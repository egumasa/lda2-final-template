"""pipeline.py — the plumbing for the mini-project.

You do NOT need to edit anything in this file. It holds the "boring but necessary"
helpers that move data around: reading the gold file, sampling a balanced subset,
loading a prompt, asking the LLM, building few-shot examples, plotting, and writing
the report. The interesting evaluation math lives in metrics.py.

Read it if you are curious — every function is written the long way, on purpose.

-------------------------------------------------------------------------------
THE CALL-FORM RULE (please read before changing any signature)

Every call form taught in the Day 1-3 course notebooks must run here UNCHANGED.
That is what lets you carry your own code over from the tutorials instead of
learning a second dialect during the project. Concretely, these must both work:

    predictions = run_prompt(PROMPT, gold)            # Day 3
    evaluate(gold, predictions, ordered=True)         # Day 2 S6 / Day 3

So when a function here needs more information than the Day-3 version did, the
extra arguments are added as OPTIONAL KEYWORDS at the END of the parameter list,
and sensible values are worked out automatically when they are left off.
`scripts/_check_call_forms.py` enforces this — run it after any signature edit.
-------------------------------------------------------------------------------
"""

import datetime
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The pieces of the method that a group may need to read and change. Re-exported, so
# `from pipeline import extract_label` keeps working exactly as Day 3 taught it.
from _study import extract_label

# If we are running OUTSIDE Colab, load a local .env so GEMINI_API_KEY is available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# The pinned model. gemini-3.1-flash-lite gets 15 requests/minute and ~500/day on
# the free tier. NOT gemini-2.5-flash: its free tier is 5/minute and 20/day, so a
# single 42-item run would need two days of quota. Override with LLM_MODEL if you
# must, but then say so in your report — the model is part of your method.
MODEL_ID = "gemini-3.1-flash-lite"


# ----------------------------------------------------------------------------------
# LLM backend
# ----------------------------------------------------------------------------------
def _looks_like_rate_limit(error: Exception) -> bool:
    """True if an exception looks like a 'too many requests' / quota error.

    We match on the text rather than a specific exception class, so the SAME guard
    works for both backends (the Colab built-in Gemini and the Gemini API), which
    raise different exception types.

    Args:
        error: the exception the model call raised.

    Returns:
        True when the message names a rate limit or a quota.
    """
    text = str(error).lower()
    for signal in ["429", "resource_exhausted", "rate limit", "quota", "too many requests"]:
        if signal in text:
            return True
    return False


def _looks_like_daily_quota(error: Exception) -> bool:
    """True if the rate-limit error is the PER-DAY cap (not the per-minute one).

    A daily cap cannot be waited out in a single sitting, so retrying is pointless -
    we want to stop immediately and tell the user how to actually get unblocked.
    Gemini names the daily quota 'GenerateRequestsPerDay...' / 'PerDay' in the error.

    Args:
        error: the exception the model call raised.

    Returns:
        True when the message names a per-day limit.
    """
    text = str(error).lower()
    return "perday" in text or "per day" in text or "requests_per_day" in text


def _suggested_delay(error: Exception, fallback: float) -> float:
    """Pull the server's own 'please retry in Ns' hint out of the error, if present.

    Gemini includes a RetryInfo like 'Please retry in 7.17s.' - honoring it is more
    accurate than a guessed backoff.

    Args:
        error: the exception the model call raised.
        fallback: seconds to wait when the error names no delay.

    Returns:
        Seconds to wait before trying again.
    """
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(error).lower())
    if match:
        return float(match.group(1)) + 1.0   # a small cushion over the server's figure
    return fallback


# --- Talking to the model, at a pace the free tier allows ---------------------------
# Free-tier Gemini caps how many requests you may send per minute, so every call goes
# through generate_text() below, which waits its turn and retries if it is told off.
#
# These four remembered values are the whole of that machinery. They are module-level
# rather than passed around because there is exactly one connection per notebook, and
# threading a clock through every call would put it in front of you at every step.
_CALL_MODEL = None        # the function that actually sends a prompt (set by make_backend)
_BACKEND_NAME = ""        # what we connected to, for printing
_MIN_INTERVAL = 4.4       # seconds to leave between calls
_LAST_CALL_TIME = 0.0     # when the last call went out (time.monotonic)


def _wait_our_turn() -> None:
    """Sleep until at least _MIN_INTERVAL seconds have passed since the last call."""
    global _LAST_CALL_TIME
    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_CALL_TIME)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TIME = time.monotonic()


def generate_text(prompt: str) -> str:
    """Send one prompt to the model and return its text reply.

    Three guards, in order:
      1) wait our turn, so we stay under the per-minute cap in the first place
      2) on a per-minute rate-limit error, sleep (honoring the delay the server
         suggests, when it gives one) and try again a few times
      3) on a PER-DAY quota error, stop at once - retrying cannot help today - and say
         how to get unblocked.

    Args:
        prompt: the text to send to the model.

    Returns:
        The model's reply, as text.

    Raises:
        RuntimeError: when the daily quota is exhausted, or after the last retry.

    Example:
        >>> reply = generate_text("Label this sentence: I like cats.")
    """
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    for attempt in range(max_retries + 1):
        _wait_our_turn()
        try:
            return _CALL_MODEL(prompt)
        except Exception as error:
            # Non-rate-limit errors are real bugs - let them surface.
            if not _looks_like_rate_limit(error):
                raise
            # A DAILY cap will not clear today - stop now with advice.
            if _looks_like_daily_quota(error):
                raise RuntimeError(
                    "Gemini free-tier DAILY quota is exhausted for this model. "
                    "Retrying will not help until it resets (~midnight Pacific). "
                    "To keep working now: hand the 'driver' role to another group "
                    "member and use their key (your files stay put in the shared "
                    "Drive folder), lower N_PER_CLASS so each run uses fewer calls, "
                    "or run in Google Colab without a key (free built-in Gemini - "
                    "but NOT reproducible, so it must not be your final frozen run)."
                ) from error
            if attempt == max_retries:
                raise RuntimeError(
                    "The model refused " + str(max_retries + 1) + " times in a row, "
                    "every time because we were sending requests too fast.\n"
                    "Wait a couple of minutes and run the cell again. If it keeps "
                    "happening, another member of your group is probably running their "
                    "own calls on the same key at the same time - agree who is driving."
                ) from error
            backoff = _suggested_delay(error, _MIN_INTERVAL * (attempt + 1))
            print("  (rate limited - waiting", round(backoff), "s then retrying)")
            time.sleep(backoff)


def _resolve_gemini_key() -> str | None:
    """Find a Gemini API key: Colab Secrets first, then the environment.

    The Colab Secrets panel (the little key icon in the sidebar) is where Day 3 tells
    you to put your key - but Colab does NOT copy secrets into the environment, so we
    have to ask for it explicitly. That is the only reason this function exists.

    Returns:
        The key, or None when neither place has one.
    """
    try:
        from google.colab import userdata      # only exists inside Colab
        key = userdata.get("GEMINI_API_KEY")   # what you saved in the Secrets panel
        if key:
            return key
    except Exception:
        pass                                    # not in Colab, or the secret is not set
    return os.environ.get("GEMINI_API_KEY")     # last resort: an environment variable


# What model the connection is actually using. Logged into your test log, so that the
# number in your report can be traced to the model that produced it.
_MODEL_IN_USE = ""


def _connect_to_gemini_api(key: str,
                           temperature: float = 0.0,
                           seed: int = 42,
                           model: str = "") -> str:
    """Set up the API-key connection.

    temperature=0 + a fixed seed = the same prompt gives the same answer every run.
    That is what "reproducible" means in practice, and it is the whole reason we prefer
    the key path over Colab's built-in model.

    Raising the temperature is a real experiment rather than a mistake: run the same
    prompt twice at 0 and twice at 1, and count how many labels changed. What you learn
    is how much of any difference between two of your rounds was the prompt and how
    much was the model. Just report the temperature you actually used.

    Args:
        key: your Gemini API key.
        temperature: 0 makes the model take its most likely answer every time. Higher
            values let it vary.
        seed: the same seed with the same prompt asks for the same answer.
        model: the model id to pin. Left empty, the course default is used.

    Returns:
        The model id the connection will use.
    """
    global _GENAI_CLIENT, _GENAI_CONFIG
    from google import genai
    from google.genai import types
    _GENAI_CLIENT = genai.Client(api_key=key)
    _GENAI_CONFIG = types.GenerateContentConfig(temperature=temperature, seed=seed)
    if model:
        return model
    return os.environ.get("LLM_MODEL", MODEL_ID)


_GENAI_CLIENT = None
_GENAI_CONFIG = None

# The (temperature, seed, model) the live connection was made with. Compared on every
# make_backend call, so that changing a setting reconnects instead of being ignored.
_BACKEND_SETTINGS = None


def _describe_settings(settings: tuple | None) -> str:
    """One line naming a connection's temperature, seed and model.

    Args:
        settings: the (temperature, seed, model) tuple, or None before connecting.

    Returns:
        A readable description, for the message printed when they change.
    """
    if settings is None:
        return "not connected yet"
    temperature, seed, model = settings
    return ("temperature=" + str(temperature) + ", seed=" + str(seed)
            + ", model=" + (model or "course default"))


def _call_gemini_api(prompt: str) -> str:
    """Send one prompt through the API-key connection."""
    response = _GENAI_CLIENT.models.generate_content(
        model=_MODEL_IN_USE, contents=prompt, config=_GENAI_CONFIG)
    return response.text


def _call_colab_gemini(prompt: str) -> str:
    """Send one prompt through Colab's built-in, keyless Gemini."""
    from google.colab import ai
    return ai.generate_text(prompt)


def make_backend(temperature: float = 0.0,
                 seed: int = 42,
                 model: str = "") -> tuple:
    """Connect to the model.

    We prefer an API KEY, because the API lets us pin the temperature and the seed,
    which is what makes a run reproducible. Colab's keyless built-in Gemini is only a
    fallback: it works with zero setup but exposes no temperature or seed, so the same
    prompt can give different answers - fine for a quick look, NOT for your final
    frozen run.

    Safe to call more than once: with the SAME settings it hands back the connection it
    already made, so re-running the SETUP cell does not reset the pacing clock and let a
    burst of calls through. With DIFFERENT settings it reconnects and says so - because
    the alternative is that `setup(temperature=1)` after `setup()` quietly keeps sending
    at temperature 0, and every number after it describes a run you did not make.

    Args:
        temperature: 0 makes the model take its most likely answer every time.
        seed: the same seed with the same prompt asks for the same answer.
        model: the model id to pin. Left empty, the course default is used.

    Returns:
        Two things: the generate_text function, and the name of what we connected to.

    Raises:
        RuntimeError: when there is no key and no Colab built-in model.
    """
    global _CALL_MODEL, _BACKEND_NAME, _MIN_INTERVAL, _MODEL_IN_USE, _BACKEND_SETTINGS
    wanted = (temperature, seed, model)
    if _CALL_MODEL is not None:
        if wanted == _BACKEND_SETTINGS:
            return generate_text, _BACKEND_NAME   # same settings - reuse it, stay paced
        print("Settings changed since the last connection, so reconnecting:")
        print("  was:", _describe_settings(_BACKEND_SETTINGS))
        print("  now:", _describe_settings(wanted))
        _CALL_MODEL = None

    # Option 1 (preferred): the Gemini API with your own key. Reproducible.
    key = _resolve_gemini_key()
    if key:
        _MODEL_IN_USE = _connect_to_gemini_api(key, temperature, seed, model)
        _CALL_MODEL = _call_gemini_api
        _BACKEND_SETTINGS = wanted
        _BACKEND_NAME = ("Gemini API (" + _MODEL_IN_USE + ", temperature="
                         + str(temperature) + ", seed=" + str(seed) + ")")
        # 4.4s between calls keeps us under the 15-per-minute free-tier cap.
        _MIN_INTERVAL = 4.4
        return generate_text, _BACKEND_NAME

    # Option 2 (fallback): Colab's free built-in Gemini. No key, but not reproducible.
    try:
        from google.colab import ai            # noqa: F401 - just checking it is there
    except ImportError:
        ai = None
    if ai is not None:
        print("WARNING: no API key found, so we are using Colab's built-in Gemini.")
        print("         It has no temperature or seed setting, so the same prompt can")
        print("         give different answers - your numbers will NOT be reproducible.")
        print("         Put your key in the Colab Secrets panel as GEMINI_API_KEY")
        print("         before your final run. A free key: aistudio.google.com/apikey")
        if temperature != 0.0:
            print("         It also cannot honour temperature=" + str(temperature)
                  + ". Whatever this backend does is not what you asked for.")
        _MODEL_IN_USE = "colab built-in gemini"
        _CALL_MODEL = _call_colab_gemini
        _BACKEND_SETTINGS = wanted
        _BACKEND_NAME = "Colab Gemini (non-reproducible)"
        _MIN_INTERVAL = 13.2                   # no published limit, so pace carefully
        return generate_text, _BACKEND_NAME

    # Option 3: nothing available - tell the user what to do.
    raise RuntimeError(
        "No LLM backend found. Either set GEMINI_API_KEY - in Colab via the Secrets "
        "panel (the key icon in the sidebar), or in a .env file when running locally - "
        "or run this notebook in Google Colab, which has a free built-in Gemini that "
        "needs no key. A free key takes a minute: https://aistudio.google.com/apikey"
    )


def setup(temperature: float = 0.0,
          seed: int = 42,
          model: str = "") -> None:
    """Connect to the model and say what we connected to. Run this once, at the top.

    The three settings are part of your study, not of this session: the temperature
    that produced your reported number is a number your report has to state. So they
    come from `config.yaml`, where your group agrees them once, and the notebook hands
    them over here in the open rather than hiding them inside the connection.

    Args:
        temperature: 0 makes the model take its most likely answer every time, which is
            what makes a run repeatable. Higher lets it vary.
        seed: the same seed with the same prompt asks for the same answer.
        model: the model id to pin. Left empty, the course default is used.

    Returns:
        Nothing. It prints the backend, the model, and the settings it connected with.

    Example:
        >>> setup(temperature=TEMPERATURE, seed=SEED, model=MODEL)
    """
    _, backend_name = make_backend(temperature, seed, model)
    print("LLM backend:", backend_name)


def model_in_use() -> str:
    """The model id the current connection is using, for the test log.

    Returns:
        The model id, or the pinned default when nothing is connected yet.

    Example:
        >>> model_in_use()
    """
    if not _MODEL_IN_USE:
        return MODEL_ID
    return _MODEL_IN_USE


def _default_backend():
    """The function that sends a prompt, for when a caller did not pass one in.

    This is generate_text - the paced, retrying one - never the raw connection.

    Returns:
        The function to call with a prompt.
    """
    sender, _ = make_backend()
    return sender


# ----------------------------------------------------------------------------------
# Reading data
# ----------------------------------------------------------------------------------
def load_gold(url_or_path: str) -> list[dict[str, str]]:
    """Read a gold/pool file. Each item looks like {"id": 1, "text": "...", "label": "..."}.

    Args:
        url_or_path: a web address, or the path to a file on this machine.

    Returns:
        The items, each with "id", "text" and "label".

    Raises:
        FileNotFoundError: when the file is not there. The message says which
            notebook writes it.
        ValueError: when the file opened but has nothing in it.

    Example:
        >>> gold = load_gold(GOLD_PATH)
    """
    # If the location is a web address, download it; otherwise open a local file.
    if str(url_or_path).startswith("http"):
        raw_bytes = urllib.request.urlopen(url_or_path).read()
        raw_text = raw_bytes.decode("utf-8")
        gold = json.loads(raw_text)
    else:
        # No pool or gold data ships with the template - every file here is one you
        # build or make yourself. If it is missing, say WHY and WHAT TO DO instead
        # of a bare traceback.
        if not Path(url_or_path).exists():
            raise FileNotFoundError(
                "File not found: " + str(url_or_path) + "\n"
                "No data ships with the template - you build all of it:\n"
                "  * A full-size POOL to sample from - notebook 01_build_pool_<track>, "
                "or the\n"
                "    shortcut `python scripts/prep_datasets.py <track>`. It lands in "
                "data/pools/.\n"
                "  * A small DEMO pool, if you just want to watch the pipeline run - "
                "add\n"
                "    `--demos` to that same command.\n"
                "  * YOUR SAMPLE - notebook 02_sample writes it to data/gold/.\n"
                "  * YOUR GOLD SET - notebook 03_annotate writes it, after you have "
                "annotated\n"
                "    and adjudicated. Nothing can make this one for you.\n"
                "Check the paths in config.py. To just see the pipeline run, build "
                "the demo pool and load DEMO_POOL_PATH instead of POOL_PATH."
            )
        opened_file = open(url_or_path, encoding="utf-8")
        gold = json.loads(opened_file.read())
        opened_file.close()
    if len(gold) == 0:
        raise ValueError(
            "The file " + str(url_or_path) + " opened, but there is nothing in it.\n"
            "Whichever notebook was supposed to write it has not finished - go back and "
            "run its save cell, then come back to this one.")
    print("Loaded", len(gold), "items. First one:", gold[0])
    return gold


def load_prompt(path: str) -> str:
    """Read a prompt template from a text file in the prompts/ folder.

    Keeping the prompt in its own file (instead of pasting it into the notebook)
    means you iterate by editing the file — and each version is easy to save and
    compare.

    Args:
        path: the prompt file to read. It must contain the placeholder {text},
            where each sentence is slotted in.

    Returns:
        The prompt text, with leading and trailing blank space removed.

    Example:
        >>> prompt = load_prompt(PROMPT_FILE)
    """
    prompt = Path(path).read_text(encoding="utf-8").strip()
    if "{text}" not in prompt:
        print("WARNING: the prompt file has no {text} placeholder — the sentence "
              "will not be inserted anywhere.")
    print("Loaded prompt from", path, "(", len(prompt), "characters ).")
    return prompt


def save_prompt(prompt: str, path: str) -> None:
    """Write one version of your prompt to a file, so a later notebook can read it.

    Nothing survives between notebooks except what is on disk. A prompt that only ever
    existed as a string in this session is one `05_test.ipynb` cannot load and your
    report cannot quote, so every version you might want to test gets saved.

    Unlike the other saves in this project, this one DOES overwrite: you will rewrite
    the same version several times while you are working on it, and refusing would mean
    inventing a new filename each time.

    Args:
        prompt: the prompt text, containing {text} where the sentence should go.
        path: where to write it. Give each version its own name - v1, v2 - so that
            the file your report names is still the file you can re-run.

    Example:
        >>> save_prompt(PROMPT_v1, ROOT / "prompts" / "my_v1.txt")
    """
    if "{text}" not in prompt:
        print("WARNING: this prompt has no {text} placeholder — every item would be "
              "sent the same sentence-less prompt. Saving it anyway.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt.strip() + "\n", encoding="utf-8")
    print("Saved your prompt to", path, "(", len(prompt), "characters ).")


# ----------------------------------------------------------------------------------
# Not overwriting work you already did
# ----------------------------------------------------------------------------------
# Re-running a cell is the most ordinary thing you can do in a notebook, and until this
# check existed it silently replaced whatever was already in the file. A gold set is a
# morning of two people's annotation; nothing warned you it had gone. So every save in
# this file stops instead, and tells you the two ways forward.
def _refuse_to_overwrite(path: Path, overwrite: bool, what: str) -> None:
    """Stop rather than replace a file that already exists.

    Args:
        path: the file about to be written.
        overwrite: True to allow the replacement and do nothing here.
        what: the word for the contents, used in the message ("predictions").

    Raises:
        FileExistsError: when the file is there and overwrite is False.
    """
    if overwrite:
        return None
    if not path.exists():
        return None
    written = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    raise FileExistsError(
        "\n" + path.name + " already exists.\n"
        "  written: " + written.strftime("%Y-%m-%d %H:%M") + "\n"
        "  holds:   " + _describe_contents(path, what) + "\n"
        "\n"
        "Saving now would replace it, and what is in it cannot be got back.\n"
        "\n"
        "  * Want to KEEP it? Open config.yaml and change  run: to the next version\n"
        "    (v1 -> v2). Everything you save from then on gets a new name, and this\n"
        "    file stays where it is. Then re-run the SETUP cell.\n"
        "    Careful: run: is part of EVERY filename - your sample, your annotation\n"
        "    sheet, your gold set and its dev/test split included. Bumping it means\n"
        "    starting the study over, not redoing one step.\n"
        "  * Meant to REPLACE it? Add overwrite=True inside the brackets of the call\n"
        "    you just ran."
    )


def _refuse_to_load_missing(path: str | Path, what: str, made_by: str) -> None:
    """Explain a missing handoff file: which notebook makes it, and what to do.

    Notebooks 04 and 05 read files that an EARLIER notebook wrote. Arriving at 05
    without having finished 04 is the ordinary way to meet this, and a bare
    FileNotFoundError pointing into open() does not say that.

    Args:
        path: the file about to be read.
        what: the word for the contents ("frozen predictions").
        made_by: which notebook writes it ("notebook 04_develop").

    Raises:
        FileNotFoundError: when the file is not there.
    """
    if Path(path).exists():
        return
    raise FileNotFoundError(
        "File not found: " + str(path) + "\n"
        "This is the " + what + " file, and " + made_by + " writes it.\n"
        "Nothing is wrong with your setup - that notebook has just not been run to\n"
        "the end yet on this config.\n"
        "  * Go and finish " + made_by + ", including its SAVE cell. A run you did\n"
        "    not save is a run this notebook cannot see: the variables live in that\n"
        "    session's memory, not on disk.\n"
        "  * Already ran it? Then it saved somewhere else. `run:` and `group:` in\n"
        "    config.yaml are part of every output filename, so changing either one\n"
        "    since that run points these paths at a file that was never written.\n"
        "    Check the name above against what is actually in outputs/."
    )


def _describe_contents(path: Path, what: str) -> str:
    """Say what is in a file, for the message above - how many items, or how big.

    Args:
        path: the file to look at.
        what: the word for the contents ("predictions").

    Returns:
        A short phrase like "40 predictions", or a byte count when it is not JSON.
    """
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return str(len(existing)) + " " + what
    except Exception:
        # Not JSON, or unreadable. The size still tells you it is not empty.
        return str(path.stat().st_size) + " bytes"


# ----------------------------------------------------------------------------------
# Freezing predictions
# ----------------------------------------------------------------------------------
# A hosted LLM is only best-effort reproducible, even at temperature=0. So once your
# prompt is final you run the model ONCE, save its predictions to a file, and do all
# your evaluation off that file. Your reported numbers then hold still, and anyone
# (including whoever grades it) can re-run your analysis on exactly the outputs you
# saw. In Day 2 you loaded a frozen file we made for you; now you make your own.
def save_predictions(predictions: list[str],
                     path: str | Path,
                     overwrite: bool = False) -> Path:
    """Write a list of predicted labels to a JSON file - this is 'freezing' a run.

    Args:
        predictions: one predicted label per gold item, in gold order.
        path: where to write the file.
        overwrite: True to replace a file that is already there. Left False, an
            existing file stops the save instead.

    Returns:
        The path it wrote.

    Raises:
        FileExistsError: when the file is there and overwrite is False.

    Example:
        >>> save_predictions(predictions, PRED_PATH)
    """
    output_path = Path(path)
    _refuse_to_overwrite(output_path, overwrite, "predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Froze", len(predictions), "predictions to", str(output_path))
    return output_path


def load_predictions(url_or_path: str | Path) -> list[str]:
    """Read a frozen predictions list back - a local path or a URL.

    Args:
        url_or_path: a web address, or the path to a file on this machine.

    Returns:
        One predicted label per gold item, in gold order.

    Example:
        >>> predictions = load_predictions(PRED_PATH)
    """
    if str(url_or_path).startswith("http"):
        raw_bytes = urllib.request.urlopen(url_or_path).read()
        predictions = json.loads(raw_bytes.decode("utf-8"))
    else:
        _refuse_to_load_missing(url_or_path, "frozen predictions",
                                "notebook 04_develop")
        opened_file = open(url_or_path, encoding="utf-8")
        predictions = json.loads(opened_file.read())
        opened_file.close()
    print("Loaded", len(predictions), "frozen predictions.")
    return predictions


# ----------------------------------------------------------------------------------
# Scoring the held-out set, on the record
# ----------------------------------------------------------------------------------
# Nothing here stops you running the test set twice. Stopping you would be the wrong
# design: at four o'clock on the last day, a group that hit a genuine mistake needs a
# way forward, and the way forward must not be to bump `run:` in config.yaml - that
# renames the sample, the sheet, the gold set and the split too, and walks off from a
# week of annotation.
#
# So instead: nothing is overwritten, every attempt is kept, and every scoring writes a
# line to a log that goes in your submission. A second attempt is allowed. It is just
# not invisible.
def save_test_run(predictions: list[str],
                  path: str | Path,
                  overwrite: bool = False) -> tuple:
    """Freeze a run on the TEST set.

    First call writes the path you gave it. A second call does not replace that file -
    it writes ..._predictions_attempt2.json beside it, and says so.

    Args:
        predictions: one predicted label per test item, in test order.
        path: where to write the first attempt.
        overwrite: True to replace the first attempt rather than write a new one.

    Returns:
        Two things: the path it wrote, and which attempt number this was.

    Example:
        >>> path, attempt = save_test_run(predictions, TEST_PRED_PATH)
    """
    first_path = Path(path)
    first_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not first_path.exists():
        attempt = 1
        output_path = first_path
        if overwrite and first_path.exists():
            print("OVERWROTE", first_path.name, "on purpose (overwrite=True).")
    else:
        attempt = _next_attempt_number(first_path)
        output_path = first_path.with_name(
            first_path.stem + "_attempt" + str(attempt) + first_path.suffix)

    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Froze", len(predictions), "test predictions to", str(output_path))

    if attempt > 1:
        print("")
        print("This is test scoring attempt", str(attempt) + ".", first_path.name,
              "is untouched.")
        print("Every attempt is bundled with your submission, and the test log records")
        print("all of them. If the number you report is not attempt 1, section 5 of your")
        print("report has to say why there was more than one.")
    return output_path, attempt


def _next_attempt_number(first_path: Path) -> int:
    """Find the lowest attempt number that is not on disk yet.

    Args:
        first_path: the path the first attempt was written to.

    Returns:
        The number to give the next attempt.
    """
    highest = 1
    pattern = first_path.stem + "_attempt*" + first_path.suffix
    for existing in first_path.parent.glob(pattern):
        digits = existing.stem.split("_attempt")[-1]
        if digits.isdigit() and int(digits) > highest:
            highest = int(digits)
    return highest + 1


def log_test_run(log_path: str | Path, record: dict) -> Path:
    """Append one record to the test log. Never refuses, never replaces.

    One JSON object per line (a .jsonl file), rather than one JSON list, because a list
    would have to be read and rewritten whole every time - and a rewrite interrupted
    halfway loses exactly the history this file exists to keep.

    Args:
        log_path: the .jsonl log file to append to.
        record: what to write as one line.

    Returns:
        The path it appended to.
    """
    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    log_file = open(output_path, "a", encoding="utf-8")
    log_file.write(line + "\n")
    log_file.close()
    return output_path


def read_test_log(log_path: str | Path) -> pd.DataFrame:
    """Read the test log back as a table, oldest scoring first.

    Args:
        log_path: the .jsonl log file written by record_test_scoring.

    Returns:
        One row per scoring. An empty table when the log does not exist yet.

    Example:
        >>> read_test_log(TESTLOG_PATH)
    """
    output_path = Path(log_path)
    if not output_path.exists():
        print("No test log at", str(output_path), "- notebook 05 writes it when you")
        print("score the held-out set. Notebook 04 never touches it.")
        return pd.DataFrame()
    records = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return pd.DataFrame(records)


def record_test_scoring(log_path: str | Path,
                        macro_f1: float,
                        attempt: int,
                        pred_path: str | Path,
                        prompt: str,
                        prompt_file: str | Path,
                        gold_items: list[dict[str, str]],
                        dev_f1: float | None = None,
                        round_key: str = "",
                        note: str = "",
                        predictions: list[str] | None = None) -> dict:
    """Write down one scoring of the held-out set, and print it back.

    Args:
        log_path: the .jsonl test log to append to.
        macro_f1: the score this run got.
        attempt: which test-scoring attempt this was.
        pred_path: the frozen predictions file the score came from.
        prompt: the prompt text, fingerprinted so two attempts can be told apart.
        prompt_file: the file the prompt was read from.
        gold_items: the held-out items, counted into the record.
        dev_f1: the dev score, when you have one to compare against.
        round_key: the name this round has in the rounds table.
        note: why there was more than one attempt, in your own words.
        predictions: the predicted labels, so "??" replies can be counted.

    Returns:
        The record it wrote.
    """
    # How many replies no label could be read out of. A macro-F1 computed over a run
    # with eight "??" in it is a different claim from the same number with none.
    unparseable = 0
    if predictions is not None:
        for predicted in predictions:
            if predicted == "??":
                unparseable = unparseable + 1
    fingerprint = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    record = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "attempt": attempt,
        "pred_file": Path(pred_path).name,
        "n_test": len(gold_items),
        "prompt_file": Path(prompt_file).name,
        # The point of the whole log. Scoring test twice with the SAME prompt is that
        # prompt run twice; scoring it twice with a DIFFERENT one is a prompt tuned
        # after seeing the held-out set. Nothing else in your submission tells them
        # apart, and the difference is the whole reason the split exists.
        "prompt_sha1": fingerprint,
        "prompt_chars": len(prompt),
        "model": model_in_use(),   # what we actually connected to
        "macro_f1": macro_f1,
        "dev_f1": dev_f1,
        "round_key": round_key,
        "unparseable": unparseable,
        "note": note,
    }
    log_test_run(log_path, record)

    print("Logged test scoring", str(attempt), "· macro-F1", macro_f1,
          "· prompt", fingerprint)
    if dev_f1 is not None:
        print("        dev was", dev_f1, "- the gap between them is a finding, not a")
        print("        failure. Say what you make of it in the report.")
    if attempt > 1:
        print("        This is test scoring number", str(attempt) + ". All of them are")
        print("        in the log, and the log is in your submission.")
    return record


# The held-out run, end to end. Running the final prompt, freezing the predictions,
# reading them back, scoring them, logging the attempt and saving the rounds table used
# to be four notebook cells and three function names, for what is one decision: this
# prompt, on the test set, once. Four cells is also four chances to stop halfway and
# leave the log disagreeing with the predictions file.
#
# The paths are passed in rather than read from config.yaml, so that this file stays
# runnable on its own - _check_call_forms.py exercises it against synthetic data.
def freeze_test_run(prompt: str,
                    test: list[dict[str, str]],
                    f1_by_round: dict[str, float],
                    pred_path: str | Path,
                    log_path: str | Path,
                    rounds_path: str | Path,
                    prompt_file: str | Path,
                    dev_f1: float | None = None,
                    note: str = "",
                    ordered: bool = False,
                    labels: list[str] | None = None,
                    key: str = "FINAL test (held out)",
                    generate_text=None,
                    extract=None) -> float:
    """Run `prompt` on the held-out set, freeze it, score it, and log the attempt.

    `f1_by_round` is updated in place, with the test score added LAST so the table
    reads as the dev rounds in order with the held-out score at the bottom, and then
    written to `rounds_path`, which notebook 06 prints as report section 2.

    Args:
        prompt: the final prompt, containing {text}.
        test: the held-out items.
        f1_by_round: your per-round scores. The test score is added to it in place.
        pred_path: where to freeze the predictions.
        log_path: the .jsonl test log to append to.
        rounds_path: where to write the rounds table, for notebook 06 to print.
        prompt_file: the file the prompt was read from.
        dev_f1: the dev score, when you have one to compare against.
        note: why there was more than one attempt. A second attempt with the SAME
            prompt is that prompt run twice; with a DIFFERENT one it is a prompt
            tuned after seeing the held-out set. Say which it was.
        ordered: True when the labels sit on a scale.
        labels: the labels to score, in scale order.
        key: the name to give this round in the table.
        generate_text: the function that sends a prompt. Left out, the connected
            backend is used.
        extract: the function that turns one reply into one label. Pass the same one
            you used while iterating - a held-out run scored with a different reader
            than the dev rounds is not comparable with them.

    Returns:
        The macro-F1 on the held-out set. The raw replies are frozen beside the
        predictions, as `..._predictions_replies.json`.

    Example:
        >>> freeze_test_run(prompt, test, f1_by_round, PRED_PATH, TESTLOG_PATH,
        ...                 ROUNDS_PATH, PROMPT_FILE, dev_f1=dev_score)
    """
    from metrics import evaluate      # imported here: metrics imports this module

    predictions = run_prompt(prompt, test, labels=labels, generate_text=generate_text,
                             extract=extract)
    written_path, attempt = save_test_run(predictions, pred_path)
    print("Froze to", Path(written_path).name, "· attempt", attempt)

    # Freeze what the model SAID as well as what we made of it. The predictions are our
    # reading of the replies; if anyone later doubts a label - or you want to show that
    # a "??" was the model's fault and not the extractor's - the reply is the evidence,
    # and it is gone as soon as this runtime resets.
    replies_path = Path(written_path).with_name(
        Path(written_path).stem + "_replies.json")
    save_json(last_replies(), replies_path, what="raw model replies", overwrite=True)

    # Scored from the file, not from the list still in memory. This is the one check
    # that the file you will quote in your report is the file you think it is.
    print("")
    print("Reading", Path(written_path).name, "back off disk, and scoring that...")
    pred_final = load_predictions(written_path)

    macro_f1 = evaluate(test, pred_final, ordered=ordered, labels=labels)
    f1_by_round[key] = macro_f1

    record_test_scoring(log_path, macro_f1=macro_f1, attempt=attempt,
                        pred_path=written_path, prompt=prompt, prompt_file=prompt_file,
                        gold_items=test, predictions=pred_final, dev_f1=dev_f1,
                        round_key=key, note=note)

    # The rounds table has existed only in this session's memory until now, and
    # notebook 06 prints it as report section 2. Overwrite: re-running this cell means
    # re-running the whole held-out scoring, and the table has to match the log it was
    # written beside.
    save_json(f1_by_round, rounds_path, what="rounds", overwrite=True)
    return macro_f1


# ----------------------------------------------------------------------------------
# Handing work from one notebook to the next
# ----------------------------------------------------------------------------------
# Notebooks 01-05 run in separate sessions, often on separate days and different
# people's runtimes. Anything one of them produces and another needs has to go through
# a FILE - a variable in someone else's Colab is not a handoff. These two do that for
# any JSON-able thing: your sample, your gold set, your per-round F1 table.
def save_json(data: list | dict, path: str | Path, what: str = "items",
              overwrite: bool = False) -> Path:
    """Write anything JSON-able to a file, making the folder if it is missing.

    Args:
        data: what to write - a list of items, or a dict of scores.
        path: where to write it.
        what: the word for the contents, used when printing and in the refusal
            message ("items", "gold", "rounds").
        overwrite: True to replace a file that is already there.

    Returns:
        The path it wrote.

    Raises:
        FileExistsError: when the file is there and overwrite is False.

    Example:
        >>> save_json(sample, SAMPLE_PATH, what="sampled items")
    """
    output_path = Path(path)
    _refuse_to_overwrite(output_path, overwrite, what)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", len(data), what, "to", str(output_path))
    return output_path


def load_json(path: str | Path, what: str = "items",
              made_by: str | None = None) -> list | dict:
    """Read back what save_json wrote.

    Args:
        path: the file to read.
        what: the word for the contents, used when printing.
        made_by: which notebook writes this file, so a missing one says where it
            comes from. Left out, the message stays general.

    Returns:
        Whatever was saved - a list of items, or a dict of scores.

    Raises:
        FileNotFoundError: when the file is not there.

    Example:
        >>> gold = load_json(GOLD_PATH, what="gold items", made_by="notebook 03")
    """
    if made_by is None:
        made_by = "an earlier notebook"
    _refuse_to_load_missing(path, what, made_by)
    opened_file = open(path, encoding="utf-8")
    data = json.loads(opened_file.read())
    opened_file.close()
    print("Loaded", len(data), what, "from", str(path))
    return data


# ----------------------------------------------------------------------------------
# Sampling a balanced subset
# ----------------------------------------------------------------------------------
def reid(items: list[dict[str, str]],
         start: int = 1) -> list[dict[str, str]]:
    """Give the items fresh id numbers start, start+1, ... in their current order.

    The id an item had in the POOL is kept as `source_id`. The sample gets its own ids
    1, 2, 3 ... because they are what you read in the annotation sheet, but that
    renumbering is also what would otherwise make it impossible to tell later which pool
    rows you have already drawn. `source_id` is that link, and notebook 02b uses it to
    draw items you do not have yet.

    An item that already carries a `source_id` keeps the one it has. sample_more
    renumbers items that have been through here once already, and overwriting would
    replace the pool id with a sample position.

    Args:
        items: the items to renumber. They are copied, not changed in place.
        start: the number to give the first item. Left alone for a first draw;
            sample_more passes the number after the ids you already have.

    Returns:
        The same items with new ids, each carrying the id it came in with.

    Example:
        >>> items = reid(items)
    """
    renumbered = []
    next_id = start
    for item in items:
        new_item = dict(item)          # make a copy so we do not change the original
        if "source_id" not in new_item and "id" in new_item:
            new_item["source_id"] = new_item["id"]
        new_item["id"] = next_id
        renumbered.append(new_item)
        next_id = next_id + 1
    return renumbered


# The three samplers, behind one name. Notebook 02 used to spell this out as an
# if/elif over three calls, which meant three hard-coded sizes (40 items, 10 documents,
# 4 sentences each) that had nothing to do with n_per_class in config.yaml - so
# choosing "random" or "by_document" silently made that setting dead. Here every
# strategy is sized from the same setting, and the choice stays one word.
def sample(pool: list[dict[str, str]],
           strategy: str,
           n_per_class: int,
           seed: int = 42,
           n_per_doc: int = 4) -> list[dict[str, str]]:
    """Draw a sample using one of the three strategies, all sized from n_per_class.

    Args:
        pool: the items to draw from.
        strategy: which draw to make.
            "balanced" - up to n_per_class items of EACH label.
            "random" - the same TOTAL, drawn without regard to label.
            "by_document" - whole passages, enough to reach that total. cars50 and
            raamove only; the other tracks have no documents.
        n_per_class: how many items per label. Every strategy is sized from this.
        seed: same seed gives the same draw; a different seed gives a different one.
        n_per_doc: how many sentences to take from each document, for "by_document".

    Returns:
        The drawn items, renumbered from 1.

    Raises:
        ValueError: when strategy is not one of the three names.

    Example:
        >>> items = sample(pool, "balanced", N_PER_CLASS, seed=SEED)
    """
    n_labels = len(label_set(pool))
    n_total = n_per_class * n_labels

    if strategy == "balanced":
        return sample_pool(pool, n_per_class, seed)
    if strategy == "random":
        return sample_random(pool, n_total, seed)
    if strategy == "by_document":
        # Enough documents to reach the same total, rounded up so the draw is never
        # smaller than the other two strategies would have given.
        n_docs = max(1, -(-n_total // n_per_doc))
        return sample_by_document(pool, n_docs, n_per_doc, seed)

    raise ValueError(
        "strategy has to be balanced, random or by_document, and it says "
        + repr(strategy) + ". Fix the line above and run this cell again.")


def sample_pool(pool: list[dict[str, str]],
                n_per_class: int,
                seed: int = 42) -> list[dict[str, str]]:
    """Pick up to n_per_class items for EACH label, chosen at random.

    Rare labels simply give fewer items - that is a property of the data.

    This is the BALANCED strategy: it forces the classes level so that per-class
    precision and recall are readable and the confusion matrix is not dominated by
    one huge class. The cost is that your sample no longer looks like the corpus.
    See sample_random and sample_by_document for the other two positions.

    Args:
        pool: the items to draw from, each {"id", "text", "label"}.
        n_per_class: how many to take per label.
        seed: same seed gives the same draw; a different seed gives a different one,
            so different groups can get different subsets.

    Returns:
        The drawn items, shuffled and renumbered from 1.

    Example:
        >>> items = sample_pool(pool, N_PER_CLASS, seed=SEED)
    """
    random_generator = random.Random(seed)

    # Step 1: sort every pool item into a bucket named after its label.
    items_by_label = {}
    for item in pool:
        label = item["label"]
        if label not in items_by_label:
            items_by_label[label] = []
        items_by_label[label].append(item)

    # Step 2: from each bucket, shuffle and keep up to n_per_class items.
    sampled = []
    for label in sorted(items_by_label):
        items_with_this_label = items_by_label[label]
        random_generator.shuffle(items_with_this_label)
        kept_items = items_with_this_label[:n_per_class]
        for item in kept_items:
            sampled.append(item)

    # Step 3: shuffle the whole set and renumber the ids from 1.
    random_generator.shuffle(sampled)
    sampled = reid(sampled)

    # Step 4: say what we drew, and warn if we took most of the pool. A sample is only
    # meaningful if there is a pool left over - both because a near-total sample is not
    # a sample, and because build_fewshot needs unused items for its examples.
    _report_draw(sampled, pool, "Sampled balanced by label")
    return sampled


def _report_draw(sampled: list[dict[str, str]],
                 pool: list[dict[str, str]],
                 what: str) -> None:
    """Print what a draw produced, and warn if it swallowed the pool.

    Shared by all three sampling strategies, so that whichever one a group picks,
    they see the same two things: the per-label counts their choice produced, and a
    warning if there is no pool left for build_fewshot to draw examples from.

    Args:
        sampled: the items that were drawn.
        pool: the items they were drawn from.
        what: how to describe the draw ("Sampled at random").

    Returns:
        Nothing. It prints the counts and any warning.
    """
    counts = {}
    for item in sampled:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    print(what + ":", len(sampled), "items. Per-label counts:", counts)

    if len(pool) > 0 and len(sampled) > 0.5 * len(pool):
        print("WARNING: you just took", len(sampled), "of", len(pool), "pool items.")
        print("         That is most of the pool, which leaves few or no spare items")
        print("         for few-shot examples. Are you pointing at a small DEMO file")
        print("         instead of a full pool? Build the real one with:")
        print("             python scripts/prep_datasets.py <track>")


def sample_random(pool: list[dict[str, str]],
                  n_total: int,
                  seed: int = 42) -> list[dict[str, str]]:
    """Draw n_total items at random, ignoring the labels entirely.

    The corpus as it actually is: every item equally likely, so each label turns up
    roughly as often as it does in the pool. That is the honest thing if you want to
    say something about the corpus - and the awkward thing if a label is rare, because
    a rare label will come back with one or two items, or none at all, and precision
    and recall on a class of one mean very little.

    Compare sample_pool, which forces the classes level instead.

    Args:
        pool: the items to draw from. It is copied, not shuffled in place.
        n_total: how many items to draw altogether.
        seed: same seed gives the same draw.

    Returns:
        The drawn items, renumbered from 1.

    Example:
        >>> items = sample_random(pool, 40, seed=SEED)
    """
    random_generator = random.Random(seed)

    # Copy before shuffling: shuffling the caller's pool in place would quietly change
    # the order of the list they are still holding.
    shuffled = list(pool)
    random_generator.shuffle(shuffled)
    sampled = reid(shuffled[:n_total])

    _report_draw(sampled, pool, "Sampled at random")
    return sampled


def sample_by_document(pool: list[dict[str, str]],
                       n_docs: int,
                       n_per_doc: int,
                       seed: int = 42) -> list[dict[str, str]]:
    """Pick whole documents first, then sentences inside them.

    Forty sentences drawn from forty abstracts and forty sentences drawn from three
    are both "forty sentences", and they support very different claims. This strategy
    makes that choice explicit: n_docs documents, n_per_doc sentences from each.

    Args:
        pool: the items to draw from. Every item must carry a "doc_id".
        n_docs: how many documents to draw.
        n_per_doc: how many sentences to take from each of them.
        seed: same seed gives the same draw.

    Returns:
        The drawn items, renumbered from 1.

    Raises:
        ValueError: when any item has no doc_id. Only cars50, cars50_step and
            raamove record which document a sentence came from.

    Example:
        >>> items = sample_by_document(pool, 10, 4, seed=SEED)
    """
    # Say it here, in terms of the track, rather than dying on a KeyError inside the
    # loop below - which would read as "the code is broken" rather than "this corpus
    # does not record which document a sentence came from".
    items_without_doc = 0
    for item in pool:
        if "doc_id" not in item:
            items_without_doc = items_without_doc + 1
    if items_without_doc > 0:
        raise ValueError(
            "sample_by_document needs to know which document each item came from, "
            "and " + str(items_without_doc) + " of these " + str(len(pool)) + " items "
            "do not carry a doc_id.\n"
            "Only the rhetorical-move tracks record that: cars50, cars50_step and "
            "raamove. On this track a sentence is not part of a passage in the data, "
            "so there are no documents to stratify by.\n"
            "Use sample_pool (balanced across labels) or sample_random instead, and "
            "say in PLAN.md which you chose.")

    random_generator = random.Random(seed)

    # Step 1: group the pool into documents.
    items_by_doc = {}
    for item in pool:
        doc_id = item["doc_id"]
        if doc_id not in items_by_doc:
            items_by_doc[doc_id] = []
        items_by_doc[doc_id].append(item)

    # Step 2: choose the documents. Sorted first, so the seed alone decides the draw -
    # dict order would otherwise depend on what order the pool happened to be built in.
    doc_ids = sorted(items_by_doc)
    random_generator.shuffle(doc_ids)
    chosen_docs = doc_ids[:n_docs]
    if len(chosen_docs) < n_docs:
        print("NOTE: you asked for", n_docs, "documents and the pool has only",
              len(chosen_docs), "- using all of them.")

    # Step 3: from each chosen document, take up to n_per_doc sentences.
    sampled = []
    for doc_id in chosen_docs:
        items_in_doc = list(items_by_doc[doc_id])
        random_generator.shuffle(items_in_doc)
        for item in items_in_doc[:n_per_doc]:
            sampled.append(item)

    random_generator.shuffle(sampled)
    sampled = reid(sampled)

    _report_draw(sampled, pool, "Sampled by document")
    print("        from", len(chosen_docs), "documents, up to", n_per_doc, "each.")
    # The labels were never controlled for, so say what that cost. A group that draws
    # by document and then reports per-class F1 needs to have seen this line.
    print("        Note the label counts above: this strategy balances DOCUMENTS,")
    print("        not labels, so a rare move stays rare.")
    return sampled


# --- Drawing a SECOND time, into a sheet you are already annotating in --------------
# Notebook 02b. Your first draw is annotated, there is time left, and more annotated
# items is more evidence. The whole difficulty is that by then the session that drew the
# first sample is gone, so the only record of what you already have is the sample file -
# and the ids in it are 1..N, not the ids those items had in the pool. That is what
# `source_id` is for.
def remaining_pool(pool: list[dict[str, str]],
                   sampled: list[dict[str, str]]) -> list[dict[str, str]]:
    """The pool items you have NOT already drawn.

    Matched two ways, and an item is excluded if EITHER matches:

      source_id  the id the item had in the pool, recorded when it was drawn.
      text       for samples drawn before source_id was recorded, and as a check on
                 a pool that has been rebuilt with different ids since you drew.

    Args:
        pool: the full pool for your track.
        sampled: the items you already have, read back from your sample file.

    Returns:
        The pool items that are in neither of those, in pool order.

    Example:
        >>> left = remaining_pool(pool, sampled)
    """
    taken_ids = set()
    taken_texts = set()
    without_source_id = 0
    for item in sampled:
        if "source_id" in item:
            taken_ids.add(item["source_id"])
        else:
            without_source_id = without_source_id + 1
        taken_texts.add(str(item["text"]))

    remaining = []
    for item in pool:
        if item.get("id") in taken_ids:
            continue
        if str(item["text"]) in taken_texts:
            continue
        remaining.append(item)

    excluded = len(pool) - len(remaining)
    print("pool:", len(pool), "· already drawn:", len(sampled),
          "· left to draw from:", len(remaining))

    if without_source_id > 0:
        print("NOTE:", without_source_id, "of your", len(sampled), "sampled items carry")
        print("      no source_id, so they were matched to the pool by their text.")

    # More pool rows excluded than you drew means the pool holds the same text twice.
    # Nothing you have annotated can be drawn again, which is the direction you want,
    # but it does take items out of reach - so say it rather than let the count puzzle
    # someone later.
    if excluded > len(sampled):
        print("NOTE:", excluded, "pool items were excluded but you have only",
              len(sampled), "sampled.")
        print("      The pool holds some texts more than once, so the copies of what")
        print("      you already drew were excluded too.")
    return remaining


def sample_more(pool: list[dict[str, str]],
                sampled: list[dict[str, str]],
                strategy: str,
                n_per_class: int,
                seed: int = 42,
                n_per_doc: int = 4) -> list[dict[str, str]]:
    """Draw MORE items, from the part of the pool you have not drawn yet.

    The same three strategies as your first draw, sized the same way from n_per_class.
    The choice is yours again and it needs the same one-sentence reason in PLAN.md -
    and if you pick a different strategy this time, your gold set was built in two
    stages and the report has to say so.

    Args:
        pool: the full pool for your track.
        sampled: the items you already have, read back from your sample file.
        strategy: "balanced", "random" or "by_document", as in notebook 02.
        n_per_class: how many more items per label.
        seed: use a different one from your first draw, and record both.
        n_per_doc: how many sentences per document, for "by_document".

    Returns:
        Only the NEW items. Their ids carry on from the highest id you already have,
        so nothing in your sheet is renumbered and no two rows share an id.

    Raises:
        ValueError: when `sampled` is empty, when the pool is used up, or when the
            new items collide with the ones you already have.

    Example:
        >>> extra = sample_more(pool, sampled, "balanced", 5, seed=SEED + 1)
    """
    if not sampled:
        raise ValueError(
            "sample_more adds to a draw you already have, and the list you passed is "
            "empty.\n"
            "If this is your first draw, use 02_sample.ipynb instead. If it is not, "
            "the sample file did not load - check the cell above.")

    ### Step 1: what is left ###
    remaining = remaining_pool(pool, sampled)
    if not remaining:
        raise ValueError(
            "Every item in the pool is already in your sample, so there is nothing "
            "left to add.\n"
            "This is the end of what this track can give you. Say so in the report - "
            "a sample that is the whole pool is a census, not a sample.")

    ### Step 2: has a label run out? ###
    # Worth saying BEFORE the draw. A balanced top-up simply will not contain that
    # label, and a random one is sized from the labels that are left, so it comes out
    # smaller than the same call on the full pool would have given.
    gone = []
    labels_left = label_set(remaining)
    for label in label_set(pool):
        if label not in labels_left:
            gone.append(label)
    if gone:
        print("NOTE: no items left in the pool for:", ", ".join(gone))
        print("      A balanced top-up will not contain them, and a random one is")
        print("      sized from the", len(labels_left), "labels that are left.")
        print("      Your combined sample stops being balanced. That belongs in the")
        print("      limitations of your report, not in a change of strategy.")

    ### Step 3: the same draw you made the first time, over what is left ###
    extra = sample(remaining, strategy, n_per_class, seed, n_per_doc)

    ### Step 4: ids that carry on from the ones you already have ###
    # Not from 1. Two rows sharing an id are merged into one when the sheet is read
    # back, so a restart here would quietly cost you half your annotation.
    highest = 0
    for item in sampled:
        highest = max(highest, int(item["id"]))
    extra = reid(extra, start=highest + 1)

    ### Step 5: check the new items really are new ###
    _refuse_overlapping_draw(extra, sampled)

    last = extra[len(extra) - 1]["id"]
    print("")
    print("Adding", len(extra), "items, ids", str(extra[0]["id"]) + ".." + str(last) +
          ". The", len(sampled), "you already have keep theirs.")
    return extra


def _refuse_overlapping_draw(extra: list[dict[str, str]],
                             sampled: list[dict[str, str]]) -> None:
    """Stop if a new item repeats an id, a source_id or a text you already have.

    Any of the three means the new rows would not be new. An id clash costs you rows
    when the sheet is read back; a source_id or text clash means two rows of the same
    item, annotated twice.

    Args:
        extra: the newly drawn items.
        sampled: the items you already have.

    Returns:
        Nothing. It raises if anything overlaps.

    Raises:
        ValueError: naming what clashed.
    """
    old_ids = set()
    old_sources = set()
    old_texts = set()
    for item in sampled:
        old_ids.add(int(item["id"]))
        if "source_id" in item:
            old_sources.add(item["source_id"])
        old_texts.add(str(item["text"]))

    clashing_ids = []
    repeated = []
    for item in extra:
        if int(item["id"]) in old_ids:
            clashing_ids.append(item["id"])
        if item.get("source_id") in old_sources or str(item["text"]) in old_texts:
            repeated.append(item["id"])

    if clashing_ids:
        raise ValueError(
            "The new items would reuse ids you already have: "
            + str(clashing_ids[:10]) + "\n"
            "Two rows with the same id are merged into one when the sheet is read "
            "back, so this would cost you annotation. Are the ids in your sample file "
            "still 1, 2, 3 ...? Nothing was written.")
    if repeated:
        raise ValueError(
            "These new items are ones you have already drawn: "
            + str(repeated[:10]) + "\n"
            "Usually this means the pool was rebuilt after your first draw, so the "
            "source_ids no longer point at the same rows. Nothing was written - "
            "check that data/pools/ still holds the pool you sampled from.")


def label_set(gold: list[dict[str, str]]) -> list[str]:
    """Return the sorted list of labels that appear in a gold set.

    Args:
        gold: the items to look through, each with a "label" key.

    Returns:
        Every label that appears, once each, sorted alphabetically.

    Example:
        >>> label_set(gold)
    """
    labels = []
    for item in gold:
        label = item["label"]
        if label not in labels:
            labels.append(label)
    labels.sort()
    return labels


# ----------------------------------------------------------------------------------
# Drawing the line: dev and test
# ----------------------------------------------------------------------------------
# You will change your prompt because of what you saw it get wrong. That is the work.
# But a score measured on the same items you kept adjusting against is not a measure of
# how good the prompt is - it is a measure of how long you kept adjusting. The fix is a
# line drawn through the gold set BEFORE any of that starts:
#
#   dev   the items you may look at. Iterate here, as many rounds as you like.
#   test  opened once, at the end. Whatever it says is what you report.
#
# The line is drawn after annotation, so it costs no extra coding - both halves were
# double-coded in the same sheet. What it costs is items you are allowed to learn from.
def split_dev_test(gold: list[dict[str, str]],
                   dev: int | float,
                   seed: int = 42,
                   by_document: bool = False) -> tuple:
    """Split an adjudicated gold set into (dev, test).

    Both forms stratify by label, so every label is represented on both sides of the
    line wherever the data allows it. The ids are NOT renumbered: notebook 06 joins on
    them to ask which of the model's errors are also items your coders disagreed about.

    Args:
        gold: the adjudicated gold items.
        dev: how big the dev half is, and how you write it says which you meant.
            A whole number (dev=3) is that many dev items per label. A decimal
            between 0 and 1 (dev=0.35) is that proportion of each label's items.
        seed: same seed gives the same split.
        by_document: True if you drew your sample with sample_by_document, so that
            no passage has some sentences in dev and the rest in test.

    Returns:
        Two lists: the dev items and the test items.

    Raises:
        ValueError: when `dev` is neither a count nor a proportion, or when
            by_document=True and the items carry no doc_id.

    Example:
        >>> dev, test = split_dev_test(gold, DEV, seed=SEED)
    """
    dev_per_class, dev_fraction = _read_dev_size(dev)

    if by_document:
        dev, test = _split_by_document(gold, dev_per_class, dev_fraction, seed)
    else:
        dev, test = _split_by_label(gold, dev_per_class, dev_fraction, seed)

    _report_split(dev, test, by_document)
    return dev, test


def _read_dev_size(dev: int | float) -> tuple:
    """Read one `dev` setting into a count and a proportion - exactly one of them set.

    A count and a proportion are one decision, so config.yaml holds one key and the way
    the number is written says which was meant: 3 is three items per label, 0.35 is a
    third of each label. The two names survive below this line only because the split
    functions need to tell the cases apart.

    Args:
        dev: the setting from config.yaml.

    Returns:
        Two things: the per-label count and the proportion. One of them is None.

    Raises:
        ValueError: when dev is neither a count nor a proportion, or is out of range.
    """
    # bool before int: True is an int in Python, and `dev: yes` in YAML would otherwise
    # be read as "1 dev item per label" without complaining.
    if isinstance(dev, bool) or not isinstance(dev, (int, float)):
        raise ValueError(
            "dev=" + repr(dev) + " is neither a count nor a proportion.\n"
            + _DEV_HELP)
    if isinstance(dev, float):
        if not 0 < dev < 1:
            raise ValueError(
                "dev=" + str(dev) + ". A decimal is read as a proportion of each label, "
                "so it has to be strictly between 0 and 1. For a fixed count per label, "
                "write it without a decimal point.\n" + _DEV_HELP)
        return None, dev
    if dev < 1:
        raise ValueError(
            "dev=" + str(dev) + " leaves no dev set at all.\n" + _DEV_HELP)
    return dev, None


_DEV_HELP = (
    "Set `dev:` in config.yaml to one of these:\n"
    "    dev: 3       a whole number - that many dev items per label\n"
    "    dev: 0.35    a decimal between 0 and 1 - that proportion of each label\n"
    "Then re-run the SETUP cell at the top of this notebook."
)


def _dev_target(bucket_size: int,
                dev_per_class: int | None,
                dev_fraction: float | None) -> int:
    """How many of a label's items should go to dev, after the rare-class clamp.

    Args:
        bucket_size: how many items this label has.
        dev_per_class: the per-label count, when that is what was set.
        dev_fraction: the proportion, when that is what was set.

    Returns:
        How many to put in dev, never more than bucket_size - 1: a label missing
        from TEST drops out of the macro average without saying so, and test is the
        number you report, so test is served first.
    """
    if dev_per_class is not None:
        wanted = dev_per_class
    else:
        # Round half UP, written out rather than left to round(). Python's round() does
        # banker's rounding: round(0.5) is 0 and round(1.5) is 2, which is not something
        # you want to have to explain in the Q&A about why one class got no dev items.
        wanted = int(dev_fraction * bucket_size + 0.5)

    # The clamp, and the one asymmetric decision in this function. A label missing from
    # TEST drops out of the macro average without saying so, and test is the number you
    # report - so test gets served first. A label missing from DEV only means you get no
    # feedback on it while you iterate, which you can live with.
    if bucket_size <= 1:
        return 0
    return max(0, min(wanted, bucket_size - 1))


def _split_by_label(gold: list[dict[str, str]],
                    dev_per_class: int | None,
                    dev_fraction: float | None,
                    seed: int) -> tuple:
    """Stratified split: each label is divided in the same proportion.

    Args:
        gold: the items to split.
        dev_per_class: the per-label dev count, when that is what was set.
        dev_fraction: the proportion, when that is what was set.
        seed: same seed gives the same split.

    Returns:
        Two lists: the dev items and the test items. The ids are not renumbered.
    """
    random_generator = random.Random(seed)

    # Same bucket-then-sorted-then-shuffle shape as sample_pool, so the two read alike
    # and the seed - not the order the list happened to be built in - decides the draw.
    items_by_label = {}
    for item in gold:
        label = item["label"]
        if label not in items_by_label:
            items_by_label[label] = []
        items_by_label[label].append(item)

    dev = []
    test = []
    for label in sorted(items_by_label):
        items_with_this_label = items_by_label[label]
        random_generator.shuffle(items_with_this_label)
        n_dev = _dev_target(len(items_with_this_label), dev_per_class, dev_fraction)

        if n_dev == 0 and len(items_with_this_label) > 0:
            print("NOTE: label", label, "has only", len(items_with_this_label),
                  "item(s), so all of them went to TEST - test is the number you")
            print("      report, and a label missing from it drops out of the macro")
            print("      average without saying so.")

        for item in items_with_this_label[:n_dev]:
            dev.append(item)
        for item in items_with_this_label[n_dev:]:
            test.append(item)

    # Shuffle each half so the labels are not left in blocks, but do NOT renumber the
    # ids. Every id here was fixed when the annotation sheet was built, and notebook 06
    # joins on them to ask which of the model's errors are also the items your two
    # coders disagreed about. Renumbering would leave that join silently meaningless.
    random_generator.shuffle(dev)
    random_generator.shuffle(test)
    return dev, test


def _split_by_document(gold: list[dict[str, str]],
                       dev_per_class: int | None,
                       dev_fraction: float | None,
                       seed: int) -> tuple:
    """Split whole documents, so no passage has sentences on both sides of the line.

    Args:
        gold: the items to split. Every item must carry a "doc_id".
        dev_per_class: the per-label dev count, when that is what was set.
        dev_fraction: the proportion, when that is what was set.
        seed: same seed gives the same split.

    Returns:
        Two lists: the dev items and the test items. Whole documents cannot be cut,
        so the dev size is a target to reach or pass, not a size to hit exactly.

    Raises:
        ValueError: when any item has no doc_id.
    """
    items_without_doc = 0
    for item in gold:
        if "doc_id" not in item:
            items_without_doc = items_without_doc + 1
    if items_without_doc > 0:
        raise ValueError(
            "by_document=True needs to know which document each item came from, and "
            + str(items_without_doc) + " of these " + str(len(gold)) + " items do not "
            "carry a doc_id.\n"
            "Only the rhetorical-move tracks record that: cars50, cars50_step and "
            "raamove. On this track a sentence is not part of a passage in the data.\n"
            "Leave by_document off and the split stratifies by label instead.")

    # How many dev items we are aiming for overall. Whole documents cannot be cut, so
    # this is a target to reach or pass, not a size to hit exactly.
    if dev_per_class is not None:
        wanted = dev_per_class * len(label_set(gold))
    else:
        wanted = int(dev_fraction * len(gold) + 0.5)

    items_by_doc = {}
    for item in gold:
        doc_id = item["doc_id"]
        if doc_id not in items_by_doc:
            items_by_doc[doc_id] = []
        items_by_doc[doc_id].append(item)

    doc_ids = sorted(items_by_doc)
    random.Random(seed).shuffle(doc_ids)

    dev = []
    test = []
    for doc_id in doc_ids:
        if len(dev) < wanted:
            for item in items_by_doc[doc_id]:
                dev.append(item)
        else:
            for item in items_by_doc[doc_id]:
                test.append(item)

    print("Split by document:", len(doc_ids), "documents, none of them straddling the")
    print("        line. No label balancing is possible this way - check the counts")
    print("        below and say in your report what they cost you.")
    return dev, test


def _report_split(dev: list[dict[str, str]],
                  test: list[dict[str, str]],
                  by_document: bool = False) -> None:
    """Print what the split produced, and warn about labels that fell off one side.

    Args:
        dev: the dev half.
        test: the test half.
        by_document: True when the split was made by document, which changes the
            advice given for a label that is missing from test.

    Returns:
        Nothing. It prints the per-label counts and any warnings.
    """
    dev_counts = _label_counts(dev)
    test_counts = _label_counts(test)
    print("Split:", len(dev), "dev ·", len(test), "test ·",
          len(dev) + len(test), "annotated in total.")
    print("        dev  per label:", dev_counts)
    print("        test per label:", test_counts)

    for label in sorted(dev_counts):
        if label not in test_counts:
            print("WARNING: label", label, "is in dev but NOT in test, so it cannot")
            print("         appear in the score you report. Fix this before notebook 04")
            if by_document:
                print("         - a different seed will land the documents differently.")
            else:
                print("         - lower your dev size, or redraw with a bigger sample.")

    if len(dev) < 5 or len(test) < 5:
        print("NOTE: halves this small usually mean you are on the DEMO pool. Fine for")
        print("      watching the pipeline run, useless for a number you would report.")

    print("        dev is what you iterate against in notebook 04. test is opened in")
    print("        notebook 05, once, and scored there.")


def _label_counts(items: list[dict[str, str]]) -> dict[str, int]:
    """Count items per label - the shared shape behind every 'per-label counts' line.

    Args:
        items: the items to count, each with a "label" key.

    Returns:
        How many items carry each label.
    """
    counts = {}
    for item in items:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


# ----------------------------------------------------------------------------------
# Asking the model and reading its answer
# ----------------------------------------------------------------------------------
# `extract_label` is imported at the top of this file from _study.py, where the other
# pieces of the method live: the rule inside it - which label a reply is pointing at -
# is a judgment a group may need to change. `run_prompt` falls back to it when
# `extract` is left out.

# The raw replies from the most recent run_prompt call. Kept here rather than returned,
# so that `predictions = run_prompt(prompt, dev)` stays the one-line call Day 3 taught.
_LAST_REPLIES = []


def last_replies() -> list[str]:
    """What the model actually said in the most recent run, one string per item.

    `run_prompt` hands back the labels it read out of these. When one of them is "??",
    or when a label looks wrong, this is where you find out why - the reply is the
    evidence and the label is only our reading of it.

    Returns:
        The raw replies, in the order the items were sent. Empty before the first run.

    Example:
        >>> last_replies()[3]
    """
    return _LAST_REPLIES


def _one_prediction_line(item: dict[str, str], predicted: str, reply: str) -> str:
    """One item's result as a single line: what it was, what we read, what it said.

    Args:
        item: the gold item just sent.
        predicted: the label read out of the reply.
        reply: what the model actually replied.

    Returns:
        The line to print.
    """
    gold_label = str(item.get("label", "?"))
    # Collapse the reply onto one line - models like to answer in paragraphs.
    said = " ".join(str(reply).split())
    if len(said) > 60:
        said = said[:57] + "..."

    mark = "     "
    if predicted == "??":
        mark = "  ?? "
    elif gold_label != "?" and predicted != gold_label:
        mark = " MISS"

    return ("  " + str(item.get("id", "-")).rjust(4)
            + "  " + gold_label.ljust(10) + " -> " + str(predicted).ljust(10)
            + mark + "  | " + said)


def run_prompt(prompt: str,
               gold: list[dict[str, str]],
               labels: list[str] | None = None,
               generate_text=None,
               extract=None) -> list[str]:
    """Ask the model to label every item, and collect the predicted labels.

    Same call as Day 3: run_prompt(PROMPT, gold). The optional arguments are worked out
    for you, so you only pass one if you want something different.

    It prints one line per item while it runs - the gold label, the label it read out of
    the reply, and the beginning of the reply itself. Watch the third column when the
    second one says "??": that is the model answering in a shape `extract_label` cannot
    read, which is a finding about your prompt rather than a bug.

    `extract` is how you act on that. `extract_label` decides that "This looks like Move
    2 to me" means `Move 2`, and if your model keeps answering in some shape it misses,
    copy it into a cell, change it, and pass your version in. Pass it rather than only
    redefining it: when this function is the one imported from a file, a redefinition in
    your notebook does not reach the copy this loop calls, and you would get the old
    labels back with no sign anything had been ignored. Passing it also puts the rule
    that produced your numbers in the call, where a reader of the notebook can see it.

    Args:
        prompt: your prompt, containing {text} where the sentence should go, and
            {context} for its passage on the tracks that carry one.
        gold: the items to label.
        labels: the labels your scheme allows. Left out, they are read off `gold`.
        generate_text: the function that sends a prompt. Left out, the backend
            connected by the Setup cell is used.
        extract: the function that turns one reply into one label. Left out,
            `extract_label` is used.

    Returns:
        One predicted label per gold item, in the same order. Replies no label could be
        read out of come back as "??". The raw replies are kept as well - see
        `last_replies()`.

    Example:
        >>> predictions = run_prompt(prompt, dev)
    """
    global _LAST_REPLIES
    if labels is None:
        labels = label_set(gold)
    if generate_text is None:
        generate_text = _default_backend()
    if extract is None:
        extract = extract_label

    # A prompt that asks for {context} on a track whose items have none would quietly
    # send the model an empty passage, once per item, and report a number as if it had
    # tested something. Say so instead.
    if "{context}" in prompt and not any(item.get("context") for item in gold):
        print("WARNING: this prompt uses {context}, but none of these items carry one. "
              "Only the rhetorical-move tracks (cars50, raamove) do. The model is about "
              "to be shown an empty passage " + str(len(gold)) + " times.")

    predictions = []
    replies = []
    total = len(gold)
    position = 0
    # One line per item is readable for a project-sized set and a wall of text for a
    # whole pool, so past that we fall back to a count every ten.
    show_each = total <= 40

    for item in gold:
        position = position + 1
        # Put this item's sentence into the prompt where {text} is - and its passage
        # where {context} is, on the tracks that carry one. A prompt that does not
        # mention {context} simply ignores it.
        filled_prompt = prompt.format(text=item["text"],
                                      context=item.get("context", ""))
        reply = generate_text(filled_prompt)
        predicted_label = extract(reply, labels)
        predictions.append(predicted_label)
        replies.append(str(reply))

        if show_each:
            print(_one_prediction_line(item, predicted_label, reply))
        elif position % 10 == 0:
            print("  ...", position, "/", total, "done")

    # Keep the raw replies. They are the evidence behind every extraction decision -
    # what you look at when a label comes back "??" - and the reproducibility checklist
    # asks for what the model actually said, not only what we made of it.
    _LAST_REPLIES = replies

    # Count how many replies we could not turn into a valid label.
    number_unparseable = 0
    for label in predictions:
        if label == "??":
            number_unparseable = number_unparseable + 1
    print("Got " + str(len(predictions)) + " predictions ("
          + str(number_unparseable) + " could not be parsed).")
    if number_unparseable:
        print("  The ?? rows are replies no label could be read out of. Read what the")
        if show_each:
            print("  model actually said in the right-hand column above before you")
            print("  change anything.")
        else:
            # No per-item lines were printed for a set this size, so pointing at a
            # column that is not there would send them looking for nothing.
            print("  model actually said - `last_replies()` - before you change")
            print("  anything.")
    return predictions


# ----------------------------------------------------------------------------------
# Few-shot examples
# ----------------------------------------------------------------------------------
def build_fewshot(base_prompt: str,
                  pool: list[dict[str, str]],
                  gold: list[dict[str, str]],
                  labels: list[str] | None = None,
                  shots_per_class: int = 1,
                  seed: int = 42) -> str:
    """Put a few labeled examples (taken from the pool) in front of the prompt.

    We NEVER use an item that is in the gold set as an example, otherwise we
    would be showing the model the very answers we are testing it on. Items are
    matched by their TEXT, not their id, because sampling renumbers the ids.

    Args:
        base_prompt: the prompt to put the examples in front of.
        pool: the items to draw examples from.
        gold: the items being tested. None of these is used as an example.
        labels: the labels to find examples for. Left out, read off `gold`.
        shots_per_class: how many examples to show for each label.
        seed: same seed gives the same examples.

    Returns:
        The example block followed by your prompt.

    Example:
        >>> prompt = build_fewshot(base_prompt, pool, gold, shots_per_class=2)
    """
    if labels is None:
        labels = label_set(gold)

    # Step 1: collect the texts that are already in the gold set.
    gold_texts = []
    for item in gold:
        gold_texts.append(item["text"])

    # Step 2: group the remaining pool items by label (skipping any gold items).
    examples_by_label = {}
    for item in pool:
        if item["text"] in gold_texts:
            continue
        label = item["label"]
        if label not in examples_by_label:
            examples_by_label[label] = []
        examples_by_label[label].append(item)

    # Step 3: for each label, shuffle and take a few examples.
    random_generator = random.Random(seed)
    lines = ["Here are labeled examples:"]
    labels_with_no_examples = []
    for label in labels:
        if label in examples_by_label:
            examples = examples_by_label[label]
        else:
            examples = []
        random_generator.shuffle(examples)
        chosen_examples = examples[:shots_per_class]
        if len(chosen_examples) < shots_per_class:
            labels_with_no_examples.append(label)
        for item in chosen_examples:
            lines.append("Sentence: " + item["text"] + "\nLabel: " + label)

    # Step 4: if the pool could not supply enough spare examples, say so loudly -
    # a few-shot prompt missing whole labels is not the prompt you think it is.
    if len(labels_with_no_examples) > 0:
        print("WARNING: not enough spare pool items for", shots_per_class,
              "example(s) of:", ", ".join(labels_with_no_examples))
        print("         Those labels get fewer examples (or none). This usually means")
        print("         POOL_PATH points at a small DEMO file rather than a full pool.")

    # Step 5: glue the example block in front of the base prompt.
    example_block = "\n\n".join(lines)
    return example_block + "\n\nNow classify this one.\n\n" + base_prompt


# ----------------------------------------------------------------------------------
# Plotting (drawing only — the counts come from metrics.py)
# ----------------------------------------------------------------------------------
def plot_confusion_matrix(matrix,
                          labels: list[str],
                          title: str,
                          xlabel: str = "Predicted",
                          ylabel: str = "Gold") -> None:
    """Draw a confusion matrix as a heatmap (rows = gold, columns = predicted).

    Args:
        matrix: the counts, as sklearn's confusion_matrix returns them.
        labels: the label names, in the same order as the matrix.
        title: the heading to put above it.
        xlabel: what the columns are.
        ylabel: what the rows are.

    Returns:
        Nothing. It draws the picture.
    """
    plt.figure(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------
# Writing the report
# ----------------------------------------------------------------------------------
# The four things that can be wrong when a model gets an item wrong. Fixed on purpose:
# a triage is a judgment you make from a menu, not an essay, and fixed words mean the
# counts say the same thing in every group's report.
#
#   model      - the label is clear, both coders agreed at once, the model still missed
#   scheme     - the item is genuinely borderline under YOUR scheme
#   wording    - the label NAME misleads; a different word might fix it
#   ambiguous  - the item itself is unclear, in a way no scheme would settle
#
# Note what the difference between `scheme` and `wording` is worth: a wording error is
# something your next prompt round can fix, and a scheme error is not.
TRIAGE_CATEGORIES = ["model", "scheme", "wording", "ambiguous"]

def triage_category(reason: str,
                    categories: list[str] | tuple = TRIAGE_CATEGORIES) -> str | None:
    """The category word a triage line starts with, or None if it is not one of ours.

    Args:
        reason: one line of your triage, e.g. "scheme - Move 1/Move 2 boundary".
        categories: the words that count. Left out, the four for the model's errors.
            Pass your own list to recognise a different set of words.

    Returns:
        The category word, or None when the line does not start with one.
    """
    first_word = str(reason).strip().split(" ")[0]
    first_word = first_word.strip("-—:,.").lower()
    if first_word in categories:
        return first_word
    return None



def export_results(track: str,
                   gold: list[dict[str, str]],
                   predictions: list[str],
                   macro_f1_by_round: dict[str, float],
                   out_dir: str | Path,
                   group: str = "",
                   run: str = "",
                   overwrite: bool = False,
                   triage: dict[int, str] | None = None,
                   dev: list[dict[str, str]] | None = None) -> dict[str, Path]:
    """Write your gold set, a predictions CSV, and a one-page report scaffold.

    These three files are what you submit.

    Args:
        track: which dataset track this is.
        gold: whatever you scored - which, from notebook 05, is your TEST half.
        predictions: one predicted label per item, in the same order.
        macro_f1_by_round: your per-round scores, in the order you ran them.
        out_dir: the folder to write the three files into.
        group: added to every filename, so several groups can share one folder.
        run: added to every filename, so a second attempt does not replace the first.
        overwrite: True to replace files that are already there.
        triage: your group's own reading of the errors,
            {item id: "category - reason"}. Give it and section 4 becomes your
            analysis, with the counts at the top and your reason beside each item.
            Leave it off and section 4 is a placeholder asking for exactly that.
        dev: your dev half. Pass it and the report says how many items you tuned on,
            how many you reported on, and that the rounds table is a dev trail with
            one test row at the bottom. A reader cannot tell those apart from the
            numbers alone.

    Returns:
        Where each of the three files was written: {"gold", "csv", "report"}.

    Raises:
        FileExistsError: when any of the three is already there and overwrite is
            False. All three are checked before any is written.

    Example:
        >>> export_results(TRACK, test, predictions, f1_by_round, OUT_DIR,
        ...                group=GROUP, run=RUN, triage=triage, dev=dev)
    """
    output_folder = Path(out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    ### Step 1: work out the three filenames, and refuse to replace any of them ###
    gold_path, csv_path, report_path = _export_paths(output_folder, track, group, run,
                                                     dev)
    # Checked BEFORE writing any of them. Stopping halfway would leave a report that
    # describes one run sitting beside the gold set of another.
    _refuse_to_overwrite(gold_path, overwrite, "items")
    _refuse_to_overwrite(csv_path, overwrite, "rows")
    _refuse_to_overwrite(report_path, overwrite, "sections")

    ### Step 2: the items themselves - the numbers below mean nothing without them ###
    gold_path.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

    ### Step 3: one row per item, as a spreadsheet you can open and sort ###
    records = _prediction_records(gold, predictions)
    pd.DataFrame(records).to_csv(csv_path, index=False)

    ### Step 4: the report ###
    report = _build_report(track, group, gold, dev, records, macro_f1_by_round, triage)
    report_path.write_text(report, encoding="utf-8")

    print("Wrote", gold_path.name + ",", csv_path.name, "and", report_path.name,
          "to", str(output_folder) + "/")
    return {"gold": gold_path, "csv": csv_path, "report": report_path}


def _export_paths(output_folder: Path,
                  track: str,
                  group: str,
                  run: str,
                  dev: list[dict[str, str]] | None) -> tuple:
    """The three files export_results writes: the items, the CSV and the report.

    Args:
        output_folder: the folder they go in.
        track: which dataset track this is.
        group: the group name, or "" when called by hand.
        run: the run name, or "" when called by hand.
        dev: the dev half. With a split, the items saved beside the results are the
            TEST half, so the file is named _test rather than _gold.

    Returns:
        Three paths: the items, the predictions CSV and the report.
    """
    # The same stem config.py builds, from the same three pieces, so the names cannot
    # drift apart. Any of the three may be empty when export_results is called by hand.
    stem = track
    if group != "":
        stem = stem + "_" + group
    if run != "":
        stem = stem + "_" + run

    # With a split, the items saved beside the results are the TEST half, and calling
    # that copy "_gold" would misdescribe it.
    if dev is None:
        gold_path = output_folder / (stem + "_gold.json")
    else:
        gold_path = output_folder / (stem + "_test.json")
    return (gold_path,
            output_folder / (stem + "_predictions.csv"),
            output_folder / (stem + "_report.md"))


def _prediction_records(gold: list[dict[str, str]],
                        predictions: list[str]) -> list[dict]:
    """One row per item: what it was, what the model said, and whether that matched.

    Args:
        gold: the scored items.
        predictions: one predicted label per item, in the same order.

    Returns:
        One dict per item: id, gold, pred, correct, text.
    """
    records = []
    for item, predicted in zip(gold, predictions):
        record = {
            "id": item["id"],
            "gold": item["label"],
            "pred": predicted,
            "correct": item["label"] == predicted,
            "text": item["text"],
        }
        records.append(record)
    return records


def _count_labels(gold: list[dict[str, str]]) -> dict[str, int]:
    """How many gold items carry each label.

    Args:
        gold: the items to count.

    Returns:
        How many items carry each label.
    """
    counts = {}
    for item in gold:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


def _rounds_table(macro_f1_by_round: dict[str, float]) -> str:
    """The rows of the "F1 per round" table, as Markdown.

    Args:
        macro_f1_by_round: your per-round scores, in the order you ran them.

    Returns:
        The table rows, one per round, joined by newlines.
    """
    lines = []
    for round_name in macro_f1_by_round:
        score = macro_f1_by_round[round_name]
        lines.append("| " + round_name + " | " + format(score, ".3f") + " |")
    return "\n".join(lines)


def _final_f1(macro_f1_by_round: dict[str, float]) -> float:
    """The score of the last round we ran.

    Args:
        macro_f1_by_round: your per-round scores, in the order you ran them.

    Returns:
        The last score, or nan when there are no rounds.
    """
    if len(macro_f1_by_round) == 0:
        return float("nan")
    all_scores = list(macro_f1_by_round.values())
    return all_scores[-1]


def _error_line(record: dict) -> str:
    """One wrong item, as a Markdown bullet.

    Args:
        record: one row from _prediction_records.

    Returns:
        The bullet, with the text cut to 120 characters.
    """
    snippet = str(record["text"])[:120]
    return ("- **id " + str(record["id"]) + "** gold `" + str(record["gold"])
            + "` -> pred `" + str(record["pred"]) + "`: " + snippet)


def _error_examples(wrong_records: list[dict],
                    triage: dict[int, str] | None) -> str:
    """The wrong items, as Markdown.

    Args:
        wrong_records: the rows the model got wrong.
        triage: your group's reading of them. With a triage, EVERY triaged one is
            listed with your reason beside it - that judgment is the section.
            Without one, the first five are listed as a prompt to go and make it.

    Returns:
        The bullets, joined by newlines.
    """
    lines = []
    if triage:
        for record in wrong_records:
            reason = triage.get(record["id"])
            if reason is None:
                reason = triage.get(str(record["id"]))   # keys survive a JSON trip as text
            if reason is None:
                continue
            lines.append(_error_line(record))
            lines.append("  - " + str(reason))
    else:
        for record in wrong_records:
            if len(lines) >= 5:
                break
            lines.append(_error_line(record))

    if len(lines) == 0:
        return "- (no errors to show)"
    return "\n".join(lines)


def _triage_summary(triage: dict[int, str] | None,
                    wrong_records: list[dict]) -> str:
    """The headline of the error section: what the errors were CAUSED by.

    "6 of 14 are the scheme's fault" is a finding; "F1 was 0.62" is not.

    Args:
        triage: your group's reading of the errors.
        wrong_records: the rows the model got wrong, counted into the headline.

    Returns:
        The headline line, or "" when there is no triage.
    """
    if not triage:
        return ""
    all_categories = TRIAGE_CATEGORIES + ["uncategorised"]
    counts = {}
    for category in all_categories:
        counts[category] = 0
    for item_id in triage:
        category = triage_category(triage[item_id])
        if category is None:
            category = "uncategorised"
        counts[category] = counts[category] + 1
    parts = []
    for category in all_categories:
        if counts[category] > 0:
            parts.append(str(counts[category]) + " " + category)
    return ("- **Your triage of " + str(len(triage)) + " of " + str(len(wrong_records))
            + " errors:** " + " / ".join(parts) + "\n\n")


def _build_report(track: str,
                  group: str,
                  gold: list[dict[str, str]],
                  dev: list[dict[str, str]] | None,
                  records: list[dict],
                  macro_f1_by_round: dict[str, float],
                  triage: dict[int, str] | None) -> str:
    """The one-page report, section by section.

    Anything in _italics_ is a placeholder YOU replace - a section left as the
    placeholder scores zero.

    Args:
        track: which dataset track this is.
        group: the group name.
        gold: the scored items.
        dev: the dev half, when there was a split.
        records: the rows from _prediction_records.
        macro_f1_by_round: your per-round scores.
        triage: your group's reading of the errors.

    Returns:
        The whole report, as Markdown.
    """
    labels = label_set(gold)
    label_counts = _count_labels(gold)
    round_rows = _rounds_table(macro_f1_by_round)
    final_f1 = _final_f1(macro_f1_by_round)

    wrong_records = []
    for record in records:
        if not record["correct"]:
            wrong_records.append(record)

    report = ""
    report = report + "# One-page report - " + track + "\n\n"
    if group != "":
        report = report + "Group: " + group + "\n\n"
    report = report + "## 1. Scheme & gold\n"
    report = report + "- **Labels:** " + ", ".join(labels) + "\n"
    if dev is None:
        report = report + ("- **Gold set:** " + str(len(gold))
                           + " items sampled from the pool; per-label counts: "
                           + str(label_counts) + "\n")
    else:
        report = report + ("- **Gold set:** " + str(len(dev) + len(gold))
                           + " items sampled and double-coded, split into "
                           + str(len(dev)) + " dev (tuned on) and " + str(len(gold))
                           + " test (held out, scored once). Per-label counts of the "
                           "test half: " + str(label_counts) + "\n")
    report = report + ("- **QC / adjudication:** _<your percent agreement and kappa, how "
                       "many labels your adjudication changed, which label pair caused "
                       "the most disagreement, and what your scheme now says about it>_\n\n")
    report = report + "## 2. Prompt iterations\n"
    report = report + "| Round | Macro-F1 |\n|---|---|\n" + round_rows + "\n\n"
    if dev is not None:
        report = report + ("_Every round above is a **dev** score. The last row is the "
                           "held-out test set, scored once._\n\n")
    report = report + ("_For each round: what did you change, and WHY did you expect it to "
                       "help?_\n\n")
    report = report + "## 3. Evaluation\n"
    if dev is None:
        report = report + ("- **Final macro-F1:** " + format(final_f1, ".3f") + " on "
                           + str(len(gold)) + " gold items.\n")
    else:
        report = report + ("- **Final macro-F1:** " + format(final_f1, ".3f") + " on "
                           + str(len(gold)) + " held-out test items.\n")
        report = report + ("- _How far below your best dev round did that land? That "
                           "distance is roughly how much of your improvement was tuning "
                           "to those particular dev items._\n")
    report = report + ("- Per-class precision/recall/F1 and the confusion matrix are in "
                       "the notebook output.\n")
    report = report + "- _Which class did worst, and what did it get confused with?_\n\n"
    report = report + "## 4. Error analysis\n"
    report = report + _triage_summary(triage, wrong_records)
    report = report + _error_examples(wrong_records, triage) + "\n\n"
    if triage:
        report = report + ("_What do the **scheme** errors have in common? That pattern "
                           "is the finding - say what your scheme would have to add to "
                           "settle those items._\n\n")
    else:
        report = report + ("_For each miss: is it the **model's** fault or the "
                           "**scheme's** (a genuinely borderline item)? Give a reason, "
                           "not a verdict. Passing `triage=` to export_results puts your "
                           "reasons here automatically._\n\n")
    report = report + "## 5. Limitations\n"
    report = report + ("_Replace these three generic lines with at least two limitations "
                       "that apply to YOUR run._\n")
    report = report + "- LLM output is stochastic; a re-run can shift the numbers.\n"
    report = report + ("- Contamination risk: these are published datasets the model may "
                       "have seen.\n")
    report = report + ("- " + str(len(gold)) + " items is a small sample - treat per-class "
                       "scores for rare labels with caution.\n")
    if dev is not None:
        report = report + ("- The prompt was tuned on " + str(len(dev)) + " dev items and "
                           "reported on " + str(len(gold)) + " held-out ones. A study that "
                           "could support a claim about the corpus would need hundreds of "
                           "each; at roughly 4.4 seconds per API call, that was not "
                           "available in a five-day course. The discipline here is real; "
                           "the confidence interval on the number is wide.\n")

    return report
