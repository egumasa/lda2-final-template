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
def _looks_like_rate_limit(error):
    """True if an exception looks like a 'too many requests' / quota error.

    We match on the text rather than a specific exception class, so the SAME guard
    works for both backends (the Colab built-in Gemini and the Gemini API), which
    raise different exception types.
    """
    text = str(error).lower()
    for signal in ["429", "resource_exhausted", "rate limit", "quota", "too many requests"]:
        if signal in text:
            return True
    return False


def _looks_like_daily_quota(error):
    """True if the rate-limit error is the PER-DAY cap (not the per-minute one).

    A daily cap cannot be waited out in a single sitting, so retrying is pointless -
    we want to stop immediately and tell the user how to actually get unblocked.
    Gemini names the daily quota 'GenerateRequestsPerDay...' / 'PerDay' in the error.
    """
    text = str(error).lower()
    return "perday" in text or "per day" in text or "requests_per_day" in text


def _suggested_delay(error, fallback):
    """Pull the server's own 'please retry in Ns' hint out of the error, if present.

    Gemini includes a RetryInfo like 'Please retry in 7.17s.' - honoring it is more
    accurate than a guessed backoff. Falls back to `fallback` seconds when absent.
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


def _wait_our_turn():
    """Sleep until at least _MIN_INTERVAL seconds have passed since the last call."""
    global _LAST_CALL_TIME
    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_CALL_TIME)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TIME = time.monotonic()


def generate_text(prompt):
    """Send one prompt to the model and return its text reply.

    Three guards, in order:
      1) wait our turn, so we stay under the per-minute cap in the first place
      2) on a per-minute rate-limit error, sleep (honoring the delay the server
         suggests, when it gives one) and try again a few times
      3) on a PER-DAY quota error, stop at once - retrying cannot help today - and say
         how to get unblocked.
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


def _resolve_gemini_key():
    """Find a Gemini API key: Colab Secrets first, then the environment.

    The Colab Secrets panel (the little key icon in the sidebar) is where Day 3 tells
    you to put your key - but Colab does NOT copy secrets into the environment, so we
    have to ask for it explicitly. That is the only reason this function exists.
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


def _connect_to_gemini_api(key):
    """Set up the API-key connection. Returns the model id we will be using.

    temperature=0 + a fixed seed = the same prompt gives the same answer every run.
    That is what "reproducible" means in practice, and it is the whole reason we prefer
    the key path over Colab's built-in model.
    """
    global _GENAI_CLIENT, _GENAI_CONFIG
    from google import genai
    from google.genai import types
    _GENAI_CLIENT = genai.Client(api_key=key)
    _GENAI_CONFIG = types.GenerateContentConfig(temperature=0, seed=42)
    return os.environ.get("LLM_MODEL", MODEL_ID)


_GENAI_CLIENT = None
_GENAI_CONFIG = None


def _call_gemini_api(prompt):
    """Send one prompt through the API-key connection."""
    response = _GENAI_CLIENT.models.generate_content(
        model=_MODEL_IN_USE, contents=prompt, config=_GENAI_CONFIG)
    return response.text


def _call_colab_gemini(prompt):
    """Send one prompt through Colab's built-in, keyless Gemini."""
    from google.colab import ai
    return ai.generate_text(prompt)


def make_backend():
    """Connect to the model. Returns (generate_text, backend_name).

    We prefer an API KEY, because the API lets us pin temperature=0 and a fixed seed,
    which is what makes a run reproducible. Colab's keyless built-in Gemini is only a
    fallback: it works with zero setup but exposes no temperature or seed, so the same
    prompt can give different answers - fine for a quick look, NOT for your final
    frozen run.

    Safe to call more than once: after the first time it just hands back the connection
    it already made, so re-running the SETUP cell does not reset the pacing clock and
    let a burst of calls through.
    """
    global _CALL_MODEL, _BACKEND_NAME, _MIN_INTERVAL, _MODEL_IN_USE
    if _CALL_MODEL is not None:
        return generate_text, _BACKEND_NAME   # already connected - reuse it, stay paced

    # Option 1 (preferred): the Gemini API with your own key. Reproducible.
    key = _resolve_gemini_key()
    if key:
        _MODEL_IN_USE = _connect_to_gemini_api(key)
        _CALL_MODEL = _call_gemini_api
        _BACKEND_NAME = ("Gemini API (" + _MODEL_IN_USE
                         + ", temperature=0, seed=42)")
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
        _MODEL_IN_USE = "colab built-in gemini"
        _CALL_MODEL = _call_colab_gemini
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


def setup():
    """Connect to the model and say what we connected to. Run this once, at the top."""
    _, backend_name = make_backend()
    print("LLM backend:", backend_name)


def model_in_use():
    """The model id the current connection is using, for the test log."""
    if not _MODEL_IN_USE:
        return MODEL_ID
    return _MODEL_IN_USE


def _default_backend():
    """The function that sends a prompt, for when a caller did not pass one in.

    This is generate_text - the paced, retrying one - never the raw connection.
    """
    sender, _ = make_backend()
    return sender


# ----------------------------------------------------------------------------------
# Reading data
# ----------------------------------------------------------------------------------
def load_gold(url_or_path):
    """Read a gold/pool file. Each item looks like {"id": 1, "text": "...", "label": "..."}."""
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


def load_prompt(path):
    """Read a prompt template from a text file in the prompts/ folder.

    Keeping the prompt in its own file (instead of pasting it into the notebook)
    means you iterate by editing the file — and each version is easy to save and
    compare. The file must contain the placeholder {text}, where each sentence
    is slotted in.
    """
    prompt = Path(path).read_text(encoding="utf-8").strip()
    if "{text}" not in prompt:
        print("WARNING: the prompt file has no {text} placeholder — the sentence "
              "will not be inserted anywhere.")
    print("Loaded prompt from", path, "(", len(prompt), "characters ).")
    return prompt


# ----------------------------------------------------------------------------------
# Not overwriting work you already did
# ----------------------------------------------------------------------------------
# Re-running a cell is the most ordinary thing you can do in a notebook, and until this
# check existed it silently replaced whatever was already in the file. A gold set is a
# morning of two people's annotation; nothing warned you it had gone. So every save in
# this file stops instead, and tells you the two ways forward.
def _refuse_to_overwrite(path, overwrite, what):
    """Stop rather than replace a file that already exists."""
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


def _refuse_to_load_missing(path, what, made_by):
    """Explain a missing handoff file: which notebook makes it, and what to do.

    Notebooks 04 and 05 read files that an EARLIER notebook wrote. Arriving at 05
    without having finished 04 is the ordinary way to meet this, and a bare
    FileNotFoundError pointing into open() does not say that.
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


def _describe_contents(path, what):
    """Say what is in a file, for the message above - how many items, or how big."""
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
def save_predictions(predictions, path, overwrite=False):
    """Write a list of predicted labels to a JSON file - this is 'freezing' a run."""
    output_path = Path(path)
    _refuse_to_overwrite(output_path, overwrite, "predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Froze", len(predictions), "predictions to", str(output_path))
    return output_path


def load_predictions(url_or_path):
    """Read a frozen predictions list back - a local path or a URL."""
    if str(url_or_path).startswith("http"):
        raw_bytes = urllib.request.urlopen(url_or_path).read()
        predictions = json.loads(raw_bytes.decode("utf-8"))
    else:
        _refuse_to_load_missing(url_or_path, "frozen predictions",
                                "notebook 04_prompt")
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
def save_test_run(predictions, path, overwrite=False):
    """Freeze a run on the TEST set. Returns (path_written, attempt_number).

    First call writes the path you gave it. A second call does not replace that file -
    it writes ..._predictions_attempt2.json beside it, and says so.
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


def _next_attempt_number(first_path):
    """Find the lowest attempt number that is not on disk yet."""
    highest = 1
    pattern = first_path.stem + "_attempt*" + first_path.suffix
    for existing in first_path.parent.glob(pattern):
        digits = existing.stem.split("_attempt")[-1]
        if digits.isdigit() and int(digits) > highest:
            highest = int(digits)
    return highest + 1


def log_test_run(log_path, record):
    """Append one record to the test log. Never refuses, never replaces.

    One JSON object per line (a .jsonl file), rather than one JSON list, because a list
    would have to be read and rewritten whole every time - and a rewrite interrupted
    halfway loses exactly the history this file exists to keep.
    """
    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    log_file = open(output_path, "a", encoding="utf-8")
    log_file.write(line + "\n")
    log_file.close()
    return output_path


def read_test_log(log_path):
    """Read the test log back as a table, oldest scoring first."""
    output_path = Path(log_path)
    if not output_path.exists():
        print("No test log at", str(output_path), "- notebook 04 writes it when you")
        print("score the held-out set.")
        return pd.DataFrame()
    records = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return pd.DataFrame(records)


def record_test_scoring(log_path, macro_f1, attempt, pred_path, prompt, prompt_file,
                        gold_items, dev_f1=None, round_key="", note="",
                        predictions=None):
    """Write down one scoring of the held-out set, and print it back."""
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
def freeze_test_run(prompt, test, f1_by_round, pred_path, log_path, rounds_path,
                    prompt_file, dev_f1=None, note="", ordered=False, labels=None,
                    key="FINAL test (held out)", generate_text=None):
    """Run `prompt` on the held-out set, freeze it, score it, and log the attempt.

    Returns the macro-F1. `f1_by_round` is updated in place, with the test score added
    LAST so the table reads as the dev rounds in order with the held-out score at the
    bottom, and then written to `rounds_path` for notebook 05.

    `note` is not decoration. A second attempt with the SAME prompt is that prompt run
    twice; a second attempt with a DIFFERENT one is a prompt tuned after seeing the
    held-out set. Say which it was.
    """
    from metrics import evaluate      # imported here: metrics imports this module

    predictions = run_prompt(prompt, test, labels=labels, generate_text=generate_text)
    written_path, attempt = save_test_run(predictions, pred_path)
    print("Froze to", Path(written_path).name, "· attempt", attempt)

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
    # notebook 05 needs it. Overwrite: re-running this cell means re-running the whole
    # held-out scoring, and the table has to match the log it was written beside.
    save_json(f1_by_round, rounds_path, what="rounds", overwrite=True)
    return macro_f1


# ----------------------------------------------------------------------------------
# Handing work from one notebook to the next
# ----------------------------------------------------------------------------------
# Notebooks 01-05 run in separate sessions, often on separate days and different
# people's runtimes. Anything one of them produces and another needs has to go through
# a FILE - a variable in someone else's Colab is not a handoff. These two do that for
# any JSON-able thing: your sample, your gold set, your per-round F1 table.
def save_json(data, path, what="items", overwrite=False):
    """Write anything JSON-able to a file, making the folder if it is missing."""
    output_path = Path(path)
    _refuse_to_overwrite(output_path, overwrite, what)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", len(data), what, "to", str(output_path))
    return output_path


def load_json(path, what="items", made_by=None):
    """Read back what save_json wrote.

    `made_by` names the notebook that writes this file, so that a missing one says
    where it comes from. Leave it off and the message stays general.
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
def reid(items):
    """Give the items fresh id numbers 1, 2, 3, ... in their current order."""
    renumbered = []
    next_id = 1
    for item in items:
        new_item = dict(item)          # make a copy so we do not change the original
        new_item["id"] = next_id
        renumbered.append(new_item)
        next_id = next_id + 1
    return renumbered


# The three samplers, behind one name. Notebook 02 used to spell this out as an
# if/elif over three calls, which meant three hard-coded sizes (40 items, 10 documents,
# 4 sentences each) that had nothing to do with n_per_class in config.yaml - so
# choosing "random" or "by_document" silently made that setting dead. Here every
# strategy is sized from the same setting, and the choice stays one word.
def sample(pool, strategy, n_per_class, seed=42, n_per_doc=4):
    """Draw a sample using one of the three strategies, all sized from n_per_class.

        "balanced"      up to n_per_class items of EACH label
        "random"        the same TOTAL, drawn without regard to label
        "by_document"   whole passages, enough of them to reach that total
                        (cars50 and raamove only - other tracks have no documents)
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


def sample_pool(pool, n_per_class, seed=42):
    """Pick up to n_per_class items for EACH label, chosen at random.

    Rare labels simply give fewer items - that is a property of the data.
    The same seed always gives the same sample; a different seed gives a
    different draw (so different groups can get different subsets).

    This is the BALANCED strategy: it forces the classes level so that per-class
    precision and recall are readable and the confusion matrix is not dominated by
    one huge class. The cost is that your sample no longer looks like the corpus.
    See sample_random and sample_by_document for the other two positions.
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


def _report_draw(sampled, pool, what):
    """Print what a draw produced, and warn if it swallowed the pool.

    Shared by all three sampling strategies, so that whichever one a group picks,
    they see the same two things: the per-label counts their choice produced, and a
    warning if there is no pool left for build_fewshot to draw examples from.
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


def sample_random(pool, n_total, seed=42):
    """Draw n_total items at random, ignoring the labels entirely.

    The corpus as it actually is: every item equally likely, so each label turns up
    roughly as often as it does in the pool. That is the honest thing if you want to
    say something about the corpus - and the awkward thing if a label is rare, because
    a rare label will come back with one or two items, or none at all, and precision
    and recall on a class of one mean very little.

    Compare sample_pool, which forces the classes level instead.
    """
    random_generator = random.Random(seed)

    # Copy before shuffling: shuffling the caller's pool in place would quietly change
    # the order of the list they are still holding.
    shuffled = list(pool)
    random_generator.shuffle(shuffled)
    sampled = reid(shuffled[:n_total])

    _report_draw(sampled, pool, "Sampled at random")
    return sampled


def sample_by_document(pool, n_docs, n_per_doc, seed=42):
    """Pick whole documents first, then sentences inside them.

    Forty sentences drawn from forty abstracts and forty sentences drawn from three
    are both "forty sentences", and they support very different claims. This strategy
    makes that choice explicit: n_docs documents, n_per_doc sentences from each.

    Only the tracks whose items remember where they came from can do this - cars50,
    cars50_step and raamove carry `doc_id`. On the others it stops and says so.
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


def label_set(gold):
    """Return the sorted list of labels that appear in a gold set."""
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
def split_dev_test(gold, dev, seed=42, by_document=False):
    """Split an adjudicated gold set into (dev, test).

    `dev` says how big the dev half is, and how you write it says which you meant:

        dev=3        a whole number — that many dev items per label
        dev=0.35     a decimal between 0 and 1 — that proportion of each label's items

    Both stratify by label, so every label is represented on both sides of the line
    wherever the data allows it. The same seed always gives the same split.

    Set by_document=True if you drew your sample with sample_by_document, so that no
    passage has some of its sentences in dev and the rest in test.
    """
    dev_per_class, dev_fraction = _read_dev_size(dev)

    if by_document:
        dev, test = _split_by_document(gold, dev_per_class, dev_fraction, seed)
    else:
        dev, test = _split_by_label(gold, dev_per_class, dev_fraction, seed)

    _report_split(dev, test, by_document)
    return dev, test


def _read_dev_size(dev):
    """Read one `dev` setting as (dev_per_class, dev_fraction) - exactly one of them set.

    A count and a proportion are one decision, so config.yaml holds one key and the way
    the number is written says which was meant: 3 is three items per label, 0.35 is a
    third of each label. The two names survive below this line only because the split
    functions need to tell the cases apart.
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


def _dev_target(bucket_size, dev_per_class, dev_fraction):
    """How many of a label's items should go to dev, before the rare-class clamp."""
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


def _split_by_label(gold, dev_per_class, dev_fraction, seed):
    """Stratified split: each label is divided in the same proportion."""
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
    # ids. Every id here was fixed when the annotation sheet was built, and notebook 05
    # joins on them to ask which of the model's errors are also the items your two
    # coders disagreed about. Renumbering would leave that join silently meaningless.
    random_generator.shuffle(dev)
    random_generator.shuffle(test)
    return dev, test


def _split_by_document(gold, dev_per_class, dev_fraction, seed):
    """Split whole documents, so no passage has sentences on both sides of the line."""
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


def _report_split(dev, test, by_document=False):
    """Print what the split produced, and warn about labels that fell off one side."""
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

    print("        dev is what you iterate against in notebook 04. test is opened once,")
    print("        in the last step of that notebook, and scored once.")


def _label_counts(items):
    """Count items per label - the shared shape behind every 'per-label counts' line."""
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
def extract_label(reply, labels):
    """Figure out which of the known labels the model's reply is pointing at.

    Returns "??" when we cannot find any known label in the reply.
    """
    reply_text = str(reply).strip()
    reply_lowercased = reply_text.lower()

    # Step 1: collect every known label whose name appears in the reply.
    labels_found = []
    for label in labels:
        if label.lower() in reply_lowercased:
            labels_found.append(label)

    # Step 2: if we found one or more, keep the longest (most specific) one.
    if len(labels_found) > 0:
        longest_label = labels_found[0]
        for label in labels_found:
            if len(label) > len(longest_label):
                longest_label = label
        return longest_label

    # Step 3: special case for "Move 1/2/3" labels - look for a bare digit.
    has_move_labels = False
    for label in labels:
        if label.lower().startswith("move "):
            has_move_labels = True
    if has_move_labels:
        match = re.search(r"\b([1-9])\b", reply_text)
        if match is not None:
            candidate = "Move " + match.group(1)
            if candidate in labels:
                return candidate

    # Step 4: nothing matched.
    return "??"


def run_prompt(prompt, gold, labels=None, generate_text=None):
    """Ask the model to label every item, and collect the predicted labels.

    Same call as Day 3: run_prompt(PROMPT, gold). The two optional arguments are
    worked out for you - `labels` from the gold set, and the model connection from
    the Setup cell - so you only pass them if you want something different.
    """
    if labels is None:
        labels = label_set(gold)
    if generate_text is None:
        generate_text = _default_backend()

    # A prompt that asks for {context} on a track whose items have none would quietly
    # send the model an empty passage, once per item, and report a number as if it had
    # tested something. Say so instead.
    if "{context}" in prompt and not any(item.get("context") for item in gold):
        print("WARNING: this prompt uses {context}, but none of these items carry one. "
              "Only the rhetorical-move tracks (cars50, raamove) do. The model is about "
              "to be shown an empty passage " + str(len(gold)) + " times.")

    predictions = []
    total = len(gold)
    position = 0
    for item in gold:
        position = position + 1
        # Put this item's sentence into the prompt where {text} is - and its passage
        # where {context} is, on the tracks that carry one. A prompt that does not
        # mention {context} simply ignores it.
        filled_prompt = prompt.format(text=item["text"],
                                      context=item.get("context", ""))
        reply = generate_text(filled_prompt)
        predicted_label = extract_label(reply, labels)
        predictions.append(predicted_label)
        # Print a small progress note every 10 items.
        if position % 10 == 0:
            print("  ...", position, "/", total, "done")

    # Count how many replies we could not turn into a valid label.
    number_unparseable = 0
    for label in predictions:
        if label == "??":
            number_unparseable = number_unparseable + 1
    print("Got", len(predictions), "predictions (", number_unparseable, "could not be parsed).")
    return predictions


# ----------------------------------------------------------------------------------
# Few-shot examples
# ----------------------------------------------------------------------------------
def build_fewshot(base_prompt, pool, gold, labels=None, shots_per_class=1, seed=42):
    """Put a few labeled examples (taken from the pool) in front of the prompt.

    We NEVER use an item that is in the gold set as an example, otherwise we
    would be showing the model the very answers we are testing it on. Items are
    matched by their TEXT, not their id, because sampling renumbers the ids.
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
def plot_confusion_matrix(matrix, labels, title, xlabel="Predicted", ylabel="Gold"):
    """Draw a confusion matrix as a heatmap (rows = gold, columns = predicted)."""
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


def triage_category(reason):
    """The category word a triage line starts with, or None if it is not one of ours."""
    first_word = str(reason).strip().split(" ")[0]
    first_word = first_word.strip("-—:,.").lower()
    if first_word in TRIAGE_CATEGORIES:
        return first_word
    return None



def export_results(track, gold, predictions, macro_f1_by_round, out_dir, group="",
                   run="", overwrite=False, triage=None, dev=None):
    """Write your gold set, a predictions CSV, and a one-page report scaffold.

    `group` and `run` are added to every filename, so several groups can drop their
    results in one folder without overwriting each other, and a second attempt does not
    replace your first. These files are what you submit.

    `triage` is your group's own reading of the errors - {item id: "category - reason"}.
    Give it and section 4 becomes your analysis, with the counts at the top and your
    reason beside each item. Leave it off and section 4 is a placeholder asking for
    exactly that, which is worth less to you and to whoever reads the report.

    `gold` here is whatever you scored - which, from notebook 05, is your TEST half.
    Pass `dev=` as well and the report says so: how many items you tuned on, how many
    you reported on, and that the rounds table is a dev trail with one test row at the
    bottom. A reader cannot tell those apart from the numbers alone.
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


def _export_paths(output_folder, track, group, run, dev):
    """The three files export_results writes: the items, the CSV and the report."""
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


def _prediction_records(gold, predictions):
    """One row per item: what it was, what the model said, and whether that matched."""
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


def _count_labels(gold):
    """How many gold items carry each label."""
    counts = {}
    for item in gold:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


def _rounds_table(macro_f1_by_round):
    """The rows of the "F1 per round" table, as Markdown."""
    lines = []
    for round_name in macro_f1_by_round:
        score = macro_f1_by_round[round_name]
        lines.append("| " + round_name + " | " + format(score, ".3f") + " |")
    return "\n".join(lines)


def _final_f1(macro_f1_by_round):
    """The score of the last round we ran."""
    if len(macro_f1_by_round) == 0:
        return float("nan")
    all_scores = list(macro_f1_by_round.values())
    return all_scores[-1]


def _error_line(record):
    """One wrong item, as a Markdown bullet."""
    snippet = str(record["text"])[:120]
    return ("- **id " + str(record["id"]) + "** gold `" + str(record["gold"])
            + "` -> pred `" + str(record["pred"]) + "`: " + snippet)


def _error_examples(wrong_records, triage):
    """The wrong items, as Markdown.

    With a triage, EVERY triaged one is listed with your group's reason beside it -
    that judgment is the section. Without one, the first five are listed as a prompt to
    go and make it.
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


def _triage_summary(triage, wrong_records):
    """The headline of the error section: what the errors were CAUSED by.

    "6 of 14 are the scheme's fault" is a finding; "F1 was 0.62" is not.
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


def _build_report(track, group, gold, dev, records, macro_f1_by_round, triage):
    """The one-page report, section by section.

    Anything in _italics_ is a placeholder YOU replace - a section left as the
    placeholder scores zero.
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
