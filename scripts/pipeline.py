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


def _throttle_and_retry(call_model, default_min_interval):
    """Wrap a raw 'prompt -> text' function with free-tier friendliness.

    Free-tier Gemini caps requests per minute. If we fire faster than that, we get a
    429 RESOURCE_EXHAUSTED / rate-limit error. Guards:
      1) MIN_INTERVAL - wait at least this many seconds between calls, so we stay
         under the per-minute cap in the first place.
      2) on a per-minute rate-limit error, sleep (honoring the server's suggested
         delay when given) and retry a few times.
      3) on a PER-DAY quota error, stop immediately - retrying cannot help today -
         and raise a clear message about how to get unblocked.

    `default_min_interval` depends on which backend we picked (the API key path can
    go faster than colab.ai). Both it and the retry count are tunable via the
    LLM_MIN_INTERVAL / LLM_MAX_RETRIES environment variables.
    """
    min_interval = float(os.environ.get("LLM_MIN_INTERVAL", str(default_min_interval)))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    last_call_time = [0.0]   # a list so the inner function can update it

    def generate_text(prompt):
        for attempt in range(max_retries + 1):
            # Guard 1: pace ourselves to at most one call per min_interval seconds.
            wait = min_interval - (time.monotonic() - last_call_time[0])
            if wait > 0:
                time.sleep(wait)
            try:
                last_call_time[0] = time.monotonic()
                return call_model(prompt)
            except Exception as error:
                # Non-rate-limit errors are real bugs - let them surface.
                if not _looks_like_rate_limit(error):
                    raise
                # Guard 3: a DAILY cap will not clear today - stop now with advice.
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
                # Guard 2: a per-minute cap - back off (honoring the server hint) and retry.
                if attempt == max_retries:
                    raise
                backoff = _suggested_delay(error, min_interval * (attempt + 1))
                print("  (rate limited - waiting", round(backoff), "s then retrying)")
                time.sleep(backoff)
        # Should be unreachable, but keeps the function honest.
        raise RuntimeError("The model kept returning rate-limit errors after retries.")

    return generate_text


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


# Once make_backend() has run, its result is remembered here so that re-running the
# Setup cell does not build a second backend (which would reset the pacing clock and
# risk a burst of calls). run_prompt() reads this when you do not pass a backend in.
_BACKEND = None


def make_backend():
    """Return (generate_text, backend_name) - a function that sends a prompt and
    returns the model's text reply, plus a label describing what we connected to.

    We prefer an API KEY, because the API lets us pin temperature=0 and a fixed seed,
    which is what makes a run reproducible. Colab's keyless built-in Gemini is only a
    fallback: it works with zero setup but exposes no temperature or seed, so the same
    prompt can give different answers - fine for a quick look, NOT for your final
    frozen run.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND                      # already built - reuse it, stay paced

    # Option 1 (preferred): the Gemini API with your own key. Reproducible.
    key = _resolve_gemini_key()
    if key:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        model = os.environ.get("LLM_MODEL", MODEL_ID)
        # temperature=0 + a fixed seed = the same prompt gives the same answer every
        # run. This is what "reproducible" means in practice, and it is the whole
        # reason we prefer the key path over Colab's built-in model.
        config = types.GenerateContentConfig(temperature=0, seed=42)

        def call_model(prompt):
            response = client.models.generate_content(
                model=model, contents=prompt, config=config)
            return response.text

        label = "Gemini API (" + model + ", temperature=0, seed=42)"
        # 4.4s between calls keeps us under the 15-requests-per-minute free-tier cap.
        _BACKEND = (_throttle_and_retry(call_model, 4.4), label)
        return _BACKEND

    # Option 2 (fallback): Colab's free built-in Gemini. No key, but not reproducible.
    try:
        from google.colab import ai

        def call_model(prompt):
            return ai.generate_text(prompt)

        print("WARNING: no API key found, so we are using Colab's built-in Gemini.")
        print("         It has no temperature or seed setting, so the same prompt can")
        print("         give different answers - your numbers will NOT be reproducible.")
        print("         Put your key in the Colab Secrets panel as GEMINI_API_KEY")
        print("         before your final run. A free key: aistudio.google.com/apikey")
        # colab.ai publishes no rate limit, so pace conservatively.
        _BACKEND = (_throttle_and_retry(call_model, 13.2), "Colab Gemini (non-reproducible)")
        return _BACKEND
    except ImportError:
        pass

    # Option 3: nothing available - tell the user what to do.
    raise RuntimeError(
        "No LLM backend found. Either set GEMINI_API_KEY - in Colab via the Secrets "
        "panel (the key icon in the sidebar), or in a .env file when running locally - "
        "or run this notebook in Google Colab, which has a free built-in Gemini that "
        "needs no key. A free key takes a minute: https://aistudio.google.com/apikey"
    )


def setup():
    """Connect to the model and say what we connected to. Run this once, at the top.

    Safe to re-run: the backend is built only the first time, so re-running this cell
    will not reset the pacing clock.
    """
    generate_text, backend_name = make_backend()
    print("LLM backend:", backend_name)


def _default_backend():
    """The backend to use when a caller did not pass one in explicitly."""
    generate_text, backend_name = make_backend()
    return generate_text


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
        "  * Meant to REPLACE it? Add overwrite=True inside the brackets of the call\n"
        "    you just ran."
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
        opened_file = open(url_or_path, encoding="utf-8")
        predictions = json.loads(opened_file.read())
        opened_file.close()
    print("Loaded", len(predictions), "frozen predictions.")
    return predictions


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


def load_json(path, what="items"):
    """Read back what save_json wrote."""
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
        raise KeyError(
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
                   run="", overwrite=False, triage=None):
    """Write your gold set, a predictions CSV, and a one-page report scaffold.

    `group` and `run` are added to every filename, so several groups can drop their
    results in one folder without overwriting each other, and a second attempt does not
    replace your first. These files are what you submit.

    `triage` is your group's own reading of the errors - {item id: "category - reason"}.
    Give it and section 4 becomes your analysis, with the counts at the top and your
    reason beside each item. Leave it off and section 4 is a placeholder asking for
    exactly that, which is worth less to you and to whoever reads the report.
    """
    output_folder = Path(out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    labels = label_set(gold)

    # Every file we write starts with the same stem, e.g. "cars50_kimura_v1".
    stem = track
    if group != "":
        stem = stem + "_" + group
    if run != "":
        stem = stem + "_" + run

    gold_path = output_folder / (stem + "_gold.json")
    csv_path = output_folder / (stem + "_predictions.csv")
    report_path = output_folder / (stem + "_report.md")

    # Check all three BEFORE writing any of them. Stopping halfway would leave a report
    # that describes one run sitting beside the gold set of another.
    _refuse_to_overwrite(gold_path, overwrite, "items")
    _refuse_to_overwrite(csv_path, overwrite, "rows")
    _refuse_to_overwrite(report_path, overwrite, "sections")

    # Save the gold set alongside the results: the numbers below mean nothing
    # without the exact items they were computed on.
    gold_path.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build one row per item for the CSV.
    records = []
    for item, predicted in zip(gold, predictions):
        is_correct = (item["label"] == predicted)
        record = {
            "id": item["id"],
            "gold": item["label"],
            "pred": predicted,
            "correct": is_correct,
            "text": item["text"],
        }
        records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(csv_path, index=False)

    # Count how many gold items carry each label.
    label_counts = {}
    for item in gold:
        label = item["label"]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] = label_counts[label] + 1

    # Build the rows of the "F1 per round" table.
    round_lines = []
    for round_name in macro_f1_by_round:
        score = macro_f1_by_round[round_name]
        round_lines.append("| " + round_name + " | " + format(score, ".3f") + " |")
    round_rows = "\n".join(round_lines)

    # The final F1 is the score of the last round we ran.
    if len(macro_f1_by_round) > 0:
        all_scores = list(macro_f1_by_round.values())
        final_f1 = all_scores[-1]
    else:
        final_f1 = float("nan")

    # The wrong items. With a triage, EVERY triaged one is listed, with your group's
    # reason beside it - that judgment is the section. Without one, the first five are
    # listed as a prompt to go and make it.
    wrong_records = []
    for record in records:
        if not record["correct"]:
            wrong_records.append(record)

    error_lines = []
    if triage:
        for record in wrong_records:
            reason = triage.get(record["id"])
            if reason is None:
                reason = triage.get(str(record["id"]))     # keys survive a JSON trip as text
            if reason is None:
                continue
            snippet = str(record["text"])[:120]
            error_lines.append("- **id " + str(record["id"]) + "** gold `"
                               + str(record["gold"]) + "` -> pred `"
                               + str(record["pred"]) + "`: " + snippet)
            error_lines.append("  - " + str(reason))
    else:
        for record in wrong_records:
            if len(error_lines) >= 5:
                break
            snippet = str(record["text"])[:120]
            error_lines.append("- **id " + str(record["id"]) + "** gold `"
                               + str(record["gold"]) + "` -> pred `"
                               + str(record["pred"]) + "`: " + snippet)

    if len(error_lines) == 0:
        error_examples = "- (no errors to show)"
    else:
        error_examples = "\n".join(error_lines)

    # The headline of the error section: what the errors were CAUSED by, in your own
    # judgment. "6 of 14 are the scheme's fault" is a finding; "F1 was 0.62" is not.
    triage_summary = ""
    if triage:
        counts = {}
        for category in TRIAGE_CATEGORIES + ["uncategorised"]:
            counts[category] = 0
        for item_id in triage:
            category = triage_category(triage[item_id])
            if category is None:
                category = "uncategorised"
            counts[category] = counts[category] + 1
        parts = []
        for category in TRIAGE_CATEGORIES + ["uncategorised"]:
            if counts[category] > 0:
                parts.append(str(counts[category]) + " " + category)
        triage_summary = ("- **Your triage of " + str(len(triage)) + " of "
                          + str(len(wrong_records)) + " errors:** "
                          + " / ".join(parts) + "\n\n")

    # Assemble the one-page report, section by section. Anything in _italics_ is a
    # placeholder YOU replace - a section left as the placeholder scores zero.
    report = ""
    report = report + "# One-page report - " + track + "\n\n"
    if group != "":
        report = report + "Group: " + group + "\n\n"
    report = report + "## 1. Scheme & gold\n"
    report = report + "- **Labels:** " + ", ".join(labels) + "\n"
    report = report + ("- **Gold set:** " + str(len(gold))
                       + " items sampled from the pool; per-label counts: "
                       + str(label_counts) + "\n")
    report = report + ("- **QC / adjudication:** _<your percent agreement and kappa, how "
                       "many labels your adjudication changed, which label pair caused "
                       "the most disagreement, and what your scheme now says about it>_\n\n")
    report = report + "## 2. Prompt iterations\n"
    report = report + "| Round | Macro-F1 |\n|---|---|\n" + round_rows + "\n\n"
    report = report + ("_For each round: what did you change, and WHY did you expect it to "
                       "help?_\n\n")
    report = report + "## 3. Evaluation\n"
    report = report + ("- **Final macro-F1:** " + format(final_f1, ".3f") + " on "
                       + str(len(gold)) + " gold items.\n")
    report = report + ("- Per-class precision/recall/F1 and the confusion matrix are in "
                       "the notebook output.\n")
    report = report + "- _Which class did worst, and what did it get confused with?_\n\n"
    report = report + "## 4. Error analysis\n"
    report = report + triage_summary
    report = report + error_examples + "\n\n"
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

    report_path.write_text(report, encoding="utf-8")
    print("Wrote", gold_path.name + ",", csv_path.name, "and", report_path.name,
          "to", str(output_folder) + "/")
    return {"gold": gold_path, "csv": csv_path, "report": report_path}
