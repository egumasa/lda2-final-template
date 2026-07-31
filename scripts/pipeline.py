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
        print("         before your final run. See resources/tools/gemini-api-key.md.")
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
        "needs no key. See resources/tools/gemini-api-key.md."
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
        # Only the small DEMO pools ship with the template; every other file is one
        # you build or make yourself. If it is missing, say WHY and WHAT TO DO instead
        # of a bare traceback.
        if not Path(url_or_path).exists():
            raise FileNotFoundError(
                "File not found: " + str(url_or_path) + "\n"
                "Only data/pools/<track>_demo_pool.json ships with the template. The "
                "rest you build:\n"
                "  * A full-size POOL to sample from - notebook 01_build_pool_<track>, "
                "or the\n"
                "    shortcut `python scripts/prep_datasets.py <track>`. It lands in "
                "data/pools/.\n"
                "  * YOUR SAMPLE - notebook 02_sample writes it to data/gold/.\n"
                "  * YOUR GOLD SET - notebook 03_annotate writes it, after you have "
                "annotated\n"
                "    and adjudicated. Nothing can make this one for you.\n"
                "Check the paths in config.py. To just see the pipeline run, load "
                "DEMO_POOL_PATH instead of POOL_PATH."
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
# Freezing predictions
# ----------------------------------------------------------------------------------
# A hosted LLM is only best-effort reproducible, even at temperature=0. So once your
# prompt is final you run the model ONCE, save its predictions to a file, and do all
# your evaluation off that file. Your reported numbers then hold still, and anyone
# (including whoever grades it) can re-run your analysis on exactly the outputs you
# saw. In Day 2 you loaded a frozen file we made for you; now you make your own.
def save_predictions(predictions, path):
    """Write a list of predicted labels to a JSON file - this is 'freezing' a run."""
    output_path = Path(path)
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
def save_json(data, path, what="items"):
    """Write anything JSON-able to a file, making the folder if it is missing."""
    output_path = Path(path)
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

    # Step 4: report how many of each label we ended up with.
    counts = {}
    for item in sampled:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    print("Sampled", len(sampled), "items. Per-label counts:", counts)

    # Step 5: warn if we took most of the pool. A sample is only meaningful if there
    # is a pool left over - both because a near-total sample is not a sample, and
    # because build_fewshot needs unused items to draw its examples from.
    if len(pool) > 0 and len(sampled) > 0.5 * len(pool):
        print("WARNING: you just took", len(sampled), "of", len(pool), "pool items.")
        print("         That is most of the pool, which leaves few or no spare items")
        print("         for few-shot examples. Are you pointing at a small DEMO file")
        print("         instead of a full pool? Build the real one with:")
        print("             python scripts/prep_datasets.py <track>")
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
def export_results(track, gold, predictions, macro_f1_by_round, out_dir, group=""):
    """Write your gold set, a predictions CSV, and a one-page report scaffold.

    `group` is added to every filename, so several groups can drop their results in
    one folder without overwriting each other. These files are what you submit.
    """
    output_folder = Path(out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    labels = label_set(gold)

    # Every file we write starts with the same stem, e.g. "raamove_groupA".
    if group == "":
        stem = track
    else:
        stem = track + "_" + group

    # Save the gold set alongside the results: the numbers below mean nothing
    # without the exact items they were computed on.
    gold_path = output_folder / (stem + "_gold.json")
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
    csv_path = output_folder / (stem + "_predictions.csv")
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

    # Collect up to five wrong items as concrete error examples.
    error_lines = []
    for record in records:
        if len(error_lines) >= 5:
            break
        if not record["correct"]:
            snippet = str(record["text"])[:120]
            line = ("- **id " + str(record["id"]) + "** gold `" + str(record["gold"])
                    + "` -> pred `" + str(record["pred"]) + "`: " + snippet)
            error_lines.append(line)
    if len(error_lines) == 0:
        error_examples = "- (no errors to show)"
    else:
        error_examples = "\n".join(error_lines)

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
    report = report + error_examples + "\n\n"
    report = report + ("_For each miss: is it the **model's** fault or the **scheme's** "
                       "(a genuinely borderline item)? Give a reason, not a verdict._\n\n")
    report = report + "## 5. Limitations\n"
    report = report + ("_Replace these three generic lines with at least two limitations "
                       "that apply to YOUR run._\n")
    report = report + "- LLM output is stochastic; a re-run can shift the numbers.\n"
    report = report + ("- Contamination risk: these are published datasets the model may "
                       "have seen.\n")
    report = report + ("- " + str(len(gold)) + " items is a small sample - treat per-class "
                       "scores for rare labels with caution.\n")

    report_path = output_folder / (stem + "_report.md")
    report_path.write_text(report, encoding="utf-8")
    print("Wrote", gold_path.name + ",", csv_path.name, "and", report_path.name,
          "to", str(output_folder) + "/")
    return {"gold": gold_path, "csv": csv_path, "report": report_path}
