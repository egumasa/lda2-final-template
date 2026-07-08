"""pipeline.py — the plumbing for the mini-project.

You do NOT need to edit anything in this file. It holds the "boring but necessary"
helpers that move data around: reading the gold file, sampling a balanced subset,
loading a prompt, asking the LLM, building few-shot examples, plotting, and writing
the report. The interesting evaluation math lives in evaluate.py.

Read it if you are curious — every function is written the long way, on purpose.
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


def _throttle_and_retry(call_model):
    """Wrap a raw 'prompt -> text' function with free-tier friendliness.

    Free-tier Gemini caps requests per minute (roughly 10 for flash). If we fire
    faster than that, we get a 429 RESOURCE_EXHAUSTED / rate-limit error. Guards:
      1) MIN_INTERVAL - wait at least this many seconds between calls, so we stay
         under the per-minute cap in the first place.
      2) on a per-minute rate-limit error, sleep (honoring the server's suggested
         delay when given) and retry a few times.
      3) on a PER-DAY quota error, stop immediately - retrying cannot help today -
         and raise a clear message about how to get unblocked.
    MIN_INTERVAL and MAX_RETRIES are tunable via environment variables.
    """
    min_interval = float(os.environ.get("LLM_MIN_INTERVAL", "4.5"))
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
                        "To keep working now: switch models by setting "
                        "LLM_MODEL=gemini-3.1-flash-lite in your .env (a separate daily "
                        "bucket with a much higher limit, ~500/day), run the notebook in "
                        "Google Colab (free built-in Gemini), or enable billing. "
                        "You can also lower N_PER_CLASS to make each run use fewer calls."
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


def make_backend():
    """Return a function that sends a prompt to an LLM and returns its text reply.

    Whichever backend we pick is wrapped by _throttle_and_retry, so EVERY notebook
    gets the same free-tier pacing and rate-limit backoff for free.
    """
    # Option 1: inside Google Colab, use the free built-in Gemini (no API key).
    try:
        from google.colab import ai

        def call_model(prompt):
            return ai.generate_text(prompt)

        return _throttle_and_retry(call_model), "Colab Gemini"
    except ImportError:
        pass

    # Option 2: outside Colab, use the Gemini API with a key from .env / the environment.
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        from google import genai
        client = genai.Client(api_key=key)
        # gemini-3.1-flash-lite has a far higher free daily quota (500/day vs 20/day
        # for the plain flash models) - so it is the sensible default for a task that
        # sends dozens of requests per run. Override with LLM_MODEL in your .env.
        model = os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")

        def call_model(prompt):
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text

        return _throttle_and_retry(call_model), "Gemini API (" + model + ")"

    # Option 3: nothing available - tell the user what to do.
    raise RuntimeError(
        "No LLM backend found. Run this in Google Colab (free built-in Gemini), "
        "or set GEMINI_API_KEY in a .env file / your environment."
    )


# ----------------------------------------------------------------------------------
# Reading data
# ----------------------------------------------------------------------------------
def load_gold(path_or_url):
    """Read a gold/pool file. Each item looks like {"id": 1, "text": "...", "label": "..."}."""
    # If the location is a web address, download it; otherwise open a local file.
    if str(path_or_url).startswith("http"):
        raw_bytes = urllib.request.urlopen(path_or_url).read()
        raw_text = raw_bytes.decode("utf-8")
        gold = json.loads(raw_text)
    else:
        # Only cefr_pool.json ships with the template; other tracks are student-built.
        # If the file is missing, say WHY and WHAT TO DO instead of a bare traceback.
        if not Path(path_or_url).exists():
            raise FileNotFoundError(
                "Pool file not found: " + str(path_or_url) + "\n"
                "Only the CEFR track ships with data. To run this track you must first "
                "BUILD its pool file:\n"
                "  1. Use the dataset download/preprocess notebooks (see the course repo's "
                "sources/resources/datasets/ and data/gold/README.md).\n"
                "  2. Save the result as data/gold/<track>_pool.json in the canonical "
                '{"id", "text", "label"} shape.\n'
                "  3. Make sure POOL_PATH in the CONFIG cell points at it.\n"
                "To just see the pipeline run end to end, switch TRACK to 'cefr'."
            )
        opened_file = open(path_or_url, encoding="utf-8")
        gold = json.loads(opened_file.read())
        opened_file.close()
    print("Loaded", len(gold), "items. The first one is:", gold[0])
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


def sample_pool(pool, n_per_class, seed):
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


def run_prompt(prompt, gold, labels, generate_text):
    """Ask the model to label every item, and collect the predicted labels."""
    predictions = []
    total = len(gold)
    position = 0
    for item in gold:
        position = position + 1
        # Put this item's sentence into the prompt where {text} is.
        filled_prompt = prompt.format(text=item["text"])
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
def build_fewshot(base_prompt, pool, gold, labels, shots_per_class, seed):
    """Put a few labeled examples (taken from the pool) in front of the prompt.

    We NEVER use an item that is in the gold set as an example, otherwise we
    would be showing the model the very answers we are testing it on.
    """
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
    for label in labels:
        if label in examples_by_label:
            examples = examples_by_label[label]
        else:
            examples = []
        random_generator.shuffle(examples)
        chosen_examples = examples[:shots_per_class]
        for item in chosen_examples:
            lines.append("Sentence: " + item["text"] + "\nLabel: " + label)

    # Step 4: glue the example block in front of the base prompt.
    example_block = "\n\n".join(lines)
    return example_block + "\n\nNow classify this one.\n\n" + base_prompt


# ----------------------------------------------------------------------------------
# Plotting (drawing only — the counts come from evaluate.py)
# ----------------------------------------------------------------------------------
def plot_confusion_matrix(matrix, labels, title):
    """Draw a confusion matrix as a heatmap (rows = gold, columns = predicted)."""
    plt.figure(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------
# Writing the report
# ----------------------------------------------------------------------------------
def export_results(track, gold, predictions, macro_f1_by_round, out_dir):
    """Write a predictions CSV and a one-page report with the five required sections."""
    output_folder = Path(out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    labels = label_set(gold)

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
    csv_path = output_folder / (track + "_predictions.csv")
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

    # Assemble the one-page report, section by section.
    report = ""
    report = report + "# One-page report - " + track + "\n\n"
    report = report + "## 1. Scheme & gold\n"
    report = report + "- **Labels:** " + ", ".join(labels) + "\n"
    report = report + ("- **Gold set:** " + str(len(gold))
                       + " items sampled from the pool; per-label counts: "
                       + str(label_counts) + "\n")
    report = report + ("- **QC / adjudication:** _<what your independent re-check changed; "
                       "disagreements with the published label>_\n\n")
    report = report + "## 2. Prompt iterations\n"
    report = report + "| Round | Macro-F1 |\n|---|---|\n" + round_rows + "\n\n"
    report = report + "## 3. Evaluation\n"
    report = report + ("- **Final macro-F1:** " + format(final_f1, ".3f") + " on "
                       + str(len(gold)) + " held-out gold items.\n")
    report = report + ("- Per-class precision/recall/F1 and the confusion matrix are in "
                       "the notebook output.\n\n")
    report = report + "## 4. Error analysis\n"
    report = report + error_examples + "\n\n"
    report = report + ("_For each miss: is it the **model's** fault or the **scheme's** "
                       "(a genuinely borderline item)?_\n\n")
    report = report + "## 5. Limitations\n"
    report = report + "- LLM output is stochastic; a re-run can shift the numbers.\n"
    report = report + ("- Contamination risk: these are published datasets the model may "
                       "have seen.\n")
    report = report + ("- " + str(len(gold)) + " items is a small sample - treat per-class "
                       "scores for rare labels with caution.\n")

    report_path = output_folder / (track + "_report.md")
    report_path.write_text(report, encoding="utf-8")
    print("Wrote", csv_path.name, "and", report_path.name, "to", str(output_folder) + "/")
    return {"csv": csv_path, "report": report_path}
