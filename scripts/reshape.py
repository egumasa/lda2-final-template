"""reshape.py — turning each raw corpus into the canonical schema.

    [{"id": 1, "text": "...", "label": "..."}, ...]

This is the ONE place the reshaping decisions live. `prep_datasets.py` calls these
functions, and `_generate_pool_notebooks.py` embeds their source into the
`notebooks/01_build_pool_<track>.ipynb` walkthroughs — so the command-line shortcut and the
step-by-step notebooks can never drift apart, because they are the same code.

Reshaping is not a mechanical format conversion. Each function below makes real
gold-standard decisions - which annotations to trust, how fine-grained the labels
should be, what to do with items that do not fit cleanly. Those decisions are
commented where they happen, because they are the interesting part.

You do NOT need to edit this file (but it is worth reading before you defend your
gold set in the Q&A).
"""

import csv
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

SEED = 42            # fixed, so every rebuild gives the same sample


# ----------------------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------------------
def reid(items):
    """Renumber ids sequentially from 1, keeping the current order."""
    renumbered = []
    next_id = 1

    for item in items:
        ### Copy before writing ###
        new_item = dict(item)                    # Work on a copy, so the caller's item is left alone.

        ### Stamp the id ###
        new_item["id"] = next_id                 # Overwrite whatever id was there with the running number.
        renumbered.append(new_item)              # Keep it in the order it arrived.
        next_id = next_id + 1                    # Advance, so the next item gets a fresh id.

    return renumbered


def balanced_sample(items, per_label, seed=SEED):
    """Up to `per_label` items for each label, drawn with a fixed seed."""
    random_generator = random.Random(seed)
    by_label = {}
    for item in items:
        label = item["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item)

    out = []
    for label in sorted(by_label):
        bucket = by_label[label]
        random_generator.shuffle(bucket)
        for item in bucket[:per_label]:
            out.append(item)
    random_generator.shuffle(out)
    return out


def validate(items, allowed=None):
    """Check the canonical schema, and raise on the first problem found.

    Deliberately explicit rather than `assert`: assertions vanish under `python -O`,
    and a silently unvalidated dataset is exactly the kind of thing that surfaces as a
    baffling metric three days later.
    """
    seen_ids = set()
    for position, item in enumerate(items):
        missing = {"id", "text", "label"} - set(item)
        if missing:
            raise ValueError("item " + str(position) + " is missing "
                             + str(sorted(missing)) + ": " + repr(item))
        if item["id"] in seen_ids:
            raise ValueError("duplicate id " + str(item["id"]))
        seen_ids.add(item["id"])
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("empty text at id " + str(item["id"]))
        if not isinstance(item["label"], str) or not item["label"]:
            raise ValueError("empty label at id " + str(item["id"]))
        if allowed is not None and item["label"] not in allowed:
            raise ValueError("label " + repr(item["label"]) + " at id "
                             + str(item["id"]) + " is not in " + str(sorted(allowed)))


def label_counts(items):
    """How many items carry each label - the first thing to look at after a build."""
    counts = {}
    for item in items:
        label = item["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


# ----------------------------------------------------------------------------------
# CEFR-SP (Wiki-Auto portion) -> CEFR level A1-C2
# ----------------------------------------------------------------------------------
CEFR_NUM = {"1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1", "6": "C2"}
CEFR_LABELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def reshape_cefr(wiki_auto_dir):
    """Read the Wiki-Auto TSV files and keep only the sentences both annotators agreed on.

    Each line is:  sentence <TAB> label_by_annotator_A <TAB> label_by_annotator_B
    with labels as digits, 1=A1 ... 6=C2.

    Three real decisions here:
      1. TRUST ONLY AGREEMENT. A sentence is kept only when both annotators chose the
         same level, so every label is unambiguous. That is what makes this the gentle
         on-ramp track - and it also means the track is easier than the data really is.
      2. HUMAN-READABLE LABELS. 1 -> A1, so a prompt can name the levels the way a
         person would.
      3. WIKI-AUTO ONLY. CEFR-SP also ships a SCoRE portion, but it is CC BY-NC-SA
         (non-commercial), so we deliberately do not touch it - see data/SOURCES.md.
    """
    source_dir = Path(wiki_auto_dir)
    rows = []

    ### Read every TSV in the folder ###
    # Sorted, so a rebuild reads the files in the same order and ids stay stable.
    for path in sorted(source_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():   # One sentence per line.

            ### Split the line on its tabs ###
            parts = line.split("\t")             # -> [sentence, label_A, label_B]
            if len(parts) < 3:                   # A short line is malformed; skip it rather than crash.
                continue

            ### Pull out the three fields ###
            text = parts[0].strip()              # The sentence itself.
            label_a = parts[1].strip()           # Annotator A's level, as a digit "1".."6".
            label_b = parts[2].strip()           # Annotator B's level, same encoding.

            ### Keep only what both annotators agreed on ###
            if text and label_a == label_b and label_a in CEFR_NUM:   # Disagreements are dropped entirely.
                rows.append({"id": 0, "text": text,
                             "label": CEFR_NUM[label_a]})             # Digit -> human-readable level ("3" -> "B1").

    return reid(rows)                            # Hand back with ids running 1..N.


# ----------------------------------------------------------------------------------
# RAAMove -> rhetorical move in an RA abstract (8 classes)
# ----------------------------------------------------------------------------------
RAAMOVE_LABELS = {
    "BAC": "Background", "GAP": "Gap", "MTD": "Method", "PUR": "Purpose",
    "RST": "Result", "CLN": "Conclusion", "CTN": "Contribution", "IMP": "Implication",
}


def reshape_raamove(raamove_dir):
    """Read RAAMove's per-domain JSON files and expand the 3-letter move codes.

    The corpus ships two domains (Intelligence, Engineering) as separate files. We pool
    them, because a move is meant to be a rhetorical function rather than a
    discipline-specific one - but that IS an assumption, and comparing the two domains
    separately would be a perfectly good extension.

    A move is a rhetorical function WITHIN an abstract, so each item also carries the
    abstract it came from - see the note on the two-pass loop below.
    """
    source_dir = Path(raamove_dir)
    rows = []

    ### Read both discipline files into one pool ###
    for filename in ("Intelligence.json", "Engineering.json"):
        path = source_dir / filename
        if not path.exists():                    # A missing file is not fatal; use whichever shipped.
            continue

        ### Parse the JSON ###
        data = json.loads(path.read_text(encoding="utf-8"))   # -> a list of {"idx": ..., "text": ..., "labels": ...} records.

        ### PASS 1: group the flat record list back into abstracts ###
        # The file is one long list of sentences, but `idx` is the abstract number, and
        # the sentences of one abstract sit together in reading order. So a new idx means
        # a new abstract - which is all the grouping we need, and it does not care that
        # idx starts again at 0 in the other discipline file.
        abstracts = []
        for record in data:
            if not abstracts or abstracts[-1][0] != record["idx"]:
                abstracts.append((record["idx"], []))          # Start collecting a new abstract.
            abstracts[-1][1].append(record)                    # Same idx: same abstract as the line before.

        ### PASS 2: emit one item per sentence, with its abstract attached ###
        for number, records in abstracts:
            texts = [record["text"].strip() for record in records]   # The abstract, sentence by sentence.
            context = "\n".join(texts)           # One string, newlines kept so the sentences stay visible.
            doc_id = path.stem + "-" + str(number)   # e.g. "Intelligence-0". idx alone is NOT unique across the two files.

            for position, record in enumerate(records):
                code = record["labels"]          # e.g. "BAC".
                if code in RAAMOVE_LABELS:
                    label = RAAMOVE_LABELS[code]     # The name your prompt and annotation sheet will use.
                else:
                    label = code        # an unexpected code: keep it and let validate() complain
                rows.append({"id": 0, "text": texts[position], "label": label,
                             "doc_id": doc_id,           # Which abstract this sentence is from.
                             "sent_index": position,     # Where in it - 0 is the first sentence.
                             "n_sents": len(texts),      # How long the abstract is.
                             "context": context})        # The abstract itself.

    return reid(rows)                            # Hand back with ids running 1..N.


# ----------------------------------------------------------------------------------
# CaRS-50 -> Swales CARS Move (3 classes) or Move+Step (11 classes)
# ----------------------------------------------------------------------------------
def reshape_cars50(cars50_dir):
    """Parse the 50 XML introductions into TWO datasets: moves, and move+step.

    XML shape:
        <sentence><sentenceID/><text/><step>1b</step></sentence>

    The `step` code is like "1b": the leading DIGIT is the Move, the whole code is the
    Step. So one parse gives two granularities, and which you use is a scheme decision:
    3 classes is a fair task, 11 classes is the stretch version. Returns
    (move_rows, step_rows).

    A move is a rhetorical function WITHIN an introduction, so each item also carries the
    introduction it came from - see the note on the two-pass loop below.
    """
    source_dir = Path(cars50_dir)
    move_rows = []
    step_rows = []

    ### Walk the 50 XML files ###
    for xml_path in sorted(source_dir.glob("*.xml")):   # One file per article introduction.

        ### Parse one file into a tree ###
        tree = ET.parse(xml_path)                # ElementTree turns the XML into a navigable tree.
        doc_id = xml_path.stem                   # e.g. "text001". The FILENAME - the <sentenceID> tags are not reliable.

        ### PASS 1: read the whole introduction, in order ###
        # Every sentence that has text goes in here, including ones we are about to drop
        # for having no usable code. They belong in the passage because a reader saw them:
        # leaving them out would hand the model a doctored introduction.
        passage = []
        for sentence in tree.iter("sentence"):   # .iter() finds them at any depth, so the paragraph nesting does not matter.
            text_element = sentence.find("text")     # The <text> child, or None if absent.
            step_element = sentence.find("step")     # The <step> child, or None if absent.
            text = (text_element.text or "").strip() if text_element is not None else ""   # `or ""` guards an empty tag, whose .text is None.
            if text:
                passage.append((text, step_element))

        texts = [text for text, _ in passage]    # Just the sentences.
        context = "\n".join(texts)               # One string, newlines kept so the sentences stay visible.

        ### PASS 2: emit an item for each sentence that carries a usable code ###
        # Position comes from enumerate(), never from <sentenceID>: those ids are padded
        # three different ways, mix two widths inside text038.xml, and t025s020 appears
        # twice in text025.xml.
        for position, (text, step_element) in enumerate(passage):
            code = (step_element.text or "").strip() if step_element is not None else ""   # e.g. "1b".

            # Skip anything unlabelled, or whose code does not start with a move digit.
            if not code or not code[0].isdigit():
                continue

            ### Where this sentence sits - the same for both granularities ###
            where = {"doc_id": doc_id,           # Which introduction this sentence is from.
                     "sent_index": position,     # Where in it - 0 is the first sentence.
                     "n_sents": len(texts),      # How long the introduction is.
                     "context": context}         # The introduction itself.

            ### Record the SAME sentence at both granularities ###
            move_rows.append({"id": 0, "text": text,
                              "label": "Move " + code[0], **where})   # Leading digit only -> 3 classes.
            step_rows.append({"id": 0, "text": text,
                              "label": code, **where})                # Whole code -> 11 classes.

    return reid(move_rows), reid(step_rows)      # Two datasets, each with ids running 1..N.


# ----------------------------------------------------------------------------------
# AutoErrorAnalyzer -> L2 error category (4 classes) or error detection (2 classes)
# ----------------------------------------------------------------------------------
# The published 23-code taxonomy, collapsed into three broader categories.
L2_COARSE = {}
for _code in "ART PREP NUM TENSE VFORM WO AGR DET POSS MOD CONJ STRUCT".split():
    L2_COARSE[_code] = "Grammatical"
for _code in "N ADJ ADV V REF EXPR".split():
    L2_COARSE[_code] = "Lexical"
for _code in "SP MIS UNN CWS PUNC".split():
    L2_COARSE[_code] = "Mechanical"

L2_LABELS = {"Grammatical", "Lexical", "Mechanical", "No error"}
L2_DETECTION_LABELS = {"Has error", "No error"}


def _l2_coarse_label(human_field):
    """Collapse a sentence's comma-separated error codes to ONE broader category.

    Returns None when the sentence cannot get a single clean label - either it has no
    codes, or its codes span more than one broader category. Those get dropped, which
    keeps this a single-label task. It also means the dataset under-represents exactly
    the messiest sentences, and that is worth a line in your limitations section.
    """
    ### Split the comma-separated code list ###
    codes = []
    for code in human_field.split(","):          # "ART,SP" -> ["ART", "SP"].
        code = code.strip()
        if code:                                 # Drop empties from a trailing comma.
            codes.append(code)

    ### Two easy cases first ###
    if not codes:                                # Nothing to go on -> no label.
        return None
    if codes[0] == "NO_ERROR":                   # The explicit "this sentence is clean" marker.
        return "No error"

    ### Map every code to its broader category ###
    categories = set()
    for code in codes:
        if code in L2_COARSE:                    # Codes you did not list are silently ignored here...
            categories.add(L2_COARSE[code])

    ### One category, or nothing ###
    if len(categories) == 1:                     # All the errors agree -> that is the label.
        return categories.pop()
    return None                       # no codes we recognise, or a mixed-category sentence


def reshape_l2_errors(csv_path):
    """Read data_category.csv into TWO datasets: 4-way categories, and yes/no detection.

    The CSV also carries `AEA_ErrorCategories` - the published tool's own predictions -
    so this is the one track where you can benchmark your LLM against both a human gold
    standard AND an existing system. Returns (category_rows, detection_rows).
    """
    category_rows = []
    detection_rows = []
    # utf-8-sig: the file ships with a byte-order mark, which would otherwise end up
    # glued to the first column name and break the lookup.
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:

        ### Read the CSV a row at a time ###
        for record in csv.DictReader(handle):    # DictReader gives each row as {column: value}.

            ### Pull the two columns that matter ###
            sentence = (record.get("Sentence") or "").strip()                  # The text.
            human = (record.get("Human_ErrorCategories") or "").strip()        # The human codes, comma-separated.
            if not sentence or not human:        # No text or no annotation -> unusable either way.
                continue

            ### Dataset 1: the coarse category ###
            label = _l2_coarse_label(human)      # Returns None for mixed-category sentences...
            if label is not None:                # ...and those get no row here at all.
                category_rows.append({"id": 0, "text": sentence, "label": label})

            ### Dataset 2: did it have ANY error? ###
            if human == "NO_ERROR":              # This one does not depend on your grouping,
                detection_label = "No error"     # so every annotated sentence gets a row.
            else:
                detection_label = "Has error"
            detection_rows.append({"id": 0, "text": sentence, "label": detection_label})

    return reid(category_rows), reid(detection_rows)   # Two datasets, each with ids running 1..N.


# ----------------------------------------------------------------------------------
# ICNALE GRA -> holistic score band
# ----------------------------------------------------------------------------------
ICNALE_LABELS = {"Low", "Mid", "High"}


def reshape_icnale(csv_path, low_below=4.0, mid_below=7.0):
    """Band a numeric holistic score into Low / Mid / High.

    THE CUT-OFFS ARE PLACEHOLDERS. 4 and 7 are not from the ICNALE rubric - they are
    round numbers. Where you put the boundaries decides how hard the task is and how
    balanced the classes are, so set them from the rubric you are actually using and
    say what you chose in your report.
    """
    rows = []
    skipped = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:

        ### Read the CSV a row at a time ###
        for record in csv.DictReader(handle):    # DictReader gives each row as {column: value}.

            ### Pull the essay and its score ###
            text = (record.get("text") or "").strip()
            raw_score = (record.get("score") or "").strip()   # Still a string at this point.
            if not text or not raw_score:        # Nothing to band -> skip.
                continue

            ### Turn the score into a number ###
            try:
                score = float(raw_score)
            except ValueError:
                skipped = skipped + 1      # a non-numeric cell: report it, do not crash
                continue

            ### Apply the two cut-offs ###
            if score < low_below:                # Everything below the first boundary.
                label = "Low"
            elif score < mid_below:              # Between the two boundaries.
                label = "Mid"
            else:                                # At or above the second boundary.
                label = "High"
            rows.append({"id": 0, "text": text, "label": label})
    if skipped:
        print("  note: skipped", skipped, "row(s) whose score cell was not a number.")
    return reid(rows)
