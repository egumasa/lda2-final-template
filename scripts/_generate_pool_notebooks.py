#!/usr/bin/env python3
"""Generate notebooks/01_build_pool_<track>.ipynb — one per track.

    python scripts/_generate_pool_notebooks.py

Notebook 01 is where a track's pool comes from: download -> look at the raw format ->
reshape to {id, text, label} -> check the label balance -> save
data/pools/<track>_pool.json. Notebook 02 picks it up from there.

The reshaping code is stdlib only and travels inside the notebook, but the notebook is
not standalone: like 02-05 it opens with the SETUP cell, mounts your group's Drive
folder and imports config.py. That is on purpose. POOL_PATH is where notebook 02 comes
looking, and a pool written anywhere else - a Colab runtime that is about to reset, a
loose file in someone's Drive root - is a pool nobody reads.

WHAT IS BLANK, AND WHY
----------------------
The download and the parsing are written out: nobody learns anything from retyping a
`User-Agent` header or an XML glob, and every track's raw format is different anyway.

What is left blank is the handful of places where a real gold-standard DECISION gets
made — which annotations to trust, what to call each label, how fine-grained the scheme
is, where a numeric scale gets cut. Those are the choices a group has to be able to
defend in the Q&A, so they type them, and PLAN.md asks them to write down why.

The mechanism is deliberately simple: the reshaping FUNCTIONS come out of reshape.py by
`inspect.getsource`, but the CONSTANTS they read are left undefined, and a blank cell
asks for them. Python resolves them from the notebook's globals at call time, so a
filled-in notebook runs the same code prep_datasets.py runs.

WHY THIS IS GENERATED
---------------------
Each notebook needs the reshaping code *inside it*, because it has to run without the
rest of the repo. Pasting five copies would mean keeping five copies in sync by hand.
Instead we read the real functions out of reshape.py at generation time. There is one
implementation; these notebooks are renderings of it.

Never hand-edit a 01_*.ipynb - edit reshape.py or this file and re-run.

To check nothing has drifted:

    python scripts/_generate_pool_notebooks.py && git diff --exit-code notebooks/
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reshape
from _setup_cell import SCRATCH, SETUP_MD_LINES, setup_lines

OUT = Path(__file__).resolve().parent.parent / "notebooks"


# ----------------------------------------------------------------------------------
# Notebook building blocks
# ----------------------------------------------------------------------------------
def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def save(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    blanks = sum(1 for c in cells if "✏️" in "".join(c["source"])
                 and c["cell_type"] == "code")
    print("wrote", name, "(" + str(len(cells)), "cells,", blanks, "blank)")


def lead(*lines):
    """A markdown signpost immediately above a code cell: what we are about to do.

    Every code cell in these notebooks has one. A cell a student meets with no idea
    what it is for is a cell they run and scroll past, and running a cell you cannot
    describe is not the skill this course is trying to build.
    """
    return md(*lines)


def reading_note(what):
    """The markdown that introduces a run of embedded-source cells."""
    return md(
        "### The code that does it — read it, then run it",
        "",
        what,
        "",
        "It is read straight out of `scripts/reshape.py` when this notebook is "
        "generated, so it is not a simplified copy: it is the code that runs.",
        "",
        "It arrives one function per cell, so you can take them one at a time. **None "
        "of these cells print anything.** They only give the functions their names — "
        "that is what `def` does. You will see no output until the cell *after* them, "
        "which calls one.")


def source_cells(described, imports=()):
    """One cell per embedded function, each with its own one-line signpost.

    `described` is a list of (object, sentence) pairs. Splitting them up matters: the
    single cell these replaced ran to over a hundred lines and defined four unrelated
    things, which is a wall rather than something anyone reads.

    Functions come through `inspect.getsource`, so what the notebook shows is literally
    what prep_datasets.py runs. `imports` are the module-level names that source needs -
    reshape.py has them at the top of the file, which inspect.getsource does not carry
    across, so without this the first cell dies on `NameError: Path`.
    """
    cells = []
    if imports:
        cells.append(lead("First, the two library modules the code below needs. "
                          "`import` is how Python is told to load one."))
        cells.append(code(*imports))
    for obj, sentence in described:
        cells.append(lead(sentence))
        cells.append(code(*inspect.getsource(obj).rstrip("\n").split("\n")))
    return cells


def blank(signpost, title, does, produces, hints=(), notes=(), starter=None):
    """A decision cell, with a markdown signpost above it. Returns BOTH cells.

    The header says what the cell does and what it names for later cells. Everything
    else - why the decision matters, what turns on it, what to watch out for - goes in
    the signpost above, where it can be written as sentences.

    Never leave a cell without saying what the next one expects to find - a student
    stuck on a NAME has learned nothing about annotation.

    `hints` is not a function list. It shows the SHAPE of the thing being edited (a
    dict of code-to-category, a pair of band boundaries), which the code alone does not
    make obvious. Drop it wherever the starter already shows the shape.

    `starter` ships COMPLETE and runnable - every argument filled in with a real,
    defensible value. There are no blanks to fill.

    That is deliberate. A blank standing in for an argument tests whether you can copy
    a word out of the comment beside it, and a student who does that has made no
    decision at all. The decision these cells are actually for is a different one:
    whether the value that is already there is the RIGHT one for the study you are
    describing in PLAN.md - whether those band boundaries, those category names, that
    sampling strategy are what you would defend in the Q&A. So the cell runs on first
    execution, prints what it did, and the work is reading the result and arguing about
    it. Changing nothing is a choice too, and one you still have to justify.
    """
    rule = "─" * max(4, 60 - len(title))
    lines = ["# ✏️ " + title + " " + rule]
    lines.extend("# " + line for line in does)
    for index, hint in enumerate(hints):
        lines.append(("# Looks like: " if index == 0 else "#             ") + hint)
    if produces:
        lines.append("# Creates: " + produces)
    for note in notes:
        lines.append("# " + note)
    lines.append("")
    lines.append("# ✏️ this runs as written — the work is deciding whether it should")
    lines.append("")
    for line in starter:
        lines.append(line)
    lines.append("")
    return [lead(signpost), code(*lines)]


SCHEMA_NOTE = (
    "Every dataset in this course is reshaped into the **same canonical schema**, so one "
    "pipeline works for all of them:\n\n"
    "```json\n"
    "[{\"id\": 1, \"text\": \"...\", \"label\": \"...\"}]\n"
    "```\n\n"
    "The *raw* data, though, looks different every time. **That difference is the "
    "lesson** — half of building a gold standard is getting messy real data into a "
    "clean, consistent shape.\n\n"
    "Those three keys are required on every track. Two tracks add more: `cars50` and "
    "`raamove` ask what a sentence *does in a passage*, which is not always decidable "
    "from the sentence on its own, so their items also carry `doc_id`, `sent_index`, "
    "`n_sents` and `context`. Extra keys are safe everywhere — nothing in the pipeline "
    "checks for keys it does not need."
)

GENERATED_NOTE = (
    "> The reshaping code below is read straight out of `scripts/reshape.py` — it is the "
    "same code `scripts/prep_datasets.py` runs, not a copy of it. What is *missing* from "
    "it is missing on purpose: the ✏️ cells are the decisions, and they are yours. "
    "(Generated by `scripts/_generate_pool_notebooks.py`; edit that or `reshape.py`, "
    "never the `.ipynb`.)"
)

WHERE_THIS_FITS = (
    "### Where this sits\n\n"
    "```\n"
    "▶ 01 build the pool  →  02 sample  →  03 annotate  →  04 prompt  →  05 report\n"
    "```\n\n"
    "You run **01 once per group**, for your own track only. It ends by writing "
    "`data/pools/<track>_pool.json` — the file notebook 02 opens."
)


def header(title, subtitle, what, licence_line, citation, difficulty):
    return md(
        "# 01 · " + title,
        "",
        "*" + subtitle + "*",
        "",
        WHERE_THIS_FITS,
        "",
        "---",
        "",
        "**What it is.** " + what,
        "",
        "**Difficulty of the labeling judgment:** " + difficulty,
        "",
        "**Licence:** " + licence_line + "  ",
        "**Cite:** " + citation,
        "",
        "---",
        "",
        SCHEMA_NOTE,
        "",
        GENERATED_NOTE,
    )


def inspect_cells():
    """Two cells: how big the pool is and how it is balanced, then what an item looks
    like. Two questions, so two cells."""
    return [
        lead("Now we count what we have got: how many items, how many of each label, "
             "and which fields every item carries."),
        code(
            "# Count how many items carry each label, one item at a time.",
            "label_counts = {}",
            'for item in rows:',
            '    label = item["label"]',
            "    if label not in label_counts:",
            "        label_counts[label] = 0",
            "    label_counts[label] = label_counts[label] + 1",
            "",
            'print("total items:", len(rows))',
            'print("label counts:", label_counts)',
            'print("fields per item:", list(rows[0].keys()))'),
        lead("Now we look at three whole items, to see the shape of one.",
             "",
             "Where a track carries a `context` (the whole passage a sentence came "
             "from), it is shortened here so it does not bury everything else. That "
             "only changes what is **printed** — `rows` itself is untouched."),
        code(
            "for item in rows[:3]:",
            "    preview = dict(item)          # a copy, so trimming it changes nothing",
            '    if preview.get("context"):',
            '        preview["context"] = preview["context"][:70] + " …"',
            "    print(preview)",
            '    print("---")'),
    ]


def parser_note(heading, intro, structure, api_steps, demos=()):
    """Teach the library this track's raw format needs, before the reshaping code.

    Every track hands you a different format, and the reshaping function further down
    reads it with whatever stdlib module fits. That module is the one genuinely new
    thing on the page, so it gets named and mapped BEFORE the code that uses it -
    otherwise the function below reads as magic.

    `structure` is a bulleted map of the format; `api_steps` is a numbered list of the
    calls the function makes. `demos` is a list of (signpost, source lines) pairs, each
    becoming a lead-in and a runnable cell - give them only when the track's Step 2 cell
    does not already exercise the library.
    """
    lines = ["### " + heading, "", intro, ""]
    for item in structure:
        lines.append(item)
    lines.append("")
    lines.append("The reshaping function below uses exactly these:")
    lines.append("")
    for index, item in enumerate(api_steps):
        lines.append(str(index + 1) + ". " + item)
    cells = [md(*lines)]
    for signpost, source in demos:
        cells.append(lead(signpost))
        cells.append(code(*source))
    return cells


def setup(track):
    """The SETUP cell, plus the markdown that explains the shared folder.

    01 imports `config.py` for the same reason 02-05 do: POOL_PATH is where notebook
    02 will come looking. A pool written anywhere else is a pool nobody reads.
    """
    return [md(*SETUP_MD_LINES), code(*setup_lines(workdir=SCRATCH))]


def save_cell(track, note="", variants=()):
    """Write the pool to POOL_PATH — the exact path notebook 02 opens.

    Note the track guard. POOL_PATH is built from `track` in config.yaml, so running
    this notebook with config.yaml still set to another track would write, say, a
    RAAMove pool into cars50_pool.json. Everything downstream would then run perfectly
    on the wrong data, and the first sign of trouble would be labels that make no sense
    in notebook 03 - by which point two people have annotated forty items.

    `variants` are the other track values this notebook may legitimately be run under -
    the second granularity a couple of tracks offer ("cars50_step"). Setting `track` to
    the variant is what puts the pool in its own file, so the guard has to allow it.
    """
    allowed = [track] + list(variants)
    cells = [
        lead("**First, a safety check.** `POOL_PATH` is built from the `track:` line in "
             "`config.yaml`. If that still says another track, saving now would write "
             + track + " data into a file belonging to something else — and everything "
             "downstream would run perfectly on the wrong data. The first sign of "
             "trouble would be labels that make no sense in notebook 03, by which point "
             "two people have annotated forty items.",
             "",
             "If this cell stops you: open `config.yaml`, set `track:` to `" + track
             + "`, save it, then re-run the SETUP cell at the top of this notebook."),
        code(
            "if TRACK not in " + repr(allowed) + ":",
            '    raise RuntimeError(',
            '        "config.yaml says  track: " + str(TRACK) + "  but this is the '
            + track + ' "',
            '        "notebook, so saving now would put ' + track + ' data into "',
            '        + POOL_PATH.name + ", which belongs to another track.\\n"',
            '        "Open config.yaml, set  track: to one of ' + " · ".join(allowed)
            + ',"',
            '        " save it, then re-run the SETUP cell at the top of this notebook.")',
            "",
            'print("config.yaml agrees: this is the", TRACK, "track.")'),
        lead("**Now we check the shape of every item.** Everything downstream — the "
             "sampling, the annotation sheet, the scoring — assumes each item has an "
             "`id`, a `text` and a `label`. A pool that breaks that assumption does not "
             "fail here; it fails in notebook 03, after two people have annotated forty "
             "items.",
             "",
             "`validate` says nothing when all is well. Silence is the pass."),
        code("validate(rows)",
             'print("All", len(rows), "items have an id, a text and a label.")'),
        lead("**Now we write the pool** into your group's Drive folder, under the exact "
             "name notebook 02 will look for. Both notebooks get that name from "
             "`config.yaml`, so there is nothing to copy or paste between them."),
    ]
    write_lines = [
        "import json",
        "",
        "POOL_PATH.parent.mkdir(parents=True, exist_ok=True)",
        'with open(POOL_PATH, "w", encoding="utf-8") as f:',
        "    json.dump(rows, f, ensure_ascii=False, indent=2)",
        'print("Saved", len(rows), "items to", POOL_PATH)',
    ]
    if note:
        write_lines = write_lines + ["", "# " + note]
    cells.append(code(*write_lines))
    return cells


def handoff(track):
    return md(
        "## What you just built, and what happens to it",
        "",
        "This is the **pool** — everything usable in the corpus, with its natural label "
        "imbalance intact. It is **not** your gold set, and its labels are **not** your "
        "labels: they are the original corpus authors' judgment, and you have not yet "
        "agreed with them about anything.",
        "",
        "What those labels are for is narrow, and worth being precise about:",
        "",
        "1. **Stratifying the draw** in notebook 02 — you cannot sample evenly across "
        "classes without knowing what the classes are.",
        "2. **A comparison** in notebook 03 — once you have annotated blind and "
        "adjudicated, `compare_to_published` shows you every item where your group "
        "landed somewhere different. That gap is evidence, and one of the more "
        "interesting things you can put in a report.",
        "",
        "They are never the answer key you score the model against. That file does not "
        "exist yet — you make it in notebook 03.",
        "",
        "---",
        "",
        "**Next:** open `02_sample.ipynb`. It reads `POOL_PATH` — the file the cell "
        "above just wrote, in your group's Drive folder. Nothing to copy, nothing to "
        "paste: that path is the handoff, and both notebooks get it from the same "
        "`config.yaml`.")


# ===================================================================== RAAMove
save("01_build_pool_raamove.ipynb", [
    header(
        "RAAMove — build the pool",
        "Rhetorical moves in research-article abstracts (8 classes)",
        "400 RA abstracts, annotated sentence by sentence with one of eight rhetorical "
        "moves (Background, Gap, Purpose, Method, Result, Conclusion, Contribution, "
        "Implication). Reported annotator agreement: κ = 0.785.",
        "CC BY 4.0",
        "Liu, J. et al. (2024), *LREC-COLING*. github.com/ljk1228/RAAMove",
        "★★☆ — moderate. Moves are functional categories, so neighbouring sentences "
        "can be genuinely hard to separate.",
    ),
    *setup("raamove"),
    md("## Step 1 — Download the raw data",
       "",
       "Now we fetch the corpus itself. It lives in a public GitHub repository, and "
       "`git clone` copies the whole thing down into the folder this notebook is "
       "working in.",
       "",
       "The `!` at the front is not Python. In Colab it means *run this line as a "
       "terminal command*, and you will see it whenever a cell reaches outside Python "
       "to fetch or install something."),
    code("!git clone --depth 1 https://github.com/ljk1228/RAAMove"),
    md("## Step 2 — Look at the raw format",
       "",
       "This one is **JSON**, split into two files by discipline "
       "(`Intelligence.json`, `Engineering.json`). Each record has a `text`, a "
       "three-letter move code in `labels`, and an `idx` — the number of the abstract "
       "the sentence came from.",
       "",
       "**Read the codes the cell below prints** — you are about to name every one. "
       "Then look at the `idx` values: the file is one long list of sentences, but the "
       "sentences of an abstract sit together, in order. That is the only reason the "
       "abstracts can be put back together at all."),
    code('RAW_DIR = "RAAMove"          # the folder git clone just made',
         "",
         "import json",
         "from pathlib import Path",
         "",
         "# Read the file as one long string, then turn that string into Python lists",
         "# and dicts. Two steps, so you can see where the file stops and the data",
         "# starts.",
         'raw_text = Path(RAW_DIR + "/Intelligence.json").read_text(encoding="utf-8")',
         "data = json.loads(raw_text)",
         'print("records:", len(data))'),
    lead("Now we look at what is in those records: which move codes appear, and how "
         "many separate abstracts the sentences came from."),
    code("# Count the move codes, and collect the abstract numbers, one record at a time.",
         "code_counts = {}",
         "abstract_numbers = []",
         "for record in data:",
         '    move_code = record["labels"]',
         "    if move_code not in code_counts:",
         "        code_counts[move_code] = 0",
         "    code_counts[move_code] = code_counts[move_code] + 1",
         '    if record["idx"] not in abstract_numbers:',
         '        abstract_numbers.append(record["idx"])',
         "",
         'print("codes:", code_counts)',
         'print("abstracts:", len(abstract_numbers))'),
    lead("Now we print three whole records, so you can see the shape of one."),
    code("for record in data[:3]:",
         "    print(record)"),
    *parser_note(
        "Reading JSON with `json.loads`",
        "`json.loads(text)` turns a JSON string into ordinary Python objects — a JSON "
        "array becomes a `list`, a JSON object becomes a `dict`. Nothing else is "
        "needed: once it is loaded you index it exactly like any other list of dicts, "
        "which is what the cell above did.",
        ["**One record looks like this:**",
         "",
         "```json",
         "{\"idx\": 0, \"text\": \"Recent work has shown ...\", \"labels\": \"BAC\"}",
         "```",
         "",
         "* `record[\"text\"]` — the sentence from the abstract",
         "* `record[\"labels\"]` — the move, as a three-letter code (singular value, "
         "despite the plural key)",
         "* `record[\"idx\"]` — which abstract it came from. **Careful:** `idx` starts "
         "again at 0 in the second file, so it is not on its own a unique id."],
        ["`path.read_text()` then `json.loads(...)` — the file as a list of records.",
         "`record[\"labels\"]` — the raw code, e.g. `BAC`.",
         "`RAAMOVE_LABELS[code]` — your dict, turning that code into the move name your "
         "prompt will use.",
         "`record[\"idx\"]` — used to group the flat list back into abstracts, so each "
         "sentence can carry the one it came from. See step 3.",
         "The two discipline files are read into one pool — an assumption, not a fact. "
         "See the note in step 3."]),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Two decisions:",
       "",
       "1. ✏️ **What each code is called.** `BAC` → `Background`. The expansion is not "
       "cosmetic: it is the wording your prompt will use, your annotators will read on "
       "the sheet, and your confusion matrix will be labelled with. `Gap` and "
       "`Establishing a niche` describe the same category and will not get you the same "
       "predictions.",
       "2. **Pool the two disciplines** — the code below reads both files into one set, "
       "treating a move as a rhetorical function rather than a discipline-specific one. "
       "That *is* an assumption. It is in the code rather than in a ✏️ cell only "
       "because unpicking it makes a better extension than a starting point: comparing "
       "Intelligence against Engineering separately would be a real finding.",
       "",
       "### Each sentence keeps its abstract",
       "",
       "A move is not a property of a sentence. *\"We used a mixed-effects model\"* is a "
       "`Method`; *\"the mixed-effects model showed no effect\"* is a `Result`; and "
       "plenty of real sentences sit between the two and are settled only by what came "
       "before them. So the code below does not throw the abstract away. It reads each "
       "file **twice** — once to group the flat list of sentences back into abstracts "
       "using `idx`, once to emit the items — and every item comes out carrying four "
       "extra fields on top of the canonical three:",
       "",
       "| field | what it is |",
       "|---|---|",
       "| `doc_id` | which abstract, e.g. `Intelligence-0` |",
       "| `sent_index` | where in it, counting from 0 |",
       "| `n_sents` | how many sentences the abstract has |",
       "| `context` | the abstract itself, one sentence per line |",
       "",
       "Those fields travel with the item all the way: notebook 03 shows the abstract "
       "to your two coders, and notebook 04 can put it in the prompt. Whether you "
       "*use* it is a decision for `PLAN.md` — `prompts/raamove.txt` shows the model "
       "the sentence alone, `prompts/raamove_context.txt` shows it the abstract first, "
       "and running both is one of the cleanest experiments this track offers."),
    *blank(
        "## Step 3a — Name the moves\n\nNow we write the label set: one name per "
        "three-letter code. The cell below already runs; the work is reading those "
        "eight names as a *scheme* and deciding whether they are the ones your coders "
        "should see.\n\n"
        "The names are not cosmetic. They are the wording your prompt uses, what your "
        "coders read on the sheet, and what your confusion matrix is labelled with. "
        "`Gap` and `Establishing a niche` name the same category and will not get you "
        "the same predictions.\n\n"
        "Eight classes is a lot. To **merge** two, give them the same name — "
        "`{\"RST\": \"Finding\", \"CLN\": \"Finding\"}` makes one class out of two. Decide "
        "that here and record it in `PLAN.md`, not after you have seen the model do "
        "badly on them.",
        "Step 3a · Name the moves",
        ["Sets the name your prompt, your sheet and your confusion matrix will use for",
         "each of the eight three-letter codes."],
        "RAAMOVE_LABELS (a dict)",
        starter=['# The corpus\'s own names, spelled out. They run as they are — but this',
                 '# wording goes into your prompt, onto your coders\' sheet and onto your',
                 '# confusion matrix, so read them as a scheme rather than as a given.',
                 '# Rewrite any that your coders would read differently, and give two',
                 '# codes the SAME name to merge them into one class.',
                 'RAAMOVE_LABELS = {',
                 '    "BAC": "Background",',
                 '    "GAP": "Gap",',
                 '    "MTD": "Method",',
                 '    "PUR": "Purpose",',
                 '    "RST": "Result",',
                 '    "CLN": "Conclusion",',
                 '    "CTN": "Contribution",',
                 '    "IMP": "Implication",',
                 '}',
                 '',
                 'print(RAAMOVE_LABELS)']),
    reading_note("Three functions. The middle one is the reshaping itself, and it reads "
                 "the `RAAMOVE_LABELS` you just defined — so if you skip the cell above, "
                 "this one will stop on `NameError: RAAMOVE_LABELS is not defined`."),
    *source_cells(
        [(reshape.reid,
          "`reid` renumbers items 1, 2, 3 … so that every item has an id of its own. "
          "Everything downstream joins on those ids."),
         (reshape.reshape_raamove,
          "`reshape_raamove` is the work: it reads both discipline files, groups the "
          "flat list of sentences back into abstracts, and emits one item per sentence "
          "with its abstract attached. Read the two passes."),
         (reshape.validate,
          "`validate` checks that every item has an id, a text and a label. Nothing "
          "calls it here — step 5 does, just before saving.")],
        imports=["import json", "from pathlib import Path"]),
    lead("## Step 3b — Run it",
         "",
         "Now we run the reshaping on the files we downloaded. It hands back a list of "
         "items in the canonical shape, which we call `rows`."),
    code("rows = reshape_raamove(RAW_DIR)",
         'print("built", len(rows), "items")'),
    md("## Step 4 — Check the label balance",
       "",
       "Now we look at what came out, because the balance decides what you can sample. "
       "This corpus is very imbalanced: `Method` is the biggest class by far, and "
       "`Implication` has only a couple of dozen sentences — which is a ceiling on any "
       "balanced draw you make in notebook 02."),
    *inspect_cells(),
    md("## Step 5 — Save it",
       "",
       "Three short cells: check that `config.yaml` agrees which track this is, check "
       "the shape of every item, then write the file."),
    *save_cell("raamove"),
    handoff("raamove"),
])

# ===================================================================== CaRS-50
save("01_build_pool_cars50.ipynb", [
    header(
        "CaRS-50 — build the pool",
        "Swales CARS moves in research-article introductions (3 or 11 classes)",
        "50 BioRxiv article introductions, annotated sentence by sentence with Swales' "
        "CARS Move and Step scheme. **The annotators themselves reached only κ ≈ 0.43** "
        "— so on this track, \"the model is wrong\" and \"the scheme is fuzzy\" are both "
        "live explanations, and telling them apart is the interesting part.",
        "CC BY 4.0",
        "Lam, C. & Nnamoko, N. (2025). *Mendeley Data*, V1. doi:10.17632/kwr9s5c4nk.1",
        "★★★ — hard. Judging moves in an introduction needs more context than a single "
        "sentence gives you.",
    ),
    *setup("cars50"),
    md("## Step 1 — Download the raw data",
       "",
       "This one is on **Mendeley Data**, which has a public API. We ask it for the "
       "dataset's file list, then download each file. The CDN refuses requests that do "
       "not look like a browser, hence the `User-Agent` header."),
    code("import json, urllib.request, pathlib",
         "",
         'RAW_DIR = pathlib.Path("cars50")     # the folder to download into',
         "RAW_DIR.mkdir(exist_ok=True)"),
    lead("Now we write a small function of our own. `fetch` opens one web address and "
         "hands back what is there — with a `User-Agent` header, because the server "
         "refuses requests that do not look like they came from a browser.",
         "",
         "A `def` gives a name to a few lines so they can be used more than once. This "
         "cell prints nothing; the next one calls it."),
    code("def fetch(url):",
         '    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})',
         "    return urllib.request.urlopen(request, timeout=60)"),
    lead("Now we ask Mendeley what files this dataset has, then download each one."),
    code("# One call to the Mendeley API, unpacked a step at a time.",
         'catalogue_url = "https://data.mendeley.com/public-api/datasets/kwr9s5c4nk"',
         "raw_text = fetch(catalogue_url).read()",
         "catalogue = json.loads(raw_text)",
         "",
         'for record in catalogue["files"]:',
         '    target = RAW_DIR / record["filename"]',
         "    if not target.exists():",
         '        download_url = record["content_details"]["download_url"]',
         "        target.write_bytes(fetch(download_url).read())",
         "",
         'xml_files = sorted(RAW_DIR.glob("*.xml"))',
         'print("downloaded", len(xml_files), "XML files")'),
    md("## Step 2 — Look at the raw format",
       "",
       "**XML** this time. Each sentence carries a `step` code like `1b`:",
       "",
       "```xml",
       "<sentence><sentenceID/><text/><step>1b</step></sentence>",
       "```",
       "",
       "Now we print the beginning of the first file, so you can see the real thing "
       "rather than that sketch of it."),
    code("first_file = xml_files[0]",
         'raw_xml = first_file.read_text(encoding="utf-8")',
         "print(raw_xml[:900])"),
    *parser_note(
        "Reading XML with `ElementTree`",
        "XML is nested, so unlike a TSV or a CSV you cannot get at a field by position. "
        "Python's built-in `xml.etree.ElementTree` parses the file into a tree of "
        "elements you then navigate by tag name.",
        ["**The nesting, outermost first:**",
         "",
         "* `<biology_intro>` — the root element, one per file",
         "* `<fulltext>` — the introduction itself",
         "* `<paragraph>` — one or more per introduction",
         "* `<sentence>` — one or more per paragraph, each carrying:",
         "  * `<sentenceID>` — an identifier",
         "  * `<text>` — the sentence",
         "  * `<step>` — the rhetorical step code, e.g. `1b`"],
        ["`ET.parse(path)` — read one file into a tree.",
         "`tree.iter(\"sentence\")` — every `<sentence>` at **any** depth, so you never "
         "have to walk the paragraphs yourself. It yields them in document order, "
         "which is what lets each sentence keep its place in the introduction.",
         "`element.find(\"text\")` — the first child with that tag, or `None` if it is "
         "missing. That `None` is why the reshaping code checks before using it.",
         "`element.text` — the string inside a tag. It is `None` for an empty tag, "
         "hence the `(… or \"\")` guard before `.strip()`.",
         "`path.stem` — the filename without `.xml`, e.g. `text001`. That is the "
         "document id. **Not** `<sentenceID>`: those are padded three different ways, "
         "mix two widths inside `text038.xml`, and `t025s020` appears twice in "
         "`text025.xml`. Position comes from counting, not from reading an id."],
        demos=[
            ("Now we parse that same file into a tree, so we can ask it for tags by "
             "name instead of hunting through the text.",
             ["import xml.etree.ElementTree as ET",
              "",
              "tree = ET.parse(first_file)",
              'print("reading", first_file.name)']),
            ("Now we walk the first three sentences and print what each one carries.",
             ["# Walk the first three sentences and print their three child tags.",
               'for position, sentence in enumerate(tree.iter("sentence")):',
               "    if position >= 3:                # just the first three, to keep this short",
               "        break",
               '    for tag in ("sentenceID", "text", "step"):',
               "        child = sentence.find(tag)",
               "        # A tag can be missing (child is None) or present but empty",
               "        # (child.text is None). Both mean there is nothing to read.",
               "        if child is None or child.text is None:",
               '            value = "MISSING"',
               "        else:",
               "            value = child.text.strip()",
               '        print(" ", tag + ":", value)',
               '    print()']),
        ]),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "The parsing is written for you, and it gives you **both granularities at once**:",
       "",
       "- the leading digit of `1b` is the **Move** → 3 classes;",
       "- the whole code `1b` is the **Step** → 11 classes.",
       "",
       "Sentences with no code, or a code that does not start with a move digit, are "
       "dropped either way. In *this* corpus that guard never actually fires — all 1297 "
       "sentences are coded — so do not write \"we dropped N malformed sentences\" in "
       "your report without checking the number first. The guard is there because the "
       "next corpus you meet will need it.",
       "",
       "✏️ **Which one you study is the decision**, and on this track it is the whole "
       "shape of the project. Three classes with a few hundred items each is a fair task "
       "you can sample 40 items from comfortably. Eleven classes over the same "
       "sentences means some steps have barely a dozen examples, a confusion matrix with "
       "121 cells, and an annotation job your two coders will find genuinely hard — "
       "remember the original annotators managed only κ ≈ 0.43 at this granularity.",
       "",
       "Neither is the safe answer. The 11-class version makes a better project **if** "
       "you have the time to annotate it properly and the nerve to report a low F1 with "
       "a good explanation. Decide now, write it in `PLAN.md`, and do not switch after "
       "you have seen the numbers.",
       "",
       "### Each sentence keeps its introduction",
       "",
       "The difficulty note at the top of this notebook says judging a move needs more "
       "context than a single sentence gives you. So the code below does not throw the "
       "introduction away. It reads each file **twice** — once to collect the whole "
       "introduction in order, once to emit the items — and every item comes out "
       "carrying four extra fields on top of the canonical three:",
       "",
       "| field | what it is |",
       "|---|---|",
       "| `doc_id` | which introduction, e.g. `text001` |",
       "| `sent_index` | where in it, counting from 0 |",
       "| `n_sents` | how many sentences the introduction has |",
       "| `context` | the introduction itself, one sentence per line |",
       "",
       "One detail worth arguing about: `context` keeps **every** sentence that has "
       "text, including the ones dropped for having no usable code. They belong there "
       "because a reader saw them — filtering them out would hand the model a doctored "
       "introduction that never existed. Both granularities get identical fields; it is "
       "the same sentence in the same passage, just labelled two ways.",
       "",
       "Those fields travel with the item all the way: notebook 03 shows the "
       "introduction to your two coders, and notebook 04 can put it in the prompt. "
       "`prompts/cars50.txt` shows the model the sentence alone, "
       "`prompts/cars50_context.txt` shows it the introduction first. These are 26 "
       "sentences on average and up to 47, so the context condition is noticeably "
       "slower to run — worth knowing before you start it at 4pm."),
    reading_note("Three functions. `reshape_cars50` is the one to read: it is where the "
                 "one parse becomes two datasets."),
    *source_cells(
        [(reshape.reid,
          "`reid` renumbers items 1, 2, 3 … so that every item has an id of its own."),
         (reshape._tag_text,
          "`_tag_text` reads the text inside one XML tag. It is separate because a tag "
          "can be missing *or* present-but-empty, and both have to come back as \"\". "
          "The leading underscore is a convention meaning \"a helper for the function "
          "below\" — it is not a typo, and nothing stops you calling it."),
         (reshape.reshape_cars50,
          "`reshape_cars50` is the work: one walk through the XML, two lists out — the "
          "3-class moves and the 11-class steps. Read the two passes."),
         (reshape.validate,
          "`validate` checks that every item has an id, a text and a label. Nothing "
          "calls it here — step 5 does, just before saving.")],
        imports=["import xml.etree.ElementTree as ET", "from pathlib import Path"]),
    lead("## Step 3a — Run it",
         "",
         "Now we run the parsing over all 50 files. It hands back **two** lists at once "
         "— that is what the comma on the left of the `=` means — one labelled at each "
         "granularity."),
    code("move_rows, step_rows = reshape_cars50(RAW_DIR)",
         'print("moves:", len(move_rows), " steps:", len(step_rows))'),
    lead("Now we count the classes in each, because how many items the smallest class "
         "has is the ceiling on any balanced draw you make in notebook 02."),
    code("def count_labels(items):",
         "    counts = {}",
         "    for item in items:",
         '        label = item["label"]',
         "        if label not in counts:",
         "            counts[label] = 0",
         "        counts[label] = counts[label] + 1",
         "    return counts",
         "",
         'print("move classes:", count_labels(move_rows))',
         'print("step classes:", count_labels(step_rows))'),
    *blank(
        "## Step 3b — Choose your granularity\n\nNow we pick which of the two schemes "
        "your group will actually study — the 3 moves or the 11 steps — and give it the "
        "name `rows` that the rest of the notebook uses.\n\n"
        "It is one word to change. Spend the time on the argument instead: `PLAN.md` "
        "asks you to justify the choice in a sentence.\n\n"
        "**Whichever you pick**, the label names in your prompt and on your annotation "
        "sheet have to match these exactly — `Move 1`, or `1b`.",
        "Step 3b · Choose your granularity",
        ["Names one of the two schemes as the one you will study, and prints how many",
         "items it has."],
        "rows (a list) — either move_rows or step_rows",
        starter=['# Change this one word to "step" for the 11-class version.',
                 '# It is written as a choice rather than two lines you delete one of,',
                 '# because with two live lines the SECOND one silently wins — and you',
                 '# would not find out until notebook 03, halfway through annotating.',
                 'GRANULARITY = "move"',
                 "",
                 'if GRANULARITY == "move":',
                 "    rows = move_rows          # 3 classes: Move 1 · Move 2 · Move 3",
                 'elif GRANULARITY == "step":',
                 "    rows = step_rows          # 11 classes: 1a · 1b · 2a … the finer scheme",
                 "else:",
                 '    raise ValueError(',
                 '        "GRANULARITY has to be either \\"move\\" or \\"step\\", and it says "',
                 '        + repr(GRANULARITY) + ". Fix the line above and run this cell again.")',
                 "",
                 'print("studying the", GRANULARITY, "scheme:", len(rows), "items")']),
    md("## Step 4 — Check the label balance",
       "",
       "Now we look at what you chose, in the shape it will actually be annotated in."),
    *inspect_cells(),
    md("## Step 5 — Save it",
       "",
       "Three short cells: check that `config.yaml` agrees which track this is, check "
       "the shape of every item, then write the file."),
    *save_cell("cars50",
              'If you chose steps, set  track: cars50_step  in config.yaml before running '
              "this: POOL_PATH then becomes cars50_step_pool.json, and the finer scheme "
              "gets its own file rather than overwriting the 3-move one. Notebook 04 "
              "then reads prompts/cars50_step.txt, which is already there — a baseline "
              "naming the eleven codes, for you to improve on.",
               variants=("cars50_step",)),
    handoff("cars50"),
])

# ===================================================================== L2 errors
save("01_build_pool_l2_errors.ipynb", [
    header(
        "AutoErrorAnalyzer — build the pool",
        "Error type in a learner sentence (4 classes), or error detection (2 classes)",
        "~100 Japanese-EFL essays, annotated with a 26-category error taxonomy "
        "(Krippendorff's α ≈ .92). The file also holds **the published tool's own "
        "predictions**, so this is the one track where you can benchmark your LLM "
        "against both a human gold standard *and* an existing system.",
        "CC BY 4.0",
        "Mizumoto, A. (2025). *Studies in Second Language Acquisition, 47*(3), 867–884. "
        "OSF: osf.io/jyf3r",
        "★★★ — hard. Many error types, and a sentence can carry several at once.",
    ),
    *setup("l2_errors"),
    md("## Step 1 — Download the raw data",
       "",
       "The annotations live on the paper's **OSF** project. Now we fetch one CSV "
       "directly by its OSF link."),
    code("import urllib.request",
         "",
         'RAW_FILE = "data_category.csv"',
         'urllib.request.urlretrieve("https://osf.io/download/gezat/", RAW_FILE)',
         'print("downloaded", RAW_FILE)'),
    md("## Step 2 — Look at the raw format",
       "",
       "A **CSV**. The columns that matter are `Sentence`, `Human_ErrorCategories` (the "
       "gold) and `AEA_ErrorCategories` (the tool's prediction). A sentence can carry "
       "several comma-separated error codes, or `NO_ERROR`.",
       "",
       "The cell below prints **every code in the file, with its frequency**. You need "
       "that list in front of you for step 3 — it is the taxonomy you are about to "
       "collapse."),
    code("import csv",
         "",
         '# DictReader hands you each row as {column name: value}.',
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)'),
    lead("Now we count how often each error code appears, which is the taxonomy you are "
         "about to collapse in step 3.",
         "",
         "One row can carry several codes, comma-separated. A row with no error has "
         "nothing in that column at all, which Python reads as `None` — so `or \"\"` "
         "below stands in an empty string for it, because you cannot split `None`."),
    code("codes = {}",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    for row in csv.DictReader(f):",
         '        human_field = row["Human_ErrorCategories"] or ""',
         '        for code in human_field.split(","):',
         "            code = code.strip()",
         "            if code:",
         "                if code not in codes:",
         "                    codes[code] = 0",
         "                codes[code] = codes[code] + 1",
         "",
         'print(len(codes), "distinct codes:")',
         "for code, n in codes.most_common():",
         '    print("   ", code, n)'),
    *parser_note(
        "Reading CSV with `csv.DictReader`",
        "`csv.DictReader` reads a CSV using its header row, so each row arrives as a "
        "dict keyed by column name — `row[\"Sentence\"]` rather than `row[3]`. That is "
        "what the cell above used to print `fieldnames` and count the codes.",
        ["**The columns that matter:**",
         "",
         "* `Sentence` — the text",
         "* `Human_ErrorCategories` — the human annotation, and your gold. One sentence "
         "can carry **several comma-separated codes**, or the single marker `NO_ERROR`.",
         "* `AEA_ErrorCategories` — the published tool's own prediction, which is what "
         "lets this track compare an LLM against an existing system as well as against "
         "humans"],
        ["`open(..., encoding=\"utf-8-sig\")` — this file ships with a byte-order mark. "
         "Without `-sig` it gets glued to the first column name and every lookup on it "
         "fails.",
         "`csv.DictReader(handle)` — iterate rows as `{column: value}` dicts.",
         "`(row.get(col) or \"\").strip()` — `.get` survives a missing column and the "
         "`or \"\"` survives an empty cell, which would otherwise be `None`.",
         "`human_field.split(\",\")` — one sentence's codes into a list, which "
         "`_l2_coarse_label` then maps through **your** `L2_COARSE` grouping."]),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Two decisions, and the first is the biggest single judgment call in any of the "
       "four tracks:",
       "",
       "1. ✏️ **Collapse the ~23 codes into a handful of categories.** The full "
       "taxonomy is too fine-grained to prompt for reliably at this scale, so you group "
       "it. Where you draw those boundaries decides what your study is *about*: "
       "Grammatical / Lexical / Mechanical / No error is one defensible cut, and it is "
       "the one this track was written around — but it is not the only one. Is a wrong "
       "preposition grammatical or lexical? Is spelling mechanical, or a lexical problem "
       "wearing a mechanical hat? Whatever you decide, **every code you do not list is a "
       "sentence you throw away**.",
       "2. **Drop mixed-category sentences** — a sentence whose codes span more than one "
       "of your categories gets no label, so the task stays single-label. That is in the "
       "code below. It also means the dataset under-represents exactly the messiest "
       "sentences, which **belongs in your limitations section**.",
       "",
       "You also get a binary detection version for free (any error at all: yes/no), "
       "which does not depend on your grouping at all.",
       "",
       "> Read `_l2_coarse_label` below before you write your mapping. It returns `None` "
       "— i.e. drops the sentence — when the codes span more than one of your "
       "categories, so a *coarser* grouping keeps more data and a *finer* one keeps "
       "less. That trade-off is yours to make and to report."),
    *blank(
        "## Step 3a — Group the error codes\n\nNow we collapse the published taxonomy "
        "into the handful of categories your group will annotate: one line per code "
        "from step 2 that you want to keep, with your category name on the right. "
        "`NO_ERROR` is handled separately, so do not list it. This is the decision on "
        "this track.\n\n"
        "**A code you leave out is not an error — it is a dropped sentence.** Compare "
        "your total in step 4 against the detection count to see how many.\n\n"
        "The four category names in the cell are the cut this track was written around, "
        "and they are only a starting point. Rename them, merge them, or use two "
        "categories instead of four — what matters is that you can say why.\n\n"
        "Put the grouping in `PLAN.md` as a table, with a one-line justification for "
        "any code a reasonable person would file somewhere else. That table is report "
        "section 1.",
        "Step 3a · Group the error codes",
        ["Maps each raw error code you are keeping to the broader category you will",
         "study, and prints how many codes and categories that leaves you with."],
        "L2_COARSE (a dict)",
        starter=[
            "# One line per code you are KEEPING. Scroll up to step 2 for the full list",
            "# with frequencies — the frequent codes are the ones worth arguing over.",
            "#",
            "# The right-hand side is your category name, and repeating one is how you",
            "# merge: every code you send to \"Grammatical\" becomes one class.",
            "",
            "L2_COARSE = {",
            '    "ART": "Grammatical",      # articles — given as an example of the shape',
            '    "SP": "Mechanical",        # spelling — mechanical, or lexical? your call',
            '    "PREP": "Grammatical",     # prepositions — grammatical, or lexical?',
            '    "TENSE": "Grammatical",    # verb tense',
            '    "N": "Lexical",            # wrong noun choice',
            "    # … keep going, one line per code from step 2 that you want to study.",
            "    # Four lines is not a study. Argue about each one before you add it.",
            "}",
            "",
            'print(len(L2_COARSE), "codes kept ·", sorted(set(L2_COARSE.values())))',
        ]),
    reading_note("Four functions. The first two read the `L2_COARSE` you just defined, "
                 "so if you skip the cell above they will stop on "
                 "`NameError: L2_COARSE is not defined`."),
    *source_cells(
        [(reshape._l2_coarse_label,
          "`_l2_coarse_label` turns a sentence's comma-separated codes into ONE broader "
          "category, or `None` when the codes span more than one — those get dropped. "
          "The leading underscore is a convention meaning \"a helper for the function "
          "below\"; it is not a typo, and nothing stops you calling it."),
         (reshape.reid,
          "`reid` renumbers items 1, 2, 3 … so that every item has an id of its own."),
         (reshape.reshape_l2_errors,
          "`reshape_l2_errors` is the work: one pass over the CSV, two lists out — your "
          "categories, and the yes/no detection version."),
         (reshape.validate,
          "`validate` checks that every item has an id, a text and a label. Nothing "
          "calls it here — step 5 does, just before saving.")],
        imports=["import csv"]),
    lead("## Step 3b — Run it",
         "",
         "Now we run the reshaping over the CSV. It hands back **two** lists at once — "
         "that is what the comma on the left of the `=` means."),
    code("category_rows, detection_rows = reshape_l2_errors(RAW_FILE)",
         'print("categories:", len(category_rows), " detection:", len(detection_rows))'),
    *blank(
        "## Step 3c — Choose your task\n\nNow we pick which of the two tasks your group "
        "will study — the n-way categorisation from 3a, or the yes/no detection task — "
        "and give it the name `rows` that the rest of the notebook uses.\n\n"
        "Detection is a genuinely easier task and a smaller project. If you take it, "
        "plan an extension: benchmarking against the published tool's own predictions "
        "is right there in the CSV.",
        "Step 3c · Choose your task",
        ["Names one of the two tasks as the one you will study, and prints how many",
         "items it has."],
        "rows (a list) — either category_rows or detection_rows",
        starter=['# Change this one word to "detection" for the yes/no version.',
                 '# It is written as a choice rather than two lines you delete one of,',
                 '# because with two live lines the SECOND one silently wins — and you',
                 '# would not find out until notebook 03, halfway through annotating.',
                 'TASK = "category"',
                 "",
                 'if TASK == "category":',
                 "    rows = category_rows      # your categories from 3a",
                 'elif TASK == "detection":',
                 "    rows = detection_rows     # Has error / No error",
                 "else:",
                 '    raise ValueError(',
                 '        "TASK has to be either \\"category\\" or \\"detection\\", and it says "',
                 '        + repr(TASK) + ". Fix the line above and run this cell again.")',
                 "",
                 'print("studying the", TASK, "task:", len(rows), "items")']),
    md("## Step 4 — Check the label balance",
       "",
       "Note how many sentences were dropped: compare your category count against the "
       "detection count, which keeps everything. The gap is your mixed-category "
       "sentences, and it is a direct consequence of the grouping you wrote in 3a."),
    *inspect_cells(),
    md("## Step 5 — Save it",
       "",
       "Three short cells: check that `config.yaml` agrees which track this is, check "
       "the shape of every item, then write the file."),
    *save_cell("l2_errors",
              'For the binary version, set  track: l2_error_detection  in config.yaml '
              "before running this, so POOL_PATH becomes l2_error_detection_pool.json "
              "and the two versions do not overwrite each other. Notebook 04 then reads "
              "prompts/l2_error_detection.txt, which is already there as a baseline.",
               variants=("l2_error_detection",)),
    handoff("l2_errors"),
])

# ===================================================================== ICNALE GRA
save("01_build_pool_icnale.ipynb", [
    header(
        "ICNALE GRA — build the pool",
        "Holistic essay score band (Low / Mid / High)",
        "Asian-learner L2 English essays, each rated on holistic and analytic scales by "
        "many trained raters. This is an **automated writing evaluation** task: whole "
        "essays, not sentences.",
        "⚠️ **Research use only — NOT redistributable.** Requires registration. Nothing "
        "derived from it may be committed to git or included in your submission bundle.",
        "Ishikawa, S. *The ICNALE Global Rating Archives.*",
        "★★☆ — moderate, but a different shape of task: long texts and an ordered scale.",
    ),
    *setup("icnale"),
    md("## Step 1 — Get the data (this one is manual)",
       "",
       "ICNALE GRA is released for research use behind a registration form that emails "
       "you a password. There is nothing to automate, and that is deliberate — the "
       "licence does not permit redistribution.",
       "",
       "1. Register at "
       "<https://language.sakura.ne.jp/icnale/download.html> and wait for the password.",
       "2. Download and unpack `ICNALE_GRA_2.x.zip`.",
       "3. From its rating tables, export a CSV with **exactly two columns**, `text` and "
       "`score`.",
       "",
       "In Colab, the cell below opens a file picker. Every other track downloads its "
       "corpus in one command and so keeps the raw data in the runtime — this one you "
       "cannot re-fetch without going back through the registration form, so it is "
       "worth keeping the file in your group's Drive folder and uploading it only "
       "once. The second option below does that.",
       "",
       "⚠️ `data/raw/` is excluded from git and from your submission bundle, and "
       "anything with `icnale` in the name is excluded twice over. Leave it that way — "
       "the licence does not permit redistribution."),
    lead("**First, get the file into this session.** Run the cell below and a file "
         "picker opens; choose your `essays_scores.csv`.",
         "",
         "Skip this cell if you have already put the file in your group's Drive folder "
         "— the next cell points at it there."),
    code("from google.colab import files",
         "",
         "files.upload()"),
    lead("**Now we say where the file is.** One of the two lines below is live and the "
         "other is commented out. Keep the first if you just uploaded the file; switch "
         "to the second, by moving the `#`, once you have put a copy in your group's "
         "Drive folder and want to stop uploading it every session."),
    code('RAW_FILE = "essays_scores.csv"        # the copy you just uploaded',
         "",
         "# The copy in your group's Drive folder, if you put one there:",
         '# RAW_FILE = str(ROOT / "data" / "raw" / "icnale" / "essays_scores.csv")',
         "",
         'print("using", RAW_FILE)'),
    lead("**Now we check the file is really there.** We say so here rather than three "
         "cells down, where the same problem arrives as a bare `FileNotFoundError` from "
         "inside `open`, with nothing to tell you what to do about it."),
    code("import os",
         "",
         "if not os.path.isfile(RAW_FILE):",
         "    raise FileNotFoundError(",
         '        RAW_FILE + " is not here.\\n"',
         '        "This is the one track with no automatic download: register at "',
         '        "https://language.sakura.ne.jp/icnale/download.html, export a CSV "',
         '        "with a text column and a score column, then either uncomment the "',
         '        "upload line above and run this cell again, or put the file in "',
         '        "data/raw/icnale/ in your group\'s Drive folder and use the second "',
         '        "RAW_FILE line.")',
         "",
         'print("found it:", RAW_FILE)'),
    md("## Step 2 — Look at the raw format",
       "",
       "The cell prints the **distribution** of the scores, not just a couple of rows. "
       "You need that before step 3: it is what tells you where cutting the scale leaves "
       "you with three usable classes rather than one big one and two nearly empty "
       "ones."),
    code("import csv",
         "",
         "# Collect the scores. They arrive as strings, so each one is converted to a",
         "# number - and a cell that is not a number at all is counted, not ignored.",
         "scores = []",
         "not_a_number = 0",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)',
         "    for row in reader:",
         "        try:",
         '            scores.append(float(row["score"]))',
         "        except (TypeError, ValueError):",
         "            not_a_number = not_a_number + 1",
         "",
         'print(len(scores), "usable scores ·", not_a_number, "cells that were not numbers")'),
    lead("Now we look at how those scores are spread out.",
         "",
         "Read the percentiles carefully — they are what step 3 asks you to cut the "
         "scale with. Cutting at the 33rd and 67th gives you three classes of roughly "
         "equal size; cutting anywhere else does not, and you will see that in step 4."),
    code("scores.sort()             # smallest first, so we can pick positions out of it",
         'print("min", scores[0], "· max", scores[-1])',
         "",
         "for q in (10, 25, 33, 50, 67, 75, 90):",
         "    # A rough percentile: the score q% of the way along the sorted list.",
         "    position = int(len(scores) * q / 100)",
         '    print("   ", str(q) + "th percentile:", scores[position])'),
    *parser_note(
        "Reading CSV with `csv.DictReader`",
        "`csv.DictReader` reads a CSV using its header row, so each row arrives as a "
        "dict keyed by column name — `row[\"score\"]` rather than `row[1]`. That is what "
        "the cell above used to collect the scores.",
        ["**The two columns that matter:**",
         "",
         "* `text` — the essay",
         "* `score` — the holistic score. **It arrives as a string**, even when it looks "
         "like a number, so it has to be converted before it can be compared to a "
         "boundary."],
        ["`open(..., encoding=\"utf-8-sig\")` — strips the byte-order mark this file "
         "ships with, which would otherwise be glued to the first column name.",
         "`csv.DictReader(handle)` — iterate rows as `{column: value}` dicts.",
         "`float(raw_score)` inside a `try` — a non-numeric cell is counted and skipped "
         "rather than crashing the run. The function prints how many it skipped; if that "
         "number is not small, look at the file before trusting the rest.",
         "`score < low_below` / `score < mid_below` — the two cut-offs you are about to "
         "choose. Everything else in the function is fixed; this is the whole decision."]),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "One decision, and it is entirely yours: ✏️ **where do the band boundaries go?**",
       "",
       "There is no right answer sitting in the data waiting to be found. Two honest "
       "ways to choose, and they disagree:",
       "",
       "- **From the rubric** — if the scale you are using says what a Low essay is, use "
       "that. Your classes will come out uneven, possibly badly, but they mean something "
       "outside your own study.",
       "- **From the distribution** — cut at the 33rd and 67th percentiles (printed "
       "above) and your classes come out balanced. Your F1 is then easier to read, and "
       "your bands mean nothing except \"bottom third of this sample\".",
       "",
       "Pick one, say which in `PLAN.md`, and report the boundaries as numbers. A band "
       "definition that exists only as an unexplained `4.0` in a notebook is not a "
       "scheme.",
       "",
       "⚠️ These labels are **ordered** (Low < Mid < High) but they are *not* "
       "alphabetical. List them under `labels_order:` in `config.yaml` — Low, then Mid, "
       "then High — or the weighted κ gets computed over `High < Low < Mid`, which "
       "means nothing."),
    reading_note("Three functions. `reshape_icnale` is the one to read: the two "
                 "boundaries you are about to choose are its only arguments."),
    *source_cells(
        [(reshape.reid,
          "`reid` renumbers items 1, 2, 3 … so that every item has an id of its own."),
         (reshape.reshape_icnale,
          "`reshape_icnale` reads the CSV and turns each score into a band. Find the "
          "`if / elif / else` — those three lines are the whole scheme."),
         (reshape.validate,
          "`validate` checks that every item has an id, a text and a label. Nothing "
          "calls it here — step 5 does, just before saving.")],
        imports=["import csv"]),
    *blank(
        "## Step 3a — Cut the scale\n\nNow we turn each numeric score into one of three "
        "bands. The two boundaries are yours to choose, and the percentiles printed "
        "above are the evidence for choosing them.\n\n"
        "The defaults, 4.0 and 7.0, are round numbers rather than a rubric — leaving "
        "them is as much a decision as changing them, and you have to defend it either "
        "way. Run it a couple of ways and look at step 4 each time: watching the counts "
        "move as you shift a boundary is the point.\n\n"
        "**Whatever you settle on goes in `PLAN.md`** as two numbers and a reason. Do "
        "not re-cut the scale after you have seen your F1.",
        "Step 3a · Cut the scale",
        ["Turns each numeric score into a Low, Mid or High band, split at the two",
         "boundaries you give it."],
        "rows (a list)",
        starter=["# A score below low_below is Low; below mid_below is Mid; the rest High.",
                 "#",
                 "# 4.0 and 7.0 are the defaults, and they are round numbers rather than",
                 "# a rubric — leaving them is as much a decision as changing them, and",
                 "# you have to defend it either way. Re-run with the percentiles printed",
                 "# above and watch the counts in step 4 move.",
                 "rows = reshape_icnale(RAW_FILE, low_below=4.0, mid_below=7.0)",
                 "",
                 'print(len(rows), "essays")']),
    md("## Step 4 — Check the label balance",
       "",
       "Now we look at what your two boundaries actually produced. If one band has "
       "almost everything in it, go back to step 3a and move a cut-off."),
    *inspect_cells(),
    md("## Step 5 — Save it",
       "",
       "Three short cells: check that `config.yaml` agrees which track this is, check "
       "the shape of every item, then write the file.",
       "",
       "⚠️ Keep this file **out of git** and **out of your submission bundle**. "
       "`.gitignore` and `scripts/make_submission.py` both exclude anything with "
       "`icnale` in the name — please leave that in place."),
    *save_cell("icnale"),
    handoff("icnale"),
])

print("\nDone. To check nothing drifted from reshape.py:")
print("  python scripts/_generate_pool_notebooks.py && git diff --exit-code notebooks/")
