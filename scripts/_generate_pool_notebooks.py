#!/usr/bin/env python3
"""Generate notebooks/01_build_pool_<track>.ipynb — one per track.

    python scripts/_generate_pool_notebooks.py

Notebook 01 is where a track's pool comes from: download -> look at the raw format ->
reshape to {id, text, label} -> check the label balance -> save
data/pools/<track>_pool.json. Notebook 02 picks it up from there.

It is standalone (stdlib only, no repo needed), so it runs in a fresh Colab.

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


def embed(*objects, imports=()):
    """Render live functions/constants from reshape.py as notebook source text.

    Functions come through `inspect.getsource`, so what the notebook shows is literally
    what prep_datasets.py runs. Constants are rendered from their real values - but the
    ones that encode a decision are NOT passed here; a blank cell asks for those.

    `imports` are the module-level names the embedded source needs. reshape.py has them
    at the top of the file, which inspect.getsource does not carry across - so without
    this the cell dies on `NameError: Path` the first time it is run.
    """
    blocks = []
    if imports:
        blocks.append("\n".join(imports))
    for obj in objects:
        if isinstance(obj, tuple):          # ("NAME", value) -> a constant
            name, value = obj
            blocks.append(name + " = " + repr(value))
        else:
            blocks.append(inspect.getsource(obj).rstrip("\n"))
    return "\n\n".join(blocks)


def blank(title, goal, produces, hints, notes=()):
    """A blank cell: what to write, what it must be called, and what it is FOR.

    Never blank something without saying what the next cell expects to find - a
    student stuck on a NAME has learned nothing about annotation.
    """
    rule = "─" * max(4, 60 - len(title))
    lines = ["# ✏️ " + title + " " + rule,
             "# Goal      : " + goal]
    for index, hint in enumerate(hints):
        lines.append(("# Shape     : " if index == 0 else "#             ") + hint)
    lines.append("# Produce   : " + produces + "      ← later cells use this name")
    for note in notes:
        lines.append("# " + note)
    lines.append("")
    lines.append("# ✏️ your code here")
    lines.append("")
    return code(*lines)


SCHEMA_NOTE = (
    "Every dataset in this course is reshaped into the **same canonical schema**, so one "
    "pipeline works for all of them:\n\n"
    "```json\n"
    "[{\"id\": 1, \"text\": \"...\", \"label\": \"...\"}]\n"
    "```\n\n"
    "The *raw* data, though, looks different every time. **That difference is the "
    "lesson** — half of building a gold standard is getting messy real data into a "
    "clean, consistent shape."
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


def inspect_cell():
    return code(
        "from collections import Counter",
        "",
        'print("total items:", len(rows))',
        'print("label counts:", dict(Counter(item["label"] for item in rows)))',
        "rows[:3]        # peek at the first three reshaped items")


def balance_blank(hint):
    """Step 4b is blank on every track: read your own counts and react to them."""
    return blank(
        "Step 4b · React to the balance",
        "decide what the counts you just printed mean for your study.",
        "MIN_PER_CLASS (an int)",
        ["MIN_PER_CLASS = <the size of your SMALLEST class>",
         "that is the ceiling on N_PER_CLASS in config.py — a balanced",
         "sample cannot draw more from a class than the class has"],
        notes=["Note      : " + hint,
               "Note      : if the rarest class is tiny, say so in PLAN.md. Merging it",
               "            away or living with fewer items are both defensible;",
               "            not noticing is not."])


def save_cell(track, note=""):
    lines = [
        "# Save the pool. Two places you might want it:",
        '#   * this repo, if you cloned it:  "../data/pools/' + track + '_pool.json"',
        "#   * your Google Drive, so it survives the Colab runtime resetting",
        "import json, pathlib",
        "",
        'OUT_FILE = "../data/pools/' + track + '_pool.json"',
        "",
        "# In Colab WITHOUT the repo, uncomment these two to write straight to Drive:",
        '# from google.colab import drive; drive.mount("/content/drive")',
        '# OUT_FILE = "/content/drive/MyDrive/' + track + '_pool.json"',
        "",
        "pathlib.Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)",
        'with open(OUT_FILE, "w", encoding="utf-8") as f:',
        "    json.dump(rows, f, ensure_ascii=False, indent=2)",
        'print("Saved", len(rows), "items to", OUT_FILE)',
    ]
    if note:
        lines = lines + ["", "# " + note]
    return code(*lines)


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
        "**Next:** set `TRACK = \"" + track + "\"` in `config.py`, then open "
        "`02_sample.ipynb`.")


# ===================================================================== CEFR-SP
save("01_build_pool_cefr.ipynb", [
    header(
        "CEFR-SP — build the pool",
        "The on-ramp track: sentence proficiency level (A1–C2)",
        "Sentences annotated with a CEFR level by two trained annotators. We use the "
        "openly-shipped **Wiki-Auto** portion.",
        "CC BY-SA 3.0 (Wiki-Auto portion) — **share-alike**, so anything you "
        "redistribute from it inherits the same licence.",
        "Arase, Uchida & Kajiwara (2022), *EMNLP*. github.com/yukiar/CEFR-SP",
        "★☆☆ — easy. Levels are concrete and the annotators usually agree.",
    ),
    md("## Step 1 — Download the raw data",
       "",
       "The corpus lives in a GitHub repository, so we clone it. (`!` runs a shell "
       "command from inside the notebook.)"),
    code("!git clone --depth 1 https://github.com/yukiar/CEFR-SP"),
    md("## Step 2 — Look at the raw format",
       "",
       "Note the folder path: cloning `CEFR-SP` gives you a `CEFR-SP` folder *inside* "
       "`CEFR-SP`. Easy to trip over.",
       "",
       "The Wiki-Auto files are **tab-separated text**, one sentence per line:",
       "",
       "```",
       "sentence <TAB> label_by_annotator_A <TAB> label_by_annotator_B",
       "```",
       "",
       "Labels are digits: `1`=A1, `2`=A2, … `6`=C2. **Look at the output before you "
       "write anything below** — the mapping you are about to type has to match what is "
       "actually in the file."),
    code('RAW_DIR = "CEFR-SP/CEFR-SP/Wiki-Auto"',
         "",
         'with open(RAW_DIR + "/CEFR-SP_Wikiauto_dev.txt", encoding="utf-8") as f:',
         "    for _ in range(5):",
         "        print(repr(next(f)))"),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Three decisions are baked into this track, and the first is yours to write:",
       "",
       "1. ✏️ **What the levels are called.** The file says `1`; a prompt that says "
       "`A1` needs far less explaining than one that says `1`. You supply that mapping "
       "— and its keys are also a filter: a row whose digit is not in your mapping gets "
       "dropped, so leaving a level out silently removes it from your study.",
       "2. **Trust only agreement.** Each sentence has *two* annotators, and the code "
       "below keeps a row only when both chose the same level (`label_a == label_b`). "
       "Every remaining label is then unambiguous — which is what makes this the gentle "
       "track, and also makes it easier than the data really is. Read that line and "
       "make sure you can say why it is there; keeping the disagreements would have "
       "meant deciding whose label wins.",
       "3. **Wiki-Auto only** — the repo also ships a `SCoRE/` folder under a "
       "*non-commercial* licence, and we deliberately never read it. Notice `RAW_DIR` "
       "points at `Wiki-Auto` specifically rather than at the repo root: that is what "
       "keeps the two apart."),
    blank(
        "Step 3a · Name the levels",
        "map the digit in the file to the label a prompt can actually use.",
        "CEFR_NUM (a dict)",
        ['CEFR_NUM = {"1": "A1", ...}   # keys are STRINGS — the file is text',
         "six entries, one per level"],
        notes=["Careful   : a digit you leave out is a level you silently DROP.",
               "            Check the counts in step 4 against what you expected.",
               "Why blank : this is the label set your prompt, your annotation sheet and",
               "            your confusion matrix all inherit. Put it in PLAN.md."]),
    md("Now the reshaping function itself. It reads the `CEFR_NUM` you just defined — "
       "if the cell below raises `NameError: CEFR_NUM`, go back and run the one above."),
    code(embed(reshape.reid, reshape.reshape_cefr,
               imports=["from pathlib import Path"])),
    code("rows = reshape_cefr(RAW_DIR)",
         'print("kept", len(rows), "sentences where both annotators agreed")'),
    md("## Step 4 — Check the label balance",
       "",
       "Look at the counts before you trust anything downstream. This corpus is "
       "**heavily imbalanced** — B1 and B2 dominate, and A1/C2 are scarce."),
    inspect_cell(),
    balance_blank("the agreement filter costs you rows unevenly — the levels "
                  "annotators argue about lose the most."),
    md("## Step 5 — Save it"),
    save_cell("cefr"),
    handoff("cefr"),
])

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
    md("## Step 1 — Download the raw data"),
    code("!git clone --depth 1 https://github.com/ljk1228/RAAMove"),
    md("## Step 2 — Look at the raw format",
       "",
       "This one is **JSON**, split into two files by discipline "
       "(`Intelligence.json`, `Engineering.json`). Each record has a `text` and a "
       "three-letter move code in `labels`.",
       "",
       "**Read the codes the cell below prints** — you are about to name every one."),
    code('RAW_DIR = "RAAMove"',
         "",
         "import json",
         "from collections import Counter",
         "",
         'data = json.loads(open(RAW_DIR + "/Intelligence.json", encoding="utf-8").read())',
         'print("records:", len(data))',
         'print("codes:", Counter(record["labels"] for record in data))',
         "data[:3]"),
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
       "Intelligence against Engineering separately would be a real finding."),
    blank(
        "Step 3a · Name the moves",
        "map each three-letter code to the move name your prompt will use.",
        "RAAMOVE_LABELS (a dict)",
        ['RAAMOVE_LABELS = {"BAC": "Background", ...}',
         "one entry per code you saw in step 2"],
        notes=["Careful   : a code you leave out is NOT dropped — it is kept as the raw",
               "            code, so it shows up as a stray label like \"CTN\" in step 4.",
               "            That is your check that you got them all.",
               "Note      : eight classes is a lot. If you plan to merge any (Result +",
               "            Conclusion, say), decide it HERE and say so in PLAN.md — not",
               "            after you have seen the model do badly on them."]),
    md("The function below reads the `RAAMOVE_LABELS` you just defined."),
    code(embed(reshape.reid, reshape.reshape_raamove,
               imports=["import json", "from pathlib import Path"])),
    code("rows = reshape_raamove(RAW_DIR)"),
    md("## Step 4 — Check the label balance",
       "",
       "Very imbalanced: `Method` is the biggest class by far, and `Implication` has "
       "only a couple of dozen sentences."),
    inspect_cell(),
    balance_blank("with eight classes and a rare tail, the smallest class decides "
                  "everything — 7 per class is about the most this pool supports."),
    md("## Step 5 — Save it"),
    save_cell("raamove"),
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
    md("## Step 1 — Download the raw data",
       "",
       "This one is on **Mendeley Data**, which has a public API. We ask it for the "
       "dataset's file list, then download each file. The CDN refuses requests that do "
       "not look like a browser, hence the `User-Agent` header."),
    code("import json, urllib.request, pathlib",
         "",
         'RAW_DIR = pathlib.Path("cars50")',
         "RAW_DIR.mkdir(exist_ok=True)",
         "",
         "def fetch(url):",
         '    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})',
         "    return urllib.request.urlopen(request, timeout=60)",
         "",
         'meta = json.loads(fetch("https://data.mendeley.com/public-api/datasets/kwr9s5c4nk").read())',
         'for record in meta["files"]:',
         '    target = RAW_DIR / record["filename"]',
         "    if not target.exists():",
         '        target.write_bytes(fetch(record["content_details"]["download_url"]).read())',
         'print("downloaded", len(list(RAW_DIR.glob("*.xml"))), "XML files")'),
    md("## Step 2 — Look at the raw format",
       "",
       "**XML** this time. Each sentence carries a `step` code like `1b`:",
       "",
       "```xml",
       "<sentence><sentenceID/><text/><step>1b</step></sentence>",
       "```"),
    code('print(open(sorted(RAW_DIR.glob("*.xml"))[0], encoding="utf-8").read()[:900])'),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "The parsing is written for you, and it gives you **both granularities at once**:",
       "",
       "- the leading digit of `1b` is the **Move** → 3 classes;",
       "- the whole code `1b` is the **Step** → 11 classes.",
       "",
       "Sentences with no code, or a code that does not start with a move digit, are "
       "dropped either way.",
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
       "you have seen the numbers."),
    code(embed(reshape.reid, reshape.reshape_cars50,
               imports=["import xml.etree.ElementTree as ET",
                        "from pathlib import Path"])),
    code("move_rows, step_rows = reshape_cars50(RAW_DIR)",
         'print("moves:", len(move_rows), " steps:", len(step_rows))',
         "",
         "from collections import Counter",
         'print("move classes:", Counter(r["label"] for r in move_rows))',
         'print("step classes:", Counter(r["label"] for r in step_rows))'),
    blank(
        "Step 3a · Choose your granularity",
        "pick the version of the scheme your group will actually study.",
        "rows (a list) — either move_rows or step_rows",
        ["rows = move_rows    # 3 classes",
         "rows = step_rows    # 11 classes"],
        notes=["Note      : one line. Spend the time on the ARGUMENT, not the typing —",
               "            PLAN.md asks you to justify it in a sentence.",
               "Careful   : whichever you pick, the label names in your prompt and your",
               "            annotation sheet must match these exactly (\"Move 1\", or",
               "            \"1b\")."]),
    md("## Step 4 — Check the label balance"),
    inspect_cell(),
    balance_blank("if you chose steps, look at how thin the rare ones are before you "
                  "commit."),
    md("## Step 5 — Save it"),
    save_cell("cars50",
              "If you chose steps, save as cars50_step_pool.json and point config.py "
              "at that name."),
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
    md("## Step 1 — Download the raw data",
       "",
       "The annotations live on the paper's **OSF** project. We fetch one CSV directly "
       "by its OSF link."),
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
         "from collections import Counter",
         "",
         "codes = Counter()",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)',
         "    for row in reader:",
         '        for code in (row["Human_ErrorCategories"] or "").split(","):',
         "            if code.strip():",
         "                codes[code.strip()] = codes[code.strip()] + 1",
         "",
         'print(len(codes), "distinct codes:")',
         "for code, n in codes.most_common():",
         '    print("   ", code, n)'),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Two decisions, and the first is the biggest single judgment call in any of the "
       "five tracks:",
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
    blank(
        "Step 3a · Group the error codes",
        "map each raw error code to the broader category you will actually study.",
        "L2_COARSE (a dict)",
        ['L2_COARSE = {"ART": "Grammatical", "SP": "Mechanical", ...}',
         "one entry per code from step 2 that you want to keep",
         "(NO_ERROR is handled separately — do not list it)"],
        notes=["Careful   : a code you omit is not an error — it is a DROPPED sentence.",
               "            Compare your total in step 4 against the detection count.",
               "Note      : put the grouping in PLAN.md as a table, with a one-line",
               "            justification for any code a reasonable person would file",
               "            somewhere else. That table is report section 1."]),
    md("The two functions below read the `L2_COARSE` you just defined."),
    code(embed(reshape._l2_coarse_label, reshape.reid, reshape.reshape_l2_errors,
               imports=["import csv"])),
    code("category_rows, detection_rows = reshape_l2_errors(RAW_FILE)",
         'print("categories:", len(category_rows), " detection:", len(detection_rows))'),
    blank(
        "Step 3b · Choose your task",
        "the n-way categorisation, or the yes/no detection task.",
        "rows (a list) — either category_rows or detection_rows",
        ["rows = category_rows     # your categories from 3a",
         "rows = detection_rows    # Has error / No error"],
        notes=["Note      : detection is a genuinely easier task and a smaller project.",
               "            If you take it, plan an extension — benchmarking against the",
               "            published tool's own predictions is right there in the CSV."]),
    md("## Step 4 — Check the label balance",
       "",
       "Note how many sentences were dropped: compare your category count against the "
       "detection count, which keeps everything. The gap is your mixed-category "
       "sentences, and it is a direct consequence of the grouping you wrote in 3a."),
    inspect_cell(),
    balance_blank("if the gap between the two counts is large, your grouping is "
                  "cutting deeply — say so in limitations, or regroup."),
    md("## Step 5 — Save it"),
    save_cell("l2_errors",
              "For the binary version, save as l2_error_detection_pool.json and point "
              "config.py at that name."),
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
       "`score`, and upload it here (or put it in `data/raw/icnale/essays_scores.csv`).",
       "",
       "In Colab, the cell below opens a file picker."),
    code("# In Colab: uncomment to upload your essays_scores.csv",
         "# from google.colab import files; files.upload()",
         "",
         'RAW_FILE = "essays_scores.csv"'),
    md("## Step 2 — Look at the raw format",
       "",
       "The cell prints the **distribution** of the scores, not just a couple of rows. "
       "You need that before step 3: it is what tells you where cutting the scale leaves "
       "you with three usable classes rather than one big one and two nearly empty "
       "ones."),
    code("import csv",
         "",
         "scores = []",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)',
         "    for row in reader:",
         "        try:",
         '            scores.append(float(row["score"]))',
         "        except (TypeError, ValueError):",
         "            pass",
         "",
         "scores.sort()",
         'print(len(scores), "scores · min", scores[0], "· max", scores[-1])',
         "for q in (10, 25, 33, 50, 67, 75, 90):",
         '    print("   ", str(q) + "th percentile:", scores[int(len(scores) * q / 100)])'),
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
       "alphabetical. Set `LABELS_ORDER = [\"Low\", \"Mid\", \"High\"]` in `config.py`, "
       "or the weighted κ gets computed over `High < Low < Mid`, which means nothing."),
    code(embed(reshape.reid, reshape.reshape_icnale, imports=["import csv"])),
    blank(
        "Step 3a · Cut the scale",
        "turn a numeric score into three bands, and be able to defend where.",
        "rows (a list)",
        ["rows = reshape_icnale(RAW_FILE, low_below=..., mid_below=...)",
         "the defaults (4.0 / 7.0) are ROUND NUMBERS, not a rubric —",
         "using them unchanged is a choice you would have to defend too"],
        notes=["Note      : run it a couple of ways and look at step 4 each time. Seeing",
               "            the counts move as you shift a boundary is the point.",
               "Careful   : whatever you settle on goes in PLAN.md as two numbers and a",
               "            reason. Do not re-cut after seeing your F1."]),
    md("## Step 4 — Check the label balance"),
    inspect_cell(),
    balance_blank("these counts are a direct function of the two numbers you just "
                  "chose — if you do not like them, change the cuts NOW, not after "
                  "step 5."),
    md("## Step 5 — Save it",
       "",
       "⚠️ Keep this file **out of git** and **out of your submission bundle**. "
       "`.gitignore` and `scripts/make_submission.py` both exclude anything with "
       "`icnale` in the name — please leave that in place."),
    save_cell("icnale"),
    handoff("icnale"),
])

print("\nDone. To check nothing drifted from reshape.py:")
print("  python scripts/_generate_pool_notebooks.py && git diff --exit-code notebooks/")
