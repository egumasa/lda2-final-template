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


def blank(title, goal, produces, hints, notes=(), starter=None):
    """A blank cell: what to write, what it must be called, and what it is FOR.

    Never blank something without saying what the next cell expects to find - a
    student stuck on a NAME has learned nothing about annotation.

    `starter` is a skeleton to leave in the cell instead of an empty line. Use it where
    the DECISION is what goes in the gaps and the surrounding structure is just typing -
    a beginner retyping eight dict keys from scratch is being tested on dict syntax,
    which is not what any of these cells are for.
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
    if starter:
        lines.append("# ✏️ replace each ... below")
        for line in starter:
            lines.append(line)
    else:
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


def inspect_cell():
    return code(
        "from collections import Counter",
        "",
        'print("total items:", len(rows))',
        'print("label counts:", dict(Counter(item["label"] for item in rows)))',
        'print("fields per item:", list(rows[0]))',
        "",
        "# Peek at the first three. `context`, where a track has one, is trimmed: it is",
        "# the whole passage and would bury everything else in this output.",
        "for item in rows[:3]:",
        "    preview = dict(item)",
        '    if preview.get("context"):',
        '        preview["context"] = preview["context"][:70] + " …"',
        "    print(preview)")


def parser_note(heading, intro, structure, api_steps, demo=None):
    """Teach the library this track's raw format needs, before the reshaping code.

    Every track hands you a different format, and the reshaping function further down
    reads it with whatever stdlib module fits. That module is the one genuinely new
    thing on the page, so it gets named and mapped BEFORE the code that uses it -
    otherwise the function below reads as magic.

    `structure` is a bulleted map of the format; `api_steps` is a numbered list of the
    calls the function makes. `demo` (a list of source lines) adds a runnable cell -
    give it only when the track's Step 2 cell does not already exercise the library.
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
    if demo:
        cells.append(code(*demo))
    return cells


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
       "(`Intelligence.json`, `Engineering.json`). Each record has a `text`, a "
       "three-letter move code in `labels`, and an `idx` — the number of the abstract "
       "the sentence came from.",
       "",
       "**Read the codes the cell below prints** — you are about to name every one. "
       "Then look at the `idx` values: the file is one long list of sentences, but the "
       "sentences of an abstract sit together, in order. That is the only reason the "
       "abstracts can be put back together at all."),
    code('RAW_DIR = "RAAMove"',
         "",
         "import json",
         "from collections import Counter",
         "",
         'data = json.loads(open(RAW_DIR + "/Intelligence.json", encoding="utf-8").read())',
         'print("records:", len(data))',
         'print("codes:", Counter(record["labels"] for record in data))',
         'print("abstracts:", len({record["idx"] for record in data}))',
         "data[:3]"),
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
    blank(
        "Step 3a · Name the moves",
        "fill in the move name your prompt will use for each three-letter code.",
        "RAAMOVE_LABELS (a dict)",
        ['the eight codes are given — you write the eight names',
         'RAAMOVE_LABELS = {"BAC": "Background", "GAP": ..., ...}'],
        notes=["Note      : the names are not cosmetic. They are the wording your prompt",
               "            uses, your coders read on the sheet, and your confusion matrix",
               "            is labelled with. \"Gap\" and \"Establishing a niche\" name the",
               "            same category and will not get you the same predictions.",
               "Note      : eight classes is a lot. To MERGE two, give them the same name",
               "            — {\"RST\": \"Finding\", \"CLN\": \"Finding\"} makes one class of",
               "            two. Decide that HERE and say so in PLAN.md, not after you",
               "            have seen the model do badly on them."],
        starter=['RAAMOVE_LABELS = {',
                 '    "BAC": ...,      # e.g. "Background" — the wording your prompt will use',
                 '    "GAP": ...,',
                 '    "MTD": ...,',
                 '    "PUR": ...,',
                 '    "RST": ...,',
                 '    "CLN": ...,',
                 '    "CTN": ...,',
                 '    "IMP": ...,',
                 '}']),
    md("The function below reads the `RAAMOVE_LABELS` you just defined."),
    code(embed(reshape.reid, reshape.reshape_raamove,
               imports=["import json", "from pathlib import Path"])),
    code("rows = reshape_raamove(RAW_DIR)"),
    md("## Step 4 — Check the label balance",
       "",
       "Very imbalanced: `Method` is the biggest class by far, and `Implication` has "
       "only a couple of dozen sentences."),
    inspect_cell(),
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
        demo=["import xml.etree.ElementTree as ET",
              "",
              'first_file = sorted(RAW_DIR.glob("*.xml"))[0]',
              "tree = ET.parse(first_file)",
              "",
              'print("reading", first_file.name, "\\n")',
              'for i, sentence in enumerate(tree.iter("sentence")):',
              "    if i >= 3:                       # just the first three, to keep the output short",
              "        break",
              '    for tag in ("sentenceID", "text", "step"):',
              "        child = sentence.find(tag)",
              '        print(" ", tag + ":", child.text.strip() if child is not None and child.text else "MISSING")',
              '    print()']),
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
