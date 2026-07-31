#!/usr/bin/env python3
"""Generate notebooks/02_sample · 03_annotate · 04_prompt · 05_report.ipynb.

    python scripts/_generate_project_notebooks.py

Never hand-edit the .ipynb files: edit this and re-run. (Same rule as the course repo's
day notebooks - see planning/course_planning/notebook-coding-principles.md.)

THE FIVE-NOTEBOOK SPINE
-----------------------
    01 build pool  ->  02 sample  ->  03 annotate  ->  04 prompt  ->  05 report

01 is per-track and lives in _generate_pool_notebooks.py. 02-05 are track-agnostic:
everything from sampling on runs off {id, text, label}, so one copy serves all tracks.

The 02/03 boundary is the ANNOTATION PAUSE, and it is a hard boundary on purpose.
02 draws the sample and creates the sheet in one sitting - those belong together, and
splitting them only forced a reload of a file written ten minutes earlier. Then days
pass while two people annotate. 03 picks up from the saved file.

Putting the pause INSIDE one notebook was tried and is wrong: a student returning to
adjudicate hits "Run all", the sampling cells re-run, and if anything in config.yaml
moved in the meantime the sample silently stops matching the sheet they annotated.

Each notebook LOADS a file its predecessor wrote and SAVES a file its successor opens.
That is not ceremony - the group works across several days and several people's Colab
runtimes, and a variable in someone else's session is not a handoff. It also means a
group that gets stuck in 03 can still be handed a gold set and carry on in 04.

WHAT IS BLANK, AND WHY
----------------------
The plumbing is written: the setup cell, the load-the-previous-file cell, and the
save-for-the-next-notebook cell are all filled in, because a group stranded on a path
string has learned nothing and lost an afternoon.

Blank is the pipeline itself - what each step consumes, what it produces, and in what
order. Every call has the same form it had in Days 1-3, and each blank names where it
was run before. The SHAPE of the pipeline is the thing worth being able to reconstruct,
and the thing they have to narrate in the Q&A.
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import pipeline
from _setup_cell import REPO, SETUP_MD_LINES, setup_lines

OUT = Path(__file__).resolve().parent.parent / "notebooks"

NOTEBOOKS = ["01_build_pool_<track>", "02_sample", "03_annotate", "04_prompt",
             "05_report"]


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def step(number, title, goal, available, pointer, produces, extra=(), source=None,
         starter=()):
    """A step cell: the shape of the code, with the decisions left blank.

    `starter` is the skeleton the cell ships with, and no step cell goes without one.
    An empty cell asks a beginner to remember the assignment form, the exact spelling
    of a global, and the syntax around the call, all at once - and only the last of
    those three is what the step is about. So the left-hand side is always given (it
    is the name the next cell depends on, and must not be mistypeable), any syntax the
    course has not taught is typed out in full, and `____` marks the decision itself.

    `source` names the file and function the step calls, so a student who wants to
    know what happens inside it can go and read it.
    """
    rule = "═" * max(4, 62 - len(title) - len(str(number)))
    lines = [
        "# ══ STEP " + str(number) + " · " + title + " " + rule,
        "# Goal      : " + goal,
    ]
    for index, line in enumerate(available):
        lines.append(("# Available : " if index == 0 else "#             ") + line)
    if source:
        lines.append("# Source    : " + source)
    lines.append("# Pointer   : " + pointer)
    lines.append("# Produce   : " + produces + "      ← later cells use these names")
    for line in extra:
        lines.append("# " + line)
    lines.append("")
    if starter:
        lines.append("# ✏️ your code here — fill in each ____")
        lines.append("")
        for line in starter:
            lines.append(line)
    else:
        lines.append("# ✏️ your code here")
    lines.append("")
    return code(*lines)


def embed(*objects, imports=(), why=""):
    """Render the real pipeline functions into the notebook, from the live source.

    Read straight out of scripts/ with `inspect.getsource`, so what the notebook shows
    is what actually runs - the two cannot drift, exactly as in the 01 notebooks.

    The split is deliberate. Plumbing is imported: nobody learns anything from reading
    a retry loop or a path check. The steps of the STUDY - drawing a balanced sample,
    asking the model and reading its reply, scoring the result - are put in front of
    the student instead, because those are the methods they have to describe in the
    report, and a method you cannot read is one you cannot defend.
    """
    blocks = []
    if why:
        blocks.append(why)
    if imports:
        blocks.append("\n".join(imports))
    for obj in objects:
        blocks.append(inspect.getsource(obj).rstrip("\n"))
    return code(*("\n\n".join(blocks)).split("\n"))


def read_me_md(what, points):
    lines = ["### The code that does it — read it, then run it",
             "",
             what,
             "",
             "It is read straight out of `scripts/` when this notebook is generated, so "
             "it is not a simplified copy: it is the code that runs. Two things to look "
             "for as you read:",
             ""]
    for point in points:
        lines.append("- " + point)
    lines.append("")
    lines.append("Run the cell to define these, then use them in the step below.")
    return md(*lines)


def spine(current):
    """The five-notebook map, with the one you are in marked."""
    parts = []
    for name in NOTEBOOKS:
        if name.startswith(current):
            parts.append("▶ " + name)
        else:
            parts.append("  " + name)
    return "```\n" + "  →".join(parts) + "\n```"


def title_cell(number, name, title, one_line, reads, writes, body):
    """Every notebook opens the same way: where am I, what do I read, what do I write."""
    lines = [
        "# " + number + " · " + title,
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/" + REPO + "/blob/main/notebooks/"
        + name + ".ipynb)",
        "",
        one_line,
        "",
        spine(number),
        "",
        "| | |",
        "|---|---|",
        "| **Reads** | " + reads + " |",
        "| **Writes** | " + writes + " |",
        "",
        "---",
        "",
    ]
    return md(*(lines + list(body)))


SETUP_MD = SETUP_MD_LINES


def setup_cell(extra_imports=()):
    return code(*setup_lines(extra_imports))


CONFIG_MD = md(
    "> **Everything above comes from `config.yaml`** — one small file at the top of the "
    "repo, which you edit once as a group, and the only file in the plumbing you touch. "
    "That is deliberate: the seed that drew your sample has to be the seed you report, "
    "and five copies of a number in five notebooks is five chances for them to "
    "disagree. "
    "Your settings are also the filenames — `track: cars50`, `group: kimura`, `run: v1` "
    "means this notebook reads and writes `cars50_kimura_v1_...`. If the line it just "
    "printed is not your track, your group and your seed, fix `config.yaml` and re-run "
    "this cell.")


def handoff_md(what, target, next_notebook, why):
    return md(
        "## Save it — this is the handoff",
        "",
        why,
        "",
        "**Next:** open `" + next_notebook + ".ipynb`. It starts by loading `" + target
        + "`.")


# ==================================================================================
# 02 — sample
# ==================================================================================
cells = [
    title_cell(
        "02", "02_sample", "Draw your sample, and put it in front of your coders",
        "Choose which items your study is about — and be able to say why those.",
        "`data/pools/<track>_pool.json` (from 01)",
        "`data/gold/<track>_<group>_sample.json`, and a Google Sheet to annotate",
        ["**One sitting, start to finish.** You pick a sampling strategy, draw the "
         "sample, save it, and create the annotation sheet — then the notebook is done, "
         "and the week's real work happens in the sheet.",
         "",
         "Notebook 03 is where you come back to, once both coders have finished. Keeping "
         "that separate is deliberate: re-running the cells below *after* annotating "
         "would redraw the sample your sheet was built from.",
         "",
         "> Budget real time for the annotation itself. Forty items, two annotators, plus "
         "the argument afterwards. It is the most valuable thing you will make this week "
         "and the easiest to rush.",
         "",
         "### Working as a group",
         "",
         "These notebooks live in a shared Drive folder, so **all of you can edit at "
         "once** — Colab syncs edits like a Google Doc. Two things do *not* work that "
         "way:",
         "",
         "- **Runtimes are per-person.** Seeing `sampled` in a saved output does not "
         "mean `sampled` exists in *your* session. Whoever runs the cells is the "
         "**driver**.",
         "- **Files are last-write-wins.** `data/`, `prompts/` and `outputs/` are "
         "ordinary files, not Google Docs. Two of you writing the same one does not "
         "merge them — Drive keeps one and may quietly leave the other beside it as "
         "`… (1).json`, which nothing downstream will ever read. Let the driver be the "
         "only one running cells that write.",
         "",
         "The **annotation Sheet is the exception** — that is a real Google Sheet, so "
         "annotate it together, all at once."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Reading and writing files, and making the Google Sheet, are plumbing, so",
        "# they are imported. The sampling itself is a method you have to defend, so",
        "# it is not — you will read it, further down, before you call it.",
        "from pipeline import load_gold, save_json",
        "from annotate import create_annotation_sheet",
    ]),
    CONFIG_MD,
    md(
        "## What this notebook is for",
        "",
        "A **pool** is everything the corpus has, with its natural imbalance. A "
        "**sample** is the balanced subset you actually study — equal items per label, "
        "so precision, recall, F1 and the confusion matrix all stay readable. Rare "
        "labels simply yield fewer; that is a property of the data, and it belongs in "
        "your limitations.",
        "",
        "Keeping the two separate is also what leaves unused items available as few-shot "
        "examples in notebook 04, without showing the model the answers you are testing "
        "it on. That is why you sample rather than just taking the first 40 rows.",
        "",
        "> **Do not have a pool yet?** Run `01_build_pool_<track>.ipynb` first. To see "
        "this notebook work before then, point at `DEMO_POOL_PATH` in the cell below — "
        "but the demo pools are small enough that a real sample would eat most of them, "
        "and `sample_pool` will warn you when it does."),
    step(
        1, "Load the pool",
        "open what notebook 01 wrote, and see its natural imbalance.",
        ["load_gold(POOL_PATH)  ->  a list of {id, text, label}",
         "POOL_PATH · DEMO_POOL_PATH   (both come from config.yaml)"],
        "Day 2 S5 step F · Day 3 setup — the same call.",
        "pool",
        source="scripts/pipeline.py · load_gold",
        extra=["Note      : `Counter` below counts how many items carry each label. It",
               "            is typed out for you — the decision in this cell is which of",
               "            the two paths to open, not how to count."],
        starter=[
            "pool = load_gold(____)          # POOL_PATH  ·  or DEMO_POOL_PATH to try it out",
            "",
            "from collections import Counter",
            'print(len(pool), "items")',
            'print(Counter(item["label"] for item in pool))',
        ]),
    md(
        "## The decision this notebook turns on: **how do you draw the sample?**",
        "",
        "Forty items out of a few thousand. *Which* forty is not a technicality — it "
        "decides what your F1 is a statement about, and it is the first thing a reader "
        "of your report is entitled to ask. Three defensible answers, and they disagree "
        "with each other:",
        "",
        "| Strategy | What you get | What you can claim | What it costs |",
        "|---|---|---|---|",
        "| **Balanced by label** | equal items per label | clean per-class precision and "
        "recall; a readable confusion matrix | your sample no longer looks like the "
        "corpus — you have over-sampled the rare moves on purpose |",
        "| **Simple random** | the corpus as it is | \"this is how the model does on "
        "text like this\" | a rare label may arrive with two items, or none, and F1 on a "
        "class of two means very little |",
        "| **By document** | whole passages, sampled | \"forty sentences from forty "
        "abstracts\" rather than from three | labels are not controlled at all; only "
        "`cars50` and `raamove` can do it |",
        "",
        "### The one that is easy to get wrong",
        "",
        "Forty sentences drawn from three introductions and forty drawn from forty are "
        "both \"n = 40\", and they support very different claims. Three long "
        "introductions is *three* texts, three authors, three topics — and neighbouring "
        "sentences in one passage are not independent observations of anything. If you "
        "sample balanced or at random on `cars50` or `raamove`, print how many distinct "
        "`doc_id`s you ended up with before you write \"we sampled 40 sentences\".",
        "",
        "### And the ceiling on size",
        "",
        "Whatever you choose, the smallest class is a hard ceiling on a balanced draw: "
        "ask for more than a class has and you get everything it has, and your sample is "
        "quietly no longer balanced. The other ceiling is time — every item is one API "
        "call in notebook 04, times the number of rounds, and one row two of you must "
        "annotate by hand below. **Around 40 items total** is the size this project is "
        "built for. Set `n_per_class` in `config.yaml` and re-run the setup cell.",
        "",
        "Whichever you pick: say so in `PLAN.md`, and do not change it after you have "
        "seen the numbers."),
    read_me_md(
        "All three strategies, read out of `scripts/pipeline.py`. This is your "
        "**sampling method** — report section 1 has to describe it and the Q&A may well "
        "ask you to defend it, so it is in front of you rather than behind an import.",
        ["**Where the balance comes from** in `sample_pool`: step 1 sorts the pool into "
         "one bucket per label, step 2 takes up to `n_per_class` from each. A label with "
         "fewer than that gives all it has — which is exactly why a rare class comes "
         "back short, and why that is data rather than a bug.",
         "**Where the reproducibility comes from**, in all three: `random.Random(seed)` "
         "— one generator, made from your seed, doing every shuffle. Same seed, same "
         "draw, on anyone's machine. That one line is the whole of why your sample is "
         "checkable by someone else.",
         "**What `sample_by_document` refuses to do.** Look at the check it opens with. "
         "On a track whose items are loose sentences there are no documents to stratify "
         "by, so it stops and says so rather than inventing an answer."]),
    embed(pipeline.reid, pipeline._report_draw, pipeline.sample_pool,
          pipeline.sample_random, pipeline.sample_by_document, pipeline.label_set,
          imports=["import random"],
          why="# Read straight out of scripts/pipeline.py — this IS the code that runs."),
    step(
        2, "Draw your sample — and say why that way",
        "commit to a sampling strategy, and to the sentence that defends it.",
        ["sample_pool(pool, N_PER_CLASS, SEED)              balanced across labels",
         "sample_random(pool, n_total, SEED)                the corpus as it is",
         "sample_by_document(pool, n_docs, n_per_doc, SEED) cars50 · raamove only",
         "label_set(items)  ->  the sorted list of labels present"],
        "Day 4 Part A — sample_pool is the one you ran there.",
        "sampled · LABELS",
        source="the cell just above · scripts/pipeline.py",
        extra=["Careful   : pass SEED. A sample nobody can redraw is a sample nobody can",
               "            check, and your report has to state the number.",
               "Note      : each strategy PRINTS its per-label counts. Run more than one",
               "            and look at the difference — that difference is the argument.",
               "Note      : if it WARNS that you took most of the pool, you are almost",
               "            certainly still pointed at DEMO_POOL_PATH."],
        starter=[
            "# ── Delete the two you are NOT using. ────────────────────────────────",
            "sampled = sample_pool(pool, N_PER_CLASS, SEED)",
            "sampled = sample_random(pool, ____, SEED)                  # how many in total?",
            "sampled = sample_by_document(pool, ____, ____, SEED)       # how many docs? how many from each?",
            "",
            "LABELS = label_set(sampled)",
            'print("labels:", LABELS)',
            "",
            "# ── Why this one, in a sentence. It is report section 1, and PLAN.md ──",
            "# ── asks for it. Write it now, while the reason is still in your head. ─",
            "#",
            "#   >>> We sampled ____ because ____",
            "#",
        ]),
    md(
        "### Sanity-check what you drew",
        "",
        "Three questions worth answering before you commit forty hand-annotations to it:",
        "",
        "1. **Did you get what you asked for?** Compare the counts against your "
        "intention. A short class is your rare label hitting its ceiling.",
        "2. **How many distinct texts is this?** On `cars50` and `raamove` — see above. "
        "Forty sentences from four documents is a much narrower claim.",
        "3. **Is there pool left over?** `build_fewshot` in notebook 04 draws its "
        "examples from items you did *not* sample. If the sample is most of the pool, "
        "there is nothing uncontaminated left to draw from."),
    step(
        3, "Check the draw",
        "confirm the counts, the spread across documents, and the leftover pool.",
        ["len(pool) · len(sampled)  ·  Counter(item[\"label\"] for item in sampled)"],
        "Day 4 Part A — the same counts you printed there.",
        "nothing to name — this is a check, not a stage",
        extra=["Ask       : does the shortfall on any label match the smallest class in",
               "            the pool counts from step 1?"],
        starter=[
            "from collections import Counter",
            "",
            'print("pool:", len(pool), "· sampled:", len(sampled),',
            '      "· left over:", len(pool) - len(sampled))',
            'print(Counter(item["label"] for item in ____))      # sampled',
            "",
            "# On cars50 / raamove only — how many distinct texts did your 40 come from?",
            'if "doc_id" in sampled[0]:',
            '    docs = {item["doc_id"] for item in sampled}',
            '    print("drawn from", len(docs), "distinct documents")',
        ]),
    md(
        "## Save the sample — it is what notebook 03 comes back to",
        "",
        "The cell below writes the sample to a file. That is not ceremony even though "
        "the next cell uses it directly: notebook 03 runs **days later**, in a fresh "
        "Colab runtime, possibly for a different member of your group, and `sampled` "
        "will not exist in it. Adjudication needs these exact items back, to re-attach "
        "what the sheet does not carry.",
        "",
        "Even with a fixed seed, save it — a seed reproduces a draw only as long as "
        "nobody edits the pool underneath it."),
    code(
        "save_json(sampled, SAMPLE_PATH, what=\"sampled items\")",
        "",
        "# It still carries the PUBLISHED label at this point. The sheet below",
        "# deliberately does not copy that in — you annotate blind — but notebook 03",
        "# uses it at the very end, to show you where your group disagreed with the",
        "# corpus. That comparison is one of the more interesting things in your report.",
        ""),
    md(
        "## Step 4 — Create the annotation sheet",
        "",
        "A real Google Sheet in your own Drive, one row per item, with blank `CoderA`, "
        "`CoderB`, `Final` and `Note` columns. All of you can have it open at once.",
        "",
        "The published label is **deliberately not copied in**. Two people annotate "
        "independently, without seeing each other's column or the corpus's answer — "
        "that is what makes the agreement number mean anything. Decide who is CoderA "
        "and who is CoderB before you start, and do not look across.",
        "",
        "**On `cars50` and `raamove`** the sheet gets one extra column, `Context`: the "
        "passage each sentence came from, with the sentence you are labelling marked "
        "`>>>`. Read it. A move is a rhetorical function *within* a passage, and if the "
        "sentence alone is too thin for the model to judge, it is just as thin for the "
        "two of you — and your labels are the answer key everything else gets measured "
        "against.",
        "",
        "The first time you run this, Colab asks for permission to use your Google "
        "account. That is `gspread` authorising against your own Drive.",
        "",
        "The sheet is created in the Drive of whoever runs the cell, so pass "
        "`share_with=MEMBERS` — the Google accounts you put in `config.yaml` — or your "
        "second coder will open the link and be told they need access. Pass "
        "`remember=SHEET_PATH` too, and the link is written to a file instead of living "
        "in this cell's output, where a runtime reset can lose it."),
    step(
        4, "Create the sheet",
        "make a blind annotation sheet, shared with your group, one row per item.",
        ["create_annotation_sheet(title, items, labels,",
         "                        share_with=..., remember=...)  ->  the sheet URL",
         "MEMBERS · SHEET_PATH   (from config.yaml)"],
        "Day 2 S5 step A — the same call, plus the two sharing arguments.",
        "a sheet URL — and a note of it on disk, for the step after next",
        source="scripts/annotate.py · create_annotation_sheet",
        extra=["Note      : the title is built for you from your track, group and run.",
               "            You will have several of these by the end of the week.",
               "Careful   : run this ONCE. Running it again makes a SECOND sheet, and",
               "            half your annotations end up in the one nobody read back."],
        starter=[
            'title = TRACK + " · " + GROUP + " · " + RUN + " annotation"',
            "",
            "url = create_annotation_sheet(title, sampled, LABELS,",
            "                              share_with=____,      # MEMBERS",
            "                              remember=____)        # SHEET_PATH",
            "print(url)",
        ]),
    md(
        "---",
        "",
        "## 🛑 This notebook is finished. Now go and annotate.",
        "",
        "Open the sheet, and label every row. Rules of the exercise:",
        "",
        "- **Each coder works in their OWN tab** — `CoderA`, `CoderB`. Different "
        "people, no discussion, and do not open a colleague's tab to see what they "
        "put. The sheet cannot stop you; the κ you report is only worth something if "
        "you do not.",
        "- **Leave the `Final` tab alone** until everyone has finished and you have "
        "talked.",
        "- **A third coder?** Duplicate an EMPTY tab (right-click ▸ *Duplicate*) and "
        "rename it `CoderC`. Copying a tab somebody has already filled in gives them "
        "that person's answers, and notebook 03 will tell you so in a warning you "
        "will not enjoy.",
        "- **Use `Note`** when you hesitate. The item you were unsure about is the item "
        "you will want to quote in your error analysis, and you will not remember which "
        "one it was.",
        "- Labels must be spelled exactly as `LABELS` prints them. `to_canonical` will "
        "tell you about typos, but it is quicker not to make them.",
        "",
        "This is where the week's actual work happens, and it takes days rather than "
        "minutes.",
        "",
        "> **Do not run this notebook again.** When you come back, open "
        "`03_annotate.ipynb` — it picks up from the file you just saved. Re-running the "
        "cells above would draw the sample *again*, and if anyone has touched "
        "`config.yaml` or the strategy line in the meantime, you would end up with a "
        "sample that no longer matches the sheet two people have been annotating. "
        "That is the kind of mistake you find out about in the Q&A.",
        "",
        "**Next:** `03_annotate.ipynb`, once both columns are full."),
]

# ==================================================================================
# 03 — agreement, adjudication, gold
# ==================================================================================
cells_03 = [
    title_cell(
        "03", "03_annotate", "Agreement, adjudication → *your* gold set",
        "The part no model can do for you, and the part the Q&A will ask about.",
        "`data/gold/<track>_<group>_sample.json` and the sheet (both from 02)",
        "`data/gold/<track>_<group>_gold.json`",
        ["**Come here when both coders have finished.** Notebook 02 drew the sample and "
         "made the sheet; this one turns two people's labels into one gold set.",
         "",
         "The published labels are somebody else's judgment. You re-annotated the sample "
         "blind; now you find out how far apart the two of you were, and argue out the "
         "rows you disagreed on.",
         "",
         "What comes out is *your* gold set — and the disagreements tell you which label "
         "boundaries are genuinely fuzzy. That is what lets you say, later, whether a "
         "model's miss is the **model's** fault or the **scheme's**. Nothing else in the "
         "project can tell you that, and notebook 05 asks you for it directly."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# The Google Sheets round trip is plumbing, so it is imported. The judgment it",
        "# exists to support is not in any of these files.",
        "from pipeline import load_gold, label_set, save_json",
        "from annotate import (remembered_sheet, load_coder_sheets, to_canonical,",
        "                      annotator_agreement, disagreements,",
        "                      compare_to_published)",
    ]),
    CONFIG_MD,
    md(
        "## First — your sample, back from the file",
        "",
        "Notebook 02 saved it, and this is the moment that was for. Days have passed, "
        "the runtime that drew the sample is long gone, and the person running this cell "
        "may not be the person who ran 02.",
        "",
        "It matters that this is a **load and not a redraw**: the sheet your coders "
        "filled in was built from these exact forty items, and adjudication puts their "
        "labels back onto them one by one. `to_canonical` also uses this list to restore "
        "what the sheet does not carry — on `cars50` and `raamove`, the passage each "
        "sentence came from."),
    code(
        "sampled = load_gold(SAMPLE_PATH)",
        "LABELS = label_set(sampled)",
        "",
        'print(len(sampled), "items ·", LABELS)',
        "",
        "# These still carry the PUBLISHED label. Ignore it for now — you compare",
        "# against it in step 4, once your own labels are settled.",
        ""),
    code(
        "# The sheet notebook 02 created, read back from the file it wrote. This is why "
        "it",
        "# wrote one: whoever runs this notebook need not be the person who ran",
        "# and need not still have that cell's output on screen.",
        "SHEET_ID = remembered_sheet(SHEET_PATH)",
        "",
        "# Working on a sheet someone made before this file existed? Paste its URL (or",
        "# just the long id from it) here instead:",
        '# SHEET_ID = ""',
        "",
        'print("sheet:", SHEET_ID or "-- none saved yet: run notebook 02 --")',
        ""),
    md(
        "## Step 1 — Measure agreement",
        "",
        "Each coder has their **own tab**, so the first thing to do is line them up "
        "side by side. `load_coder_sheets` reads one tab per name you give it and joins "
        "them by item id into a single table — one column per coder, plus `Final`.",
        "",
        "> **Who annotated?** Change the list if your group is not two people. If a "
        "third coder joined, duplicate an **empty** tab in the sheet (right-click ▸ "
        "*Duplicate*), rename it `CoderC`, and add `\"CoderC\"` to the list. You did not "
        "have to decide this when the sheet was made, and you do not have to rebuild "
        "anything now.",
        "",
        "Then the numbers. With **two** coders: raw percent agreement, Cohen's κ "
        "(agreement corrected for what you would get by guessing), and a "
        "coder-vs-coder confusion matrix whose off-diagonal cells show *which* label "
        "pairs you confuse. With **three or more**: Fleiss' κ for the group as a whole, "
        "then Cohen's κ for every pair, then the matrix for the pair that agreed least — "
        "which is usually where your scheme is leaking.",
        "",
        "**Write these down now** — they are report section 1, and they do not survive "
        "a runtime reset. A κ around .8 is strong; around .4 means the scheme, not the "
        "annotators, is doing something wrong. Either is a reportable finding. A low κ "
        "you can explain beats a high one you cannot."),
    step(
        1, "Measure agreement",
        "how often did you agree, and on which labels did you not?",
        ["load_coder_sheets(SHEET_ID, CODERS)  ->  rows   (one column per coder)",
         "annotator_agreement(rows, coders=CODERS)  ·  disagreements(rows, coders=CODERS)",
         "SHEET_ID   (set in the cell just above)"],
        "Day 2 S5 steps D–E — the same calls, now naming who annotated.",
        "CODERS · rows · disagreed",
        source="scripts/annotate.py · load_coder_sheets, annotator_agreement",
        extra=["Note      : run this once EVERY coder's tab is filled in. Rows that not",
               "            everyone labelled are dropped from the comparison.",
               "Careful   : if it warns that two coders gave identical labels to every",
               "            item, somebody duplicated a tab that was already filled in.",
               "            That agreement is a copy, not a measurement — fix it before",
               "            you report anything.",
               "Keep      : `disagreed` matters again in notebook 05 — the rows you",
               "            argued about are the ones to check your model's errors",
               "            against. Save the list, or write the ids down."],
        starter=[
            "# Who annotated. One name per tab in the sheet.",
            'CODERS = ["CoderA", "CoderB"]        # add "CoderC" if a third person joined',
            "",
            "rows = load_coder_sheets(SHEET_ID, ____)         # CODERS",
            "",
            "annotator_agreement(rows, coders=CODERS)         # prints agreement and κ",
            "disagreed = disagreements(rows, coders=CODERS)",
            "disagreed",
        ]),
    md(
        "## Step 2 — Adjudicate",
        "",
        "Go back to the sheet and fill in `Final` for **every** row:",
        "",
        "- Where you agreed, `Final` is that label.",
        "- Where you did not, talk it out and decide. If you cannot agree, the scheme is "
        "underspecified — write down *why* in `Note` and pick one. That note is worth "
        "more to your report than the label is.",
        "",
        "Then re-read the sheet and canonicalise it. `to_canonical` reports blanks and "
        "invalid labels rather than silently dropping them; fix them in the sheet and "
        "re-run until it says **0 blank, 0 invalid**."),
    step(
        2, "Adjudicate, then canonicalise",
        "agree a Final label for every row, then turn the sheet into gold.",
        ["load_coder_sheets(SHEET_ID, CODERS)  ->  rows   (re-read after editing)",
         "to_canonical(rows, LABELS, source=sampled)  ->  gold"],
        "Day 2 S5 step F — identical calls.",
        "gold",
        source="scripts/annotate.py · to_canonical",
        starter=[
            "# Re-read: `rows` from step 1 was fetched before you filled in Final.",
            "rows = load_coder_sheets(SHEET_ID, CODERS)",
            "",
            "gold = to_canonical(rows, LABELS, source=____)      # sampled",
        ],
        extra=["Careful   : re-read the sheet first. `rows` from step 1 is a snapshot",
               "            from before you filled in Final.",
               "Note      : keep going until it prints 0 blank and 0 invalid. A blank",
               "            row is an item silently missing from your study.",
               "Note      : pass source=sampled. Gold is rebuilt from the SHEET, which",
               "            holds only the id, the text and your label — anything else",
               "            the item carried (on cars50/raamove, its passage) is put",
               "            back from `sampled` by id. Harmless on the other tracks."]),
    md(
        "## Step 3 — Where do you differ from the published labels?",
        "",
        "Now — and only now, with your own labels settled — look at what the corpus "
        "said. `compare_to_published` matches by text and shows you every row where "
        "your group landed somewhere else.",
        "",
        "**Disagreement here is not an error.** You annotated forty items carefully "
        "against a scheme you had thought about; the original annotators worked at "
        "scale under different guidelines. Where you differ, one of three things is "
        "true, and saying which is exactly the analytical work this project is for:",
        "",
        "1. **Your scheme drifted** from theirs — you read a category boundary "
        "differently. Say where.",
        "2. **The item is genuinely ambiguous** — it would split any pair of annotators.",
        "3. **One of you is wrong.** It happens, in both directions.",
        "",
        "This table is report section 1, and it is the one that most often produces a "
        "sentence worth saying out loud in the Q&A."),
    step(
        3, "Compare against the published labels",
        "see where your gold set and the corpus disagree, and work out why.",
        ["compare_to_published(gold, sampled)  ->  a table of the rows that differ"],
        "Day 2 S5 step F — the same call.",
        "differences",
        source="scripts/annotate.py · compare_to_published",
        starter=[
            "differences = compare_to_published(gold, ____)      # sampled — not pool",
            "differences",
        ],
        extra=["Careful   : compare against `sampled`, not `pool`. Sampling renumbers",
               "            the ids, so `pool` would line your item 7 up against a",
               "            different sentence entirely.",
               "Note      : pick two or three and write down which of the three cases",
               "            above they are. Do it now, while you remember the argument."]),
    handoff_md(
        "gold set", "data/gold/<track>_<group>_gold.json", "04_prompt",
        "This file is the single most valuable thing your group makes all week — hours "
        "of judgment, and the only thing in the project that could not have been "
        "produced by a script. Every number in notebooks 04 and 05 is measured against "
        "it, and it goes in your submission bundle."),
    code(
        "save_json(gold, GOLD_PATH, what=\"gold items\")",
        "",
        "# It is git-ignored — it is your work, not part of the template. If you cloned",
        "# into Google Drive it is already saved across sessions; if not, download it.",
        ""),
    md(
        "---",
        "",
        "## 🛑 The `PLAN.md` gate",
        "",
        "Notebook 04 starts calling the model. **Do not open it until your `PLAN.md` "
        "has been read and signed off.** It takes two minutes and it is not busywork: a "
        "mismatched label set or an unstated sampling seed costs an hour to unpick "
        "*after* you have burned quota on it.",
        "",
        "Check, out loud, that these three agree: the label set in `PLAN.md`, the labels "
        "`label_set` actually returned above, and the labels your prompt file names. And "
        "that `PLAN.md` records **which sampling strategy you chose, and why**."),
]

# ==================================================================================
# 04 — prompt
# ==================================================================================
cells_04 = [
    title_cell(
        "03", "03_prompt", "Baseline, read the errors, iterate, freeze",
        "Write the plainest prompt that could work, then improve it for reasons you can "
        "state.",
        "`data/gold/<track>_<group>_gold.json` (from 02) · the pool (from 01)",
        "`outputs/<track>_<group>_predictions.json` · `..._rounds.json`",
        ["Everything from here on is measured against **your** gold set, not the "
         "corpus's labels. That is the point of the last two notebooks.",
         "",
         "> **Free-tier pacing.** The backend waits a few seconds between calls and "
         "retries on rate-limit errors, so a full run takes minutes and may print "
         "`(rate limited - waiting Ns then retrying)`. That is normal. Keep "
         "`N_PER_CLASS` small (2) while you iterate — then do **one** final run at full "
         "size."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Files in, files out, and the connection to the model: all plumbing, all",
        "# imported. Asking the model and scoring the answers is what this notebook",
        "# is FOR, so that code is in the notebook, two cells below.",
        "from pipeline import (load_gold, load_prompt, save_json,",
        "                      save_predictions, load_predictions, setup)",
        "",
        "setup()                     # connect to the model and say which backend we got",
    ]),
    md(
        "> **Check the backend line it just printed.** You want:",
        ">",
        "> ```",
        "> LLM backend: Gemini API (gemini-3.1-flash-lite, temperature=0, seed=42)",
        "> ```",
        ">",
        "> If it says **Colab Gemini** instead, no API key was found — put yours in the "
        "Colab Secrets panel (the 🔑 icon in the left sidebar) as `GEMINI_API_KEY` and "
        "re-run. The keyless backend has no temperature or seed, so the same prompt can "
        "give different answers and your numbers will not be reproducible. It must not "
        "be your final run."),
    CONFIG_MD,
    step(
        1, "Load your gold set and your pool",
        "the answers you score against, and the spare items few-shot draws from.",
        ["load_gold(GOLD_PATH)  ·  load_gold(POOL_PATH)",
         "label_set(items)  ->  the sorted list of labels present",
         "GOLD_PATH · POOL_PATH   (from config.yaml)"],
        "Day 3 setup — the same call, twice.",
        "gold · pool · LABELS",
        source="scripts/pipeline.py · load_gold",
        extra=["Note      : LABELS comes from your GOLD set, not the pool. If a label",
               "            never survived adjudication, it is not in your study."],
        starter=[
            "gold = load_gold(____)          # GOLD_PATH — your own labels, from notebook 03",
            "pool = load_gold(____)          # POOL_PATH — the spares few-shot draws from",
            "LABELS = label_set(____)        # gold, not pool",
            "",
            'print(len(gold), "gold ·", len(pool), "pool ·", LABELS)',
        ]),
    read_me_md(
        "These three functions are the measurement itself. `run_prompt` is the loop that "
        "asks the model once per item; `extract_label` is the guess about what its reply "
        "*meant*; `build_fewshot` is how examples get in front of it. Every number in "
        "your report comes out of these, so read them before you trust them.",
        ["**`extract_label` is doing more than it looks.** The model answers in prose; "
         "something has to decide that *\"This looks like Move 2 to me\"* is `Move 2`. It "
         "searches for label names, keeps the longest match, and falls back to `\"??\"`. "
         "Every `??` in your run is a reply this function could not read — and if there "
         "are many, that is a finding about your prompt, not a bug.",
         "**`build_fewshot` skips anything in your gold set**, matching by text rather "
         "than by id (sampling renumbered the ids). Without that, you would be testing "
         "the model on answers you had just shown it."]),
    embed(pipeline.extract_label, pipeline.run_prompt, pipeline.build_fewshot,
          imports=["import random, re",
                   "",
                   "# The two helpers these lean on: `label_set` you have met, and",
                   "# `_default_backend` is the connection `setup()` opened above.",
                   "from pipeline import label_set, _default_backend"],
          why="# Read straight out of scripts/pipeline.py — this IS the code that runs."),
    read_me_md(
        "And this is the scoring. `evaluate` prints per-class precision/recall/F1, "
        "Cohen's κ, and the confusion matrix — and **returns the macro-F1 as a number**, "
        "which is what lets you collect one per round. `show_errors` gives you the items "
        "it got wrong, and you will use it in **every round**, not just at the end.",
        ["**Macro-F1** is the plain average of the per-class F1 scores — every class "
         "counts the same, however rare. That is why a balanced sample and a macro "
         "average go together.",
         "**`ordered=True` adds a *weighted* κ**, which counts a near miss (Low→Mid) as "
         "a smaller error than a far one (Low→High). Use it only if your labels sit on "
         "a scale, and pass `labels=LABELS_ORDER` so it knows what that scale is.",
         "**`show_errors` is the one you will actually iterate on.** F1 tells you "
         "*whether* a round helped; only the errors tell you *what to change next*."]),
    embed(metrics.evaluate, metrics.show_errors,
          imports=["import pandas as pd",
                   "from sklearn.metrics import (classification_report, "
                   "confusion_matrix,",
                   "                             cohen_kappa_score, f1_score)",
                   "from pipeline import plot_confusion_matrix"],
          why="# Read straight out of scripts/metrics.py — this IS the code that runs."),
    md(
        "## Step 2 — The baseline (round 0)",
        "",
        "A number to beat. Write the plainest prompt that states the task and the label "
        "set, run it, score it. **Resist the urge to make it good** — the point of a "
        "baseline is that later rounds have something to be measured against, and a "
        "baseline you already tuned tells you nothing about whether tuning helped.",
        "",
        "Your prompt lives in `prompts/<track>.txt` and must contain `{text}`, where "
        "each item gets slotted in. Edit the **file**, not a string in this notebook — "
        "that is what makes each version savable and comparable, and it is the "
        "reproducibility habit from S10.",
        "",
        "In Colab you can write the file straight from a cell:",
        "",
        "```python",
        "%%writefile ../prompts/raamove_v0.txt",
        "Classify the rhetorical move of the sentence. Answer with the move name only.",
        "...",
        "",
        "Sentence: {text}",
        "```"),
    step(
        2, "Baseline prompt (round 0) — and read what it got wrong",
        "get one honest number to beat, and the errors that tell you what to fix.",
        ["load_prompt(PROMPT_FILE)  ->  PROMPT",
         "run_prompt(PROMPT, gold)  ->  predictions",
         "evaluate(gold, predictions, ordered=..., labels=LABELS_ORDER)  ->  macro-F1",
         "show_errors(gold, predictions)  ->  the items it got wrong",
         "PROMPT_FILE   (config.yaml: prompts/<track>.txt)"],
        "Day 3 Part A — the same two lines, plus the error table.",
        "f1_by_round · pred0",
        source="the two cells above · scripts/pipeline.py · scripts/metrics.py",
        extra=["Note      : `f1_by_round` is your prompt-iteration table — one entry per",
               "            round, keyed by the round's name. Notebook 04 reads it back",
               "            from a file and prints it into the report, so the keys are",
               "            what your reader sees: name them so they mean something.",
               "Note      : do not skip the last line. The errors are the ONLY thing that",
               "            tells you what to change; F1 only tells you afterwards",
               "            whether the change worked.",
               "Careful   : keep N_PER_CLASS small for this. Full size is minutes of",
               "            pure waiting per round on the free tier."],
        starter=[
            "# One entry per round from here on. Notebook 04 turns it into your table.",
            "f1_by_round = {}",
            "",
            "PROMPT = load_prompt(____)               # PROMPT_FILE",
            "print(PROMPT)",
            "",
            "pred0 = run_prompt(PROMPT, gold)",
            'f1_by_round["round0 baseline"] = evaluate(gold, pred0,',
            "                                          ordered=____,   # True only if your"
            " labels are a SCALE",
            "                                          labels=LABELS_ORDER)",
            "",
            "# Now look at what it actually got wrong. This is the cell that decides",
            "# your next round.",
            "show_errors(gold, pred0)",
        ]),
    md(
        "## Step 3 — Iterate, driven by the errors",
        "",
        "Two or three more rounds. The loop is always the same, and the middle step is "
        "the one that matters:",
        "",
        "```",
        "run  →  score  →  READ THE ERRORS  →  change ONE thing  →  run again",
        "```",
        "",
        "Before you touch the prompt, look at the error table from the round you just "
        "ran and ask **what these misses have in common**. There are only a few answers, "
        "and each points somewhere different:",
        "",
        "| What you see in the errors | What it suggests |",
        "|---|---|",
        "| One class swallows everything | the model has not understood that class's "
        "boundary — define it, or show an example of it |",
        "| Two labels traded in both directions | the *distinction* is unclear, to the "
        "model and possibly to your coders too |",
        "| Lots of `??` | the model is not answering in the format you asked for — fix "
        "the instruction, not the definitions |",
        "| Errors scattered with no pattern | you may be at the ceiling of what the "
        "prompt can do; consider whether the items are simply hard |",
        "",
        "Then change **one** thing, and say beforehand what you expect it to do. "
        "\"Added examples\" is not a reason; *\"Move 2 and Move 3 traded in both "
        "directions, so I gave it one example of each\"* is. Write it down as you go — "
        "reconstructing it afterwards from a stack of F1 numbers is much harder than it "
        "sounds, and it is report section 2.",
        "",
        "A round that made things **worse** is a result, not a mistake. Keep it in the "
        "table. It is often the most informative row you have.",
        "",
        "`build_fewshot` draws examples from the pool while avoiding anything in your "
        "gold set — otherwise you would be testing the model on answers you had just "
        "shown it."),
    step(
        3, "Iterate — 2–3 rounds, each one justified by the errors",
        "change one thing per round, for a reason you can point at in the error table.",
        ["build_fewshot(PROMPT, pool, gold)  ->  a new prompt with examples",
         "run_prompt(...)  ·  evaluate(...)  ·  show_errors(...)   (as in step 2)"],
        "Day 3 iterations 1–2. build_fewshot replaces typing the examples by hand.",
        "PROMPT (your best one) · f1_by_round",
        source="the cells above · scripts/pipeline.py · build_fewshot",
        extra=["Note      : save each version as its own prompt file (v0, v1, v2). A",
               "            prompt you overwrote is a round you cannot report.",
               "Ask       : did the confusion matrix change SHAPE, or did everything",
               "            shift a little? Those need different next moves."],
        starter=[
            "# ONE round. Copy this whole block for round 2, and again for round 3.",
            "",
            "# ── What did you see in the LAST round's errors, and what are you",
            "#    changing because of it? One line each, before you run anything. ──",
            "#",
            "#   I saw   >>> ",
            "#   So I    >>> ",
            "#   I expect>>> ",
            "",
            "# Either add examples to the prompt you have …",
            "PROMPT = build_fewshot(PROMPT, pool, gold)",
            "# … or write a new prompt file (see the %%writefile example above) and",
            "# load that instead:",
            '# PROMPT = load_prompt("../prompts/____.txt")',
            "",
            "pred1 = run_prompt(PROMPT, gold)",
            'f1_by_round["round1 ____"] = evaluate(gold, pred1,      # name the CHANGE',
            "                                      ordered=____, labels=LABELS_ORDER)",
            "",
            "# Read these before you design round 2.",
            "show_errors(gold, pred1)",
        ]),
    md(
        "## Step 4 — Freeze",
        "",
        "A hosted model is only *best-effort* reproducible, even at `temperature=0`. So "
        "once your best prompt is settled:",
        "",
        "1. Raise `n_per_class` in `config.yaml` to full size — and re-run notebooks 02 and "
        "03 if that changes your sample. (If it does, you have more annotating to do. "
        "This is why you decide the size **before** you annotate.)",
        "2. Run the model **once**, on your best prompt.",
        "3. `save_json` the predictions to a file.",
        "",
        "Every number you report from here on comes out of that file. That is what makes "
        "your F1 hold still, and what lets anyone else re-run your analysis on exactly "
        "the outputs you saw.",
        "",
        "**One person runs this.** It is the run you will be defending."),
    step(
        4, "The final run, frozen to a file",
        "one full-size run on your best prompt, saved so the numbers stop moving.",
        ["run_prompt(PROMPT, gold)  ->  predictions",
         "save_predictions(predictions, PRED_PATH)  ·  load_predictions(PRED_PATH)",
         "save_json(f1_by_round, ROUNDS_PATH, what=\"rounds\")"],
        "Day 2 S6 loaded a frozen file we made; now you make your own.",
        "pred_final",
        source="scripts/pipeline.py · save_predictions, load_predictions, save_json",
        starter=[
            "predictions = run_prompt(____, gold)          # PROMPT — your best one",
            "save_predictions(predictions, PRED_PATH)",
            "",
            "# Report from the FILE, not from the variable: this is the check that the",
            "# file you will quote is the file you think it is.",
            "pred_final = load_predictions(PRED_PATH)",
            "",
            "# The rounds table exists only in this session's memory until now.",
            'save_json(f1_by_round, ROUNDS_PATH, what="rounds")',
        ],
        extra=["Freeze    : save, then load it straight back and use THAT from now on.",
               "            Reading it back is not superstition — it is the check that",
               "            the file you will report from is the file you think it is.",
               "Careful   : save f1_by_round too. Notebook 04 needs it, and it is the",
               "            one thing here that exists only in this session's memory."]),
    md(
        "---",
        "",
        "**Next:** open `05_report.ipynb`. It loads the two files you just wrote and "
        "nothing else — so from here on, your numbers cannot move."),
]

# ==================================================================================
# 05 — report
# ==================================================================================
cells_05 = [
    title_cell(
        "04", "04_report", "Error analysis and export",
        "Show an item the model got wrong, and say whose fault it was.",
        "the gold set (02) · the frozen predictions and rounds (03)",
        "`outputs/` — the predictions CSV, the report scaffold, a copy of your gold set",
        ["This is the highest-value part of the whole project, and the one the Q&A will "
         "definitely go to. A low F1 with a clear account of *why* is worth more than a "
         "high one without."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Loading files is plumbing. The scoring, the error table and the triage are",
        "# the analysis this notebook is about, so they are in the notebook, below.",
        "from pipeline import (load_gold, load_predictions, load_json, save_json,",
        "                      export_results)",
        "from annotate import remembered_sheet, load_coder_sheets, disagreements",
    ]),
    CONFIG_MD,
    step(
        1, "Load the frozen run",
        "the gold set, the predictions file, and the per-round table.",
        ["load_gold(GOLD_PATH)  ->  gold",
         "load_predictions(PRED_PATH)  ->  pred_final",
         "load_json(ROUNDS_PATH, what=\"rounds\")  ->  f1_by_round"],
        "Day 2 S6 — loading a frozen predictions file is exactly what you did there.",
        "gold · pred_final · f1_by_round",
        source="scripts/pipeline.py · load_gold, load_predictions, load_json",
        extra=["Note      : nothing in this notebook calls the model. If a number here",
               "            differs from notebook 04, you are loading a different file —",
               "            not watching the model change its mind."],
        starter=[
            "gold = load_gold(____)                 # GOLD_PATH",
            "pred_final = load_predictions(____)    # PRED_PATH",
            'f1_by_round = load_json(____, what="rounds")      # ROUNDS_PATH',
            "",
            'print(len(gold), "gold ·", len(pred_final), "predictions")',
            "f1_by_round",
        ]),
    read_me_md(
        "The functions the rest of this notebook is made of. `evaluate` and "
        "`show_errors` are the same ones you ran in 03. `errors_on_disagreed` and "
        "`triage_counts` are new, and they are what turns a list of mistakes into an "
        "argument.",
        ["**`show_errors` keeps only the rows where gold and prediction differ**, and "
         "returns them as a `DataFrame` — a table, which is why Colab draws it nicely "
         "and why you can filter it with `errors[errors.gold == \"…\"]` below.",
         "**`errors_on_disagreed` does one join**, on the item id. Look at how little "
         "there is to it — the whole force of that number comes from the fact that you "
         "built both tables yourselves, from the same forty items.",
         "**Nothing here calls the model.** Every function takes lists you already "
         "loaded. That is what \"frozen\" means: from here on your numbers can only "
         "change if you load a different file."]),
    embed(metrics.evaluate, metrics.show_errors, metrics.errors_on_disagreed,
          metrics.triage_counts,
          imports=["import pandas as pd",
                   "from sklearn.metrics import (classification_report, "
                   "confusion_matrix,",
                   "                             cohen_kappa_score, f1_score)",
                   "from pipeline import (plot_confusion_matrix, label_set,",
                   "                      triage_category, TRIAGE_CATEGORIES)"],
          why="# Read straight out of scripts/metrics.py — this IS the code that runs."),
    step(
        2, "Score the frozen run",
        "the headline numbers, from the file rather than from a live run.",
        ["evaluate(gold, pred_final, ordered=..., labels=LABELS_ORDER)  ->  macro-F1"],
        "Day 2 S6 Part B · Day 3 — the identical call.",
        "macro_f1 — the numbers for report section 3",
        source="the cell just above · scripts/metrics.py · evaluate",
        starter=[
            "macro_f1 = evaluate(gold, pred_final,",
            "                    ordered=____,        # True only if your labels are a SCALE",
            "                    labels=LABELS_ORDER)",
            'print("macro-F1:", round(macro_f1, 3))',
        ],
        extra=["Note      : evaluate prints per-class P/R/F1 and kappa as well as the",
               "            macro average. \"Which class is it worst at\" is a more useful",
               "            sentence than \"F1 = .62\" — read the table, not just the",
               "            headline.",
               "Careful   : `annotator_agreement()` in notebook 03 compares two",
               "            ANNOTATORS. This compares gold against a MODEL. Same kind of",
               "            number, different claim — do not swap them in the report."]),
    md(
        "---",
        "",
        "# Step 3 — Error analysis",
        "",
        "**This is the part of the project the Q&A will actually go to.** A low F1 with "
        "a clear account of *why* beats a high one without, every time — and the "
        "account is only available to you because you built the gold set yourselves.",
        "",
        "`show_errors` gives you every item the model got wrong. Very different findings "
        "live in that one table, and saying which is which is the whole job:",
        "",
        "| | |",
        "|---|---|",
        "| **`model`** | the label is clear, both your coders agreed at once, and the "
        "model still missed it |",
        "| **`scheme`** | the item is genuinely borderline *under your scheme* — and you "
        "know which ones these are, because you argued about them |",
        "| **`wording`** | the label *name* misleads. `Gap` may read to a model as "
        "\"missing data\". This one your next prompt could fix |",
        "| **`ambiguous`** | the item itself is unclear in a way no scheme would settle |",
        "",
        "The difference between `scheme` and `wording` is worth being careful about: one "
        "of them is fixable by prompting and the other is not, and confusing them is how "
        "groups spend three rounds on a problem no prompt can reach."),
    step(
        3, "The errors, and where they land",
        "get the misses, and see how many fall on items your own coders split on.",
        ["show_errors(gold, pred_final)  ->  a table of the items it got wrong",
         "errors_on_disagreed(errors, disagreed)  ->  the overlapping ids",
         "load_coder_sheets(SHEET_ID, CODERS) · disagreements(...)   (as in 03)"],
        "Day 3, \"where is your best prompt still wrong?\" — identical call.",
        "errors · disagreed",
        source="the cell above · scripts/metrics.py · show_errors, errors_on_disagreed",
        extra=["Note      : the overlap number is the strongest claim this project can",
               "            support — that the model fails where your SCHEME is fuzzy,",
               "            rather than at random. Write it down either way: a LOW",
               "            overlap is just as reportable, and means something else.",
               "Careful   : re-reading the sheet needs the same coder names you used in",
               "            notebook 03. Skip these lines if the sheet is gone — the",
               "            triage below still works without it."],
        starter=[
            "errors = show_errors(gold, pred_final)",
            "errors.head(15)",
            "",
            "# `errors` is a table. To look at one label's misses, keep the rows whose",
            "# gold column equals it — put in one of YOUR labels, spelled as LABELS has",
            "# been printing them:",
            'errors[errors.gold == "____"]',
            "",
            "# Now the cross-reference: your coders' disagreements, back from the sheet.",
            'CODERS = ["CoderA", "CoderB"]        # the same names you used in 03',
            "SHEET_ID = remembered_sheet(SHEET_PATH)",
            "rows = load_coder_sheets(SHEET_ID, CODERS)",
            "disagreed = disagreements(rows, coders=CODERS)",
            "",
            "overlap = errors_on_disagreed(errors, disagreed)",
        ]),
    md(
        "## Step 4 — Triage: say what each error is *caused by*",
        "",
        "Now the judgment. Go through the errors — **all of them if there are few, at "
        "least eight or ten if there are many** — and write down, for each, which of the "
        "four things it is and how you know. Do this **together, out loud**, reading the "
        "actual sentences. It is the last genuinely analytical thing in the project and "
        "it takes about fifteen minutes.",
        "",
        "Start each line with the category word, then a reason:",
        "",
        "```python",
        "TRIAGE = {",
        "     7: \"scheme  — Move 1/Move 2 boundary; our coders split on this one too\",",
        "    12: \"model   — 'The aim of our study was' is about as clear as Move 3 gets\",",
        "    23: \"wording — the model reads any citation as Move 1\",",
        "}",
        "```",
        "",
        "The ids come from the `errors` table above, and the ones in `overlap` are the "
        "obvious candidates for `scheme` — you have independent evidence for those. "
        "**A reason, not a verdict**: *\"model — wrong\"* is not worth writing down.",
        "",
        "Two things come out of this. The counts go into report §4, so *\"of 14 errors, 6 "
        "are our scheme's\"* is a finding rather than an impression. And every `wording` "
        "line is a concrete next prompt round, which is what to say when someone asks "
        "what you would do with another week."),
    step(
        4, "Triage the errors",
        "attribute each miss, from the four categories, with a reason.",
        ["TRIAGE_CATEGORIES  ->  model · scheme · wording · ambiguous",
         "triage_counts(triage, errors)  ->  the counts, and what is left to do",
         "save_json(TRIAGE, TRIAGE_PATH, what=\"triaged errors\")"],
        "new — but it is the same judgment you made adjudicating in notebook 03.",
        "TRIAGE",
        source="the cell above · scripts/metrics.py · triage_counts",
        extra=["Note      : write it while the errors are in front of you. This is the",
               "            single hardest thing to reconstruct a week later.",
               "Ask       : do your `scheme` ids overlap with `overlap` from step 3? If",
               "            they do, say so — your own coders are the evidence."],
        starter=[
            "TRIAGE = {",
            '    ____: "____ — ____",       # id: category — why you say so',
            '    ____: "____ — ____",',
            '    ____: "____ — ____",',
            "}",
            "",
            "triage_counts(TRIAGE, errors)",
            'save_json(TRIAGE, TRIAGE_PATH, what="triaged errors")',
        ]),
    md(
        "## Step 5 — Export",
        "",
        "Writes your gold set, a per-item predictions CSV, and a one-page report "
        "scaffold with the five required sections, all stamped with your group name.",
        "",
        "The scaffold fills in what it can compute — labels, counts, the F1-per-round "
        "table, and now your triage: the category counts, and your reason printed beside "
        "each item. The *italic* placeholders are what is left for you: the QC narrative, "
        "the pattern in your `scheme` errors, and limitations that apply to **your** run "
        "rather than the generic three. A section left as the placeholder scores zero, "
        "so this is the start of the writing, not the end."),
    step(
        5, "Export",
        "write the gold set, the predictions CSV, and the report scaffold.",
        ["export_results(TRACK, gold, pred_final, f1_by_round, OUT_DIR, group=GROUP,",
         "               run=RUN, triage=TRIAGE)"],
        "new — but it only writes down what you already have.",
        "three files in ../outputs/",
        source="scripts/pipeline.py · export_results",
        extra=["Note      : pass triage=TRIAGE and section 4 becomes your analysis. Leave",
               "            it off and section 4 is a placeholder asking you for it."],
        starter=[
            "export_results(TRACK, gold, pred_final, f1_by_round, OUT_DIR,",
            "               group=GROUP, run=RUN, triage=TRIAGE)",
        ]),
    md(
        "---",
        "",
        "## Hand it in",
        "",
        "One command collects everything into a folder next to the repo, keeping the "
        "`scripts/ · prompts/ · data/ · notebooks/ · outputs/` layout — because that "
        "layout *is* the reproducibility checklist from S10, and because the notebooks' "
        "paths only resolve if it stays intact.",
        "",
        "```bash",
        "python scripts/make_submission.py --group groupA",
        "```",
        "",
        "It deliberately leaves out `.git/`, `.venv/`, your `.env` (**it holds your API "
        "key**), the big pools in `data/pools/`, and anything ICNALE-derived.",
        "",
        "Then: find the folder in Drive → right-click → **Download** → upload the zip to "
        "the *Final mini-project* assignment in Google Classroom → **Turn in**. One "
        "submission per group, with every member's name in `PLAN.md`.",
        "",
        "Before you do, check that **all five notebooks run top to bottom on a fresh "
        "runtime**, in order. If they only work in the session where you built them "
        "piece by piece, they do not yet reproduce — and 02 through 05 handing files to "
        "each other is exactly what makes that checkable."),
    code(
        "# Optional: build the bundle from here instead of a terminal.",
        "# !cd .. && python scripts/make_submission.py --group $GROUP",
        ""),
]


# ==================================================================================
def write(name, cells):
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
    blank_cells = sum(1 for c in cells
                      if c["cell_type"] == "code" and "✏️" in "".join(c["source"]))
    print("wrote", name, "(" + str(len(cells)), "cells,", blank_cells, "blank)")


write("02_sample.ipynb", cells)
write("03_annotate.ipynb", cells_03)
write("04_prompt.ipynb", cells_04)
write("05_report.ipynb", cells_05)
