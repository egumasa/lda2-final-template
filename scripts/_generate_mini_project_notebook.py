#!/usr/bin/env python3
"""Generate notebooks/mini_project.ipynb — the project skeleton.

    python scripts/_generate_mini_project_notebook.py

Never hand-edit the .ipynb: edit this file and re-run it. (Same rule as the course
repo's day notebooks — see planning/course_planning/notebook-coding-principles.md.)

WHAT THE SKELETON IS
--------------------
One notebook, all tracks. Six steps, each shipped as a BLANK SPINE: the header, the
goal in one line, the helpers that are available, a pointer to where you already ran
them during the week, and the variable names the next step expects. You write the calls.

That is deliberate. The helpers are all written for you — nobody learns anything from
retyping a rate-limit guard — but the SHAPE of the pipeline, what each step consumes and
produces, is the thing worth being able to reconstruct. It is also the thing you have to
be able to narrate in the Q&A.

Every call you need has the same form it had in Days 1–3. If a call from the tutorials
does not work here, that is a bug in the template, not in your memory of it — say so.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"
REPO = "egumasa/lda2-final-template"


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def step(number, title, goal, available, pointer, produces, extra=()):
    """A blank-spine step cell: everything except the code."""
    rule = "═" * max(4, 64 - len(title) - len(str(number)))
    lines = [
        "# ══ STEP " + str(number) + " · " + title + " " + rule,
        "# Goal      : " + goal,
    ]
    for index, line in enumerate(available):
        if index == 0:
            lines.append("# Available : " + line)
        else:
            lines.append("#             " + line)
    lines.append("# Pointer   : " + pointer)
    lines.append("# Produce   : " + produces + "      ← the next step uses these names")
    for line in extra:
        lines.append("# " + line)
    lines.append("")
    lines.append("# ✏️ your code here")
    lines.append("")
    return code(*lines)


cells = []

# ------------------------------------------------------------------ title & how to use
cells.append(md(
    "# Final mini-project — the pipeline, one step at a time",
    "",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/" + REPO + "/blob/main/notebooks/"
    "mini_project.ipynb)",
    "",
    "This is **your** notebook: one skeleton, any track. Six steps:",
    "",
    "> **sample** a balanced subset → **QC / adjudicate** it → **baseline** prompt → "
    "**iterate** and freeze → **error analysis** → **export**",
    "",
    "Each step ships with its header, its goal, the helpers available to it, and the "
    "variable names the next step expects. **You write the calls.** Every call has the "
    "same form it had in Days 1–3 — the pointer line in each step says exactly where you "
    "ran it before.",
    "",
    "You edit: the **CONFIG cell**, the **six step cells**, and your **prompt file**. "
    "You never edit anything in `scripts/`.",
    "",
    "---",
    "",
    "### Working as a group",
    "",
    "This notebook lives in a shared Drive folder, so **all of you can edit it at the "
    "same time** — Colab syncs edits like a Google Doc. Two things do *not* work that "
    "way:",
    "",
    "- **Runtimes are per-person.** Seeing `gold` in a saved output does not mean `gold` "
    "exists in *your* session. Whoever is running the cells is the **driver**.",
    "- **Files in the repo folder are last-write-wins.** `prompts/`, `outputs/` and "
    "`data/` are ordinary files, not Google Docs. Let the driver be the only one running "
    "cells that write them.",
    "",
    "The **annotation Sheet in step 2 is the exception** — that is a real Google Sheet, "
    "so annotate it together, all at once.",
    "",
    "One more reason for a single driver: your final run has to be **one run**, by one "
    "person, frozen to a file. That is step 4."))

# ------------------------------------------------------------------ setup
cells.append(md(
    "## Setup — run this first",
    "",
    "In Colab, uncomment **one** of the two clone blocks below before running the cell. "
    "Colab starts with only this one file; the clone fetches everything *around* it "
    "(`scripts/`, `data/`, `prompts/`) so that the `../` paths resolve.",
    "",
    "**Do Option A once, as a group** — then always open the copy that lives in Drive "
    "(*File ▸ Open ▸ Drive ▸ `lda2-final-template/notebooks/mini_project.ipynb`*). Your "
    "prompts, gold set and outputs then survive the runtime resetting, and everyone sees "
    "the same files.",
    "",
    "> **Free-tier pacing.** The backend waits a few seconds between calls and retries "
    "on rate-limit errors, so a full run takes minutes and may print "
    "`(rate limited - waiting Ns then retrying)`. That is normal. While you are still "
    "iterating, keep `N_PER_CLASS` small (2) — then do **one** final run at full size."))

cells.append(code(
    "# ------------------------------------------------------------------",
    "# SETUP — run me first.",
    "# ------------------------------------------------------------------",
    "# In Google Colab, UNCOMMENT one of the two blocks below, then run the cell.",
    "",
    "# --- Colab Option A: clone into your Google Drive (persists; do this once) ---",
    "# from google.colab import drive",
    '# drive.mount("/content/drive")',
    "# %cd /content/drive/MyDrive",
    "# ![ -d lda2-final-template ] || git clone https://github.com/" + REPO
    + ".git lda2-final-template",
    "# %cd /content/drive/MyDrive/lda2-final-template/notebooks",
    "",
    "# --- Colab Option B: quick, throwaway clone (changes lost on reset) ---",
    "# !git clone https://github.com/" + REPO + ".git",
    "# %cd lda2-final-template/notebooks",
    "",
    "# Put scripts/ on the import path. Works locally AND in Colab after the %cd above,",
    "# because notebooks/ and scripts/ sit side by side.",
    "import sys",
    'sys.path.append("../scripts")',
    "",
    "from pipeline import *      # load_gold, sample_pool, run_prompt, save_predictions, ...",
    "from metrics import *       # evaluate, agreement, show_errors",
    "from annotate import *      # create_annotation_sheet, annotator_agreement, ...",
    "",
    "setup()                     # connect to the model and say which backend we got",
    ""))

cells.append(md(
    "> **Check the line it just printed.** You want:",
    ">",
    "> ```",
    "> LLM backend: Gemini API (gemini-3.1-flash-lite, temperature=0, seed=42)",
    "> ```",
    ">",
    "> If it says **Colab Gemini** instead, no API key was found — put yours in the Colab "
    "Secrets panel (the 🔑 icon in the left sidebar) as `GEMINI_API_KEY` and re-run. "
    "The keyless backend has no temperature or seed, so the same prompt can give "
    "different answers, and your numbers will not be reproducible. It must not be your "
    "final run."))

# ------------------------------------------------------------------ cheat sheet
cells.append(md(
    "## The helpers, and where you already used them",
    "",
    "Everything below is written for you. Eight of these are the **same call** you made "
    "earlier in the week.",
    "",
    "| Step | Helper | Where you ran it before |",
    "|---|---|---|",
    "| 1 | `load_gold(path)` | Day 2 S5 step F · Day 3 setup |",
    "| 1 | `sample_pool(pool, n_per_class, seed)` | Day 4 Part A — as four separate steps |",
    "| 1 | `label_set(gold)` | new (one line) |",
    "| 2 | `create_annotation_sheet(title, items, labels)` | Day 2 S5 step A |",
    "| 2 | `load_annotation_sheet(sheet_id, worksheet)` | Day 2 S5 step D |",
    "| 2 | `annotator_agreement(rows)` · `disagreements(rows)` | Day 2 S5 steps D–E |",
    "| 2 | `to_canonical(rows, labels)` · `compare_to_published(gold, sampled)` | Day 2 S5 step F |",
    "| 3 | `load_prompt(path)` | new — your prompt is a *file* now |",
    "| 3–4 | `run_prompt(PROMPT, gold)` | Day 3, every iteration |",
    "| 3–4 | `evaluate(gold, pred, ordered=True)` | Day 2 S6 Part B · Day 3 |",
    "| 4 | `build_fewshot(PROMPT, pool, gold)` | Day 3 — you typed the examples by hand |",
    "| 4 | `save_predictions(pred, path)` · `load_predictions(path)` | Day 2 S6 loaded a frozen file; now you make one |",
    "| 5 | `show_errors(gold, pred)` | Day 3 |",
    "| 6 | `export_results(track, gold, pred, f1_by_round, out_dir, group=...)` | new |",
    "",
    "Need a signature? `help(sample_pool)` prints it, and every one of them is readable "
    "in `../scripts/`."))

# ------------------------------------------------------------------ CONFIG
cells.append(md(
    "## CONFIG   ✏️ YOU EDIT",
    "",
    "The one cell you change for your own track. It should match your `PLAN.md`."))

cells.append(code(
    "# ------------------------------------------------------------------",
    "# ✏️ CONFIG — set these to match your PLAN.md, then leave them alone.",
    "# ------------------------------------------------------------------",
    'TRACK       = "cefr"         # cefr · raamove · cars50 · l2_errors · icnale',
    'GROUP       = "groupA"       # your group name — it goes in every output filename',
    "SEED        = 42             # change per group so each draws a different subset",
    "N_PER_CLASS = 2              # keep SMALL while iterating; raise for the final run",
    "",
    "# Where the items come from. The demo file ships with the template so this notebook",
    "# runs immediately — but it is far too small for a real study (sampling would take",
    "# most of it and leave nothing for few-shot examples). Build the real pool with",
    "#     python scripts/prep_datasets.py cefr",
    "# then switch POOL_PATH to the pools/ line below.",
    'POOL_PATH = "../data/gold/" + TRACK + "_demo.json"      # demo — see the pipeline work',
    '# POOL_PATH = "../data/pools/" + TRACK + "_pool.json"   # ✏️ the real run',
    "",
    "# Your prompt lives in a FILE, so each version is saved and comparable.",
    'PROMPT_FILE = "../prompts/" + TRACK + ".txt"',
    'OUT_DIR     = "../outputs"',
    "",
    "# Step 2 fills these in: paste the URL your annotation sheet prints, and the tab",
    "# name for the round you are reading back.",
    'SHEET_ID = ""                # the sheet URL, or just the long id from it',
    'ROUND    = "round1"          # each re-annotation round gets its own tab',
    "",
    "# Are your labels ORDERED (on a scale), and if so in what order?",
    "#   A1..C2 and Move 1..3 are ordered AND alphabetical, so None is fine.",
    '#   Low/Mid/High is ordered but NOT alphabetical — set it, or the weighted kappa',
    '#   gets computed over "High < Low < Mid", which means nothing.',
    "LABELS_ORDER = None          # e.g. [\"Low\", \"Mid\", \"High\"] for icnale",
    ""))

# ------------------------------------------------------------------ step 1
cells.append(md(
    "## Step 1 — Sample a balanced gold subset",
    "",
    "A *pool* is everything the corpus has, with its natural imbalance. A *sample* is the "
    "balanced subset you actually study — equal items per label, so precision, recall, F1 "
    "and the confusion matrix all stay readable. Rare labels simply yield fewer; that is "
    "a property of the data, and it belongs in your limitations.",
    "",
    "Keeping the two separate is also what leaves unused items available as few-shot "
    "examples in step 4, without showing the model the answers you are testing it on."))

cells.append(step(
    1, "Sample a balanced gold subset",
    "turn the big pool into ~40 items, balanced across your labels.",
    ["load_gold(path)  ·  sample_pool(pool, n_per_class, seed)  ·  label_set(gold)"],
    "Day 4 Part A — sample_pool does those four steps in one call.",
    "pool · sampled · LABELS"))

# ------------------------------------------------------------------ step 2
cells.append(md(
    "## Step 2 — QC / adjudicate  →  *your* gold set",
    "",
    "This is the part no model can do for you, and the part the Q&A will ask about.",
    "",
    "The published labels are somebody else's judgment. Before you measure a model "
    "against them, two of you re-annotate the sample **independently and blind**, see how "
    "far apart you were, and argue out the rows you disagreed on. What comes out is *your* "
    "gold set — and the disagreements tell you which label boundaries are genuinely fuzzy. "
    "That is what lets you say, later, whether a model's miss is the **model's** fault or "
    "the **scheme's**.",
    "",
    "Three sub-steps: **2a** create the sheet · **2b** measure agreement · **2c** "
    "adjudicate and canonicalise.",
    "",
    "The sheet is a real Google Sheet, so all of you can annotate at once. Write your "
    "labels in `CoderA` and `CoderB`; leave `Final` until you have talked."))

cells.append(step(
    "2a", "Create the annotation sheet",
    "make a blind annotation sheet in your Drive, one row per sampled item.",
    ["create_annotation_sheet(title, items, labels)  ->  the sheet URL"],
    "Day 2 S5 step A. Paste the URL it prints into SHEET_ID in CONFIG.",
    "a sheet URL (nothing to name)",
    extra=["Note      : the published label is deliberately NOT copied in — you annotate blind."]))

cells.append(step(
    "2b", "Measure agreement",
    "how often did the two of you agree, and on which labels did you not?",
    ["load_annotation_sheet(sheet_id, worksheet)  ->  rows",
     "annotator_agreement(rows)  ·  disagreements(rows)"],
    "Day 2 S5 steps D–E — identical calls.",
    "rows",
    extra=["Note      : run this once both CoderA and CoderB columns are filled in.",
           "            Write the percent agreement and kappa down — report section 1."]))

cells.append(step(
    "2c", "Adjudicate, then canonicalise",
    "agree a Final label for every disagreement, then turn the sheet into gold.",
    ["load_annotation_sheet(sheet_id, worksheet)  ->  rows   (re-read after editing)",
     "to_canonical(rows, LABELS)  ->  gold",
     "compare_to_published(gold, sampled)  ->  where you differ from the source"],
    "Day 2 S5 step F — identical calls.",
    "gold",
    extra=["Careful   : compare against `sampled`, not `pool`.",
           "Note      : to_canonical reports blank and invalid rows — fix them in the",
           "            sheet and re-run until it says 0 invalid."]))

cells.append(md(
    "> **Save your gold set now** — it is the single most valuable thing you have made, "
    "and it goes in your submission:",
    ">",
    "> ```python",
    "> import json",
    '> with open("../data/gold/" + TRACK + "_" + GROUP + "_gold.json", "w",',
    '>           encoding="utf-8") as f:',
    ">     json.dump(gold, f, ensure_ascii=False, indent=2)",
    "> ```",
    ">",
    "> (`export_results` in step 6 writes a copy too, but do not wait for it — this is "
    "hours of your group's judgment.)"))

# ------------------------------------------------------------------ the gate
cells.append(md(
    "---",
    "",
    "## 🛑 Before step 3 — the `PLAN.md` gate",
    "",
    "Steps 3 onward call the model. **Do not start until your `PLAN.md` has been read and "
    "signed off.** It takes two minutes and it is not busywork: a mismatched label set or "
    "an unstated sampling seed costs an hour to unpick *after* you have burned quota on "
    "it.",
    "",
    "Check, out loud, that these agree with each other: the label set in `PLAN.md`, the "
    "labels `label_set` actually returned in step 1, and the labels your prompt file "
    "names."))

# ------------------------------------------------------------------ step 3
cells.append(md(
    "## Step 3 — Baseline prompt (round 0)",
    "",
    "A number to beat. Write the plainest prompt that states the task and the label set, "
    "run it, and score it. Resist the urge to make it good — the point of a baseline is "
    "that later rounds have something to be measured against.",
    "",
    "Your prompt lives in `prompts/<track>.txt` and must contain `{text}`, which is where "
    "each item gets slotted in. Edit the **file**, not a string in this notebook — that is "
    "what makes each version savable and comparable, and it is the reproducibility habit "
    "from S10.",
    "",
    "In Colab you can write the file straight from a cell:",
    "",
    "```python",
    "%%writefile ../prompts/cefr_v0.txt",
    "Classify the CEFR level of the sentence. Answer with the level only.",
    "...",
    "Sentence: {text}",
    "```"))

cells.append(step(
    3, "Baseline prompt (round 0)",
    "get one honest number to beat.",
    ["load_prompt(path)  ->  PROMPT",
     "run_prompt(PROMPT, gold)  ->  predictions",
     "evaluate(gold, predictions, ordered=..., labels=LABELS_ORDER)  ->  macro-F1"],
    "Day 3 Part A — the same two lines.",
    "f1_by_round · pred0",
    extra=["Note      : start f1_by_round = {} here and add a row per round, so step 6",
           "            can write the prompt-iteration table for you.",
           "Careful   : keep N_PER_CLASS small for this. Full size is ~3 min of pure",
           "            waiting per round on the free tier."]))

# ------------------------------------------------------------------ step 4
cells.append(md(
    "## Step 4 — Iterate, then **freeze**",
    "",
    "Two or three more rounds. For each one, change **one thing**, predict what it will "
    "do, and then find out. \"Added examples\" is not a reason; \"the model kept confusing "
    "B1 and B2, so I gave it one example of each\" is.",
    "",
    "`build_fewshot` draws examples from the pool while avoiding anything in your gold "
    "set — otherwise you would be testing the model on answers you had just shown it.",
    "",
    "### Then freeze it",
    "",
    "A hosted model is only *best-effort* reproducible, even at `temperature=0`. So once "
    "your best prompt is settled: raise `N_PER_CLASS`, run it **once**, and "
    "`save_predictions` to a file. Every number you report from here on comes out of that "
    "file. That is what makes your F1 stable, and what lets anyone else re-run your "
    "analysis on exactly the outputs you saw.",
    "",
    "One person runs this. It is the run you will be defending."))

cells.append(step(
    4, "Iterate, then freeze",
    "2–3 reasoned rounds, then one final full-size run saved to a file.",
    ["build_fewshot(PROMPT, pool, gold)  ->  a new prompt",
     "run_prompt(...)  ·  evaluate(...)   (as in step 3)",
     "save_predictions(predictions, path)  ·  load_predictions(path)"],
    "Day 3 iterations 1–2. build_fewshot replaces typing the examples by hand.",
    "pred_final",
    extra=["Freeze    : save to OUT_DIR + \"/\" + TRACK + \"_\" + GROUP + \"_predictions.json\"",
           "            then load it straight back and use THAT from step 5 on.",
           "Note      : record what you changed each round and why — report section 2."]))

# ------------------------------------------------------------------ step 5
cells.append(md(
    "## Step 5 — Error analysis",
    "",
    "The highest-value part of the whole project, and the one the Q&A will definitely go "
    "to: **show an item the model got wrong, and say whose fault it was.**",
    "",
    "Two different findings live in this table:",
    "",
    "- *The model's fault* — the label is clear, a competent annotator would agree, and "
    "the model still missed it.",
    "- *The scheme's fault* — the item is genuinely borderline. You know which ones these "
    "are, because they are the rows you argued about in step 2.",
    "",
    "Cross-reference your `disagreements` list against these errors. Overlap is a finding, "
    "not a failure — and a low F1 with a clear account of *why* is worth more than a high "
    "one without."))

cells.append(step(
    5, "Error analysis",
    "find the misses, and attribute each one.",
    ["show_errors(gold, pred_final)  ->  a table of the items it got wrong"],
    "Day 3, \"where is your best prompt still wrong?\" — identical call.",
    "errors",
    extra=["Note      : pick at least three, and write a REASON for each — report section 4.",
           "Try       : errors.head(15)  ·  errors[errors.gold == \"B2\"]"]))

# ------------------------------------------------------------------ step 6
cells.append(md(
    "## Step 6 — Export",
    "",
    "Writes your gold set, a per-item predictions CSV, and a one-page report scaffold "
    "with the five required sections, all stamped with your group name.",
    "",
    "The scaffold fills in what it can compute — labels, counts, the F1-per-round table. "
    "The *italic* placeholders are yours: the QC narrative, the error attributions, and "
    "limitations that apply to **your** run rather than the generic three. A section left "
    "as the placeholder scores zero, so this is the start of the writing, not the end."))

cells.append(step(
    6, "Export",
    "write the gold set, the predictions CSV, and the report scaffold.",
    ["export_results(TRACK, gold, pred_final, f1_by_round, OUT_DIR, group=GROUP)"],
    "new — but it only writes down what you already have.",
    "three files in ../outputs/"))

# ------------------------------------------------------------------ submission
cells.append(md(
    "---",
    "",
    "## Hand it in",
    "",
    "One command collects everything into a folder next to the repo, keeping the "
    "`scripts/ · prompts/ · data/ · outputs/` layout — because that layout *is* the "
    "reproducibility checklist from S10, and because the notebook's `../` paths only "
    "resolve if it stays intact.",
    "",
    "```bash",
    "python scripts/make_submission.py --group groupA",
    "```",
    "",
    "It deliberately leaves out `.git/`, `.venv/`, your `.env` (**it holds your API "
    "key**), the big pools in `data/pools/`, and anything ICNALE-derived.",
    "",
    "Then: find the folder in Drive → right-click → **Download** → upload the zip to the "
    "*Final mini-project* assignment in Google Classroom → **Turn in**. One submission "
    "per group, with every member's name in `PLAN.md`.",
    "",
    "Before you do, check that the notebook you are submitting **runs top to bottom on a "
    "fresh runtime**. If it only works in the session where you built it piece by piece, "
    "it does not yet reproduce."))

cells.append(code(
    "# Optional: build the bundle from here instead of a terminal.",
    "# !cd .. && python scripts/make_submission.py --group $GROUP",
    ""))

# ------------------------------------------------------------------ write
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
(OUT / "mini_project.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote mini_project.ipynb (" + str(len(cells)), "cells,",
      str(sum(1 for c in cells if c["cell_type"] == "code")), "code)")
