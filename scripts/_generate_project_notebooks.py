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

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def step(number, title, goal, available, pointer, produces, extra=()):
    """A blank-spine step cell: everything except the code."""
    rule = "═" * max(4, 62 - len(title) - len(str(number)))
    lines = [
        "# ══ STEP " + str(number) + " · " + title + " " + rule,
        "# Goal      : " + goal,
    ]
    for index, line in enumerate(available):
        lines.append(("# Available : " if index == 0 else "#             ") + line)
    lines.append("# Pointer   : " + pointer)
    lines.append("# Produce   : " + produces + "      ← later cells use these names")
    for line in extra:
        lines.append("# " + line)
    lines.append("")
    lines.append("# ✏️ your code here")
    lines.append("")
    return code(*lines)


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
    "> **Everything above comes from `config.py`** — one file at the top of the repo, "
    "which you edit once as a group. That is deliberate: the seed in notebook 02 has to "
    "be the seed in notebook 03, and five copies of a number in five notebooks is five "
    "chances for them to disagree. If the line it just printed is not your track, your "
    "group and your seed, fix `config.py` and re-run this cell.")


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
        "02", "02_sample", "Sample a balanced subset",
        "Turn the whole pool into the ~40 items you will actually annotate and study.",
        "`data/pools/<track>_pool.json` (from 01)",
        "`data/gold/<track>_<group>_sample.json`",
        ["### Working as a group",
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
         "The **annotation Sheet in notebook 03 is the exception** — that is a real "
         "Google Sheet, so annotate it together, all at once."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "from pipeline import *      # load_gold, sample_pool, label_set, save_json, ...",
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
        ["load_gold(path)  ->  a list of {id, text, label}",
         "POOL_PATH · DEMO_POOL_PATH   (both come from config.py)"],
        "Day 2 S5 step F · Day 3 setup — the same call.",
        "pool",
        extra=["Try       : from collections import Counter",
               "            Counter(item[\"label\"] for item in pool)"]),
    md(
        "### Before you sample — decide `N_PER_CLASS`",
        "",
        "Look at the counts you just printed and find your **smallest class**. That is "
        "a hard ceiling: a balanced sample cannot draw more from a class than the class "
        "has. Ask for more and `sample_pool` gives you everything the rare class has "
        "and moves on — your sample is then *not* balanced, and it will not say so "
        "twice.",
        "",
        "If the rarest class is thin, say so in `PLAN.md`. Merging it away or living "
        "with fewer items are both defensible; not noticing is not.",
        "",
        "The other ceiling is time. Every item is one API call in notebook 04, at a few "
        "seconds each, times the number of rounds — and every item is also a row two of "
        "you have to annotate by hand in notebook 03. Around 40 items total is the "
        "size this project is built for.",
        "",
        "Set `N_PER_CLASS` in `config.py` and re-run the setup cell if you change it."),
    step(
        2, "Draw the balanced sample",
        "turn the big pool into ~40 items, balanced across your labels.",
        ["sample_pool(pool, n_per_class, seed)  ->  sampled",
         "label_set(items)  ->  the sorted list of labels present"],
        "Day 4 Part A — sample_pool does those four steps in one call.",
        "sampled · LABELS",
        extra=["Careful   : pass SEED explicitly. A sample you cannot redraw is not a",
               "            sample anyone can check — and your report has to state it.",
               "Note      : read the per-label counts it prints. If a class came back",
               "            short, that is your rare class hitting its ceiling.",
               "Note      : if it WARNS that you took most of the pool, you are almost",
               "            certainly still pointed at DEMO_POOL_PATH."]),
    md(
        "### Sanity-check what you drew",
        "",
        "Two questions worth answering before you commit forty hand-annotations to it:",
        "",
        "1. **Is it actually balanced?** Count the labels in `sampled`.",
        "2. **Is there pool left over?** `build_fewshot` in notebook 04 draws its "
        "examples from the items you did *not* sample. If the sample is most of the "
        "pool, there is nothing uncontaminated left to draw from."),
    step(
        3, "Check the draw",
        "confirm the balance, and confirm there is pool left over.",
        ["len(pool) · len(sampled)  ·  Counter(item[\"label\"] for item in sampled)"],
        "Day 4 Part A — the same counts you printed there.",
        "nothing to name — this is a check, not a stage",
        extra=["Ask       : how many items per label did you get, and does the shortfall",
               "            match the smallest class in the pool counts from step 1?"]),
    handoff_md(
        "sample", "data/gold/<track>_<group>_sample.json", "03_annotate",
        "The cell below writes the sample to a file. Notebook 03 opens that file to "
        "build your annotation sheet, and again at the end to compare your labels "
        "against the published ones — so it has to be the same forty items, not a "
        "redraw. Even with a fixed seed, save it: a seed reproduces a draw only as long "
        "as nobody edits the pool underneath it."),
    code(
        "save_json(sampled, SAMPLE_PATH, what=\"sampled items\")",
        "",
        "# It still carries the PUBLISHED label at this point. Notebook 03 deliberately",
        "# does not copy that into your annotation sheet — you annotate blind — but it",
        "# does use it at the very end, to show you where your group disagreed with the",
        "# corpus. That comparison is one of the more interesting things in your report.",
        ""),
]

# ==================================================================================
# 03 — annotate
# ==================================================================================
cells_03 = [
    title_cell(
        "03", "03_annotate", "Annotate, adjudicate → *your* gold set",
        "The part no model can do for you, and the part the Q&A will ask about.",
        "`data/gold/<track>_<group>_sample.json` (from 02)",
        "`data/gold/<track>_<group>_gold.json`",
        ["The published labels are somebody else's judgment. Before you measure a model "
         "against them, two of you re-annotate the sample **independently and blind**, "
         "see how far apart you were, and argue out the rows you disagreed on.",
         "",
         "What comes out is *your* gold set — and the disagreements tell you which label "
         "boundaries are genuinely fuzzy. That is what lets you say, later, whether a "
         "model's miss is the **model's** fault or the **scheme's**. Nothing else in "
         "the project can tell you that.",
         "",
         "> Budget real time for this. Forty items, two annotators, plus the argument "
         "afterwards. It is the most valuable thing you will make this week and the "
         "easiest to rush."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "from pipeline import *      # load_gold, label_set, save_json, ...",
        "from annotate import *      # create_annotation_sheet, annotator_agreement, ...",
    ]),
    CONFIG_MD,
    step(
        1, "Load your sample",
        "open the forty items notebook 02 drew.",
        ["load_gold(path)  ->  a list of {id, text, label}",
         "SAMPLE_PATH   (from config.py)"],
        "Day 2 S5 step F — the same call.",
        "sampled · LABELS",
        extra=["Note      : it still carries the published label. Do not read it, and do",
               "            not print it — you are about to annotate these blind."]),
    md(
        "## Step 2 — Create the annotation sheet",
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
        "`share_with=MEMBERS` — the Google accounts you put in `config.py` — or your "
        "second coder will open the link and be told they need access. Pass "
        "`remember=SHEET_PATH` too, and the link is written to a file instead of living "
        "in this cell's output, where a runtime reset can lose it."),
    step(
        2, "Create the sheet",
        "make a blind annotation sheet, shared with your group, one row per item.",
        ["create_annotation_sheet(title, items, labels,",
         "                        share_with=..., remember=...)  ->  the sheet URL",
         "MEMBERS · SHEET_PATH   (from config.py)"],
        "Day 2 S5 step A — the same call, plus the two sharing arguments.",
        "a sheet URL — and a note of it on disk, for the step after next",
        extra=["Note      : give it a title with your group and track in it. You will",
               "            have several of these by the end of the week.",
               "Careful   : run this ONCE. Running it again makes a SECOND sheet, and",
               "            half your annotations end up in the one nobody read back."]),
    md(
        "### Now go and annotate",
        "",
        "Open the sheet, and label every row. Rules of the exercise:",
        "",
        "- **CoderA and CoderB work independently.** Different people, no discussion, "
        "no peeking at the other column.",
        "- **Leave `Final` empty** until you have both finished and talked.",
        "- **Use `Note`** when you hesitate. The item you were unsure about is the item "
        "you will want to quote in your error analysis, and you will not remember which "
        "one it was.",
        "- Labels must be spelled exactly as `LABELS` prints them. `to_canonical` will "
        "tell you about typos, but it is quicker not to make them.",
        "",
        "This is the point where the notebook stops and the week's actual work happens. "
        "Come back when both columns are full."),
    code(
        "# The sheet step 2 created, read back from the file it wrote. This is why it",
        "# wrote one: whoever runs the next cell need not be the person who ran step 2,",
        "# and need not still have that cell's output on screen.",
        "SHEET_ID = remembered_sheet(SHEET_PATH)",
        "",
        "# Working on a sheet someone made before this file existed? Paste its URL (or",
        "# just the long id from it) here instead:",
        '# SHEET_ID = ""',
        "",
        'ROUND = "round1"             # each re-annotation round gets its own tab',
        "",
        'print("sheet:", SHEET_ID or "-- none saved yet: run step 2 --")',
        ""),
    md(
        "## Step 3 — Measure agreement",
        "",
        "Two numbers and a matrix: raw percent agreement, Cohen's κ (agreement "
        "corrected for what you would get by guessing), and an annotator-vs-annotator "
        "confusion matrix whose off-diagonal cells show *which* label pairs the two of "
        "you confuse.",
        "",
        "**Write these down now** — they are report section 1, and they do not survive "
        "a runtime reset. A κ around .8 is strong; around .4 means the scheme, not the "
        "annotators, is doing something wrong. Either is a reportable finding. A low κ "
        "you can explain beats a high one you cannot."),
    step(
        3, "Measure agreement",
        "how often did the two of you agree, and on which labels did you not?",
        ["load_annotation_sheet(sheet_id, worksheet)  ->  rows",
         "annotator_agreement(rows)  ·  disagreements(rows)"],
        "Day 2 S5 steps D–E — identical calls.",
        "rows · disagreed",
        extra=["Note      : run this once BOTH CoderA and CoderB columns are filled in.",
               "            Half-finished rows are dropped from the comparison.",
               "Keep      : `disagreed` matters again in notebook 05 — the rows you",
               "            argued about are the ones to check your model's errors",
               "            against. Save the list, or write the ids down."]),
    md(
        "## Step 4 — Adjudicate",
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
        4, "Adjudicate, then canonicalise",
        "agree a Final label for every row, then turn the sheet into gold.",
        ["load_annotation_sheet(sheet_id, worksheet)  ->  rows   (re-read after editing)",
         "to_canonical(rows, LABELS, source=sampled)  ->  gold"],
        "Day 2 S5 step F — identical calls.",
        "gold",
        extra=["Careful   : re-read the sheet first. `rows` from step 3 is a snapshot",
               "            from before you filled in Final.",
               "Note      : keep going until it prints 0 blank and 0 invalid. A blank",
               "            row is an item silently missing from your study.",
               "Note      : pass source=sampled. Gold is rebuilt from the SHEET, which",
               "            holds only the id, the text and your label — anything else",
               "            the item carried (on cars50/raamove, its passage) is put",
               "            back from `sampled` by id. Harmless on the other tracks."]),
    md(
        "## Step 5 — Where do you differ from the published labels?",
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
        5, "Compare against the published labels",
        "see where your gold set and the corpus disagree, and work out why.",
        ["compare_to_published(gold, sampled)  ->  a table of the rows that differ"],
        "Day 2 S5 step F — the same call.",
        "differences",
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
        "`label_set` actually returned above, and the labels your prompt file names."),
]

# ==================================================================================
# 04 — prompt
# ==================================================================================
cells_04 = [
    title_cell(
        "04", "04_prompt", "Baseline, iterate, freeze",
        "Write the plainest prompt that could work, then improve it for reasons you can "
        "state.",
        "`data/gold/<track>_<group>_gold.json` (from 03) · the pool (from 01)",
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
        "from pipeline import *      # load_gold, load_prompt, run_prompt, save_json, ...",
        "from metrics import *       # evaluate, agreement, show_errors",
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
        ["load_gold(path)  ->  a list of {id, text, label}",
         "GOLD_PATH · POOL_PATH   (from config.py)"],
        "Day 3 setup — the same call, twice.",
        "gold · pool · LABELS",
        extra=["Note      : LABELS comes from your GOLD set, not the pool. If a label",
               "            never survived adjudication, it is not in your study."]),
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
        2, "Baseline prompt (round 0)",
        "get one honest number to beat.",
        ["load_prompt(path)  ->  PROMPT",
         "run_prompt(PROMPT, gold)  ->  predictions",
         "evaluate(gold, predictions, ordered=..., labels=LABELS_ORDER)  ->  macro-F1"],
        "Day 3 Part A — the same two lines.",
        "f1_by_round · pred0",
        extra=["Note      : start f1_by_round = {} here and add a row per round. It is",
               "            the prompt-iteration table in your report, and notebook 05",
               "            reads it from a file.",
               "Careful   : keep N_PER_CLASS small for this. Full size is minutes of",
               "            pure waiting per round on the free tier."]),
    md(
        "## Step 3 — Iterate",
        "",
        "Two or three more rounds. For each one, **change one thing, predict what it "
        "will do, then find out.** \"Added examples\" is not a reason; \"the model kept "
        "confusing B1 and B2, so I gave it one example of each\" is. Write the reason "
        "down as you go — reconstructing it afterwards from a stack of F1 numbers is "
        "much harder than it sounds, and it is report section 2.",
        "",
        "A round that made things **worse** is a result, not a mistake. Keep it in the "
        "table. It is often the most informative row you have.",
        "",
        "`build_fewshot` draws examples from the pool while avoiding anything in your "
        "gold set — otherwise you would be testing the model on answers you had just "
        "shown it."),
    step(
        3, "Iterate — 2–3 reasoned rounds",
        "improve the prompt for stated reasons, and record what each change did.",
        ["build_fewshot(PROMPT, pool, gold)  ->  a new prompt with examples",
         "run_prompt(...)  ·  evaluate(...)   (as in step 2)"],
        "Day 3 iterations 1–2. build_fewshot replaces typing the examples by hand.",
        "PROMPT (your best one) · f1_by_round",
        extra=["Note      : save each version as its own prompt file (v0, v1, v2). A",
               "            prompt you overwrote is a round you cannot report.",
               "Ask       : did the confusion matrix change SHAPE, or did everything",
               "            shift a little? Those need different next moves."]),
    md(
        "## Step 4 — Freeze",
        "",
        "A hosted model is only *best-effort* reproducible, even at `temperature=0`. So "
        "once your best prompt is settled:",
        "",
        "1. Raise `N_PER_CLASS` in `config.py` to full size — and re-run notebooks 02 "
        "and 03 if that changes your sample. (If it does, you have more annotating to "
        "do. This is why you decide the size **before** you annotate.)",
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
        extra=["Freeze    : save, then load it straight back and use THAT from now on.",
               "            Reading it back is not superstition — it is the check that",
               "            the file you will report from is the file you think it is.",
               "Careful   : save f1_by_round too. Notebook 05 needs it, and it is the",
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
        "05", "05_report", "Error analysis and export",
        "Show an item the model got wrong, and say whose fault it was.",
        "the gold set (03) · the frozen predictions and rounds (04)",
        "`outputs/` — the predictions CSV, the report scaffold, a copy of your gold set",
        ["This is the highest-value part of the whole project, and the one the Q&A will "
         "definitely go to. A low F1 with a clear account of *why* is worth more than a "
         "high one without."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "from pipeline import *      # load_gold, load_predictions, load_json, ...",
        "from metrics import *       # evaluate, show_errors",
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
        extra=["Note      : nothing in this notebook calls the model. If a number here",
               "            differs from notebook 04, you are loading a different file —",
               "            not watching the model change its mind."]),
    step(
        2, "Score the frozen run",
        "the headline numbers, from the file rather than from a live run.",
        ["evaluate(gold, pred_final, ordered=..., labels=LABELS_ORDER)  ->  macro-F1"],
        "Day 2 S6 Part B · Day 3 — the identical call.",
        "the numbers for report section 3",
        extra=["Note      : evaluate prints per-class P/R/F1 and kappa as well as the",
               "            macro average. \"Which class is it worst at\" is a more useful",
               "            sentence than \"F1 = .62\" — read the table, not just the",
               "            headline.",
               "Careful   : `agreement()` is for comparing two ANNOTATORS (two lists of",
               "            labels), not gold against predictions. You used it in 03."]),
    md(
        "## Step 3 — Error analysis",
        "",
        "`show_errors` gives you every item the model got wrong. Two very different "
        "findings live in that table, and **your job is to say which is which**:",
        "",
        "- **The model's fault** — the label is clear, both your annotators agreed on "
        "it immediately, and the model still missed it.",
        "- **The scheme's fault** — the item is genuinely borderline. You know exactly "
        "which ones these are, because they are the rows you argued about in notebook "
        "03.",
        "",
        "So cross-reference: pull up your `disagreements` list from notebook 03 and see "
        "how much of it turns up here. **Overlap is a finding, not a failure.** If the "
        "model's errors cluster on the items your own annotators could not agree on, "
        "you have measured something real about the annotation scheme — and that is a "
        "better result than a clean F1.",
        "",
        "You could not make this argument without having built the gold set yourselves. "
        "That is why notebook 03 exists."),
    step(
        3, "Error analysis",
        "find the misses, and attribute each one.",
        ["show_errors(gold, pred_final)  ->  a table of the items it got wrong"],
        "Day 3, \"where is your best prompt still wrong?\" — identical call.",
        "errors",
        extra=["Note      : pick at least three and write a REASON for each — model's",
               "            fault or scheme's fault, and how you know. Report section 4.",
               "Try       : errors.head(15)  ·  errors[errors.gold == \"B2\"]",
               "Ask       : how many of these ids are in your notebook-03",
               "            disagreements? That number is worth reporting."]),
    md(
        "## Step 4 — Export",
        "",
        "Writes your gold set, a per-item predictions CSV, and a one-page report "
        "scaffold with the five required sections, all stamped with your group name.",
        "",
        "The scaffold fills in what it can compute — labels, counts, the F1-per-round "
        "table. The *italic* placeholders are yours: the QC narrative, the error "
        "attributions, and limitations that apply to **your** run rather than the "
        "generic three. A section left as the placeholder scores zero, so this is the "
        "start of the writing, not the end."),
    step(
        4, "Export",
        "write the gold set, the predictions CSV, and the report scaffold.",
        ["export_results(TRACK, gold, pred_final, f1_by_round, OUT_DIR, group=GROUP)"],
        "new — but it only writes down what you already have.",
        "three files in ../outputs/"),
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
