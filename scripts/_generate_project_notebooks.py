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

WHAT IS LEFT TO THE STUDENT, AND WHY
-----------------------------------
Nothing is blank. Every cell ships complete and runs on first execution, because a
group stranded on a path string or on the spelling of a global has learned nothing and
lost an afternoon.

What is left to them is the DECISIONS: which sampling strategy, where the band
boundaries go, which prompt version to carry into the held-out run, what each error was
caused by. The code makes a defensible choice and prints the result; the work is
reading that result and arguing about it. Leaving a value alone is a choice too, and
needs the same defence in the report and the Q&A as changing it.

WHERE THE EXPLANATION LIVES
---------------------------
In the markdown cell above each code cell - see `lead()` and `step()`. The comment
header on a step cell says only what the cell does and what it names. Anything longer
than a line belongs in the markdown, as sentences, and must not be said twice.
"""

import builtins
import inspect
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import pipeline
from _setup_cell import REPO, setup_lines, setup_md

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


def step(number, title, does, creates="", extra=(), starter=(), signpost=""):
    """A step cell: complete, runnable code, and a two-line header over it.

    The header says what the cell does when you run it, and what it leaves behind for
    later cells to use. Nothing else. Everything a student needs to KNOW - which day
    this was first run on, which file the function lives in, what to watch out for -
    goes in the markdown cell above, where it can be written as sentences.

    That split matters. These headers used to carry the explanation as well, back when
    the code below them had blanks in it, and the result was a wall of comment above
    every cell repeating what the markdown had just said.

    `starter` ships FILLED IN - every argument a real value, nothing left blank. An
    empty cell asks a beginner to remember the assignment form, the exact spelling of a
    global and the syntax around the call all at once, none of which is what the step
    is about. But a blank standing in for an argument is no better: with the answer in
    the comment beside it, it is a copying exercise, and with the answer withheld it is
    a guessing one.

    So the cell runs on first execution and PRINTS something, and the decision lives in
    what you do next: which of the three sampling strategies to keep, whether those
    band boundaries are yours or just the defaults, whether your labels are a scale.
    The markdown above says what the choice is and what turns on it; the report and the
    Q&A are where it gets defended. Leaving a value alone is a choice as well, and it
    needs the same defence as changing it.

    `creates` is left empty on a step that names nothing new - a check, or one that
    only writes files. Say what it wrote in the markdown instead.
    """
    rule = "═" * max(4, 62 - len(title) - len(str(number)))
    lines = ["# ══ STEP " + str(number) + " · " + title + " " + rule]
    lines.extend("# " + line for line in does)
    if creates:
        lines.append("# Creates: " + creates)
    for line in extra:
        lines.append("# " + line)
    lines.append("")
    if starter:
        lines.append("# ✏️ this runs as written — the work is deciding whether it should")
        lines.append("")
        for line in starter:
            lines.append(line)
    else:
        lines.append("# ✏️ your code here")
    lines.append("")
    if signpost:
        return [lead(signpost), code(*lines)]
    return [code(*lines)]


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
        # `why` is prose. It goes into a CODE cell, so every line of it has to be a
        # comment - a bare sentence here is a SyntaxError in the student's notebook.
        commented = []
        for line in textwrap.wrap(why.lstrip("# "), width=86):
            commented.append("# " + line)
        blocks.append("\n".join(commented))
    if imports:
        blocks.append("\n".join(imports))
    for obj in objects:
        blocks.append(inspect.getsource(obj).rstrip("\n"))
    return code(*("\n\n".join(blocks)).split("\n"))


def lead(*lines):
    """A markdown signpost immediately above a code cell: what we are about to do.

    Every code cell in these notebooks has one. A cell a student meets with no idea
    what it is for is a cell they run and scroll past.
    """
    return md(*lines)


def source_cells(described, imports=()):
    """One cell per embedded function, each with its own one-line signpost.

    `described` is a list of (object, sentence) pairs. Splitting them up matters: the
    single cells these replaced ran to 140 and 180 lines and defined half a dozen
    unrelated things each, which is a wall rather than something anyone reads.
    """
    cells = []
    if imports:
        cells.append(lead("First, what the code below needs loaded."))
        cells.append(code(*imports))
    for obj, sentence in described:
        cells.append(lead(sentence))
        cells.append(code(*inspect.getsource(obj).rstrip("\n").split("\n")))
    return cells


MODULE_ALIAS = {"pandas": "pd", "numpy": "np"}


def _preamble(objects):
    """The import lines a block of redefined functions needs to be able to run.

    A function defined in a notebook cell looks its globals up in the NOTEBOOK, not
    in the module it was copied from, so every name it calls has to be there. Working
    that list out here rather than typing it at each call site means an edit to
    pipeline.py that adds a helper cannot quietly leave a student with a NameError:
    it either shows up in the preamble at the next generation, or generation stops.
    """
    defined = {obj.__name__ for obj in objects}
    from_module = {}                                  # module name -> {names}
    modules = {}                                      # import name -> alias
    for obj in objects:
        globals_ = obj.__globals__
        for name in obj.__code__.co_names:
            if name in defined or hasattr(builtins, name):
                continue
            if name not in globals_:
                # An attribute (`item["label"]`, `path.name`) rather than a global.
                # co_names holds both, and only the globals need importing.
                continue
            value = globals_[name]
            if inspect.ismodule(value):
                alias = MODULE_ALIAS.get(value.__name__, None)
                modules[value.__name__] = alias if name == alias else None
            else:
                # Import it from the module the function itself lives in, whether it
                # was defined there (`reid`) or imported into it from somewhere else
                # (sklearn's `f1_score`). That is the module whose namespace the
                # function was written against, so it is the one that must match.
                home = obj.__globals__["__name__"]
                if not hasattr(sys.modules[home], name):
                    raise RuntimeError(
                        "_preamble cannot work out where to import " + repr(name)
                        + ", which " + obj.__name__ + " calls. Add it to "
                        "MODULE_ALIAS, or import it into " + home + ".")
                from_module.setdefault(home, set()).add(name)

    lines = []
    for module_name in sorted(modules):
        alias = modules[module_name]
        if alias:
            lines.append("import " + module_name + " as " + alias)
        else:
            lines.append("import " + module_name)
    for module_name in sorted(from_module):
        names = ", ".join(sorted(from_module[module_name]))
        lines.append("from " + module_name + " import " + names)
    return lines


def study_cells(what, described, check=None):
    """The functions a group has to defend, as runnable cells ABOVE their first use.

    These used to be rendered after the call that used them, as markdown, which put
    the reader in an odd position: the sample had already been drawn by the time they
    saw what drew it, and editing the source they were reading changed nothing,
    because the name still came from the SETUP import.

    So now the block comes first and the cells are real definitions. The names are no
    longer imported in SETUP (only the plumbing is), which means these cells are where
    the functions come from - edit one, run it, and the step below behaves differently.

    Layout, for readers who met Python a few days ago: a markdown header, one small
    cell of imports the definitions need, then ONE function per cell, each with its
    own signpost. Source is still read out of scripts/ by `inspect.getsource`, so it
    cannot drift from what the rest of the pipeline runs.

    `described` is a list of (object, sentence) pairs; `check` is an optional one-line
    cell - `show(sample)` - to confirm the definitions took.
    """
    objects = [obj for obj, _ in described]
    cells = [md(what),
             lead("First, the helpers the definitions below call. Nothing to decide "
                  "here — run it and read on."),
             code(*_preamble(objects))]
    for obj, sentence in described:
        cells.append(lead(sentence))
        cells.append(code(*inspect.getsource(obj).rstrip("\n").split("\n")))
    if check:
        cells.append(lead("Run this to check the definitions above took effect. It "
                          "prints the file, the line and the source of the function "
                          "the notebook will actually use."))
        cells.append(code(check))
    return cells


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


# `save_json` as the example because all four of these notebooks import it.
SETUP_MD = setup_md("save_json")


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
        "# it is not imported at all — you define it further down, in cells you can",
        "# read and change, just before the step that calls it.",
        "# `show` prints the source of any of these: show(save_json)",
        "from pipeline import load_gold, save_json, label_set, show",
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
    lead("First we open the pool file and count what is in it. `load_gold` is the same "
         "call you ran on Day 2 S5 step F and again at the start of Day 3; it lives in "
         "`scripts/pipeline.py`. The only decision here is which of the two paths to "
         "open."),
    *step(
        1, "Load the pool",
        ["Opens the pool file notebook 01 wrote and counts the items under each label."],
        "pool",
        starter=[
            "pool = load_gold(POOL_PATH)     # or DEMO_POOL_PATH to try it out first",
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
    *study_cells(
        "### The code that draws the sample\n\n"
        "This is your **sampling method** — what report section 1 has to describe and "
        "the Q&A may well ask you to defend, so it is in front of you rather than "
        "behind an import. SETUP did not import these four: **the cells below are "
        "where they come from**, so run them before the step that uses them. They are "
        "read out of `scripts/pipeline.py` when this notebook is generated, so they "
        "are not a simplified copy — they are the code that runs.\n\n"
        "They are ordinary definitions, which means you can change one. Edit a cell, "
        "run it again, and the draw below behaves differently. (To get the original "
        "back, re-run the notebook from the top.)\n\n"
        "**Where the reproducibility comes from**, in all three strategies: "
        "`random.Random(seed)` — one generator, made from your seed, doing every "
        "shuffle. Same seed, same draw, on anyone's machine. That one line is the "
        "whole of why your sample is checkable by someone else.",
        [(pipeline.sample,
          "**`sample`** is the one you call, and the one word you change is its "
          "`strategy`. In: the pool, a strategy name, `n_per_class`, the seed. Out: "
          "the list of sampled items. It sizes all three strategies from the same "
          "`n_per_class`, so switching changes *how* the items are chosen and not "
          "*how many* — which is what makes their counts comparable. It does no "
          "drawing itself; it hands the work to one of the next three."),
         (pipeline.sample_pool,
          "**`sample_pool` — balanced across labels.** In: the pool and how many items "
          "you want per label. Out: up to that many of each. Step 1 sorts the pool "
          "into one bucket per label; step 2 takes up to `n_per_class` from each. A "
          "label with fewer than that gives all it has, which is why a rare class "
          "comes back short — that is data, not a bug."),
         (pipeline.sample_random,
          "**`sample_random` — the corpus as it is.** In: the pool and one total. Out: "
          "that many items, drawn without looking at the labels at all, so the draw "
          "keeps the pool's own imbalance. Realistic, and unkind to rare labels."),
         (pipeline.sample_by_document,
          "**`sample_by_document` — whole passages** (`cars50` and `raamove` only). "
          "In: the pool, how many documents, how many sentences from each. Out: that "
          "many sentences, drawn from that many documents. It is the longest of the "
          "four, and most of the length is the check it opens with: on a track whose "
          "items are loose sentences there are no documents to stratify by, so it "
          "stops and says so rather than inventing an answer.")],
        check="show(sample)"),
    *step(
        2, "Draw your sample",
        ["Draws the sample using whichever of the three strategies you name, and prints",
         "how many items landed under each label."],
        "sampled",
        signpost="Now we draw the sample. The cell runs as written, using the balanced "
                 "strategy — `sample_pool` is the one you ran on Day 4 Part A. The work "
                 "is deciding whether that is the strategy your study wants, and being "
                 "able to say why.\n\n"
                 "Every strategy prints its per-label counts. Run more than one and "
                 "compare: the difference between them is the argument you have to make "
                 "in your report.\n\n"
                 "All three are sized from `n_per_class` in `config.yaml` and take "
                 "`SEED`, both already passed — so switching strategy changes how the "
                 "items are chosen, not how many. A sample nobody can redraw is a "
                 "sample nobody can check, and your report has to state the seed.\n\n"
                 "**If it warns that you took most of the pool**, you are almost "
                 "certainly still pointed at `DEMO_POOL_PATH`.",
        starter=[
            "# Balanced is the DEFAULT, not the recommendation. Change this one word to",
            "# try another view of the corpus, and compare the counts each one prints.",
            "#",
            "# It is written as a choice rather than three lines you comment two of out,",
            "# because with two live lines the SECOND one silently wins.",
            "#",
            "# by_document is cars50 · raamove ONLY. The other tracks have no documents",
            "# to stratify by, and it will stop and tell you so.",
            'STRATEGY = "balanced"     # "balanced" · "random" · "by_document"',
            "",
            "sampled = sample(pool, STRATEGY, N_PER_CLASS, SEED)",
        ]),
    lead("Now we note which labels came out of that draw. `LABELS` is used again in "
         "step 4 below, and in notebook 03."),
    code("LABELS = label_set(sampled)",
         'print("labels:", LABELS)'),
    md("### Now write down why you drew it that way",
       "",
       "Not in the notebook — in `PLAN.md` §5, in a sentence. It is report section 1, "
       "and the Q&A may well ask you to defend it, so write it while the reason is "
       "still in your head.",
       "",
       "The shape of the answer (another track's, so it is not yours to copy):",
       "",
       "> We sampled by document, 10 documents × 4 sentences, because a move label "
       "describes a sentence's job *within* its abstract, and 40 loose sentences from "
       "40 different papers would have thrown that away."),
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
    *step(
        3, "Check the draw",
        ["Prints the size of the pool, the size of the sample, what is left over, and",
         "the count under each label. Nothing new is named — this is a check."],
        starter=[
            "# Count how many sampled items carry each label, one item at a time.",
            "counts = {}",
            "for item in sampled:",
            '    label = item["label"]',
            "    if label not in counts:",
            "        counts[label] = 0",
            "    counts[label] = counts[label] + 1",
            "",
            'print("pool:", len(pool))',
            'print("sampled:", len(sampled))',
            'print("left over for few-shot examples:", len(pool) - len(sampled))',
            'print("per label:", counts)',
        ]),
    md("### How many distinct texts is this? — `cars50` · `raamove` only",
       "",
       "Now we count the documents your sentences came from. Skip this cell on the "
       "other tracks: their items are loose sentences with no document attached, so "
       "there is nothing to count.",
       "",
       "Forty sentences from four introductions is a much narrower claim than forty "
       "from forty, and the difference belongs in your limitations section."),
    code("# `doc_id` is one of the extra fields the move tracks carry. Collect the",
         "# distinct ones, one item at a time.",
         "documents = []",
         "for item in sampled:",
         '    doc_id = item.get("doc_id")',
         "    if doc_id is not None and doc_id not in documents:",
         "        documents.append(doc_id)",
         "",
         "if documents:",
         '    print("drawn from", len(documents), "distinct documents")',
         "else:",
         '    print("this track carries no doc_id — nothing to count here")'),
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
        "in this cell's output, where a runtime reset can lose it.",
        "",
        "This is the Day 2 S5 step A call with those two sharing arguments added; it "
        "lives in `scripts/annotate.py`. The title is built for you out of your track, "
        "group and run, because you will have several of these sheets by the end of the "
        "week.",
        "",
        "**Run this cell once.** Running it a second time makes a second sheet, and half "
        "your annotations end up in the one nobody reads back."),
    *step(
        4, "Create the sheet",
        ["Makes a Google Sheet with one row per sampled item and blank coder columns,",
         "shares it with your group, and writes the link to a file."],
        starter=[
            'title = TRACK + " · " + GROUP + " · " + RUN + " annotation"',
            "",
            "# Run this ONCE — a second run makes a second sheet.",
            "url = create_annotation_sheet(title, sampled, LABELS,",
            "                              share_with=MEMBERS,   # your group, from config.yaml",
            "                              remember=SHEET_PATH)  # writes the link to a file",
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
        "`data/gold/<track>_<group>_gold.json`, and its `_dev.json` / `_test.json` split",
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
         "project can tell you that, and notebook 05 asks you for it directly.",
         "",
         "It ends by drawing one more line: which of your annotated items you are allowed "
         "to *look at* while you write prompts, and which are held back for the number you "
         "report. One sheet, one adjudication, then a split — it costs no extra coding."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# The Google Sheets round trip is plumbing, so it is imported. The judgment it",
        "# exists to support is not in any of these files. The dev/test split is not",
        "# imported either — you define it further down, just before you use it.",
        "# `show` prints the source of any of these: show(save_json)",
        "from pipeline import load_gold, label_set, save_json, show",
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
    md(
        "## Then — the annotation sheet, found again",
        "",
        "Now we look up the sheet notebook 02 created, from the small file it wrote the "
        "link to. That file is why the link is not lost: whoever runs this notebook need "
        "not be the person who ran notebook 02, and need not still have that cell's "
        "output on screen.",
        "",
        "If this prints *none saved yet*, notebook 02's sheet step has not been run — or "
        "was run by somebody whose copy of the folder is not this one."),
    code(
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
        "you can explain beats a high one you cannot.",
        "",
        "These are the Day 2 S5 D–E calls, with the coder names added; they live in "
        "`scripts/annotate.py`. Run the cell once **every** coder's tab is filled in — "
        "rows that not everyone labelled are dropped from the comparison.",
        "",
        "**If it warns that two coders gave every item the same label**, somebody "
        "duplicated a tab that had already been filled in. That agreement is a copy "
        "rather than a measurement, and it has to be fixed before you report anything."),
    *step(
        1, "Measure agreement",
        ["Reads one tab per coder, lines them up by item id, and prints how often you",
         "agreed, corrected for chance."],
        "rows",
        starter=[
            "# CODERS comes from config.yaml, so notebook 05 finds the same tabs. Add a",
            "# third name there if a third person joined.",
            "rows = load_coder_sheets(SHEET_ID, CODERS)       # one read per tab, merged by ID",
            "",
            "annotator_agreement(rows, coders=CODERS)         # prints agreement and κ",
        ]),
    md("### Now the rows you have to talk about",
       "",
       "`disagreements` hands back a table of the rows your coders labelled "
       "differently. That is your adjudication list for step 2.",
       "",
       "The last line of the cell is just the name `disagreed`, with no `print`. In a "
       "notebook, the value of the last line of a cell is displayed automatically — and "
       "for a table that display is much easier to read than `print` would give you. "
       "Add a line after it and the table stops appearing, which is the one thing to "
       "watch out for.",
       "",
       "The table is saved as well as shown. It comes back in notebook 05, where the "
       "rows your coders argued about are what you check the model's errors against — "
       "and saving it here means 05 does not have to sign back in to the sheet and "
       "derive the same table a second time. It also means that step still works after "
       "the sheet has been deleted, or its owner has left."),
    code("disagreed = disagreements(rows, coders=CODERS)",
         'save_json(disagreed.to_dict("records"), DISAGREED_PATH,',
         '          what="rows your coders disagreed on")',
         "disagreed"),
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
        "re-run until it says **0 blank, 0 invalid**. A blank row is an item that has "
        "gone missing from your study without telling you.",
        "",
        "The cell re-reads the sheet first, because `rows` from step 1 was fetched "
        "before you filled `Final` in. And it passes `source=sampled`: gold is rebuilt "
        "from the **sheet**, which carries only the id, the text and your label, so "
        "anything else the item had — on `cars50` and `raamove`, its passage — is put "
        "back from `sampled` by id. On the other tracks that argument does nothing.",
        "",
        "Both calls are the Day 2 S5 step F ones. `to_canonical` is in "
        "`scripts/annotate.py`."),
    *step(
        2, "Adjudicate, then canonicalise",
        ["Re-reads the sheet now that Final is filled in, and turns it into your gold",
         "set — reporting any row that is blank or has a label it does not recognise."],
        "gold",
        starter=[
            "# Re-read: `rows` from step 1 was fetched before you filled in Final.",
            "rows = load_coder_sheets(SHEET_ID, CODERS)",
            "",
            "gold = to_canonical(rows, LABELS, source=sampled)   # re-attaches what the sheet drops",
        ]),
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
        "sentence worth saying out loud in the Q&A. Pick two or three rows and write "
        "down which of the three cases above they are — now, while you still remember "
        "the argument you had about them.",
        "",
        "The comparison runs against `sampled`, not `pool`: sampling renumbered the "
        "ids, so `pool` would line your item 7 up against a completely different "
        "sentence. This is the Day 2 S5 step F call; it is in `scripts/annotate.py`."),
    *step(
        3, "Compare against the published labels",
        ["Shows every row where your group's label and the corpus's label differ."],
        "differences",
        starter=[
            "differences = compare_to_published(gold, sampled)   # sampled, not pool: same 40 items",
            "differences",
        ]),
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
        "## Step 4 — Draw the line: dev and test",
        "",
        "In notebook 04 you will change your prompt because of what you saw it get wrong. "
        "That is the work. But a score measured on the same items you kept adjusting "
        "against stops being a measure of how good your prompt is, and becomes a measure "
        "of **how long you kept adjusting**. It only ever goes up.",
        "",
        "So the line gets drawn now, before anything has been run against these items:",
        "",
        "| | what it is for |",
        "|---|---|",
        "| **dev** | the items you may look at. Iterate here, as many rounds as you like. |",
        "| **test** | opened once, in the last step of notebook 04. Whatever it says is "
        "what you report. |",
        "",
        "Both halves came out of the same sheet and the same adjudication, so the split "
        "costs you no extra annotation. What it costs is items you are allowed to learn "
        "from — which is why the ratio is a real decision and `PLAN.md` §6 asks you to "
        "defend the one you made. A bigger dev gives steadier feedback while you iterate "
        "and leaves a smaller test, so the number you finally report bounces more; a "
        "smaller dev means prompt decisions made on very few items, which is how you tune "
        "to noise and then watch the gain evaporate.",
        "",
        "This also replaces the old advice to keep `n_per_class` at 2 while iterating. "
        "**dev is the fast set now** — a dozen or so items is about a minute per round, "
        "and your sample stays at full size throughout.",
        "",
        "The split is stratified by label, so both halves keep every label wherever the "
        "data allows. Where it does not — a label with a single surviving item — that "
        "item goes to **test**, and the function says so. That asymmetry is deliberate: a "
        "label missing from test drops out of the macro average without announcing "
        "itself, while a label missing from dev only costs you feedback."),
    *study_cells(
        "### The code that draws the line\n\n"
        "SETUP did not import these two: **the cells below are where they come from**, "
        "so run them before the step that uses them. They are read out of "
        "`scripts/pipeline.py` when this notebook is generated, so they are the code "
        "that runs — and they are ordinary definitions, so if you change one and run "
        "the cell again, the split below changes with it.\n\n"
        "Three things to look for as you read: the rounding rule for a fractional "
        "`dev` is written out rather than left to `round()` (Python rounds 0.5 down "
        "and 1.5 up, and neither is something you want to have to explain in the "
        "Q&A); the rare-class clamp, and which side it favours; and the ids are "
        "**not** renumbered, because notebook 05 asks which of the model's errors are "
        "also the rows your two coders argued about, and that join runs on these ids.",
        [(pipeline.split_dev_test,
          "**`split_dev_test`** draws the line. In: your gold items, and the `dev:` "
          "setting from `config.yaml`. Out: two lists, dev and test. This is the whole "
          "of the discipline: the bookkeeping that decides whether the number in your "
          "report means anything."),
         (pipeline._read_dev_size,
          "**`_read_dev_size`** is the small function the one above calls first. In: "
          "whatever `dev:` says. Out: either a count per label or a proportion — the "
          "type of the number carries the decision, which is why one `dev:` setting "
          "can mean two things and there is no second config key to keep consistent "
          "with the first.")],
        check="show(split_dev_test)"),
    *step(
        4, "Split dev / test",
        ["Splits your gold set in two, keeping every label on both sides wherever the",
         "data allows, and prints how many items each half got."],
        "dev, test",
        signpost="Now we draw the line: which of your gold items you are allowed to "
                 "look at while iterating, and which you are not. Nothing here existed "
                 "in Days 1–3 — no set there was worth holding back. `split_dev_test` "
                 "is the function you defined and read just above.\n\n"
                 "**Run this once, and before you open notebook 04.** Splitting again "
                 "after you have iterated on dev means the held-out items have already "
                 "been seen — by you, if not by the model.\n\n"
                 "How big dev is comes from `dev:` in `config.yaml`, and how you "
                 "wrote the number says what you meant. A balanced draw "
                 "(`sample_pool`) suits a whole number, `dev: 3` — three items per "
                 "label. An uneven one (`sample_random`) suits a decimal, "
                 "`dev: 0.35` — a third of each label, because a fixed 3 per class "
                 "would eat a small class whole.\n\n"
                 "Nothing is saved yet — read the counts it prints first.",
        starter=[
            "# DEV comes from config.yaml: a whole number is items per label, a decimal",
            "# is a proportion of each label.",
            "#",
            "# Drew your sample with sample_by_document? Add by_document=True inside the",
            "# brackets below, so that no passage has some of its sentences in dev and",
            "# the rest in test.",
            "dev, test = split_dev_test(gold, DEV, seed=SEED)",
        ]),
    md("### Now save both halves",
       "",
       "Read the per-label counts the split just printed **before** you run this. A "
       "label that lands in dev but not in test cannot appear in the score you report, "
       "and this is the last easy moment to change `dev` in `config.yaml` and "
       "draw the line again.",
       "",
       "Once you are happy, save. Notebook 04 opens `dev`; notebook 05 opens `test`."),
    code('save_json(dev,  DEV_PATH,  what="dev items")',
         'save_json(test, TEST_PATH, what="test items")'),
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
        "that `PLAN.md` records **which sampling strategy you chose, and why**, and "
        "**§6: which split spec you set, the sizes it produced, and why that ratio**."),
]

# ==================================================================================
# 04 — prompt
# ==================================================================================
cells_04 = [
    title_cell(
        "04", "04_prompt", "Baseline, read the errors, iterate, freeze",
        "Write the plainest prompt that could work, then improve it for reasons you can "
        "state.",
        "`data/gold/<track>_<group>_dev.json` (from 03) · the pool (from 01)",
        "`outputs/<track>_<group>_predictions.json` · `..._rounds.json` · "
        "`..._test_log.jsonl`",
        ["Everything from here on is measured against **your** gold set, not the "
         "corpus's labels. That is the point of the last two notebooks.",
         "",
         "Steps 1–3 work on your **dev** half. The test half is not opened until step 4, "
         "and then only once.",
         "",
         "> **Free-tier pacing.** The backend waits a few seconds between calls and "
         "retries on rate-limit errors, so a run takes minutes and may print "
         "`(rate limited - waiting Ns then retrying)`. That is normal — and it is why "
         "you iterate on dev: a dozen or so items is about a minute per round, so you "
         "get enough rounds to actually learn something. Your sample stays at full size "
         "throughout; there is no longer a small-while-you-iterate phase.",
         "",
         "> **On the size of this study.** One call per item, four-and-a-bit seconds "
         "apart, no batching: forty items is minutes and four hundred is most of an "
         "afternoon of a quota you share with everyone else on the course. A study that "
         "could support a claim about a corpus needs hundreds of items per class. This "
         "one cannot, and that is a limitation to state in report §5 rather than write "
         "around. What transfers is the method — the split, the freezing, the audit "
         "trail — not the number."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Files in, files out, and the connection to the model: all plumbing.",
        "# `show` prints the source of any of these: show(save_json)",
        "from pipeline import (load_gold, label_set, load_prompt, save_json,",
        "                      load_predictions, setup, freeze_test_run, show)",
        "",
        "# Asking the model and scoring the answers is what this notebook is FOR, so",
        "# those five functions are not imported here at all. You define them yourself,",
        "# in step 2, in cells you can read and change, just before the first round.",
    ]),
    md(
        "## Connect to the model",
        "",
        "Now we open the connection this notebook will send every prompt through. It "
        "gets a cell of its own because it does something the plumbing cell above does "
        "not: it reaches out to a service, and what it prints back is worth reading.",
        "",
        "Safe to run more than once — after the first time it hands back the connection "
        "it already made."),
    code("setup()"),
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
    md("## Step 1 — Open the three files this notebook works from",
       "",
       "Now we load the dev half (what you may look at), the pool (where few-shot "
       "examples come from) and the **full** gold set.",
       "",
       "That third one looks redundant and is not. `build_fewshot` excludes your gold "
       "items from the examples it picks — by *text*, since sampling renumbered the "
       "ids. Hand it only `dev` and it can pick a **test** item as a worked example, "
       "which puts the answer to a held-out item straight into the prompt that produces "
       "your headline number. So the full gold set is loaded as an exclusion list, and "
       "scored against never.",
       "",
       "`LABELS` is read off the full gold set for the same reason: a label that "
       "happens to be thin in dev should not quietly shrink your label list.",
       "",
       "`TEST_PATH` is not opened here. It is opened once, in step 4. All three of "
       "these are the `load_gold` call from the Day 3 setup."),
    *step(
        1, "Load your dev set and your pool",
        ["Opens the three files this notebook works from, and reads your label list",
         "off the full gold set."],
        "dev, pool, gold, LABELS",
        starter=[
            "dev  = load_gold(DEV_PATH)      # what you iterate against, from notebook 03",
            "pool = load_gold(POOL_PATH)     # the spares few-shot examples are drawn from",
            "gold = load_gold(GOLD_PATH)     # ALL of it — as an EXCLUSION list, see above",
            "LABELS = label_set(gold)        # gold, not pool: what you actually adjudicated",
            "",
            'print(len(dev), "dev ·", len(pool), "pool ·", LABELS)',
        ]),
    md(
        "## Step 2 — The baseline (round 0)",
        "",
        "Your first score, before you have changed anything. Write the plainest prompt "
        "that states the task and the label set, run it, score it. **Resist the urge to "
        "make it good** — later rounds need something to be measured against, and a "
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
        "```",
        "",
        "The three cells below are the Day 3 Part A run, plus the error table. Every "
        "number in this step and the next is a **dev** number: you use it to decide "
        "what to change next, not to report.",
        "",
        "`f1_by_round` collects one score per round, keyed by the round's name. "
        "Notebook 05 reads it back from a file and prints it into your report, so those "
        "keys are what your reader sees — name them so they mean something."),
    *step(
        2, "Baseline prompt (round 0)",
        ["Starts the table of per-round scores, and loads the starting prompt for your",
         "track so you can read it before it runs."],
        "f1_by_round, PROMPT",
        starter=[
            "# One entry per round from here on. Notebook 05 turns it into your table.",
            "f1_by_round = {}",
            "",
            "PROMPT = load_prompt(PROMPT_FILE)        # the starting prompt for your track",
            "print(PROMPT)",
        ]),
    *study_cells(
        "### The five functions that produce your numbers\n\n"
        "Every number in your report comes out of these five, so read them before you "
        "quote them. SETUP did not import them: **the cells below are where they come "
        "from**, so run them before the steps that use them. They are read out of "
        "`scripts/` when this notebook is generated — not a simplified copy.\n\n"
        "They are ordinary definitions. Change one, run the cell again, and the rounds "
        "below use your version. That is worth knowing about `extract_label` in "
        "particular: if your model keeps answering in a shape it cannot read, this is "
        "where you would fix that.\n\n"
        "Five cells is more reading than the earlier notebooks asked for. Take them "
        "one at a time — each one runs on its own, and none of them calls the model.",
        [(metrics.show_errors,
          "**`show_errors` is the one you will actually iterate on.** In: your dev "
          "items and the model's predictions. Out: a `DataFrame` of just the rows "
          "where the two differ — which is why you can filter it with "
          "`errors[errors.gold == \"…\"]`. F1 tells you *whether* a round helped; only "
          "the errors tell you *what to change next*."),
         (pipeline.extract_label,
          "**`extract_label` is doing more than it looks.** In: one reply from the "
          "model, in prose. Out: one label. This is what decides that *\"This looks "
          "like Move 2 to me\"* means `Move 2` — it searches for label names, keeps "
          "the longest match, and falls back to `\"??\"`. Every `??` in your run is a "
          "reply it could not read, and if there are many, that is a finding about "
          "your prompt, not a bug."),
         (pipeline.run_prompt,
          "**`run_prompt` is the loop.** In: your prompt and a list of items. Out: one "
          "predicted label per item. One API call each, the reply passed through "
          "`extract_label`. The pacing and retrying happen inside `_default_backend`, "
          "which is the connection `setup()` opened — that part is plumbing, and it "
          "stays imported."),
         (pipeline.build_fewshot,
          "**`build_fewshot` puts worked examples in front of the model.** In: your "
          "prompt and the pool. Out: the same prompt with a few solved items added to "
          "it. It skips anything in your gold set — matched by text, because sampling "
          "renumbered the ids. Without that skip you would be testing the model on "
          "answers you had just shown it. You use it in step 3."),
         (metrics.evaluate,
          "**`evaluate`** prints per-class precision/recall/F1, Cohen's κ and the "
          "confusion matrix — and **returns the macro-F1 as a number**, which is what "
          "lets you collect one per round. Macro-F1 is the plain average of the "
          "per-class scores, so every class counts the same however rare it is; that "
          "is why a balanced sample and a macro average go together. `ordered=True` "
          "adds a weighted κ, which counts a near miss as a smaller error than a far "
          "one — use it only if your labels sit on a scale, and pass "
          "`labels=LABELS_ORDER` so it knows what that scale is.")],
        check="show(run_prompt)"),
    md("### Now send it to the model",
       "",
       "This is the slow cell: one API call per dev item, paced a few seconds apart to "
       "stay inside the free tier. Forty items takes a couple of minutes.",
       "",
       "It is on its own **deliberately**. Scoring and reading the errors are separate "
       "cells below, so that looking at your results again costs you nothing. If they "
       "shared a cell with this one, every re-read would re-run every call and spend "
       "your group's quota a second time."),
    code("pred0 = run_prompt(PROMPT, dev)"),
    md("### Now score it",
       "",
       "`evaluate` prints the table and the matrix, and hands back the macro-F1, which "
       "we store under a name we choose.",
       "",
       "`ordered=False` is the safe default. Change it to `True` if — and only if — "
       "your labels sit on a **scale** (A1 < A2 < … < C2, Low < Mid < High), where a "
       "near miss is a smaller error than a far one. Move 1 / Move 2 / Move 3 are *not* "
       "a scale: they are three different jobs, not three amounts."),
    code('f1_by_round["round0 baseline (dev)"] = evaluate(dev, pred0,',
         "                                               ordered=False,",
         "                                               labels=LABELS_ORDER)"),
    md("### Now read what it got wrong",
       "",
       "Do not skip this. The errors are the only thing that tells you *what to change*; "
       "F1 only tells you afterwards whether the change worked. This is the cell that "
       "decides your next round."),
    code("show_errors(dev, pred0)"),
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
        "shown it. It replaces typing the examples out by hand, which is how you did it "
        "in the Day 3 iterations.",
        "",
        "**Save each version as its own prompt file** (`v0`, `v1`, `v2`). A prompt you "
        "overwrote is a round you cannot report.",
        "",
        "One question to ask after each round: did the confusion matrix change **shape**, "
        "or did every cell shift a little? Those two call for different next moves."),
    *step(
        3, "Round 1 — build a new prompt",
        ["Adds worked examples to the prompt you already have, and prints the result",
         "so you can read it before you spend a round on it."],
        "PROMPT_v1",
        starter=[
            "# Either add examples to the prompt you already have …",
            "#",
            "# `gold`, not `dev`: it is the list of items NOT to use as examples, and",
            "# that has to include the test half, or the answers leak into the prompt.",
            "#",
            "# Note the NEW NAME on the left. Writing `PROMPT = build_fewshot(PROMPT, ...)`",
            "# would add examples to the prompt that already has them every time you",
            "# re-ran this cell, silently, and you would not see it in the numbers.",
            "PROMPT_v1 = build_fewshot(PROMPT, pool, gold)",
            "",
            "# … or write a new prompt file (see the %%writefile example above) and load",
            "# that instead:",
            '# PROMPT_v1 = load_prompt(ROOT / "prompts" / "my_prompt_v1.txt")',
            "",
            "print(PROMPT_v1)",
        ]),
    md("### Now run this round",
       "",
       "Again on its own, and again for the same reason: one API call per dev item, so "
       "a re-run costs your group real quota."),
    code("pred1 = run_prompt(PROMPT_v1, dev)"),
    md("### Now score it, and read the new errors",
       "",
       "Name the key for what you **changed**, not which round it was. In the report, "
       "*\"round1 four examples per label\"* is an argument; *\"round1\"* is a row "
       "number.",
       "",
       "Keep `ordered` the same as the baseline, or the two rounds are not comparable."),
    code('f1_by_round["round1 four examples per label (dev)"] = evaluate(',
         "    dev, pred1, ordered=False, labels=LABELS_ORDER)"),
    md("Now the errors again — these are what design round 2."),
    code("show_errors(dev, pred1)"),
    md("### Then do it again",
       "",
       "For round 2, copy the **three cells above** — change the prompt, run it, score "
       "it — and rename `PROMPT_v1` / `pred1` to `PROMPT_v2` / `pred2` as you go, along "
       "with the key in `f1_by_round`. Reusing a name is how a round silently scores "
       "the round before it and reports an identical F1.",
       "",
       "Whichever version wins on dev is the one you carry into step 4."),
    md(
        "## Step 4 — The held-out run",
        "",
        "This is the first time all week that `TEST_PATH` gets opened. Your prompt is "
        "settled, you run it once against items it has never been tuned against, and "
        "**whatever comes out is what you report.**",
        "",
        "Expect it to be lower than your best dev round. That is the normal result, not a "
        "failure and not a sign you did something wrong: the gap is roughly how much of "
        "your improvement was tuning to those particular dev items rather than to the "
        "task. Reporting the gap is a stronger finding than reporting a high number, and "
        "notebook 05 asks you for it.",
        "",
        "A hosted model is only *best-effort* reproducible even at `temperature=0`, so "
        "the run is frozen to a file and every number in notebook 05 comes out of that "
        "file rather than out of this session's memory.",
        "",
        "**Nothing here stops you running it twice.** Stopping you would be the wrong "
        "design — a genuine mistake at four o'clock on the last day needs a way forward. "
        "So instead: nothing is overwritten (a second run lands in "
        "`..._predictions_attempt2.json`), and every scoring appends a line to a log that "
        "goes into your submission. A second attempt is allowed. It is just not invisible, "
        "and §5 of your report has to account for it.",
        "",
        "**One person runs this.** It is the run you will be defending."),
    *step(
        4, "Open the test set and pick your prompt",
        ["Opens the held-out items — the only cell all week that does — and notes your",
         "best dev score before the test row joins the table."],
        "test, FINAL_PROMPT, best_dev",
        starter=[
            "test = load_gold(TEST_PATH)      # the first time this file is opened all week",
            "",
            "# The prompt that won on dev. Change this to PROMPT_v2, or whichever round",
            "# came out best — it is the one you are about to be judged on.",
            "FINAL_PROMPT = PROMPT_v1",
            "",
            "# Your best DEV round, noted BEFORE the test row joins the table. The gap",
            "# between this and what you are about to get is a finding in its own right.",
            "best_dev = max(f1_by_round.values())",
            'print("best dev round:", round(best_dev, 3))',
        ],
        signpost=""),
    md(
        "### Now the run itself",
        "",
        "**One cell, one call, and it is the one you will be defending.** It is on its "
        "own so that nothing else in this notebook can fail *after* it and leave you "
        "re-running it — every re-run is another line in the log and another attempt "
        "number to account for in report §5.",
        "",
        "`freeze_test_run` does five things, and it does them in one cell precisely "
        "because stopping halfway through them is the failure worth preventing — a log "
        "that disagrees with the predictions file is worse than either alone. Watch "
        "them go past as it runs:",
        "",
        "1. Runs `FINAL_PROMPT` over the test items.",
        "2. **Saves before scoring anything.** It never overwrites: a second run lands "
        "in `..._predictions_attempt2.json` beside the first, and both stay.",
        "3. Reads that file straight back off disk and scores *those* predictions, "
        "which checks that the file you will quote in your report is the file you "
        "think it is.",
        "4. Appends one line to the log that travels in your submission bundle: the "
        "score, which attempt it was, and a fingerprint of the prompt that produced "
        "it.",
        "5. Adds the test score to `f1_by_round` **last**, so the table in your report "
        "reads as the dev rounds in order with the held-out score at the bottom, and "
        "saves that table for notebook 05.",
        "",
        "The four file names it is handed are the four files it touches. `note=` is "
        "the only argument that is a decision: a second attempt with the **same** "
        "prompt is that prompt run twice; a second attempt with a **different** one is "
        "a prompt tuned after seeing the held-out set. The log fingerprints the prompt "
        "either way, so say which it was."),
    code("macro_f1 = freeze_test_run(FINAL_PROMPT, test, f1_by_round,",
         "                           PRED_PATH, TESTLOG_PATH, ROUNDS_PATH, PROMPT_FILE,",
         "                           dev_f1=best_dev,",
         "                           ordered=False,   # True only if your labels are a SCALE",
         "                           labels=LABELS_ORDER,",
         '                           note="")   # ← if this is not attempt 1, say WHY here'),
    md(
        "---",
        "",
        "**Next:** open `05_report.ipynb`. It loads the files you just wrote and nothing "
        "else — so from here on, your numbers cannot move."),
]

# ==================================================================================
# 05 — report
# ==================================================================================
cells_05 = [
    title_cell(
        "05", "05_report", "Error analysis and export",
        "Show an item the model got wrong, and say whose fault it was.",
        "the dev/test split (03) · the frozen predictions, rounds and test log (04)",
        "`outputs/` — the predictions CSV, the report scaffold, a copy of your test set",
        ["This is the highest-value part of the whole project, and the one the Q&A will "
         "definitely go to. A low F1 with a clear account of *why* is worth more than a "
         "high one without.",
         "",
         "Everything scored here is the **held-out test set**. The per-round table from "
         "notebook 04 is your dev trail — how you got to the prompt — and the two are "
         "different claims."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Loading files is plumbing.",
        "# `show` prints the source of any of these: show(save_json)",
        "from pipeline import (load_gold, load_predictions, load_json, save_json,",
        "                      export_results, read_test_log, triage_category,",
        "                      TRIAGE_CATEGORIES, show)",
        "",
        "# Scoring and the error table you defined and read in notebook 04, so here",
        "# they are just imported. The two functions that turn those errors into an",
        "# argument are not — you define them yourself at the end of step 1.",
        "import pandas as pd",
        "from metrics import evaluate, show_errors",
    ]),
    CONFIG_MD,
    md("## Step 1 — Open the frozen run",
       "",
       "Now we load the four files notebook 04 left behind: the held-out test set, the "
       "dev half (only so the report can say how big it was), the predictions file, and "
       "the per-round table.",
       "",
       "**Nothing in this notebook calls the model.** If a number here differs from "
       "what notebook 04 printed, you are loading a different file — not watching the "
       "model change its mind.",
       "",
       "The last line prints three counts, and the test count and the prediction count "
       "**must be the same**. More predictions than test items means the run you froze "
       "was a run on dev — go back to step 4 of notebook 04.",
       "",
       "Loading a frozen predictions file is what you did on Day 2 S6."),
    *step(
        1, "Load the frozen run",
        ["Opens the held-out test set, the dev half, the predictions notebook 04 froze,",
         "and the per-round table."],
        "test, dev, pred_final, f1_by_round",
        starter=[
            "test = load_gold(TEST_PATH)     # the held-out half — everything below is this",
            "dev  = load_gold(DEV_PATH)      # only so the report can say how big it was",
            "pred_final = load_predictions(PRED_PATH)",
            'f1_by_round = load_json(ROUNDS_PATH, what="rounds",',
            '                        made_by="notebook 04_prompt")',
            "",
            'print(len(dev), "dev ·", len(test), "test ·", len(pred_final), "predictions")',
        ]),
    md("Now the table of rounds, as notebook 04 left it.",
       "",
       "The cell is just the name `f1_by_round`, with no `print`. In a notebook the "
       "value of a cell's last line is displayed automatically, and for a table that is "
       "easier to read than `print` would give you."),
    code("f1_by_round"),
    md(
        "### Your test log",
        "",
        "How many times the held-out set was scored. **One row is what we expect.** More "
        "than one is allowed — that is why this file exists rather than a lock — but "
        "whichever row your headline comes from, report §5 has to account for the others.",
        "",
        "Look at `prompt_sha1`. Two rows with the *same* fingerprint are the same prompt "
        "run twice, which tells you something useful about how much the model's answers "
        "vary on their own. Two rows with *different* fingerprints are a prompt "
        "that changed after you had seen the held-out set — a different thing entirely, "
        "and one you have to say out loud."),
    code(
        "read_test_log(TESTLOG_PATH)",
        ""),
    *study_cells(
        "### The two functions this notebook adds\n\n"
        "`evaluate` and `show_errors` are the same ones you defined and read in "
        "notebook 04, so here they are imported. These two are new, and they are what "
        "turn a list of mistakes into an argument — so SETUP did not import them, and "
        "**the cells below are where they come from**. Run them before the steps "
        "further down that use them.\n\n"
        "**Nothing here calls the model.** Both take lists you already loaded. That is "
        "what \"frozen\" means: from here on your numbers can only change if you load "
        "a different file.",
        [(metrics.errors_on_disagreed,
          "**`errors_on_disagreed` does one join**, on the item id. In: the model's "
          "errors and the rows your two coders disagreed on. Out: the rows that are in "
          "both. Look at how little there is to it — the whole force of that number "
          "comes from the fact that you built both tables yourselves, from the same "
          "forty items."),
         (metrics.triage_counts,
          "**`triage_counts`** counts your judgments by category. In: your triage "
          "dictionary and the error table. Out: how many errors you put in each "
          "category, and how much of the error set you have actually been through. "
          "\"We looked at 3 of 40\" and \"we looked at all 12\" are different claims.")],
        check="show(errors_on_disagreed)"),
    md(
        "## Step 2 — The headline number",
        "",
        "**This one is the result.** It is measured on items your prompt was never tuned "
        "against, which is what makes it worth quoting; the per-round table above is the "
        "*story of how you got here*, and belongs in report §2 rather than §3.",
        "",
        "Compare it to your best dev round. If test came out lower, that is the ordinary "
        "outcome and the gap is itself a finding — roughly, how much of your improvement "
        "was tuning to those particular dev items rather than to the task. Say what you "
        "make of it in one sentence. And keep the sample size in view while you do: a few "
        "points either way on twenty-odd items is noise, so read a small gap as \"we "
        "cannot tell these apart\" rather than as a result.",
        "",
        "`evaluate` prints per-class precision, recall, F1 and κ as well as the macro "
        "average — this is the Day 2 S6 Part B call. Read the table, not just the "
        "headline: *\"which class is it worst at\"* is a more useful sentence than "
        "*\"F1 = .62\"*.",
        "",
        "**Careful with the κ.** `annotator_agreement` in notebook 03 compared two "
        "**annotators**. This compares your gold labels against a **model**. Same kind "
        "of number, different claim — do not swap them in the report."),
    *step(
        2, "Score the frozen run",
        ["Scores the frozen predictions against the held-out test set. These are the",
         "numbers for report section 3."],
        "macro_f1",
        starter=[
            "# Use the SAME `ordered` you used in notebook 04. If you switch it here,",
            "# the headline number stops matching the table you reported the rounds in.",
            "macro_f1 = evaluate(test, pred_final,",
            "                    ordered=False,       # True only if your labels are a SCALE",
            "                    labels=LABELS_ORDER)",
            'print("macro-F1 on the held-out test set:", round(macro_f1, 3))',
        ]),
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
    *step(
        3, "The errors",
        ["Lists every test item the model got wrong, and shows you the first fifteen."],
        "errors",
        starter=[
            "errors = show_errors(test, pred_final)",
            "errors.head(15)",
        ]),
    md("### Now look at one label's misses at a time",
       "",
       "`errors` is a table, and `errors[errors.gold == \"…\"]` keeps only the rows "
       "whose `gold` column equals what you put in the quotes. Start with the class "
       "`evaluate` scored worst.",
       "",
       "The cell picks that class for you, but the choice worth making is *which* label "
       "— replace the first line with e.g. `LOOK_AT = \"Move 2\"` and re-run it as often "
       "as you like."),
    code("# The label with the most misses. Counting them out, one row at a time.",
         "miss_counts = {}",
         'for label in errors["gold"]:',
         "    if label not in miss_counts:",
         "        miss_counts[label] = 0",
         "    miss_counts[label] = miss_counts[label] + 1",
         "",
         "LOOK_AT = max(miss_counts, key=miss_counts.get)   # <- or type a label yourself",
         'print("looking at:", LOOK_AT, "·", miss_counts)',
         "",
         'errors[errors["gold"] == LOOK_AT]'),
    md("### Now the cross-reference — where did YOUR coders disagree?",
       "",
       "Notebook 03 already worked this out and saved it, so this is one file being "
       "opened rather than a second trip to the Google Sheet. It works whether or not "
       "the sheet still exists, and it cannot disagree with 03 about who the coders "
       "were.",
       "",
       "**If it says the file is missing**, notebook 03 was run before this file was "
       "part of the project. Re-run its step 1, or skip this cell and the next — the "
       "triage in step 4 still works without them."),
    code("disagreed = load_json(DISAGREED_PATH)   # written by notebook 03, step 1",
         'print(len(disagreed), "rows your coders labelled differently")'),
    md("### Now the number this whole project has been building towards",
       "",
       "How many of the model's errors land on the very items your two coders could not "
       "agree on either. If they cluster there, what you have measured is a fuzzy "
       "boundary in your annotation scheme rather than a stupid model — and that is a "
       "better finding than a clean F1.",
       "",
       "Write it down **either way**: a low overlap is just as reportable, and means "
       "something else. Note too that `disagreed` covers the whole sheet while `errors` "
       "covers only the test half, so the overlap is smaller than it would have been "
       "without a split. Nothing is broken — the split kept every item's original id "
       "precisely so this join still lines up."),
    code("overlap = errors_on_disagreed(errors, disagreed)",
         'print("ids to read again before you blame the model:", overlap)'),
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
        "what you would do with another week.",
        "",
        "Write it while the errors are in front of you — it is the single hardest thing "
        "to reconstruct a week later. And if your `scheme` ids overlap with `overlap` "
        "from step 3, say so in the report: your own coders are the evidence."),
    *step(
        4, "Triage the errors",
        ["Counts your judgments by category and tells you how many errors you have",
         "still to go through. It runs empty, so add your lines and re-run it."],
        "TRIAGE",
        starter=[
            "# One line per error you have discussed: the id from the `errors` table",
            "# above, then one of the four category words, then WHY you say so.",
            "# It runs empty — and tells you how many you still owe — so add your own",
            "# and re-run as you work through them, out loud, together.",
            "TRIAGE = {",
            "    # 7: \"scheme — Move 1/Move 2 boundary; our coders split on this one too\",",
            "}",
            "",
            "triage_counts(TRIAGE, errors)",
        ]),
    md("### Now save your triage",
       "",
       "Once the counts above read the way you want them to. Step 5 reads this file "
       "straight into report §4, so your reasons end up in the report rather than in "
       "somebody's notes."),
    code('save_json(TRIAGE, TRIAGE_PATH, what="triaged errors")'),
    md(
        "## Step 5 — Export",
        "",
        "Writes your test set, a per-item predictions CSV, and a one-page report "
        "scaffold with the five required sections, all stamped with your group name.",
        "",
        "The scaffold fills in what it can compute — labels, counts, the F1-per-round "
        "table, and now your triage: the category counts, and your reason printed beside "
        "each item. The *italic* placeholders are what is left for you: the QC narrative, "
        "the pattern in your `scheme` errors, and limitations that apply to **your** run "
        "rather than the generic three. A section left as the placeholder scores zero, "
        "so this is the start of the writing, not the end.",
        "",
        "Two of the arguments change what the report says. With `triage=TRIAGE`, "
        "section 4 is your analysis; leave it off and section 4 is a placeholder asking "
        "you for it. With `dev=dev`, the report states which half is which — how many "
        "items you tuned on, how many you reported on, and that the rounds table is a "
        "dev trail with one test row at the bottom. From the numbers alone a reader "
        "cannot tell."),
    *step(
        5, "Export",
        ["Writes three files into `../outputs/`: your test set, a per-item predictions",
         "CSV, and the report scaffold."],
        starter=[
            "export_results(TRACK, test, pred_final, f1_by_round, OUT_DIR,",
            "               group=GROUP, run=RUN, triage=TRIAGE, dev=dev)",
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
