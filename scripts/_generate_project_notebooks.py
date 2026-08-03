#!/usr/bin/env python3
"""Generate notebooks/02_sample · 03_annotate · 04_develop · 05_test · 06_report.ipynb.

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
import annotate
import pipeline
import _study
from _setup_cell import REPO, setup_lines, setup_md

OUT = Path(__file__).resolve().parent.parent / "notebooks"

NOTEBOOKS = ["01_build_pool_<track>", "02_sample", "02b_add_samples", "03_annotate",
             "04_develop", "05_test", "06_report"]


def _src(lines: list[str] | tuple) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def step(number: int, title: str, does: list[str], creates: str = "",
         extra: list[str] | tuple = (), starter: list[str] | tuple = (),
         signpost: str = "", decide: str = "") -> list[dict]:
    """A step cell: a two-line header, then either code to run or a cell to fill in.

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

    `decide` names the choice this cell puts in front of the student, in one line - the
    sampling strategy to keep, whether the labels sit on a scale. It prints above the
    starter code, and it is the ONLY place a step cell may say a decision exists. Leave
    it empty and the cell says nothing.

    That restriction is the point. Every starter cell used to carry "this runs as
    written - the work is deciding whether it should", whether or not the code offered
    anything to change. Where there was nothing to change, the sentence asserted a
    decision the student had no way to make, and a cell that claims a choice it does
    not offer teaches that choosing is something you say rather than something you do.
    Name the choice, or say nothing.

    Omit `starter` entirely for a cell the student writes themselves - it becomes
    "# ✏️ your code here".

    `creates` is left empty on a step that names nothing new - a check, or one that
    only writes files. Say what it wrote in the markdown instead.

    Args:
        number: the step number, as it appears in the markdown above.
        title: the step's heading, used in the banner rule.
        does: what running the cell does, one sentence per line.
        creates: the names this cell leaves behind for later cells.
        extra: further header lines, appended after `creates`.
        starter: the code the cell ships with. Left out, the cell is empty.
        signpost: a markdown lead-in, returned as a cell above the code.
        decide: the one choice this cell offers, if it offers one.

    Returns:
        The cells to add: the code cell, preceded by a markdown lead-in when
        `signpost` is given.
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
        if decide:
            # Wrapped: a single long comment line runs off the side of a Colab cell,
            # and the one line a student is meant to read is the wrong one to hide.
            wrapped = textwrap.wrap(decide, width=72)
            lines.append("# ✏️ " + wrapped[0])
            for line in wrapped[1:]:
                lines.append("#    " + line)
            lines.append("")
        for line in starter:
            lines.append(line)
    else:
        lines.append("# ✏️ your code here")
    lines.append("")
    if signpost:
        return [lead(signpost), code(*lines)]
    return [code(*lines)]


def for_report(frame: list[str], *lines: str) -> dict:
    """The one-decision writing prompt that closes a section.

    Every place this notebook hands a methodological choice back is also a place the
    report and the Q&A will ask about it, and the answer is worth more than the number
    above it. So the section ends by asking for the sentence - with the sentence half
    written, because most of the class is working in a second language and a blank page
    is a harder task than the decision it is asking about.

    It asks for prose, not for a variable. Nothing here is saved, computed or checked:
    a `WHY = "..."` string in a notebook is a second place to write the same sentence,
    and the rubric already scores the one place that matters.

    Args:
        frame: the sentence frame, one line per sentence, blanks as `___`.
        lines: what a good answer names, one sentence per line.

    Returns:
        The markdown cell to add.
    """
    out = ["**✍️ For your report and the Q&A** — this goes in the write-up, not in a "
           "cell below."]
    out.append("")
    for line in frame:
        out.append("> " + line)
        out.append(">")
    out = out[:-1]
    if lines:
        out.append("")
        out.extend(lines)
    return md(*out)


def embed(*objects, imports: list[str] | tuple = (), why: str = "") -> list[dict]:
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


def lead(*lines: str) -> dict:
    """A markdown signpost immediately above a code cell: what we are about to do.

    Every code cell in these notebooks has one. A cell a student meets with no idea
    what it is for is a cell they run and scroll past.
    """
    return md(*lines)


def source_cells(described: list[tuple],
                 imports: list[str] | tuple = ()) -> list[dict]:
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


def _preamble(objects: tuple) -> list[str]:
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


def study_cells(what: str, described: list[tuple],
                check: list[str] | None = None) -> list[dict]:
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
    cell - `help(sample)` - to confirm the definitions took.
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
                          "prints the first line of the function the notebook will "
                          "actually use, and the description of each argument."))
        cells.append(code(check))
    return cells


def read_me_md(what: str, points: list[str]) -> dict:
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


def spine(current: int) -> dict:
    """The notebook map, with the one you are in marked.

    Matched on the number in front of the first underscore, and matched WHOLE: "02"
    must not also mark "02b_add_samples", which is what a startswith would do.
    """
    parts = []
    for name in NOTEBOOKS:
        if name.split("_")[0] == current:
            parts.append("▶ " + name)
        else:
            parts.append("  " + name)
    return "```\n" + "  →".join(parts) + "\n```"


def title_cell(number: int, name: str, title: str, one_line: str, reads: str,
               writes: str, body: list[str]) -> list[dict]:
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


def setup_cell(extra_imports: list[str] | tuple = (),
               exclude: list[str] | tuple = ()) -> list[dict]:
    """The SETUP cell for one notebook.

    `exclude` drops config names from this notebook's import. 04_develop uses it to
    leave the held-out names unbound, so its claim that it cannot reach the test set
    is a fact about the namespace rather than a promise nobody enforces.
    """
    return code(*setup_lines(extra_imports, exclude=exclude))


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


def handoff_md(what: str, target: str, next_notebook: str, why: str) -> dict:
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
        "# they are imported. So is the drawing itself: sample_pool, sample_random and",
        "# sample_by_document are three ways of picking items out of a list, and there",
        "# is no judgment inside any of them. The judgment is WHICH of the three, and",
        "# that is the one thing not imported — `sample` is defined further down, in a",
        "# cell you can read and change, just before the step that calls it.",
        "from pipeline import (load_gold, save_json, label_set,",
        "                      sample_pool, sample_random, sample_by_document)",
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
        "This is your **sampling method** — what your report's methodology section has to describe and "
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
          "comes back short — that is data, not a bug. Step 3 gives the sample its own "
          "ids, 1, 2, 3 …, and keeps the id each item had in the pool as `source_id`. "
          "That second number is what lets `02b_add_samples.ipynb` add more items "
          "later without drawing one you already have."),
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
        check="help(sample)"),
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
       "Not in the notebook — in `PLAN.md` §5, in a sentence. It belongs in your "
       "report's methodology section, "
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
         "project can tell you that, and notebook 06 asks you for it directly.",
         "",
         "It ends by drawing one more line: which of your annotated items you are allowed "
         "to *look at* while you write prompts, and which are held back for the number you "
         "report. One sheet, one adjudication, then a split — it costs no extra coding."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# The Google Sheets round trip is plumbing, so it is imported, and so is the",
        "# dev/test split: stratifying and rounding are bookkeeping, not judgment.",
        "from pipeline import (load_gold, label_set, save_json, split_dev_test,",
        "                      plot_confusion_matrix)",
        "from annotate import (remembered_sheet, load_coder_sheets, fleiss_kappa,",
        "                      adjudicated_rows)",
        "",
        "# The sheet's column headings, by the names the code below uses for them.",
        "from _study import COL_ID, COL_TEXT, COL_FINAL",
        "",
        "# `column`, `percent_agreement`, `to_canonical` and `compare_to_published` are",
        "# NOT imported. Each one decides something you have to defend — which rows count,",
        "# what a valid label is, what counts as the same item — so they are cells you can",
        "# read and change instead. Run those cells before the steps that use them.",
        "",
        "# The scoring itself comes from scikit-learn, by its own names. You checked your",
        "# hand-built versions of these against it on Day 2 S6; these are those functions.",
        "from sklearn.metrics import cohen_kappa_score, confusion_matrix",
        "",
        "# pandas, for the table the function you write in step 3 hands back.",
        "import pandas as pd",
        "",
        "# `disagreements` is NOT imported. Step 3 asks you to write it, because the rule",
        "# inside it — what counts as a disagreement — is a decision about your scheme",
        "# rather than a fact about your data.",
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
        "# against it in step 5, once your own labels are settled.",
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
    *source_cells(
        [(_study.column,
          "**`column`** turns one coder's tab into a plain list, in row order. Note "
          "what it does with a cell that coder left blank: it keeps an empty string "
          "rather than dropping the row, so two calls on the same rows stay the same "
          "length and stay lined up item by item. Dropping would silently shift one "
          "coder's labels against the other's."),
         (_study.percent_agreement,
          "**`percent_agreement`** counts how often the two matched. Read the `if` in "
          "the middle: the denominator is **rows both coders labelled**, not every row "
          "in your sample. A pair who have each finished half the sheet can show a high "
          "agreement over very few items.\n\n"
          "This matters for the cell you write in step 2. `cohen_kappa_score` does "
          "**not** apply that filter — hand it the same two lists and it counts a blank "
          "as a label like any other. Two numbers, side by side in your report, over "
          "different sets of rows.")]),
    md(
        "## Step 1 — Line the coders up, and get the first number",
        "",
        "Each coder has their **own tab**, so the first thing to do is line them up side "
        "by side. `load_coder_sheets` reads one tab per name you give it and joins them "
        "by item id into a single table — one column per coder, plus `Final`.",
        "",
        "> **Who annotated?** `CODERS` comes from `config.yaml`, so notebook 06 finds the "
        "same tabs. If a third coder joined, duplicate an **empty** tab in the sheet "
        "(right-click ▸ *Duplicate*), rename it `CoderC`, and add it there.",
        "",
        "`column(rows, name)` pulls one coder's labels out as a plain list, in row order "
        "— two lists lined up item by item, which is what every statistic takes.",
        "",
        "Then percent agreement: how often the two of them chose the same label. It is "
        "the one number every design owes, whatever else you report, and it is the only "
        "one this cell computes. **Step 2 is where you add the rest**, because which ones "
        "you owe depends on your design rather than on your data.",
        "",
        "Run it once **every** coder's tab is filled in — rows that not everyone labelled "
        "are left out of the comparison. If two coders appear to have given every item "
        "the same label, somebody duplicated a tab that had already been filled in; that "
        "agreement is a copy rather than a measurement.",
        "",
        "Now we read the tabs, line them up, and measure raw agreement."),
    *step(
        1, "Line the coders up",
        ["Reads one tab per coder, lines them up by item id, and measures how often",
         "they chose the same label."],
        "rows, a_labels, b_labels",
        starter=[
            "rows = load_coder_sheets(SHEET_ID, CODERS)   # one read per tab, merged by ID",
            "",
            "a_labels = column(rows, CODERS[0])",
            "b_labels = column(rows, CODERS[1])",
            "",
            "percent_agreement(a_labels, b_labels)",
        ]),
    md(
        "## Step 2 — The rest of the agreement your design owes",
        "",
        "Percent agreement counts lucky agreement as if you had earned it. Two coders "
        "labelling at random on a two-label scheme agree half the time; the same number "
        "on an eight-label scheme means something else entirely. So it never stands "
        "alone.",
        "",
        "**What you owe is not a free choice, and it is settled before you run anything** "
        "— by how many coders you have, and by whether `PLAN.md` §3 says your labels are "
        "a scale:",
        "",
        "| Your design | Report |",
        "|---|---|",
        "| two coders, labels with no order | percent agreement **and** Cohen's κ |",
        "| two coders, labels on a scale | those two, **and** the weighted κ |",
        "| three or more coders | percent agreement **and** Fleiss' κ, plus Cohen's κ per pair |",
        "",
        "Read that off your own design. Choosing the statistic after seeing which one "
        "flatters you is the one thing that would make the number meaningless — which is "
        "exactly why the rule above depends on nothing you are about to find out.",
        "",
        "Here is everything available. Nothing here is new: you met the last three on "
        "Day 2 S6, under these names, when you checked your own precision, recall, F1 "
        "and κ against scikit-learn's.",
        "",
        "| Call | What it gives you |",
        "|---|---|",
        "| `cohen_kappa_score(a, b)` | agreement corrected for chance, two coders |",
        "| `cohen_kappa_score(a, b, weights=\"quadratic\")` | the same, counting a near miss as a smaller error |",
        "| `fleiss_kappa([a, b, c])` | one number for three or more coders |",
        "| `confusion_matrix(a, b, labels=LABELS)` | which label **pairs** you disagree about |",
        "| `plot_confusion_matrix(matrix, LABELS, title)` | that matrix, drawn |",
        "",
        "**The confusion matrix is not optional**, whichever numbers you report. The κ "
        "says how far apart you were; only the off-diagonal cells say *which pair of "
        "labels* you disagree about, and that pair is what you go back to the sheet to "
        "argue about. Looking at it changes what you do next, so there is nothing to "
        "protect yourself from.",
        "",
        "**Write the numbers down as they print** — they belong in your report's methodology section, and they "
        "do not survive a runtime reset. A κ around .8 is strong; around .4 means the "
        "scheme, not the annotators, is doing something wrong. Either is a reportable "
        "finding. A low κ you can explain beats a high one you cannot.",
        "",
        "Now we add the statistics your design owes, and draw the matrix. `a_labels` and "
        "`b_labels` are ready from step 1.",
        "",
        "Stuck? `from answers import answer`, then `answer(\"agreement\")` — after you "
        "have tried."),
    *step(
        2, "The rest of the agreement your design owes",
        ["Adds the chance-corrected statistic your design calls for, and draws the",
         "coder-vs-coder confusion matrix."],
        "matrix"),
    for_report(
        ["Our coders agreed on ___% of items, with a ___ κ of ___.",
         "We report ___ as well as percent agreement because our labels ___.",
         "The pair we disagreed about most often was ___ and ___, which suggests our "
         "scheme ___."],
        "The second sentence comes off the table above, and the reason is what is graded "
        "— not which statistic you ran.",
        "",
        "The third is the one the Q&A goes to. It comes off the matrix, and it is a "
        "claim about your **scheme** rather than about your coders — two labels that "
        "keep swapping usually means the boundary between them is not written down "
        "clearly enough. It is also what step 3 acts on.",
        "",
        "A low κ you can explain beats a high one you cannot."),
    md(
        "## Step 3 — What counts as a disagreement?",
        "",
        "Now you need the list of rows to argue about. **This one you write**, and it is "
        "the only function in the project that you do — because the rule inside it is a "
        "decision about your scheme rather than a fact about your data, and there is no "
        "way to hand it over without answering it for you.",
        "",
        "The obvious rule: a row is a disagreement when your coders did not all choose "
        "the same label. For most schemes that is the right one.",
        "",
        "It is not the only defensible one. **If your labels sit on a scale** — A1 < A2 "
        "< … < C2, Low < Mid < High — you might decide that neighbouring labels are two "
        "people reading the same sentence much the same way, and that only a gap of two "
        "or more is worth an argument. That version hands back a shorter list and sends "
        "you to the sheet with less to settle. Which you chose, and why, is a sentence "
        "in your report; not knowing which you used is the only wrong answer.",
        "",
        "You have written this shape before. It is a loop over `rows`, keeping the ones "
        "where the labels differ:",
        "",
        "```python\n"
        "def disagreements(rows, coders):\n"
        "    out = []\n"
        "    for row in rows:\n"
        "        ...        # pull each coder's label out of this row\n"
        "        ...        # keep the row if they are not all the same\n"
        "    return pd.DataFrame(out, columns=list(rows[0]))\n"
        "```",
        "",
        "The cell below it calls yours as `disagreements(rows, coders=CODERS)` — by "
        "keyword, so that it works whether your second parameter is named `coders` or "
        "you pasted the reference version, whose first two parameters are the two "
        "column names.",
        "",
        "`column(rows, name)` is not what you want inside here — that reads a whole "
        "column down the sheet, and you are working across one row at a time. "
        "`row.get(name, \"\")` is the piece you need, and `str(...).strip()` around it "
        "drops the stray spaces a spreadsheet loves to add.",
        "",
        "**Leave out the rows nobody finished.** A blank cell is a coder who has not got "
        "there yet, not two people disagreeing, and counting it as a disagreement puts "
        "an item on your adjudication list that has nothing to adjudicate.",
        "",
        "Tried it? `from answers import answer`, then `answer(\"disagreements\")`."),
    *step(
        3, "Write the rule",
        ["Defines the function that decides which rows go on your adjudication list.",
         "Nothing runs until the cell below calls it."],
        "disagreements"),
    md("### Now run it, and keep the result",
       "",
       "This cell calls the function you just wrote and saves what comes back.",
       "",
       "The table comes back in notebook 06, where the rows your coders argued about are "
       "what you check the model's errors against — and saving it here means 05 does not "
       "have to sign back in to the sheet and derive the same table a second time. It "
       "also means that step still works after the sheet has been deleted, or its owner "
       "has left.",
       "",
       "The last line is just the name `disagreed`, with no `print`. In a notebook the "
       "value of a cell's last line is displayed automatically, and for a table that "
       "reads far better than `print` would. Add a line after it and the table stops "
       "appearing, which is the one thing to watch out for."),
    code("# `coders=` by name, not by position: the reference version in answers.py",
         "# takes the two column names first, so a pasted copy of it would read CODERS",
         "# as one coder's name if this were positional.",
         "disagreed = disagreements(rows, coders=CODERS)",
         'save_json(disagreed.to_dict("records"), DISAGREED_PATH,',
         '          what="rows your coders disagreed on")',
         "disagreed"),
    for_report(
        ["We counted a row as a disagreement when ___.",
         "That put ___ of ___ items on our adjudication list.",
         "We chose that rule rather than ___ because ___."],
        "Both rules are defensible and they hand you different lists, so the sentence "
        "that matters is the third one. If your labels are not on a scale there was only "
        "one sensible rule, and saying so is a complete answer."),
    md(
        "## Step 4 — Adjudicate",
        "",
        "The rows left on your list do not go away by rewriting the guidelines. You "
        "decide them. This is the Day 2 S5 step F, on your own data.",
        "",
        "Go back to the sheet and fill in `Final` for **every** row:",
        "",
        "- Where you agreed, `Final` is that label.",
        "- Where you did not, talk it out and decide. If you cannot agree, the scheme is "
        "underspecified — write down *why* in `Note` and pick one. That note is worth "
        "more to your report than the label is.",
        "",
        "### Before you settle them: is a second round worth it?",
        "",
        "Look again at the confusion matrix from step 2. **If one pair of labels "
        "accounts for most of your disagreements**, the boundary between those two is "
        "not written down clearly enough, and that is fixable: revise what your "
        "guidelines say about that pair, duplicate the coder tabs into a fresh round, "
        "re-annotate the affected rows there, and re-run steps 1–3. Agreement should "
        "move, and you can report by how much.",
        "",
        "**If the disagreements are scattered across many pairs**, a second round "
        "re-measures the same fuzziness and κ will barely move. Adjudicate and go on.",
        "",
        "Never overwrite a round — keep each one as its own tab, so the change is "
        "something you can show rather than assert. Either way, write down which of the "
        "two you found and what you did about it. That is the answer to *what did your "
        "QC pass change?*, which is published in advance as a Q&A question.",
        "",
        "Then re-read the sheet and canonicalise it. `to_canonical` reports blanks and "
        "invalid labels rather than silently dropping them; fix them in the sheet and "
        "re-run until it says **0 blank, 0 invalid**. A blank row is an item that has "
        "gone missing from your study without telling you.",
        "",
        "It passes `source=sampled`: gold is rebuilt from the **sheet**, which carries "
        "only the id, the text and your label, so anything else the item had — on "
        "`cars50` and `raamove`, its passage — is put back from `sampled` by id. On the "
        "other tracks that argument does nothing."),
    md(
        "### The code that builds your gold set — read it, then run it",
        "",
        "`to_canonical` is the function that decides what your gold set **is**, so what "
        "it leaves out matters as much as what it keeps. Read the three piles it sorts "
        "every row into:",
        "",
        "- A **blank** `Final` is counted and skipped. That row does not reach your gold "
        "set.",
        "- A label **not in your scheme** is reported, not repaired. `.strip()` is the "
        "only cleaning it does, so `b1` and `B11` are invalid rather than read as `B1` — "
        "it will not guess what you meant.",
        "- A row whose **ID cell** is not a number is dropped and named.",
        "",
        "So `len(gold)` can be smaller than the number of items you sampled, and nothing "
        "raises an error when it is. That is why it prints all three counts: **run it "
        "until it says 0 blank and 0 invalid**, or report the shortfall as a limitation."),
    code(*inspect.getsource(annotate.to_canonical).rstrip("\n").split("\n")),
    md("Now we re-read the sheet, with `Final` filled in, and turn it into your gold "
       "set. `rows` from step 1 was fetched before you adjudicated, so it is fetched "
       "again here."),
    *step(
        4, "Adjudicate, then canonicalise",
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
        "### Now keep the argument, not just the answer",
        "",
        "`to_canonical` took your `Final` labels into `gold` and dropped everything "
        "else — including the `Note` column. The label is what the model gets scored "
        "against; the note is the only record of **what you decided and on what "
        "grounds**, and it is what your report's methodology section asks you for.",
        "",
        "So this saves the disagreed rows with both. It is one file being written now, "
        "rather than a sentence you try to reconstruct from memory in a week — and by "
        "then the sheet may have been deleted, or its owner may have left the group.",
        "",
        "It tells you how many rows still have no `Final`, and how many have no note. "
        "A row with a label and no note is a decision nobody can check."),
    code("adjudicated = adjudicated_rows(rows, coders=CODERS)",
         'save_json(adjudicated, ADJUDICATED_PATH,',
         '          what="rows you adjudicated, with the reason")',
         "pd.DataFrame(adjudicated)"),
    for_report(
        ["Our adjudication settled ___ rows, ___ of which we recorded a reason for.",
         "Most of them were the ___ / ___ boundary, which our scheme did not settle "
         "because ___.",
         "We did / did not run a second annotation round, because ___."],
        "The second sentence is the finding: it names a boundary in your scheme, and "
        "the confusion matrix in step 2 is the evidence for it. Notebook 06 asks the "
        "same question of the model's errors, and the interesting result is when the "
        "two land on the same pair.",
        "",
        "The third comes off the decision rule above — say which of the two patterns "
        "you saw, not just what you did."),
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
        "This table belongs in your report's methodology section, and it is the one that most often produces a "
        "sentence worth saying out loud in the Q&A. Pick two or three rows and write "
        "down which of the three cases above they are — now, while you still remember "
        "the argument you had about them.",
        "",
        "The comparison runs against `sampled`, not `pool`: sampling renumbered the "
        "ids, so `pool` would line your item 7 up against a completely different "
        "sentence. This is the Day 2 S5 step F call."),
    md(
        "### The code that compares you to the published labels — read it, then run it",
        "",
        "This function prints the percentage that goes into your report's methodology section, so read "
        "how it gets there. Three decisions are inside it:",
        "",
        "- **What counts as the same item.** It matches on the **text**, because "
        "sampling renumbered the ids — an id match would line your item 7 up against a "
        "different sentence. Ids are a fallback for when the text has been edited, and "
        "it says so when that happens.",
        "- **What happens to an item it cannot match at all.** Look at the "
        "`else: continue`. That item leaves the comparison entirely — it is not counted "
        "as an agreement or a difference, and the denominator shrinks without a warning.",
        "- **What counts as differing.** Exact string equality. No case folding.",
        "",
        "The percentage it prints is therefore over the items it **could** match, not "
        "over your whole gold set. If those two numbers are not the same, say which one "
        "you are quoting."),
    code(*inspect.getsource(annotate.compare_to_published).rstrip("\n").split("\n")),
    md("Now we run it against the items you sampled — `sampled`, not `pool`, because "
       "those are the same forty items your coders saw."),
    *step(
        5, "Compare against the published labels",
        ["Shows every row where your group's label and the corpus's label differ."],
        "differences",
        starter=[
            "differences = compare_to_published(gold, sampled)   # sampled, not pool: same 40 items",
            "differences",
        ]),
    for_report(
        ["We matched ___ of our ___ gold items to the published set, and agreed with "
         "the published label on ___% of them.",
         "Where we differed, ___ of the rows were our scheme reading a boundary "
         "differently, ___ were genuinely ambiguous, and ___ were one of us being wrong.",
         "The clearest example is item ___, where we said ___ and they said ___, "
         "because ___."],
        "The second sentence is the analytical work this step exists for, and the third "
        "is what the Q&A goes to. Pick the rows now, while you still remember the "
        "argument you had about them."),
    handoff_md(
        "gold set", "data/gold/<track>_<group>_gold.json", "04_develop",
        "This file is the single most valuable thing your group makes all week — hours "
        "of judgment, and the only thing in the project that could not have been "
        "produced by a script. Every number in notebooks 04, 05 and 06 is measured against "
        "it, and it goes in your submission bundle."),
    code(
        "save_json(gold, GOLD_PATH, what=\"gold items\")",
        "",
        "# It is git-ignored — it is your work, not part of the template. If you cloned",
        "# into Google Drive it is already saved across sessions; if not, download it.",
        ""),
    md(
        "## Step 6 — Draw the line: dev and test",
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
        "| **test** | opened once, in `05_test.ipynb`. Whatever it says is "
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
    md(
        "### What `split_dev_test` does with your `dev:` setting",
        "",
        "It is imported rather than printed here, because what is inside it is "
        "bookkeeping — grouping by label, rounding a fraction to a whole number of "
        "items, and being careful about a label with only one item left. None of that "
        "is a decision you make; the ratio is, and that is in `config.yaml`.",
        "",
        "Three things it does that are worth knowing, because they show up in your "
        "numbers:",
        "",
        "- **A rare class goes to test, not dev.** A label with a single surviving item "
        "cannot be on both sides. Missing from test, it drops out of your macro average "
        "without announcing itself; missing from dev, it only costs you feedback. The "
        "second is the cheaper mistake, so that is the one it makes.",
        "- **The rounding is written out** rather than left to `round()`, which in "
        "Python rounds 0.5 down and 1.5 up — not something you want to explain in the "
        "Q&A.",
        "- **The ids are not renumbered.** Notebook 06 asks which of the model's errors "
        "are also the rows your coders argued about, and that join runs on these ids.",
        "",
        "`help(split_dev_test)` prints what to pass it. To read the code itself, run "
        "`split_dev_test??` — it prints the source of any function, imported or not."),
    *step(
        6, "Split dev / test",
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
       "Once you are happy, save. Notebook 04 opens `dev`; notebook 05 opens `test`, once."),
    code('save_json(dev,  DEV_PATH,  what="dev items")',
         'save_json(test, TEST_PATH, what="test items")'),
    for_report(
        ["We split ___ gold items into ___ dev and ___ test.",
         "We set `dev:` to ___ because ___.",
         "Every label is present on both sides except ___."],
        "There is no right ratio at this size, only one you can defend, so the second "
        "sentence is the whole answer. The third is the limitation a reader needs in "
        "order to read your per-label scores — a label that is only in test was never "
        "something you could iterate against."),
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
        "04", "04_develop", "Develop your prompt, on dev only",
        "Assemble a prompt from moves you can name, and change it for reasons you can "
        "state.",
        "`data/gold/<track>_<group>_dev.json` (from 03) · the pool (from 01)",
        "`outputs/<track>_<group>_rounds.json` · `..._round_notes.json` · your prompt "
        "files",
        ["Everything from here on is measured against **your** gold set, not the "
         "corpus's labels. That is the point of the last three notebooks.",
         "",
         "**This notebook never opens the held-out set.** That is why it is a separate "
         "file from `05_test.ipynb` rather than a section further down: a boundary you "
         "have to open another notebook to cross is one you cannot drift across while "
         "you are concentrating on something else.",
         "",
         "Every number here is a **dev** number. You use them to decide what to change "
         "next. None of them is the number you report.",
         "",
         "> **Free-tier pacing.** The backend waits a few seconds between calls and "
         "retries on rate-limit errors, so a run takes minutes and may print "
         "`(rate limited - waiting Ns then retrying)`. That is normal — and it is why "
         "you iterate on dev: a dozen or so items is about a minute per round, so you "
         "get enough rounds to actually learn something.",
         "",
         "> **On the size of this study.** One call per item, four-and-a-bit seconds "
         "apart, no batching: forty items is minutes and four hundred is most of an "
         "afternoon of a quota you share with everyone else on the course. A study that "
         "could support a claim about a corpus needs hundreds of items per class. This "
         "one cannot, and that is a limitation to state in your report's limitations section rather than write "
         "around. What transfers is the method — the split, the freezing, the audit "
         "trail — not the number."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Files in, files out, and the connection to the model: all plumbing.",
        "from pipeline import (load_gold, label_set, load_prompt, save_prompt,",
        "                      save_json, setup, build_fewshot)",
        "",
        "# Scoring. `evaluate` prints per-class precision, recall and F1, Cohen's κ and",
        "# the confusion matrix, and hands back the macro-F1 — the Day 2 S6 Part B call.",
        "from metrics import evaluate",
        "",
        "# `run_prompt`, `extract_label` and `show_errors` are NOT imported. They are the",
        "# three that decide what the model said, what counts as a label and what counts",
        "# as an error, so they are cells further down that you can read and change. Run",
        "# those cells before the rounds that use them.",
        "#",
        "# These are what those three cells call. `_default_backend` is the connection",
        "# `setup()` opened and `_one_prediction_line` is the one-line-per-item printout;",
        "# both are plumbing, so they stay imported rather than filling a cell.",
        "import re",
        "import pandas as pd",
        "from pipeline import _default_backend, _one_prediction_line",
        "",
        "# `freeze_test_run` is NOT imported here — and neither is TEST_PATH, PRED_PATH",
        "# or TESTLOG_PATH. They are left out of the config import above on purpose, so",
        "# there is no name in this notebook that reaches the held-out set. Not a rule",
        "# you have to remember: a NameError if you try. Those live in 05_test.ipynb.",
    ], exclude=["TEST_PATH", "PRED_PATH", "TESTLOG_PATH"]),
    md(
        "## Connect to the model",
        "",
        "Now we open the connection this notebook will send every prompt through. It "
        "gets a cell of its own because it does something the plumbing cell above does "
        "not: it reaches out to a service, and what it prints back is worth reading.",
        "",
        "**The three settings are handed over here in the open**, from `config.yaml`, "
        "because they are part of your study rather than of this session — the "
        "temperature that produced the number in your report is a number your report "
        "has to state, and `PLAN.md` asks for it.",
        "",
        "`temperature: 0` tells the model to take its most likely answer every time, "
        "which is what makes a run repeatable. Raising it is a real experiment rather "
        "than a mistake: run the same prompt twice at 0 and twice at 1, and count how "
        "many labels changed. That measures how much of the difference between two of "
        "your rounds was your prompt and how much was the model — worth knowing before "
        "you claim a two-point gain.",
        "",
        "Safe to run more than once. With the same settings it hands back the "
        "connection it already made; **change one and it reconnects and says so**, "
        "rather than quietly going on sending at the old temperature."),
    code("setup(temperature=TEMPERATURE, seed=SEED, model=MODEL)"),
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
       "happens to be thin in dev should not quietly shrink your label list."),
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
        "## What you have to work with",
        "",
        "Every round below is these calls, in an order you choose. Nothing else is "
        "imported, and nothing here is new — the right-hand column says where you first "
        "ran each one.",
        "",
        "| Call | What it gives you | First run |",
        "|---|---|---|",
        "| `load_prompt(path)` | a prompt file, as text | 04 |",
        "| `save_prompt(text, path)` | one version of your prompt, written to a file | 04 |",
        "| `build_fewshot(prompt, pool, gold)` | the same prompt with worked examples in front of it | Day 3, typed by hand |",
        "| `run_prompt(prompt, dev)` | one predicted label per item | Day 3 |",
        "| `evaluate(gold, pred)` | per-class P/R/F1, κ, the matrix, and the macro-F1 back | Day 2 S6 Part B |",
        "| `show_errors(gold, pred)` | just the rows it got wrong | Day 3 |",
        "| `extract_label(reply, labels)` | one label out of one reply | Day 3, as `_extract_level` |",
        "| `last_replies()` | what the model actually said, one string per item | 04 |",
        "",
        "### What you can put in a prompt",
        "",
        "A prompt is ordinary text with **placeholders** in it. `run_prompt` fills them "
        "in, once per item, just before it sends:",
        "",
        "| Placeholder | What gets slotted in | Available on |",
        "|---|---|---|",
        "| `{text}` | the sentence being classified | every track |",
        "| `{context}` | the passage it came from | `cars50`, `raamove` |",
        "",
        "> **A placeholder is not an f-string.** On Day 1 you wrote "
        "`f\"... {sentence}\"`, and the braces were filled in **on that line**. A prompt "
        "template has **no `f`**: the braces stay as they are until `run_prompt` fills "
        "them, once per item. Write `PROMPT = f\"...{text}\"` here and you either get a "
        "`NameError` or send the same frozen sentence to every item.",
        "",
        "> **Any other `{...}` is an error.** The filling is done with `.format()`, so a "
        "stray brace — a JSON example, a `{label}` you meant literally — stops the run "
        "on the first item. Write `{{` and `}}` for a literal brace.",
        "",
        "`{context}` is the cheapest experiment in the project on the two tracks that "
        "have it, and it is already written: `prompts/<track>_context.txt` is the same "
        "prompt with the passage added, one `load_prompt` away.",
        "",
        "**The number `evaluate` hands back is macro-F1**, and here that is all it is "
        "for: something to compare round 2 against round 1 with. Use it to decide what "
        "to change next; the number you report comes out of `06_report.ipynb`, where you "
        "write the scoring call yourself. If the headline you settled on in `PLAN.md` §9 "
        "is a different average, say so beside the rounds table — a dev trail measured "
        "one way and a held-out row measured another are not the same ruler.",
        "",
        "**And the prompt itself is the other half of what you assemble.** S7 called "
        "these the moves available to you; the deck's own split between what to run "
        "this week and what to know about is worth re-reading before you start:",
        "",
        "| Move | What it changes |",
        "|---|---|",
        "| **Instruction** | what you ask for. Start with a verb, name the label set |",
        "| **Context** | the background or rubric that scopes the task |",
        "| **Input data** | the sentence — and on `cars50` / `raamove`, whether the model also sees the passage it came from |",
        "| **Output indicator** | the *shape* of the answer. This is what `??` rows are about |",
        "| **Persona** | who you tell it to be |",
        "| **Few-shot** | worked examples in the prompt |",
        "| **Chain-of-thought** | asking it to reason before answering |",
        "| **Structured output** | asking for JSON or a fixed form |",
        "",
        "*(Know about, do not run this week: self-consistency, RAG, agentic multi-step "
        "— S7 says why.)*",
        "",
        "**Which of these you try, and in what order, is your experiment.** Few-shot is "
        "not the default and not the recommendation; it is one row of that table. "
        "`PLAN.md` §8 asks which move you predicted would help, and why."),
    *source_cells(
        [(_study.extract_label,
          "**`extract_label`** is what turns a reply into a label, and it is where "
          "every `??` comes from. Two rules to read: it keeps the **longest** label "
          "whose name appears anywhere in the reply — so a reply mentioning two labels "
          "is settled by length, not by which one the model meant — and on `Move N` "
          "tracks a bare digit anywhere in the reply becomes that move. When your "
          "model keeps answering in a shape this cannot read, **this is the cell you "
          "change**, and the next one is how your version reaches the loop."),
         (pipeline.run_prompt,
          "**`run_prompt`** is the loop: one API call per item, the reply passed "
          "through `extract`. Two things worth finding in it. The `prompt.format(...)` "
          "line is where `{text}` and `{context}` are filled in, once per item. And "
          "`extract` defaults to `extract_label` but is an **argument** — so an edit to "
          "the cell above only reaches this loop if you pass it in, which is what the "
          "rounds below do. The pacing and retrying happen inside the backend `setup()` "
          "opened; that part is plumbing and stays imported.\n\n"
          "It keeps the raw replies in `_LAST_REPLIES` rather than returning them, so "
          "that `predictions = run_prompt(prompt, dev)` stays the one-line call Day 3 "
          "taught. The next cell is how you read them back."),
         (pipeline.last_replies,
          "**`last_replies`** hands back what the model actually *said* in the run just "
          "finished, one string per item — the evidence behind every `??`. It is here "
          "rather than imported for a reason worth knowing: it has to read the "
          "`_LAST_REPLIES` that the `run_prompt` **above** writes to. Imported, it "
          "would read the one inside `pipeline.py`, which your notebook's loop never "
          "touches, and it would hand you an empty list after every run."),
         (_study.show_errors,
          "**`show_errors`** is the table that decides your next round. Its rule is one "
          "line — `item[\"label\"] != predicted` — so a `??` and a confidently wrong "
          "label are the same kind of error here, and on a scale a near miss counts the "
          "same as a far one. If that is not the view you want, this is the cell.")]),
    md(
        "## Step 2 — The baseline (round 0)",
        "",
        "Your first score, before you have changed anything. Write the plainest prompt "
        "that states the task and the label set, run it, score it. **Resist the urge to "
        "make it good** — later rounds need something to be measured against, and a "
        "baseline you already tuned tells you nothing about whether tuning helped.",
        "",
        "**You write the prompt here, in the cell.** It is an ordinary string, so "
        "editing it and running the cell again is all it takes to change what gets "
        "sent — there is no file to keep in step with it while you are working.",
        "",
        "It must contain `{text}`. Everything else is yours.",
        "",
        "`f1_by_round` collects one score per round and `NOTES` collects your one-line "
        "reason for each. Both are saved at the end of this notebook and printed side "
        "by side as your report's prompt-iterations section, so the keys are what your "
        "reader sees — name them "
        "for what you **changed**, not which round it was.",
        "",
        "Now we start those two tables and write the baseline prompt."),
    *step(
        2, "Baseline prompt (round 0)",
        ["Starts the two tables this notebook fills in, and holds the plainest prompt",
         "you can write for your track."],
        "f1_by_round, NOTES, PROMPT",
        decide="The wording of the baseline. Plainest thing that states the task and "
               "names the labels — save the good ideas for the rounds below.",
        starter=[
            "f1_by_round = {}     # round name -> macro-F1 on dev",
            "NOTES = {}           # round name -> why you made that change",
            "",
            "# ✏️ Edit this. Keep {text}; it is where each sentence is slotted in.",
            "PROMPT = \"\"\"Classify the sentence below.",
            "Answer with the label only.",
            "",
            "Sentence: {text}\"\"\"",
            "",
            "# … or start from the file written for your track:",
            "# PROMPT = load_prompt(PROMPT_FILE)",
            "",
            "print(PROMPT)",
        ]),
    md("### Now save it as a file",
       "",
       "`05_test.ipynb` is a different notebook, and nothing survives between notebooks "
       "except what is on disk. A prompt that only ever existed as a string in this "
       "session is one you cannot test and cannot report.",
       "",
       "Give every version its own name. `v0`, `v1`, `v2` beside each other are what "
       "let you show the reader what changed between rounds — and let you go back to "
       "the one that scored best after round 3 turned out worse.",
       "",
       "Now we write this version to `prompts/`."),
    code('save_prompt(PROMPT, ROOT / "prompts" / (TRACK + "_v0.txt"))'),
    md("### Now send it to the model",
       "",
       "This is the slow cell: one API call per dev item, paced a few seconds apart to "
       "stay inside the free tier. It prints one line per item as it goes — the gold "
       "label, the label it read out of the reply, and the beginning of the reply "
       "itself.",
       "",
       "**Watch the third column when the second says `??`.** That is the model "
       "answering in a shape `extract_label` cannot read, which is a finding about your "
       "prompt — an output-indicator problem — rather than a bug. `last_replies()` "
       "gives you the full replies when the printed line is too short to tell.",
       "",
       "Note `extract=extract_label`. It names the cell you read above, so the label "
       "rule this run used is one you can point at. Written out rather than left to "
       "the default, because the default is the copy inside `pipeline.py` — and a "
       "reader of your notebook cannot tell which one produced the numbers unless the "
       "call says so.",
       "",
       "It is on its own **deliberately**. Scoring and reading the errors are separate "
       "cells below, so that looking at your results again costs you nothing. If they "
       "shared a cell with this one, every re-read would re-run every call and spend "
       "your group's quota a second time."),
    code("pred0 = run_prompt(PROMPT, dev, extract=extract_label)"),
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
         "                                               labels=LABELS_ORDER)",
         'NOTES["round0 baseline (dev)"] = "the plainest prompt that states the task"'),
    md("### Now read what it got wrong",
       "",
       "Do not skip this. The errors are the only thing that tells you *what to change*; "
       "F1 only tells you afterwards whether the change worked. This is the cell that "
       "decides your next round."),
    code("show_errors(dev, pred0)"),
    md(
        "## Step 3 — Iterate",
        "",
        "The loop is always the same, and the middle step is the one that matters:",
        "",
        "```",
        "run  →  score  →  READ THE ERRORS  →  change ONE thing  →  run again",
        "```",
        "",
        "Before you touch the prompt, look at the error table from the round you just "
        "ran and ask **what these misses have in common**. There are only a few answers, "
        "and each points at a different row of the moves table above:",
        "",
        "| What you see in the errors | Which move it points at |",
        "|---|---|",
        "| One class swallows everything | **instruction** — define that class's boundary, or **few-shot** — show one |",
        "| Two labels traded in both directions | the *distinction* is unclear, to the model and possibly to your coders too |",
        "| Lots of `??` | **output indicator** — it is not answering in the shape you asked for. Fix the instruction, not the definitions |",
        "| Errors scattered with no pattern | you may be at the ceiling of what a prompt can do; consider whether the items are simply hard |",
        "",
        "**Three round blocks follow, and they are deliberately identical.** Fill in as "
        "many as you use. Each one asks you to say what you expect *before* you run it — "
        "that is the Day 3 Part B habit, and it is what makes a round a finding rather "
        "than a thing that happened. A prediction you got wrong is worth more than one "
        "you never wrote down.",
        "",
        "A round that made things **worse** is a result, not a mistake. Keep it in the "
        "table. It is often the most informative row you have."),
]

# Three identical round blocks. Written as a loop rather than copied three times, so
# that improving the shape improves all of them - and so the notebook never tells a
# student to "copy the three cells above", which is how a round silently scores the
# round before it and reports an identical F1.
for _round in (1, 2, 3):
    _n = str(_round)
    cells_04.extend([
        md("---",
           "",
           "### Round " + _n + " — say what you expect, then change one thing",
           "",
           "Fill in the three lines, then build the prompt. Leave the block empty if you "
           "stop before round " + _n + "."),
        *step(
            _round + 2, "Round " + _n + " — the prediction",
            ["Writes down what you are about to change and what you expect it to do,",
             "before you find out."],
            "WORST_" + _n + ", CHANGE_" + _n + ", EXPECT_" + _n,
            decide="Which move from the table, and why that one for these errors.",
            starter=[
                'WORST_' + _n + '  = "…"     # the class with the lowest F1 last round',
                'CHANGE_' + _n + ' = "…"     # the ONE move you are making, in a phrase',
                'EXPECT_' + _n + ' = "…"     # what you expect it to do, and why',
                "",
                'print("targeting", WORST_' + _n + ', "·", CHANGE_' + _n + ')',
                'print("expecting:", EXPECT_' + _n + ')',
            ]),
        md("Now write the prompt for this round. **Start from the version you are "
           "keeping** — that is `PROMPT` for round 1, and whichever of your later "
           "versions scored best after that. Change the one thing you named above and "
           "nothing else, or you will not know which change moved the number.",
           "",
           "`build_fewshot` is one option among the moves in the table, not the "
           "recommendation. It draws examples from the pool while skipping anything in "
           "your gold set, so you are not testing the model on answers you just showed "
           "it. Note that calling it twice with the same arguments gives you the same "
           "prompt both times — the examples are drawn with a fixed seed, so a round "
           "that only re-runs it is a round that changes nothing."),
        code("# ✏️ Write this round's prompt. Keep {text}.",
             "PROMPT_v" + _n + ' = """…"""',
             "",
             "# … or few-shot, from the version you are keeping. Note the NEW NAME on",
             "# the left: PROMPT = build_fewshot(PROMPT, ...) would stack examples on",
             "# examples every time you re-ran the cell, silently.",
             "#",
             "# `gold`, not `dev`: the list of items NOT to use as examples has to",
             "# include the test half, or the answers leak into the prompt.",
             "# PROMPT_v" + _n + " = build_fewshot(PROMPT, pool, gold,",
             "#                          shots_per_class=1, seed=SEED)",
             "",
             "print(PROMPT_v" + _n + ")"),
        md("Now save this version, so `05_test.ipynb` can load whichever one you end "
           "up choosing."),
        code('save_prompt(PROMPT_v' + _n + ', ROOT / "prompts" / (TRACK + "_v' + _n
             + '.txt"))'),
        md("Now run it. One API call per dev item, so a re-run costs your group real "
           "quota."),
        code("pred" + _n + " = run_prompt(PROMPT_v" + _n + ", dev, "
             "extract=extract_label)"),
        md("Now score it and read the new errors. **Name the key for what you changed** "
           "— in the report, *\"v" + _n + " one example per label\"* is an argument and "
           "*\"round " + _n + "\"* is a row number. Keep `ordered` the same as the "
           "baseline, or the rounds are not comparable."),
        code('KEY_' + _n + ' = "v' + _n + ' " + CHANGE_' + _n + ' + " (dev)"',
             "f1_by_round[KEY_" + _n + "] = evaluate(dev, pred" + _n + ", ordered=False,",
             "                            labels=LABELS_ORDER)",
             "NOTES[KEY_" + _n + "] = EXPECT_" + _n + '  + "  |  what happened: …"'),
        md("Now the errors again. Did the ones you were targeting move? Did the "
           "confusion matrix change **shape**, or did every cell shift a little? Those "
           "two call for different next moves — and whichever it was goes into the "
           "`what happened` half of `NOTES[KEY_" + _n + "]` above."),
        code("show_errors(dev, pred" + _n + ")"),
    ])

cells_04.extend([
    md("---",
       "",
       "## Step 6 — Save the trail, and stop",
       "",
       "Both tables go to disk. `05_test.ipynb` reads them back, adds the held-out row, "
       "and `06_report.ipynb` prints the two side by side as your report's prompt-iterations section.",
       "",
       "**Before you save, fill in the `what happened` half of every `NOTES` entry.** "
       "You wrote the prediction before the round; this is where you say whether it "
       "held. A row that says only what you expected is half a finding.",
       "",
       "Then decide which prompt won on dev, and check that it is **saved as a file** — "
       "the next notebook can only load files."),
    *step(
        6, "Read the trail back",
        ["Shows the rounds you ran, in order, with the score each one got.",
         "Nothing new is named — this is a check before you save."],
        signpost="First, look at what you are about to save. A round whose key says "
                 "only `round 2` is a row number; a round whose key names the move you "
                 "made is an argument. This is the last easy moment to rename one.",
        starter=[
            "f1_by_round",
        ]),
    md("Now write both tables to disk. `05_test.ipynb` reads them back and adds the "
       "held-out row; `06_report.ipynb` prints them side by side.",
       "",
       "They overwrite, because re-running this notebook re-runs every round in it — "
       "the file has to match the rounds you actually just ran, not a mixture of two "
       "sessions."),
    code('save_json(f1_by_round, ROUNDS_PATH, what="rounds", overwrite=True)',
         'save_json(NOTES, NOTES_PATH, what="round notes", overwrite=True)'),
    for_report(
        ["Our baseline scored ___ on dev.",
         "We changed ___ because ___, and expected ___; what happened was ___.",
         "The round that helped most was ___, and we think it worked because ___.",
         "One change we expected to help and it did not was ___."],
        "Write the middle two sentences once per round. The last one is worth as many "
        "marks as the rest: a prediction that failed, with a reading of why, is a "
        "finding about the model. Dropping the rounds that did not work and reporting "
        "only the winner leaves you with a number and nothing to say about it."),
    md(
        "---",
        "",
        "## 🛑 Before you open `05_test.ipynb`",
        "",
        "That notebook opens the held-out set. Once it has been scored, the number is "
        "the number — so settle these first:",
        "",
        "- Every prompt you might test is **saved as a file** in `prompts/`.",
        "- `PLAN.md` §8 says **which** of them you will test, and **how many**, and how "
        "you will pick the winner if you test more than one. Written down before you "
        "run, that is a design. Decided afterwards, it is choosing the best of several "
        "tries and reporting it as if it were one.",
        "- Every `NOTES` entry says what you expected *and* what happened.",
        "",
        "**Next:** `05_test.ipynb`."),
])

# ==================================================================================
# 05 — the held-out run
# ==================================================================================
cells_05_test = [
    title_cell(
        "05", "05_test", "The held-out run",
        "Open the test set once, score what you committed to, and stop.",
        "`data/gold/<track>_<group>_test.json` (from 03) · your prompt files · "
        "`..._rounds.json` (from 04)",
        "`outputs/<track>_<group>_predictions.json` · `..._test_log.jsonl` · "
        "`..._rounds.json` (with the test row added)",
        ["**This is the first time all week that `TEST_PATH` gets opened.** Your prompt "
         "is settled, you run it against items it has never been tuned against, and "
         "whatever comes out is what you report.",
         "",
         "Expect it to be lower than your best dev round. That is the ordinary outcome, "
         "not a failure: the gap is roughly how much of your improvement was tuning to "
         "those particular dev items rather than to the task. Reporting the gap is a "
         "stronger finding than reporting a high number, and `06_report.ipynb` asks you "
         "for it.",
         "",
         "A hosted model is only *best-effort* reproducible even at `temperature=0`, so "
         "each run is frozen to a file and every number in the report comes out of that "
         "file rather than out of this session's memory.",
         "",
         "**Nothing here stops you running it again.** Stopping you would be the wrong "
         "design — a genuine mistake at four o'clock on the last day needs a way "
         "forward. So instead nothing is overwritten, and every scoring appends a line "
         "to a log that travels in your submission. A second attempt is allowed. It is "
         "just not invisible, and the limitations section of your report has to account for it."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Files in, files out, and the connection to the model: all plumbing.",
        "from pipeline import (load_gold, load_prompt, load_json, save_json, setup,",
        "                      freeze_test_run, read_test_log)",
    ]),
    md("## Connect to the model",
       "",
       "The same settings as `04_develop.ipynb`, from the same `config.yaml`. If this "
       "line does not match the one you iterated under, the held-out score is not "
       "comparable with the dev rounds you are about to put it beside."),
    code("setup(temperature=TEMPERATURE, seed=SEED, model=MODEL)"),
    CONFIG_MD,
    md(
        "## Step 1 — Commit, before you open anything",
        "",
        "Fill this in **first**, and do not change it after the next cell has run.",
        "",
        "You may test more than one prompt on the held-out set. It is a legitimate thing "
        "to do and this notebook supports it — but it is a **different claim** from "
        "testing one, and the difference is entirely in whether you said so in advance. "
        "One prompt tested is an estimate of how that prompt does on unseen items. Three "
        "tested and the best one reported is the best of three tries, and the number is "
        "optimistic by an amount nobody can calculate afterwards.",
        "",
        "So: name every candidate here, say how you will pick between them, and **report "
        "all of their numbers in §3 — not only the winner's.** `PLAN.md` §8 should "
        "already say what you are about to type.",
        "",
        "Each candidate is a **file**, because that is the only thing that survives from "
        "the last notebook, and because a prompt you cannot produce on request is a "
        "result nobody can check.",
        "",
        "The rounds table from `04_develop.ipynb` is loaded here too, so the held-out "
        "row lands at the bottom of the dev trail rather than in a table of its own."),
    *step(
        1, "Name your candidates and your rule",
        ["Loads the dev trail from 04, and lists the prompt files you are about to",
         "test — with the rule for picking between them, written before you look."],
        "CANDIDATES, WINNER_RULE, f1_by_round, NOTES",
        decide="How many prompts you test, and how you pick the winner. Both go in "
               "your report, whichever way the numbers fall.",
        starter=[
            "# name -> prompt file. One entry is the ordinary case.",
            "CANDIDATES = {",
            '    "v1 few-shot": ROOT / "prompts" / (TRACK + ".txt"),',
            "}",
            "",
            "# If there is more than one above, how do you choose? Decide now.",
            'WINNER_RULE = "…"    # e.g. "highest macro-F1; ties go to the simpler prompt"',
            "",
            "f1_by_round = load_json(ROUNDS_PATH, what=\"rounds\",",
            '                        made_by="notebook 04_develop")',
            'NOTES = load_json(NOTES_PATH, what="round notes",',
            '                  made_by="notebook 04_develop")',
            "",
            'print(len(CANDIDATES), "candidate(s):", list(CANDIDATES))',
            'print("best dev round so far:", round(max(f1_by_round.values()), 3))',
        ]),
    md(
        "## Step 2 — Open the test set and run",
        "",
        "The cell below opens `TEST_PATH` and runs every candidate against it, freezing "
        "each one as it goes. For each, `freeze_test_run`:",
        "",
        "1. Runs the prompt over the test items.",
        "2. **Saves before scoring anything.** It never overwrites — a second run lands "
        "in `..._predictions_attempt2.json` beside the first, and both stay.",
        "3. Reads that file straight back off disk and scores *those* predictions, which "
        "checks that the file you will quote in your report is the file you think it is.",
        "4. Appends one line to the log in your submission bundle: the score, the "
        "attempt number, and a fingerprint of the prompt that produced it.",
        "5. Adds the score to `f1_by_round` and saves the table.",
        "",
        "It also freezes the model's **raw replies** beside the predictions. The "
        "predictions are our reading of those replies; if a `??` turns out to matter, "
        "the reply is the evidence and it is gone as soon as this runtime resets.",
        "",
        "**One person runs this.** It is the run you will be defending."),
    *step(
        2, "The held-out run",
        ["Opens the held-out items — the only cell all week that does — and freezes,",
         "scores and logs each candidate against them. One predictions file each."],
        "test, best_dev",
        starter=[
            "test = load_gold(TEST_PATH)      # the first time this file is opened all week",
            "",
            "# Your best DEV round, taken BEFORE any test row joins the table. Reading it",
            "# inside the loop would compare the second candidate against the first",
            "# candidate's held-out score, which is not a dev/test gap at all.",
            "best_dev = max(f1_by_round.values())",
            "",
            "# One note for every candidate: with more than one, the rule you committed",
            "# to in step 1 is the thing the log has to carry.",
            'note = ""',
            "if len(CANDIDATES) > 1:",
            "    note = WINNER_RULE",
            "",
            "for name in CANDIDATES:",
            "    print()",
            '    print("=" * 70)',
            '    print("candidate:", name)',
            '    print("=" * 70)',
            "    freeze_test_run(load_prompt(CANDIDATES[name]), test, f1_by_round,",
            "                    PRED_PATH, TESTLOG_PATH, ROUNDS_PATH, PROMPT_FILE,",
            "                    dev_f1=best_dev,",
            "                    ordered=False,   # True only if your labels are a SCALE",
            "                    labels=LABELS_ORDER,",
            '                    key="TEST · " + name,',
            "                    note=note)",
            "",
            "    # The reason this candidate was tested at all, so your prompt-iterations section has",
            "    # a line for the held-out row rather than a bare number at the bottom.",
            '    NOTES["TEST · " + name] = "held out · " + WINNER_RULE',
            "",
            'save_json(NOTES, NOTES_PATH, what="round notes", overwrite=True)',
        ]),
    md("### The log, as it now stands",
       "",
       "One line per scoring. **If there is more than one, your limitations section has to account for "
       "every one of them** — which is the whole reason this file exists rather than a "
       "lock on the cell above.",
       "",
       "Look at `prompt_sha1`. Two rows with the *same* fingerprint are one prompt run "
       "twice, which tells you something useful about how much the model varies on its "
       "own. Two rows with *different* fingerprints are two different prompts — fine if "
       "you named both in step 1, and a prompt tuned after seeing the held-out set if "
       "you did not."),
    code("read_test_log(TESTLOG_PATH)"),
    for_report(
        ["We tested ___ prompt(s) on the held-out set, decided in advance, and picked "
         "between them by ___.",
         "Their held-out scores were ___.",
         "Our best dev round was ___ and the held-out score was ___, a gap of ___, "
         "which we read as ___."],
        "The gap is the finding, in either direction. A held-out score well below the "
        "dev trail says some of the gain was tuning to those particular dev items; one "
        "that matches says the change generalised. Both are reportable, and a report "
        "that quotes only the dev number is quoting the one you cannot defend.",
        "",
        "If the log has more than one line, say here what the second run was and why."),
    md(
        "---",
        "",
        "**Next:** `06_report.ipynb`. It loads the files you just wrote and nothing "
        "else — so from here on, your numbers cannot move."),
]

# ==================================================================================
# 06 — report
# ==================================================================================
cells_06 = [
    title_cell(
        "06", "06_report", "Error analysis and export",
        "Show an item the model got wrong, and say whose fault it was.",
        "the dev/test split (03) · the rounds and notes (04) · the frozen predictions "
        "and test log (05)",
        "`outputs/` — the predictions CSV and a copy of your test set; and the numbers "
        "on screen that you write your report from",
        ["This is the highest-value part of the whole project, and the one the Q&A will "
         "definitely go to. A low F1 with a clear account of *why* is worth more than a "
         "high one without.",
         "",
         "Everything scored here is the **held-out test set**. The per-round table from "
         "notebooks 04 and 05 are your dev trail — how you got to the prompt — and the two are "
         "different claims."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Loading files is plumbing.",
        "from pipeline import (load_gold, load_predictions, load_json, save_json,",
        "                      export_results, read_test_log, label_set)",
        "",
        "# The scoring comes from scikit-learn, by its own names. You built your own",
        "# versions of these on Day 2 S6 and checked them against these very functions;",
        "# WHICH of them you report is the decision this notebook asks you to make.",
        "from sklearn.metrics import (classification_report, cohen_kappa_score,",
        "                             confusion_matrix, f1_score)",
        "",
        "# The tables: your errors, which labels the model swaps, and the join against",
        "# your coders' arguments. All met before — none of them calls the model.",
        "import pandas as pd",
        "from metrics import (show_errors, confused_pairs, errors_on_disagreed,",
        "                     labels_of)",
        "from pipeline import plot_confusion_matrix",
    ]),
    CONFIG_MD,
    md("## Step 1 — Open the frozen run",
       "",
       "Now we load the four files notebooks 04 and 05 left behind: the held-out test set, the "
       "dev half (only so you can say in the report how big it was), the predictions file, and "
       "the per-round table.",
       "",
       "**Nothing in this notebook calls the model.** If a number here differs from "
       "what notebook 05 printed, you are loading a different file — not watching the "
       "model change its mind.",
       "",
       "The last line prints three counts, and the test count and the prediction count "
       "**must be the same**. More predictions than test items means the run you froze "
       "was a run on dev — go back to step 2 of notebook 05.",
       "",
       "Loading a frozen predictions file is what you did on Day 2 S6."),
    *step(
        1, "Load the frozen run",
        ["Opens the held-out test set, the dev half, the predictions notebook 05 froze,",
         "and the per-round table."],
        "test, dev, pred_final, f1_by_round, NOTES, LABELS",
        starter=[
            "test = load_gold(TEST_PATH)     # the held-out half — everything below is this",
            "dev  = load_gold(DEV_PATH)      # only so the report can say how big it was",
            "pred_final = load_predictions(PRED_PATH)",
            'f1_by_round = load_json(ROUNDS_PATH, what="rounds",',
            '                        made_by="notebook 05_test")',
            'NOTES = load_json(NOTES_PATH, what="round notes",',
            '                  made_by="notebook 04_develop")',
            "",
            "# The label list every score below is computed over, worked out once here.",
            "# config.yaml only sets labels_order when your labels are a SCALE, so for",
            "# most schemes it is empty and the list comes off your gold set instead.",
            "LABELS = LABELS_ORDER",
            "if not LABELS:",
            "    LABELS = label_set(test)",
            "",
            'print(len(dev), "dev ·", len(test), "test ·", len(pred_final), "predictions")',
            'print("scoring over:", LABELS)',
        ]),
    md("### Your rounds, with the reason for each",
       "",
       "Now we put the two tables side by side: what each round scored, and why you made "
       "that change. **This is your prompt-iterations section**, and it is the one section you cannot "
       "reconstruct afterwards — a stack of F1 numbers with no reasons attached is a "
       "list of things that happened, not an account of what you did.",
       "",
       "A round whose reason still ends in `what happened: …` is one you have not "
       "finished. Go back to `04_develop.ipynb` and fill it in; the cell there saves it "
       "again."),
    code("for name in f1_by_round:",
         '    print(round(f1_by_round[name], 3), " ", name)',
         '    print("        ", NOTES.get(name, "— no reason recorded —"))'),
    md(
        "### Your test log",
        "",
        "How many times the held-out set was scored. **One row is what we expect.** More "
        "than one is allowed — that is why this file exists rather than a lock — but "
        "whichever row your headline comes from, your limitations section has to account for the others.",
        "",
        "Look at `prompt_sha1`. Two rows with the *same* fingerprint are the same prompt "
        "run twice, which tells you something useful about how much the model's answers "
        "vary on their own. Two rows with *different* fingerprints are a prompt "
        "that changed after you had seen the held-out set — a different thing entirely, "
        "and one you have to say out loud."),
    code(
        "read_test_log(TESTLOG_PATH)",
        ""),
    md(
        "## What you have to work with",
        "",
        "Everything below is assembled from these. Nothing here calls the model — all of "
        "it takes lists you have already loaded, which is what \"frozen\" means: from "
        "here on your numbers can only change if you load a different file.",
        "",
        "| Call | What it gives you | First run |",
        "|---|---|---|",
        "| `labels_of(test)` | the gold labels as a plain list, ready for scoring | 06 |",
        "| `classification_report(y, p, labels=LABELS)` | precision, recall and F1 for **every class** | Day 2 S6 |",
        "| `f1_score(y, p, average=\"macro\", labels=LABELS)` | one number: every class counts the same | Day 2 S6 |",
        "| `f1_score(y, p, average=\"micro\", labels=LABELS)` | one number: every **item** counts the same |Day 2 S6 |",
        "| `f1_score(y, p, average=\"weighted\", labels=LABELS)` | one number, classes weighted by how common they are | Day 2 S6 |",
        "| `cohen_kappa_score(y, p)` | agreement with your gold, corrected for chance | Day 2 S6 |",
        "| `cohen_kappa_score(y, p, weights=\"quadratic\")` | the same, counting a near miss as a smaller error | Day 2 S6 |",
        "| `confusion_matrix(y, p, labels=LABELS)` | which classes it confuses with which | Day 2 S6 |",
        "| `plot_confusion_matrix(m, LABELS, title)` | that matrix, drawn | Day 2 S6 |",
        "| `show_errors(test, pred)` | just the rows it got wrong | Day 3 |",
        "| `errors_on_disagreed(errors, disagreed)` | the errors that land where YOUR coders also disagreed | 06 |",
        "| `confused_pairs(errors)` | which label swaps the model made most often | 06 |",
        "",
        "**The three averages are three different questions, and they disagree.** Macro "
        "asks how well you do on the average *class*, so a rare class counts as much as "
        "a common one — which is why it goes with a balanced sample. Micro asks how well "
        "you do on the average *item*, so the common classes dominate. Weighted sits "
        "between them. Run all three if you like; **which one you report is a decision, "
        "and `PLAN.md` §9 should already say which.** Choosing after you have seen all "
        "three is choosing the flattering one.",
        "",
        "**Careful with the κ.** In notebook 03 it compared two **annotators**. Here it "
        "compares your gold labels against a **model**. Same kind of number, a different "
        "claim — do not swap them in the report."),
    md(
        "## Step 2 — The headline number",
        "",
        "**This one is the result.** It is measured on items your prompt was never tuned "
        "against, which is what makes it worth quoting; the per-round table above is the "
        "*story of how you got here*, and belongs in your prompt-iterations section rather than your evaluation section.",
        "",
        "Compare it to your best dev round. If test came out lower, that is the ordinary "
        "outcome and the gap is itself a finding — roughly, how much of your improvement "
        "was tuning to those particular dev items rather than to the task. Say what you "
        "make of it in one sentence. And keep the sample size in view while you do: a few "
        "points either way on twenty-odd items is noise, so read a small gap as \"we "
        "cannot tell these apart\" rather than as a result.",
        "",
        "Read the per-class table, not just the headline. *\"Which class is it worst "
        "at, and what does it confuse that class with\"* is a more useful sentence in a "
        "report than *\"F1 = .62\"*, and it is the one that says something about your "
        "scheme rather than about the model alone.",
        "",
        "**Careful with the κ.** The one you computed in notebook 03 compared two "
        "**annotators**. This compares your gold labels against a **model**. Same kind "
        "of number, different claim — do not swap them in the report."),
    *step(
        2, "Score the frozen run",
        ["Lines the gold labels up against the frozen predictions, then reports the",
         "per-class table. These are the numbers for your evaluation section."],
        "y_true, y_pred, macro_f1",
        decide="Which one number you lead with, and whether a weighted κ belongs here. "
               "Both follow from your label set, and PLAN.md §9 should already say.",
        starter=[
            "y_true = labels_of(test)      # the gold side, as a plain list",
            "y_pred = pred_final           # the model's side, already a plain list",
            "",
            "# Every class, so you can say WHICH one it is worst at. That sentence is",
            "# worth more in a report than the single number under it.",
            "print(classification_report(y_true, y_pred, labels=LABELS,",
            "                            zero_division=0))",
            "",
            "# The headline. macro = every class counts the same, which is what a",
            "# balanced sample was drawn for. Add the other averages if you want to see",
            "# them; report the one you committed to.",
            'macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS,',
            "                    zero_division=0)",
            'print("macro-F1 on the held-out test set:", round(macro_f1, 3))',
        ]),
    md("### Now the rest of what you chose to report",
       "",
       "Chance-corrected agreement with your gold, and the matrix. **Add the weighted κ "
       "only if your labels are a scale** — and if you do, `LABELS` in "
       "`config.yaml` has to be in scale order, or the weighting is computed over "
       "alphabetical order and the number means nothing.",
       "",
       "Keep whatever you decide the same as notebooks 04 and 05 used, or the held-out "
       "row stops matching the table you put it at the bottom of."),
    code("print(\"Cohen's kappa:\", round(cohen_kappa_score(y_true, y_pred), 3))",
         "",
         "# Labels on a scale? Then this one too — a near miss counts as a smaller error.",
         "# print(\"weighted kappa:\", round(cohen_kappa_score(",
         '#     y_true, y_pred, labels=LABELS, weights="quadratic"), 3))',
         "",
         "matrix = confusion_matrix(y_true, y_pred, labels=LABELS)",
         'plot_confusion_matrix(matrix, LABELS, "Gold vs model, held-out set")'),
    for_report(
        ["We report ___ as our headline number, because ___.",
         "We did not lead with ___ because ___.",
         "The model was worst on ___, which it confused with ___.",
         "We read that as ___ about our scheme."],
        "The second sentence is the one that shows you chose rather than accepted a "
        "default, and you can write it from `PLAN.md` §9 before the numbers exist. The "
        "last two come off the per-class table and the matrix, and they say something "
        "about your labels that the headline number cannot.",
        "",
        "On a sample this size, a few points either way is noise. \"We cannot tell "
        "these apart\" is an honest sentence and it scores better than a difference you "
        "cannot support."),
    md(
        "---",
        "",
        "# Step 3 — Error analysis",
        "",
        "**This is the part of the project the Q&A will actually go to.** A low F1 with "
        "a clear account of *why* beats a high one without, every time — and the "
        "account is only available to you because you built the gold set yourselves.",
        "",
        "`show_errors` gives you every item the model got wrong. The question to ask of "
        "that table is the Day 2 S6 one: **is this the model's fault, or the scheme's?**",
        "",
        "You are one of the few people who can answer it, because you built the gold set "
        "yourselves and you know which items you argued about. An item both your coders "
        "labelled at once and the model still missed is the model's. An item your coders "
        "split on is a boundary your scheme does not settle, and the model splitting on "
        "it too is evidence rather than coincidence.",
        "",
        "So this step works the same way notebook 03 step 2 did — find the label pair "
        "that keeps swapping, then say what is true of that boundary."),
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
       "the per-class table scored worst.",
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
       "part of the project. Re-run its step 3 — step 4 below asks whether your "
       "coders and the model split on the same items, and without this file there is "
       "nothing to answer it with."),
    code("disagreed = load_json(DISAGREED_PATH)   # written by notebook 03, step 3",
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
        "## Step 4 — Which boundary is this, and is it the same one?",
        "",
        "Two questions, and the notebook computes the numbers for both. What it cannot "
        "do is say what they mean, and that is the last genuinely analytical thing in "
        "the project. Do it **together, out loud, reading the actual sentences** — it "
        "takes about fifteen minutes and it is the single hardest thing to reconstruct "
        "a week later.",
        "",
        "**First: which two labels does the model confuse most often?** `confused_pairs` "
        "counts the `gold -> pred` swaps, commonest first. This is the same reading you "
        "made of the coder-vs-coder matrix in notebook 03 step 2, made now of the model.",
        "",
        "**Then: is it the same pair your own coders disagreed about?** `overlap` from "
        "step 3 already has the ids. If the two answers point at the same boundary, you "
        "have something worth saying: two people who read the guidelines and one model "
        "that did not all failed at the same place, so the problem is in what the scheme "
        "says, not in who or what was reading it. That is a better finding than a clean "
        "F1, and nobody who did not build their own gold set can report it.",
        "",
        "If they point at different boundaries, that is reportable too, and it means "
        "something else — the model is failing somewhere your coders found easy.",
        "",
        "Read the sentences behind the commonest pair before you write anything. The "
        "count tells you where to look; it does not tell you what is there."),
    *step(
        4, "The boundary the model gets wrong",
        ["Ranks the label swaps the model made, then says how many of its errors land",
         "on the items your own coders argued about."],
        "pairs",
        starter=[
            "pairs = confused_pairs(errors)",
            "",
            "# How much of the error set sits on items your coders split on too.",
            'print(len(overlap), "of", len(errors), "errors are on rows you argued about")',
            "",
            "pairs",
        ]),
    md("### Now read the items behind that pair",
       "",
       "The commonest swap, as sentences. Change the two labels to look at any other "
       "pair in the table above — the one that surprises you is often worth more than "
       "the one that is biggest."),
    code("# The commonest swap, from the table above. Or type your own two labels.",
         'GOLD_IS, PRED_WAS = pairs.iloc[0]["gold"], pairs.iloc[0]["pred"]',
         "",
         'print("items labelled", GOLD_IS, "that the model called", PRED_WAS)',
         'errors[(errors["gold"] == GOLD_IS) & (errors["pred"] == PRED_WAS)]'),
    for_report(
        ["The model most often labelled ___ items as ___, ___ times out of ___ errors.",
         "___ of its errors fell on rows our own coders had disagreed about.",
         "We read that as ___, and our scheme would have to ___ to settle those items.",
         "With another week we would ___, because ___."],
        "The second sentence is the one this whole project was built to let you write. "
        "Your coders' disagreements are independent evidence that a boundary is unclear "
        "— evidence collected before anyone saw what the model would do with it.",
        "",
        "The third is where the two readings meet: if notebook 03 named the same pair, "
        "say so explicitly. If it named a different one, say that instead — it is just "
        "as real a result, and pretending otherwise is visible in the Q&A."),
    md(
        "## Step 5 — Export",
        "",
        "Writes two files, both stamped with your group name: your test set, and a "
        "per-item predictions CSV with one row per item — id, gold label, predicted "
        "label, whether they matched, and the text.",
        "",
        "These are what make your reported F1 checkable by someone else: which items "
        "it was measured on, and what the model said about each one. The CSV is also "
        "what you sort to find the misses for your error analysis.",
        "",
        "With `dev=dev`, the saved items are named `_test` rather than `_gold`, because "
        "with a split the half you scored is the test half and calling it \"gold\" "
        "would misdescribe it.",
        "",
        "**The report itself you write, in Word.** Nothing here drafts it for you. "
        "Everything it needs is on screen in this notebook: the rounds table in step 1, "
        "the per-class scores and confusion matrix in step 3, and the errors in step 4."),
    *step(
        5, "Export",
        ["Writes two files into `../outputs/`: your test set, and the per-item",
         "predictions CSV."],
        starter=[
            "export_results(TRACK, test, pred_final, OUT_DIR,",
            "               group=GROUP, run=RUN, dev=dev)",
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
        "the *Final mini-project* assignment in Google Classroom → **Turn in**. One zip "
        "per group, with every member's name in `PLAN.md`.",
        "",
        "**Your two-page report is not in that zip.** You write it yourself, in Word, "
        "from the numbers this notebook printed above, and upload it to Classroom "
        "separately — one per person, not one per group.",
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
def write(name: str, cells: list[dict]) -> None:
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


# ==================================================================================
# 02b — add more items to a sample you have already started annotating
# ==================================================================================
cells_02b = [
    title_cell(
        "02b", "02b_add_samples", "Add more items to the sample you already have",
        "More annotated items is more evidence — drawn without repeating what you have.",
        "`data/pools/<track>_pool.json`, `data/gold/<track>_<group>_sample.json`, "
        "and the sheet link from 02",
        "the same `_sample.json`, enlarged, and new rows in the sheet you are "
        "already annotating in",
        ["**Open this only if you have time left over**, and only once the first draw "
         "is annotated or nearly so. It does not replace `02_sample.ipynb` and it does "
         "not make a second sheet — it adds rows to the bottom of the one your group is "
         "already working in.",
         "",
         "Nothing already in the sheet is renumbered, moved or overwritten. The new "
         "items get ids that carry on from the highest one you have.",
         "",
         "**One person runs this.** Everyone else can keep annotating in the sheet "
         "while it runs; the new rows appear below the ones they are working on.",
         "",
         "> If you moved columns around in the sheet, or added your own, that is fine — "
         "every column is found by the name in row 1. What is not fine is renaming "
         "`ID`, `Text`, `Label` or `Note`: notebook 03 needs those names too."]),
    md(*SETUP_MD),
    setup_cell([
        "",
        "# Same split as notebook 02: reading and writing files is plumbing, so it is",
        "# imported, and the sampling you have to defend is defined further down.",
        "from pipeline import load_gold, save_json, label_set",
        "from annotate import (append_to_annotation_sheet, remembered_sheet,",
        "                      tab_names, load_coder_sheets)",
    ]),
    CONFIG_MD,
    md(
        "## What a second draw does, and what it does not",
        "",
        "More annotated items is more evidence: your F1 rests on a bigger sample and "
        "your agreement number gets steadier. That is real, and it is why this notebook "
        "exists.",
        "",
        "Be careful about one thing when you write it up. Your gold set was now built "
        "in **two draws**, and they are only one sample of the pool if you drew them the "
        "same way, for the same reason. So:",
        "",
        "- **Same strategy, more items.** Say how many you drew in each round and why "
        "you stopped where you did. This is the simple case.",
        "- **A different strategy the second time.** That is a two-stage design, and "
        "your report has to describe it as one. It is a legitimate choice — starting "
        "balanced and topping up by document is a real thing to want — but it is not "
        "something to leave for the reader to notice from the counts.",
        "",
        "Either way, both rounds go in `PLAN.md`: strategy, size, seed, and the reason.",
        "",
        "> **A label that has run out.** If the pool has no items left under some label, "
        "a balanced top-up simply will not contain it and your combined sample stops "
        "being balanced. The cells below say so when it happens. That belongs in your "
        "limitations — do not switch strategy to hide it."),
    lead("First we open the pool and the sample you already have. Both come off disk: "
         "the session that drew the first sample is long gone, and this is what it left "
         "behind."),
    code("pool = load_gold(POOL_PATH)",
         "sampled = load_gold(SAMPLE_PATH)",
         "",
         "highest = 0",
         "for item in sampled:",
         '    highest = max(highest, int(item["id"]))',
         "",
         'print("pool:", len(pool), "items")',
         'print("already sampled:", len(sampled), "items, ids 1 to", highest)'),
    lead("Now we find the sheet your group has been annotating in, and list its tabs. "
         "Read that list before going on: the new rows are added to exactly these tabs, "
         "and a coder whose tab is not named in `CODERS` in `config.yaml` would not get "
         "them."),
    code("SHEET_ID = remembered_sheet(SHEET_PATH)",
         'print("sheet:", SHEET_ID)',
         'print("tabs in it:", tab_names(SHEET_ID))',
         'print("tabs the new rows will go in:", list(CODERS), "+ Final")'),
    *study_cells(
        "## The code that draws the extra items — read it, then run it\n\n"
        "Two functions. Neither draws anything new by itself: the actual drawing is "
        "done by the same `sample` you read and ran in notebook 02, over a smaller "
        "pool.\n\n"
        "Run `help(sample)` if you want the three strategies again — they have not "
        "changed. To read the code itself, open `scripts/pipeline.py` from the "
        "**Files** panel on the left.",
        [(pipeline.remaining_pool,
          "**`remaining_pool`** is the answer to *which items have we already got?* — "
          "which is harder than it sounds, because the ids in your sample file are 1, "
          "2, 3 … and the ids in the pool are not. It matches on `source_id`, the id "
          "each item had in the pool, and on the text as well. An item that matches "
          "either is left out, so nothing you have annotated can be drawn twice."),
         (pipeline.sample_more,
          "**`sample_more`** is the one you call. In: the pool, the items you already "
          "have, and the same three arguments as notebook 02. Out: **only the new "
          "items**, numbered on from your highest id. Step 3 is one call to the same "
          "`sample` you already read; the rest is checking. Step 5 refuses outright if "
          "any new item repeats an id, a `source_id` or a text you already have — two "
          "rows sharing an id are merged into one when the sheet is read back, which "
          "would cost you annotation you had already done.")],
        check="help(sample_more)"),
    md("### Choose a strategy again — the same decision as notebook 02",
       "",
       "| strategy | what it draws | what it means the second time |",
       "|---|---|---|",
       "| `balanced` | up to `n` more of **each** label | keeps the combined sample "
       "level, as long as no label has run out |",
       "| `random` | the same total, ignoring labels | keeps the pool's own imbalance; "
       "the top-up looks like the corpus |",
       "| `by_document` | whole passages (`cars50` · `raamove`) | new documents, so the "
       "combined sample covers more texts |",
       "",
       "Use a **different seed** from your first draw. Both go in the report: a sample "
       "nobody can redraw is a sample nobody can check, and there are two draws to "
       "redraw now.",
       "",
       "Write the second strategy and the reason in `PLAN.md` §5, beside the first — "
       "**even if it is the same word.** \"We drew 20 more the same way, because we had "
       "time and wanted a steadier κ\" is a complete answer; leaving the section as it "
       "was is not, because it now describes half of what you did."),
    *step(
        1, "Draw the extra items",
        ["Draws more items from the part of the pool you have not already sampled, and",
         "numbers them on from your highest id. Nothing is written yet."],
        "extra",
        signpost="This cell only draws. Nothing reaches the sheet or the disk until the "
                 "steps below, so run it, read the counts, and run it again with "
                 "different numbers if you do not like them.",
        starter=[
            "# The same one-word choice as notebook 02, made again.",
            'STRATEGY = "balanced"     # "balanced" · "random" · "by_document"',
            "N_MORE = 5                # how many MORE per label",
            "",
            "# A different seed from your first draw. Record both in the report.",
            "extra = sample_more(pool, sampled, STRATEGY, N_MORE, SEED + 1)",
        ]),
    md("### Check the draw before it touches anything",
       "",
       "The same three questions as notebook 02, asked of the combined sample this "
       "time. If the counts are not what you wanted, change the numbers above and run "
       "that cell again — nothing has been written yet."),
    *step(
        2, "Check what you drew",
        ["Prints the per-label counts of the new items, the ones you already had, and",
         "the two together. Nothing new is named — this is a check."],
        starter=[
            "def count_labels(items):",
            "    counts = {}",
            "    for item in items:",
            '        label = item["label"]',
            "        if label not in counts:",
            "            counts[label] = 0",
            "        counts[label] = counts[label] + 1",
            "    return counts",
            "",
            'print("you had: ", count_labels(sampled))',
            'print("adding:  ", count_labels(extra))',
            'print("total:   ", count_labels(sampled + extra))',
            'print("left over for few-shot examples:",',
            "      len(pool) - len(sampled) - len(extra))",
        ]),
    md(
        "## Now write it down — sheet first, file second",
        "",
        "The next three cells do the writing, and the order matters.",
        "",
        "1. **Keep a copy** of the sample as it is now, so a top-up that goes wrong can "
        "be undone.",
        "2. **Add the rows to the sheet.** Every tab is checked first, and if any tab "
        "cannot be written to safely then *no* tab is written to — you get the rows "
        "printed out to paste in by hand instead.",
        "3. **Save the enlarged sample file.**",
        "",
        "The sheet goes before the file on purpose. If the sheet write fails, nothing "
        "on disk has changed and you can simply run these cells again. If it succeeds "
        "and the file save then fails, the two disagree — but notebook 03 *tells you* "
        "so, because it matches the sheet's rows against this file by id. The other "
        "order would leave you with a file claiming rows the sheet never got, which "
        "shows up as rows that stay blank forever and nothing saying why."),
    *step(
        3, "Keep a copy of the sample as it is",
        ["Writes the sample as it stands now to a second file, before step 5 replaces",
         "the original. This also refuses if you have run this notebook before."],
        starter=[
            "save_json(sampled, SAMPLE_BEFORE_TOPUP_PATH,",
            '          what="the sample before the top-up")',
        ]),
    lead("If that cell refused, you have run this notebook before. Do not pass "
         "`overwrite=True` to get past it without checking — read the sheet first and "
         "work out whether the rows are already in there. Adding them twice is much "
         "harder to undo than working out what happened."),
    *step(
        4, "Add the rows to the sheet",
        ["Adds one row per new item to the bottom of every coder tab and the Final tab.",
         "Checks every tab first: if any of them fails, nothing is written anywhere."],
        starter=[
            "append_to_annotation_sheet(SHEET_ID, extra,",
            "                           coders=CODERS,",
            "                           # catches a sheet that has drifted from the file",
            "                           expected_rows=len(sampled))",
        ]),
    *step(
        5, "Save the enlarged sample",
        ["Replaces the sample file with the old items plus the new ones.",
         "Overwriting is deliberate here — step 3 kept the copy."],
        "sampled_all",
        signpost="The rows are in the sheet, so now the file has to match them. This is "
                 "the one cell in the project that overwrites a file you already have, "
                 "and it is why step 3 kept a copy first.",
        starter=[
            "sampled_all = sampled + extra",
            "save_json(sampled_all, SAMPLE_PATH, what=\"sampled items\",",
            "          overwrite=True)   # step 3 kept the old one",
        ]),
    md("### Check the sheet and the file agree",
       "",
       "This reads both back and compares them. It is the one check that the file "
       "notebook 03 will work from holds the same items as the sheet you are annotating "
       "in.",
       "",
       "A ✗ on the last line is the serious one: two rows sharing an id are merged into "
       "one when the sheet is read back, so it would quietly cost you annotation."),
    *step(
        6, "Read both back and compare",
        ["Reads the sheet and the sample file off disk and checks they hold the same",
         "ids. Prints a ✓ or a ✗ per check. Nothing new is named — this is a check."],
        starter=[
            "rows = load_coder_sheets(SHEET_ID, CODERS)",
            "on_disk = load_gold(SAMPLE_PATH)",
            "",
            "sheet_ids = []",
            "for row in rows:",
            '    sheet_ids.append(int(row["ID"]))',
            "file_ids = []",
            "for item in on_disk:",
            '    file_ids.append(int(item["id"]))',
            "",
            "def tick(passed):",
            '    return "✓" if passed else "✗"',
            "",
            'print(tick(len(sheet_ids) == len(file_ids)),',
            '      "sheet has", len(sheet_ids), "rows · file has", len(file_ids))',
            'print(tick(set(sheet_ids) == set(file_ids)),',
            '      "the sheet and the file hold the same ids")',
            'print(tick(len(sheet_ids) == len(set(sheet_ids))),',
            '      "no id appears twice in the sheet")',
        ]),
    md(
        "---",
        "",
        "## 🛑 This notebook is finished. Go and annotate the new rows.",
        "",
        "- **Annotate the new rows only.** Everything above them is already done.",
        "- **Spell the labels the same way** as the first round. `to_canonical` will "
        "tell you about a typo, but a label spelled two ways is two labels until then.",
        "- **Check the drop-down reached the new rows.** If your `Label` column has one, "
        "click a `Label` cell in a new row. Google Sheets does not always carry a "
        "drop-down down to rows added later; if it is missing, copy a `Label` cell from "
        "a row above and paste it over the new ones. Same for any colour rules you set "
        "up.",
        "- **Both coders again.** A row labelled by one person does not contribute to "
        "agreement, and a gold set where half the items were double-coded and half were "
        "not is one you have to explain.",
        "",
        "> **Do not run this notebook again**, and do not run `02_sample.ipynb` again "
        "either. When the new rows are annotated, go back to `03_annotate.ipynb` — it "
        "needs no changes at all, and picks up the enlarged sample and the same sheet.",
        "",
        "**Next:** `03_annotate.ipynb`, once the new rows are labelled too."),
]

write("02_sample.ipynb", cells)
write("02b_add_samples.ipynb", cells_02b)
write("03_annotate.ipynb", cells_03)
write("04_develop.ipynb", cells_04)
write("05_test.ipynb", cells_05_test)
write("06_report.ipynb", cells_06)
