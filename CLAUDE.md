# CLAUDE.md

Guidance for working in this repository.

## What this is

The group final-project scaffold for **Linguistic Data Analysis II** (Tohoku University,
Graduate School of International Cultural Studies). A group copies this whole folder into
their shared Google Drive, edits `config.yaml`, and works through notebooks 01-05: build a
pool, draw a balanced sample, annotate it to a gold standard, prompt an LLM against it,
and report precision/recall/F1.

## Who reads this code

Graduate applied-linguistics students with **almost no Python background**, working in
Google Colab, often for the first time. Assume the reader is guessing at what any given
line does and cannot tell a bug from something they typed wrong.

That audience is the design constraint. It outranks robustness, generality, and
cleverness.

## Say it plainly — no metaphors

Follows from the audience above. Many readers are working in a second language, and a
figure of speech costs them a translation step and buys nothing.

**Never dress a technical statement up as an image, a wager, or a piece of equipment.**
Say what the thing is and what it is for.

- ❌ "get one honest number to beat" → ✅ "a first score, before you change anything"
- ❌ "your steering wheel, not your result" → ✅ "use it to decide what to change next; do
  not report it"
- ❌ "a re-roll of the dice" → ✅ "the same prompt run a second time"
- ❌ "reading it back is not superstition" → ✅ "reading it back checks that the file you
  will report from is the file you think it is"
- ❌ "F1 is the scoreboard" · "the model's blind spot" · "the prompt is a contract"

The same rule covers flourishes that carry no information: no "and that is the whole
trick", no "here is where it gets interesting", no one-word sentences for emphasis.
Emphasis is **bold**, not drama.

This binds every kind of text here — markdown cells, code comments, error messages,
docstrings a student reads, and README prose.

## Simplicity comes first

Prefer the fewest moving parts that work.

- **Do not add validation, abstraction, configuration, or error handling that no observed
  failure calls for.** A value that *could* be malformed is not a reason to check it. The
  cost of a check is not the lines — it is one more concept in front of a student who has
  not got a mental model yet.
- Add a check only when, without it, a student would **lose work** or **silently get wrong
  numbers**. Those two are worth real machinery. Nothing else is.
- Basic functional style. No classes, no decorators, no nested comprehensions, no
  `*args`/`**kwargs` in anything a notebook calls. Explicit loops beat clever one-liners.
- One function should do one nameable thing, and its name should say it.

When in doubt, cut it. A student who reads a short file and understands it is further
along than one protected by code they cannot read.

## Signatures — type hints and docstrings

Students call these functions from a notebook and cannot see the file. The signature and
the docstring are the only description they get, so both are written for `Shift+Tab`.

Every `def` in `config.py` and `scripts/*.py` carries a type hint on **every** parameter
and on the return. Builtin generics, spelled out in full — no `typing` import, no
aliases, no `Any`. `-> None` when the function only prints. The recurring shapes:

| Thing | Hint |
|--|--|
| one item | `dict[str, str]` (`{"id", "text", "label"}`; id may be `int`) |
| gold, a pool, a sample | `list[dict[str, str]]` |
| predictions | `list[str]` — one label per gold item, in the same order |
| a label list | `list[str]` |
| a path | `str` |

This is an addition to the functional-style rule above, not an exception to it: hints
describe the data, they do not introduce classes, generics of our own, or protocols.

Three things are left unannotated, because writing them costs an import and a concept
without telling a student anything they can act on:

- **A parameter that takes a function** (`generate_text=`, `check(description,
  function)`) — would need `collections.abc.Callable`.
- **A third-party object we only pass through** — a gspread worksheet, an ElementTree
  element, sklearn's confusion matrix. Naming these means importing them for the
  annotation alone, and guessing (`list[list[int]]` for a matrix) would be wrong.
- **A value whose type depends on which key was asked for** — `_setting` reads
  config.yaml, where `seed` is an int and `members` is a list.

Their `Args:` and `Returns:` lines say what to pass and what comes back instead.

Docstrings are Google style, so Colab renders a parameter list rather than one run-on
paragraph. The existing plain second-person register stays as the summary and the
"why / what goes wrong" paragraphs; the sections go below them:

```python
def sample_pool(pool: list[dict[str, str]],
                n_per_class: int,
                seed: int = 42) -> list[dict[str, str]]:
    """Pick up to n_per_class items for EACH label, chosen at random.

    Rare labels simply give fewer items - that is a property of the data.
    The same seed always gives the same sample.

    Args:
        pool: items to draw from, each {"id", "text", "label"}.
        n_per_class: how many to take per label.
        seed: same seed gives the same draw; a different seed gives a different one.

    Returns:
        The drawn items, in pool order.

    Example:
        >>> sample = sample_pool(pool, 5)
    """
```

Order: summary · blank line · existing prose · `Args:` · `Returns:` · `Raises:` (only
where the function raises deliberately) · `Example:`. `Example:` is required on every
helper listed on the pipeline cheat-sheet; private `_helpers` do not need one.

**A hinting pass may not change a call form.** Hints must not move, rename, or add a
parameter — see the call-form rule below. Run `python scripts/_check_call_forms.py`
after touching any signature.

## Errors are teaching text

Every message a student can actually hit must say three things: what happened, which file
to open, and what to put in it. Plain language — no traceback vocabulary, no type names,
no "invalid input".

Compare:

- ❌ `ValueError: invalid track`
- ✅ `config.yaml says track: cars5o, and there is no pool by that name. Open config.yaml and check it against your PLAN.md.`

## Never destroy student work

A gold set is a morning of two people's annotation. Losing it is the worst thing this
template can do, and it happens through an ordinary-looking re-run of a notebook.

**Saves refuse to overwrite an existing file.** They stop, name the file and when it was
written, and offer the two ways forward: bump `run:` in `config.yaml` to keep both, or
pass `overwrite=True` to replace it deliberately. This is the one place extra machinery
earns its keep.

## Layout

- `config.yaml` — **the only file students edit** in the plumbing. Eight settings.
- `config.py` — reads `config.yaml` and derives every path. Students never edit it, and
  never type a file path in a notebook; the file notebook 02 writes is by construction the
  file notebook 03 opens.
- `notebooks/` — **generated. Do not hand-edit.** Edit `scripts/_generate_pool_notebooks.py`
  or `scripts/_generate_project_notebooks.py` and re-run. `scripts/_setup_cell.py` builds
  the shared SETUP cell; `scripts/_check_notebooks.py` asserts every
  cell runs, fits a screen and has a lead-in; `scripts/_check_call_forms.py` validates that the notebooks only
  call functions in the forms they teach.
- `scripts/pipeline.py`, `metrics.py`, `annotate.py` — the functions notebooks call.
- `prompts/`, `data/pools/`, `data/gold/`, `outputs/` — one file per track, named from
  `config.yaml`.

## What is imported, what is on screen, and what the student writes

This template is a **scaffolded project walkthrough**, not a tutorial. Groups have two days,
so the scaffold carries the mechanics — paths, file naming, the Sheets round-trip, API
pacing, the order the notebooks run in, what each hands to the next. That is what makes a
real study fit in the time.

What it must **not** carry is the method. Which agreement statistic, what counts as a
disagreement, which prompt move next, which number the report leads with: those are what the
project assesses, and a scaffold that pre-fills them leaves nothing to do but run cells.

Four tiers, and every function belongs to exactly one:

| Tier | What | Where it lives |
|---|---|---|
| **1 · Infrastructure** | the Sheets round-trip, the API backend, pacing and retry, paths, `save_json` / `load_gold`, `freeze_test_run`, `export_results`, `plot_confusion_matrix` | `scripts/`, imported, unread |
| **2 · The real library** | `classification_report`, `f1_score`, `cohen_kappa_score`, `confusion_matrix` | scikit-learn, **called by its real name** |
| **3 · Small algorithms carrying a judgment** | `percent_agreement`, `disagreements`, `column`, `show_errors`, `extract_label`, `build_fewshot` | in the notebook, as code |
| **4 · The composition** | the sequence of tier-2 and tier-3 calls | written by the student |

**The test for tier 1 against tier 3:** if a reader of the composed cell would have to open
the function to know what *method* was used, it is hiding a decision and belongs in the
notebook. If opening it would only show retry logic or path handling, it belongs in
`scripts/`.

**No course-specific wrapper may exist for anything scikit-learn already does.** A private
name for `f1_score` teaches an API nobody will use again, in place of the one the field uses
— and the course already treats sklearn as the ground truth, since Day 2 S6 has students
check their hand-built metrics against it. `evaluate` and `annotator_agreement` survive in
`scripts/` only because Days 2–3 call them and `_check_call_forms.py` requires those forms
to keep working; notebooks 03 and 06 do not call them.

**Adapting is editing the cell.** Tier-3 code is on screen, so there is no import to shadow,
no `importlib.reload`, no runtime restart, and nothing in the shared Drive folder that one
member can break for the group. The student's version is in the notebook they submit, which
is where a change of method should be visible, and `scripts/` stays pristine.

`scripts/_study.py` is the single source of tier-3 code. Neither repository imports it —
`inspect.getsource` renders it into the notebooks at generation time, so a notebook
*contains* the code rather than calling it. It is vendored byte-identical into the course
repo's `sources/notebooks/`, and `scripts/_check_study_source.py` fails if the two drift.

**One injection point is required.** `run_prompt` is tier 1 (its loop is pacing and retry)
but it calls `extract_label`, which is tier 3. Left alone, a student who edits
`extract_label` in a cell gets the old one anyway, silently, and reports numbers their own
code did not produce. So `run_prompt` takes `extract=None`. This is one of the two cases
"Simplicity comes first" says deserves real machinery — without it a student *silently gets
wrong numbers*.

## Where a decision may be placed

**Looking is allowed when it changes your next action. Looking is not allowed when it
changes what you claim.**

Reading the coder confusion matrix to decide which label boundary to argue about is exactly
what it is for. Choosing which agreement statistic to report *after* seeing what each one
gives you is selective reporting, and no reader of the finished report could detect it.

So a decision that determines a reported number goes where the inputs are known in advance:
`PLAN.md`, `config.yaml`, or a notebook section that runs before the number exists. Never
build a cell that prints every option and then asks the student to pick one.

The justification is **prompted, never stored**. A section that hands back a decision ends
with `for_report(...)` — a markdown cell with a half-written sentence frame, addressed to
the report and the Q&A. No `WHY` variables, no `save_decisions()`: the rubric and
`make_submission.py` already grade the one place that matters, and a second enforcement
mechanism only means writing the same sentence twice.

## One thing per cell, and say what it is first

Two rules about the generated notebooks, both checked by `scripts/_check_notebooks.py`:

- **A code cell does one nameable thing.** Not "load the pool and count the labels" —
  two cells. Not four functions stacked in one embedded-source cell — four cells. The
  only exemptions are the SETUP cell, which is honestly labelled as plumbing, and a cell
  holding exactly one `def`, which cannot be split any further.
- **Every code cell has a markdown cell above it that says what is about to happen**, in
  the form *"Now we will do X"*, as its last and plainest sentence. The rationale can be
  as long as it needs to be; the signpost goes at the bottom of it, next to the code.

Two consequences worth stating outright, because they cost real money and real work:

- **A cell that calls the model gets a cell to itself.** If scoring shares a cell with
  `run_prompt`, then re-reading your results re-sends every item, and in step 4 of
  notebook 04 it re-runs the audited held-out run.
- **Never offer a choice as two live assignment lines** with "delete the one you are not
  using". The second silently wins. Use a named choice and an explicit `if`.

## No audience labels

This whole repository is distributed to students, so **assume they read every file in it.**
Never describe a file as "instructor-facing", "instructor-only", or "for students" — not in
prose, headings, table columns, directory-tree comments, docstrings, or notebook cells.
Describe files by what they do and when you would open them instead. If something genuinely
must not reach students, `.gitignore` it; a label is not access control.

Writing *to* or *about* the instructor as a person is fine ("report the result to the
instructor").
