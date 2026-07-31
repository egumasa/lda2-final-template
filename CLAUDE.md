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

- `config.yaml` — **the only file students edit** in the plumbing. Seven settings.
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
