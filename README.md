# LDA II — Final mini-project template

The scaffold for the final group mini-project in **Linguistic Data Analysis II**. You run
a small LLM-annotation study end to end, across six numbered stages:

```
01 build pool  →  02 sample  →  03 annotate  →  04 develop  →  05 test  →  06 report
```

> **build** a pool from a real corpus → **choose how to sample** it and defend that →
> **annotate and adjudicate** it yourselves → **split it into dev and test** → write a
> **baseline prompt** → **read what it got wrong** and iterate on the *dev* half over 2–3
> rounds → **freeze** one run on the *held-out* half → **read what it got wrong** →
> **export the results you write your two-page report from**

The plumbing is written for you. What you supply is the judgment: which items, which
labels, which prompt, and what the errors mean.

Note where the model first appears — **notebook 04**. Notebooks 01–03 are the study; the
LLM is the thing being measured by it.

## What's in here

```
lda2-final-template/
├── config.yaml                     # ✏️ the ONE file you edit first: track, group, seed
├── config.py                       # turns config.yaml into every path (you do not edit it)
├── notebooks/                      # YOUR six stages, ten files. Run them in order.
│                                   #   (01_build_pool_* has one variant per track —
│                                   #    you run only the one matching your track)
│   ├── 01_build_pool_<track>.ipynb #   corpus → data/pools/<track>_pool.json
│   ├── 02_sample.ipynb             #   pool → your sample → the annotation sheet
│   ├── 02b_add_samples.ipynb       #   time left over? more items into the SAME sheet
│   ├── 03_annotate.ipynb           #   the filled-in sheet → YOUR gold standard,
│   │                               #   split into dev and test
│   ├── 04_develop.ipynb            #   dev → prompt rounds → the versions you might test
│   ├── 05_test.ipynb               #   the held-out run, once, frozen to a file
│   └── 06_report.ipynb             #   the frozen run → error analysis + report
├── scripts/                        # the code the notebooks call. You do NOT edit these —
│                                   #   but see scripts/README.md for the parts to READ
│   ├── pipeline.py                 #   load, sample, prompt, freeze, export
│   ├── metrics.py                  #   precision/recall/F1, kappa, confusion matrix
│   ├── annotate.py                 #   the Google Sheets annotation round-trip
│   ├── reshape.py · download.py    #   turning each raw corpus into {id, text, label}
│   ├── prep_datasets.py            #   one command to rebuild a track's pool
│   └── make_submission.py          #   collect what you hand in
├── data/
│   ├── pools/                      # the pools you build (git-ignored), full-size
│   │                               #   plus optional small <track>_demo_pool.json
│   ├── gold/                       # YOUR sample, YOUR gold set, and its dev/test
│   │                               #   split (git-ignored)
│   └── raw/                        # original downloads (git-ignored)
├── prompts/<track>.txt             # your prompt lives here, as a file you edit
└── outputs/                        # predictions, the test-scoring log, report, figures
```

You edit **`config.yaml`**, the **✏️ cells** in the notebooks, and your **prompt file**.
Nothing in `scripts/`.

## What is left to you, and why

**Every cell is complete and runs as written.** There are no blanks to fill in. Each ✏️
cell arrives with a real, defensible value already in it, a short header saying what the
cell does and what it names for later cells, and a markdown cell above it explaining what
the choice is and what turns on it:

```python
# ══ STEP 2 · Draw your sample ═════════════════════════════════════════════
# Draws the sample using whichever of the three strategies you name, and prints
# how many items landed under each label.
# Creates: sampled

# ✏️ this runs as written — the work is deciding whether it should

STRATEGY = "balanced"     # "balanced" · "random" · "by_document"
```

A blank standing in for an argument would test whether you can copy a word out of the
comment beside it, which is not a decision. What is left to you is the **decision itself**
— which strategy, which labels, where a scale gets cut, which of your errors is the
scheme's fault — and the sentence that defends it. **Leaving a value alone is a choice
too**, and it needs the same defence as changing it.

`Creates:` is there because the next cell expects that exact name, and a typo in it costs
you an afternoon.

Two kinds of decision, asking for different things:

- **In 01**, the ✏️ cells are the *reshaping decisions*: what each label is called, which
  annotations to trust, how fine-grained the scheme is, where a numeric scale gets cut.
  The download and the parsing are written for you; nobody learns anything from retyping
  a `User-Agent` header. `PLAN.md` asks you to justify what you chose.
- **In 02–05**, they are the *research decisions*: how to draw the sample and why, what to
  change in the prompt and what the errors told you to change, and which of the model's
  mistakes are your scheme's fault rather than the model's. Every call has the **same form
  it had in Days 1–3**. If something you typed in the tutorials does not work here, that
  is a bug in this template, not in your memory of it — please say so.

### The code that does the work is in the notebook, not hidden

Notebooks 02, 04 and 05 each print the functions they are about to call — the three
sampling strategies, `run_prompt` and `extract_label`, `evaluate` and `show_errors` —
read straight out of `scripts/` when the notebook is generated. Not a simplified copy:
the code that runs.

That is because those functions *are* your method. "We drew a balanced sample" and "we
scored it with macro-F1" are claims you have to defend, and the commented steps inside
`sample_pool` are the honest answer to how. The plumbing around them — retries,
path checks, the Sheets round trip — stays imported, because nobody learns anything from
reading it. See [`scripts/README.md`](scripts/README.md) for which is which.

### Each notebook hands a file to the next

02 loads what 01 wrote; 03 loads what 02 wrote, and so on. That is not ceremony. Your
group works across several days and several people's Colab runtimes, and a variable in
someone else's session is not a handoff. The 02/03 boundary is the clearest case: 02
draws the sample and makes the sheet, then *days* pass while two of you annotate, and 03
picks up from the file. It also means a group that stalls in 03 can be handed a gold set
and carry on in 04, and that "does it run top to bottom on a fresh runtime?" is a
question you can actually check.

## Run it in Google Colab

Your group works in **one shared Google Drive folder**, and every notebook's Setup cell
goes and finds it. That is not a convenience: a Colab runtime is temporary storage, and
files written to it look completely normal right up until the runtime resets, at which
point a morning's annotation is gone and nobody else ever saw it. So if a notebook is
running in Colab from anywhere other than that folder, it **stops** rather than write
work you are going to lose.

Set it up once, at the start of the week.

**One member of the group:**

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/egumasa/lda2-final-template.git /content/drive/MyDrive/lda2-final-template
```

Then, in Drive: right-click the new `lda2-final-template` folder ▸ **Share** ▸ add the
rest of your group as **Editors**.

**Everyone else:** open Drive ▸ *Shared with me* ▸ right-click the folder ▸ **Add
shortcut to Drive** ▸ *My Drive*.

> Keep the shortcut's name exactly `lda2-final-template`. That is what makes
> `/content/drive/MyDrive/lda2-final-template/` mean the same thing for all of you. If
> Drive names it `lda2-final-template (1)`, rename it — otherwise the Setup cell will
> not find it, and you will end up working on separate copies without noticing.

After that, open notebooks from the folder itself — *File ▸ Open notebook ▸ Drive ▸
`lda2-final-template/notebooks/`* — rather than from the GitHub badge, which always
gives you a fresh copy rather than your group's.

**Nobody pushes.** git is just how you get the code; everything you produce stays in
your Drive folder and is handed in via `make_submission.py`.

### Working as a group

Colab syncs notebook edits live, like a Google Doc, so you can all work in a notebook at
once. Two things do not work that way:

- **Runtimes are per-person.** Seeing `gold` in a saved output does not mean `gold` exists
  in *your* session. Whoever runs the cells is the **driver**.
- **Files in the folder are last-write-wins.** `prompts/`, `outputs/` and `data/` are
  ordinary files, not Google Docs. Two of you writing the same one does not merge them —
  Drive keeps one and may leave the other beside it as `… (1).json`, which nothing
  downstream will ever read. Let the driver be the only one running cells that write.

The **annotation Sheet in notebook 03 is the exception** — a real Google Sheet, so
annotate it together. Notebook 03 shares it with the addresses you put in `MEMBERS` in
`config.yaml`; leave that list empty and the sheet sits in one person's Drive where the
second coder cannot open it. And your final run has to be *one* run by one person anyway, frozen
to a file.

### Your API key

Put it in the Colab **Secrets** panel (the 🔑 icon) as `GEMINI_API_KEY`. The Setup cell
should print:

```
LLM backend: Gemini API (gemini-3.1-flash-lite, temperature=0, seed=42)
```

If it says *Colab Gemini* instead, no key was found. That backend works, but it has no
temperature or seed, so your numbers will not be reproducible — fine for a quick look,
not for your final run.

Free tier: ~500 requests/day per key, ~15/minute. The backend paces itself and retries, so
a run takes minutes and may print `(rate limited - waiting Ns then retrying)` — that is
normal. **Iterate on your dev half**, which is a dozen or so items and about a minute per
round; `n_per_class` stays at its final value throughout, and the one run on the held-out
half happens at the end. If you run out of quota, hand the driver role to another member;
the files stay put in Drive.

### About the size of this study

One API call per item, four-and-a-bit seconds apart, no batching. Forty items is a few
minutes; four hundred is most of an afternoon of a quota shared across the course. So this
template cannot produce a study that would support a claim about a corpus — that needs
hundreds of items per class, and the confidence interval on a macro-F1 over twenty-odd
held-out items is wide enough that a five-point difference means nothing.

That is stated rather than worked around, and groups are asked to state it in the limitations section of the report
too. What the project is really rehearsing is the **method**: an annotation scheme you can
defend, a gold set two people built and argued over, a line drawn between the items you
tune on and the items you report on, a frozen run, and an audit trail. Those transfer to a
study of any size. The number does not.

## The tracks

| Track | Task | Labels | Difficulty |
|---|---|---|---|
| `raamove` | RA-abstract rhetorical moves | 8 moves | ★★☆ |
| `cars50` | CARS moves in RA intros (Kim & Lu replication) | Move 1–3, or 11 steps | ★★★ the annotators themselves got κ ≈ 0.43 |
| `l2_errors` | L2 error type, or error detection | 4 classes, or yes/no | ★★★ can also benchmark against the published tool |
| `icnale` | Holistic essay score band | Low/Mid/High | ★★☆ needs a manual, registered download |

No data ships with the template — your first step is building a pool in
`01_build_pool_<track>.ipynb`. If you want to watch 02–05 run before that, build the
small stand-ins with `python scripts/prep_datasets.py <track> --demos`; they are
*smaller than the sample you are meant to draw from a pool*, so they are for seeing the
pipeline work, not for running your study.

If you only want the file and not the reasoning — rebuilding a pool you have already
thought about, say — there is a shortcut that makes every decision for you:

```bash
python scripts/prep_datasets.py raamove     # or cars50 · l2_errors · icnale
```

See [`data/pools/README.md`](data/pools/README.md) and
[`data/SOURCES.md`](data/SOURCES.md) for licences and provenance.

## Run it locally (optional)

```bash
git clone https://github.com/egumasa/lda2-final-template.git && cd lda2-final-template
cp env.example .env          # then put your Gemini key in .env
uv sync
uv run jupyter lab notebooks/
```

Get a free key at <https://aistudio.google.com/apikey>. Note that notebook 03's Sheets
round-trip is designed for Colab, where your Google account is already available.

## Deliverables

- **Presentation + Q&A** — the main deliverable, one per group.
- **Two-page report** — written **individually**, in Word, from the numbers notebook 06
  prints on screen. Six sections: intro, methodology, prompt iterations, evaluation,
  error analysis, limitations. Each member uploads their own to Google Classroom; it is
  not part of the group bundle below.
- **Completed notebooks**, run in order, each one top to bottom.

Collect the group's work with:

```bash
python scripts/make_submission.py --group groupA
```

That builds `../lda2_project_groupA/` keeping the folder structure intact, and
deliberately leaves out your `.env`, the big pools, and anything ICNALE-derived. Download
it from Drive as a zip and turn it in on Google Classroom.

Run it again after filling a gap and it will stop, because rebuilding the folder deletes
everything in it — including your slides, if you dropped them in by hand. Add
`--overwrite` when that is what you want.

## For maintainers

All eight notebooks are **generated** — never hand-edit an `.ipynb`:

```bash
python scripts/_generate_pool_notebooks.py      # 01_build_pool_<track>.ipynb ×4
python scripts/_generate_project_notebooks.py   # 02_sample … 06_report
python scripts/_check_call_forms.py     # the contract test — run after ANY signature change
python scripts/_check_notebooks.py      # every cell runs, fits a screen, and is introduced
```

`_check_notebooks.py` asserts the three things that make a notebook followable: every code
cell is valid Python, no cell is longer than a screen, and every cell has a markdown
lead-in above it saying what is about to happen.

`_check_call_forms.py` asserts that every call form taught in Days 1–3 still runs here
unchanged. It needs no API key and no network. If it fails, fix the signature rather than
the tutorials.

The reshaping logic lives once, in `reshape.py`; `_generate_pool_notebooks.py` embeds its
source into the 01 notebooks with `inspect.getsource`, so the two cannot drift. The
judgment-call blanks work by *omitting* the constants those functions read
(`RAAMOVE_LABELS`, `L2_COARSE`) — Python resolves them from the notebook's globals, so a
filled-in notebook runs exactly the code `prep_datasets.py` runs. To check for drift:

```bash
python scripts/_generate_pool_notebooks.py && git diff --exit-code notebooks/
```

The course repo keeps its own older copy of the dataset-prep code, so its already-
published pages continue to work. **This repo's copy is the authoritative one** for the
mini-project.
