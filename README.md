# LDA II — Final mini-project template

The scaffold for the final group mini-project in **Linguistic Data Analysis II**. You run
a small LLM-annotation study end to end, across five numbered notebooks:

```
01 build pool  →  02 sample  →  03 annotate  →  04 prompt  →  05 report
```

> **build** a pool from a real corpus → **sample** a balanced subset → **annotate and
> adjudicate** it yourselves → write a **baseline prompt** → **iterate** it over 2–3
> rounds (P/R/F1 + confusion matrix each round) → **freeze** the predictions →
> **error analysis** → **export a one-page report**

The plumbing is written for you. What you supply is the judgment: which items, which
labels, which prompt, and what the errors mean.

Note where the model first appears — **notebook 04**. Notebooks 01–03 are the study; the
LLM is the thing being measured by it.

## What's in here

```
lda2-final-template/
├── config.py                       # ✏️ the ONE file you edit first: track, group, seed
├── notebooks/                      # YOUR five notebooks. Run them in order.
│   ├── 01_build_pool_<track>.ipynb #   corpus → data/pools/<track>_pool.json
│   ├── 02_sample.ipynb             #   pool → a balanced sample
│   ├── 03_annotate.ipynb           #   sample → YOUR gold standard
│   ├── 04_prompt.ipynb             #   gold → a frozen set of predictions
│   └── 05_report.ipynb             #   predictions → error analysis + report
├── scripts/                        # the plumbing. You do NOT edit these.
│   ├── pipeline.py                 #   load, sample, prompt, freeze, export
│   ├── metrics.py                  #   precision/recall/F1, kappa, confusion matrix
│   ├── annotate.py                 #   the Google Sheets annotation round-trip
│   ├── reshape.py · download.py    #   turning each raw corpus into {id, text, label}
│   ├── prep_datasets.py            #   one command to rebuild a track's pool
│   └── make_submission.py          #   collect what you hand in
├── data/
│   ├── pools/                      # the pools you build (git-ignored), plus small
│   │                               #   <track>_demo_pool.json files so 02–05 run on clone
│   ├── gold/                       # YOUR sample and YOUR gold set (git-ignored)
│   └── raw/                        # original downloads (git-ignored)
├── prompts/<track>.txt             # your prompt lives here, as a file you edit
└── outputs/                        # predictions, report, figures
```

You edit **`config.py`**, the **✏️ cells** in the notebooks, and your **prompt file**.
Nothing in `scripts/`.

## The notebooks are skeletons, on purpose

Each notebook ships with all of its prose, its plumbing, and — for every blank — a header
saying what it is for, what is available to write it with, and **what to call the thing
you produce**, because the next cell expects that name. What is missing is the calls.

Two kinds of blank, and they are asking for different things:

- **In 01**, the ✏️ cells are the *reshaping decisions*: what each label is called, which
  annotations to trust, how fine-grained the scheme is, where a numeric scale gets cut.
  The download and the parsing are written for you; nobody learns anything from retyping
  a `User-Agent` header. `PLAN.md` asks you to justify what you chose.
- **In 02–05**, the blanks are the *pipeline itself*: what each step consumes and
  produces. Every call has the **same form it had in Days 1–3**, and each blank names
  where you used it before. `run_prompt(PROMPT, gold)` and
  `evaluate(gold, pred, ordered=True)` are the same two lines you ran on Day 3. If
  something you typed in the tutorials does not work here, that is a bug in this
  template, not in your memory of it — please say so.

### Each notebook hands a file to the next

02 loads what 01 wrote; 03 loads what 02 wrote, and so on. That is not ceremony. Your
group works across several days and several people's Colab runtimes, and a variable in
someone else's session is not a handoff. It also means a group that stalls in 03 can be
handed a gold set and carry on in 04, and that "does it run top to bottom on a fresh
runtime?" is a question you can actually check.

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
`config.py`; leave that list empty and the sheet sits in one person's Drive where the
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
a full run takes minutes and may print `(rate limited - waiting Ns then retrying)` — that
is normal. **Iterate at `N_PER_CLASS = 2`**, then do one final run at full size. If you
run out, hand the driver role to another member; the files stay put in Drive.

## The tracks

| Track | Task | Labels | Difficulty |
|---|---|---|---|
| `raamove` | RA-abstract rhetorical moves | 8 moves | ★★☆ |
| `cars50` | CARS moves in RA intros (Kim & Lu replication) | Move 1–3, or 11 steps | ★★★ the annotators themselves got κ ≈ 0.43 |
| `l2_errors` | L2 error type, or error detection | 4 classes, or yes/no | ★★★ can also benchmark against the published tool |
| `icnale` | Holistic essay score band | Low/Mid/High | ★★☆ needs a manual, registered download |

Notebooks 02–05 run immediately on the shipped `<track>_demo_pool.json` files — but the
demos are *smaller than the sample you are meant to draw from a pool*, so your real study
starts by building the real thing in `01_build_pool_<track>.ipynb`.

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

- **Presentation + Q&A** — the main deliverable.
- **One-page report** — `export_results` writes a scaffold with the five required
  sections; you fill in the QC narrative, the error attributions, and the limitations.
- **Completed notebooks**, run in order, each one top to bottom.

Collect it all with:

```bash
python scripts/make_submission.py --group groupA
```

That builds `../lda2_project_groupA/` keeping the folder structure intact, and
deliberately leaves out your `.env`, the big pools, and anything ICNALE-derived. Download
it from Drive as a zip and turn it in on Google Classroom.

## For maintainers

All eight notebooks are **generated** — never hand-edit an `.ipynb`:

```bash
python scripts/_generate_pool_notebooks.py      # 01_build_pool_<track>.ipynb ×4
python scripts/_generate_project_notebooks.py   # 02_sample … 05_report
python scripts/_check_call_forms.py     # the contract test — run after ANY signature change
```

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
