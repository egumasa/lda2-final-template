# LDA II — Final mini-project template

The scaffold for the final group mini-project in **Linguistic Data Analysis II**. You run
a small LLM-annotation study end to end:

> **sample** a balanced gold subset → **QC / adjudicate** it yourselves → write a
> **baseline prompt** → **iterate** it over 2–3 rounds (P/R/F1 + confusion matrix each
> round) → **freeze** the predictions → **error analysis** → **export a one-page report**

The plumbing is written for you. What you supply is the judgment: which items, which
labels, which prompt, and what the errors mean.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/egumasa/lda2-final-template/blob/main/notebooks/mini_project.ipynb)

## What's in here

```
lda2-final-template/
├── notebooks/
│   ├── mini_project.ipynb          # YOUR notebook — one skeleton, any track
│   └── download_<track>.ipynb      # build a track's data, step by step (Colab-ready)
├── scripts/                        # the plumbing. You do NOT edit these.
│   ├── pipeline.py                 #   load, sample, prompt, freeze, export
│   ├── metrics.py                  #   precision/recall/F1, kappa, confusion matrix
│   ├── annotate.py                 #   the Google Sheets annotation round-trip
│   ├── reshape.py · download.py    #   turning each raw corpus into {id, text, label}
│   ├── prep_datasets.py            #   one command to build a track's pool
│   └── make_submission.py          #   collect what you hand in
├── data/
│   ├── gold/<track>_demo.json      # small demo sets, so every track runs on clone
│   ├── pools/                      # the full pools you build (git-ignored)
│   └── raw/                        # original downloads (git-ignored)
├── prompts/<track>.txt             # your prompt lives here, as a file you edit
└── outputs/                        # predictions, report, figures
```

You edit the notebook's **CONFIG cell**, its **six step cells**, and your **prompt file**.
Nothing in `scripts/`.

## The notebook is a skeleton, on purpose

Each of the six steps ships with its header, its goal, the helpers available to it, and
the variable names the next step expects — but not the calls. You write those.

Every call has the **same form it had in Days 1–3**, and each step names where you used it
before. `run_prompt(PROMPT, gold)` and `evaluate(gold, pred, ordered=True)` are the same
two lines you ran on Day 3. If something you typed in the tutorials does not work here,
that is a bug in this template, not in your memory of it — please say so.

## Run it in Google Colab (recommended)

Open `notebooks/mini_project.ipynb` in Colab (badge above, or *File ▸ Open notebook ▸
GitHub*), then run the Setup cell — Colab starts with only that one file, so the cell
clones everything *around* it.

**Do this once, as a group:** clone into Google Drive, then always open the copy that
lives there (*File ▸ Open ▸ Drive ▸ `lda2-final-template/notebooks/mini_project.ipynb`*).
Your prompts, gold set and outputs then survive the runtime resetting.

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
![ -d lda2-final-template ] || git clone https://github.com/egumasa/lda2-final-template.git
%cd /content/drive/MyDrive/lda2-final-template/notebooks
```

One member clones and shares the folder with the group. **Nobody pushes** — git is just
how you get the code.

### Working as a group

Colab syncs notebook edits live, like a Google Doc, so you can all work in the notebook at
once. Two things do not work that way:

- **Runtimes are per-person.** Seeing `gold` in a saved output does not mean `gold` exists
  in *your* session. Whoever runs the cells is the **driver**.
- **Files in the repo folder are last-write-wins.** `prompts/`, `outputs/` and `data/` are
  ordinary files. Let the driver be the only one running cells that write them.

The **annotation Sheet in step 2 is the exception** — a real Google Sheet, so annotate it
together. And your final run has to be *one* run by one person anyway, frozen to a file.

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
| `cefr` | CEFR sentence level | A1–C2 | ★☆☆ the on-ramp |
| `raamove` | RA-abstract rhetorical moves | 8 moves | ★★☆ |
| `cars50` | CARS moves in RA intros (Kim & Lu replication) | Move 1–3, or 11 steps | ★★★ the annotators themselves got κ ≈ 0.43 |
| `l2_errors` | L2 error type, or error detection | 4 classes, or yes/no | ★★★ can also benchmark against the published tool |
| `icnale` | Holistic essay score band | Low/Mid/High | ★★☆ needs a manual, registered download |

Every track runs immediately on its shipped demo file — but the demos are *smaller than
the sample you are meant to draw from a pool*, so before your real run:

```bash
python scripts/prep_datasets.py cefr        # or raamove · cars50 · l2_errors · icnale
```

Then point `POOL_PATH` at `../data/pools/<track>_pool.json`. See
[`data/gold/README.md`](data/gold/README.md) and [`data/SOURCES.md`](data/SOURCES.md) for
licences and provenance.

## Run it locally (optional)

```bash
git clone https://github.com/egumasa/lda2-final-template.git && cd lda2-final-template
cp env.example .env          # then put your Gemini key in .env
uv sync
uv run jupyter lab notebooks/mini_project.ipynb
```

Get a free key at <https://aistudio.google.com/apikey>. Note that step 2's Sheets
round-trip is designed for Colab, where your Google account is already available.

## Deliverables

- **Presentation + Q&A** — the main deliverable.
- **One-page report** — `export_results` writes a scaffold with the five required
  sections; you fill in the QC narrative, the error attributions, and the limitations.
- **Completed notebook**, run top to bottom.

Collect it all with:

```bash
python scripts/make_submission.py --group groupA
```

That builds `../lda2_project_groupA/` keeping the folder structure intact, and
deliberately leaves out your `.env`, the big pools, and anything ICNALE-derived. Download
it from Drive as a zip and turn it in on Google Classroom.

## For maintainers

Both notebooks are **generated** — never hand-edit an `.ipynb`:

```bash
python scripts/_generate_mini_project_notebook.py
python scripts/_generate_download_notebooks.py
python scripts/_check_call_forms.py     # the contract test — run after ANY signature change
```

`_check_call_forms.py` asserts that every call form taught in Days 1–3 still runs here
unchanged. It needs no API key and no network. If it fails, fix the signature rather than
the tutorials.

The reshaping logic lives once, in `reshape.py`; `_generate_download_notebooks.py` embeds
its source into the notebooks with `inspect.getsource`, so the two cannot drift. To check:

```bash
python scripts/_generate_download_notebooks.py && git diff --exit-code notebooks/
```

The course repo keeps its own older copy of the dataset-prep code, so its already-
published pages continue to work. **This repo's copy is the authoritative one** for the
mini-project.
