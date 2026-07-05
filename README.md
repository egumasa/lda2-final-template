# LDA II — Final mini-project template

A ready-to-run scaffold for the final group mini-project in **Linguistic Data Analysis
II**. You run a small LLM-annotation study end to end:

> **sample** a balanced gold subset → **QC / adjudicate** it → write a **baseline prompt**
> → **iterate** it with few-shot rounds (precision / recall / F1 + confusion matrix each
> round) → **error analysis** → **export a one-page report.**

You do not write the plumbing. It is already here. You choose a track, drop in your data,
design a prompt, and read the numbers.

## What's in here

```
lda2-final-template/
├── scripts/
│   ├── pipeline.py     # plumbing: read data, sample, ask the LLM, few-shot, export (GIVEN)
│   └── evaluate.py     # scoring: precision/recall/F1, confusion matrix, kappa (GIVEN, uses sklearn)
├── notebooks/
│   ├── cefr.ipynb  raamove.ipynb  cars50.ipynb  l2_errors.ipynb   # one per track — pick one
├── data/gold/          # your pool of labeled items lives here (a CEFR sample ships with it)
├── prompts/            # your prompt lives here as a text file you edit
└── outputs/            # your predictions CSV + one-page report get written here
```

You edit **prompts**, the **CONFIG cell** of a notebook, and (optionally) the QC cell.
You do **not** edit `scripts/`.

## Run it in Google Colab (recommended)

The course assumes no local Python. Open your track's notebook in Colab and run the setup
cell — it offers two ways in:

**Option A — clone into your Google Drive (recommended).** Your sampled gold, edited
prompts, and outputs then **persist across sessions** (this is the same "keep your data in
Drive" workflow from the course).

```python
from google.colab import drive
drive.mount('/content/drive')
# one-time clone — safe to skip if the folder already exists:
%cd /content/drive/MyDrive
![ -d lda2-final-template ] || git clone https://github.com/egumasa/lda2-final-template.git lda2-final-template
%cd /content/drive/MyDrive/lda2-final-template/notebooks
import sys; sys.path.append("../scripts")
```

> After you edit a file in `scripts/`, restart the runtime (or `import importlib; importlib.reload(...)`)
> so the change is picked up.

**Option B — quick, ephemeral clone.** Simplest, but anything you change is lost when the
runtime resets.

```python
!git clone https://github.com/egumasa/lda2-final-template.git
%cd lda2-final-template/notebooks
import sys; sys.path.append("../scripts")
```

Either way, the notebook then does `from pipeline import *` and `from evaluate import *`.

## Run it locally (optional, for advanced users)

```bash
git clone https://github.com/egumasa/lda2-final-template.git && cd lda2-final-template
cp env.example .env        # then put your Gemini key in .env
uv sync                    # installs jupyterlab, pandas, seaborn, matplotlib, scikit-learn, google-genai, ...
uv run python -m ipykernel install --user --name lda2 --display-name "LDA2 (uv)"
uv run jupyter lab notebooks/cefr.ipynb   # then select the "LDA2 (uv)" kernel
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. In Colab you don't
need one — Colab has a built-in Gemini backend.

## The tracks

| Notebook | Task | Labels | Pool file |
|---|---|---|---|
| `cefr.ipynb` | CEFR sentence level (easy on-ramp) | A1–C2 | `cefr_pool.json` *(ships with template)* |
| `raamove.ipynb` | RA-abstract rhetorical moves | 8 moves | `raamove_pool.json` |
| `cars50.ipynb` | CARS moves (Kim & Lu replication) | Move 1/2/3 | `cars50_pool.json` |
| `l2_errors.ipynb` | L2 error type | Grammatical/Lexical/Mechanical/No error | `l2_errors_pool.json` |

The `cefr` notebook runs immediately on the shipped sample. For any other track, build the
pool file and drop it in `data/gold/` — see [`data/gold/README.md`](data/gold/README.md).

## Deliverables

- **Presentation + Q&A** (the main deliverable).
- **One-page report** — `export_results` writes a scaffold to `outputs/<track>_report.md`
  with the five required sections; you fill in the QC and error-analysis prose.
- **Completed notebook**, run end to end.
