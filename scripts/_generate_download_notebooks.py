#!/usr/bin/env python3
"""Generate notebooks/download_<track>.ipynb — one per track.

    python scripts/_generate_download_notebooks.py

These notebooks do the same job as `prep_datasets.py`, but one step at a time and with
the reasoning spelled out: download -> look at the raw format -> reshape to
{id, text, label} -> check the label balance -> save. They are standalone (stdlib only,
no repo needed), so they run in a fresh Colab.

WHY THIS IS GENERATED
---------------------
Each notebook needs the reshaping code *inside it*, because it has to run without the
rest of the repo. The obvious way to do that is to paste the code into each notebook -
which is what the course repo's version does, and its README admits the five copies
have to be kept in sync by hand.

So instead we read the real functions out of reshape.py at generation time with
`inspect.getsource`. There is one implementation; these notebooks are renderings of it.
Never hand-edit a download_*.ipynb - edit reshape.py and re-run this script.

To check nothing has drifted:

    python scripts/_generate_download_notebooks.py && git diff --exit-code notebooks/
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reshape

OUT = Path(__file__).resolve().parent.parent / "notebooks"


# ----------------------------------------------------------------------------------
# Notebook building blocks
# ----------------------------------------------------------------------------------
def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def save(name, cells):
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
    print("wrote", name, "(" + str(len(cells)), "cells)")


def embed(*objects):
    """Render live functions/constants from reshape.py as notebook source text.

    Functions come through `inspect.getsource`, so what the notebook shows is literally
    what prep_datasets.py runs. Constants are rendered from their real values.
    """
    blocks = []
    for obj in objects:
        if isinstance(obj, tuple):          # ("NAME", value) -> a constant
            name, value = obj
            blocks.append(name + " = " + repr(value))
        else:
            blocks.append(inspect.getsource(obj).rstrip("\n"))
    return "\n\n".join(blocks)


SCHEMA_NOTE = (
    "Every dataset in this course is reshaped into the **same canonical schema**, so one "
    "pipeline works for all of them:\n\n"
    "```json\n"
    "[{\"id\": 1, \"text\": \"...\", \"label\": \"...\"}]\n"
    "```\n\n"
    "The *raw* data, though, looks different every time. **That difference is the "
    "lesson** — half of building a gold standard is getting messy real data into a "
    "clean, consistent shape."
)

GENERATED_NOTE = (
    "> This notebook is **generated** from `scripts/reshape.py`. The reshaping code "
    "below is the same code `scripts/prep_datasets.py` runs — not a copy of it. If you "
    "want to change how the data is reshaped, edit `reshape.py` and re-run "
    "`scripts/_generate_download_notebooks.py`."
)


def header(title, subtitle, what, licence_line, citation, difficulty):
    return md(
        "# " + title,
        "",
        "*" + subtitle + "*",
        "",
        "**What it is.** " + what,
        "",
        "**Difficulty of the labeling judgment:** " + difficulty,
        "",
        "**Licence:** " + licence_line + "  ",
        "**Cite:** " + citation,
        "",
        "---",
        "",
        SCHEMA_NOTE,
        "",
        GENERATED_NOTE,
    )


def inspect_cell():
    return code(
        "from collections import Counter",
        "",
        'print("total items:", len(rows))',
        'print("label counts:", dict(Counter(item["label"] for item in rows)))',
        "rows[:3]        # peek at the first three reshaped items")


def save_cell(track, note=""):
    lines = [
        "# Save the pool. Two places you might want it:",
        '#   * this repo, if you cloned it:  "../data/pools/' + track + '_pool.json"',
        "#   * your Google Drive, so it survives the Colab runtime resetting",
        "import json",
        "",
        'OUT_FILE = "' + track + '_pool.json"',
        "",
        "# In Colab, uncomment these two lines to write straight to your Drive:",
        '# from google.colab import drive; drive.mount("/content/drive")',
        '# OUT_FILE = "/content/drive/MyDrive/' + track + '_pool.json"',
        "",
        'with open(OUT_FILE, "w", encoding="utf-8") as f:',
        "    json.dump(rows, f, ensure_ascii=False, indent=2)",
        'print("Saved", len(rows), "items to", OUT_FILE)',
    ]
    if note:
        lines = lines + ["", "# " + note]
    return code(*lines)


def balance_note(track):
    return md(
        "## A note on what you just built",
        "",
        "This is the **pool** — everything usable in the corpus, with its natural label "
        "imbalance intact. It is *not* your gold set.",
        "",
        "Your gold set comes next, in the project notebook: `sample_pool` draws a "
        "*balanced* subset from this pool (equal items per label), which is what makes "
        "precision, recall, F1 and the confusion matrix readable. Keeping the two "
        "separate also leaves the unsampled items free to serve as few-shot examples "
        "without leaking the answers you are testing on.",
        "",
        "So: build the pool once, here. Sample from it there.")


# ===================================================================== CEFR-SP
save("download_cefr.ipynb", [
    header(
        "CEFR-SP — download & preprocess",
        "The on-ramp track: sentence proficiency level (A1–C2)",
        "Sentences annotated with a CEFR level by two trained annotators. We use the "
        "openly-shipped **Wiki-Auto** portion.",
        "CC BY-SA 3.0 (Wiki-Auto portion) — **share-alike**, so anything you "
        "redistribute from it inherits the same licence.",
        "Arase, Uchida & Kajiwara (2022), *EMNLP*. github.com/yukiar/CEFR-SP",
        "★☆☆ — easy. Levels are concrete and the annotators usually agree.",
    ),
    md("## Step 1 — Download the raw data",
       "",
       "The corpus lives in a GitHub repository, so we clone it. (`!` runs a shell "
       "command from inside the notebook.)"),
    code("!git clone --depth 1 https://github.com/yukiar/CEFR-SP"),
    md("## Step 2 — Look at the raw format",
       "",
       "Note the folder path: cloning `CEFR-SP` gives you a `CEFR-SP` folder *inside* "
       "`CEFR-SP`. Easy to trip over.",
       "",
       "The Wiki-Auto files are **tab-separated text**, one sentence per line:",
       "",
       "```",
       "sentence <TAB> label_by_annotator_A <TAB> label_by_annotator_B",
       "```",
       "",
       "Labels are digits: `1`=A1, `2`=A2, … `6`=C2."),
    code('RAW_DIR = "CEFR-SP/CEFR-SP/Wiki-Auto"',
         "",
         'with open(RAW_DIR + "/CEFR-SP_Wikiauto_dev.txt", encoding="utf-8") as f:',
         "    for _ in range(5):",
         "        print(repr(next(f)))"),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Three real decisions happen in the code below, and they are commented where "
       "they happen:",
       "",
       "1. **Trust only agreement** — keep a sentence only when *both* annotators chose "
       "the same level. Every label is then unambiguous, which is what makes this the "
       "gentle track. It also means the track is easier than the data really is.",
       "2. **Human-readable labels** — `1` → `A1`, so your prompt can name the levels "
       "the way a person would.",
       "3. **Wiki-Auto only** — the repo also ships a `SCoRE/` folder under a "
       "*non-commercial* licence. We deliberately never read it. Notice that the code "
       "is pointed at the `Wiki-Auto` folder specifically, rather than at the repo root: "
       "that is what keeps the two apart."),
    code(embed(reshape.reid, ("CEFR_NUM", reshape.CEFR_NUM), reshape.reshape_cefr)),
    code("rows = reshape_cefr(RAW_DIR)",
         'print("kept", len(rows), "sentences where both annotators agreed")'),
    md("## Step 4 — Check the label balance",
       "",
       "Look at the counts before you trust anything downstream. This corpus is "
       "**heavily imbalanced** — B1 and B2 dominate, and A1/C2 are scarce. That is a "
       "fact about the data, and it is why the project samples a balanced subset rather "
       "than using the pool directly."),
    inspect_cell(),
    balance_note("cefr"),
    md("## Step 5 — Save it"),
    save_cell("cefr"),
])

# ===================================================================== RAAMove
save("download_raamove.ipynb", [
    header(
        "RAAMove — download & preprocess",
        "Rhetorical moves in research-article abstracts (8 classes)",
        "400 RA abstracts, annotated sentence by sentence with one of eight rhetorical "
        "moves (Background, Gap, Purpose, Method, Result, Conclusion, Contribution, "
        "Implication). Reported annotator agreement: κ = 0.785.",
        "CC BY 4.0",
        "Liu, J. et al. (2024), *LREC-COLING*. github.com/ljk1228/RAAMove",
        "★★☆ — moderate. Moves are functional categories, so neighbouring sentences "
        "can be genuinely hard to separate.",
    ),
    md("## Step 1 — Download the raw data"),
    code("!git clone --depth 1 https://github.com/ljk1228/RAAMove"),
    md("## Step 2 — Look at the raw format",
       "",
       "This one is **JSON**, split into two files by discipline "
       "(`Intelligence.json`, `Engineering.json`). Each record has a `text` and a "
       "three-letter move code in `labels`."),
    code('RAW_DIR = "RAAMove"',
         "",
         "import json",
         'data = json.loads(open(RAW_DIR + "/Intelligence.json", encoding="utf-8").read())',
         'print("records:", len(data))',
         "data[:3]"),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Two decisions:",
       "",
       "1. **Expand the codes** — `BAC` → `Background`. A prompt that says "
       "\"Background\" needs far less explaining than one that says \"BAC\".",
       "2. **Pool the two disciplines** — we treat a move as a rhetorical function "
       "rather than a discipline-specific one. That *is* an assumption. Comparing "
       "Intelligence against Engineering separately would be a good extension."),
    code(embed(reshape.reid,
               ("RAAMOVE_LABELS", reshape.RAAMOVE_LABELS),
               reshape.reshape_raamove)),
    code("rows = reshape_raamove(RAW_DIR)"),
    md("## Step 4 — Check the label balance",
       "",
       "Very imbalanced: `Method` is the biggest class by far, and `Implication` has "
       "only a couple of dozen sentences. With eight classes and a rare tail, a "
       "balanced sample of 7 per class is about the most this pool will support."),
    inspect_cell(),
    balance_note("raamove"),
    md("## Step 5 — Save it"),
    save_cell("raamove"),
])

# ===================================================================== CaRS-50
save("download_cars50.ipynb", [
    header(
        "CaRS-50 — download & preprocess",
        "Swales CARS moves in research-article introductions (3 or 11 classes)",
        "50 BioRxiv article introductions, annotated sentence by sentence with Swales' "
        "CARS Move and Step scheme. **The annotators themselves reached only κ ≈ 0.43** "
        "— so on this track, \"the model is wrong\" and \"the scheme is fuzzy\" are both "
        "live explanations, and telling them apart is the interesting part.",
        "CC BY 4.0",
        "Lam, C. & Nnamoko, N. (2025). *Mendeley Data*, V1. doi:10.17632/kwr9s5c4nk.1",
        "★★★ — hard. Judging moves in an introduction needs more context than a single "
        "sentence gives you.",
    ),
    md("## Step 1 — Download the raw data",
       "",
       "This one is on **Mendeley Data**, which has a public API. We ask it for the "
       "dataset's file list, then download each file. The CDN refuses requests that do "
       "not look like a browser, hence the `User-Agent` header."),
    code("import json, urllib.request, pathlib",
         "",
         'RAW_DIR = pathlib.Path("cars50")',
         "RAW_DIR.mkdir(exist_ok=True)",
         "",
         "def fetch(url):",
         '    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})',
         "    return urllib.request.urlopen(request, timeout=60)",
         "",
         'meta = json.loads(fetch("https://data.mendeley.com/public-api/datasets/kwr9s5c4nk").read())',
         'for record in meta["files"]:',
         '    target = RAW_DIR / record["filename"]',
         "    if not target.exists():",
         '        target.write_bytes(fetch(record["content_details"]["download_url"]).read())',
         'print("downloaded", len(list(RAW_DIR.glob("*.xml"))), "XML files")'),
    md("## Step 2 — Look at the raw format",
       "",
       "**XML** this time. Each sentence carries a `step` code like `1b`:",
       "",
       "```xml",
       "<sentence><sentenceID/><text/><step>1b</step></sentence>",
       "```"),
    code('print(open(sorted(RAW_DIR.glob("*.xml"))[0], encoding="utf-8").read()[:900])'),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "The interesting decision here is **granularity**, and one parse gives you both:",
       "",
       "- the leading digit of `1b` is the **Move** → 3 classes, a fair task;",
       "- the whole code `1b` is the **Step** → 11 classes, the stretch version.",
       "",
       "Which one you pick is a scheme decision, and it belongs in your `PLAN.md`. "
       "Sentences with no code, or a code that does not start with a move digit, are "
       "dropped."),
    code(embed(reshape.reid, reshape.reshape_cars50)),
    code("move_rows, step_rows = reshape_cars50(RAW_DIR)",
         "",
         "rows = move_rows          # 3 classes. Swap in step_rows for the 11-class version.",
         'print("moves:", len(move_rows), " steps:", len(step_rows))'),
    md("## Step 4 — Check the label balance"),
    inspect_cell(),
    balance_note("cars50"),
    md("## Step 5 — Save it"),
    save_cell("cars50",
              "For the 11-class version, set rows = step_rows above and save as "
              "cars50_step_pool.json."),
])

# ===================================================================== L2 errors
save("download_l2_errors.ipynb", [
    header(
        "AutoErrorAnalyzer — download & preprocess",
        "Error type in a learner sentence (4 classes), or error detection (2 classes)",
        "~100 Japanese-EFL essays, annotated with a 26-category error taxonomy "
        "(Krippendorff's α ≈ .92). The file also holds **the published tool's own "
        "predictions**, so this is the one track where you can benchmark your LLM "
        "against both a human gold standard *and* an existing system.",
        "CC BY 4.0",
        "Mizumoto, A. (2025). *Studies in Second Language Acquisition, 47*(3), 867–884. "
        "OSF: osf.io/jyf3r",
        "★★★ — hard. Many error types, and a sentence can carry several at once.",
    ),
    md("## Step 1 — Download the raw data",
       "",
       "The annotations live on the paper's **OSF** project. We fetch one CSV directly "
       "by its OSF link."),
    code("import urllib.request",
         "",
         'RAW_FILE = "data_category.csv"',
         'urllib.request.urlretrieve("https://osf.io/download/gezat/", RAW_FILE)',
         'print("downloaded", RAW_FILE)'),
    md("## Step 2 — Look at the raw format",
       "",
       "A **CSV**. The columns that matter are `Sentence`, `Human_ErrorCategories` (the "
       "gold) and `AEA_ErrorCategories` (the tool's prediction). A sentence can carry "
       "several comma-separated error codes, or `NO_ERROR`."),
    code("import csv",
         "",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)',
         "    for row, _ in zip(reader, range(3)):",
         '        print(row["Sentence"][:70], "|", row["Human_ErrorCategories"])'),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "Two decisions, and the second one has a consequence worth writing down:",
       "",
       "1. **Collapse 23 codes into 4 categories** — Grammatical / Lexical / Mechanical "
       "/ No error. The full taxonomy is too fine-grained to prompt for reliably at this "
       "scale.",
       "2. **Drop mixed-category sentences** — a sentence whose codes span more than one "
       "broader category gets no label, so the task stays single-label. That means the "
       "dataset under-represents exactly the messiest sentences, which **belongs in your "
       "limitations section**.",
       "",
       "You also get a binary detection version for free (any error at all: yes/no)."),
    code(embed(reshape.reid,
               ("L2_COARSE", reshape.L2_COARSE),
               reshape._l2_coarse_label,
               reshape.reshape_l2_errors)),
    code("category_rows, detection_rows = reshape_l2_errors(RAW_FILE)",
         "",
         "rows = category_rows      # 4 classes. Swap in detection_rows for yes/no.",
         'print("categories:", len(category_rows), " detection:", len(detection_rows))'),
    md("## Step 4 — Check the label balance",
       "",
       "Note how many sentences were dropped: compare the category count against the "
       "detection count, which keeps everything. The gap is the mixed-category "
       "sentences."),
    inspect_cell(),
    balance_note("l2_errors"),
    md("## Step 5 — Save it"),
    save_cell("l2_errors",
              "For the binary version, set rows = detection_rows above and save as "
              "l2_error_detection_pool.json."),
])

# ===================================================================== ICNALE GRA
save("download_icnale.ipynb", [
    header(
        "ICNALE GRA — download & preprocess",
        "Holistic essay score band (Low / Mid / High)",
        "Asian-learner L2 English essays, each rated on holistic and analytic scales by "
        "many trained raters. This is an **automated writing evaluation** task: whole "
        "essays, not sentences.",
        "⚠️ **Research use only — NOT redistributable.** Requires registration. Nothing "
        "derived from it may be committed to git or included in your submission bundle.",
        "Ishikawa, S. *The ICNALE Global Rating Archives.*",
        "★★☆ — moderate, but a different shape of task: long texts and an ordered scale.",
    ),
    md("## Step 1 — Get the data (this one is manual)",
       "",
       "ICNALE GRA is released for research use behind a registration form that emails "
       "you a password. There is nothing to automate, and that is deliberate — the "
       "licence does not permit redistribution.",
       "",
       "1. Register at "
       "<https://language.sakura.ne.jp/icnale/download.html> and wait for the password.",
       "2. Download and unpack `ICNALE_GRA_2.x.zip`.",
       "3. From its rating tables, export a CSV with **exactly two columns**, `text` and "
       "`score`, and upload it here (or put it in `data/raw/icnale/essays_scores.csv`).",
       "",
       "In Colab, the cell below opens a file picker."),
    code("# In Colab: uncomment to upload your essays_scores.csv",
         "# from google.colab import files; files.upload()",
         "",
         'RAW_FILE = "essays_scores.csv"'),
    md("## Step 2 — Look at the raw format"),
    code("import csv",
         "",
         'with open(RAW_FILE, encoding="utf-8-sig", newline="") as f:',
         "    reader = csv.DictReader(f)",
         '    print("columns:", reader.fieldnames)',
         "    for row, _ in zip(reader, range(2)):",
         '        print(round(float(row["score"]), 2), "|", row["text"][:100], "...")'),
    md("## Step 3 — Reshape into the canonical schema",
       "",
       "One decision, and it is entirely yours: **where do the band boundaries go?**",
       "",
       "The defaults below (`< 4` = Low, `< 7` = Mid, else High) are **placeholders** — "
       "round numbers, not the ICNALE rubric. Where you cut decides how hard the task is "
       "and how balanced the classes are, so set them from the rubric you are actually "
       "using and **say what you chose in your report**.",
       "",
       "⚠️ These labels are **ordered** (Low < Mid < High) but they are *not* "
       "alphabetical. So pass the order explicitly when you evaluate — "
       "`LABELS_ORDER = [\"Low\", \"Mid\", \"High\"]` — or the weighted κ will be "
       "computed over `High < Low < Mid`, which means nothing."),
    code(embed(reshape.reid, reshape.reshape_icnale)),
    code("rows = reshape_icnale(RAW_FILE)      # try low_below=..., mid_below=... too"),
    md("## Step 4 — Check the label balance"),
    inspect_cell(),
    balance_note("icnale"),
    md("## Step 5 — Save it",
       "",
       "⚠️ Keep this file **out of git** and **out of your submission bundle**. "
       "`.gitignore` and `scripts/make_submission.py` both exclude anything with "
       "`icnale` in the name — please leave that in place."),
    save_cell("icnale"),
])

print("\nDone. To check nothing drifted from reshape.py:")
print("  python scripts/_generate_download_notebooks.py && git diff --exit-code notebooks/")
