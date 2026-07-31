# scripts/

The code the notebooks call. You do not edit anything in here — but you are meant to
**read** some of it, and the notebooks tell you when.

The split is deliberate:

- **Plumbing is imported.** Downloading a corpus, retrying a rate-limited request,
  refusing to overwrite yesterday's gold file — none of that is what you are learning,
  and reading it would not tell you anything about annotation.
- **The study is not.** Drawing a balanced sample, asking the model and reading its
  reply, scoring the result: those are the methods you have to describe in your report
  and defend in the Q&A. So notebooks 02, 04 and 05 put that code **in the notebook**,
  read straight out of the files below when the notebook is generated. It is the same
  code, not a simplified copy — a method you cannot read is one you cannot defend.

| File | What is in it | When you would open it |
|---|---|---|
| `pipeline.py` | Loading and saving, the connection to the model, `sample_pool`, `run_prompt`, `extract_label`, `build_fewshot`, `export_results` | To see how the sample is drawn, or what happens to the model's reply between "it answered" and "that counts as `Move 2`" |
| `metrics.py` | `evaluate` (per-class P/R/F1, κ, confusion matrix), `agreement`, `show_errors` | To see exactly what your headline numbers are computed from |
| `annotate.py` | The Google Sheets round trip: a tab per coder, `Final` to adjudicate in, and the agreement statistics | When the sheet does something surprising — who it was shared with, how blank rows are treated, how labels are matched back — or to see how Fleiss' κ is computed for three or more coders |
| `reshape.py` | Each corpus → the canonical `{id, text, label}` | Notebook 01 already shows you the part that applies to your track |
| `download.py` | One fetch per track | Rarely. ICNALE is deliberately manual — the licence says ask first |
| `prep_datasets.py` | `python scripts/prep_datasets.py <track>` — builds a pool with every decision already made | When you want the file and not the reasoning: rebuilding a pool you have already thought about, or `--demos` for the small stand-ins |
| `make_submission.py` | Collects what you hand in | At the end, once |
| `_generate_pool_notebooks.py`, `_generate_project_notebooks.py`, `_setup_cell.py` | Build the eight notebooks | Only if you are changing the template itself. Never hand-edit an `.ipynb` |
| `_check_call_forms.py` | Asserts every call form taught on Days 1–3 still runs here unchanged | After changing any signature. Needs no API key and no network |

If a call you learned in the tutorials does not work in these notebooks, that is a bug
here rather than a gap in your memory of it — please say so.
