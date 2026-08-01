# scripts/

The code the notebooks call. You do not edit anything in here — but you are meant to
**read** some of it, and the notebooks tell you when.

The split is deliberate:

- **Plumbing is imported.** Downloading a corpus, retrying a rate-limited request,
  refusing to overwrite yesterday's gold file — none of that is what you are learning,
  and reading it would not tell you anything about annotation.
- **The study is not.** Drawing a balanced sample, deciding what counts as a
  disagreement, choosing which statistic to report: those are the methods you have to
  describe in your report and defend in the Q&A. So notebook 02 puts the sampling code
  **in the notebook**, read straight out of the files below when it is generated; 03
  asks you to write the disagreement rule yourself; and 03 and 06 call scikit-learn by
  its own names rather than through a course wrapper, so the composed cell says what
  method produced your number. A method you cannot read is one you cannot defend.

| File | What is in it | When you would open it |
|---|---|---|
| `pipeline.py` | Loading and saving, the connection to the model, `sample_pool`, `run_prompt`, `extract_label`, `build_fewshot`, `export_results` | To see how the sample is drawn, or what happens to the model's reply between "it answered" and "that counts as `Move 2`" |
| `metrics.py` | `show_errors`, `errors_on_disagreed`, `triage_counts` — and `evaluate`, the Day-2 composite that notebook 04 uses for its dev-round score | To see what the dev trail is measured with. Your **reported** numbers come from scikit-learn, written out in notebook 06 |
| `_study.py` | `column`, `percent_agreement`, `disagreements`, `show_errors`, `labels_of` — the small pieces of the method, in one place | It is the source the notebooks and the Day-2 tutorials are both built from, so what you read here is what ran |
| `answers.py` | One worked version of each cell you are asked to write yourself | **After** you have tried: `from answers import answer`, then `answer("disagreements")` |
| `annotate.py` | The Google Sheets round trip: a tab per coder, `Final` to adjudicate in, `append_to_annotation_sheet` for a second draw, and the agreement statistics | When the sheet does something surprising — who it was shared with, how blank rows are treated, how labels are matched back, why adding rows refuses when a column has been renamed — or to see how Fleiss' κ is computed for three or more coders |
| `reshape.py` | Each corpus → the canonical `{id, text, label}` | Notebook 01 already shows you the part that applies to your track |
| `download.py` | One fetch per track | Rarely. ICNALE is deliberately manual — the licence says ask first |
| `prep_datasets.py` | `python scripts/prep_datasets.py <track>` — builds a pool with every decision already made | When you want the file and not the reasoning: rebuilding a pool you have already thought about, or `--demos` for the small stand-ins |
| `make_submission.py` | Collects what you hand in | At the end, once |
| `_generate_pool_notebooks.py`, `_generate_project_notebooks.py`, `_setup_cell.py` | Build the ten notebooks — four pool builders and the six numbered ones | Only if you are changing the template itself. Never hand-edit an `.ipynb` |
| `_check_call_forms.py` | Asserts every call form taught on Days 1–3 still runs here unchanged | After changing any signature. Needs no API key and no network |
| `_check_undefined_names.py` | Asserts every name a notebook cell uses was imported or defined by an earlier cell | After changing a SETUP cell's imports or adding a call to a step. Catches the `NameError` a student would hit on their first run |
| `_check_notebooks.py` | Asserts every code cell is valid Python, fits on a screen, and has a markdown cell above it saying what it does | After regenerating. Catches a cell that would not run, and a generator change that quietly produced a wall of source |
| `_check_study_source.py` | Asserts `_study.py` is byte-identical to the course repo's copy | After editing `_study.py`. The two are one source rendered into two sets of notebooks; if they drift, a tutorial teaches one thing and the project runs another |

If a call you learned in the tutorials does not work in these notebooks, that is a bug
here rather than a gap in your memory of it — please say so.
