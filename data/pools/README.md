# data/pools/

A **pool** is everything usable in a corpus, in the canonical schema, with its natural
label imbalance left intact:

```json
[
  {"id": 1, "text": "There are three main types of chocolate ...", "label": "A1"},
  {"id": 2, "text": "He died on 6 October 1837 in Paris .",        "label": "A1"}
]
```

Two kinds of file live here:

| File | What it is | Where it comes from |
|---|---|---|
| `<track>_pool.json` | the full pool for a track (~1,000–3,200 items) | you build it, in `notebooks/01_build_pool_<track>.ipynb` |
| `<track>_demo_pool.json` | a small stand-in (60–72 items) | ships with the template |

The full pools are **not committed** — they are large, and mostly not ours to
redistribute. Rebuild one any time.

## The labels in here are not your gold standard

Worth being exact about, because the file format is identical and it would be easy to
assume otherwise.

The `label` on a pool item is the **original corpus authors' judgment**. Your gold
standard is something your group makes, in `notebooks/03_annotate.ipynb`, by
re-annotating a sample blind and adjudicating the disagreements. It lands in
[`../gold/`](../gold/).

The pool's labels get used for exactly two things:

1. **Stratifying the draw** in notebook 02 — you cannot sample evenly across classes
   without knowing what the classes are.
2. **A comparison** at the end of notebook 03 — `compare_to_published` shows you every
   item where your group landed somewhere different. That gap is evidence, and one of
   the more interesting things you can put in a report.

They are never the answer key you score the model against.

## Building the full pool for your track

Open `notebooks/01_build_pool_<track>.ipynb` and work through it. It downloads the
original data, walks you through the raw format, and asks you to make the reshaping
**decisions** yourself — which annotations to trust, what each label is called, how
fine-grained the scheme is. Those are the ✏️ cells, and `PLAN.md` asks you to justify
them.

If you only want the file and not the reasoning:

```bash
python scripts/prep_datasets.py raamove     # or cars50 · l2_errors · icnale
```

That runs the same reshaping code with the decisions already made for you. It is the
right tool for rebuilding a pool you have already thought about, and the wrong one for
building it the first time.

Roughly what you get: raamove ≈ 3,100 items · cars50 ≈ 1,300 · l2_errors ≈ 1,000. ICNALE needs a manual download first — see below.

## The demo pools

One per track, so notebooks 02–05 run the moment you clone, before anyone has built
anything:

- **`raamove_demo_pool.json`** — rhetorical moves in RA abstracts (64 sentences, 8 each:
  Background, Gap, Purpose, Method, Result, Conclusion, Contribution, Implication).
- **`cars50_demo_pool.json`** — Swales CARS moves in RA introductions (60 sentences, 20
  each: Move 1–3).
- **`l2_errors_demo_pool.json`** — L2 error category (60 sentences, 15 each:
  Grammatical, Lexical, Mechanical, No error).

> **Use these to watch the pipeline work, not to run your study.** They are *smaller
> than the sample you are meant to draw from a pool*, so sampling 40 items takes most of
> the file and leaves nothing uncontaminated for few-shot examples. `sample_pool` and
> `build_fewshot` will both warn you when this happens. Build the real pool first.

Point at one by using `DEMO_POOL_PATH` instead of `POOL_PATH` — both come from
[`config.py`](../../config.py).

## ICNALE is different

ICNALE GRA is research-use-only and requires registration, so it cannot be downloaded
automatically and must **never** be committed or submitted. There is no demo pool for
it. `notebooks/01_build_pool_icnale.ipynb` tells you what to put where. Both
`.gitignore` and `scripts/make_submission.py` exclude ICNALE-derived files by name.

## Licences

The data here is **not** covered by the repository's MIT licence. All three demo pools
are CC BY 4.0. Per-file terms are in [`LICENSE`](LICENSE); provenance and citations are in
[`../SOURCES.md`](../SOURCES.md). If you report results, cite the original source and
say that the data were reshaped.
