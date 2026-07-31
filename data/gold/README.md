# data/gold/

This folder holds **gold sets** — items with a label attached. Two different kinds live
here, and it is worth keeping them straight:

| File | What it is | Where it comes from |
|---|---|---|
| `<track>_demo.json` | a small balanced demo set (60–72 items) | ships with the template |
| `<track>_<group>_gold.json` | **your** sampled + adjudicated gold set | step 2 of your notebook writes it |

The full-size **pools** you sample from live one folder up, in [`../pools/`](../pools/) —
they are not committed, because they are large and mostly not ours to redistribute.

Every file, in both folders, uses the same canonical shape:

```json
[
  {"id": 1, "text": "There are three main types of chocolate ...", "label": "A1"},
  {"id": 2, "text": "He died on 6 October 1837 in Paris .",        "label": "A1"}
]
```

## The demo sets

One per track, so **every** track runs the moment you clone the repo:

- **`cefr_demo.json`** — CEFR sentence level (72 sentences, 12 per level: A1–C2).
- **`raamove_demo.json`** — rhetorical moves in RA abstracts (64 sentences, 8 each:
  Background, Gap, Purpose, Method, Result, Conclusion, Contribution, Implication).
- **`cars50_demo.json`** — Swales CARS moves in RA introductions (60 sentences, 20 each:
  Move 1–3).
- **`l2_errors_demo.json`** — L2 error category (60 sentences, 15 each: Grammatical,
  Lexical, Mechanical, No error).

> **Use these to watch the pipeline work, not to run your study.** They are *smaller than
> the sample you are meant to draw from a pool*, so sampling 40 items takes most of the
> file and leaves nothing uncontaminated for few-shot examples. `sample_pool` and
> `build_fewshot` will both warn you when this happens. Build the real pool first.

## Getting the full pool for your track

```bash
python scripts/prep_datasets.py cefr        # or raamove · cars50 · l2_errors · icnale
```

This downloads the original data into `../raw/`, reshapes it, and writes
`../pools/<track>_pool.json`. Then point `POOL_PATH` in the notebook's CONFIG cell at it:

```python
POOL_PATH = "../data/pools/" + TRACK + "_pool.json"
```

Prefer to see the steps rather than run one command? The matching
`notebooks/download_<track>.ipynb` does the same job cell by cell, and works in Colab.

Roughly what you get: cefr ≈ 3,200 items · raamove ≈ 3,100 · cars50 ≈ 1,300 ·
l2_errors ≈ 1,000. ICNALE needs a manual download first — see below.

## ICNALE is different

ICNALE GRA is research-use-only and requires registration, so it cannot be downloaded
automatically and must **never** be committed or submitted. The builder will tell you what
to put where. Both `.gitignore` and `scripts/make_submission.py` exclude ICNALE-derived
files by name.

## Licences

The data here is **not** covered by the repository's MIT licence. The CEFR demo carries a
**share-alike** obligation (CC BY-SA 3.0); the other three are CC BY 4.0. Per-file terms
are in [`LICENSE`](LICENSE); provenance and citations are in
[`../SOURCES.md`](../SOURCES.md). If you report results, cite the original source and say
that the data were reshaped.

## Your own gold set

Your adjudicated gold (`<track>_<group>_gold.json`) is **git-ignored** — it is your work,
not part of the template. It goes into your submission bundle instead, which
`scripts/make_submission.py` assembles for you. If you cloned this repo into your Google
Drive, it is already saved across sessions.
