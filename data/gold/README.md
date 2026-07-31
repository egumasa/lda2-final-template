# data/gold/

Everything in this folder is **your group's work**. Nothing here ships with the
template, and nothing here is committed to git — it goes into your submission bundle
instead, which `scripts/make_submission.py` assembles for you.

Two files land here, in this order:

| File | What it is | Written by |
|---|---|---|
| `<track>_<group>_sample.json` | the subset you drew from the pool | `notebooks/02_sample.ipynb` |
| `<track>_<group>_gold.json` | **your gold standard** — annotated blind, adjudicated | `notebooks/03_annotate.ipynb` |

Both use the same canonical shape as everything else in the project:

```json
[
  {"id": 1, "text": "There are three main types of chocolate ...", "label": "A1"},
  {"id": 2, "text": "He died on 6 October 1837 in Paris .",        "label": "A1"}
]
```

## The difference between the two, which matters

They look identical. They are not.

- In **`_sample.json`**, the `label` is still the **published** one — whatever the
  original corpus authors decided. Notebook 02 carries it along so that notebook 03 can
  compare against it at the very end. Nobody in your group has agreed with it yet.
- In **`_gold.json`**, the `label` is **yours**: two of you annotated the same items
  independently and blind, measured how far apart you were, and argued out every row you
  disagreed on. That is the file every number in notebooks 04 and 05 is measured
  against.

The pools you sample from — including the labels in the first file — live one folder up,
in [`../pools/`](../pools/).

## Look after the gold file

It is the single most valuable thing your group makes all week: hours of judgment, and
the only artefact in the project that could not have been produced by a script. It is
already safe: the Setup cell will not let a notebook run in Colab from anywhere but your
group's shared Drive folder, so this file is written there and stays there.

Alongside it you will find `<track>_<group>_sheet.json`, the link to the annotation Sheet
notebook 03 created. That is why it is a file too — the sheet is where the judgment
happens, and a link that lives only in one person's notebook output is a link the group
can lose.

`export_results` in notebook 05 writes a copy into `outputs/` too, but do not wait for
that — notebook 03 saves it the moment you have it, on purpose.
