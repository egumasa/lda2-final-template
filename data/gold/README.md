# data/gold/

This folder holds the **pool** of labeled items your notebook samples from. Each file is
a JSON list in the course's canonical shape:

```json
[
  {"id": 1, "text": "There are three main types of chocolate ...", "label": "A1"},
  {"id": 2, "text": "He died on 6 October 1837 in Paris .",        "label": "A1"}
]
```

## What ships with the template

- **`cefr_pool.json`** — a ready-to-run CEFR sample (72 sentences, 12 per level) so the
  `cefr` notebook works the moment you clone the repo. Use it to see the whole pipeline
  run end to end before you switch to your own track.

## Adding your track's data

Every notebook's `POOL_PATH` points here, e.g. `data/gold/raamove_pool.json`. To run a
track other than CEFR:

1. **Build the pool file** for your track using the download + preprocess notebooks in the
   course repo (`sources/resources/datasets/` — see that folder's `README.md` and
   `SOURCES.md` for provenance and licenses). They reshape the raw data into the canonical
   `{id, text, label}` shape.
2. **Drop the file in here** with the name the notebook expects
   (`<track>_pool.json` — e.g. `raamove_pool.json`, `cars50_pool.json`,
   `l2_errors_pool.json`).
3. Set `POOL_PATH` in the notebook's CONFIG cell to match.

## Keep your own data in Google Drive

The pools you build for the real project can be large and are **not** committed to git
(see `.gitignore`). Follow the course's *Housing Your Data in Google Drive* guide: mount
Drive, save your pool there, and point `POOL_PATH` at the Drive location. If you cloned
this template **into** your Drive, files you save here already persist across sessions.
