# Datasets — sources, licences & attribution

Every file under `gold/` and `pools/` is **derived** from one of the open datasets below,
reshaped into this course's canonical schema:

```json
[{"id": 1, "text": "...", "label": "..."}]
```

If you report results from any of them, **cite the original source** and note that the
data were reshaped.

## What is committed, and what you build

| Folder | Contents | In git? |
|---|---|---|
| `pools/<track>_demo_pool.json` | small balanced demo pools (60–72 items) so notebooks 02–05 run the moment you clone | ✅ committed |
| `gold/<track>_<group>_gold.json` | **your** sampled + adjudicated gold set | ❌ yours, stays local |
| `pools/<track>_pool.json` | the full-size pool you sample from (1,000–3,200 items) | ❌ build it yourself |
| `raw/` | the original downloads the builder reshapes | ❌ build it yourself |

Build any pool with:

```bash
python scripts/prep_datasets.py <track>      # cefr · raamove · cars50 · l2_errors · icnale
```

The demo sets are committed because they are small and their licences permit
redistribution with attribution. The full pools are not: they are large, and for ICNALE
redistribution is **not permitted at all**.

> **Demo files are for seeing the pipeline work, not for your study.** A 60-item "pool"
> is smaller than the sample you are meant to draw from it, which leaves nothing for
> few-shot examples. `sample_pool` and `build_fewshot` will warn you. Build the real pool
> before your final run.

## Licences

| Track | `label` | Source | Licence |
|---|---|---|---|
| `cefr` | CEFR level A1–C2 | CEFR-SP, Wiki-Auto portion | **CC BY-SA 3.0** (share-alike) |
| `raamove` | RA-abstract move (8 classes) | RAAMove | CC BY 4.0 |
| `cars50` | RA-intro Move (3) / Move+Step (11) | CaRS-50 | CC BY 4.0 |
| `l2_errors` | L2 error category / detection | AutoErrorAnalyzer (OSF) | CC BY 4.0 |
| `icnale` | Holistic score band (AWE) | ICNALE GRA | **research use only — do not redistribute** |

Notes:

- **CEFR-SP** ships text only for the **Wiki-Auto** (CC BY-SA 3.0) and SCoRE
  (CC BY-NC-SA 4.0) portions; the Newsela portion is access-gated. Our files are built
  from **Wiki-Auto only**, keeping sentences where **both annotators agree** (clean,
  unambiguous → a good on-ramp). Derived files inherit **CC BY-SA 3.0 (share-alike)**,
  which is why `gold/` carries its own `LICENSE` file separate from the code.
- **L2 errors** is built from the OSF `Analysis/data_category.csv`, which holds each
  sentence's **human gold** error codes *and* the published tool's predictions
  (`AEA_ErrorCategories`) — so you can compare your LLM not only to the human gold but to
  the original tool. The 23 error codes are collapsed to broader categories
  (Grammatical / Lexical / Mechanical / No error); sentences spanning more than one
  broader category are dropped, to keep it a clean single-label task.
- **ICNALE GRA** requires registration (a password-protected zip). It cannot be
  downloaded automatically, and it must never be committed or included in a submission
  bundle. `.gitignore` and `scripts/make_submission.py` both exclude it by name.

## Citations

- **RAAMove** — Liu, J. et al. *RAAMove: A Corpus for Analyzing Moves in Research Article
  Abstracts.* LREC-COLING 2024. (Public release: 400 abstracts / 3,069 sentences, 8-move
  scheme BAC/GAP/MTD/PUR/RST/CLN/CTN/IMP; κ = 0.785.) CC BY 4.0.
- **CaRS-50** — Lam, C. & Nnamoko, N. (2025). *CaRS-50 Dataset: Annotated corpus of
  rhetorical Moves and Steps in 50 article introductions.* Mendeley Data, V1.
  doi:10.17632/kwr9s5c4nk.1. (50 BioRxiv intros, sentence-level Swales CARS Move+Step;
  inter-rater κ ≈ 0.43.) CC BY 4.0.
- **CEFR-SP** — Arase, Y., Uchida, S., & Kajiwara, T. (2022). *CEFR-based Sentence
  Difficulty Annotation and Assessment.* EMNLP 2022.
- **AutoErrorAnalyzer** — Mizumoto, A. (2025). *Automated analysis of common errors in L2
  learner production: Prototype web application development.* Studies in Second Language
  Acquisition, 47(3), 867–884. (26-category error taxonomy; ~100 Japanese-EFL essays;
  gold annotations, Krippendorff's α ≈ .92.)
- **ICNALE GRA** — Ishikawa, S. *The ICNALE Global Rating Archives.* (Asian-learner L2
  English essays/speeches rated on holistic + analytic scales by many trained raters.)

## Motivating study (not openly available)

- **Kim, M. & Lu, X. (2024).** *Exploring the potential of using ChatGPT for rhetorical
  move-step analysis: The impact of prompt refinement, few-shot learning, and fine-tuning.*
  Journal of English for Academic Purposes, 71, 101422. doi:10.1016/j.jeap.2024.101422.
  No open replication package (the corpus is the non-public *Corpus of Social Science RA
  Introductions*, Lu et al. 2021). **CaRS-50** is the open stand-in for the same task.
