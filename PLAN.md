# PLAN — <group>, track <track>

Members: …

> Fill this in **before your group calls the model even once**, then get it signed off.
> It is *not graded* — it is a gate. Full rationale:
> <https://egumasa.github.io/linguistic-data-analysis-II-2026/final-project/plan.html>
>
> This is the 作戦シート from the S3, S6 and S9 huddles, finally given a file.

## 1. Unit of analysis

(sentence / whole text) — and why:

## 2. Label set

Exactly the strings that appear in your gold file:

## 3. Are the labels ORDERED?

If yes, in what order? This decides `LABELS_ORDER` in the notebook's CONFIG cell.

> Watch out: labels that are ordered but not alphabetical (`Low`/`Mid`/`High`) will otherwise
> be scored over `High < Low < Mid`, and the weighted κ reported to three decimal places
> will be quietly wrong.

## 4. Gold

- Pool file:
- `N_PER_CLASS`:
- `SEED`:
- Expected total items:

## 5. QC

- CoderA:
- CoderB:
- Adjudicator:

## 6. Prompt plan

Your baseline idea:

The **one** change you predict will help — and why you think so:

> Write the prediction down before you find out. A prompt change that helps for a reason you
> predicted is a finding; one that helps for no reason you can name is a lucky guess, and it
> is hard to defend in the Q&A.

## The pipeline, as an I/O chain

| # | Step | Consumes | Produces |
|---|------|----------|----------|
| 1 | sample           | POOL_PATH, N_PER_CLASS, SEED | pool, sampled, LABELS |
| 2 | QC / adjudicate  | sampled, LABELS, the sheet   | gold |
| 3 | baseline         | PROMPT, gold                 | pred0, f1_by_round["0…"] |
| 4 | iterate + freeze | PROMPT, pool, gold           | pred_final (a JSON file) |
| 5 | error analysis   | gold, pred_final             | the errors table |
| 6 | export           | all of the above             | outputs/…_report.md |

This table is the point of the exercise. If your group can say aloud *"step 2 consumes
`sampled` and the sheet, and produces `gold`"*, you understand the pipeline — and that is what
the end of Session 11 checks.

---

**The rule:** no group calls the model until the instructor has read this file. Steps 1 and 2
need no model, so there is plenty to get on with while you wait. `PLAN.md` travels in your
submission bundle as evidence the gate was passed — a final run made before sign-off is not
accepted as the final run.
