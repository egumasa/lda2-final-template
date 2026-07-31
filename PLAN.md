# PLAN — <group>, track <track>

Members: …

> Fill this in **before your group calls the model even once**, then get it signed off.
> It is *not graded* — it is a gate. Full rationale:
> <https://egumasa.github.io/linguistic-data-analysis-II-2026/final-project/plan.html>
>
> This is the 作戦シート (battle plan) your group sketched in the Day 1–3 huddles,
> finally given a file.

## 1. Unit of analysis

(sentence / whole text) — and why:

> **`cars50` and `raamove` only.** Your items carry the passage they came from, so you
> have a second decision: does the model see the sentence alone (`<track>.txt`) or the
> whole passage first (`<track>_context.txt`)? Scoring is per sentence either way, so
> running both is a real experiment — but say here which one is your baseline, and
> predict in §7 which way it will go before you find out.

## 2. Label set

Exactly the strings that appear in your gold file:

## 3. Are the labels ORDERED?

If yes, in what order? This decides `labels_order` in `config.yaml`.

> Ordered labels get a **weighted κ** as well as the ordinary one: an agreement score that
> counts a near miss (`Low` → `Mid`) as a smaller error than a far one (`Low` → `High`).
> It can only do that if it knows the order.
>
> Watch out: labels that are ordered but not alphabetical (`Low`/`Mid`/`High`) will otherwise
> be scored over `High < Low < Mid`, and the weighted κ reported to three decimal places
> will be quietly wrong.

## 4. The decisions you made building the pool

Notebook 01 left a few cells blank on purpose. Write down what you put in them, and why —
this is the part a script could not have done for you, and the part the Q&A goes to.

- The ✏️ decision(s) your track asked for (the label mapping / the code grouping / the
  granularity / the band cut-offs):
- Why that and not the obvious alternative:
- What it cost you — items dropped, classes merged, rare classes left thin:

## 5. Sampling — and why that way

- Pool file:
- **Which strategy**, and the one-sentence defence (this is report section 1):
  - [ ] `sample_pool` — balanced by label. Readable per-class scores; not what the
    corpus looks like.
  - [ ] `sample_random` — the corpus as it is. Realistic; a rare label may arrive with
    one or two items, and F1 on a class of two means very little.
  - [ ] `sample_by_document` — whole passages (`cars50` · `raamove` only). Forty
    sentences from forty abstracts is a wider claim than forty from three.
  - Because:
- Size of your smallest class (notebook 02 step 1 prints the pool counts) — this is the
  ceiling on a balanced draw, because it cannot take more from a class than the class has:
- `N_PER_CLASS` (or the totals your strategy takes):
- `SEED`:
- Expected total items:
- On `cars50`/`raamove`: how many distinct documents do you expect your items to come
  from? Neighbouring sentences in one passage are not independent observations:

## 6. QC

- CoderA:
- CoderB:
- Adjudicator:

## 7. Prompt plan

Your baseline idea:

The **one** change you predict will help — and why you think so:

> Write the prediction down before you find out. A prompt change that helps for a reason you
> predicted is a finding; one that helps for no reason you can name is a lucky guess, and it
> is hard to defend in the Q&A.

## The pipeline, as an I/O chain

Five notebooks, and each one hands a **file** to the next. Fill in the right-hand column
with the names your group actually used.

| Notebook | Consumes | Produces | Your file |
|---|---|---|---|
| 01 build pool | the raw corpus + your ✏️ decisions | `data/pools/<track>_pool.json` | |
| 02 sample     | the pool + your sampling choice     | `data/gold/…_sample.json`, and the sheet | |
| 03 annotate   | the filled-in sheet + your adjudication | `data/gold/…_gold.json`     | |
| 04 prompt     | the gold set, the pool, your prompt  | `outputs/…_predictions.json`   | |
| 05 report     | the gold set + the frozen run        | `outputs/…_report.md`          | |

Every one of those files lives in your group's shared Drive folder, and every notebook
finds it through `config.yaml` — so there is nothing to email, paste or re-upload between
steps. (The annotation Sheet is the one thing that is not a file; notebook 02 writes its
link to `data/gold/<track>_<group>_sheet.json`, and notebook 03 reads it back, so that it
behaves like one.)

This table is the point of the exercise. If your group can say aloud *"03 consumes the
filled-in sheet and your adjudication, and produces the gold set"*, you understand the pipeline — and
that is what the end of Session 11 checks.

Note where the model appears: **not until 04**. Notebooks 01–03 are the study; the LLM is
the thing being measured by it.

---

**The rule:** no group calls the model until the instructor has read this file. Notebooks 01–03
need no model at all, so there is plenty to get on with while you wait. `PLAN.md` travels in your
submission bundle as evidence the gate was passed — a final run made before sign-off is not
accepted as the final run.
