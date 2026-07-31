# prompts/

Your prompt lives here as a **text file**, not pasted inside the notebook. This is on
purpose: prompt engineering is iteration, and keeping each version in a file makes the
changes easy to see, save, and compare.

There is one starter prompt per track — `raamove.txt`, `cars50.txt`,
`l2_errors.txt` (write your own `icnale.txt` if you take that track). `04_prompt.ipynb`
loads the one matching `track:` in `config.yaml`: `PROMPT_FILE = ROOT / "prompts" / (TRACK + ".txt")`.

## How it works

The notebook loads a prompt with `load_prompt(PROMPT_FILE)`. The file must contain the
placeholder `{text}` — that is where each sentence gets slotted in when the model is asked
to classify it.

```
Classify the sentence into exactly one of: Background, Gap, Purpose, Method,
Result, Conclusion, Contribution, Implication. Answer with the move only.

Sentence: {text}
```

## `{context}`, on the two rhetorical-move tracks

`cars50` and `raamove` ask what a sentence *does in a passage* — `Method` vs `Result`,
`Move 1` vs `Move 3`. That is not always decidable from the sentence on its own, so items
on those two tracks carry the passage they came from, and a prompt may use a second
placeholder, `{context}`, to show it.

Both conditions ship, and the pair is a ready-made experiment:

| file | the model sees |
|---|---|
| `raamove.txt` · `cars50.txt` | the sentence alone — **the baseline** |
| `raamove_context.txt` · `cars50_context.txt` | the whole passage, then the sentence |

Scoring is per sentence either way, so the two runs are directly comparable — which is the
whole point. Write down which way you expect it to go **before** you run the second one;
that prediction is what `PLAN.md` §7 is asking for.

Two things to watch:

- `{context}` is only filled in on those two tracks. Use it in an `l2_errors`
  prompt and `run_prompt` will warn you that the model is being shown an empty passage.
- Changing the prompt **and** switching on few-shot at the same time changes two things
  at once, and you will not know which one moved the number. `build_fewshot` shows its
  examples as bare sentences, without their passages.

## How to iterate

Two easy ways:

1. **Edit your track's file in place** (e.g. `raamove.txt`) and re-run. Simplest.
2. **Save each version as its own file** — `raamove_v0.txt`,
   `raamove_v1_added_definitions.txt`, `raamove_v2_stricter.txt`, and `raamove.txt` for
   your final — then point `PROMPT_FILE` at the
   one you want. This keeps a record of what you tried, which is exactly what the "Prompt
   iterations" table in your report needs. `make_submission.py` collects every
   `<track>*.txt`, so the whole trail is handed in.

In Colab you can write a version straight from a notebook cell, which means the wording
sits in the notebook where your group can co-edit and argue about it:

```python
%%writefile ../prompts/raamove_v1.txt
Classify the rhetorical move of the sentence. Answer with the move name only.
...
Sentence: {text}
```

The few-shot rounds in the notebook take whichever prompt you loaded and automatically add
labeled examples in front of it (`build_fewshot`), so you do **not** write the examples by
hand — just design the base instruction here.

> Match the label names in your prompt to the labels in your gold file.
