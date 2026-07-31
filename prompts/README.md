# prompts/

Your prompt lives here as a **text file**, not pasted inside the notebook. This is on
purpose: prompt engineering is iteration, and keeping each version in a file makes the
changes easy to see, save, and compare.

There is one starter prompt per track — `cefr.txt`, `raamove.txt`, `cars50.txt`,
`l2_errors.txt` (write your own `icnale.txt` if you take that track). `mini_project.ipynb`
loads the one matching its CONFIG cell: `PROMPT_FILE = "../prompts/" + TRACK + ".txt"`.

## How it works

The notebook loads a prompt with `load_prompt(PROMPT_FILE)`. The file must contain the
placeholder `{text}` — that is where each sentence gets slotted in when the model is asked
to classify it.

```
Classify the sentence into exactly one of: A1, A2, B1, B2, C1, C2.
Answer with the level only.

Sentence: {text}
```

## How to iterate

Two easy ways:

1. **Edit your track's file in place** (e.g. `cefr.txt`) and re-run. Simplest.
2. **Save each version as its own file** — `cefr_v0.txt`, `cefr_v1_added_definitions.txt`,
   `cefr_v2_stricter.txt`, and `cefr.txt` for your final — then point `PROMPT_FILE` at the
   one you want. This keeps a record of what you tried, which is exactly what the "Prompt
   iterations" table in your report needs. `make_submission.py` collects every
   `<track>*.txt`, so the whole trail is handed in.

In Colab you can write a version straight from a notebook cell, which means the wording
sits in the notebook where your group can co-edit and argue about it:

```python
%%writefile ../prompts/cefr_v1.txt
Classify the CEFR level of the sentence. Answer with the level only.
...
Sentence: {text}
```

The few-shot rounds in the notebook take whichever prompt you loaded and automatically add
labeled examples in front of it (`build_fewshot`), so you do **not** write the examples by
hand — just design the base instruction here.

> Match the label names in your prompt to the labels in your gold file.
