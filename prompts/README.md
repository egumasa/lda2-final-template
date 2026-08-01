# prompts/

Every version of your prompt ends up here as a **text file**. You write the prompt in
the notebook cell, where editing it is what changes what gets sent, and then
`save_prompt` writes it here — because `05_test.ipynb` is a different notebook and
nothing survives between notebooks except what is on disk.

There is one starter prompt per track — `raamove.txt`, `cars50.txt`, `cars50_step.txt`,
`l2_errors.txt`, `l2_error_detection.txt` (write your own `icnale.txt` if you take that
track, since only your group knows what its bands are called).

`04_develop.ipynb` can open the one whose name matches `track:` in `config.yaml`, with
`load_prompt(PROMPT_FILE)`. It never asks you for a path: `PROMPT_FILE` is built for you
as `prompts/` + your track + `.txt`, so setting `track: cars50` is what makes it read
`prompts/cars50.txt`.

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

**Write the prompt in the notebook cell**, as an ordinary string. Editing it and running
the cell again is what changes what gets sent, so your group can co-edit and argue about
the wording in the place where it takes effect:

```python
PROMPT_v1 = """Classify the rhetorical move of the sentence.
Answer with the move name only.

Sentence: {text}"""
```

**Then save each version as its own file**, with `save_prompt`:

```python
save_prompt(PROMPT_v1, ROOT / "prompts" / "raamove_v1.txt")
```

Give the versions names you can tell apart — `raamove_v0.txt`,
`raamove_v1_added_definitions.txt`, `raamove_v2_stricter.txt`. This keeps a record of
what you tried, which is exactly what the "Prompt iterations" table in your report needs,
and `make_submission.py` collects every `<track>*.txt`, so the whole trail is handed in.

Saving is not optional bookkeeping: `05_test.ipynb` is a different notebook and can only
load a prompt from disk. A version that only ever existed as a string in one session is
one you cannot test.

`build_fewshot` takes whichever prompt you give it and adds labeled examples in front of
it, so you do **not** write the examples by hand — just design the base instruction. Note
that it draws them with a fixed seed, so calling it twice with the same arguments gives
you the same prompt both times.

> Match the label names in your prompt to the labels in your gold file.
