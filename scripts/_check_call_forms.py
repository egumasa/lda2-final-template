"""_check_call_forms.py — the contract test for this template.

Run it after ANY change to a signature in pipeline.py / metrics.py / annotate.py:

    uv run python scripts/_check_call_forms.py

WHY THIS EXISTS
---------------
Students arrive at the project having spent three days typing a particular set of
calls in the Day 1-3 notebooks. If a helper here quietly grows a required argument,
their muscle memory breaks -- during the highest-stakes session of the week -- with a
TypeError that has nothing to teach them. Worse, if an argument changes MEANING
rather than arity (a bool flag becoming a label list, say), nothing raises at all and
the reported numbers are simply wrong.

So this file is the executable form of the rule in pipeline.py's docstring: every
call form taught in Days 1-3 must run here unchanged. It needs no API key and makes
no network calls -- the model is replaced by a stub.
"""

import json
import sys
import tempfile
from pathlib import Path

# evaluate() draws a confusion matrix. There is no screen here, so draw to an
# off-screen buffer and make show() do nothing -- otherwise every call warns.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
from pipeline import (
    build_fewshot,
    label_set,
    load_gold,
    load_predictions,
    reid,
    run_prompt,
    sample_pool,
    save_predictions,
)
from metrics import agreement, evaluate, show_errors

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

_failures = []
_checks = 0


def check(description, function):
    """Run one check. Record the failure and keep going, so one run reports everything."""
    global _checks
    _checks = _checks + 1
    try:
        function()
        print("  ok   ", description)
    except Exception as error:
        print("  FAIL ", description)
        print("         ", type(error).__name__ + ":", error)
        _failures.append(description)


def install_stub_backend():
    """Replace the LLM with a stub, by pre-filling the cache make_backend() uses.

    This is also a check in itself: if the caching mechanism is ever removed,
    run_prompt would try to reach the real API here and this file would hang.
    """
    call_count = [0]

    def stub_generate_text(prompt):
        call_count[0] = call_count[0] + 1
        # Cycle through the labels so predictions are neither all-right nor all-wrong.
        return LEVELS[call_count[0] % len(LEVELS)]

    pipeline._BACKEND = (stub_generate_text, "stub backend (no network)")
    return call_count


def make_gold(n_per_label=2):
    """A synthetic gold set in the canonical {id, text, label} shape."""
    items = []
    next_id = 1
    for label in LEVELS:
        for k in range(n_per_label):
            items.append({
                "id": next_id,
                "text": "Sentence number " + str(next_id) + " for level " + label + ".",
                "label": label,
            })
            next_id = next_id + 1
    return items


def make_pool():
    """A pool big enough that sampling leaves spare items for few-shot examples."""
    return make_gold(n_per_label=10)


def main():
    call_count = install_stub_backend()
    gold = make_gold()
    pool = make_pool()
    labels = label_set(gold)
    PROMPT = "Classify this sentence.\n\nSentence: {text}"

    work_dir = Path(tempfile.mkdtemp(prefix="lda2_check_"))

    print("\nDay 1-3 call forms (these must never break)")

    def gold_roundtrip():
        path = work_dir / "gold.json"
        path.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        loaded = load_gold(path)
        assert loaded == gold, "load_gold did not return what was written"
    check("load_gold(url_or_path) -> list", gold_roundtrip)

    predictions = []

    def day3_run_prompt():
        # THE Day-3 form. Two arguments, no labels, no backend threaded through.
        result = run_prompt(PROMPT, gold)
        assert isinstance(result, list), "run_prompt must return a list"
        assert len(result) == len(gold), "one prediction per gold item"
        predictions.extend(result)
    check("run_prompt(PROMPT, gold) -> list[str]", day3_run_prompt)

    def day2_evaluate():
        score = evaluate(gold, predictions)
        assert isinstance(score, float), "evaluate must return the macro-F1 as a float"
    check("evaluate(gold, pred) -> float", day2_evaluate)

    def day3_evaluate_ordered():
        score = evaluate(gold, predictions, ordered=True)
        assert isinstance(score, float), "evaluate must return the macro-F1 as a float"
    check("evaluate(gold, pred, ordered=True) -> float", day3_evaluate_ordered)

    def day3_show_errors():
        table = show_errors(gold, predictions)
        assert hasattr(table, "shape"), "show_errors must return a DataFrame"
    check("show_errors(gold, pred) -> DataFrame", day3_show_errors)

    print("\nThe template's own extended forms (both arities must work)")

    def legacy_evaluate():
        # The pre-alignment 4-positional form, kept working via the shim.
        score = evaluate(gold, predictions, labels, "a title")
        assert isinstance(score, float), "the legacy shim must still return a float"
    check("evaluate(gold, pred, labels, title) -> float  [legacy shim]", legacy_evaluate)

    def run_prompt_extended():
        stub = pipeline._BACKEND[0]
        result = run_prompt(PROMPT, gold, labels, stub)
        assert len(result) == len(gold)
    check("run_prompt(prompt, gold, labels, generate_text)", run_prompt_extended)

    def sampling():
        a = sample_pool(pool, 3, 42)
        b = sample_pool(pool, 3)          # seed defaulted
        assert len(a) == len(b), "the default seed must be the documented one (42)"
        assert a == b, "same seed must give the same sample"
        assert [item["id"] for item in a] == list(range(1, len(a) + 1)), \
            "sample_pool must renumber ids from 1"
    check("sample_pool(pool, n, seed) and sample_pool(pool, n)", sampling)

    def fewshot():
        sampled = sample_pool(pool, 3, 42)
        short = build_fewshot(PROMPT, pool, sampled)
        long = build_fewshot(PROMPT, pool, sampled, labels, 1, 42)
        assert "{text}" in short, "the {text} placeholder must survive"
        assert short == long, "the defaults must match the explicit arguments"
        # No gold text may appear among the examples, or we are testing on the answers.
        example_block = short.split("Now classify this one.")[0]
        for item in sampled:
            assert item["text"] not in example_block, \
                "build_fewshot leaked a gold item into the examples"
    check("build_fewshot(P, pool, gold) and the 6-argument form", fewshot)

    def freeze_roundtrip():
        path = work_dir / "frozen.json"
        save_predictions(predictions, path)
        assert load_predictions(path) == predictions, \
            "a frozen run must load back identically"
    check("save_predictions -> load_predictions round-trip", freeze_roundtrip)

    def agreement_keys():
        result = agreement(["A1", "A2", "B1"], ["A1", "B1", "B1"])
        assert "kappa" in result, "the 'kappa' alias key is missing"
        assert "cohen_kappa" in result, "the 'cohen_kappa' key is missing"
        assert result["kappa"] == result["cohen_kappa"]
    check("agreement(a, b) -> dict with both kappa keys", agreement_keys)

    print("\nEdge cases that used to produce a wrong number silently")

    def single_label_kappa():
        one = [{"id": 1, "text": "x", "label": "A1"}, {"id": 2, "text": "y", "label": "A1"}]
        score = evaluate(one, ["A1", "A1"])
        assert isinstance(score, float), "a one-label gold set must not crash"
    check("evaluate on a single-label gold set (kappa undefined, not nan)",
          single_label_kappa)

    def reid_is_a_copy():
        original = [{"id": 99, "text": "x", "label": "A1"}]
        reid(original)
        assert original[0]["id"] == 99, "reid must not mutate its input"
    check("reid(items) leaves the original untouched", reid_is_a_copy)

    print("\nThe annotation round-trip (annotate.py)")
    try:
        from annotate import annotator_agreement, disagreements, to_canonical
    except ImportError as error:
        print("  SKIP  annotate.py not present yet —", error)
    else:
        rows = [
            {"ID": "1", "Text": "first", "CoderA": "A1", "CoderB": "A1",
             "Final": "A1", "Note": ""},
            {"ID": "2", "Text": "second", "CoderA": "A2", "CoderB": "B1",
             "Final": "A2", "Note": "borderline"},
            {"ID": "3", "Text": "third", "CoderA": "B1", "CoderB": "B1",
             "Final": "B1", "Note": ""},
        ]

        def canonical():
            result = to_canonical(rows, labels)
            assert len(result) == 3, "all three rows are usable"
            for item in result:
                assert set(item.keys()) == {"id", "text", "label"}, \
                    "to_canonical must emit exactly {id, text, label}"
        check("to_canonical(rows, labels) -> canonical gold", canonical)

        def canonical_keeps_context():
            # The rhetorical-move tracks carry a passage the SHEET cannot hold: it has
            # only ID, Text and your label. Without source= it is dropped here and
            # notebook 04 prompts with nothing.
            source = [{"id": 1, "text": "first", "label": "A1",
                       "doc_id": "text001", "sent_index": 0, "n_sents": 2,
                       "context": "first\nsecond"}]
            result = to_canonical(rows, labels, source=source)
            first = result[0]
            assert first["context"] == "first\nsecond", "context must survive"
            assert first["label"] == "A1", "the SHEET's adjudicated label still wins"
            assert set(result[1].keys()) == {"id", "text", "label"}, \
                "an item with nothing extra stays bare"
        check("to_canonical(..., source=sampled) carries context into gold",
              canonical_keeps_context)

        def sheet_agreement():
            result = annotator_agreement(rows)
            assert "kappa" in result
        check("annotator_agreement(rows) -> dict", sheet_agreement)

        def sheet_disagreements():
            table = disagreements(rows)
            assert len(table) == 1, "exactly one row disagrees"
        check("disagreements(rows) -> DataFrame", sheet_disagreements)

        def canonical_rejects_bad_labels():
            bad = [{"ID": "1", "Text": "first", "CoderA": "", "CoderB": "",
                    "Final": "b1", "Note": ""}]     # lowercase: not in the label set
            result = to_canonical(bad, labels)
            assert len(result) == 0, \
                "to_canonical must reject a label that is not in the label set"
        check("to_canonical rejects an invalid label (lowercase 'b1')",
              canonical_rejects_bad_labels)

        def canonical_survives_typed_over_id():
            bad = [{"ID": "not a number", "Text": "first", "CoderA": "", "CoderB": "",
                    "Final": "A1", "Note": ""}]
            result = to_canonical(bad, labels)     # must report, not raise
            assert len(result) == 0
        check("to_canonical survives a typed-over ID cell",
              canonical_survives_typed_over_id)

        from annotate import compare_to_published

        def published_matches_by_text():
            # THE REGRESSION THIS GUARDS: sample_pool renumbers ids from 1, so matching
            # a 42-item gold set to a 3183-item pool BY ID pairs your item 7 with pool
            # item 7 - two unrelated sentences - and reports a meaningless percentage
            # without ever failing. Matching on text is what makes the number real.
            sampled = sample_pool(pool, 3, 42)
            # Our labels agree with the pool's by construction, so a correct
            # text-based match must report 100%.
            table = compare_to_published(sampled, pool)
            assert table is not None, "the sampled items must be findable in the pool"
            assert len(table) == 0, (
                "every sampled item carries its pool label, so nothing should differ - "
                "a non-empty table means items are being paired by id, not text")
        check("compare_to_published pairs items by text after reid()",
              published_matches_by_text)

    print()
    if _failures:
        print(len(_failures), "of", _checks, "checks FAILED:")
        for description in _failures:
            print("  -", description)
        print("\nA broken call form means a student's Day-2/3 code will not run in the")
        print("template. Fix the signature rather than the notebook.")
        return 1
    print("All", _checks, "checks passed. Day-1-3 call forms still run unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
