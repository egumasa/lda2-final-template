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

import builtins
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
    export_results,
    label_set,
    load_gold,
    load_json,
    load_predictions,
    read_test_log,
    record_test_scoring,
    reid,
    run_prompt,
    sample_pool,
    save_json,
    save_predictions,
    save_test_run,
    split_dev_test,
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
    """Replace the LLM with a stub, so this file needs no key and makes no calls.

    It stands in for the raw connection make_backend() would build, which means the
    pacing and retry wrapper around it still runs and is still exercised here. Pacing
    is turned right down, or forty stub calls would take three minutes.

    This is also a check in itself: if the remembered-connection mechanism is ever
    removed, run_prompt would try to reach the real API and this file would hang.
    """
    call_count = [0]

    def stub_call_model(prompt):
        call_count[0] = call_count[0] + 1
        # Cycle through the labels so predictions are neither all-right nor all-wrong.
        return LEVELS[call_count[0] % len(LEVELS)]

    pipeline._CALL_MODEL = stub_call_model
    pipeline._BACKEND_NAME = "stub backend (no network)"
    pipeline._MODEL_IN_USE = "stub"
    pipeline._MIN_INTERVAL = 0.0
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

    def run_prompt_extended():
        sender = pipeline._default_backend()
        result = run_prompt(PROMPT, gold, labels, sender)
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

    print("\nThe dev/test split, and the audit trail on the held-out run")

    def split_partitions():
        dev, test = split_dev_test(gold, dev_per_class=1)
        dev_ids = [item["id"] for item in dev]
        test_ids = [item["id"] for item in test]
        assert len(set(dev_ids) & set(test_ids)) == 0, "dev and test must not overlap"
        assert sorted(dev_ids + test_ids) == sorted(item["id"] for item in gold), \
            "every gold item must land on exactly one side of the line"
    check("split_dev_test(gold, dev_per_class=n) partitions the gold set", split_partitions)

    def split_fraction():
        dev, test = split_dev_test(gold, dev_fraction=0.5)
        assert len(dev) > 0 and len(test) > 0, "both halves must be non-empty"
        assert len(dev) + len(test) == len(gold)
    check("split_dev_test(gold, dev_fraction=f) partitions the gold set", split_fraction)

    def split_spec_is_exclusive():
        for bad_call in (lambda: split_dev_test(gold),
                         lambda: split_dev_test(gold, dev_per_class=1, dev_fraction=0.3)):
            try:
                bad_call()
            except ValueError:
                continue
            raise AssertionError("exactly one of dev_per_class / dev_fraction is required")
    check("split_dev_test rejects both specs, and neither", split_spec_is_exclusive)

    def split_is_seeded():
        a = split_dev_test(gold, dev_per_class=1, seed=42)
        b = split_dev_test(gold, dev_per_class=1)          # seed defaulted
        c = split_dev_test(gold, dev_per_class=1, seed=7)
        assert a == b, "the default seed must be the documented one (42)"
        assert a != c, "a different seed must give a different split"
    check("split_dev_test(gold, ..., seed) is reproducible", split_is_seeded)

    def split_keeps_ids():
        # THE REGRESSION THIS GUARDS: the three samplers all call reid(), and copying
        # that here would renumber the gold ids. Notebook 05 joins the model's errors
        # against the ids in the annotation sheet, and that join would silently start
        # comparing unrelated rows.
        dev, test = split_dev_test(gold, dev_per_class=1)
        by_id = {}
        for item in gold:
            by_id[item["id"]] = item["text"]
        for item in dev + test:
            assert by_id[item["id"]] == item["text"], \
                "split_dev_test must NOT renumber ids - notebook 05's join runs on them"
    check("split_dev_test preserves the gold ids", split_keeps_ids)

    def split_serves_test_first():
        # A label with a single item cannot appear on both sides. It has to be the one
        # that survives in TEST, or it drops out of the reported macro average unseen.
        thin = make_gold(1)[:1] + make_gold(3)[1:]
        dev, test = split_dev_test(thin, dev_per_class=2)
        test_labels = set(item["label"] for item in test)
        for label in label_set(thin):
            assert label in test_labels, \
                "every label present in gold must survive into test"
    check("split_dev_test keeps every label in test", split_serves_test_first)

    def split_by_document():
        docs = []
        for index, item in enumerate(make_gold(3)):
            with_doc = dict(item)
            with_doc["doc_id"] = "doc" + str(index % 4)
            docs.append(with_doc)
        dev, test = split_dev_test(docs, dev_fraction=0.4, by_document=True)
        dev_docs = set(item["doc_id"] for item in dev)
        test_docs = set(item["doc_id"] for item in test)
        assert len(dev_docs & test_docs) == 0, \
            "a document must not have sentences on both sides of the line"
    check("split_dev_test(..., by_document=True) keeps documents whole", split_by_document)

    def fewshot_excludes_test():
        # Handed only `dev`, build_fewshot could pick a TEST item as a worked example -
        # putting the answer to a held-out item into the prompt that produces the
        # headline number. Notebook 04 passes the full gold set for exactly this reason.
        sampled = sample_pool(pool, 3, 42)
        dev, test = split_dev_test(sampled, dev_per_class=1)
        example_block = build_fewshot(PROMPT, pool, sampled).split(
            "Now classify this one.")[0]
        for item in test:
            assert item["text"] not in example_block, \
                "build_fewshot(P, pool, gold) leaked a held-out item into the examples"
    check("build_fewshot(P, pool, gold) leaks no test item", fewshot_excludes_test)

    def test_run_auto_versions():
        path = work_dir / "attempts" / "predictions.json"
        first, attempt_one = save_test_run(predictions, path)
        first_bytes = first.read_bytes()
        second, attempt_two = save_test_run(predictions, path)
        assert attempt_one == 1 and attempt_two == 2, "attempts must count up"
        assert second != first, "a second run must not land on the first one's path"
        assert first.read_bytes() == first_bytes, "attempt 1 must be left untouched"
    check("save_test_run never overwrites - it versions", test_run_auto_versions)

    def test_log_appends():
        log_path = work_dir / "test_log.jsonl"
        for attempt in (1, 2):
            record_test_scoring(log_path, macro_f1=0.5, attempt=attempt,
                                pred_path=work_dir / "predictions.json",
                                prompt=PROMPT, prompt_file="prompts/track.txt",
                                gold_items=gold, dev_f1=0.6,
                                predictions=predictions)
        table = read_test_log(log_path)
        assert len(table) == 2, "every scoring appends one row, and none replaces another"
        assert table["prompt_sha1"].nunique() == 1, \
            "the same prompt must fingerprint the same both times"
    check("record_test_scoring appends, read_test_log reads it back", test_log_appends)

    def export_both_forms():
        # The form that existed before the split - it must keep working untouched.
        written = export_results("track", gold, predictions, {"round0": 0.5},
                                 work_dir / "export_plain", group="g", run="v1")
        assert written["gold"].name.endswith("_gold.json")
        # And the split-aware form, which reports the test half and names it as such.
        dev, test = split_dev_test(gold, dev_per_class=1)
        test_predictions = predictions[:len(test)]
        with_dev = export_results("track", test, test_predictions, {"round0": 0.5},
                                  work_dir / "export_split", group="g", run="v1",
                                  dev=dev)
        assert with_dev["gold"].name.endswith("_test.json"), \
            "with a split, the copy saved beside the results is the TEST half"
        report = with_dev["report"].read_text(encoding="utf-8")
        assert "held-out" in report, "the report must say which half the number is on"
    check("export_results(...) and export_results(..., dev=dev)", export_both_forms)

    def freeze_roundtrip():
        path = work_dir / "frozen.json"
        save_predictions(predictions, path)
        assert load_predictions(path) == predictions, \
            "a frozen run must load back identically"
    check("save_predictions -> load_predictions round-trip", freeze_roundtrip)

    def json_roundtrip():
        path = work_dir / "rounds.json"
        save_json({"round0 baseline": 0.61}, path, what="rounds")
        # Both call forms: the one taught in the tutorials, and the one notebook 05
        # uses now that a missing file names the notebook that writes it.
        assert load_json(path, what="rounds") == {"round0 baseline": 0.61}
        assert load_json(path, what="rounds", made_by="notebook 04_prompt") == \
            {"round0 baseline": 0.61}
    check("save_json -> load_json round-trip, both call forms", json_roundtrip)

    def missing_handoff_files_explain_themselves():
        # Arriving at notebook 05 without having finished 04 is ordinary, and the
        # message has to say which notebook writes the file rather than tracebacking
        # into open().
        for call in [lambda: load_predictions(work_dir / "not_written_yet.json"),
                     lambda: load_json(work_dir / "not_written_yet.json",
                                       what="rounds", made_by="notebook 04_prompt")]:
            try:
                call()
                raise AssertionError("a missing handoff file must raise")
            except FileNotFoundError as problem:
                assert "notebook 04_prompt" in str(problem), \
                    "the message must name the notebook that writes the file"
                assert "config.yaml" in str(problem), \
                    "the message must mention the run/group naming trap"
    check("a missing predictions or rounds file says which notebook makes it",
          missing_handoff_files_explain_themselves)

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

        from annotate import create_annotation_sheet, remembered_sheet

        def sheet_creation_stays_three_args():
            # Day 2 S5 taught create_annotation_sheet(title, items, labels). Sharing and
            # remembering were added afterwards; if either ever loses its default, that
            # three-argument call breaks in the middle of the annotation session.
            import inspect
            parameters = inspect.signature(create_annotation_sheet).parameters
            required = [name for name, p in parameters.items()
                        if p.default is inspect.Parameter.empty]
            assert required == ["title", "items", "labels"], \
                "create_annotation_sheet grew a required argument: " + str(required)
        check("create_annotation_sheet(title, items, labels) still needs only three",
              sheet_creation_stays_three_args)

        def coder_tabs_merge_into_one_table():
            # load_coder_sheets reads one TAB per coder and joins them by id. Stub the
            # per-tab reader so this needs no network and no Google account: what is
            # being checked is the JOIN, and that its output is the shape everything
            # downstream already consumes.
            import annotate
            tabs = {
                "CoderA": [{"ID": 1, "Text": "first", "Label": "A1", "Note": ""},
                           {"ID": 2, "Text": "second", "Label": "A2", "Note": ""}],
                "CoderB": [{"ID": 1, "Text": "first", "Label": "A1", "Note": ""},
                           {"ID": 2, "Text": "second", "Label": "B1", "Note": "unsure"}],
                "Final": [{"ID": 1, "Text": "first", "Final": "A1", "Note": ""},
                          {"ID": 2, "Text": "second", "Final": "A2", "Note": "talked"}],
            }
            real_loader = annotate.load_annotation_sheet
            real_tab_names = annotate.tab_names
            annotate.load_annotation_sheet = lambda sheet_id, tab: tabs[tab]
            annotate.tab_names = lambda sheet_id: list(tabs)
            try:
                merged = annotate.load_coder_sheets("ignored", ["CoderA", "CoderB"])
            finally:
                annotate.load_annotation_sheet = real_loader
                annotate.tab_names = real_tab_names

            assert len(merged) == 2, "one row per item, not one per tab"
            assert merged[0]["CoderA"] == "A1" and merged[0]["CoderB"] == "A1", \
                "each coder's tab becomes a column named after that coder"
            assert merged[1]["Final"] == "A2", "the Final tab supplies the Final column"
            # The shape to_canonical and disagreements already expect.
            gold = to_canonical(merged, labels)
            assert len(gold) == 2, "merged rows must still canonicalise"
            assert len(disagreements(merged)) == 1, "row 2 disagrees"
        check("load_coder_sheets merges per-coder tabs into the usual row shape",
              coder_tabs_merge_into_one_table)

        def duplicated_tab_is_called_out():
            # Adding a coder by duplicating a tab that is already filled in hands them
            # somebody else's answers. Perfect agreement then reads as excellent
            # reliability instead of as a photocopy, and nothing downstream can tell.
            import annotate
            same = []
            for number in range(6):
                same.append({"ID": number + 1, "Text": "t", "Label": "A1", "Note": ""})
            tabs = {"CoderA": same, "CoderB": same, "Final": []}
            real_loader = annotate.load_annotation_sheet
            real_tab_names = annotate.tab_names
            annotate.load_annotation_sheet = lambda sheet_id, tab: tabs[tab]
            annotate.tab_names = lambda sheet_id: list(tabs)
            printed = []
            real_print = builtins.print
            builtins.print = lambda *args, **kwargs: printed.append(" ".join(
                str(a) for a in args))
            try:
                annotate.load_coder_sheets("ignored", ["CoderA", "CoderB"])
            finally:
                annotate.load_annotation_sheet = real_loader
                annotate.tab_names = real_tab_names
                builtins.print = real_print
            assert any("WARNING" in line for line in printed), \
                "two identical coder columns must be called out, not passed through"
        check("load_coder_sheets warns when one coder's tab is a copy of another",
              duplicated_tab_is_called_out)

        def sheet_id_round_trip():
            # The sheet id is the one handoff that is not a data file. If this stops
            # round-tripping, notebook 03 step 3 silently gets "" and reads no sheet.
            path = work_dir / "sheet.json"
            assert remembered_sheet(path) == "", \
                "a sheet that was never created must come back as empty, not raise"
            url = "https://docs.google.com/spreadsheets/d/abc123/edit"
            path.write_text(json.dumps({"url": url}), encoding="utf-8")
            assert remembered_sheet(path) == url, "the saved URL did not come back"
        check("remembered_sheet(path) round-trips, and tolerates no file yet",
              sheet_id_round_trip)

        def sheet_agreement():
            result = annotator_agreement(rows)
            assert "kappa" in result
        check("annotator_agreement(rows) -> dict", sheet_agreement)

        def sheet_disagreements():
            table = disagreements(rows)
            assert len(table) == 1, "exactly one row disagrees"
        check("disagreements(rows) -> DataFrame", sheet_disagreements)

        # ---- three or more coders -------------------------------------------------
        # Each coder gets their own TAB, and how many there are is decided when the
        # sheet is READ, not when it is made. Two coders must keep behaving exactly as
        # before; three must not fall back to comparing only the first two.
        three = [
            {"ID": "1", "Text": "first", "CoderA": "A1", "CoderB": "A1",
             "CoderC": "A1", "Final": "", "Note": ""},
            {"ID": "2", "Text": "second", "CoderA": "A2", "CoderB": "B1",
             "CoderC": "A2", "Final": "", "Note": ""},
            {"ID": "3", "Text": "third", "CoderA": "B1", "CoderB": "B1",
             "CoderC": "A1", "Final": "", "Note": ""},
            {"ID": "4", "Text": "fourth", "CoderA": "A1", "CoderB": "",
             "CoderC": "A1", "Final": "", "Note": ""},   # half-finished: must drop out
        ]
        names = ["CoderA", "CoderB", "CoderC"]

        def agreement_with_three_coders():
            result = annotator_agreement(three, coders=names)
            assert result["n"] == 3, \
                "the row one coder skipped must drop out, leaving 3"
            assert "kappa" in result, "the 'kappa' key must survive for older code"
            assert "fleiss_kappa" in result, "three coders get a Fleiss kappa"
            assert len(result["pairwise_kappa"]) == 3, \
                "three coders make three pairs"
        check("annotator_agreement(rows, coders=[A, B, C]) -> Fleiss + pairwise",
              agreement_with_three_coders)

        def two_named_coders_behave_as_before():
            named = annotator_agreement(rows, coders=["CoderA", "CoderB"])
            plain = annotator_agreement(rows)
            assert named["kappa"] == plain["kappa"], \
                "naming the two coders must not change the number"
        check("annotator_agreement(rows, coders=[A, B]) == annotator_agreement(rows)",
              two_named_coders_behave_as_before)

        def disagreements_with_three_coders():
            table = disagreements(three, coders=names)
            assert len(table) == 2, \
                "rows 2 and 3 are not unanimous; row 4 is half-finished"
        check("disagreements(rows, coders=[A, B, C]) -> not-unanimous rows",
              disagreements_with_three_coders)

        def fleiss_matches_the_published_example():
            # Fleiss (1971): 10 subjects, 14 raters, 5 categories, kappa = 0.210.
            from annotate import fleiss_kappa
            counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6],
                      [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0],
                      [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0],
                      [0, 2, 2, 3, 7]]
            raters = []
            for _ in range(14):
                raters.append([])
            for row in counts:
                seat = 0
                for category in range(len(row)):
                    for _ in range(row[category]):
                        raters[seat].append(str(category))
                        seat = seat + 1
            kappa = fleiss_kappa(raters)
            assert abs(kappa - 0.210) < 0.001, \
                "Fleiss' kappa should be 0.210 on the published example, got " + str(kappa)
        check("fleiss_kappa reproduces the published worked example",
              fleiss_matches_the_published_example)

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
