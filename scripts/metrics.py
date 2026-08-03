"""metrics.py — scoring the model's predictions.

For the final project we use scikit-learn's battle-tested implementations of the
evaluation metrics. (In the Day-2 tutorial and the Corpus Lab you coded these
formulas by hand — precision, recall, F1, Cohen's kappa — so you know exactly what
these functions compute. Here we trust the library and get on with the analysis.)

You do NOT need to edit this file.

The call forms are the same ones you used in the tutorials:

    evaluate(gold, predictions)                 # Day 2 S6
    evaluate(gold, predictions, ordered=True)   # Day 2 S6 / Day 3, ordered labels
    show_errors(gold, predictions)              # Day 3

One difference worth knowing: `evaluate` here also RETURNS the macro-F1 as a number,
so you can collect it round by round (`f1_by_round[...] = evaluate(...)`). It still
prints everything it printed in the tutorials.

(This file used to be called evaluate.py. It was renamed because `from evaluate
import *` made the module name and the function name collide.)
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)

import pandas as pd

from pipeline import plot_confusion_matrix, label_set

# show_errors and labels_of live in _study.py, the one copy the notebook generators
# render into cells. Re-exported here so that `from metrics import show_errors` - the
# Day 3 call - keeps working unchanged.
from _study import show_errors, labels_of


def evaluate(gold: list[dict[str, str]],
             predictions: list[str],
             ordered: bool = False,
             labels: list[str] | None = None,
             title: str = "Confusion matrix") -> float:
    """Score predictions against gold: per-class P/R/F1 + macro, Cohen's kappa, and a
    confusion-matrix heatmap. Returns the macro-F1 as a number.

    ordered=True adds QUADRATIC WEIGHTED kappa — use it only when the labels sit on a
    scale (A1 < A2 < ... < C2), so that a near miss counts as a smaller error than a
    far one. For unordered categories, plain kappa is the one to report.

    IMPORTANT for ordered=True: the scale is taken from `labels`, in the order given.
    Left off, `labels` is read off the gold set and sorted ALPHABETICALLY — which is
    correct for A1..C2 and Move 1..3, but wrong for something like Low/Mid/High
    (alphabetical puts High first). If your labels are ordered and not alphabetical,
    pass them yourself: evaluate(gold, pred, ordered=True, labels=LABELS_ORDER).

    Args:
        gold: the gold items, each with a "label" key.
        predictions: one predicted label per gold item, in the same order.
        ordered: True when the labels sit on a scale.
        labels: the labels to score, in scale order. Left out, they are read off the
            gold set and sorted alphabetically.
        title: the heading to put on the confusion matrix.

    Returns:
        The macro-F1, so you can collect it round by round.

    Example:
        >>> f1_by_round["1 zero-shot"] = evaluate(gold, predictions, ordered=True)
    """
    ### Step 1: line the two label lists up, gold first ###
    y_true = []                          # the correct labels, from the gold set
    for item in gold:
        y_true.append(item["label"])
    y_pred = predictions                 # the model's labels, in the same order

    if labels is None:
        labels = label_set(gold)

    ### Step 2: per-class precision / recall / F1, as a text table ###
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    ### Step 3: one overall number — agreement corrected for chance ###
    # Only meaningful if there is more than one label to be right or wrong about.
    if len(set(labels)) < 2:
        print("Cohen's kappa            undefined (only one label present)")
    else:
        print(f"Cohen's kappa            {cohen_kappa_score(y_true, y_pred):.3f}")
        if ordered:                      # only when the labels sit on a scale
            weighted = cohen_kappa_score(y_true, y_pred, labels=labels,
                                         weights="quadratic")   # near misses hurt less
            print(f"Cohen's kappa (weighted) {weighted:.3f}   <- labels are ordered")
            # Say WHICH order we used, so a wrong one is visible rather than silent.
            print("  scale order used:", " < ".join(labels))

    ### Step 4: draw the same information as a picture ###
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(matrix, labels, title)

    ### Step 5: one number to carry from round to round ###
    macro_f1 = f1_score(y_true, y_pred, labels=labels,
                        average="macro", zero_division=0)
    return macro_f1


def agreement(labels_a: list[str], labels_b: list[str]) -> dict[str, float]:
    """Compare two annotators: percent agreement and Cohen's kappa.

    Takes two plain lists of labels, in the same item order. (If your labels are
    still sitting in a Google Sheet, use annotator_agreement() from annotate.py
    instead — it pulls the two columns out for you and also draws the matrix.)

    Args:
        labels_a: the first annotator's labels.
        labels_b: the second annotator's labels, for the same items in the same order.

    Returns:
        {"percent_agreement", "kappa", "cohen_kappa"} — the last two are the same
        number under two names. Kappa is nan when only one label was used.

    Raises:
        ValueError: when the two lists are different lengths, which would pair the
            wrong sentences together.

    Example:
        >>> agreement(coder_a_labels, coder_b_labels)
    """
    a = list(labels_a)
    b = list(labels_b)
    if len(a) != len(b):
        raise ValueError(
            "The two lists of labels are different lengths: the first has "
            + str(len(a)) + " and the second has " + str(len(b)) + ". They have to line "
            "up item by item, or the comparison pairs the wrong sentences together.\n"
            "Most often this means one coder left rows blank at the bottom of their "
            "tab. Open your annotation Sheet, fill in the missing rows, and run the "
            "cell again.")

    # Count the positions where the two annotators chose the same label.
    number_of_matches = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            number_of_matches = number_of_matches + 1
    percent_agreement = number_of_matches / len(a)

    # Kappa needs at least two distinct labels across the two annotators; with only
    # one, sklearn returns nan and a warning, which reads like a bug rather than a
    # property of the data.
    all_labels_used = set(a) | set(b)
    if len(all_labels_used) < 2:
        print("Percent agreement:", format(percent_agreement, ".1%"),
              "  Cohen's kappa: undefined (both annotators used only one label)")
        kappa = float("nan")
    else:
        kappa = cohen_kappa_score(a, b)
        print("Percent agreement:", format(percent_agreement, ".1%"),
              "  Cohen's kappa:", format(kappa, ".3f"))
    # "kappa" and "cohen_kappa" are the same number under two names, so code written
    # against either the tutorials or this file keeps working.
    return {"percent_agreement": percent_agreement,
            "kappa": kappa,
            "cohen_kappa": kappa}


def confused_pairs(errors: pd.DataFrame) -> pd.DataFrame:
    """Which gold -> predicted swaps the model made most often.

    The same reading you made of the coder-vs-coder confusion matrix in notebook 03,
    made here of the model. One row per label pair, commonest first, so the pattern in
    a long error table is one line rather than something you have to spot by eye.

    Args:
        errors: the table from show_errors, with `gold` and `pred` columns.

    Returns:
        A table of `gold`, `pred` and `n`, sorted commonest first.

    Example:
        >>> confused_pairs(errors)
    """
    counts = {}
    for _, row in errors.iterrows():
        pair = (str(row["gold"]), str(row["pred"]))
        counts[pair] = counts.get(pair, 0) + 1

    out = []
    for pair in counts:
        out.append({"gold": pair[0], "pred": pair[1], "n": counts[pair]})
    out.sort(key=lambda entry: -entry["n"])

    if out:
        top = out[0]
        print("Most confused: " + top["gold"] + " -> " + top["pred"],
              "(" + str(top["n"]) + " of " + str(len(errors)) + " errors)")
    else:
        print("No errors to read.")
    return pd.DataFrame(out, columns=["gold", "pred", "n"])


def errors_on_disagreed(errors: pd.DataFrame,
                        disagreed: pd.DataFrame | list[dict[str, str]]) -> list[int]:
    """How many of the model's errors land on items YOUR OWN coders disagreed about.

    This is the most interesting number in the project. If the model's misses cluster
    on the items CoderA and CoderB could not agree on either, then what you have
    measured is a fuzzy boundary in the annotation scheme, not a stupid model - and
    that is a better finding than a clean F1.

    Both tables carry the same ids: the sheet was built from the sampled items, and
    the gold set was rebuilt from the sheet, so nothing has been renumbered in between.
    The sheet returns its ID column as text, though, so it is converted here.

    Args:
        errors: the table from show_errors, with an "id" column.
        disagreed: the table from disagreements, with an "ID" column. Notebook 03
            saves it to a file and notebook 05 loads it back as a list of dicts, so
            either form is accepted.

    Returns:
        The ids that appear in both, sorted. Empty when either table is empty.

    Example:
        >>> errors_on_disagreed(errors, disagreed)
    """
    # Both of these tables are empty in the HAPPY case - no model errors, or no coder
    # disagreements - and an empty table has no columns at all, so reading errors["id"]
    # off one would fail. Check before reading, not after.
    if errors is None or len(errors) == 0:
        print("The model got every item right, so there are no errors to compare.")
        return []
    if disagreed is None or len(disagreed) == 0:
        print("Your coders agreed on every item, so there is nothing to compare the",
              "model's errors against.")
        return []

    # Notebook 03 saves this table to a file and notebook 05 loads it back, which makes
    # it a plain list of dicts by the time it arrives here rather than the DataFrame
    # `disagreements` handed back. Accept either.
    if not isinstance(disagreed, pd.DataFrame):
        disagreed = pd.DataFrame(list(disagreed))

    error_ids = []
    for value in errors["id"]:
        error_ids.append(int(value))

    disagreed_ids = []
    for value in disagreed["ID"]:
        try:
            disagreed_ids.append(int(value))
        except (TypeError, ValueError):
            # A typed-over ID cell. to_canonical reports these too; skip it here
            # rather than lose the whole comparison over one bad row.
            pass

    overlap = []
    for item_id in error_ids:
        if item_id in disagreed_ids and item_id not in overlap:
            overlap.append(item_id)
    overlap.sort()

    share = len(overlap) / len(error_ids)
    print(len(error_ids), "errors.", len(overlap), "of them",
          "(" + format(share, ".0%") + ")",
          "are on items your two coders also disagreed about.")
    if overlap:
        print("  those ids:", overlap)
        print("  Read those items again before you blame the model.")
    return overlap
