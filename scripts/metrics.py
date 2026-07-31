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

from pipeline import (plot_confusion_matrix, label_set, triage_category,
                      TRIAGE_CATEGORIES)


def evaluate(gold, predictions, ordered=False, labels=None, title="Confusion matrix"):
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
    """
    # --- Compatibility with the older 4-positional call form -----------------------
    # An earlier version of this file took evaluate(gold, predictions, labels, title).
    # If we were called that way, argument 3 is a list of labels rather than a
    # true/false flag. Rather than fail with a confusing error - or worse, silently
    # treat a non-empty list as "ordered=True" - detect it and shuffle the arguments.
    if isinstance(ordered, (list, tuple)):
        print("NOTE: old call form evaluate(gold, pred, labels, title) — treating "
              "argument 3 as labels. The current form is "
              "evaluate(gold, pred, ordered=..., labels=...).")
        if isinstance(labels, str):
            title = labels
        labels = list(ordered)
        ordered = False

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


def agreement(labels_a, labels_b):
    """Compare two annotators: percent agreement and Cohen's kappa.

    Takes two plain lists of labels, in the same item order. (If your labels are
    still sitting in a Google Sheet, use annotator_agreement() from annotate.py
    instead — it pulls the two columns out for you and also draws the matrix.)
    """
    a = list(labels_a)
    b = list(labels_b)
    assert len(a) == len(b), "the two label lists must be the same length"

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


def triage_counts(triage, errors=None):
    """Count a triage by category, and say what it adds up to.

    `triage` is the dict your group writes by hand: {item id: "category - reason"}.
    The categories are fixed (model / scheme / wording / ambiguous) so that the counts
    mean the same thing across groups, and so this is a judgment you make from a menu
    rather than an essay you write.

    Pass `errors` (the table from show_errors) and it will also tell you how much of
    the error set you have actually been through.
    """
    counts = {}
    for category in TRIAGE_CATEGORIES:
        counts[category] = 0

    unrecognised = []
    for item_id in triage:
        category = triage_category(triage[item_id])
        if category is None:
            unrecognised.append(item_id)
        else:
            counts[category] = counts[category] + 1

    parts = []
    for category in TRIAGE_CATEGORIES:
        if counts[category] > 0:
            parts.append(str(counts[category]) + " " + category)
    if parts:
        print("Triaged " + str(len(triage)) + " errors: " + " / ".join(parts))
    else:
        print("Triaged 0 errors.")

    if unrecognised:
        print("NOTE: these lines do not start with one of",
              ", ".join(TRIAGE_CATEGORIES) + ":", unrecognised)
        print("      Start each line with the category word, then the reason:")
        print('        7: "scheme - Move 1/Move 2 boundary, our coders split too"')

    # "We looked at 3 of 40 errors" and "we looked at all 12" are different claims, and
    # the report should not let the first quietly read as the second.
    if errors is not None and hasattr(errors, "shape"):
        total = errors.shape[0]
        if len(triage) < total:
            print("You have triaged", len(triage), "of", total, "errors. Say so in the",
                  "report, or work through the rest.")
    return counts


def errors_on_disagreed(errors, disagreed):
    """How many of the model's errors land on items YOUR OWN coders disagreed about.

    This is the most interesting number in the project. If the model's misses cluster
    on the items CoderA and CoderB could not agree on either, then what you have
    measured is a fuzzy boundary in the annotation scheme, not a stupid model - and
    that is a better finding than a clean F1.

    Both tables carry the same ids: the sheet was built from the sampled items, and
    the gold set was rebuilt from the sheet, so nothing has been renumbered in between.
    The sheet returns its ID column as text, though, so it is converted here.
    """
    if errors is None or disagreed is None:
        print("Nothing to compare - one of the two tables is empty.")
        return []

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

    if len(error_ids) == 0:
        print("No errors to compare.")
        return []

    share = len(overlap) / len(error_ids)
    print(len(error_ids), "errors.", len(overlap), "of them",
          "(" + format(share, ".0%") + ")",
          "are on items your two coders also disagreed about.")
    if overlap:
        print("  those ids:", overlap)
        print("  Read those items again before you blame the model.")
    return overlap


def show_errors(gold, predictions):
    """The items the model got wrong, as a table you can read and argue about."""
    rows = []
    for item, predicted in zip(gold, predictions):
        if item["label"] != predicted:
            row = {
                "id": item["id"],
                "gold": item["label"],
                "pred": predicted,
                "text": item["text"],
            }
            rows.append(row)
    print(f"{len(rows)} of {len(gold)} wrong.")
    return pd.DataFrame(rows)             # a table, so Colab displays it nicely
