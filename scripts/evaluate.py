"""evaluate.py — scoring the model's predictions.

For the final project we use scikit-learn's battle-tested implementations of the
evaluation metrics. (In the Day-2 tutorial and the Corpus Lab you coded these
formulas by hand — precision, recall, F1, Cohen's kappa — so you know exactly what
these functions compute. Here we trust the library and get on with the analysis.)

You do NOT need to edit this file.
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)

import pandas as pd

from pipeline import plot_confusion_matrix


def evaluate(gold, predictions, labels, title):
    """Print precision/recall/F1, draw a confusion matrix, return the macro-F1 score."""
    # The true labels come from the gold set, in the same order as the predictions.
    true_labels = []
    for item in gold:
        true_labels.append(item["label"])
    predicted_labels = predictions

    # Per-class precision / recall / F1 as a text table.
    print(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0))

    # Confusion matrix (rows = gold, columns = predicted), drawn by pipeline.py.
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plot_confusion_matrix(matrix, labels, title)

    # One overall number: the macro-averaged F1 (every class counts equally).
    macro_f1 = f1_score(true_labels, predicted_labels, labels=labels,
                        average="macro", zero_division=0)
    return macro_f1


def agreement(labels_a, labels_b):
    """Compare two annotators: percent agreement and Cohen's kappa."""
    a = list(labels_a)
    b = list(labels_b)
    assert len(a) == len(b), "the two label lists must be the same length"

    # Count the positions where the two annotators chose the same label.
    number_of_matches = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            number_of_matches = number_of_matches + 1
    percent_agreement = number_of_matches / len(a)

    kappa = cohen_kappa_score(a, b)
    print("Percent agreement:", format(percent_agreement, ".1%"),
          "  Cohen's kappa:", format(kappa, ".3f"))
    return {"percent_agreement": percent_agreement, "cohen_kappa": kappa}


def show_errors(gold, predictions):
    """Return a table of the items the model got wrong."""
    wrong_rows = []
    for item, predicted in zip(gold, predictions):
        if item["label"] != predicted:
            row = {
                "id": item["id"],
                "gold": item["label"],
                "pred": predicted,
                "text": item["text"],
            }
            wrong_rows.append(row)
    print(len(wrong_rows), "of", len(gold), "were wrong.")
    return pd.DataFrame(wrong_rows)
