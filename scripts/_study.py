"""_study.py — the small pieces of the method, in one place.

These are the functions that carry a research judgment rather than plumbing: what
counts as a disagreement, how two coders' labels are lined up, which items the model
got wrong. They are short on purpose, because they are meant to be READ — every one of
them ends up printed into a notebook cell where a student can change it.

Nothing imports this file at run time from a notebook. It is a **build input**: the
notebook generators read the source out of it with `inspect.getsource` and render it
into cells. Both this repository and the course repository keep a copy, and
`scripts/_check_study_source.py` fails the build if the two ever stop matching. That
is why there is one file rather than two hand-maintained sets of the same code.

`annotate.py` and `metrics.py` re-export these, so `from annotate import
disagreements` and `from metrics import show_errors` keep working exactly as the
tutorials taught them.

Two rules govern what may go in here, and both exist to stop a student's edit being
silently ignored:

1. **Nothing in this file calls anything else in this file.** A student who copies
   `percent_agreement` into a cell and changes it gets their version wherever the
   NOTEBOOK calls it. If some other function here called it instead, that function
   would go on using this copy, and the student would report a number their own code
   did not produce. Composition happens in the notebook, not in here.
2. **Explicit loops, no comprehensions.** Whoever reads these is a term into Python at
   most. A one-line comprehension is not shorter to a reader who has to unpack it.
"""

import pandas as pd


# The annotation sheet's column names, used as the default coder columns below.
# annotate.py imports these rather than defining its own, so the sheet it writes and
# the columns these functions read cannot drift apart.
COL_ID, COL_TEXT = "ID", "Text"
COL_A, COL_B = "CoderA", "CoderB"
COL_FINAL, COL_NOTES = "Final", "Note"


def column(rows: list[dict[str, str]], name: str) -> list[str]:
    """Pull one coder's labels out of the merged sheet rows, in row order.

    The sheet arrives as one dict per item, with a column per coder. Almost everything
    you want to measure needs two plain lists lined up item by item instead, so this is
    usually the first line of an agreement cell.

    Rows where that coder left the cell blank come back as an empty string rather than
    being dropped, so two calls on the same rows stay the same length and stay aligned.

    Args:
        rows: the merged rows, one per item, as load_coder_sheets returns them.
        name: the column to read, e.g. "CoderA".

    Returns:
        That column's labels, one per row, in row order.

    Example:
        >>> a_labels = column(rows, "CoderA")
    """
    labels = []
    for row in rows:
        labels.append(str(row.get(name, "")).strip())
    return labels


def labels_of(items: list[dict[str, str]]) -> list[str]:
    """The "label" of every item, in order — the gold side of any scoring call.

    sklearn's metrics take two plain lists, so this is what turns your gold items into
    the first of them. The second is the predictions list, which is already in that
    shape.

    Args:
        items: gold items, each with a "label".

    Returns:
        One label per item, in the same order.

    Example:
        >>> y_true = labels_of(test)
    """
    labels = []
    for item in items:
        labels.append(item["label"])
    return labels


def percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """How often two coders chose the same label, counting only rows both labelled.

    This is the number to report next to a kappa, never instead of one: it counts the
    agreement you would get by chance as though you had earned it. Two coders using one
    label for nine items in ten agree 90% of the time without reading anything.

    It is written out rather than taken from sklearn's accuracy_score, which computes
    the same fraction. "Accuracy" names one of the two lists as correct, and when the
    two are coders neither of them is.

    Args:
        labels_a: the first coder's labels.
        labels_b: the second coder's labels, same items, same order.

    Returns:
        The proportion of doubly-labelled rows where the two match, 0.0 to 1.0.

    Raises:
        ValueError: when the two lists are different lengths, which would compare
            the wrong sentences with each other.

    Example:
        >>> percent_agreement(a_labels, b_labels)
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            "The two lists of labels are different lengths: the first has "
            + str(len(labels_a)) + " and the second has " + str(len(labels_b))
            + ". They have to line up item by item, or the comparison pairs the wrong "
            "sentences together.\nMost often one coder left rows blank at the bottom "
            "of their tab. Fill them in and run the cell again.")

    both_labelled = 0
    matched = 0
    for index in range(len(labels_a)):
        if labels_a[index] and labels_b[index]:      # skip rows nobody finished
            both_labelled = both_labelled + 1
            if labels_a[index] == labels_b[index]:
                matched = matched + 1

    if both_labelled == 0:
        print("No rows where both coders have labelled. Nothing to compare yet.")
        return 0.0

    share = matched / both_labelled
    print(both_labelled, "doubly-annotated ·", matched, "matched · agreement",
          format(share, ".1%"))
    return share


def disagreements(rows: list[dict[str, str]],
                  a: str = COL_A,
                  b: str = COL_B,
                  coders: list[str] | None = None) -> pd.DataFrame:
    """The rows your coders labelled differently — your adjudication list.

    A row counts as a disagreement when everyone labelled it and they did not all
    choose the same label. That rule is a decision, not a fact: on labels that sit on a
    scale, you may decide that neighbouring labels are close enough to leave alone, and
    that a disagreement means a gap of two or more. Changing this rule changes which
    rows you spend the afternoon arguing about.

    Args:
        rows: the merged rows, one per item.
        a: the column holding the first coder's labels.
        b: the column holding the second coder's labels.
        coders: for three or more coders, their names. A row needs adjudicating when
            they do not ALL agree. With two coders the behaviour is unchanged.

    Returns:
        A table of the rows to adjudicate. The columns are named even when your coders
        agreed on everything.

    Example:
        >>> disagreed = disagreements(rows, coders=CODERS)
    """
    if coders is None:
        names = [a, b]
    else:
        names = list(coders)

    out = []
    for row in rows:
        labels = []
        for name in names:
            labels.append(str(row.get(name, "")).strip())

        everyone_labelled = True
        for label in labels:
            if not label:
                everyone_labelled = False

        they_differ = len(set(labels)) > 1

        if everyone_labelled and they_differ:
            out.append(row)

    print(len(out), "rows to adjudicate. Agree on a `Final` label for each in the sheet.")
    # Name the columns even when no rows come back, so that a group whose coders agreed
    # on everything gets an empty table rather than a table with nothing in it at all.
    if rows:
        return pd.DataFrame(out, columns=list(rows[0]))
    return pd.DataFrame(out)


def show_errors(gold: list[dict[str, str]],
                predictions: list[str]) -> pd.DataFrame:
    """The items the model got wrong, as a table you can read and argue about.

    F1 tells you whether a round helped. Only this tells you what to change next.

    Args:
        gold: the gold items, each with "id", "text" and "label".
        predictions: one predicted label per gold item, in the same order.

    Returns:
        A table with one row per mistake: id, gold, pred, text. The columns are named
        even when there are no mistakes.

    Example:
        >>> errors = show_errors(gold, predictions)
    """
    rows = []
    for index in range(len(gold)):
        item = gold[index]
        predicted = predictions[index]
        if item["label"] != predicted:
            rows.append({"id": item["id"],
                         "gold": item["label"],
                         "pred": predicted,
                         "text": item["text"]})

    print(len(rows), "of", len(gold), "wrong.")
    # Name the columns even when there are no rows. A table built from an empty list
    # has no columns at all, and then errors["gold"] in the report notebook fails for
    # the one group whose model got everything right - the least deserving group to
    # break on.
    return pd.DataFrame(rows, columns=["id", "gold", "pred", "text"])
