"""answers.py — one worked version of each cell you are asked to write yourself.

Most of this project is assembling calls you have already met. A few cells ask you to
write something instead, because the thing being written IS the decision: what counts as
a disagreement, which statistics your design owes. There is no way to hand those over in
advance without answering them for you.

So they are here instead, and the order matters: **try it first, then look.** A group
that reads this before writing gets a working notebook and learns nothing the Q&A will
credit; a group that writes something, compares, and can say why theirs differs has done
the actual work. Your instructor will tell you when to open it.

It is not imported by the SETUP cell. To use it:

    from answers import answer

    answer()                    # what references are available
    answer("agreement")         # the assembled cell for 03 step 2
    answer("disagreements")     # the function 03 step 3 asks you to write

Some of what it prints is one right answer among several. Where that is true it says so,
and says what the alternatives turn on. `answer("agreement")` reports Cohen's kappa
because most groups have two coders and unordered labels; if yours are a scale, the
weighted kappa belongs there too and your cell is not wrong for having it.
"""

import inspect

import _study


# Cells whose reference is the ASSEMBLED CALLS rather than a function. Each is the code,
# then what the choices in it turned on - because the code alone reads as the only
# possible answer, and for most of these it is not.
COMPOSITIONS = {

    "agreement": (
        """print("Cohen's kappa:", round(cohen_kappa_score(a_labels, b_labels), 3))

matrix = confusion_matrix(a_labels, b_labels, labels=LABELS)
plot_confusion_matrix(matrix, LABELS, CODERS[0] + " vs " + CODERS[1])""",

        """This is step 2. Step 1 already ran percent_agreement and left `a_labels` and
`b_labels` behind, so what is left is the chance-corrected number and the matrix.

Both numbers, not one. Percent agreement on its own counts the agreement you would get
by guessing as though you had earned it - two coders who use one label for nine items in
ten agree 90% of the time without reading anything. Kappa on its own is hard to
interpret without the raw figure beside it.

Which kappa is not a free choice, and it is settled before you run anything:

  two coders, labels with no order   cohen_kappa_score(a, b)
  two coders, labels on a scale      cohen_kappa_score(a, b, weights="quadratic")
  three or more coders               fleiss_kappa(...), plus Cohen's for each pair

One thing the two numbers do NOT share: percent_agreement skips rows either coder left
blank, and cohen_kappa_score does not. Read percent_agreement in the cell above - the
`if` in the middle is the difference. On a finished sheet they agree; on a half-finished
one they are computed over different rows, and only one of them says so.

The matrix is not decoration. The two numbers say how far apart you were; only the
off-diagonal cells say WHICH pair of labels you disagree about, and that pair is what
step 3 sends you back to the sheet to argue about."""),

    "split": (
        """dev, test = split_dev_test(gold, DEV, seed=SEED)

save_json(dev,  DEV_PATH,  what="dev items")
save_json(test, TEST_PATH, what="test items")""",

        """Read the per-label counts it prints BEFORE saving. A label that lands in dev but
not in test cannot appear in the score you report, and this is the last easy moment to
change `dev:` in config.yaml and draw the line again.

If you drew your sample with sample_by_document, add by_document=True, or a passage will
have some of its sentences in dev and the rest in test - and a model that has seen half
a paragraph is not being tested on the other half."""),
}


# Cells whose reference is a FUNCTION. Printed from _study.py rather than copied here,
# so there is one implementation and this file cannot drift from the one that runs.
FUNCTIONS = {
    "disagreements": (
        _study.disagreements,
        """The rule inside it is the decision, and the obvious rule is not the only
defensible one. This version calls a row a disagreement when the coders did not all
choose the same label.

On labels that sit on a scale, you may decide that neighbouring labels are close enough
to leave alone - that A2 against B1 is two people reading the same sentence much the
same way, and only a gap of two or more is worth an argument. That version finds fewer
rows and sends you to the sheet with a shorter list.

Either is defensible. Which you chose, and why, is a sentence in your report. What is
not defensible is not knowing which one you used."""),

    "percent_agreement": (
        _study.percent_agreement,
        """Two things worth noticing. It skips rows where either coder left the cell
blank, rather than counting them as disagreements - an unfinished row is not evidence
about your scheme. And it is written out rather than taken from sklearn's
accuracy_score, which computes the same fraction: "accuracy" names one of the two lists
as correct, and when the two are coders neither of them is."""),
}


def answer(name: str = "") -> None:
    """Print one worked version of a cell you were asked to write.

    Args:
        name: which one. Left out, it lists what is available.

    Returns:
        Nothing. It prints the code and what the choices in it turned on.

    Example:
        >>> answer("disagreements")
    """
    if not name:
        _list_them()
        return None

    if name in COMPOSITIONS:
        code, why = COMPOSITIONS[name]
        _print_block(name, code, why)
        return None

    if name in FUNCTIONS:
        function, why = FUNCTIONS[name]
        _print_block(name, inspect.getsource(function).rstrip("\n"), why)
        return None

    print("There is no reference called " + repr(name) + ".")
    _list_them()
    return None


def _list_them() -> None:
    """Print the names `answer` accepts.

    Returns:
        Nothing. It prints one line per available reference.
    """
    print("Available:")
    for key in sorted(list(COMPOSITIONS) + list(FUNCTIONS)):
        print("  answer(" + repr(key) + ")")
    return None


def _print_block(name: str, code: str, why: str) -> None:
    """Print one reference: a heading, the code, then what it turned on.

    Args:
        name: the reference's name, for the heading.
        code: the code to show.
        why: the explanation printed under it.

    Returns:
        Nothing.
    """
    rule = "─" * 74
    print(rule)
    print("answer(" + repr(name) + ") — one worked version. Yours may differ and be right.")
    print(rule)
    print(code)
    print(rule)
    print(why.strip())
    print(rule)
    return None
