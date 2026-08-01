"""_check_study_source.py — the two copies of _study.py must not drift.

`_study.py` holds the small pieces of the method that end up printed into notebook
cells: `column`, `percent_agreement`, `disagreements`, `show_errors`, `labels_of`. Both
repositories keep a copy, because each has to build on its own:

    lda2-final-template/scripts/_study.py                  rendered into 03 and 06
    linguistic-data-analysis-II-2026/sources/notebooks/_study.py   rendered into the day notebooks

Two copies is the price of two repositories that ship separately. What we refuse to pay
is the usual consequence. Before this file existed, `annotator_agreement`,
`disagreements`, `to_canonical`, `evaluate` and `show_errors` were maintained by hand in
both places and had already drifted: the tutorial's `annotator_agreement` took two
arguments and the template's took four, and nothing anywhere noticed. A student would
have met one function on Day 2 and a different one with the same name on Day 5.

So: same bytes, or the build stops.

Run it:

    python scripts/_check_study_source.py

If the course repository is not checked out beside this one there is nothing to compare,
and it says so and passes - cloning only the template is a normal thing to do.
"""

import os
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent
OURS = HERE / "_study.py"

# Where the other copy usually is: the course repository, checked out beside this one.
# Set LDA2_COURSE_REPO to point somewhere else.
DEFAULT_COURSE_REPO = HERE.parent.parent / "linguistic-data-analysis-II-2026"
THEIR_PATH_IN_REPO = pathlib.Path("sources") / "notebooks" / "_study.py"


def course_copy() -> pathlib.Path | None:
    """Find the course repository's copy of _study.py.

    Returns:
        The path to it, or None when the course repository is not checked out here.
    """
    override = os.environ.get("LDA2_COURSE_REPO")
    if override:
        candidate = pathlib.Path(override) / THEIR_PATH_IN_REPO
    else:
        candidate = DEFAULT_COURSE_REPO / THEIR_PATH_IN_REPO
    if candidate.is_file():
        return candidate
    return None


def first_difference(ours: str, theirs: str) -> str:
    """The first line that differs, as a message naming the line number.

    Args:
        ours: this repository's copy.
        theirs: the course repository's copy.

    Returns:
        A description of where they part company.
    """
    our_lines = ours.split("\n")
    their_lines = theirs.split("\n")
    for number in range(max(len(our_lines), len(their_lines))):
        one = our_lines[number] if number < len(our_lines) else "(end of file)"
        two = their_lines[number] if number < len(their_lines) else "(end of file)"
        if one != two:
            return ("First difference at line " + str(number + 1) + ":\n"
                    "  template: " + one + "\n"
                    "  course:   " + two)
    return "The files differ in trailing whitespace only."


def main() -> bool:
    """Compare the two copies.

    Returns:
        True when they match, or when there is nothing to compare.
    """
    theirs_path = course_copy()
    if theirs_path is None:
        print("The course repository is not checked out beside this one, so there is")
        print("no second copy of _study.py to compare against. Nothing to check.")
        return True

    ours = OURS.read_text(encoding="utf-8")
    theirs = theirs_path.read_text(encoding="utf-8")

    if ours == theirs:
        print("_study.py matches in both repositories.")
        return True

    print("The two copies of _study.py have drifted apart.")
    print()
    print("  template: " + str(OURS))
    print("  course:   " + str(theirs_path))
    print()
    print(first_difference(ours, theirs))
    print()
    print("Copy whichever one is right over the other, then run this again:")
    print("  cp " + str(OURS) + " " + str(theirs_path))
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
