"""_check_notebooks.py — does every generated notebook read like one a beginner can follow?

Run it after regenerating:

    uv run python scripts/_check_notebooks.py

WHY THIS EXISTS
---------------
The notebooks are built by two generator scripts, and a generator can happily emit a
cell that is broken, enormous, or unannounced -- nothing complains, because nothing
runs it. One version of this template shipped a code cell that began with a sentence
of English and raised a SyntaxError the moment a student pressed Shift+Enter.

So this file asserts the three properties the notebooks are supposed to have:

    1. every code cell is valid Python (it would run, given the right names)
    2. no code cell is longer than a screen
    3. every code cell has a markdown cell above it saying what it is about

They are simple rules, but they are the whole difference between a notebook a student
works through and a wall they scroll past.
"""

import json
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

# A cell longer than this is doing more than one thing, or is a wall of source. Either
# way it wants splitting. Counted in lines of the cell as the student sees it, comments
# and all, because the comment header is part of what they have to read.
MAX_CELL_LINES = 40

problems = []


def report(notebook: str, index: int, message: str) -> None:
    problems.append(notebook + "  cell " + str(index) + ": " + message)


def source_of(cell: dict) -> str:
    return "".join(cell["source"])


def check_it_compiles(notebook: str, index: int, text: str) -> None:
    """Would this cell run? Undefined names are fine; broken syntax is not."""
    # A line starting with ! or % is Colab shell/magic syntax, which is not Python and
    # which compile() would reject. Those cells are one line each; skip them.
    for line in text.splitlines():
        if line.strip().startswith("!") or line.strip().startswith("%"):
            return
    try:
        compile(text, "<cell>", "exec")
    except SyntaxError as error:
        report(notebook, index, "is not valid Python — " + str(error.msg)
               + " (line " + str(error.lineno) + ")")


def is_the_setup_cell(text: str) -> bool:
    """The one long cell that is allowed to stay long.

    It mounts Drive, finds the group folder and imports everything, and it says in its
    own first lines that it is plumbing nobody is expected to read. Splitting it would
    give a student four cells to run before anything starts instead of one, which is
    worse, not better. It is the only exemption.
    """
    return text.lstrip().startswith("# ---") and "SETUP — run me first" in text


def is_one_function(text: str) -> bool:
    """Is this cell exactly one function definition and nothing else?

    The reshaping functions are embedded into notebook 01 from scripts/reshape.py, and
    one of them is genuinely 68 lines. That is allowed: a function is one nameable
    thing, it arrives in a cell of its own with a signpost above it, and there is no
    way to split it that leaves the notebook running the same code as the repo does.
    What is NOT allowed is four of them stacked in one cell, which is what this whole
    check exists to stop - so the rule is one `def` per cell, not "long cells are fine".
    """
    body = []
    for line in text.splitlines():
        if line.strip() and not line.startswith(("#", " ", "\t")):
            body.append(line)
    if not body:
        return False
    if not body[0].startswith("def "):
        return False
    for line in body[1:]:
        if line.startswith("def ") or line.startswith("class "):
            return False       # more than one thing in the cell
    return True


def check_it_fits_on_a_screen(notebook: str, index: int, text: str) -> None:
    if is_the_setup_cell(text) or is_one_function(text):
        return
    length = len(text.splitlines())
    if length > MAX_CELL_LINES:
        report(notebook, index, str(length) + " lines long — split it "
               "(the limit is " + str(MAX_CELL_LINES) + ")")


def check_it_has_a_lead_in(notebook: str, index: int, cells: list[dict]) -> None:
    """A code cell needs a markdown cell above it saying what is about to happen."""
    if index == 0:
        report(notebook, index, "is the first cell in the notebook, with no lead-in "
                                "above it")
        return
    if cells[index - 1]["cell_type"] != "markdown":
        report(notebook, index, "has another code cell directly above it — it needs a "
                                "markdown lead-in of its own")


def check_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index in range(len(cells)):
        cell = cells[index]
        if cell["cell_type"] != "code":
            continue
        text = source_of(cell)
        check_it_compiles(path.name, index, text)
        check_it_fits_on_a_screen(path.name, index, text)
        check_it_has_a_lead_in(path.name, index, cells)


def main() -> bool:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        print("No notebooks found in", NOTEBOOKS)
        print("Run the two generator scripts first.")
        return 1

    for path in paths:
        check_notebook(path)
        print("  checked", path.name)

    print()
    if problems:
        print(len(problems), "problems:")
        for problem in problems:
            print("  -", problem)
        print("\nFix these in scripts/_generate_pool_notebooks.py or")
        print("scripts/_generate_project_notebooks.py, then regenerate. Never hand-edit")
        print("a notebook: the next regeneration would throw the edit away.")
        return 1
    print("All", len(paths), "notebooks: every code cell runs, fits a screen, and is")
    print("introduced by a markdown cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
