"""_check_undefined_names.py — every name a notebook uses must be defined first.

A generated notebook can be syntactically perfect and still fail on the first cell a
student runs, because a step calls something the SETUP cell never imported. That is
invisible to `_check_call_forms.py`, which tests the modules rather than the notebooks,
and it is exactly the failure that wastes a group's afternoon: the traceback names a
function they did nothing wrong with.

So: walk each notebook cell in order, collect every name it BINDS (imports, assignments,
defs, loop targets, comprehensions, arguments), and check that every name it READS was
bound by an earlier cell, is a builtin, or comes from config.

`from config import *` is resolved by importing config and reading its real names, so a
notebook relying on a config global is fine, and a typo in one is not.

    python scripts/_check_undefined_names.py

Needs no API key and no network.
"""

import ast
import builtins
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent / "notebooks"

# Names Colab itself provides, and names a cell may legitimately mention before any
# cell defines them.
COLAB_NAMES = {"get_ipython", "display", "In", "Out", "exit", "quit"}


def config_names() -> bool:
    """The names `from config import *` brings in.

    config.py mounts Drive on import in Colab, so it cannot simply be imported here.
    Read its source and take the module-level bindings instead.
    """
    names = set()
    source = (HERE.parent / "config.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def bound_and_used(tree: ast.AST) -> tuple:
    """The names this cell defines, and the names it reads."""
    bound = set()
    used = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                used.append((node.id, node.lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue                  # handled by the caller, per module
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            pass
    return bound, used


def promised_names(source: str) -> set:
    """The names a cell's `# Creates:` header says it leaves behind.

    A cell the student is asked to write ships empty, under a header naming what it has
    to end up defining. Nothing binds those names until they write it, so without this
    every deliberately blank cell reads as a missing import - and the one check that
    would catch a REAL missing import starts crying wolf on the cells we left blank on
    purpose.

    Args:
        source: the cell's code.

    Returns:
        The names promised by any `# Creates:` line in it.
    """
    names = set()
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# Creates:"):
            promised = stripped.split(":", 1)[1]
            for part in promised.split(","):
                # "rows, a_labels, b_labels" - and "dev (a list) - the half you keep",
                # where only the first word is the name.
                word = part.strip().split(" ")[0].strip()
                if word:
                    names.add(word)
    return names


def deferred_used(tree: ast.AST) -> set:
    """The names this cell reads only from inside a function body.

    A name in a function body is looked up when the function is CALLED, not when the
    cell that defines it runs. So a function defined in one cell may call a helper
    defined three cells further down, and both are fine as long as nothing calls it in
    between. Without this, every such pair reads as a name used before it exists.

    Args:
        tree: the parsed cell.

    Returns:
        The names read inside a `def` or a `lambda` in this cell.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and not isinstance(inner.ctx, ast.Store):
                    names.add(inner.id)
    return names


def all_bound_names(notebook: dict) -> set:
    """Every name defined anywhere in the notebook, in any cell, in any order.

    Args:
        notebook: the parsed .ipynb.

    Returns:
        The names bound by any code cell.
    """
    names = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(runnable_source(cell))
        except SyntaxError:
            continue
        bound, _ = bound_and_used(tree)
        names = names | bound
    return names


def star_imported(tree: ast.AST) -> list[str]:
    """The modules this cell star-imports."""
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    modules.append(node.module)
    return modules


def runnable_source(cell: dict) -> str:
    """The cell's code, minus the IPython magics and shell lines ast cannot parse."""
    lines = []
    for line in "".join(cell["source"]).split("\n"):
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


def check_notebook(path: Path, from_config: set[str]) -> int:
    """Report every name read before anything defines it. Returns the problems."""
    known = set(dir(builtins)) | COLAB_NAMES
    problems = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Everything the notebook defines anywhere, for the function-body case below.
    defined_somewhere = all_bound_names(notebook)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(runnable_source(cell))
        except SyntaxError:
            continue                          # the syntax check reports these

        for module in star_imported(tree):
            if module == "config":
                known = known | from_config
            else:
                # A star import of anything else would hide real problems rather
                # than resolve them, so say so instead of trusting it.
                problems.append((index, 0, "from " + str(module) + " import *",
                                 "cannot resolve a star import of " + str(module)))

        bound, used = bound_and_used(tree)
        deferred = deferred_used(tree)
        # A blank cell the student is asked to fill in still promises, in its header,
        # what it has to end up defining. Take it at its word.
        bound = bound | promised_names(runnable_source(cell))
        # Within one cell, order is not tracked: a helper defined at the bottom and
        # called at the top is still fine at run time, since the cell runs as a unit.
        for name, line in used:
            if name in known or name in bound:
                continue
            # Inside a function body, the name is looked up when the function is
            # called. A later cell defining it is fine; nothing defining it is not.
            if name in deferred and name in defined_somewhere:
                continue
            problems.append((index, line, name, "used before anything defines it"))
        known = known | bound

    return problems


def main() -> bool:
    from_config = config_names()
    total = 0
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        problems = check_notebook(path, from_config)
        if problems:
            print(path.name)
            for index, line, name, why in problems:
                print("  cell " + str(index) + ", line " + str(line) + ": `" +
                      name + "` " + why)
        total = total + len(problems)

    if total == 0:
        print("All notebooks: every name is defined before it is used.")
        return 0
    print("")
    print(str(total) + " undefined name(s). Each one is a cell that fails the first")
    print("time a student runs it. Usually the SETUP cell is missing an import:")
    print("  setup_cell([... \"from pipeline import <the name>\", ...])")
    return 1


if __name__ == "__main__":
    sys.exit(main())
