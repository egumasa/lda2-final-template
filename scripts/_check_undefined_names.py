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


def config_names():
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


def bound_and_used(tree):
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


def star_imported(tree):
    """The modules this cell star-imports."""
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    modules.append(node.module)
    return modules


def runnable_source(cell):
    """The cell's code, minus the IPython magics and shell lines ast cannot parse."""
    lines = []
    for line in "".join(cell["source"]).split("\n"):
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


def check_notebook(path, from_config):
    """Report every name read before anything defines it. Returns the problems."""
    known = set(dir(builtins)) | COLAB_NAMES
    problems = []
    notebook = json.loads(path.read_text(encoding="utf-8"))

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
        # Within one cell, order is not tracked: a helper defined at the bottom and
        # called at the top is still fine at run time, since the cell runs as a unit.
        for name, line in used:
            if name in known or name in bound:
                continue
            problems.append((index, line, name, "used before anything defines it"))
        known = known | bound

    return problems


def main():
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
