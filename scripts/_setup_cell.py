"""_setup_cell.py — the bootstrap cell that opens every notebook.

Both notebook generators import this, so all nine notebooks start the same way and
there is exactly one copy of the rule they enforce:

    a group's work lives in the group's Google Drive folder, or it does not run.

A Colab runtime is temporary storage. Files written to it look completely normal
until the runtime resets, at which point a morning's annotation is gone and nobody
else in the group ever saw it. The old version of this cell offered that as one of
two commented-out options; this one mounts Drive, finds the shared folder, and stops
if it cannot.

The one thing that does NOT go in Drive is the raw corpus a 01 notebook downloads:
it is large, it is mostly not ours to redistribute, and it is one command to fetch
again. Derived artefacts persist, raw inputs do not. That is why `setup_lines` takes
a `workdir` — notebooks 02-05 work inside the project, 01 works in a scratch folder
in the runtime and reaches the project through the absolute paths in `config.py`.

`setup_lines()` returns source lines; each generator wraps them with its own `code()`
helper, because the two generators build cell dicts slightly differently.
"""

FOLDER = "lda2-final-template"     # the shared folder's name, in every member's Drive
REPO = "egumasa/lda2-final-template"
SCRATCH = "/content/raw"           # where a 01 notebook downloads a corpus, in Colab

SETUP_MD_LINES = [
    "## Setup — run this first",
    "",
    "This cell mounts your Google Drive and finds your group's shared folder, "
    "`" + FOLDER + "`. Everything the project produces — the pool, the gold set, your "
    "prompts, the outputs — is an ordinary file in there, which is what makes it "
    "survive the runtime resetting *and* lets the rest of your group see it.",
    "",
    "**One member sets the folder up once:**",
    "",
    "1. That member runs the `git clone` line this cell prints if the folder is "
    "missing, which puts it in their own Drive.",
    "2. They share it with the group (right-click ▸ *Share*), with edit access.",
    "3. Everyone else opens *Shared with me*, right-clicks the folder, and chooses "
    "**Add shortcut to Drive** ▸ *My Drive*.",
    "",
    "Keep that shortcut's name exactly `" + FOLDER + "`. It is what makes the same path "
    "work for all of you — if Drive renames it to `" + FOLDER + " (1)`, this cell will "
    "not find it.",
    "",
    "From then on, open notebooks from the folder itself (*File ▸ Open notebook ▸ "
    "Drive*) rather than from the GitHub badge, so you are working on your group's "
    "copy and not a fresh one.",
]


def setup_md(show_example: str | None = None) -> dict:
    """The markdown above the SETUP cell.

    `show_example` names a function the notebook imports, and adds the paragraph on
    reading a helper. Leave it off for a notebook that imports nothing from
    `scripts/` — a 01 notebook has the code it uses written out in front of it, so
    there is nothing there to go and look up.
    """
    if show_example is None:
        return list(SETUP_MD_LINES)
    return SETUP_MD_LINES + [
        "",
        "**Looking inside a helper.** The functions this cell imports are defined in "
        "`scripts/`. Two ways to read one, both the same ones you used on Day 2:",
        "",
        "- `help(" + show_example + ")` prints its first line — what to pass in and "
        "what comes back — and the description of each argument. Typing "
        "`" + show_example + "(` and pressing **Shift+Tab** shows the same thing in a "
        "pop-up.",
        "- To read the code itself, open `scripts/pipeline.py` from the **Files** "
        "panel on the left. Colab lists the functions in that file down the side, so "
        "you can click straight to the one you want.",
    ]


CONFIG_NAMES = [
    "TRACK", "GROUP", "RUN", "SEED", "N_PER_CLASS", "DEV", "CODERS",
    "MEMBERS", "LABELS_ORDER", "TEMPERATURE", "MODEL", "ROOT", "OUT_DIR",
    "POOL_PATH", "DEMO_POOL_PATH", "SAMPLE_PATH", "GOLD_PATH",
    "SAMPLE_BEFORE_TOPUP_PATH",
    "DEV_PATH", "TEST_PATH", "DISAGREED_PATH", "PRED_PATH",
    "ROUNDS_PATH", "NOTES_PATH", "TESTLOG_PATH",
    "PROMPT_FILE", "SHEET_PATH", "TRIAGE_PATH", "describe",
]


def _config_import(exclude: list[str] | tuple = ()) -> list[str]:
    """The `from config import (...)` block, wrapped, minus anything excluded.

    Excluding is not tidiness. `04_develop` tells the student it cannot reach the
    held-out set, and that is only true if the names are absent - a bound `TEST_PATH`
    makes `load_gold(TEST_PATH)` work in any new cell, so the promise the notebook
    makes about itself would be false.

    Args:
        exclude: names to leave out of this notebook's import.

    Returns:
        The source lines of the import statement.
    """
    wanted = []
    for name in CONFIG_NAMES:
        if name not in exclude:
            wanted.append(name)

    lines, current = [], "from config import ("
    for index, name in enumerate(wanted):
        piece = name + (")" if index == len(wanted) - 1 else ", ")
        if len(current) + len(piece) > 79:
            lines.append(current.rstrip())
            current = " " * len("from config import (")
        current = current + piece
    lines.append(current.rstrip())
    return lines


def setup_lines(extra_imports: list[str] | tuple = (),
                workdir: str | None = None,
                exclude: list[str] | tuple = ()) -> list[str]:
    """The source of the SETUP cell.

    `extra_imports` are appended after the config import.
    `workdir` is where to work in Colab: None means the project's own `notebooks/`
    (notebooks 02-06); pass SCRATCH for a 01 notebook, which downloads a corpus into
    the working directory and must not put that in Drive.
    `exclude` drops config names from this notebook's import - see `_config_import`.
    """
    if workdir is None:
        target = 'PROJECT + "/notebooks"'
        why = "# Work inside the project folder, where the notebooks live."
    else:
        target = '"' + workdir + '"'
        why = ("# Work in the RUNTIME, not in Drive: the next cells download a whole\n"
               "# corpus, and raw data is big, mostly not ours to redistribute, and one\n"
               "# command to fetch again. The pool you build from it is what persists.")

    lines = [
        "# ------------------------------------------------------------------",
        "# SETUP — run me first. You are not expected to read it.",
        "# ------------------------------------------------------------------",
        "# This cell is plumbing, and it is the only cell in the project that is.",
        "# It finds your group's shared folder in Google Drive, because everything",
        "# this project keeps goes in there: a Colab runtime is wiped when it resets,",
        "# and nobody else in your group can see inside it. Then it makes the",
        "# project's own code importable. Run it and move on; nothing below asks you",
        "# to have understood it.",
        "",
        'FOLDER = "' + FOLDER + '"     # the shared folder, in every member\'s Drive',
        "",
        "import os, sys",
        "",
        'PROJECT = ".."                              # running locally: it is just above us',
        "",
        "try:",
        "    from google.colab import drive           # only exists inside Colab",
        "except ImportError:",
        "    pass",
        "else:",
        '    drive.mount("/content/drive")',
        '    PROJECT = "/content/drive/MyDrive/" + FOLDER',
        "    if not os.path.isdir(PROJECT):",
        "        raise RuntimeError(",
        '            "Could not find " + PROJECT + "\\n\\n"',
        '            "Setting the folder up for your group? Run this in a new cell:\\n"',
        '            "  !git clone https://github.com/' + REPO + '.git "',
        '            + PROJECT + "\\n"',
        '            "then share the folder with the rest of your group.\\n\\n"',
        '            "Someone else already did? Open Drive, find the folder under "',
        '            "\'Shared with me\', right-click it, and choose \'Add shortcut to "',
        '            "Drive\'. Keep the name exactly " + FOLDER + ".")',
    ]
    for line in why.split("\n"):
        lines.append("    " + line)
    lines = lines + [
        "    os.makedirs(" + target + ", exist_ok=True)",
        "    os.chdir(" + target + ")",
        "",
        "# scripts/ and config.py, by their real paths - so they are found from wherever",
        "# this notebook happens to be working.",
        "sys.path.append(PROJECT)",
        'sys.path.append(PROJECT + "/scripts")',
        "",
        "# Re-read config.yaml every time this cell runs. Without the reload, Python",
        "# hands back the settings it read the FIRST time, and editing config.yaml",
        "# would appear to do nothing until you restarted the runtime.",
        "import importlib",
        "import config",
        "importlib.reload(config)",
        "",
        "# Named one by one rather than with `import *`, so that every name a cell",
        "# below uses can be traced back to the file it came from — config.yaml for",
        "# these, scripts/ for the rest.",
    ]
    lines = lines + _config_import(exclude)
    lines = lines + list(extra_imports)
    lines = lines + ["", "describe()                  # what this notebook is working on",
                     ""]
    return lines
