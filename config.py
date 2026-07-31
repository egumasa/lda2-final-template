"""config.py — the one file you edit before running the notebooks.

    from config import *

Notebooks 01–05 all start by importing this. Set it once, as a group, and every
notebook downstream agrees with every other one: same track, same seed, same file
names. That is why it is a file rather than a cell copied into five notebooks — five
copies of a seed is five chances for the sample in notebook 03 not to be the sample
that notebook 02 drew.

It should match your PLAN.md. If you change something here, re-run the setup cell of
whichever notebook you are in.
"""

import importlib.util
from pathlib import Path

# ----------------------------------------------------------------------------------
# ✏️ YOU EDIT — these four
# ----------------------------------------------------------------------------------
TRACK = "raamove"        # raamove · cars50 · l2_errors · icnale
GROUP = "groupA"         # your group name — it goes in every output filename
SEED = 42                # change per group, so each group draws a different subset
N_PER_CLASS = 2          # keep SMALL while iterating; raise for the final run

# The Google accounts of everyone in your group. Notebook 03 gives each of them edit
# access to the annotation sheet it creates. Leave it empty and the sheet lives in one
# person's Drive where the second coder cannot open it — which is the whole of step 3.
MEMBERS = []             # e.g. ["b1234567@dc.tohoku.ac.jp", "..."]

# Are your labels ORDERED (on a scale), and if so in what order?
#   Move 1..3 is ordered AND alphabetical, so None is fine.
#   Low/Mid/High is ordered but NOT alphabetical — set it, or the weighted kappa gets
#   computed over "High < Low < Mid", which means nothing.
LABELS_ORDER = None      # e.g. ["Low", "Mid", "High"] for icnale


# ----------------------------------------------------------------------------------
# Paths — worked out from the four settings above. You should not need to touch these.
# ----------------------------------------------------------------------------------
# Anchored to THIS file rather than to the working directory, so the same path works
# whether you run a notebook from notebooks/, from the repo root, or from Colab.
ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------
# The Drive check — every path above is only as durable as the folder ROOT points at.
# ----------------------------------------------------------------------------------
# A Colab runtime is temporary storage. Files written to it look completely normal
# until the runtime resets, at which point your pool, your gold set and your outputs
# are gone, and nobody else in the group ever saw them. So: if we are in Colab, ROOT
# has to be inside your group's Drive folder. If it is not, stop here rather than let
# a whole annotation round be written somewhere that will not survive lunch.
IN_COLAB = importlib.util.find_spec("google.colab") is not None
ON_DRIVE = "/content/drive/" in str(ROOT)

if IN_COLAB and not ON_DRIVE:
    raise RuntimeError(
        "This notebook is running from a COPY that lives only in the Colab runtime.\n"
        "  found: " + str(ROOT) + "\n"
        "  wanted: a folder under /content/drive/MyDrive/\n"
        "Anything you do here — the pool you build, the gold set you annotate, the "
        "prompts you write — disappears when the runtime resets, and the rest of your "
        "group cannot see any of it.\n"
        "Re-run the SETUP cell at the top of this notebook; it will mount Drive and "
        "move you into your group's shared folder."
    )

POOL_PATH = ROOT / "data" / "pools" / (TRACK + "_pool.json")           # 01 writes
SAMPLE_PATH = ROOT / "data" / "gold" / (TRACK + "_" + GROUP + "_sample.json")   # 02 writes
GOLD_PATH = ROOT / "data" / "gold" / (TRACK + "_" + GROUP + "_gold.json")       # 03 writes
PRED_PATH = ROOT / "outputs" / (TRACK + "_" + GROUP + "_predictions.json")      # 04 writes
ROUNDS_PATH = ROOT / "outputs" / (TRACK + "_" + GROUP + "_rounds.json")         # 04 writes
PROMPT_FILE = ROOT / "prompts" / (TRACK + ".txt")
OUT_DIR = ROOT / "outputs"

# Where notebook 03 writes down the annotation sheet it created. The sheet is the one
# handoff in this project that is not a file of its own — without this, the link to
# your group's annotation round survives only in one person's notebook output.
SHEET_PATH = ROOT / "data" / "gold" / (TRACK + "_" + GROUP + "_sheet.json")

# The demo pool: small, ships with the repo, lets 02–05 run before 01 has been done.
# Far too small for a real study — see data/pools/README.md.
DEMO_POOL_PATH = ROOT / "data" / "pools" / (TRACK + "_demo_pool.json")


def describe():
    """Print the settings, so every notebook can show what it is working on."""
    print("track", TRACK, "· group", GROUP, "· seed", SEED,
          "· n_per_class", N_PER_CLASS)
    print("labels order:", LABELS_ORDER)
    # Where the files go. Say it every time: it is the one setting nobody edits and
    # everybody depends on, and "which folder am I actually in" is the question behind
    # most of the ways a group loses a morning's work.
    print("files:", ROOT)
    if ON_DRIVE:
        print("        ^ in Drive — your group can see these, and they survive a reset")
    else:
        print("        ^ a local checkout (not Colab)")
    return None
