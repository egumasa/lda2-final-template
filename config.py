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

from pathlib import Path

# ----------------------------------------------------------------------------------
# ✏️ YOU EDIT — these four
# ----------------------------------------------------------------------------------
TRACK = "cefr"           # cefr · raamove · cars50 · l2_errors · icnale
GROUP = "groupA"         # your group name — it goes in every output filename
SEED = 42                # change per group, so each group draws a different subset
N_PER_CLASS = 2          # keep SMALL while iterating; raise for the final run

# Are your labels ORDERED (on a scale), and if so in what order?
#   A1..C2 and Move 1..3 are ordered AND alphabetical, so None is fine.
#   Low/Mid/High is ordered but NOT alphabetical — set it, or the weighted kappa gets
#   computed over "High < Low < Mid", which means nothing.
LABELS_ORDER = None      # e.g. ["Low", "Mid", "High"] for icnale


# ----------------------------------------------------------------------------------
# Paths — worked out from the four settings above. You should not need to touch these.
# ----------------------------------------------------------------------------------
# Anchored to THIS file rather than to the working directory, so the same path works
# whether you run a notebook from notebooks/, from the repo root, or from Colab.
ROOT = Path(__file__).resolve().parent

POOL_PATH = ROOT / "data" / "pools" / (TRACK + "_pool.json")           # 01 writes
SAMPLE_PATH = ROOT / "data" / "gold" / (TRACK + "_" + GROUP + "_sample.json")   # 02 writes
GOLD_PATH = ROOT / "data" / "gold" / (TRACK + "_" + GROUP + "_gold.json")       # 03 writes
PRED_PATH = ROOT / "outputs" / (TRACK + "_" + GROUP + "_predictions.json")      # 04 writes
ROUNDS_PATH = ROOT / "outputs" / (TRACK + "_" + GROUP + "_rounds.json")         # 04 writes
PROMPT_FILE = ROOT / "prompts" / (TRACK + ".txt")
OUT_DIR = ROOT / "outputs"

# The demo pool: small, ships with the repo, lets 02–05 run before 01 has been done.
# Far too small for a real study — see data/pools/README.md.
DEMO_POOL_PATH = ROOT / "data" / "pools" / (TRACK + "_demo_pool.json")


def describe():
    """Print the settings, so every notebook can show what it is working on."""
    print("track", TRACK, "· group", GROUP, "· seed", SEED,
          "· n_per_class", N_PER_CLASS)
    print("labels order:", LABELS_ORDER)
    return None
