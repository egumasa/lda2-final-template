#!/usr/bin/env python3
"""prep_datasets.py — build the full-size pool for a track, in one command.

    python scripts/prep_datasets.py                 # every track that can be built
    python scripts/prep_datasets.py raamove         # just one
    python scripts/prep_datasets.py raamove cars50  # a few

It downloads the original corpus into data/raw/, reshapes it into the canonical schema,
and writes data/pools/<track>_pool.json. Nothing here needs a pip install beyond what
`uv sync` already gave you: the download and reshape steps are plain standard library.

The notebooks/download_<track>.ipynb files do exactly the same work, one step at a time,
with the reasoning spelled out - and they run in Colab. Use those if you want to SEE the
reshaping; use this if you just want the file.

    --demos    ALSO rebuild the small data/pools/<track>_demo_pool.json files.
               Off by default: those files are committed, so rebuilding them changes
               what ships with the template. You almost certainly do not want this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import reshape
from reshape import balanced_sample, label_counts, reid, validate

# Paths, relative to the repo root (this file lives in scripts/, so root is its parent).
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
POOLS = ROOT / "data" / "pools"


def write_json(directory, name, items, allowed=None):
    """Validate, write, and report. Every build goes through here."""
    validate(items, allowed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    import json
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    # Report the path relative to the REPO ROOT. (Repo A's version printed relative to
    # the script's own folder, which raises ValueError as soon as the output directory
    # is not underneath it - which is the case here.)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print("  wrote", shown, " (" + str(len(items)), "items) ", label_counts(items))
    return path


# ----------------------------------------------------------------------------------
# One builder per track. Each returns nothing and prints what it wrote.
# ----------------------------------------------------------------------------------
def build_raamove(demos=False):
    source = download.download_raamove(RAW)
    pool = reshape.reshape_raamove(source)
    allowed = set(reshape.RAAMOVE_LABELS.values())
    write_json(POOLS, "raamove_pool.json", pool, allowed)
    if demos:
        demo = reid(balanced_sample(pool, per_label=8))
        write_json(POOLS, "raamove_demo_pool.json", demo, allowed)


def build_cars50(demos=False):
    source = download.download_cars50(RAW)
    move_rows, step_rows = reshape.reshape_cars50(source)
    if not move_rows:
        raise RuntimeError(
            "Parsed 0 sentences from the CaRS-50 XML. Look inside one of the files in "
            + str(RAW / "cars50") + " - the tag names may have changed.")
    write_json(POOLS, "cars50_pool.json", move_rows)
    # The 11-class Move+Step version, for the harder variant of this track.
    write_json(POOLS, "cars50_step_pool.json", step_rows)
    if demos:
        write_json(POOLS, "cars50_demo_pool.json", reid(balanced_sample(move_rows, per_label=20)))


def build_l2_errors(demos=False):
    source = download.download_l2_errors(RAW)
    category_rows, detection_rows = reshape.reshape_l2_errors(source)
    write_json(POOLS, "l2_errors_pool.json", category_rows, reshape.L2_LABELS)
    # The binary yes/no variant of this track.
    write_json(POOLS, "l2_error_detection_pool.json", detection_rows,
               reshape.L2_DETECTION_LABELS)
    if demos:
        write_json(POOLS, "l2_errors_demo_pool.json",
                   reid(balanced_sample(category_rows, per_label=15)), reshape.L2_LABELS)


def build_icnale(demos=False):
    source = download.download_icnale(RAW)     # raises with instructions if not prepared
    pool = reshape.reshape_icnale(source)
    # NOTE: ICNALE is research-use-only. .gitignore keeps this file out of git and
    # make_submission.py keeps it out of your bundle. Please leave both in place.
    write_json(POOLS, "icnale_pool.json", pool, reshape.ICNALE_LABELS)
    if demos:
        print("  (no demo file for icnale: it is not redistributable)")


BUILDERS = {
    "raamove": build_raamove,
    "cars50": build_cars50,
    "l2_errors": build_l2_errors,
    "icnale": build_icnale,
}


def main(argv):
    demos = False
    targets = []
    for argument in argv:
        if argument == "--demos":
            demos = True
        elif argument in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            targets.append(argument)

    if not targets:
        targets = list(BUILDERS)

    if demos:
        print("--demos given: the committed data/pools/*_demo_pool.json files will be "
              "REBUILT and may change.\n")

    failures = []
    for name in targets:
        if name not in BUILDERS:
            print("unknown track " + repr(name) + "; choose from " + str(list(BUILDERS)))
            failures.append(name)
            continue
        print("[" + name + "]")
        try:
            BUILDERS[name](demos=demos)
        except Exception as error:
            # One track failing must not stop the others - but it must not look like
            # success either. Repo A's version printed a SKIP and exited 0, so a
            # missing track went unnoticed until a notebook could not find its pool.
            print("  FAILED:", error)
            failures.append(name)

    print()
    if failures:
        print("Could not build:", ", ".join(failures))
        print("Everything else was written to data/pools/.")
        return 1
    print("Built:", ", ".join(targets), "-> data/pools/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
