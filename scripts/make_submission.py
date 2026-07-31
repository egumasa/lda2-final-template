#!/usr/bin/env python3
"""make_submission.py — collect everything you hand in, into one folder.

    python scripts/make_submission.py --group groupA

It builds ../lda2_project_<group>/ next to this repo, keeping the
    scripts/ · prompts/ · data/ · notebooks/ · outputs/
layout intact. Then, in Drive: right-click the folder -> Download -> upload the zip to
the *Final mini-project* assignment in Google Classroom -> Turn in.

WHY A SCRIPT RATHER THAN "COPY THESE NINE FILES"
------------------------------------------------
Two of the things that must NOT be submitted are easy to include by accident and
awkward to undo:

  * your .env, which holds your Gemini API key;
  * data/pools/ and anything ICNALE-derived — large, and in ICNALE's case not licensed
    for redistribution at all.

So the rule here is an ALLOWLIST: nothing is copied unless it is named below. Adding a
new kind of output means adding it here on purpose, rather than discovering later that
a whole corpus went up to Classroom.

WHY THE FOLDER STRUCTURE IS KEPT
--------------------------------
Partly because that layout IS the S10 reproducibility checklist made physical - code,
prompts, data and outputs, separated so each can be pointed at. And partly because the
notebooks read "../scripts" and "../data/...": flatten the folder and the submitted
copy no longer runs.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Never copied, wherever they appear. The ICNALE rule is a licence obligation, not
# tidiness - please do not remove it.
EXCLUDE_NAMES = {".git", ".venv", ".env", "__pycache__", ".ipynb_checkpoints",
                 ".DS_Store", "raw", "pools"}
EXCLUDE_SUBSTRINGS = ["icnale"]


def _is_excluded(path):
    if path.name in EXCLUDE_NAMES:
        return True
    lowered = path.name.lower()
    for substring in EXCLUDE_SUBSTRINGS:
        if substring in lowered:
            return True
    return False


def copy_file(source, destination):
    if not source.exists() or _is_excluded(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_matching(source_dir, destination_dir, pattern):
    """Copy the files in one directory matching a glob, skipping excluded ones."""
    copied = []
    if not source_dir.exists():
        return copied
    for path in sorted(source_dir.glob(pattern)):
        if path.is_dir() or _is_excluded(path):
            continue
        if copy_file(path, destination_dir / path.name):
            copied.append(path.name)
    return copied


def main(argv):
    parser = argparse.ArgumentParser(
        description="Collect your mini-project files into a submission folder.")
    parser.add_argument("--group", required=True,
                        help="your group name — must match group in config.yaml")
    parser.add_argument("--track", default=None,
                        help="track name; inferred from your output files if omitted")
    parser.add_argument("--run", default=None,
                        help="which run to bundle, e.g. v1; inferred (newest) if omitted")
    parser.add_argument("--out", default=None,
                        help="where to write the folder (default: next to this repo)")
    args = parser.parse_args(argv)

    group = args.group
    if args.out:
        bundle = Path(args.out).resolve()
    else:
        bundle = ROOT.parent / ("lda2_project_" + group)

    # Work out the track and the run from whatever the export step wrote, unless told.
    # Report files are named <track>_<group>_<run>_report.md, so both are in the name.
    # Sorting puts the newest run last when they are v1, v2, v3 - so take the last one
    # rather than the first, and say which, because a group that ran twice should not
    # have to guess which attempt got submitted.
    track = args.track
    run = args.run
    reports = sorted((ROOT / "outputs").glob("*_" + group + "_*_report.md"))
    if reports:
        newest = reports[-1].name
        if track is None:
            track = newest.split("_" + group + "_")[0]
        if run is None:
            run = newest.split("_" + group + "_")[1][:-len("_report.md")]
        if len(reports) > 1:
            print("Found", len(reports), "runs. Bundling the newest:", newest)
    if track is None or run is None:
        print("Could not work out which track and run this is. Either run the export "
              "(export_results) first, or pass --track and --run.")
        return 1

    print("Track:", track, " Group:", group, " Run:", run)
    print("Building:", bundle)
    if bundle.exists():
        shutil.rmtree(bundle)          # rebuild from scratch, so nothing stale survives
    bundle.mkdir(parents=True)

    stem = track + "_" + group + "_" + run
    found = {}
    missing = []

    # --- the plan (the gate) ------------------------------------------------------
    if copy_file(ROOT / "PLAN.md", bundle / "PLAN.md"):
        found["PLAN.md"] = ["PLAN.md"]
    else:
        missing.append("PLAN.md — the 作戦シート. It travels with the bundle as "
                       "evidence the gate was passed.")

    # --- the notebook -------------------------------------------------------------
    # All five, filled in. 01 is per-track, so only the one you actually ran is asked
    # for; 02-05 are the same file for everyone and all four should be there.
    notebooks = copy_matching(ROOT / "notebooks", bundle / "notebooks", "0*.ipynb")
    if notebooks:
        found["notebooks/"] = notebooks
    else:
        missing.append("notebooks/0*.ipynb — your completed notebooks 01-05.")
    for stage in ("02_sample", "03_annotate", "04_prompt", "05_report"):
        if not any(name.startswith(stage) for name in notebooks):
            missing.append("notebooks/" + stage + ".ipynb — filled in.")

    # --- the prompts, including the iteration trail --------------------------------
    prompts = copy_matching(ROOT / "prompts", bundle / "prompts", track + "*.txt")
    if prompts:
        found["prompts/"] = prompts
    else:
        missing.append("prompts/" + track + "*.txt — your prompt file(s).")

    # --- your adjudicated gold set --------------------------------------------------
    gold = copy_matching(ROOT / "data" / "gold", bundle / "data" / "gold", stem + "_gold.json")
    if not gold:
        gold = copy_matching(ROOT / "outputs", bundle / "data" / "gold", stem + "_gold.json")
    if gold:
        found["data/gold/"] = gold
    else:
        missing.append("data/gold/" + stem + "_gold.json — your ADJUDICATED gold set "
                       "(notebook 03). Without it the numbers cannot be checked.")

    # --- outputs: frozen predictions, CSV, report ------------------------------------
    # Named patterns rather than "<stem>*": export_results also drops a copy of the gold
    # set in outputs/, and shipping it in two places invites the question of which one
    # the numbers were actually computed against. It belongs in data/gold/.
    outputs = []
    for pattern in ("_predictions.json", "_predictions.csv", "_report.md"):
        outputs = outputs + copy_matching(ROOT / "outputs", bundle / "outputs",
                                          stem + pattern)
    if outputs:
        found["outputs/"] = outputs
    if not any(name.endswith("_predictions.json") for name in outputs):
        missing.append("outputs/" + stem + "_predictions.json — your FROZEN predictions "
                       "(notebook 04). This is the file your reported F1 must come from.")
    if not any(name.endswith("_report.md") for name in outputs):
        missing.append("outputs/" + stem + "_report.md — the one-page report (notebook 05).")

    # --- the plumbing, so the bundle actually runs -------------------------------------
    scripts = copy_matching(ROOT / "scripts", bundle / "scripts", "*.py")
    found["scripts/"] = scripts

    # config.yaml is your group's settings; config.py turns them into every path the
    # notebooks use. Without both, the bundle is a set of notebooks that die on their
    # first line, and the track, seed and n_per_class the numbers came from are nowhere
    # in the submission.
    if copy_file(ROOT / "config.yaml", bundle / "config.yaml"):
        found["config.yaml"] = ["config.yaml"]
    else:
        missing.append("config.yaml — your group's track, seed and n_per_class. It "
                       "records the settings your results came from.")
    if copy_file(ROOT / "config.py", bundle / "config.py"):
        found["config.py"] = ["config.py"]
    else:
        missing.append("config.py — reads config.yaml and builds every path. Every "
                       "notebook imports it.")

    # --- slides ------------------------------------------------------------------------
    slides = []
    for candidate in ("slides.pdf", "slides.pptx", "slides.key"):
        if copy_file(ROOT / candidate, bundle / candidate):
            slides.append(candidate)
    if slides:
        found["slides"] = slides
    else:
        missing.append("slides.pdf — your 5 slides. Put them in the repo root, or add "
                       "them to the folder by hand.")

    # --- report ---------------------------------------------------------------------
    print()
    total = 0
    for section in sorted(found):
        names = found[section]
        total = total + len(names)
        print(" ", section)
        for name in names:
            print("     ", name)

    print()
    print(total, "file(s) collected into", bundle)

    # A report still full of the scaffold's placeholder prose is the most common way to
    # lose marks, and the easiest to check mechanically.
    report_path = bundle / "outputs" / (stem + "_report.md")
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        placeholders = text.count("_<") + text.count("_Replace") + text.count("_For each")
        if placeholders:
            print()
            print("WARNING: your report still contains", placeholders,
                  "placeholder passage(s) in italics.")
            print("         Those are the sections you are meant to write. A section "
                  "left as the")
            print("         scaffold's own prose scores zero.")

    if missing:
        print()
        print("STILL MISSING:")
        for line in missing:
            print("  -", line)
        print()
        print("The folder was still built with what exists, so you can re-run this "
              "after filling the gaps.")
        return 1

    print()
    print("Nothing missing. Next: find", bundle.name, "in Drive, right-click ->")
    print("Download, then upload the zip to the Final mini-project assignment.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
