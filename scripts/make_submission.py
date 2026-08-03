#!/usr/bin/env python3
"""make_submission.py — collect everything you hand in, into one folder.

    python scripts/make_submission.py --group groupA

Run it again after filling a gap and it stops rather than rebuild, because rebuilding
deletes anything you put in the folder by hand. Add --overwrite when you mean it.

It builds ../lda2_project_<group>/ next to this repo, keeping the
    scripts/ · prompts/ · data/ · notebooks/ · outputs/
layout intact. Then, in Drive: right-click the folder -> Download -> upload the zip to
the *Final mini-project* assignment in Google Classroom -> Turn in. One zip per group.

Your two-page report is not in this bundle. It is written individually, in Word, from
the numbers notebook 06 prints on screen, and each member uploads their own.

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

The dev/test split and the test-scoring log are on the allowlist deliberately. They are
what makes the reported number auditable: which items it was measured on, and how many
times that set was scored before the number was quoted.

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


def _is_excluded(path: Path) -> bool:
    """Should this file be kept out of the bundle?

    Args:
        path: the file being considered.

    Returns:
        True when its name is on the exclusion list.
    """
    if path.name in EXCLUDE_NAMES:
        return True
    lowered = path.name.lower()
    for substring in EXCLUDE_SUBSTRINGS:
        if substring in lowered:
            return True
    return False


def copy_file(source: Path, destination: Path) -> bool:
    """Copy one file, unless it is missing or excluded.

    Args:
        source: the file to copy.
        destination: where to put it. Its folder is made if missing.

    Returns:
        True when the file was copied.
    """
    if not source.exists() or _is_excluded(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_matching(source_dir: Path, destination_dir: Path,
                  pattern: str) -> list[str]:
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


def parse_arguments(argv: list[str]):
    """Read the command line.

    Args:
        argv: the command-line arguments.

    Returns:
        The parsed arguments, with .group, .track, .run, .out and .overwrite.
    """
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
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild the folder even though it already exists")
    return parser.parse_args(argv)


def infer_track_and_run(group: str, track: str | None, run: str | None) -> tuple:
    """Work out which track and run to bundle, from whatever the export step wrote.

    Prediction CSVs are named <track>_<group>_<run>_predictions.csv, so both are in
    the name. A group that ran twice should not have to guess which attempt got
    submitted, so the choice is always announced.
    """
    suffix = "_predictions.csv"
    found = sorted((ROOT / "outputs").glob("*_" + group + "_*" + suffix))
    if not found:
        return track, run
    newest = found[-1].name
    if track is None:
        track = newest.split("_" + group + "_")[0]
    if run is None:
        run = newest.split("_" + group + "_")[1][:-len(suffix)]
    if len(found) > 1:
        print("Found", len(found), "runs. Bundling the newest:", newest)
        print("  (pass --run to bundle a different one.)")
    return track, run


def prepare_bundle_folder(bundle: Path, overwrite: bool) -> bool:
    """Make an empty folder to build in, refusing to wipe one you may have added to.

    The folder is rebuilt from scratch so nothing stale survives - which also means
    anything you dropped in by hand, like slides, goes with it. So it will not do that
    behind your back.

    Args:
        bundle: the folder to build in.
        overwrite: True to rebuild a folder that already exists.

    Returns:
        True when the folder is ready. False when it already existed and overwrite
        was not given - nothing has been deleted.
    """
    if bundle.exists() and not overwrite:
        print("That folder already exists:")
        print("   ", bundle)
        print("Rebuilding it deletes everything inside, including anything you added by")
        print("hand (your slides, say). If that is what you want, run the command again")
        print("with --overwrite on the end. Otherwise rename or move the old folder")
        print("first.")
        return False
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    return True


def collect_the_plan(bundle: Path, found: dict, missing: list[str]) -> None:
    """PLAN.md - the gate. It travels with the bundle as evidence the gate was passed."""
    if copy_file(ROOT / "PLAN.md", bundle / "PLAN.md"):
        found["PLAN.md"] = ["PLAN.md"]
    else:
        missing.append("PLAN.md — the 作戦シート. It travels with the bundle as "
                       "evidence the gate was passed.")


def collect_notebooks(bundle: Path, found: dict, missing: list[str]) -> None:
    """All six notebooks, filled in.

    01 is per-track, so only the one you actually ran is asked for; 02-06 are the same
    files for everyone and all of them should be there.
    """
    notebooks = copy_matching(ROOT / "notebooks", bundle / "notebooks", "0*.ipynb")
    if notebooks:
        found["notebooks/"] = notebooks
    else:
        missing.append("notebooks/0*.ipynb — your completed notebooks 01-06.")
    for stage in ("02_sample", "03_annotate", "04_develop", "05_test", "06_report"):
        if not any(name.startswith(stage) for name in notebooks):
            missing.append("notebooks/" + stage + ".ipynb — filled in.")


def collect_prompts(bundle: Path, track: str, found: dict,
                    missing: list[str]) -> None:
    """Your prompt file(s), including the iteration trail."""
    prompts = copy_matching(ROOT / "prompts", bundle / "prompts", track + "*.txt")
    if prompts:
        found["prompts/"] = prompts
    else:
        missing.append("prompts/" + track + "*.txt — your prompt file(s).")


def collect_gold(bundle: Path, stem: str, found: dict,
                 missing: list[str]) -> None:
    """Your adjudicated gold set, and the line you drew through it.

    All three files, because the split is part of the claim: the headline F1 was computed on
    the test half, and a reader who cannot see which items those were cannot check it.
    """
    gold = []
    for name in (stem + "_gold.json", stem + "_dev.json", stem + "_test.json"):
        copied = copy_matching(ROOT / "data" / "gold", bundle / "data" / "gold", name)
        if not copied:
            copied = copy_matching(ROOT / "outputs", bundle / "data" / "gold", name)
        gold = gold + copied
    if gold:
        found["data/gold/"] = gold
    if not any(name.endswith("_gold.json") for name in gold):
        missing.append("data/gold/" + stem + "_gold.json — your ADJUDICATED gold set "
                       "(notebook 03). Without it the numbers cannot be checked.")
    if not any(name.endswith("_test.json") for name in gold):
        missing.append("data/gold/" + stem + "_test.json — your HELD-OUT test set "
                       "(notebook 03). Your headline F1 was computed on it, so it has "
                       "to be checkable.")
    if not any(name.endswith("_dev.json") for name in gold):
        missing.append("data/gold/" + stem + "_dev.json — the dev half you tuned on "
                       "(notebook 03).")


def collect_outputs(bundle: Path, stem: str, found: dict,
                    missing: list[str]) -> None:
    """The frozen predictions, the CSV and the test log.

    Named patterns rather than "<stem>*": export_results also drops a copy of the
    # scored items in outputs/, and shipping them in two places invites the question of
    # which one the numbers were computed against. They belong in data/gold/.
    # "_predictions*.json" and not "_predictions.json": a second scoring of the held-out
    # set lands in _predictions_attempt2.json rather than replacing the first, and the
    whole point of keeping it is that it does not go missing here.
    """
    outputs = []
    for pattern in ("_predictions*.json", "_predictions.csv", "_test_log.jsonl"):
        outputs = outputs + copy_matching(ROOT / "outputs", bundle / "outputs",
                                          stem + pattern)
    if outputs:
        found["outputs/"] = outputs
    if not any(name.endswith("_predictions.json") for name in outputs):
        missing.append("outputs/" + stem + "_predictions.json — your FROZEN predictions "
                       "(notebook 05). This is the file your reported F1 must come from.")
    if not any(name.endswith("_predictions.csv") for name in outputs):
        missing.append("outputs/" + stem + "_predictions.csv — the per-item table "
                       "(notebook 06). It is what your error analysis is read off.")
    if not any(name.endswith("_test_log.jsonl") for name in outputs):
        missing.append("outputs/" + stem + "_test_log.jsonl — the test-scoring log "
                       "(notebook 05). It records how many times the held-out set was "
                       "scored, and that is part of the method.")


def collect_plumbing(bundle: Path, found: dict, missing: list[str]) -> None:
    """The code, so the bundle actually runs."""
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


def collect_slides(bundle: Path, found: dict, missing: list[str]) -> None:
    """Your 5 slides."""
    slides = []
    for candidate in ("slides.pdf", "slides.pptx", "slides.key"):
        if copy_file(ROOT / candidate, bundle / candidate):
            slides.append(candidate)
    if slides:
        found["slides"] = slides
    else:
        missing.append("slides.pdf — your 5 slides. Put them in the repo root, or add "
                       "them to the folder by hand.")


def print_what_was_collected(found: dict, bundle: Path) -> None:
    """List every file that went in, section by section."""
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


def remind_about_the_report() -> None:
    """The report is written individually and handed in separately, so it is not in
    this bundle and nothing here can check it. Saying so is the point."""
    print()
    print("This bundle is the group's work: the plan, the notebooks, the gold set,")
    print("the predictions and the slides. Your two-page report is NOT in it.")
    print("Each member writes their own and uploads it to Classroom separately.")


def main(argv: list[str]) -> int:
    """Collect your mini-project files into a submission folder.

    Args:
        argv: the command-line arguments.

    Returns:
        0 when nothing was missing, 1 otherwise.
    """
    args = parse_arguments(argv)
    group = args.group
    if args.out:
        bundle = Path(args.out).resolve()
    else:
        bundle = ROOT.parent / ("lda2_project_" + group)

    track, run = infer_track_and_run(group, args.track, args.run)
    if track is None or run is None:
        print("Could not work out which track and run this is. Either run the export "
              "(export_results) first, or pass --track and --run.")
        return 1

    print("Track:", track, " Group:", group, " Run:", run)
    print("Building:", bundle)
    if not prepare_bundle_folder(bundle, args.overwrite):
        return 1

    stem = track + "_" + group + "_" + run
    found = {}
    missing = []

    collect_the_plan(bundle, found, missing)
    collect_notebooks(bundle, found, missing)
    collect_prompts(bundle, track, found, missing)
    collect_gold(bundle, stem, found, missing)
    collect_outputs(bundle, stem, found, missing)
    collect_plumbing(bundle, found, missing)
    collect_slides(bundle, found, missing)

    print_what_was_collected(found, bundle)
    remind_about_the_report()

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
