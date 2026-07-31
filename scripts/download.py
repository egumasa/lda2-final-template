"""download.py — fetching each raw corpus into data/raw/.

One function per track. Each returns the path that `reshape.py` needs, and each is safe
to re-run: if the data is already there, nothing is downloaded again.

Three of the five download automatically. Two do not, for different reasons:
  * ICNALE GRA is password-gated behind a registration form, so there is nothing to
    automate - the function detects whether you have done it and says what to do.
  * Nothing here is scraped or worked around. If a licence says ask first, we ask first.

You do NOT need to edit this file.
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path


def _run_git_clone(url, destination):
    """Clone a repo, and if git fails, show WHY rather than a bare exit code."""
    print("  cloning", url, "...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # capture_output hides stderr, which is where the actual reason lives (no
        # network, a proxy, git not installed). Put it back in front of the reader.
        raise RuntimeError(
            "git clone failed for " + url + "\n"
            "  exit code: " + str(result.returncode) + "\n"
            "  git said: " + (result.stderr or "(nothing)").strip() + "\n"
            "If git is not installed, or this machine is behind a proxy, download the "
            "repository by hand and unpack it into " + str(destination) + "."
        )


def _fetch_with_browser_agent(url, timeout=60):
    """Open a URL pretending to be a browser (some CDNs refuse anything else)."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(request, timeout=timeout)


def download_raamove(raw_dir):
    """Clone RAAMove and return the folder holding its per-domain JSON files."""
    destination = Path(raw_dir) / "raamove"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git_clone("https://github.com/ljk1228/RAAMove", destination)
    # The JSON files may sit at the repo root or one level down.
    if (destination / "Intelligence.json").exists():
        return destination
    matches = sorted(destination.rglob("Intelligence.json"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        "Cloned RAAMove but found no Intelligence.json under " + str(destination) + "."
    )


def download_cars50(raw_dir):
    """Download the 50 CaRS-50 XML files from Mendeley Data and return their folder."""
    destination = Path(raw_dir) / "cars50"
    existing = sorted(destination.glob("*.xml")) if destination.exists() else []
    if existing:
        return destination

    print("  downloading CaRS-50 from the Mendeley public API ...")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(_fetch_with_browser_agent(
        "https://data.mendeley.com/public-api/datasets/kwr9s5c4nk").read())

    failures = []
    for file_record in metadata["files"]:
        target = destination / file_record["filename"]
        if target.exists():
            continue
        url = file_record["content_details"]["download_url"]
        last_error = None
        for attempt in range(3):          # the CDN drops the occasional connection
            try:
                target.write_bytes(_fetch_with_browser_agent(url).read())
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(2)
        if last_error is not None:
            # Do NOT swallow this. A silent partial download produces a smaller pool
            # than you think you have, and nothing downstream would notice.
            failures.append(file_record["filename"] + ": " + str(last_error))

    downloaded = sorted(destination.glob("*.xml"))
    if failures:
        print("  WARNING:", len(failures), "file(s) failed to download:")
        for line in failures[:5]:
            print("    -", line)
    if not downloaded:
        raise RuntimeError(
            "Downloaded no CaRS-50 XML files. Fetch them by hand from "
            "https://data.mendeley.com/datasets/kwr9s5c4nk/1 into " + str(destination)
        )
    print("  have", len(downloaded), "XML file(s).")
    return destination


def download_l2_errors(raw_dir):
    """Download the AutoErrorAnalyzer annotations CSV from OSF and return its path."""
    destination = Path(raw_dir) / "l2_errors"
    target = destination / "data_category.csv"
    if target.exists():
        return target
    destination.mkdir(parents=True, exist_ok=True)
    print("  downloading data_category.csv from OSF ...")
    # The direct file link from the OSF project (osf.io/jyf3r, Analysis folder).
    urllib.request.urlretrieve("https://osf.io/download/gezat/", str(target))
    if target.stat().st_size == 0:
        target.unlink()
        raise RuntimeError(
            "OSF returned an empty file. Download Analysis/data_category.csv by hand "
            "from https://osf.io/jyf3r into " + str(destination) + "."
        )
    return target


def download_icnale(raw_dir):
    """Check whether the ICNALE CSV has been prepared, and explain it if not.

    There is nothing to automate here, by design: ICNALE GRA is released for research
    use via a registration form that emails you a password. That also means it must
    never be committed or included in a submission bundle.
    """
    destination = Path(raw_dir) / "icnale"
    target = destination / "essays_scores.csv"
    if target.exists():
        return target
    destination.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        "ICNALE GRA cannot be downloaded automatically - it is research-use-only and "
        "password-gated. To use this track:\n"
        "  1. Register at https://language.sakura.ne.jp/icnale/download.html and wait "
        "for the password.\n"
        "  2. Download and unpack ICNALE_GRA_2.x.zip.\n"
        "  3. From its rating tables, export a CSV with exactly two columns, `text` "
        "and `score`, to:\n"
        "       " + str(target) + "\n"
        "  4. Re-run this builder.\n"
        "Note: ICNALE-derived files are git-ignored and are excluded from submission "
        "bundles. Do not work around that - the licence does not permit "
        "redistribution."
    )
