"""annotate.py — the Google Sheets annotation round-trip.

This is step 2 of the project: you have a sampled subset, and now two of you label it
independently, measure how far apart you were, argue about the rows you disagreed on,
and write down what you decided. That is the part a model cannot do for you, and it
is where you find out whether a wrong label is the MODEL's fault or your SCHEME's.

It is the same round-trip you ran in Day 2 S5, with the same function names:

    create_annotation_sheet(title, items, labels)   ->  a sheet URL
    remembered_sheet(path)                          ->  that URL, next time
    load_annotation_sheet(sheet_id, worksheet)      ->  a list of row dicts
    annotator_agreement(rows)                       ->  percent agreement, kappa, matrix
    disagreements(rows)                             ->  your adjudication list
    to_canonical(rows, labels)                      ->  canonical {id, text, label} gold
    compare_to_published(gold, sampled)             ->  where you differ from the source

You do NOT need to edit this file.

Note on working as a group: the SHEET is a real Google Sheets document, so all of you
can annotate it at the same time — unlike the repo's files, where concurrent writes
overwrite each other. Annotate together in the sheet; let one person run the notebook.
"""

import json
import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Sheet column headers (the annotation template uses these exact names):
COL_ID, COL_TEXT = "ID", "Text"
COL_A, COL_B = "CoderA", "CoderB"
COL_FINAL, COL_NOTES = "Final", "Note"
COL_CONTEXT = "Context"
ANNOTATION_HEADER = [COL_ID, COL_TEXT, COL_A, COL_B, COL_FINAL, COL_NOTES]
# Context is added on the END, and only for tracks that carry it (the rhetorical-move
# ones). Last, because it is a long cell: put it before CoderA and the columns you
# actually type in get pushed off the side of the screen.


def _sheets_client():
    """Authorise gspread with your Google account (a pop-up asks for permission)."""
    # In Colab, the account you are already signed in with is used directly.
    try:
        from google.colab import auth
        import google.auth
        import gspread
        auth.authenticate_user()           # the pop-up: "let Colab use your Sheets"
        creds, _ = google.auth.default()   # the permission slip that pop-up produced
        return gspread.authorize(creds)    # a logged-in connection to Google Sheets
    except ImportError:
        pass

    # Running locally instead. gspread can do its own browser sign-in, but it needs an
    # OAuth client file first, which is a one-time setup step most people skip.
    import gspread
    try:
        return gspread.oauth()
    except Exception as error:
        raise RuntimeError(
            "Could not sign in to Google Sheets from this machine.\n"
            "The annotation step is designed to run in Google Colab, where your "
            "Google account is already available - open the notebook there and this "
            "will just work.\n"
            "To do it locally instead you need a gspread OAuth client file; see "
            "https://docs.gspread.org/en/latest/oauth2.html\n"
            "Original error: " + str(error)
        ) from error


def marked_context(item):
    """The item's passage, numbered, with the sentence being judged marked `>>>`.

    Only for display in the sheet - the item's own `context` stays untouched. A move is
    a function within a passage, so a coder judging one sentence needs to see the rest;
    but handing them 26 unbroken sentences and asking "which one was it again?" trades
    one problem for another. Returns "" for tracks that carry no context.
    """
    context = item.get("context")
    if not context:
        return ""
    target = item.get("sent_index")
    lines = []
    for position, sentence in enumerate(context.splitlines()):
        number = str(position + 1) + ". "
        if position == target:
            lines.append(">>> " + number + sentence)   # The one you are labelling.
        else:
            lines.append("    " + number + sentence)
    return "\n".join(lines)


def remembered_sheet(path):
    """The sheet URL create_annotation_sheet() wrote down, or "" if there is not one.

    This exists because the sheet id is the only handoff in the project that is not a
    file. Without it, the link to your group's annotation round lives in one person's
    notebook output, and a runtime reset - or simply a different member opening the
    notebook - loses it.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("url", "")
    except FileNotFoundError:
        return ""            # step 2 has not been run yet; that is not an error


def create_annotation_sheet(title, items, labels, share_with=(), remember=None,
                            overwrite=False):
    """Create a Sheet in YOUR Drive: one row per item, blank columns to label.

    `items` are {"id", "text", ...} dicts - any existing label is deliberately NOT
    copied across, so you annotate blind. Returns the sheet URL.

    `share_with` is a list of Google account addresses - your group, from MEMBERS in
    config.yaml. The sheet is created in the Drive of whoever runs this cell, so without
    this the second coder cannot open it, and step 3 needs two coders.

    `remember` is a path to write the URL to, so the notebook can find the sheet again
    without anyone pasting a link.

    Items that carry a `context` (the rhetorical-move tracks) get one extra column
    showing the passage, so the two coders judge the sentence on the same evidence the
    model will get.
    """
    ### Step 0: is there already a sheet from an earlier run? ###
    # Checked BEFORE the sheet is created, not after. Overwriting the remembered URL
    # would strand the sheet your group has already been annotating in - it would still
    # exist in Drive, but nothing in the project would point at it any more.
    if remember is not None and not overwrite and pathlib.Path(remember).exists():
        raise FileExistsError(
            "\n" + pathlib.Path(remember).name + " already exists, so your group has "
            "made an annotation sheet before.\n"
            "  it points at: " + remembered_sheet(remember) + "\n"
            "\n"
            "Making a new one would lose that link, and any annotation already in that "
            "sheet with it.\n"
            "\n"
            "  * Going back to the sheet you already have? You do not need this cell —\n"
            "    open the link above.\n"
            "  * Starting a fresh round on purpose? Open config.yaml and change\n"
            "    run: to the next version (v1 -> v2), then re-run the SETUP cell.\n"
            "  * Really want to forget the old sheet? Add overwrite=True inside the\n"
            "    brackets of this call."
        )

    ### Step 1: make an empty spreadsheet in your own Drive ###
    sheet = _sheets_client().create(title)
    worksheet = sheet.sheet1
    worksheet.update_title("round1")   # first round lives in the 'round1' tab

    ### Step 2: is this a track that carries context? ###
    with_context = any(item.get("context") for item in items)   # All-or-nothing within a track.
    if with_context:
        header = ANNOTATION_HEADER + [COL_CONTEXT]
    else:
        header = ANNOTATION_HEADER                # The other tracks' sheets are unchanged.

    ### Step 3: one row per item - id and text filled in, label columns left blank ###
    rows = []
    for item in items:
        #      id            text          CoderA CoderB Final Note
        row = [item["id"], item["text"], "", "", "", ""]
        if with_context:
            row.append(marked_context(item))      # The passage, with this sentence marked.
        rows.append(row)

    ### Step 4: write it all in one go, then pin the header row ###
    # value_input_option="RAW" tells Sheets to store the text EXACTLY as given. Without
    # it, a sentence starting with "=", "+", "-" or "'" can be read as a formula and
    # mangled - which happens for real in learner-error and move-annotation data.
    worksheet.update([header] + rows, value_input_option="RAW")
    worksheet.freeze(rows=1)                       # header stays put as you scroll
    if with_context:
        # Without this the passage is one clipped line you can only read in the formula
        # bar, which is a good way to make sure nobody reads it.
        last_column = chr(ord("A") + len(header) - 1)
        worksheet.format(last_column + "2:" + last_column,
                         {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"})
        worksheet.columns_auto_resize(0, len(header) - 2)
    ### Step 5: let the rest of your group in ###
    # The sheet was created in the Drive of whoever ran this cell. Everyone else gets
    # "you need access" until they are named here.
    shared = []
    for address in share_with:
        address = str(address).strip()
        if not address:
            continue
        try:
            sheet.share(address, perm_type="user", role="writer")
            shared.append(address)
        except Exception as error:
            # One bad address should not cost you the other invitations.
            print("  could not share with", address + ":", error)

    ### Step 6: write the URL down, so nobody has to keep it in a notebook cell ###
    if remember is not None:
        path = pathlib.Path(remember)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"url": sheet.url, "title": title, "rounds": ["round1"]}, f,
                      ensure_ascii=False, indent=2)

    print("Created '" + title + "' with", len(rows), "rows in tab 'round1'.")
    print("Allowed labels:", ", ".join(labels))
    if with_context:
        print("This track carries context: the 'Context' column shows each sentence's "
              "passage, with the one you are labelling marked '>>>'. Read it.")
    if shared:
        print("Shared (edit access) with:", ", ".join(shared))
    else:
        print("NOT shared with anyone: this sheet is in your Drive only, so your second "
              "coder cannot open it. Share it by hand now (the Share button, top "
              "right), and put your group's Google accounts in MEMBERS in config.yaml so "
              "the next round does it for you. Do NOT re-run this cell to fix it - that "
              "makes a second, empty sheet.")
    if remember is not None:
        print("Wrote the link to", remember, "- the next step finds it there.")
    print("Open it:", sheet.url)
    return sheet.url


def load_annotation_sheet(sheet_id, worksheet="round1"):
    """Read one TAB of your annotation sheet back as a list of row dicts.

    `sheet_id` is the long id in the sheet's URL:
        docs.google.com/spreadsheets/d/<THIS PART>/edit
    Pasting the whole URL works too - either way opens the exact sheet, so two copies
    that share a name ("Copy of ...") are never confused.

    `worksheet` is the TAB name (a "round"): each round lives in its own tab, so
    re-annotating in round2 never overwrites round1 - the analysis stays reproducible.
    """
    ### Step 1: open the sheet - a pasted URL and a bare id both work ###
    client = _sheets_client()
    if str(sheet_id).startswith("http"):
        sheet = client.open_by_url(sheet_id)
    else:
        sheet = client.open_by_key(sheet_id)

    ### Step 2: find the tab (the "round") - and say which tabs exist if it is missing ###
    try:
        ws = sheet.worksheet(worksheet)
    except Exception:
        tabs = []
        for w in sheet.worksheets():
            tabs.append(w.title)
        raise ValueError("No tab named " + repr(worksheet)
                         + ". Tabs in this sheet: " + str(tabs))

    ### Step 3: read every row as a dict keyed by the header names ###
    try:
        rows = ws.get_all_records()    # [{"ID": 1, "Text": "...", "CoderA": "B1", ...}]
    except Exception as error:
        # This is almost always a header problem: a duplicated or blank column name.
        raise ValueError(
            "Could not read the rows of tab " + repr(worksheet) + ". This usually means "
            "the header row has a DUPLICATE or BLANK column name. The columns must "
            "include: " + " · ".join(ANNOTATION_HEADER) + " (plus " + COL_CONTEXT
            + " on the rhetorical-move tracks) - fix the header in the sheet and "
            "re-run.\nOriginal error: " + str(error)
        ) from error
    print("Read", len(rows), "rows from tab '" + worksheet + "'.")
    return rows


def to_canonical(rows, labels, column=COL_FINAL, source=None):
    """Turn annotation rows into canonical gold: [{"id", "text", "label"}, ...].

    Blank rows are skipped; labels outside `labels` are reported, not silently kept.

    `source` is the list of items the sheet was BUILT from (your sampled items). Pass it
    on a track that carries context: gold is rebuilt from the sheet, which holds only the
    id, the text and your label, so anything else the item was carrying would be dropped
    here and notebook 04 would never see it. The extra fields are copied from `source` by
    id rather than read back out of the sheet - the sheet's Context column is a
    marked-up display copy, and a coder may have edited it.
    """
    ### Step 0: look up what each sampled item was carrying, if we were given them ###
    extras_by_id = {}
    if source is not None:
        for item in source:
            extras = {}
            for key in item:
                if key not in ("id", "text", "label"):
                    extras[key] = item[key]
            extras_by_id[item["id"]] = extras

    ### Step 1: sort every row into one of three piles ###
    gold = []          # usable rows
    blank = 0          # not labelled yet
    invalid = []       # typos, wrong case, labels that are not in the scheme
    bad_ids = []       # rows whose ID cell is not a number
    for row in rows:
        label = str(row.get(column, "")).strip()   # .strip() drops stray spaces
        if not label:
            blank = blank + 1                      # nobody has filled this row in yet
        elif label not in labels:
            invalid.append((row.get(COL_ID), label))   # e.g. "b1" or "B11"
        else:
            # The ID column should hold the number the sheet was created with. If
            # someone has typed over it, say which row rather than crash.
            try:
                item_id = int(row[COL_ID])
            except (KeyError, TypeError, ValueError):
                bad_ids.append(row.get(COL_ID))
                continue
            gold_item = {
                "id": item_id,
                "text": str(row[COL_TEXT]),
                "label": label,
            }
            gold_item.update(extras_by_id.get(item_id, {}))   # Put back whatever the sheet could not carry.
            gold.append(gold_item)

    ### Step 2: report every count, so nothing is dropped silently ###
    print(len(gold), "usable ·", blank, "still blank ·", len(invalid), "invalid")
    if invalid:
        print("  fix these in the sheet, then re-run:", invalid[:10])   # first 10
        print("  allowed labels:", ", ".join(labels))
    if bad_ids:
        print("  these rows have a non-numeric ID cell (did something get typed over "
              "it?):", bad_ids[:10])

    ### Step 3: if we were given the sampled items, check they actually matched ###
    # A silent miss here is nasty: gold would come out looking fine, just without the
    # context, and only notebook 04 would notice - by prompting with an empty passage.
    if source is not None:
        unmatched = 0
        for item in gold:
            if item["id"] not in extras_by_id:
                unmatched = unmatched + 1
        if unmatched:
            print("  WARNING:", unmatched, "row(s) had no match in `source` by id, so "
                  "they carry no context. Is `source` the same sampled items this sheet "
                  "was created from?")
    return gold


def annotator_agreement(rows, a=COL_A, b=COL_B):
    """Percent agreement + Cohen's kappa between the two annotator columns, PLUS an
    annotator-vs-annotator confusion matrix (the diagonal is where you agreed;
    off-diagonal cells show which label pairs the two of you confuse).
    """
    ### Step 1: keep only the rows where BOTH annotators actually chose a label ###
    a_labels = []
    b_labels = []
    for row in rows:
        label_a = str(row.get(a, "")).strip()
        label_b = str(row.get(b, "")).strip()
        if label_a and label_b:            # drop half-finished rows
            a_labels.append(label_a)
            b_labels.append(label_b)
    if len(a_labels) == 0:
        print("No rows where BOTH annotators have labelled. Nothing to compare yet.")
        return None

    ### Step 2: two metrics - raw agreement, and agreement corrected for chance ###
    number_of_matches = 0
    for i in range(len(a_labels)):
        if a_labels[i] == b_labels[i]:
            number_of_matches = number_of_matches + 1
    percent = number_of_matches / len(a_labels)

    # Kappa needs at least two distinct labels between you; with only one it is
    # undefined, which is a property of the data rather than a bug.
    labels_used = set(a_labels) | set(b_labels)
    if len(labels_used) < 2:
        kappa = float("nan")
        print(len(a_labels), "doubly-annotated · agreement", format(percent, ".1%"),
              "· Cohen's kappa undefined (only one label used)")
    else:
        kappa = cohen_kappa_score(a_labels, b_labels)
        print(len(a_labels), "doubly-annotated · agreement", format(percent, ".1%"),
              "· Cohen's kappa", format(kappa, ".3f"))

    ### Step 3: draw WHICH labels you two confuse, not just how often ###
    labels = sorted(labels_used)           # every label either of you used
    matrix = confusion_matrix(a_labels, b_labels, labels=labels)
    plt.figure(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Annotator B")
    plt.ylabel("Annotator A")              # the diagonal is where you agreed
    plt.title("Annotator-vs-annotator confusion matrix")
    plt.tight_layout()
    plt.show()
    # "kappa" and "cohen_kappa" are the same number under two names, so code written
    # against either the tutorials or metrics.agreement() keeps working.
    return {"n": len(a_labels), "percent_agreement": percent,
            "kappa": kappa, "cohen_kappa": kappa}


def disagreements(rows, a=COL_A, b=COL_B):
    """The rows your two annotators labelled differently - your adjudication list."""
    out = []
    for row in rows:
        label_a = str(row.get(a, "")).strip()
        label_b = str(row.get(b, "")).strip()
        # keep a row only if both annotators labelled it AND they chose differently
        if label_a and label_b and label_a != label_b:
            out.append(row)
    print(len(out), "rows to adjudicate. Agree on a `Final` label for each in the sheet.")
    return pd.DataFrame(out)


def compare_to_published(gold, published):
    """How often does YOUR final label match the published one, item by item?

    Items are matched by their TEXT, not their id. Sampling renumbers the ids from 1,
    so an id-based match would line YOUR item 7 up against POOL item 7 - two unrelated
    sentences - and report a meaningless number without ever failing. (Ids are still
    used as a fallback, for the case where the texts have been edited.)
    """
    ### Step 1: look up the published label for every text ###
    label_by_text = {}
    label_by_id = {}
    for item in published:
        label_by_text[str(item["text"])] = item["label"]
        label_by_id[item["id"]] = item["label"]

    ### Step 2: pair each of your items with its published label ###
    matched_rows = []
    matched_by_id_only = 0
    for item in gold:
        text = str(item["text"])
        if text in label_by_text:
            theirs = label_by_text[text]
        elif item["id"] in label_by_id:
            theirs = label_by_id[item["id"]]
            matched_by_id_only = matched_by_id_only + 1
        else:
            continue
        matched_rows.append({
            "id": item["id"],
            "yours": item["label"],
            "published": theirs,
            "text": item["text"],
        })

    if len(matched_rows) == 0:
        print("None of your items could be matched to the published set. Are you "
              "comparing against the same data you sampled from?")
        return None
    if matched_by_id_only > 0:
        print("  note:", matched_by_id_only, "item(s) matched by id because the text "
              "no longer matches exactly.")

    ### Step 3: count the matches, then show only the rows where you differ ###
    agree = 0
    differences = []
    for row in matched_rows:
        if row["yours"] == row["published"]:
            agree = agree + 1
        else:
            differences.append(row)
    print(agree, "/", len(matched_rows), "match the published label",
          "(" + format(agree / len(matched_rows), ".1%") + ")")
    return pd.DataFrame(differences)
