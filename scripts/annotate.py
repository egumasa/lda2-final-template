"""annotate.py — the Google Sheets annotation round-trip.

This is step 2 of the project: you have a sampled subset, and now two of you label it
independently, measure how far apart you were, argue about the rows you disagreed on,
and write down what you decided. That is the part a model cannot do for you, and it
is where you find out whether a wrong label is the MODEL's fault or your SCHEME's.

It is the same round-trip you ran in Day 2 S5, with the same function names:

    create_annotation_sheet(title, items, labels)   ->  a sheet URL
    remembered_sheet(path)                          ->  that URL, next time
    load_annotation_sheet(sheet_id, worksheet)      ->  ONE tab, as row dicts
    load_coder_sheets(sheet_id, coders)             ->  every coder's tab, joined
    annotator_agreement(rows, coders=...)           ->  agreement, kappa, matrix
    disagreements(rows, coders=...)                 ->  your adjudication list
    to_canonical(rows, labels)                      ->  canonical {id, text, label} gold
    compare_to_published(gold, sampled)             ->  where you differ from the source

You do NOT need to edit this file.

ONE TAB PER CODER
-----------------
The sheet has a tab per coder (CoderA, CoderB, ...) and a Final tab to adjudicate in.
Each coder types into their own tab, so nobody is reading a colleague's answer in the
next column along while they decide - which the agreement number below depends on.

Be exact about what that buys, because it is easy to oversell: Google Sheets can stop
somebody EDITING a tab, but anyone who can open the sheet can read every tab in it.
This makes peeking deliberate rather than accidental. The rest is your agreement with
each other, and it is worth saying out loud before you start.

How many coders is decided when you READ the sheet, not when you make it - so a group
that gains a third coder in week two duplicates an empty tab, renames it, and adds the
name to the list it passes to load_coder_sheets. Nothing else changes, and nothing
downstream counts the tabs.

Two coders get percent agreement and Cohen's kappa, as before. Three or more get
Fleiss' kappa for the group, Cohen's kappa for each pair, and the confusion matrix of
the pair that agreed least - which is where the scheme is usually leaking.

Note on working as a group: the SHEET is a real Google Sheets document, so all of you
can annotate at the same time — unlike the repo's files, where concurrent writes
overwrite each other. Annotate together in the sheet; let one person run the notebook.
"""

import json
import pathlib

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from pipeline import plot_confusion_matrix

# Sheet column headers (the annotation template uses these exact names):
COL_ID, COL_TEXT = "ID", "Text"
COL_A, COL_B = "CoderA", "CoderB"
COL_LABEL = "Label"
COL_FINAL, COL_NOTES = "Final", "Note"
COL_CONTEXT = "Context"
ANNOTATION_HEADER = [COL_ID, COL_TEXT, COL_A, COL_B, COL_FINAL, COL_NOTES]
# Context is added on the END, and only for tracks that carry it (the rhetorical-move
# ones). Last, because it is a long cell: put it before CoderA and the columns you
# actually type in get pushed off the side of the screen.

# ONE TAB PER CODER, and a shared tab to adjudicate in.
#
# Each coder types into their own tab, which has just one label column - so nobody is
# looking at a colleague's answer in the next column along while they decide. Be honest
# with yourselves about what that does and does not buy: Google Sheets can stop somebody
# EDITING a tab, but anyone who can open the sheet can read every tab in it. This makes
# peeking deliberate rather than accidental. The rest is your agreement with each other.
CODER_HEADER = [COL_ID, COL_TEXT, COL_LABEL, COL_NOTES]
FINAL_HEADER = [COL_ID, COL_TEXT, COL_FINAL, COL_NOTES]
FINAL_TAB = "Final"
DEFAULT_CODERS = ("CoderA", "CoderB")


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
                            overwrite=False, coders=DEFAULT_CODERS):
    """Create a Sheet in YOUR Drive: one TAB per coder, plus a tab to adjudicate in.

    `items` are {"id", "text", ...} dicts - any existing label is deliberately NOT
    copied across, so you annotate blind. Returns the sheet URL.

    `share_with` is a list of Google account addresses - your group, from MEMBERS in
    config.yaml. The sheet is created in the Drive of whoever runs this cell, so without
    this the second coder cannot open it, and the agreement step needs two coders.

    `remember` is a path to write the URL to, so the notebook can find the sheet again
    without anyone pasting a link.

    `coders` names the tabs. Two is the usual number and the default. You do NOT have to
    decide this now: to add a third coder later, duplicate a tab in the Sheets UI
    (right-click the tab -> Duplicate), rename it, and add the name to the list you pass
    when you READ the sheet back. Nothing downstream counts the tabs.

        Duplicate a tab that is still EMPTY. Copying a tab somebody has already
        filled in gives the new coder their answers, and two coders who agree
        perfectly because one is a photocopy of the other produce a kappa near 1.0
        that means nothing. `load_coder_sheets` looks for that and says so, but it
        is much easier not to do it.

    Items that carry a `context` (the rhetorical-move tracks) get one extra column
    showing the passage, so every coder judges the sentence on the same evidence the
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

    ### Step 0: whose tabs are we making? ###
    coder_names = _normalise_coder_names(
        coders,
        "The list of coder names is empty, so there would be no tab to annotate in.\n"
        "Open config.yaml and put your group's coders in `members:`, or type the names "
        'into this call: coders=["CoderA", "CoderB"]')
    if FINAL_TAB in coder_names:
        raise ValueError(
            "One of your coders is called '" + FINAL_TAB + "', which is also the name of "
            "the tab you adjudicate in - so one would overwrite the other.\n"
            "Open config.yaml and change that name in `members:` to something else, "
            "then re-run the SETUP cell and this one.")

    ### Step 1: make an empty spreadsheet in your own Drive ###
    sheet = _sheets_client().create(title)

    ### Step 2: is this a track that carries context? ###
    with_context = any(item.get("context") for item in items)   # All-or-nothing within a track.

    ### Step 3: one tab per coder, then the tab you adjudicate in ###
    number_of_rows = _write_all_tabs(sheet, coder_names, items, with_context)

    ### Step 4: let the rest of your group in ###
    shared = _share_sheet(sheet, share_with)

    ### Step 5: write the URL down, so nobody has to keep it in a notebook cell ###
    if remember is not None:
        _remember_url(remember, sheet.url, title, coder_names)

    _announce_sheet(title, sheet.url, number_of_rows, coder_names, labels, with_context,
                    shared, remember)
    return sheet.url


def _write_tab(worksheet, columns, items, with_context):
    """Fill one tab: the header, one row per item, and make it readable.

    Every tab is the same shape apart from its one label column, so this is written
    once. The id and the text are filled in; the columns you type into are blank.
    """
    header = list(columns)
    if with_context:
        header = header + [COL_CONTEXT]

    rows = []
    blanks = [""] * (len(columns) - 2)          # every column after ID and Text
    for item in items:
        row = [item["id"], item["text"]] + list(blanks)
        if with_context:
            row.append(marked_context(item))    # The passage, this sentence marked.
        rows.append(row)

    # value_input_option="RAW" tells Sheets to store the text EXACTLY as given.
    # Without it, a sentence starting with "=", "+", "-" or "'" can be read as a
    # formula and mangled - which happens for real in learner-error and
    # move-annotation data.
    worksheet.update([header] + rows, value_input_option="RAW")
    worksheet.freeze(rows=1)                    # header stays put as you scroll
    if with_context:
        # Without this the passage is one clipped line you can only read in the
        # formula bar, which is a good way to make sure nobody reads it.
        last_column = chr(ord("A") + len(header) - 1)
        worksheet.format(last_column + "2:" + last_column,
                         {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"})
        worksheet.columns_auto_resize(0, len(header) - 2)
    return len(rows)


def _write_all_tabs(sheet, coder_names, items, with_context):
    """One tab per coder, then Final. Returns how many rows each tab got.

    The first coder reuses the spreadsheet's default tab; the rest are added. Final goes
    last so it sits on the right, out of the way while you are still annotating blind -
    and so nobody fills it in before the two of you have talked.
    """
    height = len(items) + 1
    first_tab = sheet.sheet1
    first_tab.update_title(coder_names[0])
    number_of_rows = _write_tab(first_tab, CODER_HEADER, items, with_context)
    for name in coder_names[1:]:
        new_tab = sheet.add_worksheet(title=name, rows=height,
                                      cols=len(CODER_HEADER) + 1)
        _write_tab(new_tab, CODER_HEADER, items, with_context)
    final_tab = sheet.add_worksheet(title=FINAL_TAB, rows=height,
                                    cols=len(FINAL_HEADER) + 1)
    _write_tab(final_tab, FINAL_HEADER, items, with_context)
    return number_of_rows


def _share_sheet(sheet, share_with):
    """Give the rest of the group edit access. Returns the addresses that worked.

    The sheet was created in the Drive of whoever ran the cell. Everyone else gets
    "you need access" until they are named here.
    """
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
    return shared


def _remember_url(remember, url, title, coder_names):
    """Write the sheet's link to a file, so nobody has to keep it in a notebook cell."""
    path = pathlib.Path(remember)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "coders": coder_names}, f,
                  ensure_ascii=False, indent=2)


def _announce_sheet(title, url, number_of_rows, coder_names, labels, with_context,
                    shared, remember):
    """Say what was made and how to annotate in it."""
    print("Created '" + title + "' with", number_of_rows, "rows.")
    print("Tabs:", " · ".join(coder_names + [FINAL_TAB]))
    print("Allowed labels:", ", ".join(labels))
    print("Each coder types in their OWN tab, so nobody is reading a colleague's answer")
    print("in the next column while they decide. Adjudicate in '" + FINAL_TAB + "', but")
    print("only once everyone has finished.")
    print("A third coder later? Duplicate an EMPTY tab (right-click the tab ->")
    print("Duplicate), rename it, and add the name where you read the sheet back.")
    if with_context:
        print("This track carries context: the 'Context' column shows each sentence's "
              "passage, with the one you are labelling marked '>>>'. Read it.")
    if shared:
        print("Shared (edit access) with:", ", ".join(shared))
    else:
        print("NOT shared with anyone: this sheet is in your Drive only, so your second "
              "coder cannot open it. Share it by hand now (the Share button, top "
              "right), and put your group's Google accounts in MEMBERS in config.yaml so "
              "next time it is done for you. Do NOT re-run this cell to fix it - that "
              "makes a second, empty sheet.")
    if remember is not None:
        print("Wrote the link to", remember, "- the next step finds it there.")
    print("Open it:", url)


def _normalise_coder_names(coders, if_empty):
    """Tidy a list of coder names: strip the spaces, drop the blanks and the repeats.

    Every place that takes a list of coders needs the same three things done to it, and
    a name typed with a trailing space is otherwise a tab nobody can find.
    """
    names = []
    for name in coders:
        name = str(name).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError(if_empty)
    return names


def _open_sheet(sheet_id):
    """Open the spreadsheet. A pasted URL and a bare id both work."""
    client = _sheets_client()
    if str(sheet_id).startswith("http"):
        return client.open_by_url(sheet_id)
    return client.open_by_key(sheet_id)


def tab_names(sheet_id):
    """The names of the tabs in your annotation sheet, as a list of strings."""
    names = []
    for worksheet in _open_sheet(sheet_id).worksheets():
        names.append(worksheet.title)
    return names


def load_annotation_sheet(sheet_id, worksheet=DEFAULT_CODERS[0]):
    """Read ONE TAB of your annotation sheet back as a list of row dicts.

    `sheet_id` is the long id in the sheet's URL:
        docs.google.com/spreadsheets/d/<THIS PART>/edit
    Pasting the whole URL works too - either way opens the exact sheet, so two copies
    that share a name ("Copy of ...") are never confused.

    `worksheet` is the TAB name, which is now a CODER: "CoderA", "CoderB", or "Final".

    Usually you want `load_coder_sheets` instead - it calls this once per coder and
    joins the tabs into one table. Reach for this one to look at a single tab.
    """
    ### Step 1: open the sheet ###
    sheet = _open_sheet(sheet_id)

    ### Step 2: find the tab - and say which tabs exist if it is missing ###
    try:
        ws = sheet.worksheet(worksheet)
    except Exception:
        tabs = []
        for w in sheet.worksheets():
            tabs.append(w.title)
        raise ValueError(
            "Your annotation sheet has no tab called '" + str(worksheet) + "'.\n"
            "The tabs it does have are: " + " · ".join(tabs) + "\n"
            "Either rename a tab in the sheet to match, or change the list of coder "
            "names in the cell above so it matches the tabs you actually made.")

    ### Step 3: read every row as a dict keyed by the header names ###
    try:
        rows = ws.get_all_records()    # [{"ID": 1, "Text": "...", "CoderA": "B1", ...}]
    except Exception:
        # This is almost always a header problem: a duplicated or blank column name.
        raise ValueError(
            "The tab '" + str(worksheet) + "' opened, but its rows could not be read.\n"
            "Nearly always this is the header row (row 1): two columns with the SAME "
            "name, or a column with no name at all.\n"
            "Open that tab and make row 1 read exactly: "
            + " · ".join(ANNOTATION_HEADER) + " (plus " + COL_CONTEXT
            + " on the rhetorical-move tracks). Then run this cell again.")
    print("Read", len(rows), "rows from tab '" + worksheet + "'.")
    return rows


def load_coder_sheets(sheet_id, coders=DEFAULT_CODERS, final=FINAL_TAB):
    """Read every coder's tab, plus the Final tab, and line them up as one table.

        rows = load_coder_sheets(SHEET_ID, ["CoderA", "CoderB"])

    This is the function to use now that each coder has their own tab. It calls
    `load_annotation_sheet` once per tab - the same function, the same call form - and
    joins the results by ID into exactly the shape everything downstream already
    expects: one row per item, with a column per coder plus Final and Note.

    `coders` is a list of TAB NAMES, given here rather than when the sheet was created.
    That is deliberate: your group does not have to know how many coders it has before
    it starts. Gained a third? Duplicate an empty tab, rename it "CoderC", add "CoderC"
    to this list. Nothing else changes.
    """
    coder_names = _normalise_coder_names(
        coders,
        "The list of coder names is empty, so there is no tab to read.\n"
        "Open config.yaml and put your group's coders in `members:`, or type the tab "
        'names into the cell above: CODERS = ["CoderA", "CoderB"]')

    ### Step 1: read each coder's tab, and remember their label for every item ###
    merged = {}          # id -> the row being built
    order = []           # ids, in the order the first coder's tab had them
    for name in coder_names:
        for row in load_annotation_sheet(sheet_id, name):
            item_id = str(row.get(COL_ID, "")).strip()
            if not item_id:
                continue                     # a blank trailing row in the sheet
            if item_id not in merged:
                merged[item_id] = {COL_ID: row.get(COL_ID), COL_TEXT: row.get(COL_TEXT),
                                   COL_FINAL: "", COL_NOTES: ""}
                if row.get(COL_CONTEXT):
                    merged[item_id][COL_CONTEXT] = row.get(COL_CONTEXT)
                order.append(item_id)
            merged[item_id][name] = str(row.get(COL_LABEL, "")).strip()

    ### Step 2: the Final tab - your adjudicated label, and the note behind it ###
    # Ask whether the tab EXISTS before trying to read it. Catching the read failure
    # instead would treat "you have no Final tab yet" and "your Final tab has a broken
    # header" as the same thing - and the second one would then be reported as "that is
    # fine", quietly throwing away a morning of adjudication.
    if final in tab_names(sheet_id):
        for row in load_annotation_sheet(sheet_id, final):
            item_id = str(row.get(COL_ID, "")).strip()
            if item_id in merged:
                merged[item_id][COL_FINAL] = str(row.get(COL_FINAL, "")).strip()
                merged[item_id][COL_NOTES] = str(row.get(COL_NOTES, "")).strip()
    else:
        # The normal state before you adjudicate, so it is a note rather than an error -
        # the agreement step below works without it.
        print("(no '" + final + "' tab yet - that is fine until you adjudicate)")

    rows = []
    for item_id in order:
        rows.append(merged[item_id])

    ### Step 3: did somebody duplicate a tab that was already filled in? ###
    # Copying a colleague's tab hands you their answers. The two columns then agree
    # perfectly, kappa comes out near 1.0, and it reads as excellent reliability rather
    # than as a photocopy. Nothing else downstream can tell the difference, so say it here.
    for first in range(len(coder_names)):
        for second in range(first + 1, len(coder_names)):
            a_name, b_name = coder_names[first], coder_names[second]
            labelled = 0
            same = 0
            for row in rows:
                a_label, b_label = row.get(a_name, ""), row.get(b_name, "")
                if a_label and b_label:
                    labelled = labelled + 1
                    if a_label == b_label:
                        same = same + 1
            if labelled >= 5 and same == labelled:
                print("WARNING:", a_name, "and", b_name, "gave the SAME label to all",
                      labelled, "items they both labelled.")
                print("         Two people annotating independently essentially never do")
                print("         that. Did one tab get duplicated from the other after it")
                print("         was filled in? If so their agreement is a copy, not a")
                print("         measurement, and the kappa below means nothing.")

    print("Merged", len(rows), "items from", len(coder_names), "coder tab(s):",
          " · ".join(coder_names))
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


def fleiss_kappa(label_lists):
    """Fleiss' kappa: agreement among THREE OR MORE annotators, as one number.

    `label_lists` is one list per annotator, all the same length, all labelling the same
    items in the same order.

    Cohen's kappa compares exactly two people, so with three coders there is no single
    Cohen's number to report - you get one per pair. Fleiss' answers the other question:
    how much do the whole group agree, over and above what people picking at random in
    the same proportions would manage? Read it on the same scale you read Cohen's.

    Written out here because scikit-learn does not provide it. The formula:

        P(i)    for each item, the share of annotator PAIRS on that item who agree
        P_bar   the average of those - observed agreement
        P_e     what you would expect from chance, given how often each label is used
        kappa   (P_bar - P_e) / (1 - P_e)
    """
    number_of_annotators = len(label_lists)
    number_of_items = len(label_lists[0])
    if number_of_annotators < 2 or number_of_items == 0:
        return float("nan")

    labels = set()
    for one_annotator in label_lists:
        for label in one_annotator:
            labels.add(label)
    labels = sorted(labels)
    if len(labels) < 2:
        return float("nan")          # everyone used one label: undefined, not zero

    ### Step 1: for each item, count how many annotators chose each label ###
    counts_per_item = []
    for position in range(number_of_items):
        counts = {}
        for label in labels:
            counts[label] = 0
        for one_annotator in label_lists:
            counts[one_annotator[position]] = counts[one_annotator[position]] + 1
        counts_per_item.append(counts)

    ### Step 2: observed agreement - the share of annotator pairs that agree ###
    pairs_per_item = number_of_annotators * (number_of_annotators - 1)
    agreement_total = 0.0
    for counts in counts_per_item:
        agreeing_pairs = 0
        for label in labels:
            agreeing_pairs = agreeing_pairs + counts[label] * (counts[label] - 1)
        agreement_total = agreement_total + agreeing_pairs / pairs_per_item
    observed = agreement_total / number_of_items

    ### Step 3: chance agreement - how often each label was used overall ###
    expected = 0.0
    total_judgements = number_of_items * number_of_annotators
    for label in labels:
        used = 0
        for counts in counts_per_item:
            used = used + counts[label]
        share = used / total_judgements
        expected = expected + share * share

    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def _draw_coder_matrix(a_labels, b_labels, name_a, name_b, title):
    """One coder's labels against another's, as a heatmap. The diagonal is agreement."""
    labels = sorted(set(a_labels) | set(b_labels))
    matrix = confusion_matrix(a_labels, b_labels, labels=labels)
    plot_confusion_matrix(matrix, labels, title, xlabel=name_b, ylabel=name_a)


def _coder_columns(rows, coders):
    """Which columns hold labels, and every row where ALL of them are filled in."""
    names = _normalise_coder_names(
        coders,
        "The list of coder names is empty, so there are no columns to compare.\n"
        'Set CODERS in the cell above, e.g. CODERS = ["CoderA", "CoderB"]')

    complete = []
    for row in rows:
        labels = []
        for name in names:
            labels.append(str(row.get(name, "")).strip())
        if all(labels):                     # drop half-finished rows, as with two coders
            complete.append(labels)
    return names, complete


def _labels_by_coder(names, complete):
    """Turn one list of labels per ROW into one list of labels per CODER."""
    by_coder = []
    for position in range(len(names)):
        one = []
        for labels in complete:
            one.append(labels[position])
        by_coder.append(one)
    return by_coder


def _kappa_of_pair(a_labels, b_labels):
    """Cohen's kappa for two coders, or nan when they only ever used one label."""
    if len(set(a_labels) | set(b_labels)) < 2:
        return float("nan")
    return cohen_kappa_score(a_labels, b_labels)


def _pairwise_kappas(names, by_coder):
    """Cohen's kappa for every pair of coders, printed and returned.

    Also returns which pair agreed LEAST, because "who diverges from whom" is the
    question that leads somewhere - a single group-wide number does not tell you which
    label boundary to go and argue about.
    """
    print("  pairwise Cohen's kappa:")
    pairwise = {}
    worst_pair = None
    worst_kappa = None
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            kappa = _kappa_of_pair(by_coder[first], by_coder[second])
            pair = names[first] + " - " + names[second]
            pairwise[pair] = kappa
            print("   ", pair, " ", format(kappa, ".3f"))
            # kappa == kappa is False only when kappa is nan - that is, when this pair
            # used a single label between them and there is nothing to rank.
            is_a_real_number = kappa == kappa
            if is_a_real_number and (worst_kappa is None or kappa < worst_kappa):
                worst_kappa = kappa
                worst_pair = (first, second)
    return pairwise, worst_pair


def _agreement_for_many(rows, coders):
    """Fleiss' kappa across the group, plus Cohen's kappa for every pair."""
    names, complete = _coder_columns(rows, coders)
    if not complete:
        print("No rows where ALL", len(names), "coders have labelled. Nothing to compare "
              "yet.")
        return None

    by_coder = _labels_by_coder(names, complete)

    ### The whole group, as one number ###
    overall = fleiss_kappa(by_coder)
    print(len(complete), "items labelled by all", len(names), "coders")
    print("  Fleiss' kappa (all coders):", format(overall, ".3f"))

    ### Then pair by pair ###
    pairwise, worst_pair = _pairwise_kappas(names, by_coder)

    ### One matrix, for the pair that agrees least - that is where the scheme leaks ###
    if worst_pair is not None:
        first, second = worst_pair
        _draw_coder_matrix(by_coder[first], by_coder[second], names[first], names[second],
                           "Least agreement: " + names[first] + " vs " + names[second])
        print("  The matrix above is your LEAST agreeing pair - the label boundary to "
              "argue about first.")

    # "kappa" is here under that name so code written against the two-coder version, and
    # the tutorials, keeps working. For a group it is the Fleiss number.
    return {"n": len(complete), "coders": names, "kappa": overall,
            "fleiss_kappa": overall, "pairwise_kappa": pairwise}


def annotator_agreement(rows, a=COL_A, b=COL_B, coders=None):
    """Percent agreement + Cohen's kappa between the two annotator columns, PLUS an
    annotator-vs-annotator confusion matrix (the diagonal is where you agreed;
    off-diagonal cells show which label pairs the two of you confuse).

    `coders` is for groups of three or more: pass the list of coder names and you get
    Fleiss' kappa for the group, then Cohen's kappa for every pair, then the confusion
    matrix of the pair that agreed least. With two coders - named or not - the output is
    exactly what it has always been.
    """
    if coders is not None and len(coders) > 2:
        return _agreement_for_many(rows, coders)
    if coders is not None and len(coders) == 2:
        a, b = coders[0], coders[1]
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
    # The diagonal is where you agreed; everything off it is a boundary to talk about.
    _draw_coder_matrix(a_labels, b_labels, str(a), str(b),
                       "Annotator-vs-annotator confusion matrix")
    # "kappa" and "cohen_kappa" are the same number under two names, so code written
    # against either the tutorials or metrics.agreement() keeps working.
    return {"n": len(a_labels), "percent_agreement": percent,
            "kappa": kappa, "cohen_kappa": kappa}


def disagreements(rows, a=COL_A, b=COL_B, coders=None):
    """The rows your annotators labelled differently - your adjudication list.

    `coders` is for groups of three or more: a row needs adjudicating when they do not
    ALL agree. With two coders the behaviour is unchanged.
    """
    if coders is not None and len(coders) > 2:
        out = _disagreements_for_many(rows, coders)
    else:
        if coders is not None and len(coders) == 2:
            a, b = coders[0], coders[1]
        out = _disagreements_for_two(rows, a, b)
    print(len(out), "rows to adjudicate. Agree on a `Final` label for each in the sheet.")
    # Name the columns even when no rows come back, so that a group whose coders agreed
    # on everything gets an empty table rather than a table with nothing in it at all.
    return pd.DataFrame(out, columns=list(rows[0]) if rows else None)


def _disagreements_for_two(rows, a, b):
    """The rows where two coders both labelled, and chose differently."""
    out = []
    for row in rows:
        label_a = str(row.get(a, "")).strip()
        label_b = str(row.get(b, "")).strip()
        if label_a and label_b and label_a != label_b:
            out.append(row)
    return out


def _disagreements_for_many(rows, coders):
    """The rows where everyone labelled, and they did not all choose the same label."""
    names, _ = _coder_columns(rows, coders)
    out = []
    for row in rows:
        labels = []
        for name in names:
            labels.append(str(row.get(name, "")).strip())
        everyone_labelled = all(labels)
        they_differ = len(set(labels)) > 1
        if everyone_labelled and they_differ:
            out.append(row)
    return out


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
