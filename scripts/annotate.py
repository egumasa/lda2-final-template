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

# The sheet's column names and the two functions that read a judgment out of it live in
# _study.py, which is the one copy the notebook generators render into cells. They are
# re-exported here so that `from annotate import disagreements` - the Day 2 S5 call -
# keeps working unchanged.
from _study import (COL_ID, COL_TEXT, COL_A, COL_B, COL_FINAL, COL_NOTES,
                    column, percent_agreement, disagreements)

# Sheet column headers (the annotation template uses these exact names):
COL_LABEL = "Label"
COL_CONTEXT = "Context"
COL_SOURCE = "SourceID"
ANNOTATION_HEADER = [COL_ID, COL_TEXT, COL_A, COL_B, COL_FINAL, COL_NOTES]
# Context is added after these, and only for tracks that carry it (the rhetorical-move
# ones). Late, because it is a long cell: put it before CoderA and the columns you
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
# SourceID goes on the far right, after Context: it is the id the item had in the pool,
# which nothing types into and nothing reads back out of the sheet. It is there so a row
# can be traced to the pool row it came from, and so notebook 02b can add more items
# without drawing one you already have. Keep the column; you never need to fill it in.
OWNED_COLUMNS = [COL_ID, COL_TEXT, COL_CONTEXT, COL_SOURCE]
FINAL_TAB = "Final"
DEFAULT_CODERS = ("CoderA", "CoderB")


def _sheets_client():
    """Authorise gspread with your Google account (a pop-up asks for permission).

    Returns:
        A logged-in connection to Google Sheets.

    Raises:
        RuntimeError: when signing in from a local machine fails. The annotation
            step is designed to run in Colab.
    """
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


def marked_context(item: dict[str, str]) -> str:
    """The item's passage, numbered, with the sentence being judged marked `>>>`.

    Only for display in the sheet - the item's own `context` stays untouched. A move is
    a function within a passage, so a coder judging one sentence needs to see the rest;
    but handing them 26 unbroken sentences and asking "which one was it again?" trades
    one problem for another.

    Args:
        item: one item, which may carry a "context" and a "sent_index".

    Returns:
        The numbered passage, or "" for tracks that carry no context.
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


def remembered_sheet(path: str) -> str:
    """The sheet URL create_annotation_sheet() wrote down, or "" if there is not one.

    This exists because the sheet id is the only handoff in the project that is not a
    file. Without it, the link to your group's annotation round lives in one person's
    notebook output, and a runtime reset - or simply a different member opening the
    notebook - loses it.

    Args:
        path: the file create_annotation_sheet wrote the URL to.

    Returns:
        The URL, or "" when step 2 has not been run yet.

    Example:
        >>> remembered_sheet(SHEET_PATH)
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("url", "")
    except FileNotFoundError:
        return ""            # step 2 has not been run yet; that is not an error


def create_annotation_sheet(title: str,
                            items: list[dict[str, str]],
                            labels: list[str],
                            share_with: list[str] | tuple = (),
                            remember: str | None = None,
                            overwrite: bool = False,
                            coders: list[str] | tuple = DEFAULT_CODERS) -> str:
    """Create a Sheet in YOUR Drive: one TAB per coder, plus a tab to adjudicate in.

    Items that carry a `context` (the rhetorical-move tracks) get one extra column
    showing the passage, so every coder judges the sentence on the same evidence the
    model will get.

    To add a third coder later, duplicate a tab in the Sheets UI (right-click the tab ->
    Duplicate), rename it, and add the name to the list you pass when you READ the sheet
    back. Duplicate a tab that is still EMPTY: copying one somebody has already filled in
    gives the new coder their answers, and two coders who agree perfectly because one is
    a copy of the other produce a kappa near 1.0 that means nothing.

    Args:
        title: the name to give the new spreadsheet.
        items: the items to annotate, each with "id" and "text". Any existing label
            is deliberately NOT copied across, so you annotate blind.
        labels: the labels your scheme allows, printed as a reminder.
        share_with: Google account addresses to give edit access - your group, from
            MEMBERS in config.yaml. The sheet is created in the Drive of whoever runs
            this cell, so without this the second coder cannot open it.
        remember: a path to write the URL to, so the notebook can find the sheet
            again without anyone pasting a link.
        overwrite: True to forget a sheet an earlier run remembered.
        coders: names the tabs, one per coder. Two is the usual number.

    Returns:
        The URL of the sheet it created.

    Raises:
        FileExistsError: when `remember` already points at a sheet and overwrite is
            False. Replacing it would strand the sheet you have been annotating in.
        ValueError: when the coder list is empty, or a coder is named "Final",
            which is also the name of the tab you adjudicate in.

    Example:
        >>> url = create_annotation_sheet(title, items, LABELS_ORDER,
        ...                               share_with=MEMBERS, remember=SHEET_PATH)
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


def _cell_values(item: dict[str, str], with_context: bool) -> dict[str, str]:
    """What this project fills in for one item, keyed by the column's header name.

    These four columns are the ones the code writes: the ones nobody types into. Any
    other column in a tab is either a column you type into or one your group added, and
    neither is ever written to. Used both when the sheet is created and when notebook
    02b adds rows to it, so the two cannot disagree about what belongs where.

    Args:
        item: one item to write a row for.
        with_context: True when this track carries passages.

    Returns:
        The values to write, keyed by column heading.
    """
    values = {COL_ID: item["id"], COL_TEXT: item["text"]}
    if "source_id" in item:
        values[COL_SOURCE] = item["source_id"]
    if with_context:
        values[COL_CONTEXT] = marked_context(item)   # The passage, this sentence marked.
    return values


def _tab_header(columns: list[str], with_context: bool) -> list[str]:
    """The header row for a tab: its own columns, then Context, then SourceID.

    Args:
        columns: the tab's own columns, which differ between a coder tab and Final.
        with_context: True when this track carries passages.

    Returns:
        The column headings, in order.
    """
    header = list(columns)
    if with_context:
        header = header + [COL_CONTEXT]
    return header + [COL_SOURCE]


def _write_tab(worksheet, columns: list[str], items: list[dict[str, str]],
               with_context: bool) -> int:
    """Fill one tab: the header, one row per item, and make it readable.

    Every tab is the same shape apart from its one label column, so this is written
    once. The id, the text and the pool id are filled in; the columns you type into are
    blank. Rows are built by looking each column up by NAME rather than by counting
    positions, which is the same way notebook 02b adds rows later.

    Args:
        worksheet: the gspread tab to fill.
        columns: the tab's own columns.
        items: the items to write one row each for.
        with_context: True when this track carries passages.

    Returns:
        How many rows it wrote.
    """
    header = _tab_header(columns, with_context)

    rows = []
    for item in items:
        values = _cell_values(item, with_context)
        row = []
        for name in header:
            row.append(values.get(name, ""))
        rows.append(row)

    # value_input_option="RAW" tells Sheets to store the text EXACTLY as given.
    # Without it, a sentence starting with "=", "+", "-" or "'" can be read as a
    # formula and mangled - which happens for real in learner-error and
    # move-annotation data.
    worksheet.update([header] + rows, value_input_option="RAW")
    worksheet.freeze(rows=1)                    # header stays put as you scroll
    if with_context:
        # Without this the passage is one clipped line you can only read in the
        # formula bar, which is a good way to make sure nobody reads it. Found by name:
        # Context is no longer the last column, so counting to the end would land on
        # SourceID and leave the passage clipped.
        context_column = _column_letter(header.index(COL_CONTEXT) + 1)
        worksheet.format(context_column + "2:" + context_column,
                         {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"})
        worksheet.columns_auto_resize(0, len(header) - 1)
    return len(rows)


def _column_letter(number: int) -> str:
    """Column number to its letter, 1 -> A, 27 -> AA.

    A sheet with a Context column and a few added columns can pass column Z, where
    chr(ord("A") + n) stops being a column name.

    Args:
        number: the column's position, counting from 1.

    Returns:
        The column letter.
    """
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _write_all_tabs(sheet, coder_names: list[str] | tuple,
                    items: list[dict[str, str]], with_context: bool) -> int:
    """One tab per coder, then Final.

    The first coder reuses the spreadsheet's default tab; the rest are added. Final goes
    last so it sits on the right, out of the way while you are still annotating blind -
    and so nobody fills it in before the two of you have talked.

    Args:
        sheet: the gspread spreadsheet to add tabs to.
        coder_names: one tab name per coder.
        items: the items to write into every tab.
        with_context: True when this track carries passages.

    Returns:
        How many rows each tab got.
    """
    # Room for the tab's own columns plus Context and SourceID, and some spare rows so
    # notebook 02b can add items later without having to grow the grid first.
    height = len(items) + 1
    coder_width = len(_tab_header(CODER_HEADER, with_context))
    final_width = len(_tab_header(FINAL_HEADER, with_context))
    first_tab = sheet.sheet1
    first_tab.update_title(coder_names[0])
    number_of_rows = _write_tab(first_tab, CODER_HEADER, items, with_context)
    for name in coder_names[1:]:
        new_tab = sheet.add_worksheet(title=name, rows=height, cols=coder_width)
        _write_tab(new_tab, CODER_HEADER, items, with_context)
    final_tab = sheet.add_worksheet(title=FINAL_TAB, rows=height, cols=final_width)
    _write_tab(final_tab, FINAL_HEADER, items, with_context)
    return number_of_rows


def _share_sheet(sheet, share_with: list[str] | tuple) -> list[str]:
    """Give the rest of the group edit access.

    The sheet was created in the Drive of whoever ran the cell. Everyone else gets
    "you need access" until they are named here.

    Args:
        sheet: the gspread spreadsheet to share.
        share_with: Google account addresses to invite.

    Returns:
        The addresses that worked. One bad address does not cost you the others.
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


def _remember_url(remember: str, url: str, title: str,
                  coder_names: list[str]) -> None:
    """Write the sheet's link to a file, so nobody has to keep it in a notebook cell.

    Args:
        remember: where to write it.
        url: the sheet's URL.
        title: the sheet's name.
        coder_names: the tab names, so a later step knows what to read.

    Returns:
        Nothing.
    """
    path = pathlib.Path(remember)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "coders": coder_names}, f,
                  ensure_ascii=False, indent=2)


def _announce_sheet(title: str, url: str, number_of_rows: int,
                    coder_names: list[str], labels: list[str], with_context: bool,
                    shared: list[str], remember: str | None) -> None:
    """Say what was made and how to annotate in it.

    Args:
        title: the sheet's name.
        url: the sheet's URL.
        number_of_rows: how many rows each tab got.
        coder_names: the tab names.
        labels: the labels the scheme allows.
        with_context: True when this track carries passages.
        shared: the addresses the sheet was shared with.
        remember: where the URL was written, or None.

    Returns:
        Nothing. It prints how to annotate and where the sheet is.
    """
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


def _normalise_coder_names(coders: list[str] | tuple, if_empty: str) -> list[str]:
    """Tidy a list of coder names: strip the spaces, drop the blanks and the repeats.

    Every place that takes a list of coders needs the same three things done to it, and
    a name typed with a trailing space is otherwise a tab nobody can find.

    Args:
        coders: the names as given.
        if_empty: the message to raise when nothing is left.

    Returns:
        The tidied names.

    Raises:
        ValueError: when no usable name is left.
    """
    names = []
    for name in coders:
        name = str(name).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError(if_empty)
    return names


def _open_sheet(sheet_id: str):
    """Open the spreadsheet. A pasted URL and a bare id both work.

    Args:
        sheet_id: the id in the sheet's URL, or the whole URL.

    Returns:
        The gspread spreadsheet.
    """
    client = _sheets_client()
    if str(sheet_id).startswith("http"):
        return client.open_by_url(sheet_id)
    return client.open_by_key(sheet_id)


def tab_names(sheet_id: str) -> list[str]:
    """The names of the tabs in your annotation sheet.

    Args:
        sheet_id: the id in the sheet's URL, or the whole URL.

    Returns:
        Every tab name, in the order the sheet has them.

    Example:
        >>> tab_names(SHEET_ID)
    """
    names = []
    for worksheet in _open_sheet(sheet_id).worksheets():
        names.append(worksheet.title)
    return names


# ----------------------------------------------------------------------------------
# Adding rows to a sheet you are already annotating in
# ----------------------------------------------------------------------------------
# Notebook 02b. By the time a group wants more items, the sheet has a morning's work in
# it and, very often, columns the group added or moved themselves. So two rules:
#
#   Find every column by the NAME in row 1, in each tab, at the moment of writing.
#     Nothing may assume "Text is column B" - that is only true until someone inserts a
#     column, and a positional write after that puts the text in the wrong place.
#
#   Write only the four columns this file owns, one column at a time, below the last
#     row that has data. Never rewrite a whole tab: a rewrite would revert whatever a
#     coder typed while the cell was running, and would replace any formula in a column
#     the group added with the value it happened to have.
def append_to_annotation_sheet(sheet_id: str,
                               items: list[dict[str, str]],
                               coders: list[str] | tuple = DEFAULT_CODERS,
                               final: str = FINAL_TAB,
                               expected_rows: int | None = None) -> int:
    """Add rows to the BOTTOM of an annotation sheet your group is already using.

    Every tab is checked before anything is written, and if any tab cannot be read with
    confidence then NO tab is written to - the rows are printed for you to paste in
    instead. A half-appended sheet is much worse than one you have to paste into.

    Args:
        sheet_id: the sheet's id or URL, as remembered_sheet returns it.
        items: the new items, already carrying the ids they should get.
        coders: the coder tabs to add the rows to.
        final: the name of the tab you adjudicate in.
        expected_rows: how many rows each tab should already have. Pass len(sampled)
            and a tab that has drifted from your sample file is caught before writing.

    Returns:
        The number of rows added to each tab.

    Raises:
        RuntimeError: when any tab fails its check. Nothing has been written.

    Example:
        >>> append_to_annotation_sheet(SHEET_ID, extra, coders=CODERS,
        ...                            expected_rows=len(sampled))
    """
    if not items:
        raise ValueError("There are no items to add, so there is nothing to do.")
    coder_names = _normalise_coder_names(
        coders,
        "The list of coder names is empty, so there is no tab to add rows to.\n"
        "Open config.yaml and put your group's coders in `members:`, or type the tab "
        'names into the cell above: CODERS = ["CoderA", "CoderB"]')

    sheet = _open_sheet(sheet_id)
    existing_tabs = {}
    for worksheet in sheet.worksheets():
        existing_tabs[worksheet.title] = worksheet

    ### Step 1: check every tab. Write nothing yet. ###
    wanted = list(coder_names)
    if final in existing_tabs:
        wanted.append(final)
    else:
        print("(no '" + final + "' tab, so the new rows are not added there)")

    plans = []
    problems = []
    for name in wanted:
        if name not in existing_tabs:
            problems.append("Tab '" + name + "' does not exist. The tabs in this sheet "
                            "are: " + " · ".join(sorted(existing_tabs)))
            continue
        try:
            plans.append(_plan_append(existing_tabs[name], name, len(items),
                                      expected_rows))
        except ValueError as error:
            problems.append(str(error))

    if problems:
        _refuse_to_append(problems, items, plans, existing_tabs, wanted)

    ### Step 2: every tab checked, so now write ###
    with_context = any(item.get("context") for item in items)
    written = []
    for plan in plans:
        try:
            _append_rows_to_tab(plan, items, with_context)
        except Exception as error:
            _report_half_written(written, plan, error)
            raise
        written.append(plan)
        print("  " + plan["name"] + ": added", len(items), "rows, from row",
              plan["first_free_row"])

    print("")
    print("Added", len(items), "rows to", len(plans), "tab(s).")
    print("Annotate the NEW rows only - the ones above them are already done.")
    print("If your Label column has a drop-down, click a Label cell in one of the new")
    print("rows and check it is there. Sheets does not always carry a drop-down down to")
    print("rows added later; if it is missing, copy a Label cell from a row above and")
    print("paste it over the new ones.")
    return len(items)


def _refuse_to_append(problems: list[str],
                      items: list[dict[str, str]],
                      plans: list[dict],
                      existing_tabs: dict,
                      wanted: list[str]) -> None:
    """Explain what stopped the write, print the rows to paste, and raise.

    Reached whenever any tab fails its check. Nothing has been written to any tab by
    this point, and nothing will be - so the group is never left with the rows in some
    tabs and not others, and pasting them in by hand is always a complete answer.

    Args:
        problems: one message per tab that failed.
        items: the rows that were going to be added.
        plans: the tabs that passed, so their real column order can be used.
        existing_tabs: every tab in the sheet, by name.
        wanted: the tabs the rows were meant for.

    Returns:
        Nothing. It always raises.

    Raises:
        RuntimeError: after printing.
    """
    print("Could not add the rows safely. NOTHING was written to any tab.")
    print("")
    for problem in problems:
        print(problem)
        print("")

    planned = {}
    for plan in plans:
        planned[plan["name"]] = plan

    print("Paste the rows in by hand instead. There are", len(items), "of them.")
    print("")
    for name in wanted:
        if name not in existing_tabs:
            continue
        _print_paste_block(name, items, planned.get(name))

    raise RuntimeError(
        "Nothing was written. Either fix the tab named above and run this cell again, "
        "or paste the rows in by hand from the output above - both give you the same "
        "sheet. If you paste, do NOT run this cell again afterwards.")


def _print_paste_block(name: str,
                       items: list[dict[str, str]],
                       plan: dict | None) -> None:
    """Print the new rows for one tab, tab-separated, ready to paste.

    Printed per tab, because each tab keeps the columns wherever its own row 1 says -
    and a block pasted in the wrong column order is exactly the mess this whole path
    exists to avoid.

    Context is left out on purpose: it holds line breaks, which paste as new rows and
    would scatter one item across a dozen of them. It is a display copy of the passage
    and nothing reads it back off the sheet.

    Args:
        name: the tab's name.
        items: the rows to print.
        plan: what _plan_append worked out for this tab, or None when it failed.

    Returns:
        Nothing. It prints.
    """
    print("--- tab '" + name + "' " + "-" * 40)

    if plan is None:
        # Row 1 could not be read, so the tab's own column order is unknown. Name the
        # columns instead and let the reader line them up.
        _print_paste_rows([COL_ID, COL_TEXT, COL_SOURCE], items,
                          "Put these under the matching headers, one column at a time. "
                          "Check each one lines up before you paste the next.")
        print("")
        return

    # The owned columns this tab actually has, in this tab's own order, minus Context.
    ordered = []
    for owned, column in sorted(plan["columns"].items(), key=lambda pair: pair[1]):
        if owned != COL_CONTEXT:
            ordered.append((column, owned))

    # Split into runs of neighbouring columns. A group that put its Label column between
    # Text and SourceID gets two blocks, each pasted where it belongs, rather than one
    # block that would overwrite the labels in between.
    runs = []
    for column, owned in ordered:
        neighbouring = runs and column == runs[len(runs) - 1]["last_column"] + 1
        if neighbouring:
            runs[len(runs) - 1]["names"].append(owned)
            runs[len(runs) - 1]["last_column"] = column
        else:
            runs.append({"names": [owned], "first_column": column,
                         "last_column": column})

    for run in runs:
        cell = _column_letter(run["first_column"]) + str(plan["first_free_row"])
        _print_paste_rows(run["names"], items,
                          "Click cell " + cell + " in tab '" + name
                          + "', then Edit > Paste.")
    print("")


def _print_paste_rows(column_names: list[str],
                      items: list[dict[str, str]],
                      instruction: str) -> None:
    """Print one block of tab-separated rows, plus how to paste it.

    Args:
        column_names: the columns to print, in the tab's own order.
        items: the rows to print.
        instruction: where to click before pasting.

    Returns:
        Nothing. It prints.
    """
    print(instruction)
    print("")
    with_context = any(item.get("context") for item in items)
    flattened = 0
    lines = [ "\t".join(column_names) ]
    for item in items:
        values = _cell_values(item, with_context)
        cells = []
        for column_name in column_names:
            cell = str(values.get(column_name, ""))
            tidy = cell.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            if tidy != cell:
                flattened = flattened + 1
            cells.append(tidy)
        lines.append("\t".join(cells))
    for line in lines:
        print("    " + line)
    print("")
    if flattened:
        print("    (" + str(flattened) + " cell(s) had a tab or a line break in them, "
              "replaced with a space so they paste as one row each.)")
        print("")


def _plan_append(worksheet, name: str, n_items: int,
                 expected_rows: int | None) -> dict:
    """Work out where the new rows go in one tab, without writing anything.

    Args:
        worksheet: the tab to look at.
        name: its name, for the messages.
        n_items: how many rows are about to be added.
        expected_rows: how many rows this tab should already have, or None.

    Returns:
        What the write needs: the tab, its name, the column of each owned header, and
        the first row that has no data in it.

    Raises:
        ValueError: when the tab cannot be written to safely, saying why.
    """
    header = worksheet.row_values(1)
    columns = _header_columns(header, name)

    for required in (COL_ID, COL_TEXT):
        if required not in columns:
            raise ValueError(
                "Tab '" + name + "' has no column called '" + required + "'.\n"
                "  row 1 reads: " + " · ".join(header) + "\n"
                "  Nothing can be added by hand-counting columns instead - that is how "
                "rows end up with the text in the wrong place.\n"
                "  Rename the column back to '" + required + "'. Notebook 03 needs that "
                "name too, so the rename would have broken the agreement step as well.")

    first_free = _first_free_row(worksheet, columns[COL_ID], name)
    if expected_rows is not None and first_free - 2 != expected_rows:
        raise ValueError(
            "Tab '" + name + "' has " + str(first_free - 2) + " rows of data, and your "
            "sample file has " + str(expected_rows) + ".\n"
            "  They should be the same. Either rows were deleted from the sheet, or "
            "somebody has already added rows to it.\n"
            "  Nothing was written. Sort out which of the two is right before adding "
            "more.")
    return {"worksheet": worksheet, "name": name, "columns": columns,
            "first_free_row": first_free, "n_items": n_items}


def _header_columns(header: list[str], name: str) -> dict[str, int]:
    """Map each owned header name to its column number, 1-based.

    Args:
        header: the tab's row 1, as a list of strings.
        name: the tab's name, for the messages.

    Returns:
        {header name: column number} for the columns this file writes.

    Raises:
        ValueError: on a repeated header name, or a blank one with data to its right.
    """
    seen = {}
    for position, cell in enumerate(header):
        cell = str(cell).strip()
        if not cell:
            continue
        if cell in seen and cell in OWNED_COLUMNS:
            raise ValueError(
                "Tab '" + name + "' has two columns called '" + cell + "' (columns "
                + _column_letter(seen[cell]) + " and "
                + _column_letter(position + 1) + ").\n"
                "  There is no way to tell which one the rows belong in. Reading the "
                "sheet refuses this too, so notebook 03 would fail on it as well.\n"
                "  Delete or rename one of them.")
        seen[cell] = position + 1

    # A blank header with data underneath it is the other shape that breaks reading the
    # sheet back, so catch it here rather than let notebook 03 find it later.
    for position, cell in enumerate(header):
        if not str(cell).strip():
            for later in header[position + 1:]:
                if str(later).strip():
                    raise ValueError(
                        "Tab '" + name + "' has a column with no name in row 1 (column "
                        + _column_letter(position + 1) + "), and named columns after "
                        "it.\n"
                        "  Reading the sheet back refuses this, so notebook 03 would "
                        "fail on it too. Give it a name, or delete the column.")
            break

    columns = {}
    for owned in OWNED_COLUMNS:
        if owned in seen:
            columns[owned] = seen[owned]
    return columns


def _first_free_row(worksheet, id_column: int, name: str) -> int:
    """The first row of this tab with no id in it.

    Args:
        worksheet: the tab to measure.
        id_column: the column number the ids are in.
        name: the tab's name, for the message.

    Returns:
        The row number to start writing at.

    Raises:
        ValueError: when the id column has gaps in the middle, so where the data ends
            cannot be told from where a row was cleared.
    """
    # col_values stops at the last non-empty cell, so its length IS the last used row.
    values = worksheet.col_values(id_column)
    filled = 0
    for value in values[1:]:
        if str(value).strip():
            filled = filled + 1
    if filled != len(values) - 1:
        raise ValueError(
            "The " + COL_ID + " column in tab '" + name + "' has "
            + str(len(values) - 1 - filled) + " blank row(s) in the middle of it.\n"
            "  That makes it unclear where your data ends, and rows added in the wrong "
            "place are hard to undo.\n"
            "  Either fill those ids back in or delete the empty rows, then run this "
            "cell again.")
    return len(values) + 1


def _append_rows_to_tab(plan: dict, items: list[dict[str, str]],
                        with_context: bool) -> None:
    """Write the new rows into one tab, one owned column at a time.

    One ranged write per column, rather than one rectangle, because the owned columns
    are wherever this tab happens to keep them - and because a column the group added
    in between must not be written to at all.

    Args:
        plan: what _plan_append worked out for this tab.
        items: the new items.
        with_context: True when this track carries a passage per item.

    Returns:
        Nothing. It writes to the sheet.
    """
    worksheet = plan["worksheet"]
    first_row = plan["first_free_row"]
    last_row = first_row + len(items) - 1

    # Grow the grid first if the new rows would fall off the bottom of it.
    if last_row > worksheet.row_count:
        worksheet.add_rows(last_row - worksheet.row_count)

    for owned, column in sorted(plan["columns"].items(), key=lambda pair: pair[1]):
        column_values = []
        for item in items:
            values = _cell_values(item, with_context)
            column_values.append([values.get(owned, "")])
        letter = _column_letter(column)
        # RAW for the same reason the sheet was created with it: a sentence starting
        # with "=" or "-" must be stored as text, not read as a formula.
        worksheet.update(range_name=letter + str(first_row) + ":" + letter
                         + str(last_row),
                         values=column_values, value_input_option="RAW")


def _report_half_written(written: list[dict], failed_plan: dict,
                         error: Exception) -> None:
    """Say exactly which rows landed where, when a write fails partway through.

    The checks before the write make this rare, but it cannot be ruled out - a network
    can drop between one tab and the next. Whoever reads this has to be able to undo it
    by hand, so name the tabs and the rows.

    Args:
        written: the tabs that were written to before the failure.
        failed_plan: the tab the write failed on.
        error: what went wrong.

    Returns:
        Nothing. It prints what to undo by hand.
    """
    print("")
    print("The write FAILED on tab '" + failed_plan["name"] + "':", error)
    if not written:
        print("Nothing was written to any tab. Run the cell again.")
        return
    print("These tabs were already written to, and the rows are still there:")
    for plan in written:
        last = plan["first_free_row"] + plan["n_items"] - 1
        print("  " + plan["name"] + ": rows", plan["first_free_row"], "to", last)
    print("Delete exactly those rows in exactly those tabs, then run the cell again.")
    print("Do not run it again before deleting them, or the rows will be added twice.")


def load_annotation_sheet(sheet_id: str,
                          worksheet: str = DEFAULT_CODERS[0]
                          ) -> list[dict[str, str]]:
    """Read ONE TAB of your annotation sheet back as a list of row dicts.

    Usually you want `load_coder_sheets` instead - it calls this once per coder and
    joins the tabs into one table. Reach for this one to look at a single tab.

    Args:
        sheet_id: the long id in the sheet's URL
            (docs.google.com/spreadsheets/d/<THIS PART>/edit). The whole URL works
            too - either way opens the exact sheet, so two copies that share a name
            ("Copy of ...") are never confused.
        worksheet: the TAB name, which is a coder: "CoderA", "CoderB", or "Final".

    Returns:
        One dict per row, keyed by the column headings.

    Raises:
        ValueError: when the sheet has no tab by that name (the message lists the
            tabs it does have), or when the header row has a duplicated or blank
            column name.

    Example:
        >>> rows = load_annotation_sheet(SHEET_ID, "CoderA")
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


def load_coder_sheets(sheet_id: str,
                      coders: list[str] | tuple = DEFAULT_CODERS,
                      final: str = FINAL_TAB) -> list[dict[str, str]]:
    """Read every coder's tab, plus the Final tab, and line them up as one table.

    This is the function to use now that each coder has their own tab. It calls
    `load_annotation_sheet` once per tab - the same function, the same call form - and
    joins the results by ID into exactly the shape everything downstream already
    expects.

    Args:
        sheet_id: the id in the sheet's URL, or the whole URL.
        coders: the TAB NAMES to read, given here rather than when the sheet was
            created. That is deliberate: your group does not have to know how many
            coders it has before it starts. Gained a third? Duplicate an empty tab,
            rename it "CoderC", add "CoderC" to this list. Nothing else changes.
        final: the name of the tab you adjudicate in.

    Returns:
        One row per item, with a column per coder plus Final and Note.

    Example:
        >>> rows = load_coder_sheets(SHEET_ID, ["CoderA", "CoderB"])
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


def to_canonical(rows: list[dict[str, str]],
                 labels: list[str],
                 column: str = COL_FINAL,
                 source: list[dict[str, str]] | None = None
                 ) -> list[dict[str, str]]:
    """Turn annotation rows into canonical gold: [{"id", "text", "label"}, ...].

    Blank rows are skipped; labels outside `labels` are reported, not silently kept.

    Args:
        rows: the rows read back by load_coder_sheets or load_annotation_sheet.
        labels: the labels your scheme allows. Anything else is reported as invalid.
        column: which column holds the adjudicated label.
        source: the items the sheet was BUILT from - your sampled items. Pass it on a
            track that carries context: gold is rebuilt from the sheet, which holds
            only the id, the text and your label, so anything else the item was
            carrying would be dropped here and notebook 04 would never see it. The
            extra fields are copied from `source` by id rather than read back out of
            the sheet, because the sheet's Context column is a marked-up display copy
            a coder may have edited.

    Returns:
        The usable rows as gold items, each {"id", "text", "label"} plus whatever
        `source` was carrying.

    Example:
        >>> gold = to_canonical(rows, LABELS_ORDER, source=sampled)
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


def fleiss_kappa(label_lists: list[list[str]]) -> float:
    """Fleiss' kappa: agreement among THREE OR MORE annotators, as one number.

    Cohen's kappa compares exactly two people, so with three coders there is no single
    Cohen's number to report - you get one per pair. Fleiss' answers the other question:
    how much do the whole group agree, over and above what people picking at random in
    the same proportions would manage? Read it on the same scale you read Cohen's.

    Written out here because scikit-learn does not provide it. The formula:

        P(i)    for each item, the share of annotator PAIRS on that item who agree
        P_bar   the average of those - observed agreement
        P_e     what you would expect from chance, given how often each label is used
        kappa   (P_bar - P_e) / (1 - P_e)

    Args:
        label_lists: one list per annotator, all the same length, all labelling the
            same items in the same order.

    Returns:
        The kappa, or nan when there are fewer than two annotators, no items, or only
        one label was used - undefined, not zero.

    Example:
        >>> fleiss_kappa([coder_a, coder_b, coder_c])
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


def _draw_coder_matrix(a_labels: list[str], b_labels: list[str], name_a: str,
                       name_b: str, title: str) -> None:
    """One coder's labels against another's, as a heatmap. The diagonal is agreement.

    Args:
        a_labels: the first coder's labels.
        b_labels: the second coder's labels, for the same items in the same order.
        name_a: what to call the rows.
        name_b: what to call the columns.
        title: the heading to put above it.

    Returns:
        Nothing. It draws the picture.
    """
    labels = sorted(set(a_labels) | set(b_labels))
    matrix = confusion_matrix(a_labels, b_labels, labels=labels)
    plot_confusion_matrix(matrix, labels, title, xlabel=name_b, ylabel=name_a)


def _coder_columns(rows: list[dict[str, str]],
                   coders: list[str] | tuple) -> tuple:
    """Which columns hold labels, and every row where ALL of them are filled in.

    Args:
        rows: the merged rows, one per item.
        coders: the coder names, which are also the column names.

    Returns:
        Two things: the tidied coder names, and one list of labels per complete row.
        Half-finished rows are dropped.

    Raises:
        ValueError: when the coder list is empty.
    """
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


def _labels_by_coder(names: list[str],
                     complete: list[list[str]]) -> list[list[str]]:
    """Turn one list of labels per ROW into one list of labels per CODER.

    Args:
        names: the coder names, in column order.
        complete: one list of labels per row, in the same column order.

    Returns:
        One list of labels per coder, which is what fleiss_kappa wants.
    """
    by_coder = []
    for position in range(len(names)):
        one = []
        for labels in complete:
            one.append(labels[position])
        by_coder.append(one)
    return by_coder


def _kappa_of_pair(a_labels: list[str], b_labels: list[str]) -> float:
    """Cohen's kappa for two coders.

    Args:
        a_labels: the first coder's labels.
        b_labels: the second coder's labels, for the same items in the same order.

    Returns:
        The kappa, or nan when they only ever used one label between them.
    """
    if len(set(a_labels) | set(b_labels)) < 2:
        return float("nan")
    return cohen_kappa_score(a_labels, b_labels)


def _pairwise_kappas(names: list[str], by_coder: list[list[str]]) -> tuple:
    """Cohen's kappa for every pair of coders, printed and returned.

    Args:
        names: the coder names.
        by_coder: one list of labels per coder, in the same order.

    Returns:
        Two things: the kappa for each pair, and which pair agreed LEAST. The second
        is there because "who diverges from whom" is the question that leads
        somewhere - a single group-wide number does not tell you which label boundary
        to go and argue about. It is None when no pair produced a real number.
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


def _agreement_for_many(rows: list[dict[str, str]],
                        coders: list[str] | tuple) -> dict | None:
    """Fleiss' kappa across the group, plus Cohen's kappa for every pair.

    Args:
        rows: the merged rows, one per item.
        coders: the coder names, three or more.

    Returns:
        {"n", "coders", "kappa", "fleiss_kappa", "pairwise_kappa"}, or None when no
        row has every coder filled in. "kappa" is the Fleiss number, under that name
        so code written against the two-coder version keeps working.
    """
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


def annotator_agreement(rows: list[dict[str, str]],
                        a: str = COL_A,
                        b: str = COL_B,
                        coders: list[str] | None = None) -> dict | None:
    """Percent agreement + Cohen's kappa between the two annotator columns, PLUS an
    annotator-vs-annotator confusion matrix (the diagonal is where you agreed;
    off-diagonal cells show which label pairs the two of you confuse).

    Args:
        rows: the merged rows, one per item.
        a: the column holding the first annotator's labels.
        b: the column holding the second annotator's labels.
        coders: for groups of three or more. Pass the list of coder names and you get
            Fleiss' kappa for the group, then Cohen's kappa for every pair, then the
            confusion matrix of the pair that agreed least. With two coders - named or
            not - the output is exactly what it has always been.

    Returns:
        {"n", "percent_agreement", "kappa"} for two coders, or the group-wide dict
        for three or more. None when no row has every coder filled in.

    Example:
        >>> annotator_agreement(rows, coders=CODERS)
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


def compare_to_published(gold: list[dict[str, str]],
                         published: list[dict[str, str]]) -> pd.DataFrame | None:
    """How often does YOUR final label match the published one, item by item?

    Items are matched by their TEXT, not their id. Sampling renumbers the ids from 1,
    so an id-based match would line YOUR item 7 up against POOL item 7 - two unrelated
    sentences - and report a meaningless number without ever failing. (Ids are still
    used as a fallback, for the case where the texts have been edited.)

    Args:
        gold: your own gold items, from to_canonical.
        published: the published items, from load_gold.

    Returns:
        A table of the items where you and the published label differ, or None when
        nothing could be matched.

    Example:
        >>> compare_to_published(my_gold, published)
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
