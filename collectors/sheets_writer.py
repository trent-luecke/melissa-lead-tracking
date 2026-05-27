"""Google Sheets writer — writes approved CRM updates back to Melissa's LEAD TRACKER."""

from __future__ import annotations

import json
import logging
from datetime import date

from google.oauth2 import service_account
from googleapiclient.discovery import build

_log = logging.getLogger(__name__)

# Maps field names to their column letters in the sheet.
# F (Days Until Follow-Up) and G (Days Since Last Contact) are formula columns — excluded.
# Notes (O) is handled separately via read-modify-write.
_COL_MAP: dict[str, str] = {
    "Stage":                "C",
    "Follow-Up Priority":   "D",
    "Next Action Date":     "E",
    "Last Contact Date":    "H",
    "Initial Contact Date": "I",
    "Branch / Sector":      "J",
    "Email":                "M",
}

# Formula columns that must never be written to.
_FORMULA_COLUMNS = {"Days Until Follow-Up", "Days Since Last Contact"}


def write_updates(
    service_account_json: str,
    spreadsheet_id: str,
    tab_name: str,
    row_index: int | None,
    updates: dict,
    note: str,
) -> None:
    """Write approved updates to Google Sheet.

    # NOTE: This function is not concurrency-safe for writes to the same row.
    # The nightly job and reply handler must not overlap (enforced at the scheduler level).

    Args:
        service_account_json: JSON string of service account credentials.
        spreadsheet_id: The Google Sheet ID.
        tab_name: Name of the tab/worksheet (e.g. "LEAD TRACKER").
        row_index: 1-based sheet row number (e.g. 2 = first data row).
                   None raises ValueError — new row creation is not implemented.
        updates: Mapping of column_name → value (without "Notes").
        note: New one-sentence note to PREPEND to the existing Notes column (column O).
    """
    if row_index is None:
        raise ValueError("row_index is required — new row creation not implemented")

    # Guard formula columns.
    for col in _FORMULA_COLUMNS:
        if col in updates:
            raise ValueError(f"Cannot write to formula column: {col}")

    # Build the service client.
    creds_dict = json.loads(service_account_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    service = build("sheets", "v4", credentials=creds)

    errors: list[tuple[str, Exception]] = []

    # Build value ranges for batch update — collect all, then send one request.
    value_ranges = []
    skipped = []
    for field_name, value in updates.items():
        col_letter = _COL_MAP.get(field_name)
        if not col_letter:
            skipped.append(field_name)
            continue
        value_ranges.append({
            "range": f"'{tab_name}'!{col_letter}{row_index}",
            "values": [[value]],
        })

    for field_name in skipped:
        _log.warning("Unknown column '%s' — skipping", field_name)

    if value_ranges:
        try:
            service.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "valueInputOption": "USER_ENTERED",
                    "data": value_ranges,
                },
            ).execute()
        except Exception as exc:
            _log.error("Batch update failed: %s", exc)
            errors.append(("batch", exc))

    # Notes column (O) — read-then-prepend pattern.
    if note and note.strip():
        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"'{tab_name}'!O{row_index}",
            ).execute()
            current_notes: str = (result.get("values") or [[""]])[0][0]

            date_str = date.today().strftime("%m/%d/%Y")
            new_note_line = f"[{date_str}] {note}"

            if current_notes.strip():
                updated_notes = f"{new_note_line}\n{current_notes.strip()}"
            else:
                updated_notes = new_note_line

            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"'{tab_name}'!O{row_index}",
                valueInputOption="USER_ENTERED",
                body={"values": [[updated_notes]]},
            ).execute()
        except Exception as exc:
            _log.error("Failed to write Notes: %s", exc)
            errors.append(("Notes", exc))
    else:
        _log.debug("No note to write for row %s — skipping Notes column", row_index)

    if errors:
        failed_fields = [f for f, _ in errors]
        raise RuntimeError(f"Failed to write {len(errors)} field(s): {failed_fields}")
