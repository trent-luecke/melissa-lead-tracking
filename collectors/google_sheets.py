"""Google Sheets collector — loads Melissa's LEAD TRACKER tab.

Returns a dict keyed by both normalized lead name and normalized org name so
callers can match Avoma participant names against either column.
"""

from __future__ import annotations

import json
import logging
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_log = logging.getLogger(__name__)

# Column indices (0-based after converting from 1-based spec)
_COL_LEAD_NAME = 0        # A
_COL_ORG = 1              # B
_COL_STAGE = 2            # C
_COL_FOLLOW_UP_PRIORITY = 3  # D
_COL_NEXT_ACTION_DATE = 4    # E
# F (5) = Days Until Follow-Up — formula, skip
# G (6) = Days Since Last Contact — formula, skip
_COL_LAST_CONTACT_DATE = 7   # H
_COL_INITIAL_CONTACT_DATE = 8  # I
_COL_BRANCH_SECTOR = 9    # J
# K (10), L (11) — skip
_COL_EMAIL = 12           # M
# N (13) — skip
_COL_NOTES = 14           # O

_ROW_LENGTH = 15  # columns A(0) through O(14)


def normalize_name(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower().strip())


def load_lead_tracker(
    service_account_json: str,
    spreadsheet_id: str,
    tab_name: str = "LEAD TRACKER",
) -> dict[str, dict]:
    """Return a dict keyed by normalized lead name (and org name as secondary key).

    Each value is:
    {
        "row_index": int,       # 1-based sheet row number (2 = first data row)
        "lead_name": str,
        "organization": str,
        "stage": str,
        "follow_up_priority": str,
        "next_action_date": str,
        "last_contact_date": str,
        "initial_contact_date": str,
        "branch_sector": str,
        "email": str,
        "notes": str,
    }

    Duplicate normalized keys get a _2, _3, ... suffix; a warning is printed.
    """
    try:
        creds_dict = json.loads(service_account_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"service_account_json is not valid JSON: {e}") from e
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    service = build("sheets", "v4", credentials=creds)

    try:
        response = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'!A:O")
            .execute()
        )
    except HttpError as e:
        raise RuntimeError(f"Sheets API call failed (spreadsheet_id={spreadsheet_id}): {e}") from e

    rows = response.get("values", [])
    if not rows:
        return {}

    # Row 0 is the header; data starts at row index 1 in this list (sheet row 2)
    data_rows = rows[1:]

    result: dict[str, dict] = {}

    def _insert(key: str, value: dict) -> None:
        if not key:
            return
        if key not in result:
            result[key] = value
            return
        if result[key] is value:
            # same row — lead_name and org normalize to same string; skip silently
            return
        # true duplicate — find next available suffix
        i = 2
        while f"{key}_{i}" in result:
            i += 1
        new_key = f"{key}_{i}"
        _log.warning("duplicate normalized key '%s', storing as '%s'", key, new_key)
        result[new_key] = value

    for list_idx, row in enumerate(data_rows):
        # Pad to 15 elements so all column accesses are safe
        row = row + [''] * (_ROW_LENGTH - len(row))

        sheet_row = list_idx + 2  # row 1 is header, data starts at row 2

        lead_name: str = row[_COL_LEAD_NAME]
        org: str = row[_COL_ORG]

        # Skip completely blank rows
        if not any(cell.strip() for cell in row):
            continue

        row_data: dict = {
            "row_index": sheet_row,
            "lead_name": lead_name,
            "organization": org,
            "stage": row[_COL_STAGE],
            "follow_up_priority": row[_COL_FOLLOW_UP_PRIORITY],
            "next_action_date": row[_COL_NEXT_ACTION_DATE],
            "last_contact_date": row[_COL_LAST_CONTACT_DATE],
            "initial_contact_date": row[_COL_INITIAL_CONTACT_DATE],
            "branch_sector": row[_COL_BRANCH_SECTOR],
            "email": row[_COL_EMAIL],
            "notes": row[_COL_NOTES],
        }

        if lead_name.strip():
            _insert(normalize_name(lead_name), row_data)

        if org.strip():
            _insert(normalize_name(org), row_data)

    return result
