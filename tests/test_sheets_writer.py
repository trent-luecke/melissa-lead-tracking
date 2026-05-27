"""Tests for collectors/sheets_writer.py — all run against a mocked Sheets API."""

from __future__ import annotations

import json
import logging
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SA_JSON = json.dumps({
    "type": "service_account",
    "project_id": "fake-project",
    "private_key_id": "key-id",
    "private_key": (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4PAtEsHAEBMf1d5NMMDuBGptFEi\n"
        "-----END RSA PRIVATE KEY-----\n"
    ),
    "client_email": "fake@fake-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
})

_SPREADSHEET_ID = "1o2mcdeHV6Lxm70J6pvcByGNBcY44d_QZ6r5NYHOLTJs"
_TAB = "LEAD TRACKER"
_ROW = 5


def _make_service(notes_value: str = "") -> MagicMock:
    """Build a mock service where get().execute() returns the given notes value."""
    mock_service = MagicMock()

    # The 'get' call for reading notes
    get_execute = MagicMock(return_value={"values": [[notes_value]]} if notes_value else {"values": [[""]]})
    mock_get = MagicMock()
    mock_get.return_value.execute = get_execute

    # The 'update' call — used for the Notes column write
    update_execute = MagicMock(return_value={})
    mock_update = MagicMock()
    mock_update.return_value.execute = update_execute

    # The 'batchUpdate' call — used for all non-Notes field writes
    batch_update_execute = MagicMock(return_value={})
    mock_batch_update = MagicMock()
    mock_batch_update.return_value.execute = batch_update_execute

    mock_values = MagicMock()
    mock_values.return_value.get = mock_get
    mock_values.return_value.update = mock_update
    mock_values.return_value.batchUpdate = mock_batch_update

    mock_service.spreadsheets.return_value.values = mock_values
    return mock_service


def _run_write(updates: dict, note: str, notes_value: str = "", row_index=_ROW):
    """Patch creds + build, run write_updates, return the mock service."""
    mock_service = _make_service(notes_value=notes_value)
    with (
        patch("collectors.sheets_writer.service_account.Credentials.from_service_account_info"),
        patch("collectors.sheets_writer.build", return_value=mock_service),
    ):
        from collectors.sheets_writer import write_updates
        write_updates(
            service_account_json=_FAKE_SA_JSON,
            spreadsheet_id=_SPREADSHEET_ID,
            tab_name=_TAB,
            row_index=row_index,
            updates=updates,
            note=note,
        )
    return mock_service


def _get_update_calls(mock_service: MagicMock) -> list[dict]:
    """Extract the kwargs from every .update(...).execute() call chain (Notes column only)."""
    update_mock = mock_service.spreadsheets.return_value.values.return_value.update
    return [c.kwargs for c in update_mock.call_args_list]


def _get_batch_update_calls(mock_service: MagicMock) -> list[dict]:
    """Extract the kwargs from every .batchUpdate(...).execute() call chain."""
    batch_mock = mock_service.spreadsheets.return_value.values.return_value.batchUpdate
    return [c.kwargs for c in batch_mock.call_args_list]


def _batch_ranges(mock_service: MagicMock) -> dict[str, object]:
    """Return a mapping of range → value from the batchUpdate body's data list."""
    calls = _get_batch_update_calls(mock_service)
    result = {}
    for call_kwargs in calls:
        for item in call_kwargs.get("body", {}).get("data", []):
            result[item["range"]] = item["values"][0][0]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStageUpdate:
    """Test 1: Stage update writes to the correct A1 range (C{row}) via batchUpdate."""

    def test_stage_writes_to_column_c(self):
        mock_service = _run_write(updates={"Stage": "Proposal"}, note="Sent proposal")

        # Non-Notes fields go through batchUpdate
        batch_calls = _get_batch_update_calls(mock_service)
        assert len(batch_calls) == 1

        data = batch_calls[0]["body"]["data"]
        assert len(data) == 1
        assert data[0]["range"] == f"'{_TAB}'!C{_ROW}"
        assert data[0]["values"] == [["Proposal"]]
        assert batch_calls[0]["body"]["valueInputOption"] == "USER_ENTERED"

        # Notes goes through update
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 1
        assert update_calls[0]["range"] == f"'{_TAB}'!O{_ROW}"


class TestNotesPrepend:
    """Test 2: Notes read-then-prepend — existing notes preserved above new note."""

    def test_existing_notes_are_preserved(self):
        existing = "Old note from last week"
        mock_service = _run_write(
            updates={},
            note="Followed up via email",
            notes_value=existing,
        )
        # No non-Notes fields → batchUpdate not called
        batch_calls = _get_batch_update_calls(mock_service)
        assert len(batch_calls) == 0

        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 1
        written_value: str = update_calls[0]["body"]["values"][0][0]

        # New note is first, old note follows
        lines = written_value.split("\n")
        assert len(lines) == 2
        assert lines[0].endswith("] Followed up via email")
        assert lines[1] == existing

    def test_new_note_is_prepended_not_appended(self):
        existing = "Prior entry"
        mock_service = _run_write(
            updates={},
            note="Brand new note",
            notes_value=existing,
        )
        update_calls = _get_update_calls(mock_service)
        written_value: str = update_calls[0]["body"]["values"][0][0]
        assert written_value.startswith("[")  # new note at top, starts with [date]
        assert written_value.endswith(existing)


class TestEmptyExistingNotes:
    """Test 3: Empty existing notes — just writes new note without trailing newline."""

    def test_empty_notes_no_trailing_newline(self):
        mock_service = _run_write(updates={}, note="First contact made", notes_value="")
        update_calls = _get_update_calls(mock_service)
        written_value: str = update_calls[0]["body"]["values"][0][0]

        assert "\n" not in written_value
        assert written_value.endswith("] First contact made")

    def test_whitespace_only_notes_treated_as_empty(self):
        mock_service = _run_write(updates={}, note="Initial outreach", notes_value="   ")
        update_calls = _get_update_calls(mock_service)
        written_value: str = update_calls[0]["body"]["values"][0][0]

        assert "\n" not in written_value


class TestEmptyNote:
    """Test: Empty or whitespace-only note skips the Notes column entirely."""

    def test_empty_string_note_skips_notes_column(self):
        mock_service = _run_write(updates={"Stage": "Prospect"}, note="")
        # Notes update (values().update) must not be called
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 0
        # get() for reading notes must also not be called
        get_mock = mock_service.spreadsheets.return_value.values.return_value.get
        assert get_mock.call_count == 0

    def test_whitespace_note_skips_notes_column(self):
        mock_service = _run_write(updates={}, note="   ")
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 0
        get_mock = mock_service.spreadsheets.return_value.values.return_value.get
        assert get_mock.call_count == 0

    def test_empty_note_with_field_updates_still_writes_fields(self):
        mock_service = _run_write(updates={"Stage": "Closed Won"}, note="")
        # batchUpdate should still fire for the Stage field
        batch_calls = _get_batch_update_calls(mock_service)
        assert len(batch_calls) == 1
        # But Notes update must not happen
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 0


class TestFormulaColumnGuard:
    """Test 4: Formula columns raise ValueError."""

    def test_days_until_follow_up_raises(self):
        with pytest.raises(ValueError, match="formula column"):
            _run_write(updates={"Days Until Follow-Up": "5"}, note="some note")

    def test_days_since_last_contact_raises(self):
        with pytest.raises(ValueError, match="formula column"):
            _run_write(updates={"Days Since Last Contact": "3"}, note="some note")


class TestUnknownColumn:
    """Test 5: Unknown column name → logged warning, skipped (not crash)."""

    def test_unknown_column_does_not_crash(self):
        # Should complete without raising
        _run_write(updates={"NonExistent Column": "value"}, note="note")

    def test_unknown_column_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="collectors.sheets_writer"):
            _run_write(updates={"NonExistent Column": "value"}, note="note")
        assert "nonexistent column" in caplog.text.lower() or "unknown" in caplog.text.lower()

    def test_unknown_column_not_written_to_sheet(self):
        mock_service = _run_write(updates={"Bogus Field": "value"}, note="note")
        # batchUpdate should not be called (no valid ranges)
        batch_calls = _get_batch_update_calls(mock_service)
        assert len(batch_calls) == 0
        # Only the Notes update should have happened
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 1
        assert update_calls[0]["range"].endswith("O5")


class TestNoneRowIndex:
    """Test 6: row_index=None raises ValueError."""

    def test_none_row_index_raises(self):
        with pytest.raises(ValueError, match="row_index is required"):
            _run_write(updates={}, note="note", row_index=None)


class TestMultipleFields:
    """Test 7: Multiple fields written in one batchUpdate — all cells updated."""

    def test_all_fields_written(self):
        updates = {
            "Stage": "Closed Won",
            "Follow-Up Priority": "Low",
            "Last Contact Date": "05/23/2026",
            "Email": "coach@gym.com",
        }
        mock_service = _run_write(updates=updates, note="Deal closed")

        # All non-Notes fields go through a single batchUpdate call
        batch_calls = _get_batch_update_calls(mock_service)
        assert len(batch_calls) == 1

        data = batch_calls[0]["body"]["data"]
        written_ranges = {item["range"] for item in data}
        assert f"'{_TAB}'!C{_ROW}" in written_ranges   # Stage
        assert f"'{_TAB}'!D{_ROW}" in written_ranges   # Follow-Up Priority
        assert f"'{_TAB}'!H{_ROW}" in written_ranges   # Last Contact Date
        assert f"'{_TAB}'!M{_ROW}" in written_ranges   # Email

        # Notes goes through the separate update() call
        update_calls = _get_update_calls(mock_service)
        assert len(update_calls) == 1
        assert update_calls[0]["range"] == f"'{_TAB}'!O{_ROW}"

    def test_each_field_value_correct(self):
        updates = {
            "Stage": "Demo",
            "Email": "test@example.com",
        }
        mock_service = _run_write(updates=updates, note="Scheduled demo")

        range_to_value = _batch_ranges(mock_service)
        assert range_to_value[f"'{_TAB}'!C{_ROW}"] == "Demo"
        assert range_to_value[f"'{_TAB}'!M{_ROW}"] == "test@example.com"


class TestNotesDateFormat:
    """Test 8: Notes date format is [MM/DD/YYYY] with leading zeros."""

    def test_date_format_has_leading_zeros(self):
        # Freeze the date to a day/month with single digits to verify leading zeros.
        fixed_date = date(2026, 5, 4)  # month=05, day=04
        with patch("collectors.sheets_writer.date") as mock_date:
            mock_date.today.return_value = fixed_date
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            mock_service = _run_write(updates={}, note="Test note")

        update_calls = _get_update_calls(mock_service)
        written_value: str = update_calls[0]["body"]["values"][0][0]
        # Should have [05/04/2026] — two-digit month and day
        assert "[05/04/2026]" in written_value

    def test_date_format_pattern(self):
        """Date string matches MM/DD/YYYY pattern."""
        import re
        mock_service = _run_write(updates={}, note="Pattern check")
        update_calls = _get_update_calls(mock_service)
        written_value: str = update_calls[0]["body"]["values"][0][0]
        # Extract the date portion from [MM/DD/YYYY]
        match = re.search(r'\[(\d{2}/\d{2}/\d{4})\]', written_value)
        assert match is not None, f"Expected [MM/DD/YYYY] pattern in: {written_value!r}"
