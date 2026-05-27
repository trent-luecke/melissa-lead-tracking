"""Tests for collectors/google_sheets.py — all run against a mocked Sheets API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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

# A realistic header row matching the spec
_HEADER = [
    "Lead Name", "Organization", "Stage", "Follow-Up Priority",
    "Next Action Date", "Days Until Follow-Up", "Days Since Last Contact",
    "Last Contact Date", "Initial Contact Date", "Branch / Sector",
    "", "", "Email", "", "Notes",
]


def _make_row(
    lead_name="",
    org="",
    stage="",
    follow_up_priority="",
    next_action_date="",
    days_until="",
    days_since="",
    last_contact_date="",
    initial_contact_date="",
    branch_sector="",
    col_k="",
    col_l="",
    email="",
    col_n="",
    notes="",
) -> list[str]:
    return [
        lead_name, org, stage, follow_up_priority, next_action_date,
        days_until, days_since, last_contact_date, initial_contact_date,
        branch_sector, col_k, col_l, email, col_n, notes,
    ]


def _mock_service(rows: list[list[str]]):
    """Return a mock googleapiclient service whose values().get().execute() returns rows."""
    mock_execute = MagicMock(return_value={"values": rows})
    mock_get = MagicMock()
    mock_get.return_value.execute = mock_execute
    mock_values = MagicMock()
    mock_values.return_value.get = mock_get
    mock_spreadsheets = MagicMock()
    mock_spreadsheets.return_value.values = mock_values
    mock_service = MagicMock()
    mock_service.spreadsheets = mock_spreadsheets
    return mock_service


def _call_loader(rows: list[list[str]], tab_name: str = "LEAD TRACKER") -> dict:
    """Patch credentials + build and call load_lead_tracker."""
    mock_service = _mock_service(rows)
    with (
        patch("collectors.google_sheets.service_account.Credentials.from_service_account_info"),
        patch("collectors.google_sheets.build", return_value=mock_service),
    ):
        from collectors.google_sheets import load_lead_tracker
        return load_lead_tracker(_FAKE_SA_JSON, _SPREADSHEET_ID, tab_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalRow:
    """Fields map to the correct columns."""

    def test_fields_parsed_correctly(self):
        rows = [
            _HEADER,
            _make_row(
                lead_name="Jane Smith",
                org="ACME Corp",
                stage="Demo",
                follow_up_priority="High",
                next_action_date="2026-06-01",
                last_contact_date="2026-05-20",
                initial_contact_date="2026-04-15",
                branch_sector="College",
                email="jane@acme.com",
                notes="Very interested",
            ),
        ]
        result = _call_loader(rows)

        assert "jane smith" in result
        entry = result["jane smith"]
        assert entry["lead_name"] == "Jane Smith"
        assert entry["organization"] == "ACME Corp"
        assert entry["stage"] == "Demo"
        assert entry["follow_up_priority"] == "High"
        assert entry["next_action_date"] == "2026-06-01"
        assert entry["last_contact_date"] == "2026-05-20"
        assert entry["initial_contact_date"] == "2026-04-15"
        assert entry["branch_sector"] == "College"
        assert entry["email"] == "jane@acme.com"
        assert entry["notes"] == "Very interested"

    def test_all_expected_keys_present(self):
        rows = [_HEADER, _make_row(lead_name="Bob")]
        result = _call_loader(rows)
        expected_keys = {
            "row_index", "lead_name", "organization", "stage",
            "follow_up_priority", "next_action_date", "last_contact_date",
            "initial_contact_date", "branch_sector", "email", "notes",
        }
        assert set(result["bob"].keys()) == expected_keys


class TestRowIndex:
    """row_index must be 1-based sheet row number."""

    def test_first_data_row_is_row_2(self):
        rows = [_HEADER, _make_row(lead_name="Alice")]
        result = _call_loader(rows)
        assert result["alice"]["row_index"] == 2

    def test_third_data_row_is_row_4(self):
        rows = [
            _HEADER,
            _make_row(lead_name="Alice"),
            _make_row(lead_name="Bob"),
            _make_row(lead_name="Carol"),
        ]
        result = _call_loader(rows)
        assert result["carol"]["row_index"] == 4


class TestDualKeyLookup:
    """Both normalized lead name and normalized org name are keys."""

    def test_both_keys_present(self):
        rows = [_HEADER, _make_row(lead_name="John Doe", org="Fitness Club")]
        result = _call_loader(rows)
        assert "john doe" in result
        assert "fitness club" in result

    def test_both_keys_point_to_same_data(self):
        rows = [_HEADER, _make_row(lead_name="John Doe", org="Fitness Club")]
        result = _call_loader(rows)
        assert result["john doe"] is result["fitness club"]

    def test_only_name_key_when_org_empty(self):
        rows = [_HEADER, _make_row(lead_name="Solo Rep", org="")]
        result = _call_loader(rows)
        assert "solo rep" in result
        # No org key added (empty string normalizes to "")
        assert "" not in result

    def test_only_org_key_when_name_empty(self):
        rows = [_HEADER, _make_row(lead_name="", org="Nameless Org")]
        result = _call_loader(rows)
        assert "nameless org" in result
        assert "" not in result

    def test_case_insensitive_normalization(self):
        rows = [_HEADER, _make_row(lead_name="  ALICE  JONES  ", org="BIG GYM")]
        result = _call_loader(rows)
        assert "alice jones" in result
        assert "big gym" in result


class TestDuplicateHandling:
    """Duplicate normalized keys get _2, _3 suffix; warning is logged."""

    def test_duplicate_lead_name_gets_suffix(self):
        rows = [
            _HEADER,
            _make_row(lead_name="Jane Smith", org="Org A"),
            _make_row(lead_name="Jane Smith", org="Org B"),
        ]
        result = _call_loader(rows)
        assert "jane smith" in result
        assert "jane smith_2" in result

    def test_duplicate_warning_logged(self, caplog):
        import logging
        rows = [
            _HEADER,
            _make_row(lead_name="Jane Smith", org="Org A"),
            _make_row(lead_name="Jane Smith", org="Org B"),
        ]
        with caplog.at_level(logging.WARNING, logger="collectors.google_sheets"):
            _call_loader(rows)
        assert "duplicate" in caplog.text.lower()
        assert "jane smith" in caplog.text

    def test_triple_duplicate_gets_3_suffix(self):
        rows = [
            _HEADER,
            _make_row(lead_name="Repeat Name", org="Org 1"),
            _make_row(lead_name="Repeat Name", org="Org 2"),
            _make_row(lead_name="Repeat Name", org="Org 3"),
        ]
        result = _call_loader(rows)
        assert "repeat name" in result
        assert "repeat name_2" in result
        assert "repeat name_3" in result

    def test_original_entry_keeps_correct_row_index(self):
        rows = [
            _HEADER,
            _make_row(lead_name="Jane Smith", org="Org A"),
            _make_row(lead_name="Jane Smith", org="Org B"),
        ]
        result = _call_loader(rows)
        assert result["jane smith"]["row_index"] == 2
        assert result["jane smith_2"]["row_index"] == 3

    def test_self_collision_inserts_only_one_key(self):
        """When lead_name and org normalize to the same string, only one key is stored."""
        rows = [
            _HEADER,
            _make_row(lead_name="ACME", org="acme"),
        ]
        result = _call_loader(rows)
        assert "acme" in result
        assert "acme_2" not in result
        assert len([k for k in result if k.startswith("acme")]) == 1


class TestEmptyCellPadding:
    """Rows shorter than 15 columns must be padded safely."""

    def test_short_row_does_not_raise(self):
        rows = [_HEADER, ["Only Name"]]  # Only 1 column
        result = _call_loader(rows)
        assert "only name" in result

    def test_short_row_empty_fields_are_empty_string(self):
        rows = [_HEADER, ["Short Row"]]
        result = _call_loader(rows)
        entry = result["short row"]
        assert entry["organization"] == ""
        assert entry["stage"] == ""
        assert entry["email"] == ""
        assert entry["notes"] == ""

    def test_partial_row_correct_values_for_present_columns(self):
        # Provide A–C only
        rows = [_HEADER, ["Alice", "Best Gym", "Demo"]]
        result = _call_loader(rows)
        entry = result["alice"]
        assert entry["stage"] == "Demo"
        assert entry["follow_up_priority"] == ""


class TestEmptySheet:
    """Empty sheet (no data rows) returns empty dict."""

    def test_no_values_key_in_response(self):
        mock_execute = MagicMock(return_value={})  # No 'values' key
        mock_get = MagicMock()
        mock_get.return_value.execute = mock_execute
        mock_values = MagicMock()
        mock_values.return_value.get = mock_get
        mock_spreadsheets = MagicMock()
        mock_spreadsheets.return_value.values = mock_values
        mock_service = MagicMock()
        mock_service.spreadsheets = mock_spreadsheets

        with (
            patch("collectors.google_sheets.service_account.Credentials.from_service_account_info"),
            patch("collectors.google_sheets.build", return_value=mock_service),
        ):
            from collectors.google_sheets import load_lead_tracker
            result = load_lead_tracker(_FAKE_SA_JSON, _SPREADSHEET_ID)

        assert result == {}

    def test_only_header_row_returns_empty_dict(self):
        rows = [_HEADER]
        result = _call_loader(rows)
        assert result == {}

    def test_empty_values_list_returns_empty_dict(self):
        rows = []
        result = _call_loader(rows)
        assert result == {}

    def test_all_blank_data_rows_returns_empty_dict(self):
        rows = [_HEADER, ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]]
        result = _call_loader(rows)
        assert result == {}
