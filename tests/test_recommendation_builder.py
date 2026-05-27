"""Tests for processors/recommendation_builder.py — all Claude API calls are mocked."""

from __future__ import annotations

import re
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from collectors.avoma import AvomaTranscript
from processors.recommendation_builder import Recommendation, build_recommendation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_API_KEY = "sk-fake-key"


def _make_transcript(**kwargs) -> AvomaTranscript:
    defaults = dict(
        uuid="uuid-001",
        title="Demo Call with Acme",
        start_at="2026-05-23T14:00:00Z",
        participants=["Jane Smith", "Bob Rep"],
        call_type="demo",
        summary="Called to demo Strength module. Prospect showed strong interest.",
        buying_signals=["Asked about pricing", "Requested contract terms"],
        objections=["Price concern"],
        action_items=["Send pricing doc", "Schedule follow-up"],
        competitors=[],
    )
    defaults.update(kwargs)
    return AvomaTranscript(**defaults)


def _make_current_row(**kwargs) -> dict:
    defaults = dict(
        row_index=5,
        lead_name="Jane Smith",
        organization="Acme Gym",
        stage="Demo",
        follow_up_priority="High",
        next_action_date="2026-06-01",
        last_contact_date="2026-05-10",
        initial_contact_date="2026-04-01",
        branch_sector="College",
        email="jane@acme.com",
        notes="Very interested",
    )
    defaults.update(kwargs)
    return defaults


def _make_tool_use_block(input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "propose_crm_updates"
    block.input = input_data
    return block


def _valid_claude_response(proposed_updates: dict | None = None) -> MagicMock:
    """Build a mock Anthropic response with a valid propose_crm_updates tool call."""
    today = date.today().isoformat()
    updates = proposed_updates if proposed_updates is not None else {
        "Stage": "Demo",
        "Follow-Up Priority": "High",
        "Last Contact Date": today,
    }
    tool_input = {
        "match_confidence": "high",
        "lead_name": "Jane Smith",
        "organization": "Acme Gym",
        "proposed_updates": updates,
        "note": "Conducted demo; prospect requested pricing doc.",
        "reasoning": "Moved to Demo stage due to strong buying signals.",
    }
    response = MagicMock()
    response.content = [_make_tool_use_block(tool_input)]
    return response


def _mock_anthropic(response: MagicMock):
    """Context manager that patches anthropic.Anthropic and returns the given response."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = response
    return patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client)


# ---------------------------------------------------------------------------
# Test 1: Normal case — valid recommendation returned
# ---------------------------------------------------------------------------

class TestNormalCase:
    def test_returns_recommendation_dataclass(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        response = _valid_claude_response()

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert isinstance(rec, Recommendation)

    def test_match_confidence_correct(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        response = _valid_claude_response()

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec.match_confidence == "high"

    def test_lead_name_and_org_correct(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        response = _valid_claude_response()

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec.lead_name == "Jane Smith"
        assert rec.organization == "Acme Gym"

    def test_sheet_row_from_current_row(self):
        transcript = _make_transcript()
        current_row = _make_current_row(row_index=7)
        response = _valid_claude_response()

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec.sheet_row == 7

    def test_note_and_reasoning_populated(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        response = _valid_claude_response()

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec.note == "Conducted demo; prospect requested pricing doc."
        assert "Demo stage" in rec.reasoning

    def test_proposed_updates_passed_through(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Demo",
            "Follow-Up Priority": "High",
            "Last Contact Date": today,
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec.proposed_updates["Stage"] == "Demo"
        assert rec.proposed_updates["Follow-Up Priority"] == "High"
        assert rec.proposed_updates["Last Contact Date"] == today


# ---------------------------------------------------------------------------
# Test 2: Formula columns dropped
# ---------------------------------------------------------------------------

class TestFormulaColumnsDropped:
    def test_days_until_follow_up_removed(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Trial",
            "Last Contact Date": today,
            "Days Until Follow-Up": "5",
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert "Days Until Follow-Up" not in rec.proposed_updates

    def test_days_since_last_contact_removed(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Trial",
            "Last Contact Date": today,
            "Days Since Last Contact": "3",
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert "Days Since Last Contact" not in rec.proposed_updates

    def test_both_formula_columns_removed_and_rest_kept(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Quoted",
            "Last Contact Date": today,
            "Days Until Follow-Up": "2",
            "Days Since Last Contact": "0",
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert "Days Until Follow-Up" not in rec.proposed_updates
        assert "Days Since Last Contact" not in rec.proposed_updates
        assert rec.proposed_updates["Stage"] == "Quoted"
        assert rec.proposed_updates["Last Contact Date"] == today


# ---------------------------------------------------------------------------
# Test 3: Returns None on malformed Claude output (no tool_use block)
# ---------------------------------------------------------------------------

class TestMalformedOutput:
    def test_returns_none_when_no_tool_use_block(self):
        transcript = _make_transcript()
        current_row = _make_current_row()

        # Response with a text block instead of tool_use
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"match_confidence": "high"}'

        response = MagicMock()
        response.content = [text_block]

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None

    def test_returns_none_when_content_is_empty(self):
        transcript = _make_transcript()
        current_row = _make_current_row()

        response = MagicMock()
        response.content = []

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None

    def test_returns_none_when_tool_use_missing_required_fields(self):
        transcript = _make_transcript()
        current_row = _make_current_row()

        # Tool use block missing "note" and "reasoning"
        incomplete_input = {
            "match_confidence": "high",
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": {},
            # "note" and "reasoning" missing
        }
        response = MagicMock()
        response.content = [_make_tool_use_block(incomplete_input)]

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None

    def test_returns_none_when_wrong_tool_name(self):
        transcript = _make_transcript()
        current_row = _make_current_row()

        wrong_tool_block = MagicMock()
        wrong_tool_block.type = "tool_use"
        wrong_tool_block.name = "some_other_tool"
        wrong_tool_block.input = {"foo": "bar"}

        response = MagicMock()
        response.content = [wrong_tool_block]

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None

    def test_returns_none_when_api_raises_exception(self):
        transcript = _make_transcript()
        current_row = _make_current_row()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("network error")

        with patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None


# ---------------------------------------------------------------------------
# Test 4: current_row=None (new lead) — prompt includes "No existing row"
# ---------------------------------------------------------------------------

class TestNewLead:
    def test_prompt_includes_no_existing_row(self):
        transcript = _make_transcript()

        mock_client = MagicMock()
        response = MagicMock()
        today = date.today().isoformat()
        tool_input = {
            "match_confidence": "new_lead",
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": {
                "Stage": "Initial Contact",
                "Last Contact Date": today,
                "Initial Contact Date": today,
            },
            "note": "Initial demo held; prospect requested follow-up.",
            "reasoning": "New lead with no prior sheet entry.",
        }
        response.content = [_make_tool_use_block(tool_input)]
        mock_client.messages.create.return_value = response

        with patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client):
            build_recommendation(transcript, None, _MODEL, _API_KEY)

        # Verify the prompt passed to Claude contains the "No existing row" message
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        assert "No existing row" in user_content

    def test_sheet_row_is_none_for_new_lead(self):
        transcript = _make_transcript()

        today = date.today().isoformat()
        tool_input = {
            "match_confidence": "new_lead",
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": {
                "Stage": "Initial Contact",
                "Last Contact Date": today,
            },
            "note": "Initial demo held.",
            "reasoning": "New lead.",
        }
        response = MagicMock()
        response.content = [_make_tool_use_block(tool_input)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response

        with patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client):
            rec = build_recommendation(transcript, None, _MODEL, _API_KEY)

        assert rec is not None
        assert rec.sheet_row is None

    def test_match_confidence_is_new_lead(self):
        transcript = _make_transcript()
        today = date.today().isoformat()
        tool_input = {
            "match_confidence": "new_lead",
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": {"Last Contact Date": today},
            "note": "Initial call.",
            "reasoning": "New lead.",
        }
        response = MagicMock()
        response.content = [_make_tool_use_block(tool_input)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response

        with patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client):
            rec = build_recommendation(transcript, None, _MODEL, _API_KEY)

        assert rec.match_confidence == "new_lead"


# ---------------------------------------------------------------------------
# Test 5: Last Contact Date is always today
# ---------------------------------------------------------------------------

class TestLastContactDateIsToday:
    def test_last_contact_date_in_proposed_updates(self):
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Demo",
            "Last Contact Date": today,
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert "Last Contact Date" in rec.proposed_updates

    def test_last_contact_date_is_today_format(self):
        """Verify the date looks like YYYY-MM-DD."""
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response(proposed_updates={
            "Stage": "Demo",
            "Last Contact Date": today,
        })

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        last_contact = rec.proposed_updates["Last Contact Date"]
        # Should match YYYY-MM-DD format
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", last_contact), (
            f"Expected YYYY-MM-DD, got {last_contact!r}"
        )

    def test_prompt_contains_todays_date(self):
        """The user prompt instructs Claude to use today's date."""
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()
        response = _valid_claude_response()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response

        with patch("processors.recommendation_builder.anthropic.Anthropic", return_value=mock_client):
            build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_content = messages[0]["content"]
        assert today in user_content


# ---------------------------------------------------------------------------
# Test 6: New validation paths
# ---------------------------------------------------------------------------

class TestNewValidationPaths:
    def test_returns_none_when_proposed_updates_is_not_dict(self):
        """Claude returns proposed_updates as a string (e.g. 'none') — should return None."""
        transcript = _make_transcript()
        current_row = _make_current_row()

        tool_input = {
            "match_confidence": "high",
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": "none",  # invalid — should be a dict
            "note": "Demo held.",
            "reasoning": "Strong interest.",
        }
        response = MagicMock()
        response.content = [_make_tool_use_block(tool_input)]

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None

    def test_returns_none_when_match_confidence_is_invalid(self):
        """Claude returns an unexpected match_confidence value — should return None."""
        transcript = _make_transcript()
        current_row = _make_current_row()
        today = date.today().isoformat()

        tool_input = {
            "match_confidence": "unknown",  # invalid value
            "lead_name": "Jane Smith",
            "organization": "Acme Gym",
            "proposed_updates": {"Last Contact Date": today},
            "note": "Demo held.",
            "reasoning": "Strong interest.",
        }
        response = MagicMock()
        response.content = [_make_tool_use_block(tool_input)]

        with _mock_anthropic(response):
            rec = build_recommendation(transcript, current_row, _MODEL, _API_KEY)

        assert rec is None
