"""Tests for outputs/slack_reply_handler.py — all Claude API calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from outputs.slack_reply_handler import ReplyAction, interpret_reply

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_API_KEY = "sk-fake-key"

_BASE_REC = {
    "thread_ts": "1234567890.123456",
    "lead_name": "Jane Smith",
    "organization": "Acme Gym",
    "sheet_row": 5,
    "proposed_updates": {
        "Stage": "Demo",
        "Last Contact Date": "2026-05-23",
        "Follow-Up Priority": "High",
    },
    "note": "Demo held; prospect requested pricing doc.",
    "reasoning": "Moved to Demo stage.",
    "match_confidence": "high",
}


def _make_tool_use_block(action: str, updates: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "interpret_reply"
    block.input = {"action": action, "updates": updates}
    return block


def _mock_claude_response(action: str, updates: dict) -> MagicMock:
    response = MagicMock()
    response.content = [_make_tool_use_block(action, updates)]
    return response


def _mock_anthropic(response: MagicMock):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = response
    return patch("outputs.slack_reply_handler.anthropic.Anthropic", return_value=mock_client)


# ---------------------------------------------------------------------------
# Test 1: "confirm" action → updates = original proposed_updates
# ---------------------------------------------------------------------------

class TestConfirmAction:
    def test_confirm_returns_original_proposed_updates(self):
        """confirm ignores Claude's updates field and uses original proposed_updates."""
        # Claude returns different updates — should be ignored
        claude_updates = {"Stage": "Trial"}
        response = _mock_claude_response("confirm", claude_updates)

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.action == "confirm"
        assert result.updates == _BASE_REC["proposed_updates"]

    def test_confirm_updates_match_proposed_updates_exactly(self):
        """Confirm: updates must exactly equal recommendation proposed_updates."""
        response = _mock_claude_response("confirm", {})

        with _mock_anthropic(response):
            result = interpret_reply("looks good, confirm", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.updates["Stage"] == "Demo"
        assert result.updates["Last Contact Date"] == "2026-05-23"
        assert result.updates["Follow-Up Priority"] == "High"

    def test_confirm_action_is_set(self):
        response = _mock_claude_response("confirm", {})

        with _mock_anthropic(response):
            result = interpret_reply("confirmed", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.action == "confirm"


# ---------------------------------------------------------------------------
# Test 2: "edit" action → updates = Claude's merged updates
# ---------------------------------------------------------------------------

class TestEditAction:
    def test_edit_returns_claude_updates(self):
        """edit uses Claude's returned updates (merged/corrected by Claude)."""
        claude_updates = {
            "Stage": "Trial",
            "Last Contact Date": "2026-05-23",
        }
        response = _mock_claude_response("edit", claude_updates)

        with _mock_anthropic(response):
            result = interpret_reply("change stage to Trial", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.action == "edit"
        assert result.updates["Stage"] == "Trial"
        assert result.updates["Last Contact Date"] == "2026-05-23"

    def test_edit_does_not_use_original_proposed_updates(self):
        """edit uses Claude's output, not the original proposed_updates."""
        claude_updates = {"Stage": "Quoted", "Follow-Up Priority": "Low"}
        response = _mock_claude_response("edit", claude_updates)

        with _mock_anthropic(response):
            result = interpret_reply("update priority to low and stage to quoted", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.updates == {"Stage": "Quoted", "Follow-Up Priority": "Low"}


# ---------------------------------------------------------------------------
# Test 3: "reject" action → updates = {}
# ---------------------------------------------------------------------------

class TestRejectAction:
    def test_reject_returns_empty_updates(self):
        """reject always returns empty updates dict."""
        response = _mock_claude_response("reject", {"Stage": "Demo"})  # Claude might fill; should be ignored

        with _mock_anthropic(response):
            result = interpret_reply("no, skip this", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.action == "reject"
        assert result.updates == {}

    def test_reject_action_is_set(self):
        response = _mock_claude_response("reject", {})

        with _mock_anthropic(response):
            result = interpret_reply("discard this recommendation", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert result.action == "reject"


# ---------------------------------------------------------------------------
# Test 4: Returns None on malformed Claude output
# ---------------------------------------------------------------------------

class TestMalformedOutput:
    def test_returns_none_when_no_tool_use_block(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"action": "confirm"}'

        response = MagicMock()
        response.content = [text_block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_returns_none_when_content_is_empty(self):
        response = MagicMock()
        response.content = []

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_returns_none_when_api_raises_exception(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("network error")

        with patch("outputs.slack_reply_handler.anthropic.Anthropic", return_value=mock_client):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_returns_none_when_updates_is_not_dict(self):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "interpret_reply"
        block.input = {"action": "confirm", "updates": "none"}  # invalid

        response = MagicMock()
        response.content = [block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_returns_none_when_wrong_tool_name(self):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "some_other_tool"
        block.input = {"action": "confirm", "updates": {}}

        response = MagicMock()
        response.content = [block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None


# ---------------------------------------------------------------------------
# Test 5: Formula columns dropped from edit updates
# ---------------------------------------------------------------------------

class TestFormulaColumnsDropped:
    def test_days_until_follow_up_dropped_from_edit(self):
        claude_updates = {
            "Stage": "Demo",
            "Days Until Follow-Up": "5",
        }
        response = _mock_claude_response("edit", claude_updates)

        with _mock_anthropic(response):
            result = interpret_reply("change stage to demo", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert "Days Until Follow-Up" not in result.updates
        assert result.updates["Stage"] == "Demo"

    def test_days_since_last_contact_dropped_from_edit(self):
        claude_updates = {
            "Stage": "Trial",
            "Days Since Last Contact": "3",
        }
        response = _mock_claude_response("edit", claude_updates)

        with _mock_anthropic(response):
            result = interpret_reply("stage to trial", _BASE_REC, _MODEL, _API_KEY)

        assert result is not None
        assert "Days Since Last Contact" not in result.updates
        assert result.updates["Stage"] == "Trial"

    def test_formula_columns_dropped_from_confirm_if_present_in_proposed(self):
        """If proposed_updates somehow contains formula columns, they get stripped."""
        rec_with_formula = dict(_BASE_REC)
        rec_with_formula["proposed_updates"] = {
            "Stage": "Demo",
            "Days Until Follow-Up": "2",
        }
        response = _mock_claude_response("confirm", {})

        with _mock_anthropic(response):
            result = interpret_reply("yes", rec_with_formula, _MODEL, _API_KEY)

        assert result is not None
        assert "Days Until Follow-Up" not in result.updates
        assert result.updates["Stage"] == "Demo"


# ---------------------------------------------------------------------------
# Test 6: action validation — invalid value returns None
# ---------------------------------------------------------------------------

class TestActionValidation:
    def test_invalid_action_returns_none(self):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "interpret_reply"
        block.input = {"action": "approve", "updates": {}}  # not in enum

        response = MagicMock()
        response.content = [block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_none_action_returns_none(self):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "interpret_reply"
        block.input = {"action": None, "updates": {}}

        response = MagicMock()
        response.content = [block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None

    def test_missing_action_returns_none(self):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "interpret_reply"
        block.input = {"updates": {}}  # no "action" key

        response = MagicMock()
        response.content = [block]

        with _mock_anthropic(response):
            result = interpret_reply("yes", _BASE_REC, _MODEL, _API_KEY)

        assert result is None
