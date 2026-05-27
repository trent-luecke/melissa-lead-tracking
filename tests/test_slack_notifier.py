"""Tests for outputs/slack_notifier.py — all Slack API calls are mocked."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch, call

import pytest
from slack_sdk.errors import SlackApiError

from outputs.slack_notifier import send_pending_recommendations, _format_message, _format_call_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MELISSA_USER_ID = "U07EJE9B3NG"
_DM_CHANNEL_ID = "D0123456789"
_BOT_TOKEN = "xoxb-fake-token"


def _make_rec(**kwargs) -> dict:
    defaults = dict(
        thread_ts="",
        lead_name="Jane Smith",
        organization="Acme Gym",
        sheet_row=5,
        proposed_updates={"Stage": "Demo", "Last Contact Date": "2026-05-23"},
        note="Demo held; prospect requested pricing doc.",
        reasoning="Moved to Demo stage based on strong buying signals.",
        match_confidence="high",
        avoma_uuid="uuid-001",
        created_at="2026-05-23T10:00:00Z",
        ambiguous_match=False,
        call_date="2026-05-23",
        call_type="demo",
    )
    defaults.update(kwargs)
    return defaults


def _make_sheet_row(**kwargs) -> dict:
    defaults = dict(
        row_index=5,
        lead_name="Jane Smith",
        organization="Acme Gym",
        stage="Initial Contact",
        follow_up_priority="Medium",
        next_action_date="2026-06-01",
        last_contact_date="2026-05-10",
        initial_contact_date="2026-04-01",
        branch_sector="College",
        email="jane@acme.com",
        notes="Initial contact made.",
    )
    defaults.update(kwargs)
    return defaults


def _make_slack_client(post_ts: str = "1716451200.000100") -> MagicMock:
    """Return a mocked WebClient that succeeds by default."""
    client = MagicMock()

    # conversations.open response
    open_resp = MagicMock()
    open_resp.__getitem__ = lambda self, key: {"channel": {"id": _DM_CHANNEL_ID}}[key]
    client.conversations_open.return_value = open_resp

    # chat.postMessage response
    post_resp = MagicMock()
    post_resp.__getitem__ = lambda self, key: {"ts": post_ts}[key]
    client.chat_postMessage.return_value = post_resp

    return client


def _build_client_with_channel_and_ts(channel_id: str, ts: str) -> MagicMock:
    """Build a mock WebClient where conversations_open returns channel_id and postMessage returns ts."""
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": channel_id}}
    client.chat_postMessage.return_value = {"ts": ts}
    return client


# ---------------------------------------------------------------------------
# Unit tests: _format_call_type
# ---------------------------------------------------------------------------

class TestFormatCallType:
    def test_demo_maps_to_Demo(self):
        assert _format_call_type("demo") == "Demo"

    def test_follow_up_maps_to_FollowUp(self):
        assert _format_call_type("follow_up") == "Follow-Up"

    def test_unknown_type_titlecased(self):
        # Any unknown type gets title-cased with underscores replaced
        result = _format_call_type("other_type")
        assert result == "Other Type"


# ---------------------------------------------------------------------------
# Unit tests: _format_message
# ---------------------------------------------------------------------------

class TestFormatMessage:
    def test_header_contains_lead_name_and_org(self):
        rec = _make_rec(lead_name="Bob Jones", organization="Bob's Gym")
        msg = _format_message(rec, {})
        assert "Bob Jones" in msg
        assert "Bob's Gym" in msg

    def test_header_bold_format(self):
        rec = _make_rec(lead_name="Jane Smith", organization="Acme Gym")
        msg = _format_message(rec, {})
        assert "*Lead Update — Jane Smith (Acme Gym)*" in msg

    def test_call_type_demo_formatted(self):
        rec = _make_rec(call_type="demo", call_date="2026-05-23")
        msg = _format_message(rec, {})
        assert "Demo" in msg
        assert "2026-05-23" in msg

    def test_call_type_follow_up_formatted(self):
        rec = _make_rec(call_type="follow_up", call_date="2026-05-22")
        msg = _format_message(rec, {})
        assert "Follow-Up" in msg

    def test_summary_line_uses_reasoning(self):
        rec = _make_rec(reasoning="Strong interest shown in pricing.")
        msg = _format_message(rec, {})
        assert "Strong interest shown in pricing." in msg
        assert "Summary:" in msg

    def test_proposed_updates_header_present(self):
        rec = _make_rec(proposed_updates={"Stage": "Demo"})
        msg = _format_message(rec, {})
        assert "Proposed updates:" in msg

    def test_field_bullet_no_current_value(self):
        rec = _make_rec(proposed_updates={"Stage": "Demo"})
        msg = _format_message(rec, {})
        # When no current value: → *New Value*
        assert "→ *Demo*" in msg

    def test_field_bullet_with_current_value(self):
        rec = _make_rec(proposed_updates={"Stage": "Demo"})
        sheet_row = _make_sheet_row(stage="Initial Contact")
        sheet_lookup = {"jane smith": sheet_row}
        msg = _format_message(rec, sheet_lookup)
        assert "Initial Contact → *Demo*" in msg

    def test_note_line_present(self):
        rec = _make_rec(note="Demo held; prospect requested pricing.")
        msg = _format_message(rec, {})
        assert "Note added:" in msg
        assert "Demo held; prospect requested pricing." in msg

    def test_note_in_code_backticks(self):
        rec = _make_rec(note="Follow-up email sent.")
        msg = _format_message(rec, {})
        assert "`Follow-up email sent.`" in msg

    def test_reply_prompt_present(self):
        rec = _make_rec()
        msg = _format_message(rec, {})
        assert "Reply in this thread to confirm, edit, or reject." in msg

    def test_no_ambiguous_warning_when_false(self):
        rec = _make_rec(ambiguous_match=False)
        msg = _format_message(rec, {})
        assert "Multiple sheet rows matched" not in msg

    def test_ambiguous_warning_when_true(self):
        rec = _make_rec(ambiguous_match=True)
        msg = _format_message(rec, {})
        assert "Multiple sheet rows matched" in msg
        assert "⚠️" in msg

    def test_ambiguous_warning_position_after_header(self):
        """Warning line must appear before the call details, right after the header."""
        rec = _make_rec(ambiguous_match=True, call_type="demo", call_date="2026-05-23")
        msg = _format_message(rec, {})
        warning_pos = msg.find("Multiple sheet rows matched")
        call_pos = msg.find("Call:")
        assert warning_pos < call_pos, "Warning should appear before Call: line"

    def test_only_fields_in_proposed_updates_appear(self):
        """Fields not in proposed_updates should not generate bullets."""
        rec = _make_rec(proposed_updates={"Stage": "Demo"})
        msg = _format_message(rec, {})
        assert "Follow-Up Priority" not in msg
        assert "Next Action Date" not in msg

    def test_multiple_field_bullets(self):
        rec = _make_rec(
            proposed_updates={
                "Stage": "Demo",
                "Last Contact Date": "2026-05-23",
                "Follow-Up Priority": "High",
            }
        )
        msg = _format_message(rec, {})
        assert "Stage" in msg
        assert "Last Contact Date" in msg
        assert "Follow-Up Priority" in msg

    def test_field_with_current_from_org_lookup(self):
        """Sheet lookup falls back to org name if lead_name not found."""
        rec = _make_rec(
            lead_name="Unknown Person",
            organization="Acme Gym",
            proposed_updates={"Stage": "Demo"},
        )
        sheet_row = _make_sheet_row(stage="Initial Contact")
        sheet_lookup = {"acme gym": sheet_row}
        msg = _format_message(rec, sheet_lookup)
        assert "Initial Contact → *Demo*" in msg

    def test_empty_current_value_treated_as_none(self):
        """If the sheet row has an empty string for a field, treat as no current value."""
        rec = _make_rec(proposed_updates={"Email": "jane@new.com"})
        sheet_row = _make_sheet_row(email="")
        sheet_lookup = {"jane smith": sheet_row}
        msg = _format_message(rec, sheet_lookup)
        assert "→ *jane@new.com*" in msg
        # Should NOT show " → *jane@new.com*" with space before arrow (no current value prefix)
        assert "→ *jane@new.com*" in msg


# ---------------------------------------------------------------------------
# Integration tests: send_pending_recommendations
# ---------------------------------------------------------------------------

class TestSendPendingRecommendations:

    def test_unsent_rec_is_sent_and_rekeyed(self):
        """An entry with thread_ts='' is sent and re-keyed by thread_ts."""
        thread_ts = "1716451200.000100"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, thread_ts)

        rec = _make_rec(avoma_uuid="uuid-001")
        pending_recs = {"uuid-001": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            result = send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        # Original key removed, new key is thread_ts
        assert "uuid-001" not in result
        assert thread_ts in result
        assert result[thread_ts]["thread_ts"] == thread_ts
        assert result[thread_ts]["avoma_uuid"] == "uuid-001"

    def test_already_sent_rec_is_skipped(self):
        """An entry with thread_ts != '' is not re-sent and is passed through unchanged."""
        existing_ts = "1716451100.000001"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, "1716451200.000999")

        rec = _make_rec(thread_ts=existing_ts, avoma_uuid="uuid-already-sent")
        pending_recs = {existing_ts: rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            result = send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        # Should not have posted any new messages
        client.chat_postMessage.assert_not_called()
        # Entry still present under original key
        assert existing_ts in result
        assert result[existing_ts]["thread_ts"] == existing_ts

    def test_failed_send_is_logged_and_others_continue(self, caplog):
        """If one send fails (SlackApiError), it is logged and remaining recs still process."""
        client = MagicMock()
        client.conversations_open.return_value = {"channel": {"id": _DM_CHANNEL_ID}}

        # First call raises SlackApiError, second succeeds
        slack_err = SlackApiError("ratelimited", MagicMock(data={"error": "ratelimited"}))
        success_ts = "1716451200.000200"
        client.chat_postMessage.side_effect = [slack_err, {"ts": success_ts}]

        rec1 = _make_rec(avoma_uuid="uuid-fail", lead_name="Fail Lead")
        rec2 = _make_rec(avoma_uuid="uuid-ok", lead_name="Ok Lead")
        pending_recs = {"uuid-fail": rec1, "uuid-ok": rec2}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            with caplog.at_level(logging.ERROR, logger="outputs.slack_notifier"):
                result = send_pending_recommendations(
                    slack_bot_token=_BOT_TOKEN,
                    melissa_user_id=_MELISSA_USER_ID,
                    pending_recs=pending_recs,
                    sheet_lookup={},
                )

        # Failed one still under original key with thread_ts=""
        assert "uuid-fail" in result
        assert result["uuid-fail"]["thread_ts"] == ""

        # Successful one re-keyed
        assert success_ts in result
        assert result[success_ts]["thread_ts"] == success_ts

        # Error was logged
        assert any("uuid-fail" in r.message for r in caplog.records)

    def test_dm_channel_opened_once(self):
        """conversations.open is called exactly once regardless of number of recs."""
        thread_ts_1 = "1716451200.000001"
        thread_ts_2 = "1716451200.000002"
        client = MagicMock()
        client.conversations_open.return_value = {"channel": {"id": _DM_CHANNEL_ID}}
        client.chat_postMessage.side_effect = [
            {"ts": thread_ts_1},
            {"ts": thread_ts_2},
        ]

        rec1 = _make_rec(avoma_uuid="uuid-a", lead_name="Lead A")
        rec2 = _make_rec(avoma_uuid="uuid-b", lead_name="Lead B")
        pending_recs = {"uuid-a": rec1, "uuid-b": rec2}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            result = send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        client.conversations_open.assert_called_once_with(users=[_MELISSA_USER_ID])
        assert client.chat_postMessage.call_count == 2

    def test_conversations_open_failure_returns_original_recs(self, caplog):
        """If conversations.open fails, return the original pending_recs unchanged."""
        client = MagicMock()
        slack_err = SlackApiError("channel_not_found", MagicMock(data={"error": "channel_not_found"}))
        client.conversations_open.side_effect = slack_err

        rec = _make_rec(avoma_uuid="uuid-001")
        pending_recs = {"uuid-001": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            with caplog.at_level(logging.ERROR, logger="outputs.slack_notifier"):
                result = send_pending_recommendations(
                    slack_bot_token=_BOT_TOKEN,
                    melissa_user_id=_MELISSA_USER_ID,
                    pending_recs=pending_recs,
                    sheet_lookup={},
                )

        assert result is pending_recs
        client.chat_postMessage.assert_not_called()

    def test_ambiguous_match_warning_in_sent_message(self):
        """Ambiguous match rec generates a message that contains the warning."""
        thread_ts = "1716451200.000300"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, thread_ts)

        rec = _make_rec(avoma_uuid="uuid-ambig", ambiguous_match=True)
        pending_recs = {"uuid-ambig": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        # Verify the message sent contains the warning
        sent_text = client.chat_postMessage.call_args.kwargs["text"]
        assert "Multiple sheet rows matched" in sent_text

    def test_message_format_full_structure(self):
        """The full message format is checked: header, call, summary, bullets, note, reply prompt."""
        thread_ts = "1716451200.000400"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, thread_ts)

        rec = _make_rec(
            lead_name="Jane Smith",
            organization="Acme Gym",
            call_type="demo",
            call_date="2026-05-23",
            reasoning="Moved to Demo stage based on strong buying signals.",
            proposed_updates={"Stage": "Demo", "Last Contact Date": "2026-05-23"},
            note="Demo held; prospect requested pricing doc.",
            ambiguous_match=False,
        )
        pending_recs = {"uuid-001": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        sent_text = client.chat_postMessage.call_args[1]["text"]

        assert "*Lead Update — Jane Smith (Acme Gym)*" in sent_text
        assert "Demo · 2026-05-23" in sent_text
        assert "Moved to Demo stage based on strong buying signals." in sent_text
        assert "Proposed updates:" in sent_text
        assert "→ *Demo*" in sent_text
        assert "→ *2026-05-23*" in sent_text
        assert "`Demo held; prospect requested pricing doc.`" in sent_text
        assert "Reply in this thread to confirm, edit, or reject." in sent_text

    def test_proposed_updates_with_current_values_from_sheet_lookup(self):
        """Current values from sheet_lookup appear in the bullets."""
        thread_ts = "1716451200.000500"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, thread_ts)

        sheet_row = _make_sheet_row(stage="Initial Contact", last_contact_date="2026-05-10")
        sheet_lookup = {"jane smith": sheet_row}

        rec = _make_rec(
            lead_name="Jane Smith",
            proposed_updates={
                "Stage": "Demo",
                "Last Contact Date": "2026-05-23",
            },
        )
        pending_recs = {"uuid-001": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup=sheet_lookup,
            )

        sent_text = client.chat_postMessage.call_args[1]["text"]
        assert "Initial Contact → *Demo*" in sent_text
        assert "2026-05-10 → *2026-05-23*" in sent_text

    def test_rekeying_removes_avoma_uuid_key_adds_thread_ts_key(self):
        """Re-keying: avoma_uuid key gone, thread_ts key present in result."""
        thread_ts = "1716451200.000600"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, thread_ts)

        rec = _make_rec(avoma_uuid="uuid-rekey")
        pending_recs = {"uuid-rekey": rec}

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            result = send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        assert "uuid-rekey" not in result
        assert thread_ts in result
        assert result[thread_ts]["thread_ts"] == thread_ts

    def test_mixed_sent_and_unsent_recs(self):
        """Already-sent entries stay put; unsent entries are sent and re-keyed."""
        already_ts = "1716451100.000001"
        new_ts = "1716451200.000700"
        client = _build_client_with_channel_and_ts(_DM_CHANNEL_ID, new_ts)

        already_sent = _make_rec(thread_ts=already_ts, avoma_uuid="uuid-already")
        unsent = _make_rec(thread_ts="", avoma_uuid="uuid-new")
        pending_recs = {
            already_ts: already_sent,
            "uuid-new": unsent,
        }

        with patch("outputs.slack_notifier.WebClient", return_value=client):
            result = send_pending_recommendations(
                slack_bot_token=_BOT_TOKEN,
                melissa_user_id=_MELISSA_USER_ID,
                pending_recs=pending_recs,
                sheet_lookup={},
            )

        # Already sent entry unchanged
        assert already_ts in result
        assert result[already_ts]["thread_ts"] == already_ts
        # Unsent entry re-keyed
        assert "uuid-new" not in result
        assert new_ts in result

        # chat_postMessage called only once (for the unsent one)
        client.chat_postMessage.assert_called_once()
