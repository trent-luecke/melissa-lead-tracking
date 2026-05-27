"""Tests for handle_reply.py — all external calls are mocked."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from outputs.slack_reply_handler import ReplyAction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV = {
    "ANTHROPIC_API_KEY": "sk-fake-key",
    "SLACK_BOT_TOKEN": "xoxb-fake-token",
    "GOOGLE_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}',
    "GOOGLE_SHEET_ID": "fake-sheet-id",
    "THREAD_TS": "1234567890.123456",
    "REPLY_TEXT": "yes, looks good",
}

_CONFIG = {
    "avoma": {},
    "google_sheets": {
        "lead_tracker_tab": "LEAD TRACKER",
    },
    "slack": {"melissa_user_id": "U07EJE9B3NG"},
    "ai_model": "claude-sonnet-4-6",
    "pending_recs_file": "data/pending_recs.json",
    "processed_uuids_file": "data/processed_uuids.json",
    "pending_rec_ttl_days": 7,
}

_SAMPLE_REC = {
    "thread_ts": "1234567890.123456",
    "lead_name": "Jane Smith",
    "organization": "Acme Gym",
    "sheet_row": 5,
    "proposed_updates": {
        "Stage": "Demo",
        "Last Contact Date": "2026-05-23",
    },
    "note": "Demo held; prospect requested pricing doc.",
    "reasoning": "Moved to Demo stage.",
    "match_confidence": "high",
    "avoma_uuid": "uuid-001",
}


def _make_pending_recs(thread_ts: str = "1234567890.123456", rec: dict | None = None) -> dict:
    r = rec if rec is not None else dict(_SAMPLE_REC)
    return {thread_ts: r}


def _patched_open(config: dict, data_dir: Path):
    """
    Return a wrapper for builtins.open that intercepts config.json reads
    and passes all other opens through.
    """
    real_open = open

    def wrapper(file, mode="r", *args, **kwargs):
        file_str = str(file)
        if file_str.endswith("config.json") and "w" not in mode:
            import io
            return io.StringIO(json.dumps(config))
        return real_open(file, mode, *args, **kwargs)

    return wrapper


def _run_handle_reply(
    tmp_path: Path,
    pending_recs: dict,
    env_overrides: dict | None = None,
    config_overrides: dict | None = None,
    interpret_reply_return=None,
    write_updates_side_effect=None,
    dm_channel_id: str = "D123CHANNEL",
):
    """
    Set up state files and mocks, then run handle_reply.main().
    Returns (pending_recs_after, dm_calls) where dm_calls is the list of
    chat_postMessage call kwargs.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pending_path = data_dir / "pending_recs.json"
    pending_path.write_text(json.dumps(pending_recs))

    cfg = dict(_CONFIG)
    cfg["pending_recs_file"] = str(pending_path)
    if config_overrides:
        cfg.update(config_overrides)

    env = dict(_ENV)
    if env_overrides:
        env.update(env_overrides)

    # Build mock Slack client
    mock_slack_client = MagicMock()
    mock_slack_client.conversations_open.return_value = {
        "channel": {"id": dm_channel_id}
    }
    mock_slack_client.chat_postMessage.return_value = {"ts": "9999999999.000001"}

    import handle_reply as hr

    with (
        patch.dict(os.environ, env, clear=False),
        patch("handle_reply.load_dotenv"),
        patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
        patch("handle_reply.interpret_reply", return_value=interpret_reply_return),
        patch("handle_reply.WebClient", return_value=mock_slack_client),
        patch(
            "handle_reply.sheets_writer.write_updates",
            side_effect=write_updates_side_effect,
        ),
    ):
        hr.main()

    pending_after = json.loads(pending_path.read_text()) if pending_path.exists() else {}
    dm_calls = mock_slack_client.chat_postMessage.call_args_list
    return pending_after, dm_calls


# ---------------------------------------------------------------------------
# Test 1: thread_ts not in pending → exits cleanly (exit 0)
# ---------------------------------------------------------------------------

class TestThreadTsNotFound:
    def test_exits_cleanly_when_thread_ts_not_found(self, tmp_path):
        env = dict(_ENV, THREAD_TS="9999999999.000000")
        pending_recs = _make_pending_recs(thread_ts="1234567890.123456")

        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_path.write_text(json.dumps(pending_recs))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        with (
            patch.dict(os.environ, env, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply") as mock_interp,
            patch("handle_reply.WebClient"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                hr.main()

        assert exc_info.value.code == 0
        # interpret_reply should NOT have been called
        mock_interp.assert_not_called()

    def test_pending_recs_unchanged_when_thread_not_found(self, tmp_path):
        env = dict(_ENV, THREAD_TS="9999999999.000000")
        original_recs = _make_pending_recs(thread_ts="1234567890.123456")

        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_path.write_text(json.dumps(original_recs))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        with (
            patch.dict(os.environ, env, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply"),
            patch("handle_reply.WebClient"),
        ):
            with pytest.raises(SystemExit):
                hr.main()

        # pending recs file should not have been rewritten with different content
        recs_after = json.loads(pending_path.read_text())
        assert recs_after == original_recs


# ---------------------------------------------------------------------------
# Test 2: Claude returns None → exit 1, pending rec kept
# ---------------------------------------------------------------------------

class TestClaudeReturnsNone:
    def test_exits_with_code_1_when_interpret_returns_none(self, tmp_path):
        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_recs = _make_pending_recs()
        pending_path.write_text(json.dumps(pending_recs))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=None),
            patch("handle_reply.WebClient"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                hr.main()

        assert exc_info.value.code == 1

    def test_pending_rec_kept_when_claude_returns_none(self, tmp_path):
        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_recs = _make_pending_recs()
        pending_path.write_text(json.dumps(pending_recs))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=None),
            patch("handle_reply.WebClient"),
        ):
            with pytest.raises(SystemExit):
                hr.main()

        # Pending rec should still be there
        recs_after = json.loads(pending_path.read_text())
        assert _ENV["THREAD_TS"] in recs_after


# ---------------------------------------------------------------------------
# Test 2b: DM channel open failure → exit 1
# ---------------------------------------------------------------------------

class TestDmChannelOpenFailure:
    def test_exits_1_when_dm_channel_open_fails(self, tmp_path):
        """If _open_dm_channel returns None, main() must exit 1 for any action."""
        action = ReplyAction(action="confirm", updates={"Stage": "Demo"})

        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_path.write_text(json.dumps(_make_pending_recs()))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        mock_slack = MagicMock()
        # conversations_open raises SlackApiError to simulate failure
        from slack_sdk.errors import SlackApiError
        mock_slack.conversations_open.side_effect = SlackApiError("fail", {"error": "fail"})

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=action),
            patch("handle_reply.WebClient", return_value=mock_slack),
            patch("handle_reply.sheets_writer.write_updates"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                hr.main()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Test 3: reject → rec removed, confirmation DM sent
# ---------------------------------------------------------------------------

class TestRejectAction:
    def test_rec_removed_on_reject(self, tmp_path):
        action = ReplyAction(action="reject", updates={})
        pending_after, _ = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert _ENV["THREAD_TS"] not in pending_after

    def test_confirmation_dm_sent_on_reject(self, tmp_path):
        action = ReplyAction(action="reject", updates={})
        _, dm_calls = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert len(dm_calls) == 1
        dm_text = dm_calls[0].kwargs.get("text", dm_calls[0][1].get("text", ""))
        assert "Jane Smith" in dm_text
        assert "skipped" in dm_text

    def test_reject_dm_contains_lead_name(self, tmp_path):
        action = ReplyAction(action="reject", updates={})
        rec = dict(_SAMPLE_REC, lead_name="Bob Jones")
        _, dm_calls = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(rec=rec),
            interpret_reply_return=action,
        )
        dm_text = dm_calls[0].kwargs.get("text", "")
        assert "Bob Jones" in dm_text


# ---------------------------------------------------------------------------
# Test 4: confirm → write_updates called with correct args, rec removed, DM sent
# ---------------------------------------------------------------------------

class TestConfirmAction:
    def test_write_updates_called_on_confirm(self, tmp_path):
        updates = {"Stage": "Demo", "Last Contact Date": "2026-05-23"}
        action = ReplyAction(action="confirm", updates=updates)

        write_mock = MagicMock()

        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_path.write_text(json.dumps(_make_pending_recs()))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        mock_slack = MagicMock()
        mock_slack.conversations_open.return_value = {"channel": {"id": "D123"}}

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=action),
            patch("handle_reply.WebClient", return_value=mock_slack),
            patch("handle_reply.sheets_writer.write_updates", wraps=write_mock),
        ):
            hr.main()

        write_mock.assert_called_once()
        call_kwargs = write_mock.call_args.kwargs
        assert call_kwargs["row_index"] == 5
        assert call_kwargs["updates"] == updates
        assert call_kwargs["spreadsheet_id"] == "fake-sheet-id"
        assert call_kwargs["tab_name"] == "LEAD TRACKER"

    def test_rec_removed_on_confirm(self, tmp_path):
        action = ReplyAction(action="confirm", updates={"Stage": "Demo"})
        pending_after, _ = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert _ENV["THREAD_TS"] not in pending_after

    def test_confirmation_dm_sent_on_confirm(self, tmp_path):
        action = ReplyAction(action="confirm", updates={"Stage": "Demo", "Last Contact Date": "2026-05-23"})
        _, dm_calls = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert len(dm_calls) == 1
        dm_text = dm_calls[0].kwargs.get("text", "")
        assert "Jane Smith" in dm_text
        assert "Acme Gym" in dm_text
        assert "Stage" in dm_text

    def test_confirm_dm_contains_fields_written(self, tmp_path):
        updates = {"Stage": "Demo", "Follow-Up Priority": "High"}
        action = ReplyAction(action="confirm", updates=updates)
        _, dm_calls = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        dm_text = dm_calls[0].kwargs.get("text", "")
        assert "Stage" in dm_text
        assert "Follow-Up Priority" in dm_text

    def test_unexpected_runtime_error_propagates(self, tmp_path):
        """Unexpected RuntimeError from write_updates propagates — rec stays in pending."""
        import handle_reply as hr

        action = ReplyAction(action="confirm", updates={"Stage": "Demo"})

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_recs = _make_pending_recs()
        pending_path.write_text(json.dumps(pending_recs))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})

        mock_slack = MagicMock()
        mock_slack.conversations_open.return_value = {"channel": {"id": "D123"}}

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=action),
            patch("handle_reply.WebClient", return_value=mock_slack),
            patch(
                "handle_reply.sheets_writer.write_updates",
                side_effect=RuntimeError("unexpected DB error"),
            ),
        ):
            with pytest.raises(RuntimeError, match="unexpected DB error"):
                hr.main()

        # Rec must NOT have been removed — the write failed
        recs_after = json.loads(pending_path.read_text())
        assert _ENV["THREAD_TS"] in recs_after


# ---------------------------------------------------------------------------
# Test 5: edit → same flow as confirm but with edited updates
# ---------------------------------------------------------------------------

class TestEditAction:
    def test_rec_removed_on_edit(self, tmp_path):
        action = ReplyAction(action="edit", updates={"Stage": "Trial", "Last Contact Date": "2026-05-23"})
        pending_after, _ = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert _ENV["THREAD_TS"] not in pending_after

    def test_write_updates_called_with_edited_updates(self, tmp_path):
        edited_updates = {"Stage": "Trial", "Follow-Up Priority": "Low"}
        action = ReplyAction(action="edit", updates=edited_updates)

        write_mock = MagicMock()

        import handle_reply as hr

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pending_path = data_dir / "pending_recs.json"
        pending_path.write_text(json.dumps(_make_pending_recs()))

        cfg = dict(_CONFIG, **{"pending_recs_file": str(pending_path)})
        mock_slack = MagicMock()
        mock_slack.conversations_open.return_value = {"channel": {"id": "D123"}}

        with (
            patch.dict(os.environ, _ENV, clear=False),
            patch("handle_reply.load_dotenv"),
            patch("builtins.open", wraps=_patched_open(cfg, data_dir)),
            patch("handle_reply.interpret_reply", return_value=action),
            patch("handle_reply.WebClient", return_value=mock_slack),
            patch("handle_reply.sheets_writer.write_updates", wraps=write_mock),
        ):
            hr.main()

        write_mock.assert_called_once()
        call_kwargs = write_mock.call_args.kwargs
        assert call_kwargs["updates"] == edited_updates

    def test_confirmation_dm_sent_on_edit(self, tmp_path):
        action = ReplyAction(action="edit", updates={"Stage": "Trial"})
        _, dm_calls = _run_handle_reply(
            tmp_path,
            pending_recs=_make_pending_recs(),
            interpret_reply_return=action,
        )
        assert len(dm_calls) == 1
        dm_text = dm_calls[0].kwargs.get("text", "")
        assert "Jane Smith" in dm_text
