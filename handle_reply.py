#!/usr/bin/env python3
"""Reply handler entry point — called by reply.yml GitHub Actions workflow."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from collectors import sheets_writer
from outputs.slack_reply_handler import interpret_reply

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

_REQUIRED_ENV = [
    "ANTHROPIC_API_KEY",
    "SLACK_BOT_TOKEN",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SHEET_ID",
    "THREAD_TS",
    "REPLY_TEXT",
]


# ---------------------------------------------------------------------------
# State file helpers (same pattern as main.py)
# ---------------------------------------------------------------------------

def _load_json_file(path: str, default):
    path_obj = Path(path)
    if not path_obj.exists():
        return default
    try:
        with open(path_obj) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Could not read state file %s (%s) — starting fresh", path, exc)
        return default


def _save_json_file(path: str, data) -> None:
    path = Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Slack helper
# ---------------------------------------------------------------------------

def _send_dm_to_channel(
    slack_bot_token: str,
    channel_id: str,
    thread_ts: str,
    text: str,
) -> None:
    """Post a threaded reply in the given channel."""
    client = WebClient(token=slack_bot_token)
    try:
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=text,
        )
    except SlackApiError as exc:
        _log.error("Failed to send Slack confirmation DM: %s", exc)


def _cleanup_kv(thread_ts: str) -> None:
    """Delete the thread_ts key from Cloudflare KV after a reply is handled.

    No-ops silently if Cloudflare env vars are not set.
    """
    cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    cf_namespace_id = os.getenv("CLOUDFLARE_KV_NAMESPACE_ID")
    cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if cf_account_id and cf_namespace_id and cf_api_token:
        from collectors.cloudflare_kv import delete_thread_ts_from_kv
        delete_thread_ts_from_kv(cf_account_id, cf_namespace_id, cf_api_token, thread_ts)


def _open_dm_channel(slack_bot_token: str, user_id: str) -> str | None:
    """Open a DM channel with a user and return the channel ID."""
    client = WebClient(token=slack_bot_token)
    try:
        resp = client.conversations_open(users=[user_id])
        return resp["channel"]["id"]
    except SlackApiError as exc:
        _log.error("Failed to open DM channel: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    # --- Load env vars ---
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        _log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
    service_account_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    thread_ts = os.environ["THREAD_TS"]
    reply_text = os.environ["REPLY_TEXT"]

    # --- Load config ---
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _log.error("Failed to load config.json: %s", e)
        sys.exit(1)

    model: str = config.get("ai_model", "claude-sonnet-4-6")
    sheets_cfg = config.get("google_sheets", {})
    sheet_tab: str = sheets_cfg.get("lead_tracker_tab", "LEAD TRACKER")
    melissa_user_id: str = config.get("slack", {}).get("melissa_user_id", "")
    pending_recs_path = str(base_dir / config.get("pending_recs_file", "data/pending_recs.json"))

    # --- Load pending_recs ---
    pending_recs: dict = _load_json_file(pending_recs_path, {})

    # --- Look up thread_ts ---
    if thread_ts not in pending_recs:
        _log.info(
            "thread_ts=%s not in pending recs, may have been already processed",
            thread_ts,
        )
        sys.exit(0)

    rec = pending_recs[thread_ts]

    # --- Interpret the reply with Claude ---
    action_result = interpret_reply(
        reply_text=reply_text,
        recommendation=rec,
        model=model,
        anthropic_api_key=anthropic_api_key,
    )

    if action_result is None:
        _log.error(
            "Claude failed to interpret reply for thread_ts=%s — leaving in pending for retry",
            thread_ts,
        )
        sys.exit(1)

    lead_name = rec.get("lead_name", "Unknown")
    organization = rec.get("organization", "Unknown")

    # --- Open DM channel — required for all actions ---
    dm_channel_id: str | None = _open_dm_channel(slack_bot_token, melissa_user_id)
    if dm_channel_id is None:
        _log.error("Failed to open DM channel to Melissa — cannot send confirmation")
        sys.exit(1)

    if action_result.action == "reject":
        _log.info(
            "Rejected recommendation for thread_ts=%s lead=%r",
            thread_ts,
            lead_name,
        )
        del pending_recs[thread_ts]
        _cleanup_kv(thread_ts)
        _save_json_file(pending_recs_path, pending_recs)

        _send_dm_to_channel(
            slack_bot_token=slack_bot_token,
            channel_id=dm_channel_id,
            thread_ts=thread_ts,
            text=f"Got it — skipped *{lead_name}*.",
        )

    else:
        # confirm or edit
        _log.info(
            "Writing updates for thread_ts=%s lead=%r action=%s fields=%s",
            thread_ts,
            lead_name,
            action_result.action,
            list(action_result.updates.keys()),
        )

        # Extract note separately so updates dict stays clean
        note_to_write = action_result.updates.pop("Notes", rec.get("note"))

        sheets_writer.write_updates(
            service_account_json=service_account_json,
            spreadsheet_id=sheet_id,
            tab_name=sheet_tab,
            row_index=rec.get("sheet_row"),
            updates=action_result.updates,
            note=note_to_write,
        )

        del pending_recs[thread_ts]
        _cleanup_kv(thread_ts)
        _save_json_file(pending_recs_path, pending_recs)

        fields_written = ", ".join(action_result.updates.keys()) if action_result.updates else "(none)"
        confirmation_text = (
            f"✅ Updated: *{lead_name}* ({organization})\n"
            f"Fields written: {fields_written}"
        )

        _send_dm_to_channel(
            slack_bot_token=slack_bot_token,
            channel_id=dm_channel_id,
            thread_ts=thread_ts,
            text=confirmation_text,
        )

    _log.info("handle_reply complete for thread_ts=%s action=%s", thread_ts, action_result.action)


if __name__ == "__main__":
    main()
