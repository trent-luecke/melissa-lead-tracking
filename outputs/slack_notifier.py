"""Slack notifier — sends pending recommendations to Melissa as DMs.

For each entry in pending_recs where thread_ts == "", sends a formatted DM to
Melissa via Slack's chat.postMessage API, captures the resulting thread_ts, and
re-keys the entry from avoma_uuid → thread_ts.
"""

from __future__ import annotations

import logging
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

_log = logging.getLogger(__name__)

_FIELD_LABELS = {
    "Stage": "Stage",
    "Follow-Up Priority": "Follow-Up Priority",
    "Next Action Date": "Next Action Date",
    "Last Contact Date": "Last Contact Date",
    "Initial Contact Date": "Initial Contact Date",
    "Branch / Sector": "Branch / Sector",
    "Email": "Email",
}

_CALL_TYPE_TITLES = {
    "demo": "Demo",
    "follow_up": "Follow-Up",
}

_field_to_row_key = {
    "Stage": "stage",
    "Follow-Up Priority": "follow_up_priority",
    "Next Action Date": "next_action_date",
    "Last Contact Date": "last_contact_date",
    "Initial Contact Date": "initial_contact_date",
    "Branch / Sector": "branch_sector",
    "Email": "email",
}


def _format_call_type(call_type: str) -> str:
    return _CALL_TYPE_TITLES.get(call_type, call_type.replace("_", " ").title())


def _format_message(rec: dict, sheet_lookup: dict) -> str:
    """Build the Slack message text for a single recommendation."""
    lead_name = rec.get("lead_name", "Unknown")
    organization = rec.get("organization", "Unknown")
    call_type_title = _format_call_type(rec.get("call_type", ""))
    call_date = rec.get("call_date", "")
    reasoning = rec.get("reasoning", "")
    proposed_updates = rec.get("proposed_updates", {})
    note = rec.get("note", "")
    ambiguous = rec.get("ambiguous_match", False)

    # Look up current row values from sheet_lookup using lead_name or org
    current_row: Optional[dict] = None
    normalized_lead = lead_name.lower().strip()
    normalized_org = organization.lower().strip()
    if normalized_lead in sheet_lookup:
        current_row = sheet_lookup[normalized_lead]
    elif normalized_org in sheet_lookup:
        current_row = sheet_lookup[normalized_org]

    # Also accept current_row if it was embedded in the rec (for inline use)
    # (not used here — sheet_lookup is the source of truth)

    lines: list[str] = []

    # Header
    lines.append(f"📋 *Lead Update — {lead_name} ({organization})*")
    lines.append("")

    # Ambiguous match warning (immediately after header blank line)
    if ambiguous:
        lines.append("⚠️ _Multiple sheet rows matched — Melissa may need to clarify which lead this is._")
        lines.append("")

    # Call info
    lines.append(f"Call: {call_type_title} · {call_date}")
    lines.append(f"Summary: {reasoning}")
    lines.append("")

    # Proposed updates
    lines.append("Proposed updates:")

    # Field bullets with current → new
    for field_key, label in _FIELD_LABELS.items():
        if field_key not in proposed_updates:
            continue
        new_value = proposed_updates[field_key]

        # Determine current value from sheet row
        current_value: Optional[str] = None
        if current_row is not None:
            row_key = _field_to_row_key.get(field_key)
            if row_key:
                val = current_row.get(row_key, "")
                if val:
                    current_value = val

        if current_value:
            lines.append(f"• {label}: {current_value} → *{new_value}*")
        else:
            lines.append(f"• {label}: → *{new_value}*")

    # Note line
    if note:
        lines.append(f"• Note added: `{note}`")

    lines.append("")
    lines.append("Reply in this thread to confirm, edit, or reject.")

    return "\n".join(lines)


def send_pending_recommendations(
    slack_bot_token: str,
    melissa_user_id: str,
    pending_recs: dict,
    sheet_lookup: dict,
) -> dict:
    """Send DMs for all unsent recs. Returns updated pending_recs dict re-keyed by thread_ts.

    Args:
        slack_bot_token: Slack bot OAuth token.
        melissa_user_id: Slack user ID for Melissa.
        pending_recs: {avoma_uuid: rec_dict} — entries with thread_ts == "" will be sent.
        sheet_lookup: Result of load_lead_tracker, used to look up current field values.

    Returns:
        Updated pending_recs dict where sent entries are re-keyed from avoma_uuid to thread_ts.
        Entries that were already sent (thread_ts != "") are passed through unchanged.
        Entries that fail to send are left under their avoma_uuid key with thread_ts still "".
    """
    client = WebClient(token=slack_bot_token)

    # Open DM channel to Melissa once
    try:
        open_resp = client.conversations_open(users=[melissa_user_id])
        dm_channel_id: str = open_resp["channel"]["id"]
    except SlackApiError as exc:
        _log.error("Failed to open DM channel to Melissa (%s): %s", melissa_user_id, exc)
        return pending_recs

    updated_recs: dict = {}

    for key, rec in list(pending_recs.items()):
        thread_ts = rec.get("thread_ts", "")

        # Already sent — pass through unchanged under original key
        if thread_ts != "":
            updated_recs[key] = rec
            continue

        # Build and send the message
        message_text = _format_message(rec, sheet_lookup)

        try:
            response = client.chat_postMessage(
                channel=dm_channel_id,
                text=message_text,
            )
            new_thread_ts: str = response["ts"]
        except SlackApiError as exc:
            _log.error(
                "Failed to send Slack DM for avoma_uuid=%s: %s",
                rec.get("avoma_uuid", key),
                exc,
            )
            # Keep the entry under its original key so it can be retried next run
            updated_recs[key] = rec
            continue

        # Re-key from avoma_uuid → thread_ts
        new_rec = dict(rec)
        new_rec["thread_ts"] = new_thread_ts
        updated_recs[new_thread_ts] = new_rec
        _log.info(
            "Sent Slack DM for avoma_uuid=%s → thread_ts=%s",
            rec.get("avoma_uuid", key),
            new_thread_ts,
        )

    return updated_recs
