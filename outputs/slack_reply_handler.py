"""Slack reply handler — interprets Melissa's threaded Slack replies via Claude.

Given a reply text and the original recommendation dict, calls Claude with
tool use to determine the intent (confirm / edit / reject) and returns a
structured ReplyAction dataclass, or None on failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import anthropic

_log = logging.getLogger(__name__)

# Formula columns that must never appear in updates
_FORMULA_COLUMNS = {"Days Until Follow-Up", "Days Since Last Contact"}

_VALID_ACTIONS = {"confirm", "edit", "reject"}

_INTERPRET_TOOL = {
    "name": "interpret_reply",
    "description": "Interpret the user's reply and determine what CRM updates to write.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["confirm", "edit", "reject"],
            },
            "updates": {
                "type": "object",
                "description": "Final column values to write. Empty if reject.",
            },
        },
        "required": ["action", "updates"],
    },
}


@dataclass
class ReplyAction:
    action: str   # "confirm" | "edit" | "reject"
    updates: dict  # final values to write; {} if reject


def interpret_reply(
    reply_text: str,
    recommendation: dict,  # the full rec dict from pending_recs.json
    model: str,
    anthropic_api_key: str,
) -> Optional[ReplyAction]:
    """Return a ReplyAction, or None on Claude failure."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    recommendation_json = json.dumps(recommendation, indent=2)

    user_message = f"""\
A sales rep received this CRM update recommendation and replied with a free-text response.

Original recommendation:
{recommendation_json}

User's reply: "{reply_text}"

Determine their intent and use the interpret_reply tool.

If action is "confirm": updates = original proposed_updates unchanged
If action is "edit": updates = original proposed_updates with user's corrections applied
If action is "reject": updates = {{}}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system="You are interpreting a user's reply to a CRM update recommendation.",
            tools=[_INTERPRET_TOOL],
            tool_choice={"type": "tool", "name": "interpret_reply"},
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )
    except Exception:
        _log.exception("Claude API call failed in interpret_reply")
        return None

    # Extract the tool_use block
    tool_input: Optional[dict] = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "interpret_reply":
            tool_input = block.input
            break

    if tool_input is None:
        _log.warning("Claude returned no interpret_reply tool_use block")
        return None

    # Validate action field
    action = tool_input.get("action")
    if action not in _VALID_ACTIONS:
        _log.warning("Claude returned invalid action: %r", action)
        return None

    # Validate updates field is a dict
    updates_raw = tool_input.get("updates")
    if not isinstance(updates_raw, dict):
        _log.warning("Claude returned non-dict updates: %r", updates_raw)
        return None

    # Canonical overrides
    if action == "confirm":
        # Always use original proposed_updates, ignore what Claude returned
        updates = dict(recommendation.get("proposed_updates", {}))
    elif action == "reject":
        updates = {}
    else:
        # edit — use Claude's merged updates
        updates = dict(updates_raw)

    # Drop formula columns silently
    updates = {k: v for k, v in updates.items() if k not in _FORMULA_COLUMNS}

    return ReplyAction(action=action, updates=updates)
