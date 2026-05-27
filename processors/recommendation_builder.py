"""Recommendation builder — calls Claude to propose CRM updates from an Avoma transcript.

Given an AvomaTranscript and an optional current sheet row, calls Claude using
tool use to get a structured recommendation. Returns a Recommendation dataclass
or None if Claude returns malformed output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Optional

import anthropic

if TYPE_CHECKING:
    from collectors.avoma import AvomaTranscript

_log = logging.getLogger(__name__)

# Formula columns that must never appear in proposed_updates
_FORMULA_COLUMNS = {"Days Until Follow-Up", "Days Since Last Contact"}

_VALID_CONFIDENCE = {"high", "medium", "low", "new_lead"}

_RECOMMEND_TOOL = {
    "name": "propose_crm_updates",
    "description": "Propose CRM updates for a lead tracker based on a sales call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "match_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "new_lead"],
            },
            "lead_name": {"type": "string"},
            "organization": {"type": "string"},
            "proposed_updates": {
                "type": "object",
                "description": (
                    "Column name → value. Valid column names: "
                    "'Stage', 'Follow-Up Priority', 'Next Action Date', 'Last Contact Date', "
                    "'Initial Contact Date', 'Branch / Sector', 'Email'. "
                    "Only include columns where the call gives clear signal."
                ),
            },
            "note": {
                "type": "string",
                "description": "One sentence, past tense, outcome-focused.",
            },
            "reasoning": {"type": "string"},
        },
        "required": [
            "match_confidence",
            "lead_name",
            "organization",
            "proposed_updates",
            "note",
            "reasoning",
        ],
    },
}

_SYSTEM_PROMPT = (
    "You are analyzing a sales call transcript to propose CRM updates for a lead tracker "
    "for TeamBuildr Strength — workout and training program software for athletes, coaches, "
    "and strength & conditioning facilities."
)


def _build_user_prompt(transcript: "AvomaTranscript", current_row: Optional[dict]) -> str:
    if current_row is not None:
        current_row_text = json.dumps(current_row, indent=2)
    else:
        current_row_text = "No existing row — this may be a new lead."

    buying_signals = transcript.buying_signals or []
    objections = transcript.objections or []
    action_items = transcript.action_items or []
    participants = transcript.participants or []

    buying_signals_str = ", ".join(buying_signals) if buying_signals else "None"
    objections_str = ", ".join(objections) if objections else "None"
    action_items_str = ", ".join(action_items) if action_items else "None"
    participants_str = ", ".join(participants) if participants else "None"

    today = date.today().isoformat()

    return f"""\
Current row for this lead (if found):
{current_row_text}

Call analysis:
- Call type: {transcript.call_type}
- Summary: {transcript.summary}
- Buying signals: {buying_signals_str}
- Objections: {objections_str}
- Action items: {action_items_str}
- Participants: {participants_str}

Propose updates ONLY to fields where the call gives clear signal.
For Stage: use exactly one of [Initial Contact, Demo, Trial, Quoted]
For Follow-Up Priority: use exactly one of [High, Medium, Low]
Do not propose updates to: Days Until Follow-Up, Days Since Last Contact, Initial Contact Date (unless new_lead with no existing row).
Do not propose updates to Email or Branch / Sector if the existing row already has values.
Always set Last Contact Date = {today} (YYYY-MM-DD format).
"""


@dataclass
class Recommendation:
    match_confidence: str        # "high" | "medium" | "low" | "new_lead"
    lead_name: str               # matched lead name from sheet (or Avoma participant name if new)
    organization: str            # from sheet or inferred from Avoma
    sheet_row: Optional[int]     # sheet row index, or None if new_lead
    proposed_updates: dict       # column_name -> value
    note: str                    # one-sentence past-tense note for Notes column
    reasoning: str               # one sentence explaining stage/priority calls


def build_recommendation(
    transcript: "AvomaTranscript",
    current_row: Optional[dict],
    model: str,
    anthropic_api_key: str,
) -> Optional[Recommendation]:
    """Return a Recommendation, or None if Claude returns malformed output."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=_SYSTEM_PROMPT,
            tools=[_RECOMMEND_TOOL],
            tool_choice={"type": "tool", "name": "propose_crm_updates"},
            messages=[
                {
                    "role": "user",
                    "content": _build_user_prompt(transcript, current_row),
                }
            ],
        )
    except Exception:
        _log.exception("Claude API call failed for transcript %s", transcript.uuid)
        return None

    # Extract the tool_use block
    tool_input: Optional[dict] = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "propose_crm_updates":
            tool_input = block.input
            break

    if tool_input is None:
        _log.warning("Claude returned no tool_use block for transcript %s", transcript.uuid)
        return None

    # Validate required fields
    required = {"match_confidence", "lead_name", "organization", "proposed_updates", "note", "reasoning"}
    if not required.issubset(tool_input.keys()):
        missing = required - tool_input.keys()
        _log.warning("Claude response missing fields %s for transcript %s", missing, transcript.uuid)
        return None

    # Validate match_confidence enum
    if tool_input["match_confidence"] not in _VALID_CONFIDENCE:
        _log.warning(
            "Unexpected match_confidence value '%s' for transcript %s",
            tool_input["match_confidence"],
            transcript.uuid,
        )
        return None

    # Validate proposed_updates is a dict
    proposed_updates_raw = tool_input.get("proposed_updates")
    if not isinstance(proposed_updates_raw, dict):
        _log.warning("proposed_updates is not a dict for transcript %s", transcript.uuid)
        return None

    # Drop formula columns silently
    proposed_updates: dict = {
        k: v
        for k, v in proposed_updates_raw.items()
        if k not in _FORMULA_COLUMNS
    }

    sheet_row: Optional[int] = current_row.get("row_index") if current_row else None

    return Recommendation(
        match_confidence=tool_input["match_confidence"],
        lead_name=tool_input["lead_name"],
        organization=tool_input["organization"],
        sheet_row=sheet_row,
        proposed_updates=proposed_updates,
        note=tool_input["note"],
        reasoning=tool_input["reasoning"],
    )
