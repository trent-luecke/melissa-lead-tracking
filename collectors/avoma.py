"""Avoma meeting transcript collector — raw transcript + Claude analysis.

Fetches external sales calls involving specific reps, then uses Claude to
extract structured call data: call type, gaps, objections, buying signals,
competitors, and action items.

48h lookback window is the default. UUID-based deduplication and call_type
filtering (demo/follow_up only) are handled by the caller (main.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import anthropic
import requests

_BASE_URL = "https://api.avoma.com"
_TIMEOUT = 15
_TRANSCRIPT_CHAR_LIMIT = 30_000  # ~7k tokens — fits well within any Claude context window

_EXTRACT_TOOL: dict = {
    "name": "extract_call_analysis",
    "description": "Extract structured analysis from a sales call transcript.",
    "input_schema": {
        "type": "object",
        "properties": {
            "call_type": {
                "type": "string",
                "enum": ["demo", "onboarding", "follow_up", "other"],
            },
            "summary": {"type": "string", "description": "2-3 sentence outcome summary."},
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product limitations or missing features raised by the prospect.",
            },
            "objections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Price, timing, competition, missing decision-maker, etc.",
            },
            "buying_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Positive intent signals — asking about contracts, pricing, next steps.",
            },
            "competitors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any competing product or vendor names mentioned.",
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete next steps, with owner where identifiable.",
            },
        },
        "required": [
            "call_type", "summary",
            "gaps", "objections",
            "buying_signals", "competitors",
            "action_items",
        ],
    },
}

_SYSTEM_PROMPT = """\
You are analyzing sales call transcripts for TeamBuildr, a B2B SaaS company.

TeamBuildr's primary product is **Strength**: workout and training program software for athletes, coaches, and strength & conditioning facilities.

Your job: analyze the transcript and call the extract_call_analysis tool with structured data.

Key rules:
- For gaps: product limitations or missing features the prospect raised as concerns.
- For objections: price, timing, competition, decision-maker not present, etc.
- For buying_signals: asking about contracts, implementation timelines, pricing details, references.
- For action_items: concrete next steps with owner where identifiable (e.g. "Rep to send pricing doc").
- For competitors: any competing product or vendor name mentioned.
- For summary: 2-3 sentences on call outcome and current status.\
"""


@dataclass
class AvomaTranscript:
    uuid: str
    title: str
    start_at: str  # ISO 8601 UTC
    participants: list[str] = field(default_factory=list)
    call_type: str = ""  # demo | onboarding | follow_up | other
    summary: str = ""
    gaps: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    buying_signals: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


def _get(api_key: str, path: str, params: dict | None = None) -> dict:
    resp = requests.get(
        f"{_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params or {},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_transcript(api_key: str, meeting_uuid: str) -> tuple[list[dict], list[dict]]:
    """Return (speakers, utterances) for a meeting, or ([], []) on failure."""
    try:
        body = _get(api_key, "/v1/transcriptions", {"meeting_uuid": meeting_uuid})
        # API returns a bare list, not {"results": [...]}
        results = body if isinstance(body, list) else body.get("results", [])
        if not results:
            return [], []
        data = results[0]
        return data.get("speakers", []), data.get("transcript", [])
    except Exception:
        return [], []


def _format_transcript(speakers: list[dict], utterances: list[dict]) -> str:
    """Format raw Avoma transcript data into labeled dialog text."""
    speaker_map: dict[str, str] = {}
    for s in speakers:
        sid = str(s.get("id") or s.get("speaker_id", ""))
        name = s.get("name") or s.get("email", "Unknown")
        prefix = "Rep" if s.get("is_rep") else "Prospect"
        speaker_map[sid] = f"[{prefix} - {name}]"

    lines: list[str] = []
    for utt in utterances:
        sid = str(utt.get("speaker_id", ""))
        label = speaker_map.get(sid, "[Unknown]")
        text = (utt.get("transcript") or "").strip()
        if text:
            lines.append(f"{label}: {text}")

    return "\n".join(lines)[:_TRANSCRIPT_CHAR_LIMIT]


def _analyze_with_claude(
    anthropic_api_key: str,
    model: str,
    title: str,
    formatted_transcript: str,
) -> dict | None:
    """Run Claude extraction. Returns the tool input dict, or None on failure."""
    if not formatted_transcript.strip():
        return None
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_call_analysis"},
            messages=[{
                "role": "user",
                "content": f"Meeting title: {title}\n\nTranscript:\n{formatted_transcript}",
            }],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_call_analysis":
                return block.input
        return None
    except Exception:
        return None


def fetch_recent_meetings(
    api_key: str,
    anthropic_api_key: str,
    model: str,
    lookback_hours: int = 48,
    sales_rep_emails: list[str] | None = None,
    filter_internal: bool = True,
) -> list[AvomaTranscript]:
    """Return analyzed Avoma meetings from the past ``lookback_hours`` hours.

    Filters to calls involving ``sales_rep_emails`` (if provided) and
    external meetings only (if ``filter_internal=True``). Returns all external
    meetings with a transcript. Callers are responsible for UUID-based
    deduplication and call_type filtering.
    """
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(hours=lookback_hours)

    rep_emails_lower = {e.lower() for e in (sales_rep_emails or [])}

    params: dict | None = {
        "from_date": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_size": 100,
        "o": "-start_at",
    }
    if filter_internal:
        params["is_internal"] = "false"

    headers = {"Authorization": f"Bearer {api_key}"}
    url: str | None = f"{_BASE_URL}/v1/meetings"
    transcripts: list[AvomaTranscript] = []

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()

        for m in body.get("results", []):
            if not m.get("transcript_ready"):
                continue

            uuid = m.get("uuid", "")
            if not uuid:
                continue

            attendees = m.get("attendees", [])

            # Must include at least one configured sales rep
            if rep_emails_lower:
                attendee_emails = {(a.get("email") or "").lower() for a in attendees}
                if not rep_emails_lower.intersection(attendee_emails):
                    continue

            participants = [
                a.get("name") or a.get("email", "")
                for a in attendees
                if a.get("name") or a.get("email")
            ]

            speakers, utterances = _fetch_transcript(api_key, uuid)
            if not utterances:
                continue

            formatted = _format_transcript(speakers, utterances)
            title = m.get("subject") or "Untitled Meeting"

            result = _analyze_with_claude(anthropic_api_key, model, title, formatted)
            if not result:
                continue

            transcripts.append(AvomaTranscript(
                uuid=uuid,
                title=title,
                start_at=m.get("start_at", ""),
                participants=participants,
                call_type=result.get("call_type", "other"),
                summary=result.get("summary", ""),
                gaps=result.get("gaps", []),
                objections=result.get("objections", []),
                buying_signals=result.get("buying_signals", []),
                competitors=result.get("competitors", []),
                action_items=result.get("action_items", []),
            ))

        # params are already encoded in the next URL
        url = body.get("next")
        params = None

    return transcripts
