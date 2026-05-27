"""Nightly orchestrator — Avoma → name matching → recommendation builder → pending_recs.json.

Reads config.json and environment variables, fetches Avoma transcripts, matches
participants against the Google Sheets lead tracker, builds recommendations via
Claude, and persists results to data/pending_recs.json keyed by avoma_uuid.

Slack delivery is handled separately in Task 5 (slack_notifier.py).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from collectors.avoma import AvomaTranscript, fetch_recent_meetings
from collectors.google_sheets import load_lead_tracker, normalize_name
from outputs.slack_notifier import send_pending_recommendations
from processors.recommendation_builder import build_recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

_REQUIRED_ENV = [
    "AVOMA_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SHEET_ID",
]

_VALID_CALL_TYPES = {"demo", "follow_up"}


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

def load_json_file(path: str, default):
    path_obj = Path(path)
    if not path_obj.exists():
        return default
    try:
        with open(path_obj) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Could not read state file %s (%s) — starting fresh", path, exc)
        return default


def save_json_file(path: str, data) -> None:
    path = Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

def load_env() -> dict[str, str]:
    """Load required environment variables. Raises RuntimeError on missing vars."""
    load_dotenv()
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    return {k: os.environ[k] for k in _REQUIRED_ENV}


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

def expire_stale_recs(
    pending_recs: dict,
    ttl_days: int,
    now: datetime | None = None,
) -> dict:
    """Remove entries whose created_at is older than ttl_days. Returns cleaned dict."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ttl_days)
    cleaned: dict = {}
    for key, rec in pending_recs.items():
        created_at_str = rec.get("created_at", "")
        try:
            # Parse ISO 8601 with trailing Z
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Unparseable timestamp — keep it to avoid silent data loss
            cleaned[key] = rec
            continue
        if created_at < cutoff:
            _log.info(
                "Expiring stale pending rec for uuid=%s (created_at=%s, ttl=%d days)",
                key,
                created_at_str,
                ttl_days,
            )
        else:
            cleaned[key] = rec
    return cleaned


# ---------------------------------------------------------------------------
# Sheet matching
# ---------------------------------------------------------------------------

def match_participant_to_sheet(
    participants: list[str],
    sheet: dict[str, dict],
) -> tuple[dict | None, bool]:
    """Return (current_row, ambiguous).

    - Exactly one match → (row, False)
    - Zero matches → (None, False)
    - Multiple matches → (None, True)
    """
    matched_rows: list[dict] = []
    seen_row_indices: set[int] = set()

    for name in participants:
        normalized = normalize_name(name)
        if normalized in sheet:
            row = sheet[normalized]
            row_idx = row.get("row_index")
            if row_idx not in seen_row_indices:
                seen_row_indices.add(row_idx)
                matched_rows.append(row)

    if len(matched_rows) == 0:
        return None, False
    if len(matched_rows) == 1:
        return matched_rows[0], False
    # Multiple distinct rows matched
    return None, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve paths relative to the directory containing this file
    base_dir = Path(__file__).parent

    # --- Load config ---
    config_path = base_dir / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _log.error("Failed to load config.json: %s", e)
        sys.exit(1)

    # --- Load env ---
    try:
        env = load_env()
    except RuntimeError as e:
        _log.error("%s", e)
        sys.exit(1)

    avoma_api_key = env["AVOMA_API_KEY"]
    anthropic_api_key = env["ANTHROPIC_API_KEY"]
    google_service_account_json = env["GOOGLE_SERVICE_ACCOUNT_JSON"]
    google_sheet_id = env["GOOGLE_SHEET_ID"]

    model: str = config.get("ai_model", "claude-sonnet-4-6")
    avoma_cfg = config.get("avoma", {})
    sheets_cfg = config.get("google_sheets", {})
    ttl_days: int = config.get("pending_rec_ttl_days", 7)

    pending_recs_path = str(base_dir / config.get("pending_recs_file", "data/pending_recs.json"))
    processed_uuids_path = str(base_dir / config.get("processed_uuids_file", "data/processed_uuids.json"))

    # --- Load state ---
    processed_uuids: set[str] = set(load_json_file(processed_uuids_path, []))
    pending_recs: dict = load_json_file(pending_recs_path, {})

    # --- Expire stale pending recs ---
    pending_recs = expire_stale_recs(pending_recs, ttl_days)

    # --- Fetch Avoma transcripts ---
    _log.info("Fetching Avoma transcripts (lookback=%dh)...", avoma_cfg.get("lookback_hours", 48))
    try:
        transcripts: list[AvomaTranscript] = fetch_recent_meetings(
            api_key=avoma_api_key,
            anthropic_api_key=anthropic_api_key,
            model=model,
            lookback_hours=avoma_cfg.get("lookback_hours", 48),
            sales_rep_emails=avoma_cfg.get("sales_rep_emails"),
            filter_internal=avoma_cfg.get("filter_internal", True),
        )
    except Exception as e:
        _log.error("Failed to fetch Avoma transcripts: %s", e)
        sys.exit(1)

    _log.info("Fetched %d transcript(s) from Avoma.", len(transcripts))

    # --- Filter: call_type and already-processed UUIDs ---
    qualifying: list[AvomaTranscript] = []
    skipped_call_type = 0
    skipped_dedup = 0

    for t in transcripts:
        if t.uuid in processed_uuids:
            _log.info("Skipping already-processed UUID: %s", t.uuid)
            skipped_dedup += 1
            continue
        if t.call_type not in _VALID_CALL_TYPES:
            _log.info(
                "Skipping UUID=%s call_type=%s (not demo/follow_up).",
                t.uuid,
                t.call_type,
            )
            skipped_call_type += 1
            # Still mark as processed so we don't re-check it each run
            processed_uuids.add(t.uuid)
            continue
        qualifying.append(t)

    _log.info(
        "After filtering: %d qualifying, %d skipped (wrong call_type), %d skipped (already processed).",
        len(qualifying),
        skipped_call_type,
        skipped_dedup,
    )

    if not qualifying:
        _log.info("No qualifying transcripts to process. Saving state and exiting.")
        save_json_file(processed_uuids_path, list(processed_uuids))
        save_json_file(pending_recs_path, pending_recs)
        return

    # --- Load the sheet once ---
    _log.info("Loading Google Sheets lead tracker...")
    try:
        sheet = load_lead_tracker(
            service_account_json=google_service_account_json,
            spreadsheet_id=google_sheet_id,
            tab_name=sheets_cfg.get("lead_tracker_tab", "LEAD TRACKER"),
        )
    except RuntimeError as e:
        _log.error("Failed to load Google Sheets lead tracker: %s", e)
        sys.exit(1)

    _log.info("Loaded %d row(s) from lead tracker.", len(sheet))

    # --- Process each qualifying transcript ---
    recs_built = 0

    for transcript in qualifying:
        current_row, ambiguous = match_participant_to_sheet(transcript.participants, sheet)

        if ambiguous:
            _log.info(
                "UUID=%s has ambiguous participant match (multiple sheet rows). Using current_row=None.",
                transcript.uuid,
            )

        try:
            rec = build_recommendation(
                transcript=transcript,
                current_row=current_row,
                model=model,
                anthropic_api_key=anthropic_api_key,
            )
        except Exception as exc:
            _log.exception("Unexpected error building recommendation for %s: %s", transcript.uuid, exc)
            rec = None

        # Always mark UUID as processed — even if rec is None — to avoid retrying bad transcripts
        processed_uuids.add(transcript.uuid)

        if rec is None:
            _log.warning(
                "build_recommendation returned None for UUID=%s. Skipping.",
                transcript.uuid,
            )
            continue

        if len(transcript.start_at) < 10:
            _log.warning("start_at too short for transcript %s: %r", transcript.uuid, transcript.start_at)

        pending_recs[transcript.uuid] = {
            "thread_ts": "",  # filled in by slack_notifier (Task 5)
            "lead_name": rec.lead_name,
            "organization": rec.organization,
            "sheet_row": rec.sheet_row,
            "proposed_updates": rec.proposed_updates,
            "note": rec.note,
            "reasoning": rec.reasoning,
            "match_confidence": rec.match_confidence,
            "avoma_uuid": transcript.uuid,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ambiguous_match": ambiguous,
            "call_date": transcript.start_at[:10] if len(transcript.start_at) >= 10 else transcript.start_at,  # YYYY-MM-DD
            "call_type": transcript.call_type,
        }
        recs_built += 1
        _log.info(
            "Built recommendation for UUID=%s lead=%r org=%r match_confidence=%s",
            transcript.uuid,
            rec.lead_name,
            rec.organization,
            rec.match_confidence,
        )

    _log.info("Built %d recommendation(s).", recs_built)

    # --- Send pending recs to Melissa via Slack (optional — skip if token not set) ---
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    melissa_user_id: str = config.get("slack", {}).get("melissa_user_id", "")

    if slack_bot_token:
        _log.info("Sending pending recommendations to Melissa via Slack...")
        try:
            pending_recs = send_pending_recommendations(
                slack_bot_token=slack_bot_token,
                melissa_user_id=melissa_user_id,
                pending_recs=pending_recs,
                sheet_lookup=sheet,
            )
        except Exception as exc:
            _log.exception("Unexpected error in send_pending_recommendations: %s", exc)
    else:
        _log.info("SLACK_BOT_TOKEN not set — skipping Slack delivery.")

    # --- Sync active thread_ts keys to Cloudflare KV so the worker can validate replies ---
    cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    cf_namespace_id = os.getenv("CLOUDFLARE_KV_NAMESPACE_ID")
    cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if cf_account_id and cf_namespace_id and cf_api_token:
        from collectors.cloudflare_kv import sync_thread_ts_to_kv
        active_thread_ts = [k for k in pending_recs if k and k != ""]
        sync_thread_ts_to_kv(cf_account_id, cf_namespace_id, cf_api_token, active_thread_ts)
    else:
        _log.warning("CLOUDFLARE env vars not set — skipping KV sync (replies will not trigger GHA)")

    # --- Save state ---
    save_json_file(processed_uuids_path, list(processed_uuids))
    save_json_file(pending_recs_path, pending_recs)
    _log.info("State saved. Done.")


if __name__ == "__main__":
    main()
