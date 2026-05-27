"""Tests for main.py — all external calls (Avoma, Sheets, Claude) are mocked."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from collectors.avoma import AvomaTranscript
from processors.recommendation_builder import Recommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = {
    "avoma": {
        "enabled": True,
        "lookback_hours": 48,
        "sales_rep_emails": ["melissa@teambuildr.com"],
        "filter_internal": True,
    },
    "google_sheets": {
        "spreadsheet_id": "fake-sheet-id",
        "lead_tracker_tab": "LEAD TRACKER",
    },
    "slack": {"melissa_user_id": "U07EJE9B3NG"},
    "ai_model": "claude-sonnet-4-6",
    "pending_recs_file": "data/pending_recs.json",
    "processed_uuids_file": "data/processed_uuids.json",
    "pending_rec_ttl_days": 7,
}

_ENV = {
    "AVOMA_API_KEY": "avoma-fake-key",
    "ANTHROPIC_API_KEY": "anthropic-fake-key",
    "GOOGLE_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}',
    "GOOGLE_SHEET_ID": "fake-sheet-id",
}


def _make_transcript(**kwargs) -> AvomaTranscript:
    defaults = dict(
        uuid="uuid-001",
        title="Demo Call with Acme",
        start_at="2026-05-23T14:00:00Z",
        participants=["Jane Smith", "Melissa Rep"],
        call_type="demo",
        summary="Demo call; prospect showed interest.",
        buying_signals=["Asked about pricing"],
        objections=["Price concern"],
        action_items=["Send pricing doc"],
        competitors=[],
        gaps=[],
    )
    defaults.update(kwargs)
    return AvomaTranscript(**defaults)


def _make_recommendation(**kwargs) -> Recommendation:
    defaults = dict(
        match_confidence="high",
        lead_name="Jane Smith",
        organization="Acme Gym",
        sheet_row=5,
        proposed_updates={"Stage": "Demo", "Last Contact Date": "2026-05-23"},
        note="Demo held; prospect requested pricing doc.",
        reasoning="Moved to Demo stage.",
    )
    defaults.update(kwargs)
    return Recommendation(**defaults)


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


def _run_main(
    tmp_path: Path,
    transcripts: list[AvomaTranscript],
    sheet: dict,
    recommendation: Recommendation | None,
    initial_pending_recs: dict | None = None,
    initial_processed_uuids: list | None = None,
    config_overrides: dict | None = None,
):
    """
    Wire up all mocks, write temp state files, patch config/env paths, and run main().
    Returns the pending_recs and processed_uuids dicts written to disk.
    """
    # Write state files into tmp_path
    pending_path = tmp_path / "data" / "pending_recs.json"
    uuids_path = tmp_path / "data" / "processed_uuids.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    if initial_pending_recs is not None:
        pending_path.write_text(json.dumps(initial_pending_recs))
    if initial_processed_uuids is not None:
        uuids_path.write_text(json.dumps(initial_processed_uuids))

    # Build config with paths pointing to tmp_path
    cfg = dict(_CONFIG)
    cfg["pending_recs_file"] = str(pending_path)
    cfg["processed_uuids_file"] = str(uuids_path)
    if config_overrides:
        cfg.update(config_overrides)

    import main as main_module

    with (
        patch.dict(os.environ, _ENV),
        patch("main.load_dotenv"),
        patch("main.fetch_recent_meetings", return_value=transcripts),
        patch("main.load_lead_tracker", return_value=sheet),
        patch("main.build_recommendation", return_value=recommendation),
        patch("builtins.open", wraps=_patched_open(cfg, pending_path, uuids_path)),
    ):
        main_module.main()

    pending_recs = json.loads(pending_path.read_text()) if pending_path.exists() else {}
    processed_uuids = json.loads(uuids_path.read_text()) if uuids_path.exists() else []
    return pending_recs, processed_uuids


def _patched_open(config: dict, pending_path: Path, uuids_path: Path):
    """
    Return a wrapper for builtins.open that intercepts config.json reads and
    passes all other opens through to the real filesystem.
    """
    real_open = open  # capture real built-in before patch

    def wrapper(file, mode="r", *args, **kwargs):
        file_str = str(file)
        if file_str.endswith("config.json") and "w" not in mode:
            import io
            return io.StringIO(json.dumps(config))
        return real_open(file, mode, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Direct unit tests for helper functions (no file I/O, no subprocess)
# ---------------------------------------------------------------------------

class TestExpireStaleRecs:
    """Unit tests for main.expire_stale_recs."""

    def test_recent_rec_is_kept(self):
        from main import expire_stale_recs

        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        recent = (now - timedelta(days=3)).isoformat().replace("+00:00", "Z")
        recs = {"uuid-001": {"created_at": recent, "lead_name": "Jane"}}
        result = expire_stale_recs(recs, ttl_days=7, now=now)
        assert "uuid-001" in result

    def test_stale_rec_is_removed(self):
        from main import expire_stale_recs

        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        stale = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
        recs = {"uuid-old": {"created_at": stale, "lead_name": "Old Lead"}}
        result = expire_stale_recs(recs, ttl_days=7, now=now)
        assert "uuid-old" not in result

    def test_exactly_at_boundary_is_kept(self):
        """A rec created exactly ttl_days ago (to the second) is NOT expired — must be strictly older."""
        from main import expire_stale_recs

        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        at_boundary = (now - timedelta(days=7)).isoformat().replace("+00:00", "Z")
        recs = {"uuid-boundary": {"created_at": at_boundary, "lead_name": "Boundary"}}
        result = expire_stale_recs(recs, ttl_days=7, now=now)
        # Exactly at boundary (created_at == cutoff) is not strictly older, so it is kept
        assert "uuid-boundary" in result

    def test_unparseable_timestamp_is_kept(self):
        """Recs with unparseable created_at should be kept (don't silently drop data)."""
        from main import expire_stale_recs

        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        recs = {"uuid-bad": {"created_at": "not-a-date", "lead_name": "Bad Date"}}
        result = expire_stale_recs(recs, ttl_days=7, now=now)
        assert "uuid-bad" in result

    def test_mixed_keeps_recent_removes_stale(self):
        from main import expire_stale_recs

        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        recent = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
        stale = (now - timedelta(days=14)).isoformat().replace("+00:00", "Z")
        recs = {
            "uuid-recent": {"created_at": recent, "lead_name": "Recent"},
            "uuid-stale": {"created_at": stale, "lead_name": "Stale"},
        }
        result = expire_stale_recs(recs, ttl_days=7, now=now)
        assert "uuid-recent" in result
        assert "uuid-stale" not in result


class TestMatchParticipantToSheet:
    """Unit tests for main.match_participant_to_sheet."""

    def test_single_match_returns_row(self):
        from main import match_participant_to_sheet

        row = _make_sheet_row(row_index=5, lead_name="Jane Smith")
        sheet = {"jane smith": row}
        current_row, ambiguous = match_participant_to_sheet(["Jane Smith", "Melissa Rep"], sheet)
        assert current_row is row
        assert ambiguous is False

    def test_no_match_returns_none(self):
        from main import match_participant_to_sheet

        sheet = {"john doe": _make_sheet_row(lead_name="John Doe")}
        current_row, ambiguous = match_participant_to_sheet(["Unknown Person"], sheet)
        assert current_row is None
        assert ambiguous is False

    def test_multiple_matches_returns_ambiguous(self):
        from main import match_participant_to_sheet

        row_a = _make_sheet_row(row_index=3, lead_name="Jane Smith")
        row_b = _make_sheet_row(row_index=7, lead_name="Bob Jones")
        sheet = {"jane smith": row_a, "bob jones": row_b}
        current_row, ambiguous = match_participant_to_sheet(["Jane Smith", "Bob Jones"], sheet)
        assert current_row is None
        assert ambiguous is True

    def test_same_row_matched_twice_not_ambiguous(self):
        """If two participant names map to the same sheet row, it's not ambiguous."""
        from main import match_participant_to_sheet

        row = _make_sheet_row(row_index=5, lead_name="Jane Smith", organization="Acme Gym")
        sheet = {"jane smith": row, "acme gym": row}
        current_row, ambiguous = match_participant_to_sheet(
            ["Jane Smith", "Acme Gym rep"], sheet
        )
        # "acme gym rep" won't match, "jane smith" will → single match
        assert current_row is row
        assert ambiguous is False

    def test_empty_participants_returns_none(self):
        from main import match_participant_to_sheet

        sheet = {"jane smith": _make_sheet_row()}
        current_row, ambiguous = match_participant_to_sheet([], sheet)
        assert current_row is None
        assert ambiguous is False


# ---------------------------------------------------------------------------
# Integration-style tests using _run_main
# ---------------------------------------------------------------------------

class TestCallTypeFilter:
    """call_type filter: demo and follow_up are processed; other and onboarding are skipped."""

    def test_demo_is_processed(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-demo", call_type="demo")
        rec = _make_recommendation()

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=rec,
        )

        assert "uuid-demo" in pending_recs
        assert "uuid-demo" in processed_uuids

    def test_follow_up_is_processed(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-followup", call_type="follow_up")
        rec = _make_recommendation()

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=rec,
        )

        assert "uuid-followup" in pending_recs
        assert "uuid-followup" in processed_uuids

    def test_other_is_skipped(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-other", call_type="other")

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={},
            recommendation=None,
        )

        assert "uuid-other" not in pending_recs
        # Still added to processed_uuids so we don't re-evaluate it
        assert "uuid-other" in processed_uuids

    def test_onboarding_is_skipped(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-onboard", call_type="onboarding")

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={},
            recommendation=None,
        )

        assert "uuid-onboard" not in pending_recs
        assert "uuid-onboard" in processed_uuids


class TestUUIDDedup:
    """Transcript whose UUID is already in processed_uuids.json is skipped."""

    def test_already_processed_uuid_is_skipped(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-dup", call_type="demo")

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=_make_recommendation(),
            initial_processed_uuids=["uuid-dup"],
        )

        # Should not be in pending_recs — was already processed
        assert "uuid-dup" not in pending_recs

    def test_new_uuid_is_processed(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-new", call_type="demo")

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=_make_recommendation(),
            initial_processed_uuids=["uuid-other"],
        )

        assert "uuid-new" in pending_recs
        assert "uuid-new" in processed_uuids


class TestTTLExpiry:
    """Pending recs older than ttl_days are removed; recent ones are kept."""

    def test_stale_rec_removed_from_pending(self, tmp_path):
        now = datetime.now(timezone.utc)
        stale_ts = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
        initial_pending = {
            "uuid-stale": {
                "created_at": stale_ts,
                "lead_name": "Old Lead",
                "avoma_uuid": "uuid-stale",
            }
        }

        pending_recs, _ = _run_main(
            tmp_path,
            transcripts=[],
            sheet={},
            recommendation=None,
            initial_pending_recs=initial_pending,
        )

        assert "uuid-stale" not in pending_recs

    def test_recent_rec_kept_in_pending(self, tmp_path):
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
        initial_pending = {
            "uuid-recent": {
                "created_at": recent_ts,
                "lead_name": "Recent Lead",
                "avoma_uuid": "uuid-recent",
            }
        }

        pending_recs, _ = _run_main(
            tmp_path,
            transcripts=[],
            sheet={},
            recommendation=None,
            initial_pending_recs=initial_pending,
        )

        assert "uuid-recent" in pending_recs


class TestSheetMatching:
    """Sheet match behavior passed to build_recommendation."""

    def test_matched_current_row_passed_to_builder(self, tmp_path):
        """When participant matches sheet, current_row is the matched row."""
        import main as main_module

        transcript = _make_transcript(uuid="uuid-match", participants=["Jane Smith"])
        sheet_row = _make_sheet_row(row_index=5, lead_name="Jane Smith")
        sheet = {"jane smith": sheet_row}
        rec = _make_recommendation()

        captured_current_row = {}

        def fake_build_recommendation(transcript, current_row, model, anthropic_api_key):
            captured_current_row["value"] = current_row
            return rec

        with (
            patch.dict(os.environ, _ENV),
            patch("main.load_dotenv"),
            patch("main.fetch_recent_meetings", return_value=[transcript]),
            patch("main.load_lead_tracker", return_value=sheet),
            patch("main.build_recommendation", side_effect=fake_build_recommendation),
            patch("builtins.open", wraps=_patched_open(
                dict(_CONFIG, **{
                    "pending_recs_file": str(tmp_path / "data" / "pending_recs.json"),
                    "processed_uuids_file": str(tmp_path / "data" / "processed_uuids.json"),
                }),
                tmp_path / "data" / "pending_recs.json",
                tmp_path / "data" / "processed_uuids.json",
            )),
        ):
            (tmp_path / "data").mkdir(parents=True, exist_ok=True)
            main_module.main()

        assert captured_current_row["value"] is sheet_row

    def test_no_match_passes_none_to_builder(self, tmp_path):
        """When no participant matches, current_row=None is passed."""
        import main as main_module

        transcript = _make_transcript(uuid="uuid-nomatch", participants=["Unknown Person"])
        sheet = {"jane smith": _make_sheet_row(lead_name="Jane Smith")}
        rec = _make_recommendation(sheet_row=None)

        captured_current_row = {}

        def fake_build_recommendation(transcript, current_row, model, anthropic_api_key):
            captured_current_row["value"] = current_row
            return rec

        with (
            patch.dict(os.environ, _ENV),
            patch("main.load_dotenv"),
            patch("main.fetch_recent_meetings", return_value=[transcript]),
            patch("main.load_lead_tracker", return_value=sheet),
            patch("main.build_recommendation", side_effect=fake_build_recommendation),
            patch("builtins.open", wraps=_patched_open(
                dict(_CONFIG, **{
                    "pending_recs_file": str(tmp_path / "data" / "pending_recs.json"),
                    "processed_uuids_file": str(tmp_path / "data" / "processed_uuids.json"),
                }),
                tmp_path / "data" / "pending_recs.json",
                tmp_path / "data" / "processed_uuids.json",
            )),
        ):
            (tmp_path / "data").mkdir(parents=True, exist_ok=True)
            main_module.main()

        assert captured_current_row["value"] is None


class TestBuildRecommendationReturnsNone:
    """When build_recommendation returns None, UUID is still added to processed_uuids."""

    def test_uuid_added_to_processed_when_rec_is_none(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-no-rec", call_type="demo")

        pending_recs, processed_uuids = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={},
            recommendation=None,
        )

        assert "uuid-no-rec" not in pending_recs
        assert "uuid-no-rec" in processed_uuids


class TestPendingRecsStructure:
    """pending_recs.json written with correct structure after a successful run."""

    def test_pending_rec_has_all_required_keys(self, tmp_path):
        transcript = _make_transcript(
            uuid="uuid-struct",
            call_type="demo",
            start_at="2026-05-23T14:00:00Z",
        )
        rec = _make_recommendation(
            lead_name="Jane Smith",
            organization="Acme Gym",
            sheet_row=5,
            proposed_updates={"Stage": "Demo", "Last Contact Date": "2026-05-23"},
            note="Demo held.",
            reasoning="Strong signals.",
            match_confidence="high",
        )

        pending_recs, _ = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=rec,
        )

        assert "uuid-struct" in pending_recs
        entry = pending_recs["uuid-struct"]

        required_keys = {
            "thread_ts",
            "lead_name",
            "organization",
            "sheet_row",
            "proposed_updates",
            "note",
            "reasoning",
            "match_confidence",
            "avoma_uuid",
            "created_at",
            "ambiguous_match",
            "call_date",
            "call_type",
        }
        assert required_keys.issubset(entry.keys()), (
            f"Missing keys: {required_keys - entry.keys()}"
        )

    def test_pending_rec_values_correct(self, tmp_path):
        transcript = _make_transcript(
            uuid="uuid-vals",
            call_type="follow_up",
            start_at="2026-05-23T14:00:00Z",
        )
        rec = _make_recommendation(
            lead_name="Bob Jones",
            organization="Bob's Gym",
            sheet_row=8,
            match_confidence="medium",
            note="Follow-up completed.",
            reasoning="Mid-stage follow-up.",
        )

        pending_recs, _ = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"bob jones": _make_sheet_row(row_index=8, lead_name="Bob Jones")},
            recommendation=rec,
        )

        entry = pending_recs["uuid-vals"]
        assert entry["lead_name"] == "Bob Jones"
        assert entry["organization"] == "Bob's Gym"
        assert entry["sheet_row"] == 8
        assert entry["match_confidence"] == "medium"
        assert entry["avoma_uuid"] == "uuid-vals"
        assert entry["call_date"] == "2026-05-23"
        assert entry["call_type"] == "follow_up"
        assert entry["thread_ts"] == ""  # not yet filled in by slack_notifier
        assert entry["ambiguous_match"] is False

    def test_pending_rec_created_at_is_iso_format(self, tmp_path):
        transcript = _make_transcript(uuid="uuid-ts", call_type="demo")
        rec = _make_recommendation()

        pending_recs, _ = _run_main(
            tmp_path,
            transcripts=[transcript],
            sheet={"jane smith": _make_sheet_row()},
            recommendation=rec,
        )

        entry = pending_recs["uuid-ts"]
        created_at = entry["created_at"]
        # Should end with Z (UTC)
        assert created_at.endswith("Z"), f"created_at={created_at!r} does not end with Z"
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        assert parsed is not None

    def test_ambiguous_match_flag_set_correctly(self, tmp_path):
        """When two different participants match two different sheet rows, ambiguous=True."""
        import main as main_module

        row_a = _make_sheet_row(row_index=3, lead_name="Jane Smith")
        row_b = _make_sheet_row(row_index=7, lead_name="Bob Jones")
        sheet = {"jane smith": row_a, "bob jones": row_b}

        transcript = _make_transcript(
            uuid="uuid-ambig",
            call_type="demo",
            participants=["Jane Smith", "Bob Jones"],
        )
        rec = _make_recommendation()

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.dict(os.environ, _ENV),
            patch("main.load_dotenv"),
            patch("main.fetch_recent_meetings", return_value=[transcript]),
            patch("main.load_lead_tracker", return_value=sheet),
            patch("main.build_recommendation", return_value=rec),
            patch("builtins.open", wraps=_patched_open(
                dict(_CONFIG, **{
                    "pending_recs_file": str(data_dir / "pending_recs.json"),
                    "processed_uuids_file": str(data_dir / "processed_uuids.json"),
                }),
                data_dir / "pending_recs.json",
                data_dir / "processed_uuids.json",
            )),
        ):
            main_module.main()

        pending_recs = json.loads((data_dir / "pending_recs.json").read_text())
        assert pending_recs["uuid-ambig"]["ambiguous_match"] is True
