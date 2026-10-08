import json
from pathlib import Path

import pytest

from experiments.namuwiki.resolve_manual_not_found import preflight, resolve


def _fixture(tmp_path: Path, *, status: str = "needs_review"):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    song = raw / "artist_song_123"
    song.mkdir()
    original = {
        "song_id": "123",
        "metadata": {"title": "song", "artist": ["artist"]},
        "namuwiki": {
            "schema_version": "namuwiki_v3", "status": status,
            "collected_at": "2026-09-29T10:00:00+00:00", "facts": [],
            "error_code": "title_only_page_artist_not_verified",
            "source_url": "https://example.invalid/w/wrong",
        },
    }
    path = song / "meta.json"
    raw_bytes = (json.dumps(original, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw_bytes)
    unresolved = tmp_path / "artifacts/context/unresolved.json"
    unresolved.parent.mkdir(parents=True)
    unresolved.write_text(json.dumps({
        "updated_at": "2026-09-29T12:00:00+00:00", "unresolved_count": 1,
        "needs_review_song_ids": ["123"], "unresolved_song_ids": ["123"],
        "collection_error_song_ids": [], "artifact_pending_song_ids": [],
        "pending_collection_song_ids": [],
        "songs": [{"song_id": "123", "namuwiki_status": "needs_review"}],
    }), encoding="utf-8")
    return raw, unresolved, path, raw_bytes


def test_dry_run_is_read_only_and_apply_preserves_review_backup(tmp_path):
    raw, unresolved, path, before = _fixture(tmp_path)
    backups = tmp_path / "data/context/manual_review_backups"
    plan = resolve(raw, unresolved, backups, 1, apply=False, confirm=False)
    assert plan["status"] == "dry_run"
    assert path.read_bytes() == before
    assert not backups.exists()

    applied = resolve(raw, unresolved, backups, 1, apply=True, confirm=True)
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert applied["status"] == "applied"
    assert meta["namuwiki"]["status"] == "not_found"
    assert meta["namuwiki"]["facts"] == []
    assert "source_url" not in meta["namuwiki"]
    assert "error_code" not in meta["namuwiki"]
    assert meta["namuwiki"]["collected_at"] == "2026-09-29T10:00:00+00:00"
    backup = Path(applied["backup_dir"])
    assert (backup / "123.meta.json").read_bytes() == before
    assert json.loads((backup / "review_manifest.json").read_text(encoding="utf-8"))["items"][0]["song_id"] == "123"


def test_refuse_other_unresolved_work_before_writes(tmp_path):
    raw, unresolved, path, before = _fixture(tmp_path)
    payload = json.loads(unresolved.read_text(encoding="utf-8"))
    payload["collection_error_song_ids"] = ["999"]
    unresolved.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="other work"):
        resolve(raw, unresolved, tmp_path / "backups", 1, apply=True, confirm=True)
    assert path.read_bytes() == before


def test_apply_requires_explicit_manual_confirmation(tmp_path):
    raw, unresolved, path, before = _fixture(tmp_path)
    backups = tmp_path / "backups"
    with pytest.raises(ValueError, match="confirm-all-reviewed-no-page"):
        resolve(raw, unresolved, backups, 1, apply=True, confirm=False)
    assert path.read_bytes() == before
    assert not backups.exists()


def test_refuse_stale_status_and_count_without_partial_writes(tmp_path):
    raw, unresolved, path, before = _fixture(tmp_path)
    with pytest.raises(ValueError, match="count changed"):
        preflight(raw, unresolved, 2)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["namuwiki"]["status"] = "not_found"
    changed["namuwiki"]["error_code"] = None
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="manual decision"):
        preflight(raw, unresolved, 1)
    assert path.read_bytes() != before
