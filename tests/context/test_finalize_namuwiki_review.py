"""Data mutation guards for private, hash-bound manual review decisions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pytest

from src.crawler.scripts_py import finalize_namuwiki_review as review


def _song(raw: Path, song_id: str) -> Path:
    folder = raw / f"song_{song_id}"
    folder.mkdir(parents=True)
    path = folder / "meta.json"
    path.write_text(json.dumps({
        "id": song_id,
        "metadata": {"title": f"Song {song_id}", "artist": ["Singer"]},
        "namuwiki": {
            "schema_version": "namuwiki_v3",
            "status": "needs_review",
            "source_url": "https://namu.wiki/w/unverified",
            "error_code": "ambiguous_binding",
            "collected_at": "2026-09-20T00:00:00+00:00",
            "facts": [],
        },
        "lyrics_data": {"full_lyrics": "original private data"},
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _decide(csv_path: Path, *, song_ids: set[str] | None = None) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if song_ids is None or row["song_id"] in song_ids:
            row["decision"] = "not_found"
            row["review_note"] = "Confirmed no matching song article by title and artist"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def catalogue(tmp_path: Path, monkeypatch):
    raw = tmp_path / "data/raw"
    one, two = _song(raw, "101"), _song(raw, "102")
    # These tiny fixtures exercise review mutation. The production metadata
    # validator still runs against full song metadata and media files.
    monkeypatch.setattr(review, "validate_meta_document", lambda *_a, **_k: [])
    cache = tmp_path / "data/context"
    decisions = cache / "review_decisions_step9.csv"
    review.execute(command="export", data_dir=raw, cache_dir=cache, csv_path=decisions)
    return raw, cache, decisions, one, two


def _run(catalogue, command: str):
    raw, cache, decisions, *_ = catalogue
    return review.execute(command=command, data_dir=raw, cache_dir=cache, csv_path=decisions)


def test_explicit_review_preserves_private_metadata_and_creates_exact_backup(catalogue):
    raw, cache, decisions, first, second = catalogue
    original = first.read_bytes()
    with pytest.raises(ValueError, match="explicit decision"):
        _run(catalogue, "validate")
    _decide(decisions)
    assert _run(catalogue, "validate")["pending"] == 2
    result = _run(catalogue, "apply")
    assert result["applied"] == 2
    new = json.loads(first.read_text(encoding="utf-8"))
    assert new["lyrics_data"]["full_lyrics"] == "original private data"
    assert new["namuwiki"]["status"] == "not_found"
    assert new["namuwiki"]["facts"] == []
    assert new["namuwiki"].get("source_url") is None
    assert new["namuwiki"]["error_code"] == review.DECISION_CODE
    backup = cache / "meta_backups/101" / f"{hashlib.sha256(original).hexdigest()}.json"
    assert backup.read_bytes() == original
    assert json.loads(second.read_text(encoding="utf-8"))["namuwiki"]["status"] == "not_found"
    assert _run(catalogue, "apply")["already_applied"] == 2
    with pytest.raises(ValueError, match="already exists"):
        _run(catalogue, "export")


def test_queue_must_be_complete_and_hashes_fresh_before_any_write(catalogue):
    raw, cache, decisions, first, second = catalogue
    before = first.read_bytes()
    _decide(decisions, song_ids={"101"})
    with pytest.raises(ValueError, match="explicit decision"):
        _run(catalogue, "apply")
    assert first.read_bytes() == before
    _decide(decisions)
    second.write_bytes(second.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metadata changed"):
        _run(catalogue, "apply")
    assert first.read_bytes() == before


def test_restart_after_write_before_receipt_uses_validated_backup(catalogue, monkeypatch):
    raw, cache, decisions, first, second = catalogue
    _decide(decisions)
    original_writer = review.write_meta
    once = {"done": False}

    def fail_after_write(*args):
        result = original_writer(*args)
        if not once["done"]:
            once["done"] = True
            raise RuntimeError("simulated power loss after atomic replace")
        return result

    monkeypatch.setattr(review, "write_meta", fail_after_write)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        _run(catalogue, "apply")
    monkeypatch.setattr(review, "write_meta", original_writer)
    result = _run(catalogue, "apply")
    assert result["applied"] == 1
    assert result["already_applied"] == 1
    assert json.loads(first.read_text(encoding="utf-8"))["namuwiki"]["status"] == "not_found"
    assert (cache / "manual_resolutions/101.json").exists()


def test_existing_review_receipt_cannot_be_reinterpreted(catalogue):
    raw, cache, decisions, first, second = catalogue
    _decide(decisions)
    _run(catalogue, "apply")
    with decisions.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["review_note"] = "A different reviewer conclusion was entered later"
    with decisions.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="receipt conflicts"):
        _run(catalogue, "validate")


def test_core_invalid_song_is_excluded_without_mutation(catalogue, monkeypatch):
    raw, cache, decisions, first, second = catalogue
    invalid = _song(raw, "103")
    before = invalid.read_bytes()
    monkeypatch.setattr(
        review, "validate_meta_document",
        lambda meta, **_k: ["invalid core"] if meta["id"] == "103" else [],
    )
    _decide(decisions)
    assert _run(catalogue, "apply")["core_invalid_skipped"] == 1
    assert invalid.read_bytes() == before
