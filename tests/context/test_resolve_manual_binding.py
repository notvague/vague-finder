import json
from pathlib import Path

import pytest

from experiments.namuwiki.resolve_manual_binding import resolve
from src.crawler.context.store import Store

URL = "https://namu.wiki/w/bomb(ILLIT)"
BODY = (
    "<h2>1. 개요</h2><p>ILLIT의 미니 3집이다.</p>"
    "<h2>2. 수록곡</h2>"
    "<h3>2.1. 밤소풍</h3><p>작사 홍길동</p>"
    "<h4>2.1.1. 여담</h4><p>예능 프로그램의 배경음악으로 사용되었다.</p>"
    "<h3>2.2. oops!</h3><p>작사 김철수</p>"
    "<h4>2.2.1. 여담</h4><p>인터넷 밈으로 유행했다.</p>"
)


def _html(title: str, body: str) -> bytes:
    return (
        f"<!doctype html><html><head><title>{title} - 나무위키</title></head>"
        f"<body><nav>메뉴</nav><article><h1>{title}</h1>{body}</article><footer>푸터</footer></body></html>"
    ).encode()


def _fixture(tmp_path: Path, decision: dict, *, status: str = "needs_review", cache_page: bool = True):
    raw = tmp_path / "data/raw"
    song = raw / "ILLIT_밤소풍_123"
    song.mkdir(parents=True)
    namuwiki = {
        "schema_version": "namuwiki_v3", "status": status,
        "collected_at": "2026-10-13T10:00:00+00:00", "facts": [],
    }
    if status == "needs_review":
        namuwiki["error_code"] = "page_title_mismatch"
    meta = {"song_id": "123", "metadata": {"title": "밤소풍", "artist": ["ILLIT"]}, "namuwiki": namuwiki}
    path = song / "meta.json"
    before = (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(before)

    cache = tmp_path / "data/context"
    if cache_page:
        Store(cache).save_snapshot(
            _html("bomb(ILLIT)", BODY), requested_url=URL, acquisition_method="http",
            final_url=URL, http_status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps([{"song_id": "123", "url": URL, **decision}]), encoding="utf-8")
    return raw, cache, decisions, path, before


def test_dry_run_is_read_only_and_apply_binds_only_the_reviewed_section(tmp_path):
    raw, cache, decisions, path, before = _fixture(
        tmp_path, {"scope": "section", "section": ["수록곡", "밤소풍"]},
    )
    backups = tmp_path / "data/context/manual_review_backups"
    plan = resolve(raw, cache, decisions, backups, apply=False, confirm=False)
    assert plan["status"] == "dry_run"
    assert plan["items"][0]["status"] == "ok"
    assert plan["items"][0]["fact_count"] == 1
    assert path.read_bytes() == before
    assert not backups.exists()

    applied = resolve(raw, cache, decisions, backups, apply=True, confirm=True)
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert applied["status"] == "applied"
    assert meta["namuwiki"]["status"] == "ok"
    assert meta["namuwiki"]["source_url"] == URL
    assert "error_code" not in meta["namuwiki"]
    texts = [fact["text"] for fact in meta["namuwiki"]["facts"]]
    assert len(texts) == 1 and "배경음악" in texts[0]
    backup = Path(applied["backup_dir"])
    assert (backup / "123.meta.json").read_bytes() == before
    manifest = json.loads((backup / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["section"] == ["수록곡", "밤소풍"]


def test_page_scope_takes_every_trivia_fact_of_the_document(tmp_path):
    raw, cache, decisions, _, _ = _fixture(tmp_path, {"scope": "page"})
    plan = resolve(raw, cache, decisions, tmp_path / "backups", apply=False, confirm=False)
    assert plan["items"][0]["fact_count"] == 2


def test_scope_none_records_the_page_without_borrowing_its_trivia(tmp_path):
    raw, cache, decisions, path, _ = _fixture(tmp_path, {"scope": "none"})
    resolve(raw, cache, decisions, tmp_path / "backups", apply=True, confirm=True)
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["namuwiki"]["status"] == "no_trivia"
    assert meta["namuwiki"]["facts"] == []
    assert meta["namuwiki"]["source_url"] == URL


def test_apply_requires_explicit_manual_confirmation(tmp_path):
    raw, cache, decisions, path, before = _fixture(tmp_path, {"scope": "page"})
    with pytest.raises(ValueError, match="confirm-all-reviewed-pages"):
        resolve(raw, cache, decisions, tmp_path / "backups", apply=True, confirm=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize("decision, kwargs, message", [
    ({"scope": "section", "section": ["수록곡", "없는 곡"]}, {}, "matches 0 sections"),
    ({"scope": "page"}, {"status": "not_found"}, "cannot be applied to this status"),
    ({"scope": "page"}, {"cache_page": False}, "not in the snapshot cache"),
    ({"scope": "section"}, {}, "'section' is required"),
    ({"scope": "whole"}, {}, "needs a url and a scope"),
])
def test_refuses_before_any_write(tmp_path, decision, kwargs, message):
    raw, cache, decisions, path, before = _fixture(tmp_path, decision, **kwargs)
    with pytest.raises(ValueError, match=message):
        resolve(raw, cache, decisions, tmp_path / "backups", apply=True, confirm=True)
    assert path.read_bytes() == before
    assert not (tmp_path / "backups").exists()
