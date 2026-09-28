"""
tests/test_audit_collection.py

수집 결과 점검 도구(audit_collection)의 회귀 테스트.

배경 (2026-09-18 점검): 시드마다 부분일치하는 **첫** 레코드만 돌려줬다. 정상 'My Love'
뒤에 잘못 수집된 'My Love (Duet Ver.)'가 있으면 정상 레코드만 검사하고 끝났다.

이제 시드에 연결된 레코드를 전부 점검한다. 수집 당시 요청 정보(crawl_status.input_artist/
input_title)가 있으면 그것으로 연결하고, 없는 예전 레코드는 제목 부분일치로 찾는다.

실행:
    venv/bin/python -m pytest tests/test_audit_collection.py -v
"""
from __future__ import annotations

import json

from src.crawler.scripts_py import audit_collection as audit


def record(song_id, artist, title, album="Album", crawl_input=None, youtube_title=""):
    rec = {
        "song_id": song_id,
        "metadata": {"artist": [artist], "title": title, "album": album, "genre": ["발라드"]},
        "links": {"youtube_title": youtube_title},
    }
    if crawl_input:
        rec["crawl_status"] = {"input_artist": crawl_input[0], "input_title": crawl_input[1]}
    return rec


def test_every_record_of_a_seed_is_inspected() -> None:
    records = [
        record("1", "이승철", "My Love"),
        record("2", "이승철", "My Love (Duet Ver.)", album="35주년 기념 앨범"),
    ]
    index = audit.index_by_artist(records)
    found = audit.find_records("이승철", "My Love", index)
    assert [r["song_id"] for r in found] == ["1", "2"]

    flagged, clean, unmatched, duplicates = audit.audit([("이승철", "My Love")], index)
    assert clean == 0                     # 정상 레코드도 '둘 중 하나'라는 경고로 점검 대상이 된다
    assert unmatched == 0 and duplicates == 1
    assert {f[2]["song_id"] for f in flagged} == {"1", "2"}
    duet = next(f for f in flagged if f[2]["song_id"] == "2")
    assert duet[3].hard_reject is True        # 요청하지 않은 듀엣판은 '재수집 권장'이다
    assert next(f for f in flagged if f[2]["song_id"] == "1")[3].hard_reject is False


def test_seed_link_uses_the_recorded_request_when_present() -> None:
    """제목이 부분일치하지 않아도 수집 당시 요청 정보로 연결된다."""
    records = [record("7", "아이유", "밤편지 (Night Letter)", crawl_input=("아이유", "밤편지"))]
    index = audit.index_by_artist(records)
    seed_index = audit.index_by_seed(records)
    assert [r["song_id"] for r in audit.find_records("아이유", "밤편지", index, seed_index)] == ["7"]
    # 같은 레코드가 제목 일치로도 잡히지만 한 번만 나온다
    assert len(audit.find_records("아이유", "밤편지", index, seed_index)) == 1


def test_records_without_a_request_link_fall_back_to_title_match() -> None:
    records = [record("9", "박효신", "야생화")]
    found = audit.find_records("박효신", "야생화", audit.index_by_artist(records), audit.index_by_seed(records))
    assert [r["song_id"] for r in found] == ["9"]


def test_duplicate_jsonl_lines_collapse_to_the_last_one(tmp_path) -> None:
    path = tmp_path / "all_songs.jsonl"
    lines = [
        record("1", "가수", "곡", album="첫 번째"),
        record("1", "가수", "곡", album="두 번째"),
        record("2", "가수", "다른곡"),
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
    loaded = audit.load_records(path)
    assert sorted(r["song_id"] for r in loaded) == ["1", "2"]
    assert next(r for r in loaded if r["song_id"] == "1")["metadata"]["album"] == "두 번째"
