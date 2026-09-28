"""
tests/test_melon_comments.py

멜론 댓글 수집 품질 회귀 테스트.

배경 (수집분 961곡 점검에서 발견):

1. HTML 엔티티가 그대로 남았다. 태그(`<...>`)만 지우고 엔티티는 풀지 않아서
   425곡 640개 댓글에 `&hellip;` 500회(382개 댓글), `&#39;` 144회, `&quot;` 127회 등이 섞여 있었다.
   유튜브 쪽은 0건이라 멜론 API 응답 특성이다.

2. 최신순으로 긁고 있었다. `sortType=0`은 최신순이라 추천수 0~8짜리가 오고,
   `sortType=1`(추천순)은 2454, 1967, 1723을 준다. 유튜브는 이미 좋아요순으로
   정렬해 LLM에 넘기고 있었는데 멜론만 정렬 없이 API 순서 그대로였다.

3. 추천수를 버렸다. `like_count`를 0으로 하드코딩해서 정렬 근거가 없었다.

테스트 문장은 모두 새로 지어낸 것이다. 실제 사용자 댓글이나 가사를 옮기지 않는다.

실행:
    venv/bin/python -m pytest tests/test_melon_comments.py -v
"""
from __future__ import annotations

import json
import re
import sys

import pytest

from src.crawler.scripts_py import backfill_comment_entities as backfill
from src.crawler.scripts_py import collect_melon_data as cmd
from src.crawler.scripts_py.collect_melon_data import preprocess_melon_comment_candidates

ENTITY = re.compile(r"&(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);")


def raw(text, recm=0):
    return {"AUTH_CNTTS": text, "RECM_CNT": recm}


# --- 1. HTML 엔티티 ------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("새벽에 들으면 마음이 조용해지는 노래&hellip; 계속 듣게 된다",
         "새벽에 들으면 마음이 조용해지는 노래… 계속 듣게 된다"),
        ("&quot;괜찮다&quot;는 말을 스스로에게 하게 되는 곡",
         '"괜찮다"는 말을 스스로에게 하게 되는 곡'),
        ("비 오는 날이면 &#39;그 골목&#39;이 떠오른다",
         "비 오는 날이면 '그 골목'이 떠오른다"),
        ("산책할 때 &amp; 버스 기다릴 때 자주 듣는다",
         "산책할 때 & 버스 기다릴 때 자주 듣는다"),
    ],
)
def test_html_entities_are_unescaped(text, expected) -> None:
    out = preprocess_melon_comment_candidates([raw(text)])
    assert len(out) == 1, out
    assert out[0]["text"] == expected
    assert not ENTITY.search(out[0]["text"]), out[0]["text"]


def test_entities_do_not_break_the_length_filter() -> None:
    """엔티티를 푼 뒤의 길이로 재야 한다. '&hellip;'은 8자가 아니라 1자다."""
    # 엔티티를 안 풀면 8자 이상이지만, 풀면 8자 미만이라 걸러져야 한다
    assert preprocess_melon_comment_candidates([raw("음&hellip;")]) == []


# --- 2·3. 추천순 요청과 추천수 보존 ------------------------------------------------

def test_recommend_count_is_preserved() -> None:
    """추천수를 버리면 LLM에 넘길 순서를 정할 근거가 없다."""
    out = preprocess_melon_comment_candidates([raw("이 노래 들으면 그때가 생각난다", 2454)])
    assert out[0]["like_count"] == 2454


def test_missing_recommend_count_defaults_to_zero() -> None:
    out = preprocess_melon_comment_candidates([{"AUTH_CNTTS": "이 노래 들으면 그때가 생각난다"}])
    assert out[0]["like_count"] == 0


def test_sort_uses_recommend_count(monkeypatch) -> None:
    """LLM에 넘기기 전 추천순으로 정렬해야 한다(유튜브 filter_comments와 동일)."""
    seen = {}

    def fake_select(candidates, **kwargs):
        seen["order"] = [c["like_count"] for c in candidates]
        return [c["text"] for c in candidates]

    monkeypatch.setattr(cmd, "select_emotional_comments_with_llm", fake_select)
    cmd.filter_melon_comments([
        raw("이 노래 들으면 그 겨울이 떠오른다", 3),
        raw("새벽에 혼자 듣기 참 마음이 편해지는 곡", 900),
        raw("헤어진 날 계속 돌려 들었던 기억이 난다", 120),
    ])
    assert seen["order"] == [900, 120, 3]


def test_fetch_sends_recommended_sort_and_carries_recmcnt(monkeypatch) -> None:
    """실제로 보낸 요청과 API 응답의 추천수가 선별 단계까지 가는지 본다.

    fetch_melon_comments는 예외를 삼키고 []를 돌려주므로, 요청과 결과가 실제로 있었는지
    먼저 확인해야 스텁이 깨졌을 때 조용히 통과하지 않는다.
    """
    calls = []

    class FakeResponse:
        status_code = 200   # 댓글 API도 상태를 본다. 403/429는 차단, 5xx는 일시 오류다.

        def __init__(self, params):
            self.params = params

        def json(self):
            if self.params["pageNo"] == "1":
                return {"result": {"cmtList": [
                    {"cmtInfo": {"cmtCont": "이 노래 들으면 그 겨울이 떠오른다", "recmCnt": 3}},
                    {"cmtInfo": {"cmtCont": "새벽에 혼자 듣기 참 마음이 편해지는 곡", "recmCnt": 900}},
                ]}}
            return {"result": {"cmtList": []}}

    def fake_get(url, params=None, **kwargs):
        calls.append(dict(params))
        return FakeResponse(params)

    seen = {}

    def fake_select(candidates, **kwargs):
        seen["likes"] = [c["like_count"] for c in candidates]
        return [c["text"] for c in candidates]

    monkeypatch.setattr(cmd.requests, "get", fake_get)
    monkeypatch.setattr(cmd, "select_emotional_comments_with_llm", fake_select)

    out = cmd.fetch_melon_comments("123", max_pages=3)

    assert calls, "requests.get이 호출되지 않았다"
    assert len(out) == 2
    # sortType 0은 최신순이라 추천수 0~8짜리를 긁어 온다. 1이 추천순이다.
    assert [c["sortType"] for c in calls] == ["1", "1"]
    assert [c["pageNo"] for c in calls] == ["1", "2"]
    # cmtInfo.recmCnt -> RECM_CNT -> like_count, 그리고 추천순 정렬
    assert seen["likes"] == [900, 3]


# --- 이미 저장된 댓글 정제 (backfill) ---------------------------------------------

def test_backfill_unescapes_stored_melon_comments() -> None:
    record = {"comments": {"melon": ["퇴근길에 듣다가 울컥했다&hellip;", "엔티티 없는 댓글"],
                           "youtube": ["&quot;건드리지 않음&quot;"]}}
    assert backfill.clean_comments(record) == 1
    assert record["comments"]["melon"] == ["퇴근길에 듣다가 울컥했다…", "엔티티 없는 댓글"]
    assert record["comments"]["youtube"] == ["&quot;건드리지 않음&quot;"]   # 유튜브는 원래 0건


@pytest.mark.parametrize(
    "text",
    [
        "창밖 보면서 듣는 곡&amp;hellip; 계속 듣는다",   # 이중 이스케이프
        "웃는 표시는 &amp;lt;3 이렇게 쓴다고 배웠음",      # 사용자가 일부러 쓴 '&lt;3'
    ],
)
def test_backfill_on_old_data_matches_the_fixed_collector(text) -> None:
    """고치기 전 수집기가 저장한 원문(엔티티 그대로)에 backfill을 한 번 적용하면, 고친 수집기가
    저장했을 결과와 같아야 한다. 여러 번 풀면 사용자가 쓴 '&lt;3'이 '<3'이 된다."""
    record = {"comments": {"melon": [text]}}
    backfill.clean_comments(record)
    collected = preprocess_melon_comment_candidates([raw(text)])[0]["text"]
    assert record["comments"]["melon"] == [collected]


def test_backfill_must_not_run_on_comments_saved_by_the_fixed_collector() -> None:
    """한계를 고정한다: 고친 수집기로 이미 한 번 풀어 저장한 댓글은 구분할 수 없어 또 풀린다.

    그래서 backfill은 수집기 수정 전 데이터에 한 번만 돌린다(모듈 설명). 이 동작이 바뀌면
    모듈 설명도 함께 고쳐야 한다.
    """
    saved = preprocess_melon_comment_candidates([raw("웃는 표시는 &amp;lt;3 이렇게 쓴다고 배웠음")])[0]["text"]
    assert saved == "웃는 표시는 &lt;3 이렇게 쓴다고 배웠음"
    record = {"comments": {"melon": [saved]}}
    assert backfill.clean_comments(record) == 1
    assert record["comments"]["melon"] == ["웃는 표시는 <3 이렇게 쓴다고 배웠음"]


def test_backfill_is_idempotent_on_existing_data() -> None:
    record = {"comments": {"melon": ["눈 오는 날 &#39;첫 장면&#39;이 생각난다"]}}
    assert backfill.clean_comments(record) == 1
    assert backfill.clean_comments(record) == 0


def test_backfill_is_dry_run_by_default_and_write_keeps_a_backup(tmp_path, monkeypatch) -> None:
    path = tmp_path / "all_songs.jsonl"
    lines = [
        json.dumps({"song_id": "1", "comments": {"melon": ["퇴근길에 듣다가 울컥했다&hellip;"]}},
                   ensure_ascii=False),
        json.dumps({"song_id": "2", "comments": {"melon": ["엔티티 없는 댓글"]}}, ensure_ascii=False),
    ]
    original = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(original)

    # 옵션 없이 실행하면 아무것도 쓰지 않는다
    monkeypatch.setattr(sys, "argv", ["backfill", "--jsonl", str(path)])
    assert backfill.main() == 0
    assert path.read_bytes() == original
    assert not (tmp_path / "all_songs.jsonl.bak").exists()

    # --write: 원본을 .bak으로 남기고, 바뀌지 않는 줄은 그대로 둔다
    monkeypatch.setattr(sys, "argv", ["backfill", "--jsonl", str(path), "--write"])
    assert backfill.main() == 0
    assert (tmp_path / "all_songs.jsonl.bak").read_bytes() == original
    rewritten = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(rewritten[0])["comments"]["melon"] == ["퇴근길에 듣다가 울컥했다…"]
    assert rewritten[1] == lines[1]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_backfill_reads_jsonl_split_on_newlines_only(tmp_path, separator) -> None:
    """json.dumps(ensure_ascii=False)는 U+2028 등을 이스케이프하지 않는다.
    str.splitlines()로 나누면 레코드 중간이 잘려 미리보기조차 실패했다."""
    path = tmp_path / "all_songs.jsonl"
    record = {
        "song_id": "1",
        "metadata": {"album_description": f"첫 줄{separator}둘째 줄"},
        "comments": {"melon": ["퇴근길에 듣다가 울컥했다&hellip;"]},
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    songs, total, _ = backfill.process_jsonl(path, write=False)
    assert (songs, total) == (1, 1)


def test_backfill_raw_dir_respects_dry_run(tmp_path) -> None:
    song_dir = tmp_path / "가수_곡_123"
    song_dir.mkdir()
    meta = song_dir / "meta.json"
    original = json.dumps({"comments": {"melon": ["비 오는 날 &#39;그 골목&#39;"]}},
                          ensure_ascii=False, indent=2).encode("utf-8")
    meta.write_bytes(original)

    assert backfill.process_raw_dir(tmp_path, write=False) == (1, 1)
    assert meta.read_bytes() == original
    assert not (song_dir / "meta.json.bak").exists()

    assert backfill.process_raw_dir(tmp_path, write=True) == (1, 1)
    assert (song_dir / "meta.json.bak").read_bytes() == original
    assert json.loads(meta.read_text(encoding="utf-8"))["comments"]["melon"] == ["비 오는 날 '그 골목'"]


# --- 댓글 API 장애가 '댓글 없음'으로 저장되면 안 된다 ---------------------------------------
# 예전에는 HTTP 상태를 보지 않고 JSON 파싱 실패만 잡아 []를 돌려줬다. 403·429·503이 모두
# 조용히 빈 목록이 됐고, 그 곡은 '댓글 없는 곡'으로 굳었다.

class _StatusResponse:
    def __init__(self, status_code, payload=None, body=""):
        self.status_code = status_code
        self._payload = payload
        self.text = body

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value")
        return self._payload


@pytest.mark.parametrize("status", [403, 429])
def test_comment_api_block_stops_the_batch(monkeypatch, status) -> None:
    monkeypatch.setattr(cmd.time, "sleep", lambda s: None)
    monkeypatch.setattr(cmd.requests, "get", lambda *a, **k: _StatusResponse(status))
    with pytest.raises(cmd.MelonAccessError):
        cmd.fetch_melon_comments("123")


def test_comment_api_server_error_is_retried_then_raised(monkeypatch) -> None:
    monkeypatch.setattr(cmd.time, "sleep", lambda s: None)
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return _StatusResponse(503)

    monkeypatch.setattr(cmd.requests, "get", fake_get)
    with pytest.raises(cmd.MelonTransientError):
        cmd.fetch_melon_comments("123")
    assert len(calls) == cmd.PAGE_RETRIES + 1


def test_comment_api_non_json_body_is_transient_not_empty(monkeypatch) -> None:
    """200으로 오는 오류 안내 HTML. 빈 목록으로 돌려주면 '댓글 없는 곡'이 된다."""
    monkeypatch.setattr(cmd.time, "sleep", lambda s: None)
    monkeypatch.setattr(cmd.requests, "get",
                        lambda *a, **k: _StatusResponse(200, body="<html>error</html>"))
    with pytest.raises(cmd.MelonTransientError):
        cmd.fetch_melon_comments("123")
