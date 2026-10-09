"""표지 질의 진단 — 같은 표지를 한 번만 세고, 대상 질의마다 시각 조건 문장이 있다."""
from __future__ import annotations

import json

from src.eval.schema import V05_QUERY_SETS
from src.retrieval.diagnose_image_queries import QUERIES, VISUAL_ONLY, cover_hash, cover_key


def test_cover_key_ignores_size_and_resize_suffix() -> None:
    a = "https://cdnimg.melon.co.kr/cm2/album/images/110/33/394/11033394_20220916124707_500.jpg?abc/melon/resize/500"
    b = "https://cdnimg.melon.co.kr/cm2/album/images/110/33/394/11033394_20220916124707_500.jpg/melon/resize/282"
    assert cover_key(a) == cover_key(b)
    # 기록에는 주소가 아니라 해시만 남는다 — 공개 저장소에 표지 주소를 올리지 않는다
    assert cover_hash(a) == cover_hash(b)
    assert "melon" not in cover_hash(a) and len(cover_hash(a)) == 12
    assert cover_hash("") == ""


def test_every_image_query_has_a_visual_only_sentence() -> None:
    queries = json.load(open(QUERIES, encoding="utf-8"))["queries"]
    # 진단(v25)은 v0.5 세트의 표지 질의를 v22·v23 기록과 대조한다. v09(2026-10)의 표지 질의는
    # 그 기록에 없고 시각 조건 문장도 아직 쓰지 않았다 — 대상에서 뺀다.
    wanted = {
        q["query_id"] for q in queries
        if q.get("modality_focus") in ("image", "multimodal") and q.get("positives")
        and q.get("label_status") == "labeled" and q.get("query_set") in V05_QUERY_SETS
    }
    assert wanted == set(VISUAL_ONLY)


def test_link_check_reports_missing_rows_and_rank_mismatch(monkeypatch) -> None:
    """현재 인덱스와 과거 기록을 합치기 전에 연결을 확인한다 — 빠진 행·다른 순위는 멈춘다."""
    import src.retrieval.diagnose_image_queries as diag

    monkeypatch.setattr(diag, "V22_RUNINFO", {})   # 측정 조건 대조는 로컬 기록에 기대므로 여기서는 뺀다
    queries = [{"query_id": "x", "positives": ["a", "b"]}]
    trace = {("x", "a"): {"pool_rank": "3"}, ("x", "b"): {"pool_rank": ""}}
    explains = {("x", "a"): {"rank_before_rerank": 3}, ("x", "b"): {"rank_before_rerank": "out"}}
    assert diag.check_links(queries, trace, {"x": {"candidate_rank@30": "3"}}, explains, 10) == []

    problems = diag.check_links(queries, trace, {"x": {"candidate_rank@30": "5"}}, explains, 10)
    assert any("candidate_rank@30" in p for p in problems)

    del trace[("x", "b")]
    problems = diag.check_links(queries, trace, {"x": {"candidate_rank@30": "3"}}, explains, 10)
    assert any("x/b" in p for p in problems)


def test_link_check_compares_every_answer_song_and_the_analysis(monkeypatch) -> None:
    """복수 정답은 곡마다 비교한다(최솟값만 보면 다른 곡의 어긋남이 가려진다). 분석은 내용으로 비교한다."""
    import json as _json

    import src.retrieval.diagnose_image_queries as diag
    from src.backend.schemas.query import QueryAnalysis

    monkeypatch.setattr(diag, "V22_RUNINFO", {})
    queries = [{"query_id": "x", "positives": ["a", "b"]}]
    trace = {("x", "a"): {"pool_rank": "18"}, ("x", "b"): {"pool_rank": "25"}}
    explains = {("x", "a"): {"rank_before_rerank": 18}, ("x", "b"): {"rank_before_rerank": 24}}
    analysis = QueryAnalysis(original_query="q", intent_type="mood", image_english_query="blue",
                             audio_english_query="", vocal_gender="여성")
    detail = {"x": {"candidate_rank@30": "18", "analysis_json": analysis.model_dump_json()}}
    problems = diag.check_links(queries, trace, detail, explains, 10, {"x": analysis})
    assert problems == ["x/b: v23 후보 순위 25 ≠ v22 리랭킹 전 순위 24"]

    explains[("x", "b")]["rank_before_rerank"] = 25
    changed = analysis.model_copy(update={"vocal_gender": "남성"})
    problems = diag.check_links(queries, trace, detail, explains, 10, {"x": changed})
    assert problems == ["x: 분석 캐시 내용이 v22 측정 때와 다르다"]
    assert _json.loads(detail["x"]["analysis_json"])["vocal_gender"] == "여성"
