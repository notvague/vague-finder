"""소리 경로의 융합 배율(`AUDIO_FUSION_SCALE`) — 실험 스위치.

분석이 소리 가중을 0.7로 주면 가중 RRF에서 텍스트 1위(0.3/61)가 소리 경로 상위 30곡(0.7/61~0.7/90)
전부보다 낮아져 후보에서 밀린다(v10 t058). 배율은 분석을 건드리지 않고 융합에서만 소리 가중을 줄인다.
기본 1.0은 현행과 같아야 한다. 모델·DB 없음.
"""
from __future__ import annotations

import pytest

from src.retrieval import search_router as sr
from src.retrieval.explain import ExplainRecorder
from tests.test_search_explain import _analysis, _router, _run, _track

AUDIO_HEAVY = {"text": 0.3, "image": 0.0, "audio": 0.7}


def _search(monkeypatch, scale: str | None):
    if scale is None:
        monkeypatch.delenv("AUDIO_FUSION_SCALE", raising=False)
    else:
        monkeypatch.setenv("AUDIO_FUSION_SCALE", scale)
    text_hits = [_track(f"t{i}", title=f"텍스트{i}") for i in range(1, 61)]
    audio_ids = [f"a{i}" for i in range(1, 31)]      # 텍스트와 겹치지 않는다
    rec = ExplainRecorder("q")
    analysis = _analysis(audio_english_query="a reggae groove with offbeat guitar", modality_weights=AUDIO_HEAVY)
    results = _run(_router(text_hits, audio_ids=audio_ids), analysis, top_k=30, use_rerank=False, recorder=rec)
    return [t.id for t in results], rec.record.modality_weights, analysis


def test_default_keeps_the_current_behaviour(monkeypatch) -> None:
    """배율이 없으면 지금과 같다 — 텍스트 1위가 소리 경로 30곡에 밀려 후보 30 밖이다."""
    ids, weights, _ = _search(monkeypatch, None)
    assert "t1" not in ids
    assert ids[:3] == ["a1", "a2", "a3"]
    assert weights == {"text": 0.3, "image": 0.0, "audio": 0.7}


def test_zero_turns_the_audio_path_off(monkeypatch) -> None:
    ids, weights, analysis = _search(monkeypatch, "0")
    assert ids[:3] == ["t1", "t2", "t3"]
    assert not any(i.startswith("a") for i in ids), "소리 경로를 돌리지 않는다"
    assert weights == {"text": 1.0, "image": 0.0, "audio": 0.0}
    # 분석은 그대로다 — 분석 가중을 읽는 다른 규칙(표지 지배 판정 등)이 따라 바뀌지 않는다.
    assert analysis.modality_weights.model_dump() == AUDIO_HEAVY
    assert analysis.audio_english_query


def test_a_partial_scale_shrinks_only_the_audio_weight(monkeypatch) -> None:
    """0.5면 0.3 : 0.35 → 소리 경로는 여전히 돌고, 텍스트 1위가 후보 30 안으로 돌아온다."""
    ids, weights, _ = _search(monkeypatch, "0.5")
    assert "t1" in ids and "a1" in ids
    assert weights["text"] == pytest.approx(0.3 / 0.65)
    assert weights["audio"] == pytest.approx(0.35 / 0.65)


@pytest.mark.parametrize("raw, expected", [("", 1.0), ("0", 0.0), ("0.5", 0.5), ("-1", 0.0), ("없음", 1.0)])
def test_the_switch_reads_the_environment_safely(monkeypatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("AUDIO_FUSION_SCALE", raw)
    assert sr._audio_fusion_scale() == expected


def test_runinfo_records_the_scale_actually_applied(monkeypatch) -> None:
    from src.retrieval.evaluate_search_accuracy import _ranking_switches

    monkeypatch.delenv("AUDIO_FUSION_SCALE", raising=False)
    assert _ranking_switches(None)["audio_fusion_scale"] == 1.0
    monkeypatch.setenv("AUDIO_FUSION_SCALE", "0")
    assert _ranking_switches(None)["audio_fusion_scale"] == 0.0


# --- 텍스트 경로의 BM25 질의(`TEXT_SPARSE_SOURCE`) — 재 보고 채택하지 않은 실험 스위치 ---------------

class _SparseSpy:
    """텍스트 검색에 넘어온 BM25 질의를 적어 둔다."""

    def __init__(self) -> None:
        self.sparse_queries: list = []

    def search_text(self, query, top_k=30, alpha=0.5, sparse_query=None, **kw):
        self.sparse_queries.append(sparse_query)
        return [_track(f"t{i}") for i in range(1, 4)]

    def track_from_match(self, match):
        return _track(str(match.get("id")))


@pytest.mark.parametrize("raw, expected", [
    (None, "이별 그리움"),                      # 현행: 모델이 뽑은 태그만
    ("tags", "이별 그리움"),
    ("query", "비 오는 날 발라드"),
    ("both", "이별 그리움 비 오는 날 발라드"),
    ("없는값", "이별 그리움"),
])
def test_text_sparse_source_picks_what_bm25_receives(monkeypatch, raw, expected: str) -> None:
    if raw is None:
        monkeypatch.delenv("TEXT_SPARSE_SOURCE", raising=False)
    else:
        monkeypatch.setenv("TEXT_SPARSE_SOURCE", raw)
    spy = _SparseSpy()
    router = _router([])
    router._text_svc = spy
    _run(router, _analysis(korean_tags=["이별", "그리움"]), top_k=3, use_rerank=False)
    assert spy.sparse_queries[0] == expected
