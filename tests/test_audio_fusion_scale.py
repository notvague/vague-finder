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


# --- PR #49 리뷰 반영 ------------------------------------------------------------------------

_SCALE_HELPERS = [
    ("AUDIO_FUSION_SCALE", sr._audio_fusion_scale, 1.0),
    ("BOOST_SCALE", sr._explicit_boost_scale, 1.0),
    ("ATTR_BOOST_SCALE", sr._attr_boost_scale, 1.0),
    ("LYRIC_BOOST_SCALE", sr._lyric_boost_scale, 1.0),
    ("GENDER_MISMATCH_SCALE", sr._gender_mismatch_scale, 0.0),
]


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "1e999", "infinity"])
@pytest.mark.parametrize("name, helper, default", _SCALE_HELPERS)
def test_a_non_finite_scale_falls_back_to_the_default(monkeypatch, name, helper, default, raw: str) -> None:
    """`float()`는 nan·inf를 예외 없이 받는다. 예전에는 nan이 0(스위치 끔)이 되고 inf가 그대로 곱해졌다."""
    monkeypatch.setenv(name, raw)
    assert helper() == default


@pytest.mark.parametrize("name, helper, default", _SCALE_HELPERS)
def test_scale_helpers_keep_their_old_reading_of_ordinary_values(monkeypatch, name, helper, default) -> None:
    for raw, expected in (("", default), ("0", 0.0), ("0.2", 0.2), ("3", 3.0), ("-1", 0.0), ("없음", default)):
        monkeypatch.setenv(name, raw)
        assert helper() == expected, (name, raw)


def test_an_infinite_scale_no_longer_poisons_the_fusion_weights(monkeypatch) -> None:
    """inf가 통과하면 inf/inf = NaN 가중이 된다. 이제 기본값으로 읽혀 현행과 같은 결과다."""
    ids, weights, _ = _search(monkeypatch, "inf")
    assert weights == {"text": 0.3, "image": 0.0, "audio": 0.7}
    assert ids[:3] == ["a1", "a2", "a3"]


def test_a_scale_above_one_raises_the_audio_share(monkeypatch) -> None:
    ids, weights, _ = _search(monkeypatch, "2")
    assert weights["audio"] == pytest.approx(1.4 / 1.7) and weights["text"] == pytest.approx(0.3 / 1.7)
    assert "t1" not in ids


def test_forced_weights_ignore_the_scale(monkeypatch) -> None:
    """가중을 직접 주는 ablation 실행(`force_weights`)은 준 가중 그대로 돈다 — 배율을 0으로 둬도 소리 경로가 돈다."""
    monkeypatch.setenv("AUDIO_FUSION_SCALE", "0")
    text_hits = [_track(f"t{i}", title=f"텍스트{i}") for i in range(1, 61)]
    rec = ExplainRecorder("q")
    analysis = _analysis(audio_english_query="a reggae groove", modality_weights=AUDIO_HEAVY)
    results = _run(
        _router(text_hits, audio_ids=[f"a{i}" for i in range(1, 31)]), analysis,
        top_k=30, use_rerank=False, recorder=rec, force_weights=[0.3, 0.0, 0.7],
    )
    assert [t.id for t in results][:3] == ["a1", "a2", "a3"]
    assert rec.record.modality_weights == {"text": 0.3, "image": 0.0, "audio": 0.7}


def test_the_scale_applies_before_the_lyric_meaning_clamp(monkeypatch) -> None:
    """가사 뜻만 말한 mixed 질의는 text ≥ 0.6 · audio ≤ 0.4로 묶인다. 배율은 그 앞에서 곱해진다."""
    def weights(scale):
        if scale is None:
            monkeypatch.delenv("AUDIO_FUSION_SCALE", raising=False)
        else:
            monkeypatch.setenv("AUDIO_FUSION_SCALE", scale)
        rec = ExplainRecorder("q")
        analysis = _analysis(
            intent_type="mixed", lyric_semantic_query="헤어진 사람을 그리워하는 내용",
            audio_english_query="a slow piano ballad", modality_weights=AUDIO_HEAVY,
        )
        text_hits = [_track(f"t{i}", title=f"텍스트{i}") for i in range(1, 31)]
        _run(_router(text_hits, audio_ids=["a1"]), analysis, top_k=10, use_rerank=False, recorder=rec)
        return rec.record.modality_weights

    assert weights(None) == {"text": 0.6, "image": 0.0, "audio": 0.4}          # 현행: 0.3/0.7 → 묶임
    scaled = weights("0.25")                                                    # 0.7 × 0.25 = 0.175 → 묶임 아래
    assert scaled["text"] == pytest.approx(0.6 / 0.775) and scaled["audio"] == pytest.approx(0.175 / 0.775)
