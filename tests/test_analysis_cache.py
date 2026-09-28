"""질의 분석 캐시 — 측정에서 분석 단계를 고정한다.

이 캐시가 지키는 것은 하나다: **측정은 저장된 분석만 읽는다.** 한 질의라도 측정 중에
새로 분석되면 그 질의만 다른 조건으로 측정되므로, 누락·원문 불일치·규칙 폴백은
모두 시작 전에 멈춰야 한다.
"""

from __future__ import annotations

import asyncio
import csv
import json

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval import evaluate_search_accuracy as ev
from src.retrieval import query_analyzer as qa
from src.retrieval.analysis_cache import (
    CACHE_FORMAT,
    AnalysisCacheError,
    analyzer_fingerprint,
    load_cache,
    looks_like_fallback,
    new_cache,
)
from src.retrieval.explain import RERANK_APPLIED, RerankRun
from src.retrieval.reranker import RerankerConfig


def _analysis(query: str = "비 오는 날 발라드", **kwargs) -> QueryAnalysis:
    base = dict(
        original_query=query,
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
    )
    base.update(kwargs)
    return QueryAnalysis(**base)


# ---------------------------------------------------------------------------
# 캐시 자체
# ---------------------------------------------------------------------------

def test_cache_roundtrip_preserves_the_analysis(tmp_path):
    cache = new_cache(source="queries.csv", split="dev")
    cache.put("q1", "비 오는 날 발라드", _analysis(korean_tags=["비", "발라드"]))
    path = tmp_path / "cache.json"
    cache.save(path)

    loaded = load_cache(path)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == CACHE_FORMAT
    got = loaded.get("q1", "비 오는 날 발라드")
    assert got.korean_tags == ["비", "발라드"]
    assert loaded.meta["split"] == "dev"
    assert loaded.meta["analyzer"]["temperature"] == 0.0


def test_cache_rejects_a_changed_query_text(tmp_path):
    """질문이 바뀐 것을 모르고 옛 분석으로 측정하는 것이 가장 나쁘다."""
    cache = new_cache(source="queries.csv", split="dev")
    cache.put("q1", "원래 질의", _analysis("원래 질의"))
    cache.save(tmp_path / "c.json")

    loaded = load_cache(tmp_path / "c.json")
    with pytest.raises(AnalysisCacheError) as exc:
        loaded.get("q1", "바뀐 질의")
    assert "원문이 캐시와 다릅니다" in str(exc.value)


def test_cache_reports_missing_queries():
    cache = new_cache(source="q.csv", split="dev")
    cache.put("q1", "질의1", _analysis("질의1"))
    assert cache.missing(["q1", "q2", "q3"]) == ["q2", "q3"]


def test_cache_marks_a_rule_fallback_separately():
    """API 장애로 나온 규칙 폴백은 분석 결과가 아니다. 구분해서 담는다."""
    fallback = qa._fallback("비 오는 날 노래")
    assert looks_like_fallback(fallback) is True
    assert looks_like_fallback(_analysis(confidence=0.9)) is False

    cache = new_cache(source="q.csv", split="dev")
    cache.put("q1", "비 오는 날 노래", fallback)
    cache.put("q2", "다른 질의", _analysis("다른 질의", confidence=0.9))
    assert cache.fallback_ids() == ["q1"]
    assert cache.fallback_ids(["q2"]) == []


def test_low_confidence_model_answer_is_not_called_a_fallback():
    """확신도만 보면 모델의 낮은 확신도 분석까지 폴백으로 몰게 된다."""
    low = _analysis(confidence=0.0, korean_tags=["모델이 준 태그"])
    assert looks_like_fallback(low) is False


def test_cache_detects_a_changed_analysis_condition(tmp_path):
    """캐시를 만든 조건이 달라졌으면 사람이 판단할 수 있게 알려야 한다."""
    cache = new_cache(source="q.csv", split="dev")
    cache.put("q1", "질의", _analysis("질의"))
    cache.meta["analyzer"]["prompt_sha"] = "deadbeef0000"
    cache.save(tmp_path / "c.json")

    drift = load_cache(tmp_path / "c.json").fingerprint_drift()
    assert drift == ["prompt_sha"]
    assert analyzer_fingerprint()["prompt_sha"] != "deadbeef0000"


def test_cache_rejects_an_unknown_format(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"format": 99, "meta": {}, "entries": {}}), encoding="utf-8")
    with pytest.raises(AnalysisCacheError):
        load_cache(path)


# ---------------------------------------------------------------------------
# 평가 스크립트와의 결합
# ---------------------------------------------------------------------------

def _queries_csv(path, rows):
    path.write_text(
        "query_id,split,query_type,query,relevant_ids\n"
        + "".join(f"{qid},dev,search,{query},s1\n" for qid, query in rows),
        encoding="utf-8",
    )


class _Router:
    async def search(self, analysis, **kw):
        return [
            MatchingTrack(id=f"s{i}", score=1.0 / i, title=f"곡{i}") for i in range(1, 4)
        ]


class _Reranker:
    enabled = True
    uses_clarify_answers = False
    config = RerankerConfig()

    def load(self):
        return self

    def rerank_run(self, query, tracks, top_k):
        out = [t.model_copy(update={"rerank_score": 0.9}) for t in reversed(list(tracks))]
        return RerankRun(out[:top_k], RERANK_APPLIED, [t.id for t in tracks])

    def rerank(self, query, tracks, top_k):
        return self.rerank_run(query, tracks, top_k).tracks


class _ExplodingAnalyzer:
    """측정 중에 불리면 안 되는 분석기."""

    def analyze(self, query):
        raise AssertionError("캐시를 쓰는 측정이 분석기를 불렀다")


def _wire(monkeypatch, analyzer=None):
    monkeypatch.setattr(ev, "get_search_router", lambda: _Router())
    monkeypatch.setattr(ev, "get_reranker", lambda: _Reranker())
    monkeypatch.setattr(
        ev, "get_query_analyzer", lambda: analyzer or _ExplodingAnalyzer()
    )
    # 코퍼스 규모 조회는 실제 벡터 DB를 열므로 테스트에서는 막는다.
    monkeypatch.setattr(ev, "_count_points", lambda vsettings: 952)


def _args(tmp_path, csv_path, cache_path, extra=()):
    return ev.build_parser().parse_args(
        [
            "--input", str(csv_path),
            "--output-dir", str(tmp_path / "out"),
            "--split", "dev",
            "--top-k", "3",
            "--candidate-k", "3",
            "--analysis-cache", str(cache_path),
            *extra,
        ]
    )


def test_eval_refuses_to_start_when_a_query_is_missing(tmp_path, monkeypatch):
    """누락 질의를 측정 중에 새로 분석하면 그 질의만 다른 조건으로 측정된다."""
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의"), ("q2", "둘째 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", _analysis("첫 질의"))
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    with pytest.raises(AnalysisCacheError) as exc:
        asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))
    assert "q2" in str(exc.value)
    assert not (tmp_path / "out" / "search_eval_dev_detail.csv").exists()


def test_eval_refuses_a_rule_fallback_unless_allowed(tmp_path, monkeypatch):
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", qa._fallback("첫 질의"))
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    with pytest.raises(AnalysisCacheError) as exc:
        asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))
    assert "폴백" in str(exc.value)

    # 의도한 경우에는 통과시킨다 — 다만 명시해야 한다.
    asyncio.run(
        ev.evaluate(
            _args(tmp_path, csv_path, tmp_path / "c.json", ["--allow-fallback-analysis"])
        )
    )
    assert (tmp_path / "out" / "search_eval_dev_detail.csv").exists()


def test_eval_with_a_cache_never_calls_the_analyzer(tmp_path, monkeypatch):
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의"), ("q2", "둘째 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", _analysis("첫 질의"))
    cache.put("q2", "둘째 질의", _analysis("둘째 질의"))
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)   # 분석기는 불리면 AssertionError

    asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))

    detail = list(
        csv.DictReader(
            (tmp_path / "out" / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")
        )
    )
    assert [row["query_id"] for row in detail] == ["q1", "q2"]


def test_eval_records_what_the_measurement_ran_against(tmp_path, monkeypatch):
    """숫자만 남기면 905곡 측정과 952곡 측정을 구분할 수 없다."""
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", _analysis("첫 질의"))
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))

    info = json.loads(
        (tmp_path / "out" / "search_eval_dev_runinfo.json").read_text(encoding="utf-8")
    )
    assert info["analysis"]["mode"] == "cache"
    assert info["analysis"]["meta"]["analyzer"]["prompt_sha"]
    assert info["corpus"]["point_count"] == 952
    assert info["reranker"]["class"] == "_Reranker"
    assert info["reranker"]["config"]["rerank_weight"] == pytest.approx(
        RerankerConfig().rerank_weight
    )
    assert info["reranker"]["config"]["spread_ref"] == pytest.approx(
        RerankerConfig().spread_ref
    )
    assert "sha256_12" in info["bm25"] or info["bm25"]["exists"] is False


def test_eval_validates_every_cache_entry_before_touching_anything(tmp_path, monkeypatch):
    """원문 불일치를 **시작 전에** 잡아야 한다.

    루프 안에서 처음 만나면 모델·DB를 올리고 파일을 쓴 뒤 중단된다 — 첫 질의 결과만
    남은 결과 폴더가 생기고, 그것이 정상 산출물처럼 보인다.
    """
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의"), ("q2", "바뀐 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", _analysis("첫 질의"))
    cache.put("q2", "옛 질의", _analysis("옛 질의"))       # 원문이 바뀌었다
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    with pytest.raises(AnalysisCacheError) as exc:
        asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))
    assert "q2" in str(exc.value)
    assert "원문이 캐시와 다릅니다" in str(exc.value)

    out = tmp_path / "out"
    assert not (out / "search_eval_dev_detail.csv").exists(), "결과 파일이 남았다"
    assert not (out / "search_eval_dev_runinfo.json").exists()


def test_eval_reports_every_bad_entry_at_once(tmp_path, monkeypatch):
    """한 건씩 고치게 하면 53건짜리 캐시에서 53번 다시 돌려야 한다."""
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "새 질의1"), ("q2", "새 질의2")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "옛 질의1", _analysis("옛 질의1"))
    cache.put("q2", "옛 질의2", _analysis("옛 질의2"))
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    with pytest.raises(AnalysisCacheError) as exc:
        asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))
    message = str(exc.value)
    assert "2건" in message
    assert "q1" in message and "q2" in message


def test_eval_rejects_a_cache_entry_with_a_broken_schema(tmp_path, monkeypatch):
    csv_path = tmp_path / "q.csv"
    _queries_csv(csv_path, [("q1", "첫 질의")])
    cache = new_cache(source=str(csv_path), split="dev")
    cache.put("q1", "첫 질의", _analysis("첫 질의"))
    cache.entries["q1"]["analysis"]["intent_type"] = "존재하지_않는_값"
    cache.save(tmp_path / "c.json")
    _wire(monkeypatch)

    with pytest.raises(AnalysisCacheError) as exc:
        asyncio.run(ev.evaluate(_args(tmp_path, csv_path, tmp_path / "c.json")))
    assert "q1" in str(exc.value)
    assert not (tmp_path / "out" / "search_eval_dev_detail.csv").exists()


def test_split_cannot_be_omitted():
    """생략 시 전체를 돌리던 예전 동작에서는 dev만 재려던 실행이 test까지 측정했다."""
    with pytest.raises(SystemExit):
        ev.build_parser().parse_args(["--input", "x.csv"])
    assert ev.build_parser().parse_args(["--split", "all"]).split == "all"


def test_default_input_is_the_reference_set():
    """기본값이 3행 스모크 파일이면 3질의 측정을 'dev 재측정'으로 읽게 된다."""
    args = ev.build_parser().parse_args(["--split", "dev"])
    assert args.input.endswith("eval_queries_v05.csv")


# ---------------------------------------------------------------------------
# 항목별 생성 이력 — **중단돼도 남아야 한다**
#
# 조건이 어긋난 캐시를 이어서 채우면 한 파일 안에 서로 다른 분석기가 만든 항목이
# 섞인다. 그 사실을 작업이 **끝나야** 기록하면, 한 건을 저장한 직후 끊겼을 때
# 그 항목만 이력에서 빠진다(실제로 재현했다). 항목과 같은 저장에 들어가야 한다.
# ---------------------------------------------------------------------------

import argparse

from src.retrieval import build_analysis_cache as bc
from src.retrieval.analysis_cache import fingerprint_sha, load_cache, new_cache


class _Interrupted(Exception):
    """중단을 흉내 낸다. 사용자가 Ctrl+C를 누른 자리다."""


class _CountingAnalyzer:
    def __init__(self, die_on=None):
        self.calls = 0
        self.die_on = die_on

    def analyze(self, query):
        self.calls += 1
        if self.die_on and self.calls == self.die_on:
            raise _Interrupted()
        return _analysis(query, confidence=0.9)


def _drifted_cache(path, csv_path):
    """조건이 어긋난 기존 캐시. 항목 하나가 옛 분석기로 만들어져 있다."""
    cache = new_cache(source=str(csv_path), split="dev")
    cache.meta["analyzer"]["postprocess_sha"] = "옛날해시"
    cache.put("q0", "옛 질의", _analysis("옛 질의"))
    cache.entries["q0"].pop("analyzer_sha")  # 항목별 기록이 없던 시절의 캐시
    cache.save(path)


def _build_args(csv_path, cache_path):
    return argparse.Namespace(
        input=str(csv_path), output=str(cache_path), split="dev", query_ids="",
        force=False, keep_fallback=False, allow_floating_year=True,
        reuse_despite_drift=True, drift_reason="시험",
    )


def test_every_new_entry_records_which_analyzer_made_it(tmp_path, monkeypatch):
    csv_path, cache_path = tmp_path / "q.csv", tmp_path / "c.json"
    _queries_csv(csv_path, [("q1", "질의 하나"), ("q2", "질의 둘")])
    _drifted_cache(cache_path, csv_path)
    monkeypatch.setattr(bc, "get_query_analyzer", lambda: _CountingAnalyzer())

    bc.build(_build_args(csv_path, cache_path))

    cache = load_cache(cache_path)
    assert cache.entries["q1"]["analyzer_sha"] == cache.entries["q2"]["analyzer_sha"]
    assert cache.entries["q1"]["analyzer_sha"] != "(meta)"
    # 옛 항목은 지문이 없다 — meta가 만든 것이다
    assert "analyzer_sha" not in cache.entries["q0"]


def test_provenance_survives_an_interrupted_build(tmp_path, monkeypatch):
    """한 건을 저장한 직후 끊고 재개해도 **두 건 모두** 이력이 남아야 한다."""
    csv_path, cache_path = tmp_path / "q.csv", tmp_path / "c.json"
    _queries_csv(csv_path, [("q1", "질의 하나"), ("q2", "질의 둘")])
    _drifted_cache(cache_path, csv_path)
    args = _build_args(csv_path, cache_path)

    monkeypatch.setattr(bc, "get_query_analyzer", lambda: _CountingAnalyzer(die_on=2))
    with pytest.raises(_Interrupted):
        bc.build(args)
    assert "q1" in load_cache(cache_path).entries, "한 건씩 저장하지 않았다"

    monkeypatch.setattr(bc, "get_query_analyzer", lambda: _CountingAnalyzer())
    bc.build(args)

    cache = load_cache(cache_path)
    missing = [q for q in ("q1", "q2") if not cache.entries[q].get("analyzer_sha")]
    assert not missing, f"중단 때문에 이력이 빠진 항목: {missing}"
    groups = cache.provenance()
    assert groups["(meta)"] == ["q0"]
    assert sorted(v for ids in groups.values() for v in ids) == ["q0", "q1", "q2"]


def test_provenance_groups_entries_by_analyzer():
    cache = new_cache(source="x", split="dev")
    cache.put("a", "가", _analysis("가"))
    cache.put("b", "나", _analysis("나"))
    cache.entries["old"] = {"query": "다", "fallback": False, "analyzed_at": "-",
                            "analysis": _analysis("다").model_dump(mode="json")}
    groups = cache.provenance()
    assert groups["(meta)"] == ["old"]
    assert len(groups) == 2


def test_provenance_reads_the_legacy_drift_note(tmp_path):
    """항목별 기록을 넣기 **전에** 만든 캐시도 바르게 묶여야 한다.

    v19 캐시가 이 모양이다 — 조건이 어긋난 뒤 추가된 항목이 `drift_notes`에만
    적혀 있다. 그것을 읽지 않으면 다른 분석기가 만든 항목까지 meta 묶음으로
    들어간다(실제로 dev 4건·test 2건이 그렇게 묶였다).
    """
    cache = new_cache(source="x", split="dev")
    other = dict(cache.meta["analyzer"], postprocess_sha="다른해시")
    cache.meta["drift_notes"] = [{
        "drifted": ["postprocess_sha"],
        "analyzer_now": other,
        "added_after_drift": ["새1", "새2"],
    }]
    for qid in ("옛1", "새1", "새2"):
        cache.put(qid, qid, _analysis(qid))
        cache.entries[qid].pop("analyzer_sha")  # 항목별 기록이 없던 시절

    groups = cache.provenance()
    assert groups["(meta)"] == ["옛1"]
    assert groups[fingerprint_sha(other)] == ["새1", "새2"]


def test_entry_stamp_wins_over_the_legacy_note(tmp_path):
    """항목에 지문이 붙어 있으면 그쪽이 먼저다 — 옛 기록보다 정확하다."""
    cache = new_cache(source="x", split="dev")
    cache.meta["drift_notes"] = [{
        "analyzer_sha_now": "옛기록",
        "added_after_drift": ["q1"],
    }]
    cache.put("q1", "질의", _analysis("질의"))
    stamped = cache.entries["q1"]["analyzer_sha"]

    groups = cache.provenance()
    assert groups[stamped] == ["q1"]
    assert "옛기록" not in groups


def test_a_later_drift_note_overrides_an_earlier_one():
    """같은 질의가 두 번 다시 분석됐으면 **마지막** 조건이 그 항목을 만든 것이다."""
    cache = new_cache(source="x", split="dev")
    cache.meta["drift_notes"] = [
        {"analyzer_sha_now": "먼저", "added_after_drift": ["q1"]},
        {"analyzer_sha_now": "나중", "added_after_drift": ["q1"]},
    ]
    cache.entries["q1"] = {"query": "질의", "fallback": False, "analyzed_at": "-",
                           "analysis": _analysis("질의").model_dump(mode="json")}
    assert cache.provenance() == {"나중": ["q1"]}
