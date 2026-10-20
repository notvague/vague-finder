"""후보 밖 정답의 단계 판정 — 실행 기록만 읽어서 나눈다."""
from __future__ import annotations

from src.retrieval.explain import ExplainRecorder
from src.retrieval.trace_candidate_drop import (
    FINAL_CUT,
    FUSION_CUT,
    IN_POOL,
    NO_PATH,
    rank_by,
    stage_of,
)


def _record():
    rec = ExplainRecorder("q")
    for song_id, rank in (("a", 1), ("c", 40), ("d", 2), ("e", 3)):
        rec.path(song_id, "text_hybrid", rank, 1.0 / (60 + rank))
    rec.mark_paths_dropped(["c"])          # RRF 절단에서 빠졌다
    rec.set_fused("a", 0.03)
    rec.set_fused("d", 0.02)
    rec.set_fused("e", 0.01)
    rec.adjust("d", "vocal_gender_match", 0.02)
    rec.adjust("a", "vocal_gender_mismatch", -0.02)
    return rec.record


def test_stages() -> None:
    record = _record()
    pool = ["d", "e"]
    assert stage_of("d", record, pool) == IN_POOL
    assert stage_of("b", record, pool) == NO_PATH
    assert stage_of("c", record, pool) == FUSION_CUT
    assert stage_of("a", record, pool) == FINAL_CUT


def test_rank_before_and_after_adjustments() -> None:
    record = _record()
    assert rank_by(record, "a", boosted=False) == 1
    assert rank_by(record, "d", boosted=True) == 1
    assert rank_by(record, "a", boosted=True) == 3
    assert rank_by(record, "c", boosted=True) is None
