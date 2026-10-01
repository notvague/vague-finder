"""허용 정답(allowed) — 이중 정답 구조의 확장 지표.

여기서 고정하는 것
1. 엄격 지표는 원래 타깃(positives)만, 확장 지표는 positives + allowed를 정답으로 센다.
2. 허용 정답은 원래 타깃·함정(negatives)과 겹칠 수 없다.
3. 기준 세트 CSV(v06)에는 allowed_ids 열이 없다 — 허용 정답은 --with-allowed로만 내보낸다.
"""
from __future__ import annotations

import pytest

from src.eval.export_csv import ALLOWED_COLUMN, COLUMNS, to_rows
from pathlib import Path

from src.eval.relaxed_metrics import evaluate, scores, tag_for
from src.eval.schema import EvalQuery

TOP = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def _query(**kw) -> EvalQuery:
    base = dict(query_id="x1", query="질의", category="mixed", split="dev", query_set="v04", positives=["h"])
    base.update(kw)
    return EvalQuery(**base)


def test_scores_follow_first_hit() -> None:
    s = scores(TOP, {"c"})
    assert (s["hit@1"], s["hit@3"], s["hit@5"], s["hit@10"]) == (0.0, 1.0, 1.0, 1.0)
    assert s["mrr@10"] == pytest.approx(1 / 3)
    assert scores(TOP, {"zz"})["mrr@10"] == 0.0


def test_relaxed_counts_allowed_but_strict_does_not() -> None:
    detail = [{"query_id": "x1", "relevant_ids": "h", "baseline_top_ids": "|".join(TOP), "rerank_top_ids": "|".join(TOP)}]
    labels = {"x1": {"relevant_ids": "h", "allowed_ids": "b|zz"}}
    per_query, summary = evaluate(detail, labels)
    q = per_query[0]
    assert (q["rerank_strict_rank"], q["rerank_relaxed_rank"]) == (8, 2)
    mrr = next(r for r in summary if r["metric"] == "MRR@10")
    assert mrr["rerank_strict"] == pytest.approx(1 / 8, abs=1e-6)
    assert mrr["rerank_relaxed"] == pytest.approx(1 / 2, abs=1e-6)


def test_label_mismatch_is_an_error() -> None:
    """detail을 잰 라벨과 다른 라벨로 재집계하면 숫자가 조용히 틀어진다."""
    detail = [{"query_id": "x1", "relevant_ids": "h", "baseline_top_ids": "a", "rerank_top_ids": "a"}]
    with pytest.raises(ValueError):
        evaluate(detail, {"x1": {"relevant_ids": "a", "allowed_ids": ""}})


def test_allowed_cannot_overlap_positives_or_negatives() -> None:
    with pytest.raises(ValueError):
        _query(allowed=["h"])
    with pytest.raises(ValueError):
        _query(allowed=["n"], negatives=["n"], negative_reason="함정")
    assert _query(allowed=["b"]).allowed == ["b"]


def test_default_export_has_no_allowed_column() -> None:
    q = _query(allowed=["b", "c"])
    assert ALLOWED_COLUMN not in COLUMNS
    assert ALLOWED_COLUMN not in to_rows([q])[0]
    assert to_rows([q], with_allowed=True)[0][ALLOWED_COLUMN] == "b|c"


def test_result_files_carry_the_label_version() -> None:
    """라벨 버전이 바뀌어도 이전 버전 결과 파일을 덮어쓰지 않는다."""
    assert tag_for(Path("experiments/reranking/eval_queries_v08.csv")) == "relaxed_v08"
    assert tag_for(Path("experiments/reranking/eval_queries_v07.csv")) == "relaxed_v07"
    assert tag_for(Path("labels_custom.csv")) == "relaxed_labels_custom"
