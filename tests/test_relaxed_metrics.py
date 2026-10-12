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


from src.eval.relaxed_metrics import (  # noqa: E402
    TOP10_COLUMNS,
    load_result_rows,
    scope_summary,
    summarize,
    top10_path,
    write_top10,
)


def _detail(qid: str, top: list, relevant: str = "h") -> dict:
    return {"query_id": qid, "split": "dev", "relevant_ids": relevant, "baseline_top_ids": "|".join(top),
            "rerank_top_ids": "|".join(top), "rerank_top_titles": "제목은 경량본에 남지 않는다"}


def test_top10_file_keeps_only_ids_and_reproduces_the_detail(tmp_path) -> None:
    detail = [_detail("x1", TOP), _detail("x2", TOP[::-1])]
    labels = {"x1": {"relevant_ids": "h", "allowed_ids": "b"}, "x2": {"relevant_ids": "h", "allowed_ids": ""}}
    write_top10(detail, top10_path(tmp_path, "dev"))
    rows, source = load_result_rows(tmp_path, "dev")
    assert source.name == "search_eval_dev_top10.csv"
    assert tuple(rows[0]) == TOP10_COLUMNS          # 제목·설명 열은 없다
    assert evaluate(rows, labels) == evaluate(detail, labels)


def test_detail_is_read_first_when_both_exist(tmp_path) -> None:
    write_top10([_detail("x1", TOP)], top10_path(tmp_path, "dev"))
    assert load_result_rows(tmp_path, "dev")[1].name == "search_eval_dev_top10.csv"
    (tmp_path / "search_eval_dev_detail.csv").write_text("query_id\nx1\n", encoding="utf-8")
    assert load_result_rows(tmp_path, "dev")[1].name == "search_eval_dev_detail.csv"
    assert load_result_rows(tmp_path, "test") is None


def test_scope_summary_adds_groups_without_changing_the_whole() -> None:
    """범주형 집계는 부분 집합을 더할 뿐이다 — all은 원래 요약과 같다."""
    detail = [_detail("s1", TOP), _detail("k1", TOP, relevant="zz"), _detail("k2", TOP, relevant="a")]
    labels = {q["query_id"]: {"relevant_ids": q["relevant_ids"], "allowed_ids": ""} for q in detail}
    scopes = {"s1": "specific", "k1": "categorical", "k2": "categorical"}
    per_query, summary = evaluate(detail, labels, scopes)
    rows = scope_summary(per_query)
    pick = lambda scope, metric: next(r for r in rows if r["scope"] == scope and r["metric"] == metric)  # noqa: E731
    assert [{k: v for k, v in r.items() if k not in ("scope", "n_queries")} for r in rows if r["scope"] == "all"] == summary
    assert pick("categorical", "Hit@10")["n_queries"] == 2
    assert pick("categorical", "Hit@10")["rerank_strict"] == 0.5     # k2만 찾는다
    assert pick("specific", "MRR@10")["rerank_strict"] == pytest.approx(1 / 8, abs=1e-6)


def test_empty_scope_is_blank_not_zero() -> None:
    assert all(r["rerank_strict"] == "" for r in summarize([]))


def test_query_missing_from_queries_json_is_an_error() -> None:
    with pytest.raises(KeyError):
        evaluate([_detail("x1", TOP)], {"x1": {"relevant_ids": "h", "allowed_ids": ""}}, scopes={})
