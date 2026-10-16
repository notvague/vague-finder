"""v61 — 텍스트 하이브리드 **호출 단위** 비교: 현행 구현(A)이 돌려준 top-k와 독립 정확 합산(`exact_hybrid`)을 같은 질의 벡터·필터로 대조한다.

A가 기록한 질의 벡터(`calls_vectors.npz`)와 실제 두 조회가 돌려준 id 집합(dense/sparse hit ids)으로
  - top-k의 ID·순서·점수 차이(허용 오차 `TOTAL_TOL`, 동점 묶음 안의 순서는 차이로 세지 않음)
  - 기준 top-k에 들어야 하는 곡이 A에서 어떤 모양으로 빠졋는지: dense 몫 누락 / sparse 몫 누락 / 양쪽 누락(결과에서 사라짐)
를 분류한다. 또 B(평가기 안에서 기준 구현을 끼운 실행)와 C(상한=point 수)의 같은 호출 결과가 오프라인 기준과 같은지 확인한다
— B는 주입이 제대로 됐는지, C는 "실제 대상 수를 반영한 조회"가 이 자료에서 기준과 같은지.

호출 짝 맞추기: 같은 질의의 호출은 경로가 스레드풀에서 동시에 돌아 순서가 바뀌므로, (query_id, top_k, 필터, dense 벡터·sparse 벡터의 해시)로 맞춘다.

출력: `text_calls_diff.csv`(호출별) · `truncation_songs.csv`(누락 곡별) · `text_calls_summary.json`
실행(로컬 복사본을 연다 — 다른 프로세스가 열고 있으면 안 된다): venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/compare_text_calls.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import v61_guard  # noqa: E402

v61_guard.fixed_env()
v61_guard.install()

from qdrant_client import QdrantClient  # noqa: E402

from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

from exact_hybrid import TOTAL_TOL, ExactHybridReference, allowed_point_ids, compare_rankings  # noqa: E402

SETS = ("v06", "v09", "v10")


def _load_calls(arm: str, set_name: str) -> Tuple[List[Dict[str, Any]], Any]:
    folder = HERE / f"cand_{arm}" / f"{set_name}_dev"
    calls = [json.loads(line) for line in (folder / "calls.jsonl").open(encoding="utf-8")]
    vectors = np.load(folder / "calls_vectors.npz")
    return calls, vectors


def _key(call: Dict[str, Any], vectors: Any) -> Tuple:
    seq = call["seq"]
    digest = hashlib.sha256()
    digest.update(vectors[f"call{seq}_dense"].tobytes())
    digest.update(vectors[f"call{seq}_sparse_idx"].tobytes())
    digest.update(vectors[f"call{seq}_sparse_val"].tobytes())
    return (call["query_id"], call["top_k"], call["filter"], digest.hexdigest()[:16])


def main() -> int:
    client = QdrantClient(path=os.environ["QDRANT_PATH"])
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    reference = ExactHybridReference.from_client(client, collection)
    fingerprint = reference.fingerprint()
    print("기준 색인 지문:", fingerprint)

    per_call_rows: List[Dict[str, Any]] = []
    song_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"reference_fingerprint": fingerprint, "sets": {}, "tolerance": {"total": TOTAL_TOL}}
    filter_cache: Dict[str, Set[int]] = {}
    grand = Counter()

    for set_name in SETS:
        calls_a, vec_a = _load_calls("A", set_name)
        calls_b, vec_b = _load_calls("B", set_name)
        try:
            calls_c, vec_c = _load_calls("C", set_name)
        except FileNotFoundError:
            calls_c, vec_c = [], None
        by_key_b = {_key(c, vec_b): c for c in calls_b}
        by_key_c = {_key(c, vec_c): c for c in calls_c} if calls_c else {}
        fp_a = {k for k in by_key_b if k not in {_key(c, vec_a) for c in calls_a}}
        counts = Counter()
        for call in calls_a:
            key = _key(call, vec_a)
            seq = call["seq"]
            dense_q = vec_a[f"call{seq}_dense"].astype(np.float64)
            sparse_q = {"indices": vec_a[f"call{seq}_sparse_idx"].tolist(), "values": vec_a[f"call{seq}_sparse_val"].astype(np.float64).tolist()}
            allowed: Optional[Set[int]] = None
            if call["filter"]:
                allowed = filter_cache.get(call["filter"])
                if allowed is None:
                    allowed = allowed_point_ids(client, collection, qb.translate_filter(json.loads(call["filter"])))
                    filter_cache[call["filter"]] = allowed
            table = reference.score(dense_q, sparse_q, allowed_point_ids=allowed)
            exact_top = table.ranked(call["top_k"])
            actual = [{"id": i, "score": s} for i, s in zip(call["result_ids"], call["result_scores"])]
            cmp = compare_rankings(actual, exact_top, tol=TOTAL_TOL)
            # 실제 두 조회가 돌려준 id 집합(A 기록) — 누락 모양 분류
            dense_ids = set(vec_a[f"call{seq}_dense_hit_ids"].tolist()) if f"call{seq}_dense_hit_ids" in vec_a else None
            sparse_ids = set(vec_a[f"call{seq}_sparse_hit_ids"].tolist()) if f"call{seq}_sparse_hit_ids" in vec_a else None
            ranks = table.ranks()
            actual_rank = {sid: i + 1 for i, sid in enumerate(call["result_ids"])}
            exact_by_id = {item["id"]: item for item in exact_top}
            shapes = Counter()
            affected = [item["id"] for item in exact_top if item["id"] in cmp["missing"] or any(m["id"] == item["id"] for m in cmp["score_mismatch"])]
            for sid in affected:
                item = exact_by_id[sid]
                pid = item["point_id"]
                in_dense = (pid in dense_ids) if dense_ids is not None else None
                in_sparse = (pid in sparse_ids) if sparse_ids is not None else None
                if item["sparse_matched"]:
                    if in_dense is False and in_sparse is False:
                        shape = "both_missing_absent"
                    elif in_dense is False:
                        shape = "dense_portion_missing"
                    elif in_sparse is False:
                        shape = "sparse_portion_missing"
                    else:
                        shape = "present_but_score_differs"
                else:
                    shape = "dense_only_song_absent" if in_dense is False else "present_but_score_differs"
                shapes[shape] += 1
                song_rows.append({
                    "set": set_name, "query_id": call["query_id"], "call_seq": seq, "top_k": call["top_k"], "filtered": int(bool(call["filter"])),
                    "song_id": sid, "shape": shape, "exact_rank": ranks[sid].get("total_rank"), "exact_score": round(item["score"], 6),
                    "exact_dense": round(item["dense"], 6), "exact_sparse": round(item["sparse"], 6),
                    "dense_rank_in_corpus": ranks[sid].get("dense_rank"), "sparse_rank_among_matched": ranks[sid].get("sparse_rank", ""),
                    "impl_rank": actual_rank.get(sid, ""), "impl_score": next((s for i, s in zip(call["result_ids"], call["result_scores"]) if i == sid), ""),
                })
            # B·C 같은 호출의 결과가 오프라인 기준과 같은가
            b_call = by_key_b.get(key)
            c_call = by_key_c.get(key)
            b_cmp = compare_rankings([{"id": i, "score": s} for i, s in zip(b_call["result_ids"], b_call["result_scores"])], exact_top, tol=TOTAL_TOL) if b_call else None
            c_cmp = compare_rankings([{"id": i, "score": s} for i, s in zip(c_call["result_ids"], c_call["result_scores"])], exact_top, tol=TOTAL_TOL) if c_call else None
            dense_truncated = (call.get("dense_hits") or 0) >= (call.get("dense_limit") or 0) and (len(table.point_ids) > (call.get("dense_limit") or 0))
            sparse_truncated = (call.get("sparse_hits") or 0) >= (call.get("sparse_limit") or 0) and int(table.sparse_matched.sum()) > (call.get("sparse_limit") or 0)
            row = {
                "set": set_name, "query_id": call["query_id"], "call_seq": seq, "top_k": call["top_k"], "filtered": int(bool(call["filter"])),
                "candidates_after_filter": len(table.point_ids), "sparse_matched_songs": int(table.sparse_matched.sum()),
                "dense_hits": call.get("dense_hits"), "sparse_hits": call.get("sparse_hits"),
                "dense_truncated": int(dense_truncated), "sparse_truncated": int(sparse_truncated),
                "dense_score_min_returned": call.get("dense_score_min"), "exact_dense_min_all": round(float(table.dense.min()), 6),
                "exact_dense_negative_songs": int((table.dense < 0).sum()),
                "A_identical_ids": int(cmp["identical_ids"]), "A_same_set": int(cmp["same_set"]),
                "A_missing_from_topk": len(cmp["missing"]), "A_score_mismatch": len(cmp["score_mismatch"]),
                "A_order_differs_beyond_ties": int(cmp["order_differs_beyond_ties"]), "A_max_abs_score_delta": round(cmp["max_abs_score_delta_common"], 9),
                "shapes": json.dumps(dict(shapes), ensure_ascii=False) if shapes else "",
                "B_matches_exact": int(bool(b_cmp) and not b_cmp["missing"] and not b_cmp["extra"] and not b_cmp["score_mismatch"] and not b_cmp["order_differs_beyond_ties"]) if b_cmp else "",
                "C_matches_exact": int(bool(c_cmp) and not c_cmp["missing"] and not c_cmp["extra"] and not c_cmp["score_mismatch"] and not c_cmp["order_differs_beyond_ties"]) if c_cmp else "",
                "A_ms": call["elapsed_ms"], "B_ms": b_call["elapsed_ms"] if b_call else "", "C_ms": c_call["elapsed_ms"] if c_call else "",
            }
            per_call_rows.append(row)
            counts["calls"] += 1
            counts["dense_truncated"] += int(dense_truncated)
            counts["sparse_truncated"] += int(sparse_truncated)
            counts["filtered"] += int(bool(call["filter"]))
            counts["A_identical"] += int(cmp["identical_ids"])
            counts["A_same_set"] += int(cmp["same_set"])
            counts["A_any_diff"] += int(not (cmp["same_set"] and not cmp["score_mismatch"] and not cmp["order_differs_beyond_ties"]))
            counts["A_missing_songs"] += len(cmp["missing"])
            counts["A_score_mismatch_songs"] += len(cmp["score_mismatch"])
            counts["B_matches_exact"] += int(row["B_matches_exact"] == 1)
            counts["B_compared"] += int(b_cmp is not None)
            counts["C_matches_exact"] += int(row["C_matches_exact"] == 1)
            counts["C_compared"] += int(c_cmp is not None)
            counts["exact_dense_negative_songs"] += row["exact_dense_negative_songs"]
            for shape, n in shapes.items():
                counts[f"shape:{shape}"] += n
        counts["B_calls_without_A_match"] = len(fp_a)
        summary["sets"][set_name] = dict(counts)
        grand.update(counts)
        print(set_name, dict(counts))

    summary["all"] = dict(grand)
    queries_affected = sorted({(r["set"], r["query_id"]) for r in per_call_rows if not (r["A_same_set"] and r["A_score_mismatch"] == 0 and not r["A_order_differs_beyond_ties"])})
    summary["queries_with_any_call_diff"] = [f"{s}:{q}" for s, q in queries_affected]
    with (HERE / "text_calls_diff.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_call_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_call_rows)
    with (HERE / "truncation_songs.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["set", "query_id", "call_seq", "top_k", "filtered", "song_id", "shape", "exact_rank", "exact_score", "exact_dense", "exact_sparse",
                  "dense_rank_in_corpus", "sparse_rank_among_matched", "impl_rank", "impl_score"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(song_rows)
    (HERE / "text_calls_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n전체:", json.dumps(summary["all"], ensure_ascii=False))
    print("호출 차이가 있는 질의:", len(queries_affected))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
