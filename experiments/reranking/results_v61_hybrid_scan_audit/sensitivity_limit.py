"""v61 — 상한 민감도(오프라인 모사). "지금 dev에서 차이가 거의 없다"가 "상한이 안전하다"는 뜻이 아님을 수치로 보인다.

기록된 A 호출의 질의 벡터(필터 없는 호출)에 대해 독립 기준 표(`exact_hybrid`)의 **순위만으로** 상한 L의 효과를 모사한다:
  dense 몫은 dense 순위 ≤ L일 때만, sparse 몫은 (일치한 곡 중) sparse 순위 ≤ L일 때만 남고, 둘 다 밖이면 곡이 사라진다.
이것은 구현을 다시 쓴 것이 아니라 "상위 L개만 받는다"는 조회 계약의 모사다(경계 동점은 무시). L=5000에서 모사가 실제 A 결과와
같은 호출에서 같은 차이를 내는지 먼저 확인하고(`validated_at_5000`), 그다음 코퍼스 대비 상한 비율을 낮춰(= 코퍼스가 늘고
상한은 그대로인 상황) top-30 · top-90에서 달라지는 호출·사라지는 곡 수를 센다.

실행: venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/sensitivity_limit.py   → sensitivity_limit.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from exact_hybrid import TOTAL_TOL, ExactHybridReference, compare_rankings  # noqa: E402

SETS = ("v06", "v09", "v10")
LIMITS = (5143, 5000, 4500, 4000, 3500, 3000, 2500)


def emulate(table, limit: int, top_k: int) -> List[Dict[str, Any]]:
    n = len(table.point_ids)
    dense_order = np.lexsort((table.point_ids, -table.dense))
    dense_rank = np.empty(n, dtype=np.int64)
    dense_rank[dense_order] = np.arange(1, n + 1)
    matched = np.flatnonzero(table.sparse_matched)
    sparse_rank = np.full(n, n + 1, dtype=np.int64)
    if len(matched):
        order = matched[np.lexsort((table.point_ids[matched], -table.sparse[matched]))]
        sparse_rank[order] = np.arange(1, len(order) + 1)
    keep_dense = dense_rank <= limit
    keep_sparse = table.sparse_matched & (sparse_rank <= limit)
    present = keep_dense | keep_sparse
    score = np.where(keep_dense, table.dense, 0.0) + np.where(keep_sparse, table.sparse, 0.0)
    rows = np.flatnonzero(present)
    order = rows[np.lexsort((table.point_ids[rows], -score[rows]))][:top_k]
    return [{"id": table.song_ids[i], "score": float(score[i])} for i in order]


def main() -> int:
    client = QdrantClient(path=os.environ["QDRANT_PATH"])
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    reference = ExactHybridReference.from_client(client, collection)
    n_points = len(reference.point_ids)
    out: Dict[str, Any] = {"points": n_points, "limits": {}, "validated_at_5000": None, "tolerance": TOTAL_TOL}
    tables = []
    actual_at_5000 = []
    for set_name in SETS:
        folder = HERE / "cand_A" / f"{set_name}_dev"
        calls = [json.loads(line) for line in (folder / "calls.jsonl").open(encoding="utf-8")]
        vectors = np.load(folder / "calls_vectors.npz")
        for call in calls:
            if call["filter"]:
                continue
            seq = call["seq"]
            table = reference.score(vectors[f"call{seq}_dense"].astype(np.float64),
                                    {"indices": vectors[f"call{seq}_sparse_idx"].tolist(), "values": vectors[f"call{seq}_sparse_val"].astype(np.float64).tolist()})
            tables.append((set_name, call["query_id"], call["top_k"], table))
            actual_at_5000.append([{"id": i, "score": s} for i, s in zip(call["result_ids"], call["result_scores"])])
    out["calls"] = len(tables)
    # 모사 검증: L=5000 모사 vs 실제 A 결과
    agree = 0
    for (set_name, qid, top_k, table), actual in zip(tables, actual_at_5000):
        em = emulate(table, 5000, top_k)
        cmp = compare_rankings(actual, em, tol=TOTAL_TOL)
        agree += int(cmp["same_set"] and not cmp["score_mismatch"])
    out["validated_at_5000"] = {"calls": len(tables), "emulation_matches_actual_A": agree}
    for limit in LIMITS:
        stats = {"ratio_limit_over_points": round(limit / n_points, 3), "calls_top30_changed_set_or_score": 0, "calls_top30_missing_song": 0,
                 "songs_missing_from_top30": 0, "calls_top90_changed_set_or_score": 0, "songs_missing_from_top90": 0, "calls_top30_order_changed": 0}
        for set_name, qid, top_k, table in tables:
            for k, label in ((30, "top30"), (90, "top90")):
                exact = table.ranked(k)
                em = emulate(table, limit, k)
                cmp = compare_rankings(em, exact, tol=TOTAL_TOL)
                changed = bool(cmp["missing"] or cmp["score_mismatch"])
                stats[f"calls_{label}_changed_set_or_score"] += int(changed)
                stats[f"songs_missing_from_{label}"] += len(cmp["missing"])
                if k == 30:
                    stats["calls_top30_missing_song"] += int(bool(cmp["missing"]))
                    stats["calls_top30_order_changed"] += int([i["id"] for i in em] != [i["id"] for i in exact] and not cmp["missing"])
        out["limits"][str(limit)] = stats
        print(limit, stats)
    (HERE / "sensitivity_limit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("validated_at_5000:", out["validated_at_5000"])
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
