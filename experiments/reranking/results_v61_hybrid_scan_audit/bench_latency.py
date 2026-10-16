"""v61 — 하이브리드 조회 지연 비교(한 프로세스 · 로컬 Qdrant 복사본 · 임베딩 없음).

A(상한 5000)와 C(상한 = point 수, count는 준비 단계 1회)를 **같은 구현**으로, 기록된 질의 벡터(A 실행의 `calls_vectors.npz`)를 입력으로
번갈아 실행한다. D는 수정안 §7-A의 모양 — **요청마다** `count(exact)`를 부르고 그 값을 상한으로 조회(count 비용 포함). 필터 호출의
`count(filter)` 비용은 A 기록의 서로 다른 필터마다 10회씩 따로 잰다. B(독립 기준, numpy 행렬곱)도 함께 재지만 **운영 수정안의 비용이
아니다** — 벡터를 메모리에 다 올린 다른 방식이다.
- 예열: 각 구성 2회 호출 후 측정 시작. 준비 시간(복사본 열기·기준 행렬 적재)은 따로 적는다
- 순서: 호출마다 A→C→D→B / D→C→A→B 를 교대로(순서 편향 완화). 두 구성을 동시에 돌리지 않는다
- 기록: 호출별 ms, 조회 횟수(dense·sparse 각 1회)와 반환 건수, 구성별 중앙값·p95, 프로세스 ru_maxrss

실행: venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/bench_latency.py [--set v06] [--max-calls 60]
출력: latency_calls.csv · latency_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
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

from exact_hybrid import ExactHybridReference, allowed_point_ids  # noqa: E402


def _rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="v06")
    parser.add_argument("--max-calls", type=int, default=60, help="측정할 호출 수(필터 없는 호출만, 앞에서부터)")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rss0 = _rss()
    client = QdrantClient(path=os.environ["QDRANT_PATH"])
    index = qb.QdrantIndex(client, TEXT_HYBRID_INDEX_NAME, exact=False)
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    point_count = int(client.count(collection, exact=True).count)
    open_s = time.perf_counter() - t0
    rss_open = _rss()
    t1 = time.perf_counter()
    reference = ExactHybridReference.from_client(client, collection)
    ref_build_s = time.perf_counter() - t1
    rss_ref = _rss()

    folder = HERE / "cand_A" / f"{args.set}_dev"
    calls = [json.loads(line) for line in (folder / "calls.jsonl").open(encoding="utf-8")]
    vectors = np.load(folder / "calls_vectors.npz")
    inputs: List[Dict[str, Any]] = []
    for call in calls:
        if call["filter"]:
            continue
        seq = call["seq"]
        inputs.append({
            "seq": seq, "query_id": call["query_id"], "top_k": call["top_k"],
            "dense": vectors[f"call{seq}_dense"].astype(float).tolist(),
            "sparse": {"indices": vectors[f"call{seq}_sparse_idx"].tolist(), "values": vectors[f"call{seq}_sparse_val"].astype(float).tolist()},
        })
        if len(inputs) >= args.max_calls:
            break

    def run_impl(limit: int, item: Dict[str, Any]) -> Dict[str, Any]:
        qb.HYBRID_SCAN_LIMIT = limit
        counts = {"calls": 0, "returned": 0}
        orig = index._client.query_points

        def spy(*a, **kw):
            resp = orig(*a, **kw)
            counts["calls"] += 1
            counts["returned"] += len(resp.points)
            return resp

        index._client.query_points = spy  # type: ignore[assignment]
        try:
            started = time.perf_counter()
            res = index.query(vector=item["dense"], sparse_vector=item["sparse"], top_k=item["top_k"], include_metadata=True, namespace=NAMESPACE)
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            index._client.query_points = orig  # type: ignore[assignment]
        return {"ms": elapsed, "qdrant_calls": counts["calls"], "returned_points": counts["returned"], "n": len(res["matches"])}

    def run_ref(item: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        ranked = reference.top(item["dense"], item["sparse"], item["top_k"])
        payloads = index._payloads(collection, [r["point_id"] for r in ranked])
        elapsed = (time.perf_counter() - started) * 1000
        return {"ms": elapsed, "qdrant_calls": 1, "returned_points": len(payloads), "n": len(ranked)}

    def run_count_then_impl(item: Dict[str, Any]) -> Dict[str, Any]:
        """D: 수정안 §7-A 모양 — 요청마다 count(exact)로 대상 수를 얻고 그 값을 상한으로 조회한다(count 비용 포함)."""
        started = time.perf_counter()
        n = int(client.count(collection, exact=True).count)
        count_ms = (time.perf_counter() - started) * 1000
        out = run_impl(n, item)
        out["ms"] += count_ms
        out["count_ms"] = count_ms
        out["qdrant_calls"] += 1
        return out

    arms = {"A": lambda it: run_impl(5000, it), "C": lambda it: run_impl(point_count, it), "D": run_count_then_impl, "B": run_ref}
    for _ in range(args.warmup):
        for name in ("A", "C", "D", "B"):
            arms[name](inputs[0])

    # 필터 count 비용(수정안이 필터 호출에서 추가로 내는 비용) — A 기록의 필터 호출에서 서로 다른 필터를 모아 각 10회
    filter_counts: List[Dict[str, Any]] = []
    seen_filters = []
    for call in calls:
        if call["filter"] and call["filter"] not in seen_filters:
            seen_filters.append(call["filter"])
    for raw in seen_filters:
        qf = qb.translate_filter(json.loads(raw))
        for _ in range(10):
            started = time.perf_counter()
            n = int(client.count(collection, count_filter=qf, exact=True).count)
            filter_counts.append({"filter": raw, "matched": n, "ms": (time.perf_counter() - started) * 1000})

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(inputs):
        order = ("A", "C", "D", "B") if i % 2 == 0 else ("D", "C", "A", "B")
        for name in order:
            out = arms[name](item)
            rows.append({"idx": i, "query_id": item["query_id"], "seq": item["seq"], "top_k": item["top_k"], "arm": name,
                         "order": "-".join(order), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}})
    rss_end = _rss()
    qb.HYBRID_SCAN_LIMIT = 5000

    summary: Dict[str, Any] = {
        "set": args.set, "calls_measured": len(inputs), "warmup_per_arm": args.warmup, "point_count": point_count,
        "prepare_seconds": {"open_copy": round(open_s, 3), "reference_build": round(ref_build_s, 3)},
        "ru_maxrss_bytes": {"start": rss0, "after_open": rss_open, "after_reference": rss_ref, "end": rss_end},
        "arms": {},
        "note": "A·C는 같은 구현(상한만 다름, C의 count는 준비 단계 1회). D = 요청마다 count(exact) + 상한=그 값(수정안 §7-A 모양, count 비용 포함). "
                "B는 메모리 행렬 기준 구현이며 운영 수정안의 비용 추정이 아니다. 로컬 파일 모드 측정이며 서버 모드 수치가 아니다",
        "filter_count_ms": {
            "distinct_filters": len(seen_filters), "repeats_each": 10,
            "median_ms": round(float(np.median([f["ms"] for f in filter_counts])), 3) if filter_counts else None,
            "p95_ms": round(float(np.percentile([f["ms"] for f in filter_counts], 95)), 3) if filter_counts else None,
            "matched_range": [min(f["matched"] for f in filter_counts), max(f["matched"] for f in filter_counts)] if filter_counts else None,
        },
    }
    for name in ("A", "C", "D", "B"):
        ms = np.array([r["ms"] for r in rows if r["arm"] == name])
        summary["arms"][name] = {
            "median_ms": round(float(np.median(ms)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "mean_ms": round(float(ms.mean()), 2), "min_ms": round(float(ms.min()), 2), "max_ms": round(float(ms.max()), 2),
            "qdrant_calls_per_query": float(np.mean([r["qdrant_calls"] for r in rows if r["arm"] == name])),
            "returned_points_per_query_mean": float(np.mean([r["returned_points"] for r in rows if r["arm"] == name])),
        }
    for other in ("C", "D"):
        diffs = np.array([
            next(r["ms"] for r in rows if r["idx"] == i and r["arm"] == other) - next(r["ms"] for r in rows if r["idx"] == i and r["arm"] == "A")
            for i in range(len(inputs))
        ])
        summary[f"{other}_minus_A_ms"] = {"median": round(float(np.median(diffs)), 2), "p95": round(float(np.percentile(diffs, 95)), 2),
                                          "mean": round(float(diffs.mean()), 2), "negative_share": round(float((diffs < 0).mean()), 3)}
    d_counts = [r["count_ms"] for r in rows if r["arm"] == "D"]
    summary["D_count_ms"] = {"median": round(float(np.median(d_counts)), 3), "p95": round(float(np.percentile(d_counts, 95)), 3)}
    with (HERE / "latency_calls.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0].keys(), "count_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "latency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
