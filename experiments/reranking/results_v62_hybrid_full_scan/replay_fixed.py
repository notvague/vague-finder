"""v62 — 저장 호출 311건을 **수정된 `QdrantIndex.query`에 실제로 재생**하고 독립 정확 합산 기준과 대조한다.

v61의 `replay_calls.py`와 같은 입력(v61 워크트리의 A 실행 `calls.jsonl` · `calls_vectors.npz`, 읽기 전용)·같은 기준(`exact_hybrid.py`)이지만
  - 지금 디스크의 구현(v62: limit = 요청마다 count)을 **환경변수 우회 없이** 부른다(`QDRANT_HYBRID_SCAN_LIMIT` 없음)
  - 호출마다 `count`가 돌려준 값과 두 조회의 `limit`·반환 건수를 가로채 기록한다(limit == count, dense 반환 == count 확인)
  - 차이가 있으면 동점(허용 오차 안 묶음의 순서) · 부동소수점 오차(허용 오차 안 점수 차이) · 실제 누락(집합·점수 불일치)을 나눠 적는다
  - 색인 복사본 지문이 v61과 같고 분석 캐시가 지금 분석기와 맞는지 먼저 확인한다
출력(이 폴더): replay_fixed.json · replay_fixed_calls.csv — v61 폴더의 `replay_current.json` 등은 건드리지 않는다.
실행: venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/replay_fixed.py [--label fixed]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v62_common as common  # noqa: E402

APPLIED_ENV, GUARD_BEFORE = common.guard()

from qdrant_client import QdrantClient  # noqa: E402

from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

from exact_hybrid import TOTAL_TOL, ExactHybridReference, allowed_point_ids, compare_rankings, tie_groups  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="fixed")
    parser.add_argument("--source-arm", default="A")
    parser.add_argument("--expected-calls", type=int, default=311,
                        help="재생해야 하는 저장 호출 수(v61 A: 105+102+104). 적게 읽혔으면 입력이 빠진 것이라 실패로 끝낸다(0이면 검사 안 함)")
    args = parser.parse_args()
    if os.getenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV):
        raise SystemExit(f"{qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV}가 설정돼 있다 — 우회 없이 재생해야 하므로 지우고 다시 실행")

    cache_compat = common.analysis_cache_compat()
    t0 = time.perf_counter()
    client = QdrantClient(path=os.environ["QDRANT_PATH"])
    index = qb.QdrantIndex(client, TEXT_HYBRID_INDEX_NAME, exact=False)
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    point_count = int(client.count(collection, exact=True).count)
    open_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    reference = ExactHybridReference.from_client(client, collection)
    fingerprint = common.check_fingerprint(reference.fingerprint())
    ref_s = time.perf_counter() - t1

    # 호출 가로채기 — 구현이 실제로 어떤 count·limit으로 조회했는지
    trace: Dict[str, Any] = {}
    real_count, real_query = client.count, client.query_points

    def spy_count(*a: Any, **kw: Any) -> Any:
        res = real_count(*a, **kw)
        trace.setdefault("count", []).append(int(res.count))
        return res

    def spy_query(*a: Any, **kw: Any) -> Any:
        res = real_query(*a, **kw)
        trace.setdefault("limits", []).append(int(kw["limit"]))
        trace.setdefault("returned", []).append(len(res.points))
        trace.setdefault("using", []).append(kw["using"])
        return res

    client.count = spy_count  # type: ignore[assignment]
    client.query_points = spy_query  # type: ignore[assignment]

    filter_cache: Dict[str, Set[int]] = {}
    counts: Counter = Counter()
    diffs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    elapsed: List[float] = []
    max_delta_all = 0.0
    for set_name in common.SETS:
        calls, vectors = common.load_calls(set_name, args.source_arm)
        for call in calls:
            dense_q, sparse_q, flt = common.call_inputs(call, vectors)
            trace.clear()
            started = time.perf_counter()
            res = index.query(vector=dense_q, sparse_vector=sparse_q, top_k=call["top_k"], include_metadata=False, namespace=NAMESPACE,
                              **({"filter": flt} if flt else {}))
            ms = (time.perf_counter() - started) * 1000
            elapsed.append(ms)
            allowed: Optional[Set[int]] = None
            if flt:
                allowed = filter_cache.get(call["filter"])
                if allowed is None:
                    allowed = allowed_point_ids(client, collection, qb.translate_filter(flt))
                    filter_cache[call["filter"]] = allowed
            exact_top = reference.top(dense_q, sparse_q, call["top_k"], allowed_point_ids=allowed)
            actual = [{"id": m["id"], "score": m["score"]} for m in res["matches"]]
            cmp = compare_rankings(actual, exact_top, tol=TOTAL_TOL)
            ok = cmp["same_set"] and not cmp["score_mismatch"] and not cmp["order_differs_beyond_ties"]
            # 차이 분류 — 집합/점수(허용 오차 밖) 불일치 = 실제 누락, 동점 묶음 안 순서 = 동점, 허용 오차 안 점수 차이 = 부동소수점
            if cmp["missing"] or cmp["extra"] or cmp["score_mismatch"] or cmp["order_differs_beyond_ties"]:
                kind = "actual_mismatch"
            elif not cmp["identical_ids"]:
                kind = "tie_order_only"
            elif cmp["max_abs_score_delta_common"] > 0:
                kind = "float_within_tol"
            else:
                kind = "identical"
            max_delta_all = max(max_delta_all, cmp["max_abs_score_delta_common"])
            counted = trace.get("count", [None])[0]
            limits = trace.get("limits", [])
            returned = trace.get("returned", [])
            expected_targets = point_count if allowed is None else len(allowed)
            limit_ok = (counted == expected_targets) and (limits == [counted, counted])
            dense_full = bool(returned) and returned[0] == counted
            ties = sum(1 for g in tie_groups(exact_top, TOTAL_TOL) if len(g) > 1)
            counts["calls"] += 1
            counts["matches_exact"] += int(ok)
            counts[f"kind_{kind}"] += 1
            counts["limit_equals_count"] += int(limit_ok)
            counts["dense_returned_all_targets"] += int(dense_full)
            counts["filtered_calls"] += int(flt is not None)
            counts["calls_with_tie_groups"] += int(ties > 0)
            counts["missing_songs"] += len(cmp["missing"])
            counts["score_mismatch_songs"] += len(cmp["score_mismatch"])
            rows.append({
                "set": set_name, "query_id": call["query_id"], "seq": call["seq"], "top_k": call["top_k"], "filtered": int(flt is not None),
                "targets_expected": expected_targets, "count_returned": counted, "dense_limit": limits[0] if limits else "",
                "sparse_limit": limits[1] if len(limits) > 1 else "", "dense_returned": returned[0] if returned else "",
                "sparse_returned": returned[1] if len(returned) > 1 else "", "matches": len(actual),
                "ok": int(ok), "kind": kind, "max_abs_score_delta": f"{cmp['max_abs_score_delta_common']:.3e}",
                "tie_groups_in_reference": ties, "ms": round(ms, 3),
            })
            if not ok:
                diffs.append({"set": set_name, "query_id": call["query_id"], "seq": call["seq"], "top_k": call["top_k"], "kind": kind,
                              "missing": cmp["missing"], "extra": cmp["extra"], "score_mismatch": cmp["score_mismatch"],
                              "order_differs_beyond_ties": cmp["order_differs_beyond_ties"]})
    client.count = real_count  # type: ignore[assignment]
    client.query_points = real_query  # type: ignore[assignment]
    guard_after = common.verify_guard()

    out = {
        "label": args.label, "git_sha": common.git_sha(), "source_arm": args.source_arm, "source_inputs": common.source_inputs_digest(),
        "hybrid_scan_limit_compat_value": qb.HYBRID_SCAN_LIMIT, "env_QDRANT_HYBRID_SCAN_LIMIT": os.getenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV, ""),
        "qdrant_path": os.environ["QDRANT_PATH"], "point_count": point_count, "reference_fingerprint": fingerprint,
        "analysis_cache_compat": cache_compat, "tolerance": TOTAL_TOL, "counts": dict(counts), "max_abs_score_delta_common": max_delta_all,
        "diffs": diffs, "replayed_call_ms": {"median": round(float(np.median(elapsed)), 2), "p95": round(float(np.percentile(elapsed, 95)), 2)},
        "prepare_seconds": {"open_copy": round(open_s, 3), "reference_build": round(ref_s, 3)},
        "env": APPLIED_ENV, "guard_before": GUARD_BEFORE, "guard_after": guard_after,
    }
    (common.HERE / f"replay_{args.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with (common.HERE / f"replay_{args.label}_calls.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: out[k] for k in ("label", "git_sha", "hybrid_scan_limit_compat_value", "counts", "max_abs_score_delta_common", "replayed_call_ms")},
                     ensure_ascii=False))
    for d in diffs:
        print("  diff:", d)
    client.close()

    # 검증 실패는 비정상 종료다 — 저장만 하고 0으로 끝내면 상위 스크립트가 "통과"로 읽는다
    problems: List[str] = []
    if args.expected_calls and counts["calls"] != args.expected_calls:
        problems.append(f"재생 호출 수 {counts['calls']} ≠ 기대 {args.expected_calls}(입력 누락)")
    if counts["matches_exact"] != counts["calls"]:
        problems.append(f"기준 불일치 {counts['calls'] - counts['matches_exact']}건(누락 곡 {counts['missing_songs']} · 점수 불일치 곡 {counts['score_mismatch_songs']})")
    if counts["limit_equals_count"] != counts["calls"]:
        problems.append(f"limit ≠ count인 호출 {counts['calls'] - counts['limit_equals_count']}건")
    if counts["dense_returned_all_targets"] != counts["calls"]:
        problems.append(f"dense 반환 ≠ 대상 수인 호출 {counts['calls'] - counts['dense_returned_all_targets']}건")
    out["verdict"] = {"ok": not problems, "problems": problems}
    (common.HERE / f"replay_{args.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if problems:
        print("REPLAY FAILED:", " / ".join(problems), file=sys.stderr)
        return 2
    print("REPLAY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
