"""v61 — **수정 코드 검증용 재생**: 저장된 질의 벡터(A 실행의 `calls_vectors.npz`·`calls.jsonl`)를 **지금 디스크에 있는** 서비스 구현
(`src.vector_db.qdrant_backend.QdrantIndex.query`)에 다시 넣어 결과를 새로 만들고, 독립 정확 합산 기준(`exact_hybrid.py`)과 대조한다.

`compare_text_calls.py`는 저장된 A/B/C 결과만 읽으므로 운영 코드를 고친 뒤 그것만 다시 돌려도 새 구현은 호출되지 않는다. 수정 뒤에는
**이 스크립트**를 돌려야 한다. 서비스 코드는 수정하지 않고, 모듈 상수를 바꾸고 싶으면 `QDRANT_HYBRID_SCAN_LIMIT` 환경변수로 준다.

기대: 현행 코드(상한 5000) → 호출 차이 4건(RUN_INFO §5-1과 같아야 한다) · 수정 코드(상한 = 대상 수) → 0건.
실행: [QDRANT_HYBRID_SCAN_LIMIT=5143] venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/replay_calls.py [--label fixed]
출력: replay_<label>.json(호출별 대조 요약·색인 지문·사용한 상한)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current")
    parser.add_argument("--source-arm", default="A", help="질의 벡터를 읽을 저장 실행(기본 A)")
    args = parser.parse_args()

    client = QdrantClient(path=os.environ["QDRANT_PATH"])
    index = qb.QdrantIndex(client, TEXT_HYBRID_INDEX_NAME, exact=False)
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    reference = ExactHybridReference.from_client(client, collection)
    fingerprint = reference.fingerprint()
    filter_cache: Dict[str, Set[int]] = {}
    counts = Counter()
    diffs: List[Dict[str, Any]] = []
    elapsed: List[float] = []
    for set_name in SETS:
        folder = HERE / f"cand_{args.source_arm}" / f"{set_name}_dev"
        calls = [json.loads(line) for line in (folder / "calls.jsonl").open(encoding="utf-8")]
        vectors = np.load(folder / "calls_vectors.npz")
        for call in calls:
            seq = call["seq"]
            dense_q = vectors[f"call{seq}_dense"].astype(float).tolist()
            sparse_q = {"indices": vectors[f"call{seq}_sparse_idx"].tolist(), "values": vectors[f"call{seq}_sparse_val"].astype(float).tolist()}
            flt = json.loads(call["filter"]) if call["filter"] else None
            started = time.perf_counter()
            res = index.query(vector=dense_q, sparse_vector=sparse_q, top_k=call["top_k"], include_metadata=False, namespace=NAMESPACE,
                              **({"filter": flt} if flt else {}))
            elapsed.append((time.perf_counter() - started) * 1000)
            allowed: Optional[Set[int]] = None
            if flt:
                allowed = filter_cache.get(call["filter"])
                if allowed is None:
                    allowed = allowed_point_ids(client, collection, qb.translate_filter(flt))
                    filter_cache[call["filter"]] = allowed
            exact_top = reference.top(dense_q, sparse_q, call["top_k"], allowed_point_ids=allowed)
            cmp = compare_rankings([{"id": m["id"], "score": m["score"]} for m in res["matches"]], exact_top, tol=TOTAL_TOL)
            ok = cmp["same_set"] and not cmp["score_mismatch"] and not cmp["order_differs_beyond_ties"]
            counts["calls"] += 1
            counts["matches_exact"] += int(ok)
            counts["missing_songs"] += len(cmp["missing"])
            counts["score_mismatch_songs"] += len(cmp["score_mismatch"])
            if not ok:
                diffs.append({"set": set_name, "query_id": call["query_id"], "seq": seq, "top_k": call["top_k"], "missing": cmp["missing"],
                              "score_mismatch": [m["id"] for m in cmp["score_mismatch"]], "order_differs_beyond_ties": cmp["order_differs_beyond_ties"]})
    out = {
        "label": args.label, "source_arm": args.source_arm, "hybrid_scan_limit_in_effect": qb.HYBRID_SCAN_LIMIT,
        "env_QDRANT_HYBRID_SCAN_LIMIT": os.getenv("QDRANT_HYBRID_SCAN_LIMIT", ""), "reference_fingerprint": fingerprint,
        "tolerance": TOTAL_TOL, "counts": dict(counts), "diffs": diffs,
        "replayed_call_ms": {"median": round(float(np.median(elapsed)), 2), "p95": round(float(np.percentile(elapsed, 95)), 2)},
    }
    (HERE / f"replay_{args.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("label", "hybrid_scan_limit_in_effect", "counts", "replayed_call_ms")}, ensure_ascii=False))
    for d in diffs:
        print("  diff:", d)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
