"""v62 — 하이브리드 조회 지연: **수정 전**(v61 기준 커밋 fc20e86의 `qdrant_backend.py`, 상한 5000) ↔ **수정 후**(지금 디스크 구현, limit = count).

같은 Qdrant 클라이언트 위에 두 구현의 `QdrantIndex`를 만들어, 저장 질의 벡터(v61 A 실행, 읽기 전용)를 호출마다 **교대로**(전→후 / 후→전)
넣는다. 두 구성을 동시에 돌리지 않는다. include_metadata=True(서비스 모양)로 payload 조회까지 포함한 전체 시간을 재고, 클라이언트의
`count` · `query_points`(dense·sparse) · `retrieve`를 감싸 **단계별 시간**과 반환 항목 수를 기록한다.

  --mode local   : 로컬 파일 모드 복사본(`QDRANT_PATH`, v61_guard가 고정) · exact 파라미터 없음(로컬은 항상 전수)
  --mode server  : `--url`의 전용 Docker Qdrant(server_setup_v62.py가 띄움) · **서비스와 같은 `exact=True`**(`QdrantVectorClient.Index`가 서버 모드에서 쓰는 설정)
필터 없는 호출(`--set`의 앞 `--max-calls`건)과 필터 있는 호출(세 세트의 필터 호출 전부 × `--filter-repeats`)을 나눠 잰다.
예열(각 구성 `--warmup`회)은 측정에서 뺀다. 결과 ID·점수가 두 구현에서 같은지도 함께 적는다(v61에서 확인된 4건은 다를 수 있다).
출력: bench_<mode>_calls.csv · bench_<mode>_summary.json
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
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v62_common as common  # noqa: E402

APPLIED_ENV, GUARD_BEFORE = common.guard()

from qdrant_client import QdrantClient  # noqa: E402

from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

STEP_NAMES = ("count", "dense", "sparse", "retrieve")


class StepSpy:
    """클라이언트의 count·query_points·retrieve를 감싸 단계별 ms·반환 항목 수를 모은다."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.reset()
        self._real = {"count": client.count, "query_points": client.query_points, "retrieve": client.retrieve}
        client.count = self._wrap("count", self._real["count"], lambda r: 1)
        client.query_points = self._wrap_query(self._real["query_points"])
        client.retrieve = self._wrap("retrieve", self._real["retrieve"], lambda r: len(r))

    def reset(self) -> None:
        self.ms: Dict[str, float] = {k: 0.0 for k in STEP_NAMES}
        self.calls: Dict[str, int] = {k: 0 for k in STEP_NAMES}
        self.returned: Dict[str, int] = {k: 0 for k in STEP_NAMES}

    def _wrap(self, name: str, real: Callable, size: Callable) -> Callable:
        def inner(*a: Any, **kw: Any) -> Any:
            t = time.perf_counter()
            out = real(*a, **kw)
            self.ms[name] += (time.perf_counter() - t) * 1000
            self.calls[name] += 1
            self.returned[name] += int(size(out))
            return out
        return inner

    def _wrap_query(self, real: Callable) -> Callable:
        def inner(*a: Any, **kw: Any) -> Any:
            t = time.perf_counter()
            out = real(*a, **kw)
            name = "dense" if kw.get("using") == qb.DENSE_VECTOR else "sparse"
            self.ms[name] += (time.perf_counter() - t) * 1000
            self.calls[name] += 1
            self.returned[name] += len(out.points)
            return out
        return inner

    def restore(self) -> None:
        for k, v in self._real.items():
            setattr(self.client, k, v)


def _pct(values: List[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "server"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6362")
    parser.add_argument("--set", default="v06")
    parser.add_argument("--max-calls", type=int, default=60, help="필터 없는 호출 수(앞에서부터)")
    parser.add_argument("--filter-repeats", type=int, default=5, help="세 세트의 필터 호출(16건)을 각각 몇 번 반복할지")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if os.getenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV):
        raise SystemExit(f"{qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV}가 설정돼 있다 — 지우고 실행")

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if args.mode == "server":
        client = QdrantClient(url=args.url)
        exact = True
        server_version = client.info().version
    else:
        client = QdrantClient(path=os.environ["QDRANT_PATH"])
        exact = False
        server_version = None
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    point_count = int(client.count(collection, exact=True).count)
    open_s = time.perf_counter() - t0

    legacy = common.load_legacy_backend(common.HERE / "logs")
    assert legacy.HYBRID_SCAN_LIMIT == 5000, legacy.HYBRID_SCAN_LIMIT
    before = legacy.QdrantIndex(client, TEXT_HYBRID_INDEX_NAME, exact=exact)
    after = qb.QdrantIndex(client, TEXT_HYBRID_INDEX_NAME, exact=exact)
    impls = {"before": before, "after": after}

    # 입력 — 필터 없는 호출(한 세트)과 필터 있는 호출(세 세트 전부)
    plain: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []
    for set_name in common.SETS:
        calls, vectors = common.load_calls(set_name)
        for call in calls:
            dense, sparse, flt = common.call_inputs(call, vectors)
            item = {"set": set_name, "seq": call["seq"], "query_id": call["query_id"], "top_k": call["top_k"], "dense": dense, "sparse": sparse, "filter": flt}
            if flt:
                filtered.append(item)
            elif set_name == args.set and len(plain) < args.max_calls:
                plain.append(item)

    spy = StepSpy(client)

    def run(name: str, item: Dict[str, Any]) -> Dict[str, Any]:
        spy.reset()
        started = time.perf_counter()
        res = impls[name].query(vector=item["dense"], sparse_vector=item["sparse"], top_k=item["top_k"], include_metadata=True, namespace=NAMESPACE,
                                **({"filter": item["filter"]} if item["filter"] else {}))
        total = (time.perf_counter() - started) * 1000
        out = {"ms": total, "n": len(res["matches"]), "ids": [m["id"] for m in res["matches"]], "scores": [round(m["score"], 7) for m in res["matches"]]}
        for k in STEP_NAMES:
            out[f"{k}_ms"] = spy.ms[k]
            out[f"{k}_calls"] = spy.calls[k]
            out[f"{k}_returned"] = spy.returned[k]
        out["qdrant_calls"] = sum(spy.calls.values())
        out["returned_points"] = spy.returned["dense"] + spy.returned["sparse"]
        return out

    # 예열(측정 제외)
    t_w = time.perf_counter()
    for _ in range(args.warmup):
        for name in ("before", "after"):
            run(name, plain[0])
            if filtered:
                run(name, filtered[0])
    warmup_s = time.perf_counter() - t_w

    rows: List[Dict[str, Any]] = []

    def measure(group: str, items: List[Dict[str, Any]], repeats: int) -> None:
        idx = 0
        for rep in range(repeats):
            for item in items:
                order = ("before", "after") if idx % 2 == 0 else ("after", "before")
                outs = {}
                for name in order:
                    outs[name] = run(name, item)
                same = outs["before"]["ids"] == outs["after"]["ids"] and outs["before"]["scores"] == outs["after"]["scores"]
                for name in order:
                    o = outs[name]
                    rows.append({"group": group, "idx": idx, "rep": rep, "set": item["set"], "query_id": item["query_id"], "seq": item["seq"],
                                 "top_k": item["top_k"], "filtered": int(bool(item["filter"])), "impl": name, "order": "-".join(order),
                                 "same_result_before_after": int(same),
                                 **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k not in ("ids", "scores")}})
                idx += 1

    measure("plain", plain, 1)
    measure("filtered", filtered, args.filter_repeats)
    spy.restore()
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    guard_after = common.verify_guard()

    def summarize(group: str) -> Dict[str, Any]:
        g = [r for r in rows if r["group"] == group]
        out: Dict[str, Any] = {"requests_per_impl": len(g) // 2, "distinct_inputs": len({(r["set"], r["seq"]) for r in g}), "impl": {}}
        for name in ("before", "after"):
            rr = [r for r in g if r["impl"] == name]
            ms = [r["ms"] for r in rr]
            out["impl"][name] = {
                "median_ms": _pct(ms, 50), "p95_ms": _pct(ms, 95), "mean_ms": round(float(np.mean(ms)), 2), "min_ms": round(min(ms), 2), "max_ms": round(max(ms), 2),
                "qdrant_calls_per_request": round(float(np.mean([r["qdrant_calls"] for r in rr])), 2),
                "returned_points_per_request_mean": round(float(np.mean([r["returned_points"] for r in rr])), 1),
                "matches_per_request_mean": round(float(np.mean([r["n"] for r in rr])), 1),
                "step_median_ms": {k: _pct([r[f"{k}_ms"] for r in rr if r[f"{k}_calls"]], 50) if any(r[f"{k}_calls"] for r in rr) else None for k in STEP_NAMES},
                "step_p95_ms": {k: _pct([r[f"{k}_ms"] for r in rr if r[f"{k}_calls"]], 95) if any(r[f"{k}_calls"] for r in rr) else None for k in STEP_NAMES},
            }
        by_idx: Dict[int, Dict[str, float]] = {}
        for r in g:
            by_idx.setdefault(r["idx"], {})[r["impl"]] = r["ms"]
        diffs = [v["after"] - v["before"] for v in by_idx.values() if "after" in v and "before" in v]
        out["after_minus_before_ms"] = {"median": _pct(diffs, 50), "p95": _pct(diffs, 95), "mean": round(float(np.mean(diffs)), 2),
                                        "negative_share": round(float(np.mean([d < 0 for d in diffs])), 3), "pairs": len(diffs)}
        out["same_result_before_after"] = sum(1 for r in g if r["impl"] == "after" and r["same_result_before_after"])
        out["different_result_pairs"] = sorted({(r["set"], r["query_id"], r["seq"]) for r in g if r["impl"] == "after" and not r["same_result_before_after"]})
        return out

    label = f"_{args.label}" if args.label else ""
    summary = {
        "mode": args.mode, "url": args.url if args.mode == "server" else None, "server_version": server_version,
        "qdrant_client": __import__("importlib.metadata").metadata.version("qdrant-client"),
        "exact_search_params": exact, "git_sha": common.git_sha(), "legacy_commit": common.V61_BASE_COMMIT, "legacy_scan_limit": legacy.HYBRID_SCAN_LIMIT,
        "point_count": point_count, "set": args.set, "warmup_per_impl": args.warmup,
        "prepare_seconds": {"open_client": round(open_s, 3), "warmup": round(warmup_s, 3)},
        "ru_maxrss_bytes": {"start": rss0, "end": rss_end},
        "plain": summarize("plain"), "filtered": summarize("filtered") if filtered else None,
        "note": ("같은 클라이언트 위에서 두 구현을 호출마다 교대로 실행(동시 실행 없음). 전체 ms = count+dense+sparse+retrieve+합산. "
                 "로컬 PC(루프백)의 Docker 서버라 배포 서버의 네트워크·하드웨어 수치가 아니다" if args.mode == "server" else
                 "로컬 파일 모드 한 프로세스. 서버 모드 수치가 아니다"),
        "env": APPLIED_ENV, "guard_before": GUARD_BEFORE, "guard_after": guard_after,
    }
    with (common.HERE / f"bench_{args.mode}{label}_calls.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (common.HERE / f"bench_{args.mode}{label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "point_count", "plain", "filtered")}, ensure_ascii=False, indent=1))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
