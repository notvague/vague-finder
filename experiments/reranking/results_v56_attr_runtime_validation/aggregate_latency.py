"""§5 집계 — `latency_requests.csv`(클라이언트)와 `timing_{A,D}.jsonl`(서버 구간 기록)을 request_id 로 합쳐
질의별 A/D 시간·차이, 범주별 중앙값·p95·최대, 경로 구간(path.attr_filter · path.text · search · search.fuse)을 낸다. 모델·DB 없음.

규칙: warm=1(준비 요청)만 제외. 그 밖의 요청은 느려도 전부 포함한다. 오류·시간 초과는 따로 센다.
실행: venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/aggregate_latency.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
SPANS = ("analysis", "search", "search.fuse", "path.text", "path.attr_filter", "path.context", "path.image", "path.audio",
         "path.lyrics", "path.performance", "path.metadata", "explain", "clarify", "route")


def load_timing(arm: str) -> Dict[str, Dict[str, Any]]:
    path = HERE / f"timing_{arm}.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        spans = {}
        for span in rec.get("spans", []):
            name = span["name"]
            if name in spans:  # 같은 이름이 여러 번이면 합산(재시도 등)
                spans[name]["run_ms"] += span.get("run_ms", 0.0)
                spans[name]["wait_ms"] += span.get("wait_ms", 0.0)
            else:
                spans[name] = {"run_ms": span.get("run_ms", 0.0), "wait_ms": span.get("wait_ms", 0.0), "attrs": span.get("attrs", {})}
        out[rec["request_id"]] = {"total_ms": rec.get("total_ms"), "spans": spans, "attrs": rec.get("attrs", {})}
    return out


def pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"n": len(values), "median": round(statistics.median(values), 1), "p95": round(pct(values, 0.95), 1),
            "max": round(max(values), 1), "mean": round(statistics.mean(values), 1)}


def main() -> int:
    rows = list(csv.DictReader((HERE / "latency_requests.csv").open(encoding="utf-8")))
    timing = {arm: load_timing(arm) for arm in ("A", "D")}
    joined: List[Dict[str, Any]] = []
    unmatched = 0
    for r in rows:
        t = timing[r["arm"]].get(r["request_id"])
        rec = dict(r)
        rec["client_ms"] = float(r["client_ms"]) if r["client_ms"] else float("nan")
        if t is None:
            unmatched += 1
            rec["server_total_ms"] = ""
            for s in SPANS:
                rec[f"{s}_run_ms"] = ""
            rec["attr_filter_wait_ms"] = ""
        else:
            rec["server_total_ms"] = t["total_ms"]
            for s in SPANS:
                rec[f"{s}_run_ms"] = round(t["spans"][s]["run_ms"], 1) if s in t["spans"] else ""
            rec["attr_filter_wait_ms"] = round(t["spans"]["path.attr_filter"]["wait_ms"], 1) if "path.attr_filter" in t["spans"] else ""
            rec["fuse_candidates"] = (t["spans"].get("search.fuse", {}).get("attrs") or {}).get("candidates", "")
        joined.append(rec)
    fields = list(joined[0].keys())
    with (HERE / "latency_joined.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields); w.writeheader(); w.writerows(joined)

    measured = [r for r in joined if r["warm"] == "0"]
    errors = [r for r in measured if r["status"] != "200"]
    # 질의별 A/D (회차별 짝 + 중앙값)
    by_q: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in measured:
        by_q[(r["category"], r["query_id"])][r["arm"]].append(r)
    per_query: List[Dict[str, Any]] = []
    for (cat, qid), arms in sorted(by_q.items()):
        a, d = arms.get("A", []), arms.get("D", [])
        def med(rs, key):
            vals = [float(x[key]) for x in rs if x.get(key) not in ("", None)]
            return round(statistics.median(vals), 1) if vals else ""
        rec = {
            "category": cat, "query_id": qid, "n_A": len(a), "n_D": len(d),
            "A_client_med": med(a, "client_ms"), "D_client_med": med(d, "client_ms"),
            "A_server_total_med": med(a, "server_total_ms"), "D_server_total_med": med(d, "server_total_ms"),
            "A_search_med": med(a, "search_run_ms"), "D_search_med": med(d, "search_run_ms"),
            "D_attr_filter_run_med": med(d, "path.attr_filter_run_ms"), "D_attr_filter_wait_med": med(d, "attr_filter_wait_ms"),
            "A_text_med": med(a, "path.text_run_ms"), "D_text_med": med(d, "path.text_run_ms"),
            "A_fuse_med": med(a, "search.fuse_run_ms"), "D_fuse_med": med(d, "search.fuse_run_ms"),
            "A_candidates": "/".join(sorted({x["n_candidates"] for x in a})), "D_candidates": "/".join(sorted({x["n_candidates"] for x in d})),
            "A_digest": "/".join(sorted({x["candidate_digest"] for x in a})), "D_digest": "/".join(sorted({x["candidate_digest"] for x in d})),
            "digest_same_A_D": len({x["candidate_digest"] for x in a} | {x["candidate_digest"] for x in d}) == 1,
            "A_rounds_client": "|".join(str(x["client_ms"]) for x in sorted(a, key=lambda x: int(x["round"]))),
            "D_rounds_client": "|".join(str(x["client_ms"]) for x in sorted(d, key=lambda x: int(x["round"]))),
        }
        for key in ("client", "server_total", "search"):
            av, dv = rec[f"A_{key}_med"], rec[f"D_{key}_med"]
            rec[f"diff_{key}_D_minus_A"] = round(dv - av, 1) if av != "" and dv != "" else ""
        per_query.append(rec)
    with (HERE / "latency_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(per_query[0].keys())); w.writeheader(); w.writerows(per_query)

    # 범주별 집계(요청 단위 — 회차 3개 × 질의)
    summary: List[Dict[str, Any]] = []
    cats = ["closed", "open_hits", "open_empty", "open_empty_synthetic", "ALL"]
    for cat in cats:
        for arm in ("A", "D"):
            rs = [r for r in measured if (cat == "ALL" or r["category"] == cat) and r["arm"] == arm and r["status"] == "200"]
            if not rs:
                continue
            for key, label in (("client_ms", "client"), ("server_total_ms", "server_total"), ("search_run_ms", "search"),
                                ("path.attr_filter_run_ms", "attr_filter_run"), ("path.text_run_ms", "text_run"), ("search.fuse_run_ms", "fuse")):
                vals = [float(r[key]) for r in rs if r.get(key) not in ("", None)]
                if not vals:
                    continue
                summary.append({"category": cat, "arm": arm, "metric": label, **stats(vals)})
        # 질의별 짝 차이(D−A)의 분포
        diffs = [p[f"diff_client_D_minus_A"] for p in per_query if (cat == "ALL" or p["category"] == cat) and p["diff_client_D_minus_A"] != ""]
        if diffs:
            summary.append({"category": cat, "arm": "D-A", "metric": "client_diff_per_query_median_of_3", **stats(diffs)})
        diffs = [p[f"diff_search_D_minus_A"] for p in per_query if (cat == "ALL" or p["category"] == cat) and p["diff_search_D_minus_A"] != ""]
        if diffs:
            summary.append({"category": cat, "arm": "D-A", "metric": "search_diff_per_query_median_of_3", **stats(diffs)})
    with (HERE / "latency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["category", "arm", "metric", "n", "median", "p95", "max", "mean"]); w.writeheader(); w.writerows(summary)

    warm = [r for r in joined if r["warm"] == "1"]
    print(f"requests={len(rows)} measured={len(measured)} warm={len(warm)} errors={len(errors)} timing_unmatched={unmatched}")
    for r in warm:
        print(f"  warm {r['arm']} {r['query_id']} client={r['client_ms']} server={r['server_total_ms']} search={r['search_run_ms']}")
    for e in errors:
        print(f"  ERROR {e['arm']} {e['query_id']} r{e['round']} status={e['status']} {e['error'][:80]}")
    print(f"{'category':>22} {'arm':>4} {'metric':>18} {'n':>3} {'median':>8} {'p95':>8} {'max':>8}")
    for s in summary:
        print(f"{s['category']:>22} {s['arm']:>4} {s['metric']:>18} {s['n']:>3} {s['median']:>8} {s['p95']:>8} {s['max']:>8}")
    same = sum(1 for p in per_query if p["digest_same_A_D"])
    print(f"candidate digest same A==D: {same}/{len(per_query)}; per-arm repeat-consistent: "
          f"A {sum(1 for p in per_query if '/' not in p['A_digest'])}/{len(per_query)}, D {sum(1 for p in per_query if '/' not in p['D_digest'])}/{len(per_query)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
