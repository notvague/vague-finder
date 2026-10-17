"""§5-3 지연 집계 — `latency_requests.csv` + 서버 구간(`logs/main/timing_{A,D}.jsonl`)을 request_id로 합친다.

구분해서 낸다(섞지 않는다):
  (가) 요청 단위 분포 — 범주·구성별 n=30(10질의 × 3회): 중앙값 / p95 / 최대
  (나) 질의 단위 D−A — 질의마다 A 3회 중앙값과 D 3회 중앙값의 차이, 범주별 10개: 중앙값 / p95 / 최대
  (다) 짝 단위 D−A — 같은 회차·같은 질의의 A·D 요청 차이 30쌍: 중앙값 / p95
구간: client(클라이언트 벽시계) · server(미들웨어 total_ms) · search(라우터 검색 단계) · count_unfiltered / count_filtered(`qdrant.count` 실행 합) ·
attr_filter(`path.attr_filter` 실행) · text(`path.text` 실행). p95는 numpy 선형 보간. 준비 요청(warm=1)은 따로 적고 집계에서 뺀다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/aggregate_latency.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "main"
METRICS = ("client", "server", "search", "count_unfiltered", "count_filtered", "attr_filter", "text")


def spans_of(rec):
    out = {"server": rec["total_ms"], "search": None, "count_unfiltered": 0.0, "count_filtered": 0.0,
           "attr_filter": None, "text": None, "n_count_filtered": 0, "n_count_unfiltered": 0}
    for s in rec["spans"]:
        name = s["name"]
        if name == "search":
            out["search"] = s["run_ms"]
        elif name == "path.attr_filter":
            out["attr_filter"] = s["run_ms"]
        elif name == "path.text":
            out["text"] = s["run_ms"]
        elif name == "qdrant.count":
            key = "filtered" if (s.get("attrs") or {}).get("filtered") else "unfiltered"
            out[f"count_{key}"] += s["run_ms"]
            out[f"n_count_{key}"] += 1
    return out


def stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": round(float(np.median(vals)), 1), "p95": round(float(np.percentile(vals, 95)), 1),
            "max": round(float(max(vals)), 1)}


def main() -> int:
    timing = {}
    for arm in ("A", "D"):
        for line in (LOG / f"timing_{arm}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            timing[rec["request_id"]] = rec
    rows = list(csv.DictReader((HERE / "latency_requests.csv").open(encoding="utf-8")))
    joined, unmatched, bad = [], [], []
    for r in rows:
        rec = timing.get(r["request_id"])
        if r["status"] != "200":
            bad.append(r)
        if rec is None:
            unmatched.append(r["request_id"])
            continue
        j = {**r, "client": float(r["client_ms"]), **spans_of(rec)}
        joined.append(j)
    with (HERE / "latency_joined.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "set", "category", "round", "arm", "order_in_pair", "warm", "status", "request_id", "candidate_digest",
                  *METRICS, "n_count_unfiltered", "n_count_filtered"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(joined)
    measured = [j for j in joined if j["warm"] == "0"]
    out = {"requests": len(rows), "measured": len(measured), "warm": [
        {k: j[k] for k in ("arm", "query_id", "client", "server", "search")} for j in joined if j["warm"] == "1"],
        "non_200": len(bad), "timing_unmatched": len(unmatched), "categories": {}}
    for category in ("closed", "open_hits", "open_empty", "open_empty_synthetic", "all"):
        sel = [j for j in measured if category == "all" or j["category"] == category]
        if not sel:
            out["categories"][category] = "없음"
            continue
        cat = {"request": {}, "query_diff": {}, "pair_diff": {}, "count_calls": {}}
        for arm in ("A", "D"):
            arm_rows = [j for j in sel if j["arm"] == arm]
            cat["request"][arm] = {m: stats([j[m] for j in arm_rows]) for m in METRICS}
            cat["count_calls"][arm] = {"filtered_per_request": dict(sorted(statistics.multimode([j["n_count_filtered"] for j in arm_rows]) and
                                                                      {n: sum(1 for j in arm_rows if j["n_count_filtered"] == n) for n in {j["n_count_filtered"] for j in arm_rows}}.items())),
                                       "unfiltered_per_request": {n: sum(1 for j in arm_rows if j["n_count_unfiltered"] == n) for n in sorted({j["n_count_unfiltered"] for j in arm_rows})}}
        per_q = defaultdict(lambda: defaultdict(list))
        for j in sel:
            per_q[(j["category"], j["query_id"])][j["arm"]].append(j)
        for m in ("client", "server", "search"):
            diffs = []
            for key, arms in per_q.items():
                a = [x[m] for x in arms["A"]]
                d = [x[m] for x in arms["D"]]
                if a and d:
                    diffs.append(float(np.median(d)) - float(np.median(a)))
            cat["query_diff"][m] = stats(diffs) | {"min": round(min(diffs), 1)} if diffs else None
            pairs = defaultdict(dict)
            for j in sel:
                pairs[(j["category"], j["query_id"], j["round"])][j["arm"]] = j[m]
            pd_ = [p["D"] - p["A"] for p in pairs.values() if "A" in p and "D" in p]
            cat["pair_diff"][m] = stats(pd_) if pd_ else None
        digests = defaultdict(set)
        for j in sel:
            digests[(j["query_id"], j["category"], j["arm"])].add(j["candidate_digest"])
        cat["candidate_digest_stable"] = sum(1 for v in digests.values() if len(v) == 1)
        cat["candidate_digest_groups"] = len(digests)
        a_eq_d = 0
        qs = {(q, c) for (q, c, _a) in digests}
        for q, c in qs:
            if digests.get((q, c, "A")) == digests.get((q, c, "D")):
                a_eq_d += 1
        cat["A_eq_D_candidates"] = f"{a_eq_d}/{len(qs)}"
        out["categories"][category] = cat
    (HERE / "latency_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for category, cat in out["categories"].items():
        if isinstance(cat, str):
            print(category, cat)
            continue
        print(f"== {category}  A=D cand {cat['A_eq_D_candidates']}  stable {cat['candidate_digest_stable']}/{cat['candidate_digest_groups']}")
        for arm in ("A", "D"):
            r = cat["request"][arm]
            print(f"  {arm} client {r['client']} | server {r['server']} | search {r['search']} | cnt_unf {r['count_unfiltered']} | cnt_f {r['count_filtered']} | attr {r['attr_filter']} | text {r['text']}")
            print(f"     count calls {cat['count_calls'][arm]}")
        for m in ("client", "server", "search"):
            print(f"  D-A {m}: query {cat['query_diff'][m]}  pair {cat['pair_diff'][m]}")
    print("warm", out["warm"], "non200", out["non_200"], "unmatched", out["timing_unmatched"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
