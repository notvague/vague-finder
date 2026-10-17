"""예산 장부(budget/ledger.jsonl) 요약 — 실행별 HTTP 시도·성공·실패·토큰·확정/미확인 비용·지연. 모델 호출 없음.

    venv/bin/python budget_summary.py [--ledger budget/ledger.jsonl] → budget_summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(HERE / "budget/ledger.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(l) for l in Path(args.ledger).read_text(encoding="utf-8").splitlines() if l.strip()]
    reserves = {r["id"]: r for r in rows if r.get("event") == "reserve"}
    closed = {r["id"]: r for r in rows if r.get("event") in {"settle", "fail"}}
    per = defaultdict(lambda: {"attempts": 0, "generate": 0, "count_tokens": 0, "other": 0, "settled": 0, "failed": 0,
                               "open": 0, "errors": Counter(), "prompt_tokens": 0, "candidates_tokens": 0,
                               "thoughts_tokens": 0, "cached_tokens": 0, "settled_usd": 0.0, "unverified_usd": 0.0,
                               "reserved_usd_sum": 0.0, "latency": [], "model_versions": Counter()})
    for rid, res in reserves.items():
        p = per[res["run"]]
        p["attempts"] += 1
        p[res.get("kind", "other")] += 1
        p["reserved_usd_sum"] += res.get("reserve_usd", 0.0)
        c = closed.get(rid)
        if c is None:
            p["open"] += 1
            p["unverified_usd"] += res.get("reserve_usd", 0.0)
            continue
        if c["event"] == "settle":
            p["settled"] += 1
            p["settled_usd"] += c.get("cost_usd", 0.0)
            for k in ("prompt_tokens", "candidates_tokens", "thoughts_tokens", "cached_tokens"):
                p[k] += int(c.get(k) or 0)
            if res.get("kind") == "generate":
                p["latency"].append(c.get("latency_ms", 0.0))
                p["model_versions"][str(c.get("model_version"))] += 1
        else:
            p["failed"] += 1
            p["unverified_usd"] += c.get("unverified_usd", 0.0)
            p["errors"][f"{c.get('error')}:{c.get('error_code', '')}"] += 1
    out = {"runs": {}, "halts": [r for r in rows if r.get("event") == "halt"]}
    tot = Counter()
    for run, p in sorted(per.items()):
        lat = p.pop("latency")
        p["errors"] = dict(p["errors"])
        p["model_versions"] = dict(p["model_versions"])
        p["generate_latency_ms_median"] = round(statistics.median(lat), 1) if lat else None
        p["generate_latency_ms_p95"] = pct(lat, 0.95)
        for k in ("settled_usd", "unverified_usd", "reserved_usd_sum"):
            p[k] = round(p[k], 6)
        out["runs"][run] = p
        for k in ("attempts", "generate", "count_tokens", "settled", "failed", "open", "prompt_tokens",
                  "candidates_tokens", "thoughts_tokens", "cached_tokens"):
            tot[k] += p[k]
        tot["settled_usd"] += p["settled_usd"]
        tot["unverified_usd"] += p["unverified_usd"]
    out["total"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in tot.items()}
    out["total"]["committed_usd"] = round(tot["settled_usd"] + tot["unverified_usd"], 6)
    (HERE / "budget_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["total"], ensure_ascii=False))
    for run, p in out["runs"].items():
        print(run, {k: p[k] for k in ("attempts", "generate", "count_tokens", "settled", "failed", "open", "errors",
                                      "settled_usd", "unverified_usd", "generate_latency_ms_median")})


if __name__ == "__main__":
    main()
