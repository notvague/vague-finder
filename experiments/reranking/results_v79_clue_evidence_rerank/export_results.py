"""원기록(runs/, Git 제외)에서 추적할 요약을 만든다. 모델 호출 없음."""
import csv, json
from pathlib import Path
R = Path(__file__).resolve().parent
RUNS = ("off_rep1", "off_rep2", "hits_rep1", "hits_rep2", "dsyn_rep1", "dsyn_rep2")
SETS = ("v06", "v09", "v10")
BAD = {"failed", "fallback", "budget_exhausted", "partial_failed", "model_partial"}
rows = {}
summary = {}
for run in RUNS:
    got = {}
    state = {}
    for s in SETS:
        p = R / "runs" / run / s / "search_eval_dev_detail.csv"
        if not p.exists():
            state[s] = "없음"; continue
        detail = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        bad = [r for r in detail if r["rerank_outcome"] in BAD or float(r.get("rerank_fell_back") or 0) > 0]
        state[s] = f"{len(detail)}건·실패 {len(bad)}"
        if bad:
            continue
        for r in detail:
            got[r["query_id"]] = r
            rows.setdefault(r["query_id"], {"query_id": r["query_id"], "set": s, "candidate_rank_off": ""})[f"{run}_rank"] = r.get("rerank_rank", "")
            if run == "off_rep1":
                rows[r["query_id"]]["candidate_rank_off"] = r.get("candidate_rank@30", "")
    f = lambda k: sum(float(r[k] or 0) > 0 for r in got.values())
    summary[run] = {"sets": state, "n": len(got), "hit@1": f("rerank_hit@1"), "hit@5": f("rerank_hit@5"), "hit@10": f("rerank_hit@10"),
                    "mrr": round(sum(float(r["rerank_mrr@10"] or 0) for r in got.values()) / max(1, len(got)), 4)}
cols = ["query_id", "set", "candidate_rank_off"] + [f"{r}_rank" for r in RUNS]
with (R / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for q in sorted(rows):
        w.writerow({c: rows[q].get(c, "") for c in cols})
led = [json.loads(l) for l in (R / "budget/ledger.jsonl").open() if l.strip()]
cost = {}
for x in led:
    if x.get("event") == "settle":
        key = x.get("run", "?").rsplit("_", 1)[0]
        cost[key] = round(cost.get(key, 0.0) + x.get("cost_usd", 0.0), 4)
summary["budget"] = {"settled_usd": round(sum(cost.values()), 4), "by_run": cost, "attempts": sum(1 for x in led if x.get("event") == "reserve"),
                     "failed_attempts": sum(1 for x in led if x.get("event") == "fail"), "halt": next((x.get("reason") for x in led if x.get("event") == "halt"), "")}
(R / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: (v.get("hit@10") if "hit@10" in v else v.get("settled_usd")) for k, v in summary.items()})
