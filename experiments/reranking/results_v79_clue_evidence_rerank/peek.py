"""끝난 실행의 지표와 비용을 본다. 판정은 analyze.py가 한다."""
import csv, json, sys, collections
from pathlib import Path
R = Path(__file__).resolve().parent
def load(run):
    rows = {}
    for s in ("v06", "v09", "v10"):
        p = R / "runs" / run / s / "search_eval_dev_detail.csv"
        if not p.exists():
            return None
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            rows[r["query_id"]] = r
    return rows
fl = lambda v: float(v or 0) > 0
base = None
for run in ("off_rep1", "hits_rep1", "hits_rep2", "off_rep2", "syn_rep1", "syn_rep2"):
    d = load(run)
    if not d:
        continue
    print(run, "n", len(d), "hit1", sum(fl(r["rerank_hit@1"]) for r in d.values()), "hit5", sum(fl(r["rerank_hit@5"]) for r in d.values()), "hit10", sum(fl(r["rerank_hit@10"]) for r in d.values()),
          "mrr", round(sum(float(r["rerank_mrr@10"] or 0) for r in d.values()) / len(d), 4), "cand", sum(fl(r["candidate_recall@30"]) for r in d.values()),
          dict(collections.Counter(r["rerank_outcome"] for r in d.values())), "ms중앙", round(sorted(float(r.get("rerank_only_ms") or 0) for r in d.values())[len(d) // 2]))
    h = {q for q, r in d.items() if fl(r["rerank_hit@10"])}
    if base is None:
        base = h
    else:
        print("   off_rep1 대비 얻음", sorted(h - base), "잃음", sorted(base - h))
led = [json.loads(l) for l in (R / "budget/ledger.jsonl").open() if l.strip()]
c = collections.Counter()
for x in led:
    if x.get("event") == "settle":
        c[x.get("run", "?").rsplit("_", 1)[0]] += x.get("cost_usd", 0.0)
print({k: round(v, 3) for k, v in c.items()}, "합", round(sum(c.values()), 3), "실패", collections.Counter(x.get("run") for x in led if x.get("event") == "fail"))
