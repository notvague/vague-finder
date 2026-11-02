"""D(A + 동의어) 판정. RUN_INFO §7 기준. 모델 호출 없음. 리랭킹 실패가 섞인 D2 세트는 쓰지 않는다."""
import csv, json
from pathlib import Path
R = Path(__file__).resolve().parent
CHANGED = ["q210", "q215", "q308", "q312", "c707", "n006", "n021", "t083", "t100"]
fl = lambda v: float(v or 0) > 0
BAD = {"failed", "fallback", "budget_exhausted", "partial_failed", "model_partial"}
def load(run):
    out, status = {}, {}
    for s in ("v06", "v09", "v10"):
        p = R / "runs" / run / s / "search_eval_dev_detail.csv"
        if not p.exists():
            status[s] = "없음"; continue
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        bad = [r["query_id"] for r in rows if r["rerank_outcome"] in BAD or fl(r.get("rerank_fell_back"))]
        status[s] = f"{len(rows)}건·실패 {len(bad)}"
        if not bad:
            for r in rows:
                out[r["query_id"]] = r
    return out, status
def rank(r):
    try:
        return int(float(r["rerank_rank"]))
    except (TypeError, ValueError):
        return None
runs = {n: load(n) for n in ("off_rep1", "off_rep2", "dsyn_rep1", "dsyn_rep2")}
for n, (rows, st) in runs.items():
    print(n, st)
A1, A2, D1, D2 = (runs[n][0] for n in ("off_rep1", "off_rep2", "dsyn_rep1", "dsyn_rep2"))
m = lambda d: {"n": len(d), "hit1": sum(fl(r["rerank_hit@1"]) for r in d.values()), "hit5": sum(fl(r["rerank_hit@5"]) for r in d.values()), "hit10": sum(fl(r["rerank_hit@10"]) for r in d.values()), "mrr": round(sum(float(r["rerank_mrr@10"] or 0) for r in d.values()) / max(1, len(d)), 4)}
for n, d in (("A1", A1), ("A2", A2), ("D1", D1)):
    print(n, m(d))
print("\n후보가 달라진 9건: 정답 순위 A1 / A2 / D1 / D2")
for q in CHANGED:
    print(" ", q, rank(A1[q]), rank(A2[q]), rank(D1[q]), rank(D2[q]) if q in D2 else "—", "|", A1[q]["query"][:44])
h = lambda d: {q for q, r in d.items() if fl(r["rerank_hit@10"])}
print("D1 Hit@10 얻음(A1·A2 모두 실패):", sorted(h(D1) - h(A1) - h(A2)), "잃음(A1·A2 모두 성공):", sorted((h(A1) & h(A2)) - h(D1)))
other = [q for q in D1 if q not in CHANGED]
print("입력이 같은 167건의 출력 변동(정답 순위가 다른 질의) D1↔A1:", [q for q in other if rank(D1[q]) != rank(A1[q])][:14], "| A1↔A2:", [q for q in A1 if rank(A1[q]) != rank(A2[q])][:14])
common = [q for q in D2 if q in D1]
print("D1↔D2(반복, 완료 세트만)", len(common), "건 정답 순위 다름:", [q for q in common if rank(D1[q]) != rank(D2[q])][:12])
ledger = [json.loads(l) for l in (R / "budget/ledger.jsonl").open() if l.strip()]
print("확정 비용", round(sum(x.get("cost_usd", 0.0) for x in ledger if x.get("event") == "settle"), 3), "시도", sum(1 for x in ledger if x.get("event") == "reserve"))
