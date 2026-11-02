"""단서별 검색의 후보 변화(§9 기준). 모델 호출 없음."""
import csv
from pathlib import Path
R = Path(__file__).resolve().parent
fl = lambda v: float(v or 0) > 0
def load(run):
    o = {}
    for s in ("v06", "v09", "v10"):
        for r in csv.DictReader((R / "runs" / run / s / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")):
            o[r["query_id"]] = r
    return o
base, clue, a1, a2 = load("offline_syn_off"), load("offline_clue_on"), load("off_rep1"), load("off_rep2")
assert set(base) == set(clue) and len(base) == 176
def rk(r):
    try:
        return int(float(r["candidate_rank@30"]))
    except (TypeError, ValueError):
        return None
print("후보 회수 끔", sum(rk(r) is not None for r in base.values()), "켬", sum(rk(r) is not None for r in clue.values()))
gain = sorted(q for q in base if rk(base[q]) is None and rk(clue[q]) is not None)
lost = sorted(q for q in base if rk(base[q]) is not None and rk(clue[q]) is None)
print("얻음", [(q, rk(clue[q])) for q in gain]); print("잃음", [(q, rk(base[q])) for q in lost])
success = {q for q in a1 if fl(a1[q]["rerank_hit@10"]) and fl(a2[q]["rerank_hit@10"])}
print("A1·A2 모두 성공인데 후보에서 잃음", sorted(set(lost) & success))
moved = [(q, rk(base[q]), rk(clue[q])) for q in base if rk(base[q]) is not None and rk(clue[q]) is not None and rk(base[q]) != rk(clue[q])]
up = [m for m in moved if m[2] < m[1]]; down = [m for m in moved if m[2] > m[1]]
print("정답 후보 순위 변화: 오름", len(up), "내림", len(down), "같음", 176 - len(gain) - len(lost) - len(moved) - sum(rk(base[q]) is None and rk(clue[q]) is None for q in base))
print("크게 내림(5칸 이상)", sorted(down, key=lambda m: m[1] - m[2])[:12])
print("크게 오름(5칸 이상)", sorted(up, key=lambda m: m[2] - m[1])[:12])
changed = [q for q in base if base[q].get("candidate_digest") != clue[q].get("candidate_digest")]
print("후보 목록이 달라진 질의", len(changed))
print("기준:", "통과" if len(gain) > len(lost) and not (set(lost) & success) else "미통과")
