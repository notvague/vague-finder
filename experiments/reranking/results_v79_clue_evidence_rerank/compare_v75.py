"""로컬 Qdrant 후보(이번 A 조건)가 v75 A(서버 Qdrant)와 같은지 질의별로 맞춘다. 모델 호출 없음."""
import csv, glob, json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
v75 = Path(sys.argv[1])
ours = {}
for s in ("v06", "v09", "v10"):
    for r in csv.DictReader((R / "runs/offline_syn_off" / s / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")):
        ours[r["query_id"]] = r
theirs = {}
for path in glob.glob(str(v75 / "**/search_eval_dev_detail.csv"), recursive=True):
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        theirs[r["query_id"]] = r
print("v75 행", len(theirs), "이번 행", len(ours))
def rk(r):
    try:
        return int(float(r["candidate_rank@30"]))
    except (TypeError, ValueError):
        return None
same_rank = [q for q in ours if q in theirs and rk(ours[q]) == rk(theirs[q])]
diff = [(q, rk(theirs[q]), rk(ours[q])) for q in ours if q in theirs and rk(ours[q]) != rk(theirs[q])]
dg = [q for q in ours if q in theirs and ours[q].get("candidate_digest") != theirs[q].get("candidate_digest")]
print("정답 후보 순위 같음", len(same_rank), "다름", diff)
print("후보 회수 v75", sum(rk(r) is not None for r in theirs.values()), "이번", sum(rk(r) is not None for r in ours.values()))
print("후보 목록 지문 다름", len(dg), dg[:12])
