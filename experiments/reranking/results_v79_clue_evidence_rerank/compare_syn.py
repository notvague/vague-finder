"""동의어 확장의 후보 변화(모델 호출 없음). 끔·켬 실행의 정답 후보 순위와 sparse 입력이 달라진 질의를 비교한다."""
import csv, json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[2]))
from src.retrieval.user_explanation import SHORT_CLUE_VALUES
def load(arm):
    rows = {}
    for s in ("v06", "v09", "v10"):
        for r in csv.DictReader((R / "runs" / f"offline_syn_{arm}" / s / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")):
            rows[r["query_id"]] = r
    return rows
off, on = load("off"), load("on")
assert set(off) == set(on) and len(off) == 176, (len(off), len(on))
def rank(r):
    try:
        return int(float(r["candidate_rank@30"]))
    except (TypeError, ValueError):
        return None
cands = lambda rows: sum(rank(r) is not None for r in rows.values())
affected = {}
for s in ("v06", "v09", "v10"):
    entries = json.load((R.parent / "results_v49_vertex_baseline" / f"analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"]
    for qid, e in entries.items():
        tags = [t for t in e["analysis"].get("korean_tags", []) if str(t).strip() in SHORT_CLUE_VALUES]
        if tags:
            affected[qid] = tags
print("후보 회수 끔", cands(off), "켬", cands(on), "| sparse 입력이 바뀌는 질의", len(affected), affected)
gain = [q for q in off if rank(off[q]) is None and rank(on[q]) is not None]
lost = [q for q in off if rank(off[q]) is not None and rank(on[q]) is None]
moved = [(q, rank(off[q]), rank(on[q])) for q in off if rank(off[q]) is not None and rank(on[q]) is not None and rank(off[q]) != rank(on[q])]
digest = [q for q in off if off[q].get("candidate_digest") != on[q].get("candidate_digest")]
print("얻음", gain, "잃음", lost, "순위 변화", moved)
print("후보 목록이 달라진 질의", len(digest), digest, "| 영향 질의 밖에서 달라짐", [q for q in digest if q not in affected])
