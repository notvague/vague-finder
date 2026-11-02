"""B에서 순위가 달라진 질의를 읽는다. 모델 호출 없음."""
import csv, json, os, sys, types
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[2]))
from analyze import _song_loader
from src.backend.schemas.query import QueryAnalysis
from src.retrieval.user_explanation import clue_evidence_for_rerank
songs = _song_loader(Path(sys.argv[1]))
titles = {}
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        r = json.loads(line); titles[str(r["song_id"])] = f'{r["metadata"]["title"]}/{"·".join(r["metadata"]["artist"])}'
analyses = {}
for s in ("v06", "v09", "v10"):
    for q, e in json.load((R.parent / f"results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"].items():
        analyses[q] = QueryAnalysis.model_validate(e["analysis"])
def load(run):
    o = {}
    for s in ("v06", "v09", "v10"):
        for r in csv.DictReader((R / "runs" / run / s / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")):
            o[r["query_id"]] = r
        for r in csv.DictReader((R / "runs" / run / s / "search_eval_dev_top10.csv").open(encoding="utf-8-sig")):
            o[r["query_id"]]["top"] = r["rerank_top_ids"].split("|")
    return o
A, B = load("off_rep1"), load("hits_rep1")
def rk(r):
    try:
        return int(float(r["rerank_rank"]))
    except (TypeError, ValueError):
        return None
def stat(a, sid):
    if sid not in songs:
        return "자료없음"
    c = clue_evidence_for_rerank(a.original_query, a, songs[sid])
    return f'확인{len(c.get("found", []))}·부분{len(c.get("partial", []))}·못찾음{len(c.get("not_found", []))}'
for q in ("n057", "q202", "t071", "t107", "t044"):
    a = analyses[q]; gold = A[q]["relevant_ids"].split("|")[0]
    print(f'\n[{q}] {a.original_query[:78]}')
    print(f'  순위 A {rk(A[q])} → B {rk(B[q])} | 태그 {a.korean_tags}')
    print(f'  정답 {titles.get(gold)} : {stat(a, gold)}')
    for sid in B[q]["top"][:3]:
        print(f'  B top: {titles.get(sid, sid)[:34]} : {stat(a, sid)}')
# 전체 경향: 질의마다 정답과 B 1위의 단서 확인 수
up = down = same = 0; diff = []
gold_better = gold_worse = gold_equal = 0
for q, r in B.items():
    ra, rb = rk(A[q]), rk(r)
    if ra is None and rb is None:
        continue
    ra, rb = ra or 99, rb or 99
    up += rb < ra; down += rb > ra; same += rb == ra
print(f'\n정답 순위 변화(양쪽 중 하나라도 10위 안): 오름 {up} · 내림 {down} · 같음 {same}')
for q, r in B.items():
    a = analyses[q]; gold = A[q]["relevant_ids"].split("|")[0]
    if gold not in songs or not r["top"] or r["top"][0] == gold or r["top"][0] not in songs:
        continue
    def n(sid):
        c = clue_evidence_for_rerank(a.original_query, a, songs[sid])
        return len(c.get("found", [])) + len(c.get("partial", []))
    gn, tn = n(gold), n(r["top"][0])
    gold_better += gn > tn; gold_worse += gn < tn; gold_equal += gn == tn
print(f'정답이 1위가 아닌 질의에서 단서 확인 수: 정답이 많음 {gold_better} · 같음 {gold_equal} · 1위 곡이 많음 {gold_worse}')
