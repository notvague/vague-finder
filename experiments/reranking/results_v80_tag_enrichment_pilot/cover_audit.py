"""평가 단서 중 표지 묘사에서 온 것을 찾는다. 모델 호출 없음."""
import csv, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from clap_labels import LABELS
from pilot_lib import failed_query_gold
from src.retrieval.modality_queries import has_explicit_visual_clue, text_outside_artwork
from src.retrieval.user_explanation import _norm, _said_in_query
SKIP = re.compile(r"남성|여성|남자|여자|보컬$|^발라드|^댄스|힙합|^록$|^인디|알앤비|R&B|걸그룹|보이그룹|솔로|그룹|듀오|듀엣|밴드$|\d|년대|앨범|표지|커버|사진|일러스트|OST")
ent = {}
for s in ("v06", "v09", "v10"):
    ent.update(json.load((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"])
sample = json.loads((HERE / "runs/sample.json").read_text(encoding="utf-8"))
gold = failed_query_gold(ROOT / "experiments/reranking/results_v79_clue_evidence_rerank/per_query.csv", ROOT / "experiments/reranking")
print("== 파일럿 F 단서 중 표지 질의에서 온 것")
for qid, ids in gold.items():
    e = ent[qid]; q = e["query"]; visual = bool(e["analysis"].get("has_visual_clue")) or has_explicit_visual_clue(q)
    terms = [t for t in e["analysis"].get("korean_tags", []) if len(_norm(t)) >= 2 and not SKIP.search(t) and _norm(t) in _norm(q)]
    n = sum(1 for g in ids if g in sample["F"])
    if not n or not terms or not visual:
        continue
    outside = text_outside_artwork(q)
    print(qid, f"곡{n}", [(t, "노래절" if _said_in_query(t, outside) else "표지/불명") for t in terms], "|", q[:74])
print("== CLAP 평가 사례(dev 전체) 중 표지 질의")
cases = json.loads((HERE / "runs/clap_cases.json").read_text(encoding="utf-8"))
words = {l[0]: l[3] for l in LABELS}
seen = set(); bad = 0
for c in cases:
    e = ent[c["query_id"]]; q = e["query"]
    if not (e["analysis"].get("has_visual_clue") or has_explicit_visual_clue(q)):
        continue
    outside = text_outside_artwork(q)
    inside = any(_norm(w) in _norm(outside) for w in words[c["label"]])
    bad += not inside
    print(" ", c["query_id"], c["label"], "노래절" if inside else "표지/불명", "tagged" if c["tagged"] else "태그없음", "top15" if c["top15"] else "-", "|", q[:62])
print("표지/불명 사례", bad, "/", len(cases))
