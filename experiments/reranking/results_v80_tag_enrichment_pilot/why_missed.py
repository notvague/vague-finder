"""실패 질의 정답 곡의 단서가 어디까지 있는지 나눈다. 모델 호출 없음."""
import collections, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from pilot_lib import failed_query_gold, load_songs, old_tags
from src.retrieval.user_explanation import _norm, _term_in_tag
songs = load_songs(Path(sys.argv[1]))
ok = {json.loads(l)["song_id"]: json.loads(l) for l in (HERE / "runs/new_tags.jsonl").open(encoding="utf-8") if l.strip()}
sample = json.loads((HERE / "runs/sample.json").read_text(encoding="utf-8"))
SKIP = re.compile(r"남성|여성|남자|여자|보컬$|^발라드|^댄스|힙합|^록$|^인디|알앤비|R&B|걸그룹|보이그룹|솔로|그룹|듀오|듀엣|밴드$|\d|년대|앨범|표지|커버|사진|일러스트|OST")
ent = {}
for s in ("v06", "v09", "v10"):
    ent.update(json.load((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"])
gold = failed_query_gold(ROOT / "experiments/reranking/results_v79_clue_evidence_rerank/per_query.csv", ROOT / "experiments/reranking")
flat = lambda t: [x for v in t.values() for x in v]
buckets = collections.Counter(); names = collections.defaultdict(list); tagless = 0; fq = 0
for qid, ids in gold.items():
    e = ent[qid]; q = _norm(e["query"])
    for sid in ids:
        if sid not in sample["F"]:
            continue
        fq += 1
        terms = [t for t in e["analysis"].get("korean_tags", []) if len(_norm(t)) >= 2 and not SKIP.search(t) and _norm(t) in q]
        tagless += not terms
        row = songs[sid]
        o = flat(old_tags(row)); n = flat(ok[sid]["tags"])
        src = _norm(" ".join([row["lyrics_data"].get("full_lyrics") or "", row["metadata"].get("album_description") or "", *((row.get("comments") or {}).get("melon") or []), *((row.get("comments") or {}).get("youtube") or [])]))
        for t in terms:
            if any(_term_in_tag(_norm(t), x) == "tag" for x in o):
                k = "기존 태그에 있음"
            elif any(_term_in_tag(_norm(t), x) == "tag" for x in n):
                k = "새 태그로 얻음"
            elif _norm(t) in src:
                k = "원자료에 있으나 모델이 안 뽑음"
            else:
                k = "원자료에 글자로 없음"
            buckets[k] += 1; names[k].append(t)
print("정답 곡-질의 쌍", fq, "단서 없는 쌍", tagless, "단서", sum(buckets.values()))
for k, v in buckets.most_common():
    print(f"  {k}: {v} {names[k][:26]}")
out = {"pairs": fq, "pairs_without_clue": tagless, "buckets": {k: {"count": v, "clues": names[k]} for k, v in buckets.items()}}
(HERE / "why_missed.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
