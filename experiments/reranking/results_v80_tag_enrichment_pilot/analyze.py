"""파일럿 판정(RUN_INFO §4). 모델 호출 없음. 미완료·실패가 있으면 그대로 보고한다."""
import collections, json, re, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from pilot_lib import TAG_LIMITS, evaluation_clues, failed_query_gold, load_songs, old_tags
from src.retrieval.user_explanation import _term_in_tag, _norm
songs = load_songs(Path(sys.argv[1]))
recs = {}
for line in (HERE / "runs/new_tags.jsonl").open(encoding="utf-8"):
    if line.strip():
        r = json.loads(line); recs[r["song_id"]] = r
sample = json.loads((HERE / "runs/sample.json").read_text(encoding="utf-8"))
ok = {k: r for k, r in recs.items() if r["status"] == "ok"}
expected = len(sample["F"]) + len(sample["R"])
failed = [k for k, r in recs.items() if r["status"] != "ok"]
missing = [k for ids in sample.values() for k in ids if k not in recs]
flat = lambda tags: [t for values in tags.values() for t in values]
# 단서: pilot_lib.evaluation_clues(속성 낱말·표지 묘사 제외)
ent = {}
for s in ("v06", "v09", "v10"):
    ent.update(json.load((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"])
gold = failed_query_gold(ROOT / "experiments/reranking/results_v79_clue_evidence_rerank/per_query.csv", ROOT / "experiments/reranking")
clues = collections.defaultdict(list)
for qid, ids in gold.items():
    e = ent[qid]
    terms = evaluation_clues(e["query"], e["analysis"])
    for song_id in ids:
        if song_id in sample["F"]:
            clues[song_id].extend(terms)
def hit(term, tags):
    return any(_term_in_tag(_norm(term), t) == "tag" for t in tags)
rows = []
F_ok = [k for k in sample["F"] if k in ok]; R_ok = [k for k in sample["R"] if k in ok]
old_f = new_f = old_r = new_r = n_f = n_r = 0
per = []
for song_id in F_ok:
    o = flat(old_tags(songs[song_id])); u = o + flat(ok[song_id]["tags"])
    for term in clues[song_id]:
        n_f += 1; a = hit(term, o); b = hit(term, u); old_f += a; new_f += b
        if b and not a:
            per.append((term, songs[song_id]["metadata"]["title"]))
        for other in R_ok:
            oo = flat(old_tags(songs[other])); uu = oo + flat(ok[other]["tags"])
            n_r += 1; old_r += hit(term, oo); new_r += hit(term, uu)
cov = lambda a, n: round(a / n, 4) if n else None
lift = lambda f, r: round(f / r, 2) if r else None
df = collections.Counter()
for row in songs.values():
    df.update(set(flat(old_tags(row))))
added = []; source_hit = 0
for song_id, r in ok.items():
    o = {_norm(t) for t in flat(old_tags(songs[song_id]))}
    row = songs[song_id]
    src = _norm(" ".join([row["lyrics_data"].get("full_lyrics") or "", row["metadata"].get("album_description") or "", *((row.get("comments") or {}).get("melon") or []), *((row.get("comments") or {}).get("youtube") or [])]))
    for t in flat(r["tags"]):
        if _norm(t) not in o:
            added.append(t); source_hit += _norm(t) in src
generic = sum(df[t] > 0.05 * len(songs) for t in added)
led = [json.loads(l) for l in (HERE / "budget/ledger.jsonl").open() if l.strip()]
settle = [x for x in led if x.get("event") == "settle" and x.get("kind") == "generate"]
cost = sum(x.get("cost_usd", 0.0) for x in led if x.get("event") == "settle")
per_song = cost / max(1, len(ok))
result = {
    "songs": {"expected": expected, "ok": len(ok), "failed": failed, "missing": missing, "retries": sum(r["attempts"] - 1 for r in recs.values()), "dropped_tags": sum(r.get("dropped", 0) for r in ok.values())},
    "tags_per_song": {"old": round(st.mean(len(flat(old_tags(songs[k]))) for k in ok), 1), "new": round(st.mean(len(flat(ok[k]["tags"])) for k in ok), 1), "added": round(len(added) / max(1, len(ok)), 1)},
    "clue_cover": {"clues": n_f, "F_old": cov(old_f, n_f), "F_union": cov(new_f, n_f), "R_old": cov(old_r, n_r), "R_union": cov(new_r, n_r), "lift_old": lift(cov(old_f, n_f), cov(old_r, n_r)), "lift_union": lift(cov(new_f, n_f), cov(new_r, n_r))},
    "added": {"count": len(added), "generic_share": cov(generic, len(added)), "literal_in_source_share": cov(source_hit, len(added)), "new_clue_hits": per[:30]},
    "cost": {"settled_usd": round(cost, 4), "per_song": round(per_song, 6), "full_5143_standard": round(per_song * 5143, 2), "full_5143_batch": round(per_song * 5143 / 2, 2), "input_tokens_median": st.median(x.get("prompt_tokens", 0) for x in settle) if settle else None, "output_tokens_median": st.median(x.get("candidates_tokens", 0) for x in settle) if settle else None, "thoughts_tokens_median": st.median(x.get("thoughts_tokens", 0) or 0 for x in settle) if settle else None, "latency_ms_median": st.median(x.get("latency_ms", 0) for x in settle) if settle else None},
}
c = result["clue_cover"]
result["gates"] = {
    "G1_stability": len(failed) + len(missing) <= 1,
    "G2_clue_cover": (c["F_union"] - c["F_old"]) >= 0.10 if n_f else False,
    "G3_discrimination": (c["lift_union"] or 0) >= (c["lift_old"] or 0),
    "G4_generic": (result["added"]["generic_share"] or 1) <= 0.40,
    "G5_cost": result["cost"]["full_5143_standard"] <= 20,
    "G6_human": "대기",
}
(HERE / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False)[:2600])
# 사람 검토 표: 무작위 R 12곡(F는 단서를 안 곡이라 넣지 않는다)
lines = ["# 태그 재생성 검토 표본 (무작위 12곡)", "", "기존 태그와 **새로 추가된 태그**입니다. 곡을 아는 사람이 보고 아래를 판단합니다.", "", "- 추가된 태그가 그 곡을 제대로 설명하는가", "- 틀리거나 근거 없어 보이는 태그가 있는가", "- 기존보다 곡을 더 잘 가르는가", ""]
for song_id in R_ok[:12]:
    row = songs[song_id]; o = old_tags(row); oset = {_norm(t) for t in flat(o)}
    lines += [f"## {row['metadata']['title']} / {'·'.join(row['metadata']['artist'])}", "", "| 칸 | 기존 | 추가 |", "|---|---|---|"]
    for name in TAG_LIMITS:
        add = [t for t in ok[song_id]["tags"][name] if _norm(t) not in oset]
        lines.append(f"| {name.replace('_tags','')} | {' '.join(o[name])} | {' '.join(add)} |")
    lines.append("")
(HERE / "review_sample.md").write_text("\n".join(lines), encoding="utf-8")
