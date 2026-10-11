"""run1 detail + candidate_drop + split_v10 → 질의별 표(dev_first_results.csv)와 층별 집계.

엄격 지표는 원래 정답만, 확장 지표는 allowed_v10.csv의 허용 정답까지 센다.
(v10은 queries.json에 아직 없어 src.eval.relaxed_metrics를 쓸 수 없다 — 0.7.0 병합 뒤에는 그쪽으로 잰다.)

  venv/bin/python experiments/reranking/results_v38_v10_dev_first/summarize.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

OUT = Path(__file__).resolve().parent


def _s(v: object) -> str:
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def _norm(s: str) -> str:
    return re.sub(r"[\s()\[\]\-_.,'\"!?]+", "", s.lower())


def _rank(x: str) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def main() -> None:
    songs: Dict[str, tuple] = {}
    with open("data/all_songs.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m = d.get("metadata", {})
            songs[str(d["song_id"])] = (_s(m.get("title")), _s(m.get("artist")), _s(m.get("release_date"))[:4])
    tr = {r["query_id"]: r for r in csv.DictReader(open(OUT / "candidate_drop_dev.csv", encoding="utf-8"))}
    sp = {r["query_id"]: r for r in csv.DictReader(open(OUT / "split_v10.csv", encoding="utf-8"))}
    det = list(csv.DictReader(open(OUT / "run1/search_eval_dev_detail.csv", encoding="utf-8-sig")))
    allowed = {r["query_id"]: set(r["allowed_ids"].split("|")) - {""}
               for r in csv.DictReader(open(OUT / "allowed_v10.csv", encoding="utf-8"))}
    cache = json.load(open("experiments/reranking/analysis_cache_v10_dev.json", encoding="utf-8"))["entries"]

    def label(sid: str) -> str:
        return f"{songs[sid][0]} - {songs[sid][1]} ({songs[sid][2]})" if sid in songs else sid

    out: List[Dict[str, object]] = []
    for r in det:
        q, pid = r["query_id"], r["relevant_ids"]
        t, s = tr[q], sp[q]
        raw = r["rerank_top_ids"]
        top = [str(x) for x in (json.loads(raw) if raw.startswith("[") else re.split(r"[|,;\s]+", raw)) if str(x)][:10]
        at = songs[pid]
        same = [x for x in top if x != pid and x in songs and _norm(songs[x][0]) == _norm(at[0])
                and _norm(songs[x][1])[:3] == _norm(at[1])[:3]]
        ok = {pid} | allowed.get(q, set())
        relaxed = next((i + 1 for i, x in enumerate(top) if x in ok), None)
        entry = cache.get(q, {})
        fallback = bool(entry.get("is_fallback") or entry.get("fallback"))
        out.append({
            "query_id": q, "corpus": s["corpus"], "modality": s["modality_focus"], "tier": s["tier"],
            "category": s["category"], "clue_type": s["clue_type"], "scope": s["target_scope"],
            "analysis_fallback": int(fallback), "stage": t["stage"],
            "text_rank": t["text_hybrid_rank"], "image_rank": t["image_rank"], "audio_rank": t["audio_rank"],
            "pool_rank_deep": t["pool_rank_deep"], "cand_rank": r["candidate_rank@30"],
            "base_rank": r["baseline_rank"], "final_rank": r["rerank_rank"],
            "relaxed_rank": relaxed or "", "allowed_ids": "|".join(sorted(allowed.get(q, set()))), "rerank_outcome": r["rerank_outcome"],
            "answer": label(pid), "top1": label(top[0]) if top else "", "same_song_other_id": "|".join(same),
            "weights": r["modality_weights"], "query": r["query"],
        })
    with open(OUT / "dev_first_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    def cand(g): return sum(_rank(o["cand_rank"]) is not None for o in g)
    def hit(g, k, col="final_rank"): return sum((_rank(str(o[col])) or 999) <= k for o in g)
    print(f"전체 n={len(out)} 후보30={cand(out)} Hit@1={hit(out,1)} Hit@5={hit(out,5)} Hit@10={hit(out,10)}")
    mrr = lambda col: sum(1 / r for o in out if (r := _rank(str(o[col]))) and r <= 10) / len(out)
    print(f"확장(허용 정답 포함) Hit@1={hit(out,1,'relaxed_rank')} Hit@5={hit(out,5,'relaxed_rank')} "
          f"Hit@10={hit(out,10,'relaxed_rank')} · MRR@10 엄격 {mrr('final_rank'):.3f} 확장 {mrr('relaxed_rank'):.3f}")
    print("허용 정답으로 순위가 달라진 질의:", [(o["query_id"], o["final_rank"], o["relaxed_rank"]) for o in out
                                  if str(o["final_rank"]) != str(o["relaxed_rank"])])
    print("후보 안인데 10위 밖:", [(o["query_id"], o["cand_rank"], o["final_rank"]) for o in out
                             if _rank(o["cand_rank"]) and (_rank(o["final_rank"]) or 999) > 10])
    print("같은 곡 다른 ID가 top-10:", [(o["query_id"], o["same_song_other_id"], o["final_rank"]) for o in out if o["same_song_other_id"]])
    print("분석 폴백:", [o["query_id"] for o in out if o["analysis_fallback"]])
    print("단계:", Counter(o["stage"] for o in out), "· 리랭킹:", Counter(o["rerank_outcome"] for o in out))
    for col in ("corpus", "modality", "tier", "category", "clue_type", "scope"):
        print(col)
        for k in sorted({o[col] for o in out}):
            g = [o for o in out if o[col] == k]
            print(f"   {k:18s} n={len(g):2d} 후보30={cand(g):2d} Hit@1={hit(g,1):2d} Hit@10={hit(g,10):2d}")


if __name__ == "__main__":
    main()
