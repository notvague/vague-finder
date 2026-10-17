"""v66 후1 실제 실행의 최종 10위 밖 질의 → 단계 진단표 + 사람 검토용 자료 묶음. 모델 호출 없음(서버 Qdrant 읽기만).

읽는 것: v66 `live/live_after_server_rep1` detail(실제 Vertex 최종 순위) · v66 후2 detail(같음 확인) · 이 폴더 `trace/trace_on.csv` ·
`runs/on_rep1` 기록(보호·정책 생략·리랭커 그룹) · v49 분석 캐시 · 서버 Qdrant payload(텍스트 색인 · Context 사실 · 오디오·이미지 점 존재) ·
가사 스냅샷(가사 존재) · `data/all_songs.jsonl`(메타) · v50 `failure_diagnosis.csv`.

산출:
- `diagnosis/failure_stage_table.csv`(추적, 곡 ID·순위·코드만) — 정답 곡마다 한 줄, 질의 대표 줄 표시
- `diagnosis_private/review_bundle.jsonl`(Git 제외) — 질의 원문·분석·정답 곡 자료. 사람 검토용

    venv/bin/python build_failure_table.py --v66 <v66 결과 폴더>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN_ROOT = Path("/Users/ozzing727/Desktop/Developer/졸작-RISE/vague-finder")
SETS = ("v06", "v09", "v10")
QDRANT = "http://127.0.0.1:6370"
SNAPSHOT = Path("/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot")
STAGE_ORDER = {"후보 안": 0, "보정 뒤 절단": 1, "융합 절단": 2, "경로 밖": 3}


def detail(run: Path) -> Dict[str, Dict[str, str]]:
    out = {}
    for s in SETS:
        for r in csv.DictReader(open(run / f"{s}_dev" / "search_eval_dev_detail.csv", encoding="utf-8-sig")):
            r["set"] = s
            out[r["query_id"]] = r
    return out


def scroll(collection: str, ids: List[str] | None = None, payload: Any = True) -> List[dict]:
    if ids is not None:
        r = requests.post(f"{QDRANT}/collections/{collection}/points", json={"ids": [int(i) for i in ids],
                                                                             "with_payload": payload, "with_vector": False})
        r.raise_for_status()
        return r.json()["result"]
    out, offset = [], None
    while True:
        body = {"limit": 1000, "with_payload": payload, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT}/collections/{collection}/points/scroll", json=body)
        r.raise_for_status()
        res = r.json()["result"]
        out += res["points"]
        offset = res.get("next_page_offset")
        if offset is None:
            return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v66", type=Path, required=True)
    args = ap.parse_args()
    live1 = detail(args.v66 / "live/live_after_server_rep1")
    live2 = detail(args.v66 / "live/live_after_server_rep2")
    if len(live1) != 176 or set(live1) != set(live2):
        raise SystemExit("v66 detail 건수·질의 집합 이상")
    if any(live1[q]["rerank_top_ids"] != live2[q]["rerank_top_ids"] for q in live1):
        raise SystemExit("v66 후1 ≠ 후2")

    def rank(r: Dict[str, str], col: str) -> int | None:
        return int(float(r[col])) if r.get(col) else None

    failures = [q for q, r in live1.items() if not (rank(r, "rerank_rank") and rank(r, "rerank_rank") <= 10)]
    # 평가기 rerank_rank를 최종 top-10 ID와 정답으로 다시 계산해 맞춰 본다
    for q, r in live1.items():
        rel = {x for x in r["relevant_ids"].split("|") if x}
        top = r["rerank_top_ids"].split("|")[:10]
        hit = any(x in rel for x in top)
        if hit != (q not in failures):
            raise SystemExit(f"{q}: top-10 재계산과 rerank_rank 불일치")
    out_pool = [q for q in failures if not rank(live1[q], "candidate_rank@30")]
    in_pool = [q for q in failures if rank(live1[q], "candidate_rank@30")]
    print(f"최종 10위 밖 {len(failures)} = 후보 밖 {len(out_pool)} + 후보 내 {len(in_pool)}", file=sys.stderr)

    trace: Dict[str, List[dict]] = {}
    for r in csv.DictReader(open(HERE / "trace/trace_on.csv", encoding="utf-8")):
        trace.setdefault(r["query_id"], []).append(r)
    trace_skip: Dict[str, Dict[str, dict]] = {}
    for r in csv.DictReader(open(HERE / "trace/trace_skip.csv", encoding="utf-8")):
        trace_skip.setdefault(r["query_id"], {})[r["song_id"]] = r
    runrows: Dict[str, dict] = {}
    for s in SETS:
        for line in open(HERE / f"runs/on_rep1/{s}_dev/v67_queries.jsonl", encoding="utf-8"):
            d = json.loads(line)
            runrows[d["query_id"]] = d
    caches = {}
    for s in SETS:
        caches.update(json.loads((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json")
                                 .read_text(encoding="utf-8"))["entries"])
    v50 = {r["query_id"]: r for r in csv.DictReader(open(ROOT / "experiments/reranking/results_v50_vertex_context/failure_diagnosis.csv",
                                                          encoding="utf-8-sig"))}
    v50_review = {r["query_id"]: r for r in csv.DictReader(open(ROOT / "experiments/reranking/results_v50_vertex_context/failure_review.csv",
                                                                 encoding="utf-8-sig"))}

    gold_ids = sorted({sid for q in failures for sid in live1[q]["relevant_ids"].split("|") if sid})
    text_payload = {str(p["id"]): p["payload"] for p in scroll("vaguefinder-text-1024-koe5__dev", gold_ids)}
    image_ids = {str(p["id"]) for p in scroll("vaguefinder-image-768__dev", gold_ids, payload=False)}
    audio_ids = {str(p["id"]) for p in scroll("vaguefinder-audio-512__dev", gold_ids, payload=False)}
    facts = Counter()
    fact_text: Dict[str, List[str]] = {}
    for p in scroll("vaguefinder-context-dense-1024-koe5__dev", None, payload=True):
        sid = str(p["payload"].get("song_id"))
        facts[sid] += 1
        if sid in gold_ids:
            fact_text.setdefault(sid, []).append(json.dumps({k: v for k, v in p["payload"].items()
                                                             if k in ("relation", "fact_text", "text", "subject", "object", "kind",
                                                                      "category", "evidence", "media_name")},
                                                            ensure_ascii=False)[:300])
    profiles = {str(p["payload"].get("song_id")) for p in scroll("vaguefinder-context-sparse-bm25__dev", None, payload=["song_id"])}
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    docs = pickle.loads(gzip.decompress((SNAPSHOT / manifest["file"]["name"]).read_bytes()))
    lyrics = {str(d.get("song_id")): len(str((d.get("lyrics_data") or {}).get("full_lyrics") or "")) for d in docs}
    meta: Dict[str, dict] = {}
    for line in open(MAIN_ROOT / "data/all_songs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        sid = str(d.get("song_id") or d.get("metadata", {}).get("song_id") or "")
        if sid in gold_ids:
            meta[sid] = {k: d.get(k) for k in ("metadata", "community_feedback") if k in d}

    table, bundle = [], []
    for q in failures:
        r = live1[q]
        rows = trace[q]
        # 가장 멀리 간 정답 곡: 후보 안 > 보정 뒤 절단 > 융합 절단 > 경로 밖, 같으면 전체 깊이 가산 뒤 순위가 좋은 곡
        rep = min(rows, key=lambda x: (STAGE_ORDER[x["stage"]], int(x["pool_rank"] or 999),
                                       int(x["boosted_rank_deep"] or 99999)))
        run = runrows[q]
        prot = run.get("protection") or {}
        for t in rows:
            sid = t["song_id"]
            pay = text_payload.get(sid)
            best_deep = [int(x.split(":")[1]) for x in t["paths_deep"].split("|") if x and x.split(":")[0] in
                         ("text_hybrid", "image", "audio")]
            if t["stage"] == "후보 안":
                auto = "K 후보 내 최종 순위 손실"
            elif t["stage"] in ("융합 절단", "보정 뒤 절단"):
                auto = f"F {t['stage']}"
            elif t["boosted_rank_deep"] and int(t["boosted_rank_deep"]) <= 30:
                auto = "P 경로 깊이 밖(전체 깊이 가산 뒤 30 안)"
            else:
                auto = "경로 밖·전체 깊이에서도 30 밖(검토)"
            ts = trace_skip.get(q, {}).get(sid, {})
            table.append({
                "set": r["set"], "query_id": q, "song_id": sid, "relevant_count": t["relevant_count"],
                "representative": int(t is rep), "in_pool_v66": int(q in in_pool),
                "v66_cand_rank": r["candidate_rank@30"], "v66_pre_rerank_rank": r["baseline_rank"],
                "v66_final_rank": r["rerank_rank"], "v66_outcome": r["rerank_outcome"], "policy_skip": prot.get("policy_skip", ""),
                "protected_in_pool": len(prot.get("protected_ids") or []), "rerank_groups": len(run.get("rerank_runs") or []),
                "stage": t["stage"], "pool_rank": t["pool_rank"], "paths_now": t["paths_now"], "paths_deep": t["paths_deep"],
                "best_main_rank_deep": min(best_deep) if best_deep else "",
                "path_lengths_now": t["path_lengths_now"],
                "fused_rank_now": t["fused_rank_now"], "boosted_rank_now": t["boosted_rank_now"],
                "fused_rank_deep": t["fused_rank_deep"], "boosted_rank_deep": t["boosted_rank_deep"],
                "pool_rank_deep": t["pool_rank_deep"], "adjust_now": t["adjust_now"],
                "auto_stage": auto,
                "skip_pool_rank": ts.get("pool_rank", ""), "skip_stage": ts.get("stage", ""),
                "data_text_payload": int(pay is not None), "data_image_point": int(sid in image_ids),
                "data_audio_point": int(sid in audio_ids), "data_context_facts": facts.get(sid, 0),
                "data_context_profile": int(sid in profiles), "data_lyrics_chars": lyrics.get(sid, 0),
                "data_jsonl_record": int(sid in meta),
                "v50_auto": (v50.get(q) or {}).get("auto_category", "(v50 성공)" if q in v50 and not (v50[q].get("auto_category") or "").strip() else ""),
                "v50_target_id": (v50.get(q) or {}).get("target_id", ""),
                "v50_review": (v50_review.get(q) or {}).get("category", ""),
            })
        a = caches[q]["analysis"]
        bundle.append({
            "query_id": q, "set": r["set"], "query": r["query"], "in_pool": q in in_pool,
            "analysis": {k: a.get(k) for k in ("intent_type", "korean_tags", "lyric_clues", "song_title", "title_constraints",
                                                "artist_name", "artist_name_alt", "vocal_gender", "genre", "image_english_query",
                                                "audio_english_query", "has_visual_clue", "modality_weights", "release_era",
                                                "artist_type", "title_meaning_clue", "performance_clues", "context_clues",
                                                "life_stage")},
            "gold": [{"song_id": t["song_id"], "stage": t["stage"], "pool_rank": t["pool_rank"], "paths_now": t["paths_now"],
                      "paths_deep": t["paths_deep"], "boosted_rank_deep": t["boosted_rank_deep"], "adjust_now": t["adjust_now"],
                      "payload": text_payload.get(t["song_id"]), "context_facts": fact_text.get(t["song_id"], []),
                      "lyrics_chars": lyrics.get(t["song_id"], 0),
                      "jsonl_metadata": (meta.get(t["song_id"]) or {}).get("metadata")} for t in rows],
            "v66_final_top10_titles": r["rerank_top_titles"],
            "v50": {"auto": (v50.get(q) or {}).get("auto_category", ""), "review": (v50_review.get(q) or {}).get("category", ""),
                    "evidence": (v50_review.get(q) or {}).get("evidence", "")},
        })

    (HERE / "diagnosis").mkdir(exist_ok=True)
    (HERE / "diagnosis_private").mkdir(exist_ok=True)
    with open(HERE / "diagnosis/failure_stage_table.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(HERE / "diagnosis_private/review_bundle.jsonl", "w", encoding="utf-8") as fh:
        for b in bundle:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    reps = [t for t in table if t["representative"]]
    print(json.dumps({"failures": len(failures), "out_of_pool": len(out_pool), "in_pool": len(in_pool),
                      "gold_rows": len(table), "multi_gold_failures": sum(1 for t in reps if t["relevant_count"] != "1"),
                      "stage_rep": Counter(t["auto_stage"] for t in reps),
                      "missing_text_payload": [t["song_id"] for t in table if not t["data_text_payload"]],
                      "missing_jsonl": [t["song_id"] for t in table if not t["data_jsonl_record"]]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
