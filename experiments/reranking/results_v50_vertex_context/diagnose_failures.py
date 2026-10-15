"""질의별 진단표 — 정답이 후보 검색과 리랭킹 중 어디에서 탈락했는지. 모델 호출 없음.

읽는 것(모두 이 폴더 안): 리랭킹 실행 detail(`rerank_ctx{0,0.5}_rep{1,2}`), 후보 단계(`cand_ctx*`), 탈락 단계 추적(`trace_ctx*/candidate_drop_*.csv`),
Context 조회(`context_lookup.csv`), 분석 캐시(v49 Vertex). 기준 실행은 B(Context 켬) rep1이고 A도 나란히 적는다.

    venv/bin/python <이 파일> --output failure_diagnosis.csv [--review failure_review.csv]

자동 분류(`auto_category`, B rep1 기준 · 최종 10위 밖인 질의만):
  E1 후보 30 안·리랭킹 전 10위 안이었는데 리랭킹이 10위 밖으로 밀어냄
  E2 후보 30 안·리랭킹 전에도 10위 밖(11~30위)·리랭킹이 못 올림
  D  회수했지만 융합·가산·후보 절단에서 탈락(trace stage = 융합 절단 / 보정 뒤 절단)
  C1 근거는 있지만 검색 경로에서 회수하지 못함 — 어떤 기본 경로에도 30 안에 없지만, 전체 깊이(5,143)에서 가산까지 더하면 30위 안(boosted_rank_deep)
  C2 같은 조건인데 전체 깊이에서도 가산 뒤 30위 밖 — 기본 경로가 멀어 깊이로는 닿지 않음(표지·소리 서술 질의가 많다)
  X  전체 깊이에서도 어떤 경로에도 없음 — 자료 부족인지 분석 오류인지 자동으로는 가를 수 없다(미확인)
질의 분석 오류(A)와 자료 부족(B)은 자동 판정하지 않는다. `--review`(query_id, category, evidence)를 주면 그 판정을 `review_category`·`review_evidence`로
덧붙인다 — 사람이 분석 JSON·정답 곡 자료를 보고 적은 것이고, 비어 있으면 미확인이다. Context 단서 질의에서 정답에 나무위키 사실이 없으면
`context_data_gap=1`로 따로 표시한다(자료 부족의 한 종류이지 전체 판정이 아니다).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def read_detail(run: str, set_name: str) -> dict:
    path = HERE / run / f"{set_name}_dev" / "search_eval_dev_detail.csv"
    if not path.exists():
        return {}
    return {r["query_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def read_trace(run: str, set_name: str) -> dict:
    path = HERE / run / f"candidate_drop_{set_name}.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["query_id"], {})[r["song_id"]] = r
    return out


def rank_of(row: dict | None, column: str) -> int | None:
    if not row or not row.get(column):
        return None
    return int(float(row[column]))


def first_rank(row: dict | None, ids_column: str, relevant: set[str]) -> int | None:
    if not row:
        return None
    ids = row.get(ids_column, "").split("|")
    return next((i for i, sid in enumerate(ids, 1) if sid in relevant), None)


def classify(b_row: dict | None, trace: dict | None) -> str:
    final = rank_of(b_row, "rerank_rank")
    if final is not None and final <= 10:
        return "hit"
    cand = rank_of(b_row, "candidate_rank@30")
    base = rank_of(b_row, "baseline_rank")
    if cand:
        return "E1 리랭킹이 10위 밖으로 밀어냄" if (base and base <= 10) else "E2 후보 11~30·리랭킹이 못 올림"
    if trace is None:
        return "? trace 없음"
    stage = trace.get("stage", "")
    if stage in ("융합 절단", "보정 뒤 절단"):
        return f"D 회수 뒤 탈락({stage})"
    if stage == "경로 밖":
        if not trace.get("best_main_rank"):
            return "X 전체 깊이에도 없음(미확인)"
        boosted = int(trace.get("boosted_rank_deep") or 99999)
        # 전체 깊이에서 가산까지 더한 순위가 30 안이면 "경로 깊이만 넓히면 후보에 든다"는 뜻이고,
        # 그래도 30 밖이면 기본 경로 자체가 멀어 깊이로는 안 된다(표지·소리 서술·자료 부족 쪽).
        return ("C1 경로 깊이 밖·전체 깊이 가산 뒤 30위 안" if boosted <= 30
                else "C2 경로 깊이 밖·전체 깊이에서도 가산 뒤 30위 밖")
    return f"? {stage}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "failure_diagnosis.csv")
    ap.add_argument("--review", type=Path, default=HERE / "failure_review.csv")
    args = ap.parse_args()

    review = {}
    if args.review.exists():
        review = {r["query_id"]: r for r in csv.DictReader(open(args.review, encoding="utf-8-sig"))}
    lookup = {}
    lookup_path = HERE / "context_lookup.csv"
    if lookup_path.exists():
        lookup = {(r["set"], r["query_id"], r["target_id"]): r for r in csv.DictReader(open(lookup_path, encoding="utf-8-sig"))}

    rows_out = []
    for set_name in SETS:
        cache = json.loads((CACHES / f"analysis_cache_{set_name}_dev_vertex.json").read_text(encoding="utf-8"))["entries"]
        details = {run: read_detail(run, set_name) for run in
                   ("rerank_ctx0_rep1", "rerank_ctx0_rep2", "rerank_ctx0.5_rep1", "rerank_ctx0.5_rep2", "cand_ctx0", "cand_ctx0.5")}
        traces = {run: read_trace(run, set_name) for run in ("trace_ctx0", "trace_ctx0.5")}
        queries = [r for r in csv.DictReader(open(ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", encoding="utf-8-sig"))
                   if r["split"] == "dev"]
        for q in queries:
            qid = q["query_id"].strip()
            relevant = {x.strip() for x in q["relevant_ids"].split("|") if x.strip()}
            target = sorted(relevant)[0]
            analysis = (cache.get(qid) or {}).get("analysis") or {}
            b1 = details["rerank_ctx0.5_rep1"].get(qid)
            a1 = details["rerank_ctx0_rep1"].get(qid)
            tb = (traces["trace_ctx0.5"].get(qid) or {}).get(target)
            ta = (traces["trace_ctx0"].get(qid) or {}).get(target)
            lk = lookup.get((set_name, qid, target), {})
            context_clue = bool(analysis.get("context_clues"))
            row = {
                "set": set_name, "query_id": qid, "query": q["query"].strip(), "target_id": target,
                "relevant_count": len(relevant),
                "has_context_clue": int(context_clue),
                "target_facts": lk.get("target_facts", ""),
                "context_hits": lk.get("n_context_hits", ""),
                "target_context_rank": lk.get("target_context_rank", ""),
                "context_data_gap": int(context_clue and lk.get("target_facts", "") in ("0", "")) if lk else "",
                "cand_A": rank_of(details["cand_ctx0"].get(qid), "candidate_rank@30") or "",
                "cand_B": rank_of(details["cand_ctx0.5"].get(qid), "candidate_rank@30") or "",
                "pre_rerank_A": rank_of(a1, "baseline_rank") or "",
                "pre_rerank_B": rank_of(b1, "baseline_rank") or "",
                "final_A1": rank_of(a1, "rerank_rank") or "",
                "final_A2": rank_of(details["rerank_ctx0_rep2"].get(qid), "rerank_rank") or "",
                "final_B1": rank_of(b1, "rerank_rank") or "",
                "final_B2": rank_of(details["rerank_ctx0.5_rep2"].get(qid), "rerank_rank") or "",
                "outcome_B1": (b1 or {}).get("rerank_outcome", ""),
                "outcome_A1": (a1 or {}).get("rerank_outcome", ""),
                "trace_stage_A": (ta or {}).get("stage", ""),
                "trace_stage_B": (tb or {}).get("stage", ""),
                "paths_now_B": (tb or {}).get("paths_now", ""),
                "text_rank_deep": (tb or {}).get("text_hybrid_rank", ""),
                "image_rank_deep": (tb or {}).get("image_rank", ""),
                "audio_rank_deep": (tb or {}).get("audio_rank", ""),
                "aux_paths_deep_B": (tb or {}).get("aux_paths_deep", ""),
                "fused_rank_deep_B": (tb or {}).get("fused_rank_deep", ""),
                "boosted_rank_deep_B": (tb or {}).get("boosted_rank_deep", ""),
                "adjustments_B": (tb or {}).get("adjustments", ""),
                "auto_category": classify(b1, tb),
                "auto_category_A": classify(a1, ta),
                "review_category": (review.get(qid) or {}).get("category", ""),
                "review_evidence": (review.get(qid) or {}).get("evidence", ""),
                "analysis_intent": analysis.get("intent_type", ""),
                "analysis_tags": "|".join(analysis.get("korean_tags") or []),
                "analysis_lyric_clues": "|".join(str(c) for c in (analysis.get("lyric_clues") or [])),
                "analysis_gender": analysis.get("vocal_gender", ""),
                "analysis_genre": analysis.get("genre", ""),
                "analysis_era": json.dumps(analysis.get("release_era"), ensure_ascii=False) if analysis.get("release_era") else "",
            }
            rows_out.append(row)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    from collections import Counter
    fails = [r for r in rows_out if r["auto_category"] != "hit"]
    print(f"B rep1 기준 10위 밖 {len(fails)}/{len(rows_out)}")
    for cat, n in Counter(r["auto_category"] for r in fails).most_common():
        print(f"  {n:3d}  {cat}")
    print("  그중 Context 단서 질의", sum(1 for r in fails if r["has_context_clue"]),
          "· 정답에 나무위키 사실 없음", sum(1 for r in fails if r["context_data_gap"] == 1))
    reviewed = Counter(r["review_category"] for r in fails if r["review_category"])
    if reviewed:
        print("  사람 검토:", dict(reviewed), "· 미검토", sum(1 for r in fails if not r["review_category"]))
    print(f"기록: {args.output}")


if __name__ == "__main__":
    main()
