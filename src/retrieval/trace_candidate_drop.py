"""
후보 밖으로 빠진 정답이 **어느 단계에서** 빠졌는지 남긴다 (리랭킹 없음, 순위 계산을 바꾸지 않음).

평가 기록은 "정답이 후보 풀에 없음"까지만 말한다. 검색이 못 찾은 것인지, 찾았는데
융합·보정에서 밀린 것인지는 고치는 방법이 다르므로 나눠서 본다.

같은 분석으로 검색을 두 번 한다. 최종 후보 수(candidate_k)는 두 번 모두 같다.

1. 지금 깊이 (path_k 없음) — 정답이 최종 후보에 들었나, 아니면 어디서 빠졌나
     후보 안       : 최종 후보에 있다
     경로 밖       : 어떤 경로(기본·보조)의 목록에도 없었다
     융합 절단     : 기본 경로 목록에는 있었지만 RRF 절단에서 빠졌다
     보정 뒤 절단  : 융합 목록에는 있었지만 보조 경로·가산·감점 뒤 최종 절단에서 빠졌다
2. 전체 깊이 (path_k = --deep-k) — 세 기본 경로 안에서 정답의 순위, 그리고 깊이
   제한이 없을 때 융합 직후·보정 뒤 순위. "몇 위까지 봐야 닿는가"와 "닿으면
   후보에 드는가"를 함께 본다

  python -m src.retrieval.trace_candidate_drop --split dev \\
      --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \\
      --output experiments/reranking/results_v23_path_depth/candidate_drop_dev.csv

--sweep 35,40,50을 주면 단계 기록 대신 경로 깊이별로 최종 후보에 정답이 몇 위로 드는지 남긴다.
확장 열은 원래 타깃 + 허용 정답(--labels의 allowed_ids) 기준이다. 리랭킹은 하지 않는다.

곡 ID와 순위만 남긴다(제목 없음).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.backend.api.dependencies import get_search_router
from src.retrieval.analysis_cache import load_cache
from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries
from src.retrieval.explain import ExplainRecorder, SearchExplain

DEFAULT_LABELS = Path("experiments/reranking/eval_queries_v08.csv")

MAIN_PATHS = ("text_hybrid", "image", "audio")

IN_POOL = "후보 안"
NO_PATH = "경로 밖"
FUSION_CUT = "융합 절단"
FINAL_CUT = "보정 뒤 절단"


def stage_of(song_id: str, record: SearchExplain, pool: List[str]) -> str:
    if song_id in pool:
        return IN_POOL
    song = record.songs.get(song_id)
    if song is None or not song.paths:
        return NO_PATH
    if song.fused_score is None:
        return FUSION_CUT
    return FINAL_CUT


def paths_text(record: SearchExplain, song_id: str, main_only: bool = False) -> str:
    """경로:순위. 융합 절단에서 빠진 기여는 *를 붙인다."""
    song = record.songs.get(song_id)
    if song is None:
        return ""
    return "|".join(
        f"{p.path}:{p.rank}{'*' if p.dropped else ''}"
        for p in song.paths
        if not main_only or p.path not in MAIN_PATHS
    )


def rank_by(record: SearchExplain, song_id: str, boosted: bool) -> Optional[int]:
    """융합 목록 안의 순위. boosted면 가산·감점까지 더한 점수로 센다(최종 절단 직전)."""

    def score(song) -> float:
        return song.fused_score + (song.adjustment_total() if boosted else 0.0)

    target = record.songs.get(song_id)
    if target is None or target.fused_score is None:
        return None
    mine = score(target)
    return 1 + sum(
        1 for s in record.songs.values()
        if s.fused_score is not None and s.song_id != song_id and score(s) > mine
    )


def main_path_ranks(record: SearchExplain, song_id: str) -> Dict[str, Optional[int]]:
    song = record.songs.get(song_id)
    ranks: Dict[str, Optional[int]] = {name: None for name in MAIN_PATHS}
    if song is not None:
        for p in song.paths:
            if p.path in ranks and ranks[p.path] is None:
                ranks[p.path] = p.rank
    return ranks


def active_paths(record: SearchExplain) -> set:
    return {p.path for s in record.songs.values() for p in s.paths}


async def trace(args: argparse.Namespace) -> List[dict]:
    rows = read_queries(Path(args.input), args.split)
    cache = load_cache(Path(args.analysis_cache))
    router = get_search_router()
    out: List[dict] = []
    for index, row in enumerate(rows, start=1):
        qid, query = row["query_id"].strip(), row["query"].strip()
        relevant = sorted(parse_relevant_ids(row["relevant_ids"]))
        analysis = cache.get(qid, query)

        runs = {}
        for name, path_k in (("now", None), ("deep", args.deep_k)):
            recorder = ExplainRecorder(query)
            pool: List[str] = []
            await router.search(
                analysis, top_k=args.candidate_k, use_rerank=False,
                candidate_k=args.candidate_k, candidate_ids_out=pool,
                recorder=recorder, path_k=path_k,
            )
            runs[name] = (recorder.record, pool)
        now, now_pool = runs["now"]
        deep, deep_pool = runs["deep"]
        active = active_paths(deep)

        for song_id in relevant:
            ranks = main_path_ranks(deep, song_id)
            main = [r for name, r in ranks.items() if r is not None]
            song = deep.songs.get(song_id)
            out.append({
                "split": row["split"],
                "query_id": qid,
                "song_id": song_id,
                "pool_rank": now_pool.index(song_id) + 1 if song_id in now_pool else "",
                "stage": stage_of(song_id, now, now_pool),
                "paths_now": paths_text(now, song_id),
                "active_main_paths": "|".join(p for p in MAIN_PATHS if p in active),
                **{f"{name}_rank": ("" if ranks[name] is None else ranks[name]) for name in MAIN_PATHS},
                "best_main_rank": min(main) if main else "",
                "aux_paths_deep": paths_text(deep, song_id, main_only=True),
                "fused_rank_deep": rank_by(deep, song_id, boosted=False) or "",
                "boosted_rank_deep": rank_by(deep, song_id, boosted=True) or "",
                "pool_rank_deep": deep_pool.index(song_id) + 1 if song_id in deep_pool else "",
                "adjustments": "" if song is None else "|".join(
                    f"{a.rule}:{a.delta:+.4f}" for a in song.adjustments
                ),
            })
        hit = [r for r in out if r["query_id"] == qid and r["pool_rank"]]
        print(f"[{index}/{len(rows)}] {qid} {'후보 안' if hit else '후보 밖'}", flush=True)
    return out


async def sweep(args: argparse.Namespace) -> List[dict]:
    """경로 깊이별 최종 후보 안 정답 순위. path_k 없음(지금 깊이)을 항상 첫 줄로 둔다."""
    rows = read_queries(Path(args.input), args.split)
    cache = load_cache(Path(args.analysis_cache))
    with open(args.labels, encoding="utf-8-sig") as f:
        allowed = {r["query_id"]: parse_relevant_ids(r.get("allowed_ids") or "") for r in csv.DictReader(f)}
    router = get_search_router()
    depths: List[Optional[int]] = [None] + [int(x) for x in args.sweep.split(",") if x.strip()]
    out: List[dict] = []
    for index, row in enumerate(rows, start=1):
        qid, query = row["query_id"].strip(), row["query"].strip()
        strict = parse_relevant_ids(row["relevant_ids"])
        relaxed = strict | allowed.get(qid, set())
        analysis = cache.get(qid, query)
        for path_k in depths:
            pool: List[str] = []
            await router.search(
                analysis, top_k=args.candidate_k, use_rerank=False,
                candidate_k=args.candidate_k, candidate_ids_out=pool, path_k=path_k,
            )
            first = lambda ids: next((i for i, s in enumerate(pool, 1) if s in ids), "")  # noqa: E731
            out.append({
                "split": row["split"], "query_id": qid,
                "path_k": path_k or args.candidate_k,
                "pool_rank": first(strict), "pool_rank_relaxed": first(relaxed),
                "pool_ids": "|".join(pool),
            })
        print(f"[{index}/{len(rows)}] {qid}", flush=True)
    return out


def summarize_sweep(rows: List[dict]) -> None:
    base = {r["query_id"]: r for r in rows if r["path_k"] == rows[0]["path_k"]}
    print(f"\n{'경로 깊이':>8}{'후보 안(엄격)':>12}{'(확장)':>8}  얻음 / 잃음 (엄격)")
    for depth in dict.fromkeys(r["path_k"] for r in rows):
        at = [r for r in rows if r["path_k"] == depth]
        gained = [r["query_id"] for r in at if r["pool_rank"] and not base[r["query_id"]]["pool_rank"]]
        lost = [r["query_id"] for r in at if not r["pool_rank"] and base[r["query_id"]]["pool_rank"]]
        print(f"{depth:>8}{sum(1 for r in at if r['pool_rank']):>12}{sum(1 for r in at if r['pool_rank_relaxed']):>8}"
              f"  +{gained} -{lost}")


def summarize(rows: List[dict], candidate_k: int) -> None:
    by_query: Dict[str, List[dict]] = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(r)
    missed = {q: rs for q, rs in by_query.items() if not any(r["pool_rank"] for r in rs)}
    print(f"\n정답이 후보 {candidate_k} 밖인 질의 {len(missed)}/{len(by_query)}건")
    stages = Counter(r["stage"] for rs in missed.values() for r in rs)
    print("  정답 곡 기준 단계: " + ", ".join(f"{k} {v}" for k, v in stages.most_common()))
    print(f"  {'질의':<6}{'곡':>10}  {'단계':<8}{'text':>6}{'image':>6}{'audio':>6}{'보정뒤(깊이무제한)':>18}")
    for qid, rs in missed.items():
        for r in rs:
            print(f"  {qid:<6}{r['song_id']:>10}  {r['stage']:<8}{str(r['text_hybrid_rank']):>6}"
                  f"{str(r['image_rank']):>6}{str(r['audio_rank']):>6}{str(r['boosted_rank_deep']):>18}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="후보 밖 정답의 단계별 위치 기록 (리랭킹 없음)")
    p.add_argument("--input", default="experiments/reranking/eval_queries_v06.csv")
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--analysis-cache", required=True)
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--deep-k", type=int, default=3010, help="전체 깊이 실행의 path_k (코퍼스 곡 수 이상이면 전수)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sweep", default="", help="쉼표로 구분한 경로 깊이 목록 (예: 35,40,50,60)")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="--sweep의 확장 열에 쓸 allowed_ids 파일")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = asyncio.run(sweep(args) if args.sweep else trace(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.sweep:
        summarize_sweep(rows)
    else:
        summarize(rows, args.candidate_k)
    print(f"\n기록: {args.output}")


if __name__ == "__main__":
    main()
