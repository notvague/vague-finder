"""Cross-Encoder **배치 크기**가 시간과 점수에 무엇을 하는가.

왜 이것부터인가. 2026-09-23 지연 계측에서 요청의 대부분이 질의 분석 2.6초와
CE 2.7초였다. 둘 중 CE 쪽은 **분석 결과를 건드리지 않으므로** 기존 분석 캐시를
그대로 쓸 수 있다 — 같은 질의·같은 후보·같은 문서로 전후를 비교할 수 있다는 뜻이다.

**점수가 바뀔 수 있다.** `MusicReranker.score()`는 `padding=True`로 토크나이즈하는데,
이것은 *배치 안에서 가장 긴 입력*에 맞춰 패딩한다. 배치 크기가 달라지면 한 배치에
묶이는 곡이 달라지고, 패딩 길이도 달라진다. 그래서 시간만 재면 안 되고 **점수와
순위가 같은지**를 함께 봐야 한다.

통제하는 것: 같은 분석 캐시 · 같은 후보(한 번 검색해 얼려 둔다) · 같은 문서 ·
같은 장치 · 모델은 한 번만 올리고 설정만 바꾼다 · 예열 후 반복 측정.
후보 수와 `max_length`는 **건드리지 않는다**(이번 실험의 변수는 배치 크기 하나다).

    venv/bin/python -m src.retrieval.measure_rerank_batch \
        --input experiments/reranking/eval_queries_v06.csv --split dev \
        --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
        --output-dir experiments/reranking/results_v20_ce_batch
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import get_reranker, get_search_router
from src.retrieval.analysis_cache import load_cache
from src.retrieval.evaluate_search_accuracy import read_queries

REFERENCE_BATCH = 8


def _collect_candidates(rows, cache, candidate_k: int) -> List[Dict[str, Any]]:
    """후보를 **한 번만** 검색해 얼려 둔다.

    배치 크기마다 다시 검색하면 검색 쪽 변동이 CE 비교에 섞인다.
    """
    router = get_search_router()
    frozen: List[Dict[str, Any]] = []

    async def run() -> None:
        for index, row in enumerate(rows, start=1):
            query_id, query = row["query_id"].strip(), row["query"].strip()
            analysis = cache.get(query_id, query)
            tracks = await router.search(
                analysis, top_k=candidate_k, use_rerank=False,
                candidate_k=candidate_k,
            )
            print(f"  [{index}/{len(rows)}] {query_id}: 후보 {len(tracks)}곡", flush=True)
            if tracks:
                frozen.append({"query_id": query_id, "query": query, "tracks": tracks})

    asyncio.run(run())
    return frozen


def _token_profile(reranker, frozen) -> Dict[str, Any]:
    """CE 입력이 실제로 몇 토큰인가. **배치 크기 결과를 읽는 데 꼭 필요하다.**

    `padding=True`는 배치 안 최장 입력에 맞춘다. 그런데 입력이 죄다 `max_length`
    근처면 배치를 어떻게 묶든 패딩 길이가 비슷해서, 배치 크기를 바꿔도 계산량이
    거의 그대로다. 시간이 안 줄어든 이유를 이 분포가 설명한다.
    """
    lengths: List[int] = []
    for item in frozen:
        for track in item["tracks"]:
            document = reranker.build_document(track)
            lengths.append(
                len(reranker._tokenizer(item["query"], document, truncation=False)["input_ids"])
            )
    lengths.sort()
    limit = reranker.config.max_length
    pick = lambda p: lengths[min(int(len(lengths) * p), len(lengths) - 1)]
    return {
        "pairs": len(lengths),
        "max_length": limit,
        "p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99),
        "max": lengths[-1],
        "truncated_pairs": sum(1 for n in lengths if n > limit),
    }


def _run_batch(reranker, frozen, batch_size: int, top_k: int, repeats: int) -> Dict[str, Any]:
    """한 배치 크기로 전부 재정렬한다. 시간은 반복해 재고, 결과는 한 번만 남긴다."""
    reranker.config = replace(reranker.config, batch_size=batch_size)
    per_query: Dict[str, Dict[str, Any]] = {}
    timings: Dict[str, List[float]] = {}

    for repeat in range(repeats):
        for item in frozen:
            started = time.perf_counter()
            run = reranker.rerank_run(item["query"], item["tracks"], top_k)
            elapsed = (time.perf_counter() - started) * 1000
            timings.setdefault(item["query_id"], []).append(elapsed)
            if repeat == 0:
                # 결과는 결정적이어야 한다. 첫 회차 것만 남기고 나머지는 시간만 쓴다.
                per_query[item["query_id"]] = {
                    "order": [t.id for t in run.tracks],
                    "scores": {t.id: t.rerank_score for t in run.tracks},
                }
    return {
        "batch_size": batch_size,
        "per_query": per_query,
        # 중앙값을 쓴다 — 첫 회차에 남은 초기화 비용이 평균을 끌어올린다.
        "median_ms": {qid: statistics.median(v) for qid, v in timings.items()},
        "all_ms": timings,
    }


def _compare(result: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    """기준 배치와 견준다 — 순서가 같은가, 점수가 얼마나 다른가."""
    changed_order, max_delta, compared = [], 0.0, 0
    for query_id, got in result["per_query"].items():
        ref = reference["per_query"].get(query_id)
        if ref is None:
            continue
        if got["order"] != ref["order"]:
            changed_order.append(query_id)
        for song_id, score in got["scores"].items():
            other = ref["scores"].get(song_id)
            if score is None or other is None:
                continue
            compared += 1
            max_delta = max(max_delta, abs(score - other))
    return {
        "changed_order": changed_order,
        "max_score_delta": max_delta,
        "scores_compared": compared,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="평가 질의 CSV. 기본값에 기대지 않는다")
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--analysis-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-sizes", default="4,8,16")
    ap.add_argument("--queries", type=int, default=12, help="앞에서 몇 건을 쓸지")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=30)
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    if REFERENCE_BATCH not in sizes:
        print(f"오류: 기준 배치 {REFERENCE_BATCH}이 --batch-sizes에 없습니다.", file=sys.stderr)
        return 2

    cache = load_cache(Path(args.analysis_cache))
    rows = read_queries(Path(args.input), args.split)[: args.queries]
    print(f"질의 {len(rows)}건 · 배치 {sizes} · 반복 {args.repeats}회")

    print("후보를 한 번만 검색해 얼려 둔다:")
    frozen = _collect_candidates(rows, cache, args.candidate_k)
    if not frozen:
        print("오류: 후보가 있는 질의가 없습니다.", file=sys.stderr)
        return 2

    reranker = get_reranker()
    reranker.load()
    original = reranker.config
    # 예열 — 첫 추론에 붙는 커널 컴파일 비용을 측정 밖으로 뺀다.
    reranker.rerank_run(frozen[0]["query"], frozen[0]["tracks"], args.top_k)

    results: Dict[int, Dict[str, Any]] = {}
    try:
        for size in sizes:
            print(f"\n배치 {size} …", flush=True)
            results[size] = _run_batch(reranker, frozen, size, args.top_k, args.repeats)
    finally:
        reranker.config = original

    reference = results[REFERENCE_BATCH]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"{'배치':>4} {'CE 중앙값':>10} {'질의합':>9} {'vs 8':>8}  {'순서 바뀐 질의':>14}  {'점수 최대차':>10}")
    print("-" * 72)
    summary_rows = []
    for size in sizes:
        result = results[size]
        medians = list(result["median_ms"].values())
        per_query = statistics.median(medians)
        total = sum(medians)
        ref_total = sum(reference["median_ms"].values())
        diff = f"{(total - ref_total) / ref_total * 100:+.1f}%" if size != REFERENCE_BATCH else "—"
        cmp = _compare(result, reference)
        print(f"{size:>4} {per_query:>9.0f}ms {total:>8.0f}ms {diff:>8}  "
              f"{len(cmp['changed_order']):>12}건  {cmp['max_score_delta']:>10.2e}")
        summary_rows.append({
            "batch_size": size,
            "ce_median_ms_per_query": round(per_query, 1),
            "ce_total_ms": round(total, 1),
            "vs_reference_pct": diff,
            "order_changed_queries": len(cmp["changed_order"]),
            "order_changed_ids": "|".join(cmp["changed_order"]),
            "max_score_delta": cmp["max_score_delta"],
            "scores_compared": cmp["scores_compared"],
        })

    csv_path = out_dir / f"ce_batch_{args.split}_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    profile = _token_profile(reranker, frozen)
    share = profile["truncated_pairs"] / profile["pairs"] * 100
    print(f"\nCE 입력 토큰: 중앙값 {profile['p50']} · p90 {profile['p90']} · "
          f"최대 {profile['max']} (창 {profile['max_length']})")
    print(f"  창을 넘겨 잘리는 쌍 {profile['truncated_pairs']}건 ({share:.0f}%) — "
          "입력이 창을 거의 채우므로 배치를 어떻게 묶어도 패딩 길이가 비슷하다")

    info = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "token_profile": profile,
        "queries": [item["query_id"] for item in frozen],
        "candidates_per_query": {i["query_id"]: len(i["tracks"]) for i in frozen},
        "reranker": {
            "model_name": original.model_name,
            "max_length": original.max_length,
            "rerank_weight": original.rerank_weight,
            "spread_ref": original.spread_ref,
            "device": str(getattr(reranker, "_device", "?")),
        },
        "note": (
            "후보는 배치 크기마다 다시 검색하지 않고 한 번만 검색해 얼렸다. "
            "모델은 한 번만 올리고 config.batch_size만 바꿨다."
        ),
    }
    (out_dir / f"ce_batch_{args.split}_runinfo.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n요약: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
