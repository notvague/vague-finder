"""
Search quality evaluator -- 메트릭 + 지연시간 측정 통합 러너.

사용 예:
    # 실제 SearchRouter + QueryAnalyzer 로 평가
    python -m src.eval.evaluate

    # 합성 oracle 로 메트릭 파이프라인 검증
    python -m src.eval.evaluate --synthetic perfect
    python -m src.eval.evaluate --synthetic random

    # 결과 JSON 저장 (베이스라인 박제)
    python -m src.eval.evaluate --output reports/baseline_2026-05-25.json

설계:
- EvalRunner 는 search_fn(query, top_k) -> List[song_id] 콜러블만 받음.
  -> 검색 시스템 변경 없이 평가 로직 단독 테스트 가능.
- CLI 는 SearchRouter + QueryAnalyzer 인스턴스화 후 콜러블로 래핑.
- 지연시간 분해: query_analysis / search / total 3 단계.
  (router 내부 stage breakdown 은 차후 SearchRouter 에 recorder 주입 시 확장)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from src.eval.latency import LatencyRecorder
from src.eval.loader import load_eval_set, load_song_catalog
from src.eval.metrics import (
    diversity_at_k,
    mean_negative_rank_when_present,
    mrr,
    ndcg_at_k,
    negative_hit_rate_at_k,
    recall_at_k,
)
from src.eval.schema import EvalQuery, EvalSet, QuerySet, V05_QUERY_SETS

logger = logging.getLogger(__name__)

# (query, top_k) -> ranked song_id list
SearchFn = Callable[[str, int], Awaitable[List[str]]]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    query: str
    category: str
    tier: str
    modality_focus: str
    ranked_ids: List[str]
    n_positives: int
    n_negatives: int
    positives: List[str]        # ✅/❌ 표시용
    negatives: List[str]        # ✅/❌ 표시용
    recall_at_10: float
    mrr_score: float
    ndcg_at_10: float
    neg_hit_rate_at_10: float
    mean_neg_rank_when_present: Optional[float]
    diversity_at_10: float
    latency_ms: float


@dataclass
class GroupAgg:
    label: str
    n: int = 0
    recall_at_10: float = 0.0
    mrr_score: float = 0.0
    ndcg_at_10: float = 0.0
    neg_hit_rate_at_10: float = 0.0
    diversity_at_10: float = 0.0


def _aggregate(label: str, results: List[QueryResult]) -> GroupAgg:
    if not results:
        return GroupAgg(label=label, n=0)
    n = len(results)
    return GroupAgg(
        label=label,
        n=n,
        recall_at_10=sum(r.recall_at_10 for r in results) / n,
        mrr_score=sum(r.mrr_score for r in results) / n,
        ndcg_at_10=sum(r.ndcg_at_10 for r in results) / n,
        neg_hit_rate_at_10=sum(r.neg_hit_rate_at_10 for r in results) / n,
        diversity_at_10=sum(r.diversity_at_10 for r in results) / n,
    )


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    def __init__(
        self,
        eval_set: EvalSet,
        search_fn: SearchFn,
        catalog: Optional[Dict[str, dict]] = None,
        top_k: int = 10,
    ):
        self._eval_set = eval_set
        self._search_fn = search_fn
        self._catalog = catalog or {}
        self._top_k = top_k
        self._recorder = LatencyRecorder()

    @property
    def recorder(self) -> LatencyRecorder:
        return self._recorder

    def _get_artist(self, sid: str) -> str:
        if sid in self._catalog:
            a = self._catalog[sid].get("artist") or []
            if a:
                return a[0]
        return sid  # catalog 미존재 시 song_id 자체를 unique key 로 사용

    async def run(self) -> List[QueryResult]:
        results: List[QueryResult] = []
        for q in self._eval_set.queries:
            r = await self._run_one(q)
            results.append(r)
        return results

    async def _run_one(self, q: EvalQuery) -> QueryResult:
        # End-to-end 측정 — search_fn 내부에 instrumented 가 있다면 stage별 자동 누적
        t0 = time.perf_counter()
        try:
            ranked = await self._search_fn(q.query, self._top_k)
        except Exception as e:
            logger.error("[EvalRunner] query=%s 검색 실패: %s", q.query_id, e)
            ranked = []
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._recorder.record("total", latency_ms)

        pos = set(q.positives)
        neg = set(q.negatives)
        has_positives = bool(pos)

        return QueryResult(
            query_id=q.query_id,
            query=q.query,
            category=q.category,
            tier=q.tier,
            modality_focus=q.modality_focus,
            ranked_ids=list(ranked),
            n_positives=len(pos),
            n_negatives=len(neg),
            positives=list(q.positives),
            negatives=list(q.negatives),
            # positives 없으면 0.0 (low_signal). 리포트에서 N/A 로 표시.
            recall_at_10=recall_at_k(ranked, pos, 10) if has_positives else 0.0,
            mrr_score=mrr(ranked, pos) if has_positives else 0.0,
            ndcg_at_10=ndcg_at_k(ranked, pos, 10) if has_positives else 0.0,
            neg_hit_rate_at_10=negative_hit_rate_at_k(ranked, neg, 10),
            mean_neg_rank_when_present=mean_negative_rank_when_present(ranked, neg),
            diversity_at_10=diversity_at_k(ranked, self._get_artist, 10),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def format_report(results: List[QueryResult], recorder: LatencyRecorder) -> str:
    lines: List[str] = []

    def line(s: str = "") -> None:
        lines.append(s)

    line("=" * 92)
    line(f"Search Quality Evaluation Report  (n={len(results)} queries)")
    line("=" * 92)

    # 1) Tier x Modality 그리드 — low_signal 은 분리
    grid: Dict[Tuple[str, str], List[QueryResult]] = defaultdict(list)
    for r in results:
        grid[(r.tier, r.modality_focus)].append(r)

    line("\n[Tier × Modality 메트릭]")
    line(f"{'Group':38s} {'N':>4} {'R@10':>7} {'MRR':>7} {'NDCG':>7} {'NegHit':>7} {'Div':>6}")
    line("-" * 92)
    for (tier, mod), rs in sorted(grid.items()):
        label = f"{tier:<14} x {mod:<10}"
        agg = _aggregate(label, rs)
        if tier == "low_signal":
            # positives 없으니 R/MRR/NDCG 의미 없음 — N/A 표시
            line(
                f"{label:38s} {agg.n:>4} {'N/A':>7} {'N/A':>7} {'N/A':>7} "
                f"{agg.neg_hit_rate_at_10:>7.3f} {agg.diversity_at_10:>6.2f}"
            )
        else:
            line(
                f"{label:38s} {agg.n:>4} {agg.recall_at_10:>7.3f} {agg.mrr_score:>7.3f} "
                f"{agg.ndcg_at_10:>7.3f} {agg.neg_hit_rate_at_10:>7.3f} {agg.diversity_at_10:>6.2f}"
            )

    # 2) 주력 KPI (tier=vague)
    vague = [r for r in results if r.tier == "vague"]
    if vague:
        agg = _aggregate("vague (all)", vague)
        line("\n[주력 KPI — tier=vague]")
        line(f"  N         = {agg.n}")
        line(f"  Recall@10 = {agg.recall_at_10:.3f}")
        line(f"  MRR       = {agg.mrr_score:.3f}")
        line(f"  NDCG@10   = {agg.ndcg_at_10:.3f}")
        line(f"  NegHit@10 = {agg.neg_hit_rate_at_10:.3f}  (낮을수록 좋음)")

    # 3) 시나리오별 보조 지표
    line("\n[시나리오 별]")
    for tier in ("baseline", "misinformation", "low_signal"):
        rs = [r for r in results if r.tier == tier]
        if not rs:
            continue
        agg = _aggregate(tier, rs)
        if tier == "low_signal":
            line(
                f"  {tier:14s} n={agg.n:>2}  Diversity@10={agg.diversity_at_10:.2f}  "
                f"NegHit@10={agg.neg_hit_rate_at_10:.3f}"
            )
        else:
            line(
                f"  {tier:14s} n={agg.n:>2}  R@10={agg.recall_at_10:.3f}  "
                f"MRR={agg.mrr_score:.3f}  NegHit={agg.neg_hit_rate_at_10:.3f}"
            )

    # 4) Latency
    line("\n[Latency (ms)]")
    line(f"  {'Stage':18s} {'N':>4} {'p50':>7} {'p95':>7} {'p99':>7} {'mean':>7}")
    line("  " + "-" * 56)
    for label in recorder.labels():
        s = recorder.stats(label)
        line(
            f"  {label:18s} {s['n']:>4} {s['p50']:>7.1f} {s['p95']:>7.1f} "
            f"{s['p99']:>7.1f} {s['mean']:>7.1f}"
        )

    # 5) Negative 등장한 query 상세 (회귀 디버깅용)
    leaked = [r for r in results if r.neg_hit_rate_at_10 > 0]
    if leaked:
        line(f"\n[⚠️ Negative leak — top-10 에 함정이 들어온 {len(leaked)}개 query]")
        for r in leaked:
            rank = r.mean_neg_rank_when_present
            line(
                f"  {r.query_id}  rate={r.neg_hit_rate_at_10:.2f}  "
                f"meanRank={rank if rank is None else f'{rank:.1f}'}  '{r.query}'"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hit details reporter
# ---------------------------------------------------------------------------

def format_hits(results: List[QueryResult], catalog: Dict[str, dict]) -> str:
    """쿼리별 실제 반환 곡 목록을 제목 + 아티스트 + ✅/❌ 표시로 출력."""
    lines: List[str] = []

    def line(s: str = "") -> None:
        lines.append(s)

    line("=" * 92)
    line("Per-Query Hit Details  (✅ positive  ❌ negative  —  uncategorized)")
    line("=" * 92)

    for r in results:
        pos_set = set(r.positives)
        neg_set = set(r.negatives)
        has_pos = bool(pos_set)

        metrics_str = (
            f"R@10={r.recall_at_10:.3f}  MRR={r.mrr_score:.3f}"
            if has_pos
            else "R@10=N/A  (low_signal)"
        )
        line(f"\n[{r.query_id}] {r.tier} / {r.modality_focus}  |  {metrics_str}")
        line(f'  쿼리: "{r.query}"')

        if not r.ranked_ids:
            line("  ⚠️  결과 없음")
            continue

        for rank, sid in enumerate(r.ranked_ids, 1):
            song = catalog.get(sid, {})
            title = song.get("title") or sid
            artist_list = song.get("artist") or []
            artist = artist_list[0] if artist_list else "?"

            if sid in pos_set:
                mark = "✅"
            elif sid in neg_set:
                mark = "❌"
            else:
                mark = "—"

            line(f"  {rank:2}. {mark} {title:<28} {artist}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic search_fn (메트릭 파이프라인 검증용)
# ---------------------------------------------------------------------------

def make_perfect_search_fn(eval_set: EvalSet) -> SearchFn:
    """각 query 의 positives 를 그대로 top-K 로 반환 — 메트릭 상한 검증."""
    lookup = {q.query: list(q.positives) for q in eval_set.queries}

    async def fn(query: str, top_k: int) -> List[str]:
        return lookup.get(query, [])[:top_k]

    return fn


def make_random_search_fn(
    catalog: Dict[str, dict], seed: int = 42
) -> SearchFn:
    """카탈로그에서 무작위 K개 — 베이스라인 하한 검증."""
    rng = random.Random(seed)
    pool = list(catalog.keys())

    async def fn(query: str, top_k: int) -> List[str]:
        return rng.sample(pool, min(top_k, len(pool)))

    return fn


# ---------------------------------------------------------------------------
# CLI: 실제 SearchRouter + QueryAnalyzer 래핑
# ---------------------------------------------------------------------------

def build_real_search_fn(
    recorder: LatencyRecorder,
    force_weights: Optional[List[float]] = None,
    disable_boost: bool = False,
) -> SearchFn:
    """실제 시스템 인스턴스화 — Qdrant 적재본·Gemini·HuggingFace 모델이 필요하다.

    force_weights/disable_boost: ablation 용 — router.search 로 그대로 전달.
    """
    from src.embedding.models.audio_clap import CLAPAudioEmbedder
    from src.embedding.models.image_siglip2 import SigLIP2Embedder
    from src.embedding.models.text_bm25 import BM25SparseEncoder
    from src.embedding.models.text_koe5 import KoE5Embedder
    from src.retrieval.query_analyzer import QueryAnalyzer
    from src.retrieval.search_router import SearchRouter
    from src.retrieval.search_service import SearchService
    # 앱과 같은 벡터 클라이언트를 쓴다. 로컬 Qdrant는 저장 폴더를 한 프로세스에서
    # 하나만 열 수 있으므로 따로 만들면 안 된다.
    from src.backend.api.dependencies import get_vector_client

    pc = get_vector_client()
    ko_e5 = KoE5Embedder()
    bm25 = BM25SparseEncoder(params_path=Path("artifacts/bm25_params.json"))
    try:
        bm25.load()
    except Exception as e:
        logger.warning("[evaluate] BM25 load 실패: %s", e)

    siglip = SigLIP2Embedder()
    clap = CLAPAudioEmbedder()
    search_svc = SearchService(
        vector_client=pc, text_embedder=ko_e5, bm25_encoder=bm25
    )
    router = SearchRouter(
        search_service=search_svc,
        image_embedder=siglip,
        audio_embedder=clap,
        vector_client=pc,
    )
    analyzer = QueryAnalyzer()

    async def fn(query: str, top_k: int) -> List[str]:
        # Stage 별 분해 측정 — router 내부 stage 는 추후 확장
        with recorder.time("query_analysis"):
            analysis = analyzer.analyze(query)
        with recorder.time("search"):
            tracks = await router.search(
                analysis, top_k,
                force_weights=force_weights,
                disable_boost=disable_boost,
            )
        # MatchingTrack 필드명 호환 (id 또는 song_id)
        return [getattr(t, "id", None) or getattr(t, "song_id", None) for t in tracks]

    return fn


def select_queries(
    queries: List[EvalQuery],
    query_sets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[EvalQuery]:
    """이 진입점이 돌릴 질의를 고른다. 세트를 안 주면 v0.5 세트(v04·modality_v1·clarify_v1)다.

    `queries.json` 전체를 기본으로 두면 v09 봉인 test 38건이 함께 돌아간다(PR #16~#22 리뷰).
    봉인 세트는 최종 설정 하나로 한 번만 여는 것이므로 `--query-set v09 --split test`로 명시해야 돈다.
    `export_csv.select`와 달리 정답 없는 질의도 남긴다 — 이 러너는 그 질의를 따로 집계한다.
    """
    chosen = tuple(query_sets) if query_sets else V05_QUERY_SETS
    picked = [q for q in queries if q.query_set in chosen]
    if splits:
        picked = [q for q in picked if q.split in splits]
    return picked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description="Vague-Finder 검색 품질 평가 러너")
    p.add_argument("--queries", type=Path, default=Path("docs/eval/queries.json"))
    p.add_argument("--catalog", type=Path, default=Path("docs/eval/song_catalog.json"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--query-set",
        action="append",
        choices=list(QuerySet.__args__),
        default=None,
        help="질의 세트(반복 가능). 기본은 v0.5 세트. v09 봉인 test는 --query-set v09 --split test로만 돈다",
    )
    p.add_argument(
        "--split",
        action="append",
        choices=("dev", "test"),
        default=None,
        help="split(반복 가능). 기본은 고른 세트의 모든 split",
    )
    p.add_argument(
        "--synthetic",
        choices=("perfect", "random"),
        default=None,
        help="외부 의존성 없는 가상 검색기로 메트릭 파이프라인 검증",
    )
    p.add_argument("--output", type=Path, default=None, help="결과 JSON 저장")
    p.add_argument(
        "--show-hits",
        action="store_true",
        default=False,
        help="쿼리별 반환 곡 목록을 제목+아티스트+✅/❌ 표시로 출력",
    )
    p.add_argument(
        "--force-weights",
        type=str,
        default=None,
        help="ablation: RRF 가중치 수동 주입 'text,image,audio' (예: '1,0,0'). 게이팅도 우회.",
    )
    p.add_argument(
        "--disable-boost",
        action="store_true",
        default=False,
        help="ablation: RRF 이후 정확매칭 부스팅 비활성화 (순수 경로 기여만).",
    )
    args = p.parse_args()

    full_set = load_eval_set(args.queries)
    picked = select_queries(full_set.queries, args.query_set, args.split)
    if not picked:
        p.error("고른 세트·split에 질의가 없습니다")
    sealed = [q.query_id for q in picked if q.query_set == "v09" and q.split == "test"]
    if sealed:
        logger.warning("v09 봉인 test %d건이 포함됐다 — 최종 설정 하나로 한 번만 연다", len(sealed))
    eval_set = full_set.model_copy(update={"queries": picked})
    catalog = load_song_catalog(args.catalog)
    logger.info("Eval set v%s, %d/%d queries (sets=%s, splits=%s) / Catalog %d songs",
                eval_set.version, len(picked), len(full_set.queries),
                ",".join(args.query_set or V05_QUERY_SETS), ",".join(args.split or ["all"]), len(catalog))

    if args.synthetic == "perfect":
        search_fn: SearchFn = make_perfect_search_fn(eval_set)
        logger.info("[synthetic=perfect] positives 를 그대로 반환 (메트릭 상한)")
    elif args.synthetic == "random":
        search_fn = make_random_search_fn(catalog)
        logger.info("[synthetic=random] catalog 에서 무작위 (베이스라인 하한)")
    else:
        recorder_for_build = LatencyRecorder()
        fw: Optional[List[float]] = None
        if args.force_weights:
            fw = [float(x) for x in args.force_weights.split(",")]
            if len(fw) != 3:
                p.error("--force-weights 는 'text,image,audio' 3개 값이어야 합니다")
        search_fn = build_real_search_fn(
            recorder_for_build, force_weights=fw, disable_boost=args.disable_boost,
        )
        logger.info(
            "[real] SearchRouter + QueryAnalyzer 인스턴스화 완료 (force_weights=%s, disable_boost=%s)",
            fw, args.disable_boost,
        )

    runner = EvalRunner(eval_set, search_fn, catalog, top_k=args.top_k)
    results = asyncio.run(runner.run())

    report = format_report(results, runner.recorder)
    print("\n" + report + "\n")

    if args.show_hits:
        hits_report = format_hits(results, catalog)
        print("\n" + hits_report + "\n")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_set_version": eval_set.version,
            "top_k": args.top_k,
            "synthetic": args.synthetic,
            "ablation": {
                "force_weights": args.force_weights,
                "disable_boost": args.disable_boost,
            },
            "n_queries": len(results),
            "per_query": [asdict(r) for r in results],
            "latency_stats": runner.recorder.all_stats(),
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("결과 저장: %s", args.output)


if __name__ == "__main__":
    main()
