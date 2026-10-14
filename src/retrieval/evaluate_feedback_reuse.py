"""피드백 후보 보충의 비교 측정 — 수집 전용(off)과 후보 보충(on).

세 가지 명령이 있다.

    compare    같은 질의·같은 분석·같은 사례 판으로 off/on을 번갈아 N회 재고 지표·변화·지연을 낸다
    calibrate  질의와 사례 문장의 유사도 분포만 본다(검색 없음). 유사도 문턱을 고를 때 쓴다
    harness    사례 파일을 끼운 채 기존 측정기(evaluate_search_accuracy·evaluate_clarification)를 돌린다.
               측정기는 라우터에 후보 전체를 요청하므로 표시 수(`--display-k`, 기본은 측정기의 --top-k)를 함께 알려 준다

사례는 둘 중 하나에서 온다.

- `--cases 파일.jsonl` — 합성 사례나 내보낸 사례. **격리된 메모리 저장소**에 올린다(운영 Mongo·Qdrant에 쓰지 않는다).
- `--cases live` — 운영 저장소에 승인·색인돼 있는 사례. 검색 서버와 같은 경로로 읽기만 한다.

질의는 사례와 **따로 쓴 것**이어야 한다. 사례를 만든 원문을 다시 넣거나 말만 바꾼 질의로 잰 결과는 기능
확인이지 검색 개선의 근거가 아니다. **dev 분할만 연다** — 그 밖의 값(test · all · 빈 값)은 `--sealed-final`
없이는 거부한다. `harness`가 넘기는 측정기 인자의 `--split`도 같은 규칙으로 본다(축약·`=` 형식 포함).

사례 파일의 줄마다 `expires_at`이 있어야 한다. 합성 사례 파일만 `--synthetic-cases`로 그 검사를 뺀다 —
실사용 사례를 내보낸 파일에서 만료를 빠뜨리면 원래 보관 기간을 넘겨 쓰게 된다.

`--synthetic`은 운영 자료 없이 도는 자체 완결 실행이다 — 합성 곡 402개를 메모리 Qdrant에 올리고 실제 KoE5와
실제 라우터로 돈다(분석은 고정, 리랭커 없음). **기능 확인용이며 품질 개선의 근거가 아니다.**

산출물(`--output-dir`): `feedback_reuse_detail.csv` · `_summary.csv` · `_changes.csv` · `_runinfo.json`.
운영 사례(live)로 잰 산출물에는 사례 ID·재사용 문장을 쓰지 않는다 — 곡 ID와 건수만 남는다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import runpy
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from src.backend.schemas.query import QueryAnalysis
from src.feedback.eval_cases import CASE_GROUPS, EvalCase, FileRetriever, build_file_retriever, cases_digest, load_cases
from src.feedback.settings import EMBEDDING_VERSION, MAX_DISPLAY_K, RetrievalSettings, measurement_display
from src.retrieval.evaluate_search_accuracy import (
    first_relevant_rank,
    hit_at_k,
    mrr_at_k,
    parse_relevant_ids,
    read_queries,
)
from src.retrieval.explain import ExplainRecorder

SYNTHETIC_DIR = Path("experiments/feedback_reuse/synthetic")
OPEN_SPLIT = "dev"
SEALED_MESSAGE = "dev 분할만 연다. 봉인 분할은 최종 설정 하나로 한 번만 재며, 그때만 --sealed-final을 준다."


def require_open_split(split: Optional[str], sealed_final: bool) -> None:
    """dev가 아니면 거부한다. "test면 거부"가 아니다 — all·빈 값·오타도 test 행을 읽을 수 있다(리뷰 FR-03)."""
    if (split or "").strip().lower() != OPEN_SPLIT and not sealed_final:
        raise SystemExit(SEALED_MESSAGE)


def harness_splits(rest: Sequence[str]) -> List[str]:
    """측정기에 넘길 인자에서 `--split` 값을 모두 읽는다. `--split=test`·`--spl test` 같은 형식도 argparse가 읽는 대로."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--split", action="append", default=[])
    return list(probe.parse_known_args(list(rest))[0].split)


def harness_display_k(explicit: Optional[int], rest: Sequence[str]) -> int:
    """측정기가 결과로 쓰는 곡 수. 명시값 → 측정기 인자의 `--top-k` → 측정기 기본값(10) 순으로 정한다.

    기존 측정기는 라우터에 **후보 전체**를 요청하고(top_k = candidate_k) 자기 쪽에서 리랭킹한 뒤 상위 N곡만 쓴다.
    라우터가 그 N을 모르면 "후보 수 − 표시 수"가 0이 되어 보충이 한 건도 일어나지 않고, 측정은 조용히 off와 같아진다.
    """
    if explicit is not None:
        return _display_k(explicit, "--display-k")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--top-k", type=int)
    top_k = probe.parse_known_args(list(rest))[0].top_k
    return MEASURE_TOP_K if top_k is None else _display_k(top_k, "측정기 인자의 --top-k")


def _display_k(value: int, source: str) -> int:
    """쓸 수 없는 표시 수는 거부한다. 그대로 넘기면 보충이 0건인 채 조용히 돌아 on이 off와 같아진다."""
    if not 1 <= value <= MAX_DISPLAY_K:
        raise SystemExit(f"{source}는 1~{MAX_DISPLAY_K}이어야 한다: {value}")
    return value


def cases_are_synthetic(args: argparse.Namespace) -> bool:
    """만료 없는 사례 파일을 받아도 되는가. 명시했거나, 저장소에 든 합성 세트일 때만."""
    if getattr(args, "synthetic_cases", False) or getattr(args, "synthetic", False):
        return True
    try:
        return Path(args.cases).resolve().is_relative_to(SYNTHETIC_DIR.resolve())
    except (OSError, ValueError):
        return False
MODES = ("off", "on")
SWITCH = "FEEDBACK_RETRIEVAL_ENABLED"
# 기존 측정기의 --top-k 기본값. 측정기 인자에 --top-k가 없으면 이 값을 표시 수로 본다.
MEASURE_TOP_K = 10
# 채택 기준의 지연 한도(초안). off 대비 중앙값·p95 증가가 각각 이 값 이내여야 한다.
LATENCY_BUDGET_MS = 1000.0


@dataclass
class Bench:
    """측정에 쓰는 검색 스택 한 벌."""

    router: Any
    analyze: Callable[[dict], QueryAnalysis]
    file_cases: Optional[FileRetriever]
    info: Dict[str, Any]
    close: Callable[[], None] = field(default=lambda: None)


# ---------------------------------------------------------------------------
# 검색 스택
# ---------------------------------------------------------------------------

@contextmanager
def installed_cases(cases_arg: str, synthetic: bool) -> Iterator[Optional[FileRetriever]]:
    """사례 파일을 검색 스택의 조회 서비스 자리에 끼웠다가, **끝나면 원래 팩토리로 되돌린다.** live면 아무것도 바꾸지 않는다.

    `dependencies.get_search_router()`는 `get_feedback_retriever`라는 이름을 부를 때 찾으므로 그 이름을 바꿔 끼운다.
    바꿔 끼우는 것도 **같은 모양의 팩토리**(`dependencies.singleton` — `cache_clear`·`peek`이 있다)여야 한다.
    `close_vector_client()`가 이 팩토리의 캐시를 비우기 때문이다. 일반 함수로 바꾸면 측정을 다 돌린 뒤 정리에서
    `AttributeError`로 죽는다(리뷰 FR-09).
    """
    if cases_arg == "live":
        yield None
        return
    from src.backend.api import dependencies

    cases = load_cases(Path(cases_arg), require_expiry=not synthetic)
    file_cases = build_file_retriever(
        cases, embedder=dependencies.get_text_embedder(), song_client=dependencies.get_vector_client())
    original = dependencies.get_feedback_retriever
    dependencies.get_feedback_retriever = dependencies.singleton(lambda: file_cases.retriever)
    # 이미 만들어 둔 라우터가 있으면 예전 조회 서비스를 쥐고 있다. 다음 get_search_router()가 새로 만들게 한다.
    _drop_cached_router(dependencies)
    try:
        yield file_cases
    finally:
        dependencies.get_feedback_retriever = original
        # 파일 사례를 쥔 라우터가 캐시에 남으면 같은 프로세스의 다음 호출자가 그것을 받는다.
        _drop_cached_router(dependencies)
        file_cases.close()


def _drop_cached_router(dependencies: Any) -> None:
    """캐시된 검색 라우터를 **모두** 내려놓는다(모델별로 여럿일 수 있다). 스레드풀을 세우고(돌던 조회가 끝난 뒤) 캐시를 비운다."""
    for router in dependencies.get_search_router.values():
        router.shutdown()
    dependencies.get_search_router.cache_clear()


def real_bench(args: argparse.Namespace, file_cases: Optional[FileRetriever]) -> Bench:
    """현재 코드·현재 코퍼스. 분석은 캐시에서 읽는다(같은 측정을 두 번 해도 분석이 갈리지 않게).

    사례 파일은 호출부가 `installed_cases`로 이미 끼워 둔 상태다(`file_cases`는 기록용).
    """
    from src.backend.api import dependencies
    from src.retrieval.analysis_cache import load_cache

    router = dependencies.get_search_router()
    cache = load_cache(Path(args.analysis_cache))
    info = {
        "corpus": "live", "analysis_cache": str(args.analysis_cache), "analysis_cache_meta": cache.meta,
        "reranker": type(dependencies.get_reranker()).__name__ if args.rerank else None,
        "gemini_retrieval_backend": os.getenv("GEMINI_RETRIEVAL_BACKEND", "auto"),
    }
    return Bench(router, lambda row: cache.get(row["query_id"].strip(), row["query"].strip()), file_cases, info,
                 dependencies.close_vector_client)


@contextmanager
def open_bench(args: argparse.Namespace) -> Iterator[Bench]:
    """측정 스택을 열고, 끝나면 닫는다. **정리 중의 오류는 경고로만 남긴다** — 이미 잰 결과를 버리지 않는다.

    호출부는 이 블록 안에서 산출물까지 쓴다. 닫는 것은 그 뒤다.
    """
    if args.synthetic:
        bench = synthetic_bench(args)
        try:
            yield bench
        finally:
            _close_bench(bench)
        return
    with installed_cases(args.cases, cases_are_synthetic(args)) as file_cases:
        bench = real_bench(args, file_cases)
        try:
            yield bench
        finally:
            _close_bench(bench)


def _close_bench(bench: Bench) -> None:
    try:
        bench.close()
    except Exception as exc:
        print(f"경고: 측정 스택을 닫는 중 오류({type(exc).__name__}). 산출물은 닫기 전에 이미 썼다.", file=sys.stderr, flush=True)


def synthetic_bench(args: argparse.Namespace) -> Bench:
    """합성 곡을 메모리 Qdrant에 올린 자체 완결 스택. 운영 Mongo·Qdrant·Gemini를 쓰지 않는다."""
    from qdrant_client import models

    from src.embedding.models.text_koe5 import KoE5Embedder
    from src.retrieval.search_router import SearchRouter
    from src.retrieval.search_service import SearchService
    from src.vector_db.qdrant_backend import DENSE_VECTOR, QdrantVectorClient, collection_name, point_id
    from src.vector_db.settings import TEXT_DENSE_DIM, TEXT_HYBRID_INDEX_NAME

    songs = [json.loads(line) for line in (args.synthetic_dir / "songs.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    embedder = KoE5Embedder()
    client = QdrantVectorClient(path=":memory:")
    client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=TEXT_DENSE_DIM)
    texts = [f"{song['title']} {song['artist']} {song['genre']} {song['search_style_summary']}" for song in songs]
    vectors = embedder.embed_passages(texts, batch_size=16, normalize=True)
    client.client.upsert(
        collection_name=collection_name(TEXT_HYBRID_INDEX_NAME),
        points=[models.PointStruct(id=point_id(song["song_id"]), vector={DENSE_VECTOR: vector.tolist()}, payload=song)
                for song, vector in zip(songs, vectors)],
    )
    file_cases = build_file_retriever(
        load_cases(Path(args.cases), require_expiry=False), embedder=embedder, song_client=client)
    router = SearchRouter(
        # BM25 통계는 운영 코퍼스의 것이라 합성 곡에는 쓰지 않는다 — dense만으로 찾는다.
        search_service=SearchService(client, embedder, SimpleNamespace(_is_fitted=False)),
        image_embedder=None, audio_embedder=None, vector_client=client, reranker=None,
        feedback_retriever=file_cases.retriever,
    )

    def analyze(row: dict) -> QueryAnalysis:
        # 분석기를 부르지 않는다. 텍스트 경로만 도는 고정 분석이다.
        return QueryAnalysis(original_query=row["query"].strip(), intent_type="mood",
                             image_english_query="", audio_english_query="")

    def close() -> None:
        router.shutdown()
        file_cases.close()
        client.close()

    info = {"corpus": "synthetic", "songs": len(songs), "analysis": "fixed(text-only)", "reranker": None}
    return Bench(router, analyze, file_cases, info, close)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def _search_once(bench: Bench, analysis: QueryAnalysis, args: argparse.Namespace, *,
                       rerank: bool, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    pool: List[str] = []
    feedback: list = []
    recorder = ExplainRecorder(analysis.original_query)
    started = time.perf_counter()
    results = await bench.router.search(
        analysis, top_k=args.top_k, candidate_k=args.candidate_k, use_rerank=rerank, exclude_ids=exclude,
        candidate_ids_out=pool, feedback_out=feedback, recorder=recorder,
    )
    record = feedback[0] if feedback else None
    # 들어온 곡 가운데 리랭커가 실제로 평가한 곡. 보충 곡은 후보 끝에 서므로, 리랭커가 후보 앞쪽만 평가하는
    # 설정(GEMINI_RERANK_MAX_CANDIDATES < 후보 수 등)에서는 평가받지 못해 표시될 수 없다.
    judged = [
        song_id for song_id in (record.entered_ids if record is not None else [])
        if recorder.record.songs.get(song_id) is not None and recorder.record.songs[song_id].group_rank_before is not None
    ]
    return {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "results": [str(track.id) for track in results], "pool": pool,
        "feedback": record, "stage": recorder.record.reorder_stage,
        "rerank_runs": list(recorder.record.rerank_runs), "entered_judged": judged,
    }


async def run_compare(bench: Bench, rows: List[dict], args: argparse.Namespace,
                      checkpoint: Optional[Callable[[List[dict]], None]] = None) -> List[dict]:
    """질의마다 off → on을 번갈아 잰다. 한 실행 안에서 두 방식이 같은 시각대의 부하를 받는다.

    `checkpoint`를 주면 질의 하나가 끝날 때마다 지금까지의 detail을 넘긴다 — 중간에 죽어도 끝난 질의의 기록은
    남게(리랭킹을 켠 측정은 질의마다 Gemini 호출이 든다).
    """
    detail: List[dict] = []
    live = bench.file_cases is None
    for run in range(1, args.runs + 1):
        for index, row in enumerate(rows, start=1):
            query_id, relevant = row["query_id"].strip(), parse_relevant_ids(row["relevant_ids"])
            analysis = bench.analyze(row)
            print(f"[run {run}] [{index}/{len(rows)}] {query_id}", flush=True)
            for mode in MODES:
                os.environ[SWITCH] = "1" if mode == "on" else "0"
                out = await _search_once(bench, analysis, args, rerank=args.rerank)
                record = out["feedback"]
                matched = list(record.cases) if record is not None else []
                cases = [case for case in matched if case.added]         # 후보에 넣으려고 점수를 준(후보 밖) 곡
                reappeared: Optional[int] = None
                if mode == "on" and args.reject_check and out["results"]:
                    # 보여준 곡을 모두 거절한 다음 턴. 거절한 곡이 보충으로 되돌아오면 안 된다.
                    again = await _search_once(bench, analysis, args, rerank=False, exclude=out["results"])
                    reappeared = len(set(out["results"]) & (set(again["pool"]) | set(again["results"])))
                detail.append({
                    "run": run, "mode": mode, "query_id": query_id, "relevant_ids": "|".join(sorted(relevant)),
                    "candidate_rank": first_relevant_rank(out["pool"], relevant) or "",
                    "candidate_hit": int(any(song_id in relevant for song_id in out["pool"])),
                    "rank": first_relevant_rank(out["results"], relevant) or "",
                    "hit1": int(hit_at_k(out["results"], relevant, 1)),
                    "hit10": int(hit_at_k(out["results"], relevant, 10)),
                    "hit15": int(hit_at_k(out["results"], relevant, 15)),
                    "mrr10": round(mrr_at_k(out["results"], relevant, 10), 6),
                    "latency_ms": round(out["latency_ms"], 1),
                    "reorder_stage": out["stage"], "rerank_runs": "|".join(out["rerank_runs"]),
                    "feedback_status": record.status if record is not None else "",
                    "supplemented_ids": "|".join(case.song_id for case in cases),
                    "already_candidate_ids": "|".join(case.song_id for case in matched if case.already_candidate),
                    # 사례 ID는 합성·파일 사례일 때만 쓴다. 운영 사례는 곡 ID와 건수만 남긴다.
                    "supplemented_cases": "" if live else "|".join(
                        f"{bench.file_cases.case_id_of(case.feedback_id)}:{bench.file_cases.group_of(case.feedback_id)}"
                        for case in cases),
                    "supplement_similarity": "|".join(f"{case.similarity:.4f}" for case in cases),
                    "entered_ids": "|".join(record.entered_ids) if record is not None else "",
                    # 리랭킹을 돌렸을 때만 뜻이 있다. 들어온 곡이 있는데 이 칸이 비면 평가 범위 밖이었다.
                    "entered_judged_ids": "|".join(out["entered_judged"]),
                    "displaced_ids": "|".join(record.displaced_ids) if record is not None else "",
                    "feedback_wait_ms": round(record.elapsed_ms, 1) if record is not None else "",
                    "rejected_reappeared": "" if reappeared is None else reappeared,
                    "candidate_ids": "|".join(out["pool"]), "result_ids": "|".join(out["results"]),
                })
            if checkpoint is not None:
                checkpoint(detail)
    os.environ[SWITCH] = "0"
    return detail


def summarize(detail: List[dict], runs: int) -> List[dict]:
    """방식·회차별 지표와, 회차를 합친 평균 줄."""
    summary: List[dict] = []
    for mode in MODES:
        per_run: List[dict] = []
        for run in range(1, runs + 1):
            rows = [row for row in detail if row["mode"] == mode and row["run"] == run]
            if not rows:
                continue
            latencies = [row["latency_ms"] for row in rows]
            per_run.append({
                "mode": mode, "run": run, "queries": len(rows),
                "candidate_recall": round(statistics.fmean(row["candidate_hit"] for row in rows), 4),
                "hit1": round(statistics.fmean(row["hit1"] for row in rows), 4),
                "hit10": round(statistics.fmean(row["hit10"] for row in rows), 4),
                "hit15": round(statistics.fmean(row["hit15"] for row in rows), 4),
                "mrr10": round(statistics.fmean(row["mrr10"] for row in rows), 4),
                "latency_median_ms": round(statistics.median(latencies), 1),
                "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
                "rerank_failed": sum(1 for row in rows if "failed" in row["rerank_runs"].split("|")),
                "supplement_applied": sum(1 for row in rows if row["feedback_status"] == "applied"),
                "entered": sum(len(row["entered_ids"].split("|")) for row in rows if row["entered_ids"]),
                "entered_judged": sum(
                    len(row["entered_judged_ids"].split("|")) for row in rows if row.get("entered_judged_ids")),
            })
        summary.extend(per_run)
        if per_run:
            keys = [key for key in per_run[0] if key not in {"mode", "run"}]
            summary.append({"mode": mode, "run": "mean",
                            **{key: round(statistics.fmean(row[key] for row in per_run), 4) for key in keys}})
    return summary


def changes(detail: List[dict], runs: int) -> List[dict]:
    """질의별 off → on 변화. 회복·손실, 보충으로 Top-10에 든 오답, 거절 곡 재등장."""
    by_key = {(row["run"], row["mode"], row["query_id"]): row for row in detail}
    out: List[dict] = []
    for run in range(1, runs + 1):
        for query_id in dict.fromkeys(row["query_id"] for row in detail):
            off, on = by_key.get((run, "off", query_id)), by_key.get((run, "on", query_id))
            if off is None or on is None:
                continue
            relevant = set(off["relevant_ids"].split("|"))
            on_top10, off_top10 = on["result_ids"].split("|")[:10], off["result_ids"].split("|")[:10]
            wrong = [song_id for song_id in on["supplemented_ids"].split("|")
                     if song_id and song_id not in relevant and song_id in on_top10 and song_id not in off_top10]

            def moved(key: str) -> str:
                if off[key] == on[key]:
                    return ""
                return "recovered" if on[key] > off[key] else "lost"

            out.append({
                "run": run, "query_id": query_id,
                "candidate": moved("candidate_hit"), "hit10": moved("hit10"), "hit15": moved("hit15"),
                "rank_off": off["rank"], "rank_on": on["rank"],
                "candidate_rank_off": off["candidate_rank"], "candidate_rank_on": on["candidate_rank"],
                "feedback_status": on["feedback_status"], "supplemented_cases": on["supplemented_cases"],
                "entered_ids": on["entered_ids"], "displaced_ids": on["displaced_ids"],
                "wrong_supplement_in_top10": "|".join(wrong),
                "rejected_reappeared": on["rejected_reappeared"],
            })
    return out


def gates(detail: List[dict], summary: List[dict], change_rows: List[dict], runs: int, reranked: bool) -> Dict[str, Any]:
    """채택 기준 점검표. 잴 수 있는 것만 판정하고 나머지는 None으로 둔다."""
    means = {row["mode"]: row for row in summary if row["run"] == "mean"}
    off_runs = [row for row in summary if row["mode"] == "off" and row["run"] != "mean"]
    # off는 피드백 경로를 타지 않는다. 회차 사이에 후보 목록이 같아야 한다(검색은 결정적이다).
    off_pools = {}
    off_changed = 0
    for row in detail:
        if row["mode"] != "off":
            continue
        first = off_pools.setdefault(row["query_id"], row["candidate_ids"])
        off_changed += int(first != row["candidate_ids"])
    reappear = [row["rejected_reappeared"] for row in change_rows if row["rejected_reappeared"] != ""]
    spread = (max(row["hit10"] for row in off_runs) - min(row["hit10"] for row in off_runs)) if off_runs else 0.0
    gain = means["on"]["hit10"] - means["off"]["hit10"] if len(means) == 2 else 0.0
    failed = sum(row["rerank_failed"] for row in summary if row["run"] != "mean")
    on_runs = [row for row in summary if row["mode"] == "on" and row["run"] != "mean"]
    entered_total = sum(row["entered"] for row in on_runs)
    judged_total = sum(row.get("entered_judged", 0) for row in on_runs)
    return {
        "hit10_lost_queries": sum(1 for row in change_rows if row["hit10"] == "lost"),
        "candidate_lost_queries": sum(1 for row in change_rows if row["candidate"] == "lost"),
        "hit10_no_loss": all(row["hit10"] != "lost" for row in change_rows),
        "rejected_reappeared_total": sum(reappear) if reappear else None,
        "rejected_reappear_zero": (sum(reappear) == 0) if reappear else None,
        "off_candidate_changes_between_runs": off_changed,
        "off_unchanged": off_changed == 0 if runs > 1 else None,
        "latency_median_delta_ms": round(means["on"]["latency_median_ms"] - means["off"]["latency_median_ms"], 1),
        "latency_p95_delta_ms": round(means["on"]["latency_p95_ms"] - means["off"]["latency_p95_ms"], 1),
        "latency_within_budget": (
            means["on"]["latency_median_ms"] - means["off"]["latency_median_ms"] <= LATENCY_BUDGET_MS
            and means["on"]["latency_p95_ms"] - means["off"]["latency_p95_ms"] <= LATENCY_BUDGET_MS),
        "hit10_gain": round(gain, 4), "off_hit10_spread_between_runs": round(spread, 4),
        # 반복 변동보다 커야 개선이라 할 수 있다. 리랭킹 없이 잰 값은 순위 지표의 근거가 아니다.
        "gain_exceeds_run_variation": (gain > spread) if (reranked and runs > 1) else None,
        "rerank_failed_rows": failed,
        "comparable": failed == 0,
        # 리랭킹을 돌린 측정에서 들어온 곡이 리랭커 평가 범위 안에 있었는가. 범위 밖이 있으면 "보충은 됐는데
        # 효과 없음"이 사례 탓이 아니라 설정 탓일 수 있다(리랭커가 후보 끝까지 평가하지 않았다).
        "entered_total": entered_total,
        "entered_judged_by_reranker": judged_total if reranked else None,
        "entrants_outside_judged_range": (judged_total < entered_total) if reranked else None,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _case_info(bench: Bench, args: argparse.Namespace) -> Dict[str, Any]:
    if bench.file_cases is None:
        # 운영 사례는 지문·건수를 여기서 세지 않는다 — 원문을 읽지 않고, 측정 뒤에 남는 것은 곡 ID와 집계뿐이다.
        return {"source": "live", "embedding_version": EMBEDDING_VERSION}
    cases: List[EvalCase] = bench.file_cases.cases
    return {
        "source": str(args.cases), "digest": cases_digest(cases), "count": len(cases),
        "by_group": {group: sum(1 for case in cases if case.group == group) for group in CASE_GROUPS},
        "embedding_version": EMBEDDING_VERSION,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    require_open_split(args.split, args.sealed_final)
    if args.runs < 2:
        print("경고: 한 번만 재면 반복 변동을 알 수 없다. 채택 판단에는 --runs 2 이상을 쓴다.", flush=True)
    rows = read_queries(Path(args.input), args.split)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "feedback_reuse_detail.csv"

    # 산출물은 **스택을 닫기 전에** 쓴다. 닫다가 난 오류가 다 잰 결과를 버리지 않게(리뷰 FR-09).
    with open_bench(args) as bench:
        detail = asyncio.run(run_compare(bench, rows, args, checkpoint=lambda done: _write_csv(detail_path, done)))
        summary = summarize(detail, args.runs)
        change_rows = changes(detail, args.runs)
        report = gates(detail, summary, change_rows, args.runs, args.rerank)
        _write_csv(detail_path, detail)
        _write_csv(output / "feedback_reuse_summary.csv", summary)
        _write_csv(output / "feedback_reuse_changes.csv", change_rows)
        settings = RetrievalSettings.from_env()
        (output / "feedback_reuse_runinfo.json").write_text(json.dumps({
            "git_head": _git_head(), "input": str(args.input), "input_sha256_16": _sha16(Path(args.input)),
            "split": args.split, "queries": len(rows), "runs": args.runs, "top_k": args.top_k,
            "candidate_k": args.candidate_k, "rerank": args.rerank, "reject_check": args.reject_check,
            "settings": settings.__dict__, "cases": _case_info(bench, args), "bench": bench.info, "gates": report,
            "note": ("합성 자료로 돈 기능 확인이다. 검색 품질 개선의 근거로 쓰지 않는다." if args.synthetic else ""),
        }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print("\n방식  회차  후보Recall  Hit@1  Hit@10  Hit@15  MRR@10  중앙값ms  p95ms  보충  진입")
    for row in summary:
        print(f"{row['mode']:<4}  {str(row['run']):<4}  {row['candidate_recall']:<10}  {row['hit1']:<5}  "
              f"{row['hit10']:<6}  {row['hit15']:<6}  {row['mrr10']:<6}  {row['latency_median_ms']:<8}  "
              f"{row['latency_p95_ms']:<6} {row['supplement_applied']:<4}  {row['entered']}")
    print("\n채택 기준 점검:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\n산출물: {output}")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def similarity_rows(embedder: Any, cases: List[EvalCase], rows: List[dict]) -> List[dict]:
    """질의 × 사례의 코사인 유사도. 검색과 같은 규약(질의는 `query: `, 사례는 `passage: `)으로 임베딩한다."""
    import numpy as np

    case_vectors = np.asarray(embedder.embed_passages([case.reusable_text for case in cases], normalize=True))
    query_vectors = np.asarray(embedder.embed_passages(
        [f"query: {row['query'].strip()}" for row in rows], add_e5_prefix=False, normalize=True))
    scores = query_vectors @ case_vectors.T
    out: List[dict] = []
    for q_index, row in enumerate(rows):
        relevant = parse_relevant_ids(row["relevant_ids"])
        for c_index, case in enumerate(cases):
            out.append({
                "query_id": row["query_id"].strip(), "case_id": case.case_id, "group": case.group,
                "same_song": int(case.song_id in relevant), "similarity": round(float(scores[q_index, c_index]), 4),
            })
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    require_open_split(args.split, args.sealed_final)
    from src.embedding.models.text_koe5 import KoE5Embedder

    cases = load_cases(Path(args.cases), require_expiry=not cases_are_synthetic(args))
    rows = read_queries(Path(args.input), args.split)
    pairs = similarity_rows(KoE5Embedder(), cases, rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "feedback_reuse_similarity.csv", pairs)
    threshold = RetrievalSettings.from_env().min_similarity
    print(f"질의 {len(rows)}건 × 사례 {len(cases)}건 · 현재 문턱 {threshold}")
    print("묶음                       정답곡  쌍수   최소    중앙값  최대    문턱 이상")
    table: List[dict] = []
    for group in CASE_GROUPS:
        for same in (1, 0):
            values = [pair["similarity"] for pair in pairs if pair["group"] == group and pair["same_song"] == same]
            if not values:
                continue
            passing = sum(1 for value in values if value >= threshold)
            table.append({"group": group, "same_song": same, "pairs": len(values), "min": min(values),
                          "median": round(statistics.median(values), 4), "max": max(values), "at_or_above_threshold": passing})
            print(f"{group:<26} {same:<6}  {len(values):<5}  {min(values):<6}  {statistics.median(values):<6.4f}  "
                  f"{max(values):<6}  {passing}")
    _write_csv(output / "feedback_reuse_similarity_summary.csv", table)
    print(f"\n산출물: {output}")
    return 0


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def cmd_harness(args: argparse.Namespace) -> int:
    """사례 파일을 끼우고 스위치를 켠 채 기존 측정기를 그대로 돌린다(재질문 손실·표준 정확도 측정용)."""
    allowed = {"src.retrieval.evaluate_search_accuracy", "src.retrieval.evaluate_clarification",
               "src.retrieval.trace_candidate_drop"}
    if args.module not in allowed:
        raise SystemExit(f"돌릴 수 있는 측정기: {sorted(allowed)}")
    rest = [item for item in args.rest if item != "--"]
    # 측정기마다 --split의 기본값이 다르고 all을 받는 것도 있다. 값을 명시하게 하고, 전부 dev일 때만 통과시킨다.
    splits = harness_splits(rest)
    if not args.sealed_final and (not splits or any(value.strip().lower() != OPEN_SPLIT for value in splits)):
        raise SystemExit(f"측정기 인자에 --split dev를 명시한다. {SEALED_MESSAGE}")
    display_k = harness_display_k(args.display_k, rest)
    print(f"피드백 후보 보충을 켜고 {args.module}을(를) 돌린다 — 표시 수 {display_k}곡 기준으로 보충 자리를 정한다.", flush=True)
    # 끼운 사례·표시 수·스위치·argv는 측정기가 끝나면(실패해도) 원래대로 돌려놓는다. 표시 수는 환경 변수가 아니라
    # 이 프로세스 안의 값으로만 넘긴다(`measurement_display`) — 서버 환경으로 새어 나갈 길이 없다.
    previous_switch, previous_argv = os.environ.get(SWITCH), list(sys.argv)
    with installed_cases(args.cases, cases_are_synthetic(args)), measurement_display(display_k):
        os.environ[SWITCH] = "1"
        sys.argv = [args.module, *rest]
        try:
            runpy.run_module(args.module, run_name="__main__")
        finally:
            sys.argv = previous_argv
            if previous_switch is None:
                os.environ.pop(SWITCH, None)
            else:
                os.environ[SWITCH] = previous_switch
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cases", default=str(SYNTHETIC_DIR / "cases.jsonl"), help="사례 JSONL 경로 또는 live")
        p.add_argument("--input", default=str(SYNTHETIC_DIR / "queries.csv"), help="평가 질의 CSV")
        p.add_argument("--split", default=OPEN_SPLIT, choices=[OPEN_SPLIT, "test"])
        p.add_argument("--output-dir", required=True)
        p.add_argument("--sealed-final", action="store_true", help="봉인 test를 여는 최종 1회 측정일 때만 준다")
        p.add_argument("--synthetic-cases", action="store_true",
                       help="사례 파일이 합성 자료다(줄마다 expires_at이 없어도 받는다). 실사용 사례에는 주지 않는다")

    compare = sub.add_parser("compare", help="off/on 비교 측정")
    common(compare)
    compare.add_argument("--analysis-cache", help="분석 캐시(실제 코퍼스 측정에 필수)")
    compare.add_argument("--runs", type=int, default=2)
    compare.add_argument("--top-k", type=int, default=15)
    compare.add_argument("--candidate-k", type=int, default=30)
    compare.add_argument("--rerank", action="store_true", help="리랭킹까지 돌린다(Gemini 호출). 없으면 후보 단계만")
    compare.add_argument("--reject-check", action="store_true", help="보여준 곡을 거절한 다음 턴에 그 곡이 되돌아오는지 본다")
    compare.add_argument("--synthetic", action="store_true", help="합성 곡으로 도는 자체 완결 실행(기능 확인용)")
    compare.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR)
    compare.set_defaults(func=cmd_compare)

    calibrate = sub.add_parser("calibrate", help="질의·사례 유사도 분포(검색 없음)")
    common(calibrate)
    calibrate.set_defaults(func=cmd_calibrate)

    harness = sub.add_parser("harness", help="사례 파일을 끼우고 기존 측정기를 돌린다")
    harness.add_argument("--cases", required=True, help="사례 JSONL 경로 또는 live")
    harness.add_argument("--module", required=True)
    harness.add_argument("--sealed-final", action="store_true")
    harness.add_argument("--synthetic-cases", action="store_true", help="사례 파일이 합성 자료다(만료 없이 받는다)")
    harness.add_argument("--display-k", type=int,
                         help="측정기가 결과로 쓰는 곡 수(1~100). 생략하면 측정기 인자의 --top-k(없으면 10). 보충 곡은 이 수 뒤의 자리에만 선다")
    harness.add_argument("rest", nargs=argparse.REMAINDER, help="-- 뒤에 그 측정기의 인자를 그대로 적는다")
    harness.set_defaults(func=cmd_harness)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare" and not args.synthetic and not args.analysis_cache:
        raise SystemExit("실제 코퍼스 측정에는 --analysis-cache가 필요하다(분석이 갈리면 off/on 차이와 섞인다).")
    if args.command == "compare" and args.synthetic and args.cases == "live":
        raise SystemExit("--synthetic에는 사례 파일이 필요하다.")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
