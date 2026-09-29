#!/usr/bin/env python
"""평가 질의를 한 번 분석해 캐시로 저장한다.

    venv/bin/python -m src.retrieval.build_analysis_cache \
        --input experiments/reranking/eval_queries_v06.csv --split dev \
        --output experiments/reranking/analysis_cache_v06_dev.json

왜 따로 만드나: 측정 스크립트가 분석까지 하면 같은 측정을 두 번 해도 숫자가 달라진다
(자세한 이유는 `analysis_cache.py` 문서). 분석을 먼저 고정해 두면 검색·랭킹 변경의
전후 비교가 성립한다.

기본적으로 **이미 있는 항목은 다시 분석하지 않는다.** Gemini 호출은 쿼터를 쓰므로,
누락·폴백만 채우는 것이 기본 동작이다. 전부 다시 만들려면 `--force`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import get_query_analyzer
from src.retrieval.analysis_cache import (
    AnalysisCacheError,
    analyzer_fingerprint,
    analyzer_sha,
    load_cache,
    looks_like_fallback,
    new_cache,
)
from src.retrieval.evaluate_search_accuracy import read_queries


def build(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.getenv("SEARCH_REFERENCE_YEAR", "").strip():
        # 상대 시기("작년", "최근")를 해석하는 기준이 실행 시점의 연도가 된다.
        # 고정하지 않으면 해가 바뀌는 순간 같은 명령이 다른 캐시를 만든다.
        message = (
            "SEARCH_REFERENCE_YEAR가 설정되지 않았습니다. 상대 시기 해석 기준이 "
            "실행 시점의 연도가 되므로 캐시를 다시 만들 때 결과가 달라질 수 있습니다."
        )
        if not args.allow_floating_year:
            print(f"오류: {message}", file=sys.stderr)
            print(
                "  .env에 SEARCH_REFERENCE_YEAR=2026 처럼 고정하거나, "
                "의도한 것이면 --allow-floating-year를 주세요.",
                file=sys.stderr,
            )
            return 2
        print(f"경고: {message}", flush=True)

    rows = read_queries(input_path, None if args.split == "all" else args.split)
    if args.query_ids:
        wanted = {p.strip() for p in args.query_ids.split(",") if p.strip()}
        rows = [row for row in rows if row["query_id"].strip() in wanted]
        if not rows:
            print("오류: --query-ids에 해당하는 질의가 없습니다.", file=sys.stderr)
            return 2

    cache = new_cache(source=str(input_path), split=args.split)
    reused = 0
    if output_path.exists() and not args.force:
        try:
            existing = load_cache(output_path)
        except AnalysisCacheError as exc:
            print(f"오류: 기존 캐시를 읽을 수 없습니다 — {exc}", file=sys.stderr)
            return 2
        drift = existing.fingerprint_drift()
        if drift and not args.reuse_despite_drift:
            print(
                "경고: 기존 캐시를 만든 조건이 지금과 다릅니다 "
                f"({', '.join(drift)}). 재사용하지 않고 전부 다시 분석합니다.\n"
                "  분석 결과를 정하는 코드가 그대로임을 확인했다면 "
                "--reuse-despite-drift로 유지할 수 있습니다.",
                flush=True,
            )
        else:
            cache.meta = existing.meta
            cache.entries = dict(existing.entries)
            reused = len(cache.entries)
            if drift:
                # **숨기지 않는다.** 옛 분석을 그대로 들고 가되, 무엇이 어긋난
                # 채로 그렇게 했는지 캐시에 남긴다. 이 캐시로 낸 숫자를 나중에
                # 되짚는 사람이 판단할 수 있어야 한다.
                print(
                    f"조건이 어긋났지만({', '.join(drift)}) 기존 분석을 유지합니다 "
                    "(--reuse-despite-drift).",
                    flush=True,
                )
                cache.meta.setdefault("drift_notes", []).append({
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "drifted": drift,
                    "kept_entries": len(existing.entries),
                    "analyzer_now": analyzer_fingerprint(),
                    # 이 지문으로 만들어진 항목은 entries의 analyzer_sha로 찾는다.
                    # 목록을 여기 따로 적지 않는다 — 작업이 끝나야 쓰는 기록은
                    # 중간에 끊기면 사라진다.
                    "analyzer_sha_now": analyzer_sha(),
                    "reason": args.drift_reason,
                })

    todo = []
    for row in rows:
        query_id = row["query_id"].strip()
        query = row["query"].strip()
        entry = cache.entries.get(query_id)
        if entry is None or entry["query"] != query:
            todo.append((query_id, query))
        elif entry.get("fallback") and not args.keep_fallback:
            # 폴백은 API 장애의 결과지 분석 결과가 아니다. 다시 시도한다.
            todo.append((query_id, query))

    # 그대로 쓰는 항목 = 이번 질의 중 캐시에 있고 다시 분석하지 않는 것.
    # `reused - len(todo)`로 세면 캐시에 없던 질의가 todo에 들어오는 만큼
    # 음수가 나온다(실제로 "재사용 -1건"이 찍혔다).
    todo_ids = {query_id for query_id, _ in todo}
    kept = sum(
        1 for row in rows
        if row["query_id"].strip() in cache.entries
        and row["query_id"].strip() not in todo_ids
    )
    print(
        f"질의 {len(rows)}건 | 재사용 {kept}건 | 새로 분석 {len(todo)}건",
        flush=True,
    )
    if not todo:
        cache.save(output_path)
        print(f"캐시: {output_path}")
        return 0

    analyzer = get_query_analyzer()
    fallbacks = []
    for index, (query_id, query) in enumerate(todo, start=1):
        print(f"[{index}/{len(todo)}] {query_id}: {query}", flush=True)
        analysis = analyzer.analyze(query)
        cache.put(query_id, query, analysis)
        if looks_like_fallback(analysis):
            fallbacks.append(query_id)
            print("  경고: 규칙 폴백이 반환됐습니다(Gemini 실패).", flush=True)
        # 한 건씩 저장한다 — 중간에 끊겨도 쓴 만큼은 남아야 다시 부르지 않는다.
        cache.save(output_path)

    print()
    print(f"캐시: {output_path} (총 {len(cache.entries)}건)")
    groups = cache.provenance()
    if len(groups) > 1:
        # 한 캐시 안에 서로 다른 분석기가 만든 항목이 섞여 있다. 숨기지 않는다.
        print("항목을 만든 분석기가 둘 이상이다:")
        for sha, ids in sorted(groups.items()):
            head = ", ".join(ids[:6]) + (" …" if len(ids) > 6 else "")
            print(f"  {sha}: {len(ids)}건 ({head})")
    remaining = cache.fallback_ids()
    if remaining:
        print(
            f"폴백으로 남은 질의 {len(remaining)}건: {', '.join(remaining)}\n"
            "  이 질의는 제목·가사 단서가 비어 있어 측정값이 검색과 무관한 이유로\n"
            "  떨어집니다. 다시 실행해 채우거나, 의도한 것이면 측정에서\n"
            "  --allow-fallback-analysis를 주세요."
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="평가 질의의 QueryAnalysis를 한 번 분석해 캐시로 저장합니다."
    )
    parser.add_argument(
        "--input",
        default="experiments/reranking/eval_queries_v06.csv",
        help="평가 질의 CSV 경로. 기본값이 기준 세트다",
    )
    parser.add_argument("--output", required=True, help="저장할 캐시 JSON 경로")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        required=True,
        help="분석할 데이터 분할. 측정 쪽과 같은 이유로 생략할 수 없다",
    )
    parser.add_argument(
        "--query-ids", default="", help="쉼표로 구분한 query_id만 분석"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 캐시를 무시하고 전부 다시 분석한다(Gemini 호출이 늘어난다)",
    )
    parser.add_argument(
        "--keep-fallback",
        action="store_true",
        help="폴백으로 저장된 항목을 다시 시도하지 않는다",
    )
    parser.add_argument(
        "--reuse-despite-drift",
        action="store_true",
        help=(
            "분석기 조건이 어긋나도 기존 분석을 유지한다. **전후 비교용이다** — "
            "같은 질의를 같은 분석으로 재야 랭킹·라벨 변경만 남는다. "
            "결과를 정하는 코드(프롬프트·safeguards·modality)가 그대로임을 "
            "확인했을 때만 쓴다. 어긋난 내용은 캐시 meta에 남는다"
        ),
    )
    parser.add_argument(
        "--drift-reason",
        default="",
        help="--reuse-despite-drift를 쓴 이유. 캐시 meta에 그대로 남는다",
    )
    parser.add_argument(
        "--allow-floating-year",
        action="store_true",
        help="SEARCH_REFERENCE_YEAR 없이 진행한다(재현성 보장 안 됨)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(build(build_parser().parse_args()))
