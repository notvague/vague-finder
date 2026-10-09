"""재질문 답변 보너스가 리랭커에 넘기는 후보 30곡의 **순서**를 얼마나 바꾸는지 — 첫 검색 조건과 거절 뒤 조건.

RUN_INFO "왜 m402 발라드·솔로는 남는가"의 근거 표(`candidate_order_after_reject.csv`)를 만든다. (a) 변형(보너스 전 순서 고정)을
재현할 때도 같은 계산이 필요하다. 라우터의 후보 검색만 돌린다(`use_rerank=False`) — 모델·Qdrant·Mongo는 올리지만 Gemini는 부르지 않는다.
서버가 떠 있으면 로컬 Qdrant 때문에 실패한다.

실행 (레포 루트):
    venv/bin/python experiments/reranking/results_clarify_v10_corrections/candidate_order_after_reject.py \\
        [--query-ids m402,c603] [--initial-from results_clarify_v10_corrections/base_r1] [--queries docs/eval/queries.json] [--out <csv>]

- 거절 대상(첫 검색 Top-10)은 `--initial-from` 폴더의 clarify_detail.csv `initial` 행 shown_ids를 쓴다 — 측정과 같은 거절 집합
- 목표 곡은 하네스와 같이 `queries.json`의 positives[0], 답은 하네스와 같은 `oracle_value`(정답 곡 메타데이터)
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

# 파일 경로로 실행하므로 레포 루트를 import 경로에 넣는다 (src 패키지)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")

from src.backend.api.dependencies import get_search_router  # noqa: E402
from src.backend.schemas.query import QueryAnalysis  # noqa: E402
from src.backend.schemas.search import ClarifyAnswer  # noqa: E402
from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set  # noqa: E402
from src.retrieval.evaluate_clarification import DEFAULT_CORPUS, SLOTS, load_corpus, oracle_value  # noqa: E402

ROOT = Path("experiments/reranking")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-ids", default="m402,c603")
    p.add_argument("--analysis-cache", default=str(ROOT / "analysis_cache_v06_dev.json"))
    p.add_argument("--queries", default=str(DEFAULT_EVAL_PATH), help="목표 곡은 하네스와 같이 queries.json의 positives[0]")
    p.add_argument("--initial-from", default=str(ROOT / "results_clarify_v10_corrections/base_r1"))
    p.add_argument("--out", default=str(ROOT / "results_clarify_v10_corrections/candidate_order_after_reject.csv"))
    return p.parse_args()


async def _candidates(router, analysis: QueryAnalysis, answers: list[ClarifyAnswer], exclude: list[str]) -> list[str]:
    pool: list[str] = []
    tracks: list = []
    await router.search(
        analysis, top_k=30, use_rerank=False, candidate_k=30, exclude_ids=exclude,
        candidate_ids_out=pool, candidate_tracks_out=tracks, answers=answers,
    )
    return [str(t.id) for t in tracks] or pool


async def main() -> None:
    args = _parse()
    query_ids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    cache = json.load(open(args.analysis_cache, encoding="utf-8"))
    entries = cache.get("entries") or cache.get("queries")
    positives = {q.query_id: q.positives for q in load_eval_set(Path(args.queries)).queries}
    corpus = load_corpus(DEFAULT_CORPUS)
    initial = {
        r["query_id"]: r["shown_ids"].split("|")
        for r in csv.DictReader(open(Path(args.initial_from) / "clarify_detail.csv", encoding="utf-8-sig"))
        if r["policy"] == "initial"
    }
    router = get_search_router()
    rows: list[dict] = []
    for qid in query_ids:
        entry = entries[qid]
        analysis = QueryAnalysis(**(entry.get("analysis", entry)))
        target = positives[qid][0]  # 하네스(evaluate_clarification)와 같은 출처
        song = corpus.get(target)
        for condition, exclude in (("first_search", []), ("after_reject", initial[qid])):
            base = await _candidates(router, analysis, [], exclude)
            for slot in SLOTS:
                value = oracle_value(song, slot) if song else None
                if not value:
                    continue
                with_answer = await _candidates(router, analysis, [ClarifyAnswer(slot=slot, value=value)], exclude)
                rows.append({
                    "query_id": qid, "condition": condition, "slot": slot, "value": value,
                    "same_set": set(with_answer) == set(base),
                    "positions_changed": sum(1 for i in range(min(len(with_answer), len(base))) if with_answer[i] != base[i]),
                    "target_rank_none": base.index(target) + 1 if target in base else "",
                    "target_rank_answer": with_answer.index(target) + 1 if target in with_answer else "",
                })
    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out} ({len(rows)}행)")


if __name__ == "__main__":
    asyncio.run(main())
