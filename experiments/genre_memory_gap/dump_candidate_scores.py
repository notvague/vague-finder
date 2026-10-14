#!/usr/bin/env python
"""후보 30곡의 ID·점수를 그대로 적는다 — 기반 커밋과 이 브랜치가 기본값에서 같은 결과를 내는지 견줄 때 쓴다.

질의마다 세 가지 검색을 한다(리랭킹 없음, Gemini 호출 없음).

    initial   최초 검색
    reject    앞 10곡을 거절한 재검색
    answer    거절 + 재질문 답변 — 남은 후보에서 가장 많은 성별 하나, 가장 많은 첫 등록 장르 하나를 차례로 답한다

점수는 반올림하지 않고 `repr`로 적는다. 두 체크아웃에서 만든 파일이 바이트까지 같으면 후보·순서·점수가 같다.
이 스크립트는 기반 커밋에도 있는 API만 쓴다(`router.search`의 인자, `ClarifyAnswer`) — 그 폴더를 cwd로 두고 실행한다.

    (cd <체크아웃> && venv/bin/python <이 파일> --input <질의.csv> --analysis-cache <캐시.json> --output <파일.csv>)

dev 분할만 읽는다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.getcwd())


async def run(args: argparse.Namespace) -> None:
    from src.backend.api.dependencies import get_search_router
    from src.backend.schemas.search import ClarifyAnswer
    from src.retrieval.analysis_cache import load_cache

    with open(args.input, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get("split") or "").strip().lower() == "dev" and (r.get("query_type") or "").strip() == "search"
                and (r.get("relevant_ids") or "").strip()]
    cache = load_cache(Path(args.analysis_cache))
    router = get_search_router()
    out: List[dict] = []

    async def search(analysis, kind: str, qid: str, exclude: Optional[List[str]] = None, answers=None) -> List:
        tracks: List = []
        await router.search(
            analysis, top_k=args.candidate_k, use_rerank=False, candidate_k=args.candidate_k,
            exclude_ids=exclude, answers=answers, candidate_tracks_out=tracks,
        )
        out.append({
            "query_id": qid, "search": kind,
            "candidates": "|".join(f"{t.id}:{float(t.score)!r}" for t in tracks),
        })
        return tracks

    for index, row in enumerate(rows, start=1):
        qid = row["query_id"].strip()
        analysis = cache.get(qid, row["query"].strip())
        first = await search(analysis, "initial", qid)
        rejected = [t.id for t in first[:args.top_k]]
        after = await search(analysis, "reject", qid, exclude=rejected)
        genders = Counter((t.vocal_gender or "").strip() for t in after if (t.vocal_gender or "").strip())
        genres = Counter((t.genre or "").split(",")[0].strip() for t in after if (t.genre or "").strip())
        for slot, counts in (("vocal_gender", genders), ("genre", genres)):
            if counts:
                value = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                await search(analysis, f"answer:{slot}={value}", qid, exclude=rejected,
                             answers=[ClarifyAnswer(slot=slot, value=value)])
        print(f"[{index}/{len(rows)}] {qid}", flush=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "search", "candidates"])
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)}행 → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--analysis-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
