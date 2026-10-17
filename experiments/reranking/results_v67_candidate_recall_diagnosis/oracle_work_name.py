"""[진단 전용 · 오라클] 간접 지칭(작품·프로그램을 줄거리로만 말함) 질의에서 **작품명을 분석이 풀었다면** 후보에 드는가 — 상한 추정.

**후보 생성 알고리즘이 아니다.** 작품명은 사람이 질의 서술에서 알아볼 수 있는 고유명(드라마·영화·예능·경연 프로그램)을 적은 것이고,
곡 제목·가수명은 넣지 않는다(정답 직접 주입 방지). 분석의 `korean_tags` 끝에 그 이름 하나를 더해 BM25에만 닿게 하고,
나머지 분석·경로·가중은 그대로 둔다. 리랭킹 없음, 모델 호출 0회. 대조로 정답 자료에 이름이 없는 질의(c607·t043)를 함께 넣는다.

    V67_AUDIO=on venv/bin/python oracle_work_name.py --out analysis/oracle_work_name.csv --expect-server-state server_state_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v67_eval_runtime as rt  # noqa: E402
import server_state  # noqa: E402

# 질의 서술이 가리키는 작품·프로그램 이름(사람이 서술에서 알아볼 수 있는 것). 정답 자료에 이름이 있는지는 진단표 근거 열 참조.
WORK_NAMES = {
    "c602": "쇼미더머니", "c603": "슈퍼스타K", "n033": "눈물의 여왕", "n043": "슬기로운 의사생활", "t038": "슈퍼스타K",
    "t040": "청연", "c701": "별에서 온 그대", "n040": "좋니", "n096": "봄날은 간다",
    "c705": "베토벤 바이러스", "c707": "냄새를 보는 소녀",
    # 대조 — 정답 자료(album_summary·Context 사실)에 이름이 없다
    "c607": "응답하라 1988", "t043": "케이팝 데몬 헌터스",
}
# n096의 영화명은 곡 제목과 같다 — 결과를 따로 표시한다(제목 일치로 닿는 것이라 상한 해석에서 뺀다)
TITLE_OVERLAP = {"n096"}


async def run() -> list:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries
    import src.backend.api.dependencies as deps

    router = deps.get_search_router()
    out = []
    for s in rt.SETS:
        cache = load_cache(rt.CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for row in read_queries(rt.ROOT / f"experiments/reranking/eval_queries_{s}.csv", "dev"):
            qid = row["query_id"].strip()
            if qid not in WORK_NAMES:
                continue
            rel = parse_relevant_ids(row["relevant_ids"])
            base = cache.get(qid, row["query"].strip())
            res = {}
            for name, analysis in (("base", base),
                                   ("oracle", base.model_copy(update={"korean_tags": [*base.korean_tags, WORK_NAMES[qid]]}))):
                pool: list = []
                await router.search(analysis, top_k=30, use_rerank=False, candidate_k=30, candidate_ids_out=pool)
                res[name] = next((i for i, x in enumerate(pool, 1) if x in rel), "")
            out.append({"query_id": qid, "work_name_added": "(이름)", "title_overlap": int(qid in TITLE_OVERLAP),
                        "control_no_name_in_data": int(qid in ("c607", "t043")),
                        "pool_rank_base": res["base"], "pool_rank_oracle": res["oracle"]})
            print(qid, res, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-server-state", type=Path, required=True)
    args = ap.parse_args()
    os.chdir(rt.ROOT)
    if server_state.diff(server_state.capture(), json.loads(args.expect_server_state.read_text(encoding="utf-8"))):
        return 4
    rt.install_guards()
    rt.code_identity()
    rt.verify_guards()
    rt.install_audio_config()
    rt.install_lyrics_snapshot()
    rows = asyncio.run(run())
    if len(rows) != len(WORK_NAMES):
        raise SystemExit("질의 누락")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 1 if rt.NET_ATTEMPTS else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
