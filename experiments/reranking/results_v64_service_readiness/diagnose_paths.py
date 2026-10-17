"""§5-2 진단 2 — 로컬↔서버 후보 차이가 **어느 경로**에서 생기는가. 경로별 곡 순위(ExplainRecorder)를 모드마다 덤프한다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/diagnose_paths.py <local|server> qid,qid,...
출력: diagnose_paths_<mode>.json — {qid: {path: [[song_id, rank], ...]}}
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], sys.argv[1], *sys.argv[2:]]
import diagnose_candidates as dc  # noqa: E402  (모드·환경·차단 설정을 그대로 쓴다)
from src.retrieval.explain import ExplainRecorder  # noqa: E402


def main() -> int:
    wanted = sys.argv[2].split(",")
    analyzer = dc.v64_runtime.get_cached_analyzer()
    by_id = {qid: query for query, (qid, *_rest) in analyzer.by_query.items()}
    router = dc.deps.get_search_router()
    out = {}
    for qid in wanted:
        analysis = analyzer.analyze(by_id[qid])
        recorder = ExplainRecorder(by_id[qid])
        tracks = asyncio.run(router.search(analysis, top_k=30, use_rerank=False, candidate_k=30, recorder=recorder))
        paths: dict = {}
        for song_id, song in recorder.record.songs.items():
            for contribution in song.paths:
                paths.setdefault(contribution.path, []).append([song_id, contribution.rank, round(contribution.delta, 9)])
        out[qid] = {"final": [t.id for t in tracks],
                    "paths": {p: sorted(v, key=lambda x: (x[1], x[0])) for p, v in sorted(paths.items())}}
    (HERE / f"diagnose_paths_{dc.MODE}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    router.shutdown()
    print("ok", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
