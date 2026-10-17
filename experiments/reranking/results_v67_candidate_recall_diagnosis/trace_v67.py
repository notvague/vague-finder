"""진단용 단계 추적 — 정답 곡마다 경로별 순위(현행 깊이 / 전체 깊이 5,143)·단계·융합/가산 뒤 순위. 리랭킹 없음, 모델 호출 0회.

**실제 후보 생성과 분리된 별도 실행이다.** 라우터를 `use_rerank=False`로 두 번(현행 깊이 · `path_k=5143`) 부르고 explain 기록만 읽는다.
정답 ID는 기록을 읽을 때만 쓰며, 검색에 넘기거나 가산하지 않는다. 단계 정의는 `src.retrieval.trace_candidate_drop`의 것을 그대로 쓴다.
`src/retrieval/trace_candidate_drop.py`와 다른 점: 정답이 여럿이면 **모든 정답 곡**을 적는다(v50 진단표는 정렬 첫 곡 하나), 모든 경로(보조 포함)의
현행/전체 깊이 순위와 경로별 받은 목록 길이를 남긴다, v67 하네스(서버·차단·가사 스냅샷·소리 구성)를 쓴다.

    V67_AUDIO=on venv/bin/python trace_v67.py --out <폴더>/trace_on.csv --expect-server-state server_state_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v67_eval_runtime as rt  # noqa: E402
import server_state  # noqa: E402

DEEP_K = 5143


def path_lengths(record) -> dict:
    """경로마다 받은 목록 길이(= 그 경로에서 기록된 최대 순위)."""
    out: dict = {}
    for song in record.songs.values():
        for p in song.paths:
            out[p.path] = max(out.get(p.path, 0), p.rank)
    return out


def path_ranks(record, song_id: str) -> dict:
    song = record.songs.get(song_id)
    out: dict = {}
    if song is not None:
        for p in song.paths:
            if p.path not in out or p.rank < out[p.path][0]:
                out[p.path] = (p.rank, p.dropped)
    return out


async def run(args) -> list:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries
    from src.retrieval.explain import ExplainRecorder
    from src.retrieval.trace_candidate_drop import rank_by, stage_of
    import src.backend.api.dependencies as deps

    router = deps.get_search_router()
    out = []
    for set_name in rt.SETS:
        rows = read_queries(rt.ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", "dev")
        cache = load_cache(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for row in rows:
            qid, query = row["query_id"].strip(), row["query"].strip()
            relevant = sorted(parse_relevant_ids(row["relevant_ids"]))
            analysis = cache.get(qid, query)
            runs = {}
            for name, path_k in (("now", None), ("deep", DEEP_K)):
                recorder = ExplainRecorder(query)
                pool: list = []
                started = time.perf_counter()
                await router.search(analysis, top_k=30, use_rerank=False, candidate_k=30, candidate_ids_out=pool,
                                    recorder=recorder, path_k=path_k)
                runs[name] = (recorder.record, pool, round((time.perf_counter() - started) * 1000, 1))
            now, now_pool, _ = runs["now"]
            deep, deep_pool, _ = runs["deep"]
            lengths_now = path_lengths(now)
            for sid in relevant:
                rn, rd = path_ranks(now, sid), path_ranks(deep, sid)
                song_deep = deep.songs.get(sid)
                song_now = now.songs.get(sid)
                out.append({
                    "set": set_name, "query_id": qid, "song_id": sid, "relevant_count": len(relevant),
                    "pool_rank": now_pool.index(sid) + 1 if sid in now_pool else "",
                    "stage": stage_of(sid, now, now_pool),
                    "paths_now": "|".join(f"{p}:{r}{'*' if d else ''}" for p, (r, d) in sorted(rn.items())),
                    "paths_deep": "|".join(f"{p}:{r}" for p, (r, _) in sorted(rd.items())),
                    "path_lengths_now": "|".join(f"{p}:{n}" for p, n in sorted(lengths_now.items())),
                    "weights_now": json.dumps(now.modality_weights, sort_keys=True),
                    "fused_rank_now": rank_by(now, sid, boosted=False) or "",
                    "boosted_rank_now": rank_by(now, sid, boosted=True) or "",
                    "fused_rank_deep": rank_by(deep, sid, boosted=False) or "",
                    "boosted_rank_deep": rank_by(deep, sid, boosted=True) or "",
                    "pool_rank_deep": deep_pool.index(sid) + 1 if sid in deep_pool else "",
                    "adjust_now": "" if song_now is None else "|".join(f"{a.rule}:{a.delta:+.4f}" for a in song_now.adjustments),
                    "adjust_deep": "" if song_deep is None else "|".join(f"{a.rule}:{a.delta:+.4f}" for a in song_deep.adjustments),
                    "pool30_boosted_cut_score": "",
                })
            print(f"[trace {rt.AUDIO}] {set_name} {qid}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-server-state", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"이미 있다: {args.out}", file=sys.stderr)
        return 2
    import os

    os.chdir(rt.ROOT)
    if server_state.diff(server_state.capture(), json.loads(args.expect_server_state.read_text(encoding="utf-8"))):
        print("서버 상태가 기준과 다르다", file=sys.stderr)
        return 4
    rt.install_guards()
    rt.code_identity()
    rt.verify_guards()
    info = rt.install_audio_config()
    rt.install_lyrics_snapshot()
    rows = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"audio": info, "rows": len(rows), "queries": len({r["query_id"] for r in rows}), "deep_k": DEEP_K,
            "network_attempts": list(rt.NET_ATTEMPTS)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if rt.NET_ATTEMPTS:
        print("외부 연결 시도가 있었다", file=sys.stderr)
        return 1
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os_exit = __import__("os")._exit
    os_exit(code)
