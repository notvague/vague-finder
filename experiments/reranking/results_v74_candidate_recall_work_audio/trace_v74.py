"""진단용 단계 추적 — 정답 곡마다 경로별 순위(현행 깊이 / 전체 깊이 5,143)·단계·융합/가산 뒤 순위. 리랭킹 없음, 모델 호출 0회.

**실제 후보 생성과 분리된 별도 실행이다.** 라우터를 `use_rerank=False`로 두 번(현행 깊이 · `path_k=5143`) 부르고 explain 기록만 읽는다.
정답 ID는 기록을 읽을 때만 쓰며, 검색에 넘기거나 가산하지 않는다. 단계 정의는 `src.retrieval.trace_candidate_drop`의 것을 그대로 쓴다.
`src/retrieval/trace_candidate_drop.py`와 다른 점: 정답이 여럿이면 **모든 정답 곡**을 적는다(v50 진단표는 정렬 첫 곡 하나), 모든 경로(보조 포함)의
현행/전체 깊이 순위와 경로별 받은 목록 길이를 남긴다, v74 하네스(서버·차단·가사 스냅샷·작품명×소리 구성)를 쓴다.

v67 `trace_v67.py`(PR #82 병합) 사본. v74에서 바꾼 것:
- 하네스 `v74_eval_runtime`(작품명 off|work × 소리 on|skip). 소리 구성은 환경(`AUDIO_FUSION_SCALE`)으로 정해진다.
- **예열**: 평가 실행과 같은 고정 질의(q105) 한 번을 기록 전에 검색한다(작품명 작업 대기 cold 시간 초과 방지 — v72).
- **작품명 해석 상태 대조**: 두 검색(현행·전체 깊이) 모두 질의별 해석 상태를 받아 기준 파일(`--expect-work-status`)과 대조하고,
  다르거나 실패 상태면 멈춘다(캐시 누락·경로 오류를 정상 빈 결과로 숨기지 않는다).
- **평가기 후보 대조**: `--expect-candidates-run <실행 폴더>`를 주면 현행 깊이 후보 30(순서 포함)이 그 실행의 기록(`v74_queries.jsonl`)과
  176/176 같아야 끝난다. 다르면 0이 아닌 코드로 끝낸다.
- 경로 실패(`_path_result`가 삼킨 예외)를 질의별로 센다. 0이 아니면 멈춘다.

    V74_CONFIG=work V74_AUDIO=on venv/bin/python trace_v74.py --out <폴더>/trace_work_on.csv \
        --expect-server-state server_state_baseline.json --expect-work-status inputs/expected_work_status.json \
        --expect-candidates-run runs/work_on_server_rep1
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v74_eval_runtime as rt  # noqa: E402
import server_state  # noqa: E402

DEEP_K = 5143
POOLS: dict = {}


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


PATH_FAILURES: list = []
WORK_STATUS_PROBLEMS: list = []


def install_path_failure_counter() -> None:
    import src.retrieval.search_router as sr

    orig = sr._path_result

    def path_result(name, result, recorder=sr.NULL_RECORDER):  # type: ignore[no-untyped-def]
        if isinstance(result, BaseException):
            PATH_FAILURES.append(f"{name}: {type(result).__name__}: {str(result)[:160]}")
        return orig(name, result, recorder)

    sr._path_result = path_result  # type: ignore[assignment]


def check_work_status(qid: str, kind: str, out: list, expected: dict) -> str:
    status = out[0].status if out else None
    want = expected.get(qid)
    if rt.CONFIG == "off":
        if status is not None:
            WORK_STATUS_PROBLEMS.append(f"{qid}/{kind}: off인데 {status!r}")
    elif status is None or str(status).startswith("failed") or status == "no_base_candidates":
        WORK_STATUS_PROBLEMS.append(f"{qid}/{kind}: 상태 {status!r}")
    elif want is not None and str(status) != want:
        WORK_STATUS_PROBLEMS.append(f"{qid}/{kind}: {status!r} ≠ 기준 {want!r}")
    return "(none)" if status is None else str(status)


async def run(args) -> list:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries
    from src.retrieval.explain import ExplainRecorder
    from src.retrieval.trace_candidate_drop import rank_by, stage_of
    import src.backend.api.dependencies as deps

    router = deps.get_search_router()
    expected = (json.loads(args.expect_work_status.read_text(encoding="utf-8"))["status_by_config"][rt.CONFIG]
                if args.expect_work_status else {})
    # 예열(평가 실행과 같은 고정 질의 · 기록 밖)
    wcache = load_cache(rt.CACHES / "analysis_cache_v06_dev_vertex.json")
    wrow = next(r for r in read_queries(rt.ROOT / "experiments/reranking/eval_queries_v06.csv", "dev")
                if r["query_id"].strip() == "q105")
    await router.search(wcache.get("q105", wrow["query"].strip()), top_k=30, use_rerank=False, candidate_k=30)
    out = []
    for set_name in rt.SETS:
        rows = read_queries(rt.ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", "dev")
        cache = load_cache(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for row in rows:
            qid, query = row["query_id"].strip(), row["query"].strip()
            relevant = sorted(parse_relevant_ids(row["relevant_ids"]))
            analysis = cache.get(qid, query)
            runs = {}
            statuses = {}
            for name, path_k in (("now", None), ("deep", DEEP_K)):
                recorder = ExplainRecorder(query)
                pool: list = []
                work_out: list = []
                started = time.perf_counter()
                await router.search(analysis, top_k=30, use_rerank=False, candidate_k=30, candidate_ids_out=pool,
                                    recorder=recorder, path_k=path_k, work_resolution_out=work_out)
                runs[name] = (recorder.record, pool, round((time.perf_counter() - started) * 1000, 1))
                statuses[name] = check_work_status(qid, name, work_out, expected)
            POOLS[qid] = list(runs["now"][1])
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
                    "work_status": statuses["now"],
                    "work_status_deep": statuses["deep"],
                    "search_ms_now": runs["now"][2],
                    "search_ms_deep": runs["deep"][2],
                })
            print(f"[trace {rt.CODE}] {set_name} {qid}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-server-state", type=Path, required=True)
    ap.add_argument("--expect-work-status", type=Path, default=None)
    ap.add_argument("--expect-candidates-run", type=Path, default=None)
    args = ap.parse_args()
    if args.out.exists():
        print(f"이미 있다: {args.out}", file=sys.stderr)
        return 2
    os.chdir(rt.ROOT)
    if server_state.diff(server_state.capture(), json.loads(args.expect_server_state.read_text(encoding="utf-8"))):
        print("서버 상태가 기준과 다르다", file=sys.stderr)
        return 4
    rt.install_guards()
    rt.code_identity()
    rt.verify_guards()
    info = {"code": rt.CODE, "AUDIO_FUSION_SCALE": os.environ.get("AUDIO_FUSION_SCALE"),
            "WORK_NAME_RESOLUTION_ENABLED": os.environ.get("WORK_NAME_RESOLUTION_ENABLED"),
            "WORK_NAME_RESOLVER": os.environ.get("WORK_NAME_RESOLVER", "")}
    rt.install_lyrics_snapshot()
    install_path_failure_counter()
    rows = asyncio.run(run(args))
    pool_check = {}
    if args.expect_candidates_run:
        ref = {}
        for set_name in rt.SETS:
            for line in (args.expect_candidates_run / f"{set_name}_dev/v74_queries.jsonl").read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                ref[rec["query_id"]] = rec
        diff = [q for q in POOLS if q not in ref or [str(c["id"]) for c in ref[q]["candidates"]] != POOLS[q]]
        pool_check = {"run": str(args.expect_candidates_run), "queries": len(POOLS), "different": diff}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"config": info, "rows": len(rows), "queries": len({r["query_id"] for r in rows}), "deep_k": DEEP_K,
            "network_attempts": list(rt.NET_ATTEMPTS), "path_failures": list(PATH_FAILURES),
            "work_status_problems": list(WORK_STATUS_PROBLEMS), "work_resolver_calls": list(rt.WORK_RESOLVER_CALLS),
            "pool_check_vs_eval_run": pool_check}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    bad = []
    if rt.NET_ATTEMPTS:
        bad.append("외부 연결 시도")
    if PATH_FAILURES:
        bad.append(f"경로 실패 {len(PATH_FAILURES)}")
    if WORK_STATUS_PROBLEMS:
        bad.append(f"작품명 상태 {WORK_STATUS_PROBLEMS[:3]}")
    if pool_check and (pool_check["different"] or pool_check["queries"] != 176):
        bad.append(f"평가기 후보와 다름 {pool_check['different'][:5]}")
    if bad:
        print("추적 무효: " + " · ".join(bad), file=sys.stderr)
        return 1
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os_exit = __import__("os")._exit
    os_exit(code)
