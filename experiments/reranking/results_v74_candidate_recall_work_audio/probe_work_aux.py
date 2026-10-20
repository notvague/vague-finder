"""작품명 보조 목록이 후보 30까지 어디서 떨어지는가 — 해석 '채택' 질의 전부(정답과 무관하게 고른 집합). 리랭킹 없음, 모델 호출 0회.

**진단 전용이며 검색 입력을 바꾸지 않는다.** 대상 질의는 기준 해석 상태 파일(`inputs/expected_work_status.json`)에서 상태가 `applied`인
질의 전부다(정답을 보고 고르지 않는다). 라우터를 `use_rerank=False`로 한 번 부르고 explain 기록에서 `work_name_aux` 경로에 든 곡마다
보조 순위 · 다른 경로 유무 · 융합 순위 · 보정 뒤 순위 · 후보 30 안 여부를 적는다. 정답 여부(`is_relevant`)는 기록을 다 모은 뒤에 붙인다.

    V74_CONFIG=work V74_AUDIO=on venv/bin/python probe_work_aux.py --out trace/work_aux_on.csv --expect-server-state server_state_baseline.json
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
import v74_eval_runtime as rt  # noqa: E402
import server_state  # noqa: E402


async def run(targets: dict) -> list:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import read_queries
    from src.retrieval.explain import ExplainRecorder
    from src.retrieval.trace_candidate_drop import rank_by
    import src.backend.api.dependencies as deps

    router = deps.get_search_router()
    wcache = load_cache(rt.CACHES / "analysis_cache_v06_dev_vertex.json")
    wrow = next(r for r in read_queries(rt.ROOT / "experiments/reranking/eval_queries_v06.csv", "dev")
                if r["query_id"].strip() == "q105")
    await router.search(wcache.get("q105", wrow["query"].strip()), top_k=30, use_rerank=False, candidate_k=30)  # 예열
    out = []
    for set_name in rt.SETS:
        rows = read_queries(rt.ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", "dev")
        cache = load_cache(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for row in rows:
            qid = row["query_id"].strip()
            if qid not in targets:
                continue
            recorder = ExplainRecorder(row["query"].strip())
            pool: list = []
            work_out: list = []
            await router.search(cache.get(qid, row["query"].strip()), top_k=30, use_rerank=False, candidate_k=30,
                                candidate_ids_out=pool, recorder=recorder, work_resolution_out=work_out)
            status = work_out[0].status if work_out else None
            if status != "applied":
                raise RuntimeError(f"{qid}: 상태 {status!r} ≠ applied")
            rec = recorder.record
            entered = set(work_out[0].entered_ids or [])
            displaced = len(work_out[0].displaced_ids or [])
            # 30위 경계 점수(보정 뒤): 후보 30번째 곡의 보정 뒤 점수
            for song in rec.songs.values():
                aux = [p for p in song.paths if p.path == "work_name_aux"]
                if not aux:
                    continue
                others = sorted({p.path for p in song.paths if p.path != "work_name_aux"})
                out.append({
                    "set": set_name, "query_id": qid, "song_id": song.song_id, "aux_rank": aux[0].rank,
                    "other_paths": "|".join(others), "aux_only": int(not others),
                    "fused_rank": rank_by(rec, song.song_id, boosted=False) or "",
                    "boosted_rank": rank_by(rec, song.song_id, boosted=True) or "",
                    "in_pool": int(song.song_id in pool), "pool_rank": pool.index(song.song_id) + 1 if song.song_id in pool else "",
                    "entered_by_aux": int(song.song_id in entered), "query_displaced_n": displaced,
                })
            print(f"[probe {rt.CODE}] {qid}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-server-state", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"이미 있다: {args.out}", file=sys.stderr)
        return 2
    if rt.CONFIG != "work":
        print("V74_CONFIG=work만", file=sys.stderr)
        return 2
    status = json.loads((HERE / "inputs/expected_work_status.json").read_text(encoding="utf-8"))["status_by_config"]["work"]
    targets = {q: s for q, s in status.items() if s == "applied"}
    os.chdir(rt.ROOT)
    if server_state.diff(server_state.capture(), json.loads(args.expect_server_state.read_text(encoding="utf-8"))):
        print("서버 상태가 기준과 다르다", file=sys.stderr)
        return 4
    rt.install_guards()
    rt.code_identity()
    rt.verify_guards()
    rt.install_lyrics_snapshot()
    rows = asyncio.run(run(targets))
    # 정답 표시는 기록을 다 모은 뒤에만
    rel = {}
    for s in rt.SETS:
        with open(rt.ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    rel[r["query_id"].strip()] = {x.strip() for x in r["relevant_ids"].split("|") if x.strip()}
    for r in rows:
        r["is_relevant"] = int(r["song_id"] in rel[r["query_id"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"code": rt.CODE, "queries": len(targets), "rows": len(rows), "network_attempts": list(rt.NET_ATTEMPTS),
            "work_resolver_calls": list(rt.WORK_RESOLVER_CALLS)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 1 if rt.NET_ATTEMPTS else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
