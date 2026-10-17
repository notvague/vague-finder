"""v63 실행 하나 = (arm A|B) × (mode offline|live) × 반복 번호 — dev 176(v06 57 · v09 59 · v10 60)을 한 프로세스에서 순서대로.

기존 평가기(`src.retrieval.evaluate_search_accuracy.evaluate`, 리랭킹 포함 경로)를 그대로 부른다. 바뀌는 것은 `v63_runtime`의
자료원·arm·기록과(offline이면) 가짜 모델 응답, (live면) `v63_budget`뿐이다.

    venv/bin/python run_v63.py --arm A --mode offline --rep 1 --out-root <폴더>
    (live) venv/bin/python with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- venv/bin/python run_v63.py --arm A --mode live --rep 1 --out-root <폴더>

산출: <out-root>/<mode>_<arm>_rep<n>/<set>_dev/(평가기 산출물 + v63_queries.jsonl · v63_hybrid_calls.jsonl) ·
      <out-root>/<mode>_<arm>_rep<n>/run_conditions.json · DONE(모든 세트가 끝나고 건수 확인을 통과했을 때만, 마지막에 쓴다)
종료 코드: 0 성공 · 1 실패/중단(DONE 없음) · 2 덮어쓰기 거부. `--overwrite`는 실행 스크립트가 OVERWRITE=1일 때만 넘긴다.
`V63_FAIL_INJECT=<set>`이면 그 세트 시작 때 일부러 예외를 낸다(실패 전달 검증용).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
PROCESS_STARTED_WALL = time.time()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import v63_runtime as rt  # noqa: E402


def read_rows(set_name: str):
    path = ROOT / f"experiments/reranking/eval_queries_{set_name}.csv"
    with path.open(encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["split"].strip() == "dev"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--mode", choices=["offline", "live"], required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sets", default="v06,v09,v10")
    parser.add_argument("--query-ids", default="", help="스모크 전용(완료 표식에 부분 실행으로 적힌다)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cost-cap", type=float, default=10.0)
    parser.add_argument("--attempt-cap", type=int, default=1200)
    parser.add_argument("--deadline-epoch", type=float, default=0.0, help="live 신규 호출 마감(epoch 초)")
    parser.add_argument("--ledger", default="", help="live 예산 장부(실행 사이 공유)")
    args = parser.parse_args()

    sets = [s for s in args.sets.split(",") if s]
    run_name = f"{args.mode}_{args.arm}_rep{args.rep}"
    run_dir = Path(args.out_root) / run_name
    existing = [p.name for p in run_dir.glob("*")] if run_dir.exists() else []
    if existing and not args.overwrite:
        print(f"출력 폴더에 이미 산출물이 있다: {run_dir} — 다른 --out-root 또는 OVERWRITE=1", file=sys.stderr)
        return 2
    if args.overwrite and run_dir.exists():
        (run_dir / "DONE").unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    env_shown = rt.fixed_env(args.mode)
    rt.install_guards(args.mode)
    conditions: dict = {
        "run": run_name, "arm": args.arm, "mode": args.mode, "rep": args.rep, "sets": sets,
        "query_ids": args.query_ids, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_shown, "git": rt.git_state(), "status": "running",
    }

    def write_conditions() -> None:
        (run_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")

    guard = None
    try:
        if args.mode == "live":
            if not args.ledger or not args.deadline_epoch:
                raise SystemExit("live 실행은 --ledger와 --deadline-epoch가 필요하다")
            import v63_budget

            guard = v63_budget.install(Path(args.ledger), cost_cap_usd=args.cost_cap, attempt_cap=args.attempt_cap,
                                       deadline_epoch=args.deadline_epoch, run_tag=run_name,
                                       model=os.environ["GEMINI_MODEL_NAME"])
            conditions["budget_before"] = guard.totals()
            conditions["budget_limits"] = {"cost_cap_usd": args.cost_cap, "attempt_cap": args.attempt_cap,
                                           "deadline": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                                     time.localtime(args.deadline_epoch))}
            if guard.halted:
                raise rt.RunAborted(f"예산 장부가 정지 상태: {guard.halt_reason}")
        conditions["guard_before"] = rt.verify_guards(args.mode)

        import src.backend.api.dependencies as deps  # noqa: E402
        from src.retrieval import evaluate_search_accuracy as esa  # noqa: E402
        from src.vector_db import qdrant_backend as qb  # noqa: E402
        from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

        if args.mode == "offline":
            esa.preflight_gemini = lambda *a, **k: {"offline": "skipped — 모델 호출 없음"}  # type: ignore[assignment]

        conditions["arm_impl"] = rt.install_arm(args.arm, run_dir / "_legacy")
        lyrics_source = rt.install_lyrics_snapshot()
        conditions["lyrics_snapshot"] = lyrics_source.fingerprint()

        t0 = time.perf_counter()
        client = deps.get_vector_client().client
        collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
        conditions["qdrant_open_seconds"] = round(time.perf_counter() - t0, 3)
        conditions["point_count"] = int(client.count(collection, exact=True).count)
        reference, fp = rt.index_fingerprint(client, collection)
        conditions["index_fingerprint"] = fp

        recorder = rt.Recorder(args.mode, reference)
        recorder.install()

        conditions["sets_done"] = {}
        conditions["cold_before_first_query_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        for set_name in sets:
            if os.getenv("V63_FAIL_INJECT") == set_name:
                raise RuntimeError(f"V63_FAIL_INJECT={set_name} — 일부러 낸 실패")
            rows = read_rows(set_name)
            recorder.query_ids = {r["query"].strip(): r["query_id"].strip() for r in rows}
            recorder.rows = {}
            recorder.hybrid_calls = []
            recorder.seq = 0
            out_dir = run_dir / f"{set_name}_dev"
            eval_args = esa.build_parser().parse_args([
                "--input", f"experiments/reranking/eval_queries_{set_name}.csv",
                "--split", "dev",
                "--analysis-cache", str(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json"),
                "--output-dir", str(out_dir),
                *(["--query-ids", args.query_ids] if args.query_ids else []),
            ])
            set_started = time.perf_counter()
            asyncio.run(esa.evaluate(eval_args))
            seconds = round(time.perf_counter() - set_started, 3)
            summary = recorder.save(out_dir)
            # 건수·질의 집합 확인 — 부분 실행이 완료로 보이지 않게
            with (out_dir / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
                detail = list(csv.DictReader(fh))
            expected_ids = [r["query_id"].strip() for r in rows]
            if args.query_ids:
                wanted = [q for q in args.query_ids.split(",") if q]
                expected_ids = [q for q in expected_ids if q in wanted]
            got_ids = [r["query_id"] for r in detail]
            if got_ids != expected_ids:
                raise RuntimeError(f"{set_name}: detail 질의 {len(got_ids)}건 ≠ 기대 {len(expected_ids)}건")
            if summary["queries_recorded"] != len(expected_ids) or summary["unattributed_rows"]:
                raise RuntimeError(f"{set_name}: 기록 질의 {summary['queries_recorded']} ≠ {len(expected_ids)} 또는 귀속 없는 기록")
            if summary["lyrics_errors"]:
                raise RuntimeError(f"{set_name}: 가사 검색 예외 {summary['lyrics_errors']}건 — 자료원 실패")
            runinfo = json.loads((out_dir / "search_eval_dev_runinfo.json").read_text(encoding="utf-8"))
            if runinfo["lyrics_source"].get("content_sha256_16") != conditions["lyrics_snapshot"]["content_sha256_16"]:
                raise RuntimeError(f"{set_name}: runinfo 가사 지문이 스냅샷과 다르다")
            if runinfo["reranker"]["backend"] != "gemini_listwise" or not runinfo["reranker"]["enabled"]:
                raise RuntimeError(f"{set_name}: 리랭커가 gemini_listwise·켜짐이 아니다: {runinfo['reranker']}")
            if runinfo["analysis"].get("fallback_query_ids") or runinfo["analysis"].get("fingerprint_drift"):
                raise RuntimeError(f"{set_name}: 분석 캐시 폴백·drift: {runinfo['analysis'].get('fingerprint_drift')}")
            conditions["sets_done"][set_name] = {"seconds": seconds, "rows": len(detail), **summary}
            write_conditions()
            print(f"[v63] {run_name} {set_name} 완료 {len(detail)}건 {seconds}초 | {json.dumps(summary, ensure_ascii=False)[:300]}",
                  flush=True)

        conditions["lyrics_source_reads"] = lyrics_source.reads
        lyr = deps.get_lyrics_exact_search_service.peek()  # type: ignore[attr-defined]
        conditions["lyrics_service_status"] = lyr.status() if lyr is not None else None
        conditions["guard_after"] = rt.verify_guards(args.mode)
        if guard is not None:
            conditions["budget_local"] = dict(guard.local)
            conditions["budget_after"] = guard.totals()
            if guard.halted:
                raise rt.RunAborted(f"실행 중 정지: {guard.halt_reason}")
        conditions["status"] = "complete"
        conditions["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        conditions["total_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        conditions["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_conditions()
        partial = bool(args.query_ids) or sets != list(rt.SETS)
        (run_dir / "DONE").write_text(json.dumps({"partial": partial, "finished_at": conditions["finished_at"]}) + "\n",
                                      encoding="utf-8")
        try:
            deps.close_lyrics_exact_search_service(timeout=5)
            deps.close_vector_client()
        except Exception:  # noqa: BLE001
            pass
        return 0
    except BaseException as exc:  # noqa: BLE001 - 어떤 실패든 완료 표식 없이 기록하고 1
        conditions["status"] = "failed"
        conditions["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        conditions["traceback_tail"] = traceback.format_exc()[-2000:]
        if guard is not None:
            conditions["budget_local"] = dict(guard.local)
            conditions["budget_after"] = guard.totals()
        write_conditions()
        print(f"[v63] {run_name} 실패: {conditions['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # 남은 백그라운드 스레드(가사 갱신 등)를 기다리지 않는다 — 결과는 이미 기록했다
