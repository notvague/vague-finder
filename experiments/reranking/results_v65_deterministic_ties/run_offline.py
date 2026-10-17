"""v65 평가기 실행 하나 = 코드 판(before|after) × Qdrant 모드(local|server) × 반복 — dev 176을 한 프로세스에서 순서대로.

기존 평가기(`src.retrieval.evaluate_search_accuracy.evaluate`, 리랭킹 포함 경로)를 그대로 부른다. 모델 응답은 결정적 가짜
(입력 순서 그대로)라 **품질 측정이 아니다** — 후보·리랭커 입력·프롬프트가 같은지 보는 실행이다.

    V65_CODE=after V65_MODE=server venv/bin/python run_offline.py --rep 1 --out-root <폴더>

산출: <out-root>/<code>_<mode>_rep<n>/<set>_dev/(평가기 산출물 + v65_queries.jsonl · v65_qdrant_calls.jsonl) ·
      run_conditions.json · DONE(모든 세트가 끝나고 건수 확인을 통과했을 때만)
종료 코드: 0 성공 · 1 실패 · 2 덮어쓰기 거부
"""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v65_eval_runtime as rt  # noqa: E402  (코드 판·환경을 src import 전에 정한다)

ROOT = rt.ROOT
os.chdir(ROOT)


def read_rows(set_name: str):
    with (ROOT / f"experiments/reranking/eval_queries_{set_name}.csv").open(encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["split"].strip() == "dev"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sets", default="v06,v09,v10")
    parser.add_argument("--query-ids", default="")
    parser.add_argument("--overwrite", action="store_true")
    # live 전용(v65에서는 실행하지 않았다 — LIVE_REMEASURE_PLAN.md). 사용자 승인 표시 V65_LIVE_APPROVED=1 없이는 시작하지 않는다
    parser.add_argument("--ledger", default="")
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument("--cost-cap", type=float, default=0.0)
    parser.add_argument("--attempt-cap", type=int, default=0)
    args = parser.parse_args()
    if rt.MODEL == "live" and (os.environ.get("V65_LIVE_APPROVED") != "1" or not args.ledger or not args.deadline_epoch
                               or args.cost_cap <= 0 or args.attempt_cap <= 0):
        print("live 실행은 V65_LIVE_APPROVED=1 · --ledger · --deadline-epoch · --cost-cap · --attempt-cap이 모두 필요하다", file=sys.stderr)
        return 2
    sets = [s for s in args.sets.split(",") if s]
    run_name = f"{'live_' if rt.MODEL == 'live' else ''}{rt.CODE}_{rt.MODE}_rep{args.rep}"
    run_dir = Path(args.out_root) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(f"이미 산출물이 있다: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "DONE").unlink(missing_ok=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "src", "tests"], capture_output=True, text=True, cwd=ROOT).stdout
    conditions: dict = {"run": run_name, "code": rt.CODE, "mode": rt.MODE, "rep": args.rep, "sets": sets,
                        "query_ids": args.query_ids, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "env": {k: os.environ.get(k, "") for k in sorted(rt.FIXED_ENV) + ["QDRANT_PATH", "QDRANT_URL"]},
                        "git_head": head, "worktree_src_tests_changes": dirty.splitlines(), "status": "running"}

    def write() -> None:
        (run_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")

    guard = None
    try:
        rt.install_guards()
        conditions["model"] = rt.MODEL
        if rt.MODEL == "live":
            import v65_budget

            guard = v65_budget.install(Path(args.ledger), cost_cap_usd=args.cost_cap, attempt_cap=args.attempt_cap,
                                       deadline_epoch=args.deadline_epoch, run_tag=run_name,
                                       model=os.environ["GEMINI_MODEL_NAME"])
            conditions["budget_before"] = guard.totals()
            conditions["budget_limits"] = {"cost_cap_usd": args.cost_cap, "attempt_cap": args.attempt_cap,
                                           "deadline_epoch": args.deadline_epoch}
            if guard.halted:
                raise RuntimeError(f"예산 장부가 정지 상태: {guard.halt_reason}")
        conditions["guard_before"] = rt.verify_guards()
        conditions["code_identity"] = rt.code_identity()
        import src.backend.api.dependencies as deps
        from src.retrieval import evaluate_search_accuracy as esa
        from src.vector_db import qdrant_backend as qb
        from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME
        from qdrant_reference import ReferenceStore

        if rt.MODEL == "fake":
            esa.preflight_gemini = lambda *a, **k: {"offline": "skipped — 모델 호출 없음"}  # type: ignore[assignment]
        lyrics = rt.install_lyrics_snapshot()
        conditions["lyrics_snapshot"] = lyrics.fingerprint()
        t0 = time.perf_counter()
        client = deps.get_vector_client().client
        conditions["qdrant_open_seconds"] = round(time.perf_counter() - t0, 3)
        collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
        t0 = time.perf_counter()
        store = ReferenceStore(client, collection)
        fp = store.hybrid.fingerprint()
        conditions["reference_build_seconds"] = round(time.perf_counter() - t0, 3)
        conditions["index_fingerprint"] = fp
        if fp["content_sha256_16"] != rt.EXPECTED_INDEX_FINGERPRINT:
            raise RuntimeError(f"색인 지문 {fp['content_sha256_16']} ≠ {rt.EXPECTED_INDEX_FINGERPRINT}")
        conditions["collections"] = {n: len(r.point_ids) for n, r in store.by_name.items()}
        recorder = rt.Recorder(store)
        recorder.install()
        conditions["sets_done"] = {}
        conditions["cold_before_first_query_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        for set_name in sets:
            rows = read_rows(set_name)
            recorder.query_ids = {r["query"].strip(): r["query_id"].strip() for r in rows}
            recorder.rows, recorder.qdrant_calls = {}, []
            out_dir = run_dir / f"{set_name}_dev"
            eval_args = esa.build_parser().parse_args([
                "--input", f"experiments/reranking/eval_queries_{set_name}.csv", "--split", "dev",
                "--analysis-cache", str(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json"),
                "--output-dir", str(out_dir), *(["--query-ids", args.query_ids] if args.query_ids else []),
            ])
            started = time.perf_counter()
            asyncio.run(esa.evaluate(eval_args))
            seconds = round(time.perf_counter() - started, 3)
            summary = recorder.save(out_dir)
            with (out_dir / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
                detail = list(csv.DictReader(fh))
            expected = [r["query_id"].strip() for r in rows]
            if args.query_ids:
                wanted = set(args.query_ids.split(","))
                expected = [q for q in expected if q in wanted]
            if [r["query_id"] for r in detail] != expected:
                raise RuntimeError(f"{set_name}: detail 질의 수가 기대와 다르다")
            if summary["queries_recorded"] != len(expected) or summary["unattributed_rows"]:
                raise RuntimeError(f"{set_name}: 기록 질의 수 불일치 또는 귀속 없는 기록")
            if summary["lyrics_errors"]:
                raise RuntimeError(f"{set_name}: 가사 검색 예외")
            runinfo = json.loads((out_dir / "search_eval_dev_runinfo.json").read_text(encoding="utf-8"))
            if runinfo["lyrics_source"].get("content_sha256_16") != rt.EXPECTED_LYRICS:
                raise RuntimeError(f"{set_name}: runinfo 가사 지문 불일치")
            if runinfo["reranker"]["backend"] != "gemini_listwise" or not runinfo["reranker"]["enabled"]:
                raise RuntimeError(f"{set_name}: 리랭커 설정이 다르다: {runinfo['reranker']}")
            if runinfo["analysis"].get("fallback_query_ids") or runinfo["analysis"].get("fingerprint_drift"):
                raise RuntimeError(f"{set_name}: 분석 캐시 폴백·drift")
            conditions["sets_done"][set_name] = {"seconds": seconds, "rows": len(detail), **summary}
            write()
            print(f"[v65] {run_name} {set_name} {len(detail)}건 {seconds}초 {json.dumps(summary, ensure_ascii=False)[:400]}",
                  flush=True)
        conditions["guard_after"] = rt.verify_guards()
        if guard is not None:
            conditions["budget_local"] = dict(guard.local)
            conditions["budget_after"] = guard.totals()
            if guard.halted:
                raise RuntimeError(f"실행 중 정지: {guard.halt_reason}")
        conditions["network_attempts"] = list(rt.NET_ATTEMPTS)
        conditions["lyrics_source_reads"] = lyrics.reads
        conditions["status"] = "complete"
        conditions["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        conditions["total_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        conditions["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write()
        partial = bool(args.query_ids) or sets != list(rt.SETS)
        (run_dir / "DONE").write_text(json.dumps({"partial": partial}) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001
        conditions["status"] = "failed"
        conditions["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        conditions["traceback_tail"] = traceback.format_exc()[-3000:]
        write()
        print(f"[v65] {run_name} 실패: {conditions['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
