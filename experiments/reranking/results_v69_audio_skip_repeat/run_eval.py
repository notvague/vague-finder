"""v69 평가기 실행 하나 = 소리 구성(on|skip) × 서버 Qdrant × 반복 — dev 176(v06 57 + v09 59 + v10 60)을 한 프로세스에서.

v65 `run_offline.py`를 옮긴 것이다. 기존 평가기(`evaluate_search_accuracy.evaluate`, 리랭킹 포함 경로)를 그대로 부른다.

- `V69_MODEL=fake`(기본): 결정적 가짜 응답 — **품질 측정이 아니다.** 후보·보호·리랭커 입력·프롬프트 지문을 남긴다.
- `V69_MODEL=live`: 실제 Vertex(`gemini-3.1-flash-lite`). `V69_LIVE_APPROVED=1` · `--ledger` · `--cost-cap` · `--attempt-cap` ·
  `--deadline-window-seconds`가 모두 있어야 시작한다. **부분 실행(--query-ids·--sets)은 live에서 거부한다.**

호출 전에 확인하는 것(하나라도 어긋나면 모델·네트워크 호출 없이 멈춘다):
  1. 서버 상태 = `--expect-server-state`(컨테이너 ID·시작 시각·재시작 수·컬렉션 세그먼트·색인 상태·별칭)
  2. 코드 판 사본 = 기준 커밋의 src 트리(blob 단위, `code_identity`)
  3. 세 세트 분석 캐시 — 누락·규칙 폴백·항목 오류·조건 drift(평가기는 drift를 경고만 하므로 여기서 멈춘다)
  4. 색인 내용 지문 `238299e6c8c17f2a` · 가사 스냅샷 지문 `e0d58f28f1f4a563` · 차단 확인
실행 뒤에도 서버 상태를 다시 떠서 기준과 같아야 완료(`DONE`)로 표시한다.

    V69_AUDIO=skip venv/bin/python run_eval.py --rep 1 --out-root <폴더> --expect-server-state <기준 파일>

산출: <out-root>/<run>/<set>_dev/(평가기 산출물 + v69_queries.jsonl · v69_qdrant_calls.jsonl) · run_conditions.json ·
      server_state_before.json · server_state_after.json · DONE(모든 세트가 끝나고 건수·조건 확인을 통과했을 때만)
종료 코드: 0 성공 · 1 실패 · 2 거부(덮어쓰기·게이트·부분 live) · 4 서버 상태 불일치(호출 전)
자체 검사용: `V69_FAIL_INJECT=<set>`이면 그 세트 직전에 실패를 일으킨다(모델 호출 없음, fake 전용).
v69 자체 검사용: `V69_MODEL=fault V69_FAULT_PLAN=504|403` — live와 같은 예산 통제·리랭커 경로를 가짜 전송으로(네트워크 없음).
  승인 표시 V69_LIVE_APPROVED가 있거나 장부가 이 결과 폴더 안이면 거부한다(실제 실행 묶음이 쓸 수 없게). 지표 집계는 live만 받는다.
"""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v69_eval_runtime as rt  # noqa: E402  (코드 판·환경을 src import 전에 정한다)
import server_state  # noqa: E402

ROOT = rt.ROOT
os.chdir(ROOT)
EXPECTED_ROWS = {"v06": 57, "v09": 59, "v10": 60}


def read_rows(set_name: str):
    with (ROOT / f"experiments/reranking/eval_queries_{set_name}.csv").open(encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["split"].strip() == "dev"]


def precheck_caches(esa, sets, query_ids: str) -> dict:
    """세 세트의 분석 캐시를 **모델 호출 전에** 모두 검사한다. 평가기는 drift를 경고만 하고 세트마다 preflight를 먼저 부르므로."""
    out = {}
    for set_name in sets:
        rows = read_rows(set_name)
        if query_ids:
            wanted = set(query_ids.split(","))
            rows = [r for r in rows if r["query_id"].strip() in wanted]
        path = rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json"
        cache = esa.load_cache(path)
        needed = [r["query_id"].strip() for r in rows]
        missing = cache.missing(needed)
        fallbacks = cache.fallback_ids(needed)
        problems = []
        for r in rows:
            try:
                cache.get(r["query_id"].strip(), r["query"].strip())
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{r['query_id']}: {type(exc).__name__}")
        drift = cache.fingerprint_drift()
        if missing or fallbacks or problems or drift:
            raise RuntimeError(f"{set_name} 분석 캐시 문제: missing={list(missing)[:5]} fallback={list(fallbacks)[:5]} "
                               f"problems={problems[:5]} drift={drift}")
        out[set_name] = {"path": str(path.relative_to(ROOT)), "queries": len(needed),
                         "file_sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                         "missing": 0, "fallback": 0, "drift": []}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-name", default="", help="기본 <code>_server_rep<n>(live면 live_ 접두)")
    parser.add_argument("--sets", default="v06,v09,v10")
    parser.add_argument("--query-ids", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expect-server-state", required=True)
    parser.add_argument("--ledger", default="")
    parser.add_argument("--deadline-window-seconds", type=float, default=0.0)
    parser.add_argument("--cost-cap", type=float, default=0.0)
    parser.add_argument("--attempt-cap", type=int, default=0)
    args = parser.parse_args()
    sets = [s for s in args.sets.split(",") if s]
    partial = bool(args.query_ids) or sets != list(rt.SETS)
    if rt.MODEL == "fault":
        if os.environ.get("V69_LIVE_APPROVED"):
            print("fault는 승인 표시(V69_LIVE_APPROVED)와 함께 쓸 수 없다", file=sys.stderr)
            return 2
        if (not args.ledger or args.deadline_window_seconds <= 0 or args.cost_cap <= 0 or args.attempt_cap <= 0
                or Path(args.ledger).resolve().is_relative_to(HERE)):
            print("fault는 결과 폴더 밖 임시 장부와 한도 인자가 모두 필요하다", file=sys.stderr)
            return 2
        if os.environ.get("V69_FAULT_PLAN") not in rt.FAULT_PLANS:
            print(f"V69_FAULT_PLAN은 {rt.FAULT_PLANS} 중 하나", file=sys.stderr)
            return 2
    elif os.environ.get("V69_FAULT_PLAN"):
        print("V69_FAULT_PLAN은 fault에서만", file=sys.stderr)
        return 2
    if rt.MODEL == "live":
        if (os.environ.get("V69_LIVE_APPROVED") != "1" or not args.ledger or args.deadline_window_seconds <= 0
                or args.cost_cap <= 0 or args.attempt_cap <= 0):
            print("live 실행은 V69_LIVE_APPROVED=1 · --ledger · --deadline-window-seconds · --cost-cap · --attempt-cap이 모두 필요하다",
                  file=sys.stderr)
            return 2
        if partial:
            print("live 실행은 dev 176 전체만 허용한다(--query-ids·--sets 거부)", file=sys.stderr)
            return 2
        if os.environ.get("V69_FAIL_INJECT"):
            print("V69_FAIL_INJECT는 fake에서만", file=sys.stderr)
            return 2
        if not rt.LIVE_CREDENTIALS_PRESENT:
            print("live 실행에는 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS가 필요하다(with_env.py)", file=sys.stderr)
            return 2
    if os.environ.get("V69_PATH_FAIL_INJECT") and rt.MODEL != "fake":
        print("V69_PATH_FAIL_INJECT는 fake에서만", file=sys.stderr)
        return 2
    if os.environ.get("V69_FAIL_INJECT") and rt.MODEL != "fake":
        print("V69_FAIL_INJECT는 fake에서만", file=sys.stderr)
        return 2
    prefix = {"live": "live_", "fault": "fault_"}.get(rt.MODEL, "")
    run_name = args.run_name or f"{prefix}{rt.CODE}_server_rep{args.rep}"
    run_dir = Path(args.out_root) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(f"이미 산출물이 있다: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "DONE").unlink(missing_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "src", "tests"], capture_output=True, text=True, cwd=ROOT).stdout
    conditions: dict = {"run": run_name, "code": rt.CODE, "commit": rt.COMMITS[rt.CODE], "mode": rt.MODE,
                        "model": rt.MODEL, "rep": args.rep, "sets": sets, "query_ids": args.query_ids, "partial": partial,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "env": {k: os.environ.get(k, "") for k in sorted(rt.FIXED_ENV) + ["QDRANT_PATH", "QDRANT_URL"]},
                        "worktree_head": head, "worktree_src_tests_changes": dirty.splitlines(), "status": "running"}
    if rt.MODEL == "live":
        conditions["env"]["GCP_PROJECT_ID"] = "(운영 .env에서 넘김 — 값 기록 안 함)"

    def write() -> None:
        (run_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")

    # 1) 서버 상태 — 어떤 import·호출보다 먼저
    expected_state = json.loads(Path(args.expect_server_state).read_text(encoding="utf-8"))
    state_before = server_state.capture()
    (run_dir / "server_state_before.json").write_text(json.dumps(state_before, ensure_ascii=False, indent=1), encoding="utf-8")
    changed = server_state.diff(state_before, expected_state)
    conditions["server_state_before_matches_baseline"] = not changed
    if changed:
        conditions["status"] = "refused_server_state"
        conditions["error"] = f"서버 상태가 기준과 다르다: {changed}"
        write()
        print(conditions["error"], file=sys.stderr)
        return 4

    guard = None
    try:
        rt.install_guards()
        # live: 예산 통제를 가장 먼저 건다(사전 확인 단계에서 혹시 Vertex 시도가 생겨도 장부·한도 밖으로 나가지 않게).
        # 마감 기준점은 장부의 첫 HTTP 시도 때 정해지므로, 건다고 시간이 흐르기 시작하지는 않는다.
        if rt.MODEL in ("live", "fault"):
            if rt.MODEL == "fault":
                fault_seen = rt.install_fault_transport()  # 예산 통제가 이 가짜 전송을 감싼다
                conditions["fault_plan"] = os.environ["V69_FAULT_PLAN"]
            import v69_budget

            guard = v69_budget.install(Path(args.ledger), cost_cap_usd=args.cost_cap, attempt_cap=args.attempt_cap,
                                       deadline_window_seconds=args.deadline_window_seconds, run_tag=run_name,
                                       model=os.environ["GEMINI_MODEL_NAME"])
            conditions["budget_before"] = guard.totals()
            conditions["budget_limits"] = {"cost_cap_usd": args.cost_cap, "attempt_cap": args.attempt_cap,
                                           "deadline_window_seconds": args.deadline_window_seconds}
            if guard.halted:
                raise RuntimeError(f"예산 장부가 정지 상태: {guard.halt_reason}")
            if guard.deadline_passed():
                raise RuntimeError("신규 호출 마감이 지났다 — 실행을 시작하지 않는다")
        conditions["code_identity"] = rt.code_identity()
        from src.retrieval import evaluate_search_accuracy as esa

        conditions["analysis_caches"] = precheck_caches(esa, sets, args.query_ids)
        conditions["guard_before"] = rt.verify_guards()
        import src.backend.api.dependencies as deps
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
        if guard is not None:
            conditions["budget_before_first_set"] = guard.totals()
            if guard.local["attempts"]:
                raise RuntimeError(f"사전 확인 단계에서 Vertex 시도 {guard.local['attempts']}회가 생겼다")
        conditions["sets_done"] = {}
        conditions["cold_before_first_query_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        write()
        for set_name in sets:
            if os.environ.get("V69_FAIL_INJECT") == set_name:
                raise RuntimeError(f"V69_FAIL_INJECT={set_name}(자체 검사)")
            rows = read_rows(set_name)
            if not args.query_ids and len(rows) != EXPECTED_ROWS[set_name]:
                raise RuntimeError(f"{set_name}: dev {len(rows)}건 ≠ {EXPECTED_ROWS[set_name]}")
            recorder.query_ids = {r["query"].strip(): r["query_id"].strip() for r in rows}
            if len(recorder.query_ids) != len(rows):
                raise RuntimeError(f"{set_name}: 질의 문장이 겹쳐 귀속이 모호하다")
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
                raise RuntimeError(f"{set_name}: detail 질의 목록이 기대와 다르다")
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
            # 소리 구성 실행 검사(v67과 같은 기준): 경로 실패는 라우터가 빈 결과로 삼키므로 실제 조회 수로 확인한다
            if summary["path_failures"]:
                raise RuntimeError(f"{set_name}: 경로 실패 {summary['path_failures'][:3]}")
            if rt.AUDIO == "skip" and (summary["audio_collection_queries"] or summary["queries_with_clap_embed"]):
                raise RuntimeError(f"{set_name}: skip인데 CLAP·오디오 조회가 있었다")
            if rt.AUDIO == "on" and summary["audio_collection_queries"] == 0:
                raise RuntimeError(f"{set_name}: on인데 오디오 조회가 0회다")
            if summary["audio_collection_queries"] != summary["queries_with_clap_embed"]:
                raise RuntimeError(f"{set_name}: 오디오 조회 수 ≠ CLAP 임베딩 질의 수")
            conditions["sets_done"][set_name] = {"seconds": seconds, "rows": len(detail), **summary}
            if guard is not None:
                conditions["sets_done"][set_name]["budget_after_set"] = guard.totals()
            write()
            print(f"[v69] {run_name} {set_name} {len(detail)}건 {seconds}초 {json.dumps(summary, ensure_ascii=False)[:300]}",
                  flush=True)
        conditions["guard_after"] = rt.verify_guards()
        if guard is not None:
            conditions["budget_local"] = dict(guard.local)
            conditions["budget_after"] = guard.totals()
            if guard.halted:
                raise RuntimeError(f"실행 중 정지: {guard.halt_reason}")
        conditions["network_attempts"] = list(rt.NET_ATTEMPTS)
        if rt.MODEL == "fault":
            conditions["fault_transport_seen"] = dict(fault_seen)
        conditions["clap"] = dict(recorder.clap)
        conditions["lyrics_source_reads"] = lyrics.reads
        state_after = server_state.capture()
        (run_dir / "server_state_after.json").write_text(json.dumps(state_after, ensure_ascii=False, indent=1), encoding="utf-8")
        changed = server_state.diff(state_after, expected_state)
        conditions["server_state_after_matches_baseline"] = not changed
        if changed:
            raise RuntimeError(f"실행 뒤 서버 상태가 기준과 다르다: {changed}")
        conditions["status"] = "complete"
        conditions["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        conditions["total_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        conditions["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write()
        (run_dir / "DONE").write_text(json.dumps({"partial": partial, "model": rt.MODEL}) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001 - RunAborted(BaseException) 포함
        conditions["status"] = "failed"
        conditions["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        conditions["traceback_tail"] = traceback.format_exc()[-3000:]
        if guard is not None:
            conditions["budget_local"] = dict(guard.local)
            conditions["budget_after"] = guard.totals()
        if rt.MODEL == "fault":
            conditions["fault_transport_seen"] = dict(fault_seen) if "fault_seen" in locals() else None
            conditions["network_attempts"] = list(rt.NET_ATTEMPTS)
        try:
            state_after = server_state.capture()
            (run_dir / "server_state_after.json").write_text(json.dumps(state_after, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
            conditions["server_state_after_matches_baseline"] = not server_state.diff(state_after, expected_state)
        except Exception as state_exc:  # noqa: BLE001
            conditions["server_state_after_error"] = type(state_exc).__name__
        write()
        print(f"[v69] {run_name} 실패: {conditions['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
