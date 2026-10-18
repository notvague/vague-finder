"""v72 평가기 실행 하나 = 구성(off|work) × 서버 Qdrant × 반복 — v69 `run_eval.py`(리뷰된 실행기) 사본.

기존 평가기(`evaluate_search_accuracy.evaluate`, 리랭킹 포함 경로)를 그대로 부른다. v69에서 바꾼 것:
- 구성 `V72_CONFIG=off|work`(소리 구성 대신). work는 v71 반복 1 해석 캐시로 작품명 해석을 켠다(`v72_eval_runtime`).
- **고정 부분집합 실행**: `--subset-file <json>`(세트별 질의 id). live는 이 파일이 있어야 하고, 파일 바이트 sha16이
  `--expect-subset-sha16`과 같아야 시작한다(실행 묶음 `run_live.sh`에 고정). 그 밖의 부분 실행(--query-ids·--sets)은 live에서 거부한다.
  부분집합에 질의가 없는 세트는 평가기를 부르지 않는다(preflight countTokens도 없음). 평가기는 질의를 하나씩 독립으로 돌리므로
  후보 수·그룹 구성·보호 정책은 전체 실행과 같다 — 실행 전 무료 검증에서 전체 실행과 프롬프트 지문이 같은지 따로 확인한다.
- 작품명 해석 확인: 실행 전 라우터의 해석기 종류(off는 없음, work는 `cache:`), 실행 뒤 runinfo의 해석 스위치·버전·저장소 지문,
  실제 해석 어댑터 호출 0.
- 그 밖(서버 상태 전후 비교 · 코드 사본 blob 대조 · 분석 캐시 검사 · 색인·가사 지문 · 차단 확인 · 예산 통제 · DONE 표시)은 v69 그대로.

    V72_CONFIG=work venv/bin/python run_eval.py --rep 1 --out-root <폴더> --expect-server-state <기준 파일> [--subset-file …]

산출: <out-root>/<run>/<set>_dev/(평가기 산출물 + v72_queries.jsonl · v72_qdrant_calls.jsonl) · run_conditions.json ·
      server_state_before.json · server_state_after.json · DONE
종료 코드: 0 성공 · 1 실패 · 2 거부(덮어쓰기·게이트·부분 live·부분집합 지문) · 4 서버 상태 불일치(호출 전)
자체 검사용: `V72_FAIL_INJECT=<set>`(fake 전용) · `V72_MODEL=fault V72_FAULT_PLAN=504|403`(가짜 전송, 네트워크 없음).
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
import v72_eval_runtime as rt  # noqa: E402  (코드 판·환경을 src import 전에 정한다)
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


def warmup(esa, deps, subset: dict) -> dict:
    from src.retrieval.work_name_resolution import gate_indirect_work_reference

    excluded = {q for ids in subset.values() for q in ids}
    for set_name in rt.SETS:
        cache = esa.load_cache(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for r in read_rows(set_name):
            qid = r["query_id"].strip()
            if qid in excluded:
                continue
            analysis = cache.get(qid, r["query"].strip())
            if not gate_indirect_work_reference(analysis).triggered:
                continue
            router = deps.get_search_router()
            out: list = []
            started = time.perf_counter()
            asyncio.run(router.search(analysis, top_k=30, use_rerank=False, candidate_k=30, work_resolution_out=out))
            return {"query_id": qid, "set": set_name, "seconds": round(time.perf_counter() - started, 3),
                    "work_status": out[0].status if out else "(none)"}
    raise RuntimeError("예열 질의를 고르지 못했다")


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
    parser.add_argument("--subset-file", default="")
    parser.add_argument("--expect-subset-sha16", default="")
    args = parser.parse_args()
    sets = [s for s in args.sets.split(",") if s]
    partial = bool(args.query_ids) or sets != list(rt.SETS)
    subset: dict = {}
    subset_sha16 = ""
    if args.subset_file:
        if partial:
            print("--subset-file은 --query-ids·--sets와 함께 쓰지 않는다", file=sys.stderr)
            return 2
        raw = Path(args.subset_file).read_bytes()
        subset_sha16 = hashlib.sha256(raw).hexdigest()[:16]
        if args.expect_subset_sha16 and subset_sha16 != args.expect_subset_sha16:
            print(f"부분집합 지문 {subset_sha16} ≠ 기대 {args.expect_subset_sha16}", file=sys.stderr)
            return 2
        subset = {k: list(v) for k, v in json.loads(raw)["query_ids_by_set"].items()}
        if set(subset) - set(rt.SETS):
            print(f"부분집합에 모르는 세트: {sorted(set(subset) - set(rt.SETS))}", file=sys.stderr)
            return 2
        sets = [s for s in rt.SETS if subset.get(s)]
    if rt.MODEL == "fault":
        if os.environ.get("V72_LIVE_APPROVED"):
            print("fault는 승인 표시(V72_LIVE_APPROVED)와 함께 쓸 수 없다", file=sys.stderr)
            return 2
        if (not args.ledger or args.deadline_window_seconds <= 0 or args.cost_cap <= 0 or args.attempt_cap <= 0
                or Path(args.ledger).resolve().is_relative_to(HERE)):
            print("fault는 결과 폴더 밖 임시 장부와 한도 인자가 모두 필요하다", file=sys.stderr)
            return 2
        if os.environ.get("V72_FAULT_PLAN") not in rt.FAULT_PLANS:
            print(f"V72_FAULT_PLAN은 {rt.FAULT_PLANS} 중 하나", file=sys.stderr)
            return 2
    elif os.environ.get("V72_FAULT_PLAN"):
        print("V72_FAULT_PLAN은 fault에서만", file=sys.stderr)
        return 2
    if rt.MODEL == "live":
        if (os.environ.get("V72_LIVE_APPROVED") != "1" or not args.ledger or args.deadline_window_seconds <= 0
                or args.cost_cap <= 0 or args.attempt_cap <= 0):
            print("live 실행은 V72_LIVE_APPROVED=1 · --ledger · --deadline-window-seconds · --cost-cap · --attempt-cap이 모두 필요하다",
                  file=sys.stderr)
            return 2
        if partial:
            print("live 실행은 고정 부분집합(--subset-file)만 허용한다(--query-ids·--sets 거부)", file=sys.stderr)
            return 2
        if not args.subset_file or not args.expect_subset_sha16:
            print("live 실행은 --subset-file과 --expect-subset-sha16이 모두 필요하다", file=sys.stderr)
            return 2
        if os.environ.get("V72_FAIL_INJECT"):
            print("V72_FAIL_INJECT는 fake에서만", file=sys.stderr)
            return 2
        if not rt.LIVE_CREDENTIALS_PRESENT:
            print("live 실행에는 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS가 필요하다(with_env.py)", file=sys.stderr)
            return 2
    if os.environ.get("V72_PATH_FAIL_INJECT") and rt.MODEL != "fake":
        print("V72_PATH_FAIL_INJECT는 fake에서만", file=sys.stderr)
        return 2
    if os.environ.get("V72_FAIL_INJECT") and rt.MODEL != "fake":
        print("V72_FAIL_INJECT는 fake에서만", file=sys.stderr)
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
                        "subset_file": Path(args.subset_file).name if args.subset_file else "", "subset_sha16": subset_sha16,
                        "subset_query_ids": subset, "work_inputs": rt.WORK_INPUTS,
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
                conditions["fault_plan"] = os.environ["V72_FAULT_PLAN"]
            import v72_budget

            guard = v72_budget.install(Path(args.ledger), cost_cap_usd=args.cost_cap, attempt_cap=args.attempt_cap,
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

        conditions["analysis_caches"] = {s: precheck_caches(esa, [s], ",".join(subset[s]) if subset else args.query_ids)[s]
                                         for s in sets}
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
        # 예열(v72): 새 프로세스의 첫 판정 통과 질의는 인코더 지연 로드를 작품명 작업 대기(약 4초+여유) 안에서 떠안아
        # failed:timeout이 될 수 있다(부분집합 1차 가짜 실행 m103에서 확인). 전체 실행에서는 앞 질의들이 예열해 준다.
        # 그래서 기록을 걸기 전에 고정 규칙으로 고른 질의 하나(부분집합 밖·판정 통과·dev 순서 첫 질의)를 리랭킹 없이 한 번 검색한다.
        # 정답은 보지 않는다. 기록·평가기 산출에 들어가지 않는다. live에서도 모델 호출이 없다(아래 사전 시도 0 확인).
        conditions["warmup"] = warmup(esa, deps, subset)
        recorder = rt.Recorder(store)
        recorder.install()
        # 작품명 해석기 종류 — 라우터 공장이 만든 것을 실행 전에 본다(off: 없음, work: cache 해석기)
        router_probe = deps.get_search_router()
        work_rt = getattr(router_probe, "_work_names", None)
        resolver_name = getattr(getattr(work_rt, "resolver", None), "name", "") if work_rt is not None else ""
        conditions["work_resolver"] = resolver_name or "(none)"
        if rt.CONFIG == "off" and work_rt is not None:
            raise RuntimeError("off인데 라우터에 작품명 해석기가 있다")
        if rt.CONFIG == "work" and not resolver_name.startswith("cache:"):
            raise RuntimeError(f"work인데 해석기가 캐시가 아니다: {resolver_name!r}")
        if guard is not None:
            conditions["budget_before_first_set"] = guard.totals()
            if guard.local["attempts"]:
                raise RuntimeError(f"사전 확인 단계에서 Vertex 시도 {guard.local['attempts']}회가 생겼다")
        conditions["sets_done"] = {}
        conditions["cold_before_first_query_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        write()
        for set_name in sets:
            if os.environ.get("V72_FAIL_INJECT") == set_name:
                raise RuntimeError(f"V72_FAIL_INJECT={set_name}(자체 검사)")
            rows = read_rows(set_name)
            if len(rows) != EXPECTED_ROWS[set_name]:
                raise RuntimeError(f"{set_name}: dev {len(rows)}건 ≠ {EXPECTED_ROWS[set_name]}")
            # 평가기는 --query-ids 순서로 돈다 — 전체 실행과 같은 dev 순서로 넘긴다
            set_query_ids = (",".join(r["query_id"].strip() for r in rows if r["query_id"].strip() in set(subset[set_name]))
                             if subset else args.query_ids)
            if subset and not set(subset[set_name]) <= {r["query_id"].strip() for r in rows}:
                raise RuntimeError(f"{set_name}: 부분집합 질의가 dev에 없다")
            recorder.query_ids = {r["query"].strip(): r["query_id"].strip() for r in rows}
            if len(recorder.query_ids) != len(rows):
                raise RuntimeError(f"{set_name}: 질의 문장이 겹쳐 귀속이 모호하다")
            recorder.rows, recorder.qdrant_calls = {}, []
            out_dir = run_dir / f"{set_name}_dev"
            eval_args = esa.build_parser().parse_args([
                "--input", f"experiments/reranking/eval_queries_{set_name}.csv", "--split", "dev",
                "--analysis-cache", str(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json"),
                "--output-dir", str(out_dir), *(["--query-ids", set_query_ids] if set_query_ids else []),
            ])
            started = time.perf_counter()
            asyncio.run(esa.evaluate(eval_args))
            seconds = round(time.perf_counter() - started, 3)
            summary = recorder.save(out_dir)
            with (out_dir / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
                detail = list(csv.DictReader(fh))
            expected = [r["query_id"].strip() for r in rows]
            if set_query_ids:
                wanted = set(set_query_ids.split(","))
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
            wsw = runinfo["ranking_switches"]["work_name_resolution"]
            if rt.CONFIG == "off" and wsw != {"enabled": False}:
                raise RuntimeError(f"{set_name}: off인데 해석 스위치 {wsw}")
            if rt.CONFIG == "work":
                bad = {k: wsw.get(k) for k, v in rt.EXPECTED_WORK_VERSIONS.items() if wsw.get(k) != v}
                if (not wsw.get("enabled") or not str(wsw.get("resolver", "")).startswith("cache:") or bad
                        or wsw.get("aux_weight") != 0.5 or wsw.get("store_sha256_16") != rt.WORK_MENTIONS_SHA16):
                    raise RuntimeError(f"{set_name}: 해석 스위치가 고정 조건과 다르다: {wsw}")
            if rt.WORK_RESOLVER_CALLS:
                raise RuntimeError(f"{set_name}: 실제 작품명 해석 호출 {rt.WORK_RESOLVER_CALLS}")
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
            print(f"[v72] {run_name} {set_name} {len(detail)}건 {seconds}초 {json.dumps(summary, ensure_ascii=False)[:300]}",
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
        print(f"[v72] {run_name} 실패: {conditions['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
