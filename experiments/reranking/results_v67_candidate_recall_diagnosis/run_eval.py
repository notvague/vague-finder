"""v67 평가기 실행 하나 = 소리 경로 구성(V67_AUDIO) × 반복 — dev 176(v06 57 + v09 59 + v10 60), 모델 호출 0회.

v66 `run_eval.py`(fake 경로)를 옮겼다. 기존 평가기(`evaluate_search_accuracy.evaluate`)를 그대로 부르고, 리랭커는 결정적 가짜 응답이다.
**가짜 최종 순서는 품질이 아니다.** 쓰는 것은 후보 30(리랭킹 전)·가사 보호·정책 생략·리랭커 그룹 입력·소리 경로 실행 기록이다.

호출 전에 확인(하나라도 어긋나면 멈춘다): 서버 상태 = `--expect-server-state` · 코드 판 사본 = bb7e6f0 src 트리(= v66 수정 후) ·
세 세트 분석 캐시(누락·폴백·항목 오류·drift 0) · 색인 지문 `238299e6c8c17f2a` · 가사 지문 `e0d58f28f1f4a563` · 차단 확인.

    V67_AUDIO=on venv/bin/python run_eval.py --rep 1 --out-root <폴더> --expect-server-state <기준 파일>

종료 코드: 0 성공 · 1 실패 · 2 거부(덮어쓰기) · 4 서버 상태 불일치(호출 전)
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
import v67_eval_runtime as rt  # noqa: E402  (코드 판·환경을 src import 전에 정한다)
import server_state  # noqa: E402

ROOT = rt.ROOT
os.chdir(ROOT)
EXPECTED_ROWS = {"v06": 57, "v09": 59, "v10": 60}


def read_rows(set_name: str):
    with (ROOT / f"experiments/reranking/eval_queries_{set_name}.csv").open(encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["split"].strip() == "dev"]


def precheck_caches(esa, sets) -> dict:
    out = {}
    for set_name in sets:
        rows = read_rows(set_name)
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
                         "file_sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expect-server-state", required=True)
    args = parser.parse_args()
    sets = list(rt.SETS)
    run_name = f"{rt.AUDIO}_rep{args.rep}"
    run_dir = Path(args.out_root) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"이미 산출물이 있다: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "src", "tests"], capture_output=True, text=True, cwd=ROOT).stdout
    conditions: dict = {"run": run_name, "audio": rt.AUDIO, "commit": rt.COMMIT, "mode": "server", "model": "fake",
                        "rep": args.rep, "sets": sets, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "env": {k: os.environ.get(k, "") for k in sorted(rt.FIXED_ENV) + ["QDRANT_PATH", "QDRANT_URL"]},
                        "worktree_head": head, "worktree_src_tests_changes": dirty.splitlines(), "status": "running"}

    def write() -> None:
        (run_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")

    expected_state = json.loads(Path(args.expect_server_state).read_text(encoding="utf-8"))
    state_before = server_state.capture()
    (run_dir / "server_state_before.json").write_text(json.dumps(state_before, ensure_ascii=False, indent=1), encoding="utf-8")
    changed = server_state.diff(state_before, expected_state)
    if changed:
        conditions["status"] = "refused_server_state"
        conditions["error"] = f"서버 상태가 기준과 다르다: {changed}"
        write()
        print(conditions["error"], file=sys.stderr)
        return 4
    try:
        rt.install_guards()
        conditions["code_identity"] = rt.code_identity()
        from src.retrieval import evaluate_search_accuracy as esa

        conditions["analysis_caches"] = precheck_caches(esa, sets)
        conditions["guard_before"] = rt.verify_guards()
        conditions["audio_config"] = rt.install_audio_config()
        import src.backend.api.dependencies as deps
        from src.vector_db import qdrant_backend as qb
        from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME
        from qdrant_reference import ReferenceStore

        esa.preflight_gemini = lambda *a, **k: {"offline": "skipped — 모델 호출 없음"}  # type: ignore[assignment]
        lyrics = rt.install_lyrics_snapshot()
        conditions["lyrics_snapshot"] = lyrics.fingerprint()
        t0 = time.perf_counter()
        client = deps.get_vector_client().client
        conditions["qdrant_open_seconds"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        store = ReferenceStore(client, qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE))
        fp = store.hybrid.fingerprint()
        conditions["reference_build_seconds"] = round(time.perf_counter() - t0, 3)
        conditions["index_fingerprint"] = fp
        if fp["content_sha256_16"] != rt.EXPECTED_INDEX_FINGERPRINT:
            raise RuntimeError(f"색인 지문 {fp['content_sha256_16']} ≠ {rt.EXPECTED_INDEX_FINGERPRINT}")
        recorder = rt.Recorder(store)
        recorder.install()
        t0 = time.perf_counter()
        deps.get_search_router()  # 라우터·임베더 객체 생성(평가기도 같은 캐시를 쓴다). CLAP은 첫 임베딩 때 지연 로드된다
        conditions["router_build_seconds"] = round(time.perf_counter() - t0, 3)
        conditions["sets_done"] = {}
        conditions["cold_before_first_query_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        write()
        for set_name in sets:
            rows = read_rows(set_name)
            if len(rows) != EXPECTED_ROWS[set_name]:
                raise RuntimeError(f"{set_name}: dev {len(rows)}건 ≠ {EXPECTED_ROWS[set_name]}")
            recorder.query_ids = {r["query"].strip(): r["query_id"].strip() for r in rows}
            if len(recorder.query_ids) != len(rows):
                raise RuntimeError(f"{set_name}: 질의 문장이 겹쳐 귀속이 모호하다")
            recorder.rows, recorder.qdrant_calls = {}, []
            out_dir = run_dir / f"{set_name}_dev"
            eval_args = esa.build_parser().parse_args([
                "--input", f"experiments/reranking/eval_queries_{set_name}.csv", "--split", "dev",
                "--analysis-cache", str(rt.CACHES / f"analysis_cache_{set_name}_dev_vertex.json"),
                "--output-dir", str(out_dir),
            ])
            started = time.perf_counter()
            asyncio.run(esa.evaluate(eval_args))
            seconds = round(time.perf_counter() - started, 3)
            summary = recorder.save(out_dir)
            with (out_dir / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
                detail = list(csv.DictReader(fh))
            expected = [r["query_id"].strip() for r in rows]
            if [r["query_id"] for r in detail] != expected:
                raise RuntimeError(f"{set_name}: detail 질의 목록이 기대와 다르다")
            if summary["queries_recorded"] != len(expected) or summary["unattributed_rows"]:
                raise RuntimeError(f"{set_name}: 기록 질의 수 불일치 또는 귀속 없는 기록")
            if summary["lyrics_errors"]:
                raise RuntimeError(f"{set_name}: 가사 검색 예외")
            runinfo = json.loads((out_dir / "search_eval_dev_runinfo.json").read_text(encoding="utf-8"))
            if runinfo["lyrics_source"].get("content_sha256_16") != rt.EXPECTED_LYRICS:
                raise RuntimeError(f"{set_name}: runinfo 가사 지문 불일치")
            if runinfo["analysis"].get("fallback_query_ids") or runinfo["analysis"].get("fingerprint_drift"):
                raise RuntimeError(f"{set_name}: 분석 캐시 폴백·drift")
            # 소리 경로 실행 검사: 경로 실패는 라우터가 빈 결과로 삼키므로(`_path_result`) 구성마다 실제 조회 수로 확인한다
            path_failures = summary["path_failures"]
            if rt.AUDIO == "skip" and (summary["audio_collection_queries"] or summary["queries_with_clap_embed"]):
                raise RuntimeError(f"{set_name}: skip인데 CLAP·오디오 조회가 있었다")
            if rt.AUDIO in ("on", "weight0") and summary["audio_collection_queries"] == 0:
                raise RuntimeError(f"{set_name}: {rt.AUDIO}인데 오디오 조회가 0회다")
            if path_failures:
                raise RuntimeError(f"{set_name}: 경로 실패 기록 {path_failures[:5]}")
            if summary["audio_collection_queries"] != summary["queries_with_clap_embed"]:
                raise RuntimeError(f"{set_name}: 오디오 조회 수 ≠ CLAP 임베딩 질의 수")
            conditions["sets_done"][set_name] = {"seconds": seconds, "rows": len(detail), **summary}
            write()
            print(f"[v67] {run_name} {set_name} {len(detail)}건 {seconds}초 {json.dumps(summary, ensure_ascii=False)[:300]}",
                  flush=True)
        conditions["guard_after"] = rt.verify_guards()
        conditions["network_attempts"] = list(rt.NET_ATTEMPTS)
        conditions["clap"] = dict(recorder.clap)
        conditions["lyrics_source_reads"] = lyrics.reads
        state_after = server_state.capture()
        (run_dir / "server_state_after.json").write_text(json.dumps(state_after, ensure_ascii=False, indent=1), encoding="utf-8")
        changed = server_state.diff(state_after, expected_state)
        if changed:
            raise RuntimeError(f"실행 뒤 서버 상태가 기준과 다르다: {changed}")
        if rt.NET_ATTEMPTS:
            raise RuntimeError(f"차단된 외부 연결 시도 {len(rt.NET_ATTEMPTS)}회")
        conditions["status"] = "complete"
        conditions["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        conditions["total_seconds"] = round(time.perf_counter() - PROCESS_STARTED, 3)
        conditions["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write()
        (run_dir / "DONE").write_text(json.dumps({"audio": rt.AUDIO, "model": "fake"}) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001
        conditions["status"] = "failed"
        conditions["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        conditions["traceback_tail"] = traceback.format_exc()[-3000:]
        write()
        print(f"[v67] {run_name} 실패: {conditions['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
