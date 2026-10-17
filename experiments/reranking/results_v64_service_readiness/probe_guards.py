"""§3 S2 — 서버 대체물이 가리는 실제 경로가 차단 아래에서 외부로 나가지 않는가(별도 프로세스, 모델·Mongo 호출 0).

서버(`v64_app`)는 분석기·기동 검사·가사 자료원을 대체하므로, 그 자리에 실물이 있었다면 무엇이 일어나는지 여기서 확인한다.
  1. 실제 기동 검사 함수 `gemini_client._verify_vertex_setup`(대체 전의 본체) — ADC 읽기·countTokens 전에 차단 예외로 끝나야 한다
  2. 실제 `QueryAnalyzer()`로 캐시에 없는 질의 분석(동기·비동기) — 클라이언트 생성 차단 → 규칙 폴백 또는 예외. 외부 연결 0
  3. 스냅샷을 꽂지 않은 실제 `LyricsExactSearchService` 판 적재 — 가사 전용 Mongo 클라이언트 생성 차단 → 실패 상태
  4. 실제 lifespan 기동 검사 경로(`main.lifespan`)를 대체 없이 TestClient로 — 기동이 멈추는가/경고인가, 외부 연결 0
네트워크 차단 기록(`NET_ATTEMPTS`)이 비어 있어야 한다 — 시도가 있었다면 차단돼 기록되므로 숨겨지지 않는다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/probe_guards.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
for key in ("MONGO_URI", "GOOGLE_APPLICATION_CREDENTIALS", "QDRANT_PATH", "QDRANT_URL"):
    os.environ.pop(key, None)
os.environ.update({
    "GEMINI_RETRIEVAL_BACKEND": "vertex", "GCP_PROJECT_ID": "v64-blocked-project", "GCP_LOCATION": "global",
    "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "RERANKER_BACKEND": "gemini_listwise", "HF_HUB_OFFLINE": "1",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "SEARCH_WARMUP": "0", "CONTEXT_WEIGHT": "0",
    # 4번(lifespan)은 벡터 DB를 열지 않게 아무것도 없는 서버 주소를 준다(연결 실패 = 경고 경로). 루프백이라 차단 대상이 아니다
    "QDRANT_URL": "http://127.0.0.1:1",
})

import v64_runtime  # noqa: E402

v64_runtime.install_guards()
GUARD = v64_runtime.verify_guards()


def outcome(fn):
    started = time.perf_counter()
    try:
        value = fn()
        return {"result": "returned", "value": value, "ms": round((time.perf_counter() - started) * 1000, 1)}
    except BaseException as exc:  # noqa: BLE001
        return {"result": "raised", "error": type(exc).__name__, "message": str(exc)[:200],
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "where": [f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_tb(exc.__traceback__)[-4:]]}


def main() -> int:
    report = {"guard": GUARD, "checks": {}}
    import src.common.gemini_client as gc

    report["checks"]["verify_vertex_setup_real"] = outcome(lambda: gc._verify_vertex_setup(gc.RETRIEVAL, extra_models=["gemini-3.1-flash-lite"]))

    from src.retrieval.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    query = "v64 캐시에 없는 질의 — 실제 분석기 차단 확인용 비 오는 날 듣던 발라드"

    def analyze_sync():
        a = analyzer.analyze(query)
        return {"intent_type": a.intent_type, "fallback_marker": getattr(a, "analysis_source", None) or getattr(a, "source", None)}

    report["checks"]["query_analyzer_sync"] = outcome(analyze_sync)
    report["checks"]["query_analyzer_async"] = outcome(lambda: asyncio.run(analyzer.analyze_async(query)).intent_type)
    report["checks"]["query_analyzer_configured"] = analyzer._configured

    from src.retrieval.lyrics_exact_search import LyricsExactSearchService

    service = LyricsExactSearchService(cache_ttl_seconds=300.0)
    report["checks"]["lyrics_snapshot_without_patch"] = outcome(lambda: len(service.snapshot().documents) if hasattr(service.snapshot(), "documents") else "built")
    report["checks"]["lyrics_status_after"] = {k: v for k, v in service.status().items() if k in ("state", "documents", "last_error", "failures", "refresh_failures")}
    service.close(1.0)

    def lifespan_real_verify():
        from fastapi.testclient import TestClient
        import src.backend.main as main_mod

        with TestClient(main_mod.app) as client:
            health = client.get("/health").json()
        return {"context": health.get("context", {}).get("state"), "warmup": health.get("warmup", {}).get("state")}

    report["checks"]["lifespan_with_real_verify"] = outcome(lifespan_real_verify)
    report["network_attempts"] = list(v64_runtime.NET_ATTEMPTS)
    report["blocked_client_calls"] = list(v64_runtime.BLOCKED_CALLS)
    (HERE / "probe_guards.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
