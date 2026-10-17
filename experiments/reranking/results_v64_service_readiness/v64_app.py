"""v64 검증용 서버 진입점 — **실제 FastAPI 앱(`src.backend.main:app`)** 을 그대로 쓰고 외부 의존만 바꿔 끼운다.

바뀌는 것(`v64_runtime` 설명): 네트워크·Mongo·Gemini·ADC·Claude 차단, 기동 검사 대체, 가사 자료원 = v63 스냅샷,
분석기 = 저장 캐시(dependency_overrides), 리랭커 = 운영 `GeminiListwiseReranker` + 가짜 SDK, Qdrant·가사 구간 기록.
라우트·미들웨어·SearchRouter·Context 기동 검사·예열·가사 판 수명·리랭킹 예산·직렬화는 운영 코드 그대로다.

실행(워크트리 루트, 환경변수는 start_servers.sh):
  venv/bin/uvicorn v64_app:app --app-dir experiments/reranking/results_v64_service_readiness --port 8064
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

PROCESS_STARTED = time.time()

import v64_runtime  # noqa: E402

v64_runtime.install_guards()
GUARD = v64_runtime.verify_guards()
LYRICS = v64_runtime.install_lyrics_snapshot()
v64_runtime.install_observers()

from src.common import gemini_client as gc  # noqa: E402

gc.verify_vertex_setup = v64_runtime.fake_verify_vertex_setup  # type: ignore[assignment]

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v64_runtime.make_fake_reranker)  # type: ignore[assignment]

import src.backend.main as main_mod  # noqa: E402
from src.backend import context_startup  # noqa: E402

app = main_mod.app
app.dependency_overrides[deps.get_query_analyzer] = v64_runtime.get_cached_analyzer
IMPORTED = time.time()


@app.get("/v64/status")
async def v64_status() -> dict:
    """검증 보조 — 차단·외부 연결 시도·설정·가짜 호출 수·분석 캐시 누락·가사 자료원. 운영 앱에는 없는 경로다."""
    client = v64_runtime.get_fake_client()
    analyzer = v64_runtime.get_cached_analyzer()
    source = v64_runtime.lyrics_source()
    return {
        "guard": GUARD,
        "network_attempts": list(v64_runtime.NET_ATTEMPTS),
        "blocked_client_calls": list(v64_runtime.BLOCKED_CALLS),
        "config": v64_runtime.describe_config(),
        "fake_calls": len(client.calls),
        "analysis_cache_sources": analyzer.sources,
        "analysis_misses": len(analyzer.misses),
        "lyrics_source": source.describe() if source else None,
        "lyrics_calls": len(v64_runtime.LYRICS_CALLS),
        "reranker_type": type(deps.get_reranker()).__name__,
        "context_startup": context_startup.status(),
        "process_started": PROCESS_STARTED,
        "imported": IMPORTED,
    }


@app.get("/v64/lyrics_calls")
async def v64_lyrics_calls(since: int = 0) -> dict:
    return {"total": len(v64_runtime.LYRICS_CALLS), "calls": v64_runtime.LYRICS_CALLS[since:]}


print(f"[v64_app] guard={GUARD} lyrics={LYRICS.describe()['content_sha256_16']} import={IMPORTED - PROCESS_STARTED:.1f}s", flush=True)
