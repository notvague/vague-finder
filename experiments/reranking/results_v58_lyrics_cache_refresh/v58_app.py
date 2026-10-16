"""v58 검증용 서버 진입점 — **실제 FastAPI 앱(`src.backend.main:app`)** 에 가짜 가사 자료원을 꽂는다.

바뀌는 것(요청 경로 밖 또는 자료·모델 자리):
  - 가사 서비스 → 운영과 같은 `LyricsExactSearchService` + 합성 문서 자료원(`v58_runtime.FakeLyricsSource`, 제어 파일로 느리게/실패)
      `V58_OLD_SERVICE=1`이면 수정 전 코드로 같은 서비스(전후 비교)
  - 질의 분석기 → 고정 분석(`FixedLyricsAnalyzer`, 검증 질의 외에는 500)
  - 리랭커 → `GeminiListwiseReranker` + 가짜 SDK 클라이언트(v56과 같음, 요청은 use_rerank=false)
  - 기동 검사 `verify_vertex_setup` → 네트워크 없는 대체 · 실제 Gemini/Claude/Mongo 호출은 예외로 차단
라우트·SearchRouter·Qdrant(서버 모드)·예열·lifespan 종료는 운영 코드 그대로다.

실행: venv/bin/uvicorn v58_app:app --app-dir experiments/reranking/results_v58_lyrics_cache_refresh --port 8058 (환경변수는 start_server.sh)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v56_runtime  # noqa: E402
import v58_runtime  # noqa: E402

v56_runtime.block_real_models()
v58_runtime.block_mongo()
GUARD = v56_runtime.verify_guard()
GUARD["mongodb.get_collection"] = v58_runtime.verify_mongo_guard()
if GUARD["mongodb.get_collection"] != "blocked":
    raise RuntimeError(f"v58 Mongo 차단 확인 실패: {GUARD}")

from src.common import gemini_client as gc  # noqa: E402

gc.verify_vertex_setup = v56_runtime.fake_verify_vertex_setup  # type: ignore[assignment]

import src.backend.api.dependencies as deps  # noqa: E402

ORIGINAL_GET_LYRICS = deps.get_lyrics_exact_search_service
deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]
# 라우터(`get_search_router`)·예열(`warmup._stages`)·종료(`close_lyrics_exact_search_service`)·/health는 모듈 전역 이름으로 부른다 →
# 여기서 바꾸면 모두 같은 가짜 자료원 서비스를 본다. 라우트의 `Depends(...)`는 원래 함수 객체를 쥐고 있어 override로 잇는다.
deps.get_lyrics_exact_search_service = deps.singleton(v58_runtime.make_lyrics_service)  # type: ignore[assignment]

import src.backend.main as main_mod  # noqa: E402

app = main_mod.app
app.dependency_overrides[deps.get_query_analyzer] = v58_runtime.FixedLyricsAnalyzer
app.dependency_overrides[ORIGINAL_GET_LYRICS] = deps.get_lyrics_exact_search_service


@app.get("/v58/status")
async def v58_status() -> dict:
    """검증 보조 — 차단 상태 · 설정 · 가사 서비스 상태 · 자료원 호출 수. 운영 앱에는 없는 경로다."""
    service = deps.get_lyrics_exact_search_service.peek()  # type: ignore[attr-defined]
    status = getattr(service, "status", None)
    return {
        "guard": GUARD,
        "config": v58_runtime.describe_config(),
        "service_type": None if service is None else type(service).__name__,
        "service_module": None if service is None else type(service).__module__,
        "lyrics": status() if callable(status) else {"state": "no_status_method(old code)"},
        "source_calls": v58_runtime.get_source().calls,
        "fake_reranker_calls": len(v56_runtime.get_fake_client().calls),
    }


print(f"[v58_app] guard={GUARD} old_service={v58_runtime.os.getenv('V58_OLD_SERVICE', '0')}", flush=True)
