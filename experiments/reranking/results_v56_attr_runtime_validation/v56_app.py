"""v56 검증용 서버 진입점 — **실제 FastAPI 앱(`src.backend.main:app`)** 을 그대로 쓰되 모델만 바꿔 끼운다.

바뀌는 것(전부 요청 경로 밖 또는 모델 자리):
  - 질의 분석기 → 저장 분석 캐시(`CachedAnalyzer`, 캐시에 없으면 500)
  - 리랭커 → 운영과 같은 `GeminiListwiseReranker` + 가짜 SDK 클라이언트(`tests/fake_gemini_client.py`)
  - 기동 검사 `verify_vertex_setup` → 네트워크 없는 대체
  - 실제 Gemini/Claude 클라이언트 생성은 예외로 차단(`v56_runtime.block_real_models`)
라우트·SearchRouter·Qdrant·Mongo·시간 제한·폴백·보호 배치는 운영 코드 그대로다.

실행(워크트리 루트에서, 환경변수는 start_servers.sh):
  venv/bin/uvicorn v56_app:app --app-dir experiments/reranking/results_v56_attr_runtime_validation --port 8056
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

v56_runtime.block_real_models()
GUARD = v56_runtime.verify_guard()

from src.common import gemini_client as gc  # noqa: E402

gc.verify_vertex_setup = v56_runtime.fake_verify_vertex_setup  # type: ignore[assignment]

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]

import src.backend.main as main_mod  # noqa: E402

app = main_mod.app
app.dependency_overrides[deps.get_query_analyzer] = v56_runtime.get_cached_analyzer


@app.get("/v56/status")
async def v56_status() -> dict:
    """검증 보조 — 차단 상태·설정·가짜 호출 수·분석 캐시 누락. 운영 앱에는 없는 경로다."""
    client = v56_runtime.get_fake_client()
    analyzer = v56_runtime.get_cached_analyzer()
    return {
        "guard": GUARD,
        "config": v56_runtime.describe_config(),
        "fake_calls": len(client.calls),
        "analysis_cache_sources": analyzer.sources,
        "analysis_misses": analyzer.misses,
        "reranker_type": type(deps.get_reranker()).__name__,
    }


print(f"[v56_app] guard={GUARD}", flush=True)
