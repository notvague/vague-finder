"""v61 공용 — 외부 호출 차단과 실행 조건 고정.

이 진단은 **모델·Mongo·운영 Qdrant·네트워크를 쓰지 않는다.** 테스트·측정 코드가 우회적으로 외부 호출을
만들지 않도록, 서비스 모듈을 import 하기 **전에** `install()`을 부른다. 차단 대상:

  1. `src.common.gemini_client.make_genai_client` / `google.genai.Client` — Gemini(Vertex·AI Studio) 클라이언트 생성
  2. `anthropic.Anthropic` / `AsyncAnthropic` — Claude
  3. `pymongo.MongoClient` + `src.common.mongodb._connect` — Mongo(가사 표면 검색 · 평가기의 가사 지문 · 피드백 저장)
  4. `QDRANT_URL` 제거 — 서버 모드 금지(로컬 복사본만), `HF_HUB_OFFLINE=1` — 모델 다운로드 금지(로컬 캐시만)

`verify()`는 1~3이 실제로 막혔는지 확인한다(막히지 않았으면 RuntimeError). 측정 조건(`fixed_env`)은 A/B/C 세 구성에
똑같이 적용된다 — Context 경로는 서비스 기본값(0.5) 그대로, 속성 보조 경로는 꺼짐(기본값), 리랭커 꺼짐, 피드백 꺼짐.
"""
from __future__ import annotations

import os
from typing import Any, Dict

ROOT_ENV_FIXED: Dict[str, str] = {
    # 벡터 DB — 로컬 복사본만. 운영 색인(artifacts/qdrant)·서버는 열지 않는다
    "QDRANT_PATH": "artifacts/qdrant_v61_audit",
    # 다른 경로의 설정 고정(A/B/C 공통). 서비스 기본값과 같다
    "CONTEXT_WEIGHT": "0.5",
    "ATTR_FILTER_WEIGHT": "0",
    "AUDIO_FUSION_SCALE": "1",
    "SEARCH_REFERENCE_YEAR": "2026",
    # 모델 리랭킹 없음(후보 30개까지만). 분석은 저장 캐시만 읽는다
    "RERANKER_ENABLED": "false",
    "RERANKER_BACKEND": "gemini_listwise",
    "GEMINI_RERANK_RARE_FACT_VERIFY": "0",
    "GEMINI_RETRIEVAL_BACKEND": "vertex",
    # 피드백 수집·보충 경로 끔(Mongo·피드백 컬렉션 접근 없음)
    "FEEDBACK_COLLECTION_ENABLED": "0",
    "FEEDBACK_INDEX_WORKER_ENABLED": "0",
    # 로컬 모델 캐시만
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

REMOVE_ENV = ("QDRANT_URL", "CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY")


class BlockedExternalCall(RuntimeError):
    """외부 클라이언트 생성이 시도됐다 — 진단 조건 위반이므로 조용히 넘기지 않는다."""


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise BlockedExternalCall("v61: 외부 클라이언트 생성이 차단됐다")


class _BlockedClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise BlockedExternalCall(f"v61: {type(self).__name__} 생성이 차단됐다")


_installed = False


def fixed_env(overrides: Dict[str, str] | None = None) -> Dict[str, str]:
    """측정 조건을 환경변수에 **강제로** 넣는다(.env 값보다 우선). 적용된 값을 돌려준다."""
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    applied = dict(ROOT_ENV_FIXED)
    if overrides:
        applied.update(overrides)
    for key, value in applied.items():
        os.environ[key] = value
    return applied


def install() -> None:
    """외부 클라이언트 생성을 막는다. 서비스 모듈을 import 하기 전에 부른다."""
    global _installed
    if _installed:
        return
    import src.common.gemini_client as gc

    gc.make_genai_client = _blocked  # type: ignore[assignment]
    try:
        from google import genai
        import google.genai.client as genai_client_mod

        genai.Client = _BlockedClient  # type: ignore[attr-defined]
        genai_client_mod.Client = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass
    try:
        import anthropic

        anthropic.Anthropic = _BlockedClient  # type: ignore[attr-defined]
        anthropic.AsyncAnthropic = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass
    import pymongo

    pymongo.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    import src.common.mongodb as mongodb

    mongodb.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    mongodb._connect = _blocked  # type: ignore[assignment]
    _installed = True


def verify() -> Dict[str, str]:
    """차단이 실제로 걸렸는지 확인한다. 하나라도 열려 있으면 RuntimeError."""
    result: Dict[str, str] = {}
    import src.common.gemini_client as gc

    try:
        gc.make_genai_client(purpose="retrieval")
        result["make_genai_client"] = "NOT BLOCKED"
    except BlockedExternalCall:
        result["make_genai_client"] = "blocked"
    try:
        from google import genai

        genai.Client(vertexai=True, project="x", location="global")
        result["google.genai.Client"] = "NOT BLOCKED"
    except BlockedExternalCall:
        result["google.genai.Client"] = "blocked"
    except Exception as exc:  # noqa: BLE001 - 차단이 아니라 설정 오류로 실패한 것
        result["google.genai.Client"] = f"NOT BLOCKED ({type(exc).__name__})"
    try:
        import pymongo

        pymongo.MongoClient("mongodb://127.0.0.1:1")
        result["pymongo.MongoClient"] = "NOT BLOCKED"
    except BlockedExternalCall:
        result["pymongo.MongoClient"] = "blocked"
    try:
        import src.common.mongodb as mongodb

        mongodb.get_collection("songs")
        result["src.common.mongodb"] = "NOT BLOCKED"
    except BlockedExternalCall:
        result["src.common.mongodb"] = "blocked"
    result["QDRANT_URL"] = "unset" if not os.getenv("QDRANT_URL") else "SET"
    if any(v not in {"blocked", "unset"} for v in result.values()):
        raise RuntimeError(f"v61 외부 호출 차단 확인 실패: {result}")
    return result
