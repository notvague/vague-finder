"""Gemini는 AI Studio API 키만 사용한다. GCP/ADC/Vertex 설정은 사용하지 않는다."""
from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_CRAWL_MODEL = "gemini-3.5-flash-lite"
RETRIEVAL = "retrieval"


def crawl_model_name() -> str:
    return os.getenv("GEMINI_CRAWL_MODEL_NAME", "").strip() or DEFAULT_CRAWL_MODEL


def env_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def backend_setting(purpose: Optional[str] = None) -> str:
    """옛 auto/api_key 설정은 호환하고, 제거된 Vertex를 명시하면 거절한다."""
    if purpose is not None:
        raw = os.getenv("GEMINI_RETRIEVAL_BACKEND", "").strip().lower()
        if raw not in {"", "auto", "api_key", "ai_studio"}:
            raise ValueError("GEMINI_RETRIEVAL_BACKEND는 api_key만 지원합니다. Vertex AI는 제거되었습니다.")
    return "api_key"


def gemini_backend(api_key: Optional[str] = None, purpose: Optional[str] = None) -> str:
    backend_setting(purpose)
    return "api_key" if (api_key or env_api_key()) else "none"


def gemini_configured(api_key: Optional[str] = None, purpose: Optional[str] = None) -> bool:
    return gemini_backend(api_key, purpose) != "none"


def make_genai_client(*, api_key: Optional[str] = None, http_options: Any = None,
                      purpose: Optional[str] = None):
    from google import genai

    if not gemini_configured(api_key, purpose):
        raise ValueError("Gemini 설정이 없습니다 — GEMINI_API_KEY가 필요합니다.")
    # SDK가 GOOGLE_GENAI_USE_VERTEXAI 환경변수를 읽더라도 Vertex로 전환하지 않는다.
    return genai.Client(vertexai=False, api_key=api_key or env_api_key(), http_options=http_options)


def gemini_route_info(purpose: Optional[str] = None) -> dict:
    info = {"backend": gemini_backend(purpose=purpose)}
    if purpose is not None:
        info["setting"] = backend_setting(purpose)
    return info
