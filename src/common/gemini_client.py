"""Gemini 클라이언트를 만드는 곳 — Vertex AI(GCP 크레딧) 또는 AI Studio API 키.

`GCP_PROJECT_ID`가 있으면 Vertex AI(`aiplatform.googleapis.com`, ADC 인증)로, 없으면 `GEMINI_API_KEY`로
AI Studio(`generativelanguage.googleapis.com`)를 부른다. AI Studio 키 경로에는 GCP 무료 체험 크레딧이
적용되지 않아 Vertex로 옮겼다(2026-10-09). 크레딧이 끝나면 `.env`에서 `GCP_PROJECT_ID`를 비우고
`GEMINI_API_KEY`를 되살리면 코드 수정 없이 돌아간다.

호출부는 클라이언트를 직접 만들지 않고 `make_genai_client`를 부른다. "Gemini를 쓸 수 있나"도
`gemini_configured`로 본다 — `GEMINI_API_KEY`만 보면 Vertex로 옮긴 뒤 키를 주석 처리했을 때 질의 분석은
규칙 폴백, 리랭커는 CE로 **조용히** 내려간다.

호출부가 키를 **직접 넘기면**(테스트, 예비 키) 환경과 상관없이 그 키로 AI Studio를 부른다.
"""
from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_GCP_LOCATION = "global"


def gcp_project_id() -> str:
    return os.getenv("GCP_PROJECT_ID", "").strip()


def gcp_location() -> str:
    return os.getenv("GCP_LOCATION", "").strip() or DEFAULT_GCP_LOCATION


def env_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def gemini_backend(api_key: Optional[str] = None) -> str:
    """실제로 쓸 경로 — `vertex` / `api_key` / `none`. 측정 runinfo에도 이 값을 적는다."""
    if api_key:
        return "api_key"
    if gcp_project_id():
        return "vertex"
    if env_api_key():
        return "api_key"
    return "none"


def gemini_configured(api_key: Optional[str] = None) -> bool:
    """Vertex 설정이나 API 키 중 하나라도 있으면 True."""
    return gemini_backend(api_key) != "none"


def make_genai_client(*, api_key: Optional[str] = None, http_options: Any = None):
    """`google.genai.Client`를 만든다. 경로 선택은 `gemini_backend`와 같다.

    http_options(타임아웃·재시도)는 두 경로에 똑같이 넘긴다. 설정이 하나도 없으면 ValueError —
    호출부는 그 전에 `gemini_configured`로 걸러 폴백한다.
    """
    from google import genai

    backend = gemini_backend(api_key)
    if backend == "vertex":
        return genai.Client(
            vertexai=True,
            project=gcp_project_id(),
            location=gcp_location(),
            http_options=http_options,
        )
    if backend == "api_key":
        return genai.Client(api_key=api_key or env_api_key(), http_options=http_options)
    raise ValueError("Gemini 설정이 없다 — GCP_PROJECT_ID(Vertex) 또는 GEMINI_API_KEY가 필요하다")


def gemini_route_info() -> dict:
    """측정 runinfo용 — 어느 경로·프로젝트·위치로 불렀는지. 키 값은 적지 않는다."""
    backend = gemini_backend()
    info = {"backend": backend}
    if backend == "vertex":
        info.update(project=gcp_project_id(), location=gcp_location())
    return info
