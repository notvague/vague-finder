"""Gemini 클라이언트를 만드는 곳 — Vertex AI(GCP 크레딧) 또는 AI Studio API 키.

`GCP_PROJECT_ID`가 있으면 Vertex AI(`aiplatform.googleapis.com`, ADC 인증)로, 없으면 `GEMINI_API_KEY`로
AI Studio(`generativelanguage.googleapis.com`)를 부른다. AI Studio 키 경로에는 GCP 무료 체험 크레딧이
적용되지 않아 Vertex로 옮겼다(2026-10-09). 크레딧이 끝나면 `.env`에서 `GCP_PROJECT_ID`를 비우고
`GEMINI_API_KEY`를 되살리면 코드 수정 없이 돌아간다.

**용도별로 경로를 고정할 수 있다.** 검색(질의 분석·리랭커)은 `GEMINI_RETRIEVAL_BACKEND`(auto / api_key / vertex)를
따른다. 같은 `gemini-3.1-flash-lite`라도 AI Studio와 Vertex는 출력이 달랐고, Vertex는 같은 설정을 두 번 돌려도
Top-10이 달랐다(results_clarify_v11_prebonus). 검색 기준선(v32~v35·봉인 test·재질문 v09)은 AI Studio로 잰 것이라,
크롤링은 Vertex에 두고 검색만 `api_key`로 고정하면 기준선과 같은 경로에서 잴 수 있다. 비우거나 auto면 위 규칙 그대로다.

호출부는 클라이언트를 직접 만들지 않고 `make_genai_client`를 부른다. "Gemini를 쓸 수 있나"도
`gemini_configured`로 본다 — `GEMINI_API_KEY`만 보면 Vertex로 옮긴 뒤 키를 주석 처리했을 때 질의 분석은
규칙 폴백, 리랭커는 CE로 **조용히** 내려간다.

호출부가 키를 **직접 넘기면**(테스트, 예비 키) 환경과 상관없이 그 키로 AI Studio를 부른다.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_GCP_LOCATION = "global"

# 크롤링(정제·댓글 선별) 모델. 검색 모델(GEMINI_MODEL_NAME — 질의 분석·리랭커)과 따로 둔다: 검색 모델이 바뀌면 기준선과 비교할 수 없다.
# gemini-3.5-flash-lite(2026-07 출시): 10/9 Vertex에서 실제 크롤링 프롬프트(곡 3개)로 잰 중앙값이 정제 3.3초·댓글 선별 3.8초
# (3.1-flash-lite 12.9초·18.6초), thinking 0, 응답 형식 6/6. 3.8-flash가 더 새롭지만 기본 thinking(1.2k~2.9k 토큰)으로 14~19초였다.
# `*-latest` 별칭은 가리키는 모델이 조용히 바뀌므로 버전을 적는다.
DEFAULT_CRAWL_MODEL = "gemini-3.5-flash-lite"

# 용도 — 호출부가 넘기는 값. 없으면(None) 용도 설정을 보지 않고 기본 규칙을 따른다(크롤링).
RETRIEVAL = "retrieval"
_PURPOSE_ENV = {RETRIEVAL: "GEMINI_RETRIEVAL_BACKEND"}
# 사람이 "AI Studio"로 부르는 경로를 그대로 적어도 받는다. 기록(runinfo)은 api_key로 통일한다.
_SETTING_ALIASES = {"": "auto", "auto": "auto", "api_key": "api_key", "ai_studio": "api_key", "vertex": "vertex"}
_warned: set[tuple[str, str]] = set()


def crawl_model_name() -> str:
    """크롤링 Gemini 호출의 모델 ID — `GEMINI_CRAWL_MODEL_NAME`, 없으면 DEFAULT_CRAWL_MODEL."""
    return os.getenv("GEMINI_CRAWL_MODEL_NAME", "").strip() or DEFAULT_CRAWL_MODEL


def gcp_project_id() -> str:
    return os.getenv("GCP_PROJECT_ID", "").strip()


def gcp_location() -> str:
    return os.getenv("GCP_LOCATION", "").strip() or DEFAULT_GCP_LOCATION


def env_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def backend_setting(purpose: Optional[str] = None) -> str:
    """용도별 경로 설정 — auto(기본 규칙) / api_key / vertex. 모르는 값은 ValueError(오타가 조용히 auto가 되면 안 된다)."""
    if purpose is None:
        return "auto"
    env = _PURPOSE_ENV[purpose]
    raw = os.getenv(env, "").strip().lower()
    if raw not in _SETTING_ALIASES:
        raise ValueError(f"{env}={raw!r} — auto / api_key / vertex 중 하나여야 한다")
    return _SETTING_ALIASES[raw]


def gemini_backend(api_key: Optional[str] = None, purpose: Optional[str] = None) -> str:
    """실제로 쓸 경로 — `vertex` / `api_key` / `none`. 측정 runinfo에도 이 값을 적는다.

    용도 설정으로 경로를 고정했는데 그 경로의 설정이 없으면 다른 경로로 바꿔 타지 않고 `none`이다 —
    "검색은 AI Studio"라고 정했는데 키가 빠져 Vertex로 돌면 기준선과 다른 경로에서 잰 숫자가 조용히 섞인다.
    """
    if api_key:
        return "api_key"
    setting = backend_setting(purpose)
    if setting == "api_key":
        return "api_key" if env_api_key() else _missing(purpose, setting, "GEMINI_API_KEY")
    if setting == "vertex":
        return "vertex" if gcp_project_id() else _missing(purpose, setting, "GCP_PROJECT_ID")
    if gcp_project_id():
        return "vertex"
    if env_api_key():
        return "api_key"
    return "none"


def _missing(purpose: Optional[str], setting: str, needed: str) -> str:
    key = (purpose or "", setting)
    if key not in _warned:
        _warned.add(key)
        logger.warning("[gemini_client] %s=%s인데 %s가 없다 — Gemini 없음으로 본다(다른 경로로 바꾸지 않는다)",
                       _PURPOSE_ENV.get(purpose or "", "용도 설정"), setting, needed)
    return "none"


def gemini_configured(api_key: Optional[str] = None, purpose: Optional[str] = None) -> bool:
    """이 용도에 쓸 경로의 설정이 있으면 True."""
    return gemini_backend(api_key, purpose) != "none"


def make_genai_client(*, api_key: Optional[str] = None, http_options: Any = None, purpose: Optional[str] = None):
    """`google.genai.Client`를 만든다. 경로 선택은 `gemini_backend`와 같다.

    http_options(타임아웃·재시도)는 두 경로에 똑같이 넘긴다. 설정이 하나도 없으면 ValueError —
    호출부는 그 전에 `gemini_configured`로 걸러 폴백한다.
    """
    from google import genai

    backend = gemini_backend(api_key, purpose)
    if backend == "vertex":
        return genai.Client(
            vertexai=True,
            project=gcp_project_id(),
            location=gcp_location(),
            http_options=http_options,
        )
    if backend == "api_key":
        return genai.Client(api_key=api_key or env_api_key(), http_options=http_options)
    if backend_setting(purpose) != "auto":
        raise ValueError(f"Gemini 설정이 없다 — {_PURPOSE_ENV[purpose]}={backend_setting(purpose)}에 맞는 설정이 필요하다")
    raise ValueError("Gemini 설정이 없다 — GCP_PROJECT_ID(Vertex) 또는 GEMINI_API_KEY가 필요하다")


def gemini_route_info(purpose: Optional[str] = None) -> dict:
    """측정 runinfo용 — 어느 경로·위치로 불렀는지. 키 값도, GCP 프로젝트 ID도 적지 않는다.

    프로젝트 ID는 공개 레포에 올라가는 runinfo에 남으면 안 되는 식별자다(10/9 v11 runinfo 9개에서 손으로 지웠다).
    비교 조건으로 필요한 것은 backend(api_key / vertex)와 location뿐이다. 용도를 주면 그 용도의 설정값(setting)도 적는다 —
    auto로 돌았는지 고정으로 돌았는지가 남아야 .env를 바꾼 뒤의 실행을 가려낼 수 있다.
    """
    backend = gemini_backend(purpose=purpose)
    info = {"backend": backend}
    if backend == "vertex":
        info["location"] = gcp_location()
    if purpose is not None:
        info["setting"] = backend_setting(purpose)
    return info
