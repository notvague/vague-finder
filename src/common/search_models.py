"""검색 모델 선택과 기본값. 요청별 선택에 환경변수를 변경하지 않는다."""
from __future__ import annotations

import os
from typing import Any, Dict, Literal

from src.common.claude_client import claude_configured, claude_model_name
from src.common.gemini_client import RETRIEVAL, gemini_configured

ModelProvider = Literal["claude", "gemini"]
PROVIDERS: tuple[str, ...] = ("claude", "gemini")
_KEY_HINT = {"claude": "CLAUDE_API(또는 ANTHROPIC_API_KEY)", "gemini": "GEMINI_API_KEY"}


def default_provider() -> ModelProvider:
    value = os.getenv("SEARCH_MODEL_PROVIDER", "claude").strip().lower()
    if value not in PROVIDERS:
        raise ValueError("SEARCH_MODEL_PROVIDER는 claude 또는 gemini여야 합니다.")
    return value  # type: ignore[return-value]


def require_default_provider() -> ModelProvider:
    """기본 모델과 그 키가 **모두** 있어야 한다. 서버 기동 검사용.

    철자만 보면 Gemini 키만 있는 환경에서 서버가 정상으로 뜬 뒤 `model_provider`를 생략한 모든 검색이
    503으로 떨어진다. 다시 시도해도 낫지 않는 설정 오류라 첫 검색이 아니라 기동에서 멈춘다(PR #31 리뷰와 같은 이유).
    """
    provider = default_provider()
    if not provider_configured(provider):
        other = "gemini" if provider == "claude" else "claude"
        raise ValueError(
            f"SEARCH_MODEL_PROVIDER={provider}인데 {_KEY_HINT[provider]}가 없습니다 — "
            f"키를 넣거나 SEARCH_MODEL_PROVIDER={other}로 둡니다."
        )
    return provider


def model_name(provider: ModelProvider) -> str:
    if provider == "claude":
        return claude_model_name()
    return os.getenv("GEMINI_MODEL_NAME", "").strip() or "gemini-3.1-flash-lite"


def provider_configured(provider: ModelProvider) -> bool:
    return claude_configured() if provider == "claude" else gemini_configured(purpose=RETRIEVAL)


def model_options() -> Dict[str, Any]:
    """화면 선택창용 — 모델 이름·기본값·키 설정 여부만. 응답 모양은 `SearchModelOptions`(schemas)가 검증한다."""
    return {
        "default": default_provider(),
        "models": [
            {"provider": p, "label": "Claude" if p == "claude" else "Gemini",
             "model": model_name(p), "available": provider_configured(p)}
            for p in PROVIDERS
        ],
    }
