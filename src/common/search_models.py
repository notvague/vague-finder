"""검색 모델 선택과 기본값. 요청별 선택에 환경변수를 변경하지 않는다."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

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
    """기본 모델을 지금 설정으로 **쓸 수 있어야** 한다. 서버 기동 검사용.

    철자만 보면 Gemini 키만 있는 환경에서 서버가 정상으로 뜬 뒤 `model_provider`를 생략한 모든 검색이
    503으로 떨어진다. 다시 시도해도 낫지 않는 설정 오류라 첫 검색이 아니라 기동에서 멈춘다(PR #31 리뷰와 같은 이유).
    """
    provider = default_provider()
    reason = unavailable_reason(provider)
    if reason:
        other = "gemini" if provider == "claude" else "claude"
        raise ValueError(
            f"SEARCH_MODEL_PROVIDER={provider}인데 {reason} — 설정을 고치거나 SEARCH_MODEL_PROVIDER={other}로 둡니다."
        )
    return provider


def model_name(provider: ModelProvider) -> str:
    if provider == "claude":
        return claude_model_name()
    return os.getenv("GEMINI_MODEL_NAME", "").strip() or "gemini-3.1-flash-lite"


def unavailable_reason(provider: ModelProvider) -> Optional[str]:
    """이 모델을 지금 설정으로 쓸 수 없는 이유. None이면 쓸 수 있다.

    키 유무만 보지 않는다 — Claude 리랭커는 Google Search를 쓰는 옵션(GEMINI_RERANK_USE_SEARCH ·
    GEMINI_RERANK_RARE_FACT_VERIFY)과 함께 만들 수 없어, 그 상태로 Claude를 고르면 첫 검색이 500으로 떨어진다(#63 리뷰).
    기동 검사(`require_default_provider`)·요청의 503·`/models`의 `available`이 모두 이 함수를 본다.
    """
    if provider == "claude":
        if not claude_configured():
            return f"{_KEY_HINT['claude']}가 없습니다"
        blockers = _claude_reranker_blockers()
        if blockers:
            return f"{' · '.join(blockers)}가 켜져 있어 Claude 리랭킹을 쓸 수 없습니다"
        return None
    if not gemini_configured(purpose=RETRIEVAL):
        return f"{_KEY_HINT['gemini']}가 없습니다"
    return None


def _claude_reranker_blockers() -> List[str]:
    """Claude listwise와 함께 쓸 수 없는 리랭커 옵션 중 켜진 것. 리랭킹이 꺼져 있으면 해당 없다."""
    # 함수 안 import — gemini_listwise_reranker → schemas.search → 이 모듈로 돌아오는 순환이 있다.
    from src.retrieval.gemini_listwise_reranker import GeminiListwiseRerankerConfig

    config = GeminiListwiseRerankerConfig.from_env()
    if not config.enabled:
        return []
    flags = [("GEMINI_RERANK_USE_SEARCH", config.use_search_grounding),
             ("GEMINI_RERANK_RARE_FACT_VERIFY", config.rare_fact_verification)]
    return [name for name, on in flags if on]


def provider_configured(provider: ModelProvider) -> bool:
    """지금 설정으로 쓸 수 있는가 — `unavailable_reason`이 None."""
    return unavailable_reason(provider) is None


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
