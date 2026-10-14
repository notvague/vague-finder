"""공용 listwise 프롬프트·파서·보호 규칙을 사용하는 Claude 리랭커.

기본은 Sonnet + structured outputs. CLAUDE_MODEL_NAME으로 모델 ID를 바꾼다.
Google 도구는 지원하지 않으며 활성화된 경우 호출 전에 거절한다.
스키마는 JSON 구조를 보장하고 후보 누락·중복 처리는 공용 파서에 맡긴다.
"""
from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Dict, Optional

from src.common.claude_client import (
    claude_configured, claude_model_name, claude_structured_output, make_claude_genai_client,
)
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig

# structured outputs 제약(2026-10-14 문서): 객체마다 additionalProperties=false, 수치 · 길이 제약은 쓸 수 없다.
RANKING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query_confidence": {"type": "number"},
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "relevance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "relevance", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["query_confidence", "ranking"],
    "additionalProperties": False,
}

MISSING_CONFIG = "Claude 설정이 없다 — CLAUDE_API(또는 ANTHROPIC_API_KEY)가 필요하다"


class ClaudeListwiseReranker(GeminiListwiseReranker):
    def __init__(self, config: Optional[GeminiListwiseRerankerConfig] = None, client: Any = None):
        base = config or GeminiListwiseRerankerConfig.from_env()
        if base.use_search_grounding or base.rare_fact_verification:
            raise ValueError(
                "claude_listwise는 Google Search를 쓰는 기능과 함께 쓸 수 없다 — "
                "GEMINI_RERANK_USE_SEARCH · GEMINI_RERANK_RARE_FACT_VERIFY를 끈다"
            )
        if client is None and base.enabled and not claude_configured():
            # 키가 없다고 다른 모델로 바꿔 타지 않는다(설정 오류). 리랭킹을 끈 실행(RERANKER_ENABLED=false)은
            # 키 없이도 만들어진다 — 검색 순서만 내는 재현 실행이 Claude 키를 요구할 이유가 없다.
            raise ValueError(MISSING_CONFIG)
        self.structured_output = claude_structured_output()
        self._client_lock = threading.Lock()
        super().__init__(config=replace(base, model_name=claude_model_name()), client=client)

    def _env_configured(self) -> bool:
        return claude_configured()

    def _missing_config_message(self) -> str:
        return "Claude 설정 없음(CLAUDE_API 또는 ANTHROPIC_API_KEY)"

    @property
    def _gemini(self) -> Any:
        """어댑터는 첫 호출에서 만든다 — 만드는 것만으로 키를 요구하므로 생성 시점에 만들면 꺼진 리랭커도 키 없이 죽는다.
        어댑터 하나가 HTTP 풀을 쥐므로 겹친 첫 호출이 두 벌을 만들지 않게 잠근다."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = make_claude_genai_client(
                        tag="rerank", timeout_seconds=self.config.request_timeout_seconds
                    )
        return self._client

    def _new_client(self) -> Any:
        # 어댑터는 연결 풀을 자기 루프 스레드에 묶어 프로세스 동안 재사용한다 — 호출마다 새로 만들지 않는다.
        return self._gemini

    def _generate(self, *, contents: str, config: Any, deadline: Optional[float]):
        # 순위 호출만 JSON을 요청한다(Search evidence · 희소 사실 검증은 이 백엔드에서 꺼져 있다).
        if self.structured_output == "json_schema" and isinstance(config, dict) \
                and config.get("response_mime_type") == "application/json":
            config = {**config, "response_json_schema": RANKING_SCHEMA}
        return super()._generate(contents=contents, config=config, deadline=deadline)
