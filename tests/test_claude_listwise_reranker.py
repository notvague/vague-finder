"""Claude listwise 백엔드 — 선택했을 때만 만들어지고, Gemini listwise와 같은 프롬프트·파서를 쓴다."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.backend.schemas.search import MatchingTrack
from src.retrieval.claude_listwise_reranker import ClaudeListwiseReranker
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.explain import RERANK_APPLIED, RERANK_FAILED


def _track(song_id: str, title: str, score: float) -> MatchingTrack:
    return MatchingTrack(id=song_id, title=title, artist="artist", score=score, retrieval_score=score)


class _Response:
    def __init__(self, text: str):
        self.text = text


class _FakeClaude:
    """`models.generate_content`만 가진 가짜 — 어댑터와 같은 면이다(`aio` 없음)."""

    def __init__(self, texts: List[str]):
        self._texts = list(texts)
        self.calls: List[Dict[str, Any]] = []
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: Any = None) -> _Response:
        self.calls.append({"model": model, "contents": contents, "config": config})
        text = self._texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return _Response(text)


def _ranking(ids: List[str]) -> str:
    return json.dumps({"query_confidence": 0.9,
                       "ranking": [{"id": i, "relevance": 1.0 - 0.1 * n, "reason": "이유"} for n, i in enumerate(ids)]})


TRACKS = [_track("1", "가", 0.9), _track("2", "나", 0.8), _track("3", "다", 0.7)]


def _clean_env(monkeypatch) -> None:
    for name in ("CLAUDE_MODEL_NAME", "VF_CLAUDE_THINKING", "VF_CLAUDE_EFFORT", "CLAUDE_STRUCTURED_OUTPUT", "GEMINI_RERANK_USE_SEARCH",
                 "GEMINI_RERANK_RARE_FACT_VERIFY", "GEMINI_RERANK_MODEL_NAME", "GEMINI_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)


def test_same_prompt_as_gemini_and_claude_model_name(monkeypatch):
    _clean_env(monkeypatch)
    fake = _FakeClaude([_ranking(["3", "1", "2"])])
    reranker = ClaudeListwiseReranker(client=fake)
    run = reranker.rerank_run("흐릿한 기억의 노래", TRACKS, 3)
    assert run.status == RERANK_APPLIED and [t.id for t in run.tracks][0] == "3"
    assert fake.calls[0]["model"] == "claude-sonnet-5-5" == reranker.config.model_name
    # 프롬프트는 Gemini listwise가 만드는 것과 글자까지 같다 — 모델만 다르다.
    rank = {t.id: n for n, t in enumerate(TRACKS, start=1)}
    assert fake.calls[0]["contents"] == GeminiListwiseReranker._build_prompt("흐릿한 기억의 노래", TRACKS, rank)
    # 정책·답변 반영 속성도 물려받는다.
    assert reranker.skips_image_dominant and reranker.skips_unverifiable_mv_scene and reranker.uses_clarify_answers


def test_non_json_output_retries_with_correction_then_fails(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setattr("src.retrieval.gemini_listwise_reranker.time.sleep", lambda _s: None)
    # 본 순위 파서는 엄격한 JSON만 받는다(Gemini는 JSON 응답 스위치로 보장받는다). 코드 울타리로 감싼 답도
    # 형식 위반으로 세고 다시 묻는다 — 어댑터가 울타리를 벗겨 주지 않으므로 그대로 드러난다.
    fenced = "```json\n" + _ranking(["2", "1", "3"]) + "\n```"
    fake = _FakeClaude([fenced, _ranking(["2", "1", "3"])])
    run = ClaudeListwiseReranker(client=fake).rerank_run("질의", TRACKS, 3)
    assert run.status == RERANK_APPLIED and len(fake.calls) == 2
    assert fake.calls[1]["contents"].startswith(fake.calls[0]["contents"])        # 같은 프롬프트 + 정정 문장
    assert fake.calls[1]["contents"] != fake.calls[0]["contents"]

    failing = _FakeClaude([RuntimeError("x"), RuntimeError("x"), RuntimeError("x")])
    run = ClaudeListwiseReranker(client=failing).rerank_run("질의", TRACKS, 3)
    # 실패는 검색 순서 그대로 + failed — 성공으로 세지 않는다.
    assert run.status == RERANK_FAILED and [t.id for t in run.tracks] == ["1", "2", "3"]


def test_search_features_are_refused(monkeypatch):
    _clean_env(monkeypatch)
    base = GeminiListwiseRerankerConfig.from_env()
    for change in ({"use_search_grounding": True}, {"rare_fact_verification": True}):
        config = GeminiListwiseRerankerConfig(**{**base.__dict__, **change})
        with pytest.raises(ValueError, match="Google Search"):
            ClaudeListwiseReranker(config=config, client=_FakeClaude([]))
        # 꺼진 리랭커는 옵션을 쓰지 않으므로 만들어진다 — 검색 순서만 내는 재현 실행
        off = GeminiListwiseRerankerConfig(**{**config.__dict__, "enabled": False})
        assert not ClaudeListwiseReranker(config=off, client=_FakeClaude([])).enabled


def test_measurement_announces_the_reranker_and_how_to_get_the_gemini_baseline(monkeypatch):
    """RERANKER_BACKEND를 비우면 CLI 리랭커가 SEARCH_MODEL_PROVIDER(claude)를 따른다 — 예전 재현 명령이 조용히 Sonnet으로
    재지 않게 시작 줄에 무엇으로 재는지와 Gemini 기준선 재현법을 찍는다(#63 리뷰)."""
    from src.retrieval.evaluate_search_accuracy import describe_reranker

    _clean_env(monkeypatch)
    monkeypatch.delenv("SEARCH_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("RERANKER_BACKEND", raising=False)
    line = describe_reranker(ClaudeListwiseReranker(client=_FakeClaude([])))
    assert line.startswith("리랭커: claude_listwise (SEARCH_MODEL_PROVIDER 기본값 claude)") and "gemini_listwise" in line
    monkeypatch.setenv("RERANKER_BACKEND", "gemini_listwise")
    gemini = describe_reranker(GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), client=_FakeClaude([])))
    assert gemini == "리랭커: gemini_listwise (RERANKER_BACKEND=gemini_listwise)"


def test_factory_defaults_to_claude_and_allows_explicit_override(monkeypatch):
    from src.backend.api import dependencies as deps
    from src.retrieval.evaluate_search_accuracy import _actual_backend
    from src.retrieval.reranker import MusicReranker

    _clean_env(monkeypatch)
    monkeypatch.delenv("SEARCH_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "시험용")
    monkeypatch.setenv("CLAUDE_API", "test-key-not-real")
    try:
        # 기본값은 Claude이며 CLI에서 공급자를 명시적으로 바꿀 수 있다.
        monkeypatch.delenv("RERANKER_BACKEND", raising=False)
        deps.get_reranker.cache_clear()
        default = deps.get_reranker()
        assert type(default) is ClaudeListwiseReranker and _actual_backend(default) == "claude_listwise"
        monkeypatch.setenv("RERANKER_BACKEND", "gemini_listwise")
        deps.get_reranker.cache_clear()
        assert type(deps.get_reranker()) is GeminiListwiseReranker

        monkeypatch.setenv("RERANKER_BACKEND", "claude_listwise")
        deps.get_reranker.cache_clear()
        chosen = deps.get_reranker()
        assert type(chosen) is ClaudeListwiseReranker and _actual_backend(chosen) == "claude_listwise"
        assert chosen.enabled and chosen.config.model_name == "claude-sonnet-5-5"

        # Claude 키가 없으면 다른 모델로 바꿔 타지 않고 설정 오류를 낸다.
        monkeypatch.delenv("CLAUDE_API", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        deps.get_reranker.cache_clear()
        with pytest.raises(ValueError, match="Claude 설정"):
            deps.get_reranker()
    finally:
        deps.get_reranker.cache_clear()


def test_disabled_reranker_needs_no_key_and_the_client_is_built_on_first_call(monkeypatch):
    """어댑터는 첫 호출에서 만든다. 리랭킹을 끈 실행(검색 순서 재현)이 Claude 키를 요구하며 죽지 않는다 —
    Gemini listwise는 설정이 없으면 꺼진 채로 만들어지는데 Claude만 생성에서 죽으면 get_search_router()가 통째로 죽는다."""
    _clean_env(monkeypatch)
    monkeypatch.delenv("CLAUDE_API", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base = GeminiListwiseRerankerConfig.from_env()
    off = GeminiListwiseRerankerConfig(**{**base.__dict__, "enabled": False})
    disabled = ClaudeListwiseReranker(config=off)
    assert not disabled.enabled and disabled._client is None
    # 켜져 있는데 키가 없으면 다른 모델로 바꿔 타지 않고 설정 오류다
    with pytest.raises(ValueError, match="Claude 설정"):
        ClaudeListwiseReranker(config=base)

    monkeypatch.setenv("CLAUDE_API", "test-key-not-real")
    reranker = ClaudeListwiseReranker(config=base)
    assert reranker.enabled and reranker._client is None, "만들기만 해서는 어댑터(HTTP 풀)가 없다"
    client = reranker._new_client()
    assert client is reranker._client and reranker._new_client() is client, "호출마다 새로 만들지 않는다"
    assert client._tag == "rerank" and client._timeout == base.request_timeout_seconds


def test_rerank_budget_cuts_a_slow_claude_request(monkeypatch, tmp_path):
    """리랭킹 예산이 요청 제한보다 먼저 끝나면, 진행 중인 Claude 요청을 끊고 검색 순서 + failed를 낸다.

    어댑터에 `aio`가 있어야 리랭커의 취소 경로(`asyncio.wait_for(…, min(요청 제한, 남은 예산))`)를 탄다 —
    없던 때는 동기 호출이라 HTTP 제한(60초)까지 기다렸다.
    """
    import asyncio
    import time

    import httpx

    from src.common import claude_client as cc

    _clean_env(monkeypatch)
    for name in ("CLAUDE_BUDGET_USD", "CLAUDE_TIMEOUT_SECONDS", "CLAUDE_RAW_LOG", "CLAUDE_JSON_FENCE"):
        monkeypatch.delenv(name, raising=False)
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CLAUDE_USAGE_LOG", str(log))
    monkeypatch.setattr("src.retrieval.gemini_listwise_reranker.time.sleep", lambda _s: None)
    cancelled: List[bool] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200, json={})

    client = cc.ClaudeGenaiClient(api_key="test-key-not-real", tag="rerank", timeout_seconds=60,
                                  transport=httpx.MockTransport(handler))
    base = GeminiListwiseRerankerConfig.from_env()
    config = GeminiListwiseRerankerConfig(**{**base.__dict__, "request_timeout_seconds": 60.0,
                                             "time_budget_seconds": 0.5, "max_retries": 3})
    reranker = ClaudeListwiseReranker(config=config, client=client)
    started = time.monotonic()
    run = reranker.rerank_run("질의", TRACKS, 3)
    assert time.monotonic() - started < 2.0
    assert run.status == RERANK_FAILED and [t.id for t in run.tracks] == ["1", "2", "3"]
    deadline = time.monotonic() + 3
    while not log.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["error"] for r in records] == ["cancelled"] and cancelled == [True]


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _walk_objects(value)


def test_structured_output_changes_only_the_output_format(monkeypatch):
    """스키마는 순위 호출에만 붙고, 프롬프트 글자는 그대로다. 끄면(기본) 붙지 않는다."""
    from src.retrieval.claude_listwise_reranker import RANKING_SCHEMA

    _clean_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_STRUCTURED_OUTPUT", "off")
    plain = _FakeClaude([_ranking(["3", "1", "2"])])
    ClaudeListwiseReranker(client=plain).rerank_run("흐릿한 기억의 노래", TRACKS, 3)
    assert "response_json_schema" not in plain.calls[0]["config"]

    monkeypatch.setenv("CLAUDE_STRUCTURED_OUTPUT", "json_schema")
    fake = _FakeClaude([_ranking(["3", "1", "2"])])
    run = ClaudeListwiseReranker(client=fake).rerank_run("흐릿한 기억의 노래", TRACKS, 3)
    assert run.status == RERANK_APPLIED
    assert fake.calls[0]["config"]["response_json_schema"] == RANKING_SCHEMA
    assert fake.calls[0]["contents"] == plain.calls[0]["contents"]
    assert {k: v for k, v in fake.calls[0]["config"].items() if k != "response_json_schema"} == plain.calls[0]["config"]


def test_ranking_schema_follows_structured_output_limits():
    from src.retrieval.claude_listwise_reranker import RANKING_SCHEMA

    banned = {"minimum", "maximum", "multipleOf", "minLength", "maxLength", "maxItems", "enum"}
    text = json.dumps(RANKING_SCHEMA)
    assert not any(f'"{word}"' in text for word in banned)
    for node in _walk_objects(RANKING_SCHEMA):
        assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
    # 프롬프트가 요구하는 모양과 같다 — 파서가 읽는 칸이 모두 있다.
    item = RANKING_SCHEMA["properties"]["ranking"]["items"]["properties"]
    assert set(item) == {"id", "relevance", "reason"} and "query_confidence" in RANKING_SCHEMA["properties"]
