"""재시도 정정 지시는 받은 응답이 쓸 수 없을 때만 붙는다.

504·429·시간 초과처럼 응답이 없던 실패 뒤 재시도는 같은 프롬프트를 다시 보낸다. 전에는 실패 종류와 상관없이
"Your previous output was invalid" 지시를 붙여, 같은 질의의 재시도 입력이 달라졌다(v73 실제 실행에서 확인).
"""
import json

import pytest

from src.backend.schemas.search import MatchingTrack
from src.retrieval import gemini_listwise_reranker as mod
from src.retrieval.gemini_listwise_reranker import (
    RERANK_APPLIED,
    RERANK_FAILED,
    RETRY_CORRECTION,
    GeminiListwiseReranker,
    GeminiListwiseRerankerConfig,
)


def _track(song_id: str, score: float) -> MatchingTrack:
    return MatchingTrack(id=song_id, title=f"곡{song_id}", artist="artist", score=score, retrieval_score=score)


TRACKS = [_track("a", 1.0), _track("b", 0.8), _track("c", 0.6)]


class _ServerError(Exception):
    """SDK의 504 DEADLINE_EXCEEDED를 흉내 낸다(리랭커는 예외 종류로 가르지 않는다)."""


def _ranking(contents: str) -> str:
    tail = contents.split("Candidates (JSON):\n", 1)[1].split("\n\nReturn one JSON", 1)[0]
    ids = [str(c["id"]) for c in json.loads(tail)]
    return json.dumps({"query_confidence": 0.9,
                       "ranking": [{"id": i, "relevance": 1.0 - 0.1 * k} for k, i in enumerate(reversed(ids))]})


class _ScriptedModels:
    """호출마다 정해 둔 동작을 한다: "error"(응답 없음) · "timeout"(응답 없음) · "invalid"(JSON 아님) · "ok"."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents)
        step = self.script.pop(0)
        if step == "error":
            raise _ServerError("504 DEADLINE_EXCEEDED")
        if step == "timeout":
            raise TimeoutError()
        text = "not json" if step == "invalid" else _ranking(contents)
        return type("Response", (), {"text": text})()


def _reranker(models, *, max_retries=3):
    client = type("Client", (), {"models": models})()
    return GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(passes=1, use_search_grounding=False, rare_fact_verification=False,
                                            max_retries=max_retries, time_budget_seconds=30.0),
        client=client,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_retry_after_call_failure_resends_the_same_prompt(failure):
    models = _ScriptedModels([failure, "ok"])
    run = _reranker(models).rerank_run("잔잔한 발라드", TRACKS, 3)
    assert run.status == RERANK_APPLIED
    assert len(models.prompts) == 2
    assert models.prompts[1] == models.prompts[0], "응답이 없던 실패 뒤 재시도 입력이 바뀌면 안 된다"
    assert RETRY_CORRECTION not in models.prompts[1]


def test_retry_after_invalid_output_adds_the_correction():
    models = _ScriptedModels(["invalid", "ok"])
    run = _reranker(models).rerank_run("잔잔한 발라드", TRACKS, 3)
    assert run.status == RERANK_APPLIED
    assert models.prompts[1] == models.prompts[0] + RETRY_CORRECTION


def test_correction_from_an_invalid_output_survives_a_later_call_failure():
    """무효 응답 → 504 → 성공: 마지막으로 받은 응답이 무효였으므로 정정 지시는 그대로 남는다."""
    models = _ScriptedModels(["invalid", "error", "ok"])
    run = _reranker(models).rerank_run("잔잔한 발라드", TRACKS, 3)
    assert run.status == RERANK_APPLIED
    base = models.prompts[0]
    assert models.prompts[1:] == [base + RETRY_CORRECTION, base + RETRY_CORRECTION]


def test_all_call_failures_fall_back_without_any_correction():
    models = _ScriptedModels(["error", "timeout", "error"])
    run = _reranker(models).rerank_run("잔잔한 발라드", TRACKS, 3)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == ["a", "b", "c"]
    assert len(set(models.prompts)) == 1 and RETRY_CORRECTION not in models.prompts[0]
