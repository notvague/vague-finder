"""
tests/test_comment_selection.py

LLM 댓글 선별(llm_utils.select_emotional_comments_with_llm)의 회귀 테스트.

배경 (2026-09-18 점검): 선별 결과가 비면 '안전 fallback'이 판정 목록에서 좋아요순으로
댓글을 되살렸다. keep=false, score=0으로 버린 댓글이 최종 결과에 들어갔다.

이제 품질 기준에 맞는 댓글이 없으면 빈 목록이다. API 실패로 판정하지 못한 댓글은
따로 세고, 판정된 것처럼 돌려주지 않는다.

Gemini는 가짜다. 댓글 문장은 지어낸 것이다.

실행:
    venv/bin/python -m pytest tests/test_comment_selection.py -v
"""
from __future__ import annotations

import json
import types

import pytest

from src.crawler.scripts_py import llm_utils


def comment(text, likes=0):
    return {"text": text, "like_count": likes}


@pytest.fixture
def fake_gemini(monkeypatch):
    """배치마다 verdicts(list) 또는 예외를 순서대로 돌려준다."""
    monkeypatch.setattr(llm_utils, "GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(llm_utils.time, "sleep", lambda s: None)
    calls = []

    def install(responses):
        queue = list(responses)

        class _Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return types.SimpleNamespace(text=json.dumps(item, ensure_ascii=False))

        class _Client:
            def __init__(self, api_key=None):
                self.models = _Models()

        monkeypatch.setattr(llm_utils.genai, "Client", _Client)
        return calls
    return install


def verdict(index, keep, score):
    return {"index": index, "keep": keep, "score": score, "reason": "", "aspects": []}


def test_rejected_comments_never_come_back(fake_gemini) -> None:
    fake_gemini([[verdict(0, False, 0), verdict(1, False, 1)]])
    result = llm_utils.select_emotional_comments_detailed(
        [comment("보러 옴", 900), comment("2024년에 듣는 사람", 800)], target_count=5, batch_size=10,
    )
    assert result.texts == []
    assert (result.evaluated, result.unevaluated) == (2, 0)


def test_api_failure_does_not_pass_comments_through(fake_gemini) -> None:
    """판정을 못 한 것은 통과한 것이 아니다. 예전에는 raw fallback으로 원본이 그대로 들어갔다."""
    fake_gemini([RuntimeError("503")] * (llm_utils.BATCH_RETRIES + 1))
    result = llm_utils.select_emotional_comments_detailed(
        [comment("아무 댓글", 900), comment("다른 댓글", 1)], target_count=5, batch_size=10,
    )
    assert result.texts == []
    assert (result.evaluated, result.unevaluated, result.failed_batches) == (0, 2, 1)
    assert result.api_failed_entirely is True


def test_partial_failure_keeps_only_evaluated_results(fake_gemini) -> None:
    fake_gemini([
        [verdict(0, True, 5)],
        *([RuntimeError("timeout")] * (llm_utils.BATCH_RETRIES + 1)),
    ])
    result = llm_utils.select_emotional_comments_detailed(
        [comment("비 오는 새벽에 창가에서 듣기 좋은 노래", 10), comment("평가 못 한 댓글", 5)],
        target_count=5, batch_size=1,
    )
    assert result.texts == ["비 오는 새벽에 창가에서 듣기 좋은 노래"]
    assert (result.evaluated, result.unevaluated) == (1, 1)


def test_flaky_batch_is_retried(fake_gemini) -> None:
    calls = fake_gemini([RuntimeError("once"), [verdict(0, True, 4)]])
    result = llm_utils.select_emotional_comments_with_llm([comment("겨울밤 공기 같은 노래", 1)], target_count=5)
    assert result == ["겨울밤 공기 같은 노래"]
    assert len(calls) == 2


def test_score_fallback_still_excludes_low_scores(fake_gemini) -> None:
    """keep=false여도 score 3 이상은 부족할 때 채우고, 0~1은 절대 넣지 않는다."""
    # 배치는 좋아요순으로 정렬된 뒤 만들어지므로 index는 정렬 후 순서다
    fake_gemini([[verdict(0, False, 3), verdict(1, False, 1), verdict(2, True, 5)]])
    result = llm_utils.select_emotional_comments_with_llm(
        [comment("혼자 걷는 밤에 어울린다", 900), comment("최고다", 5), comment("이별 후 밤에 듣는 곡", 1)],
        target_count=5, batch_size=10,
    )
    assert result == ["이별 후 밤에 듣는 곡", "혼자 걷는 밤에 어울린다"]


def test_non_list_response_counts_as_a_failed_batch(fake_gemini) -> None:
    fake_gemini([{"keep": True}] * (llm_utils.BATCH_RETRIES + 1))
    result = llm_utils.select_emotional_comments_detailed([comment("아무 댓글", 1)], target_count=5)
    assert result.texts == [] and result.unevaluated == 1


def test_empty_input_short_circuits(fake_gemini) -> None:
    calls = fake_gemini([])
    assert llm_utils.select_emotional_comments_with_llm([]) == []
    assert calls == []


# --- 판정 불가는 예외로 올라간다 -----------------------------------------------------------
# 호출부(filter_comments / filter_melon_comments)는 얇은 래퍼만 쓴다. 래퍼가 빈 목록을
# 돌려주면 '쓸 댓글이 없는 곡'과 구분되지 않아, 몇 분짜리 쿼터 초과가 곡을 영구히 그 상태로
# 굳힌다. 래퍼가 올리는 예외를 여기서 고정한다.

def test_wrapper_raises_when_nothing_could_be_evaluated(fake_gemini) -> None:
    fake_gemini([RuntimeError("503")] * (llm_utils.BATCH_RETRIES + 1))
    with pytest.raises(llm_utils.CommentSelectionUnavailable) as info:
        llm_utils.select_emotional_comments_with_llm(
            [comment("아무 댓글", 900), comment("다른 댓글", 1)], target_count=5, batch_size=10,
        )
    assert info.value.candidates == 2 and info.value.failed_batches == 1


def test_wrapper_returns_an_empty_list_when_the_llm_actually_judged(fake_gemini) -> None:
    """판정을 했고 쓸 만한 댓글이 없는 경우는 예외가 아니다. 빈 목록이 정답이다."""
    fake_gemini([[verdict(0, False, 0)]])
    assert llm_utils.select_emotional_comments_with_llm([comment("보러 옴", 900)], target_count=5) == []


def test_wrapper_keeps_evaluated_results_despite_a_failed_batch(fake_gemini) -> None:
    fake_gemini([
        [verdict(0, True, 5)],
        *([RuntimeError("timeout")] * (llm_utils.BATCH_RETRIES + 1)),
    ])
    assert llm_utils.select_emotional_comments_with_llm(
        [comment("비 오는 새벽에 창가에서 듣기 좋은 노래", 10), comment("평가 못 한 댓글", 5)],
        target_count=5, batch_size=1,
    ) == ["비 오는 새벽에 창가에서 듣기 좋은 노래"]


def test_melon_filter_propagates_the_outage(fake_gemini) -> None:
    """collect_melon_data.fetch_melon_comments가 이 예외를 삼키면 '댓글 0개' 곡이 저장된다."""
    from src.crawler.scripts_py import collect_melon_data as cmd
    fake_gemini([RuntimeError("429")] * (llm_utils.BATCH_RETRIES + 1))
    with pytest.raises(llm_utils.CommentSelectionUnavailable):
        cmd.filter_melon_comments([{"AUTH_CNTTS": "새벽에 혼자 듣기 좋은 노래예요", "RECM_CNT": 10}])


def test_youtube_filter_propagates_the_outage(fake_gemini) -> None:
    from src.crawler.scripts_py import collect_reaction as cr
    fake_gemini([RuntimeError("429")] * (llm_utils.BATCH_RETRIES + 1))
    with pytest.raises(llm_utils.CommentSelectionUnavailable):
        cr.filter_comments([{"text": "새벽에 혼자 듣기 좋은 노래예요", "like_count": 10}])
