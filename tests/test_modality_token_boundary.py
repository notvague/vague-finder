"""한 글자 낱말(록·락·별·달·해·비)을 다른 낱말의 조각에서 읽지 않는다.

조각을 단서로 읽으면 규칙이 "소리(또는 표지) 단서가 있다"고 판정하고, 모델이 내지 않은
프롬프트를 요구하다가 분석 전체가 규칙 폴백으로 떨어진다(results_v38: t009·t071·t086·t099,
v09 n002). 여기서는 판정만 본다 — 모델 호출 없음.
"""
from __future__ import annotations

import pytest

from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    build_fallback_modality_queries,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)

_NO_PROMPTS = {"image_english_query": "", "audio_english_query": "", "modality_weights": {"text": 1.0, "image": 0.0, "audio": 0.0}}


@pytest.mark.parametrize("query", [
    "FIFA 07 축구게임에서 한국어 곡 최초로 수록된 노래",
    "초록색 물감을 붓으로 마구 덧칠한 초록 숲 이미지",
    "쨍한 빨간 드레스 입은 여자가 초록 풀밭에 엎드려 턱 괴고 있는 이미지",
    "새끼손가락에 반지를 낀 손이 나오는 이미지",
])
def test_rock_inside_another_word_is_not_an_audio_clue(query: str) -> None:
    assert not has_explicit_audio_clue(query)
    # 모델이 소리 프롬프트를 내지 않아도 검증을 통과한다(예전에는 여기서 예외 → 3회 뒤 폴백).
    out = apply_modality_query_safeguards(query, dict(_NO_PROMPTS))
    assert out["audio_english_query"] == ""
    # 폴백으로 가더라도 "rock style" 프롬프트를 만들지 않는다.
    assert "rock" not in build_fallback_modality_queries(query)[1]


@pytest.mark.parametrize("query", [
    "90년대 초반 록밴드 노래",
    "펑크록처럼 일렉기타가 거칠게 긁히는 곡",
    "하드록 느낌으로 내지르는 노래",
    "마지막에 터지듯 내지르는 록발라드야",
    "팝록인데 후렴이 시원해",
    "신나는 락 음악",
    "모던 록 밴드 노래",
])
def test_rock_as_a_genre_word_still_counts(query: str) -> None:
    assert has_explicit_audio_clue(query)
    with pytest.raises(ModalityQueryValidationError):
        apply_modality_query_safeguards(query, dict(_NO_PROMPTS))


@pytest.mark.parametrize("query", [
    "작곡가가 객원 보컬 데려와서 앨범 내는 프로젝트, 거기서 남자가 부른 이별 노래",
    "원래 보컬이 앨범 준비하다가 사고로 세상을 떠났다는 사연",
    "그 앨범에서 제일 유명해",
    "앨범마다 분위기가 달라지는 가수",
    "특별한 날에 듣던 앨범",
    "앨범은 별로 안 떴는데 이 노래만 유명했어",
])
def test_syllable_inside_another_word_is_not_a_cover_clue(query: str) -> None:
    assert not has_explicit_visual_clue(query)


@pytest.mark.parametrize("query", [
    "별이 잔뜩 그려진 앨범",
    "노란 달이 크게 떠 있는 앨범",
    "해가 지는 바닷가 사진이 있는 앨범",
    "비가 내리는 창문 그림이었던 앨범",
    "별무늬가 박힌 앨범",
    "분홍빛 배경에 흑백 얼굴 사진이 있던 앨범",
])
def test_one_syllable_noun_as_a_word_still_counts(query: str) -> None:
    assert has_explicit_visual_clue(query)


def test_colour_word_keeps_cover_query_free_of_audio_clue() -> None:
    query = "앨범 커버가 초록색 바탕이었고 흰색 기하학 도형 네 개가 크게 배치돼 있었어"
    assert has_explicit_visual_clue(query)
    assert not has_explicit_audio_clue(query)
