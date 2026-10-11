"""'표지'라는 말 없이 "…한 이미지"로 맺은 표지 묘사를 표지 단서로 읽는다.

못 읽으면 모델이 낸 표지 프롬프트를 규칙이 지워 표지 경로가 꺼진다(results_v38 t086·t099).
'이미지'는 인상("청순한 이미지의 가수")도 뜻하므로 그 쓰임은 읽지 않아야 한다 — 읽으면 규칙이
모델에게 없는 표지 프롬프트를 요구하다가 분석 전체가 폴백으로 떨어진다. 판정만 본다(모델 호출 없음).
"""
from __future__ import annotations

import pytest

from src.retrieval.modality_queries import (
    apply_modality_query_safeguards,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)


@pytest.mark.parametrize("query", [
    "쨍한 빨간 드레스 입은 여자가 초록 풀밭에 엎드려 턱 괴고 있는 이미지",
    "초록색 물감을 붓으로 마구 덧칠한 초록 숲 이미지",
    "파란 바다 위에 하얀 배가 떠 있는 이미지였어",
    "검은 배경에 노란 꽃 한 송이가 있는 이미지인데 노래는 잔잔해",
    "분홍 구름 사이로 여자가 누워 있는 이미지야",
    "노란 우산 쓴 남자가 서 있는 이미지 같아",
])
def test_description_ending_on_image_noun_is_a_cover_clue(query: str) -> None:
    assert has_explicit_visual_clue(query)


@pytest.mark.parametrize("query", [
    # 인상을 뜻하는 '이미지' — 뒤에 꾸밈받는 명사가 오거나 조사가 붙는다.
    "청순한 이미지의 여자 가수가 부른 발라드",
    "이미지가 좋은 남자 배우가 부른 노래",
    "빨간 머리 여자 이미지로 유명한 가수",
    "청순한 이미지였던 여자 가수가 부른 빨간 맛 같은 노래",
    "이미지 변신한 걸그룹의 파란색 콘셉트 노래",
    "노래가 주는 이미지는 파란 바다 같아",
    # 맺음꼴이어도 다른 시각 축이 하나뿐이면 받지 않는다.
    "센 이미지였어, 여자 래퍼",
    "청순한 이미지였어 여자 솔로였고",
    # 떠올린 심상(v06 q215).
    "이 노래 들으면 생각나는 이미지가 검은색?",
    # 재료 낱말이 비유로 쓰인 경우.
    "물감처럼 번지는 멜로디의 노래",
])
def test_image_as_impression_is_not_a_cover_clue(query: str) -> None:
    assert not has_explicit_visual_clue(query)


def test_cover_only_description_keeps_the_models_image_prompt() -> None:
    query = "쨍한 빨간 드레스 입은 여자가 초록 풀밭에 엎드려 턱 괴고 있는 이미지"
    raw = {
        "image_english_query": "Album cover showing a woman in a red dress lying on green grass.",
        "audio_english_query": "",
        "modality_weights": {"text": 0.3, "image": 0.7, "audio": 0.0},
    }
    out = apply_modality_query_safeguards(query, raw)
    assert out["image_english_query"].startswith("Album cover")
    assert out["has_visual_clue"] is True
    assert out["modality_weights"]["image"] > 0
    # "초록"은 소리 단서가 아니다(#42) — 둘이 함께여야 이 질의가 표지 경로로 간다.
    assert not has_explicit_audio_clue(query)
