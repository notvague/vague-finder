"""Context usage and cover performances must not trigger the album-art path."""

import json
from types import SimpleNamespace

import pytest

from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    extract_audio_evidence_text,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)


NW008 = (
    "김윤아가 어떤 청춘 드라마의 모티브가 됐다고 말했고, "
    "나중에 윤하가 불후의 명곡에서 커버해 우승한 자우림 곡이 뭐였더라?"
)
NW010 = (
    "르세라핌 뮤직비디오에서 'DO YOU THINK IM FRAGILE?'이라는 글귀로 "
    "다음 앨범을 암시했다던데, 그 뮤비의 노래가 뭐였지?"
)
NW013 = (
    "쇼미더머니에서 첫 경연에 탈락해서 원래 음원으로는 못 나올 곡인데, "
    "제작진 실수로 공개돼 버린 노래가 있다고 들었어."
)
NW017 = (
    "후렴의 영어 가사가 '엄마엄마가'처럼 들린다면서 밈이 돌았던 "
    "뉴진스 노래가 뭐였지? 제목은 기억이 안 나."
)
NW016 = (
    "MSG워너비 M.O.M이 부른 곡인데 초기 기획 때는 김종국과 KCM이 "
    "둘이 부르는 안도 있었다더라. 어떤 노래였지?"
)


@pytest.mark.parametrize("query", [
    NW008,
    NW010,
    "윤하가 방송에서 커버해 우승한 곡",
    "윤하가 방송에서 커버해도 좋은 곡",
    "다음 앨범을 예고하는 뮤비가 나온 곡",
])
def test_external_facts_do_not_require_an_image_prompt(query):
    assert not has_explicit_visual_clue(query)
    clues = [{"target": "방송", "relation": "커버", "search_query": query,
              "confidence": 0.8}]
    result = apply_modality_query_safeguards(query, {
        "context_clues": clues,
        "image_english_query": "",
        "audio_english_query": "",
        "modality_weights": {"text": 1.0, "image": 0.0, "audio": 0.0},
    })
    assert result["image_english_query"] == ""
    assert result["has_visual_clue"] is False
    assert result["modality_weights"]["image"] == 0
    assert result["context_clues"] == clues


@pytest.mark.parametrize("query", [
    "빨간 앨범 커버 노래",
    "앨범 커버 해상도가 낮았는데 배경은 분홍색이었어",
    "앨범이 파란색이고 흰 글씨가 적혀 있었어",
    "손으로 그린 꽃, 거친 종이 질감, 손글씨 제목의 앨범",
    "뮤직비디오는 못 봤지만 앨범 표지에 붉은 꽃이 그려졌어",
])
def test_actual_artwork_still_requires_an_image_prompt(query):
    assert has_explicit_visual_clue(query)
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, {
            "image_english_query": "",
            "audio_english_query": "",
        })


@pytest.mark.parametrize("query,expected_clues", [(NW008, 2), (NW010, 1)])
def test_analyzer_accepts_text_only_model_output_for_external_facts(query, expected_clues):
    from src.retrieval.query_analyzer import QueryAnalyzer

    model_output = {
        "intent_type": "mixed", "image_english_query": "", "audio_english_query": "",
        "has_visual_clue": False,
        "modality_weights": {"text": 1.0, "image": 0.0, "audio": 0.0},
        "confidence": 0.8, "context_clues": [],
    }
    analysis = QueryAnalyzer(api_key="placeholder")._parse(
        query, SimpleNamespace(text=json.dumps(model_output, ensure_ascii=False))
    )
    assert analysis.confidence == 0.8
    assert len(analysis.context_clues) == expected_clues
    assert not analysis.has_visual_clue
    assert analysis.image_english_query == ""


@pytest.mark.parametrize("query", [
    NW013,
    NW017,
    "경연에서 탈락한 뒤 음원 공개가 취소됐던 곡",
    "그 가수에게 연락해서 공개 여부를 물어본 곡",
    "수록 기록만 남은 곡",
    "게임 사운드트랙에만 실린 곡",
    "후렴의 영어 가사만 기억나는 곡",
    "힙합 경연 프로그램에서 우승한 사람이 발표한 곡",
    "배우가 출연한 작품에서 노래하는 장면에 나오는 곡",
    "오래된 밴드가 부르던 노래를 나중에 다시 불렀던 곡",
    "클래식 연주자가 출연하는 드라마에 나온 노래",
])
def test_release_history_and_lyric_text_do_not_require_an_audio_prompt(query):
    assert not has_explicit_audio_clue(query)
    result = apply_modality_query_safeguards(query, {
        "audio_english_query": "", "image_english_query": "",
        "modality_weights": {"text": 1, "image": 0, "audio": 0},
    })
    assert result["audio_english_query"] == ""
    assert result["modality_weights"]["audio"] == 0


@pytest.mark.parametrize("query", [
    "록 밴드의 기타 소리가 강했던 곡",
    "록밴드 노래",
    "락밴드 노래",
    "하드록 곡",
    "모던락 음악",
    "사운드가 거친 노래",
    "후렴에서 악기가 확 터지는 노래",
    "힙합 사운드트랙인데 빠른 비트가 있는 곡",
    "힙합 그룹의 노래",
    "힙합 경연 프로그램에서 랩 파트가 강했던 곡",
])
def test_audible_music_still_requires_an_audio_prompt(query):
    assert has_explicit_audio_clue(query)
    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        apply_modality_query_safeguards(query, {
            "audio_english_query": "", "image_english_query": "",
        })


@pytest.mark.parametrize("query,expected_clues", [(NW013, 1), (NW017, 1)])
def test_analyzer_accepts_text_only_model_output_for_context_and_lyrics(query, expected_clues):
    from src.retrieval.query_analyzer import QueryAnalyzer

    model_output = {
        "intent_type": "mixed", "image_english_query": "", "audio_english_query": "",
        "has_visual_clue": False,
        "modality_weights": {"text": 1.0, "image": 0.0, "audio": 0.0},
        "confidence": 0.8, "context_clues": [],
    }
    analysis = QueryAnalyzer(api_key="placeholder")._parse(
        query, SimpleNamespace(text=json.dumps(model_output, ensure_ascii=False))
    )
    assert analysis.confidence == 0.8
    assert len(analysis.context_clues) == expected_clues
    assert analysis.audio_english_query == ""
    assert analysis.modality_weights.audio == 0


@pytest.mark.parametrize("query", [
    "힙합 경연 프로그램에서 우승한 사람이 발표한 곡",
    "배우가 출연한 작품에서 노래하는 장면에 나오는 곡",
    "오래된 밴드가 부르던 노래를 나중에 다시 불렀던 곡",
])
def test_analyzer_does_not_reinterpret_background_history_as_song_audio(query):
    from src.retrieval.query_analyzer import QueryAnalyzer

    model_output = {
        "intent_type": "mixed", "image_english_query": "", "audio_english_query": "",
        "modality_weights": {"text": 1, "image": 0, "audio": 0},
        "confidence": 0.8,
        # The model might infer a vocal role from a stage scene; discard it.
        "performance_clues": {"vocal_count": "solo", "confidence": 0.7},
    }
    result = QueryAnalyzer(api_key="placeholder")._parse(
        query, SimpleNamespace(text=json.dumps(model_output, ensure_ascii=False))
    )
    assert result.confidence == 0.8
    assert result.audio_english_query == ""
    assert result.modality_weights.audio == 0


def test_actual_vocal_count_still_requires_audio_prompt_in_analyzer():
    from src.retrieval.query_analyzer import QueryAnalyzer

    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        QueryAnalyzer(api_key="placeholder")._parse(
            "여자 둘이 함께 부르는 노래",
            SimpleNamespace(text=json.dumps({
                "intent_type": "mixed", "image_english_query": "",
                "audio_english_query": "", "confidence": 0.8,
            }, ensure_ascii=False)),
        )


@pytest.mark.parametrize("query", [
    NW016,
    "초기 기획 당시에는 두 사람이 듀엣할 예정이었다는 노래",
    "당초 계획 때 두 사람이 듀엣으로 부르는 안이 있었지만 발표 때 바뀌었어",
    "기획 단계에서 남녀가 듀엣으로 부르는 안도 나왔던 노래",
])
def test_unrealized_performance_is_not_audible_evidence(query):
    assert "듀엣" not in extract_audio_evidence_text(query)
    assert "둘이 부르" not in extract_audio_evidence_text(query)
    assert not has_explicit_audio_clue(query)
    result = apply_modality_query_safeguards(query, {
        "audio_english_query": "", "image_english_query": "",
        "modality_weights": {"text": 1, "image": 0, "audio": 0},
    })
    assert result["modality_weights"]["audio"] == 0


def test_nw016_model_output_keeps_production_context_without_inventing_duet():
    from src.retrieval.query_analyzer import QueryAnalyzer

    model_output = {
        "intent_type": "mixed", "image_english_query": "", "audio_english_query": "",
        "modality_weights": {"text": 1, "image": 0, "audio": 0},
        "confidence": 0.8, "context_clues": [],
        # A model may incorrectly infer a duet from the abandoned plan.
        "performance_clues": {"vocal_count": "duet", "confidence": 0.8},
    }
    analysis = QueryAnalyzer(api_key="placeholder")._parse(
        NW016, SimpleNamespace(text=json.dumps(model_output, ensure_ascii=False))
    )
    assert analysis.original_query == NW016
    assert analysis.confidence == 0.8
    assert len(analysis.context_clues) == 1
    assert "M.O.M" in analysis.context_clues[0].search_query
    assert "초기 기획" in analysis.context_clues[0].search_query
    assert analysis.performance_clues.vocal_count is None
    assert analysis.audio_english_query == ""
    assert analysis.modality_weights.audio == 0


@pytest.mark.parametrize("query", [
    "초기 기획 때는 둘이 부르는 안도 있었지만 실제 음원은 피아노 반주에 여자 보컬이었어",
    "기획 당시에는 피아노로 연주할 예정이었다고 하지만 실제 음원에서는 기타 소리가 들려",
])
def test_actual_recording_auditory_clues_survive_unrealized_plan(query):
    audible = extract_audio_evidence_text(query)
    assert "실제" in audible
    assert has_explicit_audio_clue(query)
    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        apply_modality_query_safeguards(query, {
            "audio_english_query": "", "image_english_query": "",
        })


def test_released_duet_without_planning_context_still_requires_audio_prompt():
    query = "실제 음원에서 여자 둘이 함께 부르는 듀엣곡"
    assert "둘이" in extract_audio_evidence_text(query)
    from src.retrieval.query_analyzer import QueryAnalyzer

    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        QueryAnalyzer(api_key="placeholder")._parse(
            query, SimpleNamespace(text=json.dumps({
                "intent_type": "mixed", "image_english_query": "",
                "audio_english_query": "", "confidence": 0.8,
            }, ensure_ascii=False)),
        )
