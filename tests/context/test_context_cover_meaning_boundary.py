"""Cover renditions and artwork use independent, source-bound evidence."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.namuwiki.query_boundary_matrix import boundary_cases
from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.modality_queries import (
    ModalityQueryValidationError, apply_modality_query_safeguards,
    fallback_modality_payload, has_explicit_artwork_reference,
    has_explicit_audio_clue, has_explicit_visual_clue,
)
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback


def model_payload(*, audio="", image="", context=None):
    return dict(intent_type="mixed", confidence=0.89,
                context_clues=[] if context is None else context,
                audio_english_query=audio, image_english_query=image,
                modality_weights=dict(text=0.6, image=0.2 if image else 0.0,
                                      audio=0.2 if audio else 0.0))


@pytest.mark.parametrize("case", boundary_cases(), ids=lambda case: case.name)
def test_crossed_context_artwork_and_heard_sound(case):
    raw = model_payload(audio=case.audio, image=case.image)
    before = deepcopy(raw)
    result = QueryAnalyzer(api_key="synthetic-test-key")._parse(
        case.query, SimpleNamespace(text=json.dumps(raw)),
    )
    assert raw == before and result.original_query == case.query
    assert result.confidence == 0.89
    assert bool(result.context_clues) is case.context
    assert bool(result.image_english_query) is bool(case.image)
    assert bool(result.audio_english_query) is bool(case.audio)
    assert result.has_visual_clue is bool(case.image)
    assert len(result.context_clues) <= 4
    weights = result.modality_weights.model_dump()
    assert all(0 <= value <= 1 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert (weights["image"] > 0) is bool(case.image)
    assert (weights["audio"] > 0) is bool(case.audio)


@pytest.mark.parametrize("rendition", [
    "커버가 화제가 돼", "커버가 화제가 됐어", "커버는 화제가 된 뒤",
    "커버를 불렀대", "커버를 불러", "커버를 해", "커버를 했던",
    "커버 해서", "커버 했던", "커버한", "커버된", "커버를 공개했대",
    "커버 영상이 공개된 뒤", "커버가 차트에서 다시 올라오고 있다는",
    "커버가 역주행 계기가 됐다는", "커버 버전으로 다시 알려진",
])
def test_bare_cover_morphology_is_a_rendition_not_an_album_picture(rendition):
    query = f"어떤 가수의 {rendition} 노래를 찾고 있어."
    assert not has_explicit_artwork_reference(query)
    assert not has_explicit_visual_clue(query)
    assert not has_explicit_audio_clue(query)
    clues = apply_context_query_safeguards(query, model_payload())["context_clues"]
    assert clues and all(clue["target"] == "" for clue in clues)


@pytest.mark.parametrize("query", [
    "커버가 파란색이었던 곡을 찾아줘.",
    "커버가 빨갛고 중앙에 꽃이 그려져 있었어.",
    "커버는 검은 바탕에 흰 글씨였어.",
    "커버에 여자 얼굴 사진이 있었어.",
    "커버 중앙에 남자의 실루엣이 보였어.",
    "커버 색깔은 노란색이었어.",
    "빨간 커버였던 곡이 기억나.",
    "손글씨가 적힌 커버였어.",
    "커버 이미지가 유명했던 노래야.",
    "커버 아트가 유명했던 곡이야.",
    "커버 디자인이 화제가 됐던 노래야.",
    "앨범의 커버가 화제가 됐어.",
])
def test_actual_cover_appearance_and_explicit_artwork_still_require_image(query):
    assert has_explicit_visual_clue(query)
    assert apply_context_query_safeguards(query, model_payload())["context_clues"] == []
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, model_payload())


@pytest.mark.parametrize("query", [
    "커버였던 것만 기억나. 무슨 노래지?",
    "누구 커버인지 잘 모르겠어.",
    "커버를 찾아줘.",
    "커버가 화제라는 말이 가사에 있었어.",
    "가사에 커버가 화제가 됐다는 말이 있어.",
    "제목에 커버가 화제가 됐다는 문장이 들어 있어.",
    "커버해 보려고 노래를 고르고 있어.",
    "커버를 공개하려고 곡을 찾고 있어.",
    "커버를 불러 볼 노래를 추천해줘.",
    "커버할 노래를 추천해줘.",
])
def test_ambiguous_literal_or_future_cover_does_not_invent_context_or_art(query):
    assert not has_explicit_visual_clue(query)
    result = apply_context_query_safeguards(query, model_payload())
    assert result["context_clues"] == []


@pytest.mark.parametrize("audio", [
    "Male background vocals.", "A violin melody in the foreground.",
    "Soft piano in the background with vocals in the foreground.",
])
def test_rendition_event_keeps_legitimate_audio_spatial_wording(audio):
    query = "가수의 커버가 화제가 돼 역주행한 곡이야. 피아노와 바이올린 소리에 남자 보컬이 들렸어."
    result = apply_modality_query_safeguards(query, model_payload(audio=audio))
    assert result["audio_english_query"] and not result["image_english_query"]


@pytest.mark.parametrize("audio", [
    "Male background vocals behind a red cover image.",
    "Piano in the background beside a drawing of flowers.",
    "Violin in the foreground of a black and white photograph.",
])
def test_cover_performance_does_not_whitelist_image_content_in_audio(audio):
    query = "가수의 커버가 화제가 돼 역주행한 곡이야. 남자 보컬에 피아노와 바이올린 반주가 들렸어."
    with pytest.raises(ModalityQueryValidationError, match="visual-only wording"):
        apply_modality_query_safeguards(query, model_payload(audio=audio))


@pytest.mark.parametrize("async_call", [False, True])
def test_rendition_with_sound_passes_real_analyzer_first_attempt(async_call):
    query = "어떤 가수의 커버가 화제가 돼 차트에서 다시 올라온 노래야. 오르골과 바이올린 소리가 들리는 발라드야."
    raw = model_payload(audio="A music box melody and violin accompaniment in a ballad.")
    response = SimpleNamespace(text=json.dumps(raw))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return response

    async def generate_async(**kwargs):
        return generate(**kwargs)

    analyzer = QueryAnalyzer(api_key="synthetic-test-key")
    analyzer._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate),
                                      aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_async)))
    result = asyncio.run(analyzer.analyze_async(query)) if async_call else analyzer.analyze(query)
    assert len(calls) == 1 and result.confidence == 0.89
    assert result.original_query == query and result.context_clues
    assert result.audio_english_query and not result.image_english_query


def test_fallback_keeps_source_bound_context_and_sound_when_model_is_unavailable():
    query = "어떤 가수의 커버가 화제가 돼 역주행한 노래야. 피아노와 바이올린 소리가 들렸어."
    result = rule_fallback(query)
    assert result.context_clues and result.audio_english_query
    assert result.original_query == query and not result.image_english_query
    assert not fallback_modality_payload(query)["image_english_query"]


def test_model_cannot_replace_the_rendition_with_an_invented_work_or_answer():
    query = "다른 가수의 커버가 화제가 돼 역주행한 노래야."
    invented = [dict(target="없는작품", relation="OST", search_query="없는작품 OST 가상정답", confidence=0.99)]
    clues = apply_context_query_safeguards(query, model_payload(context=invented))["context_clues"]
    assert clues and "가상정답" not in str(clues) and "없는작품" not in str(clues)


@pytest.mark.parametrize("appearance", [
    "커버가 어둡고 차가운 느낌이었어", "커버는 밝은 분위기였어",
    "커버가 몽환적인 분위기였어", "어두운 커버였어", "따뜻한 커버였어",
    "미니멀한 커버였어", "화려한 커버였어",
])
def test_cover_appearance_mood_keeps_the_real_image_route(appearance):
    query = appearance + ", 노래는 현악기가 깔린 남자 발라드였어."
    image = "A dark, moody cover with a cool color palette."
    audio = "A male vocal ballad with string accompaniment."
    result = QueryAnalyzer(api_key="synthetic-test-key")._parse(
        query, SimpleNamespace(text=json.dumps(model_payload(image=image, audio=audio))),
    )
    assert result.image_english_query == image
    assert result.audio_english_query == audio
    assert result.modality_weights.image > 0 and not result.context_clues


@pytest.mark.parametrize("query", [
    "커버가 차가운 음색이었어.", "커버는 따뜻한 소리였어.",
    "커버가 어둡고 차가운 보컬 느낌이었어.",
    "커버가 밝은 멜로디와 사운드였어.",
])
def test_sound_descriptors_do_not_make_ambiguous_cover_into_artwork(query):
    assert not has_explicit_visual_clue(query)


@pytest.mark.parametrize("event", [
    "가수의 커버가 화제가 돼 역주행한 곡인데",
    "가수의 커버가 화제가 된 노래이고",
    "커버 영상이 공개된 곡인데",
    "커버 대회에서 우승한 곡인데",
])
@pytest.mark.parametrize("art", [
    "앨범 커버에는 검은 배경에 꽃 그림이 있었어",
    "커버에 흰 글씨와 빨간 꽃이 그려져 있었어",
])
def test_repeated_cover_senses_in_one_clause_keep_both_paths_without_art_in_context(event, art):
    query = f"{event} {art}. 오르골과 바이올린 소리가 들렸어."
    raw = model_payload(image="Cover artwork with flowers and white lettering.",
                        audio="Music box and violin accompaniment.")
    result = QueryAnalyzer(api_key="synthetic-test-key")._parse(
        query, SimpleNamespace(text=json.dumps(raw)),
    )
    assert result.context_clues and result.image_english_query and result.audio_english_query
    assert result.original_query == query
    assert all("꽃" not in clue.search_query and "글씨" not in clue.search_query
               and "오르골" not in clue.search_query for clue in result.context_clues)


@pytest.mark.parametrize("query", [
    "아마 어떤 가수의 커버가 화제가 돼 역주행한 곡이야.",
    "어떤 가수의 커버가 화제가 돼 역주행한 곡이었던 것 같아.",
    "어떤 가수의 커버가 화제가 돼 역주행한 곡인지 모르겠어.",
])
def test_uncertain_rendition_keeps_conservative_source_confidence(query):
    result = apply_context_query_safeguards(query, model_payload())
    assert result["context_clues"]
    assert all(clue["confidence"] <= 0.6 for clue in result["context_clues"])
