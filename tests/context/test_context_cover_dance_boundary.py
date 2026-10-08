"""Cover choreography is Context evidence; heard sound and artwork stay separate."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    extract_audio_evidence_text,
    fallback_modality_payload,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)


RADIO_QUERY = (
    "김가영 기상캐스터가 굿모닝FM 장성규입니다 보이는 라디오에서 "
    "크리스마스 이브에 커버댄스한 방탄소년단 곡은? "
    "펑키한 베이스 소리가 있는 노래였어."
)


def model_payload(*, audio="", image=""):
    return {
        "intent_type": "mixed", "confidence": 0.8,
        "image_english_query": image, "audio_english_query": audio,
        "has_visual_clue": bool(image),
        "modality_weights": {"text": 0.7, "image": 0.0, "audio": 0.3}
        if audio else {"text": 1.0, "image": 0.0, "audio": 0.0},
        "context_clues": [],
    }


@pytest.mark.parametrize("performance", [
    "커버댄스", "커버 댄스", "커버\t댄스", "커버\n댄스", "커버댄싱",
    "커버 춤", "커버 안무", "댄스커버", "댄스 커버", "댄싱 커버",
    "춤 커버", "안무 커버",
])
def test_choreography_history_is_not_artwork_or_recording_genre(performance):
    query = f"방송에서 {performance}한 노래가 뭐였지?"
    assert not has_explicit_visual_clue(query)
    assert not has_explicit_audio_clue(query)
    result = apply_modality_query_safeguards(query, model_payload())
    assert not result["has_visual_clue"]
    assert result["image_english_query"] == result["audio_english_query"] == ""
    assert result["modality_weights"] == {"text": 1.0, "image": 0.0, "audio": 0.0}


@pytest.mark.parametrize("performance", [
    "커버댄스", "커버 댄스", "커버댄싱", "커버 춤", "커버 안무",
    "댄스커버", "댄스 커버", "댄싱 커버", "춤 커버", "안무 커버",
])
def test_performed_choreography_survives_context_safeguards(performance):
    from src.retrieval.context_query import apply_context_query_safeguards

    query = f"누가 방송에서 {performance}한 곡이 뭐였지?"
    result = apply_context_query_safeguards(query, model_payload())
    assert len(result["context_clues"]) == 1
    assert result["context_clues"][0]["relation"] == "방송·공연 일화"
    assert performance in result["context_clues"][0]["search_query"]
    assert result["context_clues"][0]["target"] == ""


@pytest.mark.parametrize("query", [
    "커버댄스하기 좋은 곡을 찾아줘",
    "라디오에서 커버댄스하려고 해. 어떤 곡이 좋을까?",
    "댄스 커버할 곡을 고르고 싶어",
    "커버댄스한 사람의 사진이 앨범 표지에 인쇄되어 있었어",
    "가사에 커버댄스한 모습을 이야기하는 노래",
    "앨범 커버에 댄스 커버 사진이 있는 노래",
])
def test_planned_activity_artwork_and_lyrics_do_not_invent_context(query):
    from src.retrieval.context_query import apply_context_query_safeguards

    assert apply_context_query_safeguards(query, model_payload())["context_clues"] == []


@pytest.mark.parametrize("query", [
    "커버댄스 영상에서 파란 배경에 얼굴 사진이 보이는 곡",
    "댄스 커버에서 분홍빛 배경 중앙에 사람의 실루엣이 보였어",
    "안무 커버에서 검은 배경 왼쪽에 빨간 글씨가 적혀 있었어",
])
def test_performance_video_description_does_not_become_album_art(query):
    assert not has_explicit_visual_clue(query)
    result = apply_modality_query_safeguards(query, model_payload(image="A blue stage."))
    assert result["image_english_query"] == ""
    assert result["modality_weights"]["image"] == 0


@pytest.mark.parametrize("query", [
    "커버댄스한 곡인데 앨범 커버에는 빨간 꽃이 있었어",
    "댄스 커버는 못 봤지만 앨범 표지에 파란 달이 그려졌어",
    "커버댄스와 댄스 커버 영상이 있던 곡. 커버에는 흰 글씨가 적혀 있었어",
    "앨범 커버에 댄스 커버 사진이 인쇄되어 있었어",
    "댄스 커버 사진을 찍었다고 해. 앨범에는 흰 글씨가 적혀 있었어",
    "빨간 앨범 커버 노래",
    "앨범 커버 해상도가 낮고 배경이 분홍색인 곡",
    "앨범 표지 중앙에 피아노 그림이 있었어",
])
def test_independent_real_artwork_still_requires_its_image_prompt(query):
    assert has_explicit_visual_clue(query)
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, model_payload())
    result = apply_modality_query_safeguards(
        query, model_payload(image="Album artwork with printed illustration and lettering.")
    )
    assert result["has_visual_clue"]
    assert result["modality_weights"]["image"] > 0


@pytest.mark.parametrize("query", [
    RADIO_QUERY,
    "댄스 커버했던 곡인데 피아노 반주가 있고 목소리가 허스키했어",
    "커버댄스한 곡인데 느린 템포와 베이스 소리가 기억나",
    "앨범 커버에 커버댄스 사진이 있었어. 노래에서는 드럼 소리가 크게 났어",
    "커버댄스했지만 표지는 기억 안 나. 피아노 소리가 났어",
    "빠른 비트가 들리는 댄스곡",
])
def test_heard_sound_survives_choreography_context(query):
    assert has_explicit_audio_clue(query)
    image = "Printed album artwork showing performers." if has_explicit_visual_clue(query) else ""
    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        apply_modality_query_safeguards(query, model_payload(image=image))
    result = apply_modality_query_safeguards(
        query, model_payload(audio="Piano, bass and drums with a husky vocal.", image=image)
    )
    assert result["audio_english_query"]
    assert result["modality_weights"]["audio"] > 0


def test_radio_failure_preserves_audio_context_and_input_payload():
    clue = {"target": "굿모닝FM", "relation": "방송·공연 일화",
            "search_query": RADIO_QUERY, "confidence": 0.8}
    payload = model_payload(audio="Funky bass groove.")
    payload["context_clues"] = [clue]
    original = deepcopy(payload)
    result = apply_modality_query_safeguards(RADIO_QUERY, payload)
    assert payload == original
    assert result["context_clues"] == [clue]
    assert result["image_english_query"] == ""
    assert result["audio_english_query"] == "Funky bass groove."
    assert result["modality_weights"] == original["modality_weights"]
    assert "베이스" in extract_audio_evidence_text(RADIO_QUERY)


def test_fallback_does_not_generate_an_image_from_cover_choreography():
    result = fallback_modality_payload(RADIO_QUERY)
    assert result["image_english_query"] == ""
    assert not result["has_visual_clue"]
    assert result["modality_weights"]["image"] == 0


def test_model_parse_retains_context_and_audible_bass_without_image():
    from src.retrieval.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer(api_key="synthetic-test-key")
    analysis = analyzer._parse(
        RADIO_QUERY,
        SimpleNamespace(text=json.dumps(model_payload(audio="Funky bass groove."))),
    )
    assert analysis.original_query == RADIO_QUERY
    assert analysis.confidence == 0.8
    assert analysis.context_clues
    assert analysis.audio_english_query
    assert analysis.image_english_query == ""
    assert not analysis.has_visual_clue


@pytest.mark.parametrize("async_call", [False, True])
def test_model_analysis_accepts_the_valid_mixed_response_on_first_attempt(async_call):
    from src.retrieval.query_analyzer import QueryAnalyzer

    calls = []
    response = SimpleNamespace(text=json.dumps(model_payload(audio="Funky bass groove.")))

    def generate(**kwargs):
        calls.append(kwargs)
        return response

    async def generate_async(**kwargs):
        return generate(**kwargs)

    analyzer = QueryAnalyzer(api_key="synthetic-test-key")
    analyzer._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate),
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_async)),
    )
    analysis = asyncio.run(analyzer.analyze_async(RADIO_QUERY)) if async_call else analyzer.analyze(RADIO_QUERY)
    assert len(calls) == 1
    assert analysis.confidence == 0.8
    assert analysis.context_clues
    assert analysis.audio_english_query
    assert not analysis.has_visual_clue


def test_real_missing_artwork_prompt_still_falls_back_after_validation_retries(monkeypatch):
    from src.retrieval import query_analyzer as qa

    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(model_payload()))

    analyzer = qa.QueryAnalyzer(api_key="synthetic-test-key")
    analyzer._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(qa.time, "sleep", lambda _: None)
    analysis = analyzer.analyze("커버댄스한 곡인데 앨범 표지에 빨간 꽃이 그려졌어")
    assert len(calls) == 3
    assert analysis.confidence == 0.0


def test_modality_guard_change_is_part_of_analyzer_cache_provenance(monkeypatch):
    from src.retrieval import analysis_cache as cache

    original_read = cache.Path.read_text
    def previous_guard(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        return text + "\n# different guard version\n" if path.name == "modality_queries.py" else text

    current = cache.analyzer_fingerprint()
    monkeypatch.setattr(cache.Path, "read_text", previous_guard)
    previous = cache.analyzer_fingerprint()
    assert current["prompt_sha"] == previous["prompt_sha"]
    assert current["postprocess_sha"] != previous["postprocess_sha"]
