"""Audible mix positions survive validation; actual artwork leakage still fails."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
)


AUDIO_QUERY = "노래 중간에 남자 코러스가 들렸어."
MIXED_QUERY = (
    "노래 중간에 남자 코러스가 나오는데, 녹음 과정에서 다른 가수가 "
    "즉흥적으로 참여했다는 제작 비화가 있어."
)


def payload(audio="", *, image=""):
    return {
        "intent_type": "mixed", "confidence": 0.87, "context_clues": [],
        "audio_english_query": audio, "image_english_query": image,
        "modality_weights": {"text": 0.65, "audio": 0.35, "image": 0.0},
    }


@pytest.mark.parametrize("prompt", [
    "Female lead singing with male background vocals joining in the middle.",
    "Background vocals support the lead voice.",
    "Subtle background male vocals.",
    "Background soft female voices.",
    "Layered background vocal harmonies.",
    "Background backing vocals.",
    "BACKGROUND CHORUS with a soft lead voice.",
    "Background instrumental layers.",
    "Background acoustic accompaniment.",
    "Background sounds and foreground vocals.",
    "Background-vocals and foreground-bass.",
    "Foreground vocals are louder than background music.",
    "A guitar playing in the background.",
    "A piano softly playing in the background.",
    "Male voices are heard in the background.",
    "Vocal harmonies sit in the foreground.",
    "Instrumental layers in the foreground.",
    "Whispers quietly in the background.",
    "A faint chorus from the background.",
    "Drums gradually move into the foreground.",
    "Background ambience under the lead vocal.",
    "Background ambiance under the lead vocal.",
])
def test_audible_spatial_descriptions_preserve_prompt_and_weights(prompt):
    raw = payload(prompt)
    original = deepcopy(raw)
    result = apply_modality_query_safeguards(AUDIO_QUERY, raw)
    assert raw == original
    assert result["audio_english_query"] == prompt
    assert result["image_english_query"] == ""
    assert not result["has_visual_clue"]
    assert result["modality_weights"] == original["modality_weights"]


@pytest.mark.parametrize("prompt", [
    "A red background with vocals.",
    "A male singer with an illustrated background.",
    "A background image behind a singer.",
    "Foreground lettering beside a singer.",
    "Vocals with a background.",
    "Vocals with a foreground.",
    "Vocals. In the background stands a person.",
    "Background vocals with a red background.",
    "Male background vocals, plus a foreground image.",
    "Foreground vocals with background typography.",
    "Vocals in the foreground of an album cover.",
    "Guitar playing in the background of a portrait.",
    "Background music with cover art.",
    "Background instrumental layers with printed lettering.",
    "Background vocals beside a photograph.",
    "Background vocals alongside photographs.",
    "Background vocals with paper texture.",
    "Background vocals and geometric shapes.",
    "Background vocals evoking flowers on the cover.",
    "Background vocals plus album artwork.",
    "Background vocals accompanied by an image.",
    "Background vocals with pictures.",
    "A portrait, followed by background vocals.",
    "A background image, followed by foreground vocals.",
    "Vocals in the background; an image in the foreground.",
    "Foreground vocals; in the background there is a silhouette.",
])
def test_visual_background_or_other_leakage_is_still_rejected(prompt):
    with pytest.raises(ModalityQueryValidationError, match="visual-only wording"):
        apply_modality_query_safeguards(AUDIO_QUERY, payload(prompt))


@pytest.mark.parametrize("connector", [". ", "; ", ", ", " and "])
def test_an_audio_noun_in_another_clause_cannot_whitelist_a_visual_background(connector):
    with pytest.raises(ModalityQueryValidationError, match="visual-only wording"):
        apply_modality_query_safeguards(
            AUDIO_QUERY, payload("A chorus" + connector + "a background image")
        )


def test_separate_real_album_art_and_background_chorus_keep_both_modalities():
    query = "앨범 표지에는 검은 배경의 빨간 꽃이 있었고, 노래 중간에 남자 코러스가 들렸어."
    raw = payload("Male background vocals enter in the middle.",
                  image="Album artwork with red flowers on a black background.")
    raw["modality_weights"] = {"text": 0.6, "audio": 0.2, "image": 0.2}
    result = apply_modality_query_safeguards(query, raw)
    assert result["has_visual_clue"]
    assert result["audio_english_query"] == raw["audio_english_query"]
    assert result["image_english_query"] == raw["image_english_query"]
    assert result["modality_weights"] == raw["modality_weights"]


@pytest.mark.parametrize("query", [
    "앨범 표지에 검은 배경과 빨간 꽃이 있었어.",
    "옛 드라마 OST로 쓰였던 노래를 찾아줘.",
    "가사에 '보고 싶어'라는 말이 들어가는 노래를 찾아줘.",
])
def test_generated_background_vocals_do_not_create_user_audio_evidence(query):
    image = "Album artwork with red flowers on a black background." if "표지" in query else ""
    result = apply_modality_query_safeguards(query, payload("Background vocals.", image=image))
    assert result["audio_english_query"] == ""
    assert result["modality_weights"]["audio"] == 0


def test_real_sound_still_requires_a_dedicated_audio_prompt():
    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        apply_modality_query_safeguards(AUDIO_QUERY, payload())


def test_real_artwork_still_requires_a_dedicated_image_prompt():
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(
            AUDIO_QUERY + " 앨범 표지에 빨간 꽃이 있었어.", payload("Background vocals.")
        )


@pytest.mark.parametrize("async_call", [False, True])
def test_real_analyzer_accepts_spatial_audio_and_context_on_first_attempt(async_call):
    from src.retrieval.query_analyzer import QueryAnalyzer

    response = SimpleNamespace(text=json.dumps(payload(
        "Female lead singing with male background vocals joining in the middle."
    )))
    calls = []

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
    result = asyncio.run(analyzer.analyze_async(MIXED_QUERY)) if async_call else analyzer.analyze(MIXED_QUERY)
    assert len(calls) == 1
    assert result.confidence == 0.87
    assert result.context_clues
    assert result.audio_english_query
    assert result.image_english_query == ""
    assert result.modality_weights.audio > 0


def test_actual_visual_leakage_still_retries_and_returns_fallback(monkeypatch):
    from src.retrieval import query_analyzer as qa

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(payload("Male background vocals on a red album cover.")))

    analyzer = qa.QueryAnalyzer(api_key="synthetic-test-key")
    analyzer._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(qa.time, "sleep", lambda _: None)
    result = analyzer.analyze(MIXED_QUERY)
    assert len(calls) == 3
    assert result.confidence == 0.0


def test_real_context_clue_and_input_are_not_modified_by_audio_spatial_validation():
    clue = {"target": "", "relation": "제작·발매 비화",
            "search_query": MIXED_QUERY, "confidence": 0.8}
    raw = payload("Male background vocals enter in the middle.")
    raw["context_clues"] = [clue]
    before = deepcopy(raw)
    result = apply_modality_query_safeguards(MIXED_QUERY, raw)
    assert result["context_clues"] == [clue]
    assert raw == before


def test_spatial_guard_update_changes_cache_provenance_without_changing_prompt(monkeypatch):
    from src.retrieval import analysis_cache as cache

    original = cache.Path.read_text

    def previous_guard(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return text + "\n# different spatial validation\n" if path.name == "modality_queries.py" else text

    before = cache.analyzer_fingerprint()
    monkeypatch.setattr(cache.Path, "read_text", previous_guard)
    after = cache.analyzer_fingerprint()
    assert before["prompt_sha"] == after["prompt_sha"]
    assert before["postprocess_sha"] != after["postprocess_sha"]
