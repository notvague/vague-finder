"""A cover competition/performance is not an album-cover appearance clue."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    explicit_artwork_reference_start,
    has_explicit_artwork_reference,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)


def payload(*, audio="", image="", context=None):
    return {
        "intent_type": "mixed", "confidence": 0.89,
        "context_clues": [] if context is None else context,
        "image_english_query": image, "audio_english_query": audio,
        "modality_weights": {"text": 0.7, "image": 0.0, "audio": 0.3}
        if audio else {"text": 1.0, "image": 0.0, "audio": 0.0},
    }


@pytest.mark.parametrize("event", ["서바이벌", "공모전", "경연", "대회", "콘테스트"])
@pytest.mark.parametrize("space", ["", " ", "\t", "\n"])
def test_cover_contest_does_not_create_image_or_audio_evidence(event, space):
    query = f"한 플랫폼이 커버{space}{event}을 열었던 곡을 찾아줘."
    assert not has_explicit_artwork_reference(query)
    assert not has_explicit_visual_clue(query)
    assert not has_explicit_audio_clue(query)
    result = apply_modality_query_safeguards(query, payload(image="A drawing of a singer."))
    assert result["image_english_query"] == result["audio_english_query"] == ""
    assert result["modality_weights"] == {"text": 1.0, "image": 0.0, "audio": 0.0}


@pytest.mark.parametrize("event", ["서바이벌", "공모전", "경연", "대회", "콘테스트"])
@pytest.mark.parametrize("space", ["", " ", "\t", "\n"])
def test_reported_contest_stays_context_when_model_omits_the_clue(event, space):
    query = f"어떤 회사가 커버{space}{event}을 열었던 노래가 뭐였지?"
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1
    assert clues[0]["relation"] == "방송·공연 일화"
    assert "열었던" in clues[0]["search_query"] and event in clues[0]["search_query"]
    assert clues[0]["target"] == ""


@pytest.mark.parametrize("kind", ["영상", "라이브", "연주", "공연"])
@pytest.mark.parametrize("space", ["", " "])
def test_cover_recording_is_not_artwork_and_past_release_is_context(kind, space):
    query = f"다른 가수가 커버{space}{kind}을 공개했던 곡이 뭐지?"
    assert not has_explicit_visual_clue(query)
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1 and clues[0]["relation"] == "커버·리메이크·답가"
    assert "공개했던" in clues[0]["search_query"]


@pytest.mark.parametrize("query", [
    "커버 서바이벌이라는 패러디 공모전을 열었던 곡은?",
    "노래 커버 대회에서 우승한 곡을 찾아줘.",
    "커버경연에서 참가했던 노래가 뭐지?",
    "커버 콘테스트를 개최했던 곡을 찾아줘.",
    "커버 공모전에 선정된 곡을 찾아줘.",
    "커버-서바이벌을 열었던 곡은?",
    "커버·서바이벌을 열었던 곡은?",
    '"커버" 서바이벌을 열었던 노래는?',
    "커버\n댄스한 노래가 뭐였지?",
    "댄스\n커버했던 노래가 뭐였지?",
])
def test_performance_variants_keep_a_single_grounded_context_clue(query):
    assert not has_explicit_visual_clue(query)
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1 and 0 < clues[0]["confidence"] <= 0.8


@pytest.mark.parametrize("query", [
    "커버 서바이벌 사진에는 파란 배경의 남자가 있었어.",
    "커버 경연 영상에서 남자가 빨간 그림 앞에서 노래했어.",
    "커버 공모전 영상에 검은 배경과 손글씨가 보였어.",
    "커버영상에는 꽃 사진과 손글씨 자막이 있었어.",
    "커버 라이브에서 하얀 배경에 남자 실루엣이 보였어.",
])
def test_visuals_of_a_cover_performance_do_not_open_album_art(query):
    assert not has_explicit_visual_clue(query)


@pytest.mark.parametrize("query", [
    "커버 서바이벌에 나왔던 노래. 앨범 커버에는 빨간 꽃이 있었어.",
    "커버 대회에서 우승한 노래의 앨범 표지에 남자의 그림이 있었어.",
    "커버 공모전에 사용된 곡인데 커버에는 검은 배경과 흰 글씨가 있었어.",
    "커버영상을 봤던 노래인데 앨범 커버에는 파란 꽃이 있었어.",
    "앨범 커버 공모전에 나온 표지는 빨간 그림과 손글씨였어.",
    "앨범의 커버 공모전에 나온 빨간 꽃 그림이 기억나.",
    "앨범 커버 영상에는 실제 표지 사진이 나왔어.",
    "앨범의 커버 영상에서 빨간 꽃 표지를 봤어.",
    "커버 공모전에 제출한 앨범 커버 디자인에는 빨간 꽃이 있었어.",
    "커버 디자인 공모전의 앨범 표지에 하얀 글씨가 있었어.",
    "커버 라이브 곡의 앨범 커버 해상도가 낮았고 배경은 분홍색이었어.",
    "커버 서바이벌을 열었던 곡인데 손으로 그린 꽃과 손글씨 제목을 기억해.",
])
def test_independent_album_art_and_cover_design_still_require_image(query):
    assert has_explicit_visual_clue(query)
    audio = "Male vocals with piano." if has_explicit_audio_clue(query) else ""
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, payload(audio=audio))


@pytest.mark.parametrize("query, event_text", [
    ("커버 대회에서 우승한 곡인데 앨범 커버에는 빨간 꽃이 있었어.", "우승한"),
    ("커버 공모전에 사용된 노래의 앨범 표지에는 파란 꽃이 있었어.", "사용된"),
    ("커버 라이브를 공개했던 곡인데 앨범 표지에 흰 글씨가 있었어.", "공개했던"),
    ("커버 영상을 올린 노래의 앨범 커버에는 빨간 꽃이 있었어.", "올린"),
])
def test_a_past_event_and_real_artwork_in_one_clause_keep_both_paths(query, event_text):
    image = "Printed album artwork with flowers and lettering."
    result = apply_modality_query_safeguards(query, payload(image=image))
    assert result["has_visual_clue"] and result["image_english_query"] == image
    clues = apply_context_query_safeguards(query, result)["context_clues"]
    assert len(clues) == 1 and event_text in clues[0]["search_query"]
    assert "앨범" not in clues[0]["search_query"] and "꽃" not in clues[0]["search_query"]


def test_negated_artwork_does_not_hide_a_later_genuine_artwork_offset():
    query = "커버 공모전을 열었던 노래야. 표지는 기억 안 나지만 앨범 커버에 빨간 꽃이 있었어."
    start = explicit_artwork_reference_start(query)
    assert start == query.index("앨범 커버")
    assert query[start:].startswith("앨범 커버")


@pytest.mark.parametrize("query", [
    "커버 대회에 참가할 곡을 추천해줘.",
    "커버 서바이벌을 열려고 곡을 고르고 있어.",
    "커버 공모전에 참가하려고 노래를 찾고 있어.",
    "커버 경연에서 부를 노래를 찾고 있어.",
    "커버 콘테스트에 낼 곡을 추천해줘.",
    "커버영상을 만들려고 노래를 고르고 있어.",
    "커버 라이브를 할 곡을 추천해줘.",
    "가사에 커버 대회에서 우승했다는 말이 나오는 노래.",
    "제목에 커버 서바이벌을 열었다는 말이 있는 노래.",
    "앨범 표지에 커버 공모전에서 우승했다는 글자가 있어.",
    "커버 공모전을 개최했던 사람의 사진이 앨범 표지에 그려져 있었어.",
])
def test_future_activities_lyrics_title_and_artwork_do_not_create_context(query):
    assert apply_context_query_safeguards(query, payload())["context_clues"] == []


@pytest.mark.parametrize("query", [
    "커버 공연을 공개하려고 노래를 고르고 있어.",
    "커버 공연을 제작할 곡을 추천해줘.",
    "커버 서바이벌 공모전을 패러디하려고 곡을 찾아줘.",
    "커버 대회에 참가하려고 곡을 찾고 있어.",
    "커버댄스할 곡을 골라줘.",
])
def test_model_cannot_promote_a_planned_cover_activity_into_a_reported_fact(query):
    invented = [{"target": "", "relation": "공개", "search_query": query, "confidence": 0.95}]
    assert apply_context_query_safeguards(query, payload(context=invented))["context_clues"] == []


def test_future_activity_does_not_hide_an_independent_past_usage_event():
    query = "커버 공연을 공개하려고 곡을 찾고 있어. 드라마 가상작품 OST였던 곡이야."
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1 and clues[0]["relation"] == "삽입곡·배경음악"
    assert "커버" not in clues[0]["search_query"]


def test_event_does_not_remove_a_real_audio_description_or_mutate_raw_payload():
    query = "커버 서바이벌을 열었던 곡이야. 노래 중간에 남자 코러스가 들렸어."
    raw = payload(audio="Male background vocals enter in the middle.")
    before = deepcopy(raw)
    result = apply_modality_query_safeguards(query, raw)
    assert raw == before
    assert result["audio_english_query"] == raw["audio_english_query"]
    assert result["modality_weights"] == raw["modality_weights"]
    assert not result["has_visual_clue"]
    clues = apply_context_query_safeguards(query, result)["context_clues"]
    assert len(clues) == 1 and "열었던" in clues[0]["search_query"]


def test_cover_contest_cannot_whitelist_actual_visual_leak_in_audio_prompt():
    query = "커버 서바이벌에서 불렀던 곡이야. 남자 코러스가 들렸어."
    with pytest.raises(ModalityQueryValidationError, match="visual-only wording"):
        apply_modality_query_safeguards(query, payload(audio="Male background vocals on a red album cover."))


def test_model_cannot_add_an_answer_artist_or_work_to_the_contest():
    query = "커버 서바이벌이라는 패러디 공모전을 열었던 곡이 뭐지?"
    invented = [{"target": "없는작품", "relation": "없는작품 OST",
                 "search_query": "없는작품 OST 없는가수 가상정답", "confidence": 0.99}]
    clues = apply_context_query_safeguards(query, payload(context=invented))["context_clues"]
    assert len(clues) == 1
    assert "가상정답" not in str(clues) and "없는가수" not in str(clues)
    assert "없는작품" not in str(clues) and "OST" not in str(clues)


@pytest.mark.parametrize("async_call", [False, True])
def test_real_analyzer_accepts_the_contest_on_first_attempt_without_image(async_call):
    from src.retrieval.query_analyzer import QueryAnalyzer

    query = "어떤 플랫폼이 커버 서바이벌이라는 패러디 공모전을 열었던 곡이 뭐지?"
    response = SimpleNamespace(text=json.dumps(payload()))
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
    result = asyncio.run(analyzer.analyze_async(query)) if async_call else analyzer.analyze(query)
    assert len(calls) == 1 and result.confidence == 0.89
    assert result.original_query == query and result.context_clues
    assert result.image_english_query == result.audio_english_query == ""
