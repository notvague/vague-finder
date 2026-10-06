"""External song events survive parsing without opening album-art by name."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    extract_audio_evidence_text,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)


def payload(*, audio="", image="", context=None):
    return {
        "intent_type": "mixed", "confidence": 0.88,
        "context_clues": [] if context is None else context,
        "image_english_query": image, "audio_english_query": audio,
        "modality_weights": {"text": 0.65, "image": 0.0, "audio": 0.35}
        if audio else {"text": 1.0, "image": 0.0, "audio": 0.0},
    }


BROADCAST_QUERY = (
    "어떤 가수가 밤의 스케치북에서 부른 뒤 역주행했다는 곡을 찾아줘. "
    "피아노와 현악기 반주가 있고 감성적인 남자 발라드였어."
)
AUDIO_PROMPT = "An emotional male ballad with piano and string accompaniment."


@pytest.mark.parametrize("venue", [
    "스케치북", "수채화극장", "사진의밤", "그림음악회", "초상라이브",
])
@pytest.mark.parametrize("verb", ["부른", "불렀던", "연주한", "커버한"])
def test_performed_venue_name_does_not_supply_an_artwork_medium(venue, verb):
    query = f"{venue}에서 {verb} 곡을 찾고 있어. 피아노 반주의 남자 발라드야."
    assert not has_explicit_visual_clue(query)
    assert has_explicit_audio_clue(query)
    result = apply_modality_query_safeguards(query, payload(audio=AUDIO_PROMPT))
    assert not result["has_visual_clue"]
    assert result["image_english_query"] == ""
    assert result["audio_english_query"]
    assert "피아노" in extract_audio_evidence_text(query)


@pytest.mark.parametrize("query", [
    '프로그램 "파란사진"에서 부른 곡. 남자 보컬이 들렸어.',
    '프로그램 "그림과 사진의 밤"에서 부른 곡. 남자 보컬이 들렸어.',
    "방송 ‘새벽의 스케치북’에서 부른 곡인데 여자 보컬이었어.",
    "빛의 스케치북에서 부른 노래인데 여자 보컬과 바이올린 반주였어.",
    "그림라디오에서도 연주한 곡이래. 허스키한 남자 목소리였어.",
    "그림라디오에서는 커버한 곡인데 부드러운 여자 보컬이 들렸어.",
    "앨범 발매 후 수채화극장에서 부른 곡인데 남자 발라드였어.",
])
def test_programme_name_spelling_quotes_and_particles_stay_nonvisual(query):
    assert not has_explicit_visual_clue(query)


@pytest.mark.parametrize("query", [
    "스케치북에 남자의 초상이 그려져 있었어.",
    "스케치북에서 그린 꽃 그림과 손글씨 제목이 기억나.",
    "사진에서 남자가 노래를 부른 듯한 모습이 보였어.",
    "수채화로 그린 남자가 왼쪽에 있었어.",
    "스케치북에서 부른 곡인데, 앨범 표지에는 빨간 꽃이 있었어.",
    "스케치북에서 부른 곡이야. 손으로 그린 꽃과 손글씨 제목이 기억나.",
    "스케치북에서 부른 곡인데 손글씨 제목과 그린 꽃이 있었어.",
    "스케치북에서 부른 곡. 앨범에는 흰 글씨가 찍혀 있었어.",
])
def test_real_artwork_survives_next_to_or_without_a_performance(query):
    assert has_explicit_visual_clue(query)
    audio = AUDIO_PROMPT if has_explicit_audio_clue(query) else ""
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, payload(audio=audio))


def test_invented_image_is_removed_but_audio_and_source_are_preserved():
    raw = payload(audio=AUDIO_PROMPT, image="A drawing of a man.")
    # Detailed sound already uses the pre-existing Audio weight floor. Start
    # at that valid balance so this assertion isolates the image-name fix.
    raw["modality_weights"] = {"text": 0.5, "image": 0.0, "audio": 0.5}
    before = deepcopy(raw)
    result = apply_modality_query_safeguards(BROADCAST_QUERY, raw)
    assert raw == before
    assert result["image_english_query"] == ""
    assert result["audio_english_query"] == AUDIO_PROMPT
    assert result["modality_weights"] == before["modality_weights"]


@pytest.mark.parametrize("query, relation, anchor", [
    ("작곡할 때 가이드가 장난스러운 이름이었다는 곡을 찾아줘.", "제작·발매 비화", "가이드"),
    ("작사 비화가 있는데 작사가가 마지막 구절을 바꿨다는 곡이야.", "제작·발매 비화", "작사 비화"),
    ("작사가가 원래의 구절을 다른 말로 바꾼 곡을 찾아줘.", "제작·발매 비화", "바꾼"),
    ("신인 프로듀서팀의 데뷔곡이자 작곡 능력을 인정받게 한 곡이래.", "제작·발매 비화", "데뷔곡"),
    ("뮤비를 해외 도시에서 찍고 다른 배우가 출연한 노래였어.", "뮤직비디오 속 사건", "뮤비"),
    ("이 노래의 뮤비를 해안 복구 프로젝트에 참여하면서 찍었어.", "뮤직비디오 속 사건", "프로젝트"),
    ("뮤비에서 시계 모양의 건물이 등장하는 연출을 했던 노래야.", "뮤직비디오 속 사건", "시계"),
    ("게임에 들어갔다가 판권 만료로 삭제된 노래를 찾아줘.", "다른 작품에 사용", "판권 만료"),
    ("라이선스가 만료돼 리듬 게임에서 삭제됐다는 곡이 있었어.", "다른 작품에 사용", "삭제"),
    ("무대에서 공이 관객석으로 갔다가 다시 튕겨온 방송사고를 봤어. 그때 부른 곡은?", "방송·공연 일화", "방송사고"),
    ("콘서트에서 가수가 어린 관객들에게 인기 많다고 말했던 곡이야.", "방송·공연 일화", "말했던"),
    ("경연 명명식에서 어떤 팀이 부른 대표곡을 찾고 있어.", "방송·공연 일화", "명명식"),
    ("라디오에서 다른 가수가 부르던 곡을 찾아줘.", "방송·공연 일화", "부르던"),
    ("광고에서 소화제 이름으로 개사한 댄스곡을 찾아줘.", "다른 작품에 사용", "개사한"),
    ("한 선수를 위해 응원가로 개사된 아이돌 노래를 찾아줘.", "다른 작품에 사용", "응원가"),
    ("이 곡 중간의 남자 코러스가 친한 가수가 즉흥적으로 녹음한 거라는 여담이 있어.", "보컬 참여 일화", "녹음한"),
    ("백보컬을 다른 가수가 녹음했다는 곡이 있었어.", "보컬 참여 일화", "백보컬"),
    ("한 가수에게 갈 예정이었는데 군입대 때문에 다른 가수에게 돌아갔다는 곡을 찾아줘.", "제작·발매 비화", "군입대"),
    ("지역을 살리는 프로젝트에 참여한 노래를 찾아줘.", "제작·활동에 얽힌 사건", "프로젝트"),
])
def test_reported_external_facts_survive_even_when_model_omits_context(query, relation, anchor):
    raw = payload()
    before = deepcopy(raw)
    clues = apply_context_query_safeguards(query, raw)["context_clues"]
    assert raw == before
    assert clues and clues[0]["relation"] == relation
    assert any(anchor in clue["search_query"] for clue in clues)
    assert all(0 < clue["confidence"] <= 0.8 for clue in clues)
    assert all(not clue["target"] or clue["target"] in query for clue in clues)


@pytest.mark.parametrize("query", [
    "남자 코러스가 들리고 피아노 반주가 있었어.",
    "중간에 즉흥적인 코러스가 들리는 노래였어.",
    "남자 코러스를 녹음한 잔잔한 노래를 찾아줘.",
    "부드러운 백보컬을 녹음한 곡을 찾아줘.",
    "가수 이름만 기억나는데 조용한 노래를 찾고 있어.",
    "작사 비화라는 제목의 노래를 찾아줘.",
    "제목에 작사 비화라는 말이 들어가는 노래.",
    "가사에 '작사 비화'라는 말이 나오는 노래.",
    "가사에 '녹음했다는 곡'이라는 구절이 들어가 있어.",
    "앨범 표지에 작사 비화라는 글자가 있어.",
    "앨범 커버에 뮤비라는 글자가 인쇄되어 있어.",
    "가사에 뮤비를 찍었다는 말이 나오는 노래.",
    "뮤비처럼 들리는 기타 소리를 찾고 있어.",
    "광고 같은 느낌으로 들리는 댄스곡.",
    "광고에서 쓸 노래를 추천해줘.",
    "응원가로 개사할 노래를 고르고 싶어.",
    "방송에서 부를 곡을 추천해줘.",
    "명명식에서 부르려고 노래를 고르고 있어.",
    "프로젝트에 참여하려고 곡을 찾고 있어.",
    "노래방에서 내가 부른 곡인데 기억이 안 나.",
    "내 방에서 기타를 연주한 노래인데 제목을 잊었어.",
])
def test_audio_titles_lyrics_artwork_and_future_activities_do_not_invent_context(query):
    # Even a fabricated model clue must be backed by a real event in the source.
    invented = [{"target": "없는작품", "relation": "제작 비화",
                 "search_query": "없는작품 녹음한 가상정답", "confidence": 0.99}]
    result = apply_context_query_safeguards(query, payload(context=invented))
    assert result["context_clues"] == []


def test_real_lyric_revision_with_quoted_before_and_after_still_is_an_event():
    query = "가사에 '내 꿈'이라는 구절이 있었는데 작사가가 '네 꿈'으로 바꿨다는 제작 비화야."
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1
    assert clues[0]["relation"] == "제작·발매 비화"
    assert "바꿨" in clues[0]["search_query"]


def test_text_shown_inside_an_actual_video_still_is_an_external_fact():
    query = "뮤비에서 다음 앨범 제목이라는 문장이 나오는 연출을 한 곡이래."
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1 and clues[0]["relation"] == "뮤직비디오 속 사건"
    assert "다음 앨범" in clues[0]["search_query"]


def test_grounded_model_search_cannot_turn_words_inside_a_title_into_an_event():
    query = "제목에 '광고에서 부른 노래'라는 말이 들어가는 곡이야."
    grounded = [{"target": "", "relation": "부른",
                 "search_query": "광고에서 부른 노래", "confidence": 0.9}]
    assert apply_context_query_safeguards(query, payload(context=grounded))["context_clues"] == []


def test_audio_visual_and_context_can_all_be_explicit_in_one_query():
    from src.retrieval.query_analyzer import QueryAnalyzer

    query = (
        "콘서트에서 다른 가수가 부른 노래야. "
        "앨범 표지에는 빨간 꽃이 그려져 있었고 남자 코러스가 들렸어."
    )
    raw = payload(audio="Male background vocals.", image="Album artwork with red flowers.")
    raw["modality_weights"] = {"text": 0.6, "image": 0.2, "audio": 0.2}
    result = QueryAnalyzer(api_key="synthetic-test-key")._parse(
        query, SimpleNamespace(text=json.dumps(raw))
    )
    assert result.original_query == query
    assert result.context_clues and result.has_visual_clue
    assert result.image_english_query and result.audio_english_query
    assert result.modality_weights.model_dump() == raw["modality_weights"]


def test_new_fact_rule_still_rejects_invented_answer_names_and_relations():
    query = "노래 중간 코러스를 친한 가수가 즉흥적으로 녹음한 여담이 있어."
    invented = [{"target": "없는영화", "relation": "없는영화 OST",
                 "search_query": "없는영화 OST 가상정답 가상가수", "confidence": 0.99}]
    clues = apply_context_query_safeguards(query, payload(context=invented))["context_clues"]
    assert len(clues) == 1
    assert "녹음한" in clues[0]["search_query"]
    assert "없는영화" not in str(clues) and "가상정답" not in str(clues)
    assert "OST" not in str(clues)


def test_multiple_event_types_remain_separate_and_literal():
    query = "뮤비를 해외에서 찍었어, 광고에서 다른 말로 개사한 곡이야."
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 2
    assert [clue["relation"] for clue in clues] == ["뮤직비디오 속 사건", "다른 작품에 사용"]
    assert "광고" not in clues[0]["search_query"]
    assert "뮤비" not in clues[1]["search_query"]


@pytest.mark.parametrize("video", ["뮤비", "뮤직비디오", "MV", "mv"])
@pytest.mark.parametrize("followup", ["그 {}의 노래가 뭐였지?", "그 {} 곡을 찾아줘."])
def test_video_followup_question_does_not_add_a_second_event(video, followup):
    query = "뮤직비디오를 해외 도시에서 찍었어, " + followup.format(video)
    clues = apply_context_query_safeguards(query, payload())["context_clues"]
    assert len(clues) == 1
    assert "해외 도시" in clues[0]["search_query"]


@pytest.mark.parametrize("async_call", [False, True])
def test_real_analyzer_accepts_broadcast_and_sound_on_first_attempt(async_call):
    from src.retrieval.query_analyzer import QueryAnalyzer

    response = SimpleNamespace(text=json.dumps(payload(audio=AUDIO_PROMPT)))
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
    result = asyncio.run(analyzer.analyze_async(BROADCAST_QUERY)) if async_call else analyzer.analyze(BROADCAST_QUERY)
    assert len(calls) == 1
    assert result.original_query == BROADCAST_QUERY
    assert result.confidence == 0.88
    assert result.context_clues and result.audio_english_query
    assert result.image_english_query == "" and not result.has_visual_clue


def test_recording_participant_does_not_replace_audible_chorus_with_an_image():
    from src.retrieval.query_analyzer import QueryAnalyzer

    query = "중간의 남자 코러스가 다른 가수가 즉흥적으로 녹음한 거라는 여담을 들었어."
    prompt = "A soft lead voice with male background vocals in the middle."
    analyzer = QueryAnalyzer(api_key="synthetic-test-key")
    result = analyzer._parse(query, SimpleNamespace(text=json.dumps(payload(audio=prompt))))
    assert result.original_query == query
    assert result.audio_english_query == prompt
    assert result.context_clues and "녹음한" in result.context_clues[0].search_query
    assert not result.has_visual_clue and not result.image_english_query
