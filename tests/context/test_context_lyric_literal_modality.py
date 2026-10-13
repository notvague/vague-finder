"""Invented lyrics cannot supply audio or artwork evidence."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.retrieval.modality_queries import (
    ModalityQueryValidationError, _mask_lyric_literals, apply_modality_query_safeguards,
    build_fallback_modality_queries, explicit_artwork_reference_start,
    extract_audio_evidence_text, has_explicit_audio_clue, has_explicit_visual_clue,
)
from src.retrieval.lyrics_query import extract_lyric_clues
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback


FRAGMENTS = ("따뜻한 저녁을 기다려", "피아노 기타 드럼이 잠든다", "앨범 커버엔 붉은 꽃이 보인다")
QUOTES = (('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'), ('「', '」'), ('『', '』'), ('`', '`'))


def model(**kwargs):
    raw = dict(intent_type="lyrics", image_english_query="", audio_english_query="",
               confidence=0.8, modality_weights=dict(text=1, image=0, audio=0))
    raw.update(kwargs)
    return raw


@pytest.mark.parametrize("fragment", FRAGMENTS)
@pytest.mark.parametrize("quotes", QUOTES)
@pytest.mark.parametrize("form", ('가사에 {literal}라고 나와', '{literal}라는 가사가 있어'))
def test_attributed_lyrics_are_text_only_for_supported_quotes(fragment, quotes, form):
    query = form.format(literal=quotes[0] + fragment + quotes[1])
    masked = _mask_lyric_literals(query)
    assert len(masked) == len(query) and fragment not in masked
    assert not has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    assert build_fallback_modality_queries(query) == ("", "")
    raw = model()
    before = deepcopy(raw)
    result = apply_modality_query_safeguards(query, raw)
    assert result["modality_weights"] == dict(text=1, image=0, audio=0)
    assert raw == before


@pytest.mark.parametrize("query", (
    '가사에 따뜻한 저녁이라고 나와', '가사에 피아노 기타라는 말이 나와',
    '후렴은 피아노를 기다려 하고 나왔어', '따뜻한 저녁이라는 가사가 있었어',
    '가사: 따뜻한 피아노를 기다려', '노랫말：앨범 커버에 붉은 꽃이 보인다',
))
def test_unquoted_attributed_fragments_and_dedicated_lines(query):
    assert not has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    assert build_fallback_modality_queries(query) == ("", "")


@pytest.mark.parametrize("lead,prompt", (
    ("피아노 반주가 있는 발라드인데", "A ballad with piano accompaniment."),
    ("피아노로 시작하는데", "A song that opens with piano."),
    ("잔잔한 노랜데", "A calm, gentle song."),
    ("남자가 부르는 발라드였고", "A ballad with male vocals."),
    ("템포가 빠르지만", "A fast tempo."),
    ("남자 목소리인데", "Male vocals."),
))
def test_description_before_an_unquoted_lyric_keeps_its_audio(lead, prompt):
    query = f"{lead} 사랑해라는 가사가 나와"
    masked = _mask_lyric_literals(query)
    assert len(masked) == len(query) and lead in masked and "사랑해" not in masked
    assert _mask_lyric_literals(masked) == masked
    assert has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    result = apply_modality_query_safeguards(query, model(audio_english_query=prompt))
    assert result["audio_english_query"] and result["modality_weights"]["audio"] > 0


def test_description_before_an_unquoted_lyric_keeps_its_artwork():
    query = "표지가 파란색인 앨범이고 사랑해라는 가사가 있어"
    masked = _mask_lyric_literals(query)
    assert "사랑해" not in masked and _mask_lyric_literals(masked) == masked
    assert has_explicit_visual_clue(query) and not has_explicit_audio_clue(query)
    assert explicit_artwork_reference_start(query) == 0
    assert build_fallback_modality_queries(query) == ("Album cover with blue tones.", "")
    result = apply_modality_query_safeguards(query, model(image_english_query="Album cover with blue tones."))
    assert result["image_english_query"] and result["modality_weights"]["image"] > 0


@pytest.mark.parametrize("query", (
    "노래는 잘 모르겠는데 피아노 기타 드럼이 잠든다라는 가사가 있어",
    "제목은 기억 안 나지만 앨범 커버엔 붉은 꽃이 보인다라는 가사가 나와",
    "어둠 가운데 피아노 소리라는 가사가 있었어",
    "피아노가 잠들었지만이라는 가사가 있어",
    # A clause link inside the lyric does not end the lyric.
    "피아노를 쳤는데 눈물이 났어라는 가사가 있어",
    "네 목소리가 들리는데 볼 수 없어라는 가사가 있어",
    "슬픈데 웃는다라는 가사가 있어",
    "피아노 소리   라는 가사가 있어",
    "앨범 커버엔 붉은 꽃이 보인다     라는 구절이 있어",
    "붉은 꽃 그림을 그렸지만 지워버렸어라는 가사가 있어",
    # Everything after a lyric marker is sung, whatever it names.
    "가사에 피아노를 쳤는데 눈물이 났어라고 나와",
    "가사에서 반주가 멈췄는데 눈물이 나라고 나와",
    "가사는 따뜻한 밤인데 피아노가 운다라고 나와",
    "가사는 기억 안 나지만 피아노를 쳤는데 눈물이 났어라는 말이 있었어",
))
def test_unquoted_lyric_words_stay_text_only(query):
    masked = _mask_lyric_literals(query)
    assert len(masked) == len(query) and _mask_lyric_literals(masked) == masked
    assert not has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    assert build_fallback_modality_queries(query) == ("", "")


def test_an_unknown_lyric_remark_does_not_open_the_lyric():
    query = "가사는 잘 모르겠는데 피아노 반주가 있고 사랑해라는 말이 나와"
    masked = _mask_lyric_literals(query)
    assert "피아노 반주" in masked and "사랑해" not in masked and _mask_lyric_literals(masked) == masked
    assert has_explicit_audio_clue(query)
    assert [clue.text for clue in extract_lyric_clues(query)] == ["사랑해"]


def test_description_ends_where_the_lyric_with_its_own_clause_link_begins():
    query = "피아노 반주가 있는 발라드인데 기타를 쳤는데 눈물이 났어라는 가사가 있어"
    evidence = extract_audio_evidence_text(query)
    assert "피아노 반주" in evidence and "기타" not in evidence and "눈물" not in evidence
    assert has_explicit_audio_clue(query)
    assert [clue.text for clue in extract_lyric_clues(query)][0] == "기타를 쳤는데 눈물이 났어"


@pytest.mark.parametrize("lyric", ('"사랑해"', "'사랑해'", "사랑해"))
def test_rule_fallback_keeps_the_description_before_a_lyric(lyric):
    sound = rule_fallback(f"잔잔한 발라드인데 {lyric}라는 가사가 있어")
    assert "ballad" in sound.audio_english_query and sound.modality_weights.audio > 0
    art = rule_fallback(f"표지가 파란색인 앨범이고 {lyric}라는 가사가 있어")
    assert art.image_english_query == "Album cover with blue tones." and not art.audio_english_query
    for analysis in (sound, art):
        assert "사랑해" in [clue.text for clue in analysis.lyric_clues]
        assert not any("발라드" in clue.text or "표지" in clue.text for clue in analysis.lyric_clues)


@pytest.mark.parametrize("quotes", QUOTES)
def test_words_before_a_quoted_lyric_are_never_taken_for_a_second_lyric(quotes):
    lyric = quotes[0] + "사랑해" + quotes[1]
    sound = f"잔잔한 피아노 발라드 {lyric}라는 가사가 있어"
    assert has_explicit_audio_clue(sound) and "ballad" in build_fallback_modality_queries(sound)[1]
    art = f"파란 표지 앨범 {lyric}라는 가사가 있어"
    assert has_explicit_visual_clue(art) and not has_explicit_audio_clue(art)
    assert build_fallback_modality_queries(art) == ("Album cover with blue tones.", "")


def test_actual_parser_keeps_the_model_lyric_beside_a_described_cover():
    query = "표지가 파란색인 앨범이고 사랑해라는 가사가 있어"
    raw = model(image_english_query="Album cover with blue tones.",
                modality_weights=dict(text=0.6, image=0.4, audio=0),
                lyric_clues=[dict(text="사랑해", kind="partial", variants=[], confidence=0.9, source="model")])
    analysis = QueryAnalyzer(api_key="placeholder")._parse(query, SimpleNamespace(text=json.dumps(raw, ensure_ascii=False)))
    assert "사랑해" in [clue.text for clue in analysis.lyric_clues]
    assert not any("표지" in clue.text for clue in analysis.lyric_clues)
    assert analysis.image_english_query and analysis.modality_weights.image > 0


def test_connected_quotes_apostrophes_and_newlines_keep_offsets():
    query = '가사에 "따뜻한 피아노"와 "붉은 앨범 커버"가 나와\n노랫말이 “기타\n드럼”이라고 들렸어'
    masked = _mask_lyric_literals(query)
    assert len(masked) == len(query)
    assert [i for i,c in enumerate(masked) if c == '\n'] == [i for i,c in enumerate(query) if c == '\n']
    assert not has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    assert 'piano' not in _mask_lyric_literals("가사에 'don't play the piano'가 나와")


@pytest.mark.parametrize("fragment", FRAGMENTS)
def test_outside_audio_is_retained_and_missing_prompt_is_still_rejected(fragment):
    query = f'가사에 "{fragment}"가 나왔고, 실제 소리는 피아노 반주에 허스키한 남자 보컬이었어.'
    assert fragment not in extract_audio_evidence_text(query)
    assert '피아노 반주' in extract_audio_evidence_text(query)
    assert has_explicit_audio_clue(query) and not has_explicit_visual_clue(query)
    with pytest.raises(ModalityQueryValidationError, match="explicit auditory clue"):
        apply_modality_query_safeguards(query, model())
    result = apply_modality_query_safeguards(query, model(audio_english_query="Husky male vocals with piano."))
    assert result["audio_english_query"] and result["modality_weights"]["audio"] > 0


def test_artwork_and_sound_do_not_inherit_lyric_features():
    query = '가사에 "기타와 붉은 꽃"이 나왔어. 표지에는 파란 배경에 흰 글씨가 있고, 피아노 반주가 들렸어.'
    assert has_explicit_audio_clue(query) and has_explicit_visual_clue(query)
    fallback = rule_fallback(query)
    assert 'blue' in fallback.image_english_query and 'red' not in fallback.image_english_query
    assert 'piano' in fallback.audio_english_query and 'guitar' not in fallback.audio_english_query
    assert explicit_artwork_reference_start(query) == query.index('표지')
    with pytest.raises(ModalityQueryValidationError, match="album-art visual clue"):
        apply_modality_query_safeguards(query, model())


@pytest.mark.parametrize("query", (
    '목소리가 "따뜻한" 노래', '반주가 "피아노와 기타"였어',
    '후렴에서 피아노 반주가 강하게 들렸어', '후렴은 "피아노와 현악기" 소리가 강했어',
    '가사에 "차가운 밤"이 나왔는데, 보컬은 "따뜻한" 느낌이었어',
    '앨범 커버에 "피아노"라는 글씨가 있고 노래는 잔잔했어',
))
def test_general_quotes_and_audible_choruses_still_open_audio(query):
    assert has_explicit_audio_clue(query)


@pytest.mark.parametrize("query", (
    '앨범 커버에 "기타"라는 글씨와 파란 꽃이 보였어',
    '가사에 "피아노"가 나왔는데 표지는 "붉은 꽃" 그림이었어',
))
def test_album_typography_quotes_still_open_image(query):
    assert has_explicit_visual_clue(query)


@pytest.mark.parametrize("fragment", FRAGMENTS[:2])
def test_actual_parser_preserves_lyrics_and_clears_inferred_performance(fragment):
    query = f'가사에 "{fragment}"라고 나오는 가상가수 노래 찾아줘.'
    raw = model(performance_clues=dict(vocal_count="solo", vocal_roles=["남성노래"], confidence=0.8))
    analysis = QueryAnalyzer(api_key="placeholder")._parse(query, SimpleNamespace(text=json.dumps(raw, ensure_ascii=False)))
    assert analysis.confidence == 0.8 and analysis.original_query == query
    assert any(item.text == fragment for item in analysis.lyric_clues)
    assert analysis.audio_english_query == analysis.image_english_query == ""
    assert analysis.performance_clues.vocal_count is None
    assert analysis.modality_weights.text == 1
    fallback = rule_fallback(query)
    assert fallback.audio_english_query == fallback.image_english_query == "" and fallback.confidence == 0


def test_independent_audio_quote_is_retained():
    query = '가사에 "따뜻한 저녁"이 나와. 그런데 반주는 "피아노와 현악기"야'
    assert '피아노와 현악기' in _mask_lyric_literals(query)
    assert has_explicit_audio_clue(query)


def test_attributed_chorus_quote_remains_literal():
    assert not has_explicit_audio_clue('후렴에 "피아노가 잠드는 밤"이라고 나왔어')


@pytest.mark.parametrize("quotes", QUOTES)
def test_multiline_lyrics_do_not_move_outside_artwork(quotes):
    fragment = "따뜻한 피아노\n기타와 붉은 앨범 커버"
    query = f'가사에 {quotes[0]}{fragment}{quotes[1]}가 나와. 표지는 파란 꽃 그림이야.'
    masked = _mask_lyric_literals(query)
    assert len(masked) == len(query) and _mask_lyric_literals(masked) == masked
    assert not has_explicit_audio_clue(query) and has_explicit_visual_clue(query)
    assert explicit_artwork_reference_start(query) == query.index('표지')


@pytest.mark.parametrize("word", ("아닌", "아니라", "말고"))
def test_explicitly_not_lyrics_are_not_hidden(word):
    assert has_explicit_audio_clue(f'가사가 {word} "따뜻한" 목소리였어')


@pytest.mark.parametrize("asynchronous", (False, True))
def test_complete_analyzer_accepts_text_only_lyrics_on_first_response(monkeypatch, asynchronous):
    query = '가사에 "따뜻한 저녁을 기다려"라고 나오는 가상가수 노래 찾아줘.'
    calls = []
    response = SimpleNamespace(text=json.dumps(model(), ensure_ascii=False))
    def generate(**kwargs):
        calls.append(kwargs)
        return response
    async def generate_async(**kwargs):
        return generate(**kwargs)
    analyzer = QueryAnalyzer(api_key="placeholder")
    monkeypatch.setattr(analyzer, "_client", SimpleNamespace(
        models=SimpleNamespace(generate_content=generate),
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_async)),
    ))
    analysis = asyncio.run(analyzer.analyze_async(query)) if asynchronous else analyzer.analyze(query)
    assert len(calls) == 1 and analysis.confidence == 0.8 and analysis.original_query == query
    assert any(item.text == "따뜻한 저녁을 기다려" for item in analysis.lyric_clues)
    assert analysis.modality_weights.text == 1
    assert analysis.image_english_query == analysis.audio_english_query == ""
