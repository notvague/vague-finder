"""Synthetic completed adaptations and specific event evidence boundaries."""

from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import _related, select_context_fact_for_result
from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback


def clue(query, relation="다른 작품에 사용", target=""):
    return ContextClue(target=target, relation=relation, search_query=query, confidence=0.8)


def fact(text, *, sid="synthetic", rid="synthetic", category="media_usage", **overrides):
    values = dict(song_id=sid, record_id=f"nw:{sid}:{rid}", fact_text=text,
                  source_url="https://namu.wiki/w/Synthetic", title="가상 곡",
                  artists=("가상 가수",), category=category, section="여담", score=0.7)
    return NS(**{**values, **overrides})


def route(cue, facts=(), *, sid="synthetic", alternatives=()):
    dense = NS(best_fact=facts[0]) if facts else None
    return NS(song_id=sid, clue=cue, alternatives=alternatives,
              fused_hit=NS(song_id=sid, dense_song=dense, dense_facts=facts))


@pytest.mark.parametrize("event", [
    "광고에서 가사를 바꿔 부른 노래야",
    "광고에서 노랫말을 바꾸어 부른 노래야",
    "광고에서 가삿말을 고쳐서 불렀던 곡이야",
    "광고에 맞춰 가사 일부를 바꾼 곡이야",
    "광고에서 가사를 변경한 노래야",
    "광고에서 노랫말을 수정했던 노래야",
    "가사를 바꿔서 부른 광고 노래야",
    "노랫말을 고쳐 녹음한 광고 노래야",
    "응원가로 가사를 바꿔 부른 노래야",
    "로고송에서 가사를 바꿔 부른 노래야",
    "CM송에서 노랫말을 바꿔 부른 곡이야",
    "CF에서 가사를 고쳐 부른 노래야",
])
@pytest.mark.parametrize("mixed", [False, True])
def test_completed_commercial_adaptation_recovered_without_model_context(event, mixed):
    query = "가상 가수가 2020년 " + event
    if mixed:
        query += ". 앨범 표지에 빨간 꽃이 있었어. 피아노 반주가 들렸어."
    raw = dict(intent_type="mixed", confidence=0.9, context_clues=[],
               image_english_query="Red flowers on album artwork." if mixed else "",
               audio_english_query="Soft piano accompaniment." if mixed else "",
               modality_weights=dict(text=0.6 if mixed else 1.0,
                                     image=0.2 if mixed else 0.0, audio=0.2 if mixed else 0.0))
    frozen = deepcopy(raw)
    analysis = QueryAnalyzer(api_key="synthetic-test-key")._parse(query, NS(text=json.dumps(raw)))
    assert raw == frozen and analysis.original_query == query and analysis.confidence == 0.9
    assert len(analysis.context_clues) == 1
    assert analysis.context_clues[0].relation in {"다른 작품에 사용", "삽입곡·배경음악"}
    assert "가상 가수" in analysis.context_clues[0].search_query
    assert "빨간 꽃" not in analysis.context_clues[0].search_query
    assert bool(analysis.audio_english_query) is mixed
    assert bool(analysis.image_english_query) is mixed


@pytest.mark.parametrize("query", [
    "광고에서 가사를 바꿔 부를 노래를 추천해줘.",
    "광고에 쓰려고 가사를 바꾸어 부르고 싶어.",
    "응원가로 가사를 바꿔 부를 곡을 찾고 있어.",
    "가사에 '광고에서 가사를 바꿔 부른 노래'라는 문장이 있어.",
    "제목에 '광고에서 가사를 바꿔 부른 곡'이라는 말이 있어.",
    "앨범 표지에 '광고에서 가사를 바꿔 부른 곡'이라는 문구가 있어.",
    "가사에 광고에서 가사를 바꿔 부른 곡이라는 문장이 있어.",
    "광고처럼 들리는 밝은 피아노 노래야.",
])
def test_planned_literal_and_auditory_mentions_do_not_invent_an_external_event(query):
    result = apply_context_query_safeguards(query, dict(context_clues=[]))
    assert result["context_clues"] == []


def test_model_answer_invention_and_confidence_are_still_rejected():
    query = "광고에서 가사를 바꿔 부른 곡인지 모르겠어."
    invented = dict(target="없는 작품", relation="OST", search_query="없는 작품 OST 정답", confidence=1)
    result = apply_context_query_safeguards(query, dict(context_clues=[invented]))
    assert len(result["context_clues"]) == 1
    assert result["context_clues"][0]["confidence"] <= 0.6
    assert "없는 작품" not in str(result) and "정답" not in str(result)
    assert rule_fallback(query).context_clues


@pytest.mark.parametrize("query", [
    "가사를 바꿔 부른 광고 노래 찾아줘.",
    "노랫말을 고쳐 녹음한 광고 노래였어.",
    "가사 일부를 수정한 응원가 노래였어.",
])
def test_external_adaptation_can_precede_the_advertisement_without_becoming_a_lyric_search(query):
    result = apply_context_query_safeguards(query, dict(context_clues=[]))
    assert len(result["context_clues"]) == 1


@pytest.mark.parametrize("revision", ["개사하여", "가사를 바꾸어 부르며", "노랫말을 수정하여"])
def test_adaptation_fact_retains_actual_record_text_and_url(revision):
    cue = clue("가상 가수가 2020년 가상상품 광고에서 가사를 바꿔 부른 노래")
    value = fact(f"2020년 가상 가수가 {revision} 가상상품 광고를 촬영했다.", category="version")
    assert _related(cue, value, "synthetic")
    selected = select_context_fact_for_result(route(cue, (value,)), query_clues=[cue])
    assert selected is value and selected.source_url == value.source_url


@pytest.mark.parametrize("cause", ["판권 만료", "라이선스 종료", "저작권 계약 만료"])
def test_deletion_does_not_prove_the_stated_license_reason(cause):
    cue = clue(f"게임 가상게임에서 {cause}로 삭제된 노래", target="가상게임")
    missing = fact("가상게임에 이 곡이 수록되었다가 삭제되었다.")
    assert not _related(cue, missing, "synthetic")
    supported = fact(f"가상게임에서 {cause}로 이 곡이 삭제되었다.")
    assert _related(cue, supported, "synthetic")


def test_generic_deletion_is_still_valid_without_inventing_a_license_reason():
    cue = clue("가상게임에서 삭제된 노래", target="가상게임")
    assert _related(cue, fact("가상게임에서 이 곡이 수록되었다가 삭제되었다."), "synthetic")
    assert not _related(cue, fact("가상게임에서 이 곡이 수록되었다."), "synthetic")


def test_license_reason_in_another_sentence_cannot_explain_this_removal():
    cue = clue("가상게임에서 판권 만료로 삭제된 노래", target="가상게임")
    value = fact("가상게임에서 이 곡이 삭제되었다. 다른 곡의 판권 만료도 있었다.")
    assert not _related(cue, value, "synthetic")


def test_license_expiry_alone_does_not_assert_the_requested_removal():
    cue = clue("가상게임에서 판권 만료로 삭제된 노래", target="가상게임")
    assert not _related(cue, fact("가상게임에서 이 곡의 판권이 만료되었다."), "synthetic")


@pytest.mark.parametrize("text", [
    "가상게임에서 이 곡이 삭제되었다.",
    "가상게임에서 이 곡의 노랫말을 수정했다.",
    "가상게임에서 관련 게시물이 삭제되었다.",
])
def test_withdrawal_and_adaptation_alone_do_not_certify_a_generic_ost_use(text):
    cue = clue("가상게임 OST", relation="OST 삽입", target="가상게임")
    assert not _related(cue, fact(text), "synthetic")


def test_a_removed_article_is_not_a_removed_track_even_for_a_withdrawal_request():
    cue = clue("가상게임에서 삭제된 노래", target="가상게임")
    assert not _related(cue, fact("가상게임에서 관련 게시물이 삭제되었다."), "synthetic")


def test_shared_concert_without_an_explicit_speech_claim_keeps_the_existing_scope():
    cue = clue("은빛 하늘 콘서트에서 부른 노래", relation="방송·공연 일화")
    assert _related(cue, fact("은빛 하늘 콘서트에서 부른 곡이다.", category="performance"), "synthetic")


def test_speech_content_mentioned_elsewhere_cannot_substantiate_the_same_event():
    cue = clue("은빛 하늘 콘서트에서 초등학생들에게 인기 많다고 말했던 노래", relation="방송·공연 일화")
    value = fact("은빛 하늘 콘서트에서 옛 곡에 대해 말했다. 다른 노래는 초등학생들에게 인기가 많았다.", category="performance")
    assert not _related(cue, value, "synthetic")


@pytest.mark.parametrize("reporting", ["라고 말했던", "라고 이야기했던", "라고 설명한", "라고 언급한"])
@pytest.mark.parametrize("quoted", [False, True])
def test_speech_content_cannot_be_replaced_by_a_shared_concert(reporting, quoted):
    content = "이 노래 초등학생들에게 인기 많다면서요"
    if quoted:
        content = f'"{content}"'
    cue = clue(f"가상 가수가 은빛 하늘 콘서트에서 {content}{reporting} 노래",
               relation="방송·공연 일화")
    unrelated = fact("은빛 하늘 콘서트에서 가상 가수가 옛 곡의 느낌을 재현하려고 노력했다고 말했다.",
                     category="performance")
    correct = fact(f'은빛 하늘 콘서트에서 가상 가수가 "이 노래 초등학생들에게 인기 많다면서요"라고 말하기도.',
                   category="performance")
    assert not _related(cue, unrelated, "synthetic")
    assert _related(cue, correct, "synthetic")
    # The first retrieved fact is higher-ranked. Selection still skips it and
    # returns the supported alternative from this request's existing snapshot.
    correct.record_id = "nw:synthetic:correct"
    assert select_context_fact_for_result(route(cue, (unrelated, correct)), query_clues=[cue]) is correct


@pytest.mark.parametrize("failure", ["empty", "unsafe_url", "other_song", "wrong_record", "negative", "unsupported_number"])
def test_new_conditions_do_not_bypass_identity_url_negation_or_numeric_guards(failure):
    cue = clue("가상게임에서 2020년 판권 만료로 삭제된 노래", target="가상게임")
    value = fact("가상게임에서 2020년 판권 만료로 이 곡이 삭제되었다.")
    if failure == "empty": value.fact_text = ""
    elif failure == "unsafe_url": value.source_url = "https://example.invalid/w/Source"
    elif failure == "other_song": value.song_id = "another"
    elif failure == "wrong_record": value.record_id = "nw:another:record"
    elif failure == "negative": value.fact_text = "가상게임에서 2020년 판권 만료로 삭제되지 않았다."
    else: value.fact_text = "가상게임에서 2021년 판권 만료로 삭제되었다."
    assert not _related(cue, value, "synthetic")
    assert select_context_fact_for_result(route(cue, (value,)), query_clues=[cue]) is None


def test_sparse_only_is_never_fact_evidence_and_other_snapshot_keeps_one_bound_fact():
    cue = clue("가상게임에서 판권 만료로 삭제된 노래", target="가상게임")
    sparse = route(cue)
    assert select_context_fact_for_result(sparse, query_clues=[cue]) is None
    good = fact("가상게임에서 판권 만료로 이 곡이 삭제되었다.")
    dense = route(cue, (good,))
    alternative = route(cue, alternatives=((cue, dense.fused_hit),))
    assert select_context_fact_for_result(alternative, query_clues=[cue]) is good
    wrong_song = route(cue, (good,), sid="another")
    assert select_context_fact_for_result(route(cue, alternatives=((cue, wrong_song.fused_hit),)), query_clues=[cue]) is None
