"""A retrieved Context candidate becomes a citation only after relevance checks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import context_evidence_for_result
from src.retrieval.context_qdrant_search import (
    ContextFactHit, ContextProfileHit, ContextSongCandidates,
)
from src.retrieval.context_ranking import (
    ContextDenseSongHit, ContextFusedSongHit, fuse_context_song_candidates,
)
from src.retrieval.context_route import ContextRouteHit, combine_context_clues


def clue(*, target="크레용 신짱", relation="삽입곡·배경음악",
         query="크레용 신짱 OST 삽입곡", confidence=0.8):
    return ContextClue(target=target, relation=relation, search_query=query,
                       confidence=confidence)


def fact(*, song_id="101", record_id=None,
         text="짱구는 못말려에서 선생님과 함께 이 곡이 흐른다.",
         category="media_usage", section="삽입곡", url="https://namu.wiki/w/%EA%B0%80%EC%83%81"):
    return ContextFactHit(
        song_id=song_id, record_id=record_id or f"nw:{song_id}:a", score=0.9,
        fact_text=text, source_url=url, title="가상곡", artists=("가상 가수",),
        category=category, section=section, quality="ok", source_fact_indices=(0,),
    )


def route(*, clue_=None, facts=None, sparse=True, song_id="101"):
    items = tuple(facts) if facts is not None else (fact(song_id=song_id),)
    best = ContextDenseSongHit(song_id, items[0].score, items[0]) if items else None
    fused = ContextFusedSongHit(
        song_id=song_id, score=0.02, dense_rank=1 if items else None,
        sparse_rank=1 if sparse else None, dense_song=best,
        sparse_profile=ContextProfileHit(song_id, f"nws:{song_id}", 0.4,
                                         "가상곡", ("가상 가수",)) if sparse else None,
        dense_facts=items,
    )
    return ContextRouteHit(song_id, 0.01, clue_ or clue(), fused)


def test_named_work_alias_yields_actual_fact_and_source():
    evidence = context_evidence_for_result(route())
    assert evidence is not None
    assert evidence.record_id == "nw:101:a"
    assert evidence.fact_text == fact().fact_text
    assert evidence.source_url == fact().source_url
    assert evidence.category == "media_usage"


def test_rule_fallback_subject_works_without_explicit_target():
    from src.retrieval.query_analyzer import rule_fallback

    analysis = rule_fallback("크레용 신짱 OST 삽입곡")
    assert analysis.context_clues
    evidence = context_evidence_for_result(route(clue_=analysis.context_clues[0]))
    assert evidence is not None
    assert evidence.record_id == "nw:101:a"


@pytest.mark.parametrize("changes", [
    {"text": "다른 작품에서 삽입곡으로 이 곡이 흘렀다."},
    {"text": "짱구는 못말려에서 이 곡을 부르지는 않았다."},
    {"text": "짱구는 못말려의 등장인물을 그림으로 그렸다."},
    {"text": "짱구는 못말려에서 창밖을 보며 OST로 나오지 않았다."},
    {"category": "production"},
    {"song_id": "202", "record_id": "nw:202:wrong"},
    {"record_id": "nw:202:wrong"},
    {"url": ""},
    {"url": "http://namu.wiki/w/test"},
    {"url": "javascript:alert(1)"},
    {"url": "https://namu.wiki.evil.test/w/test"},
    {"url": "https://attacker@namu.wiki/w/test"},
    {"url": "https://namu.wiki/other"},
])
def test_unrelated_or_untrusted_fact_is_hidden(changes):
    assert context_evidence_for_result(route(facts=[fact(**changes)])) is None


def test_section_and_song_title_cannot_stand_in_for_fact_text():
    wrong = fact(text="다른 영화에서 이 곡이 OST로 사용되었다.", section="짱구는 못말려")
    assert context_evidence_for_result(route(facts=[wrong])) is None


def test_work_and_event_must_describe_the_same_clause():
    wrong = fact(text=(
        "짱구는 못말려의 캐릭터가 등장한다. "
        "다른 드라마에서만 이 곡이 OST로 흘렀다."
    ))
    assert context_evidence_for_result(route(facts=[wrong])) is None


def test_specific_scene_must_overlap_fact_text():
    remembered = clue(query="크레용 신짱 OST 삽입곡 창밖을 보며 울던 장면")
    unrelated_scene = fact(text="짱구는 못말려에서 운동회 장면에 OST로 흘렀다.")
    assert context_evidence_for_result(route(clue_=remembered,
                                             facts=[unrelated_scene])) is None
    matching_scene = replace(unrelated_scene, fact_text="짱구는 못말려에서 창밖을 보며 이 곡이 흘렀다.")
    assert context_evidence_for_result(route(clue_=remembered,
                                             facts=[matching_scene])) is not None


def test_wrong_highest_scoring_fact_does_not_block_relevant_second_fact():
    wrong = fact(text="다른 드라마에서 OST로 쓰였다.", record_id="nw:101:top")
    right = replace(fact(), record_id="nw:101:second", score=0.3)
    candidate = route(facts=(wrong, right))
    evidence = context_evidence_for_result(candidate)
    assert candidate.fused_hit.dense_song.best_fact.record_id == "nw:101:top"
    assert evidence is not None and evidence.record_id == "nw:101:second"


def test_sparse_only_cannot_be_a_citation():
    assert context_evidence_for_result(route(facts=())) is None


def test_unknown_relation_and_zero_confidence_abstain():
    assert context_evidence_for_result(route(clue_=clue(relation="알 수 없음"))) is None
    assert context_evidence_for_result(route(clue_=clue(confidence=0))) is None


def test_generic_target_and_generic_relation_cannot_validate_unrelated_fact():
    generic = clue(target="애니메이션", query="애니메이션 삽입곡")
    assert context_evidence_for_result(route(clue_=generic)) is None


def test_performance_detail_can_validate_without_named_target():
    c = clue(target="", relation="축제 앙코르 무대",
             query="한양대 축제에서 앙코르를 못 했다가 다시 무대에 올라 불렀다")
    relevant = fact(text="한양대 축제에서 앙코르를 못 했다가 다시 무대에 올라 불렀다.",
                    category="performance")
    assert context_evidence_for_result(route(clue_=c, facts=[relevant])) is not None
    assert context_evidence_for_result(route(clue_=c, facts=[
        replace(relevant, fact_text="어느 지역 축제에서 앙코르를 불렀다.")])) is None


def test_other_named_relation_accepts_matching_person_and_event():
    c = clue(target="가상 가수", relation="뮤직비디오 NG",
             query="가상 가수 뮤직비디오 NG 안무")
    matching = fact(text="가상 가수의 뮤직비디오에서 NG 안무가 영상에 실렸다.",
                    category="music_video")
    assert context_evidence_for_result(route(clue_=c, facts=[matching])) is not None
    assert context_evidence_for_result(route(clue_=c, facts=[
        replace(matching, category="production")])) is None


@pytest.mark.parametrize("relation,category,text", [
    ("다른 작품에 사용", "media_usage", "가상 작품에 이 곡이 삽입되었다."),
    ("등장곡", "media_usage", "가상 작품에서 이 곡이 등장곡으로 사용됐다."),
    ("뮤직비디오 속 사건", "music_video", "가상 작품의 뮤직비디오에서 촬영했다."),
    ("제작·발매 비화", "production", "가상 작품의 제작 과정에 사용됐다."),
    ("방송·공연 일화", "performance", "가상 작품을 공연에서 불렀다."),
    ("커버·리메이크·답가", "version", "가상 작품을 커버한 버전이다."),
    ("유행·밈", "meme", "가상 작품이 밈으로 유행했다."),
    ("기록·영향", "record", "가상 작품이 차트 1위를 기록했다."),
    ("제작·활동에 얽힌 사건", "production", "가상 작품을 제작했다."),
])
def test_context_relation_families(relation, category, text):
    c = clue(target="가상 작품", relation=relation, query=f"가상 작품 {relation}")
    evidence = context_evidence_for_result(route(clue_=c, facts=[
        fact(text=text, category=category)]))
    assert evidence is not None
    assert evidence.fact_text == text


def test_different_clue_can_validate_same_record_without_extra_rank_vote():
    bad = clue(target="다른 작품", query="다른 작품 OST")
    good = clue(confidence=0.6)
    fused = route().fused_hit
    ranked = combine_context_clues([(bad, (fused,)), (good, (fused,))])
    assert len(ranked) == 1
    assert ranked[0].clue == bad
    assert ranked[0].score == pytest.approx(bad.confidence / 61)
    assert context_evidence_for_result(ranked[0]) is not None


def test_fusion_keeps_ranks_and_all_same_snapshot_facts_for_verification():
    wrong = fact(record_id="nw:101:top", text="다른 드라마에서 OST로 흘렀다.")
    right = replace(fact(), record_id="nw:101:second", score=0.3)
    dense = ContextDenseSongHit("101", wrong.score, wrong)
    sparse = ContextProfileHit("101", "nws:101", 3.0, "가상곡", ("가상 가수",))
    with_facts = ContextSongCandidates((dense,), (sparse,), (wrong, right))
    without = ContextSongCandidates((dense,), (sparse,))
    fused = fuse_context_song_candidates(with_facts)
    assert [(h.song_id, h.score) for h in fused] == [
        (h.song_id, h.score) for h in fuse_context_song_candidates(without)
    ]
    assert [h.record_id for h in fused[0].dense_facts] == [
        "nw:101:top", "nw:101:second"
    ]
    assert context_evidence_for_result(ContextRouteHit("101", 0.01, clue(), fused[0])).record_id == "nw:101:second"
