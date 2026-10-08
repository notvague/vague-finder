"""A citation must support the event the listener remembers."""

from types import SimpleNamespace

import pytest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import context_evidence_for_result
from src.retrieval.context_query import (
    context_path_weight, expand_context_search_query, has_grounded_media_work_clue,
)
from src.retrieval.context_qdrant_search import ContextFactHit
from src.retrieval.context_ranking import ContextDenseSongHit, ContextFusedSongHit
from src.retrieval.context_route import ContextRouteHit


def _clue(query: str, relation: str, target: str = "") -> ContextClue:
    return ContextClue(target=target, relation=relation, search_query=query,
                       confidence=0.8)


def _route(clues: list[ContextClue], facts: list[tuple[str, str]]) -> ContextRouteHit:
    hits = tuple(ContextFactHit(
        song_id="100", record_id=f"nw:100:{i}", score=1 - i * 0.1,
        fact_text=text, source_url="https://namu.wiki/w/example",
        title="", artists=(), category=category, section="여담",
        quality="ok", source_fact_indices=(i,),
    ) for i, (category, text) in enumerate(facts))
    fused = ContextFusedSongHit(
        "100", 0.02, 1, None, ContextDenseSongHit("100", 1, hits[0]),
        None, hits,
    )
    return ContextRouteHit("100", 0.02, clues[0], fused,
                           tuple((clue, fused) for clue in clues[1:]))


def _evidence(clues: list[ContextClue], facts: list[tuple[str, str]]):
    return context_evidence_for_result(_route(clues, facts), query_clues=clues)


def test_specific_production_clue_cannot_be_replaced_by_generic_ost_fact():
    broad = _clue("드라마 가상작 OST", "삽입곡·배경음악", "가상작")
    specific = _clue("제작진에게 여러 버전을 보내고 전부 퇴짜를 맞았다",
                     "제작·발매 비화")
    generic = ("media_usage", "가상작 마지막 장면에 노래가 BGM으로 흘렀다.")
    production = ("production", "가상작 제작진에게 여러 버전을 보냈지만 전부 퇴짜를 맞았다.")
    assert _evidence([broad, specific], [generic]) is None
    verified = _evidence([broad, specific], [generic, production])
    assert verified is not None and verified.record_id == "nw:100:1"


def test_different_election_is_not_evidence_for_named_city_race():
    clue = _clue("양쪽 서울시장 후보가 같은 노래를 선거 로고송으로 썼다",
                 "삽입곡·배경음악")
    unrelated = ("media_usage", "국회의원 선거에서 어떤 정당이 로고송으로 썼다.")
    matching = ("version", "서울시장 양쪽 후보가 모두 이 노래를 선거 로고송으로 사용했다.")
    assert _evidence([clue], [unrelated]) is None
    assert _evidence([clue], [unrelated, matching]).record_id == "nw:100:1"


def test_unrelated_lyrics_or_artist_career_are_not_evidence_for_comeback():
    clue = _clue("가수가 밴드 해체 후 개 분양업을 하다 곡이 인기를 얻어 음악을 재개했다",
                 "제작·활동에 얽힌 사건")
    unrelated = ("production", "작사가의 옛 연인을 생각하며 만든 노래다.")
    similar_artist = ("production", "가수가 밴드 보컬로 활동했다.")
    relevant = ("influence", "밴드 해체 후 개 분양업을 하던 중 곡이 역주행하자 음악을 재개했다.")
    assert _evidence([clue], [unrelated, similar_artist]) is None
    assert _evidence([clue], [unrelated, similar_artist, relevant]).record_id == "nw:100:2"


def test_other_song_facts_cannot_pass_on_one_generic_overlap():
    clue = _clue("펌프 잇 업 FIESTA EX에도 수록됐고 피에스타 2까지 플레이",
                 "다른 작품에 사용")
    another = ("media_usage", "펌프 잇 업 FIESTA 2에도 수록되었다.")
    exact = ("media_usage", "펌프 잇 업 FIESTA EX에 수록되어 피에스타 2까지 플레이할 수 있었다.")
    assert _evidence([clue], [another]) is None
    assert _evidence([clue], [another, exact]).record_id == "nw:100:1"


def test_context_work_alias_retains_original_words_and_ignores_unrelated_title():
    assert expand_context_search_query("짱구 애니메이션 이별 장면 OST") == (
        "짱구 애니메이션 이별 장면 OST 짱구는 못말려"
    )
    assert expand_context_search_query("짱구는 못말려 장면") == "짱구는 못말려 장면"
    assert expand_context_search_query("짱구 캐릭터 같은 앨범 커버") == (
        "짱구 캐릭터 같은 앨범 커버"
    )


def test_named_media_trial_requires_one_grounded_work_and_media_relation():
    def qualifies(original, clues):
        return has_grounded_media_work_clue(SimpleNamespace(
            original_query=original, context_clues=clues,
        ))

    media = _clue("짱구 애니메이션 이별 장면 OST", "OST 삽입", "짱구 애니메이션")
    assert qualifies("짱구 애니메이션에서 이별 장면 OST로 나왔어", [media])
    assert qualifies("짱구는 못말려에서 창밖의 선생님과 노래가 흘러나오던 장면", [
        _clue("짱구는 못말려 나미리 선생님", "다른 작품에 사용", "짱구는 못말려")
    ])
    assert not qualifies("무한도전 가요제 유재석 참여 곡", [
        _clue("무한도전 가요제 유재석", "참여 곡", "무한도전 가요제")
    ])
    assert not qualifies("짱구 애니메이션 OST", [
        media, _clue("제작진과 버전 문제", "제작·발매 비화")
    ])
    assert not qualifies("어떤 애니에서 OST였대", [media])
    assert not qualifies("짱구 캐릭터처럼 생긴 앨범 표지", [
        _clue("짱구 캐릭터 표지", "삽입곡", "짱구 캐릭터")
    ])

    media_analysis = SimpleNamespace(original_query="짱구 애니메이션 OST로 나왔어",
                                     context_clues=[media])
    generic_analysis = SimpleNamespace(original_query="무한도전 가요제 참여곡",
                                       context_clues=[_clue("무한도전 가요제 유재석",
                                                            "참여 곡", "무한도전 가요제")])
    assert context_path_weight(.5, .8, 1.5, media_analysis) == pytest.approx(.6)
    assert context_path_weight(.5, .8, 1.5, generic_analysis) == pytest.approx(.4)
