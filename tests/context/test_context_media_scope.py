"""Equivalent relations and literal work names receive identical protections.

Use invented works/songs, never evaluation IDs or answer lists. Test the real
parser, evidence boundary and router with controlled index adapters.
"""

from dataclasses import replace

import pytest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import (
    context_candidate_matches_media_description, context_evidence_for_result,
)
from src.retrieval.context_media_match import media_name_in_text
from src.retrieval.context_query import (
    apply_context_query_safeguards, context_media_description_requires_support,
    is_media_usage_relation,
)
from tests.context.test_context_fixed_followup_retrieval import (
    analysis, clue, route, router, run,
)


RELATIONS = ("삽입곡", "OST", "다른 작품에 사용", "쓰인", "쓰였던", "사용된", "나온", "나오던")
DESCRIPTIONS = (
    "유령 탐정이 나오는 드라마에 쓰인 곡",
    "재즈 연주자들이 나오는 영화에 쓰인 곡",
    "겨울 골목 이별 장면 드라마 삽입곡",
)


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("description", DESCRIPTIONS)
def test_equivalent_usage_verbs_cannot_bypass_qualified_fact_checks(relation, description):
    cue = clue(description, relation=relation)
    assert is_media_usage_relation(relation)
    assert context_media_description_requires_support(cue)
    for hit in (route(cue), route(cue, "다른 드라마의 OST로 사용되었다.")):
        assert not context_candidate_matches_media_description(hit, query_clues=[cue])
        assert context_evidence_for_result(hit, query_clues=[cue]) is None


@pytest.mark.parametrize("relation", RELATIONS)
def test_supported_descriptions_keep_the_fact_and_support_synonymous_relations(relation):
    cue = clue("유령 탐정 드라마에 쓰인 곡", relation=relation)
    hit = route(cue, "유령 탐정 드라마의 OST로 사용되었다.")
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is not None


@pytest.mark.parametrize("verb", ("쓰인", "쓰였던", "사용된", "나온"))
@pytest.mark.parametrize("rewrite", ("드라마에 {verb} 곡", "유령 탐정이 나오는 드라마에 {verb} 곡"))
def test_model_only_clues_cannot_erase_qualified_user_details(verb, rewrite):
    query = f"유령 탐정이 나오는 드라마에 {verb} 곡"
    raw = dict(context_clues=[dict(target="", relation=verb,
                                  search_query=rewrite.format(verb=verb), confidence=0.8)])
    result = apply_context_query_safeguards(query, raw)
    assert result["context_clues"]
    for item in result["context_clues"]:
        cue = ContextClue(**item)
        assert "유령" in cue.search_query and "탐정" in cue.search_query
        assert not context_candidate_matches_media_description(route(cue), query_clues=[cue])


@pytest.mark.parametrize("query", ("드라마 OST 삽입곡", "영화에 쓰인 곡", "드라마에 나온 노래", "게임 배경음악"))
def test_broad_enumeration_still_supports_sparse_recall_without_fake_work_names(query):
    cue = clue(query, relation="쓰인" if "쓰인" in query else "삽입곡")
    assert not context_media_description_requires_support(cue)
    hit = route(cue)
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("terms", ((), ("ost", "다른_작품"), ("가상작품2",), ("가상작품불",)))
def test_named_work_rejects_generic_or_other_work_profiles(relation, terms):
    cue = clue("가상작품 드라마 OST", target="가상작품", relation=relation)
    hit = route(cue, sparse_terms=terms)
    assert not context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


@pytest.mark.parametrize("terms", (("가상_작품",), ("가상 작품 OST",), ("가상작품",)))
def test_literal_work_in_profile_preserves_sparse_only_recall_but_not_citation(terms):
    cue = clue("가상 작품 드라마 OST", target="가상 작품")
    hit = route(cue, sparse_terms=terms)
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


def test_literal_work_in_dense_fact_can_supply_a_retrieval_anchor():
    cue = clue("가상 작품 드라마 OST", target="가상 작품")
    hit = route(cue, "가상 작품의 OST로 사용되었다.")
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is not None


def test_wrong_song_cannot_supply_the_profile_or_fact_anchor():
    cue = clue("가상 작품 OST", target="가상 작품")
    no_match = route(cue)
    wrong = route(cue, "가상 작품의 OST로 사용되었다.", sid="different", sparse_terms=("가상_작품",))
    hit = replace(no_match, alternatives=((cue, wrong.fused_hit),))
    assert not context_candidate_matches_media_description(hit, query_clues=[cue])


@pytest.mark.parametrize("name,text,expected", (
    ("가상 작품", "가상_작품", True),
    ("가상 작품", "가상작품에서 흐른 노래", True),
    ("가상 작품", "가상 작품의 삽입곡", True),
    ("가상 작품", "가상 작품불 OST", False),
    ("가상 작품", "새가상 작품 OST", False),
    ("MY SHOW", "My_Show", True),
    ("MY SHOW", "MY SHOWCASE", False),
    ("ＡＢ", "ab OST", True),
    ("東京", "東京の物語", False),
    ("東京", "東京 OST", True),
    ("Café", "Café OST", True),
    ("Café", "Caféteria OST", False),
    ("가상 작품", "字가상 작품 OST", False),
    ("a", "a OST", False),
))
def test_name_matching_handles_spacing_particles_case_and_distinct_titles(name, text, expected):
    assert media_name_in_text(name, text) is expected


@pytest.mark.parametrize("relation", ("쓰인", "나온", "사용된", "삽입곡"))
def test_router_preserves_the_entire_old_pool_when_a_plot_has_no_fact_support(router, relation):
    cue = clue(DESCRIPTIONS[0], relation=relation)
    router._context_search.hits = tuple(route(cue, sid=f"new{i}").fused_hit for i in range(10))
    query = analysis(cue)
    frozen = query.model_dump()
    off, old_ids, _ = run(router, query, use_context=False)
    on, new_ids, hits = run(router, query)
    assert new_ids == old_ids and len(new_ids) == 30 and not hits
    assert [(t.id, t.score) for t in off] == [(t.id, t.score) for t in on]
    assert query.model_dump() == frozen


def test_named_router_filters_unrelated_profile_but_retains_related_sparse_candidate_once(router):
    cue = clue("가상 작품 OST", target="가상 작품")
    router._context_search.hits = (
        route(cue, sid="unrelated", sparse_terms=("ost",)).fused_hit,
        route(cue, sid="related", sparse_terms=("가상_작품",)).fused_hit,
    )
    _, ids, hits = run(router, analysis(cue), exclude_ids=["old0", "old1", "old2"])
    assert len(ids) == 30 and len(ids) == len(set(ids))
    assert "related" in ids and "unrelated" not in ids
    assert not {"old0", "old1", "old2"}.intersection(ids)
    assert [hit.song_id for hit in hits] == ["related"]
    assert context_evidence_for_result(hits[0], query_clues=[cue]) is None
