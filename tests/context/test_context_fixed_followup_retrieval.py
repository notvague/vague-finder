"""Media descriptions need fact support; named lookups still cast one vote.

All names/facts here are synthetic. Real corpus ranks are measured separately
with the disclosed 106 questions and then the new independent evaluation.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from src.backend.schemas.query import ContextClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.context_evidence import (
    context_candidate_matches_media_description, context_evidence_for_result,
)
from src.retrieval.context_qdrant_search import ContextFactHit, ContextProfileHit
from src.retrieval.context_query import (
    context_media_description_requires_support, context_media_target_terms,
    context_search_queries, has_specific_media_target,
)
from src.retrieval.context_ranking import ContextDenseSongHit, ContextFusedSongHit
from src.retrieval.context_route import ContextRouteHit
from src.retrieval.explain import ExplainRecorder
from src.retrieval.search_router import SearchRouter


def clue(query, *, target="", relation="삽입곡·배경음악", confidence=0.8):
    return ContextClue(target=target, relation=relation,
                       search_query=query, confidence=confidence)


def route(cue, text=None, *, category="media_usage", sid="fictional"):
    fact = None if text is None else ContextFactHit(
        sid, f"nw:{sid}:synthetic", 0.7, text, "https://namu.wiki/w/fictional",
        "가상 곡", ("가상 가수",), category, "여담", "ok", (0,),
    )
    fused = ContextFusedSongHit(
        sid, 0.02, 1 if fact else None, 1,
        ContextDenseSongHit(sid, fact.score, fact) if fact else None,
        ContextProfileHit(sid, f"nws:{sid}", 0.2, "임시 제목", ()),
        (fact,) if fact else (),
    )
    return ContextRouteHit(sid, 0.01, cue, fused)


@pytest.mark.parametrize("target,query", [
    ("", "드라마 OST"), ("겨울 드라마", "겨울 드라마 OST"),
    ("특별한 후각을 가진 주인공", "특별한 후각을 가진 주인공 드라마 삽입곡"),
    ("가상 작품", "다른 작품 OST"),
])
def test_plot_season_or_unsupplied_name_cannot_enable_a_named_filter(target, query):
    cue = clue(query, target=target)
    assert not has_specific_media_target(cue, original_query=query)
    assert context_media_target_terms(cue, original_query=query) == ()
    assert len(context_search_queries(cue, original_query=query)) == 1


@pytest.mark.parametrize("name", ["가상 작품", "가상 작품 드라마", "드라마 가상 작품"])
def test_media_type_is_removed_without_inventing_a_work(name):
    cue = clue(f"{name}에서 이별 장면에 삽입된 노래", target=name)
    assert context_media_target_terms(cue, original_query=cue.search_query) == ("가상 작품",)
    queries = context_search_queries(cue, original_query=cue.search_query)
    assert len(queries) == 2 and queries[0] == cue.search_query
    assert queries[1] == "가상 작품 배경음악 삽입곡"


def test_vetted_work_equivalence_keeps_the_original_scene_query():
    cue = clue("짱구 애니메이션 이별 장면 OST", target="짱구 애니메이션", relation="OST 삽입")
    before = cue.model_dump()
    assert context_media_target_terms(cue, original_query=cue.search_query) == (
        "짱구는 못말려", "크레용 신짱",
    )
    assert context_search_queries(cue, original_query=cue.search_query) == (
        "짱구 애니메이션 이별 장면 OST 짱구는 못말려", "짱구는 못말려 OST 삽입곡",
    )
    assert cue.model_dump() == before


def test_rule_clue_without_target_uses_only_literal_work_plus_relation_syntax():
    cue = clue("크레용 신짱 OST 삽입곡")
    assert context_media_target_terms(cue, original_query=cue.search_query)
    assert len(context_search_queries(cue, original_query=cue.search_query)) == 2


@pytest.mark.parametrize("query", [
    "겨울 드라마 OST",
    "옛 동네 골목을 배경으로 한 추억 드라마에 삽입된 노래",
    "특별한 후각을 가진 주인공이 나오는 드라마 삽입곡",
    "랩으로 유명한 여자 가수가 랩 없이 부른 드라마 삽입곡",
    "2000년대 초반 겨울 드라마 OST였는데",
])
@pytest.mark.parametrize("text", [None, "가상 드라마에 OST로 사용되었다."])
def test_generic_profile_or_unrelated_ost_fact_cannot_promote_a_detailed_recollection(query, text):
    cue = clue(query)
    hit = route(cue, text)
    assert context_media_description_requires_support(cue)
    assert not context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


def test_two_distinct_details_in_one_use_fact_allow_promotion_and_citation():
    cue = clue("후각 탐정 드라마에 삽입된 노래")
    hit = route(cue, "후각 탐정 드라마의 배경음악으로 이 곡이 삽입되었다.")
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]).record_id == hit.fused_hit.dense_song.best_fact.record_id


@pytest.mark.parametrize("text", [
    "후각 탐정이 등장하는 드라마다. 이 곡은 다른 영화의 배경음악으로 사용되었다.",
    "후각 탐정이 등장하는 드라마다; 이 곡은 다른 영화의 OST로 사용되었다.",
    "후각 탐정 드라마에는 이 곡이 삽입되지 않았다.",
])
def test_details_elsewhere_or_a_negated_use_do_not_substantiate_the_event(text):
    cue = clue("후각 탐정 드라마에 삽입된 노래")
    hit = route(cue, text)
    assert not context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


def test_ost_poll_is_not_a_use_fact_even_when_its_heading_says_media_usage():
    cue = clue("가상 작품 OST", target="가상 작품")
    hit = route(cue, "가상 작품 OST는 드라마 OST 인기 조사에서 1위를 차지했다.")
    assert context_evidence_for_result(hit, query_clues=[cue]) is None
    valid = route(cue, "가상 작품의 OST로 사용되었다.")
    assert context_evidence_for_result(valid, query_clues=[cue]) is not None


def test_a_production_fact_cannot_bypass_an_unsupported_media_description():
    media = clue("옛 동네 골목 추억 드라마 삽입곡")
    production = clue("데모 녹음 기획 일화", relation="제작·발매 비화")
    hit = route(production, "데모를 녹음하여 기획한 곡이다.", category="production")
    assert context_candidate_matches_media_description(hit, query_clues=[production])
    assert not context_candidate_matches_media_description(hit, query_clues=[media, production])


@pytest.mark.parametrize("cue", [
    clue("드라마 OST"), clue("드라마 OST 삽입곡"),
    clue("팬 요청으로 더블 타이틀을 바꾼 곡", relation="제작·발매 비화"),
    clue("가상 작품 OST", target="가상 작품"),
])
def test_broad_or_named_or_non_media_queries_keep_sparse_only_candidates(cue):
    hit = route(cue)
    assert context_candidate_matches_media_description(hit, query_clues=[cue])
    assert context_evidence_for_result(hit, query_clues=[cue]) is None


def test_support_from_another_snapshot_is_allowed_only_for_the_same_song():
    cue = clue("후각 탐정 드라마에 삽입된 노래")
    sparse = route(cue)
    dense = route(cue, "후각 탐정 드라마의 배경음악으로 이 곡이 삽입되었다.")
    alternative = replace(sparse, alternatives=((cue, dense.fused_hit),))
    assert context_candidate_matches_media_description(alternative, query_clues=[cue])
    other_song = route(cue, dense.fused_hit.dense_song.best_fact.fact_text, sid="other")
    wrong = replace(sparse, alternatives=((cue, other_song.fused_hit),))
    assert not context_candidate_matches_media_description(wrong, query_clues=[cue])


@pytest.fixture
def router():
    class Catalogue:
        def fetch_tracks_by_ids(self, ids):
            return {sid: MatchingTrack(id=sid, title=f"정식 제목 {sid}", score=0.0,
                                       artist="가상 가수") for sid in ids}

    class Search:
        def __init__(self):
            self.calls = []
            self.hits = ()

        def search_fused_songs(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return self.hits

    value = SearchRouter.__new__(SearchRouter)
    value._text_svc, value._context_search = Catalogue(), Search()
    value._context_weight, value._context_named_media_multiplier = 0.5, 2.0
    value._context_fact_k = value._context_sparse_k = 100
    value._lyrics_svc = value._reranker = None
    value._pool = ThreadPoolExecutor(max_workers=2)
    value._search_text = lambda _query, _limit: [
        MatchingTrack(id=f"old{i}", title=f"가상 기존 곡 {i}", artist="가상 가수", score=1.0 - i / 1000)
        for i in range(40)
    ]
    for method in ("_search_image", "_search_audio", "_search_lyrics", "_search_performance_clues",
                   "_search_performance_metadata", "_search_balanced_semantic", "_search_title_constrained",
                   "_search_title_presence", "_search_title_meaning"):
        setattr(value, method, lambda _query, _limit: [])
    yield value
    value._pool.shutdown(wait=True)


def analysis(cue):
    return QueryAnalysis(original_query=cue.search_query, intent_type="mixed",
                         image_english_query="", audio_english_query="", context_clues=[cue])


def run(router, value, **kwargs):
    ids, hits = [], []
    results = asyncio.run(router.search(value, top_k=10, candidate_k=30, use_rerank=False,
                                        candidate_ids_out=ids, context_hits_out=hits, **kwargs))
    return results, ids, hits


def test_filtered_promotions_preserve_all_existing_candidates_and_scores(router):
    cue = clue("특별한 후각을 가진 주인공 드라마 삽입곡")
    router._context_search.hits = tuple(route(cue, sid=f"new{i}").fused_hit for i in range(20))
    baseline, old_ids, _ = run(router, analysis(cue), use_context=False)
    results, ids, hits = run(router, analysis(cue))
    assert ids == old_ids and len(ids) == 30 and not hits
    assert [(r.id, r.score) for r in results] == [(r.id, r.score) for r in baseline]


def test_focused_named_query_is_filtered_and_still_gives_one_outer_vote(router):
    cue = clue("짱구 애니메이션 이별 장면 OST", target="짱구 애니메이션", relation="OST 삽입")
    hit = route(cue, "짱구는 못말려의 이별 장면에 이 곡이 OST로 삽입되었다.")
    router._context_search.hits = (hit.fused_hit,)
    router._search_text = lambda *_: []
    recorder = ExplainRecorder(cue.search_query)
    before = analysis(cue)
    frozen = before.model_dump()
    results, ids, hits = run(router, before, recorder=recorder)
    assert ids == [hit.song_id] and len(hits) == 1 and results[0].title == "정식 제목 fictional"
    calls = router._context_search.calls
    assert len(calls) == 2
    assert calls[0][1] == {"fact_k": 100, "sparse_k": 100}
    assert calls[1][1] == {"fact_k": 100, "sparse_k": 100,
                          "media_targets": ("짱구는 못말려", "크레용 신짱")}
    paths = [p for p in recorder.record.get(hit.song_id).live_paths() if p.path == "context"]
    assert len(paths) == 1 and paths[0].delta == pytest.approx(0.5 * 0.8 * 2 / 61)
    assert before.model_dump() == frozen
    assert context_evidence_for_result(hits[0], query_clues=[cue]) is not None


def test_rejected_ids_are_still_refilled_and_never_returned(router):
    cue = clue("드라마 OST")
    router._context_search.hits = (route(cue).fused_hit,)
    _, ids, _ = run(router, analysis(cue), exclude_ids=["old0", "old1", "old2"])
    assert len(ids) == 30 and not {"old0", "old1", "old2"} & set(ids)
