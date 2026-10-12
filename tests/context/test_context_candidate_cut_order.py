"""Low-weight Context candidates receive metadata boosts before the pool cut.

The retrieval services return controlled candidates; the real router executes
fusion, catalogue lookup, metadata boosts, exclusions and the final trim.
No embeddings, model downloads or live Qdrant are needed for these tests.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.backend.schemas.query import ContextClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.context_ranking import ContextFusedSongHit
from src.retrieval.context_qdrant_search import ContextProfileHit
from src.retrieval.explain import ExplainRecorder
from src.retrieval.search_router import SearchRouter


def _analysis(*, artist="가상 가수", context=True, image=False, audio=False):
    return QueryAnalysis(
        original_query="가상 가수의 제작 일화가 있는 곡",
        intent_type="mixed", artist_name=artist,
        image_english_query="a blue album cover" if image else "",
        audio_english_query="acoustic piano music" if audio else "",
        has_visual_clue=image,
        context_clues=[ContextClue(
            target="", relation="제작·발매 비화",
            search_query="팬 요청으로 더블 타이틀로 바꾼 곡",
            confidence=0.8,
        )] if context else [],
    )


class _Catalogue:
    def fetch_tracks_by_ids(self, song_ids):
        return {
            song_id: MatchingTrack(
                id=song_id, score=0.0, title=f"정식 제목 {song_id}",
                artist="가상 가수" if song_id.startswith("target") else "다른 가수",
            )
            for song_id in song_ids if song_id != "missing"
        }


class _ContextSearch:
    song_ids = ("target", "missing")

    def search_fused_songs(self, _query, **_kwargs):
        return tuple(ContextFusedSongHit(
            song_id=song_id, score=1 / (60 + rank), dense_rank=None,
            sparse_rank=rank, dense_song=None,
            sparse_profile=ContextProfileHit(
                song_id, f"nws:{song_id}", 1 / rank, "프로필의 임시 제목", ("임시 가수",),
            ),
        ) for rank, song_id in enumerate(self.song_ids, 1))


@pytest.fixture
def router():
    router = SearchRouter.__new__(SearchRouter)
    router._text_svc = _Catalogue()
    router._context_search = _ContextSearch()
    router._context_weight = 0.5
    router._context_fact_k = 100
    router._context_sparse_k = 100
    router._context_named_media_multiplier = 2.0
    router._lyrics_svc = None
    router._reranker = None
    router._pool = ThreadPoolExecutor(max_workers=4)
    router._search_text = lambda _analysis, top_k: [
        MatchingTrack(id=f"text{i:03d}", score=1.0 - i / 1000,
                      title=f"기존 후보 {i}", artist="다른 가수")
        for i in range(top_k)
    ]
    empty = lambda _analysis, _top_k: []
    for name in (
        "_search_image", "_search_audio", "_search_lyrics",
        "_search_performance_clues", "_search_performance_metadata",
        "_search_balanced_semantic", "_search_title_constrained",
        "_search_title_presence", "_search_title_meaning",
    ):
        setattr(router, name, empty)
    yield router
    router._pool.shutdown(wait=True)


def _run(router, analysis=None, **kwargs):
    ids, tracks, routes = [], [], []
    result = asyncio.run(router.search(
        analysis or _analysis(), top_k=10, candidate_k=30,
        use_rerank=False, candidate_ids_out=ids,
        candidate_tracks_out=tracks, context_hits_out=routes, **kwargs,
    ))
    return result, ids, tracks, routes


def test_low_weight_context_only_song_receives_artist_boost_before_cut(router):
    recorder = ExplainRecorder("controlled query")
    result, ids, tracks, routes = _run(router, recorder=recorder)
    assert len(ids) == len(set(ids)) == 30
    assert ids[0] == result[0].id == "target"
    assert "missing" not in ids
    target = next(track for track in tracks if track.id == "target")
    assert (target.title, target.artist) == ("정식 제목 target", "가상 가수")
    assert target.retrieval_score == pytest.approx(0.4 / 61 + 1 / 61)
    assert [route.song_id for route in routes] == ["target"]
    paths = [path for path in recorder.record.get("target").live_paths()
             if path.path == "context"]
    assert len(paths) == 1 and paths[0].delta == pytest.approx(0.4 / 61)


def test_no_reserved_slot_or_extra_context_bonus(router):
    _, ids, _, routes = _run(router, disable_boost=True)
    assert len(ids) == 30 and "target" not in ids and not routes
    _, ids, _, _ = _run(router, _analysis(artist=""))
    assert "target" not in ids


@pytest.mark.parametrize("mode", ["no_clue", "disabled", "empty", "zero_weight", "ablation"])
def test_inactive_context_preserves_candidate_order_and_scores(router, mode):
    baseline, baseline_ids, _, _ = _run(router, use_context=False)
    query, options = _analysis(), {}
    if mode == "no_clue":
        query = _analysis(context=False)
    elif mode == "disabled":
        options["use_context"] = False
    elif mode == "empty":
        router._context_search.song_ids = ()
    elif mode == "zero_weight":
        router._context_weight = 0.0
    else:
        options["force_weights"] = [1.0, 0.0, 0.0]
    actual, ids, _, routes = _run(router, query, **options)
    assert ids == baseline_ids and not routes
    assert [(track.id, track.score) for track in actual] == [
        (track.id, track.score) for track in baseline
    ]


def test_non_context_candidates_keep_their_original_scores(router):
    _, _, baseline, _ = _run(router, use_context=False)
    _, _, active, _ = _run(router)
    old = {track.id: track.retrieval_score for track in baseline}
    assert all(track.retrieval_score == old[track.id]
               for track in active if track.id in old)


@pytest.mark.parametrize("exclude", [
    ["target"], ["text000", "text001", "text002", "text003", "text004"],
    ["target", "text000", "text001", "text002", "text003"],
])
def test_exclusions_refill_the_final_pool_after_boosting(router, exclude):
    _, ids, _, routes = _run(router, exclude_ids=exclude)
    assert len(ids) == len(set(ids)) == 30
    assert not set(ids) & set(exclude)
    assert all(route.song_id not in exclude for route in routes)
    if "target" not in exclude:
        assert ids[0] == "target"


def test_sparse_only_context_candidate_can_enter_but_has_no_fact(router):
    _, ids, _, routes = _run(router)
    assert "target" in ids
    assert routes[0].fused_hit.dense_song is None
    from src.retrieval.context_evidence import context_evidence_for_result
    assert context_evidence_for_result(routes[0]) is None


def test_multiple_context_songs_are_boosted_then_trimmed_once(router):
    router._context_search.song_ids = tuple(f"target{i:03d}" for i in range(40))
    _, ids, tracks, routes = _run(router)
    assert len(ids) == len(set(ids)) == 30
    assert ids == [f"target{i:03d}" for i in range(30)]
    assert [route.song_id for route in routes] == ids
    assert len(tracks) == 30


def test_context_failure_falls_back_to_the_same_existing_results(router):
    _, baseline, _, _ = _run(router, use_context=False)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("controlled publication failure")
    router._context_search.search_fused_songs = unavailable
    _, actual, _, routes = _run(router)
    assert actual == baseline and not routes


def test_existing_three_modalities_keep_one_vote_each_with_context(router):
    def common(_analysis, _top_k):
        return [MatchingTrack(id="common", score=0.9,
                              title="기존 멀티모달 곡", artist="다른 가수")]
    router._search_text = router._search_image = router._search_audio = common
    query = _analysis(image=True, audio=True)
    recorder = ExplainRecorder(query.original_query)
    _, _, baseline, _ = _run(router, query, use_context=False)
    _, ids, active, _ = _run(router, query, recorder=recorder)
    assert ids == ["target", "common"]
    retained = next(track for track in active if track.id == "common")
    assert retained.retrieval_score == baseline[0].retrieval_score
    common_paths = recorder.record.get("common").live_paths()
    assert {path.path for path in common_paths} == {"text_hybrid", "image", "audio"}
    assert sum(path.delta for path in common_paths) == pytest.approx(1 / 61)
