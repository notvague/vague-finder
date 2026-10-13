"""Context joins the existing candidate pool once, before it is trimmed."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.backend.schemas.query import ContextClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.context_ranking import ContextFusedSongHit
from src.retrieval.context_route import combine_context_clues
from src.retrieval.explain import ExplainRecorder
from src.retrieval.search_router import SearchRouter


def clue(query: str = "애니메이션 삽입곡", confidence: float = 0.8) -> ContextClue:
    return ContextClue(
        target="애니메이션", relation="삽입곡", search_query=query,
        confidence=confidence,
    )


def fused(song_id: str) -> ContextFusedSongHit:
    return ContextFusedSongHit(
        song_id=song_id, score=0.02, dense_rank=None, sparse_rank=1,
        dense_song=None, sparse_profile=None,
    )


def analysis(*clues: ContextClue) -> QueryAnalysis:
    return QueryAnalysis(
        original_query="애니메이션 삽입곡", intent_type="mixed",
        image_english_query="", audio_english_query="",
        context_clues=list(clues),
    )


class ContextSearch:
    def __init__(self, song_ids):
        self.song_ids = song_ids
        self.calls = []

    def search_fused_songs(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return tuple(fused(song_id) for song_id in self.song_ids)


class TextService:
    def __init__(self, indexed_ids):
        self.indexed_ids = indexed_ids
        self.calls = []

    def fetch_tracks_by_ids(self, song_ids):
        self.calls.append(song_ids)
        return {
            song_id: MatchingTrack(
                id=song_id, score=0.0, title=f"공식 제목 {song_id}",
                artist="공식 가수",
            )
            for song_id in song_ids if song_id in self.indexed_ids
        }


@pytest.fixture
def router():
    router = SearchRouter.__new__(SearchRouter)
    router._text_svc = TextService({"target", *(f"s{i:03d}" for i in range(100))})
    router._context_search = ContextSearch(["target", "missing"])
    router._context_weight = 1.0
    router._context_fact_k = 100
    router._context_sparse_k = 100
    router._lyrics_svc = None
    router._reranker = None
    router._pool = ThreadPoolExecutor(max_workers=2)
    router._search_text = lambda _analysis, top_k: [
        MatchingTrack(id=f"s{i:03d}", score=1.0 - i / 1000,
                      title=f"텍스트 곡 {i}", artist="텍스트 가수")
        for i in range(top_k)
    ]
    empty = lambda _analysis, _top_k: []
    for path in (
        "_search_image", "_search_audio", "_search_lyrics",
        "_search_performance_clues", "_search_performance_metadata",
        "_search_balanced_semantic", "_search_title_constrained",
        "_search_title_presence", "_search_title_meaning",
    ):
        setattr(router, path, empty)
    yield router
    router._pool.shutdown(wait=True)


def run(router, query, **kwargs):
    candidates = []
    result = asyncio.run(router.search(
        query, top_k=10, candidate_k=30, use_rerank=False,
        candidate_ids_out=candidates, **kwargs,
    ))
    return result, candidates


def test_context_enters_before_pool_cut_with_canonical_metadata(router):
    query = analysis(clue())
    recorder = ExplainRecorder(query.original_query)
    tracks = []
    context_routes = []
    _, ids = run(router, query, recorder=recorder, candidate_tracks_out=tracks,
                 context_hits_out=context_routes)
    assert len(ids) == 30 and len(set(ids)) == 30
    assert "target" in ids and "missing" not in ids
    target = next(item for item in tracks if item.id == "target")
    assert (target.title, target.artist) == ("공식 제목 target", "공식 가수")
    assert [route.song_id for route in context_routes] == ["target"]
    assert context_routes[0].fused_hit.song_id == "target"
    assert router._context_search.calls[0][1] == {"fact_k": 100, "sparse_k": 100}
    assert any(path.path == "context" and path.rank == 1
               and path.delta == pytest.approx(0.8 / 61)
               for path in recorder.record.get("target").live_paths())


def test_no_context_and_disabled_context_keep_old_scores(router):
    plain, plain_ids = run(router, analysis())
    disabled, disabled_ids = run(router, analysis(clue()), use_context=False)
    assert plain_ids == disabled_ids
    assert [(item.id, item.score) for item in plain] == [
        (item.id, item.score) for item in disabled
    ]
    assert router._context_search.calls == []


def test_non_context_song_is_not_penalized(router):
    query = analysis(clue())
    _, baseline = run(router, query, use_context=False)
    _, with_context = run(router, query)
    assert baseline != with_context
    baseline_tracks = []
    context_tracks = []
    asyncio.run(router.search(query, candidate_k=30, top_k=10,
                              use_rerank=False, use_context=False,
                              candidate_tracks_out=baseline_tracks))
    asyncio.run(router.search(query, candidate_k=30, top_k=10,
                              use_rerank=False,
                              candidate_tracks_out=context_tracks))
    old_scores = {track.id: track.retrieval_score for track in baseline_tracks}
    for track in context_tracks:
        if track.id.startswith("s") and track.id in old_scores:
            assert track.retrieval_score == old_scores[track.id]


def test_rejected_songs_are_replaced_without_losing_pool_width(router):
    query = analysis(clue())
    _, initial = run(router, query)
    rejected = initial[:5]
    _, after = run(router, query, exclude_ids=rejected)
    assert len(after) == 30
    assert not set(rejected) & set(after)


def test_same_query_runs_once_and_multiple_clues_are_one_path(router):
    a = clue("A", 0.8)
    b = clue("A", 0.4)
    c = clue("B", 0.7)
    router._context_search.song_ids = ["target"]
    recorder = ExplainRecorder("A")
    run(router, analysis(a, b, c), recorder=recorder)
    assert [query for query, _ in router._context_search.calls] == ["A", "B"]
    assert len([path for path in recorder.record.get("target").live_paths()
                if path.path == "context"]) == 1


def test_merge_repeated_song_keeps_strongest_clue():
    a, b = clue("A", 0.2), clue("B", 0.9)
    hits = combine_context_clues([(a, (fused("song"),)),
                                  (b, (fused("other"), fused("song")))])
    assert hits[0].song_id == "other"
    assert hits[1].song_id == "song"
    assert hits[1].clue == b


def test_empty_high_confidence_clue_does_not_amplify_other_clue(router):
    router._search_text = lambda _analysis, _top_k: []
    router._context_search.search_fused_songs = (
        lambda query, **_kwargs: (fused("target"),) if query == "weak" else ()
    )
    recorder = ExplainRecorder("query")
    run(router, analysis(clue("strong", 1.0), clue("weak", 0.1)),
        recorder=recorder)
    target_paths = [p for p in recorder.record.get("target").live_paths()
                    if p.path == "context"]
    assert len(target_paths) == 1
    assert target_paths[0].delta == pytest.approx(0.1 / 61)


def test_force_weights_stays_a_three_modality_ablation(router):
    run(router, analysis(clue()), force_weights=[1.0, 0.0, 0.0])
    assert router._context_search.calls == []


def test_context_failure_keeps_existing_search_and_explains_failure(router):
    query = analysis(clue())
    _, expected = run(router, query, use_context=False)

    def unavailable(_query, **_kwargs):
        raise RuntimeError("context publication unavailable")

    router._context_search.search_fused_songs = unavailable
    recorder = ExplainRecorder(query.original_query)
    _, actual = run(router, query, recorder=recorder)
    assert actual == expected
    assert "context publication unavailable" in recorder.record.failed_paths["context"]


def test_context_weight_validation():
    with pytest.raises(ValueError, match="context_weight"):
        SearchRouter(None, None, None, type("Client", (), {"Index": lambda *_: None})(),
                     context_weight=float("nan"))


def test_default_candidate_width_is_configurable_but_explicit_request_wins(router):
    router._default_candidate_k = 50
    unconfigured = []
    explicit = []
    query = analysis()
    asyncio.run(router.search(query, top_k=10, candidate_k=None,
                              use_rerank=False, candidate_ids_out=unconfigured))
    asyncio.run(router.search(query, top_k=10, candidate_k=30,
                              use_rerank=False, candidate_ids_out=explicit))
    assert len(unconfigured) == 50
    assert len(explicit) == 30
