"""Context rank fusion with synthetic songs and facts only."""

import pytest

from src.retrieval.context_qdrant_search import (
    ContextFactHit,
    ContextProfileHit,
    ContextQdrantSearch,
    ContextSongCandidates,
)
from src.retrieval.context_ranking import (
    ContextDenseSongHit,
    fuse_context_song_candidates,
)


def _dense(song_id: str, score: float) -> ContextDenseSongHit:
    fact = ContextFactHit(
        song_id=song_id,
        record_id=f"nw:{song_id}:fact",
        score=score,
        fact_text=f"가상 작품에 사용된 곡 {song_id}",
        source_url="https://example.org/test",
        title="가상 제목",
        artists=("가상 가수",),
        category="media_usage",
        section="여담",
        quality="ok",
        source_fact_indices=(0,),
    )
    return ContextDenseSongHit(song_id, score, fact)


def _sparse(song_id: str, score: float) -> ContextProfileHit:
    return ContextProfileHit(song_id, f"nws:{song_id}", score, "가상 제목", ("가상 가수",))


def test_overlapping_song_uses_one_contribution_per_context_rank():
    dense_a, dense_b = _dense("101", 0.91), _dense("202", 0.72)
    sparse_b, sparse_c = _sparse("202", 435.0), _sparse("303", 40.0)
    fused = fuse_context_song_candidates(
        ContextSongCandidates((dense_a, dense_b), (sparse_b, sparse_c))
    )

    assert [hit.song_id for hit in fused] == ["202", "101", "303"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].dense_rank == 2 and fused[0].sparse_rank == 1
    assert fused[0].dense_song is dense_b
    assert fused[0].sparse_profile is sparse_b
    assert fused[0].dense_song.best_fact.record_id == "nw:202:fact"
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[2].dense_song is None and fused[2].dense_rank is None
    assert fused[2].sparse_profile is sparse_c


def test_rrf_uses_ranks_even_when_score_scales_change():
    ids = (("101", "202"), ("202", "303"))
    normal = ContextSongCandidates(
        (_dense(ids[0][0], 0.9), _dense(ids[0][1], 0.2)),
        (_sparse(ids[1][0], 3.0), _sparse(ids[1][1], 0.1)),
    )
    changed = ContextSongCandidates(
        (_dense(ids[0][0], 1e9), _dense(ids[0][1], -1e8)),
        (_sparse(ids[1][0], 1e-6), _sparse(ids[1][1], -1e9)),
    )
    assert [(hit.song_id, hit.score) for hit in fuse_context_song_candidates(normal)] == [
        (hit.song_id, hit.score) for hit in fuse_context_song_candidates(changed)
    ]


def test_limit_is_applied_after_fusion_and_equal_ranks_favor_dense_order():
    candidates = ContextSongCandidates((_dense("101", 0.7),), (_sparse("202", 3.0),))
    assert [hit.song_id for hit in fuse_context_song_candidates(candidates)] == ["101", "202"]
    assert [hit.song_id for hit in fuse_context_song_candidates(candidates, limit=1)] == ["101"]
    assert fuse_context_song_candidates(ContextSongCandidates((), ())) == ()


def test_weights_can_disable_one_path_without_leaking_its_candidate_or_fact():
    candidates = ContextSongCandidates((_dense("101", 0.7),), (_sparse("202", 3.0),))
    sparse_only = fuse_context_song_candidates(candidates, dense_weight=0)
    assert [hit.song_id for hit in sparse_only] == ["202"]
    assert sparse_only[0].dense_song is None
    dense_only = fuse_context_song_candidates(candidates, sparse_weight=0)
    assert [hit.song_id for hit in dense_only] == ["101"]
    assert dense_only[0].sparse_profile is None


@pytest.mark.parametrize("kwargs", [
    {"limit": 0},
    {"rrf_k": 0},
    {"rrf_k": 60.0},
    {"dense_weight": float("nan")},
    {"sparse_weight": -0.1},
    {"dense_weight": 0, "sparse_weight": 0},
])
def test_invalid_fusion_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fuse_context_song_candidates(ContextSongCandidates((), ()), **kwargs)


@pytest.mark.parametrize("candidates, branch", [
    (ContextSongCandidates((_dense("101", 0.5), _dense("101", 0.4)), ()), "Dense"),
    (ContextSongCandidates((), (_sparse("101", 2.0), _sparse("101", 1.0))), "Sparse"),
])
def test_duplicate_song_does_not_get_a_second_contribution(candidates, branch):
    with pytest.raises(ValueError, match=branch):
        fuse_context_song_candidates(candidates)


def test_search_exposes_one_fused_context_path_and_forwards_fetch_widths(monkeypatch):
    candidates = ContextSongCandidates((_dense("101", 0.8),), (_sparse("101", 17.0),))
    calls = []

    def fake_candidates(self, query, *, fact_k, sparse_k, song_k):
        calls.append((query, fact_k, sparse_k, song_k))
        return candidates

    monkeypatch.setattr(ContextQdrantSearch, "search_song_candidates", fake_candidates)
    searcher = object.__new__(ContextQdrantSearch)
    fused = searcher.search_fused_songs(
        "가상 작품 삽입곡", fact_k=70, sparse_k=30, song_k=20, limit=1,
        dense_weight=2.0, sparse_weight=1.0,
    )
    assert calls == [("가상 작품 삽입곡", 70, 30, 20)]
    assert len(fused) == 1 and fused[0].song_id == "101"
    assert fused[0].score == pytest.approx(3 / 61)
    assert fused[0].dense_song.best_fact.fact_text
