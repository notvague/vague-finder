"""Aggregate Namuwiki facts by song and fuse the two Context rankings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.retrieval.context_qdrant_search import (
        ContextFactHit,
        ContextProfileHit,
        ContextSongCandidates,
    )


CONTEXT_RRF_K = 60


@dataclass(frozen=True)
class ContextDenseSongHit:
    """A song's highest scoring fact and its original evidence.

    ``score`` is a raw Dense similarity; it must not be added to a Sparse
    profile score. Context fusion uses the song's Dense rank.
    """

    song_id: str
    score: float
    best_fact: ContextFactHit


@dataclass(frozen=True)
class ContextFusedSongHit:
    """One Context candidate; the score is internal RRF, not a Qdrant score.

    ``dense_song.best_fact`` is only a retrieved fact candidate. A later
    relevance check must approve it before it can be shown as evidence.
    Sparse-only candidates have no fact or source URL.
    """

    song_id: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_song: ContextDenseSongHit | None
    sparse_profile: ContextProfileHit | None
    # Same search snapshot, retained for verification only; never extra votes.
    dense_facts: tuple[ContextFactHit, ...] = ()


def aggregate_dense_facts_by_song(
    facts: Iterable[ContextFactHit], *, limit: int | None = None
) -> tuple[ContextDenseSongHit, ...]:
    """Select one best fact per song, ordered by descending Dense score.

    Equal scores retain the first song's order in the input, and equal-scoring
    facts of the same song retain the first fact as evidence. Apply ``limit``
    after grouping, so duplicate facts never consume a song slot.
    """

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    best_by_song: dict[str, ContextDenseSongHit] = {}
    for fact in facts:
        if not math.isfinite(fact.score):
            raise ValueError("context Dense fact score must be finite")
        previous = best_by_song.get(fact.song_id)
        if previous is None or fact.score > previous.score:
            best_by_song[fact.song_id] = ContextDenseSongHit(
                song_id=fact.song_id,
                score=fact.score,
                best_fact=fact,
            )

    ranked = sorted(best_by_song.values(), key=lambda hit: -hit.score)
    return tuple(ranked[:limit])


def fuse_context_song_candidates(
    candidates: ContextSongCandidates,
    *,
    limit: int | None = None,
    rrf_k: int = CONTEXT_RRF_K,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> tuple[ContextFusedSongHit, ...]:
    """Fuse song ranks with RRF, keeping Context as one downstream path.

    Each song contributes at most once per ranked list. Input order, not raw
    Dense cosine or Sparse BM25 scores, sets the rank. Ties retain Dense-first
    encounter order. Zero weight disables a branch, including its candidates.
    Downstream fusion must use this *one* ranked list, not both input lists.
    """

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    if (
        not math.isfinite(dense_weight)
        or not math.isfinite(sparse_weight)
        or dense_weight < 0
        or sparse_weight < 0
        or (dense_weight == 0 and sparse_weight == 0)
    ):
        raise ValueError("Context RRF weights must be finite, nonnegative, and not both zero")

    scores: dict[str, float] = {}
    dense_by_id: dict[str, tuple[int, ContextDenseSongHit]] = {}
    sparse_by_id: dict[str, tuple[int, ContextProfileHit]] = {}

    for rank, hit in enumerate(candidates.dense_songs, start=1):
        if not hit.song_id or hit.song_id in dense_by_id:
            raise ValueError("Context Dense list needs unique nonempty song_ids")
        if not math.isfinite(hit.score):
            raise ValueError("Context Dense song score must be finite")
        dense_by_id[hit.song_id] = (rank, hit)
        if dense_weight:
            scores[hit.song_id] = dense_weight / (rrf_k + rank)

    for rank, hit in enumerate(candidates.sparse_songs, start=1):
        if not hit.song_id or hit.song_id in sparse_by_id:
            raise ValueError("Context Sparse list needs unique nonempty song_ids")
        if not math.isfinite(hit.score):
            raise ValueError("Context Sparse profile score must be finite")
        sparse_by_id[hit.song_id] = (rank, hit)
        if sparse_weight:
            scores[hit.song_id] = (
                scores.get(hit.song_id, 0.0) + sparse_weight / (rrf_k + rank)
            )

    ranked = sorted(scores, key=lambda song_id: -scores[song_id])
    if limit is not None:
        ranked = ranked[:limit]
    facts_by_song: dict[str, list[ContextFactHit]] = {}
    for fact in candidates.dense_facts:
        if fact.song_id in dense_by_id:
            facts_by_song.setdefault(fact.song_id, []).append(fact)
    result = []
    for song_id in ranked:
        dense_rank, dense_hit = (
            dense_by_id.get(song_id, (None, None)) if dense_weight else (None, None)
        )
        sparse_rank, sparse_hit = (
            sparse_by_id.get(song_id, (None, None)) if sparse_weight else (None, None)
        )
        result.append(ContextFusedSongHit(
            song_id=song_id,
            score=scores[song_id],
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            dense_song=dense_hit,
            sparse_profile=sparse_hit,
            dense_facts=tuple(facts_by_song.get(song_id, ())) if dense_hit else (),
        ))
    return tuple(result)
