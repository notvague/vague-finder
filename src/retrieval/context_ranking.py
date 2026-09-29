"""Rank Namuwiki fact hits by song before later Dense/Sparse fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.retrieval.context_qdrant_search import ContextFactHit


@dataclass(frozen=True)
class ContextDenseSongHit:
    """A song's highest scoring fact and its original evidence.

    ``score`` is a raw Dense similarity; it must not be added to a Sparse
    profile score. A later fusion stage should use the song's Dense rank.
    """

    song_id: str
    score: float
    best_fact: ContextFactHit


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
