"""Collapse multiple background clues into one Context song ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.backend.schemas.query import ContextClue
from src.retrieval.context_ranking import CONTEXT_RRF_K, ContextFusedSongHit


@dataclass(frozen=True)
class ContextRouteHit:
    """A single song candidate with the clue and fact candidate that found it.

    The fact in ``fused_hit`` is not display-ready evidence. Step 7 must check
    its relevance to ``clue`` before attaching it to an API result.
    """

    song_id: str
    score: float
    clue: ContextClue
    fused_hit: ContextFusedSongHit
    # Other clues for this song are evidence candidates, never extra RRF votes.
    alternatives: tuple[tuple[ContextClue, ContextFusedSongHit], ...] = ()


def combine_context_clues(
    rankings: Iterable[tuple[ContextClue, Sequence[ContextFusedSongHit]]],
    *,
    limit: int | None = None,
) -> tuple[ContextRouteHit, ...]:
    """Use each song's strongest confidence-adjusted clue rank.

    Repeated/paraphrased clues must not turn into extra paths or accumulate
    bonus for a song. Preserve the first clue on a tie; only the final sorted
    Context list is sent to the outer SearchRouter RRF.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    best: dict[str, ContextRouteHit] = {}
    alternatives: dict[str, list[tuple[ContextClue, ContextFusedSongHit]]] = {}
    for clue, hits in rankings:
        if clue.confidence <= 0:
            continue
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.song_id in seen:
                raise ValueError("Context clue ranking contains a repeated song_id")
            seen.add(hit.song_id)
            alternatives.setdefault(hit.song_id, []).append((clue, hit))
            score = clue.confidence / (CONTEXT_RRF_K + rank)
            previous = best.get(hit.song_id)
            if previous is None or score > previous.score:
                best[hit.song_id] = ContextRouteHit(hit.song_id, score, clue, hit)

    ordered = sorted(best.values(), key=lambda item: -item.score)
    return tuple(
        ContextRouteHit(
            item.song_id, item.score, item.clue, item.fused_hit,
            tuple(alternatives[item.song_id]),
        )
        for item in ordered[:limit]
    )
