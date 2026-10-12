"""Shared Context hit types must resolve in runtime annotation inspection."""

from typing import get_type_hints

from src.backend.schemas.search import ContextEvidence
from src.retrieval import context_qdrant_search as qdrant
from src.retrieval import context_ranking as ranking
from src.retrieval.context_evidence import context_evidence_for_result
from src.retrieval.context_route import ContextRouteHit


def test_context_hit_exports_and_annotations_resolve():
    # The public Qdrant imports remain compatible while the hit definitions
    # live in the ranking module, breaking the former circular type import.
    assert qdrant.ContextFactHit is ranking.ContextFactHit
    assert qdrant.ContextProfileHit is ranking.ContextProfileHit
    assert qdrant.ContextSongCandidates is ranking.ContextSongCandidates

    assert get_type_hints(ranking.ContextDenseSongHit)["best_fact"] is ranking.ContextFactHit
    assert get_type_hints(ranking.ContextSongCandidates)["dense_facts"] == tuple[
        ranking.ContextFactHit, ...
    ]
    assert get_type_hints(ranking.fuse_context_song_candidates)["candidates"] is (
        ranking.ContextSongCandidates
    )
    assert get_type_hints(context_evidence_for_result)["hit"] is ContextRouteHit
    assert get_type_hints(context_evidence_for_result)["return"] == ContextEvidence | None
