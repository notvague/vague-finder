"""경로 깊이(path_k)를 최종 후보 수와 떼어 넓힌다.

여기서 고정하는 것
1. path_k를 주지 않거나 원래 깊이와 같게 주면 결과가 그대로다.
2. path_k를 넓혀도 최종 후보 수(candidate_k)는 그대로다.
3. 깊은 곳의 곡은 가산을 받아 후보에 들어올 수 있고, 원래 있던 곡의 점수는 바뀌지 않는다.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.search_router import SearchRouter

DEEP_FEMALE = "s35"  # 텍스트 35위 — 기본 깊이 30에서는 경로에 잡히지 않는다


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="여자 가수의 잔잔한 노래",
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
        vocal_gender="여성",
    )


def _hits() -> List[MatchingTrack]:
    return [
        MatchingTrack(
            id=f"s{i}", score=1.0 - i / 100, title=f"곡{i}", artist="A",
            vocal_gender="여성" if f"s{i}" == DEEP_FEMALE else "남성",
        )
        for i in range(1, 41)
    ]


def _run(path_k: Optional[int]) -> tuple[List[str], dict, List[int]]:
    router = SearchRouter.__new__(SearchRouter)
    for attr in ("_text_svc", "_img_emb", "_audio_emb", "_img_idx", "_audio_idx",
                 "_lyrics_svc", "_reranker"):
        setattr(router, attr, None)
    router._pool = ThreadPoolExecutor(max_workers=2)
    asked: List[int] = []

    def text(analysis, k):
        asked.append(k)
        return [t.model_copy() for t in _hits()[:k]]

    router._search_text = text
    for path in ("_search_image", "_search_audio", "_search_balanced_semantic",
                 "_search_performance_clues", "_search_performance_metadata",
                 "_search_title_constrained", "_search_title_presence",
                 "_search_title_meaning"):
        setattr(router, path, lambda a, k: [])
    pool: List[str] = []
    tracks: List[MatchingTrack] = []
    try:
        asyncio.run(router.search(
            _analysis(), top_k=10, use_rerank=False, candidate_k=30,
            candidate_ids_out=pool, candidate_tracks_out=tracks, path_k=path_k,
        ))
    finally:
        router.shutdown()
    return pool, {t.id: t.retrieval_score for t in tracks}, asked


def test_default_depth_is_unchanged() -> None:
    pool, scores, asked = _run(None)
    same_pool, same_scores, _ = _run(30)
    assert asked == [30]
    assert (pool, scores) == (same_pool, same_scores)
    assert DEEP_FEMALE not in pool


def test_deeper_paths_keep_candidate_count() -> None:
    base_pool, base_scores, _ = _run(None)
    pool, scores, asked = _run(60)
    assert asked == [60]
    assert len(pool) == len(base_pool) == 30
    assert DEEP_FEMALE in pool, "성별 가산을 받는 깊은 곡이 후보에 들어와야 한다"
    for song_id in set(pool) & set(base_pool):
        assert scores[song_id] == base_scores[song_id], song_id
