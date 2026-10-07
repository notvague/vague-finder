"""The real router must not promote generic OST hits for a collapsed memory.

Index adapters are deterministic; the real production router performs candidate
fusion, metadata lookup and rejected-ID refill. All songs/queries are synthetic.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.backend.schemas.query import ContextClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.context_qdrant_search import ContextProfileHit
from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.context_ranking import ContextFusedSongHit
from src.retrieval.search_router import SearchRouter


@pytest.fixture
def router():
    class Catalogue:
        def fetch_tracks_by_ids(self, ids):
            return {sid: MatchingTrack(id=sid, title=f"가상 곡 {sid}",
                                       artist="가상 가수", score=0.0) for sid in ids}

    class Background:
        def __init__(self):
            self.calls = []

        def search_fused_songs(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return tuple(ContextFusedSongHit(
                sid, 0.02, None, rank, None,
                ContextProfileHit(sid, f"nws:{sid}", 0.2, "가상 곡", ()), (),
            ) for rank, sid in enumerate(("old10", "old11", "old12"), start=1))

    value = SearchRouter.__new__(SearchRouter)
    value._text_svc, value._context_search = Catalogue(), Background()
    value._context_weight, value._context_named_media_multiplier = 0.5, 2.0
    value._context_fact_k = value._context_sparse_k = 100
    value._lyrics_svc = value._reranker = None
    value._pool = ThreadPoolExecutor(max_workers=2)
    value._search_text = lambda *_: [MatchingTrack(
        id=f"old{i}", title=f"가상 곡 old{i}", artist="가상 가수", score=1.0 - i / 1000,
    ) for i in range(40)]
    for method in ("_search_image", "_search_audio", "_search_lyrics", "_search_performance_clues",
                   "_search_performance_metadata", "_search_balanced_semantic", "_search_title_constrained",
                   "_search_title_presence", "_search_title_meaning"):
        setattr(value, method, lambda *_: [])
    yield value
    value._pool.shutdown(wait=True)


def analyzed(query, *, collapsed):
    data = dict(context_clues=[dict(target="", relation="삽입곡·배경음악",
                                   search_query="OST 삽입곡", confidence=0.8)])
    data = data if collapsed else apply_context_query_safeguards(query, data)
    return QueryAnalysis(original_query=query, intent_type="mixed", image_english_query="",
                         audio_english_query="", context_clues=[ContextClue(**c) for c in data["context_clues"]])


def run(router, value, **kwargs):
    candidates, hits = [], []
    results = asyncio.run(router.search(value, top_k=10, candidate_k=30, use_rerank=False,
                                        candidate_ids_out=candidates, context_hits_out=hits, **kwargs))
    return results, candidates, hits


@pytest.mark.parametrize("query", (
    "랩으로 유명한 남자 가수가 랩 없이 부른 영화 삽입곡인데 1990년대 후반 곡이었어",
    "군인이 파병 나가는 드라마 삽입곡",
    "특별한 후각을 가진 주인공이 나오는 드라마 삽입곡",
))
def test_parser_specificity_repair_removes_bonus_without_penalizing_existing_search(router, query):
    broken = analyzed(query, collapsed=True)
    _, shifted_ids, _ = run(router, broken)
    assert shifted_ids.index("old9") + 1 > 10  # Reproduce the collapsed-clue regression.
    repaired = analyzed(query, collapsed=False)
    frozen = repaired.model_dump()
    baseline, off_ids, _ = run(router, repaired, use_context=False)
    results, on_ids, hits = run(router, repaired)
    assert on_ids == off_ids and len(on_ids) == 30 and not hits
    assert on_ids.index("old9") + 1 == 10
    assert [(r.id, r.score) for r in results] == [(r.id, r.score) for r in baseline]
    assert repaired.model_dump() == frozen
    assert all(kw["fact_k"] == kw["sparse_k"] == 100 for _, kw in router._context_search.calls)
    assert any("OST" != q and "삽입곡" in q for q, _ in router._context_search.calls)


def test_rejected_ids_are_refilled_after_unsupported_context_promotion_is_removed(router):
    value = analyzed("군인이 파병 나가는 드라마 삽입곡", collapsed=False)
    _, off_ids, _ = run(router, value, use_context=False, exclude_ids=["old0", "old1", "old2"])
    _, on_ids, hits = run(router, value, exclude_ids=["old0", "old1", "old2"])
    assert on_ids == off_ids and len(on_ids) == 30 and not hits
    assert not {"old0", "old1", "old2"}.intersection(on_ids)
