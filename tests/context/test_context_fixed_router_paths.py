"""Observe the real router's modality methods through its worker pool.

Embedders/indices are injected deterministic adapters; no model downloads or
network calls are needed. Model inference coverage is measured by the CLI.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from experiments.namuwiki.context_fixed_core import inspect_path
from experiments.namuwiki.evaluate_context_fixed import _arm
from src.backend.schemas.query import ModalityWeights, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.search_router import SearchRouter


class Embedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self, queries, **kwargs):
        self.calls.append((queries, kwargs))
        return np.array([[1.0, 0.0]], dtype=float)


class Index:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("injected DB failure")
        return {"matches": [dict(id="answer", score=0.8, metadata={"title": "synthetic title"})]}


@pytest.fixture
def router():
    value = SearchRouter.__new__(SearchRouter)
    value._pool = ThreadPoolExecutor(max_workers=2)
    value._img_emb, value._audio_emb = Embedder(), Embedder()
    value._img_idx, value._audio_idx = Index(), Index()
    value._lyrics_svc = value._reranker = value._context_search = None
    class Text:
        def track_from_match(self, match):
            return MatchingTrack(id=match["id"], score=match["score"], title=match["metadata"]["title"])
    value._text_svc = Text()
    for path in (
        "_search_text", "_search_lyrics", "_search_performance_clues",
        "_search_performance_metadata", "_search_balanced_semantic", "_search_title_constrained",
        "_search_title_presence", "_search_title_meaning",
    ):
        setattr(value, path, lambda *_: [])
    yield value
    value._pool.shutdown(wait=True)


def query(kind):
    return QueryAnalysis(original_query="synthetic modality query", intent_type="mixed",
                         confidence=0.9, has_visual_clue=kind == "image",
                         image_english_query="a white album cover" if kind == "image" else "",
                         audio_english_query="soft piano music" if kind == "audio" else "",
                         modality_weights=ModalityWeights(text=0.0, image=float(kind == "image"), audio=float(kind == "audio")))


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_router_worker_transfers_timing_to_actual_embedding_and_query_methods(router, kind):
    result = asyncio.run(_arm(router, query(kind), False))
    check = inspect_path(result["spans"], kind)
    assert check.verified
    assert result["top"] == ["answer"]
    embedder = router._img_emb if kind == "image" else router._audio_emb
    index = router._img_idx if kind == "image" else router._audio_idx
    assert len(embedder.calls) == len(index.calls) == 1
    assert index.calls[0]["include_metadata"] is True
    assert result["contributions"]["answer"][0]["path"] == kind


def test_real_router_records_empty_prompt_skip_as_missing_coverage(router):
    value = query("image")
    value.image_english_query = ""
    result = asyncio.run(_arm(router, value, False))
    assert not inspect_path(result["spans"], "image").verified
    assert not router._img_idx.calls and not router._img_emb.calls


def test_real_router_catches_db_error_but_coverage_still_fails(router):
    router._img_idx.fail = True
    result = asyncio.run(_arm(router, query("image"), False))
    check = inspect_path(result["spans"], "image")
    assert check.executed and check.queried and not check.verified
    assert "TimeoutError" in check.error
    assert result["top"] == []
