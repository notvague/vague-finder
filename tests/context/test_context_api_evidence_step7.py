"""Check the /search response boundary without a live vector DB or LLM."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import (
    get_lyrics_exact_search_service, get_query_analyzer, get_search_router,
)
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.query import ContextClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.context_qdrant_search import ContextFactHit
from src.retrieval.context_ranking import ContextDenseSongHit, ContextFusedSongHit
from src.retrieval.context_route import ContextRouteHit


class Analyzer:
    def analyze(self, query):
        return QueryAnalysis(
            original_query=query, intent_type="mixed", image_english_query="",
            audio_english_query="", context_clues=[ContextClue(
                target="가상 작품", relation="OST 삽입곡",
                search_query="가상 작품 OST 삽입곡", confidence=0.8,
            )] if "작품" in query else [],
        )


def hit(song_id: str, text: str) -> ContextRouteHit:
    clue = Analyzer().analyze("가상 작품 OST").context_clues[0]
    fact = ContextFactHit(
        song_id=song_id, record_id=f"nw:{song_id}:a", score=0.7,
        fact_text=text, source_url="https://namu.wiki/w/test",
        title="임시 제목", artists=("임시 가수",), category="media_usage",
        section="삽입곡", quality="ok", source_fact_indices=(0,),
    )
    fused = ContextFusedSongHit(
        song_id=song_id, score=0.02, dense_rank=1, sparse_rank=1,
        dense_song=ContextDenseSongHit(song_id, 0.7, fact),
        sparse_profile=None, dense_facts=(fact,),
    )
    return ContextRouteHit(song_id, 0.01, clue, fused)


class Router:
    def __init__(self, *, rerank=False):
        self.rerank = rerank
        self.hits = {
            "101": hit("101", "다른 애니메이션에서 이 곡이 OST로 흘렀다."),
            "202": hit("202", "가상 작품에 이 곡이 OST로 삽입되었다."),
        }
        self.calls = []

    async def search(self, analysis, **kwargs):
        self.calls.append(kwargs)
        all_ids = ["101", "202", "303"]
        ids = [song_id for song_id in all_ids
               if song_id not in kwargs.get("exclude_ids", [])]
        kwargs["candidate_ids_out"].extend(ids)
        kwargs["candidate_tracks_out"].extend(
            MatchingTrack(id=song_id, score=0.1, title=f"공식 제목 {song_id}")
            for song_id in ids
        )
        if analysis.context_clues:
            kwargs["context_hits_out"].extend(
                self.hits[song_id] for song_id in ids if song_id in self.hits
            )
        final = ([song_id for song_id in ["202", "101", "303"] if song_id in ids]
                 if self.rerank and kwargs.get("use_rerank") else ids)
        return [MatchingTrack(id=song_id, score=0.1, title=f"공식 제목 {song_id}")
                for song_id in final[:kwargs["top_k"]]]


def test_search_api_exposes_only_verified_fact_on_final_results():
    searcher = Router()
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = Analyzer
    app.dependency_overrides[get_search_router] = lambda: searcher
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: None
    with TestClient(app) as client:
        response = client.post("/search", json={
            "query": "가상 작품 OST", "top_k": 2, "use_rerank": False,
        })
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert [row["id"] for row in rows] == ["101", "202"]
        assert rows[0]["context_evidence"] is None
        assert rows[1]["context_evidence"]["record_id"] == "nw:202:a"
        assert rows[1]["context_evidence"]["fact_text"] == (
            "가상 작품에 이 곡이 OST로 삽입되었다."
        )
        assert rows[1]["context_evidence"]["source_url"] == "https://namu.wiki/w/test"
        assert rows[1]["title"] == "공식 제목 202"
        assert searcher.calls[-1]["context_hits_out"]

        with_explain = client.post("/search", json={
            "query": "가상 작품 OST", "top_k": 2,
            "use_rerank": False, "explain": True,
        })
        assert with_explain.status_code == 200, with_explain.text
        assert with_explain.json()["results"][1]["context_evidence"] == rows[1]["context_evidence"]

        cutoff = client.post("/search", json={"query": "가상 작품 OST", "top_k": 1})
        assert cutoff.status_code == 200
        assert [row["id"] for row in cutoff.json()["results"]] == ["101"]
        assert cutoff.json()["results"][0]["context_evidence"] is None

        rejected = client.post("/search", json={
            "query": "가상 작품 OST", "top_k": 2, "rejected_ids": ["101"],
        })
        assert rejected.status_code == 200, rejected.text
        assert [row["id"] for row in rejected.json()["results"]] == ["202", "303"]
        assert rejected.json()["results"][0]["context_evidence"] is not None
        assert rejected.json()["results"][1]["context_evidence"] is None

        control = client.post("/search", json={"query": "잔잔한 분위기 노래", "top_k": 2})
        assert control.status_code == 200, control.text
        assert all(row["context_evidence"] is None for row in control.json()["results"])


def test_reranking_attaches_evidence_to_shown_song_and_bad_fact_fails_closed():
    searcher = Router(rerank=True)
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = Analyzer
    app.dependency_overrides[get_search_router] = lambda: searcher
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: None
    with TestClient(app) as client:
        reranked = client.post("/search", json={"query": "가상 작품 OST", "top_k": 1})
        assert reranked.status_code == 200, reranked.text
        result = reranked.json()["results"][0]
        assert result["id"] == "202"
        assert result["context_evidence"]["record_id"] == "nw:202:a"

        # Simulate corrupt Qdrant payload. Verification must suppress the
        # citation and still return the canonical search result.
        old = searcher.hits["202"]
        bad = old.fused_hit.dense_facts[0]
        from dataclasses import replace
        corrupt = replace(bad, source_url=123)
        fused = replace(old.fused_hit, dense_facts=(corrupt,))
        searcher.hits["202"] = replace(old, fused_hit=fused)
        healthy_search = client.post("/search", json={"query": "가상 작품 OST", "top_k": 1})
        assert healthy_search.status_code == 200, healthy_search.text
        assert healthy_search.json()["results"][0]["context_evidence"] is None
