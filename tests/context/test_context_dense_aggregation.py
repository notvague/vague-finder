"""Song-level Context Dense ranking with synthetic, shareable facts."""

from dataclasses import dataclass

import pytest

from src.retrieval.context_ranking import aggregate_dense_facts_by_song


@dataclass(frozen=True)
class Fact:
    song_id: str
    record_id: str
    score: float
    fact_text: str
    source_url: str


def _fact(song_id: str, record_id: str, score: float) -> Fact:
    return Fact(song_id, record_id, score, f"공개 테스트 사실 {record_id}", "https://example.org/test")


def test_selects_max_fact_per_song_and_keeps_its_evidence():
    facts = [
        _fact("101", "first", 0.4),
        _fact("202", "other", 0.7),
        _fact("101", "best", 0.9),
        _fact("101", "same-score", 0.9),
        _fact("303", "last", 0.1),
    ]

    ranked = aggregate_dense_facts_by_song(facts, limit=2)

    assert [(hit.song_id, hit.score) for hit in ranked] == [("101", 0.9), ("202", 0.7)]
    assert ranked[0].best_fact is facts[2]
    assert ranked[0].best_fact.fact_text == "공개 테스트 사실 best"
    assert ranked[0].best_fact.source_url == "https://example.org/test"
    assert len(facts) == 5


def test_equal_song_scores_preserve_first_seen_order():
    facts = [
        _fact("202", "one", 0.6),
        _fact("101", "two", 0.6),
        _fact("202", "three", 0.6),
    ]

    ranked = aggregate_dense_facts_by_song(iter(facts))

    assert [hit.song_id for hit in ranked] == ["202", "101"]
    assert ranked[0].best_fact is facts[0]


def test_empty_and_negative_scores():
    assert aggregate_dense_facts_by_song(()) == ()
    ranked = aggregate_dense_facts_by_song(
        [_fact("101", "low", -0.8), _fact("202", "high", -0.2)]
    )
    assert [hit.song_id for hit in ranked] == ["202", "101"]


@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_nonpositive_limit(limit):
    with pytest.raises(ValueError, match="limit must be positive"):
        aggregate_dense_facts_by_song((), limit=limit)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        aggregate_dense_facts_by_song([_fact("101", "bad", score)])


def test_context_search_can_return_ranked_songs_with_evidence(monkeypatch):
    from src.retrieval.context_qdrant_search import ContextFactHit, ContextQdrantSearch

    def hit(song_id: str, record_id: str, score: float) -> ContextFactHit:
        return ContextFactHit(
            song_id=song_id,
            record_id=record_id,
            score=score,
            fact_text=f"공개 테스트 사실 {record_id}",
            source_url="https://example.org/test",
            title="가상 제목",
            artists=("가상 가수",),
            category="media_usage",
            section="여담",
            quality="ok",
            source_fact_indices=(0,),
        )

    facts = (hit("101", "a", 0.5), hit("101", "b", 0.9), hit("202", "c", 0.6))
    calls = []

    def fake_search(self, query: str, *, limit: int):
        calls.append((query, limit))
        return facts[:limit]

    monkeypatch.setattr(ContextQdrantSearch, "search_dense_facts", fake_search)
    searcher = object.__new__(ContextQdrantSearch)
    ranked = searcher.search_dense_songs("가상 작품 삽입곡", fact_k=3, song_k=2)

    assert calls == [("가상 작품 삽입곡", 3)]
    assert [song.song_id for song in ranked] == ["101", "202"]
    assert ranked[0].best_fact is facts[1]
    assert ranked[0].best_fact.fact_text == "공개 테스트 사실 b"

    with pytest.raises(ValueError, match="song_k must be positive"):
        searcher.search_dense_songs("가상 작품", song_k=0)
