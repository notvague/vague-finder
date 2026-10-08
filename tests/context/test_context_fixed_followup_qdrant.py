"""Named-work filtering with real Qdrant, synthetic embeddings and generation changes."""
from pathlib import Path
import pytest
from src.vector_db.qdrant_backend import QdrantVectorClient
from tests.context.test_context_qdrant_load import FakeEmbedder, _sources, _load, _hash


@pytest.fixture
def qdrant():
    client = QdrantVectorClient(path=":memory:")
    yield client
    client.close()


def test_named_media_dense_filter_uses_real_qdrant_and_keeps_sparse_unchanged(
    tmp_path: Path, qdrant: QdrantVectorClient, monkeypatch,
):
    from src.retrieval.context_qdrant_search import ContextQdrantSearch
    sources = _sources(tmp_path)
    _load(tmp_path, sources, qdrant)
    searcher = ContextQdrantSearch(
        qdrant, namespace="test", dense_dir=sources[1], sparse_dir=sources[2],
        state_manifest=tmp_path / "state.json", text_embedder=FakeEmbedder(),
        hash_fn=_hash, expected_dense_dim=4,
    )
    query = "테스트 프로그램 배경음악"
    unrestricted = searcher.search(query, dense_k=4, sparse_k=2)
    assert len(unrestricted.dense_facts) == 4
    snapshot_calls = []
    snapshot = searcher._snapshot
    def tracked_snapshot():
        value = snapshot()
        snapshot_calls.append(value)
        return value
    monkeypatch.setattr(searcher, "_snapshot", tracked_snapshot)
    candidates = searcher.search_song_candidates(
        query, fact_k=4, sparse_k=2, media_targets=("프로그램 101",),
    )
    assert len(snapshot_calls) == 1
    assert len(candidates.dense_facts) == 1
    assert candidates.dense_facts[0].song_id == "101"
    assert "101화" in candidates.dense_facts[0].fact_text
    assert candidates.sparse_songs == unrestricted.sparse_profiles
    fused = searcher.search_fused_songs(
        query, fact_k=4, sparse_k=2, media_targets=("프로그램 101",),
    )
    assert len(snapshot_calls) == 2
    assert len({hit.song_id for hit in fused}) == len(fused) == 2
    assert fused[0].song_id == "101" and fused[0].dense_rank == 1
    assert fused[1].song_id == "202" and fused[1].dense_song is None
    # A nonexistent work never broadens the Dense filter into all facts.
    absent = searcher.search(query, dense_k=4, sparse_k=2,
                             media_targets=("존재하지 않는 작품",))
    assert absent.dense_facts == ()
    assert absent.sparse_profiles == unrestricted.sparse_profiles
    union = searcher.search(query, dense_k=4, sparse_k=2,
                            media_targets=("프로그램 101", "프로그램 202"))
    assert {f.song_id for f in union.dense_facts} == {"101", "202"}
    assert len(union.dense_facts) == 2
    # The name cache follows the published generation, not the old aliases.
    old_key = searcher._media_index_key
    grown = _sources(tmp_path, song_ids=("101", "202", "303"))
    _load(tmp_path, grown, qdrant)
    newer = searcher.search(query, dense_k=6, sparse_k=3,
                            media_targets=("프로그램 303",))
    assert searcher._media_index_key != old_key
    assert len(newer.dense_facts) == 1 and newer.dense_facts[0].song_id == "303"


@pytest.mark.parametrize("targets", [
    ("",), ("a",), ("!!",), ("x" * 81,), (123,), ("aa",) * 5, ["valid"], "valid",
])
def test_invalid_named_media_filter_is_rejected_before_opening_qdrant(targets):
    from src.retrieval.context_qdrant_search import ContextQdrantSearch
    searcher = ContextQdrantSearch.__new__(ContextQdrantSearch)
    with pytest.raises(ValueError, match="media_targets"):
        searcher.search("가상 작품 OST", media_targets=targets)


