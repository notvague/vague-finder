from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import models

from src.embedding.context_artifacts import ContextArtifactStore
from src.embedding.context_dense import embed_context_dense
from src.embedding.context_sparse import build_context_bm25
from src.vector_db.context_qdrant import load_context_qdrant
from src.vector_db.qdrant_backend import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantVectorClient,
)


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_passages(self, passages, **_kwargs):
        rows = []
        for index, text in enumerate(passages):
            seed = float(sum(text.encode("utf-8")) % 89 + index + 1)
            vector = np.arange(seed, seed + self.dim, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows).astype(np.float32)


def _hash(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


def _song(song_id: str) -> dict:
    return {
        "id": song_id,
        "metadata": {
            "title": f"테스트곡 {song_id}",
            "artist": [f"가수 {song_id}"],
        },
        "namuwiki": {
            "schema_version": "namuwiki_v3",
            "status": "ok",
            "source_url": f"https://namu.wiki/w/test-{song_id}",
            "collected_at": "2026-09-21T00:00:00+00:00",
            "facts": [
                {
                    "category": "media_usage",
                    "section": "여담 > 삽입곡",
                    "text": f"테스트 프로그램 {song_id}화의 배경음악으로 사용되었다.",
                },
                {
                    "category": "production",
                    "section": "여담",
                    "text": f"프로듀서 {song_id}이 데모를 다시 녹음해 완성했다.",
                },
            ],
            "error_code": None,
        },
    }


def _sources(tmp_path: Path, song_ids=("101", "202"), *, pending: int = 0):
    context = tmp_path / "context"
    dense = tmp_path / "dense"
    sparse = tmp_path / "sparse"
    songs = [_song(song_id) for song_id in song_ids]
    store = ContextArtifactStore(context)
    with store.writer():
        record_count = 0
        for song in songs:
            record_count += int(store.sync_song(song)["record_count"])
        store.publish_manifest(
            coverage={
                "scope_total": len(songs) + pending,
                "terminal_meta": len(songs),
                "artifact_ready": len(songs),
                "pending": pending,
                "completion_percent": round(
                    len(songs) / (len(songs) + pending) * 100.0, 1
                ),
                "context_record_count": record_count,
                "status_counts": {"ok": len(songs)},
                "coverage_error_count": 0,
                "coverage_errors": [],
            },
            active_song_ids=set(song_ids),
        )
    embed_context_dense(
        context_dir=context,
        output_dir=dense,
        expected_dim=4,
        embedder=FakeEmbedder(),
    )
    build_context_bm25(
        context_dir=context,
        output_dir=sparse,
        hash_fn=_hash,
    )
    return context, dense, sparse


@pytest.fixture()
def qdrant():
    client = QdrantVectorClient(path=":memory:")
    yield client
    client.close()


def _aliases(client: QdrantVectorClient) -> dict[str, str]:
    return {
        alias.alias_name: alias.collection_name
        for alias in client.client.get_aliases().aliases
    }


def _load(tmp_path: Path, sources, client, **kwargs):
    context, dense, sparse = sources
    return load_context_qdrant(
        context_dir=context,
        dense_dir=dense,
        sparse_dir=sparse,
        state_manifest=tmp_path / "state.json",
        namespace="test",
        expected_dense_dim=4,
        batch_size=2,
        vector_client=client,
        **kwargs,
    )


def test_dry_run_validates_without_opening_or_writing_qdrant(tmp_path: Path):
    sources = _sources(tmp_path, pending=3)
    state = tmp_path / "state.json"

    result = load_context_qdrant(
        context_dir=sources[0],
        dense_dir=sources[1],
        sparse_dir=sources[2],
        state_manifest=state,
        namespace="test",
        expected_dense_dim=4,
        qdrant_path=str(tmp_path / "must-not-exist"),
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["qdrant_opened"] is False
    assert result["dense_record_count"] == 4
    assert result["sparse_profile_count"] == 2
    assert not (tmp_path / "must-not-exist").exists()
    assert not state.exists()


def test_separate_fact_and_song_collections_are_atomic_and_reusable(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
):
    sources = _sources(tmp_path)
    first = _load(tmp_path, sources, qdrant)

    assert first["action"] == "build"
    aliases = _aliases(qdrant)
    assert aliases[first["dense_alias"]] == first["dense_collection"]
    assert aliases[first["sparse_alias"]] == first["sparse_collection"]
    assert qdrant.client.count(first["dense_alias"], exact=True).count == 4
    assert qdrant.client.count(first["sparse_alias"], exact=True).count == 2

    dense_hits = qdrant.client.query_points(
        collection_name=first["dense_alias"],
        query=[0.5, 0.5, 0.5, 0.5],
        using=DENSE_VECTOR,
        limit=4,
        with_payload=True,
    ).points
    assert len(dense_hits) == 4
    assert {hit.payload["point_kind"] for hit in dense_hits} == {"context_fact"}
    assert {hit.payload["song_id"] for hit in dense_hits} == {"101", "202"}

    sparse_hits = qdrant.client.query_points(
        collection_name=first["sparse_alias"],
        query=models.SparseVector(indices=[_hash("배경음악")], values=[1.0]),
        using=SPARSE_VECTOR,
        limit=2,
        with_payload=True,
    ).points
    assert all(hit.payload["point_kind"] == "context_song_profile" for hit in sparse_hits)

    collections_before = {
        collection.name for collection in qdrant.client.get_collections().collections
    }
    second = _load(tmp_path, sources, qdrant)
    collections_after = {
        collection.name for collection in qdrant.client.get_collections().collections
    }
    assert second["action"] == "reuse"
    assert collections_after == collections_before


def test_sparse_profiles_feed_song_candidates_without_becoming_fact_evidence(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
):
    from src.retrieval.context_qdrant_search import ContextQdrantSearch

    sources = _sources(tmp_path)
    _load(tmp_path, sources, qdrant)
    searcher = ContextQdrantSearch(
        qdrant,
        namespace="test",
        dense_dir=sources[1],
        sparse_dir=sources[2],
        state_manifest=tmp_path / "state.json",
        text_embedder=FakeEmbedder(),
        hash_fn=_hash,
        expected_dense_dim=4,
    )

    candidates = searcher.search_song_candidates(
        "테스트 프로그램 배경음악", fact_k=1, sparse_k=2, song_k=1
    )

    assert len(candidates.dense_songs) == 1
    assert candidates.dense_songs[0].best_fact.fact_text
    assert candidates.dense_songs[0].best_fact.source_url
    assert candidates.dense_facts
    assert candidates.dense_songs[0].best_fact in candidates.dense_facts
    assert {hit.song_id for hit in candidates.sparse_songs} == {"101", "202"}
    assert all(hit.profile_id and hit.title and hit.artists for hit in candidates.sparse_songs)
    assert any(
        hit.song_id not in {dense.song_id for dense in candidates.dense_songs}
        for hit in candidates.sparse_songs
    )
    assert all(not hasattr(hit, "fact_text") for hit in candidates.sparse_songs)
    assert all(not hasattr(hit, "source_url") for hit in candidates.sparse_songs)

    fused = searcher.search_fused_songs(
        "테스트 프로그램 배경음악", fact_k=1, sparse_k=2
    )
    assert len(fused) == 2
    assert {hit.song_id for hit in fused} == {hit.song_id for hit in candidates.sparse_songs}
    assert fused[0].dense_song is not None and fused[0].sparse_profile is not None
    assert fused[0].dense_facts
    assert fused[0].dense_song.best_fact in fused[0].dense_facts
    assert fused[1].dense_song is None and fused[1].sparse_profile is not None
    assert fused[0].score > fused[1].score > 0

    empty = searcher.search_song_candidates("  ", fact_k=1, sparse_k=2)
    assert empty.dense_songs == empty.sparse_songs == ()
    assert searcher.search_fused_songs("  ") == ()


def test_song_candidate_boundary_rejects_duplicate_sparse_profiles(monkeypatch):
    from src.retrieval.context_qdrant_search import (
        ContextProfileHit,
        ContextQdrantSearch,
        ContextSearchHits,
    )

    profile = ContextProfileHit("101", "nws:101", 0.5, "가상 제목", ("가상 가수",))
    calls = []

    def fake_search(self, query, *, dense_k, sparse_k):
        calls.append((query, dense_k, sparse_k))
        return ContextSearchHits((), (profile, profile))

    monkeypatch.setattr(ContextQdrantSearch, "search", fake_search)
    searcher = object.__new__(ContextQdrantSearch)
    with pytest.raises(RuntimeError, match="more than one profile per song"):
        searcher.search_song_candidates("가상 작품 OST", fact_k=5, sparse_k=9)
    assert calls == [("가상 작품 OST", 5, 9)]

    with pytest.raises(ValueError, match="song_k must be positive"):
        searcher.search_song_candidates("가상 작품 OST", song_k=0)
    assert len(calls) == 1


def test_failed_repair_never_moves_the_active_alias_pair(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
    monkeypatch,
):
    import src.vector_db.context_qdrant as module

    sources = _sources(tmp_path)
    first = _load(tmp_path, sources, qdrant)
    aliases_before = _aliases(qdrant)
    original = module._upsert_points

    def fail_sparse(client, collection, iterator, **kwargs):
        if kwargs["label"] == "sparse":
            raise RuntimeError("injected sparse load failure")
        return original(client, collection, iterator, **kwargs)

    monkeypatch.setattr(module, "_upsert_points", fail_sparse)
    with pytest.raises(RuntimeError, match="injected sparse"):
        _load(tmp_path, sources, qdrant, force=True)

    assert _aliases(qdrant) == aliases_before
    assert aliases_before[first["dense_alias"]] == first["dense_collection"]
    assert aliases_before[first["sparse_alias"]] == first["sparse_collection"]


def test_new_full_generation_drops_stale_song_points_but_keeps_rollback(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
):
    first_sources = _sources(tmp_path / "v1", song_ids=("101", "202"))
    first = _load(tmp_path / "v1", first_sources, qdrant)
    second_sources = _sources(tmp_path / "v2", song_ids=("101",))
    second = _load(tmp_path / "v2", second_sources, qdrant)

    assert second["action"] == "build"
    assert second["context_build_id"] != first["context_build_id"]
    assert qdrant.client.count(second["dense_alias"], exact=True).count == 2
    assert qdrant.client.count(second["sparse_alias"], exact=True).count == 1
    assert qdrant.client.collection_exists(first["dense_collection"])
    assert qdrant.client.collection_exists(first["sparse_collection"])

    dense_rows, _ = qdrant.client.scroll(
        second["dense_alias"], limit=10, with_payload=True, with_vectors=False
    )
    sparse_rows, _ = qdrant.client.scroll(
        second["sparse_alias"], limit=10, with_payload=True, with_vectors=False
    )
    assert {row.payload["song_id"] for row in dense_rows} == {"101"}
    assert {row.payload["song_id"] for row in sparse_rows} == {"101"}


def test_final_gate_rejects_partial_catalogue_before_db_mutation(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
):
    sources = _sources(tmp_path, pending=3)

    with pytest.raises(ValueError, match="catalogue is incomplete"):
        _load(
            tmp_path,
            sources,
            qdrant,
            require_complete_scope=True,
        )

    assert qdrant.client.get_collections().collections == []
    assert qdrant.client.get_aliases().aliases == []


def test_source_manifest_hash_mismatch_is_rejected_before_db_mutation(
    tmp_path: Path,
    qdrant: QdrantVectorClient,
):
    sources = _sources(tmp_path)
    dense_manifest_path = sources[1] / "manifest.json"
    manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    manifest["source_manifest_sha256"] = "0" * 64
    digest_input = dict(manifest)
    digest_input.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = hashlib.sha256(json.dumps(
        digest_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    dense_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="another context manifest"):
        _load(tmp_path, sources, qdrant)

    assert qdrant.client.get_collections().collections == []
