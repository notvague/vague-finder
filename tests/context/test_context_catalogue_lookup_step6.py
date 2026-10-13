"""A Context candidate must resolve to an existing Text index song."""

from qdrant_client import models

from src.retrieval.search_service import SearchService
from src.vector_db.qdrant_backend import (
    DENSE_VECTOR, QdrantVectorClient, point_id,
)
from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME


def test_context_lookup_uses_text_payload_and_skips_missing_or_mismatched_ids():
    client = QdrantVectorClient(path=":memory:")
    try:
        collection = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=2)
        client.client.upsert(collection, points=[
            models.PointStruct(
                id=point_id("123"),
                vector={DENSE_VECTOR: [1.0, 0.0]},
                payload={"song_id": "123", "title": "공식 곡명", "artist": "공식 가수"},
            ),
            models.PointStruct(
                id=point_id("124"),
                vector={DENSE_VECTOR: [0.0, 1.0]},
                payload={"song_id": "999", "title": "잘못 연결된 곡"},
            ),
            models.PointStruct(
                id=point_id("125"),
                vector={DENSE_VECTOR: [1.0, 0.0]},
                payload={"song_id": "125", "artist": "제목 없음"},
            ),
        ])
        search_service = SearchService.__new__(SearchService)
        search_service.text_idx = client.Index(TEXT_HYBRID_INDEX_NAME)
        tracks = search_service.fetch_tracks_by_ids(["123", "124", "125", "404"])
        assert list(tracks) == ["123"]
        assert tracks["123"].title == "공식 곡명"
        assert tracks["123"].artist == "공식 가수"
    finally:
        client.close()
