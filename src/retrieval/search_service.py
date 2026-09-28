from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from pinecone import Pinecone

from src.retrieval import timing
from src.backend.schemas.search import MatchingTrack
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.models.text_koe5 import KoE5Embedder
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME

logger = logging.getLogger(__name__)


def hybrid_score(dense: List[float], sparse: Dict[str, Any], alpha: float):
    """
    Pinecone Hybrid Search를 위한 Dense, Sparse 벡터 가중치 조절
    """
    h_dense = [v * alpha for v in dense]
    h_sparse = {
        "indices": sparse.get("indices", []),
        "values": [v * (1.0 - alpha) for v in sparse.get("values", [])]
    }
    return h_dense, h_sparse


def _get(obj: Any, key: str, default=None):
    """Pinecone dict 응답과 객체 응답을 모두 지원한다."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


class SearchService:
    def __init__(self, pinecone_client: Any, text_embedder: KoE5Embedder, bm25_encoder: BM25SparseEncoder):
        self.pc = pinecone_client
        self.text_idx = self.pc.Index(TEXT_HYBRID_INDEX_NAME)
        self.text_embedder = text_embedder
        self.bm25_encoder = bm25_encoder

    @staticmethod
    def track_from_match(match: Any) -> MatchingTrack:
        """Pinecone match를 텍스트/이미지/오디오 경로 공용 MatchingTrack으로 변환한다."""
        metadata = _get(match, "metadata", {}) or {}
        match_id = str(_get(match, "id", ""))
        score = float(_get(match, "score", 0.0) or 0.0)

        return MatchingTrack(
            id=match_id,
            score=score,
            title=metadata.get("title", "Unknown"),
            artist=metadata.get("artist"),
            album=metadata.get("album"),
            release_date=metadata.get("release_date"),
            genre=metadata.get("genre"),
            type=metadata.get("type"),
            vocal_gender=metadata.get("vocal_gender"),
            artist_types=_parse_list(metadata.get("type")),
            search_style_summary=metadata.get("search_style_summary"),
            mood_tags=_parse_list(metadata.get("mood_tags")),
            time_weather_tags=_parse_list(metadata.get("time_weather_tags")),
            place_activity_tags=_parse_list(metadata.get("place_activity_tags")),
            emotion_tags=_parse_list(metadata.get("emotion_tags")),
            vibe_tags=_parse_list(metadata.get("vibe_tags")),
            relation_context_tags=_parse_list(metadata.get("relation_context_tags")),
            color_tags=_parse_list(metadata.get("color_tags")),
            sound_tags=_parse_list(metadata.get("sound_tags")),
            melon_playlist_tags=_parse_list(metadata.get("melon_playlist_tags")),
            visual_imagery=_parse_list(metadata.get("visual_imagery")),
            lyrics_highlight=metadata.get("lyrics_highlight"),
            lyrics_summary=metadata.get("lyrics_summary"),
            sentiment_summary=metadata.get("sentiment_summary"),
            fans_tags=_parse_list(metadata.get("fans_tags")),
            major_emotion=metadata.get("major_emotion"),
            fame=metadata.get("fame"),
            melon_url=metadata.get("melon_url"),
            youtube_url=metadata.get("youtube_url"),
            cover_url=metadata.get("cover_url"),
        )

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        sparse_query: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[MatchingTrack]:
        """
        Dense + Sparse Hybrid Search.

        - query: dense(KoE5) 의미 검색용 자연어 텍스트
        - sparse_query: sparse(BM25) 키워드 매칭용 텍스트 (없으면 query 재사용)
        - alpha: dense 비중 (1.0=dense 전용, 0.0=sparse 전용)
        """
        # 1. Dense Embedding — E5 계열은 query에 "query: " prefix 사용
        #
        # 임베딩과 DB 조회를 따로 잰다. 한 덩어리로 재면 "텍스트 경로가 느리다"까지만
        # 알 뿐, 모델을 바꿔야 하는지 인덱스를 손봐야 하는지 알 수 없다.
        with timing.step("text.embed_dense"):
            dense_vecs = self.text_embedder.embed_passages(
                [f"query: {query}"], add_e5_prefix=False, normalize=True
            )
            dense_vec = dense_vecs[0].tolist()

        # 2. Sparse Embedding — BM25 미학습 시 dense 전용으로 폴백
        sparse_vec = None
        if getattr(self.bm25_encoder, "_is_fitted", False):
            with timing.step("text.encode_sparse"):
                encoded = self.bm25_encoder.encode_queries(sparse_query or query)
                if isinstance(encoded, list):
                    encoded = encoded[0]
                if encoded.get("indices"):
                    sparse_vec = encoded

        # 3. Hybrid Weighting (Alpha = 0.5 기본)
        if sparse_vec is not None:
            h_dense, h_sparse = hybrid_score(dense_vec, sparse_vec, alpha)
        else:
            h_dense, h_sparse = dense_vec, None

        # 4. Pinecone Query
        query_params: Dict[str, Any] = {
            "vector": h_dense,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": NAMESPACE,
        }
        if metadata_filter:
            query_params["filter"] = metadata_filter

        if h_sparse is not None:
            query_params["sparse_vector"] = h_sparse

        try:
            with timing.step("text.query"):
                res = self.text_idx.query(**query_params)
        except Exception as e:
            if "does not support sparse values" in str(e):
                logger.warning("Pinecone 인덱스가 Sparse 벡터를 지원하지 않아 Dense 전용으로 자동 전환합니다.")
                query_params.pop("sparse_vector", None)
                query_params["vector"] = dense_vec
                res = self.text_idx.query(**query_params)
            else:
                raise

        # 5. Parse Results
        matches = _get(res, "matches", []) or []
        return [self.track_from_match(match) for match in matches]
