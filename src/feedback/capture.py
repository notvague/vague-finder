"""검색 응답 직전의 표시 결과를 저장한다. 저장소 장애가 검색으로 전파되지 않는다."""
import asyncio
import hashlib
import logging
import os
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

from src.feedback.store import collection_enabled, get_feedback_store

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def source_version():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


async def capture_search(request, analysis, results, candidate_ids, turn, search_router):
    if not collection_enabled() or not results:
        return None
    try:
        # 출생 연도 슬롯 원값 대신 실제 검색에서 사용하는 연도 창을 기록한다.
        from src.retrieval.clarify import apply_birth_year_answers
        from src.retrieval.analysis_cache import analyzer_fingerprint

        effective = apply_birth_year_answers(analysis, request.answers)
        reranker = getattr(search_router, "_reranker", None)
        config = getattr(reranker, "config", None)
        snapshot = {
            "query": request.query, "analysis": effective.model_dump(mode="json"),
            "answers": [a.model_dump(mode="json") for a in request.answers if a.slot != "birth_year"],
            "asked_slots": request.asked_slots, "rejected_ids": request.rejected_ids,
            "turn": turn, "shown_ids": [str(t.id) for t in results],
            "shown": [{"id": str(t.id), "rank": rank, "title": t.title, "artist": t.artist}
                      for rank, t in enumerate(results, 1)],
            "candidate_ids": list(candidate_ids),
            "versions": {
                "source_sha": source_version(), "analyzer": analyzer_fingerprint(),
                "corpus": os.getenv("FEEDBACK_CORPUS_VERSION", "unknown"),
                "reranker": type(reranker).__name__ if reranker else None,
                "reranker_model": getattr(config, "model_name", None),
                "reranker_config": asdict(config) if is_dataclass(config) else None,
                "retrieval_backend": os.getenv("GEMINI_RETRIEVAL_BACKEND", "auto"),
                "feedback_index": None,
            },
            "options": {"top_k": request.top_k, "candidate_k": request.candidate_k,
                        "use_rerank": request.use_rerank},
        }
        return await asyncio.to_thread(lambda: get_feedback_store().record_search(snapshot, request.feedback_parent))
    except Exception as exc:
        # DB 예외에는 URI·명령 문서가 포함될 수 있어 예외 종류만 남긴다.
        logger.warning("feedback capture unavailable: %s", type(exc).__name__)
        return None
