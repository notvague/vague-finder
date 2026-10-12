"""검색 응답 직전의 표시 결과를 저장한다. 저장소 장애가 검색으로 전파되지 않는다."""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

from src.feedback.store import collection_enabled, get_feedback_store
from src.retrieval import timing

logger = logging.getLogger(__name__)

# 기록에 실패한 뒤 이 시간 동안은 저장소를 다시 두드리지 않는다. 저장소가 내려가 있으면 결과가 있는
# 검색마다 연결 제한(0.5~0.75초)을 기다리게 된다 — 피드백을 남기지 않는 사용자도 그 값을 낸다.
RETRY_AFTER_SECONDS = 30.0
_unavailable_until = 0.0


@lru_cache(maxsize=1)
def source_version():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def _analyzer_fingerprint():
    """서버가 떠 있는 동안 분석기 소스와 설정은 바뀌지 않는다. 검색마다 소스 두 개를 다시 읽지 않는다.

    `analyzer_fingerprint()` 자체는 캐시하지 않는다 — 측정 도구는 환경 변수가 바뀐 것을 봐야 한다.
    """
    from src.retrieval.analysis_cache import analyzer_fingerprint

    return analyzer_fingerprint()


def _plain(value):
    """Mongo가 받는 값만 남긴다. 설정에 Path·Enum·set 같은 필드가 생겨도 기록이 통째로 실패하지 않게."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _capture_search(request, analysis, results, candidate_ids, turn, search_router):
    # 파일 읽기·해시·직렬화부터 DB 기록까지 같은 작업 스레드에서 처리한다.
    # 출생 연도 슬롯 원값 대신 실제 검색에서 사용하는 연도 창을 기록한다.
    from src.retrieval.clarify import apply_birth_year_answers

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
            "source_sha": source_version(), "analyzer": dict(_analyzer_fingerprint()),
            "corpus": os.getenv("FEEDBACK_CORPUS_VERSION", "unknown"),
            "reranker": type(reranker).__name__ if reranker else None,
            "reranker_model": getattr(config, "model_name", None),
            "reranker_config": _plain(asdict(config)) if is_dataclass(config) else None,
            "retrieval_backend": os.getenv("GEMINI_RETRIEVAL_BACKEND", "auto"),
            "feedback_index": None,
        },
        "options": {"top_k": request.top_k, "candidate_k": request.candidate_k,
                    "use_rerank": request.use_rerank},
    }
    return get_feedback_store().record_search(snapshot, request.feedback_parent)


async def capture_search(request, analysis, results, candidate_ids, turn, search_router):
    global _unavailable_until
    if not collection_enabled() or not results:
        return None
    if time.monotonic() < _unavailable_until:
        return None  # 방금 실패했다. 이 검색은 제출 권한 없이 내보내고 저장소를 기다리지 않는다.
    try:
        # 기록이 저장돼야 제출 권한을 발급할 수 있으므로 완료를 기다린다. 기다린 시간은 요청 시간
        # 기록에 따로 남긴다 — 검색 단계에 묻히면 "검색이 느리다"와 구분되지 않는다.
        with timing.step("feedback.capture"):
            return await asyncio.to_thread(
                _capture_search, request, analysis, results, candidate_ids, turn, search_router,
            )
    except Exception as exc:
        _unavailable_until = time.monotonic() + RETRY_AFTER_SECONDS
        # DB 예외에는 URI·명령 문서가 포함될 수 있어 예외 종류만 남긴다.
        logger.warning(
            "feedback capture unavailable: %s (다음 시도까지 %.0f초)", type(exc).__name__, RETRY_AFTER_SECONDS,
        )
        return None
