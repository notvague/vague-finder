from functools import wraps
import logging
import os
import threading
from pathlib import Path

from typing import Any, Callable, Optional, TypeVar

from fastapi import HTTPException
from src.backend.schemas.search import SearchModelSelection
from src.common.search_models import ModelProvider, default_provider, provider_configured
from src.common.claude_client import ClaudeGenaiClient
from src.retrieval.claude_listwise_reranker import ClaudeListwiseReranker

from src.common.gemini_client import RETRIEVAL, gemini_configured
from src.embedding.models.audio_clap import CLAPAudioEmbedder
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.models.text_koe5 import KoE5Embedder
from src.feedback.case_index import FeedbackCaseIndex
from src.feedback.retrieval import FeedbackRetriever
from src.feedback.store import get_feedback_store
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.lyrics_exact_search import LyricsExactSearchService
from src.retrieval.reranker import MusicReranker
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker
from src.retrieval.search_router import SearchRouter
from src.retrieval.context_qdrant_search import ContextQdrantSearch
from src.retrieval.search_service import SearchService
from src.vector_db.qdrant_backend import get_qdrant_client
from src.retrieval.context_search_settings import load_context_settings


T = TypeVar("T")


def singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """호출이 겹쳐도 **한 번만** 만든다. lru_cache 자리에 그대로 쓴다.

    @lru_cache()는 첫 호출의 중복 실행을 막지 못한다. 서버가 뜬 직후 요청 두 개가 동시에
    들어오면 두 스레드가 모두 캐시 미스를 보고 각각 factory를 실행한다. 결과가 하나로
    수렴하더라도 그 과정에서 객체가 두 개 만들어진다.

    로컬 Qdrant에서는 그것이 곧 실패다 — 저장 폴더는 한 프로세스에서 하나만 열 수 있어서
    두 번째 클라이언트가 "Storage folder ... is already accessed"로 죽는다. 임베딩 모델도
    같은 이유로 곤란하다(KoE5·SigLIP2·CLAP를 두 벌 올리면 메모리가 두 배로 든다).

    lru_cache와 호환되도록 cache_clear()를 남겨 둔다(테스트가 쓴다).
    """
    lock = threading.Lock()
    holder: dict = {}

    @wraps(factory)
    def wrapper() -> T:
        if "value" in holder:
            return holder["value"]
        with lock:
            if "value" not in holder:
                holder["value"] = factory()
        return holder["value"]

    wrapper.cache_clear = holder.clear  # type: ignore[attr-defined]
    # 아직 안 만들어졌으면 만들지 않고 None을 준다. 종료 처리가 쓴다.
    wrapper.peek = lambda: holder.get("value")  # type: ignore[attr-defined]
    # model_singleton과 같은 면 — 종료 처리가 두 종류의 팩토리를 같은 식으로 훑는다.
    wrapper.values = lambda: tuple(holder.values())  # type: ignore[attr-defined]
    return wrapper


def model_singleton(factory: Callable[..., T]) -> Callable[..., T]:
    """모델(provider)별로 한 객체만 만든다. FastAPI 의존성과 CLI 진입점을 겸한다.

    - FastAPI: 검색 본문의 `model_provider`만 읽는다(`SearchModelSelection` — 본문 전체를 다시 검증하지 않는다).
      매개변수 이름은 라우트의 `request: SearchRequest`와 **같아야 한다.** FastAPI는 본문 필드를 이름으로 묶어서,
      이름이 다르면 본문이 `{"request": …}`로 감싸인 모양으로 바뀐다(test_search_models가 평평한 본문으로 지킨다).
      선택한 모델에 키가 없으면 503 — 다른 모델로 바꿔 타지 않는다.
    - CLI(인자 없음): 서버 기본 모델. `RERANKER_BACKEND` 같은 CLI 전용 덮어쓰기는 이 경로에만 적용된다.
    - `for_provider(provider)`: CLI에서 모델을 지정한다 — 분석 캐시를 **그 캐시의 모델**로 이어 채울 때.
    """
    lock = threading.Lock()
    holder: dict[tuple[str, bool], T] = {}

    def acquire(provider: Optional[ModelProvider], *, for_request: bool) -> T:
        provider = provider or default_provider()
        if for_request and not provider_configured(provider):
            raise HTTPException(status_code=503, detail=f"{provider} API 키가 설정되지 않았습니다.")
        key = (provider, for_request)
        with lock:
            if key not in holder:
                holder[key] = factory(provider, for_request=for_request)
            return holder[key]

    def wrapper(request: Optional[SearchModelSelection] = None) -> T:
        if request is None:
            return acquire(None, for_request=False)
        return acquire(request.model_provider, for_request=True)

    wrapper.__name__ = factory.__name__
    wrapper.__doc__ = factory.__doc__
    wrapper.for_provider = lambda provider: acquire(provider, for_request=False)  # type: ignore[attr-defined]
    wrapper.cache_clear = holder.clear  # type: ignore[attr-defined]
    # CLI 기본 모델의 객체. 아직 없으면 만들지 않고 None — evaluate_feedback_reuse와 테스트가 본다.
    wrapper.peek = lambda: holder.get((default_provider(), False))  # type: ignore[attr-defined]
    wrapper.values = lambda: tuple(holder.values())  # type: ignore[attr-defined]
    return wrapper


logger = logging.getLogger(__name__)

@singleton
def get_vector_client() -> Any:
    """검색에 쓸 벡터 DB(Qdrant) 클라이언트.

    검색 코드는 `Index(name).query(...)`만 부른다(qdrant_backend 참고).

    로컬 Qdrant는 저장 폴더를 하나만 열 수 있다. 앱 시작 시(lifespan) 한 번 열어 두고
    요청들이 공유하며, 종료 시 close_vector_client()로 닫는다.
    """
    return get_qdrant_client()


def _vector_client_dependents() -> tuple:
    """get_vector_client()가 만든 객체를 붙들고 있는 팩토리들.

    클라이언트를 닫을 때 이들의 캐시도 비워야 한다. 비우지 않으면 같은 프로세스에서 앱을
    다시 시작할 때(반복 lifespan, TestClient 여러 번) 새 클라이언트를 만들어 놓고도 검색은
    닫힌 클라이언트를 쥔 예전 객체를 재사용해 "QdrantLocal instance is closed"로 죽는다.

    이 목록에서 빠지면 테스트가 잡는다
    (test_every_factory_holding_the_vector_client_is_cleared_on_close).
    정의 순서 때문에 함수 안에서 이름을 찾는다.
    """
    return (get_search_service, get_search_router, get_feedback_case_index, get_feedback_retriever)


def close_vector_client() -> None:
    """벡터 클라이언트를 닫고, 그것을 쥔 캐시까지 비운다.

    로컬 Qdrant는 저장 폴더를 잠근다. 프로세스가 끝나기 전에 닫지 않으면 다음 프로세스가
    그 폴더를 열지 못한다. 아직 만들어지지 않았으면 **새로 만들지 않는다.**

    순서가 중요하다. 스레드풀을 먼저 세운다 — 아직 도는 검색이 클라이언트를 쓰고 있을 수
    있다. 그다음 캐시를 비우고, 마지막에 클라이언트를 닫는다.
    """
    # 모델별 라우터가 여럿일 수 있다(요청용 claude·gemini, CLI용). 전부 세운다.
    routers = get_search_router.values()  # type: ignore[attr-defined]
    for router in routers:
        shutdown = getattr(router, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[경고] 검색 라우터 스레드풀을 정리하는 중 오류: {exc}")

    # Claude 어댑터는 요청 사이 HTTP 풀을 재사용하므로 서버가 내려갈 때 닫는다 — 분석기, 라우터의 리랭커, CLI 리랭커가 쥔다.
    owners = [*get_query_analyzer.values(), *(getattr(r, "_reranker", None) for r in routers),  # type: ignore[attr-defined]
              get_reranker.peek()]  # type: ignore[attr-defined]
    clients = {id(c): c for owner in owners
               if isinstance((c := getattr(owner, "_client", None)), ClaudeGenaiClient)}
    for client in clients.values():
        try:
            client.close()
        except Exception:
            logger.warning("Claude 연결 풀 종료 실패", exc_info=True)
    get_query_analyzer.cache_clear()  # type: ignore[attr-defined]
    get_reranker.cache_clear()  # type: ignore[attr-defined]

    for factory in _vector_client_dependents():
        factory.cache_clear()

    client = get_vector_client.peek()  # type: ignore[attr-defined]
    get_vector_client.cache_clear()  # type: ignore[attr-defined]
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # 종료 경로에서 예외를 올리면 애플리케이션 종료가 막힌다
        print(f"[경고] 벡터 클라이언트를 닫는 중 오류: {exc}")


@singleton
def get_text_embedder() -> KoE5Embedder:
    return KoE5Embedder()


@singleton
def get_image_embedder() -> SigLIP2Embedder:
    return SigLIP2Embedder()


@singleton
def get_audio_embedder() -> CLAPAudioEmbedder:
    return CLAPAudioEmbedder()


@singleton
def get_bm25_encoder() -> BM25SparseEncoder:
    params_path = Path(os.getenv("BM25_PARAMS_PATH", "artifacts/bm25_params.json"))
    encoder = BM25SparseEncoder(params_path=params_path)
    try:
        encoder.load()
    except Exception as exc:
        print(
            f"[경고] {params_path} 경로에서 BM25 학습 파라미터를 "
            f"불러오지 못했습니다. 오류 원인: {exc}"
        )
    return encoder


@singleton
def get_search_service() -> SearchService:
    return SearchService(
        vector_client=get_vector_client(),
        text_embedder=get_text_embedder(),
        bm25_encoder=get_bm25_encoder(),
    )


@model_singleton
def get_query_analyzer(provider: ModelProvider, *, for_request: bool = False) -> QueryAnalyzer:
    return QueryAnalyzer(provider=provider)


@singleton
def get_lyrics_exact_search_service() -> LyricsExactSearchService:
    # Mongo 연결과 full_lyrics snapshot 로딩은 첫 가사 표면 검색 시 지연 수행된다.
    return LyricsExactSearchService(
        fuzzy_threshold=float(os.getenv("LYRICS_FUZZY_THRESHOLD", "0.78")),
        cache_ttl_seconds=float(os.getenv("LYRICS_CACHE_TTL_SECONDS", "300")),
    )


@singleton
def get_feedback_case_index() -> FeedbackCaseIndex:
    """승인 사례 컬렉션. 색인 작업(쓰기)과 검색(읽기)이 **같은 객체**를 써야 한다 — 잠금이 여기에 있다.

    만들기만 해서는 Qdrant에 아무것도 하지 않는다. 컬렉션은 색인 작업이 처음 돌 때 만든다.
    """
    return FeedbackCaseIndex(get_vector_client())


@singleton
def get_feedback_retriever() -> FeedbackRetriever:
    """승인 사례로 후보를 보충하는 조회 서비스. `FEEDBACK_RETRIEVAL_ENABLED`가 꺼져 있으면 라우터가 부르지 않는다.

    Mongo 저장소와 KoE5 모델은 **함수로 넘긴다** — 꺼져 있는 동안에는 연결도 모델 로딩도 일어나지 않는다.
    """
    return FeedbackRetriever(
        case_index=get_feedback_case_index(),
        store=get_feedback_store,
        embedder=get_text_embedder,
        vector_client=get_vector_client(),
    )


def _make_reranker(backend: str) -> Any:
    if backend in {"claude_listwise", "claude-listwise", "claude"}:
        return ClaudeListwiseReranker()
    if backend in {"gemini_listwise", "gemini-listwise", "gemini"}:
        if gemini_configured(purpose=RETRIEVAL):
            return GeminiListwiseReranker()
        logger.warning("[reranker] GEMINI_API_KEY 없음 — 로컬 Cross-Encoder로 내려간다")
        return MusicReranker()
    if backend == "cross_encoder":
        return MusicReranker()
    raise ValueError("RERANKER_BACKEND는 claude_listwise / gemini_listwise / cross_encoder 중 하나여야 합니다.")


@singleton
def get_reranker() -> Any:
    """CLI 평가 기본값. RERANKER_BACKEND는 과거 기준선 재현용 명시적 덮어쓰기다."""
    backend = os.getenv("RERANKER_BACKEND", "").strip().lower() or f"{default_provider()}_listwise"
    return _make_reranker(backend)


@model_singleton
def get_search_router(provider: ModelProvider, *, for_request: bool = False) -> SearchRouter:
    context_settings = load_context_settings()
    return SearchRouter(
        search_service=get_search_service(),
        image_embedder=get_image_embedder(),
        audio_embedder=get_audio_embedder(),
        vector_client=get_vector_client(),
        lyrics_search_service=get_lyrics_exact_search_service(),
        reranker=(get_reranker() if not for_request
                  else _make_reranker(f"{provider}_listwise")),
        context_search=ContextQdrantSearch(
            get_vector_client(), text_embedder=get_text_embedder()
        ),
        context_weight=context_settings.weight,
        context_fact_k=context_settings.fact_k,
        context_sparse_k=context_settings.sparse_k,
        context_named_media_multiplier=context_settings.named_media_multiplier,
        default_candidate_k=context_settings.candidate_k,
        feedback_retriever=get_feedback_retriever(),
    )
