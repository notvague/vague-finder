from functools import wraps
import logging
import os
import threading
from pathlib import Path

from typing import Any, Callable, TypeVar

from src.common.gemini_client import RETRIEVAL, gemini_configured
from src.embedding.models.audio_clap import CLAPAudioEmbedder
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.models.text_koe5 import KoE5Embedder
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.lyrics_exact_search import LyricsExactSearchService
from src.retrieval.reranker import MusicReranker
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker
from src.retrieval.search_router import SearchRouter
from src.retrieval.search_service import SearchService
from src.vector_db.qdrant_backend import get_qdrant_client


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
    return (get_search_service, get_search_router)


def close_vector_client() -> None:
    """벡터 클라이언트를 닫고, 그것을 쥔 캐시까지 비운다.

    로컬 Qdrant는 저장 폴더를 잠근다. 프로세스가 끝나기 전에 닫지 않으면 다음 프로세스가
    그 폴더를 열지 못한다. 아직 만들어지지 않았으면 **새로 만들지 않는다.**

    순서가 중요하다. 스레드풀을 먼저 세운다 — 아직 도는 검색이 클라이언트를 쓰고 있을 수
    있다. 그다음 캐시를 비우고, 마지막에 클라이언트를 닫는다.
    """
    router = get_search_router.peek()  # type: ignore[attr-defined]
    if router is not None:
        shutdown = getattr(router, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[경고] 검색 라우터 스레드풀을 정리하는 중 오류: {exc}")

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


@singleton
def get_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


@singleton
def get_lyrics_exact_search_service() -> LyricsExactSearchService:
    # Mongo 연결과 full_lyrics snapshot 로딩은 첫 가사 표면 검색 시 지연 수행된다.
    return LyricsExactSearchService(
        fuzzy_threshold=float(os.getenv("LYRICS_FUZZY_THRESHOLD", "0.78")),
        cache_ttl_seconds=float(os.getenv("LYRICS_CACHE_TTL_SECONDS", "300")),
    )


DEFAULT_RERANKER_BACKEND = "gemini_listwise"


@singleton
def get_reranker():
    """환경변수로 Gemini listwise 또는 Cross-Encoder를 선택한다. **기본은 Gemini listwise.**

    2026-10-09 전환. 세 질의 세트(v06 dev 57 · v09 dev 59 · v09 봉인 test 38 = 154건)에서
    listwise 1패스·Google Search 끔이 CE 대비 엄격 Hit@10 97 → 114(0.630 → 0.740), Hit@1 53 → 78,
    Top-10 이탈 0건이었다(experiments/reranking/results_v32~v34). CE는 이 질의 유형에서 점수
    폭이 0.003 수준이라 후보를 가르지 못했다(results_v27 E4). 대가는 리랭킹 중앙값 1.8초 → 4.5~5.4초.

    CE로 돌리려면 RERANKER_BACKEND=cross_encoder. v22·v27~v31 기준선은 CE로 잰 것이라 재현에 필요하다.

    **Gemini 설정(GCP_PROJECT_ID 또는 GEMINI_API_KEY — `GEMINI_RETRIEVAL_BACKEND`로 고정했으면 그 경로의 설정)이 없으면 CE로 내려간다.** 설정 없는 listwise는
    입력 순서를 그대로 돌려주므로 리랭킹이 사실상 꺼진 채 조용히 뜨게 된다 — 그보다는 로컬 CE가 낫고,
    로그에 남긴다. 실행 중 Gemini 호출이 모두 실패하면 listwise가 검색 순서를 유지하고 상태를 failed로
    남긴다(`GeminiListwiseReranker.rerank_run`). 그때 CE로 바꿔 타지는 않는다 — 두 모델을 함께 올리지 않는다.
    """
    backend = os.getenv("RERANKER_BACKEND", DEFAULT_RERANKER_BACKEND).strip().lower()
    if backend in {"gemini_listwise", "gemini-listwise", "gemini"}:
        if gemini_configured(purpose=RETRIEVAL):
            return GeminiListwiseReranker()
        logger.warning(
            "[reranker] RERANKER_BACKEND=%s인데 Gemini 설정(GCP_PROJECT_ID·GEMINI_API_KEY, GEMINI_RETRIEVAL_BACKEND)이 없다 — Cross-Encoder로 내려간다",
            backend,
        )
    # 기존 모델은 첫 요청 때 지연 로딩한다.
    return MusicReranker()


@singleton
def get_search_router() -> SearchRouter:
    return SearchRouter(
        search_service=get_search_service(),
        image_embedder=get_image_embedder(),
        audio_embedder=get_audio_embedder(),
        vector_client=get_vector_client(),
        lyrics_search_service=get_lyrics_exact_search_service(),
        reranker=get_reranker(),
    )
