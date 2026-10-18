import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.backend.api.dependencies import (
    close_gemini_clients,
    close_lyrics_exact_search_service,
    close_vector_client,
    get_vector_client,
    load_work_name_runtime,
    lyrics_cache_status,
    reranker_backend,
)
from src.common import gemini_client as gc
from src.common.gemini_client import RETRIEVAL, GeminiConfigError
from src.retrieval.gemini_listwise_reranker import GeminiListwiseRerankerConfig
from src.backend.api.routes import search
from src.backend.api.routes import feedback
from src.feedback import worker as feedback_worker
from src.feedback.health import feedback_health
from src.feedback.store import close_feedback_store, prepare_indexes
from src.backend import context_startup, warmup
from src.retrieval import timing

STATIC_DIR = Path(__file__).resolve().parents[1] / "frontend" / "static"


def _gemini_reranker_models() -> list[str]:
    """기동 검사에서 질의 분석 모델과 함께 확인할 리랭커 모델. 리랭커 백엔드 값 검증도 겸한다(오타·제거된 값은 여기서 ValueError).

    리랭커는 `GEMINI_RERANK_MODEL_NAME`을 먼저 읽어 분석 모델(`GEMINI_MODEL_NAME`)과 다를 수 있다. 분석 모델만 확인하면
    리랭커 모델 오타에도 서버가 정상으로 뜨고 모든 리랭킹이 404 → failed로 조용히 내려간다(리뷰). 리랭커가 Gemini listwise가
    아니거나 꺼져 있으면 빈 목록이다. 조용히 CE로 내려가는 경우(프로젝트 없음)는 기동 검사가 먼저 프로젝트 없음으로 멈춘다.
    """
    if reranker_backend() != "gemini_listwise":
        return []
    config = GeminiListwiseRerankerConfig.from_env()
    return [config.model_name] if config.enabled and config.model_name else []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """벡터 DB 클라이언트를 시작할 때 한 번 열고, 종료할 때 닫는다.

    로컬 Qdrant는 저장 폴더를 한 프로세스에서 하나만 열 수 있다. 첫 요청에서 만들면
    동시에 들어온 요청 두 개가 각각 열려다 한쪽이
    "Storage folder ... is already accessed"로 실패한다. 요청이 오기 전에 열어 둔다.

    실패해도 서버는 뜬다. /health와 정적 페이지는 벡터 DB 없이도 응답해야 하고, 검색은
    첫 요청에서 다시 시도한다.

    종료는 **기다린다.** close_vector_client()가 검색 스레드풀이 비기를 기다린 뒤 클라이언트를
    닫기 때문에 블로킹이다. 이벤트 루프를 막지 않도록 별도 스레드에서 돌리고 완료를 기다린다.

    종료 순서: **예열을 먼저 세우고 그다음에 클라이언트를 닫는다.** 예열 도중에 서버를
    내리면 뒤쪽 vector_db 단계가 종료 처리 뒤에 실행되어 방금 닫은 저장소를 다시 연다.
    로컬 Qdrant는 폴더를 한 번에 하나만 열 수 있으므로 그 핸들이 다음 기동을 막는다.

    피드백 색인 작업(`FEEDBACK_INDEX_WORKER_ENABLED`)도 같은 이유로 **벡터 클라이언트보다 먼저**
    세운다. 이 프로세스의 Qdrant 클라이언트로 승인 사례를 쓰는 스레드라, 돌고 있는 묶음이 끝난 뒤에
    클라이언트를 닫아야 한다. Mongo는 그 작업과 색인 준비가 모두 끝난 뒤 마지막에 닫는다.
    """
    # Gemini(Vertex AI) 설정 오류는 **기동에서** 실패시킨다 — 프로젝트 없음 · ADC 없음 · 경로 설정(GEMINI_RETRIEVAL_BACKEND)
    # 오타 · 프로젝트 권한/모델 거절. 분석기는 첫 요청에서 만들어지고 예열은 리랭커 예외를 로그로만 남겨, 그대로 두면
    # 서버가 정상으로 떠 있다가 첫 검색이 500으로 떨어지거나 분석·리랭킹이 조용히 폴백으로만 돈다(PR #31 리뷰).
    # 벡터 DB 실패와 달리 다시 시도해도 낫지 않는 설정 오류라 서버를 띄우지 않는다. 연결 불가·429 같은 일시 장애는 경고만 —
    # 요청 경로에서 다시 시도한다(실패하면 규칙 폴백·검색 순서 + failed로 기록된다).
    # 리랭커 백엔드 값의 오타·제거된 값(claude_listwise)도 기동에서 멈춘다 — 조용히 Cross-Encoder로 내려가면 Vertex로 쟀다고 믿게 된다.
    # try 바깥에서 — 리랭커 백엔드 값 오류(ValueError)는 연결 문제가 아니라 설정 오류라 경고로 삼키면 안 된다.
    reranker_models = _gemini_reranker_models()
    try:
        route = gc.verify_vertex_setup(RETRIEVAL, extra_models=reranker_models)
    except GeminiConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - 설정이 아니라 연결 문제
        print(f"[경고] Vertex AI 확인 실패(첫 검색에서 다시 시도): {type(exc).__name__}: {exc}")
    else:
        print(f"[Gemini] Vertex AI 확인: location={route.get('location')} models={route.get('models') or [route.get('model')]} probe={route.get('probe')}")

    vector_client = None
    try:
        vector_client = get_vector_client()
    except Exception as exc:
        print(f"[경고] 벡터 DB 초기화 실패(첫 검색 요청에서 다시 시도): {exc}")

    # Context(배경 지식) 색인·별칭 기동 검사 — Context를 쓰도록 설정됐을 때만(`CONTEXT_WEIGHT>0`).
    # v56에서 서버 Qdrant로 옮기며 별칭을 빠뜨렸는데 서버가 정상으로 뜨고 검색마다 경고만 남긴 채 Context 없이
    # 돌았다. 별칭 누락·잘못된 대상·manifest 불일치는 다시 시도해도 낫지 않는 설정 오류라 서버를 띄우지 않는다.
    # 연결 불가·시간 초과는 위 벡터 DB 실패와 같은 일시 장애로 보고 경고만 남긴다(확인 불가로 기록, 확인 완료 아님).
    # 예열·색인 작업·피드백 스레드보다 **먼저** 돈다 — 실패하면 아직 띄운 것이 벡터 클라이언트뿐이라 그것만 닫으면
    # 로컬 Qdrant 저장 폴더 잠금이 풀리고 같은 프로세스에서 다시 띄울 수 있다. 모델을 올리지 않는다.
    # 설정 오류(ContextStartupError)뿐 아니라 검사 코드의 **예상 밖 예외**(리뷰: 자료형 오류로 TypeError가 난 사례)에도
    # 닫는다 — 어떤 이유로든 기동이 여기서 끊기면 클라이언트는 남지 않아야 한다. 원래 예외를 그대로 다시 올린다.
    try:
        context_startup.run(vector_client)
        # 간접 지칭 작품명 해석(실험, 기본 꺼짐) — 켰으면 저장소·해석기를 여기서 불러온다. 저장소 없음·sha 불일치·
        # 설정값 오류는 다시 시도해도 낫지 않으므로 기동 실패다(꺼져 있으면 아무것도 읽지 않는다). 기동마다 환경을 다시 읽는다.
        load_work_name_runtime.cache_clear()  # type: ignore[attr-defined]
        load_work_name_runtime()
    except Exception:
        await asyncio.to_thread(close_vector_client)
        raise

    # 모델 예열은 기동을 막지 않는다. 첫 검색 하나가 11초를 혼자 내지 않게 하는
    # 것이 목적이고, 그 사이에 들어온 요청은 예열 없던 때와 같은 값을 낼 뿐이다.
    # 진행 상태는 /health에 있다. SEARCH_WARMUP=0이면 건너뛴다.
    warmup.start_background()

    # 피드백 보관 기간(TTL) 색인도 요청이 오기 전에 만든다. 요청 경로에서는 0.75초 예산 안에서만
    # 시도하기 때문에 거기에만 맡기면 큰 컬렉션의 색인이 끝내 안 만들어질 수 있다. 수집이 꺼져 있으면
    # 연결하지 않고, 실패해도 기동을 막지 않는다. 기다리지 않는다 — 종료할 때만 정리한다.
    index_task = asyncio.create_task(asyncio.to_thread(prepare_indexes))

    # 승인 사례 색인과 만료·무효 사례 정리. 꺼져 있으면 스레드도 연결도 만들지 않는다. 기동을 막지 않고,
    # 뜨자마자 한 주기를 돌려 꺼져 있던 동안 밀린 색인·정리를 처리한다.
    feedback_worker.start_background()

    try:
        yield
    finally:
        # 순서가 중요하다 — 예열이 남은 단계로 넘어가지 못하게 먼저 세운다.
        # **제한을 두지 않고 기다린다.** 돌고 있는 단계가 벡터 클라이언트를 만지는
        # 중일 수 있어서, 여기서 시간을 끊으면 닫은 뒤에 그 단계가 저장소를 다시 연다.
        await asyncio.to_thread(warmup.stop_and_wait)
        # 가사 판 백그라운드 갱신(TTL 만료 뒤 Mongo 전체 읽기)을 멈춘다. 예열이 끝난 뒤여야
        # 한다 — 예열 lyrics 단계가 최초 적재 중일 수 있다. 닫힌 뒤에는 새 갱신을 시작하지
        # 않고 돌던 갱신은 결과를 게시하지 않는다. 대기는 유한하다 — 가사 읽기는 전용 Mongo
        # 클라이언트의 socket 제한(한 배치) 안에 반환되거나 실패하므로 "제한 + 여유"만 기다리고,
        # 그 안에 안 끝나면 경고(False)로 남긴다(join만 끊고 읽기를 남기는 것이 아니라 읽기 자체가 끝난다).
        await asyncio.to_thread(close_lyrics_exact_search_service)
        # 색인 작업은 asyncio 작업이 아니라 스레드다. 멈춤 신호를 주고 **끝날 때까지 기다린 뒤**
        # 벡터 클라이언트를 닫는다 — 돌던 묶음이 닫힌 저장소에 쓰지 않게.
        await asyncio.to_thread(feedback_worker.stop_and_wait)
        # 분석기가 쥔 Gemini 연결 풀(동기·비동기)은 이 루프에서 닫는다 — 비동기 풀은 이 루프에 묶여 있다.
        await close_gemini_clients()
        await asyncio.to_thread(close_vector_client)
        # 색인 작업이 아직 돌고 있으면 끝난 뒤에 닫는다(최대 INDEX_BUDGET_SECONDS). 먼저 닫으면
        # 그 작업이 닫힌 클라이언트를 다시 열어 둔 채로 남는다.
        await asyncio.gather(index_task, return_exceptions=True)
        await asyncio.to_thread(close_feedback_store)


app = FastAPI(
    title="Vague-Finder API",
    description="Multi-modal RAG Search Engine API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware Setup
origins = [
    "http://localhost:3000",  # Next.js defaults
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 검색 한 건의 시간만 잰다. **정확히 이 경로만** — 접두사로 고르면 질문 조회(/api/v1/search/clarify)가
# 질의도 구간도 없는 행으로 섞여 검색 시간 집계를 흐린다.
TIMED_PATHS = frozenset({"/api/v1/search"})


@app.middleware("http")
async def record_request_timing(request: Request, call_next):
    """검색 요청의 **전체** 시간을 잰다 — 라우트 바깥까지.

    라우트 안에서만 재면 의존성 해결(첫 요청의 모델 로딩이 여기서 일어난다),
    요청 본문 검증, 응답 직렬화가 빠진다. 15.7초의 정체를 찾는 중이므로 빠지는
    구간이 있으면 안 된다. 미들웨어 시간에서 라우트 시간을 빼면 그 바깥이 남는다.

    검색 경로만 잰다. 정적 파일까지 한 줄씩 남기면 정작 볼 줄이 묻힌다. 검색 아래의 질문 조회
    (`/search/clarify`)도 재지 않는다 — 검색이 아니라서 한 줄로 남기면 검색 분포에 0초대 행이 낀다.

    `X-Request-Id`를 돌려주는 이유: 측정 스크립트가 클라이언트에서 잰 벽시계
    시간과 서버가 남긴 구간 기록을 같은 요청끼리 맞춰야 한다.
    """
    if request.url.path not in TIMED_PATHS:
        return await call_next(request)

    timer = timing.TimingRecorder()
    request.state.timing = timer
    try:
        response = await call_next(request)
    except BaseException:
        timer.note(failed=True)
        timing.emit(timer)
        raise
    timer.note(status=response.status_code)
    response.headers["X-Request-Id"] = timer.request_id
    response.headers["Cache-Control"] = "no-store"  # 실행별 피드백 제출 권한을 캐시에 남기지 않는다
    # 본문을 내보내기 직전까지가 서버가 책임지는 시간이다. 네트워크 왕복과
    # 화면 렌더링은 여기에 안 잡히므로 클라이언트에서 따로 재야 한다.
    timing.emit(timer)
    return response


@app.get("/")
async def root():
    return {"message": "Welcome to Vague-Finder API v0.1.0"}

@app.get("/health")
async def health_check():
    """떠 있는가, 그리고 **빠를 준비가 됐는가**.

    status는 그대로 ok다 — 예열 중에도 검색은 된다(느릴 뿐이다). 발표 전에
    warmup.state가 ready인지 보고 첫 검색을 시작하면 11초짜리 첫 요청을 피한다.
    """
    return {
        "status": "ok",
        "service": "vague-finder-backend",
        "warmup": warmup.status(),
        # 가사 판(Mongo full_lyrics 캐시)의 상태 — fresh/stale, 곡 수, 마지막 성공 시각,
        # 백그라운드 갱신 진행 여부, 실패·재시도 대기. 아직 안 만들어졌으면 만들지 않는다.
        "lyrics_cache": lyrics_cache_status(),
        # 피드백 수집·색인 작업·검색 반영의 준비 상태. 건수·시각·상태 코드뿐이다 —
        # 원문·내부 오류 본문·연결 정보는 싣지 않는다.
        "feedback": feedback_health(),
        # Context(배경 지식) 색인·별칭의 **기동 검사** 결과 — disabled / verified / unavailable
        # (config_error면 서버가 뜨지 않으므로 여기 보이지 않는다). 저장된 결과를 돌려줄 뿐, 조회마다
        # Qdrant를 다시 묻지 않는다. `checked_at`·`note`가 기동 시점 결과임을 말한다.
        "context": context_startup.status(),
    }

app.include_router(search.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/map")
async def map_page() -> FileResponse:
    """everynoise 스타일 노래 맵 페이지.

    no-cache는 "저장하지 마라"가 아니라 "쓰기 전에 물어봐라"다. ETag가 있으므로
    바뀐 게 없으면 304만 오간다(본문 없음).

    이게 없으면 브라우저가 Last-Modified만 보고 어림짐작으로 캐시를 쓴다. 그러면
    index.html이 옛 판으로 남고, 그 안의 ?v= 번호도 옛것이라 map.js·style.css까지
    통째로 옛 판이 뜬다. 정적 파일의 캐시 무효화를 ?v= 에 맡기고 있으니
    index.html만은 늘 새로 확인해야 한다.
    """
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
