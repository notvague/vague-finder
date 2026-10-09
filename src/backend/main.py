import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.backend.api.dependencies import close_vector_client, get_vector_client
from src.common.gemini_client import RETRIEVAL, backend_setting
from src.backend.api.routes import search
from src.backend import warmup
from src.retrieval import timing

STATIC_DIR = Path(__file__).resolve().parents[1] / "frontend" / "static"


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
    """
    # 검색 Gemini 경로 설정(GEMINI_RETRIEVAL_BACKEND)의 오타는 **기동에서** 실패시킨다. 분석기는 첫 요청에서 만들어지고
    # 예열은 리랭커 예외를 로그로만 남겨, 그대로 두면 서버가 정상으로 떠 있다가 첫 검색이 500으로 떨어진다(PR #31 리뷰).
    # 벡터 DB 실패와 달리 다시 시도해도 낫지 않는 설정 오류라 서버를 띄우지 않는다.
    backend_setting(RETRIEVAL)

    try:
        get_vector_client()
    except Exception as exc:
        print(f"[경고] 벡터 DB 초기화 실패(첫 검색 요청에서 다시 시도): {exc}")

    # 모델 예열은 기동을 막지 않는다. 첫 검색 하나가 11초를 혼자 내지 않게 하는
    # 것이 목적이고, 그 사이에 들어온 요청은 예열 없던 때와 같은 값을 낼 뿐이다.
    # 진행 상태는 /health에 있다. SEARCH_WARMUP=0이면 건너뛴다.
    warmup.start_background()

    try:
        yield
    finally:
        # 순서가 중요하다 — 예열이 남은 단계로 넘어가지 못하게 먼저 세운다.
        # **제한을 두지 않고 기다린다.** 돌고 있는 단계가 벡터 클라이언트를 만지는
        # 중일 수 있어서, 여기서 시간을 끊으면 닫은 뒤에 그 단계가 저장소를 다시 연다.
        await asyncio.to_thread(warmup.stop_and_wait)
        await asyncio.to_thread(close_vector_client)


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

TIMED_PATHS = ("/api/v1/search",)


@app.middleware("http")
async def record_request_timing(request: Request, call_next):
    """검색 요청의 **전체** 시간을 잰다 — 라우트 바깥까지.

    라우트 안에서만 재면 의존성 해결(첫 요청의 모델 로딩이 여기서 일어난다),
    요청 본문 검증, 응답 직렬화가 빠진다. 15.7초의 정체를 찾는 중이므로 빠지는
    구간이 있으면 안 된다. 미들웨어 시간에서 라우트 시간을 빼면 그 바깥이 남는다.

    검색 경로만 잰다. 정적 파일까지 한 줄씩 남기면 정작 볼 줄이 묻힌다.

    `X-Request-Id`를 돌려주는 이유: 측정 스크립트가 클라이언트에서 잰 벽시계
    시간과 서버가 남긴 구간 기록을 같은 요청끼리 맞춰야 한다.
    """
    if not request.url.path.startswith(TIMED_PATHS):
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
    }

app.include_router(search.router, prefix="/api/v1")

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
