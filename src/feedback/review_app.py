"""운영자용 로컬 검토 화면. **공개 검색 서버(`src.backend.main`)와 다른 진입점이다.**

실행: `venv/bin/python -m src.feedback.review_app` → `http://127.0.0.1:8001/`

관리자 인증이 없다. 그래서 이 프로세스 안에서 접근을 좁힌다.

1. **loopback에만 붙는다.** `FEEDBACK_REVIEW_HOST`가 loopback 주소가 아니면 뜨지 않는다.
2. **Host 헤더를 검사한다.** 바깥 페이지가 자기 도메인을 127.0.0.1로 풀어 같은 출처인 척 붙는 것
   (DNS rebinding)을 막는다. 주소창에 적은 호스트와 포트가 이 서버의 것이 아니면 거절한다.
3. **쓰기 요청은 세 가지를 모두 본다.** 다른 사이트에서 온 요청이 아닐 것(`Sec-Fetch-Site`·`Origin`),
   이 화면이 받은 토큰을 실을 것(`X-Review-Token`), 본문이 JSON일 것. CORS 응답 헤더를 주지 않는
   것만으로는 부족하다 — 브라우저는 응답을 못 읽게 할 뿐 단순 요청은 그대로 보낸다.
4. 읽기 API도 다른 사이트에서 온 요청은 거절한다(원질의·사유가 실린다).

검토자 이름은 로컬 설정값(`FEEDBACK_REVIEWER`)이다. 확인된 신원이 아니며 화면에도 그렇게 적는다.
원격 운영자 접근은 인증을 갖춘 뒤의 별도 작업이다.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.backend.schemas.feedback import SearchId
from src.feedback.review import REVIEW_STATES
from src.feedback.review_service import MAX_PAGE_SIZE, MongoSongCatalog, ReviewService
from src.feedback.settings import MAX_REUSABLE_TEXT, MAX_REVIEW_NOTE, env_int
from src.feedback.store import FeedbackStore, ReviewError

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "review_static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
TOKEN_HEADER = "x-review-token"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# 주소창에 적을 수 있는 이 서버의 이름. 그 밖의 Host는 받지 않는다.
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    # 인라인 스크립트·스타일을 쓰지 않는다. 사용자 원문이 마크업으로 해석될 길을 한 번 더 막는다.
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


class ReviewBody(BaseModel):
    """검토 저장 요청. 길이·빈 문장 검증의 기준은 저장소(`apply_review`)이고 여기서는 형식만 본다."""

    model_config = ConfigDict(extra="forbid")

    state: str = Field(..., max_length=20)
    reusable_text: str = Field(default="", max_length=MAX_REUSABLE_TEXT * 4)
    note: str = Field(default="", max_length=MAX_REVIEW_NOTE * 4)
    expected_revision: int = Field(..., ge=1, le=21)
    expected_review_version: int = Field(..., ge=0, le=100_000)


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _error(status: int, code: str, message: str, current: Any = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if current is not None:
        body["current"] = current
    return JSONResponse(body, status_code=status, headers=SECURITY_HEADERS)


def create_app(
    service_factory: Optional[Callable[[], ReviewService]] = None,
    *,
    port: int = DEFAULT_PORT,
    csrf_token: Optional[str] = None,
    on_close: Optional[Callable[[], None]] = None,
) -> FastAPI:
    """검토 앱. `service_factory`를 주지 않으면 설정된 Mongo에 붙는다(첫 요청 때).

    토큰은 프로세스가 뜰 때 만들어 메모리에만 둔다. 다시 띄우면 열려 있던 화면은 새로 고쳐야 한다.
    """
    token = csrf_token or secrets.token_urlsafe(32)
    allowed_hosts = frozenset(f"{name}:{port}" for name in LOOPBACK_NAMES)
    holder: dict = {}
    holder_lock = threading.Lock()

    def service() -> ReviewService:
        # 화면은 첫 요청 몇 개를 동시에 보낸다. 서비스(=Mongo 클라이언트)는 한 번만 만든다.
        with holder_lock:
            if "service" not in holder:
                holder["service"] = (service_factory or _default_service_factory(holder))()
            return holder["service"]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        try:
            yield
        finally:
            for close in (holder.get("close"), on_close):
                if close is not None:
                    close()

    app = FastAPI(title="Vague-Finder 피드백 검토(로컬)", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Any]) -> Any:
        host = request.headers.get("host", "").lower()
        if host not in allowed_hosts:
            return _error(403, "forbidden_host", "이 주소로는 검토 화면에 접근할 수 없습니다.")
        is_api = request.url.path.startswith("/api/")
        site = request.headers.get("sec-fetch-site")
        origin = request.headers.get("origin")
        if is_api:
            # 브라우저가 붙이는 값이다. 없으면(구형 브라우저·명령줄) 아래 Origin·토큰 검사로 넘어간다.
            if site is not None and site not in {"same-origin", "none"}:
                return _error(403, "cross_site", "다른 사이트에서 온 요청은 받지 않습니다.")
            # 같은 출처여야 한다 — localhost로 연 화면이 127.0.0.1로 보내는 것도 다른 출처다.
            if origin is not None and origin.lower() != f"http://{host}":
                return _error(403, "cross_origin", "다른 출처에서 온 요청은 받지 않습니다.")
        if request.method in UNSAFE_METHODS:
            if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, "").encode(), token.encode()):
                return _error(403, "bad_token", "검토 화면을 새로 고친 뒤 다시 시도해 주세요.")
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                return _error(415, "json_only", "JSON 본문만 받습니다.")
        elif request.method not in {"GET", "HEAD"}:
            return _error(405, "method_not_allowed", "허용하지 않는 요청입니다.")
        try:
            response = await call_next(request)
        except Exception as exc:
            # DB 예외에는 URI·명령 문서가 들어갈 수 있다. 종류만 남기고 응답에는 사유를 싣지 않는다.
            logger.warning("review request failed: %s", type(exc).__name__)
            return _error(503, "storage_unavailable", "저장소에 연결하지 못했습니다. 잠시 뒤 다시 시도해 주세요.")
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        return response

    @app.exception_handler(ReviewError)
    async def review_error(_: Request, exc: ReviewError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.current)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid_request", "요청 형식이 올바르지 않습니다.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = "찾을 수 없습니다." if exc.status_code == 404 else "요청을 처리하지 못했습니다."
        return _error(exc.status_code, "http_error", message)

    @app.get("/api/session")
    def session() -> dict:
        return {**service().session(), "csrf_token": token}

    @app.get("/api/status")
    def status() -> dict:
        return service().status()

    @app.get("/api/feedback")
    def list_feedback(
        state: Annotated[str, Query(max_length=20)] = "pending",
        page: Annotated[int, Query(ge=1, le=10_000)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    ) -> dict:
        return service().list(state, page, page_size)

    @app.get("/api/feedback/{search_id}")
    def feedback_detail(search_id: SearchId) -> dict:
        return service().detail(search_id)

    @app.put("/api/feedback/{search_id}/review")
    def save_review(search_id: SearchId, body: ReviewBody) -> dict:
        if body.state not in REVIEW_STATES:
            raise ReviewError(422, "invalid_state", "검토 결과는 대기·승인·보류·제외 중 하나여야 합니다.")
        return service().review(
            search_id, state=body.state, reusable_text=body.reusable_text, note=body.note,
            expected_revision=body.expected_revision, expected_review_version=body.expected_review_version,
        )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    return app


def _default_service_factory(holder: dict) -> Callable[[], ReviewService]:
    def build() -> ReviewService:
        from pymongo import MongoClient

        from src.common.mongodb import DB_NAME, MONGO_URI

        # 검색 서버의 0.75초 예산과 달리 넉넉히 둔다 — 사람이 보는 화면이고 검색 응답을 붙잡지 않는다.
        client = MongoClient(MONGO_URI, connect=False, tz_aware=True, serverSelectionTimeoutMS=3000,
                             connectTimeoutMS=3000, timeoutMS=5000)
        holder["close"] = client.close
        database = client[DB_NAME]
        return ReviewService(
            FeedbackStore(database, client), MongoSongCatalog(database["songs"]),
            reviewer=os.getenv("FEEDBACK_REVIEWER", "").strip() or "local",
        )

    return build


def main() -> None:
    import uvicorn

    host = os.getenv("FEEDBACK_REVIEW_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = env_int("FEEDBACK_REVIEW_PORT", DEFAULT_PORT, low=1, high=65535)
    if not is_loopback_host(host):
        raise SystemExit(
            f"FEEDBACK_REVIEW_HOST={host!r}: 검토 화면은 로컬 전용입니다. 127.0.0.1(기본값)로만 띄웁니다."
        )
    print(f"피드백 검토 화면: http://{host}:{port}/  (로컬 전용 · 인증 없음)")
    uvicorn.run(create_app(port=port), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
