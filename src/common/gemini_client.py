"""Gemini 클라이언트 — **Vertex AI 전용** (2026-10-15).

운영용 생성형 호출(크롤링 정제·댓글 선별·Context 추출, 질의 분석·리랭킹)은 모두 `GCP_PROJECT_ID`의 Vertex AI
(`aiplatform.googleapis.com`, 위치 `GCP_LOCATION` 기본 global)를 ADC(Application Default Credentials)로 부른다.

- **AI Studio API 키 경로는 없다.** `GEMINI_API_KEY`·`GOOGLE_API_KEY`가 환경에 남아 있어도 읽지 않고(한 번 경고),
  호출부가 키를 넘겨 다른 경로로 나갈 방법도 없다. Vertex 호출이 실패해도 다른 공급자(AI Studio·Anthropic)로
  바꿔 타지 않는다 — 호출부의 기존 폴백(질의 분석은 규칙, 리랭커는 검색 순서 + `failed`)으로 내려가고 그렇게 기록된다.
- 설정이 없거나 잘못되면(프로젝트 없음 · ADC 없음 · 경로 설정 오타) **원인이 드러나는 오류**(`GeminiConfigError`)를 낸다.
  서버는 기동에서, 분석 캐시 생성·측정 도구는 시작에서 멈춘다 — 정상으로 떠 있다가 첫 검색이 500으로 떨어지거나
  176건이 전부 규칙 폴백으로 채워지지 않게(PR #31 리뷰).
- `GEMINI_RETRIEVAL_BACKEND`는 호환용이다. 비우거나 `vertex`만 받는다. 옛 `api_key`·`auto`는 설정 오류다 —
  AI Studio로 잰 기준선(v32~v47 · 봉인 test · 재질문 v09)과 같은 경로로 돌아갈 수 있다는 뜻이 되면 안 된다.
- 호출부는 클라이언트를 직접 만들지 않고 `make_genai_client`를 부른다. "Gemini를 쓸 수 있나"는 `gemini_configured`.
- 인증은 두 가지다. ① 개인 ADC(`gcloud auth application-default login`) ② **서비스 계정 키 파일**
  (`GOOGLE_APPLICATION_CREDENTIALS=secrets/vertex-sa.json`, 2026-10-15) — 팀원 PC·Docker처럼 gcloud 로그인과 IAM 초대 없이
  같은 프로젝트를 부를 때 쓴다. 상대 경로는 레포 루트 기준으로 푼다(venv의 cwd와 컨테이너 `/app`이 같은 파일을 가리키게).
  파일을 가리키는데 없으면 설정 오류다 — google-auth가 조용히 다른 ADC로 넘어가지 않게 한다.
- 자격 증명은 프로세스에서 한 번만 읽어(`vertex_credentials`) 모든 클라이언트가 공유한다. 리랭커는 호출마다 새
  클라이언트를 만들므로(이벤트 루프마다) 매번 ADC를 다시 읽고 토큰을 새로 받으면 호출마다 0.3초 + 왕복 하나가 붙는다.

측정 runinfo·피드백 기록에는 `gemini_route_info`가 backend(vertex)·location·setting을 적는다. 프로젝트 ID는 적지 않는다 —
공개 레포에 올라가는 기록에 남으면 안 되는 식별자다(10/9 v11 runinfo 9개에서 손으로 지웠다).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

VERTEX = "vertex"
DEFAULT_GCP_LOCATION = "global"

# 검색(질의 분석·리랭커) 모델. 바꾸면 검색 기준선과 비교할 수 없다.
DEFAULT_RETRIEVAL_MODEL = "gemini-3.1-flash-lite"
# 크롤링(정제·댓글 선별·Context 추출) 모델. 검색 모델과 따로 둔다: 검색 모델이 바뀌면 기준선과 비교할 수 없다.
# gemini-3.5-flash-lite(2026-07 출시): 10/9 Vertex에서 실제 크롤링 프롬프트(곡 3개)로 잰 중앙값이 정제 3.3초·댓글 선별 3.8초
# (3.1-flash-lite 12.9초·18.6초), thinking 0, 응답 형식 6/6. 3.8-flash가 더 새롭지만 기본 thinking(1.2k~2.9k 토큰)으로 14~19초였다.
# `*-latest` 별칭은 가리키는 모델이 조용히 바뀌므로 버전을 적는다.
DEFAULT_CRAWL_MODEL = "gemini-3.5-flash-lite"

# 용도 — 호출부가 넘기는 값. 없으면(None) 용도 설정을 보지 않는다(크롤링).
RETRIEVAL = "retrieval"
_PURPOSE_ENV = {RETRIEVAL: "GEMINI_RETRIEVAL_BACKEND"}
# 제거된 AI Studio 경로를 가리키던 옛 값들. 오류 메시지에 이름만 적는다.
_REMOVED_SETTINGS = ("api_key", "ai_studio", "auto")
_STALE_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# 서비스 계정 키 파일 경로. google-auth의 표준 변수라 이름을 바꾸지 않는다. 상대 경로는 레포 루트 기준.
CREDENTIALS_FILE_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_credentials_lock = threading.Lock()
_credentials: Any = None
_warned: set[str] = set()


class GeminiConfigError(ValueError):
    """설정·인증 오류. 다시 시도해도 낫지 않으므로 호출부는 재시도하지 않고 멈춘다(서버는 기동에서)."""


def crawl_model_name() -> str:
    """크롤링 Gemini 호출의 모델 ID — `GEMINI_CRAWL_MODEL_NAME`, 없으면 DEFAULT_CRAWL_MODEL."""
    return os.getenv("GEMINI_CRAWL_MODEL_NAME", "").strip() or DEFAULT_CRAWL_MODEL


def retrieval_model_name() -> str:
    """검색(질의 분석·리랭커) Gemini 호출의 모델 ID — `GEMINI_MODEL_NAME`, 없으면 DEFAULT_RETRIEVAL_MODEL."""
    return os.getenv("GEMINI_MODEL_NAME", "").strip() or DEFAULT_RETRIEVAL_MODEL


def gcp_project_id() -> str:
    return os.getenv("GCP_PROJECT_ID", "").strip()


def gcp_location() -> str:
    return os.getenv("GCP_LOCATION", "").strip() or DEFAULT_GCP_LOCATION


def backend_setting(purpose: Optional[str] = None) -> str:
    """용도별 경로 설정. 비우거나 `vertex`만 허용한다 — 다른 값은 GeminiConfigError.

    제거된 옛 값(`api_key`·`ai_studio`·`auto`)은 이름을 적어 되돌려 준다. 그 밖의 값은 메시지에 넣지 않는다 —
    이 칸에 API 키를 잘못 넣으면 기동·측정 로그에 키가 그대로 남는다.
    """
    if purpose is None:
        return VERTEX
    env = _PURPOSE_ENV[purpose]
    raw = os.getenv(env, "").strip().lower()
    if raw in ("", VERTEX):
        return VERTEX
    if raw in _REMOVED_SETTINGS:
        raise GeminiConfigError(
            f"{env}={raw}는 더 이상 쓰지 않는다 — AI Studio 경로는 제거됐다(2026-10-15). "
            f"이 줄을 지우거나 {env}=vertex로 둔다"
        )
    raise GeminiConfigError(f"{env} 값을 알 수 없다 — 비우거나 vertex여야 한다")


def gemini_backend(purpose: Optional[str] = None) -> str:
    """실제로 쓸 경로 — `vertex` 또는 `none`(프로젝트 없음). 설정 오타는 GeminiConfigError."""
    backend_setting(purpose)
    return VERTEX if gcp_project_id() else "none"


def gemini_configured(purpose: Optional[str] = None) -> bool:
    """이 용도로 Vertex를 부를 설정(프로젝트)이 있으면 True. ADC는 보지 않는다 — 그것은 `verify_vertex_setup`."""
    return gemini_backend(purpose) != "none"


def warn_stale_keys() -> None:
    """옛 AI Studio 키가 환경에 남아 있으면 한 번 알린다. 읽지는 않는다."""
    stale = [name for name in _STALE_KEY_ENVS if os.getenv(name, "").strip()]
    if stale and "stale_keys" not in _warned:
        _warned.add("stale_keys")
        logger.warning(
            "[gemini_client] %s가 환경에 있지만 쓰지 않는다 — Gemini는 Vertex AI(GCP_PROJECT_ID + ADC)만 부른다. .env에서 지운다",
            "·".join(stale),
        )


def require_vertex(purpose: Optional[str] = None) -> dict:
    """설정만 본다(네트워크 없음) — 경로 설정 · 프로젝트. 없으면 GeminiConfigError. 반환은 `gemini_route_info`."""
    backend_setting(purpose)
    if not gcp_project_id():
        raise GeminiConfigError(
            "GCP_PROJECT_ID가 없다 — Gemini는 Vertex AI 전용이라 크레딧이 연결된 GCP 프로젝트가 필요하다(.env.example 3). "
            "AI Studio 키(GEMINI_API_KEY)로는 돌지 않는다"
        )
    warn_stale_keys()
    return gemini_route_info(purpose)


def credentials_file_path() -> Optional[Path]:
    """`GOOGLE_APPLICATION_CREDENTIALS`가 가리키는 키 파일의 절대 경로. 비어 있으면 None(개인 ADC를 쓴다).

    상대 경로는 cwd가 아니라 **레포 루트** 기준이다 — venv(`uvicorn`을 레포 루트에서)와 Docker(`/app` 바인드 마운트)가
    `.env`의 같은 한 줄(`secrets/vertex-sa.json`)로 같은 파일을 가리킨다. 측정 CLI를 다른 폴더에서 띄워도 어긋나지 않는다.
    """
    raw = os.getenv(CREDENTIALS_FILE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _load_default_credentials() -> Any:
    """자격 증명을 읽는다(파일 읽기뿐, 토큰 요청은 첫 호출 때). 없으면 어떻게 만드는지까지 적은 GeminiConfigError.

    키 파일 변수가 있으면 그 파일이어야 한다 — 없으면 멈춘다. google-auth는 변수가 가리키는 파일이 없을 때도
    DefaultCredentialsError를 내지만, 메시지가 ADC 안내로 뭉개져 팀원이 엉뚱한 로그인을 하게 된다.
    """
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    key_file = credentials_file_path()
    if key_file is not None:
        if not key_file.is_file():
            raise GeminiConfigError(
                f"{CREDENTIALS_FILE_ENV}가 가리키는 서비스 계정 키 파일이 없다: {key_file} — "
                "팀 드라이브의 vertex-sa.json을 레포의 secrets/ 폴더에 두거나 .env의 경로를 고친다"
            )
        # google-auth는 이 변수를 cwd 기준으로 읽는다 — 레포 루트 기준으로 푼 절대 경로를 넘긴다.
        os.environ[CREDENTIALS_FILE_ENV] = str(key_file)
    try:
        credentials, _adc_project = google.auth.default(scopes=_SCOPES)
    except DefaultCredentialsError as exc:
        raise GeminiConfigError(
            "Vertex AI 인증이 없다 — 둘 중 하나를 한다. ① 팀 드라이브의 서비스 계정 키를 secrets/vertex-sa.json에 두고 "
            f".env에 {CREDENTIALS_FILE_ENV}=secrets/vertex-sa.json을 적는다(팀원·Docker 권장). ② 크레딧이 있는 Google 계정으로 "
            "`gcloud auth application-default login`을 한 번 실행한다(브라우저 로그인만으로는 바뀌지 않는다). "
            f"원인: {exc}"
        ) from exc
    return credentials


def vertex_credentials() -> Any:
    """프로세스에서 한 번만 읽는 ADC. 모든 Vertex 클라이언트가 같은 객체를 공유한다(토큰 갱신도 한 곳에서)."""
    global _credentials
    if _credentials is None:
        with _credentials_lock:
            if _credentials is None:
                _credentials = _load_default_credentials()
    return _credentials


def reset_credentials_cache() -> None:
    """테스트·계정 교체용. 다음 `vertex_credentials()`가 ADC를 다시 읽는다."""
    global _credentials
    with _credentials_lock:
        _credentials = None


def make_genai_client(*, http_options: Any = None, purpose: Optional[str] = None):
    """`google.genai.Client`를 Vertex AI로 만든다. 설정이 없으면 GeminiConfigError — 호출부는 그 전에 `gemini_configured`로 거른다.

    프로젝트·위치·자격 증명을 모두 명시한다. SDK는 명시된 프로젝트/자격 증명이 있으면 환경의 `GOOGLE_API_KEY`·
    `GEMINI_API_KEY`·`GOOGLE_CLOUD_PROJECT`·`GOOGLE_GENAI_USE_VERTEXAI`를 쓰지 않는다 — 옛 키가 남아 있어도
    AI Studio나 Vertex express(API 키) 모드로 새지 않는다. http_options(타임아웃·재시도)는 그대로 넘긴다.
    """
    from google import genai

    require_vertex(purpose)
    return genai.Client(
        vertexai=True,
        project=gcp_project_id(),
        location=gcp_location(),
        credentials=vertex_credentials(),
        http_options=http_options,
    )


# 확인 호출에서 **설정 오류**로 보는 HTTP 코드 — 잘못된 인자·인증·권한·없는 모델. 다시 시도해도 낫지 않는다.
# 429(RESOURCE_EXHAUSTED)는 쿼터·용량 부족으로 지나가는 오류라 넣지 않는다 — 기동 검사에서 한 번 났다고 서버를 띄우지 않으면
# 일시 장애가 영구 장애가 된다(리뷰 P1). 408·5xx도 같은 이유로 그대로 올린다.
_CONFIG_HTTP_CODES = frozenset({400, 401, 403, 404})
# 확인 호출의 SDK 재시도 횟수(원 요청 포함). 429·5xx 같은 재시도 대상 코드만 1~4초 지수 대기로 다시 부른다.
PROBE_RETRY_ATTEMPTS = 3


def probe_model(model: str, *, purpose: Optional[str] = None) -> dict:
    """모델 하나를 **무료 호출**(countTokens)로 건드려 본다 — 토큰 과금 없음.

    프로젝트 권한 · Vertex API 활성화 · 이 위치에 그 모델이 있는가 · ADC 토큰 갱신이 한 번에 확인된다.
    설정·권한 쪽 실패(`_CONFIG_HTTP_CODES`: 400·401·403·404)는 GeminiConfigError로 바꾼다. 429·5xx·연결 불가는 SDK가
    제한 횟수(`PROBE_RETRY_ATTEMPTS`)만큼 다시 시도한 뒤에도 남으면 **그대로 올린다** — 호출부가 "설정 오류(멈춤)"와
    "지금 못 닿음(경고만 남기고 진행)"을 가른다.
    """
    from google.auth.exceptions import RefreshError
    from google.genai import errors, types

    client = make_genai_client(
        purpose=purpose,
        http_options=types.HttpOptions(
            timeout=10_000,
            retry_options=types.HttpRetryOptions(attempts=PROBE_RETRY_ATTEMPTS, initial_delay=1.0, max_delay=4.0),
        ),
    )
    try:
        counted = client.models.count_tokens(model=model, contents="ping")
    except RefreshError as exc:
        # ADC 파일은 있는데 토큰 갱신이 거절된 경우. invalid_grant(철회·만료·계정 변경)는 다시 시도해도 낫지 않는 인증 오류라
        # 멈춘다 — 경고만 남기고 뜨면 정상 서버처럼 보이면서 모든 모델 호출이 실패한다(리뷰 P2). google-auth가 retryable로
        # 표시한 인증 서버 일시 장애(internal_failure·server_error·temporarily_unavailable)는 그대로 올린다.
        if getattr(exc, "retryable", False):
            raise
        raise GeminiConfigError(
            "Vertex AI 인증(ADC) 토큰을 갱신하지 못했다 — 자격 증명이 철회·만료됐거나 다른 계정의 것이다. "
            "크레딧이 있는 Google 계정으로 `gcloud auth application-default login`을 다시 실행한다. "
            f"원인: {exc}"
        ) from exc
    except errors.ClientError as exc:
        if exc.code not in _CONFIG_HTTP_CODES:
            raise  # 429 등 — 설정이 아니라 지금 못 닿는 것. 호출부의 일시 장애 처리로 넘긴다
        raise GeminiConfigError(
            f"Vertex AI가 모델 {model}(location={gcp_location()}) 호출을 거절했다 — HTTP {exc.code}. "
            "프로젝트 권한(roles/aiplatform.user 이상), Vertex AI API 활성화, 모델 이름·위치, ADC 계정을 확인한다. "
            f"원인: {exc.message}"
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return {"model": model, "location": gcp_location(), "probe": "ok",
            "probe_tokens": getattr(counted, "total_tokens", None)}


def _verify_vertex_setup(
    purpose: Optional[str] = None,
    *,
    model: Optional[str] = None,
    extra_models: Sequence[str] = (),
    probe: bool = True,
) -> dict:
    """기동·측정 시작 검사: 설정(경로·프로젝트) → ADC → (probe) 모델마다 무료 호출 한 번.

    `model`은 이 용도의 기본 모델(검색이면 질의 분석 모델). `extra_models`는 같은 실행이 부를 다른 모델 — 리랭커가
    `GEMINI_RERANK_MODEL_NAME`으로 분석 모델과 다른 모델을 쓰면 그것도 확인한다(리뷰: 리랭커 모델 오타면 서버는 정상으로
    뜨고 모든 리랭킹이 404 → failed로 조용히 내려갔다). 같은 이름은 한 번만 부른다.

    GeminiConfigError는 다시 시도해도 낫지 않는 오류라 호출부가 멈춘다. 그 밖의 예외(연결 불가 등)는 호출부가 경고만
    남기고 진행한다 — 요청 경로의 폴백·실패 기록이 그대로 남는다.
    """
    info = require_vertex(purpose)
    vertex_credentials()
    model = model or (retrieval_model_name() if purpose == RETRIEVAL else crawl_model_name())
    models = list(dict.fromkeys([model, *[m for m in extra_models if m]]))
    info["model"] = model
    info["models"] = models
    if probe:
        for name in models:
            probed = probe_model(name, purpose=purpose)
            if name == model:
                info.update(probed)
        info["probe"] = "ok"
    else:
        info["probe"] = "skipped"
    return info


def verify_vertex_setup(
    purpose: Optional[str] = None,
    *,
    model: Optional[str] = None,
    extra_models: Sequence[str] = (),
    probe: bool = True,
) -> dict:
    """공개 진입점. 테스트 conftest가 통째로 바꿔 끼운다 — 테스트는 ADC도 Vertex도 건드리지 않는다."""
    return _verify_vertex_setup(purpose, model=model, extra_models=extra_models, probe=probe)


def gemini_route_info(purpose: Optional[str] = None) -> dict:
    """측정 runinfo·피드백 기록용 — 어느 경로·위치로 불렀는지. 키 값도, GCP 프로젝트 ID도 적지 않는다.

    비교 조건으로 필요한 것은 backend(vertex)와 location이다. 용도를 주면 그 용도의 설정값(setting)도 적는다 —
    .env를 바꾼 뒤의 실행을 가려낼 수 있어야 한다. 같은 모델 이름이라도 AI Studio(옛 기준선 `api_key`)와
    Vertex는 출력이 달랐다(results_clarify_v11_prebonus) — backend가 다른 실행끼리는 비교하지 않는다.
    """
    backend = gemini_backend(purpose)
    info: dict = {"backend": backend}
    if backend == VERTEX:
        info["location"] = gcp_location()
    if purpose is not None:
        info["setting"] = backend_setting(purpose)
    return info
