"""Gemini 클라이언트 — Vertex AI 전용(GCP_PROJECT_ID + ADC). 실제 호출은 하지 않는다.

`google.genai.Client`를 가짜로 바꿔 어떤 인자로 만들어지는지 보고, ADC·기동 검사는 private 구현
(`_load_default_credentials` · `_verify_vertex_setup`)을 직접 부른다 — 공개 이름은 conftest가 모든 테스트에서 바꿔 끼운다.
"""
from types import SimpleNamespace

import pytest

from src.common import gemini_client as gc
from src.common.gemini_client import GeminiConfigError
# 모듈 맨 위에서 import한다 — 아래 모듈은 import될 때 load_dotenv()를 부른다. 시험 함수 안에서 처음 import하면
# fixture가 env를 지운 뒤에 .env가 다시 읽혀 GCP_PROJECT_ID 등이 되살아나고, monkeypatch가 되돌리지 못해 뒤 시험까지 샌다.
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.query_analyzer import QueryAnalyzer
from src.backend import main as backend_main  # 같은 이유로 맨 위에서 — 서버 기동 시험용

# conftest의 autouse fixture가 바꿔 끼우기 **전**(모듈 import 시점)의 실제 구현.
_real_load_default_credentials = gc._load_default_credentials
_real_verify = gc._verify_vertex_setup

ENV_NAMES = (
    "GCP_PROJECT_ID", "GCP_LOCATION", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_RETRIEVAL_BACKEND",
    "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_MODEL_NAME", "GEMINI_CRAWL_MODEL_NAME",
)


class _FakeModels:
    def __init__(self, fail_with=None, total_tokens=1):
        self.fail_with, self.total_tokens, self.calls = fail_with, total_tokens, []

    def count_tokens(self, *, model, contents):
        self.calls.append((model, contents))
        if self.fail_with is not None:
            raise self.fail_with
        return SimpleNamespace(total_tokens=self.total_tokens)


class _FakeClient:
    def __init__(self, models=None, **kwargs):
        self.kwargs, self.models, self.closed = kwargs, models or _FakeModels(), False

    def close(self):
        self.closed = True


@pytest.fixture
def captured(monkeypatch):
    """`genai.Client` 생성 인자를 모은다. ADC는 conftest가 끼운 오프라인 자격 증명이다."""
    import google.genai as genai_mod

    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return _FakeClient(**kwargs)

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return calls


def test_vertex_client_uses_project_location_and_shared_credentials(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-옛-키")             # 남아 있어도 읽지 않는다
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")       # SDK 환경변수도 무시된다 — vertexai=True 명시
    gc.make_genai_client(http_options="opts")
    assert len(captured) == 1
    call = captured[0]
    assert call["vertexai"] is True and call["project"] == "proj-1" and call["location"] == "global"
    assert call["http_options"] == "opts" and "api_key" not in call
    assert call["credentials"] is gc.vertex_credentials(), "프로세스 공용 ADC 객체를 넘긴다"
    assert gc.gemini_backend() == "vertex" and gc.gemini_configured()
    # 프로젝트 ID는 runinfo(공개 레포)에 남으면 안 되는 식별자라 적지 않는다 — backend·location만
    assert gc.gemini_route_info() == {"backend": "vertex", "location": "global"}
    assert gc.gemini_route_info(gc.RETRIEVAL) == {"backend": "vertex", "location": "global", "setting": "vertex"}
    assert "proj-1" not in str(gc.gemini_route_info(gc.RETRIEVAL))


def test_location_comes_from_env(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    gc.make_genai_client()
    assert captured[0]["location"] == "us-central1"
    assert gc.gemini_route_info()["location"] == "us-central1"


def test_api_keys_never_select_a_provider(monkeypatch, captured):
    """옛 AI Studio 키가 있어도 Gemini 없음이다 — 키로 우회하는 경로가 없다."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-옛-키")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy-옛-키-2")
    assert not gc.gemini_configured() and gc.gemini_backend() == "none"
    assert gc.gemini_route_info(gc.RETRIEVAL) == {"backend": "none", "setting": "vertex"}
    with pytest.raises(GeminiConfigError, match="GCP_PROJECT_ID") as err:
        gc.make_genai_client()
    assert "AIzaSy" not in str(err.value)
    with pytest.raises(TypeError):  # 키를 받는 자리 자체가 없다
        gc.make_genai_client(api_key="직접")
    assert captured == []
    assert not QueryAnalyzer()._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled


@pytest.mark.parametrize("value", ["", "vertex", " VERTEX "])
def test_retrieval_setting_accepts_empty_or_vertex(monkeypatch, captured, value):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", value)
    assert gc.backend_setting(gc.RETRIEVAL) == "vertex"
    assert gc.gemini_backend(purpose=gc.RETRIEVAL) == "vertex"
    gc.make_genai_client(purpose=gc.RETRIEVAL)
    assert captured[0]["vertexai"] is True


@pytest.mark.parametrize("value", ["api_key", "ai_studio", "auto"])
def test_removed_ai_studio_settings_are_config_errors(monkeypatch, captured, value):
    """옛 값이 조용히 Vertex가 되면 안 된다 — .env를 고치라고 이름을 적어 알린다. 크롤링(용도 없음)은 이 설정을 읽지 않는다."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", value)
    with pytest.raises(GeminiConfigError, match=f"GEMINI_RETRIEVAL_BACKEND={value}.*AI Studio"):
        gc.gemini_backend(purpose=gc.RETRIEVAL)
    with pytest.raises(GeminiConfigError):
        gc.make_genai_client(purpose=gc.RETRIEVAL)
    assert captured == []
    assert gc.gemini_backend() == "vertex", "용도 없는 호출(크롤링)은 이 설정을 읽지 않는다"


def test_unknown_retrieval_setting_is_rejected_without_echoing_the_value(monkeypatch, captured):
    """API 키를 이 칸에 잘못 넣어도 에러 메시지(→ 기동·측정 로그)에 키가 남지 않는다."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "AIzaSy-시험용-가짜-키")
    with pytest.raises(GeminiConfigError) as err:
        gc.gemini_backend(purpose=gc.RETRIEVAL)
    assert "aizasy" not in str(err.value).lower() and "가짜" not in str(err.value)


def test_analyzer_and_reranker_follow_the_vertex_setting(monkeypatch, captured):
    """프로젝트가 있으면 켜지고 없으면 규칙 폴백·꺼짐이다. `configured=`는 환경과 무관하게 고정한다(테스트용)."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    assert QueryAnalyzer()._configured
    assert GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled
    assert not QueryAnalyzer(configured=False)._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), configured=False).enabled
    monkeypatch.delenv("GCP_PROJECT_ID")
    assert not QueryAnalyzer()._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled
    assert QueryAnalyzer(configured=True)._configured
    assert GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), configured=True).enabled
    assert GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), client=object()).enabled


def test_analyzer_and_reranker_build_clients_on_vertex(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-옛-키")
    QueryAnalyzer()._gemini  # noqa: B018 — 클라이언트 생성
    GeminiListwiseReranker(config=GeminiListwiseRerankerConfig())._gemini  # noqa: B018
    assert len(captured) == 2
    assert all(c["vertexai"] is True and c["project"] == "proj-1" and "api_key" not in c for c in captured)


def test_missing_adc_is_a_config_error_that_names_the_fix(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_adc(**kwargs):
        raise DefaultCredentialsError("no adc")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(GeminiConfigError, match="application-default login"):
        _real_load_default_credentials()


def test_credentials_are_loaded_once_per_process(monkeypatch):
    loads = []
    monkeypatch.setattr(gc, "_load_default_credentials", lambda: loads.append(1) or object())
    gc.reset_credentials_cache()
    first = gc.vertex_credentials()
    assert gc.vertex_credentials() is first and loads == [1]
    gc.reset_credentials_cache()
    assert gc.vertex_credentials() is not first and loads == [1, 1]


def test_probe_model_is_a_free_count_tokens_call_and_maps_rejections(monkeypatch, captured):
    import google.genai as genai_mod
    from google.genai import errors

    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    models = _FakeModels(total_tokens=3)
    clients = []

    def fake_client(**kwargs):
        client = _FakeClient(models=models, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    info = gc.probe_model("gemini-x")
    assert info == {"model": "gemini-x", "location": "global", "probe": "ok", "probe_tokens": 3}
    assert models.calls == [("gemini-x", "ping")] and clients[-1].closed
    assert getattr(clients[-1].kwargs["http_options"], "timeout", None) == 10_000

    models.fail_with = errors.ClientError(403, {"error": {"message": "denied", "status": "PERMISSION_DENIED"}})
    with pytest.raises(GeminiConfigError, match="403"):
        gc.probe_model("gemini-x")
    assert clients[-1].closed, "거절돼도 클라이언트는 닫는다"

    models.fail_with = ConnectionError("offline")  # 설정이 아니라 연결 문제 — 그대로 올린다
    with pytest.raises(ConnectionError):
        gc.probe_model("gemini-x")


def test_probe_model_treats_429_as_transient_and_lets_the_sdk_retry(monkeypatch, captured):
    """429(RESOURCE_EXHAUSTED)는 쿼터·용량 부족이라 설정 오류가 아니다 — 기동 검사에서 한 번 났다고 서버를 띄우지 않으면 안 된다(리뷰 P1)."""
    import google.genai as genai_mod
    from google.genai import errors

    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    models = _FakeModels(fail_with=errors.ClientError(429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}))
    clients = []

    def fake_client(**kwargs):
        client = _FakeClient(models=models, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    with pytest.raises(errors.ClientError) as err:
        gc.probe_model("gemini-x")
    assert err.value.code == 429 and not isinstance(err.value, GeminiConfigError)
    assert clients[-1].closed
    retry = clients[-1].kwargs["http_options"].retry_options
    assert retry.attempts == gc.PROBE_RETRY_ATTEMPTS >= 2, "429·5xx는 SDK가 제한 횟수만큼 다시 시도한다"
    assert retry.initial_delay == 1.0 and retry.max_delay == 4.0

    # 잘못된 인자·인증·권한·없는 모델은 그대로 설정 오류다
    for code in (400, 401, 403, 404):
        models.fail_with = errors.ClientError(code, {"error": {"message": "bad", "status": "x"}})
        with pytest.raises(GeminiConfigError, match=str(code)):
            gc.probe_model("gemini-x")


def _revoked(retryable: bool):
    from google.auth.exceptions import RefreshError

    return RefreshError("invalid_grant: Token has been expired or revoked.", {"error": "invalid_grant"}, retryable=retryable)


def test_probe_model_treats_a_revoked_adc_as_a_config_error_but_auth_outages_as_transient(monkeypatch, captured):
    """ADC 파일이 있어도 토큰 갱신이 invalid_grant로 거절되면 재로그인이 필요한 영구 오류다(리뷰 P2).

    HTTP 오류만 보면 이 예외는 '경고 후 계속 기동'으로 넘어가 정상 서버처럼 보이면서 모든 모델 호출이 실패한다.
    google-auth가 retryable로 표시한 인증 서버 일시 장애는 그대로 올린다.
    """
    import google.genai as genai_mod
    from google.auth.exceptions import RefreshError

    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    models = _FakeModels(fail_with=_revoked(retryable=False))
    monkeypatch.setattr(genai_mod, "Client", lambda **kwargs: _FakeClient(models=models, **kwargs))
    with pytest.raises(GeminiConfigError, match="application-default login") as err:
        gc.probe_model("gemini-x")
    assert "invalid_grant" in str(err.value) and isinstance(err.value.__cause__, RefreshError)

    models.fail_with = _revoked(retryable=True)
    with pytest.raises(RefreshError) as transient:
        gc.probe_model("gemini-x")
    assert not isinstance(transient.value, GeminiConfigError)


def test_server_refuses_to_start_on_a_revoked_adc_but_starts_on_an_auth_outage(monkeypatch):
    """실제 검사 경로(_verify_vertex_setup)로 — 철회된 ADC는 기동에서 멈추고, 인증 서버 일시 장애는 경고만 남기고 뜬다."""
    import google.genai as genai_mod
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)
    models = _FakeModels(fail_with=_revoked(retryable=False))
    monkeypatch.setattr(genai_mod, "Client", lambda **kwargs: _FakeClient(models=models, **kwargs))
    monkeypatch.setattr(gc, "verify_vertex_setup", _real_verify)
    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))

    with pytest.raises(GeminiConfigError, match="application-default login"):
        with TestClient(backend_main.app):
            pass
    assert opened == [], "설정 검사가 벡터 DB를 열기 전에 끝나야 한다"

    models.fail_with = _revoked(retryable=True)
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
    assert opened == [1]


def test_server_starts_when_the_startup_probe_hits_429(monkeypatch):
    """일시 장애는 경고만 남기고 뜬다 — 요청 경로에서 다시 시도한다."""
    from fastapi.testclient import TestClient
    from google.genai import errors

    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))

    def quota(purpose=None, **kwargs):
        raise errors.ClientError(429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(gc, "verify_vertex_setup", quota)
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
    assert opened == [1]


@pytest.mark.parametrize("value", ["claude_listwise", "gemini_listwse", "cross encoder"])
def test_server_refuses_to_start_on_removed_or_unknown_reranker_backend(monkeypatch, value):
    """빈 값은 기본(gemini_listwise)이고, 제거된 값·오타는 기동에서 멈춘다 — 조용히 Cross-Encoder로 내려가면 Vertex로 쟀다고 믿게 된다(리뷰 P2)."""
    from fastapi.testclient import TestClient

    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))
    monkeypatch.setenv("RERANKER_BACKEND", value)
    with pytest.raises(ValueError, match="RERANKER_BACKEND"):
        with TestClient(backend_main.app):
            pass
    assert opened == [], "설정 검사가 벡터 DB를 열기 전에 끝나야 한다"

    monkeypatch.setenv("RERANKER_BACKEND", "")
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200


def test_verify_vertex_setup_checks_settings_before_touching_the_network(monkeypatch, captured):
    with pytest.raises(GeminiConfigError, match="GCP_PROJECT_ID"):
        _real_verify(gc.RETRIEVAL)
    assert captured == []
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    with pytest.raises(GeminiConfigError, match="AI Studio"):
        _real_verify(gc.RETRIEVAL)
    assert captured == []
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND")
    monkeypatch.setenv("GEMINI_MODEL_NAME", "검색-모델")
    assert _real_verify(gc.RETRIEVAL, probe=False) == {
        "backend": "vertex", "location": "global", "setting": "vertex", "model": "검색-모델", "probe": "skipped",
    }
    assert captured == []
    info = _real_verify(gc.RETRIEVAL)
    assert info["probe"] == "ok" and info["model"] == "검색-모델" and captured[-1]["vertexai"] is True
    # 크롤링 용도(없음)는 크롤링 모델을 본다
    monkeypatch.setenv("GEMINI_CRAWL_MODEL_NAME", "크롤-모델")
    assert _real_verify()["model"] == "크롤-모델"


def test_crawl_and_retrieval_models_are_separate(monkeypatch):
    for name in ("GEMINI_CRAWL_MODEL_NAME", "GEMINI_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    assert gc.retrieval_model_name() == gc.DEFAULT_RETRIEVAL_MODEL == "gemini-3.1-flash-lite"
    assert gc.crawl_model_name() == gc.DEFAULT_CRAWL_MODEL == "gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL_NAME", " 검색-모델 ")
    monkeypatch.setenv("GEMINI_CRAWL_MODEL_NAME", " 다른-모델 ")
    assert gc.retrieval_model_name() == "검색-모델" and gc.crawl_model_name() == "다른-모델"
    assert QueryAnalyzer(configured=False)._model_name == "검색-모델"


def test_stale_api_key_is_reported_once_without_the_value(monkeypatch, caplog):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-옛-키")
    gc._warned.discard("stale_keys")
    with caplog.at_level("WARNING", logger="src.common.gemini_client"):
        gc.require_vertex()
        gc.require_vertex()
    hits = [r.getMessage() for r in caplog.records if "GEMINI_API_KEY" in r.getMessage()]
    assert len(hits) == 1 and "AIzaSy" not in hits[0]


def test_server_refuses_to_start_on_vertex_config_error(monkeypatch):
    """설정 오류는 기동에서 실패한다 — 정상으로 떠 있다가 첫 검색이 500으로 떨어지면 안 된다(PR #31 리뷰)."""
    from fastapi.testclient import TestClient

    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))

    def refuse(purpose=None, **kwargs):
        raise GeminiConfigError("GCP_PROJECT_ID가 없다")

    monkeypatch.setattr(gc, "verify_vertex_setup", refuse)
    with pytest.raises(GeminiConfigError, match="GCP_PROJECT_ID"):
        with TestClient(backend_main.app):
            pass
    assert opened == [], "설정 검사가 벡터 DB를 열기 전에 끝나야 한다"

    # 연결 문제는 경고만 — 요청 경로에서 다시 시도한다
    def offline(purpose=None, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(gc, "verify_vertex_setup", offline)
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
    assert opened == [1]


def test_server_closes_the_analyzer_gemini_client_on_shutdown(monkeypatch):
    from fastapi.testclient import TestClient
    from src.backend.api import dependencies as deps

    monkeypatch.setattr(backend_main, "get_vector_client", lambda: None)
    events = []

    class _Aio:
        async def aclose(self):
            events.append("aclose")

    class _Client:
        aio = _Aio()

        def close(self):
            events.append("close")

    deps.get_query_analyzer.cache_clear()
    analyzer = deps.get_query_analyzer()
    analyzer._client = _Client()
    try:
        with TestClient(backend_main.app):
            pass
        assert events == ["aclose", "close"] and analyzer._client is None
    finally:
        deps.get_query_analyzer.cache_clear()
