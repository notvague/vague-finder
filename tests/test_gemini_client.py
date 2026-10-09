"""Gemini 클라이언트 경로 선택 — Vertex AI(GCP_PROJECT_ID) 또는 AI Studio API 키, 검색 용도 고정(GEMINI_RETRIEVAL_BACKEND).

실제 호출은 하지 않는다. `google.genai.Client`를 가짜로 바꿔 어떤 인자로 만들어지는지만 본다.
"""
import pytest

from src.common import gemini_client as gc
# 모듈 맨 위에서 import한다 — 두 모듈은 import될 때 load_dotenv()를 부른다. 시험 함수 안에서 처음 import하면
# fixture가 env를 지운 뒤에 .env가 다시 읽혀 GEMINI_API_KEY 등이 되살아나고, monkeypatch가 되돌리지 못해 뒤 시험까지 샌다.
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.query_analyzer import QueryAnalyzer
from src.backend import main as backend_main  # 같은 이유로 맨 위에서 — 서버 기동 시험용


@pytest.fixture
def captured(monkeypatch):
    import google.genai as genai_mod

    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    for name in ("GCP_PROJECT_ID", "GCP_LOCATION", "GEMINI_API_KEY", "GEMINI_RETRIEVAL_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_vertex_when_project_is_set(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "키도-있음")
    gc.make_genai_client(http_options="opts")
    assert captured == [{"vertexai": True, "project": "proj-1", "location": "global", "http_options": "opts"}]
    assert gc.gemini_backend() == "vertex"
    # 프로젝트 ID는 runinfo(공개 레포)에 남으면 안 되는 식별자라 적지 않는다 — backend·location만
    assert gc.gemini_route_info() == {"backend": "vertex", "location": "global"}


def test_location_comes_from_env(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    gc.make_genai_client()
    assert captured[0]["location"] == "us-central1"


def test_api_key_when_no_project(monkeypatch, captured):
    """크레딧이 끝나면 GCP_PROJECT_ID를 비우고 키를 되살리는 롤백 경로."""
    monkeypatch.setenv("GEMINI_API_KEY", "키")
    gc.make_genai_client()
    assert captured == [{"api_key": "키", "http_options": None}]
    assert gc.gemini_backend() == "api_key"
    assert gc.gemini_route_info() == {"backend": "api_key"}, "키 값은 적지 않는다"


def test_explicit_key_wins_over_vertex(monkeypatch, captured):
    """호출부가 키를 직접 넘기면(테스트·예비 키) 환경과 상관없이 그 키를 쓴다."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    gc.make_genai_client(api_key="직접")
    assert captured[0] == {"api_key": "직접", "http_options": None}


def test_nothing_configured(captured):
    assert not gc.gemini_configured()
    assert gc.gemini_backend() == "none"
    with pytest.raises(ValueError):
        gc.make_genai_client()
    assert captured == []


def test_vertex_only_counts_as_configured_for_analyzer_and_reranker(monkeypatch, captured):
    """키를 주석 처리하고 Vertex만 남겨도 분석기·리랭커가 규칙 폴백·꺼짐으로 내려가지 않는다."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    assert QueryAnalyzer()._configured
    assert GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled
    monkeypatch.delenv("GCP_PROJECT_ID")
    assert not QueryAnalyzer()._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled


def _both_configured(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "키")


def test_retrieval_can_stay_on_api_key_while_crawling_uses_vertex(monkeypatch, captured):
    """검색만 AI Studio로 고정 — 기준선(AI Studio)과 같은 경로에서 재고, 크롤링(용도 없음)은 Vertex 그대로."""
    _both_configured(monkeypatch)
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    assert gc.gemini_backend() == "vertex"
    assert gc.gemini_backend(purpose=gc.RETRIEVAL) == "api_key"
    gc.make_genai_client(purpose=gc.RETRIEVAL)
    gc.make_genai_client()
    assert captured == [{"api_key": "키", "http_options": None},
                        {"vertexai": True, "project": "proj-1", "location": "global", "http_options": None}]
    assert gc.gemini_route_info(gc.RETRIEVAL) == {"backend": "api_key", "setting": "api_key"}


def test_retrieval_setting_auto_or_empty_follows_default_rule(monkeypatch, captured):
    _both_configured(monkeypatch)
    for value in ("", "auto", " AUTO "):
        monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", value)
        assert gc.gemini_backend(purpose=gc.RETRIEVAL) == "vertex"
    assert gc.gemini_route_info(gc.RETRIEVAL) == {"backend": "vertex", "location": "global", "setting": "auto"}


def test_retrieval_setting_accepts_ai_studio_alias_and_vertex(monkeypatch, captured):
    _both_configured(monkeypatch)
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "ai_studio")
    assert gc.gemini_backend(purpose=gc.RETRIEVAL) == "api_key"
    assert gc.gemini_route_info(gc.RETRIEVAL)["setting"] == "api_key", "기록은 api_key로 통일"
    monkeypatch.delenv("GCP_PROJECT_ID")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "vertex")
    assert gc.gemini_backend() == "api_key"
    assert gc.gemini_backend(purpose=gc.RETRIEVAL) == "none"


def test_pinned_retrieval_backend_without_its_settings_does_not_switch_paths(monkeypatch, captured):
    """api_key로 고정했는데 키가 없으면 Vertex로 바꿔 타지 않는다 — 다른 경로의 숫자가 조용히 섞이면 안 된다."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    assert not gc.gemini_configured(purpose=gc.RETRIEVAL)
    assert gc.gemini_route_info(gc.RETRIEVAL) == {"backend": "none", "setting": "api_key"}
    with pytest.raises(ValueError, match="GEMINI_RETRIEVAL_BACKEND=api_key"):
        gc.make_genai_client(purpose=gc.RETRIEVAL)
    assert captured == []
    assert not QueryAnalyzer()._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled


def test_unknown_retrieval_setting_is_an_error(monkeypatch, captured):
    """오타가 조용히 auto(Vertex)가 되면 경로를 고정한 뜻이 사라진다."""
    _both_configured(monkeypatch)
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "aistudio")
    with pytest.raises(ValueError, match="auto / api_key / vertex"):
        gc.gemini_backend(purpose=gc.RETRIEVAL)

    # API 키를 이 칸에 잘못 넣어도 에러 메시지(→ 기동·측정 로그)에 키가 남지 않는다
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "AIzaSy-시험용-가짜-키")
    with pytest.raises(ValueError) as err:
        gc.gemini_backend(purpose=gc.RETRIEVAL)
    assert "aizasy" not in str(err.value).lower() and "가짜" not in str(err.value)
    assert gc.gemini_backend() == "vertex", "용도 없는 호출(크롤링)은 이 설정을 읽지 않는다"


def test_analyzer_and_reranker_build_clients_on_the_retrieval_path(monkeypatch, captured):
    _both_configured(monkeypatch)
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    QueryAnalyzer()._gemini
    GeminiListwiseReranker(config=GeminiListwiseRerankerConfig())._gemini
    assert [c.get("api_key") for c in captured] == ["키", "키"]
    assert not any(c.get("vertexai") for c in captured)


def test_crawl_model_is_separate_from_retrieval_model(monkeypatch):
    monkeypatch.delenv("GEMINI_CRAWL_MODEL_NAME", raising=False)
    monkeypatch.setenv("GEMINI_MODEL_NAME", "검색-모델")
    assert gc.crawl_model_name() == gc.DEFAULT_CRAWL_MODEL == "gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_CRAWL_MODEL_NAME", " 다른-모델 ")
    assert gc.crawl_model_name() == "다른-모델"


def test_server_refuses_to_start_on_retrieval_setting_typo(monkeypatch):
    """오타는 기동에서 실패한다 — 서버가 정상으로 떠 있다가 첫 검색이 500으로 떨어지면 안 된다(PR #31 리뷰)."""
    from fastapi.testclient import TestClient

    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "aistudio")
    with pytest.raises(ValueError, match="GEMINI_RETRIEVAL_BACKEND"):
        with TestClient(backend_main.app):
            pass
    assert opened == [], "설정 검사가 벡터 DB를 열기 전에 끝나야 한다"

    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
