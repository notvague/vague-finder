"""Gemini 클라이언트 경로 선택 — Vertex AI(GCP_PROJECT_ID) 또는 AI Studio API 키.

실제 호출은 하지 않는다. `google.genai.Client`를 가짜로 바꿔 어떤 인자로 만들어지는지만 본다.
"""
import pytest

from src.common import gemini_client as gc
# 모듈 맨 위에서 import한다 — 두 모듈은 import될 때 load_dotenv()를 부른다. 시험 함수 안에서 처음 import하면
# fixture가 env를 지운 뒤에 .env가 다시 읽혀 GEMINI_API_KEY 등이 되살아나고, monkeypatch가 되돌리지 못해 뒤 시험까지 샌다.
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.query_analyzer import QueryAnalyzer


@pytest.fixture
def captured(monkeypatch):
    import google.genai as genai_mod

    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    for name in ("GCP_PROJECT_ID", "GCP_LOCATION", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_vertex_when_project_is_set(monkeypatch, captured):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.setenv("GEMINI_API_KEY", "키도-있음")
    gc.make_genai_client(http_options="opts")
    assert captured == [{"vertexai": True, "project": "proj-1", "location": "global", "http_options": "opts"}]
    assert gc.gemini_backend() == "vertex"
    assert gc.gemini_route_info() == {"backend": "vertex", "project": "proj-1", "location": "global"}


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
