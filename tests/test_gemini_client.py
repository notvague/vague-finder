"""Vertex/ADC 환경이 남아 있어도 Gemini가 AI Studio만 사용하는지 확인한다."""
import pytest

from src.common import gemini_client as gc
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.query_analyzer import QueryAnalyzer


@pytest.fixture
def captured(monkeypatch):
    import google.genai
    calls = []
    monkeypatch.setattr(google.genai, 'Client', lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setenv('GEMINI_API_KEY', '')
    monkeypatch.setenv('GEMINI_RETRIEVAL_BACKEND', '')
    monkeypatch.setenv('GCP_PROJECT_ID', 'old-project')
    monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', 'true')
    return calls


@pytest.mark.parametrize('purpose', [None, gc.RETRIEVAL])
def test_api_key_only_even_with_vertex_environment(monkeypatch, captured, purpose):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    gc.make_genai_client(purpose=purpose, http_options='opts')
    assert captured == [{'vertexai': False, 'api_key': 'test-key', 'http_options': 'opts'}]
    assert gc.gemini_backend(purpose=purpose) == 'api_key'
    assert 'old-project' not in str(gc.gemini_route_info(purpose))


def test_project_and_adc_do_not_count_as_credentials(captured):
    assert not gc.gemini_configured()
    assert not QueryAnalyzer(provider='gemini')._configured
    assert not GeminiListwiseReranker(config=GeminiListwiseRerankerConfig()).enabled
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        gc.make_genai_client()
    assert captured == []


def test_explicit_key(captured):
    gc.make_genai_client(api_key='explicit')
    assert captured[0]['api_key'] == 'explicit'
    assert captured[0]['vertexai'] is False


@pytest.mark.parametrize('setting', ['', 'auto', 'api_key', 'ai_studio'])
def test_legacy_settings_never_select_vertex(monkeypatch, captured, setting):
    monkeypatch.setenv('GEMINI_RETRIEVAL_BACKEND', setting)
    assert gc.backend_setting(gc.RETRIEVAL) == 'api_key'
    assert gc.gemini_backend(purpose=gc.RETRIEVAL) == 'none'


@pytest.mark.parametrize('setting', ['vertex', 'secret-pasted-by-mistake'])
def test_removed_or_invalid_route_rejected_without_echoing_value(monkeypatch, captured, setting):
    monkeypatch.setenv('GEMINI_RETRIEVAL_BACKEND', setting)
    with pytest.raises(ValueError) as error:
        gc.make_genai_client(api_key='test', purpose=gc.RETRIEVAL)
    assert setting not in str(error.value)
    assert captured == []


def test_crawl_model_is_separate(monkeypatch):
    monkeypatch.setenv('GEMINI_MODEL_NAME', 'search-model')
    monkeypatch.delenv('GEMINI_CRAWL_MODEL_NAME', raising=False)
    assert gc.crawl_model_name() == gc.DEFAULT_CRAWL_MODEL
    monkeypatch.setenv('GEMINI_CRAWL_MODEL_NAME', 'crawl-model')
    assert gc.crawl_model_name() == 'crawl-model'
