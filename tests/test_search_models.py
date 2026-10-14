"""선택 모델이 분석·리랭킹 전체에 적용되고 요청 간 섞이지 않는지 검증한다."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.backend import main as backend_main  # 맨 위에서 — import 때 load_dotenv()가 돌므로 fixture가 env를 지운 뒤 .env가 되살아나지 않게
from src.backend.api import dependencies as deps
from src.backend.api.routes.search import available_models
from src.backend.schemas.search import SearchRequest
from src.common import claude_client as cc
from src.common.search_models import default_provider, unavailable_reason
from src.retrieval.analysis_cache import analyzer_fingerprint
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback


@pytest.fixture
def selection(monkeypatch):
    monkeypatch.setenv('SEARCH_MODEL_PROVIDER', 'claude')
    monkeypatch.setenv('CLAUDE_API', 'fake-claude')
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-gemini')
    monkeypatch.setenv('GEMINI_RETRIEVAL_BACKEND', 'api_key')
    # CLI의 과거 설정은 명시적인 웹 선택을 덮어쓰면 안 된다.
    monkeypatch.setenv('RERANKER_BACKEND', 'gemini_listwise')
    monkeypatch.setattr(deps, 'QueryAnalyzer', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(deps, '_make_reranker', lambda backend: SimpleNamespace(backend=backend))
    monkeypatch.setattr(deps, 'SearchRouter', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(deps, 'ContextQdrantSearch', lambda *a, **kw: None)
    for name in ('get_search_service', 'get_image_embedder', 'get_audio_embedder',
                 'get_vector_client', 'get_lyrics_exact_search_service', 'get_text_embedder', 'get_feedback_retriever'):
        monkeypatch.setattr(deps, name, lambda: object())
    deps.get_query_analyzer.cache_clear()
    deps.get_search_router.cache_clear()
    app = FastAPI()

    @app.post('/probe')
    def probe(request: SearchRequest, analyzer=Depends(deps.get_query_analyzer), router=Depends(deps.get_search_router)):
        return {'provider': analyzer.provider, 'reranker': router.reranker.backend}

    with TestClient(app) as client:
        yield client
    deps.get_query_analyzer.cache_clear()
    deps.get_search_router.cache_clear()


def test_flat_request_body_and_default_selection(selection):
    for provider in ('claude', 'gemini', None):
        body = {'query': '노래', 'model_provider': provider}
        response = selection.post('/probe', json=body)
        assert response.status_code == 200, response.text
        expected = provider or 'claude'
        assert response.json() == {'provider': expected, 'reranker': f'{expected}_listwise'}


def test_concurrent_provider_requests_are_isolated(selection):
    choices = ['claude', 'gemini'] * 12
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: selection.post('/probe', json={'query': '노래', 'model_provider': p}).json(), choices))
    assert [r['provider'] for r in results] == choices
    assert [r['reranker'] for r in results] == [f'{p}_listwise' for p in choices]
    assert len(deps.get_search_router.values()) == 2


def test_unavailable_choice_does_not_fall_back_to_other_provider(selection, monkeypatch):
    monkeypatch.setenv('CLAUDE_API', '')
    monkeypatch.setenv('ANTHROPIC_API_KEY', '')
    for payload in ({'query': '노래'}, {'query': '노래', 'model_provider': 'claude'}):
        response = selection.post('/probe', json=payload)
        assert response.status_code == 503 and 'CLAUDE_API' in response.json()['detail']
    assert selection.post('/probe', json={'query': '노래', 'model_provider': 'gemini'}).status_code == 200
    assert selection.post('/probe', json={'query': '노래', 'model_provider': 'vertex'}).status_code == 422


def test_search_options_make_claude_unavailable_instead_of_failing_the_first_search(selection, monkeypatch):
    """Google Search를 쓰는 리랭커 옵션은 Claude listwise와 함께 만들 수 없다. 전에는 그 ValueError가 첫 Claude 검색의
    의존성에서 올라와 500이었다(#63 리뷰). 이제 기동 검사·503·/models가 같은 판정을 본다."""
    for flag in ('GEMINI_RERANK_USE_SEARCH', 'GEMINI_RERANK_RARE_FACT_VERIFY'):
        with monkeypatch.context() as m:
            m.setenv(flag, '1')
            assert flag in unavailable_reason('claude') and unavailable_reason('gemini') is None
            response = selection.post('/probe', json={'query': '노래', 'model_provider': 'claude'})
            assert response.status_code == 503 and flag in response.json()['detail']
            assert selection.post('/probe', json={'query': '노래', 'model_provider': 'gemini'}).status_code == 200
            options = available_models().model_dump()
            assert [(o['provider'], o['available']) for o in options['models']] == [('claude', False), ('gemini', True)]
            # 리랭킹을 끈 실행은 옵션을 쓰지 않는다
            m.setenv('RERANKER_ENABLED', '0')
            assert unavailable_reason('claude') is None


def test_options_are_public_metadata_only(selection, monkeypatch):
    monkeypatch.setenv('CLAUDE_MODEL_NAME', 'claude-sonnet-5-5')
    data = available_models().model_dump()
    assert data['default'] == 'claude'
    assert all(m['available'] for m in data['models'])
    assert data['models'][0]['model'] == 'claude-sonnet-5-5'
    assert 'fake-' not in json.dumps(data)


def test_default_and_fingerprint_follow_configuration(monkeypatch):
    monkeypatch.delenv('SEARCH_MODEL_PROVIDER', raising=False)
    assert default_provider() == 'claude'
    claude, gemini = analyzer_fingerprint('claude'), analyzer_fingerprint('gemini')
    assert claude['provider'] != gemini['provider']
    assert claude['temperature'] is None and gemini['temperature'] == 0
    monkeypatch.setenv('CLAUDE_MODEL_NAME', 'replacement-model')
    assert analyzer_fingerprint('claude')['model_name'] == 'replacement-model'
    monkeypatch.setenv('SEARCH_MODEL_PROVIDER', 'invalid')
    with pytest.raises(ValueError):
        default_provider()


def test_cli_entry_points_pick_the_default_or_an_explicit_model(selection, monkeypatch):
    """인자 없는 호출은 서버 기본 모델(CLI), `for_provider`는 지정한 모델 — 요청용 객체와는 따로 둔다."""
    monkeypatch.setenv('RERANKER_BACKEND', '')
    monkeypatch.setattr(deps, 'get_reranker', lambda: SimpleNamespace(backend='cli'))
    assert deps.get_query_analyzer().provider == 'claude'
    assert deps.get_query_analyzer.for_provider('gemini').provider == 'gemini'
    assert deps.get_query_analyzer.for_provider('gemini') is deps.get_query_analyzer.for_provider('gemini')
    assert deps.get_search_router().reranker.backend == 'cli'
    assert deps.get_search_router.peek() is deps.get_search_router()
    selection.post('/probe', json={'query': '노래', 'model_provider': 'claude'})
    assert deps.get_search_router().reranker.backend == 'cli', "요청용 라우터와 CLI 라우터는 다른 객체다"
    assert len(deps.get_search_router.values()) == 2


def test_server_refuses_to_start_on_model_setting_errors(monkeypatch):
    """설정 오류는 **기동에서** 실패한다 — 서버가 정상으로 떠 있다가 첫 검색이 500·503으로 떨어지면 안 된다(PR #31 리뷰).

    기본 모델의 키가 없는 경우도 같다: 철자만 보면 Gemini 키만 있는 환경에서 서버가 뜬 뒤 model_provider를
    생략한 모든 검색이 503이 된다(2026-10-14 리뷰).
    """
    from fastapi.testclient import TestClient

    opened = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: opened.append(1))
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    monkeypatch.setenv("SEARCH_WARMUP", "0")
    monkeypatch.setenv("CLAUDE_API", "fake-claude")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    monkeypatch.delenv("SEARCH_MODEL_PROVIDER", raising=False)
    for env, expected in [
        ({"SEARCH_MODEL_PROVIDER": "vertex"}, "SEARCH_MODEL_PROVIDER"),
        ({"GEMINI_RETRIEVAL_BACKEND": "vertex"}, "GEMINI_RETRIEVAL_BACKEND"),
        ({"CLAUDE_API": "", "ANTHROPIC_API_KEY": ""}, "CLAUDE_API"),        # 기본 모델(claude)의 키가 없다
        ({"GEMINI_RERANK_RARE_FACT_VERIFY": "1"}, "GEMINI_RERANK_RARE_FACT_VERIFY"),   # Claude와 못 쓰는 리랭커 옵션
    ]:
        with monkeypatch.context() as m:
            for name, value in env.items():
                m.setenv(name, value)
            with pytest.raises(ValueError, match=expected):
                with TestClient(backend_main.app):
                    pass
    assert opened == [], "설정 검사가 벡터 DB를 열기 전에 끝나야 한다"

    # Gemini 키만 있는 환경은 기본 모델을 gemini로 두면 뜬다. 선택지는 Claude를 사용 불가로 보고한다.
    monkeypatch.setenv("CLAUDE_API", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SEARCH_MODEL_PROVIDER", "gemini")
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
        options = client.get("/api/v1/search/models").json()
    assert options["default"] == "gemini"
    assert [(m["provider"], m["available"]) for m in options["models"]] == [("claude", False), ("gemini", True)]
    assert opened == [1]


def test_claude_analysis_sync_and_async_never_build_google_client(monkeypatch):
    import src.retrieval.query_analyzer as qa
    monkeypatch.setenv('CLAUDE_API', 'fake')
    for name in ('CLAUDE_BUDGET_USD', 'CLAUDE_USAGE_LOG', 'CLAUDE_RAW_LOG'):
        monkeypatch.setenv(name, '')
    def forbidden(**kw):
        raise AssertionError('Google client must not be built')
    monkeypatch.setattr(qa, 'make_genai_client', forbidden)
    seen = []
    payload = rule_fallback('슬픈 발라드').model_dump()
    payload['confidence'] = 0.8
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'content': [{'type': 'text', 'text': '```json\n'+json.dumps(payload)+'\n```'}],
                                        'stop_reason': 'end_turn', 'usage': {'input_tokens': 10, 'output_tokens': 20}})
    adapter = cc.ClaudeGenaiClient(api_key='fake', transport=httpx.MockTransport(respond))
    monkeypatch.setattr(qa, 'make_claude_genai_client', lambda **kw: adapter)
    analyzer = QueryAnalyzer(provider='claude')
    assert analyzer.analyze('슬픈 발라드').confidence == 0.8
    assert asyncio.run(analyzer.analyze_async('슬픈 발라드')).confidence == 0.8
    assert len(seen) == 2
    assert all(r['model'] == cc.claude_model_name() and 'temperature' not in r for r in seen)


def test_claude_billing_failure_is_not_retried_or_sent_to_google(monkeypatch):
    import src.retrieval.query_analyzer as qa
    monkeypatch.setenv('CLAUDE_API', 'fake')
    analyzer = QueryAnalyzer(provider='claude')
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise cc.ClaudeAPIError(402, 'billing_error', 'insufficient credit')
    analyzer._client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    result = analyzer.analyze('슬픈 발라드')
    assert result.confidence == 0.0
    assert len(calls) == 1


def test_shutdown_closes_all_provider_routers_and_clears_model_caches(monkeypatch):
    closed = []
    routers = [SimpleNamespace(shutdown=lambda p=p: closed.append(p)) for p in ('claude', 'gemini')]
    monkeypatch.setattr(deps.get_search_router, 'values', lambda: routers)
    deps.close_vector_client()
    assert closed == ['claude', 'gemini']
    assert deps.get_query_analyzer.values() == ()


def test_claude_http_pool_is_closed_on_owner_loop(monkeypatch):
    client = cc.ClaudeGenaiClient(api_key='fake', transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': '{}'}],
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    })))
    for name in ('CLAUDE_BUDGET_USD', 'CLAUDE_USAGE_LOG', 'CLAUDE_RAW_LOG'):
        monkeypatch.setenv(name, '')
    client.generate(model='claude-sonnet-5-5', contents='test')
    transport = client._http
    client.close()
    assert transport.is_closed
    assert client._http is None
    client.close()
