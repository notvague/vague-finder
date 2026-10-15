"""테스트 전역 기본값.

**테스트는 모델을 올리지 않는다.** `lifespan`을 타는 테스트가 몇 개 있는데, 거기서
예열이 켜지면 백그라운드 스레드가 KoE5·SigLIP2·CLAP·CE를 내려받아 올리고 로컬
Qdrant 저장 폴더까지 연다. 개발 서버가 떠 있는 채로 테스트를 돌리면 그 폴더를 두
프로세스가 여는 셈이라 한쪽이 죽는다.

`setdefault`이므로 일부러 켜서 돌리는 것은 그대로 된다.

**테스트는 Vertex AI도 ADC도 건드리지 않는다.** `src.common.gemini_client`의 ADC 읽기(`_load_default_credentials`)와
기동·측정 시작 검사(`verify_vertex_setup`)를 모든 테스트에서 바꿔 끼운다 — .env에 프로젝트가 있는 PC에서 `QueryAnalyzer()`·
리랭커가 실제 Vertex를 부르거나, ADC가 없는 PC에서 앱 기동 테스트가 떨어지지 않게. 바꿔 끼운 자격 증명은 토큰 요청에서
예외를 내므로 어떤 경로로든 실제 요청이 나가려 하면 네트워크 전에 멈추고, 호출부의 폴백(규칙 분석 · 검색 순서 + failed)으로
끝난다. 실제 구현을 시험하려면 private 이름(`_load_default_credentials` · `_verify_vertex_setup`)을 직접 부른다
(tests/test_gemini_client.py).
"""

import os

import google.auth.credentials
import pytest

os.environ.setdefault("SEARCH_WARMUP", "0")
os.environ.setdefault("FEEDBACK_COLLECTION_ENABLED", "0")


class OfflineCredentials(google.auth.credentials.Credentials):
    """토큰을 받지 않는 자격 증명. 요청이 나가려는 순간 예외 — 네트워크 전에 멈춘다."""

    def refresh(self, request):
        raise RuntimeError("테스트는 Vertex AI를 부르지 않는다 — ADC 토큰 요청 차단(tests/conftest.py)")


def _stub_verify(purpose=None, *, model=None, probe=True):
    return {"backend": "vertex", "location": "test", "setting": "vertex",
            "model": model or "test-model", "probe": "skipped-in-tests"}


@pytest.fixture(autouse=True)
def _offline_vertex(monkeypatch):
    from src.common import gemini_client as gc

    gc.reset_credentials_cache()
    monkeypatch.setattr(gc, "_load_default_credentials", OfflineCredentials)
    monkeypatch.setattr(gc, "verify_vertex_setup", _stub_verify)
    # 개발자 .env의 옛 값(GEMINI_RETRIEVAL_BACKEND=api_key)은 지금 설정 오류라 분석기·리랭커 생성에서 예외가 난다 —
    # 테스트는 .env와 무관해야 한다. 값을 시험하는 테스트는 직접 setenv한다.
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)
    yield
    gc.reset_credentials_cache()
