"""측정 runinfo의 `gemini` 칸과 시작 검사(preflight)가 **실제로 Gemini를 부르는 실행**에서만 경로·모델을 적고 확인하는지.

PR #64 리뷰: CE 리랭커 + 분석 캐시(v22·v27~v31 CE 기준선 재현)도 .env에 프로젝트가 있으면 runinfo에 `gemini.backend=vertex`가
적혀, "경로가 다른 실행끼리는 비교하지 않는다"는 규칙이 무너졌다. main(#63)에서는 이 경우 `unused`였다.
"""
import argparse

import pytest

from src.common import gemini_client as gc
from src.retrieval import evaluate_search_accuracy as esa
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from src.retrieval.reranker import MusicReranker


@pytest.fixture
def runinfo_env(monkeypatch, tmp_path):
    """Mongo·Qdrant를 건드리지 않는 collect_run_info."""
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.setenv("BM25_PARAMS_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(esa, "_lyrics_fingerprint", lambda: {"stub": True})
    monkeypatch.setattr(esa, "_count_points", lambda settings: 0)
    return argparse.Namespace(input="q.csv", split="dev")


def _gemini_reranker(model_name="rerank-model", enabled=True):
    return GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(model_name=model_name, enabled=enabled), configured=True)


def test_runinfo_marks_gemini_unused_when_nothing_calls_it(runinfo_env):
    cache = {"mode": "cache", "path": "c.json"}
    live = {"mode": "live"}
    # CE + 캐시: Gemini 호출 0 — 프로젝트가 있어도 vertex로 적지 않는다
    assert esa.collect_run_info(runinfo_env, reranker=MusicReranker(), analysis_source=cache)["gemini"] == {
        "backend": "unused", "setting": "vertex",
    }
    # 리랭킹을 끈 Gemini listwise + 캐시도 호출이 없다
    assert esa.collect_run_info(runinfo_env, reranker=_gemini_reranker(enabled=False), analysis_source=cache)["gemini"]["backend"] == "unused"
    # 실시간 분석이거나 켜진 Gemini listwise면 실제 경로다
    for reranker, source in ((MusicReranker(), live), (_gemini_reranker(), cache)):
        assert esa.collect_run_info(runinfo_env, reranker=reranker, analysis_source=source)["gemini"] == {
            "backend": "vertex", "location": "global", "setting": "vertex",
        }


def test_runinfo_unused_still_records_an_invalid_setting(runinfo_env, monkeypatch):
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    info = esa.collect_run_info(runinfo_env, reranker=MusicReranker(), analysis_source={"mode": "cache"})
    assert info["gemini"] == {"backend": "unused", "setting": "invalid"}


def test_preflight_checks_the_reranker_model_and_skips_runs_without_gemini(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-1")
    calls = []

    def fake_verify(purpose=None, *, model=None, extra_models=(), probe=True):
        calls.append((purpose, model, list(extra_models)))
        return {"backend": "vertex", "location": "global", "model": "analysis-model",
                "models": ["analysis-model", *extra_models], "probe": "ok"}

    monkeypatch.setattr(gc, "verify_vertex_setup", fake_verify)
    assert esa.preflight_gemini(MusicReranker(), live_analysis=False) is None
    assert esa.preflight_gemini(_gemini_reranker(enabled=False), live_analysis=False) is None
    assert calls == [], "Gemini를 부르지 않는 실행은 확인 호출도 하지 않는다"

    assert esa.preflight_gemini(_gemini_reranker("rerank-model"), live_analysis=False)["models"] == ["analysis-model", "rerank-model"]
    assert esa.preflight_gemini(MusicReranker(), live_analysis=True)["models"] == ["analysis-model"]
    assert calls == [(gc.RETRIEVAL, None, ["rerank-model"]), (gc.RETRIEVAL, None, [])]
