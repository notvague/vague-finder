"""v56 운영 검증 공용 — 실제 모델 호출 차단 · 저장 분석 캐시 분석기 · 가짜 클라이언트를 꽂은 **실제** Gemini listwise 리랭커.

서비스 코드(`src/`)는 바꾸지 않는다. 서버 프로세스(`v56_app.py`)와 사전 조회(`classify_queries.py`)가 같은 함수를 쓴다.

차단 순서가 중요하다 — `block_real_models()`를 **서비스 모듈을 import 하기 전에** 부른다.
  1. `src.common.gemini_client.make_genai_client` → 예외(분석기·리랭커가 실제 클라이언트를 만들 수 없다)
  2. `google.genai.Client` → 예외(어떤 경로로든 SDK 클라이언트를 만들면 바로 드러난다)
  3. `anthropic.Anthropic` 류 → 예외(남아 있을지 모르는 Claude 경로)
  4. 기동 검사 `verify_vertex_setup`은 네트워크 없는 대체 함수로 — 운영 검사 대상이 아니다(설정 점검일 뿐)
`verify_guard()`가 1·2가 실제로 막혔는지 확인한다. 질의 분석은 저장 캐시에만 의존하며 **없는 질의는 예외**(규칙 폴백으로 조용히 채우지 않는다).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("v56")

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

DEFAULT_CACHES = [
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v06_dev_vertex.json",
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v09_dev_vertex.json",
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v10_dev_vertex.json",
    # 봉인 test 42건의 Vertex 분석(v55에서 1회 생성). t014 · t049 장애 재현에만 쓴다(§6) — 새 평가·튜닝에 쓰지 않는다.
    Path(os.getenv("V56_TEST_ANALYSIS_CACHE", str(
        ROOT / "experiments/reranking/results_v55_attr_final_validation/analysis_cache_v10_test_vertex.json"
    ))),
]


class BlockedModelCall(RuntimeError):
    """실제 생성형 모델 클라이언트 생성이 시도됐다 — 검증 조건 위반이므로 조용히 넘기지 않는다."""


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise BlockedModelCall("v56: 실제 Gemini/Claude 클라이언트 생성이 차단됐다(가짜 클라이언트만 허용)")


class _BlockedClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise BlockedModelCall("v56: google.genai.Client 생성이 차단됐다")


_installed = False


def block_real_models() -> None:
    global _installed
    if _installed:
        return
    import src.common.gemini_client as gc

    gc.make_genai_client = _blocked  # type: ignore[assignment]
    try:
        from google import genai
        import google.genai.client as genai_client_mod

        genai.Client = _BlockedClient  # type: ignore[attr-defined]
        genai_client_mod.Client = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass
    try:
        import anthropic

        anthropic.Anthropic = _BlockedClient  # type: ignore[attr-defined]
        anthropic.AsyncAnthropic = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass
    _installed = True


def verify_guard() -> Dict[str, str]:
    """차단이 실제로 걸렸는지 — 둘 다 예외여야 한다. 결과를 돌려주고, 아니면 RuntimeError."""
    import src.common.gemini_client as gc

    result: Dict[str, str] = {}
    try:
        gc.make_genai_client(purpose="retrieval")
        result["make_genai_client"] = "NOT BLOCKED"
    except BlockedModelCall:
        result["make_genai_client"] = "blocked"
    try:
        from google import genai

        genai.Client(vertexai=True, project="x", location="global")
        result["google.genai.Client"] = "NOT BLOCKED"
    except BlockedModelCall:
        result["google.genai.Client"] = "blocked"
    except Exception as exc:  # 다른 예외면 차단이 아니라 설정 오류로 실패한 것 — 차단 아님
        result["google.genai.Client"] = f"NOT BLOCKED ({type(exc).__name__})"
    if any(v != "blocked" for v in result.values()):
        raise RuntimeError(f"v56 모델 호출 차단 확인 실패: {result}")
    return result


def fake_verify_vertex_setup(*_args: Any, **_kwargs: Any) -> dict:
    """기동 검사 대체 — 네트워크 없음. 운영의 설정 점검을 검증하는 것이 아니라 서버를 띄우기 위한 것이다."""
    return {"backend": "blocked(v56)", "location": "none", "model": "fake", "models": ["fake"], "probe": "skipped(v56)"}


# --- 분석 캐시 분석기 ---------------------------------------------------------

class CachedAnalyzer:
    """저장된 분석 캐시로만 답하는 질의 분석기. 라우트의 `_analyze`가 쓰는 인터페이스(`analyze`, `analysis_budget_seconds`)만 제공한다.

    `analyze_async`는 일부러 두지 않는다 — 라우트가 동기 분석기를 스레드로 옮기는 경로(실물과 다른 분기)를 지난다.
    캐시에 없는 질의·규칙 폴백 항목은 **예외**로 요청을 500으로 끝낸다(조용한 폴백 금지).
    """

    analysis_budget_seconds = 20.0

    def __init__(self, cache_paths: Optional[List[Path]] = None) -> None:
        from src.retrieval.analysis_cache import load_cache

        self.by_query: Dict[str, Tuple[str, Dict[str, Any], bool, str]] = {}
        self.sources: List[Dict[str, Any]] = []
        for path in cache_paths or DEFAULT_CACHES:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"분석 캐시 없음: {path}")
            cache = load_cache(path)
            for qid, entry in cache.entries.items():
                self.by_query[entry["query"]] = (qid, entry["analysis"], bool(entry.get("fallback")), str(path))
            analyzer = (cache.meta or {}).get("analyzer", {})
            self.sources.append({
                "path": str(path), "split": (cache.meta or {}).get("split"), "entries": len(cache.entries),
                "gemini_backend": analyzer.get("gemini_backend"), "prompt_sha": analyzer.get("prompt_sha"),
                "postprocess_sha": analyzer.get("postprocess_sha"),
            })
        self.misses: List[str] = []

    def query_id(self, query: str) -> Optional[str]:
        hit = self.by_query.get(query)
        return hit[0] if hit else None

    def analyze(self, query: str):
        from src.backend.schemas.query import QueryAnalysis
        from src.retrieval.analysis_cache import AnalysisCacheError

        hit = self.by_query.get(query)
        if hit is None:
            self.misses.append(query)
            raise AnalysisCacheError(f"v56: 분석 캐시에 없는 질의 — 모델 호출 없이 중단: {query!r}")
        qid, analysis, fallback, _ = hit
        if fallback:
            raise AnalysisCacheError(f"v56: {qid}는 규칙 폴백 분석이라 쓰지 않는다")
        return QueryAnalysis(**analysis)


_analyzer: Optional[CachedAnalyzer] = None


def get_cached_analyzer() -> CachedAnalyzer:
    global _analyzer
    if _analyzer is None:
        extra = [Path(p) for p in os.getenv("V56_ANALYSIS_CACHES", "").split(os.pathsep) if p.strip()]
        _analyzer = CachedAnalyzer(extra or None)
    return _analyzer


# --- 가짜 클라이언트를 꽂은 실제 리랭커 -------------------------------------------

_fake_client = None


def get_fake_client():
    global _fake_client
    if _fake_client is None:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tests.fake_gemini_client import FakeGeminiClient

        _fake_client = FakeGeminiClient(
            mode=os.getenv("V56_FAKE_MODE", "normal"),
            mode_file=os.getenv("V56_FAKE_MODE_FILE") or None,
            log_path=os.getenv("V56_FAKE_LOG") or None,
        )
    return _fake_client


def make_fake_reranker():
    """운영과 같은 `GeminiListwiseReranker`(설정은 환경변수 그대로) + 가짜 SDK 클라이언트. 재시도·파서·예산·상태 보고는 실물이다."""
    from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker

    return GeminiListwiseReranker(client=get_fake_client(), configured=True)


def describe_config() -> Dict[str, Any]:
    from src.retrieval.gemini_listwise_reranker import GeminiListwiseRerankerConfig

    cfg = GeminiListwiseRerankerConfig.from_env()
    keys = ("ATTR_FILTER_WEIGHT", "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY",
            "QDRANT_URL", "QDRANT_PATH", "CONTEXT_WEIGHT", "SEARCH_REFERENCE_YEAR", "RERANKER_BACKEND", "RERANKER_ENABLED",
            "FEEDBACK_COLLECTION_ENABLED", "FEEDBACK_RETRIEVAL_ENABLED", "LYRICS_CACHE_TTL_SECONDS", "SEARCH_WARMUP",
            "GEMINI_RERANK_TIMEOUT_SECONDS", "GEMINI_RERANK_BUDGET_SECONDS", "V56_FAKE_MODE_FILE", "V56_FAKE_LOG", "SEARCH_TIMING_LOG")
    return {
        "env": {k: os.getenv(k, "") for k in keys},
        "reranker": {
            "model_name": cfg.model_name, "enabled": cfg.enabled, "passes": cfg.passes, "max_retries": cfg.max_retries,
            "max_candidates": cfg.max_candidates, "request_timeout_seconds": cfg.request_timeout_seconds,
            "time_budget_seconds": cfg.time_budget_seconds, "rerank_weight": cfg.rerank_weight,
            "use_search_grounding": cfg.use_search_grounding, "rare_fact_verification": cfg.rare_fact_verification,
        },
    }


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
