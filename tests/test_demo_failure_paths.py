"""발표 중에 터질 수 있는 실패 경로 — 서버가 무엇을 남기고 언제 포기하는가.

2026-09-24에 `src/backend/demo_faults.py`로 장애를 주입하고 실제 화면을 돌려
확인한 것들의 회귀 시험이다. 화면 쪽 확인은
`venv/bin/python -m src.backend.check_demo_faults`가 브라우저로 한다.

여기서 고정하는 것 둘.

1. **질의 분석이 규칙 폴백이면 기록에 남는다.** 쿼터가 터져도 검색은 그대로
   돌아서, 결과만 봐서는 폴백인지 알 수 없다. 기록에 없으면 화면이 말할 수 없다.
2. **분석은 정해진 시간 안에 끝난다.** 제한이 없으면 Gemini가 응답하지 않을 때
   요청이 끝나지 않고 화면이 "검색 중…"에서 멈춘다.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval import query_analyzer as qa

QUERY = "비 오는 날 듣기 좋은 노래"


def _model_analysis(**kw) -> QueryAnalysis:
    """Gemini가 제대로 답했을 때의 모양. 필수 필드 기본값은 다른 시험과 같은 관례."""
    base = dict(
        original_query=QUERY,
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
        confidence=0.9,
    )
    base.update(kw)
    return QueryAnalysis(**base)


# ---------------------------------------------------------------------------
# 1. 분석이 폴백이면 기록에 남는다
# ---------------------------------------------------------------------------

class _FixedAnalyzer:
    def __init__(self, analysis: QueryAnalysis) -> None:
        self._analysis = analysis
        self.calls = 0

    def analyze(self, query: str) -> QueryAnalysis:
        self.calls += 1
        return self._analysis


class _StubRouter:
    """검색은 이 시험의 관심사가 아니다. 고정 결과만 돌려준다."""

    async def search(self, analysis, **kw) -> List[MatchingTrack]:
        tracks = [MatchingTrack(id=f"s{i}", score=1.0 - i * 0.01, title=f"곡{i}")
                  for i in range(3)]
        for name in ("candidate_ids_out", "candidate_tracks_out"):
            out = kw.get(name)
            if out is not None:
                out.extend(t.id if name.endswith("ids_out") else t for t in tracks)
        return tracks[: kw.get("top_k", 10)]


def _client(analysis: QueryAnalysis) -> TestClient:
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: _FixedAnalyzer(analysis)
    app.dependency_overrides[get_search_router] = _StubRouter
    return TestClient(app)


def _explain(analysis: QueryAnalysis, **body: Any) -> Dict[str, Any]:
    with _client(analysis) as client:
        payload = {"query": QUERY, "top_k": 3, "explain": True, **body}
        response = client.post("/search", json=payload)
        assert response.status_code == 200, response.text
        return response.json()["explain"]


def test_rule_fallback_analysis_is_flagged():
    """쿼터 초과·타임아웃이면 `_fallback()`이 온다. 그것을 기록이 말해야 한다."""
    explain = _explain(qa._fallback(QUERY))
    assert explain["analysis_fallback"] is True
    assert explain["analysis_fallback_label"]


def test_model_analysis_is_not_flagged():
    assert _explain(_model_analysis())["analysis_fallback"] is False


def test_low_confidence_model_analysis_is_not_called_a_fallback():
    """confidence만 보면 안 된다 — 모델이 낮은 확신도를 준 정상 분석까지 폴백이 된다."""
    analysis = qa._fallback(QUERY).model_copy(update={"song_title": "모델이 찾은 제목"})
    assert analysis.confidence == 0.0
    assert _explain(analysis)["analysis_fallback"] is False


def test_prior_analysis_that_was_a_fallback_is_still_flagged():
    """재질문 턴은 분석을 다시 하지 않는다. 1턴의 폴백이 2턴에도 그대로 쓰인다."""
    fallback = qa._fallback(QUERY)
    explain = _explain(
        _model_analysis(),
        prior_analysis=json.loads(fallback.model_dump_json()),
        turn=1,
    )
    assert explain["analysis_fallback"] is True


# ---------------------------------------------------------------------------
# 2. 분석은 정해진 시간 안에 끝난다
# ---------------------------------------------------------------------------

class _FakeGemini:
    """호출마다 config를 기록하고, 정해진 만큼 자다가 실패한다."""

    def __init__(self, sleep: float = 0.0) -> None:
        self.configs: List[Dict[str, Any]] = []
        self._sleep = sleep
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: Dict[str, Any]):
        self.configs.append(config)
        if self._sleep:
            time.sleep(self._sleep)
        raise RuntimeError("호출 실패(시험)")

    @property
    def timeouts(self) -> List[int]:
        return [c["http_options"]["timeout"] for c in self.configs]


def test_every_call_carries_a_shrinking_timeout(monkeypatch):
    """제한을 넘기지 않으면 라이브러리는 **무한정** 기다린다. 그리고 그 제한은
    남은 예산이어야 한다 — 시도마다 같은 값을 주면 3회가 예산의 3배가 된다.

    재시도 대기는 짧게 바꿔 둔다. 1초를 실제로 자는 것은 이 시험의 관심사가 아니다.
    """
    monkeypatch.setenv("QUERY_ANALYSIS_TIMEOUT_SECONDS", "20")
    # `qa.time`은 time 모듈 자체다 — 진짜 sleep을 먼저 붙잡아야 자기 자신을 부르지 않는다
    real_sleep = time.sleep
    monkeypatch.setattr(qa.time, "sleep", lambda s: real_sleep(min(s, 0.05)))
    analyzer = qa.QueryAnalyzer(api_key="시험용")
    fake = _FakeGemini()
    analyzer._client = fake

    analyzer.analyze(QUERY)

    assert len(fake.timeouts) == 3, "빠른 실패는 3회 다 시도해야 한다"
    for timeout in fake.timeouts:
        assert 0 < timeout <= 20_000, f"예산 밖의 제한 시간: {timeout}ms"
    assert fake.timeouts == sorted(fake.timeouts, reverse=True), fake.timeouts
    assert fake.timeouts[-1] < fake.timeouts[0], "남은 예산으로 줄어들지 않았다"


def test_a_spent_budget_stops_the_retries(monkeypatch):
    """한 번에 예산을 다 쓰면 더 시도하지 않고 규칙 폴백으로 내려온다."""
    monkeypatch.setenv("QUERY_ANALYSIS_TIMEOUT_SECONDS", "0.3")
    analyzer = qa.QueryAnalyzer(api_key="시험용")
    fake = _FakeGemini(sleep=0.4)
    analyzer._client = fake

    started = time.monotonic()
    analysis = analyzer.analyze(QUERY)
    elapsed = time.monotonic() - started

    assert len(fake.configs) == 1, f"예산을 다 쓰고도 {len(fake.configs)}회 시도했다"
    assert analysis.confidence == 0.0
    assert elapsed < 2.0, f"예산 0.3초인데 {elapsed:.1f}초가 걸렸다"


def test_no_api_key_falls_back_without_calling(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    analyzer = qa.QueryAnalyzer(api_key="")
    assert analyzer.analyze(QUERY).confidence == 0.0


# ---------------------------------------------------------------------------
# 3. 예산은 **벽시계 상한**이어야 한다
#
# 분석기에 넘긴 값은 라이브러리를 거쳐 httpx의 timeout이 된다. httpx는 연결·읽기
# 같은 단계마다 따로 세고, 읽기는 *조각 하나*를 기다리는 시간이다. 그래서 응답이
# 조금씩 도착하면 예산을 넘겨 끝난다 — 0.1초 제한에 0.68초를 확인했다.
# 재시도 사이의 deadline 검사도 **진행 중인** 호출은 끊지 못한다.
# ---------------------------------------------------------------------------

import asyncio

from src.backend.api.routes import search as search_route_module
from src.backend.api.routes.search import _analyze


class _SlowAnalyzer:
    """예산을 한참 넘겨 도는 분석기. 취소되면 그것을 기록한다."""

    def __init__(self, budget: float = 0.2, runs_for: float = 5.0) -> None:
        self.analysis_budget_seconds = budget
        self.runs_for = runs_for
        self.cancelled = False

    async def analyze_async(self, query: str) -> QueryAnalysis:
        try:
            await asyncio.sleep(self.runs_for)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return _model_analysis()


def test_a_call_that_overruns_is_capped_by_the_wall_clock(monkeypatch):
    """진행 중인 호출이 예산을 넘겨도 요청은 **예산+여유** 안에 끝난다."""
    monkeypatch.setattr(
        search_route_module, "ANALYSIS_DEADLINE_MARGIN_SECONDS", 0.1
    )
    analyzer = _SlowAnalyzer(budget=0.2, runs_for=5.0)

    async def scenario():
        started = time.monotonic()
        analysis = await _analyze(analyzer, QUERY)
        return time.monotonic() - started, analysis

    elapsed, analysis = asyncio.run(scenario())
    assert elapsed < 1.0, f"벽시계 상한이 걸리지 않았다: {elapsed:.1f}초"
    assert elapsed >= 0.2, "예산보다 먼저 끊으면 멀쩡한 분석까지 자른다"
    assert analysis.confidence == 0.0, "규칙 분석으로 검색을 계속해야 한다"


def test_the_overrunning_call_is_cancelled_not_just_abandoned(monkeypatch):
    """남은 일을 어떻게 하는가 — **끊는다.**

    기다림만 끊고 호출을 두면 연결이 남고 결과는 버려진다. 비동기 경로를 쓰는
    이유가 이것이다.
    """
    monkeypatch.setattr(
        search_route_module, "ANALYSIS_DEADLINE_MARGIN_SECONDS", 0.1
    )
    analyzer = _SlowAnalyzer(budget=0.1, runs_for=5.0)
    asyncio.run(_analyze(analyzer, QUERY))
    assert analyzer.cancelled, "진행 중인 분석이 취소되지 않았다"


def test_a_real_dripping_response_does_not_outlast_the_budget(monkeypatch):
    """**실제 SDK와 httpx**에 조금씩 흘리는 응답을 물려 본다.

    네트워크만 모의한다. `http_options.timeout`만으로는 여기서 못 막는다 —
    읽기 제한은 조각마다 다시 세기 때문이다.
    """
    import http.server
    import socketserver
    import threading

    monkeypatch.setattr(
        search_route_module, "ANALYSIS_DEADLINE_MARGIN_SECONDS", 0.1
    )

    class _Drip(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 규약
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(40):          # 0.05초 × 40 = 2초
                    time.sleep(0.05)
                    self.wfile.write(b"1\r\n \r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                          # 취소되면 여기로 온다 — 정상이다

        def log_message(self, *a):
            return

    with socketserver.TCPServer(("127.0.0.1", 0), _Drip) as server:
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        # 키는 **아스키여야 한다.** 한글 키는 HTTP 헤더 인코딩에서 먼저 터져
        # 드립 서버에 닿지도 못한다 — 그러면 이 시험은 엉뚱한 이유로 통과한다.
        analyzer = qa.QueryAnalyzer(api_key="drip-test-key")
        analyzer._budget_seconds = 0.3
        analyzer._client = qa.genai.Client(
            api_key="drip-test-key",
            http_options={"base_url": f"http://127.0.0.1:{port}"},
        )
        started = time.monotonic()
        analysis = asyncio.run(_analyze(analyzer, QUERY))
        elapsed = time.monotonic() - started
        server.shutdown()

    assert elapsed < 1.5, f"예산 0.3초인데 {elapsed:.1f}초가 걸렸다"
    assert analysis.confidence == 0.0, "규칙 분석으로 내려와야 한다"


# ---------------------------------------------------------------------------
# 4. 재시도가 앞서 만든 정정을 잃지 않는다
#
# 모달리티 경계를 어기면 다음 프롬프트에 정정을 붙인다. 그 사이에 통신 오류가
# 한 번 나면 **정정이 사라져** 마지막 시도가 최초 프롬프트로 돌아갔다
# (정정 포함 여부 False→True→False). 같은 오류를 그대로 되풀이하게 된다.
# ---------------------------------------------------------------------------

from src.retrieval.modality_queries import ModalityQueryValidationError


class _CorrectionSequence:
    """1차 모달리티 검증 실패 → 2차·3차 통신 오류. 시도마다 프롬프트를 기록한다."""

    def __init__(self) -> None:
        self.saw_correction: List[bool] = []
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: Dict[str, Any]):
        self.saw_correction.append("REQUIRED CORRECTION" in contents)
        if len(self.saw_correction) == 1:
            raise ModalityQueryValidationError("image 질의에 소리 묘사가 섞였다")
        raise RuntimeError("일시적 통신 오류")

    @property
    def aio(self):
        outer = self

        class _AsyncModels:
            async def generate_content(self, **kw):
                return outer.generate_content(**kw)

        return type("_Aio", (), {"models": _AsyncModels()})()


def _analyzer_with(fake, monkeypatch) -> qa.QueryAnalyzer:
    real_sleep = time.sleep
    monkeypatch.setattr(qa.time, "sleep", lambda s: real_sleep(0))
    analyzer = qa.QueryAnalyzer(api_key="시험용")
    analyzer._client = fake
    return analyzer


def test_a_transient_error_keeps_the_correction_from_the_previous_attempt(monkeypatch):
    fake = _CorrectionSequence()
    _analyzer_with(fake, monkeypatch).analyze("빨간 앨범 커버 노래")
    assert fake.saw_correction == [False, True, True], (
        f"통신 오류가 정정을 지웠다: {fake.saw_correction}"
    )


def test_the_async_loop_keeps_the_correction_too(monkeypatch):
    fake = _CorrectionSequence()
    analyzer = _analyzer_with(fake, monkeypatch)
    asyncio.run(analyzer.analyze_async("빨간 앨범 커버 노래"))
    assert fake.saw_correction == [False, True, True], (
        f"통신 오류가 정정을 지웠다: {fake.saw_correction}"
    )


# ---------------------------------------------------------------------------
# 5. 주석이 **Python 3.11에서도** 평가된다
#
# 로컬은 3.14라 주석 평가가 지연되지만(PEP 649) Dockerfile은 3.11이다. 3.11은
# def 시점에 주석을 평가하므로, import 하나가 빠지면 **모듈을 부르는 순간**
# NameError가 난다 — 로컬 시험은 전부 통과한 채로 배포가 깨진다.
# 실제로 `_parse(..., response: Any)`를 넣고 `Any`를 안 가져와 그렇게 됐다.
# ---------------------------------------------------------------------------

def test_annotations_resolve_on_an_older_python():
    """지금 세션이 이미 불러온 우리 모듈의 주석이 전부 풀리는지 본다.

    새로 import 하지 않는다 — 모델을 올리는 모듈까지 건드리면 시험이 무거워진다.
    시험이 지나간 모듈은 거의 다 여기 들어 있다.
    """
    import inspect
    import sys
    import typing

    broken: List[str] = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("src.") or module is None:
            continue
        for holder_name, holder in vars(module).items():
            if getattr(holder, "__module__", None) != name:
                continue
            targets = [holder] if inspect.isfunction(holder) else []
            if inspect.isclass(holder):
                targets = [
                    member for member in vars(holder).values()
                    if inspect.isfunction(member)
                ]
            for target in targets:
                if not getattr(target, "__annotations__", None):
                    continue
                try:
                    typing.get_type_hints(target)
                except Exception as exc:  # NameError가 대부분이다
                    broken.append(f"{name}.{holder_name}.{target.__name__}: {exc}")

    assert not broken, "주석을 풀 수 없다(3.11에서 import가 깨진다):\n  " + "\n  ".join(broken)
