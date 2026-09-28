"""요청 시간 계측 — 대기와 실행을 가르고, 겹친 시간을 더하지 않는다.

이 계측이 지켜야 할 것은 둘이다.

1. **재는 행위가 결과를 바꾸지 않는다.** 끈 자리(`NULL_TIMER`)에서는 감싼 함수가
   그대로 돌고 아무것도 쌓이지 않아야 한다. 평가 스크립트와 테스트가 그 경로다.
2. **겹쳐 도는 경로의 시간을 합치지 않는다.** 스레드풀이 4칸인데 최대 10개가
   제출되므로, 합치면 실제 경과보다 크게 나오고 없는 병목을 만들어 낸다.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.retrieval import timing


def _sleep_then(value, seconds=0.05):
    def run():
        time.sleep(seconds)
        return value

    return run


def test_null_timer_runs_the_work_and_records_nothing():
    """끈 자리에서도 감싼 함수는 그대로 돈다.

    계측을 넣느라 평가 경로의 동작이 달라지면 v17 숫자를 다시 방어해야 한다.
    """
    called = []

    def work(a, b):
        called.append((a, b))
        return a + b

    runner = timing.NULL_TIMER.job("path.text", work, 1, 2)
    assert runner() == 3
    assert called == [(1, 2)]
    assert timing.NULL_TIMER.to_dict() == {}
    with timing.NULL_TIMER.step("무시"):
        pass


def test_pool_wait_is_separated_from_run_time():
    """칸이 하나뿐인 풀에서 뒤에 선 작업은 대기가 실행과 따로 잡힌다.

    이걸 합쳐 버리면 멀쩡한 경로가 "느린 경로"로 보인다 — 실제로는 앞 작업이
    끝나기를 기다린 시간이다.
    """
    timer = timing.TimingRecorder(query="q")
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(timer.job("path.first", _sleep_then("a", 0.08)))
        second = pool.submit(timer.job("path.second", _sleep_then("b", 0.01)))
        assert first.result() == "a"
        assert second.result() == "b"

    waited = timer.find("path.second")
    assert waited is not None
    # 앞 작업(80ms)이 끝나야 시작하므로 대기가 실행보다 훨씬 크다.
    assert waited.wait_ms > 40
    assert waited.run_ms < 40
    assert timer.find("path.first").wait_ms < 40


def test_parallel_paths_are_not_summed_into_the_total():
    """같이 돈 두 경로의 합은 실제 경과보다 크다. 그래서 더하지 않는다."""
    timer = timing.TimingRecorder()
    with timer.step("search.paths"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(timer.job(f"path.{i}", _sleep_then(i, 0.06)))
                for i in range(2)
            ]
            for future in futures:
                future.result()

    paths = [s for s in timer.spans if s.name.startswith("path.")]
    assert len(paths) == 2
    parallel_sum = sum(s.run_ms for s in paths)
    elapsed = timer.find("search.paths").run_ms
    assert parallel_sum > elapsed
    # 겹침을 보려면 길이만으로는 부족하다. 시작 오프셋이 함께 남아야 한다.
    dumped = timer.to_dict()["spans"]
    assert all("at_ms" in span for span in dumped)


def test_nested_steps_attach_to_the_job_that_started_them():
    """작업 스레드 안에서 남긴 구간이 그 작업의 자식으로 붙는다.

    `run_in_executor`는 컨텍스트를 옮겨주지 않는다. 다시 묶지 않으면 경로 안의
    임베딩·조회 시간이 부모 없이 떠돌아 어느 경로의 것인지 알 수 없다.
    """
    timer = timing.TimingRecorder()

    def work():
        with timing.step("text.embed_dense"):
            time.sleep(0.01)
        with timing.step("text.query"):
            time.sleep(0.01)
        return "done"

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(timer.job("path.text", work)).result() == "done"

    path = timer.find("path.text")
    children = [s for s in timer.spans if s.parent == path.id]
    assert sorted(s.name for s in children) == ["text.embed_dense", "text.query"]
    # 자식이 부모 안에 들어 있어야 한다.
    assert sum(s.run_ms for s in children) <= path.run_ms + 1.0


def test_unbound_step_records_nothing():
    """묶인 기록기가 없으면 조용히 지나간다.

    `SearchService`는 평가 스크립트에서도 불린다. 그 경로에는 기록기가 없다.
    """
    before = timing.current()
    with timing.step("text.query"):
        pass
    assert timing.current() is before is None


def test_failed_span_keeps_its_time_and_names_the_error():
    """실패한 단계의 시간도 응답 지연에는 그대로 들어간다."""
    timer = timing.TimingRecorder()
    with pytest.raises(ValueError):
        with timer.step("analysis.gemini", attempt=1):
            time.sleep(0.01)
            raise ValueError("boom")

    span = timer.find("analysis.gemini")
    assert span.error == "ValueError"
    assert span.run_ms >= 5
    assert span.attrs["attempt"] == 1


def test_emit_appends_one_json_line_per_request(tmp_path, monkeypatch):
    """측정 기록은 요청당 한 줄이다. 붙여 쓰므로 53건이 한 파일에 쌓인다."""
    path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("SEARCH_TIMING_LOG", str(path))

    for index in range(2):
        timer = timing.TimingRecorder(query=f"질의{index}")
        timer.note(top_k=10)
        with timer.step("search"):
            time.sleep(0.005)
        timing.emit(timer)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["query"] == "질의0"
    assert first["attrs"]["top_k"] == 10
    assert first["total_ms"] > 0
    assert [s["name"] for s in first["spans"]] == ["search"]


def test_emit_without_a_log_path_does_not_write(tmp_path, monkeypatch):
    """환경변수가 없으면 파일을 만들지 않는다 — 측정할 때만 쌓인다."""
    monkeypatch.delenv("SEARCH_TIMING_LOG", raising=False)
    timer = timing.TimingRecorder()
    timing.emit(timer)
    assert list(tmp_path.iterdir()) == []


def test_mark_closes_a_span_that_has_many_exits():
    """리랭킹처럼 빠져나가는 자리가 여럿인 구간을 위한 기록 방식."""
    timer = timing.TimingRecorder()
    with timer.step("search"):
        started = time.perf_counter()
        time.sleep(0.01)
        timer.mark("search.rerank", started, shape="plain")

    rerank = timer.find("search.rerank")
    assert rerank.run_ms >= 5
    assert rerank.parent == timer.find("search").id
    assert rerank.attrs["shape"] == "plain"


# ---------------------------------------------------------------------------
# 배선 — 미들웨어부터 라우트까지 실제 앱 경로로 지난다
# ---------------------------------------------------------------------------


class _FakeAnalyzer:
    """분석기 자리. 안에서 구간을 남겨 라우트가 묶어 주는지도 같이 본다."""

    def __init__(self, sleep: float = 0.02) -> None:
        self.sleep = sleep

    def analyze(self, query: str):
        from src.backend.schemas.query import QueryAnalysis

        with timing.step("analysis.gemini", attempt=1):
            time.sleep(self.sleep)
        return QueryAnalysis(
            original_query=query,
            intent_type="mood",
            image_english_query="",
            audio_english_query="",
        )


class _FakeSearchRouter:
    def __init__(self) -> None:
        self.timers = []

    async def search(self, analysis, **kwargs):
        from src.backend.schemas.search import MatchingTrack

        timer = kwargs["timer"]
        self.timers.append(timer)
        with timer.step("search.paths", submitted=2):
            time.sleep(0.01)
        tracks_out = kwargs.get("candidate_tracks_out")
        if tracks_out is not None:
            tracks_out.extend(
                MatchingTrack(id=f"s{i}", score=1.0, title=f"곡{i}") for i in range(20)
            )
        return [
            MatchingTrack(id=f"s{i}", score=1.0, title=f"곡{i}")
            for i in range(kwargs.get("top_k", 10))
        ]


@pytest.fixture()
def timed_app(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.backend.api.dependencies import get_query_analyzer, get_search_router
    from src.backend.api.routes.search import router as search_route
    from src.backend.main import record_request_timing

    monkeypatch.setenv("SEARCH_TIMING_LOG", str(tmp_path / "timing.jsonl"))

    searcher = _FakeSearchRouter()
    app = FastAPI()
    # 실서비스와 같은 경로여야 한다 — 미들웨어가 경로로 대상을 고른다.
    app.middleware("http")(record_request_timing)
    app.include_router(search_route, prefix="/api/v1")
    app.dependency_overrides[get_query_analyzer] = lambda: _FakeAnalyzer()
    app.dependency_overrides[get_search_router] = lambda: searcher

    with TestClient(app) as client:
        yield client, searcher, tmp_path / "timing.jsonl"


def test_request_records_the_whole_path_and_returns_its_id(timed_app):
    """요청 하나가 한 줄로 남고, 그 줄을 응답 헤더의 id로 찾을 수 있다.

    측정 스크립트는 클라이언트에서 잰 벽시계 시간과 서버 기록을 맞춰야 한다.
    id가 없으면 53건을 순서에 기대어 맞춰야 하는데, 재시도 한 번이면 어긋난다.
    """
    client, searcher, log = timed_app

    response = client.post("/api/v1/search", json={"query": "비 오는 날 발라드"})
    assert response.status_code == 200

    request_id = response.headers["X-Request-Id"]
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["request_id"] == request_id
    assert record["query"] == "비 오는 날 발라드"

    names = [span["name"] for span in record["spans"]]
    assert {"route", "analysis", "analysis.gemini", "search", "search.paths"} <= set(
        names
    )
    # 라우트가 받은 타이머가 그대로 검색까지 간다 — 여기서 끊기면 경로 시간이 없다.
    assert searcher.timers[0].request_id == request_id


def test_route_time_is_smaller_than_the_whole_request(timed_app):
    """미들웨어 시간 − 라우트 시간 = 프레임워크가 쓴 시간.

    의존성 해결·요청 검증·응답 직렬화가 그 차이에 들어간다. 라우트 안만 재면
    그만큼이 통째로 사라진다.
    """
    client, _, log = timed_app
    client.post("/api/v1/search", json={"query": "가사 검색", "explain": True})

    record = json.loads(log.read_text(encoding="utf-8").strip())
    route = next(s for s in record["spans"] if s["name"] == "route")
    assert route["run_ms"] < record["total_ms"]
    assert record["attrs"]["status"] == 200
    assert record["attrs"]["explain"] is True


def test_untimed_paths_leave_no_record(timed_app):
    """정적 파일·헬스체크까지 남기면 정작 볼 줄이 묻힌다."""
    client, _, log = timed_app
    client.get("/api/v1/does-not-exist")
    assert not log.exists()


def test_analysis_does_not_block_the_event_loop():
    """동시 요청 2건이 겹쳐서 돈다 — 질의 분석이 루프를 잡지 않는다.

    `analyze()`는 Gemini 왕복과 재시도 대기를 포함한 **동기** 함수다. async
    라우트에서 직접 부르면 그동안 프로세스가 다른 요청을 받지 못한다. 2026-09-23
    측정에서 실제로 뒤 요청이 앞 요청의 분석이 끝난 뒤에야 라우트에 들어왔다.

    여기서는 분석을 0.3초 자는 것으로 대신한다. 루프가 막히면 두 건이 줄을 서서
    0.6초를 넘고, 스레드로 나가면 0.3초대에서 함께 끝난다.
    """
    import asyncio

    import httpx
    from fastapi import FastAPI

    from src.backend.api.dependencies import get_query_analyzer, get_search_router
    from src.backend.api.routes.search import router as search_route

    app = FastAPI()
    app.include_router(search_route, prefix="/api/v1")
    app.dependency_overrides[get_query_analyzer] = lambda: _FakeAnalyzer(sleep=0.3)
    app.dependency_overrides[get_search_router] = lambda: _FakeSearchRouter()

    async def scenario() -> float:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            started = time.perf_counter()
            await asyncio.gather(
                c.post("/api/v1/search", json={"query": "첫 질의"}),
                c.post("/api/v1/search", json={"query": "둘째 질의"}),
            )
            return (time.perf_counter() - started) * 1000

    wall_ms = asyncio.run(scenario())
    # 두 번 자면 600ms. 겹치면 300ms대. 300과 600 사이에서 넉넉히 가른다.
    assert wall_ms < 450, f"분석이 루프를 잡고 있다: {wall_ms:.0f}ms"
