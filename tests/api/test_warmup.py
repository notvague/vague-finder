"""기동 예열 — 첫 사용자가 11초를 혼자 내지 않게 한다.

예열이 지켜야 할 것은 셋이다.

1. **꺼면 아무것도 안 올린다.** 모델 없이 화면만 손보는 자리, 그리고 테스트가 그 경로다.
2. **한 단계가 실패해도 나머지는 올린다.** 예열은 편의이지 전제가 아니다 —
   Mongo가 없다고 KoE5까지 안 올릴 이유가 없다.
3. **기동을 막지 않는다.** `--reload` 개발이 저장마다 11초씩 느려지면 안 된다.
"""

from __future__ import annotations

import time

from src.backend import warmup


def test_disabled_by_env_records_skipped(monkeypatch):
    monkeypatch.setenv("SEARCH_WARMUP", "0")
    assert warmup.enabled() is False
    assert warmup.run()["state"] == "skipped"


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("SEARCH_WARMUP", raising=False)
    assert warmup.enabled() is True


def test_one_failed_stage_does_not_stop_the_rest_but_is_not_ready(monkeypatch):
    """가사 스냅샷을 못 읽어도 모델은 올라간다. 다만 **ready라고 하지 않는다.**

    발표 전에 `/health`의 이 값만 보고 준비됐다고 판단하므로, 실패를 삼키고
    ready로 끝내면 예열이 반만 된 서버를 준비된 서버로 오인한다.
    """
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    done = []

    def ok():
        done.append("ok")

    def boom():
        raise RuntimeError("Mongo 없음")

    monkeypatch.setattr(
        warmup, "_stages", lambda: [("koe5", ok), ("lyrics", boom), ("clap", ok)]
    )
    status = warmup.run()

    assert status["state"] == warmup.PARTIAL_FAILURE
    assert done == ["ok", "ok"]
    assert status["stages"]["koe5"]["outcome"] == "ok"
    assert status["stages"]["lyrics"]["outcome"].startswith("failed:")
    assert "Mongo 없음" in status["stages"]["lyrics"]["outcome"]
    assert status["stages"]["clap"]["outcome"] == "ok"


def test_intentional_skip_is_not_a_failure(monkeypatch):
    """꺼 둔 리랭커는 실패가 아니다.

    실패로 세면 진짜 실패(모델 로딩 오류)와 구분이 안 되고, 설정대로 동작하는
    서버가 늘 partial_failure로 보인다.
    """
    monkeypatch.setenv("SEARCH_WARMUP", "1")

    def skip():
        raise warmup.SkipStage("리랭커가 꺼져 있다(설정)")

    monkeypatch.setattr(warmup, "_stages", lambda: [("reranker", skip)])
    status = warmup.run()

    assert status["state"] == warmup.READY
    assert status["stages"]["reranker"]["outcome"].startswith("skipped:")


def test_remote_reranker_is_skipped_not_called(monkeypatch):
    """RERANKER_BACKEND=gemini_listwise면 예열이 Gemini를 부르지 않는다.

    원격 호출은 미리 불러도 다음이 빨라지지 않는다. 서버를 띄울 때마다 쿼터만 쓴다.
    """
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    called = []

    class _FakeGeminiReranker:
        enabled = True

        def load(self):
            called.append("load")

        def rerank(self, *args, **kwargs):
            called.append("rerank")
            return []

    from src.backend.api import dependencies

    monkeypatch.setattr(dependencies, "get_reranker", lambda: _FakeGeminiReranker())
    stage = dict(warmup._stages())["reranker"]
    status = warmup._run_stage("reranker", stage)

    assert status.startswith("skipped:")
    assert called == []


def test_shutdown_stops_the_remaining_stages(monkeypatch):
    """종료 신호가 걸리면 남은 단계로 넘어가지 않는다.

    마지막 `vector_db` 단계가 종료 처리 뒤에 돌면 방금 닫은 저장소를 다시 연다.
    로컬 Qdrant는 폴더를 하나만 열 수 있어서 그 핸들이 다음 기동을 막는다.
    """
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    ran = []

    def slow():
        ran.append("slow")
        time.sleep(0.2)

    def must_not_run():
        ran.append("vector_db")

    monkeypatch.setattr(
        warmup, "_stages", lambda: [("slow", slow), ("vector_db", must_not_run)]
    )

    warmup.start_background()
    time.sleep(0.05)  # 첫 단계가 돌고 있는 동안에 내린다
    warmup.stop_and_wait(timeout=5)

    assert ran == ["slow"]
    assert warmup.status()["state"] == warmup.STOPPED


def test_lifespan_stops_warmup_before_closing_the_client(monkeypatch):
    """종료 순서: 예열을 세운 **뒤에** 벡터 클라이언트를 닫는다."""
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main

    order = []

    monkeypatch.setenv("SEARCH_WARMUP", "1")
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())
    monkeypatch.setattr(
        backend_main, "close_vector_client", lambda: order.append("close")
    )

    def stage():
        order.append("stage")
        time.sleep(0.3)

    monkeypatch.setattr(warmup, "_stages", lambda: [("slow", stage), ("slow2", stage)])

    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
        time.sleep(0.05)  # 첫 단계가 돌고 있는 중에 내린다

    # 두 번째 단계가 늦게라도 도는지 본다. 종료 뒤에 도는 단계가 곧 버그다.
    time.sleep(0.6)
    assert order == ["stage", "close"], order


def test_background_start_returns_before_the_work_finishes(monkeypatch):
    """예열은 기동을 막지 않는다."""
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    monkeypatch.setattr(
        warmup, "_stages", lambda: [("slow", lambda: time.sleep(0.3))]
    )

    started = time.perf_counter()
    warmup.start_background()
    returned_ms = (time.perf_counter() - started) * 1000

    assert returned_ms < 100
    assert warmup.status()["state"] in {"running", "ready"}
    # 데몬 스레드라 여기서 기다리지 않아도 되지만, 상태가 ready로 끝나는지는 본다.
    deadline = time.perf_counter() + 3
    while warmup.status()["state"] == "running" and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert warmup.status()["state"] == "ready"


def test_health_reports_the_warmup_state(monkeypatch):
    """발표 전에 "빠를 준비가 됐는지"를 한 번에 본다."""
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main

    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    monkeypatch.setenv("SEARCH_WARMUP", "0")

    with TestClient(backend_main.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["warmup"]["state"] == "skipped"


def test_shutdown_waits_for_a_stage_that_outlives_any_deadline(monkeypatch):
    """돌고 있는 단계가 길어도 **끝난 뒤에** 클라이언트를 닫는다.

    앞서 종료 대기에 30초 제한을 두었는데, 제한이 지나면 스레드가 살아 있는 채로
    클라이언트를 닫았다. 그러면 이미 시작한 `vector_db` 단계가 닫힌 저장소에 질의를
    보낸다 — 제한을 둔 채로는 고치려던 문제가 그대로 돌아온다.

    여기서는 DB를 만지는 단계가 대기보다 확실히 오래 걸리게 두고, 닫기가 그 뒤에
    오는지만 본다.
    """
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main

    order = []
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())
    monkeypatch.setattr(
        backend_main, "close_vector_client", lambda: order.append("close")
    )

    def db_stage():
        order.append("db-시작")
        time.sleep(0.8)
        # 닫힌 뒤에 여기 오면 안 된다. 실서비스에서는 이 줄이 Qdrant 질의다.
        order.append("db-질의")

    monkeypatch.setattr(warmup, "_stages", lambda: [("vector_db", db_stage)])

    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
        time.sleep(0.05)

    assert order == ["db-시작", "db-질의", "close"], order


def test_timeout_reports_failure_and_keeps_the_handle(monkeypatch):
    """제한을 두고 부른 쪽은 **실패를 돌려받고**, 핸들은 남는다.

    핸들을 버리면 다음 기동이 살아 있는 스레드를 모르는 채로 시작한다.
    False를 받은 쪽은 클라이언트를 닫으면 안 된다.
    """
    monkeypatch.setenv("SEARCH_WARMUP", "1")
    monkeypatch.setattr(warmup, "_stages", lambda: [("slow", lambda: time.sleep(0.6))])

    warmup.start_background()
    time.sleep(0.05)
    assert warmup.stop_and_wait(timeout=0.05) is False
    assert warmup._thread is not None and warmup._thread.is_alive()

    # 아직 도는 스레드가 있으면 새 예열을 시작하지 않는다 — 시작하면서 지우는
    # 종료 신호를 그 스레드도 보고 있기 때문이다.
    warmup.start_background()
    assert warmup.stop_and_wait() is True


def test_shutdown_joins_the_warmup_thread_without_a_deadline(monkeypatch):
    """종료 경로가 `join()`에 **제한을 주지 않는지**를 직접 고정한다.

    "오래 걸리는 단계를 기다린다"만 보는 시험은 제한이 넉넉하기만 하면 통과한다
    (30초 제한을 되살려도 0.8초짜리 단계는 통과한다). 이번 결함은 제한이 있다는
    것 자체였으므로, 넘어가는 값이 None인지를 본다.
    """
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main

    monkeypatch.setenv("SEARCH_WARMUP", "1")
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    monkeypatch.setattr(warmup, "_stages", lambda: [("slow", lambda: time.sleep(0.3))])

    joins = []

    with TestClient(backend_main.app):
        thread = warmup._thread
        assert thread is not None, "lifespan이 예열을 시작하지 않았다"
        original_join = thread.join

        def spy(timeout=None):
            joins.append(timeout)
            return original_join(timeout)

        thread.join = spy  # 이 스레드에만 건다 — 전역 Thread를 건드리지 않는다

    assert joins == [None], f"종료가 제한을 걸고 기다린다: {joins}"
