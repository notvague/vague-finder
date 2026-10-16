"""v58 — 가사 판(Mongo full_lyrics 캐시) TTL 만료 시 **기존 판으로 검색하며 뒤에서 한 번만 갱신**.

v56 운영 검증에서 TTL 300초가 지난 뒤 첫 가사 요청이 전체 재적재(약 350초)를 `_load_lock` 안에서
기다렸다. 여기서 고정하는 것:

- 최초 적재(판 없음)는 예전처럼 블로킹이고 실패는 예외로 올라간다(조용히 빈 결과 아님)
- 판이 있으면 TTL이 지나도 요청은 기다리지 않는다 — 갱신이 이벤트로 막혀 있어도 돌아온다
- 갱신은 서비스 하나에 동시에 하나, 실패하면 지금 판 유지 + 재시도 간격, 부분 판은 게시하지 않는다
- 검색이 쓴 판(`snapshot_out`)으로만 인용한다 — 교체 뒤에도 같은 판
- 명시적 `refresh()`는 무효화다 — 진행 중 갱신 결과가 되살아나지 않는다
- 종료 뒤에는 새 갱신을 시작하지 않고 결과를 게시하지 않는다

실제 Mongo·모델·350초 대기 없음 — 자료원은 주입한 가짜, 시계는 가짜, 갱신은 `threading.Event`로 막는다.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytest

from src.backend.schemas.query import LyricClue
from src.retrieval.lyrics_exact_search import (
    DEFAULT_REFRESH_RETRY_SECONDS,
    LyricsExactSearchService,
    LyricsServiceClosed,
    LyricsSnapshot,
)

# 가상의 가사(실제 곡 아님). 기존 test_lyric_evidence와 같은 문장을 쓴다.
LYRICS_V1 = "모래처럼 흩어진 오후의 약속은\n창문을 여는 척 웃었고!\nI FOUND  the WAY, oh\n"
LYRICS_V2 = "첫 줄이 통째로 바뀌었고 훨씬 길어졌습니다\n창문을 여는 척 웃었고!\n"
OTHER = "전혀 다른 노래의 가사다\n두 번째 줄\n"

CLUE = LyricClue(text="창문을 여는 척", kind="verbatim", confidence=1.0)
# 마지막 글자만 다른 가사와 fuzzy로 맞는 단서(3-gram 겹침 10/11 · 연속 12/13 → 0.91 ≥ 0.78)
FUZZY_CLUE = LyricClue(text="모래처럼 흩어진 오후의 약속은", kind="partial", confidence=0.9)
WAIT = 3.0  # 스레드 동기화 대기 상한. 통과하는 실행은 밀리초 단위다.


def _doc(song_id: str, lyrics: str, title: str = "가상 곡") -> Dict[str, Any]:
    return {"song_id": song_id, "metadata": {"title": title, "artist": "가수"},
            "lyrics_data": {"full_lyrics": lyrics}}


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSource:
    """주입 가능한 자료원. 호출마다 `docs`를 낸다.

    gate: 설정돼 있으면 `gate_after`번째 문서를 낸 뒤 이벤트가 열리기까지 **읽기 안에서** 막힌다
          (Mongo cursor의 긴 getMore를 흉내). `started`는 그 지점에 닿았음을 알린다.
    fail_after: 그만큼 낸 뒤 예외(자료원 오류). 0이면 첫 문서 전에.
    """

    def __init__(self, docs: List[Mapping[str, Any]]) -> None:
        self.docs = list(docs)
        self.calls = 0
        self.closed = 0
        self.gate: Optional[threading.Event] = None
        self.gate_after = 1
        self.started = threading.Event()
        self.fail_after: Optional[int] = None
        self.fail_once = False
        self._lock = threading.Lock()

    def __call__(self) -> Iterable[Mapping[str, Any]]:
        with self._lock:
            self.calls += 1
            fail_after = self.fail_after
            if self.fail_once:
                self.fail_after = None
        return self._iter(list(self.docs), fail_after)

    def _iter(self, docs: List[Mapping[str, Any]], fail_after: Optional[int]):
        source = self

        class _Cursor:
            def __init__(self) -> None:
                self._index = 0

            def __iter__(self):
                return self

            def __next__(self):
                # gate를 먼저 본다 — gate가 열린 직후 같은 왕복에서 실패하는 모양(늦은 실패)을 만들 수 있다.
                if source.gate is not None and self._index == source.gate_after:
                    source.started.set()
                    if not source.gate.wait(WAIT * 4):
                        raise AssertionError("테스트가 gate를 열지 않았다")
                if fail_after is not None and self._index == fail_after:
                    raise ConnectionError("자료원 읽기 실패(가짜)")
                if self._index >= len(docs):
                    raise StopIteration
                item = docs[self._index]
                self._index += 1
                return item

            def close(self) -> None:
                source.closed += 1

        return _Cursor()


def _service(source: FakeSource, clock: FakeClock, ttl: float = 300.0, retry: float = 60.0,
             **kw: Any) -> LyricsExactSearchService:
    return LyricsExactSearchService(cache_ttl_seconds=ttl, documents_source=source, clock=clock,
                                    refresh_retry_seconds=retry, **kw)


def _ids(tracks) -> List[str]:
    return [t.id for t in tracks]


def _join_refresh(service: LyricsExactSearchService, timeout: float = WAIT) -> None:
    thread = service._refresh_thread
    if thread is not None:
        thread.join(timeout)
        assert not thread.is_alive(), "갱신 스레드가 끝나지 않았다"


def _timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


# ---------------------------------------------------------------------------
# 최초 적재
# ---------------------------------------------------------------------------

def test_initial_load_blocks_reads_once_and_serves_results():
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    service = _service(source, FakeClock())
    assert service.status()["state"] == "empty"

    used: List[LyricsSnapshot] = []
    tracks = service.search([CLUE], snapshot_out=used)

    assert _ids(tracks) == ["s1"] and tracks[0].lyric_match_type == "exact"
    assert source.calls == 1 and source.closed == 1, "자료원 cursor는 읽은 뒤 닫는다"
    assert used[0].documents and used[0] is service.snapshot()
    status = service.status()
    assert status["state"] == "fresh" and status["documents"] == 2 and status["refreshing"] is False
    assert status["last_success_epoch"] is not None


def test_initial_load_failure_raises_and_next_search_retries():
    """판이 없을 때의 실패는 예외다 — 가사 경로를 빈 결과로 조용히 생략하지 않는다(라우터가 경로 실패로 기록)."""
    source = FakeSource([_doc("s1", LYRICS_V1)])
    source.fail_after, source.fail_once = 0, True
    service = _service(source, FakeClock())

    with pytest.raises(ConnectionError):
        service.search([CLUE])
    assert service.status()["state"] == "empty" and source.closed == 1

    tracks = service.search([CLUE])  # 다음 검색은 다시 읽는다(최초 적재 계약 그대로)
    assert _ids(tracks) == ["s1"] and source.calls == 2


def test_concurrent_initial_loads_read_the_source_once():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    source.gate, source.gate_after = threading.Event(), 1
    service = _service(source, FakeClock())
    results: List[List[str]] = []

    def worker() -> None:
        results.append(_ids(service.search([CLUE])))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    assert source.started.wait(WAIT)
    time.sleep(0.05)
    source.gate.set()
    for t in threads:
        t.join(WAIT)
    assert results == [["s1"]] * 4 and source.calls == 1


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

def test_no_reload_before_ttl_expires():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock, ttl=300.0)
    first = service.snapshot()
    for _ in range(5):
        clock.advance(59.0)
        service.search([CLUE])
    assert source.calls == 1 and service.snapshot() is first
    assert service.status()["state"] == "fresh" and service.status()["refreshing"] is False


def test_expired_snapshot_is_served_immediately_while_refresh_is_blocked():
    """핵심 — 갱신이 **막혀 있어도** 요청은 지금 판으로 바로 돌아온다(v56에서는 350초를 기다렸다)."""
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    clock = FakeClock()
    service = _service(source, clock, ttl=300.0)
    before = service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2), _doc("s2", OTHER)]
    source.gate, source.gate_after = threading.Event(), 1   # 갱신은 첫 문서 뒤에 막힌다
    clock.advance(301.0)

    used: List[LyricsSnapshot] = []
    (tracks, elapsed) = _timed(lambda: service.search([CLUE], snapshot_out=used))

    assert _ids(tracks) == ["s1"] and used[0] is before, "지금 판으로 검색했다"
    assert elapsed < 1.0, f"만료 뒤 요청이 갱신을 기다렸다({elapsed:.2f}s)"
    assert source.started.wait(WAIT) and source.calls == 2
    status = service.status()
    assert status["state"] == "stale" and status["refreshing"] is True
    assert status["refresh_running_seconds"] is not None

    # 막힌 동안 더 들어온 요청도 바로, 같은 판으로. 갱신은 여전히 하나.
    (_, elapsed2) = _timed(lambda: service.search([CLUE]))
    assert elapsed2 < 1.0 and service.snapshot() is before and source.calls == 2
    assert [t.name for t in threading.enumerate()].count("lyrics_refresh") == 1

    source.gate.set()
    _join_refresh(service)
    after = service.snapshot()
    assert after is not before and after.get("s1").lyrics_sha16 != before.get("s1").lyrics_sha16
    status = service.status()
    assert status["state"] == "fresh" and status["refreshing"] is False and status["refresh_successes"] == 1
    assert source.closed == 2


def test_many_concurrent_requests_after_expiry_start_exactly_one_refresh():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    before = service.snapshot()
    source.gate, source.gate_after = threading.Event(), 0
    clock.advance(600.0)

    seen: List[LyricsSnapshot] = []
    lock = threading.Lock()

    def worker() -> None:
        out: List[LyricsSnapshot] = []
        service.search([CLUE], snapshot_out=out)
        with lock:
            seen.append(out[0])

    threads = [threading.Thread(target=worker) for _ in range(12)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert all(not t.is_alive() for t in threads) and time.perf_counter() - started < 2.0
    assert len(seen) == 12 and all(s is before for s in seen)
    assert source.calls == 2, "갱신 읽기는 한 번"
    source.gate.set()
    _join_refresh(service)
    assert source.calls == 2


def test_requests_after_successful_refresh_use_the_new_snapshot():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    old = service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2), _doc("s3", "새로 들어온 곡의 가사\n창문을 여는 척\n")]
    clock.advance(301.0)
    service.search([CLUE])
    _join_refresh(service)

    used: List[LyricsSnapshot] = []
    tracks = service.search([CLUE], snapshot_out=used)
    assert used[0] is not old and used[0].loaded_at == clock.now
    assert set(_ids(tracks)) == {"s1", "s3"}
    assert service.excerpt("s1", "창문을여는척", used[0]).data_version == used[0].get("s1").lyrics_sha16


# ---------------------------------------------------------------------------
# 갱신 실패
# ---------------------------------------------------------------------------

def test_source_error_during_refresh_keeps_the_old_snapshot_and_publishes_no_partial():
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    clock = FakeClock()
    service = _service(source, clock)
    before = service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2), _doc("s2", OTHER)]
    source.fail_after = 1  # 첫 문서를 낸 뒤 실패 — 부분 판이 생길 수 있는 자리
    clock.advance(301.0)

    tracks = service.search([CLUE])
    _join_refresh(service)

    assert _ids(tracks) == ["s1"]
    assert service.snapshot() is before, "부분 판이 게시됐다"
    assert len(service.snapshot().documents) == 2
    status = service.status()
    assert status["refresh_failures"] == 1 and status["refreshing"] is False
    assert status["last_refresh_error"].startswith("ConnectionError")
    assert status["retry_in_seconds"] == pytest.approx(60.0, abs=0.2)
    assert source.closed == 2, "실패해도 cursor는 닫는다"


def test_failed_refresh_is_not_retried_before_the_interval_and_recovers_after():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock, ttl=300.0, retry=60.0)
    before = service.snapshot()
    source.fail_after = 0
    clock.advance(301.0)
    service.search([CLUE])
    _join_refresh(service)
    assert source.calls == 2 and service.status()["refresh_failures"] == 1

    # 간격 안: 요청이 와도 다시 읽지 않는다(폭주 방지). 지금 판으로 답한다.
    for _ in range(5):
        clock.advance(10.0)
        assert _ids(service.search([CLUE])) == ["s1"]
    assert source.calls == 2 and service.snapshot() is before
    assert service.status()["retry_in_seconds"] == pytest.approx(10.0, abs=0.2)

    # 간격 뒤: 다시 시도 → 자료원이 살아나면 새 판
    source.fail_after = None
    source.docs = [_doc("s1", LYRICS_V2)]
    clock.advance(10.0)
    service.search([CLUE])
    _join_refresh(service)
    assert source.calls == 3
    status = service.status()
    assert service.snapshot() is not before and status["state"] == "fresh"
    assert status["refresh_failures"] == 1 and status["refresh_successes"] == 1
    assert status["last_refresh_error"] is None and status["retry_in_seconds"] is None


def test_late_failure_of_an_invalidated_refresh_does_not_block_the_new_generation():
    """옛 갱신 시작 → refresh() → 새 판 적재 → 옛 갱신이 늦게 실패. 옛 실패가 새 판의 재시도 대기를 걸면 안 된다."""
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER), _doc("s3", OTHER)])
    clock = FakeClock()
    service = _service(source, clock, ttl=300.0, retry=60.0)
    service.snapshot()
    source.gate, source.gate_after = threading.Event(), 1
    source.fail_after, source.fail_once = 1, True    # gate가 열린 직후 같은 왕복에서 실패(늦은 실패)
    clock.advance(301.0)
    service.search([CLUE])                             # 옛 세대 갱신 시작(막힘)
    assert source.started.wait(WAIT)

    service.refresh()                                  # 무효화 → 다음 검색이 새 판을 블로킹 적재
    source.gate_after = 10**6                          # 새 읽기는 gate에 걸리지 않게
    service.search([CLUE])
    assert service.status()["state"] == "fresh" and source.calls == 3
    source.gate.set()                                  # 옛 갱신을 풀어 늦게 실패하게 한다
    _join_refresh(service)

    status = service.status()
    assert status["refresh_failures"] == 0 and status["last_refresh_error"] is None
    assert status["retry_in_seconds"] is None and status["refresh_discarded"] == 1

    clock.advance(301.0)                               # 새 판 만료 → 갱신이 막히지 않고 시작해야 한다
    service.search([CLUE])
    _join_refresh(service)
    assert source.calls == 4 and service.status()["refresh_successes"] == 1


def test_status_and_logs_carry_only_the_exception_type(caplog):
    """오류 메시지에 호스트·계정이 섞일 수 있다 — /health와 로그에는 형식 이름만 남긴다."""
    import json
    import logging

    secret = "mongodb://user:pw@cluster.example.net:27017 timed out"

    class _Source(FakeSource):
        def __call__(self):
            self.calls += 1
            if self.calls > 1:
                raise ConnectionError(secret) from TimeoutError("socket")
            return super().__call__()

    source = _Source([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    service.snapshot()
    clock.advance(301.0)
    with caplog.at_level(logging.INFO, logger="src.retrieval.lyrics_exact_search"):
        service.search([CLUE])
        _join_refresh(service)
    status = service.status()
    assert status["last_refresh_error"] == "ConnectionError <- TimeoutError"
    dumped = json.dumps(status, default=str)
    assert "pw@" not in dumped and "cluster.example.net" not in dumped
    assert "cluster.example.net" not in caplog.text and "pw@" not in caplog.text


def test_mongo_path_uses_a_dedicated_client_with_socket_timeout_and_batch_size(monkeypatch):
    from src.retrieval import lyrics_exact_search as mod

    made: List[Dict[str, Any]] = []

    class _Collection:
        def __init__(self) -> None:
            self.find_calls: List[Dict[str, Any]] = []

        def find(self, filter, projection, **kw):
            self.find_calls.append({"filter": filter, "projection": projection, **kw})
            return iter([_doc("s1", LYRICS_V1)])

    class _Client:
        def __init__(self) -> None:
            self.collection = _Collection()
            self.closed = 0

        def __getitem__(self, name):
            return {mod.COLLECTION: self.collection}

        def close(self) -> None:
            self.closed += 1

    client = _Client()

    def fake_factory(socket_timeout_seconds: float):
        made.append({"socket_timeout_seconds": socket_timeout_seconds})
        return client

    monkeypatch.setattr(mod, "_make_lyrics_mongo_client", fake_factory)
    service = LyricsExactSearchService(cache_ttl_seconds=300.0, mongo_socket_timeout_seconds=45.0,
                                       mongo_batch_size=150, close_wait_margin_seconds=5.0)
    assert _ids(service.search([CLUE])) == ["s1"]
    assert made == [{"socket_timeout_seconds": 45.0}]
    assert client.collection.find_calls[0]["batch_size"] == 150
    assert "lyrics_data.full_lyrics" in client.collection.find_calls[0]["projection"]
    assert service.close_wait_seconds == 50.0 and service.status()["close_wait_seconds"] == 50.0
    assert service.close() is True and client.closed == 1
    assert len(made) == 1, "클라이언트는 한 번만 만든다"


def test_close_without_timeout_waits_only_socket_timeout_plus_margin():
    """제한 없는(주입) 자료원이 멈춰 있으면 기본 대기 뒤 False — 무기한 대기하지 않는다."""
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    clock = FakeClock()
    service = _service(source, clock, mongo_socket_timeout_seconds=1.0, close_wait_margin_seconds=0.2)
    service.snapshot()
    source.gate, source.gate_after = threading.Event(), 1
    clock.advance(301.0)
    service.search([CLUE])
    assert source.started.wait(WAIT)
    (finished, elapsed) = _timed(service.close)
    assert finished is False and 1.0 <= elapsed < 2.5
    source.gate.set()
    _join_refresh(service)


def test_retry_interval_is_configurable():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock, ttl=10.0, retry=5.0)
    service.snapshot()
    source.fail_after = 0
    clock.advance(11.0)
    service.search([CLUE])
    _join_refresh(service)
    clock.advance(4.0)
    service.search([CLUE])
    assert source.calls == 2
    clock.advance(1.5)
    service.search([CLUE])
    _join_refresh(service)
    assert source.calls == 3
    assert DEFAULT_REFRESH_RETRY_SECONDS == 60.0


# ---------------------------------------------------------------------------
# 검색과 인용의 판 일관성
# ---------------------------------------------------------------------------

def test_request_that_searched_the_old_snapshot_quotes_from_it_after_replacement():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2)]
    source.gate, source.gate_after = threading.Event(), 0
    clock.advance(301.0)

    used: List[LyricsSnapshot] = []
    tracks = service.search([CLUE], snapshot_out=used)
    old_document = used[0].get("s1")
    assert source.started.wait(WAIT)
    source.gate.set()
    _join_refresh(service)                       # 그 사이 판이 교체됐다
    assert service.snapshot() is not used[0]

    found = service.excerpt(tracks[0].id, tracks[0].lyric_match_detail.matched_phrase, used[0])
    assert found is not None
    assert found.data_version == old_document.lyrics_sha16, "다른 판을 인용했다"
    assert found.char_start == LYRICS_V1.index("창문을")
    assert LYRICS_V1[found.char_start:found.char_end] == found.quote == "창문을 여는 척"

    new_used: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=new_used)
    new_found = service.excerpt("s1", "창문을여는척", new_used[0])
    assert new_found.data_version != found.data_version
    assert new_found.char_start == LYRICS_V2.index("창문을")


def test_identical_data_gives_identical_results_ranking_and_hashes_across_refresh():
    docs = [_doc("a", "사랑 이야기\n창문을 여는 척 웃었고"), _doc("b", "모래처럼 흩어진 오후의 약속을\n다른 줄"),
            _doc("c", "전혀 무관한 가사"), _doc("d", "창문을 여는 척\n또 한 줄")]
    source = FakeSource(docs)
    clock = FakeClock()
    service = _service(source, clock)

    def observe():
        used: List[LyricsSnapshot] = []
        tracks = service.search([CLUE, FUZZY_CLUE], snapshot_out=used)
        return (
            [(t.id, t.lyric_match_type, round(float(t.lyric_match_score), 6),
              t.lyric_match_detail.matched_phrase, t.lyric_match_detail.corpus_match_count) for t in tracks],
            {t.id: service.excerpt(t.id, t.lyric_match_detail.matched_phrase, used[0]) for t in tracks},
            used[0],
        )

    first, quotes_first, snap_first = observe()
    clock.advance(301.0)
    service.search([CLUE])
    _join_refresh(service)
    second, quotes_second, snap_second = observe()

    assert snap_second is not snap_first and source.calls == 2
    assert first == second and {"exact", "fuzzy"} <= {row[1] for row in first}
    assert quotes_first == quotes_second
    assert {d.track.id: d.lyrics_sha16 for d in snap_first.documents} == \
           {d.track.id: d.lyrics_sha16 for d in snap_second.documents}


# ---------------------------------------------------------------------------
# 명시적 refresh()
# ---------------------------------------------------------------------------

def test_explicit_refresh_drops_the_snapshot_and_next_search_reloads_synchronously():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    used: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=used)
    source.docs = [_doc("s1", LYRICS_V2)]

    service.refresh()
    assert service.status()["state"] == "empty"
    assert service.excerpt("s1", "창문을여는척", used[0]).quote == "창문을 여는 척", "나간 판은 그대로 산다"

    new_used: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=new_used)
    assert source.calls == 2 and new_used[0].generation == 1
    assert new_used[0].get("s1").lyrics_sha16 != used[0].get("s1").lyrics_sha16


def test_explicit_refresh_during_background_refresh_discards_that_result():
    """무효화 뒤에, 무효화 **전에** 시작한 읽기의 결과가 게시되면 안 된다."""
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2)]
    source.gate, source.gate_after = threading.Event(), 1   # 마지막 문서까지 읽은 뒤 막힘 — 끝까지 읽은 결과가 나온다
    clock.advance(301.0)
    service.search([CLUE])
    assert source.started.wait(WAIT)

    (_, elapsed) = _timed(service.refresh)
    assert elapsed < 0.5, "refresh()가 진행 중 갱신을 기다렸다"
    source.docs = [_doc("s1", "무효화 뒤에 바뀐 가사\n창문을 여는 척\n")]
    source.gate.set()
    _join_refresh(service)

    status = service.status()
    assert status["state"] == "empty" and status["refresh_discarded"] + status["refresh_cancelled"] == 1
    assert status["refresh_successes"] == 0

    used: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=used)
    assert source.calls == 3 and "무효화 뒤에" in used[0].get("s1").full_lyrics


def test_explicit_refresh_during_initial_load_is_not_overwritten_by_that_load():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    source.gate, source.gate_after = threading.Event(), 1
    clock = FakeClock()
    service = _service(source, clock)
    got: List[LyricsSnapshot] = []
    thread = threading.Thread(target=lambda: service.search([CLUE], snapshot_out=got))
    thread.start()
    assert source.started.wait(WAIT)
    service.refresh()
    source.docs = [_doc("s1", LYRICS_V2)]
    source.gate.set()
    thread.join(WAIT)

    assert got and got[0].get("s1").full_lyrics == LYRICS_V1, "그 요청은 자기가 읽은 판을 쓴다"
    assert service.status()["state"] == "empty", "무효화 이전 읽기가 게시됐다"
    assert service.snapshot().get("s1").full_lyrics == LYRICS_V2 and source.calls == 2


# ---------------------------------------------------------------------------
# 종료
# ---------------------------------------------------------------------------

def test_close_stops_a_running_refresh_and_blocks_new_ones():
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER), _doc("s3", OTHER)])
    clock = FakeClock()
    service = _service(source, clock)
    before = service.snapshot()
    source.gate, source.gate_after = threading.Event(), 1
    clock.advance(301.0)
    service.search([CLUE])
    assert source.started.wait(WAIT)

    assert service.close(timeout=0.1) is False, "읽기가 아직 돈다 — 취소 완료가 아니다"
    assert service.closed and service.status()["closed"] is True
    source.gate.set()                      # 자료원이 풀리면 다음 문서에서 취소된다
    assert service.close(timeout=WAIT) is True
    status = service.status()
    assert status["refresh_cancelled"] == 1 and status["refresh_successes"] == 0
    assert source.closed == 2, "취소돼도 cursor는 닫는다"

    # 닫힌 뒤: 마지막 판으로는 답하지만 새 갱신은 없다
    clock.advance(1000.0)
    used: List[LyricsSnapshot] = []
    assert _ids(service.search([CLUE], snapshot_out=used)) == ["s1"] and used[0] is before
    assert source.calls == 2 and service.status()["refreshing"] is False


def test_refresh_that_finishes_after_close_does_not_publish():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    before = service.snapshot()
    source.docs = [_doc("s1", LYRICS_V2)]
    source.gate, source.gate_after = threading.Event(), 1   # 끝까지 읽고 나서 막힘
    clock.advance(301.0)
    service.search([CLUE])
    assert source.started.wait(WAIT)
    assert service.close(timeout=0.05) is False
    source.gate.set()
    _join_refresh(service)
    assert service.snapshot() is before and service.status()["refresh_discarded"] == 1


def test_closed_service_without_a_snapshot_refuses_to_read():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    service = _service(source, FakeClock())
    assert service.close() is True
    with pytest.raises(LyricsServiceClosed):
        service.search([CLUE])
    assert source.calls == 0


def test_close_is_idempotent_and_cheap_without_refresh():
    service = _service(FakeSource([_doc("s1", LYRICS_V1)]), FakeClock())
    service.snapshot()
    assert service.close() is True and service.close() is True


# ---------------------------------------------------------------------------
# 경계값
# ---------------------------------------------------------------------------

def test_empty_source_builds_an_empty_snapshot_and_searches_return_nothing():
    source = FakeSource([])
    clock = FakeClock()
    service = _service(source, clock)
    assert service.search([CLUE]) == [] and service.status()["documents"] == 0
    clock.advance(301.0)
    assert service.search([CLUE]) == []      # 빈 판도 판이다 — 만료되면 뒤에서 갱신
    _join_refresh(service)
    assert source.calls == 2 and service.status()["state"] == "fresh"


def test_ttl_zero_means_no_cache_and_reloads_synchronously_each_time():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    service = _service(source, FakeClock(), ttl=0.0)
    first: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=first)
    source.docs = [_doc("s1", LYRICS_V2)]
    second: List[LyricsSnapshot] = []
    service.search([CLUE], snapshot_out=second)
    assert source.calls == 2 and second[0].get("s1").full_lyrics == LYRICS_V2
    assert service.status()["state"] == "no_cache" and service.status()["refreshing"] is False
    assert [t.name for t in threading.enumerate()].count("lyrics_refresh") == 0


def test_seed_documents_constructor_still_works():
    service = LyricsExactSearchService(documents=[_doc("s1", LYRICS_V1)])
    assert _ids(service.search([CLUE])) == ["s1"]


def test_no_thread_is_created_per_request():
    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    service.snapshot()
    source.gate, source.gate_after = threading.Event(), 0
    clock.advance(301.0)
    baseline = threading.active_count()
    service.search([CLUE])
    assert source.started.wait(WAIT)
    for _ in range(20):
        service.search([CLUE])
    assert threading.active_count() <= baseline + 1
    source.gate.set()
    _join_refresh(service)


# ---------------------------------------------------------------------------
# HTTP 경로(TestClient) — 실제 라우트 + 실제 SearchRouter + 실제 가사 서비스, 모델·Qdrant는 가짜
# ---------------------------------------------------------------------------

class _TextService:
    def search_text(self, query, top_k=10, alpha=0.5, sparse_query=None, metadata_filter=None):
        return []


class _Embedder:
    def embed_texts(self, texts, l2_normalize=True):
        return [[0.0] * 4 for _ in texts]


class _Index:
    def query(self, **kwargs):
        return {"matches": []}

    def fetch(self, ids, namespace=None, fields=None):
        return {"matches": []}


class _Client:
    def Index(self, name):  # noqa: N802
        return _Index()


def _http_app(service: LyricsExactSearchService):
    from fastapi import FastAPI

    from src.backend.api.dependencies import (
        get_lyrics_exact_search_service,
        get_query_analyzer,
        get_search_router,
    )
    from src.backend.api.routes.search import router as search_route
    from src.backend.schemas.query import QueryAnalysis
    from src.retrieval import search_router as sr

    router = sr.SearchRouter(
        search_service=_TextService(), image_embedder=_Embedder(), audio_embedder=_Embedder(),
        vector_client=_Client(), lyrics_search_service=service, reranker=None, max_workers=2,
    )

    class _Analyzer:
        analysis_budget_seconds = 20.0

        def analyze(self, query: str) -> QueryAnalysis:
            return QueryAnalysis(
                original_query=query, intent_type="lyrics", korean_tags=["가사"],
                lyric_clues=[CLUE], image_english_query="", audio_english_query="",
                modality_weights={"text": 1.0, "image": 0.0, "audio": 0.0}, confidence=0.9,
            )

    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: _Analyzer()
    app.dependency_overrides[get_search_router] = lambda: router
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: service
    return app, router


def _post(client, query: str = "가사에 창문을 여는 척이 나오는 노래") -> Dict[str, Any]:
    response = client.post("/search", json={"query": query, "explain": True, "use_rerank": False})
    assert response.status_code == 200, response.text
    return response.json()


def _quote(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    evidence = body["results"][0]["explain"]["evidence"]
    return evidence[0] if evidence else None


def test_http_request_after_expiry_returns_old_snapshot_without_waiting_for_refresh():
    from fastapi.testclient import TestClient

    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    clock = FakeClock()
    service = _service(source, clock)
    app, router = _http_app(service)
    try:
        with TestClient(app) as client:
            first = _post(client)
            assert [r["id"] for r in first["results"]] == ["s1"]
            assert _quote(first)["quote"] == "창문을 여는 척" and source.calls == 1
            old_version = _quote(first)["data_version"]

            source.docs = [_doc("s1", LYRICS_V2), _doc("s2", OTHER)]
            source.gate, source.gate_after = threading.Event(), 0
            clock.advance(301.0)

            (second, elapsed) = _timed(lambda: _post(client))
            assert elapsed < 2.0, f"만료 뒤 HTTP 요청이 갱신을 기다렸다({elapsed:.2f}s)"
            assert [r["id"] for r in second["results"]] == ["s1"]
            assert _quote(second)["data_version"] == old_version, "검색에 쓴 판과 다른 판을 인용했다"
            assert source.started.wait(WAIT) and service.status()["refreshing"] is True

            (third, elapsed3) = _timed(lambda: _post(client))
            assert elapsed3 < 2.0 and _quote(third)["data_version"] == old_version and source.calls == 2

            source.gate.set()
            _join_refresh(service)
            fourth = _post(client)
            assert _quote(fourth)["data_version"] != old_version
            assert _quote(fourth)["quote"] == "창문을 여는 척"
            assert LYRICS_V2[_quote(fourth)["char_start"]:_quote(fourth)["char_end"]] == "창문을 여는 척"
    finally:
        router.shutdown()
        service.close()


def test_http_quote_is_from_the_snapshot_the_search_used_even_if_replaced_mid_request():
    """검색은 옛 판, 인용은 그 뒤 — 사이에 판이 교체돼도 인용은 검색이 본 판에서 뜬다."""
    from fastapi.testclient import TestClient

    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock)
    app, router = _http_app(service)
    original_search = service.search

    def search_then_replace(clues, top_k=100, snapshot_out=None):
        tracks = original_search(clues, top_k=top_k, snapshot_out=snapshot_out)
        # 검색이 끝난 직후, 인용 전에 갱신이 끝나 판이 교체된 상황을 만든다.
        if source.gate is not None:
            source.gate.set()
            _join_refresh(service)
        return tracks

    service.search = search_then_replace  # type: ignore[method-assign]
    try:
        with TestClient(app) as client:
            first = _post(client)
            old_version = _quote(first)["data_version"]
            source.docs = [_doc("s1", LYRICS_V2)]
            source.gate, source.gate_after = threading.Event(), 1
            clock.advance(301.0)
            second = _post(client)
            assert service.snapshot().get("s1").lyrics_sha16 != old_version, "판이 교체되지 않았다"
            assert _quote(second)["data_version"] == old_version
            assert _quote(second)["char_start"] == LYRICS_V1.index("창문을")
    finally:
        router.shutdown()
        service.close()


def test_http_refresh_failure_keeps_serving_and_health_style_status_reports_it():
    from fastapi.testclient import TestClient

    source = FakeSource([_doc("s1", LYRICS_V1)])
    clock = FakeClock()
    service = _service(source, clock, retry=60.0)
    app, router = _http_app(service)
    try:
        with TestClient(app) as client:
            _post(client)
            source.fail_after = 0
            clock.advance(301.0)
            body = _post(client)
            _join_refresh(service)
            assert [r["id"] for r in body["results"]] == ["s1"]
            status = service.status()
            assert status["state"] == "stale" and status["refresh_failures"] == 1
            assert status["last_refresh_error"].startswith("ConnectionError")
            clock.advance(30.0)
            assert [r["id"] for r in _post(client)["results"]] == ["s1"] and source.calls == 2
    finally:
        router.shutdown()
        service.close()


# ---------------------------------------------------------------------------
# 의존성·수명 주기 배선
# ---------------------------------------------------------------------------

def test_dependency_close_and_status_do_not_create_the_service(monkeypatch):
    from src.backend.api import dependencies

    dependencies.get_lyrics_exact_search_service.cache_clear()
    assert dependencies.lyrics_cache_status() == {"state": "not_created"}
    assert dependencies.close_lyrics_exact_search_service() is True
    assert dependencies.get_lyrics_exact_search_service.peek() is None


def test_dependency_close_closes_the_singleton_and_clears_it(monkeypatch):
    from src.backend.api import dependencies

    monkeypatch.setenv("LYRICS_REFRESH_RETRY_SECONDS", "7")
    monkeypatch.setenv("LYRICS_MONGO_SOCKET_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("LYRICS_MONGO_BATCH_SIZE", "77")
    dependencies.get_lyrics_exact_search_service.cache_clear()
    service = dependencies.get_lyrics_exact_search_service()
    assert service._refresh_retry_seconds == 7.0
    assert service._mongo_socket_timeout_seconds == 33.0 and service._mongo_batch_size == 77
    assert dependencies.lyrics_cache_status()["state"] == "empty"
    assert dependencies.close_lyrics_exact_search_service() is True
    assert service.closed and dependencies.get_lyrics_exact_search_service.peek() is None
    dependencies.get_lyrics_exact_search_service.cache_clear()


def test_lifespan_closes_the_lyrics_service_after_warmup_and_before_the_vector_client(monkeypatch):
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main
    from src.backend import warmup

    order: List[str] = []
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())
    monkeypatch.setattr(warmup, "start_background", lambda: None)
    monkeypatch.setattr(warmup, "stop_and_wait", lambda timeout=None: order.append("warmup") or True)
    monkeypatch.setattr(backend_main, "close_lyrics_exact_search_service",
                        lambda timeout=None: order.append("lyrics") or True)
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: order.append("vector"))
    monkeypatch.setattr(backend_main, "lyrics_cache_status", lambda: {"state": "fresh", "documents": 3})

    with TestClient(backend_main.app) as client:
        body = client.get("/health").json()
        assert body["lyrics_cache"] == {"state": "fresh", "documents": 3}
    assert order == ["warmup", "lyrics", "vector"]


# ---------------------------------------------------------------------------
# 리뷰 2차 — 종료 뒤 클라이언트 생성 거부 · 최초 적재 추적 · cursor 정리 로그
# ---------------------------------------------------------------------------

def test_close_refuses_to_create_a_mongo_client_after_shutdown_and_waits_for_initial_load(monkeypatch):
    """최초 적재가 클라이언트 생성 직전에 멈춘 채 close() → 재개돼도 클라이언트를 열지 않고 적재는 닫힘으로 끝난다."""
    from src.retrieval import lyrics_exact_search as mod

    factory_calls: List[float] = []
    monkeypatch.setattr(mod, "_make_lyrics_mongo_client", lambda t: factory_calls.append(t) or object())

    before_client = threading.Event()
    reached = threading.Event()

    class _Service(LyricsExactSearchService):
        def _source_documents(self):
            reached.set()
            assert before_client.wait(WAIT), "테스트가 gate를 열지 않았다"
            return super()._source_documents()

    service = _Service(cache_ttl_seconds=300.0, mongo_socket_timeout_seconds=1.0, close_wait_margin_seconds=0.1)
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        try:
            service.search([CLUE])
            outcome["result"] = "returned"
        except Exception as exc:  # noqa: BLE001
            outcome["result"] = type(exc).__name__

    thread = threading.Thread(target=worker)
    thread.start()
    assert reached.wait(WAIT)
    assert service.status()["initial_loads_running"] == 1

    (finished, elapsed) = _timed(lambda: service.close(timeout=0.3))
    assert finished is False and 0.3 <= elapsed < 1.5, "진행 중 최초 적재를 기다리지 않고 True를 돌려줬다"

    before_client.set()
    thread.join(WAIT)
    assert outcome["result"] == "LyricsServiceClosed"
    assert factory_calls == [], "종료 뒤에 Mongo 클라이언트를 만들었다"
    assert service._mongo_client is None
    assert service.close() is True and service.status()["initial_loads_running"] == 0


def test_initial_load_in_progress_is_cancelled_by_close_and_counted():
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER), _doc("s3", OTHER)])
    source.gate, source.gate_after = threading.Event(), 1
    service = _service(source, FakeClock(), mongo_socket_timeout_seconds=1.0, close_wait_margin_seconds=0.1)
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        try:
            service.search([CLUE])
            outcome["result"] = "returned"
        except Exception as exc:  # noqa: BLE001
            outcome["result"] = type(exc).__name__

    thread = threading.Thread(target=worker)
    thread.start()
    assert source.started.wait(WAIT)
    assert service.close(timeout=0.2) is False
    source.gate.set()
    thread.join(WAIT)
    assert outcome["result"] == "LyricsServiceClosed"
    assert source.closed == 1, "취소된 최초 적재도 cursor를 닫는다"
    status = service.status()
    assert status["state"] == "empty" and status["refresh_cancelled"] == 1
    assert service.close() is True


def test_cursor_close_failure_logs_only_the_exception_type(caplog):
    import logging

    secret = "mongodb://user:pw@cluster.example.net:27017"

    class _Source(FakeSource):
        def _iter(self, docs, fail_after):
            cursor = super()._iter(docs, fail_after)

            def bad_close() -> None:
                raise RuntimeError(secret)

            cursor.close = bad_close  # type: ignore[method-assign]
            return cursor

    service = _service(_Source([_doc("s1", LYRICS_V1)]), FakeClock())
    with caplog.at_level(logging.DEBUG, logger="src.retrieval.lyrics_exact_search"):
        assert _ids(service.search([CLUE])) == ["s1"]
    assert "cursor 닫기 실패" in caplog.text and "RuntimeError" in caplog.text
    assert "pw@" not in caplog.text and "cluster.example.net" not in caplog.text


# ---------------------------------------------------------------------------
# 리뷰 3차 — 종료 확인과 적재 등록 사이의 경합
# ---------------------------------------------------------------------------

class _HookLock:
    """`_state_lock` 자리에 끼워 **잠금이 풀리는 순간**에 훅을 돌린다 — 임계 구역 바로 뒤를 재현한다."""

    def __init__(self, inner: threading.Lock, hook) -> None:
        self._inner = inner
        self._hook = hook

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self._inner.acquire(*args, **kwargs)

    def release(self) -> None:
        self._inner.release()
        self._hook()

    def __enter__(self):
        self._inner.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def locked(self) -> bool:
        return self._inner.locked()


def test_close_racing_with_initial_load_registration_either_refuses_or_waits():
    """적재 스레드가 종료 확인 구간을 막 벗어난 순간 close()가 끼어드는 경합.

    되돌리면 깨진다: 종료 확인 뒤 잠금을 풀었다가 다시 잡아 등록하던 판에서는 그 틈에 close()가
    "진행 중 적재 0건"으로 True를 돌려주고, 적재가 종료 뒤에 자료원을 읽기 시작했다.
    """
    source = FakeSource([_doc("s1", LYRICS_V1), _doc("s2", OTHER)])
    service = _service(source, FakeClock(), mongo_socket_timeout_seconds=1.0, close_wait_margin_seconds=0.1)
    inner = service._state_lock
    state: Dict[str, Any] = {"fired": False, "close_result": None, "loads_seen": None, "releases": 0}
    loader_ident: List[int] = []

    def hook() -> None:
        # 적재 스레드의 잠금 해제 순서: ① `_load()`의 판 확인 → ② `_load_blocking()`의 종료 확인·세대·등록.
        # ②를 벗어나는 순간에 한 번만 close()를 끼워 넣는다 — 예전 코드에서는 이 자리가 "종료 확인은 했지만
        # 아직 등록 전"이어서 close()가 True를 돌려줬다.
        if state["fired"] or threading.get_ident() not in loader_ident:
            return
        state["releases"] += 1
        if state["releases"] < 2:
            return
        state["fired"] = True
        state["loads_seen"] = service._blocking_loads
        state["close_result"] = service.close(timeout=0.2)

    service._state_lock = _HookLock(inner, hook)  # type: ignore[assignment]
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        loader_ident.append(threading.get_ident())
        try:
            service.search([CLUE])
            outcome["result"] = "returned"
        except Exception as exc:  # noqa: BLE001
            outcome["result"] = type(exc).__name__

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(WAIT)
    assert not thread.is_alive() and state["fired"]
    assert state["loads_seen"] == 1, "close()가 끼어든 시점에 적재가 이미 등록돼 있어야 한다"
    assert state["close_result"] is False, "등록된 적재를 두고 close()가 True를 돌려줬다"
    assert outcome["result"] == "LyricsServiceClosed" and service.status()["state"] == "empty"
    assert service.close() is True and service.status()["initial_loads_running"] == 0


def test_close_never_returns_true_before_a_registered_load_finishes_under_stress():
    """경합을 반복해도 "close()가 True를 돌려준 뒤 자료원 읽기가 시작"되는 순서는 없어야 한다."""
    violations = 0
    for _ in range(150):
        docs = [_doc("s1", LYRICS_V1)]
        closed_true_at: List[float] = []

        class _Source(FakeSource):
            def __call__(self):
                if closed_true_at:
                    nonlocal_violation[0] += 1
                return super().__call__()

        nonlocal_violation = [0]
        source = _Source(docs)
        service = _service(source, FakeClock(), mongo_socket_timeout_seconds=1.0, close_wait_margin_seconds=0.1)
        thread = threading.Thread(target=lambda: _safe_search(service))
        thread.start()
        if service.close() is True:
            closed_true_at.append(time.perf_counter())
        thread.join(WAIT)
        violations += nonlocal_violation[0]
        assert service.status()["initial_loads_running"] == 0
    assert violations == 0


def _safe_search(service: LyricsExactSearchService) -> None:
    try:
        service.search([CLUE])
    except LyricsServiceClosed:
        pass
