"""피드백 저장 계약과 API. 모델·기존 Mongo 데이터·Qdrant를 열지 않는다."""
import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from importlib import import_module

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from src.backend.schemas.feedback import FeedbackRequest
from src.backend.schemas.search import MatchingTrack, SearchRequest
from src.feedback import capture
from src.feedback.store import FeedbackError, FeedbackStore, MAX_REVISIONS
from src.retrieval.query_analyzer import rule_fallback


class Collection:
    """Mongo의 _id 유일성과 CAS를 잠금 안에서 재현한다."""
    def __init__(self):
        self.documents = {}
        self.indexes = []
        self.lock = threading.RLock()

    def create_index(self, name, **options):
        self.indexes.append((name, options))

    def insert_one(self, document):
        with self.lock:
            if document["_id"] in self.documents:
                raise DuplicateKeyError("duplicate")
            self.documents[document["_id"]] = copy.deepcopy(document)

    def find_one(self, query):
        with self.lock:
            doc = self.documents.get(query["_id"])
            if doc is None:
                return None
            for key, value in query.items():
                if isinstance(value, dict):
                    if not doc[key] > value["$gt"]:
                        return None
                elif doc.get(key) != value:
                    return None
            return copy.deepcopy(doc)

    def find_one_and_update(self, query, update, **options):
        with self.lock:
            doc = self.find_one(query)
            if doc is None:
                return None
            doc.update(copy.deepcopy(update.get("$set", {})))
            for key, value in update.get("$push", {}).items():
                doc.setdefault(key, []).append(copy.deepcopy(value))
            self.documents[doc["_id"]] = doc
            return copy.deepcopy(doc)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    # 모듈에 남는 상태 — 앞 시험의 실패 대기·분석기 지문이 다음 시험으로 새지 않게 한다.
    monkeypatch.setattr(capture, "_unavailable_until", 0.0)
    capture._analyzer_fingerprint.cache_clear()
    database = {"search_runs": Collection(), "search_feedback": Collection()}
    store = FeedbackStore(database)
    receipt = store.record_search({"query": "휘파람 노래", "shown_ids": ["a", "b"], "candidate_ids": ["a", "b", "hidden"]})
    return store, receipt, database


def submission(receipt, **changes):
    return FeedbackRequest(**{
        "search_id": receipt.search_id, "token": receipt.token, "submission_id": "test_submission_1",
        "selected_song_id": "a", "reason": "도입부 휘파람", **changes,
    })


def test_only_displayed_songs_and_matching_search_token_are_accepted(setup):
    store, receipt, db = setup
    for changes, status in [({"selected_song_id": "hidden"}, 422), ({"token": "x" * 43}, 404),
                            ({"search_id": "a" * 32}, 404)]:
        with pytest.raises(FeedbackError) as error:
            store.submit(submission(receipt, **changes))
        assert error.value.status == status
    assert not db["search_feedback"].documents
    result = store.submit(submission(receipt))
    assert result.status == "pending"
    stored = db["search_runs"].documents[receipt.search_id]
    assert "token" not in stored and stored["token_hash"] != receipt.token
    assert db["search_runs"].indexes == [("expires_at", {"expireAfterSeconds": 0})]
    assert db["search_feedback"].indexes == [("expires_at", {"expireAfterSeconds": 0})]


def test_expiry_is_enforced_before_mongo_ttl_sweeps(setup):
    store, receipt, db = setup
    db["search_runs"].documents[receipt.search_id]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(FeedbackError, match="보관 기간"):
        store.submit(submission(receipt))


def test_retries_edits_and_withdrawal_preserve_history_and_invalidate_review(setup):
    store, receipt, db = setup
    first = store.submit(submission(receipt))
    assert store.submit(submission(receipt)) == first
    saved = db["search_feedback"].documents[receipt.search_id]
    saved["review"], saved["reusable_text"] = {"approved": True}, "old"
    second = store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1, selected_song_id="b", reason=""))
    assert second.selected_song_id == "b" and second.revision == 2
    assert store.submit(submission(receipt)) == second  # 오래된 재시도가 현재 선택을 되돌리지 않는다
    saved = db["search_feedback"].documents[receipt.search_id]
    assert saved["review"] is None and saved["reusable_text"] is None
    assert [e["selected_song_id"] for e in saved["history"]] == ["a", "b"]
    withdrawn = store.submit(submission(receipt, submission_id="test_submission_3", expected_revision=2,
                                       action="withdraw", selected_song_id=None, reason=""))
    assert withdrawn.status == "withdrawn" and withdrawn.reason == "" and withdrawn.selected_song_id is None
    assert len(db["search_feedback"].documents) == 1
    assert db["search_feedback"].documents[receipt.search_id]["expires_at"] == db["search_runs"].documents[receipt.search_id]["expires_at"]


def test_reused_key_with_different_content_and_stale_revision_conflict(setup):
    store, receipt, _ = setup
    store.submit(submission(receipt))
    for changes in [{"selected_song_id": "b"}, {"submission_id": "test_submission_2"}]:
        with pytest.raises(FeedbackError) as error:
            store.submit(submission(receipt, **changes))
        assert error.value.status == 409
        assert error.value.current.selected_song_id == "a"


@pytest.mark.parametrize("revision", [0, 1])
def test_concurrent_submissions_never_overwrite_each_other(setup, revision):
    store, receipt, db = setup
    if revision:
        store.submit(submission(receipt, submission_id="initial_submission"))
    barrier = threading.Barrier(2)
    original = store.feedback.find_one
    calls = threading.local()

    def simultaneous_read(query):
        result = original(query)
        if not getattr(calls, "read", False):
            calls.read = True
            barrier.wait(timeout=5)
        return result

    store.feedback.find_one = simultaneous_read

    def send(key):
        try:
            return store.submit(submission(receipt, submission_id=key, expected_revision=revision))
        except FeedbackError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(send, ["test_submission_1", "test_submission_2"]))
    assert sum(value == 409 for value in answers) == 1
    assert len(db["search_feedback"].documents[receipt.search_id]["history"]) == revision + 1


def test_last_revision_can_always_be_withdrawn(setup, monkeypatch):
    store, receipt, _ = setup
    for index in range(MAX_REVISIONS):
        store.submit(submission(receipt, submission_id=f"submission_{index:08d}", expected_revision=index))
    with pytest.raises(FeedbackError) as error:
        store.submit(submission(receipt, submission_id="submission_over_limit", expected_revision=MAX_REVISIONS))
    assert error.value.code == "revision_limit"
    route = import_module("src.backend.api.routes.feedback")
    monkeypatch.setattr(route, "get_feedback_store", lambda: store)
    app = FastAPI()
    app.include_router(route.router)
    with TestClient(app) as client:
        response = client.post("/feedback", json=submission(
            receipt, submission_id="submission_over_limit", expected_revision=MAX_REVISIONS,
        ).model_dump())
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "revision_limit", "message": "이 검색의 수정 횟수를 모두 사용했습니다.",
        "current": error.value.current.model_dump(),
    }
    # 한도에 도달한 마지막 제출도 응답 유실 시 같은 키로 재시도할 수 있다.
    replay = store.submit(submission(receipt, submission_id=f"submission_{MAX_REVISIONS - 1:08d}",
                                    expected_revision=MAX_REVISIONS - 1))
    assert replay.revision == MAX_REVISIONS
    result = store.submit(submission(receipt, submission_id="submission_withdraw", expected_revision=MAX_REVISIONS,
                                    action="withdraw", selected_song_id=None, reason=""))
    assert result.status == "withdrawn"


def test_parent_link_requires_same_query_and_valid_token(setup):
    store, parent, db = setup
    child = store.record_search({"query": "휘파람 노래", "shown_ids": ["c"]}, parent)
    assert db["search_runs"].documents[child.search_id]["conversation_id"] == parent.search_id
    for query, invalid in [("휘파람 노래", parent.model_copy(update={"token": "x" * 43})), ("다른 검색", parent)]:
        other = store.record_search({"query": query, "shown_ids": ["d"]}, invalid)
        assert db["search_runs"].documents[other.search_id]["parent_search_id"] is None


def test_http_submit_validation_conflict_and_db_failure(setup, monkeypatch):
    store, receipt, _ = setup
    route = import_module("src.backend.api.routes.feedback")
    monkeypatch.setattr(route, "get_feedback_store", lambda: store)
    app = FastAPI()
    app.include_router(route.router)
    with TestClient(app) as client:
        body = submission(receipt).model_dump()
        ok = client.post("/feedback", json=body)
        assert ok.status_code == 200 and ok.headers["Cache-Control"] == "no-store"
        assert client.post("/feedback", json={**body, "selected_song_id": "b"}).status_code == 409
        for bad in [{"reason": "a" * 1001}, {"selected_song_id": " "}, {"action": "withdraw"}, {"unexpected": True}]:
            assert client.post("/feedback", json={**body, **bad}).status_code == 422
        monkeypatch.setattr(store, "submit", lambda request: (_ for _ in ()).throw(RuntimeError("secret URI")))
        response = client.post("/feedback", json=body)
        assert response.status_code == 503 and "secret" not in response.text
        monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
        assert client.post("/feedback", json=body).status_code == 503


def test_capture_off_does_not_connect_and_failure_does_not_escape(setup, monkeypatch):
    store, _, _ = setup
    calls = []
    def unavailable():
        calls.append(True)
        raise TimeoutError("unavailable")
    monkeypatch.setattr(capture, "get_feedback_store", unavailable)
    request = SearchRequest(query="휘파람")
    analysis = rule_fallback(request.query)
    tracks = [MatchingTrack(id="a", title="A", score=1)]
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
    assert asyncio.run(capture.capture_search(request, analysis, tracks, ["a"], 1, None)) is None
    assert calls == []
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    assert asyncio.run(capture.capture_search(request, analysis, tracks, ["a"], 1, None)) is None
    assert calls == [True]


def test_capture_keeps_effective_era_but_not_birth_year_answer(setup, monkeypatch):
    store, _, db = setup
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    analysis = rule_fallback("중학교 때 노래")
    # 실제 스키마는 규칙 분석에서 추출된 생애 단계 값을 사용한다.
    assert analysis.has_life_stage
    request = SearchRequest(query=analysis.original_query, prior_analysis=analysis,
                            answers=[{"slot": "birth_year", "value": "1998"}])
    receipt = asyncio.run(capture.capture_search(request, analysis, [MatchingTrack(id="a", title="A", score=1)], ["a"], 2, None))
    assert receipt is not None
    stored = db["search_runs"].documents[receipt.search_id]
    assert stored["answers"] == [] and "birth_year" not in stored
    assert stored["analysis"]["release_era"] is not None


def test_capture_file_reads_and_storage_run_outside_event_loop(setup, monkeypatch):
    store, _, db = setup
    from src.retrieval import analysis_cache
    loop_thread = threading.get_ident()
    observed = []

    def fingerprint(name, result):
        observed.append((name, threading.get_ident()))
        return result

    original_record = store.record_search
    def record(*args):
        observed.append(("store", threading.get_ident()))
        return original_record(*args)

    monkeypatch.setattr(capture, "source_version", lambda: fingerprint("source", "source-version"))
    monkeypatch.setattr(analysis_cache, "analyzer_fingerprint", lambda: fingerprint("analyzer", {"version": "test"}))
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    monkeypatch.setattr(store, "record_search", record)
    request = SearchRequest(query="휘파람")
    analysis = rule_fallback(request.query)
    tracks = [MatchingTrack(id="a", title="A", score=1)]

    receipt = asyncio.run(capture.capture_search(request, analysis, tracks, ["a"], 1, None))
    assert receipt is not None
    assert [name for name, _ in observed] == ["source", "analyzer", "store"]
    assert all(thread_id != loop_thread for _, thread_id in observed)
    versions = db["search_runs"].documents[receipt.search_id]["versions"]
    assert versions["source_sha"] == "source-version" and versions["analyzer"] == {"version": "test"}

    observed.clear()
    assert asyncio.run(capture.capture_search(request, analysis, [], [], 1, None)) is None
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
    assert asyncio.run(capture.capture_search(request, analysis, tracks, ["a"], 1, None)) is None
    assert observed == []


def test_real_search_route_receipt_submission_and_storage_failure(setup, monkeypatch):
    store, _, db = setup
    search = import_module("src.backend.api.routes.search")
    feedback = import_module("src.backend.api.routes.feedback")
    from src.backend.api.dependencies import get_query_analyzer, get_search_router, get_lyrics_exact_search_service
    from types import SimpleNamespace

    class Router:
        async def search(self, analysis, **kw):
            tracks = [MatchingTrack(id=sid, title=sid, score=1) for sid in ["a", "b", "hidden"]]
            kw["candidate_ids_out"].extend(t.id for t in tracks)
            kw["candidate_tracks_out"].extend(tracks)
            return tracks[:kw["top_k"]]

    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    monkeypatch.setattr(feedback, "get_feedback_store", lambda: store)
    app = FastAPI()
    app.include_router(search.router)
    app.include_router(feedback.router)
    app.dependency_overrides[get_query_analyzer] = lambda: SimpleNamespace(analyze=rule_fallback)
    app.dependency_overrides[get_search_router] = Router
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: None
    with TestClient(app) as client:
        args = {"query": "휘파람 노래", "top_k": 2, "defer_clarify": True}
        first = client.post("/search", json=args)
        assert first.status_code == 200
        data = first.json()
        receipt = data["feedback"]
        body = {"search_id": receipt["search_id"], "token": receipt["token"],
                "submission_id": "http_submission_1", "selected_song_id": "a"}
        assert client.post("/feedback", json=body).status_code == 200
        assert client.post("/feedback", json={**body, "selected_song_id": "hidden"}).status_code == 422
        next_data = client.post("/search", json={**args, "feedback_parent": receipt}).json()
        assert next_data["feedback"]["search_id"] != receipt["search_id"]
        assert db["search_runs"].documents[next_data["feedback"]["search_id"]]["parent_search_id"] == receipt["search_id"]

        def fail(*args):
            raise TimeoutError("database unavailable")
        monkeypatch.setattr(store, "record_search", fail)
        failed_capture = client.post("/search", json=args)
        assert failed_capture.status_code == 200
        assert failed_capture.json()["feedback"] is None
        assert failed_capture.json()["results"] == data["results"]
        monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
        assert client.post("/search", json=args).json()["results"] == data["results"]


def test_retry_after_database_write_succeeded_but_acknowledgement_was_lost(setup, monkeypatch):
    store, receipt, db = setup
    original = store.feedback.insert_one
    def write_then_fail(doc):
        original(doc)
        raise TimeoutError("acknowledgement lost")
    monkeypatch.setattr(store.feedback, "insert_one", write_then_fail)
    with pytest.raises(TimeoutError):
        store.submit(submission(receipt))
    result = store.submit(submission(receipt))
    assert result.revision == 1
    assert len(db["search_feedback"].documents[receipt.search_id]["history"]) == 1


# --- 리뷰 반영: 저장소가 느리거나 연결 값이 틀려도 검색은 그대로 나간다 -------------------------

def _search_args(query="휘파람"):
    request = SearchRequest(query=query)
    return request, rule_fallback(query), [MatchingTrack(id="a", title="A", score=1)]


def test_searches_racing_the_first_index_build_still_get_receipts(setup):
    """기동 직후 동시에 들어온 검색. 한 요청이 색인을 만드는 동안 나머지는 기다리지 않고 기록한다.

    예전에는 0.05초만 기다리다 TimeoutError를 내서 그 검색들이 제출 권한 없이 나갔다.
    """
    database = {"search_runs": Collection(), "search_feedback": Collection()}
    store = FeedbackStore(database)
    building, release = threading.Event(), threading.Event()
    original = database["search_runs"].create_index

    def slow_create_index(name, **options):
        building.set()
        assert release.wait(5)
        original(name, **options)

    database["search_runs"].create_index = slow_create_index
    snapshot = {"query": "휘파람 노래", "shown_ids": ["a"], "candidate_ids": ["a"]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(store.record_search, dict(snapshot))
        assert building.wait(5)
        racing = store.record_search(dict(snapshot))      # 색인이 아직 만들어지는 중이다
        assert racing.search_id in database["search_runs"].documents
        assert not store._indexed
        release.set()
        assert first.result(5).search_id in database["search_runs"].documents
    assert store._indexed
    assert len(database["search_runs"].indexes) == 1 and len(database["search_feedback"].indexes) == 1


def test_a_failed_index_build_does_not_block_recording_and_is_retried(setup, caplog):
    """색인은 지난 문서를 지우는 용도다. 못 만들어도 기록은 되고, 다음 기록에서 다시 만든다."""
    database = {"search_runs": Collection(), "search_feedback": Collection()}
    store = FeedbackStore(database)
    original = database["search_runs"].create_index
    attempts = []

    def flaky(name, **options):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("mongodb://secret-uri timed out")
        original(name, **options)

    database["search_runs"].create_index = flaky
    snapshot = {"query": "휘파람 노래", "shown_ids": ["a"], "candidate_ids": ["a"]}
    assert store.record_search(dict(snapshot)).search_id in database["search_runs"].documents
    assert not store._indexed
    assert "secret-uri" not in caplog.text, "예외 본문에는 URI가 들어갈 수 있다 — 종류만 남긴다"
    store.record_search(dict(snapshot))
    assert store._indexed and len(attempts) == 2


def test_startup_prepares_the_ttl_index_without_connecting_when_collection_is_off(setup, monkeypatch):
    from src.feedback import store as store_module

    database = {"search_runs": Collection(), "search_feedback": Collection()}
    store = FeedbackStore(database)
    opened = []
    monkeypatch.setattr(store_module, "get_feedback_store", lambda: opened.append(True) or store)

    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
    assert store_module.prepare_indexes() is False and opened == []

    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    assert store_module.prepare_indexes() is True
    assert database["search_runs"].indexes == [("expires_at", {"expireAfterSeconds": 0})]

    # 실패해도 기동을 막지 않는다.
    monkeypatch.setattr(store_module, "get_feedback_store", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert store_module.prepare_indexes() is False


def test_an_unavailable_store_is_not_retried_on_every_search(setup, monkeypatch):
    """저장소가 내려가 있으면 결과가 있는 검색마다 연결 제한을 기다리게 된다. 한 번 실패하면 잠시 쉰다."""
    store, _, _ = setup
    calls = []

    def unavailable():
        calls.append(True)
        raise TimeoutError("unavailable")

    clock = [1000.0]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(capture, "get_feedback_store", unavailable)
    args = _search_args()

    assert asyncio.run(capture.capture_search(*args, ["a"], 1, None)) is None
    assert asyncio.run(capture.capture_search(*args, ["a"], 1, None)) is None
    assert calls == [True], "두 번째 검색은 저장소를 기다리지 않는다"

    clock[0] += capture.RETRY_AFTER_SECONDS + 1
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    assert asyncio.run(capture.capture_search(*args, ["a"], 1, None)) is not None, "대기가 끝나면 다시 기록한다"


def test_the_analyzer_fingerprint_is_read_once_per_process(setup, monkeypatch):
    store, _, _ = setup
    from src.retrieval import analysis_cache

    reads = []
    monkeypatch.setattr(analysis_cache, "analyzer_fingerprint", lambda: reads.append(True) or {"version": "test"})
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    args = _search_args()
    for _ in range(3):
        assert asyncio.run(capture.capture_search(*args, ["a"], 1, None)) is not None
    assert reads == [True]


def test_reranker_config_is_stored_as_plain_values(setup, monkeypatch):
    """설정에 Mongo가 못 받는 값이 생기면 insert가 실패해 수집 전체가 조용히 꺼진다."""
    from dataclasses import dataclass, field
    from enum import Enum
    from pathlib import Path
    from types import SimpleNamespace

    class Mode(Enum):
        FAST = "fast"

    @dataclass(frozen=True)
    class Config:
        model_name: str = "m"
        cache_dir: Path = Path("/tmp/cache")
        mode: Mode = Mode.FAST
        tags: frozenset = field(default_factory=lambda: frozenset({"a"}))

    store, _, db = setup
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    router = SimpleNamespace(_reranker=SimpleNamespace(config=Config()))
    receipt = asyncio.run(capture.capture_search(*_search_args(), ["a"], 1, router))
    stored = db["search_runs"].documents[receipt.search_id]["versions"]["reranker_config"]
    assert stored["model_name"] == "m" and stored["cache_dir"] == "/tmp/cache"
    assert all(isinstance(v, (str, int, float, bool, list, dict, type(None))) for v in stored.values())


@pytest.mark.parametrize("parent", [
    {"search_id": "0" * 32, "token": "잘린토큰"},
    {"search_id": "not-hex", "token": "A" * 43},
    {"search_id": "0" * 32},
    "문자열",
    [],
])
def test_an_unusable_feedback_parent_does_not_reject_the_search(parent):
    """연결용 값이다. 형식이 틀리면 버리고 검색은 받는다 — 예전에는 요청 전체가 422였다."""
    request = SearchRequest(query="휘파람", feedback_parent=parent)
    assert request.feedback_parent is None and request.query == "휘파람"


def test_a_valid_feedback_parent_is_kept():
    parent = {"search_id": "0" * 32, "token": "A" * 43}
    request = SearchRequest(query="휘파람", feedback_parent=parent)
    assert (request.feedback_parent.search_id, request.feedback_parent.token) == ("0" * 32, "A" * 43)
