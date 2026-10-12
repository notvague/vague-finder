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


def test_last_revision_can_always_be_withdrawn(setup):
    store, receipt, _ = setup
    for index in range(MAX_REVISIONS):
        store.submit(submission(receipt, submission_id=f"submission_{index:08d}", expected_revision=index))
    with pytest.raises(FeedbackError):
        store.submit(submission(receipt, submission_id="submission_over_limit", expected_revision=MAX_REVISIONS))
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
