"""실제 Mongo의 TTL 삭제와 디스크에 쓰는 로컬 Qdrant의 만료 정리. **대역으로는 검증할 수 없는 것만** 본다.

`FEEDBACK_TEST_MONGO_URI`가 있을 때만 돈다(없으면 건너뛴다). 시험마다 무작위 이름의 DB를 만들고 끝나면
지운다 — 운영 DB·운영 Qdrant 폴더를 쓰지 않는다. TTL 청소 주기는 Mongo 기본 60초라 최대 150초를 기다린다.

    docker run -d --rm --name vf-feedback-test-mongo -p 127.0.0.1:27917:27017 mongo:7 \
        --setParameter ttlMonitorSleepSecs=5
    FEEDBACK_TEST_MONGO_URI=mongodb://127.0.0.1:27917 venv/bin/python -m pytest tests/feedback/test_feedback_mongo_integration.py -q
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.feedback.case_index import FeedbackCaseIndex
from src.feedback.indexer import FeedbackIndexer
from src.feedback.retrieval import FeedbackRetriever
from src.feedback.settings import EMBEDDING_VERSION, RetrievalSettings, WorkerSettings
from src.feedback.store import TTL_NOT_TTL, TTL_READY, TTL_WRONG_SECONDS, FeedbackStore
from tests.feedback_fakes import MONGO_URI_ENV, TEST_DIM, KeywordEmbedder, _real_database, approve, seed_songs, submission

pytestmark = pytest.mark.skipif(not os.getenv(MONGO_URI_ENV), reason=f"{MONGO_URI_ENV}가 없다 — 실제 Mongo 통합 시험을 건너뛴다")

TTL_WAIT_SECONDS = 150


@pytest.fixture
def mongo():
    client, database, name = _real_database()
    try:
        yield database
    finally:
        client.drop_database(name)
        client.close()


def _wait_until(condition, seconds=TTL_WAIT_SECONDS):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(1)
    return condition()


def test_ttl_index_is_created_and_reported_ready(mongo):
    store = FeedbackStore(mongo)
    assert store.ensure_indexes() is True
    assert store.ttl_status() == {"search_runs": TTL_READY, "search_feedback": TTL_READY}
    info = mongo["search_feedback"].index_information()
    assert any(spec.get("expireAfterSeconds") == 0 and spec["key"] == [("expires_at", 1)] for spec in info.values())


def test_an_existing_non_ttl_index_blocks_creation_and_is_reported(mongo, caplog):
    """같은 키에 TTL이 아닌 색인이 이미 있으면 Mongo가 TTL 색인 생성을 거부한다. 기록은 계속되고 상태에 드러난다."""
    mongo["search_feedback"].create_index("expires_at")
    mongo["search_runs"].create_index("expires_at", expireAfterSeconds=3600)
    store = FeedbackStore(mongo)
    assert store.ensure_indexes() is False
    assert store.ttl_status() == {"search_runs": TTL_WRONG_SECONDS, "search_feedback": TTL_NOT_TTL}
    receipt = store.record_search({"query": "합성 질의", "shown_ids": ["a"], "candidate_ids": ["a"]})
    assert store.submit(submission(receipt)).status == "pending", "TTL이 준비되지 않아도 수집 자체는 된다"

    # 색인 작업은 이 상태에서 만료 문서를 직접 지운다(TTL의 대체 수단).
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    mongo["search_runs"].update_one({"_id": receipt.search_id}, {"$set": {"expires_at": past}})
    mongo["search_feedback"].update_one({"_id": receipt.search_id}, {"$set": {"expires_at": past}})
    from src.vector_db.qdrant_backend import QdrantVectorClient

    client = QdrantVectorClient(path=":memory:")
    try:
        report = FeedbackIndexer(store, FeedbackCaseIndex(client, dim=TEST_DIM), KeywordEmbedder(), WorkerSettings()).run_once()
    finally:
        client.close()
    assert report.mongo_expired_deleted == 2
    assert mongo["search_feedback"].count_documents({}) == 0 and mongo["search_runs"].count_documents({}) == 0


def test_mongo_ttl_deletes_run_feedback_review_and_text_together(mongo):
    """만료 시각이 지나면 원문·선택·사유·이력·검토 결과·재사용 문장이 든 문서가 통째로 사라진다."""
    store = FeedbackStore(mongo)
    receipt = store.record_search({"query": "합성 질의", "shown_ids": ["a"], "candidate_ids": ["a"]})
    store.submit(submission(receipt))
    approve(store, receipt.search_id, "합성 재사용 문장")
    expires = datetime.now(timezone.utc) + timedelta(seconds=3)
    for name in ("search_runs", "search_feedback"):
        mongo[name].update_one({"_id": receipt.search_id}, {"$set": {"expires_at": expires}})
    assert _wait_until(lambda: mongo["search_feedback"].count_documents({}) == 0
                       and mongo["search_runs"].count_documents({}) == 0), "TTL이 만료 문서를 지우지 않았다"


def test_expiry_end_to_end_with_real_mongo_and_on_disk_qdrant(mongo, tmp_path):
    """승인 → 색인 → 검색 사용 → 만료. 만료 뒤에는 검색에서 빠지고, Mongo와 Qdrant 양쪽에서 실제로 지워진다."""
    from src.vector_db.qdrant_backend import QdrantVectorClient

    store = FeedbackStore(mongo)
    client = QdrantVectorClient(path=str(tmp_path / "qdrant"))       # 운영 폴더가 아닌 임시 경로
    try:
        seed_songs(client, {"a": {"title": "합성 곡", "artist": "합성 가수"}})
        index = FeedbackCaseIndex(client, dim=TEST_DIM)
        indexer = FeedbackIndexer(store, index, KeywordEmbedder(), WorkerSettings())
        receipt = store.record_search({"query": "합성 질의", "shown_ids": ["a"], "candidate_ids": ["a"]})
        store.submit(submission(receipt))
        # 만료를 먼저 당겨 둔다 — 승인·색인이 기간을 늘리지 않으므로 포인트도 이 시각을 물려받는다.
        expires = datetime.now(timezone.utc) + timedelta(seconds=8)
        for name in ("search_runs", "search_feedback"):
            mongo[name].update_one({"_id": receipt.search_id}, {"$set": {"expires_at": expires}})
        approve(store, receipt.search_id, "휘파람 전주로 시작하는 노래")
        assert indexer.run_once().indexed == 1
        payload = index.retrieve([receipt.search_id])[receipt.search_id]
        assert payload["expires_at_ts"] == pytest.approx(expires.timestamp(), abs=0.002)
        assert payload["embedding_version"] == EMBEDDING_VERSION

        retriever = FeedbackRetriever(index, lambda: store, KeywordEmbedder, client,
                                      settings=lambda: RetrievalSettings(budget_seconds=5.0))
        used = retriever.validate(retriever.find_cases("휘파람 불던 노래"))
        assert [item.track.id for item in used.items] == ["a"], "만료 전에는 검색이 이 사례를 쓴다"

        time.sleep(max(0.0, (expires - datetime.now(timezone.utc)).total_seconds()) + 0.5)
        assert retriever.find_cases("휘파람 불던 노래").hits == [], "만료 시각부터 조회에서 빠진다(삭제를 기다리지 않는다)"
        assert _wait_until(lambda: mongo["search_feedback"].count_documents({}) == 0), "Mongo TTL이 지우지 않았다"
        assert index.count() == 1, "Mongo TTL 삭제는 Qdrant로 전파되지 않는다"
        report = indexer.run_once()
        assert report.deleted_expired_points == 1 and index.count() == 0 and report.expired_points_remaining == 0
    finally:
        client.close()

    # 다시 열어도 지워진 상태다(디스크에 반영됐다).
    reopened = QdrantVectorClient(path=str(tmp_path / "qdrant"))
    try:
        assert FeedbackCaseIndex(reopened, dim=TEST_DIM).count() == 0
    finally:
        reopened.close()
