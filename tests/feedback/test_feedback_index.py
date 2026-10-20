"""승인 사례 색인과 자동 정리(③-B). Qdrant는 메모리 모드 실물, Mongo는 대역 + (있으면) 실물.

KoE5는 올리지 않는다 — 주제어로 벡터를 정하는 가짜 임베딩을 쓴다(`tests/feedback_fakes.py`).
"""
import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.feedback import indexer as indexer_module
from src.feedback import store as store_module
from src.feedback import worker
from src.feedback.case_index import FeedbackCaseIndex
from src.feedback.indexer import ERROR_EMBEDDING, ERROR_QDRANT, ERROR_STORE, FeedbackIndexer, retry_delay
from src.feedback.review import CaseVersion, point_uuid
from src.feedback.settings import EMBEDDING_VERSION, WorkerSettings
from tests.feedback_fakes import (  # noqa: F401
    TEST_DIM,
    KeywordEmbedder,
    approve,
    expire,
    memory_qdrant,
    store_backend,
    submission,
    submit_case,
)

PAYLOAD_KEYS = {"feedback_id", "song_id", "source_revision", "review_version", "embedding_version",
                "expires_at_ts", "expires_at", "source"}


class Clock:
    def __init__(self):
        self.offset = timedelta(0)

    def __call__(self):
        return datetime.now(timezone.utc) + self.offset

    def advance(self, **kwargs):
        self.offset += timedelta(**kwargs)


@pytest.fixture
def rig(store_backend):
    store, database = store_backend
    client = memory_qdrant()
    case_index = FeedbackCaseIndex(client, dim=TEST_DIM)
    embedder = KeywordEmbedder()
    clock = Clock()
    indexer = FeedbackIndexer(store, case_index, embedder, WorkerSettings(), clock)
    yield type("Rig", (), dict(store=store, database=database, index=case_index, embedder=embedder,
                               clock=clock, indexer=indexer, client=client))
    client.close()


def _doc(rig, search_id):
    return rig.database["search_feedback"].find_one({"_id": search_id})


def _point(rig, search_id):
    return rig.index.retrieve([search_id]).get(search_id)


def _approved(rig, **kw):
    receipt, _ = submit_case(rig.store, **kw)
    approve(rig.store, receipt.search_id)
    return receipt


def test_an_approved_case_is_indexed_without_copying_user_text(rig):
    receipt = _approved(rig)
    report = rig.indexer.run_once()
    assert report.ok and report.indexed == 1 and report.points == 1
    payload = _point(rig, receipt.search_id)
    assert set(payload) == PAYLOAD_KEYS, "원질의·사유·재사용 문장은 Qdrant에 복사하지 않는다"
    assert (payload["song_id"], payload["source_revision"], payload["review_version"]) == ("a", 1, 1)
    assert payload["embedding_version"] == EMBEDDING_VERSION and payload["source"] == "shown_result"
    saved = _doc(rig, receipt.search_id)
    assert payload["expires_at_ts"] == pytest.approx(saved["expires_at"].timestamp()), "만료는 검색 기록의 것을 그대로"
    assert saved["index"]["state"] == "indexed" and saved["index"]["embedding_version"] == EMBEDDING_VERSION
    assert rig.embedder.calls == [["휘파람 전주로 시작하는 잔잔한 노래"]], "운영자가 쓴 재사용 문장만 임베딩한다"
    records = rig.client.client.retrieve(rig.index.collection, ids=[point_uuid(receipt.search_id)], with_vectors=True)
    assert len(records[0].vector["dense"]) == TEST_DIM


def test_pending_hold_rejected_and_withdrawn_cases_are_never_indexed(rig):
    pending, _ = submit_case(rig.store, query="대기")
    held, _ = submit_case(rig.store, query="보류")
    rig.store.apply_review(held.search_id, state="on_hold", reusable_text="보류 중 초안", note="",
                           expected_revision=1, expected_review_version=0, reviewer="운영자")
    withdrawn = _approved(rig, query="철회")
    rig.store.submit(submission(withdrawn, submission_id="test_submission_2", expected_revision=1,
                                action="withdraw", selected_song_id=None, reason=""))
    report = rig.indexer.run_once()
    assert report.indexed == 0 and report.points == 0 and rig.embedder.calls == []


def test_running_the_cycle_again_does_not_duplicate_or_reembed(rig):
    receipt = _approved(rig)
    rig.indexer.run_once()
    for _ in range(3):
        report = rig.indexer.run_once()
        assert report.ok and report.indexed == 0 and report.points == 1 and report.repaired == 0
    assert len(rig.embedder.calls) == 1
    # 응답을 잃어 같은 사례를 다시 올려도 포인트는 하나다(사례마다 정해진 ID).
    saved = _doc(rig, receipt.search_id)
    case = CaseVersion(receipt.search_id, "a", 1, 1)
    rig.index.upsert([(case, [1.0] + [0.0] * (TEST_DIM - 1), saved["expires_at"])], embedding_version=EMBEDDING_VERSION)
    assert rig.index.count() == 1


@pytest.mark.parametrize("failing, code", [("embed", ERROR_EMBEDDING), ("upsert", ERROR_QDRANT)])
def test_failures_are_recorded_and_retried_after_a_backoff(rig, monkeypatch, failing, code):
    receipt = _approved(rig)
    original_upsert = rig.index.upsert
    if failing == "embed":
        rig.embedder.fail = RuntimeError("model crashed: mongodb://secret")
    else:
        monkeypatch.setattr(rig.index, "upsert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    report = rig.indexer.run_once()
    assert report.errors == [code] and report.index_failed == 1 and report.points == 0
    saved = _doc(rig, receipt.search_id)["index"]
    assert (saved["state"], saved["attempts"], saved["error"]) == ("failed", 1, code)
    assert "secret" not in str(saved)

    rig.embedder.fail = None
    monkeypatch.setattr(rig.index, "upsert", original_upsert)
    assert rig.indexer.run_once().indexed == 0, "재시도 시각 전에는 다시 시도하지 않는다"
    rig.clock.advance(seconds=retry_delay(1) + 1)
    report = rig.indexer.run_once()
    assert report.ok and report.indexed == 1 and _doc(rig, receipt.search_id)["index"]["state"] == "indexed"


def test_repeated_failures_back_off_exponentially(rig):
    receipt = _approved(rig)
    rig.embedder.fail = RuntimeError("down")
    for attempt in (1, 2, 3):
        rig.indexer.run_once()
        saved = _doc(rig, receipt.search_id)["index"]
        assert saved["attempts"] == attempt
        rig.clock.advance(seconds=retry_delay(attempt) + 1)
    assert [retry_delay(n) for n in (1, 2, 3, 20)] == [60.0, 120.0, 240.0, 3600.0]


def test_a_restarted_worker_picks_up_approvals_it_never_saw(rig):
    """승인은 Mongo에 있다. 작업이 죽어 있던 동안의 승인도 새 프로세스의 첫 주기에서 색인된다."""
    receipts = [_approved(rig, query=f"재시작 {index}") for index in range(3)]
    fresh = FeedbackIndexer(rig.store, rig.index, rig.embedder, WorkerSettings(embed_batch=2), rig.clock)
    report = fresh.run_once()
    assert report.indexed == 3 and [len(call) for call in rig.embedder.calls] == [2, 1], "작은 묶음으로 임베딩한다"
    assert all(_point(rig, receipt.search_id) for receipt in receipts)


def test_a_cycle_handles_only_a_bounded_batch(rig):
    for index in range(5):
        _approved(rig, query=f"묶음 {index}")
    small = FeedbackIndexer(rig.store, rig.index, rig.embedder, WorkerSettings(index_batch=2), rig.clock)
    assert [small.run_once().indexed for _ in range(3)] == [2, 2, 1]


def test_an_edit_during_embedding_is_not_indexed(rig, monkeypatch):
    receipt = _approved(rig)
    rig.embedder.before = lambda: rig.store.submit(submission(
        receipt, submission_id="test_submission_2", expected_revision=1, selected_song_id="b"))
    uploads = []
    original = rig.index.upsert
    monkeypatch.setattr(rig.index, "upsert", lambda entries, **kw: (uploads.append(len(entries)), original(entries, **kw))[1])
    report = rig.indexer.run_once()
    assert report.indexed == 0 and report.skipped_stale == 1 and report.points == 0
    assert uploads == [], "올리기 직전 확인에서 걸러야 한다 — 올렸다가 지우는 것이 아니다"
    assert _doc(rig, receipt.search_id)["review"] is None


def test_completion_is_not_marked_for_a_case_that_expired(rig):
    """완료 표시의 조건에는 미만료가 들어 있다. 올리는 사이 만료된 사례를 완료로 적지 않는다."""
    receipt = _approved(rig)
    case = CaseVersion(receipt.search_id, "a", 1, 1)
    assert rig.store.mark_indexed(case, EMBEDDING_VERSION) is True
    rig.database["search_feedback"].update_one({"_id": receipt.search_id}, {"$set": {"index": None}})
    expire(rig.database, receipt.search_id)
    assert rig.store.mark_indexed(case, EMBEDDING_VERSION) is False
    assert rig.store.mark_index_failed(case, error="qdrant_error", attempts=1,
                                       next_attempt_at=datetime.now(timezone.utc)) is False
    assert rig.store.reset_index_pending(case) is False


def test_expiry_during_embedding_is_not_indexed(rig):
    receipt = _approved(rig)
    rig.embedder.before = lambda: expire(rig.database, receipt.search_id)
    report = rig.indexer.run_once()
    assert report.indexed == 0 and report.points == 0


def test_a_change_between_upload_and_completion_removes_the_uploaded_point(rig, monkeypatch):
    receipt = _approved(rig)
    original = rig.index.upsert

    def upload_then_user_withdraws(entries, **kwargs):
        original(entries, **kwargs)
        rig.store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                    action="withdraw", selected_song_id=None, reason=""))

    monkeypatch.setattr(rig.index, "upsert", upload_then_user_withdraws)
    report = rig.indexer.run_once()
    assert report.indexed == 0 and report.skipped_stale == 1
    assert _point(rig, receipt.search_id) is None, "완료로 표시하지 못한 판은 남기지 않는다"
    assert _doc(rig, receipt.search_id)["status"] == "withdrawn"


def test_a_late_job_for_an_old_version_cannot_mark_or_keep_it(rig, monkeypatch):
    """예전 판 작업이 최신 판을 덮어쓴 뒤에 끝난다 — 완료 표시는 실패하고, 최신 판이 다시 올라간다."""
    receipt = _approved(rig)
    # 다른 프로세스의 작업을 흉내 낸다 — 같은 DB·같은 컬렉션을 보는 별도 저장소 객체.
    late = FeedbackIndexer(store_module.FeedbackStore(rig.database), rig.index, KeywordEmbedder(), WorkerSettings(), rig.clock)
    original = rig.index.upsert
    state = {"interleaved": False}

    def old_job_is_overtaken(entries, **kwargs):
        if not state["interleaved"]:
            state["interleaved"] = True
            # 예전 판 작업이 올리기 직전: 운영자가 문장을 고치고(검토 판 2) 다른 작업이 그 판을 먼저 끝낸다.
            approve(rig.store, receipt.search_id, "피아노로 시작하는 노래", review_version=1)
            assert rig.indexer.run_once().indexed == 1
            assert _point(rig, receipt.search_id)["review_version"] == 2
        original(entries, **kwargs)          # 예전 판이 최신 판 포인트를 덮어쓴다

    monkeypatch.setattr(rig.index, "upsert", old_job_is_overtaken)
    stale_work = rig.store.find_index_work(EMBEDDING_VERSION, 10)
    report = indexer_module.CycleReport()
    # 올리기 직전 확인은 통과한 뒤에 추월당하는 순서를 만든다(확인 → 추월 → 적재).
    monkeypatch.setattr(late.store, "fetch_cases", lambda ids, **kw: {doc["_id"]: doc for doc in stale_work})
    late._ensure_collection()
    late._index_batch(stale_work, report)
    monkeypatch.undo()

    assert report.indexed == 0 and report.skipped_stale == 1, "예전 판 작업은 최신 판을 완료로 표시하지 못한다"
    saved = _doc(rig, receipt.search_id)
    assert saved["review_version"] == 2 and saved["index"]["review_version"] == 2
    assert _point(rig, receipt.search_id) is None, "덮어쓴 예전 판 포인트는 그 작업이 지운다"
    report = rig.indexer.run_once()          # 완료라고 적혀 있는데 포인트가 없다 → 색인 대기로 되돌린다
    assert report.repaired == 1
    assert rig.indexer.run_once().indexed == 1
    assert _point(rig, receipt.search_id)["review_version"] == 2


def test_a_stale_point_left_behind_is_removed_and_the_current_version_restored(rig):
    """지우지 못하고 남은 예전 판 포인트(작업이 죽은 경우)는 대조에서 지워지고 최신 판이 다시 올라간다."""
    receipt = _approved(rig)
    rig.indexer.run_once()
    saved = _doc(rig, receipt.search_id)
    approve(rig.store, receipt.search_id, "피아노로 시작하는 노래", review_version=1)
    rig.indexer.run_once()
    old = CaseVersion(receipt.search_id, "a", 1, 1)
    rig.index.upsert([(old, [1.0] + [0.0] * (TEST_DIM - 1), saved["expires_at"])], embedding_version=EMBEDDING_VERSION)
    report = rig.indexer.run_once()
    assert report.deleted_stale_points == 1 and report.repaired == 1 and report.indexed == 1
    assert _point(rig, receipt.search_id)["review_version"] == 2


def test_a_delete_for_an_old_version_does_not_remove_the_reapproved_point(rig):
    receipt = _approved(rig)
    rig.indexer.run_once()
    approve(rig.store, receipt.search_id, "다시 승인한 문장", review_version=1)
    rig.indexer.run_once()
    rig.index.delete_version(receipt.search_id, 1, 1)        # 늦게 도착한 예전 판 삭제
    assert _point(rig, receipt.search_id)["review_version"] == 2


def test_withdrawal_removes_the_point_and_a_new_selection_can_be_approved_again(rig):
    receipt = _approved(rig)
    rig.indexer.run_once()
    rig.store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                action="withdraw", selected_song_id=None, reason=""))
    report = rig.indexer.run_once()
    assert report.deleted_stale_points == 1 and report.points == 0
    rig.store.submit(submission(receipt, submission_id="test_submission_3", expected_revision=2,
                                selected_song_id="b", reason="다시 골랐어요"))
    approve(rig.store, receipt.search_id, "기타로 시작하는 노래", revision=3, review_version=1)
    assert rig.indexer.run_once().indexed == 1
    payload = _point(rig, receipt.search_id)
    assert (payload["song_id"], payload["source_revision"], payload["review_version"]) == ("b", 3, 2)


def test_approval_cancellation_and_user_edit_remove_the_point(rig):
    cancelled, edited = _approved(rig, query="승인 취소"), _approved(rig, query="사용자 수정")
    rig.indexer.run_once()
    rig.store.apply_review(cancelled.search_id, state="rejected", reusable_text="", note="취소",
                           expected_revision=1, expected_review_version=1, reviewer="운영자")
    rig.store.submit(submission(edited, submission_id="test_submission_2", expected_revision=1, selected_song_id="b"))
    report = rig.indexer.run_once()
    assert report.deleted_stale_points == 2 and report.points == 0


def test_expired_points_are_deleted_even_after_mongo_dropped_the_original(rig):
    """Mongo TTL 삭제는 Qdrant로 전파되지 않는다. 포인트에 실린 만료 시각만으로 지운다."""
    receipt = _approved(rig)
    rig.indexer.run_once()
    rig.database["search_feedback"].delete_many({"_id": receipt.search_id})   # TTL이 먼저 지웠다
    rig.database["search_runs"].delete_many({"_id": receipt.search_id})
    rig.clock.advance(days=31)
    report = rig.indexer.run_once()
    assert report.deleted_expired_points == 1 and report.points == 0 and report.expired_points_remaining == 0


def test_an_orphan_point_is_removed_before_its_expiry(rig):
    receipt = _approved(rig)
    rig.indexer.run_once()
    rig.database["search_feedback"].delete_many({"_id": receipt.search_id})
    report = rig.indexer.run_once()
    assert report.deleted_stale_points == 1 and report.points == 0


def test_points_are_kept_when_mongo_cannot_be_read(rig, monkeypatch):
    """'문서가 없다'와 '읽지 못했다'를 구분하지 못하는 주기에는 지우지 않는다."""
    receipt = _approved(rig)
    rig.indexer.run_once()

    def down(*args, **kwargs):
        raise RuntimeError("mongodb://user:secret@host unreachable")

    for name in ("fetch_cases", "find_index_work", "find_indexed", "count_expired", "write_ops_status", "read_ops_status"):
        monkeypatch.setattr(rig.store, name, down)
    report = rig.indexer.run_once()
    assert report.errors == [ERROR_STORE] and report.deleted_stale_points == 0
    assert _point(rig, receipt.search_id) is not None


def test_ttl_not_ready_is_reported_and_expired_documents_are_removed_directly(rig, monkeypatch):
    kept, _ = submit_case(rig.store, query="아직 유효")
    gone, _ = submit_case(rig.store, query="만료됨")
    expire(rig.database, gone.search_id)
    monkeypatch.setattr(rig.store, "ttl_status", lambda: {"search_runs": "ready", "search_feedback": "not_ttl"})
    report = rig.indexer.run_once()
    assert report.mongo_expired_deleted == 2 and report.expired_documents_remaining == 0
    assert _doc(rig, gone.search_id) is None and _doc(rig, kept.search_id) is not None
    assert rig.database["search_runs"].find_one({"_id": gone.search_id}) is None
    assert store_module.ttl_report()["search_feedback"] == "not_ttl" and not store_module.ttl_ready()
    assert rig.store.read_ops_status()["ttl"]["search_feedback"] == "not_ttl"


def test_ttl_ready_leaves_expired_documents_to_mongo(rig, monkeypatch):
    gone, _ = submit_case(rig.store, query="만료됨")
    monkeypatch.setattr(rig.store, "ttl_status", lambda: {"search_runs": "ready", "search_feedback": "ready"})
    monkeypatch.setattr(rig.store, "count_expired", lambda now=None: {"search_runs": 1, "search_feedback": 1})
    monkeypatch.setattr(rig.store, "delete_expired", lambda now=None: pytest.fail("TTL이 준비됐으면 직접 지우지 않는다"))
    report = rig.indexer.run_once()
    assert report.mongo_expired_deleted == 0 and report.expired_documents_remaining == 2


def test_ttl_status_reads_the_real_index_shape(store_backend):
    store, database = store_backend
    assert set(store.ttl_status().values()) == {"missing"}
    assert store.ensure_indexes() is True
    assert store.ttl_status() == {"search_runs": "ready", "search_feedback": "ready"}


def test_an_old_embedding_version_is_reindexed(rig):
    receipt = _approved(rig)
    rig.indexer.run_once()
    saved = _doc(rig, receipt.search_id)
    rig.database["search_feedback"].update_one(
        {"_id": receipt.search_id}, {"$set": {"index.embedding_version": "koe5:old"}})
    rig.index.upsert([(CaseVersion(receipt.search_id, "a", 1, 1), [0.0, 1.0] + [0.0] * (TEST_DIM - 2), saved["expires_at"])],
                     embedding_version="koe5:old")
    report = rig.indexer.run_once()
    assert report.indexed == 1 and _point(rig, receipt.search_id)["embedding_version"] == EMBEDDING_VERSION


def test_a_deleted_collection_is_recreated_and_refilled_from_mongo(rig):
    """사례 컬렉션은 검색용 사본이다. 통째로 사라져도 Mongo의 승인 기록에서 다시 채운다(서버 재시작 없이)."""
    receipts = [_approved(rig, query=f"복구 {index}") for index in range(3)]
    assert rig.indexer.run_once().indexed == 3
    rig.client.client.delete_collection(rig.index.collection)
    broken = rig.indexer.run_once()
    assert ERROR_QDRANT in broken.errors and broken.repaired == 3, "사라진 것을 알아차리고 색인 대기로 되돌린다"
    healed = rig.indexer.run_once()
    assert healed.ok and healed.indexed == 3 and healed.points == 3
    assert all(_point(rig, receipt.search_id) for receipt in receipts)


def test_status_summary_has_counts_and_keeps_the_last_success_time(rig, monkeypatch):
    receipt = _approved(rig)
    rig.indexer.run_once()
    first = rig.store.read_ops_status()
    assert first["last_error"] is None and first["cycle"]["indexed"] == 1 and first["cycle"]["points"] == 1
    assert first["embedding_version"] == EMBEDDING_VERSION
    assert "휘파람" not in str(first) and receipt.search_id not in str(first), "원문·사례 ID는 상태에 넣지 않는다"
    monkeypatch.setattr(rig.index, "delete_expired", lambda now: (_ for _ in ()).throw(RuntimeError("qdrant down")))
    rig.clock.advance(seconds=30)
    rig.indexer.run_once()
    second = rig.store.read_ops_status()
    assert second["last_error"] == ERROR_QDRANT and second["last_success_at"] == first["last_success_at"]
    assert second["last_cycle_at"] > first["last_cycle_at"]


def test_a_stop_signal_ends_the_cycle_before_the_next_step(rig):
    _approved(rig)
    stop = threading.Event()
    stop.set()
    report = rig.indexer.run_once(stop)
    assert report.indexed == 0 and rig.embedder.calls == []


def test_a_collection_with_other_vector_settings_is_refused(rig):
    FeedbackCaseIndex(rig.client, dim=TEST_DIM).ensure()
    with pytest.raises(RuntimeError, match="벡터 설정"):
        FeedbackCaseIndex(rig.client, dim=TEST_DIM * 2).ensure()


def test_non_finite_or_wrong_sized_vectors_are_refused(rig):
    rig.index.ensure()
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    case = CaseVersion("a" * 32, "a", 1, 1)
    for vector in ([float("nan")] * TEST_DIM, [float("inf")] + [0.0] * (TEST_DIM - 1), [1.0] * (TEST_DIM - 1)):
        with pytest.raises(ValueError):
            rig.index.upsert([(case, vector, expires)], embedding_version=EMBEDDING_VERSION)
    assert rig.index.count() == 0


# ---------------------------------------------------------------------------
# 작업 루프와 서버 종료 순서
# ---------------------------------------------------------------------------

class _Cycle:
    def __init__(self):
        self.started, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()
        self.calls = 0
        self.errors = []

    def run_once(self, stop):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        self.finished.set()
        return indexer_module.CycleReport(indexed=1, errors=list(self.errors))


@pytest.fixture
def loop_worker(monkeypatch):
    yield
    worker._stop.set()
    assert worker.stop_and_wait(5)
    worker._reset_state()


def test_the_worker_does_not_start_or_connect_when_disabled(monkeypatch, loop_worker):
    monkeypatch.delenv("FEEDBACK_INDEX_WORKER_ENABLED", raising=False)
    built = []
    assert worker.start_background(lambda: built.append(True)) is False
    assert built == [] and worker.status()["state"] == "disabled" and worker._thread is None


def test_the_worker_runs_at_startup_and_stop_waits_for_the_running_cycle(monkeypatch, loop_worker):
    monkeypatch.setenv("FEEDBACK_INDEX_WORKER_ENABLED", "1")
    cycle = _Cycle()
    assert worker.start_background(lambda: cycle, interval=3600) is True
    assert worker.start_background(lambda: cycle, interval=3600) is False, "두 번 띄우지 않는다"
    assert cycle.started.wait(5), "기다리지 않고 첫 주기를 바로 돌린다"
    assert worker.status()["state"] == "running"

    stopped = []
    waiter = threading.Thread(target=lambda: stopped.append(worker.stop_and_wait()))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive() and not cycle.finished.is_set(), "돌던 주기가 끝나기 전에는 돌아오지 않는다"
    cycle.release.set()
    waiter.join(5)
    assert stopped == [True] and cycle.finished.is_set() and cycle.calls == 1
    status = worker.status()
    assert status["state"] == "stopped" and status["cycles"] == 1 and status["last_error"] is None
    assert status["last_report"]["indexed"] == 1 and status["last_success_at"]


def test_the_worker_survives_a_failing_cycle_and_reports_a_code_only(monkeypatch, loop_worker, caplog):
    monkeypatch.setenv("FEEDBACK_INDEX_WORKER_ENABLED", "1")
    attempts = []

    def build():
        attempts.append(True)
        raise RuntimeError("mongodb://user:secret@host refused")

    assert worker.start_background(build, interval=0.01)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=5)
    while len(attempts) < 3 and datetime.now(timezone.utc) < deadline:
        threading.Event().wait(0.01)
    assert worker.stop_and_wait(5)
    assert len(attempts) >= 3, "실패해도 다음 주기에 다시 시도한다"
    status = worker.status()
    assert status["last_error"] == worker.ERROR_WORKER and status["last_success_at"] is None
    assert "secret" not in str(status) and "secret" not in caplog.text


def test_shutdown_stops_the_worker_before_closing_qdrant_and_mongo(monkeypatch):
    from src.backend import main

    order = []
    # Vertex 기동 검사(main.gc.verify_vertex_setup)는 tests/conftest.py가 모든 테스트에서 바꿔 끼운다.
    monkeypatch.setattr(main, "get_vector_client", lambda: None)
    monkeypatch.setattr(main.warmup, "start_background", lambda: None)
    monkeypatch.setattr(main.warmup, "stop_and_wait", lambda: order.append("warmup_stopped"))
    monkeypatch.setattr(main, "prepare_indexes", lambda: order.append("ttl_prepared"))
    monkeypatch.setattr(main.feedback_worker, "start_background", lambda: order.append("worker_started"))
    monkeypatch.setattr(main.feedback_worker, "stop_and_wait", lambda: order.append("worker_stopped"))
    monkeypatch.setattr(main, "close_vector_client", lambda: order.append("qdrant_closed"))
    monkeypatch.setattr(main, "close_feedback_store", lambda: order.append("mongo_closed"))

    async def run():
        async with main.lifespan(main.app):
            await asyncio.sleep(0.05)

    asyncio.run(run())
    assert order.index("worker_started") < order.index("worker_stopped") < order.index("qdrant_closed") < order.index("mongo_closed")
    assert order.index("warmup_stopped") < order.index("qdrant_closed")


@pytest.fixture
def ttl_state():
    """프로세스 전역의 TTL 보고를 시험 전후로 비운다."""
    def reset():
        with store_module._ttl_report_lock:
            store_module._ttl_report.clear()
            store_module._ttl_checked_at = None
    reset()
    yield
    reset()


def test_public_health_reports_feedback_state_without_internals(monkeypatch, loop_worker, ttl_state):
    from src.feedback.health import feedback_health

    for name in ("FEEDBACK_COLLECTION_ENABLED", "FEEDBACK_INDEX_WORKER_ENABLED", "FEEDBACK_RETRIEVAL_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(store_module, "get_feedback_store", lambda: pytest.fail("꺼져 있으면 저장소를 열지 않는다"))
    off = feedback_health()
    assert off["collection"] == {"enabled": False, "ready": False, "ttl": None, "ttl_checked_at": None}
    assert off["index_worker"]["enabled"] is False and off["retrieval"] == {"enabled": False}

    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    store_module._set_ttl_report({"search_runs": "ready", "search_feedback": "unauthorized"})
    on = feedback_health()
    assert on["collection"]["enabled"] is True and on["collection"]["ready"] is False, \
        "TTL이 준비되지 않았으면 수집 준비 완료가 아니다"
    assert on["collection"]["ttl"]["search_feedback"] == "unauthorized" and on["collection"]["ttl_checked_at"]
    store_module._set_ttl_report({"search_runs": "ready", "search_feedback": "ready"})
    assert feedback_health()["collection"]["ready"] is True


def test_the_ttl_report_follows_the_store_when_no_worker_is_running(monkeypatch, ttl_state):
    """리뷰 FR-04. 색인 작업을 켜지 않은 서버에서도 `/health`의 TTL 보고가 기동 시점 값에 굳지 않는다."""
    from src.feedback.health import feedback_health
    from tests.feedback_fakes import FakeDatabase

    database = FakeDatabase()
    store = store_module.FeedbackStore(database)
    monkeypatch.setattr(store_module, "get_feedback_store", lambda: store)
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    monkeypatch.delenv("FEEDBACK_INDEX_WORKER_ENABLED", raising=False)
    runs = database["search_runs"]
    create, information = runs.create_index, runs.index_information

    def unreachable(*args, **kwargs):
        raise TimeoutError("mongodb://user:secret@host unreachable")

    # 기동할 때 Mongo에 닿지 못했다.
    runs.create_index = runs.index_information = unreachable
    assert store_module.prepare_indexes() is False
    assert store_module.ttl_report()["search_runs"] == "unreachable" and feedback_health()["collection"]["ready"] is False

    # 연결이 돌아오고 검색 기록이 저장된다 — 그때 색인이 만들어지고 보고도 따라온다(재시작 없이).
    runs.create_index, runs.index_information = create, information
    snapshot = {"query": "합성 질의", "shown_ids": ["a"], "candidate_ids": ["a"]}
    store.record_search(dict(snapshot))
    health = feedback_health()["collection"]
    assert health["ready"] is True and health["ttl"] == {"search_runs": "ready", "search_feedback": "ready"}
    first_check = health["ttl_checked_at"]
    assert first_check

    # 준비된 뒤 색인이 사라졌다. 재확인 간격이 지나면 다시 만들고, 그 전에는 명령을 보내지 않는다.
    database["search_feedback"].index_info.clear()
    store.record_search(dict(snapshot))
    assert database["search_feedback"].index_info == {}, "간격 안에서는 다시 확인하지 않는다"
    store._verified_at -= store_module.TTL_RECHECK_SECONDS + 1
    store.record_search(dict(snapshot))
    assert "expires_at_1" in database["search_feedback"].index_info and store_module.ttl_ready()

    # 재확인이 실패하면 준비됐다고 남겨 두지 않는다. 기록은 계속된다.
    runs.create_index = unreachable
    store._verified_at -= store_module.TTL_RECHECK_SECONDS + 1
    assert store.record_search(dict(snapshot)).search_id in runs.documents
    assert store_module.ttl_report() == {"search_runs": "unknown", "search_feedback": "unknown"}
    assert feedback_health()["collection"]["ready"] is False
    runs.create_index = create
    store.record_search(dict(snapshot))
    assert store_module.ttl_ready(), "다음 기록에서 바로 다시 시도한다"

    # 기동할 때 읽은 구체적인 사유는 실패한 재시도가 지우지 않고, 다른 저장소 객체는 이 보고를 바꾸지 않는다.
    store_module._set_ttl_report({"search_runs": "ready", "search_feedback": "not_ttl"})
    store_module._note_ttl_verified(store, False)
    assert store_module.ttl_report() == {"search_runs": "unknown", "search_feedback": "not_ttl"}
    store_module.FeedbackStore(FakeDatabase()).ensure_indexes()
    assert store_module.ttl_report() == {"search_runs": "unknown", "search_feedback": "not_ttl"}


def test_a_worker_only_server_prepares_the_ttl_index(monkeypatch, ttl_state):
    """수집은 끄고 색인 작업만 켠 서버(쌓인 자료를 정리하는 중)도 기동할 때 TTL 색인을 만든다."""
    from tests.feedback_fakes import FakeDatabase

    database = FakeDatabase()
    store = store_module.FeedbackStore(database)
    monkeypatch.setattr(store_module, "get_feedback_store", lambda: store)
    monkeypatch.delenv("FEEDBACK_COLLECTION_ENABLED", raising=False)
    monkeypatch.delenv("FEEDBACK_INDEX_WORKER_ENABLED", raising=False)
    assert store_module.prepare_indexes() is False and database["search_runs"].index_info == {}, \
        "둘 다 꺼져 있으면 건드리지 않는다"
    monkeypatch.setenv("FEEDBACK_INDEX_WORKER_ENABLED", "1")
    assert store_module.prepare_indexes() is True
    assert database["search_runs"].index_info["expires_at_1"]["expireAfterSeconds"] == 0
    assert store_module.ttl_ready()
