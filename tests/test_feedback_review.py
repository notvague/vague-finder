"""검토 계약(③-A) — 상태 전이·판 조건·사용자 제출과의 경합. 모델·Qdrant를 열지 않는다.

`store_backend`가 대역과 (FEEDBACK_TEST_MONGO_URI가 있으면) 실제 Mongo로 같은 시험을 돌린다.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src.feedback.review import (
    REVIEW_APPROVED,
    REVIEW_ON_HOLD,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    effective_review_state,
    point_uuid,
    reusable_case,
)
from src.feedback.settings import MAX_REUSABLE_TEXT, MAX_REVIEW_NOTE
from src.feedback.store import REVIEW_HISTORY_LIMIT, FeedbackError, ReviewError
from tests.feedback_fakes import approve, expire, record_run, store_backend, submission, submit_case  # noqa: F401


def _now():
    return datetime.now(timezone.utc)


def _doc(database, search_id):
    return database["search_feedback"].find_one({"_id": search_id})


def test_a_submission_without_review_reads_as_pending_and_can_be_approved(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    saved = _doc(database, receipt.search_id)
    assert saved["review"] is None and "review_version" not in saved
    assert effective_review_state(saved) == REVIEW_PENDING and reusable_case(saved, _now()) is None

    approved = approve(store, receipt.search_id, "  휘파람   전주로\n시작하는 노래 ")
    assert approved["reusable_text"] == "휘파람 전주로 시작하는 노래", "공백을 정리해 저장한다"
    assert approved["review_version"] == 1 and approved["review"]["source_revision"] == 1
    assert approved["review"]["reviewed_by"] == "시험 운영자"
    assert approved["index"]["state"] == "pending", "승인과 색인 예약은 같은 쓰기다"
    assert approved["status"] == "pending", "사용자 응답의 상태에는 검토 결과를 섞지 않는다"
    case = reusable_case(_doc(database, receipt.search_id), _now())
    assert (case.song_id, case.source_revision, case.review_version) == ("a", 1, 1)


def test_documents_written_before_the_review_contract_are_pending(store_backend):
    """#47 때 쌓인 자료는 `review=null`이고 검토 판 필드가 없다. 모양이 다른 값도 승인으로 읽지 않는다."""
    store, database = store_backend
    receipt, _ = submit_case(store)
    database["search_feedback"].update_one({"_id": receipt.search_id}, {"$set": {"review": {"approved": True}}})
    saved = _doc(database, receipt.search_id)
    assert effective_review_state(saved) == REVIEW_PENDING and reusable_case(saved, _now()) is None
    listed, total = store.list_for_review(REVIEW_PENDING, page=1, page_size=10)
    assert total == 1 and listed[0]["_id"] == receipt.search_id
    assert approve(store, receipt.search_id)["review"]["state"] == REVIEW_APPROVED


def test_hold_reject_and_rereview_bump_the_review_version_each_time(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    sid = receipt.search_id
    held = store.apply_review(sid, state=REVIEW_ON_HOLD, reusable_text="", note="가수 확인 필요",
                              expected_revision=1, expected_review_version=0, reviewer="운영자")
    assert held["review"]["state"] == REVIEW_ON_HOLD and held["index"] is None and held["reusable_text"] is None
    approved = approve(store, sid, review_version=1)
    assert approved["review_version"] == 2
    edited = approve(store, sid, "문장만 고친 승인", review_version=2)
    assert edited["review_version"] == 3 and edited["index"]["review_version"] == 3, "문장만 고쳐도 판이 오른다"
    rejected = store.apply_review(sid, state=REVIEW_REJECTED, reusable_text="문장만 고친 승인", note="개인 기억뿐",
                                  expected_revision=1, expected_review_version=3, reviewer="운영자")
    assert rejected["review_version"] == 4 and rejected["index"] is None, "승인 취소도 판을 올리고 색인 예약을 지운다"
    assert reusable_case(_doc(database, sid), _now()) is None
    back = store.apply_review(sid, state=REVIEW_PENDING, reusable_text="", note="", expected_revision=1,
                              expected_review_version=4, reviewer="운영자")
    assert effective_review_state(back) == REVIEW_PENDING
    assert [event["state"] for event in back["review_history"]] == [
        REVIEW_ON_HOLD, REVIEW_APPROVED, REVIEW_APPROVED, REVIEW_REJECTED, REVIEW_PENDING]
    assert all("reusable_text" not in event and "note" not in event for event in back["review_history"])


@pytest.mark.parametrize("fields, code", [
    ({"state": "approved", "reusable_text": "   \n "}, "text_required"),
    ({"state": "approved", "reusable_text": "가" * (MAX_REUSABLE_TEXT + 1)}, "text_too_long"),
    ({"state": "on_hold", "note": "가" * (MAX_REVIEW_NOTE + 1)}, "note_too_long"),
    ({"state": "auto_approved"}, "invalid_state"),
])
def test_invalid_reviews_are_rejected_before_any_write(store_backend, fields, code):
    store, database = store_backend
    receipt, _ = submit_case(store)
    args = {"state": "on_hold", "reusable_text": "", "note": "", **fields}
    with pytest.raises(ReviewError) as error:
        store.apply_review(receipt.search_id, expected_revision=1, expected_review_version=0, reviewer="운영자", **args)
    assert (error.value.status, error.value.code) == (422, code)
    assert _doc(database, receipt.search_id)["review"] is None


def test_a_stale_screen_cannot_approve_a_selection_the_user_changed(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1, selected_song_id="b"))
    with pytest.raises(ReviewError) as error:
        approve(store, receipt.search_id)                       # 화면은 revision 1(곡 a)을 보고 있었다
    assert (error.value.status, error.value.code) == (409, "stale_revision")
    assert error.value.current["selected_song_id"] == "b" and error.value.current["revision"] == 2
    assert _doc(database, receipt.search_id)["review"] is None


def test_user_edit_after_approval_invalidates_it_but_keeps_the_review_version(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    approve(store, receipt.search_id)
    store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1, selected_song_id="b"))
    saved = _doc(database, receipt.search_id)
    assert saved["review"] is None and saved["reusable_text"] is None and saved["index"] is None
    assert saved["review_version"] == 1, "판을 되돌리면 같은 번호가 다른 문장을 가리킨다"
    assert reusable_case(saved, _now()) is None
    again = approve(store, receipt.search_id, "곡 b의 단서", revision=2, review_version=1)
    assert again["review_version"] == 2 and reusable_case(again, _now()).song_id == "b"


def test_withdrawn_and_expired_cases_cannot_be_approved(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    approve(store, receipt.search_id)
    store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                            action="withdraw", selected_song_id=None, reason=""))
    assert reusable_case(_doc(database, receipt.search_id), _now()) is None, "철회는 검토 상태와 무관하게 재사용을 막는다"
    with pytest.raises(ReviewError) as error:
        approve(store, receipt.search_id, revision=2, review_version=1)
    assert (error.value.status, error.value.code) == (409, "withdrawn")

    other, _ = submit_case(store, query="다른 검색")
    expire(database, other.search_id)
    with pytest.raises(ReviewError) as error:
        approve(store, other.search_id)
    assert (error.value.status, error.value.code) == (410, "expired")
    assert store.get_for_review(other.search_id) is None, "만료된 사례는 지워지기 전이라도 보여주지 않는다"
    with pytest.raises(ReviewError) as error:
        approve(store, "f" * 32)
    assert (error.value.status, error.value.code) == (404, "not_found")


def test_approval_does_not_extend_the_retention_period(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    before = _doc(database, receipt.search_id)["expires_at"]
    approve(store, receipt.search_id)
    approve(store, receipt.search_id, "문장 수정", review_version=1)
    assert _doc(database, receipt.search_id)["expires_at"] == before
    assert database["search_runs"].find_one({"_id": receipt.search_id})["expires_at"] == before


def test_an_approved_case_stops_being_reusable_at_its_expiry_time(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    saved = approve(store, receipt.search_id)
    expires = saved["expires_at"]
    assert reusable_case(saved, expires - timedelta(seconds=1)) is not None
    assert reusable_case(saved, expires) is None, "만료 시각부터 쓰지 않는다(TTL 삭제를 기다리지 않는다)"
    assert reusable_case({**saved, "expires_at": None}, _now()) is None, "기한을 모르는 자료는 쓰지 않는다"


def test_concurrent_reviews_of_the_same_version_let_exactly_one_through(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)

    def review(state):
        try:
            return store.apply_review(receipt.search_id, state=state, reusable_text="동시 검토", note=state,
                                      expected_revision=1, expected_review_version=0, reviewer=state)
        except ReviewError as exc:
            return exc

    for _ in range(5):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(review, [REVIEW_APPROVED, REVIEW_REJECTED]))
        conflicts = [r for r in results if isinstance(r, ReviewError)]
        saved = _doc(database, receipt.search_id)
        assert len(conflicts) == 1 and conflicts[0].code == "stale_review"
        assert conflicts[0].current["review"]["state"] == saved["review"]["state"], "충돌 응답은 최신 문서를 싣는다"
        assert saved["review_version"] == 1
        database["search_feedback"].update_one(
            {"_id": receipt.search_id}, {"$set": {"review": None, "review_version": 0, "index": None}})


def test_user_edit_racing_an_approval_never_leaves_the_old_selection_approved(store_backend):
    """어느 쪽이 먼저 들어가든 결과는 하나다 — 새 선택(revision 2)이 검토 없이 남는다."""
    store, database = store_backend
    for round_number in range(8):
        receipt, _ = submit_case(store, query=f"경합 {round_number}")

        def edit():
            return store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                           selected_song_id="b", reason="다른 곡이었어요"))

        def review():
            try:
                return approve(store, receipt.search_id)
            except ReviewError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(edit), pool.submit(review)] if round_number % 2 else [pool.submit(review), pool.submit(edit)]
            outcomes = [future.result(10) for future in futures]
        saved = _doc(database, receipt.search_id)
        assert saved["revision"] == 2 and saved["selected_song_id"] == "b"
        assert saved["review"] is None and saved["reusable_text"] is None
        assert reusable_case(saved, _now()) is None
        errors = [o for o in outcomes if isinstance(o, ReviewError)]
        assert all(error.code == "stale_revision" for error in errors)


def test_withdrawal_racing_an_approval_never_leaves_a_reusable_case(store_backend):
    store, database = store_backend
    for round_number in range(6):
        receipt, _ = submit_case(store, query=f"철회 경합 {round_number}")

        def withdraw():
            return store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                           action="withdraw", selected_song_id=None, reason=""))

        def review():
            try:
                return approve(store, receipt.search_id)
            except ReviewError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(withdraw), pool.submit(review)]:
                future.result(10)
        saved = _doc(database, receipt.search_id)
        assert saved["status"] == "withdrawn" and reusable_case(saved, _now()) is None


def test_a_retry_after_a_lost_response_is_applied_once(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    first = approve(store, receipt.search_id, note="확인함")
    replay = approve(store, receipt.search_id, note="확인함")     # 응답을 못 받아 같은 값으로 다시 보냈다
    assert replay["review_version"] == first["review_version"] == 1
    assert len(_doc(database, receipt.search_id)["review_history"]) == 1
    with pytest.raises(ReviewError) as error:                    # 내용이 다르면 재시도가 아니라 충돌이다
        approve(store, receipt.search_id, "다른 문장", note="확인함")
    assert error.value.code == "stale_review"


def test_review_history_is_bounded(store_backend):
    store, database = store_backend
    receipt, _ = submit_case(store)
    for version in range(REVIEW_HISTORY_LIMIT + 3):
        approve(store, receipt.search_id, f"문장 {version}", review_version=version)
    history = _doc(database, receipt.search_id)["review_history"]
    assert len(history) == REVIEW_HISTORY_LIMIT and history[-1]["version"] == REVIEW_HISTORY_LIMIT + 3


def test_review_lists_filter_by_state_page_and_hide_expired(store_backend):
    store, database = store_backend
    receipts = [submit_case(store, query=f"목록 {index}")[0] for index in range(5)]
    approve(store, receipts[0].search_id)
    store.apply_review(receipts[1].search_id, state=REVIEW_ON_HOLD, reusable_text="", note="", expected_revision=1,
                       expected_review_version=0, reviewer="운영자")
    store.submit(submission(receipts[2], submission_id="test_submission_2", expected_revision=1,
                            action="withdraw", selected_song_id=None, reason=""))
    expire(database, receipts[3].search_id)

    def ids(state, **kw):
        rows, total = store.list_for_review(state, page=kw.get("page", 1), page_size=kw.get("page_size", 10))
        return [row["_id"] for row in rows], total

    assert ids(REVIEW_APPROVED) == ([receipts[0].search_id], 1)
    assert ids(REVIEW_ON_HOLD) == ([receipts[1].search_id], 1)
    assert ids("withdrawn") == ([receipts[2].search_id], 1)
    assert ids(REVIEW_PENDING) == ([receipts[4].search_id], 1)
    everything, total = ids("all")
    assert total == 4 and receipts[3].search_id not in everything, "만료된 문서는 어느 목록에도 없다"
    first, _ = ids("all", page=1, page_size=3)
    second, _ = ids("all", page=2, page_size=3)
    assert len(first) == 3 and len(second) == 1 and not set(first) & set(second)
    rows, _ = store.list_for_review("all", page=1, page_size=10)
    assert all("history" not in row and "review_history" not in row for row in rows)
    with pytest.raises(ValueError):
        store.list_for_review("unknown", page=1, page_size=10)


def test_review_reads_never_return_the_submission_token_hash(store_backend):
    store, _ = store_backend
    receipt, _ = submit_case(store)
    feedback, run = store.get_for_review(receipt.search_id)
    assert "token_hash" not in run and run["query"] == "휘파람으로 시작하는 노래"
    assert "token_hash" not in store.runs_by_id([receipt.search_id])[receipt.search_id]
    assert feedback["reason"] == "도입부 휘파람"


def test_public_submit_contract_is_unchanged_by_review(store_backend):
    """공개 응답의 status는 pending/withdrawn뿐이고, 검토 뒤에도 수정·철회 흐름이 그대로다."""
    store, _ = store_backend
    receipt, first = submit_case(store)
    approve(store, receipt.search_id)
    assert store.submit(submission(receipt)) == first, "같은 제출의 재시도는 검토와 무관하게 같은 응답이다"
    assert set(first.model_dump()) == {"search_id", "revision", "status", "selected_song_id", "reason"}
    with pytest.raises(FeedbackError) as error:
        store.submit(submission(receipt, submission_id="test_submission_2", selected_song_id="b"))
    assert error.value.status == 409 and error.value.current.status == "pending"


def test_judgement_holds_for_documents_the_store_would_never_write():
    """저장소를 거치면 생기지 않는 모양(수동 수정·복구한 백업). 판정 함수가 그대로 믿지 않는다."""
    expires = _now() + timedelta(days=1)
    approved = {"_id": "a" * 32, "status": "pending", "revision": 2, "review_version": 3, "selected_song_id": "s",
                "reusable_text": "단서", "expires_at": expires,
                "review": {"state": "approved", "source_revision": 2, "version": 3}}
    assert reusable_case(approved, _now()) is not None
    stale_revision = {**approved, "review": {**approved["review"], "source_revision": 1}}     # 예전 선택을 본 검토
    stale_version = {**approved, "review": {**approved["review"], "version": 2}}              # 검토 판이 어긋남
    for document in (stale_revision, stale_version):
        assert effective_review_state(document) == REVIEW_PENDING and reusable_case(document, _now()) is None
    for broken in ({**approved, "status": "withdrawn"}, {**approved, "selected_song_id": None},
                   {**approved, "reusable_text": "  "}, {**approved, "revision": "2"}, {**approved, "expires_at": "2099"}):
        assert reusable_case(broken, _now()) is None
    assert reusable_case(None, _now()) is None and reusable_case({}, _now()) is None


def test_point_ids_are_deterministic_uuids():
    import uuid

    first, second = point_uuid("a" * 32), point_uuid("b" * 32)
    assert first == point_uuid("a" * 32) and first != second
    assert str(uuid.UUID(first)) == first, "Qdrant가 받는 UUID 문자열이어야 한다"
