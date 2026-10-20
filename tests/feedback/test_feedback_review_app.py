"""로컬 검토 앱(③-A) — 접근 제한·쓰기 보호·응답 내용. Mongo 대역만 쓴다."""
import json

import pytest
from fastapi.testclient import TestClient

from src.feedback import review_app
from src.feedback.review_service import ReviewService
from src.feedback.store import FeedbackStore
from tests.feedback_fakes import FakeDatabase, approve, expire, submission, submit_case

BASE = "http://127.0.0.1:8001"
TOKEN = "t" * 43
XSS = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


class Catalog:
    def __init__(self):
        self.songs = {"a": {"title": "현재 제목 A", "artist": "가수 A", "album": "앨범", "release_date": "2019-01-01",
                            "genre": "발라드"}}
        self.fail = False

    def lookup(self, ids):
        if self.fail:
            raise RuntimeError("mongodb://user:secret@host timed out")
        return {song_id: self.songs[song_id] for song_id in ids if song_id in self.songs}


@pytest.fixture
def env():
    database = FakeDatabase()
    store = FeedbackStore(database)
    catalog = Catalog()
    app = review_app.create_app(lambda: ReviewService(store, catalog, reviewer="정현(로컬)"), port=8001, csrf_token=TOKEN)
    with TestClient(app, base_url=BASE) as client:
        yield client, store, database, catalog


def _write_headers(**extra):
    return {"X-Review-Token": TOKEN, "Origin": BASE, "Sec-Fetch-Site": "same-origin", **extra}


def _review(client, search_id, **body):
    payload = {"state": "approved", "reusable_text": "휘파람 전주로 시작하는 노래", "note": "",
               "expected_revision": 1, "expected_review_version": 0, **body}
    return client.put(f"/api/feedback/{search_id}/review", json=payload, headers=_write_headers())


def test_the_public_search_server_does_not_expose_review_routes():
    from src.backend.main import app as public_app

    paths = set(public_app.openapi()["paths"])
    assert "/api/v1/feedback" in paths and "/api/v1/search" in paths, "공개 제출 API는 그대로 있다"
    assert not any("review" in path or path.startswith("/api/feedback") for path in paths)
    # `with` 없이 연다 — 기동 절차(벡터 DB 열기·모델 예열)를 돌리지 않고 경로만 본다.
    client = TestClient(public_app, raise_server_exceptions=False)
    for path in ["/api/feedback", "/api/session", "/api/status", "/api/v1/feedback/review", "/static/review.js"]:
        assert client.get(path).status_code == 404, path


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1:9000", "192.168.0.10:8001", "127.0.0.1.evil.example:8001", ""])
def test_requests_for_another_host_are_refused(env, host):
    """DNS rebinding — 바깥 도메인이 127.0.0.1로 풀려도 Host는 그 도메인이다."""
    client, *_ = env
    for path in ["/", "/api/session", "/api/feedback", "/static/review.js"]:
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 403 and response.json()["error"]["code"] == "forbidden_host"


@pytest.mark.parametrize("host", ["127.0.0.1:8001", "localhost:8001", "[::1]:8001"])
def test_loopback_names_on_the_configured_port_are_served(env, host):
    client, *_ = env
    assert client.get("/api/session", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("headers", [
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
    {"Origin": "https://evil.example"},
    {"Origin": "http://localhost:8001"},          # 같은 기계의 다른 이름도 다른 출처다
    {"Origin": "null"},
])
def test_cross_site_reads_are_refused(env, headers):
    client, store, *_ = env
    receipt, _ = submit_case(store)
    for path in ["/api/session", "/api/feedback", f"/api/feedback/{receipt.search_id}", "/api/status"]:
        response = client.get(path, headers=headers)
        assert response.status_code == 403, path
        assert "csrf_token" not in response.text and "휘파람" not in response.text


def test_writes_need_same_origin_the_page_token_and_a_json_body(env):
    client, store, database, _ = env
    receipt, _ = submit_case(store)
    url = f"/api/feedback/{receipt.search_id}/review"
    body = {"state": "approved", "reusable_text": "휘파람 전주", "expected_revision": 1, "expected_review_version": 0}
    refused = [
        ({"Origin": BASE}, {"json": body}, "bad_token"),                                    # 토큰 없음
        (_write_headers(**{"X-Review-Token": "wrong"}), {"json": body}, "bad_token"),
        (_write_headers(Origin="https://evil.example"), {"json": body}, "cross_origin"),
        (_write_headers(**{"Sec-Fetch-Site": "cross-site"}), {"json": body}, "cross_site"),
        # 폼으로 보내는 단순 요청(사전 요청 없이 나간다). 토큰이 맞아도 JSON이 아니면 받지 않는다.
        (_write_headers(**{"Content-Type": "text/plain"}), {"content": json.dumps(body)}, "json_only"),
        (_write_headers(**{"Content-Type": "application/x-www-form-urlencoded"}), {"content": "state=approved"}, "json_only"),
    ]
    for headers, kwargs, code in refused:
        response = client.put(url, headers=headers, **kwargs)
        assert response.status_code in {403, 415} and response.json()["error"]["code"] == code
        assert database["search_feedback"].documents[receipt.search_id]["review"] is None
    assert client.post(url, json=body, headers={"Origin": "https://evil.example"}).status_code == 403
    ok = client.put(url, json=body, headers=_write_headers())
    assert ok.status_code == 200 and ok.json()["review_state"] == "approved"


def test_preflight_gets_no_cors_permission(env):
    client, store, *_ = env
    receipt, _ = submit_case(store)
    response = client.options(f"/api/feedback/{receipt.search_id}/review", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "content-type,x-review-token"})
    assert response.status_code in {403, 405}
    assert not any(name.lower().startswith("access-control-") for name in response.headers)


def test_every_response_carries_the_security_headers(env):
    client, *_ = env
    for response in [client.get("/"), client.get("/static/review.js"), client.get("/api/session"),
                     client.get("/api/feedback/" + "0" * 32), client.get("/missing")]:
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "script-src 'self'" in response.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    page = client.get("/").text
    assert "<script src=\"/static/review.js\"" in page and "onclick" not in page and "<style" not in page


def test_list_omits_raw_text_and_detail_shows_the_case(env):
    client, store, database, _ = env
    receipt, _ = submit_case(store, query="휘파람으로 시작하는 노래", shown=("b", "a"))
    listing = client.get("/api/feedback?state=pending")
    assert listing.status_code == 200
    item = listing.json()["items"][0]
    assert item["selected_song"] == {"id": "a", "title": "곡 a", "artist": "가수 a", "rank": 2}
    assert item["has_reason"] is True and item["review_state"] == "pending" and item["remaining_seconds"] > 29 * 86400
    assert "휘파람" not in listing.text and "token" not in listing.text, "목록에는 원질의·사유 본문을 싣지 않는다"

    detail = client.get(f"/api/feedback/{receipt.search_id}").json()
    assert detail["query"] == "휘파람으로 시작하는 노래" and detail["reason"] == "도입부 휘파람"
    assert detail["current_song"] == {"id": "a", "found": True, **Catalog().songs["a"]}
    assert detail["context"]["answers"] == [] and detail["reviewable"] is True and detail["review"] is None
    assert "token_hash" not in json.dumps(detail)
    session = client.get("/api/session").json()
    assert session["csrf_token"] == TOKEN and session["reviewer_verified"] is False


def test_user_text_is_returned_as_json_text_never_as_markup(env):
    client, store, *_ = env
    receipt = store.record_search({"query": XSS, "shown_ids": ["a"], "candidate_ids": ["a"],
                                   "shown": [{"id": "a", "rank": 1, "title": XSS, "artist": XSS}]})
    store.submit(submission(receipt, reason=XSS))
    response = _review(client, receipt.search_id, reusable_text=XSS, note=XSS)
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    detail = response.json()
    assert detail["query"] == XSS and detail["reason"] == XSS and detail["reusable_text"] == XSS
    assert detail["selected_song"]["title"] == XSS and detail["review"]["note"] == XSS


def test_conflicts_return_the_latest_case_and_blocked_states_are_reported(env):
    client, store, database, _ = env
    receipt, _ = submit_case(store)
    store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1, selected_song_id="b"))
    stale = _review(client, receipt.search_id)
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_revision"
    assert stale.json()["current"]["revision"] == 2 and stale.json()["current"]["selected_song"]["id"] == "b"
    assert _review(client, receipt.search_id, expected_revision=2).status_code == 200
    concurrent = _review(client, receipt.search_id, expected_revision=2, reusable_text="다른 검토자의 문장")
    assert concurrent.status_code == 409 and concurrent.json()["error"]["code"] == "stale_review"
    assert concurrent.json()["current"]["review"]["state"] == "approved"

    store.submit(submission(receipt, submission_id="test_submission_3", expected_revision=2,
                            action="withdraw", selected_song_id=None, reason=""))
    withdrawn = _review(client, receipt.search_id, expected_revision=3, expected_review_version=1)
    assert withdrawn.status_code == 409 and withdrawn.json()["error"]["code"] == "withdrawn"
    assert withdrawn.json()["current"]["reviewable"] is False

    other, _ = submit_case(store, query="만료될 검색")
    expire(database, other.search_id)
    assert _review(client, other.search_id).status_code in {404, 410}
    assert client.get(f"/api/feedback/{other.search_id}").status_code == 404
    assert other.search_id not in client.get("/api/feedback?state=all").text


def test_validation_errors_use_one_error_shape(env):
    client, store, *_ = env
    receipt, _ = submit_case(store)
    cases = [
        _review(client, receipt.search_id, reusable_text="   "),
        _review(client, receipt.search_id, state="auto"),
        _review(client, receipt.search_id, expected_revision=0),
        _review(client, receipt.search_id, unexpected=True),
        client.put("/api/feedback/not-an-id/review", json={}, headers=_write_headers()),
        client.get("/api/feedback?state=nonsense"),
        client.get("/api/feedback?page=0"),
    ]
    for response in cases:
        assert response.status_code == 422, response.text
        assert set(response.json()["error"]) == {"code", "message"}


def test_storage_failures_do_not_leak_connection_details(env, caplog):
    client, store, _, catalog = env
    receipt, _ = submit_case(store)
    catalog.fail = True
    detail = client.get(f"/api/feedback/{receipt.search_id}")
    assert detail.status_code == 200 and detail.json()["current_song"] == {"id": "a", "found": None}, \
        "곡 정보를 못 읽어도 검토는 할 수 있다"

    def boom(*args, **kwargs):
        raise RuntimeError("mongodb://user:secret@host timed out")

    store.list_for_review = boom
    response = client.get("/api/feedback")
    assert response.status_code == 503 and response.json()["error"]["code"] == "storage_unavailable"
    assert "secret" not in response.text and "secret" not in caplog.text


def test_status_reports_ttl_readiness_and_index_counts(env):
    client, store, database, _ = env
    receipt, _ = submit_case(store)
    approve(store, receipt.search_id)
    status = client.get("/api/status").json()
    assert status["ttl_ready"] is True and status["ttl"] == {"search_runs": "ready", "search_feedback": "ready"}
    assert status["index"] == {"approved": 1, "indexed": 0, "failed": 0} and status["worker"] is None
    database["search_feedback"].index_info.clear()
    status = client.get("/api/status").json()
    assert status["ttl_ready"] is False and status["ttl"]["search_feedback"] == "missing"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.10", "::", "example.com"])
def test_the_review_app_refuses_to_bind_a_public_address(monkeypatch, host):
    """loopback 검사가 빠지면 이 시험은 **실패해야 한다** — 서버를 실제로 띄운 채 멈추면 안 된다(리뷰 FR-06)."""
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append((a, k)))
    monkeypatch.setenv("FEEDBACK_REVIEW_HOST", host)
    with pytest.raises(SystemExit) as error:
        review_app.main()
    assert "로컬 전용" in str(error.value) and started == [], "거부했으면 서버를 띄우지 않는다"


def test_the_review_app_binds_loopback_by_default(monkeypatch):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **k: started.append(k))
    monkeypatch.delenv("FEEDBACK_REVIEW_HOST", raising=False)
    monkeypatch.setenv("FEEDBACK_REVIEW_PORT", "8123")
    review_app.main()
    assert [(k["host"], k["port"]) for k in started] == [("127.0.0.1", 8123)]


def test_loopback_host_check():
    assert review_app.is_loopback_host("127.0.0.1") and review_app.is_loopback_host("::1")
    assert not review_app.is_loopback_host("192.168.0.2") and not review_app.is_loopback_host("example.com")
