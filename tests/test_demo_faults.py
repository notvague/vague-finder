"""실패 경로 모의 서버(`src/backend/demo_faults.py`) — 실물을 열지 않고, 화면과 같은 왕복이 돈다.

모의 서버는 개발 서버 옆에서 함께 띄운다. 로컬 Qdrant 폴더는 한 프로세스만 열 수 있어서, 라우트가
쓰는 의존성 하나라도 대체를 빠뜨리면 모의 서버가 폴더를 잡거나(개발 서버·적재가 막힌다) 500을 낸다.
질문 조회(/search/clarify)를 추가했을 때 실제로 그랬다. 여기서 실물이 열리는 순간 실패하게 고정한다.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.backend import demo_faults, demo_rehearsal
from src.backend.api import dependencies

QUERY = "앨범이 파란 잔잔한 노래"

# 모의 서버에서 만들어지면 안 되는 실물 — 벡터 DB, 임베딩 모델, 분석기, 리랭커, 검색기.
REAL_FACTORIES = [
    "get_qdrant_client",
    "KoE5Embedder", "SigLIP2Embedder", "CLAPAudioEmbedder", "BM25SparseEncoder",
    "QueryAnalyzer", "MusicReranker", "GeminiListwiseReranker", "SearchRouter", "SearchService",
]


@pytest.fixture()
def server(monkeypatch):
    opened: List[str] = []

    def forbid(name: str):
        def opener(*args: Any, **kw: Any):
            opened.append(name)
            raise AssertionError(f"모의 서버가 실물을 열었다: {name}")
        return opener

    for name in REAL_FACTORIES:
        monkeypatch.setattr(dependencies, name, forbid(name))
    demo_faults.fault.set("ok", None, "")
    with TestClient(demo_faults.create_app()) as client:
        yield client, opened
    demo_faults.fault.set("ok", None, "")


def search(client: TestClient, **body: Any) -> Dict[str, Any]:
    response = client.post("/api/v1/search", json={
        "query": QUERY, "top_k": 10, "explain": True, "defer_clarify": True, **body})
    assert response.status_code == 200, response.text
    return response.json()


def lookup(client: TestClient, reply: Dict[str, Any]):
    """화면이 "이 중에는 없어요"를 누를 때 보내는 질문 조회."""
    return client.post("/api/v1/search/clarify", json={
        "asked_slots": reply["asked_slots"],
        "previous_candidate_ids": reply["candidate_ids"][:30],
        "shown_ids": [r["id"] for r in reply["results"]],
        "rejected_ids": reply["rejected_ids"],
        "turn": reply["turn"],
    })


def test_every_real_backend_the_routes_depend_on_is_replaced(server):
    """검색 → 질문 조회 → 답 → 다시 검색. 화면의 왕복 전체가 실물 없이 돈다."""
    client, opened = server
    first = search(client)
    question = lookup(client, first)
    assert question.status_code == 200, question.text
    slot = question.json()["clarify"]["slot"]
    second = search(client, prior_analysis=first["analysis"], turn=first["turn"],
                    asked_slots=first["asked_slots"], answers=[{"slot": slot, "value": "", "skipped": True}],
                    previous_candidate_ids=first["candidate_ids"][:30],
                    rejected_ids=[r["id"] for r in first["results"]])
    assert second["turn"] == 2 and len(second["results"]) == 10
    assert lookup(client, second).status_code == 200
    assert client.get("/health").status_code == 200
    assert opened == []


def test_question_matches_the_candidates_the_mock_search_returned(server):
    """질문의 선택지 수는 모의 검색이 낸 후보의 메타데이터에서 나온다 — 화면의 목록과 어긋나지 않는다."""
    client, _ = server
    reply = search(client)
    assert reply["clarify"] is None and reply["clarify_deferred"] is True
    shown = {r["id"] for r in reply["results"]}
    by_id = {str(song["id"]): song for song in demo_faults._songs()}
    remaining = [demo_faults._track(by_id[i], 0.0) for i in reply["candidate_ids"] if i not in shown]
    assert len(remaining) == 20

    question = lookup(client, reply).json()["clarify"]
    assert question["slot"] == "vocal_gender"
    assert {o["value"]: o["count"] for o in question["options"]} == dict(Counter(t.vocal_gender for t in remaining))
    # 결과 목록에 실린 값과도 같은 규칙이다
    for track in reply["results"]:
        assert track["vocal_gender"] == demo_faults._metadata(by_id[track["id"]])["vocal_gender"]


@pytest.mark.parametrize("mode, search_ok", [("clarify_fail", True), ("server_error", False)])
def test_question_lookup_fault_modes(server, mode, search_ok):
    client, opened = server
    reply = search(client)
    demo_faults.fault.set(mode, None, None)
    failed = lookup(client, reply)
    assert failed.status_code == 500 and "질문" in failed.json()["detail"]
    again = client.post("/api/v1/search", json={"query": QUERY, "defer_clarify": True})
    assert (again.status_code == 200) is search_ok, "clarify_fail은 질문 조회만 고장 낸다"
    demo_faults.fault.set("ok", None, None)
    assert lookup(client, reply).json()["clarify"] is not None
    assert opened == []


def test_hanging_lookup_can_be_released(server):
    client, _ = server
    reply = search(client)
    demo_faults.fault.set("clarify_hang", 30.0, None)
    demo_faults.fault.release.set()      # 밖에서 풀어 주면 기다리지 않고 답한다
    assert lookup(client, reply).json()["clarify"] is not None


def test_rehearsal_walks_the_same_round_trip_as_the_screen(server, monkeypatch):
    """`demo_rehearsal`의 재질문 단계 — defer_clarify로 검색하고, /search/clarify로 질문을 받아 답한다."""
    client, opened = server
    sent: List[tuple] = []

    def post(url: str, body: Dict[str, Any], timeout: float, path: str = demo_rehearsal.SEARCH_PATH):
        sent.append((path, body))
        response = client.post(path, json=body)
        return 0.01, response.status_code, response.json()

    monkeypatch.setattr(demo_rehearsal, "_post", post)
    monkeypatch.setattr(demo_rehearsal, "_health", lambda url: client.get("/health").json())

    steps = demo_rehearsal.rehearse("http://mock", timeout=5)

    paths = [path for path, _ in sent]
    searches = len(demo_rehearsal.SEQUENCE)
    assert paths == [demo_rehearsal.SEARCH_PATH] * searches + [demo_rehearsal.CLARIFY_PATH, demo_rehearsal.SEARCH_PATH]
    assert all(body["defer_clarify"] is True for path, body in sent if path == demo_rehearsal.SEARCH_PATH)

    first = client.post(demo_rehearsal.SEARCH_PATH, json=sent[0][1]).json()
    shown = [r["id"] for r in first["results"]]
    assert sent[searches][1] == {
        "asked_slots": [], "previous_candidate_ids": first["candidate_ids"][:30],
        "shown_ids": shown, "rejected_ids": [], "turn": 1,
    }
    answer = sent[-1][1]
    assert answer["rejected_ids"] == shown and answer["turn"] == 1
    assert answer["answers"][0]["slot"] == "vocal_gender" and not answer["answers"][0]["skipped"]
    assert answer["prior_analysis"] == first["analysis"]

    last = steps[-1]
    assert last.name.startswith("4.") and last.status == 200 and last.results == 10
    assert last.problems == [], last.problems
    assert opened == []
