"""지연 조회 — 데이터 질문은 "이 중에는 없어요"를 누를 때 /search/clarify가 계산한다.

고정하는 것:

1. **/search는 데이터 질문을 계산하지 않는다**(defer_clarify). 정답을 찾아 멈추면 그 계산은 일어나지 않는다.
2. **연령대는 후보를 통해 반영된다.** 질문 조회는 출생 연도를 받지 않는다 — 연도로 다시 찾은 결과의
   후보(candidate_ids)를 받아 그 후보가 어떻게 갈리는지로 질문을 만든다.
3. **질문 조회는 상태를 바꾸지 않는다.** 검색·분석을 다시 하지 않고, 턴·거절 목록도 그대로다.
4. **/search의 clarify_deferred와 /search/clarify의 판정이 같다**(슬롯·턴·거절 한도, 남은 후보).
5. 잘못된 입력은 500이 아니라 422이거나, 없는 곡으로 조용히 빠진다.
"""
from importlib import import_module
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import get_query_analyzer, get_search_router, get_vector_client
from src.backend.schemas.search import MAX_REJECTED_IDS, MatchingTrack
from src.retrieval.clarify import ALLOWED_SLOTS, QUESTION_FIELDS, slot_value
from src.retrieval.query_analyzer import _fallback
from src.retrieval.search_service import SearchService

routes = import_module("src.backend.api.routes.search")
QUERY = "중학교 때 듣던 노래"

# 연령대(시기 창) 없이 찾은 후보 — 전부 남성이라 성별로는 못 가르고 장르로 갈린다.
BASE_POOL = [
    ("100", "남성", "발라드"), ("101", "남성", "댄스"), ("102", "남성", "발라드"),
    ("103", "남성", "댄스"), ("104", "남성", "발라드"), ("105", "남성", "댄스"),
]
# 연령대를 넣어 다시 찾은 후보 — 다른 곡들이고 성별로 갈린다.
ERA_POOL = [
    ("200", "여성", "발라드"), ("201", "남성", "발라드"), ("202", "여성", "발라드"),
    ("203", "남성", "댄스"), ("204", "여성", "댄스"), ("205", "남성", "발라드"),
]
METADATA: Dict[str, Dict[str, Any]] = {
    sid: {"title": f"곡{sid}", "vocal_gender": gender, "genre": genre, "lyrics_summary": "질문과 무관한 긴 칸"}
    for sid, gender, genre in BASE_POOL + ERA_POOL
}


def _track(sid: str) -> MatchingTrack:
    return MatchingTrack(id=sid, score=1.0, **{k: v for k, v in METADATA[sid].items() if k != "lyrics_summary"})


class AgeAwareRouter:
    """시기 창이 들어오면 다른 후보를 낸다 — 실제 검색에서 연령대 가산이 후보를 바꾸는 것의 자리."""

    def __init__(self) -> None:
        self.seen: List[Any] = []
        self.pool_size: Optional[int] = None

    async def search(self, analysis, top_k=10, exclude_ids=None, candidate_ids_out=None,
                     candidate_tracks_out=None, **kw) -> List[MatchingTrack]:
        self.seen.append(analysis)
        excluded = set(exclude_ids or [])
        ids = [sid for sid, _, _ in (ERA_POOL if analysis.has_release_era else BASE_POOL) if sid not in excluded]
        tracks = [_track(sid) for sid in ids[: self.pool_size]]
        if candidate_ids_out is not None:
            candidate_ids_out.extend(t.id for t in tracks)
        if candidate_tracks_out is not None:
            candidate_tracks_out.extend(tracks)
        return tracks[:top_k]


class CandidateIndex:
    """`QdrantIndex.fetch` 자리. 무엇을 읽으라고 했는지 남긴다."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.fail = False

    def fetch(self, ids, namespace=None, fields=None) -> Dict[str, Any]:
        self.calls.append((list(ids), tuple(fields or ())))
        if self.fail:
            raise RuntimeError("DB unavailable")
        return {"matches": [
            {"id": sid, "score": 0.0,
             "metadata": {k: v for k, v in METADATA[sid].items() if not fields or k in fields}}
            for sid in ids if sid in METADATA
        ]}


@pytest.fixture()
def setup():
    app = FastAPI()
    app.include_router(routes.router)
    index, router = CandidateIndex(), AgeAwareRouter()
    app.dependency_overrides[get_vector_client] = lambda: type("V", (), {"Index": lambda self, name: index})()
    app.dependency_overrides[get_search_router] = lambda: router
    app.dependency_overrides[get_query_analyzer] = lambda: type("A", (), {"analyze": staticmethod(_fallback)})()
    with TestClient(app) as client:
        yield client, index, router, app


def search(client: TestClient, **body: Any) -> Dict[str, Any]:
    response = client.post("/search", json={"query": QUERY, "defer_clarify": True, "top_k": 2, **body})
    assert response.status_code == 200, response.text
    return response.json()


def lookup_body(reply: Dict[str, Any]) -> Dict[str, Any]:
    """화면(map.js requestClarification)이 보내는 것 — 직전 검색 응답의 상태를 그대로 돌려준다."""
    return {
        "asked_slots": reply["asked_slots"],
        "previous_candidate_ids": reply["candidate_ids"][:30],
        "shown_ids": [r["id"] for r in reply["results"]],
        "rejected_ids": reply["rejected_ids"],
        "turn": reply["turn"],
    }


def lookup(client: TestClient, reply: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    response = client.post("/search/clarify", json=lookup_body(reply))
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"clarify"}, "질문 조회는 질문만 돌려준다 — 턴·거절 상태가 없다"
    return response.json()["clarify"]


# --- 1. /search는 데이터 질문을 계산하지 않는다 -------------------------------------

def test_search_never_computes_a_data_question_when_deferred(setup, monkeypatch):
    client, index, _, _ = setup

    def forbidden(*args, **kw):
        pytest.fail("정답을 찾아 멈추면 데이터 질문은 계산되지 않아야 한다")

    monkeypatch.setattr(routes, "pick_question", forbidden)
    monkeypatch.setattr(routes, "pick_data_question", forbidden)

    first = search(client)
    assert first["clarify"]["slot"] == "birth_year", "출생 연도만 결과와 함께 묻는다"
    second = search(client, prior_analysis=first["analysis"], birth_year=1998)
    assert second["clarify"] is None and second["clarify_deferred"] and second["turn"] == 1
    third = search(client, prior_analysis=second["analysis"], birth_year=1998, turn=1,
                   rejected_ids=[r["id"] for r in second["results"]],
                   answers=[{"slot": "vocal_gender", "value": "남성"}])
    assert third["clarify"] is None and third["clarify_deferred"] and third["turn"] == 2
    assert index.calls == [], "검색은 질문용 후보 메타데이터를 읽지 않는다"


# --- 2. 연령대는 후보를 통해 반영된다 ---------------------------------------------

def test_question_is_built_from_the_candidates_refreshed_by_the_birth_year(setup):
    client, index, router, app = setup
    first = search(client)
    refreshed = search(client, prior_analysis=first["analysis"], birth_year=1998)
    era = router.seen[-1].release_era
    assert (era.start_year, era.end_year) == (2010, 2014), "연도는 검색에서 시기 창이 된다"
    assert refreshed["analysis"]["release_era"]["start_year"] is None, "응답의 분석에는 연도가 굳어 들어가지 않는다"
    assert refreshed["candidate_ids"] == [sid for sid, _, _ in ERA_POOL]
    assert refreshed["candidate_ids"] != first["candidate_ids"], "연령대를 넣으면 후보가 달라진다(이 시험의 전제)"

    def forbidden():
        pytest.fail("질문 조회는 검색기·분석기를 만들지 않는다")

    app.dependency_overrides[get_search_router] = forbidden
    app.dependency_overrides[get_query_analyzer] = forbidden
    searches = len(router.seen)

    body = lookup_body(refreshed)
    assert "birth_year" not in body and "prior_analysis" not in body, "연도·분석을 보내지 않는다"
    question = lookup(client, refreshed)
    # 연령대 반영 뒤의 후보(200번대)에서 보여준 곡을 뺀 나머지 — 성별로 갈린다
    assert index.calls[-1] == (["202", "203", "204", "205"], tuple(QUESTION_FIELDS))
    assert question["slot"] == "vocal_gender"
    assert {o["value"]: o["count"] for o in question["options"]} == {"여성": 2, "남성": 2}

    # 대조: 연령대를 넣기 전의 후보(100번대)였다면 전부 남성이라 장르를 물었을 것이다
    stale = lookup(client, first | {"asked_slots": ["birth_year"]})
    assert index.calls[-1][0] == ["102", "103", "104", "105"]
    assert stale["slot"] == "genre"
    assert len(router.seen) == searches, "질문 조회는 검색을 다시 하지 않는다"


def test_question_fields_cover_everything_the_slots_read():
    """질문 조회는 QUESTION_FIELDS만 읽는다. 슬롯이 다른 칸을 읽기 시작하면 여기서 걸린다."""
    for sid, metadata in METADATA.items():
        full = SearchService.track_from_match({"id": sid, "score": 0.0, "metadata": metadata})
        narrow = SearchService.track_from_match(
            {"id": sid, "score": 0.0, "metadata": {k: metadata[k] for k in QUESTION_FIELDS}})
        for slot in ALLOWED_SLOTS:
            assert slot_value(narrow, slot) == slot_value(full, slot) is not None


# --- 3. 질문 조회는 상태를 바꾸지 않는다 -------------------------------------------

def test_lookup_can_be_repeated_without_consuming_a_turn_or_rejections(setup):
    client, index, router, _ = setup
    reply = search(client, birth_year=1998)
    searches = len(router.seen)
    answers = [lookup(client, reply) for _ in range(2)]  # 질문을 닫고 다시 눌러도 같은 질문이다
    assert answers[0] == answers[1] and answers[0]["slot"] == "vocal_gender"
    assert index.calls[0] == index.calls[1]
    assert len(router.seen) == searches
    # 다음 검색은 조회 전과 같은 상태에서 이어진다 — 턴은 거절할 때 한 번만 간다
    after = search(client, prior_analysis=reply["analysis"], birth_year=1998, turn=reply["turn"],
                   rejected_ids=[r["id"] for r in reply["results"]])
    assert after["turn"] == 2 and after["rejected_ids"] == ["200", "201"]


def test_lookup_failure_is_reported_and_can_be_retried(setup):
    client, index, _, _ = setup
    reply = search(client, birth_year=1998)
    index.fail = True
    failed = client.post("/search/clarify", json=lookup_body(reply))
    assert failed.status_code == 500 and "질문" in failed.json()["detail"]
    index.fail = False
    assert lookup(client, reply)["slot"] == "vocal_gender"


# --- 4. /search의 clarify_deferred와 /search/clarify의 판정이 같다 -------------------

def _ids(n: int) -> List[str]:
    return [f"r{i}" for i in range(n)]


@pytest.mark.parametrize("label, request_state, pool_size, expected", [
    ("첫 검색", {}, None, True),
    ("연도를 건너뜀", {"asked_slots": ["birth_year"]}, None, True),
    ("연도를 건너뛰고 성별까지 답함 — 장르가 남았다",
     {"asked_slots": ["birth_year", "vocal_gender"], "rejected_ids": _ids(2), "turn": 1}, None, True),
    ("데이터 슬롯이 아닌 것이 둘 — 슬롯 수는 남았고 데이터 슬롯도 남았다",
     {"asked_slots": ["birth_year", "type"]}, None, True),
    ("데이터 슬롯을 다 물었다", {"asked_slots": ["vocal_gender", "genre"]}, None, False),
    ("슬롯 수 한도", {"asked_slots": ["birth_year", "type", "vocal_gender"]}, None, False),
    ("턴 한도", {"rejected_ids": _ids(2), "turn": 2}, None, False),
    ("다음 거절을 담을 자리가 딱 남았다",
     {"rejected_ids": _ids(MAX_REJECTED_IDS - 2), "turn": 1}, None, True),
    ("다음 거절을 담을 자리가 없다",
     {"rejected_ids": _ids(MAX_REJECTED_IDS - 1), "turn": 1}, None, False),
    ("보여준 곡 말고 남은 후보가 없다", {}, 2, False),
])
def test_deferred_flag_and_lookup_agree(setup, label, request_state, pool_size, expected):
    client, index, router, _ = setup
    router.pool_size = pool_size
    if request_state:
        request_state = {"prior_analysis": _fallback(QUERY).model_dump(), **request_state}
    reply = search(client, birth_year=1998, **request_state)
    assert reply["clarify_deferred"] is expected, label

    question = lookup(client, reply)
    assert (question is not None) is expected, f"{label}: 조회하라고 했으면 질문이 나오고, 아니면 나오지 않는다"
    if not expected:
        assert index.calls == [], f"{label}: 물을 수 없는 상태에서는 DB를 읽지 않는다"


# --- 5. 입력 검증 ------------------------------------------------------------------

@pytest.mark.parametrize("patch", [
    {"previous_candidate_ids": [""]},
    {"shown_ids": ["x" * 65]},
    {"rejected_ids": [123456789012345678901234567890]},   # 문자열이 아니다
    {"previous_candidate_ids": [str(i) for i in range(31)]},
    {"asked_slots": ["a", "b", "c", "d"]},
    {"turn": 0},
    {"query": QUERY},            # 검색 필드는 받지 않는다 — 실어 보내고 반영됐다고 믿지 않게
    {"top_k": 5},
    {"birth_year": 1998},
])
def test_lookup_rejects_malformed_requests(setup, patch):
    client, index, _, _ = setup
    body = {"previous_candidate_ids": ["200", "201"], "shown_ids": ["200"], **patch}
    assert client.post("/search/clarify", json=body).status_code == 422
    assert index.calls == []


def test_lookup_needs_nothing_but_candidates(setup):
    client, _, _, _ = setup
    assert client.post("/search/clarify", json={}).json() == {"clarify": None}
    body = {"previous_candidate_ids": [sid for sid, _, _ in ERA_POOL]}
    assert client.post("/search/clarify", json=body).json()["clarify"]["slot"] == "vocal_gender"


def test_ids_that_are_not_in_the_corpus_are_skipped_not_errors():
    """'²'는 `isdigit()`이 참이지만 정수가 아니다 — 실제 Qdrant 백엔드까지 지나도 500이 아니어야 한다."""
    from qdrant_client import models

    from src.vector_db.qdrant_backend import DENSE_VECTOR, QdrantVectorClient, point_id
    from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

    vector_client = QdrantVectorClient(path=":memory:")
    try:
        collection = vector_client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, recreate=True)
        stored = [("200", "여성"), ("201", "남성"), ("non-numeric", "여성"), ("203", "남성")]
        vector_client.client.upsert(collection_name=collection, points=[
            models.PointStruct(id=point_id(sid), vector={DENSE_VECTOR: [1.0, 0.0, 0.0]},
                               payload={"song_id": sid, "title": sid, "vocal_gender": gender, "genre": "발라드"})
            for sid, gender in stored
        ])
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[get_vector_client] = lambda: vector_client
        with TestClient(app) as client:
            odd = ["²", "①", "１２３", "9" * 30, "-5", "404", " "]
            response = client.post("/search/clarify", json={
                "previous_candidate_ids": [*odd, *(sid for sid, _ in stored)],
            })
        assert response.status_code == 200, response.text
        question = response.json()["clarify"]
        assert question["slot"] == "vocal_gender"
        assert {o["value"]: o["count"] for o in question["options"]} == {"여성": 2, "남성": 2}, \
            "숫자 id와 비숫자 id는 그대로 읽히고, 없는 id만 빠진다"
    finally:
        vector_client.close()
