"""v62 — 하이브리드 조회의 limit이 **요청마다 센 대상 수**로 정해지는지, 호출 모양과 결과 두 층에서 확인한다.

두 종류의 자료원을 쓴다.
  - **가짜 클라이언트**(`FakeQdrantClient`): `count` · `query_points` · `retrieve` 호출을 그대로 기록하고, 저장 벡터로 점수를
    계산해 `limit`만큼 **실제로 자른다**. 호출 순서·limit·필터·payload 플래그·실패 전파처럼 "어떤 호출을 어떤 인자로 했나"를
    본다. count 값을 임의로 돌려줄 수 있어 "count와 조회가 한 스냅샷이 아니다"라는 전제도 그대로 드러낸다.
  - **임시 로컬 Qdrant**(`:memory:` · `tmp_path` 파일 모드): 실제 엔진에서 점수·순서가 직접 내적 기준(`_exact`)과 같은지,
    요청 사이에 점이 늘면 다음 요청의 limit이 따라가는지, payload를 최종 top_k만 읽는지를 본다.

기준값은 구현을 따라 쓰지 않는다 — 원본 벡터로 직접 내적한 값이다. 옛 결함(고정 상한 5000)의 재현은 v61 기준 커밋
`fc20e86`(PR #76)의 `tests/vector_db/test_v61_hybrid_scan_audit.py`에 있고, 수정 뒤의 같은 자료 검증은 현재 그 파일에 있다.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from qdrant_client import models

from src.vector_db import qdrant_backend as qb
from src.vector_db.qdrant_backend import DENSE_VECTOR, SPARSE_VECTOR, QdrantIndex, QdrantVectorClient, point_id
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME

TOL = 1e-5
Song = Tuple[str, Sequence[float], Dict[int, float], Dict[str, object]]
COLLECTION = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)


def _corpus() -> List[Song]:
    """v61과 같은 7곡. 질의 dense=[0.5,0,0] · sparse {7: 0.5}에서 직접 합산 4=1.15 · 5=0.85 · 2=0.65 · 7=0.60 · 6=0.55 · 1=0.50 · 3=0.35."""
    return [
        ("1", [0.90, 0.10, 0.00], {7: 0.10}, {"g": "a"}),
        ("2", [0.80, 0.20, 0.00], {7: 0.50}, {"g": "a"}),
        ("3", [0.70, 0.30, 0.00], {8: 5.00}, {"g": "b"}),
        ("4", [0.30, 0.90, 0.00], {7: 2.00}, {"g": "b"}),
        ("5", [0.20, 0.00, 0.90], {7: 1.50}, {"g": "b"}),
        ("6", [0.10, 0.00, 0.95], {7: 1.00}, {"g": "c"}),
        ("7", [0.60, 0.50, 0.00], {7: 0.60}, {"g": "c"}),
    ]


DENSE_Q = [0.5, 0.0, 0.0]
SPARSE_Q = {"indices": [7], "values": [0.5]}


def _exact(songs: Sequence[Song], dense_q: Sequence[float], sparse_q: Optional[Dict[str, Sequence]],
           allowed: Optional[set] = None) -> List[Tuple[str, float]]:
    """(id, 합산) 합산 내림차순 — 원본 벡터로 직접 계산. 동점은 id 순(재현용)."""
    terms = dict(zip(sparse_q.get("indices", []), sparse_q.get("values", []))) if sparse_q else {}
    rows = []
    for song_id, dense, sparse, _payload in songs:
        if allowed is not None and song_id not in allowed:
            continue
        d = sum(a * b for a, b in zip(dense, dense_q))
        s = sum(v * terms[i] for i, v in sparse.items() if i in terms)
        rows.append((song_id, d + s))
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def _assert_exact(res: Dict[str, Any], expected: List[Tuple[str, float]]) -> None:
    assert [m["id"] for m in res["matches"]] == [r[0] for r in expected]
    got = {m["id"]: m["score"] for m in res["matches"]}
    for song_id, total in expected:
        assert got[song_id] == pytest.approx(total, abs=TOL)


# ---------------------------------------------------------------------------
# 가짜 클라이언트 — 호출을 기록하고 limit만큼 실제로 자른다
# ---------------------------------------------------------------------------

class _Count:
    def __init__(self, count: int) -> None:
        self.count = count


class _Response:
    def __init__(self, points: List[models.ScoredPoint]) -> None:
        self.points = points


class _Record:
    def __init__(self, pid: int, payload: Dict[str, Any]) -> None:
        self.id = pid
        self.payload = payload


class FakeQdrantClient:
    """`QdrantIndex`가 쓰는 세 메서드(`count` · `query_points` · `retrieve`)만 흉내 낸다.

    - `count`: 기본은 실제 점 수. `count_override`를 주면 그 값을 돌려준다(한 스냅샷이 아닌 상황을 흉내), `count_error`를 주면 던진다
    - `query_points`: 저장 벡터로 점수를 계산해 내림차순 상위 `limit`개만 돌려준다(진짜 엔진처럼 자른다). sparse 미일치 곡은 돌려주지 않는다
    - 필터는 기록만 한다(의미 해석은 실제 엔진 테스트가 맡는다). 단, 결과 수를 필터와 맞추는 데는 `count_override`를 쓴다
    """

    def __init__(self, songs: Sequence[Song]) -> None:
        self.songs: Dict[int, Song] = {int(point_id(s[0])): s for s in songs}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.count_override: Optional[int] = None
        self.count_error: Optional[Exception] = None

    # --- 기록 보기 ---------------------------------------------------------
    def names(self) -> List[str]:
        return [name for name, _ in self.calls]

    def of(self, name: str) -> List[Dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]

    # --- 흉내 ---------------------------------------------------------------
    def count(self, collection_name: str, count_filter: Any = None, exact: bool = True, **_: Any) -> _Count:
        self.calls.append(("count", {"collection_name": collection_name, "count_filter": count_filter, "exact": exact}))
        if self.count_error is not None:
            raise self.count_error
        return _Count(len(self.songs) if self.count_override is None else self.count_override)

    def query_points(self, collection_name: str, query: Any, using: str, limit: int, query_filter: Any = None,
                     with_payload: Any = False, search_params: Any = None, **_: Any) -> _Response:
        self.calls.append(("query_points", {"collection_name": collection_name, "using": using, "limit": limit,
                                            "query_filter": query_filter, "with_payload": with_payload}))
        scored: List[Tuple[int, float]] = []
        for pid, (_sid, dense, sparse, _payload) in self.songs.items():
            if using == DENSE_VECTOR:
                scored.append((pid, float(sum(a * b for a, b in zip(dense, query)))))
            else:
                terms = dict(zip(query.indices, query.values))
                hit = [(i, v) for i, v in sparse.items() if i in terms]
                if hit:
                    scored.append((pid, float(sum(v * terms[i] for i, v in hit))))
        scored.sort(key=lambda r: (-r[1], r[0]))
        points = [
            models.ScoredPoint(id=pid, version=0, score=score,
                               payload=({"song_id": self.songs[pid][0], **self.songs[pid][3]} if with_payload else None))
            for pid, score in scored[:limit]
        ]
        return _Response(points)

    def retrieve(self, collection_name: str, ids: List[Any], with_payload: Any = True, with_vectors: bool = False, **_: Any) -> List[_Record]:
        self.calls.append(("retrieve", {"collection_name": collection_name, "ids": list(ids), "with_payload": with_payload}))
        return [_Record(pid, {"song_id": self.songs[pid][0], **self.songs[pid][3]}) for pid in ids if pid in self.songs]


def _fake_index(songs: Sequence[Song]) -> Tuple[FakeQdrantClient, QdrantIndex]:
    fake = FakeQdrantClient(songs)
    return fake, QdrantIndex(fake, TEXT_HYBRID_INDEX_NAME, exact=True)   # type: ignore[arg-type]


def test_both_scan_limits_equal_the_counted_target_and_count_comes_first():
    fake, index = _fake_index(_corpus())
    res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=3)
    assert fake.names() == ["count", "query_points", "query_points", "retrieve"]
    dense, sparse = fake.of("query_points")
    assert (dense["using"], sparse["using"]) == (DENSE_VECTOR, SPARSE_VECTOR)
    assert dense["limit"] == sparse["limit"] == 7 == len(_corpus())
    assert fake.of("count")[0]["exact"] is True
    _assert_exact(res, _exact(_corpus(), DENSE_Q, SPARSE_Q)[:3])


def test_the_same_filter_goes_to_count_dense_and_sparse():
    fake, index = _fake_index(_corpus())
    fake.count_override = 3    # 필터 통과 3곡(의미 해석은 실제 엔진 테스트가 맡는다)
    index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=5, filter={"g": {"$eq": "b"}})
    expected = qb.translate_filter({"g": {"$eq": "b"}})
    assert expected is not None
    assert fake.of("count")[0]["count_filter"] == expected
    for call in fake.of("query_points"):
        assert call["query_filter"] == expected
        assert call["limit"] == 3


def test_zero_targets_return_empty_without_search_or_payload():
    fake, index = _fake_index(_corpus())
    fake.count_override = 0
    assert index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=5, filter={"g": {"$eq": "zzz"}}) == {"matches": []}
    assert fake.names() == ["count"]


def test_dense_only_query_does_not_count_and_uses_top_k_as_limit():
    fake, index = _fake_index(_corpus())
    res = index.query(vector=DENSE_Q, top_k=2)
    assert fake.names() == ["query_points"]
    # v65: 단일 벡터 조회는 경계 동점을 보려고 top_k+1개를 받는다(동점이 없으면 그것으로 끝, count 없음)
    assert fake.of("query_points")[0]["limit"] == 3
    assert fake.of("query_points")[0]["with_payload"] is True      # dense 전용은 조회에서 바로 payload를 받는다(기존 동작)
    assert [m["id"] for m in res["matches"]] == ["1", "2"]


def test_count_failure_is_propagated_and_no_search_runs(caplog):
    fake, index = _fake_index(_corpus())
    fake.count_error = RuntimeError("count unavailable")
    caplog.set_level(logging.ERROR, logger=qb.__name__)
    with pytest.raises(RuntimeError, match="count unavailable"):
        index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=3)
    assert fake.names() == ["count"]
    assert any("count" in r.getMessage() and "실패" in r.getMessage() for r in caplog.records)


def test_scores_are_only_read_and_payload_is_fetched_for_the_final_top_k_only():
    fake, index = _fake_index(_corpus())
    res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=2)
    assert [c["with_payload"] for c in fake.of("query_points")] == [False, False]
    retrieve = fake.of("retrieve")
    assert len(retrieve) == 1
    assert retrieve[0]["ids"] == [point_id("4"), point_id("5")]        # 합산 상위 2곡만
    assert [m["id"] for m in res["matches"]] == ["4", "5"]
    assert res["matches"][0]["metadata"] == {"g": "b"}

    fake.calls.clear()
    res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=2, include_metadata=False)
    # v65: 메타데이터를 안 받아도 곡 id(동점 순서 키)는 payload song_id에서 — 경계까지의 점만, song_id 필드만 읽는다
    assert fake.names() == ["count", "query_points", "query_points", "retrieve"]
    assert fake.of("retrieve")[0]["ids"] == [point_id("4"), point_id("5")]
    assert fake.of("retrieve")[0]["with_payload"] == ["song_id"]
    assert [m["id"] for m in res["matches"]] == ["4", "5"]
    assert res["matches"][0]["metadata"] == {}


def test_a_changed_target_count_is_used_by_the_very_next_request():
    """요청 사이에 count가 바뀌면(점 추가) 다음 요청의 두 limit이 그 값이다 — 캐시가 없다."""
    fake, index = _fake_index(_corpus()[:3])
    index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
    assert [c["limit"] for c in fake.of("query_points")] == [3, 3]
    for song in _corpus()[3:]:
        fake.songs[int(point_id(song[0]))] = song
    fake.calls.clear()
    res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
    assert fake.names()[0] == "count"
    assert [c["limit"] for c in fake.of("query_points")] == [7, 7]
    _assert_exact(res, _exact(_corpus(), DENSE_Q, SPARSE_Q))


def test_premise_a_stale_count_is_trusted_so_concurrent_loading_can_still_truncate():
    """전제의 반대 사례를 그대로 적는다: count가 조회 시점의 점 수보다 작으면(동시 적재) 그 요청은 잘린다 — 이번 수정이 보장하지 않는 범위.

    반환 건수(= limit)로는 알 수 없다. 운영 조건(적재 중 검색 중단)이나 발행 설계(세대 + 별칭)로 막아야 한다.
    """
    fake, index = _fake_index(_corpus())
    fake.count_override = 3   # 조회 시점에는 7곡인데 count는 3을 돌려준 상황
    res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
    assert [c["limit"] for c in fake.of("query_points")] == [3, 3]
    got = {m["id"]: m["score"] for m in res["matches"]}
    assert "7" not in got                                          # 양쇽 목록 밖 곡이 사라진다(옛 결함과 같은 모양)
    assert got["4"] == pytest.approx(1.0, abs=TOL)                 # dense 몫이 빠진다
    assert len(res["matches"]) == 6                                # 반환 건수만으로는 "전수가 아니었다"를 알 수 없다


def test_old_env_limit_set_small_does_not_truncate_the_fake_either(monkeypatch):
    monkeypatch.setenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV, "2")
    try:
        reloaded = importlib.reload(qb)
        assert reloaded.HYBRID_SCAN_LIMIT == 2
        fake = FakeQdrantClient(_corpus())
        index = reloaded.QdrantIndex(fake, TEXT_HYBRID_INDEX_NAME, exact=True)   # type: ignore[arg-type]
        res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
        assert [c["limit"] for c in fake.of("query_points")] == [7, 7]
        _assert_exact(res, _exact(_corpus(), DENSE_Q, SPARSE_Q))
    finally:
        monkeypatch.undo()
        importlib.reload(qb)


# ---------------------------------------------------------------------------
# 임시 로컬 Qdrant — 실제 엔진에서 결과·limit·payload 확인
# ---------------------------------------------------------------------------

def _fill(client: QdrantVectorClient, songs: Sequence[Song], recreate: bool = True) -> None:
    collection = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, with_sparse=True, recreate=recreate)
    client.client.upsert(collection_name=collection, points=[
        models.PointStruct(
            id=point_id(song_id),
            payload={"song_id": song_id, "title": f"T{song_id}", **payload},
            vector={DENSE_VECTOR: list(dense),
                    SPARSE_VECTOR: models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))},
        )
        for song_id, dense, sparse, payload in songs
    ])


class _Spy:
    """실제 클라이언트의 count·query_points·retrieve를 감싸 인자를 기록한다."""

    def __init__(self, raw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.limits: List[int] = []
        self.count_filters: List[Any] = []
        self.query_filters: List[Any] = []
        self.payload_flags: List[Any] = []
        self.retrieved: List[List[Any]] = []
        real_count, real_query, real_retrieve = raw.count, raw.query_points, raw.retrieve

        def count(*a: Any, **kw: Any) -> Any:
            self.count_filters.append(kw.get("count_filter"))
            return real_count(*a, **kw)

        def query_points(*a: Any, **kw: Any) -> Any:
            self.limits.append(kw["limit"])
            self.query_filters.append(kw.get("query_filter"))
            self.payload_flags.append(kw.get("with_payload"))
            return real_query(*a, **kw)

        def retrieve(*a: Any, **kw: Any) -> Any:
            self.retrieved.append(list(kw["ids"]))
            return real_retrieve(*a, **kw)

        monkeypatch.setattr(raw, "count", count)
        monkeypatch.setattr(raw, "query_points", query_points)
        monkeypatch.setattr(raw, "retrieve", retrieve)


@pytest.fixture()
def client():
    c = QdrantVectorClient(path=":memory:")
    yield c
    c.close()


def test_real_engine_limits_follow_the_count_and_results_match_the_exact_sum(client, monkeypatch):
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)      # 옛 상수는 아무 효과가 없어야 한다
    spy = _Spy(client.client, monkeypatch)
    res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
    assert spy.limits == [7, 7]
    assert spy.payload_flags == [False, False]
    assert spy.count_filters == [None]
    _assert_exact(res, _exact(songs, DENSE_Q, SPARSE_Q))


def test_real_engine_applies_the_same_filter_to_count_and_both_scans(client, monkeypatch):
    songs = _corpus()
    _fill(client, songs)
    spy = _Spy(client.client, monkeypatch)
    res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=5, filter={"g": {"$eq": "b"}})
    assert spy.limits == [3, 3]                            # 통과 3곡(3·4·5)
    assert spy.count_filters[0] is not None and spy.query_filters == [spy.count_filters[0]] * 2
    _assert_exact(res, _exact(songs, DENSE_Q, SPARSE_Q, allowed={"3", "4", "5"}))


def test_real_engine_payload_is_read_only_for_the_final_top_k(client, monkeypatch):
    _fill(client, _corpus())
    spy = _Spy(client.client, monkeypatch)
    res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=2)
    assert spy.retrieved == [[point_id("4"), point_id("5")]]
    assert [m["metadata"]["title"] for m in res["matches"]] == ["T4", "T5"]


def test_real_engine_dense_only_does_not_count(client, monkeypatch):
    _fill(client, _corpus())
    spy = _Spy(client.client, monkeypatch)
    res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=DENSE_Q, top_k=2)
    assert spy.count_filters == [] and spy.limits == [3]   # v65: top_k+1(경계 동점 확인), count 없음
    assert [m["id"] for m in res["matches"]] == ["1", "2"]


def test_points_added_between_requests_are_counted_by_the_next_request_in_file_mode(tmp_path, monkeypatch):
    """파일 모드 임시 Qdrant: 3곡으로 조회 → 4곡 추가 → 다음 조회의 limit이 7이고 결과가 7곡 기준과 같다."""
    c = QdrantVectorClient(path=str(tmp_path / "qdrant"))
    try:
        songs = _corpus()
        _fill(c, songs[:3])
        spy = _Spy(c.client, monkeypatch)
        index = c.Index(TEXT_HYBRID_INDEX_NAME)
        _assert_exact(index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7), _exact(songs[:3], DENSE_Q, SPARSE_Q))
        assert spy.limits == [3, 3]
        _fill(c, songs[3:], recreate=False)
        res = index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=7)
        assert spy.limits == [3, 3, 7, 7]
        _assert_exact(res, _exact(songs, DENSE_Q, SPARSE_Q))
    finally:
        c.close()


def test_negative_dense_is_summed_exactly_on_the_real_engine(client):
    songs: List[Song] = [
        ("1", [1.0, 0.0, 0.0], {7: 0.5}, {}),
        ("2", [0.5, 0.5, 0.0], {7: 0.5}, {}),
        ("3", [-1.0, 0.0, 0.0], {7: 3.0}, {}),
    ]
    _fill(client, songs)
    dense_q, sparse_q = [1.0, 0.0, 0.0], {"indices": [7], "values": [1.0]}
    res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=dense_q, sparse_vector=sparse_q, top_k=3)
    _assert_exact(res, _exact(songs, dense_q, sparse_q))
    assert {m["id"]: m["score"] for m in res["matches"]}["3"] == pytest.approx(2.0, abs=TOL)


def test_sparse_cases_on_the_real_engine(client):
    """sparse 미일치(합산 = dense) · 빈 sparse(설계상 사례) · top_k 초과(전부 1회) · 동점(둘 다 남고 점수 같음)."""
    songs = _corpus()
    _fill(client, songs)
    index = client.Index(TEXT_HYBRID_INDEX_NAME)
    _assert_exact(index.query(vector=DENSE_Q, sparse_vector={"indices": [99], "values": [0.5]}, top_k=7),
                  _exact(songs, DENSE_Q, {"indices": [99], "values": [0.5]}))
    assert [m["id"] for m in index.query(vector=DENSE_Q, sparse_vector={"indices": [], "values": []}, top_k=7)["matches"]] == \
        [r[0] for r in _exact(songs, DENSE_Q, None)]
    ids = [m["id"] for m in index.query(vector=DENSE_Q, sparse_vector=SPARSE_Q, top_k=50)["matches"]]
    assert sorted(ids) == sorted(s[0] for s in songs) and len(ids) == len(set(ids))

    ties: List[Song] = [("1", [1.0, 0.0, 0.0], {7: 1.0}, {}), ("2", [1.0, 0.0, 0.0], {7: 1.0}, {}), ("3", [0.0, 1.0, 0.0], {8: 1.0}, {})]
    _fill(client, ties)
    res = index.query(vector=[1.0, 0.0, 0.0], sparse_vector={"indices": [7], "values": [1.0]}, top_k=3)
    got = {m["id"]: m["score"] for m in res["matches"]}
    assert got["1"] == pytest.approx(got["2"], abs=TOL) == pytest.approx(2.0, abs=TOL)
    assert {m["id"] for m in res["matches"][:2]} == {"1", "2"}
