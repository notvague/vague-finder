"""v65 — Qdrant 조회 결과의 동점 순서와 조회 상한 경계 동점이 **백엔드 반환 순서와 무관하게** 정해지는지 본다.

계약(`src/vector_db/qdrant_backend.py` 모듈 설명 "동점 순서"):
  점수 내림차순 → 원래 곡 id(`song_id_sort_key`: ASCII 숫자열은 숫자 값, 그 밖은 뒤에 코드포인트 순).
  Context 사실은 → song_id → `record_id`, 프로필은 → song_id → `profile_id`. 동점 = 점수가 정확히 같을 때만.
  상위 k개는 k번째 점수와 같은 점을 모두 받은 뒤 자른다.

기준값은 구현을 따라 쓰지 않는다. 저장한 벡터로 직접 점수를 계산하고(`_reference`) 계약의 키를 이 파일에서 따로 적어
정렬한다. "어떤 정확 검색 백엔드든"을 흉내 내려고 `AdversarialBackend`가 실제 엔진의 전체 결과를 받아 **동점 안의 순서를
무작위로 섞은 뒤 limit에서 자른다** — 로컬 파일 모드와 서버 모드의 반환 순서 차이(v64 §7-①)를 일반화한 것이다.
실제 서버·파일 모드·재적재·재시작은 `experiments/reranking/results_v65_deterministic_ties/synthetic_backends.py`가 본다.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from qdrant_client import models

from src.vector_db.qdrant_backend import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantVectorClient,
    complete_top_points,
    point_id,
    song_id_sort_key,
)
from src.vector_db.settings import IMAGE_INDEX_NAME, TEXT_HYBRID_INDEX_NAME


def _key(song_id: str) -> Tuple[int, int, str]:
    """계약의 키를 구현과 따로 적는다."""
    s = str(song_id).strip()
    return (0, int(s), s) if s.isascii() and s.isdigit() else (1, 0, s)


# ---------------------------------------------------------------------------
# song_id_sort_key
# ---------------------------------------------------------------------------

def test_numeric_ids_compare_by_value_not_by_string():
    ids = ["10000", "9999", "123456789", "4321", "30001234"]
    assert sorted(ids, key=song_id_sort_key) == ["4321", "9999", "10000", "30001234", "123456789"]
    assert sorted(ids) != sorted(ids, key=song_id_sort_key)  # 문자열 순서와 실제로 다르다


def test_non_numeric_ids_follow_numeric_ids_in_code_point_order():
    ids = ["b", "12", "A", "a-1", "3"]
    assert sorted(ids, key=song_id_sort_key) == ["3", "12", "A", "a-1", "b"]


def test_leading_zero_whitespace_and_non_ascii_digits():
    assert song_id_sort_key(" 42 ") == song_id_sort_key("42") == (0, 42, "42")
    assert sorted(["7", "007"], key=song_id_sort_key) == ["007", "7"]  # 같은 값이면 문자열 순 — 결정적
    # '²'·전각 숫자는 int()가 받거나 isdigit()이 참이어도 ASCII가 아니다 → 비숫자 쪽
    for odd in ("²", "１２"):
        assert song_id_sort_key(odd)[0] == 1
    assert [song_id_sort_key(x) for x in ("1", "x")] == [_key("1"), _key("x")]


# ---------------------------------------------------------------------------
# complete_top_points — 백엔드 흉내
# ---------------------------------------------------------------------------

def _point(pid: Any, score: float, song_id: Optional[str]) -> models.ScoredPoint:
    payload = {} if song_id is None else {"song_id": song_id}
    return models.ScoredPoint(id=pid, version=0, score=score, payload=payload)


class FakeExactSearch:
    """정확 검색처럼 점수 내림차순 상위 limit개를 주되, 동점 안의 순서는 `seed`로 섞는다."""

    def __init__(self, points: Sequence[models.ScoredPoint], seed: int) -> None:
        self.points = list(points)
        self.rng = random.Random(seed)
        self.limits: List[int] = []

    def __call__(self, limit: int) -> List[models.ScoredPoint]:
        self.limits.append(limit)
        shuffled = list(self.points)
        self.rng.shuffle(shuffled)
        shuffled.sort(key=lambda p: -p.score)  # 안정 정렬 — 동점 안 순서는 섞인 그대로
        return shuffled[:limit]


def _identity(hit: models.ScoredPoint):
    sid = str((hit.payload or {}).get("song_id", "")).strip()
    return song_id_sort_key(sid) if sid else None


def _expected(points: Sequence[models.ScoredPoint], k: int) -> List[str]:
    valid = [p for p in points if str((p.payload or {}).get("song_id", "")).strip()]
    return [p.payload["song_id"] for p in sorted(valid, key=lambda p: (-p.score, _key(p.payload["song_id"])))][:k]


def _ids(result) -> List[str]:
    return [hit.payload["song_id"] for hit, _ in result]


@pytest.mark.parametrize("seed", range(8))
def test_tie_group_larger_than_k_selects_the_same_set_for_any_backend_order(seed):
    points = [_point(i, 0.9, str(1000 + i)) for i in range(12)] + [_point(100, 0.95, "77"), _point(101, 0.1, "5")]
    search = FakeExactSearch(points, seed)
    got = complete_top_points(search, 4, _identity)
    assert _ids(got) == _expected(points, 4) == ["77", "1000", "1001", "1002"]
    # 첫 조회 k+1=5 → 5번째도 동점 → 10 → 20(대상 14 < 20, 끝). 고정 배수 한 번이 아니라 종료 조건까지 넓힌다
    assert search.limits == [5, 10, 20]


def test_non_tie_just_below_the_boundary_costs_one_extra_point():
    points = [_point(1, 0.9, "1"), _point(2, 0.8, "2"), _point(3, 0.7000001, "3"), _point(4, 0.7, "4")]
    search = FakeExactSearch(points, 0)
    assert _ids(complete_top_points(search, 3, _identity)) == ["1", "2", "3"]
    assert search.limits == [4]


def test_tie_exactly_at_the_boundary_expands_until_a_lower_score_is_seen():
    points = [_point(1, 0.9, "1"), _point(2, 0.5, "30"), _point(3, 0.5, "4"), _point(4, 0.5, "200"), _point(5, 0.1, "9")]
    for seed in range(5):
        search = FakeExactSearch(points, seed)
        assert _ids(complete_top_points(search, 2, _identity)) == ["1", "4"]
        assert search.limits == [3, 6]  # 6 > 대상 5 → 끝


def test_tie_above_but_not_at_the_boundary_needs_no_expansion():
    points = [_point(1, 0.9, "20"), _point(2, 0.9, "3"), _point(3, 0.5, "1"), _point(4, 0.4, "2")]
    for seed in range(5):
        search = FakeExactSearch(points, seed)
        assert _ids(complete_top_points(search, 3, _identity)) == ["3", "20", "1"]
        assert search.limits == [4]


def test_empty_k_at_least_total_and_zero_k():
    assert complete_top_points(FakeExactSearch([], 0), 5, _identity) == []
    points = [_point(i, 0.5, str(i)) for i in range(3)]
    search = FakeExactSearch(points, 1)
    assert _ids(complete_top_points(search, 10, _identity)) == ["0", "1", "2"]
    assert search.limits == [11]
    zero = FakeExactSearch(points, 1)
    assert complete_top_points(zero, 0, _identity) == [] and zero.limits == []


def test_points_without_song_id_are_not_candidates_and_do_not_shrink_the_result():
    # 무효 점이 상위를 차지해도 유효한 k개를 경계 동점까지 채운다
    points = [_point(1, 0.99, None), _point(2, 0.98, ""), _point(3, 0.9, "8"), _point(4, 0.8, "3"),
              _point(5, 0.8, "2"), _point(6, 0.1, "1")]
    for seed in range(5):
        search = FakeExactSearch(points, seed)
        assert _ids(complete_top_points(search, 2, _identity)) == ["8", "2"]


def test_duplicate_keys_keep_only_the_first_position():
    points = [_point(1, 0.9, "5"), _point(2, 0.9, "5"), _point(3, 0.8, "6")]
    got = complete_top_points(FakeExactSearch(points, 0), 2, _identity)
    assert _ids(got) == ["5", "6"]


# ---------------------------------------------------------------------------
# 실제 엔진(로컬 메모리 모드) + 반환 순서를 섞는 백엔드
# ---------------------------------------------------------------------------

class AdversarialBackend:
    """실제 엔진의 정확한 전체 결과를 받아 동점 안을 섞은 뒤 limit에서 자른다(어떤 정확 검색 백엔드든)."""

    def __init__(self, client: Any, seed: int, *, equal_scores: bool = False) -> None:
        self.client = client
        self.real = client.query_points
        self.rng = random.Random(seed)
        self.equal_scores = equal_scores
        self.limits: List[int] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        limit = int(kwargs.pop("limit"))
        self.limits.append(limit)
        name = kwargs.get("collection_name") or args[0]
        total = self.client.count(collection_name=name, exact=True).count
        response = self.real(*args, limit=max(1, total), **kwargs)
        points = list(response.points)
        if self.equal_scores:
            points = [p.model_copy(update={"score": 1.0}) for p in points]
        self.rng.shuffle(points)
        points.sort(key=lambda p: -p.score)
        response.points = points[:limit]
        return response


def _unit(*xs: float) -> List[float]:
    v = np.asarray(xs, dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


# 같은 표지(같은 벡터) 묶음 7곡 + 앞뒤 비동점. 질의 [1,0,0]에서 묶음 점수가 정확히 같다
IMAGE_SONGS: List[Tuple[str, List[float]]] = (
    [("9001", _unit(1.0, 0.0, 0.0))]
    + [(sid, _unit(0.6, 0.8, 0.0)) for sid in ("700", "12", "123456789", "45", "9999", "10000", "88")]
    + [("31", _unit(0.0, 0.0, 1.0)), ("32", _unit(0.1, 0.0, 0.9))]
)


def _fill_image(client: QdrantVectorClient, songs, *, reverse: bool = False, payload_extra=None) -> None:
    collection = client.ensure_collection(IMAGE_INDEX_NAME, dim=3, recreate=True)
    rows = list(reversed(songs)) if reverse else list(songs)
    client.client.upsert(collection_name=collection, points=[
        models.PointStruct(id=point_id(sid), vector={DENSE_VECTOR: vec},
                           payload={"song_id": sid, "title": f"T{sid}", **(payload_extra or {}).get(sid, {})})
        for sid, vec in rows
    ])


def _reference(songs, query: Sequence[float], k: int, allowed: Optional[set] = None) -> List[str]:
    q = np.asarray(query, dtype=np.float64)
    q = q / np.linalg.norm(q)
    scored = [(sid, float(np.dot(np.asarray(vec), q))) for sid, vec in songs if allowed is None or sid in allowed]
    # 같은 벡터면 같은 값 — float64로 직접. 계약 키로 정렬
    return [sid for sid, _ in sorted(scored, key=lambda r: (-r[1], _key(r[0])))][:k]


@pytest.fixture()
def mem():
    c = QdrantVectorClient(path=":memory:")
    yield c
    c.close()


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 10, 20])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_image_like_tie_group_matches_the_reference_for_any_return_order(mem, monkeypatch, k, seed):
    _fill_image(mem, IMAGE_SONGS, reverse=bool(seed % 2))
    backend = AdversarialBackend(mem.client, seed)
    monkeypatch.setattr(mem.client, "query_points", backend)
    res = mem.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], top_k=k)
    assert [m["id"] for m in res["matches"]] == _reference(IMAGE_SONGS, [1.0, 0.0, 0.0], k)
    assert res["matches"] and all(m["metadata"]["title"] == f"T{m['id']}" for m in res["matches"])


def test_insertion_order_reversal_gives_the_same_result_on_the_real_engine():
    results = []
    for reverse in (False, True):
        c = QdrantVectorClient(path=":memory:")
        try:
            _fill_image(c, IMAGE_SONGS, reverse=reverse)
            results.append([m["id"] for m in c.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], top_k=4)["matches"]])
        finally:
            c.close()
    assert results[0] == results[1] == _reference(IMAGE_SONGS, [1.0, 0.0, 0.0], 4) == ["9001", "12", "45", "88"]


def test_filtered_tie_group_and_zero_matching_filter(mem, monkeypatch):
    extra = {sid: {"g": "x" if sid in {"700", "12", "9999", "31"} else "y"} for sid, _ in IMAGE_SONGS}
    _fill_image(mem, IMAGE_SONGS, payload_extra=extra)
    monkeypatch.setattr(mem.client, "query_points", AdversarialBackend(mem.client, 3))
    index = mem.Index(IMAGE_INDEX_NAME)
    res = index.query(vector=[1.0, 0.0, 0.0], top_k=2, filter={"g": {"$eq": "x"}})
    allowed = {sid for sid, e in extra.items() if e["g"] == "x"}
    assert [m["id"] for m in res["matches"]] == _reference(IMAGE_SONGS, [1.0, 0.0, 0.0], 2, allowed) == ["12", "700"]
    assert index.query(vector=[1.0, 0.0, 0.0], top_k=2, filter={"g": {"$eq": "none"}})["matches"] == []


def test_non_numeric_song_ids_use_payload_ids_not_uuid_point_ids(mem, monkeypatch):
    songs = [(sid, _unit(1.0, 1.0, 0.0)) for sid in ("zeta", "Alpha", "mid-1", "42", "7")]
    _fill_image(mem, songs)
    for seed in range(4):
        monkeypatch.setattr(mem.client, "query_points", AdversarialBackend(mem.client, seed))
        res = mem.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 1.0, 0.0], top_k=4)
        # UUID point id 순서가 아니라 원래 id 키 순서
        assert [m["id"] for m in res["matches"]] == ["7", "42", "Alpha", "mid-1"]


def test_metadata_off_still_identifies_songs_by_payload_song_id(mem, monkeypatch):
    _fill_image(mem, IMAGE_SONGS)
    real = mem.client.query_points
    seen: List[Any] = []

    def spy(*a, **kw):
        seen.append(kw.get("with_payload"))
        return real(*a, **kw)

    monkeypatch.setattr(mem.client, "query_points", spy)
    res = mem.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], top_k=3, include_metadata=False)
    assert [m["id"] for m in res["matches"]] == ["9001", "12", "45"]
    assert all(m["metadata"] == {} for m in res["matches"])
    assert set(map(str, seen)) == {"['song_id']"}  # 곡 id만 읽는다


def test_point_without_song_id_payload_is_dropped_not_named_by_point_id(mem, caplog):
    collection = mem.ensure_collection(IMAGE_INDEX_NAME, dim=3, recreate=True)
    mem.client.upsert(collection_name=collection, points=[
        models.PointStruct(id=1, vector={DENSE_VECTOR: _unit(1, 0, 0)}, payload={"title": "no id"}),
        models.PointStruct(id=2, vector={DENSE_VECTOR: _unit(1, 0.1, 0)}, payload={"song_id": "2"}),
        models.PointStruct(id=3, vector={DENSE_VECTOR: _unit(1, 0.2, 0)}, payload={"song_id": "3"}),
    ])
    res = mem.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], top_k=2)
    assert [m["id"] for m in res["matches"]] == ["2", "3"]
    assert any("song_id" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 하이브리드 — 전수 합산 + 같은 키
# ---------------------------------------------------------------------------

def _fill_text(client: QdrantVectorClient, songs) -> None:
    collection = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, with_sparse=True, recreate=True)
    client.client.upsert(collection_name=collection, points=[
        models.PointStruct(id=point_id(sid), payload={"song_id": sid, "title": sid},
                           vector={DENSE_VECTOR: dense, SPARSE_VECTOR: models.SparseVector(indices=list(sp), values=list(sp.values()))})
        for sid, dense, sp in songs
    ])


TEXT_SONGS = [
    ("500", [0.5, 0.0, 0.0], {7: 1.0}),
    # 아래 4곡은 dense·sparse가 같다 → 합산이 정확히 같다(경계 동점)
    ("90", [0.4, 0.0, 0.0], {7: 0.5}),
    ("1000", [0.4, 0.0, 0.0], {7: 0.5}),
    ("8", [0.4, 0.0, 0.0], {7: 0.5}),
    ("300", [0.4, 0.0, 0.0], {7: 0.5}),
    ("2", [0.1, 0.0, 0.0], {9: 3.0}),
]


@pytest.mark.parametrize("seed", range(4))
def test_hybrid_ties_are_ordered_by_song_id_and_payload_is_read_only_to_the_boundary(mem, monkeypatch, seed):
    _fill_text(mem, TEXT_SONGS)
    monkeypatch.setattr(mem.client, "query_points", AdversarialBackend(mem.client, seed))
    real_retrieve = mem.client.retrieve
    retrieved: List[int] = []

    def spy(*a, **kw):
        retrieved.append(len(list(kw["ids"])))
        return real_retrieve(*a, **kw)

    monkeypatch.setattr(mem.client, "retrieve", spy)
    res = mem.Index(TEXT_HYBRID_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], sparse_vector={"indices": [7], "values": [1.0]},
                                                  top_k=3)
    # 500: 0.5+1.0 · 동점 4곡: 0.4+0.5 → 8 < 90 < 300 < 1000(숫자 값)
    assert [m["id"] for m in res["matches"]] == ["500", "8", "90"]
    assert [round(m["score"], 6) for m in res["matches"]] == [1.5, 0.9, 0.9]
    assert retrieved == [5]  # top_k 3 + 경계 동점 2곡까지만(전체 6곡 중)


def test_hybrid_metadata_off_identifies_by_payload_song_id(mem):
    _fill_text(mem, TEXT_SONGS)
    res = mem.Index(TEXT_HYBRID_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], sparse_vector={"indices": [7], "values": [1.0]},
                                                  top_k=2, include_metadata=False)
    assert [(m["id"], m["metadata"]) for m in res["matches"]] == [("500", {}), ("8", {})]


# ---------------------------------------------------------------------------
# Context — 같은 곡 여러 사실·사실 동점·프로필 동점
# ---------------------------------------------------------------------------

def _context_searcher(tmp_path, client, song_ids):
    from tests.context.test_context_qdrant_load import FakeEmbedder, _hash, _load, _sources
    from src.retrieval.context_qdrant_search import ContextQdrantSearch

    sources = _sources(tmp_path, song_ids=song_ids)
    _load(tmp_path, sources, client)
    return ContextQdrantSearch(client, namespace="test", dense_dir=sources[1], sparse_dir=sources[2],
                               state_manifest=tmp_path / "state.json", text_embedder=FakeEmbedder(), hash_fn=_hash,
                               expected_dense_dim=4)


CONTEXT_SONGS = ("303", "1001", "22", "4004")


@pytest.mark.parametrize("seed", range(4))
def test_context_fact_ties_follow_song_then_record_id_for_any_backend_order(tmp_path, mem, monkeypatch, seed):
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    # 백엔드가 모든 사실에 같은 점수를 준다고 치고(정확한 동점), 반환 순서를 섞는다
    monkeypatch.setattr(mem.client, "query_points", AdversarialBackend(mem.client, seed, equal_scores=True))
    facts = searcher.search_dense_facts("테스트 프로그램 배경음악", limit=3)
    all_facts = sorted(
        ((str(p.payload["song_id"]), str(p.payload["record_id"])) for p in _all_points(mem, "context-dense")),
        key=lambda r: (_key(r[0]), r[1]),
    )
    assert [(f.song_id, f.record_id) for f in facts] == all_facts[:3]
    assert all_facts[0][0] == all_facts[1][0] == "22"  # 같은 곡 사실 2개가 record_id 순으로
    candidates = searcher.search_song_candidates("테스트 프로그램 배경음악", fact_k=3, sparse_k=3)
    assert [s.song_id for s in candidates.dense_songs] == ["22", "303"]
    assert candidates.dense_songs[0].best_fact.record_id == all_facts[0][1]  # 동점 사실 중 record_id가 작은 쪽
    assert [s.song_id for s in candidates.sparse_songs] == ["22", "303", "1001"]
    fused = searcher.search_fused_songs("테스트 프로그램 배경음악", fact_k=3, sparse_k=3)
    assert [h.song_id for h in fused] == ["22", "303", "1001"]
    assert [f.record_id for f in fused[0].dense_facts] == [all_facts[0][1], all_facts[1][1]]  # 근거 순서도 고정


def _all_points(client: QdrantVectorClient, prefix: str):
    names = [c.name for c in client.client.get_collections().collections if prefix in c.name]
    assert len(names) == 1
    points, _ = client.client.scroll(names[0], limit=1000, with_payload=True, with_vectors=False)
    return points


def test_context_real_scores_are_unchanged_by_the_ordering_layer(tmp_path, mem):
    """동점이 없으면 순서는 점수만으로 정해진다 — 이전 동작과 같다(기준: 저장 벡터로 직접 계산)."""
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    facts = searcher.search_dense_facts("테스트 프로그램 배경음악", limit=8)
    scores = [f.score for f in facts]
    assert scores == sorted(scores, reverse=True)
    keys = [(-f.score, _key(f.song_id), f.record_id) for f in facts]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 코드 리뷰 반영(2026-10-17) — 서버 정확 검색 · 피드백 사례 경계 동점 · k 밖 무효 점 · 하이브리드 중복 곡 · 경고 한 번
# ---------------------------------------------------------------------------

def _spy_kwargs(client: Any, monkeypatch) -> List[Dict[str, Any]]:
    real = client.query_points
    seen: List[Dict[str, Any]] = []

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(dict(kw))
        return real(*a, **kw)

    monkeypatch.setattr(client, "query_points", spy)
    return seen


def test_context_searches_request_exact_search_on_a_server_client(tmp_path, mem, monkeypatch):
    """서버(HNSW가 생길 수 있음)에서는 사실·프로필 조회가 정확 검색을 요청한다 — 경계 확인의 전제다."""
    mem.is_local = False  # 서버 클라이언트처럼(엔진은 메모리 — search_params는 경고만 남기고 무시된다)
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    seen = _spy_kwargs(mem.client, monkeypatch)
    searcher.search_song_candidates("테스트 프로그램 배경음악", fact_k=3, sparse_k=3)
    assert seen and all(kw.get("search_params") == models.SearchParams(exact=True) for kw in seen)


def test_context_searches_do_not_send_search_params_in_local_mode(tmp_path, mem, monkeypatch):
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    seen = _spy_kwargs(mem.client, monkeypatch)
    searcher.search_song_candidates("테스트 프로그램 배경음악", fact_k=3, sparse_k=3)
    assert seen and all(kw.get("search_params") is None for kw in seen)


class _WithCorruptPoint:
    """실제 결과 뒤(또는 앞)에 세대가 맞지 않는 점 하나를 정해진 점수로 끼운다."""

    def __init__(self, client: Any, score: float) -> None:
        self.real = client.query_points
        self.score = score

    def __call__(self, *a: Any, **kw: Any) -> Any:
        res = self.real(*a, **kw)
        bad = models.ScoredPoint(id=999999, version=0, score=self.score,
                                 payload={"context_build_id": "other-generation", "song_id": "1"})
        points = sorted(list(res.points) + [bad], key=lambda p: -p.score)
        res.points = points[: kw["limit"]]
        return res


def test_an_incompatible_point_below_the_k_th_score_does_not_fail_the_context_path(tmp_path, mem, monkeypatch):
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    facts = searcher.search_dense_facts("테스트 프로그램 배경음악", limit=8)
    floor = min(f.score for f in facts[:3])
    monkeypatch.setattr(mem.client, "query_points", _WithCorruptPoint(mem.client, floor - 1e-3))
    # 예전에는 limit=3이라 받지 않던 4번째 자리 — 경계 확인용으로 읽혀도 실패로 만들지 않는다
    got = searcher.search_dense_facts("테스트 프로그램 배경음악", limit=3)
    assert [f.record_id for f in got] == [f.record_id for f in facts[:3]]


def test_an_incompatible_point_inside_the_top_k_still_fails_as_before(tmp_path, mem, monkeypatch):
    searcher = _context_searcher(tmp_path, mem, CONTEXT_SONGS)
    monkeypatch.setattr(mem.client, "query_points", _WithCorruptPoint(mem.client, 10.0))
    with pytest.raises(RuntimeError, match="incompatible"):
        searcher.search_dense_facts("테스트 프로그램 배경음악", limit=3)


def test_hybrid_with_two_points_for_one_song_still_returns_top_k_distinct_songs(mem):
    collection = mem.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, with_sparse=True, recreate=True)
    rows = [(1, "10", [0.9, 0, 0]), (2, "10", [0.85, 0, 0]), (3, "20", [0.8, 0, 0]), (4, "30", [0.7, 0, 0]), (5, "40", [0.6, 0, 0])]
    mem.client.upsert(collection_name=collection, points=[
        models.PointStruct(id=pid, payload={"song_id": sid}, vector={DENSE_VECTOR: v, SPARSE_VECTOR: models.SparseVector(indices=[1], values=[0.1])})
        for pid, sid, v in rows])
    index = mem.Index(TEXT_HYBRID_INDEX_NAME)
    hybrid = index.query(vector=[1.0, 0, 0], sparse_vector={"indices": [1], "values": [1.0]}, top_k=3)
    assert [m["id"] for m in hybrid["matches"]] == ["10", "20", "30"]  # 중복 곡 때문에 2곡으로 줄지 않는다
    dense = index.query(vector=[1.0, 0, 0], top_k=3)
    assert [m["id"] for m in dense["matches"]] == ["10", "20", "30"]  # 단일 벡터 경로와 같은 규칙


def test_points_without_song_id_are_reported_once_per_query(mem, caplog):
    collection = mem.ensure_collection(IMAGE_INDEX_NAME, dim=3, recreate=True)
    mem.client.upsert(collection_name=collection, points=[
        models.PointStruct(id=i, vector={DENSE_VECTOR: _unit(1, 0.01 * i, 0)}, payload={"title": "x"}) for i in range(1, 40)
    ] + [models.PointStruct(id=100, vector={DENSE_VECTOR: _unit(0, 1, 0)}, payload={"song_id": "100"})])
    caplog.set_level("WARNING")
    res = mem.Index(IMAGE_INDEX_NAME).query(vector=[1.0, 0.0, 0.0], top_k=3)
    assert [m["id"] for m in res["matches"]] == ["100"]
    warnings = [r for r in caplog.records if "song_id" in r.getMessage()]
    assert len(warnings) == 1 and "39" in warnings[0].getMessage()


@pytest.mark.parametrize("seed", range(4))
def test_feedback_cases_tied_at_the_boundary_are_chosen_by_feedback_id(monkeypatch, seed):
    import time as _time
    from datetime import datetime, timedelta, timezone

    from src.feedback.case_index import FeedbackCaseIndex
    from src.feedback.review import CaseVersion
    from tests.feedback_fakes import TEST_DIM, memory_qdrant

    client = memory_qdrant()
    try:
        index = FeedbackCaseIndex(client, dim=TEST_DIM)
        index.ensure()
        same = [1.0] + [0.0] * (TEST_DIM - 1)
        expires = datetime.now(timezone.utc) + timedelta(days=3)
        ids = ["fb-k", "fb-c", "fb-x", "fb-a", "fb-m"]
        index.upsert([(CaseVersion(fid, f"s{i}", 1, 1), same, expires) for i, fid in enumerate(ids)], embedding_version="v")
        monkeypatch.setattr(client.client, "query_points", AdversarialBackend(client.client, seed))
        hits = index.search(same, top_k=2, now_ts=_time.time(), embedding_version="v", min_similarity=0.5)
        assert [h.feedback_id for h in hits] == ["fb-a", "fb-c"]
    finally:
        client.close()
