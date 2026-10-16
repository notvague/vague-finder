"""v61 — 하이브리드 합산의 조회 상한(`HYBRID_SCAN_LIMIT`)이 점수·후보를 누락시키는지, 작은 합성 자료로 결정적으로 확인한다.

기준값은 구현을 따라 쓰지 않는다 — 원본 벡터로 **직접 내적**을 계산한 값이다(`_exact`). 구현(`QdrantIndex.query`)에는 메모리
Qdrant에 같은 벡터를 넣고 모듈 상수 `HYBRID_SCAN_LIMIT`만 작게 바꿔 부른다. 상한이 자료 수 이상이면 기준과 같아야 하고,
작으면 어떤 모양으로 어긋나는지(한쪽 점수 누락 · 양쪽 후보 누락)를 그대로 보인다.

구분:
  - "현재 구현에서 실제로 일어나는 사례" — 상한 < 자료 수일 때 한쪽/양쪽 누락, 음수 dense에서 누락이 점수를 **올리는** 방향
  - "설계상 가능하지만 서비스 경로는 닿지 않는 사례" — 빈 sparse 질의(서비스는 sparse가 비면 sparse_vector를 넘기지 않는다)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest
from qdrant_client import models

from src.vector_db import qdrant_backend as qb
from src.vector_db.qdrant_backend import DENSE_VECTOR, SPARSE_VECTOR, QdrantVectorClient, point_id
from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

TOL = 1e-5

# (song_id, dense(3차원), {sparse 인덱스: 값}, payload 일부)
Song = Tuple[str, Sequence[float], Dict[int, float], Dict[str, object]]


def _exact(songs: Sequence[Song], dense_q: Sequence[float], sparse_q: Optional[Dict[str, Sequence]],
           allowed: Optional[set] = None) -> List[Tuple[str, float, float, float]]:
    """원본 벡터로 직접 계산한 (id, 합산, dense, sparse) — 합산 내림차순. 동점은 id 순(의미 없음, 재현용)."""
    sparse_terms = dict(zip(sparse_q.get("indices", []), sparse_q.get("values", []))) if sparse_q else {}
    rows = []
    for song_id, dense, sparse, _payload in songs:
        if allowed is not None and song_id not in allowed:
            continue
        d = sum(a * b for a, b in zip(dense, dense_q))
        s = sum(value * sparse_terms[index] for index, value in sparse.items() if index in sparse_terms)
        rows.append((song_id, d + s, d, s))
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def _fill(client: QdrantVectorClient, songs: Sequence[Song]) -> None:
    collection = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, with_sparse=True, recreate=True)
    client.client.upsert(collection_name=collection, points=[
        models.PointStruct(
            id=point_id(song_id),
            payload={"song_id": song_id, "title": f"T{song_id}", **payload},
            vector={
                DENSE_VECTOR: list(dense),
                SPARSE_VECTOR: models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
            },
        )
        for song_id, dense, sparse, payload in songs
    ])


@pytest.fixture()
def client():
    c = QdrantVectorClient(path=":memory:")
    yield c
    c.close()


def _query(client, dense_q, sparse_q, top_k, **kwargs):
    return client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=dense_q, sparse_vector=sparse_q, top_k=top_k, **kwargs)


def _scores(res) -> Dict[str, float]:
    return {m["id"]: m["score"] for m in res["matches"]}


# ---------------------------------------------------------------------------
# 자료 — dense는 x축에 가까운 정도, sparse는 인덱스 7의 값. 상한 L=3으로 자른다.
# ---------------------------------------------------------------------------

def _corpus() -> List[Song]:
    return [
        # id,  dense,              sparse,       payload
        # 질의 dense=[0.5,0,0] · sparse {7: 0.5}일 때 — dense 순위: 1>2>3>7>4>5>6, sparse(7) 순위: 4>5>6>7>2>1 (3은 미일치)
        # 직접 합산: 4=1.15 · 5=0.85 · 2=0.65 · 7=0.60 · 6=0.55 · 1=0.50 · 3=0.35 (동점 없음)
        ("1", [0.90, 0.10, 0.00], {7: 0.10},    {"g": "a"}),   # dense 1위 · sparse 6위
        ("2", [0.80, 0.20, 0.00], {7: 0.50},    {"g": "a"}),   # dense 2위 · sparse 5위
        ("3", [0.70, 0.30, 0.00], {8: 5.00},    {"g": "b"}),   # dense 3위, 질의 sparse(7)와 미일치
        ("4", [0.30, 0.90, 0.00], {7: 2.00},    {"g": "b"}),   # dense 5위(상한 3 밖) · sparse 1위 → 합산 1위
        ("5", [0.20, 0.00, 0.90], {7: 1.50},    {"g": "b"}),   # dense 6위(밖) · sparse 2위
        ("6", [0.10, 0.00, 0.95], {7: 1.00},    {"g": "c"}),   # dense 7위(밖) · sparse 3위
        ("7", [0.60, 0.50, 0.00], {7: 0.60},    {"g": "c"}),   # dense 4위 · sparse 4위 — 양쪽 상한 밖인데 합산은 4위(6·1·3보다 위)
    ]


def test_limit_at_least_corpus_matches_exact_sum(client, monkeypatch):
    """상한 ≥ 자료 수면 구현 = 직접 내적 합산(id·점수)."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    expected = _exact(songs, dense_q, sparse_q)
    res = _query(client, dense_q, sparse_q, top_k=len(songs))
    assert [m["id"] for m in res["matches"]] == [r[0] for r in expected]
    for (song_id, total, _d, _s) in expected:
        assert _scores(res)[song_id] == pytest.approx(total, abs=TOL)


def test_dense_portion_is_dropped_for_songs_outside_the_dense_limit(client, monkeypatch):
    """한쪽(dense) 목록에서 잘린 곡은 sparse 점수만 남는다 — 점수가 기준보다 작고 순위가 밀린다. (현재 구현에서 실제로 일어남)"""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    exact = {r[0]: r for r in _exact(songs, dense_q, sparse_q)}
    res = _query(client, dense_q, sparse_q, top_k=7)
    got = _scores(res)
    # 4번: dense 5위(상한 3 밖)·sparse 1위 → 구현은 sparse 몫(2.0*0.5=1.0)만, 기준은 1.0 + 0.15
    assert exact["4"][1] == pytest.approx(1.15, abs=TOL)
    assert got["4"] == pytest.approx(exact["4"][3], abs=TOL)           # sparse 몫만 남았다
    assert got["4"] < exact["4"][1] - 0.1                               # dense 몫(0.15)이 빠졌다
    # 반대쪽도 같다 — 1·2번은 dense 상위지만 sparse 순위(6·5위)가 상한 밖이라 sparse 몫이 빠진다
    for song_id in ("1", "2"):
        assert got[song_id] == pytest.approx(exact[song_id][2], abs=TOL)   # dense 몫만
        assert got[song_id] < exact[song_id][1]
    # 3번은 dense 상한 안이고 sparse는 애초에 미일치라 정확하다
    assert got["3"] == pytest.approx(exact["3"][1], abs=TOL)


def test_song_outside_both_lists_is_missing_even_when_its_sum_is_top(client, monkeypatch):
    """양쪽 개별 상위 목록에는 없지만 합산하면 상위권인 곡이 결과에서 **사라진다**. (현재 구현에서 실제로 일어남)"""
    songs = _corpus()
    _fill(client, songs)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    exact = _exact(songs, dense_q, sparse_q)
    exact_rank = {r[0]: i + 1 for i, r in enumerate(exact)}
    # 7번은 dense 0.30(4위: 1·2·3 다음) · sparse 0.30(4위: 4·5·6 다음) — 상한 3이면 양쪽 모두 밖. 직접 합산은 4위(0.60)다
    assert exact_rank["7"] == 4, exact

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)
    truncated = _query(client, dense_q, sparse_q, top_k=7)
    truncated_ids = [m["id"] for m in truncated["matches"]]
    assert "7" not in truncated_ids                       # 결과 전체(top_k=7)에서 사라졌다
    assert {"6", "1", "3"} <= set(truncated_ids)          # 7보다 낮은 곡들은 남아 있다 — 단순 하위 절단이 아니다

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    full = _query(client, dense_q, sparse_q, top_k=7)
    assert [m["id"] for m in full["matches"]][exact_rank["7"] - 1] == "7"


def test_dropped_negative_dense_portion_inflates_the_score(client, monkeypatch):
    """음수 dense 점수(DOT는 음수가 나온다)가 상한 밖으로 잘리면 합산이 기준보다 **커진다** — 누락이 순위를 올리는 방향. (설계상 가능 · 실제 색인은 음수 없음)"""
    songs: List[Song] = [
        ("1", [1.0, 0.0, 0.0], {7: 0.5}, {}),
        ("2", [0.5, 0.5, 0.0], {7: 0.5}, {}),
        ("3", [-1.0, 0.0, 0.0], {7: 3.0}, {}),   # dense 최하위(음수)·sparse 최상위
    ]
    _fill(client, songs)
    dense_q, sparse_q = [1.0, 0.0, 0.0], {"indices": [7], "values": [1.0]}
    exact = {r[0]: r for r in _exact(songs, dense_q, sparse_q)}
    assert exact["3"][1] == pytest.approx(2.0, abs=TOL)            # -1.0 + 3.0

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    full = _scores(_query(client, dense_q, sparse_q, top_k=3))
    assert full["3"] == pytest.approx(2.0, abs=TOL)                 # 상한이 넉넉하면 음수도 정확히 더해진다

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 2)
    truncated = _scores(_query(client, dense_q, sparse_q, top_k=3))
    assert truncated["3"] == pytest.approx(3.0, abs=TOL)            # 음수 몫이 빠져 2.0 → 3.0으로 부풀었다
    assert truncated["3"] > exact["3"][1]


def test_sparse_without_any_match_falls_back_to_dense_order(client, monkeypatch):
    """sparse 질의가 어느 곡과도 안 겹치면 합산 = dense. 상한이 충분하면 기준과 같다."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [99], "values": [0.5]}
    exact = _exact(songs, dense_q, sparse_q)
    res = _query(client, dense_q, sparse_q, top_k=7)
    assert [m["id"] for m in res["matches"]] == [r[0] for r in exact]
    assert all(r[3] == 0.0 for r in exact)


def test_empty_sparse_query_is_design_only_and_equals_dense(client, monkeypatch):
    """빈 sparse(indices=[])는 서비스 경로가 보내지 않는다(sparse가 비면 dense 전용). 백엔드에 직접 주면 dense 순서와 같다 — 설계상 사례."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    dense_q = [0.5, 0.0, 0.0]
    res = _query(client, dense_q, {"indices": [], "values": []}, top_k=7)
    assert [m["id"] for m in res["matches"]] == [r[0] for r in _exact(songs, dense_q, None)]


def test_filter_makes_the_limit_apply_to_the_filtered_set(client, monkeypatch):
    """필터가 있으면 상한은 **필터를 통과한 곡**에 적용된다 — 통과 수 ≤ 상한이면 전체가 작아도 정확하다."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)        # 전체 7곡보다 작다
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    allowed = {s[0] for s in songs if s[3].get("g") == "b"}  # 3·4·5 — 3곡 ≤ 상한
    exact = _exact(songs, dense_q, sparse_q, allowed=allowed)
    res = _query(client, dense_q, sparse_q, top_k=5, filter={"g": {"$eq": "b"}})
    assert [m["id"] for m in res["matches"]] == [r[0] for r in exact]
    for song_id, total, _d, _s in exact:
        assert _scores(res)[song_id] == pytest.approx(total, abs=TOL)


def test_filter_with_no_match_returns_empty(client, monkeypatch):
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)
    res = _query(client, [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}, top_k=5, filter={"g": {"$eq": "zzz"}})
    assert res == {"matches": []}


def test_top_k_larger_than_corpus_returns_everything_once(client, monkeypatch):
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    res = _query(client, [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}, top_k=50)
    ids = [m["id"] for m in res["matches"]]
    assert sorted(ids) == sorted(s[0] for s in songs)
    assert len(ids) == len(set(ids))


def test_ties_keep_both_songs_with_equal_scores_and_do_not_lose_either(client, monkeypatch):
    """동점 곡은 둘 다 남고 점수가 같다. 동점 안의 순서는 정의하지 않으므로 단언하지 않는다."""
    songs: List[Song] = [
        ("1", [1.0, 0.0, 0.0], {7: 1.0}, {}),
        ("2", [1.0, 0.0, 0.0], {7: 1.0}, {}),   # 1과 완전히 같은 벡터
        ("3", [0.0, 1.0, 0.0], {8: 1.0}, {}),
    ]
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", len(songs))
    res = _query(client, [1.0, 0.0, 0.0], {"indices": [7], "values": [1.0]}, top_k=3)
    got = _scores(res)
    assert got["1"] == pytest.approx(got["2"], abs=TOL) == pytest.approx(2.0, abs=TOL)
    assert set(m["id"] for m in res["matches"][:2]) == {"1", "2"}


def test_the_limit_is_read_from_the_environment_name_used_in_docs(monkeypatch):
    """운영 문서·.env.example의 이름(`QDRANT_HYBRID_SCAN_LIMIT`)과 모듈 상수가 같은 값을 가리킨다 — 진단이 가정한 설정 경로 확인."""
    import importlib

    monkeypatch.setenv("QDRANT_HYBRID_SCAN_LIMIT", "7")
    try:
        assert importlib.reload(qb).HYBRID_SCAN_LIMIT == 7
    finally:
        # 환경을 먼저 되돌린 뒤 다시 읽어 다른 테스트에 7이 남지 않게 한다. 되돌린 값은 실행 환경에 따르므로 여기서 단언하지 않는다
        monkeypatch.undo()
        importlib.reload(qb)


def test_the_default_limit_is_5000_when_the_environment_variable_is_absent(monkeypatch):
    """변수가 없을 때의 기본값이 진단이 전제한 5000이다. 셸·CI에 변수가 설정돼 있어도 지우고 재서 영향을 받지 않는다."""
    import importlib

    monkeypatch.delenv("QDRANT_HYBRID_SCAN_LIMIT", raising=False)
    try:
        assert importlib.reload(qb).HYBRID_SCAN_LIMIT == 5000
    finally:
        monkeypatch.undo()
        importlib.reload(qb)
