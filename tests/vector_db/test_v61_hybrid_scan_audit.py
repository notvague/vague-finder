"""v61 → v62 — 하이브리드 합산이 **옛 조회 상한에 더는 묶이지 않음**을 작은 합성 자료로 결정적으로 확인한다.

기준값은 구현을 따라 쓰지 않는다 — 원본 벡터로 **직접 내적**을 계산한 값이다(`_exact`). 구현(`QdrantIndex.query`)에는 메모리
Qdrant에 같은 벡터를 넣고 부른다. v62부터 두 조회의 limit은 요청마다 `count(exact)`로 센 대상 수이므로, 옛 상수
`HYBRID_SCAN_LIMIT`을 자료 수보다 작게 둬도(여기서는 monkeypatch로 3·2) 결과가 기준과 같아야 한다.

**옛 결함의 재현은 v61 기준 커밋에 보존돼 있다** — `git show fc20e86:tests/test_v61_hybrid_scan_audit.py`(PR #76, 2026-10-16).
그 판에서는 같은 자료·같은 `_exact` 기준으로 상한 3일 때 4번 곡의 dense 몫(0.15)이 빠져 1.15 → 1.00이 되고, 7번 곡이 결과에서
사라지며, 음수 dense 곡의 합산이 2.0 → 3.0으로 부풀었다. 여기서는 그 세 사례가 **기준과 같아졌는지**를 같은 숫자로 단언한다.

구분:
  - "옛 구현에서 실제로 일어났던 사례" — 한쪽 점수 누락 · 양쪽 목록 밖 곡 소실 · 음수 dense 부풀기 → 이제 기준과 같음
  - "설계상 가능하지만 서비스 경로는 닿지 않는 사례" — 빈 sparse 질의(서비스는 sparse가 비면 sparse_vector를 넘기지 않는다)
"""
from __future__ import annotations

import importlib
import logging
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


def _assert_matches_exact(res, expected: List[Tuple[str, float, float, float]]) -> None:
    assert [m["id"] for m in res["matches"]] == [r[0] for r in expected]
    got = _scores(res)
    for song_id, total, _d, _s in expected:
        assert got[song_id] == pytest.approx(total, abs=TOL)


# ---------------------------------------------------------------------------
# 자료 — dense는 x축에 가까운 정도, sparse는 인덱스 7의 값. 옛 상한 L=3이면 양쪽 목록이 잘리던 자료다.
# ---------------------------------------------------------------------------

def _corpus() -> List[Song]:
    return [
        # id,  dense,              sparse,       payload
        # 질의 dense=[0.5,0,0] · sparse {7: 0.5}일 때 — dense 순위: 1>2>3>7>4>5>6, sparse(7) 순위: 4>5>6>7>2>1 (3은 미일치)
        # 직접 합산: 4=1.15 · 5=0.85 · 2=0.65 · 7=0.60 · 6=0.55 · 1=0.50 · 3=0.35 (동점 없음)
        ("1", [0.90, 0.10, 0.00], {7: 0.10},    {"g": "a"}),   # dense 1위 · sparse 6위
        ("2", [0.80, 0.20, 0.00], {7: 0.50},    {"g": "a"}),   # dense 2위 · sparse 5위
        ("3", [0.70, 0.30, 0.00], {8: 5.00},    {"g": "b"}),   # dense 3위, 질의 sparse(7)와 미일치
        ("4", [0.30, 0.90, 0.00], {7: 2.00},    {"g": "b"}),   # dense 5위(옛 상한 3 밖) · sparse 1위 → 합산 1위
        ("5", [0.20, 0.00, 0.90], {7: 1.50},    {"g": "b"}),   # dense 6위(밖) · sparse 2위
        ("6", [0.10, 0.00, 0.95], {7: 1.00},    {"g": "c"}),   # dense 7위(밖) · sparse 3위
        ("7", [0.60, 0.50, 0.00], {7: 0.60},    {"g": "c"}),   # dense 4위 · sparse 4위 — 옛 상한 3이면 양쪽 밖인데 합산은 4위
    ]


def test_implementation_matches_exact_sum_without_any_limit_setting(client):
    """상수를 건드리지 않은 기본 상태 — 구현 = 직접 내적 합산(id·점수)."""
    songs = _corpus()
    _fill(client, songs)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    _assert_matches_exact(_query(client, dense_q, sparse_q, top_k=len(songs)), _exact(songs, dense_q, sparse_q))


def test_songs_outside_the_old_dense_limit_keep_their_dense_portion(client, monkeypatch):
    """옛 결함 ① 한쪽(dense) 목록에서 잘린 곡은 sparse 점수만 남았다(4번 1.15 → 1.00). 이제 옛 상수를 3으로 둬도 전부 기준과 같다."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    exact = {r[0]: r for r in _exact(songs, dense_q, sparse_q)}
    got = _scores(_query(client, dense_q, sparse_q, top_k=7))
    assert exact["4"][1] == pytest.approx(1.15, abs=TOL)
    assert got["4"] == pytest.approx(1.15, abs=TOL)                 # 옛 구현은 1.00(sparse 몫만)이었다
    assert got["4"] == pytest.approx(exact["4"][2] + exact["4"][3], abs=TOL)
    for song_id in ("1", "2"):                                        # 반대쪽(sparse 순위가 옛 상한 밖)도 같다
        assert got[song_id] == pytest.approx(exact[song_id][1], abs=TOL)
        assert got[song_id] > exact[song_id][2]                       # dense 몫만이 아니다
    assert got["3"] == pytest.approx(exact["3"][1], abs=TOL)


def test_song_outside_both_old_lists_is_kept_at_its_exact_rank(client, monkeypatch):
    """옛 결함 ② 양쪽 개별 상위 목록에 없지만 합산은 상위권인 곡(7번, 합산 4위)이 결과에서 사라졌다. 이제 정확한 자리에 있다."""
    songs = _corpus()
    _fill(client, songs)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    exact = _exact(songs, dense_q, sparse_q)
    exact_rank = {r[0]: i + 1 for i, r in enumerate(exact)}
    assert exact_rank["7"] == 4, exact

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 3)
    res = _query(client, dense_q, sparse_q, top_k=7)
    ids = [m["id"] for m in res["matches"]]
    assert ids[exact_rank["7"] - 1] == "7"
    _assert_matches_exact(res, exact)


def test_negative_dense_portion_is_summed_exactly_even_with_a_small_old_limit(client, monkeypatch):
    """옛 결함 ③ 음수 dense(DOT)가 상한 밖으로 잘리면 합산이 2.0 → 3.0으로 **부풀었다**. 이제 옛 상수가 2여도 2.0이다."""
    songs: List[Song] = [
        ("1", [1.0, 0.0, 0.0], {7: 0.5}, {}),
        ("2", [0.5, 0.5, 0.0], {7: 0.5}, {}),
        ("3", [-1.0, 0.0, 0.0], {7: 3.0}, {}),   # dense 최하위(음수)·sparse 최상위
    ]
    _fill(client, songs)
    dense_q, sparse_q = [1.0, 0.0, 0.0], {"indices": [7], "values": [1.0]}
    exact = {r[0]: r for r in _exact(songs, dense_q, sparse_q)}
    assert exact["3"][1] == pytest.approx(2.0, abs=TOL)            # -1.0 + 3.0

    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 2)
    got = _scores(_query(client, dense_q, sparse_q, top_k=3))
    assert got["3"] == pytest.approx(2.0, abs=TOL)                  # 옛 구현은 3.0
    assert got["3"] == pytest.approx(exact["3"][1], abs=TOL)


def test_sparse_without_any_match_falls_back_to_dense_order(client):
    """sparse 질의가 어느 곡과도 안 겹치면 합산 = dense."""
    songs = _corpus()
    _fill(client, songs)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [99], "values": [0.5]}
    exact = _exact(songs, dense_q, sparse_q)
    assert all(r[3] == 0.0 for r in exact)
    _assert_matches_exact(_query(client, dense_q, sparse_q, top_k=7), exact)


def test_empty_sparse_query_is_design_only_and_equals_dense(client):
    """빈 sparse(indices=[])는 서비스 경로가 보내지 않는다(sparse가 비면 dense 전용). 백엔드에 직접 주면 dense 순서와 같다 — 설계상 사례."""
    songs = _corpus()
    _fill(client, songs)
    dense_q = [0.5, 0.0, 0.0]
    res = _query(client, dense_q, {"indices": [], "values": []}, top_k=7)
    assert [m["id"] for m in res["matches"]] == [r[0] for r in _exact(songs, dense_q, None)]


def test_filter_restricts_the_sum_to_the_filtered_songs(client, monkeypatch):
    """필터가 있으면 대상 수는 **필터를 통과한 곡**이고 합산도 그 곡들로만 — 옛 상수가 작아도 정확하다."""
    songs = _corpus()
    _fill(client, songs)
    monkeypatch.setattr(qb, "HYBRID_SCAN_LIMIT", 1)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    allowed = {s[0] for s in songs if s[3].get("g") == "b"}  # 3·4·5
    exact = _exact(songs, dense_q, sparse_q, allowed=allowed)
    _assert_matches_exact(_query(client, dense_q, sparse_q, top_k=5, filter={"g": {"$eq": "b"}}), exact)


def test_filter_with_no_match_returns_empty(client):
    songs = _corpus()
    _fill(client, songs)
    res = _query(client, [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}, top_k=5, filter={"g": {"$eq": "zzz"}})
    assert res == {"matches": []}


def test_top_k_larger_than_corpus_returns_everything_once(client):
    songs = _corpus()
    _fill(client, songs)
    res = _query(client, [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}, top_k=50)
    ids = [m["id"] for m in res["matches"]]
    assert sorted(ids) == sorted(s[0] for s in songs)
    assert len(ids) == len(set(ids))


def test_ties_keep_both_songs_with_equal_scores_and_do_not_lose_either(client):
    """동점 곡은 둘 다 남고 점수가 같다. 동점 안의 순서는 정의하지 않으므로 단언하지 않는다."""
    songs: List[Song] = [
        ("1", [1.0, 0.0, 0.0], {7: 1.0}, {}),
        ("2", [1.0, 0.0, 0.0], {7: 1.0}, {}),   # 1과 완전히 같은 벡터
        ("3", [0.0, 1.0, 0.0], {8: 1.0}, {}),
    ]
    _fill(client, songs)
    res = _query(client, [1.0, 0.0, 0.0], {"indices": [7], "values": [1.0]}, top_k=3)
    got = _scores(res)
    assert got["1"] == pytest.approx(got["2"], abs=TOL) == pytest.approx(2.0, abs=TOL)
    assert set(m["id"] for m in res["matches"][:2]) == {"1", "2"}


# ---------------------------------------------------------------------------
# 설정 경로 — `QDRANT_HYBRID_SCAN_LIMIT`은 폐기 예정 호환값이다. 읽히지만 조회를 자르지 않는다.
# ---------------------------------------------------------------------------

def test_the_old_environment_variable_is_read_as_a_compat_value_but_does_not_truncate(monkeypatch, caplog):
    """옛 이름으로 2를 주고 모듈을 다시 읽으면 상수는 2(호환값)지만, 7곡 하이브리드 결과는 기준과 같다. 경고는 import 때 한 번."""
    monkeypatch.setenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV, "2")
    caplog.set_level(logging.WARNING, logger=qb.__name__)
    try:
        reloaded = importlib.reload(qb)
        assert reloaded.HYBRID_SCAN_LIMIT == 2
        warnings = [r for r in caplog.records if qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV in r.getMessage()]
        assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
        caplog.clear()
        c = reloaded.QdrantVectorClient(path=":memory:")
        try:
            songs = _corpus()
            _fill(c, songs)
            dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
            for _ in range(3):   # 요청을 반복해도 경고가 더 쌓이지 않는다
                _assert_matches_exact(_query(c, dense_q, sparse_q, top_k=7), _exact(songs, dense_q, sparse_q))
            assert not [r for r in caplog.records if qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV in r.getMessage()]
        finally:
            c.close()
    finally:
        # 환경을 먼저 되돌린 뒤 다시 읽어 다른 테스트에 호환값이 남지 않게 한다. 되돌린 값은 실행 환경에 따르므로 단언하지 않는다
        monkeypatch.undo()
        importlib.reload(qb)


def test_without_the_old_environment_variable_there_is_no_compat_value_and_no_warning(monkeypatch, caplog):
    """변수가 없으면 호환값은 None이고 경고도 없다. 셸·CI에 변수가 설정돼 있어도 지우고 재서 영향을 받지 않는다."""
    monkeypatch.delenv(qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV, raising=False)
    caplog.set_level(logging.WARNING, logger=qb.__name__)
    try:
        assert importlib.reload(qb).HYBRID_SCAN_LIMIT is None
        assert not [r for r in caplog.records if qb.DEPRECATED_HYBRID_SCAN_LIMIT_ENV in r.getMessage()]
    finally:
        monkeypatch.undo()
        importlib.reload(qb)
