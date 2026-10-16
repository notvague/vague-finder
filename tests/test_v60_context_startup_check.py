"""v60 — Context 색인·별칭 **기동 검사**(2026-10-16).

v56 운영 검증에서 서버 Qdrant로 옮기며 Context 별칭 2개를 빠뜨렸는데 서비스는 정상으로 뜨고
검색마다 경고 로그만 남긴 채 Context 없이 돌았다. 여기서 고정하는 것:

- 저수준 검색기 `ContextQdrantSearch.verify_publication()` — 설정 오류는 `ContextPublicationError`
  (+ `reason` 코드), 연결 장애는 클라이언트의 예외형 그대로. 모델을 만들지 않는다.
- 기동 모듈 `src.backend.context_startup.run()` — disabled / verified / config_error(기동 실패) /
  unavailable 을 섞지 않는다. 비활성이면 Qdrant에 묻지 않는다.
- 실제 lifespan — 검사가 돌고, 설정 오류면 기동이 실패하며 **로컬 Qdrant 잠금을 풀어** 같은
  프로세스에서 다시 띄울 수 있다. 예열을 꺼도 검사는 돈다. `/health.context`가 저장 결과와 같다.
- 정상 환경의 Context 후보 ID·순서는 검사 전후 같다.

모든 Qdrant는 `:memory:` 또는 임시 폴더, 발행 자료는 합성(tests/context/test_context_qdrant_load 도구).
Vertex·Mongo·모델 호출은 없다(conftest가 Vertex 확인을 대체하고, 여기서 피드백 색인·작업 스레드도 대체).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from src.backend import context_startup
from src.backend.context_startup import ContextStartupError
from src.retrieval.context_qdrant_search import (
    ContextPublicationError,
    ContextPublicationStatus,
    ContextQdrantSearch,
)
from src.vector_db.context_qdrant import _manifest_content_digest
from src.vector_db.qdrant_backend import QdrantVectorClient
from tests.context.test_context_qdrant_load import FakeEmbedder, _hash, _load, _sources

NAMESPACE = "test"
DENSE_ALIAS = "vaguefinder-context-dense-1024-koe5__test"
SPARSE_ALIAS = "vaguefinder-context-sparse-bm25__test"
QUERIES = ("테스트 프로그램 배경음악", "프로듀서가 데모를 다시 녹음", "삽입곡")


# ---------------------------------------------------------------------------
# 도구
# ---------------------------------------------------------------------------

def _publish(tmp_path: Path, client: QdrantVectorClient, song_ids=("101", "202")):
    sources = _sources(tmp_path, song_ids=song_ids)
    result = _load(tmp_path, sources, client)
    state = tmp_path / "state.json"
    manifest = json.loads(state.read_text(encoding="utf-8"))
    return SimpleNamespace(client=client, sources=sources, state=state, result=result,
                           manifest=manifest, tmp=tmp_path)


@pytest.fixture
def published(tmp_path: Path):
    client = QdrantVectorClient(path=":memory:")
    pub = _publish(tmp_path, client)
    yield pub
    client.close()


def _searcher(pub, client=None, *, dense_dir=None, sparse_dir=None, state=None, embedder=None):
    kwargs = {"text_embedder": embedder} if embedder is not None else {}
    return ContextQdrantSearch(
        client or pub.client, namespace=NAMESPACE,
        dense_dir=dense_dir or pub.sources[1], sparse_dir=sparse_dir or pub.sources[2],
        state_manifest=state or pub.state, hash_fn=_hash, expected_dense_dim=4, **kwargs,
    )


def _delete_alias(client: QdrantVectorClient, alias: str) -> None:
    client.client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
    ])


def _point_alias(client: QdrantVectorClient, alias: str, collection: str) -> None:
    ops = []
    if alias in {a.alias_name for a in client.client.get_aliases().aliases}:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(
        alias_name=alias, collection_name=collection)))
    client.client.update_collection_aliases(change_aliases_operations=ops)


def _rewrite_state(path: Path, mutate) -> None:
    """상태 manifest를 고치고 내용 해시를 다시 맞춘다(해시 불일치가 아닌 **내용** 불일치를 시험할 때)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    data.pop("manifest_content_sha256", None)
    data["manifest_content_sha256"] = _manifest_content_digest(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class _RawProxy:
    """진짜 로컬 Qdrant 위에 얹는 얇은 대역 — 호출 수를 세고, 지정한 메서드만 바꿔 끼운다."""

    def __init__(self, raw, **overrides):
        self._raw = raw
        self._overrides = overrides
        self.calls: list[str] = []

    def __getattr__(self, name):
        self.calls.append(name)
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._raw, name)


class _ProxyVectorClient:
    def __init__(self, vector_client: QdrantVectorClient, **overrides):
        self.client = _RawProxy(vector_client.client, **overrides)
        self.is_local = vector_client.is_local

    def close(self) -> None:
        pass


def _raise_connection(*_args, **_kwargs):
    raise ResponseHandlingException("http://qdrant-internal:6333 connection refused api_key=secret")


def _fused_ids(searcher: ContextQdrantSearch) -> dict[str, list[str]]:
    return {q: [hit.song_id for hit in searcher.search_fused_songs(q, fact_k=4, sparse_k=2)]
            for q in QUERIES}


# ---------------------------------------------------------------------------
# 1. verify_publication — 저수준 검색기 경계
# ---------------------------------------------------------------------------

def test_verified_status_describes_the_published_pair_without_building_a_model(published):
    searcher = _searcher(published)
    status = searcher.verify_publication()
    q = published.manifest["qdrant"]
    assert isinstance(status, ContextPublicationStatus)
    assert status.dense_alias == DENSE_ALIAS and status.sparse_alias == SPARSE_ALIAS
    assert status.dense_collection == q["dense_collection"]
    assert status.sparse_collection == q["sparse_collection"]
    assert status.build_id == published.manifest["context_build_id"]
    assert status.dense_dimension == 4
    # KoE5 임베더는 만들지 않는다 — 검사는 메타데이터 읽기만
    assert searcher._embedder is None and searcher._embedder_build_id is None


def test_metadata_only_no_point_scan_or_search(published):
    proxy = _ProxyVectorClient(published.client)
    _searcher(published, client=proxy).verify_publication()
    assert "scroll" not in proxy.client.calls
    assert "query_points" not in proxy.client.calls
    assert "retrieve" not in proxy.client.calls
    assert "count" not in proxy.client.calls
    assert "update_collection_aliases" not in proxy.client.calls
    assert "create_collection" not in proxy.client.calls


def test_unpublished_catalogue_is_an_empty_search_but_a_startup_config_error(tmp_path):
    """기존 계약 유지: 별칭도 manifest도 없는 환경은 검색에는 빈 결과. 기동 검사에는 설정 오류."""
    client = QdrantVectorClient(path=":memory:")
    try:
        searcher = ContextQdrantSearch(
            client, namespace=NAMESPACE, dense_dir=tmp_path / "d", sparse_dir=tmp_path / "s",
            state_manifest=tmp_path / "none.json", expected_dense_dim=4,
        )
        assert searcher.search("아무 질의", dense_k=2, sparse_k=2).dense_facts == ()
        with pytest.raises(ContextPublicationError) as info:
            searcher.verify_publication()
        assert info.value.reason == "not_published"
    finally:
        client.close()


def test_both_aliases_missing_with_manifest_is_a_config_error(published):
    _delete_alias(published.client, DENSE_ALIAS)
    _delete_alias(published.client, SPARSE_ALIAS)
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "aliases_missing"
    # 검색 경로도 같은 예외형(RuntimeError 호환)으로 떨어진다 — 라우터는 이것을 빈 결과로 바꾼다
    with pytest.raises(RuntimeError):
        _searcher(published).search("질의", dense_k=2, sparse_k=2)


@pytest.mark.parametrize("missing", [DENSE_ALIAS, SPARSE_ALIAS])
def test_one_alias_missing_is_a_config_error(published, missing):
    _delete_alias(published.client, missing)
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "aliases_partial"


def test_alias_pointing_at_another_collection_is_a_config_error(published):
    published.client.client.create_collection(
        "stray__gen", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.DOT)},
    )
    _point_alias(published.client, DENSE_ALIAS, "stray__gen")
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "publication_mismatch"


def test_alias_target_collection_missing_is_a_config_error(published):
    """서버가 별칭은 남겼는데 컬렉션이 없다(수동 삭제·부분 이전). 로컬 Qdrant는 함께 지우므로 대역으로 재현."""
    target = published.manifest["qdrant"]["dense_collection"]
    real_exists = published.client.client.collection_exists
    proxy = _ProxyVectorClient(
        published.client, collection_exists=lambda name: False if name == target else real_exists(name),
    )
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published, client=proxy).verify_publication()
    assert info.value.reason == "dense_collection_missing"


def test_alias_target_with_wrong_vector_schema_is_a_config_error(published):
    published.client.client.create_collection(
        "wrongdim__gen", vectors_config={"dense": models.VectorParams(size=8, distance=models.Distance.DOT)},
    )
    _point_alias(published.client, DENSE_ALIAS, "wrongdim__gen")
    _rewrite_state(published.state, lambda d: d["qdrant"].__setitem__("dense_collection", "wrongdim__gen"))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "dense_collection_schema_mismatch"


def test_sparse_target_with_dense_schema_is_a_config_error(published):
    published.client.client.create_collection(
        "notsparse__gen", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.DOT)},
    )
    _point_alias(published.client, SPARSE_ALIAS, "notsparse__gen")
    _rewrite_state(published.state, lambda d: d["qdrant"].__setitem__("sparse_collection", "notsparse__gen"))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "sparse_collection_schema_mismatch"


def test_state_manifest_missing_is_a_config_error(published):
    published.state.unlink()
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "manifest_missing"


def test_state_manifest_corrupt_is_a_config_error(published):
    published.state.write_text("{not json", encoding="utf-8")
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "manifest_invalid"


def test_state_manifest_hash_mismatch_is_a_config_error(published):
    data = json.loads(published.state.read_text(encoding="utf-8"))
    data["qdrant"]["namespace"] = "edited-without-rehash"
    published.state.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "manifest_hash_mismatch"


@pytest.mark.parametrize("field, value", [
    ("dense_vector_name", "vector"),
    ("sparse_vector_name", "bm25"),
    ("dense_dimension", 1024),
    ("payload_schema_version", "context_qdrant_payload_v0"),
    ("dense_alias", "vaguefinder-context-dense-1024-koe5__prod"),
])
def test_state_manifest_content_mismatch_is_a_config_error(published, field, value):
    _rewrite_state(published.state, lambda d: d["qdrant"].__setitem__(field, value))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "publication_mismatch"


def test_state_manifest_not_ok_status_is_a_config_error(published):
    _rewrite_state(published.state, lambda d: d.__setitem__("status", "failed"))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "publication_mismatch"


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("qdrant", None),
    lambda d: d.__setitem__("qdrant", "dense"),
    lambda d: d.__setitem__("source_hashes", []),
    lambda d: d.__setitem__("source_hashes", None),
    lambda d: d.__setitem__("context_build_id", 123),
    lambda d: d.__setitem__("context_build_id", ""),
    lambda d: d.__setitem__("context_build_id", None),
    lambda d: d["source_hashes"].__setitem__("dense_manifest_content_sha256", 42),
    lambda d: d["source_hashes"].__setitem__("sparse_manifest_content_sha256", ["x"]),
], ids=["qdrant_null", "qdrant_str", "hashes_list", "hashes_null", "build_id_int", "build_id_empty",
        "build_id_null", "dense_sha_int", "sparse_sha_list"])
def test_state_manifest_with_wrong_nested_types_is_a_config_error_not_a_crash(published, mutate):
    """리뷰(P2): 해시를 맞춘 manifest의 중첩 자료형 오류가 AttributeError/TypeError로 새면 기동 모듈이
    연결 장애(unavailable)로 잘못 분류해 기동을 허용한다. 설정 오류형으로 올라와야 한다."""
    _rewrite_state(published.state, mutate)
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "manifest_invalid"
    # 검색 경로도 같은 형으로(RuntimeError 호환) 떨어진다
    with pytest.raises(ContextPublicationError):
        _searcher(published).search("질의", dense_k=2, sparse_k=2)


@pytest.mark.parametrize("value", [None, "koe5", []])
def test_dense_manifest_with_wrong_embedding_block_is_a_config_error(published, value):
    _rewrite_state(published.sources[1] / "manifest.json", lambda d: d.__setitem__("embedding", value))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published).verify_publication()
    assert info.value.reason == "manifest_invalid"
    with pytest.raises(ContextPublicationError):
        _searcher(published, embedder=FakeEmbedder()).search("질의", dense_k=2, sparse_k=2)


def test_local_dense_artifact_from_another_build_is_a_config_error(published, tmp_path):
    other = _sources(tmp_path / "other", song_ids=("101", "202", "303"))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published, dense_dir=other[1]).verify_publication()
    assert info.value.reason == "dense_artifact_mismatch"


def test_local_sparse_artifact_from_another_build_is_a_config_error(published, tmp_path):
    other = _sources(tmp_path / "other", song_ids=("101", "202", "303"))
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published, sparse_dir=other[2]).verify_publication()
    assert info.value.reason == "sparse_artifact_mismatch"


def test_missing_local_artifacts_are_config_errors(published, tmp_path):
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published, dense_dir=tmp_path / "no-dense").verify_publication()
    assert info.value.reason == "manifest_missing"
    with pytest.raises(ContextPublicationError) as info:
        _searcher(published, sparse_dir=tmp_path / "no-sparse").verify_publication()
    assert info.value.reason == "sparse_artifact_invalid"


def test_connection_failure_keeps_the_clients_exception_type(published):
    proxy = _ProxyVectorClient(published.client, get_aliases=_raise_connection)
    with pytest.raises(ResponseHandlingException):
        _searcher(published, client=proxy).verify_publication()


def test_connection_failure_after_aliases_is_still_not_a_config_error(published):
    proxy = _ProxyVectorClient(published.client, collection_exists=_raise_connection)
    with pytest.raises(ResponseHandlingException):
        _searcher(published, client=proxy).verify_publication()


def test_candidate_ids_and_order_are_unchanged_by_the_check(published):
    """정상 환경에서 검사는 읽기만 한다 — 같은 검색기도, 새 검색기도 같은 후보·순서를 낸다."""
    baseline = _fused_ids(_searcher(published, embedder=FakeEmbedder()))
    assert any(baseline.values()), baseline
    searcher = _searcher(published, embedder=FakeEmbedder())
    before = _fused_ids(searcher)
    searcher.verify_publication()
    after = _fused_ids(searcher)
    assert before == after == baseline
    # 검사 전용 인스턴스(임베더 없음)가 같은 클라이언트를 쓴 뒤에도 검색 결과는 그대로다
    _searcher(published).verify_publication()
    assert _fused_ids(_searcher(published, embedder=FakeEmbedder())) == baseline


# ---------------------------------------------------------------------------
# 2. context_startup.run — 상태 구분
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_startup_state():
    context_startup.reset()
    yield
    context_startup.reset()


def _factory_for(pub):
    return lambda client: _searcher(pub, client=client)


def test_disabled_context_makes_no_qdrant_call(published, monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0")
    proxy = _ProxyVectorClient(published.client)
    called = []

    def factory(client):
        called.append(client)
        return _searcher(published, client=client)

    result = context_startup.run(proxy, searcher_factory=factory)
    assert result["state"] == context_startup.DISABLED
    assert result["reason"] == "context_weight_zero"
    assert proxy.client.calls == [] and called == []
    assert context_startup.status()["state"] == context_startup.DISABLED


def test_enabled_and_correct_publication_is_verified(published, monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    result = context_startup.run(published.client, searcher_factory=_factory_for(published))
    assert result["state"] == context_startup.VERIFIED
    assert result["source"] == "startup" and result["checked_at"] and result["note"]
    q = published.manifest["qdrant"]
    assert result["publication"] == {
        "dense_alias": DENSE_ALIAS, "sparse_alias": SPARSE_ALIAS,
        "dense_collection": q["dense_collection"], "sparse_collection": q["sparse_collection"],
        "context_build_id": published.manifest["context_build_id"], "dense_dimension": 4,
    }
    assert context_startup.status() == result


def test_default_weight_counts_as_enabled(published, monkeypatch):
    monkeypatch.delenv("CONTEXT_WEIGHT", raising=False)
    assert context_startup.enabled() is True
    result = context_startup.run(published.client, searcher_factory=_factory_for(published))
    assert result["state"] == context_startup.VERIFIED


@pytest.mark.parametrize("break_it, reason", [
    (lambda pub: (_delete_alias(pub.client, DENSE_ALIAS), _delete_alias(pub.client, SPARSE_ALIAS)), "aliases_missing"),
    (lambda pub: _delete_alias(pub.client, SPARSE_ALIAS), "aliases_partial"),
    (lambda pub: pub.state.unlink(), "manifest_missing"),
    (lambda pub: pub.state.write_text("{", encoding="utf-8"), "manifest_invalid"),
    (lambda pub: _rewrite_state(pub.state, lambda d: d["qdrant"].__setitem__("dense_vector_name", "x")), "publication_mismatch"),
])
def test_config_errors_fail_startup_with_a_reason(published, monkeypatch, break_it, reason):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    break_it(published)
    with pytest.raises(ContextStartupError) as info:
        context_startup.run(published.client, searcher_factory=_factory_for(published))
    assert info.value.reason == reason
    status = context_startup.status()
    assert status["state"] == context_startup.CONFIG_ERROR and status["reason"] == reason


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("qdrant", None),
    lambda d: d.__setitem__("source_hashes", []),
    lambda d: d.__setitem__("context_build_id", 123),
], ids=["qdrant_null", "hashes_list", "build_id_int"])
def test_manifest_type_errors_fail_startup_as_config_errors_not_unavailable(published, monkeypatch, mutate):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    _rewrite_state(published.state, mutate)
    with pytest.raises(ContextStartupError) as info:
        context_startup.run(published.client, searcher_factory=_factory_for(published))
    assert info.value.reason == "manifest_invalid"
    assert context_startup.status()["state"] == context_startup.CONFIG_ERROR


def test_unpublished_pair_fails_startup_when_context_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    client = QdrantVectorClient(path=":memory:")
    try:
        factory = lambda c: ContextQdrantSearch(  # noqa: E731
            c, namespace=NAMESPACE, dense_dir=tmp_path / "d", sparse_dir=tmp_path / "s",
            state_manifest=tmp_path / "none.json", expected_dense_dim=4,
        )
        with pytest.raises(ContextStartupError) as info:
            context_startup.run(client, searcher_factory=factory)
        assert info.value.reason == "not_published"
    finally:
        client.close()


def test_connection_failure_is_unavailable_not_config_error_and_does_not_fail_startup(published, monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    proxy = _ProxyVectorClient(published.client, get_aliases=_raise_connection)
    result = context_startup.run(proxy, searcher_factory=_factory_for(published))
    assert result["state"] == context_startup.UNAVAILABLE
    assert result["reason"] == "backend_error"
    assert result["error_type"] == "ResponseHandlingException"
    dumped = json.dumps(result, ensure_ascii=False)
    assert "http://" not in dumped and "secret" not in dumped and "refused" not in dumped


def test_missing_vector_client_is_unavailable_without_asking_qdrant(monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    result = context_startup.run(None, searcher_factory=lambda c: pytest.fail("must not build a searcher"))
    assert result["state"] == context_startup.UNAVAILABLE
    assert result["reason"] == "vector_client_unavailable"


@pytest.mark.parametrize("value", ["abc", "-1", "nan"])
def test_invalid_context_weight_is_a_config_error_before_any_qdrant_call(published, monkeypatch, value):
    monkeypatch.setenv("CONTEXT_WEIGHT", value)
    proxy = _ProxyVectorClient(published.client)
    with pytest.raises(ContextStartupError) as info:
        context_startup.run(proxy, searcher_factory=_factory_for(published))
    assert info.value.reason == "settings_invalid"
    assert proxy.client.calls == []
    assert context_startup.status()["state"] == context_startup.CONFIG_ERROR


def test_status_before_any_run_is_not_checked():
    assert context_startup.status() == {"state": context_startup.NOT_CHECKED}


def test_status_is_a_copy_and_does_not_touch_qdrant(published, monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    proxy = _ProxyVectorClient(published.client)
    context_startup.run(proxy, searcher_factory=_factory_for(published))
    calls = list(proxy.client.calls)
    first = context_startup.status()
    first["state"] = "tampered"
    assert context_startup.status()["state"] == context_startup.VERIFIED
    assert proxy.client.calls == calls


# ---------------------------------------------------------------------------
# 3. 실제 lifespan · /health
# ---------------------------------------------------------------------------

def _offline_app(monkeypatch):
    """lifespan이 밖으로 나가지 않게 — 피드백 색인·작업 스레드·Mongo 종료를 대체한다(Vertex는 conftest)."""
    from src.backend import main as backend_main

    monkeypatch.setattr(backend_main, "prepare_indexes", lambda: False)
    monkeypatch.setattr(backend_main.feedback_worker, "start_background", lambda *a, **k: False)
    monkeypatch.setattr(backend_main.feedback_worker, "stop_and_wait", lambda *a, **k: True)
    monkeypatch.setattr(backend_main, "close_feedback_store", lambda: None)
    monkeypatch.setenv("SEARCH_WARMUP", "0")
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "0")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return backend_main


def _local_publication(tmp_path: Path, monkeypatch):
    """임시 폴더의 **진짜 로컬 Qdrant**에 발행하고 닫는다. 앱이 같은 폴더를 연다."""
    storage = tmp_path / "qdrant"
    client = QdrantVectorClient(path=str(storage))
    pub = _publish(tmp_path, client)
    client.close()
    monkeypatch.setenv("QDRANT_PATH", str(storage))
    monkeypatch.setattr(context_startup, "ContextQdrantSearch",
                        lambda vector_client: _searcher(pub, client=vector_client))
    return pub, storage


@pytest.fixture
def fresh_dependencies():
    from src.backend.api import dependencies

    dependencies.close_vector_client()
    yield dependencies
    dependencies.close_vector_client()


def test_lifespan_verifies_a_correct_local_publication_and_health_reports_it(tmp_path, monkeypatch, fresh_dependencies):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    pub, _storage = _local_publication(tmp_path, monkeypatch)
    assert backend_main.warmup.enabled() is False  # 예열을 꺼도 검사는 돈다

    with TestClient(backend_main.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["context"] == context_startup.status()
        assert health["context"]["state"] == "verified"
        assert health["context"]["source"] == "startup"
        assert health["context"]["publication"]["dense_collection"] == pub.manifest["qdrant"]["dense_collection"]
        assert health["context"]["publication"]["context_build_id"] == pub.manifest["context_build_id"]
        assert health["warmup"]["state"] == "skipped"
    assert fresh_dependencies.get_vector_client.peek() is None  # 종료가 닫았다


def test_lifespan_fails_on_missing_aliases_releases_the_lock_and_can_restart(tmp_path, monkeypatch, fresh_dependencies):
    """v56 재현 → 수정 동작: 별칭 없는 저장소로는 서버가 **뜨지 않고**, 잠금이 풀려 고친 뒤 같은 프로세스에서 다시 뜬다."""
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    pub, storage = _local_publication(tmp_path, monkeypatch)
    q = pub.manifest["qdrant"]

    broken = QdrantVectorClient(path=str(storage))
    _delete_alias(broken, DENSE_ALIAS)
    _delete_alias(broken, SPARSE_ALIAS)
    broken.close()

    with pytest.raises(ContextStartupError) as info:
        with TestClient(backend_main.app):
            pytest.fail("Context가 켜졌는데 별칭이 없으면 서버가 떠서는 안 된다")
    assert info.value.reason == "aliases_missing"
    assert context_startup.status()["state"] == "config_error"
    assert fresh_dependencies.get_vector_client.peek() is None  # 클라이언트를 닫고 캐시를 비웠다

    # 잠금이 풀렸다 — 같은 폴더를 다시 열어 별칭을 복구할 수 있다(로컬 Qdrant는 한 프로세스 하나만)
    repair = QdrantVectorClient(path=str(storage))
    _point_alias(repair, DENSE_ALIAS, q["dense_collection"])
    _point_alias(repair, SPARSE_ALIAS, q["sparse_collection"])
    repair.close()

    with TestClient(backend_main.app) as client:
        assert client.get("/health").json()["context"]["state"] == "verified"


def test_lifespan_unexpected_check_exception_still_closes_the_client_and_releases_the_lock(
    tmp_path, monkeypatch, fresh_dependencies,
):
    """리뷰(P2): 설정 오류형이 아닌 예상 밖 예외로 기동이 끊겨도 클라이언트를 닫아야 한다 — 전에는 캐시가 남고
    같은 저장 폴더 재개방이 잠금 오류로 실패했다. 원래 예외형이 그대로 올라온다."""
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    _pub, storage = _local_publication(tmp_path, monkeypatch)

    real_run = context_startup.run

    def exploding_run(vector_client, **_kwargs):
        raise TypeError("unexpected: 'int' object is not subscriptable")

    monkeypatch.setattr(backend_main.context_startup, "run", exploding_run)
    with pytest.raises(TypeError):
        with TestClient(backend_main.app):
            pytest.fail("검사가 예외를 던지면 서버가 떠서는 안 된다")
    assert fresh_dependencies.get_vector_client.peek() is None  # 캐시가 남지 않았다
    QdrantVectorClient(path=str(storage)).close()  # 잠금이 풀렸다

    # 검사를 되돌리면 같은 프로세스에서 다시 뜬다
    monkeypatch.setattr(backend_main.context_startup, "run", real_run)
    with TestClient(backend_main.app) as client:
        assert client.get("/health").json()["context"]["state"] == "verified"


def test_lifespan_manifest_with_numeric_build_id_fails_as_config_error_and_releases_the_lock(
    tmp_path, monkeypatch, fresh_dependencies,
):
    """리뷰가 발견한 실제 사례: context_build_id=123 — 전에는 로그 출력에서 TypeError가 나고 잠금이 남았다."""
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    pub, storage = _local_publication(tmp_path, monkeypatch)
    _rewrite_state(pub.state, lambda d: d.__setitem__("context_build_id", 123))
    with pytest.raises(ContextStartupError) as info:
        with TestClient(backend_main.app):
            pass
    assert info.value.reason == "manifest_invalid"
    assert fresh_dependencies.get_vector_client.peek() is None
    QdrantVectorClient(path=str(storage)).close()


def test_lifespan_fails_on_partial_aliases_too(tmp_path, monkeypatch, fresh_dependencies):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    _pub, storage = _local_publication(tmp_path, monkeypatch)
    broken = QdrantVectorClient(path=str(storage))
    _delete_alias(broken, SPARSE_ALIAS)
    broken.close()
    with pytest.raises(ContextStartupError) as info:
        with TestClient(backend_main.app):
            pass
    assert info.value.reason == "aliases_partial"
    QdrantVectorClient(path=str(storage)).close()  # 잠금이 풀렸다


def test_lifespan_runs_the_check_before_warmup_with_the_shared_vector_client(monkeypatch):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    order = []
    shared = object()
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: shared)
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: order.append("close"))

    def fake_run(vector_client, **_kwargs):
        order.append(("context", vector_client))
        return {"state": "verified"}

    monkeypatch.setattr(backend_main.context_startup, "run", fake_run)
    monkeypatch.setattr(backend_main.warmup, "start_background", lambda: order.append("warmup"))
    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
    assert order[:2] == [("context", shared), "warmup"]
    assert order[-1] == "close"


def test_lifespan_with_context_disabled_never_touches_the_vector_client(monkeypatch):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0")

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"Context가 꺼져 있으면 벡터 클라이언트를 만지지 않는다: {name}")

    monkeypatch.setattr(backend_main, "get_vector_client", lambda: Untouchable())
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    with TestClient(backend_main.app) as client:
        context = client.get("/health").json()["context"]
    assert context["state"] == "disabled" and context["reason"] == "context_weight_zero"


def test_lifespan_connection_failure_starts_the_server_as_unavailable_without_leaking_the_uri(published, monkeypatch):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    proxy = _ProxyVectorClient(published.client, get_aliases=_raise_connection)
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: proxy)
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    monkeypatch.setattr(context_startup, "ContextQdrantSearch",
                        lambda vector_client: _searcher(published, client=vector_client))
    with TestClient(backend_main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        context = health.json()["context"]
    assert context["state"] == "unavailable" and context["error_type"] == "ResponseHandlingException"
    text = health.text
    assert "http://" not in text and "secret" not in text and "qdrant-internal" not in text


def test_lifespan_vector_client_failure_leaves_context_unavailable(monkeypatch):
    """벡터 DB 초기화 실패는 기존 정책대로 경고 + 기동. Context는 확인 불가로 남고 확인 완료로 적히지 않는다."""
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")

    def failing():
        raise RuntimeError("Storage folder is already accessed")

    monkeypatch.setattr(backend_main, "get_vector_client", failing)
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    with TestClient(backend_main.app) as client:
        context = client.get("/health").json()["context"]
    assert context["state"] == "unavailable" and context["reason"] == "vector_client_unavailable"


def test_health_does_not_rerun_the_check(published, monkeypatch, fresh_dependencies):
    from fastapi.testclient import TestClient

    backend_main = _offline_app(monkeypatch)
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.5")
    proxy = _ProxyVectorClient(published.client)
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: proxy)
    monkeypatch.setattr(backend_main, "close_vector_client", lambda: None)
    monkeypatch.setattr(context_startup, "ContextQdrantSearch",
                        lambda vector_client: _searcher(published, client=vector_client))
    with TestClient(backend_main.app) as client:
        after_startup = len(proxy.client.calls)
        for _ in range(3):
            assert client.get("/health").json()["context"]["state"] == "verified"
        assert len(proxy.client.calls) == after_startup
