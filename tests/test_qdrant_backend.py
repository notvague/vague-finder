"""Qdrant 백엔드가 Pinecone과 같은 모양·같은 점수로 답하는지 확인한다."""
import pytest

from src.vector_db.qdrant_backend import QdrantVectorClient, translate_filter
from src.vector_db.settings import IMAGE_INDEX_NAME, TEXT_HYBRID_INDEX_NAME


# ---------------------------------------------------------------------------
# 필터 변환
# ---------------------------------------------------------------------------

def test_translate_filter_handles_eq_and_and():
    flt = translate_filter({"$and": [{"title_script": {"$eq": "latin"}}, {"title_char_count": {"$eq": 3}}]})
    keys = {c.key: c.match.value for c in flt.must}
    assert keys == {"title_script": "latin", "title_char_count": 3}

    assert translate_filter(None) is None
    assert translate_filter({}) is None
    # 값을 그대로 준 형태도 동등 비교로 받는다
    assert translate_filter({"vocal_gender": "여성"}).must[0].match.value == "여성"


def test_translate_filter_rejects_unknown_operators():
    # 모르는 연산자를 무시하면 필터 없이 검색되어 결과가 조용히 달라진다.
    with pytest.raises(ValueError):
        translate_filter({"title": {"$regex": "^a"}})
    with pytest.raises(ValueError):
        translate_filter({"$nor": [{"a": {"$eq": 1}}]})


# ---------------------------------------------------------------------------
# 검색 — 메모리 모드
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    c = QdrantVectorClient(path=":memory:")
    yield c
    c.close()


def _unit(*values):
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def _fill_text(client):
    from qdrant_client import models

    from src.vector_db.qdrant_backend import DENSE_VECTOR, SPARSE_VECTOR

    collection = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=3, with_sparse=True, recreate=True)
    points = [
        # a: dense가 강함 / b: sparse가 강함 / c: 둘 다 약함
        models.PointStruct(id=1, payload={"song_id": "a", "title": "A", "vocal_gender": "여성"},
                           vector={DENSE_VECTOR: _unit(1.0, 0.0, 0.0),
                                   SPARSE_VECTOR: models.SparseVector(indices=[10], values=[1.0])}),
        models.PointStruct(id=2, payload={"song_id": "b", "title": "B", "vocal_gender": "남성"},
                           vector={DENSE_VECTOR: _unit(0.6, 0.8, 0.0),
                                   SPARSE_VECTOR: models.SparseVector(indices=[10], values=[9.0])}),
        models.PointStruct(id=3, payload={"song_id": "c", "title": "C", "vocal_gender": "여성"},
                           vector={DENSE_VECTOR: _unit(0.0, 0.0, 1.0),
                                   SPARSE_VECTOR: models.SparseVector(indices=[99], values=[5.0])}),
    ]
    client.client.upsert(collection_name=collection, points=points)


def test_hybrid_score_is_dense_plus_sparse(client):
    """Pinecone 하이브리드 = 스케일한 dense·sparse의 내적 합. 그 합을 재현해야 한다."""
    _fill_text(client)
    index = client.Index(TEXT_HYBRID_INDEX_NAME)

    alpha = 0.5
    dense_q = [v * alpha for v in _unit(1.0, 0.0, 0.0)]
    sparse_q = {"indices": [10], "values": [1.0 * (1 - alpha)]}

    res = index.query(vector=dense_q, top_k=3, sparse_vector=sparse_q)
    scores = {m["id"]: m["score"] for m in res["matches"]}

    # a: 0.5*1.0 + 0.5*1.0 = 1.0 / b: 0.5*0.6 + 0.5*9 = 4.8 / c: 0 + 0 = 0
    assert scores["b"] == pytest.approx(4.8, abs=1e-5)
    assert scores["a"] == pytest.approx(1.0, abs=1e-5)
    assert [m["id"] for m in res["matches"]][:2] == ["b", "a"]
    # sparse가 하나도 안 걸린 곡도 dense 점수로 남는다
    assert scores["c"] == pytest.approx(0.0, abs=1e-5)


def test_sparse_only_hit_outside_dense_top_k_is_not_lost(client):
    """dense 상위만 합산하면 sparse로만 걸린 곡을 놓친다. 전수 합산이어야 한다."""
    _fill_text(client)
    index = client.Index(TEXT_HYBRID_INDEX_NAME)
    res = index.query(
        vector=[0.0, 0.0, 0.05],          # c만 살짝 닮은 dense
        top_k=1,
        sparse_vector={"indices": [10], "values": [1.0]},  # a·b가 걸린다
    )
    assert [m["id"] for m in res["matches"]] == ["b"]


def test_filter_and_metadata_flags(client):
    _fill_text(client)
    index = client.Index(TEXT_HYBRID_INDEX_NAME)

    res = index.query(vector=_unit(1.0, 0.0, 0.0), top_k=3, filter={"vocal_gender": {"$eq": "남성"}})
    assert [m["id"] for m in res["matches"]] == ["b"]
    assert res["matches"][0]["metadata"]["title"] == "B"
    # song_id는 메타데이터가 아니라 match id로 나간다 (Pinecone과 같은 모양)
    assert "song_id" not in res["matches"][0]["metadata"]

    res = index.query(vector=_unit(1.0, 0.0, 0.0), top_k=1, include_metadata=False)
    assert res["matches"][0]["metadata"] == {}


def test_dense_only_index_matches_cosine_order(client):
    from qdrant_client import models

    from src.vector_db.qdrant_backend import DENSE_VECTOR

    collection = client.ensure_collection(IMAGE_INDEX_NAME, dim=2, recreate=True)
    client.client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(id=1, payload={"song_id": "near"}, vector={DENSE_VECTOR: _unit(1.0, 0.1)}),
            models.PointStruct(id=2, payload={"song_id": "far"}, vector={DENSE_VECTOR: _unit(0.0, 1.0)}),
        ],
    )
    res = client.Index(IMAGE_INDEX_NAME).query(vector=_unit(1.0, 0.0), top_k=2)
    assert [m["id"] for m in res["matches"]] == ["near", "far"]
    assert res["matches"][0]["score"] > res["matches"][1]["score"]


# ---------------------------------------------------------------------------
# 실제 검색 경로가 쓰는 필터 — 범위·OR (리뷰 지적)
# ---------------------------------------------------------------------------

def test_translate_filter_supports_range_and_or_used_by_title_paths():
    """제목 단서 경로는 $gte(문장형)와 $or(의성어)를 쓴다.

    회귀 이력: $eq·$and만 지원해 두 경로가 ValueError로 죽었고, 라우터가 빈
    결과로 넘겨 **제목 단서 후보가 통째로 사라졌다**. 검색은 성공한 것처럼 보인다.
    """
    from src.common.title_features import build_title_meaning_metadata_filter

    class _Clue:
        def __init__(self, kind):
            self.kind = kind

    for kind in ("sentence", "onomatopoeia", "foreign_person_name"):
        raw = build_title_meaning_metadata_filter(_Clue(kind))
        assert raw is not None, kind
        assert translate_filter(raw) is not None, f"{kind} 필터가 변환되지 않았다: {raw}"


def test_range_and_or_filters_select_the_right_songs(client):
    from qdrant_client import models

    from src.vector_db.qdrant_backend import DENSE_VECTOR

    collection = client.ensure_collection(IMAGE_INDEX_NAME, dim=2, recreate=True)
    client.client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(id=1, vector={DENSE_VECTOR: _unit(1.0, 0.0)},
                               payload={"song_id": "long", "title_word_count": 4,
                                        "title_repeated_char": False, "title_repeated_word": False}),
            models.PointStruct(id=2, vector={DENSE_VECTOR: _unit(1.0, 0.0)},
                               payload={"song_id": "short", "title_word_count": 1,
                                        "title_repeated_char": False, "title_repeated_word": False}),
            models.PointStruct(id=3, vector={DENSE_VECTOR: _unit(1.0, 0.0)},
                               payload={"song_id": "repeat", "title_word_count": 1,
                                        "title_repeated_char": True, "title_repeated_word": False}),
        ],
    )
    index = client.Index(IMAGE_INDEX_NAME)

    got = {m["id"] for m in index.query(
        vector=_unit(1.0, 0.0), top_k=10,
        filter={"title_word_count": {"$gte": 3}})["matches"]}
    assert got == {"long"}

    got = {m["id"] for m in index.query(
        vector=_unit(1.0, 0.0), top_k=10,
        filter={"$or": [{"title_repeated_char": {"$eq": True}},
                        {"title_repeated_word": {"$eq": True}}]})["matches"]}
    assert got == {"repeat"}

    # $and 안에 범위와 동등 비교가 섞여도 AND 의미가 유지된다
    got = {m["id"] for m in index.query(
        vector=_unit(1.0, 0.0), top_k=10,
        filter={"$and": [{"title_word_count": {"$gte": 3}},
                         {"title_repeated_char": {"$eq": True}}]})["matches"]}
    assert got == set()


def test_eval_runner_uses_the_configured_backend(monkeypatch):
    """구 평가 실행기도 VECTOR_BACKEND를 따라야 한다.

    회귀 이력: get_pinecone_client()를 직접 불러, qdrant 설정으로 돌려도
    Pinecone을 때렸다(한도 소진 + Qdrant 성능 미측정).
    """
    import inspect

    from src.eval import evaluate as eval_mod

    source = inspect.getsource(eval_mod)
    assert "get_vector_client" in source
    assert "get_pinecone_client()" not in source


def test_loader_fails_when_embedding_files_are_missing(tmp_path, monkeypatch):
    """벡터 파일이 없으면 적재가 실패해야 한다.

    회귀 이력: 없으면 전부 건너뛰고 종료 코드 0으로 끝나, 빈 컬렉션이 만들어진 줄
    모른 채 검색이 0건을 돌려줬다. 벡터는 git에 없고 드라이브에서 받는다.
    """
    import sys

    from src.vector_db.cli import qdrant_load

    monkeypatch.setattr(sys, "argv", [
        "qdrant_load",
        "--artifacts", str(tmp_path / "없는디렉터리"),
        "--qdrant-path", str(tmp_path / "store"),
    ])
    with pytest.raises(SystemExit) as exc:
        qdrant_load.main()
    assert "임베딩 디렉터리가 없습니다" in str(exc.value)


# --- 적재 경로 검증 (2026-09-18 점검) ------------------------------------------------

def write_npy(directory, song_id, dim):
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{song_id}.npy", np.ones(dim, dtype="float32"))


def write_sparse_npz(directory, song_id):
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{song_id}.npz", indices=np.array([1, 2]), values=np.array([0.5, 0.5]))


def build_artifacts(tmp_path, song_ids, *, image_dim=None, audio_dim=None, sparse_ids=None):
    """세 모달리티 .npy와 sparse .npz, 코퍼스 JSONL을 만든다."""
    import json

    from src.vector_db.settings import AUDIO_DIM, IMAGE_DIM, TEXT_DENSE_DIM

    artifacts = tmp_path / "artifacts"
    for song_id in song_ids:
        write_npy(artifacts / "text_dense" / "m", song_id, TEXT_DENSE_DIM)
        write_npy(artifacts / "image" / "m", song_id, image_dim or IMAGE_DIM)
        write_npy(artifacts / "audio" / "m", song_id, audio_dim or AUDIO_DIM)
    for song_id in (sparse_ids if sparse_ids is not None else song_ids):
        write_sparse_npz(artifacts / "text_sparse" / "bm25", song_id)
    return artifacts


def write_corpus(tmp_path, song_ids):
    import json

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"song_id": sid, "metadata": {"title": f"곡{sid}", "artist": ["가수"]}},
                       ensure_ascii=False)
            for sid in song_ids
        ) + "\n",
        encoding="utf-8",
    )
    return corpus


def run_loader(monkeypatch, artifacts, corpus, store, *extra):
    import sys

    from src.vector_db.cli import qdrant_load

    monkeypatch.setattr(sys, "argv", [
        "qdrant_load",
        "--artifacts", str(artifacts),
        "--corpus", str(corpus),
        "--qdrant-path", str(store),
        "--sparse-from-artifacts",
        *extra,
    ])
    return qdrant_load.main()


def count_points(store, index_name):
    from src.vector_db.qdrant_backend import QdrantVectorClient, collection_name

    client = QdrantVectorClient(path=str(store))
    try:
        name = collection_name(index_name)
        if not client.client.collection_exists(name):
            return 0
        return client.client.count(collection_name=name).count
    finally:
        client.close()


def test_loader_refuses_songs_without_metadata(tmp_path, monkeypatch):
    """dense에는 있고 코퍼스에는 없는 곡을 song_id만 넣어 적재하면 안 된다.

    회귀 이력: 종료 코드 0으로 성공하고, 제목·가수가 빈 곡이 검색 1위로 나왔다.
    """
    artifacts = build_artifacts(tmp_path, ["1", "2"])
    corpus = write_corpus(tmp_path, ["1"])          # 2번 곡의 메타데이터가 없다
    store = tmp_path / "store"

    with pytest.raises(SystemExit) as exc:
        run_loader(monkeypatch, artifacts, corpus, store)
    assert "메타데이터가 없는 곡" in str(exc.value)
    # 검증에서 막혔으면 컬렉션을 만들지도 않는다
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 0


def test_loader_allows_missing_metadata_only_with_the_flag(tmp_path, monkeypatch):
    artifacts = build_artifacts(tmp_path, ["1", "2"])
    corpus = write_corpus(tmp_path, ["1"])
    store = tmp_path / "store"

    assert run_loader(monkeypatch, artifacts, corpus, store, "--allow-missing-metadata") == 0
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 2


def test_loader_refuses_when_sparse_files_are_absent(tmp_path, monkeypatch):
    """--sparse-from-artifacts인데 sparse 디렉터리가 비어 있으면 막는다.

    회귀 이력: 빈 결과를 그대로 적재해, 하이브리드 검색이 dense 점수만으로 돌았다.
    """
    artifacts = build_artifacts(tmp_path, ["1"], sparse_ids=[])
    corpus = write_corpus(tmp_path, ["1"])
    store = tmp_path / "store"

    with pytest.raises(SystemExit) as exc:
        run_loader(monkeypatch, artifacts, corpus, store)
    assert "sparse 벡터가 없는 곡" in str(exc.value)
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 0


def test_loader_validates_every_modality_before_touching_a_collection(tmp_path, monkeypatch):
    """뒤쪽 모달리티의 차원 오류를 앞쪽 컬렉션을 바꾸기 전에 발견해야 한다.

    회귀 이력: --recreate에서 이미지 차원이 틀리면 텍스트는 이미 새 데이터로 교체되고
    이미지·오디오는 예전 데이터로 남아, 세 컬렉션이 서로 다른 데이터를 가졌다.
    """
    store = tmp_path / "store"

    # 1회차: 곡 1개로 정상 적재
    good = build_artifacts(tmp_path / "v1", ["1"])
    corpus1 = write_corpus(tmp_path / "v1", ["1"])
    (tmp_path / "v1").mkdir(parents=True, exist_ok=True)
    assert run_loader(monkeypatch, good, corpus1, store, "--recreate") == 0
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 1

    # 2회차: 곡 2개인데 이미지 차원이 틀렸다
    bad = build_artifacts(tmp_path / "v2", ["1", "2"], image_dim=7)
    corpus2 = write_corpus(tmp_path / "v2", ["1", "2"])
    with pytest.raises(ValueError) as exc:
        run_loader(monkeypatch, bad, corpus2, store, "--recreate")
    assert "차원이 7" in str(exc.value)

    # 텍스트 컬렉션이 그대로다 — 갈라지지 않았다
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 1


def test_loader_warns_when_a_modality_is_missing_songs(tmp_path, monkeypatch, capsys):
    """모달리티 사이 ID 차이는 막지 않지만 세어서 알린다."""
    from src.vector_db.settings import AUDIO_DIM, IMAGE_DIM, TEXT_DENSE_DIM

    artifacts = tmp_path / "artifacts"
    for sid in ["1", "2"]:
        write_npy(artifacts / "text_dense" / "m", sid, TEXT_DENSE_DIM)
        write_npy(artifacts / "audio" / "m", sid, AUDIO_DIM)
        write_sparse_npz(artifacts / "text_sparse" / "bm25", sid)
    write_npy(artifacts / "image" / "m", "1", IMAGE_DIM)      # 2번 곡 커버가 없다

    corpus = write_corpus(tmp_path, ["1", "2"])
    assert run_loader(monkeypatch, artifacts, corpus, tmp_path / "store") == 0
    assert "다른 모달리티에는 있고 여기엔 없는 곡" in capsys.readouterr().out


# --- 벡터 클라이언트 단일 생성 --------------------------------------------------------

def test_vector_client_is_created_once_even_when_requests_overlap(monkeypatch):
    """@lru_cache()는 첫 호출의 중복 실행을 막지 못한다.

    회귀 이력: 서버 시작 직후 요청 두 개가 겹치면 로컬 Qdrant 저장 폴더를 두 번 열어
    한쪽이 "Storage folder ... is already accessed"로 실패했다.
    """
    import threading
    import time

    from src.backend.api import dependencies

    dependencies.get_vector_client.cache_clear()
    builds = []

    def slow_build():
        builds.append(1)
        time.sleep(0.05)          # 두 스레드가 겹칠 틈을 만든다
        return object()

    monkeypatch.setattr(dependencies, "get_pinecone_client", slow_build)
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")

    results = []
    threads = [threading.Thread(target=lambda: results.append(dependencies.get_vector_client()))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"클라이언트가 {len(builds)}번 만들어졌다"
    assert len({id(r) for r in results}) == 1
    dependencies.get_vector_client.cache_clear()


def test_close_vector_client_does_not_create_one(monkeypatch):
    """종료 처리가 클라이언트를 새로 만들면 안 된다(폴더를 다시 잠근다)."""
    from src.backend.api import dependencies

    dependencies.get_vector_client.cache_clear()
    builds = []
    monkeypatch.setattr(dependencies, "get_pinecone_client", lambda: builds.append(1) or object())
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")

    dependencies.close_vector_client()
    assert builds == []

    class Closable:
        closed = False

        def close(self):
            Closable.closed = True

    monkeypatch.setattr(dependencies, "get_pinecone_client", Closable)
    dependencies.get_vector_client()
    dependencies.close_vector_client()
    assert Closable.closed is True
    assert dependencies.get_vector_client.peek() is None


# --- 같은 프로세스에서 앱 재시작 -------------------------------------------------------

class Stub:
    """모델·서비스 대역. 무엇을 받아도 만들어지고 load()도 받는다."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def load(self):
        return self


class StubClient(Stub):
    closed = False

    def close(self):
        self.closed = True


class StubRouter(Stub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shut = False

    def shutdown(self):
        self.shut = True


def stub_search_stack(monkeypatch):
    """모델을 올리지 않고 검색 스택을 만들 수 있게 대역으로 바꾼다."""
    from src.backend.api import dependencies

    for name in ("KoE5Embedder", "BM25SparseEncoder", "SigLIP2Embedder", "CLAPAudioEmbedder",
                 "LyricsExactSearchService", "MusicReranker", "QueryAnalyzer", "SearchService"):
        monkeypatch.setattr(dependencies, name, Stub)
    monkeypatch.setattr(dependencies, "SearchRouter", StubRouter)
    monkeypatch.setattr(dependencies, "get_pinecone_client", StubClient)
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    for factory in (dependencies.get_vector_client, dependencies.get_text_embedder,
                    dependencies.get_image_embedder, dependencies.get_audio_embedder,
                    dependencies.get_bm25_encoder, dependencies.get_search_service,
                    dependencies.get_query_analyzer, dependencies.get_lyrics_exact_search_service,
                    dependencies.get_reranker, dependencies.get_search_router):
        factory.cache_clear()
    return dependencies


def test_restarting_in_one_process_rebuilds_the_search_stack(monkeypatch):
    """종료 후 다시 시작하면 검색 스택이 새 클라이언트로 다시 만들어져야 한다.

    회귀 이력: close_vector_client()가 벡터 클라이언트 캐시만 비워서, 두 번째 시작에서
    새 클라이언트를 만들어 놓고도 get_search_service()/get_search_router()가 닫힌
    클라이언트를 쥔 예전 객체를 재사용했다 ("QdrantLocal instance is closed").
    """
    dependencies = stub_search_stack(monkeypatch)

    first_client = dependencies.get_vector_client()
    first_router = dependencies.get_search_router()
    first_service = dependencies.get_search_service()
    assert first_router.kwargs["pinecone_client"] is first_client
    assert first_service.kwargs["pinecone_client"] is first_client

    dependencies.close_vector_client()
    assert first_client.closed is True
    assert first_router.shut is True          # 스레드풀도 정리했다

    second_client = dependencies.get_vector_client()
    second_router = dependencies.get_search_router()
    second_service = dependencies.get_search_service()

    assert second_client is not first_client
    assert second_router is not first_router
    assert second_service is not first_service
    # 닫힌 클라이언트를 쥔 객체가 남아 있지 않다
    assert second_router.kwargs["pinecone_client"] is second_client
    assert second_service.kwargs["pinecone_client"] is second_client

    dependencies.close_vector_client()


def test_every_factory_holding_the_vector_client_is_cleared_on_close():
    """get_vector_client()를 쓰는 팩토리가 종료 목록에서 빠지면 잡는다.

    앞으로 클라이언트를 쓰는 팩토리가 추가될 때 _vector_client_dependents()에 넣는 것을
    잊으면, 같은 프로세스 재시작에서 닫힌 클라이언트가 되살아난다.
    """
    import ast
    import inspect

    from src.backend.api import dependencies

    tree = ast.parse(inspect.getsource(dependencies))
    holders = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "get_vector_client"):
                holders.add(node.name)
    holders -= {"close_vector_client"}

    cleared = {f.__name__ for f in dependencies._vector_client_dependents()}
    assert holders == cleared, (
        f"닫을 때 비우지 않는 팩토리: {sorted(holders - cleared)} / "
        f"클라이언트를 쓰지 않는데 목록에 있는 것: {sorted(cleared - holders)}"
    )


# --- NaN·Inf 벡터 (2026-09-18 3차 점검) ------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["NaN", "Inf"])
def test_loader_refuses_non_finite_dense_before_touching_a_collection(tmp_path, monkeypatch, bad):
    """차원이 맞아도 값이 NaN·Inf면 컬렉션을 건드리기 전에 막는다.

    회귀 이력: 차원만 검사해서 사전 검증을 통과했고, --recreate가 텍스트를 교체하고
    이미지 컬렉션을 비운 뒤 upsert에서 "Vector contains NaN values"로 죽었다.
    세 컬렉션이 서로 다른 상태로 갈렸다.
    """
    import numpy as np

    from src.vector_db.settings import IMAGE_DIM

    store = tmp_path / "store"

    # 1회차: 정상 적재
    good = build_artifacts(tmp_path / "v1", ["1"])
    corpus1 = write_corpus(tmp_path / "v1", ["1"])
    assert run_loader(monkeypatch, good, corpus1, store, "--recreate") == 0
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 1

    # 2회차: 이미지 벡터의 차원은 맞지만 값이 망가졌다
    broken = build_artifacts(tmp_path / "v2", ["1", "2"])
    corpus2 = write_corpus(tmp_path / "v2", ["1", "2"])
    vec = np.ones(IMAGE_DIM, dtype="float32")
    vec[3] = bad
    np.save(broken / "image" / "m" / "2.npy", vec)

    with pytest.raises(ValueError) as exc:
        run_loader(monkeypatch, broken, corpus2, store, "--recreate")
    assert "NaN/Inf" in str(exc.value)

    # 텍스트도 이미지도 그대로다 — 갈라지지 않았다
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 1
    assert count_points(store, IMAGE_INDEX_NAME) == 1


def test_loader_refuses_non_finite_sparse_values(tmp_path, monkeypatch):
    """sparse 값에 NaN이 있으면 막는다."""
    import numpy as np

    artifacts = build_artifacts(tmp_path, ["1"], sparse_ids=[])
    sparse_dir = artifacts / "text_sparse" / "bm25"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    np.savez(sparse_dir / "1.npz", indices=np.array([1, 2]),
             values=np.array([0.5, float("nan")]))
    corpus = write_corpus(tmp_path, ["1"])
    store = tmp_path / "store"

    with pytest.raises(SystemExit) as exc:
        run_loader(monkeypatch, artifacts, corpus, store)
    assert "sparse 값에 NaN/Inf" in str(exc.value)
    assert count_points(store, TEXT_HYBRID_INDEX_NAME) == 0


# --- 종료가 실행 중인 검색을 기다리는가 ------------------------------------------------

def test_router_shutdown_waits_for_running_searches():
    """shutdown()은 돌고 있는 작업이 끝날 때까지 기다린다.

    회귀 이력: wait=False였다. 요청이 취소돼도 검색 스레드는 계속 돌아서, 그 작업이 이미
    닫힌 Qdrant 클라이언트를 만지고 "QdrantLocal instance is closed"로 죽었다.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from src.retrieval.search_router import SearchRouter

    router = SearchRouter.__new__(SearchRouter)          # __init__의 무거운 의존을 건너뛴다
    router._pool = ThreadPoolExecutor(max_workers=2)

    finished = []

    def slow_search():
        time.sleep(0.2)
        finished.append("done")

    router._pool.submit(slow_search)
    time.sleep(0.02)                                     # 작업이 시작되게 한다
    router.shutdown()
    # 기다렸으므로 이 시점에 이미 끝나 있다
    assert finished == ["done"]


def test_closing_the_client_waits_for_the_search_pool(monkeypatch):
    """클라이언트를 닫기 전에 스레드풀이 빈다 — 순서와 대기를 함께 고정한다."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    dependencies = stub_search_stack(monkeypatch)
    events = []

    class WaitingRouter(Stub):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._pool.submit(self._slow)

        def _slow(self):
            time.sleep(0.15)
            events.append("검색 끝")

        def shutdown(self, wait: bool = True):
            self._pool.shutdown(wait=wait, cancel_futures=True)
            events.append("풀 정리")

    class RecordingClient(Stub):
        def close(self):
            events.append("클라이언트 닫힘")

    monkeypatch.setattr(dependencies, "SearchRouter", WaitingRouter)
    monkeypatch.setattr(dependencies, "get_pinecone_client", RecordingClient)

    dependencies.get_vector_client()
    dependencies.get_search_router()
    dependencies.close_vector_client()

    assert events == ["검색 끝", "풀 정리", "클라이언트 닫힘"], events


def test_lifespan_runs_the_shutdown_and_it_completes(monkeypatch):
    """lifespan 종료가 close_vector_client()를 부르고, 끝난 뒤에 앱이 내려간다.

    close_vector_client()는 검색 스레드풀이 비기를 기다리므로 블로킹이다. 그래서 별도
    스레드로 넘기고 await한다.

    한계: await를 빼고 run_in_executor나 create_task로 던져 놓아도 이 테스트는 통과한다.
    anyio 포털이 루프를 닫을 때 남은 작업을 정리해 주기 때문이다(직접 확인). 즉 이 테스트가
    고정하는 것은 '종료가 실행되고 완료된다'까지이고, '기다린다'는 코드로만 보장된다.
    스레드풀이 클라이언트보다 먼저 비는지는
    test_closing_the_client_waits_for_the_search_pool이 고정한다.
    """
    import time

    from fastapi.testclient import TestClient

    from src.backend import main as backend_main

    done = []

    def slow_close():
        time.sleep(0.15)
        done.append("닫힘")

    monkeypatch.setattr(backend_main, "close_vector_client", slow_close)
    monkeypatch.setattr(backend_main, "get_vector_client", lambda: object())

    with TestClient(backend_main.app) as client:
        assert client.get("/health").status_code == 200
        assert done == []

    # with를 벗어난 시점에 종료가 이미 끝나 있어야 한다
    assert done == ["닫힘"]
