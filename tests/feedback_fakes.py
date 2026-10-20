"""피드백 시험이 함께 쓰는 Mongo 대역과 합성 자료.

대역은 **저장소가 실제로 쓰는 연산만** 흉내 낸다 — 점 표기 경로, `$gt/$gte/$lt/$lte/$ne/$in/$nin`,
`$or/$and`, `$set/$push($each·$slice)`, 정렬·건너뛰기·개수 제한, 색인 정보. 실제 Mongo와 갈릴 수 있는
자리(없는 필드와 null의 비교, `$nin`·`$ne`가 없는 필드를 맞추는 것)는 Mongo 규칙대로 구현했고,
같은 계약 시험을 `FEEDBACK_TEST_MONGO_URI`가 있으면 **실제 Mongo로도** 돌린다(`store_backend` 픽스처).
대역 통과만으로 TTL 삭제가 검증된 것은 아니다 — 그것은 `tests/feedback/test_feedback_mongo_integration.py`가 본다.
"""
from __future__ import annotations

import copy
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pymongo.errors import DuplicateKeyError

from src.backend.schemas.feedback import FeedbackRequest
from src.feedback.store import FeedbackStore

_MISSING = object()


def _lookup(document, path):
    value = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _comparable(left, right):
    """Mongo는 타입이 다른 값을 크기 비교로 맞추지 않는다. 없는 필드·null도 마찬가지다."""
    if left is _MISSING or left is None or right is None:
        return False
    numbers = (int, float)
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool)
    if isinstance(left, numbers) and isinstance(right, numbers):
        return True
    return type(left) is type(right)


def _equals(value, expected):
    if expected is None:
        return value is _MISSING or value is None
    return value is not _MISSING and value == expected


def _match_condition(value, condition):
    if not isinstance(condition, dict) or not any(key.startswith("$") for key in condition):
        return _equals(value, condition)
    for op, operand in condition.items():
        if op == "$gt":
            ok = _comparable(value, operand) and value > operand
        elif op == "$gte":
            ok = _comparable(value, operand) and value >= operand
        elif op == "$lt":
            ok = _comparable(value, operand) and value < operand
        elif op == "$lte":
            ok = _comparable(value, operand) and value <= operand
        elif op == "$ne":
            ok = not _equals(value, operand)
        elif op == "$in":
            ok = any(_equals(value, item) for item in operand)
        elif op == "$nin":
            ok = not any(_equals(value, item) for item in operand)
        elif op == "$exists":
            ok = (value is not _MISSING) == bool(operand)
        else:
            raise NotImplementedError(f"대역이 모르는 조건: {op}")
        if not ok:
            return False
    return True


def matches(document, query):
    for key, condition in (query or {}).items():
        if key == "$or":
            if not any(matches(document, branch) for branch in condition):
                return False
        elif key == "$and":
            if not all(matches(document, branch) for branch in condition):
                return False
        elif key.startswith("$"):
            raise NotImplementedError(f"대역이 모르는 연산자: {key}")
        elif not _match_condition(_lookup(document, key), condition):
            return False
    return True


def _project(document, projection):
    if not projection:
        return copy.deepcopy(document)
    include = {key for key, flag in projection.items() if flag and key != "_id"}
    exclude = {key for key, flag in projection.items() if not flag}
    if include:
        out = {key: copy.deepcopy(document[key]) for key in include if key in document}
        if "_id" not in exclude:
            out["_id"] = document["_id"]
        return out
    return {key: copy.deepcopy(value) for key, value in document.items() if key not in exclude}


def _set_path(document, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        document = document.setdefault(part, {})
    document[parts[-1]] = copy.deepcopy(value)


def _apply_update(document, update):
    for op, fields in update.items():
        if op == "$set":
            for path, value in fields.items():
                _set_path(document, path, value)
        elif op == "$inc":
            for path, amount in fields.items():
                current = _lookup(document, path)
                _set_path(document, path, (0 if current is _MISSING else current) + amount)
        elif op == "$push":
            for path, value in fields.items():
                current = _lookup(document, path)
                items = [] if current is _MISSING else list(current)
                if isinstance(value, dict) and "$each" in value:
                    items.extend(copy.deepcopy(value["$each"]))
                    if "$slice" in value:
                        size = value["$slice"]
                        items = items[size:] if size < 0 else items[:size]
                else:
                    items.append(copy.deepcopy(value))
                _set_path(document, path, items)
        else:
            raise NotImplementedError(f"대역이 모르는 갱신: {op}")


class _Cursor:
    def __init__(self, documents):
        self._documents = documents
        self._skip = 0
        self._limit = None

    def sort(self, keys, direction=None):
        if isinstance(keys, str):
            keys = [(keys, direction or 1)]
        for key, order in reversed(keys):
            def sort_key(document, key=key):
                value = _lookup(document, key)
                return (value is not _MISSING and value is not None, value if value not in (_MISSING, None) else 0)
            self._documents.sort(key=sort_key, reverse=order < 0)
        return self

    def skip(self, count):
        self._skip = count
        return self

    def limit(self, count):
        self._limit = count or None
        return self

    def __iter__(self):
        end = None if self._limit is None else self._skip + self._limit
        return iter(self._documents[self._skip:end])


class FakeCollection:
    """`_id` 유일성과 문서 단위 원자적 갱신을 잠금 안에서 재현한다."""

    def __init__(self):
        self.documents = {}
        self.index_info = {}
        self.lock = threading.RLock()

    # --- 색인 ---
    def create_index(self, key, **options):
        name = f"{key}_1"
        with self.lock:
            self.index_info[name] = {"key": [(key, 1)], **options}
        return name

    def index_information(self):
        with self.lock:
            return {"_id_": {"key": [("_id", 1)]}, **copy.deepcopy(self.index_info)}

    # --- 읽기 ---
    def _find(self, query):
        return [doc for doc in self.documents.values() if matches(doc, query)]

    def find_one(self, query=None, projection=None):
        with self.lock:
            found = self._find(query)
            return _project(found[0], projection) if found else None

    def find(self, query=None, projection=None):
        with self.lock:
            return _Cursor([_project(doc, projection) for doc in self._find(query)])

    def count_documents(self, query):
        with self.lock:
            return len(self._find(query))

    # --- 쓰기 ---
    def insert_one(self, document):
        with self.lock:
            if document["_id"] in self.documents:
                raise DuplicateKeyError("duplicate")
            self.documents[document["_id"]] = copy.deepcopy(document)

    def find_one_and_update(self, query, update, **options):
        with self.lock:
            found = self._find(query)
            if not found:
                return None
            _apply_update(found[0], update)
            return copy.deepcopy(found[0])

    def update_one(self, query, update, upsert=False):
        with self.lock:
            found = self._find(query)
            if not found:
                return SimpleNamespace(matched_count=0, modified_count=0)
            _apply_update(found[0], update)
            return SimpleNamespace(matched_count=1, modified_count=1)

    def replace_one(self, query, replacement, upsert=False):
        with self.lock:
            found = self._find(query)
            if found:
                self.documents[found[0]["_id"]] = copy.deepcopy(replacement)
            elif upsert:
                self.documents[replacement["_id"]] = copy.deepcopy(replacement)
            return SimpleNamespace(matched_count=len(found[:1]))

    def delete_many(self, query):
        with self.lock:
            doomed = [doc["_id"] for doc in self._find(query)]
            for key in doomed:
                del self.documents[key]
            return SimpleNamespace(deleted_count=len(doomed))


class FakeDatabase(dict):
    """`database["이름"]`으로 컬렉션을 준다. 없는 이름은 만들어 준다(Mongo와 같다)."""

    def __missing__(self, name):
        self[name] = FakeCollection()
        return self[name]


# ---------------------------------------------------------------------------
# 저장소 픽스처 — 대역과 (있으면) 실제 Mongo를 같은 시험으로 돌린다
# ---------------------------------------------------------------------------

MONGO_URI_ENV = "FEEDBACK_TEST_MONGO_URI"


def _real_database():
    """시험 전용 DB. 이름에 무작위 값을 붙여 운영 DB와 겹치지 않게 하고, 끝나면 DB째 지운다."""
    from pymongo import MongoClient

    client = MongoClient(os.environ[MONGO_URI_ENV], tz_aware=True, serverSelectionTimeoutMS=3000)
    name = f"vf_feedback_test_{uuid.uuid4().hex[:12]}"
    return client, client[name], name


@pytest.fixture(params=["fake", "mongo"])
def store_backend(request):
    """(저장소, 데이터베이스). `mongo`는 `FEEDBACK_TEST_MONGO_URI`가 있을 때만 돈다."""
    if request.param == "fake":
        database = FakeDatabase()
        yield FeedbackStore(database), database
        return
    if not os.getenv(MONGO_URI_ENV):
        pytest.skip(f"{MONGO_URI_ENV}가 없어 실제 Mongo 계약 시험을 건너뛴다")
    client, database, name = _real_database()
    try:
        yield FeedbackStore(database, client), database
    finally:
        client.drop_database(name)
        client.close()


# ---------------------------------------------------------------------------
# 합성 사례
# ---------------------------------------------------------------------------

def record_run(store, *, query="휘파람으로 시작하는 노래", shown=("a", "b"), candidates=None):
    """검색 기록 하나. 내용은 모두 합성이다."""
    shown = list(shown)
    return store.record_search({
        "query": query, "shown_ids": shown,
        "shown": [{"id": sid, "rank": rank, "title": f"곡 {sid}", "artist": f"가수 {sid}"}
                  for rank, sid in enumerate(shown, 1)],
        "candidate_ids": list(candidates or shown), "answers": [], "asked_slots": [], "rejected_ids": [],
        "turn": 1, "analysis": {"original_query": query},
    })


def submission(receipt, **changes):
    return FeedbackRequest(**{
        "search_id": receipt.search_id, "token": receipt.token, "submission_id": "test_submission_1",
        "selected_song_id": "a", "reason": "도입부 휘파람", **changes,
    })


def submit_case(store, **run_kwargs):
    """검색 기록 + 정답 제출 한 건. (영수증, 제출 응답)을 돌려준다."""
    receipt = record_run(store, **run_kwargs)
    return receipt, store.submit(submission(receipt))


def approve(store, search_id, text="휘파람 전주로 시작하는 잔잔한 노래", *, revision=1, review_version=0, **kw):
    return store.apply_review(
        search_id, state="approved", reusable_text=text, note=kw.pop("note", ""),
        expected_revision=revision, expected_review_version=review_version,
        reviewer=kw.pop("reviewer", "시험 운영자"), **kw,
    )


def expire(database, search_id, *, seconds_ago=1):
    """문서의 만료 시각을 과거로 옮긴다(시험용 문서에만 쓴다)."""
    past = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    for name in ("search_runs", "search_feedback"):
        database[name].update_one({"_id": search_id}, {"$set": {"expires_at": past}})


# ---------------------------------------------------------------------------
# 색인·검색 시험용 — 실제 Qdrant(메모리 모드)와 결정적인 가짜 임베딩
# ---------------------------------------------------------------------------

TEST_DIM = 8
# 문장에 이 말이 있으면 그 축의 성분이 선다. 겹치는 말이 많을수록 코사인 유사도가 높다.
TOPICS = ("휘파람", "피아노", "여름", "이별", "기타", "새벽", "댄스", "발라드")


class KeywordEmbedder:
    """KoE5 대역. `embed_passages` 호출 모양만 같고, 벡터는 주제어 유무로 정해진다(L2 정규화)."""

    def __init__(self):
        self.calls = []
        self.fail = None          # 예외를 넣으면 다음 호출에서 던진다
        self.before = None        # 임베딩 직전에 부를 함수(경합 시험용)

    def embed_passages(self, passages, *, add_e5_prefix=True, normalize=True, **_):
        import numpy as np

        self.calls.append(list(passages))
        if self.before is not None:
            self.before()
        if self.fail is not None:
            raise self.fail
        rows = []
        for text in passages:
            vector = np.array([1.0 if topic in text else 0.0 for topic in TOPICS], dtype=np.float32)
            if not vector.any():
                vector[:] = 1.0 / TEST_DIM        # 주제어가 없는 문장: 모든 축에 고르게
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)

    # 색인 작업은 `embed(texts)` 모양으로 부른다.
    def __call__(self, texts):
        return self.embed_passages(list(texts))


def memory_qdrant():
    """디스크에 남지 않는 Qdrant. 로컬 파일 모드와 같은 엔진이다."""
    from src.vector_db.qdrant_backend import QdrantVectorClient

    return QdrantVectorClient(path=":memory:")


def seed_songs(vector_client, songs):
    """곡 텍스트 컬렉션에 곡 정보를 넣는다. `songs`: {song_id: {title, artist, ...}}"""
    from qdrant_client import models

    from src.vector_db.qdrant_backend import DENSE_VECTOR, collection_name, point_id
    from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

    vector_client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=TEST_DIM)
    vector_client.client.upsert(
        collection_name=collection_name(TEXT_HYBRID_INDEX_NAME),
        points=[
            models.PointStruct(id=point_id(song_id), vector={DENSE_VECTOR: [1.0] + [0.0] * (TEST_DIM - 1)},
                               payload={"song_id": song_id, **metadata})
            for song_id, metadata in songs.items()
        ],
    )
