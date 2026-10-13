"""승인 사례로 검색 후보를 보충하는 경로(④). Qdrant는 메모리 모드 실물, Mongo는 대역 + (있으면) 실물.

두 가지를 본다. ① 조회 서비스가 **지금 유효한 사례만** 고르는가 ② 라우터가 그 곡을 최종 후보 절단 전에
합치되, 꺼져 있거나 쓸 사례가 없거나 실패했을 때는 기존 후보·점수·리랭커 입력이 그대로인가.
"""
import asyncio
import math
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.backend.schemas.explain import SearchExplainOut, to_track_explain
from src.backend.schemas.query import LifeStageClue
from src.backend.schemas.search import ClarifyAnswer
from src.feedback import retrieval as retrieval_module
from src.feedback.case_index import CaseIndexBusy, FeedbackCaseIndex
from src.feedback.indexer import FeedbackIndexer
from src.feedback.retrieval import (
    DROP_DUPLICATE,
    DROP_INVALID,
    DROP_MISSING,
    DROP_OVER_LIMIT,
    DROP_REJECTED,
    DROP_SONG_GONE,
    FeedbackRetriever,
    build_case_query,
)
from src.feedback.review import CaseVersion
from src.feedback.settings import EMBEDDING_VERSION, RetrievalSettings, WorkerSettings, env_float, env_int
from src.retrieval import search_router as sr
from src.retrieval.explain import ExplainRecorder
from tests.feedback_fakes import (  # noqa: F401
    TEST_DIM,
    KeywordEmbedder,
    approve,
    expire,
    memory_qdrant,
    seed_songs,
    store_backend,
    submission,
    submit_case,
)
from tests.test_search_explain import (
    _analysis,
    _FakeClient,
    _FakeEmbedder,
    _FakeLyricsService,
    _FakeTextService,
    _StubReranker,
    _track,
)

SONGS = {
    "f1": {"title": "휘파람 부는 밤", "artist": "가수 F", "vocal_gender": "여성", "genre": "발라드"},
    "f2": {"title": "피아노 앞에서", "artist": "가수 G", "vocal_gender": "남성", "genre": "발라드"},
    "f3": {"title": "여름 기타", "artist": "가수 H", "vocal_gender": "혼성", "genre": "댄스"},
    # 텍스트 경로가 돌려주는 s2("곡2")와 **다른 제목**이다 — 이미 후보인 곡의 곡 정보가 색인 값으로 바뀌는지 본다.
    "s2": {"title": "색인에 적힌 다른 제목", "artist": "색인의 가수"},
}
SETTINGS = RetrievalSettings(top_cases=20, max_songs=3, min_similarity=0.86, fusion_weight=1.0, budget_seconds=2.0)
STEP = sr._FEEDBACK_ORDER_STEP


class Rig:
    def __init__(self, store, database):
        self.store, self.database = store, database
        self.client = memory_qdrant()
        seed_songs(self.client, SONGS)
        self.index = FeedbackCaseIndex(self.client, dim=TEST_DIM)
        self.embedder = KeywordEmbedder()
        self.indexer = FeedbackIndexer(store, self.index, KeywordEmbedder(), WorkerSettings())
        self.settings = SETTINGS
        self.store_calls = 0
        self.retriever = FeedbackRetriever(
            self.index, self._store, lambda: self.embedder, self.client, settings=lambda: self.settings)

    def _store(self):
        self.store_calls += 1
        return self.store

    def case(self, song_id, text, *, query="합성 질의", index=True):
        """표시 곡 중 `song_id`를 정답으로 고르고 운영자가 `text`로 승인한 사례. 모두 합성이다."""
        receipt = self.store.record_search({
            "query": query, "shown_ids": [song_id, "zz"], "candidate_ids": [song_id, "zz"],
            "shown": [{"id": song_id, "rank": 1, "title": song_id, "artist": ""}],
        })
        self.store.submit(submission(receipt, selected_song_id=song_id, reason="합성 사유"))
        approve(self.store, receipt.search_id, text)
        if index:
            assert self.indexer.run_once().ok
        return receipt

    def supplement(self, query="휘파람 불던 노래", **kw):
        return self.retriever.validate(self.retriever.find_cases(query), **kw)


@pytest.fixture
def rig(store_backend, monkeypatch):
    monkeypatch.setenv("FEEDBACK_RETRIEVAL_ENABLED", "1")
    made = Rig(*store_backend)
    yield made
    made.client.close()


# ---------------------------------------------------------------------------
# 질의문
# ---------------------------------------------------------------------------

def test_case_query_uses_the_effective_analysis_and_answers_but_no_birth_year():
    analysis = _analysis(
        original_query="중학교 때 듣던 휘파람 노래", lyric_semantic_query="헤어진 뒤의 그리움",
        life_stage=LifeStageClue(stage="middle", text="중학교 때", spans=[[0, 5]], confidence=0.9),
    )
    answers = [ClarifyAnswer(slot="birth_year", value="1998"), ClarifyAnswer(slot="vocal_gender", value="여성"),
               ClarifyAnswer(slot="genre", value="", skipped=True), ClarifyAnswer(slot="release_era", value="2010년대")]
    text = build_case_query(analysis, answers)
    assert text.splitlines() == ["듣던 휘파람 노래", "가사 의미: 헤어진 뒤의 그리움", "가수 성별: 여성", "발매 시기: 2010년대"]
    assert "1998" not in text and "중학교" not in text and "장르" not in text
    assert build_case_query(_analysis(original_query="휘파람 노래")) == "휘파람 노래"


# ---------------------------------------------------------------------------
# 조회 서비스 — 유효한 사례만
# ---------------------------------------------------------------------------

def test_a_similar_approved_case_supplies_its_song_with_current_metadata(rig):
    receipt = rig.case("f1", "휘파람 전주로 시작하는 잔잔한 노래")
    supplement = rig.supplement("휘파람 불던 노래")
    assert supplement.status == "valid_cases" and supplement.scanned == 1, "들어왔는지는 라우터가 절단 뒤에 정한다"
    item = supplement.items[0]
    assert (item.track.id, item.track.title, item.track.vocal_gender) == ("f1", "휘파람 부는 밤", "여성")
    assert item.case == CaseVersion(receipt.search_id, "f1", 1, 1) and item.rank == 1
    assert item.similarity == pytest.approx(1.0, abs=1e-5)
    assert rig.embedder.calls == [["query: 휘파람 불던 노래"]], "질의에는 E5 query 접두사를 쓴다"


def test_broad_similarity_below_the_threshold_supplies_nothing(rig):
    rig.case("f1", "휘파람 전주로 시작하는 잔잔한 노래")
    # 휘파람·피아노·여름을 함께 말한 질의는 '휘파람' 한 축만 가진 사례와 0.58만큼만 겹친다.
    supplement = rig.supplement("휘파람 피아노 여름")
    assert supplement.status == "no_case" and supplement.items == [] and rig.store_calls == 0, \
        "문턱을 넘는 사례가 없으면 Mongo도 읽지 않는다"
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "min_similarity": 0.5})
    assert rig.supplement("휘파람 피아노 여름").status == "valid_cases"


def test_several_cases_for_one_song_count_once_with_the_closest_case(rig):
    close = rig.case("f1", "휘파람 전주로 시작하는 노래", query="사례 1")
    for number in range(3):
        rig.case("f1", "휘파람과 피아노가 같이 나오는 노래", query=f"먼 사례 {number}")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "min_similarity": 0.6})
    supplement = rig.supplement("휘파람 불던 노래")
    assert [item.track.id for item in supplement.items] == ["f1"]
    assert supplement.items[0].case.feedback_id == close.search_id and supplement.items[0].similarity > 0.99
    assert supplement.dropped == {DROP_DUPLICATE: 3}, "제출이 여러 건이어도 곡 하나에 한 번이다"


def test_valid_songs_are_ranked_by_similarity(rig):
    rig.case("f3", "휘파람 피아노 여름 노래")
    rig.case("f1", "휘파람 노래")
    rig.case("f2", "휘파람 피아노 노래")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "min_similarity": 0.5, "max_songs": 2})
    supplement = rig.supplement("휘파람 불던 노래")
    # 곡 수 상한은 여기서 자르지 않는다 — 이미 후보인 곡을 뺀 뒤 라우터가 적용한다.
    assert [(item.track.id, item.rank) for item in supplement.items] == [("f1", 1), ("f2", 2), ("f3", 3)]
    assert supplement.dropped == {}


def test_rejected_and_deleted_songs_are_not_supplied(rig):
    rig.case("f1", "휘파람 노래")
    rig.case("gone", "휘파람 부는 노래")              # 곡 색인에 없는 곡(삭제됨)
    supplement = rig.supplement("휘파람 불던 노래", excluded_ids=["f1"])
    assert supplement.items == [] and supplement.status == "no_valid_case"
    assert supplement.dropped == {DROP_REJECTED: 1, DROP_SONG_GONE: 1}


@pytest.mark.parametrize("change", ["withdraw", "user_edit", "approval_cancelled", "text_edited", "expired", "deleted"])
def test_a_case_that_changed_after_indexing_is_not_used_even_if_its_point_remains(rig, change):
    """Qdrant 포인트 삭제는 한 주기 늦다. 그 사이의 검색은 Mongo의 최신 상태로 걸러야 한다."""
    receipt = rig.case("f1", "휘파람 노래")
    sid = receipt.search_id
    if change == "withdraw":
        rig.store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                    action="withdraw", selected_song_id=None, reason=""))
    elif change == "user_edit":
        rig.store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                    selected_song_id="zz", reason="다른 곡이었어요"))
    elif change == "approval_cancelled":
        rig.store.apply_review(sid, state="rejected", reusable_text="", note="취소", expected_revision=1,
                               expected_review_version=1, reviewer="운영자")
    elif change == "text_edited":
        approve(rig.store, sid, "피아노 노래", review_version=1)     # 검토 판 2. 포인트는 아직 판 1
    elif change == "expired":
        expire(rig.database, sid)                                    # TTL이 아직 지우지 않았다
    else:
        rig.database["search_feedback"].delete_many({"_id": sid})
    assert rig.index.retrieve([sid]), "포인트는 아직 남아 있다"
    supplement = rig.supplement("휘파람 불던 노래")
    assert supplement.items == [] and supplement.status == "no_valid_case"
    assert supplement.dropped == {DROP_MISSING if change == "deleted" else DROP_INVALID: 1}


def test_a_point_resurrected_by_a_late_job_is_not_used(rig):
    receipt = rig.case("f1", "휘파람 노래")
    saved = rig.database["search_feedback"].find_one({"_id": receipt.search_id})
    approve(rig.store, receipt.search_id, "피아노 노래", review_version=1)
    assert rig.indexer.run_once().indexed == 1                       # 검토 판 2가 올라갔다
    # 늦게 끝난 예전 판 작업이 판 1 벡터(휘파람)를 다시 올렸다.
    rig.index.upsert([(CaseVersion(receipt.search_id, "f1", 1, 1), [1.0] + [0.0] * (TEST_DIM - 1), saved["expires_at"])],
                     embedding_version=EMBEDDING_VERSION)
    supplement = rig.supplement("휘파람 불던 노래")
    assert supplement.items == [] and supplement.dropped == {DROP_INVALID: 1}


def test_expired_points_and_other_embedding_versions_are_filtered_by_qdrant(rig):
    receipt = rig.case("f1", "휘파람 노래", index=False)
    saved = rig.database["search_feedback"].find_one({"_id": receipt.search_id})
    case, vector = CaseVersion(receipt.search_id, "f1", 1, 1), [1.0] + [0.0] * (TEST_DIM - 1)
    rig.index.ensure()
    rig.index.upsert([(case, vector, datetime.now(timezone.utc) - timedelta(seconds=1))], embedding_version=EMBEDDING_VERSION)
    assert rig.retriever.find_cases("휘파람 불던 노래").hits == [], "만료 시각이 지난 포인트는 조회에서 빠진다"
    rig.index.upsert([(case, vector, saved["expires_at"])], embedding_version="koe5:old")
    assert rig.retriever.find_cases("휘파람 불던 노래").hits == [], "임베딩 판이 다른 포인트는 쓰지 않는다"
    rig.index.upsert([(case, vector, saved["expires_at"])], embedding_version=EMBEDDING_VERSION)
    assert len(rig.retriever.find_cases("휘파람 불던 노래").hits) == 1


def test_expiry_is_checked_at_validation_time_not_lookup_time(rig):
    rig.case("f1", "휘파람 노래")
    lookup = rig.retriever.find_cases("휘파람 불던 노래")
    assert len(lookup.hits) == 1
    clock = lambda: datetime.now(timezone.utc) + timedelta(days=31)   # noqa: E731 - 확인하는 사이 만료됐다
    late = FeedbackRetriever(rig.index, lambda: rig.store, lambda: rig.embedder, rig.client,
                             settings=lambda: SETTINGS, clock=clock)
    assert late.validate(lookup).items == []


def test_mongo_failure_supplies_nothing(rig, monkeypatch):
    rig.case("f1", "휘파람 노래")
    lookup = rig.retriever.find_cases("휘파람 불던 노래")
    monkeypatch.setattr(rig.store, "fetch_cases", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("mongo down")))
    with pytest.raises(TimeoutError):
        rig.retriever.validate(lookup)


def test_no_collection_yet_means_no_cases_no_embedding_and_no_mongo_read(rig):
    """색인 작업이 한 번도 돌지 않은 서버. 검색마다 KoE5를 헛돌리지 않는다."""
    assert rig.retriever.find_cases("휘파람 불던 노래").hits == []
    assert rig.supplement().status == "no_case" and rig.store_calls == 0
    assert rig.embedder.calls == [], "사례 컬렉션이 없으면 질의를 임베딩하지도 않는다"


def test_a_busy_index_is_reported_instead_of_blocking_the_search(rig):
    rig.case("f1", "휘파람 노래")
    held, release = threading.Event(), threading.Event()

    def hold():
        with rig.index._lock:
            held.set()
            release.wait(5)

    writer = threading.Thread(target=hold)
    writer.start()
    assert held.wait(5)
    try:
        with pytest.raises(CaseIndexBusy):
            rig.index.search([1.0] + [0.0] * (TEST_DIM - 1), top_k=5, now_ts=time.time(),
                             embedding_version=EMBEDDING_VERSION, min_similarity=0.5, lock_timeout=0.05)
    finally:
        release.set()
        writer.join(5)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-1", "abc", "1e9", ""])
def test_unsafe_settings_fall_back_to_defaults(monkeypatch, raw):
    for name in ("FEEDBACK_MIN_SIMILARITY", "FEEDBACK_FUSION_WEIGHT", "FEEDBACK_LOOKUP_BUDGET_SECONDS",
                 "FEEDBACK_TOP_CASES", "FEEDBACK_SUPPLEMENT_SONGS"):
        monkeypatch.setenv(name, raw)
    assert RetrievalSettings.from_env() == RetrievalSettings()
    monkeypatch.setenv("FEEDBACK_WORKER_INTERVAL_SECONDS", raw)
    monkeypatch.setenv("FEEDBACK_INDEX_BATCH", raw)
    assert WorkerSettings.from_env() == WorkerSettings()


def test_valid_settings_are_read_and_zero_weight_is_refused(monkeypatch):
    monkeypatch.setenv("FEEDBACK_MIN_SIMILARITY", "0.9")
    monkeypatch.setenv("FEEDBACK_FUSION_WEIGHT", "0")
    monkeypatch.setenv("FEEDBACK_SUPPLEMENT_SONGS", "0")
    monkeypatch.setenv("FEEDBACK_TOP_CASES", "7")
    settings = RetrievalSettings.from_env()
    assert settings.min_similarity == 0.9 and settings.top_cases == 7 and settings.display_k is None
    assert settings.fusion_weight == RetrievalSettings().fusion_weight and settings.max_songs == RetrievalSettings().max_songs
    assert env_float("X_UNSET", 0.3, low=0, high=1) == 0.3 and env_int("X_UNSET", 4, low=1, high=9) == 4
    assert all(math.isfinite(value) for value in (settings.min_similarity, settings.fusion_weight, settings.budget_seconds))


def test_the_display_count_cannot_come_from_the_environment(monkeypatch):
    """표시 수는 측정 프로세스가 `measurement_display`로만 정한다. 환경 변수는 읽지 않는다.

    측정기와 서버가 같은 `.env`를 읽으므로, 환경 변수로 받으면 서버에서도 후보 전체 요청(top_k = candidate_k)의
    표시 구간 보호가 풀린다.
    """
    from src.feedback.settings import measurement_display

    monkeypatch.setenv("FEEDBACK_DISPLAY_K", "10")
    assert RetrievalSettings.from_env().display_k is None
    assert RetrievalSettings.from_env().display_count(30, 30) == 30, "서버가 읽는 설정으로는 숨은 자리가 생기지 않는다"
    with measurement_display(10):
        assert RetrievalSettings.from_env().display_k == 10
        with measurement_display(5):
            assert RetrievalSettings.from_env().display_k == 5
        assert RetrievalSettings.from_env().display_k == 10
        with pytest.raises(RuntimeError):
            with measurement_display(7):
                raise RuntimeError("측정이 죽어도 되돌린다")
        assert RetrievalSettings.from_env().display_k == 10
    assert RetrievalSettings.from_env().display_k is None
    for bad in (0, -4, 101, True, 2.0, "10", None):
        with pytest.raises(ValueError):
            with measurement_display(bad):
                pass
    assert RetrievalSettings.from_env().display_k is None


# ---------------------------------------------------------------------------
# 라우터 — 최종 후보 절단 전에 합류, 그 밖에는 기존 그대로
# ---------------------------------------------------------------------------

def _hits(count=40, **extra):
    """텍스트 경로가 돌려줄 곡 40개. 서비스 설정(후보 30·표시 15)에서 31위부터는 후보 밖이다."""
    return [_track(f"s{i}", title=f"곡{i}", score=1.0 - i * 0.01, **extra) for i in range(1, count + 1)]


BASELINE_POOL = [f"s{i}" for i in range(1, 31)]


def _router(hits, retriever=None, *, reranker=None, lyrics_hits=None):
    return sr.SearchRouter(
        search_service=_FakeTextService(hits), image_embedder=_FakeEmbedder(), audio_embedder=_FakeEmbedder(),
        vector_client=_FakeClient(), reranker=reranker, max_workers=1, feedback_retriever=retriever,
        lyrics_search_service=_FakeLyricsService(lyrics_hits) if lyrics_hits is not None else None,
    )


def _search(router, analysis=None, **kw):
    """한 번 검색하고 비교할 것을 모두 모은다 — 결과·후보·점수·리랭커 입력·실행 기록."""
    out = {"pool": [], "tracks": [], "rerank_in": [], "feedback": []}
    recorder = kw.pop("recorder", None) or ExplainRecorder("q")
    kw.setdefault("top_k", 15)
    kw.setdefault("candidate_k", 30)
    try:
        results = asyncio.run(router.search(
            analysis or _analysis(original_query="휘파람 불던 노래"),
            candidate_ids_out=out["pool"], candidate_tracks_out=out["tracks"],
            rerank_input_tracks_out=out["rerank_in"], feedback_out=out["feedback"], recorder=recorder, **kw,
        ))
    finally:
        router.shutdown()
    out["results"] = [(t.id, t.score) for t in results]
    out["scores"] = [(t.id, t.score, t.retrieval_score) for t in out["tracks"]]
    out["rerank_in"] = [(t.id, t.score) for t in out["rerank_in"]]
    out["meta"] = [(t.id, t.title, t.artist, t.vocal_gender) for t in out["tracks"]]
    out["shown"] = [t.id for t in results]
    out["record"] = recorder.record
    return out


def _same_as_baseline(run, baseline):
    for key in ("pool", "scores", "rerank_in", "results", "meta"):
        assert run[key] == baseline[key], key


class _Spy:
    """꺼져 있을 때 아무것도 부르지 않는지 센다."""

    def __init__(self, enabled):
        self._enabled = enabled
        self.calls = []

    @property
    def enabled(self):
        self.calls.append("enabled")
        return self._enabled

    def settings(self):
        self.calls.append("settings")
        return SETTINGS

    def find_cases(self, *a, **k):
        self.calls.append("find_cases")
        return retrieval_module.CaseLookup(settings=SETTINGS)

    def validate(self, *a, **k):
        self.calls.append("validate")
        raise AssertionError("사례가 없으면 확인하지 않는다")


def test_switch_off_makes_no_feedback_calls_and_changes_nothing(rig, monkeypatch):
    rig.case("f1", "휘파람 노래")
    rig.store_calls = 0
    monkeypatch.setenv("FEEDBACK_RETRIEVAL_ENABLED", "0")
    reranker, baseline_reranker = _StubReranker(), _StubReranker()
    baseline = _search(_router(_hits(), reranker=baseline_reranker))
    off = _search(_router(_hits(), rig.retriever, reranker=reranker))
    _same_as_baseline(off, baseline)
    assert reranker.seen == baseline_reranker.seen
    assert rig.embedder.calls == [] and rig.store_calls == 0, "임베딩도 Mongo 조회도 없다"
    assert off["feedback"] == [] and off["record"].feedback is None
    assert SearchExplainOut.of(off["record"]).feedback_supplement is None

    spy = _Spy(enabled=False)
    _search(_router(_hits(), spy))
    assert spy.calls == ["enabled"], "스위치만 읽고 설정·조회·확인은 부르지 않는다"


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("FEEDBACK_RETRIEVAL_ENABLED", raising=False)
    from src.feedback.settings import retrieval_enabled

    assert retrieval_enabled() is False


def test_no_matching_case_leaves_candidates_scores_and_reranker_input_unchanged(rig):
    rig.case("f1", "피아노 노래")                       # 질의(휘파람)와 겹치지 않는 사례
    reranker, baseline_reranker = _StubReranker(), _StubReranker()
    baseline = _search(_router(_hits(), reranker=baseline_reranker))
    on = _search(_router(_hits(), rig.retriever, reranker=reranker))
    _same_as_baseline(on, baseline)
    assert reranker.seen == baseline_reranker.seen
    assert on["feedback"][0].status == "no_case" and on["feedback"][0].cases == []
    assert rig.store_calls == 0


def test_storage_failure_falls_back_to_the_existing_candidates(rig, monkeypatch, caplog):
    rig.case("f1", "휘파람 노래")
    monkeypatch.setattr(rig.store, "fetch_cases",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mongodb://user:secret@host down")))
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    _same_as_baseline(on, baseline)
    assert on["feedback"][0].status == "failed" and on["feedback"][0].cases == []
    assert "f1" not in on["pool"] and "secret" not in caplog.text
    assert not any(p.path == "feedback_case" for song in on["record"].songs.values() for p in song.paths)


@pytest.mark.parametrize("stage", ["embedding", "qdrant", "song_lookup"])
def test_lookup_failures_fall_back_to_the_existing_candidates(rig, monkeypatch, stage):
    rig.case("f1", "휘파람 노래")
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))   # noqa: E731
    if stage == "embedding":
        rig.embedder.fail = RuntimeError("model down")
    elif stage == "qdrant":
        monkeypatch.setattr(rig.index, "search", boom)
    else:
        monkeypatch.setattr(rig.client, "Index", boom)
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    _same_as_baseline(on, baseline)
    assert on["feedback"][0].status == "failed"


def test_a_slow_lookup_is_abandoned_within_the_budget(rig, monkeypatch):
    rig.case("f1", "휘파람 노래")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "budget_seconds": 0.15})
    original = rig.retriever.find_cases

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(rig.retriever, "find_cases", slow)
    baseline = _search(_router(_hits()))
    started = time.monotonic()
    on = _search(_router(_hits(), rig.retriever))
    _same_as_baseline(on, baseline)
    assert on["feedback"][0].status == "timeout" and on["feedback"][0].elapsed_ms < 400
    assert time.monotonic() - started < 2.0     # shutdown은 돌던 조회가 끝나길 기다린다(0.6초)


def test_a_slow_status_check_is_abandoned_within_the_budget(rig, monkeypatch):
    rig.case("f1", "휘파람 노래")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "budget_seconds": 0.15})
    original = rig.retriever.validate
    monkeypatch.setattr(rig.retriever, "validate", lambda *a, **k: (time.sleep(0.5), original(*a, **k))[1])
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    _same_as_baseline(on, baseline)
    assert on["feedback"][0].status == "timeout"


def test_a_failure_while_fusing_leaves_no_partial_scores(rig, monkeypatch):
    rig.case("f1", "휘파람 노래")
    rig.case("s2", "휘파람 부는 노래")                  # 이미 후보에 있는 곡도 보충 대상이다
    monkeypatch.setattr(sr, "fuse_feedback_supplement", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    _same_as_baseline(on, baseline)
    assert on["feedback"][0].status == "failed" and on["feedback"][0].cases == []


def _item(song_id, feedback_id, similarity, rank):
    return retrieval_module.SupplementItem(_track(song_id, title=f"곡 {song_id}"), CaseVersion(feedback_id * 32, song_id, 1, 1),
                                           similarity, rank)


def test_fusing_starts_new_songs_at_the_tail_and_does_not_mutate_inputs():
    # 보정 전 융합 점수. 보충 없는 최종 후보(순위 순)는 s1·s2·s3이고, s9는 융합 목록에는 있지만 후보 밖이다.
    fused = [("s1", 0.016), ("s2", 0.015), ("s3", 0.012), ("s9", 0.004)]
    meta = {sid: _track(sid) for sid in ("s1", "s2", "s3", "s9")}
    before = (list(fused), dict(meta))
    items = [_item("s2", "a", 0.97, 1), _item("f1", "b", 0.95, 2), _item("s9", "c", 0.9, 3), _item("f2", "d", 0.8, 4)]
    ranked, updates, cases = sr.fuse_feedback_supplement(fused, items, 1.0, meta, ["s1", "s2", "s3"], 2, 3)
    assert (fused, meta) == before
    scores = dict(ranked)
    assert (scores["s1"], scores["s2"], scores["s3"]) == (0.016, 0.015, 0.012), "기존 후보의 점수는 그대로다"
    # 두 곡이 들어오면 꼬리 두 곡(s2·s3)이 밀려난다. 그 맨 앞인 s2의 점수 바로 위에서 출발한다.
    assert scores["f1"] == pytest.approx(0.015 * (1 + 2 * STEP), rel=1e-12)
    assert scores["s9"] == pytest.approx(0.015 * (1 + 1 * STEP), rel=1e-12)
    assert scores["s1"] > scores["f1"] > scores["s9"] > scores["s2"] and "f2" not in scores
    assert list(updates) == ["f1"], "새 곡만 곡 정보를 채운다"
    assert [(c.song_id, c.outcome, c.new_to_pool, c.delta > 0) for c in cases] == [
        ("s2", "already_candidate", False, False), ("f1", "added", True, True),
        ("s9", "added", False, True), ("f2", "over_limit", True, False)]

    # 가중이 1보다 작으면 꼬리 아래에서 출발한다 — 명시 단서 보정이 더 많아야만 들어온다.
    low = dict(sr.fuse_feedback_supplement(fused, items[1:2], 0.5, meta, ["s1", "s2", "s3"], 2, 3)[0])
    assert low["f1"] == pytest.approx(0.012 * 0.5 * (1 + STEP)) and low["f1"] < low["s3"]
    # 후보가 꽉 차지 않았으면 밀려날 곡이 없다. 마지막 후보 바로 위에 선다.
    roomy = dict(sr.fuse_feedback_supplement(fused, items[1:2], 1.0, meta, ["s1", "s2", "s3"], 2, 30)[0])
    assert roomy["s3"] < roomy["f1"] < roomy["s2"]
    # 이미 꼬리 자리보다 높은 융합 점수를 가진 후보 밖 곡은 그대로 둔다.
    high = [("s1", 0.016), ("s2", 0.015), ("s9", 0.02)]
    ranked, _, cases = sr.fuse_feedback_supplement(high, [_item("s9", "c", 0.9, 1)], 1.0, meta, ["s1", "s2"], 2, 2)
    assert dict(ranked)["s9"] == 0.02 and cases[0].outcome == "no_gain" and cases[0].delta == 0.0
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            sr.fuse_feedback_supplement(fused, items, bad, meta, ["s1", "s2", "s3"], 2, 3)


def test_a_song_outside_the_candidates_enters_before_the_final_cut(rig):
    """핵심 동작. 기존 경로가 찾지 못한 정답이 최종 후보 30곡 안으로 들어오고, 30위였던 곡이 밀려난다."""
    receipt = rig.case("f1", "휘파람 전주로 시작하는 잔잔한 노래")
    reranker = _StubReranker(identity=True)
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever, reranker=reranker))
    assert "f1" not in baseline["pool"] and baseline["pool"] == BASELINE_POOL
    assert "f1" in on["pool"] and len(on["pool"]) == 30, "후보 수(candidate_k)는 그대로다"
    assert len(on["results"]) == 15, "표시 수(top_k)도 그대로다"
    assert set(baseline["pool"]) - set(on["pool"]) == {"s30"}
    assert on["pool"] == BASELINE_POOL[:29] + ["f1"], "보충 곡은 꼬리에 서고 기존 후보의 순서는 그대로다"
    assert on["scores"][:29] == baseline["scores"][:29], "남은 기존 후보의 점수도 그대로다"

    record = on["feedback"][0]
    assert record.status == "applied" and record.entered_ids == ["f1"] and record.displaced_ids == ["s30"]
    case = record.cases[0]
    assert (case.feedback_id, case.source_revision, case.review_version) == (receipt.search_id, 1, 1)
    assert case.new_to_pool and case.in_candidates and case.entered and case.outcome == "added"
    assert case.delta == pytest.approx((1 / 90) * (1 + STEP), rel=1e-12), "밀려날 꼬리 후보(s30)의 융합 점수 바로 위"
    assert on["record"].feedback is record

    # 리랭커는 현재 곡 정보로 판단한다. 재사용 문장·사유는 어디에도 실리지 않는다.
    entered = next(t for t in on["tracks"] if t.id == "f1")
    assert (entered.title, entered.artist, entered.vocal_gender) == ("휘파람 부는 밤", "가수 F", "여성")
    assert reranker.seen[0] == on["pool"]
    blob = " ".join(str(t.model_dump()) for t in on["tracks"])
    assert "전주로 시작하는 잔잔한" not in blob and "합성 사유" not in blob and receipt.search_id not in blob


def test_reranker_gets_the_original_query_and_no_feedback_text(rig):
    rig.case("f1", "휘파람 전주로 시작하는 잔잔한 노래")
    seen = []

    class Reranker(_StubReranker):
        def rerank_run(self, query, tracks, top_k):
            seen.append(query)
            return super().rerank_run(query, tracks, top_k)

    _search(_router(_hits(), rig.retriever, reranker=Reranker(identity=True)))
    assert seen == ["휘파람 불던 노래"]


def test_a_song_already_in_the_candidates_is_left_exactly_where_it_was(rig):
    """첫 버전의 피드백은 후보에 넣는 것까지다. 이미 후보인 곡의 점수·순위를 올리지 않는다."""
    rig.case("s2", "휘파람 노래")
    reranker, baseline_reranker = _StubReranker(), _StubReranker()
    baseline = _search(_router(_hits(), reranker=baseline_reranker))
    on = _search(_router(_hits(), rig.retriever, reranker=reranker))
    _same_as_baseline(on, baseline)
    assert reranker.seen == baseline_reranker.seen
    record = on["feedback"][0]
    assert record.status == "already_in_candidates" and record.entered_ids == [] and record.displaced_ids == []
    assert record.dropped == {"already_candidate": 1}
    assert [(c.song_id, c.outcome, c.delta, c.entered, c.in_candidates) for c in record.cases] == [
        ("s2", "already_candidate", 0.0, False, True)]
    assert not any(p.path == "feedback_case" for p in on["record"].songs["s2"].paths), "기여가 없으면 경로로 적지 않는다"
    assert SearchExplainOut.of(on["record"]).feedback_supplement.supplemented == 0


def test_only_the_tail_changes_and_remaining_candidates_keep_their_scores(rig):
    """보충을 켰을 때 달라지는 것은 들어온 곡과 밀려난 꼬리뿐이다. 남은 후보의 점수와 순서는 그대로다."""
    rig.case("s2", "휘파람 부는 노래")                  # 이미 후보
    rig.case("f1", "휘파람 노래")
    rig.case("f2", "휘파람 피아노 노래")
    rig.case("f3", "휘파람 피아노 여름 노래")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "min_similarity": 0.5, "max_songs": 2})
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    record = on["feedback"][0]
    assert record.status == "applied" and record.entered_ids == ["f1", "f2"] and record.displaced_ids == ["s29", "s30"]
    assert record.dropped == {"already_candidate": 1, DROP_OVER_LIMIT: 1}, "상한은 후보 밖 곡에만 센다"
    by_song = {c.song_id: c for c in record.cases}
    assert by_song["f1"].delta == pytest.approx((1 / 89) * (1 + 2 * STEP), rel=1e-12)
    assert by_song["f2"].delta == pytest.approx((1 / 89) * (1 + 1 * STEP), rel=1e-12)
    assert by_song["f3"].delta == 0.0 and by_song["s2"].delta == 0.0
    assert on["pool"] == BASELINE_POOL[:28] + ["f1", "f2"], "유사도가 높은 사례의 곡이 앞에 선다"
    assert on["scores"][:28] == baseline["scores"][:28], "남은 기존 후보는 점수도 순서도 그대로다"
    paths = {sid: [p.rank for p in on["record"].songs[sid].paths if p.path == "feedback_case"] for sid in ("f1", "f2")}
    assert paths == {"f1": [1], "f2": [2]}


def test_a_weak_supplement_does_not_force_its_way_in(rig):
    """가중이 1보다 작으면 꼬리 아래에서 출발해 절단을 넘지 못한다. 보충은 자리를 보장하지 않는다."""
    rig.case("f1", "휘파람 노래")
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "fusion_weight": 0.9})
    baseline = _search(_router(_hits()))
    on = _search(_router(_hits(), rig.retriever))
    assert on["pool"] == baseline["pool"] and "f1" not in on["pool"]
    record = on["feedback"][0]
    assert record.status == "not_entered" and record.entered_ids == [], "점수만 받고 못 들어온 것은 applied가 아니다"
    assert [(c.song_id, c.outcome, c.in_candidates, c.entered) for c in record.cases] == [("f1", "added", False, False)]
    summary = SearchExplainOut.of(on["record"]).feedback_supplement
    assert summary.label == "이전 사용자 제보를 참고했지만 후보에 새로 포함된 곡은 없습니다" and summary.entered == 0


def test_a_rejected_song_is_never_supplied_back(rig):
    rig.case("f1", "휘파람 노래")
    on = _search(_router(_hits(), rig.retriever), exclude_ids=["f1", "s1"])
    assert "f1" not in on["pool"] and "s1" not in on["pool"] and len(on["pool"]) == 30
    assert on["feedback"][0].dropped == {DROP_REJECTED: 1} and on["feedback"][0].cases == []


def test_a_case_withdrawn_while_the_search_is_running_is_dropped_before_fusing(rig, monkeypatch):
    """상태 확인 시점은 후보에 합치기 직전이다. 조회와 그 사이에 철회되면 이 요청에서도 빠진다."""
    receipt = rig.case("f1", "휘파람 노래")
    original = rig.retriever.find_cases

    def lookup_then_user_withdraws(*args, **kwargs):
        found = original(*args, **kwargs)
        rig.store.submit(submission(receipt, submission_id="test_submission_2", expected_revision=1,
                                    selected_song_id=None, action="withdraw", reason=""))
        return found

    monkeypatch.setattr(rig.retriever, "find_cases", lookup_then_user_withdraws)
    on = _search(_router(_hits(), rig.retriever))
    assert "f1" not in on["pool"] and on["feedback"][0].status == "no_valid_case"
    assert on["feedback"][0].dropped == {DROP_INVALID: 1}


def _ledger(song):
    """설명의 장부: 최종 점수에 남은 경로 기여 + 가감점."""
    return sum(p.delta for p in song.live_paths()) + sum(a.delta for a in song.adjustments)


def _female_at(*positions):
    """텍스트 경로 40곡 가운데 지정한 순번만 여성 가수다(나머지 남성)."""
    return [_track(f"s{i}", title=f"곡{i}", score=1.0 - i * 0.01, vocal_gender="여성" if i in positions else "남성")
            for i in range(1, 41)]


@pytest.mark.parametrize("positions", [(10, 20, 30, 40), tuple(range(1, 41, 2))], ids=["여성 4곡", "절반이 여성"])
@pytest.mark.parametrize("fallback", ["no_reranker", "reranker_raises", "reranker_inner_fallback"])
def test_an_entrant_matching_the_explicit_clue_still_stays_at_the_tail(rig, positions, fallback):
    """리뷰 FR-01. 질의가 말한 성별과 맞는 보충 곡은 보정을 받아 절단을 넘지만, 자리는 후보 끝이다.

    보정 한 번(1/61)이 후보 30곡의 융합 점수 폭과 비슷해서, 자리를 고정하지 않으면 '단서가 맞는 묶음'
    안으로 올라가 리랭킹이 없거나 실패한 요청에서 검색 근거 없이 표시된다.
    """
    rig.case("f1", "휘파람 노래")                       # f1은 여성
    analysis = _analysis(original_query="휘파람 불던 노래", vocal_gender="여성")
    reranker = {"no_reranker": None, "reranker_raises": _StubReranker(fail=True),
                "reranker_inner_fallback": _StubReranker(inner_fallback=True)}[fallback]
    baseline = _search(_router(_female_at(*positions)), analysis)
    on = _search(_router(_female_at(*positions), rig.retriever, reranker=reranker), analysis)
    assert on["pool"][-1] == "f1" and on["pool"][:29] == baseline["pool"][:29], "들어오되 끝에, 나머지는 그대로"
    assert on["scores"][:29] == baseline["scores"][:29]
    assert "f1" not in on["shown"] and len(on["shown"]) == 15, "리랭킹 없이는 표시 구간에 들지 않는다"
    record = on["feedback"][0]
    assert record.status == "applied" and record.entered_ids == ["f1"] and record.displaced_ids == [baseline["pool"][-1]]
    song = on["record"].songs["f1"]
    rules = [a.rule for a in song.adjustments]
    assert "vocal_gender_match" in rules and "feedback_tail" in rules, "보정은 받았고(진입 판정), 그만큼 끝으로 내렸다"
    assert _ledger(song) == pytest.approx(song.retrieval_score, abs=1e-12), "장부(경로 기여 + 가감점)가 검색 점수와 맞는다"
    assert song.retrieval_score < min(score for sid, score, _ in on["scores"] if sid != "f1")


def test_with_clarify_answers_the_entrant_replaces_the_final_tail_not_an_answer_match(rig):
    """재질문 답이 있는 요청. 절단 시점의 꼴찌(s30)는 답(여성)과 맞아 보너스로 3위가 될 곡이다.

    보충 곡이 그 곡을 밀어내면 표시 결과가 바뀐다. 자리를 내주는 것은 **답변 보너스까지 반영한 순서의 꼴찌**여야 하고,
    끝자리를 뺀 나머지는 보충이 없을 때와 점수·순서가 같아야 한다.
    """
    rig.case("f1", "휘파람 노래")                       # f1도 여성이다 — 답과 맞지만 자리는 끝이다
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
    baseline = _search(_router(_female_at(10, 20, 30, 40)), answers=answers)
    on = _search(_router(_female_at(10, 20, 30, 40), rig.retriever), answers=answers)
    assert baseline["pool"][:3] == ["s10", "s20", "s30"], "보충이 없으면 답과 맞는 세 곡이 앞에 선다"
    assert on["pool"] == baseline["pool"][:29] + ["f1"] and on["scores"][:29] == baseline["scores"][:29]
    assert on["shown"] == baseline["shown"] and "f1" not in on["shown"]
    assert on["rerank_in"][:29] == baseline["rerank_in"][:29] and on["rerank_in"][-1][0] == "f1", "리랭커 입력에서도 끝이다"
    record = on["feedback"][0]
    assert record.entered_ids == ["f1"] and record.displaced_ids == [baseline["pool"][-1]] != ["s30"]
    song = on["record"].songs["f1"]
    assert "answer_match" not in [a.rule for a in song.adjustments], "들어온 곡에는 답변 보너스를 주지 않는다"
    assert _ledger(song) == pytest.approx(song.retrieval_score, abs=1e-12)
    assert song.retrieval_score < on["scores"][28][1], "남은 후보의 최저점보다 아래다"


def test_context_facts_follow_the_final_candidates_after_the_tail_is_replaced(rig):
    """Context 경로(배경 지식)와 함께 돌 때. 응답에 실을 Context 사실은 **최종 후보에 남은 곡**의 것만 내보낸다.

    절단 직후에 모으면 어긋난다 — 재질문 답이 있으면 절단 시점의 꼴찌(s30, 답과 맞아 3위가 될 곡)는 남고
    답변 보너스 뒤의 꼴찌(s29)가 자리를 내주기 때문이다. 보충으로 들어온 곡이 Context 경로에도 걸렸으면
    그 사실도 함께 나간다. Context 가중을 아주 작게 둬 순위는 텍스트 경로 그대로다.
    """
    from src.backend.schemas.query import ContextClue
    from src.retrieval.context_route import ContextRouteHit

    clue = ContextClue(target="", relation="삽입곡", search_query="드라마에 나온 노래", confidence=0.9)
    context_ids = ["s5", "s29", "s30", "f1"]            # f1은 텍스트 경로에 없다 — Context에만 걸린 후보 밖 곡
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]

    def run(retriever=None):
        router = _router(_female_at(10, 20, 30, 40), retriever)
        router._context_search, router._context_weight = object(), 1e-9
        router._search_context = lambda analysis, pool_k: [
            sr._ContextTrack(
                ContextRouteHit(song_id, 1.0 / (60 + rank), clue, None),
                _track(song_id, title=f"색인 제목 {song_id}", score=0.0, vocal_gender="여성" if song_id in ("s30", "f1") else "남성"),
            )
            for rank, song_id in enumerate(context_ids, start=1)
        ]
        facts = []
        out = _search(router, _analysis(original_query="휘파람 불던 노래", context_clues=[clue]),
                      answers=answers, context_hits_out=facts)
        out["facts"] = [hit.song_id for hit in facts]
        return out

    rig.case("f1", "휘파람 노래")
    baseline, on = run(), run(rig.retriever)
    assert baseline["pool"][:3] == ["s10", "s20", "s30"] and baseline["pool"][-1] == "s29"
    assert baseline["facts"] == ["s5", "s29", "s30"], "보충이 없으면 후보에 남은 Context 곡 그대로다(f1은 후보 밖)"
    assert on["pool"] == baseline["pool"][:29] + ["f1"] and on["scores"][:29] == baseline["scores"][:29]
    assert on["feedback"][0].entered_ids == ["f1"] and on["feedback"][0].displaced_ids == ["s29"]
    assert on["facts"] == ["s5", "s30", "f1"], "밀려난 s29의 사실은 빠지고, 남은 s30과 들어온 f1의 사실은 실린다"
    assert on["shown"] == baseline["shown"]


def test_the_pre_bonus_order_given_to_an_answer_aware_reranker_has_the_same_songs(rig, monkeypatch):
    """실험 스위치 CLARIFY_RERANK_INPUT_ORDER=pre_bonus — 리랭커가 받는 '보너스 전 순서'도 같은 곡들이고 보충 곡은 끝이다."""
    from tests.test_router_rerank_input_order import _AnswerReranker

    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rig.case("f1", "휘파람 노래")
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
    baseline_reranker, reranker = _AnswerReranker(), _AnswerReranker()
    baseline = _search(_router(_female_at(10, 20, 30, 40), reranker=baseline_reranker), answers=answers)
    on = _search(_router(_female_at(10, 20, 30, 40), rig.retriever, reranker=reranker), answers=answers)
    assert set(reranker.seen[0]) == set(on["pool"]) and reranker.seen[0][-1] == "f1"
    kept = [sid for sid in baseline_reranker.seen[0] if sid in set(on["pool"])]
    assert reranker.seen[0][:29] == kept, "보너스 전 순서에서도 남은 곡의 순서는 그대로다"


def test_an_entrant_contradicting_the_explicit_clue_does_not_enter(rig):
    """진입 여부는 보정까지 본 점수로 정한다. 후보가 모두 여성인데 보충 곡(f2)만 남성이면 절단을 넘지 못한다."""
    rig.case("f2", "휘파람 부는 노래")                  # f2는 남성
    analysis = _analysis(original_query="휘파람 불던 노래", vocal_gender="여성")
    baseline = _search(_router(_hits(vocal_gender="여성")), analysis)
    on = _search(_router(_hits(vocal_gender="여성"), rig.retriever), analysis)
    _same_as_baseline(on, baseline)
    record = on["feedback"][0]
    assert record.status == "not_entered" and record.entered_ids == [] and record.displaced_ids == []
    assert [(c.song_id, c.outcome, c.in_candidates) for c in record.cases] == [("f2", "added", False)]
    summary = SearchExplainOut.of(on["record"]).feedback_supplement
    assert (summary.status, summary.supplemented, summary.entered) == ("not_entered", 1, 0)
    assert "포함한 곡이 있습니다" not in summary.label, "들어온 곡이 없으면 포함했다고 말하지 않는다(리뷰 FR-05)"


def test_candidate_and_display_sizes_are_kept_and_answers_only_reorder(rig):
    rig.case("f1", "휘파람 노래")
    plain = _search(_router(_hits(vocal_gender="남성"), rig.retriever), top_k=3, candidate_k=20)
    answered = _search(_router(_hits(vocal_gender="남성"), rig.retriever),
                       answers=[ClarifyAnswer(slot="vocal_gender", value="남성")], top_k=3, candidate_k=20)
    assert len(answered["pool"]) == 20 and len(answered["results"]) == 3, "candidate_k·top_k 계약은 그대로다"
    assert set(answered["pool"]) == set(plain["pool"]), "답변은 후보 풀을 바꾸지 않고 순위만 바꾼다"
    assert plain["pool"][-1] == answered["pool"][-1] == "f1"


@pytest.mark.parametrize("top_k, candidate_k, entrants", [(3, 3, 0), (5, 5, 0), (15, 16, 1), (15, 17, 2), (15, 30, 3)])
def test_supplements_never_take_displayed_slots(rig, top_k, candidate_k, entrants):
    """리뷰 FR-02. 보충 곡 수는 '리랭커 없이는 표시되지 않는 자리 수'(후보 수 − 표시 수)를 넘지 않는다.

    후보 수 = 표시 수면 보충하지 않는다. 그러지 않으면 후보 전체가 보충 곡으로 바뀔 수 있다.
    """
    for song_id, text in (("f1", "휘파람 노래"), ("f2", "휘파람 피아노 노래"), ("f3", "휘파람 피아노 여름 노래")):
        rig.case(song_id, text)
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "min_similarity": 0.5})
    baseline = _search(_router(_hits()), top_k=top_k, candidate_k=candidate_k)
    on = _search(_router(_hits(), rig.retriever), top_k=top_k, candidate_k=candidate_k)
    record = on["feedback"][0]
    assert on["shown"] == baseline["shown"], "표시 구간은 그대로다"
    assert len(record.entered_ids) == entrants and on["pool"][:candidate_k - entrants] == baseline["pool"][:candidate_k - entrants]
    assert on["pool"][candidate_k - entrants:] == ["f1", "f2", "f3"][:entrants]
    assert record.status == ("applied" if entrants else "not_entered")
    assert record.dropped.get("over_limit", 0) == 3 - entrants


def test_a_whole_pool_request_needs_the_display_count_to_supplement(rig, monkeypatch):
    """측정기처럼 후보 전체를 요청하면(top_k = candidate_k) 표시 수를 따로 알려 줘야 보충 자리가 생긴다.

    서비스 요청(top_k < 후보 수)에서는 그 값을 읽지 않는다 — 표시 구간 보호를 이 설정으로 풀 수 없다.
    """
    for song_id, text in (("f1", "휘파람 노래"), ("f2", "휘파람 피아노 노래"), ("f3", "휘파람 피아노 여름 노래")):
        rig.case(song_id, text)
    base = {**SETTINGS.__dict__, "min_similarity": 0.5}
    baseline = _search(_router(_hits()), top_k=30, candidate_k=30)

    rig.settings = RetrievalSettings(**base)                       # 표시 수를 모른다 → 숨은 자리 0
    unknown = _search(_router(_hits(), rig.retriever), top_k=30, candidate_k=30)
    assert unknown["pool"] == baseline["pool"] and unknown["feedback"][0].status == "not_entered"

    monkeypatch.setenv("FEEDBACK_DISPLAY_K", "10")                 # 서버 환경에 잘못 들어간 값 — 읽지 않는다
    rig.settings = RetrievalSettings(**{**base, "display_k": RetrievalSettings.from_env().display_k})
    leaked = _search(_router(_hits(), rig.retriever), top_k=30, candidate_k=30)
    assert leaked["pool"] == baseline["pool"] and leaked["shown"] == baseline["shown"]

    rig.settings = RetrievalSettings(**{**base, "display_k": 10})  # 측정기가 상위 10곡을 쓴다
    known = _search(_router(_hits(), rig.retriever), top_k=30, candidate_k=30)
    assert known["pool"] == baseline["pool"][:27] + ["f1", "f2", "f3"], "끝자리 세 곡만 바뀐다"
    assert known["feedback"][0].settings["display_k"] == 10

    rig.settings = RetrievalSettings(**{**base, "display_k": 1})   # 서비스 요청에서는 무시한다
    service = _search(_router(_hits(), rig.retriever), top_k=15, candidate_k=16)
    assert len(service["feedback"][0].entered_ids) == 1 and service["shown"] == _search(
        _router(_hits()), top_k=15, candidate_k=16)["shown"]
    assert service["feedback"][0].settings["display_k"] == 15

    settings = RetrievalSettings(display_k=10)
    assert (settings.display_count(15, 30), settings.display_count(30, 30), settings.display_count(5, 5)) == (15, 10, 5)
    assert RetrievalSettings().display_count(30, 30) == 30 and RetrievalSettings(display_k=50).display_count(30, 30) == 30


def test_no_supplement_when_there_are_fewer_candidates_than_displayed(rig):
    """기존 후보가 표시 수보다 적으면 끝에 세워도 표시 구간 안이다. 보충하지 않는다."""
    rig.case("f1", "휘파람 노래")
    baseline = _search(_router(_hits(10)))
    on = _search(_router(_hits(10), rig.retriever))
    _same_as_baseline(on, baseline)
    record = on["feedback"][0]
    assert record.status == "not_entered" and record.dropped == {"over_limit": 1}


def test_pinning_moves_only_entrants_and_never_raises_a_score():
    ranked = [("a", 0.05), ("f1", 0.04), ("b", 0.03), ("f2", 0.02), ("c", 0.01), ("f3", 0.001)]
    recorder = ExplainRecorder("q")
    pinned = sr.pin_feedback_entrants(ranked, ["f2", "f1", "f3"], recorder)
    assert [sid for sid, _ in pinned] == ["a", "b", "c", "f2", "f1", "f3"], "넘겨준 순서(유사도 순)대로 끝에 선다"
    scores = dict(pinned)
    assert (scores["a"], scores["b"], scores["c"]) == (0.05, 0.03, 0.01)
    assert 0.01 > scores["f2"] > scores["f1"] > scores["f3"] == 0.001, "이미 더 낮은 점수는 올리지 않는다"
    assert [score for _, score in pinned] == sorted((score for _, score in pinned), reverse=True), "점수 순서와 어긋나지 않는다"
    deltas = {sid: [a.delta for a in recorder.record.songs[sid].adjustments] for sid in ("f1", "f2")}
    assert deltas["f1"] == [pytest.approx(scores["f1"] - 0.04)] and deltas["f2"] == [pytest.approx(scores["f2"] - 0.02)]
    assert recorder.record.songs["f3"].adjustments == [], "바뀌지 않은 점수에는 조정을 적지 않는다"
    assert all(recorder.record.songs[sid].feedback_entrant for sid in ("f1", "f2", "f3")), "들어온 곡이라는 표시는 셋 다 남는다"
    assert "a" not in recorder.record.songs
    assert sr.pin_feedback_entrants(ranked, []) == ranked and sr.pin_feedback_entrants(ranked, ["zz"]) == ranked


def test_the_feedback_pool_is_drained_on_shutdown(rig, monkeypatch):
    """조회 스레드도 벡터 클라이언트를 쓴다. shutdown은 돌던 조회가 끝난 뒤에 돌아온다."""
    rig.case("f1", "휘파람 노래")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = rig.retriever.find_cases

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(rig.retriever, "find_cases", slow)
    rig.settings = RetrievalSettings(**{**SETTINGS.__dict__, "budget_seconds": 0.05})
    router = _router(_hits(), rig.retriever)
    asyncio.run(router.search(_analysis(original_query="휘파람 불던 노래"), top_k=15, candidate_k=30))  # 시간 초과로 끝난다
    assert started.is_set() and not finished.is_set()
    closer = threading.Thread(target=router.shutdown)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive(), "돌던 조회가 끝나기 전에는 shutdown이 돌아오지 않는다"
    release.set()
    closer.join(5)
    assert not closer.is_alive() and finished.is_set()


def test_exact_lyric_protection_is_kept(rig):
    rig.case("f1", "휘파람 노래")
    lyric = _track("s4", title="곡4", lyric_match_type="exact", lyric_match_score=0.95)
    analysis = _analysis(original_query="휘파람 불던 노래",
                         lyric_clues=[{"text": "휘파람 불던", "kind": "verbatim", "confidence": 0.95}])
    baseline = _search(_router(_hits(), reranker=_StubReranker(), lyrics_hits=[lyric]), analysis)
    on = _search(_router(_hits(), rig.retriever, reranker=_StubReranker(), lyrics_hits=[lyric]), analysis)
    assert baseline["results"][0][0] == on["results"][0][0] == "s4", "가사 exact 보호 곡이 그대로 1위다"
    assert "f1" in on["pool"]


def test_ablation_modes_do_not_use_feedback(rig):
    rig.case("f1", "휘파람 노래")
    for kwargs in ({"force_weights": [1.0, 0.0, 0.0]}, {"disable_boost": True}):
        run = _search(_router(_hits(), rig.retriever), **kwargs)
        assert run["feedback"] == [] and "f1" not in run["pool"]
    assert rig.embedder.calls == []


def test_recording_does_not_change_the_supplemented_result(rig):
    from src.retrieval.explain import NULL_RECORDER

    rig.case("f1", "휘파람 노래")
    recorded = _search(_router(_hits(), rig.retriever, reranker=_StubReranker()))
    silent = _search(_router(_hits(), rig.retriever, reranker=_StubReranker()), recorder=NULL_RECORDER)
    _same_as_baseline(silent, recorded)
    assert silent["feedback"][0].entered_ids == ["f1"], "기록을 꺼도 실행 기록 통로로는 받을 수 있다"


def test_user_facing_output_names_only_the_source(rig):
    receipt = rig.case("f1", "휘파람 전주로 시작하는 잔잔한 노래")
    on = _search(_router(_hits(), rig.retriever))
    summary = SearchExplainOut.of(on["record"]).feedback_supplement
    assert summary.model_dump() == {
        "status": "applied", "label": "이전 사용자 제보를 참고해 후보에 포함한 곡이 있습니다",
        "supplemented": 1, "entered": 1,
    }
    public = SearchExplainOut.of(on["record"]).model_dump_json()
    track = to_track_explain(on["record"].songs["f1"], reorder_stage=on["record"].reorder_stage)
    path = next(p for p in track.paths if p.path == "feedback_case")
    assert path.label == "이전 사용자 제보 참고" and path.detail == ""
    for text in (public, track.model_dump_json()):
        assert receipt.search_id not in text and "전주로 시작하는" not in text and "합성 사유" not in text
        assert "정답" not in text and "확인됨" not in text

    from src.retrieval.user_explanation import build_user_explanation

    explanation = build_user_explanation(
        query="휘파람 불던 노래", analysis=_analysis(original_query="휘파람 불던 노래"),
        search_analysis=_analysis(original_query="휘파람 불던 노래"), answers=[],
        track=next(t for t in on["tracks"] if t.id == "f1"), explain=on["record"].songs["f1"],
        evidence=[], rerank_skip_reason="",
    )
    assert "이전 사용자 제보를 참고해 후보에 포함한 곡이에요" in explanation.summary
    assert "사실로 확인한 것은 아니에요" in explanation.summary


def _explain_entrant(on, analysis, answers=()):
    from src.retrieval.user_explanation import build_user_explanation

    return build_user_explanation(
        query=analysis.original_query, analysis=analysis, search_analysis=analysis, answers=list(answers),
        track=next(t for t in on["tracks"] if t.id == "f1"), explain=on["record"].songs["f1"],
        evidence=[], rerank_skip_reason="",
    )


def test_an_entrant_with_matching_clues_still_names_its_source_and_claims_no_ranking(rig):
    """리뷰 FR-07. 성별 단서가 맞는 보충 곡 — 보이는 설명이 출처를 말해야 하고, 상쇄된 보정을 '검색 점수에 반영'이라 하면 안 된다."""
    from src.retrieval.user_explanation import BASIS_ENTRY_ONLY, FEEDBACK_SOURCE_SENTENCE

    rig.case("f1", "휘파람 노래")                       # f1은 여성
    analysis = _analysis(original_query="여자 가수가 부른 휘파람 불던 노래", vocal_gender="여성")
    on = _search(_router(_female_at(10, 20, 30, 40), rig.retriever, reranker=_StubReranker()), analysis)
    assert on["shown"][0] == "f1", "뒤집는 리랭커가 끝자리 곡을 1위로 올렸다"
    song = on["record"].songs["f1"]
    assert song.feedback_entrant and [a.rule for a in song.adjustments] == ["vocal_gender_match", "feedback_tail"]

    explanation = _explain_entrant(on, analysis)
    assert explanation.summary.startswith(FEEDBACK_SOURCE_SENTENCE)
    assert "여성 보컬 단서는 곡 정보와 일치해요" in explanation.summary, "곡 자료와 맞는다는 사실은 말한다"
    assert "검색 점수에 반영" not in explanation.summary
    gender = next(c for c in explanation.clue_checks if c.slot == "vocal_gender")
    assert gender.status == "match" and gender.basis_label == BASIS_ENTRY_ONLY
    assert [b.code for b in gender.ranking_basis] == ["vocal_gender_match"], "가산 기록과의 연결은 지우지 않는다"
    assert all("검색 점수에 반영" not in c.basis_label for c in explanation.clue_checks)

    track = to_track_explain(song, reorder_stage=on["record"].reorder_stage, user_explanation=explanation)
    assert track.feedback_entrant is True and track.user_explanation.summary == explanation.summary

    # 같은 질의의 기존 후보(검색이 찾은 여성 곡)는 예전 그대로 말한다.
    other = on["record"].songs["s10"]
    from src.retrieval.user_explanation import build_user_explanation
    plain = build_user_explanation(query=analysis.original_query, analysis=analysis, search_analysis=analysis, answers=[],
                                   track=next(t for t in on["tracks"] if t.id == "s10"), explain=other)
    assert not other.feedback_entrant and "검색 점수에 반영됐어요" in plain.summary
    assert "이전 사용자 제보" not in plain.summary


def test_an_entrant_reached_only_through_an_answer_names_its_source_too(rig):
    from src.retrieval.user_explanation import FEEDBACK_SOURCE_SENTENCE

    rig.case("f1", "휘파람 노래")
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
    analysis = _analysis(original_query="휘파람 불던 노래")
    on = _search(_router(_female_at(10, 20, 30, 40), rig.retriever, reranker=_StubReranker()), analysis, answers=answers)
    assert on["record"].songs["f1"].feedback_entrant, "점수를 낮추지 않았어도(조정 기록 없음) 들어온 곡이라는 표시는 남는다"
    assert on["record"].songs["f1"].adjustments == []
    explanation = _explain_entrant(on, analysis, answers)
    assert explanation.summary.startswith(FEEDBACK_SOURCE_SENTENCE) and "검색 점수에 반영" not in explanation.summary
    gender = next(c for c in explanation.clue_checks if c.slot == "vocal_gender")
    assert gender.ranking_basis == [] and gender.basis_label == "순위 반영 기록 없음", \
        "가산이 없던 행은 그대로다 — 모든 행을 '후보에 드는 데만 쓰임'으로 바꾸지 않는다"


def test_an_entrant_with_overlapping_lyric_keywords_states_the_overlap_without_a_ranking_claim(rig):
    """가사 키워드가 발췌와 겹친 보충 곡. 겹친다는 사실만 말하고 '검색 점수에 반영'이라고 하지 않는다."""
    from src.retrieval.user_explanation import BASIS_ENTRY_ONLY, FEEDBACK_SOURCE_SENTENCE, build_user_explanation

    seed_songs(rig.client, {"f4": {"title": "가사가 있는 곡", "artist": "가수 K", "lyrics_highlight": "휘파람 불며 걷던 길"}})
    rig.case("f4", "휘파람 노래")
    analysis = _analysis(original_query="휘파람 불던 노래", lyric_keywords=["휘파람"])
    on = _search(_router(_hits(), rig.retriever, reranker=_StubReranker()), analysis)
    song = on["record"].songs["f4"]
    assert song.feedback_entrant and "lyric_keyword_tokens" in [a.rule for a in song.adjustments]
    explanation = build_user_explanation(query=analysis.original_query, analysis=analysis, search_analysis=analysis, answers=[],
                                         track=next(t for t in on["tracks"] if t.id == "f4"), explain=song)
    assert explanation.summary.startswith(FEEDBACK_SOURCE_SENTENCE)
    assert "가사 키워드 일부가 발췌·요약과 겹쳐요" in explanation.summary and "검색 점수에 반영" not in explanation.summary
    keyword_row = next(c for c in explanation.clue_checks if c.slot == "lyrics")
    assert keyword_row.basis_label == BASIS_ENTRY_ONLY


def test_an_entrant_that_also_had_a_search_path_names_its_source_not_the_path(rig):
    """융합 목록에는 있었지만(텍스트 35위) 후보 밖이던 곡이 제보로 들어온 경우. '텍스트 검색이 비슷하게 평가'라고만 하면 안 된다."""
    from src.retrieval.user_explanation import FEEDBACK_SOURCE_SENTENCE, build_user_explanation

    rig.case("s35", "휘파람 노래")
    seed_songs(rig.client, {"s35": {"title": "곡35", "artist": "A"}})
    analysis = _analysis(original_query="휘파람 불던 노래")
    on = _search(_router(_hits(), rig.retriever, reranker=_StubReranker()), analysis, path_k=40)
    record = on["feedback"][0]
    assert record.entered_ids == ["s35"] and [(c.song_id, c.new_to_pool, c.outcome) for c in record.cases] == [("s35", False, "added")]
    song = on["record"].songs["s35"]
    assert {p.path for p in song.live_paths()} == {"text_hybrid", "feedback_case"} and song.feedback_entrant
    explanation = build_user_explanation(query=analysis.original_query, analysis=analysis, search_analysis=analysis, answers=[],
                                         track=next(t for t in on["tracks"] if t.id == "s35"), explain=song)
    assert explanation.summary == FEEDBACK_SOURCE_SENTENCE


def test_mixed_cases_report_not_entered_rather_than_already_in_candidates(rig):
    """이미 후보인 곡의 사례 + 들어오지 못한 곡의 사례가 섞이면 '이미 후보에 있었다'가 아니다."""
    rig.case("s2", "휘파람 부는 노래")                  # 이미 후보
    rig.case("f2", "휘파람 노래")                       # 남성 — 후보가 모두 여성이라 절단을 못 넘는다
    analysis = _analysis(original_query="휘파람 불던 노래", vocal_gender="여성")
    on = _search(_router(_hits(vocal_gender="여성"), rig.retriever), analysis)
    record = on["feedback"][0]
    assert sorted((c.song_id, c.outcome) for c in record.cases) == [("f2", "added"), ("s2", "already_candidate")]
    assert record.status == "not_entered" and record.entered_ids == []

    # 가리킨 곡이 **모두** 이미 후보일 때만 already_in_candidates다.
    rig.database["search_feedback"].delete_many({"selected_song_id": "f2"})
    only_existing = _search(_router(_hits(vocal_gender="여성"), rig.retriever), analysis)
    assert only_existing["feedback"][0].status == "already_in_candidates"


def test_search_run_snapshot_keeps_song_ids_but_no_case_ids(rig):
    from src.feedback import capture

    receipt = rig.case("f1", "휘파람 노래")
    record = _search(_router(_hits(), rig.retriever))["feedback"][0]
    stored = capture._feedback_supplement(record)
    assert stored["status"] == "applied" and stored["supplemented_ids"] == ["f1"]
    assert stored["entered_ids"] == ["f1"] and stored["displaced_ids"] == ["s30"]
    assert receipt.search_id not in str(stored)
    assert capture._feedback_supplement(None) is None


def test_search_route_stores_the_supplement_summary_with_the_run(monkeypatch):
    """실제 /search 라우트. 보충을 조회한 검색의 기록에 곡 ID 요약이 남고, 꺼져 있던 검색에는 없다."""
    from importlib import import_module
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.backend.api.dependencies import get_lyrics_exact_search_service, get_query_analyzer, get_search_router
    from src.backend.schemas.search import MatchingTrack
    from src.feedback import capture
    from src.feedback.store import FeedbackStore
    from src.retrieval.explain import FeedbackCaseExplain, FeedbackExplain
    from src.retrieval.query_analyzer import rule_fallback
    from tests.feedback_fakes import FakeDatabase

    database = FakeDatabase()
    store = FeedbackStore(database)
    monkeypatch.setenv("FEEDBACK_COLLECTION_ENABLED", "1")
    monkeypatch.setattr(capture, "_unavailable_until", 0.0)
    monkeypatch.setattr(capture, "get_feedback_store", lambda: store)
    state = {"supplement": True}

    class Router:
        async def search(self, analysis, **kw):
            tracks = [MatchingTrack(id=sid, title=sid, score=1) for sid in ["a", "f1"]]
            kw["candidate_ids_out"].extend(t.id for t in tracks)
            kw["candidate_tracks_out"].extend(tracks)
            if state["supplement"]:
                kw["feedback_out"].append(FeedbackExplain(
                    status="applied", entered_ids=["f1"], displaced_ids=["z"], settings={"fusion_weight": 0.8},
                    cases=[FeedbackCaseExplain(song_id="f1", feedback_id="c" * 32, source_revision=1, review_version=1,
                                               similarity=0.93, rank=1, delta=0.0111, new_to_pool=True,
                                               outcome="added", in_candidates=True, entered=True)]))
            return tracks

    search = import_module("src.backend.api.routes.search")
    app = FastAPI()
    app.include_router(search.router)
    app.dependency_overrides[get_query_analyzer] = lambda: SimpleNamespace(analyze=rule_fallback)
    app.dependency_overrides[get_search_router] = Router
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: None
    with TestClient(app) as client:
        body = {"query": "휘파람 노래", "top_k": 2, "defer_clarify": True}
        first = client.post("/search", json=body).json()
        stored = database["search_runs"].documents[first["feedback"]["search_id"]]
        assert stored["feedback_supplement"]["supplemented_ids"] == ["f1"]
        assert stored["feedback_supplement"]["entered_ids"] == ["f1"]
        assert stored["versions"]["feedback_index"] == EMBEDDING_VERSION
        assert "c" * 32 not in str(stored), "다른 사용자의 사례 ID는 검색 기록에 남기지 않는다"
        state["supplement"] = False
        second = client.post("/search", json=body).json()
        stored = database["search_runs"].documents[second["feedback"]["search_id"]]
        assert stored["feedback_supplement"] is None and stored["versions"]["feedback_index"] is None


def test_health_endpoint_includes_feedback_state():
    from src.backend import main

    body = asyncio.run(main.health_check())
    assert body["status"] == "ok" and set(body["feedback"]) == {"collection", "index_worker", "retrieval"}
    assert body["feedback"]["retrieval"]["enabled"] in {True, False}
