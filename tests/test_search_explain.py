"""검색 실행 기록(explain)의 불변 조건과 기록 내용을 검증한다.

완료 기준: **기록을 켜도 같은 입력·후보에서 순위와 점수가 변하지 않는다.**
기록기가 점수에 영향을 주면 설명 기능이 곧 랭킹 변경이 되므로, 그것부터 막는다.
"""

from __future__ import annotations

import math

import pytest

from src.backend.schemas.explain import SearchExplainOut
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval import search_router as sr
from src.retrieval.clarify import apply_answer_bonus
from src.retrieval.explain import (
    NULL_RECORDER,
    ExplainRecorder,
    SongExplain,
    render_ko,
)


def _hits(*pairs):
    return [(song_id, score) for song_id, score in pairs]


def _analysis(**kwargs) -> QueryAnalysis:
    """QueryAnalysis 필수 필드 기본값. tests/test_clarify_question.py와 같은 관례."""
    base = dict(
        original_query="비 오는 날 발라드",
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
    )
    base.update(kwargs)
    return QueryAnalysis(**base)


# ---------------------------------------------------------------------------
# 불변 조건
# ---------------------------------------------------------------------------

def test_rrf_fuse_unchanged_when_recording():
    """기록기를 붙여도 RRF 결과가 비트 단위로 같아야 한다."""
    lists = [
        _hits(("a", 0.9), ("b", 0.8), ("c", 0.7)),
        _hits(("b", 0.95), ("d", 0.5)),
        _hits(("c", 0.6), ("a", 0.4)),
    ]
    weights = [0.6, 0.2, 0.2]

    off = sr._rrf_fuse(ranked_lists=lists, weights=weights, top_k=None)
    rec = ExplainRecorder("q")
    on = sr._rrf_fuse(
        ranked_lists=lists,
        weights=weights,
        top_k=None,
        labels=["text_hybrid", "image", "audio"],
        recorder=rec,
    )

    assert [sid for sid, _ in off] == [sid for sid, _ in on]
    for (sid_a, score_a), (sid_b, score_b) in zip(off, on):
        assert sid_a == sid_b
        assert score_a == score_b, "점수가 비트 단위로 같아야 한다"


def test_fuse_add_records_actual_delta():
    """기록된 delta는 재계산이 아니라 점수에 실제로 반영된 차이여야 한다."""
    scores = {"a": 0.5}
    rec = ExplainRecorder("q")
    sr._fuse_add(scores, "a", 0.25, rec, "title", 3, "확신도 0.90")

    assert scores["a"] == 0.75
    path = rec.record.get("a").paths[0]
    assert path.path == "title"
    assert path.rank == 3
    assert path.delta == pytest.approx(0.25)
    assert path.detail == "확신도 0.90"


def test_fuse_add_delta_matches_float_reality():
    """부동소수 반올림이 있어도 기록값 = 실제 증가분이어야 한다."""
    scores = {"a": 1e16}
    rec = ExplainRecorder("q")
    sr._fuse_add(scores, "a", 1.0, rec, "metadata", 1)

    # 1e16 + 1.0 은 반올림으로 증가분이 1.0이 아니다. 기록은 그 사실을 따라야 한다.
    recorded = rec.record.get("a").paths[0].delta
    assert recorded == scores["a"] - 1e16


def test_explicit_boosts_unchanged_when_recording():
    """명시 보정도 기록 여부와 무관하게 같은 점수를 내야 한다."""
    analysis = _analysis(
        original_query="남자가 부른 비 오는 날 노래",
        vocal_gender="남성",
        genre="발라드",
    )
    meta = {
        "s1": MatchingTrack(id="s1", score=0.0, title="비", artist="가수A",
                            vocal_gender="남성", genre="발라드"),
        "s2": MatchingTrack(id="s2", score=0.0, title="눈", artist="가수B",
                            vocal_gender="여성", genre="댄스"),
        "s3": MatchingTrack(id="s3", score=0.0, title="바람", artist="가수C",
                            vocal_gender="혼성", genre="발라드"),
    }
    fused = _hits(("s1", 0.010), ("s2", 0.009), ("s3", 0.008))

    off = sr.SearchRouter._apply_explicit_boosts(analysis, fused, meta)
    rec = ExplainRecorder("q")
    on = sr.SearchRouter._apply_explicit_boosts(analysis, fused, meta, recorder=rec)

    assert off == on, "기록을 켜도 (id, score) 목록이 동일해야 한다"


def test_penalty_is_recorded_as_negative(monkeypatch):
    """감점을 켜면 기록해야 한다 — 왜 밀렸는지가 설명의 절반이다."""
    monkeypatch.setenv("GENDER_MISMATCH_SCALE", "1")
    analysis = _analysis(original_query="여자가 부른 노래", vocal_gender="여성")
    meta = {
        "s1": MatchingTrack(id="s1", score=0.0, title="곡", artist="A",
                            vocal_gender="남성"),
    }
    rec = ExplainRecorder("q")
    sr.SearchRouter._apply_explicit_boosts(
        analysis, _hits(("s1", 0.01)), meta, recorder=rec
    )

    adjustments = rec.record.get("s1").adjustments
    assert adjustments, "성별 불일치 감점이 기록되어야 한다"
    mismatch = [a for a in adjustments if a.rule == "vocal_gender_mismatch"]
    assert mismatch and mismatch[0].delta < 0


def test_no_penalty_and_no_record_by_default(monkeypatch):
    """기본값(감점 0)에서는 점수도 기록도 없다 — 0점 감점을 설명하면 이유를 지어내는 것이다."""
    monkeypatch.delenv("GENDER_MISMATCH_SCALE", raising=False)
    analysis = _analysis(original_query="여자가 부른 노래", vocal_gender="여성")
    meta = {
        "s1": MatchingTrack(id="s1", score=0.0, title="곡", artist="A",
                            vocal_gender="남성"),
        "s2": MatchingTrack(id="s2", score=0.0, title="곡", artist="B",
                            vocal_gender="여성"),
    }
    rec = ExplainRecorder("q")
    scored = dict(sr.SearchRouter._apply_explicit_boosts(
        analysis, _hits(("s1", 0.01), ("s2", 0.01)), meta, recorder=rec
    ))
    assert scored["s1"] == 0.01, "불일치 곡의 점수는 그대로여야 한다"
    assert scored["s2"] > 0.01, "일치 가산은 그대로 준다"
    rules = [a.rule for a in rec.record.get("s1").adjustments]
    assert "vocal_gender_mismatch" not in rules


def test_answer_bonus_unchanged_when_recording():
    analysis_tracks = {
        "s1": MatchingTrack(id="s1", score=0.0, title="곡1", artist="A",
                            vocal_gender="남성"),
        "s2": MatchingTrack(id="s2", score=0.0, title="곡2", artist="B",
                            vocal_gender="여성"),
    }
    answers = [ClarifyAnswer(slot="vocal_gender", value="남성")]
    scored = _hits(("s1", 0.01), ("s2", 0.011))

    off = apply_answer_bonus(scored, analysis_tracks, answers, 1.0 / 61)
    rec = ExplainRecorder("q")
    on = apply_answer_bonus(scored, analysis_tracks, answers, 1.0 / 61, recorder=rec)

    assert off == on
    assert any(a.rule == "answer_match" for a in rec.record.get("s1").adjustments)


def test_null_recorder_records_nothing():
    """기본값은 아무것도 남기지 않아야 한다 — 켜지 않으면 비용도 없다."""
    NULL_RECORDER.path("a", "text_hybrid", 1, 0.5)
    NULL_RECORDER.adjust("a", "title_exact", 0.5)
    NULL_RECORDER.order("a", "lyrics_exact_priority")
    NULL_RECORDER.set_fused("a", 1.0)

    assert NULL_RECORDER.record.songs == {}
    assert NULL_RECORDER.enabled is False


# ---------------------------------------------------------------------------
# 문장화 — 기록에 없는 것은 말하지 않는다
# ---------------------------------------------------------------------------

def test_render_reports_no_evidence_when_empty():
    assert render_ko(SongExplain(song_id="s1")) == "기록된 근거가 없습니다."


def test_render_mentions_rank_change_from_rerank():
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 2, 0.0161, "가중치 0.60")
    rec.adjust("s1", "vocal_gender_match", 0.0164, "남성")
    rec.set_rank_before(["s9", "s1"])
    rec.set_rank_after(["s1", "s9"])
    rec.set_reorder_attempted(True)
    rec.note_rerank_run(RERANK_APPLIED, ["s1", "s9"])
    # 모델이 본 묶음이 후보 전체와 같을 때만 순위 변화를 모델에 귀속할 수 있다.
    rec.set_group_ranks(["s9", "s1"], ["s1", "s9"], ["s1", "s9"])
    rec.commit_reorder()

    text = render_ko(
        rec.record.get("s1"), reorder_stage=REORDER_MODEL_APPLIED
    )
    assert "텍스트(의미+키워드) 경로 2위" in text
    assert "말한 성별과 일치" in text
    assert "2위에서 1위로 올림" in text


def test_render_puts_order_rule_first():
    """점수로는 보이지 않는 순서 규칙이 가장 앞에 와야 한다."""
    rec = ExplainRecorder("q")
    rec.path("s1", "audio", 7, 0.003)
    rec.order("s1", "lyrics_exact_priority", "일반 후보 아래로 내려가지 않음")

    text = render_ko(rec.record.get("s1"))
    assert text.startswith("기억한 구절이 가사와 일치해 먼저 배치")


def test_path_and_adjustment_totals():
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.path("s1", "audio", 5, 0.003)
    rec.adjust("s1", "title_exact", 0.082)
    rec.adjust("s1", "vocal_gender_mismatch", -0.008)

    explain = rec.record.get("s1")
    assert math.isclose(explain.path_total(), 0.019)
    assert math.isclose(explain.adjustment_total(), 0.074)


# ---------------------------------------------------------------------------
# 실제 search() 흐름 회귀 — 설명이 실행과 어긋나지 않는지
# ---------------------------------------------------------------------------

import asyncio

import numpy as np

from src.retrieval.explain import (
    REORDER_MODEL_APPLIED,
    REORDER_MODEL_FAILED,
    REORDER_NOT_ATTEMPTED,
    REORDER_ORDER_RULES_ONLY,
    RERANK_APPLIED,
    RERANK_FAILED,
    RERANK_SKIPPED,
    RERANK_UNKNOWN,
    RerankRun,
)


def _track(sid, title="곡", artist="A", **kw):
    return MatchingTrack(id=sid, score=kw.pop("score", 0.0), title=title,
                         artist=artist, **kw)


class _FakeTextService:
    """search_text가 돌려줄 결과를 미리 정해 둔다."""

    def __init__(self, hits):
        self._hits = hits

    def search_text(self, *a, **kw):
        top_k = kw.get("top_k") or 30
        return [t.model_copy() for t in self._hits[:top_k]]

    def track_from_match(self, match):  # 이미지·오디오 경로용
        return _track(str(match.get("id")))


class _FakeIndex:
    def __init__(self, ids=()):
        self._ids = list(ids)

    def query(self, *a, **kw):
        return {"matches": [{"id": sid} for sid in self._ids]}


class _FakeClient:
    def __init__(self, audio_ids=()):
        self._audio_ids = list(audio_ids)

    def Index(self, name):  # noqa: N802 - 호출부가 쓰는 이름을 따른다
        from src.vector_db.settings import AUDIO_INDEX_NAME
        return _FakeIndex(self._audio_ids if name == AUDIO_INDEX_NAME else ())


class _FakeEmbedder:
    def embed_texts(self, texts, **kw):
        return np.zeros((len(texts), 4), dtype=np.float32)


class _FakeLyricsService:
    """가사 경로의 결과만 흉내 낸다.

    `snapshot_out`을 받기만 하고 **채우지 않는다.** 여기에는 원문 가사가 없어서
    인용할 판 자체가 없다 — 라우트는 판이 없으면 인용을 비운다. 인용까지 보는
    시험은 `tests/test_lyric_evidence.py`가 실물 서비스로 한다.
    """

    def __init__(self, hits):
        self._hits = hits

    def search(self, clues, top_k=None, snapshot_out=None):
        return [t.model_copy() for t in self._hits]


class _StubReranker:
    """호출 여부를 세는 리랭커. 순서를 뒤집어 '리랭킹이 있었다'를 눈에 보이게 한다.

    inner_fallback=True는 Gemini의 실제 동작을 흉내 낸다 — 재시도가 전부 실패해도
    **예외를 던지지 않고 입력 순서를 그대로 돌려준다.**
    """

    enabled = True
    uses_clarify_answers = False

    def __init__(self, fail=False, inner_fallback=False, fail_after=None,
                 max_judged=None, identity=False):
        self.identity = identity   # 성공했지만 입력 순서를 그대로 돌려준다
        self.calls = 0
        self.seen = []
        self.fail = fail
        self.inner_fallback = inner_fallback
        self.fail_after = fail_after      # 이 호출 번호부터 예외를 던진다(1-based)
        self.max_judged = max_judged      # Gemini처럼 앞쪽 N곡만 평가한다

    def rerank_run(self, query, tracks, top_k):
        self.calls += 1
        self.seen.append([t.id for t in tracks])
        if self.fail or (self.fail_after and self.calls >= self.fail_after):
            raise RuntimeError("리랭킹 실패(시험)")
        if self.inner_fallback:
            return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])
        judged = list(tracks) if self.max_judged is None else list(tracks[: self.max_judged])
        tail = [] if self.max_judged is None else list(tracks[self.max_judged :])
        out = list(judged) if self.identity else list(reversed(judged))
        out = [t.model_copy(update={"rerank_score": 0.9}) for t in out]
        return RerankRun(
            [*out, *tail][:top_k],
            RERANK_APPLIED,
            [t.id for t in judged],
        )

    def rerank(self, query, tracks, top_k):
        return self.rerank_run(query, tracks, top_k).tracks


class _LegacyReranker:
    """rerank_with_status가 없는 예전 백엔드. 상태를 알 수 없다."""

    enabled = True
    uses_clarify_answers = False

    def __init__(self):
        self.calls = 0
        self.seen = []

    def rerank(self, query, tracks, top_k):
        self.calls += 1
        self.seen.append([t.id for t in tracks])
        out = list(reversed(list(tracks)))[:top_k]
        return [t.model_copy(update={"rerank_score": 0.9}) for t in out]


def _router(text_hits, *, lyrics_hits=None, reranker=None, audio_ids=(),
            image_embedder=None):
    return sr.SearchRouter(
        search_service=_FakeTextService(text_hits),
        image_embedder=image_embedder or _FakeEmbedder(),
        audio_embedder=_FakeEmbedder(),
        vector_client=_FakeClient(audio_ids),
        lyrics_search_service=(
            _FakeLyricsService(lyrics_hits) if lyrics_hits is not None else None
        ),
        reranker=reranker,
        max_workers=1,
    )


def _run(router, analysis, **kw):
    try:
        return asyncio.run(router.search(analysis, **kw))
    finally:
        router.shutdown()


def test_no_rerank_claim_when_rerank_disabled():
    """결함1: use_rerank=False인데 '리랭킹이 N위 유지'라고 말하면 안 된다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    router = _router(hits, reranker=_StubReranker())
    results = _run(router, _analysis(), top_k=3, use_rerank=False, recorder=rec)

    assert results
    assert rec.record.reorder_stage == REORDER_NOT_ATTEMPTED
    assert rec.record.rerank_runs == []
    assert rec.record.rerank_applied is False

    text = render_ko(rec.record.get(results[0].id), reorder_stage=rec.record.reorder_stage)
    assert "리랭킹" not in text


def test_rerank_failure_is_explained_as_failure():
    """결함1: 예외가 났으면 '실패해 검색 순서 유지'로 말해야 한다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    stub = _StubReranker(fail=True)
    results = _run(_router(hits, reranker=stub), _analysis(), top_k=3, recorder=rec)

    assert stub.calls == 1
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED
    assert rec.record.rerank_applied is False
    assert "리랭킹 실패(시험)" in rec.record.rerank_error

    text = render_ko(rec.record.get(results[0].id), reorder_stage=rec.record.reorder_stage)
    assert "실패해 검색 순서 유지" in text


def test_rerank_applied_only_when_model_ran():
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    stub = _StubReranker()
    _run(_router(hits, reranker=stub), _analysis(), top_k=3, recorder=rec)

    assert stub.calls >= 1
    assert len(rec.record.rerank_runs) == stub.calls
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED
    assert rec.record.rerank_applied is True


def test_dropped_path_contribution_excluded_from_total():
    """결함2: 절단에서 탈락한 기여가 최종 합계에 들어가면 안 된다."""
    rec = ExplainRecorder("q")
    lists = [[(f"s{i}", 1.0 / i) for i in range(1, 11)]]
    fused = sr._rrf_fuse(
        ranked_lists=lists, weights=[1.0], top_k=3,
        labels=["text_hybrid"], recorder=rec,
    )
    survivors = {sid for sid, _ in fused}

    for sid, score in fused:
        assert rec.record.get(sid).path_total() == pytest.approx(score)
        assert not rec.record.get(sid).dropped_paths()

    dropped = [sid for sid in (f"s{i}" for i in range(1, 11)) if sid not in survivors]
    assert dropped
    for sid in dropped:
        explain = rec.record.get(sid)
        assert explain.dropped_paths(), "탈락한 기여는 표시되어야 한다"
        assert explain.path_total() == 0.0, "최종 합계에서 제외되어야 한다"


def test_rejoined_candidate_total_matches_fused_score():
    """결함2 끝단: 절단에서 탈락 후 가사 경로로 재합류한 곡을 직접 검증한다.

    force_weights를 쓰면 안 된다 — 그 경로는 가사 검색 자체를 끄므로
    (`search_router`의 `force_weights is None` 게이트) 재합류가 일어나지 않고,
    fused_score=None인 곡을 건너뛰면 검사가 통과해 버린다.

    절단은 **여러 경로의 합집합이 pool_k를 넘을 때** 생긴다. 텍스트와 오디오가
    서로 겹치지 않는 후보를 내도록 두고, 오디오에서만 들어와 잘린 곡을 가사
    경로가 다시 부르게 한다.
    """
    text_hits = [_track(f"t{i}", title=f"텍스트{i}") for i in range(1, 61)]
    audio_ids = [f"a{i}" for i in range(1, 31)]
    target = "a30"
    rejoined = _track(target, title="오디오30", lyric_match_type="exact",
                      lyric_match_score=0.95)

    rec = ExplainRecorder("q")
    analysis = _analysis(
        audio_english_query="rainy dawn ballad",
        lyric_clues=[{"kind": "phonetic", "text": "샤라라"}],
    )
    results = _run(
        _router(text_hits, lyrics_hits=[rejoined], audio_ids=audio_ids),
        analysis, top_k=5, use_rerank=False, recorder=rec,
    )

    explain = rec.record.get(target)

    # (1) 절단 전 기여가 탈락으로 표시됐다
    dropped = explain.dropped_paths()
    assert dropped, "절단 전 경로 기여가 탈락으로 표시되어야 한다"
    assert any(p.path == "audio" for p in dropped)

    # (2) 가사 경로로 다시 들어왔다
    assert any(p.path == "lyrics_surface" for p in explain.live_paths()), (
        "가사 경로 재진입이 기록되어야 한다"
    )

    # (3) 최종 후보에 포함됐다
    assert explain.fused_score is not None, "재합류한 곡이 후보 풀에 있어야 한다"
    assert target in {t.id for t in results}, "재합류한 곡이 결과에 있어야 한다"

    # (4) 설명의 기여 합계가 실제 융합 점수와 같다
    assert explain.path_total() == pytest.approx(explain.fused_score, abs=1e-9)
    assert "재합류" in render_ko(explain, reorder_stage=rec.record.reorder_stage)

    # 후보 전체에서도 어긋남이 없어야 한다.
    checked = 0
    for sid, ex in rec.record.songs.items():
        if ex.fused_score is None:
            continue
        checked += 1
        assert ex.path_total() == pytest.approx(ex.fused_score, abs=1e-9), sid
    assert checked > 1, "후보가 여러 곡 검사되어야 한다"


def test_summary_only_lyric_match_is_not_called_lyrics():
    """결함3: 요약에만 있는 문구를 '가사에 있음'이라고 하면 안 된다."""
    track = _track(
        "s1",
        lyrics_highlight="전혀 다른 구절",
        lyrics_summary="이별 후의 새벽을 그린 곡",
    )
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "이별 후의 새벽"}])
    rec = ExplainRecorder("q")
    sr.SearchRouter._apply_explicit_boosts(
        analysis, _hits(("s1", 0.01)), {"s1": track}, recorder=rec
    )

    rules = [a.rule for a in rec.record.get("s1").adjustments]
    assert "lyric_clue_in_summary" in rules
    assert "lyric_clue_in_highlight" not in rules

    text = render_ko(rec.record.get("s1"))
    assert "가사 요약과 표기 정규화 후 일치" in text
    assert "이 곡 가사에 있음" not in text


def test_highlight_lyric_match_is_reported_as_excerpt():
    track = _track("s1", lyrics_highlight="비가 내리던 새벽", lyrics_summary="요약")
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "비가 내리던 새벽"}])
    rec = ExplainRecorder("q")
    sr.SearchRouter._apply_explicit_boosts(
        analysis, _hits(("s1", 0.01)), {"s1": track}, recorder=rec
    )

    rules = [a.rule for a in rec.record.get("s1").adjustments]
    assert "lyric_clue_in_highlight" in rules
    assert "가사 발췌와 표기 정규화 후 일치" in render_ko(rec.record.get("s1"))


def test_phonetic_clue_is_distinguished_from_verbatim():
    """음차 추정과 원문 일치를 구분해야 한다."""
    track = _track("s1", lyrics_highlight="shawty like a melody")
    analysis = _analysis(
        lyric_clues=[{"kind": "phonetic", "text": "shawty like a melody"}]
    )
    rec = ExplainRecorder("q")
    sr.SearchRouter._apply_explicit_boosts(
        analysis, _hits(("s1", 0.01)), {"s1": track}, recorder=rec
    )

    rules = [a.rule for a in rec.record.get("s1").adjustments]
    assert "lyric_phonetic_in_highlight" in rules
    assert "음차 추정" in render_ko(rec.record.get("s1"))


def test_every_recorded_rule_has_a_label():
    """미등록 규칙명이 사용자에게 그대로 나가지 않도록 커버리지를 테스트로 고정한다."""
    from src.retrieval import explain as ex

    used = {
        "lyric_clue_in_highlight", "lyric_clue_in_summary", "lyric_clue_in_combined",
        "lyric_phonetic_in_highlight", "lyric_phonetic_in_summary",
        "lyric_phonetic_in_combined", "lyric_keyword_tokens",
        "title_exact", "title_constraints", "title_hanja_presence", "title_meaning",
        "artist_match", "vocal_gender_match", "vocal_gender_partial",
        "vocal_gender_mismatch", "release_era", "artist_type", "performance_clues",
        "genre_match", "answer_match", "lyrics_exact_priority",
    }
    missing = used - set(ex.RULE_LABELS)
    assert not missing, f"라벨이 없는 규칙: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 백엔드 내부 폴백 · 곡별 적용 여부 · 취소된 규칙
# ---------------------------------------------------------------------------

def test_backend_inner_fallback_is_not_reported_as_success():
    """결함1: Gemini는 재시도가 전부 실패해도 예외 없이 입력 순서를 돌려준다.

    호출 횟수만 보면 성공과 구분되지 않는다. 백엔드가 보고한 상태를 써야 한다.
    """
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    stub = _StubReranker(inner_fallback=True)
    results = _run(_router(hits, reranker=stub), _analysis(), top_k=3, recorder=rec)

    assert stub.calls == 1, "호출은 있었다"
    assert rec.record.rerank_runs == [RERANK_FAILED]
    assert rec.record.rerank_applied is False
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED

    text = render_ko(rec.record.get(results[0].id), reorder_stage=rec.record.reorder_stage)
    assert "실패해 검색 순서 유지" in text
    assert "유지" in text and "리랭킹이 1위 유지" not in text


def test_unknown_status_backend_is_not_claimed_as_applied():
    """상태를 보고하지 않는 백엔드는 '리랭킹했다'고 단정하지 않는다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    stub = _LegacyReranker()
    results = _run(_router(hits, reranker=stub), _analysis(), top_k=3, recorder=rec)

    assert stub.calls == 1
    assert rec.record.rerank_applied is False
    assert rec.record.reorder_stage == "model_unknown"

    text = render_ko(rec.record.get(results[0].id), reorder_stage=rec.record.reorder_stage)
    assert "확인할 수 없음" in text


def test_song_not_sent_to_model_is_not_described_as_reranked():
    """결함2: 단독 보호 그룹의 곡은 모델을 보지 않는다 — 리랭킹했다고 하면 안 된다."""
    protected = _track("p", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.95)
    others = [_track("a", title="일반A"), _track("b", title="일반B")]
    rec = ExplainRecorder("q")
    stub = _StubReranker()
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])

    results = _run(
        _router(others, lyrics_hits=[protected], reranker=stub),
        analysis, top_k=3, recorder=rec,
    )

    assert results[0].id == "p", "보호곡이 먼저 배치되어야 한다"
    # 모델에 들어간 것은 일반 후보뿐이다.
    sent = {sid for batch in stub.seen for sid in batch}
    assert "p" not in sent
    assert rec.record.get("p").reorder_applied is False
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED, "요청 전체로는 모델이 돌았다"

    text = render_ko(rec.record.get("p"), reorder_stage=rec.record.reorder_stage)
    assert "리랭킹이" not in text
    assert "기억한 구절이 가사와 일치해 먼저 배치" in text
    assert "리랭커에 넘기지 않음" in text


def test_cancelled_order_rule_is_not_recorded_on_fallback():
    """결함3: 리랭킹이 실패해 폴백하면 우선 배치는 **성사되지 않았다.**

    보호 배치는 리랭킹 경로 안에서 완성된다. 예외가 나면 그 경로를 빠져나와
    검색 순서로 돌아가므로, 배치 규칙이 하나라도 기록돼 있으면 거짓이다.
    최종 순위가 우연히 같더라도 "규칙이 적용돼서 그 자리"는 아니다.
    """
    protected = _track("p", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.95)
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    stub = _StubReranker(fail=True)
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])

    _run(
        _router(others, lyrics_hits=[protected], reranker=stub),
        analysis, top_k=4, recorder=rec,
    )

    assert rec.record.reorder_stage == REORDER_MODEL_FAILED
    assert rec.record.order_rules_applied is False

    recorded = {
        sid: [r.rule for r in ex.order_rules]
        for sid, ex in rec.record.songs.items()
        if ex.order_rules
    }
    assert not recorded, f"폴백했는데 배치 규칙이 남았다: {recorded}"

    text = render_ko(rec.record.get("p"), reorder_stage=rec.record.reorder_stage)
    assert "내려가지 않음" not in text
    assert "실패해 검색 순서 유지" in text


def test_order_rule_recorded_only_for_songs_in_final_result():
    """보호곡이 top_k 밖으로 잘리면 그 곡에는 배치 규칙을 남기지 않는다."""
    protected = _track("p", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.95)
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])

    results = _run(
        _router(others, lyrics_hits=[protected], reranker=_StubReranker()),
        analysis, top_k=4, recorder=rec,
    )

    for sid, explain in rec.record.songs.items():
        if explain.order_rules:
            assert sid in {t.id for t in results}, (
                f"{sid}: 최종 결과에 없는데 배치 규칙이 기록됐다"
            )


# ---------------------------------------------------------------------------
# 복합 경로 — "실행이 성공했다"와 "최종 결과에 반영됐다"는 다르다
# ---------------------------------------------------------------------------

def test_partial_group_success_then_total_fallback_revokes_marks():
    """결함1: 앞 그룹은 성공했는데 뒤에서 예외가 나면 **전체가 폴백**한다.

    성공 이력이 남아 있다고 적용됐다고 말하면 안 된다 — 최종 결과는 검색 순서다.
    """
    # 보호 곡 2개를 **같은 tier**에 두어야 그룹 리랭킹 호출이 생긴다
    # (tier 안에 1곡뿐이면 call_reranker를 건너뛴다). 그 호출이 성공한 뒤
    # 일반 후보 호출에서 예외가 나는 것이 이 시나리오다.
    protected = [
        _track("p1", title="보호1", vocal_gender="남성",
               lyric_match_type="exact", lyric_match_score=0.95),
        _track("p2", title="보호2", vocal_gender="남성",
               lyric_match_type="exact", lyric_match_score=0.95),
    ]
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    # 첫 호출은 성공, 두 번째부터 예외 → 앞 그룹만 성공한 상태에서 전체 폴백
    stub = _StubReranker(fail_after=2)
    analysis = _analysis(
        vocal_gender="남성",
        lyric_clues=[{"kind": "verbatim", "text": "한 소절"}],
    )

    _run(
        _router(others, lyrics_hits=protected, reranker=stub),
        analysis, top_k=5, recorder=rec,
    )

    assert stub.calls >= 2, "성공 1회 + 실패 1회가 나야 하는 시나리오다"
    assert RERANK_APPLIED in rec.record.rerank_runs, "앞 그룹은 성공했다"
    assert rec.record.reorder_committed is False, "최종 결과에 반영되지 않았다"
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED

    applied = [sid for sid, ex in rec.record.songs.items() if ex.reorder_applied]
    assert not applied, f"폴백했는데 적용 표시가 남았다: {applied}"

    for sid in ("p1", "p2"):
        text = render_ko(rec.record.get(sid), reorder_stage=rec.record.reorder_stage)
        assert "리랭킹이" not in text or "실패" in text
        assert "위 유지" not in text.replace("검색 순서 유지", "")


def _displacement_setup():
    """보호곡이 2위, 일반곡이 1위인 상태를 만든다.

    보호곡의 가사 보너스(boost_unit×8×0.80 ≈ 0.105)보다 일반곡의 명시 단서 합
    (제목 완전일치 5 + 가수 1 + 성별 1 + 장르 1 = 8배 ≈ 0.131)이 커야 한다.
    그래야 보호 배치가 순위를 실제로 뒤집고, 일반곡이 밀려난다.
    """
    protected = _track("p", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.80)
    strong = _track("o1", title="센곡", artist="가수A",
                    vocal_gender="남성", genre="발라드")
    weak = _track("o2", title="약곡", artist="가수B")
    analysis = _analysis(
        song_title="센곡", artist_name="가수A", vocal_gender="남성", genre="발라드",
        lyric_clues=[{"kind": "verbatim", "text": "한 소절"}],
    )
    return protected, [strong, weak], analysis


def test_inner_fallback_with_order_rule_reports_both():
    """결함2: 백엔드가 예외 없이 실패해도 보호 배치는 적용된다.

    보호곡은 2→1위로 올라가고 일반곡은 1→2위로 내려간다. 둘 다 "검색 순서 유지"가
    아니며, 일반곡의 하락 원인은 모델이 아니라 보호 규칙이다.
    """
    protected, others, analysis = _displacement_setup()
    rec = ExplainRecorder("q")
    results = _run(
        _router(others, lyrics_hits=[protected],
                reranker=_StubReranker(inner_fallback=True)),
        analysis, top_k=3, recorder=rec,
    )

    ids = [t.id for t in results]
    assert ids[0] == "p" and ids[1] == "o1", f"보호 배치가 순위를 뒤집어야 한다: {ids}"
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED

    # 보호곡 — 규칙이 이유고, 모델은 실패했다
    p_ex = rec.record.get("p")
    assert p_ex.order_rules
    assert p_ex.reorder_applied is False
    p_text = render_ko(p_ex, reorder_stage=rec.record.reorder_stage)
    assert "기억한 구절이 가사와 일치해 먼저 배치" in p_text
    assert "리랭킹은 실패" in p_text
    assert "검색 순서 유지" not in p_text

    # 밀린 일반곡 — 하락을 모델에 귀속하면 안 된다
    o_ex = rec.record.get("o1")
    assert o_ex.behind_protected == 1, "보호곡 1곡에 밀렸다는 사실이 기록되어야 한다"
    assert o_ex.reorder_applied is False
    o_text = render_ko(o_ex, reorder_stage=rec.record.reorder_stage)
    assert "최종 순위 1→2위" in o_text
    assert "가사 일치 보호 곡 뒤에 배치 1곡" in o_text
    assert "리랭킹이 1위에서 2위로 내림" not in o_text, (
        "하락 원인은 모델이 아니라 보호 규칙이다"
    )
    assert "검색 순서 유지" not in o_text


def test_displaced_song_not_attributed_to_model_on_success():
    """결함: 모델이 **성공**하고 묶음 순서를 그대로 돌려준 경우.

    일반곡의 1→2위 하락은 모델의 정렬이 아니라 보호 배치 때문이다.
    모델에 귀속하면 "리랭킹이 1위에서 2위로 내림"이라는 거짓이 된다.
    """
    protected, others, analysis = _displacement_setup()
    rec = ExplainRecorder("q")
    results = _run(
        _router(others, lyrics_hits=[protected],
                reranker=_StubReranker(identity=True)),
        analysis, top_k=3, recorder=rec,
    )

    ids = [t.id for t in results]
    assert ids[0] == "p" and ids[1] == "o1", f"보호 배치가 순위를 뒤집어야 한다: {ids}"
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED

    o_ex = rec.record.get("o1")
    assert o_ex.reorder_applied is True, "이 곡은 모델에 들어갔다"
    assert o_ex.behind_protected == 1
    # 모델이 본 묶음(일반 후보만) 안에서는 1위 그대로다
    assert o_ex.group_rank_before == 1 and o_ex.group_rank_after == 1
    # 그런데 요청 전체로는 1→2위다 — 두 숫자가 다르므로 분리해 말해야 한다
    assert o_ex.rank_before_rerank == 1 and o_ex.rank_after_rerank == 2

    text = render_ko(o_ex, reorder_stage=rec.record.reorder_stage)
    assert "최종 순위 1→2위" in text
    assert "리랭킹 적용(묶음 안 순서 유지)" in text
    assert "가사 일치 보호 곡 뒤에 배치 1곡" in text
    assert "리랭킹이 1위에서 2위로 내림" not in text


def test_unjudged_tail_is_not_marked_as_reranked():
    """결함3: Gemini는 max_candidates까지만 평가하고 뒷부분은 그대로 붙인다.

    모델이 본 적 없는 곡에 "리랭킹이 N위 유지"를 붙이면 거짓이다.
    """
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 15)]
    rec = ExplainRecorder("q")
    stub = _StubReranker(max_judged=10)
    results = _run(
        _router(hits, reranker=stub), _analysis(), top_k=12, recorder=rec,
    )

    assert stub.calls == 1
    judged = set(stub.seen[0][:10])
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED

    marked = {sid for sid, ex in rec.record.songs.items() if ex.reorder_applied}
    assert marked == judged, f"평가한 곡만 표시해야 한다 (차이: {marked ^ judged})"

    # 11위 이후 곡은 모델이 보지 않았다.
    tail_ids = [t.id for t in results if t.id not in judged]
    assert tail_ids, "평가되지 않은 뒷부분이 결과에 있어야 하는 시나리오다"
    for sid in tail_ids:
        text = render_ko(rec.record.get(sid), reorder_stage=rec.record.reorder_stage)
        assert "리랭킹" not in text, f"{sid}: 평가되지 않았는데 리랭킹을 말한다 — {text}"


def test_unchanged_rank_is_not_called_displacement():
    """결함: 보호 곡이 처음부터 1위면 2위 곡은 2위 그대로다.

    `behind_protected`는 하락 폭이 아니라 앞에 배치된 보호 곡 수다.
    순위가 안 바뀌었는데 "밀려남"이라고 하면 거짓이다.
    """
    # 보호곡이 가사 보너스로 압도적 1위 → 일반곡들은 원래 자리 그대로다.
    protected = _track("p", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.95)
    others = [_track("o1", title="일반1"), _track("o2", title="일반2")]
    rec = ExplainRecorder("q")
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])

    results = _run(
        _router(others, lyrics_hits=[protected],
                reranker=_StubReranker(identity=True)),
        analysis, top_k=3, recorder=rec,
    )
    assert [t.id for t in results][0] == "p"

    o_ex = rec.record.get("o1")
    assert o_ex.behind_protected == 1, "보호 곡 뒤에 배치된 사실은 남는다"
    assert o_ex.rank_before_rerank == o_ex.rank_after_rerank, (
        "이 시나리오에서는 순위가 바뀌지 않아야 한다"
    )

    text = render_ko(o_ex, reorder_stage=rec.record.reorder_stage)
    assert "가사 일치 보호 곡 뒤에 배치 1곡" in text
    assert "밀려남" not in text, "순위가 그대로인데 밀렸다고 하면 안 된다"
    assert "최종 순위" not in text, "바뀌지 않은 순위를 변화로 적으면 안 된다"


def test_judged_but_dropped_from_results_is_not_called_unchanged():
    """결함: 평가는 됐지만 반환 목록에 없는 곡을 "순서 유지"라고 하면 안 된다.

    정답이 결과 밖으로 밀린 이유를 분석할 때 정확히 이 문장이 거짓이 된다.
    """
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    # 후보 3곡을 뒤집어 Top-1만 반환 → s1·s2는 평가됐지만 결과 밖
    _run(_router(hits, reranker=_StubReranker()), _analysis(), top_k=1, recorder=rec)

    dropped = rec.record.get("s1")
    assert dropped.reorder_applied is True, "모델이 평가한 곡이다"
    assert dropped.group_rank_after is None, "반환 목록에 없어 위치를 모른다"

    text = render_ko(dropped, reorder_stage=rec.record.reorder_stage)
    assert "평가됐으나 반환 결과 밖" in text
    assert "순서 유지" not in text


# ---------------------------------------------------------------------------
# 혼합 점수 — "리랭커가 순서를 정했다"고 말할 수 있는 범위
# ---------------------------------------------------------------------------

from src.retrieval.explain import ScoreMix
from src.retrieval.reranker import MusicReranker, RerankerConfig


class _FixedScoreReranker(MusicReranker):
    """CE 점수를 미리 정해 둔 실제 리랭커. 합성 산술은 원본 그대로 돈다."""

    def __init__(self, scores, **config):
        super().__init__(RerankerConfig(**config))
        self._scores = scores

    def load(self):  # 모델을 부르지 않는다
        return self

    def score(self, query, tracks):
        return [self._scores[str(track.id)] for track in tracks]


def _ce_tracks():
    return [
        _track("a", title="가", score=0.030),
        _track("b", title="나", score=0.020),
        _track("c", title="다", score=0.010),
    ]


def test_cross_encoder_mix_matches_the_final_score():
    """기록된 합성식이 **실제 점수를 만든 그 식**이어야 한다."""
    reranker = _FixedScoreReranker({"a": 0.62, "b": 0.99, "c": 0.41})
    run = reranker.rerank_run("q", _ce_tracks(), top_k=3)

    by_id = {str(track.id): track for track in run.tracks}
    assert set(run.mixes) == {"a", "b", "c"}
    for song_id, mix in run.mixes.items():
        assert mix.backend == "cross_encoder"
        assert mix.final == pytest.approx(by_id[song_id].score), "기록 = 실제 점수"
        recomputed = (
            mix.weight * mix.rerank_component
            + (1.0 - mix.weight) * mix.retrieval_component
        )
        assert recomputed == pytest.approx(mix.final)


def test_highest_rerank_score_is_not_always_first():
    """이 기록이 필요한 이유 그 자체.

    리랭커 점수가 가장 높은 곡이 1위가 아닐 수 있다. 실제 질의에서 CE 0.998인 곡이
    2위였다. 합성 비율을 남기지 않으면 그 순서를 설명할 방법이 없다.
    """
    reranker = _FixedScoreReranker({"a": 0.62, "b": 0.99}, rerank_weight=0.30)
    run = reranker.rerank_run("q", _ce_tracks()[:2], top_k=2)

    first, second = run.tracks
    assert str(first.id) == "a"
    assert first.rerank_score < second.rerank_score, "리랭커 최고점 곡이 1위가 아니다"

    mix = run.mixes["a"]
    assert mix.retrieval_component > mix.rerank_component, "검색 쪽이 끌어올렸다"
    assert mix.weight == pytest.approx(0.30)


def test_raw_ce_score_is_not_called_normalized():
    """spread_ref를 끄면 CE **원점수**를 그대로 섞는다.

    그때 "정규화된 리랭커 점수"라고 적으면 거짓이다 — 0.62가 0~1로 환산된 값이
    아니라 모델이 낸 값 그 자체다. (운영 기본값은 spread_ref=0.01이라 정규화를
    쓴다. 이 테스트는 끈 경우의 표기를 고정한다.)
    """
    reranker = _FixedScoreReranker({"a": 0.62, "b": 0.99}, spread_ref=0.0)
    run = reranker.rerank_run("q", _ce_tracks()[:2], top_k=2)

    assert run.mixes["a"].rerank_normalized is False
    assert run.mixes["a"].rerank_component == pytest.approx(0.62)

    # 운영 기본값에서는 정규화된 값을 섞는다.
    default_run = _FixedScoreReranker({"a": 0.62, "b": 0.99}).rerank_run(
        "q", _ce_tracks()[:2], top_k=2
    )
    assert default_run.mixes["a"].rerank_normalized is True


def test_low_confidence_tail_is_not_called_reordered():
    """점수는 합성됐지만 **순서에는 쓰이지 않은** 곡을 구분한다.

    저신뢰 전략은 상위 N개만 다시 정렬하고 그 아래는 검색 순서를 그대로 둔다.
    둘을 구분하지 않으면 "리랭커가 이 자리에 뒀다"가 되는데 사실이 아니다.
    """
    reranker = _FixedScoreReranker(
        {"a": 0.50, "b": 0.45, "c": 0.40},
        spread_ref=0.5,          # spread(0.10) < spread_ref → 저신뢰 경로
        low_conf_top_n=2,
    )
    run = reranker.rerank_run("q", _ce_tracks(), top_k=3)

    assert run.mixes["a"].reordered is True
    assert run.mixes["c"].reordered is False
    assert "상위 2개만" in run.mixes["c"].strategy
    assert run.mixes["c"].confidence < 1.0

    rec = ExplainRecorder("q")
    rec.note_rerank_run(RERANK_APPLIED, run.judged_ids)
    rec.set_score_mix("c", run.mixes["c"])
    # 리랭커가 자리를 정하지 않았어도 순위 자체는 그대로일 수 있다. 그때 "리랭킹이
    # N위 유지"라고 하면 하지 않은 일을 말하는 것이다.
    rec.set_rank_before(["a", "b", "c"])
    rec.set_rank_after(["a", "b", "c"])
    rec.set_group_ranks(["a", "b", "c"], ["a", "b", "c"], run.judged_ids)

    text = render_ko(rec.record.get("c"), reorder_stage=REORDER_MODEL_APPLIED)
    assert "리랭커 점수는 합성됐으나 순서는 검색 순서 유지" in text
    assert "상위 2개만" in text
    assert "리랭킹이 3위 유지" not in text, "자리를 정하지 않은 곡을 리랭커에 귀속했다"
    assert "축소" in text, "가중치를 깎았으면 그 사실도 남아야 한다"

    # 같은 실행의 상위 곡은 실제로 리랭커가 정렬했다 — 그쪽은 귀속해도 된다.
    rec.set_score_mix("a", run.mixes["a"])
    head = render_ko(rec.record.get("a"), reorder_stage=REORDER_MODEL_APPLIED)
    assert "리랭킹이 1위 유지" in head


# ---------------------------------------------------------------------------
# 모델이 스스로 쓴 문장 — 검증된 근거와 섞지 않는다
# ---------------------------------------------------------------------------

class _NoteReranker:
    """합성식과 모델 문장을 함께 보고하는 백엔드(Gemini 경로의 모양)."""

    enabled = True
    uses_clarify_answers = False

    def __init__(self, mix=None, notes=None, fail_after=None):
        self.mix = mix
        self.notes = notes or {}
        self.fail_after = fail_after
        self.calls = 0

    def rerank_run(self, query, tracks, top_k):
        self.calls += 1
        if self.fail_after and self.calls >= self.fail_after:
            raise RuntimeError("리랭킹 실패(시험)")
        ids = [str(track.id) for track in tracks]
        out = [
            track.model_copy(update={"rerank_score": 0.9})
            for track in reversed(list(tracks))
        ][:top_k]
        return RerankRun(
            out,
            RERANK_APPLIED,
            ids,
            mixes={sid: self.mix for sid in ids} if self.mix else {},
            model_notes={sid: list(self.notes.get(sid, [])) for sid in ids},
        )

    def rerank(self, query, tracks, top_k):
        return self.rerank_run(query, tracks, top_k).tracks


def _mix(**kw):
    base = dict(
        backend="gemini_listwise",
        weight=0.85,
        configured_weight=0.85,
        rerank_component=0.9,
        retrieval_component=0.5,
        final=0.84,
        strategy="후보 전체 재정렬",
    )
    base.update(kw)
    return ScoreMix(**base)


def test_model_note_is_kept_out_of_the_summary():
    """모델의 문장은 검증된 근거와 같은 문장에 들어가면 안 된다.

    섞으면 어느 쪽이 확인된 사실인지 구별할 수 없다. 별도 필드로만 내보낸다.
    """
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    reranker = _NoteReranker(
        mix=_mix(), notes={"s1": ["비 오는 새벽 정서가 일치"]}
    )
    _run(_router(hits, reranker=reranker), _analysis(), top_k=3, recorder=rec)

    explain = rec.record.get("s1")
    assert explain.model_notes == ["비 오는 새벽 정서가 일치"], "보존은 한다"

    text = render_ko(explain, reorder_stage=rec.record.reorder_stage)
    assert "비 오는 새벽" not in text, "검증되지 않은 문장이 설명에 섞였다"
    assert "최종 점수는" in text, "검증된 합성식은 설명에 들어간다"


def test_mix_and_model_note_are_revoked_on_total_fallback():
    """전체가 폴백하면 합성 점수도 모델의 문장도 최종 순서와 무관해진다."""
    protected = [
        _track("p1", title="보호1", vocal_gender="남성",
               lyric_match_type="exact", lyric_match_score=0.95),
        _track("p2", title="보호2", vocal_gender="남성",
               lyric_match_type="exact", lyric_match_score=0.95),
    ]
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    reranker = _NoteReranker(
        mix=_mix(), notes={"p1": ["가사가 그대로 일치"]}, fail_after=2
    )
    analysis = _analysis(
        vocal_gender="남성",
        lyric_clues=[{"kind": "verbatim", "text": "한 소절"}],
    )

    _run(
        _router(others, lyrics_hits=protected, reranker=reranker),
        analysis, top_k=5, recorder=rec,
    )

    assert rec.record.reorder_committed is False
    for song_id, explain in rec.record.songs.items():
        assert explain.score_mix is None, f"{song_id}: 쓰이지 않은 합성식이 남았다"
        assert explain.model_notes == [], f"{song_id}: 버려진 순서의 문장이 남았다"


def test_rule_move_after_the_model_is_not_attributed_to_the_model():
    """모델 순서 **뒤에** 규칙이 옮긴 위치를 모델의 정렬로 기록하면 안 된다.

    Gemini 경로는 모델 순서에 rescue 규칙을 연달아 적용한다. 반환 순서를 모델의
    순서로 쓰면 "리랭킹이 3위에서 1위로 올림"처럼 모델이 하지 않은 일을 말하게 된다.
    """

    class _RescueReranker:
        enabled = True
        uses_clarify_answers = False

        def rerank_run(self, query, tracks, top_k):
            model_order = list(reversed(list(tracks)))        # 모델이 낸 순서
            final = [model_order[-1], *model_order[:-1]]      # 규칙이 꼴찌를 1위로
            return RerankRun(
                final[:top_k],
                RERANK_APPLIED,
                [str(track.id) for track in tracks],
                order_notes=[
                    (str(final[0].id), "gemini_rare_fact_rescue", "support=0.97")
                ],
                model_order_ids=[str(track.id) for track in model_order],
            )

        def rerank(self, query, tracks, top_k):
            return self.rerank_run(query, tracks, top_k).tracks

    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    results = _run(
        _router(hits, reranker=_RescueReranker()), _analysis(), top_k=3, recorder=rec
    )

    assert [track.id for track in results] == ["s1", "s3", "s2"]
    rescued = rec.record.get("s1")
    assert rescued.group_rank_before == 1
    assert rescued.group_rank_after == 3, "모델은 이 곡을 꼴찌로 내렸다"
    assert rescued.rank_after_rerank == 1, "1위로 만든 것은 규칙이다"

    text = render_ko(rescued, reorder_stage=rec.record.reorder_stage)
    assert "외부 검증된 희소 단서" in text, "규칙이 올렸다는 사실이 먼저 나와야 한다"
    assert "1위에서 3위로 내림" in text, "모델이 한 일은 내린 것이다"
    assert "3위에서 1위로 올림" not in text, "규칙의 이동을 모델에 귀속했다"


# ---------------------------------------------------------------------------
# 응답 노출 — explain=true
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.explain import to_track_explain


def test_track_explain_separates_dropped_contributions():
    """반영된 기여와 탈락한 기여를 다른 칸에 담는다.

    한 목록에 섞으면 합계가 실제 점수보다 커진다(실측 0.081967 vs 0.090032).
    """
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.path("s1", "image", 4, 0.003)
    rec.mark_paths_dropped(["s1"])          # 1차 융합에서 탈락
    rec.path("s1", "lyrics_surface", 1, 0.100)   # 가사 경로로 재합류
    rec.set_fused("s1", 0.100)

    out = to_track_explain(rec.record.get("s1"))

    assert [p.path for p in out.paths] == ["lyrics_surface"]
    assert {p.path for p in out.dropped_paths} == {"text_hybrid", "image"}
    assert out.path_total == pytest.approx(out.fused_score)


def test_track_explain_labels_every_rule_it_carries():
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.adjust("s1", "title_exact", 0.08)
    rec.order("s1", "lyrics_exact_priority", "일반 후보 아래로 내려가지 않음")

    out = to_track_explain(rec.record.get("s1"))

    assert out.paths[0].label == "텍스트(의미+키워드)"
    assert out.adjustments[0].label == "제목 완전일치"
    assert out.order_rules[0].label == "기억한 구절이 가사와 일치해 먼저 배치"


def test_track_explain_names_the_lyric_match_type_for_the_screen():
    """일치 유형의 이름을 서버가 내보낸다.

    화면이 직접 문구를 지으면 "가사 원문에 그대로 있다"처럼 실제보다 강한 말이
    나온다. 비교는 normalize_lyric_surface()를 거친 뒤에 한 것이라 원문 일치가
    아니다. 이름을 한곳에서만 정하면 그 오류가 다시 생길 자리가 없다.
    """
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.note_lyric_match("s1", {"match_type": "exact", "clue_text": "나의 운명인 사랑"})

    out = to_track_explain(rec.record.get("s1"))

    assert out.lyric_match_label == "기억한 구절이 가사와 표기 정규화 후 일치"
    assert "원문" not in out.lyric_match_label


def test_track_explain_leaves_the_lyric_label_empty_without_a_match():
    """가사 일치가 없으면 이름도 없다. 빈 값이 곧 "해당 없음"이다."""
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)

    out = to_track_explain(rec.record.get("s1"))

    assert out.lyric_match is None
    assert out.lyric_match_label == ""


def test_track_explain_labels_the_model_note_as_unverified():
    """모델의 문장은 나가되, 검증되지 않았다는 표시와 **함께만** 나간다."""
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.note_model_reason("s1", ["분위기가 비슷하다"])

    out = to_track_explain(rec.record.get("s1"))

    assert out.model_notes == ["분위기가 비슷하다"]
    assert "검증" in out.model_notes_caption
    assert "분위기가 비슷하다" not in out.summary, "확인된 근거와 섞였다"


class _FakeAnalyzer:
    def analyze(self, query):
        return _analysis(original_query=query)


def _client(router):
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: _FakeAnalyzer()
    app.dependency_overrides[get_search_router] = lambda: router
    return TestClient(app)


def _post(body):
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    router = _router(hits, reranker=_StubReranker())
    try:
        with _client(router) as client:
            response = client.post("/search", json={"query": "비 오는 날", **body})
        assert response.status_code == 200, response.text
        return response.json()
    finally:
        router.shutdown()


def test_route_omits_explain_unless_requested():
    """기본값은 끔. None은 "근거가 없다"가 아니라 "기록을 켜지 않았다"는 뜻이다."""
    body = _post({"top_k": 3})

    assert body["explain"] is None
    assert all(track["explain"] is None for track in body["results"])


def test_route_explain_does_not_change_the_ranking():
    """(가) 불변 조건을 API 경계에서 확인한다."""
    off = _post({"top_k": 3})
    on = _post({"top_k": 3, "explain": True})

    assert [t["id"] for t in off["results"]] == [t["id"] for t in on["results"]]
    assert [t["score"] for t in off["results"]] == [t["score"] for t in on["results"]]


def test_route_explain_carries_the_recorded_evidence():
    body = _post({"top_k": 3, "explain": True})

    top = body["results"][0]
    explain = top["explain"]
    assert explain["song_id"] == top["id"]
    assert explain["paths"][0]["label"] == "텍스트(의미+키워드)"
    assert explain["path_total"] == pytest.approx(explain["fused_score"])
    assert explain["reorder_stage"] == "model_applied"
    assert explain["reorder_stage_label"], "코드만 내보내면 화면이 다시 매핑해야 한다"
    assert "리랭킹" in explain["summary"]

    summary = body["explain"]
    assert summary["modality_weights"]["text"] > 0
    assert summary["rerank_calls"] == summary["rerank_calls_applied"] == 1
    assert summary["rerank_error"] == ""


def test_route_explain_is_filled_for_every_result_not_only_top1():
    """"설명하기"는 top10 어느 곡에서도 눌릴 수 있다."""
    body = _post({"top_k": 3, "explain": True})

    assert len(body["results"]) == 3
    for track in body["results"]:
        explain = track["explain"]
        assert explain is not None
        assert explain["song_id"] == track["id"]
        assert explain["summary"] != "기록된 근거가 없습니다."


# ---------------------------------------------------------------------------
# 평가 결과에 같은 기록 저장 — 952곡 재측정의 진단 도구
# ---------------------------------------------------------------------------

import src.retrieval.evaluate_search_accuracy as ev


def test_eval_rerank_helper_records_the_same_things_as_the_router():
    """평가용 사본도 라우터와 같은 것을 기록해야 한다.

    같은 기록이 아니면 평가 결과의 설명과 실서비스의 설명이 다른 것을 말하게 된다.
    """
    protected = _track("p1", title="보호곡", lyric_match_type="exact",
                       lyric_match_score=0.95)
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")

    final = ev.rerank_preserving_exact_lyrics(
        _StubReranker(), _analysis(), [protected, *others], 4, recorder=rec,
    )

    assert rec.record.reorder_attempted is True
    assert rec.record.reorder_committed is True
    assert rec.record.order_rules_applied is True
    assert [rule.rule for rule in rec.record.get("p1").order_rules] == [
        "lyrics_exact_priority"
    ]
    assert rec.record.get("o1").behind_protected == 1
    assert [track.id for track in final][0] == "p1"
    assert rec.record.get("p1").rank_after_rerank == 1


def test_eval_payload_keeps_the_relevant_song_even_outside_the_results():
    """정답이 결과 밖에 있어도 그 곡의 근거를 함께 저장한다.

    재측정에서 알아야 하는 것은 "1위가 왜 1위인가"보다 "정답이 왜 그 자리인가"다.
    """
    candidates = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    for rank, track in enumerate(candidates, start=1):
        rec.path(track.id, "text_hybrid", rank, 1.0 / (60 + rank))
    rec.set_rank_before([track.id for track in candidates])

    payload = ev.build_explain_payload(
        query_id="q1", split="dev", query="비 오는 날",
        relevant_ids={"s5"}, record=rec.record,
        ordered=candidates, candidates=candidates, top_k=2,
    )

    roles = {row["song_id"]: row["role"] for row in payload["songs"]}
    assert roles == {"s1": "result", "s2": "result", "s5": "relevant"}
    answer = next(row for row in payload["songs"] if row["song_id"] == "s5")
    assert answer["rank"] == 5
    assert answer["in_candidates"] is True
    assert answer["explain"]["paths"][0]["rank"] == 5


def test_eval_payload_distinguishes_a_relevant_song_never_retrieved():
    """후보 풀에 없는 것과 순위가 밀린 것은 고치는 방법이 다르다."""
    candidates = [_track("s1", title="곡1")]
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)

    payload = ev.build_explain_payload(
        query_id="q1", split="dev", query="비 오는 날",
        relevant_ids={"zzz"}, record=rec.record,
        ordered=candidates, candidates=candidates, top_k=1,
    )

    answer = next(row for row in payload["songs"] if row["song_id"] == "zzz")
    assert answer["in_candidates"] is False
    assert answer["rank"] is None
    assert answer["explain"] is None, "검색이 닿지 않은 곡에 근거를 만들면 안 된다"


def test_eval_payload_does_not_invent_records_for_unseen_songs():
    """기록을 조회하는 것만으로 빈 기록이 생기면 안 된다.

    record.get()은 없는 곡을 만들어 넣는다. 저장 단계에서 그것을 쓰면 후보 밖 곡이
    "근거 없음"으로 기록에 들어앉아, 다음에 읽는 사람이 후보였다고 오해한다.
    """
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    before = set(rec.record.songs)

    ev.build_explain_payload(
        query_id="q1", split="dev", query="q",
        relevant_ids={"zzz"}, record=rec.record,
        ordered=[_track("s1")], candidates=[_track("s1")], top_k=1,
    )

    assert set(rec.record.songs) == before


# ---------------------------------------------------------------------------
# 평가 경로의 실패 처리 — 한 질의의 추론 실패가 측정 전체를 멈추면 안 된다
# ---------------------------------------------------------------------------

from src.backend.schemas.explain import SearchExplainOut
from src.retrieval.explain import REORDER_STAGE_LABELS


class _BoomReranker:
    """추론에서 예외를 던지는 백엔드."""

    enabled = True
    uses_clarify_answers = False

    def rerank_run(self, query, tracks, top_k):
        raise RuntimeError("추론 실패(시험)")

    def rerank(self, query, tracks, top_k):
        return self.rerank_run(query, tracks, top_k)


def test_eval_falls_back_instead_of_aborting_the_whole_run():
    """실서비스와 같이 기존 검색 순서로 돌아가고, 다음 질의를 계속해야 한다.

    예외를 올리면 질의 하나의 추론 실패로 측정 전체가 멈추고 요약 파일도 만들어지지
    않는다. 실제로 2건 중 두 번째를 실패시켰을 때 첫 질의만 저장됐다.
    """
    candidates = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    rec = ExplainRecorder("q")
    errors: list = []

    final = ev.rerank_preserving_exact_lyrics(
        _BoomReranker(), _analysis(), candidates, 3,
        recorder=rec, errors=errors,
    )

    assert [track.id for track in final] == ["s1", "s2", "s3"], "검색 순서로 폴백"
    assert errors and "RuntimeError" in errors[0]
    assert rec.record.reorder_committed is False, "반영된 것이 없다"
    assert rec.record.reorder_stage == "model_failed"
    assert rec.record.rerank_error.startswith("RuntimeError")
    assert rec.record.get("s1").rank_after_rerank == 1, "폴백 순서도 기록한다"


def test_eval_fallback_revokes_the_provisional_record():
    """앞 그룹이 성공한 뒤 실패해도 잠정 기록은 남지 않아야 한다."""
    protected = [
        _track("p1", title="보호1", lyric_match_type="exact", lyric_match_score=0.95),
        _track("p2", title="보호2", lyric_match_type="exact", lyric_match_score=0.95),
    ]
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 3)]
    rec = ExplainRecorder("q")
    errors: list = []

    ev.rerank_preserving_exact_lyrics(
        _StubReranker(fail_after=2), _analysis(), [*protected, *others], 4,
        recorder=rec, errors=errors,
    )

    assert errors, "폴백 사유가 보고돼야 한다"
    for song_id, explain in rec.record.songs.items():
        assert explain.reorder_applied is False, f"{song_id}: 적용 표시가 남았다"
        assert explain.order_rules == [], f"{song_id}: 취소된 순서 규칙이 남았다"
        assert explain.score_mix is None, f"{song_id}: 쓰이지 않은 합성식이 남았다"


def test_eval_columns_are_blank_when_recording_is_off():
    """기록을 끈 실행에 "리랭킹 없었음"을 저장하면 안 된다.

    NULL_RECORDER의 빈 기록은 "리랭킹이 없었다"가 아니라 "기록하지 않았다"다.
    실제로 리랭킹이 돈 실행에 not_attempted / 호출 0회가 남았다.
    """
    off = ev.explain_columns(NULL_RECORDER.record, enabled=False)
    assert off["reorder_stage"] == ""
    assert off["rerank_calls"] == ""
    assert off["rerank_calls_applied"] == ""

    # 폴백 사실은 기록을 껐어도 남는다 — 숫자를 읽을 때 반드시 알아야 한다.
    with_error = ev.explain_columns(
        NULL_RECORDER.record, enabled=False, rerank_error="RuntimeError: x"
    )
    assert with_error["rerank_error"] == "RuntimeError: x"


def test_eval_columns_report_the_real_state_when_recording_is_on():
    rec = ExplainRecorder("q")
    rec.set_reorder_attempted(True)
    rec.note_rerank_run(RERANK_APPLIED, ["s1"])
    rec.commit_reorder()

    on = ev.explain_columns(rec.record, enabled=True)
    assert on["reorder_stage"] == "model_applied"
    assert on["rerank_calls"] == 1
    assert on["rerank_calls_applied"] == 1


# ---------------------------------------------------------------------------
# 요청 단위 라벨 — 최종 배치를 말하지 않는다
# ---------------------------------------------------------------------------

def test_stage_label_does_not_deny_the_order_rule_that_did_apply():
    """모델이 실패해도 가사 보호 규칙은 그대로 적용된다.

    그때 요청 단위 라벨이 "검색 순서를 사용"이라고 하면 곡별 설명("최종 순위 2→1위")과
    정면으로 모순한다. 라벨은 리랭킹 단계에 무슨 일이 있었는지까지만 말한다.
    """
    rec = ExplainRecorder("q")
    rec.set_reorder_attempted(True)
    rec.note_rerank_run(RERANK_FAILED, [])       # 예외 없이 실패
    rec.order("p", "lyrics_exact_priority", "일반 후보 아래로 내려가지 않음")
    rec.note_order_rules_applied()
    rec.commit_reorder()                          # 보호 배치는 성사됐다
    rec.set_rank_before(["o", "p"])
    rec.set_rank_after(["p", "o"])

    out = SearchExplainOut.of(rec.record)
    assert out.reorder_stage == REORDER_MODEL_FAILED
    assert out.reorder_stage_label == "리랭킹 실패"
    assert "검색 순서" not in out.reorder_stage_label

    song = render_ko(rec.record.get("p"), reorder_stage=out.reorder_stage)
    assert "최종 순위 2→1위" in song, "곡별 설명은 실제 배치를 말한다"


def test_no_stage_label_claims_the_final_order():
    """라벨은 단계만 말한다 — 최종 순서는 요청 단위로 알 수 없다."""
    for stage, label in REORDER_STAGE_LABELS.items():
        assert "검색 순서" not in label, f"{stage}: 최종 순서를 단정했다"
        assert "순서를 정" not in label, f"{stage}: 최종 순서를 단정했다"


def test_eval_script_saves_every_query_and_the_summary_when_one_rerank_fails(
    tmp_path, monkeypatch
):
    """스크립트 끝단: 2건 중 두 번째 리랭킹이 실패해도 둘 다 저장되고 요약이 나온다.

    고치기 전에는 두 번째에서 예외가 올라와 첫 질의만 저장되고 요약 파일은 아예
    만들어지지 않았다. 952곡 측정에서 이런 일이 생기면 몇 시간이 버려진다.
    """
    csv_path = tmp_path / "queries.csv"
    csv_path.write_text(
        "query_id,split,query_type,query,relevant_ids\n"
        "q1,dev,search,첫 질의,s1\n"
        "q2,dev,search,둘째 질의,s2\n"
        "q3,dev,search,셋째 질의,s3\n",
        encoding="utf-8",
    )

    class _Router:
        async def search(self, analysis, **kw):
            return [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]

    class _Reranker:
        enabled = True
        uses_clarify_answers = False
        config = RerankerConfig()

        def __init__(self):
            self.queries = []

        def load(self):
            return self

        def rerank_run(self, query, tracks, top_k):
            self.queries.append(query)
            if len(self.queries) == 2:
                raise RuntimeError("추론 실패(시험)")
            if len(self.queries) == 3:
                # Gemini 내부 폴백 — 예외 없이 실패 상태를 돌려준다.
                return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])
            out = [
                track.model_copy(update={"rerank_score": 0.9})
                for track in reversed(list(tracks))
            ]
            return RerankRun(out[:top_k], RERANK_APPLIED, [t.id for t in tracks])

        def rerank(self, query, tracks, top_k):
            return self.rerank_run(query, tracks, top_k).tracks

    class _Analyzer:
        def analyze(self, query):
            return _analysis(original_query=query)

    reranker = _Reranker()
    monkeypatch.setattr(ev, "get_query_analyzer", lambda: _Analyzer())
    monkeypatch.setattr(ev, "get_search_router", lambda: _Router())
    monkeypatch.setattr(ev, "get_reranker", lambda: reranker)

    args = ev.build_parser().parse_args(
        ["--input", str(csv_path), "--output-dir", str(tmp_path),
         "--split", "dev", "--top-k", "3", "--candidate-k", "3"]
    )
    asyncio.run(ev.evaluate(args))

    import csv as _csv
    detail = list(
        _csv.DictReader((tmp_path / "search_eval_dev_detail.csv").open(encoding="utf-8-sig"))
    )
    assert [row["query_id"] for row in detail] == ["q1", "q2", "q3"], "모두 저장"
    assert detail[0]["rerank_error"] == "", "첫 질의는 정상"
    assert detail[0]["rerank_fell_back"] == "0"
    assert "RuntimeError" in detail[1]["rerank_error"], "예외 사유가 남는다"
    assert detail[1]["reorder_stage"] == "model_failed"
    assert detail[1]["rerank_fell_back"] == "1"
    # 예외 없이 실패한 질의도 폴백으로 잡혀야 한다.
    assert detail[2]["rerank_error"] == "", "예외는 없었다"
    assert detail[2]["rerank_run_statuses"] == "failed"
    assert detail[2]["rerank_fell_back"] == "1", "내부 폴백이 집계에서 빠졌다"

    summary_path = tmp_path / "search_eval_dev_summary.csv"
    assert summary_path.exists(), "요약 파일이 생성돼야 한다"
    summary = {
        row["metric"]: row["rerank"]
        for row in _csv.DictReader(summary_path.open(encoding="utf-8-sig"))
    }
    assert summary["Rerank fallback query count"] == "2", (
        "예외 1건 + 내부 폴백 1건. 폴백을 따로 적지 않으면 실패가 "
        "'리랭킹 효과 없음'으로 읽힌다"
    )
    assert summary["Rerank partial failure query count"] == "0"


def test_eval_rerank_helper_is_unchanged_by_recording():
    """(가) 불변 조건을 평가 경로에서도 확인한다.

    같은 후보를 넣으면 기록 여부와 무관하게 같은 순서·같은 점수가 나와야 한다.
    (실제 스크립트를 두 번 돌리면 결과가 갈릴 수 있는데, 그것은 기록이 아니라 질의
    분석의 비결정성 때문이다 — `audio_english_query`가 달라지면 오디오 경로 순위가
    달라진다. 그래서 여기서는 분석을 고정하고 후보를 직접 넣어 비교한다.)
    """
    candidates = [
        _track("a", title="가", score=0.030),
        _track("b", title="나", score=0.020),
        _track("c", title="다", score=0.010),
    ]
    reranker = _FixedScoreReranker({"a": 0.62, "b": 0.99, "c": 0.41})

    off = ev.rerank_preserving_exact_lyrics(
        reranker, _analysis(), candidates, 3,
    )
    on = ev.rerank_preserving_exact_lyrics(
        reranker, _analysis(), candidates, 3, recorder=ExplainRecorder("q"),
    )

    assert [t.id for t in off] == [t.id for t in on]
    for left, right in zip(off, on):
        assert left.score == right.score, "점수가 비트 단위로 같아야 한다"
        assert left.rerank_score == right.rerank_score


# ---------------------------------------------------------------------------
# 백엔드 내부 폴백 집계 — 기록 스위치와 무관해야 한다
# ---------------------------------------------------------------------------

class _InnerFallbackReranker:
    """Gemini처럼 **예외 없이** 실패 상태를 돌려주는 백엔드."""

    enabled = True
    uses_clarify_answers = False

    def rerank_run(self, query, tracks, top_k):
        return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])

    def rerank(self, query, tracks, top_k):
        return self.rerank_run(query, tracks, top_k).tracks


@pytest.mark.parametrize("recording_on", [True, False])
def test_inner_fallback_is_counted_even_when_recording_is_off(recording_on):
    """예외 없이 실패한 리랭킹도 폴백으로 세어야 한다.

    예외만 세면 Gemini 내부 폴백이 집계에서 빠지고, 실패가 "리랭킹 효과 없음"으로
    조용히 섞인다. 상태는 기록기와 **별개 통로**로 받아야 기록을 꺼도 남는다.
    """
    candidates = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 3)]
    rec = ExplainRecorder("q") if recording_on else NULL_RECORDER
    errors: list = []
    statuses: list = []

    ev.rerank_preserving_exact_lyrics(
        _InnerFallbackReranker(), _analysis(), candidates, 2,
        recorder=rec, errors=errors, statuses=statuses,
    )

    assert errors == [], "예외는 없었다"
    assert statuses == [RERANK_FAILED], "백엔드가 실패를 보고했다"

    columns = ev.explain_columns(
        rec.record, enabled=recording_on, statuses=statuses
    )
    assert columns["rerank_run_statuses"] == "failed"
    assert columns["rerank_fell_back"] == 1
    assert columns["rerank_partial_failure"] == 0


def test_applied_run_is_not_counted_as_fallback():
    statuses = [RERANK_APPLIED]
    assert ev.rerank_fell_back(statuses, "") is False
    assert ev.rerank_partially_failed(statuses, "") is False


def test_partial_group_failure_is_counted_separately():
    """보호 그룹이 여럿일 때 일부만 실패할 수 있다. 전부 폴백과 구별한다."""
    statuses = [RERANK_APPLIED, RERANK_FAILED]
    assert ev.rerank_fell_back(statuses, "") is False, "반영된 것이 있다"
    assert ev.rerank_partially_failed(statuses, "") is True


def test_no_run_at_all_is_not_called_a_fallback():
    """호출 자체가 없으면 폴백이 아니다 — 이미지 지배 질의처럼 건너뛴 경우."""
    assert ev.rerank_fell_back([], "") is False
    assert ev.rerank_fell_back([], "RuntimeError: x") is True


def test_normal_skip_is_not_counted_as_a_failure():
    """후보가 1곡이거나 리랭킹이 꺼져 있으면 백엔드는 정상적으로 건너뛴다.

    그것을 폴백으로 세면 아무 문제 없는 질의가 실패 목록에 올라간다. Gemini는
    후보 1개를 받으면 호출 없이 skipped를 돌려준다.
    """
    assert ev.rerank_outcome([RERANK_SKIPPED], "") == ev.OUTCOME_SKIPPED
    assert ev.rerank_fell_back([RERANK_SKIPPED], "") is False
    assert ev.rerank_partially_failed([RERANK_SKIPPED], "") is False

    # 일부 그룹이 정상 건너뜀인 것도 부분 실패가 아니다.
    mixed = [RERANK_APPLIED, RERANK_SKIPPED]
    assert ev.rerank_outcome(mixed, "") == ev.OUTCOME_APPLIED
    assert ev.rerank_partially_failed(mixed, "") is False


def test_unknown_status_is_not_counted_as_a_failure():
    """상태를 보고하지 않는 백엔드는 '모른다'다. 모르는 것을 실패로 단정하지 않는다."""
    assert ev.rerank_outcome([RERANK_UNKNOWN], "") == ev.OUTCOME_UNKNOWN
    assert ev.rerank_fell_back([RERANK_UNKNOWN], "") is False


def test_only_real_failure_is_counted():
    assert ev.rerank_outcome([RERANK_FAILED], "") == ev.OUTCOME_FALLBACK
    assert ev.rerank_fell_back([RERANK_FAILED], "") is True
    assert ev.rerank_outcome([RERANK_APPLIED, RERANK_FAILED], "") == (
        ev.OUTCOME_PARTIAL_FAILURE
    )
    assert ev.rerank_outcome([RERANK_APPLIED], "RuntimeError: x") == ev.OUTCOME_EXCEPTION


def test_lyric_surface_score_is_not_called_a_similarity():
    """exact·phonetic의 점수는 문자 유사도가 아니라 **분석 단서의 확신도**다.

    `lyrics_exact_search.py`는 정규화 후 부분문자열로 발견되면 `clue.confidence`를
    그대로 점수로 쓰고, fuzzy일 때만 유사도를 곱한다. 전부 "일치도"라고 적으면
    확신도를 문자 유사도로 읽게 된다 — 실제로 q219를 그렇게 잘못 설명했다.
    """
    hits = [_track("s1", title="정답곡")]
    lyrics = [
        _track("s1", title="정답곡", lyric_match_type="phonetic", lyric_match_score=0.85)
    ]
    rec = ExplainRecorder("q")
    _run(
        _router(hits, lyrics_hits=lyrics),
        _analysis(lyric_clues=[{"kind": "phonetic", "text": "아파운더웨이"}]),
        top_k=1, recorder=rec,
    )

    surface = next(
        p for p in rec.record.get("s1").paths if p.path == "lyrics_surface"
    )
    assert "단서 확신도 0.85" in surface.detail
    assert "일치도" not in surface.detail, "확신도를 문자 유사도로 적었다"
    assert "음차 추정 표기가 가사와 표기 정규화 후 일치" in surface.detail


def test_fuzzy_lyric_score_is_labelled_as_a_blend():
    """fuzzy만 유사도가 섞인다 — 그것도 순수 유사도는 아니다(유사도 × 확신도)."""
    hits = [_track("s1", title="정답곡")]
    lyrics = [
        _track("s1", title="정답곡", lyric_match_type="fuzzy", lyric_match_score=0.72)
    ]
    rec = ExplainRecorder("q")
    _run(
        _router(hits, lyrics_hits=lyrics),
        _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}]),
        top_k=1, recorder=rec,
    )

    surface = next(
        p for p in rec.record.get("s1").paths if p.path == "lyrics_surface"
    )
    assert "유사도×확신도 0.72" in surface.detail
    assert "가사와 근사 일치" in surface.detail


# ---------------------------------------------------------------------------
# 보호 규칙 실험 스위치 — 판정은 한 곳에만 있어야 한다
# ---------------------------------------------------------------------------

def _lyric(sid, match_type, score, **kw):
    return _track(sid, lyric_match_type=match_type, lyric_match_score=score, **kw)


def test_only_exact_is_protected_by_default():
    """0.80은 문자 유사도가 아니라 단서 확신도 기준이다."""
    candidates = [
        _lyric("a", "phonetic", 0.95),      # 검색 1위지만 기본값에서는 보호 안 됨
        _lyric("b", "exact", 0.95),
        _lyric("c", "exact", 0.70),         # 확신도 미달
        _lyric("d", "fuzzy", 0.99),
    ]
    assert sr.select_protected_lyric_ids(candidates) == {"b"}


def test_phonetic_top1_is_protected_only_when_switched_on(monkeypatch):
    """q219: 가사 경로 기여가 텍스트의 7배인데 보호를 못 받아 1위→3위로 내려갔다."""
    candidates = [_lyric("a", "phonetic", 0.85), _lyric("b", "exact", 0.95)]
    assert sr.select_protected_lyric_ids(candidates) == {"b"}

    monkeypatch.setenv("LYRIC_PROTECT_PHONETIC_TOP1", "1")
    assert sr.select_protected_lyric_ids(candidates) == {"a", "b"}


def test_phonetic_protection_is_limited_to_the_top_search_result(monkeypatch):
    """음차는 추정이라 틀릴 수 있고 같은 구절이 여러 곡에 걸친다. 전면 보호가 아니다."""
    monkeypatch.setenv("LYRIC_PROTECT_PHONETIC_TOP1", "1")

    # 1위가 아니면 보호하지 않는다.
    assert sr.select_protected_lyric_ids(
        [_track("z"), _lyric("a", "phonetic", 0.95)]
    ) == set()

    # 1위여도 확신도가 낮으면 보호하지 않는다.
    assert sr.select_protected_lyric_ids([_lyric("a", "phonetic", 0.50)]) == set()


def test_router_and_eval_share_one_protection_rule(monkeypatch):
    """규칙을 한쪽만 고치면 평가와 서비스가 다르게 동작한다. 같은 함수를 쓴다."""
    import inspect

    assert "select_protected_lyric_ids" in inspect.getsource(sr.SearchRouter.search)
    assert "select_protected_lyric_ids" in inspect.getsource(
        ev._rerank_with_lyric_protection
    )


def test_protected_phonetic_survives_the_reranker(monkeypatch):
    """스위치를 켜면 리랭커가 뒤집어도 보호 곡이 앞에 남는다."""
    answer = _lyric("ans", "phonetic", 0.85, title="정답곡")
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]

    def run():
        rec = ExplainRecorder("q")
        results = _run(
            _router(others, lyrics_hits=[answer], reranker=_StubReranker()),
            _analysis(lyric_clues=[{"kind": "phonetic", "text": "아파운더웨이"}]),
            top_k=4, recorder=rec,
        )
        return [t.id for t in results], rec

    off_ids, _ = run()
    monkeypatch.setenv("LYRIC_PROTECT_PHONETIC_TOP1", "1")
    on_ids, on_rec = run()

    assert off_ids[0] != "ans", "기본값에서는 리랭커가 뒤집는다"
    assert on_ids[0] == "ans", "보호를 켜면 앞에 남는다"
    # 음차에 exact용 규칙을 재사용하면 "가사가 그대로 일치"가 붙어 두 번 틀린다.
    assert [r.rule for r in on_rec.record.get("ans").order_rules] == [
        "lyrics_phonetic_priority"
    ]
    text = render_ko(on_rec.record.get("ans"), reorder_stage=on_rec.record.reorder_stage)
    assert "음차 추정 표기의 가사 일치를 보호해 먼저 배치" in text
    assert "그대로 일치" not in text


# ---------------------------------------------------------------------------
# 작은 spread에서 재정렬 생략
# ---------------------------------------------------------------------------

def test_tiny_spread_skips_reordering_when_switched_on():
    """CE가 후보를 거의 구분하지 못하면 순서를 정하지 않는다.

    점수도 검색 점수를 그대로 둔다 — 합성 점수만 넣고 순서를 유지하면 점수와 순서가
    어긋나 "이 곡이 왜 위에 있나"를 설명할 수 없다.
    """
    reranker = _FixedScoreReranker(
        {"a": 0.50000, "b": 0.50002, "c": 0.50001}, min_spread=0.002
    )
    tracks = _ce_tracks()
    run = reranker.rerank_run("q", tracks, top_k=3)

    assert [t.id for t in run.tracks] == ["a", "b", "c"], "검색 순서 그대로"
    assert run.status == RERANK_SKIPPED, "실패가 아니고 적용도 아니다"
    assert run.judged_ids == [], "최종 순서에 반영된 것이 없다"
    assert run.mixes == {}, "쓰이지 않은 합성식을 남기지 않는다"

    assert [t.score for t in run.tracks] == [t.score for t in tracks], "점수 유지"
    assert all(t.rerank_score is not None for t in run.tracks), "진단용 점수는 붙인다"


def test_reordering_still_happens_above_the_threshold():
    """임계값 위에서는 재정렬이 일어난다.

    구체적인 순열은 합성 계수에 따라 달라지므로(운영 기본값 0.45/0.01) 고정하지
    않는다. 확인할 것은 **검색 순서를 그대로 두지 않았다**는 것이다.
    """
    tracks = _ce_tracks()
    reranker = _FixedScoreReranker(
        {"a": 0.10, "b": 0.90, "c": 0.50}, min_spread=0.002
    )
    run = reranker.rerank_run("q", tracks, top_k=3)

    assert run.status == RERANK_APPLIED
    assert [t.id for t in run.tracks] != [t.id for t in tracks]
    assert run.tracks[0].id == "b", "CE 최고점 곡이 맨 앞에 온다"
    assert all(mix.reordered for mix in run.mixes.values())


def test_min_spread_is_off_by_default():
    """기본 동작을 바꾸지 않는다 — 실험 스위치다."""
    assert RerankerConfig().min_spread == 0.0
    run = _FixedScoreReranker({"a": 0.50000, "b": 0.50002, "c": 0.50001}).rerank_run(
        "q", _ce_tracks(), top_k=3
    )
    assert run.status == RERANK_APPLIED


def test_skipped_reorder_is_not_counted_as_a_failure():
    """정책에 따른 생략이지 실패가 아니다. 폴백 집계에 들어가면 안 된다."""
    assert ev.rerank_outcome([RERANK_SKIPPED], "") == ev.OUTCOME_SKIPPED
    assert ev.rerank_fell_back([RERANK_SKIPPED], "") is False


def test_phonetic_protection_pins_a_wrong_song_too(monkeypatch):
    """**이 규칙의 비용을 명시한다.** 보호는 곡이 맞는지 보지 않는다.

    음차는 추정 표기이고 비교도 표기 정규화 후에 하므로, 짧은 구절이 엉뚱한 곡에
    걸려 검색 1위가 되면 그 곡이 그대로 고정된다. dev 53건에는 이 경우가 없어서
    측정으로는 드러나지 않는다 — 그래서 여기에 남긴다.
    """
    monkeypatch.setenv("LYRIC_PROTECT_PHONETIC_TOP1", "1")
    wrong = _lyric("wrong", "phonetic", 0.90, title="엉뚱한 곡")
    right = _track("right", title="정답곡")

    rec = ExplainRecorder("q")
    results = _run(
        _router([right], lyrics_hits=[wrong], reranker=_StubReranker()),
        _analysis(lyric_clues=[{"kind": "phonetic", "text": "짧은구절"}]),
        top_k=2, recorder=rec,
    )

    assert results[0].id == "wrong", "보호는 곡이 맞는지 보지 않는다"
    assert rec.record.get("wrong").reorder_applied is False, (
        "보호 곡은 일반 풀에서 빠져 리랭커가 고칠 수 없다"
    )
    # 설명은 그 사실을 감추지 않는다.
    text = render_ko(rec.record.get("wrong"), reorder_stage=rec.record.reorder_stage)
    assert "음차 추정 표기의 가사 일치를 보호해 먼저 배치" in text


def test_protection_can_be_disabled_for_diagnosis(monkeypatch):
    """보호 규칙 자체의 비용을 재려면 끌 수 있어야 한다."""
    candidates = [_lyric("a", "exact", 0.95)]
    assert sr.select_protected_lyric_ids(candidates) == {"a"}

    monkeypatch.setenv("LYRIC_PROTECT_MIN_CONFIDENCE", "1.01")
    assert sr.select_protected_lyric_ids(candidates) == set()

    monkeypatch.setenv("LYRIC_PROTECT_MIN_CONFIDENCE", "말도안되는값")
    assert sr.select_protected_lyric_ids(candidates) == {"a"}, "잘못된 값은 기본값으로"


# ---------------------------------------------------------------------------
# 가사 부스트 계측 — 관측만 한다. 랭킹은 바뀌지 않는다
# ---------------------------------------------------------------------------

from src.backend.schemas.query import LyricClue
from src.backend.schemas.search import LyricSurfaceMatch
from src.retrieval.lyrics_exact_search import LyricsExactSearchService


def _corpus(*pairs):
    return [
        {"song_id": sid, "metadata": {"title": sid}, "lyrics_data": {"full_lyrics": text}}
        for sid, text in pairs
    ]


def test_clue_kind_and_match_type_are_recorded_separately():
    """q211: **partial 단서가 exact로 판정된다.** 둘을 한 칸에 담으면 구분이 사라진다."""
    service = LyricsExactSearchService(
        documents=_corpus(("a", "오늘도 위험하다 그렇게"), ("b", "전혀 다른 가사"))
    )
    (track,) = service.search([LyricClue(text="위험하다", kind="partial", confidence=0.9)])

    detail = track.lyric_match_detail
    assert detail.clue_kind == "partial", "사용자 단서는 조각이었다"
    assert detail.match_type == "exact", "검색은 정규화 후 부분문자열로 찾았다"
    assert detail.normalized_length == 4


def test_corpus_match_count_is_taken_before_candidate_cutoff():
    """흔한 조각인지 알려면 후보를 자르기 **전** 전체 코퍼스에서 세야 한다."""
    service = LyricsExactSearchService(
        documents=_corpus(
            ("a", "사랑 이야기"), ("b", "사랑해"), ("c", "사랑은"), ("d", "무관한 가사")
        )
    )
    results = service.search([LyricClue(text="사랑", kind="partial", confidence=0.9)])

    assert {t.id for t in results} == {"a", "b", "c"}
    assert all(t.lyric_match_detail.corpus_match_count == 3 for t in results)
    # 겹쳐도 점수는 단서 확신도 그대로다 — 이것이 계측이 드러내는 사실이다.
    assert all(t.lyric_match_score == pytest.approx(0.9) for t in results)


def test_variant_match_is_marked_as_a_variant():
    """음차는 원문이 아니라 추정 표기로 맞을 수 있다. 그 사실을 남긴다."""
    service = LyricsExactSearchService(documents=_corpus(("a", "I FOUND\nTHE WAY!")))
    (track,) = service.search(
        [
            LyricClue(
                text="아파운더웨이", kind="phonetic", confidence=0.85,
                variants=["I found the way"],
            )
        ]
    )

    detail = track.lyric_match_detail
    assert detail.is_variant is True
    assert detail.matched_phrase == "ifoundtheway"
    assert detail.clue_text == "아파운더웨이", "사용자가 적은 것은 보존한다"


def test_lyric_instrumentation_does_not_change_the_ranking():
    """(가) 불변 조건 — 계측을 켜도 순위와 점수가 같아야 한다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    lyrics = [
        _track("s3", title="곡3", lyric_match_type="exact", lyric_match_score=0.9)
    ]
    analysis = _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])

    plain = _run(_router(hits, lyrics_hits=lyrics), analysis, top_k=3)

    detailed = [
        t.model_copy(
            update={
                "lyric_match_detail": LyricSurfaceMatch(
                    clue_text="한 소절", clue_kind="verbatim", matched_phrase="한소절",
                    is_variant=False, normalized_length=3, match_type="exact",
                    confidence=0.9, corpus_match_count=42,
                )
            }
        )
        for t in lyrics
    ]
    rec = ExplainRecorder("q")
    with_detail = _run(
        _router(hits, lyrics_hits=detailed), analysis, top_k=3, recorder=rec
    )

    assert [t.id for t in plain] == [t.id for t in with_detail]
    for a, b in zip(plain, with_detail):
        assert a.score == b.score, "점수가 비트 단위로 같아야 한다"

    # 기록에는 남는다.
    match = rec.record.get("s3").lyric_match
    assert match["corpus_match_count"] == 42
    assert match["clue_kind"] == "verbatim"
    surface = next(p for p in rec.record.get("s3").paths if p.path == "lyrics_surface")
    assert "3자" in surface.detail and "42곡에 겹침" in surface.detail


def test_boost_ranks_show_how_far_the_boost_lifted_the_song():
    """부스트가 몇 칸을 올렸는지가 과도함을 판단하는 재료다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    lyrics = [
        _track("s5", title="곡5", lyric_match_type="exact", lyric_match_score=0.9)
    ]
    rec = ExplainRecorder("q")
    _run(
        _router(hits, lyrics_hits=lyrics),
        _analysis(lyric_clues=[{"kind": "verbatim", "text": "한 소절"}]),
        top_k=5, recorder=rec,
    )

    explain = rec.record.get("s5")
    assert explain.lyric_boost_rank_before == 5
    assert explain.lyric_boost_rank_after == 1, "부스트가 꼴찌를 1위로 올렸다"


# ---------------------------------------------------------------------------
# 죽은 경로 — 기여가 "없는" 것과 경로가 "죽은" 것은 다른 사실이다
#
# 2026-09-24 실패 경로 확인에서 나온 것: 이미지 경로가 예외로 죽어도 응답에는
# image 가중치가 그대로 남아 있었다. 화면에는 그 경로 줄만 사라지므로,
# 보는 사람은 "경로는 돌았지만 이 곡엔 기여가 없었다"로 읽는다.
# ---------------------------------------------------------------------------

class _BrokenEmbedder:
    """이 경로만 죽인다. 다른 경로는 멀쩡해야 검색이 계속되는 것을 볼 수 있다."""

    def embed_texts(self, texts, **kw):
        raise RuntimeError("SigLIP2 추론 실패(시험)")


def test_path_result_records_the_failure_not_just_the_log():
    """되돌려 보면 깨진다: `note_path_failed` 호출을 빼면 기록이 비어 순위 설명이
    '기여 없음'과 구분되지 않는다."""
    rec = ExplainRecorder("q")
    assert sr._path_result("image", RuntimeError("boom"), rec) == []
    assert rec.record.failed_paths == {"image": "RuntimeError: boom"}


def test_path_result_passes_healthy_results_through_untouched():
    rec = ExplainRecorder("q")
    hits = [_track("s1")]
    assert sr._path_result("image", hits, rec) is hits
    assert rec.record.failed_paths == {}


def test_dead_image_path_is_recorded_while_the_search_continues():
    """이미지 경로가 죽어도 검색은 계속되고, **죽었다는 사실이 기록에 남는다.**"""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 6)]
    rec = ExplainRecorder("q")
    analysis = _analysis(
        image_english_query="a blue album cover",
        has_visual_clue=True,
    )
    results = _run(
        _router(hits, image_embedder=_BrokenEmbedder()),
        analysis,
        top_k=3,
        recorder=rec,
    )

    assert results, "한 경로가 죽었다고 검색 전체가 멈추면 안 된다"
    assert rec.record.failed_paths.keys() == {"image"}
    assert "RuntimeError" in rec.record.failed_paths["image"]

    # 핵심: 가중치는 남아 있다. 그래서 기록 없이는 두 상황을 구분할 수 없다.
    assert rec.record.modality_weights["image"] > 0
    top = rec.record.get(results[0].id)
    assert not [p for p in top.paths if p.path == "image"]


def test_response_names_the_dead_path_in_korean():
    """화면이 문구를 짓지 않도록 서버가 이름을 붙여 보낸다."""
    rec = ExplainRecorder("q")
    rec.set_weights(0.6, 0.2, 0.2)
    rec.note_path_failed("image", "RuntimeError: boom")

    out = SearchExplainOut.of(rec.record)
    assert [p.label for p in out.failed_paths] == ["앨범 이미지 경로 실패"]
    assert out.failed_paths[0].path == "image"
    assert "RuntimeError" in out.failed_paths[0].reason
    # 가중치는 그대로 나간다 — 둘을 함께 봐야 읽을 수 있다
    assert out.modality_weights["image"] == 0.2


def test_healthy_run_reports_no_dead_paths():
    rec = ExplainRecorder("q")
    rec.set_weights(0.6, 0.2, 0.2)
    assert SearchExplainOut.of(rec.record).failed_paths == []
