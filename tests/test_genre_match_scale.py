"""장르만 떼어 본 두 실험 배율 — `GENRE_MATCH_SCALE`(최초 질의의 장르 일치 가산) · `GENRE_ANSWER_SCALE`(재질문 장르 답변 가산).

지키는 것: 기본 1.0에서 점수·순서가 그대로다 · 배율은 해당 장르 가산 한 줄만 바꾼다 · 0이면 가산도 기록도 없다 ·
등록 장르가 맞는다는 사실과 점수 가산은 따로 말한다 · 성별·형태·거절·가사 보호·피드백 끝자리 계약은 그대로다.
곡별 기대 순위를 고정하는 시험이 아니다 — 실제 품질은 experiments/genre_memory_gap에서 잰다. 모델·DB 없음.
"""
from __future__ import annotations

import pytest

from src.backend.schemas.query import ArtistTypeClue
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval import search_router as sr
from src.retrieval.clarify import ANSWER_MATCH_MULTIPLIER, answer_matches, apply_answer_bonus
from src.retrieval.explain import ExplainRecorder, render_ko
from src.retrieval.user_explanation import (
    BASIS_NONE,
    BASIS_SCORED,
    ORIGIN_ANSWER,
    SLOT_GENRE,
    STATUS_MATCH,
    build_user_explanation,
)
from src.feedback.settings import RetrievalSettings
from tests.feedback_fakes import store_backend  # noqa: F401 - rig가 쓰는 fixture
from tests.test_feedback_retrieval import SETTINGS as FEEDBACK_SETTINGS
from tests.test_feedback_retrieval import _hits as _feedback_hits
from tests.test_feedback_retrieval import _router as _feedback_router
from tests.test_feedback_retrieval import _search as _feedback_search
from tests.test_feedback_retrieval import rig  # noqa: F401
from tests.test_search_explain import _analysis, _router, _run, _track

UNIT = 1.0 / 61
SCALES = ("GENRE_MATCH_SCALE", "GENRE_ANSWER_SCALE")


@pytest.fixture(autouse=True)
def _default_scales(monkeypatch):
    """시험마다 두 배율을 기본값에서 시작한다(다른 가산 배율도 환경에서 새지 않게)."""
    for name in (*SCALES, "BOOST_SCALE", "ATTR_BOOST_SCALE", "GENDER_MISMATCH_SCALE"):
        monkeypatch.delenv(name, raising=False)


def _meta() -> dict:
    return {
        "ballad": MatchingTrack(id="ballad", score=0.0, title="곡1", artist="A", vocal_gender="여성",
                                genre="발라드", artist_types=["그룹"], release_date="2024.06.26"),
        "rnb": MatchingTrack(id="rnb", score=0.0, title="곡2", artist="B", vocal_gender="여성",
                             genre="R&B/Soul", artist_types=["그룹"], release_date="2024.06.26"),
        "indie": MatchingTrack(id="indie", score=0.0, title="곡3", artist="C", vocal_gender="남성",
                               genre="인디음악, 록/메탈", artist_types=["밴드"], release_date="2017.04.24"),
        "ost": MatchingTrack(id="ost", score=0.0, title="곡4", artist="D", vocal_gender="남성",
                             genre="발라드, 국내드라마", artist_types=["솔로"], release_date="2021.10.11"),
    }


FUSED = [("rnb", 0.0105), ("ballad", 0.0100), ("ost", 0.0095), ("indie", 0.0090)]


def _boost(analysis, *, record: bool = True):
    rec = ExplainRecorder("q") if record else None
    kwargs = {"recorder": rec} if rec is not None else {}
    scored = sr.SearchRouter._apply_explicit_boosts(analysis, FUSED, _meta(), **kwargs)
    deltas = {
        sid: {a.rule: a.delta for a in rec.record.get(sid).adjustments} for sid, _ in FUSED
    } if rec is not None else {}
    return scored, deltas


def _ballad_query():
    return _analysis(original_query="여자 그룹의 발라드 노래", vocal_gender="여성", genre="발라드",
                     artist_type=ArtistTypeClue(values=["그룹"], confidence=0.85))


# ---------------------------------------------------------------------------
# 배율 읽기
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("helper, name", [(sr._genre_match_scale, "GENRE_MATCH_SCALE"),
                                          (sr._genre_answer_scale, "GENRE_ANSWER_SCALE")])
@pytest.mark.parametrize("raw, expected", [
    ("", 1.0), ("1", 1.0), ("1.0", 1.0), ("0.5", 0.5), ("0.25", 0.25), ("0", 0.0),
    ("-1", 0.0),                                        # 음수는 0(기존 배율들과 같은 규칙)
    ("2", 1.0), ("1.0001", 1.0),                        # 실험 범위는 0~1 — 가산을 키우지 않는다
    ("없음", 1.0), ("nan", 1.0), ("inf", 1.0), ("-inf", 1.0), ("1e999", 1.0),   # 못 읽는 값은 기본값
])
def test_scales_read_the_environment_safely(monkeypatch, helper, name, raw: str, expected: float) -> None:
    monkeypatch.setenv(name, raw)
    assert helper() == expected


def test_runinfo_records_the_values_actually_applied(monkeypatch) -> None:
    from src.retrieval.evaluate_search_accuracy import _ranking_switches

    switches = _ranking_switches(None)
    assert (switches["genre_match_scale"], switches["genre_answer_scale"]) == (1.0, 1.0)
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.25")
    monkeypatch.setenv("GENRE_ANSWER_SCALE", "7")        # 범위 밖 — 적용값(1.0)이 적혀야 한다
    switches = _ranking_switches(None)
    assert (switches["genre_match_scale"], switches["genre_answer_scale"]) == (0.25, 1.0)


# ---------------------------------------------------------------------------
# 최초 질의의 장르 일치 가산
# ---------------------------------------------------------------------------

def test_default_keeps_scores_and_order_bit_for_bit(monkeypatch) -> None:
    """배율이 없을 때와 1.0일 때가 같고, 장르 가산은 예전 식(`attr_unit * 1`) 그대로다."""
    unset, deltas = _boost(_ballad_query())
    monkeypatch.setenv("GENRE_MATCH_SCALE", "1.0")
    explicit, _ = _boost(_ballad_query())
    assert unset == explicit
    assert deltas["ballad"]["genre_match"] == UNIT * 1
    assert deltas["ost"]["genre_match"] == UNIT * 1, "'발라드, 국내드라마'도 부분 일치다"
    assert "genre_match" not in deltas["rnb"] and "genre_match" not in deltas["indie"]
    assert [sid for sid, _ in unset][0] == "ballad", "현행: 등록 장르가 발라드인 곡이 R&B 곡을 넘는다"


@pytest.mark.parametrize("scale", ["0.5", "0.25"])
def test_a_partial_scale_changes_only_the_genre_delta(monkeypatch, scale: str) -> None:
    _, base = _boost(_ballad_query())
    monkeypatch.setenv("GENRE_MATCH_SCALE", scale)
    scored, scaled = _boost(_ballad_query())
    for sid in base:
        others = {rule: delta for rule, delta in base[sid].items() if rule != "genre_match"}
        assert {rule: delta for rule, delta in scaled[sid].items() if rule != "genre_match"} == others, sid
    assert scaled["ballad"]["genre_match"] == pytest.approx(UNIT * float(scale), rel=1e-12)
    assert scaled["ost"]["genre_match"] == pytest.approx(UNIT * float(scale), rel=1e-12)
    # 기록한 값이 점수에 실제로 들어간 값이다(장부).
    fused = dict(FUSED)
    for sid, score in scored:
        assert score == pytest.approx(fused[sid] + sum(scaled[sid].values()), rel=1e-12)


def test_zero_gives_no_bonus_and_no_record_but_leaves_other_boosts(monkeypatch) -> None:
    _, base = _boost(_ballad_query())
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0")
    scored, off = _boost(_ballad_query())
    assert all("genre_match" not in rules for rules in off.values()), "0점 가산을 '장르 일치'로 적지 않는다"
    for sid in base:
        assert off[sid] == {rule: delta for rule, delta in base[sid].items() if rule != "genre_match"}
    order = [sid for sid, _ in scored]
    assert order.index("rnb") < order.index("ballad"), "성별·형태 가산이 같으면 융합 순서로 돌아간다"
    # 불일치 감점은 원래도 없다 — 등록 장르가 다른 곡의 점수는 어느 배율에서도 같다.
    assert dict(scored)["rnb"] == pytest.approx(0.0105 + sum(base["rnb"].values()), rel=1e-12)


def test_recording_does_not_change_scores_at_any_scale(monkeypatch) -> None:
    for scale in ("1", "0.5", "0"):
        monkeypatch.setenv("GENRE_MATCH_SCALE", scale)
        assert _boost(_ballad_query(), record=False)[0] == _boost(_ballad_query())[0]


@pytest.mark.parametrize("stated", ["인디", "록"])
def test_a_song_registered_under_several_genres_gets_one_scaled_bonus(monkeypatch, stated: str) -> None:
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.5")
    _, deltas = _boost(_analysis(original_query=f"{stated} 노래", genre=stated))
    assert deltas["indie"] == {"genre_match": pytest.approx(UNIT * 0.5, rel=1e-12)}


def test_band_type_bonus_is_kept_when_the_genre_bonus_is_off(monkeypatch) -> None:
    """'인디 밴드'의 두 단서는 따로 가산된다. 장르 배율은 가수 형태 가산을 건드리지 않는다."""
    indie_band = _analysis(original_query="인디 밴드가 부른 노래", genre="인디",
                           artist_type=ArtistTypeClue(values=["밴드"], confidence=0.85))
    _, base = _boost(indie_band)
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0")
    _, off = _boost(indie_band)
    assert base["indie"]["artist_type"] == off["indie"]["artist_type"] == pytest.approx(UNIT * 0.85)
    assert "genre_match" in base["indie"] and "genre_match" not in off["indie"]


def test_it_multiplies_with_the_attribute_scale(monkeypatch) -> None:
    monkeypatch.setenv("ATTR_BOOST_SCALE", "0.2")
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.5")
    _, deltas = _boost(_ballad_query())
    assert deltas["ballad"]["genre_match"] == pytest.approx(UNIT * 0.2 * 0.5, rel=1e-12)
    assert deltas["ballad"]["vocal_gender_match"] == pytest.approx(UNIT * 0.2, rel=1e-12), "성별은 장르 배율과 무관하다"


def test_the_exact_lyric_tier_score_is_held_fixed(monkeypatch) -> None:
    """가사 exact 후보를 묶는 보조 점수의 장르 +1.0은 이번 실험에서 바꾸지 않는다(문서에 고정 요인으로 적었다)."""
    track = _meta()["ballad"]
    before = sr.exact_lyric_constraint_score(_ballad_query(), track)
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0")
    monkeypatch.setenv("GENRE_ANSWER_SCALE", "0")
    assert sr.exact_lyric_constraint_score(_ballad_query(), track) == before == 3.0   # 성별 2.0 + 장르 1.0


def test_exact_lyric_protection_is_kept(monkeypatch) -> None:
    """가사가 그대로 일치한 곡은 장르 배율과 무관하게 보호 대상이고 후보에 남는다."""
    hits = [_track(f"s{i}", title=f"곡{i}", genre="발라드") for i in range(1, 6)]
    lyric = _track("lyric", title="가사곡", genre="댄스", lyric_match_type="exact", lyric_match_score=0.9)
    analysis = _analysis(original_query="발라드인데 '너를 사랑해'라는 가사", genre="발라드",
                         lyric_clues=[{"text": "너를 사랑해", "kind": "verbatim", "confidence": 0.9}])
    for scale in ("1", "0"):
        monkeypatch.setenv("GENRE_MATCH_SCALE", scale)
        tracks: list = []
        _run(_router(hits, lyrics_hits=[lyric]), analysis, top_k=5, use_rerank=False, candidate_tracks_out=tracks)
        assert tracks[0].id == "lyric"
        assert sr.select_protected_lyric_ids(tracks) == {"lyric"}


def test_the_router_ledger_balances_at_a_partial_scale(monkeypatch) -> None:
    """실제 search() 흐름에서도 경로 기여 + 가감점 = 검색 점수다. 기록한 장르 가산이 곧 적용한 값이다."""
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.5")
    hits = [_track(f"s{i}", title=f"곡{i}", genre="발라드" if i % 2 else "댄스") for i in range(1, 9)]
    rec = ExplainRecorder("q")
    tracks: list = []
    _run(_router(hits), _analysis(genre="발라드"), top_k=8, use_rerank=False, recorder=rec, candidate_tracks_out=tracks)
    for track in tracks:
        song = rec.record.songs[track.id]
        assert track.score == pytest.approx(
            sum(p.delta for p in song.live_paths()) + sum(a.delta for a in song.adjustments), rel=1e-12)
        genre = [a.delta for a in song.adjustments if a.rule == "genre_match"]
        assert genre == ([pytest.approx(UNIT * 0.5, rel=1e-12)] if track.genre == "발라드" else [])


# ---------------------------------------------------------------------------
# 재질문 장르 답변 가산
# ---------------------------------------------------------------------------

def _answer_pool():
    tracks = {
        "b": MatchingTrack(id="b", score=0.0, title="곡", artist="A", vocal_gender="여성", genre="발라드"),
        "r": MatchingTrack(id="r", score=0.0, title="곡", artist="B", vocal_gender="여성", genre="R&B/Soul"),
        "m": MatchingTrack(id="m", score=0.0, title="곡", artist="C", vocal_gender="남성", genre="발라드"),
    }
    return [("r", 0.012), ("b", 0.011), ("m", 0.010)], tracks


def test_answer_bonus_default_is_unchanged() -> None:
    scored, tracks = _answer_pool()
    answers = [ClarifyAnswer(slot="genre", value="발라드")]
    rec = ExplainRecorder("q")
    default = apply_answer_bonus(scored, tracks, answers, UNIT, recorder=rec)
    assert default == apply_answer_bonus(scored, tracks, answers, UNIT, genre_scale=1.0)
    assert dict(default)["b"] == 0.011 + UNIT * ANSWER_MATCH_MULTIPLIER
    assert rec.record.get("b").adjustments[0].delta == UNIT * ANSWER_MATCH_MULTIPLIER


def test_answer_scale_touches_only_genre_answers() -> None:
    scored, tracks = _answer_pool()
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성"), ClarifyAnswer(slot="genre", value="발라드")]
    full = dict(apply_answer_bonus(scored, tracks, answers, UNIT))
    rec = ExplainRecorder("q")
    half = dict(apply_answer_bonus(scored, tracks, answers, UNIT, recorder=rec, genre_scale=0.5))
    bonus = UNIT * ANSWER_MATCH_MULTIPLIER
    assert half["r"] == full["r"] == pytest.approx(0.012 + bonus), "성별 답만 맞는 곡은 그대로다"
    assert half["b"] == pytest.approx(0.011 + bonus + bonus * 0.5)
    assert half["m"] == pytest.approx(0.010 + bonus * 0.5)
    recorded = {(a.slot, a.value): a.delta for a in rec.record.get("b").adjustments}
    assert recorded == {("vocal_gender", "여성"): pytest.approx(bonus), ("genre", "발라드"): pytest.approx(bonus * 0.5)}


def test_answer_scale_zero_gives_no_genre_bonus_without_changing_what_matches() -> None:
    scored, tracks = _answer_pool()
    genre = ClarifyAnswer(slot="genre", value="발라드")
    rec = ExplainRecorder("q")
    off = apply_answer_bonus(scored, tracks, [genre], UNIT, recorder=rec, genre_scale=0.0)
    assert off == scored, "가산이 없으면 순서도 점수도 그대로다"
    assert all(not rec.record.get(sid).adjustments for sid, _ in scored), "0점 가산을 기록하지 않는다"
    assert answer_matches(tracks["b"], genre) and not answer_matches(tracks["r"], genre), "일치 판정은 그대로다"
    # 성별 답은 0이어도 반영된다 — 다른 슬롯의 정책을 함께 바꾸지 않는다.
    gender = dict(apply_answer_bonus(scored, tracks, [ClarifyAnswer(slot="vocal_gender", value="남성")], UNIT,
                                     genre_scale=0.0))
    assert gender["m"] == 0.010 + UNIT * ANSWER_MATCH_MULTIPLIER


def test_answer_scale_zero_does_not_swallow_answers_that_follow_the_genre_answer() -> None:
    """장르 답이 먼저 오고 다른 슬롯의 답이 뒤따라도, 배율 0은 장르 답 하나만 건너뛴다(뒤의 답은 그대로 반영)."""
    scored, tracks = _answer_pool()
    answers = [ClarifyAnswer(slot="genre", value="발라드"), ClarifyAnswer(slot="vocal_gender", value="여성")]
    rec = ExplainRecorder("q")
    off = dict(apply_answer_bonus(scored, tracks, answers, UNIT, recorder=rec, genre_scale=0.0))
    bonus = UNIT * ANSWER_MATCH_MULTIPLIER
    assert off["b"] == pytest.approx(0.011 + bonus), "장르·성별 모두 맞는 곡 — 성별 답의 가산은 받는다"
    assert off["r"] == pytest.approx(0.012 + bonus) and off["m"] == 0.010
    assert [(a.slot, a.value) for a in rec.record.get("b").adjustments] == [("vocal_gender", "여성")]
    # 순서를 바꿔도 같다.
    assert off == dict(apply_answer_bonus(scored, tracks, answers[::-1], UNIT, genre_scale=0.0))


def test_a_skipped_genre_answer_never_scores() -> None:
    scored, tracks = _answer_pool()
    skipped = [ClarifyAnswer(slot="genre", skipped=True)]
    for scale in (1.0, 0.5, 0.0):
        assert apply_answer_bonus(scored, tracks, skipped, UNIT, genre_scale=scale) == scored


def _round_trip(monkeypatch, scale, **kw):
    if scale is None:
        monkeypatch.delenv("GENRE_ANSWER_SCALE", raising=False)
    else:
        monkeypatch.setenv("GENRE_ANSWER_SCALE", scale)
    hits = [_track(f"s{i}", title=f"곡{i}", genre="발라드" if i >= 6 else "댄스",
                   vocal_gender="여성" if i % 2 else "남성") for i in range(1, 11)]
    pool: list = []
    rec = ExplainRecorder("q")
    _run(_router(hits), _analysis(original_query="분위기 좋은 노래"), top_k=10, use_rerank=False,
         candidate_k=10, candidate_ids_out=pool, recorder=rec, **kw)
    return pool, rec.record


def test_router_round_trip_applies_the_answer_scale_and_keeps_rejections(monkeypatch) -> None:
    """실제 search() 왕복: 답은 후보 집합을 바꾸지 않고 순서만 바꾼다. 거절한 곡은 어느 배율에서도 돌아오지 않는다."""
    genre = [ClarifyAnswer(slot="genre", value="발라드")]
    plain, _ = _round_trip(monkeypatch, None, exclude_ids=["s1"])
    answered, record = _round_trip(monkeypatch, None, exclude_ids=["s1"], answers=genre)
    off, off_record = _round_trip(monkeypatch, "0", exclude_ids=["s1"], answers=genre)
    assert set(answered) == set(off) == set(plain) and "s1" not in plain
    assert answered[:5] == ["s6", "s7", "s8", "s9", "s10"], "현행: 답과 등록 장르가 같은 곡이 올라온다"
    assert off == plain, "0이면 장르 답은 순서를 바꾸지 않는다"
    assert any(a.rule == "answer_match" for a in record.songs["s6"].adjustments)
    assert not any(a.rule == "answer_match" for s in off_record.songs.values() for a in s.adjustments)

    # 성별 답은 장르 답변 배율과 무관하다.
    gender = [ClarifyAnswer(slot="vocal_gender", value="남성")]
    assert _round_trip(monkeypatch, "0", answers=gender)[0] == _round_trip(monkeypatch, None, answers=gender)[0]
    # "잘 모르겠어요"는 어느 배율에서도 거절만 한 것과 같다.
    skipped = [ClarifyAnswer(slot="genre", skipped=True)]
    assert _round_trip(monkeypatch, "0.5", exclude_ids=["s1"], answers=skipped)[0] == plain


def test_the_two_scales_are_independent(monkeypatch) -> None:
    """최초 질의의 장르 가산과 재질문 장르 답변 가산은 다른 요인이다 — 한쪽 배율이 다른 쪽을 바꾸지 않는다."""
    monkeypatch.setenv("GENRE_ANSWER_SCALE", "0")
    _, deltas = _boost(_ballad_query())
    assert deltas["ballad"]["genre_match"] == UNIT * 1
    monkeypatch.delenv("GENRE_ANSWER_SCALE")
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0")
    _, record = _round_trip(monkeypatch, None, answers=[ClarifyAnswer(slot="genre", value="발라드")])
    assert [a.delta for a in record.songs["s6"].adjustments if a.rule == "answer_match"] == [
        UNIT * ANSWER_MATCH_MULTIPLIER]


# ---------------------------------------------------------------------------
# 설명 — 등록 장르가 맞는다는 사실과 점수 가산은 따로 말한다
# ---------------------------------------------------------------------------

def _genre_row(analysis, track, record, answers=()):
    explanation = build_user_explanation(
        query=analysis.original_query, analysis=analysis, search_analysis=analysis, answers=list(answers),
        track=track, explain=record.get(track.id),
    )
    rows = [c for c in explanation.clue_checks if c.slot == SLOT_GENRE]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("scale, label", [("1", BASIS_SCORED), ("0.5", BASIS_SCORED), ("0", BASIS_NONE)])
def test_explanation_shows_the_delta_actually_applied(monkeypatch, scale: str, label: str) -> None:
    monkeypatch.setenv("GENRE_MATCH_SCALE", scale)
    analysis = _analysis(original_query="발라드 노래 찾아줘", genre="발라드")
    track = _meta()["ballad"]
    rec = ExplainRecorder("q")
    scored = dict(sr.SearchRouter._apply_explicit_boosts(analysis, [("ballad", 0.01)], {"ballad": track}, recorder=rec))
    row = _genre_row(analysis, track, rec.record)
    assert row.status == STATUS_MATCH and "등록 장르: 발라드" in row.observed, "등록 장르가 맞는다는 사실은 배율과 무관하다"
    assert row.basis_label == label
    applied = scored["ballad"] - 0.01
    assert [b.delta for b in row.ranking_basis] == ([] if scale == "0" else [pytest.approx(applied, rel=1e-9)])
    assert applied == pytest.approx(UNIT * float(scale), rel=1e-9)
    if scale == "0":
        # 점수에 반영되지 않은 단서를 가산 근거라고 말하지 않는다(내부 기록 문장에도 없다).
        assert "장르 일치" not in render_ko(rec.record.get("ballad"), reorder_stage=rec.record.reorder_stage)


def test_explanation_for_a_genre_answer_without_bonus_claims_no_ranking(monkeypatch) -> None:
    analysis = _analysis(original_query="분위기 좋은 노래")
    track = _meta()["ballad"]
    answer = ClarifyAnswer(slot="genre", value="발라드")
    for scale, label in ((1.0, BASIS_SCORED), (0.0, BASIS_NONE)):
        rec = ExplainRecorder("q")
        rec.path("ballad", "text_hybrid", 1, UNIT)
        apply_answer_bonus([("ballad", 0.01)], {"ballad": track}, [answer], UNIT, recorder=rec, genre_scale=scale)
        row = _genre_row(analysis, track, rec.record, answers=[answer])
        assert (row.status, row.clue_origin, row.basis_label) == (STATUS_MATCH, ORIGIN_ANSWER, label)


# ---------------------------------------------------------------------------
# 피드백 후보 보충 — 진입 판정이 명시 단서 보정(장르 가산 포함)을 읽는다
# ---------------------------------------------------------------------------

def test_feedback_entry_uses_the_scaled_genre_bonus_and_keeps_the_tail_contract(rig, monkeypatch) -> None:  # noqa: F811
    """보충 곡의 진입은 '밀려날 꼬리 후보의 융합 점수 + 명시 단서 보정'으로 정한다. 후보가 모두 말한 장르로
    등록돼 있으면 등록 장르가 다른 보충 곡은 현행에서 못 들어오고, 장르 가산을 끄면 들어온다 — 장르 배율이
    피드백 보충에 닿는 길은 이 하나다. 어느 배율에서도 들어온 곡은 후보 끝에 서고 나머지 후보는 그대로다."""
    rig.case("f3", "휘파람 부는 노래")                       # f3의 등록 장르는 댄스
    analysis = _analysis(original_query="휘파람 불던 노래", genre="발라드")
    entered = {}
    for scale in ("1", "0"):
        monkeypatch.setenv("GENRE_MATCH_SCALE", scale)
        baseline = _feedback_search(_feedback_router(_feedback_hits(genre="발라드")), analysis)
        on = _feedback_search(_feedback_router(_feedback_hits(genre="발라드"), rig.retriever), analysis)
        entered[scale] = on["feedback"][0].entered_ids
        kept = 30 - len(entered[scale])
        assert on["pool"][:kept] == baseline["pool"][:kept], scale
        assert on["scores"][:kept] == baseline["scores"][:kept], "끝자리를 뺀 후보는 점수도 순서도 보충이 없을 때와 같다"
        assert on["shown"] == baseline["shown"], "리랭커 없이 나가는 표시 결과는 보충이 없을 때와 같다"
        if entered[scale]:
            assert on["pool"][-1] == "f3" and on["record"].songs["f3"].feedback_entrant
    assert entered == {"1": [], "0": ["f3"]}


def test_feedback_tail_is_computed_with_the_same_answer_scale(rig, monkeypatch) -> None:  # noqa: F811
    """보충 곡이 들어올 때 라우터는 "보충이 없을 때의 답변 반영 순서"를 따로 계산해 그 끝자리를 내준다. 그 계산도 같은
    장르 답변 배율을 읽어야 한다 — 어느 배율에서도 끝자리를 뺀 후보는 같은 배율로 잰 보충 없는 검색과 같다."""
    rig.case("f1", "휘파람 노래")
    # 사례 조회문에는 답("장르: 발라드")이 붙는다 — 짧은 합성 사례와의 유사도가 내려가므로 문턱을 낮춰 둔다.
    rig.settings = RetrievalSettings(**{**FEEDBACK_SETTINGS.__dict__, "min_similarity": 0.5})
    hits = [t.model_copy(update={"genre": "발라드" if i % 3 == 0 else "댄스"})
            for i, t in enumerate(_feedback_hits(), start=1)]
    answer = [ClarifyAnswer(slot="genre", value="발라드")]
    pools = {}
    for scale in (None, "0"):
        if scale is not None:
            monkeypatch.setenv("GENRE_ANSWER_SCALE", scale)
        baseline = _feedback_search(_feedback_router(hits), answers=answer)
        on = _feedback_search(_feedback_router(hits, rig.retriever), answers=answer)
        assert on["feedback"][0].entered_ids == ["f1"], scale
        assert on["pool"] == baseline["pool"][:29] + ["f1"] and on["scores"][:29] == baseline["scores"][:29], scale
        assert on["feedback"][0].displaced_ids == [baseline["pool"][-1]], scale
        pools[scale] = baseline["pool"]
    assert pools[None][:3] == ["s3", "s6", "s9"], "현행: 답과 등록 장르가 같은 곡이 앞에 선다"
    assert pools["0"][:3] == ["s1", "s2", "s3"], "0이면 장르 답은 순서를 바꾸지 않는다"
