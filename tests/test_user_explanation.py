"""사용자용 검색 근거 설명(`src/retrieval/user_explanation.py`) — 계획서 §6 필수 검증.

고정하는 것: 곡 정보 일치와 실제 순위 기여를 따로 말한다. 기록이 없으면 "반영"을 쓰지 않는다.
자료 없음은 미확인이다. 정정한 최신 답이 현재 단서다. 없는 인용을 만들지 않는다. 입력을 바꾸지 않는다.
"""
from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace

import pytest

from src.backend.schemas.explain import to_track_explain
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval.clarify import with_birth_year
from src.retrieval.explain import RERANK_SKIP_MV_SCENE_UNVERIFIABLE, ExplainRecorder
from src.retrieval.lyrics_exact_search import LyricExcerpt
from src.retrieval.user_explanation import (
    BASIS_NEAR,
    BASIS_NONE,
    BASIS_PROTECTED,
    BASIS_RERANK_SKIPPED,
    BASIS_SCORED,
    BASIS_SEMANTIC,
    BASIS_SUPPLEMENTARY,
    ORIGIN_ANSWER,
    ORIGIN_BIRTH_YEAR,
    ORIGIN_QUERY,
    SLOT_ERA,
    SLOT_GENDER,
    SLOT_GENRE,
    SLOT_LYRICS,
    SLOT_MV,
    KIND_ERA_EARLIER_THAN_ESTIMATE,
    KIND_LYRIC_COMBINED,
    KIND_LYRIC_HIGHLIGHT_PHONETIC,
    KIND_LYRIC_SUMMARY,
    KIND_LYRIC_SUMMARY_PHONETIC,
    KIND_LYRIC_UNATTRIBUTED,
    KIND_SLOT_CONFLICT,
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_PARTIAL,
    STATUS_UNVERIFIED,
    build_user_explanation,
)

UPSIDE_DOWN_QUERY = "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"


def _analysis(**kw) -> QueryAnalysis:
    base = dict(original_query="비 오는 날 발라드", intent_type="mood",
                image_english_query="", audio_english_query="")
    base.update(kw)
    return QueryAnalysis(**base)


def _track(**kw) -> MatchingTrack:
    base = dict(id="s1", score=0.1, title="Upside Down", artist="가수",
                vocal_gender="남성", genre=["댄스"], release_date="2019.03.01",
                lyrics_highlight="옥상 위로 올라가 수박을 먹어")
    base.update(kw)
    return MatchingTrack(**base)


def _record(song_id="s1", query="q", *, adjust=(), paths=(("text_hybrid", 1, 0.016),), order=(), lyric_match=None):
    rec = ExplainRecorder(query)
    for path, rank, delta in paths:
        rec.path(song_id, path, rank, delta)
    for item in adjust:
        rule, delta = item[0], item[1]
        detail = item[2] if len(item) > 2 else ""
        extra = item[3] if len(item) > 3 else {}
        rec.adjust(song_id, rule, delta, detail, **extra)
    for rule, detail in order:
        rec.order(song_id, rule, detail)
    if lyric_match:
        rec.note_lyric_match(song_id, lyric_match)
    return rec


def _build(query, analysis, track, rec, *, answers=(), evidence=(), skip="", search_analysis=None):
    return build_user_explanation(
        query=query, analysis=analysis, search_analysis=search_analysis or analysis,
        answers=list(answers), track=track, explain=rec.record.get(track.id),
        evidence=list(evidence), rerank_skip_reason=skip,
    )


def _row(ux, slot):
    rows = [c for c in ux.clue_checks if c.slot == slot]
    assert rows, f"{slot} 행이 없다: {[c.slot for c in ux.clue_checks]}"
    return rows[0]


# ---------------------------------------------------------------------------
# Upside Down 사례 — 남성 일치, 인디·장면 미확인. 옥상 가사로 MV 주장을 만들지 않는다
# ---------------------------------------------------------------------------

def test_upside_down_case_matches_gender_and_leaves_indie_and_scene_unverified():
    analysis = _analysis(original_query=UPSIDE_DOWN_QUERY, vocal_gender="남성", genre="인디")
    ux = _build(UPSIDE_DOWN_QUERY, analysis, _track(), _record(), skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE)

    gender, genre, mv = _row(ux, SLOT_GENDER), _row(ux, SLOT_GENRE), _row(ux, SLOT_MV)
    assert gender.status == STATUS_MATCH and gender.clue == "남성 보컬" and gender.clue_origin == ORIGIN_QUERY
    assert gender.basis_label == BASIS_NONE, "가산 기록이 없으면 반영이라고 말하지 않는다"
    assert genre.status == STATUS_UNVERIFIED and "인디 여부" in genre.observed and "댄스" in genre.observed
    assert "아니" not in genre.observed, "인디가 아니라고 단정하면 안 된다"
    assert mv.status == STATUS_UNVERIFIED and mv.basis_label == BASIS_RERANK_SKIPPED
    assert "의자" in mv.clue, "사용자가 말한 장면을 그대로 보인다"
    assert "옥상" not in mv.observed and "옥상" not in ux.summary, "가사 속 장면이 MV 근거로 둔갑했다"

    assert ux.summary.startswith("남성 보컬 단서는 곡 정보와 일치해요.")
    assert "반영" not in ux.summary
    assert "뮤직비디오 장면" in ux.summary and "인디 여부" in ux.summary
    assert ux.summary.count(".") <= 2, ux.summary


def test_mv_row_appears_even_when_rerank_was_not_skipped_but_never_claims_evidence():
    analysis = _analysis(original_query=UPSIDE_DOWN_QUERY)
    ux = _build(UPSIDE_DOWN_QUERY, analysis, _track(), _record(), skip="")
    mv = _row(ux, SLOT_MV)
    assert mv.status == STATUS_UNVERIFIED and mv.basis_label == BASIS_NONE and not mv.ranking_basis


def test_non_scene_mv_mention_does_not_create_an_mv_row():
    query = "남자 가수인데 뮤비가 유명한 노래"
    ux = _build(query, _analysis(original_query=query), _track(), _record())
    assert all(c.slot != SLOT_MV for c in ux.clue_checks)


# ---------------------------------------------------------------------------
# 같은 남성 메타데이터, 가산 기록 있음/없음 — 일치는 같고 "점수 반영" 문구만 다르다
# ---------------------------------------------------------------------------

def test_same_metadata_match_differs_only_in_ranking_basis_when_adjustment_is_recorded():
    query = "남자 발라드"
    analysis = _analysis(original_query=query, vocal_gender="남성", genre="발라드")
    track = _track(genre=["발라드"])

    without = _build(query, analysis, track, _record())
    with_adj = _build(query, analysis, track, _record(adjust=[
        ("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"}),
    ]))

    g0, g1 = _row(without, SLOT_GENDER), _row(with_adj, SLOT_GENDER)
    assert g0.status == g1.status == STATUS_MATCH and g0.observed == g1.observed
    assert g0.basis_label == BASIS_NONE and not g0.ranking_basis
    assert g1.basis_label == BASIS_SCORED and g1.ranking_basis[0].code == "vocal_gender_match"
    assert "검색 점수에 반영" not in without.summary
    assert without.summary == "남성 보컬·발라드 단서는 곡 정보와 일치해요."
    # 반영된 단서와 일치만 한 단서를 한 문장 안에서 갈라 말한다
    assert with_adj.summary == "남성 보컬 단서는 곡 정보와 일치해 검색 점수에 반영됐고, 발라드 단서는 곡 정보와 일치해요."
    assert "순위가 올라" not in with_adj.summary and "1위" not in with_adj.summary


def test_gender_adjustment_without_structured_value_falls_back_to_the_search_analysis_value():
    """옛 기록(value 없음)은 검색용 분석의 성별로 비교한다 — 같을 때만 연결."""
    query = "남자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="남성")
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성")])
    ux = _build(query, analysis, _track(), rec)
    assert _row(ux, SLOT_GENDER).basis_label == BASIS_SCORED


# ---------------------------------------------------------------------------
# 혼성 · 다름 · 등록 정보 없음
# ---------------------------------------------------------------------------

def test_mixed_vocal_is_partial_not_full_match_and_links_only_the_partial_rule():
    query = "여자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="여성")
    rec = _record(adjust=[("vocal_gender_partial", 0.001, "혼성", {"slot": "vocal_gender", "value": "여성"})])
    ux = _build(query, analysis, _track(vocal_gender="혼성"), rec)
    row = _row(ux, SLOT_GENDER)
    assert row.status == STATUS_PARTIAL and "혼성" in row.observed
    assert row.ranking_basis[0].code == "vocal_gender_partial"
    assert "부분적으로만 일치" in ux.summary and "곡 정보와 일치해요" not in ux.summary


def test_opposite_gender_is_mismatch_and_unknown_gender_is_unverified():
    query = "여자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="여성")
    mismatch = _row(_build(query, analysis, _track(vocal_gender="남성"), _record()), SLOT_GENDER)
    assert mismatch.status == STATUS_MISMATCH and "등록 정보와 달라요" in _build(
        query, analysis, _track(vocal_gender="남성"), _record()).summary
    unknown = _row(_build(query, analysis, _track(vocal_gender="unknown"), _record()), SLOT_GENDER)
    assert unknown.status == STATUS_UNVERIFIED and "없음" in unknown.observed


def test_uncertain_gender_in_query_creates_no_gender_row():
    query = "남자 가수인지 모르겠어 발라드야"
    ux = _build(query, _analysis(original_query=query, vocal_gender="남성"), _track(), _record())
    assert all(c.slot != SLOT_GENDER for c in ux.clue_checks), "불확실한 표현은 확정 단서가 아니다"


# ---------------------------------------------------------------------------
# 재질문 정정 · 건너뛰기 · 과거 가산
# ---------------------------------------------------------------------------

def test_corrected_answer_is_the_current_clue_and_old_gender_bonus_is_not_linked():
    query = "남자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="남성")   # 검색은 남성 기준으로 돌았다
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"})])
    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
    ux = _build(query, analysis, _track(vocal_gender="남성"), rec, answers=answers)

    row = _row(ux, SLOT_GENDER)
    assert row.clue == "여성 보컬" and row.clue_origin == ORIGIN_ANSWER
    assert row.status == STATUS_MISMATCH
    assert not row.ranking_basis and row.basis_label == BASIS_NONE, "남성 기준 가산을 여성 단서의 근거로 쓰면 안 된다"
    assert "반영" not in ux.summary and "달라요" in ux.summary
    # 과거 계산은 상세 기록에 그대로 남는다
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    assert [a.rule for a in out.adjustments] == ["vocal_gender_match"]
    assert out.adjustments[0].value == "남성"


def test_skipped_answer_is_not_a_correction():
    query = "남자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="남성")
    answers = [ClarifyAnswer(slot="vocal_gender", value="", skipped=True)]
    row = _row(_build(query, analysis, _track(), _record(), answers=answers), SLOT_GENDER)
    assert row.clue == "남성 보컬" and row.clue_origin == ORIGIN_QUERY


def test_latest_valid_answer_wins_over_an_earlier_one_and_links_its_own_bonus():
    query = "분위기 좋은 노래"
    analysis = _analysis(original_query=query)
    rec = _record(adjust=[
        ("answer_match", 0.02, "vocal_gender=남성", {"slot": "vocal_gender", "value": "남성"}),
        ("answer_match", 0.02, "vocal_gender=여성", {"slot": "vocal_gender", "value": "여성"}),
    ])
    answers = [ClarifyAnswer(slot="vocal_gender", value="남성"), ClarifyAnswer(slot="vocal_gender", value="여성")]
    row = _row(_build(query, analysis, _track(vocal_gender="여성"), rec, answers=answers), SLOT_GENDER)
    assert row.clue == "여성 보컬" and row.status == STATUS_MATCH
    assert [b.code for b in row.ranking_basis] == ["answer_match"]
    assert row.ranking_basis[0].delta == pytest.approx(0.02)


def test_genre_answer_from_clarification_is_compared_with_registered_genre():
    query = "분위기 좋은 노래"
    analysis = _analysis(original_query=query)
    rec = _record(adjust=[("answer_match", 0.02, "genre=발라드", {"slot": "genre", "value": "발라드"})])
    row = _row(_build(query, analysis, _track(genre=["발라드", "국내드라마"]), rec,
                      answers=[ClarifyAnswer(slot="genre", value="발라드")]), SLOT_GENRE)
    assert row.status == STATUS_MATCH and row.clue_origin == ORIGIN_ANSWER and row.basis_label == BASIS_SCORED


# ---------------------------------------------------------------------------
# 장르 — 인디/댄스는 미확인, 인디/인디음악은 일치
# ---------------------------------------------------------------------------

def test_indie_request_with_indie_registered_matches_and_links_genre_bonus():
    query = "남자 인디 노래"
    analysis = _analysis(original_query=query, genre="인디")
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "인디"})])
    row = _row(_build(query, analysis, _track(genre=["인디음악", "록/메탈"]), rec), SLOT_GENRE)
    assert row.status == STATUS_MATCH and "인디음악" in row.observed and row.basis_label == BASIS_SCORED


def test_genre_mismatch_is_never_reported_as_contradiction():
    query = "발라드인데 가사가 슬퍼"
    row = _row(_build(query, _analysis(original_query=query, genre="발라드"), _track(genre=["댄스"]), _record()), SLOT_GENRE)
    assert row.status == STATUS_UNVERIFIED and "발라드 여부" in row.observed


def test_genre_bonus_for_a_different_genre_is_not_linked_to_the_current_clue():
    query = "분위기 좋은 노래"
    analysis = _analysis(original_query=query, genre="발라드")
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "발라드"})])
    row = _row(_build(query, analysis, _track(genre=["발라드", "댄스"]), rec,
                      answers=[ClarifyAnswer(slot="genre", value="댄스")]), SLOT_GENRE)
    assert row.status == STATUS_MATCH and not row.ranking_basis


# ---------------------------------------------------------------------------
# 가사 — exact · 음차 · 인용 실패 · 보충 · 탈락 · 의미
# ---------------------------------------------------------------------------

def _lyric_clue(text, kind="verbatim", variants=()):
    return {"text": text, "kind": kind, "variants": list(variants), "confidence": 0.9, "source": "model"}


def _excerpt(song_id="s1", quote="나의 운명인 사랑", line="너는 나의 운명인 사랑이야"):
    return LyricExcerpt(song_id=song_id, source="mongo.full_lyrics", quote=quote, line=line,
                        char_start=3, char_end=3 + len(quote), data_version="abcd1234abcd1234",
                        matched_phrase="나의운명인사랑")


def test_exact_lyric_match_with_quote_scores_and_is_reported_as_reflected():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.12), ("text_hybrid", 3, 0.01)],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "나의운명인사랑",
                               "is_variant": False, "clue_kind": "verbatim"})
    ux = _build("q", analysis, _track(), rec, evidence=[_excerpt()])
    row = _row(ux, SLOT_LYRICS)
    assert row.status == STATUS_MATCH and "“나의 운명인 사랑”" in row.observed
    assert row.source.startswith("mongo.full_lyrics@abcd1234")
    assert row.basis_label == BASIS_SCORED and row.ranking_basis[0].code == "lyrics_surface"
    assert ux.summary.startswith("기억하신 구절이 가사에서 확인되어 검색 점수에 반영됐어요.")


def test_exact_match_without_quote_does_not_fabricate_a_quotation():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.12)],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "나의운명인사랑"})
    row = _row(_build("q", analysis, _track(), rec, evidence=[]), SLOT_LYRICS)
    assert row.status == STATUS_MATCH
    assert "“" not in row.observed and "인용은 확보하지 못함" in row.observed


def test_phonetic_match_is_partial_and_worded_as_an_estimated_transcription():
    analysis = _analysis(lyric_clues=[_lyric_clue("아이 파운드 더 웨이", kind="phonetic", variants=["I found the way"])])
    rec = _record(paths=[("lyrics_surface", 1, 0.08)],
                  lyric_match={"match_type": "phonetic", "clue_text": "아이 파운드 더 웨이",
                               "matched_phrase": "ifoundtheway", "is_variant": True})
    ux = _build("q", analysis, _track(), rec)
    row = _row(ux, SLOT_LYRICS)
    assert row.status == STATUS_PARTIAL and "추정" in row.observed and "ifoundtheway" in row.observed
    assert "비슷한 표기가 가사에서 확인되어 검색 점수에 반영" in ux.summary
    assert "기억하신 구절이 가사에서 확인되어" not in ux.summary


def test_order_rule_only_is_reported_as_protection_not_as_score():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.0), ("text_hybrid", 2, 0.01)],
                  order=[("lyrics_exact_priority", "일반 후보 아래로 내려가지 않음")],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("q", analysis, _track(), rec)
    row = _row(ux, SLOT_LYRICS)
    assert row.basis_label == BASIS_PROTECTED and [b.kind for b in row.ranking_basis] == ["order_rule"]
    assert "먼저 배치됐어요" in ux.summary and "검색 점수에 반영" not in ux.summary


def test_dropped_lyric_contribution_is_supplementary_not_a_ranking_reason():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = ExplainRecorder("q")
    rec.path("s1", "lyrics_surface", 1, 0.12)
    rec.mark_paths_dropped(["s1"])
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.note_lyric_match("s1", {"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("q", analysis, _track(), rec, evidence=[_excerpt()])
    row = _row(ux, SLOT_LYRICS)
    assert row.status == STATUS_MATCH and row.basis_label == BASIS_SUPPLEMENTARY and not row.ranking_basis
    assert "이 순위 계산에는 쓰이지 않았어요" in ux.summary


def test_zero_delta_lyric_path_is_supplementary():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.0), ("text_hybrid", 1, 0.016)],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    assert _row(_build("q", analysis, _track(), rec), SLOT_LYRICS).basis_label == BASIS_SUPPLEMENTARY


def test_highlight_adjustment_is_match_but_summary_adjustment_is_partial():
    clue = _lyric_clue("옥상 위로 올라가")
    analysis = _analysis(lyric_clues=[clue])
    hi = _record(adjust=[("lyric_clue_in_highlight", 0.03, "verbatim ×4.0", {"slot": "lyric_clue", "value": "옥상 위로 올라가"})])
    su = _record(adjust=[("lyric_clue_in_summary", 0.03, "verbatim ×4.0", {"slot": "lyric_clue", "value": "옥상 위로 올라가"})])
    hi_row = _row(_build("q", analysis, _track(), hi), SLOT_LYRICS)
    su_row = _row(_build("q", analysis, _track(), su), SLOT_LYRICS)
    assert hi_row.status == STATUS_MATCH and hi_row.source == "track.lyrics_highlight" and hi_row.basis_label == BASIS_SCORED
    assert su_row.status == STATUS_PARTIAL and "원문 확인 아님" in su_row.observed


def test_unverified_lyric_clue_says_not_confirmed_rather_than_absent():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    ux = _build("q", analysis, _track(), _record())
    row = _row(ux, SLOT_LYRICS)
    assert row.status == STATUS_UNVERIFIED and "확인하지 못함" in row.observed and "없" not in row.observed
    assert "기억하신 구절은 이 곡 가사에서 확인하지 못했어요" in ux.summary


def test_semantic_only_search_gives_a_limited_explanation_without_detection_claims():
    analysis = _analysis(lyric_clues=[_lyric_clue("비 오는 날 헤어지는 느낌", kind="semantic")])
    ux = _build("q", analysis, _track(), _record(paths=[("text_hybrid", 2, 0.015), ("audio", 5, 0.003)]))
    row = _row(ux, SLOT_LYRICS)
    assert row.status == STATUS_UNVERIFIED and "2위로 평가" in row.observed and row.basis_label == BASIS_SEMANTIC
    assert ux.summary == "텍스트 의미·키워드 검색이 비슷하게 평가한 후보예요(특정 구절이나 장면을 확인한 것은 아니에요)."
    for word in ("장면", "검출", "확인되어"):
        assert word not in ux.summary.split("특정")[0]


def test_no_clues_at_all_gives_a_limited_sentence_from_the_strongest_path():
    ux = _build("분위기 좋은 노래", _analysis(original_query="분위기 좋은 노래"), _track(), _record(paths=[("image", 1, 0.02)]))
    assert ux.clue_checks == []
    assert ux.summary == "앨범 이미지 검색이 비슷하게 평가한 후보예요(특정 구절이나 장면을 확인한 것은 아니에요)."


def test_other_recorded_adjustment_is_summarized_conservatively():
    ux = _build("가수 이름 노래", _analysis(original_query="가수 이름 노래", artist_name="가수"), _track(),
                _record(adjust=[("artist_match", 0.016, "")]))
    assert ux.summary == "‘가수 일치’ 기록이 검색 점수에 반영됐어요.", "사실 일치 규칙에는 '직접 확인한 것은 아니다'를 붙이지 않는다"


# ---------------------------------------------------------------------------
# 시기 — 출생 연도 창은 추정으로, 근접 가산은 범위 안 일치와 구분
# ---------------------------------------------------------------------------

def _life_stage_analysis(**kw):
    return _analysis(
        original_query="중학교 때 듣던 노래",
        life_stage={"stage": "middle", "text": "중학교 때", "spans": [[0, 5]], "age_from": 13, "age_to": 15, "confidence": 0.8},
        **kw,
    )


def test_birth_year_window_is_labelled_as_an_estimate_and_in_range_year_matches():
    analysis = _life_stage_analysis()
    search_analysis = with_birth_year(analysis, 1998)       # 2010~2014
    assert search_analysis is not analysis and search_analysis.has_release_era
    rec = _record(adjust=[("release_era", 0.003, "시기 유사 1.00", {"slot": "release_era", "value": "2010~2014"})])
    ux = _build(analysis.original_query, analysis, _track(release_date="2012.05.01"), rec, search_analysis=search_analysis)
    row = _row(ux, SLOT_ERA)
    assert row.clue_origin == ORIGIN_BIRTH_YEAR and "추정" in row.clue and "2010~2014" in row.clue and "중학교 때" in row.clue
    assert row.status == STATUS_MATCH and "추정 기간 안" in row.observed and row.basis_label == BASIS_SCORED
    assert ux.summary == "추정한 시기 단서는 곡 정보와 일치해 검색 점수에 반영됐어요.", "출생 연도 창은 요약에서도 추정이다"


def test_near_miss_year_is_partial_with_near_basis_not_full_match():
    analysis = _life_stage_analysis()
    search_analysis = with_birth_year(analysis, 1998)       # 2010~2014
    rec = _record(adjust=[("release_era", 0.002, "시기 유사 0.50", {"slot": "release_era", "value": "2010~2014"})])
    row = _row(_build(analysis.original_query, analysis, _track(release_date="2016.01.01"), rec,
                      search_analysis=search_analysis), SLOT_ERA)
    assert row.status == STATUS_PARTIAL and "2년 차이(근접)" in row.observed
    assert row.basis_label == BASIS_NEAR and row.ranking_basis


def test_far_year_is_mismatch_and_explicit_window_is_not_called_an_estimate():
    analysis = _analysis(original_query="2000년대 초반 노래", release_era={"start_year": 2000, "end_year": 2004, "confidence": 0.8})
    row = _row(_build(analysis.original_query, analysis, _track(release_date="2019.03.01"), _record()), SLOT_ERA)
    assert row.status == STATUS_MISMATCH and "추정" not in row.clue and row.clue == "2000~2004년 무렵 발매"
    assert "기억한 기간 밖 15년 차이" in row.observed


def test_life_stage_without_birth_year_is_unverified_and_distinguishes_listening_from_release():
    analysis = _life_stage_analysis()
    row = _row(_build(analysis.original_query, analysis, _track(release_date="2012.05.01"), _record()), SLOT_ERA)
    assert row.status == STATUS_UNVERIFIED and row.clue == "중학교 때에 듣던 노래" and "출생 연도" in row.observed


def test_missing_release_date_is_unverified():
    analysis = _analysis(original_query="2000년대 초반 노래", release_era={"start_year": 2000, "end_year": 2004, "confidence": 0.8})
    row = _row(_build(analysis.original_query, analysis, _track(release_date=None), _record()), SLOT_ERA)
    assert row.status == STATUS_UNVERIFIED and row.observed == "발매일 정보 없음"


# ---------------------------------------------------------------------------
# 불변 조건 — 입력을 바꾸지 않는다 · 응답 모델에 그대로 실린다 · 모델 문장은 쓰지 않는다
# ---------------------------------------------------------------------------

def test_builder_does_not_mutate_its_inputs():
    analysis = _analysis(original_query=UPSIDE_DOWN_QUERY, vocal_gender="남성", genre="인디")
    track = _track()
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"})])
    rec.note_model_reason("s1", ["뮤직비디오에 옥상에서 의자를 던지는 장면이 나온다"])
    answers = [ClarifyAnswer(slot="vocal_gender", value="남성")]
    before = (copy.deepcopy(analysis.model_dump()), copy.deepcopy(track.model_dump()),
              copy.deepcopy(rec.record.get("s1")), [a.model_dump() for a in answers])

    ux = _build(UPSIDE_DOWN_QUERY, analysis, track, rec, answers=answers, skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE)

    assert analysis.model_dump() == before[0] and track.model_dump() == before[1]
    assert rec.record.get("s1") == before[2] and [a.model_dump() for a in answers] == before[3]
    assert "의자를 던지는" not in ux.summary and all("의자를 던지는" not in c.observed for c in ux.clue_checks)


def test_track_explain_carries_the_user_explanation_and_keeps_the_numbers():
    analysis = _analysis(original_query=UPSIDE_DOWN_QUERY)
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"})])
    ux = _build(UPSIDE_DOWN_QUERY, analysis, _track(), rec)
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    dumped = out.model_dump()
    assert dumped["user_explanation"]["summary"] == ux.summary
    assert [c["status_label"] for c in dumped["user_explanation"]["clue_checks"]] == ["일치", "미확인", "미확인"]
    assert dumped["paths"] and dumped["adjustments"][0]["slot"] == "vocal_gender"
    assert dumped["model_notes"] == []
    assert to_track_explain(rec.record.get("s1")).user_explanation is None, "문맥 없는 호출자는 비운다"


# ---------------------------------------------------------------------------
# 회차 1 검토 지적 (UFE-01 ~ UFE-09, UFE-T2, UFE-T3)
# ---------------------------------------------------------------------------

def _sentences(text: str) -> int:
    return text.count(".")


def test_ufe01_summary_level_lyric_hits_are_not_called_confirmed_in_lyrics():
    """요약·경계 일치, 키워드 겹침은 "가사에서 확인되어"가 아니다. 요약(lyrics_summary)은 가사가 아니다."""
    clue = _lyric_clue("나의 운명인 사랑")
    analysis = _analysis(lyric_clues=[clue])
    expected = {
        "lyric_clue_in_summary": (KIND_LYRIC_SUMMARY, "기억하신 구절이 가사 요약 문구와 겹쳐 검색 점수에 반영됐어요."),
        "lyric_clue_in_combined": (KIND_LYRIC_COMBINED, "기억하신 구절이 가사 발췌·요약 문구와 겹쳐 검색 점수에 반영됐어요."),
        "lyric_phonetic_in_summary": (
            KIND_LYRIC_SUMMARY_PHONETIC, "기억하신 구절과 비슷한 표기가 가사 요약 문구와 겹쳐 검색 점수에 반영됐어요."),
    }
    for rule, (kind, sentence) in expected.items():
        rec = _record(adjust=[(rule, 0.03, "x", {"slot": "lyric_clue", "value": "나의 운명인 사랑"})])
        ux = _build("q", analysis, _track(), rec)
        assert "가사에서 확인" not in ux.summary and "발췌에서 확인" not in ux.summary, (rule, ux.summary)
        assert ux.summary == sentence, (rule, ux.summary)
        assert _row(ux, SLOT_LYRICS).detail_kind == kind and _row(ux, SLOT_LYRICS).status == STATUS_PARTIAL

    # 키워드 겹침만 — 구절 확인 문장의 근거가 아니다
    kw = _analysis(lyric_keywords=["사랑", "이별", "눈물"])
    ux = _build("q", kw, _track(), _record(adjust=[("lyric_keyword_tokens", 0.01, "겹침 1/3", {})]))
    assert "가사에서 확인" not in ux.summary and "구절" not in ux.summary
    assert ux.summary == "가사 키워드 일부가 발췌·요약과 겹쳐 검색 점수에 반영됐어요."

    # 구절은 미확인 + 키워드 겹침 — "확인하지 못했어요"가 남는다
    both = _analysis(lyric_clues=[clue], lyric_keywords=["사랑", "이별"])
    ux = _build("q", both, _track(), _record(adjust=[("lyric_keyword_tokens", 0.01, "겹침 1/2", {})]))
    assert "가사에서 확인되어" not in ux.summary and "가사에서 확인됐" not in ux.summary
    assert ux.summary == ("가사 키워드 일부가 발췌·요약과 겹쳐 검색 점수에 반영됐어요. "
                          "기억하신 구절은 이 곡 가사에서 확인하지 못했어요.")
    assert _row(ux, SLOT_LYRICS).status == STATUS_UNVERIFIED


def test_ufe01_fuzzy_is_worded_as_overlap_not_confirmation():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.05)],
                  lyric_match={"match_type": "fuzzy", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("q", analysis, _track(), rec)
    assert ux.summary == "기억하신 구절과 가사가 근사하게 겹쳐 검색 점수에 반영됐어요."


def test_ufe02_rule_value_differing_from_search_value_is_not_asserted_gender():
    """원문 규칙값(혼성)과 검색이 쓴 값(여성)이 다르면 일치·다름을 단정하지 않는다. 실제 가산은 검색 해석 기준으로만 연결."""
    cases = [
        ("남자 가수 이별 노래가 크게 뜨고 나서 그 노래 답가로 나온 여자 버전 노래 있잖아. 2017년 겨울쯤", "여성", "여성"),
        ("의사 다섯 명이 밴드 하는 드라마에서 배우가 기타 치면서 부른 노래. 원래는 2000년대 초 혼성 그룹 노래였대, 2020년에 다시 부른 걸로 찾고 싶어", "남성", "남성"),
    ]
    for query, used, actual in cases:
        analysis = _analysis(original_query=query, vocal_gender=used)
        rec = _record(adjust=[("vocal_gender_match", 0.004, actual, {"slot": "vocal_gender", "value": used})])
        ux = _build(query, analysis, _track(vocal_gender=actual), rec)
        row = _row(ux, SLOT_GENDER)
        assert row.status == STATUS_UNVERIFIED and row.detail_kind == KIND_SLOT_CONFLICT, (query, row)
        # 혼성은 사용자가 쓴 글자가 아니라 규칙이 읽은 값이다 — "원문 표현"이라고 부르지 않는다 (UFE-12 b)
        assert "원문에서 읽은 값(혼성)" in row.observed and f"검색이 해석한 값({used})" in row.observed
        assert "원문 표현" not in row.observed
        assert row.ranking_basis and row.basis_label == "검색 점수에 반영(검색이 해석한 값 기준)"
        assert "달라요" not in ux.summary and "일치해" not in ux.summary and "보컬 성별은 확인할 자료" not in ux.summary


def test_ufe02_rule_value_differing_from_search_value_is_not_asserted_genre():
    """원문의 어느 장르 진술도 검색이 쓴 값과 맞지 않으면 단정하지 않는다(dev 원문)."""
    query = "도입부가 EDM 처럼 신나는 남자 노랜데 가사가 슬퍼"
    analysis = _analysis(original_query=query, genre="랩/힙합")
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "랩/힙합"})])
    ux = _build(query, analysis, _track(genre=["랩/힙합"]), rec)
    row = _row(ux, SLOT_GENRE)
    assert row.status == STATUS_UNVERIFIED and row.detail_kind == KIND_SLOT_CONFLICT
    assert row.clue == "랩/힙합(검색이 해석한 값)" and "원문에서 읽은 값(일렉트로니카)" in row.observed
    assert row.ranking_basis and row.basis_label == "검색 점수에 반영(검색이 해석한 값 기준)"
    # 장르에 대해서는 일치·다름·자료 없음 어느 것도 말하지 않는다(성별 문장은 따로 나올 수 있다)
    assert "일렉트로니카" not in ux.summary and "힙합" not in ux.summary
    # 표가 단정하지 않은 줄의 가산은 상세에서도 무엇을 기준으로 했는지 적힌다 (UFE-12 a)
    assert [(n.rule, n.note) for n in ux.adjustment_notes] == [("genre_match", "검색이 해석한 값(랩/힙합) 기준")]


def test_ufe02_second_stated_genre_is_the_clue_when_search_used_it():
    """dev 원문: "일렉트로닉 사운드가 강한 힙합풍 곡" — 사용자는 장르를 둘 말했고 검색은 둘째(힙합)를 썼다.
    첫 값(일렉트로니카)만 보고 틀린 단서를 단정하지 않는다. 진술마다 한 줄이고, 실제 가산은 맞는 줄에만 연결된다."""
    query = "신디사이저랑 일렉트로닉 사운드가 강한 힙합풍 곡인데 남녀 목소리가 섞여 나와"
    analysis = _analysis(original_query=query, genre="랩/힙합", vocal_gender="혼성")
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "랩/힙합"})])
    ux = _build(query, analysis, _track(genre=["랩/힙합"], vocal_gender="혼성"), rec)
    rows = {c.clue: c for c in ux.clue_checks if c.slot == SLOT_GENRE}
    assert set(rows) == {"일렉트로니카", "힙합"}
    assert rows["힙합"].status == STATUS_MATCH and rows["힙합"].basis_label == BASIS_SCORED
    assert [b.code for b in rows["힙합"].ranking_basis] == ["genre_match"]
    assert rows["일렉트로니카"].status == STATUS_UNVERIFIED and not rows["일렉트로니카"].ranking_basis
    assert "일렉트로니카 여부를 확인할 자료는 없음" in rows["일렉트로니카"].observed
    assert all(c.detail_kind != KIND_SLOT_CONFLICT for c in ux.clue_checks)
    assert ux.adjustment_notes == [], "다른 줄에 연결된 가산에 '다른 값 기준' 표시를 붙이면 안 된다"
    assert ux.summary == ("힙합 단서는 곡 정보와 일치해 검색 점수에 반영됐고, 혼성 보컬 단서는 곡 정보와 일치해요. "
                          "일렉트로니카 여부는 확인할 자료가 없어요.")


def test_multiple_stated_genres_get_one_row_each_and_same_registered_genre_is_grouped():
    query = "R&B 발라드인데 여자 가수야"
    analysis = _analysis(original_query=query, genre="발라드", vocal_gender="여성")
    ux = _build(query, analysis, _track(genre=["발라드"], vocal_gender="여성"), _record())
    rows = {c.clue: c for c in ux.clue_checks if c.slot == SLOT_GENRE}
    assert {k: v.status for k, v in rows.items()} == {"알앤비": STATUS_UNVERIFIED, "발라드": STATUS_MATCH}
    assert all("원문에서 읽은 값" not in c.observed for c in rows.values()), "둘 다 사용자가 말한 장르다 — 충돌이 아니다"

    query = "록 메탈 느낌의 남자 밴드 노래"
    ux = _build(query, _analysis(original_query=query, genre="록"), _track(genre=["록/메탈"]),
                _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "록"})]))
    genre_rows = [c for c in ux.clue_checks if c.slot == SLOT_GENRE]
    assert [c.clue for c in genre_rows] == ["록·메탈"], "같은 등록 표기를 가리키는 표현은 한 줄이다"
    assert genre_rows[0].status == STATUS_MATCH and len(genre_rows[0].ranking_basis) == 1


def test_ufe02_rule_value_is_used_when_search_value_is_missing_or_agrees():
    query = "남자 인디 노래"
    fallback = _analysis(original_query=query)                       # 규칙 폴백 — 분석 값 없음
    assert _row(_build(query, fallback, _track(), _record()), SLOT_GENDER).clue_origin == ORIGIN_QUERY
    agree = _analysis(original_query=query, vocal_gender="남성", genre="인디음악")
    rows = _build(query, agree, _track(genre=["인디음악"]), _record()).clue_checks
    assert [r.status for r in rows[:2]] == [STATUS_MATCH, STATUS_MATCH]


def test_ufe03_song_released_before_the_estimated_window_is_not_a_contradiction():
    analysis = _life_stage_analysis()
    search_analysis = with_birth_year(analysis, 1998)       # 2010~2014
    earlier = _row(_build(analysis.original_query, analysis, _track(release_date="2005.01.01"), _record(),
                          search_analysis=search_analysis), SLOT_ERA)
    assert earlier.status == STATUS_UNVERIFIED and earlier.detail_kind == KIND_ERA_EARLIER_THAN_ESTIMATE
    assert "5년 먼저 발매" in earlier.observed and "들었을 수 있음" in earlier.observed
    ux = _build(analysis.original_query, analysis, _track(release_date="2005.01.01"), _record(), search_analysis=search_analysis)
    assert "달라요" not in ux.summary and "발매 시기" not in ux.summary and "추정한 시기는 확인할 자료" not in ux.summary

    later = _row(_build(analysis.original_query, analysis, _track(release_date="2021.01.01"), _record(),
                        search_analysis=search_analysis), SLOT_ERA)
    assert later.status == STATUS_MISMATCH, "들었던 시절보다 나중에 나온 곡만 모순이다"
    ux = _build(analysis.original_query, analysis, _track(release_date="2021.01.01"), _record(), search_analysis=search_analysis)
    assert ux.summary == "추정한 시기 단서는 등록 정보와 달라요."

    explicit = _analysis(original_query="2010년대 초 노래", release_era={"start_year": 2010, "end_year": 2014, "confidence": 0.8})
    assert _row(_build(explicit.original_query, explicit, _track(release_date="2005.01.01"), _record()), SLOT_ERA).status == STATUS_MISMATCH


def test_ufe04_mv_clue_cell_holds_only_the_scene_clause():
    cases = {
        "뮤비에 의자가 하늘에 떠있는 장면이 있어 남자 가수였고 2010년쯤 노래야": "뮤비에 의자가 하늘에 떠있는 장면이 있어",
        "뮤직비디오에서 의자를 던져, 누가 불렀는지는 모르겠어": "뮤직비디오에서 의자를 던져",
        "뮤비에서 교복 입고 춤추는 여자 아이돌 노래 찾아줘": "뮤비에서 교복 입고 춤추는",
    }
    for query, expected in cases.items():
        ux = _build(query, _analysis(original_query=query), _track(), _record())
        assert _row(ux, SLOT_MV).clue == expected, (query, _row(ux, SLOT_MV).clue)


def test_ufe06_fact_rules_do_not_get_the_not_directly_confirmed_hedge():
    ux = _build("q", _analysis(song_title="제목"), _track(), _record(adjust=[("title_exact", 0.08, "")]))
    assert ux.summary == "‘제목 완전일치’ 기록이 검색 점수에 반영됐어요."
    ux = _build("q", _analysis(), _track(), _record(adjust=[("performance_clues", 0.01, "단서 유사 0.50")]))
    assert ux.summary.endswith("(말씀하신 단서를 직접 확인한 것은 아니에요).")


def test_ufe07_missing_birth_year_is_not_reported_as_missing_song_data():
    analysis = _life_stage_analysis()
    ux = _build(analysis.original_query, analysis, _track(release_date="2012.05.01"), _record())
    assert "발매 시기는 확인할 자료가 없어요" not in ux.summary
    assert "출생 연도" in _row(ux, SLOT_ERA).observed


def test_ufe08_summary_never_exceeds_two_sentences():
    query = UPSIDE_DOWN_QUERY
    analysis = _analysis(original_query=query, vocal_gender="남성", genre="인디",
                         lyric_clues=[_lyric_clue("나의 운명인 사랑")], lyric_keywords=["사랑"])
    rec = _record(adjust=[("lyric_keyword_tokens", 0.01, "겹침 1/1", {}), ("title_exact", 0.08, "")])
    ux = _build(query, analysis, _track(vocal_gender="unknown"), rec, skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE)
    assert _sentences(ux.summary) <= 2, ux.summary
    assert ux.summary == (
        "가사 키워드 일부가 발췌·요약과 겹쳐 검색 점수에 반영됐어요. "
        "보컬 성별, 인디 여부, 말씀하신 뮤직비디오 장면은 확인할 자료가 없고, 기억하신 구절은 이 곡 가사에서 확인하지 못했어요."
    )
    # 긍정 절이 여럿이어도 한 문장으로 묶인다
    full = _analysis(original_query="남자 발라드", vocal_gender="남성", genre="발라드",
                     lyric_clues=[_lyric_clue("나의 운명인 사랑")],
                     release_era={"start_year": 2018, "end_year": 2020, "confidence": 0.8})
    rec = _record(paths=[("lyrics_surface", 1, 0.12)], adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"})],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("남자 발라드", full, _track(genre=["발라드"], release_date="2019.03.01"), rec, evidence=[_excerpt()])
    assert _sentences(ux.summary) == 1, ux.summary
    assert ux.summary == "기억하신 구절이 가사에서 확인되어 검색 점수에 반영됐고, 남성 보컬 단서는 곡 정보와 일치해 검색 점수에 반영됐고, 발라드·발매 시기 단서는 곡 정보와 일치해요."


def test_ufe09_stale_adjustment_rows_carry_a_server_note():
    query = "남자 가수 노래"
    analysis = _analysis(original_query=query, vocal_gender="남성")
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"}),
                          ("answer_match", 0.02, "vocal_gender=남성", {"slot": "vocal_gender", "value": "남성"}),
                          ("title_exact", 0.08, "")])
    ux = _build(query, analysis, _track(), rec, answers=[ClarifyAnswer(slot="vocal_gender", value="여성")])
    # 남성은 원문에서 온 값이라 "정정 전 값", 답변 가산은 "이전 답"이다 — 남성을 답한 적 없는 경우에 "답"이라고 하지 않는다
    assert {(n.rule, n.note) for n in ux.adjustment_notes} == {
        ("vocal_gender_match", "정정 전 값(남성) 기준"), ("answer_match", "이전 답(남성) 기준"),
    }
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    notes = {a.rule: a.note for a in out.adjustments}
    assert notes == {"vocal_gender_match": "정정 전 값(남성) 기준", "answer_match": "이전 답(남성) 기준", "title_exact": ""}
    assert [a.delta for a in out.adjustments] == [0.08, 0.02, 0.004], "점수·순서는 그대로"
    # 정정이 아니라 원문 규칙값과 검색 해석이 다른 경우의 문구
    q2 = "남자 가수 이별 노래가 크게 뜨고 나서 그 노래 답가로 나온 여자 버전 노래 있잖아"
    rec = _record(adjust=[("vocal_gender_match", 0.004, "여성", {"slot": "vocal_gender", "value": "여성"})])
    ux = _build(q2, _analysis(original_query=q2, vocal_gender="여성"), _track(vocal_gender="여성"), rec)
    assert [(n.rule, n.note) for n in ux.adjustment_notes] == [("vocal_gender_match", "검색이 해석한 값(여성) 기준")]
    assert to_track_explain(rec.record.get("s1")).adjustments[0].note == "", "설명 없는 호출자는 문구도 없다"


def test_ufet2_same_status_partial_bonus_computed_for_another_value_is_not_linked():
    """답 여성 · 곡 혼성 · 남성 기준 vocal_gender_partial — 상태는 같은 부분 일치지만 값이 다르므로 연결하지 않는다."""
    query = "분위기 좋은 노래"
    analysis = _analysis(original_query=query, vocal_gender="남성")
    rec = _record(adjust=[("vocal_gender_partial", 0.001, "혼성", {"slot": "vocal_gender", "value": "남성"})])
    row = _row(_build(query, analysis, _track(vocal_gender="혼성"), rec, answers=[ClarifyAnswer(slot="vocal_gender", value="여성")]), SLOT_GENDER)
    assert row.status == STATUS_PARTIAL and not row.ranking_basis and row.basis_label == BASIS_NONE


def test_ufet3_other_songs_evidence_and_skipped_answer_values_are_ignored():
    analysis = _analysis(original_query="남자 가수 노래", vocal_gender="남성", lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    rec = _record(paths=[("lyrics_surface", 1, 0.12)],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    foreign = _excerpt(song_id="other", quote="다른 곡의 문장", line="다른 곡의 문장")
    # ClarifyAnswer는 건너뛴 답의 값을 스스로 지운다 — 값이 남은 채 들어오는 경우를 보려면 검증을 거치지 않은 객체가 필요하다
    stale_skip = SimpleNamespace(slot="vocal_gender", value="여성", skipped=True)
    ux = _build("남자 가수 노래", analysis, _track(), rec, evidence=[foreign], answers=[stale_skip])
    lyric = _row(ux, SLOT_LYRICS)
    assert "다른 곡의 문장" not in lyric.observed and lyric.source == "explain.lyric_match"
    assert _row(ux, SLOT_GENDER).clue == "남성 보컬", "값이 든 건너뛴 답은 정정이 아니다"


# ---------------------------------------------------------------------------
# 회차 2 추가 — 음차·경계 문구, 가장 강한 가사 근거, 문장 수 상한, MV 표시 구간과 정책 판정의 동치
# ---------------------------------------------------------------------------

def test_phonetic_clue_in_highlight_is_a_similar_spelling_not_the_remembered_phrase():
    clue = _lyric_clue("아이 파운드 더 웨이", kind="phonetic")
    rec = _record(adjust=[("lyric_phonetic_in_highlight", 0.02, "phonetic ×2.5", {"slot": "lyric_clue", "value": "아이 파운드 더 웨이"})])
    ux = _build("q", _analysis(lyric_clues=[clue]), _track(), rec)
    row = _row(ux, SLOT_LYRICS)
    assert row.detail_kind == KIND_LYRIC_HIGHLIGHT_PHONETIC and row.status == STATUS_PARTIAL
    assert ux.summary == "기억하신 구절과 비슷한 표기가 가사 발췌에서 확인되어 검색 점수에 반영됐어요."


def test_summary_speaks_of_the_strongest_lyric_evidence_not_the_first_row():
    weak, strong = _lyric_clue("옥상 위로"), _lyric_clue("나의 운명인 사랑")
    analysis = _analysis(lyric_clues=[weak, strong])
    rec = _record(paths=[("lyrics_surface", 1, 0.12)],
                  adjust=[("lyric_clue_in_summary", 0.03, "x", {"slot": "lyric_clue", "value": "옥상 위로"})],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("q", analysis, _track(), rec, evidence=[_excerpt()])
    assert [c.detail_kind for c in ux.clue_checks] == ["lyric_summary", "lyric_exact"], "표에는 둘 다 남는다"
    assert ux.summary == "기억하신 구절이 가사에서 확인되어 검색 점수에 반영됐어요."


def test_lyric_path_without_a_recorded_phrase_claims_neither_confirmed_nor_unconfirmed():
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑")])
    ux = _build("q", analysis, _track(), _record(paths=[("lyrics_surface", 1, 0.12)]))
    row = _row(ux, SLOT_LYRICS)
    assert row.detail_kind == KIND_LYRIC_UNATTRIBUTED and row.status == STATUS_UNVERIFIED
    assert ux.summary == "가사 구절 일치 기록이 검색 점수에 반영됐지만 어느 구절인지는 기록에 없어요."
    # 단서 행이 없을 때의 제한 문장도 유사도라고 하지 않는다
    bare = _build("q", _analysis(), _track(), _record(paths=[("lyrics_surface", 1, 0.12)]))
    assert bare.summary == "가사 구절 일치 기록이 검색 점수에 반영된 후보예요(어느 구절인지는 기록에 없어요)."


def test_limited_sentence_picks_the_right_particle_for_the_path_name():
    ux = _build("q", _analysis(), _track(), _record(paths=[("audio", 1, 0.02)]))
    assert ux.summary.startswith("소리 검색이 비슷하게 평가한 후보예요")
    ux = _build("q", _analysis(), _track(), _record(paths=[("unknown_path", 1, 0.02)]))
    assert ux.summary.startswith("unknown_path이(가) 비슷하게 평가한 후보예요")


def test_summary_is_at_most_two_sentences_across_the_clue_matrix():
    """요약 계약: 최대 2문장. 성별·장르·가사·시기·MV 조합 전부에서 고정한다."""
    life = _life_stage_analysis()
    genders = [("남자 가수", "남성"), ("남자 가수", "여성"), ("남자 가수", "혼성"), ("남자 가수", "unknown"), ("", "남성")]
    genres = [("인디", ["댄스"]), ("발라드", ["발라드"]), ("", ["댄스"])]
    lyrics = [
        dict(),
        dict(clues=[_lyric_clue("나의 운명인 사랑")]),
        dict(clues=[_lyric_clue("나의 운명인 사랑")], match="exact", paths=[("lyrics_surface", 1, 0.1)]),
        dict(clues=[_lyric_clue("나의 운명인 사랑")], adjust="lyric_clue_in_summary"),
        dict(clues=[_lyric_clue("나의 운명인 사랑")], keywords=["사랑"], adjust="lyric_keyword_tokens"),
        dict(clues=[_lyric_clue("비 오는 느낌", kind="semantic")]),
    ]
    eras = [None, 2012, 2016, 2005, 2021, "none"]
    mvs = ["", " 뮤비에 의자가 하늘에 떠있는 장면이 있어"]
    seen = 0
    for (g_text, g_actual), (genre_text, genre_reg), lyric, era, mv in itertools.product(genders, genres, lyrics, eras, mvs):
        query = " ".join(x for x in (g_text, genre_text, "노래") if x) + mv
        kw = dict(original_query=query, lyric_clues=lyric.get("clues", []), lyric_keywords=lyric.get("keywords", []))
        if era is not None:
            kw["life_stage"] = life.life_stage.model_dump()
        analysis = _analysis(**kw)
        search_analysis = with_birth_year(analysis, 1998) if era not in (None, "none") else analysis
        adjust = []
        if lyric.get("adjust") == "lyric_keyword_tokens":
            adjust.append(("lyric_keyword_tokens", 0.01, "겹침 1/1", {}))
        elif lyric.get("adjust"):
            adjust.append((lyric["adjust"], 0.03, "x", {"slot": "lyric_clue", "value": "나의 운명인 사랑"}))
        rec = _record(paths=lyric.get("paths", [("text_hybrid", 1, 0.016)]), adjust=adjust,
                      lyric_match=({"match_type": lyric["match"], "clue_text": "나의 운명인 사랑", "matched_phrase": "x"}
                                   if lyric.get("match") else None))
        release = None if era in (None, "none") else f"{era}.01.01"
        ux = _build(query, analysis, _track(vocal_gender=g_actual, genre=genre_reg, release_date=release), rec,
                    search_analysis=search_analysis, skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE if mv else "")
        assert 1 <= _sentences(ux.summary) <= 2, (query, ux.summary)
        assert ux.summary.endswith("."), ux.summary
        for banned in ("순위가 올라", "1위", "결정적", "가장 큰 이유"):
            assert banned not in ux.summary, (banned, ux.summary)
        seen += 1
    assert seen == len(genders) * len(genres) * len(lyrics) * len(eras) * len(mvs)


def test_mv_display_phrases_exist_exactly_when_the_policy_judges_a_scene_query():
    """표시용 구간은 판정을 바꾸지 않는다 — 비어 있지 않음 ⇔ is_specific_mv_scene_query."""
    from src.retrieval.rerank_policy import is_specific_mv_scene_query, mv_scene_phrases, mv_scene_spans
    import tests.test_rerank_policy as policy_tests

    targets = list(policy_tests.TARGETS)
    non_targets = [item if isinstance(item, str) else item[0] for item in policy_tests.NON_TARGETS]
    bases = targets + non_targets + ["남자 발라드", "분위기 좋은 노래"]
    queries = list(bases)
    queries += [lead + q for lead in policy_tests.LEAD_INS for q in bases]                    # 말머리
    queries += [tail(q) for tail in policy_tests.TRAILING_UNCERTAINTIES for q in bases]        # 뒤에 붙는 불확실
    queries += [mention(q) for mention in policy_tests.SCENELESS_MENTIONS for q in bases]      # 장면 없는 뮤비 언급
    queries += [f"{q} 남자 가수였고 2010년쯤 노래야" for q in targets]                           # 뒤에 다른 단서
    assert len(queries) > 300 and sum(map(is_specific_mv_scene_query, queries)) > 50
    from src.retrieval.rerank_policy import _describes_scene
    for query in queries:
        phrases, spans = mv_scene_phrases(query), mv_scene_spans(query)
        assert bool(phrases) == bool(spans) == is_specific_mv_scene_query(query), query
        assert len(phrases) <= len(spans), query
        normalized = " ".join(_strip(query).split())
        for phrase in phrases:
            assert phrase in normalized, (query, phrase)                       # 사용자가 쓴 말 그대로다
            assert _describes_scene(phrase), (query, phrase)                   # 자른 뒤에도 장면이다
            assert not _DANGLING_END.search(phrase), (query, phrase)           # 지시어·성별·가수·노래로 끝나지 않는다
            assert not any(phrase != other and phrase in other for other in phrases), (query, phrases)


def _strip(query: str) -> str:
    from src.retrieval.rerank_policy import _strip_quoted
    return _strip_quoted(query)


import re as _re

# 단서 칸이 이렇게 끝나면 장면이 아닌 말이 매달린 것이다.
_DANGLING_END = _re.compile(
    r"(?:^|\s)(?:그런|이런|저런|그|이|저|남성|여성|혼성|아이돌|가수|그룹|밴드|솔로|래퍼|듀오|듀엣|보컬|걸그룹|보이그룹|"
    r"발라드|노래\S{0,3}|곡\S{0,3})$"
)


def test_genre_spelling_differences_are_not_treated_as_a_conflict():
    """dev 원문: 분석이 "일렉트로닉"·"록발라드"라고 쓰고 원문 규칙이 "일렉트로니카"·"록"으로 읽는 것은 같은 장르다."""
    cases = [
        ("샘플링이랑 비트가 강조된 일렉트로닉 곡이고 남자들이 여러 명 부르는 느낌이었어", "일렉트로닉", ["일렉트로니카"], "일렉트로니카"),
        ("부드러운 남자 목소리로 시작해서 후렴에 노래 대신 신스 멜로디가 터지는 EDM", "EDM", ["일렉트로니카"], "일렉트로니카"),
        ("군대 가 있는 동안 역주행해서 차트 1위까지 한 남자 솔로 노래, 마지막에 터지듯 내지르는 록발라드야", "록발라드", ["록/메탈"], "록"),
        # 다른 표기가 복합 장르 안에 든 경우 — 표기 표를 읽지 않으면 "발라드"만 읽혀 록과 충돌로 보인다
        ("군대 가 있는 동안 역주행한 남자 솔로 노래, 마지막에 터지듯 내지르는 락발라드야", "락발라드", ["록/메탈"], "록"),
    ]
    for query, used, registered, clue in cases:
        ux = _build(query, _analysis(original_query=query, genre=used), _track(genre=registered), _record())
        row = _row(ux, SLOT_GENRE)
        assert row.clue == clue and row.status == STATUS_MATCH and row.detail_kind != KIND_SLOT_CONFLICT, (query, row)
        assert "원문에서 읽은 값" not in row.observed

    # 등록 장르가 발라드뿐이면 록은 미확인이다 — 충돌도 모순도 아니다
    query = cases[2][0]
    row = _row(_build(query, _analysis(original_query=query, genre="록발라드"), _track(genre=["발라드"]), _record()), SLOT_GENRE)
    assert row.status == STATUS_UNVERIFIED and row.detail_kind == "" and "록 여부를 확인할 자료는 없음" in row.observed


def test_genre_bonus_recorded_in_corpus_notation_links_to_the_user_wording():
    query = "남자 힙합 노래"
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "랩/힙합"})])
    ux = _build(query, _analysis(original_query=query, genre="랩/힙합"), _track(genre=["랩/힙합"]), rec)
    row = _row(ux, SLOT_GENRE)
    assert row.clue == "힙합" and row.status == STATUS_MATCH and [b.code for b in row.ranking_basis] == ["genre_match"]
    assert ux.adjustment_notes == []


# ---------------------------------------------------------------------------
# 회차 2 검토 지적 (UFE-11 ~ UFE-13, UFE-T4)
# ---------------------------------------------------------------------------

# UFE-11 — 리뷰 기록의 "원문 → 기대" 표와, 같은 종류의 말끝.
MV_PHRASE_TABLE = {
    "뮤비에서 의자가 하늘에 떠 있는 그 노래 뭐지": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비가 흑백이고 비가 계속 내려 그 노래 뭐야": ["뮤비가 흑백이고 비가 계속 내려"],
    "뮤비 마지막에 바다에서 춤추는 그런 노래": ["뮤비 마지막에 바다에서 춤추는"],
    "뮤직비디오에서 교복 입고 춤추는 그 곡 찾아줘": ["뮤직비디오에서 교복 입고 춤추는"],
    "뮤비에서 우산 쓰고 걷는 여자 가수 노래야": ["뮤비에서 우산 쓰고 걷는"],
    "뮤비 마지막에 교실이 나오는 남자 솔로 노래였어": ["뮤비 마지막에 교실이 나오는"],
    "뮤비에서 남자가 여자한테 곡 알려주는 장면이 있어": ["뮤비에서 남자가 여자한테 곡 알려주는 장면이 있어"],
    "뮤비에서 가수가 노래 요청을 받는 장면": ["뮤비에서 가수가 노래 요청을 받는 장면"],
    "뮤비에서 의자가 하늘에 떠 있었고 뮤직비디오 마지막에 바다가 나와": [
        "뮤비에서 의자가 하늘에 떠 있었고", "뮤직비디오 마지막에 바다가 나와"],
    # 회차 1(UFE-04)에서 고정한 문장
    "뮤비에 의자가 하늘에 떠있는 장면이 있어 남자 가수였고 2010년쯤 노래야": ["뮤비에 의자가 하늘에 떠있는 장면이 있어"],
    "뮤직비디오에서 의자를 던져, 누가 불렀는지는 모르겠어": ["뮤직비디오에서 의자를 던져"],
    "뮤비에서 교복 입고 춤추는 여자 아이돌 노래 찾아줘": ["뮤비에서 교복 입고 춤추는"],
    # 같은 종류의 말끝 — 장르 낱말, 절 경계로 안 잡히는 반말 종결, 지시어 뒤의 감상, 구간 끝의 가수
    "뮤비에서 의자가 하늘에 떠 있는 발라드 노래": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 교복 입고 춤췄어 근데 가수를 몰라": ["뮤비에서 교복 입고 춤췄어"],
    "뮤비에서 의자가 하늘에 떠 있는 그 노래 듣고 울었어": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 의자가 하늘에 떠 있었어 그 노래 듣고 울었어": ["뮤비에서 의자가 하늘에 떠 있었어"],
    "뮤비에서 의자가 하늘에 떠 있는 노래를 찾고 있어": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 우산 쓰고 걷는 여자 가수였어": ["뮤비에서 우산 쓰고 걷는"],
    "뮤비 마지막에 교실이 진짜 크게 나와 그 노래 뭐지": ["뮤비 마지막에 교실이 진짜 크게 나와"],
    # 떼면 안 되는 것 — 화면 속 사람·노래하는 행위, 표지 앞 관형절
    "뮤비에서 춤추는 사람은 여자": ["뮤비에서 춤추는 사람은 여자"],
    "뮤비에서 남자가 노래 부르면서 걷는 장면이 있어": ["뮤비에서 남자가 노래 부르면서 걷는 장면이 있어"],
    "뮤비에서 남자가 비 맞으면서 노래 부르고 있어": ["뮤비에서 남자가 비 맞으면서 노래 부르고 있어"],
    "뮤비에서 남자가 비 맞으면서 노래 부르고 있어 그 노래 뭐지": ["뮤비에서 남자가 비 맞으면서 노래 부르고 있어"],
    "교실이 나오는 뮤비였고 남자 발라드야": ["교실이 나오는 뮤비"],
    "뮤비를 봤는데 의자가 하늘에 떠 있었어": ["뮤비를 봤는데 의자가 하늘에 떠 있었어"],
    "뮤비 배경이 사막이야": ["뮤비 배경이 사막이야"],
    # 뒤에 '장면'이 오면 그 노래는 장면 안의 노래다 — 앞 절만 남기면 장면의 절반을 잃는다
    "뮤비에서 여자가 울고 그 노래 부르는 장면이 있어": ["뮤비에서 여자가 울고 그 노래 부르는 장면이 있어"],
    # 화면 속에서 부르는 노래는 꾸미는 말이 붙어도 장면이다("슬픈"에서 끊기면 안 된다)
    "뮤비에서 남자가 비 맞으면서 슬픈 노래 부르고 있어": ["뮤비에서 남자가 비 맞으면서 슬픈 노래 부르고 있어"],
    # 관형절이 꾸미는 곡 뒤에 감상이 이어져도 장면까지만
    "뮤비에서 우산 쓰고 걷는 여자 가수 노래 진짜 많이 울었어": ["뮤비에서 우산 쓰고 걷는"],
    # 줄이면 장면이 아니게 되는 문장은 줄이지 않는다("뮤비에서 걷는"만으로는 장면이 아니다)
    "뮤비에서 걷는 남자 가수 곡 끝날 때 차가 폭발해": ["뮤비에서 걷는 남자 가수 곡 끝날 때 차가 폭발해"],
    # 표지가 둘 — 앞 표지의 장면이 뒤 표지 쪽에 있으면 뒤 표지의 구간만 보인다
    "뮤비를 봤는데 뮤직비디오 마지막에 바다가 나와": ["뮤직비디오 마지막에 바다가 나와"],
    # 표지가 둘인데 뒤 표지는 부정됐다 — 줄일 수 없으면 판정 구간 그대로(비우지 않는다)
    "뮤비 보면 뮤직비디오는 기억 안 나는데 의자가 떠 있었어": ["뮤비 보면 뮤직비디오는 기억 안 나는데 의자가 떠 있었어"],
    # UFE-14 — 장면 뒤에 **다른 단서의 절**이 와도 그 절의 명사가 칸에 남지 않는다(리뷰 기록의 표)
    "뮤비에 의자가 하늘에 떠있는 장면이 있는데 남자가 불렀어": ["뮤비에 의자가 하늘에 떠있는 장면이 있는데"],
    "뮤비에서 의자가 하늘에 떠 있어 여자 목소리였어": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에 의자가 하늘에 떠있는 장면이 있어 중학교 때 듣던 노래야": ["뮤비에 의자가 하늘에 떠있는 장면이 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 바다가 생각나는 노래야": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 의자가 하늘에 떠 있는 남자 보컬 곡": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 의자가 하늘에 떠 있어 아이들이 좋아하던 노래야": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 교복 입고 춤추는데 여자 다섯 명이 불렀어": ["뮤비에서 교복 입고 춤추는데"],
    "뮤비에서 교실이 나오는 2015년쯤 나온 노래": ["뮤비에서 교실이 나오는"],
    "뮤비에서 여자가 옥상에서 뛰어내려 여름에 나온 노래야": ["뮤비에서 여자가 옥상에서 뛰어내려"],
    "뮤비에서 의자가 하늘에 떠 있어, 여자가 불렀어": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비 배경이 사막인 여자 목소리 노래": ["뮤비 배경이 사막인"],
    "뮤비 배경이 사막인 아이들이 좋아하던 노래": ["뮤비 배경이 사막인"],
    # UFE-14 — 그대로여야 하는 것: 명사가 서술된 장면(구조어 · 명사 서술 · 배경)
    "뮤비에서 교복 입고 춤추는 여자들이 나와": ["뮤비에서 교복 입고 춤추는 여자들이 나와"],
    "뮤비에서 춤추는 사람은 여자였어": ["뮤비에서 춤추는 사람은 여자였어"],
    # 회차 4에서 "원문 그대로"로 넣었던 항목. 관형형("눈 내리는")으로 닫힌 뒤의 "바다였어"는 혼자 장면이 아니므로
    # 회차 4 검토의 성질(닫힌 장면 뒤의 말은 혼자 장면일 때만 잇는다)에 따라 앞부분까지만 보인다.
    "뮤비 배경은 눈 내리는 바다였어": ["뮤비 배경은 눈 내리는"],
    "뮤비 마지막 장면은 바다였어": ["뮤비 마지막 장면은 바다였어"],
    "뮤비 마지막에 교실이 진짜 크게 나와": ["뮤비 마지막에 교실이 진짜 크게 나와"],
    # 뒤 절의 동사가 서술어 목록에 없으면 앞 절까지만(명사만 남기지 않는다)
    "뮤비에서 의자가 하늘에 떠 있고 남자가 그 밑에서 노래해": ["뮤비에서 의자가 하늘에 떠 있고"],
    # 가수 절을 사이에 두고 장면이 또 나와도, 장면이 아닌 절을 칸에 넣지 않는다 — 첫 장면까지만
    "뮤비에서 의자가 떠 있어 남자가 불렀는데 마지막에 바다가 나와": ["뮤비에서 의자가 떠 있어"],
    # 구조어는 이미 서술된 명사를 건너뛰어 붙지 않는다("뛰어내려"를 넘어 옥상에 "있어"를 붙이지 않는다)
    "뮤비에서 여자가 옥상에서 뛰어내려 유튜브에 있어": ["뮤비에서 여자가 옥상에서 뛰어내려"],
    # UFE-15 (a) 구조어·보조 용언은 어간에 어미가 붙은 낱말일 때만 — "보이그룹"·"배경음악"·"있잖아"는 아니다
    "뮤비에서 멤버들이 지하철에서 춤추는 보이그룹 노래였어": ["뮤비에서 멤버들이 지하철에서 춤추는"],
    "뮤비에서 의자가 하늘에 떠 있는 배경음악 같은 노래": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 교복 입고 춤춰 있잖아 그 노래": ["뮤비에서 교복 입고 춤춰"],
    "뮤비에서 교복 입고 춤춰 있어 보이는 노래야": ["뮤비에서 교복 입고 춤춰"],
    # UFE-15 (b) 닫힌 장면 뒤에서 가수를 말한 명사 서술은 장면이 아니다
    "뮤비에 의자가 하늘에 떠있는 장면이 있어 부른 사람은 여자야": ["뮤비에 의자가 하늘에 떠있는 장면이 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 멤버들이 남자 다섯 명이야": ["뮤비에서 의자가 하늘에 떠 있어"],
    # UFE-15 (c) 닫힌 장면 뒤의 서술어만 있는 절은 듣는 상황·분위기다
    "뮤비에서 의자가 하늘에 떠 있어 울고 싶을 때 들었어": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 춤추기 좋은 노래야": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 날아갈 것 같은 기분이 드는 노래야": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 슬퍼서 울었어": ["뮤비에서 의자가 하늘에 떠 있어"],
    "뮤비에서 의자가 하늘에 떠 있어 신나게 뛰어놀기 좋아": ["뮤비에서 의자가 하늘에 떠 있어"],
    # UFE-15 그대로여야 하는 것 — 관형형 뒤라도 이어지는 말이 혼자 장면이면 잇는다 · 열린 서술은 끝까지
    "뮤비에 눈 오는 거리에서 혼자 서 있는 여자가 나오는 2000년대 노래": ["뮤비에 눈 오는 거리에서 혼자 서 있는 여자가 나오는"],
    "뮤비가 흑백이고 비가 계속 내려": ["뮤비가 흑백이고 비가 계속 내려"],
    "MV에 사람이 거꾸로 매달린 장면": ["MV에 사람이 거꾸로 매달린 장면"],
    "뮤비에서 의자가 공중에 떠 있어 여자가 불렀어": ["뮤비에서 의자가 공중에 떠 있어"],
    "뮤비에서 남자가 피아노 앞에 앉아 있어 그 노래 뭐지": ["뮤비에서 남자가 피아노 앞에 앉아 있어"],
    "뮤비에 로봇이 나오고 남자가 그걸 부숴": ["뮤비에 로봇이 나오고 남자가 그걸 부숴"],
    "뮤비에서 편의점 앞에 앉아서 컵라면 먹어": ["뮤비에서 편의점 앞에 앉아서 컵라면 먹어"],
    # 서술어 목록에 없는 마지막 동사는 떨어진다(한계) — 뒤의 말을 잇지 않으려면 모르는 낱말에서 멈춰야 한다
    "뮤비에서 주인공이 기차 타고 떠나 춤추기 좋은 노래야": ["뮤비에서 주인공이 기차 타고"],
    # 곡을 가리키는 말 뒤에 장면이 또 와도, 그 곡을 꾸민 장면까지만(지시어 · 관형형 뒤의 곡)
    "뮤비에서 의자가 하늘에 떠 있는 그 노래 마지막에 바다가 나와": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 우산 쓰고 걷는 여자 가수 노래 마지막에 바다가 나와": ["뮤비에서 우산 쓰고 걷는"],
    # 나열 뒤의 "그 노래"는 화면 속에서 부르는 노래일 수 있다 — 서술이 이어지면 장면이다
    "뮤비에서 여자가 울고 그 노래 불러": ["뮤비에서 여자가 울고 그 노래 불러"],
    "뮤비에서 교복 입고 그 노래 듣고 울었어": ["뮤비에서 교복 입고"],
    # 닫힌 장면 뒤의 "그 노래"는 뒤에 무엇이 와도 찾는 곡이다 — 관형형 뒤든 그 밖의 연결어미 뒤든
    "뮤비에서 여자가 물에 빠지는데 그 노래 마지막에 바다가 나와": ["뮤비에서 여자가 물에 빠지는데"],
    "뮤비에서 의자가 하늘에 떠 있는 그 노래 나오면 사람들이 춤춰": ["뮤비에서 의자가 하늘에 떠 있는"],
    "뮤비에서 교복 입고 춤추는 그 노래 안무 장면이 유명해": ["뮤비에서 교복 입고 춤추는"],
    # 관형형 뒤의 노래라도 화면 속에서 부르는 노래면 장면이다
    "뮤비에서 우산 쓰고 걷는 노래 부르는 여자가 나와": ["뮤비에서 우산 쓰고 걷는 노래 부르는 여자가 나와"],
    # 낱말의 앞부분만 구조어·보조 용언과 같은 말 — "있잖아", "보이그룹", "배경음악"
    "뮤비에서 남자가 피아노 앞에 앉아 있잖아 그 노래": ["뮤비에서 남자가 피아노 앞에 앉아"],
    "뮤비에서 여자가 울고 아이들이 보이그룹 노래를 따라 해": ["뮤비에서 여자가 울고"],
    "뮤비에서 여자가 울고 바다가 배경음악처럼 깔려": ["뮤비에서 여자가 울고"],
    # 주어 조사가 없는 명사 뒤의 명사 서술은 장면의 끝이 아니다
    "뮤비에서 여자가 울고 바다에서 남자였어": ["뮤비에서 여자가 울고"],
    # 배경 뒤의 가장 가까운 명사까지 — 그 뒤 명사("아이들이")는 다른 말이다
    "뮤비가 흑백이고 배경이 사막이야": ["뮤비가 흑백이고 배경이 사막이야"],
    "뮤비가 흑백이고 배경이 사막인 아이들이 좋아하던 노래": ["뮤비가 흑백이고 배경이 사막인"],
    # 판정은 장면으로 봤는데(판정 구간이 "목소리" 앞에서 잘려 여자가 화면 속 사람으로 읽힌다) 줄일 수 없다 — 판정 구간 그대로
    "뮤비에서 춤추는 여자 목소리 노래": ["뮤비에서 춤추는 여자"],
    # 장면 질의가 아니면 비어 있다
    "뮤비가 유명한 노래": [],
    "뮤비는 안 찾아봤는데 의자가 떠 있었어": [],
}


def test_ufe11_mv_clue_phrases_match_the_review_table():
    from src.retrieval.rerank_policy import _describes_scene, is_specific_mv_scene_query, mv_scene_phrases
    for query, expected in MV_PHRASE_TABLE.items():
        phrases = mv_scene_phrases(query)
        assert phrases == expected, (query, phrases)
        assert bool(phrases) == is_specific_mv_scene_query(query), query
        assert all(_describes_scene(phrase) for phrase in phrases), (query, phrases)
    # 설명의 단서 칸에 그대로 실린다
    query = "뮤직비디오에서 교복 입고 춤추는 그 곡 찾아줘"
    ux = _build(query, _analysis(original_query=query), _track(), _record(), skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE)
    assert _row(ux, SLOT_MV).clue == "뮤직비디오에서 교복 입고 춤추는"
    two = "뮤비에서 의자가 하늘에 떠 있었고 뮤직비디오 마지막에 바다가 나와"
    ux = _build(two, _analysis(original_query=two), _track(), _record())
    assert _row(ux, SLOT_MV).clue == "뮤비에서 의자가 하늘에 떠 있었고 / 뮤직비디오 마지막에 바다가 나와"


def test_ufe11_mv_clue_phrase_is_exactly_the_scene_across_natural_endings():
    """표지 × 장면(관형형·종결형) × 말끝 × 말머리 — 단서 칸은 언제나 "표지 + 장면"이다."""
    from src.retrieval.rerank_policy import is_specific_mv_scene_query, mv_scene_phrases
    markers = ["뮤비", "뮤직비디오"]
    scenes = [  # (곡을 꾸미는 관형형, 문장으로 끝나는 종결형)
        ("에서 의자가 하늘에 떠 있는", "에서 의자가 하늘에 떠 있었어"),
        ("에서 교복 입고 춤추는", "에서 교복 입고 춤췄어"),
        ("에서 우산 쓰고 걷는", "에서 우산 쓰고 걸어"),
        (" 마지막에 교실이 나오는", " 마지막에 교실이 나와"),
        (" 마지막에 바다에서 춤추는", " 마지막에 바다에서 춤춰"),
        ("가 흑백이고 비가 계속 내리는", "가 흑백이고 비가 계속 내려"),
        (" 배경이 사막인", " 배경이 사막이야"),
        ("에 의자가 하늘에 떠있는 장면이 있는", "에 의자가 하늘에 떠있는 장면이 있어"),
        ("에서 사람들이 옥상에서 뛰어내리는", "에서 사람들이 옥상에서 뛰어내려"),
        ("에서 남자가 비 맞으면서 노래 부르는", "에서 남자가 비 맞으면서 노래 불러"),
        ("를 봤는데 의자가 하늘에 떠 있는", "를 봤는데 의자가 하늘에 떠 있었어"),
    ]
    after_attributive = [
        " 그 노래 뭐지", " 그 노래 뭐야", " 그런 노래", " 이 노래 찾아줘", " 그 곡 찾아줘", " 노래", " 노래야", " 노래 찾아줘",
        " 곡이야", " 여자 가수 노래야", " 남자 솔로 노래였어", " 여자 아이돌 노래 찾아줘", " 걸그룹 노래", " 혼성 그룹 곡이야",
        " 그 남자 가수 노래", " 남자 아이돌 곡 알려줘", " 노래를 찾고 있어", " 노래인데 제목이 뭐지", " 그 노래 듣고 울었어",
        " 노래 제목 알려줘", " 저 노래 뭔지 알아", " 발라드 노래", " 남자 가수 곡",
        # 다른 단서 — 성별 주어 · 목소리/보컬 · 생애 시기 · 발매 시점 (UFE-14)
        " 남자가 부른 노래", " 여자 목소리 노래", " 남자 보컬 곡", " 여자 음색 곡", " 중학교 때 듣던 노래",
        " 2015년쯤 나온 노래", " 여름에 나온 노래", " 아이들이 좋아하던 노래",
    ]
    after_final = [
        "", " 그 노래 뭐지", " 그 노래 뭐야", ", 누가 불렀는지는 모르겠어", " 남자 가수였고 2010년쯤 노래야", " 근데 가수를 몰라",
        " 여자 가수 노래야", " 노래 찾아줘", " 그 노래 듣고 울었어", ", 언제 나왔는지 기억이 안 나", " 이 노래 제목 알려줘",
        " 그런 노래 찾아줘", " 남자 아이돌 곡이야", " 누가 불렀지", " 색감이 예뻤어", " 진짜 좋아했어", " 2010년쯤이야",
        " 남자 발라드야", " 유튜브에서 봤어",
        # 다른 단서 — 성별 주어 · 목소리/보컬 · 생애 시기 · 발매 시점 (UFE-14)
        " 남자가 불렀어", " 여자가 부른 노래야", ", 여자가 불렀어", " 여자 다섯 명이 불렀어", " 남자 둘이 불렀어",
        " 여자 목소리였어", " 남자 보컬이었어", " 여자 음색이 좋았어", " 중학교 때 듣던 노래야", " 고등학교 때 유행했어",
        " 2015년쯤 나온 노래야", " 여름에 나온 노래야", " 작년에 나왔어", " 바다가 생각나는 노래야", " 아이들이 좋아하던 노래야",
    ]
    leads = ["", "제목은 기억 안 나는데 ", "남자 인디 노래인데 ", "가수는 모르겠고 "]
    checked = 0
    for marker, (attributive, final), lead in itertools.product(markers, scenes, leads):
        for scene, tails in ((attributive, after_attributive), (final, after_final)):
            for tail in tails:
                query = lead + marker + scene + tail
                assert is_specific_mv_scene_query(query), query
                assert mv_scene_phrases(query) == [" ".join((marker + scene).split())], (query, mv_scene_phrases(query))
                checked += 1
    assert checked == len(markers) * len(scenes) * len(leads) * (len(after_attributive) + len(after_final))


def test_ufe12_conflict_row_wording_and_detail_notes():
    query = "남자 가수 이별 노래가 크게 뜨고 나서 그 노래 답가로 나온 여자 버전 노래 있잖아. 2017년 겨울쯤"
    analysis = _analysis(original_query=query, vocal_gender="여성")
    # (a) 표가 단정하지 않은 줄의 가산 — 상세 줄과 기록 요약 문장에 기준이 적힌다
    rec = _record(adjust=[("vocal_gender_match", 0.0164, "여성", {"slot": "vocal_gender", "value": "여성"})])
    ux = _build(query, analysis, _track(vocal_gender="여성"), rec)
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    assert out.adjustments[0].note == "검색이 해석한 값(여성) 기준"
    assert "말한 성별과 일치, 여성 · 검색이 해석한 값(여성) 기준 (+0.0164)" in out.summary
    row = _row(ux, SLOT_GENDER)
    assert row.basis_label == "검색 점수에 반영(검색이 해석한 값 기준)"
    # (c) 연결된 기록이 감점이면 감점이라고 쓴다
    rec = _record(adjust=[("vocal_gender_mismatch", -0.0164, "남성", {"slot": "vocal_gender", "value": "여성"})])
    ux = _build(query, analysis, _track(vocal_gender="남성"), rec)
    row = _row(ux, SLOT_GENDER)
    assert row.status == STATUS_UNVERIFIED and row.basis_label == "검색 점수에서 감점(검색이 해석한 값 기준)"
    assert "반영" not in row.basis_label
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    assert out.adjustments[0].note == "검색이 해석한 값(여성) 기준"


def test_ufe13_search_value_of_unknown_family_is_not_a_conflict():
    """검색값의 계열을 모르면("OST"·"K-POP") 충돌이 아니다 — 사용자가 말한 장르를 등록 장르와 대조한다."""
    cases = [
        ("드라마 OST였던 남자 발라드", "OST", ["발라드", "국내드라마"], "발라드"),
        ("걸그룹 댄스곡", "K-POP", ["댄스"], "댄스"),
    ]
    for query, used, registered, clue in cases:
        ux = _build(query, _analysis(original_query=query, genre=used), _track(genre=registered), _record())
        row = _row(ux, SLOT_GENRE)
        assert row.clue == clue and row.status == STATUS_MATCH and row.clue_origin == ORIGIN_QUERY, (query, row)
        assert "원문에서 읽은 값" not in row.observed and row.detail_kind != KIND_SLOT_CONFLICT
        assert f"{clue} 단서는 곡 정보와 일치" in ux.summary, ux.summary   # 성별 절이 뒤따를 수 있다

    # "랩"은 등록 표기(랩/힙합)의 한쪽 낱말이다 — 힙합 계열로 읽어 실제 가산을 사용자 단서에 연결한다
    query = "힙합 노래"
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "랩"})])
    ux = _build(query, _analysis(original_query=query, genre="랩"), _track(genre=["랩/힙합"]), rec)
    row = _row(ux, SLOT_GENRE)
    assert row.clue == "힙합" and row.status == STATUS_MATCH and row.basis_label == BASIS_SCORED
    assert ux.adjustment_notes == []

    # 계열 밖 검색값으로 계산된 가산은 사용자 단서에 연결하지 않고, 상세 줄에 기준을 적는다
    query = "드라마 OST였던 남자 발라드"
    rec = _record(adjust=[("genre_match", 0.004, "", {"slot": "genre", "value": "OST"})])
    ux = _build(query, _analysis(original_query=query, genre="OST"), _track(genre="발라드, OST"), rec)
    row = _row(ux, SLOT_GENRE)
    assert row.clue == "발라드" and row.status == STATUS_MATCH and not row.ranking_basis and row.basis_label == BASIS_NONE
    assert [(n.rule, n.note) for n in ux.adjustment_notes] == [("genre_match", "검색이 해석한 값(OST) 기준")]
    assert "검색 점수에 반영" not in ux.summary

    # 아는 장르끼리 다르면 여전히 충돌이다(UFE-02)
    query = "도입부가 EDM 처럼 신나는 남자 노랜데 가사가 슬퍼"
    ux = _build(query, _analysis(original_query=query, genre="댄스"), _track(genre=["댄스"]), _record())
    assert _row(ux, SLOT_GENRE).detail_kind == KIND_SLOT_CONFLICT


def test_ufet4_confirmed_and_unconfirmed_phrases_do_not_contradict_in_the_summary():
    """R13 — 구절 하나는 확인, 하나는 미확인이면 요약은 확인된 것만 말한다. 미확인은 표에 남는다."""
    analysis = _analysis(lyric_clues=[_lyric_clue("나의 운명인 사랑"), _lyric_clue("다시는 볼 수 없어")])
    rec = _record(paths=[("lyrics_surface", 1, 0.12)],
                  lyric_match={"match_type": "exact", "clue_text": "나의 운명인 사랑", "matched_phrase": "x"})
    ux = _build("q", analysis, _track(), rec, evidence=[_excerpt()])
    assert ux.summary == "기억하신 구절이 가사에서 확인되어 검색 점수에 반영됐어요."
    assert "확인하지 못했어요" not in ux.summary
    assert [c.status for c in ux.clue_checks if c.slot == SLOT_LYRICS] == [STATUS_MATCH, STATUS_UNVERIFIED]


def test_ufet4_no_note_without_a_recorded_value_or_without_a_current_clue():
    # R20 — 비교값이 없는 옛 기록에는 "() 기준"을 붙이지 않는다(정정이 있어도)
    query = "남자 가수 노래"
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성")])
    ux = _build(query, _analysis(original_query=query, vocal_gender="남성"), _track(), rec,
                answers=[ClarifyAnswer(slot="vocal_gender", value="여성")])
    assert ux.adjustment_notes == []
    assert to_track_explain(rec.record.get("s1"), user_explanation=ux).adjustments[0].note == ""

    # R21 — 단서 행이 없는(분석만 성별·장르를 채운) 곡의 가산에는 표시가 없다
    query = "비 오는 날 듣기 좋은 노래"
    rec = _record(adjust=[("vocal_gender_match", 0.004, "남성", {"slot": "vocal_gender", "value": "남성"}),
                          ("genre_match", 0.004, "", {"slot": "genre", "value": "발라드"})])
    ux = _build(query, _analysis(original_query=query, vocal_gender="남성", genre="발라드"), _track(genre=["발라드"]), rec)
    assert [c.slot for c in ux.clue_checks] == [], "사용자가 말하지 않은 성별·장르는 표에 없다"
    assert ux.adjustment_notes == []
    out = to_track_explain(rec.record.get("s1"), user_explanation=ux)
    assert all(a.note == "" for a in out.adjustments) and "기준" not in out.summary


def test_ufet4_era_bonus_computed_for_another_window_is_not_linked():
    """R22 — 다른 창으로 계산된 시기 가산은 지금 창의 근거가 아니다."""
    analysis = _life_stage_analysis()
    search_analysis = with_birth_year(analysis, 1998)       # 2010~2014
    other = _record(adjust=[("release_era", 0.003, "시기 유사 1.00", {"slot": "release_era", "value": "2004~2008"})])
    row = _row(_build(analysis.original_query, analysis, _track(release_date="2012.05.01"), other,
                      search_analysis=search_analysis), SLOT_ERA)
    assert row.status == STATUS_MATCH and not row.ranking_basis and row.basis_label == BASIS_NONE
    same = _record(adjust=[("release_era", 0.003, "시기 유사 1.00", {"slot": "release_era", "value": "2010~2014"})])
    row = _row(_build(analysis.original_query, analysis, _track(release_date="2012.05.01"), same,
                      search_analysis=search_analysis), SLOT_ERA)
    assert row.basis_label == BASIS_SCORED


def test_ufet4_a_genre_stated_twice_is_one_clue():
    """R46 — "발라드 … 발라드"가 「발라드·발라드」 한 줄이 되면 안 된다."""
    from src.retrieval.known_slots import known_genre_values
    query = "슬픈 발라드인데 남자 가수가 부른 발라드야"
    assert known_genre_values(query) == ["발라드"]
    ux = _build(query, _analysis(original_query=query, genre="발라드"), _track(genre=["발라드"]), _record())
    assert [c.clue for c in ux.clue_checks if c.slot == SLOT_GENRE] == ["발라드"]
    assert "발라드·발라드" not in ux.summary


# ---------------------------------------------------------------------------
# 회차 3 검토 지적 (UFE-14) — 장면 뒤에 다른 단서의 절이 와도 MV 칸은 장면까지다
# ---------------------------------------------------------------------------

def test_ufe14_other_clue_clauses_after_the_scene_do_not_leak_into_the_mv_cell():
    """리뷰어의 조합(장면 17 × 다른 단서 말끝 73 × 표지·여는 말 4 = 2,560건)을 그대로 고정한다.

    말끝은 곡을 가리키는 말이 아니라 **다른 단서** — 가수 성별·목소리, 시기, 장르, 그룹 형태, 요청·감상이다.
    그 절의 명사("남자가", "중학교", "바다가")가 서술 없이 장면의 끝에 남으면 사용자가 장면으로 말하지 않은 것이
    "말씀하신 뮤직비디오 장면"에 들어간다.
    """
    from src.retrieval.rerank_policy import is_specific_mv_scene_query, mv_scene_phrases
    final_scenes = [
        "의자가 하늘에 떠 있어", "교복 입고 춤춰", "남자가 의자를 던지는 장면이 나와", "여자가 옥상에서 뛰어내려",
        "흑백이고 비가 계속 내려", "남자가 피아노 앞에 앉아 있어", "자동차가 폭발하는 장면이 있어", "교실이 나와",
        "여자가 바다에서 춤췄어", "아이들이 풍선 들고 뛰어다녀",
    ]
    attributive_scenes = [
        "의자가 하늘에 떠 있는", "교복 입고 춤추는", "교실이 나오는", "우산 쓰고 걷는", "여자가 바다에서 춤추는",
        "아이들이 풍선 들고 뛰어다니는", "남자가 의자를 던지는",
    ]
    after_final = [
        " 여자 목소리였어", " 남자 목소리야", " 여자 보컬이었어", " 남자 음색이 좋았어", " 여자가 불렀어", " 남자가 부른 노래야",
        " 여자 가수가 불렀어", " 2015년쯤 나온 노래야", " 2010년대 초반 노래야", " 옛날 노래야", " 중학교 때 듣던 노래야",
        " 여름에 나온 노래야", " 작년에 나왔어", " 신나는 댄스곡이야", " 잔잔한 발라드였어", " 록 느낌이었어", " 힙합이었던 것 같아",
        " 인디 밴드 노래야", " 5인조 그룹이야", " 혼성 그룹이었어", " 솔로 가수였어", " 아이돌 그룹 노래 같았어", " 걸그룹이었어",
        " 남자 아이돌이었어", " 가사에 사랑이 나와", " 후렴이 중독적이야", " 제목이 영어였어", " 제목은 모르겠어", " 찾아줘",
        " 뭔지 알려줘", " 이게 뭐였지", " 알 수 있을까", " 뭐였더라", " 유튜브에서 봤어", " 조회수가 높았어", " 되게 유명했어",
        " 드라마 OST였어", " 그 노래 제목이 뭐야", " 그 가수 이름이 뭐지", " 그때 엄청 유행했어", " 노래는 슬펐어", " 멜로디가 좋았어",
        " 분위기가 몽환적이었어",
    ]
    after_attributive = [
        " 노래", " 곡", " 그 노래", " 노래 찾아줘", " 여자 목소리 노래", " 남자 보컬 곡", " 2015년쯤 나온 노래", " 2010년대 노래",
        " 옛날 노래", " 신나는 댄스곡", " 잔잔한 발라드", " 록 느낌 노래", " 인디 밴드 노래", " 5인조 그룹 노래", " 혼성 그룹 노래",
        " 솔로 가수 노래", " 아이돌 노래", " 걸그룹 곡", " 드라마 OST", " 유명한 노래", " 슬픈 노래", " 그 유명한 노래",
        " 노래 뭐였더라", " 한국 노래", " 여자 솔로 발라드", " 남자 그룹 댄스곡", " 그 여름 노래", " 내가 찾는 노래", " 요즘 노래",
        " 일본 노래 아니고 한국 노래",
    ]
    checked = 0
    for marker, opener in itertools.product(["뮤비", "뮤직비디오"], ["에서 ", " 마지막에 "]):
        for scenes, tails in ((final_scenes, after_final), (attributive_scenes, after_attributive)):
            for scene, tail in itertools.product(scenes, tails):
                query = f"{marker}{opener}{scene}{tail}"
                assert is_specific_mv_scene_query(query), query
                assert mv_scene_phrases(query) == [" ".join(f"{marker}{opener}{scene}".split())], (query, mv_scene_phrases(query))
                checked += 1
    assert checked == 2560


def test_ufe14_the_mv_cell_in_the_explanation_carries_only_the_scene():
    """같은 표에 「남성 보컬 | 일치」가 따로 있는데 MV 칸에 "남자가"가 붙으면 안 된다."""
    query = "뮤비에 의자가 하늘에 떠있는 장면이 있는데 남자가 불렀어"
    ux = _build(query, _analysis(original_query=query, vocal_gender="남성"), _track(), _record(),
                skip=RERANK_SKIP_MV_SCENE_UNVERIFIABLE)
    assert _row(ux, SLOT_MV).clue == "뮤비에 의자가 하늘에 떠있는 장면이 있는데"
    assert _row(ux, SLOT_GENDER).clue == "남성 보컬" and _row(ux, SLOT_GENDER).status == STATUS_MATCH


# ---------------------------------------------------------------------------
# 회차 4 검토 지적 (UFE-15) — 승인 기준은 말끝 목록이 아니라 성질이다
# ---------------------------------------------------------------------------

# 종결어미로 닫힌 장면 / 관형형으로 닫힌 장면(곡을 꾸민다)
CLOSED_SCENES_FINAL = [
    "의자가 하늘에 떠 있어", "교복 입고 춤춰", "남자가 의자를 던지는 장면이 나와", "여자가 옥상에서 뛰어내려", "흑백이고 비가 계속 내려",
    "남자가 피아노 앞에 앉아 있어", "자동차가 폭발하는 장면이 있어", "교실이 나와", "여자가 바다에서 춤췄어", "아이들이 풍선 들고 뛰어다녀",
    "우산 쓰고 걸어", "배경이 사막이야", "의자가 하늘에 떠있는 장면이 있어", "사람들이 옥상에서 뛰어내려", "남자가 비 맞으면서 노래 불러",
    "사람이 거꾸로 매달려 있어", "의자만 덩그러니 있어", "고양이가 계속 따라다녀", "주인공이 기차 타고 떠나", "여자가 전화기 들고 울어",
    "춤추는 사람은 여자였어", "온통 노란 꽃밭이 나와", "의자가 공중에 떠 있었어", "남자가 달리고 여자가 울어",
]
CLOSED_SCENES_ATTRIBUTIVE = [
    "의자가 하늘에 떠 있는", "교복 입고 춤추는", "교실이 나오는", "우산 쓰고 걷는", "여자가 바다에서 춤추는", "아이들이 풍선 들고 뛰어다니는",
    "남자가 의자를 던지는", "멤버들이 지하철에서 춤추는", "배경이 사막인", "사람들이 옥상에서 뛰어내리는", "남자가 비 맞으면서 노래 부르는",
    "사람이 거꾸로 매달린", "눈 오는 놀이공원이 배경인", "창고 안에 불이 타오르는", "여자가 물속에 가라앉는",
]
# 그 밖의 연결어미(-는데 · -지만 · -아서 · -다가 · -더니 · -니까)로 끝난 장면 — 나열(-고 · -며 · -면서)과 달리 닫힌 것으로 본다.
# "-는데"를 나열로 읽으면 "교복 입고 춤추는데 | 울고 싶을 때 들었어"의 울고가 칸에 들어온다(UFE-T5).
CLOSED_SCENES_CONNECTIVE = [
    "교복 입고 춤추는데", "여자가 옥상에 올라가는데", "의자를 던지지만", "남자가 피아노 앞에 앉아서", "여자가 바다에서 춤추다가",
    "의자가 하늘에 떠 있더니", "교실이 나오는데", "아이들이 풍선 들고 뛰어다니니까", "의자가 하늘에 떠 있는데", "남자가 비를 맞고 있지만",
    "사람들이 옥상에서 뛰어내리는데", "여자가 전화기 들고 우는데",
]
# 회차 4 검토에서 새로 쓴 말끝(리뷰어의 108종) — 성질을 보는 재료로만 쓴다(기대값을 박아 두지 않는다)
REVIEW_TAILS_R4 = [
    " 남자 둘이 불렀어", " 여자 혼자 불렀어", " 남자 다섯 명이 부른 노래야", " 멤버들이 남자 다섯 명이야", " 가수는 여자였어", " 보컬이 여자야",
    " 부른 사람은 여자야", " 부른 사람이 남자였어", " 남자 그룹이 불렀어", " 여자 솔로가 불렀어", " 혼성 그룹이야", " 보이그룹이야", " 걸그룹이 불렀어",
    " 보이그룹 노래였어", " 보이그룹 곡이야", " 아이돌이 불렀어", " 밴드 노래였어", " 가수가 엄청 유명해", " 고등학교 때 유행했어",
    " 초등학교 다닐 때 들었어", " 대학교 축제에서 들었어", " 군대에서 들었어", " 버스에서 자주 들었어", " 학교에서 자주 나왔어", " 학교에서 자주 나와",
    " 카페에서 자주 나와", " 노래방에서 불렀어", " 길거리에서 많이 나왔어", " 라디오에서 나왔어", " TV에서 자주 나왔어", " 드라마에도 나왔어",
    " 예능에 나온 노래야", " 광고에 나왔어", " 영화에 나온 노래야", " 2000년대에 나왔어", " 10년 전에 나온 노래야", " 겨울에 자주 들었어",
    " 작년 여름에 나온 노래야", " 비 오는 날 듣기 좋아", " 눈 오는 날 생각나", " 밤에 듣기 좋은 노래야", " 바다 갈 때 듣던 노래야",
    " 하늘 보면서 듣던 노래야", " 별 같은 노래야", " 사람들이 다 아는 노래야", " 아이들도 따라 불렀어", " 울고 싶을 때 들었어", " 웃긴 노래였어",
    " 춤이 유명했어", " 안무가 유명해", " 춤추기 좋은 노래야", " 달리기할 때 듣던 노래야", " 걸으면서 듣기 좋아", " 차에서 자주 들었어",
    " 날아갈 것 같은 기분이 드는 노래야", " 신나게 뛰어놀기 좋아", " 슬퍼서 울었어", " 들으면 눈물 나", " 기타 소리가 좋았어", " 피아노 반주로 시작해",
    " 드럼 소리가 커", " 일렉 기타가 나와", " 랩이 나와", " 영어 가사가 나와", " 고음이 나와", " 남자 랩이 나와", " 여자 내레이션이 나와",
    " 휘파람 소리가 나와", " 아이들 합창이 나와", " 배경음악이 좋았어", " 있잖아 그 노래", " 이거 뭔지 알아", " 혹시 아는 사람", " 찾아줄 수 있어",
    " 맞나 모르겠어", " 그런 거 있었는데", " 뭐였는지 궁금해", " 보이그룹 곡이었어", " 보이그룹 노래인데 제목을 몰라", " 걸그룹 곡이었어",
    " 남자 그룹 곡인데", " 여자 솔로 노래였어", " 밴드 노래였는데", " 아이돌 곡이었던 것 같아", " 혼성 그룹 노래였어", " 힙합 곡이었어", " 발라드 곡인데",
    " 노래였는데 남자가 불렀어", " 곡인데 여자 목소리야", " 노래 중에 제일 유명한 거", " 노래 말고 다른 노래", " 그거 있잖아", " 거 뭐였지", " 게 기억나",
    " 걸 봤어", " 영상 본 적 있어", " 고등학교 때 노래", " 학교 다닐 때 유행한 노래", " 비 오는 날 듣던 노래", " 울고 싶을 때 듣던 노래",
    " 작년에 나온 노래", " 드라마에 나온 노래", " 예능에서 본 노래", " 배경음악 같은 노래", " 사람들이 다 아는 노래", " 남자 다섯 명이 부른 노래",
    " 여자가 부른 노래", " 남자 목소리가 좋은 노래",
]


def _generated_tails():
    """틀로 만든 말끝 — 장면 뒤에 흔히 이어 말하는 다른 단서(가수 · 듣던 때와 곳 · 분위기 · 요청)."""
    subjects = ["남자가", "여자가", "가수가", "멤버들이", "아이들이", "사람들이", "친구가", "남자 둘이", "여자 혼자", "보이그룹이", "걸그룹이",
                "밴드가", "남자 가수가", "여자 아이돌이", "부른 사람은", "보컬이", "멤버는"]
    verbs = ["불렀어", "부른 노래야", "좋아했어", "추천해 줬어", "자주 틀었어", "알려 줬어", "유명해", "누군지 모르겠어", "여자야", "남자였어",
             "다섯 명이야"]
    contexts = ["울고 싶을 때", "비 올 때", "운동할 때", "학교 갈 때", "밤에", "여름에", "고등학교 때", "군대에서", "카페에서", "차에서",
                "노래방에서", "달리기할 때", "걸으면서", "춤추면서", "바다 갈 때", "눈 올 때"]
    listens = ["들었어", "듣던 노래야", "자주 나왔어", "유행했어", "틀어 줬어", "듣기 좋아", "생각나"]
    moods = ["신나는", "슬픈", "잔잔한", "춤추기 좋은", "따라 부르기 쉬운", "날아갈 것 같은", "배경음악 같은", "달리고 싶어지는", "울고 싶어지는",
             "보이그룹", "걸그룹", "남자 아이돌", "여자 솔로", "혼성 그룹", "인디 밴드", "2015년쯤 나온", "작년에 나온", "드라마에 나온"]
    songs = ["노래야", "곡이야", "노래였어", "노래", "곡", "느낌이었어", "분위기였어", "노래인데 제목을 몰라"]
    asks = ["있잖아 그 노래", "찾아줘", "뭐였지", "이거 뭔지 알아", "혹시 아는 사람", "그 노래 뭐지", "그런 거 있었는데", "제목이 뭐야", "알려줘",
            "보이그룹이야", "배경음악이 좋았어", "찍먹 같은 노래야", "있어 보이는 노래야", "나온 지 오래됐어", "보여주고 싶은 노래야"]
    tails = [f" {a} {b}" for a, b in itertools.product(subjects, verbs)]
    tails += [f" {a} {b}" for a, b in itertools.product(contexts, listens)]
    tails += [f" {a} {b}" for a, b in itertools.product(moods, songs)]
    tails += [f" {a}" for a in asks]
    return tails + [", " + tail.strip() for tail in tails[::7]]     # 쉼표로 이은 꼴


def test_ufe15_property_words_after_a_closed_scene_never_enter_the_mv_cell_unless_they_are_a_scene():
    """성질(회차 4 검토의 승인 기준): 장면 S가 종결어미나 관형형으로 닫힌 뒤의 말 T가 판정 어휘로 장면이 아니면
    (`_describes_scene(T)` 거짓) 칸에 T의 글자가 하나도 들어가지 않는다 — 칸은 "표지 + S"이거나 장면이 성립하는 그 앞부분이다.
    그 밖의 연결어미(-는데 · -지만 · -아서 …)로 끝난 장면도 닫힌 것이라 같은 성질을 지킨다(회차 5 검토, UFE-T5).

    기대값을 말끝마다 박아 두지 않는다. 장면 51종(종결 24 · 관형형 15 · 그 밖의 연결어미 12) × 말끝(틀로 만든 것 +
    리뷰어의 것) × 표지·여는 말 4.
    """
    from src.retrieval.rerank_policy import _describes_scene, is_specific_mv_scene_query, mv_scene_phrases
    tails = list(dict.fromkeys(_generated_tails() + REVIEW_TAILS_R4))
    assert len(tails) > 550
    checked = exact = scene_tails = 0
    shortened = set()
    for marker, opener in itertools.product(["뮤비", "뮤직비디오"], ["에서 ", " 마지막에 "]):
        scenes = CLOSED_SCENES_FINAL + CLOSED_SCENES_ATTRIBUTIVE + CLOSED_SCENES_CONNECTIVE
        for scene, tail in itertools.product(scenes, tails):
            query = f"{marker}{opener}{scene}{tail}"
            if not is_specific_mv_scene_query(query):
                continue
            phrases = mv_scene_phrases(query)
            scene_text = " ".join(f"{marker}{opener}{scene}".split())
            if _describes_scene(tail):
                # 뒤의 말이 판정 어휘로 장면이다("학교에서 자주 나와") — 성질의 대상이 아니다(알려진 한계). 원문의 일부인지만 본다
                scene_tails += 1
                assert all(phrase in " ".join(query.split()) and _describes_scene(phrase) for phrase in phrases), (query, phrases)
                continue
            checked += 1
            assert len(phrases) == 1, (query, phrases)
            assert scene_text.startswith(phrases[0]), (query, phrases)          # T의 글자가 하나도 없다
            assert _describes_scene(phrases[0]), (query, phrases)               # 줄어도 장면이다
            if phrases[0] == scene_text:
                exact += 1
            else:
                shortened.add(scene)
    assert checked > 110_000 and scene_tails > 800
    # 앞부분으로 줄어드는 장면은 서술어 목록에 없는 마지막 동사로 끝나는 것뿐이다(알려진 한계: "기차 타고 떠나" → "기차 타고")
    assert shortened == {"주인공이 기차 타고 떠나"}, shortened
    assert exact / checked > 0.97


def test_ufe15_closed_scene_continues_only_into_words_that_are_a_scene_by_themselves():
    """관형형 뒤라도 이어지는 말이 혼자 장면이면 잇고, 종결어미 뒤에는 장면이 또 와도 잇지 않는다(첫 장면까지만)."""
    from src.retrieval.rerank_policy import mv_scene_phrases
    assert mv_scene_phrases("뮤비에 눈 오는 거리에서 혼자 서 있는 여자가 나오는 2000년대 노래") == [
        "뮤비에 눈 오는 거리에서 혼자 서 있는 여자가 나오는"]
    assert mv_scene_phrases("뮤비에서 여자가 울어 남자가 달려가") == ["뮤비에서 여자가 울어"]
    # 판정이 화제 전환어 앞에서 자른 조각만 보면 장면처럼 읽히는 경우 — 자르기 전 문장으로도 본다
    assert mv_scene_phrases("뮤비에서 의자가 하늘에 떠 있는 춤추는 여자 목소리 노래") == ["뮤비에서 의자가 하늘에 떠 있는"]
    # 앞부분만 보면 화면 속 사람으로 읽히는 "남자"(뒤에 가수가 온다)에서 끊지 않는다
    assert mv_scene_phrases("뮤비에서 걷는 남자 가수 곡 끝날 때 차가 폭발해") == ["뮤비에서 걷는 남자 가수 곡 끝날 때 차가 폭발해"]


# ---------------------------------------------------------------------------
# 회차 5 검토 지적 (UFE-T5) — 동작은 맞고, 그 동작을 지키는 테스트가 없던 두 갈래
# ---------------------------------------------------------------------------

def test_ufet5_other_connectives_close_the_scene_unlike_listing_connectives():
    """"-는데"는 나열 연결어미가 아니다 — 뒤의 서술어만 있는 말을 잇지 않는다. "-고"는 잇는다."""
    from src.retrieval.rerank_policy import mv_scene_phrases
    closed = {
        "뮤비에서 교복 입고 춤추는데 울고 싶을 때 들었어": ["뮤비에서 교복 입고 춤추는데"],
        "뮤비에서 여자가 옥상에 올라가는데 춤추기 좋은 노래야": ["뮤비에서 여자가 옥상에 올라가는데"],
        "뮤비에서 의자를 던지지만 달리기할 때 듣던 노래야": ["뮤비에서 의자를 던지지만"],
        "뮤비에서 남자가 피아노 앞에 앉아서 울고 싶어지는 노래야": ["뮤비에서 남자가 피아노 앞에 앉아서"],
        "뮤비에서 여자가 바다에서 춤추다가 걸으면서 듣기 좋아": ["뮤비에서 여자가 바다에서 춤추다가"],
    }
    for query, expected in closed.items():
        assert mv_scene_phrases(query) == expected, (query, mv_scene_phrases(query))
    # 나열 연결어미로 열려 있으면 다음 서술을 잇는다 — 같은 말이 "-고" 뒤에서는 장면이다
    assert mv_scene_phrases("뮤비에서 교복 입고 춤추고 울어") == ["뮤비에서 교복 입고 춤추고 울어"]
    # 그 밖의 연결어미 뒤라도 이어지는 말이 혼자 장면이면 잇는다
    assert mv_scene_phrases("뮤비에서 편의점 앞에 앉아서 컵라면 먹어") == ["뮤비에서 편의점 앞에 앉아서 컵라면 먹어"]
    # 나열 뒤의 다음 서술은 가까이 있어야 한다 — 사이에 낄 수 있는 말은 셋까지다
    assert mv_scene_phrases("뮤비에서 여자가 울고 계속 막 진짜 춤춰") == ["뮤비에서 여자가 울고 계속 막 진짜 춤춰"]
    assert mv_scene_phrases("뮤비에서 여자가 울고 계속 막 진짜 엄청 춤춰") == ["뮤비에서 여자가 울고"]


def test_ufet5_singer_nominal_after_an_open_scene_does_not_enter_the_mv_cell():
    """나열 연결어미로 열린 장면 뒤에 가수를 말한 명사 서술("가수는 여자", "보컬은 남자")은 칸에 들어가지 않는다.

    명사 서술이 장면의 끝이 되려면 바로 앞 어절이 **화면 속 명사**여야 한다("춤추는 사람은 여자였어"). "가수" · "보컬"은
    화면 속 명사가 아니다 — 앞 어절이 주어 조사로 끝난다는 것만으로는 부족하다.
    """
    from src.retrieval.rerank_policy import is_specific_mv_scene_query, mv_scene_phrases
    open_scenes = ["남자가 가면을 쓰고", "교복 입고 춤추고", "여자가 전화기 들고", "남자가 비 맞으면서", "의자가 하늘에 떠 있고",
                   "아이들이 풍선 들고 뛰어다니며"]
    singer_nominals = [" 가수는 여자", " 가수는 여자야", " 보컬은 남자", " 보컬은 남자였어", " 가수가 여자였어", " 래퍼는 남자",
                       " 부른 건 여자야", " 목소리는 여자야"]
    checked = 0
    for marker, opener in itertools.product(["뮤비", "뮤직비디오"], ["에서 ", " 마지막에 "]):
        for scene, tail in itertools.product(open_scenes, singer_nominals):
            query = f"{marker}{opener}{scene}{tail}"
            assert is_specific_mv_scene_query(query), query
            assert mv_scene_phrases(query) == [" ".join(f"{marker}{opener}{scene}".split())], (query, mv_scene_phrases(query))
            checked += 1
    assert checked == 4 * len(open_scenes) * len(singer_nominals)
    # 화면 속 명사가 주어면 명사 서술까지가 장면이다
    assert mv_scene_phrases("뮤비에서 춤추는 사람은 여자였어") == ["뮤비에서 춤추는 사람은 여자였어"]
    assert mv_scene_phrases("뮤비에서 의자가 하늘에 떠 있고 주인공은 여자였어") == [
        "뮤비에서 의자가 하늘에 떠 있고 주인공은 여자였어"]
