"""속성 조건 보조 후보(`ATTR_FILTER_WEIGHT`) — 실험 스위치. 모델·DB 없음.

v50 진단: 후보 30 밖 정답 15건은 기본 경로에서는 멀지만 속성 가산을 받으면 30위 안이었다. 깊이를 넓히지 않고
분석의 시기·성별·장르·형태 조건에 맞는 곡만 같은 텍스트 질의로 한 번 더 검색해 낮은 가중으로 더한다.
기본값은 가중 0(경로 없음 = 현행). 켜져 있을 때만 필터가 만들어지고 필터 검색이 한 번 더 돈다.
재질문 2턴에서는 답(성별·장르·형태·시기·출생 연도)을 반영한 분석으로 필터를 만든다 — 최초 분석의 틀린 조건이
이 경로에서만 남으면 안 된다. 텍스트 질의 자체는 원래 분석 그대로다.
"""
from __future__ import annotations

from qdrant_client.http import models

from src.backend.schemas.query import ArtistTypeClue, LifeStageClue, ReleaseEra
from src.backend.schemas.search import ClarifyAnswer
from src.retrieval import search_router as sr
from src.retrieval.clarify import BIRTH_YEAR_SLOT
from src.retrieval.explain import ExplainRecorder
from src.vector_db.qdrant_backend import translate_filter
from tests.test_search_explain import _analysis, _router, _run, _track


# ---------------------------------------------------------------------------
# 필터 만들기
# ---------------------------------------------------------------------------

def test_no_attribute_no_filter() -> None:
    assert sr._build_attribute_filter(_analysis()) is None


def test_gender_alone_does_not_open_the_path(monkeypatch) -> None:
    """성별 하나는 3,000곡이라 좁히는 힘이 없다 — 어느 게이트에서도 열리지 않는다."""
    for gate in ("era_plus", "two_axes"):
        monkeypatch.setenv("ATTR_FILTER_GATE", gate)
        assert sr._build_attribute_filter(_analysis(vocal_gender="남성")) is None
        assert sr._build_attribute_filter(_analysis(genre="발라드")) is None
    monkeypatch.delenv("ATTR_FILTER_GATE", raising=False)
    assert sr._build_attribute_filter(_analysis(vocal_gender="남성")) is None
    assert sr._build_attribute_filter(_analysis(genre="발라드")) is None


def test_era_window_alone_opens_the_path_with_year_text_conditions(monkeypatch) -> None:
    """시기만 있는 필터는 two_axes에서만 열린다(기본 era_plus는 다른 축이 하나 더 있어야 한다)."""
    monkeypatch.setenv("ATTR_FILTER_GATE", "two_axes")
    analysis = _analysis(release_era=ReleaseEra(start_year=2004, end_year=2006, confidence=0.75))
    built = sr._build_attribute_filter(analysis)
    assert built == {"$and": [{"$or": [
        {"release_date": {"$text": "2004"}},
        {"release_date": {"$text": "2005"}},
        {"release_date": {"$text": "2006"}},
    ]}]}


def test_wide_era_window_is_not_a_condition() -> None:
    """13년 이상(예: 2010~2026)은 좁히지 못하므로 빼고, 남은 축이 하나뿐이면 열지 않는다."""
    analysis = _analysis(
        release_era=ReleaseEra(start_year=2010, end_year=2026, confidence=0.5), vocal_gender="여성",
    )
    assert sr._build_attribute_filter(analysis) is None


def test_broken_era_values_are_not_a_condition() -> None:
    """끝이 시작보다 앞서거나(end < start) 한쪽이 비면 시기 축이 없는 것으로 본다 — 예외를 내지 않는다."""
    for era in (
        ReleaseEra(start_year=2008, end_year=2004, confidence=0.7),
        ReleaseEra(start_year=2008, end_year=None, confidence=0.7),
        ReleaseEra(start_year=None, end_year=2008, confidence=0.7),
        ReleaseEra(start_year=2008, end_year=2009, confidence=0.0),
    ):
        assert sr._build_attribute_filter(_analysis(release_era=era, vocal_gender="남성")) is None


def test_two_axes_without_era_open_the_path_with_boost_compatible_semantics(monkeypatch) -> None:
    """성별 남성은 혼성도 허용(vocal_gender_partial), 장르 "록"은 등록 장르 "록/메탈"(genre_match 부분 문자열),
    형태 밴드는 payload type 별칭. 시기 없는 두 축은 two_axes 게이트에서만 열린다."""
    monkeypatch.setenv("ATTR_FILTER_GATE", "two_axes")
    analysis = _analysis(
        vocal_gender="남성", genre="록",
        artist_type=ArtistTypeClue(values=["밴드"], confidence=0.7),
    )
    built = sr._build_attribute_filter(analysis)
    assert built == {"$and": [
        {"vocal_gender": {"$in": ["남성", "혼성"]}},
        {"genre": {"$in": ["록/메탈"]}},
        {"type": {"$in": ["밴드", "인디밴드"]}},
    ]}


def test_mixed_gender_and_unknown_values_are_handled(monkeypatch) -> None:
    """혼성은 혼성만, 등록 장르에 없는 말("펑크록")은 조건이 되지 않는다 — 가산 규칙과 같다.
    형태 값은 스키마(ArtistTypeClue.values)가 솔로/그룹/듀오/밴드로 막으므로 모르는 형태는 여기 오지 않는다."""
    monkeypatch.setenv("ATTR_FILTER_GATE", "two_axes")
    assert sr._build_attribute_filter(_analysis(vocal_gender="여성", genre="펑크록")) is None
    era = ReleaseEra(start_year=2006, end_year=2007, confidence=0.8)
    built = sr._build_attribute_filter(_analysis(release_era=era, vocal_gender="혼성", genre="펑크록"))
    assert built == {"$and": [
        {"$or": [{"release_date": {"$text": "2006"}}, {"release_date": {"$text": "2007"}}]},
        {"vocal_gender": {"$eq": "혼성"}},
    ]}


def test_era_plus_gate_requires_era_and_another_axis(monkeypatch) -> None:
    """기본 게이트(환경 변수 없음)가 era_plus다."""
    monkeypatch.delenv("ATTR_FILTER_GATE", raising=False)
    assert sr._attr_filter_gate() == "era_plus"
    era_only = _analysis(release_era=ReleaseEra(start_year=2006, end_year=2007, confidence=0.8))
    assert sr._build_attribute_filter(era_only) is None
    two_axes_no_era = _analysis(vocal_gender="남성", genre="록")
    assert sr._build_attribute_filter(two_axes_no_era) is None
    era_and_gender = _analysis(
        release_era=ReleaseEra(start_year=2006, end_year=2007, confidence=0.8), vocal_gender="남성",
    )
    assert sr._build_attribute_filter(era_and_gender) == {"$and": [
        {"$or": [{"release_date": {"$text": "2006"}}, {"release_date": {"$text": "2007"}}]},
        {"vocal_gender": {"$in": ["남성", "혼성"]}},
    ]}
    monkeypatch.setenv("ATTR_FILTER_GATE", "nonsense")
    assert sr._attr_filter_gate() == "era_plus"
    monkeypatch.setenv("ATTR_FILTER_GATE", "two_axes")
    assert sr._build_attribute_filter(era_only) is not None


def test_text_operator_translates_to_match_text() -> None:
    built = translate_filter({"$and": [{"$or": [{"release_date": {"$text": "2004"}}]},
                                       {"vocal_gender": {"$in": ["남성", "혼성"]}}]})
    assert isinstance(built, models.Filter)
    year_branch = built.must[0]
    assert isinstance(year_branch, models.Filter)
    cond = year_branch.should[0]
    assert isinstance(cond, models.FieldCondition) and cond.key == "release_date"
    assert isinstance(cond.match, models.MatchText) and cond.match.text == "2004"


# ---------------------------------------------------------------------------
# 라우터 경로
# ---------------------------------------------------------------------------

class _RecordingTextService:
    """필터 없는 호출은 기본 목록, 필터가 있는 호출은 다른 목록을 돌려준다. 호출 인자를 남긴다."""

    def __init__(self, hits, filtered_hits):
        self._hits = hits
        self._filtered = filtered_hits
        self.calls: list = []

    def search_text(self, *a, **kw):
        self.calls.append(kw)
        top_k = kw.get("top_k") or 30
        source = self._filtered if kw.get("metadata_filter") else self._hits
        return [t.model_copy() for t in source[:top_k]]

    def track_from_match(self, match):
        return _track(str(match.get("id")))


ERA = ReleaseEra(start_year=2004, end_year=2006, confidence=0.75)


def _setup(monkeypatch, weight: str | None, *, k: str | None = None, text_n: int = 60,
           filtered=None, lyrics_hits=None):
    for name, value in (("ATTR_FILTER_WEIGHT", weight), ("ATTR_FILTER_K", k)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    text_hits = [_track(f"t{i}", title=f"텍스트{i}", release_date="1999.01.01") for i in range(1, text_n + 1)]
    # 조건에 맞는 곡 — 텍스트 경로 깊은 곳(또는 밖)에 있다
    if filtered is None:
        filtered = [_track(f"f{i}", title=f"조건곡{i}", release_date="2005.05.05") for i in range(1, 6)]
    service = _RecordingTextService(text_hits, filtered)
    router = _router(text_hits, lyrics_hits=lyrics_hits)
    router._text_svc = service
    return router, service


def _search(monkeypatch, weight: str | None, *, k: str | None = None, **search_kw):
    router, service = _setup(monkeypatch, weight, k=k)
    rec = ExplainRecorder("q")
    analysis = search_kw.pop("analysis", None) or _analysis(release_era=ERA, vocal_gender="남성")
    results = _run(router, analysis, top_k=30, use_rerank=False, recorder=rec, **search_kw)
    return [t.id for t in results], service.calls, rec.record


def _filtered_calls(calls):
    return [c for c in calls if c.get("metadata_filter")]


def test_default_is_off(monkeypatch) -> None:
    """환경 변수가 없으면 가중 0 — 필터 검색이 없고 조건곡도 들어오지 않는다(현행과 같다)."""
    monkeypatch.delenv("ATTR_FILTER_WEIGHT", raising=False)
    assert sr._attr_filter_weight() == 0.0
    ids, calls, record = _search(monkeypatch, None)
    assert not _filtered_calls(calls)
    assert not any(sid.startswith("f") for sid in ids)
    assert not any(p.path == "attr_filter" for s in record.songs.values() for p in s.paths)


def test_weight_zero_keeps_existing_behaviour_bit_for_bit(monkeypatch) -> None:
    """가중 0은 환경 변수를 지운 것과 같아야 한다 — 호출 인자·후보 순서·기록이 모두 같다."""
    off_ids, off_calls, _ = _search(monkeypatch, None)
    zero_ids, zero_calls, record = _search(monkeypatch, "0")
    assert zero_ids == off_ids
    assert all("metadata_filter" not in c for c in zero_calls)
    assert len(zero_calls) == len(off_calls)
    assert not any(p.path == "attr_filter" for s in record.songs.values() for p in s.paths)


def test_weight_opens_a_filtered_search_and_adds_candidates(monkeypatch) -> None:
    ids, calls, record = _search(monkeypatch, "0.5", k="5")
    filtered_calls = _filtered_calls(calls)
    assert len(filtered_calls) == 1
    assert filtered_calls[0]["metadata_filter"] == sr._build_attribute_filter(_analysis(release_era=ERA, vocal_gender="남성"))
    assert filtered_calls[0]["top_k"] == 5
    # 조건곡이 후보에 들어왔고(시기 가산까지 받아 앞쪽), 기록에는 경로 이름이 남는다
    assert any(sid.startswith("f") for sid in ids)
    contributions = [p for p in record.songs["f1"].paths if p.path == "attr_filter"]
    assert contributions and contributions[0].rank == 1
    assert contributions[0].delta == 0.5 / (sr._RRF_K + 1)
    # 기존 텍스트 후보가 통째로 사라지지는 않는다(소환이지 hard filter가 아니다)
    assert "t1" in ids


def test_weight_without_attributes_changes_nothing(monkeypatch) -> None:
    monkeypatch.setenv("ATTR_FILTER_WEIGHT", "0.5")
    text_hits = [_track(f"t{i}", title=f"텍스트{i}") for i in range(1, 40)]
    service = _RecordingTextService(text_hits, [])
    router = _router(text_hits)
    router._text_svc = service
    results = _run(router, _analysis(), top_k=30, use_rerank=False)
    assert all("metadata_filter" not in c for c in service.calls)
    assert [t.id for t in results] == [f"t{i}" for i in range(1, 31)]


def test_pool_size_and_no_duplicates_when_filtered_list_overlaps(monkeypatch) -> None:
    """보조 목록이 기존 텍스트 후보와 겹쳐도 후보는 한 번씩만, 최종 후보 수는 candidate_k 그대로다.
    같은 곡은 어느 목록에서 왔든 payload가 하나다 — 겹치는 곡은 두 목록 모두 2005년으로 둔다."""
    monkeypatch.setenv("ATTR_FILTER_WEIGHT", "0.5")
    monkeypatch.delenv("ATTR_FILTER_K", raising=False)
    overlap_ids = {3, 40, 55}
    text_hits = [
        _track(f"t{i}", title=f"텍스트{i}", release_date="2005.05.05" if i in overlap_ids else "1999.01.01")
        for i in range(1, 61)
    ]
    overlap = [t for t in text_hits if int(t.id[1:]) in overlap_ids]
    fresh = [_track(f"f{i}", title=f"조건곡{i}", release_date="2005.05.05") for i in range(1, 4)]
    service = _RecordingTextService(text_hits, [*overlap, *fresh])
    router = _router(text_hits)
    router._text_svc = service
    pool: list = []
    results = _run(router, _analysis(release_era=ERA, vocal_gender="남성"), top_k=10, use_rerank=False,
                   candidate_k=30, candidate_ids_out=pool)
    assert len(_filtered_calls(service.calls)) == 1
    assert len(pool) == 30 and len(set(pool)) == 30
    assert len(results) == 10 and len({t.id for t in results}) == 10
    # 겹친 곡(t40 · t55)은 텍스트 깊이(30) 밖이라 텍스트 표는 없지만 보조 표 + 시기 가산으로 후보에 들어오고, 한 자리씩만 차지한다
    assert "t40" in pool and pool.count("t40") == 1 and "t55" in pool
    assert "t3" in pool and pool.count("t3") == 1


def test_rejected_songs_from_filtered_list_are_excluded(monkeypatch) -> None:
    """거절한 곡은 보조 목록으로 들어왔더라도 후보에서 빠지고, 거절 수만큼 뒤에서 채워진다."""
    router, service = _setup(monkeypatch, "0.5")
    pool: list = []
    _run(router, _analysis(release_era=ERA, vocal_gender="남성"), top_k=10, use_rerank=False,
         candidate_k=30, candidate_ids_out=pool, exclude_ids=["f1", "f2", "t1"])
    assert not {"f1", "f2", "t1"} & set(pool)
    assert len(pool) == 30 and len(set(pool)) == 30
    assert {"f3", "f4", "f5"} <= set(pool)


def test_exact_lyric_match_still_placed_first_with_filtered_list(monkeypatch) -> None:
    """가사 exact 보호는 보조 목록보다 위다 — 조건곡 30개가 들어와도 보호곡이 1위다."""
    protected = _track("p", title="보호곡", lyric_match_type="exact", lyric_match_score=0.95)
    filtered = [_track(f"f{i}", title=f"조건곡{i}", release_date="2005.05.05") for i in range(1, 31)]
    router, service = _setup(monkeypatch, "0.5", filtered=filtered, lyrics_hits=[protected])
    analysis = _analysis(release_era=ERA, vocal_gender="남성", lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])
    results = _run(router, analysis, top_k=10, use_rerank=False)
    assert len(_filtered_calls(service.calls)) == 1
    assert results[0].id == "p"
    assert any(t.id.startswith("f") for t in results)


# ---------------------------------------------------------------------------
# 재질문 2턴 — 답이 필터 조건에 반영된다
# ---------------------------------------------------------------------------

def test_answer_corrects_the_filter_condition(monkeypatch) -> None:
    """최초 분석 "남성"을 사용자가 "여성"으로 고치면 필터도 여성(+혼성)이다. 텍스트 질의 인자는 그대로다."""
    ids, calls, _ = _search(monkeypatch, "0.5", answers=[ClarifyAnswer(slot="vocal_gender", value="여성")])
    filtered = _filtered_calls(calls)
    assert len(filtered) == 1
    assert filtered[0]["metadata_filter"] == {"$and": [
        {"$or": [{"release_date": {"$text": str(y)}} for y in (2004, 2005, 2006)]},
        {"vocal_gender": {"$in": ["여성", "혼성"]}},
    ]}
    # 원래 분석 객체는 바뀌지 않는다(답은 사본에만 병합)
    analysis = _analysis(release_era=ERA, vocal_gender="남성")
    _run(_setup(monkeypatch, "0.5")[0], analysis, top_k=30, use_rerank=False,
         answers=[ClarifyAnswer(slot="vocal_gender", value="여성")])
    assert analysis.vocal_gender == "남성"


def test_skipped_answer_leaves_the_filter_as_is(monkeypatch) -> None:
    plain = _filtered_calls(_search(monkeypatch, "0.5")[1])[0]["metadata_filter"]
    skipped = _filtered_calls(_search(
        monkeypatch, "0.5", answers=[ClarifyAnswer(slot="vocal_gender", skipped=True)],
    )[1])[0]["metadata_filter"]
    assert skipped == plain


def test_genre_type_and_era_answers_become_conditions(monkeypatch) -> None:
    """장르·형태·연대 답은 각각 조건 축이 된다. 연대 답("2000년대")은 10년 창이라 열린다."""
    analysis = _analysis(vocal_gender="남성")  # 시기 없음 → 답이 없으면 era_plus에서 닫힘
    _, calls_without, _ = _search(monkeypatch, "0.5", analysis=analysis)
    assert not _filtered_calls(calls_without)
    answers = [
        ClarifyAnswer(slot="release_era", value="2000년대"),
        ClarifyAnswer(slot="genre", value="발라드"),
        ClarifyAnswer(slot="type", value="듀오"),
    ]
    _, calls, _ = _search(monkeypatch, "0.5", analysis=_analysis(vocal_gender="남성"), answers=answers)
    built = _filtered_calls(calls)[0]["metadata_filter"]
    assert built == {"$and": [
        {"$or": [{"release_date": {"$text": str(y)}} for y in range(2000, 2010)]},
        {"vocal_gender": {"$in": ["남성", "혼성"]}},
        {"genre": {"$in": ["발라드"]}},
        {"type": {"$in": ["듀엣", "듀오"]}},  # 별칭은 정렬된 집합
    ]}


def test_birth_year_answer_opens_the_era_axis(monkeypatch) -> None:
    """생애 단계("중학교 때") + 출생 연도 답 → 시기 창이 생겨 성별과 함께 필터가 열린다."""
    stage = LifeStageClue(stage="middle", text="중학교 때", spans=[[0, 5]], confidence=0.9, age_from=13, age_to=15)
    analysis = _analysis(original_query="중학교 때 듣던 남자 발라드", life_stage=stage, vocal_gender="남성")
    assert sr._build_attribute_filter(analysis) is None
    _, calls, _ = _search(monkeypatch, "0.5", analysis=analysis,
                          answers=[ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1990")])
    built = _filtered_calls(calls)[0]["metadata_filter"]
    # 1990 + 13 − 1 … 1990 + 15 + 1 = 2002~2006 (release_era_from_birth_year, pad 1년)
    assert built["$and"][0] == {"$or": [{"release_date": {"$text": str(y)}} for y in range(2002, 2007)]}
    assert built["$and"][1] == {"vocal_gender": {"$in": ["남성", "혼성"]}}


def test_answers_with_weight_zero_still_do_nothing(monkeypatch) -> None:
    _, calls, record = _search(monkeypatch, "0", answers=[ClarifyAnswer(slot="vocal_gender", value="여성")])
    assert not _filtered_calls(calls)
    assert not any(p.path == "attr_filter" for s in record.songs.values() for p in s.paths)


def test_switch_values_are_clamped() -> None:
    import os
    os.environ["ATTR_FILTER_K"] = "0"
    try:
        assert sr._attr_filter_k() == 1
        os.environ["ATTR_FILTER_K"] = "500"
        assert sr._attr_filter_k() == 100
        os.environ["ATTR_FILTER_K"] = "abc"
        assert sr._attr_filter_k() == 30
    finally:
        os.environ.pop("ATTR_FILTER_K", None)
