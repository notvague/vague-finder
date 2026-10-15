"""속성 조건 보조 후보(`ATTR_FILTER_WEIGHT`) — 실험 스위치. 모델·DB 없음.

v50 진단: 후보 30 밖 정답 15건은 기본 경로에서는 멀지만 속성 가산을 받으면 30위 안이었다. 깊이를 넓히지 않고
분석의 시기·성별·장르·형태 조건에 맞는 곡만 같은 텍스트 질의로 한 번 더 검색해 낮은 가중으로 더한다.
2026-10-15 채택: 기본값은 가중 0.5 · `era_plus`. 가중 0이면 채택 전(v50 B)과 비트 단위로 같아야 하고, 켜져 있을 때만
필터가 만들어지고 필터 검색이 한 번 더 돈다.
"""
from __future__ import annotations

from qdrant_client.http import models

from src.backend.schemas.query import ArtistTypeClue, ReleaseEra
from src.retrieval import search_router as sr
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


def test_unknown_genre_wording_is_dropped_not_guessed(monkeypatch) -> None:
    """등록 장르에 없는 말("펑크록")은 조건이 되지 않는다 — 가산 규칙과 같다. 남은 축이 성별뿐이면 열지 않는다."""
    monkeypatch.setenv("ATTR_FILTER_GATE", "two_axes")
    assert sr._build_attribute_filter(_analysis(vocal_gender="여성", genre="펑크록")) is None


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


def _search(monkeypatch, weight: str | None, *, k: str | None = None):
    for name, value in (("ATTR_FILTER_WEIGHT", weight), ("ATTR_FILTER_K", k)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    text_hits = [_track(f"t{i}", title=f"텍스트{i}", release_date="1999.01.01") for i in range(1, 61)]
    # 조건에 맞는 곡 — 텍스트 경로 깊은 곳(또는 밖)에 있다
    filtered = [_track(f"f{i}", title=f"조건곡{i}", release_date="2005.05.05") for i in range(1, 6)]
    service = _RecordingTextService(text_hits, filtered)
    router = _router(text_hits)
    router._text_svc = service
    rec = ExplainRecorder("q")
    results = _run(router, _analysis(release_era=ERA, vocal_gender="남성"), top_k=30, use_rerank=False, recorder=rec)
    return [t.id for t in results], service.calls, rec.record


def test_default_is_on_with_weight_half(monkeypatch) -> None:
    """환경 변수가 없으면 가중 0.5 · era_plus — 채택된 기본값."""
    monkeypatch.delenv("ATTR_FILTER_WEIGHT", raising=False)
    assert sr._attr_filter_weight() == 0.5
    ids, calls, _ = _search(monkeypatch, None)
    assert sum(1 for c in calls if c.get("metadata_filter")) == 1
    assert any(sid.startswith("f") for sid in ids)


def test_weight_zero_restores_pre_adoption_behaviour(monkeypatch) -> None:
    ids, calls, record = _search(monkeypatch, "0")
    assert all("metadata_filter" not in c for c in calls)
    assert not any(sid.startswith("f") for sid in ids)
    assert not any(p.path == "attr_filter" for s in record.songs.values() for p in s.paths)


def test_weight_opens_a_filtered_search_and_adds_candidates(monkeypatch) -> None:
    ids, calls, record = _search(monkeypatch, "0.5", k="5")
    filtered_calls = [c for c in calls if c.get("metadata_filter")]
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
