"""단계 1 — 재질문 상태 왕복 + Reject-only.

핵심 불변조건 둘을 고정한다.
1. 피드백이 없으면 기존 검색 동작이 하나도 바뀌지 않는다.
2. 거절한 곡은 자르기 전에 빠지고, 거절한 만큼 후보를 더 가져온다 —
   둘이 같이 있어야 다음 턴 후보 풀이 얇아지지 않는다.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.retrieval.clarify import merge_answer
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack, SearchRequest
from src.retrieval.search_router import SearchRouter, _reject_and_trim


# ---------------------------------------------------------------------------
# _reject_and_trim — Reject-only 기준선이 성립하는 조건
# ---------------------------------------------------------------------------

def _pool(n: int) -> List[tuple]:
    """점수 내림차순 후보 풀. id는 s000, s001, ... 순위와 같다."""
    return [(f"s{i:03d}", 1.0 - i * 0.001) for i in range(n)]


def test_no_exclusion_is_plain_trim() -> None:
    """제외가 없으면 그냥 자르기다 — 기존 동작이 바뀌면 안 된다."""
    ranked = _pool(100)
    assert _reject_and_trim(ranked, None, 30) == ranked[:30]
    assert _reject_and_trim(ranked, set(), 30) == ranked[:30]


def test_rejected_ids_are_removed() -> None:
    ranked = _pool(50)
    got = _reject_and_trim(ranked, {"s000", "s003"}, 30)
    ids = [song_id for song_id, _ in got]
    assert "s000" not in ids
    assert "s003" not in ids


def test_pool_refills_after_rejection() -> None:
    """상위 10곡을 거절해도 후보 풀은 여전히 30개다.

    이게 단계 1의 핵심이다. 자른 뒤에 걸렀다면 20개가 됐을 것이다.

    단 이 테스트만으로는 부족하다 — 여기서는 100개짜리 리스트를 직접 넣지만,
    실제 search()는 각 경로와 RRF를 candidates개로 이미 잘라서 넘긴다.
    그 상태에서는 메울 곡이 없어 풀이 20개가 된다. 실경로 보장은 아래
    "search() 실경로" 절에서 확인한다.
    """
    ranked = _pool(100)
    rejected = {song_id for song_id, _ in ranked[:10]}

    got = _reject_and_trim(ranked, rejected, 30)

    assert len(got) == 30
    # 11위였던 곡이 새 1위가 되고, 원래 40위가 새 30위로 올라온다.
    assert got[0][0] == "s010"
    assert got[-1][0] == "s039"


def test_pool_shrinks_only_when_source_is_exhausted() -> None:
    """원본이 모자라면 그건 어쩔 수 없다 — 없는 걸 만들어내진 않는다."""
    ranked = _pool(12)
    rejected = {song_id for song_id, _ in ranked[:10]}
    assert len(_reject_and_trim(ranked, rejected, 30)) == 2


def test_order_is_preserved() -> None:
    """제외는 순위를 재계산하지 않는다. 빈자리만 메운다."""
    ranked = _pool(20)
    got = _reject_and_trim(ranked, {"s005"}, 10)
    ids = [song_id for song_id, _ in got]
    assert ids == ["s000", "s001", "s002", "s003", "s004",
                   "s006", "s007", "s008", "s009", "s010"]


# ---------------------------------------------------------------------------
# search() 실경로 — 후보 풀 폭
#
# 헬퍼 단위 테스트는 100개짜리 리스트를 직접 넣기 때문에 "거르고 자른다"만
# 검증한다. 실제로는 각 검색 경로와 RRF가 이미 candidates개로 잘라서
# 넘기므로, 거절한 만큼 폭을 넓혀 오지 않으면 풀이 30 → 20으로 얇아진다.
# 그 회귀를 여기서 실제 search()로 고정한다.
# ---------------------------------------------------------------------------

_CORPUS = 200  # 인덱스에 곡이 충분히 있는 상황


def _stub_router() -> SearchRouter:
    """벡터 DB·임베딩 없이 search()의 퓨전·절단 로직만 실행하는 라우터.

    각 경로 스텁은 '요청한 top_k만큼 돌려준다' — 실제 벡터 DB 조회와 같은 계약이다.
    폭을 넓혀 요청하지 않으면 뒤쪽 후보는 애초에 존재하지 않는다.
    """
    router = SearchRouter.__new__(SearchRouter)
    router._text_svc = None
    router._img_emb = None
    router._audio_emb = None
    router._img_idx = None
    router._audio_idx = None
    router._lyrics_svc = None
    router._reranker = None
    router._pool = ThreadPoolExecutor(max_workers=2)

    def text_hits(analysis: QueryAnalysis, top_k: Optional[int]) -> List[MatchingTrack]:
        n = min(top_k or 30, _CORPUS)
        return [
            MatchingTrack(
                id=f"s{i:03d}",
                score=1.0 - i * 0.001,
                title=f"곡{i}",
                artist="가수",
            )
            for i in range(n)
        ]

    def no_hits(analysis: QueryAnalysis, top_k: Optional[int]) -> List[MatchingTrack]:
        return []

    router._search_text = text_hits
    router._search_balanced_semantic = text_hits
    for path in (
        "_search_image",
        "_search_audio",
        "_search_lyrics",
        "_search_performance_clues",
        "_search_performance_metadata",
        "_search_title_constrained",
        "_search_title_presence",
        "_search_title_meaning",
    ):
        setattr(router, path, no_hits)
    return router


@pytest.fixture()
def stub_router():
    router = _stub_router()
    try:
        yield router
    finally:
        router.shutdown()


def _search(
    router: SearchRouter,
    analysis: QueryAnalysis,
    exclude: Optional[List[str]] = None,
    candidate_k: int = 30,
) -> Tuple[List[MatchingTrack], List[str]]:
    pool: List[str] = []
    results = asyncio.run(
        router.search(
            analysis,
            top_k=10,
            use_rerank=False,
            candidate_k=candidate_k,
            exclude_ids=exclude or [],
            candidate_ids_out=pool,
        )
    )
    return results, pool


def _mood_query() -> QueryAnalysis:
    """단서가 하나도 없는 순수 무드 질의 — 재질문의 주 대상."""
    return _analysis("비 오는 날 창밖 보면서 듣기 좋은 잔잔한 곡")


def _metadata_clue_query() -> QueryAnalysis:
    analysis = _mood_query()
    analysis.artist_type.values = ["그룹"]
    analysis.artist_type.confidence = 0.8
    return analysis


def _sound_clue_query() -> QueryAnalysis:
    """q115류 — 사운드만 상세히 묘사한 질의 (deep audio fusion 경로)."""
    analysis = QueryAnalysis(
        original_query="휘파람으로 시작하는 듀엣곡",
        intent_type="mood",
        image_english_query="",
        audio_english_query="a duet with a whistling intro",
    )
    analysis.performance_clues.sound_ensemble = ["휘파람"]
    analysis.performance_clues.confidence = 0.9
    return analysis


_QUERY_KINDS: List[Tuple[str, Callable[[], QueryAnalysis]]] = [
    ("무드만", _mood_query),
    ("메타데이터 단서", _metadata_clue_query),
    ("사운드 단서", _sound_clue_query),
]


@pytest.mark.parametrize("kind, build", _QUERY_KINDS, ids=[k for k, _ in _QUERY_KINDS])
def test_pool_is_candidate_k_on_first_turn(kind, build, stub_router) -> None:
    _, pool = _search(stub_router, build())
    assert len(pool) == 30


@pytest.mark.parametrize("kind, build", _QUERY_KINDS, ids=[k for k, _ in _QUERY_KINDS])
def test_pool_stays_full_after_rejecting_top10(kind, build, stub_router) -> None:
    """Top-10을 전부 거절해도 다음 턴 후보가 30개다 — 질의 종류와 무관하게.

    회귀 이력: 예전에는 사운드 단서 질의(deep audio fusion)만 30을 유지했고
    나머지는 20으로 얇아졌다. q115가 마침 그 경로라 수동 검증에서 드러나지 않았다.
    """
    analysis = build()
    first, _ = _search(stub_router, analysis)
    rejected = [track.id for track in first[:10]]

    second, pool = _search(stub_router, analysis, exclude=rejected)

    assert len(pool) == 30
    assert not set(rejected) & set(pool)
    assert not set(rejected) & {track.id for track in second}


def test_rejection_only_shifts_survivors_up(stub_router) -> None:
    """제외는 순위를 재계산하지 않는다. 11위가 1위가 되고 40위까지 올라온다."""
    analysis = _mood_query()
    first, _ = _search(stub_router, analysis)
    rejected = [track.id for track in first[:10]]

    _, pool = _search(stub_router, analysis, exclude=rejected)

    assert pool == [f"s{i:03d}" for i in range(10, 40)]


def test_pool_survives_chained_rejections(stub_router) -> None:
    """2턴 연속 거절(누적 20곡)에도 풀은 30개다. 3턴째 질문 선택의 전제."""
    analysis = _mood_query()
    rejected = [f"s{i:03d}" for i in range(20)]

    _, pool = _search(stub_router, analysis, exclude=rejected)

    assert len(pool) == 30
    assert pool[0] == "s020"


def test_widening_does_not_touch_the_no_rejection_path(stub_router) -> None:
    """제외가 없으면 폭도 그대로다 — 기존 검색 결과가 변하면 안 된다."""
    analysis = _mood_query()
    results, pool = _search(stub_router, analysis)

    assert pool == [f"s{i:03d}" for i in range(30)]
    assert [track.id for track in results] == [f"s{i:03d}" for i in range(10)]


# ---------------------------------------------------------------------------
# 요청 스키마 검증
# ---------------------------------------------------------------------------

def _analysis(query: str = "비 오는 날 여자 가수 발라드") -> QueryAnalysis:
    return QueryAnalysis(
        original_query=query,
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
    )


def test_first_turn_needs_nothing() -> None:
    req = SearchRequest(query="아무 노래")
    assert req.answers == []
    assert req.rejected_ids == []
    assert req.turn == 1


def test_answers_require_prior_analysis() -> None:
    """답변만 오면 어디에 병합할지 알 수 없다."""
    with pytest.raises(ValidationError, match="prior_analysis"):
        SearchRequest(
            query="아무 노래",
            answers=[ClarifyAnswer(slot="vocal_gender", value="여성")],
        )


def test_rejection_without_prior_analysis_is_allowed() -> None:
    """거절만이면 분석을 다시 해도 동작한다. 막지 않고 라우터가 경고만 남긴다."""
    req = SearchRequest(query="아무 노래", rejected_ids=["s001"])
    assert req.rejected_ids == ["s001"]


def test_duplicate_rejections_are_rejected() -> None:
    """중복이 쌓이면 상한 20에 먼저 걸려 실제 거절 곡이 잘린다."""
    with pytest.raises(ValidationError, match="중복"):
        SearchRequest(query="아무 노래", rejected_ids=["s001", "s001"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("rejected_ids", [f"s{i}" for i in range(21)]),
        ("previous_candidate_ids", [f"s{i}" for i in range(31)]),
        ("asked_slots", ["genre", "type", "vocal_gender"]),
    ],
)
def test_state_field_limits(field: str, value: List[str]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**{"query": "아무 노래", field: value})


def test_answers_capped_at_two_turns() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="아무 노래",
            prior_analysis=_analysis(),
            answers=[
                ClarifyAnswer(slot="genre", value="발라드"),
                ClarifyAnswer(slot="type", value="솔로"),
                ClarifyAnswer(slot="vocal_gender", value="여성"),
            ],
        )


# ---------------------------------------------------------------------------
# merge_answer — 필드 타입 함정 회귀 가드
# ---------------------------------------------------------------------------

def test_skipped_answer_changes_nothing() -> None:
    """'잘 모르겠어요'는 슬롯을 채우지 않는다."""
    analysis = _analysis()
    before = analysis.model_dump()
    merge_answer(analysis, ClarifyAnswer(slot="vocal_gender", skipped=True))
    assert analysis.model_dump() == before


def test_artist_type_goes_into_values_list() -> None:
    """artist_type은 문자열이 아니라 ArtistTypeClue다.

    setattr(analysis, 'artist_type', '그룹')은 타입 에러였다.
    """
    analysis = _analysis()
    merge_answer(analysis, ClarifyAnswer(slot="type", value="그룹"))
    assert analysis.artist_type.values == ["그룹"]
    assert analysis.artist_type.confidence == pytest.approx(0.8)


def test_release_era_is_parsed_into_year_range() -> None:
    analysis = _analysis()
    merge_answer(analysis, ClarifyAnswer(slot="release_era", value="2010년대"))
    assert analysis.release_era.start_year == 2010
    assert analysis.release_era.end_year == 2019


def test_unknown_value_is_ignored_not_raised() -> None:
    """사용자 답이 이상해도 검색 자체는 계속돼야 한다."""
    analysis = _analysis()
    merge_answer(analysis, ClarifyAnswer(slot="vocal_gender", value="외계인"))
    assert analysis.vocal_gender is None


# ---------------------------------------------------------------------------
# 라우트 배선 — 상태가 실제로 라우터까지 흘러가는가
# ---------------------------------------------------------------------------

class _FakeSearchRouter:
    """search() 호출 인자를 기록하고 고정 결과를 돌려준다.

    rerank=True면 리랭커가 풀 뒤쪽 곡을 상위로 끌어올리는 상황을 흉내 낸다.
    실서비스는 use_rerank가 기본 True라 후보 풀 순서(candidate_ids)와 실제로
    보여준 순서(results)가 다르다. 단계 4의 질문 선택이 여기에 걸린다.
    """

    def __init__(
        self,
        pool_ids: Optional[List[str]] = None,
        rerank: bool = False,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.pool_ids = pool_ids or [f"s{i:03d}" for i in range(50)]
        self.rerank = rerank

    async def search(self, analysis, **kwargs) -> List[MatchingTrack]:
        self.calls.append({"analysis": analysis, **kwargs})

        excluded = set(kwargs.get("exclude_ids") or [])
        surviving = [i for i in self.pool_ids if i not in excluded]

        out = kwargs.get("candidate_ids_out")
        if out is not None:
            # 후보 풀은 리랭킹 전 순서로 나간다.
            out.extend(surviving)

        # 질문 선택은 후보의 성별·장르를 본다. 비워 두면 pick_question이 항상
        # None을 돌려줘 종료 조건 테스트가 무의미해진다.
        tracks_out = kwargs.get("candidate_tracks_out")
        if tracks_out is not None:
            tracks_out.extend(
                MatchingTrack(
                    id=song_id, score=1.0, title=f"곡{song_id}",
                    vocal_gender="남성" if i % 2 else "여성",
                    genre="발라드",
                )
                for i, song_id in enumerate(surviving)
            )

        top_k = kwargs.get("top_k", 10)
        shown = surviving[:top_k]
        if self.rerank and len(surviving) > top_k:
            # 풀 20위 곡이 리랭킹으로 1위가 되는 전형적인 경우.
            shown = [surviving[19], *surviving[: top_k - 1]]

        return [
            MatchingTrack(id=song_id, score=1.0, title=f"곡{song_id}")
            for song_id in shown
        ]


class _FakeAnalyzer:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def analyze(self, query: str) -> QueryAnalysis:
        self.calls.append(query)
        return _analysis(query)


def _wire(searcher: _FakeSearchRouter):
    analyzer = _FakeAnalyzer()

    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: analyzer
    app.dependency_overrides[get_search_router] = lambda: searcher

    with TestClient(app) as client:
        yield client, analyzer, searcher


@pytest.fixture()
def wired():
    yield from _wire(_FakeSearchRouter())


@pytest.fixture()
def wired_reranked():
    """리랭킹이 켜진 실서비스 기본값을 흉내 낸 배선."""
    yield from _wire(_FakeSearchRouter(rerank=True))


def test_plain_search_is_untouched(wired) -> None:
    """피드백이 없으면 제외도 없고 턴도 1이다."""
    client, analyzer, searcher = wired

    body = client.post("/search", json={"query": "비 오는 날 노래"}).json()

    assert analyzer.calls == ["비 오는 날 노래"]
    assert searcher.calls[0]["exclude_ids"] == []
    assert body["turn"] == 1
    assert body["rejected_ids"] == []
    assert len(body["results"]) == 10


def test_rejected_ids_reach_the_router(wired) -> None:
    client, _, searcher = wired
    rejected = [f"s{i:03d}" for i in range(10)]

    body = client.post(
        "/search",
        json={"query": "비 오는 날 노래", "rejected_ids": rejected},
    ).json()

    assert searcher.calls[0]["exclude_ids"] == rejected
    returned = [r["id"] for r in body["results"]]
    assert not set(returned) & set(rejected)
    # 거절한 10곡 다음 순위가 올라온다 — Reject-only의 전부다.
    assert returned[0] == "s010"
    assert body["rejected_ids"] == rejected


def test_prior_analysis_skips_the_analyzer(wired) -> None:
    """왕복의 목적. 2턴에서 Gemini를 다시 부르면 안 된다."""
    client, analyzer, _ = wired

    body = client.post(
        "/search",
        json={
            "query": "비 오는 날 노래",
            "prior_analysis": _analysis().model_dump(),
            "rejected_ids": ["s000"],
        },
    ).json()

    assert analyzer.calls == []
    assert body["turn"] == 2


def test_candidate_ids_are_wider_than_results(wired) -> None:
    client, _, _ = wired
    body = client.post("/search", json={"query": "비 오는 날 노래"}).json()

    # results(10)보다 후보 풀(30)이 넓어야 다음 턴에 승격시킬 곡이 남는다.
    assert len(body["candidate_ids"]) == 50
    assert {r["id"] for r in body["results"]} <= set(body["candidate_ids"])


def test_candidate_ids_are_pre_rerank_order(wired_reranked) -> None:
    """candidate_ids[:10]은 '사용자에게 보여준 10곡'이 아니다.

    candidate_ids는 리랭킹 전 후보 풀 순서고, results는 리랭킹 결과다.
    단계 4에서 '남은 11위 이하'를 candidate_ids[10:]로 계산하면 이미 보여준
    곡이 섞여 들어간다. 보여준 곡은 rejected_ids로만 알 수 있다.
    """
    client, _, _ = wired_reranked
    body = client.post("/search", json={"query": "비 오는 날 노래"}).json()

    shown = [r["id"] for r in body["results"]]
    pool = body["candidate_ids"]

    assert shown != pool[:10]
    # 리랭킹으로 올라온 곡은 풀에서는 여전히 11위 이하 자리에 있다.
    assert shown[0] in pool[10:]


def test_answers_reach_the_router_without_touching_the_analysis(wired) -> None:
    """답변은 분석에 병합하지 않고 따로 넘긴다.

    병합하면 analysis.genre가 보조 검색 경로의 sparse 질의로 흘러가 맞는
    답변도 순위를 해친다(q200: 후보 4위 → 15위). 답변은 후보 재정렬로만
    반영한다. 응답의 analysis도 병합되지 않은 원본이어야 한다 — 병합본을
    돌려주면 클라이언트가 다음 턴에 되돌려줘 같은 문제가 재발한다.
    """
    client, _, searcher = wired

    body = client.post(
        "/search",
        json={
            "query": "비 오는 날 노래",
            "prior_analysis": _analysis().model_dump(),
            "asked_slots": ["vocal_gender"],
            "answers": [
                {"slot": "vocal_gender", "value": "여성"},
                {"slot": "type", "value": "솔로"},
            ],
        },
    ).json()

    passed = searcher.calls[0]["analysis"]
    assert passed.vocal_gender is None, "답변이 분석에 병합됐다"
    assert passed.artist_type.values == []

    answers = searcher.calls[0]["answers"]
    assert [a.slot for a in answers] == ["vocal_gender", "type"]
    assert [a.value for a in answers] == ["여성", "솔로"]

    assert body["analysis"]["vocal_gender"] is None
    assert body["asked_slots"] == ["vocal_gender", "type"]


# ---------------------------------------------------------------------------
# 종료 조건 — 답할 수 없는 질문을 주지 않는다
# ---------------------------------------------------------------------------

def test_a_question_is_attached_when_the_budget_allows(wired) -> None:
    """아래 종료 조건 테스트들이 무의미해지지 않도록, 먼저 '질문이 나온다'를 고정한다."""
    client, _, _ = wired
    body = client.post("/search", json={
        "query": "비 오는 날 노래",
        "prior_analysis": _analysis().model_dump(),
        "rejected_ids": ["s000"],
    }).json()

    assert body["clarify"] is not None
    assert body["clarify"]["slot"] == "vocal_gender"


def test_no_question_once_the_rejection_budget_is_spent(wired) -> None:
    """거절 한도를 다 쓴 뒤에는 질문을 주지 않는다.

    회귀 이력: 종료 판정이 물어본 슬롯 수만 봤다. 질문 없이 거절만 한 턴이
    섞이면 20곡을 거절한 상태에서도 질문이 나갔고, 사용자가 답하며 다음
    페이지를 거절하는 순간 rejected_ids가 30개가 되어 요청이 422로 막혔다.
    답할 수 없는 질문을 보여주는 셈이다.
    """
    client, _, _ = wired
    rejected = [f"s{i:03d}" for i in range(20)]

    body = client.post("/search", json={
        "query": "비 오는 날 노래",
        "prior_analysis": _analysis().model_dump(),
        "rejected_ids": rejected,
    }).json()

    assert body["clarify"] is None


def test_no_question_after_the_last_turn(wired) -> None:
    """2턴 확정 — 그 뒤로는 묻지 않는다."""
    from src.backend.schemas.search import MAX_TURNS
    client, _, _ = wired

    body = client.post("/search", json={
        "query": "비 오는 날 노래",
        "prior_analysis": _analysis().model_dump(),
        "rejected_ids": ["s000"],
        "turn": MAX_TURNS + 1,
    }).json()

    assert body["turn"] > MAX_TURNS
    assert body["clarify"] is None


def test_termination_check_covers_all_three_limits() -> None:
    from src.backend.api.routes.search import _can_ask_another
    from src.backend.schemas.search import MAX_ASKED_SLOTS, MAX_REJECTED_IDS

    assert _can_ask_another(turn=1, asked_slots=[], rejected_ids=[], top_k=10)
    # 슬롯을 다 물었다
    assert not _can_ask_another(
        turn=1, asked_slots=["vocal_gender", "genre"][:MAX_ASKED_SLOTS],
        rejected_ids=[], top_k=10)
    # 2턴 응답까지는 질문을 싣는다 (그 질문은 3턴에 답한다)
    assert _can_ask_another(turn=2, asked_slots=[], rejected_ids=[], top_k=10)
    # 그 뒤로는 답할 턴이 없다
    assert not _can_ask_another(turn=3, asked_slots=[], rejected_ids=[], top_k=10)
    # 다음 거절을 담을 자리가 없다
    assert not _can_ask_another(
        turn=1, asked_slots=[],
        rejected_ids=[f"s{i}" for i in range(MAX_REJECTED_IDS - 5)], top_k=10)


def test_corrected_answer_reaches_the_exact_lyric_grouping() -> None:
    """정정한 답변이 리랭크의 가사 exact 그룹 정렬까지 닿아야 한다.

    회귀 이력: 후보 점수에는 답변 보너스가 반영되는데, 그 뒤 exact 가사 후보를
    성별·장르로 다시 묶는 자리는 **원래 분석**을 봤다. "남성"이라고 잘못
    기억했다가 "여성"으로 정정해도 남성 그룹이 먼저 와서, 보너스로 후보 1위가
    된 여성 곡이 최종 Top-10에서 통째로 밀려났다.
    """
    from src.backend.schemas.query import LyricClue
    from src.retrieval.search_router import SearchRouter

    class _PassThroughReranker:
        enabled = True
        class config:  # noqa: D106 - 테스트용 최소 스텁
            spread_ref = 0.0
            rerank_weight = 1.0

        def load(self): return self
        def rerank(self, query, candidates, top_k): return list(candidates)[:top_k]

    def _make_router():
        router = SearchRouter.__new__(SearchRouter)
        for attr in ("_text_svc", "_img_emb", "_audio_emb", "_img_idx", "_audio_idx"):
            setattr(router, attr, None)
        router._lyrics_svc = object()          # use_lyrics_surface 게이트 통과용
        router._reranker = _PassThroughReranker()
        router._pool = ThreadPoolExecutor(max_workers=2)

        def lyric_hits(analysis, top_k):
            # 성별만 다른 두 곡. 둘 다 가사 exact라 보호 그룹에 들어간다.
            return [
                MatchingTrack(id="male", score=1.0, title="남성곡",
                              vocal_gender="남성", lyric_match_type="exact",
                              lyric_match_score=1.0),
                MatchingTrack(id="female", score=0.9, title="여성곡",
                              vocal_gender="여성", lyric_match_type="exact",
                              lyric_match_score=1.0),
            ]

        router._search_text = lambda a, k: lyric_hits(a, k)
        router._search_lyrics = lyric_hits
        for path in ("_search_image", "_search_audio", "_search_balanced_semantic",
                     "_search_performance_clues", "_search_performance_metadata",
                     "_search_title_constrained", "_search_title_presence",
                     "_search_title_meaning"):
            setattr(router, path, lambda a, k: [])
        return router

    def _analysis_with_lyrics():
        a = _analysis()
        a.vocal_gender = "남성"                  # 사용자가 처음 잘못 말한 값
        a.lyric_clues = [LyricClue(text="너를 사랑해", kind="verbatim")]
        return a

    def _shown(answers):
        router = _make_router()
        try:
            return [t.id for t in asyncio.run(router.search(
                _analysis_with_lyrics(), top_k=2, candidate_k=10,
                use_rerank=True, answers=answers,
            ))]
        finally:
            router.shutdown()

    assert _shown([])[0] == "male", "정정 전에는 남성이 먼저여야 한다"
    corrected = _shown([ClarifyAnswer(slot="vocal_gender", value="여성")])
    assert corrected[0] == "female", f"정정이 그룹 정렬에 반영되지 않았다: {corrected}"


def test_search_forwards_answers_to_answer_aware_reranker() -> None:
    """라우터가 재질문 답변을 리랭커까지 넘겨야 한다 — 일반 경로와 가사 exact 경로 모두.

    회귀 이력: 리랭커 호출이 최초 질의만 넘겨서, Gemini listwise가 정정된 답을
    모른 채 질의의 틀린 성별을 따라 정답을 다시 내렸다(dev c705).
    """
    from src.backend.schemas.query import LyricClue
    from src.retrieval.search_router import SearchRouter

    class _RecordingReranker:
        enabled = True
        uses_clarify_answers = True

        class config:  # noqa: D106 - 테스트용 최소 스텁
            spread_ref = 0.0
            rerank_weight = 1.0

        def __init__(self):
            self.received = []

        def load(self): return self

        def rerank(self, query, candidates, top_k, answers=None):
            self.received.append(answers)
            return list(candidates)[:top_k]

    def _run(exact: bool):
        router = SearchRouter.__new__(SearchRouter)
        for attr in ("_text_svc", "_img_emb", "_audio_emb", "_img_idx", "_audio_idx"):
            setattr(router, attr, None)
        router._lyrics_svc = object()
        router._reranker = _RecordingReranker()
        router._pool = ThreadPoolExecutor(max_workers=2)

        def hits(analysis, top_k):
            kind = "exact" if exact else None
            score = 1.0 if exact else None
            return [
                MatchingTrack(id=f"s{i}", score=1.0 - i / 10, title=f"곡{i}",
                              lyric_match_type=kind, lyric_match_score=score)
                for i in range(4)
            ]

        router._search_text = lambda a, k: hits(a, k)
        router._search_lyrics = hits
        for path in ("_search_image", "_search_audio", "_search_balanced_semantic",
                     "_search_performance_clues", "_search_performance_metadata",
                     "_search_title_constrained", "_search_title_presence",
                     "_search_title_meaning"):
            setattr(router, path, lambda a, k: [])
        analysis = _analysis()
        if exact:
            analysis.lyric_clues = [LyricClue(text="너를 사랑해", kind="verbatim")]
        answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
        try:
            asyncio.run(router.search(analysis, top_k=3, candidate_k=10,
                                      use_rerank=True, answers=answers))
        finally:
            router.shutdown()
        return router._reranker.received, answers

    for exact in (False, True):
        received, answers = _run(exact)
        assert received, f"리랭커가 호출되지 않았다 (exact={exact})"
        assert all(r == answers for r in received), f"답변이 전달되지 않았다: {received}"


def test_turn_increments_across_chained_rejections(wired) -> None:
    """Reject-only는 연속으로 눌릴 수 있다. 턴이 그대로면 종료 판정을 못 한다."""
    client, _, _ = wired

    body = client.post(
        "/search",
        json={
            "query": "비 오는 날 노래",
            "prior_analysis": _analysis().model_dump(),
            "rejected_ids": ["s000"],
            "turn": 2,
        },
    ).json()

    assert body["turn"] == 3
