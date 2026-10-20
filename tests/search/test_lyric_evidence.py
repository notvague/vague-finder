"""가사 원문 인용 — **되짚은 구간이 정말 그 곡의 그 문장인가.**

인용은 설명을 강하게 만든다. 그래서 틀리면 더 위험하다. 여기서 고정하는 것.

1. 인용이 원문과 **글자 단위로** 같다 (대소문자·띄어쓰기·줄바꿈까지)
2. 다른 곡의 문장이 섞이지 않는다
3. 되짚지 못하면 **비어 있다** — 그럴듯한 문장으로 채우지 않는다
4. 근거를 붙여도 순위가 바뀌지 않는다
5. 순위를 만든 근거와 뒤에 덧붙인 설명을 구분한다
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import (
    get_lyrics_exact_search_service,
    get_query_analyzer,
    get_search_router,
)
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.explain import EVIDENCE_RANKING, EVIDENCE_SUPPLEMENTARY
from src.backend.schemas.query import LyricClue, QueryAnalysis
from src.backend.schemas.search import LyricSurfaceMatch, MatchingTrack
from src.retrieval.lyrics_exact_search import LYRICS_FIELD, LyricsExactSearchService

LYRICS = (
    "모래처럼 흩어진 오후의 약속은\n"
    "창문을 여는 척 웃었고!\n"
    "I FOUND  the WAY, oh\n"
)
OTHER_LYRICS = "전혀 다른 노래의 가사다\n두 번째 줄\n"


def _service(**kw: Any) -> LyricsExactSearchService:
    docs = [
        {"song_id": "s1", "metadata": {"title": "가상 곡", "artist": "가수A"},
         "lyrics_data": {"full_lyrics": LYRICS}},
        {"song_id": "s2", "metadata": {"title": "다른 곡", "artist": "B"},
         "lyrics_data": {"full_lyrics": OTHER_LYRICS}},
    ]
    return LyricsExactSearchService(documents=docs, **kw)


# ---------------------------------------------------------------------------
# 되짚기
# ---------------------------------------------------------------------------

def _excerpt(service, song_id, phrase, snapshot=None):
    """실제 라우트와 같은 모양 — 판을 받아서 그 판에서만 인용한다."""
    return service.excerpt(song_id, phrase, snapshot or service.snapshot())


def test_quote_is_the_original_span_character_for_character():
    service = _service()
    found = _excerpt(service, "s1", "창문을여는척")
    assert found is not None
    assert found.quote == "창문을 여는 척"
    assert LYRICS[found.char_start:found.char_end] == found.quote


def test_quote_keeps_case_and_double_spaces():
    """정규화본으로는 되살릴 수 없는 것들 — 여기가 인용의 존재 이유다."""
    service = _service()
    found = _excerpt(service, "s1", "ifoundtheway")
    assert found is not None
    assert found.quote == "I FOUND  the WAY"


def test_line_context_contains_the_quote():
    service = _service()
    found = _excerpt(service, "s1", "창문을여는척")
    assert found is not None
    assert found.line == "창문을 여는 척 웃었고!"
    assert found.quote in found.line


def test_quote_never_comes_from_another_song():
    service = _service()
    assert _excerpt(service, "s2", "창문을여는척") is None
    assert _excerpt(service, "없는곡", "창문을여는척") is None
    assert _excerpt(service, "s1", "전혀다른노래의") is None


def test_phrase_that_is_not_contiguous_gets_no_quote():
    """fuzzy 일치가 이렇다 — 3-gram 겹침으로 뽑혀 이어지는 구간이 원문에 없다."""
    service = _service()
    assert _excerpt(service, "s1", "모래처럼웃었고") is None


def test_source_and_version_are_recorded():
    service = _service()
    found = _excerpt(service, "s1", "창문을여는척")
    assert found is not None
    assert found.source == LYRICS_FIELD
    assert len(found.data_version) == 16


def test_version_changes_when_the_lyrics_change():
    service = _service()
    before = _excerpt(service, "s1", "창문을여는척")
    changed = LyricsExactSearchService(documents=[
        {"song_id": "s1", "metadata": {"title": "가상 곡"},
         "lyrics_data": {"full_lyrics": LYRICS.replace("창문을", "창문도")}},
    ])
    after = _excerpt(changed, "s1", "여는척웃었고")
    assert before is not None and after is not None
    assert before.data_version != after.data_version


# ---------------------------------------------------------------------------
# 판(版) 고정 — 인용은 **검색이 본 가사**에서만 뜬다
# ---------------------------------------------------------------------------

def _mutable_service(state) -> LyricsExactSearchService:
    """가사가 바뀔 수 있는 서비스. TTL 0이라 읽을 때마다 새 판이 된다."""
    service = LyricsExactSearchService(cache_ttl_seconds=0.0)
    service._source_documents = lambda: [{
        "song_id": "s1", "metadata": {"title": "가상 곡"},
        "lyrics_data": {"full_lyrics": state["lyrics"]},
    }]
    return service


def test_quote_stays_on_the_snapshot_the_search_used():
    """검색 뒤 가사가 바뀌어도 인용은 **검색이 본 판**을 가리켜야 한다.

    되돌리면 깨진다: `excerpt()`가 스스로 `_load()`를 부르던 판에서는 TTL이
    만료된 뒤 새 판을 읽어, 판 해시와 구절 위치가 **둘 다** 달라졌다.
    화면의 순위는 옛 가사로 만들어졌는데 인용은 새 가사를 보여 주는 셈이다.
    """
    before = "첫 줄은 이렇습니다\n창문을 여는 척 웃었고"
    after = "첫 줄이 통째로 바뀌었고 훨씬 길어졌습니다\n창문을 여는 척 웃었고"
    state = {"lyrics": before}
    service = _mutable_service(state)

    used: List[Any] = []
    service.search(
        [LyricClue(text="창문을 여는 척", kind="verbatim", confidence=1.0)],
        snapshot_out=used,
    )
    assert used, "검색이 쓴 판을 돌려주지 않았다"
    old_document = used[0].get("s1")

    state["lyrics"] = after          # 그 사이 가사가 바뀐다

    found = service.excerpt("s1", "창문을여는척", used[0])
    assert found is not None
    assert found.data_version == old_document.lyrics_sha16, "다른 판을 인용했다"
    assert found.char_start == before.index("창문을"), "위치가 새 판 기준이다"
    assert before[found.char_start:found.char_end] == found.quote


def test_excerpt_never_reads_the_source_again():
    """인용은 **저장소를 건드리지 않는다.**

    Mongo 왕복은 측정 7.0초짜리 블로킹이다. 비동기 라우트에서 동기로 부르면
    이벤트 루프가 그만큼 멈춘다 — ⑥B가 질의 분석에서 없앤 문제와 같은 것이다.
    """
    state = {"lyrics": LYRICS}
    service = _mutable_service(state)
    used: List[Any] = []
    service.search(
        [LyricClue(text="창문을 여는 척", kind="verbatim", confidence=1.0)],
        snapshot_out=used,
    )

    def explode():
        raise AssertionError("인용이 저장소를 다시 읽었다")

    service._source_documents = explode
    assert service.excerpt("s1", "창문을여는척", used[0]) is not None


def test_refresh_does_not_disturb_a_snapshot_already_handed_out():
    """`refresh()`는 다음 검색부터 새 판을 읽게 할 뿐, 나간 판은 그대로 산다."""
    state = {"lyrics": LYRICS}
    service = _mutable_service(state)
    used: List[Any] = []
    service.search(
        [LyricClue(text="창문을 여는 척", kind="verbatim", confidence=1.0)],
        snapshot_out=used,
    )
    service.refresh()
    found = service.excerpt("s1", "창문을여는척", used[0])
    assert found is not None and found.quote == "창문을 여는 척"


# ---------------------------------------------------------------------------
# 응답까지
# ---------------------------------------------------------------------------

def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="가사에 창문을 여는 척 웃었고가 나오는 노래",
        intent_type="lyrics",
        image_english_query="",
        audio_english_query="",
        confidence=0.9,
    )


def _track(song_id: str) -> MatchingTrack:
    return MatchingTrack(
        id=song_id, score=0.9, title=f"곡{song_id}",
        lyric_match_type="exact",
        lyric_match_score=1.0,
        lyric_match_detail=LyricSurfaceMatch(
            clue_text="창문을 여는 척",
            clue_kind="verbatim",
            matched_phrase="창문을여는척",
            normalized_length=6,
            match_type="exact",
            confidence=1.0,
            corpus_match_count=1,
        ),
    )


class _Router:
    """가사 경로가 순위를 만든 곡과, 기록만 남은 곡을 하나씩 낸다.

    실제 라우터처럼 **이 검색이 본 가사 판**을 `lyrics_snapshot_out`으로 넘긴다.
    넘기지 않으면 라우트는 인용을 비운다 — 판 없는 인용은 근거가 아니기 때문이다.
    """

    def __init__(self, service, ranking: bool = True, hand_snapshot: bool = True) -> None:
        self.service = service
        self.ranking = ranking
        self.hand_snapshot = hand_snapshot

    async def search(self, analysis, **kw) -> List[MatchingTrack]:
        recorder = kw.get("recorder")
        tracks = [_track("s1")]
        for name in ("candidate_ids_out", "candidate_tracks_out"):
            out = kw.get(name)
            if out is not None:
                out.extend(t.id if name.endswith("ids_out") else t for t in tracks)
        snapshot_out = kw.get("lyrics_snapshot_out")
        if snapshot_out is not None and self.hand_snapshot:
            snapshot_out.append(self.service.snapshot())
        if recorder is not None:
            recorder.set_weights(1.0, 0.0, 0.0)
            recorder.path("s1", "text_hybrid", 1, 0.016)
            # ranking=False는 "가사가 걸렸지만 점수에는 0을 더했다" — 부스트를
            # 꺼 둔 대조 실험(LYRIC_BOOST_SCALE=0)의 모양이다.
            recorder.path("s1", "lyrics_surface", 1, 0.13 if self.ranking else 0.0)
            # 실제 라우터와 같은 자리에서 남긴다. 인용은 결과 트랙이 아니라
            # **기록**에서 읽으므로, 여기가 비면 인용도 없다.
            recorder.note_lyric_match(
                "s1", tracks[0].lyric_match_detail.model_dump()
            )
            recorder.set_rank_before(["s1"])
            recorder.set_rank_after(["s1"])
        return tracks


def _post(ranking: bool = True, hand_snapshot: bool = True, **body: Any) -> Dict[str, Any]:
    # 라우터와 라우트가 **같은 서비스**를 봐야 판이 오간다(실서비스도 싱글턴이다).
    service = _service()
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: type(
        "A", (), {"analyze": staticmethod(lambda q: _analysis())}
    )()
    app.dependency_overrides[get_search_router] = lambda: _Router(
        service, ranking, hand_snapshot
    )
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: service
    with TestClient(app) as client:
        payload = {"query": _analysis().original_query, "explain": True, **body}
        response = client.post("/search", json=payload)
        assert response.status_code == 200, response.text
        return response.json()


def test_response_carries_the_quote():
    evidence = _post()["results"][0]["explain"]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["quote"] == "창문을 여는 척"
    assert evidence[0]["song_id"] == "s1"
    assert evidence[0]["source"] == LYRICS_FIELD


def test_ranking_evidence_is_marked_as_such():
    evidence = _post(ranking=True)["results"][0]["explain"]["evidence"]
    assert evidence[0]["kind"] == EVIDENCE_RANKING
    assert evidence[0]["kind_label"]


def test_zero_boost_is_not_called_a_ranking_reason():
    """가산점이 0이면 순위를 만든 것이 아니다.

    `_fuse_add`는 더한 양이 0이어도 경로 기여를 남긴다. 경로가 기록됐다는 것만
    보고 '이 순위를 만든 근거'라고 하면, 후보 유입에만 쓰인 가사를 순위의 이유로
    말하게 된다(LYRIC_BOOST_SCALE=0 대조 실험).
    """
    evidence = _post(ranking=False)["results"][0]["explain"]["evidence"]
    assert evidence, "인용 자체는 나와야 한다 — 사실이긴 하다"
    assert evidence[0]["kind"] == EVIDENCE_SUPPLEMENTARY


def test_lyric_protection_alone_counts_as_a_ranking_reason():
    """점수가 0이어도 보호 규칙이 자리를 바꿨으면 순위는 가사가 만든 것이다."""
    from src.retrieval.explain import LYRIC_ORDER_RULES
    from src.backend.schemas.explain import evidence_kind
    from src.retrieval.explain import ExplainRecorder

    rec = ExplainRecorder("q")
    rec.path("s1", "lyrics_surface", 1, 0.0)
    rec.order("s1", sorted(LYRIC_ORDER_RULES)[0])
    assert evidence_kind(rec.record.get("s1")) == EVIDENCE_RANKING


def test_no_snapshot_means_no_quote():
    """검색이 쓴 판을 확보하지 못하면 인용하지 않는다.

    그 판이 아닌 곳에서 뜬 인용은 화면의 순위를 만든 가사가 아닐 수 있다.
    """
    evidence = _post(hand_snapshot=False)["results"][0]["explain"]["evidence"]
    assert evidence == []


def test_evidence_does_not_change_the_order():
    """근거를 붙여도 순위가 달라지면 안 된다 — 설명이 곧 랭킹 변경이 된다."""
    with_evidence = [r["id"] for r in _post(explain=True)["results"]]
    without = [r["id"] for r in _post(explain=False)["results"]]
    assert with_evidence == without
