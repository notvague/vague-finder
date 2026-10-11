import asyncio
import logging
import time

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.backend.api.dependencies import (
    get_lyrics_exact_search_service,
    get_query_analyzer,
    get_search_router,
    get_vector_client,
)
from src.backend.schemas.explain import SearchExplainOut, to_track_explain
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import (
    MAX_ASKED_SLOTS,
    MAX_PREVIOUS_CANDIDATES,
    MAX_REJECTED_IDS,
    MAX_TURNS,
    ClarifyRequest,
    ClarifyResponse,
    MatchingTrack,
    SearchRequest,
    SearchResponse,
)
from src.retrieval import timing
from src.retrieval.analysis_cache import looks_like_fallback
from src.retrieval.clarify import (
    ALLOWED_SLOTS,
    BIRTH_YEAR_SLOT,
    QUESTION_FIELDS,
    build_birth_year_question,
    pick_data_question,
    pick_question,
    with_birth_year,
)
from src.retrieval.explain import NULL_RECORDER, ExplainRecorder, SongExplain
from src.retrieval.lyrics_exact_search import (
    LyricsExactSearchService,
    LyricsSnapshot,
)
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback
from src.retrieval.search_router import SearchRouter
from src.retrieval.search_service import SearchService
from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_search(
    request: ClarifyRequest,
    vector_client: Any = Depends(get_vector_client),
) -> ClarifyResponse:
    """현재 후보로 데이터 질문(성별·장르)만 고른다. 검색·분석·리랭킹·턴 갱신은 하지 않는다.

    **연령대를 여기서 다시 계산하지 않는다.** 출생 연도는 /search가 시기 창으로 바꿔 후보를 다시 찾을 때
    반영되고, 화면은 그 응답의 candidate_ids를 그대로 보낸다 — 질문은 그 최신 후보가 어떻게 갈리는지에서
    나온다. 출생 연도 질문 자체는 결과와 함께 /search가 이미 냈으므로 여기서는 내지 않는다.

    물을 수 없는 상태(한도 소진·남은 후보 없음)면 DB를 읽지 않고 None을 돌려준다.
    """
    rejected = set(request.rejected_ids)
    shown = [s for s in dict.fromkeys(request.shown_ids) if s not in rejected]
    # 화면이 다음에 거절할 곡은 지금 보이는 곡이다. /search가 clarify_deferred를 정할 때와 같은 판정.
    if not _can_ask_data_question(request.turn, request.asked_slots, list(rejected), len(shown)):
        return ClarifyResponse()
    excluded = rejected | set(shown)
    ids = [s for s in dict.fromkeys(request.previous_candidate_ids) if s not in excluded]
    if not ids:
        return ClarifyResponse()
    try:
        index = vector_client.Index(TEXT_HYBRID_INDEX_NAME)
        result = await asyncio.to_thread(index.fetch, ids, fields=QUESTION_FIELDS)
        candidates = [SearchService.track_from_match(m) for m in result["matches"]]
        return ClarifyResponse(clarify=pick_data_question(candidates, request.asked_slots))
    except Exception as exc:
        logger.exception("[clarify] 질문 조회 실패")
        raise HTTPException(status_code=500, detail="추가 질문을 준비하지 못했습니다.") from exc


def _can_ask_another(
    turn: int,
    asked_slots: list,
    rejected_ids: list,
    top_k: int,
) -> bool:
    """질문을 하나 더 던져도 되는 상황인가.

    슬롯 수만 보면 안 된다. 질문 없이 거절만 한 턴이 섞이면 거절 한도를 다
    쓴 뒤에도 질문이 나가고, 사용자가 그 질문에 답하며 다음 페이지를 거절하는
    순간 rejected_ids가 상한을 넘어 요청이 422로 막힌다. 답할 수 없는 질문을
    보여주는 셈이다.

    세 가지를 모두 본다 — 물어본 슬롯 수, 턴 수, 그리고 다음 거절을 담을
    자리가 남았는지.
    """
    if len(asked_slots) >= MAX_ASKED_SLOTS:
        return False
    # 응답에 실은 질문은 *다음* 턴에 쓰인다. turn이 이미 마지막이면 그 질문에
    # 답할 턴이 없다. MAX_TURNS=2면 1턴·2턴 응답까지 질문을 싣고 3턴부터 멈춘다.
    if turn > MAX_TURNS:
        return False
    return len(rejected_ids) + top_k <= MAX_REJECTED_IDS


def _can_ask_data_question(
    turn: int,
    asked_slots: list,
    rejected_ids: list,
    next_rejections: int,
) -> bool:
    """데이터 질문(성별·장르)을 낼 수 있는 상태인가 — 지연 조회의 **양쪽이 함께 쓰는** 판정.

    /search는 이것으로 `clarify_deferred`(조회하러 올 가치가 있는가)를 정하고, /search/clarify는
    같은 값(직전 응답의 turn·asked_slots·rejected_ids, 지금 보이는 곡 수)으로 다시 본다. 판정이
    갈리면 화면이 조회하러 왔다가 빈손으로 돌아가거나, 물을 수 있는데 묻지 않게 된다.

    `_can_ask_another`에 더해 아직 묻지 않은 데이터 슬롯이 남았는지를 본다. 건너뛴 출생 연도는
    asked_slots에 들어 있어 슬롯 수에는 세지만(양쪽 같다) 데이터 슬롯은 아니다.
    """
    if all(slot in asked_slots for slot in ALLOWED_SLOTS):
        return False
    return _can_ask_another(turn, asked_slots, rejected_ids, next_rejections)


def _lyric_evidence(
    lyrics_service: LyricsExactSearchService,
    explain: SongExplain,
    snapshot: Optional[LyricsSnapshot],
) -> list:
    """이 곡의 가사 일치를 **원문 인용으로** 되짚는다.

    **검색을 다시 돌리지 않는다.** 이미 메모리에 있는 같은 가사 스냅샷에서 이 곡
    하나만 본다. 그래서 인용은 화면에 보이는 그 순위를 만든 실행과 같은 데이터에서
    나온다 — 다시 검색하면 질의 분석이 흔들려 다른 실행을 설명하게 된다(②의 이유).

    돌려보는 곡은 **응답에 실린 것뿐**이다. 가사 일치는 수백 곡에 걸릴 수 있는데
    (q301: 626곡), 그 전부를 되짚으면 보여주지도 않을 곡에 시간을 쓴다.

    **검색이 쓴 판에서만 뜬다.** 서비스에게 "지금 판"을 달라고 하면 안 된다 —
    그 사이 TTL이 만료되거나 다른 요청이 갱신하면 **검색에 쓴 가사와 다른 판**을
    인용한다. 게다가 그 적재는 Mongo 왕복(측정 7.0초)이라 여기서 부르면 이벤트
    루프까지 막는다. 판은 검색 단계에서 받아 둔 것을 그대로 넘긴다.

    **기록에서 읽는다.** 결과 트랙의 `lyric_match_detail`을 쓰면 안 된다 — 가사
    경로가 찾은 트랙과 최종 결과 트랙이 같은 객체가 아니라서, 가사로 1위가 된 곡에도
    그 칸이 비어 있다(732519 '가시'가 그랬다). 기록은 가사 경로가 실제로 무엇을
    맞혔는지를 곡 id로 들고 있다.

    되짚지 못하면 빈 목록이다. fuzzy 일치가 대표적이다 — 3-gram 겹침으로 뽑힌
    것이라 이어지는 한 구간이 원문에 없을 수 있다.
    """
    phrase = (explain.lyric_match or {}).get("matched_phrase")
    if not phrase or snapshot is None:
        return []
    try:
        found = lyrics_service.excerpt(explain.song_id, phrase, snapshot)
    except Exception as exc:  # 근거를 못 만든다고 검색 응답이 실패하면 안 된다
        logger.warning("[evidence] %s 인용 실패: %s", explain.song_id, exc)
        return []
    return [found] if found is not None else []


# 분석기가 스스로 내려오지 못했을 때를 위한 여유. 분석기 안의 예산이 먼저
# 걸리는 편이 낫다 — 그쪽이 로그를 남기고 규칙 폴백으로 내려오기 때문이다.
# 이 값은 그게 안 될 때의 **벽시계 상한**이다.
ANALYSIS_DEADLINE_MARGIN_SECONDS = 2.0


async def _analyze(query_analyzer: Any, query: str) -> QueryAnalysis:
    """질의 분석을 **벽시계로** 묶는다.

    분석기 안의 예산만으로는 부족하다. 그 값은 라이브러리를 거쳐 httpx의
    timeout이 되는데, httpx는 연결·읽기 같은 **단계마다** 따로 세고 읽기는
    조각 하나를 기다리는 시간이다. 응답이 조금씩 도착하면 예산을 넘겨 끝난다
    (0.1초 제한에 0.68초로 확인). 재시도 사이의 deadline 검사도 *진행 중인*
    호출은 끊지 못한다.

    그래서 여기서 한 번 더 묶는다. 시간이 다 되면 규칙 분석으로 검색을 계속한다 —
    분석이 늦다고 검색을 통째로 버릴 이유는 없다.

    **남은 일의 처리.** `analyze_async`를 가진 분석기(실물)는 취소가 HTTP 호출까지
    전파돼 연결이 닫힌다. 동기 분석기(시험용 가짜)는 스레드로 옮기므로 루프는
    막지 않지만, 시간이 다 되면 그 스레드는 끝까지 돌고 **결과만 버려진다** —
    멈출 방법이 없어서다. 실물이 비동기인 이유가 이것이다.
    """
    budget = getattr(query_analyzer, "analysis_budget_seconds", 20.0)
    run_async = getattr(query_analyzer, "analyze_async", None)
    pending = (
        run_async(query)
        if run_async is not None
        else asyncio.to_thread(query_analyzer.analyze, query)
    )
    try:
        return await asyncio.wait_for(
            pending, timeout=budget + ANALYSIS_DEADLINE_MARGIN_SECONDS
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "[clarify] 질의 분석이 %.0f초 안에 끝나지 않았다 → 규칙 분석으로 검색한다: %r",
            budget + ANALYSIS_DEADLINE_MARGIN_SECONDS,
            query,
        )
        return rule_fallback(query)


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    http_request: Request,
    query_analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    search_router: SearchRouter = Depends(get_search_router),
    lyrics_service: LyricsExactSearchService = Depends(
        get_lyrics_exact_search_service
    ),
):
    """자연어 질의 분석 → 멀티모달 검색 → RRF → 선택적 리랭킹.

    2턴 재질문: prior_analysis가 오면 질의를 다시 분석하지 않고 그 분석에
    답변을 병합해 재검색한다. 세션 저장소 없이 클라이언트가 상태를 왕복시킨다.

    Reject-only: rejected_ids만 보내도 동작한다. 답변 없이 거절한 곡을 후보
    풀에서 빼는 것만으로 다음 순위가 올라오며, 이것이 재질문 성능의 비교
    기준선이다. 재질문은 이것보다 나아야 의미가 있다.

    explain=true: 각 결과에 선정 근거를 실어 보낸다. **다시 계산하지 않고 이번
    실행의 기록을 그대로 담는다.** 클릭 시점에 재검색하면 질의 분석이 비결정적이라
    (같은 질의에서 텍스트 경로 순위가 4·7·19위로 달랐다) 화면에 보이는 순위와
    다른 실행을 설명하게 된다.
    """
    # 미들웨어가 없는 자리(테스트에서 라우트를 직접 부를 때)에서는 빈 그릇이 온다.
    timer = getattr(http_request.state, "timing", timing.NULL_TIMER)
    timer.set_query(request.query)
    timer.note(top_k=request.top_k, explain=request.explain)
    # 의존성 해결이 끝난 시각. 미들웨어가 잰 전체에서 이것을 빼면 프레임워크가
    # 쓴 시간(의존성 초기화·요청 검증·응답 직렬화)이 남는다.
    route_started = time.perf_counter()

    try:
        carries_state = bool(
            request.prior_analysis is not None
            or request.rejected_ids
            or request.answers
        )

        if request.prior_analysis is not None:
            # 재분석하면 1턴에서 얻은 단서가 사라지고 Gemini 호출도 낭비된다.
            #
            # 답변은 여기서 병합하지 않는다. 분석에 넣으면 analysis.genre가 보조
            # 검색 경로의 sparse 질의로 흘러가 맞는 답변도 순위를 해친다
            # (q200: 후보 4위 → 15위). 답변은 search()에 따로 넘겨 후보 재정렬로만
            # 반영하고, 응답의 analysis는 Gemini가 말한 그대로 돌려보낸다 —
            # 병합본을 돌려주면 클라이언트가 다음 턴에 그걸 되돌려줘 같은 문제가 재발한다.
            analysis = request.prior_analysis.model_copy(deep=True)
            timer.note(analysis_mode="prior")
        else:
            if request.rejected_ids:
                # 막지는 않는다. 다만 왕복의 목적이 이 호출을 없애는 것이라
                # 조용히 넘어가면 낭비를 눈치채지 못한다.
                logger.warning(
                    "[clarify] prior_analysis 없이 rejected_ids %d개가 왔다 — "
                    "질의를 다시 분석한다(Gemini 호출 낭비)",
                    len(request.rejected_ids),
                )
            # **루프를 잡지 않는다.** 분석은 Gemini 왕복(중앙값 2.6초)에 더해
            # 실패 시 재시도 사이에 1초를 기다린다. 루프에서 동기로 부르면
            # 그동안 이 프로세스가 다른 요청을 아예 받지 못한다 — 2026-09-23
            # 측정에서 동시 요청 2건 중 뒤 요청은 앞 요청의 이 구간이 끝난 뒤에야
            # (2.4초) 라우트에 들어왔다.
            #
            # `_analyze`가 비동기 경로를 쓰고, 동기 분석기는 스레드로 옮긴다.
            # 벽시계 상한도 거기서 건다.
            with timer.step("analysis", mode="gemini"):
                analysis = await _analyze(query_analyzer, request.query)

        # 턴은 거절이나 답이 있을 때만 는다. prior_analysis만 보내는 재검색(출생 연도 답 뒤 화면이 같은 질의를
        # 기존 분석 + 프로필로 다시 찾는 것)은 재분석만 아끼는 것이지 대화가 진행된 게 아니다
        turn = request.turn + 1 if (request.rejected_ids or request.answers) else (request.turn if carries_state else 1)

        # 브라우저에 저장된 출생 연도(프로필). 생애 단계 질의("중학교 때")에 시기가 없을 때만 창을 만든다.
        # 검색과 질문 선택에는 창을 넣은 사본을 쓰고, **응답의 analysis는 원본**이다 — 클라이언트가 다음 턴에
        # 되돌려주는 분석에 프로필이 굳어 들어가지 않게(프로필을 지우면 바로 빠져야 한다).
        analysis_for_search = with_birth_year(analysis, request.birth_year)
        if analysis_for_search is not analysis:
            timer.note(birth_year_window=f"{analysis_for_search.release_era.start_year}-{analysis_for_search.release_era.end_year}")

        # 물어본 슬롯 누적 — 다음 턴에서 같은 것을 다시 묻지 않기 위해.
        asked_slots = list(request.asked_slots)
        for answer in request.answers:
            if answer.slot not in asked_slots:
                asked_slots.append(answer.slot)
        if len(asked_slots) > MAX_ASKED_SLOTS:
            # 넘으면 응답을 클라이언트가 그대로 되돌려줄 수 없다(요청 검증에서 걸린다).
            logger.warning("[clarify] asked_slots 초과 — 앞 %d개만 유지: %s",
                           MAX_ASKED_SLOTS, asked_slots)
            asked_slots = asked_slots[:MAX_ASKED_SLOTS]

        # 후보 풀을 받아올 통로. search()의 반환 타입을 바꾸지 않기 위한 것.
        candidate_ids: list[str] = []
        candidate_tracks: list[MatchingTrack] = []
        # 가사 경로가 **무슨 판을 봤는지**. 인용이 그 판에서만 뜨게 하려고 받는다.
        lyrics_snapshot: list[LyricsSnapshot] = []

        # 끄면 아무것도 기록하지 않는 null object가 들어간다. 라우터에 `if` 분기가
        # 없으므로 "기록을 켰을 때만 지나가는 코드"가 생기지 않는다.
        recorder = ExplainRecorder(request.query) if request.explain else NULL_RECORDER

        # 분석이 규칙 폴백이었는지를 남긴다. 쿼터가 터져도 검색은 그대로 돌아서
        # 결과만 봐서는 구분되지 않는다 — 기록에 없으면 화면이 말할 수 없다.
        #
        # confidence만 보지 않는 이유는 `looks_like_fallback`에 적혀 있다. 그 함수는
        # confidence > 0이면 바로 빠지므로, 비싼 비교는 폴백일 때만 돈다.
        if request.explain:
            recorder.set_analysis_fallback(looks_like_fallback(analysis))

        # 이 구간 안에서 시작된 검색 경로들이 이 구간의 자식으로 붙는다.
        with timer.step("search"):
            results = await search_router.search(
                analysis_for_search,
                top_k=request.top_k,
                use_rerank=request.use_rerank,
                candidate_k=request.candidate_k,
                exclude_ids=request.rejected_ids,
                candidate_ids_out=candidate_ids,
                candidate_tracks_out=candidate_tracks,
                lyrics_snapshot_out=lyrics_snapshot,
                answers=request.answers,
                recorder=recorder,
                timer=timer,
            )

        explain_summary = None
        if request.explain:
            with timer.step("explain"):
                # 단계는 요청 전체에서 파생한다. 곡별로 따로 판단하면 "리랭킹이 N위
                # 유지"처럼 하지 않은 일을 말하게 된다.
                stage = recorder.record.reorder_stage
                results = [
                    track.model_copy(
                        update={
                            "explain": to_track_explain(
                                song_explain,
                                reorder_stage=stage,
                                evidence=_lyric_evidence(
                                    lyrics_service,
                                    song_explain,
                                    lyrics_snapshot[0] if lyrics_snapshot else None,
                                ),
                            )
                        }
                    )
                    for track in results
                    for song_explain in [recorder.record.get(track.id)]
                ]
                explain_summary = SearchExplainOut.of(recorder.record)

        # 재질문은 두 방식이다.
        #
        # defer_clarify(화면): 데이터 질문은 여기서 계산하지 않는다 — 사용자가 "이 중에는 없어요"를 누를
        # 때 화면이 /search/clarify로 조회한다. 정답을 찾아 멈추면 그 계산은 아예 일어나지 않는다. 여기서는
        # 출생 연도만 결과와 함께 묻고(곡이 아니라 사용자에 대한 질문이라 거절을 기다리지 않는다),
        # 조회하러 올 가치가 있는지만 clarify_deferred로 알린다.
        #
        # 그 외(재질문 측정 `evaluate_clarification`과 같은 방식 · 직접 부르는 호출자): 질문을 지금 계산해
        # 응답에 싣는다. 질문은 보여준 곡을 뺀 나머지 후보로 만든다 — 이미 보여준 곡을 기준으로 물으면
        # 사용자가 아니라고 한 것들을 근거로 삼게 된다.
        with timer.step("clarify"):
            shown = {track.id for track in results}
            clarify = None
            clarify_deferred = False
            if request.defer_clarify:
                if (results and analysis_for_search.has_life_stage
                        and not analysis_for_search.has_release_era
                        and BIRTH_YEAR_SLOT not in asked_slots):
                    clarify = build_birth_year_question(analysis_for_search)
                # 화면은 candidate_ids의 앞 MAX_PREVIOUS_CANDIDATES개만 되돌려준다. 그 안에 보여준 곡 말고
                # 남은 것이 없으면 질문이 나올 수 없으니 조회하러 오지 않게 한다.
                has_remaining = any(
                    song_id not in shown for song_id in candidate_ids[:MAX_PREVIOUS_CANDIDATES]
                )
                clarify_deferred = (
                    bool(results)
                    and has_remaining
                    and _can_ask_data_question(turn, asked_slots, request.rejected_ids, len(results))
                )
            elif _can_ask_another(turn, asked_slots, request.rejected_ids, request.top_k):
                remaining = [t for t in candidate_tracks if t.id not in shown]
                clarify = pick_question(analysis_for_search, remaining, asked_slots)

        return SearchResponse(
            results=results,
            total_found=len(results),
            analysis=analysis,
            clarify=clarify,
            clarify_deferred=clarify_deferred,
            asked_slots=asked_slots,
            candidate_ids=candidate_ids,
            rejected_ids=list(request.rejected_ids),
            turn=turn,
            explain=explain_summary,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 서버 내부 오류가 발생했습니다: {exc}",
        ) from exc
    finally:
        # 실패해도 남긴다 — 느려서 끊긴 요청의 시간이 가장 궁금한 시간이다.
        timer.add("route", 0, route_started, route_started, time.perf_counter())
