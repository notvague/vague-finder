"""리랭킹 생략 정책(`rerank_policy.py`)과 설명 노출 제한 — 2026-10-11 사례.

사용자 질의 "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"에 리랭커가
"뮤직비디오에 옥상에서 의자를 던지는 장면이 나온다"는, 어떤 입력에도 없는 장면을 이유로 썼고
화면이 그것을 보여줬다. 이 파일이 고정하는 것은 두 가지다.

1. 모델이 쓴 문장은 어떤 confidence로 와도 일반 API·화면에 나가지 않는다. 내부 기록에는 남는다.
2. 구체적인 MV 장면 질의인데 장면을 판단할 자료가 없으면 Gemini를 **부르지 않고** 지금 후보 순서를 낸다.
   생략은 실패와 구분해 기록하고, 하지 않은 리랭킹의 점수·이유·이동을 적지 않는다.

모델·DB는 쓰지 않는다. 보호 경로의 리랭커는 호출되면 바로 실패하는 가짜다.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.retrieval.evaluate_search_accuracy as ev
from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.explain import to_track_explain
from src.backend.schemas.query import ModalityWeights
from src.backend.schemas.search import ClarifyAnswer
from src.retrieval.explain import (
    REORDER_MODEL_APPLIED,
    REORDER_NOT_ATTEMPTED,
    REORDER_POLICY_SKIPPED,
    REORDER_STAGE_LABELS,
    RERANK_APPLIED,
    RERANK_SKIP_IMAGE_DOMINANT,
    RERANK_SKIP_LABELS,
    RERANK_SKIP_MV_SCENE_UNVERIFIABLE,
    ExplainRecorder,
    RerankRun,
    ScoreMix,
    render_ko,
)
from src.retrieval.gemini_listwise_reranker import (
    GeminiListwiseReranker,
    GeminiListwiseRerankerConfig,
)
from src.retrieval.query_analyzer import rule_fallback
from src.retrieval.rerank_policy import (
    MV_SCENE_EVIDENCE_FIELDS,
    has_mv_scene_evidence,
    is_specific_mv_scene_query,
    rerank_skip_reason,
)
from src.retrieval.reranker import MusicReranker
from tests.test_search_explain import _analysis, _router, _track

USER_QUERY = "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"
FABRICATED = "뮤직비디오에 옥상에서 의자를 던지는 장면이 나온다"


# ---------------------------------------------------------------------------
# 질의 판정 — 지원 범위를 시험으로 고정한다
# ---------------------------------------------------------------------------

TARGETS = [
    # 사용자 사례와 바꿔 말한 것
    USER_QUERY,
    "의자가 하늘에 떠 있는 뮤비",
    "하늘에 의자 떠 있는 뮤비",                                   # 조사 없는 체언
    "뮤직비디오에서 의자를 던져",
    "MV에 사람이 거꾸로 매달린 장면",
    "mv에서 여자가 물속에 가라앉는 장면이 나와",
    "뮤비에 의자가 떠 있는 장면이 있는데 제목은 기억 안 나",      # 부정은 다른 화제(제목)의 것이다
    # 표지와 장면이 절을 나눠 갖는 문형
    "뮤비를 봤는데 의자가 하늘에 떠 있었어",
    "뮤비가 기억나는데 남자가 옥상에서 뛰어내려",
    "뮤직비디오가 인상적이었는데 주인공이 물에 빠지는 장면이 있었어",
    # 어미(고/만/데/며)로 끝나는 낱말 뒤 — 절을 자르지 않아야 산다
    "뮤비에서 우산 쓰고 걸어가는 여자",
    "뮤비에서 교복 입고 춤추는 노래",
    "뮤비에서 창고 안에 불이 타오르는 노래",
    "뮤비에 의자만 덩그러니 있어",
    # 활용형·낱말
    "뮤비에서 의자가 하늘에 둥둥 떠다녀",
    "뮤비에 하늘에 뜬 의자가 나와",
    "뮤비에서 의자가 공중부양해",
    "뮤직비디오 마지막에 차가 폭발해",
    "뮤비에서 남자랑 여자가 키스해",
    "뮤비에 고양이가 계속 따라다녀",
    "뮤비에서 남자가 달리고 여자가 울어",
    "뮤비가 흑백이고 비가 계속 내려",
    # 동작 없는 배경·소품도 구체적 시각 내용이다
    "뮤비 배경이 사막이야",
    "뮤비에 교실이 나오는 노래",
    "뮤비에 눈 오는 놀이공원이 배경인 노래",
    # 표기
    "M/V에서 가수가 피아노를 부숴",
    "엠브이에 풍선이 날아가는 장면",
    # 부정이 장면을 **제외하고** 말한 것 — 장면은 긍정이다
    "뮤비에서 의자를 던지는 장면 말고는 기억 안 나",
    "뮤비에 의자가 떠 있던 거 말고는 몰라",
    "뮤비에서 누군지 모르는 여자가 물에 빠져",
    # 다른 대상(제목·가수·연도·누가)의 부정·불확실은 판정에 쓰지 않는다
    "제목은 기억 안 나는데 뮤비에서 의자가 하늘에 떠 있었어",
    "가수는 모르겠고 뮤비에서 의자가 하늘에 떠 있었어",
    "누가 불렀는지 모르겠는데 뮤비에서 의자가 하늘에 떠 있었어",
    "뮤비에서 여자가 물에 빠지는데 그게 누구였는지 모르겠어",
    "몇 년도 노래인지 기억 안 나고 뮤직비디오에서 교복 입은 애들이 교실에서 춤춰",
    # 조사 뒤 첫 절이 장면을 담거나 이어지는 말
    "뮤비는 남자가 옥상에서 뛰어내리는 내용이야",
    # 여는 말 뒤에서는 시각 서술어만으로도 장면이다
    "뮤비 내내 비 맞으면서 노래 불러",
    "뮤비에서 편의점 앞에 앉아서 컵라면 먹어",
    "뮤비 마지막에 남자가 사라져",
    "뮤비에서 멤버들이 사막을 걸어가",
    "엠브이에서 다같이 버스 타고 어디 가",
    # 3차 — 새 표현·이어지는 말·다른 대상의 불확실
    "뮤비에서 주인공이 기차 타고 떠나",
    "뮤직비디오에 온통 노란 꽃밭이 나와",
    "뮤비 처음에 여자가 전화기 들고 울어",
    "뮤비에 로봇이 나오고 남자가 그걸 부숴",
    "옥상에서 다 같이 춤추는 뮤비",
    "남자가 물속으로 가라앉는 뮤직비디오였어",
    "뮤비를 보면 계속 눈이 내리고 있어",
    "뮤비는 봤는데 남자가 옥상에서 뛰어내려",                       # 부정 없는 '봤' → 뒤를 연다(아래 비대상 쌍과 대조)
    "뮤비에서 남자가 달리고 여자가 울어, 누가 불렀는지는 모르겠어",
    "뮤직비디오에서 의자를 던져, 누가 불렀는지는 모르겠어",
    "뮤비에 고래가 하늘을 날아다니는데 가수가 누군지 기억이 안 나",
    "뮤비에서 남자가 비를 맞고 있는데 그 뒤는 기억 안 나",
]

# 다른 대상의 불확실 꼬리 — TARGETS 뒤에 붙여도 대상이어야 한다.
TRAILING_UNCERTAINTIES = [
    lambda q: q + ", 누가 불렀는지는 모르겠어",
    lambda q: q + ", 언제 나왔는지 기억이 안 나",
    lambda q: q + " 근데 가수를 몰라",
]

# 뮤비 표지 앞에 붙는 흔한 말머리 — 다른 대상의 부정·불확실. 판정을 바꾸면 안 된다.
LEAD_INS = [
    "제목은 기억 안 나는데 ",
    "가수는 모르겠고 ",
    "누가 불렀는지 모르겠는데 ",
    "가수 이름은 생각이 안 나는데 ",
    "몇 년도 노래인지 기억 안 나고 ",
]

# 장면 없는 뮤비 언급 — 어떤 질의에 붙여도 대상이 되면 안 된다.
SCENELESS_MENTIONS = [
    lambda q: q.rstrip(" .?!") + ", 뮤비도 유명했어",
    lambda q: q.rstrip(" .?!") + "인데 뮤비 조회수가 높았어",
    lambda q: q.rstrip(" .?!") + " 뮤비는 조회수가 높았어",
    lambda q: q.rstrip(" .?!") + ". 뮤비도 유명했어",
    lambda q: "뮤비도 유명했어, " + q,
    lambda q: "뮤비가 화제였고 " + q,
    lambda q: "뮤비가 화제였던 곡이야, " + q,
    lambda q: "뮤비도 유명했어. " + q,
    # 3차 — 부정된 '봤', 구두점 없는 종결, 주제 조사 뒤의 표지
    lambda q: "뮤비는 안 찾아봤는데 " + q,
    lambda q: "뮤비도 유명했어 " + q,
    lambda q: q.rstrip(" .?!") + "는 뮤비도 유명했어",
    lambda q: q.rstrip(" .?!") + "은 뮤비가 화제였어",
]

NON_TARGETS = [
    ("뮤비가 유명한 노래", "MV 단어만 있고 장면이 없다"),
    ("전 세계에서 사람들이 따라 추는 춤이 유행했던 곡이야. 뮤직비디오가 코미디처럼 웃겼고 남자가 불렀어",
     "감성 묘사(dev c601)"),
    ("뮤비가 몽환적인 분위기였어", "감성 묘사"),
    ("의자가 떠 있는 것 같은 몽환적인 분위기", "뮤비 언급이 없다"),
    ("가사에 '뮤비에서 의자를 던져'라는 말이 있어", "가사 인용"),
    ("뮤비라는 말이 가사에 나오는 노래", "낱말 언급"),
    ("뮤비가 유명한 노래인데 가사에 바다가 나와", "장면은 가사 화제에 있다"),
    ("뮤비는 기억 안 나고 남자 발라드야", "뮤비 자체를 모름"),
    ("뮤비 못 봤는데 남자가 부른 힙합", "뮤비 자체를 모름"),
    ("뮤비에 의자 떠 있는 장면은 못 봤어", "부정 + 같은 절의 장면 — 보지 못했다"),
    ("뮤비는 기억 안 나는데 무대에서 남자가 의자 들고 춤췄어", "부정 + 같은 문장의 장면 — 뮤비가 아니다"),
    ("뮤비에서 의자가 떠 있었는지는 기억이 안 나", "장면이 불확실"),
    ("뮤비에 의자가 떠 있었나 잘 모르겠어", "장면이 불확실"),
    ("뮤비에 배우 김태리가 나와", "출연 사실은 장면이 아니다 — 구체 명사 없음, 범위 밖"),
    # 뮤비 밖의 설명(성별·장르·분위기·발매·다른 매체·무대)이 같은 문장에 있어도 장면이 아니다
    ("비 오는 날 듣기 좋은 여자 발라드인데 뮤비가 예뻤어", "뮤비 평가로 닫힘"),
    ("남자 그룹이고 뮤비가 유튜브에 있어", "뮤비 존재로 닫힘"),
    ("2010년쯤 나온 남자 아이돌 곡, 뮤비가 엄청 화제였어", "쉼표 앞 절은 다른 화제"),
    ("남자 솔로 가수 노래 중에 뮤비가 유명한 게 있어", "장면 없음"),
    ("하늘 높이 올라가는 고음이 유명한 남자 가수 노래고 뮤비도 유명해", "표지 앞 절은 관형절이 아니다"),
    ("드라마 키스 장면에 나와서 유명해졌고 뮤비도 화제였어", "다른 매체의 장면"),
    ("무대에서 남자가 의자 들고 춤췄는데 뮤비도 유명했어", "무대 장면"),
    ("여자 아이돌 데뷔곡인데 뮤비 찍다가 다쳤다는 기사가 났어", "제작 사실"),
    ("사람들이 노래방에서 많이 부르는 남자 발라드, 뮤비는 따로 없었던 걸로 알아", "뮤비 없음"),
    # 3차 — 부정된 '봤'은 뒤를 열지 않는다(위 대상 "뮤비는 봤는데 …"와 쌍)
    ("뮤비는 안 봤는데 남자가 옥상에서 뛰어내려", "부정 + 이어지는 말 → 뮤비 부정"),
    ("뮤비는 안 찾아봤는데 남자가 옥상에서 뛰어내리는 노래", "부정 + 이어지는 말"),
    ("뮤비는 안 보고 음원만 들었는데 남자가 옥상에서 뛰어내려", "부정 + 이어지는 말"),
    ("뮤비가 있었는지 모르겠는데 남자 아이돌이 교복 입고 춤추는 노래", "뮤비 존재 불확실 + 무대 장면"),
    # 주제 조사(는/은) 뒤의 표지는 관형절이 아니다
    ("비 오는 날 듣기 좋은 여자 발라드는 뮤비가 예뻤어", "주제 조사"),
    ("학교에서 많이 나오던 남자 발라드는 뮤비도 유명했어", "주제 조사"),
    ("남자가 무대에서 의자 들고 춤추는 노래는 뮤비도 유명해", "주제 조사 + 무대 장면"),
    # 약한 구조어(있는·나온·찍)와 가수 설명의 남자/여자
    ("뮤비가 유튜브에 있는 남자 아이돌 노래", "있는 = 존재, 남자 아이돌 = 가수"),
    ("뮤비도 있는 여자 솔로 곡", "있는 = 존재"),
    ("뮤비가 나온 지 10년 넘은 남자 발라드", "나온 = 발매"),
    ("뮤비 찍은 감독이 유명한 여자 아이돌 노래", "찍 = 제작"),
    ("뮤비에 나온 배우가 유명했던 남자 발라드", "출연 — 범위 밖"),
    ("뮤비 나오자마자 학교에서 다들 따라 불렀던 남자 아이돌 노래", "나오자마자 = 발매"),
    ("뮤비 보고 반해서 팬이 된 여자 가수 노래", "여자 가수 = 가수"),
    ("뮤비 속 여자 주인공이 예뻐서 화제였던 노래", "장면 없는 평가"),
    ("뮤비 조회수는 낮은데 길거리에서는 엄청 나오던 여자 솔로 노래", "닫힌 첫 절"),
    ("뮤비가 공개된 날 사람들이 서버 터졌다고 난리였어", "터졌 = 서버"),
    ("뮤비 댓글에 다들 울고 갔다고 써 있던 남자 발라드", "울고 = 댓글"),
    ("뮤비 없어도 차트에 계속 있는 여자 그룹 노래", "뮤비 없음"),
    # 여는 말 뒤라도 시각 서술어만으로는 장면이 아니다(구체 명사가 필요)
    ("뮤비에 돈을 엄청 쓰고 망했다는 노래", "쓰고 = 지출"),
    ("뮤비에서부터 화제가 돼서 차트에 올라가기 시작한 노래", "올라가 = 차트"),
    # 화제 전환 — 장면은 가사 화제에 있다
    ("뮤비에 대해서는 모르겠고 가사에 바다가 나와", "가사 앞에서 끊는다"),
    ("뮤비에서 여자가 옥상에서 뛰어내렸나 그랬는데 제목은 몰라", "장면이 불확실"),
    # 짧은 서술어 어간·조사 모양에 걸리던 오탐
    ("뮤비 조회수가 계속 올라가던 노래", "올라가 — 조회수"),
    ("뮤비 나온 지 얼마 안 된 노래가 있어", "얼(마)"),
    ("뮤비 얼굴 천재로 유명했던 아이돌이 부른 노래", "얼(굴) · 아이돌"),
    ("뮤비에 타고난 춤꾼이라는 댓글이 많았어", "타고(난)"),
    ("뮤비 댓글에 남달리 슬프다는 말이 많던 노래", "(남)달리"),
    ("뮤비가 유명해서 얼떨결에 알게 된 노래가 있어", "얼(떨결)"),
    ("뮤비 덕분에 사랑에 빠진 사람이 많았던 가수의 데뷔곡", "사랑에 빠진 — 관용구"),
    ("뮤비를 녹화해서 돌려 보던 노래가 있어", "녹(화)"),
    ("뮤비 공개하고 차트에서 떨어진 적이 없는 노래", "떨어진 — 차트"),
    ("MVP 받은 선수 응원가", "MVP는 MV가 아니다"),
    ("비 오는 날 발라드", "일반 감성"),
    ("남자 발라드", "일반 속성"),
    ("초록 바탕에 흰 도형이 있는 앨범 커버", "표지 질의"),
    ("2020년 초에 연예인들이 춤 영상 찍어 올리던 챌린지 노래", "영상은 MV가 아니다(dev n042)"),
    ("무슨 밴드 노래인데, 옛날에 짱구 애니메이션에서 누가 이별하는 되게 감성적인 장면에 OST로 삽입되어서 화제가 됐던 곡이야",
     "애니 장면은 MV가 아니다(dev q203)"),
]


@pytest.mark.parametrize("query", TARGETS)
def test_specific_mv_scene_queries_are_targets(query):
    assert is_specific_mv_scene_query(query), query


@pytest.mark.parametrize("query,why", NON_TARGETS)
def test_non_scene_mv_mentions_are_not_targets(query, why):
    assert not is_specific_mv_scene_query(query), f"{why}: {query}"


def _dev_queries():
    import csv
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "experiments" / "reranking"
    rows = []
    for name in ("eval_queries_v06.csv", "eval_queries_v09.csv", "eval_queries_v10.csv"):
        with (root / name).open(encoding="utf-8-sig", newline="") as fh:
            # dev 분할만 읽는다 — test 분할은 봉인이다
            rows.extend(r["query"] for r in csv.DictReader(fh) if r.get("split", "").lower() == "dev")
    assert len(rows) >= 170, "dev 질의 세트가 바뀌었다"
    return rows


def test_property_sceneless_mv_mentions_never_turn_dev_queries_into_targets():
    """오탐 성질: dev 176건 각각에 장면 없는 뮤비 언급을 앞·뒤(쉼표·연결어미·마침표·붙여쓰기)로 붙여도 대상 0건.

    표지 앞뒤 문장 전체를 장면 구간으로 보면 dev의 남자·여자·하늘·학교 같은 낱말이 장면이 된다(재리뷰 B).
    """
    rows = _dev_queries()
    assert not [q for q in rows if is_specific_mv_scene_query(q)], "dev 세트 자체에 대상이 생겼다 — CLAUDE.md 기준선 설명을 고쳐야 한다"
    for make in SCENELESS_MENTIONS:
        hits = [make(q) for q in rows if is_specific_mv_scene_query(make(q))]
        assert not hits, hits[:5]


def test_property_lead_in_negations_do_not_change_the_verdict():
    """누락 성질: 다른 대상의 부정 말머리를 붙여도 대상은 대상, 비대상은 비대상(재리뷰 A)."""
    for lead in LEAD_INS:
        missed = [lead + q for q in TARGETS if not is_specific_mv_scene_query(lead + q)]
        assert not missed, missed[:5]
        flipped = [lead + q for q, _ in NON_TARGETS if is_specific_mv_scene_query(lead + q)]
        assert not flipped, flipped[:5]


def test_property_trailing_uncertainty_about_other_things_keeps_targets():
    """누락 성질: 뒤에 붙은 다른 대상의 불확실(누가·언제·가수)은 앞 절의 장면을 지우지 않는다(3차 5)."""
    for make in TRAILING_UNCERTAINTIES:
        missed = [make(q) for q in TARGETS if not is_specific_mv_scene_query(make(q))]
        assert not missed, missed[:5]


def test_fallback_and_prior_analysis_share_the_same_verdict():
    """원문만 보므로 Gemini 분석·규칙 폴백·되돌려받은 분석(prior_analysis)에서 판정이 같다."""
    live = _analysis(original_query=USER_QUERY)
    fallback = rule_fallback(USER_QUERY)
    prior = live.model_copy(deep=True)
    gemini = GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x")
    for analysis in (live, fallback, prior):
        assert rerank_skip_reason(gemini, analysis, []) == RERANK_SKIP_MV_SCENE_UNVERIFIABLE


# ---------------------------------------------------------------------------
# 자료 — 무엇이 장면의 증거가 아닌가
# ---------------------------------------------------------------------------

def _upside_down_like():
    """사용자 사례 곡의 로컬 레코드 모양. 옥상은 **가사 요약**에, 의자는 어디에도 없다."""
    return _track(
        "602193305", title="Upside Down", artist="BOYNEXTDOOR", genre="댄스", vocal_gender="남성",
        lyrics_summary="꽉 막힌 세상 속에서 옥상 위로 올라가 수박을 먹는 등 자유를 갈구하는 청춘의 에너지를 노래한다.",
        lyrics_highlight="Upside down 돌려봐 뒤집은 청바지도 입고 엎어봐 upside down",
        visual_imagery=["교복을 입은 소년들이 옥상 위에서 수박을 먹으며 웃고 있는 모습"],
        youtube_url="https://www.youtube.com/watch?v=86NUEhixJlk",
        cover_url="https://cdnimg.melon.co.kr/x.jpg",
        search_style_summary="신나는 하이틴 댄스",
    )


def test_lyrics_imagery_cover_and_url_are_not_mv_scene_evidence():
    assert MV_SCENE_EVIDENCE_FIELDS == (), "장면 출처 필드를 추가하면 곡·판본·주장 연결 검사도 함께 넣는다"
    assert not has_mv_scene_evidence([_upside_down_like()])
    assert not has_mv_scene_evidence([])


def test_model_confidence_is_not_evidence():
    """모델이 support=0.99·verified=true를 말해도 정책 입력이 아니다 — 정책은 모델 출력을 받지 않는다."""
    gemini = GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x")
    analysis = _analysis(original_query=USER_QUERY)
    confident = _upside_down_like().model_copy(update={"rerank_score": 0.99})
    assert rerank_skip_reason(gemini, analysis, [confident]) == RERANK_SKIP_MV_SCENE_UNVERIFIABLE


# ---------------------------------------------------------------------------
# 백엔드 범위 — Gemini listwise만
# ---------------------------------------------------------------------------

def test_policy_applies_to_gemini_listwise_only():
    analysis = _analysis(original_query=USER_QUERY)
    gemini = GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x")
    assert rerank_skip_reason(gemini, analysis, []) == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert rerank_skip_reason(MusicReranker(), analysis, []) == ""
    assert rerank_skip_reason(None, analysis, []) == ""


def test_image_policy_still_wins_and_is_reported_as_its_own_reason():
    gemini = GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x")
    cover = _analysis(
        original_query="초록 바탕에 흰 도형이 있는 앨범 커버", has_visual_clue=True,
        image_english_query="green cover", modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0),
    )
    assert rerank_skip_reason(gemini, cover, []) == RERANK_SKIP_IMAGE_DOMINANT
    mixed = cover.model_copy(update={"modality_weights": ModalityWeights(text=0.5, image=0.5, audio=0.0)})
    assert rerank_skip_reason(gemini, mixed, []) == ""


def test_only_the_mv_skip_has_a_user_label():
    """표지 지배 생략은 기존 동작이라 안내 문구가 없다 — 진단 사유 코드로만 남는다."""
    assert set(RERANK_SKIP_LABELS) == {RERANK_SKIP_MV_SCENE_UNVERIFIABLE}
    assert "없다" not in RERANK_SKIP_LABELS[RERANK_SKIP_MV_SCENE_UNVERIFIABLE], "자료 없음을 장면 없음으로 말하지 않는다"
    assert REORDER_STAGE_LABELS[REORDER_POLICY_SKIPPED]


# ---------------------------------------------------------------------------
# 라우터 — 보호 경로는 모델을 부르지 않는다
# ---------------------------------------------------------------------------

class _ForbiddenReranker:
    """Gemini listwise의 정책 속성을 가진 가짜. 호출되면 곧바로 실패한다 — 보호 경로에서는 호출 자체가 없어야 한다."""

    enabled = True
    uses_clarify_answers = True
    skips_image_dominant = True
    skips_unverifiable_mv_scene = True

    def __init__(self):
        self.calls = 0

    def rerank_run(self, query, tracks, top_k, answers=None):
        self.calls += 1
        raise AssertionError("보호 경로에서 모델이 호출됐다")


class _CallingReranker(_ForbiddenReranker):
    """비대상 질의용. 호출을 세고, 순서를 뒤집어 '리랭킹이 있었다'를 보이게 하며, 허구 이유도 낸다."""

    def rerank_run(self, query, tracks, top_k, answers=None):
        self.calls += 1
        ids = [t.id for t in tracks]
        out = [t.model_copy(update={"rerank_score": 0.9}) for t in reversed(list(tracks))][:top_k]
        mix = ScoreMix(backend="gemini_listwise", weight=0.85, configured_weight=0.85,
                       rerank_component=0.9, retrieval_component=0.5, final=0.84, strategy="후보 전체 재정렬")
        return RerankRun(out, RERANK_APPLIED, ids, mixes={sid: mix for sid in ids},
                         model_notes={sid: [FABRICATED] for sid in ids})


def _hits():
    # 보너스(남성)가 s2·s4를 올린다 → 보너스 전 s1 s2 s3 s4, 보너스 뒤 s2 s4 s1 s3
    return [
        _track("s1", score=0.0160, vocal_gender="여성"),
        _track("s2", score=0.0159, vocal_gender="남성"),
        _track("s3", score=0.0158, vocal_gender="여성"),
        _track("s4", score=0.0157, vocal_gender="남성"),
    ]


def _search(router, analysis, **kw):
    try:
        return asyncio.run(router.search(analysis, **kw))
    finally:
        router.shutdown()


def test_router_skips_gemini_and_keeps_the_corrected_candidate_order(monkeypatch):
    """사용자 사례 + 재질문 답변 + 거절. 모델 호출 0회, 최종은 **보너스·거절 반영 후보 순서**다(보정 전이 아니다)."""
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")   # 입력이 보정 전 순서여도 결과는 보정 후여야 한다
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    pool, skips = [], []
    out = _search(
        _router(_hits(), reranker=rr), _analysis(original_query=USER_QUERY),
        top_k=4, candidate_k=4, answers=[ClarifyAnswer(slot="vocal_gender", value="남성")],
        exclude_ids=["s3"], candidate_ids_out=pool, rerank_skip_out=skips, recorder=rec,
    )
    assert rr.calls == 0
    assert [t.id for t in out] == pool == ["s2", "s4", "s1"]
    assert skips == [RERANK_SKIP_MV_SCENE_UNVERIFIABLE]

    record = rec.record
    assert record.rerank_policy_skip == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert record.reorder_stage == REORDER_POLICY_SKIPPED
    assert record.reorder_attempted is False and record.rerank_runs == [] and record.rerank_error == ""
    for track in out:
        song = record.get(track.id)
        assert track.rerank_score is None
        assert song.score_mix is None and song.model_notes == [] and song.order_rules == []
        assert song.rank_before_rerank == song.rank_after_rerank
        text = render_ko(song, reorder_stage=record.reorder_stage)
        assert "리랭킹" not in text and "리랭커" not in text, text


def test_router_still_reranks_non_target_queries_with_the_same_reranker():
    rr = _CallingReranker()
    rec = ExplainRecorder("q")
    out = _search(_router(_hits(), reranker=rr), _analysis(original_query="뮤비가 유명한 남자 발라드"),
                  top_k=4, candidate_k=4, recorder=rec)
    assert rr.calls == 1
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED
    assert rec.record.rerank_policy_skip == ""
    assert [t.id for t in out] == ["s4", "s3", "s2", "s1"]


def test_disabled_or_single_candidate_is_not_called_a_policy_skip():
    """부를 수 없었던 요청에 '생략'을 적지 않는다 — 하지 않을 일을 생략한 것이 아니다."""
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    skips = []
    _search(_router(_hits(), reranker=rr), _analysis(original_query=USER_QUERY),
            top_k=4, candidate_k=4, use_rerank=False, recorder=rec, rerank_skip_out=skips)
    assert skips == [] and rec.record.reorder_stage == REORDER_NOT_ATTEMPTED

    rr2 = _ForbiddenReranker()
    rec2 = ExplainRecorder(USER_QUERY)
    skips2 = []
    _search(_router(_hits()[:1], reranker=rr2), _analysis(original_query=USER_QUERY),
            top_k=4, candidate_k=4, recorder=rec2, rerank_skip_out=skips2)
    assert skips2 == [] and rec2.record.reorder_stage == REORDER_NOT_ATTEMPTED


def _lyric_hits():
    # 후보 순서 s1(일반) > p1(가사 exact) > s2. 리랭킹 경로라면 p1이 일반 후보 아래로 내려가지 않는다.
    return [
        _track("s1", score=0.0200),
        _track("p1", score=0.0160, lyric_match_type="exact", lyric_match_score=0.95),
        _track("s2", score=0.0159),
    ]


def test_mv_skip_keeps_the_lyric_protection_without_calling_the_model():
    """생략 경로에서도 가사 exact 보호 배치는 살아 있다. 모델 호출 0회, 규칙은 기록된다."""
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    out = _search(_router(_lyric_hits(), reranker=rr),
                  _analysis(original_query="가사에 '하늘 위로'가 나오고 뮤비에서 의자가 하늘에 떠 있는 장면이 있어"),
                  top_k=3, candidate_k=3, recorder=rec)
    assert rr.calls == 0
    assert [t.id for t in out] == ["p1", "s1", "s2"]
    record = rec.record
    assert record.reorder_stage == REORDER_POLICY_SKIPPED
    assert [r.rule for r in record.get("p1").order_rules] == ["lyrics_exact_priority"]
    assert record.get("s1").behind_protected == 1 and record.get("s2").behind_protected == 1
    assert record.get("p1").rank_before_rerank == 2 and record.get("p1").rank_after_rerank == 1
    text = render_ko(record.get("p1"), reorder_stage=record.reorder_stage)
    assert "먼저 배치" in text and "리랭킹 생략" in text, text
    assert "리랭커" not in text, "모델이 돌았다는 말이 섞였다: " + text


def test_mv_skip_orders_protected_songs_by_the_corrected_answer_tier():
    """보호 곡이 둘이면 명시 성별이 맞는 tier가 먼저다. 라우터는 **재질문 답변을 합친** 분석으로 tier를 계산한다 —
    최초 질의에 성별이 없고 답으로 '남성'을 말했으면 남성 보호 곡이 먼저 온다."""
    # 점수 차를 크게 둔다 — 답변 보너스(1/61 ≈ 0.016)로는 순서가 안 바뀌어, tier만이 p2를 앞세울 수 있다.
    hits = [
        _track("p1", score=0.2000, lyric_match_type="exact", lyric_match_score=0.95, vocal_gender="여성"),
        _track("p2", score=0.1000, lyric_match_type="exact", lyric_match_score=0.95, vocal_gender="남성"),
        _track("s1", score=0.0160),
    ]
    rr = _ForbiddenReranker()
    out = _search(_router(hits, reranker=rr), _analysis(original_query=USER_QUERY), top_k=3, candidate_k=3,
                  answers=[ClarifyAnswer(slot="vocal_gender", value="남성")])
    assert rr.calls == 0
    assert [t.id for t in out] == ["p2", "p1", "s1"]

    # 답이 없으면 후보 순서(점수 순) 그대로 — tier 정렬은 안정 정렬이다
    from src.retrieval.search_router import place_protected_lyrics_first
    plain = place_protected_lyrics_first(_analysis(original_query=USER_QUERY), hits, 3)
    assert [t.id for t in plain] == ["p1", "p2", "s1"]
    tiered = place_protected_lyrics_first(_analysis(original_query=USER_QUERY, vocal_gender="남성"), hits, 3)
    assert [t.id for t in tiered] == ["p2", "p1", "s1"]


def test_mv_skip_without_protected_songs_is_plain_candidate_order():
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    out = _search(_router(_hits(), reranker=rr), _analysis(original_query=USER_QUERY), top_k=4, candidate_k=4, recorder=rec)
    assert [t.id for t in out] == ["s1", "s2", "s3", "s4"]
    assert rec.record.order_rules_applied is False


def test_image_skip_keeps_its_previous_behaviour_plain_candidate_order():
    """표지 지배 생략은 기준선 동작(후보 순서 그대로)을 유지한다 — 보호 배치를 새로 넣지 않는다."""
    rr = _ForbiddenReranker()
    rec = ExplainRecorder("q")
    cover = _analysis(original_query="초록 바탕에 흰 도형이 있는 앨범 커버", has_visual_clue=True,
                      image_english_query="green cover", modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0))
    out = _search(_router(_lyric_hits(), reranker=rr), cover, top_k=3, candidate_k=3, recorder=rec)
    assert rr.calls == 0
    assert [t.id for t in out] == ["s1", "p1", "s2"]
    assert rec.record.rerank_policy_skip == RERANK_SKIP_IMAGE_DOMINANT
    assert rec.record.order_rules_applied is False


# ---------------------------------------------------------------------------
# 평가 도구 — 서비스와 같은 정책, 같은 기록
# ---------------------------------------------------------------------------

def test_eval_helper_uses_the_same_policy_and_returns_the_bonus_order():
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    statuses, skips = [], []
    pre_bonus = [_track("s1"), _track("s2"), _track("s3")]
    bonus = [_track("s2"), _track("s1"), _track("s3")]
    final = ev.rerank_preserving_exact_lyrics(
        rr, _analysis(original_query=USER_QUERY), pre_bonus, 3,
        recorder=rec, statuses=statuses, skips=skips, fallback_candidates=bonus,
    )
    assert rr.calls == 0
    assert [t.id for t in final] == ["s2", "s1", "s3"], "보정 전 입력이 아니라 후보(보너스) 순서"
    assert statuses == [] and skips == [RERANK_SKIP_MV_SCENE_UNVERIFIABLE]
    assert rec.record.reorder_stage == REORDER_POLICY_SKIPPED

    columns = ev.explain_columns(rec.record, enabled=True, statuses=statuses, skips=skips)
    assert columns["rerank_outcome"] == ev.OUTCOME_POLICY_SKIPPED
    assert columns["rerank_policy_skip"] == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert columns["rerank_fell_back"] == 0 and columns["rerank_partial_failure"] == 0
    assert columns["reorder_stage"] == REORDER_POLICY_SKIPPED

    off = ev.explain_columns(rec.record, enabled=False, statuses=statuses, skips=skips)
    assert off["rerank_outcome"] == ev.OUTCOME_POLICY_SKIPPED, "기록을 꺼도 생략 사유는 남는다"


def test_eval_helper_applies_the_lyric_protection_on_mv_skip_like_the_router():
    rr = _ForbiddenReranker()
    rec = ExplainRecorder(USER_QUERY)
    final = ev.rerank_preserving_exact_lyrics(rr, _analysis(original_query=USER_QUERY), _lyric_hits(), 3, recorder=rec)
    assert rr.calls == 0
    assert [t.id for t in final] == ["p1", "s1", "s2"]
    assert [r.rule for r in rec.record.get("p1").order_rules] == ["lyrics_exact_priority"]
    assert rec.record.reorder_stage == REORDER_POLICY_SKIPPED


def test_eval_helper_gates_the_policy_like_the_router():
    """리랭커가 꺼져 있거나 후보가 하나뿐이면 **정책 생략을 적지 않는다**(라우터는 그 요청을 not_attempted로 적는다).

    평가 사본은 그 경우에도 백엔드를 한 번 부르고 백엔드가 skipped를 보고한다(기존 동작). 여기서 고정하는 것은
    "부를 수 없는 요청에 policy_skipped가 붙지 않는다"는 점이다.
    """
    from src.retrieval.explain import RERANK_SKIPPED

    class _ReportsSkipped(_ForbiddenReranker):
        def rerank_run(self, query, tracks, top_k, answers=None):
            self.calls += 1
            return RerankRun(list(tracks)[:top_k], RERANK_SKIPPED, [])

    off = _ReportsSkipped()
    off.enabled = False
    rec = ExplainRecorder(USER_QUERY)
    skips = []
    ev.rerank_preserving_exact_lyrics(off, _analysis(original_query=USER_QUERY), _hits(), 4, recorder=rec, skips=skips)
    assert skips == [] and rec.record.rerank_policy_skip == ""
    assert rec.record.reorder_stage != REORDER_POLICY_SKIPPED

    single = _ReportsSkipped()
    rec2 = ExplainRecorder(USER_QUERY)
    skips2 = []
    ev.rerank_preserving_exact_lyrics(single, _analysis(original_query=USER_QUERY), _hits()[:1], 4, recorder=rec2, skips=skips2)
    assert skips2 == [] and rec2.record.rerank_policy_skip == ""
    assert rec2.record.reorder_stage != REORDER_POLICY_SKIPPED


def test_eval_script_writes_the_policy_skip_column(tmp_path, monkeypatch):
    """evaluate() 끝단: detail CSV에 생략 사유·결말이 남고 실패로 세지 않는다."""
    import csv as _csv

    csv_path = tmp_path / "queries.csv"
    csv_path.write_text(
        "query_id,split,query_type,query,relevant_ids\n"
        f"q1,dev,search,{USER_QUERY},s1\n"
        "q2,dev,search,비 오는 날 발라드,s2\n",
        encoding="utf-8",
    )

    class _Router:
        async def search(self, analysis, **kw):
            return [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]

    from src.retrieval.reranker import RerankerConfig

    class _Reranker(_CallingReranker):
        config = RerankerConfig()
        uses_clarify_answers = False

        def load(self):
            return self

        def rerank_run(self, query, tracks, top_k, answers=None):
            assert query != USER_QUERY, "보호 경로에서 모델이 호출됐다"
            return super().rerank_run(query, tracks, top_k, answers)

    class _Analyzer:
        def analyze(self, query):
            return _analysis(original_query=query)

    monkeypatch.setattr(ev, "get_query_analyzer", lambda: _Analyzer())
    monkeypatch.setattr(ev, "get_search_router", lambda: _Router())
    monkeypatch.setattr(ev, "get_reranker", lambda: _Reranker())
    args = ev.build_parser().parse_args(
        ["--input", str(csv_path), "--output-dir", str(tmp_path), "--split", "dev", "--top-k", "3", "--candidate-k", "3"]
    )
    asyncio.run(ev.evaluate(args))

    detail = {row["query_id"]: row for row in
              _csv.DictReader((tmp_path / "search_eval_dev_detail.csv").open(encoding="utf-8-sig"))}
    assert detail["q1"]["rerank_policy_skip"] == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert detail["q1"]["rerank_outcome"] == ev.OUTCOME_POLICY_SKIPPED
    assert detail["q1"]["rerank_fell_back"] == "0" and detail["q1"]["rerank_run_statuses"] == ""
    assert detail["q2"]["rerank_policy_skip"] == "" and detail["q2"]["rerank_outcome"] == ev.OUTCOME_APPLIED


def test_clarification_eval_counts_the_policy_skip_separately():
    import src.retrieval.evaluate_clarification as ec

    ec.RERANK_STATUS_COUNTS.clear()
    rr = _ForbiddenReranker()
    router = _router(_hits(), reranker=rr)
    try:
        result = asyncio.run(ec.run_search(router, rr, _analysis(original_query=USER_QUERY), {"s2"}, top_k=4, candidate_k=4))
    finally:
        router.shutdown()
        counts = dict(ec.RERANK_STATUS_COUNTS)
        ec.RERANK_STATUS_COUNTS.clear()
    assert rr.calls == 0
    assert counts == {f"policy_skipped:{RERANK_SKIP_MV_SCENE_UNVERIFIABLE}": 1}, counts
    assert result.shown_ids == ["s1", "s2", "s3", "s4"]


def test_eval_outcome_keeps_policy_skip_apart_from_normal_skip_and_failure():
    assert ev.rerank_outcome([], "") == ev.OUTCOME_SKIPPED
    assert ev.rerank_outcome([], "", RERANK_SKIP_MV_SCENE_UNVERIFIABLE) == ev.OUTCOME_POLICY_SKIPPED
    assert ev.rerank_outcome([], "RuntimeError: x", RERANK_SKIP_MV_SCENE_UNVERIFIABLE) == ev.OUTCOME_EXCEPTION
    assert not ev.rerank_fell_back([], "")


def test_eval_payload_keeps_the_raw_model_note_only_in_the_internal_field():
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.note_model_reason("s1", [FABRICATED])
    payload = ev.build_explain_payload(
        query_id="x", split="dev", query="q", relevant_ids={"s1"}, record=rec.record,
        ordered=[_track("s1")], candidates=[_track("s1")], top_k=1,
    )
    row = payload["songs"][0]
    assert row["explain"]["model_notes"] == []
    assert row["model_notes_internal"] == [FABRICATED]


# ---------------------------------------------------------------------------
# 설명 노출 — 어떤 문장도, 어떤 confidence로도 나가지 않는다
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("note", [
    FABRICATED,
    "뮤직비디오에 배우 OO이 카메오로 나온다",
    "사막에서 촬영한 뮤직비디오로 유명하다",
    "후렴에 빨간 풍선이 하늘로 올라가는 장면이 있다",
])
def test_any_model_note_is_kept_out_of_the_track_explain(note):
    rec = ExplainRecorder("q")
    rec.path("s1", "text_hybrid", 1, 0.016)
    rec.note_model_reason("s1", [note])
    rec.set_score_mix("s1", ScoreMix(backend="gemini_listwise", weight=0.85, configured_weight=0.85,
                                     confidence=1.0, rerank_component=0.99, retrieval_component=0.5, final=0.92))
    out = to_track_explain(rec.record.get("s1"), reorder_stage=REORDER_MODEL_APPLIED)
    dumped = out.model_dump_json()
    assert out.model_notes == []
    assert note not in dumped
    assert rec.record.get("s1").model_notes == [note], "내부 원문은 남는다"
    assert out.score_mix is not None and out.paths, "검증된 기록은 그대로 나간다"


class _FakeAnalyzer:
    def __init__(self, query_to_analysis):
        self._f = query_to_analysis

    def analyze(self, query):
        return self._f(query)


def _client(router, analyzer):
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: analyzer
    app.dependency_overrides[get_search_router] = lambda: router
    return TestClient(app)


def test_api_response_never_carries_the_fabricated_sentence_even_when_reranked():
    """비대상 질의 → 모델이 돌고 허구 이유를 냈다. 응답 JSON 어디에도 그 문장이 없어야 한다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    router = _router(hits, reranker=_CallingReranker())
    try:
        with _client(router, _FakeAnalyzer(lambda q: _analysis(original_query=q))) as client:
            body = client.post("/search", json={"query": "비 오는 날 발라드", "top_k": 3, "explain": True}).json()
    finally:
        router.shutdown()
    text = json.dumps(body, ensure_ascii=False)
    assert FABRICATED not in text
    assert body["explain"]["reorder_stage"] == REORDER_MODEL_APPLIED
    assert body["notice"] is None
    for track in body["results"]:
        assert track["explain"]["model_notes"] == []
        assert track["explain"]["score_mix"] is not None, "검증된 합성식은 그대로"


def test_api_user_case_skips_rerank_and_sends_the_fixed_notice():
    hits = [_upside_down_like(), _track("s2", title="곡2"), _track("s3", title="곡3")]
    router = _router(hits, reranker=_ForbiddenReranker())
    try:
        with _client(router, _FakeAnalyzer(lambda q: _analysis(original_query=q))) as client:
            on = client.post("/search", json={"query": USER_QUERY, "top_k": 3, "explain": True}).json()
            off = client.post("/search", json={"query": USER_QUERY, "top_k": 3}).json()
    finally:
        router.shutdown()
    assert [t["id"] for t in on["results"]] == [t["id"] for t in off["results"]] == ["602193305", "s2", "s3"]
    assert [t["score"] for t in on["results"]] == [t["score"] for t in off["results"]]
    for body in (on, off):
        assert body["notice"] == {"code": RERANK_SKIP_MV_SCENE_UNVERIFIABLE,
                                  "message": RERANK_SKIP_LABELS[RERANK_SKIP_MV_SCENE_UNVERIFIABLE]}
    summary = on["explain"]
    assert summary["reorder_stage"] == REORDER_POLICY_SKIPPED
    assert summary["rerank_skip_reason"] == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert summary["rerank_skip_label"] == RERANK_SKIP_LABELS[RERANK_SKIP_MV_SCENE_UNVERIFIABLE]
    assert summary["rerank_calls"] == 0 and summary["rerank_error"] == ""
    top = on["results"][0]["explain"]
    assert top["model_notes"] == [] and top["score_mix"] is None and top["rerank_score"] is None
    assert "리랭킹" not in top["summary"]
    assert top["reorder_stage_label"] == REORDER_STAGE_LABELS[REORDER_POLICY_SKIPPED]


def test_api_image_skip_has_no_notice_but_keeps_the_diagnostic_reason():
    """표지 지배 생략은 요청 범위 밖 — 안내 문구를 싣지 않고 explain의 사유 코드로만 남는다."""
    hits = [_track(f"s{i}", title=f"곡{i}") for i in range(1, 4)]
    router = _router(hits, reranker=_ForbiddenReranker())
    cover = _analysis(original_query="초록 바탕에 흰 도형이 있는 앨범 커버", has_visual_clue=True,
                      image_english_query="green cover", modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0))
    try:
        with _client(router, _FakeAnalyzer(lambda q: cover)) as client:
            body = client.post("/search", json={"query": cover.original_query, "top_k": 3, "explain": True}).json()
    finally:
        router.shutdown()
    assert body["notice"] is None
    assert body["explain"]["rerank_skip_reason"] == RERANK_SKIP_IMAGE_DOMINANT
    assert body["explain"]["rerank_skip_label"] == ""
    assert body["explain"]["reorder_stage"] == REORDER_POLICY_SKIPPED


def test_api_prior_analysis_turn_keeps_the_policy():
    """2턴(prior_analysis + rejected_ids)에서도 원문 판정이 같아 모델을 부르지 않는다."""
    hits = [_upside_down_like(), _track("s2", title="곡2"), _track("s3", title="곡3")]
    router = _router(hits, reranker=_ForbiddenReranker())
    prior = _analysis(original_query=USER_QUERY).model_dump()
    try:
        with _client(router, _FakeAnalyzer(lambda q: pytest.fail("prior_analysis가 있으면 재분석하지 않는다"))) as client:
            body = client.post("/search", json={
                "query": USER_QUERY, "top_k": 3, "explain": True,
                "prior_analysis": prior, "rejected_ids": ["602193305"], "turn": 1,
            }).json()
    finally:
        router.shutdown()
    assert [t["id"] for t in body["results"]] == ["s2", "s3"]
    assert body["notice"]["code"] == RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    assert body["explain"]["reorder_stage"] == REORDER_POLICY_SKIPPED
