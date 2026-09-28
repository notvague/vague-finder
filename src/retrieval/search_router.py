"""
[Retrieval] SearchRouter — 멀티모달 병렬 검색 매니저
설명: 가사 표면(full_lyrics), 텍스트(BM25+KoE5), 이미지(SigLIP2), 오디오(CLAP) 경로를
      asyncio.gather + ThreadPoolExecutor로 병렬 실행하고 RRF로 점수를 통합한 뒤,
      정확 단서 부스팅과 (선택적) Cross-Encoder 리랭킹을 적용합니다.
작성자: 황찬혁 (Full)
생성일: 2026-05-23
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pinecone import Pinecone

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.embedding.models.audio_clap import CLAPAudioEmbedder
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.retrieval import timing
from src.retrieval.explain import (
    LYRIC_MATCH_LABELS,
    LYRIC_SCORE_LABELS,
    NULL_RECORDER,
    PATH_LABELS,
    RERANK_APPLIED,
    RERANK_FAILED,
    RERANK_UNKNOWN,
    ExplainRecorder,
    RerankRun,
)
from src.retrieval.reranker import MusicReranker
from src.retrieval.lyrics_exact_search import LyricsExactSearchService
from src.retrieval.lyrics_query import normalize_lyric_surface
from src.retrieval.clarify import (
    analysis_with_answers,
    apply_answer_bonus,
    canonical_artist_types as _canonical_artist_types,
)
from src.retrieval.search_service import SearchService
from src.common.title_features import (
    build_title_meaning_metadata_filter,
    build_title_metadata_filter,
    build_title_presence_metadata_filter,
    has_title_constraints,
    title_constraint_match_ratio,
    title_meaning_match_ratio,
    title_presence_match_ratio,
)
from src.vector_db.settings import AUDIO_INDEX_NAME, IMAGE_INDEX_NAME, NAMESPACE

logger = logging.getLogger(__name__)

# (song_id, raw_score) 타입 별칭
_Hit = Tuple[str, float]

# RRF 표준 상수 (보통 60)
_RRF_K = 60




def _release_year(value: Optional[str]) -> Optional[int]:
    match = re.search(r"((?:19|20)\d{2})", str(value or ""))
    return int(match.group(1)) if match else None


def _release_era_similarity(
    analysis: QueryAnalysis,
    track: MatchingTrack,
) -> float:
    if not analysis.has_release_era:
        return 0.0

    year = _release_year(track.release_date)
    if year is None:
        return 0.0

    start = analysis.release_era.start_year
    end = analysis.release_era.end_year

    if start is None or end is None:
        return 0.0

    if start <= year <= end:
        return 1.0

    distance = start - year if year < start else year - end

    return {
        1: 0.75,
        2: 0.50,
        3: 0.25,
    }.get(distance, 0.0)


def _artist_type_similarity(
    analysis: QueryAnalysis,
    track: MatchingTrack,
) -> float:
    if not analysis.has_artist_type_clue:
        return 0.0

    wanted = set(analysis.artist_type.values)
    actual = _canonical_artist_types(track.artist_types)

    return float(bool(wanted & actual))


def _metadata_clue_strength(
    analysis: QueryAnalysis,
    track: MatchingTrack,
) -> float:
    era_score = (
        _release_era_similarity(analysis, track)
        * analysis.release_era.confidence
    )

    artist_type_score = (
        _artist_type_similarity(analysis, track)
        * analysis.artist_type.confidence
    )

    return era_score + artist_type_score


_SOUND_ALIASES: Dict[str, tuple[str, ...]] = {
    "밴드사운드": ("밴드", "록", "락", "기타", "드럼"),
    "어쿠스틱": ("어쿠스틱", "통기타", "언플러그드"),
    "오케스트라": ("오케스트라", "관현악", "웅장한 현악"),
    "전자음악": ("전자음", "일렉트로닉", "신스", "신디사이저", "edm"),
    "라이브": ("라이브", "공연", "관객"),
    "아카펠라": ("아카펠라", "무반주"),
    "브라스": ("브라스", "트럼펫", "트롬본", "색소폰"),
    "스트링": ("스트링", "현악", "바이올린", "첼로"),
    "피아노": ("피아노", "건반"),
    "기타": ("기타", "guitar"),
    "퍼커션": ("퍼커션", "드럼", "타악기"),
    "휘파람": ("휘파람", "휘슬", "whistle", "whistling"),
    "비프음": ("비프", "삐삐", "beep"),
    "벨소리": ("벨소리", "종소리", "차임", "초인종"),
    "박수": ("박수", "손뼉", "clap"),
    "핑거스냅": ("핑거스냅", "손가락", "snap"),
    "사이렌": ("사이렌", "경보음", "siren"),
    "전화음": ("전화", "연결음", "통화음", "수화기"),
    "자연음": ("빗소리", "파도", "새소리", "바람소리", "물소리"),
    "플루트": ("플루트", "플룻", "피리"),
    "하모니카": ("하모니카",),
    "국악": ("국악", "판소리", "가야금", "해금", "대금", "장구"),
    "인도풍": ("인도", "볼리우드", "시타르", "이국적"),
    "중동풍": ("중동", "아랍", "이국적"),
}


def _track_clue_text(track: MatchingTrack) -> str:
    values: List[str] = [
        track.title,
        track.artist or "",
        track.genre or "",
        track.vocal_gender or "",
        track.search_style_summary or "",
        track.lyrics_summary or "",
        *track.artist_types,
        *track.sound_tags,
        *track.mood_tags,
        *track.vibe_tags,
        *track.relation_context_tags,
    ]
    return " ".join(str(value) for value in values if value).lower()


def _has_feature_marker(track: MatchingTrack) -> bool:
    return bool(
        re.search(
            r"(?:\bfeat(?:uring)?\.?\b|\bwith\b|피처링|듀엣)",
            f"{track.title} {track.artist or ''}",
            re.IGNORECASE,
        )
    )


def _performance_clue_similarity(
    analysis: QueryAnalysis,
    track: MatchingTrack,
) -> float:
    """곡 메타데이터에서 공연 역할/효과음 단서의 양성 증거만 계산한다."""
    if not analysis.has_composite_performance_clue:
        return 0.0

    clue = analysis.performance_clues
    text = _track_clue_text(track)
    actual_gender = str(track.vocal_gender or "").strip()
    actual_types = _canonical_artist_types(track.artist_types)
    feature_marker = _has_feature_marker(track)
    genre_text = str(track.genre or "").lower()
    is_rap = any(
        token in genre_text
        for token in ("힙합", "랩", "hip hop", "hip-hop", "rap")
    )
    category_scores: List[float] = []

    if clue.vocal_count:
        if clue.vocal_count == "duet":
            count_score = max(
                1.0 if actual_gender == "혼성" else 0.0,
                0.90 if feature_marker else 0.0,
                0.75 if "듀오" in actual_types else 0.0,
            )
        elif clue.vocal_count == "solo":
            count_score = max(
                0.85 if "솔로" in actual_types and not feature_marker else 0.0,
                0.70 if actual_gender in {"남성", "여성"} and not feature_marker else 0.0,
            )
        elif clue.vocal_count == "multiple":
            count_score = max(
                0.90 if "그룹" in actual_types else 0.0,
                0.75 if actual_gender == "혼성" else 0.0,
                0.65 if feature_marker else 0.0,
            )
        else:  # choir
            count_score = 1.0 if any(
                token in text for token in ("합창", "콰이어", "성가대", "떼창")
            ) else 0.0
        category_scores.append(count_score)

    if clue.vocal_roles:
        role_scores: List[float] = []
        for role in clue.vocal_roles:
            if role == "남성랩":
                role_scores.append(
                    0.65 * float(is_rap)
                    + 0.35 * float(actual_gender in {"남성", "혼성"})
                )
            elif role == "여성랩":
                role_scores.append(
                    0.65 * float(is_rap)
                    + 0.35 * float(actual_gender in {"여성", "혼성"})
                )
            elif role == "남성노래":
                role_scores.append(float(actual_gender in {"남성", "혼성"}))
            elif role == "여성노래":
                role_scores.append(float(actual_gender in {"여성", "혼성"}))
            elif role == "피처링보컬":
                role_scores.append(float(feature_marker))
            elif role == "나레이션":
                role_scores.append(float(any(token in text for token in ("나레이션", "내레이션", "낭독"))))
            elif role in {"코러스", "합창"}:
                role_scores.append(float(any(token in text for token in ("코러스", "합창", "콰이어", "떼창"))))
        if role_scores:
            category_scores.append(sum(role_scores) / len(role_scores))

    if clue.sound_ensemble:
        sound_scores: List[float] = []
        for sound in clue.sound_ensemble:
            aliases = _SOUND_ALIASES.get(sound, (sound,))
            sound_scores.append(float(any(alias.lower() in text for alias in aliases)))
        if sound_scores:
            category_scores.append(sum(sound_scores) / len(sound_scores))

    return (
        sum(category_scores) / len(category_scores)
        if category_scores
        else 0.0
    )


def _build_performance_metadata_filter(
    analysis: QueryAnalysis,
) -> Optional[Dict[str, Any]]:
    """두 축 이상의 고신뢰 공연 단서를 좁은 additive 경로로 변환한다.

    전역 검색에는 이 필터를 적용하지 않는다. 사용자의 기억이 틀려도 기존
    text/audio 후보는 그대로 남고, 메타데이터로 직접 확인 가능한 조합만 별도
    후보 경로를 연다.

    현재 안전하게 구조화할 수 있는 조합은 다음과 같다.
      - 남녀 역할이 모두 명시된 duet -> vocal_gender=혼성
      - 그중 랩 역할도 명시됨       -> genre=랩/힙합

    단순 "혼성곡" 또는 "힙합곡" 한 축만으로는 후보군이 너무 넓으므로 이
    경로를 열지 않는다.
    """
    if not analysis.has_composite_performance_clue:
        return None

    clue = analysis.performance_clues
    roles = set(clue.vocal_roles)
    has_male_role = bool(roles & {"남성랩", "남성노래"})
    has_female_role = bool(roles & {"여성랩", "여성노래"})
    has_rap_role = bool(roles & {"남성랩", "여성랩", "랩"})
    is_cross_gender_duet = bool(
        clue.vocal_count == "duet"
        and has_male_role
        and has_female_role
    )

    if not (is_cross_gender_duet and has_rap_role):
        return None

    return {
        "$and": [
            {"vocal_gender": {"$eq": "혼성"}},
            {"genre": {"$eq": "랩/힙합"}},
        ]
    }


def _norm(s: str) -> str:
    """제목/아티스트 정확 매칭용 정규화: 공백 제거 + 소문자화.
    표기 차이(공백, 대소문자)를 흡수해 비교한다."""
    return "".join((s or "").split()).lower()


# 가사 키워드/하이라이트 토큰화용 — BM25 토크나이저와 동일 규칙(한글/영숫자 런).
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokenize(text: str) -> List[str]:
    """소문자화 후 한글/영숫자 토큰 추출. 'da da da~' → ['da','da','da'] 처럼
    의성어 후크나 통문장도 토큰 단위로 분해해 substring 매칭 실패를 방지한다."""
    return _TOKEN_RE.findall((text or "").lower())


def _dedupe_terms(values: List[str]) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_lyric_surface(text)
        if text and key and key not in seen:
            seen.add(key)
            terms.append(text)
    return terms


def _exact_lyric_terms(analysis: QueryAnalysis) -> List[str]:
    return _dedupe_terms(
        [
            clue.text
            for clue in analysis.lyric_clues
            if clue.kind in {"verbatim", "partial"}
        ]
        + list(analysis.lyric_keywords)
    )


def _phonetic_lyric_terms(analysis: QueryAnalysis) -> List[str]:
    values: List[str] = []
    for clue in analysis.lyric_clues:
        if clue.kind == "phonetic":
            values.extend([clue.text, *clue.variants])
    return _dedupe_terms(values)


def exact_lyric_constraint_score(
    analysis: QueryAnalysis,
    track: MatchingTrack,
) -> float:
    """동일 exact 구절 후보 안에서 명시 메타데이터의 적합도를 계산한다."""
    score = 0.0

    want_title = _norm(analysis.song_title)
    if want_title and _norm(track.title) == want_title:
        score += 4.0

    want_artist = _norm(analysis.artist_name)
    if want_artist and track.artist and want_artist in _norm(track.artist):
        score += 3.0

    want_gender = analysis.vocal_gender
    actual_gender = str(track.vocal_gender or "").strip()
    if want_gender and actual_gender:
        if actual_gender == want_gender:
            score += 2.0
        elif actual_gender == "혼성" and want_gender in {"남성", "여성"}:
            score += 0.5
        elif {want_gender, actual_gender} == {"남성", "여성"}:
            score -= 2.0

    want_genre = _norm(analysis.genre)
    if want_genre and track.genre and want_genre in _norm(track.genre):
        score += 1.0

    if has_title_constraints(analysis.title_constraints):
        score += title_constraint_match_ratio(
            track.title,
            analysis.title_constraints,
        ) * analysis.title_constraints.confidence

    return score


# ---------------------------------------------------------------------------
# RRF 퓨전
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: List[List[_Hit]],
    weights: List[float],
    top_k: Optional[int],
    labels: Optional[List[str]] = None,
    recorder: ExplainRecorder = NULL_RECORDER,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion으로 여러 모달리티 결과를 통합한다.

    score(d) = Σ_i  weight_i / (RRF_K + rank_i(d))

    - 절대 점수 스케일이 모달리티마다 달라도 순위(rank)만 사용하므로 안전
    - weight로 각 모달리티 기여도를 조절
    - 특정 모달리티에 song_id가 없어도 다른 경로 점수만으로 집계됨

    labels/recorder는 실행 기록용이다. 더하는 값(`contribution`)을 그대로 남기므로
    점수 계산에는 관여하지 않는다 — 기록을 켜도 결과가 같아야 한다.
    """
    scores: Dict[str, float] = {}
    for index, (hits, w) in enumerate(zip(ranked_lists, weights)):
        label = labels[index] if labels and index < len(labels) else f"path{index}"
        for rank, (song_id, _) in enumerate(hits, start=1):
            contribution = w / (_RRF_K + rank)
            scores[song_id] = scores.get(song_id, 0.0) + contribution
            recorder.path(song_id, label, rank, contribution, f"가중치 {w:.2f}")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
        # 절단에서 탈락한 곡의 기여는 **최종 점수에 남지 않는다.** 탈락한 곡이
        # 가사 경로로 다시 들어오면 점수가 0에서 시작하므로, 기록을 그대로 두면
        # 설명의 기여 합계가 실제 융합 점수보다 커진다.
        survivors = {song_id for song_id, _ in ranked}
        recorder.mark_paths_dropped(
            song_id for song_id in scores if song_id not in survivors
        )
    return ranked


def _matched_lyric_field(surfaces: List[str], track: MatchingTrack) -> str:
    """단서가 발췌(highlight)에서 맞았는지 요약(summary)에서 맞았는지 **보고용으로** 가린다.

    점수 판정은 두 필드를 합친 문자열로 하므로 이 함수는 점수에 관여하지 않는다.
    합친 경계에 걸쳐 맞은 경우만 "combined"가 된다.
    """
    normalized = [
        normalized_clue
        for surface in surfaces
        if (normalized_clue := normalize_lyric_surface(surface))
    ]
    in_highlight = any(
        clue in normalize_lyric_surface(track.lyrics_highlight or "")
        for clue in normalized
    )
    if in_highlight:
        return "highlight"
    in_summary = any(
        clue in normalize_lyric_surface(track.lyrics_summary or "")
        for clue in normalized
    )
    return "summary" if in_summary else "combined"


def _fuse_add(
    fused_scores: Dict[str, float],
    song_id: str,
    delta: float,
    recorder: ExplainRecorder,
    path: str,
    rank: int,
    detail: str = "",
) -> None:
    """보조 경로 기여를 더하면서 실제로 더해진 양을 기록한다.

    `fused_scores.get(...) + delta`를 그대로 유지한다. 기록값은 재계산이 아니라
    **적용 전후 차이**를 쓰므로, 표시되는 숫자가 점수에 반영된 것과 항상 같다.
    """
    before = fused_scores.get(song_id, 0.0)
    fused_scores[song_id] = before + delta
    recorder.path(song_id, path, rank, fused_scores[song_id] - before, detail)


def _reject_and_trim(
    ranked: List[Tuple[str, float]],
    excluded: Optional[Set[str]],
    candidates: int,
) -> List[Tuple[str, float]]:
    """거절한 곡을 뺀 뒤 후보 풀을 candidates개로 자른다.

    **자르기 전에 걸러야 한다.** 자른 뒤에 거르면 거절한 수만큼 후보 풀이
    얇아져서, Top-10을 전부 거절했을 때 다음 턴 후보가 30개가 아니라 20개가
    된다.

    단 이 순서만으로는 부족하다. `ranked`가 이미 candidates개면 걸러낸 자리를
    메울 곡이 애초에 없다. 호출부가 `pool_k`(= candidates + 거절 수)로 각
    경로와 RRF 절단을 넓혀서 넘겨줘야 비로소 풀이 다시 채워진다.
    둘이 같이 있어야 Reject-only 기준선이 성립한다.
    """
    if excluded:
        ranked = [item for item in ranked if item[0] not in excluded]
    return ranked[:candidates]


def call_reranker(
    reranker: Any,
    query: str,
    tracks: Sequence[MatchingTrack],
    top_k: int,
    answers: Optional[Sequence[ClarifyAnswer]] = None,
    recorder: ExplainRecorder = NULL_RECORDER,
    runs_out: Optional[List[str]] = None,
) -> List[MatchingTrack]:
    """리랭커를 부른다. 재질문 답변을 쓰는 리랭커에만 answers를 넘긴다.

    답변을 넘기지 않으면 LLM 리랭커는 최초 질의의 틀린 기억을 따라 정정된 답을
    덮어쓴다. Cross-Encoder처럼 답변을 받지 않는 리랭커는 시그니처를 바꾸지 않는다.

    모델 호출은 이 함수만 거친다. 상태는 **백엔드가 보고한 것**을 쓴다 — 호출 횟수로는
    성공과 내부 폴백을 구분할 수 없다(Gemini는 재시도가 전부 실패해도 예외 없이
    입력 순서를 돌려준다). 상태를 보고하지 않는 백엔드는 unknown으로 남겨 단정하지 않는다.

    runs_out에 리스트를 넘기면 실행 상태가 담긴다. **기록기와 별개의 통로다** —
    설명 기록을 끄면 NULL_RECORDER가 상태를 버리는데, 백엔드가 예외 없이 실패한
    사실(Gemini 내부 폴백)은 기록 스위치와 무관하게 집계돼야 한다. 그러지 않으면
    측정에서 실패가 "리랭킹 효과 없음"으로 조용히 섞인다.
    """
    pass_answers = bool(answers) and getattr(reranker, "uses_clarify_answers", False)
    detailed = getattr(reranker, "rerank_run", None)

    if detailed is not None:
        run = (
            detailed(query, tracks, top_k, answers=list(answers))
            if pass_answers
            else detailed(query, tracks, top_k)
        )
    else:
        # 상태를 보고하지 않는 백엔드. 평가 범위도 알 수 없으므로 judged를 비운다 —
        # 모르는 것을 "리랭킹됨"으로 표시하면 안 된다.
        tracks_out = (
            reranker.rerank(query, tracks, top_k, answers=list(answers))
            if pass_answers
            else reranker.rerank(query, tracks, top_k)
        )
        run = RerankRun(tracks_out, RERANK_UNKNOWN, [])

    if runs_out is not None:
        runs_out.append(run.status)

    recorder.note_rerank_run(run.status, run.judged_ids)
    if run.status == RERANK_APPLIED:
        # 모델에 귀속할 수 있는 변화는 **이 묶음 안에서의 이동**뿐이다.
        # 요청 전체 순위는 보호 배치 같은 다른 원인과 섞인다.
        #
        # 그 묶음 안에서도 반환 순서가 곧 모델의 순서는 아니다. Gemini는 모델 순서
        # 뒤에 rescue 규칙을 연달아 적용하므로, 규칙이 개입하기 전 순서를 보고하면
        # 그것을 쓴다. 반환 순서를 쓰면 규칙의 이동까지 모델이 한 일이 된다.
        after_ids = run.model_order_ids or [track.id for track in run.tracks]
        recorder.set_group_ranks(
            [track.id for track in tracks],
            after_ids,
            run.judged_ids,
        )
        # 최종 점수 합성식. 리랭커 점수가 곧 최종 점수가 아니라는 사실은 기록에
        # 남아야만 말할 수 있다 — CE 최고점 곡이 1위가 아닌 경우가 실제로 있었다.
        for song_id, mix in run.mixes.items():
            recorder.set_score_mix(song_id, mix)
        # 모델이 낸 순서를 규칙이 다시 바꾼 것. 순서 규칙으로 남겨 모델의 정렬과
        # 구분한다.
        for song_id, rule, detail in run.order_notes:
            recorder.order(song_id, rule, detail)
        # 모델이 스스로 쓴 문장. 검증된 근거와 같은 목록에 넣지 않는다.
        for song_id, texts in run.model_notes.items():
            recorder.note_model_reason(song_id, texts)
    return run.tracks


# 보호를 받을 최소 단서 확신도. **문자 유사도 기준이 아니다**
# (`lyrics_exact_search.py`는 정규화 후 부분문자열로 발견되면 단서의 confidence를
# 그대로 점수로 쓴다).
#
# 1.0보다 크게 주면 보호가 전부 꺼진다 — 보호 규칙 자체의 비용을 재는 진단용이다.
# q211이 그 비용의 실례다: "위험하다" 한 조각이 다른 곡에 있어서 그 곡이 보호를 받고
# 1위에 고정되고, 리랭커가 그것을 고칠 수 없다(보호 곡은 일반 풀에서 빠진다).
def _lyric_protect_min_confidence() -> float:
    raw = os.getenv("LYRIC_PROTECT_MIN_CONFIDENCE", "").strip()
    try:
        return float(raw) if raw else 0.80
    except ValueError:
        return 0.80


LYRIC_PROTECT_MIN_CONFIDENCE = 0.80  # 기본값. 실제 판정은 위 함수를 쓴다.


def _lyric_boost_scale() -> float:
    """가사 표면 부스트에 곱하는 배율. 1.0이 현행, 0이면 가산점만 끈다.

    **대조 실험용이다.** 0으로 두면 가사 매칭과 후보 유입은 그대로 두고 가산점만
    사라지므로, "부스트가 최종 결과에 무엇을 했나"를 분리해 잴 수 있다.
    보호 규칙을 끄는 것(`LYRIC_PROTECT_MIN_CONFIDENCE`)과는 **다른 실험이다** —
    보호를 꺼도 부스트는 그대로 걸린다.
    """
    raw = os.getenv("LYRIC_BOOST_SCALE", "").strip()
    try:
        return max(0.0, float(raw)) if raw else 1.0
    except ValueError:
        return 1.0


def _lyric_match_note(track: MatchingTrack) -> str:
    """부스트가 걸린 근거를 문장에 덧붙인다 — 구절 길이와 겹치는 곡 수.

    점수만 보면 "위험하다" 네 글자가 다른 곡에 걸린 경우와 한 곡에만 있는 긴 구절이
    구분되지 않는다. 둘 다 단서 확신도가 그대로 점수가 되기 때문이다.
    """
    detail = getattr(track, "lyric_match_detail", None)
    if detail is None:
        return ""
    note = f", {detail.normalized_length}자"
    if detail.corpus_match_count > 1:
        note += f"·{detail.corpus_match_count}곡에 겹침"
    if detail.clue_kind != detail.match_type:
        note += f"({detail.clue_kind} 단서)"
    return note


def lyric_priority_rule(track: MatchingTrack) -> str:
    """이 곡이 **어떤 근거로** 먼저 배치됐는지. 유형마다 다른 규칙이다.

    phonetic에 exact용 규칙을 재사용하면 "가사가 그대로 일치해 먼저 배치"가 붙는다.
    음차는 추정 표기이고 비교도 표기 정규화 후에 하므로 두 번 틀린 말이 된다.
    """
    if track.lyric_match_type == "phonetic":
        return "lyrics_phonetic_priority"
    return "lyrics_exact_priority"


def _protect_phonetic_top1() -> bool:
    """검색 1위의 phonetic 표면 일치도 보호할지. 실험용 스위치(기본 꺼짐)."""
    raw = os.getenv("LYRIC_PROTECT_PHONETIC_TOP1", "")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def select_protected_lyric_ids(candidates: Sequence[MatchingTrack]) -> set:
    """리랭킹에서 일반 후보 아래로 내려가지 않게 보호할 곡.

    **라우터와 평가 스크립트가 이 함수를 함께 쓴다.** 보호 로직은 두 곳에 사본으로
    있어서, 규칙을 한쪽만 고치면 평가와 서비스가 다른 규칙으로 동작한다. 판정만이라도
    한 곳에 두면 그 어긋남이 생기지 않는다.

    `candidates`는 **보정까지 끝난 순위 순** 후보다. 0번이 검색 1위다.

    기본은 exact만 보호한다. 0.80은 문자 유사도 기준이 아니라 **단서 확신도**
    기준이다(`lyrics_exact_search.py`는 정규화 후 부분문자열로 발견되면 단서의
    confidence를 그대로 점수로 쓴다).

    `LYRIC_PROTECT_PHONETIC_TOP1`을 켜면 **검색 1위인 phonetic 표면 일치**도 보호한다.
    q219가 그 경우였다 — 가사 경로 기여가 텍스트의 7배인데 보호를 못 받아 리랭커가
    1위에서 3위로 내렸다. phonetic은 음차 추정이라 틀릴 수 있고 같은 구절이 여러 곡에
    걸칠 수 있으므로, 전면 보호가 아니라 **검색 1위 한 곡**으로 제한한다.
    """
    protected = {
        track.id
        for track in candidates
        if track.lyric_match_type == "exact"
        and float(track.lyric_match_score or 0.0) >= _lyric_protect_min_confidence()
    }
    if (
        _protect_phonetic_top1()
        and candidates
        and candidates[0].lyric_match_type == "phonetic"
        and float(candidates[0].lyric_match_score or 0.0)
        >= _lyric_protect_min_confidence()
    ):
        protected.add(candidates[0].id)
    return protected


def should_skip_rerank_for_image(reranker: Any, analysis: QueryAnalysis) -> bool:
    """이미지 지배 질의는 텍스트 LLM 리랭킹을 건너뛴다.

    SigLIP2가 실제 앨범 커버를 보고 만든 순위를, 이미지를 직접 보지 않는
    listwise reranker가 다시 섞어 회귀시키는 것을 막는다.
    mixed 질의(image < 0.70)는 기존 리랭킹을 그대로 사용한다.

    리랭커가 `skips_image_dominant`를 켠 경우에만 적용한다. 기존 Cross-Encoder는
    v0.5 기준선부터 이미지 질의도 리랭킹해 왔으므로, 여기서 함께 끄면 백엔드를
    바꾸지 않은 사람의 검색 결과까지 달라진다.
    """
    if not getattr(reranker, "skips_image_dominant", False):
        return False
    return bool(
        analysis.has_visual_clue
        and float(analysis.modality_weights.image or 0.0) >= 0.70
    )


# ---------------------------------------------------------------------------
# SearchRouter
# ---------------------------------------------------------------------------

def _path_result(
    name: str,
    result: Any,
    recorder: ExplainRecorder = NULL_RECORDER,
) -> List[MatchingTrack]:
    """경로 하나의 결과. 예외면 빈 결과로 대체하고 **그 사실을 남긴다.**

    `return_exceptions=True`로 모은 결과라 실패가 예외 객체로 들어온다. 여기서
    빈 리스트로 바꾸면 뒤쪽 융합 코드는 "기여가 없는 경로"와 똑같이 본다 —
    의도한 동작이다(한 경로가 죽어도 검색은 계속돼야 한다).

    다만 **둘은 다른 사실이다.** 로그에만 남기면 응답에는 가중치만 남아, 보는
    사람은 그 경로가 돌았지만 이 곡을 못 올린 것으로 읽는다. 기록에 남겨야
    화면이 "이미지 경로 실패"라고 말할 수 있다.
    """
    if not isinstance(result, BaseException):
        return result
    label = PATH_LABELS.get(name, name)
    logger.warning("[SearchRouter] %s 경로 실패 (빈 결과로 대체): %s", label, result)
    recorder.note_path_failed(name, f"{type(result).__name__}: {result}")
    return []


class SearchRouter:
    """
    멀티모달 병렬 검색 매니저.

    asyncio.gather + ThreadPoolExecutor 설계 이유:
    - SigLIP2 / CLAP embed_texts, Pinecone query 모두 동기(blocking) 호출
    - run_in_executor로 각 경로를 별도 스레드에 위임해야 진짜 병렬 실행
    - return_exceptions=True -> 개별 모달리티 실패가 전체를 중단시키지 않음

    파이프라인: 후보 생성(gating) -> RRF -> 정확 단서 부스팅 -> (선택) Cross-Encoder 리랭킹.
    """
    
    """멀티모달 후보 생성, RRF, 규칙 부스팅, Cross-Encoder 리랭킹을 담당"""

    def __init__(
        self,
        search_service: SearchService,
        image_embedder: SigLIP2Embedder,
        audio_embedder: CLAPAudioEmbedder,
        pinecone_client: Any,   # Pinecone 또는 같은 모양의 백엔드(qdrant_backend)
        lyrics_search_service: Optional[LyricsExactSearchService] = None,
        reranker: Optional[Any] = None,
        max_workers: int = 4,
    ):
        self._text_svc = search_service
        self._img_emb = image_embedder
        self._audio_emb = audio_embedder
        self._lyrics_svc = lyrics_search_service
        self._img_idx = pinecone_client.Index(IMAGE_INDEX_NAME)
        self._audio_idx = pinecone_client.Index(AUDIO_INDEX_NAME)
        # reranker: src.retrieval.reranker.MusicReranker (정현님 재랭킹 PR 병합 후 DI로 주입).
        # None이면 리랭킹 단계를 건너뛰고 RRF+부스팅 결과를 그대로 반환한다.
        self._reranker = reranker
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="search_worker",
        )

    def shutdown(self, wait: bool = True) -> None:
        """FastAPI lifespan 종료 시 스레드풀을 정리한다.

        **기본으로 기다린다.** 벡터 DB 클라이언트는 이 뒤에 닫히는데, 돌고 있는 검색
        스레드가 남아 있으면 그 작업이 닫힌 클라이언트를 만져 "QdrantLocal instance is
        closed"로 죽는다. 요청이 취소돼도(클라이언트가 연결을 끊어도) 스레드는 계속 돈다.

        cancel_futures로 아직 시작하지 않은 작업은 버리고, 이미 도는 작업만 기다린다.
        wait=False는 기다릴 수 없는 자리에서만 쓴다.
        """
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def search(
        self,
        analysis: QueryAnalysis,
        top_k: int = 10,
        use_rerank: bool = True,
        candidate_k: Optional[int] = None,
        force_weights: Optional[List[float]] = None,
        disable_boost: bool = False,
        exclude_ids: Optional[Iterable[str]] = None,
        candidate_ids_out: Optional[List[str]] = None,
        candidate_tracks_out: Optional[List[MatchingTrack]] = None,
        lyrics_snapshot_out: Optional[List[Any]] = None,
        answers: Optional[Sequence[ClarifyAnswer]] = None,
        answer_multiplier: Optional[float] = None,
        recorder: ExplainRecorder = NULL_RECORDER,
        timer: Any = timing.NULL_TIMER,
    ) -> List[MatchingTrack]:
        """
        QueryAnalysis를 받아 3개 경로를 병렬 실행하고
        RRF 퓨전 → 정확 단서 부스팅 → (선택) Cross-Encoder 리랭킹 후 top_k를 반환한다.

        use_rerank: True + reranker 주입됨 + 후보 2개 이상일 때만 리랭킹 수행.
            리랭킹 실패 시 예외를 흡수하고 부스팅 결과로 폴백한다.
        candidate_k: 리랭킹 전에 유지할 후보 수. 생략하면 max(top_k*3, 30) (최대 100).
        force_weights: [text, image, audio] ablation 용 수동 가중치 오버라이드.
            지정 시 Gemini 동적 가중치와 게이팅(has_visual_clue/intent)을 **모두 우회**한다.
            단, 해당 모달리티 전용 질의가 비어 있으면 오염 방지를 위해 그 경로는
            실행하지 않고 살아 있는 경로로 재정규화한다. weight=0 인 경로도 건너뛴다.
            예) [1,0,0]=text-only, [0,1,0]=image-only.
        disable_boost: True 면 RRF 이후 정확매칭 부스팅을 끄고 순수 RRF 점수만 사용.
            ablation 에서 검색 경로 기여만 isolate 할 때 사용(부스팅 confound 제거).
        exclude_ids: 후보 풀에서 뺄 song_id. 재질문 2턴에서 사용자가 "이 중에는
            없어요"로 거절한 곡을 제외하는 데 쓴다. 제외한 수만큼 후보 검색 폭을
            넓혀(pool_k) 최종 풀이 candidates개로 유지된다.
            None이면 폭도 그대로라 기존 동작과 완전히 동일하다.
        candidate_ids_out: 리스트를 넘기면 최종 후보 풀의 id가 순위 순으로 채워진다.
            반환 타입을 바꾸지 않고 후보 풀을 밖으로 내보내기 위한 통로다
            (호출부가 eval 스크립트 포함 3곳이라 반환 시그니처를 못 바꾼다).
        candidate_tracks_out: 같은 통로의 MatchingTrack 버전. 재질문 슬롯을 고르려면
            후보의 성별·장르 메타데이터가 필요한데 id만으로는 알 수 없다.
        answers: 재질문 답변. **질의를 바꾸지 않고** 후보 풀 안에서 재정렬만 한다.
            답변을 분석에 병합해 재검색하면 맞는 답변도 순위를 해친다
            (q200: 후보 4위 → 15위). 자세한 근거는 clarify.apply_answer_bonus 참조.
        """
        """
        1) 세 모달리티 병렬 후보 검색
        2) RRF 통합
        3) 제목/아티스트/가사 등 명시 단서 부스팅
        4) candidate_k개 Cross-Encoder 리랭킹
        5) top_k 반환
        """
        loop = asyncio.get_running_loop()
        candidates = candidate_k or max(top_k * 3, 30)
        candidates = min(100, max(top_k, candidates))
        excluded = set(exclude_ids) if exclude_ids else None
        # 거절한 수만큼 후보를 더 가져온다. 이걸 하지 않으면 각 경로와 RRF가
        # 이미 candidates개로 잘라버려서, 거절 후 후보 풀이 그만큼 얇아진다
        # (30 → 20). 자르기 순서만 바꿔서는 뒤에서 채워 올릴 곡 자체가 없다.
        # 제외가 없으면 candidates와 같으므로 기존 동작은 변하지 않는다.
        pool_k = min(100, candidates + len(excluded)) if excluded else candidates

        use_metadata_clue = (
            analysis.has_metadata_clue
            and force_weights is None
            and not disable_boost
        )

        use_performance_clue = (
            analysis.has_composite_performance_clue
            and force_weights is None
            and not disable_boost
        )
        performance_metadata_filter = _build_performance_metadata_filter(
            analysis
        )
        use_performance_metadata = bool(
            performance_metadata_filter is not None
            and force_weights is None
            and not disable_boost
        )
        title_meaning_filter = build_title_meaning_metadata_filter(
            analysis.title_meaning_clue
        )
        use_title_meaning = bool(
            title_meaning_filter is not None
            and force_weights is None
            and not disable_boost
        )
        use_balanced_semantic = bool(
            analysis.needs_balanced_semantic_recall
            and force_weights is None
            and not disable_boost
        )

        text_pool_k = (
            min(100, max(candidates * 3, 60, pool_k))
            if (use_metadata_clue or use_performance_clue)
            else pool_k
        )
        audio_pool_k = (
            min(100, max(candidates * 3, 60, pool_k))
            if use_performance_clue
            else pool_k
        )

        # 경로 게이팅 — 무관한 cross-modal 경로가 RRF 노이즈가 되는 것을 차단한다.
        #  · image: 앨범 표지를 가리키는 시각 단서(has_visual_clue)가 있을 때만 실행.
        #  · audio: 가사 키워드 정확매칭(intent=lyrics)은 답이 텍스트(BM25)에 있으므로
        #           CLAP cross-modal 은 노이즈만 더한다 → 차단.
        #           (2026-05-26 진단: 오마이걸 'da da da' 검색에서 텍스트는 정답을 1위로
        #            뽑았으나 audio 노이즈 + RRF 평탄화로 정답이 3위로 밀림)
        has_image_query = bool(analysis.image_english_query.strip())
        has_audio_query = bool(analysis.audio_english_query.strip())
        if force_weights is not None:
            # [ablation] 동적 가중치·의도 게이팅을 우회하되, 빈 전용 프롬프트를
            # 원본/타 모달리티 문자열로 메우지는 않는다.
            use_image = force_weights[1] > 0 and has_image_query
            use_audio = force_weights[2] > 0 and has_audio_query
        else:
            use_image = (
                analysis.has_visual_clue
                and has_image_query
                and analysis.modality_weights.image > 0
            )
            use_audio = (
                not analysis.has_exact_lyric_clue
                and has_audio_query
                and analysis.modality_weights.audio > 0
            )

        text_fut = loop.run_in_executor(
            self._pool,
            timer.job("path.text", self._search_text, analysis, text_pool_k),
        )
        image_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job("path.image", self._search_image, analysis, pool_k),
            )
            if use_image else None
        )
        audio_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.audio", self._search_audio, analysis, audio_pool_k
                ),
            )
            if use_audio else None
        )

        performance_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.performance",
                    self._search_performance_clues,
                    analysis,
                    min(60, max(candidates, 30)),
                ),
            )
            if use_performance_clue
            else None
        )

        performance_metadata_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.performance_metadata",
                    self._search_performance_metadata,
                    analysis,
                    candidates,
                ),
            )
            if use_performance_metadata
            else None
        )

        balanced_semantic_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.balanced_semantic",
                    self._search_balanced_semantic,
                    analysis,
                    candidates,
                ),
            )
            if use_balanced_semantic
            else None
        )

        use_title_constraint = has_title_constraints(
            analysis.title_constraints
        )

        title_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.title",
                    self._search_title_constrained,
                    analysis,
                    candidates,
                ),
            )
            if use_title_constraint
            else None
        )
        use_title_presence = (
            build_title_presence_metadata_filter(
                analysis.title_constraints
            )
            is not None
        )
        title_presence_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.title_presence",
                    self._search_title_presence,
                    analysis,
                    candidates,
                ),
            )
            if use_title_presence
            else None
        )
        title_meaning_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.title_meaning",
                    self._search_title_meaning,
                    analysis,
                    min(100, max(candidates * 3, 60)),
                ),
            )
            if use_title_meaning
            else None
        )
        use_lyrics_surface = bool(
            self._lyrics_svc is not None
            and force_weights is None
            and not disable_boost
            and any(
                clue.kind in {"verbatim", "partial", "phonetic"}
                for clue in analysis.lyric_clues
            )
        )
        lyrics_fut = (
            loop.run_in_executor(
                self._pool,
                timer.job(
                    "path.lyrics", self._search_lyrics, analysis, None,
                    lyrics_snapshot_out,
                ),
            )
            if use_lyrics_surface
            else None
        )
        if not use_image:
            logger.debug(
                "[SearchRouter] image 경로 skip "
                "(visual=%s prompt=%s weight=%.3f query='%s')",
                analysis.has_visual_clue,
                has_image_query,
                analysis.modality_weights.image,
                analysis.original_query,
            )
        if not use_audio:
            logger.debug(
                "[SearchRouter] audio 경로 skip "
                "(exact_lyrics=%s prompt=%s weight=%.3f query='%s')",
                analysis.has_exact_lyric_clue,
                has_audio_query,
                analysis.modality_weights.audio,
                analysis.original_query,
            )

        # skip 한 경로는 gather 에서 제외하고, 결과를 text/image/audio 슬롯에 재배치.
        futs = [
            f for f in (
                text_fut,
                image_fut,
                audio_fut,
                performance_fut,
                performance_metadata_fut,
                balanced_semantic_fut,
                title_fut,
                title_presence_fut,
                title_meaning_fut,
                lyrics_fut,
            )
            if f is not None
        ]
        with timer.step("search.paths", submitted=len(futs)):
            done = await asyncio.gather(*futs, return_exceptions=True)
        # 경로가 다 끝난 시각. 여기서부터 리랭킹 직전까지가 융합·부스팅이다.
        fuse_started = time.perf_counter()
        _it = iter(done)
        text_res = next(_it)
        image_res = next(_it) if use_image else []
        audio_res = next(_it) if use_audio else []
        performance_res = next(_it) if use_performance_clue else []
        performance_metadata_res = (
            next(_it) if use_performance_metadata else []
        )
        balanced_semantic_res = next(_it) if use_balanced_semantic else []
        raw = [text_res, image_res, audio_res]
        title_res = (
            next(_it)
            if use_title_constraint
            else []
        )
        title_presence_res = (
            next(_it)
            if use_title_presence
            else []
        )
        title_meaning_res = (
            next(_it)
            if use_title_meaning
            else []
        )
        lyrics_res = (
            next(_it)
            if use_lyrics_surface
            else []
        )

        # 실패한 경로는 빈 결과로 대체하고 기록에 남긴다
        text_hits, image_hits, audio_hits = [
            _path_result(name, result, recorder)
            for name, result in zip(["text_hybrid", "image", "audio"], raw)
        ]

        use_deep_audio_fusion = bool(
            use_performance_clue
            and not analysis.has_visual_clue
            and not analysis.has_any_lyric_clue
            and not analysis.song_title.strip()
            and not analysis.artist_name.strip()
            and not analysis.artist_name_alt
            and not has_title_constraints(analysis.title_constraints)
            and not analysis.has_title_meaning_clue
            and not analysis.has_artist_type_clue
        )

        if use_deep_audio_fusion:
            # q115처럼 제목/가사/가수 단서 없이 사운드만 상세히 묘사한 경우.
            base_text_hits = text_hits
            base_audio_hits = audio_hits
            fusion_top_k = max(
                pool_k,
                len(base_text_hits),
                len(image_hits),
                len(base_audio_hits),
            )
        else:
            # 일반 질의는 기존 상위 candidate_k를 보호한다.
            base_text_hits = text_hits[:pool_k]
            base_audio_hits = audio_hits[:pool_k]
            fusion_top_k = pool_k

        performance_path_hits = _path_result("performance", performance_res, recorder)

        performance_metadata_hits = _path_result(
            "performance_metadata", performance_metadata_res, recorder
        )

        balanced_semantic_hits = _path_result(
            "balanced_semantic", balanced_semantic_res, recorder
        )

        metadata_hits: list[tuple[MatchingTrack, float, int]] = []

        if use_metadata_clue:
            for original_rank, track in enumerate(text_hits, start=1):
                strength = _metadata_clue_strength(analysis, track)

                if strength > 0:
                    metadata_hits.append(
                        (track, strength, original_rank)
                    )

            metadata_hits.sort(
                key=lambda item: (-item[1], item[2])
            )

            metadata_hits = metadata_hits[:min(15, candidates)]

        title_hits = _path_result("title", title_res, recorder)
        title_presence_hits = _path_result("title_presence", title_presence_res, recorder)
        title_meaning_hits = _path_result("title_meaning", title_meaning_res, recorder)
        lyrics_hits = _path_result("lyrics_surface", lyrics_res, recorder)

        performance_hit_map: Dict[
            str,
            tuple[MatchingTrack, float, int],
        ] = {}
        if use_performance_clue:
            # dedicated 경로뿐 아니라 기존 text/audio의 깊은 후보에도 같은
            # 설명 가능한 점수를 적용한다. 어느 한 임베딩 경로의 실패에 덜 민감하다.
            ranked_sources = (
                performance_path_hits,
                text_hits,
                audio_hits,
            )
            for source_index, source in enumerate(ranked_sources):
                for source_rank, track in enumerate(source, start=1):
                    strength = _performance_clue_similarity(analysis, track)
                    if strength < 0.25:
                        continue
                    stable_rank = source_rank + source_index * 100
                    current = performance_hit_map.get(track.id)
                    candidate = (track, strength, stable_rank)
                    if current is None or (-strength, stable_rank) < (
                        -current[1],
                        current[2],
                    ):
                        performance_hit_map[track.id] = candidate
        performance_hits = sorted(
            performance_hit_map.values(),
            key=lambda item: (-item[1], item[2]),
        )[:min(15, candidates)]
            
        mw = analysis.modality_weights

        # 퓨전 가중치 — skip 한 경로(image/audio)는 0 으로 두고 살아있는 경로로
        # 재정규화한다(합 1.0 유지). Gemini 가 image/audio 에 가중치를 줘도 라우터
        # 단에서 확정적으로 0% 반영을 보장.
        if force_weights is not None:
            # [ablation] 전용 프롬프트가 없어 skip 된 경로는 0으로 만든 뒤
            # 실제 실행된 경로끼리 다시 정규화한다.
            active_forced = [
                force_weights[0],
                force_weights[1] if use_image else 0.0,
                force_weights[2] if use_audio else 0.0,
            ]
            s = sum(active_forced)
            weights = (
                [w / s for w in active_forced]
                if s > 0
                else [1.0, 0.0, 0.0]
            )
        else:
            w_text = mw.text
            w_image = mw.image if use_image else 0.0
            w_audio = mw.audio if use_audio else 0.0

            if (
                analysis.intent_type == "mixed"
                and bool((analysis.lyric_semantic_query or "").strip())
                and not analysis.has_lexical_lyric_clue
                and use_audio
                and not use_image
            ):
                w_text = max(w_text, 0.60)
                w_audio = min(w_audio, 0.40)

            total = w_text + w_image + w_audio
            if total > 0:
                weights = [w_text / total, w_image / total, w_audio / total]
            else:
                # 전 경로 가중치 0 인 비정상 케이스 — text 단독으로 폴백
                weights = [1.0, 0.0, 0.0]

        # 메타데이터 캐시 — 세 경로 모두 Pinecone에서 메타를 함께 받아오므로
        # (image/audio도 include_metadata=True) 어느 경로에서만 잡힌 곡이라도
        # 리랭킹용 문서를 만들 수 있다. text 우선, 그 다음 채워진 항목을 유지.
        meta_cache: Dict[str, MatchingTrack] = {}
        for track in [
            *text_hits,
            *image_hits,
            *audio_hits,
            *performance_path_hits,
            *performance_metadata_hits,
            *balanced_semantic_hits,
            *title_hits,
            *title_presence_hits,
            *title_meaning_hits,
            *lyrics_hits,
        ]:
            current = meta_cache.get(track.id)
            if current is None or current.title == "Unknown":
                meta_cache[track.id] = track

        # text metadata가 먼저 캐시된 경우에도 exact lyrics 일치 정보는 보존한다.
        for track in lyrics_hits:
            current = meta_cache.get(track.id)
            if current is None:
                meta_cache[track.id] = track
            else:
                meta_cache[track.id] = current.model_copy(
                    update={
                        "lyric_match_type": track.lyric_match_type,
                        "lyric_match_score": track.lyric_match_score,
                    }
                )

        # RRF 퓨전 — top_k보다 넉넉히 받아 부스팅/리랭킹으로 재정렬할 여지를 둔다
        #
        # text가 dense/sparse로 갈라지지 않는 이유는 explain.PATH_LABELS 주석 참조 —
        # Qdrant 하이브리드가 두 점수를 합쳐 돌려주므로 여기서는 이미 한 경로다.
        recorder.set_weights(*(list(weights) + [0.0, 0.0, 0.0])[:3])
        fused = _rrf_fuse(
            ranked_lists=[
                [(t.id, t.score) for t in base_text_hits],
                [(t.id, t.score) for t in image_hits],
                [(t.id, t.score) for t in base_audio_hits],
            ],
            weights=weights,
            top_k=fusion_top_k,
            labels=["text_hybrid", "image", "audio"],
            recorder=recorder,
        )

        # full_lyrics에서 확인된 표면 구절 후보를 RRF 절단 전에 합류시킨다.
        # exact 후보는 일반 RRF 최대 점수보다 충분히 큰 보너스로 보호하고,
        # fuzzy/phonetic 후보는 리랭커가 뒤집을 여지를 남긴다.
        protected_lyric_ids: set[str] = set()
        if lyrics_hits:
            fused_scores = dict(fused)
            boost_unit = 1.0 / (_RRF_K + 1)
            # 부스트 직전 융합 순위. 부스트가 몇 칸을 올렸는지 보려면 필요하다.
            # **기록 전용이며 아래 계산에 쓰이지 않는다.**
            rank_before_boost = {
                song_id: index for index, (song_id, _) in enumerate(fused, start=1)
            }
            boost_scale = _lyric_boost_scale()
            for rank, track in enumerate(lyrics_hits, start=1):
                match_type = track.lyric_match_type or "fuzzy"
                match_score = float(track.lyric_match_score or 0.0)
                multiplier = {
                    "exact": 8.0,
                    "phonetic": 5.0,
                    "fuzzy": 3.0,
                }.get(match_type, 2.0)
                # exact 후보끼리는 Mongo의 자연 순서가 검색 순위를 좌우하면 안 된다.
                # 같은 구절이 여러 곡에 있는 경우 text RRF와 성별/장르/아티스트 등
                # 사용자 단서가 순위를 정하게 하고, fuzzy만 순위를 감쇠한다.
                rank_decay = (
                    1.0
                    if match_type == "exact"
                    else (_RRF_K + 1) / (_RRF_K + rank)
                )
                # match_score의 의미가 유형마다 다르다(lyrics_exact_search.py).
                # exact/phonetic은 표면 문자열이 원문에 그대로 있었던 경우라 점수가
                # **단서의 확신도**이고, fuzzy만 유사도가 섞인다. 전부 "일치도"라고
                # 적으면 확신도를 문자 유사도로 읽게 된다.
                _fuse_add(
                    fused_scores, track.id,
                    boost_unit * multiplier * match_score * rank_decay * boost_scale,
                    recorder, "lyrics_surface", rank,
                    f"{LYRIC_MATCH_LABELS.get(match_type, match_type)} "
                    f"×{multiplier:.1f}, "
                    f"{LYRIC_SCORE_LABELS.get(match_type, '점수')} {match_score:.2f}"
                    + _lyric_match_note(track),
                )
                detail_model = getattr(track, "lyric_match_detail", None)
                recorder.note_lyric_match(
                    track.id,
                    detail_model.model_dump() if detail_model is not None else None,
                )
                # 보호 판정은 여기서 하지 않는다. "검색 1위"를 알아야 하므로
                # 보정이 끝난 후보 목록이 나온 뒤 select_protected_lyric_ids()로 한다.
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            recorder.note_lyric_boost_ranks(
                rank_before_boost,
                {song_id: index for index, (song_id, _) in enumerate(fused, start=1)},
            )

        if title_hits:
            title_confidence = analysis.title_constraints.confidence

            # title constraint는 기존 모달리티를 대체하지 않는
            # 보조 신호이므로 비교적 작은 RRF 가중치를 사용.
            title_weight = 0.35 * title_confidence

            fused_scores = dict(fused)

            for rank, track in enumerate(title_hits, start=1):
                bonus = title_weight / (_RRF_K + rank)

                _fuse_add(
                    fused_scores, track.id, bonus, recorder, "title", rank,
                    f"확신도 {title_confidence:.2f}",
                )

            # 이 시점에서는 아직 candidates로 자르지 않는다.
            # title 경로에서 새로 들어온 후보를 boost까지 보내야 함.
            fused = sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

        if title_presence_hits:
            # 기존 title_script 경로는 그대로 두고, 괄호 속 한자 표기 후보만
            # 별도의 매우 약한 신호로 합류시킨다. 다른 제목 유형에는 실행되지 않는다.
            title_presence_weight = (
                0.35 * analysis.title_constraints.confidence
            )
            fused_scores = dict(fused)
            for rank, track in enumerate(title_presence_hits, start=1):
                _fuse_add(
                    fused_scores, track.id,
                    title_presence_weight / (_RRF_K + rank),
                    recorder, "title_presence", rank,
                )
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if title_meaning_hits:
            # 의미 기억은 구조 단서보다 불확실하므로 기존 결과를 제거하지 않고,
            # 해당 단서 전용의 좁은 보조 경로에만 낮은 RRF 기여를 준다.
            title_meaning_weight = (
                0.50 * analysis.title_meaning_clue.confidence
            )
            fused_scores = dict(fused)
            for rank, track in enumerate(title_meaning_hits, start=1):
                _fuse_add(
                    fused_scores, track.id,
                    title_meaning_weight / (_RRF_K + rank),
                    recorder, "title_meaning", rank,
                )
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if balanced_semantic_hits:
            # 원래 alpha 결과는 그대로 유지한다. alpha=0.60에서만 살아나는
            # 후보를 작은 보조 신호로 합류시켜 q112 유형의 회귀만 복구한다.
            fused_scores = dict(fused)
            balanced_weight = 0.30
            for rank, track in enumerate(balanced_semantic_hits, start=1):
                _fuse_add(
                    fused_scores, track.id,
                    balanced_weight / (_RRF_K + rank),
                    recorder, "balanced_semantic", rank,
                )
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if performance_hits:
            fused_scores = dict(fused)
            performance_weight = (
                0.55 * analysis.performance_clues.confidence
            )
            for auxiliary_rank, (
                track,
                strength,
                _source_rank,
            ) in enumerate(performance_hits, start=1):
                _fuse_add(
                    fused_scores, track.id,
                    performance_weight * strength / (_RRF_K + auxiliary_rank),
                    recorder, "performance", auxiliary_rank,
                    f"단서 강도 {strength:.2f}",
                )
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if performance_metadata_hits:
            # 구조 메타데이터 경로는 후보 소환만 담당한다. 전역 hard filter가
            # 아니며, 실제 순위는 아래 explicit performance/gender/genre boost가
            # 다시 검증한다. 낮은 RRF 기여로 기존 후보 순서를 최대한 보존한다.
            fused_scores = dict(fused)
            performance_metadata_weight = (
                0.5 * analysis.performance_clues.confidence
            )
            for rank, track in enumerate(
                performance_metadata_hits,
                start=1,
            ):
                _fuse_add(
                    fused_scores, track.id,
                    performance_metadata_weight / (_RRF_K + rank),
                    recorder, "performance_metadata", rank,
                )
            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if metadata_hits:
            fused_scores = dict(fused)

            metadata_weight = 0.20

            for auxiliary_rank, (
                track,
                strength,
                _original_rank,
            ) in enumerate(metadata_hits, start=1):
                bonus = (
                    metadata_weight
                    * strength
                    / (_RRF_K + auxiliary_rank)
                )

                _fuse_add(
                    fused_scores, track.id, bonus, recorder, "metadata",
                    auxiliary_rank, f"단서 강도 {strength:.2f}",
                )

            fused = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        if not fused:
            logger.error(
                "[SearchRouter] 전체 경로 결과 없음 — 빈 리스트 반환 (query='%s')",
                analysis.original_query,
            )
            return []
        
        for song_id, fused_score in fused:
            recorder.set_fused(song_id, fused_score)

        boosted = self._apply_explicit_boosts(
            analysis,
            fused,
            meta_cache,
            disable_boost=disable_boost,
            recorder=recorder,
        )

        boosted = _reject_and_trim(boosted, excluded, candidates)

        # 답변은 여기서만 반영한다 — 후보 풀은 그대로 두고 순위만 바꾼다.
        if answers:
            boosted = apply_answer_bonus(
                boosted, meta_cache, answers, 1.0 / (_RRF_K + 1),
                multiplier=answer_multiplier,
                recorder=recorder,
            )

        for song_id, retrieval_score in boosted:
            recorder.set_retrieval(song_id, retrieval_score)
        recorder.set_rank_before([song_id for song_id, _ in boosted])

        if candidate_ids_out is not None:
            candidate_ids_out.extend(song_id for song_id, _ in boosted)

        candidate_tracks: List[MatchingTrack] = []
        for song_id, retrieval_score in boosted:
            base_track = meta_cache.get(song_id)
            if base_track is None:
                base_track = MatchingTrack(
                    id=song_id,
                    score=retrieval_score,
                    title="Unknown",
                    artist="Unknown",
                )
                logger.debug("[SearchRouter] metadata 없는 id=%s", song_id)

            candidate_tracks.append(
                base_track.model_copy(
                    update={
                        "score": float(retrieval_score),
                        "retrieval_score": float(retrieval_score),
                        "rerank_score": None,
                    }
                )
            )

        if candidate_tracks_out is not None:
            candidate_tracks_out.extend(candidate_tracks)

        # 보호 판정은 평가 스크립트와 **같은 함수**로 한다(사본이 갈리지 않게).
        protected_lyric_ids = select_protected_lyric_ids(candidate_tracks)

        recorder.set_reranker(
            type(self._reranker).__name__ if self._reranker is not None else "none"
        )
        timer.mark("search.fuse", fuse_started, candidates=len(candidate_tracks))

        # 리랭킹은 빠져나가는 자리가 셋(보호 그룹·일반·폴백)이라 블록으로 묶지 않고
        # 각 출구에서 닫는다. None이면 리랭킹을 아예 시도하지 않은 것이다.
        rerank_started: Optional[float] = None
        if (
            use_rerank
            and self._reranker is not None
            and self._reranker.enabled
            and len(candidate_tracks) > 1
            and not should_skip_rerank_for_image(self._reranker, analysis)
        ):
            recorder.set_reorder_attempted(True)
            rerank_started = time.perf_counter()
            try:
                # full_lyrics 연속 구절 exact 일치는 검색의 가장 강한 증거다.
                # exact 그룹은 그룹 내부에서만 리랭킹하고 일반 후보 아래로는 내리지 않는다.
                protected_tracks = [
                    track
                    for track in candidate_tracks
                    if track.id in protected_lyric_ids
                ]
                other_tracks = [
                    track
                    for track in candidate_tracks
                    if track.id not in protected_lyric_ids
                ]

                if protected_tracks:
                    # 같은 가사 구절이 여러 곡에 실제로 존재하면 명시된 성별/가수/
                    # 장르가 일치하는 tier를 먼저 둔 뒤, 같은 tier 안에서만 리랭킹한다.
                    # 정정된 답변을 반영한 사본으로 묶는다. 원래 분석을 쓰면
                    # "남성"이라고 잘못 기억했다가 "여성"으로 고쳐도 남성 그룹이
                    # 먼저 와서, 보너스로 올라온 정답이 여기서 다시 밀려난다.
                    ranking_analysis = analysis_with_answers(analysis, answers or [])
                    tiered: Dict[float, List[MatchingTrack]] = {}
                    for track in protected_tracks:
                        tiered.setdefault(
                            exact_lyric_constraint_score(ranking_analysis, track),
                            [],
                        ).append(track)

                    ordered_protected: List[MatchingTrack] = []
                    for tier in sorted(tiered, reverse=True):
                        group = tiered[tier]
                        remaining_group_slots = top_k - len(ordered_protected)
                        if remaining_group_slots <= 0:
                            break
                        if len(group) > 1:
                            protected_job = partial(
                                call_reranker,
                                self._reranker,
                                analysis.original_query,
                                group,
                                min(remaining_group_slots, len(group)),
                                answers,
                                recorder,
                            )
                            group = await loop.run_in_executor(
                                self._pool,
                                timer.job(
                                    "rerank.protected_group",
                                    protected_job,
                                    size=len(group),
                                ),
                            )
                        ordered_protected.extend(group[:remaining_group_slots])
                    protected_tracks = ordered_protected

                    remaining = max(0, top_k - len(protected_tracks))
                    if remaining and other_tracks:
                        other_job = partial(
                            call_reranker,
                            self._reranker,
                            analysis.original_query,
                            other_tracks,
                            remaining,
                            answers,
                            recorder,
                        )
                        other_tracks = await loop.run_in_executor(
                            self._pool,
                            timer.job(
                                "rerank.others",
                                other_job,
                                size=len(other_tracks),
                            ),
                        )
                    else:
                        other_tracks = []
                    final = [*protected_tracks, *other_tracks][:top_k]
                    # 이 자리까지 와야 보호 배치가 실제로 성사된 것이다. 위에서
                    # 미리 기록하면 리랭킹이 실패해 폴백했을 때 취소된 규칙이 남는다.
                    final_ids = {t.id for t in final}
                    for track in protected_tracks:
                        if track.id in final_ids:
                            recorder.order(
                                track.id,
                                lyric_priority_rule(track),
                                "일반 후보 아래로 내려가지 않음",
                            )
                    # 보호 곡이 앞에 배치됐다는 **사실**만 남긴다. 하락 폭이
                    # 아니다 — 보호 곡이 원래 앞에 있었으면 순위는 그대로다.
                    recorder.note_behind_protected(
                        [t.id for t in other_tracks if t.id in final_ids],
                        len(protected_tracks),
                    )
                    recorder.note_order_rules_applied()
                    recorder.commit_reorder()
                    recorder.set_rank_after([t.id for t in final])
                    for t in final:
                        recorder.set_rerank_score(t.id, t.rerank_score)
                    timer.mark("search.rerank", rerank_started, shape="protected")
                    return final

                rerank_job = partial(
                    call_reranker,
                    self._reranker,
                    analysis.original_query,
                    candidate_tracks,
                    top_k,
                    answers,
                    recorder,
                )
                reranked = await loop.run_in_executor(
                    self._pool,
                    timer.job(
                        "rerank.main", rerank_job, size=len(candidate_tracks)
                    ),
                )
                recorder.commit_reorder()
                recorder.set_rank_after([t.id for t in reranked])
                for t in reranked:
                    recorder.set_rerank_score(t.id, t.rerank_score)
                timer.mark("search.rerank", rerank_started, shape="plain")
                return reranked
            except Exception as exc:
                # 폴백도 설명해야 한다 — "리랭킹이 실패해서 검색 순서 그대로"가
                # 사용자에게는 순위의 진짜 이유다.
                # 앞 그룹이 성공했어도 최종 결과는 검색 순서로 돌아간다.
                # 성공 이력만 남기고 **반영 표시는 취소**해야 설명이 맞는다.
                recorder.note_rerank_run(RERANK_FAILED, [])
                recorder.revoke_reorder()
                recorder.set_rerank_error(f"{type(exc).__name__}: {exc}")
                logger.exception(
                    "[SearchRouter] 리랭킹 실패 — 기존 검색 순서로 폴백: %s",
                    exc,
                )

        if rerank_started is not None:
            # 여기까지 왔다는 건 리랭킹이 예외로 멈췄다는 뜻이다. 실패한 시도의
            # 시간도 응답 지연에는 그대로 들어간다.
            timer.mark("search.rerank", rerank_started, shape="failed")
        final = candidate_tracks[:top_k]
        recorder.set_rank_after([t.id for t in final])
        return final

    @staticmethod
    def _apply_explicit_boosts(
        analysis: QueryAnalysis,
        fused: List[Tuple[str, float]],
        meta_cache: Dict[str, MatchingTrack],
        disable_boost: bool = False,
        recorder: ExplainRecorder = NULL_RECORDER,
    ) -> List[Tuple[str, float]]:
        """기존 RRF 점수에 사용자가 명시한 정확 단서 보너스/페널티를 적용한다.

        recorder는 어떤 규칙이 얼마를 더했는지만 남긴다. 점수 계산은 건드리지 않는다 —
        `score += delta`가 클로저 안으로 들어갔을 뿐 연산 순서와 값이 같다.
        """
        # 정확 매칭 부스팅: RRF 점수 분포가 평탄하므로(0.008~0.013 수준),
        # 사용자가 명시한 단서가 곡 메타와 정확히 맞으면 보너스를 더해 상위로 끌어올린다.
        # boost_unit ≈ RRF 1순위 기여도(1/(K+1)). 신호 강도에 따라 배수를 다르게 준다.
        # 가사 키워드를 토큰 단위로 분해 — "da da da da~" 같은 의성어 후크나 통문장도
        # 토큰('da')으로 매칭되게 한다. 기존 통문장 substring 매칭(`k in hay`)은
        # highlight 표기('da da da...\n')와 물결표·줄바꿈 차이로 항상 실패했다 (2026-05-26).
        kw_tokens = set()
        for k in analysis.lyric_keywords:
            kw_tokens.update(_tokenize(k))
        surface_clues = [
            clue
            for clue in analysis.lyric_clues
            if clue.kind in {"verbatim", "partial", "phonetic"}
        ]
        boost_unit = 1.0 / (_RRF_K + 1)
        want_title = _norm(analysis.song_title)
        want_artist = _norm(analysis.artist_name)
        want_gender = analysis.vocal_gender
        want_genre = _norm(analysis.genre)
        title_constraints = analysis.title_constraints
        title_meaning_clue = analysis.title_meaning_clue

        boosted: List[Tuple[str, float]] = []
        for song_id, rrf_score in fused:
            score = rrf_score
            track = meta_cache.get(song_id)

            def bump(rule: str, delta: float, detail: str = "", _id=song_id) -> None:
                """점수를 더하고 같은 값을 기록한다. 감점은 delta가 음수다."""
                nonlocal score
                score += delta
                recorder.adjust(_id, rule, delta, detail)

            if track is not None and not disable_boost:
                # 가사 키워드 매칭 (highlight/summary 토큰과 교집합)
                lyric_haystack = " ".join(
                    filter(None, [track.lyrics_highlight, track.lyrics_summary])
                )
                normalized_haystack = normalize_lyric_surface(lyric_haystack)

                # full_lyrics 경로가 이미 일치를 확인한 후보에는 highlight/summary
                # 보너스를 다시 주지 않는다. 같은 가사 증거를 중복 계산하면 대표 구절로
                # 선정된 곡이 성별 같은 보조 단서를 압도한다(q212 회귀 원인).
                if track.lyric_match_type is None:
                    # exact 경로가 일시적으로 실패했을 때 후보 안의 정답을 살리는 안전망.
                    for clue in surface_clues:
                        clue_surfaces = [clue.text, *clue.variants]
                        if any(
                            (normalized_clue := normalize_lyric_surface(surface))
                            and normalized_clue in normalized_haystack
                            for surface in clue_surfaces
                        ):
                            multiplier = 4.0 if clue.kind != "phonetic" else 2.5
                            # 어느 필드에서 맞았는지는 **보고용으로만** 다시 본다.
                            # 점수 조건은 위 haystack 그대로다. highlight는 발췌이고
                            # summary는 요약이므로 "가사에 있다"는 말의 근거가 다르다.
                            field = _matched_lyric_field(clue_surfaces, track)
                            kind = "phonetic" if clue.kind == "phonetic" else "clue"
                            bump(
                                f"lyric_{kind}_in_{field}",
                                boost_unit * multiplier * clue.confidence,
                                f"{clue.kind} ×{multiplier:.1f}",
                            )
                            break

                    if kw_tokens:
                        hay_tokens = set(_tokenize(lyric_haystack))
                        matched = kw_tokens & hay_tokens
                        if matched:
                            coverage = len(matched) / len(kw_tokens)
                            # 일부 흔한 단어 하나만 겹치는 경우의 과도한 부스팅을 막는다.
                            multiplier = 2.0 if coverage == 1.0 else 0.5 * coverage
                            bump("lyric_keyword_tokens", boost_unit * multiplier, f"겹침 {len(matched)}/{len(kw_tokens)}")

                # 제목 완전일치: 가장 강한 신호 → 큰 보너스
                if want_title and _norm(track.title) == want_title:
                    bump("title_exact", boost_unit * 5)

                # 제목 구조 soft boost
                if has_title_constraints(title_constraints):
                    match_ratio = title_constraint_match_ratio(
                        track.title,
                        title_constraints,
                    )

                    if match_ratio > 0:
                        bump(
                            "title_constraints",
                            boost_unit * 1.0 * match_ratio * title_constraints.confidence,
                            f"구조 일치 {match_ratio:.2f}",
                        )

                    # 대표 제목은 한글이지만 괄호 속 정식 표기가 한자인 경우.
                    # hanja 질의에서만 동작하므로 Latin/숫자/반복 제목 순위에는
                    # 영향을 주지 않으며, 잘못 기억한 성별보다 희소 제목 단서를 우선한다.
                    presence_ratio = title_presence_match_ratio(
                        track.title,
                        title_constraints,
                    )
                    if presence_ratio > 0:
                        bump(
                            "title_hanja_presence",
                            boost_unit * 2.0 * presence_ratio * title_constraints.confidence,
                            f"표기 일치 {presence_ratio:.2f}",
                        )

                # 제목 의미/유형은 확정 일치가 아니므로 양성 증거만 약하게 더한다.
                # foreign_person_name의 Latin 표기는 제목 그 자체가 아닌 근사 신호다.
                if analysis.has_title_meaning_clue:
                    title_meaning_ratio = title_meaning_match_ratio(
                        track.title,
                        title_meaning_clue,
                    )
                    if title_meaning_ratio > 0:
                        bump(
                            "title_meaning",
                            boost_unit * 1.25 * title_meaning_ratio * title_meaning_clue.confidence,
                            f"의미 일치 {title_meaning_ratio:.2f}",
                        )

                # 아티스트 일치: 같은 가수 곡이 묶이므로 단독으론 변별력↓ → 작은 보너스
                if want_artist and track.artist and want_artist in _norm(track.artist):
                    bump("artist_match", boost_unit * 1)

                # vocal_gender soft matching
                #
                # Vague-Finder의 사용자 기억은 틀릴 수 있으므로 성별 불일치를
                # 강한 hard signal로 취급하지 않는다.
                #
                # 특히 실제 곡이 "혼성"인 경우:
                #   사용자="여성", 실제="혼성"
                #   사용자="남성", 실제="혼성"
                # 는 해당 성별 보컬이 실제로 포함되어 있을 수 있으므로
                # 부분 일치로 처리한다.
                if want_gender and track.vocal_gender:
                    actual_gender = str(track.vocal_gender).strip()

                    if actual_gender == want_gender:
                        bump("vocal_gender_match", boost_unit * 1.0, f"{actual_gender}")

                    elif (
                        actual_gender == "혼성"
                        and want_gender in ("남성", "여성")
                    ):
                        bump("vocal_gender_partial", boost_unit * 0.25, f"{actual_gender}")

                    elif (
                        want_gender == "혼성"
                        and actual_gender in ("남성", "여성")
                    ):
                        bump("vocal_gender_mismatch", -(boost_unit * 0.5), f"{actual_gender}")

                    elif {
                        want_gender,
                        actual_gender,
                    } == {"남성", "여성"}:
                        bump("vocal_gender_mismatch", -(boost_unit * 1.0), f"{actual_gender}")

                # 발매 시기 soft boost
                era_similarity = _release_era_similarity(
                    analysis,
                    track,
                )

                if era_similarity > 0:
                    bump(
                        "release_era",
                        boost_unit * 1.25 * analysis.release_era.confidence * era_similarity,
                        f"시기 유사 {era_similarity:.2f}",
                    )

                # 솔로/그룹/듀오/밴드 soft boost
                artist_type_similarity = _artist_type_similarity(
                    analysis,
                    track,
                )

                if artist_type_similarity > 0:
                    bump(
                        "artist_type",
                        boost_unit * 1.0 * analysis.artist_type.confidence * artist_type_similarity,
                        f"형태 유사 {artist_type_similarity:.2f}",
                    )

                # 보컬 역할/효과음은 composite 또는 희소 단서가 있을 때만
                # 후보 확장과 동일한 설명 가능한 점수를 사용한다. 불일치 페널티는 없다.
                performance_similarity = _performance_clue_similarity(
                    analysis,
                    track,
                )
                if performance_similarity > 0:
                    bump(
                        "performance_clues",
                        boost_unit * 1.50 * analysis.performance_clues.confidence * performance_similarity,
                        f"단서 유사 {performance_similarity:.2f}",
                    )

                # genre는 멜론 세분류와 사용자 체감이 자주 어긋나므로
                # 페널티 없이 일치 시 약한 보너스만 적용
                if (
                    want_genre
                    and track.genre
                    and want_genre in _norm(track.genre)
                ):
                    bump("genre_match", boost_unit * 1)

            boosted.append((song_id, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    # ------------------------------------------------------------------
    # Private: 동기 검색 메서드 (ThreadPoolExecutor 안에서 실행)
    # ------------------------------------------------------------------

    def _search_text(self, analysis: QueryAnalysis, top_k: int) -> List[MatchingTrack]:
        """KoE5(dense) + BM25(sparse) 하이브리드 검색."""
        # dense/sparse 비중은 modality가 아닌 전용 text_alpha 사용
        alpha = analysis.text_alpha

        # sparse(BM25)에는 키워드 텍스트만 투입한다.
        # KoE5(dense)와 BM25(sparse) 모두 한국어 인덱스 — 영문 번역은 노이즈이므로 제외.
        # 대신 artist_name_alt 로 한국어↔영문 alias 를 명시적으로 결합해 토큰화 mismatch 해소
        # (q004 "뉴진스 노래" → metadata="NewJeans" 매칭 실패 → R@10=0 베이스라인 진단).
        exact_lyric_terms = _exact_lyric_terms(analysis)
        phonetic_lyric_terms = _phonetic_lyric_terms(analysis)
        if exact_lyric_terms:
            # 가사 구절과 일반 분위기 태그를 한 sparse 벡터에 섞으면 IDF 정규화로
            # 구절 신호가 희석된다. lexical lyrics 전용 질의로 분리한다.
            sparse_terms = exact_lyric_terms
            alpha = min(alpha, 0.10)
        elif phonetic_lyric_terms:
            # 음차 원문과 모델이 복원한 원어 후보를 모두 BM25에 넣되, 철자 복원이
            # 틀렸을 때 후보군 전체가 사라지지 않도록 dense를 절반 이상 유지한다.
            sparse_terms = _dedupe_terms(
                [*phonetic_lyric_terms, *analysis.korean_tags]
            )
            alpha = min(0.60, max(alpha, 0.50))
        else:
            sparse_terms = list(analysis.korean_tags)
        if analysis.song_title:
            sparse_terms.append(analysis.song_title)
        if analysis.artist_name:
            sparse_terms.append(analysis.artist_name)
        # 메타데이터 표기 형태 모두 결합 — 인덱스 토큰과 매칭 가능성 ↑
        sparse_terms.extend(a for a in analysis.artist_name_alt if a)
        sparse_query = " ".join(sparse_terms) if sparse_terms else None
        dense_query = analysis.original_query
        if analysis.lyric_semantic_query and not exact_lyric_terms:
            dense_query = (
                f"{analysis.original_query}\n"
                f"가사 의미: {analysis.lyric_semantic_query}"
            )
        if phonetic_lyric_terms and not exact_lyric_terms:
            dense_query = (
                f"{dense_query}\n"
                f"들리는 가사의 가능한 표기: {' / '.join(phonetic_lyric_terms)}"
            )

        logger.debug(
            "[SearchRouter/text] query='%s' alpha=%.2f sparse='%s' top_k=%d",
            analysis.original_query, alpha, sparse_query, top_k,
        )
        return self._text_svc.search_text(
            dense_query,
            top_k=top_k,
            alpha=alpha,
            sparse_query=sparse_query,
        )

    def _search_balanced_semantic(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """기존 dense 중심 결과를 대체하지 않는 alpha=0.60 보조 검색."""
        balanced = analysis.model_copy(update={"text_alpha": 0.60})
        return self._search_text(balanced, top_k)

    def _search_performance_clues(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """보컬 역할·희소 효과음만 압축한 조건부 text hybrid 보조 경로."""
        clue = analysis.performance_clues
        count_labels = {
            "solo": "한 명의 솔로 보컬",
            "duet": "두 사람이 함께하는 듀엣",
            "multiple": "여러 명의 보컬",
            "choir": "합창과 콰이어",
        }
        clue_terms = [
            count_labels.get(clue.vocal_count or "", ""),
            *clue.vocal_roles,
            *clue.sound_ensemble,
        ]
        clue_terms = [term for term in clue_terms if term]
        dense_parts = [
            analysis.original_query,
            f"곡에서 들리는 단서: {' / '.join(clue_terms)}",
        ]
        if analysis.lyric_semantic_query:
            dense_parts.append(
                f"가사 의미: {analysis.lyric_semantic_query}"
            )
        sparse_terms = _dedupe_terms(
            [
                *analysis.korean_tags,
                *clue.vocal_roles,
                *clue.sound_ensemble,
                analysis.genre,
            ]
        )
        logger.debug(
            "[SearchRouter/performance] clues=%s top_k=%d",
            clue_terms,
            top_k,
        )
        return self._text_svc.search_text(
            "\n".join(dense_parts),
            top_k=top_k,
            alpha=min(0.60, max(0.45, analysis.text_alpha)),
            sparse_query=" ".join(sparse_terms) if sparse_terms else None,
        )

    def _search_performance_metadata(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """공연 역할의 고신뢰 메타데이터 조합을 이용한 additive dense 경로."""
        metadata_filter = _build_performance_metadata_filter(analysis)
        if not metadata_filter:
            return []

        clue = analysis.performance_clues
        clue_terms = [
            clue.vocal_count or "",
            *clue.vocal_roles,
            *clue.sound_ensemble,
        ]
        dense_parts = [
            analysis.original_query,
            "구조적 공연 단서: "
            + " / ".join(term for term in clue_terms if term),
        ]
        if analysis.lyric_semantic_query:
            dense_parts.append(
                f"가사 의미: {analysis.lyric_semantic_query}"
            )

        logger.debug(
            "[SearchRouter/performance-metadata] filter=%s top_k=%d",
            metadata_filter,
            top_k,
        )
        return self._text_svc.search_text(
            "\n".join(dense_parts),
            top_k=top_k,
            alpha=1.0,
            metadata_filter=metadata_filter,
        )

    def _search_title_meaning(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """제목 의미/유형을 기존 메타데이터로 근사하는 additive 보조 경로."""
        metadata_filter = build_title_meaning_metadata_filter(
            analysis.title_meaning_clue
        )
        if not metadata_filter:
            return []
        sparse_terms = _dedupe_terms(
            [*analysis.korean_tags, analysis.genre]
        )
        dense_query = (
            f"{analysis.original_query}\n"
            f"제목 의미 단서: {analysis.title_meaning_clue.text}"
        )
        logger.debug(
            "[SearchRouter/title-meaning] kind=%s filter=%s top_k=%d",
            analysis.title_meaning_clue.kind,
            metadata_filter,
            top_k,
        )
        # 외국 사람 이름이라는 기억은 특정 어휘가 아니라 제목의 의미/유형
        # 단서다. Latin 존재 필터가 이미 후보를 제한하므로 BM25 일반 태그가
        # Catallena 같은 정답을 밀어내지 않도록 이 경우에만 dense-only로 찾는다.
        alpha = (
            1.0
            if analysis.title_meaning_clue.kind == "foreign_person_name"
            else min(0.65, max(0.50, analysis.text_alpha))
        )
        hits = self._text_svc.search_text(
            dense_query,
            top_k=top_k,
            alpha=alpha,
            sparse_query=" ".join(sparse_terms) if sparse_terms else None,
            metadata_filter=metadata_filter,
        )
        if analysis.title_meaning_clue.kind == "foreign_person_name":
            # Dense 유사도는 '인도풍 여자 그룹' 같은 곡 설명을 잘 잡지만
            # 제목이 실제로 사람 이름처럼 보이는지는 구분하지 못한다.
            # 전용 경로 안에서만 결정론적인 제목 형태 점수를 먼저 적용하고,
            # 동점일 때 기존 Pinecone 순서를 유지한다.
            hits = [
                track
                for _original_rank, track in sorted(
                    enumerate(hits),
                    key=lambda item: (
                        -title_meaning_match_ratio(
                            item[1].title,
                            analysis.title_meaning_clue,
                        ),
                        item[0],
                    ),
                )
            ]
        return hits

    def _search_title_constrained(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """
        사용자가 기억한 제목 구조와 일치하는 곡만 대상으로
        원래 query의 text hybrid 검색을 수행한다.

        전체 검색을 hard filter하지 않고 별도 보조 경로로만 사용한다.
        """
        metadata_filter = build_title_metadata_filter(
            analysis.title_constraints
        )

        if not metadata_filter:
            return []

        # 기존 BM25 query와 동일한 구성 사용
        exact_lyric_terms = _exact_lyric_terms(analysis)
        phonetic_lyric_terms = _phonetic_lyric_terms(analysis)
        lyric_terms = exact_lyric_terms or phonetic_lyric_terms
        sparse_terms = lyric_terms or list(analysis.korean_tags)

        if analysis.song_title:
            sparse_terms.append(analysis.song_title)

        if analysis.artist_name:
            sparse_terms.append(analysis.artist_name)

        sparse_terms.extend(
            a for a in analysis.artist_name_alt if a
        )

        sparse_query = (
            " ".join(sparse_terms)
            if sparse_terms
            else None
        )

        logger.debug(
            "[SearchRouter/title] filter=%s query='%s' top_k=%d",
            metadata_filter,
            analysis.original_query,
            top_k,
        )

        dense_query = analysis.original_query
        alpha = analysis.text_alpha
        if phonetic_lyric_terms and not exact_lyric_terms:
            dense_query = (
                f"{analysis.original_query}\n"
                f"들리는 가사의 가능한 표기: {' / '.join(phonetic_lyric_terms)}"
            )
            alpha = min(0.60, max(alpha, 0.50))

        return self._text_svc.search_text(
            dense_query,
            top_k=top_k,
            alpha=alpha,
            sparse_query=sparse_query,
            metadata_filter=metadata_filter,
        )

    def _search_title_presence(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> List[MatchingTrack]:
        """괄호/부제에 포함된 한자 표기를 찾는 비침습적 보조 경로."""
        metadata_filter = build_title_presence_metadata_filter(
            analysis.title_constraints
        )
        if not metadata_filter:
            return []

        exact_lyric_terms = _exact_lyric_terms(analysis)
        phonetic_lyric_terms = _phonetic_lyric_terms(analysis)
        sparse_terms = (
            exact_lyric_terms
            or phonetic_lyric_terms
            or list(analysis.korean_tags)
        )
        if analysis.song_title:
            sparse_terms.append(analysis.song_title)
        if analysis.artist_name:
            sparse_terms.append(analysis.artist_name)
        sparse_terms.extend(a for a in analysis.artist_name_alt if a)

        dense_query = analysis.original_query
        alpha = analysis.text_alpha
        if phonetic_lyric_terms and not exact_lyric_terms:
            dense_query = (
                f"{analysis.original_query}\n"
                f"들리는 가사의 가능한 표기: {' / '.join(phonetic_lyric_terms)}"
            )
            alpha = min(0.60, max(alpha, 0.50))

        logger.debug(
            "[SearchRouter/title-presence] filter=%s query='%s' top_k=%d",
            metadata_filter,
            analysis.original_query,
            top_k,
        )
        return self._text_svc.search_text(
            dense_query,
            top_k=top_k,
            alpha=alpha,
            sparse_query=" ".join(sparse_terms) if sparse_terms else None,
            metadata_filter=metadata_filter,
        )

    def _search_lyrics(
        self,
        analysis: QueryAnalysis,
        top_k: Optional[int],
        snapshot_out: Optional[List[Any]] = None,
    ) -> List[MatchingTrack]:
        """full_lyrics에서 표면 구절의 exact/fuzzy 후보를 찾는다.

        `snapshot_out`으로 **이 검색이 본 가사 판**을 호출부에 넘긴다. 인용은 그
        판에서만 떠야 하기 때문이다 — 나중에 다시 읽으면 다른 판을 인용한다.
        """
        if self._lyrics_svc is None:
            return []
        return self._lyrics_svc.search(
            analysis.lyric_clues, top_k=top_k, snapshot_out=snapshot_out
        )

    def _search_image(self, analysis: QueryAnalysis, top_k: int) -> List[MatchingTrack]:
        """
        SigLIP2 텍스트→이미지 공간 임베딩 후 Pinecone image 인덱스 쿼리.
        image_english_query만 사용 (SigLIP2는 영어 성능이 극대화됨).
        메타데이터를 함께 받아와(include_metadata=True) 이 경로에서만 잡힌 곡도
        "Unknown"이 아닌 실제 제목/태그로 리랭킹/응답에 사용할 수 있게 한다.
        """
        en_query = analysis.image_english_query.strip()
        if not en_query:
            logger.debug("[SearchRouter/image] empty dedicated prompt → skip")
            return []
        logger.debug("[SearchRouter/image] en_query='%s' top_k=%d", en_query, top_k)

        with timing.step("image.embed"):
            vec: List[float] = (
                self._img_emb.embed_texts([en_query], l2_normalize=True)[0].tolist()
            )
        with timing.step("image.query"):
            res = self._img_idx.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                namespace=NAMESPACE,
            )
        matches = res.get("matches", []) if hasattr(res, "get") else getattr(res, "matches", [])
        return [self._text_svc.track_from_match(m) for m in (matches or [])]

    def _search_audio(self, analysis: QueryAnalysis, top_k: int) -> List[MatchingTrack]:
        """
        CLAP 텍스트→오디오 공간 임베딩 후 Pinecone audio 인덱스 쿼리.
        audio_english_query만 사용 (CLAP은 영어 학습 기반 모델).
        메타데이터를 함께 받아와(include_metadata=True) 이 경로에서만 잡힌 곡도
        "Unknown"이 아닌 실제 제목/태그로 리랭킹/응답에 사용할 수 있게 한다.
        """
        en_query = analysis.audio_english_query.strip()
        if not en_query:
            logger.debug("[SearchRouter/audio] empty dedicated prompt → skip")
            return []
        logger.debug("[SearchRouter/audio] en_query='%s' top_k=%d", en_query, top_k)

        with timing.step("audio.embed"):
            vec: List[float] = (
                self._audio_emb.embed_texts([en_query], l2_normalize=True)[0].tolist()
            )
        with timing.step("audio.query"):
            res = self._audio_idx.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                namespace=NAMESPACE,
            )
        matches = res.get("matches", []) if hasattr(res, "get") else getattr(res, "matches", [])
        return [self._text_svc.track_from_match(m) for m in (matches or [])]
