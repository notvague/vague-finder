from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


# 현재 meta.json에서 반드시 값이 존재해야 하는 필드
CORE_REQUIRED_PATHS: tuple[str, ...] = (
    "metadata.title",
    "metadata.artist",
    "metadata.album",
    "metadata.release_date",
    "metadata.genre",
    "metadata.type",
    "metadata.vocal_gender",
    "lyrics_data.full_lyrics",
    "lyrics_data.lyrics_highlight",
    "lyrics_data.lyrics_summary",
    "semantic_analysis.search_style_summary",
    "semantic_analysis.mood_tags",
    "semantic_analysis.time_weather_tags",
    "semantic_analysis.place_activity_tags",
    "semantic_analysis.emotion_tags",
    "semantic_analysis.vibe_tags",
    "semantic_analysis.relation_context_tags",
    "semantic_analysis.color_tags",
    "semantic_analysis.sound_tags",
    "semantic_analysis.melon_playlist_tags",
    "semantic_analysis.visual_imagery",
    "community_feedback.sentiment_summary",
    "community_feedback.fans_tags",
    "community_feedback.major_emotion",
    "community_feedback.popularity.fame",
    "links.cover_url",
    "comments",
)


CORE_OBJECT_PATHS: tuple[str, ...] = (
    "metadata",
    "lyrics_data",
    "semantic_analysis",
    "community_feedback",
    "community_feedback.popularity",
    "links",
    "comments",
)


CORE_STRING_PATHS: tuple[str, ...] = (
    "metadata.title",
    "metadata.album",
    "metadata.release_date",
    "metadata.vocal_gender",

    "lyrics_data.full_lyrics",
    "lyrics_data.lyrics_highlight",
    "lyrics_data.lyrics_summary",

    "semantic_analysis.search_style_summary",

    "community_feedback.sentiment_summary",
    "community_feedback.major_emotion",
    "community_feedback.popularity.fame",

    "links.cover_url",
)


CORE_STRING_LIST_PATHS: tuple[str, ...] = (
    "metadata.artist",
    "metadata.genre",
    "metadata.type",

    "semantic_analysis.mood_tags",
    "semantic_analysis.time_weather_tags",
    "semantic_analysis.place_activity_tags",
    "semantic_analysis.emotion_tags",
    "semantic_analysis.vibe_tags",
    "semantic_analysis.relation_context_tags",
    "semantic_analysis.color_tags",
    "semantic_analysis.sound_tags",
    "semantic_analysis.melon_playlist_tags",
    "semantic_analysis.visual_imagery",

    "community_feedback.fans_tags",

    "comments.melon",
    "comments.youtube",
)


# 최신 크롤러에서 crawl_status가 존재하는 경우 검사할 하위 필드
CRAWL_STATUS_REQUIRED_PATHS: tuple[str, ...] = (
    "crawl_status.status",
    "crawl_status.warnings",
    "crawl_status.input_artist",
    "crawl_status.input_title",
    "crawl_status.crawled_at",
)


# 최신 크롤러에서 audio 블록이 존재하는 경우 검사할 하위 필드
AUDIO_EXTENSION_REQUIRED_PATHS: tuple[str, ...] = (
    "audio.source",
    "audio.preview_url",
    "audio.matched_artist",
    "audio.matched_title",
    "audio.match_score",
    "audio.is_full_track",
    "audio_status",
    "audio_path",
)


URL_PATHS = {
    "links.cover_url",
    "audio.preview_url",
}


COMMENT_PATHS = (
    "comments.melon",
    "comments.youtube",
)

# 사람이 쓴 원문이 들어오는 경로. 실패 문구 검사를 적용하지 않는다.
#
# 실패 문구 목록('댓글이없', '가사가없' 등)은 LLM이 "댓글이 없어 알 수 없습니다"처럼 답한 것을
# 잡으려고 만든 것이다. 공백을 지우고 부분 문자열로 비교하기 때문에, 흔한 한국어 댓글이 그대로
# 걸린다. 961곡 실측에서 5곡이 이 사유 하나로 탈락했다.
#   3534943 '...요즘은 이런 가사가없는 음악들뿐이라 그립다'
#   1569650 '이럴수가 내 댓글이 없었네...'
#   511798  '댓글이 없다니.. 아련하게 떠오르는 내 20대의 추억이...'
#   465286  '...가사가 없어서 종이에 가사 받아적었던거...'
#   1748336 '아니 이노래에 왜댓글이없지'
# 원문 댓글의 품질은 수집 단계(하드필터 + LLM 선별)가 담당한다. 빈 값·자료형 검사는 그대로 한다.
_USER_TEXT_PATH_PREFIXES = tuple(f"{path}[" for path in COMMENT_PATHS)

ALTERNATIVE_LINK_PATHS = (
    "links.melon_url",
    "links.youtube_url",
)

ALTERNATIVE_ALBUM_TEXT_PATHS = (
    "metadata.album_description",
    "metadata.album_summary",
)

# 댓글 합계 기준. **탈락 기준이 아니라 보고 기준이다.**
#
# 예전에는 합계 10개 미만이면 곡을 떨어뜨렸다. 이미 수집한 961곡 중 41곡이 이 규칙 하나로만
# 탈락했는데(다른 문제는 없음), 그 41곡의 sentiment_summary는 모두 실제 댓글에서 나온 구체적인
# 문장이었다(예: 멜론 2개+유튜브 4개). 댓글이 적은 것은 곡의 결함이 아니다.
#
# 크롤러도 2026-09-18부터 댓글이 꺼진 영상 때문에 정상 음원을 버리지 않는다. 두 단계가 어긋나면
# 수집은 완료인데 임베딩에서 빠지는 곡이 생기므로, 이 기준은 크롤러의 검토 기록에만 쓴다.
# 대신 아래에 '근거 없는 반응 요약'을 잡는 규칙을 둔다 — 댓글이 0개인데 반응 요약이 있으면
# 그 문장은 댓글에서 나올 수 없다.
LOW_COMMENT_COUNT = 10

# 댓글이 0개일 때 비어 있어도 되는 경로. 댓글에서 나오는 필드라서, 근거가 없으면 비우는 것이
# 맞다(크롤러 refine_data._blank_comment_fields_without_evidence가 비운다).
_COMMENT_DERIVED_PATHS = (
    "community_feedback.sentiment_summary",
    "community_feedback.fans_tags",
)


# 일반적인 빈값 검사 대신 전용 결합 규칙으로 검사할 필드
_ALLOW_EMPTY_SEQUENCE_PATHS = {
    "crawl_status.warnings",
    *COMMENT_PATHS,
}

_SKIP_GENERIC_VALUE_VALIDATION_PATHS = {
    # Background context is optional and must never decide whether the core song
    # is moved from raw to failed_raw.  Its own collector validates this block.
    "namuwiki",
    # Keep accepting the previous nested shape long enough to migrate it.
    "external_context",
    *ALTERNATIVE_LINK_PATHS,
    *ALTERNATIVE_ALBUM_TEXT_PATHS,
    "community_feedback.popularity.youtube_view_count",
}

# 빈 문자열을 정상 값으로 받는 경로. 이 검증기는 키 누락·null·문자열 아닌 값·실패 문구를
# 계속 문제로 본다.
_ALLOW_EMPTY_STRING_PATHS = {
    # 크롤러(refine_data._normalize_major_emotion)는 LLM 응답의 major_emotion을 통제 어휘로
    # 옮길 수 없으면 틀린 감정을 저장하는 대신 빈 문자열로 저장한다. 해당하는 경우: 뜻이 갈리는
    # 값인데 태그 근거가 없음, 대응 어휘가 없는 값(Obsession 등), 답 자체에 어휘 어간이 없는 값,
    # 그리고 성공 응답에서 키 누락·null·실패 문구. 크롤러는 2026-09-18부터 분석이 모두 실패한
    # 곡을 저장하지 않으므로(refine_data가 빈 사전을 돌려주고 main이 분석 단계를 재시도),
    # 새 수집분에서는 '분석실패' 문구가 나오지 않는다. 예전 수집분 검사를 위해 규칙은 남긴다.
    # 이 빈 값을 거부하면 부가 필드 하나 때문에 곡 전체가 임베딩에서 빠지고 폴더가 옮겨진다.
    # 다운스트림은 빈 값을 건너뛴다: passage_builder(감정 줄 생략), 리랭커 문서(빈 줄 생략).
    "community_feedback.major_emotion",
}


# 'unknown'을 정상 값으로 받는 경로.
#
# 크롤러 프롬프트는 이 두 필드에 한해 "If completely unknown, output 'unknown'"이라고 지시하고,
# refine_data의 정규화도 옮길 수 없는 값을 unknown으로 떨어뜨린다. 그런데 'unknown'이 실패
# 문구여서 그 곡이 통째로 탈락했다 — 수집은 done인데 임베딩에서 failed_raw로 옮겨지는,
# 댓글 수 규칙과 똑같은 어긋남이다.
# 다운스트림은 이 값을 조용히 건너뛴다: 재질문은 성별 선택지를 남성/여성/혼성으로만 만들고
# (retrieval/clarify.slot_value), 검색 가산점도 일치하는 값이 없으면 붙지 않는다.
_ALLOW_UNKNOWN_PATHS = {
    "metadata.vocal_gender",
    "metadata.type",
}


def _allows_unknown(path: str) -> bool:
    """metadata.type[0]처럼 배열 원소 경로도 허용 대상으로 본다."""
    return path in _ALLOW_UNKNOWN_PATHS or path.split("[")[0] in _ALLOW_UNKNOWN_PATHS


# 공백, 대시, 밑줄을 제거하고 소문자로 비교한다.
_EXACT_FAILURE_MARKERS = {
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "undefined",
    "unavailable",
    "미상",
    "알수없음",
    "정보없음",
    "가사없음",
    "분석실패",
    "apierror",
    "수집실패",
    "크롤링실패",
}


# LLM이 빈값 대신 문장으로 실패 사실을 생성한 경우
_FAILURE_PHRASE_MARKERS = (
    "api호출한도초과",
    "분석하지못했습니다",
    "정보가제공되지않",
    "제공된정보가없",
    "구체적인정보가없",
    "댓글이없",
    "댓글이제공되지않",
    "가사가없",
    "가사가제공되지않",
    "파악하기어렵",
    "찾기어렵",
    "확인하기어렵",
    "알수없습니다",
)


_SUCCESS_CRAWL_STATUSES = {
    "success",
    "succeeded",
    "complete",
    "completed",
    "ok",
    "full",
}


_SUCCESS_AUDIO_STATUSES = {
    "available",
    "success",
    "succeeded",
    "complete",
    "completed",
    "downloaded",
    "ok",
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    reason: str
    value_preview: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "reason": self.reason,
            "value_preview": self.value_preview,
        }


def _compact_text(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).casefold()


def _preview(value: Any, limit: int = 120) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _lookup_path(
    obj: Mapping[str, Any],
    path: str,
) -> tuple[bool, Any]:
    current: Any = obj

    for part in path.split("."):
        if not isinstance(current, Mapping):
            return False, None

        if part not in current:
            return False, None

        current = current[part]

    return True, current


def _is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )


def has_failure_marker(value: str) -> bool:
    """LLM이 "정보가 없어 답할 수 없다"고 적은 문장인가.

    크롤러(refine_data)도 같은 판정을 쓴다. 규칙이 두 곳에서 갈라지면 "수집할 때는 통과했는데
    임베딩 검증에서는 걸린다"가 생기므로 정의는 여기 하나만 둔다.
    """
    return _has_failure_marker(value)


def _has_failure_marker(value: str) -> bool:
    compact = _compact_text(value)

    if compact in _EXACT_FAILURE_MARKERS:
        return True

    return any(
        marker in compact
        for marker in _FAILURE_PHRASE_MARKERS
    )


def _append_issue(
    issues: list[ValidationIssue],
    *,
    path: str,
    reason: str,
    value: Any = "",
) -> None:
    issue = ValidationIssue(
        path=path or "<root>",
        reason=reason,
        value_preview=_preview(value),
    )

    if issue not in issues:
        issues.append(issue)


def _validate_present_value(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    allow_empty: frozenset[str] = frozenset(),
) -> None:
    """
    존재하는 값의 내부를 배열 원소까지 재귀적으로 검사한다.

    allow_empty: 이 문서에서만 빈 값을 허용할 경로. 문서 내용에 따라 달라지는 규칙에 쓴다
        (댓글이 0개인 곡의 반응 요약). 문서와 무관하게 늘 허용하는 경로는
        _ALLOW_EMPTY_STRING_PATHS / _ALLOW_EMPTY_SEQUENCE_PATHS에 둔다.
    """
    if path in _SKIP_GENERIC_VALUE_VALIDATION_PATHS:
        return
    
    if value is None:
        _append_issue(
            issues,
            path=path,
            reason="null 값",
            value=value,
        )
        return

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            if path not in _ALLOW_EMPTY_STRING_PATHS and path not in allow_empty:
                _append_issue(
                    issues,
                    path=path,
                    reason="빈 문자열",
                    value=value,
                )

        elif path.startswith(_USER_TEXT_PATH_PREFIXES):
            # 사람이 쓴 댓글 본문. 실패 문구 검사를 하지 않는다(_USER_TEXT_PATH_PREFIXES 주석).
            pass

        elif _compact_text(stripped) == "unknown" and _allows_unknown(path):
            # 이 필드에서는 'unknown'이 실패가 아니라 "모른다"는 정상 값이다.
            pass

        elif _has_failure_marker(stripped):
            _append_issue(
                issues,
                path=path,
                reason="누락/실패 대체 문구",
                value=value,
            )

        elif path in URL_PATHS and not _is_valid_http_url(stripped):
            _append_issue(
                issues,
                path=path,
                reason="유효한 HTTP(S) URL이 아님",
                value=value,
            )

        return

    # False는 정상적인 정보일 수 있다.
    # 예: audio.is_full_track == false
    if isinstance(value, bool):
        return

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            _append_issue(
                issues,
                path=path,
                reason="유한한 숫자가 아님",
                value=value,
            )
        return

    if isinstance(value, Mapping):
        if not value:
            _append_issue(
                issues,
                path=path,
                reason="빈 객체",
                value=value,
            )
            return

        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _validate_present_value(
                child,
                child_path,
                issues,
                allow_empty,
            )

        return

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        # 성공한 crawl_status는 warnings가 빈 배열이어야 정상이다.
        if not value and path not in _ALLOW_EMPTY_SEQUENCE_PATHS and path not in allow_empty:
            _append_issue(
                issues,
                path=path,
                reason="빈 배열",
                value=value,
            )
            return

        for index, child in enumerate(value):
            _validate_present_value(
                child,
                f"{path}[{index}]",
                issues,
                allow_empty,
            )

        return

    _append_issue(
        issues,
        path=path,
        reason=f"지원하지 않는 값 자료형({type(value).__name__})",
        value=value,
    )


def count_comments(meta: Mapping[str, Any]) -> tuple[dict[str, int], int]:
    """멜론·유튜브 댓글 수와 합계. 크롤러와 검증기가 같은 셈을 쓰도록 공개한다."""
    counts: dict[str, int] = {}
    total = 0

    for path in COMMENT_PATHS:
        exists, value = _lookup_path(meta, path)
        count = len(value) if exists and isinstance(value, list) else 0
        counts[path] = count
        total += count

    return counts, total


def _document_allow_empty(meta: Mapping[str, Any]) -> frozenset[str]:
    """이 문서에서만 빈 값을 허용할 경로를 정한다.

    댓글이 하나도 없으면 반응 요약과 팬 태그는 비어 있어야 정상이다. 이걸 허용하지 않으면
    "댓글이 없어 알 수 없습니다" 같은 정직한 답은 실패 문구로 걸리고, 가사에서 지어낸 답만
    통과한다. 정직한 쪽이 통과해야 한다.
    """
    _, total = count_comments(meta)

    return frozenset(_COMMENT_DERIVED_PATHS) if total == 0 else frozenset()


def _require_paths(
    meta: Mapping[str, Any],
    paths: Iterable[str],
    issues: list[ValidationIssue],
) -> None:
    for path in paths:
        exists, _ = _lookup_path(meta, path)

        if not exists:
            _append_issue(
                issues,
                path=path,
                reason="필수 필드 누락",
            )


def _validate_expected_types(
    meta: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    for path in CORE_OBJECT_PATHS:
        exists, value = _lookup_path(meta, path)

        if exists and not isinstance(value, Mapping):
            _append_issue(
                issues,
                path=path,
                reason="객체(dict) 자료형이 아님",
                value=value,
            )

    for path in CORE_STRING_PATHS:
        exists, value = _lookup_path(meta, path)

        if exists and not isinstance(value, str):
            _append_issue(
                issues,
                path=path,
                reason="문자열 자료형이 아님",
                value=value,
            )

    for path in CORE_STRING_LIST_PATHS:
        exists, value = _lookup_path(meta, path)

        if not exists:
            continue

        if not isinstance(value, list):
            _append_issue(
                issues,
                path=path,
                reason="배열(list) 자료형이 아님",
                value=value,
            )
            continue

        for index, item in enumerate(value):
            if not isinstance(item, str):
                _append_issue(
                    issues,
                    path=f"{path}[{index}]",
                    reason="배열 원소가 문자열이 아님",
                    value=item,
                )


def _validate_special_rules(
    meta: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> None:
    _validate_expected_types(meta, issues)

    song_id = meta.get("song_id") or meta.get("id")

    if song_id is None:
        _append_issue(
            issues,
            path="song_id",
            reason="song_id/id 필드 누락",
        )

    else:
        _validate_present_value(
            song_id,
            "song_id",
            issues,
        )

        if not str(song_id).strip().isdigit():
            _append_issue(
                issues,
                path="song_id",
                reason="Melon 곡 ID가 숫자가 아님",
                value=song_id,
            )

    # 앨범 텍스트는 **탈락 기준이 아니다.**
    #
    # album_description은 멜론 앨범 페이지에서 오는데, 2000년대 이전 앨범은 소개가 아예 없는
    # 경우가 흔하다(코퍼스 952곡 중 81곡). 그중 79곡은 LLM이 가사·메타데이터로 쓸 만한 요약을
    # 썼고, 2곡은 "정보가 제공되지 않아 확인할 수 없습니다"라고 정직하게 답해서 실패 문구 규칙에
    # 걸려 곡이 통째로 빠졌다. 곡 수가 늘면 옛 곡 비중이 커져 이 손실도 함께 커진다.
    #
    # 앨범 설명이 빠지는 것이 곡이 빠지는 것보다 낫다. 그래서 여기서는 검사하지 않고,
    # 실패 문구는 크롤러가 비운다(refine_data.blank_album_text_without_evidence).
    # 두 경로는 _SKIP_GENERIC_VALUE_VALIDATION_PATHS에 있어 빈 값도 정상으로 통과한다.
    # 수집 시점의 누락은 main.note_validation_gap과 review_songs.csv로 드러난다.
    # 댓글 수 자체로는 떨어뜨리지 않는다(LOW_COMMENT_COUNT 주석 참고).
    # 대신 댓글이 하나도 없는데 반응 요약이 채워져 있으면 그 문장은 댓글에서 나올 수 없다.
    comment_counts, total_comment_count = count_comments(meta)

    if total_comment_count == 0:
        fabricated: dict[str, Any] = {}

        for path in _COMMENT_DERIVED_PATHS:
            exists, value = _lookup_path(meta, path)

            if not exists:
                continue

            if isinstance(value, str) and value.strip():
                fabricated[path] = value
            elif isinstance(value, list) and value:
                fabricated[path] = value

        if fabricated:
            _append_issue(
                issues,
                path="community_feedback.sentiment_summary|community_feedback.fans_tags",
                reason="댓글이 0개인데 반응 요약/팬 태그가 채워져 있음",
                value={**fabricated, "comments": comment_counts},
            )

    # cover_url은 별도로 필수 검사한다.
    # melon_url과 youtube_url은 둘 중 하나만 유효해도 통과한다.
    valid_source_links: list[str] = []
    source_link_values: dict[str, Any] = {}

    for path in ALTERNATIVE_LINK_PATHS:
        exists, value = _lookup_path(meta, path)
        source_link_values[path] = value if exists else None

        if isinstance(value, str) and _is_valid_http_url(value):
            valid_source_links.append(path)

    if not valid_source_links:
        _append_issue(
            issues,
            path="links.melon_url|links.youtube_url",
            reason=(
                "Melon/YouTube URL 중 유효한 "
                "HTTP(S) URL이 하나도 없음"
            ),
            value=source_link_values,
        )

    # crawl_status 블록이 있는 최신 meta.json 검사
    if "crawl_status" in meta:
        _require_paths(
            meta,
            CRAWL_STATUS_REQUIRED_PATHS,
            issues,
        )

        exists, crawl_status = _lookup_path(
            meta,
            "crawl_status",
        )

        if exists and not isinstance(crawl_status, Mapping):
            _append_issue(
                issues,
                path="crawl_status",
                reason="객체(dict) 자료형이 아님",
                value=crawl_status,
            )

        for path in (
            "crawl_status.status",
            "crawl_status.input_artist",
            "crawl_status.input_title",
            "crawl_status.crawled_at",
        ):
            exists, value = _lookup_path(meta, path)

            if exists and not isinstance(value, str):
                _append_issue(
                    issues,
                    path=path,
                    reason="문자열 자료형이 아님",
                    value=value,
                )

        exists, status = _lookup_path(
            meta,
            "crawl_status.status",
        )

        if (
            exists
            and _compact_text(str(status))
            not in _SUCCESS_CRAWL_STATUSES
        ):
            _append_issue(
                issues,
                path="crawl_status.status",
                reason="완료 상태가 아님",
                value=status,
            )

        exists, warnings = _lookup_path(
            meta,
            "crawl_status.warnings",
        )

        if exists and (
            not isinstance(warnings, list)
            or len(warnings) > 0
        ):
            _append_issue(
                issues,
                path="crawl_status.warnings",
                reason="크롤링 경고가 존재함",
                value=warnings,
            )

    audio_extension_present = any(
        key in meta
        for key in (
            "audio",
            "audio_status",
            "audio_path",
        )
    )

    # audio 확장 블록이 있는 최신 meta.json 검사
    if audio_extension_present:
        _require_paths(
            meta,
            AUDIO_EXTENSION_REQUIRED_PATHS,
            issues,
        )

        exists, audio_block = _lookup_path(
            meta,
            "audio",
        )

        if exists and not isinstance(audio_block, Mapping):
            _append_issue(
                issues,
                path="audio",
                reason="객체(dict) 자료형이 아님",
                value=audio_block,
            )

        for path in (
            "audio.source",
            "audio.preview_url",
            "audio.matched_artist",
            "audio.matched_title",
            "audio_status",
            "audio_path",
        ):
            exists, value = _lookup_path(meta, path)

            if exists and not isinstance(value, str):
                _append_issue(
                    issues,
                    path=path,
                    reason="문자열 자료형이 아님",
                    value=value,
                )

        exists, is_full_track = _lookup_path(
            meta,
            "audio.is_full_track",
        )

        if exists and not isinstance(is_full_track, bool):
            _append_issue(
                issues,
                path="audio.is_full_track",
                reason="불리언(bool) 자료형이 아님",
                value=is_full_track,
            )

        exists, audio_status = _lookup_path(
            meta,
            "audio_status",
        )

        if (
            exists
            and _compact_text(str(audio_status))
            not in _SUCCESS_AUDIO_STATUSES
        ):
            _append_issue(
                issues,
                path="audio_status",
                reason="오디오 사용 가능 상태가 아님",
                value=audio_status,
            )

        exists, match_score = _lookup_path(
            meta,
            "audio.match_score",
        )

        if exists and (
            isinstance(match_score, bool)
            or not isinstance(match_score, (int, float))
            or not math.isfinite(float(match_score))
            or match_score <= 0
        ):
            _append_issue(
                issues,
                path="audio.match_score",
                reason="오디오 매칭 점수가 0보다 큰 숫자가 아님",
                value=match_score,
            )


def _validate_media_files(
    song_dir: Path,
    issues: list[ValidationIssue],
) -> None:
    """
    meta.json 외의 실제 필수 미디어 파일을 검사한다.

    meta.json 자체는 data_songs.py에서 곡 폴더를 순회하면서 먼저 검사한다.
    """
    required_files = (
        ("cover.jpg", "<file>.cover.jpg"),
        ("audio.m4a", "<file>.audio.m4a"),
    )

    for filename, path_name in required_files:
        media_path = song_dir / filename

        if not media_path.is_file():
            _append_issue(
                issues,
                path=path_name,
                reason="필수 미디어 파일 없음",
                value=str(media_path),
            )
            continue

        try:
            file_size = media_path.stat().st_size

        except OSError as exc:
            _append_issue(
                issues,
                path=path_name,
                reason=f"파일 상태 확인 실패: {exc}",
                value=str(media_path),
            )
            continue

        if file_size <= 0:
            _append_issue(
                issues,
                path=path_name,
                reason="크기가 0인 미디어 파일",
                value=str(media_path),
            )


def validate_meta_document(
    meta: Any,
    *,
    song_dir: Path | None = None,
    require_media_files: bool = True,
) -> list[ValidationIssue]:
    """
    한 곡의 meta.json과 필수 미디어 파일을 검사한다.

    반환값이 빈 배열이면 임베딩 가능하다.
    ValidationIssue가 하나라도 있으면 임베딩에서 제외한다.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(meta, Mapping):
        _append_issue(
            issues,
            path="<root>",
            reason="JSON 최상위 값이 객체가 아님",
            value=meta,
        )
        return issues

    _require_paths(
        meta,
        CORE_REQUIRED_PATHS,
        issues,
    )

    _validate_present_value(
        meta,
        "",
        issues,
        _document_allow_empty(meta),
    )

    _validate_special_rules(
        meta,
        issues,
    )

    if require_media_files:
        if song_dir is None:
            _append_issue(
                issues,
                path="<folder>",
                reason="곡 폴더 경로가 없어 미디어 파일을 검사할 수 없음",
            )
        else:
            _validate_media_files(
                Path(song_dir),
                issues,
            )

    return issues
