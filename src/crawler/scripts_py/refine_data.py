"""
[Step 3/4] LLM Refinement Module
설명: Gemini 모델을 사용하여 수집된 메타데이터, 가사, 댓글을 분석합니다.
      노래의 분위기(Tags)를 추출하고, 상황에 맞는 요약(Scene Summary)을 생성하여
      최종 메타데이터(JSON)를 완성합니다.

      나무위키 배경 정보는 이 음악-vibe LLM 입력에 넣지 않습니다. 별도 context
      artifact와 검색 인덱스에서만 처리하여 기본 음악 표현을 오염시키지 않습니다.

작성자: 이연우 (Data Engineer), 황찬혁 (Full)
생성일: 2026-01-29
수정일: 2026-08-23 (Namuwiki external context 지원)

NOTE:
- `build_comment_selection_prompt` / `select_emotional_comments_with_llm`은
  collector(`collect_melon_data`, `collect_reaction`)와 함께 사용되는 공용
  유틸이므로 순환 import 위험을 없애기 위해 `llm_utils` 모듈로 이전됨.
  본 모듈은 더 이상 해당 함수를 정의/재노출하지 않는다.
- 구형 `namuwiki_data` 입력 경로는 제거했다. context 검색은
  `src.embedding.context_*` 파이프라인의 독립 채널이다.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from src.common.emotion_vocab import canonical_emotion, vocab_prompt_line
from src.embedding.fixtures.meta_validation import has_failure_marker

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# .env 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY가 설정되지 않았습니다.")


MAX_LYRICS_CHARS = 1000
MAX_ALBUM_DESC_CHARS = 1000


def _safe_text(value: Any) -> str:
    """None/NaN성 값까지 안전하게 문자열로 변환한다."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def generate_prompt(
    metadata: Dict,
    lyrics: Optional[str],
    melon_comments: List[str],
    youtube_comments: List[str],
) -> str:
    """Gemini에 보낼 단일 통합 프롬프트를 생성한다."""
    track = metadata.get("title", metadata.get("trackName", "Unknown"))
    artist = metadata.get("artist", metadata.get("artistName", "Unknown"))

    album_desc = metadata.get("album_desc", "")

    # 텍스트 길이 제한
    lyrics_snippet = lyrics[:MAX_LYRICS_CHARS] if lyrics else "가사 없음"
    album_desc_snippet = album_desc[:MAX_ALBUM_DESC_CHARS] if album_desc else "정보 없음"

    # 댓글 블록 생성 (Top 10개만 사용)
    m_block = "\n".join([f"- {c}" for c in melon_comments[:10]])
    y_block = "\n".join([f"- {c}" for c in youtube_comments[:10]])

    prompt = f"""
role: You are a "Music Vibe Analyst".
Your goal is to analyze the song and generate rich metadata for a Korean music search engine.

INPUT DATA:
1. Song: {artist} - {track}
2. Playlist/Album Intro:
{album_desc_snippet}
...
3. Lyrics:
{lyrics_snippet}
...
4. User Comments:
   [Melon Reviews]:
{m_block}
   [YouTube Reactions]:
{y_block}

TASK:
Analyze the input and output a JSON object exactly matching this schema (ALL values in Korean):

CRITICAL RULE FOR ALL TAG LISTS:
1. NO SPACE NOUNS: Every single item in the EXISTING MUSIC-VIBE tag arrays MUST be a concise NOUN-based keyword WITHOUT ANY SPACES (띄어쓰기 절대 금지) and WITHOUT verb/adjective descriptive endings (동사형/형용사형 서술 어미 절대 금지).
BAD: "듣기 좋은", "비 오는 날", "위로를 주는", "신나는 비트", "청아한 보컬"
GOOD: "힐링", "비", "드라이브", "새벽감성", "신나는", "위로", "청아함", "어쿠스틱"

2. EVIDENCE-BASED EXTRACTION: For existing music-vibe fields, rely ONLY on the provided lyrics, album desc, and user comments. If there is absolutely no evidence or context to infer specific tags, DO NOT GUESS. Simply output "unknown" as the tag for that slot.
(EXCEPTION: For the "visual_imagery" field ONLY, you are strongly encouraged to use creative freedom to imagine 3 specific visual scenes that perfectly match the song's vibe).

3. NO NUMERICS/METRICS FOR VIBE TAGS: Never create vibe tags such as "좋아요10만", "1억뷰", "1위". Focus those tag fields on emotion, atmosphere, genre, and theme.

4. STRICT FACTUAL ACCURACY IN ALBUM SUMMARY: You must strictly base the album summary ONLY on the provided Album Intro. Focus heavily on describing the properties of THIS SPECIFIC SONG if mentioned in the intro. NEVER hallucinate or infer facts that are not explicitly written. For example, NEVER label a song as a "title track" (타이틀 곡) unless the Album Intro explicitly states it is the title track.

{{
  "album_summary": "A 2-sentence description summarizing the provided Album Intro, focusing heavily on THIS SPECIFIC SONG. Must be 100% factually accurate. Do not fabricate details like 'title track' without explicit textual evidence.",
  "artist_type": ["List", "of", "1-2", "noun artist types (e.g., 솔로, 그룹, 밴드, 듀오, 혼성). If completely unknown, output ['unknown']"],
  "vocal_gender": "A single word for vocal gender (e.g., 남성, 여성, 혼성). If completely unknown, output 'unknown'",
  "lyrics_highlight": "The most memorable 1-2 lines from the lyrics.",
  "lyrics_summary": "A 2-sentence summary of the lyrics' core emotion.",
  "search_style_summary": "A 2-sentence rich situational description of the song.",
  "mood_tags": ["List", "of", "5", "noun keywords for overall mood"],
  "time_weather_tags": ["List", "of", "4", "noun time/weather/season (e.g., 새벽, 밤, 비, 겨울)"],
  "place_activity_tags": ["List", "of", "4", "noun places/activities (e.g., 드라이브, 창가, 산책, 퇴근길)"],
  "emotion_tags": ["List", "of", "3", "noun emotions (e.g., 먹먹함, 쓸쓸함, 그리움)"],
  "vibe_tags": ["List", "of", "3", "noun vibe descriptors (e.g., 몽환적, 잔잔한, 서늘한)"],
  "relation_context_tags": ["List", "of", "3", "noun relation contexts (e.g., 이별, 짝사랑, 재회)"],
  "color_tags": ["List", "of", "3", "noun colors (e.g., 파랑, 회색, 보라)"],
  "sound_tags": ["List", "of", "3", "noun sound/vocal descriptors (e.g., 피아노, 호소력)"],
  "visual_imagery": ["List", "of", "3", "IMAGINED specific visual scene fragments inspired by the song (creative freedom allowed, e.g., 눈 덮인 겨울밤, 흐린 대낮의 카페)"],
  "sentiment_summary": "A 2-sentence summary of how users feel based on comments.",
  "fans_tags": ["List", "of", "4", "noun keywords frequently mentioned by fans (e.g., 눈물버튼, 인생곡, 위로)"],
  "major_emotion": "The dominant emotion. Pick EXACTLY ONE from this fixed Korean list, copied verbatim: __EMOTION_VOCAB__"
}}

OUTPUT FORMAT: ONLY return valid JSON. Do not include Markdown blocks.
"""
    return prompt.replace("__EMOTION_VOCAB__", vocab_prompt_line())


# --- LLM 출력 스키마 -------------------------------------------------------------------
# JSON 객체인지만 보고 저장하면 태그 배열이 문자열로 와도 통과한다. main의 no_space()는
# 문자열을 글자 단위로 쪼개 ["새","벽","감","성"] 같은 태그를 만든다.

STRING_FIELDS: tuple = (
    "album_summary", "lyrics_highlight", "lyrics_summary", "search_style_summary",
    "sentiment_summary",
)
LIST_FIELDS: tuple = (
    "artist_type", "mood_tags", "time_weather_tags", "place_activity_tags", "emotion_tags",
    "vibe_tags", "relation_context_tags", "color_tags", "sound_tags", "visual_imagery",
    "fans_tags",
)
# 다른 헬퍼가 채우는 필드. 없거나 null이어도 응답을 버리지 않는다.
_OPTIONAL_STRING_FIELDS: tuple = ("major_emotion", "fact_summary")
_OPTIONAL_LIST_FIELDS: tuple = ("context_tags",)

VOCAL_GENDERS: frozenset = frozenset({"남성", "여성", "혼성", "unknown"})

# 성별 판정은 **부분 문자열로 하면 안 된다.** 'female'에는 'male'이, 'woman'에는 'man'이
# 들어 있어서 남성 표현을 먼저 검사하면 여성 보컬이 남성으로 저장된다.
# 영어는 단어 단위로 비교하고, 한국어는 겹치지 않는 표기만 쓴다.
_MALE_KO = ("남성", "남자")
_FEMALE_KO = ("여성", "여자")
# 성별을 한쪽으로 정하지 못하는 표현. 남녀는 한 단어에 둘이 다 들어 있다.
_MIXED_KO = ("혼성", "남녀")
_MALE_EN = frozenset({"male", "man", "men", "boy", "boys", "guy", "guys"})
_FEMALE_EN = frozenset({"female", "woman", "women", "girl", "girls", "lady", "ladies"})
# 'co'는 넣지 않는다. 두 글자 토큰이라 'co-vocal'·'Co.'처럼 성별과 무관한 말에서 잘려 나와
# 명시된 성별을 뒤집는다('female co-vocal' -> 혼성).
_MIXED_EN = frozenset({"mixed", "coed", "duet"})   # duet 단독은 성별 근거가 아니다(아래 주석)
_UNKNOWN_WORDS = ("unknown", "알수없음", "알 수 없음", "미상", "정보없음")

ARTIST_TYPES: frozenset = frozenset({"솔로", "그룹", "듀오", "밴드", "혼성", "unknown"})
# 코퍼스 961곡에서 나온 표기(걸그룹, 힙합듀오, 싱어송라이터, 남성중창단 등)를 여섯 값으로 모은다.
#
# 편성(몇 명인지)을 말하는 표기. 앞에 있는 항목이 이긴다 — '혼성그룹'은 그룹으로 모으고
# 혼성 여부는 vocal_gender가 들고 있다. '듀엣'은 검색 단계의 별칭 표(retrieval/clarify.py
# ARTIST_TYPE_ALIASES)가 이미 듀오로 보고 있어 같이 맞춘다.
_ENSEMBLE_SYNONYMS = (
    (("듀오", "듀엣"), ("duo",), "듀오"),
    (("밴드",), ("band",), "밴드"),
    (("그룹", "유닛", "중창단", "프로젝트", "콜라보"), ("group", "unit", "trio", "quartet"), "그룹"),
    (("솔로",), ("solo",), "솔로"),
    (("혼성",), ("mixed", "coed"), "혼성"),
)

# 역할·직함. 편성을 말하지 않으므로 **같은 목록에 편성 표기가 없을 때만** 솔로의 근거로 쓴다.
# 예전에는 항목마다 따로 옮겨서 ['듀오','보컬리스트']가 ['듀오','솔로']가 됐다. 그 값은
# metadata.type을 읽는 재질문(clarify.answer_matches)과 검색 가산점에서, 듀오를 솔로로
# 답한 사용자에게 이 곡을 맞춰 준다.
_ROLE_SYNONYMS = (
    (("싱어송라이터", "래퍼", "랩퍼", "프로듀서", "보컬리스트", "가수", "록커"),
     ("rapper", "producer", "vocalist", "singer", "songwriter"), "솔로"),
)


def _coerce_string(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"{field}: null")
    if isinstance(value, list):
        # 가사 하이라이트를 줄 목록으로 주는 경우가 있다. 이어 붙인다.
        parts = [_safe_text(v) for v in value if isinstance(v, (str, int, float))]
        if not parts:
            raise ValueError(f"{field}: 문자열이 아닌 배열")
        return " / ".join(p for p in parts if p)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return _safe_text(value)
    raise ValueError(f"{field}: 문자열이 아님 ({type(value).__name__})")


def _coerce_string_list(value: Any, field: str) -> List[str]:
    if isinstance(value, str):
        # 문자열 하나는 배열이 아니다. no_space()가 글자 단위로 분해하는 경로다.
        raise ValueError(f"{field}: 배열이 아닌 문자열")
    if not isinstance(value, list):
        raise ValueError(f"{field}: 배열이 아님 ({type(value).__name__})")
    items: List[str] = []
    for item in value:
        if not isinstance(item, str):
            # 태그 자리의 숫자·null은 정보가 아니다. 문자열로 바꿔 넣지 않는다.
            logger.warning("%s: 문자열이 아닌 항목 제거 %r", field, item)
            continue
        text = _safe_text(item)
        if text and text not in items:
            items.append(text)
    return items


def _words(text: str) -> set:
    """영문 단어 집합. 'female vocal' -> {female, vocal}. 부분 문자열 오판을 막는다."""
    return set(re.findall(r"[a-z]+", text.lower()))


def normalize_vocal_gender(value: Any) -> str:
    """남성/여성/혼성/unknown 네 값으로 맞춘다. 모르는 값은 unknown.

    성별과 편성(듀엣·듀오)은 별개다. '남성 듀엣'은 남성이고 '여성 듀엣'은 여성이다.
    예전 구현은 듀엣을 혼성 표현으로 묶어서 둘 다 혼성으로 저장했다.
    한쪽 성별 근거 없이 'duet'만 있으면 판단 근거가 아니므로 unknown이다.
    """
    if isinstance(value, list):
        # 목록으로 오면 항목을 모두 본다. ['남성','여성'] 같은 답이 혼성이 되어야 한다.
        value = " ".join(_safe_text(v) for v in value if isinstance(v, (str, int, float)))
    text = _safe_text(value)
    if not text:
        return "unknown"

    lowered = text.lower()
    words = _words(lowered)
    male = any(k in lowered for k in _MALE_KO) or bool(words & _MALE_EN)
    female = any(k in lowered for k in _FEMALE_KO) or bool(words & _FEMALE_EN)
    mixed = any(k in lowered for k in _MIXED_KO) or bool(words & (_MIXED_EN - {"duet"}))

    if mixed or (male and female):
        result = "혼성"
    elif male:
        result = "남성"
    elif female:
        result = "여성"
    else:
        result = "unknown"
        if not any(k in lowered for k in _UNKNOWN_WORDS) and "duet" not in words and "듀엣" not in lowered:
            logger.warning("vocal_gender: 허용되지 않은 값 %r -> unknown", text)
    assert result in VOCAL_GENDERS
    return result


def _map_with(table, text: str) -> Optional[str]:
    """표기 하나를 표에서 찾는다. 한국어는 부분 문자열, 영어는 단어 단위."""
    lowered = text.lower()
    words = _words(lowered)
    for ko_keys, en_keys, canonical in table:
        if any(key in lowered for key in ko_keys) or bool(words & set(en_keys)):
            return canonical
    return None


def normalize_artist_type(value: Any) -> List[str]:
    """솔로/그룹/듀오/밴드/혼성/unknown으로 맞춘다. 비면 ['unknown'].

    편성 표기를 먼저 모으고, 하나도 없을 때만 역할 표기를 솔로의 근거로 쓴다. 그래서
    ['듀오','보컬리스트']는 ['듀오']이고 ['싱어송라이터']는 ['솔로']다. 둘을 섞어 넣으면
    한 곡이 듀오이면서 솔로라고 주장하게 된다.
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ["unknown"]

    ensemble: List[str] = []
    roles: List[str] = []
    for item in value:
        text = _safe_text(item) if isinstance(item, (str, int, float)) else ""
        if not text:
            continue
        mapped = _map_with(_ENSEMBLE_SYNONYMS, text)
        if mapped:
            if mapped not in ensemble:
                ensemble.append(mapped)
            continue
        mapped = _map_with(_ROLE_SYNONYMS, text)
        if mapped:
            if mapped not in roles:
                roles.append(mapped)
            continue
        if not any(key in text.lower() for key in _UNKNOWN_WORDS):
            logger.warning("artist_type: 허용되지 않은 값 %r", text)

    result = ensemble or roles
    assert all(item in ARTIST_TYPES for item in result)
    return result or ["unknown"]


def validate_llm_schema(result_json: Dict) -> Dict:
    """LLM 응답의 타입과 허용값을 검사한다.

    구조가 틀리면(필드 누락, 배열 자리에 문자열, null) ValueError를 올려 호출부가
    다시 요청하게 한다. 값의 표기 차이(남자 -> 남성, 걸그룹 -> 그룹)는 고쳐 준다.
    """
    if not isinstance(result_json, dict):
        raise ValueError("Gemini 응답의 최상위 값이 JSON object가 아닙니다.")

    missing = [f for f in STRING_FIELDS + LIST_FIELDS + ("artist_type", "vocal_gender")
               if f not in result_json]
    if missing:
        raise ValueError(f"필드 누락: {', '.join(sorted(set(missing)))}")

    out = dict(result_json)
    for field in STRING_FIELDS:
        out[field] = _coerce_string(out[field], field)
    for field in _OPTIONAL_STRING_FIELDS:
        value = out.get(field)
        out[field] = "" if value is None else (
            _coerce_string(value, field) if not isinstance(value, str) else value
        )
    for field in LIST_FIELDS:
        if field == "artist_type":
            continue
        out[field] = _coerce_string_list(out[field], field)
    for field in _OPTIONAL_LIST_FIELDS:
        value = out.get(field)
        out[field] = [] if value is None else _coerce_string_list(value, field)

    if isinstance(out["artist_type"], str):
        raise ValueError("artist_type: 배열이 아닌 문자열")
    out["artist_type"] = normalize_artist_type(out["artist_type"])
    out["vocal_gender"] = normalize_vocal_gender(out["vocal_gender"])
    return out


def _normalize_major_emotion(result_json: Dict) -> Dict:
    """major_emotion을 한국어 통제 어휘로 맞춘다.

    프롬프트로 목록 중 하나를 고르게 해도 LLM이 목록 밖 값('Excitement', '행복' 등)을 돌려줄 수
    있다. 후보가 여럿인 값은 **같은 응답의** emotion_tags·mood_tags를 근거로 가린다.

    옮길 수 없으면 빈 문자열로 저장한다. 해당하는 경우: 뜻이 갈리는 값인데 태그 근거가 없음,
    대응 어휘가 없는 값(Obsession 등), 답 자체에 어휘 어간이 없는 값, 그리고 키 누락·null·
    실패 문구. 키는 항상 쓴다. 틀린 감정을 저장하느니 빈 값이 낫고, 빈 값은 임베딩 전 메타
    검증에서 허용된다(meta_validation._ALLOW_EMPTY_STRING_PATHS).
    """
    raw = result_json.get("major_emotion")
    tags = [
        tag
        for field in ("emotion_tags", "mood_tags")
        for tag in (result_json.get(field) or [])
        if isinstance(tag, str)
    ]
    canonical = canonical_emotion(raw, tags)
    if raw and canonical != raw:
        if canonical:
            logger.info("major_emotion 정규화: %r -> %r", raw, canonical)
        else:
            logger.warning("major_emotion을 통제 어휘로 옮길 근거가 없어 비움: %r", raw)
    result_json["major_emotion"] = canonical
    return result_json


# **LLM이 쓴 앨범 텍스트만** 실패 문구 검사 대상이다.
#
# album_description은 멜론 앨범 페이지에서 긁어온 사람이 쓴 글이라 LLM 실패 문구일 수가 없다.
# 그런데도 검사에 넣으면 긴 평론이 오검출된다 — 나얼 '같은 시간 속의 너'의 1,318자 평론에
# "그의 가창에는 이견을 찾기 어렵다"(칭찬)가 있어서 '찾기어렵' 문구에 걸렸다. 실패 문구 목록은
# 짧은 LLM 답변("정보를 찾기 어렵습니다")을 잡으려고 만든 것이라 긴 산문에 그대로 쓰면 안 된다.
_ALBUM_TEXT_KEYS = ("album_summary",)


def blank_album_text_without_evidence(result_json: Dict) -> Dict:
    """앨범 텍스트가 "정보가 제공되지 않아..." 같은 실패 문구면 그 자리를 비운다.

    멜론에 앨범 소개가 없는 곡이 코퍼스 952곡 중 81곡이다. 그중 79곡은 LLM이 가사·메타데이터로
    쓸 만한 요약을 썼고, 2곡은 "정보가 제공되지 않아 확인할 수 없습니다"라고 정직하게 답했다.
    그 문장이 그대로 남으면 두 가지가 나쁘다. dense 패시지의 앨범 설명 자리에 들어가 엉뚱한
    질의와 매칭되고, 임베딩 전 검증의 실패 문구 규칙에 걸려 곡이 통째로 빠진다.

    blank_comment_fields_without_evidence와 같은 원칙이다 — 근거가 없으면 틀린 값이나 변명
    대신 빈 값. 다운스트림은 빈 값을 건너뛴다(passage_builder의 앨범 설명 줄).
    앨범 텍스트는 곡의 필수 정보가 아니다. 정보가 빠지는 것이 곡이 빠지는 것보다 낫다.

    이미 비워진 값에 다시 걸어도 결과가 같다.
    """
    for key in _ALBUM_TEXT_KEYS:
        value = result_json.get(key)
        if isinstance(value, str) and value.strip() and has_failure_marker(value):
            logger.info("앨범 텍스트가 실패 문구라 비움: %s=%r", key, value[:60])
            result_json[key] = ""
    return result_json


def blank_comment_fields_without_evidence(
    result_json: Dict,
    melon_comments: List,
    youtube_comments: List,
) -> Dict:
    """댓글이 하나도 없으면 반응 요약과 팬 태그를 비운다.

    댓글 블록이 비어 있을 때 LLM은 둘 중 하나를 한다. "댓글이 제공되지 않아 알 수 없습니다"라고
    정직하게 답하거나, 가사·앨범 소개에서 지어낸다. 앞쪽은 임베딩 전 검증의 실패 문구 규칙에
    걸려 곡이 탈락하고, 뒤쪽은 그대로 통과한다. 정직한 답이 손해를 보는 셈이라 그 자리를 비운다.

    major_emotion과 같은 원칙이다 — 근거가 없으면 틀린 값 대신 빈 값.
    다운스트림은 빈 값을 건너뛴다: passage_builder의 '사용자 반응' 줄과 fans_tags 블록.

    두 곳에서 부른다. 여기(LLM 응답 정규화)와 main.build_record(레코드 조립). 분석기를 바꾸거나
    가짜로 대체해도 저장되는 레코드가 임베딩 전 검증과 어긋나지 않게 하려는 것이다. 같은 함수라
    규칙이 갈라지지 않고, 이미 비워진 값에 다시 걸어도 결과가 같다.
    """
    if melon_comments or youtube_comments:
        return result_json

    dropped = {
        key: result_json.get(key)
        for key in ("sentiment_summary", "fans_tags")
        if result_json.get(key)
    }
    if dropped:
        logger.warning("댓글이 0개여서 반응 요약·팬 태그를 비움: %r", dropped)
    result_json["sentiment_summary"] = ""
    result_json["fans_tags"] = []
    return result_json


def _clear_legacy_context_output(result_json: Dict) -> Dict:
    """Keep retired flat context fields empty on the core music-vibe path."""
    result_json["context_tags"] = []
    result_json["fact_summary"] = ""
    return result_json


def refine_data(
    metadata: Dict,
    lyrics: Optional[str],
    reaction: Optional[Dict],
) -> Dict:
    """
    LLM을 호출하여 최종 데이터 가공 (Gemini).

    Args:
        metadata: Melon 수집 데이터
        lyrics: 전체 가사
        reaction: YouTube 반응 데이터
    Returns:
        음악-vibe refinement 필드. 구형 호환용 `context_tags`와 `fact_summary`가
        응답에 섞여 오더라도 아래 정규화 단계에서 빈 값으로 만든다.
        API 키가 없거나 재시도를 모두 실패하면 빈 사전. 호출부는 이를 분석 실패로 다뤄야
        하며 정상 레코드로 저장하면 안 된다.
    """
    # Melon 댓글은 metadata 안에 있음
    melon_comments = metadata.get("melon_comments", [])

    # YouTube 댓글은 reaction 딕셔너리 안에 있음
    youtube_comments = reaction.get("comments", []) if reaction else []

    prompt = generate_prompt(
        metadata,
        lyrics,
        melon_comments,
        youtube_comments,
    )

    # === Gemini 모드 ===
    if not GEMINI_API_KEY:
        logger.error("Gemini API Key 누락! .env파일 확인요망")
        return {}

    # 재시도 설정 (유료 플랜이므로 안정성을 위해 3회 재시도)
    max_retries = 3
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            result_json = json.loads(response.text)
            result_json = validate_llm_schema(result_json)

            # Legacy model responses may still contain these optional fields.
            # No verified context is ever supplied through this core path.
            result_json = _clear_legacy_context_output(result_json)

            result_json = _normalize_major_emotion(result_json)

            result_json = blank_comment_fields_without_evidence(
                result_json, melon_comments, youtube_comments
            )

            logger.info("LLM music-vibe 응답 생성 완료 (Gemini)")

            return result_json

        except Exception as e:
            logger.warning(
                "Gemini API 호출 중 오류 (시도 %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(2)  # 2초 대기 후 재시도

    logger.error("최대 재시도 횟수 초과. 분석 실패.")
    # 예전에는 여기서 '분석실패'를 채운 사전을 돌려줬고, 호출부가 그것을 정상 레코드처럼
    # 저장해 수집 완료 목록에 넣었다. 실패 목록에도 남지 않았다. 이제 빈 사전이 실패 신호다.
    # 호출부(main)는 원천 데이터를 유지한 채 분석 단계만 다시 시도한다.
    return {}
