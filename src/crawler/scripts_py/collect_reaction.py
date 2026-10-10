"""
[Step 2] YouTube Reaction Collector
설명: yt-dlp로 유튜브에서 댓글과 조회수, 좋아요 수 데이터를 수집합니다.
      이모티콘 스팸 필터링 및 Node.js 런타임 설정으로 안정성을 확보
작성자: 이연우 (Data Engineer), 황찬혁 (Full)
생성일: 2026-01-29
수정일: 2026-02-04
"""

import re
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from difflib import SequenceMatcher
import yt_dlp

from .melon_match import (
    normalize,
    remove_song_title,
    soften,
    song_title_parts,
    strip_requested_versions,
    title_conflict,
    video_title_parts,
)

from .llm_utils import select_emotional_comments_with_llm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    불필요한 공백 제거
    """
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_comment_candidates(
    comments: List[Dict],
    song_lyrics: Optional[str] = None,
) -> List[Dict]:
    filtered = []
    clean_lyrics = ""
    if song_lyrics:
        clean_lyrics = re.sub(r"\s+", "", song_lyrics).lower()

    hard_stop_patterns = [
        r"http[s]?://", r"www\.",
        r"쇼츠|숏츠|shorts",
        r"화음파트|화음 파트|확인하러|체크하러|에도\s?들으시는|에도\s?듣는",
        r"듣는\s?사람|보러\s?옴|보러\s?온|보고왔|ㅋㅋ+",
        r"빙신|병신|ㅂㅅ|개소리|꺼져|씨발|시발|ㅅㅂ|지랄|존나|개같은",
        r"예쁘|이쁘|잘생|존예|존잘|비주얼|미모|얼굴|외모",
    ]

    for c in comments:
        text = c.get("text", "") if isinstance(c, dict) else str(c)
        like_count = c.get("like_count", 0) if isinstance(c, dict) else 0

        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()

        if len(clean) < 8 or len(clean) >= 200: continue
        if any(re.search(p, clean, flags=re.IGNORECASE) for p in hard_stop_patterns): continue

        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55\u3297\u3299]")
        if len(emoji_pattern.findall(clean)) >= 8: continue

        if clean_lyrics and len(clean) >= 20:
            compact = re.sub(r"\s+", "", clean).lower()
            if len(compact) >= 25 and compact in clean_lyrics: continue

            m = SequenceMatcher(None, compact, clean_lyrics).find_longest_match(0, len(compact), 0, len(clean_lyrics))
            match_ratio = (m.size / len(compact)) if len(compact) else 0.0

            if match_ratio > 0.88 and len(compact) >= 35: continue

        filtered.append({"text": clean, "like_count": int(like_count or 0)})

    return filtered

def filter_comments(comments: List[Dict], song_lyrics: Optional[str] = None) -> List[str]:
    """LLM 판정을 한 건도 못 하면 llm_utils.CommentSelectionUnavailable이 올라간다(수집 단계 실패)."""
    candidates = preprocess_comment_candidates(comments, song_lyrics=song_lyrics)
    if not candidates:
        logger.info("[YT LLM FILTER] 1차 필터 후 후보 0개")
        return []

    candidates.sort(key=lambda x: x.get("like_count", 0), reverse=True)
    logger.info(f"[YT LLM FILTER] 1차 하드필터 후 후보 수: {len(candidates)}")

    selected = select_emotional_comments_with_llm(
        candidates,
        target_count=20,
        batch_size=30,
        source_name="YouTube"
    )

    logger.info(f"[YT LLM FILTER] 최종 LLM 선별 댓글 수: {len(selected)}")
    return selected

# --- 영상이 요청한 곡인지 ---------------------------------------------------------------
# 가수 이름은 검색어에만 들어가고 결과 검증은 제목·버전만 봤다. 그 뒤 조회수순으로 골라서,
# 요청한 가수보다 조회수가 높은 다른 가수의 동명곡이 먼저 뽑혔다. 버전 검사는 통과하지만
# 메타데이터(멜론)와 오디오(유튜브)가 다른 곡이 된다.

# 오디오 자체가 원곡이 아닌 영상. **복원하지 않는다** — 이 후보만 남았다면 수집하지 않는 게
# 맞다. 다른 연주를 원곡 자리에 넣으면 오디오 임베딩이 가사·메타와 어긋난다.
HARD_EXCLUDE_KEYWORDS = [
    "cover", "커버", "remix", "리믹스", "mashup", "매쉬업",
    "mr removed", "inst", "instrumental", "반주",
    "piano", "피아노", "guitar", "기타연주", "violin", "바이올린",
    "music box", "오르골", "연주곡", "acoustic ver",
    "reaction", "리액션", "review", "리뷰", "fancam", "직캠",
    "teaser", "티저", "trailer",
]

# 맥락상 제외하지만 오디오는 원곡일 수 있는 영상. 다른 후보가 하나도 없으면 되살린다.
SOFT_EXCLUDE_KEYWORDS = [
    "loop", "시간", "playlist", "플레이리스트", "홍보",
    "mnet", "dingo", "딩고", "kbs", "sbs", "mbc", "tvn", "jtbc", "Killing Voice", "킬링보이스",
]

EXCLUDE_KEYWORDS = HARD_EXCLUDE_KEYWORDS + SOFT_EXCLUDE_KEYWORDS

OFFICIAL_MARKERS = ("official", "m/v", "[mv]", "mv)", " mv", "뮤직비디오", "뮤비", "official audio", "topic")


def name_in_text(name: str, text: str) -> bool:
    """가수 이름이 제목·채널명에 들어 있는가.

    'IU', 'V'처럼 짧은 영문 이름은 아무 단어에나 걸리므로 단어 경계를 요구한다.
    그 밖에는 기호·공백을 지운 뒤 부분일치로 본다('아이유(IU)' 안의 '아이유').
    """
    target = normalize(name)
    if not target or not text:
        return False
    if target.isascii() and len(target) <= 3:
        return re.search(rf"\b{re.escape(target)}\b", soften(text)) is not None
    return target in normalize(text)


def _entry_names(entry: Dict) -> List[str]:
    """yt-dlp 항목에서 가수 후보가 될 문자열을 모두 뽑는다(제목, 채널, 업로더, 아티스트)."""
    names: List[str] = []
    for key in ("title", "channel", "uploader", "artist", "creator", "album_artist"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            names.append(value)
    for key in ("artists", "creators"):
        value = entry.get(key)
        if isinstance(value, list):
            names.extend(str(v) for v in value if v)
    return names


def expand_names(known_artists: Sequence[str]) -> List[str]:
    """'태연 (TAEYEON)'처럼 멜론이 병기한 이름을 '태연', 'TAEYEON'으로 나눈다.

    통째로 비교하면 영상 제목에 둘 중 하나만 있어도 못 찾는다(코퍼스 961곡의 가수명
    246개가 이 형식이다). 원문도 함께 남긴다.
    """
    names: List[str] = []
    for raw in known_artists:
        if not raw:
            continue
        for part in [raw] + re.split(r"[()（）/,&]|\bfeat\.?\b|\bwith\b", raw, flags=re.IGNORECASE):
            part = (part or "").strip()
            if part and normalize(part) and part not in names:
                names.append(part)
    return names


def artist_evidence(entry: Dict, known_artists: Sequence[str]) -> bool:
    """검색 결과 항목(제목·채널·업로더·아티스트)에 아는 가수 이름이 하나라도 있는가."""
    texts = _entry_names(entry)
    return any(name_in_text(name, text) for name in expand_names(known_artists) for text in texts)


def is_official_entry(entry: Dict) -> bool:
    """공식 M/V·음원처럼 보이는가. 'Artist - Topic' 채널은 유튜브가 만든 공식 음원이다."""
    title = str(entry.get("title") or "").lower()
    channel = str(entry.get("channel") or entry.get("uploader") or "").lower()
    return any(m in title for m in OFFICIAL_MARKERS) or channel.endswith("- topic") or "official" in channel


def rank_candidates(entries: List[Dict], known_artists: Sequence[str]) -> Tuple[List[Dict], bool]:
    """가수 일치 -> 공식 음원 -> 조회수 순으로 정렬한다.

    Returns:
        (정렬된 후보, 가수 근거가 하나라도 있었는지)

    가수 근거가 있는 항목이 하나라도 있으면 없는 항목은 버린다. 하나도 없으면 조회수순을
    그대로 쓰되 호출부가 검토 목록에 남길 수 있게 False를 알린다.
    """
    with_artist = [e for e in entries if artist_evidence(e, known_artists)]
    verified = bool(with_artist)
    pool = with_artist if verified else list(entries)
    pool.sort(
        key=lambda e: (is_official_entry(e), e.get("view_count", 0) or 0),
        reverse=True,
    )
    return pool, verified


_HANGUL = re.compile(r"[가-힣]")


def _looks_like_translation(left: str, right: str) -> bool:
    """두 제목이 같은 곡의 다른 표기로 보이는가. 글자 체계가 달라야 한다.

    '밤편지'와 'Through the Night'는 번역 관계일 수 있지만, '니 소식'과 '니 소식2'는
    둘 다 한국어로 쓴 서로 다른 제목이다.
    """
    return bool(_HANGUL.search(left)) != bool(_HANGUL.search(right))


def video_mismatch(info: Dict, melon_title: str, known_artists: Sequence[str]) -> str:
    """상세 정보(artists/track/채널)로 다른 가수·다른 곡을 걸러낸다. 사유를 돌려주고, 맞으면 ''.

    yt-dlp는 유튜브 뮤직 메타데이터가 있으면 artists/track을 준다. 그 값이 있는데 아는
    가수와 하나도 안 맞으면 다른 가수의 동명곡이다.
    """
    credited: List[str] = []
    for key in ("artists", "creators"):
        value = info.get(key)
        if isinstance(value, list):
            credited.extend(str(v) for v in value if v)
    for key in ("artist", "creator"):
        value = info.get(key)
        if isinstance(value, str) and value:
            credited.append(value)
    # 크레딧이 있는데 크레딧·제목·채널 어디에도 아는 이름이 없으면 다른 가수다.
    # 크레딧만 보고 거르지 않는다 — 멜론은 '아이유', 유튜브 크레딧은 'IU'일 수 있다.
    if credited and known_artists and not artist_evidence(info, known_artists):
        return f"영상 아티스트 불일치 ({', '.join(credited)[:60]})"

    # 유튜브 뮤직 메타데이터의 track은 업로더가 쓴 제목보다 믿을 만하다. 다만 국내 곡이
    # 영어 제목으로 등록되는 경우가 있어(멜론 '밤편지' / track 'Through the Night'),
    # 이름이 다르다고 바로 버리지 않는다.
    #
    # 예외에는 근거가 필요하다. **글자 체계가 달라야** 번역·표기 차이로 본다. 같은 언어로
    # 쓴 다른 제목은 그냥 다른 곡이다 — '니 소식'과 '니 소식2'가 영상 제목에 나란히 있어도
    # 후속곡일 뿐이다.
    track = info.get("track")
    if isinstance(track, str) and track and melon_title:
        melon_parts = song_title_parts(melon_title)
        track_parts = song_title_parts(track)
        if melon_parts and track_parts and not (melon_parts & track_parts):
            title_parts = video_title_parts(info.get("title") or "", known_artists,
                                            keep=melon_parts | track_parts)
            both_in_title = bool(melon_parts & title_parts) and bool(track_parts & title_parts)
            if not (both_in_title and _looks_like_translation(melon_title, track)):
                return f"영상 트랙이 다른 곡: {track}"
        conflict = title_conflict(melon_title, track, known_artists)
        if conflict and not conflict.startswith("곡 이름"):
            return f"영상 트랙 불일치: {conflict}"
    return ""


def fetch_best_video(
    search_query: str,
    melon_title: str = "",
    known_artists: Sequence[str] = (),
) -> Optional[Dict]:
    """
    검색 결과 중 상위 10개를 가져와서,
    [커버/리믹스/리액션] 등을 필터링하고
    남은 것 중 가수가 맞고 공식 음원인 영상을 조회수 순으로 고른다.

    버전 검사(melon_title)는 **조회수 상위 3개로 자르기 전에** 적용한다. 나중에
    적용하면 상위 3개가 전부 듀엣판일 때 4번째에 있는 원곡을 아예 보지 못한다.

    Returns:
        후보 목록(최대 3개). 각 항목에 `_artist_verified` 키를 붙인다.
    """
    # 1. 상위 10개 검색
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True, # 빠른 검색을 위해 flat info만 
        'search_sort': 'relevance', # 관련성 순
        'default_search': 'ytsearch10' # 10개 스캔
    }

    # [Cookie Support]
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = "cookies.txt"
        logger.info("cookies.txt 감지됨: 쿠키를 사용하여 검색합니다.")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f" YouTube 검색: {search_query}")
            info = ydl.extract_info(search_query, download=False)
            
            if not info or 'entries' not in info:
                logger.warning("검색 결과 없음")
                return None
                
            entries = info['entries']
    except Exception as e:
        logger.error(f"영상 검색 중 에러: {e}")
        return []

    return select_candidates(entries, melon_title, known_artists)


def select_candidates(
    entries: Iterable[Dict],
    melon_title: str,
    known_artists: Sequence[str],
    exclude_keywords: Optional[Sequence[str]] = None,
    limit: int = 3,
    soft_keywords: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """검색 결과에서 후보를 고른다. 네트워크를 타지 않아 테스트로 고정할 수 있다."""
    # 2-a. 버전 검사 (자르기 전에 전체 후보에 적용)
    #      멜론이 고른 곡과 다른 녹음이면 오디오 임베딩이 가사·메타와
    #      어긋난다(32591630 사례).
    version_ok = []
    for entry in entries:
        if not entry: continue
        if melon_title:
            conflict = title_conflict(melon_title, entry.get('title', '') or '', known_artists)
            if conflict:
                logger.warning(
                    f"[영상 제외] {conflict} | 멜론 '{melon_title}' / 영상 '{entry.get('title','')}'"
                )
                continue
        version_ok.append(entry)

    # 2-b. 키워드 필터링 (커버/편곡/리액션/방송 등)
    hard = [kw.lower() for kw in (exclude_keywords if exclude_keywords is not None
                                  else HARD_EXCLUDE_KEYWORDS)]
    soft = [kw.lower() for kw in (soft_keywords if soft_keywords is not None
                                  else SOFT_EXCLUDE_KEYWORDS)]
    candidates, softened = [], []
    for entry in version_ok:
        # 곡 이름 안의 단어는 검사에서 뺀다. 'Piano Man', '시간'처럼 제외 키워드와 같은 말이
        # 곡 이름인 경우가 있다.
        # 곡 이름과, 시드가 요청한 버전 표시를 뺀 나머지로 검사한다.
        title = remove_song_title(entry.get('title') or '', melon_title)
        title = strip_requested_versions(title, melon_title).lower()
        channel = (entry.get('channel') or '').lower()
        haystack = f"{title} {channel}"

        if any(kw in haystack for kw in hard):
            continue                      # 편곡·커버로 판정한 후보는 되살리지 않는다
        if any(kw in haystack for kw in soft):
            softened.append(entry)
            continue
        candidates.append(entry)

    if not candidates:
        # 되살리는 것은 '맥락상 제외'뿐이다. 버전 검사와 편곡·커버 판정은 완화하지 않는다 —
        # 여기서 되돌리면 걸러낸 다른 녹음이 그대로 돌아온다.
        if softened:
            logger.warning("키워드 필터로 모두 제외됨. 편곡·커버가 아닌 후보만 되살립니다.")
        candidates = softened

    # 3. 가수 일치 -> 공식 음원 -> 조회수. 조회수만 보면 다른 가수의 동명곡이 앞선다.
    ranked, verified = rank_candidates(candidates, known_artists)
    if not verified and ranked:
        logger.warning(
            "검색 결과 어디에도 가수 이름이 없음(%s). 조회수순으로 고르되 검토 대상으로 남긴다.",
            ", ".join(known_artists) or "-",
        )
    for entry in ranked:
        entry["_artist_verified"] = verified
    return ranked[:limit]


def audio_unavailable(info: Dict) -> str:
    """내려받을 수 있는 포맷이 없는 이유. 있으면 ''.

    process=True로 부르면 yt-dlp가 포맷 선택까지 해 주고, 남는 포맷이 없으면 예외를 올린다
    (YoutubeDL.process_video_result -> raise_no_formats). 그래서 예전에는 DRM·연령 제한처럼
    내려받을 수 없는 영상이 선택 단계에서 저절로 걸러졌다. 댓글을 아끼려고 process=False로
    바꾼 뒤에는 그 검사가 돌지 않아, 내려받을 수 없는 영상이 음원으로 채택되고 assets 단계에서
    실패한다. 재수집해도 같은 영상을 다시 고르므로 실행마다 같은 곡이 같은 자리에서 실패한다.

    **yt-dlp가 거르던 것만** 거른다. DRM이 걸린 포맷(has_drm이 'maybe'가 아닌 것)을 뺀 뒤
    아무것도 남지 않는 경우다. 실시간 방송처럼 yt-dlp가 받을 수 있는 것은 예전처럼 통과시킨다.
    포맷 정보가 아예 없으면(extract_flat 응답·테스트 대역) 판단하지 않고 통과시킨다.
    """
    formats = info.get("formats")
    if formats is None:
        return ""
    playable = [
        f for f in formats
        if (not f.get("has_drm") or f.get("has_drm") == "maybe")
        and (f.get("url") or f.get("manifest_url") or f.get("fragments"))
    ]
    if not playable:
        return "내려받을 수 있는 포맷 없음 (DRM·연령 제한 등)"
    return ""


def extract_comments(info: Dict) -> Optional[List[Dict]]:
    """검증을 통과한 영상의 댓글만 뒤늦게 받는다.

    yt-dlp는 getcomments가 켜져 있으면 info['__post_extractor']에 댓글 수집 함수를 넣어 두고,
    process 단계에서 그것을 부른다. extract_info(process=False)로 메타데이터만 받아 검증한 뒤
    여기서 부르면, 탈락할 영상에는 댓글 요청이 나가지 않는다. 메타데이터를 다시 받지도 않는다.

    Returns:
        댓글 목록. 댓글이 꺼져 있으면 None(호출부가 다음 후보에서 보충한다).
    """
    if "comments" in info:            # 이미 받아 둔 경우(process=True로 부른 호출부·테스트)
        return info.get("comments")
    post_extractor = info.get("__post_extractor")
    if not callable(post_extractor):
        return None
    return (post_extractor() or {}).get("comments")


def fetch_youtube_reaction(
    artist: str,
    title: str,
    song_lyrics: str = "",
    melon_title: str = "",
    known_artists: Optional[Sequence[str]] = None,
) -> Dict:
    """
    유튜브에서 검색 -> [Smart Selection] -> 영상 메타데이터 및 댓글 수집.

    오디오 선택과 댓글 수집은 별개다. 검증을 통과한 첫 영상이 오디오 출처가 되고,
    그 영상의 댓글이 꺼져 있으면 검증된 다른 후보에서 댓글을 보충한다. 어디에도
    댓글이 없으면 빈 목록으로 돌려준다 — 댓글이 없다고 정상 음원을 버리지 않는다.

    Returns:
        video_url, video_title, view_count, comments, artist_verified,
        comment_source_url(댓글을 가져온 영상. 오디오 영상과 다를 수 있다).
        검증을 통과한 영상이 없으면 {}.
    """
    query = f"{artist} {title}"
    names = [artist] + [n for n in (known_artists or []) if n and n != artist]

    # 1. 최상의 후보 영상들(최대 3개) 탐색
    candidates = fetch_best_video(f"ytsearch10:{query}", melon_title=melon_title, known_artists=names)
    
    if not candidates:
        logger.warning(f"최적의 영상을 찾지 못했습니다: {query}")
        return {}

    # 2. 상세 데이터(댓글 포함) 수집 설정
    ydl_opts_full = {
        'quiet': True,
        'no_warnings': True, # 불필요한 포맷 관련 경고 억제
        'skip_download': True,
        'getcomments': True, # 댓글 수집
        'js_runtimes': {'node': {}}, # JS 엔진 강제 지정 (Node.js)
        'extractor_args': {
            'youtube': {
                'max_comments': ['1000']
            }
        },
        'force_generic_extractor': False, 
        'noplaylist': True, 
    }
    
    # [Cookie Support]
    if os.path.exists("cookies.txt"):
        ydl_opts_full['cookiefile'] = "cookies.txt"

    audio: Optional[Dict] = None
    comments: Optional[List[str]] = None
    comment_source_url = ""

    # 3. 후보를 순서대로 검증한다. 첫 통과 영상이 오디오, 댓글이 있는 첫 통과 영상이 댓글 출처.
    for i, video_info in enumerate(candidates):
        video_id = video_info.get('id')
        if not video_id:
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f" 후보 {i+1}: {video_info.get('title')} (View: {video_info.get('view_count')}) {url}")

        # 상세 메타데이터만 먼저 받는다. process=False면 yt-dlp가 댓글 수집 후처리를
        # 실행하지 않고 호출 가능한 형태로 남겨 둔다. 검증에서 탈락할 영상에 댓글 요청
        # (최대 1,000개)을 쓰지 않으려는 것이다.
        try:
            with yt_dlp.YoutubeDL(ydl_opts_full) as ydl:
                info = ydl.extract_info(url, download=False, process=False)
        except Exception as e:
            logger.error(f"상세 수집 실패 (후보 {i+1}): {e}")
            continue
        if not info:
            continue

        reason = video_mismatch(info, melon_title, names)
        if reason:
            logger.warning(f"[영상 제외] {reason} | {info.get('title')}")
            continue

        reason = audio_unavailable(info)
        if reason:
            logger.warning(f"[영상 제외] {reason} | {info.get('title')}")
            continue

        # 검증을 통과한 영상에서만 댓글을 받는다. 댓글 수집이 실패하면 **이 후보를 버린다** —
        # 예전에는 댓글 수집이 extract_info 안에서 돌아 그 예외가 곧 '후보 탈락'이었다.
        # 음원을 먼저 확정해 두면, 댓글이 끊긴 영상이 음원 출처로 남아 예전과 다른 곡이 된다.
        try:
            raw_comments = extract_comments(info)
        except Exception as e:
            logger.error(f"댓글 수집 실패 (후보 {i+1}): {e}")
            continue

        if audio is None:
            audio = {
                "video_url": info.get("webpage_url") or url,
                "video_title": info.get("title") or video_info.get("title") or "",
                "view_count": info.get("view_count"),
                "artist_verified": bool(video_info.get("_artist_verified", True)),
            }
            logger.info(f"Best Video Selected: {audio['video_title']} (View: {audio['view_count']})")

        if raw_comments is None:
            # 댓글 사용 중지. 음원은 유효하므로 댓글만 다음 후보에서 찾는다.
            logger.info(f" [댓글 비활성화] 후보 {i+1}. 댓글은 다른 후보에서 보충한다.")
            continue

        comments = filter_comments(raw_comments, song_lyrics=song_lyrics)
        comment_source_url = info.get("webpage_url") or url
        break

    if audio is None:
        logger.error(f"검증을 통과한 영상이 없습니다: {query}")
        return {}

    if comments is None:
        logger.warning(f"모든 후보의 댓글이 꺼져 있어 유튜브 댓글 없이 진행: {query}")
        comments = []

    return {
        **audio,
        "comments": comments[:30],  # 상위 30개 반환
        "comment_source_url": comment_source_url,
    }


# --- 오디오 다운로드 ------------------------------------------------------------------------
# 다운로드 포맷은 bestaudio[ext=m4a]/bestaudio였는데 완료 판정·Mongo 경로·임베딩 입력은
# 전부 audio.m4a를 기대한다. audio.webm만 생기면 저장은 성공했다고 하고 재실행에서는
# 미완료로 판정돼 재수집과 JSONL 중복이 생겼다. 모든 단계가 audio.m4a 하나만 보게 한다.

AUDIO_FILENAME = "audio.m4a"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# 403일 때 같은 영상을 다시 받는 횟수(첫 시도 포함)와 그 사이 대기.
AUDIO_DOWNLOAD_ATTEMPTS = 3
AUDIO_RETRY_DELAY_SEC = (5.0, 10.0)


def audio_download_options(output_stem: str) -> Dict:
    """ffmpeg가 있으면 어떤 포맷이든 받아 m4a로 변환하고, 없으면 m4a만 받는다."""
    opts = {
        'outtmpl': output_stem + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        # 댓글 수집(fetch_youtube_reaction)과 같은 JS 엔진. 없으면 yt-dlp가 'JS 엔진 없는 추출은
        # 지원 중단'이라 경고하고 예비 클라이언트로만 받는다(2026-10-10 확인).
        'js_runtimes': {'node': {}},
    }
    if ffmpeg_available():
        opts['format'] = 'bestaudio[ext=m4a]/bestaudio'
        opts['postprocessors'] = [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}]
    else:
        opts['format'] = 'bestaudio[ext=m4a]'
    return opts


def looks_like_m4a(path: Path) -> bool:
    """MP4 계열 컨테이너인지 헤더로 본다(4~8바이트 'ftyp'). HTML 오류 본문이나 webm은 걸린다."""
    try:
        with path.open("rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] == b"ftyp"


def download_youtube_audio(video_url: str, save_path: str) -> bool:
    """
    유튜브 영상의 오디오를 audio.m4a로 내려받는다.
    save_path: 저장할 파일의 전체 경로 (예: .../audio.m4a). 확장자는 m4a로 고정된다.

    끝나면 save_path에 m4a 헤더를 가진 파일이 있어야 True다. 다른 확장자만 남으면
    False이고, 남은 audio.* 조각은 지운다.
    """
    path_obj = Path(save_path).with_suffix('.m4a')
    output_tmpl = str(path_obj.with_suffix(''))  # 확장자 제거한 경로
    ydl_opts = audio_download_options(output_tmpl)
    
    # [Cookie Support]
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = "cookies.txt"

    # 영상 서버(googlevideo)의 403은 무작위다 — 2026-10-10 707회 중 74회, 같은 영상을 다시 받으면
    # 대부분 성공했다. 403일 때만 잠시 쉬고 다시 받는다. 매번 새로 추출하므로 다운로드 주소도 새것이다.
    for attempt in range(1, AUDIO_DOWNLOAD_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"오디오 다운로드 시작: {video_url}"
                            + (f" (재시도 {attempt - 1}/{AUDIO_DOWNLOAD_ATTEMPTS - 1})" if attempt > 1 else ""))
                ydl.download([video_url])
            break
        except Exception as e:
            _remove_partial_audio(path_obj)
            if "HTTP Error 403" in str(e) and attempt < AUDIO_DOWNLOAD_ATTEMPTS:
                wait = random.uniform(*AUDIO_RETRY_DELAY_SEC)
                logger.warning(f"오디오 다운로드 403 -> {wait:.1f}초 뒤 다시 받는다: {e}")
                time.sleep(wait)
                continue
            logger.error(f"오디오 다운로드 실패: {e}")
            return False

    if not path_obj.is_file() or path_obj.stat().st_size == 0 or not looks_like_m4a(path_obj):
        leftovers = [p.name for p in path_obj.parent.glob(path_obj.stem + ".*")]
        logger.error(
            "오디오가 audio.m4a로 저장되지 않음 (남은 파일: %s). ffmpeg가 없으면 m4a 포맷이 없는 영상은 받을 수 없다.",
            ", ".join(leftovers) or "없음",
        )
        _remove_partial_audio(path_obj)
        return False
    return True


def _remove_partial_audio(path_obj: Path) -> None:
    """실패한 다운로드의 audio.* 조각을 지운다. 남기면 다음 판정이 이 조각을 오디오로 본다."""
    for leftover in path_obj.parent.glob(path_obj.stem + ".*"):
        try:
            leftover.unlink()
        except OSError:
            pass
