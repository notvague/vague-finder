"""
[Step 2] YouTube Reaction Collector (유튜브 반응 수집기)

역할:
1) yt-dlp로 "검색(ytsearch10)" → 후보 영상 10개를 가져옴
2) 커버/리믹스/리액션/플레이리스트 등 제외 키워드로 후보를 필터링
3) 남은 후보 중 조회수(view_count)가 가장 높은 영상 선택
4) 선택 영상의 상세 메타(조회수, 댓글 등)를 다시 yt-dlp로 가져옴
5) 댓글은 스팸/가사복붙 등을 필터링해서 상위 30개만 반환

추가 기능:
- cookies.txt가 있으면 yt-dlp에 쿠키를 넣어봄(봇탐지/연령/지역 제한 회피에 도움)
- download_youtube_audio()로 m4a 오디오 다운로드
"""

import re
import logging
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import yt_dlp
import os

from .refine_gemini import select_emotional_comments_with_llm

# 로깅 설정
# logging configured by main
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
    """
    1차 최소 필터:
    - 길이 8자 미만 제거
    - 길이 200자 이상 제거
    - URL 제거
    - 쇼츠/숏츠 제거
    - 화음파트/확인하러 제거
    - 욕설 제거
    - 외모칭찬 only 제거
    - 가사 복붙 제거

    반환:
    [
      {
        "text": "...",
        "like_count": 123
      },
      ...
    ]
    """

    filtered = []

    clean_lyrics = ""
    if song_lyrics:
        clean_lyrics = re.sub(r"\s+", "", song_lyrics).lower()

    # 명백한 제거 대상
    hard_stop_patterns = [
        # URL
        r"http[s]?://", r"www\.",

        # 쇼츠/유입
        r"쇼츠|숏츠|shorts",

        # 화음/확인류
        r"화음파트|화음 파트|확인하러|체크하러|에도\s?들으시는|에도\s?듣는",
        
        # 메타/유입 반응
        r"듣는\s?사람|보러\s?옴|보러\s?온|보고왔|ㅋㅋ+",

        # 욕설/분쟁
        r"빙신|병신|ㅂㅅ|개소리|꺼져|씨발|시발|ㅅㅂ|지랄",

        # 외모칭찬 only 성격 강한 키워드
        r"예쁘|이쁘|잘생|존예|존잘|비주얼|미모|얼굴|외모",
    ]

    for c in comments:
        text = c.get("text", "") if isinstance(c, dict) else str(c)
        like_count = c.get("like_count", 0) if isinstance(c, dict) else 0

        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()

        # 1) 길이 컷
        if len(clean) < 8:
            continue
        if len(clean) >= 200:
            continue

        # 2) URL 컷 / 쇼츠 / 화음 / 욕설 / 외모칭찬
        if any(re.search(p, clean, flags=re.IGNORECASE) for p in hard_stop_patterns):
            continue

        # 3) 이모지 과다 컷 (선택적이지만 최소 필터로 넣는 편이 안전)
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55\u3297\u3299]")
        emojis = emoji_pattern.findall(clean)
        if len(emojis) >= 8:
            continue

        # 4) 가사 복붙 제거
        if clean_lyrics and len(clean) >= 20:
            compact = re.sub(r"\s+", "", clean).lower()

            # 댓글이 가사의 일부를 길게 그대로 포함하면 제거
            if len(compact) >= 25 and compact in clean_lyrics:
                continue

            # 최장 공통 구간 기반 복붙 제거
            m = SequenceMatcher(None, compact, clean_lyrics).find_longest_match(
                0, len(compact), 0, len(clean_lyrics)
            )
            match_ratio = (m.size / len(compact)) if len(compact) else 0.0

            if match_ratio > 0.88 and len(compact) >= 35:
                continue

        filtered.append({
            "text": clean,
            "like_count": int(like_count or 0),
        })

    return filtered


def filter_comments(comments: List[Dict], song_lyrics: Optional[str] = None) -> List[str]:
    """
    하이브리드 필터:
    1) 최소 전처리/제거
    2) 좋아요 순 정렬
    3) LLM 판별
    4) 상위 20개 반환
    """

    # 1차 최소 필터
    candidates = preprocess_comment_candidates(comments, song_lyrics=song_lyrics)

    if not candidates:
        logger.info("[YT FILTER] 1차 필터 후 후보 0개")
        return []

    # 좋아요 순 정렬
    candidates.sort(key=lambda x: x.get("like_count", 0), reverse=True)

    logger.info(f"[YT FILTER] 1차 필터 후 후보 수: {len(candidates)}")

    # LLM으로 감성 서술형 댓글 선별
    selected = select_emotional_comments_with_llm(
        candidates,
        target_count=20,
        batch_size=15,
    )

    logger.info(f"[YT FILTER] 최종 선별 댓글 수: {len(selected)}")

    return selected


def fetch_best_video(search_query: str) -> Optional[Dict]:
    """
    유튜브 검색 결과 상위 N개(기본 10개)를 가져와서
    제외 키워드에 걸리는 영상(커버/리믹스/리액션/직캠/플리 등)을 제거하고
    남은 후보 중 조회수가 가장 높은 영상을 선택한다.

    search_query 예: "ytsearch10:아이유 밤편지"
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

    # 필터링 키워드 
    EXCLUDE_KEYWORDS = [
        "cover", "커버", "remix", "리믹스", "reaction", "리액션", "review", "리뷰",
        "mashup", "매쉬업", "loop", "시간", "playlist", "플레이리스트", 
        "mr removed", "inst", "instrumental", "fancam", "직캠",
        "teaser", "티저", "trailer", "홍보",
        # 방송사 및 채널명 필터링 
        "mnet", "dingo", "딩고", "kbs", "sbs", "mbc", "tvn", "jtbc", "Killing Voice" , "킬링보이스"
    ]
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f" YouTube 검색: {search_query}")
            # 검색 실행
            info = ydl.extract_info(search_query, download=False)
            
            # 결과 구조 확인
            if not info or 'entries' not in info:
                logger.warning("검색 결과 없음")
                return None
                
            entries = info['entries']
            candidates = []
            
            # 후보 필터링
            for entry in entries:
                if not entry: continue
                
                title = entry.get('title', '').lower()
                channel = entry.get('channel', '').lower()

                # 필터링 로직 (제목/채널명에 제외 키워드 포함 시 스킵)
                if any(kw in title or kw in channel for kw in EXCLUDE_KEYWORDS):
                    continue
                    
                candidates.append(entry)

            # 필터 후 후보가 없으면 그냥 첫 번째를 fallback
            if not candidates:
                logger.warning("모든 영상이 필터링되었습니다. 조회수가 높은 영상을 fallback으로 사용합니다.")
                best_video = entries[0]
            else:
                # 가장 조회수가 높은 영상 선택
                best_video = max(candidates, key=lambda x: x.get('view_count', 0) or 0)
                logger.info(f"Best Video Selected: {best_video.get('title')} (View: {best_video.get('view_count')})")

            # 최고 조회수 영상 정보를 반환 (fetch_youtube_reaction에서 상세 수집)
            return best_video
            
    except Exception as e:
        logger.error(f"영상 검색 중 에러: {e}")
        return None


def fetch_youtube_reaction(artist: str, title: str, song_lyrics: str = "") -> Dict:
    """
    유튜브에서:
    1) 검색 → best_video 선정
    2) 선정된 영상 URL로 상세 정보(조회수/댓글) 가져오기
    3) 댓글 필터링 후 반환

    반환 예:
    {
      "video_url": "https://www.youtube.com/watch?v=....",
      "view_count": 123456,
      "comments": [...]
    }
    """
    query = f"{artist} {title}"

    best_video_info = fetch_best_video(f"ytsearch10:{query}")

    if not best_video_info:
        logger.warning(f"최적의 영상을 찾지 못했습니다: {query}")
        return {}

    video_id = best_video_info.get("id")
    if not video_id:
        logger.error(f"선택된 영상에서 ID를 찾을 수 없습니다: {best_video_info}")
        return {}

    ydl_opts_full = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "getcomments": True,
        "js_runtimes": {"node": {}},
        "extractor_args": {
            "youtube": {
                "max_comments": ["1000"]
            }
        },
        "force_generic_extractor": False,
        "noplaylist": True,
    }

    if os.path.exists("cookies.txt"):
        ydl_opts_full["cookiefile"] = "cookies.txt"

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"

        with yt_dlp.YoutubeDL(ydl_opts_full) as ydl:
            logger.info(f"상세 데이터 수집 중: {url}")
            info = ydl.extract_info(url, download=False)

            if not info:
                logger.error(f"상세 정보 추출 실패: {video_id}")
                return {}

            video_url = info.get("webpage_url")
            view_count = info.get("view_count")

            raw_comments = info.get("comments", [])
            filtered_comments = filter_comments(raw_comments, song_lyrics=song_lyrics)

            return {
                "video_url": video_url,
                "view_count": view_count,
                "comments": filtered_comments[:10],
            }

    except Exception as e:
        logger.error(f"상세 수집 실패: {e}")
        return {}


def download_youtube_audio(video_url: str, save_path: str) -> bool:
    """
    유튜브 영상의 오디오를 m4a 포맷으로 다운로드
    save_path: 저장할 파일의 전체 경로 (확장자 포함 권장, 예: .../audio.m4a)
    """
    import os
    from pathlib import Path
    
    # 파일명에서 확장자 제거 (outtmpl이 알아서 붙이거나 처리함, 하지만 명시적 포맷팅 위해)
    path_obj = Path(save_path)
    output_tmpl = str(path_obj.with_suffix('')) # 확장자 제거한 경로
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio', # m4a 우선
        'outtmpl': output_tmpl + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
    }
    
    # [Cookie Support]
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = "cookies.txt"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"오디오 다운로드 시작: {video_url}")
            ydl.download([video_url])
            return True
    except Exception as e:
        logger.error(f"오디오 다운로드 실패: {e}")
        return False


