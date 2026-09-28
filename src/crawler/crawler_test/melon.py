"""
[Step 1] Melon Data Collector

역할:
- 멜론(Melon)에서 검색 후 메타데이터, 가사, 댓글을 수집한다.
- 댓글은 YouTube와 같은 형식으로:
  1) 최소 필터
  2) LLM 배치 판별
  을 거쳐 감성 서술형 댓글만 선별한다.
"""

import requests
from bs4 import BeautifulSoup
import re
import logging
from typing import Dict, List, Optional

from .refine_gemini import select_emotional_comments_with_llm

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_melon_comment_candidates(comments_list: List[Dict]) -> List[Dict]:
    """
    멜론 댓글 1차 최소 필터
    - 길이 8자 미만 제거
    - 길이 200자 이상 제거
    - URL 제거
    - 욕설 제거
    - 외모칭찬 only 제거
    - 과도한 이모지 제거

    멜론은 like_count를 안정적으로 얻기 어려워서 0으로 고정
    """
    filtered: List[Dict] = []

    hard_stop_patterns = [
        r"http[s]?://",
        r"www\.",
        r"빙신|병신|ㅂㅅ|개소리|꺼져|씨발|시발|ㅅㅂ|지랄",
        r"예쁘|이쁘|잘생|존예|존잘|비주얼|미모|얼굴|외모",
        r"스트리밍|스밍|총공|투표|1위|순위|차트|진입",
        r"출석|1빠|등수|좋아요|하트|추천",
    ]

    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55\u3297\u3299]")

    for item in comments_list:
        content = item.get("AUTH_CNTTS", "")
        content = re.sub(r"<[^>]+>", " ", content)
        content = clean_text(content)

        if len(content) < 8:
            continue
        if len(content) >= 200:
            continue

        if any(re.search(p, content, flags=re.IGNORECASE) for p in hard_stop_patterns):
            continue

        emojis = emoji_pattern.findall(content)
        if len(emojis) >= 8:
            continue

        filtered.append({
            "text": content,
            "like_count": 0,  # 멜론은 일단 0 고정
        })

    return filtered


def filter_melon_comments(comments_list: List[Dict]) -> List[str]:
    """
    멜론 댓글 하이브리드 필터:
    1) 최소 전처리
    2) LLM 배치 판별
    3) 상위 15개 반환
    """
    candidates = preprocess_melon_comment_candidates(comments_list)

    if not candidates:
        logger.info("[MELON FILTER] 1차 필터 후 후보 0개")
        return []

    logger.info(f"[MELON FILTER] 1차 필터 후 후보 수: {len(candidates)}")

    selected = select_emotional_comments_with_llm(
        candidates,
        target_count=15,
        batch_size=15,
        source_name="Melon",
    )

    logger.info(f"[MELON FILTER] 최종 선별 댓글 수: {len(selected)}")
    return selected


def fetch_melon_song_id(artist: str, title: str) -> Optional[str]:
    """
    멜론 검색 페이지에서 첫 번째 songId 추출
    """
    search_url = "https://www.melon.com/search/song/index.htm"
    params = {
        "q": f"{artist} {title}",
        "section": "song"
    }

    try:
        res = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        target_link = soup.select_one("a.btn_icon_detail")
        if not target_link:
            logger.warning(f"멜론 검색 결과 없음: {artist} - {title}")
            return None

        href = target_link.get("href", "")
        match = re.search(r"goSongDetail\('(\d+)'\)", href)

        if match:
            return match.group(1)
        return None

    except Exception as e:
        logger.error(f"멜론 검색 중 에러: {e}")
        return None


def fetch_album_desc(album_id: str) -> str:
    """
    앨범 상세 페이지에서 앨범 소개글 추출
    """
    if not album_id:
        return ""

    url = f"https://www.melon.com/album/detail.htm?albumId={album_id}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        intro_div = soup.select_one(".dtl_albuminfo")
        if intro_div:
            desc = clean_text(intro_div.get_text())
            desc = desc.replace("앨범소개", "").strip()
            return desc[:2000]
        return ""

    except Exception as e:
        logger.warning(f"앨범 소개 수집 실패: {e}")
        return ""

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        value = clean_text(item)
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    tags = re.findall(r"#([^#\s,\/]+)", text)
    return _unique_preserve_order(tags)


def _is_noise_line(text: str) -> bool:
    if not text:
        return True

    noise_patterns = [
        r"^앨범리스트$",
        r"^좋아요 총건수",
        r"^현제 페이지",
        r"^현재 페이지",
        r"^이 곡이 포함된 DJ 플레이리스트$",
        r"^댓글$",
        r"^좋아요 한 사람$",
        r"^명예의 전당$",
        r"^Image$",
    ]
    return any(re.search(pattern, text) for pattern in noise_patterns)


def _looks_like_playlist_title(text: str) -> bool:
    if not text:
        return False
    if _is_noise_line(text):
        return False
    if text.startswith("#"):
        return False
    if re.search(r"좋아요 총건수|\d+곡|\d+위|현제 페이지|현재 페이지", text):
        return False
    return True


def _extract_playlist_title_from_item(item) -> str:
    anchor_candidates = []
    for a in item.select("a"):
        text = clean_text(a.get_text(" ", strip=True))
        if not _looks_like_playlist_title(text):
            continue

        href = (a.get("href") or "") + " " + (a.get("onclick") or "")
        score = 0
        if re.search(r"plylst|playlist|mymusicdj|djplaylist", href, flags=re.IGNORECASE):
            score += 10
        if len(text) >= 4:
            score += min(len(text), 30) / 30
        anchor_candidates.append((score, text))

    if anchor_candidates:
        anchor_candidates.sort(key=lambda x: x[0], reverse=True)
        return anchor_candidates[0][1]

    raw_text = item.get_text("\n", strip=True)
    lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]

    for line in lines:
        if not _looks_like_playlist_title(line):
            continue

        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            head, tail = parts
            if (
                len(head) >= 4
                and len(tail) <= 20
                and not tail.startswith("#")
                and not re.search(r"좋아요|총건수|\d+곡", tail)
            ):
                return head
        return line

    return ""


def _extract_playlist_tags_from_item(item) -> List[str]:
    tags: List[str] = []

    for a in item.select("a"):
        text = clean_text(a.get_text(" ", strip=True))
        if text.startswith("#"):
            tags.extend(_extract_hashtags(text))

    if not tags:
        tags.extend(_extract_hashtags(item.get_text(" ", strip=True)))

    return _unique_preserve_order(tags)


def fetch_melon_playlists_from_song_page(soup: BeautifulSoup, max_items: int = 2) -> List[Dict]:
    """
    곡 상세 페이지의 '이 곡이 포함된 DJ 플레이리스트' 영역에서
    플레이리스트 제목/태그를 최대 2개까지 추출한다.

    구조가 바뀔 수 있어서
    1) selector 기반
    2) heading 이후 텍스트 기반
    둘 다 시도한다.
    """
    playlist_items = []

    selector_candidates = [
        ".wrap_djplaylist li",
        ".section_djplaylist li",
        ".song_djplaylist li",
        ".list_djplaylist li",
        ".section_playlist li",
        ".playlist_wrap li",
    ]

    for selector in selector_candidates:
        found = soup.select(selector)
        if found:
            playlist_items = found
            break

    playlists: List[Dict] = []

    # 1차: selector 기반
    if playlist_items:
        for item in playlist_items:
            title = _extract_playlist_title_from_item(item)
            tags = _extract_playlist_tags_from_item(item)

            if not title and not tags:
                continue

            playlists.append({
                "title": title,
                "tags": tags,
            })

            if len(playlists) >= max_items:
                break

    if playlists:
        return playlists

    # 2차: fallback (텍스트 라인 기반)
    lines = [clean_text(line) for line in soup.get_text("\n").splitlines() if clean_text(line)]

    try:
        start_idx = next(i for i, line in enumerate(lines) if "이 곡이 포함된 DJ 플레이리스트" in line)
    except StopIteration:
        return []

    window = lines[start_idx + 1:start_idx + 40]
    current_title = ""

    for line in window:
        if re.search(r"좋아요 한 사람|관련비디오|댓글$", line):
            break

        if line.startswith("#"):
            tags = _extract_hashtags(line)
            if current_title or tags:
                playlists.append({
                    "title": current_title,
                    "tags": tags,
                })
                current_title = ""
                if len(playlists) >= max_items:
                    break
            continue

        if _looks_like_playlist_title(line):
            if current_title:
                playlists.append({
                    "title": current_title,
                    "tags": [],
                })
                if len(playlists) >= max_items:
                    break

            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                head, tail = parts
                if (
                    len(head) >= 4
                    and len(tail) <= 20
                    and not re.search(r"좋아요|총건수|\d+곡", tail)
                ):
                    current_title = head
                else:
                    current_title = line
            else:
                current_title = line

    if current_title and len(playlists) < max_items:
        playlists.append({
            "title": current_title,
            "tags": [],
        })

    return playlists[:max_items]


def flatten_melon_playlist_data(playlists: List[Dict]) -> Dict[str, List[str]]:
    titles = _unique_preserve_order([p.get("title", "") for p in playlists if p.get("title")])
    tags = _unique_preserve_order([
        tag
        for p in playlists
        for tag in p.get("tags", [])
    ])
    return {
        "melon_playlist_titles": titles,
        "melon_playlist_tags": tags,
    }

def fetch_melon_details(song_id: str) -> Dict:
    """
    songId로 곡 상세 페이지에서 메타데이터/가사/커버URL 추출
    """
    url = f"https://www.melon.com/song/detail.htm?songId={song_id}"

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        song_name_tag = soup.select_one(".song_name")
        title = clean_text(song_name_tag.text).replace("곡명 ", "") if song_name_tag else "Unknown"

        artist_tag = soup.select_one(".artist a")
        artist = artist_tag.text if artist_tag else "Unknown"

        meta_dl = soup.select_one(".meta .list")

        release_date = ""
        genre = ""
        album_name = ""
        album_id = ""

        if meta_dl:
            dt_list = meta_dl.find_all("dt")
            dd_list = meta_dl.find_all("dd")

            for dt, dd in zip(dt_list, dd_list):
                header = clean_text(dt.text)
                value = clean_text(dd.text)

                if "발매일" in header:
                    release_date = value
                elif "장르" in header:
                    genre = value
                elif "앨범" in header:
                    album_name = value
                    link_tag = dd.select_one("a")
                    if link_tag:
                        match = re.search(r"goAlbumDetail\('(\d+)'\)", link_tag.get("href", ""))
                        if match:
                            album_id = match.group(1)
                            logger.info(f"앨범 ID 추출 성공: {album_id} ({album_name})")

        album_desc = fetch_album_desc(album_id) if album_id else ""

        melon_playlists = fetch_melon_playlists_from_song_page(soup, max_items=2)
        playlist_flat = flatten_melon_playlist_data(melon_playlists)

        cover_img = soup.select_one(".section_info .wrap_info .thumb img")
        cover_url = cover_img["src"] if cover_img else ""

        if cover_url:
            cover_url = cover_url.split("/melon/")[0] + "/melon/resize/500/optimize/90/" + cover_url.split("/")[-1]

        lyric_div = soup.select_one(".lyric")
        if lyric_div:
            for br in lyric_div.find_all("br"):
                br.replace_with("\n")

            lyrics = lyric_div.get_text().strip()
            if "가사 정보 없음" in lyrics:
                lyrics = ""
        else:
            lyrics = ""

        return {
            "title": title,
            "artist": artist,
            "lyrics": lyrics,
            "cover_url": cover_url,
            "release_date": release_date,
            "genre": genre,
            "album_name": album_name,
            "album_desc": album_desc,
            "melon_playlists": melon_playlists,
            "melon_playlist_titles": playlist_flat.get("melon_playlist_titles", []),
            "melon_playlist_tags": playlist_flat.get("melon_playlist_tags", []),
        }

    except Exception as e:
        logger.error(f"멜론 상세 수집 중 에러: {e}")
        return {}


def fetch_melon_comments(song_id: str, max_pages: int = 5, page_size: int = 50) -> List[str]:
    """
    멜론 댓글 API를 여러 페이지 수집 후 하이브리드 필터링

    기본:
    - 최대 5페이지
    - 페이지당 50개
    => 최대 250개 원댓글 확보 후 LLM 선별
    """
    api_url = "https://cmt.melon.com/cmt/api/api_listCmt.json"
    all_comments: List[Dict] = []

    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Referer": f"https://www.melon.com/song/detail.htm?songId={song_id}"
    }

    try:
        for page_no in range(1, max_pages + 1):
            params = {
                "cmtPocType": "pc.web",
                "pocId": "WP10",
                "chnlSeq": "103",
                "contsRefValue": song_id,
                "contsRefKey": song_id,
                "sortType": "0",
                "pageSize": str(page_size),
                "pageNo": str(page_no),
            }

            res = requests.get(api_url, params=params, headers=headers, timeout=5)
            data = res.json()

            result_block = data.get("result", {})
            if not result_block:
                break

            raw_list = result_block.get("cmtList", [])
            if not raw_list:
                break

            for item in raw_list:
                content = ""
                if "cmtInfo" in item:
                    content = item["cmtInfo"].get("cmtCont", "")

                if content:
                    all_comments.append({"AUTH_CNTTS": content})

        logger.info(f"[MELON COMMENTS] raw_collected={len(all_comments)}")
        return filter_melon_comments(all_comments)

    except Exception as e:
        logger.error(f"멜론 댓글 수집 중 에러: {e}")
        return []


def collect_melon_data(artist: str, title: str) -> Dict:
    """
    Step1:
    - search -> songId
    - details -> meta/lyrics/album/cover
    - comments -> hybrid filter
    """
    logger.info(f"멜론 검색 시작: {artist} - {title}")

    song_id = fetch_melon_song_id(artist, title)
    if not song_id:
        return {}

    metadata = fetch_melon_details(song_id)

    comments = fetch_melon_comments(song_id)
    metadata["melon_comments"] = comments
    metadata["id"] = song_id

    return metadata