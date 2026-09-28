"""
[Step 1] Melon Data Collector
설명: 멜론(Melon)에서 검색 후 메타데이터, 가사, 댓글을 수집합니다.
      이모티콘 스팸 필터링 기능을 추가하여 데이터 품질을 높였습니다.
작성자: 이연우 (Data Engineer), 황찬혁 (Full)
생성일: 2026-02-02
수정일: 2026-02-04 (이모티콘 필터링, Song ID 반환 추가)
"""

import html
import requests
from bs4 import BeautifulSoup
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .collect_melon_playlist import fetch_melon_playlists_from_song_page, flatten_melon_playlist_data
from .melon_match import (
    MatchScore,
    artist_matches,
    check_details,
    normalize,
    score_candidate,
)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 멜론은 User-Agent 없으면 차단당함
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 일시 오류(5xx, 타임아웃, 연결 끊김)는 이만큼 다시 시도한다.
PAGE_RETRIES = 2
RETRY_DELAY_SEC = 3.0


class MelonAccessError(Exception):
    """접근 차단(403)이나 한도 초과(429). 다음 곡을 시도해도 같은 답이 오므로
    호출부(main)는 이 예외를 받으면 배치를 멈춰야 한다."""


class AlreadyCollected(Exception):
    """채택할 곡이 이미 수집돼 있다.

    검색 1위가 검사에서 탈락하고 2위가 이미 수집된 곡인 경우가 있다. 예전에는 그 사실을
    댓글 수집과 유튜브 수집까지 끝낸 뒤에야 알았다. 채택이 확정된 직후에 확인해 남은 요청을
    아낀다.
    """

    def __init__(self, song_id: str):
        super().__init__(f"이미 수집된 곡: {song_id}")
        self.song_id = str(song_id)


class MelonTransientError(Exception):
    """재시도를 다 써도 남은 일시 오류(5xx·타임아웃·연결 끊김).

    '이 후보는 틀렸다'와 구분해야 한다. 뭉개서 빈 사전으로 돌려주면 점수 1위 후보가 잠깐의
    503 때문에 탈락하고 2위(같은 곡의 다른 릴리스)가 레코드가 된다. 그러면 다음 실행의
    중복 검사는 1위 id로 하므로 같은 시드에 레코드가 둘 남는다. 곡 단위 실패로 올려
    다음 실행에서 다시 시도한다.
    """


def _request_with_retry(url: str, params: Optional[Dict] = None,
                        headers: Optional[Dict] = None, timeout: float = 10.0,
                        retries: int = PAGE_RETRIES):
    """멜론에 요청하고 상태를 분류한다. 페이지와 댓글 API가 같은 규칙을 쓴다.

    HTTP 상태를 보지 않고 본문을 파싱하면 403 안내 페이지도 '제목 Unknown'인 정상
    상세정보처럼 읽히고, 댓글 API의 403·503도 '댓글 없는 곡'이 된다.

    Returns:
        requests.Response. 404처럼 대상이 없으면 None.
    Raises:
        MelonAccessError: 403/429.
        MelonTransientError: 재시도 후에도 남은 5xx·타임아웃·연결 끊김.
        requests.HTTPError: 재시도로 풀리지 않는 4xx.
    """
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            res = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
        except requests.RequestException as e:
            last_error = e
            logger.warning("멜론 요청 실패 (시도 %d/%d): %s", attempt + 1, retries + 1, e)
        else:
            status = res.status_code
            if status in (403, 429):
                raise MelonAccessError(f"HTTP {status}: {url}")
            if status == 404:
                return None
            if status >= 500:
                last_error = requests.HTTPError(f"HTTP {status}: {url}")
                logger.warning("멜론 서버 오류 (시도 %d/%d): HTTP %d", attempt + 1, retries + 1, status)
            elif status != 200:
                raise requests.HTTPError(f"HTTP {status}: {url}")
            else:
                return res
        if attempt < retries:
            time.sleep(RETRY_DELAY_SEC)
    raise MelonTransientError(f"멜론 요청 재시도 초과: {last_error}")


def fetch_melon_page(url: str, params: Optional[Dict] = None, timeout: float = 10.0,
                     retries: int = PAGE_RETRIES) -> Optional[str]:
    """멜론 HTML 페이지 본문. 대상이 없으면 None."""
    res = _request_with_retry(url, params=params, timeout=timeout, retries=retries)
    return res.text if res is not None else None


def fetch_melon_json(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None,
                     timeout: float = 5.0, retries: int = PAGE_RETRIES) -> Optional[Dict]:
    """멜론 JSON API 응답. 대상이 없으면 None.

    본문이 JSON이 아니면(오류 안내 HTML 등) 일시 오류로 본다. 빈 목록으로 돌려주면
    '댓글 없는 곡'이 되어 그대로 굳는다.
    """
    res = _request_with_retry(url, params=params, headers=headers, timeout=timeout, retries=retries)
    if res is None:
        return None
    try:
        data = res.json()
    except ValueError as e:
        raise MelonTransientError(f"멜론 JSON 응답을 읽지 못함: {url} ({e})")
    return data if isinstance(data, dict) else None

def clean_text(text: str) -> str:
    """공백 및 특수문자 정리"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

from .llm_utils import select_emotional_comments_with_llm

def preprocess_melon_comment_candidates(comments_list: List[Dict]) -> List[Dict]:
    filtered: List[Dict] = []
    hard_stop_patterns = [
        r"http[s]?://", r"www\.", r"빙신|병신|ㅂㅅ|개소리|꺼져|씨발|시발|ㅅㅂ|지랄|존나|개같은",
        r"예쁘|이쁘|잘생|존예|존잘|비주얼|미모|얼굴|외모",
        r"스트리밍|스밍|총공|투표|1위|순위|차트|진입|최고|레전드",
        r"출석|1빠|등수|좋아요|하트|추천|\d{4}년|시간",
    ]
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55\u3297\u3299]")

    for item in comments_list:
        content = item.get("AUTH_CNTTS", "")
        content = re.sub(r"<[^>]+>", " ", content)
        # 멜론 API는 본문을 HTML 이스케이프해서 준다. 풀지 않으면 '&hellip;', '&quot;'가
        # 그대로 남아 LLM 입력에 섞인다(수집분 425곡 640개에서 확인).
        content = html.unescape(content)
        content = clean_text(content)
        
        if len(content) < 8 or len(content) >= 200: continue
        if any(re.search(p, content, flags=re.IGNORECASE) for p in hard_stop_patterns): continue
        emojis = emoji_pattern.findall(content)
        if len(emojis) >= 8: continue
        
        filtered.append({"text": content, "like_count": int(item.get("RECM_CNT", 0) or 0)})
    return filtered

def filter_melon_comments(comments_list: List[Dict]) -> List[str]:
    """LLM 판정을 한 건도 못 하면 CommentSelectionUnavailable을 올린다(수집 단계 실패)."""
    candidates = preprocess_melon_comment_candidates(comments_list)
    if not candidates: return []
    candidates.sort(key=lambda x: x.get("like_count", 0), reverse=True)
    logger.info(
        f"[MELON LLM FILTER] 1차 하드필터 후 후보 수: {len(candidates)} "
        f"(추천수 최대 {candidates[0].get('like_count', 0)})"
    )
    selected = select_emotional_comments_with_llm(candidates, target_count=15, batch_size=30, source_name="Melon")
    logger.info(f"[MELON LLM FILTER] LLM 2차 핵심 선별 완료: {len(selected)}개")
    return selected

_ALBUM_HREF = re.compile(r"searchLog\(\s*'[^']*'\s*,\s*'[^']*'\s*,\s*'AL'")

# 19세 미만 이용불가 표시. 멜론은 곡명 앞에 배지를 붙인다.
#   <span class="bullet_icons age_19 large" title="19세 미만 청소년 이용불가"><span class="none">19금</span></span>
# 이 곡은 **가사 영역(.lyric)이 아예 없다** — 로그인·성인 인증이 필요해서 가사를 받을 수 없다.
# 수집분 961곡 중 가사가 빈 8곡이 전부 이 경우였고, 배지의 '19금' 글자가 제목에도 섞여 들어가
# '19금 BAND', '19금 그XX'처럼 저장됐다. 검색 결과 행에도 같은 배지가 있어 상세 요청 전에 걸러낸다.
_ADULT_BADGE_SELECTOR = "span.age_19"


def has_adult_badge(node) -> bool:
    """검색 결과 행이나 상세 페이지에 19금 배지가 있는가."""
    return bool(node.select(_ADULT_BADGE_SELECTOR)) if node is not None else False


@dataclass
class SongCandidate:
    """검색 결과 한 행. 채택 여부를 판단할 재료를 모두 들고 있다."""

    song_id: str
    title: str
    artists: List[str] = field(default_factory=list)
    album: str = ""
    rank: int = 0
    match: Optional[MatchScore] = None

    def describe(self) -> str:
        score = self.match.describe() if self.match else "-"
        return f"'{self.title}' / 앨범 '{self.album}' / {score}"


def _is_album_anchor(anchor) -> bool:
    """앨범 링크인지 판별. searchLog의 세 번째 인자가 'AL'이면 앨범이다."""
    return bool(_ALBUM_HREF.search(anchor.get("href") or ""))


def _row_artists(row) -> List[str]:
    """행에서 아티스트 이름만 뽑는다."""
    tags = row.select("div#artistName a.fc_mgray, div#artistName span.checkEllipsis")
    if not tags:
        # 폴백은 div.ellipsis를 통째로 훑어서 앨범 앵커까지 딸려 온다.
        # 앨범명이 아티스트로 섞이면 엉뚱한 행이 아티스트 일치로 통과한다.
        tags = [a for a in row.select("div.ellipsis a.fc_mgray") if not _is_album_anchor(a)]
    return [clean_text(a.get_text()) for a in tags if clean_text(a.get_text())]


def _row_album(row) -> str:
    """행에서 앨범명을 뽑는다.

    제목이 똑같이 완전일치하는 행이 여럿일 때 앨범이 승부를 가른다.
    예) 'Kill This Love'는 정규 앨범판과 'THE SHOW' LIVE판이 둘 다 완전일치다.
    """
    for anchor in row.select("a.fc_mgray"):
        if _is_album_anchor(anchor):
            return clean_text(anchor.get_text())

    tds = row.select("td")
    if len(tds) >= 5:
        return clean_text(tds[4].get_text(" ", strip=True))
    return ""


def fetch_melon_song_candidates(artist: str, title: str, limit: int = 5) -> List[SongCandidate]:
    """검색 결과를 전부 점수화해 상위 후보를 순서대로 돌려준다.

    이전 구현(fetch_melon_song_id)은 제목·아티스트 부분일치를 통과한 **첫 행을 즉시
    반환**했다. 그래서 최종 선택이 멜론 검색 순위에 그대로 끌려다녔고, 같은 질의가
    2026-03-02에는 원곡을, 2026-07-29에는 일본판 라이브를 가져왔다.
    """
    search_url = "https://www.melon.com/search/total/index.htm"
    params = {"q": f"{artist} {title}"}

    try:
        body = fetch_melon_page(search_url, params=params)
    except MelonAccessError:
        raise  # 배치를 멈춰야 하는 신호. 여기서 삼키면 이후 곡이 전부 '검색 결과 없음'이 된다.
    except Exception as e:
        logger.error(f"멜론 검색 중 에러: {e}")
        return []
    if body is None:
        logger.warning(f"멜론 검색 페이지 없음(404): {artist} - {title}")
        return []
    soup = BeautifulSoup(body, "html.parser")

    detail_btns = soup.select("a.btn_icon_detail")
    if not detail_btns:
        logger.warning(f"멜론 검색 결과 없음: {artist} - {title}")
        return []

    seed_norm = normalize(title)
    accepted: List[SongCandidate] = []
    rejected: List[SongCandidate] = []
    adult: List[SongCandidate] = []
    seen_ids = set()

    for rank, btn in enumerate(detail_btns):
        row = btn.find_parent("tr")
        if not row:
            continue

        id_match = re.search(r"goSongDetail\('(\d+)'\)", btn.get("href", "") or "")
        if not id_match:
            continue
        song_id = id_match.group(1)
        if song_id in seen_ids:
            continue

        title_tag = row.select_one("a.fc_gray")
        row_title = clean_text(title_tag.get_text()) if title_tag else ""
        if not row_title:
            continue

        row_artists = _row_artists(row)
        if not artist_matches(artist, row_artists):
            continue

        row_norm = normalize(row_title)
        if not (seed_norm and (seed_norm in row_norm or row_norm in seed_norm)):
            continue

        seen_ids.add(song_id)

        # 19금 곡은 가사를 받을 수 없어 채택하지 않는다. 상세 요청 전에 거른다.
        if has_adult_badge(row):
            adult.append(
                SongCandidate(song_id=song_id, title=row_title, artists=row_artists,
                              album=_row_album(row), rank=rank,
                              match=MatchScore(score=-1000, reasons=["19금(청소년 이용불가)"],
                                               hard_reject=True))
            )
            continue

        album = _row_album(row)
        candidate = SongCandidate(
            song_id=song_id,
            title=row_title,
            artists=row_artists,
            album=album,
            rank=rank,
            match=score_candidate(title, row_title, album),
        )
        (rejected if candidate.match.hard_reject else accepted).append(candidate)

    for candidate in rejected + adult:
        logger.info(f"[후보 제외] {candidate.describe()}")

    if adult and not accepted:
        logger.warning(
            f"'{artist} - {title}'의 후보가 19금 곡뿐입니다. 가사를 받을 수 없어 수집하지 않습니다."
        )

    if not accepted:
        logger.warning(
            f"검증 실패: '{artist} - {title}'와 일치하는 국내 원곡이 1페이지에 없습니다. "
            f"(제외된 후보 {len(rejected) + len(adult)}개"
            + (f", 그중 19금 {len(adult)}개)" if adult else ")")
        )
        return []

    # 점수 내림차순, 동점이면 멜론이 매긴 순위를 따른다.
    accepted.sort(key=lambda c: (-c.match.score, c.rank))
    logger.info(
        f"[후보 {len(accepted)}개] 1위 {accepted[0].describe()}"
        + (f" | 2위 {accepted[1].describe()}" if len(accepted) > 1 else "")
    )
    return accepted[:limit]


def fetch_melon_song_id(artist: str, title: str) -> Optional[str]:
    """최선 후보의 Song ID 하나만 필요할 때 쓰는 얇은 래퍼."""
    candidates = fetch_melon_song_candidates(artist, title, limit=1)
    return candidates[0].song_id if candidates else None


# 같은 실행에서 같은 앨범을 여러 곡이 참조한다. 성공한 응답만 담는다 —
# 일시 오류로 비어 버린 값을 캐시하면 그 앨범의 모든 곡이 소개 없이 저장된다.
_ALBUM_DESC_CACHE: Dict[str, str] = {}


def reset_album_desc_cache() -> None:
    _ALBUM_DESC_CACHE.clear()


def fetch_album_desc(album_id: str) -> str:
    """앨범 상세 페이지에서 앨범 소개글 추출"""
    if not album_id:
        return ""
    if album_id in _ALBUM_DESC_CACHE:
        return _ALBUM_DESC_CACHE[album_id]
        
    url = f"https://www.melon.com/album/detail.htm?albumId={album_id}"
    try:
        body = fetch_melon_page(url)
        if body is None:
            return ""
        soup = BeautifulSoup(body, "html.parser")
        
        # 앨범 소개: <div class="dtl_albuminfo"> ... </div>
        # text만 가져오기
        intro_div = soup.select_one(".dtl_albuminfo")
        if intro_div:
            # 펼치기 전/후 이슈가 있을 수 있으나 보통 원본 HTML에 다 들어있음
            desc = clean_text(intro_div.get_text())
            # "앨범소개" 타이틀은 제거
            desc = desc.replace("앨범소개", "").strip()
            # 2000자 제한
            _ALBUM_DESC_CACHE[album_id] = desc[:2000]
            return _ALBUM_DESC_CACHE[album_id]
        else:
            _ALBUM_DESC_CACHE[album_id] = ""
            return ""
            
    except MelonAccessError:
        raise
    except Exception as e:
        logger.warning(f"앨범 소개 수집 실패: {e}")
        return ""


def parse_melon_details(body: str) -> Dict:
    """상세 페이지 HTML을 사전으로 바꾼다. 네트워크를 타지 않아 테스트로 고정할 수 있다.

    앨범 소개는 다른 페이지라서 여기서 받지 않는다. album_id만 담아 주고
    fetch_melon_details가 채운다.

    제목이나 가수를 못 찾으면 빈 사전을 돌려준다. 오류 안내 페이지, 레이아웃 변경,
    빈 응답이 모두 여기 걸린다. 'Unknown'을 채워 넘기면 호출부가 정상 데이터로 채택한다.
    """
    soup = BeautifulSoup(body or "", "html.parser")

    # 1. 메타데이터
    # 제목: <div class="song_name">
    song_name_tag = soup.select_one(".song_name")
    is_adult = has_adult_badge(song_name_tag)
    if is_adult:
        # 배지의 '19금' 글자를 떼어내지 않으면 제목이 '19금 BAND'가 된다(수집분 8곡에서 확인).
        for badge in song_name_tag.select(_ADULT_BADGE_SELECTOR):
            badge.decompose()
    title = clean_text(song_name_tag.text).replace("곡명 ", "") if song_name_tag else ""
    if not title:
        logger.warning("멜론 상세 파싱 실패: 제목(.song_name)을 찾지 못함")
        return {}

    # 가수: 여러 명일 경우 배열로 분리
    # .meta .list 안의 작곡/작사가 <a> 태그와 분리하기 위해 div.artist 영역만 탐색
    artist: List[str] = []
    artist_section = soup.select_one("div.artist")
    if artist_section:
        artist_tags = artist_section.select("span.wrap_dtl a")
        if not artist_tags:  # 구버전 레이아웃 대비 fallback
            artist_tags = [artist_section.select_one("a")]
        artist = [a.text.strip() for a in artist_tags if a and a.text.strip()]
    if not artist:
        logger.warning("멜론 상세 파싱 실패: 가수(div.artist)를 찾지 못함 (제목 %r)", title)
        return {}

    details = _parse_details_rest(soup, title, artist)
    if is_adult:
        details["is_adult"] = True
    return details


def _parse_details_rest(soup, title: str, artist: List[str]) -> Dict:
    """제목·가수를 확인한 뒤의 나머지 필드(앨범, 가사, 커버, 플레이리스트)."""
    # 앨범: <div class="meta"> <dl> ...
    # 리스트 순회하며 발매일, 장르, 앨범명+ID 추출
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
                # href="javascript:melon.link.goAlbumDetail('10554246');"
                link_tag = dd.select_one("a")
                if link_tag:
                    match = re.search(r"goAlbumDetail\('(\d+)'\)", link_tag.get("href", ""))
                    if match:
                        album_id = match.group(1)
                        logger.info(f"앨범 ID 추출 성공: {album_id} ({album_name})")
                    else:
                        logger.warning(f"앨범 ID 추출 실패 (Regex Mismatch): {link_tag.get('href')}")
                else:
                    logger.warning("앨범 ID 추출 실패 (Link Tag Not Found)")
    
    melon_playlists = fetch_melon_playlists_from_song_page(soup, max_items=2)
    playlist_flat = flatten_melon_playlist_data(melon_playlists)
    
    # 커버 이미지
    # <a href="javascript:melon.openImage('...');"><img src="...">
    cover_img = soup.select_one(".section_info .wrap_info .thumb img")
    cover_url = cover_img['src'] if cover_img else ""
    # 멜론 썸네일 리사이징 해제 -> 고화질로 변경
    if cover_url:
        cover_url = cover_url.split("/melon/")[0] + "/melon/resize/500/optimize/90/" + cover_url.split("/")[-1]
    
    # 2. 가사
    # <div class="lyric" id="d_video_summary">
    lyric_div = soup.select_one(".lyric")
    if lyric_div:
        # <br> -> \n 변환
        for br in lyric_div.find_all("br"):
            br.replace_with("\n")
        lyrics = lyric_div.get_text().strip()
        # "가사 정보 없음" -> 빈 문자열로 변경
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
        # 앨범 소개는 fetch_melon_details가 별도 요청으로 채운다. 파서는 네트워크를 타지 않는다.
        "album_id": album_id,
        "album_desc": "",
        "melon_playlist_tags": playlist_flat.get("melon_playlist_tags", [])
    }


def fetch_melon_details(song_id: str, with_album_desc: bool = True) -> Dict:
    """Song ID로 상세 페이지 메타데이터 등 크롤링.

    빈 사전은 **이 후보를 버려도 된다**는 뜻이다: 페이지 없음(404), 파싱 실패, 필수 필드 누락.
    일시 오류(5xx·타임아웃)는 MelonTransientError로, 접근 차단은 MelonAccessError로 올린다.
    그 둘을 빈 사전으로 뭉개면 잠깐의 장애가 '이 후보는 틀렸다'로 읽혀 다른 음원이 채택된다.

    with_album_desc=False면 앨범 소개를 받지 않고 album_id만 남긴다. 후보 검사에서 탈락할
    곡에 앨범 페이지 요청을 쓰지 않으려는 것이다(collect_melon_data가 검사 뒤에 채운다).
    """
    url = f"https://www.melon.com/song/detail.htm?songId={song_id}"

    try:
        body = fetch_melon_page(url)
    except (MelonAccessError, MelonTransientError):
        raise
    except Exception as e:
        logger.error(f"멜론 상세 수집 중 에러: {e}")
        return {}
    if body is None:
        logger.warning(f"멜론 상세 페이지 없음(404): songId={song_id}")
        return {}

    try:
        details = parse_melon_details(body)
    except (MelonAccessError, MelonTransientError):
        raise
    except Exception as e:
        logger.error(f"멜론 상세 파싱 중 에러: {e}")
        return {}
    if not details:
        return {}

    # 앨범 소개는 별도 페이지다. 파서를 네트워크에서 떼어 놓기 위해 여기서 받는다.
    if with_album_desc:
        album_id = details.pop("album_id", "")
        details["album_desc"] = fetch_album_desc(album_id) if album_id else ""
    return details

def fetch_melon_comments(song_id: str, max_pages: int = 5, page_size: int = 50) -> List[str]:
    """다중 페이징 싹쓸이 수집 후 LLM 2차 필터 선별"""
    api_url = "https://cmt.melon.com/cmt/api/api_listCmt.json"
    all_comments: List[Dict] = []
    headers = {"User-Agent": HEADERS["User-Agent"], "Referer": f"https://www.melon.com/song/detail.htm?songId={song_id}"}
    try:
        for page_no in range(1, max_pages + 1):
            params = {
                "cmtPocType": "pc.web", "pocId": "WP10", "chnlSeq": "103",
                "contsRefValue": song_id, "contsRefKey": song_id,
                # sortType 0은 최신순이라 추천수 0~8짜리를 긁어 온다. 1이 추천순이다.
                # 유튜브 쪽은 이미 좋아요순으로 정렬해 LLM에 넘기고 있었다.
                "sortType": "1",
                "pageSize": str(page_size), "pageNo": str(page_no),
            }
            # 상세 페이지와 같은 규칙으로 상태를 본다. 403·429는 배치 중단, 5xx·타임아웃은
            # 재시도 후 곡 단위 재시도. 예전에는 셋 다 조용히 빈 목록이 됐다.
            data = fetch_melon_json(api_url, params=params, headers=headers)
            if not data: break
            result_block = data.get("result", {})
            if not result_block: break
            raw_list = result_block.get("cmtList", [])
            if not raw_list: break
            for item in raw_list:
                info = item.get("cmtInfo", {})
                content = info.get("cmtCont", "")
                if content:
                    all_comments.append({
                        "AUTH_CNTTS": content,
                        "RECM_CNT": info.get("recmCnt", 0),
                    })
        logger.info(f"[MELON COMMENTS] raw_collected={len(all_comments)}")
    except (MelonAccessError, MelonTransientError):
        raise
    except Exception as e:
        logger.error(f"멜론 댓글 수집 중 에러: {e}")
        return []
    # 선별은 try 밖에서 부른다. CommentSelectionUnavailable을 여기서 삼키면
    # 판정 장애가 '댓글 없는 곡'으로 굳는다.
    return filter_melon_comments(all_comments)

def collect_melon_data(
    artist: str,
    title: str,
    candidates: Optional[List[SongCandidate]] = None,
    already_collected=None,
) -> Dict:
    """메인 실행 함수.

    후보를 점수순으로 훑으면서 상세 페이지 검증까지 통과한 첫 곡을 채택한다.
    보통 1위에서 끝나고, 1위가 장르 검사에 걸릴 때만 다음 후보로 넘어간다.

    Args:
        candidates: 이미 검색해 둔 후보. main에서 중복 체크용으로 한 번 받아온 것을
            그대로 넘겨 멜론 검색이 곡당 두 번 나가지 않게 한다.
        already_collected: song_id를 받아 이미 수집됐는지 답하는 함수. 채택이 확정된 직후에
            물어보고, 그렇다면 AlreadyCollected를 올려 앨범 소개·댓글 수집을 건너뛴다.

    Raises:
        AlreadyCollected: 채택할 곡이 이미 수집돼 있다.
    """
    logger.info(f"멜론 검색 시작: {artist} - {title}")

    if candidates is None:
        candidates = fetch_melon_song_candidates(artist, title)
    if not candidates:
        return {}

    for candidate in candidates:
        # 앨범 소개는 검사를 통과한 뒤에 받는다. 탈락할 후보에 요청 한 번을 아낀다.
        metadata = fetch_melon_details(candidate.song_id, with_album_desc=False)
        if not metadata:
            logger.warning(f"[후보 제외] 상세 수집 실패: {candidate.describe()}")
            continue

        # 검색 결과 행에 배지가 없었더라도(레이아웃 변경 등) 상세에서 다시 본다.
        if metadata.get("is_adult"):
            logger.warning(f"[후보 제외] 19금(청소년 이용불가), 가사 수집 불가: {candidate.describe()}")
            continue

        # 검색 결과 페이지에는 장르 열이 없어 여기서 한 번 더 본다.
        ok, note = check_details(metadata.get("genre", ""))
        if not ok:
            logger.warning(f"[후보 제외] {note}: {candidate.describe()}")
            continue
        if note:
            logger.warning(f"[확인 필요] {note}: {candidate.describe()}")

        # 가사 없는 곡은 채택하지 않는다(임베딩 검증도 빈 가사를 거부한다). 댓글 수집과 LLM
        # 선별 **앞에서** 거른다 — 뒤로 미루면 버릴 곡에 요청과 Gemini 할당량을 쓰고, 선별
        # 장애가 나면 제외 대상이 재시도 대상으로 기록된다.
        # 다음 후보로 넘어간다. 같은 곡의 다른 릴리스에는 가사가 있을 수 있다.
        if not (metadata.get("lyrics") or "").strip():
            logger.warning(f"[후보 제외] 가사 미제공: {candidate.describe()}")
            continue

        # 여기서 이 후보의 채택이 확정된다. 앨범 소개·댓글을 받기 전에 중복을 확인한다.
        if already_collected is not None and already_collected(candidate.song_id):
            raise AlreadyCollected(candidate.song_id)

        album_id = metadata.pop("album_id", "")
        metadata["album_desc"] = fetch_album_desc(album_id) if album_id else ""
        metadata["melon_comments"] = fetch_melon_comments(candidate.song_id)
        metadata["id"] = candidate.song_id

        # 레코드에는 넣지 않는다. main이 로그/검토 목록에만 쓴다.
        metadata["match_audit"] = {
            "score": candidate.match.score,
            "reasons": list(candidate.match.reasons),
            "needs_review": candidate.match.needs_review or bool(note),
            "note": note,
            "rank": candidate.rank,
            "album": candidate.album,
            "candidate_count": len(candidates),
        }
        return metadata

    logger.warning(f"모든 후보가 검증에 실패했습니다: {artist} - {title}")
    return {}
