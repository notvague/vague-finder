import re
from typing import Dict, List
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
    
    # 의미 없는 태그(차트, टॉप100, 좋아요 등) 필터링
    noise_regex = re.compile(r"탑백|top\s*100|좋아요|차트|인기|최신|명곡|신곡|추천|베스트|비트|트렌드|역주행", flags=re.IGNORECASE)
    filtered_tags = [t for t in tags if not noise_regex.search(t)]
    
    return _unique_preserve_order(filtered_tags)

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
        
    noise_regex = re.compile(
        r"좋아요 총건수|\d+곡|\d+위|현제 페이지|현재 페이지|탑백|top\s*100|차트|인기|최신|명곡|신곡|베스트|히트|트렌드|좋아요10만|좋아요100만", 
        flags=re.IGNORECASE
    )
    if noise_regex.search(text):
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
