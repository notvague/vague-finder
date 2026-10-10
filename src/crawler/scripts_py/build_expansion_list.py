"""
[Expansion] 지정 아티스트 인기곡으로 추가 크롤링 목록(artist, title)을 만든다.

    # 1) 네트워크 없이: 캐시·진행 파일만으로 목록을 다시 만든다
    venv/bin/python -m src.crawler.scripts_py.build_expansion_list --offline

    # 2) 시험 수집(앞 3명). 실제 멜론 요청이 나간다 — 사용자 확인 뒤에만 실행
    venv/bin/python -m src.crawler.scripts_py.build_expansion_list --first 3 --skip-must-add

가사·음원·앨범아트는 받지 않는다. 아티스트마다 멜론 요청 2회(검색 1 + 인기순 곡 목록 1),
검색이 실패하면 alias로 1회 더.

멜론 차단 방지 (2026-10-09 이 IP가 한 번 막혔다):
- 요청은 한 번에 하나, 요청 사이 3~6초 무작위 대기. 재시도·프록시·UA 교체 없음
- 받은 응답은 전부 캐시하고, 캐시에 있으면 요청하지 않는다
- 403·429·빈 응답·차단/캡차 페이지·예상과 다른 구조가 오면 그 자리에서 전체를 멈춘다
- 요청 횟수는 진행 파일에 누적되고, 상한(기본 300)에 닿으면 멈춘다
- 크롤러(src.crawler.main)가 돌고 있으면 시작하지 않는다(같은 IP)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from bs4 import BeautifulSoup

from .collect_melon_data import HEADERS, has_adult_badge
from .melon_match import chunks, is_instrumental, labels_in

logger = logging.getLogger("build_expansion_list")

REASON_EXISTING = "기존 중복"
REASON_VARIANT = "버전 변형"
REASON_CANDIDATE_DUP = "후보 중복"
REASON_CAP = "상한 초과"
# 요청서의 네 사유 밖이다. 크롤러가 어차피 받지 못하는 곡이라 상한 자리를 차지하지 않게 뺀다.
REASON_ADULT = "19금(가사 수집 불가)"
REASON_NOT_CREDITED = "아티스트 불일치"

DEFAULT_EXISTING = Path("data/songs_3010.csv")
DEFAULT_OUT_DIR = Path("data/expansion")
DEFAULT_CACHE_DIR = Path("data/cache/melon_expansion")


# =============================================================================
# 정규화 — 네트워크 없음
# =============================================================================

_OPEN = "([{（［【「『〈<"
_CLOSE = ")]}）］】」』〉>"
_BRACKET = re.compile(r"[\(\[\{（［【「『〈<][^\)\]\}）］】」』〉>]*[\)\]\}）］】」』〉>]")
# 괄호 밖에 붙은 크레딧. 'with'는 'Dance With Me' 같은 제목이 있어 넣지 않는다.
_TRAILING_CREDIT = re.compile(r"\s+(feat\.?|ft\.|prod\.?)\s+.*$", re.IGNORECASE)
_CREDIT_SEGMENT = re.compile(r"^\s*(feat|ft|prod|with|narr)\b", re.IGNORECASE)


def norm_text(text: object) -> str:
    """비교용 정규화. 소문자·악센트 제거 뒤 글자와 숫자만 남긴다(어느 문자 체계든)."""
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[\W_]+", "", s)


def title_key(title: str) -> str:
    """제목 키. 괄호 속 부가 정보(Feat.·Prod.·OST·Remaster…)를 떼고 정규화한다.

    괄호를 떼면 아무것도 안 남는 제목('(Intro)' 같은)은 원래 제목으로 키를 만든다.
    """
    raw = unicodedata.normalize("NFKC", str(title or ""))
    stripped = _TRAILING_CREDIT.sub("", _BRACKET.sub(" ", raw))
    return norm_text(stripped) or norm_text(raw)


def split_artists(text: str) -> List[str]:
    """'아이유, 지코' 같은 협업 표기를 쪼갠다. 괄호 안의 쉼표·&는 쪼개지 않는다.

    '싹쓰리 (유두래곤, 린다G, 비룡)'은 한 아티스트다.
    """
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in str(text or ""):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        if depth == 0 and ch in ",&":
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def artist_tokens(name: str) -> Set[str]:
    """한 아티스트의 이름 열쇠들. 'TWICE (트와이스)' -> {twice트와이스, twice, 트와이스}.

    괄호 안이 멤버 목록('유두래곤, 린다G, 비룡', '노홍철 & 싸이')이면 열쇠로 쓰지 않는다.
    """
    tokens = {norm_text(name), norm_text(_BRACKET.sub(" ", name))}
    for inner in chunks(name):
        if "," not in inner and "&" not in inner:
            tokens.add(norm_text(inner))
    return {t for t in tokens if t}


class AliasMap:
    """artists_manual.csv의 alias를 대표 이름으로 묶는다.

    alias의 괄호 속은 열쇠로 쓰지 않는다 — '원필 (DAY6)'의 'day6'을 원필로 묶으면
    목록의 DAY6가 원필이 된다.
    """

    def __init__(self) -> None:
        self._to_canonical: Dict[str, str] = {}

    @classmethod
    def from_rows(cls, rows: Iterable[Dict[str, str]]) -> "AliasMap":
        amap = cls()
        for row in rows:
            artist = (row.get("artist") or "").strip()
            if not artist:
                continue
            canonical = norm_text(_BRACKET.sub(" ", artist)) or norm_text(artist)
            for name in (artist, (row.get("alias") or "").strip()):
                if not name:
                    continue
                for token in {norm_text(name), norm_text(_BRACKET.sub(" ", name))}:
                    if not token:
                        continue
                    known = amap._to_canonical.get(token)
                    if known is not None and known != canonical:
                        logger.warning("[alias 충돌] '%s'가 %s와 %s 둘에 걸린다 — 앞의 것을 쓴다",
                                       token, known, canonical)
                        continue
                    amap._to_canonical[token] = canonical
        return amap

    def canonical(self, token: str) -> str:
        return self._to_canonical.get(token, token)

    def artist_set(self, artist_field: str) -> frozenset:
        """협업 표기를 쪼개고, 이름 열쇠를 대표 이름으로 바꾼 집합."""
        return frozenset(
            self.canonical(token)
            for name in split_artists(artist_field)
            for token in artist_tokens(name)
        )


# 버전 변형으로 보는 라벨(melon_match). 듀엣판은 원곡인 경우가 많아('(Duet With 차은주)') 뺀다.
_VARIANT_LABELS = frozenset({"반주/MR", "라이브판", "편곡판", "일본어판", "중국어판", "영어판", "외국어판"})
_VERSION_WORD = re.compile(r"\b(ver|version)\b|버전", re.IGNORECASE)
_KEEP_VERSION = re.compile(r"\b(korean|original)\s*ver|한국어\s*(버전|ver)", re.IGNORECASE)
# EDM의 'Original Mix'는 원곡이다(숀 'Lunisolar (Original Mix)').
_ORIGINAL_MIX = re.compile(r"\boriginal\s*mix\b", re.IGNORECASE)
_EXTRA_VARIANT = re.compile(
    r"\b(sped\s*up|speed\s*up|slowed|nightcore|demo|guide|karaoke|radio\s*edit|extended|tv\s*size)\b",
    re.IGNORECASE,
)


def variant_reason(title: str) -> str:
    """버전 변형 표시가 있으면 그 표시를, 없으면 빈 문자열.

    괄호 속과 ' - ' 뒤만 본다. 제목 본문까지 보면 'Live My Life'·'Mr.Mr.'가 걸린다.
    """
    if is_instrumental(title):
        return "Inst/MR"
    segments = list(chunks(title))
    if " - " in str(title):
        segments.extend(str(title).split(" - ")[1:])
    for seg in segments:
        if _CREDIT_SEGMENT.match(seg) or _ORIGINAL_MIX.search(seg):
            continue
        hit = labels_in(seg) & _VARIANT_LABELS
        if hit:
            return f"{'/'.join(sorted(hit))}: ({seg})"
        if _VERSION_WORD.search(seg) and not _KEEP_VERSION.search(seg):
            return f"Ver.: ({seg})"
        m = _EXTRA_VARIANT.search(seg)
        if m:
            return f"{m.group(1)}: ({seg})"
    return ""


# =============================================================================
# 목록 조립 — 네트워크 없음
# =============================================================================

@dataclass
class Song:
    artist: str
    title: str
    song_id: str = ""
    source: str = ""            # 'existing' / 'must_add' / 아티스트 이름(artists_manual)
    rank: int = 0               # 인기순 순위(0부터)
    adult: bool = False
    credited: bool = True       # 검색한 아티스트가 곡의 아티스트에 들어 있는가


@dataclass
class Excluded:
    song: Song
    reason: str
    detail: str = ""


class DupIndex:
    """제목 키가 같고 아티스트가 한 명이라도 겹치면 같은 곡. 멜론 song_id가 같아도 같은 곡."""

    def __init__(self, aliases: AliasMap) -> None:
        self.aliases = aliases
        self._by_key: Dict[str, List[Tuple[frozenset, Song]]] = {}
        self._by_id: Dict[str, Song] = {}

    def add(self, song: Song) -> None:
        self._by_key.setdefault(title_key(song.title), []).append(
            (self.aliases.artist_set(song.artist), song)
        )
        if song.song_id:
            self._by_id[song.song_id] = song

    def find(self, song: Song) -> Optional[Song]:
        if song.song_id and song.song_id in self._by_id:
            return self._by_id[song.song_id]
        artists = self.aliases.artist_set(song.artist)
        for other_artists, other in self._by_key.get(title_key(song.title), []):
            if artists & other_artists:
                return other
        return None


@dataclass
class AssembleResult:
    new_songs: List[Song] = field(default_factory=list)
    excluded: List[Excluded] = field(default_factory=list)
    added_per_artist: Dict[str, int] = field(default_factory=dict)
    must_add_added: int = 0


def _describe(song: Song) -> str:
    sid = f" [{song.song_id}]" if song.song_id else ""
    return f"{song.artist} - {song.title}{sid}"


def assemble(
    existing: Sequence[Song],
    candidates: Dict[str, List[Song]],
    must_add: Sequence[Song],
    aliases: AliasMap,
    cap: int = 20,
) -> AssembleResult:
    """후보를 거르고 합친다.

    순서: must_add(상한 없음) → artists_manual 순서대로 아티스트별 인기순.
    must_add를 먼저 넣으므로 같은 곡이 후보에 있으면 후보 쪽이 '후보 중복'이 된다.
    must_add 곡은 해당 아티스트의 상한 자리를 쓰지 않는다.
    """
    existing_idx = DupIndex(aliases)
    for song in existing:
        existing_idx.add(song)
    accepted_idx = DupIndex(aliases)
    result = AssembleResult()

    for song in must_add:
        hit = existing_idx.find(song)
        if hit is not None:
            result.excluded.append(Excluded(song, REASON_EXISTING, _describe(hit)))
            continue
        hit = accepted_idx.find(song)
        if hit is not None:
            result.excluded.append(Excluded(song, REASON_CANDIDATE_DUP, _describe(hit)))
            continue
        warn = variant_reason(song.title)
        if warn:
            logger.warning("[must_add] 버전 표기가 있지만 그대로 넣는다: %s (%s)", _describe(song), warn)
        accepted_idx.add(song)
        result.new_songs.append(song)
        result.must_add_added += 1

    for artist, songs in candidates.items():
        added = 0
        for song in sorted(songs, key=lambda s: s.rank):
            if song.adult:
                result.excluded.append(Excluded(song, REASON_ADULT))
                continue
            if not song.credited:
                result.excluded.append(Excluded(song, REASON_NOT_CREDITED, f"검색 아티스트 {artist}"))
                continue
            reason = variant_reason(song.title)
            if reason:
                result.excluded.append(Excluded(song, REASON_VARIANT, reason))
                continue
            hit = existing_idx.find(song)
            if hit is not None:
                result.excluded.append(Excluded(song, REASON_EXISTING, _describe(hit)))
                continue
            hit = accepted_idx.find(song)
            if hit is not None:
                result.excluded.append(Excluded(song, REASON_CANDIDATE_DUP, _describe(hit)))
                continue
            if added >= cap:
                result.excluded.append(Excluded(song, REASON_CAP, f"{artist} {cap}곡 초과"))
                continue
            accepted_idx.add(song)
            result.new_songs.append(song)
            added += 1
        result.added_per_artist[artist] = added
    return result


# =============================================================================
# 입출력
# =============================================================================

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def load_existing(path: Path) -> List[Song]:
    return [
        Song(artist=r.get("artist", ""), title=r.get("title", ""), song_id=r.get("song_id", ""), source="existing")
        for r in read_csv_rows(path)
    ]


def load_must_add(path: Path) -> List[Song]:
    return [Song(artist=r["artist"], title=r["title"], source="must_add") for r in read_csv_rows(path)]


def write_outputs(result: AssembleResult, out_dir: Path) -> Tuple[Path, Path]:
    """기존 크롤링 목록과 같은 형식: UTF-8(BOM 없음), csv 모듈 기본 줄바꿈(CRLF)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / "new_songs.csv"
    exc_path = out_dir / "excluded.csv"
    with new_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title"])
        w.writerows([s.artist, s.title] for s in result.new_songs)
    with exc_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title", "reason"])
        w.writerows([e.song.artist, e.song.title, e.reason] for e in result.excluded)
    # 크롤러 입력용. song_id가 있으면 크롤러가 검색을 건너뛴다(must_add는 비어 있어 검색한다).
    with (out_dir / "new_songs_with_ids.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title", "song_id"])
        w.writerows([s.artist, s.title, s.song_id] for s in result.new_songs)
    return new_path, exc_path


# =============================================================================
# 멜론 요청 — 한 번에 하나, 캐시 우선, 이상하면 멈춤
# =============================================================================

SEARCH_URL = "https://www.melon.com/search/total/index.htm"
SONG_LIST_URL = "https://www.melon.com/artist/songPaging.htm"

_BLOCK_MARKERS = ("captcha", "자동입력 방지", "비정상적인 접근", "접근이 차단", "access denied", "too many requests")
# 정상적인 '결과 없음' 페이지 표시. 페이지 종류마다 따로 본다.
SEARCH_EMPTY_MARKERS = ("검색결과가 없습니다", "검색 결과가 없습니다")
SONG_LIST_EMPTY_MARKERS = ("곡이 없습니다", "결과가 없습니다")
_ARTIST_HREF = re.compile(r"goArtistDetail\(\s*'?(\d+)'?\s*\)")
_SONG_HREF = re.compile(r"goSongDetail\(\s*'?(\d+)'?\s*\)")
MIN_BODY_BYTES = 200


class StopCrawl(Exception):
    """전체를 멈춰야 하는 응답. 재시도하지 않는다."""

    def __init__(self, reason: str, request: str) -> None:
        super().__init__(f"{reason} | 요청: {request}")
        self.reason = reason
        self.request = request


class BudgetReached(Exception):
    """요청 상한(전체 또는 이번 실행)에 닿았다. 오류가 아니라 정상 정지."""


def _request_label(url: str, params: Dict[str, str]) -> str:
    return requests.Request("GET", url, params=params).prepare().url or url


class Progress:
    """진행 파일. 요청 횟수와 아티스트별 결과를 한 파일에 둔다(원자적 저장)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {"requests_made": 0, "artists": {}, "stops": []}

    @property
    def requests_made(self) -> int:
        return int(self.data.get("requests_made", 0))

    def count_request(self) -> None:
        self.data["requests_made"] = self.requests_made + 1
        self.save()

    def artist(self, name: str) -> Optional[Dict]:
        return self.data["artists"].get(name)

    def set_artist(self, name: str, record: Dict) -> None:
        self.data["artists"][name] = record
        self.save()

    def record_stop(self, stop: StopCrawl) -> None:
        self.data.setdefault("stops", []).append(
            {"at": datetime.now().isoformat(timespec="seconds"), "reason": stop.reason, "request": stop.request}
        )
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


class MelonClient:
    def __init__(self, cache_dir: Path, progress: Progress, *, max_requests: int, run_budget: Optional[int],
                 delay: Tuple[float, float], offline: bool) -> None:
        self.cache_dir = cache_dir / "responses"
        self.progress = progress
        self.max_requests = max_requests
        self.run_budget = run_budget
        self.delay = delay
        self.offline = offline
        self.requests_this_run = 0
        self._session = requests.Session()

    def _cache_path(self, label: str) -> Path:
        return self.cache_dir / f"{hashlib.sha1(label.encode('utf-8')).hexdigest()}.html"

    def get(self, url: str, params: Dict[str, str], headers: Dict[str, str], expect: str,
            empty_markers: Sequence[str] = ()) -> Optional[str]:
        """본문을 돌려준다. 오프라인이고 캐시가 없으면 None."""
        label = _request_label(url, params)
        cached = self._cache_path(label)
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        if self.offline:
            return None
        if self.progress.requests_made >= self.max_requests:
            raise BudgetReached(f"전체 요청 상한 {self.max_requests}회에 닿았다")
        if self.run_budget is not None and self.requests_this_run >= self.run_budget:
            raise BudgetReached(f"이번 실행 요청 상한 {self.run_budget}회에 닿았다")

        wait = random.uniform(*self.delay)
        logger.info("  %.1f초 대기 후 요청 #%d: %s", wait, self.progress.requests_made + 1, label)
        time.sleep(wait)
        # 보내기 전에 센다. 응답 도중 죽어도 요청은 나간 것이다.
        self.progress.count_request()
        self.requests_this_run += 1
        try:
            res = self._session.get(url, params=params, headers=headers, timeout=15)
        except requests.RequestException as e:
            raise StopCrawl(f"연결 오류(재시도 안 함): {e.__class__.__name__}: {e}", label)
        if res.status_code != 200:
            raise StopCrawl(f"HTTP {res.status_code}", label)
        res.encoding = res.encoding or "utf-8"
        body = res.text
        if len(body.encode("utf-8")) < MIN_BODY_BYTES:
            raise StopCrawl(f"빈 응답({len(body)}자)", label)
        lowered = body.lower()
        for marker in _BLOCK_MARKERS:
            if marker in lowered:
                self._quarantine(label, body)
                raise StopCrawl(f"차단/캡차 의심 문구 '{marker}'", label)
        if expect not in body and not any(m in body for m in empty_markers):
            self._quarantine(label, body)
            raise StopCrawl(f"예상과 다른 구조('{expect}' 없음)", label)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(body, encoding="utf-8")
        return body

    def _quarantine(self, label: str, body: str) -> None:
        """멈춘 응답은 캐시에 넣지 않고 따로 남긴다(원인 확인용)."""
        path = self.cache_dir.parent / "stopped" / f"{datetime.now():%Y%m%d-%H%M%S}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<!-- {label} -->\n{body}", encoding="utf-8")
        logger.error("  멈춘 응답 저장: %s", path)


def parse_artist_anchors(body: str) -> List[Tuple[str, str]]:
    """페이지의 아티스트 링크를 순서대로 (artist_id, 이름)."""
    soup = BeautifulSoup(body, "html.parser")
    out: List[Tuple[str, str]] = []
    for a in soup.find_all("a"):
        m = _ARTIST_HREF.search((a.get("href") or "") + " " + (a.get("onclick") or ""))
        name = " ".join(a.get_text(" ", strip=True).split())
        if m and name:
            out.append((m.group(1), name))
    return out


@dataclass
class ArtistPick:
    artist_id: str
    name: str
    hits: int
    others: List[Tuple[str, str, int]] = field(default_factory=list)   # 이름이 맞는 다른 ID(동명이인 후보)


def pick_artist(anchors: Sequence[Tuple[str, str]], wanted: Set[str], aliases: AliasMap) -> Optional[ArtistPick]:
    """이름 열쇠가 맞는 아티스트 중 페이지에 가장 많이(같으면 먼저) 나온 ID."""
    stats: Dict[str, List] = {}
    for pos, (aid, name) in enumerate(anchors):
        tokens = {aliases.canonical(t) for t in artist_tokens(name)}
        if not tokens & wanted:
            continue
        entry = stats.setdefault(aid, [name, 0, pos])
        entry[1] += 1
    if not stats:
        return None
    ranked = sorted(stats.items(), key=lambda kv: (-kv[1][1], kv[1][2]))
    (aid, (name, hits, _)), rest = ranked[0], ranked[1:]
    return ArtistPick(aid, name, hits, [(oid, oname, ohits) for oid, (oname, ohits, _) in rest])


def parse_song_rows(body: str) -> List[Dict]:
    """아티스트 곡 목록의 행을 순서대로. 각 행: song_id, title, artists[(id, name)], adult."""
    soup = BeautifulSoup(body, "html.parser")
    rows: List[Dict] = []
    seen: Set[str] = set()
    for tr in soup.select("tr"):
        song_id = ""
        check = tr.select_one("input.input_check")
        if check is not None and (check.get("value") or "").isdigit():
            song_id = check["value"]
        if not song_id:
            m = _SONG_HREF.search(str(tr))
            song_id = m.group(1) if m else ""
        if not song_id or song_id in seen:
            continue

        title = ""
        if check is not None and (check.get("title") or "").endswith("곡 선택"):
            title = check["title"][: -len("곡 선택")].strip()
        if not title:
            span = tr.select_one("a.btn_icon_detail span.odd_span")
            if span is not None:
                title = re.sub(r"\s*상세정보 페이지 이동$", "", span.get_text(strip=True))
        if not title:
            a = tr.select_one("a.fc_gray")
            title = a.get_text(strip=True) if a is not None else ""
        if not title:
            continue

        artists: List[Tuple[str, str]] = []
        for a in tr.find_all("a"):
            m = _ARTIST_HREF.search((a.get("href") or "") + " " + (a.get("onclick") or ""))
            name = " ".join(a.get_text(" ", strip=True).split())
            if m and name and (m.group(1), name) not in artists:
                artists.append((m.group(1), name))
        seen.add(song_id)
        rows.append({"song_id": song_id, "title": title, "artists": artists, "adult": has_adult_badge(tr)})
    return rows


def collect_artist(client: MelonClient, row: Dict[str, str], aliases: AliasMap, top_n: int) -> Dict:
    """한 아티스트: 검색(실패 시 alias로 한 번 더) → 인기순 곡 목록 상위 top_n."""
    name, alias = row["artist"], row.get("alias", "")
    # extra_aliases: 입력 파일에 없는 멜론 표기(--extra-alias). 검색어로는 쓰지 않고 이름 대조에만 쓴다.
    names = [name, alias, *row.get("extra_aliases", [])]
    wanted = {aliases.canonical(t) for n in names if n for t in artist_tokens(n)}
    record: Dict = {"status": "search_failed", "queries": []}

    pick: Optional[ArtistPick] = None
    for query in [q for q in (name, alias) if q]:
        body = client.get(SEARCH_URL, {"q": query}, HEADERS, expect="goArtistDetail",
                          empty_markers=SEARCH_EMPTY_MARKERS)
        if body is None:
            record["status"] = "not_cached"
            return record
        record["queries"].append(query)
        pick = pick_artist(parse_artist_anchors(body), wanted, aliases)
        if pick is not None:
            break
        logger.warning("  검색 '%s': 이름이 맞는 아티스트 없음", query)
    if pick is None:
        return record

    params = {"startIndex": "1", "pageSize": "50", "listType": "A",
              "orderBy": "POPULAR_SONG_LIST", "artistId": pick.artist_id}
    headers = dict(HEADERS)
    headers.update({"Referer": f"https://www.melon.com/artist/song.htm?artistId={pick.artist_id}",
                    "X-Requested-With": "XMLHttpRequest"})
    body = client.get(SONG_LIST_URL, params, headers, expect="goSongDetail",
                      empty_markers=SONG_LIST_EMPTY_MARKERS)
    if body is None:
        record["status"] = "not_cached"
        return record
    songs = parse_song_rows(body)
    if not songs and not any(m in body for m in SONG_LIST_EMPTY_MARKERS):
        raise StopCrawl("곡 목록을 하나도 읽지 못함(구조 확인 필요)", _request_label(SONG_LIST_URL, params))

    record.update({
        "status": "done" if songs else "no_songs",
        "melon_artist_id": pick.artist_id,
        "melon_name": pick.name,
        "search_hits": pick.hits,
        "same_name_others": pick.others,
        "songs": songs[:top_n],
    })
    top3 = ", ".join(s["title"] for s in songs[:3]) or "-"
    logger.info("  → 멜론 '%s' (ID %s, 검색 %d회 등장) · 대표곡: %s", pick.name, pick.artist_id, pick.hits, top3)
    if pick.others:
        logger.warning("  [동명이인 확인] 이름이 맞는 다른 아티스트: %s",
                       "; ".join(f"{n} (ID {i}, {h}회)" for i, n, h in pick.others))
    return record


def candidates_from_record(artist: str, record: Dict, aliases: AliasMap) -> List[Song]:
    out: List[Song] = []
    target_id = record.get("melon_artist_id", "")
    target_tokens = {aliases.canonical(t) for t in artist_tokens(record.get("melon_name", ""))}
    for rank, s in enumerate(record.get("songs", [])):
        names = [n for _, n in s["artists"]]
        ids = {i for i, _ in s["artists"]}
        credited = (target_id in ids) if ids else bool(aliases.artist_set(", ".join(names)) & target_tokens)
        out.append(Song(artist=", ".join(names) or record.get("melon_name", artist), title=s["title"],
                        song_id=s["song_id"], source=artist, rank=rank, adult=bool(s.get("adult")),
                        credited=credited))
    return out


def crawler_running() -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", "src.crawler.main"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return bool(res.stdout.strip())


# =============================================================================
# CLI
# =============================================================================

def _setup_logging(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(cache_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="지정 아티스트 인기곡으로 추가 크롤링 목록을 만든다")
    p.add_argument("--artists", type=Path, default=Path("artists_manual.csv"))
    p.add_argument("--must-add", type=Path, default=Path("must_add_songs.csv"))
    p.add_argument("--skip-must-add", action="store_true", help="must_add 파일 없이 진행(시험 수집용)")
    p.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--first", type=int, default=None, help="앞에서 N명만")
    p.add_argument("--only", default="", help="쉼표로 구분한 아티스트 이름만")
    p.add_argument("--extra-alias", action="append", default=[], metavar="이름=멜론표기",
                   help="입력 파일 밖의 별칭을 더한다(여러 번 가능). 예: 미스에이=미쓰에이")
    p.add_argument("--offline", action="store_true", help="요청하지 않고 캐시·진행 파일만 쓴다")
    p.add_argument("--reparse", action="store_true",
                   help="진행 파일의 아티스트 결과를 무시하고 캐시 응답을 다시 파싱한다(캐시에 있으면 요청 없음)")
    p.add_argument("--max-requests", type=int, default=300, help="누적 요청 상한(진행 파일 기준)")
    p.add_argument("--run-budget", type=int, default=None, help="이번 실행의 요청 상한")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--cap", type=int, default=20)
    p.add_argument("--min-delay", type=float, default=3.0)
    p.add_argument("--max-delay", type=float, default=6.0)
    args = p.parse_args(argv)

    _setup_logging(args.cache_dir)
    if args.min_delay < 3.0 or args.max_delay < args.min_delay:
        logger.error("대기는 3초 이상이어야 한다(min=%.1f max=%.1f)", args.min_delay, args.max_delay)
        return 2
    if not args.offline and crawler_running():
        logger.error("크롤러(src.crawler.main)가 돌고 있다. 같은 IP라 시작하지 않는다.")
        return 2

    artist_rows = [r for r in read_csv_rows(args.artists) if r.get("artist")]
    extra: Dict[str, List[str]] = {}
    for item in args.extra_alias:
        key, _, value = item.partition("=")
        if not key.strip() or not value.strip():
            logger.error("--extra-alias는 '이름=멜론표기' 형식이어야 한다: %s", item)
            return 2
        extra.setdefault(key.strip(), []).append(value.strip())
    for row in artist_rows:
        row["extra_aliases"] = extra.get(row["artist"], [])
    aliases = AliasMap.from_rows(
        artist_rows + [{"artist": k, "alias": v} for k, vs in extra.items() for v in vs]
    )
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        artist_rows = [r for r in artist_rows if r["artist"] in wanted]
    if args.first is not None:
        artist_rows = artist_rows[: args.first]

    must_add: List[Song] = []
    if not args.skip_must_add:
        if not args.must_add.is_file():
            logger.error("must_add 파일이 없다: %s (시험 수집이면 --skip-must-add)", args.must_add)
            return 2
        must_add = load_must_add(args.must_add)
    existing = load_existing(args.existing)

    progress = Progress(args.cache_dir / "progress.json")
    client = MelonClient(args.cache_dir, progress, max_requests=args.max_requests, run_budget=args.run_budget,
                         delay=(args.min_delay, args.max_delay), offline=args.offline)
    logger.info("대상 %d명 · 기존 %d곡 · must_add %d곡 · 누적 요청 %d/%d%s",
                len(artist_rows), len(existing), len(must_add), progress.requests_made, args.max_requests,
                " · 오프라인" if args.offline else "")

    stopped: Optional[str] = None
    # 1 = 이상 응답으로 멈춤(재시도 금지), 3 = 요청 상한 도달(이어서 돌리면 된다)
    stop_code = 0
    for i, row in enumerate(artist_rows, 1):
        name = row["artist"]
        done = progress.artist(name)
        if not args.reparse and done and done.get("status") in ("done", "no_songs", "search_failed"):
            continue
        logger.info("[%d/%d] %s%s", i, len(artist_rows), name, f" (alias {row['alias']})" if row.get("alias") else "")
        try:
            record = collect_artist(client, row, aliases, args.top_n)
        except StopCrawl as e:
            progress.record_stop(e)
            stopped = f"[멈춤] {e.reason}\n  요청: {e.request}\n  아티스트: {name}"
            stop_code = 1
            break
        except BudgetReached as e:
            stopped = f"[상한] {e}. 아티스트 '{name}'부터 이어서 할 수 있다."
            stop_code = 3
            break
        if record["status"] != "not_cached":
            progress.set_artist(name, record)

    candidates: Dict[str, List[Song]] = {}
    failed: List[str] = []
    not_collected: List[str] = []
    for row in artist_rows:
        record = progress.artist(row["artist"])
        if not record or record.get("status") == "not_cached":
            not_collected.append(row["artist"])
        elif record.get("status") in ("search_failed", "no_songs"):
            failed.append(row["artist"])
        else:
            candidates[row["artist"]] = candidates_from_record(row["artist"], record, aliases)

    result = assemble(existing, candidates, must_add, aliases, cap=args.cap)
    # 일부 아티스트만 돌린 실행은 전체 목록을 덮어쓰지 않는다.
    out_dir = args.out_dir / "partial" if (args.only or args.first is not None) else args.out_dir
    new_path, exc_path = write_outputs(result, out_dir)
    for e in result.excluded:
        logger.info("  제외 [%s] %s%s", e.reason, _describe(e.song), f" ← {e.detail}" if e.detail else "")

    print("\n===== 요약 =====")
    for row in artist_rows:
        name = row["artist"]
        if name in candidates:
            rec = progress.artist(name) or {}
            print(f"  {name:<20} +{result.added_per_artist.get(name, 0):>2}곡  (멜론 '{rec.get('melon_name')}', 후보 {len(candidates[name])})")
    print(f"  must_add             +{result.must_add_added}곡")
    reasons: Dict[str, int] = {}
    for e in result.excluded:
        reasons[e.reason] = reasons.get(e.reason, 0) + 1
    print(f"전체 추가 {len(result.new_songs)}곡 · 제외 {len(result.excluded)}건 {reasons}")
    print(f"검색 실패: {', '.join(failed) if failed else '없음'}")
    if not_collected:
        print(f"미수집({len(not_collected)}명): {', '.join(not_collected)}")
    print(f"요청: 이번 실행 {client.requests_this_run}회 · 누적 {progress.requests_made}/{args.max_requests}회")
    print(f"출력: {new_path} · {exc_path} · 로그 {args.cache_dir / 'run.log'}")
    if stopped:
        print("\n" + stopped)
    return stop_code


if __name__ == "__main__":
    raise SystemExit(main())
