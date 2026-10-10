"""
[Start] Vague-Finder Crawler Orchestrator
설명: 4단계 파이프라인 (Melon -> YouTube -> Namuwiki -> LLM)을 통합 실행하여
      CSV에 있는 노래 리스트를 순차적으로 수집하고 저장합니다.
      최종 데이터는 ID 최상위 구조("id", "metadata")로 저장되며, 커버 이미지도 함께 다운로드합니다.
      [Update] 2026-02-14:
      - 로깅 강화 (File + Console)
      - 실패 추적 (failed_songs.csv)
      - 엄격한 검증 (이미지/오디오 누락 시 즉시 중단)
      [Update] 2026-09-18: 단계별 재시작
      - 곡마다 source -> assets -> analysis -> export 네 단계를 crawl_status.json에 기록한다.
        실패한 단계만 다시 실행하고, 원천 데이터(source.json)는 유지한 채 분석만 재시도한다.
      - 진행 중인 폴더는 <DATA_DIR>_incomplete에 두고, export가 끝나야 DATA_DIR로 옮긴다.
        임베딩·Mongo 단계는 완료 폴더만 본다.
      - all_songs.jsonl은 meta.json의 파생 파일이다. 시작할 때 누락을 채우고(sync),
        --rebuild-jsonl로 통째로 다시 만들 수 있다.
      - 멜론 접근 차단(403/429)은 배치를 멈춘다. 이어서 돌리면 전부 '검색 실패'로 기록된다.
작성자: 이연우 (Data Engineer), 황찬혁 (Full)
생성일: 2026-02-02
수정일: 2026-09-18
"""
import argparse
import os
import csv
import logging
import logging.handlers
import json
import requests
import re
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

import time
import random
from dotenv import load_dotenv

load_dotenv()

# Import Modules
from src.crawler.scripts_py.collect_melon_data import (
    AlreadyCollected,
    MelonAccessError,
    MelonTransientError,
    SongCandidate,
    collect_melon_data,
    fetch_melon_song_candidates,
)
from src.crawler.scripts_py.melon_match import EXACT_TITLE_BONUS, MatchScore, title_conflict
from src.crawler.scripts_py.collect_reaction import (
    download_youtube_audio,
    fetch_youtube_reaction,
    looks_like_m4a,
)
from src.crawler.scripts_py.refine_data import (
    blank_album_text_without_evidence,
    blank_comment_fields_without_evidence,
    refine_data,
)
from src.common.gemini_client import crawl_model_name
from src.crawler.scripts_py.collect_namuwiki_data import collect_namuwiki_data
from src.crawler.scripts_py.llm_utils import CommentSelectionUnavailable
from src.crawler.scripts_py.crawl_state import (
    _seed_key,
    AUDIO_FILE,
    RebuildRefused,
    COVER_FILE,
    META_FILE,
    PIPELINE_VERSION,
    CrawlState,
    append_jsonl,
    load_meta,
    read_jsonl_ids,
    read_source,
    rebuild_jsonl,
    remove_jsonl_record,
    replace_jsonl_record,
    scan_complete,
    scan_incomplete,
    seed_index,
    sync_jsonl,
    write_source,
)
from src.embedding.fixtures.meta_validation import LOW_COMMENT_COUNT, validate_meta_document

# 데이터 저장 경로
DATA_DIR = Path(os.environ.get("VAGUEFINDER_DATA_DIR", "data/raw"))
# 진행 중인 곡 폴더. 완료 폴더와 떨어져 있어야 임베딩 단계가 반쪽 폴더를 옮겨 버리지 않는다.
STAGING_DIR = Path(
    os.environ.get("VAGUEFINDER_STAGING_DIR")
    or DATA_DIR.with_name(DATA_DIR.name + "_incomplete")
)
LOG_DIR = Path("logs")
FAILED_LOG_FILE = Path("data/failed_songs.csv")
# 수집은 했지만 원곡이 아닐 수 있는 곡. 사람이 눈으로 확인할 목록이다.
REVIEW_LOG_FILE = Path("data/review_songs.csv")
JSONL_FILE = Path("data/all_songs.jsonl")

# Namuwiki 외부 맥락 수집 설정. 기본은 **꺼짐**이다.
#
# 곡 수집 경로에 붙어 있던 flat 나무위키 수집은 은퇴했다(#67). collect_namuwiki_data는 이제
# RuntimeError만 던지는 껍데기라, 켜 두면 곡마다 실패 경고가 한 줄씩 남을 뿐 아무것도 모으지
# 않는다. 나무위키는 별도 CLI로 분리돼 data/context에 쌓는다 — meta.json과 색인은 건드리지 않는다:
#   python -m src.crawler.scripts_py.collect_song_context --help
# 아래 NAMUWIKI_* 값과 Step 3 블록은 수집기가 되살아날 때를 위해 남겨 둔다.
ENABLE_NAMUWIKI = os.environ.get("ENABLE_NAMUWIKI", "0").strip().lower() not in {"0", "false", "no", "off"}
NAMUWIKI_CANDIDATE_LIMIT = int(os.environ.get("NAMUWIKI_CANDIDATE_LIMIT", "4"))
NAMUWIKI_QUERY_LIMIT = int(os.environ.get("NAMUWIKI_QUERY_LIMIT", "3"))
NAMUWIKI_MIN_DELAY = float(os.environ.get("NAMUWIKI_MIN_DELAY", "2.0"))
NAMUWIKI_MAX_DELAY = float(os.environ.get("NAMUWIKI_MAX_DELAY", "5.0"))
NAMUWIKI_TIMEOUT = float(os.environ.get("NAMUWIKI_TIMEOUT", "15.0"))

# 이보다 작은 audio.m4a는 곡 전체가 아니다. 128kbps AAC 30초가 약 480KB다.
MIN_AUDIO_BYTES = 100_000
# 헤더의 재생 시간이 이보다 짧으면 잘린 파일이나 예고편으로 본다.
MIN_AUDIO_SECONDS = 30.0

# 403/429처럼 이후 요청도 중단해야 하는 접근 차단이 발생하면 현재 실행에서만 비활성화한다.
_namuwiki_disabled_for_run = False


def reset_namuwiki_state() -> None:
    global _namuwiki_disabled_for_run
    _namuwiki_disabled_for_run = False

def setup_logging():
    """로깅 설정: 콘솔(INFO) + 파일(INFO, ERROR)"""
    LOG_DIR.mkdir(exist_ok=True)

    # Root Logger 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [] # 기존 핸들러 제거

    # 포맷터
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 1. 전체 로그 파일 (Rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "crawler.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. 에러 로그 파일 (Rotating)
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "crawler_error.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # 3. 콘솔 출력
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def load_songs(csv_path: str) -> List[Dict]:
    """CSV 파일에서 수집 대상 노래 리스트를 읽어옴"""
    songs = []
    if not os.path.exists(csv_path):
        logger.error(f"파일을 찾을 수 없습니다: {csv_path}")
        return songs

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("artist") and row.get("title"):
                songs.append(row)
    return songs

def save_failed_song(artist: str, title: str, reason: str):
    """실패한 곡 정보를 CSV에 기록"""
    file_exists = FAILED_LOG_FILE.exists()

    # data 폴더가 없으면 생성
    FAILED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(FAILED_LOG_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "artist", "title", "reason"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                artist,
                title,
                reason
            ])
        logger.warning(f"실패 기록됨: {artist} - {title} ({reason})")
    except Exception as e:
        logger.error(f"실패 로그 저장 중 에러: {e}")

def save_review_song(artist: str, title: str, picked: str, reason: str):
    """수집은 했지만 원곡이 아닐 수 있는 곡을 검토 목록에 남긴다.

    실패로 처리해 버리면 곡을 통째로 잃는다. 라이브 음원이 유일한 릴리스인 곡도
    있어서(예: 김광석 'Live Op.4 Concert Project'), 채택은 하되 기록을 남긴다.
    """
    file_exists = REVIEW_LOG_FILE.exists()
    REVIEW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(REVIEW_LOG_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "artist", "title", "picked", "reason"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                artist,
                title,
                picked,
                reason,
            ])
        logger.warning(f"[검토 필요] {artist} - {title} -> '{picked}' ({reason})")
    except Exception as e:
        logger.error(f"검토 로그 저장 중 에러: {e}")


def clean_filename(text: str) -> str:
    """파일명으로 사용할 수 없는 문자 제거 및 공백을 언더바(_)로 변경"""
    # 1. 비허용 문자 제거/대체
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    # 2. 공백 -> _
    text = re.sub(r'\s+', "_", text)
    return text.strip()


# --- 에셋 내용 검증 -------------------------------------------------------------------
# HTTP 200과 파일 크기만 보면 HTML 오류 본문도 cover.jpg로 '저장 성공'한다.

def verify_cover(path: Path) -> str:
    """이미지로 디코딩되는지 본다. 문제면 사유, 정상이면 ''."""
    if not path.is_file() or path.stat().st_size == 0:
        return "파일 없음"
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow는 임베딩 의존성이라 보통 있다
        return ""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
    except Exception as e:
        return f"이미지 디코딩 실패: {type(e).__name__}"
    if width < 50 or height < 50:
        return f"이미지가 너무 작음 ({width}x{height})"
    return ""


def m4a_duration_seconds(path: Path) -> Optional[float]:
    """MP4 헤더의 mvhd 상자에서 재생 시간을 읽는다. 없으면 None."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    pos = data.find(b"mvhd")
    if pos < 0 or pos + 4 >= len(data):
        return None
    # 'mvhd' 뒤: version(1) flags(3) ctime mtime timescale duration. v1은 시간 필드가 8바이트다.
    version = data[pos + 4]
    try:
        if version == 1:
            timescale, duration = struct.unpack(">IQ", data[pos + 24:pos + 36])
        else:
            timescale, duration = struct.unpack(">II", data[pos + 16:pos + 24])
    except struct.error:
        return None
    if not timescale:
        return None
    return duration / timescale


def verify_audio(path: Path) -> str:
    """audio.m4a가 곡 전체를 담은 재생 가능한 파일인지 본다. 문제면 사유, 정상이면 ''."""
    if not path.is_file() or path.stat().st_size == 0:
        return "파일 없음"
    if not looks_like_m4a(path):
        return "m4a 헤더 아님"
    if path.stat().st_size < MIN_AUDIO_BYTES:
        return f"파일이 너무 작음 ({path.stat().st_size} bytes)"
    duration = m4a_duration_seconds(path)
    if duration is not None and duration < MIN_AUDIO_SECONDS:
        return f"재생 시간이 너무 짧음 ({duration:.0f}s)"
    return ""


def _download_cover(cover_url: str, save_dir: Path) -> bool:
    """커버 이미지를 내려받아 cover.jpg로 저장. 성공 여부를 반환함"""
    if not cover_url:
        logger.warning("커버 이미지 URL 없음")
        return False

    cover_path = save_dir / COVER_FILE
    if not verify_cover(cover_path):
        logger.info("커버 이미지가 이미 있음: cover.jpg")
        return True

    try:
        # 멜론 등은 헤더 필요할거 대비해서
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(cover_url, headers=headers, timeout=10)
    except Exception as e:
        logger.warning(f"이미지 다운로드 실패: {e}")
        return False

    if res.status_code != 200:
        logger.warning(f"이미지 다운로드 실패: HTTP {res.status_code}")
        return False

    content_type = res.headers.get("Content-Type", "")
    if content_type and not content_type.lower().startswith("image/"):
        logger.warning(f"이미지 다운로드 실패: Content-Type {content_type}")
        return False

    try:
        with open(cover_path, "wb") as f:
            f.write(res.content)
    except Exception as e:
        logger.warning(f"이미지 저장 실패: {e}")
        return False

    problem = verify_cover(cover_path)
    if problem:
        logger.warning(f"이미지 검증 실패: {problem}")
        cover_path.unlink(missing_ok=True)
        return False

    logger.info("커버 이미지 저장 완료: cover.jpg")
    return True


def _download_audio(youtube_url: str, save_dir: Path) -> bool:
    """유튜브 오디오를 audio.m4a로 내려받는다. 성공 여부를 반환한다."""
    if not youtube_url:
        logger.warning("유튜브 URL 없음")
        return False

    audio_path = save_dir / AUDIO_FILE
    if not verify_audio(audio_path):
        logger.info("오디오가 이미 있음: audio.m4a")
        return True

    try:
        ok = download_youtube_audio(youtube_url, str(audio_path))
    except Exception as e:
        logger.warning(f"오디오 다운로드 호출 중 에러: {e}")
        return False

    if not ok:
        logger.warning("오디오 다운로드 실패(yt-dlp)")
        return False

    problem = verify_audio(audio_path)
    if problem:
        logger.warning(f"오디오 검증 실패: {problem}")
        audio_path.unlink(missing_ok=True)
        return False

    return True


def get_popularity_tag(view_count: int) -> str:
    """조회수에 따른 인기도 태그 반환"""
    if view_count >= 100_000_000:
        return "Mega Hit"
    elif view_count >= 10_000_000:
        return "Famous"
    elif view_count >= 1_000_000:
        return "Moderate"
    else:
        return "Hidden"


# --- 수집 현황 ------------------------------------------------------------------------

class Registry:
    """이번 실행이 아는 곡들. 완료 ID, JSONL에 있는 ID, 재개 대상."""

    def __init__(self, data_dir: Path = None, staging_dir: Path = None, jsonl_path: Path = None,
                 complete: Optional[Dict[str, Path]] = None,
                 jsonl_ids: Optional[Set[str]] = None):
        self.data_dir = data_dir or DATA_DIR
        self.staging_dir = staging_dir or STAGING_DIR
        self.jsonl_path = jsonl_path or JSONL_FILE
        # 시작할 때 폴더 목록과 JSONL을 이미 읽었으면 그것을 쓴다(같은 일을 두 번 하지 않는다).
        self.complete_ids: Set[str] = set(
            scan_complete(self.data_dir) if complete is None else complete
        )
        self.jsonl_ids: Set[str] = (
            read_jsonl_ids(self.jsonl_path) if jsonl_ids is None else set(jsonl_ids)
        )
        self.incomplete: Dict[str, CrawlState] = {
            s.song_id: s for s in scan_incomplete(self.staging_dir)
        }
        # 수집 당시 요청 정보로 만든 색인. 검색 전에 완료곡을 건너뛰는 데 쓴다.
        self.seed_index: Dict[Tuple[str, str], str] = seed_index(self.jsonl_path, self.complete_ids)
        self.attempted_this_run: Set[str] = set()
        # 방금 처리한 곡에서 외부 요청이 나갔는지. 나가지 않았으면 호출부가 대기하지 않는다.
        self.made_requests: bool = False
        logger.info(f"[SKIP] 이미 수집된 곡 수: {len(self.complete_ids)}곡")
        if self.incomplete:
            logger.info(f"재개 대상 미완료 폴더: {len(self.incomplete)}개 ({self.staging_dir})")

    def is_complete(self, song_id: str) -> bool:
        return str(song_id) in self.complete_ids

    def completed_song_for(self, artist: str, title: str) -> Optional[str]:
        """이 시드로 이미 수집한 곡의 ID. 없으면 None."""
        return self.seed_index.get((_seed_key(artist), _seed_key(title)))


def load_collected_ids() -> set:
    """시작할 때 DATA_DIR을 확인하여 이미 수집이 완료된 곡의 Melon ID를 불러온다.

    meta.json, cover.jpg, audio.m4a가 모두 존재하고 파일 크기가 0보다 큰 경우에만
    수집 완료곡으로 판단한다.
    """
    return set(scan_complete(DATA_DIR))


def is_already_crawled(melon_id: str, collected_ids: set) -> bool:
    return str(melon_id) in collected_ids


# --- 1단계: 원천 수집 -------------------------------------------------------------------

# --- 수집 정책 ---------------------------------------------------------------------------
# 가사 없는 곡은 색인에 들어갈 수 없다(임베딩 전 검증이 빈 full_lyrics를 거부한다). 원인은
# 대부분 19금 곡이다 — 성인 인증이 필요해 멜론이 가사 영역을 아예 주지 않는다.
#
# 이 검사는 **한 곳에 모아** 수집·재개·내보내기가 모두 거치게 한다. 예전에는 collect_source
# 안에만 있어서, 정책 이전에 만들어진 미완료 폴더를 재개하면 검사 없이 통과해 JSONL까지 갔다.

POLICY_NO_LYRICS = "No Lyrics (가사 미제공 - 19금·연주곡 등)"
POLICY_ADULT = "Adult Only (19금 - 가사 수집 불가)"


def source_policy_error(source: Dict) -> str:
    """source.json이 정책에 걸리는 이유. 통과면 ''."""
    if (source.get("melon") or {}).get("is_adult"):
        return POLICY_ADULT
    if not (source.get("lyrics") or "").strip():
        return POLICY_NO_LYRICS
    return ""


def record_policy_error(record: Dict) -> str:
    """meta.json이 정책에 걸리는 이유. 통과면 ''."""
    if not ((record.get("lyrics_data") or {}).get("full_lyrics") or "").strip():
        return POLICY_NO_LYRICS
    return ""


def staged_policy_error(state: CrawlState, source: Optional[Dict] = None) -> str:
    """진행 중인 폴더가 정책에 걸리는 이유. 통과면 ''.

    source.json과 meta.json 중 있는 것을 본다. 재개든 내보내기든 이 함수를 거친다.
    이미 읽어 둔 source가 있으면 그것을 쓴다(같은 파일을 두 번 읽지 않는다).
    """
    if source is None:
        source = read_source(state.folder)
    if source is not None:
        error = source_policy_error(source)
        if error:
            return error
    record = load_meta(state.folder) if (state.folder / META_FILE).is_file() else None
    if record is not None:
        return record_policy_error(record)
    return ""


def discard_staged(state: CrawlState, registry: "Registry", reason: str) -> bool:
    """정책에 걸린 미완료 곡을 버린다. 다시 수집해도 같은 이유로 걸리므로 재시도하지 않는다.

    **JSONL을 먼저 정리한다.** 예전 실행에서 JSONL 기록은 성공하고 폴더 이동만 실패한 곡이
    있으면, 폴더만 지웠을 때 그 행이 남아 감사·평가·카탈로그에 제외 대상이 계속 전달된다.
    시작 시 동기화는 그 행을 고아로만 집계하고 지우지 않는다.

    정리가 실패하면 폴더를 **보존해서** 다음 실행에 다시 시도할 수 있게 한다. 폴더를 먼저
    지우면 재시도할 근거가 사라진다.

    Returns:
        True면 정리와 폐기를 끝냈다. False면 JSONL 정리에 실패해 폴더를 남겼다.
    """
    logger.warning("수집 정책 위반: %s (%s)", state.folder.name, reason)
    try:
        removed = remove_jsonl_record(registry.jsonl_path, state.song_id)
    except Exception as e:
        logger.error(
            "정책 제외 곡의 JSONL 행을 지우지 못했습니다(폴더는 남겨 둡니다): %s (%s)",
            state.song_id, e,
        )
        state.mark_failed(state.next_stage() or "export", f"Policy cleanup failed: {e}")
        save_failed_song(state.input_artist, state.input_title,
                         f"[policy] {reason} / JSONL 정리 실패: {e}")
        return False

    if removed:
        logger.warning("JSONL에서 제외 대상 %d줄 삭제: %s", removed, state.song_id)
    registry.jsonl_ids.discard(state.song_id)
    registry.complete_ids.discard(state.song_id)
    registry.incomplete.pop(state.song_id, None)

    save_failed_song(state.input_artist, state.input_title, f"[policy] {reason}")
    shutil.rmtree(state.folder, ignore_errors=True)
    return True


def collect_melon_source(artist: str, title: str, candidates,
                         already_collected=None) -> Tuple[Optional[Dict], Optional[CrawlState], str]:
    """원천 수집의 앞쪽(멜론)만 한다. 성공하면 곧바로 저장할 수 있는 형태로 돌려준다.

    유튜브 단계에서 실패해도 멜론 상세·댓글 수집과 댓글 LLM 선별을 다시 하지 않기 위해,
    여기까지의 결과를 source.json에 먼저 남긴다. 나머지는 stage_source가 이어서 채운다.

    Returns:
        (source, state, 실패 사유). 실패면 source와 state가 None이다.
    """
    # Step 1: Melon Crawling (메타 + 가사 + 멜론댓글)
    melon_data = collect_melon_data(artist, title, candidates=candidates,
                                    already_collected=already_collected)
    if not melon_data:
        return None, None, "Melon Validation Failed"

    # 채택 근거는 레코드 본문에 넣지 않고 crawl_status와 로그/검토 목록에 쓴다.
    audit = melon_data.pop("match_audit", {})
    warnings: List[str] = []
    if audit.get("needs_review"):
        note = audit.get("note") or ", ".join(audit.get("reasons", [])) or "완전일치 아님"
        save_review_song(artist, title, melon_data.get("title", ""), note)
        warnings.append(f"melon: {note}")

    melon_id = melon_data.get("id")
    if not melon_id:
        return None, None, "Melon ID Not Found"

    # [Validation] Image Check
    if not melon_data.get("cover_url"):
        return None, None, "No Image (Cover URL)"

    # 후보 단계(collect_melon_data)가 이미 19금·가사 없음을 거르지만, 여기서 한 번 더 본다.
    lyrics = melon_data.get("lyrics", "")
    policy = source_policy_error({"melon": melon_data, "lyrics": lyrics})
    if policy:
        return None, None, policy

    artist_raw = melon_data.get("artist", "Unknown")
    if isinstance(artist_raw, list):
        artist_raw = artist_raw[0] if artist_raw else "Unknown"
    folder_name = f"{clean_filename(artist_raw)}_{clean_filename(melon_data.get('title', 'Unknown'))}_{melon_id}"

    state = CrawlState(
        song_id=str(melon_id),
        input_artist=artist,
        input_title=title,
        folder=STAGING_DIR / folder_name,
        selection={
            "melon": {
                "song_id": str(melon_id),
                "title": melon_data.get("title", ""),
                "album": melon_data.get("album_name", ""),
                "score": audit.get("score"),
                "reasons": list(audit.get("reasons", [])),
                "rank": audit.get("rank"),
                "candidate_count": audit.get("candidate_count"),
            },
        },
        comment_counts={"melon": len(melon_data.get("melon_comments") or [])},
        warnings=warnings,
    )
    source = {
        "melon": melon_data,
        "lyrics": lyrics,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    return source, state, ""


def stage_source(state: CrawlState, source: Dict) -> str:
    """원천 수집의 뒤쪽(유튜브·나무위키)을 채운다. 실패 사유를 돌려주고 성공이면 ''.

    이미 채워져 있으면 아무것도 하지 않는다. 멜론 쪽은 collect_melon_source가 저장해 두므로
    유튜브 실패 뒤 재시도에서 멜론 요청과 댓글 LLM 선별을 다시 하지 않는다.
    """
    if source.get("reaction"):
        return ""

    artist, title = state.input_artist, state.input_title
    melon_data = source.get("melon") or {}
    lyrics = source.get("lyrics", "")

    # Step 2: YouTube Reaction (영상URL + 유튜브댓글 + 조회수)
    #         가수 이름은 시드와 멜론 표기를 모두 넘겨 다른 가수의 동명곡을 거른다.
    try:
        reaction_data = fetch_youtube_reaction(
            artist, title, song_lyrics=lyrics,
            melon_title=melon_data.get("title", ""),
            known_artists=list(melon_data.get("artist") or []),
        )
    except CommentSelectionUnavailable as e:
        # 판정 장애는 '댓글 0개'가 아니다. 단계 실패로 남겨 다음 실행에서 다시 시도한다.
        # 멜론 쪽 결과는 이미 저장돼 있어 다시 받지 않는다.
        return f"Comment Selection Unavailable (retry): {e}"

    # [Validation] Audio Check
    video_url = reaction_data.get("video_url", "") if reaction_data else ""
    video_title = reaction_data.get("video_title", "") if reaction_data else ""
    if not video_url:
        return "No Audio Source (YouTube URL)"

    # 멜론(가사·메타)과 유튜브(오디오)가 같은 녹음인지 대조.
    # 어긋나면 저장하지 않는다. 기록만 남기고 저장하면 텍스트 임베딩과 오디오
    # 임베딩이 다른 녹음을 가리키는 레코드가 그대로 쌓인다(32591630 사례).
    conflict = title_conflict(melon_data.get("title", ""), video_title,
                              list(melon_data.get("artist") or []))
    if conflict:
        return (
            f"Source Mismatch: {conflict} (멜론 '{melon_data.get('title','')}' / 영상 '{video_title}')"
        )

    if reaction_data.get("artist_verified") is False:
        note = f"영상 제목·채널에 가수 이름 없음 (영상 '{video_title}')"
        save_review_song(artist, title, melon_data.get("title", ""), note)
        state.warn(f"youtube: {note}")

    # 댓글 수는 채택 여부를 바꾸지 않는다. 임베딩 전 검증도 이 수로 곡을 떨어뜨리지 않는다
    # (meta_validation.LOW_COMMENT_COUNT 주석). 대신 사람이 볼 기록을 남긴다.
    melon_comment_count = len(melon_data.get("melon_comments") or [])
    youtube_comment_count = len(reaction_data.get("comments") or [])
    state.comment_counts = {"melon": melon_comment_count, "youtube": youtube_comment_count}
    comment_total = melon_comment_count + youtube_comment_count
    if comment_total == 0:
        note = "댓글 0개 (반응 요약·팬 태그는 비워 저장)"
        save_review_song(artist, title, melon_data.get("title", ""), note)
        state.warn(f"comments: {note}")
    elif comment_total < LOW_COMMENT_COUNT:
        state.warn(
            f"comments: 합계 {comment_total}개 (멜론 {melon_comment_count}, 유튜브 {youtube_comment_count})"
        )

    # Step 3: Namuwiki External Context (후보 매칭 + usable context 판정)
    global _namuwiki_disabled_for_run
    namuwiki_data = {}

    if ENABLE_NAMUWIKI and not _namuwiki_disabled_for_run:
        try:
            namuwiki_data, should_stop_namuwiki = collect_namuwiki_data(
                artist=artist,
                title=title,
                album=melon_data.get("album_name", ""),
                release_date=melon_data.get("release_date", ""),
                song_id=state.song_id,
                candidate_limit=NAMUWIKI_CANDIDATE_LIMIT,
                query_limit=NAMUWIKI_QUERY_LIMIT,
                min_delay=NAMUWIKI_MIN_DELAY,
                max_delay=NAMUWIKI_MAX_DELAY,
                timeout=NAMUWIKI_TIMEOUT,
            )

            logger.info(
                "Namuwiki 결과: status=%s usable=%s page=%s",
                namuwiki_data.get("match_status", ""),
                namuwiki_data.get("context_usable", ""),
                namuwiki_data.get("page_title", ""),
            )

            if should_stop_namuwiki:
                _namuwiki_disabled_for_run = True
                logger.warning(
                    "Namuwiki 접근 차단/레이트리밋 감지 -> 현재 크롤러 실행에서는 이후 Namuwiki 단계를 비활성화합니다."
                )
        except Exception as e:
            # 외부 소스 실패 때문에 기존 Melon/YouTube 크롤링 전체가 실패하면 안 된다.
            logger.warning("Namuwiki 수집 실패 -> 외부 맥락 없이 계속 진행: %s", e)
            namuwiki_data = {}

    source["reaction"] = reaction_data
    source["namuwiki"] = namuwiki_data or {}
    state.selection["youtube"] = {
        "video_url": video_url,
        "video_title": video_title,
        "view_count": reaction_data.get("view_count"),
        "artist_verified": bool(reaction_data.get("artist_verified", True)),
        "comment_source_url": reaction_data.get("comment_source_url", ""),
    }
    write_source(state.folder, source)
    state.save()
    return ""


# --- 2·3·4단계 ----------------------------------------------------------------------------

def stage_assets(state: CrawlState, source: Dict) -> str:
    """cover.jpg, audio.m4a. 실패 사유를 돌려주고 성공이면 ''."""
    melon = source.get("melon") or {}
    reaction = source.get("reaction") or {}
    if not _download_cover(melon.get("cover_url", ""), state.folder):
        return "Cover Download Failed"
    if not _download_audio(reaction.get("video_url", ""), state.folder):
        return "Audio Download Failed"
    return ""


def build_record(source: Dict, final_meta: Dict, state: CrawlState) -> Dict:
    """meta.json 본문. dense/sparse 임베딩 입력은 임베딩 단계(passage_builder)가
    meta.json에서 직접 조립하므로, 크롤러는 원천 데이터만 저장한다."""
    melon_data = source.get("melon") or {}
    reaction_data = source.get("reaction") or {}
    namuwiki_data = source.get("namuwiki") or {}
    lyrics = source.get("lyrics", "")

    # 댓글이 하나도 없으면 반응 요약·팬 태그를 비운다. refine_data도 같은 함수를 부르지만,
    # 레코드 불변식은 조립하는 여기서 지켜야 분석기를 바꿔도 검증기와 어긋나지 않는다.
    # 앨범 텍스트가 실패 문구면 비운다. 레코드 불변식은 조립하는 여기서 지켜야 분석기를 바꿔도
    # 임베딩 전 검증과 어긋나지 않는다(댓글 필드와 같은 이유).
    final_meta = blank_album_text_without_evidence(dict(final_meta))
    final_meta = blank_comment_fields_without_evidence(
        dict(final_meta),
        melon_data.get("melon_comments") or [],
        reaction_data.get("comments") or [],
    )
    melon_id = state.song_id
    video_url = reaction_data.get("video_url", "")
    video_title = reaction_data.get("video_title", "")
    view_count = reaction_data.get("view_count") or 0

    # 띄어쓰기(공백)를 강제로 모두 제거하는 헬퍼 함수
    def no_space(tag_list):
        if not tag_list: return []
        if isinstance(tag_list, str):  # 배열 자리에 문자열이 오면 글자 단위로 쪼개지 않는다
            tag_list = [tag_list]
        return [str(t).replace(" ", "") for t in tag_list if t]

    genre_str = melon_data.get("genre", "")
    genre_list = [g.strip() for g in genre_str.split(",")] if genre_str else []

    ordered_data = {
        "song_id": str(melon_id),
        "metadata": {
            "title": melon_data.get("title"),
            "artist": melon_data.get("artist", []),
            "album": melon_data.get("album_name", ""),
            "release_date": melon_data.get("release_date", ""),
            "genre": genre_list,
            "type": no_space(final_meta.get("artist_type", ["unknown"])),
            "vocal_gender": final_meta.get("vocal_gender", "unknown"),
            "album_description": melon_data.get("album_desc", ""),
            "album_summary": final_meta.get("album_summary", "")
        },
        "lyrics_data": {
            "full_lyrics": lyrics,
            "lyrics_highlight": final_meta.get("lyrics_highlight", ""),
            "lyrics_summary": final_meta.get("lyrics_summary", "")
        },
        "semantic_analysis": {
            "search_style_summary": final_meta.get("search_style_summary", ""),
            "mood_tags": no_space(final_meta.get("mood_tags", [])),
            "time_weather_tags": no_space(final_meta.get("time_weather_tags", [])),
            "place_activity_tags": no_space(final_meta.get("place_activity_tags", [])),
            "emotion_tags": no_space(final_meta.get("emotion_tags", [])),
            "vibe_tags": no_space(final_meta.get("vibe_tags", [])),
            "relation_context_tags": no_space(final_meta.get("relation_context_tags", [])),
            "color_tags": no_space(final_meta.get("color_tags", [])),
            "sound_tags": no_space(final_meta.get("sound_tags", [])),
            "melon_playlist_tags": no_space(melon_data.get("melon_playlist_tags", [])),
            "visual_imagery": final_meta.get('visual_imagery', [])
        },
        "community_feedback": {
            "sentiment_summary": final_meta.get("sentiment_summary", ""),
            "fans_tags": no_space(final_meta.get("fans_tags", [])),
            "major_emotion": final_meta.get("major_emotion", ""),
            "popularity": {
                "youtube_view_count": view_count,
                "fame": get_popularity_tag(view_count)
            }
        },
        "links": {
            "melon_url": f"https://www.melon.com/song/detail.htm?songId={melon_id}",
            "youtube_url": video_url,
            "youtube_title": video_title,
            "cover_url": melon_data.get("cover_url", "")
        },
        "comments": {
            "melon": melon_data.get("melon_comments", []),
            "youtube": reaction_data.get("comments", [])
        },
        # 수집 이력: 요청한 가수·제목, 고른 ID와 근거, 규칙 버전. 재수집과 품질 비교에 쓴다.
        "crawl_status": state.to_meta_block(),
    }

    # Namuwiki 원문 전체(context_excerpt)는 저장하지 않고,
    # 매칭 감사 정보 + LLM이 구조화한 사실만 저장한다.
    # 이렇게 해야 원문 대량 적재에 따른 오염/라이선스 부담을 줄일 수 있다.
    if namuwiki_data:
        ordered_data["external_context"] = {
            "namuwiki": {
                "match_status": namuwiki_data.get("match_status", ""),
                "match_type": namuwiki_data.get("match_type", ""),
                "page_title": namuwiki_data.get("page_title", ""),
                "source_url": namuwiki_data.get("page_url", ""),
                "context_usable": namuwiki_data.get("context_usable", ""),
                "context_tags": final_meta.get("context_tags", []),
                "fact_summary": final_meta.get("fact_summary", ""),
            }
        }
    return ordered_data


def stage_analysis(state: CrawlState, source: Dict) -> str:
    """Gemini 정제 -> meta.json. 실패 사유를 돌려주고 성공이면 ''.

    실패해도 source.json과 에셋은 그대로 둔다. 다음 실행이 이 단계만 다시 한다.
    """
    melon_data = dict(source.get("melon") or {})
    final_meta = refine_data(
        melon_data,
        source.get("lyrics", ""),
        source.get("reaction") or {},
    )
    if not final_meta:
        return "LLM Analysis Failed"

    record = build_record(source, final_meta, state)
    # 정제 모델을 수집 이력에 남긴다. 정제 태그(mood_tags·emotion_tags 등)는 색인·검색·리랭커 입력이라, 모델이 바뀌면
    # 그 뒤 정제한 곡만 다른 모델의 태그를 갖는다 — 곡마다 가려낼 수 있어야 한다(PR #31 리뷰).
    # 이 필드가 없는 레코드는 2026-10-09 이전 정제(gemini-3.1-flash-lite 이하)다. refine_data와 같은 프로세스·같은 env라
    # 방금 쓴 모델과 같다. build_record는 이 단계에서만 불린다(재색인은 meta.json을 그대로 쓴다).
    record["crawl_status"]["refine_model"] = crawl_model_name()
    record = note_validation_gap(state, record)

    meta_path = state.folder / META_FILE
    tmp = meta_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    tmp.replace(meta_path)
    logger.info(f"저장 완료: {meta_path}")
    return ""


def note_validation_gap(state: CrawlState, record: Dict) -> Dict:
    """이 레코드가 임베딩 전 검증에서 탈락할지 미리 보고 기록한다. 기록을 반영한 레코드를 돌려준다.

    수집은 done인데 임베딩에서 빠지는 어긋남을 **수집 시점에** 드러내려는 것이다. 지금 아는
    사례는 가사가 없는 곡(full_lyrics 빈 문자열, 코퍼스 961곡 중 8곡)이다. 채택 여부는 바꾸지
    않는다 — 가사 없는 곡을 버릴지는 색인 정책이라 사람이 정할 일이고, 조용히 버리면 곡을 잃는다.
    """
    issues = validate_meta_document(record, song_dir=state.folder, require_media_files=True)
    if not issues:
        return record

    paths = sorted({f"{issue.path}({issue.reason})" for issue in issues})
    note = "임베딩 검증 탈락 예상: " + ", ".join(paths[:4])
    logger.warning("%s (%s)", note, state.folder.name)
    state.warn(note)
    state.save()
    save_review_song(state.input_artist, state.input_title,
                     (record.get("metadata") or {}).get("title", ""), note)
    # 메모는 레코드에도 남긴다. crawl_status.json만 보면 임베딩·감사 단계에서 보이지 않는다.
    record.setdefault("crawl_status", {})["notes"] = list(state.warnings)
    return record


def stage_export(state: CrawlState, registry: Registry) -> str:
    """JSONL에 현재 meta.json을 기록하고 폴더를 완료 폴더로 옮긴다. 실패 사유, 성공이면 ''.

    JSONL을 먼저 쓰고 옮긴다. 쓰기 뒤에 죽으면 다음 실행이 같은 레코드로 덮어쓰고 옮기기만 한다.
    옮긴 뒤 JSONL이 빠지는 경우는 시작 시 sync가 채운다.

    같은 ID가 이미 있으면 **건너뛰지 않고 바꾼다.** 재수집한 곡은 meta.json이 새 분석인데
    JSONL에는 예전 분석이 남아, 두 파일이 어긋난 채로 굳었다. 파일 전체를 다시 쓰는 경로지만
    ID가 이미 있을 때만 타므로(새 곡은 덧붙인다) 정상 수집에서는 비용이 붙지 않는다.
    """
    record = load_meta(state.folder)
    if record is None:
        return "meta.json Missing"
    policy = record_policy_error(record)
    if policy:
        # 여기까지 온 레코드는 정책 이전에 만들어진 것이다. JSONL에 넣지 않는다.
        return f"Policy: {policy}"
    try:
        if state.song_id in registry.jsonl_ids:
            # 이번 실행에서 이미 썼든 예전 실행이 썼든, 지금 meta.json으로 덮어쓴다.
            # '이번 실행에서 썼으니 건너뛴다'로 판단하면, JSONL을 쓴 뒤 이동이 실패해
            # 재시도하는 곡의 바뀐 meta.json이 JSONL에 반영되지 않는다.
            replaced = replace_jsonl_record(registry.jsonl_path, record)
            logger.info("JSONL %s: %s", "기존 줄 교체" if replaced else "덧붙임", state.song_id)
        else:
            append_jsonl(registry.jsonl_path, record)
    except Exception as e:
        return f"JSONL Write Failed: {e}"
    registry.jsonl_ids.add(state.song_id)

    dest = registry.data_dir / state.folder.name
    try:
        registry.data_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            # 예전 실행이 남긴 반쪽 폴더. 완료 판정을 통과하지 못했으니 덮어쓴다.
            shutil.rmtree(dest)
        shutil.move(str(state.folder), str(dest))
    except Exception as e:
        return f"Folder Move Failed: {e}"
    state.folder = dest
    return ""


def advance(state: CrawlState, registry: Registry, source: Optional[Dict] = None) -> bool:
    """남은 단계를 순서대로 실행한다. 한 단계라도 실패하면 거기서 멈추고 False.

    단계를 돌리기 전에 정책 검사를 한다. 재개 경로는 collect_source를 타지 않으므로, 여기가
    아니면 정책 이전에 만들어진 폴더가 검사 없이 JSONL까지 간다.
    """
    # source.json은 한 번만 읽어 정책 검사와 단계 실행이 나눠 쓴다.
    if source is None:
        source = read_source(state.folder)

    policy = staged_policy_error(state, source)
    if policy:
        discard_staged(state, registry, policy)
        return False

    stage = state.next_stage()
    while stage is not None:
        if stage in ("source", "assets", "analysis") and source is None:
            state.mark_failed(stage, "source.json Missing")
            save_failed_song(state.input_artist, state.input_title, "source.json Missing")
            return False

        # 단계 함수가 사유 문자열 대신 예외를 던져도 여기서 실패로 기록한다. 예전에는 예외가
        # resume_incomplete 밖으로 빠져나가 곡 하나 때문에 남은 재개 작업까지 멈췄고,
        # 실패 횟수도 늘지 않아 같은 단계에서 영원히 멈출 수 있었다.
        try:
            if stage == "assets":
                error = stage_assets(state, source)
            elif stage == "analysis":
                error = stage_analysis(state, source)
            elif stage == "export":
                error = stage_export(state, registry)
            else:  # source: 멜론은 이미 저장돼 있고, 유튜브·나무위키를 이어서 채운다
                error = stage_source(state, source)
        except Exception as e:
            logger.error("단계 실행 중 예외 (%s [%s]): %s", state.folder.name, stage, e, exc_info=True)
            error = f"Exception: {e}"

        if error:
            state.mark_failed(stage, error)
            save_failed_song(state.input_artist, state.input_title, f"[{stage}] {error}")
            logger.warning(f"단계 실패 -> 다음 실행에서 재개: {state.folder.name} [{stage}] {error}")
            return False

        state.mark_done(stage)
        logger.info(f"단계 완료: {state.folder.name} [{stage}]")
        stage = state.next_stage()

    registry.complete_ids.add(state.song_id)
    registry.incomplete.pop(state.song_id, None)
    return True


def resume_incomplete(registry: Registry) -> Tuple[int, int]:
    """시작할 때 격리 폴더의 미완료 곡을 이어서 진행한다. (성공 수, 실패 수)"""
    ok = failed = 0
    for song_id, state in list(registry.incomplete.items()):
        registry.attempted_this_run.add(song_id)
        recollect = state.needs_recollect()
        if recollect:
            logger.warning(
                f"재개 포기: {state.folder.name} [{recollect}] {state.attempts(recollect)}회 실패. "
                f"시드에 있으면 처음부터 다시 수집한다."
            )
            failed += 1
            continue
        stuck = state.exhausted()
        if stuck:
            # 다시 수집해도 고쳐지지 않는 단계(export)다. 여기서 포기하면 seed 루프도
            # attempted_this_run 때문에 건너뛰어 곡이 영원히 격리 폴더에 남는다. 계속 재시도한다.
            logger.warning(
                f"재시도 계속: {state.folder.name} [{stuck}] {state.attempts(stuck)}회 실패. "
                f"수집 폴더({registry.data_dir})와 JSONL 경로({registry.jsonl_path})를 확인한다."
            )
        logger.info(f"재개: {state.folder.name} [{state.next_stage()}]")
        try:
            succeeded = advance(state, registry)
        except Exception as e:
            # advance가 단계 예외를 흡수하므로 여기까지 오는 것은 상태 저장·실패 기록 자체가
            # 실패한 경우다. 그래도 남은 곡의 재개는 계속한다.
            logger.error("재개 중 예외 (%s): %s", state.folder.name, e, exc_info=True)
            succeeded = False
        failed += 0 if succeeded else 1
        ok += 1 if succeeded else 0
    return ok, failed


def process_song(artist: str, title: str, registry: Registry, song_id: Optional[str] = None) -> str:
    """
    한 곡에 대해 4단계 파이프라인(Melon -> YouTube -> Namuwiki -> LLM)을 실행.

    song_id: CSV에 멜론 곡 ID가 있으면 검색하지 않고 그 곡을 쓴다. 상세 페이지 검사(19금·장르·
        가사)는 그대로 거친다. 아티스트 인기곡 목록(build_expansion_list)은 곡 ID를 이미 알고
        있는데, '엔플라잉 (N.Flying) 잔불 (Still)'처럼 괄호가 든 검색어는 멜론 결과가 0건이라
        2026-10-10 첫 798곡 중 292곡이 검색에서 실패했다.

    Returns:
        "done" | "skipped" | "failed" | "blocked"
        blocked는 멜론이 접근을 막은 경우다. 호출부는 배치를 멈춰야 한다.
    """
    query = f"{artist} {title}"
    registry.made_requests = False

    # [중복 체크 - 검색 전] 수집 당시 요청 정보가 남아 있으면 검색조차 하지 않는다.
    # 예전에는 완료곡도 검색 1회와 대기 2~5초를 썼다. 요청 정보가 없는 예전 수집분은
    # 색인에 없으므로 아래에서 검색해 판단한다.
    known = registry.completed_song_for(artist, title)
    if known:
        logger.info(f"[SKIP] 이미 수집된 곡(ID:{known}) -> 검색 없이 스킵: {query}")
        return "skipped"

    logger.info(f"수집 시작: {query}")

    try:
        # 검색은 한 번만 한다. 예전에는 중복 체크와 collect_melon_data가 각각 검색해서
        # 곡당 멜론 요청이 두 번 나갔고, 두 호출이 다른 답을 낼 여지도 있었다.
        registry.made_requests = True
        if song_id:
            logger.info(f"멜론 ID 지정({song_id}) → 검색 생략")
            candidates = [SongCandidate(song_id=str(song_id), title=title, artists=[artist],
                                        match=MatchScore(score=EXACT_TITLE_BONUS, reasons=["멜론 ID 지정"]))]
        else:
            candidates = fetch_melon_song_candidates(artist, title)
        if not candidates:
            save_failed_song(artist, title, "Melon Search Failed")
            logger.warning(f"검증 통과 후보 없음 → 스킵: {query}")
            return "failed"

        best_id = str(candidates[0].song_id)

        # [중복 체크 - 조기 스킵] 최선 후보가 이미 있으면 상세 수집을 건너뛴다.
        if registry.is_complete(best_id):
            logger.info(f"[SKIP] 이미 수집된 곡(ID:{best_id}) -> 스킵: {query}")
            return "skipped"

        # [재개] 미완료 폴더가 있으면 원천 수집을 다시 하지 않는다.
        pending = registry.incomplete.get(best_id)
        if pending is not None:
            if pending.needs_recollect():
                logger.warning(f"미완료 폴더를 버리고 처음부터: {pending.folder.name}")
                shutil.rmtree(pending.folder, ignore_errors=True)
                registry.incomplete.pop(best_id, None)
            elif best_id in registry.attempted_this_run:
                logger.info(f"이번 실행에서 이미 재개를 시도함 -> 스킵: {query}")
                return "failed"
            else:
                registry.attempted_this_run.add(best_id)
                return "done" if advance(pending, registry) else "failed"

        source, state, reason = collect_melon_source(artist, title, candidates,
                                                     already_collected=registry.is_complete)
        if source is None:
            save_failed_song(artist, title, reason)
            logger.warning(f"{reason} → 스킵: {query}")
            return "failed"

        # 혹시 collect 중 id가 달라진 경우 한번 더 체크
        if registry.is_complete(state.song_id):
            logger.info(f"이미 수집됨(ID: {state.song_id}) -> 스킵: {query}")
            return "skipped"

        # [재개 - 실제 채택 ID] 위의 재개 검사는 검색 1위(best_id)로만 했다. 1위가 상세 검사에서
        # 탈락하고 다음 후보가 채택되면 그 검사가 이 곡을 놓친다.
        #
        # 놓치면 진행 중인 폴더를 새 원천 정보로 덮어쓴다. 그때 source.json과 meta.json은 새로
        # 고른 영상을 가리키는데, _download_audio는 정상 audio.m4a를 다시 받지 않으므로 예전
        # 영상의 오디오가 그대로 남는다. 오디오와 메타데이터가 어긋난 레코드가 만들어진다.
        # 폴더 이름이 {가수}_{제목}_{id}라서 같은 곡을 다시 채택하면 경로까지 같아, 예전의
        # folder 비교만으로는 걸러지지 않았다.
        #
        # 곡의 정체는 song_id다. 폴더 이름은 거들 뿐이라 **비교하지 않는다.** 멜론이 표기를
        # 바꾸면('밤편지' -> '밤 편지') 새 이름이 나오는데, 그 이유로 폴더를 지우면 검증을
        # 끝낸 분석과 오디오를 잃고 다시 분석한다. 그 분석이 실패하면 되돌릴 수도 없다.
        # 폴더를 버리는 것은 needs_recollect()가 참일 때뿐이다.
        pending = registry.incomplete.get(state.song_id)
        if pending is not None:
            if pending.needs_recollect():
                logger.warning(f"미완료 폴더를 버리고 처음부터: {pending.folder.name}")
                shutil.rmtree(pending.folder, ignore_errors=True)
                registry.incomplete.pop(state.song_id, None)
            elif state.song_id in registry.attempted_this_run:
                logger.info(f"이번 실행에서 이미 재개를 시도함 -> 스킵: {query}")
                return "failed"
            else:
                if pending.folder != state.folder:
                    # 이어받은 폴더의 표기를 그대로 쓴다. 레코드에는 예전 제목이 남는다.
                    logger.info(
                        f"멜론 표기가 바뀌었지만 진행 중인 폴더를 이어간다: "
                        f"{pending.folder.name} (새 이름 후보 {state.folder.name})"
                    )
                logger.info(
                    f"진행 중인 폴더를 이어서 진행(ID: {state.song_id}, 검색 1위는 {best_id}): {query}"
                )
                registry.attempted_this_run.add(state.song_id)
                return "done" if advance(pending, registry) else "failed"

        # 멜론까지의 결과를 먼저 저장한다. 유튜브가 실패해도 상세·댓글 수집과 댓글 LLM
        # 선별을 다시 하지 않는다. 나머지는 source 단계가 이어서 채운다.
        write_source(state.folder, source)
        state.save()
        registry.incomplete[state.song_id] = state
        registry.attempted_this_run.add(state.song_id)

        return "done" if advance(state, registry, source) else "failed"

    except AlreadyCollected as e:
        # 검색 1위가 탈락하고 다음 후보가 이미 수집된 곡인 경우. 앨범 소개·댓글·유튜브
        # 수집 전에 멈춘다.
        logger.info(f"이미 수집됨(ID: {e.song_id}) -> 스킵: {query}")
        return "skipped"
    except MelonAccessError as e:
        save_failed_song(artist, title, f"Melon Access Blocked: {e}")
        logger.error(f"멜론 접근 차단/한도 초과 -> 배치 중단: {e}")
        return "blocked"
    except CommentSelectionUnavailable as e:
        # 댓글 판정 장애를 '댓글 0개'로 저장하면, 원본 댓글이 수십 개인 곡이 몇 분짜리
        # 쿼터 초과 때문에 영구히 댓글 없는 레코드로 굳는다. 다음 실행에서 다시 시도한다.
        save_failed_song(artist, title, f"Comment Selection Unavailable (retry): {e}")
        logger.warning(f"댓글 판정 실패 -> 다음 실행에서 재시도: {query} ({e})")
        return "failed"
    except MelonTransientError as e:
        # 1위 후보의 일시 오류를 '이 후보는 틀렸다'로 읽으면 다른 음원이 채택되고, 다음 실행의
        # 중복 검사는 1위 id로 하므로 같은 시드에 레코드가 둘 남는다.
        save_failed_song(artist, title, f"Melon Transient Error (retry): {e}")
        logger.warning(f"멜론 일시 오류 -> 다음 실행에서 재시도: {query} ({e})")
        return "failed"
    except Exception as e:
        logger.error(f"처리 중 예외 발생 ({query}): {e}", exc_info=True)
        save_failed_song(artist, title, f"Exception: {str(e)}")
        return "failed"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vague-Finder 크롤러")
    parser.add_argument("songs_file", nargs="?", default="songs.csv", help="artist,title 컬럼 CSV")
    parser.add_argument(
        "--rebuild-jsonl", action="store_true",
        help="완료 폴더의 meta.json만으로 all_songs.jsonl을 다시 만들고 종료 (기존 파일은 시각 사본)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="--rebuild-jsonl이 레코드를 크게 줄이더라도 진행한다",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="시작 시 미완료 폴더 재개를 건너뛴다",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    reset_namuwiki_state()
    args = parse_args(argv)

    if args.rebuild_jsonl:
        try:
            count = rebuild_jsonl(DATA_DIR, JSONL_FILE, force=args.force)
        except RebuildRefused as e:
            logger.error("JSONL 재생성을 멈췄습니다: %s", e)
            return 3
        logger.info(f"JSONL 재생성: {count}곡 -> {JSONL_FILE}")
        return 0

    songs_file = args.songs_file

    # 테스트용 파일 자동생성
    if not os.path.exists(songs_file) and songs_file == "songs.csv":
        with open(songs_file, "w", encoding="utf-8") as f:
            f.write("artist,title\n")
            f.write("아이유,밤편지\n")
            f.write("박효신,야생화\n")
            f.write("악뮤,오랜 날 오랜 밤\n")
        logger.info(f"{songs_file} 생성됨. 여기에 노래를 추가하세요.")
        return 0

    songs = load_songs(songs_file)
    logger.info(f"총 {len(songs)}곡 수집 예정 (파일: {songs_file}, 파이프라인 {PIPELINE_VERSION})")
    if not ENABLE_NAMUWIKI:
        # 곡마다 남기면 로그가 곡 수만큼 늘어난다. 실행당 한 번만 알린다.
        logger.info("Namuwiki 단계 비활성화 (기본값). 외부 맥락은 backfill_namuwiki_context로 따로 모은다.")

    # 완료 폴더에는 있는데 JSONL에 빠진 곡을 먼저 채운다.
    # 폴더 목록과 JSONL을 한 번만 읽어 동기화와 Registry가 나눠 쓴다.
    complete = scan_complete(DATA_DIR)
    jsonl_ids = read_jsonl_ids(JSONL_FILE)
    added, orphans = sync_jsonl(DATA_DIR, JSONL_FILE, skip_record=record_policy_error,
                                complete=complete, jsonl_ids=jsonl_ids)
    if added:
        logger.info(f"JSONL 누락 복구: {added}곡 추가")
    if orphans:
        # 완료 폴더가 통째로 안 보이는 것은 보통 드라이브 마운트 문제다. 이 상태에서 재생성을
        # 권하면 파일을 비우게 된다(재생성 자체도 거절하지만, 안내부터 그렇게 하지 않는다).
        total_known = orphans + len(complete)
        if not DATA_DIR.exists() or orphans == total_known:
            logger.error(
                f"JSONL에 {orphans}곡이 있는데 완료 폴더에서 한 곡도 찾지 못했습니다. "
                f"수집 폴더({DATA_DIR})가 마운트됐는지 먼저 확인하세요. 이 상태로 --rebuild-jsonl을 "
                f"실행하면 안 됩니다."
            )
        else:
            logger.warning(
                f"JSONL에는 있는데 완료 폴더가 없는 곡 {orphans}곡. 폴더 위치({DATA_DIR})를 확인하고, "
                f"확인한 뒤에 정리하려면 --rebuild-jsonl을 쓴다."
            )

    registry = Registry(complete=complete, jsonl_ids=jsonl_ids)

    if registry.incomplete and not args.no_resume:
        ok, failed = resume_incomplete(registry)
        logger.info(f"미완료 폴더 재개: 성공 {ok}, 실패 {failed}")

    counts = {"done": 0, "skipped": 0, "failed": 0, "blocked": 0}
    for s in songs:
        result = process_song(s["artist"], s["title"], registry,
                              song_id=(s.get("song_id") or "").strip() or None)
        counts[result] = counts.get(result, 0) + 1
        if result == "blocked":
            logger.error("멜론 접근이 막혀 배치를 중단합니다. 잠시 뒤 다시 실행하세요.")
            break

        # [Politeness Policy] 실제 요청이 나갔을 때만 기다린다. 검색 전에 건너뛴 완료곡까지
        # 기다리면 961곡을 다시 순회할 때 대기만 32~80분이 든다.
        if not registry.made_requests:
            continue
        delay = random.uniform(2, 5)
        logger.info(f"봇 탐지 방지를 위해 {delay:.2f}초 대기...")
        time.sleep(delay)

    logger.info(
        "수집 종료: 완료 %d, 건너뜀 %d, 실패 %d%s",
        counts["done"], counts["skipped"], counts["failed"],
        " (접근 차단으로 중단)" if counts["blocked"] else "",
    )
    return 2 if counts["blocked"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
