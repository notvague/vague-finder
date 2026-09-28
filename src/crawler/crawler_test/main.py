# crawler_test/main.py
"""
Vague-Finder 크롤러(분리본) - 오케스트레이터(파이프라인 총괄)

이 파일이 하는 일(큰 흐름):
  1) songs.csv(artist,title) 읽기
  2) 각 곡에 대해 순서대로:
     - Step1: 멜론에서 메타/가사/댓글 수집
     - Step2: 유튜브에서 영상 선택 + 댓글/조회수 수집 (+옵션: 오디오 다운로드)
     - Step3: Gemini로 정제(요약/태그 생성) (+옵션: LLM 끄기)
  3) 결과를 data/raw/... 폴더에 meta.json으로 저장 (+옵션: cover/audio)

중요 포인트:
- "실패하면 전체 종료"가 아니라, 곡 단위로 실패 기록하고 다음 곡으로 넘어가도록 설계됨.
- 실행 위치(현재 작업 디렉토리)가 중요함:
  - data/, logs/ 등의 상대 경로는 "실행한 폴더" 기준으로 생성됨.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import logging.handlers
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from .melon import collect_melon_data
from .youtube import fetch_youtube_reaction, download_youtube_audio
from .refine_gemini import refine_data


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    로그 설정:
    - logs/crawler.log          : INFO 이상 전체 로그
    - logs/crawler_error.log    : ERROR 이상만
    - 콘솔 출력                 : INFO 이상

    RotatingFileHandler:
    - 로그 파일이 커지면(maxBytes) 자동으로 분할/백업(backupCount)
    """
    # Logs 폴더가 없으면 생성
    log_dir.mkdir(parents=True, exist_ok=True)

    # root Logger (최상위 로거)를 가져옴
    root_logger = logging.getLogger()

    # 로그 레벨 설정 (INFO 이상 출력)
    root_logger.setLevel(logging.INFO)

    # 기존 핸들러 (중복 출력 방지) 제거
    root_logger.handlers = []

    # 로그 포맷(시간 + 레벨 + 메시지)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1) 전체 로그 파일(회전)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "crawler.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2) 에러 로그 파일(회전)
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "crawler_error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # 3) 콘솔 출력용 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger("vf_crawler")


def load_songs(csv_path: Path, max_songs: Optional[int] = None) -> List[Dict[str, str]]:
    """
    songs.csv 읽기
    - 헤더: artist,title 이어야 함
    - max_songs가 있으면 앞에서 N개만 읽어서 테스트 가능
    """
    songs: List[Dict[str, str]] = []

    # 파일 존재 여부 체크
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # CSV 열기
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 값은 있을 수도/없을 수도 있으니 안전하게 처리
            artist = (row.get("artist") or "").strip()
            title = (row.get("title") or "").strip()

            # 둘 다 존재할 때만 수집 대상으로 추가
            if artist and title:
                songs.append({"artist": artist, "title": title})

                # max_songs 제한이 있으면 그 수만큼만 읽고 break
                if max_songs and len(songs) >= max_songs:
                    break
    return songs


def clean_filename(text: str) -> str:
    """
    Windows/파일시스템에서 폴더/파일명으로 쓸 수 없는 문자 제거 + 공백을 언더바로 변환.
    예) "AKMU (악뮤)" -> "AKMU_(악뮤)"
    """
    # 파일명 금지 문자를 제거
    text = re.sub(r'[\\/*?:"<>|]', "", text)

    # 공백(연속 포함)을 '_'로 치환
    text = re.sub(r"\s+", "_", text)

    # 양끝 공백 제거
    return text.strip()


def is_already_crawled(raw_dir: Path, melon_id: str) -> bool:
    """
    중복 수집 방지:
    - data/raw 안에 "..._{melon_id}" 로 끝나는 폴더가 있고
    - 그 안에 meta.json이 있으면 이미 수집한 것으로 판단하여 스킵

    이유:
    - 폴더명에는 Artist/Title이 들어가서 완전 일치 비교가 어려움
    - 대신 멜론 ID는 고유하니 suffix로 확인
    """
    if not raw_dir.exists():
        return False
    
    # data/raw 아래 폴더들을 순회
    for path in raw_dir.iterdir():
        if path.is_dir() and path.name.endswith(f"_{melon_id}") and (path / "meta.json").exists():
            return True
    
    return False


def save_failed(log_path: Path, artist: str, title: str, reason: str) -> None:
    """
    실패한 곡을 data/failed_songs.csv에 기록(append)
    - 없으면 헤더를 먼저 씀
    - reason에 실패 원인을 문자열로 남김
    """
    # data/ 생성
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 존재 여부 -> 헤더 작성 여부 판단
    file_exists = log_path.exists()

    with log_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        # 파일이 새로 만들어지는 상황이면 헤더 작성
        if not file_exists:
            w.writerow(["timestamp", "artist", "title", "reason"])

        # 실제 실패 기록 row
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), artist, title, reason])


def download_cover(cover_url: str, save_dir: Path, logger: logging.Logger) -> None:
    """
    커버 이미지 다운로드
    - cover_url로 GET
    - 성공하면 cover.jpg 저장
    """
    if not cover_url:
        return
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(cover_url, headers=headers, timeout=15)
        if res.status_code == 200 and res.content:
            (save_dir / "cover.jpg").write_bytes(res.content)
            logger.info("커버 이미지 저장 완료: cover.jpg")
        else:
            logger.warning(f"커버 이미지 다운로드 실패: HTTP {res.status_code}")
    except Exception as e:
        logger.warning(f"커버 이미지 다운로드 실패: {e}")


def save_result(raw_dir: Path, melon_id: str, data: Dict, logger: logging.Logger, do_cover: bool, do_audio: bool) -> Path:
    """
    한 곡의 최종 결과 저장:
    - data/raw/{Artist}_{Title}_{MelonID}/meta.json 저장
    - 옵션에 따라 cover.jpg 다운로드
    - 옵션에 따라 audio.m4a 다운로드
    """
    # meta.json에 들어갈 "metadata" 블록을 꺼냄
    meta = data.get("metadata", {})

    # 폴더명에 들어갈 artist/title을 파일명 안전하게 변환
    artist = clean_filename(meta.get("artist") or "Unknown")
    title = clean_filename(meta.get("title") or "Unknown")

    # 폴더명 규칙
    folder_name = f"{artist}_{title}_{melon_id}"

    #저장 폴더 경로
    save_dir = raw_dir / folder_name
    
    # 폴더 생성
    save_dir.mkdir(parents=True, exist_ok=True)

    # meta.json 저장
    (save_dir / "meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"저장 완료: {save_dir / 'meta.json'}")

    # cover_url은 metadata 안에 있을 수도, data의 최상위에 있을 수도 있으니 둘 다 확인
    cover_url = meta.get("cover_url") or data.get("cover_url")

    # 커버 저장 옵션이 켜져 있으면 다운로드 시도
    if do_cover and cover_url:
        download_cover(cover_url, save_dir, logger)

    # 오디오 저장 옵션이 켜져 있으면 yt-dlp 다운로드 시도
    if do_audio:
        youtube_url = meta.get("youtube_url", "")
        
        if youtube_url:
            
            audio_path = save_dir / "audio.m4a"
            ok = download_youtube_audio(youtube_url, str(audio_path))
            
            if ok:
                logger.info("오디오 다운로드 완료")
        else:
            logger.warning("youtube_url이 없어 오디오 다운로드를 스킵합니다.")

    return save_dir


def popularity_tag(view_count: int) -> str:
    """
    조회수 기반으로 단순한 인기도 태그를 생성
    """
    if view_count >= 100_000_000:
        return "Mega Hit"
    if view_count >= 10_000_000:
        return "Famous"
    if view_count >= 1_000_000:
        return "Moderate"
    return "Hidden"


def build_ordered(melon_data: Dict, reaction_data: Dict, refined: Dict) -> Dict:
    """
    최종 저장 JSON 구조를 "우리가 원하는 형태"로 정렬/구성하는 함수

    - melon_data: 멜론에서 가져온 데이터(메타/가사/멜론댓글/커버URL 등)
    - reaction_data: 유튜브에서 가져온 데이터(영상URL/조회수/유튜브댓글 등)
    - refined: Gemini가 만든 정제 결과(요약/태그 등) 또는 빈 값들
    """
    # reaction_data에 view_count가 None일 수 있어 int 변환 시 안전하게 0 처리
    view_count = int(reaction_data.get("view_count") or 0)

    # 유튜브 영상 URL
    video_url = reaction_data.get("video_url") or ""

    return {
        "id": melon_data.get("id", ""),
        # "metadata"에 우리가 사용하는 필드들을 모아둠
        "metadata": {
            "title": melon_data.get("title"),
            "artist": melon_data.get("artist"),
            "release_date": melon_data.get("release_date", ""),
            "genre": melon_data.get("genre", ""),

            # 유튜브에서 얻은 조회수 + 단순 인기도 태그
            "view_count": view_count,
            "popularity_tag": popularity_tag(view_count),

            # 앨범 정보(멜론에서)
            "album_name": melon_data.get("album_name", ""),
            "album_desc": melon_data.get("album_desc", ""),
            
            # 앨범 소개를 Gemini가 요약한 결과
            "album_summary": refined.get("album_summary", ""),
            
            # 가사 원문 + 하이라이트/요약
            "lyrics": melon_data.get("lyrics"),
            "lyrics_highlight": refined.get("lyrics_highlight", ""),
            "lyrics_summary": refined.get("lyrics_summary", ""),
            
            # 댓글: 멜론/유튜브 원본 리스트
            "melon_comments": melon_data.get("melon_comments", []),
            "youtube_comments": reaction_data.get("comments", []),
            
            # 댓글 맥락 요약(정제)
            "comment_context": refined.get("comment_context", ""),
            
            # 분위기 태그 + 씬 요약(정제)
            "mood_tags": refined.get("mood_tags", []),
            "scene_summary": refined.get("scene_summary", ""),

            # 신규 필드
            "time_weather_tags": refined.get("time_weather_tags", []),
            "place_activity_tags": refined.get("place_activity_tags", []),
            "emotion_tags": refined.get("emotion_tags", []),
            "vibe_tags": refined.get("vibe_tags", []),
            "relation_context_tags": refined.get("relation_context_tags", []),
            "color_tags": refined.get("color_tags", []),
            "visual_imagery": refined.get("visual_imagery", []),
            "sound_tags": refined.get("sound_tags", []),
            "search_style_summary": refined.get("search_style_summary", ""),

            # 멜론 DJ 플레이리스트 기반 추가 단서
            "melon_playlists": melon_data.get("melon_playlists", []),
            "melon_playlist_titles": melon_data.get("melon_playlist_titles", []),
            "melon_playlist_tags": melon_data.get("melon_playlist_tags", []),

            # 커버 URL(멜론)
            "cover_url": melon_data.get("cover_url"),
            
            # 유튜브 URL(오디오 소스/댓글 소스)
            "youtube_url": video_url,
        },
    }


def process_song(
    artist: str,
    title: str,
    raw_dir: Path,
    failed_csv: Path,
    logger: logging.Logger,
    use_llm: bool,
    do_cover: bool,
    do_audio: bool,
    sleep_min: float,
    sleep_max: float,
) -> None:
    """
    "곡 1개"를 처리하는 함수.

    여기서 Step1/2/3을 순서대로 수행하고,
    실패하면 failed_songs.csv에 기록한 뒤 return(다음 곡 진행)한다.
    """
    # 사람이 보기 좋은 로그용 문자열
    query = f"{artist} {title}"
    logger.info(f"수집 시작: {query}")

    try:
        # -----------------------
        # Step 1) 멜론 수집
        # -----------------------
        melon_data = collect_melon_data(artist, title)

        # 멜론 단계 자체가 실패하면 melon_data는 {}로 올 수 있음
        if not melon_data:
            save_failed(failed_csv, artist, title, "Melon Search Failed")
            logger.error("Melon 단계 실패")
            return

        # 멜론 고유 ID (폴더명 suffix / 중복 체크에 쓰임)
        melon_id = melon_data.get("id")

        if not melon_id:
            save_failed(failed_csv, artist, title, "Melon ID Not Found")
            logger.error("Melon ID 누락")
            return

        # 이미 수집한 곡이면 스킵
        if is_already_crawled(raw_dir, melon_id):
            logger.info(f"이미 수집됨(ID: {melon_id}) -> 스킵: {query}")
            return

        # 가사 (유튜브 댓글에서 가사 복붙 필터링에 쓰임)
        lyrics = melon_data.get("lyrics", "") or ""

        # -----------------------
        # Step 2) 유튜브 수집
        # -----------------------
        reaction_data = fetch_youtube_reaction(artist, title, song_lyrics=lyrics)
        
        # video_url이 없으면 오디오 소스도 없다는 뜻이므로 실패 처리
        if not reaction_data or not reaction_data.get("video_url"):
            save_failed(failed_csv, artist, title, "No YouTube URL / Reaction Failed")
            logger.error("YouTube 단계 실패")
            return

        # -----------------------
        # Step 3) LLM 정제
        # -----------------------
        if use_llm:
            # Gemini 호출해서 mood_tags/scene_summary/... 생성
            refined = refine_data(melon_data, lyrics, reaction_data)
        else:
            # LLM을 끄는 옵션(--no-llm)일 때는 빈 값으로 채움
            refined = {
                "album_summary": "",
                "lyrics_summary": "",
                "comment_context": "",
                "mood_tags": [],
                "scene_summary": "",
                "lyrics_highlight": "",
                "time_weather_tags": [],
                "place_activity_tags": [],
                "emotion_tags": [],
                "vibe_tags": [],
                "relation_context_tags": [],
                "color_tags": [],
                "visual_imagery": [],
                "sound_tags": [],
                "search_style_summary": "",
            }

        # -----------------------
        # 최종 JSON 구조로 조립 + 저장
        # -----------------------
        ordered = build_ordered(melon_data, reaction_data, refined)
        
        # meta.json/cover/audio 저장
        save_result(raw_dir, melon_id, ordered, logger, do_cover=do_cover, do_audio=do_audio)

    except Exception as e:
        # 예상치 못한 예외는 여기로 떨어짐
        logger.error(f"처리 중 에러 ({query}): {e}", exc_info=True)
        
        # 실패 기록
        save_failed(failed_csv, artist, title, f"Exception: {e}")

    # -----------------------
    # 봇탐지 회피용 랜덤 sleep
    # -----------------------
    delay = random.uniform(sleep_min, sleep_max)
    logger.info(f"봇 탐지 방지를 위해 {delay:.2f}초 대기...")
    time.sleep(delay)


def main():
    """
    프로그램 시작점.
    - .env 로드
    - CLI 옵션 파싱
    - 로깅 설정
    - songs.csv 읽고 곡별로 process_song() 호출
    """
    load_dotenv()

    # -----------------------
    # CLI 옵션 정의
    # -----------------------
    ap = argparse.ArgumentParser()

    # songs.csv 경로(기본: 현재 폴더의 songs.csv)
    ap.add_argument("--songs", default="songs.csv", help="artist,title 컬럼을 가진 CSV 경로")

    # 결과 저장 루트(기본: data/raw)
    ap.add_argument("--out", default="data/raw", help="결과 저장 루트 폴더")

    # 로그 폴더(기본: logs)
    ap.add_argument("--logs", default="logs", help="로그 폴더")

    # 실패 기록 CSV(기본: data/failed_songs.csv)
    ap.add_argument("--failed", default="data/failed_songs.csv", help="실패 기록 CSV")

    # 앞에서 N곡만 실행(테스트용)
    ap.add_argument("--max-songs", type=int, default=0, help="테스트용으로 앞에서 N곡만 실행(0이면 전체)")

    # LLM 비활성화
    ap.add_argument("--no-llm", action="store_true", help="Gemini 정제 단계 비활성화")

    # 오디오 다운로드 비활성화
    ap.add_argument("--no-audio", action="store_true", help="오디오 다운로드 비활성화")

    # 커버 다운로드 비활성화
    ap.add_argument("--no-cover", action="store_true", help="커버 다운로드 비활성화")

    # 요청 사이 대기시간 범위
    ap.add_argument("--sleep-min", type=float, default=2.0)
    ap.add_argument("--sleep-max", type=float, default=5.0)

    # 실제 옵션 파싱 수행
    args = ap.parse_args()

    # -----------------------
    # 로깅 세팅
    # -----------------------
    logger = setup_logging(Path(args.logs))

    # -----------------------
    # songs.csv 로딩
    # -----------------------
    songs_path = Path(args.songs)

    # max-songs가 0이면 제한 없음(None)
    max_songs = args.max_songs if args.max_songs and args.max_songs > 0 else None
    
    songs = load_songs(songs_path, max_songs=max_songs)
    logger.info(f"총 {len(songs)}곡 수집 예정 (파일: {songs_path})")

    # 결과 저장 루트/실패 CSV 경로
    raw_dir = Path(args.out)
    failed_csv = Path(args.failed)

    # -----------------------
    # 곡별 처리 루프
    # -----------------------
    for s in songs:
        process_song(
            s["artist"],
            s["title"],
            raw_dir=raw_dir,
            failed_csv=failed_csv,
            logger=logger,
            use_llm=(not args.no_llm),
            do_cover=(not args.no_cover),
            do_audio=(not args.no_audio),
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
        )

# 이 파일을 직접 python crawler_test/main.py로 실행하는 경우에도 동작하게 하는 안전장치
if __name__ == "__main__":
    main()
