"""
tests/test_crawler_pipeline.py

크롤러 오케스트레이터(src/crawler/main.py)의 단계별 재시작과 저장 일관성 회귀 테스트.

배경 (2026-09-18 점검):

1. JSONL 저장이 실패해도 수집 완료로 처리됐다. save_to_jsonl()이 예외를 삼켜서
   save_result()가 True를 돌려줬고, 재실행에서는 폴더만 보고 건너뛰었다.
2. Gemini가 세 번 모두 실패하면 '분석실패' 값을 정상 레코드처럼 저장했다. 실패 목록에도
   남지 않았다.
3. 다운로드 포맷은 bestaudio[ext=m4a]/bestaudio인데 완료 판정·Mongo 경로·임베딩 입력은
   audio.m4a만 본다. audio.webm만 생기면 저장 성공 뒤 재실행에서 미완료가 됐다.
4. 커버는 HTTP 200과 파일 크기만 봐서 HTML 오류 본문도 cover.jpg로 저장됐다.
5. LLM 처리까지 끝난 뒤 오디오가 실패하면 폴더를 지워 전 단계를 다시 했다.

2026-09-18 2차 점검에서 나온 것:

6. 댓글 없는 음원을 수집 완료로 저장했지만 임베딩 검증기가 합계 10개를 요구해 곡이 빠졌다.
7. 재수집한 곡의 JSONL 줄이 예전 분석 그대로 남았다(같은 ID라는 이유로 기록을 건너뜀).
8. 단계 함수가 예외를 던지면 재개 루프 밖으로 퍼져 나머지 곡의 재개까지 멈췄다.

네트워크와 LLM은 전부 가짜다. 테스트 문장은 지어낸 것이다.

실행:
    venv/bin/python -m pytest tests/test_crawler_pipeline.py -v
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import struct
from pathlib import Path

import pytest
from PIL import Image

from src.common.gemini_client import crawl_model_name
from src.crawler import main as crawler
from src.crawler.scripts_py import crawl_state
from src.crawler.scripts_py.collect_melon_data import (
    AlreadyCollected,
    MelonAccessError,
    MelonTransientError,
    SongCandidate,
)
from src.crawler.scripts_py.llm_utils import CommentSelectionUnavailable
from src.crawler.scripts_py.melon_match import MatchScore
from src.embedding.fixtures.meta_validation import count_comments, validate_meta_document

SONG_ID = "100"


# --- 가짜 에셋 ---------------------------------------------------------------------------

def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 120), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def m4a_bytes(seconds: float = 200.0, pad: int = 150_000) -> bytes:
    """ftyp + moov/mvhd(v0)만 있는 최소 MP4. 헤더 검사와 재생 시간 검사를 통과한다."""
    ftyp = struct.pack(">I", 20) + b"ftypM4A " + b"\x00\x00\x00\x00" + b"M4A "
    mvhd_body = b"\x00\x00\x00\x00" + struct.pack(">IIII", 0, 0, 1000, int(seconds * 1000)) + b"\x00" * 80
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    return ftyp + moov + b"\x00" * pad


FULL_LLM = {
    "album_summary": "앨범 소개 요약", "artist_type": ["솔로"], "vocal_gender": "여성",
    "lyrics_highlight": "가장 기억에 남는 한 줄", "lyrics_summary": "가사 요약",
    "search_style_summary": "상황 묘사", "mood_tags": ["잔잔함"], "time_weather_tags": ["새벽"],
    "place_activity_tags": ["창가"], "emotion_tags": ["그리움"], "vibe_tags": ["몽환적"],
    "relation_context_tags": ["이별"], "color_tags": ["파랑"], "sound_tags": ["피아노"],
    "visual_imagery": ["비 오는 창가"], "sentiment_summary": "댓글 요약", "fans_tags": ["인생곡"],
    "major_emotion": "슬픔", "context_tags": [], "fact_summary": "",
}


def melon_payload(song_id: str = SONG_ID) -> dict:
    return {
        "id": song_id, "title": "밤편지", "artist": ["가수"], "lyrics": "가사 본문",
        "cover_url": "https://cdn.example/cover.jpg", "release_date": "2017.03.24",
        "genre": "발라드", "album_name": "Palette", "album_desc": "앨범 소개",
        "melon_playlist_tags": ["새벽감성"], "melon_comments": ["새벽에 듣기 좋은 곡"] * 6,
        "match_audit": {"score": 100, "reasons": [], "needs_review": False, "note": "",
                        "rank": 0, "album": "Palette", "candidate_count": 1},
    }


def reaction_payload() -> dict:
    return {
        "video_url": "https://www.youtube.com/watch?v=abc", "video_title": "가수 - 밤편지 Official",
        "view_count": 1_500_000, "comments": ["비 오는 밤에 어울리는 노래"] * 6,
        "artist_verified": True, "comment_source_url": "https://www.youtube.com/watch?v=abc",
    }


class Calls:
    def __init__(self):
        self.melon = 0
        self.youtube = 0
        self.refine = 0


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """네트워크·LLM을 전부 가짜로 바꾼 오케스트레이터. 폴더와 CSV는 tmp_path 아래."""
    data_dir = tmp_path / "raw"
    staging = tmp_path / "raw_incomplete"
    monkeypatch.setattr(crawler, "DATA_DIR", data_dir)
    monkeypatch.setattr(crawler, "STAGING_DIR", staging)
    monkeypatch.setattr(crawler, "JSONL_FILE", tmp_path / "all_songs.jsonl")
    monkeypatch.setattr(crawler, "FAILED_LOG_FILE", tmp_path / "failed_songs.csv")
    monkeypatch.setattr(crawler, "REVIEW_LOG_FILE", tmp_path / "review_songs.csv")
    monkeypatch.setattr(crawler, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(crawler, "ENABLE_NAMUWIKI", False)
    monkeypatch.setattr(crawler.time, "sleep", lambda s: None)

    calls = Calls()
    state = {"melon": melon_payload(), "reaction": reaction_payload(), "refine": dict(FULL_LLM),
             "cover": (200, "image/jpeg", jpeg_bytes()), "audio": m4a_bytes()}

    def fake_candidates(artist, title):
        return [SongCandidate(song_id=state["melon"]["id"], title="밤편지", match=MatchScore(score=100))]

    def fake_collect(artist, title, candidates=None, already_collected=None):
        calls.melon += 1
        song_id = state["melon"]["id"]
        # 실제 collect_melon_data와 같이, 채택이 확정된 직후 중복을 확인한다.
        if already_collected is not None and already_collected(song_id):
            raise AlreadyCollected(song_id)
        return json.loads(json.dumps(state["melon"]))

    def fake_reaction(artist, title, song_lyrics="", melon_title="", known_artists=None):
        calls.youtube += 1
        return json.loads(json.dumps(state["reaction"]))

    def fake_refine(metadata, lyrics, reaction, namuwiki_data=None):
        calls.refine += 1
        return json.loads(json.dumps(state["refine"]))

    class Response:
        def __init__(self, status, ctype, content):
            self.status_code, self.headers, self.content = status, {"Content-Type": ctype}, content

    def fake_get(url, headers=None, timeout=None):
        return Response(*state["cover"])

    def fake_download(video_url, save_path):
        Path(save_path).write_bytes(state["audio"])
        return True

    monkeypatch.setattr(crawler, "fetch_melon_song_candidates", fake_candidates)
    monkeypatch.setattr(crawler, "collect_melon_data", fake_collect)
    monkeypatch.setattr(crawler, "fetch_youtube_reaction", fake_reaction)
    monkeypatch.setattr(crawler, "refine_data", fake_refine)
    monkeypatch.setattr(crawler.requests, "get", fake_get)
    monkeypatch.setattr(crawler, "download_youtube_audio", fake_download)

    return crawler, state, calls, tmp_path


def failed_reasons(tmp_path: Path) -> list:
    path = tmp_path / "failed_songs.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [row["reason"] for row in csv.DictReader(f)]


def jsonl_records(tmp_path: Path) -> list:
    return list(crawl_state.iter_jsonl(tmp_path / "all_songs.jsonl"))


def issue_paths(record: dict, *prefixes: str) -> list:
    """관련 경로의 검증 이슈만 뽑는다. 테스트용 최소 레코드는 다른 필수 필드가 없어서
    전체 이슈 목록으로는 비교할 수 없다."""
    return sorted(
        issue.path
        for issue in validate_meta_document(record, require_media_files=False)
        if not prefixes or issue.path.startswith(prefixes)
    )


# --- 정상 경로 ---------------------------------------------------------------------------

def test_full_pipeline_writes_complete_folder_and_jsonl(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    registry = crawler.Registry()

    assert crawler.process_song("가수", "밤편지", registry) == "done"

    folders = list((tmp / "raw").iterdir())
    assert [f.name for f in folders] == ["가수_밤편지_100"]
    folder = folders[0]
    for name in ("meta.json", "cover.jpg", "audio.m4a", "source.json", "crawl_status.json"):
        assert (folder / name).is_file(), name
    assert not (tmp / "raw_incomplete").exists() or not list((tmp / "raw_incomplete").iterdir())

    records = jsonl_records(tmp)
    assert [r["song_id"] for r in records] == [SONG_ID]
    assert registry.is_complete(SONG_ID)
    assert failed_reasons(tmp) == []

    # 수집 이력이 레코드에 남고, 임베딩 전 검증을 통과한다
    status = records[0]["crawl_status"]
    assert (status["input_artist"], status["input_title"], status["status"]) == ("가수", "밤편지", "complete")
    assert status["selection"]["melon"]["song_id"] == SONG_ID
    assert status["selection"]["youtube"]["video_url"] == "https://www.youtube.com/watch?v=abc"
    assert status["pipeline_version"] == crawl_state.PIPELINE_VERSION
    assert status["refine_model"] == crawl_model_name(), "정제 모델이 곡마다 남아야 모델이 섞인 코퍼스를 가려낸다"
    issues = validate_meta_document(records[0], song_dir=folder, require_media_files=True)
    assert issues == [], [i.to_dict() for i in issues]

    # 재실행은 아무것도 다시 하지 않는다
    calls_before = (calls.melon, calls.youtube, calls.refine)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "skipped"
    assert (calls.melon, calls.youtube, calls.refine) == calls_before


# --- 1. JSONL 저장 실패 -----------------------------------------------------------------

def test_jsonl_append_failure_is_not_reported_as_complete(pipeline, monkeypatch) -> None:
    """JSONL을 못 쓰면 완료가 아니다. 폴더는 격리 폴더에 남고 실패 목록에 기록된다."""
    crawler, state, calls, tmp = pipeline

    def broken_append(path, record):
        raise OSError("disk full")

    monkeypatch.setattr(crawler, "append_jsonl", broken_append)
    registry = crawler.Registry()
    assert crawler.process_song("가수", "밤편지", registry) == "failed"

    assert not registry.is_complete(SONG_ID)
    assert not (tmp / "raw").exists() or not list((tmp / "raw").iterdir())
    staged = list((tmp / "raw_incomplete").iterdir())
    assert [f.name for f in staged] == ["가수_밤편지_100"]
    assert (staged[0] / "meta.json").is_file()      # 분석까지는 끝났다
    assert jsonl_records(tmp) == []
    assert any(r.startswith("[export] JSONL Write Failed") for r in failed_reasons(tmp))

    # 다음 실행: 원천·에셋·분석을 다시 하지 않고 export만 이어서 한다
    monkeypatch.setattr(crawler, "append_jsonl", crawl_state.append_jsonl)
    monkeypatch.setattr(crawler, "replace_jsonl_record", crawl_state.replace_jsonl_record)
    before = (calls.melon, calls.youtube, calls.refine)
    registry2 = crawler.Registry()
    assert crawler.resume_incomplete(registry2) == (1, 0)
    assert (calls.melon, calls.youtube, calls.refine) == before
    assert [r["song_id"] for r in jsonl_records(tmp)] == [SONG_ID]
    assert [f.name for f in (tmp / "raw").iterdir()] == ["가수_밤편지_100"]
    assert list((tmp / "raw_incomplete").iterdir()) == []


def test_export_retry_after_move_failure_does_not_duplicate_jsonl_line(pipeline, monkeypatch) -> None:
    """JSONL 추가 뒤 폴더 이동이 실패하면, 재시도는 추가를 건너뛰고 이동만 한다."""
    crawler, state, calls, tmp = pipeline

    def broken_move(src, dst):
        raise OSError("drive offline")

    monkeypatch.setattr(crawler.shutil, "move", broken_move)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert len(jsonl_records(tmp)) == 1

    monkeypatch.undo()
    monkeypatch.setattr(crawler, "DATA_DIR", tmp / "raw")
    monkeypatch.setattr(crawler, "STAGING_DIR", tmp / "raw_incomplete")
    monkeypatch.setattr(crawler, "JSONL_FILE", tmp / "all_songs.jsonl")
    monkeypatch.setattr(crawler, "FAILED_LOG_FILE", tmp / "failed_songs.csv")
    assert crawler.resume_incomplete(crawler.Registry()) == (1, 0)
    assert len(jsonl_records(tmp)) == 1
    assert [f.name for f in (tmp / "raw").iterdir()] == ["가수_밤편지_100"]


def test_startup_sync_restores_jsonl_lines_for_complete_folders(tmp_path) -> None:
    """다운로드 직후 프로세스가 죽어 JSONL만 빠진 폴더는 시작할 때 채운다."""
    data_dir = tmp_path / "raw"
    jsonl = tmp_path / "all_songs.jsonl"
    folder = data_dir / "가수_곡_7"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"song_id": "7", "metadata": {"title": "곡"}}), encoding="utf-8")
    (folder / "cover.jpg").write_bytes(jpeg_bytes())
    (folder / "audio.m4a").write_bytes(m4a_bytes())
    # 파일이 빠진 폴더는 완료가 아니라서 넣지 않는다
    partial = data_dir / "가수_곡_8"
    partial.mkdir()
    (partial / "meta.json").write_text(json.dumps({"song_id": "8"}), encoding="utf-8")

    assert crawl_state.sync_jsonl(data_dir, jsonl) == (1, 0)
    assert [r["song_id"] for r in crawl_state.iter_jsonl(jsonl)] == ["7"]
    assert crawl_state.sync_jsonl(data_dir, jsonl) == (0, 0)      # 두 번 넣지 않는다

    # JSONL에만 있는 곡은 지우지 않고 세기만 한다(드라이브가 잠시 안 보일 수 있다)
    crawl_state.append_jsonl(jsonl, {"song_id": "9"})
    assert crawl_state.sync_jsonl(data_dir, jsonl) == (0, 1)
    assert len(list(crawl_state.iter_jsonl(jsonl))) == 2


def test_rebuild_jsonl_from_folders_keeps_a_backup(tmp_path) -> None:
    data_dir = tmp_path / "raw"
    jsonl = tmp_path / "all_songs.jsonl"
    for song_id in ("1", "2"):
        folder = data_dir / f"가수_곡_{song_id}"
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({"song_id": song_id}), encoding="utf-8")
        (folder / "cover.jpg").write_bytes(jpeg_bytes())
        (folder / "audio.m4a").write_bytes(m4a_bytes())
    jsonl.write_text('{"song_id": "1"}\n{"song_id": "1"}\n{"song_id": "99"}\n', encoding="utf-8")

    assert crawl_state.rebuild_jsonl(data_dir, jsonl) == 2
    assert sorted(r["song_id"] for r in crawl_state.iter_jsonl(jsonl)) == ["1", "2"]
    # 사본은 실행 시각을 달고 남는다. 한 자리에만 남기면 두 번째 재생성이 사본을 덮어써
    # 되돌릴 수 없다.
    backups = list(tmp_path.glob("all_songs.jsonl.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8").count("\n") == 3


# --- 2. 분석 실패 -----------------------------------------------------------------------

def test_analysis_failure_keeps_source_and_retries_only_analysis(pipeline) -> None:
    """Gemini가 실패하면 저장하지 않고, 다음 실행은 원천 수집 없이 분석만 다시 한다."""
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}                       # refine_data의 실패 신호

    registry = crawler.Registry()
    assert crawler.process_song("가수", "밤편지", registry) == "failed"
    assert not registry.is_complete(SONG_ID)
    assert jsonl_records(tmp) == []
    assert any(r == "[analysis] LLM Analysis Failed" for r in failed_reasons(tmp))

    staged = tmp / "raw_incomplete" / "가수_밤편지_100"
    assert (staged / "source.json").is_file()
    assert (staged / "cover.jpg").is_file() and (staged / "audio.m4a").is_file()
    assert not (staged / "meta.json").exists()
    status = json.loads((staged / "crawl_status.json").read_text(encoding="utf-8"))
    assert status["stages"]["source"]["status"] == "done"
    assert status["stages"]["assets"]["status"] == "done"
    assert status["stages"]["analysis"] == {
        "status": "failed", "error": "LLM Analysis Failed", "attempts": 1,
        "at": status["stages"]["analysis"]["at"],
    }
    # '분석실패' 같은 대체값이 어디에도 저장되지 않았다
    assert "분석실패" not in (staged / "source.json").read_text(encoding="utf-8")

    # 다음 실행: 시드를 다시 만나도 멜론·유튜브는 부르지 않는다
    state["refine"] = dict(FULL_LLM)
    before = (calls.melon, calls.youtube)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert (calls.melon, calls.youtube) == before
    assert calls.refine == 2
    assert [r["song_id"] for r in jsonl_records(tmp)] == [SONG_ID]


def test_stage_exhausted_after_repeated_failures_recrawls_from_scratch(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}
    for _ in range(crawl_state.MAX_STAGE_ATTEMPTS):
        assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert calls.melon == 1

    # 한계에 닿은 폴더는 버리고 처음부터 다시 수집한다
    state["refine"] = dict(FULL_LLM)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert calls.melon == 2


# --- 4·추가2. 오디오·커버 내용 검증 -----------------------------------------------------

def test_webm_only_download_is_a_failure_not_a_success(pipeline, monkeypatch) -> None:
    """audio.webm만 생기면 완료가 아니다. 예전에는 저장 성공 + JSONL 추가 뒤 재실행에서
    미완료로 판정돼 다시 수집하고 JSONL에 중복이 쌓였다."""
    crawler, state, calls, tmp = pipeline

    def webm_download(video_url, save_path):
        Path(save_path).with_suffix(".webm").write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 200_000)
        return False    # 고친 다운로더는 m4a가 아니면 False다

    monkeypatch.setattr(crawler, "download_youtube_audio", webm_download)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert jsonl_records(tmp) == []
    assert not (tmp / "raw").exists() or not list((tmp / "raw").iterdir())
    assert "[assets] Audio Download Failed" in failed_reasons(tmp)


@pytest.mark.parametrize(
    "audio, problem",
    [
        (b"<html>error</html>" + b"\x00" * 200_000, "m4a 헤더 아님"),
        (m4a_bytes(pad=10), "파일이 너무 작음"),
        (m4a_bytes(seconds=12.0), "재생 시간이 너무 짧음"),
    ],
)
def test_audio_content_is_verified(tmp_path, audio, problem) -> None:
    path = tmp_path / "audio.m4a"
    path.write_bytes(audio)
    assert crawler.verify_audio(path).startswith(problem)
    path.write_bytes(m4a_bytes())
    assert crawler.verify_audio(path) == ""
    assert crawler.m4a_duration_seconds(path) == pytest.approx(200.0)


def test_html_error_body_is_not_accepted_as_cover(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["cover"] = (200, "text/html; charset=utf-8", b"<html><body>Access Denied</body></html>")
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert "[assets] Cover Download Failed" in failed_reasons(tmp)
    staged = tmp / "raw_incomplete" / "가수_밤편지_100"
    assert not (staged / "cover.jpg").exists()

    # Content-Type이 이미지라고 해도 디코딩이 안 되면 거부한다
    state["cover"] = (200, "image/jpeg", b"not really a jpeg" * 100)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert not (staged / "cover.jpg").exists()

    # 정상 이미지가 오면 그 자리에서 이어간다. 원천 수집은 한 번뿐이다.
    state["cover"] = (200, "image/jpeg", jpeg_bytes())
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert calls.melon == 1


# --- 6. 멜론 접근 차단은 배치를 멈춘다 ---------------------------------------------------

def test_melon_block_stops_the_batch(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline
    seen = []

    def blocked(artist, title):
        seen.append(title)
        raise MelonAccessError("HTTP 403: search")

    monkeypatch.setattr(crawler, "fetch_melon_song_candidates", blocked)
    songs = tmp / "songs.csv"
    songs.write_text("artist,title\n가수,첫곡\n가수,둘째곡\n가수,셋째곡\n", encoding="utf-8")

    assert crawler.main([str(songs)]) == 2
    assert seen == ["첫곡"]                     # 둘째·셋째 곡은 시도하지 않았다
    assert failed_reasons(tmp) == ["Melon Access Blocked: HTTP 403: search"]


# --- 3. 가수 미확인 영상은 검토 목록에 남긴다 ---------------------------------------------

def test_unverified_artist_video_is_kept_but_listed_for_review(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["reaction"]["artist_verified"] = False
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    with (tmp / "review_songs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and "가수 이름 없음" in rows[0]["reason"]
    record = jsonl_records(tmp)[0]
    # 검토 메모는 notes에 남는다. warnings에 넣으면 임베딩 전 검증이 곡을 뺀다.
    assert any(w.startswith("youtube:") for w in record["crawl_status"]["notes"])
    assert record["crawl_status"]["warnings"] == []
    assert validate_meta_document(record, require_media_files=False) == []
    assert record["crawl_status"]["selection"]["youtube"]["artist_verified"] is False


def test_incomplete_folders_are_resumed_at_startup_without_the_seed(pipeline) -> None:
    """시드 CSV에 없어도 격리 폴더의 미완료 곡은 시작할 때 이어서 끝낸다."""
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"

    state["refine"] = dict(FULL_LLM)
    songs = tmp / "songs.csv"
    songs.write_text("artist,title\n", encoding="utf-8")
    assert crawler.main([str(songs)]) == 0
    assert [r["song_id"] for r in jsonl_records(tmp)] == [SONG_ID]
    assert [f.name for f in (tmp / "raw").iterdir()] == ["가수_밤편지_100"]


# --- 6. 댓글 수 정책: 수집과 임베딩 검증이 같은 기준을 쓴다 -------------------------------

def test_song_with_few_comments_is_collected_and_passes_embedding_validation(pipeline) -> None:
    """멜론 6개·유튜브 0개. 예전에는 수집은 done인데 임베딩에서 failed_raw로 옮겨졌다."""
    crawler, state, calls, tmp = pipeline
    state["reaction"]["comments"] = []
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["crawl_status"]["comment_counts"] == {"melon": 6, "youtube": 0}
    issues = validate_meta_document(record, require_media_files=False)
    assert issues == [], [i.to_dict() for i in issues]


def test_zero_comments_blanks_the_response_summary_and_is_recorded(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["melon"]["melon_comments"] = []
    state["reaction"]["comments"] = []
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["crawl_status"]["comment_counts"] == {"melon": 0, "youtube": 0}
    assert any("댓글 0개" in note for note in record["crawl_status"]["notes"])
    with (tmp / "review_songs.csv").open(encoding="utf-8", newline="") as f:
        assert any("댓글 0개" in row["reason"] for row in csv.DictReader(f))
    # 반응 요약은 비워야 한다. 댓글이 없으면 그 문장은 댓글에서 나올 수 없다.
    assert record["community_feedback"]["sentiment_summary"] == ""
    assert record["community_feedback"]["fans_tags"] == []
    assert validate_meta_document(record, require_media_files=False) == []


def test_comment_count_alone_never_rejects_a_record() -> None:
    """이미 수집한 961곡 중 41곡이 이 규칙 하나로만 탈락했다(다른 문제 없음)."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {"melon": ["새벽에 듣기 좋은 곡", "비 오는 날 생각난다"], "youtube": ["겨울밤 같은 노래"]},
        "community_feedback": {"sentiment_summary": "청자들은 옛 기억을 떠올린다", "fans_tags": ["인생곡"],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    paths = {i.path for i in validate_meta_document(record, require_media_files=False)}
    assert "comments" not in paths


def test_summary_without_any_comment_is_rejected_as_fabricated() -> None:
    """0개 댓글에서 나올 수 없는 문장을 잡는다. 개수 규칙이 못 잡던 경우다."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {"melon": [], "youtube": []},
        "community_feedback": {"sentiment_summary": "청자들은 깊은 위로를 받는다", "fans_tags": ["인생곡"],
                               "major_emotion": "위로", "popularity": {"fame": "Moderate"}},
    }
    issues = [i for i in validate_meta_document(record, require_media_files=False)
              if "sentiment_summary" in i.path]
    assert len(issues) == 1 and "댓글이 0개" in issues[0].reason

    record["community_feedback"] = {"sentiment_summary": "", "fans_tags": []}
    assert [i for i in validate_meta_document(record, require_media_files=False)
            if "sentiment_summary" in i.path or "fans_tags" in i.path] == []


# --- 7. 재수집 결과가 JSONL에 반영된다 ----------------------------------------------------

def test_export_failures_retry_instead_of_discarding_the_folder(pipeline, monkeypatch) -> None:
    """export가 한도를 넘겨 실패해도 모아 둔 폴더를 버리지 않는다.

    export 실패 사유는 JSONL 쓰기 실패와 폴더 이동 실패다. 모아 둔 데이터가 잘못된 게 아니라
    디스크·마운트 문제라서 다시 수집해도 마지막에 똑같이 실패한다. 그런데도 폴더를 버리면
    검증을 통과한 meta.json과 오디오·커버가 사라지고, 재수집이 실행마다 반복돼 Gemini 호출
    스무 번이 매번 다시 나간다. 드라이브가 돌아오면 이동 한 번으로 끝나야 한다.
    """
    crawler, state, calls, tmp = pipeline
    moves = {"n": 0}
    real_move = crawler.shutil.move

    def flaky_move(src, dst):
        moves["n"] += 1
        if moves["n"] <= crawl_state.MAX_STAGE_ATTEMPTS + 1:      # 한도를 넘겨 4번 실패시킨다
            raise OSError("drive offline")
        return real_move(src, dst)

    monkeypatch.setattr(crawler.shutil, "move", flaky_move)
    staged = tmp / "raw_incomplete" / "가수_밤편지_100"

    def assert_folder_survived():
        assert staged.is_dir()
        assert (staged / "meta.json").is_file()
        assert (staged / "audio.m4a").is_file()
        # 다시 수집한 적이 없다 — 멜론·유튜브·LLM 호출은 처음 한 번뿐이다
        assert (calls.melon, calls.youtube, calls.refine) == (1, 1, 1)

    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert_folder_survived()

    # 한도에 닿을 때까지 재개로 실패시킨다
    for _ in range(crawl_state.MAX_STAGE_ATTEMPTS - 1):
        assert crawler.resume_incomplete(crawler.Registry()) == (0, 1)
        assert_folder_survived()

    # 한도를 넘긴 뒤 시드 루프가 같은 곡을 만나도 폴더를 버리지 않는다.
    # (재개를 건너뛴 Registry라 process_song이 직접 재개 경로를 탄다)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert_folder_survived()

    # 드라이브가 돌아오면 이동만 하고 끝난다
    assert crawler.resume_incomplete(crawler.Registry()) == (1, 0)
    assert (calls.melon, calls.youtube, calls.refine) == (1, 1, 1)
    assert not staged.exists()
    records = jsonl_records(tmp)
    assert len(records) == 1, records
    folder = tmp / "raw" / "가수_밤편지_100"
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8")) == records[0]


def test_export_retry_rewrites_the_row_it_already_wrote(pipeline, monkeypatch) -> None:
    """export는 JSONL을 먼저 쓰고 폴더를 옮긴다. 재시도가 '이미 썼으니 건너뛴다'로 판단하면
    같은 실행 안에서 바뀐 meta.json이 JSONL에 반영되지 않는다."""
    crawler, state, calls, tmp = pipeline
    moves = {"n": 0}
    real_move = crawler.shutil.move

    def flaky_move(src, dst):
        moves["n"] += 1
        if moves["n"] == 1:
            raise OSError("drive offline")
        return real_move(src, dst)

    monkeypatch.setattr(crawler.shutil, "move", flaky_move)
    registry = crawler.Registry()
    assert crawler.process_song("가수", "밤편지", registry) == "failed"
    assert len(jsonl_records(tmp)) == 1

    # 사람이 meta.json을 고친 뒤 같은 실행에서 재시도하는 상황
    staged = tmp / "raw_incomplete" / "가수_밤편지_100" / "meta.json"
    record = json.loads(staged.read_text(encoding="utf-8"))
    record["community_feedback"]["sentiment_summary"] = "손으로 고친 요약"
    staged.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    assert crawler.resume_incomplete(registry) == (1, 0)
    records = jsonl_records(tmp)
    assert len(records) == 1, records
    assert records[0]["community_feedback"]["sentiment_summary"] == "손으로 고친 요약"


def test_recrawl_replaces_the_stale_jsonl_record(pipeline, monkeypatch) -> None:
    """처음부터 다시 수집한 곡의 JSONL 줄이 예전 분석 그대로 남으면 안 된다.

    export가 JSONL을 쓴 뒤 이동이 실패했고, 그 사이 격리 폴더가 사라진 상황이다(드라이브 정리,
    수동 삭제). 다음 실행은 처음부터 다시 수집하고, 같은 ID라는 이유로 기록을 건너뛰면
    meta.json만 새 분석이고 JSONL은 예전 분석으로 굳는다.
    """
    crawler, state, calls, tmp = pipeline
    moves = {"n": 0}
    real_move = crawler.shutil.move

    def flaky_move(src, dst):
        moves["n"] += 1
        if moves["n"] == 1:
            raise OSError("drive offline")
        return real_move(src, dst)

    monkeypatch.setattr(crawler.shutil, "move", flaky_move)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert [r["community_feedback"]["sentiment_summary"] for r in jsonl_records(tmp)] == ["댓글 요약"]

    shutil.rmtree(tmp / "raw_incomplete" / "가수_밤편지_100")

    state["refine"]["sentiment_summary"] = "새로 분석한 댓글 요약"
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    records = jsonl_records(tmp)
    assert len(records) == 1, records
    assert records[0]["community_feedback"]["sentiment_summary"] == "새로 분석한 댓글 요약"
    folder = (tmp / "raw" / "가수_밤편지_100")
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8")) == records[0]


# --- 검색 2위 이하로 채택된 곡의 재개 -------------------------------------------------

def two_candidates(crawler, monkeypatch, adopted=SONG_ID, rejected="999"):
    """검색 1위는 상세 검사에서 탈락하고 2위가 채택되는 상황.

    process_song이 보는 best_id는 rejected이고, collect_melon_data가 돌려주는 ID는 adopted다.
    """
    from src.crawler.scripts_py.melon_match import MatchScore

    monkeypatch.setattr(crawler, "fetch_melon_song_candidates", lambda a, t: [
        SongCandidate(song_id=rejected, title="밤편지", match=MatchScore(score=100)),
        SongCandidate(song_id=adopted, title="밤편지", match=MatchScore(score=95)),
    ])


def audio_carrying(url: str) -> bytes:
    """어느 영상에서 받은 오디오인지 파일 안에 남긴다(출처 대조용)."""
    return m4a_bytes() + url.encode("utf-8")


def test_second_ranked_song_resumes_instead_of_recollecting(pipeline, monkeypatch) -> None:
    """2위로 채택된 곡이 진행 중이면 그 상태를 이어간다. 폴더를 덮어쓰면 안 된다.

    재개 검사를 검색 1위(best_id)로만 하면 이 곡을 놓친다. 놓치면 새 원천 정보로 폴더를
    덮어쓰는데, _download_audio는 정상 audio.m4a를 다시 받지 않으므로 meta.json은 새 영상을
    가리키고 오디오는 예전 영상인 레코드가 남는다.
    """
    crawler, state, calls, tmp = pipeline
    two_candidates(crawler, monkeypatch)
    monkeypatch.setattr(crawler, "download_youtube_audio",
                        lambda url, path: bool(Path(path).write_bytes(audio_carrying(url))) or True)

    first_url = state["reaction"]["video_url"]
    real_move = crawler.shutil.move

    # 1회차: 폴더 이동만 실패해 export 단계에서 멈춘다
    def broken_move(src, dst):
        raise OSError("drive offline")

    monkeypatch.setattr(crawler.shutil, "move", broken_move)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    staged = tmp / "raw_incomplete" / f"가수_밤편지_{SONG_ID}"
    assert staged.is_dir()
    assert (staged / crawl_state.AUDIO_FILE).read_bytes().endswith(first_url.encode())
    assert (calls.melon, calls.youtube, calls.refine) == (1, 1, 1)

    # 2회차: 드라이브는 돌아왔고, 그 사이 유튜브 검색 결과와 분석이 달라졌다고 가정한다.
    # monkeypatch.undo()는 fixture가 걸어 둔 패치까지 되돌리므로 shutil.move만 복구한다.
    monkeypatch.setattr(crawler.shutil, "move", real_move)
    state["reaction"]["video_url"] = "https://www.youtube.com/watch?v=OTHER"
    state["reaction"]["comment_source_url"] = "https://www.youtube.com/watch?v=OTHER"
    state["refine"]["sentiment_summary"] = "다시 분석한 요약"

    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    # 재개다 — 유튜브 재검색도 최종 LLM도 다시 돌지 않았다.
    # 멜론 상세는 한 번 더 나간다: 채택 ID를 알려면 상세를 봐야 해서 피할 수 없다.
    assert (calls.youtube, calls.refine) == (1, 1)
    assert calls.melon == 2

    record = jsonl_records(tmp)[0]
    folder = tmp / "raw" / f"가수_밤편지_{SONG_ID}"
    # 오디오 출처와 메타데이터가 같은 영상을 가리킨다
    assert record["links"]["youtube_url"] == first_url
    assert (folder / crawl_state.AUDIO_FILE).read_bytes().endswith(first_url.encode())
    assert record["community_feedback"]["sentiment_summary"] == "댓글 요약"


def test_melon_title_change_does_not_discard_the_folder(pipeline, monkeypatch) -> None:
    """멜론 표기가 바뀌어도 같은 song_id의 진행 중인 폴더를 이어간다.

    폴더 이름은 {가수}_{제목}_{id}라서 제목 띄어쓰기만 달라져도 새 이름이 나온다. 그 이유로
    폴더를 지우면 검증을 끝낸 분석과 오디오를 잃고 유튜브·LLM을 다시 부른다. 새 분석이
    실패하면 멀쩡하던 예전 분석까지 사라진다. 곡의 정체는 song_id이지 폴더 이름이 아니다.
    """
    crawler, state, calls, tmp = pipeline
    two_candidates(crawler, monkeypatch)
    real_move = crawler.shutil.move

    def broken_move(src, dst):
        raise OSError("drive offline")

    # 1회차: 분석까지 끝내고 폴더 이동만 실패한다
    monkeypatch.setattr(crawler.shutil, "move", broken_move)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    staged = tmp / "raw_incomplete" / f"가수_밤편지_{SONG_ID}"
    assert (staged / crawl_state.META_FILE).is_file()
    assert (calls.melon, calls.youtube, calls.refine) == (1, 1, 1)

    # 2회차: 멜론이 제목 표기를 바꿨다
    monkeypatch.setattr(crawler.shutil, "move", real_move)
    state["melon"]["title"] = "밤 편지"

    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    # 재개다 — 유튜브도 LLM도 다시 부르지 않았다
    assert (calls.youtube, calls.refine) == (1, 1)
    # 새 이름으로 폴더를 만들지 않고, 이어받은 폴더 그대로 내보낸다
    assert not (tmp / "raw_incomplete").exists() or not list((tmp / "raw_incomplete").iterdir())
    assert [p.name for p in (tmp / "raw").iterdir()] == [f"가수_밤편지_{SONG_ID}"]
    assert len(jsonl_records(tmp)) == 1


def test_second_ranked_song_already_tried_this_run_is_not_recollected(pipeline, monkeypatch) -> None:
    """이번 실행에서 재개를 시도한 곡은 같은 단계를 두 번 시도하지 않는다.

    이 검사가 없으면 한 실행 안에서 같은 단계가 두 번 실패한다(재개 + 시드 루프). 실패 횟수가
    두 배로 쌓여 MAX_STAGE_ATTEMPTS를 실행 한 번에 소진하고, LLM 호출도 한 번 더 나간다.
    """
    crawler, state, calls, tmp = pipeline
    two_candidates(crawler, monkeypatch)

    tried = []

    def always_failing_analysis(state_, source):
        tried.append(state_.song_id)
        return "LLM Analysis Failed"

    monkeypatch.setattr(crawler, "stage_analysis", always_failing_analysis)
    registry = crawler.Registry()
    assert crawler.process_song("가수", "밤편지", registry) == "failed"
    assert (calls.melon, calls.youtube) == (1, 1)
    assert tried == [SONG_ID]

    # 같은 Registry로 한 번 더 만나도 원천 수집을 반복하지 않고, analysis도 다시 돌지 않는다
    assert crawler.process_song("가수", "밤편지", registry) == "failed"
    assert calls.youtube == 1
    assert tried == [SONG_ID]
    staged = crawl_state.CrawlState.load(tmp / "raw_incomplete" / f"가수_밤편지_{SONG_ID}")
    assert staged is not None
    assert staged.attempts("analysis") == 1


def test_second_ranked_song_over_the_limit_is_recollected_from_scratch(pipeline, monkeypatch) -> None:
    """한도를 넘긴 단계는 2위 채택곡도 폴더를 버리고 처음부터 수집한다."""
    crawler, state, calls, tmp = pipeline
    two_candidates(crawler, monkeypatch)
    state["refine"] = {}
    for _ in range(crawl_state.MAX_STAGE_ATTEMPTS):
        assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    # 한도에 닿기 전에는 analysis만 다시 한다 — 유튜브 재검색은 없다
    assert calls.youtube == 1

    state["refine"] = dict(FULL_LLM)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert calls.youtube == 2        # 폴더를 버리고 원천부터 다시 모았다


def test_only_recollectable_stages_discard_the_folder() -> None:
    """어느 단계가 폐기 대상인지 한 곳에서 고정한다."""
    def state_stuck_at(stage):
        done = {s: {"status": "done"} for s in crawl_state.STAGES[:crawl_state.STAGES.index(stage)]}
        done[stage] = {"status": "failed", "attempts": crawl_state.MAX_STAGE_ATTEMPTS}
        return crawl_state.CrawlState(song_id="1", input_artist="가", input_title="곡",
                                      folder=Path("x"), stages=done)

    for stage in ("source", "assets", "analysis"):
        st = state_stuck_at(stage)
        assert st.needs_recollect() == stage
        assert st.exhausted() == stage

    export = state_stuck_at("export")
    assert export.needs_recollect() is None      # 폴더를 버리지 않는다
    assert export.exhausted() == "export"        # 사람에게는 알린다


def test_replace_keeps_other_lines_byte_for_byte(tmp_path) -> None:
    path = tmp_path / "all_songs.jsonl"
    others = ['{"song_id": "1", "note": "그대로"}', '{"song_id": "3", "note": "그대로"}']
    path.write_text("\n".join([others[0], '{"song_id": "2", "note": "낡음"}', others[1]]) + "\n",
                    encoding="utf-8")

    assert crawl_state.replace_jsonl_record(path, {"song_id": "2", "note": "새것"}) is True
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [lines[0], lines[2]] == others
    assert json.loads(lines[1]) == {"song_id": "2", "note": "새것"}

    # 없는 ID면 덧붙인다
    assert crawl_state.replace_jsonl_record(path, {"song_id": "9"}) is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_export_does_not_rewrite_the_file_for_new_songs(pipeline, monkeypatch) -> None:
    """새 곡은 덧붙인다. 곡마다 파일을 다시 쓰면 5,000곡에서 비용이 제곱으로 는다."""
    crawler, state, calls, tmp = pipeline
    calls_to_replace = []
    monkeypatch.setattr(crawler, "replace_jsonl_record",
                        lambda path, record: calls_to_replace.append(record) or True)
    registry = crawler.Registry()
    assert crawler.process_song("가수", "밤편지", registry) == "done"
    state["melon"]["id"] = "200"
    assert crawler.process_song("가수", "밤편지", registry) == "done"
    assert calls_to_replace == []


# --- 8. 재개 중 예외가 배치를 멈추지 않는다 -----------------------------------------------

def test_stage_exception_is_recorded_as_a_stage_failure(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline

    def exploding_analysis(state_arg, source):
        raise OSError("read-only file system")

    monkeypatch.setattr(crawler, "stage_analysis", exploding_analysis)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"

    status = json.loads((tmp / "raw_incomplete" / "가수_밤편지_100" / "crawl_status.json")
                        .read_text(encoding="utf-8"))
    analysis = status["stages"]["analysis"]
    assert analysis["status"] == "failed"
    assert analysis["attempts"] == 1          # 예전에는 늘지 않아 영원히 같은 단계에서 멈췄다
    assert "read-only file system" in analysis["error"]
    assert any("Exception" in r for r in failed_reasons(tmp))


def test_one_exploding_song_does_not_stop_the_other_resumes(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline

    # 분석 단계에서 멈춘 미완료 폴더 두 개를 만든다
    state["refine"] = {}
    for song_id in ("100", "200"):
        state["melon"]["id"] = song_id
        assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert len(list((tmp / "raw_incomplete").iterdir())) == 2

    state["refine"] = dict(FULL_LLM)
    real_analysis = crawler.stage_analysis

    def explode_for_the_first_song(state_arg, source):
        if state_arg.song_id == "100":
            raise RuntimeError("디스크 오류")
        return real_analysis(state_arg, source)

    monkeypatch.setattr(crawler, "stage_analysis", explode_for_the_first_song)
    registry = crawler.Registry()
    assert crawler.resume_incomplete(registry) == (1, 1)

    # 터진 곡은 격리 폴더에 실패로 남고, 나머지 곡은 끝까지 갔다
    assert [r["song_id"] for r in jsonl_records(tmp)] == ["200"]
    assert [f.name for f in (tmp / "raw").iterdir()] == ["가수_밤편지_200"]
    status = json.loads((tmp / "raw_incomplete" / "가수_밤편지_100" / "crawl_status.json")
                        .read_text(encoding="utf-8"))
    assert status["stages"]["analysis"]["attempts"] == 2


def test_failure_bookkeeping_error_does_not_stop_the_batch(pipeline, monkeypatch) -> None:
    """실패를 기록하다 예외가 나도(상태 파일 쓰기 실패 등) 남은 곡의 재개는 계속한다.

    advance()가 단계 예외를 흡수하므로 resume_incomplete의 방어는 이 경로에서만 쓰인다.
    """
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}
    for song_id in ("100", "200"):
        state["melon"]["id"] = song_id
        assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"

    state["refine"] = dict(FULL_LLM)
    real_save = crawl_state.CrawlState.save

    def save_that_fails_for_the_first_song(self):
        if self.song_id == "100":
            raise OSError("상태 파일을 쓸 수 없음")
        return real_save(self)

    monkeypatch.setattr(crawl_state.CrawlState, "save", save_that_fails_for_the_first_song)
    registry = crawler.Registry()
    assert crawler.resume_incomplete(registry) == (1, 1)
    assert [r["song_id"] for r in jsonl_records(tmp)] == ["200"]


# --- 2차 검증에서 확인된 것들 --------------------------------------------------------------

def test_rebuild_refuses_to_empty_the_file_when_folders_are_missing(tmp_path) -> None:
    """수집 폴더는 보통 드라이브에 있다. 마운트가 안 된 상태로 재생성하면 파일이 0바이트가 된다."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    jsonl = tmp_path / "all_songs.jsonl"
    original = "".join(json.dumps({"song_id": str(i)}, ensure_ascii=False) + "\n" for i in range(10))
    jsonl.write_text(original, encoding="utf-8")

    with pytest.raises(crawl_state.RebuildRefused):
        crawl_state.rebuild_jsonl(data_dir, jsonl)
    assert jsonl.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.bak")) == []

    # 절반 넘게 줄어드는 경우도 거절한다
    folder = data_dir / "가수_곡_1"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"song_id": "1"}), encoding="utf-8")
    (folder / "cover.jpg").write_bytes(jpeg_bytes())
    (folder / "audio.m4a").write_bytes(m4a_bytes())
    with pytest.raises(crawl_state.RebuildRefused):
        crawl_state.rebuild_jsonl(data_dir, jsonl)
    assert jsonl.read_text(encoding="utf-8") == original

    # --force면 진행한다
    assert crawl_state.rebuild_jsonl(data_dir, jsonl, force=True) == 1
    assert [r["song_id"] for r in crawl_state.iter_jsonl(jsonl)] == ["1"]


def test_main_rebuild_exits_with_an_error_instead_of_emptying(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    (tmp / "raw").mkdir(exist_ok=True)
    (tmp / "all_songs.jsonl").write_text('{"song_id": "1"}\n', encoding="utf-8")
    assert crawler.main(["--rebuild-jsonl"]) == 3
    assert (tmp / "all_songs.jsonl").read_text(encoding="utf-8") == '{"song_id": "1"}\n'


def test_comment_selection_outage_is_retryable_not_a_zero_comment_song(pipeline, monkeypatch) -> None:
    """Gemini 장애를 '댓글 0개'로 저장하면 곡이 영구히 그 상태로 굳는다."""
    crawler, state, calls, tmp = pipeline

    working_reaction = crawler.fetch_youtube_reaction

    def unavailable(artist, title, song_lyrics="", melon_title="", known_artists=None):
        raise CommentSelectionUnavailable("YouTube", 25, 2)

    monkeypatch.setattr(crawler, "fetch_youtube_reaction", unavailable)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"

    assert jsonl_records(tmp) == []
    assert not (tmp / "raw").exists() or not list((tmp / "raw").iterdir())
    assert any("Comment Selection Unavailable" in r for r in failed_reasons(tmp))

    # 멜론까지의 결과는 남아 있다. 다시 받지 않기 위해서다.
    staged = tmp / "raw_incomplete" / "가수_밤편지_100"
    source = json.loads((staged / "source.json").read_text(encoding="utf-8"))
    assert source["melon"]["melon_comments"]      # 댓글 LLM 선별 결과까지 저장됨
    assert "reaction" not in source

    # 다음 실행에서 되면 정상 수집된다 (undo()는 픽스처의 패치까지 되돌리므로 쓰지 않는다)
    monkeypatch.setattr(crawler, "fetch_youtube_reaction", working_reaction)
    before = calls.melon
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert calls.melon == before                  # 멜론 수집을 다시 하지 않았다
    assert len(jsonl_records(tmp)[0]["comments"]["youtube"]) == 6


def test_melon_transient_error_is_retryable(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline

    def transient(artist, title, candidates=None, already_collected=None):
        raise MelonTransientError("멜론 요청 재시도 초과: HTTP 503")

    monkeypatch.setattr(crawler, "collect_melon_data", transient)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert any("Melon Transient Error" in r for r in failed_reasons(tmp))
    assert jsonl_records(tmp) == []


def test_real_comment_text_is_not_mistaken_for_an_llm_failure_phrase() -> None:
    """'댓글이 없다니', '가사가 없어서'는 흔한 한국어 댓글이다. 961곡 중 5곡이 이 사유로만 탈락했다."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {
            "melon": ["아니 이노래에 왜댓글이없지", "댓글이 없다니.. 아련하게 떠오르는 추억",
                      "가사가 없어서 종이에 받아적었던 기억"],
            "youtube": ["요즘은 이런 가사가없는 음악들뿐이라 그립다"],
        },
        "community_feedback": {"sentiment_summary": "청자들은 옛 기억을 떠올린다", "fans_tags": ["인생곡"],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    assert issue_paths(record, "comments", "community_feedback") == []

    # LLM이 생성하는 필드에서는 같은 문구가 여전히 실패로 잡힌다
    record["community_feedback"]["sentiment_summary"] = "댓글이 없어 알 수 없습니다"
    assert issue_paths(record, "comments", "community_feedback") == ["community_feedback.sentiment_summary"]


@pytest.mark.parametrize(
    "path, value",
    [("vocal_gender", "unknown"), ("type", ["unknown"])],
)
def test_unknown_is_a_normal_value_for_gender_and_type(path, value) -> None:
    """프롬프트가 직접 'unknown'을 내라고 지시하는 두 필드다. 이걸 탈락시키면 수집 done인
    곡이 임베딩에서 빠진다 — 댓글 수 규칙과 같은 어긋남이다."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"], path: value},
        "comments": {"melon": ["새벽에 듣기 좋은 곡"], "youtube": []},
        "community_feedback": {"sentiment_summary": "요약", "fans_tags": ["인생곡"],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    assert issue_paths(record, f"metadata.{path}") == []

    # 다른 필드의 'unknown'은 계속 실패 문구다
    record["lyrics_data"] = {"lyrics_summary": "unknown"}
    assert issue_paths(record, "lyrics_data.lyrics_summary") == ["lyrics_data.lyrics_summary"]


def test_lyric_less_song_is_not_collected(pipeline) -> None:
    """가사 없는 곡은 채택하지 않는다. 저장해도 임베딩 검증이 빈 full_lyrics를 거부해
    수집 done인데 색인에 없는 레코드가 된다(코퍼스 8곡, 전부 19금)."""
    crawler, state, calls, tmp = pipeline
    state["melon"]["lyrics"] = "   "
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert jsonl_records(tmp) == []
    assert not (tmp / "raw").exists() or not list((tmp / "raw").iterdir())
    assert any("No Lyrics" in r for r in failed_reasons(tmp))
    assert calls.refine == 0      # 가사가 없으면 Gemini를 부르지 않는다


def test_validation_gap_is_recorded_at_collection_time(pipeline) -> None:
    """임베딩 검증에 걸릴 레코드는 수집 시점에 그 사실을 남긴다.

    필수 필드(가사 요약)가 빈 경우를 쓴다. 앨범 텍스트는 더 이상 탈락 기준이 아니다
    (test_album_text_is_not_required 참고).
    """
    crawler, state, calls, tmp = pipeline
    state["refine"]["lyrics_summary"] = ""
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    with (tmp / "review_songs.csv").open(encoding="utf-8", newline="") as f:
        reasons = [row["reason"] for row in csv.DictReader(f)]
    assert any("임베딩 검증 탈락 예상" in r and "lyrics_summary" in r for r in reasons)
    record = jsonl_records(tmp)[0]
    assert any("임베딩 검증 탈락 예상" in note for note in record["crawl_status"]["notes"])


# --- 앨범 텍스트는 필수가 아니다 --------------------------------------------------------

def test_album_text_is_not_required(pipeline) -> None:
    """앨범 소개와 요약이 둘 다 비어도 곡은 색인에 들어간다.

    회귀 이력: 둘 중 하나라도 유효해야 통과하는 결합 규칙이 있었다. 2000년대 이전 앨범은
    멜론에 소개가 없는 경우가 흔해서(952곡 중 81곡), LLM이 요약까지 못 쓴 2곡이 통째로
    빠졌다. 앨범 설명이 빠지는 것이 곡이 빠지는 것보다 낫다.
    """
    crawler, state, calls, tmp = pipeline
    state["melon"]["album_desc"] = ""
    state["refine"]["album_summary"] = ""
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["metadata"]["album_description"] == ""
    assert record["metadata"]["album_summary"] == ""
    assert issue_paths(record, "metadata.album") == []
    # 앨범 때문에 검토 목록에 올라가지도 않는다
    notes = record["crawl_status"].get("notes") or []
    assert not any("album" in note for note in notes)


def test_album_text_failure_phrase_is_blanked_not_rejected(pipeline) -> None:
    """LLM이 "정보가 제공되지 않아..."라고 답하면 그 자리를 비운다.

    그 문장이 남으면 dense 패시지의 앨범 설명 자리에 들어가 엉뚱한 질의와 매칭된다.
    정직하게 답한 곡이 손해를 보지 않게, 지어낸 값 대신 빈 값으로 만든다(댓글 필드와 같은 원칙).
    """
    crawler, state, calls, tmp = pipeline
    state["melon"]["album_desc"] = ""
    state["refine"]["album_summary"] = (
        "정보가 제공되지 않아 이 노래에 대한 앨범 상세 정보는 확인할 수 없습니다."
    )
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["metadata"]["album_summary"] == ""
    assert issue_paths(record, "metadata.album") == []


def test_blank_album_text_only_touches_the_llm_written_field() -> None:
    """비우는 대상은 album_summary뿐이다. album_description은 건드리지 않는다.

    회귀 이력: 처음에는 두 필드를 모두 검사했다. album_description은 멜론에서 긁어온 사람 글이라
    LLM 실패 문구일 수 없는데, 실패 문구 목록(짧은 LLM 답변용)을 긴 평론에 적용하니 나얼
    '같은 시간 속의 너'의 1,318자 평론이 "이견을 찾기 어렵다"(칭찬) 때문에 통째로 지워졌다.

    build_record는 album_description을 melon_data에서 따로 채우므로 이 함수만이 방어선이다
    (일회성 정리 스크립트 등 다른 호출부가 metadata 전체를 넘길 수 있다).
    """
    from src.crawler.scripts_py.refine_data import blank_album_text_without_evidence

    praise = "나얼의 가창에는 이견을 찾기 어렵다. 소울의 원형을 좇는 행보가 앨범의 가치를 높인다."
    out = blank_album_text_without_evidence({
        "album_description": praise,
        "album_summary": "정보가 제공되지 않아 확인할 수 없습니다.",
    })
    assert out["album_description"] == praise      # 사람이 쓴 글은 그대로
    assert out["album_summary"] == ""              # LLM 실패 문구는 비운다

    # 쓸 만한 요약은 그대로 둔다
    keep = blank_album_text_without_evidence({"album_summary": "잔잔한 발라드 모음."})
    assert keep["album_summary"] == "잔잔한 발라드 모음."


def test_long_melon_album_description_is_never_blanked(pipeline) -> None:
    """멜론에서 긁어온 앨범 소개는 실패 문구 검사를 하지 않는다.

    회귀 이력: 실패 문구 목록은 짧은 LLM 답변("정보를 찾기 어렵습니다")을 잡으려고 만든 것이다.
    그것을 긴 평론에 그대로 적용하니, 나얼 '같은 시간 속의 너'의 1,318자 평론에 있는
    "그의 가창에는 이견을 찾기 어렵다"(칭찬)가 '찾기어렵'에 걸려 소개 전문이 지워졌다.
    album_description은 사람이 쓴 글이라 LLM 실패 문구일 수가 없다.
    """
    crawler, state, calls, tmp = pipeline
    praise = (
        "브라운아이드소울 싱글 프로젝트의 시작! 나얼의 가창에는 이견을 찾기 어렵다. "
        "소울의 원형과 아날로그를 좇는 행보가 앨범의 가치를 높인다."
    )
    state["melon"]["album_desc"] = praise
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["metadata"]["album_description"] == praise
    assert issue_paths(record, "metadata.album") == []


def test_usable_album_summary_survives_without_a_description(pipeline) -> None:
    """멜론에 앨범 소개가 없어도 쓸 만한 요약은 그대로 둔다(코퍼스 79곡의 경우)."""
    crawler, state, calls, tmp = pipeline
    state["melon"]["album_desc"] = ""
    state["refine"]["album_summary"] = "잔잔한 발라드를 모은 앨범으로 새벽 감성이 짙다."
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    record = jsonl_records(tmp)[0]
    assert record["metadata"]["album_summary"] == "잔잔한 발라드를 모은 앨범으로 새벽 감성이 짙다."


def test_no_validation_gap_note_for_a_clean_record(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    assert "notes" not in jsonl_records(tmp)[0]["crawl_status"]


# --- 댓글 수 정책의 경계를 고정한다 (검증에서 세 가지 재도입이 통과했다) ----------------------

@pytest.mark.parametrize("melon, youtube", [(1, 0), (0, 1), (2, 0), (1, 1), (2, 4)])
def test_any_nonzero_comment_total_passes(melon, youtube) -> None:
    """합계가 1개여도 탈락시키지 않는다. 옛 규칙(10개)이 961곡 중 41곡을 떨어뜨렸다."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {"melon": ["새벽에 듣기 좋은 곡"] * melon, "youtube": ["겨울밤 같은 노래"] * youtube},
        "community_feedback": {"sentiment_summary": "청자들은 옛 기억을 떠올린다", "fans_tags": ["인생곡"],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    assert issue_paths(record, "comments", "community_feedback") == []


def test_blank_response_summary_is_only_allowed_when_there_are_no_comments() -> None:
    """댓글이 있는데 반응 요약이 비어 있으면 분석이 빠진 것이다. 허용 대상이 아니다."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {"melon": ["새벽에 듣기 좋은 곡"], "youtube": []},
        "community_feedback": {"sentiment_summary": "", "fans_tags": [],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    assert issue_paths(record, "community_feedback", "comments") == [
        "community_feedback.fans_tags", "community_feedback.sentiment_summary",
    ]


def test_both_comment_sources_count_toward_the_total() -> None:
    """멜론만 세면 '멜론 0 + 유튜브 N' 레코드가 근거 없는 요약으로 잘못 탈락한다(코퍼스 7곡)."""
    record = {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "comments": {"melon": [], "youtube": ["겨울밤 같은 노래", "비 오는 새벽에 듣는다"]},
        "community_feedback": {"sentiment_summary": "청자들은 겨울을 떠올린다", "fans_tags": ["인생곡"],
                               "major_emotion": "추억", "popularity": {"fame": "Moderate"}},
    }
    counts, total = count_comments(record)
    assert (counts["comments.melon"], counts["comments.youtube"], total) == (0, 2, 2)
    assert issue_paths(record, "comments", "community_feedback") == []


# --- 재개 경로도 수집 정책을 거친다 -----------------------------------------------------------
# 가사 검사가 collect_source에만 있으면, 정책 이전에 만들어진 미완료 폴더를 재개할 때 검사 없이
# 통과해 JSONL까지 간다.

def staged_folder_with(tmp, crawler, source_patch: dict, with_meta: bool = False):
    """분석 단계에서 멈춘 미완료 폴더를 만든 뒤 source.json을 바꿔 놓는다(정책 이전 데이터 흉내)."""
    folder = tmp / "raw_incomplete" / "가수_밤편지_100"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    source.update(source_patch)
    (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    return folder


def test_resume_applies_the_collection_policy(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"   # 분석에서 멈춤
    folder = staged_folder_with(tmp, crawler, {"lyrics": ""})
    assert folder.exists()

    state["refine"] = dict(FULL_LLM)
    registry = crawler.Registry()
    assert crawler.resume_incomplete(registry) == (0, 1)

    assert jsonl_records(tmp) == []
    assert not folder.exists()                       # 다시 수집해도 같은 이유로 걸린다
    assert not (tmp / "raw").exists() or not list((tmp / "raw").iterdir())
    assert any("[policy] No Lyrics" in r for r in failed_reasons(tmp))


def test_resume_applies_the_policy_to_adult_marked_source(pipeline) -> None:
    crawler, state, calls, tmp = pipeline
    state["refine"] = {}
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    folder = tmp / "raw_incomplete" / "가수_밤편지_100"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    source["melon"]["is_adult"] = True
    (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    state["refine"] = dict(FULL_LLM)
    assert crawler.resume_incomplete(crawler.Registry()) == (0, 1)
    assert jsonl_records(tmp) == []
    assert any("[policy] Adult Only" in r for r in failed_reasons(tmp))


def test_export_refuses_a_record_that_violates_the_policy(pipeline) -> None:
    """정책 이전에 분석까지 끝난 폴더. meta.json은 있지만 가사가 없다."""
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"
    folder = tmp / "raw" / "가수_밤편지_100"
    record = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    record["lyrics_data"]["full_lyrics"] = ""
    assert crawler.record_policy_error(record).startswith("No Lyrics")

    # export 단계는 이 레코드를 JSONL에 넣지 않는다
    state_obj = crawl_state.CrawlState(song_id="200", input_artist="가수", input_title="곡",
                                       folder=tmp / "raw_incomplete" / "가수_곡_200")
    state_obj.folder.mkdir(parents=True)
    (state_obj.folder / "meta.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    error = crawler.stage_export(state_obj, crawler.Registry())
    assert error.startswith("Policy: No Lyrics")


def test_duplicate_jsonl_rows_collapse_to_one_current_row(tmp_path) -> None:
    """같은 ID가 여러 줄이면 전부 걷어내고 한 줄로 합친다.

    첫 줄만 바꾸면 뒤에 남은 예전 줄 때문에, 마지막 줄을 쓰는 감사 도구가 갱신 뒤에도
    옛 데이터를 읽는다.
    """
    path = tmp_path / "all_songs.jsonl"
    path.write_text("\n".join([
        json.dumps({"song_id": "1", "v": "old-1"}, ensure_ascii=False),
        json.dumps({"song_id": "2", "v": "keep"}, ensure_ascii=False),
        json.dumps({"song_id": "1", "v": "old-2"}, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")

    assert crawl_state.replace_jsonl_record(path, {"song_id": "1", "v": "new"}) is True
    records = list(crawl_state.iter_jsonl(path))
    assert [r["song_id"] for r in records] == ["1", "2"]
    assert [r["v"] for r in records if r["song_id"] == "1"] == ["new"]

    # 감사 도구(마지막 줄 사용)도 새 값을 읽는다
    from src.crawler.scripts_py.audit_collection import load_records
    loaded = {r["song_id"]: r["v"] for r in load_records(path)}
    assert loaded == {"1": "new", "2": "keep"}


def test_replace_keeps_unparsable_lines(tmp_path) -> None:
    path = tmp_path / "all_songs.jsonl"
    path.write_text('{"song_id": "1"}\n깨진 줄\n{"song_id": "1", "v": "old"}\n', encoding="utf-8")
    crawl_state.replace_jsonl_record(path, {"song_id": "1", "v": "new"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "깨진 줄" in lines
    assert sum(1 for l in lines if '"song_id": "1"' in l) == 1


# --- 제외 대상은 JSONL에서도 지운다 ---------------------------------------------------------
# 정책 변경 전 실행에서 JSONL 기록은 성공하고 폴더 이동만 실패한 곡이 있다. 폴더만 지우면
# 그 행이 남아 감사·평가·카탈로그에 제외 대상이 계속 전달된다.

def _staged_with_jsonl_row(pipeline, monkeypatch):
    """JSONL 행은 있고 폴더 이동만 실패한 상태를 만든다(정책 변경 전 실행)."""
    crawler, state, calls, tmp = pipeline
    real_move = crawler.shutil.move

    def broken_move(src, dst):
        raise OSError("drive offline")

    monkeypatch.setattr(crawler.shutil, "move", broken_move)
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "failed"
    assert [r["song_id"] for r in jsonl_records(tmp)] == [SONG_ID]
    # undo()는 픽스처의 패치(DATA_DIR·JSONL_FILE 등)까지 되돌리므로 쓰지 않는다
    monkeypatch.setattr(crawler.shutil, "move", real_move)
    return tmp / "raw_incomplete" / "가수_밤편지_100"


def test_discarding_a_policy_violation_also_removes_the_jsonl_row(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline
    folder = _staged_with_jsonl_row(pipeline, monkeypatch)

    # 정책 이전 데이터를 흉내 낸다: 가사가 빈 source.json
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    source["lyrics"] = ""
    (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    registry = crawler.Registry()
    assert registry.jsonl_ids == {SONG_ID}
    assert crawler.resume_incomplete(registry) == (0, 1)

    assert jsonl_records(tmp) == []                 # 행이 남으면 후속 작업에 계속 전달된다
    assert registry.jsonl_ids == set()              # 메모리 상태도 같이 정리한다
    assert SONG_ID not in registry.complete_ids
    assert SONG_ID not in registry.incomplete
    assert not folder.exists()
    assert any("[policy] No Lyrics" in r for r in failed_reasons(tmp))


def test_jsonl_cleanup_failure_keeps_the_folder_for_a_retry(pipeline, monkeypatch) -> None:
    """정리가 실패하면 폴더를 남긴다. 먼저 지우면 재시도할 근거가 사라진다."""
    crawler, state, calls, tmp = pipeline
    folder = _staged_with_jsonl_row(pipeline, monkeypatch)
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    source["lyrics"] = ""
    (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    def broken_remove(path, song_id):
        raise OSError("read-only file system")

    monkeypatch.setattr(crawler, "remove_jsonl_record", broken_remove)
    registry = crawler.Registry()
    assert crawler.resume_incomplete(registry) == (0, 1)

    assert folder.exists()                          # 다음 실행에서 다시 시도한다
    assert [r["song_id"] for r in jsonl_records(tmp)] == [SONG_ID]
    assert any("JSONL 정리 실패" in r for r in failed_reasons(tmp))
    status = json.loads((folder / "crawl_status.json").read_text(encoding="utf-8"))
    assert "Policy cleanup failed" in status["stages"]["export"]["error"]

    # 정리가 되면 그때 지워진다
    monkeypatch.setattr(crawler, "remove_jsonl_record", crawl_state.remove_jsonl_record)
    assert crawler.resume_incomplete(crawler.Registry()) == (0, 1)
    assert not folder.exists()
    assert jsonl_records(tmp) == []


def test_remove_jsonl_record_drops_every_matching_row(tmp_path) -> None:
    path = tmp_path / "all_songs.jsonl"
    path.write_text("\n".join([
        json.dumps({"song_id": "1", "v": "a"}, ensure_ascii=False),
        json.dumps({"song_id": "2"}, ensure_ascii=False),
        "깨진 줄",
        json.dumps({"song_id": "1", "v": "b"}, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")

    assert crawl_state.remove_jsonl_record(path, "1") == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [r["song_id"] for r in crawl_state.iter_jsonl(path)] == ["2"]
    assert "깨진 줄" in lines                        # 읽지 못하는 줄은 손대지 않는다
    assert crawl_state.remove_jsonl_record(path, "1") == 0
    assert crawl_state.remove_jsonl_record(tmp_path / "없는파일.jsonl", "1") == 0


def test_sync_does_not_resurrect_a_policy_excluded_record(tmp_path) -> None:
    """정책 이전에 모은 완료 폴더가 남아 있으면, 검사 없이는 동기화가 JSONL에 되살린다."""
    data_dir = tmp_path / "raw"
    jsonl = tmp_path / "all_songs.jsonl"
    for song_id, lyrics in (("1", "가사 본문"), ("2", "")):
        folder = data_dir / f"가수_곡_{song_id}"
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(
            json.dumps({"song_id": song_id, "lyrics_data": {"full_lyrics": lyrics}}, ensure_ascii=False),
            encoding="utf-8")
        (folder / "cover.jpg").write_bytes(jpeg_bytes())
        (folder / "audio.m4a").write_bytes(m4a_bytes())

    added, orphans = crawl_state.sync_jsonl(data_dir, jsonl, skip_record=crawler.record_policy_error)
    assert (added, orphans) == (1, 0)
    assert [r["song_id"] for r in crawl_state.iter_jsonl(jsonl)] == ["1"]


# --- 낭비 제거 (같은 결과, 더 적은 요청) -----------------------------------------------------

def test_completed_song_is_skipped_without_searching(pipeline, monkeypatch) -> None:
    """완료곡은 검색도 대기도 하지 않는다. 예전에는 검색 1회와 2~5초 대기를 썼다."""
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    searches = []
    real_search = crawler.fetch_melon_song_candidates
    monkeypatch.setattr(crawler, "fetch_melon_song_candidates",
                        lambda a, t: searches.append((a, t)) or real_search(a, t))
    registry = crawler.Registry()
    assert registry.completed_song_for("가수", "밤편지") == SONG_ID
    assert crawler.process_song("가수", "밤편지", registry) == "skipped"
    assert searches == []
    assert registry.made_requests is False        # 호출부가 대기를 건너뛰는 근거


def test_main_does_not_sleep_for_songs_it_skipped_before_searching(pipeline, monkeypatch) -> None:
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    def no_sleep(seconds):
        raise AssertionError("완료곡에 대기가 발생했다")

    monkeypatch.setattr(crawler.time, "sleep", no_sleep)
    songs = tmp / "songs.csv"
    songs.write_text("artist,title\n가수,밤편지\n", encoding="utf-8")
    assert crawler.main([str(songs)]) == 0


def write_jsonl(path: Path, records: list) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")
    return path


def test_seed_index_needs_a_title_to_index_a_record(tmp_path) -> None:
    """요청 정보도 없고 제목도 없으면 색인에 넣지 않는다. 검색해서 판단해야 한다."""
    jsonl = write_jsonl(tmp_path / "all.jsonl", [
        {"song_id": "1", "crawl_status": {"input_artist": "가수", "input_title": "곡"}},
        {"song_id": "2"},
    ])
    index = crawl_state.seed_index(jsonl, only_ids={"1", "2"})
    assert list(index.values()) == ["1"]
    # 완료되지 않은 곡은 색인에서 빠진다
    assert crawl_state.seed_index(jsonl, only_ids=set()) == {}


def test_seed_index_falls_back_to_the_melon_title_for_legacy_records(tmp_path) -> None:
    """단계별 재시작 도입 전에 모은 곡은 crawl_status가 아예 없다.

    그 레코드까지 색인하지 않으면 이미 수집한 900곡이 실행마다 멜론 검색과 대기 2~5초를
    다시 쓴다(색인이 사실상 빈 채로 돈다). 레코드 자신의 멜론 가수·제목으로 채운다.
    """
    jsonl = write_jsonl(tmp_path / "all.jsonl", [
        {"song_id": "10", "metadata": {"title": "밤편지", "artist": ["아이유"]}},
    ])
    index = crawl_state.seed_index(jsonl, only_ids={"10"})
    assert index[(crawl_state._seed_key("아이유"), crawl_state._seed_key("밤편지"))] == "10"
    # 띄어쓰기·대소문자 차이는 같은 열쇠다
    assert index.get((crawl_state._seed_key("아이유"), crawl_state._seed_key("밤 편지"))) == "10"


def test_request_info_wins_over_the_melon_title(tmp_path) -> None:
    """수집 당시 요청한 시드가 있으면 그 열쇠를 쓴다. 폴백이 덮어쓰지 않는다.

    '밤편지'로 요청해서 10번(멜론 제목은 '밤편지 (Remastered)')을 받았고, 20번은 멜론 제목이
    '밤편지'다. 시드 '밤편지'가 가리키는 곡은 실제로 요청해서 받은 10번이다.
    """
    jsonl = write_jsonl(tmp_path / "all.jsonl", [
        {"song_id": "10", "metadata": {"title": "밤편지 (Remastered)", "artist": ["아이유"]},
         "crawl_status": {"input_artist": "아이유", "input_title": "밤편지"}},
        {"song_id": "20", "metadata": {"title": "밤편지", "artist": ["아이유"]}},
    ])
    index = crawl_state.seed_index(jsonl, only_ids={"10", "20"})
    assert index[(crawl_state._seed_key("아이유"), crawl_state._seed_key("밤편지"))] == "10"
    assert index[(crawl_state._seed_key("아이유"), crawl_state._seed_key("밤편지 (Remastered)"))] == "10"


def test_seed_index_drops_a_key_that_two_different_songs_share(tmp_path) -> None:
    """폴백 열쇠가 곡 둘을 가리키면 버린다. 건너뛰면 수집해야 할 곡을 빠뜨린다."""
    jsonl = write_jsonl(tmp_path / "all.jsonl", [
        {"song_id": "10", "metadata": {"title": "Love", "artist": ["가수"]}},
        {"song_id": "20", "metadata": {"title": "love!", "artist": ["가수"]}},
    ])
    assert crawl_state.seed_index(jsonl, only_ids={"10", "20"}) == {}


def test_seed_index_keeps_non_latin_titles_apart(tmp_path) -> None:
    """한글·영숫자만 남기면 일본어 제목이 통째로 빈 문자열이 되어 서로 같은 열쇠가 됐다."""
    jsonl = write_jsonl(tmp_path / "all.jsonl", [
        {"song_id": "10", "metadata": {"title": "恋", "artist": ["星野源"]}},
        {"song_id": "20", "metadata": {"title": "愛", "artist": ["宇多田ヒカル"]}},
    ])
    index = crawl_state.seed_index(jsonl, only_ids={"10", "20"})
    assert sorted(index.values()) == ["10", "20"]
    assert crawl_state._seed_key("恋") != crawl_state._seed_key("愛")


def test_legacy_record_is_skipped_without_a_melon_search(pipeline, monkeypatch) -> None:
    """crawl_status가 없는 예전 레코드도 검색 전에 건너뛴다."""
    crawler, state, calls, tmp = pipeline
    crawler.JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)
    folder = crawler.DATA_DIR / f"가수_밤편지_{SONG_ID}"
    folder.mkdir(parents=True)
    for name, body in ((crawl_state.META_FILE, None), (crawl_state.COVER_FILE, jpeg_bytes()),
                       (crawl_state.AUDIO_FILE, m4a_bytes())):
        if body is None:
            (folder / name).write_text(json.dumps(
                {"song_id": SONG_ID, "metadata": {"title": "밤편지", "artist": ["가수"]},
                 "lyrics_data": {"full_lyrics": "가사"}}, ensure_ascii=False), encoding="utf-8")
        else:
            (folder / name).write_bytes(body)
    write_jsonl(crawler.JSONL_FILE, [
        {"song_id": SONG_ID, "metadata": {"title": "밤편지", "artist": ["가수"]},
         "lyrics_data": {"full_lyrics": "가사"}},
    ])

    searches = []
    monkeypatch.setattr(crawler, "fetch_melon_song_candidates",
                        lambda a, t: searches.append((a, t)) or [])
    registry = crawler.Registry()
    assert registry.completed_song_for("가수", "밤편지") == SONG_ID
    assert crawler.process_song("가수", "밤편지", registry) == "skipped"
    assert searches == []


def test_duplicate_found_after_search_skips_comments_and_youtube(pipeline, monkeypatch) -> None:
    """검색 1위가 탈락하고 다음 후보가 이미 수집된 곡인 경우.

    예전에는 댓글·유튜브 수집까지 끝낸 뒤에야 중복으로 판정했다.
    """
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    registry = crawler.Registry()
    before = (calls.melon, calls.youtube, calls.refine)
    assert crawler.process_song("가수", "다른 시드", registry) == "skipped"
    # 멜론 상세까지는 갔지만(1 증가) 유튜브·분석은 부르지 않았다
    assert (calls.youtube, calls.refine) == (before[1], before[2])


def test_startup_reads_folders_and_jsonl_once(pipeline, monkeypatch) -> None:
    """동기화와 Registry가 같은 스캔 결과를 나눠 쓴다."""
    crawler, state, calls, tmp = pipeline
    assert crawler.process_song("가수", "밤편지", crawler.Registry()) == "done"

    scans, reads = [], []
    real_scan, real_read = crawl_state.scan_complete, crawl_state.read_jsonl_ids
    monkeypatch.setattr(crawler, "scan_complete", lambda d: scans.append(d) or real_scan(d))
    monkeypatch.setattr(crawler, "read_jsonl_ids", lambda p: reads.append(p) or real_read(p))
    songs = tmp / "songs.csv"
    songs.write_text("artist,title\n", encoding="utf-8")
    assert crawler.main([str(songs)]) == 0
    assert len(scans) == 1, scans
    assert len(reads) == 1, reads
