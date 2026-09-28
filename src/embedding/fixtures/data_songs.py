from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List

from src.embedding.fixtures.meta_validation import (
    ValidationIssue,
    validate_meta_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def resolve_raw_dir() -> Path:
    """곡 폴더가 **직접** 들어 있는 폴더.

    VAGUEFINDER_DATA_DIR은 크롤러와 같은 뜻이다 — 곡 폴더를 직접 담은 폴더
    (예: .../vague-finder/raw5). 예전에는 임베딩만 `<그 값>/raw`를 봤다. 팀 드라이브에는
    옛 배치 `raw`(341곡)가 실제로 남아 있어서, 같은 환경변수로 크롤러는 raw5에 쌓고 임베딩은
    raw를 읽는 상태가 만들어졌다. 한쪽이 조용히 엉뚱한 데이터를 쓰는 종류의 어긋남이다.

    적지 않으면 저장소 안 data/raw를 본다(예전 기본값과 같다).
    """
    env = (os.getenv("VAGUEFINDER_DATA_DIR") or "").strip()
    return Path(env) if env else DEFAULT_RAW_DIR


def resolve_failed_raw_dir(raw_dir: Path) -> Path:
    """검증에 실패한 곡 폴더를 옮길 자리.

    크롤러가 격리 폴더를 `<이름>_incomplete`로 짓는 것과 같은 방식으로 `<이름>_failed`를 쓴다.
    수집 폴더 옆에 나란히 생겨서 어느 수집분에서 나온 것인지 헷갈리지 않는다.
    """
    env = (os.getenv("VAGUEFINDER_FAILED_DIR") or "").strip()
    return Path(env) if env else raw_dir.with_name(raw_dir.name + "_failed")


@dataclass
class RejectedSong:
    song_id: str
    source_dir: Path
    destination_dir: Path | None
    issues: list[ValidationIssue]
    move_error: str | None = None

    @property
    def moved(self) -> bool:
        return (
            self.destination_dir is not None
            and self.move_error is None
        )


@dataclass
class SongLoadReport:
    raw_dir: Path
    failed_raw_dir: Path
    scanned_count: int = 0
    accepted_count: int = 0
    rejected: list[RejectedSong] = field(default_factory=list)
    audit_report_path: Path | None = None
    audit_write_errors: list[str] = field(default_factory=list)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def moved_count(self) -> int:
        return sum(
            1
            for item in self.rejected
            if item.moved
        )

    @property
    def move_failed_count(self) -> int:
        return sum(
            1
            for item in self.rejected
            if not item.moved
        )

    @property
    def rejected_ids(self) -> set[str]:
        return {
            item.song_id
            for item in self.rejected
            if item.song_id
        }


@dataclass
class SongLoadResult:
    songs: List[Dict]
    report: SongLoadReport


def _extract_song_id(
    obj: Any,
    song_dir: Path,
) -> str:
    if isinstance(obj, dict):
        value = obj.get("id") or obj.get("song_id")

        if value is not None and str(value).strip():
            return str(value).strip()

    # Artist_Title_12345678 형식이면 마지막 숫자를 사용한다.
    suffix = song_dir.name.rsplit("_", 1)[-1]

    if suffix.isdigit():
        return suffix

    return song_dir.name


def _next_destination(
    failed_raw_dir: Path,
    folder_name: str,
) -> Path:
    """
    failed_raw에 같은 폴더가 존재해도 기존 폴더를 덮어쓰지 않는다.
    """
    destination = failed_raw_dir / folder_name

    if not destination.exists():
        return destination

    index = 2

    while True:
        candidate = (
            failed_raw_dir
            / f"{folder_name}__retry_{index}"
        )

        if not candidate.exists():
            return candidate

        index += 1


def _move_rejected_song(
    song_dir: Path,
    raw_dir: Path,
    failed_raw_dir: Path,
) -> Path:
    """
    검증 실패 곡 폴더 전체를 data/failed_raw로 이동한다.
    """
    if song_dir.resolve() == raw_dir.resolve():
        raise ValueError(
            "data/raw 루트 전체를 failed_raw로 이동할 수 없습니다."
        )

    failed_raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = _next_destination(
        failed_raw_dir,
        song_dir.name,
    )

    shutil.move(
        str(song_dir),
        str(destination),
    )

    return destination


def _write_audit_record(
    report: SongLoadReport,
    rejected: RejectedSong,
) -> None:
    """
    실패한 곡의 모든 사유를 JSONL로 기록한다.
    """
    report.failed_raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        report.failed_raw_dir
        / "validation_report.jsonl"
    )

    record = {
        "rejected_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "song_id": rejected.song_id,
        "source_dir": str(rejected.source_dir),
        "destination_dir": (
            str(rejected.destination_dir)
            if rejected.destination_dir
            else None
        ),
        "moved": rejected.moved,
        "move_error": rejected.move_error,
        "issues": [
            issue.to_dict()
            for issue in rejected.issues
        ],
    }

    try:
        with audit_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        report.audit_report_path = audit_path

    except Exception as exc:
        report.audit_write_errors.append(
            f"{type(exc).__name__}: {exc}"
        )


def _reject_song(
    *,
    report: SongLoadReport,
    song_id: str,
    song_dir: Path,
    issues: list[ValidationIssue],
    move_invalid: bool,
) -> None:
    destination: Path | None = None
    move_error: str | None = None

    if move_invalid:
        try:
            destination = _move_rejected_song(
                song_dir,
                report.raw_dir,
                report.failed_raw_dir,
            )

        except Exception as exc:
            move_error = (
                f"{type(exc).__name__}: {exc}"
            )

    else:
        move_error = "이동 비활성화"

    rejected = RejectedSong(
        song_id=song_id,
        source_dir=song_dir,
        destination_dir=destination,
        issues=issues,
        move_error=move_error,
    )

    report.rejected.append(rejected)

    _write_audit_record(
        report,
        rejected,
    )


def _build_song(
    obj: Dict,
    meta_path: Path,
) -> Dict:
    """
    검증을 통과한 meta.json을 기존 임베딩 song 구조로 변환한다.
    """
    song_id = (
        obj.get("id")
        or obj.get("song_id")
        or meta_path.parent.name
    )

    metadata = dict(
        obj.get("metadata", {})
    )

    links = (
        obj.get("links", {})
        if isinstance(obj.get("links"), dict)
        else {}
    )

    cover_local = (
        meta_path.parent
        / "cover.jpg"
    )

    if cover_local.is_file():
        metadata["album_cover"] = str(
            cover_local
        )

    elif links.get("cover_url"):
        metadata["album_cover"] = links[
            "cover_url"
        ]

    audio_local = (
        meta_path.parent
        / "audio.m4a"
    )

    if audio_local.is_file():
        metadata["audio_path"] = str(
            audio_local
        )

    return {
        "id": str(song_id),
        "metadata": metadata,

        "lyrics_data": (
            obj.get("lyrics_data", {})
            or {}
        ),

        "semantic_analysis": (
            obj.get("semantic_analysis", {})
            or {}
        ),

        "community_feedback": (
            obj.get("community_feedback", {})
            or {}
        ),

        "search_vector_inputs": (
            obj.get("search_vector_inputs", {})
            or {}
        ),

        # Carried separately for a fact-level context index.  The existing
        # song-level dense/BM25 passage builders intentionally ignore it.
        "namuwiki": (
            obj.get("namuwiki", {})
            if isinstance(obj.get("namuwiki"), dict)
            else {}
        ),

        "text_values": None,
        "text_dense_values": None,
        "text_sparse_values": None,
        "image_values": None,
        "audio_values": None,
    }


def load_songs_with_report(
    raw_dir: str | Path | None = None,
    *,
    failed_raw_dir: str | Path | None = None,
    validate_meta: bool = True,
    move_invalid: bool = True,
    require_media_files: bool = True,
) -> SongLoadResult:
    """
    곡 폴더를 순회한다.

    raw_dir는 **곡 폴더를 직접 담은 폴더**다(크롤러의 VAGUEFINDER_DATA_DIR과 같은 뜻).
    생략하면 resolve_raw_dir()이 환경변수로 정한다.

    다음 조건을 모두 충족해야 임베딩 대상에 포함한다.

    1. meta.json 존재
    2. cover.jpg 존재 및 0바이트가 아님
    3. audio.m4a 존재 및 0바이트가 아님
    4. meta.json 파싱 성공
    5. meta.json 모든 핵심 필드가 정상
    6. 동일 song_id 중복 없음
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else resolve_raw_dir()
    failed_raw_dir = (
        Path(failed_raw_dir) if failed_raw_dir is not None else resolve_failed_raw_dir(raw_dir)
    )

    report = SongLoadReport(
        raw_dir=raw_dir,
        failed_raw_dir=failed_raw_dir,
    )

    # meta.json을 검색하는 것이 아니라 곡 폴더를 먼저 찾는다.
    # 그래야 meta.json 자체가 없는 폴더도 검출할 수 있다.
    song_dirs = (
        sorted(
            path
            for path in raw_dir.iterdir()
            if path.is_dir()
        )
        if raw_dir.exists()
        else []
    )

    report.scanned_count = len(song_dirs)

    songs: List[Dict] = []
    accepted_ids: set[str] = set()

    for song_dir in song_dirs:
        meta_path = (
            song_dir
            / "meta.json"
        )

        obj: Any = None

        # meta.json 자체가 없는 경우
        if not meta_path.is_file():
            issues = [
                ValidationIssue(
                    path="<file>.meta.json",
                    reason="필수 meta.json 파일 없음",
                    value_preview=str(meta_path),
                )
            ]

            _reject_song(
                report=report,
                song_id=_extract_song_id(
                    obj,
                    song_dir,
                ),
                song_dir=song_dir,
                issues=issues,
                move_invalid=move_invalid,
            )

            continue

        # meta.json 파싱
        try:
            obj = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:
            issues = [
                ValidationIssue(
                    path="meta.json",
                    reason=(
                        "JSON 읽기/파싱 실패"
                        f"({type(exc).__name__})"
                    ),
                    value_preview=str(exc),
                )
            ]

            _reject_song(
                report=report,
                song_id=_extract_song_id(
                    obj,
                    song_dir,
                ),
                song_dir=song_dir,
                issues=issues,
                move_invalid=move_invalid,
            )

            continue

        song_id = _extract_song_id(
            obj,
            song_dir,
        )

        issues: list[ValidationIssue] = []

        if validate_meta:
            issues.extend(
                validate_meta_document(
                    obj,
                    song_dir=song_dir,
                    require_media_files=(
                        require_media_files
                    ),
                )
            )

        elif (
            not isinstance(obj, dict)
            or not isinstance(
                obj.get("metadata"),
                dict,
            )
        ):
            issues.append(
                ValidationIssue(
                    path="metadata",
                    reason=(
                        "JSON 최상위/metadata "
                        "스키마 이상"
                    ),
                    value_preview=repr(
                        type(obj).__name__
                    ),
                )
            )

        # 동일 song_id 중복 검사
        if (
            song_id
            and song_id in accepted_ids
        ):
            issues.append(
                ValidationIssue(
                    path="song_id",
                    reason=(
                        "data/raw 안에 같은 "
                        "song_id가 중복됨"
                    ),
                    value_preview=repr(
                        song_id
                    ),
                )
            )

        if issues:
            _reject_song(
                report=report,
                song_id=song_id,
                song_dir=song_dir,
                issues=issues,
                move_invalid=move_invalid,
            )

            continue

        songs.append(
            _build_song(
                obj,
                meta_path,
            )
        )

        accepted_ids.add(song_id)

    report.accepted_count = len(songs)

    return SongLoadResult(
        songs=songs,
        report=report,
    )


def load_songs_from_data_dir(
    raw_dir: str | Path | None = None,
) -> List[Dict]:
    """곡 목록만 필요할 때 쓰는 얇은 래퍼. raw_dir 의미는 load_songs_with_report와 같다."""
    return load_songs_with_report(raw_dir).songs
