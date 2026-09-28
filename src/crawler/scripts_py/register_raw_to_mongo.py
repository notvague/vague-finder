from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from src.common.mongodb import get_collection

DEFAULT_RAW_ROOT = Path("data/raw")


def load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("meta.json의 최상위 값은 JSON 객체여야 합니다.")

    return data


def build_update_document(meta: dict[str, Any], song_dir: Path) -> tuple[str, dict[str, Any]]:
    song_id = str(meta.get("song_id") or meta.get("id") or "").strip()
    if not song_id:
        raise ValueError("song_id 또는 id가 없습니다.")

    metadata = meta.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    # meta.json의 현재 필드 전체를 MongoDB에 보존
    # MongoDB의 예약 필드와 별도 관리할 상태 필드만 제외
    document = {
        key: value
        for key, value in meta.items()
        if key not in {"_id", "id", "song_id", "status", "created_at", "updated_at"}
    }

    document.update(
        {
            "song_id": song_id,
            "folder_name": song_dir.name,
            "title": metadata.get("title"),
            "artist": metadata.get("artist"),
            "source": {
                "platform": "melon",
                "platform_id": song_id,
            },
            "paths": {
                "raw_dir": song_dir.as_posix(),
                "audio": (song_dir / "audio.m4a").as_posix(),
                "cover": (song_dir / "cover.jpg").as_posix(),
                "meta": (song_dir / "meta.json").as_posix(),
            },
        }
    )

    return song_id, document


def register_song(song_dir: Path, songs_col: Collection, dry_run: bool = False) -> str:
    meta_path = song_dir / "meta.json"
    if not meta_path.exists():
        return "skipped"

    meta = load_meta(meta_path)
    song_id, document = build_update_document(meta, song_dir)
    now = datetime.now(timezone.utc)

    update = {
        "$set": {
            **document,
            "status.raw_registered": True,
            "status.audio_file_exists": (song_dir / "audio.m4a").exists(),
            "status.cover_file_exists": (song_dir / "cover.jpg").exists(),
            "updated_at": now,
        },
        "$setOnInsert": {
            "created_at": now,
            "status.audio_embedded": False,
        },
    }

    if dry_run:
        title = document.get("title") or "제목 없음"
        print(f"[DRY-RUN] {song_id} | {title}")
        return "dry-run"

    result = songs_col.update_one({"_id": song_id}, update, upsert=True)

    if result.upserted_id is not None:
        return "inserted"
    if result.modified_count > 0:
        return "updated"
    return "unchanged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="data/raw의 meta.json을 MongoDB에 적재")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="raw 폴더 경로 기본값: data/raw",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 지정한 곡 수만 처리",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="MongoDB에 쓰지 않고 JSON 파싱만 검사함",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root

    if not raw_root.exists():
        raise FileNotFoundError(f"raw 폴더를 찾지 못함: {raw_root}")

    song_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if args.limit is not None:
        song_dirs = song_dirs[: max(args.limit, 0)]

    if not song_dirs:
        print(f"처리할 곡 폴더가 없습니다: {raw_root}")
        return

    songs_col = None if args.dry_run else get_collection("songs")
    counts = {
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "failed": 0,
        "dry-run": 0,
    }

    for song_dir in song_dirs:
        try:
            if songs_col is None:
                # dry-run에서는 DB 객체가 필요 없으므로 타입 검사만 우회
                class _DummyCollection:
                    pass

                result = register_song(song_dir, _DummyCollection(), dry_run=True)  # type: ignore[arg-type]
            else:
                result = register_song(song_dir, songs_col)

            counts[result] += 1
            if result not in {"dry-run", "skipped"}:
                print(f"[{result.upper()}] {song_dir.name}")
        except (OSError, ValueError, json.JSONDecodeError, PyMongoError) as error:
            counts["failed"] += 1
            print(f"[FAILED] {song_dir.name}: {error}")

    print("\n완료")
    print(f"대상: {len(song_dirs)}곡")
    print(
        "신규: {inserted}, 갱신: {updated}, 변경 없음: {unchanged}, "
        "건너뜀: {skipped}, 실패: {failed}, 점검만: {dry-run}".format(**counts)
    )


if __name__ == "__main__":
    main()
