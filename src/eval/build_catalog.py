"""
Song catalog 생성 스크립트.

사용:
    python -m src.eval.build_catalog
    python -m src.eval.build_catalog --source data/all_songs.jsonl --output docs/eval/song_catalog.json

데이터 jsonl을 읽어 eval set 작성에 필요한 최소 필드만 추출한 catalog를 만든다.
팀원이 docs/eval/queries.json을 작성할 때 이 catalog를 참고해 정답/오답 song_id를 고른다.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import List

DEFAULT_SOURCE = Path("data/all_songs(test).jsonl")
DEFAULT_OUTPUT = Path("docs/eval/song_catalog.json")


def _extract_entry(song: dict) -> dict:
    md = song.get("metadata", {}) or {}
    sa = song.get("semantic_analysis", {}) or {}
    artist = md.get("artist", [])
    if isinstance(artist, str):
        artist = [artist]
    return {
        "song_id": str(song.get("song_id") or song.get("id")),
        "title": md.get("title", ""),
        "artist": artist,
        "genre": md.get("genre", []) or [],
        "vocal_gender": md.get("vocal_gender", ""),
        "mood_tags": (sa.get("mood_tags") or [])[:8],
        "time_weather_tags": (sa.get("time_weather_tags") or [])[:6],
        "place_activity_tags": (sa.get("place_activity_tags") or [])[:6],
    }


def build(source: Path, output: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"입력 데이터 없음: {source}")

    songs: List[dict] = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            songs.append(_extract_entry(json.loads(line)))

    catalog = {
        "source": str(source),
        "generated_at": date.today().isoformat(),
        "total": len(songs),
        "songs": songs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    return catalog


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    catalog = build(args.source, args.output)
    print(f"[OK] {catalog['total']}곡 → {args.output}")


if __name__ == "__main__":
    main()
