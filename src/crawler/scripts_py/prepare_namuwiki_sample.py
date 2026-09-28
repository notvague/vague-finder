from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("experiments/namuwiki/sample_30.csv")

TARGET_GROUPS = {
    "Mega Hit": 10,
    "Moderate": 10,
    "Hidden": 10,
}

SEED = 42


def load_records() -> list[dict]:
    records = []

    for meta_path in sorted(RAW_DIR.glob("*/meta.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as exc:
            print(f"[WARN] meta.json 읽기 실패: {meta_path} ({exc})")
            continue

        metadata = obj.get("metadata", {}) or {}
        community = obj.get("community_feedback", {}) or {}
        popularity = community.get("popularity", {}) or {}

        fame = str(popularity.get("fame", "")).strip()
        song_id = str(
            obj.get("song_id")
            or obj.get("id")
            or meta_path.parent.name.rsplit("_", 1)[-1]
        )

        artist = metadata.get("artist", "")
        if isinstance(artist, list):
            artist = ", ".join(str(x) for x in artist)

        records.append(
            {
                "song_id": song_id,
                "artist": artist,
                "title": metadata.get("title", ""),
                "album": metadata.get("album", ""),
                "release_date": metadata.get("release_date", ""),
                "fame": fame,
                "meta_path": str(meta_path),

                # 이후 나무위키 검증 결과를 기록할 필드
                "match_status": "",
                "match_type": "",
                "page_title": "",
                "page_url": "",
                "confidence": "",
                "reason": "",
                "context_tags": "",
                "fact_summary": "",
            }
        )

    return records


def sample_records(records: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    selected = []

    for fame, count in TARGET_GROUPS.items():
        candidates = [
            record
            for record in records
            if record["fame"] == fame
        ]

        print(f"[POOL] {fame}: {len(candidates)}곡")

        if len(candidates) < count:
            raise RuntimeError(
                f"{fame} 표본 부족: "
                f"필요 {count}곡 / 현재 {len(candidates)}곡"
            )

        picked = rng.sample(candidates, count)

        # CSV를 읽기 편하게 그룹별 제목 정렬
        picked.sort(
            key=lambda x: (
                x["artist"],
                x["title"],
                x["song_id"],
            )
        )

        selected.extend(picked)

    return selected


def save_csv(records: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "song_id",
        "artist",
        "title",
        "album",
        "release_date",
        "fame",
        "meta_path",
        "match_status",
        "match_type",
        "page_title",
        "page_url",
        "confidence",
        "reason",
        "context_tags",
        "fact_summary",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] 표본 CSV 생성: {OUTPUT_PATH}")
    print(f"[OK] 총 {len(records)}곡")


def main():
    records = load_records()

    counts = Counter(
        record["fame"]
        for record in records
    )

    print(f"[INFO] 전체 meta.json: {len(records)}곡")
    print("[INFO] fame 분포:")

    for fame, count in sorted(counts.items()):
        print(f"  - {fame or '(empty)'}: {count}")

    selected = sample_records(records)
    save_csv(selected)


if __name__ == "__main__":
    main()