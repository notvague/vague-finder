"""
[Audit] 수집 결과 점검

설명: all_songs.jsonl을 훑어 "원곡이 아닐 수 있는" 레코드를 뽑는다. 재크롤링 없이
      돌릴 수 있어서, 이미 쌓인 데이터를 감시하거나 재수집 뒤 결과를 확인할 때 쓴다.

      판정 기준은 크롤러가 곡을 고를 때 쓰는 것과 같다(melon_match). 기준이 하나라
      "수집할 때는 통과했는데 점검에서는 걸린다" 같은 어긋남이 생기지 않는다.

사용:
    venv/bin/python -m src.crawler.scripts_py.audit_collection
    venv/bin/python -m src.crawler.scripts_py.audit_collection --csv data/audit.csv

작성자: 황찬혁 (Full)
생성일: 2026-09-17
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .melon_match import (
    MatchScore,
    check_details,
    normalize,
    score_candidate,
    source_titles_agree,
)

DEFAULT_JSONL = Path("data/all_songs.jsonl")
DEFAULT_SEED_GLOB = "data/crawl_batches/songs_*.csv"


def load_seeds(pattern: str) -> List[Tuple[str, str]]:
    seeds: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("artist") and row.get("title"):
                    seeds.append((row["artist"], row["title"]))
    return seeds


def index_by_artist(records: List[Dict]) -> Dict[str, List[Dict]]:
    index = collections.defaultdict(list)
    for record in records:
        for name in (record.get("metadata", {}).get("artist") or []):
            index[normalize(name)].append(record)
    return index


def _seed_key(artist: str, title: str) -> Tuple[str, str]:
    return normalize(artist), normalize(title)


def index_by_seed(records: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """crawl_status에 남긴 요청 가수·제목으로 색인한다. 2026-09-18 이후 수집분에만 있다."""
    index = collections.defaultdict(list)
    for record in records:
        status = record.get("crawl_status") or {}
        artist, title = status.get("input_artist"), status.get("input_title")
        if artist and title:
            index[_seed_key(artist, title)].append(record)
    return index


def find_records(
    seed_artist: str,
    seed_title: str,
    index: Dict[str, List[Dict]],
    seed_index: Optional[Dict[Tuple[str, str], List[Dict]]] = None,
) -> List[Dict]:
    """시드가 가리키는 레코드를 **전부** 찾는다.

    첫 레코드만 돌려주면 정상 'My Love' 뒤에 잘못 수집된 'My Love (Duet Ver.)'가 있어도
    정상 레코드만 검사하고 끝난다. 같은 시드로 여러 번 수집된 레코드가 모두 점검 대상이다.

    수집 당시 요청 정보(crawl_status.input_artist/input_title)가 있는 레코드는 그것으로
    연결하고, 없는 예전 레코드는 제목 부분일치로 찾는다. 둘을 합친다.
    """
    found: List[Dict] = []
    seen_ids = set()

    def _add(record: Dict) -> None:
        key = str(record.get("song_id") or id(record))
        if key not in seen_ids:
            seen_ids.add(key)
            found.append(record)

    for record in (seed_index or {}).get(_seed_key(seed_artist, seed_title), []):
        _add(record)

    target = normalize(seed_title)
    for record in index.get(normalize(seed_artist), []):
        collected = normalize(record.get("metadata", {}).get("title"))
        if target and (target in collected or collected in target):
            _add(record)
    return found


def find_record(seed_artist: str, seed_title: str, index: Dict[str, List[Dict]]) -> Optional[Dict]:
    """첫 레코드만 필요할 때의 래퍼. 점검에는 find_records를 쓴다."""
    records = find_records(seed_artist, seed_title, index)
    return records[0] if records else None


def inspect(seed_title: str, record: Dict) -> Tuple[MatchScore, List[str]]:
    """레코드 하나를 채점하고 추가 경고를 모은다."""
    meta = record.get("metadata", {})
    result = score_candidate(seed_title, meta.get("title", ""), meta.get("album", ""))

    warnings: List[str] = []
    ok, note = check_details(meta.get("genre") or [])
    if not ok:
        warnings.append(note)
    elif note:
        warnings.append(note)

    # 가사가 없는 레코드. 임베딩 전 검증이 거부하므로 색인에 들어가지 않는다.
    # 대부분 19금 곡이다 — 성인 인증이 필요해 가사 영역이 아예 오지 않고, 멜론이 곡명 앞에 붙이는
    # 배지 글자가 제목에도 섞여 '19금 BAND'처럼 저장됐다. 2026-09-18부터 후보 단계에서 거른다.
    if not (record.get("lyrics_data", {}).get("full_lyrics") or "").strip():
        collected_title = meta.get("title", "")
        if collected_title.startswith("19금"):
            warnings.append("19금 곡(가사 수집 불가) — 재수집 대상 아님, 삭제 권장")
        else:
            warnings.append("가사 없음 — 색인에서 제외됨")

    # 멜론(가사·메타)과 유튜브(오디오)가 같은 녹음인지. youtube_title이 없는
    # 예전 레코드는 판단할 근거가 없으므로 건너뛴다.
    video_title = (record.get("links") or {}).get("youtube_title", "")
    if video_title and not source_titles_agree(meta.get("title", ""), video_title,
                                               list(meta.get("artist") or [])):
        warnings.append(f"멜론/유튜브 불일치 (영상: {video_title})")

    return result, warnings


def load_records(jsonl_path: Path) -> List[Dict]:
    """JSONL을 읽는다. 같은 song_id가 여러 줄이면 마지막 줄을 쓴다."""
    by_id: Dict[str, Dict] = {}
    extra: List[Dict] = []
    with jsonl_path.open(encoding="utf-8", newline="") as f:
        for line in f.read().split("\n"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            song_id = str(record.get("song_id") or "")
            if song_id:
                by_id[song_id] = record
            else:
                extra.append(record)
    return list(by_id.values()) + extra


def audit(
    seeds: List[Tuple[str, str]],
    index: Dict[str, List[Dict]],
    seed_index: Optional[Dict[Tuple[str, str], List[Dict]]] = None,
) -> Tuple[List[Tuple[str, str, Dict, MatchScore, List[str]]], int, int, int]:
    """시드마다 연결된 레코드를 전부 채점한다.

    Returns:
        (점검 대상 목록, 통과 수, 미수집 시드 수, 레코드가 둘 이상인 시드 수)
    """
    flagged: List[Tuple[str, str, Dict, MatchScore, List[str]]] = []
    clean = unmatched = duplicates = 0

    for artist, title in seeds:
        matched = find_records(artist, title, index, seed_index)
        if not matched:
            unmatched += 1
            continue
        if len(matched) > 1:
            duplicates += 1
        for record in matched:
            result, warnings = inspect(title, record)
            if len(matched) > 1:
                warnings = warnings + [f"같은 시드의 레코드 {len(matched)}건 중 하나"]
            if result.hard_reject or result.needs_review or warnings:
                flagged.append((artist, title, record, result, warnings))
            else:
                clean += 1
    return flagged, clean, unmatched, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description="수집 결과에서 원곡이 아닐 수 있는 레코드를 찾는다")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--seeds", default=DEFAULT_SEED_GLOB)
    parser.add_argument("--csv", type=Path, default=None, help="결과를 CSV로도 저장")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"[에러] 수집 파일이 없습니다: {args.jsonl}")
        return 1

    records = load_records(args.jsonl)
    index = index_by_artist(records)
    seed_index = index_by_seed(records)
    seeds = load_seeds(args.seeds)
    if not seeds:
        print(f"[에러] 시드 CSV를 찾지 못했습니다: {args.seeds}")
        return 1

    flagged, clean, unmatched, duplicates = audit(seeds, index, seed_index)

    total = clean + len(flagged)
    print(f"수집 {len(records)}곡 / 시드 {len(seeds)}건 / 대조 {total}건 (미수집 {unmatched}건)")
    if duplicates:
        print(f"  같은 시드로 수집된 레코드가 둘 이상인 시드: {duplicates}건 (모두 점검함)")
    print(f"  통과      : {clean}건")
    print(f"  점검 대상 : {len(flagged)}건")

    rejected = [f for f in flagged if f[3].hard_reject]
    if rejected:
        print(f"\n[재수집 권장] {len(rejected)}건 — 지금 기준이면 채택하지 않았을 레코드")
        for artist, title, record, result, warnings in rejected:
            meta = record["metadata"]
            print(f"  {record['song_id']}  {artist} - {title}")
            print(f"      수집됨: {meta.get('title','')[:60]}")
            print(f"      사유  : {result.describe()}" + (f" | {'; '.join(warnings)}" if warnings else ""))

    others = [f for f in flagged if not f[3].hard_reject]
    if others:
        print(f"\n[확인 필요] {len(others)}건")
        for artist, title, record, result, warnings in sorted(others, key=lambda x: x[3].score):
            meta = record["metadata"]
            note = f" | {'; '.join(warnings)}" if warnings else ""
            print(f"  {result.score:+5d}  {record['song_id']}  {artist} - {title} -> {meta.get('title','')[:44]}{note}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["song_id", "artist", "seed_title", "collected_title",
                             "album", "genre", "score", "verdict", "reasons"])
            for artist, title, record, result, warnings in flagged:
                meta = record["metadata"]
                writer.writerow([
                    record["song_id"], artist, title, meta.get("title", ""),
                    meta.get("album", ""), ", ".join(meta.get("genre") or []),
                    result.score,
                    "재수집 권장" if result.hard_reject else "확인 필요",
                    "; ".join(result.reasons + warnings),
                ])
        print(f"\nCSV 저장: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
