"""v70 작품 라벨 점검 — v68 `new_data_check.py`(형식·누출·중복·분할)를 **먼저** 돌린 `checked.csv`에 작품명 해석 평가용 열을 확인한다.

모델·검색 호출 없음. 질의 원문을 화면에 찍지 않는다(건수만). v68 도구는 고치지 않는다 — 이 파일은 추가 열만 본다.

추가 열(`collection_template_v70.csv`):
- `work_reference` ∈ yes|no|unsure — 작성자가 질의를 다 쓴 **뒤** "작품(드라마·예능·경연·영화 등)을 떠올리며 썼는가"를 적는다.
- `work_name_after_writing` — yes면 떠올린 작품명. **질의를 쓴 뒤에** 적는다(질의에는 쓰지 않는다 — `diag_work` 규칙).
- `work_media_type` ∈ drama|film|variety|audition|animation|webtoon|musical|documentary|other|''.
- `work_check_by` · `work_check_result` ∈ same|different|unsure|'' — 두 번째 사람이 작품명이 실존하고 질의 서술과 맞는지 확인.
- `work_in_song_data` ∈ yes|no|unknown|'' — 두 번째 사람이 코퍼스 메타(앨범명·발매 정보)로 곡–작품 연결을 확인했는가. 파이프라인 산출물
  (태그·요약·Context 사실)은 보지 않는다. 확인 못 하면 unknown(확정 연결로 쓰지 않는다).

검사: 값 범위 · yes인데 작품명 없음 · 질의 원문에 작품명이 그대로 있음(`diag_work` 위반 표시) · 작성자=확인자 · 분할별 집계.
`general` 갈래의 `work_reference` 비율은 **분할 뒤·결과 전** 두 번째 사람이 붙인 값으로 센다(전체 분포에서 작품명 질의의 몫).

    venv/bin/python work_label_check.py --checked <보관자 폴더>/checked.csv --out <보관자 폴더>/work_label_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

WORK_REFERENCE = {"yes", "no", "unsure"}
MEDIA = {"drama", "film", "variety", "audition", "animation", "webtoon", "musical", "documentary", "other", ""}
CHECK = {"same", "different", "unsure", ""}
IN_DATA = {"yes", "no", "unknown", ""}
COLUMNS = ("work_reference", "work_name_after_writing", "work_media_type", "work_check_by", "work_check_result",
           "work_in_song_data")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def check_rows(rows: List[Dict[str, str]]) -> Dict[str, object]:
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"열 없음: {missing}")
    flags: Counter = Counter()
    by_split: Dict[str, Counter] = {}
    for row in rows:
        ref = row["work_reference"].strip()
        name = row["work_name_after_writing"].strip()
        split = row.get("split", "") or "unsplit"
        counter = by_split.setdefault(split, Counter())
        if ref not in WORK_REFERENCE:
            flags["bad_work_reference"] += 1
            continue
        if row["work_media_type"].strip() not in MEDIA:
            flags["bad_media_type"] += 1
        if row["work_check_result"].strip() not in CHECK:
            flags["bad_check_result"] += 1
        if row["work_in_song_data"].strip() not in IN_DATA:
            flags["bad_in_song_data"] += 1
        counter[f"{row.get('track', '')}:work_reference={ref}"] += 1
        if ref == "yes":
            if not name:
                flags["work_yes_without_name"] += 1
            elif _compact(name) in _compact(row.get("query", "")):
                # 작품명을 질의에 썼다 — 간접 지칭 평가 대상이 아니다(명시 이름은 기존 경로)
                flags["work_name_written_in_query"] += 1
                counter["explicit_name_in_query"] += 1
            if not row["work_check_by"].strip():
                flags["work_unchecked"] += 1
            elif row["work_check_by"].strip() == row.get("writer_id", "").strip():
                flags["checker_is_writer"] += 1
            counter[f"work_check={row['work_check_result'].strip() or 'none'}"] += 1
            counter[f"work_in_song_data={row['work_in_song_data'].strip() or 'none'}"] += 1
    return {"rows": len(rows), "flags": dict(flags), "by_split": {k: dict(v) for k, v in by_split.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checked", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with open(args.checked, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    report = check_rows(rows)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "flags": report["flags"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
