"""work_label_check 자체 검사 — 합성 행만 쓴다(실제 수집 자료 아님)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from work_label_check import COLUMNS, check_rows  # noqa: E402


def row(**kw):
    base = {"tmp_id": "x", "writer_id": "w1", "query": "의사들이 밴드 하는 드라마 노래", "track": "diag_work",
            "split": "policy_dev", **{c: "" for c in COLUMNS}}
    base.update(kw)
    return base


def main() -> int:
    cases = [
        ("정상", [row(work_reference="yes", work_name_after_writing="가상병원", work_media_type="drama",
                    work_check_by="w2", work_check_result="same", work_in_song_data="unknown")], {}),
        ("이름 없음", [row(work_reference="yes", work_check_by="w2")], {"work_yes_without_name": 1}),
        ("질의에 이름", [row(work_reference="yes", work_name_after_writing="밴드 하는 드라마", work_check_by="w2")],
         {"work_name_written_in_query": 1}),
        ("확인자=작성자", [row(work_reference="yes", work_name_after_writing="가상병원", work_check_by="w1")],
         {"checker_is_writer": 1}),
        ("미확인", [row(work_reference="yes", work_name_after_writing="가상병원")], {"work_unchecked": 1}),
        ("값 오류", [row(work_reference="maybe")], {"bad_work_reference": 1}),
        ("종류 오류", [row(work_reference="no", work_media_type="tv")], {"bad_media_type": 1}),
        ("no는 이름 불필요", [row(work_reference="no")], {}),
    ]
    passed = 0
    for label, rows, expected in cases:
        got = check_rows(rows)["flags"]
        ok = got == expected
        passed += ok
        print(("ok  " if ok else "FAIL"), label, got)
    try:
        check_rows([{"tmp_id": "x"}])
        print("FAIL 열 없음 거부")
    except SystemExit:
        passed += 1
        print("ok   열 없음 거부")
    total = len(cases) + 1
    print(f"{passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
