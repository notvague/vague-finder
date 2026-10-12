"""저장된 분석(폴백 제외)에 지금 코드의 모달리티 후처리를 다시 적용해, 결과가 그대로인지 본다.

이번 변경은 "끝내 검증을 통과하지 못한 응답"의 처리만 바꾼다. 통과한 응답의 분석은 한 글자도
달라지지 않아야 한다 — 달라지면 기준선(v39·v41)을 다시 재야 한다. Gemini·검색 없음.

  PYTHONPATH=. venv/bin/python <이 파일> <캐시 json> [<캐시 json> ...]

기본(strict)과 내려놓기(degrade=True) 두 길 모두 본다. 새 필드 `degraded_modalities`는 저장된
분석에 없으므로 빈 목록이어야 하고, 나머지 키는 저장된 값과 같아야 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.retrieval.modality_queries import ModalityQueryValidationError, apply_modality_query_safeguards

KEYS = ("image_english_query", "audio_english_query", "modality_weights", "has_visual_clue")


def main() -> None:
    total = same = 0
    for path in map(Path, sys.argv[1:]):
        for qid, entry in json.load(open(path, encoding="utf-8"))["entries"].items():
            if entry.get("fallback"):
                continue
            total += 1
            stored = entry["analysis"]
            try:
                strict = apply_modality_query_safeguards(entry["query"], dict(stored))
            except ModalityQueryValidationError as exc:
                print(f"  {path.name} {qid}: 검증 실패 — {exc}")
                continue
            lenient = apply_modality_query_safeguards(entry["query"], dict(stored), degrade=True)
            diff = [k for k in KEYS if strict.get(k) != stored.get(k)]
            if strict != lenient:
                diff.append("strict≠degrade")
            if strict["degraded_modalities"]:
                diff.append("degraded_modalities")
            if diff:
                print(f"  {path.name} {qid}: 달라짐 {diff}")
            else:
                same += 1
    print(f"폴백 아닌 캐시 항목 {total}건 중 그대로 {same}건")


if __name__ == "__main__":
    main()
