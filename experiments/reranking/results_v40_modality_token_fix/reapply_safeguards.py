"""캐시에 저장된 분석(폴백 제외)에 지금 코드의 모달리티 규칙을 다시 적용해, 프롬프트·가중이 그대로인지 본다.

규칙을 고친 쪽과 고치기 전 쪽에서 각각 돌려 비교한다(Gemini·검색 없음).
  PYTHONPATH=. venv/bin/python <이 파일> <캐시 폴더>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.retrieval.modality_queries import ModalityQueryValidationError, apply_modality_query_safeguards

KEYS = ("image_english_query", "audio_english_query", "modality_weights", "has_visual_clue")


def main() -> None:
    folder = Path(sys.argv[1])
    total = same = 0
    for path in sorted(folder.glob("analysis_cache_v*_*.json")):
        if "_tokens" in path.name or "_lenient" in path.name:
            continue
        for qid, e in json.load(open(path, encoding="utf-8"))["entries"].items():
            if e.get("fallback"):
                continue
            total += 1
            a = e["analysis"]
            try:
                out = apply_modality_query_safeguards(e["query"], dict(a))
            except ModalityQueryValidationError as exc:
                print(f"  {path.name} {qid}: 검증 실패 — {exc}")
                continue
            diff = [k for k in KEYS if k in a and out.get(k) != a.get(k)]
            if diff:
                print(f"  {path.name} {qid}: 달라짐 {diff}")
            else:
                same += 1
    print(f"폴백 아닌 캐시 항목 {total}건 중 그대로 {same}건")


if __name__ == "__main__":
    main()
