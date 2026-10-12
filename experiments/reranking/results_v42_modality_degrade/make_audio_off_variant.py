"""대조용 캐시: n025의 누출된 소리 프롬프트를 규칙 프롬프트로 바꾸지 않고 **끈** 경우.

채택한 동작(누출 → 규칙 프롬프트)과 나란히 재기 위한 것이다. 분석의 나머지는 같다.
  PYTHONPATH=. venv/bin/python <이 파일>
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "analysis_cache_v09_dev_degrade.json"
TARGET = HERE / "analysis_cache_v09_dev_audio_off.json"


def main() -> None:
    cache = json.loads(SOURCE.read_text(encoding="utf-8"))
    analysis = cache["entries"]["n025"]["analysis"]
    assert analysis["degraded_modalities"] == ["audio"], analysis["degraded_modalities"]
    print("채택:", analysis["audio_english_query"], analysis["modality_weights"])
    analysis["audio_english_query"] = ""
    analysis["modality_weights"] = {"text": 1.0, "image": 0.0, "audio": 0.0}
    cache["meta"].setdefault("notes", []).append(
        "n025만 손으로 바꾼 대조용 캐시 — 소리 프롬프트 없음, 가중 text 1.0 (make_audio_off_variant.py)"
    )
    TARGET.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("대조:", TARGET.name)


if __name__ == "__main__":
    main()
