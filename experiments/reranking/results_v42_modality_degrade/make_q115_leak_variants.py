"""가정 실험용 캐시: 소리 묘사가 중심인 q115(v06 dev)의 소리 프롬프트가 누출 검사에 걸렸다고 치고 네 가지로 만든다.

q115는 실제로 이 일이 있었던 질의다(results_v36 캐시 생성 로그: 모델이 "… in the background"를
써서 3회 거부 → 규칙 폴백). 지금 저장된 분석은 통과한 것이라, 그 프롬프트 끝에 "in the background"를
붙여 같은 상황을 만든다. 누출을 규칙 프롬프트로 바꿀지(채택) 끌지(대조)를 고르는 근거다.

  stored    저장된 분석 그대로(통과한 응답)
  fallback  예전 동작 — 분석 전체를 규칙 폴백으로
  degrade   채택 — 소리 프롬프트만 규칙 프롬프트로, 나머지 분석 유지
  audio_off 대조 — 소리 경로만 끔, 나머지 분석 유지

  PYTHONPATH=. venv/bin/python <이 파일>
"""
from __future__ import annotations

import json
from pathlib import Path

from src.retrieval import query_analyzer as qa
from src.retrieval.modality_queries import ModalityQueryValidationError, apply_modality_query_safeguards

HERE = Path(__file__).parent
SOURCE = HERE.parent / "analysis_cache_v06_dev.json"
QID = "q115"


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    entry = base["entries"][QID]
    query, stored = entry["query"], entry["analysis"]

    leaked = dict(stored)
    leaked["audio_english_query"] = stored["audio_english_query"].rstrip(". ") + " in the background."
    try:
        apply_modality_query_safeguards(query, dict(leaked))
        raise SystemExit("누출로 읽히지 않았다 — 가정이 성립하지 않는다")
    except ModalityQueryValidationError as exc:
        print("누출:", exc)

    degraded = apply_modality_query_safeguards(query, dict(leaked), degrade=True)
    degraded["original_query"] = query
    audio_off = dict(degraded)
    audio_off["audio_english_query"] = ""
    audio_off["modality_weights"] = {"text": 1.0, "image": 0.0, "audio": 0.0}

    variants = {
        "stored": stored,
        "fallback": qa.rule_fallback(query).model_dump(mode="json"),
        "degrade": qa.QueryAnalysis(**degraded).model_dump(mode="json"),
        "audio_off": qa.QueryAnalysis(**audio_off).model_dump(mode="json"),
    }
    for tag, analysis in variants.items():
        cache = json.loads(SOURCE.read_text(encoding="utf-8"))
        cache["entries"] = {QID: dict(entry, analysis=analysis, fallback=(tag == "fallback"))}
        cache["meta"].setdefault("notes", []).append(
            f"q115 한 건짜리 가정 실험 캐시 — {tag} (make_q115_leak_variants.py)"
        )
        path = HERE / f"analysis_cache_q115_{tag}.json"
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{tag:9s} audio={analysis['audio_english_query'][:70]!r} weights={analysis['modality_weights']}")


if __name__ == "__main__":
    main()
