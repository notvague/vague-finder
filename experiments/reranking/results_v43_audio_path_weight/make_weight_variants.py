"""후보 추적에 쓰는 분석 캐시를 만든다(Gemini 없음). 캐시는 커밋하지 않는다 — 이 파일로 다시 만든다.

  ctrl  측정에 쓰는 세 캐시의 사본(그대로)
  off   소리 프롬프트 없음(경로 끔), 가중은 text·image 비율을 지킨 채 재정규화

`off`는 라우터 스위치 `AUDIO_FUSION_SCALE=0`과 후보 순위가 176건 모두 같다(RUN_INFO §3). 스위치를
만들기 전의 첫 확인이었고, 스위치가 분석을 건드리지 않는다는 대조로 남긴다.

  PYTHONPATH=. venv/bin/python <이 파일>
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SOURCES = {
    "v06": ROOT / "analysis_cache_v06_dev.json",
    "v09": ROOT / "results_v42_modality_degrade" / "analysis_cache_v09_dev_degrade.json",
    "v10": ROOT / "results_v41_cover_description_image_noun" / "analysis_cache_v10_dev_imagenoun.json",
}


def main() -> None:
    for name, source in SOURCES.items():
        ctrl = json.loads(source.read_text(encoding="utf-8"))
        off = json.loads(source.read_text(encoding="utf-8"))
        changed = 0
        for entry in off["entries"].values():
            analysis = entry["analysis"]
            w = analysis["modality_weights"]
            if not analysis["audio_english_query"] or w["audio"] <= 0:
                continue
            total = (w["text"] + w["image"]) or 1.0
            analysis["audio_english_query"] = ""
            analysis["modality_weights"] = {
                "text": round(w["text"] / total, 6), "image": round(w["image"] / total, 6), "audio": 0.0,
            }
            changed += 1
        for tag, cache in (("ctrl", ctrl), ("off", off)):
            cache["meta"].setdefault("notes", []).append(
                f"후보 추적용 캐시 — {tag} (results_v43_audio_path_weight/make_weight_variants.py)"
            )
            (HERE / f"analysis_cache_{name}_dev_{tag}.json").write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print(name, "질의", len(ctrl["entries"]), "| 소리 경로를 끈 항목:", changed)


if __name__ == "__main__":
    main()
