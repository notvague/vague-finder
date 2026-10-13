"""#12 전의 분석 캐시와 지금 분석기로 새로 만든 캐시를 질의별로 견준다.

  PYTHONPATH=. venv/bin/python <이 파일> [--out cache_diff.csv]

보는 것: 폴백 · 내려놓은 모달리티 · context_clues가 붙은 질의 · 모달리티 가중과 경로 사용 여부가 바뀐 질의.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
OLD = {
    "v06": ROOT / "analysis_cache_v06_dev.json",
    "v09": ROOT / "results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json",
    "v10": ROOT / "results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json",
}


def weights(analysis: dict) -> tuple[float, float, float]:
    w = analysis.get("modality_weights") or {}
    return tuple(round(float(w.get(k, 0.0)), 2) for k in ("text", "image", "audio"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = []
    for name, old_path in OLD.items():
        old = json.loads(old_path.read_text(encoding="utf-8"))["entries"]
        new = json.loads((HERE / f"analysis_cache_{name}_dev_ctx.json").read_text(encoding="utf-8"))["entries"]
        for qid in sorted(new):
            o, n = old[qid]["analysis"], new[qid]["analysis"]
            ow, nw = weights(o), weights(n)
            rows.append({
                "set": name,
                "query_id": qid,
                "old_weights": "/".join(map(str, ow)),
                "new_weights": "/".join(map(str, nw)),
                "weight_shift": round(max(abs(a - b) for a, b in zip(ow, nw)), 2),
                "paths_changed": int([w > 0 for w in ow] != [w > 0 for w in nw]),
                "intent_changed": int(o.get("intent_type") != n.get("intent_type")),
                "old_fallback": int(bool(old[qid].get("fallback"))),
                "new_fallback": int(bool(new[qid].get("fallback"))),
                "new_degraded": "|".join(n.get("degraded_modalities") or []),
                "context_clues": len(n.get("context_clues") or []),
                "context_targets": " / ".join(c.get("target", "") or "(없음)" for c in n.get("context_clues") or []),
            })
    for name in (*OLD, "합계"):
        part = [r for r in rows if name == "합계" or r["set"] == name]
        print(f"{name}: {len(part)}건 · 폴백 {sum(r['old_fallback'] for r in part)} → {sum(r['new_fallback'] for r in part)}"
              f" · 내려놓음 {sum(1 for r in part if r['new_degraded'])}"
              f" · context_clues 붙음 {sum(1 for r in part if r['context_clues'])}"
              f" · 경로 구성 바뀜 {sum(r['paths_changed'] for r in part)}"
              f" · 가중 0.2 이상 이동 {sum(1 for r in part if r['weight_shift'] >= 0.2)}"
              f" · intent 바뀜 {sum(r['intent_changed'] for r in part)}")
    for r in rows:
        if r["paths_changed"] or r["new_fallback"] or r["weight_shift"] >= 0.2:
            print(f"  {r['set']} {r['query_id']}: {r['old_weights']} → {r['new_weights']}"
                  f"{' · 폴백' if r['new_fallback'] else ''}{' · 내려놓음 ' + r['new_degraded'] if r['new_degraded'] else ''}")
    if args.out:
        with open(HERE / args.out, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
