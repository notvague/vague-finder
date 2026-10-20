"""v78 런타임 산출물 적재 시간·RSS 증가(따로 잰 값, 5회). 모델·DB 호출 없음.

    <venv>/bin/python measure_load.py
"""
from __future__ import annotations

import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(__import__("os").getpid())], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.retrieval import description_semantic_aux as dsa

    out = []
    for _ in range(5):
        gc.collect()
        before = rss_mb()
        t0 = time.perf_counter()
        store = dsa.DescriptionSemanticStore.load(ROOT / dsa.DEFAULT_RUNTIME_DIR)
        sec = time.perf_counter() - t0
        out.append({"seconds": round(sec, 3), "rss_delta_mb": round(rss_mb() - before, 1)})
        del store
    result = {"runs": out, "first_run_rss_delta_mb": out[0]["rss_delta_mb"],
              "seconds_median": sorted(r["seconds"] for r in out)[2],
              "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)}
    (HERE / "analysis/load_cost.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
