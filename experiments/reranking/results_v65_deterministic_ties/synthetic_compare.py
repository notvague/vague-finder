"""v65 §4-② 합성 백엔드 대조 — synthetic/<code>_<tag>.json의 모든 실행을 독립 기준(after_load.json의 reference)과 서로 비교한다.

    venv/bin/python synthetic_compare.py → synthetic_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ref = json.loads((HERE / "synthetic/after_load.json").read_text(encoding="utf-8"))["reference"]
    out = {"keys": len(ref), "codes": {}}
    for code in ("before", "after"):
        runs = {}
        for tag in ("load", "restart", "reload"):
            data = json.loads((HERE / f"synthetic/{code}_{tag}.json").read_text(encoding="utf-8"))
            for name, res in data["runs"].items():
                runs[f"{tag}:{name}"] = res
        names = sorted(runs)
        eq_ref = {n: sum(1 for k in ref if runs[n][k] == ref[k]) for n in names}
        set_ref = {n: sum(1 for k in ref if set(runs[n][k]) == set(ref[k])) for n in names}
        all_same = sum(1 for k in ref if len({tuple(runs[n][k]) for n in names}) == 1)
        out["codes"][code] = {"runs": names, "identical_to_reference": eq_ref, "same_set_as_reference": set_ref,
                              "keys_identical_across_all_runs": all_same}
        print(code, "all-runs-identical", all_same, "/", len(ref), "| eq_ref", eq_ref)
    (HERE / "synthetic_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
