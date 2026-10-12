"""Show the five previously disputed searches from a finished private v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHECKS = {"nw001": "dev", "q203": "dev", "nw005": "dev",
          "nw021": "test", "nw024": "test"}


def inspect(output: Path, context_source: Path) -> None:
    locked = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
    key = locked["locked_setting"]
    cases = {case["query_id"]: case for case in json.loads(
        context_source.read_text(encoding="utf-8-sig"))["queries"]}
    checkpoints = {split: json.loads((output / f"{split}_checkpoint.json").read_text(encoding="utf-8"))
                   for split in {"dev", "test"}}
    if locked["verdict"]["status"] != "provisional":
        raise ValueError("The reused test gate did not pass; inspect the private report")
    print("Setting:", key, "(provisional; blind validation still required)")
    for query_id, split in CHECKS.items():
        row = checkpoints[split]["runs"][key][query_id]
        fact = row["target_evidence_record_id"]
        allowed = ({cases[query_id].get("evidence_record_id"),
                    *cases[query_id].get("additional_evidence_record_ids", [])}
                   if query_id in cases else set())
        if query_id in {"nw001", "q203"} and not (1 <= (row["on_candidate_rank"] or 0) <= 30):
            raise AssertionError(f"{query_id} is still missing at Candidate@30")
        if query_id in {"nw005", "nw021", "nw024"} and fact and fact not in allowed:
            raise AssertionError(f"{query_id} still displays an unrelated fact: {fact}")
        support = "supported" if fact and fact in allowed else "abstained"
        print(f"{query_id} ({split}): Candidate@30 {row['ref_candidate_rank']}→"
              f"{row['on_candidate_rank']}, Top10 {row['ref_top10_rank']}→"
              f"{row['on_top10_rank']}; evidence={support} {fact or '-'}")
        if fact:
            item = next((item for item in row["displayed_evidence"]
                         if item["song_id"] in row["eligible_ids"].split("|")
                         and item["record_id"] == fact), None)
            if item:
                print("  Fact:", item["fact_text"][:180])
                print("  Source:", item["source_url"])
    print("[PASS] Both recall targets at Candidate@30; three disputed facts supported or hidden")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/context_eval_step10_v2"))
    parser.add_argument("--context-source", type=Path,
                        default=Path("experiments/namuwiki/queries_context_v01.json"))
    args = parser.parse_args()
    inspect(args.output, args.context_source)


if __name__ == "__main__":
    main()
