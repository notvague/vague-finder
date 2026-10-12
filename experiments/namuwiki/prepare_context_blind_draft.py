"""Prepare a *private, unreviewed* 30-song holdout template from Context artifacts.

The sampler never reads retrieval results or prior evaluation checkpoints. It
excludes all old answer songs and deterministically balances six fact types.
Empty query/review fields intentionally make the draft fail blind registration
until a person writes and checks each new question independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


FIELDS = ("query_id", "query", "relevant_ids", "group", "clue_category",
          "evidence_record_ids", "review_note", "reference_fact_for_author")
FACT_GROUPS = ("context_context", "context_context", "context_mixed")
CONTROL_GROUPS = ("context_control", "regression_lyrics",
                  "regression_mood_sound", "regression_image")


def previous_song_ids(context_source: Path, regression_csv: Path) -> set[str]:
    data = json.loads(context_source.read_text(encoding="utf-8-sig"))
    old = {str(case["song_id"]) for case in data["queries"]}
    with regression_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            old.update(part.strip() for part in row["relevant_ids"].split("|") if part.strip())
    return old


def _order(seed: int, song_id: str, record_id: str = "") -> str:
    return hashlib.sha256(f"{seed}:{song_id}:{record_id}".encode()).hexdigest()


def draft_rows(artifacts: Path, old_ids: set[str], *, seed: int = 20261002) -> list[dict]:
    facts: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    controls: list[str] = []
    for path in sorted(artifacts.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        song_id = str(artifact["song"]["song_id"])
        if song_id in old_ids or path.stem != song_id:
            continue
        records = artifact["retrieval"]["records"]
        if artifact["status"] == "ok" and artifact["source"].get("url"):
            for record in records:
                if record.get("record_id") and record.get("evidence_text", "").strip():
                    facts[record["category"]].append(
                        (song_id, record["record_id"], record["evidence_text"])
                    )
        elif artifact["status"] in {"not_found", "no_trivia"} and not records:
            controls.append(song_id)

    counts = {category: len({item[0] for item in records})
              for category, records in facts.items()}
    categories = sorted(
        (category for category, count in counts.items() if count >= 3),
        key=lambda category: (-counts[category], category),
    )[:6]
    if len(categories) != 6:
        raise ValueError("Need at least six independently populated fact categories")
    chosen: set[str] = set()
    rows: list[dict] = []
    # Pick scarce categories first to preserve distinct songs across categories.
    for category in sorted(categories, key=lambda key: (counts[key], key)):
        records = sorted(facts[category], key=lambda row: _order(seed, row[0], row[1]))
        selected = []
        for song_id, record_id, text in records:
            if song_id not in chosen:
                chosen.add(song_id)
                selected.append((song_id, record_id, text))
            if len(selected) == 3:
                break
        if len(selected) < 3:
            raise ValueError(f"Not enough distinct songs for {category}; revise the predeclared sampling rule")
        for index, (song_id, record_id, text) in enumerate(selected):
            rows.append(dict(query_id=f"blind{len(rows)+1:03d}", query="",
                             relevant_ids=song_id, group=FACT_GROUPS[index],
                             clue_category=category, evidence_record_ids=record_id,
                             review_note="", reference_fact_for_author=text))

    remaining = sorted(set(controls) - chosen, key=lambda sid: _order(seed, sid))
    if len(remaining) < 12:
        raise ValueError("Need twelve distinct no-fact songs for control and regression prompts")
    for group in CONTROL_GROUPS:
        for _ in range(3):
            song_id = remaining.pop(0)
            rows.append(dict(query_id=f"blind{len(rows)+1:03d}", query="",
                             relevant_ids=song_id, group=group,
                             clue_category="no_context", evidence_record_ids="",
                             review_note="", reference_fact_for_author=""))
    assert len(rows) == 30 and len({row["relevant_ids"] for row in rows}) == 30
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/context/songs"))
    parser.add_argument("--context-source", type=Path,
                        default=Path("experiments/namuwiki/queries_context_v01.json"))
    parser.add_argument("--regression-csv", type=Path,
                        default=Path("experiments/reranking/eval_queries_v05.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/context/eval_blind_step10_draft.csv"))
    args = parser.parse_args()
    if not args.artifacts.is_dir():
        raise ValueError("Context artifacts are missing")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed work: {args.output}")
    # This CSV includes private fact snippets. Enforce the existing ignored data root.
    if args.output.resolve().is_relative_to(Path("data/context").resolve()) is False:
        raise ValueError("Private draft must remain under data/context/")
    rows = draft_rows(args.artifacts,
                      previous_song_ids(args.context_source, args.regression_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unreviewed slots to {args.output}; never register this draft as-is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
