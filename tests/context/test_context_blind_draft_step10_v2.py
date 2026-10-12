"""A blind draft excludes old labels and refuses to imply a human review."""

import json

from experiments.namuwiki.prepare_context_blind_draft import draft_rows


def test_balanced_draft_never_reuses_old_targets_or_pretends_to_be_reviewed(tmp_path):
    artifact_dir = tmp_path / "songs"
    artifact_dir.mkdir()
    for category_index in range(6):
        for n in range(6):
            song_id = str(1000 + category_index * 10 + n)
            artifact = {
                "song": {"song_id": song_id}, "status": "ok",
                "source": {"url": "https://namu.wiki/w/example"},
                "retrieval": {"records": [{
                    "record_id": f"nw:{song_id}:fact", "category": f"category{category_index}",
                    "evidence_text": f"test fact {song_id}",
                }]},
            }
            (artifact_dir / f"{song_id}.json").write_text(json.dumps(artifact), encoding="utf-8")
    for n in range(16):
        song_id = str(3000 + n)
        (artifact_dir / f"{song_id}.json").write_text(json.dumps({
            "song": {"song_id": song_id}, "status": "not_found",
            "retrieval": {"records": []},
        }), encoding="utf-8")

    old = {"1000", "3000"}
    first = draft_rows(artifact_dir, old)
    assert first == draft_rows(artifact_dir, old)
    assert len(first) == len({row["relevant_ids"] for row in first}) == 30
    assert not {row["relevant_ids"] for row in first} & old
    assert {row["group"] for row in first} == {
        "context_context", "context_mixed", "context_control", "regression_lyrics",
        "regression_mood_sound", "regression_image",
    }
    assert len({row["clue_category"] for row in first if row["evidence_record_ids"]}) == 6
    assert all(not row["query"] and not row["review_note"] for row in first)
    assert all(row["evidence_record_ids"] or not row["reference_fact_for_author"]
               for row in first)
