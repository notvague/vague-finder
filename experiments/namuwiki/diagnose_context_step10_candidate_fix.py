"""Paired diagnosis of the candidate-cut/evidence fix (not a new blind test).

Run with the backend stopped: embedded Qdrant only supports one owner. The
original blind labels, its failed report/checkpoint, and its old analysis cache
must remain unchanged. Only Context clues are reconstructed from that frozen
analysis using the currently installed safeguard; every other field is fixed.
The previous clue-only diagnostic is also fixed for an incremental comparison.
All outputs, including retrieved fact text, remain under ignored artifacts/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from experiments.namuwiki.evaluate_context_step10 import (
    CODE_FILES, _blind_queries, _corpus, _hash, _item, _preflight_labels, _search,
)
from experiments.namuwiki.context_step10_core import Setting
from src.backend.api.dependencies import close_vector_client
from src.backend.schemas.query import QueryAnalysis
from src.retrieval.analysis_cache import load_cache, looks_like_fallback
from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.explain import ExplainRecorder


DEFAULT_IDS = ("blind005", "blind006", "blind007", "blind011", "blind012", "blind013")
OLD = Path("artifacts/context_eval_step10_v2_blind_new")
PREVIOUS = Path("artifacts/context_eval_step10_after_fix")
OUT = Path("artifacts/context_eval_step10_candidate_fix")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _private(path: Path) -> Path:
    if not path.is_relative_to(Path("artifacts")) or ".." in path.parts:
        raise ValueError("diagnostic input/output must stay in ignored artifacts/")
    return path


def repaired_analysis(original: QueryAnalysis, query: str) -> QueryAnalysis:
    """Isolate new Context safeguards while preserving old model fields."""
    if original.original_query != query or looks_like_fallback(original):
        raise ValueError("frozen analysis differs from the query or is a fallback")
    before = original.model_dump(mode="json")
    after = QueryAnalysis.model_validate(apply_context_query_safeguards(query, before))
    updated = after.model_dump(mode="json")
    if {k: v for k, v in before.items() if k != "context_clues"} != {
        k: v for k, v in updated.items() if k != "context_clues"
    }:
        raise RuntimeError("Context-only repair changed another analysis field")
    return after


def _rank(ids: list[str], relevant: frozenset[str]) -> int | None:
    return next((rank for rank, song_id in enumerate(ids, 1)
                 if song_id in relevant), None)


class _RecordingRouter:
    """Observe the existing evaluation search without changing its logic."""
    def __init__(self, router, recorder):
        self.router, self.recorder = router, recorder

    async def search(self, *args, **kwargs):
        return await self.router.search(*args, recorder=self.recorder, **kwargs)


def _target_scores(recorder, relevant, candidates):
    results = []
    for song_id in sorted(relevant):
        song = recorder.record.get(song_id)
        results.append(dict(
            song_id=song_id, in_final_pool=song_id in candidates,
            context_paths=[dict(rank=path.rank, contribution=path.delta)
                           for path in song.live_paths() if path.path == "context"],
            fused_score=song.fused_score,
            adjustments=[dict(rule=item.rule, delta=item.delta) for item in song.adjustments],
            score_before_cut=(song.fused_score + song.adjustment_total()
                              if song.fused_score is not None else None),
        ))
    return results


def _counts(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if (row["group"] == name if name not in
                {"context_positive", "regression_all"} else
                (row["source"] == "context" and row["group"] != "context_control"
                 if name == "context_positive" else row["source"] == "regression"))]
    return {
        "n": len(selected),
        "clues": sum(row["context_clue_count"] > 0 for row in selected),
        "original_off_candidate30": sum(bool(row["original"]["off_candidate_rank"]) for row in selected),
        "original_on_candidate30": sum(bool(row["original"]["on_candidate_rank"]) for row in selected),
        "after_off_candidate30": sum(bool(row["after"]["off_candidate_rank"]) for row in selected),
        "after_on_candidate30": sum(bool(row["after"]["on_candidate_rank"]) for row in selected),
        "original_off_top10": sum(bool(row["original"]["off_top10_rank"]) for row in selected),
        "original_on_top10": sum(bool(row["original"]["on_top10_rank"]) for row in selected),
        "after_off_top10": sum(bool(row["after"]["off_top10_rank"]) for row in selected),
        "after_on_top10": sum(bool(row["after"]["on_top10_rank"]) for row in selected),
        "previous_on_candidate30": sum(bool(row["previous"]["on_candidate_rank"]) for row in selected),
        "previous_on_top10": sum(bool(row["previous"]["on_top10_rank"]) for row in selected),
        "after_off_hit10_rate": (sum(bool(row["after"]["off_top10_rank"])
                                        for row in selected) / len(selected) if selected else None),
        "after_on_hit10_rate": (sum(bool(row["after"]["on_top10_rank"])
                                       for row in selected) / len(selected) if selected else None),
        "after_off_mrr10": (sum(1 / row["after"]["off_top10_rank"] for row in selected
                                  if row["after"]["off_top10_rank"]) / len(selected) if selected else None),
        "after_on_mrr10": (sum(1 / row["after"]["on_top10_rank"] for row in selected
                                 if row["after"]["on_top10_rank"]) / len(selected) if selected else None),
        "target_evidence_labeled_true": sum(row["target_evidence_labeled"] is True for row in selected),
        "target_evidence_labeled_false": sum(row["target_evidence_labeled"] is False for row in selected),
        "after_off_to_on_candidate_gained": [row["query_id"] for row in selected
                                                if not row["after"]["off_candidate_rank"]
                                                and row["after"]["on_candidate_rank"]],
        "after_off_to_on_candidate_lost": [row["query_id"] for row in selected
                                              if row["after"]["off_candidate_rank"]
                                              and not row["after"]["on_candidate_rank"]],
        "after_off_to_on_top10_gained": [row["query_id"] for row in selected
                                         if not row["after"]["off_top10_rank"]
                                         and row["after"]["on_top10_rank"]],
        "after_off_to_on_top10_lost": [row["query_id"] for row in selected
                                       if row["after"]["off_top10_rank"]
                                       and not row["after"]["on_top10_rank"]],
    }


async def run(args: argparse.Namespace) -> int:
    old_dir = _private(args.old_output)
    previous_dir = _private(args.previous_output)
    output = _private(args.output_dir)
    if output.resolve() in {old_dir.resolve(), previous_dir.resolve()}:
        raise ValueError("choose a separate output directory; former results must stay frozen")
    original_report = _read(old_dir / "blind_report.json")
    original_state = _read(old_dir / "blind_checkpoint.json")
    previous_report_path = previous_dir / "diagnostic_report.json"
    previous_report = _read(previous_report_path)
    previous_stamp = previous_report["stamp"]
    previous_rows = {row["query_id"]: row for row in previous_report["rows"]}
    old_stamp = original_report["stamp"]
    if (original_state["stamp"] != old_stamp
            or original_report["verdict"]["status"] != "not_recommended"):
        raise ValueError("the frozen, failed blind checkpoint/report has changed")
    if _hash(args.blind_csv) != old_stamp["blind_csv_sha256"]:
        raise ValueError("blind labels differ from the frozen failed run")
    old_cache_path = old_dir / "blind_analysis_cache.json"
    if _hash(old_cache_path) != old_stamp["blind_analysis_cache_sha256"]:
        raise ValueError("the old analysis cache has changed since the failed run")
    existing, source_hashes, _ = _preflight_labels()
    if source_hashes != old_stamp["sources"]:
        raise ValueError("reviewed dev/test sources changed since the failed run")
    queries, cases = _blind_queries(args.blind_csv, existing)
    chosen = (list(queries) if args.all else list(dict.fromkeys(args.query_ids)))
    if not chosen or any(query_id not in queries for query_id in chosen):
        raise ValueError("unknown/empty query ID selection")
    cache = load_cache(old_cache_path)
    code = {str(path): _hash(path) for path in CODE_FILES}
    for path, digest in old_stamp["code_sha256"].items():
        if path not in code:
            raise ValueError(f"missing former code path: {path}")
        if not path.endswith(("search_router.py", "context_query.py", "context_evidence.py")) and code[path] != digest:
            raise ValueError(f"other retrieval/evaluation code changed since blind: {path}")
    if code["src/retrieval/context_query.py"] == old_stamp["code_sha256"]["src/retrieval/context_query.py"]:
        raise ValueError("Context query fix was not installed")

    if (previous_stamp["schema"] != "disclosed_step10_context_fix_diagnostic_v1"
            or previous_stamp["blind_csv_sha256"] != _hash(args.blind_csv)
            or previous_stamp["old_report_sha256"] != _hash(old_dir / "blind_report.json")
            or previous_stamp["frozen_analysis_sha256"] != _hash(old_cache_path)
            or previous_stamp["analysis_cache_code_sha256"] != _hash(Path("src/retrieval/analysis_cache.py"))):
        raise ValueError("previous clue diagnostic has different frozen inputs or cache code")
    for path, digest in previous_stamp["code_sha256"].items():
        if not path.endswith(("search_router.py", "context_evidence.py")) and code.get(path) != digest:
            raise ValueError(f"unexpected change since the clue-only diagnostic: {path}")
    if (code["src/retrieval/search_router.py"] == previous_stamp["code_sha256"]["src/retrieval/search_router.py"]
            or code["src/retrieval/context_evidence.py"] == previous_stamp["code_sha256"]["src/retrieval/context_evidence.py"]):
        raise ValueError("install both candidate-cut and evidence fixes before running")
    if any(query_id not in previous_rows for query_id in chosen):
        raise ValueError("run the previous clue-only diagnostic for all selected rows first")

    rows = [queries[query_id] for query_id in chosen]
    original_results = original_state["runs"][original_report["locked_setting"]]
    setting = Setting(**original_report["verdict"]["setting"])
    if setting.key != original_report["locked_setting"] or setting.candidate_k != 30:
        raise ValueError("the locked original setting is not the expected Candidate@30 trial")
    if previous_stamp["setting"] != setting.as_dict():
        raise ValueError("previous diagnostic used different ranking settings")
    corpus, router, indexed = _corpus(rows)
    if any(not (row.relevant_ids <= indexed) for row in rows):
        raise ValueError("a labeled song is missing from the Text catalogue")
    for name in ("context_build_id", "source_song_count", "dense_record_count",
                 "sparse_profile_count", "text_points", "context_manifest_sha256",
                 "state_manifest_sha256"):
        if corpus[name] != old_stamp["corpus"][name]:
            raise ValueError(f"corpus changed since original blind: {name}")
    if corpus != previous_stamp["corpus"]:
        raise ValueError("the active corpus differs from the previous clue-only diagnostic")

    stamp = dict(schema="disclosed_step10_candidate_cut_diagnostic_v1",
                 blind_csv_sha256=_hash(args.blind_csv),
                 old_report_sha256=_hash(old_dir / "blind_report.json"),
                 frozen_analysis_sha256=_hash(old_cache_path),
                 code_sha256=code,
                 diagnostic_code_sha256=_hash(Path(__file__)),
                 previous_report_sha256=_hash(previous_report_path),
                 analysis_cache_code_sha256=_hash(Path("src/retrieval/analysis_cache.py")),
                 corpus=corpus, setting=setting.as_dict(),
                 rerank=False, candidate_k=30, top_k=10,
                 analysis_method="old frozen model analysis + current Context safeguard only")
    checkpoint_path = output / "diagnostic_checkpoint.json"
    state = _read(checkpoint_path) if checkpoint_path.is_file() else dict(stamp=stamp, rows={})
    if state["stamp"] != stamp:
        raise ValueError("diagnostic checkpoint has other code/inputs; choose a new output directory")
    router._context_weight = setting.weight
    router._context_fact_k = setting.fact_k
    router._context_sparse_k = setting.sparse_k
    router._context_named_media_multiplier = setting.named_media_multiplier

    try:
        for position, row in enumerate(rows, 1):
            if row.query_id in state["rows"]:
                print(f"[{position}/{len(rows)}] {row.query_id} reused", flush=True)
                continue
            entry = cache.entries.get(row.query_id)
            if entry is None or entry.get("fallback") or row.query_id not in original_results:
                raise ValueError(f"{row.query_id}: frozen analysis or original result missing")
            analysis = repaired_analysis(cache.get(row.query_id, row.query), row.query)
            off = await _search(router, analysis, setting=None, candidate_k=30)
            recorder = ExplainRecorder(row.query)
            on = await _search(_RecordingRouter(router, recorder), analysis,
                               setting=setting, candidate_k=30)
            old = original_results[row.query_id]
            if off["candidates"] != old["ref_candidate_ids"].split("|"):
                raise RuntimeError(f"{row.query_id}: OFF@30 baseline changed; isolate the cause before comparing")
            result = _item(row, setting, analysis, off, on, off, indexed, cases)
            evidence = result.pop("displayed_evidence")
            target = row.relevant_ids
            previous = previous_rows[row.query_id]
            if len(analysis.context_clues) != previous["context_clue_count"]:
                raise ValueError(f"{row.query_id}: reconstructed clues differ from the previous diagnostic")
            earlier_evidence = next((item["retrieved"] for item in
                previous_report["evidence_needs_review"] if item["query_id"] == row.query_id), [])
            prior_records = {item["record_id"] for item in earlier_evidence}
            current_records = {item["record_id"] for item in evidence}
            state["rows"][row.query_id] = dict(
                query_id=row.query_id, group=row.group, source=row.source,
                context_clue_count=result["context_clue_count"],
                context_clues=[clue.model_dump(mode="json") for clue in analysis.context_clues],
                analysis_metadata=dict(artist_name=analysis.artist_name,
                                       artist_name_alt=analysis.artist_name_alt,
                                       song_title=analysis.song_title,
                                       vocal_gender=analysis.vocal_gender, genre=analysis.genre,
                                       modality_weights=analysis.modality_weights.model_dump(mode="json")),
                context_candidate_count=result["context_candidate_count"],
                target_had_context_hit=bool(result["target_had_context_hit"]),
                target_scores_before_cut=_target_scores(recorder, target, on["candidates"]),
                original=dict(off_candidate_rank=old["off_candidate_rank"],
                              on_candidate_rank=old["on_candidate_rank"],
                              off_top10_rank=old["off_top10_rank"],
                              on_top10_rank=old["on_top10_rank"]),
                previous=previous["after"],
                after=dict(off_candidate_rank=_rank(off["candidates"], target),
                           on_candidate_rank=_rank(on["candidates"], target),
                           off_top10_rank=_rank(off["top"], target),
                           on_top10_rank=_rank(on["top"], target)),
                off_top10_ids=off["top"], on_top10_ids=on["top"],
                evidence_removed_since_previous=sorted(prior_records - current_records),
                evidence_added_since_previous=sorted(current_records - prior_records),
                target_evidence_record_id=result["target_evidence_record_id"],
                target_evidence_labeled=result["target_evidence_labeled"],
                # This stays in ignored artifacts and needs manual fact review.
                evidence_for_review=evidence,
            )
            _write(checkpoint_path, state)
            current = state["rows"][row.query_id]
            print(f"[{position}/{len(rows)}] {row.query_id} "
                  f"clues={current['context_clue_count']} "
                  f"C@30 {current['after']['off_candidate_rank']}→"
                  f"{current['after']['on_candidate_rank']} "
                  f"Top10 {current['after']['off_top10_rank']}→"
                  f"{current['after']['on_top10_rank']} "
                  f"fact_label={current['target_evidence_labeled']}", flush=True)
        from experiments.namuwiki.evaluate_context_step10 import _assert_unchanged
        _assert_unchanged(rows, corpus)
        selected = [state["rows"][row.query_id] for row in rows]
        summary = dict(
            label="disclosed repair diagnostic; NOT a fresh blind pass or deployment gate",
            stamp=stamp, evaluated=len(selected), requested_query_ids=chosen,
            groups={name: _counts(selected, name) for name in
                    ("context_positive", "context_context", "context_mixed",
                     "context_control", "regression_all", "regression_lyrics",
                     "regression_mood_sound", "regression_image")},
            rows=[{key: value for key, value in row.items() if key != "evidence_for_review"}
                  for row in selected],
            evidence_needs_review=[dict(query_id=row["query_id"],
                                        retrieved=row["evidence_for_review"])
                                   for row in selected if row["evidence_for_review"]],
        )
        _write(output / "diagnostic_report.json", summary)
        for name, group in summary["groups"].items():
            if group["n"]:
                print(f"{name}: n={group['n']} C@30 {group['after_off_candidate30']}→"
                      f"{group['after_on_candidate30']} Hit@10 "
                      f"{group['after_off_top10']}→{group['after_on_top10']} "
                      f"gain={group['after_off_to_on_top10_gained']} "
                      f"loss={group['after_off_to_on_top10_lost']}")
        print("Disclosed candidate/evidence diagnostic only. Private report:",
              output / "diagnostic_report.json")
        return 0
    finally:
        close_vector_client()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path, default=OLD)
    parser.add_argument("--previous-output", type=Path, default=PREVIOUS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--blind-csv", type=Path,
                        default=Path("data/context/eval_blind_step10.csv"))
    parser.add_argument("--all", action="store_true", help="Diagnose all 30 exposed labels")
    parser.add_argument("--query-ids", nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
