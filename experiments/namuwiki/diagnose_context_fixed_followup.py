"""Rerun disclosed dev/test with the exact previous analysis, without Gemini.

This diagnostic isolates retrieval changes. It never edits an analysis, calls
QueryAnalyzer, treats a rerun as independent, or reads the new 60 cases.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import traceback
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv

from experiments.namuwiki import evaluate_context_fixed as fixed
from experiments.namuwiki import evaluate_context_step10 as old
from experiments.namuwiki.context_fixed_core import SETTING, fixed_report
from src.backend.schemas.query import QueryAnalysis


PRIOR_FILES = ("registration.json", "dev_report.json", "test_report.json",
               "dev_analysis_cache.json", "test_analysis_cache.json")
DEFAULT_PRIOR = Path("artifacts/context_fixed_validation_20261005_feedback.zip")


def read_prior(archive: Path) -> dict:
    """Read only named reports, verify their exported byte hashes."""
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or not set((*PRIOR_FILES, "export_manifest.json")) <= set(names):
            raise ValueError("prior feedback has missing or duplicate report members")
        manifest = json.loads(z.read("export_manifest.json").decode("utf-8-sig"))
        if manifest.get("schema") != "context_fixed_feedback_v1":
            raise ValueError("unsupported prior feedback format")
        values = {}
        for name in PRIOR_FILES:
            if z.getinfo(name).file_size > 32 * 1024 * 1024:
                raise ValueError("prior report is unexpectedly large")
            content = z.read(name)
            checksum = hashlib.sha256(content).hexdigest()
            if manifest.get("files", {}).get(name) != {"bytes": len(content), "sha256": checksum}:
                raise ValueError(f"prior feedback checksum mismatch: {name}")
            values[name] = json.loads(content.decode("utf-8-sig"))
            values[name + ".sha256"] = checksum
    if values["registration.json"].get("setting") != SETTING.as_dict():
        raise ValueError("prior feedback used different Context settings")
    if values["registration.json"].get("rerank_enabled") is not False:
        raise ValueError("prior feedback was not rerank OFF")
    return values


def frozen_analyses(prior: dict, selected: dict, phase: str) -> tuple[dict, dict]:
    report = prior[phase + "_report.json"]
    cache = prior[phase + "_analysis_cache.json"]
    registration = prior["registration.json"]
    if (report.get("registration") != registration or report.get("phase") != phase
            or report.get("evaluated") != len(selected)
            or report.get("analysis_cache_sha256") != prior[phase + "_analysis_cache.json.sha256"]):
        raise ValueError("prior report/cache registration or inventory mismatch")
    registration_sha = hashlib.sha256(json.dumps(
        registration, sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    if (cache.get("meta", {}).get("fixed_registration_sha256") != registration_sha
            or cache.get("meta", {}).get("fixed_analysis_batch") != phase):
        raise ValueError("prior cache does not belong to this registered split")
    rows = report["question_results"]
    before = {r["query_id"]: r for r in rows}
    if len(rows) != len(before) or set(before) != set(selected) or set(cache["entries"]) != set(selected):
        raise ValueError("prior report/cache has duplicate, missing or extra questions")
    analyses = {}
    for qid, query in selected.items():
        entry, row = cache["entries"][qid], before[qid]
        if (entry.get("fallback") is not False or entry.get("query") != query.query
                or row.get("query") != query.query or entry.get("analysis") != row.get("analysis")):
            raise ValueError(f"{qid}: prior analysis was fallback, edited or mismatched")
        value = QueryAnalysis.model_validate(entry["analysis"])
        if value.original_query != query.query or value.model_dump(mode="json") != entry["analysis"]:
            raise ValueError(f"{qid}: current schema would change the frozen analysis")
        analyses[qid] = value
    return analyses, before


def compare_row(before: dict, after: dict) -> dict:
    """Separate OFF baseline drift from the ON change being investigated."""
    metrics = ("candidate_rank", "top10_rank", "candidate_hit30", "hit1", "hit5", "hit10", "mrr10")
    drift = [key for key in (*metrics, "candidate_ids", "top10_ids")
             if before["off_" + key] != after["off_" + key]]
    old_facts = {(e["song_id"], e["record_id"]) for e in before["displayed_evidence"]}
    new_facts = {(e["song_id"], e["record_id"]) for e in after["displayed_evidence"]}
    return {
        "query_id": after["query_id"], "off_baseline_drift": drift,
        "previous_on_candidate_rank": before["on_candidate_rank"],
        "previous_on_top10_rank": before["on_top10_rank"],
        "fixed_on_candidate_rank": after["on_candidate_rank"],
        "fixed_on_top10_rank": after["on_top10_rank"],
        "removed_displayed_evidence": [dict(song_id=sid, record_id=rid) for sid, rid in sorted(old_facts - new_facts)],
        "retained_displayed_evidence": [dict(song_id=sid, record_id=rid) for sid, rid in sorted(old_facts & new_facts)],
    }


class ObservedContext:
    """Observe raw ranks after the real call; labels never affect the search."""
    def __init__(self, search):
        self.search = search
        self.targets = frozenset()
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.search, name)

    def search_fused_songs(self, query, **kwargs):
        hits = self.search.search_fused_songs(query, **kwargs)
        ranks = {}
        for rank, hit in enumerate(hits, 1):
            if hit.song_id in self.targets:
                ranks[hit.song_id] = dict(
                    fused_rank=rank, dense_rank=hit.dense_rank, sparse_rank=hit.sparse_rank,
                    candidate_fact_id=hit.dense_song.best_fact.record_id if hit.dense_song else None,
                )
        self.calls.append(dict(
            query=query, fact_k=kwargs.get("fact_k"), sparse_k=kwargs.get("sparse_k"),
            media_targets=list(kwargs.get("media_targets", ())),
            fused_song_count=len(hits), first10_ids=[h.song_id for h in hits[:10]],
            target_raw_ranks=ranks,
        ))
        return hits


def _stamp(prior_file: Path, corpus: dict) -> dict:
    code = {*fixed._code_paths(), Path("experiments/namuwiki/diagnose_context_fixed_followup.py"),
            *old.SOURCE_FILES}
    return dict(schema="context_fixed_frozen_diagnostic_v1",
                prior_feedback_sha256=old._hash(prior_file), corpus=corpus,
                code_sha256={str(p): old._hash(p) for p in sorted(code)},
                setting=SETTING.as_dict(), environment=fixed._environment(), runtime=fixed._runtime(),
                analysis_policy="exact prior analysis; no QueryAnalyzer calls; disclosed diagnostic only")


def _write_summary(root: Path, reports: dict, *, status: str) -> None:
    value = dict(status=status, schema="context_fixed_followup_diagnostic_v1",
                 reports=reports, independent=False, fresh_analysis=False,
                 policy="This disclosed diagnostic is not a fresh or independent performance validation.")
    old._write(root / "diagnostic_report.json", value)
    lines = ["# 기존 분석을 보존한 Context 수정 진단", "", f"판정: **{status}**", "",
             "기존 dev/test 106문항의 분석을 수정 없이 재사용했다. 새 독립 평가가 아니다.",
             "OFF 기준선 변동과 ON의 수정 효과를 별도로 검사했다.", "",
             "| 분할·군 | n | Candidate@30 OFF→ON | Hit@10 OFF→ON | MRR@10 OFF→ON |",
             "|---|---:|---:|---:|---:|"]
    for phase, report in reports.items():
        for group in ("context_positive", "regression_all", "regression_image"):
            info = report["groups"].get(group)
            if not info or not info["queries"]:
                continue
            cells = [f"{info['metrics'][key]['off']:.3f}→{info['metrics'][key]['on']:.3f}"
                     for key in ("candidate_hit30", "hit10", "mrr10")]
            lines.append(f"| {phase} {group} | {info['queries']} | " + " | ".join(cells) + " |")
        lines.extend(["", f"{phase} 문제: {report['violations'] or '없음'}",
                      f"{phase} OFF 기준선 변동: {report['off_baseline_drift_query_ids'] or '없음'}",
                      f"{phase} 추가 근거 검토: {report['evidence']['unreviewed_count']}건."])
        interesting = [r for r in report["comparisons"]
                       if r["previous_on_candidate_rank"] != r["fixed_on_candidate_rank"]
                       or r["previous_on_top10_rank"] != r["fixed_on_top10_rank"]
                       or r["removed_displayed_evidence"]]
        lines += ["", "| 질의 | 이전 ON 후보→수정 ON 후보 | 이전 ON Top10→수정 ON Top10 |",
                  "|---|---|---|"]
        for row in interesting:
            lines.append(f"| {row['query_id']} | {row['previous_on_candidate_rank']}→{row['fixed_on_candidate_rank']} | "
                         f"{row['previous_on_top10_rank']}→{row['fixed_on_top10_rank']} |")
    lines += ["", "이 진단을 통과해도 새 분석의 dev/test와 독립 60문항을 다시 측정해야 한다.",
              "raw_context_calls의 순위는 Context 내부 순위다. 최종 Candidate@30과 구분한다.", ""]
    (root / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args) -> int:
    from src.backend.api.dependencies import close_vector_client
    if os.getenv("SEARCH_REFERENCE_YEAR") != "2026":
        raise ValueError("SEARCH_REFERENCE_YEAR=2026 is required")
    load_dotenv()
    prior = read_prior(args.prior_feedback)
    queries, sources, cases = old._preflight_labels()
    if any(sha != prior["registration.json"]["source_hashes"].get(name) for name, sha in sources.items()):
        raise ValueError("disclosed evaluation inputs differ from the previous run")
    reports = {}
    try:
        for phase in ("dev", "test"):
            selected = {qid: q for qid, q in queries.items() if q.split == phase}
            analyses, previous = frozen_analyses(prior, selected, phase)
            corpus, router, indexed = old._corpus(list(selected.values()))
            corpus.update(fixed._modality_identity(corpus, list(selected.values()), cases))
            corpus.update(fixed.text_bm25_identity(router))
            if corpus != prior[phase + "_report.json"]["corpus"]:
                raise ValueError("corpus/BM25/modality content differs from the frozen baseline")
            fixed._configure(router)
            stamp = _stamp(args.prior_feedback, corpus)
            registration_file = args.output_dir / "registration.json"
            if registration_file.exists() and old._read(registration_file) != stamp:
                raise ValueError("diagnostic code/corpus/settings changed; use a new output directory")
            old._write(registration_file, stamp)
            checkpoint = args.output_dir / (phase + "_checkpoint.json")
            state = old._read(checkpoint) if checkpoint.exists() else dict(stamp=stamp, rows={}, comparisons={})
            if state["stamp"] != stamp or set(state["rows"]) - set(selected):
                raise ValueError("diagnostic checkpoint belongs to another run")
            observer = ObservedContext(router._context_search)
            underlying = router._context_search
            router._context_search = observer
            try:
                for n, (qid, query) in enumerate(selected.items(), 1):
                    if qid in state["rows"]:
                        continue
                    observer.targets, observer.calls = query.relevant_ids, []
                    required = "image" if query.group == "regression_image" else cases.get(qid, {}).get("required_path", "")
                    value = await fixed.evaluate_question(query, analyses[qid], router, indexed, cases, required_path=required)
                    value["raw_context_calls"] = observer.calls
                    state["rows"][qid] = value
                    state["comparisons"][qid] = compare_row(previous[qid], value)
                    old._write(checkpoint, state)
                    print(f"[frozen {phase} {n}/{len(selected)}] {qid} "
                          f"OFF C@30={value['off_candidate_rank']} "
                          f"ON before={previous[qid]['on_candidate_rank']} after={value['on_candidate_rank']} "
                          f"Top10 before={previous[qid]['on_top10_rank']} after={value['on_top10_rank']}", flush=True)
            finally:
                router._context_search = underlying
            old._assert_unchanged(list(selected.values()), corpus)
            if _stamp(args.prior_feedback, corpus) != stamp:
                raise ValueError("diagnostic code/input/runtime changed during measurement")
            after_modality = fixed._modality_identity(corpus, list(selected.values()), cases)
            after_bm25 = fixed.text_bm25_identity(router)
            if (after_modality != {k: v for k, v in corpus.items() if k.startswith(("image_", "audio_")) or k == "local_collection_sha256"}
                    or after_bm25 != {k: v for k, v in corpus.items() if k.startswith("text_bm25_")}):
                raise ValueError("modality/BM25 content changed during measurement")
            rows = [state["rows"][qid] for qid in selected]
            report = fixed_report(rows, phase=phase, expected=len(selected))
            comparisons = [state["comparisons"][qid] for qid in selected]
            drift = [r["query_id"] for r in comparisons if r["off_baseline_drift"]]
            report["violations"] = sorted(set(report["violations"] + [f"{qid}: prior_off_baseline_drift" for qid in drift]))
            report.update(corpus=corpus, diagnostic_registration=stamp, comparisons=comparisons,
                          off_baseline_drift_query_ids=drift, question_results=rows,
                          fresh_analysis=False, independent=False)
            report["claims"] = [c for c in report["claims"] if "fresh QueryAnalyzer" not in c]
            report["claims"].append("Exact previous analyses reused without repair. This is a disclosed diagnostic, not independent validation.")
            report["status"] = "blocked" if report["violations"] else "evidence_review_required" if report["evidence"]["unreviewed_count"] else "passed"
            old._write(args.output_dir / (phase + "_report.json"), report)
            flat = [{k: v for k, v in r.items() if k != "raw_context_calls"} for r in rows]
            fixed._write_csv(args.output_dir / (phase + "_detail.csv"), flat)
            reports[phase] = report
            fixed._print(report)
            _write_summary(args.output_dir, reports, status="in_progress")
            close_vector_client()
        status = "blocked" if any(r["violations"] for r in reports.values()) else "evidence_review_required" if any(r["evidence"]["unreviewed_count"] for r in reports.values()) else "passed_diagnostic"
        _write_summary(args.output_dir, reports, status=status)
        return 4 if status == "blocked" else 0
    finally:
        close_vector_client()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-feedback", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/context_fixed_followup_20261005/diagnostic"))
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        old._write(args.output_dir / "failure_report.json", dict(
            status="interrupted", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(),
            note="Prior reports remain unchanged. Retry without changing code/corpus/settings.",
        ))
        print(f"Frozen diagnostic interrupted: {type(exc).__name__}: {exc}", flush=True)
        return 2
    (args.output_dir / "failure_report.json").unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
