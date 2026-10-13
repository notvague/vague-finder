"""Paired, reranker-free evaluation of the published partial Context corpus.

Run from the project root with ``python -m experiments.namuwiki.evaluate_context_step8``.
The private query CSVs and all result files stay on the local machine.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


CONTEXT_CSV = Path("experiments/namuwiki/eval_queries_context_v01.csv")
REGRESSION_CSV = Path("experiments/reranking/eval_queries_v05.csv")
CONTEXT_SOURCE = Path("experiments/namuwiki/queries_context_v01.json")
OUTPUT_ROOT = Path("artifacts/context_eval_step8")

# Reviewed against eval_queries_v05.csv. These groups are disjoint; the entire
# 76-query v0.5 set is still evaluated, including the rows labelled "other".
REGRESSION_GROUP_IDS = {
    "lyrics": frozenset("q211 q212 q213 q219 q301 q308 q310 q311".split()),
    "mood_sound": frozenset(
        "q105 q110 q115 q118 q200 q214 q215 q302 q305 q306 q313 q314 "
        "m201 m202 m203".split()
    ),
    "image": frozenset(
        "q204 q205 q206 q210 q300 q304 q307 q309 q312 q315 "
        "m401 m402".split()
    ),
}
CONTEXT_MIXED = frozenset("nw004 nw005 nw008 nw012 nw016 nw025".split())
CONTEXT_CONTROL = frozenset("nw027 nw028 nw029 nw030".split())
REQUIRED_FIELDS = {"query_id", "split", "query_type", "query", "relevant_ids"}
METRICS = ("candidate_hit30", "candidate_recall30", "hit1", "hit5", "hit10", "mrr10")


@dataclass(frozen=True)
class Query:
    query_id: str
    split: str
    source: str
    group: str
    segment: str
    query: str
    relevant_ids: frozenset[str]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _parse_csv(path: Path, source: str) -> list[Query]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns {sorted(REQUIRED_FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no queries")

    groups = {
        song_id: group
        for group, song_ids in REGRESSION_GROUP_IDS.items()
        for song_id in song_ids
    }
    if sum(map(len, REGRESSION_GROUP_IDS.values())) != len(groups):
        raise ValueError("regression groups overlap")
    parsed: list[Query] = []
    seen: set[str] = set()
    for row in rows:
        query_id = row["query_id"].strip()
        query = row["query"].strip()
        split = row["split"].strip().lower()
        relevant_ids = frozenset(
            part.strip() for part in row["relevant_ids"].split("|") if part.strip()
        )
        if (not query_id or query_id in seen or not query or not relevant_ids
                or split not in {"dev", "test"} or row["query_type"].strip() != "search"):
            raise ValueError(f"{path}: invalid/duplicate search row {query_id!r}")
        seen.add(query_id)
        segment = ""
        if source == "context":
            segment = (
                "control" if query_id in CONTEXT_CONTROL else
                "mixed" if query_id in CONTEXT_MIXED else "context"
            )
            group = "context_" + segment
        else:
            group = "regression_" + groups.get(query_id, "other")
        parsed.append(Query(query_id, split, source, group, segment, query, relevant_ids))
    if source == "regression":
        missing = set(groups) - seen
        if missing:
            raise ValueError(f"{path}: expected v0.5 IDs missing: {sorted(missing)}")
    return parsed


def load_queries(
    context_csv: Path, regression_csv: Path, split: str,
    selected_ids: Iterable[str] = (), context_source: Path | None = None,
) -> tuple[list[Query], dict[str, str]]:
    context = _parse_csv(context_csv, "context")
    regression = _parse_csv(regression_csv, "regression")
    all_rows = context + regression
    ids = [row.query_id for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("query IDs overlap across CSVs")
    sources = {"context_csv_sha256_16": _sha(context_csv),
               "regression_csv_sha256_16": _sha(regression_csv)}
    if context_source is not None and context_source.is_file():
        raw = json.loads(context_source.read_text(encoding="utf-8"))
        original = {row["query_id"]: row for row in raw["queries"]}
        if len(original) != len(raw["queries"]) or set(original) != {r.query_id for r in context}:
            raise ValueError("Context CSV and source JSON have different IDs")
        for row in context:
            entry = original[row.query_id]
            if (entry["query"] != row.query or entry["split"] != row.split
                    or str(entry["song_id"]) not in row.relevant_ids
                    or entry["segment"] != row.segment):
                raise ValueError(f"Context source JSON/CSV mismatch: {row.query_id}")
        sources["context_source_sha256_16"] = _sha(context_source)
    else:
        sources["context_source_sha256_16"] = "unavailable"

    wanted = set(selected_ids)
    unknown = wanted - set(ids)
    if unknown:
        raise ValueError(f"unknown query IDs: {sorted(unknown)}")
    selected = [row for row in all_rows
                if (split == "all" or row.split == split)
                and (not wanted or row.query_id in wanted)]
    wrong_split = wanted - {row.query_id for row in selected}
    if wrong_split:
        raise ValueError(f"IDs not in split {split}: {sorted(wrong_split)}")
    if not selected:
        raise ValueError("no queries in the requested split/IDs")
    return selected, sources


def first_rank(ids: Iterable[str], answers: frozenset[str]) -> int | None:
    return next((n for n, song_id in enumerate(ids, 1) if song_id in answers), None)


def metrics(candidate_ids: list[str], top_ids: list[str], answers: frozenset[str]) -> dict[str, float]:
    if not answers:
        raise ValueError("metric denominator is empty; check Text-index availability")
    candidate = candidate_ids[:30]
    top = top_ids[:10]
    rank = first_rank(top, answers)
    return {
        "candidate_hit30": float(first_rank(candidate, answers) is not None),
        "candidate_recall30": len(set(candidate) & answers) / len(answers),
        "hit1": float(first_rank(top[:1], answers) is not None),
        "hit5": float(first_rank(top[:5], answers) is not None),
        "hit10": float(rank is not None),
        "mrr10": 1 / rank if rank is not None else 0.0,
    }


def _ids(items: Iterable[Any]) -> list[str]:
    return [str(item.id) for item in items]


async def paired_search(router: Any, analysis: Any) -> tuple[dict, dict]:
    """The same analysis, candidate cut, and router instance in both arms."""
    arms: list[dict] = []
    for enabled in (False, True):
        candidate_ids: list[str] = []
        context_hits: list[Any] = []
        results = await router.search(
            analysis, top_k=10, candidate_k=30, use_rerank=False,
            use_context=enabled, candidate_ids_out=candidate_ids,
            context_hits_out=context_hits,
        )
        top_ids = _ids(results)
        if (len(candidate_ids) > 30 or len(top_ids) > 10
                or len(set(candidate_ids)) != len(candidate_ids)
                or top_ids != candidate_ids[:len(top_ids)]):
            raise RuntimeError("SearchRouter candidate/top-10 contract changed")
        arms.append({"candidates": candidate_ids, "top": top_ids,
                     "tracks": results, "context_hits": context_hits})
    if not analysis.has_context_clue and arms[0]["candidates"] != arms[1]["candidates"]:
        raise RuntimeError("no-clue query changed even though Context could not run")
    return arms[0], arms[1]


def evaluate_pair(row: Query, baseline: dict, enabled: dict,
                  available_ids: frozenset[str], analysis: Any) -> dict:
    eligible = row.relevant_ids & available_ids
    scoreable = bool(eligible)
    values = {}
    for prefix, arm in (("off", baseline), ("on", enabled)):
        result = metrics(arm["candidates"], arm["top"], eligible) if scoreable else None
        for name in METRICS:
            values[f"{prefix}_{name}"] = result[name] if result else None
        values[f"{prefix}_candidate_rank"] = first_rank(arm["candidates"], eligible)
        values[f"{prefix}_top10_rank"] = first_rank(arm["top"], eligible)
        values[f"{prefix}_candidate_ids"] = "|".join(arm["candidates"])
        values[f"{prefix}_top10_ids"] = "|".join(arm["top"])

    return {
        "source": row.source, "group": row.group, "segment": row.segment,
        "split": row.split, "query_id": row.query_id,
        "relevant_ids": "|".join(sorted(row.relevant_ids)),
        "eligible_ids": "|".join(sorted(eligible)),
        "unavailable_ids": "|".join(sorted(row.relevant_ids - available_ids)),
        "scoreable": int(scoreable),
        "context_clue_count": len([c for c in analysis.context_clues if c.confidence > 0]),
        "context_candidate_count": len(enabled["context_hits"]),
        "target_had_context_hit": int(any(
            hit.song_id in eligible for hit in enabled["context_hits"]
        )),
        **values,
    }


def summarize(rows: list[dict], *, complete: bool = True) -> dict:
    """Only indexed answers enter denominators; missing labels are counted."""
    groups = {"all": rows,
              "context_all": [r for r in rows if r["source"] == "context"],
              "regression_all": [r for r in rows if r["source"] == "regression"],
              "no_context_clue": [r for r in rows if r["context_clue_count"] == 0]}
    for name in ("context_context", "context_mixed", "context_control",
                 "regression_lyrics", "regression_mood_sound", "regression_image",
                 "regression_other"):
        groups[name] = [r for r in rows if r["group"] == name]
    output = {}
    for name, group in groups.items():
        scored = [r for r in group if r["scoreable"]]
        metric_rows = {}
        for metric in METRICS:
            before = sum(r[f"off_{metric}"] for r in scored) / len(scored) if scored else None
            after = sum(r[f"on_{metric}"] for r in scored) / len(scored) if scored else None
            metric_rows[metric] = {
                "off": before, "on": after,
                "delta": after - before if scored else None,
                "gained": [r["query_id"] for r in scored
                           if r[f"on_{metric}"] > r[f"off_{metric}"]],
                "lost": [r["query_id"] for r in scored
                         if r[f"on_{metric}"] < r[f"off_{metric}"]],
            }
        output[name] = {
            "queries": len(group), "scored": len(scored),
            "unavailable_queries": [r["query_id"] for r in group if not r["scoreable"]],
            "partially_unavailable": [r["query_id"] for r in scored if r["unavailable_ids"]],
            "context_clue_queries": sum(r["context_clue_count"] > 0 for r in group),
            "context_hit_queries": sum(r["context_candidate_count"] > 0 for r in group),
            "metrics": metric_rows,
        }
    issues = {
        "missing_context_clue": [r["query_id"] for r in rows
                                  if r["group"] in {"context_context", "context_mixed"}
                                  and r["context_clue_count"] == 0],
        "unexpected_control_clue": [r["query_id"] for r in rows
                                     if r["group"] == "context_control" and r["context_clue_count"]],
        "unexpected_regression_clue": [r["query_id"] for r in rows
                                        if r["group"] in {"regression_lyrics", "regression_mood_sound", "regression_image"}
                                        and r["context_clue_count"]],
    }
    return {"complete": complete, "evaluated": len(rows), "groups": output, "issues": issues}


def write_detail(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def print_summary(summary: dict) -> None:
    print("\n[Step 8] Same analysis · Context OFF/ON · rerank OFF · Candidate@30 / Top-10")
    print("group                    scored/total   C@30 off→on     Top10 off→on   gain/loss (Top10)")
    for name, info in summary["groups"].items():
        if not info["queries"]:
            continue
        c = info["metrics"]["candidate_hit30"]
        t = info["metrics"]["hit10"]
        fmt = lambda value: f"{value:.3f}" if value is not None else "n/a"  # noqa: E731
        print(f"{name:<24} {info['scored']:>3}/{info['queries']:<3}    "
              f"{fmt(c['off'])}→{fmt(c['on'])}       "
              f"{fmt(t['off'])}→{fmt(t['on'])}         "
              f"+{len(t['gained'])}/-{len(t['lost'])}")
    print("Issues:", json.dumps(summary["issues"], ensure_ascii=False))
    for name in ("context_all", "regression_all"):
        group = summary["groups"][name]
        print(f"{name} Top10 lost: {group['metrics']['hit10']['lost']}; "
              f"Candidate@30 lost: {group['metrics']['candidate_hit30']['lost']}; "
              f"unavailable: {group['unavailable_queries']}")


def _live_preflight(rows: list[Query]) -> tuple[frozenset[str], dict, Any]:
    from src.backend.api.dependencies import get_search_router, get_vector_client
    from src.vector_db.qdrant_backend import collection_name
    from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME

    router = get_search_router()
    client = get_vector_client()
    text_collection = collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    count = int(client.client.count(collection_name=text_collection, exact=True).count)
    ids = sorted({song_id for row in rows for song_id in row.relevant_ids})
    indexed = frozenset(router._text_svc.fetch_tracks_by_ids(ids))
    snapshot = router._context_search._snapshot() if router._context_search else None
    if snapshot is None and any(r.source == "context" and r.segment != "control" for r in rows):
        raise RuntimeError("Context Qdrant alias/manifest is absent; load Context before evaluation")
    info = {"text_collection": text_collection, "text_points": count,
            "indexed_label_count": len(indexed), "label_count": len(ids),
            "context_build_id": snapshot.build_id if snapshot else None,
            "context_dense_collection": snapshot.dense_collection if snapshot else None,
            "context_sparse_collection": snapshot.sparse_collection if snapshot else None}
    return indexed, info, router


async def run(args: argparse.Namespace) -> int:
    ids = [part.strip() for part in args.query_ids.split(",") if part.strip()]
    rows, sources = load_queries(args.context_csv, args.regression_csv, args.split,
                                 ids, args.context_source)
    print(f"Selected {len(rows)} queries: Context {sum(r.source == 'context' for r in rows)}, "
          f"v0.5 {sum(r.source == 'regression' for r in rows)}")
    if args.dry_run:
        for row in rows:
            print(f"{row.query_id} {row.split} {row.group}")
        return 0
    if not os.getenv("SEARCH_REFERENCE_YEAR", "").strip():
        raise ValueError("Set SEARCH_REFERENCE_YEAR (e.g. -e SEARCH_REFERENCE_YEAR=2026)")

    from src.backend.api.dependencies import close_vector_client, get_query_analyzer
    from src.retrieval.analysis_cache import (
        analyzer_fingerprint, load_cache, looks_like_fallback, new_cache,
    )

    output = args.output_dir or OUTPUT_ROOT / f"results_{args.split}"
    cache_path = args.analysis_cache or OUTPUT_ROOT / "analysis_cache.json"
    output.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path) if cache_path.is_file() else new_cache(
        source="context_v01+reranking_v05", split="all"
    )
    drift = cache.fingerprint_drift()
    if drift:
        raise ValueError(f"analysis cache fingerprint changed ({drift}); rebuild a new cache")
    # Fail before loading any search models if a query was edited under the same ID.
    for row in rows:
        if row.query_id in cache.entries:
            cache.get(row.query_id, row.query)

    detail_path = output / "paired_detail.csv"
    report_path = output / "paired_summary.json"
    run_path = output / "run_info.json"
    report_path.write_text(json.dumps({"complete": False}, indent=2) + "\n", encoding="utf-8")
    detail_path.unlink(missing_ok=True)  # never leave an old complete detail beside a new partial run
    run_info = {"status": "running", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "split": args.split, "query_ids": ids, "sources": sources,
                "analysis_cache": str(cache_path), "analysis_fingerprint": analyzer_fingerprint(),
                "analysis_cache_sha256_16": None, "candidate_k": 30, "top_k": 10,
                "rerank_enabled": False, "context_conditions": [False, True]}
    run_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        indexed, corpus, router = _live_preflight(rows)
        run_info["corpus"] = corpus
        result_rows: list[dict] = []
        analyzer = None
        for position, row in enumerate(rows, 1):
            if row.query_id in cache.entries and not cache.entries[row.query_id].get("fallback"):
                analysis = cache.get(row.query_id, row.query)
            else:
                analyzer = analyzer or get_query_analyzer()
                analysis = analyzer.analyze(row.query)
                cache.put(row.query_id, row.query, analysis)
                cache.save(cache_path)
            if cache.entries[row.query_id].get("fallback") or looks_like_fallback(analysis):
                raise RuntimeError(f"{row.query_id}: rule fallback analysis; retry after analyzer is available")
            if analysis.original_query.strip() != row.query:
                raise RuntimeError(f"{row.query_id}: cached analysis belongs to a different query")
            before, after = await paired_search(router, analysis)
            result = evaluate_pair(row, before, after, indexed, analysis)
            result_rows.append(result)
            write_detail(detail_path, result_rows)
            print(f"[{position}/{len(rows)}] {row.query_id} ({row.group}) "
                  f"C@30 {result['off_candidate_rank']}→{result['on_candidate_rank']} "
                  f"Top10 {result['off_top10_rank']}→{result['on_top10_rank']} "
                  f"indexed={bool(result['scoreable'])} context_hits={result['context_candidate_count']}",
                  flush=True)
            if args.show_top:
                for label, arm in (("OFF", before), ("ON", after)):
                    print(f"  {label}: " + " | ".join(
                        f"{i}. {track.id} {track.title} / {track.artist}"
                        for i, track in enumerate(arm["tracks"], 1)
                    ), flush=True)
        summary = summarize(result_rows)
        if (summary["groups"]["context_all"]["context_clue_queries"]
                and not summary["groups"]["context_all"]["context_hit_queries"]):
            raise RuntimeError("Context produced no surviving hits on any context query; inspect index/path")
        _, corpus_after, _ = _live_preflight(rows)
        if corpus_after != corpus:
            raise RuntimeError("Text/Context Qdrant changed during paired evaluation")
        run_info["status"] = "complete"
        run_info["analysis_cache_sha256_16"] = _sha(cache_path)
        run_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_summary(summary)
        print(f"Saved: {detail_path} | {report_path} | {run_path}")
        return 0
    finally:
        close_vector_client()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test", "all"], required=True)
    p.add_argument("--context-csv", type=Path, default=CONTEXT_CSV)
    p.add_argument("--regression-csv", type=Path, default=REGRESSION_CSV)
    p.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    p.add_argument("--query-ids", default="", help="comma-separated IDs for a short smoke run")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--analysis-cache", type=Path, default=None)
    p.add_argument("--show-top", action="store_true", help="print titles from local Text Qdrant")
    p.add_argument("--dry-run", action="store_true", help="validate inputs without opening Qdrant/models")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
