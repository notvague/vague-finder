"""Fresh analysis, fixed Context settings, protected regressions, then new cases.

This evaluator never edits .env, selects weights, overwrites older experiments,
or treats a repeated/disclosed question as a new independent observation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.namuwiki.context_fixed_core import EXPECTED_NEW, SETTING, fixed_report, inspect_path
from experiments.namuwiki.evaluate_context_step8 import Query
from experiments.namuwiki import evaluate_context_step10 as old


DEFAULT_ROOT = Path("artifacts/context_fixed_validation_20261005")
NEW_CSV = Path("data/context/eval_independent_fixed_20261005.csv")
NEW_AUDIT = Path("data/context/eval_independent_fixed_20261005_audit.json")
OLD_DISCLOSED = Path("data/context/eval_blind_step10.csv")
CODE_FILES = (*old.CODE_FILES, Path("src/retrieval/analysis_cache.py"),
              Path("src/retrieval/timing.py"), Path("src/vector_db/settings.py"),
              Path("src/retrieval/modality_queries.py"), Path("src/retrieval/search_service.py"),
              Path("experiments/namuwiki/context_fixed_core.py"),
              Path("experiments/namuwiki/evaluate_context_fixed.py"))


def _environment() -> dict:
    prefixes = ("CONTEXT_", "SEARCH_", "LYRIC", "RERANKER_", "KOE5_", "SIGLIP", "CLAP_",
                "TEXT_", "IMAGE_", "AUDIO_", "BM25_", "QDRANT_")
    safe = {k: v for k, v in os.environ.items() if k.startswith(prefixes)
            and k != "QDRANT_URL"
            and not any(secret in k for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    safe.update({k: os.environ.get(k, "") for k in
                 ("VECTOR_BACKEND", "QDRANT_PATH", "GEMINI_MODEL_NAME", "QDRANT_NAMESPACE")})
    return safe


def _runtime() -> dict:
    versions = {}
    for name in ("google-genai", "qdrant-client", "kiwipiepy", "transformers", "torch",
                 "sentence-transformers", "numpy", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"python": platform.python_version(), "packages": versions}


def _code_paths() -> list[Path]:
    # Helper/schema/model changes can affect search as much as the router.
    return sorted(set(CODE_FILES) | set(Path("src").rglob("*.py"))
                  | {Path("requirements.txt")})


def _assert_registered_files(registration: dict) -> None:
    for name, sha in {**registration["source_hashes"], **registration["code_sha256"]}.items():
        if old._hash(Path(name)) != sha:
            raise RuntimeError(f"registered input/code changed during measurement: {name}")
    if _environment() != registration["environment"] or _runtime() != registration["runtime"]:
        raise RuntimeError("registered environment/runtime changed during measurement")


def load_independent(path: Path, audit_path: Path, existing: dict[str, Query],
                     disclosed: Path, *, artifact_dir: Path = Path("artifacts/context/songs")) -> tuple[dict, dict]:
    audit = old._read(audit_path)
    if (audit.get("schema") != "authored_independent_context_review_v1" or audit.get("case_count") != 60
            or audit.get("group_counts") != EXPECTED_NEW or len(audit.get("reviews", [])) != 60):
        raise ValueError("new review manifest schema/inventory changed")
    if old._hash(path) != audit["csv_sha256"] or old._hash(disclosed) != audit["excluded_disclosed_csv_sha256"]:
        raise ValueError("new CSV or disclosed exclusion input changed")
    with disclosed.open(encoding="utf-8-sig", newline="") as stream:
        prior = list(csv.DictReader(stream))
    old_ids = {q.query_id for q in existing.values()} | {r["query_id"] for r in prior}
    excluded = {i for q in existing.values() for i in q.relevant_ids}
    excluded |= {i for r in prior for i in r["relevant_ids"].split("|")}
    if excluded != set(audit["excluded_target_ids"]):
        raise ValueError("previous target inventory changed")
    fields = {"query_id", "query", "relevant_ids", "group", "clue_category",
              "evidence_record_ids", "review_note", "required_path"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("new CSV missing reviewed evaluation fields")
        raw = list(reader)
    queries, cases, seen_songs, seen_texts = {}, {}, set(), set()
    reviews = {r["query_id"]: r for r in audit["reviews"]}
    if len(reviews) != 60:
        raise ValueError("duplicate reviewed query IDs")
    for r in raw:
        qid, query, sid, group = (r[k].strip() for k in ("query_id", "query", "relevant_ids", "group"))
        if (not qid or not query or not sid or "|" in sid or sid in excluded or sid in seen_songs
                or qid in old_ids or qid in queries or query in seen_texts
                or group not in EXPECTED_NEW or not r["review_note"].strip()):
            raise ValueError(f"{qid}: duplicate, reused target, or incomplete new case")
        if r["required_path"] != {"regression_image": "image", "regression_mood_sound": "audio"}.get(group, ""):
            raise ValueError(f"{qid}: required image/audio execution check missing")
        artifact = old._read(artifact_dir / (sid + ".json"))
        if str(artifact["song"]["song_id"]) != sid:
            raise ValueError(f"{qid}: artifact belongs to another song")
        record_ids = [x for x in r["evidence_record_ids"].split("|") if x]
        review = reviews.get(qid, {})
        if str(review.get("song_id")) != sid:
            raise ValueError(f"{qid}: review belongs to another song")
        records = {x["record_id"]: x for x in artifact["retrieval"]["records"]}
        positive = group in {"context_context", "context_mixed"}
        if positive:
            if (artifact["status"] != "ok" or not record_ids or len(set(record_ids)) != len(record_ids)
                    or not artifact["source"].get("url") or any(x not in records for x in record_ids)
                    or review.get("source_url") != artifact["source"]["url"]):
                raise ValueError(f"{qid}: source-bound fact labels missing")
            expected = {x["record_id"]: x["fact_text"] for x in review.get("reviewed_facts", [])}
            if any(expected.get(x) != records[x]["evidence_text"] for x in record_ids):
                raise ValueError(f"{qid}: reviewed fact text changed")
            if any(records[x]["category"] != r["clue_category"] for x in record_ids):
                raise ValueError(f"{qid}: fact category changed")
        elif record_ids or r["clue_category"] != "no_context":
            raise ValueError(f"{qid}: non-Context case has fact labels")
        if group == "context_control" and (artifact["status"] not in {"not_found", "no_trivia"} or records):
            raise ValueError(f"{qid}: no-context control has facts")
        if group == "regression_image" and not (review.get("cover_url")
                and re.fullmatch(r"[0-9a-f]{64}", str(review.get("cover_sha256", "")))
                and review.get("cover_visual_review") == query
                and review.get("indexed_songs_with_same_cover") == 1):
            raise ValueError(f"{qid}: album cover review missing")
        source = "context" if group.startswith("context_") else "regression"
        segment = group.removeprefix("context_") if source == "context" else ""
        queries[qid] = Query(qid, "independent", source, group, segment, query, frozenset({sid}))
        cases[qid] = dict(clue_category=r["clue_category"], required_path=r["required_path"],
                          evidence_record_id=record_ids[0] if record_ids else None,
                          additional_evidence_record_ids=record_ids[1:])
        seen_songs.add(sid)
        seen_texts.add(query)
    categories = Counter(c["clue_category"] for c in cases.values() if c["clue_category"] != "no_context")
    if (Counter(q.group for q in queries.values()) != Counter(EXPECTED_NEW) or set(reviews) != set(queries)
            or dict(categories) != audit.get("fact_category_counts")):
        raise ValueError("independent inventory/review coverage must be exactly 60 questions")
    return queries, cases


def _registration(root: Path, new_csv: Path, new_audit: Path, disclosed: Path) -> tuple[dict, dict, dict, dict]:
    # QueryAnalyzer also loads .env. Capture its effective settings *before*
    # registration so a retry cannot silently register a different environment.
    load_dotenv()
    existing, sources, old_cases = old._preflight_labels()
    new, new_cases = load_independent(new_csv, new_audit, existing, disclosed)
    value = dict(schema="context_fixed_registered_v1", setting=SETTING.as_dict(),
                 source_hashes={**sources, str(new_csv): old._hash(new_csv),
                                str(new_audit): old._hash(new_audit), str(disclosed): old._hash(disclosed)},
                 code_sha256={str(p): old._hash(p) for p in _code_paths()}, runtime=_runtime(),
                 phase_counts={"dev": 74, "test": 32, "independent": 60},
                 reference_year=2026, rerank_enabled=False, environment=_environment(),
                 analysis_policy="new cache in this experiment only; same analysis in OFF/ON",
                 question_exposure="post-review reanalysis of previously observed dev/test/60 questions; not a new blind sample",
                 repeat_policy="first two pure/mixed Context, image, audio and lyric cases, one additional fresh analysis each")
    file = root / "registration.json"
    if file.exists() and old._read(file) != value:
        raise ValueError("registered code/input/settings changed; choose a new output directory, keep this run")
    if not file.exists():
        if (any(root.glob("*checkpoint.json")) or any(root.glob("*analysis_cache.json"))
                or any(p.name != "failure_report.json" for p in root.glob("*_report.json"))
                or (root / "corpus_identity.json").exists()):
            raise ValueError("unregistered old evaluation state in output directory")
        old._write(file, value)
    for target, data in ((root / "registered_new_labels.csv", new_csv.read_bytes()),
                         (root / "registered_new_review.json", new_audit.read_bytes())):
        if target.exists() and target.read_bytes() != data:
            raise ValueError("registered review/label copy changed")
        if not target.exists():
            target.write_bytes(data)
    return value, {**existing, **new}, {**old_cases, **new_cases}, existing


def _modality_identity(corpus: dict, selected: list[Query], cases: dict) -> dict:
    from src.backend.api.dependencies import get_vector_client
    from src.vector_db.qdrant_backend import collection_name
    from src.vector_db.settings import AUDIO_INDEX_NAME, IMAGE_INDEX_NAME, NAMESPACE

    client = get_vector_client()
    details = {}
    for kind, index_name in (("image", IMAGE_INDEX_NAME), ("audio", AUDIO_INDEX_NAME)):
        collection = collection_name(index_name, NAMESPACE)
        details[kind + "_collection"] = collection
        details[kind + "_points"] = int(client.client.count(collection_name=collection, exact=True).count)
        required = {sid for q in selected if cases.get(q.query_id, {}).get("required_path") == kind
                    or kind == "image" and q.group == "regression_image" for sid in q.relevant_ids}
        if required and set(client.Index(index_name).fetch(ids=sorted(required), namespace=NAMESPACE)["vectors"]) != required:
            raise ValueError(f"{kind} labels absent from the actual modality index")
    if not os.getenv("QDRANT_URL"):
        local = Path(os.getenv("QDRANT_PATH") or "artifacts/qdrant") / "collection"
        collections = {corpus[k] for k in ("text_collection", "context_dense_collection", "context_sparse_collection")}
        collections |= {details["image_collection"], details["audio_collection"]}
        details["local_collection_sha256"] = {name: old._hash(local / name / "storage.sqlite") for name in sorted(collections)}
    return details


def _configure(router: Any) -> None:
    # Validate the actual factory; never overwrite settings just to match a report.
    from src.retrieval.context_search_settings import assert_fixed_context_settings
    assert_fixed_context_settings(router)


def text_bm25_identity(router: Any) -> dict:
    encoder = router._text_svc.bm25_encoder
    path = Path(encoder.params_path)
    if not getattr(encoder, "_is_fitted", False) or not path.is_file():
        raise ValueError("Text BM25 parameters unavailable; restore the working baseline before measuring Context")
    parameters = old._read(path)
    if int(parameters.get("n_docs", 0)) <= 0:
        raise ValueError("Text BM25 document statistics are empty")
    return dict(text_bm25_path=str(path), text_bm25_sha256=old._hash(path),
                text_bm25_fit_documents=int(parameters["n_docs"]))


def lock_corpus(root: Path, corpus: dict) -> None:
    path = root / "corpus_identity.json"
    if path.exists() and old._read(path) != corpus:
        raise ValueError("registered corpus content changed; stop before new analysis/search")
    if not path.exists():
        old._write(path, corpus)


def fresh_analyses(queries: dict, path: Path, registration: dict, phase: str) -> tuple[dict, str]:
    from src.retrieval.analysis_cache import new_cache

    run_sha = hashlib.sha256(json.dumps(registration, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    binding = dict(fixed_registration_sha256=run_sha, fixed_analysis_batch=phase)
    if path.exists():
        meta = old._read(path)["meta"]
        if any(meta.get(k) != v for k, v in binding.items()):
            raise ValueError("analysis cache was not created by this registered experiment/batch")
    else:
        cache = new_cache(source="context_fixed_validation_20261005", split=phase)
        cache.meta.update(binding)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.save(path)
    return old._analyses(queries, path, populate=True)


async def _arm(router: Any, analysis: Any, enabled: bool) -> dict:
    from src.retrieval.timing import TimingRecorder
    from src.retrieval.context_evidence import context_evidence_for_result
    from src.retrieval.explain import ExplainRecorder

    timer = TimingRecorder(query=analysis.original_query)
    candidates, hits = [], []
    explain = ExplainRecorder(analysis.original_query)
    before_analysis = analysis.model_dump(mode="json")
    started = time.perf_counter()
    with timer.step("search"):
        results = await router.search(analysis, top_k=10, candidate_k=30, use_rerank=False,
                                      use_context=enabled, candidate_ids_out=candidates,
                                      context_hits_out=hits, timer=timer, recorder=explain)
    if analysis.model_dump(mode="json") != before_analysis:
        raise RuntimeError("SearchRouter mutated the shared OFF/ON analysis")
    top = [str(r.id) for r in results]
    if len(candidates) > 30 or len(top) > 10 or top != candidates[:len(top)] or len(set(candidates)) != len(candidates):
        raise RuntimeError("candidate/top-10 contract changed")
    evidence = []
    for hit in hits:
        if hit.song_id in top:
            fact = context_evidence_for_result(hit, query_clues=analysis.context_clues)
            if fact:
                evidence.append(dict(song_id=hit.song_id, record_id=fact.record_id,
                                     fact_text=fact.fact_text, source_url=fact.source_url))
    spans = timer.to_dict()["spans"]
    # Only plain data survives checkpoint serialization; no stale hit objects.
    contributions = {}
    for sid in candidates:
        entry = explain.record.get(sid)
        contributions[sid] = [dict(path=p.path, rank=p.rank, delta=p.delta)
                              for p in entry.live_paths()] if entry else []
    return dict(candidates=candidates, top=top, tracks=results, context_hits=hits,
                evidence=evidence, elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                spans=spans, contributions=contributions)


async def evaluate_question(row: Query, analysis: Any, router: Any, indexed: frozenset[str],
                            cases: dict, *, required_path: str = "") -> dict:
    # Alternate which arm runs first; search times remain exploratory.
    order = (False, True) if int(hashlib.sha256(row.query_id.encode()).hexdigest()[:2], 16) % 2 else (True, False)
    arms = {enabled: await _arm(router, analysis, enabled) for enabled in order}
    off, on = arms[False], arms[True]
    value = old._item(row, SETTING, analysis, off, on, off, indexed, cases)
    labels = {cases.get(row.query_id, {}).get("evidence_record_id"),
              *cases.get(row.query_id, {}).get("additional_evidence_record_ids", [])}
    for fact in value["displayed_evidence"]:
        fact["label_match"] = fact["song_id"] in row.relevant_ids and fact["record_id"] in labels
    value.update(query=row.query, required_path=required_path, analysis=analysis.model_dump(mode="json"),
                 arm_order=["on" if x else "off" for x in order], rerank_executed=False)
    for name, arm in (("off", off), ("on", on)):
        value[f"{name}_target_paths"] = {sid: arm["contributions"].get(sid, []) for sid in sorted(row.relevant_ids)}
        value[f"{name}_path_errors"] = [dict(name=s["name"], error=s["error"])
                                        for s in arm["spans"] if s.get("error")]
        value[f"{name}_path_spans"] = arm["spans"]
        for path in ("image", "audio", "context"):
            value[f"{name}_{path}_executed"] = any(s["name"] == "path." + path for s in arm["spans"])
        check = inspect_path(arm["spans"], required_path) if required_path else None
        value[f"{name}_required_path_verified"] = bool(check and check.verified)
        value["rerank_executed"] |= any(s["name"] == "search.rerank" for s in arm["spans"])
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    structured = {"displayed_evidence", "analysis", "off_path_spans", "on_path_spans", "arm_order",
                  "off_path_errors", "on_path_errors", "off_target_paths", "on_target_paths"}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k not in structured], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print(report: dict) -> None:
    print(f"\n[{report['phase']}] status={report['status']}; questions={report['evaluated']}")
    for group in ("context_positive", "context_control", "regression_all", "regression_image", "regression_mood_sound"):
        info = report["groups"].get(group)
        if not info or not info["queries"]:
            continue
        c, h, m = (info["metrics"][key] for key in ("candidate_hit30", "hit10", "mrr10"))
        print(f"  {group}: n={info['queries']} C@30 {c['off']:.3f}->{c['on']:.3f} "
              f"Hit@10 {h['off']:.3f}->{h['on']:.3f} MRR@10 {m['off']:.3f}->{m['on']:.3f} "
              f"gains={h['gained']} losses={h['lost']}")
    print("  Actual path coverage:", json.dumps(report["path_coverage"]))
    print("  Violations:", report["violations"])
    print("  Citation review items:", report["evidence"]["unreviewed_count"])
    print("  Exploratory Hit@10 signal:", report["exploratory_hit10_signal"])


async def measure_phase(phase: str, root: Path, registration: dict, queries: dict, cases: dict) -> dict:
    from src.backend.api.dependencies import close_vector_client

    selected = {qid: q for qid, q in queries.items() if q.split == phase}
    try:
        corpus, router, indexed = old._corpus(list(selected.values()))
        corpus.update(_modality_identity(corpus, list(selected.values()), cases))
        corpus.update(text_bm25_identity(router))
        lock_corpus(root, corpus)
        _configure(router)
        analyses, cache_sha = fresh_analyses(selected, root / f"{phase}_analysis_cache.json", registration, phase)
        stamp = dict(registration=registration, corpus=corpus, analysis_cache_sha256=cache_sha, phase=phase)
        checkpoint = root / (phase + "_checkpoint.json")
        state = old._read(checkpoint) if checkpoint.exists() else dict(stamp=stamp, rows={})
        if state["stamp"] != stamp or set(state["rows"]) - set(selected):
            raise ValueError("checkpoint code/corpus/analysis/input changed")
        for n, (qid, row) in enumerate(selected.items(), 1):
            if qid in state["rows"]:
                continue
            required = cases.get(qid, {}).get("required_path", "")
            if phase != "independent" and row.group == "regression_image":
                required = "image"
            value = await evaluate_question(row, analyses[qid], router, indexed, cases, required_path=required)
            state["rows"][qid] = value
            old._write(checkpoint, state)
            print(f"[{phase} {n}/{len(selected)}] {qid} C@30 {value['off_candidate_rank']}->{value['on_candidate_rank']} "
                  f"Top10 {value['off_top10_rank']}->{value['on_top10_rank']} "
                  f"image={value['off_image_executed']}/{value['on_image_executed']}", flush=True)
        old._assert_unchanged(list(selected.values()), corpus)
        _assert_registered_files(registration)
        if text_bm25_identity(router) != {k: v for k, v in corpus.items() if k.startswith("text_bm25_")}:
            raise RuntimeError("Text BM25 parameters changed during measurement")
        if _modality_identity(corpus, list(selected.values()), cases) != {
                k: v for k, v in corpus.items() if k.startswith(("image_", "audio_")) or k == "local_collection_sha256"}:
            raise RuntimeError("modality collection content changed during evaluation")
        rows = [state["rows"][qid] for qid in selected]
        report = fixed_report(rows, phase=phase, expected=registration["phase_counts"][phase])
        report.update(corpus=corpus, registration=registration, analysis_cache_sha256=cache_sha,
                      question_results=[{k: v for k, v in row.items() if k not in
                                         {"off_path_spans", "on_path_spans"}} for row in rows])
        old._write(root / (phase + "_report.json"), report)
        _write_csv(root / (phase + "_detail.csv"), rows)
        _print(report)
        return report
    finally:
        close_vector_client()


async def measure_variation(root: Path, registration: dict, queries: dict, cases: dict, main: dict) -> dict:
    """One additional fresh analysis for 10 predeclared questions; no best-of."""
    from src.backend.api.dependencies import close_vector_client

    selected = {}
    groups = ("context_context", "context_mixed", "regression_image", "regression_mood_sound", "regression_lyrics")
    originals = {}
    for group in groups:
        candidates = [q for q in queries.values() if q.split == "independent" and q.group == group][:2]
        for q in candidates:
            key = "repeat_" + q.query_id
            selected[key] = Query(key, "repeat", q.source, q.group, q.segment, q.query, q.relevant_ids)
            originals[key] = q.query_id
    repeat_cases = {key: cases[value] for key, value in originals.items()}
    try:
        corpus, router, indexed = old._corpus(list(selected.values()))
        corpus.update(_modality_identity(corpus, list(selected.values()), repeat_cases))
        corpus.update(text_bm25_identity(router))
        _configure(router)
        if corpus != main["corpus"]:
            raise ValueError("corpus changed before repeated analysis check")
        lock_corpus(root, corpus)
        analyses, cache_sha = fresh_analyses(selected, root / "repeat_analysis_cache.json", registration, "repeat")
        stamp = dict(registration=registration, corpus=corpus, cache_sha256=cache_sha)
        checkpoint = root / "repeat_checkpoint.json"
        state = old._read(checkpoint) if checkpoint.exists() else dict(stamp=stamp, rows={})
        if state["stamp"] != stamp or set(state["rows"]) - set(selected):
            raise ValueError("repeated analysis checkpoint changed")
        main_rows = {r["query_id"]: r for r in main["question_results"]}
        for key, q in selected.items():
            if key in state["rows"]:
                continue
            original = originals[key]
            required = cases[original].get("required_path", "")
            value = await evaluate_question(q, analyses[key], router, indexed,
                                            {key: cases[original]}, required_path=required)
            before = main_rows[original]
            state["rows"][key] = dict(original_query_id=original, analysis_changed=value["analysis"] != before["analysis"],
                                      on_order_changed=value["on_candidate_ids"] != before["on_candidate_ids"],
                                      main_on_top10_rank=before["on_top10_rank"],
                                      repeat_on_top10_rank=value["on_top10_rank"], result=value)
            old._write(checkpoint, state)
            print(f"[repeat] {original}: Top10 {before['on_top10_rank']}->{value['on_top10_rank']}", flush=True)
        old._assert_unchanged(list(selected.values()), corpus)
        _assert_registered_files(registration)
        if text_bm25_identity(router) != {k: v for k, v in corpus.items() if k.startswith("text_bm25_")}:
            raise RuntimeError("Text BM25 parameters changed during repeated analysis check")
        if _modality_identity(corpus, list(selected.values()), repeat_cases) != {
                k: v for k, v in corpus.items() if k.startswith(("image_", "audio_")) or k == "local_collection_sha256"}:
            raise RuntimeError("modality collection content changed during repeated analysis check")
        rows = list(state["rows"].values())
        guards = []
        for r in rows:
            value = r["result"]
            if value["group"] in {"context_context", "context_mixed"} and not value["context_clue_count"]:
                guards.append(r["original_query_id"] + ": repeat_missing_context_clue")
            if value["source"] == "regression" and value["context_clue_count"]:
                guards.append(r["original_query_id"] + ": repeat_unexpected_context_clue")
            if value["required_path"] and not (value["off_required_path_verified"] and value["on_required_path_verified"]):
                guards.append(r["original_query_id"] + ": repeat_required_path_not_verified")
            if value["off_path_errors"] or value["on_path_errors"] or value["rerank_executed"]:
                guards.append(r["original_query_id"] + ": repeat_retrieval_or_rerank_error")
            if value["on_hit10"] < value["off_hit10"]:
                guards.append(r["original_query_id"] + ": repeat_hit10_loss")
            if value["source"] == "regression":
                for metric in ("candidate_hit30", "hit1", "hit5", "mrr10"):
                    if value[f"on_{metric}"] < value[f"off_{metric}"]:
                        guards.append(r["original_query_id"] + ": repeat_protected_" + metric + "_loss")
        citations = [dict(query_id=r["original_query_id"], **e) for r in rows for e in r["result"]["displayed_evidence"]]
        unreviewed = [e for e in citations if e["label_match"] is not True]
        report = dict(question_count=len(rows), extra_fresh_analyses=len(rows), violations=guards,
                      analysis_changed_count=sum(r["analysis_changed"] for r in rows),
                      order_changed_count=sum(r["on_order_changed"] for r in rows), rows=rows,
                      evidence=dict(displayed_count=len(citations), unreviewed_count=len(unreviewed), review_items=unreviewed),
                      policy="Each question analyzed twice total; repeated trials are excluded from n=60 and statistical tests.")
        old._write(root / "repeat_report.json", report)
        return report
    finally:
        close_vector_client()


def _write_report(root: Path, reports: dict, registration: dict, *, status: str, variation: dict | None = None) -> None:
    result = dict(schema="context_fixed_validation_v1", status=status, registration=registration,
                  completed_phases=list(reports), reports=reports,
                  analyzer_repeat_check=variation,
                  next_action="Review displayed citations and analyzer variation before a deployment claim.")
    old._write(root / "validation_report.json", result)
    lines = ["# Context 고정 설정 재검증", "", f"판정: **{status}**", "",
             "가중치 0.5 / Dense 사실 100 / Sparse 프로필 100 / 후보 30 / 화면 10 / 작품 단서 배수 2.0 / 리랭킹 OFF.",
             "새 QueryAnalyzer 분석을 생성하고, 각 질의의 OFF/ON에는 동일한 분석을 전달했다.", "",
             "| 분할·군 | 문항 | Candidate@30 OFF→ON | Hit@10 OFF→ON | MRR@10 OFF→ON |",
             "|---|---:|---:|---:|---:|"]
    notes = []
    for phase, report in reports.items():
        for name in ("context_positive", "context_control", "regression_all", "regression_image", "regression_mood_sound"):
            group = report["groups"].get(name)
            if group and group["queries"]:
                cells = [f"{group['metrics'][key]['off']:.3f}→{group['metrics'][key]['on']:.3f}"
                         for key in ("candidate_hit30", "hit10", "mrr10")]
                lines.append(f"| {phase} {name} | {group['queries']} | " + " | ".join(cells) + " |")
        notes += ["", f"{phase} 회귀·경로 문제: {report['violations'] or '없음'}",
                  f"{phase} 표시 근거 {report['evidence']['displayed_count']}건 중 라벨 일치 "
                  f"{report['evidence']['label_matched_count']}건, 추가 검토 {report['evidence']['unreviewed_count']}건.",
                  f"{phase} 개선 신호(탐색적): {report['exploratory_hit10_signal']}; "
                  f"Hit@10 paired p={report['paired_context_tests']['hit10']['exact_two_sided_p']:.4f}.",
                  "실제 이미지·오디오 실행은 가중치와 별도로 embed/query 구간 및 오류를 검사했다."]
    lines.extend(notes)
    if variation:
        lines += ["", f"별도 재분석 {variation['question_count']}문항: 분석 변경 {variation['analysis_changed_count']}건, "
                  f"후보 순서 변경 {variation['order_changed_count']}건. 문제: {variation['violations'] or '없음'}.",
                  "재분석 결과는 독립 질의 60문항의 분모에 추가하지 않았다."]
        lines += [f"재분석 표시 근거 중 추가 검토 {variation['evidence']['unreviewed_count']}건."]
    lines += ["", "dev/test와 추가 60문항 모두 이전 실행에서 관측한 질의다. 이번 실행은 새 분석을 사용하는 회귀 재검증이며 새 블라인드 평가가 아니다.",
              "추가 60문항은 코퍼스에 근거해 작성한 테스트이며 실제 사용자 질문의 무작위 표본은 아니다. independent는 기존 파일 형식을 유지한 분할명이다.",
              "라벨 외 근거는 자동으로 오답이라 단정하지 않고 검토 대상으로 남긴다.",
              "Top-10은 제품 기본값으로 고정했다. 화면 표시 수의 최적값을 입증한 것은 아니다.",
              "원문의 진위를 검증한 수치가 아니며, 지연은 모델 초기 적재가 포함될 수 있는 검색 구간 측정이다.", ""]
    (root / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    if os.getenv("SEARCH_REFERENCE_YEAR") != "2026":
        raise ValueError("SEARCH_REFERENCE_YEAR=2026 is required")
    registration, queries, cases, _ = _registration(args.output_dir, args.new_csv, args.new_audit, args.disclosed_csv)
    if args.phase == "register":
        print("Registered fixed settings and reanalysis of 166 previously observed queries; no new blind claim.")
        return 0
    reports = {}
    for phase in ("dev", "test", "independent"):
        p = args.output_dir / (phase + "_report.json")
        if p.exists():
            reports[phase] = old._read(p)
            if reports[phase].get("registration") != registration:
                raise ValueError("saved phase report belongs to another registered run")
    phases = ("dev", "test", "independent") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "independent":
            if any(p not in reports for p in ("dev", "test")):
                _write_report(args.output_dir, reports, registration, status="blocked_before_independent")
                print("Complete dev/test reports are required before the additional disclosed 60-question reanalysis.")
                return 4
            if any(reports[p]["violations"] for p in ("dev", "test")):
                print("Existing regression gate failed. Continue the requested 60-question diagnostic with fixed settings; the final gate remains blocked.", flush=True)
        reports[phase] = await measure_phase(phase, args.output_dir, registration, queries, cases)
        if any(value["corpus"] != reports[phase]["corpus"] for value in reports.values()):
            raise ValueError("corpus or modality content changed between phases")
        # Always report both existing splits, even if dev exposes a loss.
        _write_report(args.output_dir, reports, registration, status="in_progress")
    variation = None
    if "independent" in reports:
        variation = await measure_variation(args.output_dir, registration, queries, cases, reports["independent"])
    status = ("blocked" if any(r["violations"] for r in reports.values()) or variation and variation["violations"] else
              "evidence_review_required" if any(r["evidence"]["unreviewed_count"] for r in reports.values())
              or variation and variation["evidence"]["unreviewed_count"] else
              "passed_observed_tests" if set(reports) == {"dev", "test", "independent"} else "partial")
    _write_report(args.output_dir, reports, registration, status=status, variation=variation)
    print(f"\nFinal status: {status}. Send {args.output_dir / 'validation_report.json'}")
    return 4 if status == "blocked" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("register", "dev", "test", "independent", "all"), default="all")
    parser.add_argument("--new-csv", type=Path, default=NEW_CSV)
    parser.add_argument("--new-audit", type=Path, default=NEW_AUDIT)
    parser.add_argument("--disclosed-csv", type=Path, default=OLD_DISCLOSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        # Keep completed rows/caches for a same-command retry, and export a
        # small diagnostic even when an API fallback stops the run early.
        old._write(args.output_dir / "failure_report.json", {
            "status": "interrupted", "phase": args.phase,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "resume_policy": "Retry the same command without editing settings, code, labels or corpus.",
        })
        print(f"Evaluation interrupted: {type(exc).__name__}: {exc}", flush=True)
        print(f"Diagnostic: {args.output_dir / 'failure_report.json'}", flush=True)
        return 2
    (args.output_dir / "failure_report.json").unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
