"""Full-corpus Context OFF/ON evaluation, dev selection, and one held-out test.

Run through run_context_step10.ps1 with the backend stopped for embedded Qdrant.
All queries, retrieved titles/facts, labels, and outputs remain under ignored
artifacts/; none of the evaluation data belongs in the public repository.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.namuwiki import build_eval_csv
from experiments.namuwiki.context_step10_core import (
    EXPECTED, Setting, blind_verdict, compare, heldout_verdict, select_dev,
)
from experiments.namuwiki.evaluate_context_step8 import (
    CONTEXT_CSV, CONTEXT_SOURCE, REGRESSION_CSV, Query, evaluate_pair,
    load_queries, metrics,
)


ROOT = Path("artifacts/context_eval_step10_v2")
# Fixed before examining the new final Context generation. Widths count
# retrieved facts and song profiles, respectively. The outer Candidate@30 and
# Top-10 remain separately reported even for a 50-candidate trial. The
# separately reported display-depth table is a prefix diagnostic, not a
# claim that the UI's best number of results has been determined.
GRID = tuple(Setting(weight, fact_k, sparse_k, candidate_k)
             for weight, (fact_k, sparse_k), candidate_k in itertools.product(
                 (0.25, 0.5, 1.0), ((100, 100), (160, 100), (100, 160)), (30, 50))) + tuple(
    Setting(weight, 100, 100, 30) for weight in (0.6, 0.7, 0.8, 0.9)
) + tuple(Setting(0.5, 100, 100, 30, named_media_multiplier=mult)
          for mult in (1.5, 2.0))
SOURCE_FILES = (CONTEXT_CSV, CONTEXT_SOURCE, REGRESSION_CSV)
CODE_FILES = (
    Path("experiments/namuwiki/evaluate_context_step10.py"),
    Path("experiments/namuwiki/context_step10_core.py"),
    Path("experiments/namuwiki/evaluate_context_step8.py"),
    Path("src/retrieval/search_router.py"),
    Path("src/retrieval/context_qdrant_search.py"),
    Path("src/retrieval/context_ranking.py"),
    Path("src/retrieval/context_evidence.py"),
    Path("src/retrieval/context_query.py"),
    Path("src/retrieval/query_analyzer.py"),
    Path("src/backend/api/dependencies.py"),
    Path("src/backend/api/routes/search.py"),
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _cases() -> dict[str, dict]:
    raw = _read(CONTEXT_SOURCE)
    return {case["query_id"]: case for case in raw["queries"]}


def _preflight_labels() -> tuple[dict[str, Query], dict[str, str], dict[str, dict]]:
    # Verifies all 30 evidence IDs/categories against today's published
    # artifacts, and refuses a stale CSV after recrawling.
    expected, _, _ = build_eval_csv.build_rows()
    if CONTEXT_CSV.read_bytes().decode("utf-8") != build_eval_csv.render_csv(expected):
        raise ValueError("Context CSV differs from reviewed source; run build_eval_csv.py --check")
    queries, _ = load_queries(CONTEXT_CSV, REGRESSION_CSV, "all", context_source=CONTEXT_SOURCE)
    for split, (context_count, regression_count) in EXPECTED.items():
        actual = [q for q in queries if q.split == split]
        if (sum(q.source == "context" for q in actual),
                sum(q.source == "regression" for q in actual)) != (context_count, regression_count):
            raise ValueError(f"{split}: query inventory changed; review the split before tuning")
    return {row.query_id: row for row in queries}, {str(p): _hash(p) for p in SOURCE_FILES}, _cases()


def _blind_queries(path: Path, existing: dict[str, Query]) -> tuple[dict[str, Query], dict[str, dict]]:
    """Load independently authored labels, with no reused query/song targets."""
    if not path.is_file():
        raise FileNotFoundError(f"new blind labels not found: {path}")
    fields = {"query_id", "query", "relevant_ids", "group", "clue_category",
              "evidence_record_ids", "review_note"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError(f"blind CSV needs columns: {sorted(fields)}")
        raw = list(reader)
    old_ids = {song_id for row in existing.values() for song_id in row.relevant_ids}
    seen_ids, seen_songs = set(existing), set(old_ids)
    queries, cases = {}, {}
    context_groups = {"context_context", "context_mixed", "context_control"}
    regression_groups = {"regression_lyrics", "regression_mood_sound",
                         "regression_image", "regression_other"}
    for item in raw:
        query_id = item["query_id"].strip()
        query = item["query"].strip()
        song_id = item["relevant_ids"].strip()
        group = item["group"].strip()
        category = item["clue_category"].strip()
        evidence_ids = [part.strip() for part in item["evidence_record_ids"].split("|") if part.strip()]
        if (not query_id.startswith("blind") or query_id in seen_ids or not query
                or not song_id.isdigit() or song_id in seen_songs
                or group not in context_groups | regression_groups
                or not item["review_note"].strip() or len(evidence_ids) != len(set(evidence_ids))):
            raise ValueError(f"blind row invalid, overlaps an old label, or lacks review: {query_id!r}")
        path_artifact = Path("artifacts/context/songs") / f"{song_id}.json"
        if not path_artifact.is_file():
            raise ValueError(f"blind label has no Context artifact: {query_id}")
        artifact = _read(path_artifact)
        if str(artifact["song"]["song_id"]) != song_id:
            raise ValueError(f"blind song/artifact ID mismatch: {query_id}")
        if group == "context_control":
            if (evidence_ids or category != "no_context" or artifact["status"] not in
                    {"no_trivia", "not_found"} or artifact["retrieval"]["records"]):
                raise ValueError(f"blind control has a Context fact: {query_id}")
        elif group in context_groups:
            records = {record["record_id"]: record for record in artifact["retrieval"]["records"]}
            if (not evidence_ids or not category or artifact["status"] != "ok"
                    or not artifact["source"].get("url")
                    or any(record_id not in records or
                           records[record_id]["category"] != category or
                           not records[record_id]["evidence_text"].strip()
                           for record_id in evidence_ids)):
                raise ValueError(f"blind Context fact labels require reviewed record IDs: {query_id}")
        elif evidence_ids or category != "no_context":
            raise ValueError(f"blind regression must use no_context category: {query_id}")
        source = "context" if group in context_groups else "regression"
        segment = group.removeprefix("context_") if source == "context" else ""
        queries[query_id] = Query(query_id, "blind", source, group, segment,
                                  query, frozenset({song_id}))
        cases[query_id] = {"clue_category": category,
                           "evidence_record_id": evidence_ids[0] if evidence_ids else None,
                           "additional_evidence_record_ids": evidence_ids[1:]}
        seen_ids.add(query_id)
        seen_songs.add(song_id)
    if len(queries) < 30:
        raise ValueError("blind split requires at least 30 independent reviewed queries")
    return queries, cases


def _analyses(queries: dict[str, Query], path: Path, *, populate: bool) -> tuple[dict, str]:
    from src.backend.api.dependencies import get_query_analyzer
    from src.retrieval.analysis_cache import (
        analyzer_fingerprint, fingerprint_sha, load_cache, looks_like_fallback, new_cache,
    )

    cache = load_cache(path) if path.is_file() else new_cache(source="context_v01+reranking_v05", split="all")
    if cache.fingerprint_drift():
        raise ValueError(f"analysis cache fingerprint drift: {cache.fingerprint_drift()}")
    expected_sha = fingerprint_sha(analyzer_fingerprint())
    analyzer = None
    analyses = {}
    for index, row in enumerate(queries.values(), start=1):
        entry = cache.entries.get(row.query_id)
        if entry is None:
            if not populate:
                raise ValueError(f"analysis cache missing {row.query_id}; run --phase prepare")
            analyzer = analyzer or get_query_analyzer()
            analysis = analyzer.analyze(row.query)
            if looks_like_fallback(analysis):
                raise ValueError(f"{row.query_id}: analyzer fallback; retry prepare after service recovery")
            cache.put(row.query_id, row.query, analysis)
            path.parent.mkdir(parents=True, exist_ok=True)
            cache.save(path)
            entry = cache.entries[row.query_id]
            print(f"[analysis {index}/{len(queries)}] {row.query_id} cached", flush=True)
        if entry.get("fallback") or entry.get("analyzer_sha") != expected_sha:
            raise ValueError(f"{row.query_id}: fallback or mixed analyzer provenance; rebuild a clean cache")
        analysis = cache.get(row.query_id, row.query)
        if looks_like_fallback(analysis) or analysis.original_query.strip() != row.query:
            raise ValueError(f"{row.query_id}: invalid cached analysis")
        analyses[row.query_id] = analysis
    return analyses, _hash(path)


def _corpus(queries: list[Query]) -> tuple[dict, Any, frozenset[str]]:
    from experiments.namuwiki.evaluate_context_step8 import _live_preflight
    from experiments.namuwiki.verify_context_step9 import verify
    from src.vector_db.settings import NAMESPACE

    checked = verify(context_dir=Path("artifacts/context"),
                     state_manifest=Path("artifacts/vector_db/context_qdrant") / NAMESPACE / "manifest.json")
    final = _read(Path("artifacts/context/step9_final_report.json"))
    if (final.get("verified", {}).get("status") != "verified"
            or final["verified"].get("context_build_id") != checked["context_build_id"]):
        raise ValueError("active Context build differs from the Step 9 final report")
    indexed, info, router = _live_preflight(queries)
    if (not checked["source_scope_complete"] or checked["source_song_count"] < 3016
            or info["text_points"] != checked["source_song_count"]
            or info["context_build_id"] != checked["context_build_id"]
            or info["indexed_label_count"] != info["label_count"]):
        raise ValueError("Context/Text Qdrant scope, build ID, or indexed labels differ")
    # Indexed-label counts depend on the split (dev/test/blind). They are
    # validated here but must not enter the cross-split corpus identity.
    identity = {key: value for key, value in info.items()
                if key not in {"indexed_label_count", "label_count"}}
    return {**identity, **checked,
            "context_manifest_sha256": _hash(Path("artifacts/context/manifest.json")),
            "state_manifest_sha256": _hash(Path("artifacts/vector_db/context_qdrant") / NAMESPACE / "manifest.json")}, router, indexed


def _stamp(source_hashes: dict, cache_hash: str, corpus: dict, output: Path) -> dict:
    registration = output / "blind_registration.json"
    return dict(schema="context_step10_v1", sources=source_hashes,
                analysis_cache_sha256=cache_hash, corpus=corpus,
                code_sha256={str(p): _hash(p) for p in CODE_FILES},
                grid=[item.as_dict() for item in GRID],
                blind_registration_sha256=_hash(registration) if registration.is_file() else None,
                rerank=False, reference_candidate_k=30, top_k=10)


def _assert_unchanged(rows: list[Query], corpus: dict) -> None:
    from experiments.namuwiki.evaluate_context_step8 import _live_preflight
    from src.vector_db.settings import NAMESPACE

    _, after, _ = _live_preflight(rows)
    for key in ("text_collection", "text_points", "context_build_id",
                "context_dense_collection", "context_sparse_collection"):
        if after[key] != corpus[key]:
            raise RuntimeError(f"Qdrant changed during paired evaluation: {key}")
    for path, sha in ((Path("artifacts/context/manifest.json"), corpus["context_manifest_sha256"]),
                      (Path("artifacts/vector_db/context_qdrant") / NAMESPACE / "manifest.json",
                       corpus["state_manifest_sha256"])):
        if _hash(path) != sha:
            raise RuntimeError(f"Context publication changed during paired evaluation: {path}")


def _state(path: Path, stamp: dict, split: str) -> dict:
    state = _read(path) if path.is_file() else dict(stamp=stamp, split=split, baselines={}, runs={})
    if state["stamp"] != stamp or state["split"] != split:
        raise ValueError(f"{path} targets another corpus/input/code; archive that run before retrying")
    return state


async def _search(router: Any, analysis: Any, *, setting: Setting | None,
                  candidate_k: int) -> dict:
    from src.retrieval.context_evidence import context_evidence_for_result

    candidate_ids: list[str] = []
    context_hits: list[Any] = []
    started = time.perf_counter()
    results = await router.search(analysis, top_k=10, candidate_k=candidate_k,
                                  use_rerank=False, use_context=setting is not None,
                                  candidate_ids_out=candidate_ids,
                                  context_hits_out=context_hits)
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    top = [str(item.id) for item in results]
    if (len(candidate_ids) > candidate_k or len(top) > 10
            or top != candidate_ids[:len(top)] or len(set(candidate_ids)) != len(candidate_ids)):
        raise RuntimeError("candidate/top-k ordering or uniqueness contract changed")
    evidence = []
    for hit in context_hits:
        if hit.song_id not in top:
            continue
        found = context_evidence_for_result(hit, query_clues=analysis.context_clues)
        if found:
            evidence.append(dict(song_id=hit.song_id, record_id=found.record_id,
                                 fact_text=found.fact_text, source_url=found.source_url))
    return dict(candidates=candidate_ids, top=top, tracks=results, context_hits=context_hits,
                evidence=evidence, elapsed_ms=elapsed)


def _saved_arm(value: dict) -> dict:
    return dict(candidates=value["candidates"], top=value["top"],
                tracks=[], context_hits=[], elapsed_ms=value["elapsed_ms"])


async def _baseline(row: Query, analysis: Any, width: int, router: Any,
                    state: dict, path: Path) -> dict:
    baselines = state["baselines"].setdefault(str(width), {})
    if row.query_id not in baselines:
        result = await _search(router, analysis, setting=None, candidate_k=width)
        baselines[row.query_id] = {key: result[key] for key in ("candidates", "top", "elapsed_ms")}
        _write(path, state)
    return _saved_arm(baselines[row.query_id])


def _item(row: Query, setting: Setting, analysis: Any, off: dict, on: dict,
          reference: dict, indexed: frozenset[str], cases: dict[str, dict]) -> dict:
    result = evaluate_pair(row, off, on, indexed, analysis)
    eligible = row.relevant_ids & indexed
    ref = metrics(reference["candidates"], reference["top"], eligible)
    if not analysis.has_context_clue and (off["candidates"] != on["candidates"]
                                           or off["top"] != on["top"]):
        raise RuntimeError(f"{row.query_id}: Context disabled by analysis but OFF/ON differ")
    case = cases.get(row.query_id, {})
    evidence = on["evidence"]
    target_evidence = next((e for e in evidence if e["song_id"] in eligible), None)
    label_ids = {case.get("evidence_record_id"), *case.get("additional_evidence_record_ids", [])}
    result.update(
        setting=setting.key, category=case.get("clue_category", "unlabeled"),
        off_elapsed_ms=off.get("elapsed_ms"), on_elapsed_ms=on["elapsed_ms"],
        ref_candidate_ids="|".join(reference["candidates"]),
        ref_top10_ids="|".join(reference["top"]),
        ref_candidate_rank=next((i for i, song_id in enumerate(reference["candidates"], 1)
                                 if song_id in eligible), None),
        ref_top10_rank=next((i for i, song_id in enumerate(reference["top"], 1)
                             if song_id in eligible), None),
        candidate_width=setting.candidate_k,
        off_candidate_hit_width=float(bool(eligible & set(off["candidates"]))),
        on_candidate_hit_width=float(bool(eligible & set(on["candidates"]))),
        target_evidence_record_id=target_evidence["record_id"] if target_evidence else "",
        target_evidence_labeled=(target_evidence["record_id"] in label_ids)
        if target_evidence else None,
        # These are local, ignored artifacts. They are for human fact review,
        # not an automatic precision claim about other retrieved songs.
        displayed_evidence=evidence,
    )
    for name, value in ref.items():
        result[f"ref_{name}"] = value
    return result


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        keys = [key for key in rows[0] if key != "displayed_evidence"]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def _phase(split: str, output: Path) -> int:
    from src.backend.api.dependencies import close_vector_client

    all_queries, source_hashes, cases = _preflight_labels()
    cache_path = Path("artifacts/context_eval_step8/analysis_cache.json")
    analyses, cache_hash = _analyses(all_queries, cache_path, populate=False)
    selected = [row for row in all_queries.values() if row.split == split]
    corpus, router, indexed = _corpus(selected)
    stamp = _stamp(source_hashes, cache_hash, corpus, output)
    path = output / f"{split}_checkpoint.json"
    state = _state(path, stamp, split)
    settings = GRID
    if split == "test":
        lock = _read(output / "dev_selection.json")
        if lock["stamp"] != stamp or lock["decision"]["status"] != "locked_for_test":
            raise ValueError("dev selection not locked against this corpus and query analysis")
        settings = tuple(item for item in GRID if item.key == lock["decision"]["selected"])
        if len(settings) != 1:
            raise ValueError("locked setting absent from predefined grid")
    try:
        for setting_idx, setting in enumerate(settings, 1):
            result_rows = state["runs"].setdefault(setting.key, {})
            router._context_weight = setting.weight
            router._context_named_media_multiplier = setting.named_media_multiplier
            router._context_fact_k = setting.fact_k
            router._context_sparse_k = setting.sparse_k
            for query_idx, row in enumerate(selected, 1):
                if row.query_id in result_rows:
                    continue
                analysis = analyses[row.query_id]
                reference = await _baseline(row, analysis, 30, router, state, path)
                matched_off = (reference if setting.candidate_k == 30 else
                               await _baseline(row, analysis, setting.candidate_k, router, state, path))
                on = await _search(router, analysis, setting=setting,
                                   candidate_k=setting.candidate_k)
                result_rows[row.query_id] = _item(row, setting, analysis, matched_off, on,
                                                  reference, indexed, cases)
                _write(path, state)
                r = result_rows[row.query_id]
                print(f"[{split} {setting_idx}/{len(settings)} {query_idx}/{len(selected)}] "
                      f"{setting.key} {row.query_id}: C@30 "
                      f"{r['ref_candidate_rank']}→{r['on_candidate_rank']} "
                      f"Top10 {r['ref_top10_rank']}→{r['on_top10_rank']}", flush=True)
            ordered = [result_rows[row.query_id] for row in selected]
            _csv(output / split / f"{setting.key}.csv", ordered)
            report = compare(ordered, setting)
            print(f"{setting.key}: Context Hit@10 "
                  f"{report['reference']['context_all']['metrics']['hit10']['ref']:.3f}→"
                  f"{report['reference']['context_all']['metrics']['hit10']['on']:.3f}; "
                  f"protected losses={len(report['violations'])}", flush=True)
        _assert_unchanged(selected, corpus)
        if split == "dev":
            decision = select_dev({item.key: (item, [state["runs"][item.key][row.query_id]
                                                    for row in selected]) for item in settings})
            _write(output / "dev_selection.json", dict(stamp=stamp, decision=decision))
            print(f"DEV selection: {decision['status']} {decision.get('selected')}", flush=True)
            return 0 if decision["status"] == "locked_for_test" else 3
        setting = settings[0]
        verdict = heldout_verdict([state["runs"][setting.key][row.query_id]
                                    for row in selected], setting)
        result = dict(stamp=stamp, locked_setting=setting.key, verdict=verdict)
        _write(output / "final_report.json", result)
        _write_report(output / "final_report.md", result, output / "dev_selection.json")
        print(f"TEST verdict: {verdict['status']}; violations={verdict['violations']}", flush=True)
        print(f"Private report: {output / 'final_report.md'}", flush=True)
        return 0 if verdict["status"] == "provisional" else 4
    finally:
        close_vector_client()


async def _blind_phase(output: Path, blind_csv: Path) -> int:
    from src.backend.api.dependencies import close_vector_client

    existing, sources, _ = _preflight_labels()
    registration = _read(output / "blind_registration.json")
    if (registration.get("blind_csv_sha256") != _hash(blind_csv)
            or registration.get("sources") != sources):
        raise ValueError("blind CSV was not registered before dev selection or has changed")
    queries, cases = _blind_queries(blind_csv, existing)
    all_analyses, old_cache_hash = _analyses(
        existing, Path("artifacts/context_eval_step8/analysis_cache.json"), populate=False
    )
    del all_analyses  # old analysis is verified only to protect the dev lock
    analyses, blind_cache_hash = _analyses(
        queries, output / "blind_analysis_cache.json", populate=True
    )
    selected = list(queries.values())
    corpus, router, indexed = _corpus(selected)
    stamp = _stamp(sources, old_cache_hash, corpus, output)
    lock = _read(output / "dev_selection.json")
    reused = _read(output / "final_report.json")
    if (lock["stamp"] != stamp or reused["stamp"] != stamp
            or lock["decision"]["selected"] != reused["locked_setting"]
            or reused["verdict"]["status"] != "provisional"):
        raise ValueError("dev/reused test gate has not passed for this exact corpus and code")
    setting = next(item for item in GRID if item.key == reused["locked_setting"])
    blind_stamp = {**stamp, "blind_csv_sha256": _hash(blind_csv),
                   "blind_analysis_cache_sha256": blind_cache_hash}
    path = output / "blind_checkpoint.json"
    state = _state(path, blind_stamp, "blind")
    result_rows = state["runs"].setdefault(setting.key, {})
    router._context_weight, router._context_fact_k, router._context_sparse_k = (
        setting.weight, setting.fact_k, setting.sparse_k
    )
    router._context_named_media_multiplier = setting.named_media_multiplier
    try:
        for index, row in enumerate(selected, 1):
            if row.query_id in result_rows:
                continue
            analysis = analyses[row.query_id]
            reference = await _baseline(row, analysis, 30, router, state, path)
            off = (reference if setting.candidate_k == 30 else
                   await _baseline(row, analysis, setting.candidate_k, router, state, path))
            on = await _search(router, analysis, setting=setting,
                               candidate_k=setting.candidate_k)
            result_rows[row.query_id] = _item(row, setting, analysis, off, on,
                                              reference, indexed, cases)
            _write(path, state)
            print(f"[blind {index}/{len(selected)}] {row.query_id} "
                  f"Top10 {result_rows[row.query_id]['ref_top10_rank']}→"
                  f"{result_rows[row.query_id]['on_top10_rank']}", flush=True)
        _assert_unchanged(selected, corpus)
        rows = [result_rows[row.query_id] for row in selected]
        _csv(output / "blind" / f"{setting.key}.csv", rows)
        verdict = blind_verdict(rows, setting)
        result = dict(stamp=blind_stamp, locked_setting=setting.key, verdict=verdict)
        _write(output / "blind_report.json", result)
        _write_blind_report(output / "blind_report.md", result)
        print(f"BLIND verdict: {verdict['status']}; losses={verdict['violations']}", flush=True)
        print(f"Private blind report: {output / 'blind_report.md'}", flush=True)
        return 0 if verdict["status"] == "recommended" else 4
    finally:
        close_vector_client()


def _write_blind_report(path: Path, result: dict) -> None:
    report = result["verdict"]
    setting = report["setting"]
    lines = ["# 새 독립 질의셋 Context 검증", "",
             f"- Final Context build: `{result['stamp']['corpus']['context_build_id']}`",
             f"- Locked dev setting: `{result['locked_setting']}`",
             f"- New blind queries: {report['reference']['all']['queries']}",
             f"- 판정: **{report['status']}**", "",
             "| 군 | 문항 | Candidate@30 OFF→ON | Hit@10 OFF→ON | 손실 ID |",
             "|---|---:|---:|---:|---|"]
    for name in ("context_all", "context_context", "context_mixed", "context_control",
                 "regression_all", "regression_lyrics", "regression_mood_sound",
                 "regression_image", "no_context_clue"):
        group = report["reference"].get(name)
        if not group:
            continue
        candidate, top = group["metrics"]["candidate_hit30"], group["metrics"]["hit10"]
        lines.append(f"| {name} | {group['queries']} | {candidate['ref']:.3f}→{candidate['on']:.3f} | "
                     f"{top['ref']:.3f}→{top['on']:.3f} | {', '.join(top['lost']) or '-'} |")
    lines += ["", "- 불합격/회귀: " + (", ".join(report["violations"]) or "없음"),
              f"- 라벨이 확인된 정답 사실: {len(report['evidence_audit']['labeled_target_evidence'])}",
              "- 새 사실을 보여주더라도 원문의 참·거짓과 다른 정답 가능성은 사람이 점검한다.", ""]
    for name, section in sorted(report["reference"].items()):
        if name.startswith("category_"):
            hit = section["metrics"]["hit10"]
            lines.append(f"- {name} ({section['queries']}문항): Hit@10 {hit['ref']:.3f}→{hit['on']:.3f}; "
                         f"손실 {hit['lost']}")
    lines.extend(_display_depth_lines("blind", report))
    if report["status"] == "recommended":
        lines += ["권고 `.env` 설정 (공개 저장소에 `.env`를 올리지 않음):", "", "```dotenv",
                  f"CONTEXT_WEIGHT={setting['weight']}", f"CONTEXT_FACT_K={setting['fact_k']}",
                  f"CONTEXT_SPARSE_K={setting['sparse_k']}",
                  f"CONTEXT_NAMED_MEDIA_MULTIPLIER={setting['named_media_multiplier']}",
                  f"SEARCH_DEFAULT_CANDIDATE_K={setting['candidate_k']}", "```", "",
              "화면 top_k=10은 기존 제품 기본값이다. 5/10/20 가상 표시 폭을 확인하고 UI 비용까지 검토한 뒤 적용한다.", ""]
    else:
        lines += ["설정 배포 보류. 이 블라인드셋으로 가중치를 재선택하지 않는다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(path: Path, result: dict, dev_path: Path) -> None:
    decision = _read(dev_path)["decision"]
    verdict = result["verdict"]
    setting = result["locked_setting"]
    lines = ["# 전체 코퍼스 Context 검색 비교", "",
             f"- Qdrant Context build: `{result['stamp']['corpus']['context_build_id']}`",
             f"- 현재 범위: {result['stamp']['corpus']['source_song_count']}곡 / "
             f"Dense {result['stamp']['corpus']['dense_record_count']}건 / "
             f"Sparse {result['stamp']['corpus']['sparse_profile_count']}곡",
             f"- 평가: dev {sum(EXPECTED['dev'])}문항으로 선택 후 기존 test {sum(EXPECTED['test'])}문항 재검증",
             "- 동일 분석 캐시·Text/Context 세대, 리랭킹 OFF, Context OFF@30 기준선",
             "- 화면 top-k=10 고정; Candidate@30과 Hit@1/5/10 별도 표시",
             f"- dev 선택: `{setting}`; test 판정: **{verdict['status']}**", "",
             "| 분할·군 | 문항 | C@30 OFF→ON | Hit@10 OFF→ON | 손실 ID |", "|---|---:|---:|---:|---|"]
    notes = []
    for label, report in (("dev", decision["candidates"][setting]), ("test", verdict)):
        for name in ("context_all", "context_context", "context_mixed", "context_control",
                     "regression_all", "regression_lyrics", "regression_mood_sound",
                     "regression_image", "no_context_clue"):
            section = report["reference"].get(name)
            if not section:
                continue
            c, h = (section["metrics"][key] for key in ("candidate_hit30", "hit10"))
            lines.append(f"| {label} {name} | {section['queries']} | "
                         f"{c['ref']:.3f}→{c['on']:.3f} | {h['ref']:.3f}→{h['on']:.3f} | "
                         f"{', '.join(h['lost']) or '-'} |")
        all_context = report["reference"]["context_all"]
        interval = all_context["metrics"]["hit10"]["bootstrap_95_delta"]
        notes.append(f"{label} Context Hit@10 paired Δ의 탐색적 95% 재표본 구간: "
                     f"[{interval[0]:+.3f}, {interval[1]:+.3f}]. "
                     f"선택된 후보 폭 {verdict['setting']['candidate_k']}의 후보 적중률 "
                     f"{all_context['candidate_hit_at_configured_width']['off']:.3f}→"
                     f"{all_context['candidate_hit_at_configured_width']['on']:.3f}.")
        for name, section in sorted(report["reference"].items()):
            if name.startswith("category_"):
                hit = section["metrics"]["hit10"]
                notes.append(f"- {label} {name} ({section['queries']}문항): Hit@10 "
                             f"{hit['ref']:.3f}→{hit['on']:.3f}; 손실 {hit['lost']}")
    lines.extend(["", "## 사실 유형과 불확실성", "", *notes])
    lines += ["", "## 화면 표시 폭 5/10/20 비교", "",
              "리랭킹 OFF인 동일 후보 순서의 앞부분만 계산했다. 20개 화면의 사용자 경험·지연은 별도로 확인해야 한다."]
    for label, report in (("dev", decision["candidates"][setting]), ("test", verdict)):
        lines.extend(_display_depth_lines(label, report))
    lines += ["", "## 결론과 적용 값", "",
              f"- test gate: {', '.join(verdict['violations']) or '통과'}", "- 전후 차이는 같은 전체 코퍼스에서 Context 경로만 켜고 끈 비교다.",
              "- 이전 단계에서 test/c703 결과를 이미 확인했으므로 이 test는 새 블라인드 데이터가 아니다.",
              "- 30개 Context 질의와 76개 기존 질의로 모든 자연어 표현의 품질을 보장할 수 없다.",
              "- 화면 Top-10은 기존 기본값으로 고정해 선택한 후보이며 최적 폭으로 검증되지 않았다.",
              "- Dense 사실 유사도·자동 근거 판정은 원문 사실의 참/거짓을 검증하지 않는다. 모든 표시 근거는 private CSV/체크포인트에서 수동 점검한다.",
              "- 손실이 있으면 test를 보며 가중치를 다시 고르지 않는다. 별도 dev 질의 및 새 독립 질의셋을 구성한다.",
              f"- 표시된 사실 {verdict['evidence_audit']['displayed_facts']}개; 정답 사실 라벨 일치 {len(verdict['evidence_audit']['labeled_target_evidence'])}문항; "
              f"라벨 외 정답 사실 수동 검토: {verdict['evidence_audit']['unlabeled_target_evidence_to_review']}", ""]
    if verdict["status"] == "provisional":
        values = verdict["setting"]
        lines += ["이 값은 기존 test 게이트를 통과한 잠정 설정이다. 새 독립 질의셋 검증 후 적용 여부를 결정한다:", "",
                  "```dotenv", f"CONTEXT_WEIGHT={values['weight']}",
                  f"CONTEXT_FACT_K={values['fact_k']}", f"CONTEXT_SPARSE_K={values['sparse_k']}",
                  f"CONTEXT_NAMED_MEDIA_MULTIPLIER={values['named_media_multiplier']}",
                  f"SEARCH_DEFAULT_CANDIDATE_K={values['candidate_k']}", "```", ""]
    else:
        lines += ["**적용 보류:** 현재 배포 설정을 이 평가 결과로 변경하지 않는다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _display_depth_lines(label: str, report: dict) -> list[str]:
    lines = ["", f"{label} 표시 깊이 (정답 포함 비율, 기준 OFF→Context ON):", "",
             "| 군 | 앞 5곡 | 앞 10곡 | 앞 20곡 |", "|---|---:|---:|---:|"]
    for group_name in ("context_all", "regression_all", "context_control"):
        if group_name not in report["display_depth_tradeoffs"]:
            continue
        group = report["display_depth_tradeoffs"][group_name]
        cells = [f"{group[f'hit{depth}']['ref']:.3f}→{group[f'hit{depth}']['on']:.3f}"
                 for depth in (5, 10, 20)]
        lines.append(f"| {group_name} | " + " | ".join(cells) + " |")
    return lines + [""]


async def run(args: argparse.Namespace) -> int:
    if os.getenv("SEARCH_REFERENCE_YEAR") != "2026":
        raise ValueError("Set SEARCH_REFERENCE_YEAR=2026 for the frozen analysis cache")
    queries, hashes, cases = _preflight_labels()
    if args.phase == "register":
        if args.blind_csv is None:
            raise ValueError("--blind-csv is required for registration")
        reviewed, _ = _blind_queries(args.blind_csv, queries)
        path = args.output_dir / "blind_registration.json"
        candidate = dict(blind_csv_sha256=_hash(args.blind_csv), sources=hashes,
                         query_count=len(reviewed))
        if path.is_file():
            previous = _read(path)
            if {key: previous.get(key) for key in candidate} != candidate:
                raise ValueError("blind registration changed; archive this run and register before dev again")
        else:
            _write(path, {**candidate, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"Registered {len(reviewed)} independent reviewed queries before dev selection")
        return 0
    if args.phase == "prepare":
        _, cache_hash = _analyses(queries,
                                  Path("artifacts/context_eval_step8/analysis_cache.json"),
                                  populate=True)
        print(f"Prepared {len(queries)} reviewed queries; source hashes={hashes}; cache={cache_hash}")
        return 0
    if args.phase == "blind":
        if args.blind_csv is None:
            raise ValueError("--blind-csv is required for phase blind")
        return await _blind_phase(args.output_dir, args.blind_csv)
    return await _phase(args.phase, args.output_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("register", "prepare", "dev", "test", "blind"), required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT)
    p.add_argument("--blind-csv", type=Path)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
