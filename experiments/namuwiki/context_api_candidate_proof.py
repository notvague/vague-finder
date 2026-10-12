"""Prove Context contributions in Candidate@30 separately from screen Top-10.

The candidate inspection is another real HTTP request using the first response's
analysis, the same candidate limit and state, and no reranking. It is a replay,
not a new analysis or a new accuracy evaluation. No backend/vector imports.
"""

from __future__ import annotations

from copy import deepcopy
import math


def positive_context_ids(body: dict) -> list[str]:
    ids = []
    for track in body["results"]:
        contributions = [item for item in track["explain"].get("paths", [])
                         if item.get("path") == "context"]
        if len(contributions) > 1:
            raise ValueError("A song has duplicate Context path contributions")
        for item in contributions:
            delta, rank = item.get("delta"), item.get("rank")
            if (isinstance(delta, bool) or not isinstance(delta, (int, float))
                    or not math.isfinite(delta) or isinstance(rank, bool)
                    or not isinstance(rank, int) or rank < 1):
                raise ValueError("Context contribution has an invalid delta or branch rank")
            if delta > 0 and item.get("dropped", False) is False:
                ids.append(str(track["id"]))
    return ids


def verify_context_execution(analysis: dict, record: dict) -> dict:
    clues = analysis.get("context_clues")
    if not isinstance(clues, list) or not clues:
        raise ValueError("New API analysis has no source Context clue")
    for clue in clues:
        confidence = clue.get("confidence") if isinstance(clue, dict) else None
        if (not isinstance(clue, dict)
                or not isinstance(clue.get("search_query"), str) or not clue["search_query"].strip()
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 < confidence <= 1):
            raise ValueError("Source Context clue has no valid query/confidence")
    if (record.get("query") != analysis.get("original_query")
            or record.get("attrs", {}).get("status") != 200):
        raise ValueError("Context timing is not from this successful source query")
    spans = record.get("spans")
    if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
        raise ValueError("Context timing spans are missing or malformed")
    by_id = {}
    for span in spans:
        sid = span.get("id")
        if isinstance(sid, bool) or not isinstance(sid, int) or sid <= 0 or sid in by_id:
            raise ValueError("Context timing span IDs must be unique positive integers")
        by_id[sid] = span
    for sid, span in by_id.items():
        cursor, seen = span.get("parent"), {sid}
        while True:
            if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                raise ValueError("Context timing ancestry has an invalid parent")
            if cursor == 0:
                break
            if cursor in seen or cursor not in by_id:
                raise ValueError("Context timing ancestry is cyclic or incomplete")
            seen.add(cursor)
            cursor = by_id[cursor].get("parent")
    if any(span.get("name", "").startswith("search.rerank") for span in spans):
        raise ValueError("Context proof unexpectedly executed reranking")
    jobs = [span for span in spans if span.get("name") == "path.context"]
    if len(jobs) != 1:
        raise ValueError("Expected one successful Context job in this HTTP request")
    job = jobs[0]
    if job.get("error"):
        raise ValueError("Context path failed in this HTTP request")
    for field in ("wait_ms", "run_ms"):
        value = job.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("Context timing job has an invalid duration")
    return dict(verified=True, job_id=job["id"], run_ms=job["run_ms"], wait_ms=job["wait_ms"])


def candidate_probe_payload(request: dict, response: dict) -> dict:
    if (request.get("query") != response.get("analysis", {}).get("original_query")
            or request.get("top_k") != 10 or request.get("candidate_k") != 30
            or request.get("use_rerank") is not False or request.get("explain") is not True):
        raise ValueError("Candidate proof requires the original rerank-OFF Top-10/Candidate@30 request")
    return {**deepcopy(request), "top_k": 30, "prior_analysis": deepcopy(response["analysis"])}


def verify_candidate_probe(case: dict, primary: dict, primary_record: dict,
                           probe: dict, *, primary_request: dict) -> dict:
    """Use fully validated HTTP bodies; neither timing alone nor presence suffices."""
    original_job = verify_context_execution(primary["analysis"], primary_record)
    if probe.get("request") != candidate_probe_payload(primary_request, primary):
        raise ValueError("Candidate probe changed query, state, limits, analysis or rerank settings")
    response, timing = probe.get("response"), probe.get("timing")
    if not isinstance(response, dict) or not isinstance(timing, dict):
        raise ValueError("Candidate probe HTTP response/timing is absent")
    primary_id, probe_id = primary_record.get("request_id"), probe.get("request_id")
    if (probe.get("http_status") != 200 or not isinstance(primary_id, str) or not primary_id.strip()
            or not isinstance(probe_id, str) or not probe_id.strip() or probe_id == primary_id
            or timing.get("request_id") != probe_id):
        raise ValueError("Candidate proof needs two distinct, correlated successful HTTP request IDs")
    if response.get("analysis") != primary["analysis"]:
        raise ValueError("Candidate probe did not reuse the exact original API analysis")
    if (timing.get("attrs", {}).get("analysis_mode") != "prior"
            or any(span.get("name") == "analysis" or span.get("name", "").startswith("analysis.")
                   for span in timing.get("spans", []))):
        raise ValueError("Candidate probe ran a new analysis instead of reusing prior_analysis")
    probe_job = verify_context_execution(response["analysis"], timing)
    candidates = primary["candidate_ids"]
    if response.get("candidate_ids") != candidates:
        raise ValueError("Candidate probe changed Candidate@30 membership or order")
    full_ids = [str(track["id"]) for track in response["results"]]
    if len(full_ids) != 30 or full_ids != candidates:
        raise ValueError("Candidate probe did not return all 30 candidates in their original order")
    shown = primary["results"]
    if shown != response["results"][:len(shown)]:
        raise ValueError("Candidate probe changed the original Top-10 tracks, scores, contributions or evidence")
    positive = positive_context_ids(response)
    expected = set(case["expected_targets"])
    positive_targets = [sid for sid in positive if sid in expected]
    if not positive or expected and not positive_targets:
        raise ValueError("Required target has no positive live Context contribution in Candidate@30")
    shown_ids = {str(track["id"]) for track in shown}
    return dict(schema="context_candidate_http_proof_v1", verified=True,
                primary_request_id=primary_id, probe_request_id=probe_id,
                same_analysis=True, same_candidate_order=True, same_top10=True,
                candidate_count=30, positive_context_ids=positive,
                positive_target_candidate_ranks={sid: candidates.index(sid) + 1 for sid in positive_targets},
                target_in_top10={sid: sid in shown_ids for sid in positive_targets},
                context_execution=original_job, probe_context_execution=probe_job,
                note="Paired HTTP replay; Candidate@30 contribution does not imply a Top-10 hit or prove fact relevance.")
