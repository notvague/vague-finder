"""Verify the rebuilt HTTP API after validated Context settings are applied.

Reuses already observed regression questions from private evaluation files.
Makes new real QueryAnalyzer/API requests; it is a smoke check, not another
independent sample. It never creates a second local Qdrant client.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from experiments.namuwiki.apply_validated_context_settings import (
    DEPENDENCIES, SETTING, assert_registered, read_json, sha, validate_pass,
)
from experiments.namuwiki.context_api_timing_checks import (
    read_request_timing, timing_path, verify_media_execution,
)
from experiments.namuwiki.context_api_candidate_proof import (
    candidate_probe_payload, positive_context_ids, verify_candidate_probe, verify_context_execution,
)


def request_json(url: str, body: dict | None = None, *, timeout: float = 120) -> dict:
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
                      headers={"Content-Type": "application/json"} if body is not None else {})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"HTTP status {response.status}")
        return json.loads(response.read().decode("utf-8"))


def request_search(url: str, body: dict, *, timeout: float = 120) -> dict:
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"HTTP status {response.status}")
        return dict(status=response.status, request_id=response.headers.get("X-Request-Id", ""),
                    body=json.loads(response.read().decode("utf-8")))


def wait_ready(base_url: str, seconds: float) -> None:
    deadline, last_print = time.monotonic() + seconds, 0.0
    while time.monotonic() < deadline:
        try:
            health = request_json(base_url + "/health", timeout=5)
            if health.get("status") == "ok" and health.get("warmup", {}).get("state") == "ready":
                print("[PASS] Backend health and model warmup ready", flush=True)
                return
        except Exception:
            pass
        if time.monotonic() - last_print >= 10:
            print("Waiting for backend warmup...", flush=True)
            last_print = time.monotonic()
        time.sleep(2)
    raise TimeoutError("Backend did not become ready; inspect docker compose logs --tail 80 backend")


def check_openapi(document: dict) -> None:
    def resolve(schema):
        while "$ref" in schema:
            ref = schema["$ref"]
            if not ref.startswith("#/"):
                raise ValueError("External API schema reference is not supported")
            schema = document
            for part in ref[2:].split("/"):
                schema = schema[part]
        return schema

    response = document["paths"]["/api/v1/search"]["post"]["responses"]["200"]
    root = resolve(response["content"]["application/json"]["schema"])
    items = resolve(resolve(root["properties"]["results"])["items"])
    if "context_evidence" not in items.get("properties", {}):
        raise ValueError("Step 7 context_evidence field is absent from the actual search response schema")


def select_cases(root: Path) -> list[dict]:
    entries = read_json(root / "fresh/dev_analysis_cache.json")["entries"]
    with (root / "fresh/dev_detail.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["query_id"]: row for row in rows}
    selected = []

    def add(row: dict, path: str, *, target=False, control=False,
            cache=entries, phase="dev"):
        qid = row["query_id"]
        if qid not in cache:
            raise ValueError(f"{qid}: observed analysis is missing")
        query = cache[qid]["query"]
        if row.get("query", query) != query or cache[qid]["analysis"].get("original_query", query) != query:
            raise ValueError(f"{qid}: observed source query is inconsistent")
        selected.append(dict(query_id=qid, query=query, required_path=path, source_phase=phase,
                             expected_targets=[value for value in row["eligible_ids"].split("|") if value] if target else [],
                             no_context=control))

    for qid in ("nw001", "q203"):
        if qid not in by_id:
            raise ValueError(f"Required known regression case missing: {qid}")
        add(by_id[qid], "context", target=True)
    for group, path in (("regression_image", "image"), ("regression_mood_sound", "audio")):
        options = [row for row in rows if row["group"] == group
                   and entries[row["query_id"]]["analysis"].get(path + "_english_query")]
        if not options:
            raise ValueError(f"Observed {path} regression case is missing")
        add(options[0], path, control=True)
    controls = [row for row in rows if row["group"] == "context_control"]
    if not controls:
        raise ValueError("Observed no-Context control is missing")
    add(controls[0], "", control=True)
    # Preserve q105/the original mixed-audio choice above. Add every question
    # explicitly registered as requiring a media path, selected by inventory
    # and ID order alone, without reading hit rates, ranks or API outcomes.
    cache = read_json(root / "fresh/independent_analysis_cache.json")["entries"]
    with (root / "fresh/independent_detail.csv").open(encoding="utf-8-sig", newline="") as stream:
        independent = list(csv.DictReader(stream))
    for path, group in (("image", "regression_image"), ("audio", "regression_mood_sound")):
        options = sorted((row for row in independent if row.get("required_path") == path),
                         key=lambda row: row["query_id"])
        if len(options) != 6 or len({row["query_id"] for row in options}) != 6:
            raise ValueError(f"Expected all six registered {path}-required questions")
        for row in options:
            if row["group"] != group:
                raise ValueError(f"{row['query_id']}: registered media group is inconsistent")
            add(row, path, target=True, control=True, cache=cache, phase="independent")
    if len({case["query_id"] for case in selected}) != len(selected):
        raise ValueError("Duplicate API smoke question IDs")
    return selected


def _response_contract(case: dict, body: dict, *, rejected: tuple[str, ...] = (),
                       timing_record: dict | None = None, result_k: int = 10) -> dict:
    analysis, summary = body.get("analysis"), body.get("explain")
    if not isinstance(analysis, dict) or not isinstance(summary, dict):
        raise ValueError("API analysis/explain output is missing")
    if analysis.get("original_query") != case["query"]:
        raise ValueError("The API changed the original source query")
    if summary.get("analysis_fallback") is not False or summary.get("failed_paths") != []:
        raise ValueError("The API fell back or a retrieval path failed")
    if summary.get("rerank_calls") != 0 or summary.get("rerank_calls_applied") != 0:
        raise ValueError("Reranking unexpectedly ran in a rerank-OFF check")
    candidates, results = body.get("candidate_ids"), body.get("results")
    if (not isinstance(candidates, list) or not isinstance(results, list)
            or len(candidates) != 30 or len(candidates) != len(set(candidates))
            or not 1 <= len(results) <= result_k):
        raise ValueError(f"Expected 30 distinct candidates and 1–{result_k} returned tracks")
    ids = [str(track["id"]) for track in results]
    if (len(ids) != len(set(ids)) or not set(ids) <= set(candidates)
            or set(rejected) & (set(candidates) | set(ids))):
        raise ValueError("Duplicate, out-of-pool, or rejected tracks returned")
    if case["expected_targets"] and not set(case["expected_targets"]) & set(candidates):
        raise ValueError("Required known target did not enter Candidate@30 in the new API analysis")
    if case["no_context"] and analysis.get("context_clues"):
        raise ValueError("An image/audio/control query acquired unexpected Context clues")
    positive_paths, evidence_count = set(), 0
    for track in results:
        if not isinstance(track.get("title"), str) or not track["title"].strip() or track["title"] == "Unknown":
            raise ValueError("Canonical track title is missing")
        explanation = track.get("explain")
        if not isinstance(explanation, dict):
            raise ValueError("Track explain output is missing")
        for contribution in explanation.get("paths", []):
            delta = float(contribution["delta"])
            if not math.isfinite(delta):
                raise ValueError("Nonfinite path contribution")
            if delta > 0 and not contribution.get("dropped", False):
                positive_paths.add(contribution["path"])
        fact = track.get("context_evidence")
        facts = fact if isinstance(fact, list) else [fact] if fact else []
        if facts and (case["no_context"] or not analysis.get("context_clues")):
            raise ValueError("Fact evidence appeared without a source Context clue")
        for item in facts:
            if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip()
                                                     for key in ("record_id", "fact_text", "source_url")):
                raise ValueError("Fact evidence has an empty required field")
            url = urlsplit(item["source_url"])
            if (url.scheme != "https" or url.hostname != "namu.wiki" or url.port not in (None, 443)
                    or url.username is not None or url.password is not None or not url.path.startswith("/w/")):
                raise ValueError("Fact evidence has an invalid Namuwiki source URL")
            evidence_count += 1
    required = case["required_path"]
    media = None
    executed_media = {}
    if timing_record is not None:
        if (timing_record.get("query") != case["query"]
                or timing_record.get("attrs", {}).get("status") != 200):
            raise ValueError("Timing record belongs to a different or unsuccessful query")
        spans = timing_record.get("spans")
        if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
            raise ValueError("Actual request timing spans are absent or malformed")
        if any(span.get("name") == "search.rerank" for span in spans):
            raise ValueError("Reranking executed according to the actual request timing")
        for name in ("image", "audio"):
            if any(span.get("name") == "path." + name for span in spans):
                executed_media[name] = verify_media_execution(analysis, timing_record, name)
    if required in {"image", "audio"}:
        if timing_record is None:
            raise ValueError(f"Same-request timing is required to verify {required} execution")
        if required not in executed_media:
            raise ValueError(f"{required} path did not execute in this HTTP request")
        media = executed_media[required]
    elif required and required != "context" and required not in positive_paths:
        raise ValueError(f"No positive {required} contribution observed in the real API results")
    if case["no_context"] and "context" in positive_paths:
        raise ValueError("Context contributed to a no-Context query")
    return dict(query_id=case["query_id"], required_path=required, candidate_count=len(candidates),
                result_count=len(results), positive_paths=sorted(positive_paths),
                target_candidate_ranks={sid: candidates.index(sid) + 1 for sid in case["expected_targets"] if sid in candidates},
                target_top10_ranks={sid: ids[:10].index(sid) + 1 for sid in case["expected_targets"] if sid in ids[:10]},
                evidence_count=evidence_count, rejected_ids=list(rejected),
                analysis_fallback=False, rerank_calls=0, media_execution=media, executed_media=executed_media,
                required_path_in_top10=required in positive_paths if required else None)


def check_response(case: dict, body: dict, *, rejected: tuple[str, ...] = (),
                   timing_record: dict | None = None, candidate_probe: dict | None = None,
                   primary_request: dict | None = None) -> dict:
    row = _response_contract(case, body, rejected=rejected, timing_record=timing_record)
    if case["required_path"] != "context":
        if candidate_probe is not None:
            raise ValueError("Candidate Context proof was supplied for an unrelated query")
        return row
    if timing_record is None:
        raise ValueError("Same-request timing is required to verify Context execution")
    execution = verify_context_execution(body["analysis"], timing_record)
    if candidate_probe is None:
        # A shown target with a positive vote is direct proof. A target only
        # in candidate_ids needs the paired full-candidate HTTP inspection.
        positive = positive_context_ids(body)
        if (not positive or case["expected_targets"]
                and not set(case["expected_targets"]) & set(positive)):
            raise ValueError("A target outside positive Context Top-10 results needs Candidate@30 HTTP proof")
        row["context_candidate_proof"] = dict(verified=True, scope="returned_top10", positive_context_ids=positive)
    else:
        if primary_request is None:
            raise ValueError("Original HTTP request is required for a candidate replay proof")
        _response_contract(case, candidate_probe["response"], rejected=rejected,
                           timing_record=candidate_probe["timing"], result_k=30)
        row["context_candidate_proof"] = verify_candidate_probe(
            case, body, timing_record, candidate_probe, primary_request=primary_request,
        )
    row["context_execution"] = execution
    return row


def verify(project: Path, output: Path, base_url: str, wait_seconds: float) -> dict:
    destination = output / "activation/api_smoke_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = destination.read_bytes()
        history = destination.parent / "history"
        history.mkdir(exist_ok=True)
        (history / ("api_smoke_" + sha(previous) + ".json")).write_bytes(previous)
    result = dict(schema="context_api_ready_smoke_v3", status="running", setting=SETTING,
                  recorded_at=datetime.now(timezone.utc).isoformat(), rows=[], requests=[],
                  verifier_sha256=sha(Path(__file__).read_bytes()),
                  timing_checks_sha256=sha(Path(__file__).with_name("context_api_timing_checks.py").read_bytes()),
                  candidate_proof_sha256=sha(Path(__file__).with_name("context_api_candidate_proof.py").read_bytes()),
                  policy="Same-response request ID and appended server timing prove media execution. Context additionally requires a paired prior-analysis HTTP replay returning the exact Candidate@30 order, unchanged Top-10, and positive live Context contributions for a required target. Candidate entry and Top-10 survival are reported separately.",
                  limitations="Known regression API smoke with new analyses, including all 12 registered media-required cases. This is a follow-up verification, not a new independent sample. Frozen OFF/ON reports remain the accuracy evidence. Reranking is OFF.")
    activation = None
    activation_checked = False
    try:
        report_data = (output / "fresh/validation_report.json").read_bytes()
        report = json.loads(report_data.decode("utf-8-sig"))
        registration = read_json(output / "fresh/registration.json")
        validate_pass(report, registration)
        activation = read_json(output / "activation/activation_report.json")
        if (activation.get("status") != "settings_applied" or activation.get("setting") != SETTING
                or activation.get("validation_report_sha256") != sha(report_data)):
            raise ValueError("Applied settings are not bound to this passed evaluation")
        activation_checked = True
        assert_registered(project, registration, activation, sha(report_data))
        result["dependencies_sha256"] = sha((project / DEPENDENCIES).read_bytes())
        wait_ready(base_url, wait_seconds)
        check_openapi(request_json(base_url + "/openapi.json"))
        log = timing_path(project, registration)
        result["timing_log"] = str(log.relative_to(project.resolve()))
        cases = select_cases(output)
        result["selected_cases"] = cases
        result["case_selection"] = "Original five dev smoke questions plus all six registered image and six registered audio requirements; no outcome filtering."
        failures = []
        first = None

        def run_case(case: dict, payload: dict, rejected=()) -> dict | None:
            started = time.perf_counter()
            diagnostic = dict(query_id=case["query_id"], kind="primary", request=payload, stage="http_request")
            result["requests"].append(diagnostic)
            try:
                offset = log.stat().st_size if log.is_file() else 0
                received = request_search(base_url + "/api/v1/search", payload)
                diagnostic.update(http_status=received["status"], request_id=received["request_id"],
                                  response=received["body"], stage="timing_correlation")
                record = read_request_timing(log, offset, received["request_id"], case["query"])
                diagnostic.update(timing=record, stage="response_contract")
                _response_contract(case, received["body"], rejected=tuple(rejected), timing_record=record)
                probe = None
                if case["required_path"] == "context":
                    verify_context_execution(received["body"]["analysis"], record)
                    diagnostic["stage"] = "candidate_probe"
                    probe = dict(query_id=case["query_id"], kind="context_candidate_probe",
                                 request=candidate_probe_payload(payload, received["body"]), stage="http_request")
                    result["requests"].append(probe)
                    offset = log.stat().st_size if log.is_file() else 0
                    try:
                        inspected = request_search(base_url + "/api/v1/search", probe["request"])
                        probe.update(http_status=inspected["status"], request_id=inspected["request_id"],
                                     response=inspected["body"], stage="timing_correlation")
                        probe["timing"] = read_request_timing(log, offset, inspected["request_id"], case["query"])
                        probe["stage"] = "candidate_proof_contract"
                    except Exception as exc:
                        probe.update(error_type=type(exc).__name__, error=str(exc))
                        raise
                row = check_response(case, received["body"], rejected=tuple(rejected), timing_record=record,
                                     candidate_probe=probe, primary_request=payload)
                if probe is not None:
                    probe["stage"] = "passed"
                row.update(status="passed", client_ms=round((time.perf_counter() - started) * 1000),
                           request_id=received["request_id"])
                result["rows"].append(row)
                diagnostic["stage"] = "passed"
                media = row["media_execution"]
                execution = f", {case['required_path']}_embed_query=verified, in_top10={row['required_path_in_top10']}" if media else ""
                if probe is not None:
                    execution += f", context_candidate_vote=verified, in_top10={row['required_path_in_top10']}"
                print(f"[API PASS] {row['query_id']}: candidates=30, results={row['result_count']}, "
                      f"paths={','.join(row['positive_paths'])}, targets={row['target_candidate_ranks']}"
                      f", target_top10={row['target_top10_ranks']}{execution}, {row['client_ms']}ms", flush=True)
                return received["body"]
            except Exception as exc:
                if diagnostic["stage"] == "candidate_probe" and probe is not None:
                    probe.update(error_type=type(exc).__name__, error=str(exc))
                diagnostic.update(error_type=type(exc).__name__, error=str(exc),
                                  client_ms=round((time.perf_counter() - started) * 1000))
                row = dict(query_id=case["query_id"], status="failed", stage=diagnostic["stage"],
                           error_type=type(exc).__name__, error=str(exc))
                result["rows"].append(row)
                failures.append(row)
                print(f"[API FAIL] {case['query_id']}: {type(exc).__name__}: {exc}", flush=True)
                return None
            finally:
                # Keep the actual failed analysis/HTTP body/timing, not just
                # the name of the first failure. The report stays private.
                destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        for case in cases:
            body = run_case(case, dict(query=case["query"], top_k=10, candidate_k=30,
                                      use_rerank=False, explain=True))
            if first is None and body is not None and case["query_id"] == "nw001":
                first = (case, body)
        if first is not None:
            case, body = first
            rejected = tuple(str(track["id"]) for track in body["results"][:5])
            refill_case = {**case, "query_id": case["query_id"] + ":reject_refill", "expected_targets": []}
            run_case(refill_case, dict(query=case["query"], top_k=10, candidate_k=30,
                                     use_rerank=False, explain=True, prior_analysis=body["analysis"],
                                     previous_candidate_ids=body["candidate_ids"], rejected_ids=list(rejected),
                                     turn=1), rejected=rejected)
        else:
            failures.append(dict(query_id="nw001:reject_refill", error="First Context case failed; refill check unavailable"))
        successful = [row for row in result["rows"] if row["status"] == "passed"]
        coverage = {}
        for name in ("image", "audio"):
            checked = [row for row in successful if name in row["executed_media"]]
            contributing = [row for row in checked if name in row["positive_paths"]]
            coverage[name] = dict(execution_verified_queries=[row["query_id"] for row in checked],
                                  top10_positive_verified_queries=[row["query_id"] for row in contributing])
            # A mixed query need not retain a media hit, but the complete API
            # smoke must still observe actual returned hits from both engines.
            # Successful queries against an empty/miswired index cannot pass.
            if not contributing:
                failures.append(dict(query_id=name + ":aggregate_hit_confirmation",
                                     error=f"No verified {name} execution yielded a positive returned contribution across the API smoke"))
        result["media_coverage"] = coverage
        result["primary_checks"] = len(result["rows"])
        result["candidate_probe_requests"] = sum(row["kind"] == "context_candidate_probe" for row in result["requests"])
        result["http_requests"] = len(result["requests"])
        result["failures"] = failures
        if failures:
            raise ValueError(f"{len(failures)} API checks failed; full responses and timing are in the private API report")
        result["status"] = "passed_api_smoke"
    except Exception as exc:
        result["status"] = "failed_api_smoke"
        result["error_type"], result["error"] = type(exc).__name__, str(exc)
        print(f"API smoke stopped: {result['error_type']}: {result['error']}", flush=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if activation_checked:
        activation["api_verified"] = result["status"] == "passed_api_smoke"
        activation["api_smoke_report_sha256"] = sha(destination.read_bytes())
        activation["api_verification_recorded_at"] = result["recorded_at"]
        (output / "activation/activation_report.json").write_text(
            json.dumps(activation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(f"Private API report: {destination}", flush=True)
    if result["status"] == "passed_api_smoke":
        print(f"[PASS] {result['primary_checks']} API checks / {result['http_requests']} HTTP requests: "
              "Context Candidate@30 votes, same-request Image/Audio embed+query, controls and rejected-ID refill; rerank OFF", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-ready-seconds", type=float, default=300)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = (project / args.output_dir).resolve()
    if project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        parser.error("Output directory must be inside artifacts")
    result = verify(project, output, args.base_url.rstrip("/"), args.wait_ready_seconds)
    return 0 if result["status"] == "passed_api_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
