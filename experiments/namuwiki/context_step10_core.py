"""Pure, label-aware selection rules for the full-corpus Context evaluation.

No test split is accepted by ``select_dev``. The reused test split reports
failures, while an independently prepared blind split supports the final gate.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Mapping

from experiments.namuwiki.evaluate_context_step8 import METRICS, summarize


EXPECTED = {"dev": (21, 53), "test": (9, 23)}
DECISION_METRICS = ("candidate_hit30", "hit1", "hit5", "hit10")
MUST_RECALL_AT_30 = frozenset({"nw001", "q203"})
KNOWN_EVIDENCE_ERRORS = frozenset({"nw005", "nw021", "nw024"})
DISPLAY_DEPTHS = (5, 10, 20)


@dataclass(frozen=True)
class Setting:
    weight: float
    fact_k: int
    sparse_k: int
    candidate_k: int
    top_k: int = 10
    named_media_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Context weight must be positive and finite")
        if (not math.isfinite(self.named_media_multiplier)
                or not 1 <= self.named_media_multiplier <= 3):
            raise ValueError("named media multiplier must be finite and within 1..3")
        if not (1 <= self.fact_k <= 1000 and 1 <= self.sparse_k <= 1000):
            raise ValueError("Context search widths must be within 1..1000")
        if not (10 <= self.candidate_k <= 100) or self.top_k != 10:
            raise ValueError("comparison requires candidate_k in 10..100 and top_k=10")

    @property
    def key(self) -> str:
        weight = f"{self.weight:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        key = f"w{weight}_d{self.fact_k}_s{self.sparse_k}_c{self.candidate_k}"
        if self.named_media_multiplier != 1:
            mult = f"{self.named_media_multiplier:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            key += f"_m{mult}"
        return key

    def as_dict(self) -> dict:
        return dict(weight=self.weight, fact_k=self.fact_k,
                    sparse_k=self.sparse_k, candidate_k=self.candidate_k,
                    top_k=self.top_k,
                    named_media_multiplier=self.named_media_multiplier)


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {
        "all": rows,
        "context_all": [r for r in rows if r["source"] == "context"],
        "regression_all": [r for r in rows if r["source"] == "regression"],
        "no_context_clue": [r for r in rows if not r["context_clue_count"]],
    }
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
        if row["source"] == "context":
            groups.setdefault("category_" + row["category"], []).append(row)
    return groups


def _bootstrap_delta(group: list[dict], metric: str, *, seed: int = 2026) -> list[float] | None:
    if not group:
        return None
    values = [row[f"on_{metric}"] - row[f"ref_{metric}"] for row in group]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(2000))
    return [draws[49], draws[1949]]


def display_depth_tradeoffs(rows: list[dict]) -> dict:
    """Hypothetical display prefixes from the same rerank-OFF candidate order.

    All depths use exactly the same candidate pool. This measures answer
    visibility, not UI usefulness or the latency/cost of rendering 20 items.
    """
    output = {}
    for name, group in _groups(rows).items():
        if name not in {"context_all", "regression_all", "context_control"} or not group:
            continue
        output[name] = {}
        for depth in DISPLAY_DEPTHS:
            before, after = [], []
            for row in group:
                eligible = set(row["eligible_ids"].split("|")) - {""}
                before.append(bool(eligible & set(row["ref_candidate_ids"].split("|")[:depth])))
                after.append(bool(eligible & set(row["on_candidate_ids"].split("|")[:depth])))
            output[name][f"hit{depth}"] = dict(
                queries=len(group), ref=sum(before) / len(group), on=sum(after) / len(group),
                gained=[row["query_id"] for row, old, new in zip(group, before, after)
                        if new and not old],
                lost=[row["query_id"] for row, old, new in zip(group, before, after)
                      if old and not new],
            )
    return output


def compare(rows: list[dict], setting: Setting) -> dict:
    """Report matched OFF/ON and the fixed OFF@30 production reference."""
    if not rows:
        raise ValueError("empty evaluation")
    query_ids = [r["query_id"] for r in rows]
    if len(query_ids) != len(set(query_ids)) or any(not r["scoreable"] for r in rows):
        raise ValueError("duplicate query or unavailable label; refuse incomplete score")
    matched = summarize(rows)
    groups = _groups(rows)
    reference = {}
    for name, group in groups.items():
        reference[name] = {
            "queries": len(group),
            "candidate_hit_at_configured_width": {
                "off": sum(r["off_candidate_hit_width"] for r in group) / len(group),
                "on": sum(r["on_candidate_hit_width"] for r in group) / len(group),
            },
            "timing_ms_exploratory": {
                "off_median": median(r["off_elapsed_ms"] for r in group),
                "on_median": median(r["on_elapsed_ms"] for r in group),
            },
            "metrics": {
                metric: {
                    "ref": sum(r[f"ref_{metric}"] for r in group) / len(group),
                    "on": sum(r[f"on_{metric}"] for r in group) / len(group),
                    "delta": sum(r[f"on_{metric}"] - r[f"ref_{metric}"] for r in group) / len(group),
                    "gained": [r["query_id"] for r in group
                               if r[f"on_{metric}"] > r[f"ref_{metric}"]],
                    "lost": [r["query_id"] for r in group
                             if r[f"on_{metric}"] < r[f"ref_{metric}"]],
                    "bootstrap_95_delta": _bootstrap_delta(group, metric)
                    if metric in {"candidate_hit30", "hit10"} else None,
                } for metric in METRICS
            },
        }

    violations: list[str] = []
    for row in rows:
        if row["query_id"] in MUST_RECALL_AT_30 and not row["on_candidate_hit30"]:
            violations.append(f"{row['query_id']}: required Context candidate@30 missing")
        if (row["query_id"] in KNOWN_EVIDENCE_ERRORS
                and row["target_evidence_labeled"] is False):
            violations.append(f"{row['query_id']}: displayed fact differs from reviewed event")
        if not row["context_clue_count"] and row["off_candidate_ids"] != row["on_candidate_ids"]:
            violations.append(f"{row['query_id']}: no-clue OFF/ON candidate mismatch")
        if row["source"] == "regression" or row["group"] == "context_control":
            for metric in DECISION_METRICS:
                if row[f"on_{metric}"] < row[f"ref_{metric}"]:
                    violations.append(f"{row['query_id']}: protected {metric} loss")
        elif row["on_hit10"] < row["ref_hit10"]:
            violations.append(f"{row['query_id']}: context hit10 loss")
        if row["group"] == "context_control" and row["target_evidence_record_id"]:
            violations.append(f"{row['query_id']}: control target has evidence")
    for kind in ("missing_context_clue", "unexpected_control_clue", "unexpected_regression_clue"):
        for query_id in matched["issues"][kind]:
            violations.append(f"{query_id}: {kind}")

    context = reference["context_all"]["metrics"]
    if not (context["hit10"]["delta"] > 0 and context["candidate_hit30"]["delta"] >= 0):
        violations.append("no positive Context Hit@10 improvement over OFF@30")
    evidence = {
        "displayed_facts": sum(len(r["displayed_evidence"]) for r in rows),
        "displayed_on_non_target_songs": sum(
            sum(item["song_id"] not in r["eligible_ids"].split("|")
                for item in r["displayed_evidence"]) for r in rows
        ),
        "labeled_target_evidence": [r["query_id"] for r in rows
                                    if r["target_evidence_labeled"] is True],
        "unlabeled_target_evidence_to_review": [r["query_id"] for r in rows
                                                 if r["target_evidence_labeled"] is False],
        "target_top10_without_displayed_fact": [r["query_id"] for r in rows
                                                 if r["source"] == "context"
                                                 and r["group"] != "context_control"
                                                 and r["on_hit10"] and not r["target_evidence_record_id"]],
    }
    tradeoffs = display_depth_tradeoffs(rows)
    for name, depths in tradeoffs.items():
        if depths["hit10"]["on"] != reference[name]["metrics"]["hit10"]["on"]:
            raise ValueError("candidate order and displayed Top-10 disagree")
    return dict(setting=setting.as_dict(), matched=matched, reference=reference,
                display_depth_tradeoffs=tradeoffs,
                evidence_audit=evidence, violations=violations, eligible=not violations)


def select_dev(results: Mapping[str, tuple[Setting, list[dict]]]) -> dict:
    if not results:
        raise ValueError("empty dev grid")
    reports = {}
    for key, (setting, rows) in results.items():
        if key != setting.key:
            raise ValueError("setting/key mismatch")
        if any(row["split"] != "dev" for row in rows):
            raise ValueError("test split must never enter dev selection")
        context_count = sum(row["source"] == "context" for row in rows)
        if (context_count, len(rows) - context_count) != EXPECTED["dev"]:
            raise ValueError("incomplete dev split")
        reports[key] = compare(rows, setting)
    safe = [key for key, value in reports.items() if value["eligible"]]
    if not safe:
        return dict(status="blocked", selected=None, candidates=reports,
                    reason="No dev setting passes protected regression/coverage gates")

    def order(key: str) -> tuple:
        setting, _ = results[key]
        context = reports[key]["reference"]["context_all"]["metrics"]
        # Predeclared priority: improve real top-10 answers, then candidate
        # recall and reciprocal rank; prefer smaller, cheaper settings on ties.
        return (context["hit10"]["on"], context["candidate_hit30"]["on"],
                context["mrr10"]["on"], -setting.weight,
                -setting.named_media_multiplier,
                -setting.fact_k, -setting.sparse_k, -setting.candidate_k)

    winner = max(safe, key=order)
    return dict(status="locked_for_test", selected=winner,
                setting=results[winner][0].as_dict(), candidates=reports,
                selection_rule="dev: nw001/q203 Candidate@30 required; zero protected/context Hit@10 losses; then Context Hit@10, Candidate@30, MRR@10, lower cost")


def heldout_verdict(rows: list[dict], setting: Setting) -> dict:
    if any(row["split"] != "test" for row in rows):
        raise ValueError("held-out report requires test rows only")
    context_count = sum(row["source"] == "context" for row in rows)
    if (context_count, len(rows) - context_count) != EXPECTED["test"]:
        raise ValueError("incomplete test split")
    report = compare(rows, setting)
    report["status"] = "provisional" if report["eligible"] else "not_recommended"
    report["policy"] = ("the v0.5 test split was already inspected in Step 8; "
                        "a fresh independent blind set is needed for a strong final claim; "
                        "losses block deployment and must not tune this setting")
    return report


def blind_verdict(rows: list[dict], setting: Setting) -> dict:
    groups = Counter(r["group"] for r in rows)
    required = {"context_context", "context_mixed", "context_control",
                "regression_lyrics", "regression_mood_sound", "regression_image"}
    categories = {r["category"] for r in rows if r["source"] == "context"
                  and r["group"] != "context_control"}
    if (len(rows) < 30 or any(groups[group] < 3 for group in required)
            or len(categories) < 6 or any(r["split"] != "blind" for r in rows)):
        raise ValueError("blind set needs >=30 cases, >=3 per required group and >=6 fact categories")
    report = compare(rows, setting)
    report["status"] = "recommended" if report["eligible"] else "not_recommended"
    report["policy"] = "new songs and reviewed queries, evaluated once after dev lock; no retuning on blind labels"
    return report
