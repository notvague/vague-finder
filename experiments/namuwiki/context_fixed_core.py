"""Fixed-setting evaluation rules; no parameter selection or model calls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import comb
from statistics import median
from typing import Iterable

from experiments.namuwiki.context_step10_core import Setting, _bootstrap_delta
from experiments.namuwiki.evaluate_context_step8 import summarize


SETTING = Setting(0.5, 100, 100, 30, named_media_multiplier=2.0)
EXPECTED_NEW = {
    "context_context": 24, "context_mixed": 12, "context_control": 6,
    "regression_lyrics": 6, "regression_mood_sound": 6, "regression_image": 6,
}


@dataclass(frozen=True)
class PathCheck:
    executed: bool
    queried: bool
    error: str

    @property
    def verified(self) -> bool:
        return self.executed and self.queried and not self.error


def inspect_path(spans: Iterable[dict], name: str) -> PathCheck:
    """A positive weight alone is insufficient: require embed + DB spans."""
    values = list(spans)
    branch = [s for s in values if s["name"] == "path." + name]
    children = [s for s in values if s["name"] in {name + ".embed", name + ".query"}]
    errors = sorted({str(s["error"]) for s in branch + children if s.get("error")})
    queried = {s["name"] for s in children} >= {name + ".embed", name + ".query"}
    return PathCheck(bool(branch), queried,
                     ",".join(errors))


def paired_exact_p(gained: int, lost: int) -> float:
    """Two-sided exact McNemar/binomial test on discordant question pairs."""
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n)


def fixed_report(rows: list[dict], *, phase: str, expected: int) -> dict:
    if not rows or len(rows) != expected or len({r["query_id"] for r in rows}) != expected:
        raise ValueError("incomplete or duplicate evaluation")
    if any(not r["scoreable"] or r["unavailable_ids"] for r in rows):
        raise ValueError("unindexed labels must not disappear from denominators")
    positive = [r for r in rows if r["group"] in {"context_context", "context_mixed"}]
    controls = [r for r in rows if r["group"] == "context_control"]
    regression = [r for r in rows if r["source"] == "regression"]
    groups = summarize(rows)["groups"]
    # Controls are excluded from the main effect denominator.
    if positive:
        groups["context_positive"] = summarize(positive)["groups"]["all"]
        for category in sorted({r["category"] for r in positive}):
            subset = [r for r in positive if r["category"] == category]
            groups["category_" + category] = summarize(subset)["groups"]["all"]
    issues: list[str] = []
    for row in rows:
        qid = row["query_id"]
        if row["group"] in {"context_context", "context_mixed"} and not row["context_clue_count"]:
            issues.append(f"{qid}: missing_context_clue")
        if row["group"] in {"context_context", "context_mixed"} and not row["on_context_executed"]:
            issues.append(f"{qid}: context_path_not_executed")
        if row["group"] in {"context_control", "regression_lyrics", "regression_mood_sound", "regression_image"} and row["context_clue_count"]:
            issues.append(f"{qid}: unexpected_context_clue")
        if not row["context_clue_count"] and row["off_candidate_ids"] != row["on_candidate_ids"]:
            issues.append(f"{qid}: no_clue_candidate_order_changed")
        if row["off_context_executed"]:
            issues.append(f"{qid}: context_executed_in_off_arm")
        if row["query_id"] in {"nw001", "q203"} and not row["on_candidate_hit30"]:
            issues.append(f"{qid}: required_candidate30_missing")
        if row["on_hit10"] < row["off_hit10"]:
            issues.append(f"{qid}: hit10_loss")
        if row in controls or row in regression:
            for metric in ("candidate_hit30", "hit1", "hit5", "mrr10"):
                if row[f"on_{metric}"] < row[f"off_{metric}"]:
                    issues.append(f"{qid}: protected_{metric}_loss")
        if row.get("required_path"):
            for arm in ("off", "on"):
                if not row[f"{arm}_required_path_verified"]:
                    issues.append(f"{qid}: {arm}_{row['required_path']}_path_not_verified")
        if row["group"] == "context_control" and row["displayed_evidence"]:
            issues.append(f"{qid}: control_has_displayed_evidence")
        for arm in ("off", "on"):
            if row[f"{arm}_path_errors"]:
                issues.append(f"{qid}: {arm}_retrieval_error")
        if row.get("rerank_executed"):
            issues.append(f"{qid}: rerank_executed")

    coverage = {}
    for path in ("image", "audio"):
        required = [r for r in rows if r.get("required_path") == path]
        coverage[path] = {
            "required_questions": len(required),
            "off_verified": sum(r["off_required_path_verified"] for r in required),
            "on_verified": sum(r["on_required_path_verified"] for r in required),
            "all_questions_off_executed": sum(r[f"off_{path}_executed"] for r in rows),
            "all_questions_on_executed": sum(r[f"on_{path}_executed"] for r in rows),
        }
    evidence = [dict(query_id=r["query_id"], **e) for r in rows for e in r["displayed_evidence"]]
    unreviewed = [e for e in evidence if e["label_match"] is not True]
    if phase == "independent":
        if Counter(r["group"] for r in rows) != Counter(EXPECTED_NEW):
            raise ValueError("independent group inventory changed")
        effect = groups["context_positive"]["metrics"]
        if effect["hit10"]["delta"] <= 0 or effect["candidate_hit30"]["delta"] < 0:
            issues.append("no_positive_context_hit10_improvement")
    elif phase not in {"dev", "test"}:
        raise ValueError("unknown phase")

    paired = {}
    for metric in ("candidate_hit30", "hit10"):
        gained = sum(r[f"on_{metric}"] > r[f"off_{metric}"] for r in positive)
        lost = sum(r[f"on_{metric}"] < r[f"off_{metric}"] for r in positive)
        paired[metric] = dict(gained=gained, lost=lost, exact_two_sided_p=paired_exact_p(gained, lost),
                              bootstrap_95_delta=_bootstrap_delta(positive, metric))
    paired["mrr10"] = dict(bootstrap_95_delta=_bootstrap_delta(positive, "mrr10"))
    status = "blocked" if issues else "evidence_review_required" if unreviewed else "passed"
    effect_signal = ("no_positive_observed_change" if paired["hit10"]["gained"] <= paired["hit10"]["lost"] else
                     "positive_observed_change_with_limited_statistical_support"
                     if paired["hit10"]["exact_two_sided_p"] > 0.05 else
                     "positive_change_supported_in_this_authored_sample")
    return {
        "phase": phase, "status": status, "evaluated": len(rows), "setting": SETTING.as_dict(),
        "rerank_enabled": False, "groups": groups, "violations": sorted(set(issues)),
        "path_coverage": coverage, "paired_context_tests": paired,
        "exploratory_hit10_signal": effect_signal,
        "evidence": {"displayed_count": len(evidence), "label_matched_count": len(evidence) - len(unreviewed),
                     "unreviewed_count": len(unreviewed), "review_items": unreviewed,
                     "target_top10_without_evidence": [r["query_id"] for r in positive
                                                       if r["on_hit10"] and not r["target_evidence_record_id"]]},
        "search_timing_ms_exploratory": {arm: {"median": median(r[f"{arm}_elapsed_ms"] for r in rows)}
                                          for arm in ("off", "on")},
        "claims": ["Same fresh QueryAnalyzer result for both arms of each question.",
                   "Effect denominator excludes no-context controls.",
                   "Independent questions are corpus-grounded authored tests, not sampled real users.",
                   "Unmatched evidence labels require review; they are not automatically false.",
                   "Search timings exclude QueryAnalyzer/API time and include possible first-model load.",
                   "The exact paired p-values are exploratory; repeated questions are not new samples.",
                   "Top-10 is the fixed product display size, not a demonstrated optimal UI size."],
    }
