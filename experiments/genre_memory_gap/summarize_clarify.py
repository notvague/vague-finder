#!/usr/bin/env python
"""재질문 하네스 결과 폴더들을 정책별로 나란히 놓는다 (모델·DB 없음). 첫 실행이 기준이다.

`evaluate_clarification`의 `clarify_summary.csv` · `clarify_detail.csv` · `clarify_runinfo.json`을 읽는다. 개입 대상(최초 검색에서
정답이 10위 밖인 질의)은 실행마다 다를 수 있으므로, **비교에 쓰는 모든 실행에서 개입 대상인 같은 질의**만으로 정책을 견준 표를
함께 낸다(답변의 효과를 재는 표는 이쪽이다 — CLAUDE.md의 재질문 측정 규칙).

    venv/bin/python experiments/genre_memory_gap/summarize_clarify.py \\
        base1=<폴더> base2=<폴더> answer0_1=<폴더> … --output <clarify_compare.csv>

**표의 이름이 뜻을 잃는 실행은 받지 않는다.** detail의 `hit10`은 "보여 준 `--top-k`곡 안에 정답이 있는가"이고 후보 순위 열은
`--candidate-k` 깊이다. `top_k`가 10이 아니거나 `candidate_k`가 30이 아닌 실행(또는 runinfo에 그 값이 없는 실행)이 있으면 멈춘다.

**공통 개입 대상 비교에서 빼는 실행** — 공통 대상 집합을 정할 때도 쓰지 않는다. 실행별 표에는 이름에 `!`를 붙여 남기고, 뺀
사유를 화면과 CSV(`scope=비교에서 뺌`, `excluded_reason`)에 적는다.

- 리랭킹 실패가 섞인 실행 — 검색 순서로 폴백한 결과다. 정상으로 아는 상태(applied · skipped · policy_skipped:*)가 아니면 실패로 센다
- 리랭커 상태 집계(`clarify.rerank_status_counts`)가 없는 실행 — "실패 없음"이 아니라 "모름"이다
- Gemini 호출 경로(`gemini.backend` · `gemini.setting`)가 기준 실행과 다른 실행 — 경로가 다르면 같은 모델 이름이어도 출력이 다르다

그 밖의 조건(배율 · 답변 배수 · 분석 캐시 · 잰 질의 범위 …)은 실험 변수일 수 있어 막지 않는다. 대신 runinfo의 조건을 기준 실행과
통째로 견줘 **다른 칸을 전부** 화면과 CSV(`conditions_vs_base`)에 적는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.run_conditions import (  # noqa: E402
    condition_differences, flatten, rerank_failures, run_conditions,
)

POLICIES = (
    "initial", "reject_only", "oracle:genre", "noisy:genre", "oracle:vocal_gender", "noisy:vocal_gender",
    "rule:oracle", "rule:noisy", "flow:reject", "flow:skip", "flow:oracle", "flow:noisy",
)
# 표의 열 이름이 전제하는 깊이 — `hit10`(보여 준 10곡) · `candidate_rank@30`(후보 30곡).
REQUIRED_ARGS = {"top_k": 10, "candidate_k": 30}
# Gemini 호출 경로를 정하는 칸. 기준 실행과 다르면 공통 비교에서 뺀다.
GEMINI_PATH_KEYS = ("gemini.backend", "gemini.setting")
SCOPE_PER_RUN = "실행별 개입 대상"
SCOPE_COMMON = "공통 개입 대상"
SCOPE_EXCLUDED = "비교에서 뺌"


class Comparison(NamedTuple):
    lines: List[dict]                    # CSV 행
    common: Set[str]                     # 공통 개입 대상 질의(비교에 쓴 실행끼리)
    valid: List[str]                     # 비교에 쓴 실행 이름
    excluded: Dict[str, List[str]]       # 뺀 실행 → 사유
    intervention: Dict[str, Set[str]]    # 실행 → 개입 대상 질의
    statuses: Dict[str, str]             # 실행 → 리랭커 상태 문자열
    differences: Dict[str, List[str]]    # 실행 → 기준 실행과 다른 조건


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_depths(name: str, info: dict) -> None:
    run_args = info.get("args") or {}
    for key, wanted in REQUIRED_ARGS.items():
        if run_args.get(key) != wanted:
            raise ValueError(
                f"{name}: runinfo의 args.{key}가 {run_args.get(key)!r}다(필요한 값 {wanted}) — 이 표의 `hit10` · "
                f"`candidate_rank@30`이 다른 깊이를 뜻하게 된다. 같은 깊이로 잰 실행끼리 견준다."
            )


def summarize(runs: Sequence[Tuple[str, Path]]) -> Comparison:
    """실행별 표 + (비교에 쓸 수 있는 실행끼리의) 공통 개입 대상 표."""
    infos: Dict[str, dict] = {}
    details: Dict[str, List[dict]] = {}
    for name, folder in runs:
        infos[name] = json.loads((folder / "clarify_runinfo.json").read_text(encoding="utf-8"))
        _check_depths(name, infos[name])
        details[name] = _rows(folder / "clarify_detail.csv")
    base_name = runs[0][0]
    base_flat = flatten(infos[base_name])
    base_conditions = run_conditions([infos[base_name]])
    base_queries = {r["query_id"] for r in details[base_name]}

    lines: List[dict] = []
    intervention: Dict[str, Set[str]] = {}
    excluded: Dict[str, List[str]] = {}
    statuses_text: Dict[str, str] = {}
    differences: Dict[str, List[str]] = {}
    for name, folder in runs:
        info = infos[name]
        switches = info.get("ranking_switches", {})
        statuses = (info.get("clarify") or {}).get("rerank_status_counts") or {}
        failed = rerank_failures(statuses)
        statuses_text[name] = " ".join(f"{k}={v}" for k, v in sorted(statuses.items())) or "(집계 없음)"
        intervention[name] = {r["query_id"] for r in details[name] if r["policy"] == "reject_only" and r["ran"] == "1"}

        differences[name] = condition_differences(base_conditions, run_conditions([info]))
        measured = {r["query_id"] for r in details[name]}
        if measured != base_queries:
            differences[name].append(f"잰 질의: {len(base_queries)}건 → {len(measured)}건")

        reasons: List[str] = []
        if failed is None:
            reasons.append("리랭커 상태 집계 없음(모름)")
        elif failed:
            reasons.append(f"리랭킹 실패 {failed}건")
        flat = flatten(info)
        changed = [f"{key} {base_flat.get(key, '(없음)')} → {flat.get(key, '(없음)')}"
                   for key in GEMINI_PATH_KEYS if flat.get(key) != base_flat.get(key)]
        if changed:
            reasons.append("Gemini 호출 경로가 기준 실행과 다름(" + ", ".join(changed) + ")")
        if reasons:
            excluded[name] = reasons

        run_args = info.get("args") or {}
        versus = "(기준)" if name == base_name else "; ".join(differences[name]) or "같음"
        summary = {r["policy"]: r for r in _rows(folder / "clarify_summary.csv")}
        for policy in POLICIES:
            row = summary.get(policy)
            if row is None:
                continue
            lines.append({
                "run": name + ("!" if reasons else ""), "scope": SCOPE_PER_RUN, "policy": policy,
                "genre_match_scale": switches.get("genre_match_scale"), "genre_answer_scale": switches.get("genre_answer_scale"),
                "n_intervention": len(intervention[name]), "n_executed": row["n_executed"],
                "full_hit@10": row["full_hit@10"], "int_hit@10": row["int_hit@10"],
                "int_candidate_recall@30": row["int_candidate_recall@30"],
                "rerank_failed": "" if failed is None else failed, "rerank_statuses": statuses_text[name],
                "excluded_reason": " · ".join(reasons),
                "gemini_backend": (info.get("gemini") or {}).get("backend"),
                "top_k": run_args.get("top_k"), "candidate_k": run_args.get("candidate_k"),
                "answer_multiplier": run_args.get("answer_multiplier"), "query_ids": run_args.get("query_ids"),
                "analysis_cache": (info.get("analysis") or {}).get("path"), "n_measured_queries": len(measured),
                "conditions_vs_base": versus,
            })

    # 뺀 실행은 공통 대상을 정할 때도, 견줄 때도 쓰지 않는다.
    valid = [name for name, _ in runs if name not in excluded]
    common: Set[str] = set.intersection(*(intervention[name] for name in valid)) if valid else set()
    for name in valid:
        for policy in POLICIES:
            # 질의 안에서 먼저 평균한다(flow:*는 목표 곡마다 한 행이다).
            by_query: Dict[str, List[float]] = {}
            pool: Dict[str, List[float]] = {}
            ran: Set[str] = set()
            for r in details[name]:
                if r["policy"] != policy or r["query_id"] not in common:
                    continue
                by_query.setdefault(r["query_id"], []).append(float(r["hit10"]))
                pool.setdefault(r["query_id"], []).append(1.0 if str(r.get("candidate_rank@30", "")).strip() else 0.0)
                if r["ran"] == "1":
                    ran.add(r["query_id"])
            if not by_query:
                continue
            hits = [sum(v) / len(v) for v in by_query.values()]
            pools = [sum(v) / len(v) for v in pool.values()]
            lines.append({
                # n_executed는 하네스와 같은 뜻이다 — 그 정책이 실제로 돈(ran = 1) 질의 수. 평균은 행이 있는 질의 전부로 낸다.
                "run": name, "scope": SCOPE_COMMON, "policy": policy,
                "n_intervention": len(common), "n_executed": len(ran), "n_queries": len(by_query),
                "int_hit@10": round(sum(hits) / len(hits), 4), "hit_sum": round(sum(hits), 2),
                "int_candidate_recall@30": round(sum(pools) / len(pools), 4),
                "conditions_vs_base": "(기준)" if name == base_name else "; ".join(differences[name]) or "같음",
            })
    for name, _ in runs:
        if name in excluded:
            lines.append({
                "run": name + "!", "scope": SCOPE_EXCLUDED, "policy": "",
                "n_intervention": len(intervention[name]),
                "rerank_statuses": statuses_text[name], "excluded_reason": " · ".join(excluded[name]),
            })
    return Comparison(lines, common, valid, excluded, intervention, statuses_text, differences)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="이름=결과폴더 (첫 실행이 기준)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    runs = [(item.partition("=")[0], Path(item.partition("=")[2])) for item in args.runs]
    lines, common, valid, excluded, intervention, statuses, differences = summarize(runs)

    columns = list(dict.fromkeys(key for line in lines for key in line))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: line.get(c, "") for c in columns} for line in lines)

    names = [name for name, _ in runs]
    mark = {name: "!" if name in excluded else "" for name in names}
    print("실행별 개입 대상: " + " · ".join(f"{name}{mark[name]} {len(intervention[name])}" for name in names))
    for name in names:
        if name in excluded:
            print(f"비교에서 뺀 실행: {name} — {' · '.join(excluded[name])}")
    for name in names[1:]:
        print(f"기준 실행({names[0]})과 다른 조건: {name} — {'; '.join(differences[name]) or '없음'}")
    if not valid:
        print("비교에 쓸 수 있는 실행이 없다 — 공통 개입 대상 표를 만들지 않았다.")
    else:
        print(f"공통 개입 대상 {len(common)}건(비교에 쓴 실행 {len(valid)}개 기준): {', '.join(sorted(common))}")
        common_hits = {(line["run"], line["policy"]): line["hit_sum"] for line in lines if line["scope"] == SCOPE_COMMON}
        print(f"\n{'정책':<22}" + "".join(f"{name:>12}" for name in valid)
              + "   (공통 개입 대상에서 10위 안에 든 질의 수 — 질의 안 평균의 합)")
        for policy in POLICIES:
            cells = [f"{common_hits[(name, policy)]:>12}" if (name, policy) in common_hits else f"{'':>12}" for name in valid]
            print(f"{policy:<22}" + "".join(cells))
    print("\n리랭커 상태: " + " | ".join(f"{name}{mark[name]}: {statuses[name]}" for name in names))
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
