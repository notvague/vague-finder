"""v73 실행 전 무료 검증 — 모델·DB 호출 없음(저장된 가짜 리랭커 실행 기록만 읽는다). v72 `freeze_subset.py`를 전체 A/B용으로 고친 판.

입력: `offline/full/off_full_warm`(A) · `offline/full/work_full_warm`(B) — v73 코드 사본·서버·조건, 예열 포함, 가짜 결정적 응답.
1. v72 오프라인 기준 재현: v72 `offline/full/{off,work}_full_warm`(예열 있음)·`{off,work}_full_fake`(예열 없음)과 질의 176건의
   후보·가사·보호·그룹 입력·전송 프롬프트·최종 순서·Qdrant 경로 반환이 모두 같아야 한다(`compare_inputs.compare`).
2. v71 재현: A 후보(id·순서·점수)·가사 보호 곡 = v71 `offline/off`, B = v71 `offline/live_rep1`(176/176).
3. A↔B 리랭커 입력이 다른 질의 = v72 고정 부분집합 13건(`inputs/v72_rerank_input_changed.json`)과 같아야 한다. 나머지 163건의
   구성(판정 밖 154 · 보류 5 · 검증 거부 1 · 채택이지만 입력 같음 3)을 다시 센다. 입력이 바뀌었는데 모델을 부르지 않는 질의는 0이어야 한다.
4. 후보 회수@30(리랭킹 전, 정답 id 사용): A 122 · B 125, B에서 새로 든 질의 n033·n043·n096, 잃은 질의 0이어야 한다.
5. 질의별 기준 작품명 해석 상태 파일 `inputs/expected_work_status.json`을 쓴다(질의 id → 상태, 작품명·원문 없음). live·직전 지문
   실행이 이 파일로 질의마다 상태를 대조한다. 이미 있으면 같은 내용인지만 본다(덮어쓰지 않는다).
6. 비용 계산용: 구성별 그룹 수·생성 호출 수(가짜 실행)·프롬프트 글자 수·정책 생략 수.
하나라도 어긋나면 종료 코드 3(유료 실행을 시작하지 않는다).

    venv/bin/python precheck_v73.py --v72 <v72 결과 폴더> --v71 <v71 결과 폴더>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import compare_inputs as ci  # noqa: E402

SETS = ("v06", "v09", "v10")
STATUS_FILE = HERE / "inputs/expected_work_status.json"
V72_SUBSET = HERE / "inputs/v72_rerank_input_changed.json"
V72_SUBSET_SHA16 = "7004bca4e3f7deb1"
EXPECT_RECALL = {"A": 122, "B": 125, "gained": ["n033", "n043", "n096"], "lost": []}
EXPECT_STATUS_B = {"not_triggered": 154, "applied": 16, "abstained": 5, "no_usable_work": 1}
APPLIED_SAME = ("n045", "q105", "t024")


def gold() -> Dict[str, set]:
    out = {}
    for s in SETS:
        with (ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    out[r["query_id"].strip()] = {p.strip() for p in (r["relevant_ids"] or "").split("|") if p.strip()}
    return out


def explain(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for s in SETS:
        for line in (run_dir / f"{s}_dev/search_eval_dev_explain.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["query_id"]] = r.get("work_resolution") or {}
    return out


def load_v71(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["qid"]] = {"cands": [(c["id"], round(float(c["score"]), 9)) for c in r["candidates"]],
                         "protected": sorted(r["protected_lyric_ids"])}
    return out


def same(cmp: Dict[str, Any]) -> bool:
    return (cmp["input_identical"] == 176 and not cmp["missing_either"] and not any(cmp["diff_counts"].values())
            and not cmp["path_diff_query_ids"] and not cmp["retry_prompt_changed_queries"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v72", type=Path, required=True)
    ap.add_argument("--v71", type=Path, required=True)
    args = ap.parse_args()
    runs = {c: ci.load_run(HERE / f"offline/full/{c}_full_warm") for c in ("off", "work")}
    a, b = runs["off"], runs["work"]
    problems = []
    if len(a["queries"]) != 176 or len(b["queries"]) != 176:
        problems.append("dev 176이 아니다")

    # 1) v72 오프라인 기준
    v72 = {}
    for c in ("off", "work"):
        for ref in ("full_warm", "full_fake"):
            cmp = ci.compare(runs[c], ci.load_run(args.v72 / f"offline/full/{c}_{ref}"))
            v72[f"{c}_vs_v72_{c}_{ref}"] = {"input_identical": cmp["input_identical"], "diff_counts": cmp["diff_counts"],
                                            "path_diff": {k: len(v) for k, v in cmp["path_diff_query_ids"].items()},
                                            "fields_compared": cmp["fields_compared"], "identical_all": same(cmp)}
            if not same(cmp):
                problems.append(f"v72 {c}_{ref}와 다르다")

    # 2) v71 재현
    v71 = {"off": load_v71(args.v71 / "offline/off/records.jsonl"), "work": load_v71(args.v71 / "offline/live_rep1/records.jsonl")}
    repro = {}
    for c, run in runs.items():
        cand = [q for q, rv in v71[c].items()
                if [(x["id"], round(float(x["score"]), 9)) for x in run["queries"][q]["candidates"]] != rv["cands"]]
        prot = [q for q, rv in v71[c].items()
                if sorted((run["queries"][q]["protection"] or {}).get("protected_ids") or []) != rv["protected"]]
        repro[c] = {"queries": len(v71[c]), "candidates_identical": len(v71[c]) - len(cand), "protected_identical": len(v71[c]) - len(prot)}
        if cand or prot or len(v71[c]) != 176:
            problems.append(f"v71 재현 실패 {c}: 후보 {cand[:5]} 보호 {prot[:5]}")

    # 3) 입력 변경 13 / 동일 163
    raw = V72_SUBSET.read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != V72_SUBSET_SHA16:
        problems.append("v72 부분집합 파일 지문이 다르다")
    v72_changed = sorted(q for ids in json.loads(raw)["query_ids_by_set"].values() for q in ids)
    cmp_ab = ci.compare(a, b)
    changed = cmp_ab["input_diff_queries"]
    if changed != v72_changed:
        problems.append(f"입력 변경 질의가 v72와 다르다: {changed}")
    work_b = explain(HERE / "offline/full/work_full_warm")
    work_a = explain(HERE / "offline/full/off_full_warm")
    status_b = {q: w.get("status", "(none)") for q, w in work_b.items()}
    counts_b: Dict[str, int] = {}
    for st in status_b.values():
        counts_b[st] = counts_b.get(st, 0) + 1
    if counts_b != EXPECT_STATUS_B:
        problems.append(f"B 해석 상태 분포 {counts_b}")
    if any(work_a.values()):
        problems.append("A에 작품명 해석 기록이 있다")
    unchanged = sorted(set(b["queries"]) - set(changed))
    composition: Dict[str, int] = {}
    for q in unchanged:
        key = status_b[q] if status_b[q] != "applied" else "applied_input_identical"
        composition[key] = composition.get(key, 0) + 1
    applied_same = sorted(q for q in unchanged if status_b[q] == "applied")
    if applied_same != sorted(APPLIED_SAME):
        problems.append(f"채택인데 입력 같은 질의 {applied_same}")
    no_call_changed = [q for q in changed if not b["queries"][q]["generates"] or not a["queries"][q]["generates"]]
    if no_call_changed:
        problems.append(f"입력이 바뀌었는데 모델을 부르지 않는 질의 {no_call_changed}")

    # 4) 후보 회수@30
    g = gold()
    hit = {c: sorted(q for q, r in run["queries"].items() if {x["id"] for x in r["candidates"]} & g[q]) for c, run in runs.items()}
    recall = {"A": len(hit["off"]), "B": len(hit["work"]), "gained": sorted(set(hit["work"]) - set(hit["off"])),
              "lost": sorted(set(hit["off"]) - set(hit["work"]))}
    if recall != EXPECT_RECALL:
        problems.append(f"후보 회수 {recall}")

    # 6) 비용 계산용 · 정책 생략
    calls = {}
    for c, run in runs.items():
        rows = run["queries"].values()
        calls[c] = {"groups": sum(len(r["rerank_runs"]) for r in rows), "generate_calls_fake": sum(len(r["generates"]) for r in rows),
                    "queries_with_model_call": sum(1 for r in rows if r["generates"]),
                    "policy_skipped": sum(1 for r in rows if (r["protection"] or {}).get("policy_skip")),
                    "not_eligible": sum(1 for r in rows if (r["protection"] or {}).get("eligible") is False),
                    "prompt_chars_total": sum(x["contents_chars"] for r in rows for x in r["generates"]),
                    "prompt_chars_max": max((x["contents_chars"] for r in rows for x in r["generates"]), default=0),
                    "max_groups_per_query": max((len(r["rerank_runs"]) for r in rows), default=0)}
    policy_skip_changed = [q for q in changed if (b["queries"][q]["protection"] or {}).get("policy_skip")]

    # 5) 기준 해석 상태 파일
    body = {"status_by_config": {"off": {q: "(none)" for q in sorted(a["queries"])}, "work": dict(sorted(status_b.items()))},
            "source_runs": {"off": "offline/full/off_full_warm", "work": "offline/full/work_full_warm"},
            "rule": "가짜 리랭커 dev 176 실행(예열 포함)의 질의별 작품명 해석 상태. live·직전 지문 실행이 질의마다 이 값과 같아야 한다"}
    text = json.dumps(body, ensure_ascii=False, indent=1) + "\n"
    report = {"v72_offline_reproduction": v72, "v71_reproduction": repro,
              "ab_input": {"input_identical": cmp_ab["input_identical"], "diff_counts": cmp_ab["diff_counts"],
                           "cand_set_diff_queries": cmp_ab["cand_set_diff_queries"],
                           "path_diff_queries": {k: v for k, v in cmp_ab["path_diff_query_ids"].items()}},
              "input_changed": changed, "input_changed_count": len(changed), "equals_v72_subset": changed == v72_changed,
              "unchanged_count": len(unchanged), "unchanged_composition_b": composition, "applied_but_input_identical": applied_same,
              "work_status_counts_b": counts_b, "changed_but_no_model_call": no_call_changed,
              "policy_skipped_in_changed": policy_skip_changed, "candidate_recall30": recall, "calls_fake": calls,
              "problems": problems, "ok": not problems}
    out_dir = HERE / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "precheck_v73.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("v72_offline_reproduction",)}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v["identical_all"] for k, v in v72.items()}))
    if problems:
        print("사전 검증 불일치 — 유료 실행을 시작하지 않는다", file=sys.stderr)
        return 3
    if STATUS_FILE.exists():
        if STATUS_FILE.read_text(encoding="utf-8") != text:
            print(f"{STATUS_FILE.name}가 이미 있고 내용이 다르다 — 덮어쓰지 않는다", file=sys.stderr)
            return 3
    else:
        STATUS_FILE.write_text(text, encoding="utf-8")
    print("기준 해석 상태", STATUS_FILE.name, hashlib.sha256(STATUS_FILE.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
