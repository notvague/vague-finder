"""v72 실행 전 무료 검증 · 부분집합 확정 — 모델·DB 호출 없음(저장된 가짜 리랭커 실행 기록만 읽는다).

입력: `offline/full/off_full_fake`(A) · `offline/full/work_full_fake`(B) — dev 176 전체, 같은 코드·서버·조건, 가짜 결정적 응답.
1. 재현: A 후보(id·순서·점수)·가사 보호 곡 = v71 `offline/off`, B = v71 `offline/live_rep1`(176/176이어야 한다).
2. A↔B 리랭커 입력 비교(질의 176건 각각): 후보 id·순서·점수, 가사 결과, 보호 그룹·tier·정책 생략 사유, 리랭커 그룹 입력,
   **실제 전송 프롬프트**(그룹별 첫 호출 sha16). 하나라도 다르면 "입력 변경".
3. 후보가 달라도 모델을 부르지 않는 질의(정책 생략·리랭커 대상 아님)를 따로 센다.
4. 입력 변경 질의 = 부분집합. v71 보고(채택 16건)와 대조한다. 채택인데 입력이 같은 질의는 원인(보조 곡이 A 후보 30에 없고
   경계를 넘은 곡도 0 → 후보 id·순서·점수가 그대로)이 기록으로 설명될 때만 통과한다. 설명되지 않는 차이 · 채택 아닌데 입력이 다른
   질의 · 재현 실패 · 입력이 바뀌었는데 모델을 부르지 않는 질의가 있으면 멈춘다(종료 코드 3).
   정답·모델 결과를 보지 않는다(정답 열은 읽지 않는다). 질의 id 목록만 파일로 남기고 바이트 지문을 적는다.

    venv/bin/python freeze_subset.py --v71 <v71 결과 폴더> [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_inputs as ci  # noqa: E402

SETS = ("v06", "v09", "v10")
SUBSET_FILE = HERE / "subset/rerank_input_changed.json"


def load_v71(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["qid"]] = {"set": r["set"], "cands": [(c["id"], round(float(c["score"]), 9)) for c in r["candidates"]],
                         "protected": sorted(r["protected_lyric_ids"]),
                         "work_status": (r.get("work_resolution") or {}).get("status")}
    return out


def explain_status(run_dir: Path) -> Dict[str, str]:
    out = {}
    for s in SETS:
        for line in (run_dir / f"{s}_dev/search_eval_dev_explain.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["query_id"]] = (r.get("work_resolution") or {}).get("status", "(none)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v71", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    a = ci.load_run(HERE / "offline/full/off_full_fake")
    b = ci.load_run(HERE / "offline/full/work_full_fake")
    if len(a["queries"]) != 176 or len(b["queries"]) != 176:
        raise SystemExit("dev 176이 아니다")
    v71 = {"off": load_v71(args.v71 / "offline/off/records.jsonl"),
           "live_rep1": load_v71(args.v71 / "offline/live_rep1/records.jsonl")}
    set_of = {q: v["set"] for q, v in v71["off"].items()}

    # 1) v71 재현 — 후보 id·순서·점수, 보호 곡
    repro: Dict[str, Any] = {}
    for name, run, ref in (("A_vs_v71_off", a, v71["off"]), ("B_vs_v71_live_rep1", b, v71["live_rep1"])):
        cand_diff, prot_diff = [], []
        for q, rv in ref.items():
            row = run["queries"][q]
            mine = [(c["id"], round(float(c["score"]), 9)) for c in row["candidates"]]
            if mine != rv["cands"]:
                cand_diff.append(q)
            if sorted((row["protection"] or {}).get("protected_ids") or []) != rv["protected"]:
                prot_diff.append(q)
        repro[name] = {"queries": len(ref), "candidates_identical": len(ref) - len(cand_diff), "candidate_diff": cand_diff,
                       "protected_identical": len(ref) - len(prot_diff), "protected_diff": prot_diff}

    # 2) A↔B 리랭커 입력
    cmp = ci.compare(a, b)
    changed = cmp["input_diff_queries"]
    status_b = explain_status(HERE / "offline/full/work_full_fake")
    status_a = explain_status(HERE / "offline/full/off_full_fake")
    applied = sorted(q for q, s in status_b.items() if s == "applied")
    v71_applied = sorted(q for q, v in v71["live_rep1"].items() if v["work_status"] == "applied")

    # 3) 후보가 다른데 모델을 부르지 않는 질의 · 입력 변경 질의의 정책·호출 상태
    no_call_changed, per_query = [], {}
    for q in changed:
        ra, rb = a["queries"][q], b["queries"][q]
        pa, pb = ra["protection"] or {}, rb["protection"] or {}
        groups_b = len(rb["rerank_runs"])
        calls_b = len(rb["generates"])
        per_query[q] = {"set": set_of[q], "fields": cmp["per_query"].get(q, {}).get("fields", []),
                        "work_status_b": status_b[q], "policy_skip_a": pa.get("policy_skip", ""),
                        "policy_skip_b": pb.get("policy_skip", ""), "eligible_b": pb.get("eligible"),
                        "groups_a": len(ra["rerank_runs"]), "groups_b": groups_b,
                        "generate_calls_a": len(ra["generates"]), "generate_calls_b": calls_b,
                        "prompt_chars_b": [g["contents_chars"] for g in rb["generates"]],
                        "protected_b": len(pb.get("protected_ids") or [])}
        if calls_b == 0:
            no_call_changed.append(q)
    policy_skipped_all = sorted(q for q, r in b["queries"].items() if (r["protection"] or {}).get("policy_skip"))
    unchanged = sorted(set(b["queries"]) - set(changed))
    # 입력이 같은 질의는 상태도 같아야 한다(작품명 판정 밖·보류·거부)
    unchanged_status_b = {}
    for q in unchanged:
        unchanged_status_b[status_b[q]] = unchanged_status_b.get(status_b[q], 0) + 1

    # 채택인데 입력이 같은 질의: 보조 곡이 A 후보 30에 하나도 없고 들어온 곡도 0이어야 설명된다
    work_b = {}
    for s in SETS:
        for line in (HERE / f"offline/full/work_full_fake/{s}_dev/search_eval_dev_explain.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            work_b[r["query_id"]] = r.get("work_resolution") or {}
    applied_same = {}
    for q in sorted(set(applied) - set(changed)):
        w = work_b[q]
        a_ids = {c["id"] for c in a["queries"][q]["candidates"]}
        applied_same[q] = {"aux_songs": len(w.get("aux_ids") or []), "aux_in_A_top30": sum(1 for x in w.get("aux_ids") or [] if x in a_ids),
                           "entered": len(w.get("entered_ids") or []), "displaced": len(w.get("displaced_ids") or [])}
    explained = all(v["aux_in_A_top30"] == 0 and v["entered"] == 0 and v["displaced"] == 0 for v in applied_same.values())
    ok = (all(r["candidates_identical"] == 176 and r["protected_identical"] == 176 for r in repro.values())
          and set(changed) <= set(applied) and applied == v71_applied and explained and not no_call_changed
          and not (set(changed) - set(applied)))
    subset = {"query_ids_by_set": {s: [q for q in sorted(changed) if set_of[q] == s] for s in SETS}}
    total_chars = sum(sum(v["prompt_chars_b"]) for v in per_query.values())
    total_chars_a = sum(sum(g["contents_chars"] for g in a["queries"][q]["generates"]) for q in changed)
    report = {
        "repro": repro,
        "ab_compare": {k: cmp[k] for k in ("queries", "input_identical", "diff_counts", "cand_set_diff_queries",
                                            "path_diff_query_ids", "retry_prompt_changed_queries")},
        "input_changed": changed, "input_changed_count": len(changed),
        "work_applied_b": applied, "v71_applied": v71_applied,
        "changed_equals_applied_equals_v71": changed == applied == v71_applied,
        "applied_equals_v71_applied": applied == v71_applied,
        "applied_but_input_identical": applied_same, "applied_but_input_identical_explained": explained,
        "changed_but_no_model_call": no_call_changed,
        "policy_skipped_any_b": policy_skipped_all,
        "unchanged_count": len(unchanged), "unchanged_work_status_b": unchanged_status_b,
        "work_status_a": sorted(set(status_a.values())),
        "per_query": per_query,
        "subset_generate_calls": {"A": sum(v["generate_calls_a"] for v in per_query.values()),
                                  "B": sum(v["generate_calls_b"] for v in per_query.values())},
        "subset_prompt_chars": {"A": total_chars_a, "B": total_chars},
        "ok": ok,
    }
    out_dir = HERE / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "precheck_full.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("repro", "input_changed_count", "changed_equals_applied_equals_v71",
                                             "applied_equals_v71_applied", "applied_but_input_identical",
                                             "changed_but_no_model_call", "policy_skipped_any_b", "unchanged_work_status_b",
                                             "work_status_a", "subset_generate_calls", "subset_prompt_chars", "ok")},
                     ensure_ascii=False, indent=1))
    if not ok:
        print("사전 검증 불일치 — 부분집합을 확정하지 않는다", file=sys.stderr)
        return 3
    if args.write:
        if SUBSET_FILE.exists():
            raise SystemExit(f"{SUBSET_FILE}가 이미 있다 — 덮어쓰지 않는다")
        SUBSET_FILE.parent.mkdir(exist_ok=True)
        body = {**subset, "count": len(changed),
                "rule": "A(off)↔B(work, v71 rep1 캐시) 가짜 리랭커 dev 176 실행에서 후보·가사·보호·그룹 입력·전송 프롬프트 중 하나라도 다른 질의",
                "source_runs": {"A": "offline/full/off_full_fake", "B": "offline/full/work_full_fake"},
                "precheck_sha16": hashlib.sha256((out_dir / "precheck_full.json").read_bytes()).hexdigest()[:16]}
        SUBSET_FILE.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("부분집합", SUBSET_FILE.name, hashlib.sha256(SUBSET_FILE.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
