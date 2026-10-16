"""v61 — 서비스 후보 30개(리랭킹 전) 단계의 A/B/C 비교. 모델·DB 없음(저장 파일만 읽는다).

입력: `<루트>/cand_<arm>/<set>_dev/search_eval_dev_detail.csv`(평가기 산출) + `candidates.jsonl`(후보 30개 전체, run_candidates가 기록)
  기본은 이 폴더의 A(기준) · B(비교) · C(참고). **수정 코드 검증**에서는 기존 기준선을 보존한 채 새 결과 폴더를 따로 받는다:
    venv/bin/python compare_candidates.py --baseline-root <이 폴더> --baseline-arm A \
        --compare-root /private/tmp/vague-finder-hybrid-scan-validation --compare-arm A --no-reference --suffix fixed
  → `candidate_changes_fixed.csv` · `candidate_summary_fixed.json` (열 이름의 A=기준선, B=비교 대상)
출력:
  - `candidate_changes.csv` — 질의별: 후보 지문 A=B? · 정답의 후보 내 순위 A/B · 리랭킹 전 Hit@1/5/10·MRR@10 A/B · 새로 회수/빠진 정답
    · 후보 30개 중 바뀐 id 수 · 가사 표면 경로 대상 여부(Mongo 차단 조건에서 서비스와 다른 질의)
  - `candidate_summary.json` — 세트별·전체 집계(분모 명시), A rep1 vs rep2 반복 재현성, C=B 여부
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
SETS = ("v06", "v09", "v10")
K = 30


def _detail(root: Path, arm: str, set_name: str) -> Dict[str, Dict[str, str]]:
    path = root / f"cand_{arm}" / f"{set_name}_dev" / "search_eval_dev_detail.csv"
    with path.open(encoding="utf-8-sig") as fh:
        return {row["query_id"]: row for row in csv.DictReader(fh)}


def _candidates(root: Path, arm: str, set_name: str) -> Dict[str, List[str]]:
    path = root / f"cand_{arm}" / f"{set_name}_dev" / "candidates.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["query_id"]] = row["candidate_ids"]
    return out


def _conditions(root: Path, arm: str, set_name: str) -> Dict[str, Any]:
    path = root / f"cand_{arm}" / f"{set_name}_dev" / "conditions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _int(value: str) -> Optional[int]:
    return int(value) if value not in ("", None) else None


def _f(value: str) -> float:
    return float(value or 0.0)


def _lyric_surface(row: Dict[str, str]) -> bool:
    try:
        clues = json.loads(row.get("lyric_clues") or "[]")
    except json.JSONDecodeError:
        return False
    return any(c.get("kind") in {"verbatim", "partial", "phonetic"} for c in clues)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default=str(HERE), help="기준선(열 이름 A) 실행이 있는 루트")
    parser.add_argument("--baseline-arm", default="A")
    parser.add_argument("--compare-root", default=None, help="비교 대상(열 이름 B) 실행이 있는 루트(기본: 기준선 루트)")
    parser.add_argument("--compare-arm", default="B")
    parser.add_argument("--reference-root", default=None, help="참고(열 이름 C) 실행 루트(기본: 기준선 루트)")
    parser.add_argument("--reference-arm", default="C")
    parser.add_argument("--no-reference", action="store_true", help="C 열을 만들지 않는다")
    parser.add_argument("--suffix", default="", help="출력 파일 이름 접미(예: fixed → candidate_changes_fixed.csv)")
    args = parser.parse_args()
    base_root = Path(args.baseline_root)
    cmp_root = Path(args.compare_root) if args.compare_root else base_root
    ref_root = Path(args.reference_root) if args.reference_root else base_root
    suffix = f"_{args.suffix}" if args.suffix else ""

    rows_out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"sets": {}, "all": {}, "repeatability": {}, "conditions": {},
                               "inputs": {"A": f"{base_root}/cand_{args.baseline_arm}", "B": f"{cmp_root}/cand_{args.compare_arm}",
                                          "C": None if args.no_reference else f"{ref_root}/cand_{args.reference_arm}"}}
    totals = {"queries": 0, "digest_changed_AB": 0, "digest_changed_BC": 0, "target_in30_A": 0, "target_in30_B": 0,
              "gained": 0, "lost": 0, "hit1_A": 0, "hit1_B": 0, "hit5_A": 0, "hit5_B": 0, "hit10_A": 0, "hit10_B": 0,
              "mrr_A": 0.0, "mrr_B": 0.0, "lyric_surface_queries": 0}
    for set_name in SETS:
        A, B = _detail(base_root, args.baseline_arm, set_name), _detail(cmp_root, args.compare_arm, set_name)
        C: Dict[str, Dict[str, str]] = {}
        if not args.no_reference:
            try:
                C = _detail(ref_root, args.reference_arm, set_name)
            except FileNotFoundError:
                C = {}
        cand_A, cand_B = _candidates(base_root, args.baseline_arm, set_name), _candidates(cmp_root, args.compare_arm, set_name)
        cand_C = _candidates(ref_root, args.reference_arm, set_name) if C else {}
        if set(A) != set(B):
            raise RuntimeError(f"{set_name}: A·B 질의 집합이 다르다")
        s = {k: 0 for k in totals if k != "mrr_A" and k != "mrr_B"}
        s["mrr_A"] = 0.0
        s["mrr_B"] = 0.0
        for qid in A:
            a, b = A[qid], B[qid]
            c = C.get(qid)
            relevant = set(filter(None, a["relevant_ids"].split("|")))
            ca, cb = cand_A.get(qid, []), cand_B.get(qid, [])
            in_a, in_b = relevant & set(ca), relevant & set(cb)
            gained, lost = sorted(in_b - in_a), sorted(in_a - in_b)
            digest_same_ab = a["candidate_digest"] == b["candidate_digest"]
            digest_same_bc = (c is not None) and (b["candidate_digest"] == c["candidate_digest"])
            changed_ids = len(set(ca) ^ set(cb)) if ca and cb else None
            first_pos_diff = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x != y), None) if ca and cb else None
            rank_a, rank_b = _int(a[f"candidate_rank@{K}"]), _int(b[f"candidate_rank@{K}"])
            row = {
                "set": set_name, "query_id": qid, "relevant_ids": a["relevant_ids"],
                "lyric_surface_query": int(_lyric_surface(a)),
                "digest_same_A_B": int(digest_same_ab), "digest_same_B_C": int(digest_same_bc) if c is not None else "",
                "changed_ids_in_30": changed_ids if changed_ids is not None else "",
                "first_position_diff": (first_pos_diff + 1) if first_pos_diff is not None else "",
                "candidate_rank_A": rank_a if rank_a else "", "candidate_rank_B": rank_b if rank_b else "",
                "target_in30_A": int(bool(in_a)), "target_in30_B": int(bool(in_b)),
                "gained_targets": "|".join(gained), "lost_targets": "|".join(lost),
                "hit1_A": a["baseline_hit@1"], "hit1_B": b["baseline_hit@1"],
                "hit5_A": a["baseline_hit@5"], "hit5_B": b["baseline_hit@5"],
                "hit10_A": a["baseline_hit@10"], "hit10_B": b["baseline_hit@10"],
                "mrr10_A": a["baseline_mrr@10"], "mrr10_B": b["baseline_mrr@10"],
                "retrieval_ms_A": a["retrieval_ms"], "retrieval_ms_B": b["retrieval_ms"],
                "retrieval_ms_C": c["retrieval_ms"] if c else "",
            }
            rows_out.append(row)
            s["queries"] += 1
            s["digest_changed_AB"] += int(not digest_same_ab)
            s["digest_changed_BC"] += int(c is not None and not digest_same_bc)
            s["target_in30_A"] += int(bool(in_a))
            s["target_in30_B"] += int(bool(in_b))
            s["gained"] += int(bool(gained))
            s["lost"] += int(bool(lost))
            for k in ("hit1", "hit5", "hit10"):
                s[f"{k}_A"] += int(_f(a[f"baseline_{k.replace('hit', 'hit@')}"]))
                s[f"{k}_B"] += int(_f(b[f"baseline_{k.replace('hit', 'hit@')}"]))
            s["mrr_A"] += _f(a["baseline_mrr@10"])
            s["mrr_B"] += _f(b["baseline_mrr@10"])
            s["lyric_surface_queries"] += int(_lyric_surface(a))
        n = s["queries"]
        summary["sets"][set_name] = {
            **s,
            "candidate_recall30_A": round(s["target_in30_A"] / n, 6), "candidate_recall30_B": round(s["target_in30_B"] / n, 6),
            "hit1_A_rate": round(s["hit1_A"] / n, 6), "hit1_B_rate": round(s["hit1_B"] / n, 6),
            "hit5_A_rate": round(s["hit5_A"] / n, 6), "hit5_B_rate": round(s["hit5_B"] / n, 6),
            "hit10_A_rate": round(s["hit10_A"] / n, 6), "hit10_B_rate": round(s["hit10_B"] / n, 6),
            "mrr10_A": round(s["mrr_A"] / n, 6), "mrr10_B": round(s["mrr_B"] / n, 6),
            "denominator": f"dev {set_name} 질의 전부({n}건, 모두 정답 있음)",
        }
        for k in totals:
            totals[k] += s[k]
        summary["conditions"][set_name] = {
            label: {k: v for k, v in _conditions(root, arm, set_name).items()
                    if k in ("point_count", "hybrid_scan_limit_used", "scan_limit_truncates", "reference_fingerprint", "capture",
                             "eval_seconds", "ru_maxrss_bytes", "guard_after", "git_sha")}
            for label, root, arm in (("A", base_root, args.baseline_arm), ("B", cmp_root, args.compare_arm), ("C", ref_root, args.reference_arm))
            if not (label == "C" and args.no_reference)
        }
    n = totals["queries"]
    summary["all"] = {
        **totals,
        "candidate_recall30_A": round(totals["target_in30_A"] / n, 6), "candidate_recall30_B": round(totals["target_in30_B"] / n, 6),
        "hit1_A_rate": round(totals["hit1_A"] / n, 6), "hit1_B_rate": round(totals["hit1_B"] / n, 6),
        "hit5_A_rate": round(totals["hit5_A"] / n, 6), "hit5_B_rate": round(totals["hit5_B"] / n, 6),
        "hit10_A_rate": round(totals["hit10_A"] / n, 6), "hit10_B_rate": round(totals["hit10_B"] / n, 6),
        "mrr10_A": round(totals["mrr_A"] / n, 6), "mrr10_B": round(totals["mrr_B"] / n, 6),
        "denominator": f"dev 176건 전부({n}건) — 리랭킹 전 후보·순서 지표, 최종 서비스 Hit@10이 아님",
    }
    # 반복 재현성 — A rep2가 있으면 지문 비교
    rep_dir = base_root / f"cand_{args.baseline_arm}_rep2"
    if rep_dir.exists():
        for set_name in SETS:
            path = rep_dir / f"{set_name}_dev" / "search_eval_dev_detail.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig") as fh:
                rep = {row["query_id"]: row for row in csv.DictReader(fh)}
            base = _detail(base_root, args.baseline_arm, set_name)
            same = sum(1 for qid in base if qid in rep and base[qid]["candidate_digest"] == rep[qid]["candidate_digest"])
            summary["repeatability"][set_name] = {"queries": len(base), "digest_same_rep1_rep2": same,
                                                  "differing": sorted(q for q in base if q in rep and base[q]["candidate_digest"] != rep[q]["candidate_digest"])}
    fieldnames = list(rows_out[0].keys())
    with (HERE / f"candidate_changes{suffix}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    (HERE / f"candidate_summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"all": summary["all"], "repeatability": summary["repeatability"]}, ensure_ascii=False, indent=1))
    changed = [r for r in rows_out if not r["digest_same_A_B"]]
    print(f"\n후보 지문이 달라진 질의 {len(changed)}건:")
    for r in changed:
        print(f"  {r['set']} {r['query_id']}: 바뀐 id {r['changed_ids_in_30']} · 첫 차이 위치 {r['first_position_diff']} · 정답 순위 A {r['candidate_rank_A'] or '-'} → B {r['candidate_rank_B'] or '-'}"
              + (f" · 새로 회수 {r['gained_targets']}" if r['gained_targets'] else "") + (f" · 빠짐 {r['lost_targets']}" if r['lost_targets'] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
