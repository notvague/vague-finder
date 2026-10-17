"""v68(v66 이식) 실제 모델 평가 집계 — 저장된 **최종 top-10**으로 엄격 정답 지표를 계산하고, 실행 쌍(전↔후·같은 구성 반복)의
질의별 이득·손실을 원인별로 나눈다. 모델·DB 호출 없음.

지표(완료된 실행만, 분모 = 그 부분집합의 전체 질의 수 — **실패·부분 적용·예산 소진의 폴백 결과도 포함**):
  Hit@1/5/10   최종 top-10 안 첫 정답 위치 ≤ k인 질의 수(정답이 여럿이면 하나라도)
  MRR@10       Σ(1/첫 정답 위치, 10 밖이면 0) / 질의 수
  후보 회수@30  리랭킹 전 후보 30에 정답이 하나라도 있는 질의 수
  부분집합      전체 176 · 가사 19(v63 `LYRIC_QUERY_IDS`) · 나머지 157
상태(질의 단위, 평가기 `rerank_outcome`과 그룹 상태에서): applied · partial(일부 그룹 실패·예산 소진) · failed(모델 반영 0 —
  fallback·exception) · policy_skipped · skipped · budget_exhausted(그룹 하나라도 예산 소진 — 위 분류와 겹칠 수 있어 따로 셈)
보조: 실패·부분·예산 소진 질의를 뺀 지표(분모가 줄어든 것을 함께 적는다). 기본 품질로 쓰지 않는다.

검사(하나라도 어긋나면 종료 코드 3, 결과를 쓰지 않는다): DONE·완료 상태·부분 실행 아님·live 실행·세트별 건수·질의 ID 중복/누락,
실행 조건 일치(코드 판별 커밋·환경·색인·가사·캐시·서버 기준 상태), 평가기 detail의 rerank_top_ids = 기록된 최종 순서 앞 10,
평가기 detail의 rerank_rank와 이 집계의 첫 정답 위치 일치, 세트별 평가기 summary의 Hit/MRR과 이 집계의 세트별 값 일치.

원인 분류(쌍 X↔Y, 질의마다 — 최종 top-10 순서가 다를 때):
  failure_fallback   한쪽이라도 failed·partial·예산 소진 그룹
  policy_skip_order  양쪽 모두 정책 생략(모델 없음) — 순서 차이는 후보 입력 차이에서만 온다
  input_change       후보·가사·보호·그룹 입력·첫 호출 프롬프트 중 하나가 다르다(세부: set/order/protection/prompt)
  model_variation    입력이 같은데 모델 출력(최종 순서)이 다르다

    venv/bin/python aggregate_live.py --runs B1=<dir>,A1=<dir>,A2=<dir>,B2=<dir> --pairs B1:A1,B2:A2,B1:B2,A1:A2 --out <폴더>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from compare_inputs import first_prompts  # noqa: E402

SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}
LYRIC_QUERY_IDS = (
    "q209", "q211", "q213", "q219", "n007", "n009", "n013", "n017", "n039", "n064",
    "t003", "t016", "t018", "t025", "t032", "t060", "t063", "t105", "t110",
)
FAILED_OUTCOMES = {"fallback", "exception"}
AUDIO_SCALE = {"on": "1", "skip": "0"}
SAME_CONDITION_KEYS = ("env", "index_fingerprint", "lyrics_snapshot", "analysis_caches", "collections")


class ValidationError(Exception):
    pass


def parse_ids(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split("|") if x.strip()]


def first_rank(ids: List[str], gold: set, k: int) -> Optional[int]:
    for i, sid in enumerate(ids[:k], start=1):
        if sid in gold:
            return i
    return None


def expected_query_ids(sets_root: Path = ROOT / "experiments/reranking") -> Dict[str, List[str]]:
    out = {}
    for s in SETS:
        with (sets_root / f"eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            out[s] = [r["query_id"].strip() for r in csv.DictReader(fh) if r["split"].strip() == "dev"]
    return out


def load_run(run_dir: Path, *, expected_ids: Dict[str, List[str]], expected_counts: Dict[str, int],
             require_live: bool = True) -> Dict[str, Any]:
    if not (run_dir / "DONE").exists():
        raise ValidationError(f"{run_dir.name}: 완료 표식 DONE 없음(미완료 실행)")
    marker = json.loads((run_dir / "DONE").read_text(encoding="utf-8"))
    cond = json.loads((run_dir / "run_conditions.json").read_text(encoding="utf-8"))
    if marker.get("partial") or cond.get("partial"):
        raise ValidationError(f"{run_dir.name}: 부분 실행")
    if cond.get("status") != "complete":
        raise ValidationError(f"{run_dir.name}: 상태 {cond.get('status')}")
    if require_live and (cond.get("model") != "live" or marker.get("model") != "live"):
        raise ValidationError(f"{run_dir.name}: live 실행이 아니다")
    if not cond.get("server_state_before_matches_baseline") or not cond.get("server_state_after_matches_baseline"):
        raise ValidationError(f"{run_dir.name}: 서버 상태가 기준과 다르다")
    rows: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for s in SETS:
        d = run_dir / f"{s}_dev"
        with (d / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
            detail = list(csv.DictReader(fh))
        rec = {}
        for line in (d / "v68_queries.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["query_id"] in rec:
                raise ValidationError(f"{run_dir.name}/{s}: 기록 질의 ID 중복 {r['query_id']}")
            rec[r["query_id"]] = r
        ids = [r["query_id"] for r in detail]
        if len(ids) != expected_counts[s]:
            raise ValidationError(f"{run_dir.name}/{s}: {len(ids)}건 ≠ {expected_counts[s]}")
        if len(set(ids)) != len(ids):
            raise ValidationError(f"{run_dir.name}/{s}: detail 질의 ID 중복")
        if ids != expected_ids[s]:
            raise ValidationError(f"{run_dir.name}/{s}: 질의 목록이 평가 세트와 다르다(누락·추가·순서)")
        if set(rec) != set(ids):
            raise ValidationError(f"{run_dir.name}/{s}: 하네스 기록과 detail의 질의 집합이 다르다")
        for r in detail:
            qid = r["query_id"]
            if qid in rows:
                raise ValidationError(f"{run_dir.name}: 세트 사이 질의 ID 중복 {qid}")
            h = rec[qid]
            final10 = parse_ids(r["rerank_top_ids"])
            recorded = ((h.get("final") or {}).get("ids") or [])[:10]
            if final10 != recorded:
                raise ValidationError(f"{run_dir.name}/{qid}: detail top-10 ≠ 기록된 최종 순서")
            gold = set(parse_ids(r["relevant_ids"]))
            if not gold:
                raise ValidationError(f"{run_dir.name}/{qid}: 정답 없음")
            rank = first_rank(final10, gold, 10)
            ev_rank = r.get("rerank_rank", "")
            if ev_rank not in ("", None) and int(ev_rank) <= 10 and int(ev_rank) != rank:
                raise ValidationError(f"{run_dir.name}/{qid}: 평가기 rerank_rank {ev_rank} ≠ {rank}")
            if rank is not None and (ev_rank in ("", None) or int(ev_rank) != rank):
                raise ValidationError(f"{run_dir.name}/{qid}: 평가기 rerank_rank {ev_rank!r} ≠ {rank}")
            cands = [c["id"] for c in (h.get("candidates") or [])]
            if len(cands) != 30:
                raise ValidationError(f"{run_dir.name}/{qid}: 후보 {len(cands)}개 ≠ 30")
            groups = [g.get("status") for g in h.get("rerank_runs", [])]
            rows[qid] = {
                "set": s, "gold": sorted(gold), "final10": final10, "rank": rank,
                "cand_rank": first_rank(cands, gold, 30), "outcome": r["rerank_outcome"],
                "group_statuses": groups, "policy_skip": r.get("rerank_policy_skip", ""),
                "error": r.get("rerank_error", ""), "harness": h,
            }
            order.append(qid)
        # 세트별 평가기 summary와 교차 확인(평가기 Hit/MRR은 비율 — 같은 분모로 되돌려 비교)
        summary = {}
        for row in csv.DictReader((d / "search_eval_dev_summary.csv").open(encoding="utf-8-sig")):
            if row["metric"] in ("Hit@1", "Hit@5", "Hit@10", "MRR@10"):
                summary[row["metric"]] = float(row["rerank"])
        if set(summary) != {"Hit@1", "Hit@5", "Hit@10", "MRR@10"}:
            raise ValidationError(f"{run_dir.name}/{s}: 평가기 summary에 Hit@1/5/10·MRR@10이 없다")
        mine = metrics([rows[q] for q in ids])
        for key, metric in (("hit1", "Hit@1"), ("hit5", "Hit@5"), ("hit10", "Hit@10")):
            if metric in summary and abs(summary[metric] * len(ids) - mine[key]) > 1e-6 * len(ids) + 1e-3:
                raise ValidationError(f"{run_dir.name}/{s}: {metric} 평가기 {summary[metric]} ≠ 집계 {mine[key]}/{len(ids)}")
        if "MRR@10" in summary and abs(summary["MRR@10"] - mine["mrr10"]) > 1e-5:
            raise ValidationError(f"{run_dir.name}/{s}: MRR@10 평가기 {summary['MRR@10']} ≠ 집계 {mine['mrr10']}")
    return {"name": run_dir.name, "dir": str(run_dir), "cond": cond, "rows": rows, "order": order}


def status_of(row: Dict[str, Any]) -> str:
    o = row["outcome"]
    if o in FAILED_OUTCOMES:
        return "failed"
    if o == "partial_failure":
        return "partial"
    return o  # applied · policy_skipped · skipped · unknown


def has_budget_exhausted(row: Dict[str, Any]) -> bool:
    return "budget_exhausted" in row["group_statuses"]


def is_failure_touched(row: Dict[str, Any]) -> bool:
    return status_of(row) in {"failed", "partial"} or has_budget_exhausted(row)


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ranks = [r["rank"] for r in rows]
    statuses = Counter(status_of(r) for r in rows)
    return {
        "queries": n,
        "hit1": sum(1 for x in ranks if x == 1), "hit5": sum(1 for x in ranks if x and x <= 5),
        "hit10": sum(1 for x in ranks if x and x <= 10),
        "mrr10": round(sum(1.0 / x for x in ranks if x and x <= 10) / n, 6) if n else 0.0,
        "mrr10_numerator": round(sum(1.0 / x for x in ranks if x and x <= 10), 6), "mrr10_denominator": n,
        "recall30": sum(1 for r in rows if r["cand_rank"]),
        "status": dict(sorted(statuses.items())),
        "budget_exhausted_queries": sum(1 for r in rows if has_budget_exhausted(r)),
        "group_status": dict(sorted(Counter(s for r in rows for s in r["group_statuses"]).items())),
    }


def subsets(run: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rows = [run["rows"][q] for q in run["order"]]
    lyric = [run["rows"][q] for q in run["order"] if q in LYRIC_QUERY_IDS]
    other = [run["rows"][q] for q in run["order"] if q not in LYRIC_QUERY_IDS]
    out = {"all": rows, "lyric": lyric, "nonlyric": other}
    for s in SETS:
        out[s] = [run["rows"][q] for q in run["order"] if run["rows"][q]["set"] == s]
    return out


def input_view(h: Dict[str, Any]) -> Dict[str, Any]:
    cands = h.get("candidates") or []
    return {
        "cand_ids": [c["id"] for c in cands], "cand_scores": [c["score"] for c in cands],
        "lyrics": [[(t["id"], t["type"], t["score"]) for t in e.get("top", [])] for e in h.get("lyrics", [])],
        "protection": h.get("protection"),
        "rerank_input": [(g["input_ids"], g["input_scores"], g["input_lyric"], g["top_k"]) for g in h.get("rerank_runs", [])],
        "prompts": first_prompts(h),
    }


def classify(rx: Dict[str, Any], ry: Dict[str, Any]) -> Dict[str, Any]:
    vx, vy = input_view(rx["harness"]), input_view(ry["harness"])
    prompts_differ = (len(vx["prompts"]) != len(vy["prompts"]) or any(
        a != b for a, b in zip(vx["prompts"], vy["prompts"]) if a is not None and b is not None))
    detail = []
    if set(vx["cand_ids"]) != set(vy["cand_ids"]):
        detail.append("cand_set")
    elif vx["cand_ids"] != vy["cand_ids"]:
        detail.append("cand_order")
    elif vx["cand_scores"] != vy["cand_scores"]:
        detail.append("cand_score")
    if vx["lyrics"] != vy["lyrics"]:
        detail.append("lyrics")
    if vx["protection"] != vy["protection"]:
        detail.append("protection")
    if vx["rerank_input"] != vy["rerank_input"]:
        detail.append("groups")
    if prompts_differ:
        detail.append("prompt")
    order_diff = rx["final10"] != ry["final10"]
    if not order_diff:
        cause = "same"
    elif is_failure_touched(rx) or is_failure_touched(ry):
        cause = "failure_fallback"
    elif status_of(rx) == "policy_skipped" and status_of(ry) == "policy_skipped":
        cause = "policy_skip_order"
    elif detail:
        cause = "input_change"
    else:
        cause = "model_variation"
    return {"order_diff": order_diff, "cause": cause, "input_change": detail}


def compare_pair(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    if x["order"] != y["order"]:
        raise ValidationError(f"{x['name']} vs {y['name']}: 질의 집합·순서가 다르다")
    per = []
    for q in x["order"]:
        rx, ry = x["rows"][q], y["rows"][q]
        c = classify(rx, ry)
        a, b = rx["rank"], ry["rank"]
        if a == b:
            change = "same_rank"
        elif b is not None and (a is None or b < a):
            change = "gain"
        else:
            change = "loss"
        per.append({"query_id": q, "set": rx["set"], "lyric": q in LYRIC_QUERY_IDS, "rank_x": a, "rank_y": b,
                    "change": change, "status_x": status_of(rx), "status_y": status_of(ry), **c})
    counts = Counter(p["cause"] for p in per if p["order_diff"])
    rank_changes = [p for p in per if p["change"] != "same_rank"]
    return {
        "pair": f"{x['name']} → {y['name']}",
        "final_top10_identical": sum(1 for p in per if not p["order_diff"]),
        "final_order_diff_by_cause": dict(sorted(counts.items())),
        "gold_rank_changed": len(rank_changes),
        "gains": [p["query_id"] for p in rank_changes if p["change"] == "gain"],
        "losses": [p["query_id"] for p in rank_changes if p["change"] == "loss"],
        "gold_rank_changes_by_cause": dict(sorted(Counter(p["cause"] for p in rank_changes).items())),
        "input_change_queries": [p["query_id"] for p in per if p["input_change"]],
        "input_change_detail_counts": dict(sorted(Counter(d for p in per for d in p["input_change"]).items())),
        "rank_changes": rank_changes,
        "per_query": per,
    }


def check_same_conditions(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = list(runs)
    base = runs[names[0]]["cond"]
    def view(cond: Dict[str, Any], key: str) -> Any:
        # v68: 구성 사이에 다른 것은 AUDIO_FUSION_SCALE 하나뿐이어야 한다 — env 비교에서 그 키만 빼고 아래에서 구성별 값을 따로 본다
        value = cond.get(key)
        if key == "env" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k != "AUDIO_FUSION_SCALE"}
        return value

    for n in names[1:]:
        c = runs[n]["cond"]
        for key in SAME_CONDITION_KEYS:
            if json.dumps(view(c, key), sort_keys=True) != json.dumps(view(base, key), sort_keys=True):
                raise ValidationError(f"{n}: 실행 조건 {key}가 {names[0]}과 다르다")
    for n, r in runs.items():
        want = AUDIO_SCALE.get(r["cond"].get("code"))
        got = (r["cond"].get("env") or {}).get("AUDIO_FUSION_SCALE")
        if want is None or got != want:
            raise ValidationError(f"{n}: 구성 {r['cond'].get('code')}의 AUDIO_FUSION_SCALE={got!r}(기대 {want!r})")
    fingerprints = {((r["cond"].get("code_identity") or {}).get("commit"),
                     (r["cond"].get("code_identity") or {}).get("src_py_tree_sha16")) for r in runs.values()}
    if len(fingerprints) != 1:
        raise ValidationError(f"구성 사이 코드 지문이 다르다(같은 코드여야 한다): {fingerprints}")
    by_code: Dict[str, set] = {}
    for n, r in runs.items():
        ci = r["cond"].get("code_identity") or {}
        by_code.setdefault(r["cond"]["code"], set()).add((ci.get("commit"), ci.get("src_py_tree_sha16")))
    for code, ids in by_code.items():
        if len(ids) != 1:
            raise ValidationError(f"코드 판 {code}의 실행끼리 코드 지문이 다르다: {ids}")
    return {"same_keys": list(SAME_CONDITION_KEYS), "code": {k: sorted(v) for k, v in by_code.items()}}


def aggregate(run_specs: Dict[str, Path], pairs: List[tuple], *, expected_ids=None, expected_counts=None,
              require_live: bool = True) -> Dict[str, Any]:
    expected_ids = expected_ids or expected_query_ids()
    expected_counts = expected_counts or EXPECTED
    runs = {label: load_run(path, expected_ids=expected_ids, expected_counts=expected_counts, require_live=require_live)
            for label, path in run_specs.items()}
    cond_check = check_same_conditions(runs)
    result: Dict[str, Any] = {"conditions": cond_check, "runs": {}, "pairs": {}}
    for label, run in runs.items():
        subs = subsets(run)
        entry = {"dir": run["dir"], "code": run["cond"]["code"], "rep": run["cond"]["rep"],
                 "metrics": {k: metrics(v) for k, v in subs.items()}}
        ok_rows = [r for r in subs["all"] if not is_failure_touched(r)]
        entry["aux_excluding_failure_touched"] = {**metrics(ok_rows), "excluded": len(subs["all"]) - len(ok_rows)}
        result["runs"][label] = entry
    for a, b in pairs:
        result["pairs"][f"{a}:{b}"] = compare_pair(runs[a], runs[b])
    result["_runs"] = runs
    return result


def write_outputs(result: Dict[str, Any], out: Path, labels: List[str]) -> None:
    runs = result.pop("_runs")
    out.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(result))
    for p in slim["pairs"].values():
        p.pop("per_query")
    (out / "live_metrics.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # 실행별 표(전체·가사·비가사·세트)
    with (out / "live_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "subset", "queries", "hit1", "hit5", "hit10", "mrr10", "recall30", "applied", "partial", "failed",
                    "policy_skipped", "skipped", "budget_exhausted_queries"])
        for label in labels:
            for sub, m in result["runs"][label]["metrics"].items():
                st = m["status"]
                w.writerow([label, sub, m["queries"], m["hit1"], m["hit5"], m["hit10"], m["mrr10"], m["recall30"],
                            st.get("applied", 0), st.get("partial", 0), st.get("failed", 0), st.get("policy_skipped", 0),
                            st.get("skipped", 0), m["budget_exhausted_queries"]])
    # 질의별(최종 정답 위치·상태·쌍별 원인)
    first = runs[labels[0]]
    with (out / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        pair_keys = list(result["pairs"])
        w.writerow(["query_id", "set", "lyric", "gold"] + [f"rank_{l}" for l in labels] + [f"status_{l}" for l in labels]
                   + [f"cand_rank_{l}" for l in labels] + [f"cause_{k}" for k in pair_keys]
                   + [f"input_change_{k}" for k in pair_keys] + [f"final10_{l}" for l in labels])
        per_by_pair = {k: {p["query_id"]: p for p in result["pairs"][k]["per_query"]} for k in pair_keys}
        for q in first["order"]:
            rows = [runs[l]["rows"][q] for l in labels]
            w.writerow([q, rows[0]["set"], int(q in LYRIC_QUERY_IDS), "|".join(rows[0]["gold"])]
                       + [r["rank"] if r["rank"] else "" for r in rows] + [status_of(r) for r in rows]
                       + [r["cand_rank"] if r["cand_rank"] else "" for r in rows]
                       + [per_by_pair[k][q]["cause"] for k in pair_keys]
                       + ["|".join(per_by_pair[k][q]["input_change"]) for k in pair_keys]
                       + ["|".join(r["final10"]) for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="LABEL=<dir>,...")
    parser.add_argument("--pairs", required=True, help="X:Y,... (X → Y)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"출력 폴더에 이미 결과가 있다: {out}", file=sys.stderr)
        return 2
    specs = {}
    for part in args.runs.split(","):
        label, path = part.split("=", 1)
        specs[label] = Path(path)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    try:
        result = aggregate(specs, pairs)
    except ValidationError as exc:
        print(f"검증 실패: {exc}", file=sys.stderr)
        return 3
    labels = list(specs)
    for label in labels:
        m = result["runs"][label]["metrics"]
        a, l, o = m["all"], m["lyric"], m["nonlyric"]
        print(f"{label}: 176 Hit@1/5/10 {a['hit1']}/{a['hit5']}/{a['hit10']} MRR {a['mrr10']:.4f} 후보 {a['recall30']} "
              f"상태 {a['status']} 예산소진 {a['budget_exhausted_queries']} | 가사 {l['hit1']}/{l['hit5']}/{l['hit10']} "
              f"{l['mrr10']:.4f} | 비가사 {o['hit1']}/{o['hit5']}/{o['hit10']} {o['mrr10']:.4f}")
    for k, p in result["pairs"].items():
        print(f"{k}: top10 동일 {p['final_top10_identical']} · 순서 차이 원인 {p['final_order_diff_by_cause']} · "
              f"정답 위치 변화 {p['gold_rank_changed']} (얻음 {p['gains']} 잃음 {p['losses']}) · 입력 변화 {len(p['input_change_queries'])}")
    write_outputs(result, out, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
