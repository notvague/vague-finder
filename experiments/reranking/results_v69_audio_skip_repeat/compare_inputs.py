"""v69(v68 사본) 입력 지문 대조 — 두 실행(오프라인↔오프라인, live↔오프라인)의 질의별 모델 입력을 비교한다. 모델·DB 호출 없음.

비교 항목(질의 176건마다):
  cand_ids       후보 30 ID·순서                        cand_scores   후보 융합 점수(정확히 같은 값)
  lyrics         가사 검색 결과 id·종류·점수              protection    보호 곡·tier·정책 생략 사유
  rerank_input   리랭커 그룹별 입력 id·점수·가사 종류·top_k
  prompts        그룹마다 **첫 호출**의 프롬프트 sha16(그룹 순서대로). live에서 예산 소진으로 호출이 없던 그룹은 None
  final          최종 순서 — 오프라인끼리만(가짜 결정적 응답이라 입력이 같으면 같아야 한다). live 쪽이 있으면 비교하지 않는다
경로별: Qdrant 논리 조회의 반환 id 다중집합(경로 스레드 완료 순서와 무관).
추가 검사(live 쪽): 그룹 안 재시도가 첫 호출과 같은 프롬프트를 보냈는가(`retry_prompt_changed`).

    venv/bin/python compare_inputs.py --a <실행 폴더> --b <실행 폴더> [--require-identical] [--out <json>]
종료 코드: 0 · 3(--require-identical인데 입력 차이, 또는 완료되지 않은 실행) · 2(출력 덮어쓰기 거부)

v69 변경: ① 기록 파일 이름 `v69_*.jsonl`, 없으면 v68 실행의 `v68_*.jsonl`을 읽는다(v68 유효 입력과 대조용 · 둘 다 있으면 거부)
② `--require-identical`이 Qdrant 경로 반환 차이(`path_diff_query_ids`)도 차이로 친다 — v68은 보고만 하고 멈추지 않았다
   (v68 저장 대조 14건 모두 경로 차이 0이라 v68 결과에는 영향 없음).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SETS = ("v06", "v09", "v10")
INPUT_FIELDS = ("cand_ids", "cand_scores", "lyrics", "protection", "rerank_input", "prompts")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}


class Incomplete(Exception):
    pass


def record_file(d: Path, kind: str) -> Path:
    """`kind` = queries | qdrant_calls. v69 기록을 먼저, 없으면 v68 기록."""
    new, old = d / f"v69_{kind}.jsonl", d / f"v68_{kind}.jsonl"
    if new.exists() and old.exists():
        raise Incomplete(f"v68·v69 기록이 함께 있다: {d}")
    return old if old.exists() and not new.exists() else new


def load_run(run_dir: Path, *, allow_incomplete: bool = False) -> Dict[str, Any]:
    cond_path = run_dir / "run_conditions.json"
    if not cond_path.exists():
        raise Incomplete(f"실행 기록 없음: {run_dir}")
    cond = json.loads(cond_path.read_text(encoding="utf-8"))
    done = (run_dir / "DONE").exists()
    if not done and not allow_incomplete:
        raise Incomplete(f"완료되지 않은 실행: {run_dir} (status={cond.get('status')})")
    if done:
        marker = json.loads((run_dir / "DONE").read_text(encoding="utf-8"))
        if marker.get("partial"):
            raise Incomplete(f"부분 실행: {run_dir}")
    queries: Dict[str, Dict[str, Any]] = {}
    calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in SETS:
        d = run_dir / f"{s}_dev"
        qf = record_file(d, "queries")
        if not qf.exists():
            if allow_incomplete:
                continue
            raise Incomplete(f"{s} 기록 없음: {run_dir}")
        n = 0
        for line in qf.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["query_id"] in queries:
                raise Incomplete(f"질의 ID 중복 {row['query_id']}: {run_dir}")
            queries[row["query_id"]] = row
            n += 1
        if done and n != EXPECTED[s]:
            raise Incomplete(f"{s} 건수 {n} ≠ {EXPECTED[s]}: {run_dir}")
        cf = record_file(d, "qdrant_calls")
        for line in (cf.read_text(encoding="utf-8").splitlines() if cf.exists() else []):
            c = json.loads(line)
            calls[c["query_id"]].append(c)
    return {"name": run_dir.name, "dir": str(run_dir), "cond": cond, "done": done, "queries": queries, "calls": calls,
            "live": cond.get("model") == "live"}


def first_prompts(row: Dict[str, Any]) -> List[Optional[str]]:
    groups = row.get("rerank_runs", [])
    firsts: Dict[int, str] = {}
    for g in row.get("generates", []):
        idx = g.get("group")
        if idx is not None and idx not in firsts:
            firsts[idx] = g["contents_sha16"]
    return [firsts.get(i) for i in range(len(groups))]


def retry_prompt_changed(row: Dict[str, Any]) -> bool:
    """그룹 안 재시도가 첫 호출과 다른 본문을 보냈는가(리랭커의 정정 지시 접미사는 떼고 본다)."""
    by_group: Dict[int, set] = defaultdict(set)
    for g in row.get("generates", []):
        by_group[g.get("group")].add(g.get("base_sha16", g["contents_sha16"]))
    return any(len(v) > 1 for v in by_group.values())


def view(row: Dict[str, Any]) -> Dict[str, Any]:
    cands = row.get("candidates") or []
    return {
        "cand_ids": [c["id"] for c in cands],
        "cand_scores": [c["score"] for c in cands],
        "lyrics": [[(t["id"], t["type"], t["score"]) for t in e.get("top", [])] for e in row.get("lyrics", [])],
        "protection": row.get("protection"),
        "rerank_input": [(r["input_ids"], r["input_scores"], r["input_lyric"], r["top_k"]) for r in row.get("rerank_runs", [])],
        "prompts": first_prompts(row),
        "final": (row.get("final") or {}).get("ids"),
    }


def path_signature(calls: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
    out: Dict[str, List[tuple]] = defaultdict(list)
    for c in calls:
        key = f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}"
        out[key].append((c.get("top_k"), bool(c.get("filtered") or c.get("media_filtered")), tuple(c["result_ids"])))
    return {k: sorted(v) for k, v in out.items()}


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    fields = list(INPUT_FIELDS) + ([] if (a["live"] or b["live"]) else ["final"])
    qids = sorted(set(a["queries"]) | set(b["queries"]))
    diff: Dict[str, List[str]] = {f: [] for f in fields}
    missing = [q for q in qids if q not in a["queries"] or q not in b["queries"]]
    cand_set_diff: List[str] = []
    path_diff: Dict[str, List[str]] = defaultdict(list)
    per_query: Dict[str, Any] = {}
    not_sent: Dict[str, List[int]] = {}
    retry_changed: List[str] = []
    for q in qids:
        ra, rb = a["queries"].get(q), b["queries"].get(q)
        if ra is None or rb is None:
            continue
        va, vb = view(ra), view(rb)
        changed = []
        for f in fields:
            if f == "prompts" and (a["live"] or b["live"]):
                # live 쪽에서 호출되지 않은 그룹(None)은 비교에서 빼고 따로 센다. 그룹 수가 다르면 차이다.
                pa, pb = va["prompts"], vb["prompts"]
                if len(pa) != len(pb):
                    changed.append(f)
                    continue
                skipped = [i for i, (x, y) in enumerate(zip(pa, pb)) if x is None or y is None]
                if skipped:
                    not_sent[q] = skipped
                if any(x != y for i, (x, y) in enumerate(zip(pa, pb)) if i not in skipped):
                    changed.append(f)
            elif va[f] != vb[f]:
                changed.append(f)
        for f in changed:
            diff[f].append(q)
        if set(va["cand_ids"]) != set(vb["cand_ids"]):
            cand_set_diff.append(q)
        for side in (ra, rb):
            if retry_prompt_changed(side):
                retry_changed.append(q)
                break
        pa_, pb_ = path_signature(a["calls"].get(q, [])), path_signature(b["calls"].get(q, []))
        paths = sorted(k for k in set(pa_) | set(pb_) if pa_.get(k) != pb_.get(k))
        for k in paths:
            path_diff[k].append(q)
        if changed or paths:
            per_query[q] = {"fields": changed, "paths": paths}
    input_diff = sorted({q for f in INPUT_FIELDS for q in diff[f]})
    return {
        "pair": f"{a['name']} vs {b['name']}", "a": a["dir"], "b": b["dir"],
        "a_done": a["done"], "b_done": b["done"], "queries": len(qids), "missing_either": missing,
        "fields_compared": fields,
        "input_identical": len(qids) - len(set(input_diff) | set(missing)),
        "input_diff_queries": input_diff,
        "diff_counts": {f: len(v) for f, v in diff.items()},
        "diff_queries": {f: v for f, v in diff.items() if v},
        "cand_set_diff_queries": cand_set_diff,
        "path_diff_query_ids": {k: v for k, v in sorted(path_diff.items())},
        "groups_not_sent": not_sent,
        "retry_prompt_changed_queries": retry_changed,
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="진단용 — 완료되지 않은 실행도 읽는다(요약에 표시)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out and Path(args.out).exists() and not args.overwrite:
        print(f"출력 파일이 이미 있다: {args.out}", file=sys.stderr)
        return 2
    try:
        a = load_run(Path(args.a), allow_incomplete=args.allow_incomplete)
        b = load_run(Path(args.b), allow_incomplete=args.allow_incomplete)
    except Incomplete as exc:
        print(f"비교 거부: {exc}", file=sys.stderr)
        return 3
    result = compare(a, b)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = (result["input_diff_queries"] or result["missing_either"] or result["retry_prompt_changed_queries"]
           or any(result["diff_counts"].values()) or result["path_diff_query_ids"])
    print(f"{result['pair']}: 입력 동일 {result['input_identical']}/{result['queries']} · 차이 {json.dumps(result['diff_counts'])}"
          f" · 후보 집합 {len(result['cand_set_diff_queries'])} · 경로 {json.dumps({k: len(v) for k, v in result['path_diff_query_ids'].items()})}"
          f" · 미호출 그룹 질의 {len(result['groups_not_sent'])} · 재시도 프롬프트 변화 {len(result['retry_prompt_changed_queries'])}")
    if args.require_identical and bad:
        print("입력 지문이 다르다 — 다음 모델 호출을 멈춘다", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
