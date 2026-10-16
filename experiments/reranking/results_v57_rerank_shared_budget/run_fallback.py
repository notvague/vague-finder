"""§6 리랭킹 실패·폴백 검증 — 실제 서버(`/api/v1/search`)에 요청을 보내고, 요청마다 가짜 SDK 클라이언트의 모드를 바꾼다.

가짜는 리랭커가 아니라 그 아래 SDK 클라이언트다(`tests/fake_gemini_client.py`). 재시도·파서·시간 예산·상태 보고·그룹 합치기·
가사 보호 배치·답변 가산·최종 정렬은 모두 운영 코드가 돈다. 모드는 서버가 읽는 파일(`ctl/fake_mode_<arm>.json`)로 바꾼다.

사례(사전 고정):
  t014 · t049  v55 저장 분석(test 캐시)으로 **장애 재현만** — A 후보 1위/16위 · D 후보 10위/22위였던 사례. 새 평가·튜닝 아님
  n007 · t018  가사 exact 보호 그룹이 있는 dev 질의(v54에서 호출 3회 · 2회) — 그룹 하나만 실패하는 경우 포함
  q112 · n011  재질문 답이 있는 2턴(prior_analysis + answers + rejected_ids) — 성별 답(q112) · 장르 답(n011)
  q117         표지 지배 질의 — 정책 생략(모델을 부르면 안 된다)
모드: normal → exception → slow(제한 초과) → invalid_json → foreign_ids → slow_within(8초) → normal(실패 뒤 정상)
    보호 그룹 질의에는 fail_group_first(일부 그룹 실패)를 더한다. 각 요청 뒤 가짜 호출 기록(JSONL)의 새 줄을 세어 실제 호출 수를 남긴다.

실행: venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/run_fallback.py [--arms A,D]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
SERVERS = {"A": "http://127.0.0.1:8056", "D": "http://127.0.0.1:8057"}
BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True}
TIMEOUT = 600
MODES_BASE = ["normal", "exception", "slow", "invalid_json", "foreign_ids", "slow_within", "normal"]
MODES_PROTECTED = ["normal", "exception", "fail_group_first", "slow", "normal"]
MODES_TURN2 = ["normal", "exception", "slow", "normal"]

CASES = [
    {"query_id": "t014", "kind": "repro_v55", "modes": MODES_BASE},
    {"query_id": "t049", "kind": "repro_v55", "modes": MODES_BASE},
    {"query_id": "n007", "kind": "lyric_protected", "modes": MODES_PROTECTED},
    {"query_id": "t018", "kind": "lyric_protected", "modes": MODES_PROTECTED},
    {"query_id": "q117", "kind": "policy_skip", "modes": ["exception", "normal"]},
    {"query_id": "q112", "kind": "turn2_gender", "modes": MODES_TURN2, "answers": [{"slot": "vocal_gender", "value": "남성"}]},
    {"query_id": "n011", "kind": "turn2_genre", "modes": MODES_TURN2, "answers": [{"slot": "genre", "value": "댄스"}]},
]
# v55 저장 결과(비교용, 리랭킹 전 top-10) — RUN_INFO §5-4 / top10.csv
V55_BASELINE_TOP = {
    ("A", "t014"): "609293|106465|525682|37998359|4684836|845580|454839|488134|104372|488135".split("|"),
    ("D", "t014"): "106465|621087|24670|441007|420248|317670|104419|103839|431033|609293".split("|"),
}
RELEVANT = {"t014": "609293", "t049": "3170749", "n007": None, "t018": None, "q112": None, "n011": None, "q117": None}


def load_queries() -> Dict[str, str]:
    import sys as _sys
    for p in (str(HERE.parents[2]), str(HERE)):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    import v56_runtime
    analyzer = v56_runtime.CachedAnalyzer()
    return {qid: query for query, (qid, _a, _f, _p) in analyzer.by_query.items()}


def set_mode(arm: str, mode: str, sleep: float = 8.0) -> None:
    path = HERE / "ctl" / f"fake_mode_{arm}.json"
    path.write_text(json.dumps({"mode": mode, "sleep": sleep}), encoding="utf-8")
    time.sleep(0.05)


def fake_log_lines(arm: str) -> int:
    path = HERE / f"fake_calls_{arm}.jsonl"
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def fake_log_tail(arm: str, since: int) -> List[Dict[str, Any]]:
    path = HERE / f"fake_calls_{arm}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[since:]
    return [json.loads(line) for line in lines]


def post(arm: str, body: Dict[str, Any]) -> tuple:
    started = time.perf_counter()
    resp = requests.post(f"{SERVERS[arm]}/api/v1/search", json=body, timeout=TIMEOUT)
    return resp, round((time.perf_counter() - started) * 1000, 1)


def summarize(resp, ms: float, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"client_ms": ms, "status": resp.status_code, "request_id": resp.headers.get("X-Request-Id", "")}
    row["fake_calls"] = len(calls)
    row["fake_outcomes"] = "|".join(c["outcome"] for c in calls)
    row["fake_call_sizes"] = "|".join(str(c["n_candidates"]) for c in calls)
    row["fake_elapsed_s"] = "|".join(str(c["elapsed_s"]) for c in calls)
    if resp.status_code != 200:
        row.update({"error": resp.text[:300]})
        return row
    data = resp.json()
    results = data.get("results", [])
    ids = [r["id"] for r in results]
    cands = data.get("candidate_ids", [])
    ex = data.get("explain") or {}
    row.update({
        "n_results": len(ids), "n_unique": len(set(ids)), "all_in_candidates": all(i in cands for i in ids),
        "n_candidates": len(cands), "result_ids": "|".join(ids), "candidate_top10": "|".join(cands[:10]),
        "reorder_stage": ex.get("reorder_stage", ""), "reorder_stage_label": ex.get("reorder_stage_label", ""),
        "rerank_calls": ex.get("rerank_calls"), "rerank_calls_applied": ex.get("rerank_calls_applied"),
        "rerank_error": ex.get("rerank_error", ""), "rerank_skip_reason": ex.get("rerank_skip_reason", ""),
        "notice": (data.get("notice") or {}).get("code", ""), "turn": data.get("turn"), "clarify_deferred": data.get("clarify_deferred"),
        "attr_filter_genre": json.dumps(ex.get("attr_filter_genre"), ensure_ascii=False) if ex.get("attr_filter_genre") else "",
        "analysis": data.get("analysis"), "error": "",
    })
    # 상위 곡의 설명에서 리랭킹 점수·순서 규칙을 뽑는다(가사 보호 배치 확인용)
    rules = []
    for r in results[:5]:
        te = r.get("explain") or {}
        for o in te.get("order_rules", []) or []:
            rules.append(f"{r['id']}:{o.get('rule') if isinstance(o, dict) else o}")
    row["top5_order_rules"] = "|".join(rules)[:400]
    row["rerank_scores_top3"] = "|".join(str(r.get("rerank_score")) for r in results[:3])
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A,D")
    parser.add_argument("--cases", default="", help="쉼표 구분 query_id — 주면 그 사례만")
    parser.add_argument("--modes", default="", help="쉼표 구분 모드 — 주면 사례별 모드 순서를 이것으로 대신")
    parser.add_argument("--suffix", default="", help="출력 파일 접미사(예: _subset)")
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        CASES[:] = [c for c in CASES if c["query_id"] in wanted]
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for c in CASES:
            c["modes"] = list(modes)
    queries = load_queries()
    rows: List[Dict[str, Any]] = []
    out_csv = HERE / f"fallback_results{args.suffix}.csv"
    out_jsonl = HERE / f"fallback_results{args.suffix}.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()
    for arm in arms:
        for case in CASES:
            qid = case["query_id"]
            query = queries[qid]
            prior = None
            rejected: List[str] = []
            if case.get("answers"):
                # 1턴(정상)으로 분석·거절 대상 확보 — 화면과 같은 왕복
                set_mode(arm, "normal")
                resp, _ = post(arm, dict(BODY, query=query))
                data = resp.json()
                prior = data["analysis"]
                rejected = [r["id"] for r in data["results"][:1]]
            for step, mode in enumerate(case["modes"], start=1):
                set_mode(arm, mode)
                before = fake_log_lines(arm)
                body = dict(BODY, query=query)
                if prior is not None:
                    body.update({"prior_analysis": prior, "answers": case["answers"], "rejected_ids": rejected,
                                 "asked_slots": [a["slot"] for a in case["answers"]], "turn": 1})
                resp, ms = post(arm, body)
                calls = fake_log_tail(arm, before)
                row = {"arm": arm, "query_id": qid, "kind": case["kind"], "step": step, "mode": mode,
                       "sent_at": time.strftime("%H:%M:%S"), **summarize(resp, ms, calls)}
                if (arm, qid) in V55_BASELINE_TOP and row.get("result_ids"):
                    top = row["result_ids"].split("|")[:10]
                    row["v55_baseline_top10_equal"] = top == V55_BASELINE_TOP[(arm, qid)]
                    row["v55_baseline_top10_same_set"] = set(top) == set(V55_BASELINE_TOP[(arm, qid)])
                if RELEVANT.get(qid) and row.get("result_ids"):
                    ids = row["result_ids"].split("|")
                    row["relevant_rank"] = ids.index(RELEVANT[qid]) + 1 if RELEVANT[qid] in ids else ""
                    cands = (row.get("candidate_top10") or "").split("|")
                    row["relevant_candidate_rank_top10"] = cands.index(RELEVANT[qid]) + 1 if RELEVANT[qid] in cands else ">10"
                analysis = row.pop("analysis", None)
                rows.append(row)
                with out_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**row, "analysis": analysis, "fake_calls_detail": calls}, ensure_ascii=False) + "\n")
                print(f"{arm} {qid:>5} {case['kind']:>15} {mode:>16} ms={row['client_ms']:>8} status={row['status']} "
                      f"stage={row.get('reorder_stage','')} calls={row.get('rerank_calls')}/{row.get('rerank_calls_applied')} "
                      f"fake={row['fake_calls']}[{row['fake_outcomes']}] rel={row.get('relevant_rank','-')} "
                      f"top3={'|'.join((row.get('result_ids') or '').split('|')[:3])}", flush=True)
            set_mode(arm, "normal")
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print("done", len(rows), "->", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
