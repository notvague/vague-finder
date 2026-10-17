"""§5-2 동점 순서 차이의 영향 — 리랭킹 전 후보 30에서 정답 위치를 로컬(=v63 offline B)과 서버(HTTP A·D)로 비교한다.

품질 평가가 아니다(리랭킹 전 후보 순서의 백엔드 간 차이). 라벨은 엄격 정답(`eval_queries_v06/v09/v10.csv`의 relevant_ids).
출력: tie_impact.json

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/tie_impact.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from analyze_sweep import load_v63  # noqa: E402


def rank(ids, gold):
    for i, sid in enumerate(ids, start=1):
        if sid in gold:
            return i
    return None


def main() -> int:
    gold = {}
    for name in ("v06", "v09", "v10"):
        for row in csv.DictReader((ROOT / f"experiments/reranking/eval_queries_{name}.csv").open(encoding="utf-8")):
            if row["split"] == "dev":
                gold[row["query_id"]] = set(filter(None, row["relevant_ids"].replace(";", "|").replace(",", "|").split("|")))
    v63 = load_v63()
    http = {}
    for line in (HERE / "sweep_requests.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["kind"] == "dev" and r["status"] == 200:
            http[(r["arm"], r["query_id"])] = r["candidate_ids"]
    rows, changed = {}, []
    for qid, rec in v63.items():
        local = [c["id"] for c in rec["candidates"]]
        server = http[("A", qid)]
        rl, rs = rank(local, gold[qid]), rank(server, gold[qid])
        rows[qid] = {"local_rank": rl, "server_rank": rs, "same_order": local == server}
        if rl != rs:
            changed.append({"query_id": qid, "local": rl, "server": rs})
    summary = {
        "queries": len(rows), "labels": len(gold),
        "recall30_local": sum(1 for r in rows.values() if r["local_rank"]), "recall30_server_A": sum(1 for r in rows.values() if r["server_rank"]),
        "hit1_local": sum(1 for r in rows.values() if r["local_rank"] == 1), "hit1_server_A": sum(1 for r in rows.values() if r["server_rank"] == 1),
        "hit10_local": sum(1 for r in rows.values() if r["local_rank"] and r["local_rank"] <= 10),
        "hit10_server_A": sum(1 for r in rows.values() if r["server_rank"] and r["server_rank"] <= 10),
        "gold_rank_changed": changed, "order_diff_queries": sum(1 for r in rows.values() if not r["same_order"]),
        "note": "리랭킹 전 후보(boosted) 순서 기준. 로컬 = v63 offline B(f43daf7, 로컬 파일 모드) — 같은 후보 생성 코드이며 dd5cc3e는 src 변경 없음",
    }
    (HERE / "tie_impact.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
