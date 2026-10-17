"""집계 도구 자체 검사 — 정상 실행만으로는 안 드러나는 잘못을 일부러 넣은 입력으로 확인한다(모델·서버·DB 없음).

analyze_fallback: 상태 500 · 실패 모드인데 applied · 폴백 순서 어긋남 · 호출 수 불일치 · 거절 곡 누출 · 구간 기록 없음 · 정책 생략인데 호출
aggregate_latency: 200이 아닌 요청 · 구간 기록이 없는 요청을 세는가
임시 폴더에서만 돈다. 실행: venv/bin/python experiments/reranking/results_v64_service_readiness/selftest_analyzers.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_fallback as af  # noqa: E402
import aggregate_latency as al  # noqa: E402

CANDS = [str(i) for i in range(100, 130)]


def row(**kw):
    base = {"arm": "A", "query_id": "t014", "kind": "repro_v55", "step": 1, "mode": "exception", "status": 200, "client_ms": 1.0,
            "request_id": "r1", "candidate_ids": CANDS, "result_ids": CANDS[:15], "n_results": 15, "n_unique": 15,
            "results_in_candidates": True, "reorder_stage": "model_failed", "reorder_stage_note": "", "rerank_calls": 1,
            "rerank_calls_applied": 0, "rerank_groups_failed": 1, "rerank_groups_budget_exhausted": 0, "rerank_model_calls": 3,
            "fake_calls": 3, "fake_outcomes": ["raised"] * 3, "tracks": [], "rejected_ids": [], "rerank_skip_reason": "", "rerank_error": ""}
    base.update(kw)
    return base


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "logs" / "main").mkdir(parents=True)
        rows = [
            row(step=1, mode="normal", reorder_stage="model_applied", rerank_calls_applied=1, rerank_groups_failed=0, rerank_model_calls=1, fake_calls=1),
            row(step=2, status=500, request_id="r2"),                                          # 상태
            row(step=3, reorder_stage="model_applied", request_id="r3"),                       # 실패인데 적용
            row(step=4, result_ids=list(reversed(CANDS[:15])), request_id="r4"),               # 폴백 순서
            row(step=5, rerank_model_calls=2, request_id="r5"),                                # 호출 수
            row(step=6, kind="turn2_gender", query_id="q112", mode="normal", reorder_stage="model_applied", rejected_ids=[CANDS[0]], request_id="r6"),
            row(step=7, kind="policy_skip", query_id="q117", reorder_stage="policy_skipped", fake_calls=1, rerank_model_calls=None, request_id="r7"),
            row(step=8, mode="normal", reorder_stage="model_applied", rerank_calls_applied=1, rerank_groups_failed=0, rerank_model_calls=1,
                fake_calls=1, result_ids=CANDS[1:16], request_id="r8"),                        # 누출(마지막 normal ≠ 첫 normal)
        ]
        (tmp / "fallback_requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (tmp / "fallback_status_after.json").write_text(json.dumps({"A": {"network_attempts": [], "blocked_client_calls": [], "analysis_misses": 0, "fake_calls": 0}}))
        (tmp / "logs" / "main" / "timing_A.jsonl").write_text(json.dumps({"request_id": "r1", "total_ms": 1.0, "spans": []}) + "\n")
        (tmp / "logs" / "main" / "timing_D.jsonl").write_text("")
        af.HERE, af.LOG = tmp, tmp / "logs" / "main"
        af.main()
        summary = json.loads((tmp / "fallback_summary.json").read_text(encoding="utf-8"))
        flagged = {c["step"]: c["problems"] for c in summary["contract_violations"]}
        expect = {2: "status", 3: "stage=model_applied", 4: "fallback_not_candidate_order", 5: "model_calls!=fake_calls",
                  6: "rejected_leak", 7: "policy_skip_contract"}
        for step, needle in expect.items():
            if not any(needle in p for p in flagged.get(step, [])):
                failures.append(f"analyze_fallback step {step}: '{needle}' 미검출 ({flagged.get(step)})")
        leak = [x for x in summary["leak_checks"] if x["query_id"] == "t014"]
        if not leak or leak[0]["last_normal_equals_first"]:
            failures.append(f"analyze_fallback 누출 미검출 {leak}")

        # aggregate_latency — 200 아님 · 구간 기록 없음을 센다
        fields = ["query_id", "set", "category", "round", "arm", "order_in_pair", "warm", "sent_at", "client_ms", "status", "request_id",
                  "n_results", "n_candidates", "candidate_digest", "reorder_stage", "failed_paths", "error"]
        lat = [
            {"query_id": "x", "set": "v06", "category": "closed", "round": 1, "arm": "A", "order_in_pair": 1, "warm": 0, "sent_at": "", "client_ms": 10,
             "status": 200, "request_id": "r1", "n_results": 15, "n_candidates": 30, "candidate_digest": "d", "reorder_stage": "", "failed_paths": "", "error": ""},
            {"query_id": "x", "set": "v06", "category": "closed", "round": 1, "arm": "D", "order_in_pair": 2, "warm": 0, "sent_at": "", "client_ms": 10,
             "status": 500, "request_id": "missing", "n_results": "", "n_candidates": "", "candidate_digest": "", "reorder_stage": "", "failed_paths": "", "error": "x"},
        ]
        with (tmp / "latency_requests.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(lat)
        al.HERE, al.LOG = tmp, tmp / "logs" / "main"
        al.main()
        lsum = json.loads((tmp / "latency_summary.json").read_text(encoding="utf-8"))
        if lsum["non_200"] != 1 or lsum["timing_unmatched"] != 1:
            failures.append(f"aggregate_latency non_200={lsum['non_200']} unmatched={lsum['timing_unmatched']} (기대 1·1)")
    print("SELFTEST", "FAIL" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
