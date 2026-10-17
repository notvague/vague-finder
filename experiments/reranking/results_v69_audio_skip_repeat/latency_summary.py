"""v69 지연 요약 — 실제 실행 원기록(`*_dev/v69_queries.jsonl`, v68은 `v68_queries.jsonl`)에서 검색 단계(라우터 search, 리랭킹 전) 시간과
CLAP 텍스트 임베딩 시간, 실행 조건의 CLAP 로드 시간을 구성별로 요약한다. 모델·DB 호출 없음.

평가기 경로(top_k 30·질의 순차·한 프로세스) 기준이다 — 서비스 HTTP 경로(top_k 15·동시 요청)의 지연이 아니다. 리랭킹 시간은 Vertex
응답 시간이 대부분이라 구성 차이를 보는 데 쓰지 않는다(장부 `latency_ms`는 `budget_summary.json`).

    venv/bin/python latency_summary.py --runs A3=live/live_on_server_rep3,B3=live/live_skip_server_rep3 --out analysis/latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
SETS = ("v06", "v09", "v10")


def pct(xs: List[float], q: float) -> float:
    s = sorted(xs)
    return round(s[int(q * (len(s) - 1))], 1) if s else 0.0


def load(run_dir: Path) -> Dict[str, Any]:
    search, clap, qids = [], [], []
    for s in SETS:
        d = run_dir / f"{s}_dev"
        f = d / "v69_queries.jsonl"
        if not f.exists():
            f = d / "v68_queries.jsonl"
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            qids.append(r["query_id"])
            search.append(float(r.get("search_ms") or 0.0))
            clap.append(sum(float(e.get("ms") or 0.0) for e in r.get("clap_embed", [])))
    cond = json.loads((run_dir / "run_conditions.json").read_text(encoding="utf-8"))
    if len(qids) != 176 or len(set(qids)) != 176:
        raise SystemExit(f"{run_dir}: 질의 {len(qids)}건")
    return {"qids": qids, "search_ms": search, "clap_embed_ms": clap,
            "clap_load_ms": (cond.get("clap") or {}).get("load_ms", []), "code": cond.get("code"),
            "cold_before_first_query_seconds": cond.get("cold_before_first_query_seconds")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out: Dict[str, Any] = {"note": "평가기 경로·리랭킹 전 검색 단계·질의 순차. 서비스 HTTP 지연 아님", "runs": {}}
    for part in args.runs.split(","):
        label, path = part.split("=", 1)
        r = load(Path(path))
        nz = [x for x in r["clap_embed_ms"] if x > 0]
        out["runs"][label] = {
            "code": r["code"], "queries": len(r["qids"]),
            "search_ms_median": round(statistics.median(r["search_ms"]), 1), "search_ms_p90": pct(r["search_ms"], 0.9),
            "search_ms_p95": pct(r["search_ms"], 0.95), "search_ms_mean": round(statistics.mean(r["search_ms"]), 1),
            "queries_with_clap_embed": len(nz),
            "clap_embed_ms_median_when_used": round(statistics.median(nz), 1) if nz else 0.0,
            "clap_embed_ms_sum": round(sum(nz), 1),
            "clap_load_ms": r["clap_load_ms"],
            "cold_before_first_query_seconds": r["cold_before_first_query_seconds"],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["runs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
