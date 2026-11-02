"""세트 하나가 끝날 때 리랭킹 실패를 본다. 실패가 섞인 실행은 비교에서 빠지므로 돈을 더 쓰기 전에 멈춘다(종료 코드 91)."""
import csv, sys
from pathlib import Path
rows = list(csv.DictReader((Path(sys.argv[1]) / "search_eval_dev_detail.csv").open(encoding="utf-8-sig")))
bad = [r["query_id"] for r in rows if str(r.get("rerank_fell_back", "")).strip().lower() in {"1", "1.0", "true"}
       or str(r.get("rerank_partial_failure", "")).strip().lower() in {"1", "1.0", "true"}
       or str(r.get("rerank_outcome", "")).strip() in {"failed", "budget_exhausted", "partial_failed", "model_partial"}
       or str(r.get("rerank_error", "")).strip()]
print(f"{sys.argv[1]}: {len(rows)}건, 실패 {len(bad)} {bad[:8]}")
log = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace") if len(sys.argv) > 2 else ""
path_failures = log.count("경로 실패")
if path_failures:
    print(f"검색 경로 실패 {path_failures}회 — 조건이 달라져 비교에서 뺀다")
sys.exit(91 if bad or not rows or path_failures else 0)
