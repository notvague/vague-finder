"""측정용 서버. 비용 장치를 걸고 해석 기록을 파일로 받는다. 제품 코드는 바꾸지 않는다."""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/reranking/results_v75_work_audio_live_ab")]
import v75_budget
arm = sys.argv[1]
v75_budget.install(HERE / "budget/ledger.jsonl", cost_cap_usd=1.0, attempt_cap=150, deadline_window_seconds=2400, run_tag=arm, model="gemini-3.1-flash-lite")
from src.retrieval import explain as ex
out = HERE / "runs" / arm; out.mkdir(parents=True, exist_ok=True)
orig = ex.ExplainRecorder.set_work_resolution
def record(self, rec):
    with (out / "work_resolution.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "query": getattr(self.record, "query", ""), **dict(rec)}, ensure_ascii=False, default=str) + "\n")
    return orig(self, rec)
ex.ExplainRecorder.set_work_resolution = record
import uvicorn
uvicorn.run("src.backend.main:app", host="127.0.0.1", port=8771, log_level="warning")
