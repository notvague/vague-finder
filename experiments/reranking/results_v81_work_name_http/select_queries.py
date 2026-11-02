"""상태별로 해시 순 선택. 질의 내용을 보지 않는다."""
import csv, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
status = json.load((ROOT / "experiments/reranking/results_v75_work_audio_live_ab/inputs/expected_work_status.json").open(encoding="utf-8"))["status_by_config"]["work"]
key = lambda q: hashlib.sha256(f"v81:{q}".encode()).hexdigest()
want = {"applied": 4, "abstained": 1, "no_usable_work": 1, "not_triggered": 4}
text = {}
for s in ("v06", "v09", "v10"):
    for r in csv.DictReader((ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig")):
        if r.get("split") == "dev":
            text[r["query_id"]] = r["query"]
picked = []
for st, n in want.items():
    picked += [{"query_id": q, "expected_status": st, "query": text[q]} for q in sorted((q for q, v in status.items() if v == st), key=key)[:n]]
picked.sort(key=lambda r: key("order" + r["query_id"]))
(HERE / "queries.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
print([(r["query_id"], r["expected_status"]) for r in picked])
