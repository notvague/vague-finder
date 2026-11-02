"""화면과 같은 모양으로 요청한다. 예열 완료를 기다린 뒤 질의를 차례로 보낸다."""
import json, sys, time, urllib.error, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
arm = sys.argv[1]; URL = "http://127.0.0.1:8771"
out = HERE / "runs" / arm; out.mkdir(parents=True, exist_ok=True)
for _ in range(240):
    try:
        h = json.loads(urllib.request.urlopen(f"{URL}/health", timeout=5).read())
        w = h.get("warmup") or {}
        if str(w.get("state")) not in {"not_started", "running", "None"}:
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("예열 대기 시간 초과", flush=True)
(out / "health.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
print("health", {k: h.get(k) for k in ("status", "context")}, "warmup", json.dumps(h.get("warmup"), ensure_ascii=False)[:200], flush=True)
rows = []
for q in json.loads((HERE / "queries.json").read_text(encoding="utf-8")):
    body = {"query": q["query"], "top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True}
    req = urllib.request.Request(f"{URL}/api/v1/search", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            status, payload = res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, {"error": exc.read().decode("utf-8", "replace")[:300]}
    except Exception as exc:
        status, payload = 0, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    sec = round(time.time() - t, 2)
    ex = payload.get("explain") or {}
    rows.append({"query_id": q["query_id"], "expected_status": q["expected_status"], "http": status, "seconds": sec, "analysis_fallback": ex.get("analysis_fallback"),
                 "reorder_stage": ex.get("reorder_stage"), "rerank_calls": ex.get("rerank_calls"), "rerank_calls_applied": ex.get("rerank_calls_applied"), "rerank_groups_failed": ex.get("rerank_groups_failed"), "rerank_skip_reason": ex.get("rerank_skip_reason"),
                 "failed_paths": ex.get("failed_paths"), "top_ids": [t["id"] for t in payload.get("results", [])], "error": payload.get("error", "")})
    (out / f"resp_{q['query_id']}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(q["query_id"], q["expected_status"], status, sec, ex.get("reorder_stage"), ex.get("rerank_groups_failed"), flush=True)
(out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("DONE", flush=True)
