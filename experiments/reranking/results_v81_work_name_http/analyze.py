"""OFF·ON 비교와 RUN_INFO §3 판정. 모델 호출 없음."""
import json, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
load = lambda arm: {r["query_id"]: r for r in json.loads((HERE / f"runs/{arm}/rows.json").read_text(encoding="utf-8"))}
off, on = load("off"), load("on")
queries = {q["query_id"]: q for q in json.loads((HERE / "queries.json").read_text(encoding="utf-8"))}
recs = {}
p = HERE / "runs/on/work_resolution.jsonl"
if p.exists():
    for line in p.open(encoding="utf-8"):
        r = json.loads(line); recs[r.get("query") or ""] = r
names = set()
print("질의 | 기대 | 실제 해석 | 해석기 | 해석ms | OFF초 | ON초 | 후보 같음 | 작품 ")
table = []
for qid, q in queries.items():
    a, b = off[qid], on[qid]; r = recs.get(q["query"]) or {}
    works = [w.get("name") or w.get("work") or w.get("title") or str(w) for w in r.get("works") or []]
    names.update(w for w in works if w)
    row = {"query_id": qid, "expected": q["expected_status"], "status": r.get("status"), "resolver_status": r.get("resolver_status"), "resolver_error": str(r.get("resolver_error") or "")[:80], "resolver_ms": r.get("resolver_latency_ms"), "off_s": a["seconds"], "on_s": b["seconds"], "same_top": a["top_ids"] == b["top_ids"], "aux": len(r.get("aux_ids") or []), "entered": len(r.get("entered_ids") or []), "works": works, "reorder_on": b["reorder_stage"], "reorder_off": a["reorder_stage"], "fail_on": b["rerank_groups_failed"], "fail_off": a["rerank_groups_failed"], "http": (a["http"], b["http"]), "fallback": (a["analysis_fallback"], b["analysis_fallback"]), "failed_paths": (a["failed_paths"], b["failed_paths"])}
    table.append(row)
    print(qid, row["expected"], row["status"], row["resolver_status"], row["resolver_ms"], a["seconds"], b["seconds"], row["same_top"], works[:3], row["reorder_off"], row["reorder_on"], row["resolver_error"])
trig = [t for t in table if t["expected"] != "not_triggered"]
miss = [t for t in table if t["expected"] == "not_triggered"]
d = lambda rows: round(st.median(t["on_s"] - t["off_s"] for t in rows), 2)
leak = []
for qid, q in queries.items():
    body = json.loads((HERE / f"runs/on/resp_{qid}.json").read_text(encoding="utf-8"))
    text = json.dumps({"s": [t.get("explain", {}).get("summary") for t in body.get("results", [])] + [(t.get("explain", {}).get("user_explanation") or {}).get("summary") for t in body.get("results", [])], "paths": [p for t in body.get("results", []) for p in t.get("explain", {}).get("paths", [])], "notice": body.get("notice")}, ensure_ascii=False)
    for name in names:
        if name and name in text and name not in q["query"]:
            leak.append((qid, name))
gates = {
    "H1": all(t["http"] == (200, 200) and not any(t["fallback"]) for t in table),
    "H2": all(t["status"] not in ("failed", "timeout") and t["resolver_status"] not in ("timeout", "error", "failed") for t in trig),
    "H3": all(not t["fail_on"] and not t["fail_off"] and t["reorder_on"] != "failed" and t["reorder_off"] != "failed" for t in table),
    "H4": d(miss) <= 0.5 and d(trig) <= 5 and max(t["on_s"] for t in table) <= 30,
    "H5": not leak,
}
out = {"table": table, "median_delta_not_triggered": d(miss), "median_delta_triggered": d(trig), "on_max_s": max(t["on_s"] for t in table), "off_max_s": max(t["off_s"] for t in table), "leak": leak, "gates": gates}
(HERE / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: out[k] for k in ("median_delta_not_triggered", "median_delta_triggered", "on_max_s", "off_max_s", "leak", "gates")})
led = [json.loads(l) for l in (HERE / "budget/ledger.jsonl").open() if l.strip()]
print("비용", round(sum(x.get("cost_usd", 0.0) for x in led if x.get("event") == "settle"), 4), "시도", sum(x.get("event") == "reserve" for x in led), "실패", [(x.get("run"), x.get("error")) for x in led if x.get("event") == "fail"])
