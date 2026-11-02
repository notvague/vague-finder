"""§5 작품명 ON · 소리 경로 생략 구성 판정. 모델 호출 없음(저장 응답만 읽는다)."""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
load = lambda arm: {r["query_id"]: r for r in json.loads((HERE / f"runs/{arm}/rows.json").read_text(encoding="utf-8"))}
off, on, na = load("off"), load("on"), load("on_noaudio")
queries = {q["query_id"]: q for q in json.loads((HERE / "queries.json").read_text(encoding="utf-8"))}
recs = {}
for line in (HERE / "runs/on_noaudio/work_resolution.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line); recs[r.get("query") or ""] = r

def visible_text(body):
    """analyze.py의 H5 범위(요약·경로·안내)에 #102의 단서 대조와 보정 설명을 더한다."""
    res = body.get("results", [])
    exp = [t.get("explain") or {} for t in res]
    ue = [e.get("user_explanation") or {} for e in exp]
    return json.dumps({
        "s": [e.get("summary") for e in exp] + [u.get("summary") for u in ue],
        "clues": [u.get("clue_checks") for u in ue], "notes": [u.get("adjustment_notes") for u in ue],
        "paths": [p for e in exp for p in e.get("paths") or []],
        "notice": body.get("notice"), "clarify": body.get("clarify"),
    }, ensure_ascii=False)

leak, checked = [], 0
for qid, q in queries.items():
    rec = recs.get(q["query"]) or {}
    names = [n for w in rec.get("works") or [] for n in [w.get("name"), *(w.get("aliases") or [])] if n]
    body = json.loads((HERE / f"runs/on_noaudio/resp_{qid}.json").read_text(encoding="utf-8"))
    text = visible_text(body)
    # 곡 자료에 원래 있는 이름은 H5 기준대로 제외한다
    song_text = json.dumps([{k: v for k, v in t.items() if k != "explain"} for t in body.get("results", [])], ensure_ascii=False)
    for n in names:
        checked += 1
        if n in text and n not in q["query"] and n not in song_text:
            leak.append((qid, n))

tr = [q for q, r in queries.items() if r["expected_status"] != "not_triggered"]
nt = [q for q in queries if q not in tr]
delta = lambda rows, base: round(st.median(na[q]["seconds"] - base[q]["seconds"] for q in rows), 2)
status_bad = [q for q in tr if (recs.get(queries[q]["query"]) or {}).get("resolver_error")
              or (recs.get(queries[q]["query"]) or {}).get("resolver_status") not in ("candidates", "abstain")]
out = {
    "H1": all(r["http"] == 200 and not r["analysis_fallback"] for r in na.values()),
    "H2": not status_bad,
    "H3": all(r["rerank_groups_failed"] == 0 and r.get("reorder_stage") != "failed" for r in na.values()),
    "H4": max(r["seconds"] for r in na.values()) <= 30 and delta(tr, off) <= 5,
    "H5": not leak,
    "max_seconds": max(r["seconds"] for r in na.values()),
    "median_delta_vs_off": {"triggered": delta(tr, off), "not_triggered": delta(nt, off)},
    "median_delta_vs_on": {"triggered": delta(tr, on), "not_triggered": delta(nt, on)},
    "names_checked": checked, "leak": leak,
    "resolver_status": {q: (recs.get(queries[q]["query"]) or {}).get("resolver_status") for q in tr},
}
(HERE / "summary_on_noaudio.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
