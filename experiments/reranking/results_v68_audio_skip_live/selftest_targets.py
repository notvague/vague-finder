"""analyze_targets 자체 검사 — 답을 아는 가짜 질의별 표(실제 질의 ID 176, v67 목록은 실제 파일). 모델·DB 없음."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V67 = HERE.parent / "results_v67_candidate_recall_diagnosis"
sys.path.insert(0, str(HERE))
import analyze_targets as at  # noqa: E402

ids = [r["query_id"] for r in csv.DictReader(open(V67 / "analysis/sound_types.csv", encoding="utf-8"))]
cases = {  # A1, A2, B1, B2
    "t058": ("", "", "3", "3"),      # 관측 이득
    "n018": ("2", "2", "", ""),      # 관측 손실
    "t077": ("", "", "4", ""),       # B 반복 변동 → variation
    "t010": ("5", "5", "2", "2"),    # 순위 오름
    "c706": ("1", "1", "3", "3"),    # 순위 내림
    "q203": ("12", "12", "11", "11"),  # 둘 다 밖
    "t042": ("2", "3", "2", "2"),    # A 반복 위치 변동(Hit은 같음) → position variation
    "t107": ("", "4", "4", "4"),     # A1 실패 폴백 → failure_touched(이득으로 세지 않음)
}
FAILED = {("t107", "A1")}
fails = []


def check(n, c):
    print(("ok   " if c else "FAIL ") + n)
    if not c:
        fails.append(n)


with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "per_query.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        cols = ["query_id"] + [f"{a}_{k}" for a in ("rank", "status", "cand_rank") for k in ("A1", "A2", "B1", "B2")] + ["cause_A1:B1", "cause_A2:B2"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for q in ids:
            r = cases.get(q, ("1", "1", "1", "1"))
            w.writerow({"query_id": q, **{f"rank_{k}": v for k, v in zip(("A1", "A2", "B1", "B2"), r)},
                        **{f"status_{k}": ("failed" if (q, k) in FAILED else "applied") for k in ("A1", "A2", "B1", "B2")},
                        **{f"cand_rank_{k}": "1" for k in ("A1", "A2", "B1", "B2")}, "cause_A1:B1": "input_change", "cause_A2:B2": "input_change"})
    out = Path(tmp) / "t.json"
    rc = subprocess.run([sys.executable, str(HERE / "analyze_targets.py"), "--per-query", str(p), "--v67", str(V67), "--out", str(out)],
                        capture_output=True, text=True).returncode
    res = json.loads(out.read_text())
    a = res["all"]
    check("실행 성공", rc == 0)
    check("관측 이득 = t058", a["gain_ids"] == ["t058"])
    check("관측 손실 = n018", a["loss_ids"] == ["n018"])
    check("반복 변동 = t077", a["variation_ids"] == ["t077"])
    check("순위 오름 t010 · 내림 c706", a["position_up_ids"] == ["t010"] and a["position_down_ids"] == ["c706"])
    check("A 반복 위치 변동은 오름·내림에 안 듦(t042)", "t042" not in a["position_up_ids"] + a["position_down_ids"] and a["position"].get("variation") == 1)
    check("Hit@10 합계 A1 = 176 − 4(t058·t077·q203·t107 밖)", a["hit@10"]["A1"] == 172)
    check("실패 닿은 질의 = t107(이득 아님)", a["failure_touched_ids"] == ["t107"] and "t107" not in a["gain_ids"])
    check("새 후보 5 묶음: 이득 1 · 변동 1", res["gained5"]["verdict"].get("gain") == 1 and res["gained5"]["verdict"].get("variation") == 1)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    # A2 없는 표(미완료 실행 제외): A1 하나로 판정, a_repeat_missing 표시
    noa2 = Path(tmp) / "noa2.csv"
    keep = [c for c in rows[0] if not c.endswith("_A2")]
    with open(noa2, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keep); w.writeheader(); w.writerows([{c: r[c] for c in keep} for r in rows])
    rc = subprocess.run([sys.executable, str(HERE / "analyze_targets.py"), "--per-query", str(noa2), "--v67", str(V67), "--out", str(out)],
                        capture_output=True, text=True).returncode
    r2 = json.loads(out.read_text())
    check("A2 없음: 실행 성공·A2 열 없음", rc == 0 and "A2" not in r2["all"]["hit@10"])
    check("A2 없음: n018 손실 · t058 이득 · t107 실패 닿음", r2["all"]["loss_ids"] == ["n018"] and r2["all"]["gain_ids"] == ["t058"]
          and r2["all"]["failure_touched_ids"] == ["t107"])
    check("A2 없음: a_repeat_missing 표시", r2["gained5"]["per_query"]["t058"]["a_repeat_missing"] is True)
    bad = Path(tmp) / "bad.csv"
    with open(bad, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows[:-1])
    rc = subprocess.run([sys.executable, str(HERE / "analyze_targets.py"), "--per-query", str(bad), "--v67", str(V67), "--out", str(out)],
                        capture_output=True, text=True).returncode
    check("질의 누락(175) 거부", rc != 0)
print(f"FAILURES {len(fails)}")
sys.exit(1 if fails else 0)
