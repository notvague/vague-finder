"""v67 진단·비교 도구 자체 검사 — 손으로 답을 아는 가짜 입력 + 거부해야 하는 입력. 모델·DB·네트워크 없음.

    venv/bin/python selftest_tools.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_ab  # noqa: E402
import compare_runs  # noqa: E402
import summarize_diagnosis as sd  # noqa: E402

PASSED, FAILED = [], []


def check(name: str, cond: bool) -> None:
    (PASSED if cond else FAILED).append(name)
    print(("ok   " if cond else "FAIL ") + name)


def refuses(name: str, fn) -> None:
    try:
        fn()
    except (sd.DiagnosisError, SystemExit):
        check(name, True)
        return
    check(name, False)


def stage_row(q, sid, stage, pool="", bd="", rep="1", rc="1"):
    return {"query_id": q, "song_id": sid, "stage": stage, "pool_rank": pool, "boosted_rank_deep": bd, "representative": rep,
            "relevant_count": rc}


def review_row(q, p, s="", lever="", ev="근거"):
    return {"query_id": q, "primary": p, "secondary": s, "confidence": "확인", "lever": lever, "evidence": ev}


# ---------------------------------------------------------------- summarize_diagnosis
base_stage = [
    stage_row("q1", "a", "후보 안", pool="30"),                        # 경계 30 = 후보 안 → K
    stage_row("q2", "b", "경로 밖", bd="30"),                          # 경계 30 = P
    stage_row("q3", "c", "경로 밖", bd="31"),                          # 31 = 사람 판정
    stage_row("q4", "d", "융합 절단", bd="99"),
    # 정답 둘: 하나는 경로 밖, 하나는 보정 뒤 절단 → 대표는 보정 뒤 절단 곡
    stage_row("q5", "e1", "경로 밖", bd="5", rep="0", rc="2"), stage_row("q5", "e2", "보정 뒤 절단", bd="40", rc="2"),
]
base_review = [review_row("q1", "K"), review_row("q2", "P", "A", "depth_attr|title_rule"), review_row("q3", "R", "", "cover_generic"),
               review_row("q4", "F", "", "audio_crowding"), review_row("q5", "F", "P", "audio_crowding")]
res = sd.summarize(base_stage, base_review, v50_fail={"q1", "q2", "q3", "q9"})
check("diag: 실패 5 = 후보 밖 4 + 후보 내 1", (res["failures"], res["out_of_pool"], res["in_pool"]) == (5, 4, 1))
check("diag: 주 원인 합계(겹침 없음) F2·P1·R1 / K1", res["primary_out_of_pool"] == {"F": 2, "P": 1, "R": 1} and res["primary_in_pool"] == {"K": 1})
check("diag: 보조 원인은 따로(합계에 안 들어감)", res["secondary_counts_overlapping"] == {"A": 1, "P": 1})
check("diag: 지렛대 겹침 집계 audio_crowding 2", res["levers_overlapping"].get("audio_crowding:out") == 2)
check("diag: 정답 여럿 질의 목록", res["multi_gold_failures"] == ["q5"])
check("diag: v50 대비 같음 3·새 q4,q5·해결 q9", res["v50_compare"] == {"same": 3, "new": ["q4", "q5"], "resolved": ["q9"]})

r = copy.deepcopy(base_review); r.pop()
refuses("diag 거부: 검토 누락", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review) + [review_row("q1", "K")]
refuses("diag 거부: 검토 중복", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review) + [review_row("q8", "U")]
refuses("diag 거부: 검토 여분", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[2]["primary"] = "Z"
refuses("diag 거부: 모르는 코드", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[0]["primary"] = "F"
refuses("diag 거부: 후보 안인데 K 아님", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[2]["primary"] = "K"
refuses("diag 거부: 후보 밖인데 K", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[1]["primary"] = "R"
refuses("diag 거부: 가산 뒤 30(=P)인데 사람 판정 코드", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[2]["primary"] = "P"
refuses("diag 거부: 가산 뒤 31인데 P", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[3]["primary"] = "R"
refuses("diag 거부: 융합 절단인데 F 아님", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[2]["secondary"] = "R"
refuses("diag 거부: 주 = 보조", lambda: sd.summarize(base_stage, r))
r = copy.deepcopy(base_review); r[2]["evidence"] = " "
refuses("diag 거부: 근거 없음", lambda: sd.summarize(base_stage, r))
s = copy.deepcopy(base_stage); s[4]["representative"], s[5]["representative"] = "1", "0"
refuses("diag 거부: 대표 곡이 가장 멀리 간 정답이 아님", lambda: sd.summarize(s, base_review))
s = copy.deepcopy(base_stage); s[5]["representative"] = "1"; s[4]["representative"] = "1"
refuses("diag 거부: 대표 줄 둘", lambda: sd.summarize(s, base_review))
s = copy.deepcopy(base_stage); s[0]["pool_rank"] = "31"
refuses("diag 거부: 후보 순위 31", lambda: sd.summarize(s, base_review))
s = copy.deepcopy(base_stage); s[2]["pool_rank"] = "4"
refuses("diag 거부: 경로 밖인데 후보 순위", lambda: sd.summarize(s, base_review))

# ---------------------------------------------------------------- analyze_ab helpers
check("ab: first_rank 정답 여럿이면 가장 앞", analyze_ab.first_rank(["x", "b", "a"], ["a", "b"]) == 2)
check("ab: first_rank 30위 경계", analyze_ab.first_rank([f"s{i}" for i in range(1, 31)], ["s30"]) == 30)
check("ab: first_rank 없음 → None", analyze_ab.first_rank(["x"], ["a"]) is None)
check("ab: jaccard 같음 1 · 반 겹침 1/3", analyze_ab.jaccard(["a", "b"], ["b", "a"]) == 1.0
      and abs(analyze_ab.jaccard(["a", "b"], ["b", "c"]) - 1 / 3) < 1e-9)


# ---------------------------------------------------------------- compare_runs(가짜 실행 폴더)
def fake_run(root: Path, name: str, mutate=None) -> Path:
    d = root / name
    for s, n in compare_runs.EXPECTED.items():
        (d / f"{s}_dev").mkdir(parents=True)
        rows = []
        for i in range(n):
            rows.append({"query_id": f"{s}_{i}", "candidates": [{"id": "1", "score": 0.5, "lyric": "", "lyric_score": None}],
                         "protection": {"protected_ids": [], "tiers": {}, "policy_skip": ""},
                         "rerank_runs": [{"input_ids": ["1"], "input_scores": [0.5], "input_lyric": [""]}],
                         "generates": [{"contents_sha16": "p"}]})
        if mutate:
            mutate(s, rows)
        with open(d / f"{s}_dev" / "v67_queries.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    (d / "DONE").write_text("{}")
    return d


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    a = fake_run(root, "a")
    b = fake_run(root, "b")

    def change_score(s, rows):
        if s == "v09":
            rows[3]["candidates"][0]["score"] = 0.51
            rows[3]["generates"][0]["contents_sha16"] = "q"
    c = fake_run(root, "c", change_score)
    res = compare_runs.compare(compare_runs.load_run(a), compare_runs.load_run(b))
    check("cmp: 같은 입력 176/176", res["all_inputs_same"] == 176)
    res = compare_runs.compare(compare_runs.load_run(a), compare_runs.load_run(c))
    check("cmp: 점수·프롬프트 하나 다름 → 175 · scores 1 · prompts 1 · 집합 0",
          res["all_inputs_same"] == 175 and res["differs"]["candidate_scores"]["count"] == 1
          and res["differs"]["prompts"]["count"] == 1 and res["differs"]["candidate_set"]["count"] == 0)
    d = fake_run(root, "d", lambda s, rows: rows.append(dict(rows[0])) if s == "v06" else None)
    refuses("cmp 거부: 질의 중복/건수", lambda: compare_runs.load_run(d))
    e = fake_run(root, "e"); (e / "DONE").unlink()
    refuses("cmp 거부: DONE 없음(미완료)", lambda: compare_runs.load_run(e))

print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} 통과")
sys.exit(1 if FAILED else 0)
