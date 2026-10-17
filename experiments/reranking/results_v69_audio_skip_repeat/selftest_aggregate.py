"""aggregate_live 자체 검사 — 답을 손으로 계산해 둔 작은 가짜 실행(임시 폴더)으로 지표·분모·MRR@10 절단·상태·원인 분류와
검증 실패(미완료·부분·중복·누락·순서 불일치·조건 불일치·평가기 요약 불일치)를 확인한다. 모델·DB 호출 없음.

    venv/bin/python selftest_aggregate.py   # 종료 코드 0 = 전부 통과
"""
from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_live as ag  # noqa: E402

failures = 0


def check(name, cond):
    global failures
    print(("ok   " if cond else "FAIL ") + name)
    failures += 0 if cond else 1


def fill(ids, n=30, prefix="f"):
    """앞에 ids를 두고 나머지를 겹치지 않는 채움 곡으로 채운다."""
    out = list(ids)
    i = 0
    while len(out) < n:
        sid = f"{prefix}{i}"
        if sid not in out:
            out.append(sid)
        i += 1
    return out


# 질의 정의: (set, qid, gold, final 순서(앞 10만 쓰임), 후보 30, outcome, group statuses, policy_skip)
def base_queries():
    return [
        ("v06", "a1", ["x1"], fill(["x1"], 12), fill(["x1"]), "applied", ["applied"], ""),
        ("v06", "a2", ["y5"], fill(["p1", "p2", "p3", "p4", "y5"], 12, "g"), fill(["y5"]), "applied", ["applied"], ""),
        # 정답 둘 — w가 3위, z는 12위(top-10 밖) → 첫 정답 3
        ("v06", "a3", ["z", "w"], fill(["p1", "p2", "w"] + [f"h{i}" for i in range(8)] + ["z"], 12, "k"),
         fill(["w", "z"]), "applied", ["applied"], ""),
        # 정답이 11위 → top-10 밖(MRR 0) · 후보에는 있음
        ("v09", "b1", ["g"], fill([f"m{i}" for i in range(10)] + ["g"], 12, "n"), fill(["g"]), "applied", ["applied"], ""),
        # 정답이 후보 밖 · 리랭킹 실패(폴백)
        ("v09", "b2", ["none"], fill([], 12, "r"), fill([], 30, "r"), "fallback", ["failed"], ""),
        # 가사 질의 t003 — 정답 10위(MRR 0.1)
        ("v10", "t003", ["c"], fill([f"s{i}" for i in range(9)] + ["c"], 12, "u"), fill(["c"]), "applied", ["applied"], ""),
        # 정책 생략 — 정답 2위
        ("v10", "c2", ["d"], fill(["e", "d"], 12, "v"), fill(["e", "d"]), "policy_skipped", [], "image_dominant"),
    ]


COUNTS = {"v06": 3, "v09": 2, "v10": 2}


def write_run(root: Path, name: str, queries, *, model="live", status="complete", partial=False, env=None,
              summary_override=None, top10_override=None, dup=None, code_commit=None, audio=None, clap=None,
              path_failures=None, record_name="v69_queries.jsonl", both_records=False, qdrant_url="http://127.0.0.1:6374"):
    d = root / name
    for s in ag.SETS:
        (d / f"{s}_dev").mkdir(parents=True, exist_ok=True)
    # v69: 이름에 A가 있으면 on(소리 켬), 아니면 skip. 두 구성은 같은 코드이고 env는 AUDIO_FUSION_SCALE만 다르다
    code = "on" if "A" in name else "skip"
    cond = {"run": name, "code": code, "rep": 1, "model": model, "status": status,
            "partial": partial, "server_state_before_matches_baseline": True, "server_state_after_matches_baseline": True,
            "env": {**(env or {"CONTEXT_WEIGHT": "0.5", "AUDIO_FUSION_SCALE": "1" if code == "on" else "0"}),
                    "QDRANT_URL": qdrant_url},
            "index_fingerprint": {"content_sha256_16": "x"},
            "lyrics_snapshot": {"content_sha256_16": "y"}, "analysis_caches": {}, "collections": {},
            "code_identity": {"commit": code_commit or "c", "src_py_tree_sha16": "t"}}
    # v69 소리 실행 기록: on은 v06 첫 질의에서 오디오 조회 1회(세트 기록·질의 기록 일치), skip은 0
    n_audio = (1 if code == "on" else 0) if audio is None else audio
    cond["sets_done"] = {s: {"audio_collection_queries": n_audio if s == "v06" else 0,
                             "queries_with_clap_embed": n_audio if s == "v06" else 0,
                             "path_failures": list(path_failures or []) if s == "v06" else []} for s in ag.SETS}
    cond["clap"] = clap if clap is not None else ({"load_calls": 1, "load_ms": [1.0], "embed_calls": n_audio} if code == "on"
                                                  else {"load_calls": 0, "load_ms": [], "embed_calls": 0})
    (d / "run_conditions.json").write_text(json.dumps(cond), encoding="utf-8")
    if status == "complete":
        (d / "DONE").write_text(json.dumps({"partial": partial, "model": model}), encoding="utf-8")
    for s in ag.SETS:
        qs = [q for q in queries if q[0] == s]
        if dup and dup[0] == s:
            qs = qs + [qs[0]]
        detail, harness = [], []
        for (_, qid, gold, final, cands, outcome, groups, skip) in qs:
            gs = set(gold)
            rank = ag.first_rank(final, gs, len(final))
            top10 = final[:10]
            if top10_override and qid in top10_override:
                top10 = top10_override[qid]
            detail.append({"query_id": qid, "relevant_ids": "|".join(gold), "rerank_top_ids": "|".join(top10),
                           "rerank_outcome": outcome, "rerank_rank": rank or "", "rerank_policy_skip": skip,
                           "rerank_error": ""})
            harness.append({"query_id": qid, "candidates": [{"id": c, "score": 1.0 - i / 100} for i, c in enumerate(cands)],
                            "final": {"ids": final}, "lyrics": [], "protection": {"policy_skip": skip},
                            "rerank_runs": [{"input_ids": cands[:20], "input_scores": [0] * 20, "input_lyric": [""] * 20,
                                             "top_k": 30, "status": g} for g in groups],
                            "generates": [{"group": i, "attempt": 0, "contents_sha16": f"p-{qid}-{cands[0]}"}
                                          for i, _ in enumerate(groups)],
                            "audio_queries": n_audio if (s == "v06" and qid == qs[0][1]) else 0})
        with (d / f"{s}_dev" / "search_eval_dev_detail.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(detail[0]))
            w.writeheader()
            w.writerows(detail)
        body = "".join(json.dumps(h) + "\n" for h in harness[: len(set(h["query_id"] for h in harness))])
        (d / f"{s}_dev" / record_name).write_text(body, encoding="utf-8")
        if both_records:
            (d / f"{s}_dev" / "v68_queries.jsonl").write_text(body, encoding="utf-8")
        # 평가기 요약(비율) — 손으로 맞춘 값
        rows = [{"rank": ag.first_rank(final[:10], set(gold), 10)} for (_, _, gold, final, *_rest) in qs]
        n = len(rows)
        summ = {"Hit@1": sum(1 for r in rows if r["rank"] == 1) / n, "Hit@5": sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n,
                "Hit@10": sum(1 for r in rows if r["rank"]) / n, "MRR@10": sum(1 / r["rank"] for r in rows if r["rank"]) / n}
        if summary_override and s in summary_override:
            summ.update(summary_override[s])
        with (d / f"{s}_dev" / "search_eval_dev_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "baseline", "rerank", "delta"])
            for k, v in summ.items():
                w.writerow([k, "", f"{v:.6f}", ""])
    return d


def expected_ids(queries):
    return {s: [q[1] for q in queries if q[0] == s] for s in ag.SETS}


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    qs = base_queries()
    ids = expected_ids(qs)
    X = write_run(root, "B1", qs)
    # Y: a2 정답 2위(입력 같음 → 모델 변동 이득) · b2 실패→적용 정답 4위(폴백 관련 이득) ·
    #    c2 정책 생략 순서만 다름 · a3 후보 집합이 달라지고 정답이 밖으로(입력 변화 손실)
    qy = copy.deepcopy(qs)
    qy[1] = ("v06", "a2", ["y5"], fill(["p1", "y5"], 12, "g"), fill(["y5"]), "applied", ["applied"], "")
    qy[4] = ("v09", "b2", ["none"], fill(["r1", "r2", "r3", "none"], 12, "r"), fill(["none"], 30, "r"), "applied", ["applied"], "")
    qy[6] = ("v10", "c2", ["d"], fill(["d", "e"], 12, "v"), fill(["d", "e"]), "policy_skipped", [], "image_dominant")
    qy[2] = ("v06", "a3", ["z", "w"], fill([f"q{i}" for i in range(12)], 12, "k"), fill(["q0"], 30, "k"), "applied", ["applied"], "")
    Y = write_run(root, "A1", qy)
    res = ag.aggregate({"B1": X, "A1": Y}, [("B1", "A1")], expected_ids=ids, expected_counts=COUNTS)
    m = res["runs"]["B1"]["metrics"]["all"]
    check("분모 7 = 실패 질의 포함", m["queries"] == 7 and m["mrr10_denominator"] == 7)
    check("Hit@1/5/10 = 1/4/5", (m["hit1"], m["hit5"], m["hit10"]) == (1, 4, 5))
    exp_mrr = round((1 + 1 / 5 + 1 / 3 + 1 / 10 + 1 / 2) / 7, 6)
    check(f"MRR@10 = {exp_mrr}(11위·후보 밖은 0, 둘째 정답은 첫 정답만)", m["mrr10"] == exp_mrr)
    check("후보 회수@30 = 6(후보 밖 b2만 빠짐)", m["recall30"] == 6)
    check("상태 applied 5 · failed 1 · policy_skipped 1", m["status"] == {"applied": 5, "failed": 1, "policy_skipped": 1})
    lm = res["runs"]["B1"]["metrics"]["lyric"]
    check("가사 부분집합 = t003 하나(정답 10위)", lm["queries"] == 1 and lm["hit10"] == 1 and lm["hit5"] == 0 and lm["mrr10"] == 0.1)
    check("비가사 6", res["runs"]["B1"]["metrics"]["nonlyric"]["queries"] == 6)
    aux = res["runs"]["B1"]["aux_excluding_failure_touched"]
    check("보조 지표는 실패 1건을 빼고 분모 6", aux["queries"] == 6 and aux["excluded"] == 1)
    p = res["pairs"]["B1:A1"]
    per = {r["query_id"]: r for r in p["per_query"]}
    check("a2: 입력 같고 순위 5→2 → model_variation 이득", per["a2"]["cause"] == "model_variation" and per["a2"]["change"] == "gain")
    check("b2: 실패 쪽 → failure_fallback 이득", per["b2"]["cause"] == "failure_fallback" and per["b2"]["change"] == "gain")
    check("c2: 양쪽 정책 생략 → policy_skip_order 이득(2→1)", per["c2"]["cause"] == "policy_skip_order" and per["c2"]["change"] == "gain")
    check("a3: 후보 집합 변화 → input_change 손실", per["a3"]["cause"] == "input_change" and per["a3"]["change"] == "loss"
          and "cand_set" in per["a3"]["input_change"])
    check("a1·b1·t003 순서 동일", all(per[q]["cause"] == "same" for q in ("a1", "b1", "t003")))
    check("얻음 3 · 잃음 1", sorted(p["gains"]) == ["a2", "b2", "c2"] and p["losses"] == ["a3"])

    def expect_fail(label, fn):
        try:
            fn()
            check(label, False)
        except ag.ValidationError as exc:
            check(f"{label} ({str(exc)[:60]})", True)

    D = write_run(root, "B9", qs, status="failed")
    expect_fail("미완료 실행 거부", lambda: ag.aggregate({"B9": D}, [], expected_ids=ids, expected_counts=COUNTS))
    P = write_run(root, "B8", qs, partial=True)
    expect_fail("부분 실행 거부", lambda: ag.aggregate({"B8": P}, [], expected_ids=ids, expected_counts=COUNTS))
    F = write_run(root, "B7", qs, model="fake")
    expect_fail("live가 아닌 실행 거부", lambda: ag.aggregate({"B7": F}, [], expected_ids=ids, expected_counts=COUNTS))
    U = write_run(root, "B6", qs, dup=("v06",))
    expect_fail("질의 ID 중복 거부", lambda: ag.aggregate({"B6": U}, [], expected_ids=ids, expected_counts={**COUNTS, "v06": 4}))
    expect_fail("세트 건수 불일치 거부", lambda: ag.aggregate({"B1": X}, [], expected_ids=ids, expected_counts={**COUNTS, "v09": 3}))
    missing_ids = {**ids, "v10": ["t003", "c9"]}
    expect_fail("질의 누락·목록 불일치 거부", lambda: ag.aggregate({"B1": X}, [], expected_ids=missing_ids, expected_counts=COUNTS))
    T = write_run(root, "B5", qs, top10_override={"a1": fill(["zz"], 10)})
    expect_fail("detail top-10 ≠ 기록된 최종 순서 거부", lambda: ag.aggregate({"B5": T}, [], expected_ids=ids, expected_counts=COUNTS))
    S = write_run(root, "B4", qs, summary_override={"v06": {"Hit@1": 0.0}})
    expect_fail("평가기 요약과 집계 불일치 거부", lambda: ag.aggregate({"B4": S}, [], expected_ids=ids, expected_counts=COUNTS))
    S2 = write_run(root, "B3", qs, summary_override={"v10": {"MRR@10": 0.6}})
    expect_fail("MRR@10 요약 불일치 거부(절단 확인)", lambda: ag.aggregate({"B3": S2}, [], expected_ids=ids, expected_counts=COUNTS))
    E = write_run(root, "A2", qs, env={"CONTEXT_WEIGHT": "0", "AUDIO_FUSION_SCALE": "1"})
    expect_fail("실행 조건(env, 소리 배율 밖) 불일치 거부", lambda: ag.aggregate({"B1": X, "A2": E}, [], expected_ids=ids, expected_counts=COUNTS))
    W = write_run(root, "A3", qs, env={"CONTEXT_WEIGHT": "0.5", "AUDIO_FUSION_SCALE": "0"})
    expect_fail("on 구성인데 AUDIO_FUSION_SCALE=0 거부", lambda: ag.aggregate({"B1": X, "A3": W}, [], expected_ids=ids, expected_counts=COUNTS))
    W2 = write_run(root, "B2x", qs, env={"CONTEXT_WEIGHT": "0.5", "AUDIO_FUSION_SCALE": "1"})
    expect_fail("skip 구성인데 AUDIO_FUSION_SCALE=1 거부", lambda: ag.aggregate({"B2x": W2}, [], expected_ids=ids, expected_counts=COUNTS))
    C = write_run(root, "A4", qs, code_commit="other")
    expect_fail("구성 사이 코드 지문 다름 거부", lambda: ag.aggregate({"B1": X, "A4": C}, [], expected_ids=ids, expected_counts=COUNTS))

    # ---- v69 추가: 허용 목록 밖 상태 · 소리 실행 기록 · 기록 파일 이름 · 시간대별 env
    for outcome, groups, label in (("unknown", ["unknown"], "unknown"), ("skipped", [], "skipped"),
                                   ("applied", ["model_partial"], "그룹 상태 model_partial")):
        qbad = copy.deepcopy(qs)
        qbad[0] = ("v06", "a1", ["x1"], fill(["x1"], 12), fill(["x1"]), outcome, groups, "")
        Bd = write_run(root, f"B_bad_{outcome}_{len(groups)}", qbad)
        expect_fail(f"허용 목록 밖 결말·상태({label}) 거부", lambda: ag.aggregate({"Bd": Bd}, [], expected_ids=ids, expected_counts=COUNTS))
    qps = copy.deepcopy(qs)
    qps[6] = ("v10", "c2", ["d"], fill(["e", "d"], 12, "v"), fill(["e", "d"]), "applied", ["applied"], "image_dominant")
    PS = write_run(root, "B_ps", qps)
    expect_fail("정책 생략 사유가 있는데 결말이 applied → 거부", lambda: ag.aggregate({"PS": PS}, [], expected_ids=ids, expected_counts=COUNTS))
    SA = write_run(root, "B_audio", qs, audio=1)
    expect_fail("skip인데 오디오 조회 기록 → 거부", lambda: ag.aggregate({"SA": SA}, [], expected_ids=ids, expected_counts=COUNTS))
    SC = write_run(root, "B_clap", qs, clap={"load_calls": 1, "load_ms": [1.0], "embed_calls": 0})
    expect_fail("skip인데 CLAP 로드 1 → 거부", lambda: ag.aggregate({"SC": SC}, [], expected_ids=ids, expected_counts=COUNTS))
    OA = write_run(root, "A_noaudio", qs, audio=0, clap={"load_calls": 0, "load_ms": [], "embed_calls": 0})
    expect_fail("on인데 오디오 조회 0(v67 계수기 재귀 같은 경우) → 거부", lambda: ag.aggregate({"OA": OA}, [], expected_ids=ids, expected_counts=COUNTS))
    PF = write_run(root, "A_pathfail", qs, path_failures=["audio: RuntimeError: x"])
    expect_fail("경로 실패 기록 → 거부", lambda: ag.aggregate({"PF": PF}, [], expected_ids=ids, expected_counts=COUNTS))
    BR = write_run(root, "B_both", qs, both_records=True)
    expect_fail("v68·v69 기록이 함께 있으면 거부", lambda: ag.aggregate({"BR": BR}, [], expected_ids=ids, expected_counts=COUNTS))
    V68 = write_run(root, "B_v68name", qs, record_name="v68_queries.jsonl", qdrant_url="http://127.0.0.1:6372")
    ok68 = ag.aggregate({"B1": X, "V68": V68}, [("V68", "B1")], expected_ids=ids, expected_counts=COUNTS)
    check("v68 기록 이름·다른 포트(QDRANT_URL) 실행은 같이 읽고 결과 동일",
          ok68["pairs"]["V68:B1"]["final_top10_identical"] == 7
          and ok68["conditions"]["env_per_period"]["V68"]["QDRANT_URL"].endswith(":6372"))

print("FAILURES", failures)
sys.exit(1 if failures else 0)
