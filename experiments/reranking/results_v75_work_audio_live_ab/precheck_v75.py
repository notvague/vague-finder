"""v75 실행 전 무료 검증 집계 — 가짜 리랭커 기준 실행(`offline/full/work_{on,skip}_full_warm`)과 입력 대조 파일(`precheck/*.json`)만 읽는다.
모델·DB 호출 없음. 기준값은 사용자 지시(2026-10-20)와 v74 결과에서 가져와 **이 파일에 고정**한다 — 결과에 맞춰 바꾸지 않는다.
(v73 `precheck_v73.py`는 작품명 OFF/ON 비교용이라 v75에서는 새로 썼다.)

검사(모두 참이어야 유료 호출을 시작한다):
  P1 A 입력 = v74 work_on 반복 1 = v73 B 가짜 기준(work_full_warm) = v73 3차 실제 B1·B2 전송 입력(각 176/176)
  P2 B 입력 = v74 work_skip 반복 1(176/176)
  P3 후보 회수@30 A 125 · B 130 · B가 얻음 5(t058·t077·t078·t100·t111) · 잃음 0
  P4 A: CLAP 임베딩 질의 114 = 오디오 조회 질의 114 · B: 둘 다 0
  P5 작품명 상태(두 구성 모두): 판정 밖 154 · 채택 16 · 보류 5 · 검증 거부 1 · 실패·캐시 누락·시간 초과 0 · 실제 해석 호출 0 · 경로 실패 0
  P6 A↔B 내부 입력 변화 114(기존 성공 74) · 실제 전송 프롬프트 변화 109(기존 성공 72) · 점수만 다른 5질의
      (기존 성공 = v73 3차 실제 B1·B2 모두 Hit@10, 121건 — `results_v73_work_name_full_ab/attempt3/analysis/per_query.csv`)
  P7 A↔B 미호출 그룹 0 · 재시도 프롬프트 변화 0

    venv/bin/python precheck_v75.py   → analysis/precheck_v75.json (종료 코드 0 = 모두 통과, 1 = 실패 항목 있음)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
V73_PER_QUERY = ROOT / "experiments/reranking/results_v73_work_name_full_ab/attempt3/analysis/per_query.csv"
RUNS = {"on": HERE / "offline/full/work_on_full_warm", "skip": HERE / "offline/full/work_skip_full_warm"}
CMP = HERE / "precheck"
EXPECT = {
    "recall": {"on": 125, "skip": 130}, "gained": ["t058", "t077", "t078", "t100", "t111"], "lost": [],
    "clap": {"on": 114, "skip": 0}, "audio": {"on": 114, "skip": 0},
    "work_status": {"not_triggered": 154, "applied": 16, "abstained": 5, "no_usable_work": 1},
    "input_changed": 114, "input_changed_base_hits": 74, "prompt_changed": 109, "prompt_changed_base_hits": 72,
    "score_only": ["n002", "q210", "t041", "t043", "t044"], "base_hits": 121,
}
IDENTICAL = {  # 파일 → 설명
    "v74_work_on_rep1__vs_v75_work_on.json": "A = v74 work_on 반복 1",
    "v73_work_full_warm__vs_v75_work_on.json": "A = v73 B 가짜 기준(work_full_warm)",
    "v75_work_on__vs_v73_live_B1.json": "A = v73 3차 실제 B1 전송 입력",
    "v75_work_on__vs_v73_live_B2.json": "A = v73 3차 실제 B2 전송 입력",
    "v74_work_skip_rep1__vs_v75_work_skip.json": "B = v74 work_skip 반복 1",
}


def relevant() -> dict:
    out = {}
    for s in SETS:
        with open(ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    out[r["query_id"].strip()] = {"set": s, "rel": {x.strip() for x in r["relevant_ids"].split("|") if x.strip()}}
    return out


def load(run: Path) -> dict:
    if not (run / "DONE").exists():
        raise SystemExit(f"완료되지 않은 실행: {run}")
    rows = {}
    for s in SETS:
        for line in (run / f"{s}_dev/v75_queries.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[r["query_id"]] = r
    return rows


def first_rank(ids: list, rel: set):
    return next((i + 1 for i, x in enumerate(ids) if x in rel), None)


def main() -> int:
    rel = relevant()
    if len(rel) != 176:
        raise SystemExit(f"dev 질의 {len(rel)} ≠ 176")
    runs = {k: load(v) for k, v in RUNS.items()}
    conds = {k: json.loads((v / "run_conditions.json").read_text(encoding="utf-8")) for k, v in RUNS.items()}
    with open(V73_PER_QUERY, encoding="utf-8") as fh:
        v73 = {r["qid"]: r for r in csv.DictReader(fh)}
    hit = lambda r, k: r[k].isdigit() and int(r[k]) <= 10  # noqa: E731  ("" · ">10" · ">30" = 10위 밖)
    base_hits = {q for q, r in v73.items() if hit(r, "B1_rank") and hit(r, "B2_rank")}
    b1_or_b2 = {q for q, r in v73.items() if hit(r, "B1_rank") or hit(r, "B2_rank")}
    checks: dict = {}
    problems: list = []

    def check(name: str, ok: bool, detail) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            problems.append(name)

    # P1·P2 입력 동일
    for f, desc in IDENTICAL.items():
        c = json.loads((CMP / f).read_text(encoding="utf-8"))
        check(f"P1P2 {desc}", c["queries"] == 176 and c["input_identical"] == 176 and not c["input_diff_queries"]
              and not c["path_diff_query_ids"] and not c["missing_either"],
              {"identical": c["input_identical"], "diff": len(c["input_diff_queries"]), "path_diff": len(c["path_diff_query_ids"])})
    # P3 후보 회수
    rank = {k: {q: first_rank([x["id"] for x in r["candidates"]], rel[q]["rel"]) for q, r in rows.items()} for k, rows in runs.items()}
    recall = {k: sum(1 for v in rank[k].values() if v) for k in runs}
    gained = sorted(q for q in rel if rank["skip"][q] and not rank["on"][q])
    lost = sorted(q for q in rel if rank["on"][q] and not rank["skip"][q])
    both = [q for q in rel if rank["on"][q] and rank["skip"][q]]
    down = sorted((q, rank["on"][q], rank["skip"][q]) for q in both if rank["skip"][q] > rank["on"][q])
    up = sorted((q, rank["on"][q], rank["skip"][q]) for q in both if rank["skip"][q] < rank["on"][q])
    check("P3 후보 회수@30 A 125 · B 130", recall == EXPECT["recall"], recall)
    check("P3 B 얻음 5 · 잃음 0", gained == EXPECT["gained"] and lost == EXPECT["lost"], {"gained": gained, "lost": lost})
    # P4 경로 실행
    clap = {k: sum(1 for r in rows.values() if r["clap_embed"]) for k, rows in runs.items()}
    audio = {k: sum(1 for r in rows.values() if r["audio_queries"] > 0) for k, rows in runs.items()}
    check("P4 CLAP 임베딩 질의 A 114 · B 0", clap == EXPECT["clap"], clap)
    check("P4 오디오 조회 질의 A 114 · B 0", audio == EXPECT["audio"], audio)
    # P5 작품명 상태
    for k, rows in runs.items():
        st = Counter((r.get("work_status") or "(none)") for r in rows.values())
        check(f"P5 작품명 상태 {k}", dict(st) == EXPECT["work_status"], dict(st))
        c = conds[k]
        check(f"P5 실제 해석 호출 0 · 해석기 캐시 · 상태 위반 0 · 경로 실패 0 · 외부 연결 0 ({k})",
              c["work_resolver"].startswith("cache:") and not c["work_status_violations"] and not c["network_attempts"]
              and all(not v["path_failures"] for v in c["sets_done"].values()),
              {"resolver": c["work_resolver"].split(":")[0], "violations": len(c["work_status_violations"]),
               "network_attempts": len(c["network_attempts"]),
               "path_failures": sum(len(v["path_failures"]) for v in c["sets_done"].values()),
               "warmup": c["warmup"]})
    # P6 입력·프롬프트 변화
    c = json.loads((CMP / "v75_work_on__vs_work_skip.json").read_text(encoding="utf-8"))
    changed = set(c["input_diff_queries"])
    prompt = set(c["diff_queries"]["prompts"])
    score_only = sorted(changed - prompt)
    check("P6 기존 성공 = v73 B1·B2 모두 Hit@10 121건(B1·B2 같음)", len(base_hits) == EXPECT["base_hits"] and base_hits == b1_or_b2,
          {"both": len(base_hits), "either": len(b1_or_b2)})
    detail6 = {"input_changed": len(changed), "input_changed_base_hits": len(changed & base_hits),
               "prompt_changed": len(prompt), "prompt_changed_base_hits": len(prompt & base_hits),
               "score_only": score_only, "score_only_base_hits": sorted(set(score_only) & base_hits)}
    check("P6 내부 입력 변화 114(기존 성공 74) · 전송 프롬프트 변화 109(기존 성공 72) · 점수만 5",
          (detail6["input_changed"], detail6["input_changed_base_hits"], detail6["prompt_changed"],
           detail6["prompt_changed_base_hits"], score_only)
          == (EXPECT["input_changed"], EXPECT["input_changed_base_hits"], EXPECT["prompt_changed"],
              EXPECT["prompt_changed_base_hits"], EXPECT["score_only"]), detail6)
    check("P7 A↔B 미호출 그룹 0 · 재시도 프롬프트 변화 0", not c["groups_not_sent"] and not c["retry_prompt_changed_queries"],
          {"groups_not_sent": len(c["groups_not_sent"]), "retry_prompt_changed": len(c["retry_prompt_changed_queries"])})
    # 관심 사례 고정(실행 전) — 판정에 쓰지 않고 결과 보고에만 쓴다
    down_base = [d for d in down if d[0] in base_hits]
    watch = {
        "new_pool_5": EXPECT["gained"],
        "work_gain_v73": ["n033", "n043", "n096"],
        "common_gain_c707": ["c707"],
        "past_audio_skip_loss_q215": ["q215"],
        "rank_down_all": [d[0] for d in down],
        "rank_down_base_hits": [d[0] for d in down_base],
        "in_pool_fail_4": ["q203", "q205", "c705", "t044"],
    }
    groups = {"prompt_changed": sorted(prompt), "score_only": score_only, "input_same": sorted(set(rel) - changed)}
    shape = {k: {"generate_calls_fake": sum(len(r["generates"]) for r in rows.values()),
                 "groups": sum(len(r["rerank_runs"]) for r in rows.values()),
                 "model_queries": sum(1 for r in rows.values() if r["rerank_runs"]),
                 "policy_skipped": sum(1 for r in rows.values() if (r["protection"] or {}).get("policy_skip")),
                 "max_groups_per_query": max(len(r["rerank_runs"]) for r in rows.values()),
                 "max_prompt_chars": max((g["contents_chars"] for r in rows.values() for g in r["generates"]), default=0),
                 "sum_prompt_chars": sum(g["contents_chars"] for r in rows.values() for g in r["generates"]),
                 "qdrant_logical_calls": sum(len(r["qdrant_seqs"]) for r in rows.values())}
             for k, rows in runs.items()}
    out = {"pass": not problems, "problems": problems, "checks": checks,
           "candidate_rank_changes_skip_vs_on": {"up": len(up), "down": len(down), "down_detail": down,
                                                  "down_base_hits": down_base},
           "watch_cases(실행 전 고정)": watch, "input_groups": {k: len(v) for k, v in groups.items()},
           "input_group_ids": groups, "base_hits_v73_B": sorted(base_hits), "fake_run_shape": shape}
    (HERE / "analysis").mkdir(exist_ok=True)
    (HERE / "analysis/precheck_v75.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "problems": problems, "p6": detail6, "rank": out["candidate_rank_changes_skip_vs_on"],
                      "shape": shape}, ensure_ascii=False, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
