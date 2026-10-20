"""work_label_tool 자체 검사 — 가짜 질의·가짜 라벨·가짜 모델 출력만 쓴다(실제 원문·출력 읽지 않음, 외부 호출 0).

조건이 다른 입력(덜 채운 표·고친 원문·덮어쓰기·해시 변조·판정 대기 이름·노출 unsure·CP949 저장)을 일부러 넣는다.

    venv/bin/python experiments/reranking/results_v71_work_name_labels/selftest_work_label_tool.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import work_label_tool as T  # noqa: E402

FAILS: List[str] = []


def ok(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def must_exit(fn, msg: str) -> None:
    try:
        fn()
    except SystemExit as e:
        if e.code in (0, None):
            FAILS.append(f"{msg}: 0으로 끝남")
        return
    FAILS.append(f"{msg}: 막히지 않음")


def write(path: Path, cols, rows, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)


def fill(row: Dict[str, str], status: str, intended: str = "", possible: str = "", **kw: str) -> Dict[str, str]:
    base = {"judgment_status": status, "intended_work": intended, "possible_works": possible,
            "work_type": "드라마" if status in ("단일", "복수") else "", "query_phrase": "",
            "external_check": "no", "sources": "", "confidence": "중간" if status in ("단일", "복수") else "",
            "labeler_id": "XX", "labeled_at": "2026-10-20", "saw_model_output_before": "no",
            "saw_answer_song_before": "no", "saw_experiment_results_before": "no", "saw_v67_work_list_before": "no",
            "open_questions": "질의만으로 못 정함" if status == "판단불가" else ""}
    if status in ("단일", "복수"):
        base["query_phrase"] = row["query"].split()[0]
    base.update(kw)
    return {**row, **base}


def scenario_incomplete(root: Path) -> None:
    """받은 그대로 보존(오류 포함) → 기계적 정리 → 임시 고정 → 확인 대기 행을 세지 않음 · 에이전트 보조 판정 분리."""
    priv, log = root / "p2", root / "log2.jsonl"
    common = ["--private-dir", str(priv), "--freeze-log", str(log)]
    T.main([*common, "prepare", "--source", str(root / "src.csv"), "--tracked-queue", str(root / "tracked.csv"),
            "--manifest", str(root / "m2.json")])
    sheet = T.read_csv(priv / "packet" / "labeler" / "work_label_sheet_labeler.csv")
    imap = {r["item_no"]: r["qid"] for r in T.read_csv(priv / "item_map.csv")}
    q2i = {v: k for k, v in imap.items()}
    rows = []
    for r in sheet:
        q = imap[r["item_no"]]
        if q == "t041":
            row = fill(r, "", open_questions="단서 없음")         # 상태 빈칸 → incomplete
        elif q == "c602":
            row = fill(r, "복수", "작품C", "작품씨")               # 의도 작품이 가능 목록에 없음 → needs_confirmation
        elif q == "q105":
            row = fill(r, "작품 아님")                            # 공백 표기 → 기계적 정리 대상
        elif q == "n043":
            row = fill(r, "단일", "작품 에프")                      # 모델과 표기 다름 → 에이전트 의견만 있음
        else:
            row = fill(r, "판단불가")
        row["labeler_id"] = ""
        row["labeled_at"] = ""
        rows.append(row)
    raw = root / "raw.csv"
    with open(raw, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(T.SHEET_COLS) + ["", ""])
        for r in rows:
            w.writerow([r.get(c, "") + (" " if c == "notes" else "") for c in T.SHEET_COLS] + ["", ""])
        w.writerow([""] * (len(T.SHEET_COLS) + 2))
    ok(T.main([*common, "freeze", "--role", "labeler", "--sheet", str(raw)]) == 1, "오류 표가 정식 고정됨")
    ok(T.main([*common, "freeze", "--role", "labeler", "--sheet", str(raw), "--provisional",
                      "--ai-assisted"]) == 0, "받은 그대로 보존 실패")
    e1 = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    ok(e1["provisional"] and e1["kind"] == "as_received" and e1["labeler_ids"] == [] and e1["ai_assisted"], "보존 기록")
    work = root / "work.csv"
    T.main([*common, "normalize", "--sheet", str(raw), "--out", str(work)])
    ch = T.read_csv(root / "work_changes.csv")
    kinds = {c["change"] for c in ch}
    ok({"drop_empty_column", "drop_blank_row", "status_spacing", "strip_whitespace"} <= kinds, f"정리 종류: {kinds}")
    wrows = {r["item_no"]: r for r in T.read_csv(work)}
    ok(wrows[q2i["q105"]]["judgment_status"] == "작품아님", "상태 공백 정리")
    ok(wrows[q2i["t041"]]["judgment_status"] == "", "빈 상태를 채움")
    ok(wrows[q2i["c602"]]["possible_works"] == "작품씨", "작품명을 고침")
    must_exit(lambda: T.main([*common, "normalize", "--sheet", str(raw), "--out", str(work)]), "작업용 사본 덮어씀")
    T.main([*common, "freeze", "--role", "labeler", "--sheet", str(work), "--provisional", "--kind",
            "mechanical_cleanup", "--reason", "기계적 정리"])
    e2 = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    ok(e2["revision"] == 2 and e2["rows_not_valid"] == {q2i["t041"]: "incomplete", q2i["c602"]: "needs_confirmation"},
       f"행 상태: {e2['rows_not_valid']}")
    ok(e2["ai_assisted"] is True, "AI 보조 판에서 나온 정리본의 AI 표시가 사라짐")
    model_csv, flags_csv = root / "model.csv", root / "flags.csv"
    ok(T.main([*common, "compare", "--model-outputs", str(model_csv), "--review-flags", str(flags_csv),
               "--no-checker", "--revision", "2"]) == 0, "2판으로 2단계 열기")
    pairs = T.read_csv(priv / "name_pairs.csv")
    for p in pairs:
        if T.norm_name(p["name_a"]) == T.norm_name("작품 에프"):
            p["agent_suggestion"], p["agent_basis"] = "no", "에이전트 보조 — 사람 판정 아님"
        if {T.norm_name(p["name_a"]), T.norm_name(p["name_b"])} == {T.norm_name("작품C"), T.norm_name("Work C")}:
            p["same_work"], p["decided_by"] = "yes", "ZZ"
    write(priv / "name_pairs.csv", T.PAIR_COLS, pairs)
    out = root / "s2.json"
    T.main([*common, "summarize", "--review-flags", str(flags_csv), "--out", str(out), "--revision", "2",
            "--v73-per-query", str(root / "v73.csv")])
    s = json.loads(out.read_text(encoding="utf-8"))
    sp = s["special_checks"]
    ok(sp["t041"]["valid_rows__human_pairs"]["outcome"] == "label_incomplete", "빈 상태 행이 집계됨")
    ok(sp["t041"]["as_written__human_pairs"]["outcome"] == "label_incomplete", "빈 상태 행을 그대로 판정")
    g = s["v72_v73_stable_gains"]["n043"]
    ok(g["valid_rows__human_pairs"]["outcome"] == "pending_name_pair", "에이전트 의견이 사람 판정으로 쓰임")
    ok(g["valid_rows__agent_assisted_pairs"]["outcome"] == "wrong", "에이전트 보조 집계에서 안 쓰임")
    per = {r["qid"]: r for r in T.read_csv(priv / "stage2" / "per_item_outcome__valid_rows__human_pairs.csv")}
    ok(per["c602"]["outcome"] == "label_needs_confirmation", "확인 대기 행이 주 결과에 들어감")
    per_w = {r["qid"]: r for r in T.read_csv(priv / "stage2" / "per_item_outcome__as_written__human_pairs.csv")}
    ok(per_w["c602"]["outcome"] == "correct", f"적힌 그대로 집계: {per_w['c602']['outcome']}")
    ok(per["q105"]["outcome"] == "wrong_link", "정리된 상태로 판정")
    ok(s["labeler_revision"] == 2 and s["checker"] is False and s["labeler_ai_assisted"] is True, "판·확인자·AI 보조 기록")


def unit_checks(root: Path) -> None:
    # 사람 판정이 표기 동일보다 먼저다(동명 다른 작품)
    P = T.Pairs([{"pair_id": "P1", "item_no": "L01", "name_a": "작품X", "name_b": "작품 X", "same_work": "no",
                  "decided_by": "ZZ"}])
    ok(P.same("L01", "작품X", "작품X") is False, "사람의 '다른 작품' 판정이 표기 동일에 밀림")
    ok(P.same("L02", "작품X", "작품 X") is True, "판정 없는 표기 동일")
    ok(P.same("L02", "작품X", "작품Y") is None, "판정 없는 다른 표기")
    A = T.Pairs([{"pair_id": "P1", "item_no": "L01", "name_a": "작품X", "name_b": "작품Y", "same_work": "",
                  "agent_suggestion": "yes"}], use_agent=True)
    ok(A.same("L01", "작품X", "작품Y") is True and T.Pairs(
        [{"pair_id": "P1", "item_no": "L01", "name_a": "작품X", "name_b": "작품Y", "same_work": "",
          "agent_suggestion": "yes"}]).same("L01", "작품X", "작품Y") is None, "에이전트 의견 분리")
    # 셀 안 줄바꿈은 한 칸 안의 두 작품이다
    f = root / "newline.csv"
    write(f, ("item_no", "judgment_status", "intended_work", "possible_works"),
          [{"item_no": "L01", "judgment_status": "복수", "intended_work": "", "possible_works": "작품A\n작품B"},
           {"item_no": "L02", "judgment_status": "단일", "intended_work": "작품C\r\n작품D", "possible_works": ""}])
    rows = T.read_csv(f)
    ok(len(rows) == 2, f"셀 안 줄바꿈이 행을 쪼갬: {len(rows)}")
    ok(T.label_names(rows[0]) == ["작품A", "작품B"], f"복수 작품이 합쳐짐: {T.label_names(rows[0])}")
    ok(len(T.split_multi(rows[1]["intended_work"])) == 2, "단일 칸의 두 작품을 못 봄")
    raw = f.read_bytes().decode("utf-8")
    table = list(csv.reader(io.StringIO(raw, newline="")))
    norm, _ = T.normalize_rows(table[0], table[1:])
    ok("\n" in norm[0]["possible_works"], "정리 과정에서 셀 안 줄바꿈이 사라짐")
    # 날짜 구분자만 고친다(형식이 다른 값은 그대로 둔다)
    norm, ch = T.normalize_rows(["item_no", "labeled_at"], [["L01", "2026.10.18"], ["L02", "10/18"]])
    ok(norm[0]["labeled_at"] == "2026-10-18" and norm[1]["labeled_at"] == "10/18", f"날짜 정리: {norm}")
    ok([c["change"] for c in ch] == ["date_separator"], f"날짜 정리 기록: {ch}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        priv, log = root / "private", root / "freeze_log.jsonl"
        common = ["--private-dir", str(priv), "--freeze-log", str(log)]
        src = root / "src.csv"
        tracked = root / "tracked.csv"
        write(src, ("set", "qid", "query", "v67_known_case"),
              [{"set": "vX", "qid": q, "query": f"가짜 질의 {q} 드라마 이야기", "v67_known_case": "yes"} for q in T.EXPECTED_QIDS])
        write(tracked, ("set", "qid"), [{"set": "vX", "qid": q} for q in T.EXPECTED_QIDS])
        man = root / "manifest.json"
        ok(T.main([*common, "prepare", "--source", str(src), "--tracked-queue", str(tracked), "--manifest", str(man)]) == 0,
           "prepare 실패")
        must_exit(lambda: T.main([*common, "prepare", "--source", str(src), "--tracked-queue", str(tracked),
                                  "--manifest", str(man)]), "prepare 재실행이 대응표를 덮어씀")
        blank = priv / "packet" / "labeler" / "work_label_sheet_labeler.csv"
        head = blank.read_text(encoding="utf-8-sig").splitlines()[0]
        ok("qid" not in head and "v67" not in head.replace("saw_v67_work_list_before", "") and "model" not in
           head.replace("saw_model_output_before", ""), "가린 표에 질의 id·v67·모델 열이 있다")
        sheet = T.read_csv(blank)
        ok([r["item_no"] for r in sheet] == [f"L{i:02d}" for i in range(1, 23)], "항목 번호")
        imap = {r["item_no"]: r["qid"] for r in T.read_csv(priv / "item_map.csv")}
        ok(list(imap.values()) != list(T.EXPECTED_QIDS), "섞이지 않음")
        ok(not (priv / "stage2").exists(), "준비 단계에 모델 출력 폴더가 생김")
        qid2item = {v: k for k, v in imap.items()}

        # 라벨: 질의별로 다른 경우를 만든다
        plan = {
            "q105": ("작품아님", "", ""), "n031": ("단일", "작품N", ""), "m103": ("복수", "", "작품A; 작품B"),
            "c602": ("단일", "작품 C", ""), "c603": ("판단불가", "", ""), "c607": ("단일", "작품D", ""),
            "n033": ("단일", "작품E?", ""), "n043": ("단일", "작품F", ""), "n096": ("판단불가", "", ""),
            "c707": ("단일", "작품G", ""), "t043": ("단일", "작품H", ""), "n010": ("작품아님", "", ""),
        }
        rows = []
        for r in sheet:
            q = imap[r["item_no"]]
            st, i, p = plan.get(q, ("판단불가", "", ""))
            rows.append(fill(r, st, i, p))
        # 노출 unsure 한 행 → 독립 아님
        rows[[r["item_no"] for r in rows].index(qid2item["c607"])]["saw_model_output_before"] = "unsure"
        cols = T.SHEET_COLS
        good = root / "labeler_filled.csv"
        write(good, cols, rows)
        rep = T.check_sheet(T.read_csv(good), sheet)
        ok(not rep["errors"] and rep["complete"], f"정상 표가 오류: {rep['errors'][:3]}")

        # 라벨러가 전달용 표에 바로 채워도 검사 기준은 빈 표 원본이다(빈 행·원문 변경을 놓치지 않는다)
        inplace = [dict(r) for r in rows]
        inplace[0]["query"] = inplace[0]["query"] + " 고침"
        write(blank, cols, inplace + [{c: "" for c in cols}], encoding="utf-8-sig")
        rep = T.check_sheet(T.read_csv(blank), T._packet_rows(priv))
        codes = {c for _, c in rep["errors"]}
        ok("query_changed" in codes and "unknown_item" in codes, f"전달용 표를 기준으로 씀: {codes}")
        sheet = T._packet_rows(priv)

        # 일부러 틀린 표
        bad = [dict(r) for r in rows]
        bad[0]["query"] = bad[0]["query"] + " 고침"
        bad[1].update(judgment_status="단일", intended_work="")
        bad[2].update(judgment_status="복수", possible_works="하나만", intended_work="")
        bad[3].update(judgment_status="판단불가", intended_work="억지 작품")
        bad[4].update(external_check="yes", sources="")
        bad[5].update(saw_answer_song_before="몰라")
        bad[6].update(labeled_at="10/20")
        bad[7].update(judgment_status="단일", intended_work="작품", query_phrase="질의에 없는 구절", work_type="드라마",
                      confidence="높음")
        bad[8].update(judgment_status="")
        bad.append(dict(bad[9]))
        rep = T.check_sheet(bad, sheet)
        codes = {c for _, c in rep["errors"]}
        for want in ("query_changed", "single_without_intended_work", "multiple_needs_two_or_more_works",
                     "undeterminable_with_intended_work", "external_check_without_sources",
                     "bad_saw_answer_song_before", "bad_labeled_at", "status_missing_but_row_touched", "duplicate_item"):
            ok(want in codes, f"검출 못함: {want}")
        ok(("L08" if False else bad[7]["item_no"], "query_phrase_not_verbatim") in {tuple(w) for w in rep["warnings"]},
           "근거 구절 불일치 경고 없음")
        ok(json.dumps(rep, ensure_ascii=False).find("가짜 질의") < 0, "검사 결과에 질의 원문이 찍힘")

        # 미완성 표는 고정 안 됨
        partial = [dict(r) for r in rows]
        for r in partial[:3]:
            for c in T.LABEL_COLS:
                r[c] = ""
        pf = root / "partial.csv"
        write(pf, cols, partial)
        ok(T.main([*common, "freeze", "--role", "labeler", "--sheet", str(pf)]) == 1, "미완성 표가 고정됨")

        # 2단계는 고정 전에 막힘
        model_csv = root / "model.csv"
        mcols = ("qid", "decision", "work_index", "name", "aliases", "media_type", "evidence", "accepted",
                 "reject_reason", "mention_songs", "name_in_query")
        mrows = []
        model_plan = {
            "q105": [("가짜구절", "Fake Alias", "True", "")],              # 라벨 작품아님 → wrong_link
            "n031": [("작품N", "", "True", "")],                          # 정답
            "m103": [("작품A", "", "True", ""), ("작품Z", "", "True", "")],  # 복수 라벨, 일부만 → partial
            "c602": [("Work C", "작품C", "True", "")],                    # 별칭이 띄어쓰기만 다름 → 정답
            "c607": [("작품 디", "", "True", "")],                        # 표기 다름 → 사람 판정 대기
            "n033": [("작품E", "", "True", "")],                          # 물음표만 다름 → 정답
            "n043": [("작품Y", "작품F", "True", "")],                     # 별칭만 라벨과 같고 사람이 다르다고 판정 → 오답
            "c707": [("작품G", "", "True", ""), ("작품Q", "", "True", "")],  # 단일 라벨 + 여분 → partial
            "t043": [("작품H", "", "False", "no_local_data")],             # 거부된 쪽에 라벨 작품 → missed_rejected
            "n096": [("아무거나", "", "True", "")],                       # 라벨 판단불가 → 세지 않음
        }
        for q in T.EXPECTED_QIDS:
            if q in model_plan:
                for i, (n, a, acc, rr) in enumerate(model_plan[q], 1):
                    mrows.append({"qid": q, "decision": "candidates", "work_index": i, "name": n, "aliases": a,
                                  "media_type": "drama", "evidence": "", "accepted": acc, "reject_reason": rr,
                                  "mention_songs": 1, "name_in_query": "False"})
            else:
                mrows.append({"qid": q, "decision": "abstain", "work_index": "", "name": "", "aliases": "",
                              "media_type": "", "evidence": "", "accepted": "", "reject_reason": "",
                              "mention_songs": "", "name_in_query": ""})
        write(model_csv, mcols, mrows)
        flags_csv = root / "flags.csv"
        write(flags_csv, ("qid", "v67_known_case", "v71_status", "review_flags"),
              [{"qid": q, "v67_known_case": "yes" if q.startswith("c") else "no", "v71_status": "applied",
                "review_flags": ""} for q in T.EXPECTED_QIDS])
        must_exit(lambda: T.main([*common, "compare", "--model-outputs", str(model_csv), "--review-flags",
                                  str(flags_csv)]), "고정 전 2단계가 열림")

        ok(T.main([*common, "freeze", "--role", "labeler", "--sheet", str(good)]) == 0, "정상 표 고정 실패")
        r1 = next((priv / "frozen").glob("labeler_r1_*.csv"))
        last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        ok(last["ai_assisted"] is False, "표시 없는 고정이 AI 보조로 기록됨")
        ok("XX" not in log.read_text(encoding="utf-8") and len(last["labeler_ids"]) == 1, "고정 기록에 작성자 원문이 남음")
        ok(not os.access(r1, os.W_OK), "고정본이 쓰기 가능")
        must_exit(lambda: T.main([*common, "freeze", "--role", "labeler", "--sheet", str(good)]),
                  "사유 없는 두 번째 고정이 허용됨")
        ok(T.main([*common, "freeze", "--role", "labeler", "--sheet", str(good), "--reason", "오타"]) == 0, "수정본 고정")
        ok(r1.exists() and len(list((priv / "frozen").glob("labeler_r*.csv"))) == 2, "최초 고정본이 사라짐")

        # 확인자 없으면 --no-checker 없이 못 염
        must_exit(lambda: T.main([*common, "compare", "--model-outputs", str(model_csv), "--review-flags",
                                  str(flags_csv)]), "확인자 없이 2단계가 열림")
        # 확인자: 하나는 판단불가, 하나는 다른 작품, CP949로 저장
        crow = [dict(r) for r in rows]
        idx = {r["item_no"]: i for i, r in enumerate(crow)}
        crow[idx[qid2item["n031"]]] = fill(sheet[idx[qid2item["n031"]]], "판단불가")
        crow[idx[qid2item["c602"]]] = fill(sheet[idx[qid2item["c602"]]], "단일", "전혀다른작품")
        for r in crow:
            r["labeler_id"] = "YY"
        cf = root / "checker.csv"
        write(cf, cols, crow, encoding="cp949")
        ok(T.main([*common, "freeze", "--role", "checker", "--sheet", str(cf)]) == 0, "CP949 확인자 표 고정 실패")
        agree_out = root / "agree.json"
        T.main([*common, "agree", "--out", str(agree_out)])
        ag = json.loads(agree_out.read_text(encoding="utf-8"))["agreement_counts"]
        ok(ag.get("one_undeterminable", 0) >= 1, f"한쪽 판단불가 분류 없음: {ag}")
        ok(ag.get("pending_name_pair", 0) >= 1, f"다른 이름이 사람 판정 대기로 안 감: {ag}")

        ok(T.main([*common, "compare", "--model-outputs", str(model_csv), "--review-flags", str(flags_csv)]) == 0,
           "2단계 열기 실패")
        v73 = root / "v73.csv"
        write(v73, ("qid", "group", "A_cand_rank", "B_cand_rank", "class"),
              [{"qid": q, "group": "changed13" if q in ("n033", "c707") else "same163",
                "A_cand_rank": ">30" if q == "n033" else "5", "B_cand_rank": "3",
                "class": "stable_gain" if q in ("n033", "c707") else "same_hit"} for q in T.EXPECTED_QIDS])
        out = root / "summary.json"
        summ = [*common, "summarize", "--review-flags", str(flags_csv), "--out", str(out), "--v73-per-query", str(v73)]
        ok(T.main(summ) == 2, "판정 대기 이름이 있는데 0으로 끝남")
        s = json.loads(out.read_text(encoding="utf-8"))
        ok(s["pending_name_pair_items"]["valid_rows__human_pairs"], "판정 대기 항목")
        pairs = T.read_csv(priv / "name_pairs.csv")
        ok(not any(T.norm_name(p["name_b"]) == T.norm_name("Fake Alias") for p in pairs), "별칭마다 쌍이 따로 올라감")
        alias_rows = [p for p in pairs if "별칭" in p.get("notes", "")]
        ok({T.norm_name(p["name_b"]) for p in alias_rows} >= {T.norm_name("Work C"), T.norm_name("작품Y")},
           "별칭만 같은 쌍이 사람 판정 없이 확정됨")
        ok(not any({T.norm_name(p["name_a"]), T.norm_name(p["name_b"])} == {T.norm_name("작품B"), T.norm_name("작품A")}
                   for p in pairs), "이미 맞은 모델 작품이 다른 라벨 이름과 쌍으로 올라감")

        # 사람이 이름 쌍을 판정한다: c607 같은 작품, 그 밖은 다른 작품
        for p in pairs:
            names = {T.norm_name(p["name_a"]), T.norm_name(p["name_b"])}
            p["same_work"] = "yes" if names in ({T.norm_name("작품D"), T.norm_name("작품 디")},
                                                {T.norm_name("작품 C"), T.norm_name("Work C")}) else "no"
            p["decided_by"] = "ZZ"
        write(priv / "name_pairs.csv", T.PAIR_COLS, pairs)
        ok(T.main(summ) == 0, "판정 뒤 집계 실패")
        s = json.loads(out.read_text(encoding="utf-8"))
        per = {r["qid"]: r for r in T.read_csv(priv / "stage2" / "per_item_outcome__valid_rows__human_pairs.csv")}
        expect = {"q105": "wrong_link", "n031": "correct", "m103": "partial", "c602": "correct", "c607": "correct",
                  "n033": "correct", "n043": "wrong", "c707": "partial", "t043": "missed_rejected",
                  "n096": "label_undeterminable", "c603": "label_undeterminable", "n010": "correct_abstain"}
        for q, want in expect.items():
            ok(per[q]["outcome"] == want, f"{q}: {per[q]['outcome']} != {want}")
        ok(per["t043"]["rejected_contains_label_work"] == "True", "거부 작품 속 라벨 작품 표시")
        ok(per["m103"]["accepted_wrong_works"] == "1" and per["m103"]["accepted_has_label_work"] == "True", "단계별 표시")
        prim = s["valid_rows__human_pairs"]
        ok(prim["all_22"]["not_counted"].get("label_undeterminable", 0) >= 2, "판단불가가 분모에 섞임")
        ok(prim["self_reported_independent_rows"]["items"] == 21, "노출 unsure 행이 독립 라벨에 들어감")
        g = s["v72_v73_stable_gains"]
        ok(g["n096"]["valid_rows__human_pairs"]["counts_as_interpretation_evidence"] == "unknown", "n096 판단불가 처리")
        ok(g["n043"]["valid_rows__human_pairs"]["counts_as_interpretation_evidence"] == "no", "n043 오답 연결 처리")
        ok(g["n033"]["valid_rows__human_pairs"]["counts_as_interpretation_evidence"] == "yes", "n033 정답 처리")
        ok(g["c707"]["valid_rows__human_pairs"]["counts_as_interpretation_evidence"] == "no", "c707 여분 연결 처리")
        ok(prim["other_12_not_negatives"]["items"] + prim["known_10"]["items"] == 22, "알려진/그 밖 분할")
        ok("labeler_vs_checker" in s, "확인자 대조 없음")
        ok(s["checker_independent"] is True, "2단계 전 확인자 고정본이 독립 확인으로 안 잡힘")
        sc = s["search_cross"]["valid_rows__human_pairs"]["per_query"]
        ok(sc["n033"] == {"interpretation": "correct", "candidate_recall": "gain", "hit10": "stable_gain",
                          "rerank_input": "changed13"}, f"검색 교차 n033: {sc['n033']}")
        ok(sc["c707"]["interpretation"] == "wrong_included", "검색 교차 c707")
        ok(s["model_stages"]["queries_all_rejected"] == 1 and s["model_stages"]["works_generated"] == 12, "모델 단계 수")
        ok("가짜 질의" not in out.read_text(encoding="utf-8") and "작품G" not in out.read_text(encoding="utf-8"),
           "추적 요약에 질의 원문·작품명")

        # 모델 출력 사본이 최초 공개 뒤 바뀌면 집계·재공개를 멈춘다
        opened_before = (priv / "stage2" / "opened.json").read_text(encoding="utf-8")
        copy = priv / "stage2" / "model_output_review_v71.csv"
        original = copy.read_bytes()
        os.chmod(copy, 0o644)
        copy.write_bytes(original.replace("가짜구절,Fake Alias,drama,,True".encode(), "가짜구절,Fake Alias,drama,,False".encode()))
        ok(copy.read_bytes() != original, "변조 준비 실패")
        must_exit(lambda: T.main(summ), "바뀐 모델 출력으로 집계됨")
        copy.write_bytes(original)
        ok(T.main(summ) == 0, "원래 출력으로 다시 집계 실패")
        other = root / "model_changed.csv"
        other.write_bytes(model_csv.read_bytes().replace("가짜구절,Fake Alias,drama,,True".encode(),
                                                          "가짜구절,Fake Alias,drama,,False".encode()))
        must_exit(lambda: T.main([*common, "compare", "--model-outputs", str(other), "--review-flags", str(flags_csv)]),
                  "다른 모델 출력으로 2단계를 다시 엶")
        ok(T.main([*common, "compare", "--model-outputs", str(model_csv), "--review-flags", str(flags_csv)]) == 0,
           "같은 출력으로 다시 열기 실패")
        ok((priv / "stage2" / "opened.json").read_text(encoding="utf-8") == opened_before, "최초 공개 기록이 덮임")
        ok(len((priv / "stage2" / "compare_runs.jsonl").read_text(encoding="utf-8").splitlines()) == 2, "재실행 기록")

        # 2단계를 연 뒤의 고정은 기록에 표시된다
        T.main([*common, "freeze", "--role", "checker", "--sheet", str(cf), "--reason", "2단계 뒤 수정"])
        last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        ok(last["after_stage2_opened"] is True and last["revision"] == 2, "2단계 뒤 고정 표시 없음")

        # 고정본 변조는 막는다
        os.chmod(r1, 0o644)
        r1.write_text(r1.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
        must_exit(lambda: T.main(summ), "변조된 고정본으로 집계됨")

        scenario_incomplete(root)
        unit_checks(root)

    if FAILS:
        print("FAIL", len(FAILS))
        for f in FAILS:
            print(" -", f)
        return 1
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
