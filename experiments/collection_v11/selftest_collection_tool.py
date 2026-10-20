"""collection_tool 자체 검사 — 전부 합성 자료(가짜 코퍼스·가짜 기존 세트·손으로 만든 질의). 모델·검색·외부 호출 없음.

임시 폴더에서 돌리고(`init --allow-unsafe-location`), 끝나면 지운다. 실제 보관자 폴더·실제 질의는 쓰지 않는다.

    venv/bin/python experiments/collection_v11/selftest_collection_tool.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collection_tool as ct  # noqa: E402

fails: List[str] = []
results: List[str] = []


def check(name: str, cond: bool) -> None:
    line = ("ok   " if cond else "FAIL ") + name
    print(line)
    results.append(line)
    if not cond:
        fails.append(name)


def run(*argv: str) -> tuple:
    """CLI를 그대로 부른다. (종료 코드 또는 거부 메시지, 화면 출력)"""
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            code = ct.main(list(argv))
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else f"refused: {e.code}"
    return code, buf.getvalue() + err.getvalue()


def refused(res: tuple) -> bool:
    return isinstance(res[0], str) and res[0].startswith("refused")


TMP = Path(tempfile.mkdtemp(prefix="vf_collect_selftest_"))
P = TMP / "keeper"
REPO = TMP / "repo"
CORPUS = TMP / "corpus.jsonl"

# 가짜 코퍼스: 101~120. 앨범 A는 101·102를 함께 담는다
with open(CORPUS, "w", encoding="utf-8") as f:
    for i in range(101, 121):
        album = "앨범A" if i in (101, 102) else f"앨범{i}"
        f.write(json.dumps({"song_id": str(i), "metadata": {"title": f"가짜제목{i}", "artist": [f"가짜가수{i}"],
                                                            "album": album, "release_date": "2010.01.01"},
                            "semantic_analysis": {"summary": "읽으면 안 되는 필드"}}, ensure_ascii=False) + "\n")
# 가짜 기존 세트: 정답 110, 문장 하나
(REPO / "docs/eval").mkdir(parents=True)
(REPO / "experiments/reranking").mkdir(parents=True)
(REPO / "docs/eval/queries.json").write_text(json.dumps({"queries": [
    {"query": "비 오는 날 듣던 남자 발라드인데 피아노 반주가 예뻤던 노래", "positives": ["110"]}]}, ensure_ascii=False),
    encoding="utf-8")
with open(REPO / "experiments/reranking/eval_queries_x.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "split", "query_type", "query", "relevant_ids"])
    w.writerow(["x1", "dev", "text", "체육대회 때 틀던 신나는 여자 그룹 댄스곡", "111"])

QUERIES = {}  # 상태 출력에 원문이 새지 않는지 볼 때 쓴다


def wrow(eid: str, q: str, **kw) -> Dict[str, str]:
    r = {c: "" for c in ct.WRITER_COLS}
    r.update({"entry_id": eid, "writer_code": eid[:3], "track": "general", "written_at": "2026-10-21 10:00",
              "query": q, "memory_source": "own_memory", "ai_used": "no", "written_before_lookup": "yes",
              "used_system_before": "no", "title_recall": "none", "artist_recall": "partial",
              "remembered_aspects": "melody;era", "work_reference": "no", "writer_confidence": "medium"})
    r.update(kw)
    QUERIES[eid] = r["query"]
    return r


CROWS: Dict[str, Dict[str, str]] = {}


def crow(eid: str, ids: str = "", status: str = "confirmed", in_corpus: str = "yes", **kw) -> Dict[str, str]:
    r = {c: "" for c in ct.CHECKER_COLS}
    r.update({"entry_id": eid, "answer_status": status, "answer_ids": ids, "multiple_answers": "no",
              "in_corpus": in_corpus, "evidence": "metadata", "confidence": "high", "wrong_clue_types": "none",
              "checker_code": "P09", "checked_at": "2026-10-22 10:00", "saw_search_results": "no"})
    r.update(kw)
    CROWS[eid] = r
    return r


def drow(code: str, **kw) -> Dict[str, str]:
    r = {c: "" for c in ct.DECL_COLS}
    r.update({"person_code": code, "declared_at": "2026-10-21 09:00", "relation": "acquaintance",
              "involved_in_search_dev": "no", "saw_existing_eval_queries": "no", "saw_dev_results": "no",
              "saw_failure_lists": "no", "used_system_before": "no", "consent": "yes", "consent_at": "2026-10-21 09:00"})
    r.update(kw)
    return r


n_file = [0]


def sheet(rows: List[Dict[str, str]], cols) -> Path:
    n_file[0] += 1
    path = TMP / "inbox" / f"f{n_file[0]:03d}.csv"
    ct.write_csv(path, cols, rows, bom=True)
    return path


def intake(kind: str, code: str, rows, cols, *extra: str) -> tuple:
    return run("intake", "--private-dir", str(P), "--kind", kind, "--code", code, "--file", str(sheet(rows, cols)),
               *extra)


C = ("--corpus", str(CORPUS))
R = ("--repo", str(REPO))

try:
    # ---- 위치 거부 ----
    git_tree = TMP / "gitrepo"
    (git_tree / ".git").mkdir(parents=True)
    check("Git 작업 트리 안 보관자 폴더 거부", refused(run("init", "--private-dir", str(git_tree / "keeper"))))
    check("/tmp 아래 보관자 폴더 거부(시험 플래그 없이)", refused(run("init", "--private-dir", "/tmp/vf_should_not_exist")))
    check("거부된 위치에 폴더를 만들지 않음", not Path("/tmp/vf_should_not_exist").exists())
    check("시험 위치는 플래그로만 허용", run("init", "--private-dir", str(P), "--allow-unsafe-location")[0] == 0)

    # ---- salt 약속 ----
    check("salt 약속", run("commit-salt", "--private-dir", str(P))[0] == 0)
    check("salt 두 번째 약속 거부(다시 뽑기 차단)", refused(run("commit-salt", "--private-dir", str(P))))

    # ---- 자기 보고 ----
    for code in ("P01", "P02", "P03"):
        intake("declaration", code, [drow(code)], ct.DECL_COLS)
    intake("declaration", "P04", [drow("P04", saw_dev_results="yes", relation="team", involved_in_search_dev="yes")],
           ct.DECL_COLS)

    # ---- 작성자 표 ----
    p1 = [
        wrow("P01-001", "고등학교 축제 때 다 같이 떼창하던 밴드 노래였는데 후렴이 구호 같았어"),
        wrow("P01-002", "축제에서 떼창하던 밴드 곡인데 무대에서 관객이 따라 부르던 기억"),            # 같은 곡(101)
        wrow("P01-003", "비 오는 날 듣던 남자 발라드인데 피아노 반주가 예뻤던 노래요"),               # 기존 문장과 거의 같음
        wrow("P01-004", "엄마가 차에서 자주 틀던 오래된 트로트였는데 제목이 기억 안 나"),            # 코퍼스 밖
        wrow("P01-005", "드라마 주인공이 우산 쓰고 걷던 장면에 깔리던 노래",
             work_reference="yes", work_name_after_writing="가짜 드라마 하나", work_media_type="drama"),
        wrow("P01-006", "그 노래 뭐더라",                                                           # 정답 판단 불가 + 짧음
             remembered_aspects="unsure"),
        wrow("P01-007", "원곡도 있고 리메이크도 있는 노래인데 둘 다 들어 봄"),                       # 복수 정답 102|103
        wrow("P01-008", "체육대회 때 틀던 노래 같은데 기억이 흐릿한 남자 솔로", ),                   # 기존 정답 곡 111
        wrow("P01-009", "AI한테 다듬어 달라고 한 문장: 몽환적인 신스팝 여성 보컬", ai_used="edited"),
        wrow("P01-010", "줄이 바뀐 질의\n두 번째 줄도 원문이다"),                                   # 셀 안 줄바꿈
    ]
    check("작성자 표 받기", intake("writer", "P01", p1, ct.WRITER_COLS)[0] == 0)
    p1_path = TMP / "inbox" / f"f{n_file[0]:03d}.csv"
    check("같은 파일 다시 받기는 기록을 늘리지 않음", "already_received" in run(
        "intake", "--private-dir", str(P), "--kind", "writer", "--code", "P01", "--file", str(p1_path))[1]
          and len(ct.read_log(ct.Private(P).intake_log)) == 5)
    p1b = [dict(r, writer_notes="고침") for r in p1[:1]]
    check("이미 받은 항목을 새 파일로 다시 받기 거부(수정본 절차 강제)",
          refused(intake("writer", "P01", p1b, ct.WRITER_COLS)))

    p2 = [
        wrow("P02-001", "버스에서 이어폰으로 듣던 잔잔한 어쿠스틱 곡인데 휘파람이 나왔어", writer_answer_title="가짜제목104",
             answer_noted_at="2026-10-21 10:30"),
        wrow("P02-002", "앨범 표지가 파란 바다 사진이었던 여자 솔로 곡"),                            # 105
        wrow("P02-003", "드라마 우산 장면 노래 맞나 싶은데 남자 듀엣이었던 것 같아",
             work_reference="yes", work_name_after_writing="가짜드라마하나", work_media_type="drama"),  # 같은 작품(정규화)
        wrow("P02-004", "친구가 물어봐서 같이 찾은 노래, 첫 질문", conversation_id="c1", turn_no="1",
             turn_origin="real_exchange"),
        wrow("P02-005", "남자 목소리였고 2010년쯤이었어", conversation_id="c1", turn_no="2", turn_origin="real_exchange",
             prior_question="가수가 남자야 여자야?"),
    ]
    check("두 번째 작성자 표 받기", intake("writer", "P02", p2, ct.WRITER_COLS)[0] == 0)

    bad = [
        wrow("P03-001", "시각이 빠진 질의", written_at=""),
        wrow("P03-002", "모르는 갈래 값", track="sound_only"),
        wrow("P03-003", "정답을 먼저 보고 적었는데 기억 먼저라고 표시", writer_answer_title="가짜제목106",
             answer_noted_at="2026-10-21 09:00"),
        wrow("P03-004", "대화 두 번째 턴인데 질문이 없다", conversation_id="c9", turn_no="2", turn_origin="constructed"),
    ]
    intake("writer", "P03", bad, ct.WRITER_COLS)
    check("항목 번호가 빈 행이 있는 표 거부",
          refused(intake("writer", "P04", [wrow("P04-001", "정상 행"), dict(wrow("P04-002", "번호 없음"), entry_id="")],
                         ct.WRITER_COLS)))
    check("한 파일 안 항목 번호 중복 거부",
          refused(intake("writer", "P04", [wrow("P04-001", "하나"), wrow("P04-001", "둘")], ct.WRITER_COLS)))
    check("다른 사람 코드로 낸 표 거부", refused(intake("writer", "P04", [wrow("P05-001", "남의 번호")], ct.WRITER_COLS)))
    check("열이 빠진 표는 보존 전에 거부",
          refused(intake("writer", "P04", [wrow("P04-001", "열 빠짐")], [c for c in ct.WRITER_COLS if c != "ai_used"])))
    intake("writer", "P04", [wrow("P04-001", "팀원이 쓴 질의, 개발 결과를 봤다고 보고함")], ct.WRITER_COLS)
    cp = TMP / "inbox" / "cp949.csv"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(ct.WRITER_COLS))
    w.writeheader()
    w.writerow(wrow("P06-001", "엑셀이 CP949로 다시 저장한 표의 한글 질의"))
    cp.write_bytes(buf.getvalue().encode("cp949"))
    check("엑셀 CP949 저장본 받기·한글 보존", run("intake", "--private-dir", str(P), "--kind", "writer", "--code", "P06",
                                             "--file", str(cp))[0] == 0
          and ct.current_rows(ct.Private(P), "writer")[0]["P06-001"]["query"] == QUERIES["P06-001"])

    # ---- 확인자 표 ----
    check("보관자 폴더 밖 확인자 표 거부(사본 추적)",
          refused(run("make-checker-sheet", "--private-dir", str(P), "--writer", "P01", "--out", str(TMP / "x.csv"))))
    out_sheet = P / "packets" / "checker_P01.csv"
    run("make-checker-sheet", "--private-dir", str(P), "--writer", "P01", "--out", str(out_sheet))
    sheet_rows = ct.read_csv(out_sheet)
    check("확인자 표에 검색 결과·점수 열 없음", not any(c in sheet_rows[0] for c in ("rank", "score", "candidates", "flags")))
    check("확인자 표는 한 작성자 항목만", {r["entry_id"][:3] for r in sheet_rows} == {"P01"} and len(sheet_rows) == 10)
    check("확인자 표 다시 만들기는 덮어쓰지 않음", refused(run("make-checker-sheet", "--private-dir", str(P),
                                                       "--writer", "P01", "--out", str(out_sheet))))
    pk = P / "packets" / "P01"
    run("make-writer-packet", "--private-dir", str(P), "--code", "P01", "--rows", "5", "--out-dir", str(pk))
    first_pk = ct.read_csv(next(pk.glob("writer_sheet_P01_*.csv")))
    check("작성자 빈 표는 받은 번호 다음부터 채움", first_pk[0]["entry_id"] == "P01-011" and len(first_pk) == 5)
    run("make-writer-packet", "--private-dir", str(P), "--code", "P01", "--rows", "5", "--out-dir", str(pk))
    check("빈 표를 두 번 내도 번호가 겹치지 않음", (pk / "writer_sheet_P01_016.csv").exists())
    meta_out = TMP / "corpus_meta.csv"
    run("export-corpus-meta", *C, "--out", str(meta_out))
    check("코퍼스 메타 표에 파이프라인 산출물 없음", "읽으면 안 되는" not in meta_out.read_text(encoding="utf-8-sig"))

    checks = [
        crow("P01-001", "101"), crow("P01-002", "101"), crow("P01-003", "107"),
        crow("P01-004", "", in_corpus="no", ooc_title="가짜트로트", ooc_artist="가짜가수"),
        crow("P01-005", "108"), crow("P01-006", "", status="undeterminable", in_corpus="unknown", confidence=""),
        crow("P01-007", "102|103", multiple_answers="versions"),
        crow("P01-008", "111"), crow("P01-009", "112"), crow("P01-010", "113"),
        crow("P02-001", "104"), crow("P02-002", "105", status="pending", in_corpus="unknown"),
        crow("P02-003", "109"), crow("P02-004", "114"), crow("P02-005", "114"),
        crow("P04-001", "115", checker_code="P09"),
    ]
    check("여러 작성자 항목이 섞인 확인자 표 거부", refused(intake("checker", "P09", checks, ct.CHECKER_COLS)))
    ok_all = True
    for owner in ("P01", "P02", "P04"):
        ok_all &= intake("checker", "P09", [r for r in checks if r["entry_id"].startswith(owner)],
                         ct.CHECKER_COLS)[0] == 0
    check("확인자 표 받기(작성자별 파일)", ok_all)
    weird = [crow("P03-001", "116", status="probably"),                       # 모르는 상태
             crow("P03-002", "999999", in_corpus="yes"),                      # 코퍼스에 없는 ID를 안이라고 함
             crow("P03-003", "117", checked_at=""),                           # 확인 시각 없음
             crow("P03-004", "118|119", multiple_answers="no")]               # 정답 여럿인데 복수 표시 없음
    intake("checker", "P08", [dict(r, checker_code="P08") for r in weird], ct.CHECKER_COLS)

    code, out = run("check", "--private-dir", str(P), *C)
    rep = json.loads((P / "work/check_report.json").read_text(encoding="utf-8"))
    errs = rep["errors"]
    check("검사 실패 종료 코드", code == 3)
    check("작성 시각 누락", "missing_written_at" in errs.get("P03-001", []))
    check("모르는 갈래 값", "bad_track" in errs.get("P03-002", []))
    check("정답 기록 시각이 작성보다 앞서는데 기억 먼저라고 표시", "answer_noted_before_writing" in errs.get("P03-003", []))
    check("두 번째 턴에 앞 질문 없음", "turn_without_prior_question" in errs.get("P03-004", []))
    check("1턴 없는 대화", "bad_conversation" in errs.get("P03-004", []))
    check("모르는 확인 상태를 확인됨으로 치지 않음", "bad_answer_status" in errs.get("checker:P03-001", []))
    check("코퍼스 밖 ID를 코퍼스 안이라고 적음", "in_corpus_id_not_in_corpus" in errs.get("checker:P03-002", []))
    check("확인 시각 누락", "missing_checked_at" in errs.get("checker:P03-003", []))
    check("정답 여럿인데 복수 정답 표시 없음", "one_answer_with_many_ids" in errs.get("checker:P03-004", []))
    check("셀 안 줄바꿈은 경고로 남기고 행을 합치지 않음",
          any(w.startswith("cell_newline:query") for w in rep["warnings"].get("P01-010", []))
          and rep["writer_entries"] == 21)
    check("화면 출력에 질의 원문 없음", not any(q[:8] in out for q in QUERIES.values()))

    check("엑셀식 시각 모양 허용·잘못된 날짜 거부",
          ct.canon_time("2026/10/21 9:05") == "2026-10-21 09:05" and ct.canon_time("2026.10.21. 09:05:00") ==
          "2026-10-21 09:05" and ct.canon_time("2026-02-30 10:00") == "" and ct.canon_time("어제 저녁") == "")
    check("시각 비교는 맞춘 꼴로(9:05 < 10:00)",
          "answer_noted_before_writing" in ct.validate_writer(wrow("P05-001", "시각 비교 시험", written_at="2026-10-21 10:00",
                                                                  writer_answer_title="x",
                                                                  answer_noted_at="2026/10/21 9:05"))[0])

    # ---- 수정본: 의미가 바뀌는 질의 수정은 작성자 확인이 있어야 한다 ----
    fix_noconfirm = [dict(p1[5], query="그 노래 뭐더라 후렴에 박수 소리")]
    check("작성자 확인 없는 질의 수정 거부", refused(intake("writer", "P01", fix_noconfirm, ct.WRITER_COLS,
                                                      "--reason", "작성자가 덧붙임", "--editor", "P10")))
    fix_ok = [dict(p1[5], query="그 노래 뭐더라 후렴에 박수 소리", edit_confirmed_by_writer="yes",
                   edit_note="작성자가 직접 덧붙임(2026-10-22 통화)")]
    check("작성자 확인 있는 수정본 받기", intake("writer", "P01", fix_ok, ct.WRITER_COLS, "--reason", "작성자가 덧붙임",
                                           "--editor", "P10")[0] == 0)
    fix3 = [dict(r, written_at="2026-10-21 10:00", track="general") for r in bad[:2]]
    intake("writer", "P03", fix3, ct.WRITER_COLS, "--reason", "형식 오류 되물음 뒤 작성자 회신", "--editor", "P10")
    fix_c = [dict(crow("P03-001", "116"), checker_code="P08"), dict(crow("P03-002", "116"), checker_code="P08"),
             dict(crow("P03-003", "117"), checker_code="P08"),
             dict(crow("P03-004", "118|119", multiple_answers="distinct", status="ambiguous"), checker_code="P08")]
    intake("checker", "P08", fix_c, ct.CHECKER_COLS, "--reason", "형식 오류 수정", "--editor", "P08")
    fix_w = [dict(bad[2], answer_noted_at="2026-10-21 11:00"),
             dict(bad[3], turn_no="1", prior_question="")]
    intake("writer", "P03", fix_w, ct.WRITER_COLS, "--reason", "시각·턴 번호 오기 정정", "--editor", "P10")

    # ---- build ----
    run("build", "--private-dir", str(P), *C, *R)
    merged = {r["entry_id"]: r for r in ct.read_csv(P / "work/merged.csv")}
    fl = {k: set(filter(None, v["flags"].split("|"))) for k, v in merged.items()}
    check("최초 질의 원문 보존(수정본과 따로)", merged["P01-006"]["query_original"] == "그 노래 뭐더라"
          and "query_edited" in fl["P01-006"])
    check("기존 문장과 비슷 → excluded_near_dup(조용히 지우지 않고 남김)", merged["P01-003"]["pool"] == "excluded_near_dup")
    check("기존 정답 곡 → known_song 표시, 묶음에서 빼지 않음",
          "known_song" in fl["P01-008"] and merged["P01-008"]["pool"] == "in_corpus")
    check("새 곡 → new_song 표시", "new_song" in fl["P01-001"])
    check("코퍼스 밖 → ooc 묶음으로 남음", merged["P01-004"]["pool"] == "ooc")
    check("판단불가 → undeterminable 묶음으로 남음", merged["P01-006"]["pool"] == "undeterminable")
    check("확인 대기 → pending", merged["P02-002"]["pool"] == "pending")
    check("AI가 다듬은 질의 → ai_assisted(주 세트와 분리)", merged["P01-009"]["pool"] == "ai_assisted")
    check("작품명 정규화 군집(띄어쓰기 차이)", merged["P01-005"]["work_key"] == merged["P02-003"]["work_key"] != "")
    check("앨범 군집 기록(101·102 같은 앨범)", merged["P01-001"]["album_keys"] in merged["P01-007"]["album_keys"].split(";"))
    check("짧은 질의 표시", "short" in fl["P01-006"])
    check("자기 보고 노출 표시", "writer_exposed" in fl["P04-001"] and "writer_is_developer" in fl["P04-001"])
    check("셀 줄바꿈 원문 보존", merged["P01-010"]["query"] == "줄이 바뀐 질의\n두 번째 줄도 원문이다")
    check("자기 보고가 있는 작성자는 no_declaration 아님", "no_declaration" not in fl["P01-001"])

    # ---- split ----
    code, out = run("split", "--private-dir", str(P), *C, *R)
    check("분할 실행", code == 0)
    ledger1 = (P / "logs/split_ledger.jsonl").read_text(encoding="utf-8")
    assigned = ct.assignments(ct.Private(P))
    check("AI 사용·확인 대기는 분할하지 않음", "P01-009" not in assigned and "P02-002" not in assigned)
    check("같은 곡 두 질의 같은 분할", assigned["P01-001"] == assigned["P01-002"])
    check("같은 대화의 턴 같은 분할", assigned["P02-004"] == assigned["P02-005"])
    check("같은 작품 질의 같은 분할", assigned["P01-005"] == assigned["P02-003"])
    check("코퍼스 밖·판단불가도 분할에 들어감(따로 평가)", "P01-004" in assigned and "P01-006" in assigned)
    code, out = run("split", "--private-dir", str(P), *C, *R)
    check("다시 돌려도 장부가 그대로(최초 배정 보존)",
          (P / "logs/split_ledger.jsonl").read_text(encoding="utf-8") == ledger1 and '"newly_assigned": 0' in out)

    # 다음 묶음: 이미 배정된 곡을 잇는 새 질의는 그 분할을 물려받는다
    intake("writer", "P02", [wrow("P02-006", "축제 떼창 노래 또 생각나서 적음")], ct.WRITER_COLS)
    intake("checker", "P09", [crow("P02-006", "101")], ct.CHECKER_COLS)
    run("split", "--private-dir", str(P), *C, *R)
    assigned = ct.assignments(ct.Private(P))
    check("새 묶음의 같은 곡 질의는 기존 분할을 물려받음", assigned["P02-006"] == assigned["P01-001"])

    # 두 분할을 잇는 복수 정답 → 격리
    by_split = defaultdict(list)
    merged = {r["entry_id"]: r for r in ct.read_csv(P / "work/merged.csv")}
    for eid, s in assigned.items():
        if merged.get(eid, {}).get("pool") == "in_corpus" and s in ("policy_dev", "sealed_eval"):
            by_split[s].append(merged[eid]["answer_ids"].split("|")[0])
    if by_split["policy_dev"] and by_split["sealed_eval"]:
        a, b = by_split["policy_dev"][0], by_split["sealed_eval"][0]
        intake("writer", "P02", [wrow("P02-007", "둘 중 어느 곡인지 헷갈리는 질의")], ct.WRITER_COLS)
        intake("checker", "P09", [crow("P02-007", f"{a}|{b}", status="ambiguous", multiple_answers="distinct")],
               ct.CHECKER_COLS)
        run("split", "--private-dir", str(P), *C, *R)
        check("두 분할 곡을 잇는 복수 정답 질의 → 격리(quarantine)",
              ct.assignments(ct.Private(P))["P02-007"] == "quarantine")
    else:
        check("두 분할이 모두 생겨야 격리 시험 가능", False)

    # 수정본이 배정된 항목끼리 두 분할을 잇게 하면 분할을 멈춘다
    assigned = ct.assignments(ct.Private(P))
    pd = next(e for e, s in assigned.items() if s == "policy_dev" and merged.get(e, {}).get("pool") == "in_corpus")
    se = next(e for e, s in assigned.items() if s == "sealed_eval" and merged.get(e, {}).get("pool") == "in_corpus")
    se_song = merged[se]["answer_ids"].split("|")[0]
    pd_row = dict(CROWS[pd])
    intake("checker", "P09", [dict(pd_row, answer_ids=f"{pd_row['answer_ids']}|{se_song}", multiple_answers="distinct",
                                   answer_status="ambiguous")],
           ct.CHECKER_COLS, "--reason", "다른 곡도 맞을 수 있음", "--editor", "P09")
    ledger_before = (P / "logs/split_ledger.jsonl").read_text(encoding="utf-8")
    code, out = run("split", "--private-dir", str(P), *C, *R)
    check("수정본이 두 분할을 이으면 분할 거부(배정 그대로)",
          code == 3 and (P / "logs/split_ledger.jsonl").read_text(encoding="utf-8") == ledger_before)
    run("quarantine", "--private-dir", str(P), "--entry", pd, "--reason", "수정본이 두 분할 곡을 이음", "--by", "P10")
    check("보관자 격리 결정 뒤 분할 재개", run("split", "--private-dir", str(P), *C, *R)[0] == 0)

    # 곡이 두 분할에 걸치지 않는지 전수 확인
    merged = {r["entry_id"]: r for r in ct.read_csv(P / "work/merged.csv")}
    assigned = ct.assignments(ct.Private(P))
    song_splits = defaultdict(set)
    for eid, s in assigned.items():
        if s in ("policy_dev", "sealed_eval"):
            for k in merged[eid]["group_keys"].split("|"):
                song_splits[k].add(s)
    check("곡·대화·작품 키가 두 분할에 걸치지 않음", all(len(v) == 1 for v in song_splits.values()))

    # 무작위 대량 합성: 곡·복수 정답이 얽혀도 분할 간 곡 중복 0
    rng = random.Random(20261020)
    P2 = TMP / "keeper2"
    run("init", "--private-dir", str(P2), "--allow-unsafe-location")
    run("commit-salt", "--private-dir", str(P2))
    for batch in range(3):
        ws, cs = [], []
        for k in range(40):
            eid = f"P{11 + batch:02d}-{k + 1:03d}"
            ws.append(wrow(eid, f"합성 질의 {batch} {k} {rng.random():.6f}"))
            ids = rng.sample(range(101, 121), rng.choice([1, 1, 1, 2]))
            cs.append(crow(eid, "|".join(map(str, ids)), multiple_answers="distinct" if len(ids) > 1 else "no",
                           status="ambiguous" if len(ids) > 1 else "confirmed"))
        run("intake", "--private-dir", str(P2), "--kind", "writer", "--code", f"P{11 + batch:02d}", "--file",
            str(sheet(ws, ct.WRITER_COLS)))
        run("intake", "--private-dir", str(P2), "--kind", "checker", "--code", "P09", "--file",
            str(sheet(cs, ct.CHECKER_COLS)))
        run("split", "--private-dir", str(P2), *C, *R)
    m2 = {r["entry_id"]: r for r in ct.read_csv(P2 / "work/merged.csv")}
    a2 = ct.assignments(ct.Private(P2))
    s2 = defaultdict(set)
    for eid, s in a2.items():
        if s in ("policy_dev", "sealed_eval"):
            for k in m2[eid]["group_keys"].split("|"):
                s2[k].add(s)
    check("대량 합성(120건·3묶음·복수 정답 섞임)에서 분할 간 곡 중복 0", all(len(v) == 1 for v in s2.values()))
    check("자기 보고가 없는 작성자 표시", all("no_declaration" in r["flags"] for r in m2.values()))
    check("대량 합성 모든 대상 항목이 배정 또는 격리됨", all(e in a2 for e, r in m2.items() if r["pool"] in ct.SPLIT_POOLS))

    # ---- salt·비율 변경 차단 ----
    salt_path = P / "salt.txt"
    os.chmod(salt_path, stat.S_IRUSR | stat.S_IWUSR)
    original_salt = salt_path.read_text(encoding="utf-8")
    salt_path.write_text("x" * 64 + "\n", encoding="utf-8")
    check("salt를 바꾸면 분할 거부", refused(run("split", "--private-dir", str(P), *C, *R)))
    check("salt를 바꾸면 verify 실패", run("verify", "--private-dir", str(P))[0] == 3)
    salt_path.write_text(original_salt, encoding="utf-8")
    ct.readonly(salt_path)

    # ---- 원본 해시 불일치 ----
    stored = P / ct.read_log(ct.Private(P).intake_log)[4]["stored"]
    check("보존본은 읽기 전용으로 저장됨", not os.access(stored, os.W_OK))
    os.chmod(stored, stat.S_IRUSR | stat.S_IWUSR)
    original = stored.read_bytes()
    stored.write_bytes(original.replace("떼창".encode(), "합창".encode()))
    check("보존본이 바뀌면 verify 실패", run("verify", "--private-dir", str(P))[0] == 3)
    check("보존본이 바뀌면 분할 거부", refused(run("split", "--private-dir", str(P), *C, *R)))
    stored.write_bytes(original)
    ct.readonly(stored)
    check("되돌리면 verify 통과", run("verify", "--private-dir", str(P))[0] == 0)

    # ---- 내보내기·시점 기록·봉인 ----
    check("봉인 내보내기", run("export", "--private-dir", str(P), *C, *R)[0] == 0)
    exports1 = ct.read_log(ct.Private(P).export_log)
    run("export", "--private-dir", str(P), *C, *R)
    check("내용이 같으면 새 판을 만들지 않음", ct.read_log(ct.Private(P).export_log) == exports1)
    sealed_file = P / [e for e in exports1 if e["split"] == "sealed_eval"][-1]["file"]
    check("내보낸 파일은 읽기 전용", not os.access(sealed_file, os.W_OK))
    pd_rows = ct.read_csv(P / [e for e in exports1 if e["split"] == "policy_dev"][-1]["file"])
    check("policy_dev 내보내기에 봉인 항목 없음", all(r["split"] == "policy_dev" for r in pd_rows))
    check("모델 출력 공개 기록은 policy_dev만", refused(run("event", "--private-dir", str(P), "--type",
                                                   "model_outputs_released", "--scope", "sealed_eval", "--by", "P10")))
    run("event", "--private-dir", str(P), "--type", "model_outputs_released", "--scope", "policy_dev", "--by", "P10")
    check("채택 기준 고정 전 봉인 열기 거부", refused(run("event", "--private-dir", str(P), "--type", "sealed_opened",
                                                    "--by", "P10")))
    crit = TMP / "criteria.md"
    crit.write_text("합성 채택 기준", encoding="utf-8")
    run("event", "--private-dir", str(P), "--type", "criteria_fixed", "--file", str(crit), "--configs", "A,B",
        "--by", "P10")
    check("기준 고정 뒤 봉인 열기", run("event", "--private-dir", str(P), "--type", "sealed_opened", "--by", "P10")[0] == 0)
    check("봉인 두 번째 열기 거부", refused(run("event", "--private-dir", str(P), "--type", "sealed_opened", "--by", "P10")))
    check("봉인 연 뒤 새 배정 거부", refused(run("split", "--private-dir", str(P), *C, *R)))
    ev = ct.read_log(ct.Private(P).events)
    check("데이터 고정 시점과 모델 출력 공개 시점이 따로 기록됨",
          any(e["type"] == "data_frozen" for e in ev) and any(e["type"] == "model_outputs_released" for e in ev))

    # ---- 참여 철회 ----
    run("make-checker-sheet", "--private-dir", str(P), "--writer", "P04", "--include-checked",
        "--out", str(P / "packets" / "checker_P04.csv"))
    run("make-writer-packet", "--private-dir", str(P), "--code", "P04", "--rows", "3", "--out-dir", str(P / "packets/P04"))
    check("확인자 사본 전달 기록", run("deliver", "--private-dir", str(P), "--file", "packets/checker_P04.csv",
                                  "--to", "P09", "--by", "P10")[0] == 0)
    exps = ct.read_log(ct.Private(P).export_log)
    pd_export = [e for e in exps if e["split"] == "policy_dev"][-1]
    se_export = [e for e in exps if e["split"] == "sealed_eval"][-1]
    run("deliver", "--private-dir", str(P), "--file", pd_export["file"], "--to", "dev", "--by", "P10")
    check("봉인 내보내기는 밖으로 보내기 거부",
          refused(run("deliver", "--private-dir", str(P), "--file", se_export["file"], "--to", "dev", "--by", "P10")))
    check("기록에 없는 파일 전달 거부",
          refused(run("deliver", "--private-dir", str(P), "--file", "LOCATION.json", "--to", "dev", "--by", "P10")))
    expected_requests = 1 + (1 if "P04" in pd_export["owners"] else 0)
    before = ct.read_log(ct.Private(P).export_log)
    code, out = run("withdraw", "--private-dir", str(P), "--person", "P04", "--by", "P10", "--reason", "본인 요청")
    gone = ct.withdrawn_files(ct.Private(P))
    check("철회: 그 사람 표·확인 표·자기 보고를 지움", code == 0 and len([g for g in gone if "received" in g]) == 3
          and all(not (P / g).exists() for g in gone))
    check("철회 뒤 verify 통과(지운 파일은 기록으로 설명됨)", run("verify", "--private-dir", str(P))[0] == 0)
    rows_after, _ = ct.current_rows(ct.Private(P), "writer")
    check("철회 뒤 현재 판에 그 사람 항목 없음", not any(k.startswith("P04-") for k in rows_after))
    check("철회 뒤 다른 사람 자료는 그대로", "P01-001" in rows_after and "P02-005" in rows_after)
    merged_path = P / "work/merged.csv"
    check("철회 뒤 통합 작업표에 그 사람 행·원문 없음",
          not merged_path.exists() or not any(r["entry_id"].startswith("P04-") for r in ct.read_csv(merged_path)))
    check("철회 뒤 파생 작업 파일 모두 지움(다음 build가 다시 만듦)", not any((P / "work").glob("*")))
    check("철회 뒤 보관자 폴더 안 확인자 사본·작성자 패킷 지움",
          not (P / "packets/checker_P04.csv").exists() and not any((P / "packets/P04").glob("*")))
    leftovers = [f for f in P.rglob("*") if f.is_file() and f.suffix in (".csv", ".json") and b"P04-" in f.read_bytes()]
    check("철회 뒤 보관자 폴더 어디에도 그 사람 항목 없음", leftovers == [])
    check("밖으로 보낸 사본은 삭제 요청으로 남음", len(ct.pending_deletions(ct.Private(P))) == expected_requests)
    run("build", "--private-dir", str(P), *C, *R)
    check("철회 뒤 다시 만든 통합 작업표에도 없음",
          not any(r["entry_id"].startswith("P04-") for r in ct.read_csv(merged_path)))
    st_w = TMP / "STATUS_W.md"
    run("status", "--private-dir", str(P), *C, *R, "--out", str(st_w))
    check("집계에 삭제 확인 대기 건수", f"삭제 확인 대기 {expected_requests}건" in st_w.read_text(encoding="utf-8"))
    check("모르는 삭제 요청 확인 거부",
          refused(run("confirm-deletion", "--private-dir", str(P), "--request", "DR-999", "--by", "P10", "--note", "x")))
    for r in ct.pending_deletions(ct.Private(P)):
        run("confirm-deletion", "--private-dir", str(P), "--request", r["id"], "--by", "P10", "--note", "받은 사람이 지웠다고 회신")
    check("삭제 확인 뒤 대기 0", ct.pending_deletions(ct.Private(P)) == [])

    # ---- 장부 사슬 ----
    led = P / "logs/split_ledger.jsonl"
    text = led.read_text(encoding="utf-8")
    first = json.loads(text.splitlines()[0])
    flipped = "policy_dev" if first["split"] == "sealed_eval" else "sealed_eval"
    led.write_text(text.replace(json.dumps(first, ensure_ascii=False, sort_keys=True),
                                json.dumps({**first, "split": flipped}, ensure_ascii=False, sort_keys=True), 1),
                   encoding="utf-8")
    check("분할 장부를 고쳐 쓰면 verify 실패", run("verify", "--private-dir", str(P))[0] == 3)
    led.write_text(text, encoding="utf-8")

    # ---- 익명 집계 ----
    st = TMP / "STATUS.md"
    run("status", "--private-dir", str(P), *C, *R, "--out", str(st))
    body = st.read_text(encoding="utf-8") + st.with_suffix(".json").read_text(encoding="utf-8")
    check("집계에 질의 원문 없음", not any(q[:8] in body for q in QUERIES.values()))
    check("집계에 곡 ID·제목 없음", not re.search(r"(?<![0-9a-f])1(0[1-9]|1[0-9]|20)(?![0-9a-f])", body)
          and "가짜제목" not in body and "가짜트로트" not in body)
    check("집계에 시험 위치 경고", "시험 위치" in body)
    st0 = TMP / "STATUS0.md"
    run("status", "--out", str(st0), "--keeper-note", "지정됨", "--plan-note", "시범 3명 × 10건")
    body0 = st0.read_text(encoding="utf-8")
    check("보관자 폴더 없으면 0건으로 표시·지정 상태와 계획 반영",
          "**0건**" in body0 and "| 보관자(원본·salt·봉인) | 지정됨 |" in body0 and "시범 3명 × 10건" in body0)
    check("집계에 실제 봉인 비율·군집 수", "실제 봉인 비율" in body and "군집 수" in body)

    # ---- v68 도구는 고치지 않았다 ----
    v68_self = ct.V68_DIR / "selftest_new_data.py"
    res = subprocess.run([sys.executable, str(v68_self)], capture_output=True, text=True)
    check("v68 selftest_new_data 그대로 통과", res.returncode == 0)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"PASSED {len(results) - len(fails)}/{len(results)} · FAILURES {len(fails)}")
sys.exit(1 if fails else 0)
