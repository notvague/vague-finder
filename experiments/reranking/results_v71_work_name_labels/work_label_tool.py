"""v71 작품명 해석 22질의 — 사람 라벨 자료 준비·검증·고정·대조·집계(모델·검색·외부 호출 0).

에이전트는 라벨을 채우지 않는다. 이 도구는 사람이 채운 표의 형식을 검사하고, 고정하고, 실행 전에 정한 규칙(README §5)대로 센다.
작품 이름이 같은지 문자열로 정할 수 없는 쌍은 사람이 `name_pairs.csv`에 판정한다(도구가 추정하지 않는다).

    prepare    v71 원문 대기표 → 가린 1단계 표(라벨러·확인자) + 항목 번호 대응표(비공개) + 해시 기록(추적)
    check      채운 1단계 표의 형식·일관성 검사(원문·작품명을 화면에 찍지 않는다)
    freeze     검사를 통과한 22행 표를 읽기 전용으로 고정한다. 같은 역할의 최초 고정본은 덮어쓰지 않는다
               (--provisional: 오류·미완료가 있어도 받은 그대로 보존 고정하고 행 상태를 기록)
    normalize  공백·빈 열·빈 행·상태 표기 공백만 정리한 작업용 사본 + 변경 내역(의미가 바뀌는 수정은 하지 않음)
    agree      라벨러·확인자 최초 고정본 대조(일치·불일치·판단 불가) — 문자열로 못 가르는 이름 쌍은 사람 판정 대기
    compare    라벨 고정 뒤에만: v71 모델 출력과 라벨을 맞댄 2단계 표(비공개)를 만든다
    summarize  2단계 표 + 이름 쌍 판정으로 정답·오답·보류를 센다(추적 요약에는 질의 원문·작품명 없음)

경로 기본값은 이 폴더 기준이다. 비공개 파일은 `private/`(Git 제외)에만 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import shutil
import stat
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"
SEED = 20261018
KST = dt.timezone(dt.timedelta(hours=9))

# 판정 통과 22질의(v70 gate_v2) — v71 대기표·v72 대조표와 같은 집합이어야 한다
EXPECTED_QIDS = (
    "q105", "m103", "c602", "c603", "c607", "c701", "c705", "c707",
    "n010", "n031", "n033", "n038", "n043", "n045", "n096",
    "t024", "t036", "t037", "t041", "t043", "t046", "t098",
)

STATUS = {"단일": "single", "복수": "multiple", "작품아님": "not_work", "판단불가": "undeterminable"}
CONFIDENCE = {"높음", "중간", "낮음"}
YES_NO = {"yes", "no"}
EXPOSURE = {"yes", "no", "unsure"}
EXPOSURE_COLS = (
    "saw_model_output_before", "saw_answer_song_before",
    "saw_experiment_results_before", "saw_v67_work_list_before",
)
# 이 셋 중 하나라도 yes·unsure면 그 행은 독립 라벨이 아니다. 정답 곡 노출은 v71 안내대로 행을 빼지 않는다(표시만)
NON_INDEPENDENT_IF = ("saw_model_output_before", "saw_experiment_results_before", "saw_v67_work_list_before")

LABEL_COLS = (
    "judgment_status", "intended_work", "possible_works", "work_type", "work_identifier",
    "query_phrase", "external_check", "sources", "memory_mismatch", "confidence", "open_questions",
    "labeler_id", "labeled_at", *EXPOSURE_COLS, "notes",
)
SHEET_COLS = ("item_no", "query", *LABEL_COLS)

# 2단계 표의 사람 칸 / 이름 쌍 판정 칸
PAIR_COLS = ("pair_id", "item_no", "side_a", "name_a", "side_b", "name_b", "name_b_aliases", "same_work", "decided_by",
             "decided_at", "agent_suggestion", "agent_basis", "notes")
SAME_WORK = {"yes", "no", "unsure"}


# ---------- 공통 ----------

def norm_name(text: str) -> str:
    """NFKC·소문자·공백과 문장부호 제거. 같은 작품의 다른 표기(띄어쓰기·물음표)만 맞춘다 — 별칭·약칭은 사람 판정."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def split_multi(text: str) -> List[str]:
    parts = re.split(r"[;；\n]", str(text or ""))
    return [p.strip() for p in parts if p.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_csv(path: Path) -> List[Dict[str, str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # 엑셀이 CP949로 다시 저장한 경우
        text = raw.decode("cp949")
    # 셀 안 줄바꿈(따옴표 안 \n)을 보존한다 — splitlines()는 한 칸을 두 행으로 쪼개 작품명을 합친다
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def write_csv(path: Path, cols: Sequence[str], rows: Iterable[Dict[str, str]], bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def now_kst() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


# ---------- prepare ----------

def build_packet(source_rows: List[Dict[str, str]], seed: int = SEED) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """원문 대기표 → (가린 표 행, 대응표 행). 질의 순서를 섞어 세트·사례 묶음이 번호로 드러나지 않게 한다."""
    by_qid = {r["qid"]: r for r in source_rows}
    if sorted(by_qid) != sorted(EXPECTED_QIDS) or len(source_rows) != len(EXPECTED_QIDS):
        raise SystemExit(f"원문 대기표의 질의 집합이 판정 통과 22건과 다르다: {len(source_rows)}행")
    order = list(EXPECTED_QIDS)
    random.Random(seed).shuffle(order)
    sheet, mapping = [], []
    for i, qid in enumerate(order, 1):
        item = f"L{i:02d}"
        query = by_qid[qid]["query"]
        sheet.append({"item_no": item, "query": query, **{c: "" for c in LABEL_COLS}})
        mapping.append({"item_no": item, "set": by_qid[qid]["set"], "qid": qid, "query_sha16": sha16(compact(query))})
    return sheet, mapping


def cmd_prepare(args: argparse.Namespace) -> int:
    src = Path(args.source)
    rows = read_csv(src)
    tracked_q = [r["qid"] for r in read_csv(Path(args.tracked_queue))]
    if sorted(tracked_q) != sorted(EXPECTED_QIDS):
        raise SystemExit("v71 추적 대기표의 질의 집합이 다르다")
    sheet, mapping = build_packet(rows, args.seed)
    out = Path(args.private_dir)
    if (out / "item_map.csv").exists() and not args.force:
        raise SystemExit("이미 준비됨 — 대응표를 덮어쓰지 않는다(--force는 라벨이 하나도 돌아오기 전에만)")
    src_copy = out / "source" / "work_label_queue_private.csv"
    src_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, src_copy)
    write_csv(out / "item_map.csv", ("item_no", "set", "qid", "query_sha16"), mapping)
    write_csv(out / "packet_blank" / "work_label_sheet_blank.csv", SHEET_COLS, sheet, bom=True)
    (out / "packet_blank" / "blank.sha256").write_text(sha256_file(out / "packet_blank" / "work_label_sheet_blank.csv"),
                                                       encoding="utf-8")
    for role in ("labeler", "checker"):
        write_csv(out / "packet" / role / f"work_label_sheet_{role}.csv", SHEET_COLS, sheet, bom=True)
        shutil.copyfile(HERE / "LABELER_GUIDE.md", out / "packet" / role / "LABELER_GUIDE.md")
    manifest = {
        "prepared_at": now_kst(),
        "seed": args.seed,
        "n_items": len(sheet),
        "source_sha256": sha256_file(src),
        "item_map_sha256": sha256_file(out / "item_map.csv"),
        "blank_sheet_sha256": {
            role: sha256_file(out / "packet" / role / f"work_label_sheet_{role}.csv") for role in ("labeler", "checker")
        },
        "guide_sha256": sha256_file(HERE / "LABELER_GUIDE.md"),
        "sheet_columns": list(SHEET_COLS),
        "model_output_in_this_folder": False,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(sheet), "manifest": str(args.manifest)}, ensure_ascii=False))
    return 0


# ---------- check ----------

def check_sheet(rows: List[Dict[str, str]], packet: List[Dict[str, str]]) -> Dict[str, object]:
    """사람이 채운 표를 검사한다. errors는 고정을 막고 warnings는 표시만 한다. 원문·작품명은 결과에 넣지 않는다."""
    errors: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []
    if rows:
        missing = [c for c in SHEET_COLS if c not in rows[0]]
        if missing:
            return {"rows": len(rows), "filled": 0, "complete": False,
                    "errors": [["*", f"missing_columns:{','.join(missing)}"]], "warnings": [], "status_counts": {}}
    packet_q = {r["item_no"]: compact(r["query"]) for r in packet}
    seen: Counter = Counter(r.get("item_no", "") for r in rows)
    for item, n in seen.items():
        if n > 1:
            errors.append((item, "duplicate_item"))
    for item in sorted(set(seen) - set(packet_q)):
        errors.append((item or "(빈칸)", "unknown_item"))
    for item in sorted(set(packet_q) - set(seen)):
        errors.append((item, "missing_item"))

    filled = 0
    status_counts: Counter = Counter()
    for r in rows:
        item = r.get("item_no", "")
        if item not in packet_q:
            continue
        if compact(r.get("query", "")) != packet_q[item]:
            errors.append((item, "query_changed"))
        st_raw = r.get("judgment_status", "")
        if not st_raw:
            if any(r.get(c, "") for c in LABEL_COLS):
                errors.append((item, "status_missing_but_row_touched"))
            continue
        if st_raw not in STATUS:
            errors.append((item, "bad_judgment_status"))
            continue
        filled += 1
        st = STATUS[st_raw]
        status_counts[st] += 1
        intended = r.get("intended_work", "")
        possible = split_multi(r.get("possible_works", ""))
        if st == "single":
            if not intended:
                errors.append((item, "single_without_intended_work"))
            if len(split_multi(intended)) > 1:
                errors.append((item, "single_with_several_intended"))
            if possible:
                warnings.append((item, "single_with_possible_works"))
        elif st == "multiple":
            names = {norm_name(x) for x in possible + split_multi(intended)}
            names.discard("")
            if len(names) < 2:
                errors.append((item, "multiple_needs_two_or_more_works"))
            if intended and norm_name(intended) not in {norm_name(x) for x in possible}:
                errors.append((item, "multiple_intended_not_in_possible"))
        else:
            # 작품 지칭 아님·판단 불가 — 억지로 하나를 고르지 않는다
            if intended:
                errors.append((item, f"{st}_with_intended_work"))
            if st == "not_work" and possible:
                errors.append((item, "not_work_with_possible_works"))
        if st in ("single", "multiple"):
            for col in ("work_type", "query_phrase", "confidence"):
                if not r.get(col, ""):
                    errors.append((item, f"{col}_missing"))
            q = compact(r.get("query", ""))
            for phrase in split_multi(r.get("query_phrase", "")):
                if compact(phrase.strip("\"'“”‘’")) not in q:
                    warnings.append((item, "query_phrase_not_verbatim"))
                    break
            names = split_multi(intended) + possible
            if any(norm_name(n) and norm_name(n) in norm_name(r.get("query", "")) for n in names):
                warnings.append((item, "work_name_written_in_query"))
        conf = r.get("confidence", "")
        if conf and conf not in CONFIDENCE:
            errors.append((item, "bad_confidence"))
        if st == "undeterminable" and not (r.get("open_questions", "") or r.get("notes", "")):
            warnings.append((item, "undeterminable_without_reason"))
        ext = r.get("external_check", "")
        if ext not in YES_NO:
            errors.append((item, "bad_external_check"))
        elif ext == "yes" and not r.get("sources", ""):
            errors.append((item, "external_check_without_sources"))
        if not r.get("labeler_id", ""):
            errors.append((item, "labeler_id_missing"))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{1,2}:\d{2})?", r.get("labeled_at", "")):
            errors.append((item, "bad_labeled_at"))
        for col in EXPOSURE_COLS:
            if r.get(col, "") not in EXPOSURE:
                errors.append((item, f"bad_{col}"))
    labelers = sorted({r.get("labeler_id", "") for r in rows if r.get("labeler_id", "")})
    if len(labelers) > 1:
        warnings.append(("*", "several_labeler_ids"))
    return {
        "rows": len(rows),
        "filled": filled,
        "complete": filled == len(packet_q) and not any(c in ("missing_item",) for _, c in errors),
        "status_counts": dict(status_counts),
        "errors": [list(e) for e in errors],
        "warnings": [list(w) for w in warnings],
    }


def _packet_rows(private_dir: Path) -> List[Dict[str, str]]:
    """대조 기준은 빈 표 원본(`private/packet_blank/`)이다. 라벨러가 전달용 표에 바로 채울 수 있으므로
    전달용 표는 기준으로 쓰지 않는다. 사본은 prepare 때 적은 해시와 같아야 한다."""
    blank = private_dir / "packet_blank" / "work_label_sheet_blank.csv"
    sha_file = private_dir / "packet_blank" / "blank.sha256"
    if not blank.exists() or not sha_file.exists():
        raise SystemExit("빈 표 원본(private/packet_blank/)이 없다")
    if sha256_file(blank) != sha_file.read_text(encoding="utf-8").strip():
        raise SystemExit("빈 표 원본 해시 불일치")
    return read_csv(blank)


def cmd_check(args: argparse.Namespace) -> int:
    report = check_sheet(read_csv(Path(args.sheet)), _packet_rows(Path(args.private_dir)))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report["errors"] else 0


# ---------- freeze ----------

# 작성자·작성 시각 칸은 출처 기록이다. 비어 있어도 라벨 내용 판정에는 쓰지 않고 '미기록'으로 따로 적는다
PROVENANCE_CODES = {"labeler_id_missing", "bad_labeled_at"}
INCOMPLETE_CODES = {"status_missing_but_row_touched", "bad_judgment_status"}


def _freeze_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_independent(row: Dict[str, str]) -> bool:
    return all(row.get(c, "") == "no" for c in NON_INDEPENDENT_IF)


def row_states(rows: List[Dict[str, str]], packet: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    """항목별 상태: valid · needs_confirmation(상태는 있으나 내용 모순) · incomplete(상태 없음·허용 밖) · blank(손대지 않음)."""
    report = check_sheet(rows, packet)
    codes: Dict[str, List[str]] = {}
    for item, code in report["errors"]:
        codes.setdefault(item, []).append(code)
    out: Dict[str, Dict[str, object]] = {}
    for r in rows:
        item = r.get("item_no", "")
        if not item:
            continue
        c = codes.get(item, [])
        content = [x for x in c if x not in PROVENANCE_CODES]
        touched = any(r.get(col, "") for col in LABEL_COLS)
        if not touched:
            state = "blank"
        elif set(content) & INCOMPLETE_CODES:
            state = "incomplete"
        elif content:
            state = "needs_confirmation"
        else:
            state = "valid"
        out[item] = {"state": state, "codes": c, "provenance_missing": sorted(set(c) & PROVENANCE_CODES)}
    return out


def cmd_freeze(args: argparse.Namespace) -> int:
    private_dir = Path(args.private_dir)
    sheet = Path(args.sheet)
    rows = read_csv(sheet)
    packet = _packet_rows(private_dir)
    report = check_sheet(rows, packet)
    if (report["errors"] or not report["complete"]) and not args.provisional:
        print(json.dumps({"frozen": False, "errors": report["errors"], "filled": report["filled"]}, ensure_ascii=False))
        return 1
    log_path = Path(args.freeze_log)
    log = _freeze_log(log_path)
    prior = [e for e in log if e["role"] == args.role]
    revision = len(prior) + 1
    if revision > 1 and not args.reason:
        raise SystemExit("같은 역할의 고정본이 이미 있다 — 수정본은 --reason을 적어 새 판으로만 고정한다(최초본은 그대로)")
    stage2_open = (private_dir / "stage2" / "model_vs_label.csv").exists()
    digest = sha256_file(sheet)
    dest = private_dir / "frozen" / f"{args.role}_r{revision}_{digest[:8]}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f"이미 있는 고정본: {dest.name}")
    shutil.copyfile(sheet, dest)
    os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    states = row_states(rows, packet)
    # 생성형 AI가 관여한 판에서 나온 정리본도 AI 보조로 남긴다(앞 판의 표시를 지우지 않음)
    ai_assisted = bool(args.ai_assisted) or any(e.get("ai_assisted") for e in prior)
    entry = {
        "role": args.role,
        "revision": revision,
        "kind": args.kind,
        "ai_assisted": ai_assisted,
        "provisional": bool(report["errors"]) or not report["complete"],
        "frozen_at": now_kst(),
        "file": dest.name,
        "sha256": digest,
        "rows": len(rows),
        # 고정 기록은 Git에 오른다 — 작성자 칸은 원문 대신 해시 앞 8자만(원문은 고정본·private/roles.md)
        "labeler_ids": sorted({hashlib.sha256(r["labeler_id"].encode("utf-8")).hexdigest()[:8]
                               for r in rows if r.get("labeler_id", "")}),
        "labeled_at_recorded_rows": sum(bool(r.get("labeled_at", "")) for r in rows),
        "status_counts": report["status_counts"],
        "row_states": dict(sorted(Counter(v["state"] for v in states.values()).items())),
        "rows_not_valid": {k: v["state"] for k, v in sorted(states.items()) if v["state"] != "valid"},
        "error_codes": dict(sorted(Counter(c for _, c in report["errors"]).items())),
        "independent_rows_self_reported": sum(is_independent(r) for r in rows),
        "exposure_yes_or_unsure": {
            c: sum(r.get(c) in ("yes", "unsure") for r in rows) for c in EXPOSURE_COLS
        },
        "external_check_rows": sum(r.get("external_check") == "yes" for r in rows),
        "warnings": len(report["warnings"]),
        "after_stage2_opened": stage2_open,
        "reason": args.reason or "",
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False))
    return 0


def frozen_sheet(private_dir: Path, log_path: Path, role: str, revision: int = 1) -> Optional[List[Dict[str, str]]]:
    for e in _freeze_log(log_path):
        if e["role"] == role and e["revision"] == revision:
            path = private_dir / "frozen" / e["file"]
            if sha256_file(path) != e["sha256"]:
                raise SystemExit(f"고정본 해시 불일치: {e['file']}")
            return [r for r in read_csv(path) if r.get("item_no", "")]
    return None


# ---------- normalize(기계적 정리 — 작업용 사본만) ----------

def normalize_rows(raw_header: List[str], raw_rows: List[List[str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """의미가 바뀌지 않는 정리만 한다: 빈 머리글·빈 값만 있는 열 제거, 완전히 빈 행 제거, 칸 앞뒤 공백 제거,
    판단 상태의 공백만 다른 표기(예: '작품 아님')를 허용 값으로, 날짜 구분자 'YYYY.MM.DD'를 'YYYY-MM-DD'로. 작품명 오타·빈 상태 칸은 고치지 않는다."""
    changes: List[Dict[str, str]] = []
    keep = []
    for j, h in enumerate(raw_header):
        name = h.strip().lstrip("﻿")
        if not name and all((r[j] if j < len(r) else "").strip() == "" for r in raw_rows):
            changes.append({"item_no": "*", "column": f"(빈 열 {j + 1})", "before": "", "after": "", "change": "drop_empty_column"})
            continue
        keep.append((j, name))
    out = []
    for i, r in enumerate(raw_rows):
        if not any(x.strip() for x in r):
            changes.append({"item_no": f"(행 {i + 2})", "column": "*", "before": "", "after": "", "change": "drop_blank_row"})
            continue
        row = {}
        for j, name in keep:
            v = r[j] if j < len(r) else ""
            row[name] = v
        item = row.get("item_no", "").strip()
        for name in list(row):
            v = row[name]
            new = v.strip()
            if name == "judgment_status" and new not in STATUS and re.sub(r"\s+", "", new) in STATUS:
                new = re.sub(r"\s+", "", new)
                changes.append({"item_no": item, "column": name, "before": v, "after": new, "change": "status_spacing"})
            elif name == "labeled_at" and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", new):
                new = new.replace(".", "-")
                changes.append({"item_no": item, "column": name, "before": v, "after": new, "change": "date_separator"})
            elif new != v:
                changes.append({"item_no": item, "column": name, "before": v, "after": new, "change": "strip_whitespace"})
            row[name] = new
        out.append(row)
    return out, changes


def cmd_normalize(args: argparse.Namespace) -> int:
    raw = Path(args.sheet).read_bytes().decode("utf-8-sig")
    table = list(csv.reader(io.StringIO(raw, newline="")))
    rows, changes = normalize_rows(table[0], table[1:])
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"이미 있는 작업용 사본: {out.name}")
    write_csv(out, SHEET_COLS, rows, bom=True)
    write_csv(out.with_name(out.stem + "_changes.csv"), ("item_no", "column", "before", "after", "change"), changes, bom=True)
    counts = dict(sorted(Counter(c["change"] for c in changes).items()))
    print(json.dumps({"rows": len(rows), "changes": counts, "source_sha256": sha256_file(Path(args.sheet)),
                      "out_sha256": sha256_file(out)}, ensure_ascii=False))
    return 0


# ---------- 이름 쌍(사람 판정) ----------

def label_names(row: Dict[str, str]) -> List[str]:
    st = STATUS.get(row.get("judgment_status", ""), "")
    if st == "single":
        return split_multi(row.get("intended_work", ""))[:1]
    if st == "multiple":
        names = split_multi(row.get("possible_works", "")) + split_multi(row.get("intended_work", ""))
        out, seen = [], set()
        for n in names:
            if norm_name(n) and norm_name(n) not in seen:
                seen.add(norm_name(n))
                out.append(n)
        return out
    return []


class Pairs:
    """문자열이 다른 두 이름이 같은 작품인지에 대한 판정. 사람 판정(`same_work`)이 기준이다.
    `agent_suggestion`은 에이전트 보조 의견이라 use_agent=True인 보조 집계에서만 쓴다. 둘 다 없으면 '대기'."""

    def __init__(self, rows: List[Dict[str, str]], use_agent: bool = False):
        self.decided: Dict[Tuple[str, str, str], str] = {}
        self.agent: Dict[Tuple[str, str, str], str] = {}
        self.use_agent = use_agent
        for r in rows:
            key = (r["item_no"], *sorted((norm_name(r["name_a"]), norm_name(r["name_b"]))))
            if r.get("same_work", "") not in SAME_WORK | {""}:
                raise SystemExit(f"same_work 값 오류: {r['pair_id']}")
            if r.get("same_work") and not r.get("decided_by"):
                raise SystemExit(f"판정자 없음: {r['pair_id']}")
            if r.get("agent_suggestion", "") not in SAME_WORK | {""}:
                raise SystemExit(f"agent_suggestion 값 오류: {r['pair_id']}")
            self.decided[key] = r.get("same_work", "")
            self.agent[key] = r.get("agent_suggestion", "")
        self.pending: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    def same(self, item: str, a: str, b: str) -> Optional[bool]:
        """같은 작품이면 True, 다르면 False, 판정이 없거나 unsure면 None.
        사람 판정이 먼저다(표기가 같아도 사람이 동명 다른 작품이라 하면 False). 그다음 표기 동일, 그다음 에이전트 의견."""
        na, nb = norm_name(a), norm_name(b)
        key = (item, *sorted((na, nb)))
        v = self.decided.get(key, "")
        if v in ("yes", "no"):
            return v == "yes"
        if na == nb:
            return True
        v = self.agent.get(key, "") if self.use_agent else ""
        return True if v == "yes" else False if v == "no" else None

    def register(self, item: str, a_side: str, a: str, b_side: str, b: str, b_display: str = "",
                 note: str = "") -> None:
        key = (item, *sorted((norm_name(a), norm_name(b))))
        if key not in self.pending:
            self.pending[key] = {"item_no": item, "side_a": a_side, "name_a": a, "side_b": b_side, "name_b": b,
                                 "name_b_aliases": b_display, "notes": note}


def match_sets(item: str, pairs: Pairs, a_side: str, a_names: Sequence[str], b_side: str,
               b_names: Sequence[str]) -> Tuple[Set[int], Set[int], bool]:
    """a 쪽 이름 각각이 b 쪽 어느 이름과 같은 작품인지. (맞은 a 색인, 맞은 b 색인, 판정 대기 있음)."""
    hit_a, hit_b, undecided = set(), set(), []
    for i, a in enumerate(a_names):
        for j, b in enumerate(b_names):
            s = pairs.same(item, a, b)
            if s is None:
                undecided.append((i, j))
            elif s:
                hit_a.add(i)
                hit_b.add(j)
    # 양쪽 모두 아직 짝이 없는 이름끼리만 사람 판정을 묻는다
    open_pairs = [(i, j) for i, j in undecided if i not in hit_a and j not in hit_b]
    for i, j in open_pairs:
        pairs.register(item, a_side, a_names[i], b_side, b_names[j])
    return hit_a, hit_b, bool(open_pairs)


def write_pending_pairs(path: Path, pairs: Pairs, existing: List[Dict[str, str]]) -> int:
    rows = list(existing)
    known = {(r["item_no"], *sorted((norm_name(r["name_a"]), norm_name(r["name_b"])))) for r in existing}
    added = 0
    for key, p in sorted(pairs.pending.items()):
        if key in known:
            continue
        added += 1
        rows.append({**p, "pair_id": f"P{len(rows) + 1:03d}", "same_work": "", "decided_by": "", "decided_at": "",
                     "agent_suggestion": "", "agent_basis": "", "notes": p.get("notes", "")})
    write_csv(path, PAIR_COLS, rows, bom=True)
    return added


def _load_pairs(path: Path, use_agent: bool = False) -> Tuple[Pairs, List[Dict[str, str]]]:
    rows = read_csv(path) if path.exists() else []
    return Pairs(rows, use_agent), rows


# ---------- agree ----------

def agreement(lab: Dict[str, str], chk: Dict[str, str], pairs: Pairs) -> str:
    """라벨러·확인자 한 항목의 일치 분류. 판정 대기 이름이 있으면 pending_name_pair."""
    if lab.get("judgment_status") not in STATUS or chk.get("judgment_status") not in STATUS:
        return "incomplete"
    a, b = STATUS[lab["judgment_status"]], STATUS[chk["judgment_status"]]
    if a == "undeterminable" and b == "undeterminable":
        return "both_undeterminable"
    if "undeterminable" in (a, b):
        return "one_undeterminable"
    if a == "not_work" and b == "not_work":
        return "agree"
    if "not_work" in (a, b):
        return "disagree_status"
    la, lb = label_names(lab), label_names(chk)
    hit_a, hit_b, pending = match_sets(lab["item_no"], pairs, "labeler", la, "checker", lb)
    if pending:
        return "pending_name_pair"
    if a == b == "single":
        return "agree" if hit_a else "disagree_work"
    if len(hit_a) == len(la) and len(hit_b) == len(lb):
        return "agree" if a == b else "agree_works_status_differs"
    return "partial_overlap" if hit_a else "disagree_work"


def cmd_agree(args: argparse.Namespace) -> int:
    private_dir, log_path = Path(args.private_dir), Path(args.freeze_log)
    lab = frozen_sheet(private_dir, log_path, "labeler", args.revision)
    chk = frozen_sheet(private_dir, log_path, "checker")
    if lab is None or chk is None:
        raise SystemExit("라벨러 고정본과 확인자 최초 고정본(r1)이 둘 다 있어야 한다")
    pairs_path = private_dir / "name_pairs.csv"
    pairs, pair_rows = _load_pairs(pairs_path)
    by_chk = {r["item_no"]: r for r in chk}
    out, counts = [], Counter()
    for r in lab:
        c = by_chk[r["item_no"]]
        cls = agreement(r, c, pairs)
        counts[cls] += 1
        out.append({"item_no": r["item_no"], "labeler_status": r["judgment_status"], "checker_status": c["judgment_status"],
                    "labeler_works": "; ".join(label_names(r)), "checker_works": "; ".join(label_names(c)),
                    "agreement": cls})
    write_csv(private_dir / "agreement" / "labeler_vs_checker.csv",
              ("item_no", "labeler_status", "checker_status", "labeler_works", "checker_works", "agreement"), out, bom=True)
    added = write_pending_pairs(pairs_path, pairs, pair_rows)
    summary = {"made_at": now_kst(), "labeler_revision": args.revision, "items": len(out),
               "agreement_counts": dict(sorted(counts.items())), "pending_name_pairs_added": added}
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


# ---------- compare(2단계) ----------

def load_model_outputs(path: Path) -> Dict[str, Dict[str, object]]:
    """v71 `analysis_private/model_output_review.csv`(반복 1 = 반복 2 응답 원문 동일) → 질의별 생성·채택·거부 작품."""
    out: Dict[str, Dict[str, object]] = {}
    for r in read_csv(path):
        q = out.setdefault(r["qid"], {"decision": r["decision"], "accepted": [], "rejected": [], "reasons": [],
                                      "name_in_query": False})
        if r["decision"] != "candidates" or not (r["name"] or r["aliases"]):
            continue
        names = [r["name"]] + [a for a in re.split(r"[|;]", r["aliases"]) if a.strip()]
        names = [n.strip() for n in names if n.strip()]
        if r["accepted"] == "True":
            q["accepted"].append(names)
        else:
            q["rejected"].append(names)
            q["reasons"].append(r["reject_reason"])
        q["name_in_query"] = q["name_in_query"] or r["name_in_query"] == "True"
    if sorted(out) != sorted(EXPECTED_QIDS):
        raise SystemExit("모델 출력 표의 질의 집합이 22건과 다르다")
    return out


def load_review_flags(path: Path) -> Dict[str, Dict[str, str]]:
    return {r["qid"]: r for r in read_csv(path)}


def _register_label_model_pairs(item: str, pairs: Pairs, side: str, names: Sequence[str],
                                model: Dict[str, object]) -> None:
    for group in list(model["accepted"]) + list(model["rejected"]):  # type: ignore[arg-type]
        _model_group_matches(item, pairs, side, names, group)


def cmd_compare(args: argparse.Namespace) -> int:
    private_dir, log_path = Path(args.private_dir), Path(args.freeze_log)
    lab = frozen_sheet(private_dir, log_path, "labeler", args.revision)
    if lab is None:
        raise SystemExit("라벨러 고정본이 없다 — 2단계는 라벨 고정 뒤에만 연다")
    chk = frozen_sheet(private_dir, log_path, "checker")
    if chk is None and not args.no_checker:
        raise SystemExit("확인자 고정본이 없다 — 확인자 없이 열려면 --no-checker(README §4의 기록 절차)")
    src = Path(args.model_outputs)
    dest = private_dir / "stage2" / "model_output_review_v71.csv"
    opened_path = private_dir / "stage2" / "opened.json"
    first = json.loads(opened_path.read_text(encoding="utf-8")) if opened_path.exists() else None
    if first:
        # 최초 공개 기록과 모델 출력 사본은 덮어쓰지 않는다. 다시 열 때는 같은 출력인지만 확인한다
        if sha256_file(src) != first["model_outputs_sha256"]:
            raise SystemExit("모델 출력이 최초 공개 때와 다르다 — 2단계를 다시 열지 않는다")
        _verify_model_copy(private_dir)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    model = load_model_outputs(dest)
    flags = load_review_flags(Path(args.review_flags))
    imap = {r["item_no"]: r for r in read_csv(private_dir / "item_map.csv")}
    states = row_states(lab, _packet_rows(private_dir))
    chk_by = {r["item_no"]: r for r in chk} if chk else {}
    rows = []
    for r in lab:
        qid = imap[r["item_no"]]["qid"]
        m = model[qid]
        f = flags[qid]
        c = chk_by.get(r["item_no"], {})
        rows.append({
            "item_no": r["item_no"], "set": imap[r["item_no"]]["set"], "qid": qid, "query": r["query"],
            "label_row_state": states[r["item_no"]]["state"],
            "labeler_status": r["judgment_status"], "labeler_works": "; ".join(label_names(r)),
            "labeler_identifier": r.get("work_identifier", ""),
            "checker_status": c.get("judgment_status", ""), "checker_works": "; ".join(label_names(c)) if c else "",
            "model_decision": m["decision"], "v71_status": f["v71_status"],
            "model_accepted_works": " ; ".join(" | ".join(n) for n in m["accepted"]),
            "model_rejected_works": " ; ".join(" | ".join(n) for n in m["rejected"]),
            "model_reject_reasons": ";".join(m["reasons"]), "model_name_in_query": str(m["name_in_query"]).lower(),
            "review_flags": f["review_flags"], "v67_known_case": f["v67_known_case"],
        })
    cols = ("item_no", "set", "qid", "query", "label_row_state", "labeler_status", "labeler_works", "labeler_identifier",
            "checker_status", "checker_works", "model_decision", "v71_status", "model_accepted_works",
            "model_rejected_works", "model_reject_reasons", "model_name_in_query", "review_flags", "v67_known_case")
    write_csv(private_dir / "stage2" / "model_vs_label.csv", cols, rows, bom=True)
    # 문자열로 못 가르는 이름 쌍(라벨 이름 × 모델 채택·거부 작품)을 판정표에 올린다
    pairs, pair_rows = _load_pairs(private_dir / "name_pairs.csv")
    for row in rows:
        for side, sheet in (("labeler", r_by(lab, row["item_no"])), ("checker", chk_by.get(row["item_no"]))):
            if sheet:
                _register_label_model_pairs(row["item_no"], pairs, side, label_names(sheet), model[row["qid"]])
    added = write_pending_pairs(private_dir / "name_pairs.csv", pairs, pair_rows)
    info = {"run_at": now_kst(), "labeler_revision": args.revision, "model_outputs_sha256": sha256_file(dest),
            "rows": len(rows), "checker": chk is not None, "pending_name_pairs_added": added}
    if first is None:
        opened_path.write_text(json.dumps({"opened_at": info["run_at"], **info}, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
    with open(private_dir / "stage2" / "compare_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(json.dumps(info, ensure_ascii=False))
    return 0


def _verify_model_copy(private_dir: Path) -> Dict[str, object]:
    """2단계 모델 출력 사본이 최초 공개 기록의 해시와 같은지. 다르면 집계·재공개를 멈춘다."""
    opened = json.loads((private_dir / "stage2" / "opened.json").read_text(encoding="utf-8"))
    dest = private_dir / "stage2" / "model_output_review_v71.csv"
    if not dest.exists() or sha256_file(dest) != opened["model_outputs_sha256"]:
        raise SystemExit("2단계 모델 출력 사본의 해시가 최초 공개 기록과 다르다")
    return opened


def r_by(rows: List[Dict[str, str]], item: str) -> Dict[str, str]:
    return next(r for r in rows if r["item_no"] == item)


def _model_group_matches(item: str, pairs: Pairs, side: str, names: Sequence[str],
                         group: Sequence[str]) -> Tuple[Set[int], bool]:
    """모델 작품 하나(대표 이름+별칭 묶음)가 라벨의 어느 이름과 같은 작품인지.
    (라벨 이름, 모델 대표 이름) 쌍만 본다 — 사람 판정 → 표기 동일 → (보조 집계면) 에이전트 의견 순.
    별칭은 모델이 만든 것이라 표기가 같아도 같은 작품으로 확정하지 않고, 사람 판정표에 표시만 해 올린다."""
    hits: Set[int] = set()
    undecided: List[int] = []
    for i, n in enumerate(names):
        v = pairs.same(item, n, group[0])
        if v is True:
            hits.add(i)
        elif v is None:
            undecided.append(i)
    # 이미 같은 작품으로 맞은 모델 작품은 나머지 라벨 이름과의 쌍을 사람에게 묻지 않는다
    if not hits:
        for i in undecided:
            alias_hit = any(norm_name(names[i]) == norm_name(g) for g in group[1:])
            pairs.register(item, side, names[i], "model", group[0], " | ".join(group[1:]),
                           "모델 별칭과 표기가 같음 — 별칭은 모델 생성이라 사람 판정 필요" if alias_hit else "")
    return hits, bool(undecided)


# ---------- summarize ----------

def classify(label: Dict[str, str], model: Dict[str, object], pairs: Pairs, side: str = "labeler") -> Dict[str, object]:
    """README §5-2 규칙(라벨 전 고정). 채택 작품만 검색에 쓰이므로 채택 집합으로 판정하고, 거부 작품은 따로 센다."""
    st = STATUS[label["judgment_status"]]
    names = label_names(label)
    accepted: List[List[str]] = model["accepted"]  # type: ignore[assignment]
    rejected: List[List[str]] = model["rejected"]  # type: ignore[assignment]
    linked = bool(accepted)
    res: Dict[str, object] = {"label_status": st, "model_linked": linked, "pending": False,
                              "generated_works": len(accepted) + len(rejected), "accepted_works": len(accepted),
                              "generated_has_label_work": "", "accepted_has_label_work": "",
                              "accepted_wrong_works": "", "rejected_contains_label_work": ""}
    if st == "undeterminable":
        res.update(outcome="label_undeterminable", wrong_work_link="undeterminable")
        return res
    if st == "not_work":
        res.update(outcome="wrong_link" if linked else "correct_abstain",
                   wrong_work_link="yes" if linked else "no", accepted_wrong_works=len(accepted),
                   generated_has_label_work=False, accepted_has_label_work=False)
        return res
    group_hit: List[bool] = []
    group_pending: List[bool] = []
    label_hit: Set[int] = set()
    for g in accepted:
        hits, pending = _model_group_matches(label["item_no"], pairs, side, names, g)
        group_hit.append(bool(hits))
        group_pending.append(pending and not hits)
        label_hit |= hits
    rej_hit, rej_pending = False, False
    for g in rejected:
        hits, pending = _model_group_matches(label["item_no"], pairs, side, names, g)
        rej_hit = rej_hit or bool(hits)
        rej_pending = rej_pending or (pending and not hits)
    res["rejected_contains_label_work"] = rej_hit if not (rej_pending and not rej_hit) else "pending"
    res["generated_has_label_work"] = bool(label_hit) or rej_hit
    res["accepted_has_label_work"] = bool(label_hit)
    if any(group_pending):
        res.update(outcome="pending_name_pair", wrong_work_link="pending", pending=True,
                   accepted_wrong_works="pending")
        return res
    res["accepted_wrong_works"] = sum(not h for h in group_hit)
    if not linked:
        res.update(outcome="missed_abstain" if model["decision"] == "abstain" else "missed_rejected",
                   wrong_work_link="no")
        return res
    # 채택 작품이 모두 라벨 작품이면 정답, 일부만이면 부분(잘못된 작품도 함께 연결), 하나도 아니면 오답
    all_groups_ok = all(group_hit)
    outcome = "correct" if all_groups_ok else ("partial" if label_hit else "wrong")
    res.update(outcome=outcome, wrong_work_link="no" if all_groups_ok else "yes")
    return res


V72_STABLE_GAINS = ("n033", "n043", "n096", "c707")
SPECIAL = {"q105": "name_in_query", "n031": "name_in_query", "n096": "work_name_equals_song_title",
           "n045": "applied_input_identical", "t024": "applied_input_identical",
           "c603": "abstained", "n010": "abstained", "t036": "abstained", "t041": "abstained", "t098": "abstained",
           "t043": "validation_rejected"}
# 해석 묶음(검색 효과와 교차할 때). 보류는 작품을 쓰지 않았으므로 정오 묶음과 따로 둔다
BUCKET = {"correct": "correct", "partial": "wrong_included", "wrong": "wrong_included", "wrong_link": "wrong_included",
          "correct_abstain": "abstain_correct", "missed_abstain": "abstain_missed", "missed_rejected": "abstain_missed"}


def _row_outcome(r: Dict[str, str], state: str, model: Dict[str, object], pairs: Pairs, as_written: bool,
                 side: str) -> Dict[str, object]:
    if state == "blank":
        return {"outcome": "label_blank", "wrong_work_link": "unlabeled", "model_linked": bool(model["accepted"]),
                "label_status": "", "pending": False}
    if state == "incomplete" or (state == "needs_confirmation" and not as_written):
        return {"outcome": f"label_{state}", "wrong_work_link": "unlabeled", "model_linked": bool(model["accepted"]),
                "label_status": STATUS.get(r.get("judgment_status", ""), ""), "pending": False}
    return classify(r, model, pairs, side)


def summarize(lab: List[Dict[str, str]], chk: Optional[List[Dict[str, str]]], imap: Dict[str, Dict[str, str]],
              model: Dict[str, Dict[str, object]], flags: Dict[str, Dict[str, str]], pairs_human: Pairs,
              pairs_agent: Pairs, states: Dict[str, Dict[str, object]],
              search: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, object]:
    variants = {
        "valid_rows__human_pairs": (False, pairs_human),
        "valid_rows__agent_assisted_pairs": (False, pairs_agent),
        "as_written__human_pairs": (True, pairs_human),
        "as_written__agent_assisted_pairs": (True, pairs_agent),
    }
    per_items: Dict[str, Dict[str, Dict[str, object]]] = {}
    result: Dict[str, object] = {"unit": "query (v71 반복 1 = 반복 2 응답 원문 22/22 동일 — 44건으로 세지 않음)"}
    for name, (as_written, pairs) in variants.items():
        per = []
        for r in lab:
            qid = imap[r["item_no"]]["qid"]
            c = _row_outcome(r, states[r["item_no"]]["state"], model[qid], pairs, as_written, "labeler")
            per.append({"item_no": r["item_no"], "qid": qid, "row_state": states[r["item_no"]]["state"],
                        "independent_self_reported": is_independent(r),
                        "known_case": flags[qid]["v67_known_case"] == "yes", "v71_status": flags[qid]["v71_status"], **c})
        per_items[name] = {p["qid"]: p for p in per}
        result[name] = {
            "all_22": _rates(per),
            "self_reported_independent_rows": _rates([p for p in per if p["independent_self_reported"]]),
            "known_10": _rates([p for p in per if p["known_case"]]),
            "other_12_not_negatives": _rates([p for p in per if not p["known_case"]]),
            "outcome_by_label_status": _crosstab(per, "label_status", "outcome"),
        }
    primary = per_items["valid_rows__human_pairs"]
    result["row_states"] = dict(sorted(Counter(v["state"] for v in states.values()).items()))
    result["special_checks"] = {q: {"flag": SPECIAL[q], **{n: _pick(per_items[n], q) for n in variants}} for q in SPECIAL}
    result["v72_v73_stable_gains"] = {
        q: {n: {**_pick(per_items[n], q), "counts_as_interpretation_evidence": _evidence(per_items[n][q])}
            for n in variants}
        for q in V72_STABLE_GAINS
    }
    result["pending_name_pair_items"] = {n: sorted(q for q, p in per_items[n].items() if p["outcome"] == "pending_name_pair")
                                         for n in variants}
    result["model_stages"] = _model_stages(model)
    if search is not None:
        result["search_cross"] = {n: _search_cross(per_items[n], search) for n in variants}
    if chk is not None:
        same = {r["item_no"]: agreement(r, r_by(chk, r["item_no"]), pairs_human) for r in lab}
        result["labeler_vs_checker"] = dict(sorted(Counter(same.values()).items()))
    return {"summary": result, "per_item": per_items, "primary": primary}


def _model_stages(model: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """① 생성 ② 검증 통과(채택) ③ 보조 검색 사용. 채택 작품은 언급 곡이 있어야 통과하므로 ②=③(작품 단위)이다.
    리랭커 입력이 실제로 바뀐 질의 수는 v72·v73 기록(13)에서 따로 읽는다."""
    dec = Counter(m["decision"] for m in model.values())
    return {
        "queries": len(model),
        "decision_candidates": dec.get("candidates", 0),
        "decision_abstain": dec.get("abstain", 0),
        "queries_with_accepted_work": sum(bool(m["accepted"]) for m in model.values()),
        "queries_all_rejected": sum(bool(m["rejected"]) and not m["accepted"] for m in model.values()),
        "works_generated": sum(len(m["accepted"]) + len(m["rejected"]) for m in model.values()),
        "works_accepted_used_in_aux": sum(len(m["accepted"]) for m in model.values()),
        "works_rejected": sum(len(m["rejected"]) for m in model.values()),
        "reject_reasons": dict(Counter(x for m in model.values() for x in m["reasons"])),
        "queries_name_in_query": sorted(q for q, m in model.items() if m["name_in_query"]),
    }


def _in30(rank: str) -> bool:
    return bool(rank) and rank != ">30" and int(rank) <= 30


def _search_cross(per: Dict[str, Dict[str, object]], search: Dict[str, Dict[str, str]]) -> Dict[str, object]:
    """v73 attempt3 질의별 기록(재실행 없음)과 교차. 검색 성공을 해석 정답 근거로 쓰지 않는다 — 나란히 셀 뿐이다."""
    rows = []
    for qid, p in per.items():
        s = search[qid]
        a, b = _in30(s["A_cand_rank"]), _in30(s["B_cand_rank"])
        cand = "gain" if b and not a else "loss" if a and not b else "kept_in" if a else "kept_out"
        bucket = BUCKET.get(str(p["outcome"]), "undetermined")
        rows.append({"qid": qid, "interpretation": bucket, "candidate_recall": cand, "hit10": s["class"],
                     "rerank_input": s["group"]})
    return {
        "interpretation_x_candidate": _crosstab(rows, "interpretation", "candidate_recall"),
        "interpretation_x_hit10": _crosstab(rows, "interpretation", "hit10"),
        "interpretation_x_rerank_input": _crosstab(rows, "interpretation", "rerank_input"),
        "per_query": {r["qid"]: {k: r[k] for k in ("interpretation", "candidate_recall", "hit10", "rerank_input")}
                      for r in sorted(rows, key=lambda x: x["qid"])},
    }


def _crosstab(rows: Iterable[Dict[str, object]], a: str, b: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = {}
    for r in rows:
        out.setdefault(str(r[a]) or "(none)", Counter())[str(r[b])] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _evidence(p: Dict[str, object]) -> str:
    """v72 §4-6: 잘못된 작품 연결이 확인된 이득은 의도 해석 근거로 세지 않는다. 판단 불가·판정 대기는 unknown."""
    if p["wrong_work_link"] == "yes":
        return "no"
    if p["outcome"] == "correct":
        return "yes"
    return "unknown"


def _pick(per: Dict[str, Dict[str, object]], qid: str) -> Dict[str, object]:
    p = per[qid]
    return {"outcome": p["outcome"], "wrong_work_link": p["wrong_work_link"], "label_status": p["label_status"]}


def _rates(per: List[Dict[str, object]]) -> Dict[str, object]:
    oc = Counter(p["outcome"] for p in per)
    linked = [p for p in per if p["model_linked"]]
    judgeable = [p for p in linked if p["outcome"] in ("correct", "partial", "wrong", "wrong_link")]
    not_counted = {k: v for k, v in oc.items() if k.startswith("label_") or k == "pending_name_pair"}

    def frac(n: int, d: int) -> str:
        return f"{n}/{d}" if d else "0/0"
    return {
        "items": len(per),
        "outcomes": dict(sorted(oc.items())),
        "model_linked": len(linked),
        "linked_judgeable": len(judgeable),
        "correct_of_judgeable": frac(oc["correct"], len(judgeable)),
        "partial_of_judgeable": frac(oc["partial"], len(judgeable)),
        "wrong_of_judgeable": frac(oc["wrong"] + oc["wrong_link"], len(judgeable)),
        "wrong_work_link_yes": sum(p["wrong_work_link"] == "yes" for p in per),
        "model_abstain_or_rejected": frac(sum(not p["model_linked"] for p in per), len(per)),
        "correct_abstain": oc["correct_abstain"],
        "missed": oc["missed_abstain"] + oc["missed_rejected"],
        "not_counted": dict(sorted(not_counted.items())),
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    private_dir, log_path = Path(args.private_dir), Path(args.freeze_log)
    if not (private_dir / "stage2" / "opened.json").exists():
        raise SystemExit("2단계(compare)를 먼저 연다")
    opened = _verify_model_copy(private_dir)
    lab = frozen_sheet(private_dir, log_path, "labeler", args.revision)
    if lab is None:
        raise SystemExit(f"라벨러 고정본 r{args.revision} 없음")
    chk = frozen_sheet(private_dir, log_path, "checker")
    imap = {r["item_no"]: r for r in read_csv(private_dir / "item_map.csv")}
    model = load_model_outputs(private_dir / "stage2" / "model_output_review_v71.csv")
    flags = load_review_flags(Path(args.review_flags))
    pairs_path = private_dir / "name_pairs.csv"
    pairs_h, pair_rows = _load_pairs(pairs_path)
    pairs_a, _ = _load_pairs(pairs_path, use_agent=True)
    states = row_states(lab, _packet_rows(private_dir))
    search = {r["qid"]: r for r in read_csv(Path(args.v73_per_query))} if args.v73_per_query else None
    out = summarize(lab, chk, imap, model, flags, pairs_h, pairs_a, states, search)
    added = write_pending_pairs(pairs_path, pairs_h, pair_rows)
    cols = ("item_no", "qid", "row_state", "independent_self_reported", "known_case", "v71_status", "label_status",
            "model_linked", "generated_works", "accepted_works", "generated_has_label_work", "accepted_has_label_work",
            "accepted_wrong_works", "rejected_contains_label_work", "outcome", "wrong_work_link", "pending")
    for name, per in out["per_item"].items():
        write_csv(private_dir / "stage2" / f"per_item_outcome__{name}.csv", cols,
                  sorted(per.values(), key=lambda x: x["item_no"]))
    entry = next(e for e in _freeze_log(log_path) if e["role"] == "labeler" and e["revision"] == args.revision)
    chk_entry = next((e for e in _freeze_log(log_path) if e["role"] == "checker" and e["revision"] == 1), None)
    summary = {"made_at": now_kst(), "labeler_revision": args.revision, "labeler_sha256": entry["sha256"],
               "labeler_kind": entry.get("kind", ""), "labeler_ai_assisted": bool(entry.get("ai_assisted")),
               "freeze_log_sha256": sha256_file(log_path),
               "checker": bool(chk_entry),
               # README §4: 2단계를 연 뒤 고정한 확인자 라벨은 독립 확인으로 쓰지 않는다
               "checker_independent": bool(chk_entry) and not chk_entry["after_stage2_opened"],
               "name_pairs_sha256": sha256_file(pairs_path),
               "stage2_opened_at": opened["opened_at"], "model_outputs_sha256": opened["model_outputs_sha256"],
               "v73_per_query_sha256": sha256_file(Path(args.v73_per_query)) if args.v73_per_query else "",
               **out["summary"], "pending_name_pairs_added": added}
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pending = summary["pending_name_pair_items"]["valid_rows__human_pairs"]
    print(json.dumps({"primary": summary["valid_rows__human_pairs"]["all_22"], "pending_items": pending},
                     ensure_ascii=False))
    return 2 if pending else 0


# ---------- CLI ----------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-dir", default=str(PRIVATE))
    ap.add_argument("--freeze-log", default=str(HERE / "freeze_log.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True, help="v71 labels/work_label_queue_private.csv(질의 원문, Git 제외)")
    p.add_argument("--tracked-queue", default=str(HERE.parent / "results_v71_work_name_resolution_live/labels/work_label_queue.csv"))
    p.add_argument("--manifest", default=str(HERE / "packet_manifest.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("--sheet", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--role", choices=("labeler", "checker"), required=True)
    p.add_argument("--sheet", required=True)
    p.add_argument("--reason", default="")
    p.add_argument("--provisional", action="store_true", help="형식 오류·미완료 행이 있어도 원본 보존용으로 고정(행 상태를 기록)")
    p.add_argument("--kind", default="as_received", choices=("as_received", "mechanical_cleanup", "human_adjusted"))
    p.add_argument("--ai-assisted", action="store_true", help="생성형 AI가 작성·수정에 관여한 판(같은 역할의 뒤 판에도 이어짐)")
    p = sub.add_parser("normalize")
    p.add_argument("--sheet", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("agree")
    p.add_argument("--out", default=str(HERE / "agreement_summary.json"))
    p.add_argument("--revision", type=int, default=1)
    flags_default = str(HERE.parent / "results_v72_work_name_rerank_diagnosis/labels/review_after_labels.csv")
    p = sub.add_parser("compare")
    p.add_argument("--model-outputs", required=True, help="v71 analysis_private/model_output_review.csv(Git 제외)")
    p.add_argument("--review-flags", default=flags_default)
    p.add_argument("--no-checker", action="store_true")
    p.add_argument("--revision", type=int, default=1)
    p = sub.add_parser("summarize")
    p.add_argument("--review-flags", default=flags_default)
    p.add_argument("--revision", type=int, default=1)
    p.add_argument("--v73-per-query", default=str(HERE.parent / "results_v73_work_name_full_ab/attempt3/analysis/per_query.csv"))
    p.add_argument("--out", default=str(HERE / "label_summary.json"))
    args = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "check": cmd_check, "freeze": cmd_freeze, "normalize": cmd_normalize, "agree": cmd_agree,
            "compare": cmd_compare, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
