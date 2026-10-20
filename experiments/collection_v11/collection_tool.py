"""v11 독립 평가 자료 — 보관자용 수집·검사·분할·봉인 도구(모델·검색·외부 호출 0).

에이전트·도구는 질의를 쓰거나 정답을 정하지 않는다. 사람이 채워 보낸 표를 받은 그대로 보존하고, 형식을 검사하고,
수집 전에 고정한 규칙(DATA_SPEC.md)대로 묶고 나누고 센다. 화면과 공유용 출력에는 건수만 찍는다(질의 원문·곡 ID·제목 없음).

    init                보관자 폴더 만들기(Git 작업 트리·/tmp 안이면 거부)
    commit-salt         salt 생성(또는 가져오기) + 해시 약속 기록. 두 번째 실행은 거부
    intake              받은 표를 바이트 그대로 읽기 전용 보존 + 해시 기록. 이미 받은 항목은 --reason 수정본으로만
    check               현재 판 전체 형식·일관성 검사(오류 코드·건수만)
    make-writer-packet  작성자에게 줄 빈 표(항목 번호 미리 채움)·자기 보고 표
    make-checker-sheet  확인자에게 줄 표(작성자 표에서 질의·작성자 정답 추정만 옮김 — 검색 결과 없음)
    export-corpus-meta  확인자용 코퍼스 메타 표(song_id·제목·가수·앨범·발매일만 — 파이프라인 산출물 없음)
    build               작성·확인·자기 보고·코퍼스·기존 세트를 합쳐 묶음(pool)·표시(flag)·군집 키 계산
    split               salt 기반 군집 단위 분할. 한 번 배정한 항목은 옮기지 않는다(추가 기록 장부)
    quarantine          배정 충돌 항목을 양쪽 분할에서 빼는 보관자 결정(사유 필수)
    export              policy_dev / sealed_eval 내보내기(판 번호, 읽기 전용, 덮어쓰지 않음) + 데이터 고정 시점
    event               채택 기준 고정·모델 출력 공개·봉인 열기 같은 시점 기록(봉인 열기는 1회)
    deliver             보관자 폴더 밖으로 보낸 사본 기록(봉인 내보내기는 거부)
    withdraw            참여 철회 — 받은 표·사본·내보내기·파생 작업 파일을 지우고, 밖으로 보낸 사본은 삭제 요청으로 남긴다
    confirm-deletion    밖으로 보낸 사본의 삭제 확인 기록
    status              익명 집계(COLLECTION_STATUS.md·json) — 개발 세션에 넘기는 유일한 산출물
    verify              보존본·내보내기 해시와 장부 사슬 확인

v68 `new_data_check.py`의 글자 3-gram Jaccard·기존 세트 읽기·salt 해시 배정을 그대로 가져다 쓴다(그 파일은 고치지 않는다).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V68_DIR = ROOT / "experiments/reranking/results_v68_audio_skip_live/new_data"
sys.path.insert(0, str(V68_DIR))
import new_data_check as v68  # noqa: E402  (grams · jaccard · existing_queries · split_of · NEAR_DUP)

KST = dt.timezone(dt.timedelta(hours=9))
# 엑셀·구글 시트가 바꿔 놓는 모양(2026/10/21 9:05, 2026.10.21 09:05:00)도 받아 같은 꼴로 비교한다
TIME_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\.? (\d{1,2}):(\d{2})(?::\d{2})?$")
PERSON_RE = re.compile(r"^P\d{2}$")
ENTRY_RE = re.compile(r"^(P\d{2})-(\d{3})$")
SONG_ID_RE = re.compile(r"^\d+$")

# ---------- 열과 값(DATA_SPEC.md §2와 같아야 한다) ----------

WRITER_COLS = (
    "entry_id", "writer_code", "track", "written_at", "query",
    "conversation_id", "turn_no", "prior_question", "turn_origin",
    "memory_source", "ai_used", "written_before_lookup", "used_system_before",
    "title_recall", "artist_recall", "remembered_aspects",
    "work_reference", "work_name_after_writing", "work_media_type", "writer_confidence",
    "answer_noted_at", "writer_answer_title", "writer_answer_artist", "writer_answer_other",
    "edit_confirmed_by_writer", "edit_note", "writer_notes",
)
CHECKER_INPUT_COLS = ("entry_id", "query", "prior_question", "writer_answer_title", "writer_answer_artist",
                      "writer_answer_other", "work_name_after_writing")
CHECKER_FILL_COLS = (
    "answer_status", "answer_ids", "multiple_answers", "in_corpus", "ooc_title", "ooc_artist",
    "evidence", "confidence", "wrong_clue_types", "wrong_clue_note",
    "work_check_result", "work_in_song_data", "checker_code", "checked_at", "saw_search_results", "checker_notes",
)
CHECKER_COLS = CHECKER_INPUT_COLS + CHECKER_FILL_COLS
DECL_COLS = ("person_code", "declared_at", "relation", "involved_in_search_dev", "saw_existing_eval_queries",
             "saw_dev_results", "saw_failure_lists", "used_system_before", "consent", "consent_at", "notes")

TRACKS = {"general", "diag_work", "diag_sound"}
MEMORY = {"own_memory", "helped_recall", "synthetic"}
AI_USED = {"no", "edited", "generated"}
YES_NO = {"yes", "no"}
YES_NO_UNSURE = {"yes", "no", "unsure"}
RECALL = {"exact", "partial", "none"}
ASPECTS = {"lyrics", "melody", "voice", "instrument_sound", "cover", "mv", "work", "place_situation", "era",
           "performer_type", "title_shape", "other", "unsure"}
MEDIA = {"drama", "film", "variety", "audition", "animation", "webtoon", "musical", "documentary", "game", "ad",
         "other", ""}
CONFIDENCE = {"high", "medium", "low"}
TURN_ORIGIN = {"real_exchange", "constructed"}
ANSWER_STATUS = {"confirmed", "ambiguous", "undeterminable", "pending"}
MULTI = {"no", "versions", "distinct"}
IN_CORPUS = {"yes", "no", "partial", "unknown"}
EVIDENCE = {"writer_named", "metadata", "cover", "listened", "lyrics_source", "external_source", "writer_interview",
            "other"}
WRONG = {"none", "gender", "era", "genre", "group_solo", "instrument", "voice", "lyrics", "work", "cover", "mv",
         "title_shape", "other"}
WORK_CHECK = {"same", "different", "unsure", ""}
WORK_IN_DATA = {"yes", "no", "unknown", ""}
RELATION = {"team", "acquaintance", "other"}

# 묶음(pool) — 앞에 있을수록 우선. 분할 대상은 SPLIT_POOLS뿐이다(DATA_SPEC.md §4)
SPLIT_POOLS = ("in_corpus", "ooc", "ambiguous", "undeterminable")
SHORT_LEN = 15

KEY_COL = {"writer": "entry_id", "checker": "entry_id", "declaration": "person_code"}


# ---------- 공통 ----------

def now_kst() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    """UTF-8(BOM 포함) 또는 엑셀이 다시 저장한 CP949. 셀 안 줄바꿈(따옴표 안 \\n)은 한 칸으로 보존한다."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    return [{(k or "").strip(): (v if v is not None else "") for k, v in r.items()} for r in rows]


def header_of(path: Path) -> List[str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    first = next(csv.reader(io.StringIO(text, newline="")), [])
    return [c.strip() for c in first]


def write_csv(path: Path, cols: Sequence[str], rows: Iterable[Dict[str, str]], bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def readonly(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def norm(text: str) -> str:
    """NFKC·소문자·글자와 숫자만 — 작품명·앨범명·제목 비교용."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def multi(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"[;|]", str(text or "")) if p.strip()]


def ids_of(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split("|") if p.strip()]


def stripped(row: Dict[str, str], col: str) -> str:
    return str(row.get(col, "") or "").strip()


# ---------- 추가 기록 장부(줄마다 앞 줄 해시를 품는다 — 고쳐 쓰면 verify가 잡는다) ----------

def read_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(path: Path, entry: Dict[str, object]) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    prev = sha256_text(lines[-1]) if lines else "genesis"
    entry = {**entry, "prev_sha": prev}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def chain_ok(path: Path) -> bool:
    if not path.exists():
        return True
    prev = "genesis"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("prev_sha") != prev:
            return False
        prev = sha256_text(line)
    return True


# ---------- 보관자 폴더 ----------

UNSAFE_PREFIXES = ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")


def inside_git_tree(path: Path) -> bool:
    for p in [path, *path.parents]:
        if (p / ".git").exists():
            return True
    return False


def location_problem(path: Path) -> Optional[str]:
    resolved = str(path.resolve())
    if any(resolved == u or resolved.startswith(u + "/") for u in UNSAFE_PREFIXES):
        return "임시 폴더 안(/tmp·/private/tmp·/var/folders) — 개발 세션·에이전트 스크래치와 섞인다"
    if inside_git_tree(path.resolve()):
        return "Git 작업 트리 안 — 실수로 스테이징되거나 개발 세션이 읽을 수 있다"
    return None


class Private:
    def __init__(self, root: Path):
        self.root = root
        self.logs = root / "logs"
        self.intake_log = self.logs / "intake_log.jsonl"
        self.ledger = self.logs / "split_ledger.jsonl"
        self.events = self.logs / "events.jsonl"
        self.export_log = self.logs / "export_log.jsonl"
        self.salt = root / "salt.txt"
        self.commitment = self.logs / "salt_commitment.json"
        self.roles = root / "roles.json"
        self.work = root / "work"
        self.merged = self.work / "merged.csv"

    def require(self) -> "Private":
        info = self.root / "LOCATION.json"
        if not info.exists():
            raise SystemExit(f"보관자 폴더가 초기화되지 않았다: init 먼저 ({self.root})")
        return self


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.private_dir)
    problem = location_problem(root.parent if not root.exists() else root)
    if problem and not args.allow_unsafe_location:
        raise SystemExit(f"보관자 폴더 위치 거부: {problem}. 보관자 개인 드라이브의 폴더를 쓴다")
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("received/writer", "received/checker", "received/declaration", "work", "exports", "packets", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    info = root / "LOCATION.json"
    if not info.exists():
        info.write_text(json.dumps({"created_at": now_kst(), "unsafe_location": bool(problem),
                                    "unsafe_reason": problem or ""}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"initialized": True, "unsafe_location": bool(problem)}, ensure_ascii=False))
    return 0


# ---------- salt ----------

def cmd_commit_salt(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    if p.commitment.exists() or p.salt.exists():
        raise SystemExit("salt와 해시 약속이 이미 있다 — 바꾸지 않는다(분할을 다시 뽑는 길을 막는다). 새 수집 회차는 새 보관자 폴더로")
    if args.import_file:
        salt = Path(args.import_file).read_text(encoding="utf-8").strip()
        if len(salt) < 32:
            raise SystemExit("salt가 너무 짧다(32자 이상)")
    else:
        salt = secrets.token_hex(32)
    p.salt.write_text(salt + "\n", encoding="utf-8")
    readonly(p.salt)
    commitment = {"committed_at": now_kst(), "salt_sha256": sha256_text(salt), "sealed_ratio": args.sealed_ratio}
    p.commitment.write_text(json.dumps(commitment, ensure_ascii=False) + "\n", encoding="utf-8")
    readonly(p.commitment)
    append_log(p.events, {"type": "salt_committed", "at": commitment["committed_at"],
                          "salt_sha256": commitment["salt_sha256"], "sealed_ratio": args.sealed_ratio})
    # 해시는 공개해도 salt를 알 수 없다 — 개발 쪽 상태 문서에 실어 나중에 salt를 바꾸지 않았음을 보인다
    print(json.dumps({"salt_sha256": commitment["salt_sha256"], "sealed_ratio": args.sealed_ratio}, ensure_ascii=False))
    return 0


def load_salt(p: Private) -> Tuple[str, float]:
    if not p.commitment.exists() or not p.salt.exists():
        raise SystemExit("salt 약속이 없다: commit-salt 먼저")
    commitment = json.loads(p.commitment.read_text(encoding="utf-8"))
    salt = p.salt.read_text(encoding="utf-8").strip()
    if sha256_text(salt) != commitment["salt_sha256"]:
        raise SystemExit("salt가 약속한 해시와 다르다 — 분할을 멈춘다(salt를 바꾸면 배정이 바뀐다)")
    return salt, float(commitment["sealed_ratio"])


# ---------- 받기(intake) ----------

def meaningful(row: Dict[str, str], kind: str) -> bool:
    """빈 양식 행(번호만 미리 적힌 행)은 받지 않은 것으로 본다."""
    if kind == "writer":
        return any(stripped(row, c) for c in WRITER_COLS if c not in ("entry_id", "writer_code", "track"))
    if kind == "checker":
        return any(stripped(row, c) for c in CHECKER_FILL_COLS)
    return any(stripped(row, c) for c in DECL_COLS if c != "person_code")


def withdrawn_files(p: Private) -> Set[str]:
    """참여 철회로 지운 보존본·내보내기(events 기록). 지운 파일은 해시 검사에서 빼고 현재 판에도 넣지 않는다."""
    out: Set[str] = set()
    for e in read_log(p.events):
        if e["type"] == "withdrawn":
            out |= set(e.get("files_deleted", [])) | set(e.get("exports_deleted", [])) | set(e.get("copies_deleted", []))
    return out


def current_rows(p: Private, kind: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, object]]]:
    """받은 순서대로 적용한 현재 판. 수정본은 그 항목의 행 전체를 바꾼다. meta: 최초 질의·판 번호·받은 파일."""
    state: Dict[str, Dict[str, str]] = {}
    meta: Dict[str, Dict[str, object]] = {}
    key = KEY_COL[kind]
    gone = withdrawn_files(p)
    for e in read_log(p.intake_log):
        if e["kind"] != kind or e["stored"] in gone:
            continue
        path = p.root / e["stored"]
        for r in read_csv(path):
            if not meaningful(r, kind):
                continue
            k = stripped(r, key)
            if not k:
                continue
            m = meta.setdefault(k, {"revision": 0, "files": [], "original_query": r.get("query", "")})
            m["revision"] = int(m["revision"]) + 1
            m["files"].append(e["stored"])
            m["received_at"] = e["received_at"]
            state[k] = r
    return state, meta


def cmd_intake(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    kind, code = args.kind, args.code
    if not PERSON_RE.match(code):
        raise SystemExit("사람 코드는 P01 형식이다(이름을 쓰지 않는다)")
    src = Path(args.file)
    digest = sha256_file(src)
    log = read_log(p.intake_log)
    if any(e["sha256"] == digest for e in log):
        print(json.dumps({"intake": "already_received", "sha256": digest[:12]}, ensure_ascii=False))
        return 0
    expected = {"writer": WRITER_COLS, "checker": CHECKER_COLS, "declaration": DECL_COLS}[kind]
    header = header_of(src)
    missing = [c for c in expected if c not in header]
    if missing:
        raise SystemExit(f"열 없음 {len(missing)}개: {', '.join(missing)} — 받은 파일은 보존하지 않았다")
    rows = [r for r in read_csv(src) if meaningful(r, kind)]
    key = KEY_COL[kind]
    keys = [stripped(r, key) for r in rows]
    blank_keys = sum(1 for k in keys if not k)
    if blank_keys:
        raise SystemExit(f"{key}가 빈 행 {blank_keys}건 — 형식만 되물어 다시 받는다")
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        raise SystemExit(f"한 파일 안 {key} 중복 {len(dup)}개 — 형식만 되물어 다시 받는다")
    if kind == "writer":
        bad_owner = [r for r in rows if stripped(r, "writer_code") != code
                     or not (ENTRY_RE.match(stripped(r, "entry_id")) and stripped(r, "entry_id")[:3] == code)]
        if bad_owner:
            raise SystemExit(f"작성자 코드·항목 번호가 {code}와 맞지 않는 행 {len(bad_owner)}건")
    if kind == "checker" and any(stripped(r, "checker_code") != code for r in rows):
        raise SystemExit(f"확인자 코드가 {code}가 아닌 행이 있다")
    owners = sorted({k[:3] for k in keys}) if kind == "checker" else [code]
    if kind == "checker" and len(owners) != 1:
        raise SystemExit("확인자 표 한 파일에는 한 작성자의 항목만 담는다(참여 철회 때 그 파일만 지우기 위해)")
    if kind == "declaration" and any(stripped(r, "person_code") != code for r in rows):
        raise SystemExit(f"자기 보고 표의 person_code가 {code}가 아니다")
    state, _ = current_rows(p, kind)
    overlap = sorted(set(keys) & set(state))
    changed: Dict[str, List[str]] = {}
    if overlap and not args.reason:
        raise SystemExit(f"이미 받은 항목 {len(overlap)}건 — 수정본은 --reason·--editor를 적어 새 판으로만 받는다(최초본은 그대로)")
    if args.reason:
        if not args.editor:
            raise SystemExit("수정본은 --editor(수정한 사람 코드)가 필요하다")
        new_keys = sorted(set(keys) - set(state))
        if new_keys:
            raise SystemExit(f"수정본에 처음 보는 항목 {len(new_keys)}건 — 새 항목은 수정본과 섞지 않고 따로 받는다")
        for r in rows:
            k = stripped(r, key)
            cols = [c for c in expected if c in r and r.get(c, "") != state[k].get(c, "")]
            changed[k] = cols
            if kind == "writer" and "query" in cols:
                if stripped(r, "edit_confirmed_by_writer") != "yes" or not stripped(r, "edit_note"):
                    raise SystemExit("질의 원문이 바뀐 행은 edit_confirmed_by_writer=yes와 edit_note가 있어야 한다 — 받지 않았다")
    n = sum(1 for e in log if e["kind"] == kind and e["code"] == code) + 1
    dest_rel = Path("received") / kind / f"{code}_{n:03d}_{digest[:8]}.csv"
    dest = p.root / dest_rel
    if dest.exists():
        raise SystemExit(f"이미 있는 보존본: {dest_rel}")
    shutil.copyfile(src, dest)
    readonly(dest)
    entry = {"kind": kind, "code": code, "owner": owners[0], "seq": n, "stored": str(dest_rel), "sha256": digest,
             "received_at": now_kst(), "rows": len(rows), "mode": "revision" if args.reason else "new",
             "reason": args.reason or "", "editor": args.editor or "",
             "changed_columns": dict(sorted(Counter(c for cols in changed.values() for c in cols).items()))}
    append_log(p.intake_log, entry)
    print(json.dumps({k: entry[k] for k in ("kind", "code", "seq", "rows", "mode")}, ensure_ascii=False))
    return 0


# ---------- 검사 ----------

def canon_time(text: str) -> str:
    """`YYYY-MM-DD HH:MM`으로 맞춘 값. 형식이 아니면 빈 문자열."""
    m = TIME_RE.match((text or "").strip())
    if not m:
        return ""
    y, mo, d, h, mi = (int(x) for x in m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def time_ok(text: str) -> bool:
    return bool(canon_time(text))


def validate_writer(r: Dict[str, str]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    eid = stripped(r, "entry_id")
    if not ENTRY_RE.match(eid):
        errors.append("bad_entry_id")
    if not stripped(r, "query"):
        errors.append("empty_query")
    if not stripped(r, "written_at"):
        errors.append("missing_written_at")
    elif not time_ok(stripped(r, "written_at")):
        errors.append("bad_written_at")
    for col, allowed in (("track", TRACKS), ("memory_source", MEMORY), ("ai_used", AI_USED),
                         ("written_before_lookup", YES_NO), ("used_system_before", YES_NO),
                         ("title_recall", RECALL), ("artist_recall", RECALL), ("work_reference", YES_NO_UNSURE),
                         ("writer_confidence", CONFIDENCE)):
        if stripped(r, col) not in allowed:
            errors.append(f"bad_{col}")
    if stripped(r, "work_media_type") not in MEDIA:
        errors.append("bad_work_media_type")
    aspects = multi(r.get("remembered_aspects", ""))
    if not aspects or any(a not in ASPECTS for a in aspects):
        errors.append("bad_remembered_aspects")
    if stripped(r, "work_reference") == "yes" and not stripped(r, "work_name_after_writing"):
        errors.append("work_name_missing")
    conv, turn = stripped(r, "conversation_id"), stripped(r, "turn_no")
    if conv or turn:
        if not conv or not turn.isdigit() or int(turn) < 1:
            errors.append("bad_turn")
        if stripped(r, "turn_origin") not in TURN_ORIGIN:
            errors.append("bad_turn_origin")
        if turn.isdigit() and int(turn) > 1 and not stripped(r, "prior_question"):
            errors.append("turn_without_prior_question")
    answered = any(stripped(r, c) for c in ("writer_answer_title", "writer_answer_artist", "writer_answer_other"))
    noted = stripped(r, "answer_noted_at")
    if answered:
        if not noted:
            errors.append("missing_answer_noted_at")
        elif not time_ok(noted):
            errors.append("bad_answer_noted_at")
        elif time_ok(stripped(r, "written_at")) and canon_time(noted) < canon_time(stripped(r, "written_at")) \
                and stripped(r, "written_before_lookup") == "yes":
            errors.append("answer_noted_before_writing")  # 기억 먼저 썼다는 표시와 시각이 어긋난다
    for col, val in r.items():
        if "\n" in str(val or "") or "\r" in str(val or ""):
            warnings.append(f"cell_newline:{col}")
    return errors, warnings


def validate_checker(r: Dict[str, str], writer: Optional[Dict[str, str]], corpus: Set[str]) -> List[str]:
    errors: List[str] = []
    if writer is None:
        return ["unknown_entry"]
    status = stripped(r, "answer_status")
    if status not in ANSWER_STATUS:
        errors.append("bad_answer_status")  # 모르는 상태를 확인됨으로 치지 않는다
    if not PERSON_RE.match(stripped(r, "checker_code")):
        errors.append("bad_checker_code")
    checked = stripped(r, "checked_at")
    if not checked:
        errors.append("missing_checked_at")
    elif not time_ok(checked):
        errors.append("bad_checked_at")
    else:
        for col in ("written_at", "answer_noted_at"):
            w = stripped(writer, col)
            if time_ok(w) and canon_time(checked) < canon_time(w):
                errors.append(f"checked_before_{col}")
    if stripped(r, "saw_search_results") not in YES_NO:
        errors.append("bad_saw_search_results")
    in_corpus = stripped(r, "in_corpus")
    if in_corpus not in IN_CORPUS:
        errors.append("bad_in_corpus")
    multiple = stripped(r, "multiple_answers")
    if multiple not in MULTI:
        errors.append("bad_multiple_answers")
    ids = ids_of(r.get("answer_ids", ""))
    if any(not SONG_ID_RE.match(i) for i in ids):
        errors.append("bad_answer_ids")
    if len(set(ids)) != len(ids):
        errors.append("duplicate_answer_ids")
    if stripped(r, "confidence") and stripped(r, "confidence") not in CONFIDENCE:
        errors.append("bad_confidence")
    if any(e not in EVIDENCE for e in multi(r.get("evidence", ""))):
        errors.append("bad_evidence")
    if any(w not in WRONG for w in multi(r.get("wrong_clue_types", ""))):
        errors.append("bad_wrong_clue_types")
    if stripped(r, "work_check_result") not in WORK_CHECK:
        errors.append("bad_work_check_result")
    if stripped(r, "work_in_song_data") not in WORK_IN_DATA:
        errors.append("bad_work_in_song_data")
    if status == "confirmed":
        if not stripped(r, "confidence"):
            errors.append("confirmed_without_confidence")
        if not multi(r.get("evidence", "")):
            errors.append("confirmed_without_evidence")
        if in_corpus == "unknown":
            errors.append("confirmed_with_unknown_corpus")
    if in_corpus in ("yes", "partial") and not ids:
        errors.append("in_corpus_without_ids")
    if in_corpus == "yes" and any(i not in corpus for i in ids):
        errors.append("in_corpus_id_not_in_corpus")
    if in_corpus == "partial" and not (any(i in corpus for i in ids) and stripped(r, "ooc_title")):
        errors.append("bad_partial_corpus")
    if in_corpus == "no":
        if any(i in corpus for i in ids):
            errors.append("ooc_but_id_in_corpus")
        if status == "confirmed" and not stripped(r, "ooc_title"):
            errors.append("ooc_without_title")
    if multiple in ("versions", "distinct") and len(ids) + (1 if stripped(r, "ooc_title") else 0) < 2:
        errors.append("multiple_with_one_answer")
    if multiple == "no" and len(ids) > 1 and status == "confirmed":
        errors.append("one_answer_with_many_ids")
    return errors


def corpus_meta(path: Path) -> Dict[str, Dict[str, str]]:
    """song_id와 원본 메타(제목·가수·앨범·발매일)만 읽는다. 태그·요약·가사·댓글 같은 파이프라인 산출물은 읽지 않는다."""
    if not path.exists():
        raise SystemExit(f"코퍼스 파일이 없다: {path}")
    out: Dict[str, Dict[str, str]] = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m = d.get("metadata") or {}
        artist = m.get("artist")
        artist = ", ".join(artist) if isinstance(artist, list) else str(artist or "")
        out[str(d["song_id"])] = {"title": str(m.get("title") or ""), "artist": artist,
                                  "album": str(m.get("album") or ""), "release_date": str(m.get("release_date") or "")}
    return out


def check_all(p: Private, corpus: Set[str]) -> Dict[str, object]:
    writers, wmeta = current_rows(p, "writer")
    checkers, _ = current_rows(p, "checker")
    errors: Dict[str, List[str]] = {}
    warnings: Dict[str, List[str]] = {}
    for eid, r in writers.items():
        e, w = validate_writer(r)
        if e:
            errors.setdefault(eid, []).extend(e)
        if w:
            warnings.setdefault(eid, []).extend(w)
    convs: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for eid, r in writers.items():
        if stripped(r, "conversation_id"):
            convs[f"{stripped(r, 'writer_code')}:{stripped(r, 'conversation_id')}"].append(r)
    for rows in convs.values():
        turns = [stripped(r, "turn_no") for r in rows]
        problem = (len(set(turns)) != len(turns) or "1" not in turns
                   or len({stripped(r, "track") for r in rows}) > 1
                   or len({stripped(r, "turn_origin") for r in rows}) > 1)
        if problem:
            for r in rows:
                errors.setdefault(stripped(r, "entry_id"), []).append("bad_conversation")
    for eid, r in checkers.items():
        e = validate_checker(r, writers.get(eid), corpus)
        if e:
            errors.setdefault(f"checker:{eid}", []).extend(e)
    hash_problems = []
    gone = withdrawn_files(p)
    for e in read_log(p.intake_log):
        if e["stored"] in gone:
            continue
        path = p.root / e["stored"]
        if not path.exists() or sha256_file(path) != e["sha256"]:
            hash_problems.append(e["stored"])
    return {"writer_entries": len(writers), "checker_entries": len(checkers), "errors": errors,
            "warnings": warnings, "hash_mismatch": hash_problems,
            "revised_entries": sum(1 for m in wmeta.values() if int(m["revision"]) > 1)}


def cmd_check(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    report = check_all(p, set(corpus_meta(Path(args.corpus))))
    p.work.mkdir(parents=True, exist_ok=True)
    (p.work / "check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    codes = Counter(c for v in report["errors"].values() for c in v)
    print(json.dumps({"writer_entries": report["writer_entries"], "checker_entries": report["checker_entries"],
                      "error_entries": len(report["errors"]), "error_codes": dict(sorted(codes.items())),
                      "warning_entries": len(report["warnings"]), "hash_mismatch": len(report["hash_mismatch"])},
                     ensure_ascii=False))
    return 3 if report["errors"] or report["hash_mismatch"] else 0


# ---------- 확인자 표 · 코퍼스 메타 ----------

# ---------- 사본 추적(철회 때 지울 대상) ----------

def inside_private(p: Private, path: Path) -> Path:
    """질의 원문이나 사람 코드가 든 사본은 보관자 폴더 안에만 만든다 — 어디에 몇 부 있는지 기록으로 셀 수 있게."""
    resolved = path.resolve()
    if p.root.resolve() not in (resolved, *resolved.parents):
        raise SystemExit("사본은 보관자 폴더 안(예: <보관자 폴더>/packets/)에만 만든다 — 철회 때 찾아 지울 수 있게")
    return resolved


def record_copy(p: Private, path: Path, owner: str, kind: str) -> None:
    rel = str(path.resolve().relative_to(p.root.resolve()))
    append_log(p.events, {"type": "copy_made", "at": now_kst(), "path": rel, "owners": [owner], "kind": kind})


def copies(p: Private) -> List[Dict[str, object]]:
    """만든 사본(copy_made)과 내보내기. owners = 그 파일에 항목·코드가 든 작성자."""
    out = [{"path": e["path"], "owners": e["owners"], "kind": e["kind"]} for e in read_log(p.events)
           if e["type"] == "copy_made"]
    out += [{"path": e["file"], "owners": e.get("owners", []), "kind": "export"} for e in read_log(p.export_log)]
    return out


def cmd_deliver(args: argparse.Namespace) -> int:
    """보관자 폴더 밖으로 보낸 사본(확인자 표·작성자 표·policy_dev 내보내기)을 기록한다 — 철회 때 삭제 요청 대상."""
    p = Private(Path(args.private_dir)).require()
    if not args.to or not args.by:
        raise SystemExit("--to(받은 사람 코드 또는 dev)와 --by가 필요하다")
    rel = str(inside_private(p, p.root / args.file).relative_to(p.root.resolve()))
    known = {c["path"]: c for c in copies(p)}
    if rel not in known:
        raise SystemExit("기록에 없는 파일이다 — 도구가 만든 사본·내보내기만 보낸다")
    if known[rel]["kind"] == "export" and "sealed_eval" in rel:
        raise SystemExit("봉인 내보내기는 밖으로 보내지 않는다")
    append_log(p.events, {"type": "copy_delivered", "at": now_kst(), "path": rel, "to": args.to, "by": args.by,
                          "owners": known[rel]["owners"]})
    print(json.dumps({"delivered": rel.split("/")[-1], "to": args.to}, ensure_ascii=False))
    return 0


def pending_deletions(p: Private) -> List[Dict[str, object]]:
    events = read_log(p.events)
    done = {e["request"] for e in events if e["type"] == "deletion_confirmed"}
    return [r for e in events if e["type"] == "withdrawn" for r in e.get("deletion_requests", [])
            if r["id"] not in done]


def cmd_confirm_deletion(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    if not args.by or not args.note:
        raise SystemExit("--by와 --note(누가 어떻게 지웠다고 확인했는지)가 필요하다")
    if args.request not in {r["id"] for r in pending_deletions(p)}:
        raise SystemExit(f"대기 중인 삭제 요청이 아니다: {args.request}")
    append_log(p.events, {"type": "deletion_confirmed", "at": now_kst(), "request": args.request, "by": args.by,
                          "note": args.note})
    print(json.dumps({"confirmed": args.request, "pending": len(pending_deletions(p))}, ensure_ascii=False))
    return 0


def cmd_make_writer_packet(args: argparse.Namespace) -> int:
    """작성자 한 명에게 줄 빈 표(항목 번호·작성자 코드·갈래만 미리 채움)와 자기 보고·동의 표."""
    p = Private(Path(args.private_dir)).require()
    if not PERSON_RE.match(args.code):
        raise SystemExit("사람 코드는 P01 형식이다")
    if args.track not in TRACKS:
        raise SystemExit(f"갈래는 {sorted(TRACKS)} 중 하나")
    writers, _ = current_rows(p, "writer")
    issued = [e for e in read_log(p.events) if e["type"] == "packet_issued" and e.get("code") == args.code]
    used = [int(k[4:]) for k in writers if k.startswith(args.code)]
    used += [int(e["last_no"]) for e in issued]
    start = max(used, default=0) + 1
    if start + args.rows - 1 > 999:
        raise SystemExit("항목 번호가 999를 넘는다")
    out = inside_private(p, Path(args.out_dir))
    sheet = out / f"writer_sheet_{args.code}_{start:03d}.csv"
    decl = out / f"declaration_{args.code}.csv"
    if sheet.exists():
        raise SystemExit(f"이미 있는 파일: {sheet.name}")
    rows = [{"entry_id": f"{args.code}-{n:03d}", "writer_code": args.code, "track": args.track}
            for n in range(start, start + args.rows)]
    write_csv(sheet, WRITER_COLS, rows, bom=True)
    if not decl.exists():
        write_csv(decl, DECL_COLS, [{"person_code": args.code}], bom=True)
    append_log(p.events, {"type": "packet_issued", "at": now_kst(), "code": args.code, "track": args.track,
                          "first_no": start, "last_no": start + args.rows - 1})
    for f in (sheet, decl):
        record_copy(p, f, owner=args.code, kind="writer_packet")
    print(json.dumps({"sheet": sheet.name, "rows": args.rows, "track": args.track}, ensure_ascii=False))
    return 0


def cmd_make_checker_sheet(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    writers, _ = current_rows(p, "writer")
    checkers, _ = current_rows(p, "checker")
    rows = []
    for eid in sorted(writers):
        if not eid.startswith(args.writer + "-"):
            continue  # 한 파일에 한 작성자 — 참여 철회 때 그 파일만 지운다
        if eid in checkers and not args.include_checked:
            continue
        r = writers[eid]
        rows.append({c: r.get(c, "") for c in CHECKER_INPUT_COLS})
    out = inside_private(p, Path(args.out))
    if out.exists():
        raise SystemExit(f"이미 있는 파일: {out.name} — 덮어쓰지 않는다")
    write_csv(out, CHECKER_COLS, rows, bom=True)
    record_copy(p, out, owner=args.writer, kind="checker_sheet")
    print(json.dumps({"checker_sheet_rows": len(rows)}, ensure_ascii=False))
    return 0


def cmd_export_corpus_meta(args: argparse.Namespace) -> int:
    meta = corpus_meta(Path(args.corpus))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"이미 있는 파일: {out.name}")
    write_csv(out, ("song_id", "title", "artist", "album", "release_date"),
              ({"song_id": k, **v} for k, v in sorted(meta.items(), key=lambda kv: int(kv[0]))), bom=True)
    print(json.dumps({"songs": len(meta)}, ensure_ascii=False))
    return 0


# ---------- 기존 세트(누출·중복 검사 — 보관자 쪽에서만) ----------

def existing_sets(root: Path) -> Tuple[List[Set[str]], Set[str], int]:
    """v68 읽기(queries.json + eval_queries_*.csv)에 모달리티 세트와 로컬 초안(data/eval_drafts)을 더한다. 원문은 출력하지 않는다."""
    items = v68.existing_queries(root)
    extra = root / "docs/eval/queries_modality_v1.json"
    if extra.exists():
        for q in json.loads(extra.read_text(encoding="utf-8")).get("queries", []):
            items.append({"text": q.get("query", ""), "ids": set(map(str, q.get("positives") or []))})
    drafts = root / "data/eval_drafts"
    if drafts.exists():
        for f in sorted(drafts.glob("*.csv")):
            for r in read_csv(f):
                if r.get("query"):
                    items.append({"text": r["query"], "ids": {x for x in ids_of(r.get("positive_id", ""))}})
    grams = [v68.grams(str(e["text"])) for e in items if str(e["text"]).strip()]
    ids: Set[str] = set()
    for e in items:
        ids |= {str(i) for i in e["ids"]}
    return grams, ids, len(items)


# ---------- build ----------

def roles_info(p: Private) -> Dict[str, object]:
    if not p.roles.exists():
        return {}
    return json.loads(p.roles.read_text(encoding="utf-8"))


def build(p: Private, corpus: Dict[str, Dict[str, str]], repo: Path) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    report = check_all(p, set(corpus))
    writers, wmeta = current_rows(p, "writer")
    checkers, _ = current_rows(p, "checker")
    decls, _ = current_rows(p, "declaration")
    ex_grams, ex_ids, ex_n = existing_sets(repo)
    new_grams = {eid: v68.grams(r.get("query", "")) for eid, r in writers.items()}
    out: List[Dict[str, str]] = []
    for eid in sorted(writers):
        w = writers[eid]
        c = checkers.get(eid, {})
        errs = report["errors"].get(eid, []) + report["errors"].get(f"checker:{eid}", [])
        flags: List[str] = []
        best = max((v68.jaccard(new_grams[eid], g) for g in ex_grams), default=0.0)
        status = stripped(c, "answer_status")
        in_corpus = stripped(c, "in_corpus")
        ids = ids_of(c.get("answer_ids", ""))
        if stripped(w, "ai_used") != "no" or stripped(w, "memory_source") == "synthetic":
            pool = "ai_assisted"
        elif errs:
            pool = "invalid"
        elif best >= v68.NEAR_DUP:
            pool = "excluded_near_dup"
        elif not c:
            pool = "unchecked"
        elif status == "pending":
            pool = "pending"
        elif status == "ambiguous":
            pool = "ambiguous"
        elif status == "undeterminable":
            pool = "undeterminable"
        elif in_corpus == "no":
            pool = "ooc"
        else:
            pool = "in_corpus"
        if ids and any(i in ex_ids for i in ids):
            flags.append("known_song")  # 기존 개발 세트의 정답 곡 — 새 문장 검증. 빼지 않고 따로 센다
        elif pool == "in_corpus":
            flags.append("new_song")
        if in_corpus == "partial":
            flags.append("partial_corpus")
        if stripped(w, "written_before_lookup") == "no":
            flags.append("written_after_lookup")
        if stripped(w, "used_system_before") == "yes":
            flags.append("used_system_before")
        if stripped(w, "memory_source") == "helped_recall":
            flags.append("helped_recall")
        if stripped(w, "title_recall") == "exact":
            flags.append("title_known")
        if len(re.sub(r"\s+", "", w.get("query", ""))) <= SHORT_LEN:
            flags.append("short")
        if stripped(w, "turn_origin") == "constructed":
            flags.append("constructed_turn")
        if c and stripped(c, "checker_code") == stripped(w, "writer_code"):
            flags.append("self_checked")
        if stripped(c, "saw_search_results") == "yes":
            flags.append("checker_saw_search")
        d = decls.get(stripped(w, "writer_code"), {})
        if not d:
            flags.append("no_declaration")
        elif any(stripped(d, col) in ("yes", "unsure") for col in
                 ("saw_existing_eval_queries", "saw_dev_results", "saw_failure_lists")):
            flags.append("writer_exposed")
        if d and stripped(d, "involved_in_search_dev") == "yes":
            flags.append("writer_is_developer")
        if any(f.startswith("cell_newline") for f in report["warnings"].get(eid, [])):
            flags.append("cell_newline")
        if int(wmeta[eid]["revision"]) > 1:
            flags.append("revised")
        if wmeta[eid]["original_query"] != w.get("query", ""):
            flags.append("query_edited")
        qn = norm(w.get("query", ""))
        for i in ids:
            m = corpus.get(i)
            if not m:
                continue
            names = [norm(m["title"])] + [norm(a) for a in m["artist"].split(",")]
            if any(len(n) >= 2 and n in qn for n in names):
                flags.append("names_answer_in_query")
                break
        keys: List[str] = [f"song:{i}" for i in ids]
        if stripped(c, "ooc_title"):
            keys.append(f"ooc:{norm(c.get('ooc_title', ''))}|{norm(c.get('ooc_artist', ''))}")
        if stripped(w, "conversation_id"):
            keys.append(f"conv:{stripped(w, 'writer_code')}:{stripped(w, 'conversation_id')}")
        work = norm(w.get("work_name_after_writing", "")) if stripped(w, "work_reference") == "yes" else ""
        if work:
            keys.append(f"work:{work}")
        if not keys:
            keys.append(f"entry:{eid}")
        # 앨범 군집 키: 앨범명 + 발매일(OST·모음집은 가수가 여럿이라 가수를 키에 넣지 않는다)
        albums = sorted({f"{norm(corpus[i]['album'])}|{corpus[i]['release_date']}" for i in ids if i in corpus})
        out.append({
            "entry_id": eid, "writer_code": stripped(w, "writer_code"), "track": stripped(w, "track"),
            "pool": pool, "flags": "|".join(flags), "errors": "|".join(errs),
            "max_sim_existing": f"{best:.3f}", "group_keys": "|".join(sorted(set(keys))),
            "album_keys": ";".join(albums), "work_key": work, "answer_ids": "|".join(ids),
            "multiple_answers": stripped(c, "multiple_answers"), "in_corpus": in_corpus,
            "answer_status": status, "confidence": stripped(c, "confidence"),
            "wrong_clue_types": stripped(c, "wrong_clue_types"), "remembered_aspects": stripped(w, "remembered_aspects"),
            "title_recall": stripped(w, "title_recall"), "artist_recall": stripped(w, "artist_recall"),
            "work_reference": stripped(w, "work_reference"), "conversation_id": stripped(w, "conversation_id"),
            "turn_no": stripped(w, "turn_no"), "turn_origin": stripped(w, "turn_origin"),
            "prior_question": w.get("prior_question", ""), "query": w.get("query", ""),
            "query_original": str(wmeta[eid]["original_query"]),
        })
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if v68.jaccard(new_grams[a["entry_id"]], new_grams[b["entry_id"]]) >= v68.NEAR_DUP:
                for x, y in ((a, b), (b, a)):
                    x["flags"] = "|".join(filter(None, [x["flags"], f"near_dup_in_set:{y['entry_id']}"]))
    info = {"existing_items": ex_n, "existing_song_ids": len(ex_ids), "hash_mismatch": report["hash_mismatch"]}
    return out, info


MERGED_COLS = ("entry_id", "writer_code", "track", "pool", "flags", "errors", "max_sim_existing", "group_keys",
               "album_keys", "work_key", "answer_ids", "multiple_answers", "in_corpus", "answer_status", "confidence",
               "wrong_clue_types", "remembered_aspects", "title_recall", "artist_recall", "work_reference",
               "conversation_id", "turn_no", "turn_origin", "prior_question", "query", "query_original")


def cmd_build(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    rows, info = build(p, corpus_meta(Path(args.corpus)), Path(args.repo))
    write_csv(p.merged, MERGED_COLS, rows)
    print(json.dumps({"entries": len(rows), "pools": dict(sorted(Counter(r["pool"] for r in rows).items())),
                      "existing_items": info["existing_items"], "hash_mismatch": len(info["hash_mismatch"])},
                     ensure_ascii=False))
    return 3 if info["hash_mismatch"] else 0


# ---------- split ----------

class DSU:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def components(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    dsu = DSU()
    for r in rows:
        node = f"row:{r['entry_id']}"
        for k in r["group_keys"].split("|"):
            dsu.union(node, k)
    comps: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        comps[dsu.find(f"row:{r['entry_id']}")].append(r)
    return comps


def assignments(p: Private) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in read_log(p.ledger):
        out[e["entry_id"]] = e["split"]
    return out


def sealed_opened(p: Private) -> bool:
    return any(e["type"] == "sealed_opened" for e in read_log(p.events))


def plan_split(rows: List[Dict[str, str]], assigned: Dict[str, str], salt: str, ratio: float
               ) -> Tuple[Dict[str, Tuple[str, str]], List[List[str]]]:
    """새 항목의 배정과, 이미 배정된 항목끼리 두 분할을 잇게 된 충돌 군집을 돌려준다(충돌이면 아무것도 배정하지 않는다)."""
    eligible = [r for r in rows if r["pool"] in SPLIT_POOLS]
    # 배정된 뒤 대상에서 빠진 항목(확인 대기로 되돌림 등)도 군집에 넣는다 — 그 곡을 새 항목이 다른 분할로 데려가지 못하게
    placed = [r for r in rows if assigned.get(r["entry_id"]) in ("policy_dev", "sealed_eval")]
    conflicts = []
    for comp in components(placed).values():
        if len({assigned[r["entry_id"]] for r in comp}) > 1:
            conflicts.append(sorted(r["entry_id"] for r in comp))
    if conflicts:
        return {}, conflicts
    plan: Dict[str, Tuple[str, str]] = {}
    nodes = {r["entry_id"]: r for r in eligible + placed}
    for comp in components(list(nodes.values())).values():
        new = [r for r in comp if r["entry_id"] not in assigned]
        if not new:
            continue
        have = {assigned[r["entry_id"]] for r in comp if assigned.get(r["entry_id"]) in ("policy_dev", "sealed_eval")}
        key = min(k for r in comp for k in r["group_keys"].split("|"))
        if len(have) > 1:
            for r in new:
                plan[r["entry_id"]] = ("quarantine", "bridges_both_splits")
        elif have:
            split = have.pop()
            for r in new:
                plan[r["entry_id"]] = (split, "inherits_component")
        else:
            split = "sealed_eval" if v68.split_of(key, salt, ratio) == "sealed" else "policy_dev"
            for r in new:
                plan[r["entry_id"]] = (split, "new_component")
    return plan, []


def cmd_split(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    if sealed_opened(p):
        raise SystemExit("봉인 세트를 이미 열었다 — 이 회차에는 더 배정하지 않는다. 새 수집 회차는 새 보관자 폴더·새 salt로")
    salt, ratio = load_salt(p)
    rows, info = build(p, corpus_meta(Path(args.corpus)), Path(args.repo))
    if info["hash_mismatch"]:
        raise SystemExit(f"보존본 해시 불일치 {len(info['hash_mismatch'])}건 — 분할하지 않는다")
    write_csv(p.merged, MERGED_COLS, rows)
    assigned = assignments(p)
    plan, conflicts = plan_split(rows, assigned, salt, ratio)
    if conflicts:
        print(json.dumps({"split": "refused", "conflict_components": len(conflicts),
                          "conflict_entries": sum(len(c) for c in conflicts)}, ensure_ascii=False))
        (p.work / "split_conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False) + "\n", encoding="utf-8")
        print("이미 배정된 항목이 두 분할을 잇는다(수정본 때문) — quarantine으로 보관자가 정한 뒤 다시 돌린다", file=sys.stderr)
        return 3
    batch = 1 + max((int(e.get("batch", 0)) for e in read_log(p.ledger)), default=0)
    at = now_kst()
    for eid in sorted(plan):
        split, why = plan[eid]
        append_log(p.ledger, {"entry_id": eid, "split": split, "why": why, "batch": batch, "assigned_at": at,
                              "salt_sha256": sha256_text(salt)[:16], "sealed_ratio": ratio})
    print(json.dumps({"newly_assigned": len(plan), "batch": batch if plan else None,
                      "by_split": dict(sorted(Counter(v[0] for v in plan.values()).items()))}, ensure_ascii=False))
    return 0


def cmd_quarantine(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    if not args.reason or not args.by:
        raise SystemExit("--reason과 --by(보관자 코드)가 필요하다")
    assigned = assignments(p)
    for eid in args.entry:
        if eid not in assigned:
            raise SystemExit(f"배정 기록이 없는 항목: {eid}")
    for eid in args.entry:
        append_log(p.ledger, {"entry_id": eid, "split": "quarantine", "why": "keeper_decision", "reason": args.reason,
                              "by": args.by, "assigned_at": now_kst(), "previous": assigned[eid]})
    print(json.dumps({"quarantined": len(args.entry)}, ensure_ascii=False))
    return 0


# ---------- export ----------

EXPORT_COLS = ("entry_id", "split", "track", "pool", "query", "conversation_id", "turn_no", "prior_question",
               "turn_origin", "answer_ids", "multiple_answers", "in_corpus", "answer_status", "confidence",
               "wrong_clue_types", "remembered_aspects", "title_recall", "artist_recall", "work_reference", "flags",
               "group_keys", "album_keys", "writer_code")


def cmd_export(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    rows, info = build(p, corpus_meta(Path(args.corpus)), Path(args.repo))
    if info["hash_mismatch"]:
        raise SystemExit("보존본 해시 불일치 — 내보내지 않는다")
    if not chain_ok(p.ledger):
        raise SystemExit("분할 장부 사슬이 깨졌다 — 내보내지 않는다")
    assigned = assignments(p)
    unassigned = [r for r in rows if r["pool"] in SPLIT_POOLS and r["entry_id"] not in assigned]
    if unassigned:
        raise SystemExit(f"분할 대상인데 배정이 없는 항목 {len(unassigned)}건 — split 먼저")
    result = {}
    for split in ("policy_dev", "sealed_eval"):
        part = [{**r, "split": split} for r in rows
                if assigned.get(r["entry_id"]) == split and r["pool"] in SPLIT_POOLS]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(EXPORT_COLS), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(part)
        digest = sha256_text(buf.getvalue())
        prior = [e for e in read_log(p.export_log) if e["split"] == split]
        if prior and prior[-1]["sha256"] == digest:
            result[split] = {"rows": len(part), "version": prior[-1]["version"], "new": False}
            continue
        version = len(prior) + 1
        dest_rel = Path("exports") / f"{split}_r{version:03d}_{digest[:8]}.csv"
        dest = p.root / dest_rel
        if dest.exists():
            raise SystemExit(f"이미 있는 내보내기: {dest_rel}")
        dest.write_text(buf.getvalue(), encoding="utf-8")
        readonly(dest)
        at = now_kst()
        append_log(p.export_log, {"split": split, "version": version, "file": str(dest_rel), "sha256": digest,
                                  "rows": len(part), "exported_at": at,
                                  "owners": sorted({r["entry_id"][:3] for r in part})})
        append_log(p.events, {"type": "data_frozen", "at": at, "split": split, "version": version,
                              "sha256": digest, "rows": len(part)})
        result[split] = {"rows": len(part), "version": version, "new": True}
    print(json.dumps(result, ensure_ascii=False))
    return 0


# ---------- event ----------

EVENT_TYPES = ("criteria_fixed", "model_outputs_released", "sealed_opened", "roles_updated", "note")


def cmd_event(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    events = read_log(p.events)
    entry: Dict[str, object] = {"type": args.type, "at": now_kst(), "by": args.by or "", "note": args.note or ""}
    if not args.by:
        raise SystemExit("--by(기록한 사람 코드)가 필요하다")
    if args.type == "criteria_fixed":
        if not args.file:
            raise SystemExit("채택 기준 문서(--file)가 필요하다 — 해시를 남긴다")
        entry["criteria_sha256"] = sha256_file(Path(args.file))
        entry["configs"] = args.configs or ""
    if args.type == "model_outputs_released":
        if args.scope not in ("policy_dev",):
            raise SystemExit("모델 출력 공개는 policy_dev만 기록한다 — 봉인 세트는 sealed_opened로만 연다")
        entry["scope"] = args.scope
        exports = [e for e in read_log(p.export_log) if e["split"] == "policy_dev"]
        entry["policy_dev_version"] = exports[-1]["version"] if exports else None
    if args.type == "sealed_opened":
        if any(e["type"] == "sealed_opened" for e in events):
            raise SystemExit("봉인 세트는 이미 한 번 열었다 — 다시 열지 않는다(재사용하면 개발용으로 강등)")
        crit = [e for e in events if e["type"] == "criteria_fixed"]
        if not crit:
            raise SystemExit("채택 기준·비교 구성이 고정되지 않았다(criteria_fixed 먼저)")
        exports = [e for e in read_log(p.export_log) if e["split"] == "sealed_eval"]
        if not exports:
            raise SystemExit("봉인 내보내기가 없다(export 먼저)")
        entry["criteria_sha256"] = crit[-1]["criteria_sha256"]
        entry["sealed_version"] = exports[-1]["version"]
        entry["sealed_sha256"] = exports[-1]["sha256"]
    append_log(p.events, entry)
    print(json.dumps({k: v for k, v in entry.items() if k != "note"}, ensure_ascii=False))
    return 0


# ---------- status(익명 집계) ----------

def share(n: int, d: int) -> str:
    return f"{n} ({n / d * 100:.1f}%)" if d else f"{n}"


def status_data(p: Optional[Private], corpus: Optional[Dict[str, Dict[str, str]]], repo: Path) -> Dict[str, object]:
    if p is None:
        return {"has_private": False}
    rows, info = build(p, corpus or {}, repo)
    roles = roles_info(p)
    assigned = assignments(p)
    events = read_log(p.events)
    exports = read_log(p.export_log)
    commitment = json.loads(p.commitment.read_text(encoding="utf-8")) if p.commitment.exists() else {}
    location = json.loads((p.root / "LOCATION.json").read_text(encoding="utf-8"))
    real = [r for r in rows if r["pool"] != "ai_assisted"]
    writers = Counter(r["writer_code"] for r in real)
    songs = Counter(i for r in real for i in r["answer_ids"].split("|") if i)
    albums = Counter(a for r in real for a in r["album_keys"].split(";") if a)
    works = Counter(r["work_key"] for r in real if r["work_key"])
    checked = [r for r in real if r["answer_status"]]
    confirmed = [r for r in real if r["answer_status"] == "confirmed"]
    flags = Counter(f.split(":")[0] for r in real for f in r["flags"].split("|") if f)
    split_counts = Counter((r["track"], assigned.get(r["entry_id"], "unassigned"), r["pool"]) for r in real
                           if r["pool"] in SPLIT_POOLS)
    quarantine = sum(1 for s in assigned.values() if s == "quarantine")
    # 군집 단위 분할이라 봉인 비율이 약속값과 정확히 같지 않다 — 실제 비율과 군집 구성을 함께 적는다
    main = [r for r in real if r["track"] == "general" and r["pool"] == "in_corpus"
            and assigned.get(r["entry_id"]) in ("policy_dev", "sealed_eval")]
    comp_sizes: Dict[str, List[int]] = {"policy_dev": [], "sealed_eval": []}
    placed_all = [r for r in rows if assigned.get(r["entry_id"]) in ("policy_dev", "sealed_eval")]
    for comp in components(placed_all).values():
        comp_sizes[assigned[comp[0]["entry_id"]]].append(len(comp))
    main_split = Counter(assigned[r["entry_id"]] for r in main)
    return {
        "has_private": True,
        "unsafe_location": location.get("unsafe_location", False),
        "rows_received": len(rows),
        "rows_real_memory": len(real),
        "rows_ai_assisted": len(rows) - len(real),
        "writers": len(writers),
        "per_writer_counts_desc": sorted(writers.values(), reverse=True),
        "max_writer_share": (max(writers.values()) / len(real)) if real else 0.0,
        "tracks": dict(sorted(Counter(r["track"] for r in real).items())),
        "pools": dict(sorted(Counter(r["pool"] for r in rows).items())),
        "checked": len(checked),
        "confirmed": len(confirmed),
        "corpus_in_out_among_confirmed": dict(sorted(Counter(r["in_corpus"] for r in confirmed).items())),
        "songs_with_2plus_queries": sum(1 for n in songs.values() if n > 1),
        "max_queries_per_song": max(songs.values(), default=0),
        "max_queries_per_album": max(albums.values(), default=0),
        "max_queries_per_work": max(works.values(), default=0),
        "flags": dict(sorted(flags.items())),
        "split_counts": [{"track": t, "split": s, "pool": pl, "n": n} for (t, s, pl), n in sorted(split_counts.items())],
        "quarantined": quarantine,
        "general_in_corpus_by_split": dict(sorted(main_split.items())),
        "general_in_corpus_sealed_share": (main_split["sealed_eval"] / len(main)) if main else None,
        "components": {k: {"n": len(v), "max_size": max(v, default=0)} for k, v in comp_sizes.items()},
        "withdrawals": sum(1 for e in events if e["type"] == "withdrawn"),
        "pending_deletion_requests": len(pending_deletions(p)),
        "salt_sha256": commitment.get("salt_sha256", ""),
        "sealed_ratio": commitment.get("sealed_ratio"),
        "events": [{k: e[k] for k in ("type", "at") if k in e} | ({"split": e["split"], "version": e["version"],
                                                                   "rows": e["rows"]} if e["type"] == "data_frozen" else {})
                   for e in events if e["type"] in ("salt_committed", "data_frozen", "model_outputs_released",
                                                    "criteria_fixed", "sealed_opened")],
        "exports": [{k: e[k] for k in ("split", "version", "rows", "exported_at")} | {"sha256": e["sha256"][:12]}
                    for e in exports],
        "roles": roles.get("summary", {}),
        "existing_items_compared": info["existing_items"],
        "hash_mismatch": len(info["hash_mismatch"]),
        "chains_ok": all(chain_ok(x) for x in (p.intake_log, p.ledger, p.events, p.export_log)),
    }


def render_status(s: Dict[str, object], generated_at: str) -> str:
    L = ["# v11 독립 평가 자료 — 수집 상태(익명 집계)", "",
         f"생성 {generated_at} KST · `collection_tool.py status`가 만든다. **건수만 담는다** — 질의 원문·정답 곡·제목·작성자 이름 없음.", ""]
    if not s.get("has_private"):
        keeper = s.get("keeper_note") or "미지정"
        L += ["| 단계 | 상태 |", "|---|---|",
              "| 준비(양식·안내·도구) | 완료 — 합성 자료 자체 검사 통과(`SELFTEST.md`) |",
              "| 실제 수집 | **0건** — 보관자 폴더 아직 없음, 받은 표 없음 |",
              "| 정답 확인 | **0건** |",
              "| 분할·봉인 | **안 함** — salt 미생성 |", "",
              "## 역할 지정", "", "| 역할 | 상태 |", "|---|---|",
              "| 질의 작성자 | 미지정 |", "| 정답 확인자 | 미지정 |", f"| 보관자(원본·salt·봉인) | {keeper} |", ""]
        if s.get("plan_notes"):
            L += ["## 계획", ""] + [f"- {n}" for n in s["plan_notes"]] + [""]
        L += [
              "## 한계(수집 전부터 적어 둔다)", "",
              "- 신뢰구간 표(`DATA_SPEC.md` §8)는 질의가 독립이라고 가정한 참고치다. 작성자·곡·작품으로 묶인 질의라 실제 불확실성은 더 크다. 최종 보고는 실제 건수·군집 구성 기준이다.",
              "- 실제 서비스 사용자 질의가 아니다. 팀원·지인이 시험을 위해 쓴 기억 질의이며, 실제 사용자 분포와 다를 수 있다.",
              "- 목표 규모를 채워도 작은 구성 차이는 판별하지 못할 수 있다(`DATA_SPEC.md` §8).",
              "- 집중 진단 갈래(`diag_*`) 성적은 전체 성능에 합치지 않는다.", ""]
        return "\n".join(L)
    real = int(s["rows_real_memory"])
    confirmed = int(s["confirmed"])
    opened = any(e["type"] == "sealed_opened" for e in s["events"])
    released = any(e["type"] == "model_outputs_released" for e in s["events"])
    frozen = [e for e in s["events"] if e["type"] == "data_frozen"]
    L += ["| 단계 | 상태 |", "|---|---|",
          "| 준비 | 완료 |",
          f"| 실제 수집 | 받은 항목 {s['rows_received']}건 · 실제 기억 {real}건 · AI 사용 {s['rows_ai_assisted']}건(주 세트 제외) · 작성자 {s['writers']}명 |",
          f"| 정답 확인 | 확인 기록 {s['checked']}건 · 확정 {confirmed}건 |",
          f"| 분할 | salt 약속 `{str(s['salt_sha256'])[:16] or '없음'}` · 봉인 비율 {s['sealed_ratio']} · 격리 {s['quarantined']}건 |",
          f"| 데이터 고정 | {len(frozen)}회 · 마지막 {frozen[-1]['at'] if frozen else '없음'} |",
          f"| policy_dev 모델 출력 공개 | {'있음 — policy_dev는 개발 자료(독립 최종 평가 아님)' if released else '없음'} |",
          f"| 봉인 열림 | {'열림(재사용 금지)' if opened else '닫힘'} |", ""]
    r = s.get("roles") or {}
    if r:
        L += ["## 역할(코드만)", "",
              f"- 작성자 {r.get('writers_assigned', '?')}명 · 확인자 {r.get('checkers_assigned', '?')}명 · 보관자 {'지정' if r.get('keeper_assigned') else '미지정'}"
              f" · 보관자의 검색 개발 참여 {r.get('keeper_involved_in_search_dev')}",
              f"- 여러 역할을 맡은 사람 {r.get('people_with_multiple_roles', '?')}명 — {r.get('multi_role_note', '')}", ""]
    else:
        L += ["## 역할", "", "- `roles.json` 없음 — 역할 지정 상태를 모른다(지정됨으로 치지 않는다)", ""]
    if s["unsafe_location"]:
        L += ["> **시험 위치**: 이 보관자 폴더는 임시·Git 위치에서 만들어졌다(합성 시험용). 실제 자료 집계가 아니다.", ""]
    L += ["## 묶음(pool)", "", "| 묶음 | 건수 |", "|---|---|"]
    L += [f"| {k} | {share(v, int(s['rows_received']))} |" for k, v in s["pools"].items()]
    L += ["", "## 집중도", "",
          f"- 작성자별 기여(많은 순): {s['per_writer_counts_desc']} · 최대 몫 {float(s['max_writer_share']) * 100:.1f}%"
          " — 기준 20% 이하는 **본 수집 완료 시점의 누적 `general`**에 적용한다(시범 3명 단계에서는 넘는다). 비율을 맞추려고 받은 질의를 버리지 않는다",
          f"- 같은 곡 질의 2건 이상인 곡 {s['songs_with_2plus_queries']}곡 · 곡당 최대 {s['max_queries_per_song']} · 앨범당 최대 {s['max_queries_per_album']} · 작품당 최대 {s['max_queries_per_work']}",
          f"- 갈래: {s['tracks']} — 집중 진단(`diag_*`)은 전체 성능에 합치지 않는다",
          f"- 확정 중 코퍼스 안/밖: {s['corpus_in_out_among_confirmed']}",
          f"- 표시: {s['flags']}", "",
          "## 분할별 건수", "", "| 갈래 | 분할 | 묶음 | 건수 |", "|---|---|---|---|"]
    L += [f"| {x['track']} | {x['split']} | {x['pool']} | {x['n']} |" for x in s["split_counts"]]
    share_txt = (f"{float(s['general_in_corpus_sealed_share']) * 100:.1f}%"
                 if s["general_in_corpus_sealed_share"] is not None else "없음")
    comps = s["components"]
    L += ["", f"- `general` 코퍼스 안: {s['general_in_corpus_by_split']} · 실제 봉인 비율 {share_txt}"
          f"(약속 {s['sealed_ratio']} — 군집 단위 배정이라 정확히 맞지 않는다)",
          f"- 군집 수 policy_dev {comps['policy_dev']['n']}(최대 {comps['policy_dev']['max_size']}건) · "
          f"sealed_eval {comps['sealed_eval']['n']}(최대 {comps['sealed_eval']['max_size']}건)",
          f"- 참여 철회 {s['withdrawals']}건 · 밖으로 보낸 사본 삭제 확인 대기 {s['pending_deletion_requests']}건"]
    L += ["", f"보존본 해시 불일치 {s['hash_mismatch']}건 · 장부 사슬 {'정상' if s['chains_ok'] else '깨짐'} · 비교한 기존 질의 {s['existing_items_compared']}건", "",
          "실제 서비스 사용자 질의가 아니다(팀원·지인 기억 자료). 작은 구성 차이는 판별하지 못할 수 있다.", ""]
    return "\n".join(L)


def cmd_status(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require() if args.private_dir else None
    corpus = corpus_meta(Path(args.corpus)) if p else None
    s = status_data(p, corpus, Path(args.repo))
    if p is None:
        s["keeper_note"] = args.keeper_note or ""
        s["plan_notes"] = args.plan_note or []
    out = Path(args.out)
    out.write_text(render_status(s, now_kst()), encoding="utf-8")
    Path(args.json_out or out.with_suffix(".json")).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n",
                                                               encoding="utf-8")
    print(json.dumps({"status_written": out.name, "rows_received": s.get("rows_received", 0)}, ensure_ascii=False))
    return 0


# ---------- 참여 철회 ----------

def cmd_withdraw(args: argparse.Namespace) -> int:
    """참여 철회: 그 사람이 낸 표·그 사람 항목을 담은 확인자 표·내보내기를 지우고 지운 사실만 기록한다(분할 장부의 항목 번호는 남긴다)."""
    p = Private(Path(args.private_dir)).require()
    if not (PERSON_RE.match(args.person) and args.by and args.reason):
        raise SystemExit("--person P01 · --by · --reason이 필요하다")
    gone = withdrawn_files(p)
    files = []
    for e in read_log(p.intake_log):
        if e["stored"] in gone:
            continue
        mine = (e["kind"] in ("writer", "declaration") and e["code"] == args.person) \
            or (e["kind"] == "checker" and e.get("owner") == args.person) \
            or (e["kind"] == "checker" and e["code"] == args.person and args.include_checker_work)
        if mine:
            files.append(e["stored"])
    touched = [c for c in copies(p) if args.person in c["owners"]]
    exports = [c["path"] for c in touched if c["kind"] == "export" and c["path"] not in gone
               and (p.root / c["path"]).exists()]
    local_copies = [c["path"] for c in touched if c["kind"] != "export" and (p.root / c["path"]).exists()]
    # 파생 작업 파일은 원문·항목 번호를 담는다 — 통째로 지우고 다음 build·split이 철회자 없이 다시 만든다
    derived = [str(f.relative_to(p.root)) for f in sorted(p.work.glob("*")) if f.is_file()]
    for rel in files + exports + local_copies + derived:
        path = p.root / rel
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        path.unlink()
    touched_paths = {c["path"] for c in touched}
    requests = []
    n_prior = sum(len(e.get("deletion_requests", [])) for e in read_log(p.events) if e["type"] == "withdrawn")
    for e in read_log(p.events):
        if e["type"] == "copy_delivered" and e["path"] in touched_paths:
            n_prior += 1
            requests.append({"id": f"DR-{n_prior:03d}", "path": e["path"], "to": e["to"]})
    append_log(p.events, {"type": "withdrawn", "at": now_kst(), "person": args.person, "by": args.by,
                          "reason": args.reason, "files_deleted": files, "exports_deleted": exports,
                          "copies_deleted": local_copies, "derived_deleted": derived,
                          "deletion_requests": requests, "sealed_already_opened": sealed_opened(p)})
    print(json.dumps({"files_deleted": len(files), "exports_deleted": len(exports), "copies_deleted": len(local_copies),
                      "derived_deleted": len(derived), "deletion_requests": len(requests)}, ensure_ascii=False))
    if requests:
        print("밖으로 보낸 사본의 삭제를 받은 사람에게 요청하고, 확인되면 confirm-deletion으로 기록한다", file=sys.stderr)
    return 0


# ---------- verify ----------

def cmd_verify(args: argparse.Namespace) -> int:
    p = Private(Path(args.private_dir)).require()
    problems: List[str] = []
    gone = withdrawn_files(p)
    for e in read_log(p.intake_log):
        if e["stored"] in gone:
            continue
        path = p.root / e["stored"]
        if not path.exists() or sha256_file(path) != e["sha256"]:
            problems.append(f"received_hash:{e['stored']}")
    for e in read_log(p.export_log):
        if e["file"] in gone:
            continue
        path = p.root / e["file"]
        if not path.exists() or sha256_file(path) != e["sha256"]:
            problems.append(f"export_hash:{e['file']}")
    for log in (p.intake_log, p.ledger, p.events, p.export_log):
        if not chain_ok(log):
            problems.append(f"chain:{log.name}")
    if p.commitment.exists() and p.salt.exists():
        c = json.loads(p.commitment.read_text(encoding="utf-8"))
        if sha256_text(p.salt.read_text(encoding="utf-8").strip()) != c["salt_sha256"]:
            problems.append("salt_commitment")
    print(json.dumps({"ok": not problems, "problems": problems}, ensure_ascii=False))
    return 3 if problems else 0


# ---------- CLI ----------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, func, private: bool = True, corpus: bool = False, repo: bool = False):
        sp = sub.add_parser(name)
        if private:
            sp.add_argument("--private-dir", required=True)
        if corpus:
            sp.add_argument("--corpus", default=str(ROOT / "data/all_songs.jsonl"))
        if repo:
            sp.add_argument("--repo", default=str(ROOT), help="기존 세트를 읽을 저장소 루트(누출·중복 검사)")
        sp.set_defaults(func=func)
        return sp

    sp = add("init", cmd_init)
    sp.add_argument("--allow-unsafe-location", action="store_true", help="합성 자기 검사 전용")
    sp = add("commit-salt", cmd_commit_salt)
    sp.add_argument("--import-file", help="보관자가 따로 만든 salt 파일(없으면 새로 만든다)")
    sp.add_argument("--sealed-ratio", type=float, default=0.6)
    sp = add("intake", cmd_intake)
    sp.add_argument("--kind", choices=sorted(KEY_COL), required=True)
    sp.add_argument("--code", required=True, help="보낸 사람 코드(P01). 확인자 표면 확인자 코드")
    sp.add_argument("--file", required=True)
    sp.add_argument("--reason", help="수정본일 때만: 고친 이유")
    sp.add_argument("--editor", help="수정본일 때만: 고친 사람 코드")
    add("check", cmd_check, corpus=True)
    sp = add("make-writer-packet", cmd_make_writer_packet)
    sp.add_argument("--code", required=True)
    sp.add_argument("--rows", type=int, default=20)
    sp.add_argument("--track", default="general")
    sp.add_argument("--out-dir", required=True)
    sp = add("make-checker-sheet", cmd_make_checker_sheet)
    sp.add_argument("--writer", required=True, help="한 작성자(P01)의 항목만 담는다")
    sp.add_argument("--out", required=True)
    sp.add_argument("--include-checked", action="store_true")
    sp = add("export-corpus-meta", cmd_export_corpus_meta, private=False, corpus=True)
    sp.add_argument("--out", required=True)
    add("build", cmd_build, corpus=True, repo=True)
    add("split", cmd_split, corpus=True, repo=True)
    sp = add("quarantine", cmd_quarantine)
    sp.add_argument("--entry", nargs="+", required=True)
    sp.add_argument("--reason")
    sp.add_argument("--by")
    add("export", cmd_export, corpus=True, repo=True)
    sp = add("event", cmd_event)
    sp.add_argument("--type", choices=EVENT_TYPES, required=True)
    sp.add_argument("--by")
    sp.add_argument("--note")
    sp.add_argument("--file")
    sp.add_argument("--configs")
    sp.add_argument("--scope")
    sp = sub.add_parser("status")
    sp.add_argument("--private-dir")
    sp.add_argument("--corpus", default=str(ROOT / "data/all_songs.jsonl"))
    sp.add_argument("--repo", default=str(ROOT))
    sp.add_argument("--out", required=True)
    sp.add_argument("--json-out")
    sp.add_argument("--keeper-note", help="보관자 폴더가 아직 없을 때: 보관자 지정 상태(이름은 쓰지 않는다)")
    sp.add_argument("--plan-note", action="append", help="보관자 폴더가 아직 없을 때: 계획 한 줄(여러 번)")
    sp.set_defaults(func=cmd_status)
    sp = add("deliver", cmd_deliver)
    sp.add_argument("--file", required=True, help="보관자 폴더 기준 경로(packets/… 또는 exports/…)")
    sp.add_argument("--to")
    sp.add_argument("--by")
    sp = add("confirm-deletion", cmd_confirm_deletion)
    sp.add_argument("--request", required=True)
    sp.add_argument("--by")
    sp.add_argument("--note")
    sp = add("withdraw", cmd_withdraw)
    sp.add_argument("--person", required=True)
    sp.add_argument("--by")
    sp.add_argument("--reason")
    sp.add_argument("--include-checker-work", action="store_true", help="그 사람이 확인자로 남긴 판단도 지운다")
    add("verify", cmd_verify)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
