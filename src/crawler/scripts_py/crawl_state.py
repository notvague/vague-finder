"""
[Pipeline State] 곡 단위 수집 단계 상태

설명: 한 곡의 수집을 네 단계로 나누고, 어느 단계까지 끝났는지 폴더 안의
      crawl_status.json에 남긴다. 실패한 단계만 다시 실행할 수 있다.

      source   멜론 메타·가사·댓글, 유튜브 영상·댓글, 나무위키 맥락을 모아 source.json에 저장
      assets   cover.jpg, audio.m4a 다운로드와 내용 검증
      analysis Gemini 정제 -> meta.json
      export   all_songs.jsonl에 추가하고 폴더를 완료 폴더(data/raw)로 이동

배경:
  - 예전에는 LLM 정제까지 끝난 뒤 오디오 다운로드가 실패하면 폴더를 통째로 지웠다.
    멜론·유튜브·Gemini 호출을 전부 다시 해야 했다.
  - JSONL 추가가 실패해도 '수집 완료'로 처리돼, 재실행에서는 건너뛰는데 JSONL에는
    곡이 없는 상태가 생겼다.
  - Gemini가 세 번 모두 실패하면 '분석실패' 값을 정상 레코드처럼 저장했다.

  진행 중인 폴더는 완료 폴더와 **다른 디렉터리**(기본 `<DATA_DIR>_incomplete`)에 둔다.
  임베딩 단계(data_songs.load_songs_from_data_dir)는 meta.json이 없는 폴더를 검증 실패로
  보고 META_MOVE_INVALID=1이면 failed_raw로 옮겨 버리므로, 반쪽 폴더가 data/raw에
  있으면 재개할 수 없게 된다.

  all_songs.jsonl은 meta.json에서 파생되는 파일이다. 시작할 때 완료 폴더 중 JSONL에
  없는 곡을 덧붙이고(sync_jsonl), 필요하면 통째로 다시 만든다(rebuild_jsonl).

작성자: 황찬혁 (Full)
생성일: 2026-09-18
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STAGES: Tuple[str, ...] = ("source", "assets", "analysis", "export")
STATE_FILE = "crawl_status.json"
SOURCE_FILE = "source.json"
META_FILE = "meta.json"
COVER_FILE = "cover.jpg"
AUDIO_FILE = "audio.m4a"

# 재수집 대상 단계(RECOLLECT_STAGES)가 이 횟수만큼 실패하면 재개를 포기하고 처음부터
# 다시 수집한다. 영상이 삭제됐거나 원천 데이터 자체가 잘못된 경우 무한히 같은 단계에서
# 멈추지 않게 한다.
MAX_STAGE_ATTEMPTS = 3

# 다시 수집해야 고칠 수 있는 단계. export는 여기 없다.
#
# export가 실패하는 이유는 JSONL 쓰기 실패와 폴더 이동 실패다. 모아 둔 데이터가 잘못된 게
# 아니라 디스크·마운트 문제라서, 다시 수집해도 마지막에 똑같이 실패한다. 그런데도 폴더를
# 버리면 검증을 통과한 meta.json과 오디오·커버가 한꺼번에 사라지고, 재수집이 실행마다
# 반복돼 Gemini 호출 스무 번이 매번 다시 나간다. export는 폐기하지 않고 계속 재시도한다 —
# 드라이브가 돌아오면 이동 한 번으로 끝난다.
RECOLLECT_STAGES: Tuple[str, ...] = ("source", "assets", "analysis")

# 규칙·프롬프트가 바뀌면 올린다. 레코드에 남아 어느 규칙으로 뽑혔는지 비교할 수 있다.
PIPELINE_VERSION = "2026.09.18"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class CrawlState:
    """폴더 하나의 진행 상태. 파일과 1:1이다."""

    song_id: str
    input_artist: str
    input_title: str
    folder: Path
    stages: Dict[str, Dict] = field(default_factory=dict)
    selection: Dict = field(default_factory=dict)
    comment_counts: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # --- 파일 입출력 ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.folder / STATE_FILE

    def save(self) -> None:
        self.updated_at = _now()
        self.folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "song_id": self.song_id,
            "input_artist": self.input_artist,
            "input_title": self.input_title,
            "stages": self.stages,
            "selection": self.selection,
            "comment_counts": self.comment_counts,
            "warnings": self.warnings,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @classmethod
    def load(cls, folder: Path) -> Optional["CrawlState"]:
        path = folder / STATE_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("상태 파일을 읽지 못함: %s (%s)", path, e)
            return None
        if not isinstance(data, dict) or not data.get("song_id"):
            return None
        return cls(
            song_id=str(data["song_id"]),
            input_artist=str(data.get("input_artist", "")),
            input_title=str(data.get("input_title", "")),
            folder=folder,
            stages=dict(data.get("stages") or {}),
            selection=dict(data.get("selection") or {}),
            comment_counts=dict(data.get("comment_counts") or {}),
            warnings=list(data.get("warnings") or []),
            pipeline_version=str(data.get("pipeline_version") or ""),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
        )

    # --- 단계 진행 -------------------------------------------------------------

    def done(self, stage: str) -> bool:
        return (self.stages.get(stage) or {}).get("status") == "done"

    def attempts(self, stage: str) -> int:
        return int((self.stages.get(stage) or {}).get("attempts", 0))

    def mark_done(self, stage: str) -> None:
        entry = self.stages.setdefault(stage, {})
        entry.update({"status": "done", "at": _now(), "error": ""})
        self.save()

    def mark_failed(self, stage: str, error: str) -> None:
        entry = self.stages.setdefault(stage, {})
        entry.update({
            "status": "failed",
            "at": _now(),
            "error": str(error)[:300],
            "attempts": self.attempts(stage) + 1,
        })
        self.save()

    def next_stage(self) -> Optional[str]:
        """아직 끝나지 않은 첫 단계. 모두 끝났으면 None."""
        for stage in STAGES:
            if not self.done(stage):
                return stage
        return None

    @property
    def is_complete(self) -> bool:
        return self.next_stage() is None

    def exhausted(self) -> Optional[str]:
        """재시도 한도에 닿은 단계가 있으면 그 이름. 사람에게 알리는 용도로도 쓴다.

        폐기 여부는 이것으로 판단하지 않는다 — needs_recollect를 쓴다.
        """
        stage = self.next_stage()
        if stage and self.attempts(stage) >= MAX_STAGE_ATTEMPTS:
            return stage
        return None

    def needs_recollect(self) -> Optional[str]:
        """폴더를 버리고 처음부터 다시 수집해야 하는 단계가 있으면 그 이름.

        export는 아무리 실패해도 여기 걸리지 않는다(RECOLLECT_STAGES 주석 참고).
        """
        stage = self.next_stage()
        if stage in RECOLLECT_STAGES and self.attempts(stage) >= MAX_STAGE_ATTEMPTS:
            return stage
        return None

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)

    # --- meta.json에 들어갈 블록 -------------------------------------------------

    def to_meta_block(self) -> Dict:
        """meta.json의 crawl_status.

        임베딩 전 검증(meta_validation)은 문서 전체를 재귀적으로 훑어 null·빈 문자열·빈 배열·
        빈 객체가 하나라도 있으면 곡을 뺀다. 그래서 값이 없는 키는 아예 넣지 않는다(_compact).
        검증기는 또 crawl_status.warnings에 항목이 있으면 곡을 뺀다. 검토 메모(원곡 아님
        가능성, 가수 미확인, 댓글 없음)는 곡을 색인에서 빼자는 뜻이 아니라 사람이 볼 목록이므로
        warnings는 비워 두고 notes에 적는다. 검토 목록은 review_songs.csv에도 남는다.

        status는 analysis 단계에서 쓰므로 'complete'로 고정한다. export가 실패하면
        폴더가 완료 폴더로 옮겨지지 않아 임베딩 단계가 이 파일을 보지 못한다.
        """
        block = {
            "status": "complete",
            "warnings": [],
            "input_artist": self.input_artist,
            "input_title": self.input_title,
            "crawled_at": _now(),
            "pipeline_version": self.pipeline_version,
        }
        selection = _compact(self.selection)
        if selection:
            block["selection"] = selection
        if self.comment_counts:
            # 0은 지우지 않는다. '댓글 0개'는 정보다 — 검증기의 '근거 없는 반응 요약' 규칙과
            # 사람이 보는 검토 목록이 이 숫자를 쓴다.
            block["comment_counts"] = dict(self.comment_counts)
        if self.warnings:
            block["notes"] = list(self.warnings)
        return block


def _compact(value):
    """None, 빈 문자열, 빈 배열, 빈 객체를 재귀적으로 제거한다."""
    if isinstance(value, dict):
        cleaned = {k: _compact(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        cleaned = [_compact(v) for v in value]
        return [v for v in cleaned if v not in (None, "", [], {})]
    return value


# --- 폴더 스캔 ------------------------------------------------------------------------

def song_id_of(folder: Path) -> str:
    """폴더명 Artist_Title_12345의 끝 숫자."""
    suffix = folder.name.rsplit("_", 1)[-1]
    return suffix if suffix.isdigit() else ""


def has_complete_files(folder: Path) -> bool:
    """meta.json + cover.jpg + audio.m4a가 모두 있고 비어 있지 않은가.

    상태 파일이 생기기 전에 수집한 폴더도 이 기준으로 완료로 본다.
    """
    for name in (META_FILE, COVER_FILE, AUDIO_FILE):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def scan_complete(data_dir: Path) -> Dict[str, Path]:
    """완료 폴더의 {song_id: 폴더}. 파일이 빠진 폴더는 경고만 남기고 제외한다."""
    found: Dict[str, Path] = {}
    incomplete = 0
    if not data_dir.exists():
        return found
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        song_id = song_id_of(folder)
        if not song_id:
            continue
        if not has_complete_files(folder):
            incomplete += 1
            continue
        found[song_id] = folder
    if incomplete:
        logger.warning(
            "완료 폴더 안에 파일이 빠진 폴더 %d개 (meta.json + cover.jpg + audio.m4a 조건 미충족). "
            "재실행에서 처음부터 다시 수집한다.",
            incomplete,
        )
    return found


def scan_incomplete(staging_dir: Path) -> List[CrawlState]:
    """격리 폴더의 재개 대상. 상태 파일이 없는 폴더는 재개할 근거가 없어 건너뛴다."""
    states: List[CrawlState] = []
    if not staging_dir.exists():
        return states
    for folder in sorted(staging_dir.iterdir()):
        if not folder.is_dir():
            continue
        state = CrawlState.load(folder)
        if state is None:
            logger.warning("상태 파일이 없는 미완료 폴더(재개 불가): %s", folder.name)
            continue
        states.append(state)
    return states


# --- source.json ------------------------------------------------------------------------

def write_source(folder: Path, source: Dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (SOURCE_FILE + ".tmp")
    tmp.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(folder / SOURCE_FILE)


def read_source(folder: Path) -> Optional[Dict]:
    path = folder / SOURCE_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("source.json을 읽지 못함: %s (%s)", path, e)
        return None
    return data if isinstance(data, dict) else None


# --- JSONL (meta.json에서 파생) -----------------------------------------------------------

def iter_jsonl(jsonl_path: Path) -> Iterator[Dict]:
    if not jsonl_path.is_file():
        return
    # json.dumps(ensure_ascii=False)는 U+2028 등을 이스케이프하지 않으므로 '\n'으로만 나눈다.
    with jsonl_path.open(encoding="utf-8", newline="") as f:
        for line in f.read().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning("JSONL에 깨진 줄이 있음(건너뜀): %s...", line[:60])
                continue
            if isinstance(record, dict):
                yield record


def seed_index(jsonl_path: Path, only_ids: Optional[Set[str]] = None) -> Dict[Tuple[str, str], str]:
    """(가수, 제목) -> 이미 수집한 song_id. 완료곡을 **검색 전에** 건너뛰는 데 쓴다.

    예전에는 완료곡도 멜론 검색을 한 번 하고 대기까지 한 뒤에야 건너뛰었다. 961곡을 다시
    순회하면 대기 시간만 32~80분이다.

    열쇠는 두 곳에서 만든다.

    1. crawl_status.input_artist/input_title — 수집 당시 **실제로 요청한** 시드다. 가장 정확하다.
    2. 레코드 자신의 멜론 가수·제목 — 1번이 없는 예전 수집분(단계별 재시작 도입 전에 모은 곡은
       crawl_status가 아예 없다)을 위한 폴백이다. 시드 목록 973개 중 902개가 이 열쇠로 맞는다.

    1번이 있으면 1번을 쓴다. 폴백 열쇠가 서로 다른 song_id 두 개를 가리키면 **그 열쇠는 버린다** —
    어느 곡인지 모르는 상태에서 건너뛰면 수집해야 할 곡을 빠뜨린다. 정규화 뒤 가수나 제목이 비는
    열쇠도 버린다(기호만으로 된 제목이 서로 같은 열쇠가 되는 것을 막는다).
    """
    from_status: Dict[Tuple[str, str], str] = {}
    from_record: Dict[Tuple[str, str], str] = {}
    ambiguous: Set[Tuple[str, str]] = set()

    for record in iter_jsonl(jsonl_path):
        song_id = str(record.get("song_id") or record.get("id") or "")
        if not song_id or (only_ids is not None and song_id not in only_ids):
            continue

        status = record.get("crawl_status") or {}
        artist, title = status.get("input_artist"), status.get("input_title")
        if isinstance(artist, str) and isinstance(title, str):
            key = (_seed_key(artist), _seed_key(title))
            if all(key):
                from_status[key] = song_id

        for key in _record_seed_keys(record):
            known = from_record.get(key)
            if known is not None and known != song_id:
                ambiguous.add(key)
            else:
                from_record[key] = song_id

    for key in ambiguous:
        from_record.pop(key, None)

    index = {key: song_id for key, song_id in from_record.items() if key not in from_status}
    index.update(from_status)
    return index


def _record_seed_keys(record: Dict) -> List[Tuple[str, str]]:
    """레코드 자신의 멜론 가수·제목으로 만든 열쇠. 가수가 여럿이면 각각 만든다."""
    meta = record.get("metadata") or {}
    title = meta.get("title") or record.get("title") or ""
    artists = meta.get("artist") or record.get("artist") or []
    if isinstance(artists, str):
        artists = [artists]
    keys = []
    for artist in artists:
        if not isinstance(artist, str):
            continue
        key = (_seed_key(artist), _seed_key(title))
        if all(key):
            keys.append(key)
    return keys


def _seed_key(text: str) -> str:
    """시드 대조용 정규화. 공백과 기호를 지우고 소문자로 만든다.

    글자·숫자는 어느 문자 체계든 남긴다. 한글·영숫자만 남기면 일본어·중국어 제목이 통째로
    빈 문자열이 되어 서로 다른 곡이 같은 열쇠를 갖는다(수집 목록에 일본 곡이 있다).
    """
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def read_jsonl_ids(jsonl_path: Path) -> Set[str]:
    return {str(r.get("song_id") or r.get("id") or "") for r in iter_jsonl(jsonl_path)} - {""}


def append_jsonl(jsonl_path: Path, record: Dict) -> None:
    """한 줄 추가. 실패는 예외로 올린다 — 삼키면 '수집 완료인데 JSONL에는 없는' 곡이 생긴다."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def replace_jsonl_record(jsonl_path: Path, record: Dict) -> bool:
    """같은 song_id의 줄을 새 레코드 한 줄로 바꾼다. 없으면 덧붙인다. 바꿨으면 True.

    재수집한 곡은 meta.json이 새 분석인데 JSONL에는 예전 분석이 남는다. 같은 ID가 이미 있다는
    이유로 기록을 건너뛰면 두 파일이 어긋난 채로 굳는다.

    파일 전체를 다시 쓰므로 호출부는 **ID가 이미 있을 때만** 쓴다(재수집·재시도). 새 곡은
    append_jsonl로 덧붙인다. 바꾸지 않는 줄은 읽은 그대로 옮겨 다른 곡의 표기가 변하지 않는다.
    """
    song_id = str(record.get("song_id") or record.get("id") or "")
    line = json.dumps(record, ensure_ascii=False)
    existing = jsonl_path.read_text(encoding="utf-8") if jsonl_path.is_file() else ""
    # json.dumps(ensure_ascii=False)는 U+2028 등을 이스케이프하지 않으므로 '\n'으로만 나눈다.
    lines = [raw for raw in existing.split("\n") if raw.strip()]

    # 같은 ID의 줄이 여럿이면 **전부** 걷어내고 한 줄로 합친다. 첫 줄만 바꾸면 뒤에 남은
    # 예전 줄 때문에, 마지막 줄을 쓰는 감사 도구(audit_collection.load_records)가 갱신 뒤에도
    # 옛 데이터를 읽는다.
    kept: List[str] = []
    replaced = False
    for raw in lines:
        try:
            current = json.loads(raw)
        except ValueError:
            kept.append(raw)          # 읽지 못하는 줄은 손대지 않는다
            continue
        is_same_song = (
            isinstance(current, dict)
            and str(current.get("song_id") or current.get("id") or "") == song_id
        )
        if not is_same_song:
            kept.append(raw)
            continue
        if not replaced:
            kept.append(line)         # 첫 자리를 새 레코드가 차지한다
            replaced = True

    lines = kept
    if not replaced:
        lines.append(line)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(jsonl_path)
    return replaced


def remove_jsonl_record(jsonl_path: Path, song_id: str) -> int:
    """같은 song_id의 줄을 **모두** 지운다. 지운 줄 수를 돌려준다.

    수집 정책에서 제외된 곡의 폴더만 지우면 JSONL 행이 남아, 그 파일을 읽는 감사·평가·카탈로그
    작업에 제외 대상이 계속 전달된다. 시작 시 동기화는 이 행을 고아로만 집계하고 지우지 않는다.

    읽지 못하는 줄은 손대지 않는다. 쓰기는 임시 파일로 하고 원자적으로 교체한다.
    """
    song_id = str(song_id)
    if not jsonl_path.is_file():
        return 0

    existing = jsonl_path.read_text(encoding="utf-8")
    # json.dumps(ensure_ascii=False)는 U+2028 등을 이스케이프하지 않으므로 '\n'으로만 나눈다.
    kept: List[str] = []
    removed = 0
    for raw in existing.split("\n"):
        if not raw.strip():
            continue
        try:
            current = json.loads(raw)
        except ValueError:
            kept.append(raw)
            continue
        same = (
            isinstance(current, dict)
            and str(current.get("song_id") or current.get("id") or "") == song_id
        )
        if same:
            removed += 1
        else:
            kept.append(raw)

    if not removed:
        return 0

    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    tmp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    tmp.replace(jsonl_path)
    return removed


def load_meta(folder: Path) -> Optional[Dict]:
    path = folder / META_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("meta.json을 읽지 못함: %s (%s)", path, e)
        return None
    return data if isinstance(data, dict) else None


def sync_jsonl(data_dir: Path, jsonl_path: Path,
               skip_record=None,
               complete: Optional[Dict[str, Path]] = None,
               jsonl_ids: Optional[Set[str]] = None) -> Tuple[int, int]:
    """완료 폴더 중 JSONL에 없는 곡을 덧붙인다.

    Args:
        complete: 이미 읽어 둔 완료 폴더 목록(scan_complete 결과). 없으면 직접 읽는다.
        jsonl_ids: 이미 읽어 둔 JSONL의 song_id 집합. 주면 **이 집합을 갱신한다**.
        skip_record: 레코드를 받아 제외 사유 문자열(제외 안 하면 "")을 돌려주는 함수.
            수집 정책에서 제외된 곡을 다시 넣지 않게 한다. 정책 이전에 모은 완료 폴더가
            남아 있으면, 이 검사가 없으면 여기서 JSONL에 되살아난다.

    Returns:
        (덧붙인 곡 수, JSONL에는 있는데 완료 폴더가 없는 곡 수)

    두 번째 값은 지우지 않고 알리기만 한다. 폴더가 다른 컴퓨터(드라이브)에 있거나
    잠시 마운트가 안 된 경우에 JSONL을 깎아 먹으면 안 된다. 통째로 다시 만들려면
    rebuild_jsonl을 쓴다.
    """
    # 폴더 목록과 JSONL은 호출부가 이미 읽었으면 그것을 쓴다. Registry가 곧바로 같은 일을
    # 반복하던 것을 없애려는 것이다. 여기서 덧붙인 ID는 넘겨받은 집합에도 반영된다.
    complete = scan_complete(data_dir) if complete is None else complete
    existing = read_jsonl_ids(jsonl_path) if jsonl_ids is None else jsonl_ids
    added = 0
    for song_id, folder in complete.items():
        if song_id in existing:
            continue
        meta = load_meta(folder)
        if meta is None:
            continue
        reason = skip_record(meta) if skip_record else ""
        if reason:
            logger.warning("JSONL 누락 복구 건너뜀(수집 정책 제외): %s (%s)", folder.name, reason)
            continue
        append_jsonl(jsonl_path, meta)
        existing.add(song_id)
        added += 1
        logger.info("JSONL 누락 복구: %s", folder.name)
    orphans = len(existing - set(complete))
    return added, orphans


class RebuildRefused(Exception):
    """재생성이 기존 레코드를 대량으로 지울 상황이라 멈췄다."""


def rebuild_jsonl(data_dir: Path, jsonl_path: Path, force: bool = False) -> int:
    """완료 폴더의 meta.json만으로 JSONL을 다시 만든다. 기존 파일은 시각을 붙여 사본으로 남긴다.

    수집 폴더는 보통 구글 드라이브에 있다. 마운트가 안 된 상태에서 부르면 완료 폴더가 0개로
    보이고, 그대로 쓰면 961곡짜리 파일이 0바이트가 된다. 사본을 한 자리에만 남기면 두 번째
    호출이 그 사본마저 덮어써 되돌릴 수 없다. 그래서 줄어드는 폭이 크면 거절한다.

    Raises:
        RebuildRefused: 완료 폴더에서 레코드를 못 찾았거나 절반 넘게 줄어드는데 force가 아닐 때.
    """
    complete = scan_complete(data_dir)
    records = []
    for song_id, folder in complete.items():
        meta = load_meta(folder)
        if meta is not None:
            records.append(meta)

    existing = len(read_jsonl_ids(jsonl_path))
    if existing and not force:
        if not records:
            raise RebuildRefused(
                f"완료 폴더에서 레코드를 하나도 찾지 못했다({data_dir}). 기존 {existing}곡을 지우지 않는다. "
                f"수집 폴더 경로(VAGUEFINDER_DATA_DIR)와 마운트를 확인하고, 정말 비울 생각이면 --force를 준다."
            )
        if len(records) * 2 < existing:
            raise RebuildRefused(
                f"재생성 결과가 {len(records)}곡으로 기존 {existing}곡의 절반 미만이다. "
                f"수집 폴더를 확인하고, 의도한 것이면 --force를 준다."
            )

    if jsonl_path.is_file():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(jsonl_path, jsonl_path.with_suffix(jsonl_path.suffix + f".{stamp}.bak"))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)
    return len(records)
