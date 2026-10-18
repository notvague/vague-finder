"""코퍼스의 작품 문맥 문장 저장소(`work_mentions_v1`) — 추정 작품명이 우리 자료에 나오는지 확인하는 용도.

작품명을 **추출하지 않는다.** 전체 코퍼스에 같은 규칙으로 "작품 종류·사용 표지가 있는 문장"만 곡별로 모아 두고,
조회 때 주어진 이름이 그 문장에 낱말 경계로 나오는지(`media_name_in_text`)만 본다. 이름 사전이 없으므로 평가 질의에
나오는 작품만 골라 넣을 수 없다.

- 자료원: `data/all_songs.jsonl`의 metadata `album`(앨범명은 표지 없이도 그대로 넣는다 — OST 앨범명이 작품명을 품는다)·
  `album_summary`·`album_description`, Context 산출물(`artifacts/context/songs/*.json`) records `evidence_text`.
  제목·가수 필드는 넣지 않는다 — 작품명과 곡 제목이 같은 경우가 제목 일치로 부풀지 않게.
- 곡–작품 관계는 "작품 문맥 문장에 이름이 언급됨"(`mentioned`)까지다. OST·삽입곡 확정 연결이 아니다.
- 저장소 파일 sha256과 자료원 sha256·규칙 버전을 manifest에 남긴다. 불러올 때 sha가 다르면 거부한다.

    venv/bin/python -m src.retrieval.work_mentions build --songs data/all_songs.jsonl \
        --context-dir artifacts/context/songs --out artifacts/work_names/work_mentions_v1.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.retrieval.context_media_match import media_name_in_text

SCHEMA_VERSION = "work_mentions_v1"
RULE_VERSION = "marker_sentence_v1"
SOURCES = ("album", "album_summary", "album_description", "context_fact")

# 작품 종류·사용 표지. 이 표지가 있는 문장만 저장한다(앨범명은 예외). 방송사·서비스 이름은 작품 문맥의 표지다.
MEDIA_MARKER = re.compile(
    r"OST|O\.S\.T|오에스티|사운드\s*트랙|삽입|주제가|테마곡|배경\s*음악|BGM|"
    r"드라마|영화|예능|프로그램|방송|경연|오디션|서바이벌|애니|만화|웹툰|시트콤|뮤지컬|다큐|시즌|극중|출연|"
    r"tvN|KBS|MBC|SBS|JTBC|Mnet|엠넷|넷플릭스|Netflix|티빙|웨이브|디즈니",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
MAX_SENTENCE_CHARS = 400


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def compact(text: str) -> str:
    return re.sub(r"[\s_]+", "", normalize(text))


def _sentences(text: str) -> List[str]:
    out = []
    for part in _SENTENCE_SPLIT.split(str(text or "")):
        part = " ".join(part.split())
        if part:
            out.append(part[:MAX_SENTENCE_CHARS])
    return out


def song_mention_sentences(song: Mapping, context_facts: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """곡 하나의 (자료원, 정규화 문장) 목록. 규칙은 모든 곡에 같다."""
    md = song.get("metadata") or {}
    rows: List[Tuple[str, str]] = []
    album = " ".join(str(md.get("album") or "").split())
    if album:
        rows.append(("album", normalize(album)[:MAX_SENTENCE_CHARS]))
    for source in ("album_summary", "album_description"):
        for sentence in _sentences(md.get(source) or ""):
            if MEDIA_MARKER.search(sentence):
                rows.append((source, normalize(sentence)))
    for fact in context_facts:
        for sentence in _sentences(fact):
            if MEDIA_MARKER.search(sentence):
                rows.append(("context_fact", normalize(sentence)))
    seen, unique = set(), []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _context_facts(context_dir: Optional[Path]) -> Tuple[Dict[str, List[str]], str]:
    """Context 산출물의 곡별 사실 문장과 자료 지문(파일 이름·sha 목록의 sha)."""
    facts: Dict[str, List[str]] = {}
    if context_dir is None:
        return facts, ""
    digest = hashlib.sha256()
    for path in sorted(context_dir.glob("*.json")):
        data = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(data).digest())
        doc = json.loads(data)
        song_id = str(((doc.get("song") or {}).get("song_id")) or "").strip()
        records = ((doc.get("retrieval") or {}).get("records")) or []
        texts = [str(r.get("evidence_text") or "") for r in records if isinstance(r, dict)]
        if song_id and texts:
            facts.setdefault(song_id, []).extend(texts)
    return facts, digest.hexdigest()


def build_store(songs_path: Path, context_dir: Optional[Path]) -> Tuple[dict, dict]:
    context, context_digest = _context_facts(context_dir)
    songs: Dict[str, List[List[str]]] = {}
    total = 0
    with open(songs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            song = json.loads(line)
            song_id = str(song.get("song_id") or "").strip()
            if not song_id:
                raise ValueError("song_id 없는 곡이 있다")
            if song_id in songs:
                raise ValueError(f"song_id 중복: {song_id}")
            total += 1
            songs[song_id] = [list(row) for row in song_mention_sentences(song, context.get(song_id, ()))]
    store = {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION, "songs": songs}
    counts = {source: sum(1 for rows in songs.values() for s, _ in rows if s == source) for source in SOURCES}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "song_count": total,
        "songs_with_sentences": sum(1 for rows in songs.values() if rows),
        "sentence_counts": counts,
        "sources": {
            "songs_jsonl": {"path_name": songs_path.name, "sha256": _sha256_file(songs_path)},
            "context_dir": {"path_name": context_dir.name if context_dir else "", "files_digest": context_digest,
                            "songs_with_facts": len(context)},
        },
    }
    return store, manifest


def write_store(store: dict, manifest: dict, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # 시각을 넣지 않아 같은 자료면 같은 바이트가 된다(sha 비교가 가능하다)
    out.write_bytes(gzip.compress(payload, mtime=0))
    manifest = dict(manifest, store_file=out.name, store_sha256=_sha256_file(out))
    manifest_path(out).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def manifest_path(store_path: Path) -> Path:
    name = store_path.name
    for suffix in (".json.gz", ".gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return store_path.with_name(name + ".manifest.json")


class WorkMentionStoreError(RuntimeError):
    """저장소 파일이 없거나 manifest와 맞지 않는다 — 기능을 켤 수 없다."""


@dataclass(frozen=True)
class MentionHit:
    song_id: str
    sources: Tuple[str, ...]


class WorkMentionStore:
    """곡별 작품 문맥 문장. 조회는 메모리 안에서만 한다(색인·DB 호출 없음)."""

    def __init__(self, songs: Mapping[str, Sequence[Sequence[str]]], manifest: Optional[dict] = None) -> None:
        self._rows: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self._compact: Dict[str, str] = {}
        for song_id, rows in songs.items():
            clean = tuple((str(s), normalize(t)) for s, t in rows if str(s) in SOURCES and str(t).strip())
            if clean:
                self._rows[str(song_id)] = clean
                self._compact[str(song_id)] = " ".join(compact(t) for _, t in clean)
        self.manifest = dict(manifest or {})

    @classmethod
    def load(cls, path: Path) -> "WorkMentionStore":
        path = Path(path)
        mpath = manifest_path(path)
        if not path.is_file() or not mpath.is_file():
            raise WorkMentionStoreError(f"작품 언급 저장소가 없다: {path.name}")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("rule_version") != RULE_VERSION:
            raise WorkMentionStoreError("작품 언급 저장소의 규칙 버전이 코드와 다르다")
        if manifest.get("store_sha256") != _sha256_file(path):
            raise WorkMentionStoreError("작품 언급 저장소 sha가 manifest와 다르다")
        store = json.loads(gzip.decompress(path.read_bytes()))
        if store.get("schema_version") != SCHEMA_VERSION or not isinstance(store.get("songs"), dict):
            raise WorkMentionStoreError("작품 언급 저장소 형식이 잘못됐다")
        return cls(store["songs"], manifest)

    def fingerprint(self) -> Dict[str, str]:
        return {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION,
                "store_sha256_16": str(self.manifest.get("store_sha256", ""))[:16]}

    def songs_mentioning(self, names: Iterable[str]) -> List[MentionHit]:
        """이름(또는 별칭) 하나라도 작품 문맥 문장에 낱말 경계로 나오는 곡. 곡 id 오름차순(결정적)."""
        names = [n for n in dict.fromkeys(" ".join(str(n).split()) for n in names) if len(compact(n)) >= 2]
        if not names:
            return []
        keys = [compact(n) for n in names]
        hits: List[MentionHit] = []
        for song_id, joined in self._compact.items():
            if not any(key in joined for key in keys):
                continue
            sources = tuple(sorted({source for source, text in self._rows[song_id]
                                    if any(media_name_in_text(name, text) for name in names)}))
            if sources:
                hits.append(MentionHit(song_id, sources))
        hits.sort(key=lambda h: (len(h.song_id), h.song_id))
        return hits


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--songs", type=Path, required=True)
    b.add_argument("--context-dir", type=Path)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out}가 이미 있다 — 덮어쓰지 않는다")
    store, manifest = build_store(args.songs, args.context_dir)
    manifest = write_store(store, manifest, args.out)
    print(json.dumps({k: manifest[k] for k in ("song_count", "songs_with_sentences", "sentence_counts", "store_sha256")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
