"""v77 앨범 소개 원문 의미 검색 — 실험 전용 공통 규칙(제품 코드 아님).

RUN_INFO §2에 고정한 규칙 한 판(`desc_semantic_v1`)만 담는다. 색인 생성기(`build_index.py`)·
검색기(`retrieve.py`)·채점기(`score.py`)·시험(`test_desc_semantic.py`)이 같이 쓴다.
정답 라벨은 이 모듈에서 읽지 않는다.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

RULE_VERSION = "desc_semantic_v1"
MAX_CHUNK_CHARS = 400            # Context 사실 단위 KoE5 색인의 MAX_FACT_CHARS와 같은 값
MIN_TITLE_CHARS = 2              # 정규화한 제목이 이보다 짧으면 제목 일치로 치지 않는다
GROUPS_K = 10                    # 주 깊이(v76 GROUPS_K · 작품명 보조 AUX_MAX_SONGS와 같은 값)
DIAG_DEPTH = 100                 # 진단용으로 저장하는 묶음 순위 깊이
CURVE_DEPTHS = (1, 3, 5, 10, 20, 50, 100)
MODEL_NAME = "nlpai-lab/KoE5"
MODEL_REVISION = "237b580aea358e3042fcf5baf78124c33c3afd16"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
EMBED_BATCH = 32                 # Context dense 기본 배치
BM25_K1 = 1.2
BM25_B = 0.75

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _main_checkout() -> Path:
    """자료·기존 벡터가 있는 메인 체크아웃(읽기만). `VF_MAIN_CHECKOUT`이 없으면 git 공용 폴더의 부모."""
    if os.environ.get("VF_MAIN_CHECKOUT"):
        return Path(os.environ["VF_MAIN_CHECKOUT"])
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


MAIN = _main_checkout()
CORPUS = MAIN / "data/all_songs.jsonl"
TEXT_DENSE_DIR = MAIN / "artifacts/embeddings/text_dense/nlpai-lab__KoE5"
INDEX_DIR = ROOT / "artifacts/description_semantic" / RULE_VERSION
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?。])\s+")
_PAREN = re.compile(r"[\(\[\{（【][^\)\]\}）】]*[\)\]\}）】]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_description(value: object) -> Optional[str]:
    """앞뒤 공백 제거 · 연속 공백(줄바꿈 포함) → 공백 하나. 문자열이 아니거나 비면 None(자료 없음)."""
    if not isinstance(value, str):
        return None
    text = _WS.sub(" ", value).strip()
    return text or None


def group_id(description: str) -> str:
    """소개 묶음 id = 정규화한 원문의 sha256 앞 16자."""
    return sha256_bytes(description.encode("utf-8"))[:16]


def _hard_split(sentence: str, limit: int) -> List[str]:
    """limit자를 넘는 한 문장: limit자 안의 마지막 공백에서 끊고, 공백이 없으면 limit자에서 끊는다."""
    pieces: List[str] = []
    rest = sentence
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        head, rest = rest[:cut].strip(), rest[cut:].strip()
        if head:
            pieces.append(head)
    if rest:
        pieces.append(rest)
    return pieces


def split_chunks(description: str, limit: int = MAX_CHUNK_CHARS) -> Tuple[List[str], int]:
    """문장을 순서대로 이어 limit자 이하 청크로 채운다(겹침 없음). (청크, 강제 분할한 문장 수).

    같은 소개 안에서 글자가 같은 청크는 첫 것만 남긴다. 빈 청크는 버린다.
    """
    sentences: List[str] = []
    forced = 0
    for raw in _SENT.split(description):
        sent = raw.strip()
        if not sent:
            continue
        if len(sent) > limit:
            forced += 1
            sentences.extend(_hard_split(sent, limit))
        else:
            sentences.append(sent)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}" if current else sent
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    seen = set()
    unique: List[str] = []
    for chunk in chunks:
        if chunk and chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return unique, forced


def normalize_title_key(text: str) -> str:
    """제목 일치용 정규화: NFKC · 괄호 부분 제거 · 소문자 · 글자/숫자만."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _PAREN.sub(" ", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


def normalize_chunk_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in text if ch.isalnum())


@dataclass
class Song:
    song_id: str
    title: str
    group: Optional[str]          # 소개 없으면 None


@dataclass
class Group:
    group_id: str
    songs: List[str]               # song_id(숫자 오름차순)
    chunks: List[str] = field(default_factory=list)
    forced_splits: int = 0


def song_sort_key(song_id: str) -> Tuple[int, str]:
    return (int(song_id), song_id) if song_id.isdigit() else (1 << 62, song_id)


def load_corpus(path: Path = CORPUS) -> Tuple[Dict[str, Song], Dict[str, Group], Dict[str, str]]:
    """(곡, 묶음, 묶음 id → 정규화 원문). 소개 없는 곡은 group=None."""
    songs: Dict[str, Song] = {}
    groups: Dict[str, Group] = {}
    texts: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec["song_id"])
            md = rec.get("metadata") or {}
            desc = normalize_description(md.get("album_description"))
            gid = group_id(desc) if desc else None
            if sid in songs:
                raise ValueError(f"duplicate song_id {sid}")
            songs[sid] = Song(sid, str(md.get("title") or ""), gid)
            if gid:
                texts.setdefault(gid, desc)
                groups.setdefault(gid, Group(gid, [])).songs.append(sid)
    for gid, group in groups.items():
        group.songs.sort(key=song_sort_key)
        group.chunks, group.forced_splits = split_chunks(texts[gid])
    return songs, groups, texts


def ordered_chunks(groups: Dict[str, Group]) -> List[Tuple[str, int, str]]:
    """(묶음 id, 청크 번호, 청크) — 묶음 id 오름차순 · 청크 번호 순(고정 순서)."""
    rows: List[Tuple[str, int, str]] = []
    for gid in sorted(groups):
        for i, chunk in enumerate(groups[gid].chunks):
            rows.append((gid, i, chunk))
    return rows


def rank_groups(chunk_groups: Sequence[str], chunk_index: Sequence[int], scores: Sequence[float],
                depth: int, *, require_positive: bool = False) -> List[Tuple[str, float, int]]:
    """청크 점수 → 묶음 순위. 묶음 점수 = 최댓값, 근거 청크 = 최댓값 청크(동점이면 번호가 작은 것).

    (점수 내림, 묶음 id 오름)으로 위 depth개. require_positive면 점수 ≤ 0인 묶음은 순위에 넣지 않는다(BM25).
    """
    best: Dict[str, Tuple[float, int]] = {}
    for gid, ci, sc in zip(chunk_groups, chunk_index, scores):
        sc = float(sc)
        if require_positive and sc <= 0.0:
            continue
        cur = best.get(gid)
        if cur is None or sc > cur[0] or (sc == cur[0] and ci < cur[1]):
            best[gid] = (sc, ci)
    ordered = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(gid, sc, ci) for gid, (sc, ci) in ordered[:depth]]


def pick_representative(group: Group, chunk: str, songs: Dict[str, Song],
                        dense_score: Dict[str, float]) -> Tuple[Optional[str], str]:
    """묶음 대표곡(정답 미사용). (song_id, 규칙) — 규칙은 title_unique · title_tie_dense · dense · none.

    ① 근거 청크에 제목이 나오는 곡이 하나면 그 곡. ② 여럿이면 그중, 없으면 묶음 전체에서
    기존 텍스트 dense 내적 최대(동점이면 song_id가 작은 곡). 기존 dense 벡터가 없는 곡은 ②에서 뺀다.
    """
    key = normalize_chunk_key(chunk)
    titled = [sid for sid in group.songs
              if len(normalize_title_key(songs[sid].title)) >= MIN_TITLE_CHARS
              and normalize_title_key(songs[sid].title) in key]
    if len(titled) == 1:
        return titled[0], "title_unique"
    pool = titled if titled else list(group.songs)
    scored = [sid for sid in pool if sid in dense_score]
    if not scored:
        return None, "none"
    scored.sort(key=lambda sid: (-dense_score[sid], song_sort_key(sid)))
    return scored[0], ("title_tie_dense" if titled else "dense")


def title_in_chunk(song: Song, chunk: str) -> bool:
    t = normalize_title_key(song.title)
    return len(t) >= MIN_TITLE_CHARS and t in normalize_chunk_key(chunk)


class BM25:
    """청크 단위 Okapi BM25(Lucene idf). 문서 = 토큰 목록."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = BM25_K1, b: float = BM25_B) -> None:
        import math

        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            tf: Dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, c in tf.items():
                self.postings.setdefault(tok, []).append((i, c))
        self.idf = {tok: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for tok, p in self.postings.items()}

    def scores(self, query_terms: Iterable[str]) -> List[float]:
        out = [0.0] * self.n
        for tok in sorted(set(query_terms)):     # 정렬 순서로 더해 부동소수 합을 고정
            posting = self.postings.get(tok)
            if not posting:
                continue
            idf = self.idf[tok]
            for i, c in posting:
                denom = c + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += idf * c * (self.k1 + 1) / denom
        return out


def iter_queries() -> Iterator[Tuple[str, str, object, bool]]:
    """(세트, 질의 id, QueryAnalysis 또는 None, fallback). 분석 캐시만 읽는다(정답 라벨 아님)."""
    from src.retrieval.analysis_cache import load_cache

    for set_name in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for qid in sorted(cache.entries):
            entry = cache.entries[qid]
            try:
                analysis = cache.get(qid, entry["query"])
            except Exception:  # noqa: BLE001 — 분석 누락은 질의를 빼지 않고 "분석 누락"으로 센다
                analysis = None
            yield set_name, qid, analysis, bool(entry.get("fallback"))


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def read_jsonl_gz(path: Path) -> List[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
