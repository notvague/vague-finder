#!/usr/bin/env python
"""등록 장르와 사용자가 기억하는 장르·가수 정체성의 간격 — 진단 도구.

사용자가 말하는 '발라드'는 체감 스타일일 수 있고 '인디'는 가수의 활동 배경일 수 있다. 멜론 등록 장르와
다르다는 것만으로 오기억이 아니다. 이 도구는 그런 표현 차이에서 정답을 **어느 단계에서** 잃는지 남긴다.
순위 계산은 바꾸지 않는다 — 실험 배율(`GENRE_MATCH_SCALE` · `GENRE_ANSWER_SCALE`)을 프로세스 안에서
바꿔 가며 같은 분석으로 검색을 되풀이할 뿐이다.

    export   query_sources.json → queries_dev.csv (평가기가 읽는 열 + 변형 정보)
    analyze  캐시에 없는 질의를 분석해 저장하고, 모델 원본 응답(후처리 전)을 옆에 남긴다 (Gemini 호출)
    trace    장르 일치 가산 배율별 후보 단계 추적 (리랭킹 없음 · Gemini 호출 없음)
    answers  재질문 장르 답변 가산 배율별 후보 단계 비교 (리랭킹 없음 · Gemini 호출 없음)
    report   evaluate_search_accuracy 결과 폴더들을 변형별로 묶어 비교 (모델·DB 없음)

  venv/bin/python -m src.retrieval.diagnose_genre_gap export
  venv/bin/python -m src.retrieval.diagnose_genre_gap analyze \\
      --output experiments/genre_memory_gap/analysis_cache_genre_gap_dev.json
  venv/bin/python -m src.retrieval.diagnose_genre_gap trace \\
      --analysis-cache experiments/genre_memory_gap/analysis_cache_genre_gap_dev.json \\
      --model-raw experiments/genre_memory_gap/model_raw_genre_gap_dev.jsonl \\
      --deep-k 5143 --output-dir experiments/genre_memory_gap/results_v01/trace

`trace` · `answers` · `report`는 기존 평가 CSV(`eval_queries_v06.csv` 등)도 읽는다. 변형 열이 없으면
전부 한 묶음(`all`)으로 센다.

**읽을 때 주의.** 후보 단계 수치(후보 30 안 · 후보 순서의 10위)는 리랭커가 없는 값이다. 서비스의
Hit@10은 `evaluate_search_accuracy`로 리랭커를 켜고 따로 잰다. 넓은 질의(`query_type=broad`)는 정답이
하나가 아니므로 단일 정답 지표에 넣지 않고 조건 적합성·후보 다양성으로만 본다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval import search_router as _search_router
from src.retrieval.analysis_cache import AnalysisCacheError, load_cache, looks_like_fallback, new_cache
from src.retrieval.clarify import canonical_artist_types, pick_question, slot_value
from src.retrieval.explain import ExplainRecorder, SearchExplain
from src.retrieval.run_conditions import condition_differences, rerank_state_ok, run_conditions

DEFAULT_DIR = Path("experiments/genre_memory_gap")
DEFAULT_SOURCES = DEFAULT_DIR / "query_sources.json"
DEFAULT_QUERIES = DEFAULT_DIR / "queries_dev.csv"

# 평가기(`evaluate_search_accuracy.read_queries`)가 요구하는 열을 앞에 둔다. 나머지는 이 도구만 읽는다.
QUERY_COLUMNS: Sequence[str] = (
    "query_id", "split", "query_type", "query", "relevant_ids",
    "pair_id", "song_id", "group", "variant", "target_scope", "stated_genre", "perceived_genre",
    "changed_clue", "conditions", "unverifiable", "other_version_ids", "basis", "self_check", "review_status",
)

VARIANT_REGISTERED = "registered"
VARIANT_PERCEIVED = "perceived"
VARIANT_OMITTED = "genre_omitted"
VARIANT_IDENTITY = "identity_sound"
VARIANT_CONTRAST = "contrast"
VARIANT_BROAD = "broad"
VARIANT_ALL = "all"            # 변형 열이 없는 기존 평가 CSV

SCOPE_SPECIFIC = "specific"
SCOPE_BROAD = "broad"

MAIN_PATHS: Sequence[str] = ("text_hybrid", "image", "audio")
# 보컬·편성 전용 검색(`_search_performance_clues`)에서 정답의 순위를 볼 깊이. 서비스는 이 검색의 상위 30곡만 읽는다.
PERFORMANCE_DEEP_K = 300

IN_POOL = "후보 안"
NO_PATH = "경로 밖"
FUSION_CUT = "융합 절단"
FINAL_CUT = "보정 뒤 절단"

RAW_UNAVAILABLE = "원본 미확보"

# 사용자가 고를 법한 체감 장르 → 재질문 선택지에 나오는 등록 표기(후보의 첫 등록 장르).
# 이 표는 **진단용 가정**이다 — 서비스의 일치 판정(`clarify.answer_matches`)을 바꾸지 않는다.
PERCEIVED_OPTION: Dict[str, str] = {
    "발라드": "발라드", "댄스": "댄스", "힙합": "랩/힙합", "알앤비": "R&B/Soul", "록": "록/메탈",
    "포크": "포크/블루스", "인디": "인디음악", "트로트": "성인가요/트로트",
}


# ---------------------------------------------------------------------------
# export — query_sources.json → queries_dev.csv
# ---------------------------------------------------------------------------

def build_query_rows(sources: Dict[str, Any]) -> List[Dict[str, str]]:
    """원본 묶음을 평가기용 행으로 편다.

    registered · perceived는 같은 템플릿에서 장르 낱말만 바꾸고, genre_omitted는 그 낱말만 뺀다 — 세 문장의
    차이가 장르 표현 하나임을 만들 때 보장한다.
    """
    split = str(sources.get("split", "dev"))
    review = str(sources.get("review_status", ""))
    rows: List[Dict[str, str]] = []

    def row(**values: str) -> Dict[str, str]:
        base = {column: "" for column in QUERY_COLUMNS}
        base.update({"split": split, "review_status": review})
        base.update(values)
        return base

    for song in sources.get("songs", []):
        pair, song_id = str(song["pair_id"]), str(song["song_id"])
        template = str(song["template"])
        if template.count("{genre} ") != 1 or template.count("{genre}") != 1:
            raise ValueError(f"{pair}: 템플릿에는 '{{genre}} '(뒤에 공백)가 정확히 한 번 있어야 한다")
        registered, perceived = str(song["registered_term"]), str(song.get("perceived_term") or "")
        common = dict(
            query_type="search", relevant_ids=song_id, pair_id=pair, song_id=song_id,
            group=str(song["group"]), target_scope=SCOPE_SPECIFIC, perceived_genre=perceived,
            other_version_ids="|".join(song.get("other_versions", [])),
            basis=str(song.get("basis", "")), self_check=str(song.get("self_check", "")),
        )
        rows.append(row(
            query_id=f"{pair}r", query=template.format(genre=registered), variant=VARIANT_REGISTERED,
            stated_genre=registered, changed_clue="(기준) 등록 장르 표현", **common,
        ))
        if perceived:
            rows.append(row(
                query_id=f"{pair}p", query=template.format(genre=perceived), variant=VARIANT_PERCEIVED,
                stated_genre=perceived, changed_clue=f"장르 낱말만: {registered} → {perceived}", **common,
            ))
        rows.append(row(
            query_id=f"{pair}o", query=template.replace("{genre} ", ""), variant=VARIANT_OMITTED,
            stated_genre="", changed_clue=f"장르 낱말만 뺌: {registered} → (없음)", **common,
        ))
        rows.append(row(
            query_id=f"{pair}i", query=str(song["identity_sound"]), variant=VARIANT_IDENTITY,
            stated_genre="", changed_clue="장르 낱말 대신 가수 정체성·소리 표현(여러 단서가 함께 바뀜)", **common,
        ))
        for extra in song.get("contrast", []):
            rows.append(row(
                query_id=f"{pair}{extra['suffix']}", query=str(extra["query"]), variant=VARIANT_CONTRAST,
                stated_genre=str(extra.get("stated_genre", "")), changed_clue=str(extra["label"]), **common,
            ))

    for broad in sources.get("broad", []):
        rows.append(row(
            query_id=str(broad["query_id"]), query_type=VARIANT_BROAD, query=str(broad["query"]),
            variant=VARIANT_BROAD, target_scope=SCOPE_BROAD, group=VARIANT_BROAD,
            stated_genre=str(broad.get("stated_genre", "")),
            conditions=json.dumps(broad.get("conditions", {}), ensure_ascii=False, sort_keys=True),
            unverifiable="|".join(broad.get("unverifiable", [])),
            self_check=str(broad.get("note", "")),
        ))

    ids = [r["query_id"] for r in rows]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"질의 ID가 겹친다: {duplicated}")
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns) if columns else list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    """질의 CSV. 변형 열이 없는 기존 평가 CSV도 읽는다(변형은 전부 `all`, 묶음은 파일 이름 — `eval_queries_v06` → `v06`).

    **dev 행만 읽는다.** 분할을 고르는 인자를 두지 않았다 — 봉인 test를 여는 길이 이 도구에는 없다.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    rows: List[Dict[str, str]] = []
    for item in raw:
        if (item.get("split") or "").strip().lower() != "dev":
            continue
        kind = (item.get("query_type") or "").strip().lower()
        if kind not in ("search", VARIANT_BROAD):
            continue
        relevant = (item.get("relevant_ids") or "").strip()
        if kind == "search" and not relevant:
            continue
        row = {column: (item.get(column) or "").strip() for column in QUERY_COLUMNS}
        row["variant"] = row["variant"] or (VARIANT_BROAD if kind == VARIANT_BROAD else VARIANT_ALL)
        row["target_scope"] = row["target_scope"] or (SCOPE_BROAD if kind == VARIANT_BROAD else SCOPE_SPECIFIC)
        row["group"] = row["group"] or path.stem.replace("eval_queries_", "")
        row["pair_id"] = row["pair_id"] or row["query_id"]
        rows.append(row)
    if not rows:
        raise ValueError(f"읽을 dev 질의가 없다: {path}")
    return rows


def relevant_ids(row: Dict[str, str]) -> List[str]:
    return [part.strip() for part in row["relevant_ids"].split("|") if part.strip()]


def cmd_export(args: argparse.Namespace) -> None:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    rows = build_query_rows(sources)
    write_csv(Path(args.output), rows, QUERY_COLUMNS)
    counts = Counter(r["variant"] for r in rows)
    print(f"{args.output}: {len(rows)}건 — " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))


# ---------------------------------------------------------------------------
# analyze — 분석 캐시 + 모델 원본 응답
# ---------------------------------------------------------------------------

def _capturing_analyzer() -> Any:
    """`QueryAnalyzer`와 같은 길로 분석하되, 후처리에 들어가기 **전** 모델 응답을 따로 적어 둔다.

    분석 결과는 건드리지 않는다 — `_parse`가 받은 응답을 한 번 더 읽어 보관할 뿐이다. 캐시에는 후처리를
    거친 최종 분석만 있어서, 장르·형태 단서가 모델에서 빠졌는지 후처리에서 바뀌었는지는 이 기록 없이 알 수 없다.
    """
    from src.retrieval.query_analyzer import QueryAnalyzer

    class CapturingAnalyzer(QueryAnalyzer):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: List[Dict[str, Any]] = []

        def _parse(self, query: str, response: Any, salvage: Any = None) -> QueryAnalysis:
            try:
                self.attempts.append(json.loads(response.text))
            except Exception as exc:  # noqa: BLE001 - 못 읽은 응답도 "있었다"는 사실은 남긴다
                self.attempts.append({"_unparsed": type(exc).__name__})
            return super()._parse(query, response, salvage)

    return CapturingAnalyzer()


def cmd_analyze(args: argparse.Namespace) -> int:
    """캐시에 **없는 질의만** 분석해 더한다. 이미 있는 분석은 건드리지 않는다.

    다시 분석하면 같은 질의도 분석이 달라질 수 있어(온도 0이어도) 그 캐시로 잰 숫자와 이어지지 않는다.
    그래서 기본은 추가만 하고, 전부 다시 만들 때만 `--force`를 준다. 분석기 조건이 캐시를 만든 때와
    달라졌으면 섞지 않고 멈춘다.
    """
    if not os.getenv("SEARCH_REFERENCE_YEAR", "").strip():
        print("오류: SEARCH_REFERENCE_YEAR를 고정하고 돌린다(상대 시기 해석 기준).", file=sys.stderr)
        return 2
    rows = read_rows(Path(args.input))
    output, raw_path = Path(args.output), Path(args.model_raw)

    cache = new_cache(source=str(args.input), split="dev")
    raw_lines: List[str] = []
    if output.exists() and not args.force:
        existing = load_cache(output)
        drift = existing.fingerprint_drift()
        if drift:
            print(f"오류: 캐시를 만든 분석 조건이 지금과 다르다({', '.join(drift)}). 섞어 쓰지 않는다 — "
                  "전부 다시 만들려면 --force.", file=sys.stderr)
            return 2
        changed = [r["query_id"] for r in rows
                   if r["query_id"] in existing.entries and existing.entries[r["query_id"]]["query"] != r["query"]]
        if changed:
            print(f"오류: 캐시에 있는 질의의 원문이 바뀌었다 {changed}. 질의를 고쳤으면 --force로 전부 다시 만든다.",
                  file=sys.stderr)
            return 2
        cache.meta, cache.entries = existing.meta, dict(existing.entries)
        raw_lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if raw_path.exists() else []
    todo = [row for row in rows if row["query_id"] not in cache.entries]
    print(f"질의 {len(rows)}건 | 그대로 {len(rows) - len(todo)}건 | 새로 분석 {len(todo)}건", flush=True)
    if not todo:
        return 0

    analyzer = _capturing_analyzer()
    for index, row in enumerate(todo, start=1):
        analyzer.attempts = []
        analysis = analyzer.analyze(row["query"])
        fallback = looks_like_fallback(analysis)
        cache.put(row["query_id"], row["query"], analysis)
        raw_lines.append(json.dumps({
            "query_id": row["query_id"], "query": row["query"], "fallback": fallback,
            "degraded_modalities": list(getattr(analysis, "degraded_modalities", []) or []),
            # 시도마다 한 건. 마지막 것이 최종 분석의 바탕이다(폴백이면 바탕이 아니다).
            "model_attempts": list(analyzer.attempts),
        }, ensure_ascii=False))
        print(f"[{index}/{len(todo)}] {row['query_id']} genre={analysis.genre!r} "
              f"type={analysis.artist_type.values} {'(폴백)' if fallback else ''}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    cache.save(output)
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    fallbacks = cache.fallback_ids([row["query_id"] for row in rows])
    print(f"캐시: {output} ({len(cache.entries)}건, 폴백 {len(fallbacks)}건)\n모델 원본: {raw_path}")
    if fallbacks:
        print(f"경고: 폴백 분석 {fallbacks} — 이 질의는 모델 분석이 아니다. 다시 만들어야 한다.")
        return 1
    return 0


def load_model_raw(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """질의 id → 최종 분석의 바탕이 된 모델 응답. 파일이 없으면 빈 dict(원본 미확보)."""
    if path is None or not path.exists():
        return {}
    found: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        attempts = [a for a in item.get("model_attempts", []) if isinstance(a, dict) and "_unparsed" not in a]
        if attempts and not item.get("fallback"):
            found[item["query_id"]] = attempts[-1]
    return found


def _raw_fields(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if raw is None:
        return {"raw_genre": RAW_UNAVAILABLE, "raw_artist_type": RAW_UNAVAILABLE, "raw_vocal_gender": RAW_UNAVAILABLE}
    artist_type = raw.get("artist_type") if isinstance(raw.get("artist_type"), dict) else {}
    return {
        "raw_genre": str(raw.get("genre") or ""),
        "raw_artist_type": "|".join(str(v) for v in (artist_type.get("values") or [])),
        "raw_vocal_gender": str(raw.get("vocal_gender") or ""),
    }


# ---------------------------------------------------------------------------
# trace — 장르 일치 가산 배율별 후보 단계
# ---------------------------------------------------------------------------

def _norm(text: Any) -> str:
    return "".join(str(text or "").split()).lower()


def genre_matches(analysis_genre: str, registered: Optional[str]) -> bool:
    """라우터의 `genre_match` 조건과 같은 판정(부분 일치)."""
    want = _norm(analysis_genre)
    return bool(want and registered and want in _norm(registered))


def stage_of(song_id: str, record: SearchExplain, pool: Sequence[str]) -> str:
    if song_id in pool:
        return IN_POOL
    song = record.songs.get(song_id)
    if song is None or not song.paths:
        return NO_PATH
    if song.fused_score is None:
        return FUSION_CUT
    return FINAL_CUT


def _rank_by(record: SearchExplain, song_id: str, boosted: bool) -> Optional[int]:
    target = record.songs.get(song_id)
    if target is None or target.fused_score is None:
        return None

    def score(song: Any) -> float:
        return song.fused_score + (song.adjustment_total() if boosted else 0.0)

    mine = score(target)
    return 1 + sum(
        1 for s in record.songs.values()
        if s.fused_score is not None and s.song_id != song_id and score(s) > mine
    )


def paths_text(record: SearchExplain, song_id: str) -> str:
    """이 실행에서 정답이 잡힌 경로와 그 안의 순위(`경로:순위`). 융합 절단에서 빠진 기여는 `*`를 붙인다.

    후보는 주 경로(text_hybrid · image · audio)뿐 아니라 보조 신호로도 든다 — 가수 형태·시기 단서가 텍스트 상위 목록에서
    다시 뽑는 `metadata`, 보컬·편성 전용 검색 `performance`, 가사 구절 `lyrics_surface` 등. "텍스트 31위인데 후보 2위"는
    이 칸을 봐야 읽힌다. 빈칸이면 어느 목록에도 없었다.
    """
    song = record.songs.get(song_id)
    if song is None:
        return ""
    return "|".join(f"{p.path}:{p.rank}{'*' if p.dropped else ''}" for p in song.paths)


def _main_ranks(record: SearchExplain, song_id: str) -> Dict[str, Optional[int]]:
    ranks: Dict[str, Optional[int]] = {name: None for name in MAIN_PATHS}
    song = record.songs.get(song_id)
    if song is not None:
        for p in song.paths:
            if p.path in ranks and ranks[p.path] is None:
                ranks[p.path] = p.rank
    return ranks


def _set_scale(name: str, value: Optional[float]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = repr(float(value))


@contextmanager
def _restoring(*names: str) -> Iterator[None]:
    """실험 배율을 프로세스 안에서 바꿔 가며 도는 동안만 쓰고, 끝나면(예외로 끝나도) 들어올 때의 값으로 되돌린다."""
    before = {name: os.environ.get(name) for name in names}
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def parse_scales(raw: str) -> List[float]:
    scales = [float(part) for part in raw.split(",") if part.strip()]
    if not scales:
        raise ValueError("배율 목록이 비어 있다")
    if scales[0] != 1.0:
        raise ValueError("첫 배율은 1.0(현행)이어야 한다 — 나머지를 그것과 비교한다")
    for value in scales:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"배율은 0~1이다: {value}")
    return scales


def _fmt(value: Optional[Any]) -> Any:
    return "" if value is None else value


def structural_fit(track: MatchingTrack, conditions: Dict[str, Any]) -> bool:
    """넓은 질의의 조건 중 **구조화 필드로 확인되는 것만** 본다(성별·가수 형태). 조건이 없으면 참."""
    gender = conditions.get("vocal_gender")
    if gender and str(track.vocal_gender or "").strip() != gender:
        return False
    types = conditions.get("type_any") or []
    if types and not (canonical_artist_types(track.artist_types) & set(types)):
        return False
    return True


async def cmd_trace(args: argparse.Namespace) -> None:
    from src.backend.api.dependencies import get_search_router
    from src.retrieval.evaluate_search_accuracy import collect_run_info

    rows = read_rows(Path(args.input))
    wanted = {part.strip() for part in args.variants.split(",") if part.strip()}
    if wanted:
        rows = [row for row in rows if row["variant"] in wanted]
        if not rows:
            raise ValueError(f"--variants에 해당하는 질의가 없다: {sorted(wanted)}")
    cache = load_cache(Path(args.analysis_cache))
    scales = parse_scales(args.scales)
    raw_by_id = load_model_raw(Path(args.model_raw) if args.model_raw else None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델·DB를 올리기 전에 캐시부터 본다 — 없는 질의·원문 불일치·폴백은 여기서 멈춘다.
    analyses: Dict[str, QueryAnalysis] = {}
    for row in rows:
        analyses[row["query_id"]] = cache.get(row["query_id"], row["query"])
    fallbacks = cache.fallback_ids(analyses)
    if fallbacks:
        raise AnalysisCacheError(f"규칙 폴백 분석이 섞여 있다: {fallbacks}")
    # --blank-genre: 분석의 genre 칸만 비운 사본으로 돈다(진단용 — 서비스에는 이런 동작이 없다). 명시 가산과 함께
    # 보조 경로의 BM25 질의·가사 exact 보조 점수에 들어가던 장르 값이 빠진다. 질의문과 태그의 장르 낱말은 그대로다.
    blanked: Dict[str, str] = {}
    if args.blank_genre:
        blanked = {qid: analysis.genre for qid, analysis in analyses.items()}
        analyses = {qid: analysis.model_copy(update={"genre": ""}) for qid, analysis in analyses.items()}

    router = get_search_router()
    target_ids = sorted({sid for row in rows for sid in relevant_ids(row)})
    targets = router._text_svc.fetch_tracks_by_ids(target_ids)
    missing_targets = [sid for sid in target_ids if sid not in targets]

    detail: List[Dict[str, Any]] = []
    broad: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with _restoring("GENRE_MATCH_SCALE"):
        for index, row in enumerate(rows, start=1):
            qid, analysis = row["query_id"], analyses[row["query_id"]]
            raw = _raw_fields(raw_by_id.get(qid)) if args.model_raw else _raw_fields(None)
            for scale in scales:
                _set_scale("GENRE_MATCH_SCALE", scale)
                applied = _search_router._genre_match_scale()
                recorder = ExplainRecorder(row["query"])
                pool: List[str] = []
                tracks: List[MatchingTrack] = []
                await router.search(
                    analysis, top_k=args.candidate_k, use_rerank=False, candidate_k=args.candidate_k,
                    candidate_ids_out=pool, candidate_tracks_out=tracks, recorder=recorder, path_k=args.path_k,
                )
                record = recorder.record
                pool_matched = sum(1 for t in tracks if genre_matches(analysis.genre, t.genre))
                pool_boosted = sum(
                    1 for sid in pool
                    if any(a.rule == "genre_match" for a in (record.songs[sid].adjustments if sid in record.songs else []))
                )
                base = {
                    "query_id": qid, "pair_id": row["pair_id"], "group": row["group"], "variant": row["variant"],
                    "target_scope": row["target_scope"], "stated_genre": row["stated_genre"],
                    "genre_match_scale": applied, "path_k": args.path_k or args.candidate_k,
                    "analysis_genre": analysis.genre, "blanked_genre": blanked.get(qid, ""), **raw,
                    "analysis_artist_type": "|".join(analysis.artist_type.values),
                    "analysis_vocal_gender": analysis.vocal_gender or "",
                    "analysis_intent": analysis.intent_type,
                    "analysis_tags": "|".join(analysis.korean_tags),
                    "analysis_has_audio_query": int(bool(analysis.audio_english_query.strip())),
                    "pool_size": len(pool),
                    "pool_genre_matched": pool_matched,      # 등록 장르가 분석의 장르를 담은 후보 수(배율과 무관한 사실)
                    "pool_genre_boosted": pool_boosted,      # 그중 실제로 가산 기록이 남은 수(배율 0이면 0)
                }
                if row["target_scope"] == SCOPE_BROAD:
                    conditions = json.loads(row["conditions"] or "{}")
                    for depth in (args.top_k, args.candidate_k):
                        top = tracks[:depth]
                        firsts = Counter(slot_value(t, "genre") or "(없음)" for t in top)
                        broad.append({
                            **base, "depth": depth, "n": len(top),
                            "structural_conditions": row["conditions"],
                            "structural_fit": sum(1 for t in top if structural_fit(t, conditions)),
                            "registered_genre_matches_stated": sum(
                                1 for t in top if genre_matches(analysis.genre, t.genre)
                            ),
                            "distinct_first_genres": len(firsts),
                            "first_genre_counts": " ".join(f"{k}={v}" for k, v in firsts.most_common()),
                            "top_ids": "|".join(t.id for t in top),
                        })
                    continue

                deep_record: Optional[SearchExplain] = None
                performance_ids: Optional[List[str]] = None
                if args.deep_k:
                    deep_recorder = ExplainRecorder(row["query"])
                    await router.search(
                        analysis, top_k=args.candidate_k, use_rerank=False, candidate_k=args.candidate_k,
                        recorder=deep_recorder, path_k=args.deep_k,
                    )
                    deep_record = deep_recorder.record
                    if analysis.has_composite_performance_clue:
                        # 보조 검색도 같은 질의문·태그를 쓴다. 서비스가 읽는 깊이(30)보다 깊이 받아 정답의 자리를 본다.
                        performance_ids = [
                            t.id for t in router._search_performance_clues(analysis, min(args.deep_k, PERFORMANCE_DEEP_K))
                        ]
                for song_id in relevant_ids(row):
                    target = targets.get(song_id)
                    song = record.songs.get(song_id)
                    ranks = _main_ranks(deep_record, song_id) if deep_record is not None else {}
                    main = [r for r in ranks.values() if r is not None]
                    pool_rank = pool.index(song_id) + 1 if song_id in pool else None
                    detail.append({
                        **base, "song_id": song_id,
                        "target_registered_genre": (target.genre or "") if target else "(색인에 없음)",
                        "target_artist_types": "|".join(target.artist_types) if target else "",
                        "target_genre_match": int(genre_matches(analysis.genre, target.genre)) if target else "",
                        "target_type_match": int(bool(
                            set(analysis.artist_type.values) & canonical_artist_types(target.artist_types)
                        )) if target else "",
                        "pool_rank": _fmt(pool_rank),
                        "in_top_k": int(pool_rank is not None and pool_rank <= args.top_k),
                        "stage": stage_of(song_id, record, pool),
                        "target_paths": paths_text(record, song_id),
                        "target_adjustments": "" if song is None else "|".join(
                            f"{a.rule}:{a.delta:+.4f}" for a in song.adjustments
                        ),
                        "target_genre_delta": "" if song is None else round(
                            sum(a.delta for a in song.adjustments if a.rule == "genre_match"), 6
                        ),
                        **{f"{name}_rank": _fmt(ranks.get(name)) for name in MAIN_PATHS},
                        "best_main_rank": min(main) if main else "",
                        # 보컬·편성 전용 검색에서의 순위. 그 검색이 돌지 않는 질의면 "-", 깊이 안에 없으면 빈칸.
                        "performance_search_rank": (
                            "" if not args.deep_k else "-" if performance_ids is None
                            else _fmt(performance_ids.index(song_id) + 1 if song_id in performance_ids else None)
                        ),
                        "fused_rank_deep": _fmt(_rank_by(deep_record, song_id, False)) if deep_record else "",
                        "boosted_rank_deep": _fmt(_rank_by(deep_record, song_id, True)) if deep_record else "",
                        "other_version_in_pool": "|".join(
                            sid for sid in row["other_version_ids"].split("|") if sid and sid in pool
                        ),
                        "pool_ids": "|".join(pool),
                    })
            print(f"[{index}/{len(rows)}] {qid}", flush=True)
    if detail:
        write_csv(out_dir / "diagnosis.csv", detail)
        write_csv(out_dir / "sweep_summary.csv", summarize_sweep(detail, args.top_k))
        write_csv(out_dir / "pair_summary.csv", summarize_pairs(detail, scales))
        write_csv(out_dir / "stage_summary.csv", summarize_stages(detail, scales))
    if broad:
        write_csv(out_dir / "broad_fit.csv", broad)

    info = collect_run_info(
        argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}),
        reranker=None, analysis_source={
            "mode": "cache", "path": str(args.analysis_cache), "meta": cache.meta,
            "fingerprint_drift": cache.fingerprint_drift(),
            "model_raw": str(args.model_raw) if args.model_raw else RAW_UNAVAILABLE,
            "model_raw_entries": len(raw_by_id),
        },
    )
    info["genre_gap"] = {
        "tool": "trace", "rerank": False,
        # 위 ranking_switches는 추적을 마친 뒤의 프로세스 값(기본값)이다. 검색마다 실제로 적용한 배율은 아래 목록이고,
        # 행마다 diagnosis.csv의 genre_match_scale 열에 적혀 있다. 리랭커는 만들지도 부르지도 않았다.
        "genre_match_scales_applied": scales,
        "scope_of_scale": "search_router._apply_explicit_boosts의 genre_match 한 줄",
        "held_fixed": FIXED_GENRE_INFLUENCES,
        "queries": dict(Counter(r["variant"] for r in rows)),
        "targets_missing_from_index": missing_targets,
        "deep_k": args.deep_k, "path_k": args.path_k,
        "blank_genre": bool(args.blank_genre),   # 참이면 분석의 genre 칸을 비우고 돈 진단 실행이다
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "runinfo.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print_sweep(summarize_sweep(detail, args.top_k) if detail else [])
    print(f"\n기록: {out_dir}")


# 이번 실험에서 **바꾸지 않은** 장르의 영향. runinfo와 RUN_INFO에 그대로 적는다.
FIXED_GENRE_INFLUENCES: Sequence[str] = (
    "가사 exact 후보를 묶는 보조 점수 exact_lyric_constraint_score의 장르 +1.0 (보호 그룹 안의 순서)",
    "보조 경로의 BM25 질의에 들어가는 analysis.genre (_search_performance_clues · _search_title_meaning)",
    "공연 단서 필터 _build_performance_metadata_filter의 genre=랩/힙합",
    "텍스트 경로의 dense 질의·korean_tags에 들어 있는 장르 낱말(분석이 만든 검색문)",
    "곡 passage에 들어 있는 등록 장르 문자열(임베딩·BM25 색인)",
    "리랭커 프롬프트의 후보 genre 칸과 일반 속성 충돌 지침",
    "재질문에서 장르 답을 읽는 analysis_with_answers(가사 exact 묶음) · 리랭커에 넘기는 답변",
)


def summarize_sweep(detail: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """묶음(group·variant) × 배율별 후보 단계 지표. 리랭커가 없는 값이다."""
    keys: List[Tuple[str, str]] = []
    for r in detail:
        key = (r["group"], r["variant"])
        if key not in keys:
            keys.append(key)
    scales = list(dict.fromkeys(r["genre_match_scale"] for r in detail))
    out: List[Dict[str, Any]] = []
    for group, variant in keys + [("(전체)", "(전체)")]:
        for scale in scales:
            rows = [
                r for r in detail if r["genre_match_scale"] == scale
                and (group == "(전체)" or (r["group"], r["variant"]) == (group, variant))
            ]
            by_query: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                by_query.setdefault(r["query_id"], []).append(r)
            best = [min((int(x["pool_rank"]) for x in rs if x["pool_rank"] != ""), default=None)
                    for rs in by_query.values()]
            n = len(best)
            out.append({
                "group": group, "variant": variant, "genre_match_scale": scale, "n": n,
                "in_pool": sum(1 for b in best if b is not None),
                f"pool_top{top_k}": sum(1 for b in best if b is not None and b <= top_k),
                "pool_top1": sum(1 for b in best if b == 1),
                f"pool_mrr@{top_k}": round(
                    sum(1.0 / b for b in best if b is not None and b <= top_k) / n, 4
                ) if n else "",
            })
    return out


def print_sweep(summary: Sequence[Dict[str, Any]]) -> None:
    if not summary:
        return
    top_key = next(k for k in summary[0] if k.startswith("pool_top") and k != "pool_top1")
    print(f"\n{'묶음':<10}{'변형':<16}{'배율':>6}{'n':>5}{'후보 안':>8}{'후보 10위 안':>12}{'후보 1위':>9}")
    for r in summary:
        print(f"{r['group']:<10}{r['variant']:<16}{r['genre_match_scale']:>6}{r['n']:>5}"
              f"{r['in_pool']:>8}{r[top_key]:>12}{r['pool_top1']:>9}")


def _best_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """정답이 여럿인 질의의 행들 가운데 후보 순위가 가장 좋은 정답의 행(`summarize_sweep`과 같은 기준). 모두 후보 밖이면 첫 행."""
    return min(rows, key=lambda r: int(r["pool_rank"]) if r["pool_rank"] != "" else float("inf"))


def summarize_pairs(detail: Sequence[Dict[str, Any]], scales: Sequence[float]) -> List[Dict[str, Any]]:
    """곡(pair)마다 변형별·배율별 후보 순위를 한 줄에. 빈칸은 후보 밖."""
    out: List[Dict[str, Any]] = []
    for pair in dict.fromkeys(r["pair_id"] for r in detail):
        rows = [r for r in detail if r["pair_id"] == pair]
        line: Dict[str, Any] = {
            "pair_id": pair, "group": rows[0]["group"], "song_id": rows[0]["song_id"],
            "registered_genre": rows[0]["target_registered_genre"],
        }
        worst_in_pool = True
        for qid in dict.fromkeys(r["query_id"] for r in rows):
            for scale in scales:
                hit = _best_row([r for r in rows if r["query_id"] == qid and r["genre_match_scale"] == scale])
                line[f"{hit['variant']}:{qid}@{scale:g}"] = hit["pool_rank"]
                if scale == scales[0] and hit["pool_rank"] == "":
                    worst_in_pool = False
        line["all_variants_in_pool@baseline"] = int(worst_in_pool)
        out.append(line)
    columns = list(dict.fromkeys(k for line in out for k in line))
    return [{c: line.get(c, "") for c in columns} for line in out]


def summarize_stages(detail: Sequence[Dict[str, Any]], scales: Sequence[float]) -> List[Dict[str, Any]]:
    """질의마다 **현행(첫 배율)** 단계와, 장르 가산을 가장 낮춘 배율에서 달라진 것.

    가산을 낮춰도 정답이 검색 경로에 없으면(경로 밖·융합 절단) 장르 가산의 문제가 아니다 — 그 구분을 한 줄로 남긴다.
    """
    base_scale, low_scale = scales[0], min(scales)
    out: List[Dict[str, Any]] = []
    for qid in dict.fromkeys(r["query_id"] for r in detail):
        base = _best_row([r for r in detail if r["query_id"] == qid and r["genre_match_scale"] == base_scale])
        low = _best_row([r for r in detail if r["query_id"] == qid and r["genre_match_scale"] == low_scale])
        b = int(base["pool_rank"]) if base["pool_rank"] != "" else None
        lo = int(low["pool_rank"]) if low["pool_rank"] != "" else None
        if b is None and lo is not None:
            effect = "가산을 낮추면 후보에 듦"
        elif b is not None and lo is None:
            effect = "가산을 낮추면 후보에서 빠짐"
        elif b is None and lo is None:
            effect = "어느 배율에서도 후보 밖"
        elif lo < b:
            effect = f"후보 안에서 오름({b}→{lo})"
        elif lo > b:
            effect = f"후보 안에서 내림({b}→{lo})"
        else:
            effect = "변화 없음"
        if base.get("blanked_genre"):
            analysis_note = f"진단용으로 genre를 비움(원래 {base['blanked_genre']!r})"
        elif base["raw_genre"] == RAW_UNAVAILABLE:
            analysis_note = RAW_UNAVAILABLE
        elif base["raw_genre"] != base["analysis_genre"]:
            analysis_note = f"후처리가 장르를 바꿈({base['raw_genre']!r}→{base['analysis_genre']!r})"
        elif base["stated_genre"] and not base["analysis_genre"]:
            analysis_note = "모델이 장르 낱말을 genre에 담지 않음"
        else:
            analysis_note = "모델 genre = 최종 genre"
        out.append({
            "query_id": qid, "pair_id": base["pair_id"], "group": base["group"], "variant": base["variant"],
            "stated_genre": base["stated_genre"], "analysis_genre": base["analysis_genre"],
            "raw_genre": base["raw_genre"], "analysis_note": analysis_note,
            "raw_artist_type": base["raw_artist_type"], "analysis_artist_type": base["analysis_artist_type"],
            "target_registered_genre": base["target_registered_genre"],
            "target_genre_match": base["target_genre_match"], "target_type_match": base["target_type_match"],
            "stage@baseline": base["stage"], "pool_rank@baseline": base["pool_rank"],
            "target_paths@baseline": base["target_paths"], f"target_paths@{low_scale:g}": low["target_paths"],
            f"pool_rank@{low_scale:g}": low["pool_rank"], "genre_scale_effect": effect,
            "pool_genre_matched@baseline": base["pool_genre_matched"],
            "text_rank": base["text_hybrid_rank"], "audio_rank": base["audio_rank"],
            "performance_search_rank": base["performance_search_rank"],
            "fused_rank_deep": base["fused_rank_deep"], "boosted_rank_deep@baseline": base["boosted_rank_deep"],
            f"boosted_rank_deep@{low_scale:g}": low["boosted_rank_deep"],
            "target_adjustments@baseline": base["target_adjustments"],
        })
    return out


# ---------------------------------------------------------------------------
# answers — 재질문 장르 답변 가산 배율별 (후보 단계)
# ---------------------------------------------------------------------------

async def cmd_answers(args: argparse.Namespace) -> None:
    """정답이 처음 후보 10위 밖일 때, 거절 뒤 장르를 답하면 후보 순위가 어떻게 되나.

    리랭커가 없으므로 "보여 준 10곡"은 **후보 순서의 앞 10곡**이다(서비스는 리랭킹한 10곡을 보여 준다).
    정답이 여럿인 질의는 첫 정답 하나를 목표 곡으로 본다(기존 평가 CSV의 dev 176건 중 5건이 그렇다).
    답은 셋이다 — 맞는 답(정답의 첫 등록 장르) · 체감 장르 답(질의 묶음의 perceived, 있을 때) · 답 없음(거절만).
    서비스가 실제로 장르를 물을지(`pick_question`)도 함께 적는다. 질의가 장르를 이미 말했으면 묻지 않는다.
    """
    from src.backend.api.dependencies import get_search_router
    from src.retrieval.evaluate_search_accuracy import collect_run_info

    rows = [r for r in read_rows(Path(args.input)) if r["target_scope"] == SCOPE_SPECIFIC]
    cache = load_cache(Path(args.analysis_cache))
    scales = parse_scales(args.scales)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyses = {row["query_id"]: cache.get(row["query_id"], row["query"]) for row in rows}

    router = get_search_router()
    targets = router._text_svc.fetch_tracks_by_ids(sorted({sid for row in rows for sid in relevant_ids(row)}))
    detail: List[Dict[str, Any]] = []
    with _restoring("GENRE_MATCH_SCALE", "GENRE_ANSWER_SCALE"):
        _set_scale("GENRE_MATCH_SCALE", args.genre_match_scale)
        match_scale = _search_router._genre_match_scale()

        async def search(analysis: QueryAnalysis, **kwargs: Any) -> Tuple[List[str], List[MatchingTrack]]:
            pool: List[str] = []
            tracks: List[MatchingTrack] = []
            await router.search(
                analysis, top_k=args.candidate_k, use_rerank=False, candidate_k=args.candidate_k,
                candidate_ids_out=pool, candidate_tracks_out=tracks, **kwargs,
            )
            return pool, tracks

        for index, row in enumerate(rows, start=1):
            qid, analysis = row["query_id"], analyses[row["query_id"]]
            target_id = relevant_ids(row)[0]
            target = targets.get(target_id)
            _set_scale("GENRE_ANSWER_SCALE", None)
            pool, tracks = await search(analysis)
            first_rank = pool.index(target_id) + 1 if target_id in pool else None
            intervened = first_rank is None or first_rank > args.top_k
            base = {
                "query_id": qid, "pair_id": row["pair_id"], "group": row["group"], "variant": row["variant"],
                "song_id": target_id, "genre_match_scale": match_scale,
                "initial_pool_rank": _fmt(first_rank), "intervention": int(intervened),
            }
            if not intervened or target is None:
                detail.append({**base, "answer_kind": "(개입 없음)" if target else "(색인에 없음)"})
                print(f"[{index}/{len(rows)}] {qid} 개입 없음", flush=True)
                continue
            rejected = pool[:args.top_k]
            question = pick_question(analysis, tracks[args.top_k:])
            options = [o.value for o in question.options] if question is not None and question.slot == "genre" else []
            oracle = slot_value(target, "genre") or ""
            perceived = PERCEIVED_OPTION.get(row["perceived_genre"], "")
            asked = {
                "service_would_ask": question.slot if question is not None else "",
                "genre_options": "|".join(options),
            }
            plans: List[Tuple[str, Optional[str]]] = [("reject_only", None)]
            if oracle:
                plans.append(("genre_oracle", oracle))
            if perceived and perceived != oracle:
                plans.append(("genre_perceived", perceived))
            for kind, value in plans:
                for scale in (scales if value else scales[:1]):
                    _set_scale("GENRE_ANSWER_SCALE", scale)
                    applied = _search_router._genre_answer_scale()
                    answers = [ClarifyAnswer(slot="genre", value=value)] if value else None
                    recorder = ExplainRecorder(row["query"])
                    after, _ = await search(analysis, exclude_ids=rejected, answers=answers, recorder=recorder)
                    rank = after.index(target_id) + 1 if target_id in after else None
                    song = recorder.record.songs.get(target_id)
                    detail.append({
                        **base, **asked, "answer_kind": kind, "answer_value": value or "",
                        "answer_offered": int(bool(value) and value in options) if value else "",
                        "genre_answer_scale": applied if value else "",
                        "pool_rank_after": _fmt(rank),
                        "in_top_k_after": int(rank is not None and rank <= args.top_k),
                        "target_answer_delta": "" if song is None else round(
                            sum(a.delta for a in song.adjustments if a.rule == "answer_match"), 6
                        ),
                        "pool_answer_matched": sum(
                            1 for sid in after if sid in recorder.record.songs
                            and any(a.rule == "answer_match" for a in recorder.record.songs[sid].adjustments)
                        ),
                    })
            print(f"[{index}/{len(rows)}] {qid} 개입 · 서비스 질문={asked['service_would_ask'] or '없음'}", flush=True)
    columns = list(dict.fromkeys(k for r in detail for k in r))
    write_csv(out_dir / "answers.csv", [{c: r.get(c, "") for c in columns} for r in detail], columns)
    summary = summarize_answers(detail, args.top_k)
    write_csv(out_dir / "answers_summary.csv", summary)
    info = collect_run_info(
        argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}),
        reranker=None, analysis_source={
            "mode": "cache", "path": str(args.analysis_cache), "meta": cache.meta,
            "fingerprint_drift": cache.fingerprint_drift(),
        },
    )
    info["genre_gap"] = {
        "tool": "answers", "rerank": False,
        # 위 ranking_switches는 비교를 마친 뒤의 프로세스 값이다. 실제 적용값은 아래와 answers.csv의 열에 있다.
        "genre_answer_scales_applied": scales, "genre_match_scale_applied": match_scale,
        "shown_set": "후보 순서의 앞 top_k곡(리랭커 없음 — 서비스가 보여 주는 10곡과 다를 수 있다)",
        "perceived_option_map": PERCEIVED_OPTION,
    }
    (out_dir / "answers_runinfo.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for r in summary:
        print(f"  {r['variant']:<16}{r['answer_kind']:<16}배율 {str(r['genre_answer_scale']):<5}"
              f"개입 {r['n']:>3} · 후보 안 {r['in_pool']:>3} · 후보 10위 안 {r['in_top_k']:>3}")
    print(f"\n기록: {out_dir}")


def summarize_answers(detail: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = [r for r in detail if r.get("intervention") == 1 and "pool_rank_after" in r]
    keys = list(dict.fromkeys((r["variant"], r["answer_kind"], r["genre_answer_scale"]) for r in rows))
    for variant, kind, scale in keys + [("(전체)", k, s) for k, s in dict.fromkeys((r["answer_kind"], r["genre_answer_scale"]) for r in rows)]:
        group = [
            r for r in rows if r["answer_kind"] == kind and r["genre_answer_scale"] == scale
            and (variant == "(전체)" or r["variant"] == variant)
        ]
        ranks = [int(r["pool_rank_after"]) if r["pool_rank_after"] != "" else None for r in group]
        out.append({
            "variant": variant, "answer_kind": kind, "genre_answer_scale": scale, "n": len(group),
            "in_pool": sum(1 for r in ranks if r is not None),
            "in_top_k": sum(1 for r in ranks if r is not None and r <= top_k),
            "answer_offered": sum(1 for r in group if r.get("answer_offered") == 1),
        })
    return out


# ---------------------------------------------------------------------------
# report — 리랭커를 켠 측정 결과를 변형별로
# ---------------------------------------------------------------------------

def _eval_files(run_dirs: Sequence[Path], name: str) -> List[Path]:
    """결과 폴더(들)에서 평가기 산출물을 찾는다. 폴더 바로 아래 또는 세트별 하위 폴더 한 단계 아래."""
    found: List[Path] = []
    for run_dir in run_dirs:
        here = sorted(run_dir.glob(name)) or sorted(run_dir.glob(f"*/{name}"))
        if not here:
            raise FileNotFoundError(f"{name}가 없다: {run_dir}")
        found.extend(here)
    return found


def _detail_rows(run_dirs: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in _eval_files(run_dirs, "search_eval_dev_detail.csv"):
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


# 보고서 지표(Hit@10 · MRR@10)의 깊이. 평가기의 `rerank_rank`는 `--top-k`까지의 순위라 15로 재면 11~15위도 적힌다 —
# "순위가 적혀 있다"가 곧 "10위 안"은 아니다.
REPORT_TOP_K = 10


def _run_infos(run_dirs: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    return [(path, json.loads(path.read_text(encoding="utf-8")))
            for path in _eval_files(run_dirs, "search_eval_dev_runinfo.json")]


def _runinfo_switches(infos: Sequence[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
    """실행 조건 — **실제 적용값**. 한 실행이 여러 세트 폴더로 나뉘어 있으면 값이 모두 같아야 한다.

    몇 위까지 잰 실행인지 모르거나(`args.top_k` 없음) 10위보다 얕게 잰 실행은 받지 않는다 — 6~10위의 순위가 없으면
    Hit@10이 조용히 줄어든다. 분석 캐시는 세트마다 다른 것이 정상이라 값을 늘어놓기만 한다.
    """
    seen: List[Dict[str, Any]] = []
    for path, info in infos:
        run_args = info.get("args") or {}
        top_k = run_args.get("top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError(f"{path.parent}: runinfo에 args.top_k가 없다 — 몇 위까지 잰 실행인지 모르면 "
                             f"Hit@{REPORT_TOP_K}을 세지 않는다")
        if top_k < REPORT_TOP_K:
            raise ValueError(f"{path.parent}: --top-k {top_k}로 재서 {REPORT_TOP_K}위까지의 순위가 없다 — "
                             f"Hit@{REPORT_TOP_K}을 셀 수 없다")
        switches = info.get("ranking_switches", {})
        seen.append({
            "genre_match_scale": switches.get("genre_match_scale"),
            "genre_answer_scale": switches.get("genre_answer_scale"),
            "audio_fusion_scale": switches.get("audio_fusion_scale"),
            "reranker": (info.get("reranker") or {}).get("backend"),
            "gemini_backend": (info.get("gemini") or {}).get("backend"),
            "bm25": (info.get("bm25") or {}).get("sha256_12"),
            "points": (info.get("corpus") or {}).get("point_count"),
            "top_k": top_k, "candidate_k": run_args.get("candidate_k"), "path_k": run_args.get("path_k"),
        })
    if any(item != seen[0] for item in seen[1:]):
        raise ValueError(f"한 실행 안의 폴더들이 서로 다른 조건으로 측정됐다: {seen}")
    caches = dict.fromkeys(str((info.get("analysis") or {}).get("path") or "") for _, info in infos)
    return {**seen[0], "analysis_cache": " | ".join(caches)}


def _rank(value: str) -> Optional[int]:
    return int(value) if str(value).strip() else None


def _top_rank(value: str) -> Optional[int]:
    """정답의 최종 순위가 보고서 깊이(10위) 안이면 그 순위, 그 밖이거나 없으면 None."""
    rank = _rank(value)
    return rank if rank is not None and 1 <= rank <= REPORT_TOP_K else None


def cmd_report(args: argparse.Namespace) -> None:
    """`--runs 이름=폴더[,폴더…] …`의 detail을 질의 CSV와 묶는다. 첫 실행이 기준이다.

    재는 조건이 다른 실행을 표시 없이 나란히 놓지 않는다. 후보 깊이가 다른 실행은 받지 않고(`후보` 열의 뜻이 달라진다),
    그 밖의 조건은 runinfo를 기준 실행과 통째로 견줘 다른 칸을 요약 CSV(`conditions_vs_base`)와 비교 줄에 적는다 —
    실험 변수(배율)도 거기에 그대로 보인다.
    """
    meta: Dict[str, Dict[str, str]] = {}
    for path in args.queries:
        for row in read_rows(Path(path)):
            meta[row["query_id"]] = row
    runs: List[Tuple[str, List[Path]]] = []
    for item in args.runs:
        name, _, folders = item.partition("=")
        runs.append((name, [Path(folder) for folder in folders.split(",") if folder]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run: Dict[str, Dict[str, Dict[str, str]]] = {}
    switches: Dict[str, Dict[str, Any]] = {}
    conditions: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    depth: Dict[str, str] = {}
    for name, folder in runs:
        rows = {r["query_id"]: r for r in _detail_rows(folder)}
        unknown = sorted(set(rows) - set(meta))
        if unknown:
            raise ValueError(f"{name}: 질의 CSV에 없는 질의 {unknown[:5]}")
        depths = sorted({k for r in rows.values() for k in r if k.startswith("candidate_rank@")})
        if len(depths) != 1:
            raise ValueError(f"{name}: detail의 후보 순위 열이 하나가 아니다({depths}) — 후보 깊이가 다른 폴더가 섞였다")
        infos = _run_infos(folder)
        per_run[name], depth[name] = rows, depths[0]
        switches[name] = _runinfo_switches(infos)
        conditions[name] = run_conditions([info for _, info in infos])
    base_name = runs[0][0]
    if len(set(depth.values())) > 1:
        raise ValueError(f"실행마다 후보 깊이가 다르다({depth}) — `후보` 열이 같은 뜻이 아니다. "
                         "같은 --candidate-k로 잰 실행끼리 견준다.")
    differences = {name: condition_differences(conditions[base_name], conditions[name]) for name, _ in runs}

    summary: List[Dict[str, Any]] = []
    for name, _ in runs:
        rows = per_run[name]
        versus = "(기준)" if name == base_name else "; ".join(differences[name]) or "같음"
        groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        for qid, r in rows.items():
            groups.setdefault((meta[qid]["group"], meta[qid]["variant"]), []).append(r)
        groups[("(전체)", "(전체)")] = list(rows.values())
        for (group, variant), items in groups.items():
            n = len(items)
            outcomes = Counter(r.get("rerank_outcome", "") for r in items)
            summary.append({
                "run": name, "group": group, "variant": variant, "n": n,
                "in_pool": sum(1 for r in items if _rank(r[depth[name]]) is not None),
                "hit@1": sum(1 for r in items if _top_rank(r["rerank_rank"]) == 1),
                "hit@10": sum(1 for r in items if _top_rank(r["rerank_rank"]) is not None),
                "mrr@10": round(sum(1.0 / rank for r in items if (rank := _top_rank(r["rerank_rank"]))) / n, 4),
                # 실패(예외·폴백·부분 실패·상태 없음)와 정상 생략을 섞지 않는다. 실패가 있는 실행은 비교에서 빼고 다시 잰다.
                "rerank_failed": sum(count for outcome, count in outcomes.items() if not rerank_state_ok(outcome)),
                "rerank_policy_skipped": outcomes.get("policy_skipped", 0),
                "rerank_outcomes": " ".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
                **switches[name], "conditions_vs_base": versus,
            })
    write_csv(out_dir / "rerank_summary.csv", summary)

    for name, _ in runs[1:]:
        if set(per_run[name]) != set(per_run[base_name]):
            only = sorted(set(per_run[name]) ^ set(per_run[base_name]))
            raise ValueError(f"{name}: 기준 실행({base_name})과 잰 질의가 다르다({len(only)}건, 예: {only[:5]}). "
                             "중간에 멈춘 실행이면 다시 잰다.")
    changes: List[Dict[str, Any]] = []
    for qid in per_run[base_name]:
        line: Dict[str, Any] = {
            "query_id": qid, "pair_id": meta[qid]["pair_id"], "group": meta[qid]["group"],
            "variant": meta[qid]["variant"],
        }
        for name, _ in runs:
            r = per_run[name][qid]
            line[f"{name}:pool"] = r[depth[name]]
            line[f"{name}:rank"] = r["rerank_rank"]
            line[f"{name}:outcome"] = r.get("rerank_outcome", "")
        changes.append(line)
    write_csv(out_dir / "rerank_ranks.csv", changes)

    print(f"{'실행':<14}{'묶음':<10}{'변형':<16}{'n':>4}{'후보':>6}{'Hit@1':>7}{'Hit@10':>8}{'MRR':>8}{'실패':>5}{'정책생략':>7}")
    for r in summary:
        print(f"{r['run']:<14}{r['group']:<10}{r['variant']:<16}{r['n']:>4}{r['in_pool']:>6}"
              f"{r['hit@1']:>7}{r['hit@10']:>8}{r['mrr@10']:>8}{r['rerank_failed']:>5}{r['rerank_policy_skipped']:>7}")

    def in_top(run: str, qid: str) -> bool:
        return _top_rank(per_run[run][qid]["rerank_rank"]) is not None

    failed_total = {r["run"]: r["rerank_failed"] for r in summary if r["group"] == "(전체)"}
    for name, _ in runs[1:]:
        gained = [q for q in per_run[base_name] if not in_top(base_name, q) and in_top(name, q)]
        lost = [q for q in per_run[base_name] if in_top(base_name, q) and not in_top(name, q)]
        # 실패가 섞인 실행과의 비교에는 폴백 순서가 끼어 있다 — 요약 표의 `실패` 열에만 두지 않고 이 줄에도 적는다.
        warn = "".join(f" · 주의: {run} 리랭킹 실패 {failed_total[run]}건(비교에서 빼고 다시 잰다)"
                       for run in dict.fromkeys((base_name, name)) if failed_total.get(run))
        print(f"{base_name} → {name}: {REPORT_TOP_K}위 안에 든 {gained} · 빠진 {lost}{warn}"
              f" · 다른 조건: {'; '.join(differences[name]) or '없음'}")
    print(f"\n기록: {out_dir}")


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="등록 장르·체감 장르 간격 진단 (dev 전용)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="query_sources.json → queries_dev.csv")
    p.add_argument("--sources", default=str(DEFAULT_SOURCES))
    p.add_argument("--output", default=str(DEFAULT_QUERIES))
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("analyze", help="분석 캐시 + 모델 원본 응답 (Gemini 호출)")
    p.add_argument("--input", default=str(DEFAULT_QUERIES))
    p.add_argument("--output", default=str(DEFAULT_DIR / "analysis_cache_genre_gap_dev.json"))
    p.add_argument("--model-raw", default=str(DEFAULT_DIR / "model_raw_genre_gap_dev.jsonl"))
    p.add_argument("--force", action="store_true", help="이미 있는 분석도 버리고 전부 다시 분석한다(기본: 없는 것만 추가)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("trace", help="장르 일치 가산 배율별 후보 단계 (리랭킹 없음)")
    p.add_argument("--input", default=str(DEFAULT_QUERIES))
    p.add_argument("--analysis-cache", required=True)
    p.add_argument("--model-raw", default="", help="analyze가 남긴 모델 원본. 없으면 '원본 미확보'로 적는다")
    p.add_argument("--scales", default="1,0.5,0.25,0", help="GENRE_MATCH_SCALE 목록. 첫 값은 1(현행)")
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--deep-k", type=int, default=0, help="전체 깊이 실행의 path_k(코퍼스 곡 수 이상이면 전수). 0이면 생략")
    p.add_argument("--variants", default="", help="쉼표로 구분한 변형만(예: genre_omitted). 기본: 전부")
    p.add_argument(
        "--blank-genre", action="store_true",
        help="진단용 — 분석의 genre 칸만 비운 사본으로 돈다. 명시 가산과 보조 경로 질의에 들어가던 장르 값이 함께 빠지고, "
             "질의문·태그의 장르 낱말은 남는다. 원래 값은 blanked_genre 열에 적는다",
    )
    p.add_argument(
        "--path-k", type=int, default=None,
        help="세 기본 경로의 깊이(evaluate_search_accuracy --path-k와 같다). 최종 후보 수는 --candidate-k 그대로. "
             "기본: 현행 깊이. 넓힌 깊이에서 장르 가산의 영향이 달라지는지 볼 때 쓴다",
    )
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("answers", help="재질문 장르 답변 가산 배율별 후보 단계 (리랭킹 없음)")
    p.add_argument("--input", default=str(DEFAULT_QUERIES))
    p.add_argument("--analysis-cache", required=True)
    p.add_argument("--scales", default="1,0.5,0", help="GENRE_ANSWER_SCALE 목록. 첫 값은 1(현행)")
    p.add_argument("--genre-match-scale", type=float, default=None, help="함께 둘 GENRE_MATCH_SCALE (기본: 현행)")
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_answers)

    p = sub.add_parser("report", help="evaluate_search_accuracy 결과를 변형별로 비교")
    p.add_argument("--queries", nargs="+", default=[str(DEFAULT_QUERIES)])
    p.add_argument("--runs", nargs="+", required=True,
                   help="이름=결과폴더[,결과폴더…] (첫 항목이 기준). 세트별 폴더를 쉼표로 이으면 한 실행으로 합친다")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
