"""질의 분석 캐시 — 측정에서 분석 단계를 고정한다.

## 왜 필요한가

온도는 이미 고정돼 있다(`query_analyzer.py`: `temperature=0.0`, `top_p=1.0`).
그래도 같은 질의의 분석이 달라진다. dev Q301을 두 번 돌렸을 때 달라진 필드는
`audio_english_query` **하나**였다 —

    1회차: "...slow tempo and melancholic atmosphere."
    2회차: "...melancholic and pleading tone."

이 질의는 audio 가중치가 0.3이라 오디오 경로 순위가 달라지고, 후보 순서가 **9위부터**
갈렸다(곡 집합은 10/10 동일). 같은 `analysis`로 두 번 검색하면 점수가 비트 단위로
같으므로 검색·랭킹은 결정적이다. 즉 재현성 문제는 분석 단계에 있다.

## 이 캐시가 측정하는 것과 하지 않는 것

이것은 **분석 결과를 고정한 검색·랭킹 평가**다. 질의 분석까지 포함한 서비스 전체의
변동성은 이 캐시로 없어지지 않는다 — 측정 대상에서 분리될 뿐이다. 전체 변동성은
같은 질의를 여러 번 분석해 따로 측정해야 한다.

## 규칙

- 측정은 저장된 분석만 읽는다. **없는 질의를 측정 중에 만들지 않는다** —
  한 질의만 새로 분석되면 그 질의만 다른 조건으로 측정된다.
- 질의 원문이 캐시와 다르면 **예외**다. 질문이 바뀐 것을 모르고 옛 분석으로 측정하는
  것이 가장 나쁘다.
- API 장애로 나온 규칙 폴백(`confidence=0.0`)은 별도 표시하고, 기본적으로 측정을
  거부한다. 폴백 분석은 제목·가사 단서가 비어 있어 그 질의의 점수가 검색과 무관한
  이유로 떨어진다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.backend.schemas.query import QueryAnalysis
from src.common.claude_client import claude_request_settings
from src.common.search_models import ModelProvider, default_provider, model_name
from src.retrieval import query_analyzer as qa

CACHE_FORMAT = 1
# 지문에 `provider`를 적기 전(2026-10-14)의 캐시는 모두 Gemini 분석이다.
LEGACY_PROVIDER: ModelProvider = "gemini"


class AnalysisCacheError(RuntimeError):
    """캐시를 그대로 쓸 수 없는 상태. 측정을 시작하기 전에 멈춘다."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def analyzer_fingerprint(provider: Optional[ModelProvider] = None) -> Dict[str, Any]:
    """이 분석을 만든 조건. 나중에 "무엇으로 만든 캐시인가"를 답할 수 있어야 한다. `provider`를 비우면 서버 기본 모델.

    프롬프트와 후처리를 따로 해시하는 이유: 프롬프트만 고쳐도 분석이 달라지고,
    후처리(`_apply_*_safeguards`)만 고쳐도 최종 `QueryAnalysis`가 달라진다.
    둘 중 무엇이 바뀌었는지 구분되면 캐시를 다시 만들어야 하는지 판단할 수 있다.
    """
    analyzer_src = Path(qa.__file__).read_text(encoding="utf-8")
    modality_src = Path(qa.__file__).with_name("modality_queries.py").read_text(
        encoding="utf-8"
    )
    # Context clues are produced by a separate safeguard module. Omitting it
    # silently reuses pre-change analyses after the Context rules change.
    context_src = Path(qa.__file__).with_name("context_query.py").read_text(
        encoding="utf-8"
    )
    provider = provider or default_provider()
    return {
        "provider": provider,
        "model_name": model_name(provider),
        "claude_settings": claude_request_settings() if provider == "claude" else None,
        # query_analyzer.py가 직접 넣는 값. 여기서 바꿀 수 있는 설정이 아니다.
        "temperature": 0.0 if provider == "gemini" else None,
        "top_p": 1.0 if provider == "gemini" else None,
        "reference_year": qa._reference_year(),
        "reference_year_env": os.getenv("SEARCH_REFERENCE_YEAR", "").strip(),
        "prompt_sha": _sha(qa._PROMPT_TEMPLATE),
        "postprocess_sha": _sha(analyzer_src + modality_src + context_src),
    }


def fingerprint_sha(fingerprint: Dict[str, Any]) -> str:
    """지문 하나를 짧은 문자열로. 항목별 기록과 옛 `drift_notes`가 같은 식을 쓴다."""
    return _sha(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False))


def analyzer_sha(provider: Optional[ModelProvider] = None) -> str:
    """지금 분석기 조건의 짧은 지문. **항목마다 붙인다.**

    캐시 meta의 지문은 "이 캐시를 시작한 조건"이고, 항목별 지문은 "이 항목을
    실제로 만든 조건"이다. 둘은 갈릴 수 있다 — 조건이 어긋난 캐시를 이어서
    채울 때가 그렇다(`--reuse-despite-drift`).

    매번 계산한다. 분석 코드 세 파일을 읽지만 Gemini 호출 한 번의 수천 분의 일이고,
    캐싱하면 **환경 변수가 바뀐 것을 놓친다**(기준 연도·모델 이름이 지문에 들어간다).
    """
    return fingerprint_sha(analyzer_fingerprint(provider))


def looks_like_fallback(analysis: QueryAnalysis) -> bool:
    """Gemini 실패 시의 규칙 폴백인가.

    휴리스틱이 아니다 — `_fallback()`은 규칙만 쓰므로 결정적이고, 결과를 그대로
    비교할 수 있다. confidence만 보면 모델이 낮은 확신도를 준 정상 분석까지 폴백으로
    몰게 된다.
    """
    if analysis.confidence > 0.0:
        return False
    return analysis.model_dump() == qa._fallback(analysis.original_query).model_dump()


@dataclass
class AnalysisCache:
    meta: Dict[str, Any] = field(default_factory=dict)
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def provider(self) -> ModelProvider:
        """이 캐시의 분석을 만든 모델. 적혀 있지 않으면 옛 캐시 — 모두 Gemini다."""
        return self.meta.get("analyzer", {}).get("provider") or LEGACY_PROVIDER

    # ---- 읽기 ----
    def get(self, query_id: str, query: str) -> QueryAnalysis:
        entry = self.entries.get(query_id)
        if entry is None:
            raise AnalysisCacheError(f"캐시에 없는 질의: {query_id}")
        if entry["query"] != query:
            # 질문이 바뀐 것을 모르고 옛 분석으로 측정하면 숫자가 조용히 틀린다.
            raise AnalysisCacheError(
                f"{query_id}: 질의 원문이 캐시와 다릅니다.\n"
                f"  캐시: {entry['query']!r}\n"
                f"  현재: {query!r}\n"
                "  캐시를 다시 만들어야 합니다."
            )
        return QueryAnalysis(**entry["analysis"])

    def missing(self, query_ids: Iterable[str]) -> List[str]:
        return [qid for qid in query_ids if qid not in self.entries]

    def fallback_ids(self, query_ids: Optional[Iterable[str]] = None) -> List[str]:
        wanted = set(query_ids) if query_ids is not None else set(self.entries)
        return sorted(
            qid
            for qid, entry in self.entries.items()
            if qid in wanted and entry.get("fallback")
        )

    def fingerprint_drift(self) -> List[str]:
        """캐시를 만든 조건과 지금 조건이 다른 항목. 경고용이다.

        오류로 만들지 않는 이유: 랭킹만 바꾼 전후 비교에서는 같은 캐시를 일부러
        재사용한다. 그때 분석 조건은 바뀌지 않으므로 이 목록은 비어 있어야 하고,
        비어 있지 않다면 사람이 판단할 일이다.

        서버 기본 모델이 아니라 **이 캐시의 모델** 조건과 비교한다. 기본이 Claude로 바뀌었다고(2026-10-14) 얼려 둔
        Gemini 캐시가 어긋난 것은 아니다 — 모델을 바꾸는 것은 drift가 아니라 새 캐시다. 기본값과 비교하면 기준선
        측정 명령마다 model_name·temperature가 어긋났다고 찍히고, build_analysis_cache는 전부 다시 분석한다.
        """
        now = analyzer_fingerprint(self.provider)
        stored = self.meta.get("analyzer", {})
        return sorted(
            key for key in now if key in stored and now[key] != stored[key]
        )

    def _legacy_provenance(self) -> Dict[str, str]:
        """항목별 기록이 없던 캐시의 이력을 `drift_notes`에서 되살린다.

        항목별 `analyzer_sha`를 넣기 전에 만든 캐시(v19가 그렇다)는 조건이 어긋난
        뒤 추가된 항목을 `drift_notes[*].added_after_drift`에만 남겼다. 그것을 읽지
        않으면 **다른 분석기가 만든 항목까지 meta 묶음으로 들어간다** — 실제로
        v06 dev 4건·test 2건이 그렇게 묶였다.

        나중 기록이 앞 기록을 덮는다. 같은 질의가 두 번 다시 분석됐다면 마지막이
        그 항목을 만든 조건이다.
        """
        mapped: Dict[str, str] = {}
        for note in self.meta.get("drift_notes", []) or []:
            sha = note.get("analyzer_sha_now")
            if not sha and note.get("analyzer_now"):
                sha = fingerprint_sha(note["analyzer_now"])
            if not sha:
                continue
            for query_id in note.get("added_after_drift") or []:
                mapped[query_id] = sha
        return mapped

    def provenance(self) -> Dict[str, List[str]]:
        """어느 분석기 지문으로 만든 항목이 무엇인가. 지문 → 질의 id 목록.

        읽는 순서: 항목에 붙은 `analyzer_sha` → 옛 `drift_notes` 기록 → `"(meta)"`.
        마지막 묶음은 캐시 meta의 지문이 만든 항목들이다.
        """
        legacy = self._legacy_provenance()
        grouped: Dict[str, List[str]] = {}
        for query_id, entry in sorted(self.entries.items()):
            sha = entry.get("analyzer_sha") or legacy.get(query_id) or "(meta)"
            grouped.setdefault(sha, []).append(query_id)
        return grouped

    # ---- 쓰기 ----
    def put(self, query_id: str, query: str, analysis: QueryAnalysis) -> None:
        self.entries[query_id] = {
            "query": query,
            "fallback": looks_like_fallback(analysis),
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            # **이 항목을 만든 조건**을 여기 붙인다. 전체 작업이 끝나야 남기는
            # 기록은 중간에 끊기면 사라진다 — 실제로 한 건을 저장한 직후 중단하니
            # 그 항목이 이력에서 빠졌다. 항목과 같은 저장에 함께 들어가야 한다.
            "analyzer_sha": analyzer_sha(self.provider),
            "analysis": analysis.model_dump(mode="json"),
        }

    def save(self, path: Path) -> None:
        payload = {
            "format": CACHE_FORMAT,
            "meta": self.meta,
            "entries": dict(sorted(self.entries.items())),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(path)


def load_cache(path: Path) -> AnalysisCache:
    if not path.exists():
        raise AnalysisCacheError(f"분석 캐시가 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fmt = payload.get("format")
    if fmt != CACHE_FORMAT:
        raise AnalysisCacheError(
            f"캐시 형식이 다릅니다(파일 {fmt!r}, 기대 {CACHE_FORMAT!r}): {path}"
        )
    return AnalysisCache(meta=payload.get("meta", {}), entries=payload.get("entries", {}))


def new_cache(*, source: str, split: Optional[str], provider: Optional[ModelProvider] = None) -> AnalysisCache:
    """새 캐시. `provider`를 비우면 서버 기본 모델의 조건으로 시작한다."""
    return AnalysisCache(
        meta={
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query_source": source,
            "split": split or "all",
            "analyzer": analyzer_fingerprint(provider),
        }
    )
