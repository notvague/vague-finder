"""
대표 감정(major_emotion) 통제 어휘

설명: 크롤러가 LLM에게 대표 감정을 받을 때 쓰는 한국어 고정 어휘와, 목록 밖 값을 정규화하는
      규칙이다. 사용처는 refine_data 하나다(프롬프트 + 응답 정규화).

      수집분 961곡의 major_emotion은 **전부 영어**였고 자유 생성이라 45종으로 흩어져 있었다.

          Sadness(390) Joy(125) Melancholy(101) Excitement(83) Nostalgia(65) ... 45종
          Joy(125)·Joyful(3)·Happiness(1) / Sensuality(3)·Sensual(1) / Longing(14)·Yearning(3)

      다른 태그 필드는 모두 한국어인데 이 필드만 영어였다. 새로 수집하는 곡부터 한국어 고정
      어휘 중 하나를 고르게 한다.

검색에서 이 값이 쓰이는 곳:
      - 색인: sparse passage(BM25)의 감정 줄. dense passage(KoE5)에는 없다.
      - 색인 밖: 벡터 DB 메타데이터를 거쳐 기본 리랭커(Cross-Encoder) 문서의 `대표 감정` 줄과
        검색 API 응답(`major_emotion`)에 그대로 들어간다.

      이 모듈은 색인(passage_builder)에서 쓰지 않는다. 기존 영어 값을 이 어휘로 옮겨 색인하는
      변경을 A/B로 재 보니 현재 76개 평가 질의에서는 개선을 확인하지 못했다(Hit@10 동일, 상위
      순위 2건 하락). 감정 표현이 든 질의는 8개뿐이라 일반적으로 효과가 없다는 입증은 아니지만,
      이번 PR에서는 색인 변경을 제외했다.
          근거: experiments/reranking/results_emotion_index_ab/RUN_INFO.md

      따라서 이번 PR에서는 기존 배포 색인을 다시 적재할 필요가 없다. 다만 앞으로 이 어휘로 수집한 곡은 영어
      값이던 기존 곡과 값이 달라진다(예: `Sadness` 대신 `슬픔`). 지금 passage 로직으로 색인하면
      감정 줄의 토큰이 달라지고, 리랭커 문서와 API 응답에도 영어·한국어 값이 섞인다. A/B는
      passage만 바꿨으므로(두 조건 모두 리랭커에는 영어 값) 리랭커 쪽 변화는 재지 않았다.

작성자: 황찬혁 (Full)
생성일: 2026-09-17
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

# LLM이 고를 수 있는 값. 늘리면 같은 뜻이 다시 여러 값으로 갈라질 수 있다.
EMOTION_VOCAB: Tuple[str, ...] = (
    "슬픔",
    "쓸쓸함",
    "그리움",
    "추억",
    "위로",
    "기쁨",
    "신남",
    "설렘",
    "사랑",
    "자신감",
    "열정",
    "분노",
    "몽환",
    "희망",
)

# ---------------------------------------------------------------------------
# 이미 수집된 영어 값 45종 → 한국어
#
# 매핑이 빠짐없는지와 매핑이 맞는지는 별개다. 곡마다 자체 emotion_tags·mood_tags와
# 대조해 보니(2026-09-17) 정적 표로는 틀리는 값이 있었다.
#
#   Excitement(83)  댄스곡에서는 '신남'이지만 '썸 탈꺼야'·'금요일에 만나요'는 태그가
#                   '두근거림·설렘'이다. 하나로 고정하면 3분의 1이 틀린다.
#   Obsession       에픽하이 'Fan' — 태그가 고독·광기·집착. '설렘'으로 옮기면 반대 뜻이다.
#   Sensuality      'Love Shot' — 섹시·치명적. '설렘'으로 옮기면 곡과 다른 감정이 저장된다.
#   Cynicism 등     '봄이 좋냐??' — 위트·유머. '분노'가 아니다.
#
# 그래서 영어 값은 세 갈래로 나눈다.
#   1) 뜻이 하나로 정해지는 값      — 그대로 옮긴다.
#   2) 뜻이 갈리는 값              — 그 곡의 한국어 태그에 근거가 있는 후보를 고른다.
#                                   같은 LLM 호출이 만든 태그라 곡의 맥락을 담고 있다.
#   3) 대응하는 어휘가 없는 값      — 비운다.
# 분류표에 없는 값(주로 목록을 벗어난 한국어 답: '행복', '외로움', '설레임')은 **답 자체**에
# 들어 있는 어간으로 후보를 찾는다. 후보가 하나면 그 값, 여럿이면 태그 근거로 가리고, 없으면 비운다.
# 태그만 보고 옮기지는 않는다 — 'Obsession'처럼 대응 어휘가 없는 값이 태그에 걸린 다른 감정으로
# 바뀌면 안 되기 때문이다.
# 근거 없이 억지로 옮기면 곡과 반대 뜻의 감정이 저장된다. 틀린 값보다 빈 값이 낫다.
# ---------------------------------------------------------------------------

_UNAMBIGUOUS: Dict[str, str] = {
    "sadness": "슬픔",
    "melancholy": "쓸쓸함", "loneliness": "쓸쓸함",
    "longing": "그리움", "yearning": "그리움",
    "nostalgia": "추억",
    "healing": "위로",
    "joy": "기쁨", "happiness": "기쁨", "euphoria": "기쁨",
    "love": "사랑", "affection": "사랑", "romance": "사랑",
    "empowerment": "자신감", "confidence": "자신감", "ambition": "자신감",
    "resilience": "자신감", "determination": "자신감",
    "passion": "열정",
    "resentment": "분노",
    "mysterious": "몽환",
    "hopeful": "희망",
}

# 곡 태그에서 근거가 가장 많이 나온 후보를 고른다. 동점이면 앞의 것(직역에 가까운 쪽).
# 근거가 하나도 없으면 비운다.
_NEEDS_EVIDENCE: Dict[str, Tuple[str, ...]] = {
    "excitement": ("신남", "설렘"),
    "joyful": ("기쁨", "설렘", "신남"),
    "bittersweet": ("그리움", "쓸쓸함"),
    "defiance": ("자신감", "분노"),
    "liberation": ("자신감", "희망"),
    "intensity": ("열정", "신남"),
    "angst": ("슬픔", "분노"),
    "despair": ("슬픔",),
    "frustration": ("분노",),
    "relief": ("위로",),
    "catharsis": ("위로",),
    "gratitude": ("위로", "사랑"),
    "sensuality": ("설렘",),
    "sensual": ("설렘",),
    "captivation": ("설렘",),
    "desire": ("그리움",),
    "cool": ("자신감",),
}

# 통제 어휘에 이 라벨과 같은 뜻의 값이 없어 후보를 두지 않고, 태그와 상관없이 비운다.
# 태그에 어휘 어간이 걸리는 곡은 있다(2026-09-17 수집분 9곡 중 8곡, 예: '환희' 희망·위로,
# 'Dirty Cash' 분노·열정). 이는 곡의 다른 감정을 가리킬 뿐, 라벨을 옮길 근거로 보지 않는다.
_NO_EQUIVALENT = frozenset({
    "obsession",   # 에픽하이 'Fan' — 고독·광기·집착
    "suspense",    # '미친거니' — 공포·불안·섬뜩함
    "cynicism",    # 'Dirty Cash', '그런 남자' — 씁쓸함·풍자
    "sarcasm",     # '봄이 좋냐??' — 위트·유머
    "conflict",    # '두사랑' — 혼란·갈등
    "mood",        # 감정이 아니다
})

# 근거로 인정할 어간. 태그만 본다 — 반응 요약은 '향수를 느낀다' 같은 상투구가 많아
# 그리움·추억에 거짓 근거를 만든다.
_EVIDENCE_STEMS: Dict[str, Tuple[str, ...]] = {
    "슬픔": ("슬픔", "슬프", "애절", "비통", "눈물", "상처", "비애"),
    "쓸쓸함": ("쓸쓸", "고독", "외로", "공허"),
    "그리움": ("그리움", "그립", "향수", "회상", "애틋"),
    "추억": ("추억", "향수", "회상"),
    "위로": ("위로", "위안", "치유", "힐링"),
    "기쁨": ("기쁨", "기쁘", "행복", "즐거", "희열"),
    "신남": ("신나", "신남", "흥겨", "흥분", "경쾌", "활기", "에너지", "파티"),
    "설렘": ("설렘", "설레", "두근"),
    "사랑": ("사랑", "로맨", "애정"),
    "자신감": ("자신감", "당당", "자존", "투지"),
    "열정": ("열정", "정열"),
    "분노": ("분노", "증오", "원망", "반항"),
    "몽환": ("몽환", "신비"),
    "희망": ("희망",),
}

KNOWN_ENGLISH = frozenset(_UNAMBIGUOUS) | frozenset(_NEEDS_EVIDENCE) | _NO_EQUIVALENT

_VOCAB_SET = frozenset(EMOTION_VOCAB)


def _evidence_count(candidate: str, tags: Iterable[str]) -> int:
    """후보의 어간이 걸리는 태그 수."""
    stems = _EVIDENCE_STEMS[candidate]
    return sum(1 for tag in tags if tag and any(stem in str(tag) for stem in stems))


def canonical_emotion(value: object, tags: Iterable[str] = ()) -> str:
    """대표 감정을 통제 어휘의 한국어 값으로 옮긴다.

    Args:
        value: 저장된 major_emotion (한국어 어휘, 예전 영어 값, 또는 목록 밖 한국어 답)
        tags:  그 곡의 emotion_tags + mood_tags. 후보가 여럿일 때 가리는 데만 쓴다.

    옮길 근거가 없으면 빈 문자열을 돌려준다.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if text in _VOCAB_SET:
        return text

    key = text.lower()
    if key in _UNAMBIGUOUS:
        return _UNAMBIGUOUS[key]

    if key in KNOWN_ENGLISH:
        # 뜻이 갈리는 값은 후보가 정해져 있고, 대응 어휘가 없는 값은 후보가 없다(비운다).
        candidates = _NEEDS_EVIDENCE.get(key, ())
    else:
        # 분류표에 없는 값. 답 자체에 든 어간으로만 후보를 찾는다('행복' -> 기쁨).
        candidates = tuple(
            vocab for vocab in EMOTION_VOCAB
            if any(stem in text for stem in _EVIDENCE_STEMS[vocab])
        )
        if len(candidates) == 1:
            return candidates[0]

    tag_list = list(tags or ())
    best, best_count = "", 0
    for candidate in candidates:
        count = _evidence_count(candidate, tag_list)
        if count > best_count:          # 동점이면 먼저 나온 후보를 유지한다
            best, best_count = candidate, count
    return best


def vocab_prompt_line() -> str:
    """LLM 프롬프트에 넣을 목록 문자열."""
    return " | ".join(EMOTION_VOCAB)
