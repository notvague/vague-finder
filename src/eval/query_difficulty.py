"""
평가 질의 난도 사전 점검.

2026-09-15에 신규 질의 26개를 추가했다가 측정에서 18/18이 Top-10 안에 들어와
전부 폐기했다. 원인은 단순했다 — 질의에 쓴 단어가 정답의 색인 텍스트에
**그대로** 들어 있었다.

    "색소폰 소리가 반복되고"  →  정답의 sound_tags에 '색소폰' (코퍼스 5곡)
    "국악기랑 타악기"        →  정답의 sound_tags에 '타악기' (코퍼스 3곡)

코퍼스에 드문 토큰이 정답 문서에만 있으면 BM25가 그 토큰 하나로 정답을 확정한다.
축을 여러 개 겹치는 것은 난도를 올리지 않는다. 오히려 정답을 좁혀 준다.

반대로 실제로 어려운 질의는 이 검사에서 깨끗하다.

    q101 "싸이월드 배경음악" (후보 24위)  → 희소 토큰 0개
    q203 "짱구 애니메이션 OST" (후보 23위) → 희소 토큰 0개

외부 맥락이 색인 대상이 아니기 때문이다. `passage_builder`에는 album이 없어
앨범명에만 있는 드라마·예능·영화 이름은 어떤 경로로도 매칭되지 않는다.

  python -m src.eval.query_difficulty                # 전체 질의 점검
  python -m src.eval.query_difficulty --query-set clarify_v1
"""
from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage
from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set

DEFAULT_CORPUS = Path("data/all_songs.jsonl")

# 이 비율 이하로 등장하는 토큰이 정답 문서에 있으면 사실상 정답을 지목한다.
#
# 절대 개수가 아니라 비율인 이유: 희소성은 코퍼스 상대적이다. 905곡에서 40곡은
# 4.4%지만 2,095곡에서 40곡은 1.9%라 훨씬 더 강하게 정답을 지목한다.
# 코퍼스가 커질 때 임계값을 손대지 않아도 되도록 비율로 둔다.
RARE_DF_RATIO = 0.045

# 하위 호환 및 소규모 코퍼스용 하한. 비율로 계산한 값이 이보다 작으면 이걸 쓴다.
MIN_RARE_DF = 10


def rare_df_threshold(corpus_size: int) -> int:
    """코퍼스 크기에 맞춘 '희소' 기준."""
    return max(MIN_RARE_DF, round(corpus_size * RARE_DF_RATIO))

# 난도와 무관한 기능어. 검사에서 빼지 않으면 노이즈만 는다.
STOPWORDS = {
    "노래", "곡", "느낌", "그리고", "같아", "있어", "이었어", "했어", "부르는", "부른",
    "나오는", "것", "거", "좀", "되게", "엄청", "진짜", "약간", "같은", "정도", "때",
    "그런", "무슨", "어떤", "저런", "이런", "했던", "하는", "위해", "대해",
}


def load_corpus(path: Path = DEFAULT_CORPUS) -> Dict[str, dict]:
    songs: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                song = json.loads(line)
                songs[song["song_id"]] = song
    return songs


@functools.lru_cache(maxsize=4096)
def _indexed_text(song_json: str) -> str:
    song = json.loads(song_json)
    return build_dense_passage(song) + "\n" + build_sparse_passage(song)


def indexed_text(song: dict) -> str:
    """검색이 실제로 보는 텍스트 (dense + sparse passage)."""
    return _indexed_text(json.dumps(song, ensure_ascii=False, sort_keys=True))


def content_words(query: str) -> List[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", query)
    return [w for w in dict.fromkeys(words) if len(w) >= 2 and w not in STOPWORDS]


def document_frequency(token: str, texts: List[str]) -> int:
    return sum(1 for t in texts if token in t)


def revealing_tokens(
    query: str,
    target: dict,
    texts: List[str],
    rare_df: int | None = None,
) -> List[Tuple[str, int]]:
    """정답을 지목해 버리는 토큰 목록 [(토큰, 코퍼스 DF)].

    비어 있으면 그 질의는 표면 매칭만으로는 풀 수 없다 — 의도한 상태다.
    """
    if rare_df is None:
        rare_df = rare_df_threshold(len(texts))
    text = indexed_text(target)
    found: List[Tuple[str, int]] = []
    for word in content_words(query):
        if word in text:
            df = document_frequency(word, texts)
            if df <= rare_df:
                found.append((word, df))
    return sorted(found, key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 질의 난도 사전 점검")
    parser.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--query-set", default=None)
    parser.add_argument("--rare-df", type=int, default=None,
                        help="생략하면 코퍼스 크기 × 4.5%%로 자동 계산")
    args = parser.parse_args()

    songs = load_corpus(args.corpus)
    texts = [indexed_text(s) for s in songs.values()]
    eval_set = load_eval_set(args.queries)

    flagged = 0
    checked = 0
    for q in eval_set.queries:
        if args.query_set and q.query_set != args.query_set:
            continue
        if not q.positives:
            continue
        target = songs.get(q.positives[0])
        if target is None:
            continue
        checked += 1
        found = revealing_tokens(q.query, target, texts, args.rare_df)
        if found:
            flagged += 1
            tokens = ", ".join(f"{w}(DF={d})" for w, d in found)
            print(f"⚠️  {q.query_id} [{q.query_set}] 정답을 지목하는 토큰: {tokens}")

    threshold = args.rare_df or rare_df_threshold(len(texts))
    print(f"\n코퍼스 {len(texts)}곡 · 희소 기준 DF≤{threshold}")
    print(f"점검 {checked}개 / 지목 토큰 있는 질의 {flagged}개")
    if flagged:
        print("→ 해당 토큰을 우회 표현으로 바꾸거나, 외부 맥락·오정보 유형으로 다시 쓸 것")


if __name__ == "__main__":
    main()
