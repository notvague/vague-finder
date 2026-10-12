"""폴백으로 떨어지던 5건의 질의 분석 시간(Gemini 호출 포함)을 잰다. 고치기 전·후 워크트리에서 각각 돌린다.

  PYTHONPATH=. venv/bin/python <이 파일>
"""
from __future__ import annotations

import time

from src.backend.api.dependencies import get_query_analyzer
from src.retrieval.analysis_cache import looks_like_fallback

QUERIES = {
    "t009": "작곡가가 곡마다 객원 보컬 데려와서 앨범 내는 프로젝트 있잖아, 거기서 남자 보컬이 부른 건데 이별 노래인데도 템포가 밝고 경쾌했어. 2000년대 후반",
    "t071": "FIFA 07 축구게임에서 한국어 곡 최초로 수록된 노래",
    "t086": "초록색 물감을 붓으로 마구 덧칠한 초록 숲 이미지",
    "t099": "쨍한 빨간 드레스 입은 여자가 초록 풀밭에 엎드려 턱 괴고 있는 이미지",
    "n002": "90년대 초반 록밴드 노래인데, 원래 보컬이 앨범 준비하다가 사고로 세상을 떠나서 그 동생이 대신 불렀다는 사연이 있었어.",
}


def main() -> None:
    analyzer = get_query_analyzer()
    total = 0.0
    for qid, q in QUERIES.items():
        t0 = time.monotonic()
        a = analyzer.analyze(q)
        dt = time.monotonic() - t0
        total += dt
        print(f"{qid}: {dt:5.1f}초 · 폴백={looks_like_fallback(a)}")
    print(f"평균 {total / len(QUERIES):.1f}초")


if __name__ == "__main__":
    main()
