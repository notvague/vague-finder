# 재질문 정책 비교 test_v05 — 3,010곡, 서비스 2턴 흐름 · 2026-10-01

`../results_clarify_v08/RUN_INFO.md`의 test 쌍이다. 평가기 변경·정의·해석은 그 문서에 함께 적었다.

**홀드아웃 평가가 아니다.** test split은 이미 여러 번 봤다. 여기서는 dev와 같은 방향인지 확인한다.

```bash
venv/bin/python -m src.retrieval.evaluate_clarification --split test \
    --analysis-cache experiments/reranking/analysis_cache_v06_test.json \
    --output-dir experiments/reranking/results_clarify_test_v05
```

개입 대상 11건, 목표 곡을 Top-10에서 본 질의 수(질의 단위 — q116은 정답 7곡의 평균):

| | 2턴 안 | 3턴 안 |
|---|---|---|
| `flow:reject` · `flow:skip` | 3.1 | 4.3 |
| `flow:oracle` | 4.3 | 5.3 |
| `flow:noisy` | 0 | 1.0 |

- 최초 검색은 v22 test와 25건 모두 같다
- 맞는 답의 이득 1.0(q102), 틀린 답의 손해 3.3(c606 · c703 · q214 · q116 2/7) — dev보다 손해 쪽으로 기운다
- `flow:oracle` 대화 17개 중 10개는 끝까지 목표 곡이 후보 밖(c608 · c704 · m401 · q118 · q200, q116 7곡 중 5곡)
- 허용 정답 참고값(`flow:oracle`): 마지막 화면 .494, 대화 중 한 번이라도 .727
