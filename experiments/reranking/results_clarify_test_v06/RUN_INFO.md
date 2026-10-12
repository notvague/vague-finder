# 재질문 정책 비교 test_v06 — 리랭커 기본값(Gemini listwise)으로 다시 잰다 (v06 test 25 — v09 질의 세트 아님) · 2026-10-09

`../results_clarify_v09/RUN_INFO.md`의 test 쌍이다. 조건·정의·해석은 그 문서에 함께 적었다. 비교 대상은 `../results_clarify_test_v05/`(CE).

**홀드아웃 평가가 아니다.** v06 test 25건은 이미 여러 번 봤다. 여기서는 dev와 같은 방향인지 확인한다.
(v09 봉인 test 38건은 열지 않았다.)

| | CE (test_v05) | listwise (이번) |
|---|---|---|
| 첫 검색 Hit@10 | 14 / 25 (0.560) | 19 / 25 (0.760) |
| 개입 대상 | 11 | 6 (c608 · c703 · c704 · m401 · q118 · q200) |
| 개입 대상 중 후보 30 안 (첫 검색 · 거절 뒤 재검색) | 0.480 · 0.480 | 0.167 (1건) · 0.167 |
| 두 번 거절해도 후보 밖인 대화 | 10 / 17 | 5 / 6 |
| `flow:reject` · `oracle` · `noisy` 최종 Hit@10 | 0.731 · 0.771 · 0.600 | 0.800 · 0.800 · 0.800 |

6건 중 5건이 후보 밖이라 어떤 정책도 한 건(후보 안의 것)만 찾는다. 같은 6건만 보면 CE도 거절·맞는 답은 1건을 찾고 틀린 답만 c703을 잃었다(listwise는 잃지 않음) —
전체 기준 틀린 답 손해 −0.131 → 0은 CE에서 깎이던 c606·q116·q214(합 −2.29건 = −0.091)를 listwise가 첫 검색에서 찾아 재질문까지 가지 않은 몫이 크고, 같은 질의의 c703(−1건 = −0.040)이 나머지다(q102·m105는 CE에서도 깎이지 않았다). 맞는 답이 정답을 밀어낸 경우는 test에서 0건.
코드 `c5e7fd2`, 리랭커 실패 0건(로그 확인).

```bash
venv/bin/python -m src.retrieval.evaluate_clarification --split test \
    --analysis-cache experiments/reranking/analysis_cache_v06_test.json \
    --output-dir experiments/reranking/results_clarify_test_v06
```
소요 713초.
