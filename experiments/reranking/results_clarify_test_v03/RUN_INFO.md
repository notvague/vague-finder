# 재질문 정책 비교 test_v03 — 현재 동작 (holdout)

`../results_clarify_v06/RUN_INFO.md`의 test 쌍. 리뷰 수정 3건과 평가기 type 정답 수정(2차 리뷰) 반영 후 재측정이다.

**홀드아웃 평가가 아니다.** test split은 이미 여러 번 봤다. 여기서는 바꾼 동작이
의도대로인지만 확인하며, 이 수치로 정책을 고르면 안 된다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-16 |
| 질의 | split=test, 집계 대상 23건 |
| 개입 대상 | 6건 — c608 c704 q102 q118 q200 q214 |
| 답변 보너스 | `boost_unit × 3.0` |
| 소요 | 425초 |

## 결과 (개입 6건)

| 정책 | Hit@10 | 오답 시 후보유지 |
|---|---|---|
| reject_only | 2/6 | 0.833 |
| skip | 2/6 | 0.833 |
| oracle:vocal_gender | 3/6 | 0.833 |
| oracle:genre | 5/6 | 0.833 |
| oracle:type | 4/6 | 0.833 |
| oracle:release_era | 3/6 | 0.833 |
| noisy:type | 1/6 | 0.833 |
| noisy:release_era | 0/6 | 0.833 |
| **rule:oracle** | **5/6** | 0.833 |
| rule:noisy | 1/6 | 0.833 |
| **oracle:best** (천장) | **5/6** | 0.833 |

전체 23건 기준 `rule:oracle` Hit@10은 22/23.

- **Hit@10 기준으로** 규칙이 천장과 일치한다. 순위 상단은 다르다.

  | 지표 | rule:oracle | oracle:best |
  |---|---|---|
  | 개입 Hit@1 | 0/6 | 1/6 |
  | 개입 Hit@5 | 4/6 | 5/6 |
  | 개입 Hit@10 | 5/6 | 5/6 |
  | 전체 MRR@10 | 0.618 | 0.675 |

  가장 큰 차이는 q118이다 — 규칙(성별)은 9위, 장르 정답은 2위.
- **오답 시 후보 이탈 0건** (0.833이 이 집합의 상한 — c704가 처음부터 후보 밖).
- **q200 회귀는 없다** — reject_only 4위 → rule:oracle 4위.
- `noisy:release_era`가 0/6으로 dev(0/9)와 같은 방향이다. 제외 판단을 뒷받침한다.
- `oracle:type`이 4/6으로 dev(4/9)보다 좋다. split 구성에 따라 갈리는 슬롯이다.

## ⚠️ 개입 대상이 또 바뀌었다 — 측정 비결정성

답변과 무관한 `initial` 검색이 test_v02와 비교해 23건 중 5건에서 달랐다.
개입 대상은 test_v01(6건) → test_v02(8건) → test_v03(6건, v01과 같은 명단)으로 흔들렸다.
Gemini 질의 분석의 비결정성으로 보인다. **분모가 다른 실행끼리는 직접 비교하면 안 된다.**
dev는 v04 → v05 → v06 세 번 모두 53/53건 동일했다.

v03이 대체한 `test_v02`는 type·release_era가 반영되지 않던 측정이라 이 브랜치에서 삭제했다.

## 재현

```bash
venv/bin/python -m src.retrieval.evaluate_clarification \
  --split test --top-k 10 --candidate-k 30 \
  --output-dir experiments/reranking/results_clarify_test_v03
```
