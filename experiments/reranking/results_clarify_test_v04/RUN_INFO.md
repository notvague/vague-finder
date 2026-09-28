# 재질문 정책 비교 test_v04 — Qdrant 백엔드 (holdout)

`../results_clarify_v07/RUN_INFO.md`의 test 쌍.

**홀드아웃 평가가 아니다.** test split은 이미 여러 번 봤다. 여기서는 백엔드를 바꾼 뒤에도
동작이 유지되는지만 확인한다. 이 수치로 정책을 고르면 안 된다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-17 |
| 벡터 DB | Qdrant 로컬 모드 (`artifacts/qdrant`) |
| 질의 | `docs/eval/queries.json` v0.5.1 split=test, 집계 23건 |
| 개입 대상 | 7건 — c608 c704 m401 q102 q118 q200 q214 |
| top_k / candidate_k / 답변 보너스 | 10 / 30 / ×3.0 |
| 소요 | 392초 (test_v03은 425초) |

## 결과 (개입 7건)

| 정책 | Hit@10 | 비고 |
|---|---|---|
| initial | 0/7 | |
| **reject_only** | **2/7** | c608 3위, q200 4위 |
| oracle:vocal_gender | 3/7 | |
| oracle:genre | 5/7 | |
| oracle:type | 4/7 | |
| **rule:oracle** | **5/7** | q102 2위, q118 9위, q214 2위를 추가 회수 |
| rule:noisy | 1/7 | |
| **oracle:best** (천장) | 5/7 | |

전체 23건 기준 `rule:oracle` Hit@10은 21/23.

- **Hit@10 기준으로 규칙이 천장과 같다.** 상단은 다르다 — 개입 Hit@1 0/7 vs 1/7,
  Hit@5 4/7 vs 5/7.
- **오답 시 후보 이탈 0건.** 후보유지 0.714는 이 집합의 상한이다 —
  **c704와 m401은 처음부터 후보@30 밖**이라 어떤 정책으로도 복구되지 않는다.
- q200 회귀 없음 (reject_only 4위 = rule:oracle 4위).

## test_v03(Pinecone)과 비교할 때 주의

개입 대상이 6건 → **7건**으로 바뀌었다(`m401` 추가). 분모가 다르므로 전체 Hit@10을
직접 비교하면 안 된다. 회수한 질의 자체는 v03과 같은 5건이다.

`m401`은 앨범 표지 질의이고 `label_status=pending_cover`다 — 표지 실물 확인이 끝나지
않은 라벨이라 정답 여부 자체가 미확정이다.

## 재현

```bash
VECTOR_BACKEND=qdrant QDRANT_PATH=artifacts/qdrant \
venv/bin/python -m src.retrieval.evaluate_clarification \
  --split test --top-k 10 --candidate-k 30 \
  --output-dir experiments/reranking/results_clarify_test_v04
```
