# 재질문 × Gemini listwise 리랭커 v02 — 답변 전달 후 (2건)

v01에서 확인한 문제 — 정정한 답변을 리랭커가 받지 못해 정답을 다시 내리는 것 — 를
고친 뒤 같은 2건을 다시 쟀다. **2건뿐이라 정책 비교 수치로 쓰지 않는다.**

## 무엇을 바꿨나

- `SearchRouter`가 재질문 답변을 리랭커까지 넘긴다 (`search_router.call_reranker`).
  답변을 쓰는 리랭커(`uses_clarify_answers`)에만 넘기고, Cross-Encoder는 그대로다.
- Gemini listwise 프롬프트와 희소 사실 검증 프롬프트에 "User corrections" 블록을 넣는다.
  답변은 결과를 거절한 뒤 사용자가 직접 고른 값이라 최초 질의보다 우선한다고 알린다.
  '잘 모르겠어요'는 넣지 않는다.
- 하네스도 같은 경로로 답변을 넘긴다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-16 |
| 코드 | `feat/gemini-listwise-reranker` (main 병합 + 리뷰 수정, 커밋 전 작업 트리) |
| 질의 | dev `c701`, `c705` |
| top_k / candidate_k / 답변 보너스 | 10 / 30 / ×3.0 |
| 리랭킹 호출 | 28회 (답변 포함 22회), **평균 26.2초** |
| 로그 | 호출/파싱 실패 1회(재시도로 복구, 최종 실패 0), 희소 사실 승격 3회 |

```bash
RERANKER_BACKEND=gemini_listwise \
venv/bin/python -m src.retrieval.evaluate_clarification \
  --split dev --query-ids c701,c705 \
  --output-dir experiments/reranking/results_clarify_gemini_v02
# 임베더 3종을 메인 스레드에서 미리 load() 한 뒤 실행
```

## 결과 — 정답 최종 순위 (`-`는 Top-10 밖, 괄호는 후보@30 순위)

| 질의 | 정책 | Cross-Encoder (v06) | Gemini v01 (답변 없음) | Gemini v02 (답변 전달) |
|---|---|---|---|---|
| c705 | `oracle:vocal_gender` | 2 (2) | - (2) | **1 (2)** |
| c705 | `rule:oracle` | 2 (2) | - (2) | **2 (2)** |
| c701 | `oracle:vocal_gender` | 7 (7) | - (12) | **7 (13)** |
| c701 | `rule:oracle` | 7 (7) | - (12) | **1 (13)** |
| c701 | `oracle:type` | - (15) | - (18) | 9 (18) |
| c701 | `oracle:release_era` | - (15) | - (16) | 9 (15) |
| c701 | `reject_only` | - (24) | 9 (24) | - (24) |
| c701 | `oracle:genre` | - (21) | 3 (22) | - (23) |

**답변 효과가 돌아왔다.** 성별을 정정한 네 경우가 v01에서는 전부 Top-10 밖이었는데
v02에서는 전부 Top-10 안(1 · 2 · 7 · 1위)이다. c705는 Cross-Encoder보다 한 칸 높다.

## ⚠️ 같은 입력에서도 순위가 크게 흔들린다

c701의 `oracle:vocal_gender`와 `rule:oracle`은 **완전히 같은 요청**이다(규칙이 성별을
물어 같은 답을 넣는다). 후보 순위도 13위로 같다. 그런데 최종 순위가 **7위와 1위**로 갈렸다.
c705도 같은 두 정책이 1위 / 2위다.

temperature 0이어도 Google Search 근거가 호출마다 달라지고 2-pass 결과가 흔들린다.
답변과 무관한 `reject_only`(9위 → 밖)와 `oracle:genre`(3위 → 밖)가 v01·v02 사이에
바뀐 것도 같은 원인으로 본다 — 두 정책은 이번 수정의 영향을 받지 않는다.

이 변동 폭이면 **질의 몇 건으로는 리랭커끼리 우열을 가릴 수 없다.** 켤지 판단하려면
같은 질의를 여러 번 재서 분산을 보거나, Google Search를 끈 설정과 비교해야 한다.
