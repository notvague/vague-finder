# 재질문 × Gemini listwise 리랭커 v01 — 답변 전달 전 (2건)

**Gemini 리랭커가 재질문 답변 효과를 지우는지** 확인하려고 잰 측정이다. 2건뿐이라
정책 비교 수치로 쓰지 않는다. 비교 대상은 같은 질의의 Cross-Encoder 측정
`../results_clarify_v06/`이다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-16 |
| 코드 | main `a4985f1` + #64의 `gemini_listwise_reranker.py` (설정 기본값) |
| 이미지 지배 질의 생략 | 적용 (#64와 같음, 이 2건에는 해당 없음) |
| 질의 | dev `c701`, `c705` — 둘 다 v06 개입 대상이자 성별 오정보 질의 |
| top_k / candidate_k / 답변 보너스 | 10 / 30 / ×3.0 |
| 리랭킹 호출 | 27회, **평균 24.6초** (두 질의 모두 "드라마" 단서로 Google Search 검증이 붙음) |
| API 실패로 원래 순서를 돌려준 횟수 | 0 |

```bash
RERANKER_BACKEND=gemini_listwise \
venv/bin/python -m src.retrieval.evaluate_clarification \
  --split dev --query-ids c701,c705 \
  --output-dir experiments/reranking/results_clarify_gemini_v01
# 임베더 3종을 메인 스레드에서 미리 load() 한 뒤 실행
```

## 결과 — 정답 최종 순위 (`-`는 Top-10 밖)

| 질의 | 정책 | 후보 순위 CE / Gemini | Cross-Encoder | Gemini |
|---|---|---|---|---|
| c705 | `oracle:vocal_gender` ("여성") | 2 / 2 | **2위** | **-** |
| c705 | `rule:oracle` | 2 / 2 | **2위** | **-** |
| c701 | `oracle:vocal_gender` ("여성") | 7 / 12 | **7위** | **-** |
| c701 | `rule:oracle` | 7 / 12 | **7위** | **-** |
| c701 | `reject_only` | 24 / 24 | - | 9위 |
| c701 | `oracle:genre` | 21 / 22 | - | 3위 |

**c705가 결정적이다.** 답변 보너스로 정답이 후보 **2위**까지 올라왔는데 Gemini가
Top-10 밖으로 내렸다. 질의는 "남자가 부르는"이라고 틀리게 기억하고, 리랭커 프롬프트에는
원래 질의만 들어가 답변이 없다. 최종 점수는 Gemini 순위 85% + 검색 순위 15%라서
답변으로 올린 검색 순위가 버티지 못한다.

c701은 Gemini가 거절만(9위)·장르 답변(3위)으로 찾았다. 다만 서비스 규칙은 성별을
먼저 묻기 때문에, 실제 흐름(`rule:oracle`)에서는 두 건 모두 놓친다.

후보 순위가 측정 사이에 다른 것(c701 7 / 12)은 Gemini 질의 분석의 비결정성이다.
c705는 양쪽 모두 후보 2위로 같으므로, 순위를 떨어뜨린 원인은 리랭커다.

## 다음

리랭커 프롬프트에 사용자 답변을 넣고 다시 쟀다 → `../results_clarify_gemini_v02/`
