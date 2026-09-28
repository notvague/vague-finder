# Gemini listwise 리랭커 — 첫 비교 측정 (#64)

**공식 기준선과 비교할 수 없다.** 아래 조건이 v0.5·v0.7과 달라서, 이 수치로 리랭커를
기본값으로 켤지 판단하면 안 된다. 켤지 말지는 현재 평가 세트로 다시 잰 뒤 정한다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-15 |
| 브랜치 | `feat/gemini-listwise-reranker` (base: #61, main 병합 전) |
| 질의 세트 | `experiments/reranking/eval_queries_v04.csv` — q118 제외 52건 |
| 리랭커 설정 | `GEMINI_RERANK_WEIGHT=0.85`, `PASSES=2`, `MAX_CANDIDATES=30`, Google Search 사용 |
| top_k / candidate_k | 10 / 30 |

```bash
RERANKER_BACKEND=gemini_listwise \
python -m src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_v04.csv \
  --top-k 10 --candidate-k 30 --exclude-query-ids q118 \
  --output-dir experiments/reranking/results_gemini
```

## 결과

| 지표 | 리랭커 없음 | Gemini listwise | 변화 |
|---|---|---|---|
| Hit@1 | 0.519 | 0.750 | +0.231 |
| Hit@5 | 0.846 | 0.942 | +0.096 |
| Hit@10 | 0.865 | 0.981 | +0.115 |
| MRR@10 | 0.652 | 0.837 | +0.184 |

개선 18 / 동일 32 / 악화 2 (q215 4→6위, q312 3→4위).
Top-10 밖에서 끌어올린 질의: q101 · q102 · q115 · q203 · q214 · q316.

main의 Cross-Encoder는 v0.5~v0.7 세 번 모두 Hit@10 변화가 0이었다. 방향 자체는
Cross-Encoder보다 확실히 낫다.

## 이 수치를 그대로 쓸 수 없는 이유 (2026-09-16 리뷰)

1. **옛 질의 세트다.** v04 52건만 썼다. 외부맥락·오정보 질의(`clarify_v1`)와
   모달리티 질의(`modality_v1`)가 빠져 있다.
2. **test split과 겹친다.** 52건 중 15건이 현재 `docs/eval/queries.json`의 test다
   (q102 q109 q111 q120 q200 q201 q212 q214 q300 q301 q302 q303 q305 q306 q309).
   리랭커 규칙을 만들 때 이 질의들을 봤으므로 홀드아웃이 아니다.
3. **q118을 뺐다.** v0.5 RUN_INFO의 "라벨 오류" 설명을 따랐는데, 그 설명은 틀렸다.
   q118은 성별을 일부러 틀리게 쓴 오정보 질의다. 빼면 실패 한 건이 가려진다.
4. **규칙이 평가 질의 표현과 겹친다.** 희소 단서 정규식(애니·드라마·OST 등 /
   휘파람·인트로·엔딩 등 / 알파벳·한자 제목)이 평가 질의 각각 15 / 7 / 8건에 걸린다.
5. **Google Search 결과는 바뀐다.** 같은 커밋으로 다시 재도 값이 다를 수 있다.
6. **느리다.** 리랭킹만 중앙값 11.6초, 최대 29.5초 (`rerank_only_ms`). 검색은 중앙값 2.8초.

## 재질문과의 충돌 — 해결, 단 변동이 크다

- `../results_clarify_gemini_v01/` — 답변을 받지 못해, 성별을 바로잡아 답해도 리랭커가
  질의의 틀린 성별을 따라 정답을 다시 내렸다 (c705: 후보 2위 → Top-10 밖).
- `../results_clarify_gemini_v02/` — 답변을 프롬프트로 넘긴 뒤 성별 정정 4건이 모두
  Top-10 안으로 돌아왔다. 다만 같은 요청이 1위 / 7위로 갈릴 만큼 실행마다 흔들린다.

기본 리랭커는 Cross-Encoder로 두고 Gemini는 `RERANKER_BACKEND=gemini_listwise`일 때만
켠다. 켜기 전에 확인할 것: 현재 평가 세트(dev) 재측정, 반복 측정 분산, Google Search를
끈 설정과의 속도·성능 비교.
