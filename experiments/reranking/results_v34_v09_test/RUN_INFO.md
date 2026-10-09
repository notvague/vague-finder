# v34 — 봉인 test 38건 1회: listwise 1패스(grounding 끔) vs 기준선 · 2026-10-09 오전

v09 봉인 test 38건을 **처음이자 마지막으로** 열었다. 채택 설정은 v33(dev 116건) 결과로 정한 Gemini listwise 1패스·grounding 끔이고,
이득을 읽기 위한 참조로 main 기준선(CE)을 같은 질의에 한 번 돌렸다. 둘 다 1회. **이 split은 더 이상 홀드아웃이 아니다** — 이후 설정 비교에 쓰지 않는다.

**결과: Hit@10 24 → 30 / 38 (0.632 → 0.789), Hit@1 14 → 19, MRR 0.460 → 0.609, 손실 0.** dev 두 세트(v06 +5, v09 +6)와 같은 방향·같은 크기다.

> **보정(2026-10-09, PR #16~#22 리뷰)**: "grounding 끔"은 일반 패스의 Google Search grounding만 끈 것이고, 희소 사실 교차검증은 희소 단서 질의(test 38 중 3건)에 Google Search 호출을 유지했다.
> 검증까지 끈 대조군은 `results_v35_rare_verify_off`(dev 116건 Hit@10 동일). 봉인 test에서 검증의 구조 규칙은 발동 0회라 검증을 끈 설정이어도 같은 결과다. 자세한 설명은 `results_v32_gemini_listwise/RUN_INFO.md` 보정 문단.

## 조건

| 항목 | 값 |
|---|---|
| 질의 | `eval_queries_v09.csv --split test` (38건), `analysis_cache_v09_test.json` (폴백 0) |
| 코퍼스·후보 | 3,010곡, 후보 30. 기준선은 main(PR #17·#18, CE 앞 20개·가중치 0.45) |
| 채택 설정 | `RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_USE_SEARCH=0 GEMINI_RERANK_PASSES=1` (가중치 0.85, 후보 30) |
| 명령 | `env <스위치> venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v09.csv --split test --analysis-cache experiments/reranking/analysis_cache_v09_test.json --output-dir experiments/reranking/results_v34_v09_test/<폴더>` |
| 소요 | 기준선 2분, listwise 3분 40초 |

## 지표 — v09 test 38 (엄격)

| 설정 | H@1 | H@5 | H@10 | MRR@10 | nDCG@10 | 회복 / 손실 (Top-10) | 리랭킹 중앙값 · p95 |
|---|---|---|---|---|---|---|---|
| 기준 (main, CE) | 14 | 21 | 24 | 0.460 | 0.502 | — | 1.8초 · 2.0초 |
| **listwise 1패스 (grounding 끔)** | **19** | **28** | **30** | **0.609** | **0.653** | 6 / 0 | 4.5초 · 12.2초 |

회복 6건: n037(밖 → 1위) · n050(2) · n056(1) · n059(5) · n085(2) · n092(1). 표지 주도 질의 6건은 라우터가 LLM 리랭킹을 건너뛴다(기준선 그대로).

유형별 Hit@10:

| 유형 (건수) | text (17) | audio (9) | image (6) | multimodal (4) | misinformation (2) |
|---|---|---|---|---|---|
| 기준 (main) | 12 | 3 | 6 | 2 | 1 |
| listwise 1패스 | 13 | 6 | 6 | 3 | 2 |

## 세 세트 한눈에 — 기준선 → listwise 1패스 (엄격 Hit@10)

| 세트 | 건수 | 기준선 | listwise 1패스 | 차이 |
|---|---|---|---|---|
| v06 dev | 57 | 37 (0.649) | 42 (0.737) | +5 |
| v09 dev | 59 | 36 (0.610) | 42 (0.712) | +6 |
| **v09 test (봉인, 1회)** | 38 | 24 (0.632) | 30 (0.789) | **+6** |
| 합계 | 154 | 97 (0.630) | 114 (0.740) | +17, 손실 0 |

Hit@1은 세 세트 합계 53 → 78.

## 읽는 법

- 세 세트 모두 +5~6, 손실 0. 튜닝에 쓰지 않은 봉인 test에서도 같은 크기라 **dev 숫자가 과적합이 아니었다**. 9월 리뷰의 정규식 우려는 여기서 닫는다
- test 38의 소리 질의가 3 → 6으로 올라 dev(5/12 그대로)와 다르다. 표본이 9건이라 우연일 수 있다 — 소리 질의 약점은 유지된 것으로 본다
- 후보 밖 미스 8건은 그대로다. 리랭커 범위 밖
- 리랭킹 중앙값 4.5초(p95 12.2초). 요청 전체로는 CE 1.8초를 빼고 더해 중앙값 약 7초

## 이 숫자의 위상

- v09 test 38은 이제 열렸다. **다음 홀드아웃이 필요하면 3차 세트를 새로 쓴다**(`docs/eval/query_writing_guide.md`)
- v06 test 25는 열지 않았다(이미 홀드아웃이 아니라 회귀 확인용). 필요하면 같은 설정으로 한 번
- 확장 지표(허용 정답)는 v09에 없다. 전부 엄격 기준이다

## 다음

1. 기본값 전환 코드: `RERANKER_BACKEND` 기본 `gemini_listwise`, `GEMINI_RERANK_USE_SEARCH` 기본 0, `GEMINI_RERANK_PASSES` 기본 1. CE는 `RERANKER_BACKEND=cross_encoder`로 남긴다.
   Gemini 키가 없거나 실패하면 CE로 폴백하는지, 설명 패널의 리랭커 문장 처리(NEXT_WORK §4)와 `tests/test_search_explain.py`를 확인한다
2. 재질문 평가를 listwise로 재측정(변동 확인) · 지연 run_v05 · 데모 리허설
3. 미검토 쌍 판정 → v09 허용 정답 → 확장 지표
