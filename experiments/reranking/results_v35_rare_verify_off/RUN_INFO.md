# v35 — 희소 사실 교차검증 끔(`GEMINI_RERANK_RARE_FACT_VERIFY=0`) 품질 · 2026-10-09

리랭커 기본값 전환(#23) 뒤 요청 지연 p95가 18.8초였고(`experiments/latency/run_v05_listwise_default`), 꼬리는 전부 드라마·영화·OST·예능을 말한
질의에서 **희소 사실 교차검증**(`_verify_rare_facts`)이 Google Search 도구로 15곡 단위 배치를 도는 시간이었다. 이 검증을 끄면 품질이 떨어지는지
dev 116건(v06 57 + v09 59)에서 한 번 쟀다. 코드 변경 없음(스위치만).

**결과: Hit@10 합산 84 → 84(v06 42 → 43, v09 42 → 41), Hit@1 59 → 58. 바뀐 질의는 전부 ±1 변동 범위이고 검증 때문이라고 볼 근거가 없다.
외부 맥락 질의(22건) 리랭킹 중앙값 15.3~15.7초 → 4.2~4.8초, 전체 리랭킹 p95 17.6초 → 5.8초. → 기본값을 끔으로 바꾼다.**

## 조건

| 항목 | 값 |
|---|---|
| 비교 대상 | 같은 설정에 검증만 켠 기록 — v06 `results_v32_gemini_listwise/nosearch_pass1`, v09 `results_v33_v09_dev/lw_pass1` |
| 설정 | `RERANKER_BACKEND=gemini_listwise`(기본) · 1패스 · Search 끔(기본) · **`GEMINI_RERANK_RARE_FACT_VERIFY=0`** · 가중치 0.85 · 후보 30 |
| 분석 | v06 `analysis_cache_v06_dev.json`, v09 `analysis_cache_v09_dev.json`(n002·n025 폴백, `--allow-fallback-analysis`) — 비교 대상과 동일 |
| 명령 | `env RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v0{6,9}.csv --split dev --analysis-cache … --output-dir experiments/reranking/results_v35_rare_verify_off/v0{6,9}_dev` (`run.sh`) |
| 소요 | v06 4분 54초, v09 4분 40초 (검증 켠 v32·v33 1패스는 7~9분) |
| 비교 스크립트 | `compare.py` |

## 지표

| 세트 | 지표 | 검증 켬 | 검증 끔 | 차이 |
|---|---|---|---|---|
| v06 dev 57 | Hit@1 | 0.491 (28) | 0.474 (27) | −1 |
| | Hit@5 | 0.702 | 0.702 | 0 |
| | Hit@10 | 0.737 (42) | 0.754 (43) | +1 |
| | MRR@10 | 0.560 | 0.556 | −0.003 |
| v09 dev 59 | Hit@1 | 0.525 (31) | 0.525 (31) | 0 |
| | Hit@5 | 0.678 | 0.678 | 0 |
| | Hit@10 | 0.712 (42) | 0.695 (41) | −1 |
| | MRR@10 | 0.588 | 0.589 | +0.001 |
| **합산 116** | **Hit@1 / Hit@10** | **59 / 84** | **58 / 84** | **−1 / 0** |

결과 분포는 두 세트 모두 동일(v06 applied 52 · skipped 5, v09 applied 49 · skipped 10). failed 0.

## 바뀐 질의

| 세트 | 질의 | 켬 → 끔 | 희소 단서 질의? | 읽는 법 |
|---|---|---|---|---|
| v06 | q202 (릴스 챌린지 역주행 여자아이돌) | Top-10 밖 → 7위 | 아니오 | 검증과 무관한 실행 변동 |
| v06 | c707 (후각 주인공 드라마 삽입곡) | 1위 → 2위 | 예 | 켠 기록에서도 구조 규칙은 발동하지 않았다 — 변동 |
| v09 | n033 (2024 봄 재벌집 딸 드라마 OST, 남자 R&B) | 8위 → Top-10 밖 | 예 | 켠 기록의 8위는 listwise 모델이 15위에서 올린 것이고 구조 규칙 발동 없음 — 변동 |
| v09 | n051 / n057 | 2→1 / 1→2 | 아니오 | 변동 |

순위가 조금이라도 바뀐 질의는 v06 6건, v09 4건이다. 검증을 켠 비교 기록도 같은 설정의 두 실행이 ±1 갈리던 범위(v32 `grounded` vs `grounded_rep2`)다.

## 검증이 실제로 한 일 — 3,010곡 listwise 기록 전부

검증 결과는 **`gemini_rare_fact_rescue` 구조 규칙으로만** 순서에 반영된다(검증 support·확신도가 문턱을 넘은 곡을 9위로 삽입). 그 규칙이 발동한 횟수를
3,010곡 listwise 기록(`*_explain.jsonl`) 전부에서 셌다:

| 기록 | 발동 | 올린 곡 |
|---|---|---|
| v32 `nosearch_pass1` · v33 `lw_pass1` · v34 `lw_pass1`(봉인 test) — **채택한 설정** | **0** | — |
| v32 `grounded` · `nosearch` | 0 | — |
| v32 `grounded_rep2` | 1 | q203 — **오답**을 18위 → 9위 |
| v33 `lw_nosearch`(2패스) | 1 | n044 — **오답**을 27위 → 9위 |
| v33 `lw_grounded` | 1 | n096 — **오답**을 12위 → 9위 |

열 번 안팎의 실행에서 3번 발동했고 **셋 다 오답을 Top-10에 넣었다.** 정답을 올린 적은 없다. 설계 때 겨냥한 q200형(휘파람 + 남녀 듀엣, 952곡 시절)은
3,010곡에서는 검증 없이도 1위다(q207·q219). 봉인 test(v34)에서도 발동 0이라, **끄는 것이 봉인 test 결과(30/38)를 바꾸지 않는다.**

## 비용

| | 외부 맥락 질의(각 세트 11건) 리랭킹 중앙값 | 전체 리랭킹 중앙값 · p95 |
|---|---|---|
| v06 켬 → 끔 | 15.3초 → 4.8초 | 5.4 · 17.6초 → 4.6 · 5.8초 |
| v09 켬 → 끔 | 15.7초 → 4.2초 | 5.4 · 16.5초 → 4.5 · 5.8초 |

질의당 Gemini 호출도 외부 맥락 질의에서 3회(검증 배치 2 + 패스 1) → 1회로 준다.

## 결정

`rare_fact_verification` 기본값을 **False**로 바꾼다(`GeminiListwiseRerankerConfig`, `GEMINI_RERANK_RARE_FACT_VERIFY` 기본 0). 코드는 남겨 두고
`GEMINI_RERANK_RARE_FACT_VERIFY=1`로 되돌릴 수 있다. 요청 전체 지연은 `experiments/latency/run_v06_rare_verify_off`에서 다시 쟀다.
