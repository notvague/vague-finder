# run_v05 — 리랭커 기본값 전환(Gemini listwise 1패스·Search 끔) 뒤 실제 API 지연 · 2026-10-09

기본 리랭커를 CE에서 Gemini listwise로 바꾼 코드(브랜치 `feat/listwise-default`)로 서버를 띄우고 dev 57건을 실제 API로 한 번씩 요청했다.
질의 분석은 실제 Gemini 호출(캐시 없음), 예열 뒤 단일 요청, `explain=true`.

**결과: 요청 전체 중앙값 4.2초 → 6.8초, p95 5.3초 → 18.8초, 최댓값 7.4초 → 21.2초.** 중앙값은 예상(약 7초)대로지만 꼬리가 길다.

## 조건

| 항목 | 값 |
|---|---|
| 코드 | `RERANKER_BACKEND` 기본 `gemini_listwise`, `GEMINI_RERANK_PASSES` 기본 1, `GEMINI_RERANK_USE_SEARCH` 기본 0, 희소 사실 검증 켬(기본) |
| 서버 | `venv/bin/uvicorn src.backend.main:app --port 8000` (`.claude/launch.json`). **`SEARCH_TIMING_LOG` 없이 띄워 서버 구간 기록이 없다** — 클라이언트 왕복만 쟀다 |
| 질의 | `eval_queries_v06.csv` dev 57 (`measure_search_latency --split dev`) |
| 비교 | CE 기준 `run_v04_hybrid_payload`: 요청 전체 4,212 · 5,315ms (중앙값 · p95) |

## 분포 (client_ms)

| | 중앙값 | p95 | 최댓값 |
|---|---|---|---|
| CE (run_v04) | 4,212 | 5,315 | 7,374 |
| **listwise 1패스 (이번)** | **6,770** | **18,791** | **21,154** |

접두별 중앙값: `c`(외부 맥락·오정보) 9건 9.3초 · `m` 10건 6.7초 · `q` 40건 6.7초. 가장 느린 8건(17~21초): m103 · q105 · q207 · c701 · q209 · q203 · m104 · c705 —
전부 드라마·영화·OST·예능 같은 **희소 단서 정규식에 걸리는 질의**다.

## 읽는 법

- **중앙값 +2.6초**는 CE 1.8초가 빠지고 listwise 4.5~5.4초가 들어온 차이 그대로다
- **p95가 3.5배**인 것은 외부 맥락 질의 때문이다. `GEMINI_RERANK_USE_SEARCH=0`은 일반 listwise 패스의 grounding만 끄고, **희소 사실 교차검증
  (`rare_fact_verification`)은 드라마·예능·OST 같은 희소 단서가 잡히면 여전히 Google Search 도구로 15곡 단위 검증을 돈다**
  (`gemini_listwise_reranker.py` `_verify_rare_facts`). 그 질의들이 17~21초다. 접두 `c`만이 아니라 m103·q203처럼 드라마·영화를 말한 질의 전부다
- 평가의 `rerank_only_ms` 중앙값 4.5~5.4초와 맞는다. 평가 때 p95 12~17초였던 것도 같은 질의들이다

## 다음

1. 꼬리를 줄이는 선택지는 셋이다 — ① `GEMINI_RERANK_RARE_FACT_VERIFY=0`(희소 사실 검증 끔; **품질 영향은 안 쟀다** — v32~v34는 켠 채 측정)
   ② 희소 사실 검증의 Search 도구만 끄기(코드) ③ 2단 응답(CE/검색 순서 먼저, LLM 순서 뒤이어). ①은 dev 116건 재측정 한 번으로 판단할 수 있다
2. 서버 구간 기록을 함께 남기려면 `SEARCH_TIMING_LOG`를 주고 다시 잰다(launch.json은 env를 못 넘겨 이번엔 뺐다)
3. 발표 리허설(`demo_rehearsal`)은 가장 느린 유형(외부 맥락)을 포함해 돌린다. `LYRICS_CACHE_TTL_SECONDS=86400`은 그대로
