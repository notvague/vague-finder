# run_v06 — 희소 사실 교차검증 끔 뒤 실제 API 지연 · 2026-10-09

`rare_fact_verification` 기본값을 False로 바꾼 코드로 서버를 띄우고 dev 57건을 실제 API로 한 번씩 요청했다(run_v05와 같은 방법 —
질의 분석 실제 Gemini 호출, 예열 뒤 단일 요청, `explain=true`). 품질 근거는 `experiments/reranking/results_v35_rare_verify_off`.

**결과: 요청 전체 중앙값 6.8초 → 7.0초(변동 범위), p95 18.8초 → 8.6초, 최댓값 21.2초 → 13.1초.** 꼬리가 사라졌다.

## 조건

| 항목 | 값 |
|---|---|
| 코드 | 기본 listwise 1패스 · Search 끔 · **희소 사실 검증 끔** (`GEMINI_RERANK_RARE_FACT_VERIFY` 기본 0) |
| 서버 | `venv/bin/uvicorn src.backend.main:app --port 8000` (`.claude/launch.json`). `SEARCH_TIMING_LOG` 없음 — 클라이언트 왕복만 |
| 질의 | `eval_queries_v06.csv` dev 57 (`measure_search_latency --split dev --output-dir experiments/latency/run_v06_rare_verify_off`) |
| 오류 | 0건 (전부 200) |

## 분포 (client_ms)

| | 중앙값 | p95 | 최댓값 |
|---|---|---|---|
| CE (run_v04) | 4,212 | 5,315 | 7,374 |
| listwise 1패스 · 검증 켬 (run_v05) | 6,770 | 18,791 | 21,154 |
| **listwise 1패스 · 검증 끔 (이번)** | **6,960** | **8,591** | **13,111** |

접두별 중앙값: `c`(외부 맥락·오정보) 9건 **6.5초**(run_v05 9.3초) · `m` 10건 7.0초 · `q` 40건 7.0초. run_v05에서 17~21초였던 8건(m103 · q105 · q207 · c701 · q209 · q203 · m104 · c705)은
이제 전부 7~8초대다. 가장 느린 1건 q115(13.1초)는 외부 맥락 질의가 아니라 그 실행의 질의 분석·리랭킹 변동이다.

## 읽는 법

- 남은 구간은 질의 분석(Gemini, 약 1.5~2초) + 검색(약 1초) + listwise 1회(4~5초)다. 질의당 Gemini 호출은 2회(분석 1 + 리랭킹 1)
- CE 대비 중앙값 +2.7초는 그대로다. 더 줄이려면 2단 응답(검색 순서 먼저, LLM 순서 뒤이어)이나 리랭킹 후보 축소뿐이다
