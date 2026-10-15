# results_v49 — Gemini 호출 경로를 Vertex AI로 통일한 뒤의 dev 기준선 (2026-10-15)

Claude/Gemini 선택(#63)을 철회하고 운영용 생성형 호출(질의 분석·리랭킹·크롤링 정제·댓글 선별)을 **Vertex AI의 Gemini**로 통일했다
(브랜치 `chore/vertex-only`). 옛 기준선(v32~v47·봉인 test v34·재질문 v09)은 전부 AI Studio(`api_key`)로 잰 것이고, 같은 모델 이름이라도
AI Studio와 Vertex는 출력이 다르므로(results_clarify_v11_prebonus) **Vertex 경로의 기준선을 새로 잰다.** 코드 변경(리랭킹 규칙·가산)은 없다.
봉인 test(v10 42건)는 열지 않았다.

- 질의: dev 176건(v06 57 · v09 59 · v10 60). 시범은 30건(세트당 10, `experiments/haiku_comparison/pilot_queries.csv`와 같은 질의)
- 색인: `artifacts/qdrant_5143_baseline_backup`의 복사본(`artifacts/qdrant_eval` · `_eval2`, 파일 해시 `58246e68ed2e` 동일) — v46과 같은 5,143곡 색인, **Context 색인 없음**
- BM25 `bf606b19ab02` · 가사 DB(Mongo `songs` 5,143곡) 내용 해시 `e0d58f28f1f4a563` — v46·Sonnet 비교 실행과 같다
- 리랭커: `gemini_listwise` 1패스 · Google Search 끔 · `GEMINI_RERANK_RARE_FACT_VERIFY=0` · 요청 제한 20초 · 예산 30초 · 가중 0.85 · 후보 30
- 모델: 검색 `gemini-3.1-flash-lite`(응답 `model_version`도 `gemini-3.1-flash-lite`), 생성 설정 temperature 0 · top_p 1 · JSON — v46과 같다
- 경로: `gemini.backend=vertex` · `location=global` · `setting=vertex`(runinfo). 인증 ADC(새 Google 계정), 프로젝트 ID는 기록하지 않는다
- 스위치: `AUDIO_FUSION_SCALE=1`(현행) · `SEARCH_REFERENCE_YEAR=2026` · 나머지 기본값

## 실행 단계

| 단계 | 분석 | 실행 이름 | 비교 대상 |
|---|---|---|---|
| ① 시범 30건 × 2회 | v46의 AI Studio 캐시(`results_v46_main_after_context/analysis_cache_*_dev_ctx.json`) 그대로 — **리랭커만 Vertex**, 후보 고정 | `aistudio_cache_pilot_rep1` · `_rep2` | 저장된 Gemini AI Studio 실행 `results_v46_main_after_context/new_scale1_rep1` · `_rep2`의 같은 30건 |
| ② dev 176 × 2회 | 같은 AI Studio 캐시 — 리랭커만 Vertex | `aistudio_cache_rep1` · `_rep2` | v46 `new_scale1_rep1` · `_rep2`(Hit@1 72 · Hit@10 113 · MRR 0.492) |
| ③ dev 176 × 2회 | **Vertex 분석기로 새로 만든 캐시** `analysis_cache_{v06,v09,v10}_dev_vertex.json` — 분석·리랭킹 모두 Vertex(전체 스택) | `vertex_cache_rep1` · `_rep2` | ②(분석 경로만 다름). **이것이 Vertex 전체 경로의 기준선이다** |

옛 캐시·결과는 덮어쓰지 않았다. 새 캐시의 지문(`meta.analyzer`)에는 `gemini_backend=vertex` · `gemini_location=global`이 들어간다
(옛 캐시는 이 키가 없고 AI Studio로 본다 — Vertex에서 이어 채우면 drift로 멈춘다).

## 결과

### 0. 전환 스모크(참고 — 저장된 입력, 각 경로 소수 호출)

| 경로 | 호출 | 결과 | 토큰(입력 · 출력) | 지연 | 추정 비용 |
|---|---|---|---|---|---|
| 설정·ADC·모델 확인(countTokens, 무료) | 2 | ok — `gemini-3.1-flash-lite` · `gemini-3.5-flash-lite`, location global | — | — | $0 |
| 질의 분석(`QueryAnalyzer`, v06 q101·q112·q204) | 3 | 3/3 정상(confidence 0.9~0.95, 폴백 0) | 23,653 · 1,417 | 2.0~2.8초 | $0.008 |
| 크롤링 정제(`refine_data`, 저장된 source.json 1곡) | 1 | 정상 응답 | 2,646 · 774 | 3.6초 | $0.003 |
| 댓글 선별(`select_emotional_comments_detailed`, 15건) | 1 | 5건 선별, 실패 배치 0 | 1,254 · 933 | 3.6초 | $0.003 |
| 리랭킹(`evaluate_search_accuracy` 3건) | 2 | applied 2 · policy_skipped 1(image_dominant) | 21,285 · 2,962 | 5.9·7.9초 | $0.010 |

합계 9회 · 약 $0.024(단가는 집계 사이트 기준 — 아래 §4). `src/crawler/context/extractor.py`(`GeminiExtractor`)는 main에서 호출부가 없고
`schemas.Analysis`가 없어 import 자체가 실패하는 미사용 모듈이라 실제 호출은 하지 않았다(클라이언트 생성 줄만 Vertex 전용으로 정리).

### 1. 시범 30건 × 2회 — 리랭커만 Vertex(①)

같은 질의·같은 AI Studio 분석·같은 후보(후보 지문 동일, Recall@30 23/30)에서 리랭커 호출 경로만 바꿨다.

| 실행 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 리랭킹 결과 | 리랭킹 시간(모델을 부른 23건) |
|---|---|---|---|---|---|---|---|---|
| 저장된 AI Studio `new_scale1_rep1` · `_rep2`(10/13) | 30 | — | 12 | 19 | 22 | 0.509 | (두 번 동일) | 기록 없음(당시 중앙값 4.3~4.5초) |
| Vertex `aistudio_cache_pilot_rep1` | 30 | 23 | **13** | **20** | **23** | **0.540** | applied 23 · policy_skipped 7 · failed 0 | 중앙값 5.46초 · p95 8.23초 · 최대 8.86초 |
| Vertex `aistudio_cache_pilot_rep2` | 30 | 23 | **13** | **20** | **23** | **0.540** | applied 23 · policy_skipped 7 · failed 0 | 중앙값 5.68초 · p95 8.68초 · 최대 22.79초 |

- Vertex 두 번은 30건 모두 정답 순위가 같다. AI Studio 저장본 대비 같음 25 · 나음 3 : 나쁨 2 — q205 밖 → 5위, q312 10 → 1위, n082 8 → 6위 / q101 5 → 6위, n031 2 → 3위.
- 호출 49회(질의 46 + 재시도 1 + 확인 2) 중 실패 1 — `504 DEADLINE_EXCEEDED`(17.7초 뒤) 1회, 재시도로 성공(그 질의 리랭킹 22.8초, 30초 예산 안). 최종 failed 0.
- 토큰: 리랭킹 1회 평균 입력 약 10.6k · 출력 약 1.4k(thoughts 0). 추정 비용 49회 $0.226.
- 응답 `model_version`은 모두 `gemini-3.1-flash-lite`(별도 날짜 판 없음).

**시범 판정**: 품질은 AI Studio 저장본과 같은 수준(30건에서 ±), 실패 0, 지연은 중앙값 +1초 안팎 — 176건으로 확대한다(②·③).

### 2. dev 176 × 2회 — 리랭커만 Vertex(②, 분석은 v46 AI Studio 캐시·후보 고정)

| 실행 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 리랭킹 결과 | 리랭킹 시간(모델을 부른 153건) |
|---|---|---|---|---|---|---|---|---|
| 저장된 AI Studio v46 `new_scale1_rep1` = `_rep2`(10/13) | 176 | 118 | 72 | 103 | 113 | 0.492 | 두 번 동일 | 기록 없음(당시 중앙값 4.3~4.5초) |
| Vertex `aistudio_cache_rep1` | 176 | 118 | **75** | **105** | **115** | **0.511** | applied 153 · policy_skipped 23 · failed 0 | 중앙값 5.07초 · p95 6.78초 · 최대 10.81초 |
| Vertex `aistudio_cache_rep2` | 176 | 118 | **75** | **105** | **115** | **0.511** | applied 153 · policy_skipped 23 · failed 0 | 중앙값 5.14초 · p95 6.96초 · 최대 9.02초 |

세트별(Vertex 두 번 동일): v06 44 · 30 · 42 · 0.591 / v09 39 · 25 · 38 · 0.505 / v10 35 · 20 · 35 · 0.440 (후보 · Hit@1 · Hit@10 · MRR).

- **Vertex 두 번은 176건 모두 정답 순위가 같다**(AI Studio 저장본도 두 번 같았다 — temperature 0에서 같은 경로는 결정적이다).
- AI Studio 저장본 대비: 같음 155 · 나음 14 : 나쁨 7 · 10위 안 새로 든 2 · 빠진 0 · 안에서 오름 12 · 내림 7. 세트별 Hit@1 v06 28 → 30 · v09 26 → 25 · v10 18 → 20.
  **같은 모델 이름·같은 입력·같은 설정에서 경로만 바꿔도 21/176의 순위가 움직인다** — AI Studio 기준선과 Vertex 측정을 직접 비교하지 않는 이유다.
- API 호출 322회(두 번 합계) 모두 성공, HTTP 실패·재시도 0. 일부 질의는 하네스가 리랭킹을 2~3회 실행했다(q211 · n007 · n064 · t003 · t018 · t025, 모두 applied — 재질문/가사 보호 흐름, 두 번 동일).
- 토큰 두 번 합계 입력 3,383,350 · 출력 455,244(thoughts 0) → 추정 $1.53(1회 $0.76, 모델을 부른 질의당 약 $0.005).
- 응답 `model_version` 모두 `gemini-3.1-flash-lite`. runinfo 6개 모두 `gemini.backend=vertex` · `location=global` · 가사 DB `e0d58f28f1f4a563` · 5,143 포인트.

### 3. dev 176 × 2회 — 분석·리랭킹 모두 Vertex(③, **Vertex 전체 경로 기준선**)

분석 캐시 `analysis_cache_{v06,v09,v10}_dev_vertex.json`(176건, 규칙 폴백 0, 지문 `gemini_backend=vertex` · `prompt_sha 80c464515d9c` ·
`postprocess_sha f18778f8f94d`). 캐시 생성은 호출 179회(질의 176 + 모달리티 검증 재시도 2 + `504 DEADLINE_EXCEEDED` 재시도 1),
입력 1,402,559 · 출력 87,085 토큰, 추정 $0.48, 분석 1회 중앙값 2.31초 · p95 3.24초.

| 실행 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 리랭킹 결과 | 리랭킹 시간(모델을 부른 155건) |
|---|---|---|---|---|---|---|---|---|
| ② Vertex 리랭커 + AI Studio 캐시 `aistudio_cache_rep1` = `_rep2` | 176 | 118 | 75 | 105 | 115 | 0.511 | applied 153 · skipped 23 · failed 0 | 중앙값 5.07 · 5.14초 |
| ③ `vertex_cache_rep1` | 176 | **119** | **80** | **105** | **114** | **0.522** | applied 155 · policy_skipped 21 · failed 0 | 중앙값 4.99초 · p95 6.79초 · 최대 23.81초 |
| ③ `vertex_cache_rep2` | 176 | **119** | **80** | **105** | **114** | **0.522** | applied 155 · policy_skipped 21 · failed 0 | 중앙값 4.93초 · p95 7.04초 · 최대 16.86초 |

세트별(③, 두 번 동일): v06 42 · 29 · 38 · 0.551 / v09 41 · 27 · 41 · 0.540 / v10 36 · 24 · 35 · 0.478 (후보 · Hit@1 · Hit@10 · MRR).

- **③ 두 번은 176건 모두 정답 순위가 같다.** 리랭킹 failed 0. API 호출 327회 중 `504 DEADLINE_EXCEEDED` 1회(18.3초 뒤) — 재시도로 성공(그 질의 23.8초, 30초 예산 안).
- ② → ③(분석 경로만 Vertex로, 후보는 새 분석으로 다시 검색): 같음 141 · 나음 20 : 나쁨 15 · 10위 안 새로 든 5(m104 · n044 · n057 · n091 · t062) · 빠진 6(c601 · c707 · m303 · q106 · q205 · t044) · 안에서 오름 15 · 내림 9.
  v06은 후보 44 → 42 · Hit@10 42 → 38로 내려가고 v09 · v10은 올라간다 — 세트별 방향이 갈린다.
- **③은 Vertex 경로뿐 아니라 지금 코드의 분석 후처리(`postprocess_sha f18778f8f94d`)로 만든 캐시다.** v46 캐시는 10/13 코드(`203a9248e607`)라
  ② → ③의 차이에는 경로와 후처리 변경이 섞여 있다(①·②는 캐시를 그대로 써서 후보가 고정됐지만, 분석을 다시 하면 둘을 가를 수 없다).
- v46 AI Studio 저장본(72 / 103 / 113 / 0.492) 대비 ③은 Hit@1 +8 · Hit@10 +1 · MRR +0.030, 질의별 나음 22 : 나쁨 12 — **다른 경로·다른 후처리의 숫자라 "개선"으로 읽지 않는다.**
  이후 Vertex 설정 비교(소리 끔·장르 가산 등)는 ③의 캐시·색인·스위치를 고정하고 ③과 견준다.
- 토큰 두 번 합계 입력 3,428,486 · 출력 451,122 → 추정 $1.53. 응답 `model_version` 모두 `gemini-3.1-flash-lite`.

### 4. 단가·비용 산정 방법

- 호출마다 저장소 밖 `sitecustomize` 탭이 `generate_content`(동기·비동기)를 감싸 `usage_*.jsonl`에 토큰·지연·`model_version`·`finish_reason`만 적었다
  (프롬프트·응답 본문 없음). 무료 countTokens 확인 호출은 세지 않는다.
- 단가(USD / 100만 토큰)는 **집계 사이트·Google 블로그 기준이며 Vertex 공식 가격표로 재확인하지 않았다**: `gemini-3.1-flash-lite` 입력 0.25 · 출력 1.50,
  `gemini-3.5-flash-lite` 입력 0.30 · 출력 2.50. thoughts 토큰은 출력 단가(이번 실행은 모두 0), 캐시 할인은 반영하지 않았다(보수적).
- 리랭킹 1회 ≈ 입력 10.6k · 출력 1.4k ≈ $0.0047, 질의 분석 1회 ≈ 입력 7.9k · 출력 0.5k ≈ $0.0027.
- 실제 청구액은 콘솔 결제 페이지에서 확인한다. 크레딧 잔액·적용 여부는 CLI로 확인할 수 없었다.

## 파일

| | |
|---|---|
| `run_eval.sh` | `CACHES=aistudio|vertex PILOT=1 REPS="1 2" QDRANT=artifacts/qdrant_eval zsh run_eval.sh` — 세트별 `evaluate_search_accuracy` |
| `build_caches.sh` | Vertex 분석기로 dev 176건 캐시 생성(③) |
| `compare_runs.py` | 실행끼리 엄격 지표·결과 코드·지연·순위 일치 비교(`--queries pilot`이면 시범 30건만) |
| `summarize.py` | v46의 요약 스크립트(그대로 복사) |
| `usage_*.jsonl` | 호출마다 토큰·지연·응답 모델 버전(프롬프트·응답 본문 없음). 저장소 밖 `sitecustomize` 탭으로 기록 |
| `*/{v06,v09,v10}_dev/` | `search_eval_dev_{summary,top10,runinfo}` 커밋 · detail·explain은 용량 때문에 git 제외 |
