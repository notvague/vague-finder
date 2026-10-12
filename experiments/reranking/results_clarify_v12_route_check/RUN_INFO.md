# 재질문 호출 경로 확인 v12 — 검색만 AI Studio로 고정해 v10 base 재현, m402 · c603 · 2026-10-10

`results_clarify_v11_prebonus`에서 Vertex로 잰 base가 v10 base(AI Studio)와 달랐다(m402 거절 뒤 7 → 9위, c603 4위 → 밖).
측정 시각도 달라서(13:34 vs 18:38) **시간 탓인지 호출 경로 탓인지** 가리지 못했다. #31(`GEMINI_RETRIEVAL_BACKEND`)이 머지된 뒤
검색(질의 분석·리랭커)만 AI Studio로 고정하고 v10 base와 같은 조건으로 2회 쟀다.

**결과: AI Studio는 약 11시간 뒤, 다른 API 키로도 v10 base를 정답 순위까지 그대로 재현했다(순위가 다른 행 0/52, 2회).
같은 입력의 Vertex 실행은 Top-10 52/52·순위 27행이 달랐다. 차이는 시간이 아니라 호출 경로다.
검색을 `GEMINI_RETRIEVAL_BACKEND=api_key`로 재면 기존 기준선(v32~v35·봉인 test·재질문 v09 — 모두 AI Studio)과 같은 조건이다.**

## 조건

| 항목 | 값 |
|---|---|
| 명령 | `run.sh` — `evaluate_clarification --split dev --query-ids m402,c603 --analysis-cache analysis_cache_v06_dev.json`, 2회 |
| 스위치 | `GEMINI_RETRIEVAL_BACKEND=api_key` · `GEMINI_RERANK_CORRECTIONS=all` · 입력 순서 `bonus` · 기본 type 문구 — v10 base와 같다 |
| 모델 | `gemini-3.1-flash-lite`(검색). 크롤링 모델(`GEMINI_CRAWL_MODEL_NAME`)은 측정과 무관 |
| runinfo `gemini` | `{backend: api_key, setting: api_key}` (2회 모두) |
| 리랭커 상태 | applied 60 · failed 0 (2회 모두). r2에서 호출 1회가 재시도(1/3)로 복구됐다 |
| 색인·사전 | 색인 3,010곡 · BM25 `8f0038a47cbc` · 가사 DB 3,010곡 `6997069a68d3da63` — v10·v11과 같다 |
| API 키 | v10을 잰 키는 크레딧 소진 뒤 백업으로 돌리고 **새 키**로 쟀다 |
| 시각 | r1 10/10 00:02~00:07 · r2 00:07~00:12 |

- 결과는 임시 폴더에 쓴 뒤 이 폴더로 옮겼다. runinfo `args.output_dir`만 옮긴 위치로 고쳤고 나머지 기록은 그대로다.
- 측정하는 동안 같은 작업 폴더에서 크롤러(`data/expansion/new_songs.csv`)가 돌았다. 측정은 Qdrant 색인과 원천 파일(`all_songs.jsonl`)의 정답곡 조회만 쓰고, 색인은 바뀌지 않았다(위 해시가 같다).
- 커밋되는 것은 이 문서·`run.sh`·`compare.py`·runinfo 2개다. `*.csv`·`*.log`는 레포 규칙상 로컬에만 있다.

## 결과 — v10 base_r1(10/9 13:34, AI Studio) 대비 (`compare.py`)

| 실행 | 경로 | Top-10이 다른 행 | 정답 순위가 다른 행 |
|---|---|---|---|
| v10 base_r2 (같은 날) | AI Studio | 0/52 | 0 |
| **v12 r1** (10/10 00:02) | AI Studio | **1/52** | **0** |
| **v12 r2** (10/10 00:07) | AI Studio | **2/52** | **0** |
| v11 base_rerun (10/9 18:38) | Vertex | 52/52 | 27 |

v12 r1 vs r2: Top-10 1/52, 순위 0.

Top-10이 달랐던 행은 둘 다 **정답이 Top-10 밖인 행**이라 지표는 같다.
- c603 `noisy:type`(틀린 답): 5·6위 순서와 10위 곡 (r1·r2 같은 모양)
- m402 `flow:skip`(r2만): 5·7~10위

## 정답 순위 (리랭킹 뒤, `-`는 Top-10 밖, flow는 찾은 턴)

| 질의 · 정책 | v10 base (AI Studio) | v12 r1 | v12 r2 | v11 base_rerun (Vertex) |
|---|---|---|---|---|
| m402 `reject_only` | 7 | 7 | 7 | 9 |
| m402 남성 | - | - | - | 8 |
| m402 발라드 | - | - | - | - |
| m402 솔로 | - | - | - | - |
| m402 2000년대 | 3 | 3 | 3 | 2 |
| m402 `flow:reject` | 7 t2 | 7 t2 | 7 t2 | 9 t2 |
| m402 `flow:oracle` | 1 t3 | 1 t3 | 1 t3 | 4 t3 |
| m402 `flow:noisy` | 10 t3 | 10 t3 | 10 t3 | 8 t3 |
| c603 `reject_only` | 4 | 4 | 4 | - |
| c603 남성 | 4 | 4 | 4 | - |
| c603 록/메탈 | 1 | 1 | 1 | 1 |
| c603 솔로 | - | - | - | - |
| c603 2010년대 | 6 | 6 | 6 | - |
| c603 `flow:reject` / `flow:oracle` / `flow:noisy` | 4 t2 | 4 t2 | 4 t2 | 1 t3 |

## 읽는 법

- **경로 차이로 확정한다.** 시간이 원인이면 11시간 뒤의 AI Studio도 달라야 하는데 순위까지 같았다. 키를 바꿔도 같았으므로 키가 아니라 호출 경로(AI Studio ↔ Vertex)가 출력을 가른다(새 키가 같은 프로젝트인지는 확인하지 않았다).
- **AI Studio도 완전히 결정적이지는 않다.** 같은 설정 2회가 Top-10 1/52 달랐다(v10 때는 0/52). 다만 정답 순위가 바뀐 행은 없었다. Vertex는 같은 설정 2회가 5/52 달랐다(v11).
- **v11의 Vertex 결과는 이제 서비스 경로의 결과가 아니다.** 검색이 AI Studio로 돌아왔으므로 서비스 경로의 근거는 v10과, v11 첫 측정(AI Studio, 15:28~15:47)의 (a) 숫자다. 그 파일은 덮어써졌고 숫자는 v11 RUN_INFO에만 있다.

## 다음

1. 측정은 `.env`에 `GEMINI_RETRIEVAL_BACKEND=api_key`를 두고, 결과를 읽기 전에 runinfo `gemini.backend`·`gemini.setting`과 `rerank_status_counts`(failed 0)를 본다.
2. v11 "다음"의 "Vertex에서 listwise 기본값이 유지되는지 dev 116으로 확인"은 필요 없다. 기준선은 그대로 쓴다.
3. (a)·①·②를 다시 볼 거라면 AI Studio에서 dev 57 전체를 구성마다 2회 잰다. 두 질의만으로는 정하지 않는다(v11 결론 그대로).
4. AI Studio 키의 결제 상태가 측정의 전제다. 크레딧이 떨어지면 402가 failed로 잡힌다 — 그 실행은 비교에서 뺀다.
