# results_v64 — 현재 코드의 서버 HTTP 경로 격리 검증(A/D, 모델 호출 0) (2026-10-17)

품질 실험이 아니다. 검색 정책·가중·후보 수·프롬프트·기본값·운영 `.env`·운영 색인·DB는 바꾸지 않는다.
v63 이후 남은 확인(서버 HTTP 경로·속성 보조 D의 현재 코드 비용·장애 시 반환·상태 누출)을 **실제 FastAPI 앱 + 로컬 Qdrant 서버**에서
모델 호출 없이 확인하고, 그 결과로 속성 보조 활성화 판단서(`ACTIVATION_DECISION.md`)와 서버 적재 운영안(`SERVER_LOADING_PLAN.md`)을 쓴다.
**§1~§4는 측정 전에 적었다**(§5 이후가 결과).

## 1. 시작 상태 (10/17 09:59~10:20 확인)

| 항목 | 값 |
|---|---|
| main | 원격·로컬 `dd5cc3e` = **#78(v63) 병합 커밋**(2026-10-17 09:56 KST). 그 앞 `f43daf7`(#77, v62) |
| v63 PR | **#78 병합됨**(head `8d2597a`, 스쿼시 `dd5cc3e`, `git diff 8d2597a dd5cc3e` 내용 차이 0). 자동 리뷰 1건: 필수·권장·참고 0. 열린 PR 없음 → 초안을 새로 만들지 않았다(§9) |
| 기준 커밋 | **`dd5cc3e`**. `src/` 변경 없음. v63 측정 당시 기준(`f43daf7`)은 v63 보고서 그대로 둔다 |
| 워크트리 · 브랜치 | `/private/tmp/vague-finder-service-readiness` · `experiment/service-readiness`(커밋 전). venv·artifacts(context·embeddings·map·timing·vector_db·audio 캐시·cover_stats)는 메인 체크아웃 심볼릭 링크, `map_data.json` 복사, **`.env` 없음**(검증 환경변수는 `start_servers.sh`만) |
| 다른 세션 | uvicorn·크롤러·평가 프로세스 없음(`ps`). v56·v62 Qdrant 컨테이너는 정지 상태 그대로(재사용·재설정 안 함) |
| 모델·외부 호출 예산 | **0회 · $0**. v63 예산($10·1,200회)은 승계하지 않는다 |

## 2. 검증 환경 (측정 전 고정)

- **Qdrant 서버**: Docker `qdrant/qdrant:v1.19.1` 새 컨테이너 `vague-finder-qdrant-v64`, `127.0.0.1:6364`(REST)·`6365`(gRPC),
  저장소 `artifacts/qdrant_server_v64/storage`(Git 제외). 로컬 PC 컨테이너다 — 배포 서버 성능이 아니다.
- **자료**: v63 워크트리의 색인 복사본(`qdrant_v63`)을 `artifacts/qdrant_v64_src`로 다시 복사해 `prepare_server.py`(v56 `migrate_to_server.py` 이식)로
  옮겼다. 서비스 색인(`artifacts/qdrant`)과 v63 복사본은 열지 않았다. 로컬 파일 모드 클라이언트는 그 스크립트 하나만 열었다.
  확인 결과(`server_collections.json`, `all_ok=true`): 5컬렉션 점 수·벡터 설정·payload 전량·벡터 표본 50점 일치, 곡 텍스트 내용 지문
  **`238299e6c8c17f2a`(원본=서버=v61~v63)**, Context 별칭 2개 원본과 같은 대상(`…__generation_f6424b94621a28cf`), Context manifest
  (`status=ok` · build `f6424b94621a28cf` · dense 5,054 · sparse 639) 건수·대상 일치, BM25 `bf606b19ab02`. payload 색인은 Context 두 컬렉션에만
  (서비스 적재기와 같은 필드), 곡 컬렉션에는 없음(적재기도 안 만든다).
- **가사**: v63 스냅샷(`/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot`, 읽기만) — 기동 때 파일 sha·문서 수·
  내용 지문 `e0d58f28f1f4a563`을 다시 계산해 다르면 서버가 뜨지 않는다. `LyricsExactSearchService._source_documents`만 바꾼다(판 수명·갱신·검색은 운영 코드).
  **스냅샷 적재 시간은 실제 Mongo cold 시간이 아니다.**
- **분석**: 저장 캐시 — v49 Vertex dev 3개(v06 57·v09 59·v10 60) + v55 Vertex test 42(이미 열린 세트, t014·t049 장애 재현에만). 캐시에 없으면 500.
- **리랭커**: 운영 `GeminiListwiseReranker` + 가짜 SDK(`tests/fake_gemini_client.py`, 모드 파일로 요청마다 전환). `normal`은 후보 **역순** 응답이라 품질이 아니다.
- **차단(`v64_runtime.py`)**: 루프백 밖 TCP 연결·이름 조회 → 예외 + 기록, Mongo 클라이언트 3곳·Gemini 클라이언트·ADC 읽기·Claude → 예외.
  기동 검사 `verify_vertex_setup`은 네트워크 없는 대체. 피드백 수집·색인 작업·검색 반영 0. `HF_HUB_OFFLINE=1`.
- **관측**: `QdrantIndex._count/_search/_payloads`와 가사 `search`를 `timing.step`으로 감싸 요청별 구간(`qdrant.count` 등)을 `SEARCH_TIMING_LOG`에 남긴다.
  값·순서는 바꾸지 않고 A·D 같은 조건이다.

### A / D 구성 (그 밖은 동일 — 검증 프로세스 환경에만 주입)

| 구성 | `ATTR_FILTER_WEIGHT` | `GATE` | `K` | `USE_GENRE` | `GENRE_POLICY` | 포트 |
|---|---|---|---|---|---|---|
| **A** 현행 OFF | 0 | era_plus | 30 | 1 | always | 8064 |
| **D** 활성화 후보 | 0.5 | era_plus | 30 | 1 | conditional | 8065 |

공통: `QDRANT_URL=http://127.0.0.1:6364` · `CONTEXT_WEIGHT=0.5` · `SEARCH_REFERENCE_YEAR=2026` · `RERANKER_BACKEND=gemini_listwise` · 리랭커 설정은 코드 기본값
(요청 제한 20초 · 공유 예산 30초 · 재시도 3 · 1패스 · Search 끔 · 희소 검증 끔) · 가사 TTL 기본 300초 · 예열 켬 · 요청 본문은 화면과 같게
`top_k 15 · candidate_k 30 · explain true · defer_clarify true`. `MONGO_URI`·인증 파일 unset, `GCP_PROJECT_ID`는 더미.

## 3. 검증 항목·표본·성공 기준 (측정 전 고정)

성공 기준은 "계약대로 동작했는가"다. 지연에는 합격선을 두지 않는다(결과를 본 뒤 만들지 않는다). 실패·예외 요청을 집계에서 빼지 않는다.

| # | 항목 | 표본 / 방식 | 성공 기준(계약) |
|---|---|---|---|
| S | 기동·준비 | A·D 별도 프로세스 | `/health.context=verified`, 예열 `ready`, 가사 판 상태 기록, 외부 연결 시도 0 · 차단된 클라이언트 생성 0. cold 구간(프로세스 시작→import→ready, 단계별)·첫 요청을 분리 기록 |
| S2 | 캐시 누락·실제 기동 검사 | 캐시 밖 질의 1건(HTTP) + `probe_guards.py`(별도 프로세스: 실제 `QueryAnalyzer`·실제 기동 검사 함수·스냅샷 없는 가사 서비스) | 외부 연결 시도 0. 캐시 누락은 500(조용한 모델 호출·폴백 없음). 실제 분석기·검사 함수는 차단 예외로 끝나거나 규칙 폴백 — 어느 쪽이든 기록 |
| 1 | 가사 통합 | dev 가사 단서 19질의(v63 목록) × A·D, `use_rerank=true` · 가짜 normal | HTTP 200 · 가사 검색 실행(호출 기록) · 예외 0 · 일치 0건(v63: n009·t016·t060)은 `failed_paths` 없이 정상 · 보호 그룹 질의와 그룹 수 기록 · 응답 필드(`lyric_match_type`·`lyric_match_detail`·곡별 `rerank_group_status`) 존재 |
| 1b | 평가기↔실제 HTTP | dev 176 × A의 `candidate_ids`(리랭킹 전 30개) ↔ v63 offline B 후보(같은 색인·가사·캐시·OFF) | 같은 조건이므로 **176/176 ID·순서 동일**이 기대값. 다르면 원인(서버 모드·top_k·라우트 차이)을 기록 |
| 2 | 일반 검색 | dev 176 × A·D, `use_rerank=true` · 가짜 normal + 합성 0곡 10건 | 전부 200 · 결과 15 · 중복 0 · 결과 ⊂ 후보 · 후보 수 기록 · 정책 생략(v63: image_dominant 21)은 모델 호출 0 · D의 `path.attr_filter` 구간은 사전 분류의 열림과 일치 · A에는 구간 없음 · 닫힌 질의는 A=D 후보 동일 |
| 3 | 지연 | v56 선정 규칙을 현재 코드로 재실행(`classify_queries.py`: D·1턴 기준 closed/open_hits/open_empty, 가사 단서 질의 제외, 범주별 v06→v09→v10 번갈아 10건, 합성 0곡 10건) → A/D 교대 순차 3회, 리랭킹 OFF, 준비 요청 1건 제외 | 수치만 보고한다: 범주별 HTTP 전체(클라이언트·미들웨어)·검색 단계·`qdrant.count`(필터/무필터)·`path.attr_filter`의 요청 단위 중앙값/p95와, 질의별 D−A(3회 중앙값 차이) 분포의 중앙값/p95/최대를 **구분**. v56 +37ms·v62 +34ms와 단순 합산하지 않는다 |
| 4 | 가짜 SDK 장애 | t014·t049(단일 묶음) / n007·t018(보호 그룹) / q117(정책 생략) × A·D, 모드 아래 표 | 아래 계약 표 |
| 5 | 기존 손실·2턴·누출 | t014·t049 폴백 시 정답 순위 A/D, q112(성별 "남성")·n011(장르 "댄스") 2턴(1턴 1위 거절), 모든 사례의 마지막 normal | t014 폴백 = 리랭킹 직전 후보 순서(v56: A 1위·D 10위). 2턴: 거절 곡 결과 없음 · 답 반영 · 실패 시 답·거절 반영 후보 순서. 장애 뒤 normal = 첫 normal(top-15·단계·호출 수) |

### 3-1. 장애 모드 순서와 계약 (현재 코드 v57·v59 기준으로 기대하는 동작)

| 사례 | 모드 순서 |
|---|---|
| t014 · t049 | normal → exception → invalid_json → foreign_ids → empty_ranking → wrong_types → invalid_then_normal → partial_ids → slow → slow_within(8초) → normal |
| n007 · t018 | normal → exception → fail_group_first → invalid_then_normal → slow → normal |
| q117 | exception → normal |
| q112 · n011(2턴) | 1턴 normal → 2턴 normal → exception → slow → normal |

| 모드 | 기대 계약 |
|---|---|
| exception · invalid_json · foreign_ids · empty_ranking · wrong_types | 공유 예산 안에서 재시도 후 `failed` → 단계 `model_failed`, 반환 = 리랭킹 직전 후보 순서(보호 배치 포함), 결과 15 · 중복 0 · 후보 안. 실제 SDK 호출 수(`rerank_model_calls`) = 가짜 호출 기록 수 |
| invalid_then_normal | 첫 응답 무효 → 재시도 응답(역순) 적용 → `model_applied` |
| partial_ids | 일부 누락·중복·후보 밖 혼합이지만 유효 id 있음 → 적용(누락분 보충) |
| fail_group_first | 첫 묶음만 실패 → `model_partial` · `rerank_groups_failed=1` · `reorder_stage_note` 있음 |
| slow | 요청의 리랭킹 단계 전체가 30초 마감 하나를 공유 → 뒤 묶음은 `budget_exhausted`(모델 미호출). 리랭킹 구간과 HTTP 전체 시간을 따로 기록(30초라고 가정하지 않는다) |
| slow_within | 8초 뒤 정상 → 적용 |
| 정책 생략(q117) | 모드와 무관하게 가짜 호출 0 · `policy_skipped` |

## 4. 하지 않는 것

새 실제 모델 실험·Vertex/Claude/AI Studio 호출·원격 Mongo 접근·피드백 쓰기·멜론 요청·운영 `.env`/색인/DB 변경·test 재튜닝·
동시 적재 구현. 가짜 모델 결과로 Hit/MRR 개선을 주장하지 않는다. 로컬 Docker 수치를 배포 서버 성능이라 부르지 않는다.

---

## 5. 결과 (측정 10:08~10:40)

실행 순서(같은 두 서버 프로세스, 재시작 없음): 기동 10:08:19 → 분류 → dev 176 sweep(374요청, 56초) → 지연(182요청) → 장애(§5-4) · 별도 프로세스 진단(§5-2)·`probe_guards`.

### 5-1. 기동·차단·캐시 누락 (S·S2) — 계약대로

| 확인 | A | D | 근거 |
|---|---|---|---|
| 기동 검사 | `/health.context=verified`(별칭 2개·manifest·artifact 해시) | 같음 | `startup/health_*.json` |
| 예열 | `ready` 8.96초(KoE5 4.11 · BM25 0.96 · SigLIP2 2.25 · CLAP 0.67 · 가사 0.54(스냅샷) · 벡터 DB 0.43, 리랭커 예열 생략) | 8.95초 | 같음 |
| cold | 프로세스 시작 → 앱 import 3.7초 → 기동 완료 → 예열 9초. 첫 검색(sweep 첫 요청) 서버 1,371ms 중 dense 전수 조회 788ms(Qdrant 서버 cold) — 이후 같은 조회 수십 ms | 같음 | timing 첫 줄 |
| 가사 판 | 5,143문서 · 내용 지문 `e0d58f28f1f4a563` · 파일 sha `295b7a508c12cc34` · TTL 300초 뒤 **백그라운드 갱신 1회 성공**(스냅샷 재읽기, 요청은 기다리지 않음) | 같음 | `/v64/status` |
| 외부 연결 시도 · 차단된 클라이언트 생성 | **0 · 0**(sweep·지연·장애 뒤 모두) | **0 · 0** | `sweep_status_after.json` · `fallback_status_after.json` |
| 캐시 밖 질의 | **HTTP 500**(`분석 캐시에 없는 질의 — 모델 호출 없이 중단`), 가짜 호출 0 · 외부 연결 0 | 같음 | `sweep_summary.json.cache_miss` |

`probe_guards.py`(별도 프로세스, 대체 없이 실물 경로 — `probe_guards.json`): 외부 연결 시도 **0**.
- 실제 기동 검사 `_verify_vertex_setup` → ADC 읽기에서 차단 예외(countTokens 전).
- 실제 `QueryAnalyzer`(동기·비동기) → 클라이언트 생성 차단이 재시도마다 3회씩, 약 2초 뒤 **규칙 폴백 분석을 반환**(예외 아님).
  운영에서 분석 클라이언트를 만들 수 없을 때 검색은 규칙 분석으로 계속된다는 기존 계약 그대로다.
- 스냅샷을 꽂지 않은 가사 서비스 → 가사 전용 Mongo 클라이언트 생성 차단 → 판 `empty`.
- 대체 없는 lifespan → 차단 예외는 설정 오류(`GeminiConfigError`)가 아니어서 **경고 후 기동**. 연결 장애와 같은 분기다.
  운영에서 실제 ADC가 없으면 설정 오류로 멈추는 경로는 단위 시험이 덮는다(이번 미검증).

### 5-2. dev 176 실제 HTTP (항목 1·1b·2) — 계약대로, 단 **로컬↔서버 후보 차이 발견**

`sweep_summary.json` · `sweep_per_query.csv`. A·D 각 176 + 합성 0곡 10 + 캐시 밖 1.

- **형태**: 200 = 372/372(캐시 밖 2건만 500) · 결과 15 · 후보 30 · 중복 0 · 결과 ⊄ 후보 0 · `failed_paths` 0 · 분석 폴백 0 · 라우터 예외 0.
- **상태**: A·D 모두 `model_applied` 155 · `policy_skipped(image_dominant)` 21(v63과 같은 수), 정책 생략 요청의 가짜 호출 0.
  `rerank_model_calls`가 있는 330요청에서 **가짜 클라이언트 실제 호출 수와 330/330 일치**(정책 생략 42건은 null — 세지 않은 것이 아니라 호출 없음).
- **가사 19질의(항목 1)**: A·D 모두 19/19 가사 검색 실행, 예외 0, `failed_paths` 0.
  - 일치 0건 = **n009·t016·t060**(v63과 같음) — 예외가 아니라 정상 0건으로 응답했다.
  - 질의별 가사 결과 수가 v63과 19/19 같다.
  - 응답의 `lyric_match_type`이 16질의에 있었고, `lyric_match_detail`은 phonetic·fuzzy 일치 곡에만 붙는다(exact는 detail 없음 — 스키마 설명대로).
    곡별 `rerank_group_status`와 보호 순서 규칙(`lyrics_exact_priority`·`gemini_lyric_anchor_protect`)이 응답에 실렸다.
  - 보호 묶음이 2개 이상인 질의: q211·n007·t003·t018·t025(A=D).
  - 묶음 수는 v63 평가기와 18/19 같고, **n064만 HTTP 1묶음 · 평가기 3묶음**이다. 원인: 라우트는 `top_k=15`라 첫 tier(15곡)가 결과 자리를 다 채우면 남은 tier를 부르지 않는다.
    평가기는 `top_k=30`으로 부른다. 설계대로의 차이이고, 평가기의 호출 수가 서비스보다 많게 잡힌다는 뜻이다.
- **보조 경로(항목 2)**: 사전 분류 open 40 · closed 136(dev, 가사 포함). D의 `path.attr_filter` 구간은 **open 40건과 정확히 일치**(불일치 0), A에는 0건.
  닫힌 136질의는 A=D 후보 동일, 열린 40질의는 40건 모두 다름. 합성 0곡 10건: D만 경로가 열리고(필터 count 실행) 후보 30 유지.
- **항목 1b — 평가기(v63 offline B)와 실제 HTTP A의 후보 30: 176건 중 142건만 같다.** 진단(`diagnose_candidates.py`·`diagnose_paths.py`, 모델·Mongo 0):

  | 비교(같은 코드·분석·가사 스냅샷) | 일치 |
  |---|---|
  | 서버 모드: 평가기식 호출(top_k 30·리랭킹 끔) = 라우트식 호출(top_k 15·`candidate_ids_out`) = 실제 HTTP | 176/176 |
  | 로컬 파일 모드: 평가기식 = 라우트식 = **v63 offline B** | 176/176 |
  | 서버 모드 ↔ 로컬 모드 | **142/176**(34건 순서, 그중 9건은 집합도 다름) |
  | 같은 모드 안 반복 | 176/176 |

  → **라우트·평가기 차이가 아니라 Qdrant 백엔드(로컬 파일 ↔ 서버) 차이다.** 경로별로 보면 이미지 경로 32건 · Context 경로 3건에서만 순위가 갈린다.
  갈린 곡은 모두 **점수가 정확히 같은 곡**이다. 이미지 벡터는 5,143곡 중 **1,395곡(495묶음)이 다른 곡과 바이트까지 같은 벡터**를 공유한다(같은 앨범 표지, 예: 85257·85258, 601703161~167).
  동점끼리의 반환 순서가 로컬과 서버에서 다르고, 라우터는 동점을 따로 정렬하지 않는다.
  - 영향(리랭킹 전, 엄격 정답, `tie_impact.json`): 후보 회수@30 122 = 122 · Hit@10 102 = 102 · **Hit@1 41 → 40**.
    n066 정답 85257이 쌍둥이 85258과 자리를 바꿔 1 → 2위, t107은 19 → 20위.
  - 리랭킹 뒤 품질 영향은 재지 않았다(모델 호출 0 범위).
  - v56의 "로컬↔서버 42/42 동일"은 이미지 동점 질의가 적은 21질의 표본이었다. 그 범위에서는 이번과 모순되지 않는다.
  - **함의**: 지금까지의 dev 기준선(로컬 파일 모드)은 서버 모드 배포의 후보를 그대로 재현하지 않는다. 결함·수정 범위는 §7-①.

### 5-3. 지연 — 현재 코드, 서버 Qdrant, 리랭킹 끔 (항목 3)

`latency_summary.json` · `latency_joined.csv`. 182요청(준비 2 + 측정 180), 200 = 182/182, 구간 기록 미대조 0, 후보 지문 반복 안정 60/60.
표본은 v56과 같은 30질의다(규칙 재실행 결과 동일 — `latency_sample.json`). 준비 요청 c601: A 168ms · D 161ms(집계 제외 — sweep 뒤라 프로세스는 이미 따뜻하다).

| 범주 | 구성 | HTTP(클라이언트) 중앙값 / p95 / 최대 | 검색 단계 중앙값 / p95 | 무필터 count 합 중앙값 | 필터 count 합 중앙값 / p95 | `path.attr_filter` 중앙값 / p95 | `path.text` 중앙값 |
|---|---|---|---|---|---|---|---|
| closed (10) | A | 60.8 / 113.8 / 268.9 | 56.2 / 108.8 | 2.4 | 0 | — | 55.5 |
|  | D | 60.2 / 197.1 / 289.6 | 55.7 / 192.7 | 2.4 | 0 | — | 55.0 |
| open_hits (10) | A | 127.9 / 291.3 / 365.2 | 120.7 / 284.1 | 4.7 | 0 / 43.1(다른 경로 필터 9요청) | — | 113.6 |
|  | D | 217.9 / 302.6 / 313.9 | 210.2 / 296.5 | 4.9 | **96.4 / 153.7** | 200.7 / 291.5 | 124.0 |
| 합성 0곡 (10) | A | 123.6 / 284.5 / 310.9 | 116.4 / 276.8 | 4.4 | 0 / 40.1 | — | 110.7 |
|  | D | 218.2 / 364.6 / 429.0 | 210.9 / 357.4 | 4.8 | **144.2 / 224.5** | 202.2 / 292.9 | 121.3 |

D − A 분포(ms)는 세 가지를 구분해서 적는다. 질의 단위는 질의별 A 3회 중앙값과 D 3회 중앙값의 차이 10개, 짝 단위는 같은 회차 A·D 요청 차이 30개다.

| 범주 | 질의 단위 중앙값 / p95 / 최대 (최소) | 짝 단위 중앙값 / p95 |
|---|---|---|
| closed | −0.6 / 0.4 / 0.8 (−3.8) | −0.9 / 92.6 |
| open_hits | **+67.0 / 123.9 / 131.5 (+33.6)** | +65.3 / 199.1 |
| 합성 0곡 | **+57.1 / 139.9 / 140.8 (+24.4)** | +56.5 / 212.8 |

- 닫힌 질의는 차이가 없다(구간 자체가 없다). 짝 단위 p95 92.6은 양쪽 공통의 산발 지연이 한쪽에만 걸린 짝 때문이고, 질의 단위 최대는 +0.8이다.
- **열린 질의의 D 추가 지연은 v56(+37ms, 옛 고정 상한 코드)보다 크다.** 늘어난 몫의 대부분은 v62가 추가한 보조 경로의 **필터 count**다
  (D 요청당 필터 count 합 중앙값 96ms, 0곡이면 144ms — payload 색인이 없어 발매 연도 `MatchText` 등 조건을 전 점에 대해 센다).
  보조 경로 자체(200ms)는 텍스트 경로(110~124ms)와 나란히 돌고, 벽시계에 더해지는 것은 그 차이와 경합분이다.
  v56 +37ms와 v62 필터 count +34ms를 더한 값(71)과 우연히 비슷하지만, **이 값은 합산이 아니라 현재 코드 전체 경로의 실측**이다.
- 0곡이어도 같은 비용(필터 count는 결과를 모르는 채 돈다, 오히려 count가 더 길다).
- 리랭킹을 켠 전체 요청 지연은 이번에 재지 않았다(가짜 모델). 참고로 v63 실제 Vertex 리랭킹 단계 중앙값은 4.5~4.9초였다(로컬 파일 모드).
- 로컬 PC의 Docker Qdrant(같은 기계)다. 배포 서버의 네트워크·하드웨어 성능이 아니다. A·D는 같은 Qdrant 서버를 공유하고 번갈아 순차로 보냈다.

### 5-4. 가짜 SDK 장애 — 현재 코드(v57 공유 마감 · v59 상태) (항목 4)

`fallback_summary.json` · `fallback_table.csv`. 92요청(A 46 · D 46), 전부 HTTP 200, **사전 계약 위반 0**(`analyze_fallback.py` 판정, 자체 검사 `selftest_analyzers.py` OK).
결과 15 · 중복 0 · 후보 안 · `rerank_model_calls` = 가짜 호출 수가 92/92다. 범위는 A·D를 합친 최소~최대다.

| 모드 | 사례 | HTTP 전체(ms) | 리랭킹 단계(ms) | 단계 · 묶음(전체/실패/예산 소진) | 실제 SDK 호출 | 반환 |
|---|---|---|---|---|---|---|
| exception · invalid_json · foreign_ids · empty_ranking · wrong_types | t014 · t049 | 2,177~2,380 | 2,013~2,030 | `model_failed` · 1/1/0 | **3**(무효 응답도 재시도 대상) | 후보 순서 앞 15 그대로(16/16) |
| invalid_then_normal | t014 · t049 | 1,199~1,351 | 1,010~1,015 | `model_applied` · 1/0/0 | 2 | 재시도 응답(모델 순서) |
| partial_ids | t014 · t049 | 159~328 | 1.5~1.8 | `model_applied` | 1 | 누락분 보충 |
| slow(무응답) | t014 · t049 · q112 · n011 | **30,126~30,309** | **30,003~30,005** | `model_failed` · 1/1/0 | 2(20초 취소 + 남은 약 9초 취소) | 후보 순서 그대로 |
| slow_within(8초) | t014 · t049 | 8,193~8,431 | 8,004~8,007 | `model_applied` | 1 | 모델 순서 |
| exception | n007(3묶음) · t018(2묶음) | 4,157~6,192 | 4,042~6,071 | `model_failed` · 3/3/0 · 2/2/0 | 9 · 6 | 보호 묶음 앞 + 나머지 후보 순서(보호 밖 곡은 후보 순서 유지) |
| fail_group_first | n007 · t018 | 2,177~2,267 | 2,021~2,027 | **`model_partial`** · 3/1/0 · 2/1/0, `reorder_stage_note` 있음 | 5 · 4 | 실패 묶음 입력 순서 + 나머지 모델 순서 |
| invalid_then_normal | n007 · t018 | 1,204~1,426 | 1,007~1,021 | `model_applied` | 4 · 3 | — |
| **slow** | **n007 · t018** | **30,169~30,247** | **30,002~30,006** | `model_failed` · **3/1/2 · 2/1/1** | **2**(뒤 묶음은 모델을 부르지 않음) | 후보 순서(보호 배치) |
| 정책 생략 | q117 | 69~570 | — | `policy_skipped` | **0**(exception 모드에서도) | 후보 순서 |

- **공유 마감 확인**: 보호 묶음이 여러 개인 n007·t018도 리랭킹 단계가 **30.0초 하나**로 끝났다. v56에서는 93.3초·62.2초였다.
  첫 묶음이 예산을 다 쓰면 남은 묶음은 `budget_exhausted`로 호출 없이 입력 순서로 나간다.
- **HTTP 전체 시간은 30초가 아니다**: 리랭킹 30.0초 + 분석·검색·설명 직렬화 약 0.12~0.3초 = 30.1~30.3초. 화면 제한 60초 안이다.
  다만 분석 단계가 실제 모델일 때 최대 22초가 더해질 수 있다(이번엔 캐시라 1ms대) — 최악 약 52초는 계산값이며 미측정이다.
- 무효 응답(후보 밖 id만·빈 순위·자료형 오류·JSON 아님)은 성공으로 표시되지 않고 재시도 3회 뒤 `failed`다(v56의 `foreign_ids → applied` 결함이 v59로 고쳐진 상태 확인).

### 5-5. 기존 손실·2턴·상태 누출 (항목 5)

- **t014 폴백**(exception·slow, 장애 계약 재현 — 품질 평가 아님): 정답 순위 **A 1위 · D 10위**. 후보 순위 A 1 · D 10과 같고, v56 실측과 같다.
  t049는 후보 A 16 · D 22위라 폴백 시 양쪽 모두 15위 밖이다(v56과 같음). 정상 모드 순위는 가짜 역순 응답이라 의미가 없다.
- **2턴(v56 절차: 1턴 결과 1위 거절 + 답)**: 거절 곡은 4흐름 모두 결과·후보에 없고(거절 곡은 정답이 아니었다), 2턴 후보는 1턴과 달라졌다(답 반영).
  실패 시 반환 = 답 가산·거절 반영 후보 순서.
  - n011(장르 "댄스", 맞는 답): **D에서 정답이 1턴 후보 밖 → 2턴 후보 10위**(답이 필터 조건에 들어감, v54와 같은 동작). A는 2턴에도 후보 밖.
  - **q112(성별 "남성", 맞는 답 — 정답 곡은 남성): D에서 정답이 1턴 후보 2위 → 2턴 후보 밖.** 모델 없이 필터를 다시 만들어 원인을 확인했다.
    1턴 필터는 `no_gender`라 최초 장르(랩/힙합)를 유지한다(시기 2004~2009 · 랩/힙합 · 그룹). 성별 답이 들어오면 `gender_condition`이 되어 장르를 뺀다(시기 · 남성/혼성 · 그룹).
    필터가 넓어지면서 정답이 보조 상위 30에서 빠졌다. A는 1·2턴 모두 정답이 후보 밖이다(보조 경로가 없다).
    v54의 2턴 평가에서는 q112가 1턴에 발견돼 질문 단계가 없었으므로 이 흐름이 측정되지 않았다(v54 §4).
    **conditional 정책의 설계대로의 동작이 만드는 2턴 손실 사례**다(§7-②).
- **상태 누출 없음**: 장애 모드를 거친 뒤 마지막 normal 요청이 첫 normal과 결과 15·후보 30·단계·묶음 수·SDK 호출 수까지 같다(12/12 사례 × 구성).
  공유 마감은 요청마다 새로 만들어지고 singleton 리랭커에 남지 않는다.

## 6. 판단

| 질문 | 답 |
|---|---|
| 실제 HTTP 경로(라우트·직렬화)가 평가기와 같은 후보를 만드는가 | **같은 Qdrant 백엔드에서는 176/176 같다.** 다른 것은 백엔드다. 로컬 파일 ↔ 서버 모드에서 이미지·Context 동점의 순서가 달라 후보가 34/176 다르다(리랭킹 전 정답 포함 122 = 122, Hit@1 41 → 40) |
| 가사 경로가 서버 HTTP에서 계약대로 도는가 | 예 — 19/19 실행, 예외 0, 정상 0건 3 구분, 보호 묶음·순서 규칙·응답 필드 정상 |
| 속성 보조 D가 현재 코드에서 경로를 맞게 여닫는가 | 예 — 사전 분류 open 40과 구간 40/40 일치, 닫힌 질의 A=D 136/136 |
| D의 추가 지연(현재 코드·로컬 Docker) | 닫힌 질의 0. 열린 질의는 질의 단위 D−A 중앙값 **+67ms**(p95 +124, 최대 +132), 합성 0곡 +57ms(p95 +140). 추가분의 대부분은 필터 count(요청당 중앙값 96~144ms). 합격선은 두지 않았다 |
| 장애 시 반환 | 리랭킹 직전 후보 순서(가산·거절·보호 배치 반영). 무효 응답도 실패로 표시. 여러 묶음도 리랭킹 30초 하나. **D 고유 위험은 폴백 시 보조 후보에 밀린 정답이 그대로 나가는 것(t014 1 → 10)** |
| 상태 누출 | 없음 |

- **속성 보조 활성화: 증거 부족 → 지금은 OFF 유지**(`ACTIVATION_DECISION.md`). 운영 계약에는 막는 결함이 없다. 다만 품질 이득은 정책을 고른 dev 안의 결과(117 → 125)뿐이다.
  독립 test에서는 0이었고, D에만 있는 2턴 손실 경로(§5-5 q112)를 새로 확인했다.
- **서버 적재: 초기 배포는 검색 중단 절차를 권고**(`SERVER_LOADING_PLAN.md`). BM25·오디오 중심값이 기동 때 고정되므로 적재 뒤 어차피 재기동해야 한다. 무중단이 필요하면 불변 세대 발행을 별도로 구현한다.
- 검증 서버 2개는 종료했고, 전용 컨테이너 `vague-finder-qdrant-v64`는 정지 상태로 보존했다(10:39). 외부 모델 호출은 **0회 · $0**이다.

## 7. 발견한 결함·관찰 (이번에 고치지 않았다 — 재현·영향·수정 범위만)

| # | 내용 | 재현 | 영향 | 수정 범위 제안 |
|---|---|---|---|---|
| ① | **로컬 파일 ↔ 서버 Qdrant의 동점 순서 차이로 후보가 달라진다.** 이미지 벡터는 1,395곡(495묶음)이 다른 곡과 완전히 같다(같은 표지). 라우터·`QdrantIndex`는 동점을 따로 정렬하지 않고 백엔드 반환 순서를 쓴다. Context 경로에도 동점이 있다(3건) | `diagnose_candidates.py local|server` → 서버↔로컬 142/176, 각 모드 안에서는 176/176 · `diagnose_paths.py`(이미지 32 · Context 3) · 이미지 벡터 중복 집계 | dev 176: 후보 순서 34건 · 집합 9건, 리랭킹 전 Hit@1 41 → 40(n066 정답이 쌍둥이와 자리 바꿈). **로컬 기준선이 서버 배포를 그대로 재현하지 않는다.** 재적재·서버 내부 정리 뒤에도 순서가 바뀔 수 있다(미확인). A·D 공통 | 경로 결과 정렬에 결정적 동점 기준(점수 내림차순 → song_id)을 넣는다(`QdrantIndex._search`/하이브리드 합산 정렬, Context 집계). 기준선 후보가 바뀌므로 로컬·서버 양쪽 재측정과 함께 한다. 표지가 같은 곡을 하나로 묶을지는 별도 판단 |
| ② | **D(conditional)에서 맞는 성별 답이 1턴 후보의 정답을 지울 수 있다.** 1턴에 성별이 없으면 장르를 유지하다가, 2턴 성별 답이 오면 최초 장르를 뺀다 → 필터가 넓어져 정답이 보조 상위 30에서 빠진다 | q112: D 1턴 후보 2위 → 2턴("남성", 정답 곡 남성) 후보 밖. 필터 재구성으로 원인 확인(§5-5) | dev에서 이 흐름을 탈 수 있는 질의는 v54 분류상 `no_gender`이면서 뺄 장르가 있던 q112·t053 2건(+ 2턴 성별 답). v54 2턴 평가(18건)는 q112에 질문이 없어 측정하지 못했다. D에만 있다 | 정책 결정 사항: "1턴에 장르로 연 필터는 성별 답이 와도 장르를 유지"(성별은 추가 조건) 변형을 별도 A/B로. **이번 test·dev로 재튜닝하지 않는다** — 새 독립 자료 필요 |
| ③ | 평가기와 서비스의 리랭킹 묶음 수 차이(설계대로): 서비스(top_k 15)는 보호 tier가 결과 자리를 다 채우면 남은 tier를 부르지 않는다. 평가기(top_k 30)는 다 부른다 | n064: HTTP 1묶음 · 평가기 3묶음 | 결과 상위 15는 같은 tier에서 나온다. 평가기 기반 호출 수·비용 추정이 서비스보다 많다(dev 176에서 1건) | 결함 아님. 비용 추정에 적어 둔다 |
| ④ | (v56 ④ 그대로) 서비스 `explain` 응답에 `attr_filter_genre`(D의 장르 유지/제외 사유)가 없다 | `SearchExplainOut` 필드 목록 | 켠 뒤 ②와 같은 판정을 응답으로 볼 수 없다(서버 로그·평가기만) | `SearchExplainOut`에 고정 필드 추가(관측성) |
| ⑤ | `QdrantIndex`에 `fetch`가 두 번 정의돼 뒤 정의만 쓰인다(`qdrant_backend.py` 229행·335행) | 코드 읽기 | 동작 영향 없음(앞 정의는 죽은 코드) | 정리 |

## 8. 확인하지 못한 것

1. 리랭킹을 켠 실제 모델의 품질·지연·실패율(모델 호출 0 범위). D의 폴백 손실은 실패율과 곱해야 한다. v63 실제 Vertex 655회 호출에서 실패는 0이었고, v52(504)·v54(499)에서는 일시 장애가 있었다.
2. 동점 순서 차이가 **리랭킹 뒤** 최종 품질에 주는 영향. 서버 모드 기준선 재측정이 필요하다(모델 호출 필요, 별도 예산).
3. 동시 요청·동시 적재. 두 서버는 번갈아 순차로만 받았다. 적재 중 검색의 정확성은 여전히 보장되지 않는다(`SERVER_LOADING_PLAN.md`).
4. 배포 서버 하드웨어·네트워크. 로컬 Docker, 같은 기계 loopback이다.
5. 실제 Mongo 가사의 cold 적재 시간. 스냅샷 재생은 0.54초였고, 실제 Mongo는 v56 실측 약 350초였다.
6. 실제 ADC가 없거나 만료됐을 때의 기동 실패 경로(설정 오류 분기). 이번 차단은 다른 예외라 경고 분기만 탔고, 그 경로는 기존 단위 시험이 덮는다.
7. 분석 단계가 실제 모델일 때 장애 요청의 HTTP 전체 상한(분석 22초 + 리랭킹 30초).
8. 자연 `open_empty` 질의(dev 0건, 합성으로 대신). test는 장애 계약 재현(t014·t049)에만 썼다.

## 9. v63 PR(#78) 상태와 검토 기록

- **병합됨**: 2026-10-17 09:56 KST, head `8d2597a` → 스쿼시 `dd5cc3e`(내용 차이 0). 자동 리뷰 1건(필수·권장·참고 0). 열린 PR 없음 → 새 초안은 만들지 않았다.
- **추적 파일**: 131개, 전부 `experiments/reranking/results_v63_hybrid_lyrics_validation/` 아래. `src/`·`tests/` 변경 0. 실험의 A 구현 교체(`fc20e86`의 `QdrantIndex.query`)는 실험 프로세스 안에서만 했다(`v63_runtime.install_arm`).
- **민감정보 검사**(값은 출력하지 않고 개수만 셈):
  - 걸린 것: Mongo URI 형식 1건(`v63_runtime.py`의 차단 확인용 `127.0.0.1:1` 가짜 주소).
  - 0건: 개인 키 · 서비스 계정 이메일 · API 키(Google·Anthropic) · OAuth 토큰 · refresh_token · 전체 프로젝트 ID · 개인 이메일 · `full_lyrics`/`lyrics_data` 값 · ADC 파일 경로.
  - 문서·코드 밖 파일에서 한글 150자 이상인 줄 0.
  - 추적 목록에 `*.jsonl`·explain·detail·lyric_boosts·`_legacy/`·`logs/`·`budget/`·스냅샷이 없다(`.gitignore`대로).
- **숫자 대조**: RUN_INFO의 후보 122 · 81/107/117 · MRR 0.5318(가사 19: 16 · 12/15/16 · 0.7023), 하이브리드 311/311(A 4건 불일치), 시도 668(generate 655 + countTokens 13) · 실패 0 · $3.075062, 토큰 6,879,563/903,447,
  가사 검색 19 · 0건 3 · 보호 13, 질의별 순위 차이 0이 `live_analysis/summary.json`·`budget_summary.json`·`lyrics_evidence.json`·`*/compare/summary.json`과 **모두 일치**.
- **6,572 통과 · 130 skipped · 실패 1(기존 macOS `/tmp` 경로 시험)은 v63 당시(`f43daf7`)의 과거 실행 결과**다. 이번에 전체 시험과 live A/B를 다시 돌리지 않았다.
- **남은 검토**:
  - v63 RUN_INFO 머리말의 "커밋 전"은 병합 뒤에는 오래된 설명이다. 과거 측정 기록이라 고치지 않았다.
  - v63의 재현에는 Git 제외 원기록·스냅샷(v63 워크트리 로컬 경로)이 필요하다.
  - **v63 결론("품질 재측정 종결")은 로컬 파일 모드 기준이다.** 서버 모드에서는 §7-① 때문에 후보가 34/176 다르므로, 서버 배포 품질은 별도로 확인해야 한다.

## 10. 재현 (워크트리 루트, 모델·Mongo 호출 없음)

```sh
R=experiments/reranking/results_v64_service_readiness
cp -R /private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/qdrant_v63 artifacts/qdrant_v64_src   # 색인 복사본(지문 238299e6c8c17f2a)
docker run -d --name vague-finder-qdrant-v64 -p 127.0.0.1:6364:6333 -p 127.0.0.1:6365:6334 \
  -v "$PWD/artifacts/qdrant_server_v64/storage:/qdrant/storage" qdrant/qdrant:v1.19.1        # 보존된 컨테이너면 docker start
venv/bin/python $R/prepare_server.py                  # 이전 + 정합성 → server_collections.json
zsh $R/start_servers.sh both                          # A 8064 · D 8065 (V64_LYRICS_SNAPSHOT_DIR로 스냅샷 위치 지정 가능)
venv/bin/python $R/classify_queries.py                # 표본 → latency_sample.json · query_categories.csv
venv/bin/python $R/run_sweep.py && venv/bin/python $R/analyze_sweep.py
venv/bin/python $R/run_latency.py && venv/bin/python $R/aggregate_latency.py
venv/bin/python $R/run_fallback.py && venv/bin/python $R/analyze_fallback.py
venv/bin/python $R/probe_guards.py                    # 별도 프로세스(서버와 무관)
venv/bin/python $R/diagnose_candidates.py server && venv/bin/python $R/diagnose_candidates.py local   # 로컬은 qdrant_v64_src를 연다(다른 프로세스가 열지 않을 때)
venv/bin/python $R/diagnose_paths.py server <qid,...> && venv/bin/python $R/diagnose_paths.py local <qid,...>
venv/bin/python $R/tie_impact.py && venv/bin/python $R/selftest_analyzers.py
```
출력 파일이 이미 있으면 실행 도구는 덮어쓰지 않고 멈춘다(`sweep_requests.jsonl`·`latency_requests.csv`·`fallback_requests.jsonl`).

검증한 시험(외부 연결 차단 플러그인 `v63_block_network`): `test_v56_rerank_fallback_paths` · `test_v59_rerank_status` · `test_rerank_shared_budget` · `test_v62_hybrid_full_scan` · `test_v60_context_startup_check` · `test_lyrics_cache_refresh` **199 통과**, 차단된 외부 연결 0.
`selftest_analyzers.py` OK(잘못된 입력 7종 · 누출 1종 검출, 지연 집계의 200 아님·구간 없음 계수). 전체 `tests/`는 돌리지 않았다(`src` 변경 없음).

## 11. 파일

| 파일 | 내용 | Git |
|---|---|---|
| `RUN_INFO.md` · `ACTIVATION_DECISION.md` · `SERVER_LOADING_PLAN.md` | 보고서 · 속성 보조 활성화 판단서 · 서버 적재 운영안 | 추적 |
| `v64_runtime.py` · `v64_app.py` · `start_servers.sh` | 차단·스냅샷·가짜 SDK·관측 / 실제 앱 진입점 / 서버 기동 | 추적 |
| `prepare_server.py` · `classify_queries.py` · `run_sweep.py` · `run_latency.py` · `run_fallback.py` · `probe_guards.py` | 실행 도구 | 추적 |
| `analyze_sweep.py` · `aggregate_latency.py` · `analyze_fallback.py` · `diagnose_candidates.py` · `diagnose_paths.py` · `tie_impact.py` · `selftest_analyzers.py` | 집계·진단·자체 검사 | 추적 |
| `server_collections.json` · `latency_sample.json` · `query_categories.csv` · `startup/` · `probe_guards.json` | 환경·표본·기동 기록 | 추적 |
| `sweep_summary.json` · `sweep_per_query.csv` · `latency_summary.json` · `latency_joined.csv` · `latency_requests.csv` · `fallback_summary.json` · `fallback_table.csv` · `diagnose_candidates_{local,server}.json` · `tie_impact.json` · `*_status_after.json` · `latency_runmeta.json` | 경량 결과 | 추적 |
| `sweep_requests.jsonl` · `fallback_requests.jsonl` · `diagnose_paths_*.json` · `logs/` · `ctl/` · `progress.txt` | 요청별 원기록(분석 결과·후보 전체) · 서버 로그 · 구간 기록 · 가짜 호출 기록 | **제외**(`.gitignore`) |
| `artifacts/qdrant_v64_src` · `artifacts/qdrant_server_v64` | 색인 복사본 · 전용 서버 저장소 | 제외(저장소 `.gitignore`) |
