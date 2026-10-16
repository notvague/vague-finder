# results_v56 — 속성 조건 보조 경로(D) 운영 검증: 서버 방식 Qdrant 추가 지연 · 리랭킹 실패 시 서비스 반환값 (2026-10-16)

정확도 실험이 아니다. 검색 정책·가중·후보 수·프롬프트·기본값·운영 `.env`·운영 색인은 바꾸지 않았다.
v54(PR #69, main `0237118`)가 권고하고 v55(PR #70, 병합 전)가 봉인 test에서 동률로 확인한 **D(보조 경로 켬 + 장르 조건부 정책)** 를
활성화하기 전에 남은 두 질문 — ① 서버 방식 Qdrant에서 D의 추가 지연 ② 리랭커 실패 시 실제 서비스가 돌려주는 순서 — 에 답한다.
**§1〜§3은 측정 전에 적었다**(§4 이후가 결과).

## 1. 시작 조건 · 작업 위치

| 항목 | 값 |
|---|---|
| 기준 커밋 | **main `0237118`**(#69 v54 병합 커밋, origin/main과 같음). PR #70(v55)은 열려 있고 병합 전이라 v55 결과는 `/private/tmp/vague-finder-attr-final-validation`에서 읽기만 했다. `src/` 변경 없음 |
| 워크트리 · 브랜치 | `/private/tmp/vague-finder-attr-runtime-validation` · `experiment/attr-runtime-validation`(커밋 전). venv·artifacts(context·embeddings·map·timing·vector_db·audio 캐시)는 메인 체크아웃에 심볼릭 링크, `.env`는 메인 것을 복사해 `QDRANT_PATH`만 바꿨다 |
| 추가 파일 | `tests/fake_gemini_client.py`(가짜 SDK 클라이언트) · `tests/test_v56_rerank_fallback_paths.py`(회귀 10건) · 이 폴더의 스크립트·결과 |
| 다른 세션 | 실행 중인 uvicorn·크롤러·평가 프로세스 없음 확인(시작 시 `ps`). 다른 워크트리·미커밋 변경은 건드리지 않았다 |
| 모델 호출 | **Vertex·AI Studio·Claude 신규 호출 0, Melon 호출 0.** 질의 분석은 저장 캐시(v49 Vertex dev 176 + v55 Vertex test 42)만, 리랭커는 가짜 SDK 클라이언트. 실제 클라이언트 생성은 코드에서 예외로 차단하고 서버 기동 때 확인(§3) |

### A / D 구성 (그 밖은 전부 동일)

| 구성 | `ATTR_FILTER_WEIGHT` | `ATTR_FILTER_GATE` | `ATTR_FILTER_K` | `ATTR_FILTER_USE_GENRE` | `ATTR_FILTER_GENRE_POLICY` | 포트 |
|---|---|---|---|---|---|---|
| **A** 현재 기본값 | 0 | era_plus | 30 | 1 | always | 8056 |
| **D** 활성화 후보 | 0.5 | era_plus | 30 | 1 | conditional | 8057 |

## 2. 실제 운영 경로 (코드·설정에서 확인, 변경 없음)

- **Qdrant 연결**: `src/vector_db/qdrant_backend.py` `QdrantVectorClient` — `QDRANT_URL`이 있으면 서버 모드(`QdrantClient(url=…)`, `exact=True` 검색 파라미터 명시), 없으면 **로컬 파일 모드**(`QDRANT_PATH`, 기본 `artifacts/qdrant`, 한 프로세스만 연다). 운영 `.env`는 `QDRANT_PATH=artifacts/qdrant`(로컬 모드)이고 `QDRANT_URL`은 없다 — **현재 운영은 로컬 모드다.** `docker-compose.yml`에도 Qdrant 서비스가 없다(backend + Mongo만). 하이브리드 조회는 dense·sparse를 각각 전수(`HYBRID_SCAN_LIMIT` 5000 ≥ 5,143? → 5,143곡이면 **5,000 상한에 143곡이 걸린다**, §8 참고) 훑어 코드에서 합산하고 top_k만 payload를 읽는다. 속성 필터는 `translate_filter`로 `$in`→`MatchAny`, 발매 연도는 `$text`→`MatchText`(텍스트 색인 없음 — 서버·로컬 모두 기본 토크나이저로 토큰 전체 일치)
- **서비스 진입점**: `POST /api/v1/search`(`src/backend/api/routes/search.py::search`) → `_analyze`(분석기 벽시계 20+2초, 넘으면 규칙 폴백) → `SearchRouter.search(analysis, top_k, use_rerank, candidate_k, exclude_ids, answers, recorder, timer, …)` → Context 근거 부착 → explain → clarify → `SearchResponse`. 미들웨어가 `/api/v1/search`만 전체 시간을 재고 `X-Request-Id`를 돌려준다(`SEARCH_TIMING_LOG`에 JSONL). 의존성은 `dependencies.py`의 singleton 팩토리(벡터 클라이언트·임베더·리랭커·가사 서비스). 화면(map.js)은 `top_k 15 · candidate_k 30 · explain true · defer_clarify true`로 부른다
- **보조 경로**: `SearchRouter.search` — `_attr_filter_weight() > 0`일 때만 `_build_attribute_filter_with_policy(analysis, answers)`로 필터를 만들고(None이면 닫힘), 열리면 `path.attr_filter` 작업으로 `_search_text(analysis, ATTR_FILTER_K, metadata_filter)`를 다른 경로와 **나란히** 스레드풀에 건다. 결과는 RRF 융합 뒤 `weight/(RRF_K+rank)`로 더해진다. **열렸지만 0곡이면** 필터 검색 자체는 그대로 실행되고(비용 발생) 융합에 더할 것이 없을 뿐이다
- **리랭킹 제한·재시도·폴백**(`GeminiListwiseReranker.rerank_run`): 요청 제한 20초(`GEMINI_RERANK_TIMEOUT_SECONDS`) · 전체 예산 30초(`GEMINI_RERANK_BUDGET_SECONDS`) · 재시도 3회(`max_retries`, 사이 1초) · 1패스 · Search 끔 · 희소 사실 검증 끔. 호출은 `asyncio.wait_for(min(20, 남은 예산))`로 감싸 느린 응답을 **실제로 취소**한다. 응답이 예산 뒤에 오면 버린다. **재시도가 전부 실패해도 예외를 내지 않고** 입력 순서 그대로 + `RERANK_FAILED`를 보고한다. 응답 JSON의 모르는 id·중복은 버리고 빠진 id는 입력 순서로 뒤에 붙인다(`_parse_response`) — 후보 밖 id만 돌아와도 `applied`(순위는 입력 그대로)다
- **라우터의 실패 처리**(`search_router.py` 2448~2645): 정책 생략(`rerank_skip_reason`: 표지 지배·자료 없는 MV 장면)은 모델을 부르지 않고 후보 순서 그대로(MV 생략만 가사 보호 배치 적용). 가사 exact 보호 곡(`select_protected_lyric_ids`: `lyric_match_type=exact` · 확신 ≥0.80)이 있으면 **tier별 그룹 호출 → 일반 후보 호출**로 나눠 부르고 `[*protected, *others][:top_k]`로 합친다 — 한 그룹이 failed면 그 그룹은 입력(보정) 순서, 다른 그룹은 리랭킹 결과다(답이 있어 입력 순서가 보너스 전이었을 때만 `_bonus_order`로 되돌린다). 리랭커가 예외를 **던지면** `except Exception`에서 `RERANK_FAILED` 기록·`revoke_reorder`·`rerank_error` 뒤 폴백 출구로 간다(Gemini 백엔드는 스스로 예외를 삼키므로 이 출구는 백엔드 버그일 때만 닿는다)
- **적용 순서**: 경로 병렬 조회 → RRF 융합(+ Context·보조·연주 단서·피드백 보충) → 명시 단서 가산·시기/성별/장르/형태 가산·재질문 답 가산·거절 제외(`boosted`) → `candidate_ids`/후보 목록 확정 → 보호 곡 판정 → 정책 생략 판정 → 리랭킹(보호 그룹 → 일반) 또는 폴백(`candidate_tracks[:top_k]`) → 라우트에서 Context 근거·explain·clarify. **폴백 결과 = 가산·거절까지 반영된 후보 순서**이지 원시 검색 순서가 아니다
- **평가기와 서비스 차이**: `evaluate_search_accuracy`는 라우터를 직접 부른다 — HTTP·요청 검증·미들웨어·Context 근거 부착·explain 직렬화·clarify·피드백 기록(`capture_search`)이 없고, 분석은 캐시, top-k 10. 서비스는 화면 기본 top-k 15, `defer_clarify`, 응답 직렬화까지 포함한다. 리랭킹 실패 판정 열(`rerank_outcome`)은 평가기만의 집계다. 이번 측정은 **HTTP 로 실제 라우트**를 부른다
- **Mongo 가사**: `LyricsExactSearchService` — 첫 가사 질의(또는 예열 `lyrics` 단계)에서 `vaguefinder.songs` 전곡(5,143)을 읽어 정규화·3-gram 판을 만든다. **TTL 300초(`LYRICS_CACHE_TTL_SECONDS`)** 가 지나면 다음 가사 질의가 다시 읽는다(블로킹, v54·v55 기록 ≈350초). 읽는 동안 다른 요청은 `_load_lock`을 기다린다. 가사 경로는 분석에 verbatim/partial/phonetic 단서가 있을 때만 돈다. 시간 제한은 없다(pymongo 기본 연결 제한만)

## 3. 검증 환경 (측정 전 고정)

- **Qdrant 서버 방식 검증 환경**: Docker `qdrant/qdrant:v1.19.1`(클라이언트 `qdrant-client 1.19.1`과 같은 판) 컨테이너 `vague-finder-qdrant-v56`, `127.0.0.1:6356`(REST) · `6357`(gRPC), 저장소는 워크트리 `artifacts/qdrant_server_v56/storage`(git 제외). 기존 운영 서버·컬렉션은 없다(운영은 로컬 모드) — 중단·삭제·덮어쓴 것 없음. **로컬 PC의 Docker 컨테이너**라 실제 배포 서버의 네트워크·하드웨어 성능을 재는 것이 아니다
- **자료 복사**: 원본 서비스 색인(`artifacts/qdrant`)은 열지 않고, v55 파일 복사본을 다시 복사한 `artifacts/qdrant_eval_v56_local`(226MB)에서 읽었다. 세 폴더의 내용 지문(`.lock` 제외, 상대경로+내용 sha256) **`136034d81855`로 동일**(v55 보고서의 `c39aa7eb08fb`는 다른 계산식이며 거기서도 서비스 색인과 동일했다). `migrate_to_server.py`(qdrant-client `migrate`)로 5개 컬렉션을 옮기고 확인 — 결과 `server_collections.json`: text 1024·DOT+sparse / image 768·COSINE / audio 512·COSINE 각 5,143점, Context dense 5,054 · sparse 639(`__generation_f6424b94621a28cf`), **벡터 설정 동일 · payload 전량 동일(불일치 0) · 벡터 표본 50점/컬렉션 동일**. payload 색인은 서비스 적재기가 서버 모드에서 만드는 것과 같게 Context dense(song_id·category·context_build_id)·sparse(song_id·context_build_id)에만 KEYWORD로 만들었고 곡 컬렉션 3개에는 없다(적재기도 안 만든다). `release_date`에 텍스트 색인이 없으므로 `$text` 필터 의미는 로컬과 같다(사전 조회에서 필터 적중 수가 v54 기록과 같은 모양 — t001 8곡 등, §4-0)
- **서버 2개**(`start_servers.sh`, 운영 FastAPI 앱 `src.backend.main:app`을 `v56_app.py`가 그대로 import): A 8056 · D 8057, 같은 Qdrant 서버·같은 Mongo Atlas(읽기) 공유. 운영 `.env`와 다른 점(검증 프로세스 환경변수뿐): `QDRANT_URL`(서버 모드) · `FEEDBACK_COLLECTION_ENABLED=0`(검증 검색이 운영 Mongo 피드백 기록에 남지 않게) · `SEARCH_TIMING_LOG` · `GCP_PROJECT_ID`는 더미 값, 인증 파일 unset. 그 밖(Context 가중 0.5, `SEARCH_REFERENCE_YEAR=2026`, 리랭커 제한 20/30초, 가사 TTL 300초, 예열 켬)은 운영 기본값
- **모델 차단**(`v56_runtime.block_real_models`): `gemini_client.make_genai_client`·`google.genai.Client`·`anthropic.*Client` 생성을 예외로 바꾸고 기동 때 둘 다 "blocked"인지 확인(`/v56/status`의 `guard`). 기동 검사 `verify_vertex_setup`만 네트워크 없는 대체. 분석기는 `CachedAnalyzer`(캐시에 없으면 500 — 규칙 폴백으로 채우지 않음), 리랭커는 **운영과 같은 `GeminiListwiseReranker`에 가짜 SDK 클라이언트 주입**(`tests/fake_gemini_client.py`, 비동기 면 제공 → `wait_for` 취소까지 실물 경로)
- **회귀 테스트**(모델 자격 제거 환경): `tests/test_v56_rerank_fallback_paths.py` **10 통과** — 정상/예외 3회 재시도/벽시계 취소/예산 안 느린 응답/JSON 오류/후보 밖 id/모드 파일 전환/보호 그룹 부분 실패 합치기/실패 뒤 상태 누수 없음

### 3-1. 지연 표본 선정 규칙 (`classify_queries.py` → `query_categories.csv` · `latency_sample.json`, 측정 전 고정)

- 모집단 dev 176(v06 57 · v09 59 · v10 60), v49 Vertex 분석 캐시. 범주는 **D 설정 · 1턴**에서 `_build_attribute_filter_with_policy`로 판정하고, 열린 질의는 서비스와 같은 `SearchRouter._search_text(metadata_filter)`를 Qdrant 서버에 **한 번** 던져 결과 수를 본다(모델 호출 없음)
- **제외**: 가사 표면 단서(verbatim/partial/phonetic)가 있는 19건 — 가사 경로의 Mongo 판 재적재(TTL 300초, ≈350초)가 A/D와 무관하게 비교를 가린다
- 결과: closed 136(제외 뒤 119) · open_hits 40(38) · **open_empty 0** — dev에는 "열렸지만 0곡"이 없다(v55 test에서는 t047 1건)
- 선정: 범주별 v06→v09→v10 번갈아, 세트 안 query_id 순, 10건. **closed 10**: c601 n018 t024 c602 n019 t029 c603 n020 t031 c605 · **open_hits 10**: m103 n002 t001 q101 n006 t004 q104 n010 t007 q107 · open_empty 0
- **합성 open_empty 10**(보조, 자연 질의 아님): open_hits 10건의 저장 분석에서 **시기 창만** 곡이 없는 1960~1965(서버 count 0)로 바꾼 사본을 `prior_analysis`로 보낸다(2턴 재검색과 같은 정규 입력, 새 분석 호출 없음). D에서 필터가 열리고 0곡임을 사전 조회로 확인했다. A·D에 같은 분석이 가므로 짝 비교는 성립하지만 "열렸는데 비었다"는 조건을 만든 것이지 실제 질의 분포가 아니다
- 측정 절차(`run_latency.py`): 두 서버 `warmup.state=ready` 확인 → 준비 요청(표본 첫 질의 c601, 서버당 1건, 집계 제외) → **3회차 × 질의마다 A→D(1·3회차) / D→A(2회차) 번갈아 순차**, 동시 요청 없음, 서버 재시작 없음(같은 프로세스 반복). 본문은 화면과 같고 `use_rerank=false`만 다르다. 느린 요청도 제외하지 않는다. 서버 구간은 timing JSONL을 request_id로 합친다(`aggregate_latency.py`)

### 3-2. 폴백 검증 사례 (`run_fallback.py`, 측정 전 고정)

| 사례 | 출처 | 모드 순서 |
|---|---|---|
| t014 · t049 | v55 test 분석 캐시 — **장애 재현만**(A 후보 1/16위, D 후보 10/22위) | normal → exception → slow → invalid_json → foreign_ids → slow_within(8초) → normal |
| n007 · t018 | dev, 가사 exact 보호 그룹(v54 호출 3회·2회) | normal → exception → fail_group_first → slow → normal |
| q117 | dev, 표지 지배(정책 생략) | exception → normal (호출 0이어야 한다) |
| q112(성별 답 "남성") · n011(장르 답 "댄스") | dev 2턴: 1턴 정상 응답의 analysis를 prior_analysis로, 1위를 거절, answers 포함 | normal → exception → slow → normal |

A·D 서버 모두에서 돌린다. 확인: 반환 id·순위·개수·중복·후보 포함 여부, 응답 시간(제한 30초 안인가), explain `reorder_stage`·`rerank_calls(_applied)`·`rerank_error`·`rerank_skip_reason`·`notice`, 가짜 호출 기록(횟수·취소/예외/반환), 다음 요청에 남는 상태. t014는 v55 저장 top-10(리랭킹 전)과 폴백 반환을 대조한다.

---

## 4. 환경 · 준비 결과 (측정 직전 확인)

- 기계: Apple M4 Pro · 12코어 · 24GB · macOS 27.0.1, Python 3.14.7, qdrant-client 1.19.1, Qdrant 서버 1.19.1(Docker 29.7.2) — `env_info.json`
- 서버 기동(11:23:31, A·D 동시): 예열 koe5 6.4초 · bm25 0.9 · siglip2 4.4 · clap 2.2 · reranker 건너뜀(Gemini listwise는 예열 안 함) · **lyrics(Mongo 가사 판 적재) A 354.4초 · D 350.8초** · vector_db 0.5/0.4초 → `ready` 약 11:29:40. 두 프로세스가 같은 시각에 Atlas에서 각자 읽었다
- `/v56/status`: `guard = {make_genai_client: blocked, google.genai.Client: blocked}`, 리랭커 `GeminiListwiseReranker`, 리랭커 설정 1패스 · 재시도 3 · 요청 20초 · 예산 30초 · 가중 0.85 · Search 끔 · 희소 검증 끔, 분석 캐시 4개(dev 57·59·60 + test 42, 전부 `gemini_backend=vertex` · `prompt_sha 80c464515d9c` · `postprocess_sha f18778f8f94d`). 측정 뒤 `analysis_misses = []`(캐시 밖 질의 0)
- Qdrant 서버 컬렉션 확인은 §3(`server_collections.json`, all_ok=true). BM25 `bf606b19ab02`
- 사전 조회(`query_categories.csv`, 참고값): 열린 38건의 필터 검색 42~158ms(첫 1건 7.4초 = 모델 적재). 필터 적중 수 분포: 30곡 27건 · 22~27곡 4건 · 1~8곡 7건 — v54 기록(t001 8곡)과 같은 모양

### 4-1. 서버로 옮기면서 자료·필터 의미가 보존됐는가 — 로컬↔서버 직접 대조와 **별칭 누락 발견**

- **1차 측정(11:30~11:54) 뒤 발견**: 같은 코드·같은 분석으로 라우터를 직접 불러 후보 30을 대조(`compare_local_server.py`, 리랭킹 없음)하니 **A 22건 중 4건(t049 · c601 · q101 · n010), D 22건 중 같은 4건**이 서버 모드에서 달랐다(`run1_noalias/local_vs_server_compare_noalias.json`). 로컬 모드는 다른 프로세스에서 다시 돌려도 21/21 동일(임베딩 비결정성 아님)이고 로컬 t049 후보 top-10은 v55 저장값과 글자까지 같았다
- 컬렉션 단위로 같은 벡터를 양쪽에 던지면(`probe_backends.py` · `probe_hybrid.py` · `probe_filters.py`) text dense 5,000 · sparse · image · audio · Context dense 상위 집합과 **하이브리드 합산 상위 100 · 속성/연주 필터 적중 집합이 전부 동일**(점수 차 ≤1.1e-7, 상위 60 순서 동일; 5,000 깊이에서만 동점 순서 차이)
- 조회 호출을 전부 기록해 비교(`trace_queries.py`)하니 서버 모드에는 **Context 전용 후보의 메타데이터 조회(`fetch`)가 없었다** → 서버 로그 `[SearchRouter] 배경 지식 경로 실패 (빈 결과로 대체): published context Qdrant aliases are missing`. **원인: `ContextQdrantSearch`는 Context 컬렉션을 Qdrant 별칭(`…__dev` → `…__dev__generation_f6424…`)으로 찾는데, `qdrant-client migrate`는 컬렉션만 옮기고 별칭은 옮기지 않는다.** 로컬 복사본에는 별칭 2개가 있었고 서버에는 없었다 — 검증 환경의 구성 실수이며 서비스 코드의 문제가 아니다(서비스는 경고 로그 + 빈 결과로 계속 돌았다). Context 기여가 있던 질의(17건 기여)만 후보가 달라진 것과 일치한다
- **12:30 별칭 2개를 서버에 만들고(`server_collections.json`의 `aliases_*`) 다시 대조: A 21 · D 21 전부 동일(42/42)**(t014는 가사 판 적재 6분을 피하려 제외). 따라서 별칭이 있는 서버 모드는 **로컬 모드와 후보 집합·순서가 같다** — 자료·필터 의미는 보존된다
- 영향: **1차 지연 측정과 1차 폴백 검증은 Context 경로가 빈 상태**였다(A·D 동일하게). 보조 경로 비용 자체(필터 검색 1회 추가)는 Context와 무관하지만 병렬 경로 수가 하나 적었으므로 **별칭을 만든 뒤 지연 측정을 다시 했다(§5 = 2차, 1차는 `run1_noalias/`에 보존·§5-3에 요약).** 폴백 검증은 실패 처리 구조를 확인하는 근거로 1차 결과를 남기되, Context가 후보·그룹 구성에 영향을 줄 수 있어 정상 환경 전체 검증으로 일반화하지 않는다. t014·t049의 후보 목록은 2차(별칭 있음)에서 normal·exception만 다시 확인했다(§6-2)
- 배포 교훈: 서버 모드로 적재할 때는 **`qdrant_load_context`의 정규 경로**(별칭을 원자적으로 바꾼다)를 쓰거나 별칭 2개를 함께 옮겨야 한다. 기동 시 Context 별칭이 없으면 경고 로그만 남고 검색은 Context 없이 조용히 돌아간다(§7-⑥)

## 5. 지연 비교 — Qdrant 서버 방식, A 대 D (**2차, Context 별칭 있음** — `run_latency.py` → `latency_requests.csv` · `latency_joined.csv` · `latency_per_query.csv` · `latency_summary.csv` · `latency_aggregate_stdout.md`)

서버 재기동 12:33:09(예열 lyrics A 355.0초 · D 350.7초, `ready` 12:39:32) → 측정 12:39:32~12:39:59. 182요청(준비 2 + 측정 180 = 30질의 × 2구성 × 3회차), **오류 0 · 시간 초과 0 · timing 기록 미대조 0**. 리랭킹 끔(`use_rerank=false`), 그 밖은 화면과 같은 본문(top_k 15 · candidate_k 30 · explain · defer_clarify). Mongo 가사 적재 0회(가사 질의 제외 규칙대로). 준비 요청(c601) A 1,010ms · D 989ms(기동 뒤 첫 요청, 집계 제외). Context 경로 구간은 A·D 각 21/90 요청에 있었다(Context 단서가 있는 질의만 돈다).

### 5-1. 범주별 (요청 단위 n=30 = 10질의 × 3회차, ms)

| 범주 | 구성 | 전체 요청 중앙값 / p95 / 최대 | 검색 단계(search) 중앙값 / p95 / 최대 | 보조 경로 `path.attr_filter` 실행 중앙값 / p95 / 최대 | 텍스트 경로 `path.text` 중앙값 | 질의별(3회 중앙값) D−A 전체 요청: 중앙값 / p95 / 최대 |
|---|---|---|---|---|---|---|
| closed (10) | A | 79.7 / 179.1 / 561.9 | 76.0 / 173.8 / 557.7 | — (구간 없음, 30/30) | 72.2 | **−1.4 / 3.9 / 5.1** |
|  | D | 75.2 / 148.1 / 501.1 | 71.3 / 142.3 / 497.1 | — (구간 없음, 30/30) | 70.1 |  |
| open_hits (10) | A | 125.6 / 287.7 / 337.5 | 118.8 / 281.6 / 330.1 | — | 115.9 | **+36.7 / 95.0 / 110.9** |
|  | D | 156.8 / 332.9 / 427.4 | 150.0 / 325.9 / 419.7 | 139.3 / 237.7 / 328.1 (대기 ≈1ms) | 135.4 | |
| open_empty (자연) | — | **dev에 0건** | | | | |
| open_empty 합성 (10) | A | 116.5 / 307.1 / 379.0 | 109.8 / 300.3 / 371.6 | — | 106.4 | **+40.8 / 49.9 / 51.3** |
|  | D | 153.3 / 370.8 / 409.1 | 146.4 / 363.5 / 401.6 | 134.1 / 307.9 / 325.4 (0곡 반환) | 123.9 | |
| 전체 (30) | A | 110.7 / 293.2 / 561.9 | 105.0 / 286.5 / 557.7 | — | 94.8 | +30.6 / 64.7 / 110.9 |
|  | D | 138.7 / 338.8 / 501.1 | 131.6 / 331.9 / 497.1 | 137.7 / 303.0 / 328.1 (60건) | 115.1 | |

### 5-2. 읽기

추가 지연 통계는 질의마다 A의 3회 중앙값과 D의 3회 중앙값을 구해 뺀 뒤, 범주별 10개 차이를 집계한 값이다. 개별 요청 30쌍의 차이 분포나 운영 요청 p95와 구분한다.

- **닫힌 질의는 차이가 없다**(질의별 D−A −1.4 중앙값, 최대 +5.1ms; `path.attr_filter` 구간이 없다 — 가중만 켜고 필터가 None이면 호출도 없다)
- **열린 질의는 D가 질의별 +16~+111ms(중앙값 +37ms)** 느리다. 보조 경로는 다른 경로와 **나란히** 돌고(대기 ≈1ms, 실행 100~330ms = 텍스트 경로와 같은 두 번 전수 조회 + payload 읽기) 벽시계에 더해지는 것은 그 전부가 아니라, 같은 Qdrant 서버에 전수 조회가 늘어 **텍스트 경로 자체도 느려진 만큼**(open_hits `path.text` A 115.9 → D 135.4ms)과 끝나는 시각 차이다. 가장 큰 +111ms(q101)는 Context·연주 단서·제목 경로까지 함께 도는 질의였다
- **열렸지만 0곡(합성)도 같은 비용이 든다**(+41ms, `path.attr_filter` 134ms) — 필터 검색은 결과를 알기 전에 실행되므로 0곡이라고 싸지지 않는다. 합성 입력이므로 "비용이 생기는가"의 답으로만 쓴다
- 후보 지문: 질의별 3회 **A 30/30 · D 30/30 반복 동일**, A=D는 20/30(closed 10 + 합성 10), 다른 10건은 전부 open_hits — 열림 판정이 서버 응답과 일치한다. 융합 후보 수는 전부 30
- 300ms 이상 11건은 1회차(기동 뒤 첫 회차) 또는 A·D 양쪽 같은 질의에 함께 나타났다(t031 r1 A 562/D 501, q101 r1·r3 양쪽 …) — 양쪽 공통의 산발 지연이지 D 고유가 아니다. 어느 것도 통계에서 빼지 않았다
- 검색 밖 서버 시간(의존성·검증·explain·clarify·직렬화)은 3~6ms, 클라이언트−서버 차는 1.5ms. 추가 지연 중앙값 37ms는 과거 v54·v55 리랭킹 중앙값 4.5~5.5초의 약 0.7~0.8%다. **서로 다른 측정 간 참고 비율이며, 리랭킹을 켠 전체 요청 지연은 이번에 측정하지 않았다**
- 같은 프로세스에서 3회 반복(서버 재시작 없음) · A·D는 다른 프로세스 · Qdrant 서버와 Mongo 공유 · 임베딩 모델은 프로세스마다 따로. 로컬 PC의 Docker 컨테이너라 **배포 서버의 네트워크·하드웨어 성능은 재지 않았다**
- 1차(Context 빔, §5-3)와 견주면 열린 질의의 D−A가 +29 → +37ms로 조금 커졌다(병렬 경로가 하나 늘어 경합이 커진 것과 일치), 닫힌 질의는 두 번 모두 차이 없음
- v55의 로컬 복사본 측정(경로 열린 9건 검색 중앙값 A 0.65 → D 1.14초)과 숫자를 직접 비교하지 않는다 — 그쪽은 평가기 직접 호출·로컬 파일 모드·test 질의·다른 날의 측정이다

### 5-3. 1차(11:30:04~11:33, Context 경로 빔 — `run1_noalias/`) 범주별 (요청 단위 n=30 = 10질의 × 3회차, ms)

182요청(준비 2 + 측정 180), 오류 0 · 시간 초과 0 · timing 미대조 0. 준비 요청(c601) A 166ms · D 193ms(집계 제외). Context 경로가 비어 병렬 경로가 하나 적은 조건이라 **2차와 직접 비교하지 말고 참고로만** 둔다.

| 범주 | 구성 | 전체 요청 중앙값 / p95 / 최대 | 검색 단계(search) 중앙값 / p95 / 최대 | 보조 경로 `path.attr_filter` 실행 중앙값 / p95 / 최대 | 텍스트 경로 `path.text` 중앙값 | 질의별(3회 중앙값) D−A 전체 요청: 중앙값 / p95 / 최대 |
|---|---|---|---|---|---|---|
| closed (10) | A | 69.0 / 207.3 / 274.8 | 64.5 / 200.4 / 271.0 | — (구간 없음, 30/30) | 63.8 | **−1.0 / 1.1 / 2.4** |
|  | D | 68.7 / 217.3 / 302.6 | 63.9 / 211.6 / 298.9 | — (구간 없음, 30/30) | 63.2 |  |
| open_hits (10) | A | 127.0 / 267.6 / 316.8 | 120.0 / 260.5 / 310.1 | — | 115.6 | **+28.8 / 58.8 / 61.8** |
|  | D | 151.6 / 324.5 / 352.2 | 144.8 / 317.0 / 345.6 | 134.2 / 313.0 / 337.4 (대기 ≈1ms) | 134.2 | |
| open_empty (자연) | — | **dev에 0건** | | | | |
| open_empty 합성 (10) | A | 110.3 / 280.3 / 286.0 | 103.8 / 273.2 / 279.4 | — | 100.7 | **+33.8 / 45.4 / 49.8** |
|  | D | 143.8 / 299.6 / 317.1 | 136.8 / 293.2 / 310.3 | 133.1 / 226.2 / 293.8 (0곡 반환) | 122.9 | |
| 전체 (30) | A | 102.3 / 274.1 / 316.8 | 96.1 / 268.4 / 310.1 | — | 92.3 | +22.7 / 52.7 / 61.8 |
|  | D | 135.1 / 303.5 / 352.2 | 128.1 / 298.8 / 345.6 | 133.1 / 294.4 / 337.4 (60건) | 108.6 | |

- **닫힌 질의는 차이가 없다**(질의별 D−A −7.6~+2.4ms, `path.attr_filter` 구간 자체가 없다 — 가중만 켜고 필터가 None이면 호출도 없다)
- **열린 질의는 D가 질의별로 +10~+62ms(중앙값 +29ms)** 느리다. 보조 경로는 텍스트 경로와 **나란히** 돌고(대기 ≈1ms, 실행 100~190ms = 텍스트 경로와 같은 두 번 전수 조회) 벽시계에 더해지는 것은 그 전부가 아니라, 같은 Qdrant 서버에 전수 조회가 2→4회로 늘어 **텍스트 경로 자체도 느려진 만큼**(open_hits `path.text` A 115.6 → D 134.2ms)과 끝나는 시각 차이다
- **열렸지만 0곡(합성)도 같은 비용이 든다**(+34ms, `path.attr_filter` 133ms) — 필터 검색은 결과를 알기 전에 실행되므로 0곡이라고 싸지지 않는다. 자연 질의가 아니라 시기 창만 바꾼 합성 입력이므로 분포가 아니라 "비용이 생기는가"의 답으로만 쓴다
- 후보 지문: 질의별 3회 **A 30/30 · D 30/30 반복 동일**, A=D는 20/30(closed 10 + 합성 10), 다른 10건은 전부 open_hits(보조 후보가 들어간 것) — 열림 판정이 서버 응답과 일치한다. 융합 후보 수는 전부 30
- 250ms 이상 요청 16건은 A·D에 같은 질의·같은 회차로 짝지어 나타났다(c605 r1, t007 r1, t001 r2, n019 r3, t004 r3 …) — 양쪽 공통의 산발 지연(이 PC의 다른 부하)이지 D 고유가 아니다. 어느 것도 통계에서 빼지 않았다
- 검색 밖 서버 시간(의존성·검증·explain·clarify·직렬화)은 3~5ms, 클라이언트−서버 차는 1.5ms. 과거 리랭킹 중앙값과의 비율은 참고만 가능하며, 리랭킹을 켠 전체 요청 지연은 측정하지 않았다
- 같은 프로세스에서 3회 반복(서버 재시작 없음) · A·D는 다른 프로세스 · Qdrant 서버와 Mongo 공유 · 임베딩 모델은 프로세스마다 따로. 로컬 PC의 Docker 컨테이너라 **배포 서버의 네트워크·하드웨어 성능은 재지 않았다**
- v55의 로컬 복사본 측정(경로 열린 9건 검색 중앙값 A 0.65 → D 1.14초)과 숫자를 직접 비교하지 않는다 — 그쪽은 평가기 직접 호출·로컬 파일 모드·test 질의·다른 날의 측정이다

## 6. 리랭킹 실패·폴백 — 실제 서비스 반환값 (`run_fallback.py` 11:31~11:54, **1차 조건(Context 별칭 누락, §4-1)** → `run1_noalias/fallback_results.{csv,jsonl}` · `run1_noalias/fallback_summary.md` · `run1_noalias/fake_calls_{A,D}.jsonl`; t014·t049 2차 확인은 `fallback_results_subset_run2.csv`)

68요청(A 34 · D 34), **전부 HTTP 200**(500·연결 오류 0). 가짜 SDK 클라이언트의 모드만 바꿨고 리랭커·라우터·라우트는 운영 코드다. 모든 폴백 반환은 결과 15곡 · 중복 0 · 전부 후보 안이었다.

### 6-1. 실패 방식별 서비스 동작 (단일 호출 질의: t014 · t049 · q112 · n011 · t018의 일반 그룹)

| 모드(가짜 클라이언트) | 리랭커가 실제로 한 일 | 응답 시간(A·D 양쪽, 요청 전체) | explain 단계 · 호출/적용 | 반환 순서 |
|---|---|---|---|---|
| normal(역순 응답) | 1회 호출 · JSON 파싱 · 융합 · 규칙 → applied | 0.10~0.51초 | `model_applied` · 1/1 | 모델 순서가 반영됨(정답이 후보 1위면 역순 때문에 15위 밖으로 — 가짜의 성질) |
| exception(호출마다 예외) | **3회 재시도(사이 1초)** 뒤 예외 없이 `failed` | **2.1~2.6초** | `model_failed` · 1/0 · `rerank_error` 없음(라우터 예외 아님) | **후보 순서(가산·거절 반영) 앞 top_k 그대로** — 15/15 사례에서 `결과 top-10 == candidate_ids top-10` |
| slow(응답 없음) | 1차 `wait_for` 20초 취소 → 1초 대기 → 2차 약 9초 취소(남은 예산) → 3차는 예산 초과로 생략 → `failed` | **31.2~31.3초**(예산 30초 + 대기·취소 처리) | `model_failed` · 1/0, 가짜 기록 `cancelled` 2회 | 후보 순서 그대로 |
| invalid_json(JSON 아닌 본문) | 파싱 실패 → 정정 문구 붙여 3회 → `failed` | 2.2~2.6초 | `model_failed` · 1/0 | 후보 순서 그대로 |
| foreign_ids(후보 밖 id만) | 파서가 전부 버리고 입력 순서로 채움 → **`applied`** | 0.17~0.45초 | **`model_applied` · 1/1** | 후보 순서와 같음 — 모델이 기여한 것이 없는데 단계는 "적용"으로 기록된다(§7-③) |
| slow_within(8초 뒤 정상) | 예산 안 응답 → applied | 8.1~8.3초 | `model_applied` · 1/1 | 모델 순서 반영 |
| 실패 뒤 normal | 바로 정상 | 0.17~0.26초 | `model_applied` · 1/1 | **첫 normal 과 같은 top-15**(A·D 모두 7사례 전부) — 오류 상태·순위 누출 없음 |

### 6-2. t014 · t049 — v55가 걱정한 "폴백이면 D에서 10위"가 실제로 그렇다

| 질의 | 구성 | 폴백(exception·slow·invalid_json·foreign_ids 전부 동일) 반환에서 정답 순위 | v55 저장 리랭킹 전 top-10과 | 비고 |
|---|---|---|---|---|
| **t014** | A | **1위** | top-10 **글자까지 동일** | 609293 이 후보 1위 |
|  | D | **10위** | top-10 **글자까지 동일**(106465 … 609293) | 보조 경로가 다른 곡 9개를 위로 올렸다 — 리랭커가 죽으면 사용자는 정답을 10번째에서 본다. 정상(모델 적용) 때는 v55에서 세 구성 모두 1위였다 |
| t049 | A | 1차 12위 → **2차(별칭 있음) 15위 밖** | 1차는 6번째부터 다름(Context 경로 빔, §4-1) → **2차 top-10 글자까지 동일** | v55 후보 16위와 일치(top_k 15 밖) |
|  | D | 15위 밖 | 1차 8번째부터 다름 → **2차 동일** | v55 후보 22위와 일치 |

2차 확인(`fallback_results_subset_run2.csv`, 별칭 생성 뒤 t014·t049 × normal·exception × A·D = 8요청): 네 폴백 반환 모두 v55 리랭킹 전 top-10과 **글자까지 동일**, 결과 15곡 · 중복 0 · 전부 후보 안. 1차의 t049 차이는 Context 경로가 비어 있던 검증 환경 탓이었다.

→ **폴백 반환 = 리랭킹 직전 후보 순서(보호 배치 포함)**이고, D는 보조 후보에 밀린 정답이 그대로 밀린 채 나갈 수 있다. v55의 t014는 정상 리랭킹에서 A·B·D 모두 1위였지만, 정상 응답이 항상 순위를 복구한다는 보장은 아니다. v55 test 42건에서 A→D 후보 순위 하락은 4건(t014 1→10 · t049 16→22 · t023 5→6 · t051 4→6), dev 176(v54)에서는 하락보다 상승이 많았다 — 폴백 빈도(운영 Vertex 실패율)와 곱해 봐야 하는 위험이며 이번에 실패율은 재지 않았다.

### 6-3. 가사 exact 보호 그룹 질의 — 그룹별 호출과 **제한 시간 합산**

| 질의(그룹 크기) | 모드 | 가짜 호출 | 응답 시간 | 단계 · 호출/적용 | 반환 |
|---|---|---|---|---|---|
| **n007**(보호 2 · 보호 2 · 일반 23 = 3호출) | normal | 3 returned | 0.17초 | applied · 3/3 | 보호 2그룹 앞 + 일반(역순) · 1위는 `gemini_lyric_anchor_protect` 로 되돌아옴 |
|  | exception | **9 raised**(그룹마다 3회) | **6.4초** | `model_failed` · 3/0 | 보호 그룹 tier 순 + 일반 입력 순(=후보 순). top-10 이 candidate_ids 순과 **다르다**(보호 배치가 먼저) |
|  | fail_group_first(첫 그룹만 실패) | 3 raised + 2 returned | 2.2초 | **`model_applied` · 3/2** | 실패한 그룹은 입력 순서, 나머지 두 그룹은 모델 순서 — 부분 실패가 단계에는 "적용"으로 보이고 호출 수(3/2)에만 남는다 |
|  | **slow** | 6 cancelled(그룹마다 20+9초) | **93.3초** | `model_failed` · 3/0 | 후보 순서(보호 배치 포함) |
| **t018**(보호 2 · 일반 28 = 2호출) | exception | 6 raised | 4.2초 | `model_failed` · 2/0 | 후보 순서 |
|  | fail_group_first | 3 raised + 1 returned | 2.2초 | `model_applied` · 2/1 | 보호 그룹 입력 순 + 일반 모델 순 |
|  | **slow** | 4 cancelled | **62.2초** | `model_failed` · 2/0 | 후보 순서 |

- **리랭킹 예산 30초는 `rerank_run` 호출 하나의 예산**이고, 보호 그룹 질의는 그룹마다 따로 부르므로 요청 전체는 **N × 30초**다 — n007 93초 · t018 62초. 화면의 검색 제한은 60초(`map.js SEARCH_TIMEOUT_MS = 60000`)라 **두 사례 모두 사용자는 "검색이 60초 안에 끝나지 않았습니다"를 보고, 서버는 그 뒤에도 끝까지 돈다.** A·D 공통(보조 경로와 무관)이며 지금 운영 코드의 동작이다(§7-①)
- 부분 실패(한 그룹만 실패)는 응답 어디에도 "일부 실패"로 표시되지 않는다 — `reorder_stage=model_applied`, `rerank_calls=3 · rerank_calls_applied=2`만 다르다(§7-③). 평가기는 이것을 `partial_failure`로 세지만 서비스 explain에는 그 분류가 없다

### 6-4. 정책 생략 · 2턴 · Mongo 재적재

- **q117(표지 지배)**: exception 모드에서도 **가짜 호출 0**, `policy_skipped`, 0.08~0.13초, 두 모드 반환 동일 — 정책 생략은 리랭커 상태와 무관하게 모델을 부르지 않는다
- **q112(성별 답 "남성" + 1위 거절) · n011(장르 답 "댄스" + 1위 거절)**: normal → exception → slow → normal 에서 실패 시 **답 가산·거절 제외가 반영된 후보 순서**(candidate_ids top-10과 동일)로 돌아왔고, 거절한 곡은 결과에 없었다. D의 q112는 1턴과 2턴 모두 보조 필터가 성별 조건으로 장르를 뺀 상태(`attr_filter_genre` 기록은 평가기 explain에만 있고 서비스 `explain` 응답에는 없다 — §7-④). 실패 뒤 normal 은 첫 normal 과 같은 top-15
- **Mongo 가사 판 재적재가 두 번 실측됐다**: A n007 다섯째 요청(11:34:53) **350.6초**, D t014 첫 요청(11:41:xx) **351.3초** — 둘 다 예열 적재(11:29:40 완료)에서 **TTL 300초가 지난 뒤 첫 가사 경로 요청**이었다(`path.lyrics` 구간이 350초, 리랭킹 3ms). 즉 운영 기본값에서는 **가사 판 적재 완료 후 300초가 지난 뒤 첫 가사 요청이 전체 재적재를 기다린다**. 검색을 계속해도 만료 기준은 갱신되지 않으며, 이번 환경에서는 그 대기가 약 350초였다(이 PC→Atlas 기준). 보조 경로·리랭커와 무관하며 A·D 동일. v54·v55의 평가기 기록(≈356초)과 같은 현상이 서비스 경로에서도 확인된 것이다(§7-②)

## 7. 발견한 구현상 문제·관찰 (이번 작업에서 고치지 않았다 — 최소 재현과 영향만 남긴다. 수정은 후속 작업)

| # | 내용 | 최소 재현 | 영향 범위 | 비고 |
|---|---|---|---|---|
| ① | **리랭킹 예산이 요청 단위가 아니라 호출 단위다.** 가사 exact 보호 그룹이 있는 질의는 그룹마다 `rerank_run`을 부르므로 모델이 멈추면 요청이 N×30초 걸린다. 실측 n007 93.3초 · t018 62.2초 > 화면 제한 60초 | `run_fallback.py`의 n007·t018 `slow` 행(가짜 클라이언트 `slow` 모드) / 단위: `tests/test_v56_rerank_fallback_paths.py::test_protected_group_failure…`의 구조로 slow 모드 | 가사 exact 일치가 여러 곡인 질의(dev 176 중 보호 호출 2회 이상 질의는 v54 기록에서 n007·t003·t018 등 소수). A·D 공통 | `search_router.py` 2492~2565: 그룹 호출이 각각 `call_reranker` → 예산은 `GeminiListwiseReranker.rerank_run` 안에서만 센다. 요청의 리랭킹 단계 전체가 공유하는 마감(`deadline`)과 남은 시간을 넘기는 수정이 후보. 재시도 사이 `time.sleep(1)`도 남은 예산으로 제한해야 한다(현재 30초가 약 31초로 초과되는 원인 중 하나) |
| ② | **Mongo 가사 판 만료 시 동기 전체 재적재.** TTL은 적재 완료 후 300초이며 접근으로 연장되지 않는다. 만료 후 첫 가사 요청이 약 350초의 재적재를 기다리고, 그 동안 다른 가사 질의는 `_load_lock`을 기다린다(스레드풀 4개 중 1개 점유) | 서버 예열 완료 후 5분 뒤 가사 단서 질의 1건(이번 A n007 5번째 · D t014 1번째 요청, `timing_{A,D}.jsonl`의 `path.lyrics` 350초) | 모든 가사 단서 질의(dev 176 중 19건 비율). A·D 공통, 로컬·서버 모드 공통. 지연의 대부분은 이 PC→Atlas 전곡 읽기(5,143건 full_lyrics) | `LYRICS_CACHE_TTL_SECONDS`·`lyrics_exact_search._load`. 정책 변경은 이 작업 범위 밖 — 후속에서 TTL/백그라운드 갱신/증분 읽기 중 결정 |
| ③ | **부분 실패·무효 응답이 "적용"으로 보인다.** 그룹 하나가 failed여도 `reorder_stage=model_applied`(호출 수 3/2에만 흔적), 후보 밖 id만 돌아와도 파서가 입력 순서로 채워 `applied` | `run_fallback.py` n007·t018 `fail_group_first` 행 · t014 `foreign_ids` 행 | 설명 패널·운영 집계가 실패를 과소 집계한다. 평가기는 `partial_failure`를 따로 세므로 서비스와 분류가 다르다 | `_parse_response`가 유효 id 0개면 실패로 볼지, `reorder_stage`에 부분 실패 단계를 둘지는 후속 결정 |
| ④ | 서비스 `explain` 응답에 `attr_filter_genre`(D의 장르 유지/제외 사유)가 없다 — `SearchExplainOut`에 필드가 없어 평가기 explain JSONL에만 남는다 | A/D 서버 응답의 `explain` 키 비교(`latency_requests.csv`의 `attr_filter_genre` 열 전부 빈 값) | 활성화 뒤 운영에서 정책 판정을 응답으로 확인할 수 없다(로그·평가기만) | 기능 결함은 아니고 관측 가능성 문제 |
| ⑥ | **서버 모드에서 Context 별칭이 없으면 조용히 Context 없이 돈다** — 기동 검사가 없고 요청마다 경고 로그(`배경 지식 경로 실패 (빈 결과로 대체)`) + 빈 결과. `qdrant-client migrate`·수동 복사는 별칭을 옮기지 않는다 | 서버 Qdrant에 컬렉션만 옮기고 별칭 없이 기동 → 이번 1차 측정 전체(§4-1) | 서버 모드 배포 때 Context 이득(v50: Hit@10 +3)이 통째로 빠져도 응답만 봐서는 모른다. A·D 공통 | 기동 시 `/health`나 lifespan에서 별칭 확인을 추가할지, 적재 도구로만 옮기게 문서화할지 후속 결정 |
| ⑤ | `HYBRID_SCAN_LIMIT`(기본 5000) < 5,143곡 — dense 전수 합산에서 dense 점수 최하위 143곡이 빠진다. 문서 주석("코퍼스보다 크면 전수")은 3,010곡 때 기준 | `check_scan_limit.py`: t014 텍스트 경로 상위 30이 상한 5000과 6000에서 **25번째부터 다르다**(t049는 같음; 의심 곡 4020215의 dense 순위는 1,716위라 이 문제와 무관) | 후보 30 끝자리가 바뀔 수 있다. A·D·로컬·서버 공통, 3,010곡 이후 모든 측정에 들어 있던 조건 | `QDRANT_HYBRID_SCAN_LIMIT=6000`으로 올리면 되지만 기준선 후보 지문이 바뀌므로 재측정과 함께 결정 |

## 8. 확인하지 못한 것 · 한계

1. **실제 배포 서버 성능은 재지 않았다.** 로컬 PC의 Docker Qdrant(같은 기계, loopback)라 네트워크 왕복·다른 하드웨어의 필터 검색 비용은 모른다. 숫자는 "서버 방식 코드 경로의 추가 비용" 크기이지 배포 지연 예측이 아니다
2. **리랭킹을 켠 전체 요청 지연과 Vertex 실패율은 재지 않았다**(신규 모델 호출 금지). D의 폴백 위험(§6-2)은 실패율과 곱해야 하며, v54·v55에서 Vertex 실패는 유효 실행 기준 0이었지만 v52(504)·v54(499 CANCELLED)에서 일시 장애가 있었다
3. 자연 `open_empty` 질의가 dev에 없어 합성 입력으로 대신했다. test의 t047 1건은 쓰지 않았다
4. 폴백 검증은 7질의 × 5~7모드 × 2구성이다. 리랭커 백엔드의 `except Exception` 출구(라우터가 예외를 잡는 길)는 Gemini 백엔드가 예외를 삼켜 서비스 경로로는 닿지 않았다 — 기존 단위 테스트(`_StubReranker(fail=True)`)만 그 길을 덮는다
5. 동시 요청(여러 사용자)은 넣지 않았다. 두 서버는 번갈아 순차로 받았고, ①·②의 스레드풀 점유가 동시 요청에서 어떻게 번지는지는 미측정
6. 1차 측정은 Context 별칭 누락으로 Context 경로가 빈 조건이었다(§4-1). 별칭을 만든 뒤 로컬↔서버 후보 42/42 동일을 확인하고 지연을 다시 쟀지만, 폴백 검증의 본 실행(68요청)은 1차 조건이다 — 실패 처리 구조는 확인했지만 Context에 따라 후보·보호 그룹 구성이 달라질 수 있다. t014·t049는 2차에서 확인했고, n007·t018·q112·n011은 운영 결함 수정 후 정상 별칭 환경에서 재검증해야 한다
7. 가짜 리랭커의 `normal`은 역순 응답이라 품질을 말하지 않는다. 모든 품질 숫자는 v54(dev)·v55(test) 것이다

## 9. 최종 판단 — **조건부 권고**

근거는 정확도가 아니라 추가 지연·제한 시간·폴백 동작·남은 불확실성이다(정확도는 v54 dev +2/0, v55 test 동률 그대로).

| 질문 | 답 |
|---|---|
| 서버 방식 Qdrant에서 D는 A보다 얼마나 느린가 | 닫힌 질의 0(±5ms), 열린 질의 **+37ms 중앙값 · p95 +95 · 최대 +111ms**(열린 10질의 각각의 A/D 3회 중앙값 차이 분포, 2차 로컬 Docker). 개별 요청 차이의 p95 또는 배포 지연이 아니다 |
| 경로가 열렸지만 결과가 없어도 비용이 드는가 | **든다**(+41ms, 필터 검색은 결과를 알기 전에 실행) — 합성 입력 기준, dev에 자연 사례 0 |
| 리랭커가 실패하면 무엇을 돌려주는가 | **리랭킹 직전 후보 순서(가산·거절·보호 배치 반영)** 앞 top_k. t014는 **A 1위 · D 10위**로 실측(v55 저장값과 동일). v55에서 t014의 정상 리랭킹 결과는 세 구성 모두 1위였다(일반적인 복구 보장 아님) |
| 제한 시간 안에 반환하는가 | 단일 호출 질의는 실패 모드별 **2.2초(예외·JSON) · 31초(무응답) · 0.2~0.5초(무효 id)**. 가사 보호 그룹 질의는 **그룹 수 × 30초**(93초·62초 > 화면 60초) — A·D 공통 기존 결함(§7-①) |
| 일부 그룹만 실패하면 | 실패 그룹은 입력 순서, 성공 그룹은 모델 순서로 **합쳐서** 반환. 단계는 `model_applied`, 호출/적용 수(3/2)에만 흔적(§7-③) |
| 반복 요청 사이 누출 | 없음 — 실패 뒤 normal 7사례 × 2구성 모두 첫 normal과 같은 top-15, 가짜 호출 수도 요청마다 독립 |

**조건부 권고의 조건**
1. D 고유의 비용은 열린 10질의의 A/D 3회 중앙값 차이 +37ms(이 차이 분포의 p95 +95ms)와 **리랭커 실패 시 보조 후보에 밀린 정답이 그대로 노출되는 것**(t014형)이다. 전자는 작고, 후자는 Vertex 실패율에 비례한다 — 실패율은 이번에 재지 않았다(v54·v55 유효 실행 0, v52·v54에서 일시 장애 504·499 관측). **운영 실패율을 한 번 더 확인하거나 폴백 노출을 감수한다는 결정이 있으면 켤 수 있다**
2. 켜기 전후와 무관하게 **§7-①(그룹 예산 합산 → 60초 초과)과 ②(가사 캐시 만료 후 동기 전체 재적재 → 약 6분 요청)는 A에도 있는 운영 결함**이므로 활성화 결정과 분리해 먼저 고치기를 권한다. 이 둘이 사용자가 실제로 겪을 가장 큰 지연이다
3. 서버 모드로 배포한다면 Context 별칭을 반드시 함께 옮기고 기동 때 확인한다(§7-⑥). 그렇지 않으면 Context 이득이 조용히 사라진다
4. 활성화는 `.env`에 `ATTR_FILTER_WEIGHT=0.5 · ATTR_FILTER_GENRE_POLICY=conditional`(나머지는 기본값과 같음)을 적는 것으로 충분하고, 이번 작업은 운영 `.env`·기본값·색인을 바꾸지 않았다. 켠 뒤에는 `explain`에 `attr_filter_genre`가 없으므로(§7-④) 서버 로그·평가기로만 정책 판정을 확인할 수 있다

## 재현 (모델 호출 없음)

먼저 저장 결과만 재집계한다. 아래 집계는 모델·DB·Docker를 사용하지 않는다. 지연 집계는 기존 CSV를 같은 계산식으로 다시 쓴다.

```sh
cd /private/tmp/vague-finder-attr-runtime-validation
OUT=experiments/reranking/results_v56_attr_runtime_validation
venv/bin/python "$OUT/aggregate_latency.py"
venv/bin/python "$OUT/summarize_fallback.py" --input-dir "$OUT/run1_noalias"
venv/bin/python -m pytest tests/test_v56_rerank_fallback_paths.py tests/test_v56_migration.py -q
```

**새 서비스 실행은 별도 재현 작업이다.** 아래 명령은 Mongo 읽기·로컬 임베딩·전용 Docker를 사용하며 결과 파일을 덮어쓸 수 있다. 원래 측정 폴더를 보존한 별도 체크아웃에서 실행한다. venv·모델 자료·BM25·Mongo 읽기 설정과 `artifacts/qdrant_eval_v56_local` 복사본이 필요하다. 운영 색인 폴더를 직접 열지 않는다.

v55가 병합된 체크아웃에서는 같은 레포의 v55 분석 캐시를 기본으로 읽는다. 병합 전이라면 아래 환경변수로 읽을 캐시를 지정한다(개인 워크트리 경로를 코드에 고정하지 않음).

```sh
export V56_TEST_ANALYSIS_CACHE=/private/tmp/vague-finder-attr-final-validation/experiments/reranking/results_v55_attr_final_validation/analysis_cache_v10_test_vertex.json

# 보존된 v56 컨테이너를 다시 사용할 때
docker start vague-finder-qdrant-v56
# 새 환경에 컨테이너가 없을 때만 위 명령 대신 실행:
# docker run -d --name vague-finder-qdrant-v56 -p 127.0.0.1:6356:6333 -p 127.0.0.1:6357:6334 -v "$PWD/artifacts/qdrant_server_v56/storage:/qdrant/storage" qdrant/qdrant:v1.19.1

venv/bin/python "$OUT/migrate_to_server.py"
# 컬렉션뿐 아니라 Context 별칭 복원·일치 검증까지 성공해야 계속한다.
# 기존 별칭이 다른 컬렉션을 가리키거나 원본 Context 별칭이 없으면 중단한다.
venv/bin/python "$OUT/compare_local_server.py" --mode local --arm A
venv/bin/python "$OUT/compare_local_server.py" --mode server --arm A
venv/bin/python "$OUT/compare_local_server.py" --mode local --arm D
venv/bin/python "$OUT/compare_local_server.py" --mode server --arm D
venv/bin/python "$OUT/compare_local_server.py" --compare
# 42/42 동일 여부를 확인한 뒤 서버를 시작한다.
venv/bin/python "$OUT/classify_queries.py"
zsh "$OUT/start_servers.sh" both
venv/bin/python "$OUT/run_latency.py"
venv/bin/python "$OUT/aggregate_latency.py"
venv/bin/python "$OUT/run_fallback.py"
venv/bin/python "$OUT/summarize_fallback.py"
```

### PR 전 보완 (측정 후 변경, 새 운영 측정 없음)

- 보고서의 TTL 기준·지연 통계 모집단·t014 복구 범위·Context 누락 조건의 한계를 정정했다. 측정 수치와 최종 판단은 유지했다.
- `migrate_to_server.py`에 Context 별칭 복원·검증을 추가했다. 누락 복원·멱등성·충돌 거부·원본 별칭/대상 누락·반영 실패를 가짜 클라이언트 테스트 5건으로 확인했다. 실제 Docker 재측정은 하지 않았다.
- 분석 캐시 경로를 레포 기준 + `V56_TEST_ANALYSIS_CACHE` 재정의로 바꿨고, 폴백 집계기에 `--input-dir`을 추가해 보존된 1차 결과를 그대로 읽게 했다.
- 필요한 CSV를 `.gitignore` 예외로 명시하고, `logs/`·`ctl/`·`run1_noalias/logs_run1/`·`progress.txt`는 제외했다. 1차 결과와 2차 결과는 구분해 보존한다.
- 보완 검증: 폴백 10건 + 별칭 복원 5건 = **15건 통과**. 지연 182요청(측정 180·준비 2)을 재집계해 오류·timing 미대조 0, 기존 산출물 3개 바이트 동일을 확인했다. 보존된 폴백 CSV 68행도 오프라인 집계했다. 신규 모델·Mongo·Docker 호출은 없었다. 기존 보고의 관련 테스트 499건은 이번 보완에서 다시 실행하지 않았다.
- 서비스 코드·기본값·운영 색인은 변경하지 않았다. 운영 결함 수정 후 n007·t018·q112·n011을 정상 Context 별칭 환경에서 다시 확인하는 일은 후속 작업이다.

## 파일

| | |
|---|---|
| `RUN_INFO.md` | 이 문서(§1~§3 측정 전 기록) |
| `v56_runtime.py` · `v56_app.py` · `start_servers.sh` | 호출 차단 · 캐시 분석기 · 가짜 SDK 클라이언트를 꽂은 실제 리랭커 · 실제 FastAPI 앱 래퍼 · 서버 기동 |
| `tests/fake_gemini_client.py` · `tests/test_v56_rerank_fallback_paths.py` | 가짜 SDK 클라이언트(기본·실패 모드) · 회귀 10건 |
| `tests/test_v56_migration.py` | PR 전 보완: 별칭 복원·실패 판정 테스트 5건 |
| `migrate_to_server.py` · `server_collections.json` · `env_info.json` | 서버 이전·검증(컬렉션 5 · 별칭 2) · 환경 |
| `classify_queries.py` · `query_categories.csv` · `latency_sample.json` | 표본 선정 규칙·범주·합성 open_empty |
| `run_latency.py` · `aggregate_latency.py` · `latency_requests.csv` · `latency_joined.csv` · `latency_per_query.csv` · `latency_summary.csv` · `latency_aggregate_stdout.md` · `latency_runmeta.json` · `timing_{A,D}.jsonl` | 2차 지연 측정(별칭 있음) |
| `run_fallback.py` · `summarize_fallback.py` · `run1_noalias/fallback_results.{csv,jsonl}` · `run1_noalias/fallback_summary.md` · `run1_noalias/fake_calls_{A,D}.jsonl` | 폴백 검증 본 실행(68요청, 1차 조건) |
| `fallback_results_subset_run2.{csv,jsonl}` · `fake_calls_{A,D}.jsonl` | 2차 t014·t049 확인(8요청) |
| `compare_local_server.py` · `local_vs_server_*.json` · `probe_backends.py` · `probe_hybrid.py` · `probe_filters.py` · `trace_queries.py` · `diagnose_mode_diff.py` · `check_scan_limit.py` · `*.json` | 로컬↔서버 대조·원인 진단(별칭 누락 · 상한 5000) |
| `run1_noalias/` | 1차(Context 빔) 지연·폴백·대조 산출물 보존 |
| `logs/` · `run1_noalias/logs_run1/` · `progress.txt` · `ctl/` | 서버·실행 로그(git 제외) |
| `artifacts/qdrant_server_v56/storage` · `artifacts/qdrant_eval_v56_local` | 서버 저장소(컨테이너는 종료) · 로컬 복사본(git 제외) |
