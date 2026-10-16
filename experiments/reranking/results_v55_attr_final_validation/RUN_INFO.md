# results_v55 — 속성 조건 보조 경로 최종 검증: 봉인 v10 test 42건 A/B/D 각 1회 (2026-10-16)

v54(`results_v54_attr_genre_policy`, PR #69로 main `0237118`에 병합)가 dev 176에서 권고한 **D = 보조 경로 켬 + 장르 조건부 정책**을,
사용자가 이번 작업에 한해 허용한 **봉인 v10 test 42건**에서 현행 운영 기준(A)·기존 보조 경로(B)와 함께 **각 1회** 평가한다.
목적은 운영 활성화 여부를 판단할 근거를 만드는 것이고, 운영 설정(`.env`·서비스 기본값·운영 색인)은 바꾸지 않는다.
검색 정책·프롬프트·가중·후보 수·필터 축은 바꾸지 않았다. **아래 §1〜§3은 test 질의·정답·분석 결과를 열기 전에 적었다**(§4 이후가 결과).

## 1. 사전 고정 조건 (test를 열기 전에 기록)

| 항목 | 값 |
|---|---|
| 코드 | main `0237118`(#69 v54 병합 커밋) 위 브랜치 `experiment/attr-final-validation`, 워크트리 `/private/tmp/vague-finder-attr-final-validation`. `src/` 변경 없음(이 폴더의 스크립트·보고서만 추가) |
| 모델 | 질의 분석 `gemini-3.1-flash-lite`(`GEMINI_MODEL_NAME`) · 리랭킹 `gemini-3.1-flash-lite`(`GEMINI_RERANK_MODEL_NAME` 미설정 → 같은 값). 무료 countTokens 확인 OK(2026-10-16 01:40, 두 모델) |
| Vertex | `GEMINI_RETRIEVAL_BACKEND=vertex` · `GCP_PROJECT_ID=project-0a71759e-e61e-4004-b23` · `GCP_LOCATION=global` · 인증은 팀 서비스 계정 키(`GOOGLE_APPLICATION_CREDENTIALS=secrets/vertex-sa.json`, #67). AI Studio·Claude API·Melon 신규 호출 없음(`GEMINI_API_KEY`·`CLAUDE_API` unset) |
| 색인 | 서비스 색인 `artifacts/qdrant`를 파일 복사한 **`artifacts/qdrant_eval_v55`**(워크트리 안, git 제외, 226MB). 전체 지문 `c39aa7eb08fb`(`.lock` 제외, 상대경로+내용 sha256) — 서비스 색인·v50 복사본 `qdrant_eval_{a,b,c}`(v51~v54가 쓴 것)와 **동일**. 컬렉션 5개: text 1024-koe5 · image 768 · audio 512(각 5,143점) + Context dense/sparse(`__generation_f6424b94621a28cf`). `lsof`로 서비스·다른 복사본을 연 프로세스 없음 확인, uvicorn·crawler·다른 평가 프로세스 없음 |
| Context | manifest `status=ok` · `source_scope_complete=true` · 대상 5,139곡 · dense 5,054 · sparse 639 · build_id `f6424b94621a28cf…` · `CONTEXT_WEIGHT` 기본(0.5, 켬) |
| BM25 | `artifacts/bm25_params.json` sha256 앞 12자 `bf606b19ab02`(5,143곡, git 추적본) |
| Mongo 가사 | `vaguefinder.songs` 5,143건, ping OK(0.7초). 내용 지문은 평가기가 runinfo `lyrics_source.content_sha256_16`에 기록 — v50~v54 값 `e0d58f28f1f4a563`과 같아야 하며 다르면 중단 |
| 분석 캐시 | **test 캐시 없음**(`analysis_cache_v10_test*.json` 어디에도 없음, git 이력 없음) → `build_cache.sh`로 **1회** 생성(`analysis_cache_v10_test_vertex.json`, Vertex 분석기, `SEARCH_REFERENCE_YEAR=2026`). 지금 코드의 분석기 지문 = v49 dev 캐시와 동일(`prompt_sha 80c464515d9c` · `postprocess_sha f18778f8f94d` · temperature 0 · top_p 1 · reference_year 2026)이므로 dev 기준선(v50~v54)과 같은 분석 조건이다. 규칙 폴백이 있으면 `--allow-fallback-analysis` 없이 평가기가 멈춘다 — 그 경우 **채우지 않고 보고** |
| 리랭커 | `gemini_listwise`(1패스 · Search 끔 · `GEMINI_RERANK_RARE_FACT_VERIFY=0` · rerank_weight 0.85 · max_candidates 30 · max_retries 3 · request_timeout 20s · time_budget 30s) — v49~v54와 동일. 리랭킹 정책 생략(표지 지배·자료 없는 MV 장면)은 서비스 규칙 그대로(`policy_skipped`, 실패 아님) |
| 그 밖 | `AUDIO_FUSION_SCALE=1` · `SEARCH_REFERENCE_YEAR=2026` · `TOKENIZERS_PARALLELISM=false` · top-k 10 · candidate-k 30 · path-k 기본 · explain 켬. `.env`는 메인 체크아웃 것을 복사하고 `QDRANT_PATH`만 복사본으로 바꿨다 |

### A/B/D 구성 (그 밖의 조건은 전부 동일)

| 구성 | `ATTR_FILTER_WEIGHT` | `ATTR_FILTER_GATE` | `ATTR_FILTER_K` | `ATTR_FILTER_USE_GENRE` | `ATTR_FILTER_GENRE_POLICY` | 뜻 |
|---|---|---|---|---|---|---|
| **A** | 0 | era_plus | 30 | 1 | always | 현행 운영 기준(보조 경로 끔 — 코드 기본값과 같음: 가중 0 · era_plus · 30 · 1 · always, 모델 없는 스위치 해석으로 확인) |
| **B** | 0.5 | era_plus | 30 | 1 | always | 기존 보조 경로(v51·v52·v53 B) |
| **D** | 0.5 | era_plus | 30 | 1 | conditional | 최종 후보(v54 권고 정책) |

실행 순서 **A → B → D**, 각 1회, **순차 단일 프로세스**(색인 복사본 하나를 한 번에 한 프로세스만 연다), 세 설정 모두 같은 test 캐시.
실행 스크립트 `run_eval.sh`(v54 `run_eval.sh`에서 split·캐시·출력 경로만 바꿈), 출력 `A_v10_test/` · `B_v10_test/` · `D_v10_test/`, 호출 기록 `usage_<실행>.jsonl`(`usage_tap.py`, 본문 없음).

### 봉인 test의 상태 (열기 전 확인)

- `experiments/reranking/eval_queries_v10.csv`의 `split=test` 42행(ID 접두 `t`, `query_type=search`). PR #41(`7eb7be0`)에서 dev 60과 함께 커밋됐고 그 뒤 수정 이력 없음.
- **평가 이력 없음**: `experiments/reranking` 전체의 runinfo 중 v10 입력 + `split=test`인 것 없음, v10 test 분석 캐시 없음. v38~v54의 RUN_INFO는 모두 "봉인 test는 열지 않았다"로 기록. (v09 test 38건은 v34에서 이미 열렸으므로 이번 대상이 아니다.)
- 정본 `docs/eval/queries.json`(0.6.0)에는 v10이 반영돼 있지 않아 라벨은 CSV의 `relevant_ids`뿐이다. 허용 정답(relaxed) 라벨은 v10 test에 없으므로 **엄격 지표만** 낸다.
- 이 평가 뒤 이 42건은 **더 이상 미사용 test가 아니다.** 이후 개선 비교에는 새 독립 검증 자료가 필요하다.

### 예산 · 장애 처리 규칙 (사전 산정, 캐시 할인 없음, Google Gemini API 가격표 표준 등급 — 입력 0.25 · 출력 1.50 USD/1M, Vertex 가격 페이지·실제 청구액은 미확인)

| 항목 | 근거 | 추정 |
|---|---|---|
| test 캐시 생성 42건 | v49: 176건 179호출 입력 1,402,559 · 출력 87,085 → $0.48 → 질의당 ≈ $0.0027 | **≈ $0.12**(재시도 여유 포함) |
| 리랭킹 1회 42건 × 3 | v54 B1: 176건 163호출 입력 1,714,344 · 출력 226,473 → $0.768 → 질의당 ≈ $0.0044 | **≈ $0.18 × 3 = $0.55** |
| 계획 합계 | | **≈ $0.67** |
| 복구 여유 | 실패한 최소 실행 단위(설정 1개 42건) 1회 | ≤ $0.18 → 상한 ≈ $0.85 |
| 한도 | 사용자 지정 | **$2.00** — usage 기록 누계가 넘으면 즉시 중단 |

- 자동 재시도는 기존 클라이언트 설정 그대로(SDK 재시도 + 리랭커 max_retries 3, 질의당 30초 예산). 그래도 결과가 `exception`·`fallback`·`partial_failure`면 그 실행은 **유효 비교가 아니다** — 산출물을 `failed_<실행>/`로 보존하고, 같은 조건으로 **그 설정(42건) 1회만** 복구 실행(`TAG=<설정>_v10_test_retry`). 복구 뒤에도 실패가 남거나 예산을 넘기면 중단하고 보고한다. 품질 판정은 세 설정이 모두 유효(실패 0)할 때만 내린다.
- 평가 중 구현 결함이 드러나면 봉인 평가를 멈추고 결함·영향을 보고한다. 고친 뒤 같은 test를 반복해 정책을 고르지 않는다.
- 호출 0이어야 하는 단계(스모크·집계·테스트)는 usage 파일이 생기지 않아야 한다.

## 2. 판정 기준 (결과를 보기 전에 고정, 결과를 본 뒤 바꾸지 않는다)

1. **D의 Hit@10 ≥ A의 Hit@10 그리고 D의 Hit@10 ≥ B의 Hit@10** 이면 활성화 권고 후보로 남긴다.
2. D의 Hit@1도 A·B 모두 이상인지 확인한다.
3. 어느 기준(1·2)이든 낮아지면 자동 채택하지 않고 **판단 보류**로 보고한다.
4. 기준을 통과해도 42건 표본이므로 **일반화 성능이 입증됐다고 쓰지 않는다.**
5. MRR@10 · Hit@5 · 질의별 손실(A→D, B→D) · 지연을 함께 보고하고, 기준 통과만으로 심각한 개별 회귀(예: 1위 → 10위 밖)를 무시하지 않는다.
6. 기록 지표: 후보 Recall@30 · 리랭킹 전 Hit@10 · 최종 Hit@1/5/10 · MRR@10 · A→D·B→D 질의별 이득/손실 · 후보 구성 변화(후보 지문)와 동일 후보에서의 리랭킹 변동 · 정책 적용 사유별 건수(`attr_filter_genre.reason`) · 지연(첫 질의·Mongo 가사 판 적재와 준비된 상태 분리) · 모델 호출·재시도·실패·폴백·토큰·추정 비용.
7. 최종 보고 형식: **운영 활성화 권고 / 판단 보류 / 비권고** 중 하나 + A·B 각각 대비 D의 변화 + 기준 충족 여부 + 손실 사례와 해석 한계 + 비용·실패·테스트 + 다음 검증.

## 3. 봉인 평가 전 준비 (test 미사용 — dev 자료·합성 입력만)

- **단위 테스트**(모델 자격 제거 `env -u GCP_PROJECT_ID -u GOOGLE_APPLICATION_CREDENTIALS -u CLAUDE_EFFORT GEMINI_RETRIEVAL_BACKEND=`): `tests/test_attr_filter_path.py` · `tests/test_analysis_cache.py` · `tests/test_rerank_policy.py` **359 통과**(9.4초, 실패 0). usage 파일 생성 없음.
- **기본값·스위치 우선순위**(모델 없음, `search_router` 함수 직접 호출): 환경변수 없음 → 가중 0 · era_plus · k 30 · use_genre True · always(= A). A/B/D 환경변수 → 각각 기대값(effective always/always/conditional).
- **분석 캐시 호환성**: 지금 코드의 `analyzer_fingerprint()`가 v49 Vertex 캐시 meta와 `prompt_sha`·`postprocess_sha`·온도·기준 연도 모두 일치(모델 자격을 뺀 확인이라 backend만 `none`으로 나왔고, 실행 환경에서는 vertex/global). dev 스모크의 runinfo `fingerprint_drift`로 재확인(§3-1).
- **failed·partial_failure·폴백 판정**: 평가기 `rerank_outcome` — `exception`(예외→검색 순서) · `fallback`(호출 있었으나 반영 0) · `partial_failure`(일부 그룹 실패) 는 실패, `applied` · `skipped`(부를 조건 아님) · `policy_skipped`(정책 생략) 는 정상. detail `rerank_fell_back` · `rerank_partial_failure`, summary `Rerank fallback query count` · `Rerank partial failure query count`로 집계되고 `aggregate.py`가 실패 질의 ID를 적는다.
- **접근 확인**: Mongo ping OK · 5,143건, Vertex 두 모델 countTokens OK, 색인 복사본 지문 일치(위 표).
- **격리**: 색인 복사본 전용(`QDRANT_PATH=artifacts/qdrant_eval_v55`), 출력은 이 폴더, 원본 색인·기존 결과·v49 dev 캐시는 읽기만.
- **호출·비용 기록**: `usage_tap.py`(v50~v54와 같은 파일, 토큰·지연·`model_version`·오류만) + `usage_summary.py`.

### 3-1. dev 스모크 (모델 없음, `RERANKER_ENABLED=false`, v49 v10 dev 캐시, 질의 t001·t010·t026)

`run_eval.sh`로 D(01:43:46~01:45:28) → A(01:45:28~01:47:07) 순차 실행(`smoke_D_v10_dev/` · `smoke_A_v10_dev/`). 둘 다 exit 0, usage 파일 생성 없음(호출 0).
- runinfo: `split=dev` · corpus `artifacts/qdrant_eval_v55`(5,143점) · BM25 `bf606b19ab02` · Mongo 가사 `e0d58f28f1f4a563`(5,143건) · `fingerprint_drift=[]` · 폴백 0 · `gemini.backend=unused`(리랭커 끔) · 스위치 A `attr_filter_weight 0 / always`, D `0.5 / conditional`(effective 동일).
- **후보 지문이 v54와 3/3 동일**: D(t001 8 · t010 8 · t026 21)는 v54 `cand_attr0.5_eraplus_condgenre`와, A(t001 밖 · t010 9 · t026 18)는 v54 `cand_attr0_eraplus`와 `candidate_digest`·후보 순위가 같다 — 색인 복사본·BM25·Mongo·캐시·코드가 v54와 같은 조건임을 확인. D의 explain `attr_filter_genre`는 3건 모두 `gender_condition`(뺄 장르 조건 있음).
- 지연: 첫 질의 8.1 · 9.9초(모델·Qdrant 적재), 준비된 상태 0.5~1.2초. 종료 때 `QdrantClient.__del__`의 `ImportError: sys.meta_path is None` 메시지는 v54 로그(`cand_attr0_eraplus_*.log`)에도 있는 인터프리터 종료 시점 메시지로 결과와 무관.
- `aggregate.py` 확인: A vs D 스모크에서 스위치 차이·조건 일치·쌍 비교·사유·지연 표가 나왔고, detail의 결과 코드를 `partial_failure`·`exception`으로 바꾼 **합성 입력**에서 "유효하지 않은 결과 질의"에 해당 ID가 올라오는 것을 확인(저장 결과 아님).

**→ 여기까지 test 질의·정답·분석을 열지 않았다. 아래 §4부터는 봉인 평가 결과다.**

## 4. 봉인 평가 실행 (`run_pipeline.sh`: 캐시 생성 → A → B → D, 2026-10-16 01:47:57~02:22:50, 순차 단일 프로세스)

- **test 분석 캐시** `analysis_cache_v10_test_vertex.json`: 42건 · 규칙 폴백 0 · 지문 v49 dev 캐시와 동일(`prompt_sha 80c464515d9c` · `postprocess_sha f18778f8f94d` · vertex/global · 2026). 호출 45(질의 42 + 재시도·검증 3) · 실패 0 · 입력 354,604 · 출력 21,834 · 분석 중앙값 2.15초 · 추정 **$0.121**. 내용을 보고 프롬프트·정책을 고치지 않았다(캐시 생성 뒤 바로 A 실행).
- **A → B → D**: 각 exit 0, runinfo 세 실행 모두 `split=test` · corpus `qdrant_eval_v55`(5,143) · BM25 `bf606b19ab02` · Mongo `e0d58f28f1f4a563` · 같은 캐시 · `fingerprint_drift=[]` · 폴백 0 · `gemini.backend=vertex/global`. 스위치 차이는 A↔B `attr_filter_weight 0→0.5`, B↔D `attr_filter_genre_policy always→conditional`뿐(aggregate 조건 대조).
- **리랭킹 결과**: 세 실행 모두 **applied 39 · policy_skipped 3(t091 · t093 · t095, `image_dominant` — 서비스 규칙 그대로) · exception/fallback/partial_failure 0**. 세 실행 모두 유효 → 복구 실행 없음, 품질 판정 가능.
- 로그의 `QdrantClient.__del__` ImportError는 종료 시점 메시지(§3-1)로 세 실행 모두 1회씩, 결과와 무관.

## 5. 결과 — test 42건 (`aggregate.py` → `aggregate_stdout.md` · `v10_test_summary.csv` · `v10_test_per_query.csv`)

### 5-1. 지표 (엄격 라벨, 허용 정답 없음)

| 구성 | 후보 30 안 | 리랭킹 전 Hit@10 | **Hit@1** | **Hit@5** | **Hit@10** | **MRR@10** | 리랭킹 결과 | generate 호출 · 실패 |
|---|---|---|---|---|---|---|---|---|
| A 끔(현행) | 25/42 | 18 | 15 | 22 | **22/42 (52.4%)** | 0.421 | applied 39 · policy_skipped 3 · failed 0 | 39 · 0 |
| B always | 25/42 | 18 | 15 | 22 | **22/42 (52.4%)** | 0.421 | 같음 | 39 · 0 |
| **D conditional** | 25/42 | 18 | **15** | **22** | **22/42 (52.4%)** | **0.421** | 같음 | 39 · 0 |

**세 구성의 42건 정답 순위가 전부 같다**(A→D 같음 42 · 나음 0 · 나쁨 0, B→D 같음 42 · 0 · 0, A→B도 같음 42). 10위 안에 든/빠진 질의 0, 안에서 오름/내림 0.
리랭킹 전 Hit@10은 세 구성 모두 18건이지만 **검색 순위는 동일하지 않다**. 상위 10위 기준 기록에서 A→D는 t014 1→10, t023 5→6, t051 4→6으로 달랐고 나머지 39건은 같았다(10위 밖은 상세 순위를 구분하지 않는 기록). 최종 top-10 목록이 글자까지 같은 질의는 A=B=D 34/42(A=B 34, B=D 36)다.

### 5-2. 후보 구성 변화와 동일 후보에서의 리랭킹 변동

- **후보 지문**(후보 30 집합): A↔B 다른 질의 8, B↔D 다른 질의 6(t005 · t014 · t020 · t023 · t049 · t050), A↔D 다른 질의 8. **후보 집합이 달라진 질의에서도 정답의 최종 순위는 모두 같았다**(B→D "지문 다른데 순위 같음" 6건).
- 후보 지문이 같은 질의에서 리랭킹 변동(정답 순위 또는 top-10 글자 차이): **0건** — 이번 세 실행 사이에 Vertex 반복 변동은 관측되지 않았다(dev v54에서는 t010 비정답 순서 1건 있었음).
- 후보 30 새로 든/빠진 정답: 0 / 0. 후보 밖 정답 17건(t006 · t013 · t030 · t035 · t045 · t048 · t050 · t052 · t061 · t066 · t067 · t069 · t072 · t076 · t081 · t090 · t094)은 세 구성 모두 밖 — 보조 경로가 회수한 정답은 **test에서 0건**이다(dev 176에서는 A→D 후보 +8).

### 5-3. 보조 경로·정책 적용 사유 (explain, D의 `attr_filter_genre`)

| 구분 | 건수 |
|---|---|
| 보조 경로가 열린 질의(시기 창 유효 + 성별/형태/장르 축 ≥1) — D 판정 기록 있음 | **9/42** (t005 · t014 · t020 · t023 · t047 · t049 · t050 · t051 · t054) |
| D 사유 `gender_condition`(성별 조건 있어 최초 장르 제외) | 8 — 그중 실제로 뺄 장르 조건이 있던 질의 **7**(t005 · t014 · t020 · t023 · t047 · t049 · t050), 장르 없던 1(t051) |
| D 사유 `no_gender`(성별 없어 장르 유지) | 1 (t054) |
| `answered_genre` | 0 (1턴 평가) |
| 보조 목록 기여(explain 경로 `attr_filter`)가 있는 질의 | B 7 · D 8 (t047은 둘 다 0곡 — 필터에 맞는 곡 없음) |
| 정답이 보조 목록에 | B 4(t005 · t014 · t020 · t054) · D 3(t005 · t020 · t054) |

B/D의 필터가 실제로 다른 질의 7건(뺄 장르 있던 `gender_condition`) 가운데 후보 지문이 달라진 것은 6건이고, 정답 순위는 모두 같았다.

### 5-4. 질의별 — 최종 순위 손실은 없으나 후보 순위는 하락

A→D 후보 순위 하락은 **4건**(t014·t049·t023·t051), B→D는 **2건**(t014·t049)이다. 후보 30 밖으로 이탈한 정답은 없고 최종 순위도 모두 유지됐다.

| 질의 | 유형 | 후보 A→B→D | 리랭킹 전 | 최종 | 설명 |
|---|---|---|---|---|---|
| **t014** | lyrics | **1 → 2 → 10** | 1 → 2 → 10 | **1 · 1 · 1** | 분석: 2000~2003 · 남성 · 랩/힙합. B에서는 정답이 보조 목록(10곡)에 들었고, D는 장르를 빼자 보조 목록(9곡)에서 정답이 빠지며 다른 곡들이 가산을 받아 **후보·검색 순위가 1위에서 10위로** 내려갔다. 최종은 리랭커가 세 구성 모두 1위로 — 단순히 검색 순서로 돌아간다면 D 10위·A 1위가 예상된다. **실제 서비스의 실패·폴백 반환 순서는 이번에 검증하지 않았으므로 후속 확인이 필요하다** |
| t049 | mixed | 16 → 18 → 22 | 밖 | 1 · 1 · 1 | 2011 · 혼성 · 랩/힙합. 보조 곡 B 2 · D 5, 정답은 둘 다 보조 목록 밖. 후보 순위만 내려가고 리랭커가 1위로 |
| t023 | mixed | 5 → 6 → 6 | 5 → 6 → 6 | 1 · 1 · 1 | 2010~2019 · 여성 · 록. B 보조 0곡 → D 2곡(장르 제외로 열림), 정답은 보조 밖 |
| t051 | mixed | 4 → 6 → 6 | 4 → 6 → 6 | 1 · 1 · 1 | 1990년대 · 혼성, 장르 없음 → B = D 필터. 보조 1곡 |
| t005 · t020 · t054 | mixed | 1·2·1 유지 | 유지 | 1 · 2 · 1(세 구성 같음) | 정답이 보조 목록에 있었으나 이미 1~2위라 변화 없음 |
| t050 | mixed | 밖 · 밖 · 밖 | 밖 | 밖 | 2000 · 여성 · 댄스. 보조 2곡이 들어왔지만 정답은 보조 목록에도 후보에도 없음 |
| t047 | mixed | 4 · 4 · 4 | 4 | 1 | 1990년대 · 혼성 · 댄스 — 필터에 맞는 곡 0, B = D = A |

### 5-5. 지연 (detail `retrieval_ms` · `rerank_only_ms`, 순차 단일 프로세스)

| 실행 | 첫 질의 t002(가사 질의, Mongo 가사 판 적재) | 준비된 상태 검색 중앙값 / p95 / 최대 (41건) | D에서 보조 후보가 반환된 동일 8건의 검색 중앙값 / 최대 | 리랭킹 중앙값 / p95 (39건) | 전체(적재 포함) 질의당 중앙값 / p95 / 42건 합계 |
|---|---|---|---|---|---|
| A | 352.4초 | 0.59 / 1.49 / 3.87 | 0.65 / 2.69 | 4.82 / 5.72 | 5.36 / 7.30 / 9.5분 |
| B | 351.2초 | 0.58 / 1.70 / 4.08 | 1.04 / 3.17 | 4.99 / 6.02 | 5.66 / 7.72 / 9.6분 |
| D | 351.1초 | 0.57 / 1.67 / 4.21 | 1.02 / 3.13 | 4.84 / 5.96 | 5.59 / 7.85 / 9.5분 |

- 첫 질의 t002는 가사 질의라 임베딩·Qdrant 적재(8~10초)에 더해 전곡 가사 판 적재(≈350초)가 겹쳤다 — v51~v54와 같은 적재이며 보조 경로와 무관(A도 같다). 그 밖에 5초 초과 검색 없음.
- 준비된 상태 41건의 검색 중앙값 차이는 ±0.02초 안이다. 표의 8건은 **D에서 보조 후보가 실제 반환된 질의**를 기준으로 고른 공통 집합이다(A 0.65초 · B 1.04초 · D 1.02초).
- **경로가 열린 전체 9건**에는 후보를 반환하지 못한 t047도 포함된다. 이 동일 9건의 검색 중앙값은 **A 0.65초 · B 1.17초 · D 1.14초**로 D가 A보다 약 0.49초 길었다. 첫 질의 t002는 이 9건에 속하지 않는다. 경로 열림과 실제 후보 반환을 구분해야 한다.
- 리랭킹 중앙값은 세 실행 4.8~5.0초였다. 이 지연 비교는 같은 PC에서 설정별 1회 측정한 값이며 실제 서버 성능이나 차이의 원인을 확정하는 근거는 아니다.

## 6. 비용 · 호출 · 장애 (`usage_summary.py`, 기록된 캐시 할인 적용)

| 묶음 | generate 호출 | count_tokens(무료) | 실패 | 입력 | 출력 | 추정 비용 |
|---|---|---|---|---|---|---|
| 캐시 생성 | 45 | 1 | 0 | 354,604 | 21,834 | $0.121 |
| A | 39 | 1 | 0 | 431,056 | 57,457 | $0.194 |
| B | 39 | 1 | 0 | 431,419 | 57,957 | $0.190 |
| D | 39 | 1 | 0 | 431,069 | 57,753 | $0.190 |
| **합계** | **162** | 4 | **0** | 1,648,148 | 195,001 | **$0.695** (한도 $2, 사전 추정 $0.67) |

- 기록된 캐시 입력은 **40,592토큰**이다. 위 비용 표는 이 입력에 캐시 단가를 적용한 값이며 **할인 전 $0.704539(약 $0.705), 할인 적용 $0.695405(약 $0.695)**다. 두 추정 모두 한도 $2 이내다.
- 기록된 HTTP 실패 0, 리랭킹 재시도 0(설정별 호출 39 = applied 39, 하네스 2회 실행 질의 0). 분석 캐시는 42질의에 45호출이었으므로 전체 작업의 재호출까지 0이라고 표현하지 않는다. 복구 실행 0. 실패 비용 0.
- 단가는 Gemini API 가격표 표준 등급 가정이고 **Vertex 실제 청구액은 확인하지 않았다**(v49~v54와 같은 가정). 모든 생성형 호출은 Vertex(`model_version gemini-3.1-flash-lite`), AI Studio·Claude·Melon 호출 0.
- 호출 0 단계(스모크·테스트·집계)는 usage 파일이 생기지 않았다.
- 테스트: §3(359 통과). `src/` 변경이 없어 전체 `tests/`는 다시 돌리지 않았다.

## 7. 판정 — 사전 기준(§2) 적용

| 기준 | 결과 | 충족 |
|---|---|---|
| 1. D Hit@10 ≥ A 그리고 ≥ B | 22 = 22 = 22 | **충족(동률)** |
| 2. D Hit@1 ≥ A 그리고 ≥ B | 15 = 15 = 15 | **충족(동률)** |
| 5. MRR · Hit@5 · 질의별 손실 · 지연 | MRR 0.421 = = · Hit@5 22 = = · 최종 순위 손실 0/42 · 검색 중앙값 A→D +0.49초(경로 열린 동일 9건) | 최종 순위 회귀 없음. 후보 순위 하락 A→D 4건, B→D 2건; 폴백 시 영향 미검증 |

**결론: 사전 품질 기준을 동률로 충족해 활성화 권고 후보로 남긴다. test 42건에서 D의 최종 순위 이득과 손실은 모두 0이다.** 이는 이번 표본에서 관측한 결과이며 일반적인 비열등성이나 우월성을 입증한 것은 아니다. 개선 근거는 현재 dev 176의 A 117→D 125(v54)에 한정된다.
이번 test에서는 보조 경로가 후보 밖 정답을 회수하지 못했다. 경로가 열린 9건 중 정답이 후보 밖이던 것은 t050 하나였고, 그 정답도 보조 목록에 없었다. 이 사실만으로 dev에서 이득을 본 질의 유형이 test에 없었다고 단정하지 않는다.

- **운영 판단: 활성화 결정은 추가 검증 전까지 보류한다.** 사전 Hit@10·Hit@1 기준을 변경한 것이 아니라, 함께 검토하기로 한 지연과 후보 순위 하락을 운영 판단에 반영한 것이다. 경로 열린 9건의 추가 검색 지연과 폴백 시 반환 순서를 실제 서비스 경로에서 확인한 뒤 활성화를 결정한다.
- 서비스 기본값·`.env`·운영 색인은 변경하지 않았다. dev와 test는 서로 다른 질의 집합이므로 71.0%→52.4%를 정책 때문에 성능이 떨어진 것으로 해석하지 않는다.
- A 대비 D: Hit@1/5/10 · MRR 변화 0, 후보 지문 변화 8건, 최종 순위 변화 0.
- B 대비 D: 변화 0, 후보 지문 변화 6건, 정답이 보조 목록에 든 수 4 → 3(t014).

## 8. 한계 · 검증하지 않은 것 · 다음 검증

1. **42건 표본**이고 세 구성이 전부 같은 결과라 일반화 성능은 입증되지 않았다. 동률은 "차이 없음의 증명"도 아니다(열린 질의 9건, 필터가 실제로 다른 7건의 작은 분모).
2. **이 42건은 이제 열린 자료다.** 앞으로 미사용 test라 부르지 않는다. 이 세트를 기준으로 정책·가중·필터 축을 더 튜닝하지 않는다. 추가 개선 비교에는 새 독립 검증 자료가 필요하다.
3. 1턴 평가만이다. 재질문 2턴은 v54의 영향 18건(v06·v09)만 봤고 v10은 2턴 평가기 미지원 — 전체 재질문 검증이 아니다. 성별 기억 오류(성별 조건이 틀렸을 때 장르를 빼는 손해)는 dev·test 어느 쪽에도 사례가 없어 미검증.
4. 로컬 Qdrant 복사본 측정이라 실제 서버 모드 Qdrant의 필터 검색 성능은 미검증. 지연 수치는 이 PC·이 시간대 값.
5. 허용 정답(relaxed) 라벨이 v10에 없어 엄격 지표만. 정본 `queries.json` 0.7.0 반영은 별도 작업.
6. 비용은 가격표 가정, 청구액 미확인.
7. **후속 검증(이 test로 정책 튜닝하지 않음)**: ① 모델 실패를 모의해 실제 서비스의 폴백 반환 순서를 확인한다(t014 등 후보 순위 하락 사례). 현재는 정상 실행의 후보 순위로 추정했을 뿐 실패 시 반환값을 측정하지 않았다 ② 실제 서버 Qdrant에서 보조 경로의 추가 검색 지연을 확인한다 ③ 정책을 더 개선한다면 새 독립 자료를 사용하고 질의 유형의 분포를 기록한다. 이번 test 결과에 맞춰 가중·필터·보호 규칙을 조정하지 않는다.


## 재현 명령

```
# 워크트리 /private/tmp/vague-finder-attr-final-validation (main 0237118, src 변경 없음), 색인 복사본 artifacts/qdrant_eval_v55
OUT=experiments/reranking/results_v55_attr_final_validation
$OUT/run_pipeline.sh                      # 캐시 1회 생성(build_cache.sh) → CONFIG=A/B/D run_eval.sh
venv/bin/python $OUT/aggregate.py         # 모델·DB 없음 → aggregate_stdout.md · v10_test_summary.csv · v10_test_per_query.csv
venv/bin/python $OUT/usage_summary.py     # 호출·토큰·추정 비용
```

## 파일

| | |
|---|---|
| `RUN_INFO.md` | 이 문서(§1·§2·§3는 test를 열기 전 기록) |
| `run_eval.sh` · `build_cache.sh` · `run_pipeline.sh` | 실행 스크립트(v54 run_eval.sh에서 split·캐시·출력만 변경) |
| `analysis_cache_v10_test_vertex.json` | test 42건 분석 캐시(1회 생성, 폴백 0) |
| `A_v10_test/` · `B_v10_test/` · `D_v10_test/` | summary · top10 · runinfo · conditions(커밋) / detail · explain · lyric_boosts(git 제외) |
| `smoke_A_v10_dev/` · `smoke_D_v10_dev/` | dev 3건 모델 없는 스모크(test 열기 전) |
| `aggregate.py` · `aggregate_stdout.md` · `v10_test_summary.csv` · `v10_test_per_query.csv` · `smoke_dev_*.csv` | 집계(모델 없음) |
| `usage_tap.py` · `usage_summary.py` · `usage_*.jsonl` | 호출 기록(본문 없음)·비용 집계 |
| `progress.txt` · `*.log`(git 제외) | 진행·실행 로그 |
