# v60 — Context 색인·별칭 기동 검사 (2026-10-16 밤)

워크트리 `/private/tmp/vague-finder-context-startup-check`, 브랜치 `fix/context-startup-check`(main `f4e87ff` = #74 v59 병합 기준), **커밋 전**.
모델(Vertex·AI Studio·Claude)·Mongo·운영 Qdrant·멜론 호출 **0**. 검색 결과·가중·후보 정책·프롬프트·`HYBRID_SCAN_LIMIT`·속성 보조 경로는 바꾸지 않았다.

## 1. 문제와 재현

**문제(v56 §4-1·§7-⑥).** 서버 Qdrant로 옮기면서 컬렉션만 옮기고 Context **별칭 2개**(`vaguefinder-context-dense-1024-koe5__dev` · `vaguefinder-context-sparse-bm25__dev`)를 옮기지 않았다. 서비스는 정상으로 떴고 검색마다 로그에 `[SearchRouter] 배경 지식 경로 실패 (빈 결과로 대체): published context Qdrant aliases are missing`만 남긴 채 Context 없이 돌았다. 응답만 봐서는 모르니 1차 측정 전체를 Context 빠진 조건으로 쟀다.

**왜 그랬나(현재 계약).**
- `ContextQdrantSearch._snapshot()`이 별칭·상태 manifest를 확인하지만 **검색 시점**에, 그것도 `search*()`가 불릴 때마다 확인한다. 별칭·manifest가 **둘 다** 없는 미발행 환경은 `None` → 빈 결과(정상), manifest가 있는데 별칭이 없거나 하나만 있으면 `RuntimeError`.
- 라우터 `_path_result`는 경로 하나의 예외를 **빈 결과 + 경고 로그**로 바꾼다(의도된 계약 — 한 경로가 죽어도 검색은 계속돼야 한다). Context 예외도 이 길을 탄다.
- `lifespan`은 Vertex 설정 오류는 기동에서 멈추지만 Context는 보지 않았다. 벡터 DB 초기화 실패는 경고 + 기동.
- 서비스가 Context를 실제로 쓰는 조건은 라우터의 `context_search is not None and _context_weight > 0`이고, 팩토리는 늘 `ContextQdrantSearch`를 만들므로 사실상 **`CONTEXT_WEIGHT > 0`**(기본 0.5, 메인 `.env`에 값 없음 → 켜짐)이다.

**재현(`repro_context_startup.py`, 수정 전 = main `f4e87ff` 체크아웃에서 실행 → `repro_before.json`).** 합성 Context(곡 2 · 4차원 가짜 임베더)를 임시 폴더 로컬 Qdrant에 정규 경로(`load_context_qdrant`)로 발행한 뒤 저장소를 망가뜨리고 실제 `src.backend.main.app`을 TestClient로 띄웠다(Vertex 확인 대체 · 피드백 색인/작업 끔 · Mongo 없음). **9개 시나리오 전부 `started=True`**, `/health`에 Context 항목 없음. 같은 저장소에서 검색기를 직접 부르면 `RuntimeError`(라우터가 빈 결과로 바꾸는 바로 그 예외).

## 2. 수정

| 파일 | 변경 |
|---|---|
| `src/retrieval/context_qdrant_search.py` | `ContextPublicationError(RuntimeError)` + `reason` 코드 — `_manifest`·`_snapshot`·dense/sparse artifact 해시 불일치가 이 형으로 올라간다(RuntimeError 하위라 기존 검색 경로·테스트 계약 그대로). `_snapshot`은 비교 전에 **중첩 자료형**을 검증한다(`qdrant`·`source_hashes`가 객체인지, `context_build_id`·두 해시가 비어 있지 않은 문자열인지 → 아니면 `manifest_invalid`; dense manifest의 `embedding` 블록도 같게). **`verify_publication()`** 추가: `_snapshot()` 재사용 → `None`(미발행)은 `not_published` 오류 → 두 별칭 대상 컬렉션의 **존재**(`collection_exists`)와 **벡터 설정**(`collection_config_matches`) → 로컬 dense manifest 해시·차원 → `_query_encoder()`(sparse artifact 해시, 기존 검사). 모델을 만들지 않는다(`_text_embedder` 미호출). 연결 장애는 클라이언트 예외형 그대로 올라간다 |
| `src/vector_db/context_qdrant.py` | 발행 도구의 `_collection_config_matches`를 공용 `collection_config_matches(raw_client, …)`로 노출(발행기는 위임). 검사 규칙 중복 구현 없음 |
| `src/backend/context_startup.py` (신규) | `run(vector_client)`: `load_context_settings().weight > 0`일 때만 검사. 상태 `disabled` / `verified` / `config_error`(→ `ContextStartupError`로 기동 실패) / `unavailable`(연결 장애·벡터 클라이언트 없음 — 확인 **불가**, 확인 완료 아님). `status()`는 저장값만 돌려준다(Qdrant 재조회 없음). 설정값 오류(`CONTEXT_WEIGHT=abc`)는 `settings_invalid`로 기동 실패(전에는 첫 검색에서 500) |
| `src/backend/main.py` | lifespan: 벡터 클라이언트 생성 직후·**예열/색인 작업/피드백 스레드보다 먼저** `context_startup.run(vector_client)`. 검사에서 **어떤 예외**가 올라와도(`ContextStartupError`뿐 아니라 예상 밖 예외까지) `close_vector_client()`로 클라이언트를 닫고(로컬 Qdrant 잠금 해제) 원래 예외를 다시 올린다. `/health`에 `context` 추가(`state`·`source="startup"`·`checked_at`·`note`·`reason`·`publication`/`error_type`) |
| `docs/namuwiki_context_vector_db.md` | "기동 검사" 절 — 조건별 기동 동작 표, 확인하는 것/하지 않는 것, 사유 코드별 운영자 확인 지점(자동 수정 없음) |
| `docs/vector_backend.md` · `.env.example` | 서버 이전 시 별칭 동반 이동 · `CONTEXT_WEIGHT` 주석 |
| `tests/test_v60_context_startup_check.py` (신규) | 49건(§6) |

바꾸지 않은 것: `ContextQdrantSearch.search*()`의 미발행 환경 빈 결과 계약, 라우터의 경로 실패 → 빈 결과 계약, `_path_result`, 가중·후보 수·프롬프트, 평가기. 별칭 생성·복구·재발행은 하지 않는다(문서로만 안내).

## 3. 수정 전후 기동 동작 (`repro_before.json` ↔ `repro_after.json`, 같은 스크립트)

| 시나리오 | 환경 | 수정 전 기동 | 수정 후 기동 | 수정 후 `/health.context` 또는 사유 | 실패 뒤 저장소 재개방 |
|---|---|---|---|---|---|
| ok | `CONTEXT_WEIGHT=0.5`, 정상 발행 | 뜸 | 뜸 | `verified` / `ok` + `publication`(별칭·컬렉션·build id·차원) | — |
| aliases_missing | 별칭 2개 삭제(manifest 있음) | 뜸(검색마다 경고·빈 Context) | **실패** | `ContextStartupError` / `aliases_missing` | 가능 |
| alias_partial | sparse 별칭만 삭제 | 뜸 | **실패** | `aliases_partial` | 가능 |
| alias_wrong_target | dense 별칭을 다른 컬렉션으로 | 뜸 | **실패** | `publication_mismatch` | 가능 |
| manifest_missing | 상태 manifest 삭제(별칭 있음) | 뜸 | **실패** | `manifest_missing` | 가능 |
| manifest_corrupt | manifest를 JSON 아님으로 | 뜸 | **실패** | `manifest_invalid` | 가능 |
| manifest_mismatch | `dense_vector_name` 변경 + 해시 재계산 | 뜸 | **실패** | `publication_mismatch` | 가능 |
| disabled_aliases_missing | `CONTEXT_WEIGHT=0` + 별칭 삭제 | 뜸 | 뜸 | `disabled` / `context_weight_zero`(Qdrant 조회 없음) | — |
| connection_failure | `QDRANT_URL=http://127.0.0.1:1` | 뜸 | 뜸 | `unavailable` / `backend_error`, `error_type=ResponseHandlingException`(URI 없음) | 해당 없음 |

수정 전은 아홉 사례 모두 "뜸 + `/health`에 Context 항목 없음"이었다. 수정 후 설정 오류 6종은 기동 실패이며, 실패 뒤 같은 프로세스에서 저장 폴더를 다시 열 수 있다(`storage_reopenable_after=true`). 테스트 `test_lifespan_fails_on_missing_aliases_releases_the_lock_and_can_restart`는 같은 프로세스에서 별칭을 복구한 뒤 **다시 기동 → `verified`**까지 확인한다.

같은 저장소의 **검색 경로**는 수정 전후 똑같이 동작한다(`search_probe`): 정상이면 결과, 망가진 저장소면 예외(라우터가 빈 결과로 바꿈). 이번 수정은 그 예외를 기동 경계에서 먼저 드러낼 뿐이다.

## 4. 검사하는 것 · 검사하지 않는 것

**검사(메타데이터 읽기만)**: ① dense/sparse 별칭 2개의 존재 ② 각 별칭이 가리키는 컬렉션의 존재 ③ 그 컬렉션의 벡터 설정(dense: 벡터 이름 `dense`·차원·DOT·sparse 없음, sparse: dense 없음·벡터 이름 `sparse`) ④ 상태 manifest의 schema 버전·내용 해시·`status=ok`·별칭 이름·대상 컬렉션 이름·벡터 이름·payload 스키마 버전·차원·build id·입력 해시 존재 ⑤ 로컬 dense artifact manifest(schema·해시)와 발행 세대의 일치·차원 ⑥ 로컬 sparse artifact(BM25 파라미터) manifest 해시와 발행 세대의 일치.

**검사하지 않음**: 포인트 수·payload 내용·벡터 값(스캔 없음), KoE5 질의 임베더의 모델 이름·revision(검사에서 모델을 만들지 않으므로 첫 검색의 기존 검사에 남는다), 서버 모드에서의 네트워크 지연·인증 종류. `verified`는 **기동 시점**의 결과다 — `/health`는 `checked_at`·`note`를 함께 싣고 조회마다 다시 묻지 않는다.

Context 비활성(`CONTEXT_WEIGHT=0`)이면 Qdrant에 묻지 않는다(테스트가 호출 목록 빈 것을 확인). 연결 장애는 설정 오류와 **형으로** 구분한다(`ContextPublicationError` vs 그 외) — 문자열 매칭 없음. 자동 폴백(Context를 끄고 계속)은 추가하지 않았다.

## 5. 정상 환경의 후보 ID·순서

같은 합성 발행에 세 질의로 `search_fused_songs(fact_k=4, sparse_k=2)`를 돌린 결과가 수정 전후 **동일**(`fused_candidates`, JSON 비교 `True`):

| 질의 | 후보(song_id · dense 순위 · sparse 순위) |
|---|---|
| 테스트 프로그램 배경음악 | 101(1·2) → 202(2·1) |
| 프로듀서가 데모를 다시 녹음 | 202(1·1) → 101(2·2) |
| 삽입곡 | 202(1·1) → 101(2·2) |

테스트 `test_candidate_ids_and_order_are_unchanged_by_the_check`는 같은 검색기에서 검사 전/후, 그리고 검사 전용 인스턴스가 같은 클라이언트를 쓴 뒤의 새 검색기까지 결과가 같음을 고정한다. 실제 5,143곡 색인·dev 176에서의 재측정은 하지 않았다(검색 경로 코드는 변경이 없어 필요하지 않다고 판단 — 미측정으로 남긴다).

## 6. 테스트

| 묶음 | 결과 |
|---|---|
| 신규 `tests/test_v60_context_startup_check.py` | **66 통과**(1차 49 + 리뷰 반영 17) — ① `verify_publication` 34건(정상·메타데이터만 조회·미발행=검색 빈 결과/기동 오류·별칭 2개/1개 누락·잘못된 대상·대상 컬렉션 없음·dense/sparse 스키마 불일치·manifest 누락/손상/해시/내용 5종/status·**중첩 자료형 오류 9종 + dense `embedding` 블록 오류 3종 = `manifest_invalid`(검색 경로도 같은 형)**·로컬 dense/sparse artifact 불일치/누락·연결 장애 2건·후보 순서 불변) ② `context_startup.run` 18건(비활성 = 조회 0·정상·기본값 켜짐·설정 오류 5종 기동 실패·**자료형 오류 3종 = config_error(unavailable 아님)**·미발행·연결 장애 = unavailable·클라이언트 없음·설정값 오류 3종·상태 복사) ③ 실제 lifespan/health 11건(정상 발행 verified + 예열 끔·별칭 누락 기동 실패 → 잠금 해제 → 복구 → 재기동·**검사의 예상 밖 TypeError에도 클라이언트 닫힘·잠금 해제·재기동**·**`context_build_id=123` 실제 lifespan에서 config_error + 잠금 해제**·부분 누락·검사가 예열보다 먼저 같은 클라이언트로·비활성이면 클라이언트 안 만짐·연결 장애는 뜨고 URI 비노출·벡터 클라이언트 실패는 unavailable·health 조회가 재검사 안 함) |
| 기존 관련(`tests/context/test_context_qdrant_load.py` · `test_context_fixed_followup_qdrant.py` · `test_context_fixed_followup_retrieval.py` · `tests/test_warmup.py` · `tests/test_qdrant_backend.py`) | 97 통과(수정 뒤) |
| 기존 관련 + 신규 묶음(리뷰 반영 뒤) | 163 통과 |
| 전체 `tests/` | **아래 §6-1** |

모든 Qdrant는 `:memory:` 또는 임시 폴더. 발행 자료는 합성(`tests/context/test_context_qdrant_load`의 `_sources`/`_load`). conftest가 Vertex 확인·ADC를 대체하고, lifespan 테스트는 피드백 색인·작업 스레드·Mongo 종료를 명시적으로 대체했다. 실행은 `env -u CLAUDE_EFFORT venv/bin/python -m pytest …`(셸의 `CLAUDE_EFFORT` 충돌 회피).

### 6-1. 전체 테스트

`tests/context/test_context_artifacts_v4.py::ContextArtifactV6Tests::test_default_artifact_path_follows_custom_raw_sandbox` 1건은 **수정 전 main에서도 실패**(macOS `/tmp` 경로 의존)라 deselect 했다. 결과(리뷰 반영 뒤): **6,544 통과 · 130 skip · 1 deselect · 실패 0**(40 subtests 포함, 79초). 1차(리뷰 전)는 6,527 통과. 수정 전 main에서의 전체 실행은 v59 때의 6,478 통과(동일 deselect)이고, 차이 66 = 이번 신규 테스트 수다.

## 7. 호환성 영향

- **기동 동작 변화**: `CONTEXT_WEIGHT>0`(기본)인 서버는 이제 별칭·컬렉션·manifest·로컬 artifact가 맞지 않으면 **뜨지 않는다**. 전에는 떴다. 이것이 의도다 — 조용히 Context 없이 도는 것보다 낫다. Context를 쓰지 않을 환경은 `CONTEXT_WEIGHT=0`으로 명시해야 한다(미발행 Qdrant + 기본값 0.5는 이제 `not_published`로 기동 실패).
- `CONTEXT_WEIGHT` 값 오류는 첫 검색 500 → 기동 실패.
- 연결 장애·벡터 클라이언트 초기화 실패는 기존 정책 그대로(경고 + 기동 + 요청에서 재시도). Context만 `unavailable`로 표시.
- `/health`에 `context` 키 추가(기존 키 변경 없음). `ContextQdrantSearch`의 예외는 `RuntimeError` 하위형으로 바뀌어 기존 호출부 호환.
- 기동 시간: 별칭 목록 1회 + 컬렉션 존재/설정 조회 4회 + 로컬 manifest 2개 읽기 + BM25 파라미터 로드(약 1.6MB; 검사 전용 인스턴스가 읽고, 라우터의 검색기는 첫 검색에서 따로 읽는다 — 중복 1회) — 모델 로드 없음. 실측하지 않았다.
- 평가 CLI(`evaluate_search_accuracy` 등)는 lifespan을 타지 않으므로 영향 없음.

## 8. 미검증·남은 한계

- 운영 서버 Qdrant(Docker·v56 컨테이너)에서의 실측은 하지 않았다 — 테스트·재현은 로컬 `:memory:`/임시 폴더와 가짜 연결 실패만.
- 실제 5,143곡 색인·dev/test 품질 재측정 없음(검색 코드 변경 없음).
- 연결 장애를 "그 외 예외" 전부로 보므로 서버의 4xx 같은 비일시적 응답도 `unavailable`로 분류된다(설정 오류로 승격하지 않음 — 확인 불가로만 남고, 확인 완료라 적지 않는다).
- KoE5 임베더 모델 이름/revision 대조는 첫 검색에 남아 있다(검사에서 모델을 만들지 않기로 했으므로).
- `verified`가 기동 시점 결과임을 `/health`가 말하지만, 기동 뒤 별칭이 바뀌면 검색 경로의 기존 `_snapshot` 검사가 요청마다 잡는다(그때는 전처럼 경고 + 빈 결과).
- 기동 지연 실측 없음.

## 9. 리뷰 반영 (PR 전 2건)

| # | 지적 | 반영 |
|---|---|---|
| 1 (P2) | 해시를 맞춘 manifest에 `qdrant=null`·`source_hashes=[]`를 넣으면 `_snapshot`의 `.get()`이 AttributeError를 내고, 기동 모듈이 이를 연결 장애(`unavailable`)로 분류해 Context 없이 기동을 허용 | `_snapshot`이 비교 전에 `qdrant`·`source_hashes`가 객체인지, `context_build_id`·두 입력 해시가 비어 있지 않은 문자열인지 검증해 `ContextPublicationError(reason="manifest_invalid")`로 올린다. dense artifact manifest의 `embedding` 블록도 같게(`_dense_embedding_config`, 검색 경로의 `_text_embedder`와 공유). 테스트: 자료형 오류 9종 + embedding 3종(검색 경로도 같은 형), 기동 모듈 3종 = `config_error` |
| 2 (P2) | lifespan이 `ContextStartupError`일 때만 클라이언트를 닫아, `context_build_id=123` manifest가 로그 출력에서 TypeError를 내자 캐시가 남고 같은 저장 폴더 재개방이 잠금 오류로 실패 | `except Exception`으로 넓혀 검사의 **어떤 예외**에도 `close_vector_client()` 뒤 원래 예외를 다시 올린다. (1의 자료형 검증으로 그 TypeError 자체도 사라졌다.) 테스트: 검사 함수가 TypeError를 던지는 lifespan(예외형 그대로 전파·캐시 없음·잠금 해제·같은 프로세스 재기동), `context_build_id=123` 실제 lifespan(`manifest_invalid` + 잠금 해제) |

반영 뒤 `repro_after.json`을 다시 만들었다(9시나리오 결과·후보 순서 동일). 운영자 문서의 `manifest_invalid` 설명에 중첩 자료형 오류를 추가했다.

## 10. 파일

| 파일 | 내용 |
|---|---|
| `RUN_INFO.md` | 이 문서 |
| `repro_context_startup.py` | 9개 시나리오 기동·검색·후보 순서 재현(수정 전/후 공용, 외부 호출 없음) |
| `repro_before.json` | main `f4e87ff` 체크아웃에서 실행(`has_context_startup=false`) |
| `repro_after.json` | 이 워크트리에서 실행(`has_context_startup=true`) |

재현: `cd <체크아웃 루트> && PYTHONPATH=. env -u CLAUDE_EFFORT venv/bin/python experiments/reranking/results_v60_context_startup_check/repro_context_startup.py --out <JSON>`
