# v57 — 리랭킹 단계 시간 예산 공유 수정 (2026-10-16)

v56 운영 검증(§7-①)에서 발견한 결함의 수정 기록이다. **정확도 실험이 아니다** — 검색 정책·가중·후보 수·프롬프트·운영 `.env`·기본값·색인은 바꾸지 않았고, 열린 test 자료로 재평가·튜닝을 하지 않았다. 실제 모델 호출 0(Vertex·AI Studio·Claude·Melon 없음).

- 워크트리 `/private/tmp/vague-finder-rerank-shared-budget` · 브랜치 `fix/rerank-shared-budget` · 기준 main `638114e`(#70 v55 병합). **PR #71(v56)은 병합 전**이라 v56 자료는 그 워크트리에서 읽기만 했고, v56의 가짜 SDK 클라이언트(`tests/fake_gemini_client.py`)는 **같은 내용으로 복사**했다(PR #71과 바이트 동일 — 두 PR이 모두 병합돼도 충돌 없음)
- 커밋·푸시·PR 없음(사용자가 처리)

## 1. 원인

| 위치 | 현상 |
|---|---|
| `src/retrieval/gemini_listwise_reranker.py` `rerank_run` | 호출마다 `deadline = now + time_budget_seconds(30)`를 **새로** 만들었다. 라우터(`search_router.py`)와 평가기(`evaluate_search_accuracy.py`)는 가사 exact 보호 그룹(tier마다 1회)과 일반 후보를 **순차**로 `call_reranker` → `rerank_run`으로 부르므로, 모델이 응답하지 않으면 요청 하나가 **그룹 수 × 30초**였다. v56 실측 n007(보호 2그룹 + 일반 = 3호출) 93.3초 · t018(2호출) 62.2초 > 화면 `SEARCH_TIMEOUT_MS` 60초 |
| 같은 함수의 재시도 루프 | 실패 뒤 `time.sleep(1)`을 남은 예산과 무관하게 실행했다. 호출 하나의 벽시계 제한은 min(20초, 남은 예산)이라 20초 취소 → 1초 대기 → 9초 취소 → 1초 대기로 단일 호출도 30초 예산이 **약 31초**였다(v56 q112·n011 31.2초) |

## 2. 수정 방식

원칙: **요청 하나의 리랭킹 단계 전체가 하나의 `time.monotonic()` 마감을 공유한다.** 마감은 단계 진입 때 한 번 만들고 인자로만 넘긴다 — singleton 리랭커나 라우터의 가변 상태에 저장하지 않는다(동시 요청 격리). 외부 마감이 없으면 기존 단독 호출 예산(30초)을 그대로 쓴다(평가 도구·`measure_rerank_batch`·기존 테스트 호환).

| 파일 | 변경 |
|---|---|
| `src/retrieval/search_router.py` | ① `rerank_stage_deadline(reranker)` 추가 — 리랭커가 `rerank_budget_seconds`를 알리면 `now + 예산`, 아니면 None(CE·시험 스텁은 기존 동작). ② 라우터 `search()`의 리랭킹 단계 진입(`rerank_started` 옆)에서 **한 번** 만들어 보호 그룹·일반 후보·단일 호출의 `call_reranker`에 `deadline=`으로 넘긴다. ③ `call_reranker(..., deadline=None)`: 마감이 이미 지났으면 **모델을 부르지 않고** 입력 순서를 `budget_exhausted` 상태로 돌려준다(기록기·`runs_out` 양쪽에 남김, 경고 로그). 마감을 받는 리랭커(`accepts_deadline`)에만 `deadline`을 넘긴다 |
| `src/retrieval/gemini_listwise_reranker.py` | ① `accepts_deadline = True`, `rerank_budget_seconds` 속성(설정값, 읽기 전용). ② `rerank_run(..., *, deadline=None)`: 받으면 그 마감까지만, 없으면 `now + 30초`. 앞 단계(Search evidence·희소 사실 검증) 몫 `_pass_reserve_seconds(budget)`은 **남은** 예산 기준(남은 4초면 2초). ③ 재시도 대기 `min(1초, 남은 시간)`, 남은 시간이 없으면 자지 않는다. ④ 예산 초과 메시지가 공유 예산이면 "리랭킹 단계 공유 예산(남은 N초)"으로 적힌다 |
| `src/retrieval/explain.py` | 상태 `RERANK_BUDGET_EXHAUSTED = "budget_exhausted"` 추가. `reorder_stage` 파생에서 `failed`와 같은 쪽(적용 0회면 `model_failed`). 단계 이름·라벨은 그대로다 |
| `src/retrieval/evaluate_search_accuracy.py` | 평가기 사본 `_rerank_with_lyric_protection`도 같은 함수로 마감을 만들어 세 `call_reranker`에 넘긴다(라우터와 같은 규칙). `rerank_outcome`은 `budget_exhausted`를 실패로 센다(`fallback` / 적용과 섞이면 `partial_failure`). 모르는 값으로 남기지 않는다 |

바꾸지 않은 것: 리랭킹 정책 생략(`rerank_policy`), 가사 보호 배치·tier 규칙, 답 가산·거절 제외, 후보 수·중복 방지, 부분 실패 표시 체계(§7-③), 후보 밖 id 응답 처리, Mongo 가사 캐시(§7-②), `HYBRID_SCAN_LIMIT`, 프론트 제한 60초.

## 3. 전체 예산의 적용 범위

**리랭킹 단계의 예산**이다. 질의 분석·경로 검색·융합·Mongo 가사 적재는 포함하지 않는다(검색 전체 제한이 아니다).

| 포함(한 마감 공유) | 제외(기존대로) |
|---|---|
| 가사 exact 보호 그룹 호출(tier마다) · 일반 후보 호출 · 단일 호출 | 질의 분석(`analysis_budget_seconds` 별도) |
| 호출 안의 재시도(최대 3회)와 재시도 대기 | 경로 검색·Qdrant·Context·속성 보조 경로 |
| 추가 패스(`GEMINI_RERANK_PASSES=2`) | Mongo 가사 판 적재(약 350초, §7-②) |
| 켜져 있을 때의 Search evidence·희소 사실 검증 호출(앞 단계 몫은 남은 예산 기준) | 리랭킹 뒤 규칙(title shape rescue·anchor 보호) — 모델 호출 없음 |

예산 값은 그대로 `GEMINI_RERANK_BUDGET_SECONDS`(기본 30) · 단일 호출 제한 `GEMINI_RERANK_TIMEOUT_SECONDS`(기본 20). 호출 하나의 제한 = min(20, 남은 예산). 모델 호출·재시도가 공유하는 예산은 30초이며, 이번 HTTP 재현 요청은 약 30.2초에 반환됐다. 분석·가사 적재·실행 대기 등은 별도이므로 검색 전체가 항상 화면 제한 60초 안에 끝난다는 보장은 아니다.

## 4. 성공 그룹·미실행 그룹의 반환과 상태

| 상황 | 반환 | 상태(`rerank_runs` / `rerank_run_statuses`) | 단계(`reorder_stage`) |
|---|---|---|---|
| 앞 그룹 성공, 뒤 그룹에서 예산 소진 | 성공 그룹은 **모델 순서(규칙 적용 후)** 그대로, 소진 중 그룹은 입력 순서(`failed`), 그 뒤 그룹은 호출 없이 입력 순서(`budget_exhausted`) | `applied` · `failed` · `budget_exhausted` | `model_applied`(기존 부분 실패 정책과 같다 — 호출/적용 수 3/1에 흔적) · 평가기 `rerank_outcome=partial_failure` |
| 첫 호출에서 예산 소진 | 모든 그룹 입력 순서. 리랭커 입력이 보너스 전 순서인 실험 스위치에서는 그룹별로 보너스 순서 복귀(`_bonus_order`, failed와 같은 규칙) | `failed` · `budget_exhausted`… | `model_failed` · `rerank_outcome=fallback` |
| 예산 뒤 도착한 응답 | 반영하지 않음(입력 순서, `failed`) — 기존 규칙 유지 | `failed` | |
| 정상 | 변화 없음 | `applied`… | `model_applied` |

가사 보호 배치(보호 tier 앞 → 일반), 답 가산 반영 후보 순서, 거절 제외, top_k 자르기, 중복 없음은 모두 유지된다(§6 테스트·§5-2 HTTP top-15가 v56과 전부 동일).

## 5. 수정 전후 재현 결과

### 5-1. 단위 재현(`repro_shared_budget.py`, 실제 라우터 + 실제 리랭커 + 가짜 SDK 클라이언트, 예산 1.0초 · 호출 제한 0.6초 · 재시도 3회 = 운영 30/20/3의 1/30)

| 사례 | 수정 전 (`repro_before.json`) | 수정 후 (`repro_after.json`) |
|---|---|---|
| n007형 전부 무응답(보호 2 tier + 일반 = 3호출) | **4.83초**(= 예산 4.8배) · 가짜 호출 3 전부 cancelled 0.6초 · 상태 failed×3 | **1.01초** · 가짜 호출 **1**(cancelled 0.6초, 재시도 대기 0.4초로 잘림) · 상태 failed · budget_exhausted · budget_exhausted · 단계 model_failed |
| t018형 전부 무응답(보호 1 tier + 일반 = 2호출) | 3.22초 · 호출 2 | 1.01초 · 호출 1 · failed · budget_exhausted |
| n007형 1그룹 성공 뒤 무응답 | 3.22초 · 호출 3(returned·cancelled·cancelled) · applied·failed·failed | 1.01초 · 호출 2(returned·cancelled) · applied·failed·budget_exhausted · 단계 model_applied(부분 실패) |
| 단일 호출 예외 재시도(예산 0.5초, 보호 1 tier + 일반) | 2.02초 · 호출 2(각 raise 뒤 **1초 대기**) | 0.51초 · 호출 1(대기 0.5초로 잘림) |
| n007형 정상 | 0.01초 · 3호출 applied | 동일(0.01초 · 3호출 applied · 같은 순서) |

반환 순서는 수정 전후 모든 사례에서 같다(폴백 = 입력 순서, 성공 그룹 = 모델 순서).

### 5-2. HTTP 재검증(v56 재현 환경, 운영 기본 예산 30/20/3)

- 환경: Docker Qdrant `vague-finder-qdrant-v56` 재기동(v1.19.1, 127.0.0.1:6356), **Context 별칭 2개 확인**(`vaguefinder-context-{dense-1024-koe5,sparse-bm25}__dev` → `__generation_f6424b94621a28cf`, 텍스트 5,143점). 실제 FastAPI 앱 + `v56_app.py`(이 폴더로 복사, `ROOT`가 이 워크트리의 수정 코드) · 저장 분석 캐시(v49 dev 3개 + v55 test) · 가짜 SDK 클라이언트 · 모델 클라이언트 생성 차단 확인(`guard: blocked/blocked`). **A만**(운영 기본 `ATTR_FILTER_WEIGHT=0 · always`, 포트 8056) — 수정은 리랭킹 단계라 보조 경로(D)와 무관하고 v56에서 A·D 동일 현상이었다. D는 돌리지 않았다
- 기동 16:01:59 → 예열 완료 16:08:19(가사 판 적재 약 5.7분, §7-② 그대로) → `run_fallback.py --arms A --cases n007,t018,q112,n011` 16:08:45~16:11:3x(18요청, 오류 0, 가사 TTL 300초 안이라 재적재 없음)
- 비교 대상: v56 1차(A, `run1_noalias/fallback_results.csv`). v56은 별칭 누락 조건이었지만 네 질의의 top-15는 전부 같았다(`compare_v56_v57_A.json`)

| 질의(호출 구조) | 모드 | v56 A | **v57 A** | 가짜 호출(v57) | 상태·단계(v57) | top-15 = v56 |
|---|---|---|---|---|---|---|
| **n007**(보호 2 · 보호 2 · 일반 23) | slow | **93,252ms** | **30,164ms** | 2[cancelled 20.0초 · cancelled 9.0초] — 2·3번째 그룹 호출 없음(서버 로그 "예산 소진 — 2곡 그룹 · 23곡 그룹") | 3/0 · model_failed | ✓ |
| | exception | 6,366 | 6,179 | 9 raised(3그룹 × 3회, 대기 1초씩 예산 안) | 3/0 · model_failed | ✓ |
| | fail_group_first | 2,226 | 2,198 | 5(첫 그룹 3 raised · 2 returned) | 3/2 · model_applied | ✓ |
| | normal(앞·뒤) | 166 · 350,615(가사 재적재) | 216 · 188 | 3 returned | 3/3 · model_applied | ✓ |
| **t018**(보호 2 · 일반 28) | slow | **62,228** | **30,199** | 2[cancelled 20.0 · 9.0] — 28곡 그룹 호출 없음 | 2/0 · model_failed | ✓ |
| | exception / fail_group_first / normal | 4,166 / 2,212 / 600·425 | 4,156 / 2,391 / 168·185 | 6 / 4 / 2 | 2/0 · 2/1 · 2/2 | ✓ |
| **q112**(2턴 성별 답 + 거절, 단일 30) | slow | 31,247 | **30,219** | 2[cancelled 20.0 · 9.0](재시도 대기가 남은 예산으로 잘림) | 1/0 · model_failed | ✓ |
| | exception / normal | 2,159 / 305·235 | 2,163 / 141·208 | 3 / 1 | 1/0 · 1/1 | ✓ |
| **n011**(2턴 장르 답 + 거절, 단일 30) | slow | 31,215 | **30,197** | 2[cancelled 20.0 · 9.0] | 1/0 · model_failed | ✓ |
| | exception / normal | 2,139 / 102·168 | 2,120 / 80·180 | 3 / 1 | 1/0 · 1/1 | ✓ |

- 구간 기록(`timing_A.jsonl`, n007 slow): `search.rerank` 30,006ms = `rerank.protected_group`(2곡) 30,005ms + `rerank.protected_group`(2곡) 0.24ms + `rerank.others`(23곡) 0.16ms — 뒤 두 그룹은 호출 없이 끝났다
- 18요청 모두 결과 15곡 · 중복 0 · 전부 후보 안. 각 질의의 첫 normal과 마지막 normal top-15 동일(실패 뒤 누출 없음). slow 행의 순서 = exception 행의 순서(둘 다 리랭킹 직전 후보 순서). 2턴 q112·n011의 slow·exception 폴백도 답 가산·거절 반영 순서(v56과 동일)
- 요청 시간은 리랭킹 예산 30초 + 검색 0.2초 = 약 30.2초로 화면 제한 60초 안이다. 단, 가사 판 재적재(§7-②)가 겹치면 여전히 넘는다 — 별도 작업

## 6. 실행한 테스트

| 묶음 | 결과 |
|---|---|
| **신규 `tests/test_rerank_shared_budget.py` 17건** | 통과(4회 반복 모두 통과, 약 11초). 사례: 보호 2~3그룹 전부 무응답 시 예산 공유(호출 1 · 나머지 미호출 · 상태 · 순서 · 단계) / 호출별 제한 = min(단일 제한, 남은 예산)(셋째 호출 0.2초) / 앞 그룹 성공 뒤 예산 소진(성공 그룹 모델 순서 보존 · 부분 실패 단계) / 재시도 대기 클램프(0.5초 예산에 0.5초, 2번째 시도 없음) / 충분한 예산이면 재시도 3회 유지 / 외부 마감 뒤 도착 응답 미반영 · 안 응답 반영 · 지난 마감이면 호출 0 / 마감 없으면 자기 예산(단독 호출 호환, `rerank_stage_deadline` 값) / 앞 단계 몫이 남은 예산 기준 / 예산 소진 요청 뒤 정상 요청 누출 없음 / **동시 요청**(같은 singleton 리랭커, A 소진 중 B는 3그룹 모두 applied) / 2턴 답·거절이 프롬프트·후보에 반영 / 정책 생략은 호출 0 / 마감 미지원 백엔드(스텁)는 기존대로 + 지난 마감이면 어떤 백엔드든 미호출 / `reorder_stage`·`rerank_outcome` 분류 / 평가기 사본 예산 공유 |
| 관련 기존 테스트 `test_gemini_listwise_reranker` · `test_search_explain` · `test_router_rerank_input_order` · `test_clarify_answers_for_reranker` · `test_rerank_policy` · `test_rerank_score_cap` + 신규 | **461 통과** |
| PR #71의 `tests/test_v56_rerank_fallback_paths.py` 10건(스크래치패드로 복사해 이 코드에 대고 실행) | 10 통과 |
| 전체 `tests/` | 1차 실행 6,367 통과 · 130 skip · 실패 7 → **전부 이번 변경과 무관**(아래). 자료 복사 뒤 실패했던 6건만 재실행해 통과. 이후 전체를 한 번에 다시 실행한 결과는 아님; Context 기존 실패 1건은 남음 |

- `tests/context/test_context_artifacts_v4.py::ContextArtifactV6Tests::test_default_artifact_path_follows_custom_raw_sandbox` — 손대지 않은 메인 체크아웃에서도 같은 실패(macOS `/tmp` → `/private/tmp` 해석). 기존 문제
- `tests/test_demo_faults.py` 6건(`test_every_real_backend_the_routes_depend_on_is_replaced` · `test_question_matches_the_candidates_the_mock_search_returned` · `test_question_lookup_fault_modes[clarify_fail-True]` · `[server_error-False]` · `test_hanging_lookup_can_be_released` · `test_rehearsal_walks_the_same_round_trip_as_the_screen`) — 메인 체크아웃에서는 통과. 이 워크트리에서의 실패 원인은 **git 제외 파일 `src/frontend/static/map_data.json`이 새 워크트리에 없어서**(`demo_faults._songs()`가 읽음, 500 응답). 메인의 사본을 복사한 뒤 6건 통과. 리랭킹 변경과 무관

## 7. 미검증 범위 · 남은 위험

- **실제 Vertex 호출로는 재지 않았다.** 가짜 클라이언트의 무응답은 `asyncio.sleep`이고 실제 HTTP 멈춤은 SDK 클라이언트 취소·연결 닫힘(`aclose`)을 거친다 — 그 경로 자체는 v56 이전 PR 리뷰 때 검증된 것이고 이번에 바꾸지 않았다
- HTTP 재검증은 **A·가사 질의 2건·2턴 2건**뿐이다. D(보조 경로 켬)는 돌리지 않았다. 보호 tier가 3개 이상인 질의, 추가 패스·Search grounding·희소 사실 검증을 켠 구성은 단위 테스트(가짜 시계)로만 봤다
- 동시 요청은 테스트(2요청, 스레드)로만 확인했다. 서버에서 동시 요청 부하는 재지 않았다
- 평가기(`evaluate_search_accuracy`)로 dev 세트를 다시 돌리지 않았다. 이번 정상 단위 재현과 HTTP 18요청에서는 반환 순서가 같았다. 그러나 이전에 각 그룹이 30초 안에 성공했어도 그룹 합계가 30초를 넘었다면 수정 후 뒤 그룹이 생략될 수 있다. 기존 dev 전체의 순위·정확도 영향은 미측정이다
- `budget_exhausted`는 새 상태 문자열이다. 이 저장소 안의 소비처(`reorder_stage`·`rerank_outcome`·`_rerank_reported_failure`·`_call_applied`·재질문 평가기 집계 문자열 카운트)는 다뤘지만, 저장된 옛 CSV를 읽는 외부 도구가 있으면 모르는 값을 만난다
- 남은 결함(별도 작업): §7-② Mongo 가사 판이 적재 완료 후 300초에 만료되면 다음 가사 요청이 동기 전체 재적재를 기다림(접근으로 TTL이 연장되지 않으며, 이번 예열도 약 5.7분) · §7-③ 부분 실패·후보 밖 id가 `model_applied` · `HYBRID_SCAN_LIMIT` 5000 < 5,143 · 서버 모드 Context 별칭 기동 검사 없음
- 운영 기본값 그대로이므로 **예산 30초는 이제 요청 단위**다. 보호 그룹이 많은 질의에서 모델이 느리면(실패가 아니라 느림) 뒤 그룹이 `budget_exhausted`로 리랭킹 없이 나갈 수 있다 — 전에는 그룹마다 30초를 다 썼다. 리랭킹 대기 시간을 제한하기 위한 의도한 교환이며, 검색 전체의 60초 제한을 보장하지는 않는다

### PR 전 확인

- 코드 리뷰에서 관련 테스트 461건을 다시 실행해 모두 통과했다. 신규 모델 호출은 없었다.
- HTTP CSV를 git 예외로 포함하고 실행 로그·제어 파일은 제외했다.
- 검색 전체 60초 보장과 기존 dev 결과 불변을 단정하지 않도록 보고서를 정정했다. 서비스 코드는 리뷰 후 추가 수정하지 않았다.

## 8. 재현

```bash
cd /private/tmp/vague-finder-rerank-shared-budget
# 단위 재현(수정 전 숫자는 main 638114e 코드에서 같은 스크립트로; 가짜 클라이언트는 이 브랜치의 tests/fake_gemini_client.py)
GCP_PROJECT_ID= PYTHONPATH=. venv/bin/python experiments/reranking/results_v57_rerank_shared_budget/repro_shared_budget.py after
# 회귀 테스트
GCP_PROJECT_ID= venv/bin/python -m pytest tests/test_rerank_shared_budget.py -q
# HTTP 재검증(v56 재현 환경 — 컨테이너·별칭·분석 캐시·Mongo 읽기 필요, 모델 호출 없음)
docker start vague-finder-qdrant-v56            # 별칭 2개 확인 후
zsh experiments/reranking/results_v57_rerank_shared_budget/start_servers.sh A   # /health ready(가사 적재 약 6분)까지 대기
venv/bin/python experiments/reranking/results_v57_rerank_shared_budget/run_fallback.py --arms A --cases n007,t018,q112,n011 --suffix _A_v57
kill $(cat experiments/reranking/results_v57_rerank_shared_budget/ctl/server_A.pid); docker stop vague-finder-qdrant-v56
```

## 9. 파일

| 파일 | 내용 |
|---|---|
| `RUN_INFO.md` | 이 문서 |
| `repro_shared_budget.py` · `repro_before.json` · `repro_after.json` | §5-1 단위 재현 스크립트와 수정 전·후 결과 |
| `fallback_results_A_v57.csv` · `.jsonl` | §5-2 HTTP 재검증 18요청(응답 15곡 id·후보 top-10·설명 단계·가짜 호출 기록) |
| `compare_v56_v57_A.json` | v56 1차 A 행과의 요청별 대조(시간·상태·호출·top-15 동일 여부) |
| `timing_A.jsonl` · `fake_calls_A.jsonl` · `logs/` · `ctl/` | 서버 구간 기록·가짜 호출 로그·서버/실행 로그·모드 파일(로컬 보존, git 제외). 공유 증거는 HTTP CSV/JSONL과 전후 대조 JSON에 보존 |
| `v56_runtime.py` · `v56_app.py` · `run_fallback.py` · `start_servers.sh` | v56 재현 도구 사본(경로만 이 폴더로; PR #71 원본과 `start_servers.sh`의 `OUT` 한 줄만 다르다) |
| `../../../tests/test_rerank_shared_budget.py` | 회귀 테스트 17건 |
| `../../../tests/fake_gemini_client.py` | v56 가짜 SDK 클라이언트(PR #71과 동일 내용) |
