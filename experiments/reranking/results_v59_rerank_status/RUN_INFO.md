# v59 — 리랭킹 부분 실패·후보 밖 id 응답이 "성공"으로 표시되던 문제 수정 (2026-10-16 밤)

워크트리 `/private/tmp/vague-finder-rerank-status` · 브랜치 `fix/rerank-status` · 기준 main `466fbe8`(PR #73 v58 병합, #72 v57 포함) · **커밋 전**.
v56 운영 검증(`results_v56_attr_runtime_validation` §7-③)에서 발견한 결함 ③을 수정했다. ①(v57 공유 예산)·②(v58 가사 캐시)는 병합돼 있고 이 수정은 그 위에서 돈다.

모델·Mongo·Qdrant 호출 0. 검증은 전부 가짜 SDK 클라이언트(`tests/fake_gemini_client.py`)를 실제 `GeminiListwiseReranker` 아래에 꽂아 했다.

## 1. 문제 — 셋이 한 뿌리다

| # | 증상(v56·v57 실측) | 어디서 |
|---|---|---|
| 1 | 가사 보호 묶음 하나가 `failed`여도 다른 묶음이 성공하면 `reorder_stage=model_applied`. 호출 수(3/2)에만 흔적. 평가기는 같은 조합을 `partial_failure`로 세므로 서비스·평가 분류가 달랐다 | `SearchExplain.reorder_stage` |
| 2 | 모델이 후보 밖 id만(또는 빈 `ranking`) 돌려줘도 파서가 빠진 id를 입력 순서로 채워 **정상 파싱** → `applied`. 모델이 기여한 것이 없는데 "리랭킹 적용", 곡별 설명은 "리랭킹이 N위 유지" | `GeminiListwiseReranker._parse_response` |
| 3 | `rerank_calls`는 묶음 처리 기록 수인데 화면은 "N번 불러 M번 적용"으로 호출 횟수처럼 적었다. v57의 `budget_exhausted` 묶음은 모델을 부르지 않았는데도 "불렀다"에 들어가고, 재시도 3회는 1로 보였다 | `SearchExplainOut.rerank_calls` · `map.js renderRunNote` |

## 2. 수정 — 무엇을 어디서

| 파일 | 변경 |
|---|---|
| `src/retrieval/gemini_listwise_reranker.py` | **A.** `_parse_response`: 최상위가 객체가 아니거나 `ranking`이 배열이 아니면 `InvalidRankingResponse`(ValueError). id는 문자열·정수 표기만 인정(객체·배열·bool은 형식 오류 항목). 유효·고유 id가 **0개면 보충 전에** `InvalidRankingResponse` — 호출부의 기존 `except Exception` 재시도(정정 지시 덧붙임)·공유 마감이 그대로 처리하고, 끝까지 무효면 기존 `failed` 폴백. 유효 id가 1개 이상이면 **보충 정책 그대로**(응답 순서 → 빠진 id는 입력 순서로 뒤에). 관측치 `stats_out`(valid·missing·duplicate·invalid). `rerank_run`: 실제 SDK 호출 횟수를 `_generate(tally=)`로 세어 `RerankRun.model_calls`에(앞 단계 Search evidence·희소 사실 검증·재시도 포함, 예산 소진으로 부르지 않은 호출은 제외), 어느 패스 응답에도 없던 곡을 `unreturned_ids`에, 반영된 응답의 관측치를 `response_stats`에. 실패 로그에 예외 형식 이름을 함께 적음 |
| `src/retrieval/explain.py` | **B.** 단계 `REORDER_MODEL_PARTIAL = "model_partial"`(라벨 "리랭킹 일부 적용") 추가. `reorder_stage`: 커밋됐고 적용 ≥1이면 실패·예산 소진이 섞였을 때만 `model_partial`, 아니면 `model_applied`. 커밋 취소(라우터 예외)는 전처럼 `model_failed`. `skipped`·`unknown`은 실패와 섞지 않음. 묶음 수 속성 `rerank_runs_failed`·`rerank_runs_budget_exhausted`·`rerank_runs_not_applied`, 호출 수 합 `rerank_model_calls`(세지 않은 실행이 하나라도 있으면 None). `RerankRun`에 `unreturned_ids`·`model_calls`·`response_stats`. `SongExplain`에 `rerank_group_status`(이 곡이 속한 묶음의 결과)·`rerank_unreturned`. `note_rerank_run(..., group_ids=, unreturned_ids=, model_calls=)`(기존 두 인자 호출 호환). `revoke_reorder`가 둘도 지움. 사용자 문구 `REORDER_STAGE_NOTES`(부분 적용에만). `render_ko`: 미반환 곡은 "모델 응답에 이 곡이 없어 입력 순서로 보충한 뒤 점수 합성·순서 규칙을 적용(모델이 정한 자리 아님)", 부분 적용 요청의 실패 묶음 곡은 "이 곡의 묶음은 리랭킹에 실패해 리랭킹 전 순서 사용", 예산 소진 묶음 곡은 "…예산이 소진돼 모델을 부르지 않음" — 전에는 둘 다 "순서 규칙으로 배치(리랭커에 넘기지 않음)"(거짓: 넘겼다) |
| `src/retrieval/search_router.py` | `call_reranker`가 `note_rerank_run`에 묶음 전체 id·미반환 id·호출 수를 넘김. 예산 소진 `RerankRun`은 `model_calls=0`. 폴백 규칙·`reorder_committed`/`revoke_reorder`·`_call_applied`·`_bonus_order`는 그대로 |
| `src/backend/schemas/explain.py` | **C.** `SearchExplainOut`: `reorder_stage_note`(고정 문구), `rerank_groups_failed`, `rerank_groups_budget_exhausted`, `rerank_model_calls: Optional[int]` 추가. `rerank_calls`·`rerank_calls_applied`는 값·의미 그대로, 설명만 "묶음 처리 기록 수(실제 SDK 호출 횟수가 아니다)"로 정정. `TrackExplain`: `rerank_group_status`, `rerank_unreturned` 추가 |
| `src/frontend/static/map.js` · `index.html` | 실행 기록 줄: "N번 불러 M번 적용" → "후보 묶음 N개 중 M개 리랭킹(, K개는 예산 소진으로 모델 미호출)" + 백엔드가 센 값이 있을 때만 "모델 호출 R회". 서버 문구 `reorder_stage_note` 표시. 경고색에 `model_failed`·`model_partial` 포함(예외 없는 백엔드 내부 폴백도 폴백이다). `map.js?v=rerank-status-1` |
| `src/retrieval/evaluate_search_accuracy.py` · `evaluate_clarification.py` | **변경 없음.** `rerank_outcome`(applied / partial_failure / fallback / exception / policy_skipped / skipped / unknown)이 이미 같은 조합 규칙이었고, 이번에 서비스 `reorder_stage`를 그쪽에 맞췄다. detail CSV의 `reorder_stage` 열에 `model_partial` 값이 새로 나올 수 있다(열 구성은 그대로). `rerank_run_statuses` 열은 기록 여부와 무관하게 남으므로 **explain을 꺼도 판정이 유지**된다(시험으로 고정) |
| `tests/fake_gemini_client.py` | 모드 추가 `empty_ranking`·`wrong_types`·`partial_ids`·`invalid_then_normal`. `foreign_ids` 설명 정정 |
| `tests/test_v56_rerank_fallback_paths.py` · `tests/test_rerank_shared_budget.py` | 옛 동작을 고정하던 3건 갱신(후보 밖 id → `failed`, 부분 실패 → `model_partial`) |
| `tests/test_v59_rerank_status.py`(신규 49건) · `tests/frontend/explain_panel.test.cjs`(+3건) | §6 |

바꾸지 않은 것: 정상 응답의 순위 계산(`_aggregate_passes`·`_fuse`·rescue 규칙), 부분 누락 보충 순서, 가사 보호 배치·tier, 답 가산·거절 제외, top_k·중복 방지, 재시도 횟수·대기·공유 예산(v57), 정책 생략(`rerank_policy`), 프롬프트, structured outputs 도입 없음, 가사 캐시·`HYBRID_SCAN_LIMIT`.

## 3. 수정 전후 — 같은 가짜 응답 대본

`repro_rerank_status.py`를 수정 전 main(466fbe8, 임시 워크트리)과 수정 후 코드에 각각 돌렸다(`repro_before.json`·`repro_after.json`). 예산은 운영 30초의 1/30(1초)·요청 제한 0.6초(v57 시험과 같은 축소). "sdk"는 가짜 클라이언트가 실제로 받은 호출 수.

| 사례(묶음 구성) | 가짜 응답 | 전: 단계 · calls/applied | 후: 단계 · calls/applied · failed/exhausted · model_calls | sdk | 평가기 outcome(전=후) |
|---|---|---|---|---|---|
| 모든 묶음 정상(보호 tier 2 + 일반) | 역순 | model_applied · 3/3 | model_applied · 3/3 · 0/0 · 3 | 3 | applied |
| 첫 묶음 실패·나머지 성공 | fail_group_first | **model_applied** · 2/1 | **model_partial** · 2/1 · 1/0 · 2 | 2 | partial_failure |
| 첫 묶음 성공·뒤 묶음 실패 | normal → exception | **model_applied** · 2/1 | **model_partial** · 2/1 · 1/0 · 2 | 2 | partial_failure |
| 첫 묶음 성공·뒤 묶음 예산 소진 | normal → slow | **model_applied** · 3/1 | **model_partial** · 3/1 · 1/1 · 2 | 2 | partial_failure |
| 모든 묶음 무응답(소진) | slow | model_failed · 3/0 | model_failed · 3/0 · 1/2 · 1 | 1 | fallback |
| 모든 묶음 예외 | exception | model_failed · 2/0 | model_failed · 2/0 · 2/0 · 2 | 2 | fallback |
| 앞 묶음 성공 뒤 라우터 예외(스텁) | — | model_failed · 2/1 | model_failed · 2/1 · 1/0 · **None**(세지 않는 백엔드) | — | exception |
| 후보 밖 id만 | foreign_ids | **model_applied** · 1/1 · sdk 1 | **model_failed** · 1/0 · 1/0 · 3 | 3 | **applied → fallback** |
| 빈 ranking | `{"ranking": []}` | **model_applied** · 1/1 · sdk 1 | **model_failed** · 1/0 · 1/0 · 3 | 3 | **applied → fallback** |
| 자료형 오류(최상위 배열) | `[...]` | model_failed(AttributeError 3회) | model_failed(`InvalidRankingResponse` 3회) · 1/0 · 1/0 · 3 | 3 | fallback |
| 입력 순서와 같은 정상 응답 | identity | model_applied · 1/1 | model_applied · 1/1 · 0/0 · 1 | 1 | applied |
| 일부 누락·중복·무효 혼합 | 앞 절반 역순 + 중복 1 + 후보 밖 1 | model_applied · 1/1 | model_applied · 1/1 · 0/0 · 1 (unreturned 3곡 표시) | 1 | applied |
| 무효 뒤 정상 | foreign_ids → normal | **model_applied(첫 무효 응답을 그대로 수용, sdk 1, 순서 = 입력 o1…o6)** | model_applied(재시도 응답 반영, **순서 = 모델 순서 o6…o1**) · 1/1 · 2 | **1 → 2** | applied |
| 계속 무효 + 공유 마감 1초 | foreign_ids | **model_applied** · sdk 1 | model_failed · 1/0 · 1/0 · 1(마감 안에서 끝, 1.008초) | 1 | **applied → fallback** |
| 정책 생략(표지 지배) | — | policy_skipped · 0/0 | policy_skipped · 0/0 · 0/0 · None | 0 | policy_skipped |
| 리랭킹 비활성 | — | not_attempted · 0/0 | not_attempted · 0/0 · 0/0 · None | 0 | skipped |
| 상태 미보고 백엔드 | — | model_unknown · 1/0 | model_unknown · 1/0 · 0/0 · None | — | unknown |

반환 순서는 **"무효 뒤 정상" 한 사례를 빼고 전후 동일**하다(`top_ids` 비교). 계속 무효인 사례는 "입력 순서 + applied" → "입력 순서 + failed"로 상태만 바뀌고 순서는 같다. "무효 뒤 정상"은 전에는 첫 무효 응답을 성공으로 받아 입력 순서(o1…o6)를 냈고, 후에는 재시도해 정상 응답의 모델 순서(o6…o1)를 낸다 — **재시도에서 정상 응답을 받으면 순위가 바뀌는 것은 의도한 동작**이고, 그만큼 실제 측정 순위에도 영향이 있을 수 있다(§8, 미측정).

곡별 설명 전후(발췌, `repro_*.json`의 `songs[*].summary`):

| 사례 · 곡 | 전 | 후 |
|---|---|---|
| 첫 묶음 실패 · 보호 곡 p1 | "… → 순서 규칙으로 배치(**리랭커에 넘기지 않음**)" | "… → **이 곡의 묶음은 리랭킹에 실패해 리랭킹 전 순서 사용**" |
| 뒤 묶음 예산 소진 · 일반 곡 o1 | "… → 가사 일치 보호 곡 뒤에 배치 5곡"(실패 언급 없음) | "… → **이 곡의 묶음은 리랭킹 예산이 소진돼 모델을 부르지 않음**(리랭킹 전 순서 사용) · 가사 일치 보호 곡 뒤에 배치 5곡" |
| 후보 밖 id만 · o1 | "… → **리랭킹이 1위 유지** · 최종 점수는 리랭커 85% + 검색 15% 합성" | "… → **리랭킹이 실패해 검색 순서 유지**" |
| 혼합 응답 · 응답에 없던 o4 | "… → **리랭킹이 4위 유지** · 최종 점수는 …" | "… → **모델 응답에 이 곡이 없어 입력 순서로 보충한 뒤 점수 합성·순서 규칙을 적용(모델이 정한 자리 아님)** · 최종 점수는 …" |

실제 `/search` 라우트(TestClient, 보호 tier 2 + 일반, normal → slow): 전 `reorder_stage=model_applied · rerank_calls 3 · applied 1`(그 밖 필드 없음) → 후 `model_partial · "리랭킹 일부 적용" · note="일부 후보 묶음만 리랭킹했고, 나머지 묶음은 리랭킹 전 후보 순서를 그대로 사용했습니다." · calls 3 · applied 1 · failed 1 · budget_exhausted 1 · model_calls 2 · rerank_error ""`. 반환 id `p1,p3,p2,p4,p5,o1,o2,o3` 전후 동일, 응답 1.02초. 곡별 `rerank_group_status`: p1~p3 applied · p4·p5 failed · o1~o3 budget_exhausted.

## 4. 판정표 — 묶음 결과의 조합 → 요청 단계

서비스(`SearchExplain.reorder_stage`)·API(`reorder_stage`)·평가기(`rerank_outcome`)가 같은 규칙이다(`tests/test_v59_rerank_status.py::test_stage_and_evaluator_outcome_agree`로 12조합 고정).

| 조건 | 단계 | 라벨 | 평가기 | 사용자 문구(`reorder_stage_note`) |
|---|---|---|---|---|
| 시도 없음(꺼짐·후보 1곡) | `not_attempted` | 리랭킹을 시도하지 않음 | `skipped` | — |
| 정책 생략(표지 지배·자료 없는 MV 장면) | `policy_skipped` | 리랭킹 생략(정책) | `policy_skipped` | (MV 생략은 기존 `notice`) |
| 라우터 예외 → 커밋 취소(앞 묶음 성공 여부 무관) | `model_failed` | 리랭킹 실패 | `exception` | — |
| 커밋 · 적용 ≥1 · failed/budget_exhausted 없음(skipped·unknown 섞여도) | `model_applied` | 리랭킹 적용 | `applied` | — |
| 커밋 · 적용 ≥1 · failed 또는 budget_exhausted ≥1 | **`model_partial`(신규)** | **리랭킹 일부 적용** | `partial_failure` | "일부 후보 묶음만 리랭킹했고, 나머지 묶음은 리랭킹 전 후보 순서를 그대로 사용했습니다." |
| 커밋 · 적용 0 · failed/budget_exhausted ≥1 | `model_failed` | 리랭킹 실패 | `fallback` | — |
| 커밋 · 적용 0 · unknown만 | `model_unknown` | 리랭킹 결과 확인 불가 | `unknown` | — |
| 커밋 · 적용 0 · skipped만(순서 규칙 있음/없음) | `order_rules_only` / `model_skipped` | … | `skipped` | — |

"검색 순서"라고 쓰지 않은 이유: 실패한 묶음은 **리랭킹 직전의 후보 순서**(보정·가사 보호·답 가산·거절 반영)로 나가며 1차 검색 순서와 같다는 보장이 없다. 전체 실패에도 요청 단위 문구를 두지 않았다(보호 배치가 남으면 거짓이 된다 — 곡별 기록이 말한다).

곡별(`TrackExplain`): `reorder_applied`(이 곡 자리에 모델 결과 반영) × `rerank_group_status`(속한 묶음의 결과) × `rerank_unreturned`(응답에 없어 보충된 곡).
- applied 묶음 · 반환됨 → "리랭킹이 N위에서 M위로 …" / "묶음 안에서 …"(기존)
- applied 묶음 · **미반환** → "모델 응답에 이 곡이 없어 입력 순서로 보충한 뒤 점수 합성·순서 규칙을 적용(모델이 정한 자리 아님)" — `reorder_applied`는 참으로 둔다(합성 점수는 받았고 묶음 결과가 자리를 포함해 최종에 반영됐다). 모델이 자리를 정했다고는 말하지 않는다. **"묶음 뒤에 배치"라고도 말하지 않는다** — 보충(반환된 곡 뒤, 입력 순서)은 중간 단계이고 그 뒤 검색 점수 합성·순서 규칙이 자리를 바꾼다. 리뷰 재현: 마지막 후보 하나만 신뢰도 0.1로 돌려주면 보충된 4곡이 반환된 곡보다 앞에 놓였다(저신뢰 합성에서 검색 점수 비중이 커짐). 시험 `test_unreturned_songs_may_precede_the_returned_one_…`으로 고정
- failed 묶음(부분 적용 요청) → "이 곡의 묶음은 리랭킹에 실패해 리랭킹 전 순서 사용"
- budget_exhausted 묶음 → "이 곡의 묶음은 리랭킹 예산이 소진돼 모델을 부르지 않음(리랭킹 전 순서 사용)"
- 묶음 없음(단독 보호 그룹) → "순서 규칙으로 배치(리랭커에 넘기지 않음)"(기존 — 이제 이 말은 실제로 넘기지 않은 곡에만 붙는다)
- 전체 실패(`model_failed`) → 기존 문구 그대로("리랭킹이 실패해 검색 순서 유지" / "리랭킹은 실패")

## 5. 기록 수와 실제 호출 횟수

| 필드 | 뜻 | 예(첫 묶음 성공 → 뒤 묶음 예산 소진, 묶음 3) |
|---|---|---|
| `rerank_calls` (호환) | 묶음 처리 기록 수 = `len(rerank_runs)`. 예산 소진으로 부르지 않은 묶음도 1, 재시도 3회도 1, 라우터 예외 기록도 1 | 3 |
| `rerank_calls_applied` (호환) | 모델 순서가 반영된 묶음 수 | 1 |
| `rerank_groups_failed` (신규) | 불렀는데 실패한 묶음 수(라우터 예외의 `failed` 기록 포함) | 1 |
| `rerank_groups_budget_exhausted` (신규) | 모델을 부르지 않은 묶음 수 | 1 |
| `rerank_model_calls` (신규, null 가능) | 백엔드가 센 실제 SDK 호출 수의 합(Search evidence·검증·재시도 포함). 세지 않는 백엔드(CE·스텁·`rerank()`만 있는 백엔드)가 하나라도 섞이면 null | 2 (성공 1 + 취소 1 + 미호출 0) |

차이가 드러나는 사례: 후보 밖 id만 → 기록 1 · 호출 3(재시도). 전부 무응답 n007형 → 기록 3 · 호출 1. 혼합 응답은 기록 1 · 호출 1. 화면은 `rerank_model_calls`가 숫자일 때만 "모델 호출 R회"를 적고, 묶음 수는 "후보 묶음 N개 중 M개 리랭킹(, K개는 예산 소진으로 모델 미호출)"로 적는다. 묶음 기록으로 호출 횟수를 추정하는 자리는 없다.

## 6. 검증

전부 모델·네트워크·Mongo·Qdrant 없음. `CLAUDE_EFFORT` 환경변수를 지운 셸(`env -u CLAUDE_EFFORT`)에서 돌렸다(메모리의 충돌 기록).

| 묶음 | 결과 |
|---|---|
| **신규 `tests/test_v59_rerank_status.py` 52건**(리뷰 반영 전 49 + §9의 3) | 통과. 파서 단위(빈 ranking·ranking 없음·후보 밖 id만·id 자료형 오류·bool id, 최상위 배열/문자열·ranking 객체/문자열/숫자, JSON 아님은 기존 `JSONDecodeError`, 입력 순서와 같은 정상 응답, 누락·중복·무효 혼합의 보충 순서·관측치, 이상한 스칼라 자료형) · 리랭커(무효 3모드 → 재시도 3회 뒤 failed·입력 순서·model_calls 3, 무효 뒤 정상 → applied·호출 2, 계속 무효 + 공유 마감 1초 안에 failed, 혼합 응답의 unreturned·judged·stats) · 단계 파생 12조합 = 평가기 outcome, 묶음 수 분리, 라우터 예외 = model_failed/exception, 정책 생략·미시도 · 실제 라우터(첫 묶음 실패·뒤 묶음 실패·뒤 묶음 예산 소진·전부 실패·라우터 예외 후 전체 폴백·표지 지배 생략·비활성·미보고 백엔드·미반환 곡 설명) · 평가기(기록 꺼도 무효 응답 fallback·부분 partial_failure) · `/search` 라우트 JSON(부분·실패·identity·스텁 null·호환 필드) |
| **화면 `tests/frontend/explain_panel.test.cjs` 22건(+3)** | 통과(Node 22). 일부 적용 줄의 문구·묶음 수·예산 소진·호출 수·경고색·"번 불러" 없음, 모두 적용·null 호출 수, 예외 없는 실패도 경고색·정책 생략은 비경고, `map.js` 버전 |
| **관련 기존**(설명·v56 폴백·v57 공유 예산·listwise 리랭커·사용자 설명 라우트·정책·입력 순서·데모 실패·예열·피드백·타이밍·가사 인용·재질문·사용자 설명) | **751 통과 · 67 skip · 실패 0**(갱신한 3건 포함) |
| **전체 `tests/`** | **6,475 통과 · 130 skip · 실패 0**(경로 의존 1건 deselect — §8). 첫 실행에서 `tests/test_demo_faults.py` 6건이 500으로 실패했는데 워크트리에 git 제외 파일 `src/frontend/static/map_data.json`이 없어서였다(메인에서는 6건 통과). 메인에서 복사한 뒤 재실행해 전부 통과 |

갱신한 기존 시험: `test_foreign_ids_are_dropped_and_input_order_is_kept_but_status_applied` → `test_foreign_ids_only_response_is_retried_then_reported_failed`(v56 §7-③의 옛 동작을 고정하던 것), `test_first_group_applied_then_budget_runs_out`·`test_evaluator_protected_groups_share_the_budget`의 단계 단언 `model_applied` → `model_partial`.

재현 스크립트 `repro_rerank_status.py`(이 폴더, 모델 호출 없음)와 결과 `repro_before.json`·`repro_after.json`을 git에 둔다. 수정 전 실행은 main 466fbe8을 받은 임시 워크트리(scratchpad)에서 같은 파일을 돌렸고 그 워크트리는 지웠다.

## 7. 호환성

- 기존 필드(`reorder_stage`·`reorder_stage_label`·`rerank_calls`·`rerank_calls_applied`·`rerank_error`·`rerank_skip_reason`)는 이름·값 의미 그대로. `rerank_calls`는 설명만 정정.
- `reorder_stage`에 새 값 `model_partial`이 나온다. 이 저장소 안의 소비처(API 변환·화면 라벨·`demo_rehearsal`/`check_demo_faults`(라벨만 읽음)·평가기 detail CSV 열)는 확인했다. 저장된 옛 CSV를 읽는 외부 도구가 `reorder_stage` 값 집합을 고정했다면 모르는 값을 만난다(평가기 집계는 `rerank_outcome`을 쓰므로 영향 없음).
- `/search` 응답에 새 키 `reorder_stage_note`·`rerank_groups_failed`·`rerank_groups_budget_exhausted`·`rerank_model_calls`(null 가능), 곡별 `rerank_group_status`·`rerank_unreturned`. 구형 화면은 무시한다. 새 화면은 구형 서버 응답(새 키 없음)에서도 깨지지 않는다(`rerank_groups_budget_exhausted > 0`·`typeof === "number"` 검사).
- `ExplainRecorder.note_rerank_run(status, judged_ids)` 두 인자 호출은 그대로 된다(새 인자는 키워드·기본값). `RerankRun`을 만드는 다른 백엔드(CE·스텁)는 `model_calls=None`으로 "모른다"가 되고 합계도 null이다.
- `_parse_response`에 셋째 인자(선택)만 더했다. 반환 모양(4-튜플)은 그대로.
- **동작 변화**: 유효 후보 0개 응답이 더는 성공이 아니다. 운영 예산(30초) 안에서 재시도 최대 3회를 쓴 뒤 `failed`가 되므로, 그런 응답을 내는 질의는 전보다 **재시도 대기 최대 약 2초 + 추가 모델 호출 최대 2회의 응답 시간**을 더 쓰며, 그 합은 남은 공유 예산으로 제한된다(1초 예산 사례에서 1.008초 안에 끝남). 계속 무효면 결과 순서는 같다(입력 순서). **재시도에서 정상 응답을 받으면 결과 순서가 바뀐다**(전에는 무효 응답을 성공으로 받아 입력 순서를 냈다) — 의도한 동작이고 영향은 미측정(§8).
- 화면 경고색: 예외 없는 리랭킹 실패(`model_failed`)·일부 적용에도 경고색이 붙는다(전에는 `rerank_error`가 있을 때만).

## 8. 미검증 · 남은 위험

- **실제 Gemini 응답 분포는 보지 않았다.** 후보 밖 id만·빈 ranking이 실제로 얼마나 나오는지(v56에서는 가짜로만 재현) 모른다. 그 빈도만큼 "applied"가 "failed"로 옮겨 가거나(계속 무효) 재시도 응답의 모델 순서로 바뀌며(무효 뒤 정상), 그만큼 재시도 호출·시간이 늘어난다. dev/test 정확도 재측정은 범위 밖이라 하지 않았다 — 옛 측정에서 무효 응답이 `applied`로 섞여 있었다면 `rerank_outcome` 분포가 달라지고, 재시도가 정상 응답을 받는 질의는 순위도 달라질 수 있다. **Hit/MRR에 미치는 영향은 미측정**이다("불변"이 아니다).
- 일부 누락 응답의 보충 순서는 이번에 바꾸지 않았다(요구). 누락 곡은 설명에서만 "모델이 정한 자리 아님"으로 구분한다. 누락이 많은 응답(예: 30곡 중 2곡만 반환)을 어디부터 무효로 볼지는 정하지 않았다 — 관측치(`response_stats`)가 내부 `RerankRun`에 남지만 API·CSV에는 싣지 않았다.
- `rerank_model_calls`는 Gemini listwise만 센다. CE·다른 백엔드는 null(추정하지 않음).
- 라우터 예외 폴백의 `note_rerank_run(RERANK_FAILED, [])` 기록은 `rerank_groups_failed`에 1로 들어간다(묶음이 아니라 예외 기록). 필드 설명에 적었고 `rerank_error`가 함께 있어 구분된다.
- 프론트 변경은 Node 단위 시험(실제 함수 + 최소 DOM)으로만 확인했다. 브라우저 실화면은 보지 않았다.
- 전체 `tests/` 가운데 `tests/context/test_context_artifacts_v4.py::…::test_default_artifact_path_follows_custom_raw_sandbox` 1건은 이 변경과 무관한 경로 의존(`/tmp` ↔ `/private/tmp`) 실패로, 수정 전 main에서도 같은 단언으로 실패함을 확인하고 제외했다(v58 때와 같다).
- 후속(§7-③ 밖): `HYBRID_SCAN_LIMIT` 5000 < 5,143곡, 서버 모드 Context 별칭 기동 검사, 속성 보조 경로 활성화 결정.

## 9. 리뷰 반영(2026-10-16)

| # | 지적 | 반영 |
|---|---|---|
| 1 | [P2] 클라이언트 생성 전에 호출 수를 올려 **모델 호출 0회인데 model_calls=1** | `_generate`의 `generate_content`를 실제로 부르는 두 자리(동기·비동기 `_call` 안)로 집계를 옮겼다. 생성 실패 시험(`_new_client`가 예외 → failed·model_calls 0)과 취소된 비동기 호출·동기 전용 가짜가 각 1회로 세어지는 시험 추가 |
| 2 | [P2] 누락 곡이 반드시 "묶음 뒤"에 놓이지는 않는다(저신뢰 응답에서 보충 곡 4개가 반환 곡보다 앞) | 곡별 문구를 "모델 응답에 이 곡이 없어 입력 순서로 보충한 뒤 점수 합성·순서 규칙을 적용(모델이 정한 자리 아님)"으로, `rerank_unreturned` API 설명·내부 주석·§4를 같은 뜻으로 고쳤다. 지적한 사례(마지막 후보 하나만 신뢰도 0.1)를 시험으로 고정 — 반환 곡이 1위가 아니고 문구에 "묶음 뒤"가 없다 |
| 3 | [P2] "모든 사례 반환 순서 동일"·"Hit/MRR 불변"·"지연 약 2초" | §3·§7·§8 정정: "무효 뒤 정상" 사례는 전 입력 순서 → 후 모델 순서(의도한 동작), Hit/MRR 영향은 **미측정**, 추가 지연은 재시도 대기 + 추가 모델 호출 시간이며 남은 공유 예산으로 제한 |

반영 뒤 재실행: 신규 52건 통과 · 화면 22건 통과 · **전체 `tests/` 6,478 통과 · 130 skip · 실패 0**(경로 의존 1건 deselect). `repro_after.json`은 고친 문구로 다시 만들었다(`repro_before.json`은 그대로).
