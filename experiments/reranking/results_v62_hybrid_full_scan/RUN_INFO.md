# v62 — 하이브리드 검색의 고정 조회 상한으로 인한 점수 누락 수정 (2026-10-16 밤)

워크트리 `/private/tmp/vague-finder-hybrid-full-scan`, 브랜치 `fix/hybrid-full-scan`(main `fc20e86` = **#76 v61 진단 병합** 기준), **커밋 전**.
모델(Vertex·AI Studio·Claude)·Mongo·운영 Qdrant·멜론 호출 **0**(§8). 운영 `.env`·운영 색인(`artifacts/qdrant`)·DB는 열지 않았고 바꾸지 않았다.
v61 진단([보고서](../results_v61_hybrid_scan_audit/RUN_INFO.md))의 권고 §7-A를 구현한 것이다. v61의 저장 입력(질의 벡터·후보 기록)은 v61 워크트리에서 **읽기만** 했고, v61 산출물은 덮어쓰지 않았다.

## 0. 결론 요약

| 질문 | 답 |
|---|---|
| 무엇을 고쳤는가 | `QdrantIndex.query`의 하이브리드 분기(sparse_vector가 있을 때)에서 dense·sparse 두 조회의 `limit`을 고정 상수(5,000) 대신 **요청마다 `count(exact=True)`로 센 대상 수**(필터가 있으면 같은 필터)로 정한다. 대상 0이면 조회·payload 없이 빈 결과. count 실패는 삼키지 않고 전파. 요청 사이 캐시 없음. dense 전용 조회·합산식·정렬·top_k·payload 읽기 방식·이미지/오디오/Context 경로는 그대로 |
| 정확한가 | v61 저장 호출 **311건을 수정 구현에 재생**(환경변수 우회 없음) → 독립 정확 합산 기준과 **311/311 일치**(집합·점수·순서, 누락 0 · 점수 불일치 0). 311건 전부 `limit == count`, dense 반환 == 대상 수. 남은 차이는 f32/f64 부동소수점(최대 3.0e-8) |
| 후보 구성 영향 | dev 176질의(가사 경로 빈 조건, 리랭킹 없음): 수정 전 A ↔ 수정 구현 — **후보 30개 ID·순서 176건 전부 동일**, 정답 회수 122/176 · 리랭킹 전 Hit@1/5/10 38/73/101 · MRR 0.3149 동일, 바뀐 것은 t003의 5번째 후보 융합 점수 0.0002 하나(v61 예상과 일치). 독립 기준 B와는 지문까지 **176/176 동일**. detail CSV 전 열 대조에서도 그 지문 1건 외 차이 0 |
| 지연 | 필터 없는 호출: 로컬 파일 모드 차이 중앙값 −0.02ms(잡음), **서버 모드 +1.8ms**(count 1.5ms). 필터 있는 호출: 로컬 +18ms, **서버 +34ms**(필터 count 32ms — payload 색인 없는 전수 집계). 서버 모드 절대값은 수정 후 중앙값 22ms(필터 없음)·56ms(필터) |
| 전제·한계 | count와 두 조회는 **한 스냅샷이 아니다** — 그 사이 적재가 끼면 그 요청은 잘릴 수 있고 반환 건수로는 감지 불가. **적재 중 검색 중단**이 운영 조건이며 세대+별칭 발행은 후속 설계(§2). 가사 경로(Mongo)가 있는 조건의 후보 비교는 미검증(§5-3) |

## 1. 변경 동작 (`src/vector_db/qdrant_backend.py`)

| 항목 | 수정 전(v61 기준 커밋 `fc20e86`) | 수정 후 |
|---|---|---|
| dense·sparse 조회 `limit`(하이브리드) | 모듈 상수 `HYBRID_SCAN_LIMIT = int(env QDRANT_HYBRID_SCAN_LIMIT, 기본 5000)` | 요청마다 `self._count(collection, query_filter)` = `client.count(collection_name, count_filter=필터, exact=True).count` |
| 필터 | 두 조회에만 적용 | count·dense·sparse 세 호출에 같은 `translate_filter` 결과 |
| 대상 0 | 빈 두 조회 + 빈 합산 → `{"matches": []}` | count 뒤 바로 `{"matches": []}` — 검색·payload 호출 없음 |
| count 실패 | (없음) | `_search`와 같은 모양으로 `logger.error` 뒤 **그대로 전파**(상수로 대체하지 않음). 라우터의 "경로 실패 = 빈 결과" 계약은 그대로 |
| dense 전용(sparse 없음) | `limit=top_k`, count 없음 | 변화 없음 |
| 합산·정렬·top_k·payload | dense+sparse 코드 합산 → 내림차순 → `[:top_k]` → 남은 곡만 `retrieve` | 변화 없음 |
| `QDRANT_HYBRID_SCAN_LIMIT` | 조회 상한 | **폐기 예정 호환값** — 읽히면 `HYBRID_SCAN_LIMIT`에 int(없으면 None)로 남고 **import 때 한 번** 경고(`logger.warning`), 조회에는 어떤 효과도 없다 |
| Qdrant 호출 수(하이브리드) | 2(+payload 1) | 3(+payload 1) |

**설정 호환 결정 — 왜 제거하지 않고 호환값으로 남겼나.** 사용처를 확인한 결과: `.env.example` 주석, `docs/vector_backend.md`, 운영 `.env`에는 없음(v61 §1), 진단 도구가 모듈 상수를 읽거나 바꾼다(v56 `check_scan_limit.py`·`diagnose_mode_diff.py`, v61 `run_candidates.py`·`bench_latency.py`·`repro_synthetic.py`·`replay_calls.py`). 이름을 없애면 그 도구들이 `AttributeError`로 멈추고, 옛 설정이 남은 환경은 아무 말 없이 무시된다. 그래서 **이름은 남기되 조회 경로에서는 전혀 읽지 않는다**(`query()`에 상수 참조 없음 — `grep limit=`로 확인 가능). 진단 도구가 상수를 바꿔도 운영 조회는 잘리지 않는다 — v61 `run_candidates.py` C 구성(`qb.HYBRID_SCAN_LIMIT = point_count`)과 `bench_latency.py` A/C/D, `repro_synthetic.py`는 이제 **같은 결과**를 내며(기록용 참고), `run_candidates.py`의 `scan_limit_truncates` 계산만 None에서 터지지 않게 고쳤다(그 외 v61 도구·결과 파일은 수정하지 않음).

**기존 환경변수가 설정된 사용자**: 동작은 "상한 N으로 잘림" → "항상 전수"로 바뀌고, 기동 로그에 `QDRANT_HYBRID_SCAN_LIMIT=... 은(는) 더 이상 하이브리드 조회를 자르지 않습니다` 경고가 한 번 남는다(요청마다 반복되지 않음 — 테스트로 확인). `.env`에서 지우면 경고도 없다. **옛 결과(5,143곡·상한 5,000으로 잘린 2026-10-11~10-16 측정) 재현**은 v61 기준 커밋 `fc20e86`의 코드로 돌린다 — 지금 코드에는 조회를 자르는 설정이 없다(일부러 두지 않았다: 잘림을 되살리는 스위치는 결함을 설정으로 옮기는 것이다). 문서: `docs/vector_backend.md` 점수 계산 표·주의 2항, `.env.example` 주석.

**반환 건수 == limit은 잘림 감지에 쓰지 않는다**(v61 리뷰 반영 그대로) — 전수 조회도 정확히 limit개를 돌려준다.

## 2. 정확성의 전제 — 동시 색인 변경

- **적재 방식(확인)**: `src/vector_db/cli/qdrant_load.py`는 서비스가 조회하는 **같은 컬렉션에 200건씩 직접 upsert**한다(`--recreate`면 컬렉션을 지우고 다시 만든 뒤 적재). 별도 세대·별칭 전환은 곡 컬렉션에 없다(Context 컬렉션만 `qdrant_load_context`가 세대+별칭 2개로 원자 교체, `docs/namuwiki_context_vector_db.md`). 로컬 파일 모드(현재 운영)는 한 프로세스만 폴더를 열 수 있어 **서버가 떠 있으면 적재가 애초에 실패**하고, 서버 모드(`QDRANT_URL`)에서는 적재와 검색이 동시에 될 수 있다.
- **count → dense → sparse는 원자적 스냅샷이 아니다.** 사이에 점이 추가되면 두 조회는 count보다 많은 대상에서 상위 count개만 돌려주므로 그 요청의 합산은 전수가 아닐 수 있다(옛 결함과 같은 모양 — 가짜 클라이언트 테스트 `test_premise_a_stale_count_is_trusted_…`로 그대로 적어 두었다). 점이 삭제·갱신되거나 같은 수가 추가·삭제되면 전후 count가 같아도 검출할 수 없다. 반환 건수로도 알 수 없다.
- **따라서 이번 수정은 "조회 중 색인이 변하지 않는 조건"에서의 정확성이다.** 동시 적재까지 보장하지 않는다. 적용 조건: **적재 중에는 검색을 받지 않는다**(현재 로컬 모드는 잠금으로 강제됨; 서버 모드 배포 시에는 서비스 중단 뒤 적재, 또는 곡 컬렉션에도 세대+별칭 발행 도입 — 후속 설계, 이번 PR에서 발행·배포 구조는 바꾸지 않았다). 요청 사이 count 캐시는 두지 않았다 — 무효화 설계 없는 캐시는 적재 뒤 첫 요청을 자른다.
- 문서 반영: 모듈 설명 "정확성의 전제", `docs/vector_backend.md` 주의 항목.

## 3. 회귀 테스트

`tests/test_v62_hybrid_full_scan.py` **15건**(가짜 클라이언트 9 + 임시 로컬 Qdrant 6) + `tests/test_v61_hybrid_scan_audit.py` **13건**(수정 뒤 동작으로 갱신) + 기존 `tests/test_qdrant_backend.py` 31건. 전부 통과(변수 없음 59건 · `QDRANT_HYBRID_SCAN_LIMIT=6000`에서 v61+v62 28건). 전체 `tests/` 결과는 §8.

| 필수 사례 | 테스트 | 자료원 |
|---|---|---|
| 대상 수에 맞춰 두 조회의 limit 결정 · 호출 순서 count→dense→sparse→retrieve | `test_both_scan_limits_equal_the_counted_target_and_count_comes_first` · `test_real_engine_limits_follow_the_count_…` | 가짜 · :memory: |
| 같은 필터가 count·dense·sparse에 적용 | `test_the_same_filter_goes_to_count_dense_and_sparse` · `test_real_engine_applies_the_same_filter_…`(통과 3곡 → limit 3) | 가짜 · :memory: |
| 대상 0 → 빈 결과, 검색·payload 없음 | `test_zero_targets_return_empty_without_search_or_payload`(호출 목록 == ["count"]) · v61 `test_filter_with_no_match_returns_empty` | 가짜 · :memory: |
| dense 전용은 count 없음 | `test_dense_only_query_does_not_count_…` · `test_real_engine_dense_only_does_not_count` | 가짜 · :memory: |
| 한쪽 점수 누락·양쪽 목록 밖 후보 누락 해소 | v61 `test_songs_outside_the_old_dense_limit_keep_their_dense_portion`(4번 1.00→**1.15**) · `test_song_outside_both_old_lists_is_kept_at_its_exact_rank`(7번 4위 유지) — 옛 상수를 3으로 둬도 기준과 같음 | :memory: |
| 음수 dense 몫 정확 합산 | v61 `test_negative_dense_portion_is_summed_exactly_…`(3.0→**2.0**) · v62 `test_negative_dense_is_summed_exactly_on_the_real_engine` | :memory: |
| sparse 미일치·빈 sparse·top_k 초과·동점 | v61 4건 + v62 `test_sparse_cases_on_the_real_engine` | :memory: |
| count 실패가 숨겨지지 않음 | `test_count_failure_is_propagated_and_no_search_runs`(RuntimeError 전파, 조회 0회, error 로그) | 가짜 |
| 연속 요청 사이 대상 수 변화 반영 | `test_a_changed_target_count_is_used_by_the_very_next_request`(3→7) · `test_points_added_between_requests_are_counted_…_in_file_mode`(tmp_path 파일 모드, limit [3,3]→[7,7]) | 가짜 · 파일 모드 |
| 기존 환경변수 값으로 다시 잘리지 않음 | `test_old_env_limit_set_small_does_not_truncate_the_fake_either`(변수 2 → limit [7,7]) · v61 `test_the_old_environment_variable_is_read_as_a_compat_value_but_does_not_truncate`(경고 1회, 요청 3회 반복에도 추가 경고 0) · `…_no_compat_value_and_no_warning` | 가짜 · :memory: |
| payload는 최종 top_k만 | `test_scores_are_only_read_and_payload_is_fetched_for_the_final_top_k_only`(scan with_payload [False,False] · retrieve ids == 상위 2 · include_metadata=False면 retrieve 없음) · `test_real_engine_payload_is_read_only_for_the_final_top_k` | 가짜 · :memory: |
| (전제) count가 조회 시점보다 작으면 잘린다 | `test_premise_a_stale_count_is_trusted_so_concurrent_loading_can_still_truncate` — 보장하지 않는 범위를 테스트로 명시 | 가짜 |

v61 테스트 갱신 원칙: 독립 내적 기준 `_exact`와 7곡 자료·숫자(1.15 · 0.60 · 2.0)는 그대로 두고, "잘린다"는 단언을 "같은 조건에서 기준과 같다"로 바꿨다(실패 단언만 지운 것이 아니라 같은 수치를 반대 방향으로 단언). **옛 결함 재현은 v61 기준 커밋에 보존**: `git show fc20e86:tests/test_v61_hybrid_scan_audit.py`.

## 4. 저장 호출 311건 재생 (`replay_fixed.py` → `replay_fixed.json` · `replay_fixed_calls.csv`)

- 입력(읽기 전용): v61 워크트리 `cand_A/<set>_dev/calls.jsonl` · `calls_vectors.npz`(sha는 `replay_fixed.json.source_inputs`), 색인 복사본 `artifacts/qdrant_v61_audit`(v61 복사본의 파일 복사, `.lock` 제거). **내용 지문 `238299e6c8c17f2a`** = v61 기록(5,143점 · dense 1024 · sparse nnz 1,061,090 · 어휘 44,486). 분석 캐시(v49 Vertex dev 57·59·60건): prompt `80c464515d9c`·postprocess `f18778f8f94d` 일치, 폴백 0. `fingerprint_drift`에 `gemini_backend`·`gemini_location`이 뜨는 것은 이 워크트리에 `.env`(GCP 설정)를 두지 않아 지금 환경의 호출 경로가 `none`으로 잡히기 때문이며, 분석은 캐시만 읽으므로 질의 분석에는 영향이 없다(새 분석 호출 0).
- 방법: 지금 디스크의 `QdrantIndex.query`를 311건(필터 16건 포함, top_k 30 × 178 · 90 × 133) **실제 호출**하고 `client.count`·`query_points`를 가로채 count·limit·반환 건수를 기록, `exact_hybrid.ExactHybridReference`(scroll 전수 + f64 내적)와 `compare_rankings`(허용 오차 1e-4, 동점 묶음 안 순서는 차이 아님)로 대조. `QDRANT_HYBRID_SCAN_LIMIT`는 **설정하지 않았다**(설정돼 있으면 스크립트가 멈춘다).

| 항목 | 값 |
|---|---|
| 호출 | **311**(v06 105 · v09 102 · v10 104) |
| 기준과 일치(집합·점수·순서) | **311 / 311** |
| 누락 곡 / 점수 불일치 곡 | **0 / 0** |
| `limit == count` (두 조회 모두) | 311 / 311 — 필터 없음 295건은 count 5,143, 필터 16건은 3~2,464 |
| dense 반환 == 대상 수 | 311 / 311 (전수) · sparse 반환 0~5,143(질의 토큰과 겹치는 곡만, 6건은 전 곡 일치) |
| 차이 분류 | 실제 누락 **0** · 동점 순서 차이 **0** · 부동소수점(허용 오차 안) 305 · 완전 동일 6 |
| 공통 곡 점수 최대 차이 | **2.97e-8**(Qdrant f32 누적 vs numpy f64) |
| 기준 동점 묶음이 있는 호출 | 288 / 311(묶음 안 순서는 비교에서 세지 않음 — 결과적으로 ID 순서도 전부 같았다) |
| 호출 ms(로컬 파일 모드, 임베딩 없음) | 중앙값 233.7 · p95 332.5 |

v61 같은 입력의 현행 코드 결과(차이 4건: n017·t003·t016·t018)와 비교하면, 수정 구현은 **환경변수 없이** v61에서 `QDRANT_HYBRID_SCAN_LIMIT=5143`로만 얻을 수 있던 0건을 낸다. 저장된 A/B/C 결과를 다시 집계한 것이 아니라 수정 코드를 다시 부른 결과다.

## 5. 후보 구성 영향 (`run_candidates_fixed.sh` → `cand_fixed/cand_A/<set>_dev/` · `compare_fixed.py` → `candidate_changes_fixed_vs_{A,B}.csv` · `candidate_summary_fixed_vs_{A,B}.json` · `detail_column_diff.json`)

같은 분석 캐시(v49 Vertex dev) · 같은 색인 복사본 · 같은 설정(`v61_guard.fixed_env`: `CONTEXT_WEIGHT=0.5`, `ATTR_FILTER_WEIGHT=0`, `RERANKER_ENABLED=false`, 피드백 끔)으로 v61 `run_candidates.py --arm A`를 **별도 `--out-root`(이 폴더 `cand_fixed/`)**에 돌렸다(세트당 36~40초, 하이브리드 호출 105·102·104 = 311, dense 전용 161). 기준선 `cand_A`·`cand_B`·`cand_C`와 v61 보고서는 덮어쓰지 않았다. 비교는 `compare_candidates.py --baseline-root <v61 워크트리> --compare-root cand_fixed`로 기준선 루트와 새 결과 루트를 각각 넘겼다. 분모는 dev 176질의 전부(모두 정답 있음), **리랭킹 전 후보·순서 지표**다 — 최종 서비스 Hit@10이 아니다. **가사 표면 경로(Mongo)는 빈 조건**(§7)이라 가사 단서 질의 19건의 후보 구성은 서비스와 다르다. 조건 기록(`cand_fixed/*/conditions.json`)의 `git_sha`는 HEAD `fc20e86`이고 `src/` 수정은 커밋 전이다(`hybrid_scan_limit_*`는 None = 상한 없음).

### 5-1. 수정 전 A(v61) ↔ 수정 구현

| 지표 | 수정 전 A | 수정 구현 |
|---|---|---|
| 후보 30개 **ID·순서**가 다른 질의 | — | **0 / 176** |
| 후보 지문(id·순서·점수)이 다른 질의 | — | **1**(v10 t003: 5번째 후보 융합 점수 0.0516→0.0518, 차이 0.00022, 순서 불변) — v61 §5-2가 예상한 바로 그 1건 |
| 정답이 후보 30개에 포함 | 122 (69.3%) | **122** (69.3%) |
| 새로 회수 / 빠진 정답 | — | **0 / 0** |
| 리랭킹 전 Hit@1 / @5 / @10 | 38 / 73 / 101 | **38 / 73 / 101** |
| 리랭킹 전 MRR@10 | 0.3149 | **0.3149** |
| 세트별 정답 포함(v06·v09·v10) | 43 · 41 · 38 | 43 · 41 · 38 |
| detail CSV 전체 열(시간 열 제외) 질의별 대조 | — | `candidate_digest` 1건(t003)만 다름, 나머지 모든 열(분석 JSON·`baseline_top_ids`·`explain_*`·`baseline_lyric_match_types`·`rerank_*` 등) 176건 동일 |

### 5-2. 독립 기준 B(v61) ↔ 수정 구현

후보 지문(id·순서·융합 점수) **176 / 176 동일**, detail 전체 열 차이 **0**. 즉 수정 구현의 후보 30개는 전수 벡터로 직접 계산한 기준과 서비스 융합까지 거친 뒤에도 같다. v61의 반복 재현성(A rep1 vs rep2 57/57)도 그대로 참고된다.

### 5-3. 가사 경로가 있는 조건 · 리랭커 입력

- v61과 같은 가사 경로 빈 조건에서 "후보 순서 동일·t003 점수 변화만"을 먼저 확인했다(위). **가사 경로가 있는 조건은 미검증**이다 — 저장 가사 스냅샷·고정 자료가 없고(§7) Mongo는 호출하지 않았다. v61에서 영향이 확인된 4곡은 모두 가사 질의의 top-90 보조 신호였으므로, 가사 경로가 켜진 조건에서 후보가 바뀔 수 있는 질의도 그 19건이고, 이 비교는 Mongo가 있는 환경에서 따로 해야 한다.
- 리랭커 입력: 리랭커는 꺼져 있었고(`rerank_calls` 0, `gemini.backend=unused`) 입력이 될 후보 목록·순서는 전후 동일하다. 보호 그룹 구성은 가사 경로가 빈 조건에서 `baseline_lyric_match_types`·`explain_*` 열이 전후 동일한 것으로만 간접 확인. 신규 모델 리랭킹은 하지 않았고, 후보가 같다는 것으로 최종 모델 출력이 같다고 단정하지 않는다.
- **실행 조건 사고(기록)**: 첫 실행에서는 이 워크트리에 `.env`(GCP 설정)가 없어 평가기의 리랭커 팩토리가 `GCP_PROJECT_ID` 부재로 Cross-Encoder로 내려갔고(꺼져 있어 호출은 없음), 그 결과 `candidate_digest`(CE 문서 포함 해시)·`rerank_confidence`·`effective_rerank_weight` 열이 176건 모두 v61과 달라 보였다 — 후보 ID·순서·점수는 그때도 175/176 동일(t003만). `GCP_PROJECT_ID=v62-dummy-no-calls`·`GCP_LOCATION=global`을 주어 v61과 같은 `gemini_listwise`(꺼짐) 구성으로 다시 돌린 것이 위 표다(외부 호출 차단은 그대로, `BlockedExternalCall`·`NOT BLOCKED` 로그 0건, `fingerprint_drift` 빈 목록). 같은 조건이어야 비교가 성립한다는 v61 교훈을 다시 확인했다.

## 6. 지연 (`bench_latency_v62.py` → `bench_{local,server}_calls.csv` · `bench_{local,server}_summary.json`)

같은 Qdrant 클라이언트 위에 **수정 전**(fc20e86의 `qdrant_backend.py`를 별도 모듈로 import, 상한 5,000)과 **수정 후**(지금 구현)의 `QdrantIndex`를 만들어, 저장 질의 벡터를 호출마다 **교대로**(전→후 / 후→전) 넣었다. 동시 실행 없음, 예열 각 2회는 제외, `include_metadata=True`(서비스 모양)로 count+dense+sparse+payload+합산 전체를 재고 클라이언트 호출을 감싸 단계별 시간을 기록했다. 필터 없음 = v06 A 기록의 앞 60호출(top_k 30·90 섞임, 평균 반환 59곡), 필터 있음 = 세 세트의 필터 호출 16건 × 5회 = 80쌍(통과 3~2,464곡). 두 구현의 결과(ID·점수)는 **측정한 140쌍 전부 같았다**(v61에서 다른 4건은 v09·v10의 필터 없는 top-90 호출이라 이 표본에 없음).

### 6-1. 로컬 파일 모드(복사본 `artifacts/qdrant_v61_audit`, 한 프로세스, search_params 없음 — 로컬은 항상 전수)

| 구성 | 요청 | 중앙값 ms | p95 ms | Qdrant 호출/요청 | 반환 점수 항목/요청 | 단계 중앙값 ms (count · dense · sparse · payload) |
|---|---|---|---|---|---|---|
| 필터 없음 · 수정 전 | 60 | 233.2 | 324.5 | 3 | 9,080 | — · 14.3 · 213.8 · 0.45 |
| 필터 없음 · **수정 후** | 60 | 231.8 | 324.9 | 4 | 9,238 | **0.02** · 14.7 · 212.5 · 0.78 |
| 필터 있음 · 수정 전 | 80 | 251.4 | 281.2 | 3 | 853 | — · 23.1 · 224.7 · 0.38 |
| 필터 있음 · **수정 후** | 80 | 262.7 | 295.6 | 4 | 853 | **18.4** · 23.0 · 220.5 · 0.47 |

짝 차이(수정 후 − 수정 전, 같은 호출): 필터 없음 **중앙값 −0.02ms**(평균 −8.3, p95 +97.5, 음수 50% — 잡음), 필터 있음 **중앙값 +18.2ms**(p95 +21.7, 음수 2.5%) = 필터 count 비용. 로컬 모드의 하이브리드 조회가 230ms인 이유는 sparse 조회(약 213ms, 파이썬 구현)이며 수정과 무관하다(v61 §5-3과 같은 수준). 준비: 복사본 열기 1.4초, 예열 1.9초.

### 6-2. 서버 모드(전용 Docker Qdrant 1.19.1, `127.0.0.1:6362`, 클라이언트 1.19.1, **`exact=True`** = 서비스의 서버 모드 설정, `server_setup_v62.py`)

색인: 복사본의 텍스트 컬렉션만 `qdrant_client.migrate`로 이전(2.6초) → 5,143점, 벡터 설정 동일(dense 1024 DOT + sparse), **전체 내용 지문 `238299e6c8c17f2a` 동일**(모든 점의 dense·sparse를 scroll로 읽어 해시, 3.9초). payload 색인 없음(서비스 적재기도 곡 컬렉션에는 만들지 않는다 — v56 §3). 컨테이너 `vague-finder-qdrant-v62`는 측정 뒤 종료했고 저장소는 `artifacts/qdrant_server_v62/storage`(100MB, git 제외)에 남겼다. v56 컨테이너(6356)·운영 색인·운영 서버는 건드리지 않았다.

| 구성 | 요청 | 중앙값 ms | p95 ms | 평균 ms | Qdrant 호출/요청 | 반환 점수 항목/요청 | 단계 중앙값 ms (count · dense · sparse · payload) |
|---|---|---|---|---|---|---|---|
| 필터 없음 · 수정 전 | 60 | 20.7 | 24.3 | 21.3 | 3 | 9,080 | — · 7.8 · 6.9 · 2.9 |
| 필터 없음 · **수정 후** | 60 | **22.0** | 27.5 | 23.1 | 4 | 9,238 | **1.46** · 8.0 · 7.0 · 2.9 |
| 필터 있음 · 수정 전 | 80 | 22.4 | 30.3 | 23.0 | 3 | 853 | — · 10.4 · 8.1 · 3.3 |
| 필터 있음 · **수정 후** | 80 | **56.3** | 72.5 | 58.3 | 4 | 853 | **32.0** · 10.4 · 8.3 · 3.5 |

짝 차이(수정 후 − 수정 전): 필터 없음 **중앙값 +1.8ms**(p95 +6.1, 평균 +1.8, 음수 6.7%), 필터 있음 **중앙값 +33.8ms**(p95 +43.5, 평균 +35.3, 음수 0%). 서버 모드에서 추가 비용은 거의 전부 `count` 왕복이며, **필터 count(32ms)가 필터 없는 count(1.5ms)보다 훨씬 크다** — payload 색인이 없는 필드의 `exact` 집계가 전수 필터 평가이기 때문이다. dev 311호출 중 필터 호출은 16건(5%)이고, 서비스에서 필터 경로는 제목 구조·연주 단서·속성 보조(기본 꺼짐) 경로다. 반환 항목은 +1.7%(5,000×2 → 5,143×2 수준). 준비: 클라이언트 연결 0.06초, 예열 0.3초. **로컬 PC(루프백)의 Docker라 배포 서버의 네트워크·하드웨어 수치가 아니다.**

필터 count 비용을 줄이려면 필터 필드(`vocal_gender`·`genre`·`release_date` 등)에 payload 색인을 만들거나 `exact=False` count를 쓰는 선택이 있지만, 둘 다 적재기·정확성 계약을 바꾸는 일이라 이번 범위 밖(후속)이다.

## 7. 가사 경로·리랭킹 검증 범위

- **가사 표면 경로(Mongo)는 빈 조건**이다(v61과 같음, `v61_guard`가 Mongo를 막음). 저장 가사 스냅샷·고정 자료를 찾아봤으나 없다 — `lyrics_exact_search._load`는 Mongo `full_lyrics`에서만 판을 만들고 디스크 스냅샷을 두지 않으며, 레포·`data/`에도 가사 고정 자료가 없다. 실제 Mongo는 호출하지 않았다. 따라서 **가사 단서 질의 19건의 서비스 조건(가사 경로 있음) 후보 비교는 미검증**이다. 합성 가사 테스트(기존 `tests/test_lyric_evidence.py` 등)는 보호 배치 회귀용이며 이번 변경이 그 코드를 건드리지 않았으므로 추가하지 않았다 — 그것이 서비스 전체 후보 비교를 대신한다고 보지 않는다.
- **리랭킹**: 신규 모델 리랭킹은 하지 않았다(`RERANKER_ENABLED=false`). 리랭커 입력 = 후보 30개 목록·순서이고 §5에서 전후 동일함을 확인했지만, 그것으로 최종 모델 출력이 같다고 단정하지 않는다(Vertex는 같은 입력에서도 ±1 변동). 보호 그룹 구성은 가사 경로가 빈 조건에서 detail CSV의 `baseline_lyric_match_types`·`explain_*` 열로만 간접 비교했다(§5).
- **최종 Hit@10 개선을 주장하지 않는다.** 재측정 필요 여부: §9.

## 8. 테스트·외부 호출 차단 결과

- `tests/test_v62_hybrid_full_scan.py` 15 + `tests/test_v61_hybrid_scan_audit.py` 13 + `tests/test_qdrant_backend.py` 31 = **59 통과**(`QDRANT_HYBRID_SCAN_LIMIT` 없음) · v62+v61 **28 통과**(`QDRANT_HYBRID_SCAN_LIMIT=6000`). 전체 `tests/`: **6,572 통과 · 130 skipped · 1 실패** — 실패는 `tests/context/test_context_artifacts_v4.py::…::test_default_artifact_path_follows_custom_raw_sandbox`(`/tmp` ↔ `/private/tmp` 경로 비교, macOS 심볼릭 링크) 하나이고 **수정 전 main 체크아웃에서도 같은 테스트가 같은 이유로 실패**한다(이번 변경과 무관, 기존 문제 — 고치지 않았다).
- 외부 호출 차단: 모든 측정 스크립트가 서비스 모듈 import 전에 `v61_guard.install()`(Gemini·Claude·pymongo·`mongodb._connect` 생성 차단, `QDRANT_URL`·`CLAUDE_*`·`GEMINI_API_KEY` 제거, `HF_HUB_OFFLINE=1`)을 부르고 실행 전후 `verify()`가 전부 `blocked`임을 각 JSON(`guard_before`·`guard_after`, `cand_fixed/*/conditions.json`)에 적었다. 평가기 runinfo의 `lyrics_source.error = BlockedExternalCall`로 Mongo가 실제로 막혔음이 보인다. 리랭커 꺼짐(`gemini.backend = unused`). 테스트는 `:memory:`·`tmp_path`·가짜 클라이언트만 쓴다. 워크트리에 `.env`는 두지 않았다(복사 안 함). 모델 다운로드 없음(로컬 HF 캐시, 평가 실행에만 KoE5·SigLIP·CLAP 로컬 로드).
- 운영 `.env`·DB·색인 수정 0. 속성 보조·장르 정책·가중치·후보 수·프롬프트 변경 0. v10 test 미사용. 과거 실험 결과 파일 수정 0(v61 폴더에서 바뀐 것은 `run_candidates.py`의 None 비교 한 줄과 설명 문단만 — 결과 파일 아님).

## 9. 남은 한계와 다음 단계

1. **동시 적재 비보장**(§2) — 서버 모드 배포 전에 곡 컬렉션의 세대+별칭 발행 또는 "적재 중 검색 중단" 운영 절차를 정한다.
2. **가사 경로 있는 조건 미검증**(§7) — Mongo가 있는 환경에서 `run_candidates.py`를 가사 경로 켠 채 전후 한 번 더 돌려 가사 단서 19질의의 후보 지문을 비교한다(v61 §8-4 (마)). 그 결과가 같으면 품질 재측정은 불필요하고, 다르면 바뀐 질의만 리랭킹까지 다시 잰다. v61에서 확인된 영향(top-90 dense 몫 4곡)은 가사 질의 보조 신호에 들어가므로, 가사 경로가 켜진 조건에서 후보가 바뀔 가능성이 있는 질의도 그 19건이다.
3. **필터 count 비용**(서버 +34ms) — 필터 경로 비율이 낮아 지금은 수용 가능하다고 보지만, 속성 보조 경로를 켜면 필터 호출이 늘어난다. payload 색인 도입은 적재기 변경이라 후속.
4. **반환 항목이 대상 수에 비례**하는 구조는 그대로다(1만 곡이면 2만 항목). 비용이 문제가 되면 v61 §7의 다른 설계(페이지네이션·메모리 행렬)가 필요하다.
5. 지연은 로컬 PC 측정이며 동시 요청은 없다.
6. `QdrantIndex`에 `fetch`가 두 번 정의돼 있어 앞의 것이 죽은 코드다(이번 범위 밖, 기록만).

다음: 이 PR → 병합 뒤 메인 체크아웃에서 `.env`의 `QDRANT_HYBRID_SCAN_LIMIT`(없음) 확인 → (2) 가사 경로 켠 후보 비교 → 속성 보조 경로 활성화 결정(NEXT_WORK §3-3).

## 리뷰 반영 (2026-10-16 밤, 측정 결과 변경 없음)

| 지적 | 반영 |
|---|---|
| [P2] 검증 스크립트가 실패해도 성공으로 끝나 옛 결과를 다시 비교할 수 있다 — `run_candidates_fixed.sh`는 후보 생성 실패(덮어쓰기 거부 포함) 뒤에도 계속 진행해 `ALL DONE`·종료 코드 0, `run_all.sh`는 실패를 기록만 하고 비교로 넘어감, `replay_fixed.py`는 기준 불일치여도 0 | `run_candidates_fixed.sh`: 세트 하나라도 실패하면 `FAILED … (중단)`을 남기고 **그 종료 코드로 즉시 끝남**(확인: 명령을 `false`로 바꾸면 exit 1 · 산출물 있는 기본 경로에 `OVERWRITE` 없이 돌리면 덮어쓰기 거부 exit 1에서 중단). `CAND_ROOT`(새 출력 루트)·`OVERWRITE=1` 지정 가능. `run_all.sh`: 단계마다 종료 코드를 보고 실패면 `FAILED at …`과 함께 중단, 6에서 띄운 컨테이너는 EXIT 트랩으로 실패해도 종료, `CAND_ROOT` 전달. `replay_fixed.py`: `--expected-calls`(기본 311) 미달 · 기준 불일치 · `limit ≠ count` · dense 반환 ≠ 대상 수가 하나라도 있으면 `verdict.problems`를 적고 **exit 2**(확인: `--expected-calls 1`로 돌리면 exit 2, 실제 입력은 `REPLAY OK` exit 0). `compare_fixed.py`: `--fixed-root` 인자 추가, 세트별 `candidates.jsonl`·detail·**`conditions.json`**(실행이 끝까지 간 뒤에만 쓰임)이 없으면 비교하지 않고 중단 |
| (2차) `run_candidates_fixed.sh`의 `${OVERWRITE:+--overwrite}`는 비어 있지 않은 어떤 값(0 포함)에도 켜져 기존 결과를 보호하지 못함 | `OVERWRITE`가 정확히 `1`일 때만 `--overwrite`를 넘기도록 변경. 확인(인자만 기록하는 가짜 실행기): 미설정·`0`·`yes` → 플래그 없음, `1` → 플래그 전달 |

## 재현 (저장소 루트, 모델 호출 없음, `env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT`; 어느 단계든 실패하면 그 종료 코드로 끝난다)

```
# 준비(한 번): 색인 복사본 — v61 복사본을 다시 복사. v61 저장 질의 벡터는 V61_SOURCE_ROOT(기본 v61 워크트리 폴더)에서 읽기만 한다
cp -R /private/tmp/vague-finder-hybrid-scan-audit/artifacts/qdrant_v61_audit artifacts/ && rm -f artifacts/qdrant_v61_audit/.lock
zsh experiments/reranking/results_v62_hybrid_full_scan/run_all.sh      # 테스트 → 재생 → 후보 재생성 → 비교 → 로컬 지연 → Docker 서버 기동·이전 → 서버 지연 → 종료(실패해도 컨테이너는 정리)
#   후보 출력을 다른 곳에: CAND_ROOT=/path zsh …/run_all.sh   · 같은 곳에 다시: OVERWRITE=1 zsh …/run_all.sh
# 개별
venv/bin/python -m pytest tests/test_v62_hybrid_full_scan.py tests/test_v61_hybrid_scan_audit.py tests/test_qdrant_backend.py -q
venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/replay_fixed.py --label fixed --expected-calls 311   # 불일치·건수 부족·limit≠count면 exit 2
CAND_ROOT=<새 루트> zsh experiments/reranking/results_v62_hybrid_full_scan/run_candidates_fixed.sh && venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/compare_fixed.py --fixed-root <새 루트>
venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/bench_latency_v62.py --mode local --set v06 --max-calls 60 --filter-repeats 5
venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/server_setup_v62.py && venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/bench_latency_v62.py --mode server && venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/server_setup_v62.py --stop
```

## 파일

| 파일 | 내용 |
|---|---|
| `src/vector_db/qdrant_backend.py` | 수정(하이브리드 limit = count · `_count` · 폐기 호환값 경고 · 모듈 설명) |
| `tests/test_v62_hybrid_full_scan.py` · `tests/test_v61_hybrid_scan_audit.py` | 신규 15 · 갱신 13 |
| `docs/vector_backend.md` · `.env.example` | 점수 계산 표·주의(적재 중 검색 중단·옛 결과 재현)·설정 주석 |
| `v62_common.py` | 경로·guard 재사용·v61 입력 읽기·지문 확인·옛 구현 로드 |
| `replay_fixed.py` · `replay_fixed.json` · `replay_fixed_calls.csv` | 311건 재생 대조 |
| `run_candidates_fixed.sh` · `cand_fixed/cand_A/<set>_dev/` | 수정 구현의 후보 30개(summary·top10·runinfo·conditions 커밋, detail·explain·jsonl·npz 제외) |
| `compare_fixed.py` · `candidate_changes_fixed_vs_{A,B}.csv` · `candidate_summary_fixed_vs_{A,B}.json` · `detail_column_diff.json` | 후보 비교(v61 A·B ↔ 수정) · detail 열 대조 |
| `bench_latency_v62.py` · `bench_{local,server}_calls.csv` · `bench_{local,server}_summary.json` | 지연(수정 전 fc20e86 ↔ 수정 후, 교대, 단계별) |
| `server_setup_v62.py` · `server_setup_v62.json` | 전용 Docker Qdrant 기동·텍스트 컬렉션 이전·지문 확인·종료 |
| `run_all.sh` | 전체 순서 |

커밋 제외: `artifacts/qdrant_v61_audit`(226MB) · `artifacts/qdrant_server_v62/storage`(100MB) · `cand_fixed/*/*/{calls,candidates}.jsonl` · `calls_vectors.npz` · `logs/`(실행 로그 · 옛 구현 사본).
