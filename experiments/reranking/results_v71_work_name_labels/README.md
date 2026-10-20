# v71 작품명 해석 22질의 — 사람 라벨 준비·검증·집계

> **2026-10-19 갱신**: 라벨러 표 1부를 받아 보존·검증·대조했다 → 결과는 [`LABEL_RESULTS.md`](LABEL_RESULTS.md). 유효 15 · 확인 대기 4 · 미완료 3. 작성자·작성 시각은 미기록이고 확인자는 없다(단일 라벨러). 아래 §0·§2는 준비 당시 기록이고, 바뀐 점은 §9에 있다.

모델이 사용자가 뜻한 작품을 맞게 해석했는지 사람 라벨로 확인한다. 검색 성능(Hit@10·후보 회수)과는 따로 본다.

- 이 폴더의 작업에는 모델 호출·검색·외부 요청이 없다. 제품 코드(`src/`)도 바꾸지 않는다.
- 에이전트는 라벨을 채우지 않는다(이 세션은 v71·v72 모델 출력을 봤다). 도구는 사람이 채운 표의 형식을 검사하고 고정한다. 그리고 §5 규칙대로 센다.
- v73 전체 A/B와 분리한다. 워크트리·브랜치·결과 폴더가 다르다. v73 해석 캐시·결과·NEXT_WORK는 건드리지 않는다.

## 0. 시작 상태(2026-10-18 17:30 KST 확인)

| 항목 | 값 |
|---|---|
| 기준 | `origin/main` d8c3878(v72 #86 병합 뒤). 워크트리 `/private/tmp/vague-finder-v71-work-labels`, 브랜치 `eval/v71-work-name-labels`(커밋 전) |
| 사람 라벨 | **0/22**. v71 대기표 추적판·원문판, v72 `review_after_labels.csv` 추적판·비공개판의 라벨·검토 열이 모두 비어 있다. 다른 곳에서 채운 표도 찾지 못했다 |
| 라벨러·확인자 | 시작 때 미정 → 2026-10-18 PM 지정(이름은 `private/roles.md` — 개인 이름은 추적하지 않는다, §2). 노출 상태는 아직 확인 전 |
| 읽은 자료 | `CLAUDE.md`(`AGENTS.md`는 저장소에 없다), v71 `labels/LABEL_GUIDE.md`·대기표 2종·`RUN_INFO.md` §4·§7·§8, v72 `labels/LABEL_STATUS.md`·`review_after_labels.csv`(추적·비공개)·`RUN_INFO.md` §2-3 정정(16→13)·§4·§6 |
| 기존 도구 | v70 `new_data/work_label_check.py`는 **새로 쓰는 질의**의 작성자 작품 표시용이라 이번 일(기존 질의에 제3자가 작품을 판단)에는 맞지 않는다. 이번 양식은 v71 대기표 열을 그대로 포함해 넓힌 것이다(§6) |

### 왜 v71 원문 대기표를 그대로 주지 않는가

v71 `work_label_queue_private.csv`에는 질의 id(`c6xx`·`c7xx`은 v06 간접 지칭 묶음), `v67_known_case`, `v67_prior_label=v67_answer_derived`가 함께 있다. 라벨러가 이것을 보면 어느 질의가 알려진 사례인지 알게 된다. 그래서 질의 원문만 옮겨 담은 가린 표를 새로 만들었다. 질의 순서는 고정 시드(20261018)로 섞고 중립 번호 `L01`~`L22`를 붙였다. 대응표는 `private/item_map.csv`에만 둔다.

## 1. 재는 것 · 재지 않는 것

- 잰다: 질의만 보고 사람이 판단한 의도 작품과 v71 반복 1 모델 해석(채택·거부 작품)의 일치. 두 반복은 응답 원문이 같았다. 모델이 보류한 것이 맞았는지, 라벨러와 확인자가 일치하는지도 본다.
- 재지 않는다: 검색 순위 변화, 정답 곡과 작품의 실제 연결, 라벨이 없는 다른 질의.
- 정답 곡이 그 작품에 쓰였다는 이유로 사용자 의도를 정하지 않는다. 알려진 10건 밖의 12건을 오답·음성으로 치지 않는다.

## 2. 역할과 노출 기록(사람 응답 대기)

| 역할 | 사람 | 맡는 일 | 노출 상태 |
|---|---|---|---|
| 보관자 | (PM이 정함) | 가린 표 전달, 회수, `freeze`·`compare` 실행. 대응표·모델 출력에 접근 | 노출 무관(라벨 안 씀) |
| 라벨러 | PM 지정(2026-10-18, 이름은 `private/roles.md`) | 1단계 표 22행 | **미확인**. 받은 표는 22행 모두 `saw_*`=no(자기 보고)지만 `labeler_id`·`labeled_at`이 비어 있어 작성자를 확인하지 못했다(2026-10-19) |
| 확인자 | PM 지정(2026-10-18, 이름은 `private/roles.md`) | 같은 빈 표 22행, 라벨러 결과 없이 | **미작성**(2026-10-19 기준 빈 표 그대로) |
| 이름 쌍 판정자 | (보관자 또는 확인자) | 문자열로 못 가르는 작품명 쌍이 같은 작품인지 판정(§5-1) | 2단계 뒤라 노출 무관 |
| 에이전트 | — | 자료 준비·검사·집계만 | 모델 출력·정답·결과 모두 봄 → 라벨 불가 |

라벨러·확인자를 정할 때 보관자가 아래를 먼저 묻고 이 표에 적는다(이미 봤으면 숨기지 않고 적는다 — 행은 버리지 않고 독립 라벨과 따로 센다, §5-4).

1. v71·v72 결과(작품명 해석 출력, PR #85·#86 본문, 에이전트 보고)를 본 적이 있는가
2. dev 평가 질의의 정답 곡을 아는가(질의 작성자인가, `eval_queries_*.csv`·`queries.json`을 봤는가)
3. 저장소 v67 진단 자료(`oracle_work_name.py`·`failure_review_v67.csv` — 정답 곡에서 거꾸로 적은 작품명이 있다)를 본 적이 있는가
4. 검색 순위·Hit 결과(v70~v72 질의별 표)를 본 적이 있는가

PR #84~#86 본문과 v70~v72 추적 문서에는 모델이 낸 작품명이 없다(2026-10-18 확인). 모델 작품명은 Git 제외 파일에만 있다. 반면 v67 추적 파일 2개에는 정답에서 거꾸로 적은 작품명이 있다.

## 3. 흐름과 명령

경로는 저장소 루트 기준이다. `T=experiments/reranking/results_v71_work_name_labels`로 둔다.

| 단계 | 누가 | 무엇 | 명령 |
|---|---|---|---|
| 0 준비(완료) | 에이전트 | 가린 표 2부(라벨러·확인자, 같은 내용) · 대응표 · 해시 기록 | `venv/bin/python $T/work_label_tool.py prepare --source <v71 원문 대기표>` |
| 1 전달 | 보관자 | 역할별로 `private/packet/<역할>/`의 **두 파일만**(`work_label_sheet_<역할>.csv`, `LABELER_GUIDE.md`) 보낸다. 다른 파일·폴더·저장소 링크는 보내지 않는다 | — |
| 2 회수·검사 | 보관자 | 받은 표를 `private/returned/`에 둔다. 오류가 있으면 라벨러에게 **형식만** 되묻는다(내용 힌트 없이) | `… work_label_tool.py check --sheet <표>` |
| 3 고정 | 보관자 | 오류 0 · 22행이면 고정본으로 둔다. 읽기 전용이고 해시를 `freeze_log.jsonl`에 남긴다. 같은 역할을 다시 고정하면 새 판(r2…)이 되고 사유가 필요하다. **r1은 덮어쓰지 않는다** | `… freeze --role labeler --sheet <표>` |
| 4 독립 확인 | 확인자 | 라벨러 결과·모델 출력 없이 같은 빈 표로 22건 모두. 고정 방법은 같다 | `… freeze --role checker --sheet <표>` |
| 5 일치 대조 | 보관자 | 라벨러 r1 ↔ 확인자 r1. 문자열이 다른 작품명은 `private/name_pairs.csv`에 판정 대기로 올라간다 | `… agree` |
| 6 2단계 열기 | 보관자 | 고정 뒤에만 열린다. v71 모델 출력 사본을 `private/stage2/`에 들인다. 이때부터 특별 점검 표시를 붙인다 | `… compare --model-outputs <v71 analysis_private/model_output_review.csv>` |
| 7 이름 쌍 판정 | 판정자 | `name_pairs.csv`의 `same_work`(yes · no · unsure)와 `decided_by`를 채운다 | — |
| 8 집계 | 에이전트 | §5 규칙. 판정 대기가 남으면 종료 코드 2로 끝나고 대기 수를 적는다 | `… summarize` |

## 4. 확인자가 없을 때

`compare --no-checker`로 연다. 결과에 "독립 확인 없음"을 붙이고 라벨러 1인 판단으로만 보고한다. 확인자는 2단계를 열기 **전에** 고정해야 독립 확인으로 센다. 2단계를 연 뒤에 들어온 확인자 고정본은 `after_stage2_opened=true`로 기록되고, 보고서에서 독립 확인으로 쓰지 않는다. 그 뒤에 들어온 확인자 라벨은 보관자가 모델 출력을 보여 주지 않았음을 확인할 때만 참고로 쓴다.

## 5. 집계 규칙(라벨이 오기 전 2026-10-18 고정 — 결과를 본 뒤 바꾸지 않는다)

### 5-1. 작품 이름이 같은지

- 도구는 NFKC·소문자·공백·문장부호 제거 뒤 문자열이 같을 때만 같은 작품으로 본다. 모델 작품은 이름과 별칭 중 하나라도 같으면 같다.
- 그 밖의 쌍(약칭·영문명·다른 표기)은 사람이 `name_pairs.csv`에 판정한다. `unsure`이거나 비어 있으면 그 항목은 **판정 대기**로 남는다. 정답·오답 어느 쪽에도 넣지 않는다.
- 동명 작품(리메이크·시즌)은 라벨의 `work_identifier`와 대조해 판정자가 정한다. 모델 출력에는 연도가 없다.

### 5-2. 항목 분류(라벨러 r1 기준, 모델은 v71 반복 1)

| 라벨 상태 | 모델 | 분류 | 잘못된 작품 연결 |
|---|---|---|---|
| 판단불가 | 무엇이든 | `label_undeterminable` — 세지 않음 | 판단 불가 |
| 작품아님 | 채택 있음 | `wrong_link` | yes |
| 작품아님 | 채택 없음(보류·거부) | `correct_abstain` | no |
| 단일·복수 | 채택 작품 전부가 라벨 작품(복수면 가능한 작품 중) | `correct` | no |
| 단일·복수 | 채택 작품 일부만 라벨 작품 | `partial` | yes(여분 작품이 검색에 들어감) |
| 단일·복수 | 채택 작품 중 라벨 작품 없음 | `wrong` | yes |
| 단일·복수 | 보류 | `missed_abstain` | no |
| 단일·복수 | 모두 검증 거부(`no_local_data`) | `missed_rejected`. 거부 작품 속에 라벨 작품이 있었는지 따로 표시 | no |

채택 작품만 검색에 쓰이므로 채택 집합으로 판정한다.

### 5-3. 분모와 비율

- 정답·부분·오답 비율의 분모는 **모델이 작품을 채택했고 라벨이 판단불가가 아닌 항목**이다(`linked_judgeable`). 판정 대기는 분모에서 빼고 수를 따로 적는다.
- 보류·거부율은 22건 분모(v71에서 이미 6/22)다. 그중 맞는 보류(`correct_abstain`)와 놓침(`missed_*`)을 나눈다.
- 같은 표를 다음 묶음으로 따로 낸다: 전체 22 · 독립 라벨 행만 · 알려진 10건 · 그 밖 12건(음성 아님).

### 5-4. 독립 라벨

`saw_model_output_before` · `saw_experiment_results_before` · `saw_v67_work_list_before`가 모두 `no`인 행만 독립 라벨이다. `unsure`는 독립 아님으로 센다. `saw_answer_song_before`는 v71 안내대로 행을 빼지 않고 수만 적는다. 정답 곡을 아는 라벨러가 많으면 결과에 그 사실을 붙인다.

### 5-5. 라벨러·확인자 일치

`agree`(일치) · `disagree_status`(작품아님 대 작품) · `disagree_work`(다른 작품) · `partial_overlap`(복수 집합이 일부 겹침) · `agree_works_status_differs`(작품 집합은 같고 단일·복수만 다름) · `one_undeterminable` · `both_undeterminable` · `pending_name_pair`로 나눈다. 일치율은 계산하되 **불일치를 한쪽으로 고치지 않는다**. 주 결과는 라벨러 r1이고, 확인자 r1 기준 결과와 둘이 `agree`인 항목만의 결과를 함께 적는다. 결론이 둘 사이에서 달라지면 결론을 쓰지 않고 그 항목을 적는다.

### 5-6. v72 결과와의 연결(v72 RUN_INFO §4-6)

v72 안정 이득 4건(n033·n043·n096·c707) 각각을 정한다. 잘못된 작품 연결=yes면 "의도 해석 근거로 세지 않음", `correct`면 "근거로 셀 수 있음", 판단불가·판정 대기면 "모름"이다. 검색 이득 숫자 자체는 바꾸지 않는다.

### 5-7. 특별 점검(2단계에서만 붙인다 — 라벨러 자료에는 없다)

| 질의 | 표시 | 볼 것 |
|---|---|---|
| q105 · n031 | `name_in_query` | 질의 낱말과 같은 이름이 채택됐다. 라벨이 작품아님이면 `wrong_link` |
| n096 | `work_name_equals_song_title` | 이득이 있어도 작품 의도 해석의 근거로 따로 본다 |
| n045 · t024(+q105) | `applied_input_identical` | 채택됐지만 리랭커 입력이 같다. 정오는 세지만 검색 효과는 0 |
| c603 · n010 · t036 · t041 · t098 | 보류 | 라벨이 단일이면 놓침 |
| t043 | 검증 거부 | 거부 작품 속에 라벨 작품이 있었는지 |

## 6. 기존 양식과의 대응

| v71 대기표 열 | 이 표 |
|---|---|
| `intended_work` | `intended_work` |
| `alternative_works` | `possible_works` |
| `multiple_possible` | `judgment_status=복수` |
| `undeterminable` | `judgment_status=판단불가` |
| `evidence` | `query_phrase` + `sources` |
| `labeler_id` · `labeled_at` · `saw_model_output_before` · `saw_answer_song_before` · `notes` | 그대로 |
| (없음) | `work_type` · `work_identifier` · `external_check` · `memory_mismatch` · `confidence` · `open_questions` · `saw_experiment_results_before` · `saw_v67_work_list_before` · `작품아님` 상태 |

v72 `review_after_labels.csv`의 `model_works_match_label`은 위 분류에 이렇게 대응한다: `correct`→yes, `partial`→partial, `wrong`·`wrong_link`→no, `label_undeterminable`→undeterminable. `wrong_work_link`는 같은 열이다. 이번 일로 v71·v72 추적 파일은 고치지 않는다(병합된 기록). 대응 결과는 이 폴더의 `label_summary.json`에만 쓴다.

## 7. 파일과 Git 제외

| 경로 | 내용 | Git |
|---|---|---|
| `README.md` · `LABELER_GUIDE.md` · `HANDOFF.md` | 문서. 라벨러 안내는 질의 id·모델 출력·사례 표시가 없다 | 추적 |
| `work_label_tool.py` · `selftest_work_label_tool.py` | 도구·자체 검사 | 추적 |
| `packet_manifest.json` | 가린 표·대응표·원문 대기표·안내문의 sha256, 시드 | 추적 |
| `freeze_log.jsonl` · `agreement_summary.json` · `label_summary.json` | 고정 기록·집계(질의 원문·작품명 없음) — 라벨이 온 뒤 생김 | 추적 |
| `private/` | 원문 대기표 사본, 가린 표, 대응표, 회수·고정본, 이름 쌍, 2단계 대조표·모델 출력 사본 | **제외** |
| `NEXT_WORK_SUMMARY.md` | NEXT_WORK 반영용 요약(v73 세션이 통합) | **제외** |

`private/`에는 2단계를 열기 전까지 모델 출력이 없다(`packet_manifest.json`의 `model_output_in_this_folder=false`). 2단계는 `compare` 실행 때 v71 워크트리의 `analysis_private/model_output_review.csv`(sha256 `ec7bf3a6…afe56`)를 복사해 연다.

## 8. 자체 검사

`venv/bin/python $T/selftest_work_label_tool.py` 는 가짜 자료만 쓰며 `selftest ok`를 냈다.

- 일부러 넣은 입력:
  - 덜 채운 표, 고친 원문, 형식 오류 9종
  - 사유 없는 재고정, 해시 변조, 고정 전 2단계 열기
  - 판정 대기 이름, 노출 `unsure`, CP949로 다시 저장한 표
  - (2026-10-19 추가) 전달용 표에 바로 채운 경우, 오류 포함 원본 보존, 기계적 정리, 확인 대기 행, 에이전트 의견과 사람 판정의 분리, 별칭 일치, 이미 맞은 작품의 불필요한 쌍
- 도구에 결함을 하나씩 심어 확인했다. 처음 6개와 2026-10-19 추가 6개가 모두 실패로 잡혔다. 추가분 중 하나(정식 고정에서 `--provisional` 확인 제거)는 검사 항목으로 잡힌 것이 아니라 시험 실행이 중단되는 방식으로 잡혔다.

## 9. 라벨 수령 뒤 변경(2026-10-19)

- **수령**: 라벨은 보관자 폴더의 전달용 표(`private/packet/labeler/`)에 바로 채워져 있었다. 받은 그대로 `private/returned/`에 읽기 전용으로 복사하고 r1(`--provisional --kind as_received`)으로 고정했다. 기계적 정리만 한 작업용 사본은 r2(`normalize` → `--kind mechanical_cleanup`)다. 사람 확인을 반영하는 수정은 r3(`human_adjusted`)로만 한다.
- **도구 확장**(기존 도구 안에서, 시험 추가):
  - `freeze --provisional`: 오류·미완료가 있어도 원본을 보존하고, 행 상태(valid · needs_confirmation · incomplete · blank)를 기록한다.
  - `normalize`: 빈 열·빈 행·앞뒤 공백·상태 표기 공백만 고친다. 변경 내역 CSV를 남긴다.
  - compare·summarize `--revision`.
  - 이름 쌍 표에 `agent_suggestion`·`agent_basis` 열(에이전트 보조 의견, 사람 판정 칸과 분리).
  - 모델 3단계(생성·채택·사용) 집계, v73 질의별 기록과의 교차.
- **검사 기준 정정**: 처음에는 전달용 표를 빈 표 기준으로 읽었다. 라벨러가 그 표에 바로 채워 기준이 오염됐다. 그래서 `private/packet_blank/`(prepare 해시 `7e8fe5a2…`와 같은 사본)만 기준으로 쓰도록 고쳤다. 첫 고정 기록 2줄은 2단계 전에 지우고 같은 파일로 다시 고정했다(해시 동일).
- **§5 규칙 적용 방식**: 규칙 자체는 바꾸지 않았다. 확인 대기·미완료 행은 주 결과에서 `label_needs_confirmation`·`label_incomplete`로 세지 않는다. 적힌 그대로 넣은 값은 민감도로만 따로 낸다. §5-2의 "단일·복수 + 보류 = 놓침"은 단일·복수를 나눠 보고한다(`outcome_by_label_status`).
- **2단계**는 확인자 없이 열었다(`--no-checker`, 2026-10-19 18:01). 이후 확인자 고정본은 `after_stage2_opened=true`로 기록되고 §4대로만 쓴다.
- **검토 뒤 수정(외부 검토 3건)**:
  - 집계 전 모델 출력 사본 해시 검증, 최초 공개 기록 보존.
  - 이름 판정은 사람 판정이 먼저이고, 모델 별칭만으로는 같은 작품으로 확정하지 않는다.
  - 셀 안 줄바꿈 보존.
  - 상세는 `LABEL_RESULTS.md` §10.
- **추적 파일 추가**: `LABEL_RESULTS.md` · `label_summary.json` · `freeze_log.jsonl`(질의 id·집계만). 비공개 추가: `private/returned/` · `private/frozen/` · `private/packet_blank/` · `private/stage2/`(대조표·교차표·질의별 결과) · `private/name_pairs.csv` · `private/confirmation_list.csv`.

- **보완본(2026-10-20)**: 보완본을 받은 그대로 r3(`--provisional --kind human_adjusted --ai-assisted`)로, 날짜 구분자만 고친 사본을 r4(`mechanical_cleanup`)로 고정했다. 파일이 생성형 AI 서비스에서 내려받은 것이라 사용자 지시로 AI 보조로 기록했다. 도구에 `freeze --ai-assisted`, 날짜 구분자 정리, 고정 기록의 작성자 해시화를 더했다. 결과와 표현 제한은 `LABEL_RESULTS.md` §11.
