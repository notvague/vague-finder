# v72 작품명 해석으로 입력이 바뀐 질의의 실제 리랭킹 효과 — 부분집합 진단(dev)

작성 시작 2026-10-18 13:40 KST. 워크트리 `/private/tmp/vague-finder-v72-work-name-rerank` · 브랜치 `experiment/v72-work-name-rerank-diagnosis` · **기준 커밋 `5c2d43e`**(origin/main, v71 PR #85 병합). 평가 PR이며 제품 코드(`src`·`tests`) 변경은 없다.

범위: 고정된 해석 캐시(v71 반복 1)로 **리랭커 입력이 바뀐 질의만** 실제 Vertex 리랭킹 A1→B1→B2→A2. 운영 활성화·전체 176 리랭킹·작품명 재해석·정책 조정은 하지 않는다. **§0~§4는 실제 호출 전에 썼다.**

## 0. 시작 상태(2026-10-18 13:40 KST 확인)

| 항목 | 상태 |
|---|---|
| v71 | **PR #85 MERGED**(2026-10-18 04:37 UTC, `5c2d43e`). v71 워크트리 head `fccb06a`와 병합 트리가 같다(`git diff fccb06a origin/main` 빈 출력). v71 워크트리는 읽기·복사만 했다 |
| 기준 코드 | `5c2d43e`. v70 기준 `a55e498`과 `src`·`tests`가 같다(`git diff a55e498 5c2d43e -- src tests` 0줄) → v71 결과와 같은 코드 |
| 코드 사본 | `artifacts/src_main_5c2d43e`(`git archive 5c2d43e src`, `.py` 162개 트리 지문 `eaf159683c73e5d5`). 하네스가 실행마다 사본 blob을 커밋 blob과 대조한다 |
| 해석 캐시 | v71 `resolution_cache/rep1.json`의 바이트 복사본 `resolution_cache/v71_rep1.json`(sha16 `03ad094834c986e5`, Git 제외). 반복 2 캐시는 쓰지 않는다(합치거나 덮어쓰지 않음) |
| 언급 저장소 | v71 `artifacts/work_names/work_mentions_v1.json.gz` 복사본(sha16 `bfd522ebe9da7590`, v70·v71과 같음) |
| 전용 Qdrant | `vague-finder-qdrant-v72`(127.0.0.1:6380, `qdrant/qdrant:v1.19.1`) — v71 저장소 파일 복사본(828파일, 트리 지문 원본과 같음). 컬렉션·별칭이 v71 `server_state_final.json`과 같다(`server_state_baseline.json`) |
| 메인 체크아웃 | 건드리지 않음(`.env`·색인·DB·캐시 변경 없음). `map_data.json`만 복사 |

## 1. 고정 조건

| 항목 | 값 |
|---|---|
| A | 작품명 해석 OFF(운영 기본, `WORK_NAME_RESOLUTION_ENABLED=0`) |
| B | `WORK_NAME_RESOLUTION_ENABLED=1` · `WORK_NAME_RESOLVER=cache:resolution_cache/v71_rep1.json` — **해석 호출 0**. 실제 해석 어댑터(`VertexWorkNameResolver.resolve`)는 하네스가 막는다(불리면 실행 중단) |
| 공통 | 소리 ON(`AUDIO_FUSION_SCALE=1`) · Context 0.5 · D OFF(`ATTR_FILTER_WEIGHT=0`) · 피드백 OFF · 후보 30 · 보조 가중 0.5(기본) · `gate_v2`·`validate_v3`·`resolver_prompt_v2`(`31494c3f69f77481`) · v49 Vertex 분석 캐시 · v63 가사 스냅샷 `e0d58f28f1f4a563` · 색인 지문 `238299e6c8c17f2a` |
| 리랭커 | Gemini listwise `gemini-3.1-flash-lite` · Vertex `global` · 1패스 · Search 끔 · 희소 사실 검증 끔 · 요청 제한 20초 · 질의당 예산 30초 · 그룹 재시도 최대 3(코드 기본값 그대로 — 비용을 맞추려고 바꾸지 않음) · 가사 exact 보호·리랭킹 정책 그대로 |
| 실행기 | 리뷰된 v69 하네스(`run_eval.py`·`v69_eval_runtime.py`·`v69_budget.py`·`compare_inputs.py`·`run_live.sh`) 사본. 바꾼 점은 아래 §2 |

## 2. 실행 전 무료 검증(모델 호출 0)

### 2-1. 하네스 변경(v69 대비)

- 구성 `off|work`, 코드 사본 `5c2d43e`, 전용 Qdrant 6380. 기록 모듈에 `work_name_resolution`·`work_mentions`·리랭커를 더했다.
- **고정 부분집합 실행**: `--subset-file` + `--expect-subset-sha16`. live는 둘 다 있어야 하고 지문이 같아야 시작한다. `--query-ids`·`--sets` 부분 실행은 live에서 계속 거부한다. 부분집합에 질의가 없는 세트는 평가기를 부르지 않는다.
- 해석 확인: 실행 전 라우터 해석기 종류(off = 없음, work = `cache:`), 세트마다 runinfo의 해석 스위치·버전·저장소 지문·보조 가중 0.5, 실제 해석 호출 0.
- **예열**(아래 2-4에서 발견): 기록을 걸기 전에 고정 규칙으로 고른 질의 하나를 리랭킹 없이 한 번 검색한다.
- 예산 장치 `v72_budget.py` = `v69_budget.py` 그대로(설치 표시 이름만 바꿈). 연속 504 3회 · 10분 안 504 5회 · 인증·권한·결제 첫 오류 · 연속 실패 5회 · 누적 실패 20회 · 시도 상한 · 비용 상한 · 첫 시도 기준 마감. 모델 최대 출력 65,536토큰으로 예약, 응답 `usageMetadata`로 정산, 실패·열린 예약은 미확인으로 유지(0으로 지우지 않음).

### 2-2. A·B 재현(dev 176 전체, 가짜 결정적 리랭커)

| 비교 | 후보 id·순서·점수 | 가사 보호 곡 |
|---|---:|---:|
| A(`offline/full/off_full_fake`) ↔ v71 `offline/off` | **176/176** | 176/176 |
| B(`offline/full/work_full_fake`) ↔ v71 `offline/live_rep1` | **176/176** | 176/176 |

해석 상태(B): 판정 밖 154 · 채택 16 · 보류 5 · 검증 거부 1 — v71 B1과 같다.

### 2-3. 리랭커 입력이 바뀐 질의 — **13건(v71 보고 16건과 다름 → 원인 확인)**

A↔B를 질의마다 후보 id·순서·점수, 가사 결과, 보호 그룹·tier·정책 생략 사유, 리랭커 그룹 입력, **실제 전송 프롬프트**(그룹별 첫 호출 sha16)로 비교했다(`freeze_subset.py` → `analysis/precheck_full.json`).

- 입력이 다른 질의: **13건** — c602 · c607 · c701 · c705 · c707 · m103 · n031 · n033 · n038 · n043 · n096 · t037 · t046.
- 채택 16건 중 **3건(n045 · q105 · t024)은 입력이 A와 바이트 단위로 같다.** 원인: 보조 곡(2 · 1 · 1곡)이 A 후보 30에 하나도 없고, 융합 뒤에도 30 경계를 넘지 못했다(들어온 곡 = 밀려난 곡 = 0). 그래서 후보 id·순서·점수가 그대로이고 프롬프트도 같다.
- **v71 문서의 "리랭커 입력이 바뀌는 질의는 채택 16건, 나머지 160건은 입력이 같다"는 틀렸다.** 바른 값은 **13건 / 163건**이다. v71은 "채택" 상태를 "입력 변경"과 같게 셌다(v71 §8-2의 "들어온 수 = 밀려난 수 16/16"에는 0 = 0인 3건이 들어 있었다). 코드·하네스 결함이 아니라 v71 보고의 집계 오류다. v71 문서의 본문 숫자는 병합된 기록이라 그대로 둔다. 대신 v71 `RUN_INFO.md`·`FOLLOWUP_AB_PLAN.md`·`PR_DRAFT.md` 첫머리와 틀린 문장 옆에 이 절을 가리키는 정정 메모를 달았다(v72 PR에 포함). 결과 파일(json·csv)은 고치지 않았다.
- 반대로 c607·n038은 들어온 곡이 0인데 입력이 바뀌었다. 보조 목록 곡이 이미 A 후보 30 안에 있어 점수·순서가 바뀌었기 때문이다.
- 입력이 바뀌었는데 모델을 부르지 않는 질의(정책 생략·후보 1개 이하): **0건**. B의 정책 생략 질의 21건은 모두 입력이 같은 163건 안에 있다.
- 13건 모두 그룹 1개 · 생성 호출 1회(가짜 실행 기준) · 프롬프트 25,264~25,938자 · 가사 보호 곡 0.
- 입력이 같은 163건(판정 밖 154 · 보류 5 · 검증 거부 1 · 채택이지만 입력 같음 3)은 리랭킹을 부르지 않는다.

**부분집합 확정**: `subset/rerank_input_changed.json`(13건, sha16 **`7004bca4e3f7deb1`**, 근거 `precheck_full.json` sha16 `2d7bbe7084a223ca`). 규칙은 "A↔B 입력 지문 차이"뿐이다. 정답·모델 결과를 보고 더하거나 빼지 않았다.

### 2-4. 부분집합 실행 = 전체 실행 확인과 예열

- 부분집합 1차 가짜 실행은 실행기의 질의 순서 검사에서 멈췄다(평가기는 `--query-ids` 순서로 돈다). dev 순서로 넘기도록 고쳤다. 2차 가짜 실행에서는 **B의 m103이 `failed:timeout`**이 되어 전체 실행과 입력이 달랐다(c602도 경로 반환 차이).
  - 원인: 라우터는 작품명 작업을 해석기 제한(4초) + 여유만큼 기다린다. 새 프로세스의 **첫 판정 통과 질의**는 인코더 지연 로드 등 첫 검색 비용을 그 안에서 떠안는다. 전체 실행에서는 앞 질의들이 예열해 주지만, 부분집합에서는 m103이 프로세스 첫 질의였다.
  - 서비스 측 뜻: 서버를 띄운 직후 첫 판정 통과 요청도 같은 이유로 해석을 놓칠 수 있다(미측정, 후속 기록).
  - 조치(실행기만): 기록을 걸기 전에 **부분집합 밖·판정 통과·dev 순서 첫 질의(q105)**를 리랭킹 없이 한 번 검색한다. 정답을 보지 않는 고정 규칙이다. 모델 호출이 없고, 기록·평가 산출에 들어가지 않는다. A·B 모두 같게 한다. 실패한 1·2차 가짜 산출물(모델 0회)은 지웠다(`logs/progress_offline.txt`에 기록).
- 예열 조건(A·B 같음):
  - 질의: dev 순서(v06 → v09 → v10)로 훑어 처음 나오는, 부분집합 13건 밖이면서 판정을 통과한 질의 하나 → **q105**(v06). q105는 입력이 A와 같은 3건 중 하나라 부분집합 결과에 영향이 없다.
  - 호출: `router.search(top_k=30, use_rerank=False, candidate_k=30)` 한 번. 리랭커·Gemini 분석을 부르지 않는다(분석은 고정 캐시). B에서는 작품명을 캐시 해석기로 읽는다(실제 해석 호출 0).
  - 시점: 예산 장치와 차단 장치를 건 **뒤**, 기록기를 걸기 **전**. 그래서 예열 결과는 평가기 기록·지표·top-10에 들어가지 않는다. 장부에는 예열 시도가 없다(실행당 16회 = countTokens 3 + 생성 13).
  - live 실측(`live/*/run_conditions.json`의 `warmup`): A1 2.805초 · A2 2.820초(해석 없음), B1 3.704초 · B2 3.640초(해석 `applied`).
- 예열을 넣은 전체 실행(`*_full_warm`)은 예열 없는 전체 실행과 **176/176 같다**(A·B 각각, 최종 순서 포함).
- 부분집합 실행(`offline/subset/*_subset_fake`)은 전체 실행의 같은 질의와 **13/13 같다**(A·B 각각 — 후보·가사·보호·그룹 입력·프롬프트·최종 순서·Qdrant 경로 반환). 부분집합 실행 때문에 후보 수·그룹·보호 정책이 달라지지 않는다.
- A 부분집합 ↔ B 부분집합: 13/13 입력이 다르다(의도한 차이).

### 2-5. 자체 검사

- `selftest_budget.py`(v69 사본, 대상 `v72_budget`) — FAILURES 0.
- `selftest_runner.sh` **53/53**(모델·Vertex·Mongo 호출 0, 임시 폴더만):
  - live 게이트(승인 표시·한도·부분집합 파일·지문)와 부분 실행·덮어쓰기·서버 상태·실패 주입 거부.
  - 묶음: 순서 A1→B1→B2→A2 · 한도와 부분집합 지문 고정 전달 · 장부 하나 · 단계 실패·입력 차이에서 즉시 종료.
  - 가짜 전송(fault): 504 3회 정지 · 정지 장부 상속 · 403 첫 오류 정지 · 미확인 비용 보존 · 외부 연결 0 · work 해석기 = 캐시.
  - 실제 해석 어댑터 차단(F).
- 제품 코드 변경이 없어 전체 시험(`pytest tests/`)은 다시 돌리지 않았다.

## 3. 가격·한도 사전 계산(실행 직전)

- 단가: 2026-10-18 13:57 KST 공식 Vertex 가격표를 내려받았다. 주소는 `…/gemini-enterprise-agent-platform/generative-ai/pricing`로 이동했고 본문 sha16은 `7e1efed71ff6164d`(동적 페이지라 v71과 다름). 표에서 직접 읽은 gemini-3.1-flash-lite **Standard · Global**(≤200K)은 입력 text $0.25/1M · 출력(response and reasoning) $1.50/1M으로 `v72_budget.PRICES`와 같다. 비교로 Priority는 $0.45/$2.70, Flex/Batch는 $0.125/$0.75다. 코드는 등급을 지정하지 않는다.
- 부분집합 실제 그룹 수: 질의당 1그룹 × 13 = 실행당 생성 13회. preflight countTokens는 세트당 1회 × 3세트 = 실행당 3회($0).

| 경우 | 계산 | 한도 안인가 |
|---|---|---|
| 계획 시도 수 | (13 + 3) × 4 = **64** | ≤ 200 예 |
| 최악 시도 수(그룹마다 재시도 3회 모두) | (13 × 3 + 3) × 4 = 168 | ≤ 200 예 |
| 예상 확정 비용 | 프롬프트 25.3k~25.9k자 ≈ v69 평균(약 10.5k 입력 토큰 · 1.4k 출력, 생성 1회 $0.0047) → 52회 ≈ **$0.25** | — |
| 전부 성공 시 최대 약정 | 확정 ≈ $0.25 + 열린 예약 1(질의 순차·질의당 1그룹) ≈ $0.11 = $0.36 | ≤ $1.50 예 |
| 재시도 3회가 모두 성공 응답(형식 오류 정정) | 확정 ≈ $0.74 + 예약 $0.11 = $0.85 | 예 |
| 실패 누적 | 실패 1회당 미확인 ≈ 예약액 $0.11(본문 약 58k자 × $0.25/1M + 65,536 × $1.50/1M) → 흩어진 실패 약 10회 뒤 비용 상한이 먼저 막는다. 몰린 실패는 연속 504 3회·연속 실패 5회가 먼저 막는다 | 상한을 넘는 지출 없음 |
| 시간 | 정상 약 1분/실행 + 오프라인 지문 전후 약 1분 → 약 10~15분. 최악(질의당 예산 30초 × 13) 약 9분/실행 → 약 36분 | ≤ 60분 예 |

**판단(실행 전)**: 승인 한도($1.50 · 200회 · 60분)는 계획된 네 실행을 감당한다. 반복을 줄이거나 장부를 나누지 않는다. 한도·정지 규칙으로 멈추면 완료·미완료를 나눠 그대로 보고하고, 다시 돌리지 않는다.

## 4. 판단 원칙(실행 전 고정 — 결과를 본 뒤 바꾸지 않는다)

**분모·지표**: 실행마다 부분집합 13건 전체를 분모로 Hit@1·5·10, MRR@10을 낸다. 실패·부분 실패·예산 소진·정책 생략을 빼지 않는다(실패는 평가기 그대로 검색 순서 폴백 결과로 채점). 보조로 네 실행 모두 리랭킹 `applied`인 질의만의 값을 따로 적는다.

**질의별 분류(Hit@10 기준)**:
- 안정 이득: A1·A2 둘 다 10위 밖, B1·B2 둘 다 10위 안.
- 안정 손실: A1·A2 둘 다 10위 안, B1·B2 둘 다 10위 밖.
- 불안정: 같은 구성 두 반복이 서로 다른 질의(A1≠A2 또는 B1≠B2).
- 같음: 네 실행이 모두 같은 결과.
- 실패 영향: 네 실행 중 하나라도 리랭킹 상태가 `applied`가 아닌 질의는 따로 표시하고 원인(실패·부분·예산)을 적는다. 주 지표에는 그대로 넣는다.

**변동**: A1↔A2, B1↔B2의 질의별 Hit@10 뒤집힘 수와 최종 10위 목록 차이를 센다. 뒤집힘은 (a) 한쪽이라도 `applied`가 아닌 경우와 (b) 둘 다 `applied`인데 출력이 다른 경우로 나눈다. 과거 실행의 변동을 오차 범위로 쓰지 않는다. 입력이 같은 163건은 이번에 부르지 않으므로 같은 입력 대조군은 같은 구성 반복 2쌍뿐이다.

**전체 A/B 권고 기준(모두 충족 → "계획·예산 작성 권고", 하나라도 아니면 "보류" 또는 "권고 안 함")**:
1. 안정 이득 − 안정 손실 ≥ 1.
2. 기존 성공 질의(A1·A2 둘 다 10위 안)에서 안정 손실 0. 손실이 있으면 질의별 원인을 적고 보류한다.
3. 이득이 n096(작품명 = 곡 제목) 한 사례에만 의존하지 않는다. n096을 빼면 안정 이득이 0이면 "근거 약함 — 보류".
4. 같은 구성 반복의 Hit@10 뒤집힘(A1↔A2 + B1↔B2)이 안정 이득 수 이상이면 변동 때문에 결론을 내리기 어렵다고 보고 보류한다.
5. 실패 영향 질의가 이득·손실 판정을 바꿀 수 있으면(그 질의를 반대로 셌을 때 1~4 결론이 달라지면) 보류한다.
6. 사람 라벨에서 잘못된 작품 연결이 확인된 이득은 의도 해석의 근거로 세지 않는다. 라벨이 없으면 "해석 정확도 미측정"을 결론에 붙인다.

새 후보 3건(n033·n043·n096)이 10위 안에 들어도 그 사실만으로 작품 의도를 정확히 해석했다고 쓰지 않는다. 이번 결과로 가중치·판정·프롬프트를 바꿔 다시 재지 않는다.

---

이하 §5~는 실제 호출 뒤(2026-10-18 14:07~)에 작성했다. §0~§4는 실행 전에 쓴 그대로다.

## 5. 실제 호출(승인 범위 안, 2026-10-18 13:59:46~14:06:21 KST)

`run_live.sh` 한 번으로 A1→B1→B2→A2를 순차 실행했다(13:59:17 시작, 14:06:37 `LIVE DONE`, 종료 코드 0). 실행마다 직전 지문 = 기준(13/13), 실제 실행이 보낸 입력 = 직전 지문, 직후 지문 = 직전 지문 검사를 모두 통과했다(`live_checks/`).

| 실행 | SDK HTTP 시도 | 실패 · 재시도 | 확정 비용 | 미확인 · 열린 예약 | 생성 지연 중앙값 / p95 | 리랭킹 상태(13) |
|---|---:|---:|---:|---:|---:|---|
| A1(off) | 16(count 3 + generate 13) | 0 · 0 | $0.063200 | $0 · 0 | 3,971 / 4,517ms | applied 13 |
| B1(work) | 16 | 0 · 0 | $0.063083 | $0 · 0 | 4,233 / 5,048ms | applied 13 |
| B2(work) | 16 | 0 · 0 | $0.063083 | $0 · 0 | 4,454 / 5,122ms | applied 13 |
| A2(off) | 16 | 0 · 0 | $0.063200 | $0 · 0 | 4,274 / 4,552ms | applied 13 |
| **합계(공유 장부)** | **64 / 200** | 0 · 0 | **$0.2526 / $1.50** | **$0** | — | partial · failed · budget_exhausted · policy_skipped 0 |

- 첫 외부 시도 13:59:46 → 마지막 기록 14:06:21(6분 35초, 마감 60분). 정지 기록 0 · 504 0 · 인증 오류 0 · SDK 단 재시도 0 · 리랭커 그룹 재시도 0.
- 출력 토큰(thinking 포함) 실행당 약 18.0k, thinking 토큰 0. 입력 실행당 약 144.1k 토큰.
- 작품명 해석: B1·B2 모두 13/13 `applied`(캐시 해석기), 실제 해석 호출 0. A는 해석기 없음.
- 서버 상태: 실행 뒤 `server_state_final.json` = `server_state_baseline.json`.
- 지연은 **해석 캐시를 쓴 값**이다. 실시간 해석 대기가 들어간 서비스 지연이 아니다.

## 6. 결과 — 부분집합 13건(분모 13, 실패 0이라 보조 "네 실행 모두 applied" 값도 같다)

| | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Hit@1 | 3 | **6** | **6** | 3 |
| Hit@5 | 5 | **9** | **9** | 5 |
| Hit@10 | 5 | **9** | **9** | 5 |
| MRR@10 | 0.2949 | **0.5449** | **0.5449** | 0.2949 |
| 후보 회수@30(리랭킹 전, 참고) | 7 | 10 | 10 | 7 |

평가기 산출 `search_eval_dev_top10.csv`에서 다시 센 값과 같다. **이 숫자는 이번 변경이 입력을 바꾼 13질의만의 값이다.** dev 176 지표가 아니고, 입력이 같은 163건의 과거 결과와 합쳐 dev 전체 값을 만들지 않았다.

### 6-1. 질의별(최종 순위, 30 밖은 ">30") — `analysis/per_query.csv`

| 질의 | 표시 | 후보 순위 A→B | A1 | B1 | B2 | A2 | 분류 |
|---|---|---|---:|---:|---:|---:|---|
| n033 | 새 후보 | 없음→13 | >30 | **3** | **3** | >30 | 안정 이득 |
| n043 | 새 후보 | 없음→25 | >30 | **4** | **4** | >30 | 안정 이득 |
| n096 | 새 후보 · 작품명=곡 제목 | 없음→3 | >30 | **1** | **1** | >30 | 안정 이득 |
| c707 | 기존 후보 안 실패 | 17→18 | 16 | **1** | **1** | 16 | 안정 이득 |
| n031 | name_in_query | 2→1 | 3 | 1 | 1 | 3 | 같음(10위 안, 순위 오름) |
| m103 · n038 · t037 · t046 | — | 1→1 · 4→1 · 3→2 · 12→12 | 1 · 1 · 2 · 1 | 같음 | 같음 | 같음 | 같음(10위 안) |
| c705 | — | 28→22 | 30 | 29 | 29 | 30 | 같음(10위 밖) |
| c602 · c607 · c701 | — | 없음→없음 | >30 | >30 | >30 | >30 | 같음(후보 밖) |

- **새 후보 3건(n033·n043·n096)은 두 반복 모두 10위 안**(3·4·1위)에 들었다.
- **기존 후보 안 실패의 회복: c707** 1건(16→1위, 두 반복). 정답의 후보 순위는 17→18로 거의 그대로다. 정답은 보조 목록 1위였지만 보조 가산은 순위를 올리지 못했다. 바뀐 것은 리랭커가 본 후보 집합·순서(들어온 곡 1)다. 왜 리랭커 판단이 바뀌었는지는 이 자료로 단정할 수 없다.
- **기존 성공 손실: 0.** A에서 10위 안이던 5건(m103·n031·n038·t037·t046)이 B에서도 모두 10위 안이다. n031은 3→1위로 올랐다.
- 순위가 내려간 질의: 0. c705는 30→29위(10위 밖 그대로).
- c602·c607·c701은 B에서도 정답이 후보 30 밖이다(v71과 같음). 리랭킹으로 바뀔 수 없다.

### 6-2. 반복 변동과 A/B 차이(직접 측정)

| 비교 | Hit@10 뒤집힘 | 최종 10위 목록 다름 | 응답 원문 sha 다름 |
|---|---:|---:|---:|
| A1 ↔ A2(같은 입력) | **0** | 0/13 | 0/13 |
| B1 ↔ B2(같은 입력) | **0** | 0/13 | 0/13 |
| A1 ↔ B1 | 4(c707·n033·n043·n096) | 13/13 | 13/13 |
| A2 ↔ B2 | 4(같은 4건) | 13/13 | 13/13 |

- 이번 실행에서 같은 입력의 리랭커 응답은 13/13 바이트 단위로 같았다(약 4~5분 간격 두 번). 실패가 0이라 "요청 실패 때문인 뒤집힘"도 0이다.
- 이득 4건은 **두 반복에서 공통**이다. 한 반복에서만 나타난 이득·손실은 0이다.
- 이것은 이번 7분 동안의 관측이다. 다른 시각·날짜·모델 갱신 뒤의 변동, 입력이 같은 163건의 변동은 재지 않았다. 과거 실행(v66~v69)에서는 같은 설정에서도 결과가 달랐던 적이 있다.

## 7. §4 판단 원칙 적용

| 기준 | 결과 | 충족 |
|---|---|---|
| 1. 안정 이득 − 안정 손실 ≥ 1 | 4 − 0 = 4 | 예 |
| 2. 기존 성공(A 두 번 모두 10위 안 5건)의 안정 손실 0 | 0 | 예 |
| 3. n096에만 의존하지 않음 | n096을 빼도 안정 이득 3(n033·n043·c707) | 예 |
| 4. 같은 구성 반복 뒤집힘 < 안정 이득 | 0 < 4 | 예 |
| 5. 실패가 판정을 바꾸지 않음 | 실패 영향 질의 0 | 예 |
| 6. 사람 라벨 | **0/22 — 해석 정확도 미측정** | 명시만 |

**결론: 부분집합 진단 기준은 모두 충족했다 → 전체 dev A/B의 계획·예산을 작성한다(`FULL_AB_PLAN.md`). 실행하지 않았다.** 다만 해석 정확도는 미측정이다. 이득 4건이 맞는 작품을 해석한 덕인지는 사람 라벨이 와야 말할 수 있다. 새 후보가 10위 안에 든 것은 그 작품의 언급 곡에 정답이 있었다는 뜻일 뿐이다.

## 8. 해석의 범위

- 이번 결과는 **이번 개발 변경으로 입력이 바뀐 13질의의 부분집합 진단**이다. 그 13건은 개발 자료(dev)에서 이 변경이 고른 질의라 독립 평가 자료가 아니다.
- 부분집합 Hit@10 5→9를 서비스 전체 성능이나 dev 176 개선률로 일반화하지 않는다.
- 입력이 같은 163건은 이번에 부르지 않았다. 따라서 dev 176 실측 A/B는 아직 없다.
- 지연은 해석 캐시를 쓴 평가기 경로 값이다. 실시간 해석이 들어간 서비스 지연, 서버 HTTP 경로는 검증하지 않았다.
- p값으로 판단하지 않았다(13건 · 반복 2회).

## 9. 증명하지 못한 것 · 제한

- **해석 정확도**: 사람 라벨 0/22. 잘못된 작품 연결의 피해·빈도는 모른다(v70 오답 모의에서는 c705 손실 1).
- **입력이 같은 163건의 변동**: 재지 않았다(승인 범위 밖). 같은 입력 대조군은 같은 구성 반복 2쌍(각 13건)뿐이다.
- **c707 회복의 원인**: 리랭커 입력 집합·순서 변화와 함께 나타났지만 기제를 확인하지 않았다. 순서·문맥에 민감한 결과일 수 있다.
- **시각·날짜 간 변동**: 이번 7분 동안 같은 입력 응답이 같았을 뿐이다.
- **서비스 경로**: 실시간 해석(약 1초) 대기, HTTP 경로, 해석 실패·시간 초과의 실제 빈도 미측정.
- **서버 기동 직후 첫 판정 통과 요청의 해석 시간 초과**(§2-4): 평가기에서 재현했지만 서비스 경로에서는 재지 않았다.
- **비용**: 장부의 `usageMetadata` 정산값이다. Vertex 청구서와 대조하지 않았다.
- **새 독립 자료**: 수집 0건.

## 10. 후속(이번에 바꾸지 않음)

- 전체 dev A/B: `FULL_AB_PLAN.md`(계획·예산만, 승인 필요).
- 사람 라벨: `labels/LABEL_STATUS.md` — PM이 라벨러·보관자를 정한다.
- 서비스 기동 직후 첫 판정 통과 요청의 해석 시간 초과 — 서비스 예열 또는 해석 마감 기준을 따로 검토(제품 변경이라 별도 작업).
- v71 문서의 "입력 변경 16 / 같은 입력 160" 표현 정정(이 문서 §2-3, NEXT_WORK에 반영).
- 가중·판정·프롬프트 변경(c602·c607·c701 후보 밖, c603 보류)은 새 자료에서만 본다.

## 11. 새 독립 자료 상태

| 항목 | 상태 | 사람이 할 일 |
|---|---|---|
| 실제 기억 기반 질의(`general`·`diag_work`) | **0건** | 작성자 모집(PM) |
| 별도 정답 확인 · 작품 라벨 확인 | 0건 | 두 번째 사람 확인 |
| dev 22 작품 라벨 | 0/22(대기표 준비됨) | 라벨러 지정(PM) |
| 보관자 · 개발용/봉인 분할 | 미지정 · 실행 0 | PM이 개발에 참여하지 않는 1인 지정 |

dev 개선을 새 자료의 성능 개선으로 쓰지 않는다. 채택 판단은 새 `general` 봉인 세트에서만 한다.

## 12. 산출물과 Git 제외

| 경로 | 내용 | Git |
|---|---|---|
| `RUN_INFO.md` · `FULL_AB_PLAN.md` · `PR_DRAFT.md` · `labels/LABEL_STATUS.md` | 문서 | 추적 |
| `run_eval.py` · `v72_eval_runtime.py` · `v72_budget.py` · `run_live.sh` · `compare_inputs.py` · `qdrant_reference.py` · `server_state.py` · `with_env.py` · `freeze_subset.py` · `check_subset_vs_full.py` · `analyze_v72.py` · `budget_summary.py` · `selftest_budget.py` · `selftest_runner.sh` | 실행기·검사·집계 | 추적 |
| `subset/rerank_input_changed.json` | 고정 부분집합(질의 id 13, sha16 `7004bca4e3f7deb1`) | 추적 |
| `analysis/precheck_*.json` · `analysis/live_summary.json` · `analysis/per_query.csv` · `budget_summary.json` · `labels/review_after_labels.csv` | 사전 검증·결과 요약·질의별 표·장부 요약·라벨 뒤 대조표(질의 원문·작품명·모델 원출력 없음) | 추적 |
| `live/*/run_conditions.json` · `live/*/*_dev/search_eval_dev_{runinfo.json,summary.csv,top10.csv}` · `live_checks/` · `server_state_{baseline,final}.json` | 실행 조건·지표·top-10(곡 id)·입력 대조 | 추적 |
| `../results_v71_work_name_resolution_live/{RUN_INFO,FOLLOWUP_AB_PLAN,PR_DRAFT}.md` | 16→13 정정 메모만 추가(본문 숫자·결과 파일은 그대로) | 추적(수정) |
| `offline/**` 평가기 요약·top-10 | 가짜 리랭커 순서라 품질이 아님 | 제외 |
| `resolution_cache/` · `**/v72_*.jsonl` · `**/search_eval_*_{explain.jsonl,detail.csv,lyric_boosts.csv}` · `labels_private/` · `logs/` · `budget/` | 해석 캐시 사본(모델 원출력)·원기록·질의 원문·작품명·로그·장부 원본 | **제외** |
| `artifacts/qdrant_server_v72/` · `artifacts/src_main_5c2d43e/` · `artifacts/work_names/` | Qdrant 사본·코드 사본·언급 저장소 | 제외(`artifacts/*`) |

## 13. 다시 돌리기

```bash
W=/private/tmp/vague-finder-v72-work-name-rerank; R=experiments/reranking/results_v72_work_name_rerank_diagnosis; V71=/private/tmp/vague-finder-v71-work-name-live
cp -R $V71/artifacts/qdrant_server_v71 $W/artifacts/qdrant_server_v72 && cp -R $V71/artifacts/work_names $W/artifacts/work_names
git archive 5c2d43e src | tar -x -C $W/artifacts/src_main_5c2d43e
mkdir -p $R/resolution_cache && cp $V71/experiments/reranking/results_v71_work_name_resolution_live/resolution_cache/rep1.json $R/resolution_cache/v71_rep1.json
docker run -d --name vague-finder-qdrant-v72 -p 127.0.0.1:6380:6333 -p 127.0.0.1:6381:6334 -v $W/artifacts/qdrant_server_v72/storage:/qdrant/storage qdrant/qdrant:v1.19.1
(cd $R && ../../../venv/bin/python server_state.py --wait-green 600 --out server_state_baseline.json)
for c in off work; do V72_CONFIG=$c V72_MODEL=fake venv/bin/python $R/run_eval.py --rep 1 --run-name ${c}_full_fake --out-root $R/offline/full --expect-server-state $R/server_state_baseline.json; done
(cd $R && ../../../venv/bin/python freeze_subset.py --v71 $V71/experiments/reranking/results_v71_work_name_resolution_live --write)
for c in off work; do V72_CONFIG=$c V72_MODEL=fake venv/bin/python $R/run_eval.py --rep 1 --run-name ${c}_subset_fake --out-root $R/offline/subset --expect-server-state $R/server_state_baseline.json --subset-file $R/subset/rerank_input_changed.json --expect-subset-sha16 7004bca4e3f7deb1; done
for c in off work; do venv/bin/python $R/check_subset_vs_full.py --full $R/offline/full/${c}_full_fake --subset $R/offline/subset/${c}_subset_fake; done
(cd $R && ../../../venv/bin/python selftest_budget.py) && zsh $R/selftest_runner.sh
V72_LIVE_APPROVED=1 zsh $R/run_live.sh          # 실제 호출(승인 필요)
(cd $R && ../../../venv/bin/python budget_summary.py && ../../../venv/bin/python analyze_v72.py && ../../../venv/bin/python server_state.py --out server_state_final.json --expect server_state_baseline.json)
docker stop vague-finder-qdrant-v72
```
