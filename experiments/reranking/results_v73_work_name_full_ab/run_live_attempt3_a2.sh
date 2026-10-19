#!/bin/zsh
# A2 단독 실행(2026-10-18 22:4x 사용자 결정 "A2만 지금 처음부터"). run_live_attempt3.sh가 A1·B1·B2를 완료·검증한 뒤
# A2 직전 무료 지문(가짜 응답)이 GPU 드라이버 오류(MPS tanh, SIGSEGV, exit 139)로 꺼져 멈췄다. A2는 실제 호출 0회였다.
# 같은 장부·한도·시간 창(첫 시도 21:59 + 14,400초)으로 A2 직전 지문부터 A2 전체(176)를 새로 잰다. 실행 도중을 이어 붙이지 않는다.
# 꺼진 직전 지문은 offline/live_fingerprints/pre_off_rep2_crashed_sigsegv에 보존했다.
# v73 3차(attempt3) 본 실행 — run_live_attempt2.sh 사본. 2026-10-18 사용자 요청: 평가 프로세스에서만 호출 하나 60초 · 질의 하나
# 120초(V73_PROFILE=t60), 재시도 정정 지시 수정 사본(inputs/rerank_retry_correction.patch). 모델·프롬프트·후보·가중치는 그대로.
# 소규모 확인(run_live_smoke_t60.sh → attempt3/smoke/smoke_check.json pass=true)을 통과해야 시작한다.
# 1·2차 결과·장부는 그대로 두고 새 폴더 attempt3/·새 장부로 A1부터 처음부터 잰다(이어 붙이지 않음).
# 한도(사용자 승인 2026-10-18 "확인 $1.5 + 본 실행 $6"): 비용 $6 · SDK HTTP 시도 800회 · 첫 시도부터 240분(14,400초).
# 정지 규칙은 v69 그대로(연속 504 3회 · 10분 안 504 5회 · 인증 첫 오류 · 연속 실패 5회 · 누적 실패 20회).
# (원래 머리말) v73 실제 Vertex 리랭킹 dev 176 전체 A/B 묶음(v72 run_live.sh 사본) — 서버 모드, 순서 A1(off) → B1(work) → B2(work) → A2(off)
# (순차, 동시 실행 없음). 네 실행이 장부 하나·한도 하나·마감 하나를 공유한다.
# v72에서 바꾼 것: 고정 부분집합 대신 dev 176 전체 · 기준 지문 offline/full/<구성>_full_warm · 질의별 기준 해석 상태 파일(지문 고정) ·
#   한도(사용자 승인: 비용 $8 · SDK HTTP 시도 1,000회 · 첫 외부 시도부터 180분).
# 실행마다: (1) 직전 오프라인 지문(가짜 응답, $0) → 기준(offline/full/<구성>_full_warm)과 입력 동일 확인
#           (2) 실제 실행 (3) 직후 오프라인 지문 → 직전 지문과 동일 확인 (4) 실제 실행이 보낸 입력 = 직전 지문 확인
# 어느 단계든 실패·입력 차이·서버 상태 변화·해석 상태 차이가 있으면 그 코드로 즉시 멈춘다(다음 모델 호출 없음). 이미 쓴 결과·장부는 보존한다.
# 멈춘 뒤 새 장부·새 프로세스로 이어 가지 않는다(정지 장부는 다음 실행이 상속해 시작하지 않는다 — v73_budget).
# 한도는 이 파일에 고정한다. 환경변수로 늘릴 수 없다.
# 장애 집중 정지(연속 504 3회 · 10분 안 504 5회)·인증 첫 오류 정지는 v73_budget(v69 규칙 그대로)에 고정돼 있다.
# 운영 .env에서 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS 두 키만 with_env.py로 넘긴다(값 출력·복사 없음).
#   필수: V73_LIVE_APPROVED=1
#   경로·한도·순서는 이 파일에 고정(환경변수로 바꾸지 않음)
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v73_work_name_full_ab
A2R=$R/attempt3
PY=venv/bin/python
LOGS=$A2R/logs
OUT_LIVE=$A2R/live
OUT_OFF=$A2R/offline/live_fingerprints
CHECKS=$A2R/live_checks
LEDGER=$A2R/budget/ledger.jsonl
# 입력 동일성은 기준 지문(offline/full, base 판 가짜 실행)과 176/176 비교한다 — t60 가짜 실행이 그것과 같음을 실행 전에 확인했다.
STATE=$A2R/server_state_baseline_main.json  # 본 실행 직전 다시 뜬 기준(소규모 확인 뒤 컨테이너 재시작 — 시작 시각 외 1차와 같아야 한다)
REF=$R/offline/full
WORK_STATUS=$R/inputs/expected_work_status.json
WORK_STATUS_SHA16=1fb2f754366de693
RUNS="off:2"
COST_CAP=6
ATTEMPT_CAP=800
WINDOW=14400
# 시작 조건(2026-10-18 사용자 결정, RUN_INFO §15): 품질 평가 시작 조건과 지연 보고를 분리한다.
#   소규모 확인 기록(smoke_check.json, 결과는 "기존 기준 불합격 pass=false" 그대로 보존)에서 S1~S5(인증 오류 0 · 창 만료 외 정지 없음 ·
#   실패 2회 이하·연속 없음 · 폴백 0 · 900초 이상)가 모두 참이어야 시작한다. S6(10초 초과 횟수 등 지연)은 지연 보고 항목이며
#   품질 평가 시작을 막는 조건에서는 제외한다. 통과 기록을 새로 쓰거나 고치지 않는다.
SMOKE=$A2R/smoke/smoke_check.json
[[ -e $SMOKE ]] && venv/bin/python -c "
import json,sys
c=json.load(open('$SMOKE'))['criteria']
need=('S1_auth_errors_0','S2_only_window_halt','S3_fails_le_2_no_consecutive','S4_rerank_fallbacks_0','S5_span_ge_900s')
sys.exit(0 if all(c.get(k) is True for k in need) else 1)" \
  || { echo "소규모 확인 기록이 없거나 S1~S5 중 거짓이 있다 — 본 실행을 시작하지 않는다" >&2; exit 3; }
[[ -e $A2R/budget/ledger.jsonl ]] || { echo "3차 본 실행 장부가 없다 — A2 단독 실행은 같은 장부를 이어 쓴다" >&2; exit 2; }
[[ -e $A2R/live/live_off_server_rep2 ]] && { echo "A2 실제 실행 폴더가 이미 있다 — 새로 돌리지 않는다" >&2; exit 2; }
for r in live_off_server_rep1 live_work_server_rep1 live_work_server_rep2; do [[ -e $A2R/live/$r/DONE ]] || { echo "$r 완료 기록이 없다" >&2; exit 2; }; done
: ${V73_LIVE_APPROVED:?사용자 승인 표시 V73_LIVE_APPROVED=1이 없다}
[[ "$V73_LIVE_APPROVED" == "1" ]] || { echo "V73_LIVE_APPROVED는 정확히 1이어야 한다" >&2; exit 2; }
mkdir -p $LOGS $CHECKS ${LEDGER:h}
P=$LOGS/progress_live.txt
step() { echo "$1 $(date +%T)" >> $P; }
stop() { echo "STOP $1 exit=$2 $(date +%T)" >> $P; echo "STOP $1 (exit $2) — 다음 모델 호출 중단" >&2; exit $2; }
WS=(--expect-work-status $WORK_STATUS --expect-work-status-sha16 $WORK_STATUS_SHA16)
offline() {  # offline <config> <rep> <name>
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V73_FAULT_PLAN -u V73_FAIL_INJECT -u V73_PATH_FAIL_INJECT -u V73_WORK_MISS_INJECT V73_PROFILE=t60 V73_CONFIG=$1 V73_MODEL=fake \
    $PY $R/run_eval.py --rep $2 --run-name $3 --out-root $OUT_OFF --expect-server-state $STATE "${WS[@]}" > $LOGS/$3.log 2>&1
}
for spec in ${=RUNS}; do
  cfg=${spec%%:*}; rep=${spec##*:}
  live=live_${cfg}_server_rep${rep}; pre=pre_${cfg}_rep${rep}; post=post_${cfg}_rep${rep}
  step "start $pre"
  offline $cfg $rep $pre; rc=$?; (( rc == 0 )) || stop $pre $rc
  $PY $R/compare_inputs.py --a $REF/${cfg}_full_warm --b $OUT_OFF/$pre --require-identical \
    --out $CHECKS/${pre}__vs_reference.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${pre}_vs_reference" $rc
  step "start $live"
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V73_FAULT_PLAN -u V73_FAIL_INJECT -u V73_PATH_FAIL_INJECT -u V73_WORK_MISS_INJECT V73_PROFILE=t60 V73_CONFIG=$cfg V73_MODEL=live V73_LIVE_APPROVED=1 \
    $PY $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    $PY $R/run_eval.py --rep $rep --out-root $OUT_LIVE --expect-server-state $STATE --ledger $LEDGER "${WS[@]}" \
      --deadline-window-seconds $WINDOW --cost-cap $COST_CAP --attempt-cap $ATTEMPT_CAP > $LOGS/$live.log 2>&1
  rc=$?; step "done $live exit=$rc"
  (( rc == 0 )) || stop $live $rc
  step "start $post"
  offline $cfg $rep $post; rc=$?; (( rc == 0 )) || stop $post $rc
  $PY $R/compare_inputs.py --a $OUT_OFF/$pre --b $OUT_OFF/$post --require-identical \
    --out $CHECKS/${post}__vs_pre.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${post}_vs_pre" $rc
  $PY $R/compare_inputs.py --a $OUT_OFF/$pre --b $OUT_LIVE/$live --require-identical \
    --out $CHECKS/${live}__vs_pre.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${live}_vs_pre" $rc
  step "verified $live"
done
step "LIVE DONE"
echo "LIVE DONE"
