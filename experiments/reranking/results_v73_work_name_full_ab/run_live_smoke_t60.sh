#!/bin/zsh
# v73 3차(attempt3) 소규모 확인 — 실제 평가 경로·실제 프롬프트로 약 20분 호출해 성공률·지연만 본다(정답률은 판단에 쓰지 않는다).
# 2026-10-18 사용자 요청: 평가 프로세스에서만 호출 하나 60초 · 질의 하나 120초(V73_PROFILE=t60), 재시도 정정 지시 수정 사본.
# 1·2차는 시작 7~12분 뒤 지연·504가 몰렸으므로 짧은 스모크로는 판단하지 않는다 → 창 20분(1,200초)을 끝까지 쓴다.
# 순서: A 직전 지문(가짜, $0) → 기준 지문과 입력 동일 확인 → 실제 A dev 176 → 실제 B dev 176(창이 끝나면 신규 호출 마감으로 멈춘다).
# 본 측정이 아니므로 A·B 사이 지문 단계는 두지 않는다(시간 창 안에서 호출을 끊지 않으려고). 결과 지표는 내지 않는다.
# 한도(사용자 승인 2026-10-18): 비용 $1.5 · SDK HTTP 시도 300회 · 첫 시도부터 1,200초. 정지 규칙은 v69 그대로.
# 판정은 smoke_check_t60.py(실행 전에 고정한 기준). 통과해야 run_live_attempt3.sh가 시작된다.
#   필수: V73_LIVE_APPROVED=1
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v73_work_name_full_ab
S=$R/attempt3/smoke
PY=venv/bin/python
LOGS=$S/logs
LEDGER=$S/budget/ledger.jsonl
STATE=$R/attempt3/server_state_baseline.json
REF=$R/offline/full
WORK_STATUS=$R/inputs/expected_work_status.json
WORK_STATUS_SHA16=1fb2f754366de693
COST_CAP=1.5
ATTEMPT_CAP=300
WINDOW=1200
: ${V73_LIVE_APPROVED:?사용자 승인 표시 V73_LIVE_APPROVED=1이 없다}
[[ "$V73_LIVE_APPROVED" == "1" ]] || { echo "V73_LIVE_APPROVED는 정확히 1이어야 한다" >&2; exit 2; }
[[ -e $LEDGER ]] && { echo "소규모 확인 장부가 이미 있다 — 새로 돌리지 않는다" >&2; exit 2; }
mkdir -p $LOGS $S/live_checks ${LEDGER:h}
P=$LOGS/progress_live.txt
step() { echo "$1 $(date +%T)" >> $P; }
stop() { echo "STOP $1 exit=$2 $(date +%T)" >> $P; echo "STOP $1 (exit $2)" >&2; exit $2; }
WS=(--expect-work-status $WORK_STATUS --expect-work-status-sha16 $WORK_STATUS_SHA16)
CLEAN=(env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V73_FAULT_PLAN -u V73_FAIL_INJECT -u V73_PATH_FAIL_INJECT -u V73_WORK_MISS_INJECT V73_PROFILE=t60)
step "start pre_off_smoke"
"${CLEAN[@]}" V73_CONFIG=off V73_MODEL=fake $PY $R/run_eval.py --rep 1 --run-name pre_off_smoke --out-root $S/offline \
  --expect-server-state $STATE "${WS[@]}" > $LOGS/pre_off_smoke.log 2>&1; rc=$?; (( rc == 0 )) || stop pre_off_smoke $rc
$PY $R/compare_inputs.py --a $REF/off_full_warm --b $S/offline/pre_off_smoke --require-identical \
  --out $S/live_checks/pre_off_smoke__vs_reference.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop pre_off_smoke_vs_reference $rc
for cfg in off work; do
  step "start smoke_live_$cfg"
  "${CLEAN[@]}" V73_CONFIG=$cfg V73_MODEL=live V73_LIVE_APPROVED=1 \
    $PY $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    $PY $R/run_eval.py --rep 1 --out-root $S/live --expect-server-state $STATE --ledger $LEDGER "${WS[@]}" \
      --deadline-window-seconds $WINDOW --cost-cap $COST_CAP --attempt-cap $ATTEMPT_CAP > $LOGS/smoke_live_$cfg.log 2>&1
  rc=$?; step "done smoke_live_$cfg exit=$rc"
  (( rc == 0 )) || break
done
$PY $R/smoke_check_t60.py --smoke-dir $S > $S/smoke_check.txt 2>&1; rc=$?
step "smoke_check exit=$rc"
cat $S/smoke_check.txt
exit $rc
