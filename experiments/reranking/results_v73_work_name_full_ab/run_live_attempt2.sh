#!/bin/zsh
# v73 재시도(attempt2, 2026-10-18 19:49 사용자 요청·한도 확인 "남은 승인액 $6.27") — run_live.sh 사본. 1차(B1 장애 정지) 결과·장부는
# 그대로 두고 새 폴더 attempt2/·새 장부로 A1부터 처음부터 잰다(1차와 이어 붙이지 않음). 한도는 최초 승인 $8 · 1,000회에서 1차
# 사용분($1.7301 · 186회)을 뺀 $6.27 · 814회. 시간 한도만 새 첫 시도부터 180분. 정지 규칙은 그대로.
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
A2R=$R/attempt2
PY=venv/bin/python
LOGS=$A2R/logs
OUT_LIVE=$A2R/live
OUT_OFF=$A2R/offline/live_fingerprints
CHECKS=$A2R/live_checks
LEDGER=$A2R/budget/ledger.jsonl
# 1차 정지 뒤 컨테이너를 멈췄다 다시 켰다 — 시작 시각만 다르고 컬렉션·점 수·세그먼트·별칭·버전은 1차 기준과 같다(19:51 확인).
# 서버 기준만 새로 뜨고, 입력 동일성은 그대로 기준 지문(offline/full)과 176/176 비교한다.
STATE=$A2R/server_state_baseline.json
REF=$R/offline/full
WORK_STATUS=$R/inputs/expected_work_status.json
WORK_STATUS_SHA16=1fb2f754366de693
RUNS="off:1 work:1 work:2 off:2"
COST_CAP=6.27
ATTEMPT_CAP=814
WINDOW=10800
: ${V73_LIVE_APPROVED:?사용자 승인 표시 V73_LIVE_APPROVED=1이 없다}
[[ "$V73_LIVE_APPROVED" == "1" ]] || { echo "V73_LIVE_APPROVED는 정확히 1이어야 한다" >&2; exit 2; }
mkdir -p $LOGS $CHECKS ${LEDGER:h}
P=$LOGS/progress_live.txt
step() { echo "$1 $(date +%T)" >> $P; }
stop() { echo "STOP $1 exit=$2 $(date +%T)" >> $P; echo "STOP $1 (exit $2) — 다음 모델 호출 중단" >&2; exit $2; }
WS=(--expect-work-status $WORK_STATUS --expect-work-status-sha16 $WORK_STATUS_SHA16)
offline() {  # offline <config> <rep> <name>
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V73_FAULT_PLAN -u V73_FAIL_INJECT -u V73_PATH_FAIL_INJECT -u V73_WORK_MISS_INJECT V73_CONFIG=$1 V73_MODEL=fake \
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
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V73_FAULT_PLAN -u V73_FAIL_INJECT -u V73_PATH_FAIL_INJECT -u V73_WORK_MISS_INJECT V73_CONFIG=$cfg V73_MODEL=live V73_LIVE_APPROVED=1 \
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
