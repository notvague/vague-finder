#!/bin/zsh
# v68 실제 Vertex 소리 경로 A/B 묶음 — 서버 모드만, dev 176, 순서 A1(on) → B1(skip) → B2 → A2(순차, 동시 실행 없음).
# 실행마다: (1) 직전 오프라인 지문(가짜 응답, $0) → 안정성 기준(같은 코드 판의 offline/stability rep1)과 입력 동일 확인
#           (2) 실제 실행 (3) 직후 오프라인 지문 → 직전 지문과 동일 확인 (4) 실제 실행이 보낸 입력 = 직전 지문 확인
# 어느 단계든 실패·입력 차이·서버 상태 변화가 있으면 그 코드로 즉시 멈춘다(다음 모델 호출 없음). 이미 쓴 결과·장부는 보존한다.
# 한도는 이 파일에 고정한다(사용자 지정: 비용 $4 · SDK HTTP 시도 800회 · 첫 시도부터 90분). 환경변수로 늘릴 수 없다.
# 운영 .env에서 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS 두 키만 with_env.py로 넘긴다(값 출력·복사 없음).
#   필수: V68_LIVE_APPROVED=1
#   자체 검사(selftest_runner.sh)만: PY · LOG_DIR · OUT_LIVE · OUT_OFF · CHECK_DIR · LEDGER · RUNS 를 바꾼다
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v68_audio_skip_live
PY=${PY:-venv/bin/python}
LOGS=${LOG_DIR:-$R/logs}
OUT_LIVE=${OUT_LIVE:-$R/live}
OUT_OFF=${OUT_OFF:-$R/offline/live_fingerprints}
CHECKS=${CHECK_DIR:-$R/live_checks}
LEDGER=${LEDGER:-$R/budget/ledger.jsonl}
STATE=$R/server_state_baseline.json
REF=$R/offline/stability
RUNS=${RUNS:-"on:1 skip:1 skip:2 on:2"}
COST_CAP=4
ATTEMPT_CAP=800
WINDOW=5400
: ${V68_LIVE_APPROVED:?사용자 승인 표시 V68_LIVE_APPROVED=1이 없다}
[[ "$V68_LIVE_APPROVED" == "1" ]] || { echo "V68_LIVE_APPROVED는 정확히 1이어야 한다" >&2; exit 2; }
mkdir -p $LOGS $CHECKS ${LEDGER:h}
P=$LOGS/progress_live.txt
step() { echo "$1 $(date +%T)" >> $P; }
stop() { echo "STOP $1 exit=$2 $(date +%T)" >> $P; echo "STOP $1 (exit $2) — 다음 모델 호출 중단" >&2; exit $2; }
offline() {  # offline <code> <rep> <name>
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT V68_AUDIO=$1 V68_MODEL=fake \
    $PY $R/run_eval.py --rep $2 --run-name $3 --out-root $OUT_OFF --expect-server-state $STATE > $LOGS/$3.log 2>&1
}
for spec in ${=RUNS}; do
  code=${spec%%:*}; rep=${spec##*:}
  live=live_${code}_server_rep${rep}; pre=pre_${code}_rep${rep}; post=post_${code}_rep${rep}
  step "start $pre"
  offline $code $rep $pre; rc=$?; (( rc == 0 )) || stop $pre $rc
  $PY $R/compare_inputs.py --a $REF/${code}_server_rep3 --b $OUT_OFF/$pre --require-identical \
    --out $CHECKS/${pre}__vs_stability.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${pre}_vs_stability" $rc
  step "start $live"
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT V68_AUDIO=$code V68_MODEL=live V68_LIVE_APPROVED=1 \
    $PY $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    $PY $R/run_eval.py --rep $rep --out-root $OUT_LIVE --expect-server-state $STATE --ledger $LEDGER \
      --deadline-window-seconds $WINDOW --cost-cap $COST_CAP --attempt-cap $ATTEMPT_CAP > $LOGS/$live.log 2>&1
  rc=$?; step "done $live exit=$rc"
  (( rc == 0 )) || stop $live $rc
  step "start $post"
  offline $code $rep $post; rc=$?; (( rc == 0 )) || stop $post $rc
  $PY $R/compare_inputs.py --a $OUT_OFF/$pre --b $OUT_OFF/$post --require-identical \
    --out $CHECKS/${post}__vs_pre.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${post}_vs_pre" $rc
  $PY $R/compare_inputs.py --a $OUT_OFF/$pre --b $OUT_LIVE/$live --require-identical \
    --out $CHECKS/${live}__vs_pre.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${live}_vs_pre" $rc
  step "verified $live"
done
step "LIVE DONE"
echo "LIVE DONE"
