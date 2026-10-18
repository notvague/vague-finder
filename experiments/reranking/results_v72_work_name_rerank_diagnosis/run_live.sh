#!/bin/zsh
# v72 실제 Vertex 리랭킹 부분집합 진단 묶음(v69 run_live.sh 사본) — 서버 모드, 고정 부분집합 13질의, 순서 A1(off) → B1(work) → B2(work) → A2(off)
# (순차, 동시 실행 없음). 네 실행이 장부 하나·한도 하나·마감 하나를 공유한다.
# 실행마다: (1) 직전 오프라인 지문(가짜 응답, $0) → 기준(offline/subset/<구성>_subset_fake = 전체 176 실행과 13/13 같음)과 입력 동일 확인
#           (2) 실제 실행 (3) 직후 오프라인 지문 → 직전 지문과 동일 확인 (4) 실제 실행이 보낸 입력 = 직전 지문 확인
# 어느 단계든 실패·입력 차이·서버 상태 변화가 있으면 그 코드로 즉시 멈춘다(다음 모델 호출 없음). 이미 쓴 결과·장부는 보존한다.
# 한도는 이 파일에 고정한다(사용자 승인: 비용 $1.50 · SDK HTTP 시도 200회 · 첫 외부 시도부터 60분). 환경변수로 늘릴 수 없다.
# 장애 집중 정지(연속 504 3회 · 10분 안 504 5회)·인증 첫 오류 정지는 v72_budget(v69 규칙 그대로)에 고정돼 있다.
# 부분집합 파일 지문도 이 파일에 고정한다(다른 질의 목록으로는 live가 시작하지 않는다).
# 운영 .env에서 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS 두 키만 with_env.py로 넘긴다(값 출력·복사 없음).
#   필수: V72_LIVE_APPROVED=1
#   자체 검사(selftest_runner.sh)만: PY · LOG_DIR · OUT_LIVE · OUT_OFF · CHECK_DIR · LEDGER · RUNS 를 바꾼다
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v72_work_name_rerank_diagnosis
PY=${PY:-venv/bin/python}
LOGS=${LOG_DIR:-$R/logs}
OUT_LIVE=${OUT_LIVE:-$R/live}
OUT_OFF=${OUT_OFF:-$R/offline/live_fingerprints}
CHECKS=${CHECK_DIR:-$R/live_checks}
LEDGER=${LEDGER:-$R/budget/ledger.jsonl}
STATE=$R/server_state_baseline.json
REF=$R/offline/subset
SUBSET=$R/subset/rerank_input_changed.json
SUBSET_SHA16=7004bca4e3f7deb1
RUNS=${RUNS:-"off:1 work:1 work:2 off:2"}
COST_CAP=1.5
ATTEMPT_CAP=200
WINDOW=3600
: ${V72_LIVE_APPROVED:?사용자 승인 표시 V72_LIVE_APPROVED=1이 없다}
[[ "$V72_LIVE_APPROVED" == "1" ]] || { echo "V72_LIVE_APPROVED는 정확히 1이어야 한다" >&2; exit 2; }
mkdir -p $LOGS $CHECKS ${LEDGER:h}
P=$LOGS/progress_live.txt
step() { echo "$1 $(date +%T)" >> $P; }
stop() { echo "STOP $1 exit=$2 $(date +%T)" >> $P; echo "STOP $1 (exit $2) — 다음 모델 호출 중단" >&2; exit $2; }
SUB=(--subset-file $SUBSET --expect-subset-sha16 $SUBSET_SHA16)
offline() {  # offline <config> <rep> <name>
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V72_FAULT_PLAN -u V72_FAIL_INJECT -u V72_PATH_FAIL_INJECT V72_CONFIG=$1 V72_MODEL=fake \
    $PY $R/run_eval.py --rep $2 --run-name $3 --out-root $OUT_OFF --expect-server-state $STATE "${SUB[@]}" > $LOGS/$3.log 2>&1
}
for spec in ${=RUNS}; do
  cfg=${spec%%:*}; rep=${spec##*:}
  live=live_${cfg}_server_rep${rep}; pre=pre_${cfg}_rep${rep}; post=post_${cfg}_rep${rep}
  step "start $pre"
  offline $cfg $rep $pre; rc=$?; (( rc == 0 )) || stop $pre $rc
  $PY $R/compare_inputs.py --a $REF/${cfg}_subset_fake --b $OUT_OFF/$pre --require-identical \
    --out $CHECKS/${pre}__vs_reference.json >> $LOGS/checks.log 2>&1; rc=$?; (( rc == 0 )) || stop "${pre}_vs_reference" $rc
  step "start $live"
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V72_FAULT_PLAN -u V72_FAIL_INJECT -u V72_PATH_FAIL_INJECT V72_CONFIG=$cfg V72_MODEL=live V72_LIVE_APPROVED=1 \
    $PY $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    $PY $R/run_eval.py --rep $rep --out-root $OUT_LIVE --expect-server-state $STATE --ledger $LEDGER "${SUB[@]}" \
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
