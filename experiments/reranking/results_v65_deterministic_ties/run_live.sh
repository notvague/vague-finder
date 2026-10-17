#!/bin/zsh
# v65 실제 Vertex 재측정 묶음 — **v65에서는 실행하지 않았다.** 사용자가 범위·예산을 정한 뒤에만 쓴다(LIVE_REMEASURE_PLAN.md).
# 평가기 실행(run_offline.py, V65_MODEL=live)을 RUNS 순서대로 순차 실행한다. 예산 장부 하나를 모든 실행이 공유한다.
# 운영 .env에서 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS 두 키만 with_env.py로 넘긴다(값 출력 없음, .env 복사 없음).
#
# 필수 환경: V65_LIVE_APPROVED=1 · DEADLINE_EPOCH(신규 호출 마감 epoch초) · COST_CAP(USD) · ATTEMPT_CAP(SDK HTTP 시도 수)
# 선택: RUNS="before:local:1 before:server:1 ..." (기본은 계획서 기본안 8실행 순서) · QUERY_IDS · SETS(부분 실행)
#   예) V65_LIVE_APPROVED=1 DEADLINE_EPOCH=$(( $(date +%s) + 10800 )) COST_CAP=8 ATTEMPT_CAP=1600 ./run_live.sh
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v65_deterministic_ties
: ${V65_LIVE_APPROVED:?사용자 승인 표시 V65_LIVE_APPROVED=1이 없다}
: ${DEADLINE_EPOCH:?신규 호출 마감 epoch} ${COST_CAP:?비용 상한 USD} ${ATTEMPT_CAP:?SDK 시도 상한}
RUNS=${RUNS:-"before:local:1 before:server:1 after:local:1 after:server:1 after:server:2 after:local:2 before:server:2 before:local:2"}
OUT=${OUT_ROOT:-$R/live}
LEDGER=${LEDGER:-$R/budget/ledger.jsonl}
EXTRA=()
[[ -n "${QUERY_IDS:-}" ]] && EXTRA+=(--query-ids $QUERY_IDS)
[[ -n "${SETS:-}" ]] && EXTRA+=(--sets $SETS)
mkdir -p $R/logs $R/budget
for spec in ${=RUNS}; do
  code=${spec%%:*}; rest=${spec#*:}; mode=${rest%%:*}; rep=${rest##*:}
  echo "start live_${code}_${mode}_rep${rep} $(date +%T)" >> $R/logs/progress_live.txt
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT V65_MODEL=live V65_CODE=$code V65_MODE=$mode \
    venv/bin/python $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    venv/bin/python $R/run_offline.py --rep $rep --out-root $OUT --ledger $LEDGER --deadline-epoch $DEADLINE_EPOCH \
      --cost-cap $COST_CAP --attempt-cap $ATTEMPT_CAP $EXTRA > $R/logs/live_${code}_${mode}_rep${rep}.log 2>&1
  rc=$?
  echo "done live_${code}_${mode}_rep${rep} exit=$rc $(date +%T)" >> $R/logs/progress_live.txt
  (( rc != 0 )) && { echo "FAILED live_${code}_${mode}_rep${rep} exit=$rc — 다음 실행 중단" >&2; exit $rc; }
done
echo "LIVE DONE"
