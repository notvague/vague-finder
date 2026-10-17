#!/bin/zsh
# v65 평가기 실행 묶음 — 코드 판 하나(before|after)를 로컬/서버 교대로 2회씩(L1 → S1 → S2 → L2), 순차 실행.
# 하나라도 실패하면 그 종료 코드로 끝난다. 모델 호출 없음(가짜 결정적 응답).
#   ./run_offline.sh before    ·   ./run_offline.sh after
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v65_deterministic_ties
CODE=$1
OUT=${OUT_ROOT:-$R/offline}
mkdir -p $R/logs
for spec in local:1 server:1 server:2 local:2; do
  mode=${spec%%:*}; rep=${spec##*:}
  echo "start ${CODE}_${mode}_rep${rep} $(date +%T)" >> $R/logs/progress_offline.txt
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT V65_CODE=$CODE V65_MODE=$mode \
    venv/bin/python $R/run_offline.py --rep $rep --out-root $OUT > $R/logs/offline_${CODE}_${mode}_rep${rep}.log 2>&1
  rc=$?
  echo "done ${CODE}_${mode}_rep${rep} exit=$rc $(date +%T)" >> $R/logs/progress_offline.txt
  (( rc != 0 )) && { echo "FAILED ${CODE}_${mode}_rep${rep} exit=$rc" >&2; exit $rc; }
done
echo "OFFLINE ${CODE} DONE"
