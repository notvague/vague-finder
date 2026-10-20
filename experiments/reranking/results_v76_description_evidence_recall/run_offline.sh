#!/usr/bin/env bash
# v76 오프라인 후보 A/B — A1(off) → B1(on) → B2(on) → A2(off), 순차(동시 실행 없음). 가짜 리랭커·외부 연결 차단·모델 호출 0.
# 워크트리 루트에서: bash experiments/reranking/results_v76_description_evidence_recall/run_offline.sh
set -euo pipefail
R=experiments/reranking/results_v76_description_evidence_recall
PY=${PY:-venv/bin/python}
run() {
  local desc=$1 rep=$2
  echo "start desc_${desc} rep${rep} $(date +%H:%M:%S)"
  env -u CLAUDE_EFFORT V76_DESC="$desc" "$PY" "$R/run_eval.py" --rep "$rep" --out-root "$R/runs" \
    --expect-server-state "$R/server_state_baseline.json" --expect-work-status "$R/inputs/expected_work_status.json" \
    > "$R/logs/desc_${desc}_rep${rep}.log" 2>&1
  echo "done desc_${desc} rep${rep} exit=$? $(date +%H:%M:%S)"
  test -f "$R/runs/desc_${desc}_server_rep${rep}/DONE"
}
mkdir -p "$R/logs" "$R/runs"
run off 1
run on 1
run on 2
run off 2
