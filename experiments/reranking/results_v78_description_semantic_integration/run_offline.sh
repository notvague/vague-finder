#!/usr/bin/env bash
# v78 오프라인 후보 A/B — A1(off) → B1(on) → B2(on) → A2(off), 순차(동시 실행 없음). 가짜 리랭커·외부 연결 차단·모델 호출 0.
# 워크트리 루트에서: bash experiments/reranking/results_v78_description_semantic_integration/run_offline.sh
# PY 기본값 = 메인 체크아웃 venv(git 공용 폴더의 부모). 결과 폴더가 있으면 run_eval이 덮어쓰기를 거부한다.
set -euo pipefail
R=experiments/reranking/results_v78_description_semantic_integration
MAIN=${VF_MAIN_CHECKOUT:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")}
PY=${PY:-$MAIN/venv/bin/python}
run() {
  local sem=$1 rep=$2 code=0
  echo "start sem_${sem} rep${rep} $(date +%H:%M:%S)"
  env -u CLAUDE_EFFORT V78_SEM="$sem" "$PY" "$R/run_eval.py" --rep "$rep" --out-root "$R/runs" \
    --expect-server-state "$R/server_state_baseline.json" --expect-work-status "$R/inputs/expected_work_status.json" \
    > "$R/logs/sem_${sem}_rep${rep}.log" 2>&1 || code=$?
  echo "done sem_${sem} rep${rep} exit=${code} $(date +%H:%M:%S)"
  test -f "$R/runs/sem_${sem}_server_rep${rep}/DONE"
}
mkdir -p "$R/logs" "$R/runs"
run off 1
run on 1
run on 2
run off 2
