#!/usr/bin/env bash
# v74 오프라인 4구성 × 2회(가짜 리랭커 · 외부 모델 호출 0). 워크트리 루트에서: bash <이 폴더>/run_offline.sh [구성:반복 ...]
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-venv/bin/python}"
RUNS="${*:-work:on:1 work:on:2 off:on:1 off:on:2 off:skip:1 off:skip:2 work:skip:1 work:skip:2}"
mkdir -p "$R/logs"
for spec in $RUNS; do
  IFS=: read -r cfg audio rep <<<"$spec"
  name="${cfg}_${audio}_server_rep${rep}"
  if [ -f "$R/runs/$name/DONE" ]; then echo "[skip] $name 완료됨"; continue; fi
  echo "[run] $name"
  V74_CONFIG=$cfg V74_AUDIO=$audio "$PY" "$R/run_eval.py" --rep "$rep" --out-root "$R/runs" \
    --expect-server-state "$R/server_state_baseline.json" --expect-work-status "$R/inputs/expected_work_status.json" \
    > "$R/logs/${name}.log" 2>&1
  tail -1 "$R/logs/${name}.log" | cut -c1-200
done
