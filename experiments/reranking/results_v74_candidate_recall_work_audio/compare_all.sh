#!/usr/bin/env bash
# v74 입력 대조 — 같은 구성 반복 2회(결정성, 같아야 한다) · 기준(work_on 반복 1) ↔ 다른 구성 반복 1. 모델·DB 호출 없음.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-venv/bin/python}"
mkdir -p "$R/compare"
for c in work_on off_on off_skip work_skip; do
  "$PY" "$R/compare_inputs.py" --a "$R/runs/${c}_server_rep1" --b "$R/runs/${c}_server_rep2" \
    --out "$R/compare/${c}_rep1_vs_rep2.json" --require-identical --overwrite | tail -1
done
for c in off_on off_skip work_skip; do
  "$PY" "$R/compare_inputs.py" --a "$R/runs/work_on_server_rep1" --b "$R/runs/${c}_server_rep1" \
    --out "$R/compare/work_on_vs_${c}.json" --overwrite | tail -1
done
# 작품명 효과(소리 OFF 조건): off_skip ↔ work_skip
"$PY" "$R/compare_inputs.py" --a "$R/runs/off_skip_server_rep1" --b "$R/runs/work_skip_server_rep1" \
  --out "$R/compare/off_skip_vs_work_skip.json" --overwrite | tail -1
