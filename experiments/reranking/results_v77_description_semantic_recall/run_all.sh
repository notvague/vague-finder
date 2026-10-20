#!/usr/bin/env bash
# v77 재현 절차(외부 호출 0 · 로컬 KoE5만). 워크트리 루트에서 실행한다.
#   bash experiments/reranking/results_v77_description_semantic_recall/run_all.sh
# 색인·실행 폴더가 이미 있으면 각 단계가 덮어쓰지 않고 멈춘다.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
R="$ROOT/experiments/reranking/results_v77_description_semantic_recall"
PY="${PY:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/venv/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$ROOT"
mkdir -p "$R/logs"
"$PY" -m pytest -q "$R/test_desc_semantic.py" -p no:cacheprovider
"$PY" "$R/build_index.py" > "$R/logs/build_index.log" 2>&1
"$PY" "$R/retrieve.py" --run R1 > "$R/logs/retrieve_R1.log" 2>&1
"$PY" "$R/retrieve.py" --run R2 > "$R/logs/retrieve_R2.log" 2>&1
"$PY" "$R/score.py" --run R1 --compare R2 > "$R/logs/score.log" 2>&1
"$PY" "$R/analyze.py" > "$R/logs/analyze.log" 2>&1
