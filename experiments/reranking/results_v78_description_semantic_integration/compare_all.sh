#!/usr/bin/env bash
# v78 입력 대조 — 반복 결정성 · OFF 재현(v75 A 실제 실행 2회 · v76 A1) · A↔B. 모델·DB 호출 없음.
# 워크트리 루트에서: bash experiments/reranking/results_v78_description_semantic_integration/compare_all.sh
set -euo pipefail
R=experiments/reranking/results_v78_description_semantic_integration
MAIN=${VF_MAIN_CHECKOUT:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")}
PY=${PY:-$MAIN/venv/bin/python}
V75=${V75:-/private/tmp/vague-finder-v75-work-audio-live-ab/experiments/reranking/results_v75_work_audio_live_ab}
V76=${V76:-/private/tmp/vague-finder-v76-evidence-recall/experiments/reranking/results_v76_description_evidence_recall}
mkdir -p "$R/compare"
c() { "$PY" "$R/compare_inputs.py" --a "$1" --b "$2" --out "$R/compare/$3.json" --overwrite ${4:-}; }
# 반복 결정성(같아야 한다)
c "$R/runs/sem_off_server_rep1" "$R/runs/sem_off_server_rep2" A1_vs_A2 --require-identical
c "$R/runs/sem_on_server_rep1" "$R/runs/sem_on_server_rep2" B1_vs_B2 --require-identical
# OFF 재현(같아야 한다) — v75 A 실제 실행 두 번(입력·첫 호출 프롬프트) · v76 A1(가짜 최종 순서까지)
c "$V75/live/live_work_on_server_rep1" "$R/runs/sem_off_server_rep1" v75liveA1_vs_A1 --require-identical
c "$V75/live/live_work_on_server_rep2" "$R/runs/sem_off_server_rep1" v75liveA2_vs_A1 --require-identical
c "$V76/runs/desc_off_server_rep1" "$R/runs/sem_off_server_rep1" v76A1_vs_A1 --require-identical
# A ↔ B(차이를 센다)
c "$R/runs/sem_off_server_rep1" "$R/runs/sem_on_server_rep1" A1_vs_B1
c "$R/runs/sem_off_server_rep2" "$R/runs/sem_on_server_rep2" A2_vs_B2
