#!/usr/bin/env bash
# v76 입력 대조 — 반복 결정성 · OFF 재현(v75 A 실제 실행 2회 · v75 오프라인 · v74 work_on) · A↔B. 모델·DB 호출 없음.
# 워크트리 루트에서: bash experiments/reranking/results_v76_description_evidence_recall/compare_all.sh
set -euo pipefail
R=experiments/reranking/results_v76_description_evidence_recall
PY=${PY:-venv/bin/python}
V75=${V75:-/private/tmp/vague-finder-v75-work-audio-live-ab/experiments/reranking/results_v75_work_audio_live_ab}
V74=${V74:-/private/tmp/vague-finder-v74-candidate-recall/experiments/reranking/results_v74_candidate_recall_work_audio}
mkdir -p "$R/compare"
c() { "$PY" "$R/compare_inputs.py" --a "$1" --b "$2" --out "$R/compare/$3.json" --overwrite ${4:-}; }
# 반복 결정성(같아야 한다)
c "$R/runs/desc_off_server_rep1" "$R/runs/desc_off_server_rep2" A1_vs_A2 --require-identical
c "$R/runs/desc_on_server_rep1" "$R/runs/desc_on_server_rep2" B1_vs_B2 --require-identical
# OFF 재현(같아야 한다) — v75 A 실제 실행 두 번(입력·첫 호출 프롬프트) · v75 오프라인 기준 · v74 work_on 오프라인(최종 가짜 순서까지)
c "$V75/live/live_work_on_server_rep1" "$R/runs/desc_off_server_rep1" v75liveA1_vs_A1 --require-identical
c "$V75/live/live_work_on_server_rep2" "$R/runs/desc_off_server_rep1" v75liveA2_vs_A1 --require-identical
c "$V75/offline/full/work_on_full_warm" "$R/runs/desc_off_server_rep1" v75offline_work_on_vs_A1 --require-identical
c "$V74/runs/work_on_server_rep1" "$R/runs/desc_off_server_rep1" v74work_on_vs_A1 --require-identical
# A ↔ B(차이를 센다)
c "$R/runs/desc_off_server_rep1" "$R/runs/desc_on_server_rep1" A1_vs_B1
c "$R/runs/desc_off_server_rep2" "$R/runs/desc_on_server_rep2" A2_vs_B2
