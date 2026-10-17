#!/bin/zsh
# v63 실제 Vertex 평가 — 오프라인과 같은 스냅샷·색인·캐시·설정, 모델만 실제 호출(gemini-3.1-flash-lite, Vertex 전용).
# 순서 A1 → B1 → B2 → A2(시간대 편향 완화, 순차). 예산 장부는 네 실행이 공유한다(비용 $10 · SDK 시도 1,200회 · 신규 호출 마감).
# 운영 .env에서 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS 두 키만 with_env.py로 넘긴다(.env 복사 없음, 값 출력 없음).
#   DEADLINE_EPOCH=<epoch초> 신규 호출 마감(필수) · OUT_ROOT=<폴더>(기본 이 폴더/live) · OVERWRITE=1 일 때만 덮어쓰기
#   RUNS="A:1 B:1" 처럼 일부만(예산이 모자랄 때 완결 A/B 쌍을 먼저) — 기본은 네 실행
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v63_hybrid_lyrics_validation
PY=${PY:-venv/bin/python}
LOGS=${LOG_DIR:-$R/logs}   # 자체 검사(selftest_failure.sh)는 다른 폴더를 준다
OUT_ROOT=${OUT_ROOT:-$R/live}
DEADLINE_EPOCH=${DEADLINE_EPOCH:?신규 호출 마감 epoch}
LEDGER=${LEDGER:-$R/budget/ledger.jsonl}
RUNS=${RUNS:-"A:1 B:1 B:2 A:2"}
OVR=()
[[ "${OVERWRITE:-}" == "1" ]] && OVR=(--overwrite)
mkdir -p $LOGS $R/budget
for spec in ${=RUNS}; do
  arm=${spec%%:*}; rep=${spec##*:}
  echo "start live_${arm}_rep${rep} $(date +%T)" >> $LOGS/progress_live.txt
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT $PY $R/with_env.py GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS -- \
    $PY $R/run_v63.py --arm $arm --mode live --rep $rep --out-root $OUT_ROOT \
      --ledger $LEDGER --deadline-epoch $DEADLINE_EPOCH --cost-cap 10 --attempt-cap 1200 $OVR \
    > $LOGS/live_${arm}_rep${rep}.log 2>&1
  rc=$?
  echo "done live_${arm}_rep${rep} exit=$rc $(date +%T)" >> $LOGS/progress_live.txt
  if (( rc != 0 )); then echo "FAILED at live_${arm}_rep${rep} (exit $rc) — 다음 실행·비교 중단" >&2; exit $rc; fi
done
echo "LIVE RUNS DONE"
