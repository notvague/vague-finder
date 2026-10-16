#!/bin/zsh
# v62 — 수정된 구현으로 dev 176건(v06 57 · v09 59 · v10 60)의 후보 30개를 다시 뽑는다(v61 `run_candidates.py --arm A`, 리랭킹 없음).
# 출력은 CAND_ROOT(기본: 이 폴더의 `cand_fixed/`)다 — v61 기준선 `cand_A·cand_B·cand_C`는 읽기만 하고 덮어쓰지 않는다.
# 로컬 Qdrant 복사본은 한 프로세스만 열 수 있어 순차다. 모델·Mongo·운영 Qdrant 호출은 v61_guard가 막는다.
# **한 세트라도 실패하면(산출물이 있어 덮어쓰기 거부로 멈춘 경우 포함) 즉시 그 종료 코드로 끝난다** — 이어지는 비교가 남아 있는 옛 결과를
# 읽지 않게 하기 위해서다. 같은 폴더에 다시 돌리려면 OVERWRITE=1, 다른 폴더에 돌리려면 CAND_ROOT=<경로>.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=experiments/reranking/results_v62_hybrid_full_scan
V61=experiments/reranking/results_v61_hybrid_scan_audit
PY=${PY:-venv/bin/python}
CAND_ROOT=${CAND_ROOT:-$OUT/cand_fixed}
# OVERWRITE는 정확히 1일 때만 덮어쓴다 — `${OVERWRITE:+…}`는 0이나 빈 문자열이 아닌 어떤 값에도 켜져서 기존 결과를 보호하지 못했다(리뷰 2차)
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi
unset CLAUDE_EFFORT CLAUDE_API QDRANT_HYBRID_SCAN_LIMIT
mkdir -p $OUT/logs
: > $OUT/logs/progress.txt
for set in v06 v09 v10; do
  $PY $V61/run_candidates.py --arm A --set $set --out-root "$CAND_ROOT" ${OVERWRITE_FLAG:-} > $OUT/logs/run_fixed_${set}.log 2>&1
  rc=$?
  echo "done fixed $set exit=$rc $(date +%T) out=$CAND_ROOT" >> $OUT/logs/progress.txt
  if [ $rc -ne 0 ]; then
    echo "FAILED fixed $set exit=$rc — $OUT/logs/run_fixed_${set}.log 참조 (중단)" | tee -a $OUT/logs/progress.txt >&2
    exit $rc
  fi
done
echo "ALL DONE $(date +%T)" >> $OUT/logs/progress.txt
