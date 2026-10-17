#!/bin/zsh
# v63 오프라인 후보 검증 — 모델 호출 차단(가짜 결정적 응답), 실제 가사 스냅샷 · 같은 색인 복사본 · v49 Vertex dev 캐시.
# 순서 A1 → B1 → B2 → A2(순차, 동시 실행 없음). 하나라도 실패하면 그 종료 코드로 즉시 끝나고 비교로 넘어가지 않는다.
#   OUT_ROOT=<폴더>  출력 루트(기본 이 폴더/offline)   · OVERWRITE=1 일 때만 기존 산출물을 덮어쓴다(그 외 값은 무시)
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v63_hybrid_lyrics_validation
PY=${PY:-venv/bin/python}
LOGS=${LOG_DIR:-$R/logs}   # 자체 검사(selftest_failure.sh)는 다른 폴더를 준다
OUT_ROOT=${OUT_ROOT:-$R/offline}
OVR=()
[[ "${OVERWRITE:-}" == "1" ]] && OVR=(--overwrite)
mkdir -p $LOGS
for spec in A:1 B:1 B:2 A:2; do
  arm=${spec%%:*}; rep=${spec##*:}
  echo "start offline_${arm}_rep${rep} $(date +%T)" >> $LOGS/progress_offline.txt
  env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT $PY $R/run_v63.py --arm $arm --mode offline --rep $rep --out-root $OUT_ROOT $OVR \
    > $LOGS/offline_${arm}_rep${rep}.log 2>&1
  rc=$?
  echo "done offline_${arm}_rep${rep} exit=$rc $(date +%T)" >> $LOGS/progress_offline.txt
  if (( rc != 0 )); then echo "FAILED at offline_${arm}_rep${rep} (exit $rc) — 비교하지 않고 중단" >&2; exit $rc; fi
done
$PY $R/compare_v63.py --runs $OUT_ROOT/offline_A_rep1 $OUT_ROOT/offline_B_rep1 $OUT_ROOT/offline_B_rep2 $OUT_ROOT/offline_A_rep2 \
  --pairs A1:B1,A2:B2,A1:A2,B1:B2,A1:B2,A2:B1 --out $OUT_ROOT/compare --require-repeat-identical $OVR \
  > $LOGS/offline_compare.log 2>&1
rc=$?
echo "compare offline exit=$rc $(date +%T)" >> $LOGS/progress_offline.txt
(( rc != 0 )) && { echo "FAILED at compare (exit $rc)" >&2; exit $rc; }
echo "OFFLINE ALL DONE"
