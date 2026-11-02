#!/bin/zsh
# A1 → B1 → B2 → A2 순차 실행. 한도는 이 파일에 고정한다. 장부가 있으면 시작하지 않는다.
set -u
R=${0:A:h}; V75=$R/../results_v75_work_audio_live_ab; source $R/env.zsh; cd "$MAIN_DIR"
: ${V79_LIVE_APPROVED:?승인 표시 V79_LIVE_APPROVED=1이 없다}
export V79_COST_CAP=6 V79_ATTEMPT_CAP=1500 V79_WINDOW=10800
# 장부가 있으면 멈춘다. 이미 쓴 비용을 상한에 포함한 채 이어 갈 때만 V79_RESUME_LEDGER=1을 준다(장부를 지우지 않는다).
L=$R/budget/ledger.jsonl
if [[ -z ${V79_PLAN_ONLY:-} && -e $L && ${V79_RESUME_LEDGER:-} != 1 ]]; then echo "장부가 이미 있다 — 멈춘다" >&2; exit 2; fi
mkdir -p $R/budget $R/logs; P=$R/logs/progress.txt
SPECS=(off:1 hits:1 hits:2 off:2 syn:1 syn:2)
[[ -n ${RUNS:-} ]] && SPECS=(${=RUNS})
if [[ -n ${V79_PLAN_ONLY:-} ]]; then print -l $SPECS; exit 0; fi
for spec in $SPECS; do
  cfg=${spec%%:*}; rep=${spec##*:}; name=${cfg}_rep$rep
  case $cfg in
    off) export GEMINI_RERANK_CLUE_EVIDENCE=off CLUE_SPARSE_SYNONYMS=0;;
    hits) export GEMINI_RERANK_CLUE_EVIDENCE=hits CLUE_SPARSE_SYNONYMS=0;;
    syn) export GEMINI_RERANK_CLUE_EVIDENCE=hits CLUE_SPARSE_SYNONYMS=1;;
    dsyn) export GEMINI_RERANK_CLUE_EVIDENCE=off CLUE_SPARSE_SYNONYMS=1;;
    *) echo "모르는 구성 $cfg" >&2; exit 2;;
  esac
  [[ -e $R/runs/$name ]] && { echo "STOP 실행 폴더가 이미 있다 $name $(date +%T)" >> $P; exit 3; }
  for s in v06 v09 v10; do
    echo "start $name $s $(date +%T)" >> $P; rm -f $QDRANT_PATH/.lock
    venv/bin/python $V75/with_env.py GCP_PROJECT_ID -- venv/bin/python $R/one_eval.py --set $s --out $R/runs/$name/$s --run-tag ${name}_$s --ledger $L > $R/logs/${name}_$s.log 2>&1; rc=$?
    echo "done $name $s exit=$rc $(date +%T)" >> $P
    (( rc == 0 )) || { echo "STOP $name $s exit=$rc $(date +%T)" >> $P; exit $rc; }
    venv/bin/python $R/check_run.py $R/runs/$name/$s $R/logs/${name}_$s.log >> $P 2>&1 || { echo "STOP $name $s 리랭킹 실패 $(date +%T)" >> $P; exit 91; }
  done
done
echo "ALL_DONE $(date +%T)" >> $P
