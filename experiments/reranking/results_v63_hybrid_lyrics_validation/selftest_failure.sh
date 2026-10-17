#!/bin/zsh
# v63 실행 도구 자체 검사 — 모델·DB 호출 없음.
#  1) run_offline.sh / run_live.sh 실패 전달: 가짜 실행기(인자만 기록)로 2번째 실행을 실패시키면 그 종료 코드로 끝나고 비교·다음 실행을 하지 않는다
#  2) OVERWRITE: 정확히 1일 때만 --overwrite 전달(미설정·0·yes는 전달 안 함)
#  3) 비교 실패 코드 전달(가짜 비교기 exit 3 → 스크립트 exit 3)
#  4) 실제 run_v63.py: V63_FAIL_INJECT로 첫 세트에서 실패 → exit 1 · DONE 없음 · status=failed, 같은 폴더 재실행은 덮어쓰기 거부 exit 2
#  5) 실제 compare_v63.py: 미완료 실행 → exit 3, 부분(스모크) 실행 → exit 3
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v63_hybrid_lyrics_validation
T=$(mktemp -d /private/tmp/v63_selftest.XXXXXX)
FAKE=$T/fake_py.sh
cat > $FAKE <<'EOF'
#!/bin/zsh
echo "$*" >> $FAKE_LOG
if [[ "$*" == *compare_v63.py* ]]; then exit ${FAKE_COMPARE_RC:-0}; fi
if [[ -n ${FAKE_FAIL_AT:-} && "$*" == *"--arm ${FAKE_FAIL_AT%%:*} "*"--rep ${FAKE_FAIL_AT##*:} "* ]]; then exit 7; fi
exit 0
EOF
chmod +x $FAKE
fail=0
check() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1 (기대 $3, 실제 $2)"; fail=1; fi; }

# 1) 실패 전달 — 두 번째(B1) 실패
LOG_DIR=$T/logs FAKE_LOG=$T/l1 FAKE_FAIL_AT=B:1 PY=$FAKE OUT_ROOT=$T/o1 zsh $R/run_offline.sh >/dev/null 2>&1; rc=$?
check "offline: B1 실패 → exit 7" $rc 7
check "offline: B1 뒤 실행·비교 없음" "$(grep -c . $T/l1)" 2
LOG_DIR=$T/logs FAKE_LOG=$T/l1b FAKE_FAIL_AT=B:1 PY=$FAKE OUT_ROOT=$T/o1b DEADLINE_EPOCH=1 zsh $R/run_live.sh >/dev/null 2>&1; rc=$?
check "live: B1 실패 → exit 7" $rc 7
check "live: B1 뒤 실행 없음" "$(grep -c run_v63.py $T/l1b)" 2
# 2) OVERWRITE
for v in "" 0 yes; do
  rm -f $T/l2; LOG_DIR=$T/logs FAKE_LOG=$T/l2 OVERWRITE=$v PY=$FAKE OUT_ROOT=$T/o2 zsh $R/run_offline.sh >/dev/null 2>&1
  check "OVERWRITE='$v' → --overwrite 없음" "$(grep -c -- --overwrite $T/l2)" 0
done
rm -f $T/l2; LOG_DIR=$T/logs FAKE_LOG=$T/l2 OVERWRITE=1 PY=$FAKE OUT_ROOT=$T/o2 zsh $R/run_offline.sh >/dev/null 2>&1
check "OVERWRITE=1 → 실행 4 + 비교 1 모두 --overwrite" "$(grep -c -- --overwrite $T/l2)" 5
check "OVERWRITE=1 → 새 출력 경로가 실행·비교에 전달" "$(grep -c -- "$T/o2" $T/l2)" 5
# 3) 비교 실패 전달
LOG_DIR=$T/logs FAKE_LOG=$T/l3 FAKE_COMPARE_RC=3 PY=$FAKE OUT_ROOT=$T/o3 zsh $R/run_offline.sh >/dev/null 2>&1; rc=$?
check "비교 exit 3 전달" $rc 3
# 4) 실제 실행기 실패 주입
V63_FAIL_INJECT=v06 env -u CLAUDE_EFFORT venv/bin/python $R/run_v63.py --arm B --mode offline --rep 9 --out-root $T/real > $T/real.log 2>&1; rc=$?
check "run_v63 실패 주입 → exit 1" $rc 1
check "run_v63 실패 → DONE 없음" "$([[ -e $T/real/offline_B_rep9/DONE ]] && echo yes || echo no)" no
check "run_v63 실패 → status failed" "$(venv/bin/python -c "import json;print(json.load(open('$T/real/offline_B_rep9/run_conditions.json'))['status'])")" failed
env -u CLAUDE_EFFORT venv/bin/python $R/run_v63.py --arm B --mode offline --rep 9 --out-root $T/real > $T/real2.log 2>&1; rc=$?
check "run_v63 기존 산출물 → 덮어쓰기 거부 exit 2" $rc 2
# 5) 비교기: 미완료 실행
venv/bin/python $R/compare_v63.py --runs $T/real/offline_B_rep9 --pairs B9:B9 --out $T/cmp1 > $T/cmp1.log 2>&1; rc=$?
check "compare: 미완료 실행 → exit 3" $rc 3
# 5b) 부분(스모크) 실행 → DONE partial → exit 3
env -u CLAUDE_EFFORT venv/bin/python $R/run_v63.py --arm B --mode offline --rep 8 --out-root $T/real --sets v06 --query-ids q101 > $T/smoke.log 2>&1; rc=$?
check "스모크 실행 exit 0" $rc 0
venv/bin/python $R/compare_v63.py --runs $T/real/offline_B_rep8 --pairs B8:B8 --out $T/cmp2 > $T/cmp2.log 2>&1; rc=$?
check "compare: 부분 실행 → exit 3" $rc 3
echo "selftest 폴더: $T"
exit $fail
