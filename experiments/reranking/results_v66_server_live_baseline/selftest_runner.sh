#!/bin/zsh
# v66 실행 도구 자체 검사 — 모델·Vertex·Mongo 호출 없음. 모든 산출물·로그·장부는 임시 폴더에만 쓴다(실제 logs/·budget/ 미사용).
#   A. run_eval.py(실제 파일): live 게이트 · live 부분 실행 거부 · 덮어쓰기 거부 · 서버 상태 불일치 거부 · 실패 주입(exit 1, DONE 없음)
#   B. run_live.sh(가짜 PY): 중간 실패 시 그 코드로 즉시 종료·다음 단계 없음 · 입력 대조 실패 시 live 미호출 · 한도 고정 전달 · 승인 표시 필수
#   C. compare_inputs.py: 미완료 실행 비교 거부(exit 3)
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v66_server_live_baseline
T=$(mktemp -d "${TMPDIR:-/tmp}/v66_selftest.XXXXXX")
PASS=0; FAIL=0
ok() { if eval "$2"; then echo "ok   $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi }
E=(env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V66_LIVE_APPROVED)

# ---------- A. run_eval.py
"${E[@]}" V66_CODE=after V66_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a1 --expect-server-state $R/server_state_baseline.json \
  --ledger $T/ledger.jsonl --deadline-window-seconds 60 --cost-cap 1 --attempt-cap 1 > $T/a1.log 2>&1
ok "live: 승인 표시 없으면 exit 2" "[[ $? == 2 ]]"
ok "live 게이트 거부 시 장부 없음" "[[ ! -e $T/ledger.jsonl ]]"
"${E[@]}" V66_LIVE_APPROVED=1 V66_CODE=after V66_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a2 \
  --expect-server-state $R/server_state_baseline.json > $T/a2.log 2>&1
ok "live: 한도·장부 인자 없으면 exit 2" "[[ $? == 2 ]]"
"${E[@]}" V66_LIVE_APPROVED=1 V66_CODE=after V66_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 \
  --expect-server-state $R/server_state_baseline.json --ledger $T/ledger.jsonl --deadline-window-seconds 60 --cost-cap 1 \
  --attempt-cap 1 --query-ids q101 > $T/a3.log 2>&1
ok "live: 부분 실행(--query-ids) exit 2" "[[ $? == 2 ]]"
"${E[@]}" V66_LIVE_APPROVED=1 V66_CODE=after V66_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 \
  --expect-server-state $R/server_state_baseline.json --ledger $T/ledger.jsonl --deadline-window-seconds 60 --cost-cap 1 \
  --attempt-cap 1 --sets v06 > $T/a3b.log 2>&1
ok "live: 부분 실행(--sets) exit 2" "[[ $? == 2 ]]"
ok "live 거부 시 장부·Vertex 시도 없음" "[[ ! -e $T/ledger.jsonl ]]"
mkdir -p $T/a4/after_server_rep1 && echo x > $T/a4/after_server_rep1/keep
"${E[@]}" V66_CODE=after venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a4 --expect-server-state $R/server_state_baseline.json > $T/a4.log 2>&1
ok "기존 산출물 덮어쓰기 거부 exit 2" "[[ $? == 2 && -e $T/a4/after_server_rep1/keep ]]"
venv/bin/python -c "
import json,sys; d=json.load(open('$R/server_state_baseline.json')); d['container']['started_at']='2000-01-01T00:00:00Z'
json.dump(d,open('$T/state_changed.json','w'))"
"${E[@]}" V66_CODE=before venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a5 --expect-server-state $T/state_changed.json > $T/a5.log 2>&1
ok "서버 상태(재시작) 불일치 → exit 4 · DONE 없음" "[[ $? == 4 && ! -e $T/a5/before_server_rep1/DONE ]]"
ok "서버 상태 거부 기록 status=refused_server_state" "grep -q refused_server_state $T/a5/before_server_rep1/run_conditions.json"
"${E[@]}" V66_CODE=before V66_FAIL_INJECT=v09 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a6 --expect-server-state $R/server_state_baseline.json > $T/a6.log 2>&1
ok "실패 주입(v09 직전) → exit 1" "[[ $? == 1 ]]"
ok "실패 실행에 DONE 없음·status failed" "[[ ! -e $T/a6/before_server_rep1/DONE ]] && grep -q '\"status\": \"failed\"' $T/a6/before_server_rep1/run_conditions.json"
ok "실패 전 세트(v06)는 기록이 남음" "[[ -s $T/a6/before_server_rep1/v06_dev/v66_queries.jsonl ]]"
"${E[@]}" V66_CODE=before V66_FAIL_INJECT=v06 V66_MODEL=live V66_LIVE_APPROVED=1 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a7 \
  --expect-server-state $R/server_state_baseline.json --ledger $T/ledger.jsonl --deadline-window-seconds 60 --cost-cap 1 --attempt-cap 1 > $T/a7.log 2>&1
ok "실패 주입은 live에서 거부 exit 2" "[[ $? == 2 && ! -e $T/ledger.jsonl ]]"

# ---------- C. compare_inputs
venv/bin/python $R/compare_inputs.py --a $T/a6/before_server_rep1 --b $R/offline/stability/before_server_rep1 > $T/c1.log 2>&1
ok "미완료 실행 비교 거부 exit 3" "[[ $? == 3 ]]"

# ---------- B. run_live.sh (가짜 PY — 인자를 기록하고 규칙대로 종료)
cat > $T/stub.sh <<'STUB'
#!/bin/zsh
# with_env.py 호출은 기록 없이 뒤의 명령으로 넘긴다(실제 with_env.py처럼). 넘긴 키 목록은 따로 남긴다
if [[ "$1" == *with_env.py ]]; then echo "with_env $2" >> $STUB_LOG; shift 3; exec $STUB_SELF "$@"; fi
# 단계 이름: run_eval은 --run-name 또는 live_<code>_server_rep<n>, compare_inputs는 --out 파일 이름
name=""; rep=""; args=("$@")
for (( i=1; i<=$#args; i++ )); do
  [[ "${args[i]}" == --run-name ]] && name=${args[i+1]}
  [[ "${args[i]}" == --out ]] && name=${${args[i+1]:t}%.json}
  [[ "${args[i]}" == --rep ]] && rep=${args[i+1]}
done
[[ -z "$name" && "$*" == *run_eval.py* ]] && name=live_${V66_CODE}_server_rep${rep}
echo "${V66_MODEL:--} ${V66_CODE:--} $name $*" >> $STUB_LOG
for pat in ${=STUB_FAIL:-}; do
  [[ "$name" == "${pat%%=*}" ]] && exit ${pat##*=}
done
exit 0
STUB
chmod +x $T/stub.sh
runlive() {  # runlive <tag> <STUB_FAIL> [RUNS]
  rm -f $T/$1.calls
  "${E[@]}" V66_LIVE_APPROVED=1 PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/$1.calls STUB_FAIL="$2" LOG_DIR=$T/$1/logs \
    OUT_LIVE=$T/$1/live OUT_OFF=$T/$1/off CHECK_DIR=$T/$1/checks LEDGER=$T/$1/ledger.jsonl RUNS="${3:-before:1 after:1 after:2 before:2}" \
    zsh $R/run_live.sh > $T/$1.log 2>&1
}
runlive b1 ""
ok "정상 경로 exit 0 · LIVE DONE" "[[ $? == 0 ]] && grep -q 'LIVE DONE' $T/b1/logs/progress_live.txt"
ok "live 4회 순서 전1→후1→후2→전2" "[[ \"\$(grep '^live ' $T/b1.calls | awk '{print \$3}' | tr '\n' ' ')\" == 'live_before_server_rep1 live_after_server_rep1 live_after_server_rep2 live_before_server_rep2 ' ]]"
ok "with_env로 두 키만 넘김" "[[ \$(grep -c '^with_env GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS\$' $T/b1.calls) == 4 ]]"
ok "한도 고정 전달(\$4 · 800 · 5400초)" "grep '^live ' $T/b1.calls | grep -q -- '--cost-cap 4 --attempt-cap 800' && grep '^live ' $T/b1.calls | grep -q -- '--deadline-window-seconds 5400'"
ok "live마다 직전·직후 오프라인 지문(fake 8회)" "[[ \$(grep -c '^fake .*run_eval.py' $T/b1.calls) == 8 ]]"
ok "입력 대조 12회(직전↔기준·직후↔직전·live↔직전 × 4)" "[[ \$(grep -c 'compare_inputs.py' $T/b1.calls) == 12 ]]"
runlive b2 "live_after_server_rep1=1"
rc=$?
ok "후1 live 실패 → 그 코드(1)로 종료" "[[ $rc == 1 ]]"
ok "실패 뒤 직후 지문·다음 live 없음" "! grep -q 'post_after_rep1\\|live_after_server_rep2\\|pre_after_rep2' $T/b2.calls"
ok "중단 기록 STOP" "grep -q 'STOP live_after_server_rep1' $T/b2/logs/progress_live.txt"
runlive b3 "pre_before_rep1__vs_stability=3"
rc=$?
ok "직전 지문 ≠ 기준 → exit 3 · live 미호출" "[[ $rc == 3 ]] && ! grep -q '^live ' $T/b3.calls"
runlive b4 "live_before_server_rep1__vs_pre=3"
rc=$?
ok "live 입력 ≠ 직전 지문 → exit 3 · 다음 live 없음" "[[ $rc == 3 && \$(grep -c '^live ' $T/b4.calls) == 1 ]]"
runlive b5 "post_after_rep2__vs_pre=3"
rc=$?
ok "후2 직후 지문 변화 → exit 3 · 전2 live 없음" "[[ $rc == 3 && \$(grep -c '^live ' $T/b5.calls) == 3 ]]"
"${E[@]}" PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/b6.calls LOG_DIR=$T/b6/logs zsh $R/run_live.sh > $T/b6.log 2>&1
rc=$?
ok "승인 표시 없으면 시작 안 함" "[[ $rc != 0 && ! -e $T/b6.calls ]]"
"${E[@]}" V66_LIVE_APPROVED=yes PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/b7.calls LOG_DIR=$T/b7/logs zsh $R/run_live.sh > $T/b7.log 2>&1
rc=$?
ok "승인 표시가 정확히 1이 아니면 exit 2" "[[ $rc == 2 && ! -e $T/b7.calls ]]"
ok "실제 logs/·budget/ 미사용(이 검사가 쓴 곳은 임시 폴더뿐)" "[[ ! -e $R/budget/ledger.jsonl && ! -e $R/logs/progress_live.txt ]]"

echo "PASS $PASS FAIL $FAIL (임시 폴더 $T)"
(( FAIL == 0 ))
