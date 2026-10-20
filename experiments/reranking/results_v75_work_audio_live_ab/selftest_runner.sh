#!/bin/zsh
# v75 실행 도구 자체 검사(v73 selftest_runner.sh 사본을 작품명 ON 고정 · 소리 on|skip 구성에 맞춤) — 모델·Vertex·Mongo 호출 없음.
# 모든 산출물·로그·장부는 임시 폴더에만 쓴다. v73의 "실제 logs/·budget/가 없어야 한다" 검사는 작업 폴더 상태(이미 있는 실제 장부)
# 때문에 실패할 수 있어, v75에서는 **검사 전후 결과 폴더의 파일 목록·크기·수정 시각이 같은지**로 바꿨다(이 검사가 결과 폴더에 쓰지 않음).
#   A. run_eval.py(실제 파일): live 게이트 · 부분 실행 거부 · 기준 해석 상태 파일·지문 필수 · 지문 불일치 거부 · 덮어쓰기 거부 ·
#      서버 상태 불일치 거부 · 실패 주입(exit 1, DONE 없음) · 예열 질의 고정(q105, 기록 밖) · 작품명 OFF 구성 거부
#   A2. 작품명 해석 상태: 캐시 누락 주입(failed:cache_miss) → 정지 · 기준 상태와 다른 파일 → 정지(정상 보류로 넘어가지 않음)
#   B. run_live.sh(가짜 PY): 순서 A1(on)→B1(skip)→B2(skip)→A2(on) · 한도($8·1000·14400초)·해석 상태 지문 고정 전달 · 같은 장부 ·
#      중간 실패/입력 차이 시 즉시 종료 · 승인 표시 필수 · 장부가 이미 있으면 시작 안 함
#   C. compare_inputs.py: 미완료 실행 비교 거부(exit 3)
#   D. 소리 경로 실패 주입(fake, on) → 멈춤, live에서는 거부
#   E. fault 모드(가짜 전송): 실제 리랭커·예산 통제 경로에서 504 3회 → 정지·DONE 없음·다음 실행 시작 불가 · 504 뒤 재시도는 같은 본문
#      (재시도 수정 #87) · 403 첫 오류 정지 · 평가 전용 제한 60/120초와 소리 스위치가 실행 환경에 기록됨
#   F. 실제 작품명 해석 어댑터 차단(불리면 RunAborted)
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v75_work_audio_live_ab
T=$(mktemp -d "${TMPDIR:-/tmp}/v75_selftest.XXXXXX")
[[ -n "$T" && -d "$T" ]] || { echo "임시 폴더 생성 실패" >&2; exit 2; }
PASS=0; FAIL=0
ok() { if eval "$2"; then echo "ok   $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi }
snapshot() { find $R -path $R/__pycache__ -prune -o -type f -print0 | xargs -0 stat -f '%N %z %m' | sort; }
snapshot > $T/result_dir_before.txt
E=(env -u CLAUDE_EFFORT -u QDRANT_HYBRID_SCAN_LIMIT -u V75_LIVE_APPROVED -u V75_FAULT_PLAN -u V75_CONFIG)
WS_SHA=1fb2f754366de693
SUB=(--expect-work-status $R/inputs/expected_work_status.json --expect-work-status-sha16 $WS_SHA)
LIM=(--ledger $T/ledger.jsonl --deadline-window-seconds 60 --cost-cap 1 --attempt-cap 1)
ST=(--expect-server-state $R/server_state_baseline.json)

# ---------- A. run_eval.py
"${E[@]}" V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a1 "${ST[@]}" "${LIM[@]}" "${SUB[@]}" > $T/a1.log 2>&1
ok "live: 승인 표시 없으면 exit 2" "[[ $? == 2 ]]"
ok "live 게이트 거부 시 장부 없음" "[[ ! -e $T/ledger.jsonl ]]"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=skip V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a2 "${ST[@]}" "${SUB[@]}" > $T/a2.log 2>&1
ok "live: 한도·장부 인자 없으면 exit 2" "[[ $? == 2 ]]"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" > $T/a3.log 2>&1
ok "live: 기준 해석 상태 파일 없으면 exit 2" "[[ $? == 2 ]] && grep -q 'expect-work-status' $T/a3.log"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" \
  --expect-work-status $R/inputs/expected_work_status.json > $T/a3b.log 2>&1
ok "live: 해석 상태 지문 인자 없으면 exit 2" "[[ $? == 2 ]]"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" \
  --expect-work-status $R/inputs/expected_work_status.json --expect-work-status-sha16 0000000000000000 > $T/a3c.log 2>&1
ok "live: 해석 상태 지문 불일치 exit 2" "[[ $? == 2 ]] && grep -q '기준 해석 상태 지문' $T/a3c.log"
venv/bin/python -c "
import json; d=json.load(open('$R/inputs/expected_work_status.json')); d['status_by_config']['work']['c707']='abstained'
json.dump(d,open('$T/status_changed.json','w'))"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" \
  --expect-work-status $T/status_changed.json --expect-work-status-sha16 $WS_SHA > $T/a3d.log 2>&1
ok "live: 고친 해석 상태 파일 거부 exit 2" "[[ $? == 2 ]]"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" "${SUB[@]}" --query-ids q101 > $T/a3e.log 2>&1
ok "live: --query-ids 거부 exit 2" "[[ $? == 2 ]]"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=skip V75_MODEL=live venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a3 "${ST[@]}" "${LIM[@]}" "${SUB[@]}" --sets v06 > $T/a3f.log 2>&1
ok "live: --sets 거부 exit 2" "[[ $? == 2 ]]"
ok "live 거부 시 장부·Vertex 시도 없음" "[[ ! -e $T/ledger.jsonl ]]"
"${E[@]}" V75_CONFIG=off V75_AUDIO=on venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a0 "${ST[@]}" "${SUB[@]}" > $T/a0.log 2>&1
ok "작품명 OFF 구성(V75_CONFIG=off) 시작 전 거부" "[[ $? != 0 && ! -e $T/a0 ]] && grep -q 'V75_CONFIG=work' $T/a0.log"
"${E[@]}" V75_AUDIO=half venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a0b "${ST[@]}" "${SUB[@]}" > $T/a0b.log 2>&1
ok "소리 스위치가 on|skip이 아니면 시작 전 거부" "[[ $? != 0 && ! -e $T/a0b ]]"
mkdir -p $T/a4/work_on_server_rep1 && echo x > $T/a4/work_on_server_rep1/keep
"${E[@]}" V75_AUDIO=on venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a4 "${ST[@]}" "${SUB[@]}" > $T/a4.log 2>&1
ok "기존 산출물 덮어쓰기 거부 exit 2" "[[ $? == 2 && -e $T/a4/work_on_server_rep1/keep ]]"
venv/bin/python -c "
import json; d=json.load(open('$R/server_state_baseline.json')); d['container']['started_at']='2000-01-01T00:00:00Z'
json.dump(d,open('$T/state_changed.json','w'))"
"${E[@]}" V75_AUDIO=skip venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a5 --expect-server-state $T/state_changed.json "${SUB[@]}" > $T/a5.log 2>&1
ok "서버 상태(재시작) 불일치 → exit 4 · DONE 없음" "[[ $? == 4 && ! -e $T/a5/work_skip_server_rep1/DONE ]]"
ok "서버 상태 거부 기록 status=refused_server_state" "grep -q refused_server_state $T/a5/work_skip_server_rep1/run_conditions.json"
"${E[@]}" V75_AUDIO=skip V75_FAIL_INJECT=v09 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a6 "${ST[@]}" "${SUB[@]}" > $T/a6.log 2>&1
ok "실패 주입(v09 직전) → exit 1" "[[ $? == 1 ]]"
ok "실패 실행에 DONE 없음·status failed" "[[ ! -e $T/a6/work_skip_server_rep1/DONE ]] && grep -q '\"status\": \"failed\"' $T/a6/work_skip_server_rep1/run_conditions.json"
ok "실패 전 세트(v06)는 기록이 남음(57건)·해석 상태 = 기준" "[[ \$(wc -l < $T/a6/work_skip_server_rep1/v06_dev/v75_queries.jsonl | tr -d ' ') == 57 ]] && venv/bin/python -c \"import json,sys; c=json.load(open('$T/a6/work_skip_server_rep1/run_conditions.json')); sys.exit(0 if c['sets_done']['v06']['work_status_counts']=={'not_triggered': 49, 'applied': 7, 'abstained': 1} else 1)\""
ok "skip 실행 v06: CLAP·오디오 조회 0 · AUDIO_FUSION_SCALE=0 · 평가 전용 제한 60/120 기록" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/a6/work_skip_server_rep1/run_conditions.json')); s=c['sets_done']['v06']; e=c['env']; sys.exit(0 if s['audio_collection_queries']==0 and s['queries_with_clap_embed']==0 and e['AUDIO_FUSION_SCALE']=='0' and e['GEMINI_RERANK_TIMEOUT_SECONDS']=='60' and e['GEMINI_RERANK_BUDGET_SECONDS']=='120' else 1)\""
ok "예열 질의 = q105(판정 통과·applied), 기록의 q105는 평가기 실행 1회분뿐" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/a6/work_skip_server_rep1/run_conditions.json')); w=c['warmup']; q=[json.loads(l)['query_id'] for l in open('$T/a6/work_skip_server_rep1/v06_dev/v75_queries.jsonl')]; sys.exit(0 if w['query_id']=='q105' and w['work_status']=='applied' and q.count('q105')==1 else 1)\""

# ---------- A2. 작품명 해석 상태(정상 보류로 숨기지 않음)
"${E[@]}" V75_AUDIO=skip V75_WORK_MISS_INJECT=c707 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a8 "${ST[@]}" "${SUB[@]}" > $T/a8.log 2>&1
rc=$?
ok "A2 캐시 누락 주입(c707) → exit 1 · DONE 없음 · RunAborted" "[[ $rc == 1 && ! -e $T/a8/work_skip_server_rep1/DONE ]] && grep -q 'RunAborted' $T/a8/work_skip_server_rep1/run_conditions.json"
ok "A2 정지 사유 = c707 failed:cache_miss(외부 해석 호출 0)" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/a8/work_skip_server_rep1/run_conditions.json')); v=c['work_status_violations']; sys.exit(0 if len(v)==1 and v[0].startswith('c707') and 'failed:cache_miss' in v[0] and c['work_miss_inject']=='c707' else 1)\""
"${E[@]}" V75_AUDIO=on V75_WORK_MISS_INJECT=c707 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a9 "${ST[@]}" > $T/a9.log 2>&1
rc=$?
ok "A2 기준 파일 없이도 캐시 누락은 정지(실패 상태 금지)" "[[ $rc == 1 ]] && grep -q 'failed:cache_miss' $T/a9/work_on_server_rep1/run_conditions.json"
"${E[@]}" V75_AUDIO=on venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a10 "${ST[@]}" --expect-work-status $T/status_changed.json > $T/a10.log 2>&1
rc=$?
ok "A2 기준 상태와 다른 질의(c707 기준 abstained) → exit 1 · ≠ 기준" "[[ $rc == 1 ]] && grep -q 'c707.*기준' $T/a10/work_on_server_rep1/run_conditions.json"
"${E[@]}" V75_AUDIO=on V75_WORK_MISS_INJECT=c707 V75_MODEL=live V75_LIVE_APPROVED=1 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a11 "${ST[@]}" "${LIM[@]}" "${SUB[@]}" > $T/a11.log 2>&1
ok "A2 캐시 누락 주입은 live에서 거부 exit 2" "[[ $? == 2 && ! -e $T/ledger.jsonl ]]"
"${E[@]}" V75_AUDIO=skip V75_FAIL_INJECT=v06 V75_MODEL=live V75_LIVE_APPROVED=1 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/a7 "${ST[@]}" "${LIM[@]}" "${SUB[@]}" > $T/a7.log 2>&1
ok "실패 주입은 live에서 거부 exit 2" "[[ $? == 2 && ! -e $T/ledger.jsonl ]]"

# ---------- C. compare_inputs
venv/bin/python $R/compare_inputs.py --a $T/a6/work_skip_server_rep1 --b $R/offline/full/work_skip_full_warm > $T/c1.log 2>&1
ok "미완료 실행 비교 거부 exit 3" "[[ $? == 3 ]]"

# ---------- B. run_live.sh (가짜 PY — 인자를 기록하고 규칙대로 종료)
cat > $T/stub.sh <<'STUB'
#!/bin/zsh
if [[ "$1" == *with_env.py ]]; then echo "with_env $2" >> $STUB_LOG; shift 3; exec $STUB_SELF "$@"; fi
name=""; rep=""; args=("$@")
for (( i=1; i<=$#args; i++ )); do
  [[ "${args[i]}" == --run-name ]] && name=${args[i+1]}
  [[ "${args[i]}" == --out ]] && name=${${args[i+1]:t}%.json}
  [[ "${args[i]}" == --rep ]] && rep=${args[i+1]}
done
[[ -z "$name" && "$*" == *run_eval.py* ]] && name=live_work_${V75_AUDIO}_server_rep${rep}
echo "${V75_MODEL:--} ${V75_AUDIO:--} $name $*" >> $STUB_LOG
for pat in ${=STUB_FAIL:-}; do
  [[ "$name" == "${pat%%=*}" ]] && exit ${pat##*=}
done
exit 0
STUB
chmod +x $T/stub.sh
runlive() {  # runlive <tag> <STUB_FAIL> — 호출 기록 파일은 태그마다 새 이름이라 지우지 않는다
  "${E[@]}" V75_LIVE_APPROVED=1 PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/$1.calls STUB_FAIL="$2" LOG_DIR=$T/$1/logs \
    OUT_LIVE=$T/$1/live OUT_OFF=$T/$1/off CHECK_DIR=$T/$1/checks LEDGER=$T/$1/ledger.jsonl \
    zsh $R/run_live.sh > $T/$1.log 2>&1
}
runlive b1 ""
ok "정상 경로 exit 0 · LIVE DONE" "[[ $? == 0 ]] && grep -q 'LIVE DONE' $T/b1/logs/progress_live.txt"
ok "live 4회 순서 A1(on) → B1(skip) → B2(skip) → A2(on)" "[[ \"\$(grep '^live ' $T/b1.calls | awk '{print \$3}' | tr '\n' ' ')\" == 'live_work_on_server_rep1 live_work_skip_server_rep1 live_work_skip_server_rep2 live_work_on_server_rep2 ' ]]"
ok "live 구성 스위치 = V75_AUDIO on·skip·skip·on" "[[ \"\$(grep '^live ' $T/b1.calls | awk '{print \$2}' | tr '\n' ' ')\" == 'on skip skip on ' ]]"
ok "with_env로 두 키만 넘김(4회)" "[[ \$(grep -c '^with_env GCP_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS\$' $T/b1.calls) == 4 ]]"
ok "한도 고정 전달(\$8 · 1000 · 14400초)" "[[ \$(grep '^live ' $T/b1.calls | grep -- '--cost-cap 8 --attempt-cap 1000' | grep -c -- '--deadline-window-seconds 14400') == 4 ]]"
ok "live·지문 실행 모두 같은 해석 상태 지문" "[[ \$(grep 'run_eval.py' $T/b1.calls | grep -c -- '--expect-work-status-sha16 $WS_SHA') == 12 ]]"
ok "live·지문 실행 모두 dev 176 전체(--query-ids·--sets·부분집합 없음)" "! grep 'run_eval.py' $T/b1.calls | grep -q -- '--query-ids\\|--sets\\|--subset'"
ok "live 네 실행이 같은 장부를 씀" "[[ \$(grep '^live ' $T/b1.calls | grep -o -- '--ledger [^ ]*' | sort -u | wc -l | tr -d ' ') == 1 ]]"
ok "live마다 직전·직후 오프라인 지문(fake 8회, 구성 = 같은 live의 소리 값)" "[[ \$(grep -c '^fake .*run_eval.py' $T/b1.calls) == 8 && \$(grep '^fake ' $T/b1.calls | awk '{print \$2\":\"\$3}' | grep -c 'on:pre_work_on\\|on:post_work_on\\|skip:pre_work_skip\\|skip:post_work_skip') == 8 ]]"
ok "입력 대조 12회(직전↔기준·직후↔직전·live↔직전 × 4), 기준은 같은 구성 전체 실행" "[[ \$(grep -c 'compare_inputs.py' $T/b1.calls) == 12 && \$(grep 'compare_inputs.py' $T/b1.calls | grep -c 'full/work_on_full_warm\\|full/work_skip_full_warm') == 4 ]]"
runlive b2 "live_work_on_server_rep1=1"
rc=$?
ok "A1 live 실패(장애 집중 정지 포함) → 그 코드(1)로 종료" "[[ $rc == 1 ]]"
ok "A1 실패 뒤 직후 지문·B1 없음" "! grep -q 'post_work_on_rep1\\|live_work_skip_server_rep1\\|pre_work_skip_rep1' $T/b2.calls"
ok "중단 기록 STOP" "grep -q 'STOP live_work_on_server_rep1' $T/b2/logs/progress_live.txt"
runlive b3 "pre_work_on_rep1__vs_reference=3"
rc=$?
ok "직전 지문 ≠ 기준 → exit 3 · live 미호출" "[[ $rc == 3 ]] && ! grep -q '^live ' $T/b3.calls"
runlive b4 "live_work_skip_server_rep1__vs_pre=3"
rc=$?
ok "B1 live 입력 ≠ 직전 지문 → exit 3 · B2 없음" "[[ $rc == 3 && \$(grep -c '^live ' $T/b4.calls) == 2 ]]"
runlive b5 "post_work_skip_rep2__vs_pre=3"
rc=$?
ok "B2 직후 지문 변화 → exit 3 · A2 live 없음" "[[ $rc == 3 && \$(grep -c '^live ' $T/b5.calls) == 3 ]]"
runlive b8 "live_work_on_server_rep2=1"
rc=$?
ok "A2 live 실패 → exit 1 · 직후 지문 없음" "[[ $rc == 1 ]] && ! grep -q 'post_work_on_rep2' $T/b8.calls"
mkdir -p $T/b9 && echo '{}' > $T/b9/ledger.jsonl
"${E[@]}" V75_LIVE_APPROVED=1 PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/b9.calls LOG_DIR=$T/b9/logs LEDGER=$T/b9/ledger.jsonl \
  OUT_LIVE=$T/b9/live OUT_OFF=$T/b9/off CHECK_DIR=$T/b9/checks zsh $R/run_live.sh > $T/b9.log 2>&1
rc=$?
ok "장부가 이미 있으면 시작 안 함(exit 2 · 호출 0 — 재개·새 장부 없음)" "[[ $rc == 2 && ! -e $T/b9.calls ]]"
"${E[@]}" PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/b6.calls LOG_DIR=$T/b6/logs LEDGER=$T/b6/ledger.jsonl zsh $R/run_live.sh > $T/b6.log 2>&1
rc=$?
ok "승인 표시 없으면 시작 안 함" "[[ $rc != 0 && ! -e $T/b6.calls ]]"
"${E[@]}" V75_LIVE_APPROVED=yes PY=$T/stub.sh STUB_SELF=$T/stub.sh STUB_LOG=$T/b7.calls LOG_DIR=$T/b7/logs LEDGER=$T/b7/ledger.jsonl zsh $R/run_live.sh > $T/b7.log 2>&1
rc=$?
ok "승인 표시가 정확히 1이 아니면 exit 2" "[[ $rc == 2 && ! -e $T/b7.calls ]]"

# ---------- D. 경로 실패 주입(fake, 실제 서버 읽기만)
"${E[@]}" V75_AUDIO=on V75_PATH_FAIL_INJECT=audio venv/bin/python $R/run_eval.py --rep 1 --out-root $T/d1 "${ST[@]}" "${SUB[@]}" > $T/d1.log 2>&1
rc=$?
ok "소리 경로 실패 주입(on) → exit 1 · DONE 없음(빈 결과로 넘어가지 않음)" "[[ $rc == 1 && ! -e $T/d1/work_on_server_rep1/DONE ]] && grep -q '경로 실패' $T/d1/work_on_server_rep1/run_conditions.json"
"${E[@]}" V75_AUDIO=on V75_PATH_FAIL_INJECT=audio V75_MODEL=live V75_LIVE_APPROVED=1 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/d2 "${ST[@]}" \
  --ledger $T/ledger_d2.jsonl --deadline-window-seconds 60 --cost-cap 1 --attempt-cap 1 "${SUB[@]}" > $T/d2.log 2>&1
rc=$?
ok "경로 실패 주입의 live 사용 거부 exit 2 · 장부 없음" "[[ $rc == 2 && ! -e $T/ledger_d2.jsonl ]]"

# ---------- E. fault 모드 — 실제 리랭커·예산 통제·RunAborted 경로, 가짜 전송(네트워크·자격 증명 없음)
FARGS=(--expect-server-state $R/server_state_baseline.json --deadline-window-seconds 600 --cost-cap 1.5 --attempt-cap 200 "${SUB[@]}")
"${E[@]}" V75_AUDIO=on V75_MODEL=fault V75_FAULT_PLAN=504 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e1 --ledger $T/e_ledger.jsonl "${FARGS[@]}" > $T/e1.log 2>&1
rc=$?
ok "E 504 주입(A1) → exit 1 · DONE 없음 · status failed" "[[ $rc == 1 && ! -e $T/e1/fault_work_on_server_rep1/DONE ]] && grep -q '\"status\": \"failed\"' $T/e1/fault_work_on_server_rep1/run_conditions.json"
ok "E 장부 정지 사유 = 연속 504 3회" "grep '\"halt\"' $T/e_ledger.jsonl | grep -q '연속 504 3회'"
ok "E 정지 뒤 새 시도 없음(generate 시도 = 504 실패 3)" "venv/bin/python -c \"import json,sys; r=[json.loads(l) for l in open('$T/e_ledger.jsonl')]; g=[x for x in r if x.get('event')=='reserve' and x['kind']=='generate']; f=[x for x in r if x.get('event')=='fail']; h=[i for i,x in enumerate(r) if x.get('event')=='halt']; sys.exit(0 if len(g)==3 and len(f)==3 and all(x.get('error_code')==504 for x in f) and not [x for x in r[h[0]+1:] if x.get('event')=='reserve'] else 1)\""
ok "E 504 뒤 재시도는 같은 본문(정정 지시 없음 — 재시도 수정 #87 반영 코드)" "venv/bin/python -c \"import json,sys; r=[json.loads(l) for l in open('$T/e_ledger.jsonl')]; g=[x['body_chars'] for x in r if x.get('event')=='reserve' and x['kind']=='generate']; sys.exit(0 if len(g)==3 and len(set(g))==1 else 1)\""
ok "E 실행 오류 = RunAborted(예산 통제 정지)" "grep -q 'RunAborted' $T/e1/fault_work_on_server_rep1/run_conditions.json"
ok "E 미확인 비용 = 실패 예약액 보존(0 아님)" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/e1/fault_work_on_server_rep1/run_conditions.json')); b=c['budget_after']; sys.exit(0 if b['unverified_usd']>0.2 and b['settled_usd']==0 else 1)\""
ok "E 외부 연결 시도 0 · 가짜 전송만 사용" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/e1/fault_work_on_server_rep1/run_conditions.json')); sys.exit(0 if c['network_attempts']==[] and c['fault_transport_seen']['generate']==3 else 1)\""
ok "E 실행 환경: AUDIO_FUSION_SCALE=1 · 호출 60초 · 질의 120초 · 코드 사본 81965f6(패치 없음)" "venv/bin/python -c \"import json,sys; c=json.load(open('$T/e1/fault_work_on_server_rep1/run_conditions.json')); e=c['env']; ci=c['code_identity']; sys.exit(0 if e['AUDIO_FUSION_SCALE']=='1' and e['GEMINI_RERANK_TIMEOUT_SECONDS']=='60' and e['GEMINI_RERANK_BUDGET_SECONDS']=='120' and ci['commit'].startswith('81965f6') and ci['patched_blobs']=={} and ci['src_py_tree_sha16']=='6dcd5e5c40802ab6' else 1)\""
n_before=$(wc -l < $T/e_ledger.jsonl)
"${E[@]}" V75_AUDIO=skip V75_MODEL=fault V75_FAULT_PLAN=504 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e2 --ledger $T/e_ledger.jsonl "${FARGS[@]}" > $T/e2.log 2>&1
rc=$?
ok "E 정지된 장부로 B1 시작 거부 → exit 1 · 새 시도 0" "[[ $rc == 1 && \$(grep -c '\"reserve\"' $T/e_ledger.jsonl) == \$(head -n $n_before $T/e_ledger.jsonl | grep -c '\"reserve\"') ]] && grep -q '정지 상태' $T/e2/fault_work_skip_server_rep1/run_conditions.json"
"${E[@]}" V75_AUDIO=skip V75_MODEL=fault V75_FAULT_PLAN=403 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e3 --ledger $T/e_ledger403.jsonl "${FARGS[@]}" > $T/e3.log 2>&1
rc=$?
ok "E 403 주입 → 첫 오류에서 정지(generate 시도 1) · exit 1" "[[ $rc == 1 ]] && grep '\"halt\"' $T/e_ledger403.jsonl | grep -q '인증·권한·결제' && [[ \$(grep '\"reserve\"' $T/e_ledger403.jsonl | grep -c generate) == 1 ]]"
ok "E 해석기 = 캐시(실제 해석 시도 0) · skip 실행 AUDIO_FUSION_SCALE=0" "grep -q '\"work_resolver\": \"cache:' $T/e3/fault_work_skip_server_rep1/run_conditions.json && grep -q '\"AUDIO_FUSION_SCALE\": \"0\"' $T/e3/fault_work_skip_server_rep1/run_conditions.json"
"${E[@]}" V75_LIVE_APPROVED=1 V75_AUDIO=on V75_MODEL=fault V75_FAULT_PLAN=504 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e4 --ledger $T/e4.jsonl "${FARGS[@]}" > $T/e4.log 2>&1
ok "E fault는 승인 표시와 함께 거부 exit 2" "[[ $? == 2 && ! -e $T/e4.jsonl ]]"
"${E[@]}" V75_AUDIO=on V75_MODEL=fault V75_FAULT_PLAN=504 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e5 --ledger $R/budget/selftest_should_not_exist.jsonl "${FARGS[@]}" > $T/e5.log 2>&1
ok "E fault는 결과 폴더 안 장부 거부 exit 2" "[[ $? == 2 && ! -e $R/budget/selftest_should_not_exist.jsonl ]]"
"${E[@]}" V75_AUDIO=on V75_MODEL=fake V75_FAULT_PLAN=504 venv/bin/python $R/run_eval.py --rep 1 --out-root $T/e6 "${ST[@]}" "${SUB[@]}" > $T/e6.log 2>&1
ok "E V75_FAULT_PLAN은 fault 밖에서 거부 exit 2" "[[ $? == 2 ]]"
ok "E 서버 상태 = 기준(장애 주입이 서버를 바꾸지 않음)" "venv/bin/python $R/server_state.py --out $T/e_state.json --expect $R/server_state_baseline.json > /dev/null"

# ---------- F. 실제 작품명 해석 어댑터 차단
(cd $R && "${E[@]}" V75_AUDIO=on V75_MODEL=fake ../../../venv/bin/python -c "
import v75_eval_runtime as rt
rt.install_guards()
import src.retrieval.work_name_resolution as wnr
try:
    wnr.VertexWorkNameResolver(client_factory=lambda: None).resolve('x', timeout_seconds=1.0)
    print('NOT BLOCKED')
except rt.RunAborted:
    print('blocked', rt.WORK_RESOLVER_CALLS)
" > $T/f1.log 2>&1)
ok "F 실제 해석 어댑터 호출 → RunAborted(기록 1)" "grep -q \"blocked \\['VertexWorkNameResolver'\\]\" $T/f1.log"

# ---------- 결과 폴더 불변(이 검사가 쓴 곳은 임시 폴더뿐)
snapshot > $T/result_dir_after.txt
ok "결과 폴더 파일 목록·크기·수정 시각 검사 전후 같음" "diff -q $T/result_dir_before.txt $T/result_dir_after.txt > /dev/null"

echo "PASS $PASS FAIL $FAIL (임시 폴더 $T)"
(( FAIL == 0 ))
