#!/bin/zsh
# v55 최종 검증 — 봉인 v10 test 42건을 A → B → D 각 1회(순차 단일 프로세스). v54 run_eval.sh에서 split·캐시·출력만 바꿨다.
#   CONFIG=A  ATTR_FILTER_WEIGHT=0   (ATTR_FILTER_USE_GENRE=1 · ATTR_FILTER_GENRE_POLICY=always — 현행 운영 기본값)
#   CONFIG=B  ATTR_FILTER_WEIGHT=0.5 ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 USE_GENRE=1 GENRE_POLICY=always
#   CONFIG=D  ATTR_FILTER_WEIGHT=0.5 ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 USE_GENRE=1 GENRE_POLICY=conditional
#   SPLIT=test(기본) · SET=v10(기본) · RERANK=1(기본) · IDS=... 를 주면 스모크(smoke_<CONFIG>/, dev 전용)
#   TAG=<이름> 출력 폴더 이름을 바꿀 때(복구 실행 등). 기본은 <CONFIG>_<SET>_<SPLIT>
# 색인 복사본 artifacts/qdrant_eval_v55(서비스 색인 복사, Context 포함, 지문 c39aa7eb08fb = v50 복사본). 분석 캐시는 이 폴더의
# analysis_cache_v10_test_vertex.json(build_cache.sh로 1회 생성) — dev 스모크는 v49 dev 캐시. 리랭커 설정은 v49~v54와 동일.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v55_attr_final_validation
PY=${PY:-venv/bin/python}
CONFIG=${CONFIG:?A|B|D}
SPLIT=${SPLIT:-test}
SET=${SET:-v10}
RERANK=${RERANK:-1}
IDS=${IDS:-}
case $CONFIG in
  A) ATTR=0;   POLICY=always ;;
  B) ATTR=0.5; POLICY=always ;;
  D) ATTR=0.5; POLICY=conditional ;;
  *) echo "CONFIG는 A·B·D 중 하나" >&2; exit 2 ;;
esac
if [[ $SPLIT == test ]]; then
  CACHE=$OUT/analysis_cache_${SET}_test_vertex.json
  [[ -n $IDS ]] && { echo "봉인 test는 부분 실행하지 않는다(IDS는 dev 스모크 전용)" >&2; exit 2; }
else
  CACHE=experiments/reranking/results_v49_vertex_baseline/analysis_cache_${SET}_dev_vertex.json
fi
[[ -f $CACHE ]] || { echo "분석 캐시 없음: $CACHE" >&2; exit 2; }
export QDRANT_PATH=artifacts/qdrant_eval_v55
export ATTR_FILTER_WEIGHT=$ATTR ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 ATTR_FILTER_USE_GENRE=1 ATTR_FILTER_GENRE_POLICY=$POLICY AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API CONTEXT_WEIGHT GEMINI_API_KEY
(( RERANK )) || export RERANKER_ENABLED=false
ids=(); [[ -n $IDS ]] && ids=(--query-ids $IDS)
name=${TAG:-${CONFIG}_${SET}_${SPLIT}}
[[ -n $IDS ]] && name=smoke_${name}
mkdir -p $OUT/$name
echo "start $name $(date +%T)" >> $OUT/progress.txt
GEMINI_USAGE_TAP=$OUT/usage_${name}.jsonl GEMINI_USAGE_TAG=$name \
$PY $OUT/usage_tap.py src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_$SET.csv --split $SPLIT \
  --analysis-cache $CACHE $ids \
  --output-dir $OUT/$name > $OUT/${name}.log 2>&1
rc=$?
echo "done $name exit=$rc $(date +%T)" >> $OUT/progress.txt
{ echo "CONFIG=$CONFIG"; echo "ATTR_FILTER_WEIGHT=$ATTR_FILTER_WEIGHT"; echo "ATTR_FILTER_K=$ATTR_FILTER_K"; echo "ATTR_FILTER_GATE=$ATTR_FILTER_GATE"
  echo "ATTR_FILTER_USE_GENRE=$ATTR_FILTER_USE_GENRE"; echo "ATTR_FILTER_GENRE_POLICY=$ATTR_FILTER_GENRE_POLICY"
  echo "CONTEXT_WEIGHT=${CONTEXT_WEIGHT:-0.5(default)}"; echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"
  echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"; echo "ANALYSIS_CACHE=$CACHE"; echo "SPLIT=$SPLIT"; echo "git_sha=$(git rev-parse --short HEAD)"
  echo "git_dirty_src=$(git status --short src | wc -l | tr -d ' ')"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
} > $OUT/$name/conditions.txt
exit $rc
