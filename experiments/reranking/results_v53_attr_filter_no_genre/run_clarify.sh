#!/bin/zsh
# v53 재질문 2턴 — 장르 포함(B, GENRE=1)과 장르 제외(C, GENRE=0)를 같은 조건에서 잰다. v52 run_clarify.sh에 GENRE 스위치를 더했다.
#   ATTR=0.5 (기본) | 0   GENRE=1|0   REP=1|2   COPY=a|b|c   SET=v06|v09   POLICIES=flow(기본)|all
#   IDS="q112,q115" 을 주면 그 질의만(영향 사례 비교 — 이름 clarify_sub_*). RERANK=0이면 리랭킹 끔(모델 호출 없음, 이름 *_norerank)
# Context 켬 · 분석 캐시 v49 Vertex · 리랭커 설정 v49~v52와 동일.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v53_attr_filter_no_genre
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
ATTR=${ATTR:-0.5}
GENRE=${GENRE:-1}
REP=${REP:-1}
COPY=${COPY:-a}
SET=${SET:-v06}
POLICIES=${POLICIES:-flow}
IDS=${IDS:-}
RERANK=${RERANK:-1}
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export ATTR_FILTER_WEIGHT=$ATTR ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 ATTR_FILTER_USE_GENRE=$GENRE AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API CONTEXT_WEIGHT
(( RERANK )) || export RERANKER_ENABLED=false
suffix=""; [[ $GENRE == 0 ]] && suffix=_nogenre
(( RERANK )) || suffix=${suffix}_norerank
if [[ -n "$IDS" ]]; then name=clarify_sub_attr${ATTR}${suffix}_rep$REP; ids=(--query-ids "$IDS")
else name=clarify_attr${ATTR}${suffix}_rep$REP; ids=(); fi
setopt=()
[[ $SET == v09 ]] && setopt=(--query-set v09 --labels experiments/reranking/eval_queries_v09.csv)
mkdir -p $OUT/$name
GEMINI_USAGE_TAP=$OUT/usage_${name}_$SET.jsonl GEMINI_USAGE_TAG=$name \
$PY $OUT/usage_tap.py src.retrieval.evaluate_clarification --split dev $setopt $ids \
  --analysis-cache $CACHES/analysis_cache_${SET}_dev_vertex.json --policies $POLICIES \
  --output-dir $OUT/$name/${SET}_dev > $OUT/${name}_$SET.log 2>&1
echo "done $name $SET exit=$? $(date +%T)" >> $OUT/progress.txt
{ echo "ATTR_FILTER_WEIGHT=$ATTR_FILTER_WEIGHT"; echo "ATTR_FILTER_K=$ATTR_FILTER_K"; echo "ATTR_FILTER_GATE=$ATTR_FILTER_GATE"
  echo "ATTR_FILTER_USE_GENRE=$ATTR_FILTER_USE_GENRE"; echo "CONTEXT_WEIGHT=${CONTEXT_WEIGHT:-0.5(default)}"; echo "POLICIES=$POLICIES"
  echo "IDS=$IDS"; echo "QDRANT_PATH=$QDRANT_PATH"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
  echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "git_sha=$(git rev-parse --short HEAD) (+ uncommitted v53 changes)"
  ls $QDRANT_PATH/collection | sed 's/^/collection=/'
} > $OUT/$name/conditions_$SET.txt
