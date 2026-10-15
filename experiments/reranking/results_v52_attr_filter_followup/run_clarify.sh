#!/bin/zsh
# v52 재질문 2턴 OFF/ON — 속성 조건 보조 후보(ATTR_FILTER_WEIGHT)를 끈 A와 켠 B를 같은 조건에서 잰다.
#   ATTR=0 (A, 끔 = v50 B와 같은 코드 경로) | ATTR=0.5 (B, 켬 · ATTR_FILTER_GATE=era_plus · ATTR_FILTER_K=30)
#   REP=1|2  COPY=a|b(색인 복사본, v50의 artifacts/qdrant_eval_* — 서비스 색인 복사, Context 포함)
#   SET=v06(기본 — 재질문 평가기·정본 라벨이 지원하는 dev 57) | v09(--query-set v09 + v09 라벨)
#   POLICIES=flow(기본 — 최초 검색 + 서비스 2턴 흐름 flow:*만) | all
#   IDS="q104,q107" 을 주면 스모크(smoke_clarify_attr$ATTR/)
# Context 켬(CONTEXT_WEIGHT 기본 0.5) · 분석 캐시는 v49 Vertex 캐시(v50·v51과 동일) · 리랭커 설정 v49~v51과 동일.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v52_attr_filter_followup
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
ATTR=${ATTR:-0}
REP=${REP:-1}
COPY=${COPY:-a}
SET=${SET:-v06}
POLICIES=${POLICIES:-flow}
IDS=${IDS:-}
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export ATTR_FILTER_WEIGHT=$ATTR ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API CONTEXT_WEIGHT
if [[ -n "$IDS" ]]; then name=smoke_clarify_attr$ATTR; ids=(--query-ids "$IDS")
else name=clarify_attr${ATTR}_rep$REP; ids=(); fi
setopt=()
[[ $SET == v09 ]] && setopt=(--query-set v09 --labels experiments/reranking/eval_queries_v09.csv)
mkdir -p $OUT/$name
GEMINI_USAGE_TAP=$OUT/usage_${name}_$SET.jsonl GEMINI_USAGE_TAG=$name \
$PY $OUT/usage_tap.py src.retrieval.evaluate_clarification --split dev $setopt $ids \
  --analysis-cache $CACHES/analysis_cache_${SET}_dev_vertex.json --policies $POLICIES \
  --output-dir $OUT/$name/${SET}_dev > $OUT/${name}_$SET.log 2>&1
echo "done $name $SET exit=$? $(date +%T)" >> $OUT/progress.txt
{ echo "ATTR_FILTER_WEIGHT=$ATTR_FILTER_WEIGHT"; echo "ATTR_FILTER_K=$ATTR_FILTER_K"; echo "ATTR_FILTER_GATE=$ATTR_FILTER_GATE"
  echo "CONTEXT_WEIGHT=${CONTEXT_WEIGHT:-0.5(default)}"; echo "POLICIES=$POLICIES"; echo "QDRANT_PATH=$QDRANT_PATH"
  echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "git_sha=$(git rev-parse --short HEAD) (+ uncommitted follow-up changes)"
  ls $QDRANT_PATH/collection | sed 's/^/collection=/'
} > $OUT/$name/conditions_$SET.txt
