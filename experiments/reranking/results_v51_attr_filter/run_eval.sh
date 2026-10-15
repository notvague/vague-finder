#!/bin/zsh
# v51 속성 조건 보조 후보(ATTR_FILTER_WEIGHT) — v50 B(Context 켬 · 현행)와 같은 조건에 보조 경로만 더한 A/B.
#   ATTR=0 (끔 = v50 B와 같은 코드 경로) | ATTR=0.5 (켬, 기본 폭 ATTR_FILTER_K=30). 채택(2026-10-15) 뒤 코드 기본값이 0.5 · era_plus라
#   이 스크립트는 값을 항상 명시한다 — 끔 쪽을 재현하려면 ATTR=0을 준다
#   GATE=two_axes(기본) | era_plus — 여는 조건(ATTR_FILTER_GATE). era_plus면 실행 이름에 _eraplus가 붙는다
#   RERANK=0이면 리랭킹 끔(후보 단계 · Gemini 호출 없음) → cand_attr$ATTR/, 1이면 rerank_attr${ATTR}_rep$REP/
#   COPY=a|b : 색인 복사본(v50의 artifacts/qdrant_eval_* — 서비스 색인 복사, Context 포함)
#   SETS="v06 v09 v10"  IDS_v06=... 를 주면 스모크(smoke_attr$ATTR/)
# 분석 캐시는 v49 Vertex 캐시(v50과 동일). Context 켬(CONTEXT_WEIGHT 기본 0.5). 리랭커 설정 v49·v50과 동일.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v51_attr_filter
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
ATTR=${ATTR:-0.5}
REP=${REP:-1}
COPY=${COPY:-a}
RERANK=${RERANK:-1}
GATE=${GATE:-two_axes}
SETS=(${=SETS:-v06 v09 v10})
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export ATTR_FILTER_WEIGHT=$ATTR ATTR_FILTER_GATE=$GATE AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API CONTEXT_WEIGHT
(( RERANK )) || export RERANKER_ENABLED=false
smoke=0
for set in $SETS; do idvar="IDS_$set"; [[ -n "${(P)idvar:-}" ]] && smoke=1; done
suffix=""; [[ $GATE == era_plus ]] && suffix=_eraplus
if (( smoke )); then name=smoke_attr$ATTR$suffix
elif (( RERANK )); then name=rerank_attr${ATTR}${suffix}_rep$REP
else name=cand_attr$ATTR$suffix; fi
for set in $SETS; do
  ids=(); idvar="IDS_$set"; [[ -n "${(P)idvar:-}" ]] && ids=(--query-ids ${(P)idvar})
  GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl GEMINI_USAGE_TAG=$name \
  $PY $OUT/usage_tap.py src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_vertex.json $ids \
    --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  { echo "ATTR_FILTER_WEIGHT=$ATTR_FILTER_WEIGHT"; echo "ATTR_FILTER_K=${ATTR_FILTER_K:-30(default)}"; echo "ATTR_FILTER_GATE=$ATTR_FILTER_GATE"; echo "CONTEXT_WEIGHT=${CONTEXT_WEIGHT:-0.5(default)}"
    echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
    echo "git_sha=$(git rev-parse --short HEAD)"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
  } > $OUT/$name/conditions_$set.txt
done
