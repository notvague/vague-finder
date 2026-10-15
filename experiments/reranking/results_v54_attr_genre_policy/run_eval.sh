#!/bin/zsh
# v54 속성 조건 보조 후보 — 장르 축 요청 단위 정책(ATTR_FILTER_GENRE_POLICY) 비교. v53 run_eval.sh에 POLICY 스위치를 더했다.
#   ATTR=0 (A, 끔) | ATTR=0.5 (켬, ATTR_FILTER_K=30)      GATE=era_plus(기본)
#   GENRE=1 (장르 포함) | GENRE=0 (C, 장르 축 제외 → _nogenre)
#   POLICY=always (B, 기본) | POLICY=conditional (D → _condgenre)
#   RERANK=0 후보 단계(Gemini 호출 없음) → cand_*/ · RERANK=1 → rerank_*_rep$REP/
#   COPY=a|b|c 색인 복사본(v50의 artifacts/qdrant_eval_* — 서비스 색인 복사, Context 포함, 심볼릭 링크)
#   SETS="v06 v09 v10"  IDS_v06=... 를 주면 스모크(smoke_*/)
# 분석 캐시는 v49 Vertex 캐시(v50~v53과 동일). Context 켬(CONTEXT_WEIGHT 기본 0.5). 리랭커 설정 v49~v53과 동일.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
ATTR=${ATTR:-0.5}
REP=${REP:-1}
COPY=${COPY:-a}
RERANK=${RERANK:-1}
GATE=${GATE:-era_plus}
GENRE=${GENRE:-1}
POLICY=${POLICY:-always}
SETS=(${=SETS:-v06 v09 v10})
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export ATTR_FILTER_WEIGHT=$ATTR ATTR_FILTER_GATE=$GATE ATTR_FILTER_K=30 ATTR_FILTER_USE_GENRE=$GENRE ATTR_FILTER_GENRE_POLICY=$POLICY AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API CONTEXT_WEIGHT
(( RERANK )) || export RERANKER_ENABLED=false
smoke=0
for set in $SETS; do idvar="IDS_$set"; [[ -n "${(P)idvar:-}" ]] && smoke=1; done
suffix=""; [[ $GATE == era_plus ]] && suffix=_eraplus
[[ $GENRE == 0 ]] && suffix=${suffix}_nogenre
[[ $POLICY == conditional ]] && suffix=${suffix}_condgenre
if (( smoke )); then name=smoke_attr$ATTR$suffix
elif (( RERANK )); then name=rerank_attr${ATTR}${suffix}_rep$REP
else name=cand_attr$ATTR$suffix; fi
for set in $SETS; do
  ids=(); idvar="IDS_$set"; [[ -n "${(P)idvar:-}" ]] && ids=(--query-ids ${(P)idvar})
  mkdir -p $OUT/$name
  echo "start $name $set $(date +%T)" >> $OUT/progress.txt
  GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl GEMINI_USAGE_TAG=$name \
  $PY $OUT/usage_tap.py src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_vertex.json $ids \
    --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  { echo "ATTR_FILTER_WEIGHT=$ATTR_FILTER_WEIGHT"; echo "ATTR_FILTER_K=$ATTR_FILTER_K"; echo "ATTR_FILTER_GATE=$ATTR_FILTER_GATE"
    echo "ATTR_FILTER_USE_GENRE=$ATTR_FILTER_USE_GENRE"; echo "ATTR_FILTER_GENRE_POLICY=$ATTR_FILTER_GENRE_POLICY"
    echo "CONTEXT_WEIGHT=${CONTEXT_WEIGHT:-0.5(default)}"
    echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
    echo "git_sha=$(git rev-parse --short HEAD) (+ uncommitted v54 changes)"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
  } > $OUT/$name/conditions_$set.txt
done
