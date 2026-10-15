#!/bin/zsh
# v50 리랭킹 포함 측정 — 완성 Context 색인 복사본 · Vertex 분석 캐시(v49) 고정 · Gemini listwise(Vertex) 리랭킹.
#   CTX=0 (A, Context 경로 끔) | CTX=0.5 (B, 현행 기본값)
#   REP=1|2        : 회차. 교차 실행 A1 → B1 → A2 → B2는 run_ab.sh
#   COPY=a|b       : 색인 복사본
#   QUERY_IDS="v06:q203,t071 v10:t071" 형식이 아니라 세트별 환경변수 IDS_v06 · IDS_v09 · IDS_v10 — 주면 스모크(smoke_ctx*)
#   SETS="v06 v09 v10"
# 리랭커 설정은 v49와 같다: 1패스 · Search 끔 · 희소 사실 검증 끔 · 요청 제한 20초 · 예산 30초 · 후보 30(코드 기본값).
# 사용량 탭(usage_tap.py)이 호출마다 토큰·지연·model_version을 usage_*.jsonl에 적는다(프롬프트·응답 본문 없음).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v50_vertex_context
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
CTX=${CTX:-0.5}
REP=${REP:-1}
COPY=${COPY:-a}
SETS=(${=SETS:-v06 v09 v10})
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export CONTEXT_WEIGHT=$CTX AUDIO_FUSION_SCALE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset RERANKER_ENABLED CLAUDE_EFFORT CLAUDE_API
smoke=0
for set in $SETS; do
  [[ -n "${(P)${:-IDS_$set}:-}" ]] && smoke=1
done
if (( smoke )); then name=smoke_ctx${CTX}; else name=rerank_ctx${CTX}_rep$REP; fi
for set in $SETS; do
  ids=()
  idvar="IDS_$set"
  [[ -n "${(P)idvar:-}" ]] && ids=(--query-ids ${(P)idvar})
  GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl GEMINI_USAGE_TAG=$name \
  $PY $OUT/usage_tap.py src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_vertex.json $ids \
    --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  # 실행 조건 기록 — runinfo에는 Context 스위치·Context 컬렉션 세대가 없다
  { echo "CONTEXT_WEIGHT=$CONTEXT_WEIGHT"; echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
    echo "git_sha=$(git rev-parse --short HEAD)"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
    echo "context_build_id=$(python3 -c "import json;print(json.load(open('artifacts/vector_db/context_qdrant/dev/manifest.json'))['context_build_id'][:16])")"
  } > $OUT/$name/conditions_$set.txt
done
