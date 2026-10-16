#!/bin/zsh
# v55 — 봉인 v10 test 42건의 질의 분석 캐시를 Vertex 분석기로 **한 번** 만든다(v49 build_caches.sh와 같은 조건, split만 test).
# 세 설정(A·B·D)이 같은 캐시를 쓴다. 내용을 보고 프롬프트·정책을 고치지 않는다. 생성 비용은 usage_build_cache_v10_test.jsonl에 기록.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v55_attr_final_validation
PY=${PY:-venv/bin/python}
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset CLAUDE_EFFORT CLAUDE_API GEMINI_API_KEY
[[ -f $OUT/analysis_cache_v10_test_vertex.json ]] && { echo "이미 있음 — 다시 만들지 않는다: $OUT/analysis_cache_v10_test_vertex.json" >&2; exit 2; }
echo "start build_cache v10 test $(date +%T)" >> $OUT/progress.txt
GEMINI_USAGE_TAP=$OUT/usage_build_cache_v10_test.jsonl GEMINI_USAGE_TAG=build_cache \
$PY $OUT/usage_tap.py src.retrieval.build_analysis_cache --input experiments/reranking/eval_queries_v10.csv --split test \
  --output $OUT/analysis_cache_v10_test_vertex.json > $OUT/build_cache_v10_test.log 2>&1
rc=$?
echo "done build_cache v10 test exit=$rc $(date +%T)" >> $OUT/progress.txt
exit $rc
