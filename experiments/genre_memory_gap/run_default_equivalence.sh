#!/bin/zsh
# 기본값(두 배율 1.0)에서 이 브랜치가 기반 커밋과 같은 후보·점수를 내는지 실제 색인으로 확인한다 (리랭킹 없음).
#   같은 분석 캐시·같은 색인으로 dump_candidate_scores.py를 기반 체크아웃과 이 브랜치에서 각각 돌려
#   산출 CSV(질의마다 최초 검색 · 거절 뒤 · 거절 + 성별 답 · 거절 + 장르 답의 후보 30곡 ID와 점수)를 바이트로 견준다.
#   BASE=<기반 커밋을 체크아웃한 폴더> zsh experiments/genre_memory_gap/run_default_equivalence.sh
set -u
HERE="$(git rev-parse --show-toplevel)"
BASE=${BASE:?기반 체크아웃 경로를 BASE로 준다}
cd "$HERE"
source experiments/genre_memory_gap/env.sh
case $QDRANT_PATH in /*) ;; *) QDRANT_PATH="$HERE/$QDRANT_PATH" ;; esac
case $PY in /*) ;; *) PY="$HERE/$PY" ;; esac          # 기반 체크아웃을 cwd로 두고도 같은 인터프리터를 쓴다
export QDRANT_PATH BM25_PARAMS_PATH="$HERE/artifacts/bm25_params.json" PYTHONDONTWRITEBYTECODE=1
unset GENRE_MATCH_SCALE GENRE_ANSWER_SCALE
OUT="$HERE/experiments/genre_memory_gap/${RESULTS:-results_v01}/default_equivalence"
V46=experiments/reranking/results_v46_main_after_context
mkdir -p "$OUT"
: > "$OUT/result.txt"
[[ -e "$BASE/.env" ]] || ln -s "$HERE/.env" "$BASE/.env"
for set in ${=SETS:-v06 v09 v10}; do
  for side in base branch; do
    root=$HERE; [[ $side == base ]] && root=$BASE
    (cd "$root" && $PY "$HERE/experiments/genre_memory_gap/dump_candidate_scores.py" \
      --input "$HERE/experiments/reranking/eval_queries_$set.csv" \
      --analysis-cache "$HERE/$V46/analysis_cache_${set}_dev_ctx.json" \
      --output "$OUT/candidate_scores_${set}_$side.csv") > "$OUT/${set}_$side.log" 2>&1
    echo "done $set $side exit=$? $(date +%T)" >> "$OUT/progress.txt"
  done
  if cmp -s "$OUT/candidate_scores_${set}_base.csv" "$OUT/candidate_scores_${set}_branch.csv"; then
    echo "$set: 같음 ($(($(wc -l < "$OUT/candidate_scores_${set}_branch.csv") - 1))회 검색, sha256 $(shasum -a 256 "$OUT/candidate_scores_${set}_branch.csv" | cut -c1-12))" >> "$OUT/result.txt"
  else
    echo "$set: 다름" >> "$OUT/result.txt"
  fi
done
echo "기반 $(cd "$BASE" && git rev-parse --short HEAD) · 브랜치 $(git rev-parse --short HEAD)+작업 트리 · 색인 $(basename "$QDRANT_PATH")" >> "$OUT/result.txt"
cat "$OUT/result.txt"
