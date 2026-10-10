#!/bin/zsh
# v36 — 질의 분석 프롬프트에 생애 단계 지침 한 줄(#35 초안)을 넣은 효과. v06 dev 57 · test 25.
# 실제 실행(10/10 17:50~18:21)은 이 순서를 여러 스크립트로 나눠 돌렸다(첫 시도는 캐시 빌드 도중 코드를 고쳐 지문이 섞여 폐기,
# 두 번째는 zsh에서 "$EX"가 단어 분리되지 않아 --exclude-query-ids를 못 읽음). 아래가 그 순서를 하나로 합친 것이다.
# - new : 새 프롬프트(지침 한 줄 추가, prompt_sha 341d71f166d4) 캐시 — 지침이 든 커밋 f08548f의 worktree에서 빌드.
#         이 브랜치의 현재 트리는 57fc9f5에서 지침을 뺐으므로(prompt_sha 08f0b3bc4c37) 여기서 --force로 빌드하면 new를 옛 프롬프트로 덮어쓴다(리뷰)
# - ctrl: 옛 프롬프트(main) 캐시 — main을 체크아웃한 worktree에서 같은 날 빌드(분석만 하므로 데이터 불필요, .env 복사)
# - old : 9/22 기준 캐시(analysis_cache_v06_{split}.json) 그대로
# 같은 프롬프트로 한 번 더 만든 ctrl2·new2는 ctrl·new와 0건 차이라 지웠다(cache_diff.py로 확인).
# dev는 q115 제외(새 프롬프트·옛 프롬프트 모두 오늘은 모달리티 검증 3회 실패 → 폴백, §2-6의 'background' 오탐).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v36_life_stage_prompt
export SEARCH_REFERENCE_YEAR=2026 RERANKER_BACKEND=gemini_listwise     # 검색 경로는 .env의 GEMINI_RETRIEVAL_BACKEND=api_key
Q=experiments/reranking/eval_queries_v06.csv
W=../vague-finder-main-ctrl   # git worktree add --detach $W origin/main && cp .env $W/.env
WN=../vague-finder-v36-new    # git worktree add --detach $WN f08548f && cp .env $WN/.env   (프롬프트 지침이 든 트리)
for split in dev test; do
  (cd $WN && ../vague-finder/venv/bin/python -m src.retrieval.build_analysis_cache --input $Q --split $split --force --output ../vague-finder/$OUT/analysis_cache_v06_${split}_new.json)
  (cd $W && ../vague-finder/venv/bin/python -m src.retrieval.build_analysis_cache --input $Q --split $split --force --output ../vague-finder/$OUT/analysis_cache_v06_${split}_ctrl.json)
done
for split in dev test; do
  EX=(); [ "$split" = dev ] && EX=(--exclude-query-ids q115)
  for tag in old ctrl new; do
    cache=experiments/reranking/analysis_cache_v06_${split}.json; [ "$tag" != old ] && cache=$OUT/analysis_cache_v06_${split}_${tag}.json
    venv/bin/python -m src.retrieval.evaluate_search_accuracy --input $Q --split $split --analysis-cache $cache "${EX[@]}" --output-dir $OUT/${tag}_${split}
  done
done
venv/bin/python $OUT/compare.py
