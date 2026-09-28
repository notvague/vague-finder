# 답변 보너스 배수 스윕 — ×8

세 디렉터리(`sweep_x3` · `sweep_x5` · `sweep_x8`)가 한 세트다.
배경·결과·선택 근거는 [`../results_clarify_sweep_x3/RUN_INFO.md`](../results_clarify_sweep_x3/RUN_INFO.md)에 정리돼 있다.

```bash
venv/bin/python -m src.retrieval.evaluate_clarification --split dev \
  --query-ids c607,c701,c705,c707,m402,q101,q115,q203,q316 \
  --answer-multiplier 8 \
  --output-dir experiments/reranking/results_clarify_sweep_x8
```
