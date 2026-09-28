# v0.8 — 부분 재측정 (5건)

**전체 측정이 아니다.** v0.7에서 후보@30 밖으로 이탈한 신규 질의 5건
(`c604 c701 c702 c705 c707`)에만 맞는 단서를 보강해 다시 잰 것이다.
`search_eval_dev_summary.csv`의 지표는 이 5건만의 평균이므로 기준선으로 읽으면 안 된다.

맥락과 결과 해석은 [`../results_v07/RUN_INFO.md`](../results_v07/RUN_INFO.md) 참조.

```bash
venv/bin/python -m src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_v05.csv --split dev \
  --query-ids c604,c701,c702,c705,c707 \
  --top-k 10 --candidate-k 30 --output-dir experiments/reranking/results_v08
```
