# v0.9 — 부분 재측정 (2건)

**전체 측정이 아니다.** v0.8에서도 목표 구간에 들어오지 못한 `c604`·`c702`를
한 번 더 고쳐 잰 것이다. 둘 다 실패해 `label_status=unreachable`로 집계에서 제외했다.
`search_eval_dev_summary.csv`의 지표는 이 2건만의 평균이므로 기준선이 아니다.

맥락과 결과 해석은 [`../results_v07/RUN_INFO.md`](../results_v07/RUN_INFO.md) 참조.

```bash
venv/bin/python -m src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_v05.csv --split dev \
  --query-ids c604,c702 \
  --top-k 10 --candidate-k 30 --output-dir experiments/reranking/results_v09
```
