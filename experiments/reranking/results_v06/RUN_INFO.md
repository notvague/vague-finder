# v0.6 측정 정보 (단계 2 검증용)

**공식 기준선이 아니다.** v0.5(2026-08-19)와 코드·질의 세트가 모두 다르므로
지표를 그대로 비교하면 안 된다. 비교는 `query_set=v04` 부분집합으로만 한다.
이 측정의 목적은 단계 2에서 추가한 신규 질의 26개가 실제로 어려운지 확인하는 것이었다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 커밋 | `6052c99` + 단계 2 작업 (queries.json v0.5 통합, 미커밋) |
| 측정 일시 | 2026-09-15 09:50 KST |
| 코퍼스 | 905곡 (Pinecone namespace `dev`) |
| 질의 세트 | `experiments/reranking/eval_queries_v05.csv` split=dev (62건) |
| 구성 | v04 37 + modality_v1 7 + clarify_v1 18 |
| top_k / candidate_k | 10 / 30 |
| `RERANKER_SPREAD_REF` | `0.02` (v0.5와 동일) |

v0.5 이후 머지된 변경: `#56`(멀티모달 안정성·후보 회수), `#59`(재질문 상태 왕복·
앨범커버 오판정), `#60`(나무위키 개요 보도자료 제거), `#61`(Reject-only·후보 폭).

임베더 동시 로드 크래시(v0.5 RUN_INFO 이슈 3)는 여전히 미수정이며, 이번에도
평가 전 메인 스레드에서 KoE5/SigLIP2/CLAP/리랭커를 `load()`하는 방식으로 우회했다.

## 결과

### 전체 (62건)

| 지표 | baseline | rerank |
|---|---|---|
| Hit@1 | 0.548 | 0.581 |
| Hit@5 | 0.871 | 0.887 |
| Hit@10 | 0.919 | 0.919 |
| MRR@10 | 0.674 | 0.712 |
| nDCG@10 | 0.734 | 0.764 |
| Candidate Recall@30 | 0.984 | — |

### v04 부분집합 (37건) — v0.5와 비교 가능한 유일한 축

| 지표 | v0.5 dev | v0.6 dev | 변화 |
|---|---|---|---|
| Hit@1 | 0.541 | 0.514 | −0.027 (1곡) |
| Hit@5 | 0.784 | 0.865 | **+0.081 (3곡)** |
| Hit@10 | 0.865 | 0.892 | +0.027 (1곡) |

Top-10 밖으로 나가 있던 q110·q218이 각각 2위로 진입했고, q316이 9위 → Top-10 밖으로
떨어졌다. 순손실이 아니다.

### 정답 위치

| 세트 | n | 1~5위 | 6~10위 | Top-10 밖 |
|---|---|---|---|---|
| v04 | 37 | 32 | 1 | 4 |
| modality_v1 | 7 | 5 | 1 | 1 |
| clarify_v1 | 18 | 18 | 0 | **0** |

## 판정 — 신규 질의는 목표 미달

단계 2의 목표는 "Top-10 밖에 정답이 있는 질의 20~30개 추가"였다.
**18개 측정 중 Top-10 밖은 0개**이고, 14개가 1위, 후보@30에서도 12개가 1위였다.
재질문이 개입할 여지가 전혀 없다.

원인은 질의를 쓴 방식이다. 서로 다른 축 2~3개를 겹치면 좁혀질 것이라고 봤는데,
**그 축들이 전부 색인된 메타데이터에 문자 그대로 들어 있었다.** 905곡 코퍼스에서
`sound_tags` + `vocal_gender` + `type`의 교집합은 거의 항상 곡 하나로 확정된다.
("색소폰" → sound_tags에 색소폰, "여자 그룹" → vocal_gender·type, "국악기" → 국악)

실제로 어려운 기존 질의들은 반대 성질을 갖는다.

| 질의 | 후보 순위 | 왜 어려운가 |
|---|---|---|
| q203 | 23위 | "짱구 애니메이션" — 코퍼스에 없는 외부 맥락 |
| q101 | 24위 | "싸이월드" — 문화적 맥락이 메타데이터에 없음 |
| q316 | 24위 | 오정보 (남자 아이돌이라 했지만 실제는 여성) |
| q115 | 14위 | "피아노 하나뿐인데 오케스트라가 뒤에서" — 미묘한 편성 |

다음 시도에서 지킬 것:
1. `sound_tags`·`genre`·`vocal_gender` 어휘를 **그대로 쓰지 않는다.** 우회 묘사로 바꾼다.
2. 코퍼스에 없는 외부 맥락(애니·게임·밈·광고)을 단서로 쓴다. 앨범명에 드라마가
   들어 있는 OST는 반대로 쉬워진다 — c101·c106·c107이 그래서 상위권이었다.
3. 축을 3개 겹치지 않는다. 교집합이 곧 정답 확정이다.
4. `misinformation` tier를 적극 쓴다. 기존 세트에서 가장 확실하게 어려운 유형이다.

## 부수 확인

- **m402가 후보@30 밖(recall 0.0)** — `label_status=pending_cover`인 부분 라벨이다.
  표지 확인 전이라 정답 자체가 의심스러우므로 라벨 문제와 검색 문제를 분리해야 한다.
- **리랭커 악화 2건**: q219 (1→3위, confidence 1.0 — v0.5의 음차 가사 문제 재현),
  m103 (1→2위, confidence 0.04).
- **q305는 test split이라 이번에 측정되지 않았다.** 계획서 4순위(리랭커 회귀)는
  다음 test 측정에서 확인한다.

## 재현

```bash
venv/bin/python -m src.eval.export_csv          # queries.json → eval_queries_v05.csv
# 임베더를 메인 스레드에서 미리 load() 한 뒤:
venv/bin/python -m src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_v05.csv \
  --split dev --top-k 10 --candidate-k 30 \
  --output-dir experiments/reranking/results_v06
venv/bin/python -m src.eval.recompute_baseline \
  --detail experiments/reranking/results_v06/search_eval_dev_detail.csv \
  --export-ranks experiments/reranking/results_v06/search_eval_dev_ranks.csv
```

`search_eval_dev_ranks.csv`(5KB)는 커밋 대상이므로 detail(204KB) 없이도 질의별
순위와 split별 재집계를 재현할 수 있다.
