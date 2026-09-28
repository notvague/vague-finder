# v0.5 기준선 측정 정보

Vague-Finder 검색 성능 공식 기준선. 이후 모든 성능 비교는 이 실행을 기준으로 한다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 커밋 | `0b9654b315070afdf1d217156b7ad51a7f98cd17` |
| 커밋 제목 | `[Fix] 머지 과정에서 끊긴 가사 exact 보호 경로 복구 - 황찬혁 (#50)` |
| 브랜치 | `main` (워킹트리 clean) |
| 측정 일시 | 2026-08-19 16:19 KST |
| 코퍼스 | 905곡 (Pinecone namespace `dev`) |
| 인덱스 벡터 수 | text 905 / image 905 / audio 905 |
| 질의 세트 | `experiments/reranking/eval_queries_v04.csv` (53건, split=dev) |
| top_k / candidate_k | 10 / 30 |
| `RERANKER_SPREAD_REF` | `0.02` |

### 포함된 PR

이 커밋 시점에 main에 머지되어 있던 변경:

- `#47` [Refactor] 리랭킹 성능 개선 — 이연우 (spread 기반 적응형 가중치, `type` 필드 연결)
- `#49` [Feat] 검색 후보 회수율 및 실행 환경 안정화 — 최정현 (가사 exact/음차 검색, 제목 구조 검색, 메타데이터 단서)
- `#50` [Fix] 머지 과정에서 끊긴 가사 exact 보호 경로 복구 — 황찬혁

`#46`, `#48`은 `#49`에 커밋이 포함되어 close 처리됨.

## 결과

| 지표 | baseline | rerank | delta |
|---|---|---|---|
| Hit@1 | 0.5283 | 0.5660 | +0.0377 |
| Hit@5 | 0.7736 | 0.7925 | +0.0189 |
| Hit@10 | 0.8491 | 0.8491 | 0.0 |
| MRR@10 | 0.6325 | 0.6514 | +0.0189 |
| nDCG@10 | 0.6845 | 0.6987 | +0.0142 |
| Candidate Recall@30 | 0.9811 | — | — |

개선 3 / 동일 48 / 악화 2

- 개선: q208 (2→1위), q212 (2→1위), q215 (6→1위)
- 악화: q219 (1→3위), q305 (2→3위)

## 알려진 이슈

### 1. Candidate Recall 98.1%의 미달 1건은 라벨 오류

후보 30 밖에 남은 유일한 질의는 `q118`이며, 정답 라벨이 잘못되어 있다.

```
q118: "여자가 부르는 피아노 발라드인데 제목이 한자였던 것 같아"
정답 라벨: 30461396 → G-DRAGON 「무제(無題)」 (남성 솔로)
```

`q111`과 정답 ID가 동일하다. 질의는 "여자"를 명시하는데 정답은 남성 솔로이므로
시스템이 구조적으로 맞힐 수 없는 문항이다.

**실질 Candidate Recall은 100%이다.** 라벨 수정은 코퍼스 확대(905 → 1,961곡) 시
기준선을 새로 찍을 때 함께 처리한다.

### 2. 악화 2건은 리랭커 고신뢰 구간에서 발생

q219, q305 모두 `rerank_confidence = 1.0`이라 저신뢰 보호 로직이 개입하지 않는다.
q219는 음차 가사(`phonetic`) 질의인데, `rerank_preserving_exact_lyrics()`는
`lyric_match_type == "exact"`만 보호 대상으로 삼는다.

### 3. 임베더 로딩 경쟁 조건 (측정 시 우회함)

`KoE5Embedder.load()` / `SigLIP2Embedder` / `CLAPAudioEmbedder`에 로드 락이 없어,
`SearchRouter`가 여러 검색 경로를 ThreadPoolExecutor로 동시 실행할 때
같은 모델을 동시에 로드하면 크래시한다.

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

이 측정은 메인 스레드에서 모델 4종을 사전 로드한 뒤 평가를 실행하는 방식으로 우회했다.
`MusicReranker`에는 `_load_lock`이 있으나 임베더 3종에는 없다. 실서비스에서도
동시 요청 시 재현될 수 있어 별도 수정이 필요하다.

## 재현 방법

```bash
git checkout 0b9654b
```

`.env`에 `RERANKER_SPREAD_REF=0.02` 설정 후:

```bash
python -m src.retrieval.evaluate_search_accuracy \
  --input experiments/reranking/eval_queries_v04.csv \
  --split dev --top-k 10 --candidate-k 30 \
  --output-dir experiments/reranking/results_v05
```

이슈 3으로 인해 간헐적으로 모델 로딩 크래시가 발생할 수 있다. 재시도하거나,
평가 전에 `get_text_embedder().load()` 등으로 모델을 미리 로드한다.

## 이전 기준선과의 관계

`results_v04`(2026-08-13)는 **비교 불가**하므로 폐기한다. 측정 이후
`#47`의 중복 boost 블록 제거와 `#49`의 신규 검색 경로 추가로 retrieval 결과 자체가
변경되었다. (예: q215 candidate 순위 10위 → 6위)

참고로 v0.4 대비 변화는 다음과 같으나, 코드가 다르므로 참고용으로만 본다.

| 지표 | v0.4 | v0.5 |
|---|---|---|
| Candidate Recall@30 | 69.8% | 98.1% |
| Hit@1 | 35.8% | 56.6% |
| Hit@10 | 64.2% | 84.9% |
