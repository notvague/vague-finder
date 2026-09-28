# v0.11 측정 정보 — 952곡 dev 재측정 (2026-09-22)

**이 측정의 기준선은 v0.11이다.** 질의 분석을 캐시로 고정한 첫 측정이라,
같은 명령을 다시 돌리면 같은 숫자가 나온다.

## 실행

```
venv/bin/python -m src.retrieval.build_analysis_cache \
    --input experiments/reranking/eval_queries_v05.csv --split dev \
    --output experiments/reranking/analysis_cache_dev.json

venv/bin/python -m src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_v05.csv --split dev \
    --analysis-cache experiments/reranking/analysis_cache_dev.json \
    --output-dir experiments/reranking/results_v11
```

## 실행 환경

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-22 |
| 코퍼스 | **952곡** (Qdrant local, namespace `dev`, `point_count`로 확인) |
| 질의 세트 | `eval_queries_v05.csv` split=dev **53건** |
| 질의 분석 | `analysis_cache_dev.json`로 **고정**. 폴백 0건, 조건 drift 없음 |
| 분석 조건 | `gemini-3.1-flash-lite`, temperature 0.0, top_p 1.0, `SEARCH_REFERENCE_YEAR=2026`, prompt_sha `08f0b3bc4c37` |
| BM25 | `artifacts/bm25_params.json` sha256 `4710257067c8` (286,292 B) |
| 리랭커 | Cross-Encoder `dragonkue/bge-reranker-v2-m3-ko`, `rerank_weight=0.9`, `spread_ref=0.02`, `low_conf_top_n=5` |
| top_k / candidate_k | 10 / 30 |

기계가 읽는 전체 기록은 `search_eval_dev_runinfo.json`에 있다.

## 결과

| 지표 | baseline | rerank | delta |
|---|---|---|---|
| Hit@1 | 0.45283 | 0.43396 | −0.01887 |
| Hit@5 | 0.71698 | 0.71698 | 0 |
| Hit@10 | 0.81132 | 0.81132 | 0 |
| Recall@10 | 0.81132 | 0.81132 | 0 |
| MRR@10 | 0.56365 | 0.56554 | +0.00189 |
| nDCG@10 | 0.62267 | 0.62492 | +0.00225 |
| Candidate Recall@30 | 0.98113 | — | — |

리랭킹 결말: `applied=53`. 폴백 0건, 부분 실패 0건 — 53건 전부 모델이 실제로 돌았다.

순위 변화: **개선 3건 / 동일 48건 / 악화 2건**

| 질의 | 변화 |
|---|---|
| q208 | 5위 → 2위 |
| q215 | 5위 → 1위 |
| q310 | 3위 → 2위 |
| q219 | 1위 → 3위 |
| m103 | 1위 → 2위 |

## v0.7(905곡)과의 비교 — 같은 53건으로 정규화

v0.7은 55건을 쟀다. 현재 dev에 없는 `c604`·`c702`를 빼고 **공통 53건**으로 맞춘 값이다.

| | v0.7 (905곡) | v0.11 (952곡) |
|---|---|---|
| 후보@30 안에 정답 | 49/53 | **52/53** |
| 최종 Hit@1 | 23/53 | 23/53 |
| 최종 Hit@5 | 39/53 | 38/53 |
| 최종 Hit@10 | 44/53 | 43/53 |

**후보 회수는 올랐고(49→52) 상위 순위는 한 칸씩 내려갔다.** Hit@5·Hit@10이 각각 1건 차이다.

### 이 비교의 한계

**차이를 코퍼스 증가 하나로 돌릴 수 없다.** v0.7 이후 함께 바뀐 것이 최소 세 가지다.

| 바뀐 것 | v0.7 | v0.11 |
|---|---|---|
| 코퍼스 | 905곡 | 952곡 |
| 벡터 백엔드 | Pinecone | Qdrant (local) |
| 오디오 임베딩 | 기존 | CLAP 시드 고정 후 **전량 재생성** |
| 질의 분석 | 매 실행 새로 분석 | 캐시로 고정 |

각 요인을 분리하려면 하나씩만 바꾼 측정이 필요하다. 이 문서는 그 작업을 하지 않았다.

또한 v0.11 안에서의 리랭킹 전후 비교는 위 표와 **별개**다 —
Hit@1은 24건 → 23건, Hit@5·Hit@10은 그대로다.

## 실행 기록으로 확인한 악화 2건

`search_eval_dev_explain.jsonl`에서 곡별 근거를 바로 읽을 수 있다.

### q219 — 보호받지 못한 phonetic 표면 일치 (1위 → 3위)

```
정답: 가사 구절 일치 경로 1위, 음차 추정 표기가 가사와 표기 정규화 후 일치 ×5.0,
      단서 확신도 0.85 (+0.0697)   ← 최대 기여, 텍스트 경로의 약 7배
      → 리랭킹이 1위에서 3위로 내림
```

가사 보호 규칙(`search_router.py`)은 `lyric_match_type == "exact"`만 보호한다.
이 후보는 `phonetic`이라 **보호 대상이 아니다.**

**0.85는 문자 유사도가 아니다.** `lyrics_exact_search.py`는 정규화 후 부분문자열로
발견되면 분석 단서의 `confidence`를 그대로 점수로 쓴다(fuzzy일 때만 유사도를 곱한다).
이 사례는 음차 변형 표기가 가사와 일치했고, 그 **단서의 확신도**가 0.85였다.
처음에 "일치도 0.85"로 적었고 그대로 잘못 설명했다.

**"가사 원문에 그대로 있음"도 강한 표현이었다.** 비교는 `normalize_lyric_surface()`를
거친다 — NFKC 정규화 + 소문자 변환 뒤 영숫자·한글만 남기므로 대소문자·공백·개행·
구두점이 모두 지워진다. `"I FOUND\nTHE WAY!"`는 `"I found the way"`를 문자열로
포함하지 않지만 정규화하면 둘 다 `"ifoundtheway"`가 되어 일치로 판정된다. 어간과
활용은 바꾸지 않는다. 문구를 **"가사와 표기 정규화 후 일치"**로 고쳤고, 같은 정규화를
쓰는 가사 발췌·요약 규칙 7종도 함께 맞췄다. 스키마 설명과 회귀 테스트도 갱신했다.

### m103 — 현재 합성 계수에서 CE 기여가 검색 기여를 넘었다 (1위 → 2위)

```
             retrieval_score → 정규화      CE 원점수 → 정규화
1위 정일영     0.088851 → 0.985832        0.000683 → 0.838471
2위 Ryu(정답)  0.089254 → 1.000000        0.000251 → 0.305956

유효 가중치 3.651% (신뢰도 0.0406으로 90%에서 축소), CE spread 0.000811
검색 쪽 기여 차 −0.013651  +  CE 쪽 기여 차 +0.019442  =  +0.005791
```

**spread는 약분된다.** 저신뢰 구간(`spread < spread_ref`)에서 CE 기여 차는

```
(rerank_weight × spread / spread_ref) × (CE 원점수 차 / spread)
= rerank_weight / spread_ref × CE 원점수 차
= 45 × CE 원점수 차                    ← spread가 사라진다
```

45 × 0.000432 = +0.019440. 즉 **CE 원점수 차에 곱해지는 계수가 45로 상수**이고,
spread가 작아질수록 정규화가 폭주하는 구조가 아니다. `spread ≥ spread_ref`에서는
계수가 `rerank_weight / spread`로 오히려 줄어드므로, 신뢰도 가중은 이 민감도에
**상한(45)을 두는 역할**을 한다.

따라서 이 사례는 "현재 합성 계수에서 CE 기여 차(+0.0194)가 검색 기여 차(−0.0137)를
넘어섰다"가 정확한 설명이다. 계산 오류가 아니다.

`confidence`는 점수 폭으로 만든 휴리스틱이지 정답 확률이 아니다. 검색 점수가
비슷한 상황은 리랭커가 도움이 될 수 있는 상황이기도 하다. 작은 spread에서
재정렬을 생략하는 것은 **별도의 정책 실험**이며, 이 한 건으로 규칙을 바꾸지 않는다.

## 저장소 간 불일치 — 후속 실험 전에 알아야 할 것

가사 표면 검색은 Qdrant가 아니라 **MongoDB의 `full_lyrics`를 직접 읽는다**
(`lyrics_exact_search.py`). 두 저장소를 세어 보니 어긋나 있다.

| | 곡 수 |
|---|---|
| Qdrant 텍스트 인덱스 (ns `dev`) | **952** |
| MongoDB `songs` | **905** (전량 `full_lyrics` 보유, 합계 717,291자) |
| Qdrant에만 있음 | **53** — 가사 표면 검색·가사 보호 규칙이 닿지 않는다 |
| MongoDB에만 있음 | **6** — 벡터 검색으로 찾을 수 없다 |

**이번 측정은 영향받지 않았다.** dev 정답 51곡 전부가 두 저장소에 있고 `full_lyrics`도
있다. 가사 단서가 있는 질의 중 정답이 Mongo에 없는 경우도 0건이다.

다만 서비스에서는 53곡이 가사 경로로 발견되지 않고, 6곡은 검색되지 않는다.
데이터 파이프라인 쪽에서 맞춰야 한다.

후속 실험 중에는 **Mongo 가사 데이터를 고정**해야 한다. 같은 분석 캐시·같은 Qdrant
인덱스라도 중간에 가사가 바뀌면 exact·phonetic 후보가 달라진다. 지문은
`search_eval_dev_runinfo.json`의 `lyrics_source`에 남는다(문서 수 + full_lyrics 총 문자 수).

## 다음

후속 실험은 **같은 캐시·같은 Qdrant 인덱스·같은 Mongo 가사**에서 세 갈래로 나눈다.

| 조건 | 내용 |
|---|---|
| 현행 | v0.11 그대로 (이 문서) |
| phonetic 보호만 | 검색 1위의 phonetic 표면 일치를 보호 대상에 넣는다 |
| 작은 spread 재정렬 생략만 | CE spread가 아주 작으면 재정렬을 건너뛴다 |

q219·m103의 회복만 보면 안 된다. **현재 개선된 q208(5→2)·q215(5→1)·q310(3→2)를
잃지 않는지** 함께 확인한다. 잘못 추정한 영어 가사나 여러 곡에 겹치는 구절도
검증 대상이다 — Gemini 백엔드에 있는 유사 보호 규칙은 비교 후보이지 그대로 옮겨
안전하다는 보장이 아니다.

조건 간 비교가 성립하는지부터 확인한다.

1. 세 조건의 **리랭킹 전** 후보 ID·순서·검색 점수가 질의별로 동일한가
   (`detail` CSV의 `candidate_rank@30`과 `explain.jsonl`의 `retrieval_score`).
   다르면 고정이 깨진 것이므로 지표를 비교해서는 안 된다.
2. `runinfo.json`의 `corpus.point_count` · `bm25.sha256_12` ·
   `lyrics_source.full_lyrics_total_chars` · `analysis.meta.analyzer`가 세 조건에서 같은가.

보호 규칙을 바꿀 때는 **두 곳을 같이 바꿔야 한다** — 서비스 라우터
(`search_router.py`의 `protected_lyric_ids`)와 평가용 사본
(`evaluate_search_accuracy.py`의 `_rerank_with_lyric_protection`). 한쪽만 바꾸면
평가와 서비스가 다른 규칙으로 동작한다.
