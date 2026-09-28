# 재질문 정책 비교 v01 (단계 3)

`src/retrieval/evaluate_clarification.py`의 첫 측정. "거절하고 다시 물으면 정말
나아지는가"를 정책별로 재고, 질문을 던지는 정책이 **Reject-only보다 나은지**를 본다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-15 |
| 코퍼스 | 905곡 (Pinecone namespace `dev`) |
| 질의 | `docs/eval/queries.json` split=dev, 집계 대상 53건 |
| 개입 대상 | 9건 — c607 c701 c705 c707 m402 q101 q115 q203 q316 |
| top_k / candidate_k | 10 / 30 |
| `RERANKER_SPREAD_REF` | `0.02` |
| 소요 | 673초 |

개입 대상 = 최초 검색에서 정답이 Top-10 밖인 질의. 정답이 이미 보이면 사용자가
"이 중에는 없어요"를 누르지 않으므로 모든 정책이 동일하다.

## 결과

| 정책 | 전체 Hit@10 (n=53) | 개입 Hit@10 (n=9) | 오답 시 후보@30 유지 |
|---|---|---|---|
| initial | 0.830 | 0.000 | 0.889 |
| **reject_only** | **0.887** | **0.333** | 0.889 |
| reject_twice | 0.906 | 0.444 | 0.889 |
| skip | 0.887 | 0.333 | 0.889 |
| oracle:vocal_gender | **0.962** | **0.778** | 0.889 |
| oracle:genre | 0.925 | 0.556 | 0.889 |
| oracle:release_era | 0.906 | 0.444 | 0.889 |
| oracle:type | 0.849 | 0.111 | 0.889 |
| noisy:vocal_gender | 0.849 | 0.111 | 0.889 |
| noisy:genre | 0.887 | 0.333 | 0.889 |
| noisy:release_era | 0.830 | 0.000 | **0.556** |
| noisy:type | 0.849 | 0.111 | **0.444** |
| oracle:best (천장) | 0.981 | 0.889 | 0.889 |

`skip`이 `reject_only`와 정확히 같다 — "잘 모르겠어요"가 상태를 오염시키지 않는다는
증거다(계획서 완료 판정 항목).

`m402`는 최초부터 후보@30 밖이라 어떤 정책으로도 복구되지 않는다. 표지 확인 전
부분 라벨(`label_status=pending_cover`)이므로 라벨 문제일 가능성이 있다.

## 판정 1 — 질문은 Reject-only보다 낫다. 단 슬롯에 따라 다르다

개입 집합에서 Reject-only는 9건 중 3건을 회수한다. 성별을 물으면 7건이다.
천장(`oracle:best`)은 8건이므로, **질문 선택 로직을 만들 가치가 있다.**

    Reject-only        3/9
    성별만 묻기         7/9      ← 단일 슬롯 최고
    완벽한 슬롯 선택     8/9      ← 천장

성별 하나만 물어도 천장의 대부분을 가져온다. 단계 4의 질문 선택기는 이 격차
(7 → 8, 1건)를 위해 만드는 것이므로, **복잡한 정보이득 계산보다 "성별을 먼저
묻는다"는 단순 규칙이 먼저다.**

## 판정 2 — `type`을 묻는 것은 맞게 답해도 해롭다

`oracle:type`은 개입 Hit@10이 0.111로 **Reject-only(0.333)보다 낮다.**
정답을 정확히 답해도 결과가 나빠진다.

원인은 두 가지다.

**(a) 경로 게이팅이 꺼진다.** `search_router.py:817`의 `use_deep_audio_fusion`
조건에 `not analysis.has_artist_type_clue`가 있다. 답변으로 `artist_type`이
채워지면 이 조건이 깨져 deep audio fusion 경로가 통째로 꺼진다.

    답변 전: has_artist_type_clue=False → deep fusion 가능
    답변 후: has_artist_type_clue=True  → deep fusion 불가

q115가 정확히 그 경로에 의존하는 질의이고, 실제로 Reject-only에서 4위였다가
`oracle:type`에서 Top-10 밖으로 떨어졌다.

**(b) 부스팅이 약하다.** `vocal_gender`는 `_apply_explicit_boosts`에서 명시적
속성 부스팅을 받지만, `artist_type`은 `_metadata_clue_strength`의 유사도 합산에
섞여 들어갈 뿐이다. 후보 순위를 한두 칸 올리는 데 그쳐 Top-10 진입으로 이어지지
않는다(c701 24→23, q101 18→16, q203 14→12).

## 판정 3 — "부스팅이라 안전하다"는 슬롯마다 다르다

계획서는 "하드 제외는 rejected_ids뿐이므로 답변이 틀려도 안전하다"고 적었다.
코드상 era/artist_type은 실제로 필터가 아니라 부스팅이다(하드 필터 없음 확인).
그런데 **부스팅만으로도 후보 풀 30칸에서 정답이 밀려난다.**

| 슬롯 | 오답 시 후보@30 이탈 | 이탈 질의 |
|---|---|---|
| vocal_gender | 0/9 | 없음 |
| genre | 0/9 | 없음 |
| release_era | 3/9 | c705, q115, q316 |
| type | 4/9 | c701, c705, q115, q316 |

정답이 후보 풀에서 사라지면 이후 어떤 거절·질문으로도 회수할 수 없다.
**`type`과 `release_era`는 재질문 슬롯에서 빼는 것이 안전하다.**

## 단계 4로 넘기는 결론

1. **질문 순서는 `vocal_gender` → `genre`.** 이 둘은 오답이어도 후보 풀을 지킨다.
2. **`type`·`release_era`는 제외.** 맞아도 이득이 적고 틀리면 정답을 잃는다.
   쓰려면 `search_router.py:817`의 게이팅 의존성부터 끊어야 한다.
3. 질문 선택기의 실질 헤드룸은 **1건(7→8)** 이다. 정교한 정보이득 계산보다
   슬롯 우선순위 규칙이 비용 대비 효과가 크다.
4. 표본이 9건이라 1건이 11.1%p다. **test split 측정 전에는 확정하지 말 것.**

## 재현

```bash
# 임베더를 메인 스레드에서 미리 load() 한 뒤 (RUN_INFO v0.5 이슈 3)
venv/bin/python -m src.retrieval.evaluate_clarification \
  --split dev --top-k 10 --candidate-k 30 \
  --output-dir experiments/reranking/results_clarify_v01
```

`clarify_detail.csv`(100KB)는 커밋 대상이다 — 정책 간 우열의 1차 증거이고
`analysis_json`이 없어 크기가 작다.
