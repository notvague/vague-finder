# 재질문 정책 비교 test_v01 — 재검색 방식 (holdout 1회차)

답변을 `QueryAnalysis`에 병합해 **재검색**하던 시절의 홀드아웃 측정.
현재 동작(재정렬)은 `../results_clarify_test_v03/`, dev는 `../results_clarify_v06/`.

이 디렉터리를 남기는 이유는 하나다 — **맞는 답변이 순위를 해치는 것**을 처음
포착한 측정이고, 설계를 바꾼 근거다.

| 질의 | reject_only | 장르 정답 반영 |
|---|---|---|
| q200 | 후보 4위 / 최종 4위 | 후보 15위 / **Top-10 밖** |

후보 풀은 그대로였고(`recall=1.0`) 순위만 추락했다. 원인은 `analysis.genre`가
`_search_performance_clues`의 sparse 항으로 흘러가 보조 검색 경로의 결과를
바꾸고 RRF 퓨전을 흔든 것이다.

## 실행 환경

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-15 |
| 질의 | split=test, 집계 대상 23건 / 개입 6건 |
| top_k / candidate_k | 10 / 30 |
| 답변 반영 | 분석 병합 후 재검색 (구 방식) |

## 결과 (개입 6건)

| 정책 | Hit@10 | 오답 시 후보유지 |
|---|---|---|
| reject_only | 2/6 | 0.833 |
| oracle:vocal_gender | 3/6 | 0.833 |
| oracle:type | 4/6 | 0.833 |
| rule:oracle | 4/6 | 0.833 |
| noisy:type | 1/6 | **0.500** |
| noisy:release_era | 1/6 | **0.667** |
| oracle:best | 5/6 | 0.833 |

오답 시 후보 이탈이 실제로 발생했다. 재정렬 방식(test_v03)에서는 전 정책 0.833(이 집합의 상한)으로
이탈이 사라진다.

`oracle:type`이 4/6으로 가장 높은데, dev에서는 1/9로 최하였다. dev에 deep audio
fusion 의존 질의(q115)가 있어 `has_artist_type_clue` 게이팅에 걸렸기 때문이다.
test에는 그런 질의가 없었다.
