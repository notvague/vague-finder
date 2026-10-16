## 조건 대조 (runinfo)
- A vs B: 다른 키 없음 · 같은 분석 캐시 True · 스위치 차이 {'attr_filter_weight': (0.0, 0.5)}
- A vs D: 다른 키 없음 · 같은 분석 캐시 True · 스위치 차이 {'attr_filter_weight': (0.0, 0.5), 'attr_filter_genre_policy_effective': ('always', 'conditional'), 'attr_filter_genre_policy': ('always', 'conditional')}
- A: split=test corpus=artifacts/qdrant_eval_v55(5143) bm25=bf606b19ab02 lyrics=e0d58f28f1f4a563 gemini=vertex/global cache_drift=[] fallback_ids=[] policy_eff=always
- B: split=test corpus=artifacts/qdrant_eval_v55(5143) bm25=bf606b19ab02 lyrics=e0d58f28f1f4a563 gemini=vertex/global cache_drift=[] fallback_ids=[] policy_eff=always
- D: split=test corpus=artifacts/qdrant_eval_v55(5143) bm25=bf606b19ab02 lyrics=e0d58f28f1f4a563 gemini=vertex/global cache_drift=[] fallback_ids=[] policy_eff=conditional

## 지표 (n = 질의 수)
| 구성 | n | 후보 30 안 | 리랭킹 전 Hit@10 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 리랭킹 결과 | 리랭킹 실행 수(detail rerank_calls 합, 모델 호출 수는 usage) | 유효하지 않은 결과 질의 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 42 | 25 | 18 | 15 | 22 | 22 | 0.421 | applied=39 policy_skipped=3 | 39 | 0 |
| B | 42 | 25 | 18 | 15 | 22 | 22 | 0.421 | applied=39 policy_skipped=3 | 39 | 0 |
| D | 42 | 25 | 18 | 15 | 22 | 22 | 0.421 | applied=39 policy_skipped=3 | 39 | 0 |

## 쌍 비교 (최종 순위 기준, 밖 = 10위 밖)
| 쌍 | 같음 | 나음 | 나쁨 | 10위 안 든 | 빠진 | 안에서 오름 | 내림 | 후보 지문 같음/다름 | 후보 30 새로 든 | 빠진 |
|---|---|---|---|---|---|---|---|---|---|---|
| A→D | 42 | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 34/8 | 0 [] | 0 [] |
| B→D | 42 | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 36/6 | 0 [] | 0 [] |
| A→B | 42 | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 0 [] | 34/8 | 0 [] | 0 [] |

- B→D 후보 지문 같은데 정답 순위 다름(리랭킹 변동): 0 []
- B→D 후보 지문 같은데 top-10 글자가 다름(비정답 포함 변동): 0 []
- B→D 후보 지문 다른데 정답 순위 같음: 6 ['t005', 't014', 't020', 't023', 't049', 't050']

## 정책 적용 사유 (explain의 요청 단위 장르 판정 — conditional 실행에만 기록)
- A: 사유 {'': 42} · (사유, 뺄 장르 조건 있었나) {} · 보조 경로 기여 있는 질의 0 · 정답이 보조 목록에 0
- B: 사유 {'': 42} · (사유, 뺄 장르 조건 있었나) {} · 보조 경로 기여 있는 질의 7 · 정답이 보조 목록에 4
- D: 사유 {'': 33, 'gender_condition': 8, 'no_gender': 1} · (사유, 뺄 장르 조건 있었나) {('gender_condition', 'True'): 7, ('gender_condition', 'False'): 1, ('no_gender', 'False'): 1} · 보조 경로 기여 있는 질의 8 · 정답이 보조 목록에 3

## 지연 (detail의 retrieval_ms · rerank_only_ms; 첫 질의와 5초 초과 검색은 적재로 분류)
| 실행 | 첫 질의 검색 s | 적재로 분류된 그 밖의 질의 | 준비된 상태 검색 중앙값 / p95 / 최대 (n) | 리랭킹 중앙값 / p95 (n) | 전체 질의당 검색+리랭킹 중앙값 / p95 / 합계 분 |
|---|---|---|---|---|---|
| A | 352.4 | 없음 | 0.59 / 1.49 / 3.87 (41) | 4.82 / 5.72 (39) | 5.36 / 7.30 / 9.5 |
| B | 351.2 | 없음 | 0.58 / 1.70 / 4.08 (41) | 4.99 / 6.02 (39) | 5.66 / 7.72 / 9.6 |
| D | 351.1 | 없음 | 0.57 / 1.67 / 4.21 (41) | 4.84 / 5.96 (39) | 5.59 / 7.85 / 9.5 |

## 순위가 달라진 질의 0건 (질의 원문은 적지 않는다 — detail CSV 참조)
| 질의 | 유형 | 후보/전/최종 A | 후보/전/최종 B | 후보/전/최종 D | 사유 D | A→D | B→D | A→B |
|---|---|---|---|---|---|---|---|---|

저장: v10_test_summary.csv · v10_test_per_query.csv
