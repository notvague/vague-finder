# v31 — E1′ 일반 속성 가산에만 배율 (`ATTR_BOOST_SCALE`) · 2026-10-08 밤

E1(`BOOST_SCALE`, 가산 전체 배율)이 dev +2 · test −1로 갈린 뒤의 후속이다(`../results_v27_boost_scale/RUN_INFO.md`).
E1에서 잃은 질의(q310·q111)는 제목 구조·표기 가산에 기대어 후보에 들던 곡이고, 얻은 질의는 성별·장르 같은 일반 속성 가산이
경쟁곡 수백 곡을 올린 경우였다. 그래서 **제목(`title_*`)·아티스트·가사 키워드 가산은 그대로 두고, 성별·발매 시기·솔로/그룹 형태·
보컬 역할/편성·장르 가산에만** 배율을 건다.

**결과: `ATTR_BOOST_SCALE=0.2`가 dev에서 손실 없이 확장 Hit@10 45 → 47, 엄격 37 → 39, Hit@1 +2, 확장 MRR 0.573 → 0.604.
E1 0.2가 잃던 q310은 3위 → 7위로 남는다.** test는 아직 안 쟀다 — 아래 「다음」.

## 스위치

`search_router._attr_boost_scale()` → `_apply_explicit_boosts`의 `attr_unit = boost_unit * ATTR_BOOST_SCALE`.
`attr_unit`을 쓰는 규칙: `vocal_gender_match` · `vocal_gender_partial` · `vocal_gender_mismatch`(기본 0이라 무관) · `release_era` ·
`artist_type` · `performance_clues` · `genre_match`. 그대로 `boost_unit`을 쓰는 규칙: `title_exact` · `title_constraints` ·
`title_hanja_presence` · `title_meaning` · `artist_match` · `lyric_*`. 단위 테스트 `tests/test_search_explain.py::test_attr_boost_scale_leaves_title_boosts_alone`.

## 조건

| 항목 | 값 |
|---|---|
| 기준선 | **main (PR #17, 성별 감점 0)** = `../results_v28_gender_penalty/default_dev` — 엄격 37 · 확장 45. v22(36 · 44)가 아니다 |
| 코퍼스·질의·분석·CE | v22와 같음 (3,010곡, `analysis_cache_v06_dev.json`, CE 앞 20개·가중치 0.45·spread_ref 0.01) |
| 명령 | `env ATTR_BOOST_SCALE=0.2 venv/bin/python -m src.retrieval.evaluate_search_accuracy --split dev --analysis-cache experiments/reranking/analysis_cache_v06_dev.json --output-dir experiments/reranking/results_v31_attr_boost_scale/a020` 뒤 `relaxed_metrics --split dev` |

## 지표 — dev 57

| 설정 | 엄격 H@1 | 엄격 H@5 | 엄격 H@10 | 엄격 MRR | 확장 H@1 | 확장 H@5 | 확장 H@10 | 확장 MRR | 회복 / 손실 (확장 Top-10) |
|---|---|---|---|---|---|---|---|---|---|
| 기준 (main) | 22 | 32 | 37 | 0.451 | 28 | 41 | 45 | 0.573 | — |
| E1 `BOOST_SCALE=0.2` (참고, v30 `s020_g000`) | 22 | 33 | 38 | 0.457 | 28 | 40 | 46 | 0.573 | +q202 c707 / **−q310** |
| **E1′ `ATTR_BOOST_SCALE=0.2`** | **24** | 34 | **39** | **0.488** | **30** | 41 | **47** | **0.604** | +q202(9) c707(10) / 없음 |
| E1′ 0.3 | 23 | 34 | 38 | 0.475 | 29 | 41 | 46 | 0.591 | +c707 / 없음 |
| E1′ 0.5 | 22 | 35 | 37 | 0.465 | 28 | 42 | 45 | 0.581 | 없음 |

0.2에서 순위가 바뀐 질의는 11건이다. 올라간 것: q202 밖→9 · c707 밖→10 · q101 9→1 · q316 8→1 · c706 9→4 · q218 3→2.
내려간 것: q310 3→7 · q106 4→7 · m203 1→3 · q115 5→6. **Top-10을 벗어난 질의는 없다.**

## 읽는 법

- E1과 E1′의 차이는 q310 하나로 요약된다. E1 0.2는 q310을 후보 밖으로 보냈는데(제목 구조 가산까지 0.2배), E1′는 제목 가산을 그대로
  두어 7위에 남는다. test에서 E1이 잃은 q111(한자 표기 가산 의존)도 같은 이유로 남을 것으로 보이지만 **test는 아직 재지 않았다**
- Hit@1이 +2(q101·q316)인 것은 일반 속성 가산이 작아지면서 텍스트 RRF 순위가 그대로 드러난 결과다. q101은 RRF 융합 순위가 높은데
  성별·시기·편성 가산을 받은 경쟁곡들에 밀려 9위였다
- 0.5는 Hit@10에 변화가 없고 0.3은 중간이다. 효과는 0.2에서만 나온다. 0.1은 재지 않았다 — 0.2에서 이미 잃는 질의가 없으므로
  더 줄여서 얻을 것은 "메타데이터만으로 특정하는" 질의를 잃는 것뿐이다
- E1′ 0.2의 top-10에서 첫 정답보다 위에 온 **미검토 쌍은 22쌍 (q106 3, q110 2, q112 7, q202 1, q203 1, q210 1, c602 1, c607 6)**다. 판정 전 확장 지표는 낮은 쪽으로 치우친 값이다

## 다음

1. **test 25건은 2차 질의 세트(v09, 100건)가 오기 전까지 열지 않는다.** 지금 세트는 질의 1건이 4%p라 E1의 dev +2 · test −1 같은 결과가
   한 번 더 나올 뿐이다. v09 dev 60건을 합쳐 142건으로 다시 잰 뒤, 봉인 test 40건으로 한 번 확인한다
2. 그 전까지 `ATTR_BOOST_SCALE` 기본값은 1.0(현행)으로 둔다. 코드는 main에 넣되 동작은 그대로다
3. 채택되면 미검토 쌍 판정 → 재질문·지연 재측정 → 기본값 0.2
