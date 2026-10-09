# v33 — v09 2차 세트 dev 59건에서 기준선 · E1′ · E5 재측정 · 2026-10-09

v09(2026-10-09 병합, `docs/eval/queries.json` 0.6.0, n001~n100)의 dev 59건(no_target 3 제외)에서 main 기준선, E1′(`ATTR_BOOST_SCALE=0.2`),
E5(Gemini listwise 세 설정)를 쟀다. **v06 dev 57건의 결론이 새 세트에서도 서는지**가 질문이다. 코드 변경 없음.

**결과: E1′은 서지 않는다(−1, 손실 3). Gemini listwise는 선다 — Hit@10 +6~7, Hit@1 +14~18, 손실 0. 두 세트 합산 116건에서
listwise(2패스, grounding 끔)는 Hit@10 73 → 87(0.629 → 0.750), Hit@1 39 → 60.**

## 조건

| 항목 | 값 |
|---|---|
| 질의 | `eval_queries_v09.csv` dev 59 (`--split dev`). test 38은 **봉인 — 열지 않았다** |
| 분석 | `analysis_cache_v09_dev.json`. **n002·n025는 폴백**(분석기 모달리티 검증 오탐 — NEXT_WORK §2-6) → `--allow-fallback-analysis`. 모든 설정에 동일 |
| 코퍼스·CE·후보 | v22와 같음 (3,010곡, 후보 30, CE 앞 20개·가중치 0.45·spread_ref 0.01). 기준선은 main(PR #17, 성별 감점 0) |
| listwise | `RERANKER_BACKEND=gemini_listwise`, 가중치 0.85, 후보 30. `GEMINI_RERANK_USE_SEARCH=0`·`GEMINI_RERANK_PASSES=1`로 변형 |
| 명령 | `env <스위치> venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v09.csv --split dev --analysis-cache experiments/reranking/analysis_cache_v09_dev.json --allow-fallback-analysis --output-dir experiments/reranking/results_v33_v09_dev/<폴더>` |
| 소요 | CE 설정 2분 40초, listwise 7~16분 |

확장 지표(허용 정답)는 없다 — v09에는 아직 `allowed`가 없다. 전부 엄격(원래 타깃) 기준이다.

## 지표 — v09 dev 59

| 설정 | H@1 | H@5 | H@10 | MRR@10 | nDCG@10 | 회복 / 손실 (Top-10) | 리랭킹 중앙값 · p95 |
|---|---|---|---|---|---|---|---|
| 기준 (main) | 17 | 31 | 36 | 0.385 | 0.439 | — | 1.8초 · 1.9초 |
| E1′ `ATTR_BOOST_SCALE=0.2` | 15 | 30 | 35 | 0.360 | 0.416 | 2 / **3** | 1.8초 · 1.9초 |
| listwise 1패스 (grounding 끔) | 31 | 40 | 42 | 0.588 | 0.618 | 6 / 0 | 5.4초 · 16.5초 |
| listwise 2패스 (grounding 끔) | 32 | 41 | 43 | 0.607 | 0.637 | 7 / 0 | 10.1초 · 21.5초 |
| listwise + grounding (2패스) | **35** | **42** | 43 | **0.641** | **0.662** | 7 / 0 | 13.0초 · 23.4초 |

유형별 Hit@10 (misinformation은 tier, 나머지는 modality_focus):

| 유형 (건수) | text (25) | audio (12) | image (9) | multimodal (6) | misinformation (7) |
|---|---|---|---|---|---|
| 기준 (main) | 18 | 5 | 8 | 3 | 2 |
| E1′ ATTR 0.2 | 16 | 5 | 8 | 4 | 2 |
| listwise 1패스 | 22 | 5 | 8 | 4 | 3 |
| listwise 2패스 (grounding 끔) | 22 | 5 | 8 | 5 | 3 |
| listwise + grounding | 22 | 5 | 8 | 5 | 3 |

## 두 세트 합산 — v06 dev 57 + v09 dev 59 = 116

| 설정 | v06 dev 57 H@1 / H@10 | v09 dev 59 H@1 / H@10 | 합산 116 H@1 / H@10 | 합산 MRR(가중) |
|---|---|---|---|---|
| 기준 (main) | 22 / 37 | 17 / 36 | 39 / 73 (0.629) | 0.417 |
| E1′ ATTR 0.2 | 24 / 39 | 15 / 35 | 39 / 74 (0.638) | 0.423 |
| listwise 1패스 | 28 / 42 | 31 / 42 | 59 / 84 (0.724) | 0.574 |
| listwise 2패스 (grounding 끔) | 28 / 44 | 32 / 43 | 60 / 87 (**0.750**) | 0.592 |
| listwise + grounding | 28 / 43 | 35 / 43 | 63 / 86 (0.741) | **0.608** |

v06 쪽 숫자는 `results_v28_gender_penalty/default_dev`(기준) · `results_v31_attr_boost_scale/a020` · `results_v32_gemini_listwise/*`.

## 읽는 법

- **기준선 자체가 v06보다 어렵다.** Hit@10 0.649 → 0.610, Hit@1 0.386 → 0.288, Candidate Recall@30 0.788 → 0.729. 소리 질의 12건 중 5건, 틀린 기억 7건 중 2건만 Top-10이다.
  미스 23건 중 15건은 후보 30 안에도 없다 — 리랭커가 못 건지는 구간이다
- **E1′은 v06에 맞춘 결과였다.** 잃은 n002·n031·n038은 텍스트 경로 30~37위인 정답을 시기·성별·장르·편성 가산이 Top-10까지 올리던 질의다.
  v06에는 "일반 속성 가산이 경쟁곡을 올리는" 유형이, v09에는 "일반 속성 가산 덕에 정답이 오르는" 유형이 더 많았다. **채택하지 않는다. `ATTR_BOOST_SCALE` 기본값 1.0 유지.**
  v09를 만든 목적이 이런 과적합을 거르는 것이었다
- **listwise는 세트가 바뀌어도 같은 방향·같은 크기다.** 손실 0은 두 세트 모두. 회복 7건 중 n014·n019·n025·n043은 **1위**로 올라왔다.
  폴백 분석이 된 n025(태그 전부 비움)까지 후보 18위에서 1위로 올린 것은 리랭커가 분석기 실패를 일부 보상한다는 뜻이다
- **grounding은 Hit@1·MRR에 보탠다**(31~32 → 35, 0.59~0.61 → 0.64). v06에서는 차이가 없었는데 v09의 외부 맥락 질의가 더 많아서다.
  대가는 리랭킹 13초. 1패스는 5.4초에 Hit@10 42로 이득의 대부분을 남긴다
- 틀린 기억 7건은 어느 설정에서도 2~3건이다. 리랭커가 아니라 질의 분석·후보 회수 쪽 문제다
- 리랭커의 희소 단서 정규식 과적합 우려(9월 리뷰)는 **v09가 새 질의라 여기서 1차로 해소**됐다. 최종 확인은 봉인 test 38

## 다음

1. **채택 후보: Gemini listwise.** 지연 설계를 정한다 — ① 1패스·grounding 끔 기본(요청 약 8초) ② CE 결과 먼저 내보내고 LLM 순서를 뒤이어 갱신 ③ 외부 맥락·오정보 질의에서만 LLM.
   ②·③은 코드 작업이고, 발표 데모는 ①이면 된다
2. 봉인 test 38건은 **채택 설정 하나로 한 번**. v06 test 25도 같은 설정으로 한 번
3. 재질문 평가(`evaluate_clarification`)를 listwise로 다시 잰다 — 9월 `results_clarify_gemini_v02`에서 실행마다 1위/7위로 갈리던 변동을 확인
4. 미검토 쌍 판정 뒤 v09에도 `allowed`를 붙인다. 설명 패널의 리랭커 문장 처리(§4)
5. 분석기 오탐 2건(n002 '앨범', n025 '팝스타') 수정은 별도 — 지문이 바뀌므로 전후 비교로
