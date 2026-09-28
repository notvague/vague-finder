# v0.19 측정 정보 — v06 질의 세트 기준선 (2026-09-24)

**이것이 앞으로의 기준선이다.** v05는 그대로 얼려 둔다 — v11~v18 숫자가 그 위에서
나왔고, 질의가 늘고 정답이 바뀐 세트와 직접 비교하면 안 된다.

## 실행

```
# 캐시: v05 분석을 그대로 잇고 새 6건만 분석한다
cp experiments/reranking/analysis_cache_dev.json  experiments/reranking/analysis_cache_v06_dev.json
cp experiments/reranking/analysis_cache_test.json experiments/reranking/analysis_cache_v06_test.json

venv/bin/python -m src.retrieval.build_analysis_cache \
    --input experiments/reranking/eval_queries_v06.csv --split dev \
    --output experiments/reranking/analysis_cache_v06_dev.json \
    --reuse-despite-drift --drift-reason "..."

venv/bin/python -m src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_v06.csv --split dev \
    --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
    --output-dir experiments/reranking/results_v19_v06
```

`--input`과 `--analysis-cache`를 **반드시 적는다.** 두 스크립트의 기본값은 아직 v05다.

## 실행 조건

| | |
|---|---|
| 질의 | dev **57건** / test **25건** (v05: 53 / 23) |
| 분석 | 캐시 고정 · **폴백 0건** |
| 랭킹 스위치 | `lyric_protect_phonetic_top1=false` · `min_confidence=0.8` · `boost_scale=1.0` · `reranker_min_spread=0.0` |
| 리랭커 | cross_encoder · `weight=0.45` · `spread_ref=0.01` |
| top_k / candidate_k | 10 / 30 |

v18(dev)·v17(test)과 **같은 조건**이다. 그래야 공통 집합 비교가 성립한다.

### 캐시 지문이 어긋난 채로 재사용했다

`postprocess_sha`가 달라졌다 — 분석기를 동기·비동기 두 고리로 쪼갰기 때문이다.
이 해시는 `query_analyzer.py` **파일 전체**를 보므로 호출 구조만 바꿔도 달라진다.

그래도 옛 분석을 유지했다. 근거는 **결과를 정하는 코드가 HEAD와 바이트 단위로
같다**는 대조다 — `_PROMPT_TEMPLATE` · `_apply_lyric_safeguards` ·
`_apply_metadata_safeguards` · `_fallback` · `_reference_year` ·
`modality_queries.py` 전체 · 응답→`QueryAnalysis` 후처리 순서.

숨기지 않았다. `--reuse-despite-drift`로 명시했고, 무엇이 어긋난 채로 무엇을
유지했는지가 캐시 `meta.drift_notes`에 남는다.

## A. v06 기준선

| split | 건수 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| dev | 57 | 0.404 | 0.684 | 0.789 | 0.775 | 0.536 | 0.594 |
| test | 25 | 0.560 | 0.560 | 0.680 | 0.638 | 0.577 | 0.581 |

## B. 공통 집합 — v05 대비 **결과가 완전히 같다**

질의 원문과 정답이 **둘 다 같은** 75건(dev 52 · test 23). 분석 캐시도 같은 항목이다
(공통 항목의 `analysis`가 v05 캐시와 완전히 일치함을 확인했다).

| split | 공통 | 기준 | Hit@1 | Hit@10 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| dev | 52 | v18 → v19 | 0.442 → 0.442 | 0.827 → 0.827 | 0.827 → 0.827 | 0.580 → 0.580 |
| test | 23 | v17 → v19 | 0.565 → 0.565 | 0.652 → 0.652 | 0.652 → 0.652 | 0.578 → 0.578 |

**무엇을 맞춰 봤는지 적는다** — 정답 순위(`rerank_rank`)뿐 아니라 **반환 Top-10
목록**과 **리랭킹 전 목록**까지 비교했고 셋 다 전부 같았다(dev 52 · test 23).

그래도 **이것이 확인한 범위는 좁다.**

평가기는 라우트를 거치지 않고 `SearchRouter.search()`를 직접 부르고, 캐시를 주면
`analyze()`는 **한 번도 실행되지 않는다**(`lyrics_snapshot_out`·`_lyric_evidence`·
`analyze_async`·벽시계 상한은 평가기 코드에 등장하지 않는다).

| | 이 측정이 확인한 것 |
|---|---|
| ✅ 확인됨 | **같은 분석을 넣으면 검색·융합·부스트·리랭킹 결과가 v18/v17과 같다.** 죽은 경로 기록(`_path_result`)처럼 `search()` 안에서 도는 변경이 여기 포함된다 |
| ❌ 확인 안 됨 | 분석기 리팩터링·재시도·벽시계 상한 — **실행되지 않았다.** 라우트의 원문 인용·가사 스냅샷 고정도 마찬가지다 |

그쪽은 단위 시험(`tests/test_demo_failure_paths.py`·`tests/test_lyric_evidence.py`)과
브라우저 확인(`check_demo_faults`)이 담당한다. 이 표의 숫자로 그 동작까지 검증했다고
말하면 안 된다.

## C. v05 대비 차이는 어디서 왔나 — **성능 변화가 아니다**

### dev 0.811 → 0.789 (Hit@10)

| 구성 | 건수 | Hit@1 | Hit@10 | Recall@10 |
|---|---|---|---|---|
| v05 전체 | 53 | 0.434 | 0.811 | 0.811 |
| 공통 집합 | 52 | 0.442 | 0.827 | 0.827 |
| m402 — 정답 교체 **전** | 1 | 0.000 | **0.000** | 0.000 |
| m402 — 정답 교체 **후** | 1 | 0.000 | **1.000** | 0.500 |
| 새 표지 질의 | 4 | 0.000 | 0.250 | 0.167 |
| **v06 전체** | **57** | **0.404** | **0.789** | **0.775** |

내려간 것은 **어려운 표지 질의 4건이 분모에 들어와서**다. 같은 질의는 하나도
나빠지지 않았고, m402는 오히려 0.000 → 1.000이 됐다 — 틀린 정답(겨울연가 OST)을
실제로 어두운 표지 4곡으로 바꾼 덕이다.

### test 0.652 → 0.680 (Hit@10), 0.652 → 0.638 (Recall@10)

**같은 2건이 한쪽은 올리고 한쪽은 내린다.** m301·q116 둘 다 top-10 안에 정답이
들어와 Hit@10을 올렸지만, 정답이 여러 개라 부분 회수(0.67 / 0.29)에 그쳐 Recall은
내렸다. 지표 하나만 보고 방향을 말하면 틀린다.

## D. 새 표지 질의 6건

| 질의 | split | 정답 수 | R@10 | |
|---|---|---|---|---|
| m301 온통 빨간색 | test | 9 | 0.67 | 정답이 1·2위 |
| m303 흑백 사진 | dev | 3 | 0.67 | |
| q116 파란 표지·얼굴 없음 | test | 7 | 0.29 | |
| q117 글씨만 | dev | 5 | 0.00 | |
| m302 얼굴 클로즈업 | dev | 5 | 0.00 | |
| m403 알록달록 표지·잔잔한 피아노 | dev | 2 | 0.00 | |

**색·톤 질의는 되고 구성·타이포그래피 질의는 안 된다.** 이미지 경로가 무엇을 할 수
있는지에 대한 첫 숫자다.

다만 q117·m302는 후보를 좁힌 표지 통계 자체가 약했다(글자 유무·얼굴 크기를
픽셀 통계로 가늠할 수 없어 대용값을 썼다). **정답이 빠졌을 가능성이 다른 질의보다
크고, 그 방향은 재현율 과소평가다.**
