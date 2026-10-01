# run_v04 — 텍스트 하이브리드 조회에서 페이로드를 top_k만 읽는다 · 2026-10-01

**결과: 요청 전체 중앙값 4.7~4.8초 → 4.2초, p95 5.7초 → 5.3초. 952곡 때(4.2초)와 같은 수준으로 돌아왔다.
순위·점수는 바뀌지 않는다(dev 57 · test 25 전 질의 동일).**

## 무엇을 바꿨나

`src/vector_db/qdrant_backend.py`의 하이브리드 조회. Qdrant에는 dense·sparse 합산이 없어 둘을 컬렉션 전체로 훑고
코드에서 더한다. 지금까지는 그 두 번의 전수 조회가 **곡마다 페이로드를 함께** 받았다. 3,010곡에서는 질의마다 최대 약 6,000건의
페이로드(표본 1곡 약 1.4KB)를 복사한다. 로컬 모드에서는 이 복사가 파이썬에서 일어나므로, 병렬로 도는 다른 경로와
CPU를 다퉈 API에서는 단독 측정보다 더 느려지는 것으로 보인다(단독 약 100ms, API 430ms).

이제 전수 조회는 **점수만** 받고, 합산해 남긴 top_k만 `retrieve`로 페이로드를 읽는다. 점수 계산과 정렬은 페이로드와
무관하므로 결과는 같다. `include_metadata=False`이면 예전처럼 페이로드를 읽지 않는다.

단독으로 잰 조회 비용(로컬 모드, 3,010곡, 실제 질의 크기의 sparse 3~20개 항):

| | 페이로드 받음 | 점수만 |
|---|---|---|
| dense 전수 | 40ms | 8ms |
| sparse 전수 | 56~92ms | 41~59ms |
| top_k 페이로드 읽기 (30~90곡) | — | 1ms 미만 |

## 순위가 그대로인지 — 분석 캐시로 입력을 고정해 다시 쟀다

```bash
venv/bin/python -m src.retrieval.evaluate_search_accuracy --split dev \
    --analysis-cache experiments/reranking/analysis_cache_v06_dev.json --output-dir <임시 폴더>
```

v22 detail(`results_v22_corpus3010`)과 질의마다 대조했다 — `candidate_rank@30` · `baseline_rank` · `rerank_rank` ·
`baseline_top_ids` · `rerank_top_ids` · `candidate_digest` · 리랭커 점수 최소·최대·폭 · 가사 일치 유형 · 결과 제목 ·
설명 문장(`explain_top1`·`explain_relevant`) · 재정렬 단계. **dev 57 · test 25건 모두 한 칸도 다르지 않다.** 요약 CSV도
줄바꿈 문자만 다르고 내용이 같다. 평가 경로의 검색 시간 중앙값은 dev 712 → 425ms, test 743 → 387ms다.

`tests/test_qdrant_backend.py`의 `test_hybrid_scans_scores_only_and_reads_payload_for_top_k`가 예전 방식(페이로드를
받아 합산)과 결과가 같은지, 전수 조회가 페이로드를 요청하지 않는지, 페이로드는 top_k만 읽는지를 지킨다.

## 실제 API 지연 (dev 57건, 예열 뒤 단일 요청, 질의 분석은 실제 Gemini 호출)

```bash
SEARCH_TIMING_LOG=artifacts/timing/hybrid_payload.jsonl venv/bin/uvicorn src.backend.main:app
venv/bin/python -m src.retrieval.measure_search_latency --split dev \
    --timing-log artifacts/timing/hybrid_payload.jsonl --output-dir experiments/latency/run_v04_hybrid_payload
```

| 중앙값 · p95 (ms) | 952곡 | 3,010곡 변경 전 (run_v03, 2회) | **3,010곡 변경 후** |
|---|---|---|---|
| **요청 전체 (client)** | 4,205 · 4,657 | 4,706~4,820 · 5,660~5,680 | **4,212 · 5,315** |
| 질의 분석 | 1,968 · 2,296 | 1,892~1,991 · 2,477~2,628 | 1,930 · 2,516 |
| 검색 | 2,168 · 2,610 | 2,633~2,907 · 3,340~3,526 | 2,282 · 2,881 |
| └ 경로 (가장 느린 경로) | 290 · 539 | 782~884 · 1,106~1,271 | **430 · 832** |
| └└ 텍스트 DB 조회 | 127 · 271 | 428~457 · 755~780 | **267 · 553** |
| └ 리랭킹 | 1,867 · 2,041 | 1,856~1,979 · 2,209~2,507 | 1,842 · 2,064 |
| 최댓값 | 7,265 | 8,775~8,826 | 7,374 |

실패 0건, 분석 재시도 2건. 변경 전 두 측정 사이의 중앙값 차이가 114ms였으므로, 약 500ms 감소는 측정 변동보다 크다.
변경 후는 한 번만 쟀다.

**경로가 952곡보다 아직 140ms 느리다.** 텍스트 조회 자체(sparse 전수 비교)가 곡 수에 비례하기 때문이다 — 로컬 모드는
sparse를 파이썬으로 비교한다. 서버 모드(`QDRANT_URL`)는 역색인을 써서 이 비용이 다르다.

## 산출물

- 이 폴더 — `latency_summary.csv`·`latency_runinfo.json`은 커밋, `latency_detail.csv`는 로컬
- 서버 구간 기록 `artifacts/timing/hybrid_payload.jsonl` (로컬)
