# v61 — `HYBRID_SCAN_LIMIT` 제한으로 인한 하이브리드 점수 합산 누락 진단 (2026-10-16 밤)

워크트리 `/private/tmp/vague-finder-hybrid-scan-audit`, 브랜치 `experiment/hybrid-scan-audit`(main `770843f` = #75 v60 병합 기준, 열린 PR 없음), **커밋 전**.
모델(Vertex·AI Studio·Claude)·Mongo·운영 Qdrant·멜론 호출 **0**(§6 차단 확인). **운영 검색 구현·기본값은 바꾸지 않았다** — 진단 도구·독립 기준·회귀 테스트·수정안 제안까지다.

## 0. 결론 요약

| 질문 | 답 |
|---|---|
| 조회 제한으로 후보·점수 일부가 실제로 누락되는가? | **그렇다.** 텍스트 하이브리드 조회 311건(dev 176질의) 중 **295건에서 dense 목록이 5,000개에서 잘렸고**(코퍼스 5,143점), **35건에서는 sparse 목록도 잘렸다**(질의 토큰과 겹치는 곡이 5,000을 넘음). 그 결과 **4건의 호출에서 1곡씩 dense 몫이 빠진 채 합산**됐고(기준 top-90 안의 곡, 전부 가사 질의), 그중 3건은 텍스트 top-90 안의 순서가 바뀌었다. 양쪽 목록 모두에서 사라진 곡은 **0** |
| 정확한 합산과 비교하면 서비스 후보 30개·정답 회수·지연은? | **dev 176질의의 후보 30개 ID·순서는 A(현행)와 B(정확 합산)가 전부 같다.** 정답의 후보 포함 122/176, 리랭킹 전 Hit@1/5/10·MRR@10 모두 동일. 후보 지문이 바뀐 질의는 t003 1건으로 5번째 후보의 융합 점수만 0.0516→0.0518(순서 불변). 지연: 상한 5,000 ↔ 5,143(C)의 호출당 차이 중앙값 +0.25ms(로컬 파일 모드, 교대 60회) |
| 운영 코드를 수정해야 하는가? 어떻게? | **수정이 필요하다 — 영향이 작아서가 아니라 조건이 자료에 달려 있어서다.** 현재 코퍼스에서 상한만 4,500으로 낮춰 모사하면 top-30에서 곡이 사라지는 호출이 8건 나온다(§5-4; 새 곡 추가 상황의 예측은 아니다). 권고는 **"두 조회의 상한 = 그 조회의 대상 수(`count(exact)`, 필터면 같은 필터로)"**(§7-A). 요청마다 count를 포함한 비용은 로컬에서 A 대비 중앙값 +0.9ms(필터 count는 약 17ms 추가, §5-3). 단순히 10,000으로 올리는 것은 상한이 코퍼스에 묶이는 구조를 그대로 두는 것이라 권하지 않는다. count와 조회 사이의 동시 적재 감지는 별도 설계로 남긴다 |

판단을 나눠 적는다(§8): 합성 사례에서 구현의 제한 **확인**, 현재 dev 자료에서 후보·정답 회수 영향 **없음(점수 1건·텍스트 깊은 목록 순서 4건)**, 후속 운영 수정 **필요**, 수정 뒤 검증은 §8-4(`replay_calls.py`로 수정 구현에 저장 벡터를 재생).

## 1. 현재 동작 (코드에서 확인, 변경 없음)

- **설정 이름·기본값**: `src/vector_db/qdrant_backend.py` `HYBRID_SCAN_LIMIT = int(os.getenv("QDRANT_HYBRID_SCAN_LIMIT", "5000"))` — 환경변수는 `QDRANT_HYBRID_SCAN_LIMIT`(`.env.example` 주석 "코퍼스보다 크면 전수"), 모듈 상수는 import 시점에 한 번 읽는다. 운영 `.env`에는 이 변수가 없다(기본 5000). 모듈 주석은 "코퍼스(905곡)보다 크면 전수 합산" — 3,010곡 때까지 참이었고 **5,143곡부터 거짓**이다. `docs/vector_backend.md`도 "전수를 훑어 잘림이 없다"로 적혀 있다(틀림, §7에서 함께 고칠 것)
- **적용 경로**: `QdrantIndex.query(vector, sparse_vector, top_k, filter, include_metadata)` — `sparse_vector`가 있으면 하이브리드. ① dense 조회 `query_points(using="dense", limit=HYBRID_SCAN_LIMIT, with_payload=False)` ② sparse 조회 `query_points(using="sparse", limit=HYBRID_SCAN_LIMIT)` ③ 코드에서 `merged[id] = dense + sparse`(한쪽만 있으면 그 값만) ④ 합산 내림차순 정렬 뒤 `[:top_k]` ⑤ 남은 top_k만 `retrieve`로 payload. sparse_vector가 없으면 dense만 `limit=top_k`(상한 무관). 이미지·오디오·Context 컬렉션은 이 경로를 쓰지 않는다(Context는 `context_qdrant_search.py` 별도)
- **가중치가 곱해지는 곳**: `search_service.hybrid_score(dense, sparse, alpha)` — 질의 **벡터**에 `alpha`(dense)·`1-alpha`(sparse)를 곱해 넘긴다. 백엔드는 가중을 모른다. 따라서 Qdrant가 돌려주는 dense 점수 = alpha·cos(단위 벡터·DOT), sparse 점수 = (1-alpha)·BM25 내적. 기록된 `dense_norm`(= alpha)은 0.10(가사 exact)·0.50~0.70(일반)이었다
- **호출 깊이**: 라우터 `_search_text`가 넘기는 top_k는 기본 후보 수 30(`pool_k`), 메타데이터·연주 단서 질의는 `text_pool_k = min(100, max(90, 60, pool_k))` = 90. dev 176질의에서 하이브리드 호출 311건 = top_k 30 × 178 + 90 × 133(텍스트 경로 + 균형 보조 `_search_balanced_semantic` + 연주 단서 `_search_performance_clues` + 제목 구조 필터 경로). RRF 융합에 들어가는 텍스트 목록은 `text_deep[:fusion_k]` = 상위 30(`path_k` 없음), 31~90위는 메타데이터/연주 단서 보조 신호에만 쓴다
- **필터**: `translate_filter`로 바꾼 필터가 두 조회에 모두 들어가므로 **상한은 필터를 통과한 곡에 적용**된다. 통과 수 ≤ 5,000이면 전수다. dev 311건 중 필터 호출 16건은 통과 83~713곡으로 전수였다
- **sparse 미일치**: Qdrant sparse 조회는 질의 인덱스와 겹치는 항이 하나도 없는 곡을 돌려주지 않는다(점수 0이 아니라 **없음**). 구현에서 그 곡은 dense 값만 남는다 = 수학적으로 sparse 0과 같다. 빈 sparse(`indices=[]`)는 빈 목록을 돌려주지만 서비스는 sparse 인코딩이 비면 `sparse_vector`를 넘기지 않아 이 경로에 닿지 않는다(설계상 사례, 테스트로만 확인)
- **동률**: 합산 정렬은 `sorted(..., reverse=True)`(안정)라 같은 점수면 dense 조회 순서 → sparse 전용 순서가 유지된다. Qdrant 쪽 동점 순서는 정의되지 않는다. 비교에서 허용 오차(§3) 안의 동점 묶음 안 순서 차이는 차이로 세지 않았고, 실제 dev에서는 그런 경우가 없었다
- **음수 dense**: 텍스트 컬렉션은 DOT라 음수가 나올 수 있고 상한 밖으로 잘린다. 실제 색인에서 질의 311건 중 9건에서 음수 dense 곡이 있었고(최대 107곡·최소 −0.052) 모두 하위라 상한 밖이다. **음수 몫이 잘리면 합산이 기준보다 커진다**(합성 사례로 확인, §2) — 실제 자료에서는 최상위권에 들 만큼 큰 sparse를 가진 음수 dense 곡이 없어 드러나지 않았다
- **실제 point 수**: 복사본 `artifacts/qdrant_v61_audit`(운영 색인 `artifacts/qdrant`의 파일 복사, 2026-10-16 21:34)에서 `count(exact)` = **5,143**(텍스트·이미지·오디오 각 5,143 · Context dense 5,054 · Context sparse 639). 평가기 runinfo의 `corpus.point_count`도 5,143. 전 곡에 sparse 벡터가 있다(항 수 중앙값 196, 최소 101, 최대 896, 어휘 44,486, 비영 항 1,061,090)
- **"5,143 − 5,000 = 143곡이 항상 제외"는 틀리다.** dense 조회에서 잘리는 143곡은 **질의마다 다른** dense 최하위 곡이고, 그 곡이 sparse 목록에 있으면 sparse 몫으로 남는다(한쪽 점수 누락). sparse 목록은 질의 토큰과 겹치는 곡 수(dev 중앙값 4,495곡, 35건은 5,000 초과)만큼 돌아오므로 그쪽 잘림은 질의에 따라 있거나 없다. 양쪽 모두 밖인 곡만 결과에서 사라진다

## 2. 합성 자료 재현 (`repro_synthetic.py` · `tests/test_v61_hybrid_scan_audit.py` 11건, 메모리 Qdrant · 3차원 · 7곡 · 상한 3)

기준값은 구현을 따라 쓰지 않고 **원본 벡터로 직접 내적**(`_exact`). 구현에는 모듈 상수만 작게 바꿔 부른다. 자료(질의 dense=[0.5,0,0] · sparse {7: 0.5}): dense 순위 1>2>3>7>4>5>6, sparse(7) 순위 4>5>6>7>2>1(3은 미일치); 직접 합산 4=1.15 · 5=0.85 · 2=0.65 · 7=0.60 · 6=0.55 · 1=0.50 · 3=0.35(동점 없음).

| 사례 | 결과 | 구분 |
|---|---|---|
| 상한 ≥ 자료 수(7) | 구현 = 직접 합산(id·점수, 1e-5) | 기준 일치 |
| 한쪽 목록에서 잘림 | 상한 3: 4·5·6은 dense 몫이, 1·2는 sparse 몫이 빠진 점수. 4번(sparse 1위·dense 5위)은 1.15 → 1.00 | **현재 구현에서 실제로 일어남** |
| 양쪽 상위 목록 밖인데 합산은 상위 | 7번(dense 4위·sparse 4위, 합산 4위 0.60)이 **결과(top_k=7)에서 사라진다** — 더 낮은 6·1·3은 남는다. 단순 하위 절단이 아니다 | **현재 구현에서 실제로 일어남** |
| 음수 dense | dense −1.0 · sparse 3.0인 곡: 상한 넉넉하면 2.0(정확), 상한 2면 음수 몫이 빠져 **3.0으로 부풀어** 기준보다 위로 간다 | 설계상 가능(실제 색인은 상위권 음수 없음) |
| sparse 미일치 | 질의 인덱스 99: sparse 0 → 합산 = dense, 기준과 같은 순서 | 기준 일치 |
| 빈 sparse 질의 | `indices=[]`를 백엔드에 직접 주면 dense 순서와 같다 | 설계상 사례(서비스 경로는 닿지 않음) |
| 필터 | 전체 7 > 상한 3이어도 필터 통과 3곡이면 정확. 통과 0곡이면 `{"matches": []}` | 기준 일치 |
| top_k > 결과 수 | 7곡 전부 1회씩 | 기준 일치 |
| 동점 | 같은 벡터 두 곡은 둘 다 남고 점수 같음(순서는 단언 안 함) | 기준 일치 |
| 설정 이름 | `QDRANT_HYBRID_SCAN_LIMIT=7`로 reload하면 상수 7 | 설정 경로 확인 |

`repro_synthetic.json`에 상한 3·5·7의 구현/기준/분류 표가 있다.

## 3. 독립 정확 합산 기준 (`exact_hybrid.py`)

- `ExactHybridReference.from_client`: `scroll` 페이지네이션(256/페이지, `with_vectors=[dense, sparse]`, payload는 `song_id`만)으로 **전체 벡터**를 읽는다. 페이지마다 중복 id 검사, 빈 페이지인데 다음 offset이 남으면 예외, 끝나면 `count(exact)`와 수가 같아야 한다(`ScrollIntegrityError`). dense는 float64 행렬(5,143×1,024), sparse는 역색인(인덱스 → 행·값)
- 점수: dense = `M @ q`, sparse = 질의 항별 posting 합, 합산 = 둘의 합(미일치 sparse 0). 질의 벡터는 서비스가 넘긴 그대로(alpha가 이미 곱해진 값)라 다시 가중하지 않는다
- 필터: 서비스의 `translate_filter` 결과를 `scroll(scroll_filter=…)`로 적용해 허용 id 집합을 만든다. **필터 의미는 Qdrant에 맡긴다**(이 진단의 대상이 아니고 기존 테스트가 덮는다)
- **허용 오차**: dense 1e-5, sparse·합산 1e-4(`DENSE_TOL`·`TOTAL_TOL`, Qdrant f32 누적 vs f64). 실제 dev 311건 중 상한 영향이 없는 307건의 같은 곡 점수 최대 차이는 1e-7 수준
- 동점: 합산이 허용 오차 안에서 같은 묶음 안의 순서 차이는 `compare_rankings`가 차이로 세지 않는다(`order_differs_beyond_ties`)
- **상한을 올린 조회(C)는 기준이 아니다** — C가 맞는지도 이 기준으로 확인했다(§5-2: 311/311 일치)
- 색인 지문: 복사본 텍스트 컬렉션 내용 해시 `238299e6c8c17f2a`(id 오름차순으로 dense f32·sparse 인덱스/값). B 실행 3회(세트별)·오프라인 대조·민감도 모사 모두 같은 값 — 비교 중 색인이 바뀌지 않았다. 원본 `artifacts/qdrant`·운영 DB에는 쓰지 않았다

## 4. 실제 자료 준비

| 자료 | 상태 |
|---|---|
| 분석 캐시 | v49 Vertex dev 캐시 3개(`results_v49_vertex_baseline/analysis_cache_{v06,v09,v10}_dev_vertex.json`, 57·59·60건) — `fingerprint_drift()` 빈 목록(현행 분석기 prompt `80c464515d9c`·postprocess `f18778f8f94d`와 일치), 폴백 0. **재사용** |
| 질의 벡터 | 캐시에는 없다. 로컬 HF 캐시의 KoE5(`nlpai-lab/KoE5`, `HF_HUB_OFFLINE=1`)와 `artifacts/bm25_params.json`(sha `bf606b19ab02`)로 평가 실행 중 생성 — 새 다운로드·외부 호출 없음. 생성된 벡터는 A 실행이 `calls_vectors.npz`로 저장(커밋 제외) |
| 색인 복사본 | `artifacts/qdrant_v61_audit`(운영 `artifacts/qdrant` 파일 복사, `.lock` 제거). 5,143점. 지문 §3 |
| 임베딩 시간 분리 | 평가기 `retrieval_ms`(임베딩 포함)와 캡처의 호출별 `elapsed_ms`(벡터 DB 조회+합산만)를 따로 기록 |

## 5. 실제 비교 — A(현행 5000) · B(독립 정확 합산) · C(상한 = 5,143)

같은 분석 캐시·색인 복사본·설정(`v61_guard.fixed_env`: `CONTEXT_WEIGHT=0.5`(서비스 기본), `ATTR_FILTER_WEIGHT=0`(기본·속성 보조 꺼짐), `RERANKER_ENABLED=false`, `FEEDBACK_COLLECTION_ENABLED=0`, `SEARCH_REFERENCE_YEAR=2026`, `QDRANT_URL` 제거). 기존 평가기 `evaluate_search_accuracy`로 후보 30개까지(`--candidate-k 30`, 모델 리랭킹 없음). `hybrid_capture.py`가 `QdrantIndex.query`를 감싸 A·C는 기록만, B는 하이브리드 호출만 §3 기준으로 바꿔 끼운다(dense 전용 호출은 원래 구현). 세트마다 A→B→C 순차(로컬 복사본은 한 프로세스만 연다), 끝에 A를 v06에서 한 번 더(rep2).

**측정 조건에서 서비스와 다른 점(두 구성 모두 같음)**: Mongo를 막아 **가사 표면 경로(`path.lyrics`)가 비었다.** 가사 단서 질의 19건(q209·q211·q213·q219·n007·n009·n013·n017·n039·n064·t003·t016·t018·t025·t032·t060·t063·t105·t110)의 후보 구성은 서비스(Mongo 있음)와 다르다. 따라서 아래 절대 지표는 v50 기준선과 직접 비교하지 않고 **A↔B 내부 비교**로만 읽는다. 평가기의 가사 지문도 `BlockedExternalCall`로 기록됐다(runinfo).

### 5-1. 호출 단위 — 텍스트 top-k (`compare_text_calls.py` → `text_calls_diff.csv` · `truncation_songs.csv` · `text_calls_summary.json`)

A가 기록한 질의 벡터로 기준 top-k를 다시 계산해 대조(호출 짝은 질의 ID·top_k·필터·벡터 해시로 맞춤). 311건 전부 짝이 맞았다.

| 항목 | v06 | v09 | v10 | 합 |
|---|---|---|---|---|
| 하이브리드 호출 | 105 | 102 | 104 | **311** |
| dense 목록 잘림(5,000 반환 · 대상 5,143) | 95 | 100 | 100 | **295** |
| sparse 목록 잘림(일치 곡 > 5,000) | 15 | 9 | 11 | **35** |
| 필터 호출(통과 83~713곡, 전수) | 10 | 2 | 4 | 16 |
| A top-k 집합 = 기준 | 105 | 102 | 104 | **311** |
| A top-k ID·순서 = 기준 | 105 | 102 | 101 | 308 |
| 점수 불일치 곡(모두 dense 몫 누락) | 0 | 1 | 3 | **4** |
| 양쪽 목록 밖으로 사라진 곡 | 0 | 0 | 0 | **0** |
| B(평가기 안 주입) = 오프라인 기준 | 105 | 102 | 104 | 311 |
| C(상한 5,143) = 기준 | 105 | 102 | 104 | **311** |

영향받은 4곡(`truncation_songs.csv`) — 전부 **가사 질의(alpha 0.10)**의 top_k=90 호출이고, 해당 곡은 sparse 상위·dense 전체 순위 5,027~5,097위(상한 밖)였다:

| 세트 | 질의 | 곡 | 기준 순위(top-90) | A 순위 | 기준 합산 | A 합산 | 빠진 dense 몫 |
|---|---|---|---|---|---|---|---|
| v09 | n017 | 32228058 | 76 | 76 | 0.2896 | 0.2636 | 0.0259 |
| v10 | t003 | 36310362 | 7 | 8 | 0.4846 | 0.4592 | 0.0255 |
| v10 | t016 | 34535898 | 59 | 65 | 0.6636 | 0.6434 | 0.0202 |
| v10 | t018 | 35637511 | 33 | 50 | 0.2992 | 0.2777 | 0.0215 |

왜 가사 질의에서만 드러나는가: alpha 0.10이면 dense 몫은 최대 0.1 안쪽이고 sparse(BM25×0.9)가 지배한다. sparse가 큰 곡은 dense가 최하위여도 top-90에 들고, 그 dense 몫(≈0.02)이 빠진다. 일반 질의(alpha 0.5~0.7)는 5,000위 dense 몫이 중앙값 0.166으로 커서 빠지면 큰 손실이지만, 그런 곡이 top-30/90에 들려면 sparse가 그만큼 커야 해서 dev에서는 한 번도 없었다. sparse 잘림 35건은 잘린 곡의 sparse 몫이 5,000위 값(≈0.0006~0.008)이라 top-k에 영향이 없었다.

### 5-2. 후보 30개 단계 (`compare_candidates.py` → `candidate_changes.csv` · `candidate_summary.json`)

분모: dev 176질의 전부(모두 정답 있음). 리랭킹 전 후보·순서 지표다 — **최종 서비스 Hit@10이 아니다.**

| 지표 | A(현행) | B(정확 합산) | C(5,143) |
|---|---|---|---|
| 후보 30개 ID·순서가 A와 다른 질의 | — | **0** | 0 |
| 후보 지문(id·순서·점수)이 **A와** 다른 질의 | — | 1 (t003, 5번째 후보 점수 0.05157→0.05179, 순서 불변) | 1 (같은 t003) |
| 후보 지문이 **B와** 다른 질의 | 1 (t003) | — | **0** (176건 모두 B와 같음) |
| 정답이 후보 30개에 포함 | 122 (69.3%) | 122 (69.3%) | 122 |
| 새로 회수한 정답 / 빠진 정답 | — | 0 / 0 | 0 / 0 |
| 리랭킹 전 Hit@1 / @5 / @10 | 38 / 73 / 101 | 38 / 73 / 101 | 동일 |
| 리랭킹 전 MRR@10 | 0.3149 | 0.3149 | 동일 |
| 세트별(v06·v09·v10) 정답 포함 | 43 · 41 · 38 | 43 · 41 · 38 | 동일 |

A rep1 vs rep2(v06 57질의): 후보 지문 57/57 동일 — 같은 입력에서 결과가 반복된다.

### 5-3. 지연·비용 (`bench_latency.py` → `latency_calls.csv` · `latency_summary.json`, 로컬 파일 모드 · 한 프로세스 · 임베딩 없음)

v06 A 실행이 기록한 필터 없는 하이브리드 호출 60건을 입력으로, 각 구성 예열 2회 뒤 호출마다 A→C→D→B / D→C→A→B 교대(동시 실행 없음).

| 구성 | 중앙값 ms | p95 ms | Qdrant 조회/호출 | 반환 점수 항목/호출 |
|---|---|---|---|---|
| A 상한 5,000(현행) | 231.5 | 328.3 | 2 | 9,079 |
| C 상한 5,143 — count는 준비 단계 1회(상한만 늘린 비용) | 230.8 | 321.6 | 2 | 9,238 |
| **D 요청마다 `count(exact)` + 상한 = 그 값(수정안 §7-A 모양, count 포함)** | 231.1 | 327.8 | 3 | 9,238 |
| B 메모리 행렬 기준(참고) | 1.5 | 2.1 | 1(payload) | 59 |

짝 차이(같은 호출): C−A 중앙값 +0.75ms(평균 −5.8, p95 +94 — 잡음, 음수 33%), **D−A 중앙값 +0.88ms**(평균 −0.9, p95 +96). D의 count 자체는 중앙값 0.021ms(로컬 파일 모드). 필터 호출의 `count(filter)`는 A 기록의 서로 다른 필터 9종 × 10회에서 **중앙값 17.1ms · p95 20.0ms**(통과 3~2,464곡) — 수정안이 필터 호출(dev 311건 중 16건)에서 추가로 내는 비용이다. 반환 항목 +1.7%. B의 1.5ms는 벡터를 전부 메모리에 올린 다른 방식의 수치라 운영 수정안의 비용이 아니다(§7-D 참고). **서버 모드의 count·필터 count 왕복 비용은 재지 않았다.**

준비 시간: 복사본 열기 1.29초, 기준 행렬 적재 0.71초. 메모리(프로세스 `ru_maxrss`): 열기 전 131MB → 복사본 열기 뒤 993MB(로컬 모드가 다섯 컬렉션을 메모리에 올림) → 기준 행렬 뒤 1,132MB(+139MB). 평가기 실행(모델 포함)은 A·B·C 모두 4.3~4.4GB로 구성 간 차이가 보이지 않았다.

평가기 안에서의 호출 시간(다른 경로와 동시 실행 중): A 중앙값 451ms · C 456ms · B 3.2ms; 질의당 `retrieval_ms`(임베딩 포함) 중앙값 A 535 · C 536 · B 80. 세트당 평가 시간 A 36~38초 · C 35~37초 · B 11~12초.

**서버 모드는 재지 않았다**(§8-5). 서버 모드도 같은 `query_points(limit=…)` 계약이라 잘림의 유무는 같지만, 5,000개 응답을 HTTP로 두 번 받는 비용의 크기는 다르다(v56 서버 측정 참고).

### 5-4. 상한 민감도 모사 (`sensitivity_limit.py` → `sensitivity_limit.json`)

"지금 dev에서 차이 없음"이 "안전"이 아님을 보이기 위해, **현재 코퍼스(5,143곡)에서 상한 L만 낮춘** 효과를 기준 표의 순위로 모사했다(dense 순위 ≤ L이면 dense 몫, 일치 곡 중 sparse 순위 ≤ L이면 sparse 몫, 둘 다 밖이면 사라짐). L=5,000 모사가 실제 A 결과와 **295/295 호출에서 일치**해 모사가 맞음을 먼저 확인했다. 필터 없는 295호출, top-30·top-90을 기준과 대조. **이 표는 "코퍼스가 N곡 늘면"의 예측이 아니다** — 새 곡은 dense 점수 분포와 BM25 통계(IDF)를 함께 바꾸므로 같은 비율이라도 결과가 다를 수 있다. 읽을 수 있는 것은 "상한이 대상 수보다 작아지는 정도에 따라 누락이 빠르게 커진다"는 방향이다.

| 상한 L | L/코퍼스 | top-30 집합·점수 바뀐 호출 | top-30에서 사라진 곡(호출) | top-90 바뀐 호출 | top-90에서 사라진 곡 |
|---|---|---|---|---|---|
| 5,143 | 1.000 | 0 | 0 (0) | 0 | 0 |
| **5,000(현행)** | 0.972 | 1 | 0 (0) | 7 | 1 |
| 4,500 | 0.875 | 17 | 10 (8) | 31 | 39 |
| 4,000 | 0.778 | 28 | 23 (18) | 65 | 90 |
| 3,000 | 0.583 | 58 | 79 (49) | 117 | 301 |

지금은 상한이 코퍼스의 97.2%라 영향이 top-90 끝자리·점수 1건에 그치지만, **현재 코퍼스에서 상한을 4,500으로 낮추면 8개 호출에서 top-30 곡이 사라지고, 4,000이면 18개 호출**이다. 상한이 대상 수에 묶여 있지 않으면 어느 코퍼스 크기에서 이런 구간에 들어가는지 미리 알 수 없다 — 그래서 상한을 숫자로 올리는 대신 대상 수로 맞추는 수정을 권한다.

## 6. 외부 호출 차단 (`v61_guard.py`)

서비스 모듈 import 전에 `make_genai_client`·`google.genai.Client`·`anthropic.(Async)Anthropic`·`pymongo.MongoClient`·`src.common.mongodb._connect`를 예외로 바꾸고, `QDRANT_URL`·`CLAUDE_*`·`GEMINI_API_KEY`를 제거, `HF_HUB_OFFLINE=1`. 실행 전후 `verify()`가 전부 `blocked`임을 확인해 각 `conditions.json`에 적었다(10회 실행 모두 통과). 평가기 runinfo의 `lyrics_source.error = BlockedExternalCall`이 Mongo가 실제로 막혔음을 보인다. 리랭커는 꺼져 있어(`enabled: False`) Vertex 사전 점검도 실행되지 않았다(`gemini.backend = unused`).

## 7. 수정안 비교와 권고 (운영 코드 미변경)

| 방식 | 정확성 조건 | 조회량 | 메모리 | 필터 | 코퍼스 증가 · 동시 색인 변경 |
|---|---|---|---|---|---|
| **A. 실제 대상 수를 반영한 조회** — 하이브리드 조회 전에 `count(collection, exact=True)`(필터가 있으면 `count(filter)`)로 대상 수 N을 얻어 두 조회의 `limit=N` | N이 조회 시점의 대상 수와 같으면 전수 → 정확(이 자료에서 상한 5,143 재생 = 기준 311/311, §8-4). **count와 조회 사이에 적재가 끼면 그 요청은 잘릴 수 있다 — 이것을 반환 건수로 감지할 수는 없다**(전수 조회도 정확히 N개를 돌려준다). 동시 적재 감지·대응은 별도 설계(적재 중 조회 차단, 세대/별칭 전환 뒤 조회 등) | 지금과 같은 2회 + count 1회. 로컬 count 0.02ms, 필터 count 약 17ms(§5-3). 반환 항목 N×2(지금 5,000×2 → 5,143×2) | 지금과 같은 수준(+2.9%) | count에 같은 필터를 주면 그대로 정확 | 곡이 늘어도 자동으로 따라간다. **대상 수에 비례해 반환이 늘어나는 구조는 그대로**(1만 곡이면 2만 항목) — 그 비용이 문제가 되면 그때는 B·D 같은 다른 설계가 필요하다 |
| B. 페이지네이션 합산 — `query_points`에 `offset`을 주며 끝까지 받기 | 모든 페이지를 중복·누락 없이 받으면 정확. Qdrant 점수 정렬은 결정적이지만 **페이지 사이에 적재가 끼면 중복·누락 가능**(스냅샷 없음) | 페이지 수만큼 왕복(5,143/1,000 → 6회 × 2). 로컬 모드는 `offset`이 내부에서 상위 offset+limit를 다시 계산해 **합계 비용이 A보다 크다** | 한 번에 쥐는 항목은 작아지지만 합산 사전은 결국 N개 | 그대로 | 곡 수에 비례해 왕복이 는다. 끝 판정(빈 페이지)을 반드시 검사해야 한다(§3의 scroll 검사와 같은 종류) |
| C. 누락 점수 보완 — dense·sparse 상위 L만 받고, 한쪽에만 있는 곡의 다른 쪽 점수를 `retrieve(with_vectors)`로 읽어 코드에서 내적 | 보완 대상이 "양쪽 상위 L에 들어온 곡"에 한정되므로 **양쪽 모두 밖인 곡은 여전히 사라진다**(§2 7번 사례) — 정확하지 않다. 정확해지려면 합산 상위 top_k의 하한을 증명해야 하는데, 한쪽 점수 상한(L위 점수)을 이용한 WAND식 경계 계산이 필요 | 상위 L 2회 + 보완 벡터 조회(한쪽만 있는 곡 수, 수천 건) | 벡터를 받아 와 내적하므로 항목당 비용이 크다(dense 1,024차원) | 그대로 | 경계 계산 없이는 설명 가능한 정확성이 없다. **권하지 않는다** |
| D. 메모리 행렬(이번 기준 구현 방식) — 기동 시 전체 벡터를 메모리에 올려 numpy로 합산 | 색인과 메모리 판이 같을 때만 정확. 적재·별칭 전환 때 다시 올려야 한다 | Qdrant 조회 0회(payload만) | 5,143×1,024 f32 ≈ 21MB + sparse ≈ 10MB(지금), 곡 수에 비례 | 필터는 Qdrant로 id 집합을 받거나 payload를 메모리에 함께 올려야 한다 | 색인 변경 감지·재적재가 새 운영 복잡도. 1.5ms는 매력적이지만 **벡터 DB를 우회하는 설계 변경**이라 이 결함의 "가장 작은 수정"이 아니다 |
| E. 상한만 10,000으로 | 코퍼스 < 10,000일 때만 | 지금과 같은 구조 | 같음 | 같음 | **같은 결함을 뒤로 미룬다**(§5-4). 문서 주석도 그대로 틀린다 |

**권고: A.** 근거는 §5-1(C = 기준 311/311)·§8-4(상한 5,143 재생 = 기준 311/311 — 상한을 대상 수로 맞추면 이 자료에서 기준과 같다)와 §5-3(요청별 count 포함 D−A 중앙값 +0.9ms, 로컬)이다. 구현은 `QdrantIndex.query`의 하이브리드 분기에서 `limit`을 상수 대신 **대상 수**(`count(exact)`, 필터가 있으면 같은 필터의 count)로 두는 것이다. `QDRANT_HYBRID_SCAN_LIMIT`은 제거하거나, 두면 "대상 수가 이 값보다 크면 경고"하는 **안전 상한**으로만 쓴다(조회 상한으로 쓰지 않는다). count를 컬렉션 단위로 캐시하면 적재 뒤 첫 조회가 잘릴 수 있고 그것을 반환 건수로는 감지하지 못하므로, 캐시하지 않거나 적재·별칭 전환과 묶인 무효화가 있어야 한다 — 이 부분(동시 색인 변경)은 별도 설계로 남긴다. 함께 고칠 것: 모듈 주석("905곡"), `docs/vector_backend.md`("잘림이 없다"), `.env.example` 주석. 서버 모드의 count 왕복·5,143×2 응답 비용은 수정 뒤 실측 대상이다.

## 8. 판단 (나눠서)

1. **합성 사례에서 구현의 제한을 확인했는가 — 예.** 상한 < 자료 수일 때 한쪽 점수 누락(dense·sparse 양방향)과 양쪽 목록 밖 곡의 소실, 음수 dense 몫 누락이 점수를 올리는 방향까지 결정적으로 재현(§2, 테스트 11건)
2. **현재 dev 자료에서 실제 후보·정답 회수 영향이 있었는가 — 후보 30개 ID·순서·정답 회수 영향은 없었다.** 영향은 텍스트 top-90 안의 점수 4건(순서 3건)과 그중 1건이 융합 점수에 미친 0.0002 차이(순서 불변)에 그쳤다. 단 이 비교는 (a) Mongo 가사 경로가 빈 조건, (b) dev 176질의, (c) 5,143곡·상한 5,000(97.2%) 조건이다. **모든 질의에서 문제가 없다는 뜻이 아니다** — §5-4에서 현재 코퍼스의 상한만 낮춰도 top-30 소실이 바로 생긴다
3. **후속 운영 수정이 필요한가 — 필요하다.** 결함은 자료 수가 상한을 넘는 순간부터 질의에 따라 조용히 커지며, 지금은 코퍼스가 상한을 2.9% 넘어 영향이 작은 구간이다. 권고 §7-A. 단순히 상한을 10,000으로 올리고 끝내지 않는다
4. **수정 뒤 어떤 검증이 필요한가** — (가) **`replay_calls.py`**: 저장된 질의 벡터 311건을 **수정된 구현**에 다시 넣어 결과를 새로 만들고 독립 기준과 대조한다(`compare_text_calls.py`는 저장된 A/B/C 결과만 읽으므로 수정 코드 검증이 되지 않는다). 이번 코드로 돌린 결과: 현행 상한 5,000 → 차이 4건(§5-1과 같은 네 곡, `replay_current.json`), `QDRANT_HYBRID_SCAN_LIMIT=5143` → **0건**(`replay_limit5143.json`). 수정 코드는 환경변수 없이 0건이어야 한다. (나) 합성 테스트 11건 통과 + 수정 코드의 상한 결정(대상 수·필터 count) 단위 테스트 추가, (다) 수정 코드로 `run_candidates.py --arm A`를 **다른 `--out-root`**(예: `/private/tmp/vague-finder-hybrid-scan-validation`)에 다시 돌려 후보 30개 지문을 이번 A·B와 비교한다 — 기본 출력 경로로 돌리면 이 폴더의 수정 전 A 기록을 덮어쓰므로, `run_candidates.py`는 산출물이 있는 폴더에는 `--overwrite` 없이 쓰지 않는다. 비교는 `compare_candidates.py --baseline-root <이 폴더> --baseline-arm A --compare-root <새 폴더> --compare-arm A --no-reference --suffix fixed`(열 이름 A=수정 전, B=수정 후). 가사 경로가 빈 같은 조건에서는 t003 점수 1건만 바뀌고 순서는 전부 같아야 한다 — 다르면 다른 원인, (라) 서버 모드(독립 로컬 Docker Qdrant + 복사본)에서 count 왕복·반환 5,143×2의 지연 실측(v56 방식), (마) **품질 재측정 여부는 (다)를 가사 경로(Mongo)가 있는 조건으로 한 번 더 확인한 뒤 결정한다** — 이번 결과는 가사 경로가 빈 조건이라 그것만으로 "재측정 불필요"를 단정할 수 없다. 리랭킹 전 후보 30개가 서비스 조건에서도 같다면 리랭킹 뒤 숫자도 같으므로 재측정이 필요 없고, 다르면 바뀐 질의만 다시 잰다

## 9. 한계

1. **Mongo 가사 표면 경로를 막은 조건**이라 가사 단서 질의 19건의 후보 구성이 서비스와 다르다. 하이브리드 조회 자체는 영향이 없지만 "후보 30개 동일"은 이 조건에서의 결과다. 절대 지표는 v50과 비교하지 않는다
2. 서버 모드(HTTP)는 재지 않았다. 잘림 유무는 같은 계약이지만 비용 크기(5,000개 응답 2회·count 왕복)는 다르다
3. 지연은 로컬 파일 모드 한 프로세스 60호출이며 동시 요청은 없다. p95 차이(+94~96ms)는 잡음 수준의 표본이다. D의 count는 필터 없는 호출만 요청별로 포함했고, 필터 count는 따로 잰 값이다
4. 민감도 표(§5-4)는 **현재 코퍼스에서 상한만 낮춘 순위 모사**다(L=5,000에서 실제와 295/295 일치로 검증) — 경계 동점은 무시했고, 새 곡 추가 상황(점수 분포·BM25 통계 변화)의 예측이 아니다
8. count와 조회 사이의 동시 적재로 생기는 잘림은 반환 건수로 감지할 수 없다(전수 조회도 정확히 limit개를 돌려준다). 그 감지·대응은 이번 범위 밖이다
5. 필터 호출 16건은 통과 수가 모두 상한 아래라 "필터 + 상한 잘림" 조합은 실제 자료로 보지 못했다(합성 사례 §2로만)
6. v10 test(봉인)는 쓰지 않았다. 질의별 예외·정답 하드코딩 없음
7. 메모리는 프로세스 `ru_maxrss`만 재서 구성 간 차이(5,000 vs 5,143 항목)는 보이지 않는다

## 리뷰 반영 (2026-10-16 밤, 측정 조건 변경 없음)

| 지적 | 반영 |
|---|---|
| "반환 건수 == limit"은 잘림 감지 조건이 못 된다(limit=count면 전수 조회도 정확히 limit개, C의 필터 없는 295건 전부 해당) | §0·§7-A·§9-8에서 감지 조건·정확성 보장 서술을 제거. 동시 적재 감지·대응은 별도 설계로 남김 |
| 측정한 C는 요청마다 count하는 수정안이 아니다(count는 준비 단계 1회) | `bench_latency.py`에 **D(요청마다 `count(exact)` + 상한=그 값, count 비용 포함)**와 필터 count(9종 × 10회) 추가, 다시 측정(§5-3). 서버 모드 count 비용은 미측정으로 명시 |
| "약 600곡 늘면 누락"은 이번 모사로 입증되지 않는다(새 곡은 점수 분포·BM25 통계를 바꿈) | §0·§5-4를 "현재 코퍼스에서 상한을 4,500으로 낮추면 8호출 top-30 누락"으로 한정, 600곡·11,400곡 예측 삭제 |
| `compare_text_calls.py` 재실행은 수정 코드를 검증하지 못한다(저장 결과만 읽음); 가사 경로 빈 결과로 품질 재측정 불필요 단정 불가 | **`replay_calls.py`** 추가 — 저장 질의 벡터를 지금 디스크의 구현에 재생해 기준과 대조(현행 4건 · `QDRANT_HYBRID_SCAN_LIMIT=5143` 0건). §8-4를 재생 → 후보 지문 재생성(가사 경로 있는 조건 포함) → 그 뒤 재측정 여부 결정으로 수정 |
| §5-2 표의 C 지문 변화 기준이 섞임 | A 대비 1건(t003) · B 대비 0건으로 행을 나눔 |
| (2차) 수정 코드 검증 명령이 기본 출력 경로(`cand_A/<set>_dev`)에 써서 수정 전 A 기록을 덮어씀 | `run_candidates.py`가 산출물이 있는 폴더에는 `--overwrite` 없이 쓰지 않도록 중단 조건 추가(확인: 기본 경로 재실행 → 중단). `compare_candidates.py`에 `--baseline-root/--compare-root/--reference-root`·`--*-arm`·`--no-reference`·`--suffix`를 추가해 기준선과 새 결과를 각각 받음(기본 인자 결과는 이전과 동일 확인). §8-4(다)·재현 절을 새 `--out-root` 예시로 교체 |

## 재현 (모델 호출 없음, 저장소 루트에서, `env -u CLAUDE_EFFORT`)

```
# 합성 사례 · 회귀 테스트
venv/bin/python -m pytest tests/test_v61_hybrid_scan_audit.py -q
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/repro_synthetic.py
# 색인 복사본 준비(한 번): cp -R artifacts/qdrant artifacts/qdrant_v61_audit && rm artifacts/qdrant_v61_audit/.lock
# dev 176건 A/B/C (순차, 약 6분) → cand_*/<set>_dev/
zsh experiments/reranking/results_v61_hybrid_scan_audit/run_all.sh
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/compare_candidates.py     # candidate_changes.csv · candidate_summary.json
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/compare_text_calls.py     # text_calls_diff.csv · truncation_songs.csv · text_calls_summary.json
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/bench_latency.py --set v06 --max-calls 60   # latency_calls.csv · latency_summary.json (A·C·D·B + 필터 count)
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/sensitivity_limit.py      # sensitivity_limit.json
# 수정 코드 검증 — 저장 질의 벡터를 지금 디스크의 구현에 재생해 독립 기준과 대조(수정 뒤에는 이것을 돌린다)
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/replay_calls.py --label current            # 현행: 차이 4건
QDRANT_HYBRID_SCAN_LIMIT=5143 venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/replay_calls.py --label limit5143   # 0건
# 수정 코드로 후보 30개 재생성 — 기준선(cand_A 등)을 덮어쓰지 않도록 반드시 다른 --out-root (산출물이 있는 폴더는 --overwrite 없이 거부)
for s in v06 v09 v10; do venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/run_candidates.py --arm A --set $s \
    --out-root /private/tmp/vague-finder-hybrid-scan-validation; done
venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/compare_candidates.py \
    --baseline-root experiments/reranking/results_v61_hybrid_scan_audit --baseline-arm A \
    --compare-root /private/tmp/vague-finder-hybrid-scan-validation --compare-arm A --no-reference --suffix fixed
# → candidate_changes_fixed.csv · candidate_summary_fixed.json (A=수정 전 기준선, B=수정 후)
```

## 파일

| 파일 | 내용 |
|---|---|
| `v61_guard.py` | 외부 호출 차단·실행 조건 고정·차단 확인 |
| `exact_hybrid.py` | 독립 정확 합산 기준(scroll 전수 읽기·정합 검사·지문·점수표·동점 처리 비교) |
| `hybrid_capture.py` | `QdrantIndex.query` 감싸기 — 기록(A·C)·기준 주입(B)·후보 30개 기록 |
| `run_candidates.py` · `run_all.sh` | 기존 평가기로 후보 30개 실행(A/B/C · 세트 · rep) |
| `compare_text_calls.py` · `compare_candidates.py` · `bench_latency.py` · `sensitivity_limit.py` | 호출 대조(저장 결과) · 후보 대조 · 지연(A·C·D·B, 필터 count) · 민감도 |
| `replay_calls.py` · `replay_current.json` · `replay_limit5143.json` | 저장 질의 벡터를 **현재 구현**에 재생해 기준과 대조 — 수정 코드 검증용 |
| `repro_synthetic.py` · `repro_synthetic.json` · `tests/test_v61_hybrid_scan_audit.py` | 합성 재현·회귀 테스트 11건 |
| `cand_{A,B,C,A_rep2}/<set>_dev/` | 평가기 산출(summary·top10·runinfo 커밋, detail·explain 제외) + `conditions.json`(지문·차단 확인·시간·메모리). `calls.jsonl`·`candidates.jsonl`·`calls_vectors.npz`는 커밋 제외 |
| `candidate_changes.csv` · `candidate_summary.json` · `text_calls_diff.csv` · `truncation_songs.csv` · `text_calls_summary.json` · `latency_calls.csv` · `latency_summary.json` · `sensitivity_limit.json` | 결과 표(곡 ID·질의 ID·숫자만) |

색인 복사본 `artifacts/qdrant_v61_audit`(230MB)·실행 로그 `run_*.log`·`progress.txt`는 커밋하지 않는다.
