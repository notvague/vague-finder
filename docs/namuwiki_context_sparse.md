# 나무위키 Context BM25 corpus

## 이 단계의 목적

`artifacts/context/songs/*.json`의
`retrieval.sparse_profile.terms`를 곡당 하나의 sparse 문서로 보고, 현재
manifest에 포함된 **모든 프로필을 한 번에** BM25로 fit한다.

기존 `artifacts/bm25_params.json`은 가사·분위기 검색용 corpus 통계다.
나무위키 context는 제작 비화, 매체 사용, 밈, 무대, 기록처럼 분포가 전혀 다른
단서이므로 같은 파라미터에 섞거나 기존 파일을 덮어쓰지 않는다.

검색 단위는 다음과 같다.

- dense: 사실 하나당 KoE5 벡터 하나
- sparse: 곡 하나당 BM25 문서 하나
- 최종 연결 키: `song_id`

## 왜 전체 corpus가 필요한가

BM25의 핵심은 각 단어가 전체 곡 중 몇 곡에 등장하는지 나타내는 document
frequency다. `한국`, `노래`, `1위`처럼 많은 곡에 등장하는 단어는 낮게, 특정
방송·캐릭터·인물·사건·음역처럼 드문 단서는 높게 반영된다.

따라서 새 곡이 하나라도 추가되거나 기존 `sparse_profile`이 바뀌면 document
frequency가 달라진다. dense처럼 변경된 곡만 처리하지 않고, 작은 context sparse
corpus 전체를 다시 fit하고 모든 문서 벡터를 다시 만든다. 수천 곡 × 최대 80개
단어 규모이므로 비용은 KoE5 임베딩보다 매우 작다.

문서 프로필은 이미 Kiwi 기반 정제가 끝난 토큰이다. 다시 일반 tokenizer에 넣으면
`짱구는_못말려`, `D-E-F#m` 같은 고유 단서가 갈라질 수 있으므로 context BM25는
토큰 배열을 그대로 사용한다. 문서는 동의어를 인위적으로 추가하지 않고, 질의에서만
`OST ↔ 삽입곡 ↔ 배경음악 ↔ BGM` 등의 제한된 동의어를 확장한다.

## 실행

`mmh3`가 직접 의존성으로 추가되었으므로 코드를 반영한 첫 실행에서는 backend
이미지를 다시 빌드한다. 이후 저장소 루트에서 실행한다.

```bash
docker compose up -d --build backend
```

```bash
# manifest/artifact 무결성과 현재 작업량만 확인한다. 파일을 쓰지 않는다.
docker compose exec backend python -m src.embedding.cli.fit_context_bm25 --dry-run

# 전체 context 수집을 기다리는 동안 현재 완료된 프로필로 파일럿 빌드한다.
docker compose exec backend python -m src.embedding.cli.fit_context_bm25

# 모든 곡의 context artifact 생성이 끝난 뒤 최종 corpus를 빌드한다.
docker compose exec backend python -m src.embedding.cli.fit_context_bm25 \
  --require-complete-context
```

회귀 테스트:

```bash
docker compose exec backend python -m pytest \
  tests/context/test_context_sparse_bm25.py \
  tests/context/test_context_dense_embedding.py \
  tests/context/test_context_artifacts_v4.py -q
```

주요 옵션:

```text
--context-dir PATH
--output-dir PATH
--b 0.75
--k1 1.2
--force
--dry-run
--require-complete-context
```

일반 재실행에는 `--force`를 사용하지 않는다. 현재 profile 목록·단어·설정과
params/bundle의 모든 hash가 일치하면 corpus를 재사용한다. 하나라도 달라지거나
파일이 손상되면 전체 corpus를 다시 fit한다.

## 출력

기본 경로:

```text
artifacts/embeddings/context_sparse/bm25/
├── manifest.json
├── params/
│   └── <source-hash>-<config-hash>.json
└── corpora/
    └── <source-hash>-<config-hash>.npz
```

- params JSON: `n_docs`, `avgdl`, document frequency, `b`, `k1`, tokenizer/hash 계약
- corpus NPZ: 곡별 sparse vector를 하나로 묶은 CSR 배열
- manifest: 입력 profile hash, 설정 hash, 출력 파일 hash, corpus 통계와 hash 충돌 통계

출력 파일은 source/config 기반 이름으로 먼저 완성하고 마지막에 manifest를 원자적으로
교체한다. 중단된 빌드의 불완전 파일을 검색 로더가 정상 결과로 읽지 않는다.

## 품질과 무결성 정책

- context manifest, 각 song artifact self-hash, 원본 사실 100% coverage를 먼저 검증
- `profile_id`와 `song_id` 중복 금지
- 프로필 내부 단어 중복과 빈 단어 금지
- 문서 sparse index를 정렬하고 중복 index를 합산
- params, corpus bundle, manifest의 SHA-256 연결 검증
- 현재 artifact가 바뀌면 이전 sparse corpus 로드를 거부
- 전체 최종 실행은 pending/coverage 오류/orphan artifact가 있으면 거부
- `top_document_frequency_terms`를 manifest에 남겨 전체 corpus에서 지나치게 흔한
  단어를 추후 점검할 수 있게 함

파일럿에서 `source_scope_complete=false`가 나오는 것은 정상이다. 단, 일부 곡만으로
학습한 IDF는 임시 통계이므로 서비스용 최종 sparse 검색에는 사용하지 않는다.

## 후속 검색 연결

`iter_context_sparse_embeddings()`는 각 곡의 `profile_id`, `song_id`, 원문 단어와
`sparse_values`를 검증 후 반환한다. `load_context_bm25_query_encoder()`는 빌드 때와
동일한 params와 질의 tokenizer를 로드한다.

향후 검색에서는 context dense와 context sparse를 각각 조회한 뒤 순위를 RRF로
결합한다. sparse 절대 점수는 질의 동의어 수와 corpus 크기에 영향을 받으므로 기존
가사 BM25 점수나 KoE5 점수에 그대로 더하지 않는다. context 후보를 `song_id`로
기존 멀티모달 검색 후보와 합치고, dense hit의 `evidence_text`를 사용자에게 근거로
제공한다.
