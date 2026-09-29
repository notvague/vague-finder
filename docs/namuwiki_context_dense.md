# 나무위키 context KoE5 임베딩

## 이 단계에서 추가되는 기능

`artifacts/context/songs/*.json`에서 검증된 각
`retrieval.records[].dense_text`를 1024차원 KoE5 벡터 하나로 변환한다.
기존 곡 단위 분위기·가사 인덱스는 변경하지 않으며, 아직 Qdrant에
벡터를 업로드하지도 않는다.

검색 단위는 다음처럼 유지한다.

- dense: context 사실 하나(`record_id`)당 벡터 하나
- sparse: 곡 하나당 중복 제거된 어휘 프로필 하나(후속 단계)
- 최종 결과 식별자: `song_id`
- dense 검색 후 곡 점수: 같은 곡에서 가장 높은 사실 점수(`max_per_song`)

각 dense 사실은 다음처럼 독립적으로 이해할 수 있는 최소 정체성 prefix를
포함한다.

```text
윤도현 - 사랑했나봐 | 매체 사용: 짱구는 못말려에서 창밖을 보며 눈물을 흘리는 나미리 선생님과 함께 곡이 흐른다.
```

형식은 `가수 - 곡명 | 카테고리: 사실`이다. 제목·가수는 sparse 문서마다
반복하지 않지만, 사실별 dense 벡터에는 반드시 포함한다. 벡터 옆의 metadata는
KoE5 입력으로 보이지 않기 때문에 이 prefix가 없으면 가수명이 우연히 들어간
엉뚱한 사실이 실제 관련 사실보다 높은 점수를 받을 수 있다.

한 곡의 모든 사실을 한 문단으로 합치면 드라마 사용, 제작 일화, 밈처럼 서로
다른 단서가 한 벡터에서 희석된다. 사실 단위 벡터는 정확히 맞은 근거 문장을
찾으면서도 최종적으로는 곡을 반환하게 해준다.

## 검증 정책

전체 카탈로그 완료 검사는 나머지 메타데이터·나무위키 크롤링이 끝날 때까지
미뤄도 된다. 그러나 무결성 검사는 미루지 않는다. 모든 임베딩 실행은 모델을
로드하거나 파일을 쓰기 전에 context manifest, artifact 스키마와 self-hash,
manifest-artifact 연결, 레코드 수, 원본 사실 100% coverage를 검증한다.

파일럿 실행은 `coverage.pending > 0`을 허용한다. 전체 카탈로그의 최종 실행은
반드시 `--require-complete-context`를 사용한다. 이 옵션은 pending, coverage
오류, invalid artifact, orphan artifact가 하나라도 있으면 실행을 거부한다.

## 실행 명령

저장소 루트에서 backend 컨테이너 안의 명령을 실행한다.

```bash
# 검증과 작업량 확인만 한다. KoE5를 로드하거나 파일을 쓰지 않는다.
docker compose exec backend python -m src.embedding.cli.embed_context_dense --dry-run

# 전체 크롤링을 기다리는 동안 현재 완료된 곡만 파일럿 임베딩한다.
docker compose exec backend python -m src.embedding.cli.embed_context_dense

# 전체 context 크롤링 완료 후 최종 임베딩한다.
docker compose exec backend python -m src.embedding.cli.embed_context_dense \
  --require-complete-context
```

첫 실제 실행에서는 `nlpai-lab/KoE5`를 다운로드할 수 있다. 기본 차원은
1024이며 모델 출력 차원이 다르면 즉시 중단한다. 최종 빌드는
`--model-revision`에 Hugging Face revision/commit을 지정하면 재현성을 더
엄격하게 고정할 수 있다.

`context_artifact_v5`에서 정체성 prefix가 포함된 v6로 갱신한 경우에는 먼저
나무위키 원문을 다시 받지 않고 기존 `meta.json`으로 artifact만 재생성한다.

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context \
  --song-ids 837567 1698598 --artifacts-only
```

그다음 평소와 같이 dense 임베딩 명령을 실행한다. artifact hash와 문장 hash가
달라졌으므로 기존 NPZ는 자동으로 무효화되어 두 곡만 다시 임베딩된다.

주요 옵션:

```bash
--context-dir PATH
--output-dir PATH
--batch-size 32
--model-name nlpai-lab/KoE5
--model-revision REVISION
--force
```

일반적인 이어받기 실행에는 `--force`를 사용하지 않는다. 재실행 시 변경되지
않은 곡 bundle은 검증 후 재사용하고, 새 곡·변경된 곡·손상된 곡만 다시
임베딩한다.

## 출력 계약

기본 출력 경로:

```text
artifacts/embeddings/context_dense/nlpai-lab__KoE5/
├── manifest.json
└── songs/
    └── <song_id>.npz
```

사실마다 파일을 만들지 않고 곡마다 NPZ 하나를 사용하므로 작은 파일 수만 개가
생기는 문제를 피하면서도 사실별 벡터는 유지한다. 각 bundle은 다음 값과
결합된다.

- context artifact content hash
- 임베딩 설정 hash
- 순서가 고정된 `record_id` 목록
- 각 `dense_text`의 SHA-256

벡터는 유한한 `float32`, `(record_count, 1024)` 형태, L2 정규화 상태여야
한다. bundle과 전역 manifest는 원자적으로 저장된다. 실행 도중 중단돼도 다음
실행에서 이미 완성된 곡 bundle을 재사용하며, 후속 단계는 새 전역 manifest가
완성되기 전까지 이전에 게시된 manifest만 읽는다.

향후 벡터 DB 적재기는 `iter_context_dense_embeddings()`를 사용한다. 이 로더는
바뀐 context 입력, 변경된 문장, 손상 파일, 잘못된 차원, 정규화되지 않은 벡터,
manifest/hash 불일치를 거부한다.

## 이후 검색 연결 방식

context는 기존 song-level 인덱스와 분리된 context 인덱스/namespace에 둔다.
검색 시에는 다음 순서로 사용한다.

1. 매체 사용·제작·무대·밈·뮤직비디오·기록·음악적 특징 같은 배경 단서가 있는
   질의만 context 검색으로 보낸다.
2. 같은 KoE5 모델과 `query: ` prefix로 자연어 질의를 임베딩한다.
3. 사실 벡터를 cosine/dot-product로 검색한다.
4. `song_id`별로 묶어 최고 사실 점수를 사용하고, 곡당 근거 사실 수도 작게
   제한한다.
5. context dense 순위와 별도로 학습한 context BM25 순위를 먼저 결합한다.
6. context 곡 순위와 기존 song-level 검색 순위를 결합한다.
7. 최종 결과에 일치한 `fact_text`, category, section, source URL을 근거로
   제공한다.

한 곡의 사실 점수를 모두 합하면 나무위키 내용이 많은 곡이 부당하게 유리해지므로
합산하지 않는다. context 사실을 기존 song-level hybrid 인덱스에도 섞지 않는다.
두 인덱스는 point ID와 집계 단위가 다르다. context BM25 corpus 생성 방법은
[`namuwiki_context_sparse.md`](namuwiki_context_sparse.md)를 따른다. DB 업로드,
질의 라우팅과 순위 결합은 후속 구현 단계다.
