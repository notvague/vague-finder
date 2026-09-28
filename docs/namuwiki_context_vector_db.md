# 나무위키 Context Qdrant 적재

## 역할과 경계

이 단계는 이미 생성된 두 검색 입력을 Qdrant에 게시한다.

| 입력 | 검색 단위 | Qdrant 포인트 | 용도 |
|---|---:|---:|---|
| `records[].dense_text` KoE5 | 사실 | `record_id`당 1개 | 자연어로 비슷한 배경 사실 검색 |
| `sparse_profile` BM25 | 곡 | `profile_id`당 1개 | 작품명·인물·회차·밈 같은 정확 단서 검색 |

두 단위는 다르므로 기존 곡 검색 컬렉션이나 하나의 hybrid 컬렉션에 합치지 않는다.
Dense 결과는 이후 `song_id`별 최고 사실 하나로 집계하고, Dense/Sparse 순위는
RRF로 결합한다.

이 구현은 `src/embedding/text/passage_builder.py`와 기존 text/image/audio 검색 및
점수 계산을 변경하지 않는다.

## 만들어지는 컬렉션

기본 namespace가 `dev`이면 안정적인 진입점(alias)은 다음 두 개다.

```text
vaguefinder-context-dense-1024-koe5__dev
vaguefinder-context-sparse-bm25__dev
```

실제 데이터는 입력 hash가 포함된 세대 컬렉션에 먼저 적재된다. 모든 포인트 수,
ID, payload build ID와 벡터 설정 검증을 통과한 뒤 두 alias를 한 Qdrant alias
연산으로 교체한다. 중간에 실패하면 현재 alias는 바뀌지 않는다.

- Dense payload: 곡 ID, 제목·가수, category/section, 근거 문장, keywords, 출처
- Sparse payload: 곡 ID, 제목·가수, 정제된 sparse terms
- 동일 입력 재실행: 기존 세대를 검증한 뒤 `action=reuse`
- 입력 변경: 새 세대를 만들고 alias 교체; 이전 세대는 기본적으로 rollback용 보존
- 현재 corpus에서 삭제된 곡/사실: 새 세대에는 들어가지 않아 stale point가 사라짐

## 파일럿 2곡 실행

먼저 DB를 열지 않는 전체 입력 검증을 실행한다.

```powershell
docker compose exec backend python -m src.vector_db.cli.qdrant_load_context --dry-run
```

로컬 파일 Qdrant는 한 프로세스만 같은 경로를 열 수 있다. 실제 적재 때는 backend를
멈추고 일회성 컨테이너로 실행한다.

```powershell
docker compose stop backend

docker compose run --rm --no-deps backend `
  python -m src.vector_db.cli.qdrant_load_context

docker compose up -d backend
```

정상 파일럿 결과는 다음 조건을 만족한다.

- `status: ok`
- 첫 실행 `action: build`, 같은 입력 재실행 `action: reuse`
- 현재 데이터 기준 `source_song_count: 2`
- `dense_record_count: 39`, `sparse_profile_count: 2`
- `source_scope_complete: false`는 전체 곡 수집 전 파일럿에서는 정상

`artifacts/vector_db/context_qdrant/dev/manifest.json`은 활성 세대와 세 입력
manifest hash를 기록하는 로컬 감사 파일이다. Qdrant 데이터와 함께 git에는 올리지
않는다.

## 약 3천 곡 최종 실행

전체 context artifact, Dense KoE5, 전체 corpus BM25를 차례로 완성한 뒤 최종
gate를 붙인다.

```powershell
docker compose stop backend

docker compose run --rm --no-deps backend `
  python -m src.vector_db.cli.qdrant_load_context `
  --require-complete-context

docker compose up -d backend
```

`--require-complete-context`는 pending 곡이나 coverage 오류가 하나라도 있으면 DB를
열기 전에 실패한다. 파일럿에는 이 옵션을 붙이지 않는다.

적재는 기본 256포인트 배치 스트리밍이므로 전체 Dense 벡터를 RAM에 한꺼번에
올리지 않는다. 환경에 따라 `CONTEXT_QDRANT_BATCH_SIZE` 또는 `--batch-size`로
조정한다.

## 운영 옵션

- `--force`: 동일 입력도 별도 repair 세대로 다시 적재한다. 보통은 필요 없다.
- `--prune-old-generations`: alias 교체 성공 후 비활성 세대를 삭제한다. rollback이
  필요 없음을 확인한 뒤에만 사용한다.
- `--namespace`: 개발·평가·운영 컬렉션을 분리한다.
- `QDRANT_URL`: 서버 Qdrant 사용. 설정되면 `--qdrant-path`보다 우선한다.

로컬 적재에서 `storage is already accessed`가 나오면 backend가 아직 같은
`QDRANT_PATH`를 열고 있는 것이다. backend를 중지한 뒤 다시 실행한다.

## 다음 단계

DB 적재 자체는 검색 순위를 바꾸지 않는다. 후속 검색 통합에서 다음을 구현한다.

1. 배경 단서 질의를 판별하는 Context query router
2. KoE5 `query: ` 벡터로 Dense fact top-k 조회
3. 현재 Sparse manifest와 결합된 query encoder로 song profile 조회
4. Dense hit를 `song_id`별 `max_per_song`으로 집계
5. Context Dense/Sparse 순위를 RRF로 결합
6. 기존 text/image/audio 후보와 다시 RRF 또는 검증된 가중 방식으로 결합
7. 최종 결과에 `fact_text`, category, section, source URL을 근거로 노출

두 곡 파일럿은 파이프라인 배선과 무결성만 검증한다. 최종 top-k, router threshold,
RRF 상수와 context 가중치는 전체 corpus와 모호 질의 평가셋이 준비된 뒤 고정한다.
