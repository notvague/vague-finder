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

## 검색 연결 현황과 다음 단계

`ContextQdrantSearch.search_song_candidates(query, fact_k=50, sparse_k=50)`는
한 번 확인한 적재 세대에서 Dense 사실과 Sparse 곡 프로필을 조회한다. Dense는
곡별 최고 점수의 사실을 보관하고, Sparse는 조회된 곡의 `song_id`, `profile_id`,
점수와 임시 표시 메타데이터를 순서대로 보관한다. `song_k`를 지정하면 Dense
**사실 검색 이후** 곡 수만 제한한다. Sparse 프로필 일치만으로는 특정 사실을
증명할 수 없으므로 출처나 `fact_text`를 부여하지 않는다.

`ContextQdrantSearch.search_fused_songs(query, fact_k=50, sparse_k=50)`는
두 **곡 순위**를 내부 RRF로 결합해 Context 후보 목록 **하나**를 만든다.
기본 점수는 `1/(60 + Dense 곡 순위) + 1/(60 + Sparse 곡 순위)`이고,
해당 경로에 없는 곡은 그 경로의 항이 없다. 원래의 Dense 유사도와 BM25
점수는 더하지 않는다. 같은 곡이 각 순위에 여러 번 있으면 중복 가산을
막기 위해 오류를 낸다. `dense_weight`와 `sparse_weight`는 별도로 조정할
수 있고, `limit`는 내부 결합 **이후**에만 적용한다.

반환값은 곡마다 `song_id`, 내부 RRF 점수, 두 경로의 순위와 원본 hit를
보관한다. Sparse에서만 발견한 곡에는 Dense 사실이 없다. Dense fact가
있더라도 이는 아직 표시가 검증된 근거가 아니다. `SearchRouter`는
`context_clues`에 양수 확신도가 있을 때 이 결과를 **한 개의 Context 경로**로
후보 절단 전에 결합한다. 동일 검색 문장은 한 번만 조회하고, 여러 단서에서 같은
곡이 나오면 가장 강한 단서만 유지한다. 바깥 RRF에는 내부 Dense/Sparse 점수가
아닌 최종 Context 곡 **순위**만 사용한다.

Context 곡 ID는 Text Qdrant 인덱스에서 다시 조회한다. 해당 ID가 없거나 제목이
비어 있으면 후보로 넣지 않는다. 결과의 제목·가수 등은 Context 프로필의 임시
표시 값이 아니라 기존 곡 인덱스의 정식 메타데이터다. Context 점수는 기존
Text/Image/Audio 가중치 합계에 **추가**하므로 Context에 없는 곡의 기존 점수는
변하지 않는다. Context 조회 실패는 다른 검색 경로의 실패 처리와 동일하게 빈
경로로 처리하고 실행 기록에 실패 사유를 남긴다.

현재 SearchRouter 기본값은 Context 경로 가중치 1.0에 단서의 최대 확신도를
곱하며, `fact_k=100`, `sparse_k=100` 이상을 읽는다. Dense 폭은 곡 수가 아닌
**사실 수**다. 이는 현재 부분 코퍼스에서 단서 곡이 Dense 순위 58위까지
내려간 사례를 놓치지 않기 위한 임시 기본값이며 전체 평가 후 조정한다.
`SearchRouter.search(..., use_context=False)`로 재랭킹 없는 전후 비교가 가능하다.
기존 세 모달리티 전용 `force_weights` 실험은 Context를 끈다.

실제 데이터 검증은 backend를 중지한 뒤 제공한
`Vague-Finder_Context_Router_Step6_Inspect.py`를 일회성 컨테이너에 표준 입력으로
전달해 실행한다. 기존 경로만 사용했을 때와 Context를 켰을 때의 후보 30개,
Top-10, 목표곡 위치, Text 인덱스 제목, 거절 후 후보 수를 출력한다.

결과 화면에서는 Context 후보라는 이유만으로 출처를 붙이지 않는다. 검색에 사용한
같은 Qdrant 조회에서 Dense 사실들을 보관하고, 반환할 곡에 한해 질의 단서의 관계,
작품 또는 인물, 구체적 장면 단서, 사실 유형, `song_id`, 레코드 ID, 나무위키
원문 URL을 확인한다. 가장 높은 Dense 사실이 맞지 않으면 같은 곡의 다른 검색
사실을 확인한다. 여러 Context 단서가 같은 곡을 찾았을 때도 각 단서로 확인하되
랭킹 표는 한 번만 가산한다. 확인되지 않으면 `context_evidence=null`이다.
Sparse 프로필 단독 일치, 절 제목만 일치하거나 출처가 없는 사실은 근거로 쓰지
않는다. 이 판정은 질의와 **검색된 문장의 관련성**을 확인하는 보수적 규칙이며,
외부 문서 내용 자체의 진위 보장은 아니다. 근거 검증 실패로 기존 검색 결과가
실패하지 않도록 근거만 생략한다. 검증된 근거가 있으면 `MatchingTrack`의
`context_evidence`에 `record_id`, `fact_text`, `source_url`, `section`,
`category`를 채우고, 지도 검색 결과의 사실 요약과 원문 링크에 표시한다.
이는 `explain`을 요청하지 않아도 제공된다.

다음 단계는 부분 코퍼스 Context 평가와 기존 질의 회귀 검사다. 그 후 전체 수집과
재적재를 마치고 corpus 기준으로 검색 폭·가중치를 조정한다.

두 곡 파일럿은 파이프라인 배선과 무결성만 검증한다. 최종 top-k, router threshold,
RRF 상수와 context 가중치는 전체 corpus와 모호 질의 평가셋이 준비된 뒤 고정한다.
