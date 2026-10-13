# Context 질의 분석 리뷰 검증

## 변경 범위

- `나온 노래`와 `가요제`를 외부 사실 단서로 인식한다. 가사·앨범 표지·제목에
  같은 단어가 적힌 경우는 제외한다.
- 규칙이 이미 찾은 사건과 별개로, 모델이 찾아낸 **다른 절의** 외부 사건은
  원문 관계와 검색 문장을 확인한 후 신뢰도를 최대 0.5로 제한해 추가한다.
  규칙값과 동일한 사건을 중복 추가하지 않는다.
- 일반 수식어를 작품명으로 추정하지 않는다. 분명하지 않으면 `target=""`로
  두고 검색 문장은 보존한다. 모델 대상도 같은 절에 실제 등장해야 한다.
- Context 전용 프롬프트를 압축했다. 전달받은 코드 버전에서 템플릿 길이는
  25,987자 → 23,396자(−2,591자)이며, 같은 예시 질의로 렌더한 길이는
  25,793자 → 23,212자다. 실제 모델 지연 개선은 별도 실측 대상이다.

## 로컬 파일의 선행 조건

공개 레포에는 수집한 곡·원문·임베딩·Qdrant 파일을 넣지 않는다. 리뷰 증빙은
각자의 로컬 프로젝트 안에 다음 파일이 **같은 생성본으로** 있어야 한다.

- `artifacts/context/manifest.json` 및 해당 `songs/*.json`
- `artifacts/embeddings/context_dense/nlpai-lab__KoE5/manifest.json`
- `artifacts/embeddings/context_sparse/bm25/manifest.json`
- `artifacts/vector_db/context_qdrant/dev/manifest.json`
- `artifacts/qdrant/`의 실제 컬렉션

첫 번째 manifest가 없으면 loader가 원천 입력을 검증할 수 없다. Qdrant 디렉터리만
복사해서 해결할 수 없다. 이때는 개인용 데이터 보관소에서 같은 생성본을 복원하거나
수집 자료로 artifact → Dense → Sparse → Qdrant 순서로 다시 생성한다.

## PowerShell 재현 명령

프로젝트 루트에서 실행한다. 로컬 Qdrant는 동시에 두 프로세스가 열 수 없으므로
backend를 중지한다. `docker compose up -d backend`는 오류가 나도 실행된다.

```powershell
if (-not (Test-Path .\artifacts\context\manifest.json)) { throw 'Context 원천 manifest가 없습니다.' }
if (-not (Test-Path .\artifacts\vector_db\context_qdrant\dev\manifest.json)) { throw 'Qdrant 게시 manifest가 없습니다.' }

docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T backend python -m src.vector_db.cli.qdrant_load_context --dry-run
    if ($LASTEXITCODE -ne 0) { throw '원천 artifact/Dense/Sparse 검증 실패' }

    docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.review_context_proof `
        --query '크레용 신짱 OST 삽입곡' `
        --expected-song-id '<개인 평가셋의 정답 ID>' `
        --expect-fact-fragment '짱구는 못말려'
    if ($LASTEXITCODE -ne 0) { throw '실제 Qdrant 조회 검증 실패' }
} finally {
    docker compose up -d backend
}
```

`--expected-song-id`에는 팀 드라이브의 비공개 평가셋에 있는 값을 넣는다.
출력은 **제목·fact_text·URL을 보여주지 않고**, 전체 schema/hash/벡터/point ID
검증 수와 sparse·dense 조회 순위 및 사실 텍스트·출처 존재 여부만 보여준다.
`dense_index_self_probe`는 저장된 사실 벡터를 다시 조회하는 인덱스 무결성 검사다.
자연어 Dense 품질의 증거는 `expected_dense_rank`다. KoE5 모델을 이 환경에서
로드할 수 없을 때만 `--skip-dense`를 추가하며, 이 결과를 Dense 질의 성공으로
표현하지 않는다.

검증 당시 전달받은 부분 코퍼스에서는 3,016곡 중 2,535곡의 artifact, Dense
사실 3,995개, Sparse 곡 프로필 478개가 입력과 Qdrant에서 일치했다. 모든
artifact는 schema/hash 검증을 통과했고, 필수 Qdrant payload의 비어 있음·
`None`·`unknown`과 사실 출처 URL 누락은 모두 0개였다. 예시 검색의 Sparse 정답은 3위, 해당 곡의 사실
19개 모두 출처가 있었으며 요청한 사실 구절과 Dense self-probe가 확인됐다.
`scope_complete=false`다. 개인 컴퓨터의 새 생성본에서는 숫자와 순위가
달라질 수 있으므로 각자 실행 출력으로 리뷰에 증빙한다.

위 결과는 **파생 artifact와 Qdrant** 검증이다. 전달받은 ZIP에는
`data/raw/*/meta.json`이 없어서 원본 Namuwiki 필드 3,016건의 유효성은 이
환경에서 주장할 수 없다. 원본이 있는 PC에서는 다음 집계도 실행한다.
`not_found`의 `source_url=None`처럼 허용된 선택 필드와 필수 필드의 누락을
구분한다. 출력에는 곡 메타데이터가 없다.

```powershell
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.audit_raw_namuwiki_meta `
    --raw-dir data/raw
```

`schema_valid_namuwiki_objects`, 상태별 건수, `issue_counts`를 확인하고
원본 검증 결과를 리뷰에 추가한다. `issues_found`가 나왔다면 문제를 고쳐
artifact를 재생성하기 전까지 “원본 전체가 통과했다”고 적지 않는다.

기존 비공개 v0.5 평가 CSV로 **규칙 fallback**의 해당 질의만 점검할 수 있다.
이 출력에는 질의 본문·곡 제목·정답 ID를 포함하지 않는다.

```powershell
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.audit_reviewed_queries `
    --queries experiments/reranking/eval_queries_v05.csv
```

전달받은 CSV에서 `m101`, `m102`, `m104`는 기존 규칙 0개 → 변경 후 1개였다.
`m103`, `m105`, `q201`, `c607`, `c703`의 추정 작품명은 공백으로 바뀌었고,
`q203`은 시간 부사를 제거해 명시된 작품명만 보존했다. 모델이 실제로 뽑는
결과와 검색 순위는 이 **오프라인 규칙 검사**만으로 확인되지 않는다.

## 분석기 지연 측정

프롬프트 길이 감소는 지연 개선의 증거가 아니다. 같은 Gemini 모델, API 키,
컨테이너, 평가 CSV, `--limit`으로 변경 전/후를 각각 실행하고 `p50_ms`,
`p95_ms`, `fallback_count`를 비교한다. 실행마다 API 호출이 발생한다.

```powershell
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.benchmark_context_analyzer `
    --queries experiments/namuwiki/eval_queries_context_v01.csv --limit 10
```

API 키가 없으면 명시적으로 오류를 내며 fallback 속도를 모델 지연으로
보고하지 않는다. 서버 경로의 실제 분석 예산은 기본 20초 + 바깥쪽 2초
벽시계 여유다. 지연 실측 없이 “2초 이내”라고 답하지 않는다.
