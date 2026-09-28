# 나무위키 배경지식 검색 설계

## 목적

나무위키 정보는 곡의 분위기나 가사 의미를 설명하는 본문이 아니다. 다음처럼
곡을 둘러싼 구체적인 기억을 찾는 별도 증거다.

- 어느 방송·드라마·애니메이션·게임에서 들었는지
- 제작·녹음 비화, 뮤직비디오 줄거리, 무대와 커버
- 밈·패러디·챌린지, 수상·기록·영향
- 버전 차이와 음역·코드·가창 난이도

따라서 기존 곡 단위 텍스트 벡터에 합치지 않는다. 나무위키 사실만 모은 별도
context 인덱스에서 검색한 뒤 결과를 `song_id`로 곡 단위 집계한다.

## 데이터 흐름

1. 크롤러가 원문·파싱 결과를 `data/context`에 보존한다.
2. 검증된 짧은 사실만 원곡의 `meta.json > namuwiki.facts`에 기록한다.
3. 같은 실행에서 `artifacts/context/songs/<song_id>.json`을 만든다.
4. 각 `retrieval.records[]`는 독립적인 dense 임베딩 문서가 된다.
5. sparse는 `retrieval.sparse_profile` 하나만 곡별 BM25 문서로 사용한다.
6. 전체 sparse profile corpus로 별도 context BM25 통계를 학습한다.
7. dense context 결과를 `song_id`별 최고 점수로 집계한다.
8. context dense/sparse 순위와 기존 text/image/audio 후보를 RRF로 결합한다.

## v6 검색 레코드

```json
{
  "record_id": "nw:837567:...",
  "source_fact_indices": [9],
  "category": "media_usage",
  "section": "여담 > 짱구는 못말려 국내판 삽입곡",
  "evidence_text": "정식 발매 전 SBS에서 ... 배경음악으로 사용됐다.",
  "dense_text": "윤도현 - 사랑했나봐 | 매체 사용: 정식 발매 전 SBS에서 ... 배경음악으로 사용됐다.",
  "keywords": ["짱구는_못말려", "sbs", "배경음악", "사용되다"],
  "quality": "standalone"
}
```

- `evidence_text`: 사용자에게 보여 주거나 리랭커의 근거로 쓰는 사실
- `dense_text`: 그대로 Ko-E5 passage 임베딩에 넣는 사실 단위 입력
- `keywords`: 해당 사실을 설명하거나 sparse hit의 근거를 찾는 소수 단서
- `source_fact_indices`: 원본 compact fact까지 역추적하는 위치
- `quality`: 독립 문장인지, 앞 문장/섹션으로 문맥을 보완했는지 표시

`source_fact_count == covered_source_fact_count`가 스키마 불변식이다. 정제된
`meta.json` 사실을 아티팩트 생성기가 조용히 버리면 파일 검증 자체가 실패한다.

곡명·가수명과 공통 섹션 단서를 모든 사실에 반복하지 않는다. 다음처럼 한 곡에
하나뿐인 sparse 프로필로 모은다.

```json
{
  "profile_id": "nws:837567",
  "terms": [
    "사랑했나봐", "윤도현", "짱구는_못말려", "6기", "15화",
    "나미리_선생님", "투니버스", "곰돌이_푸", "강남스타일", "a4"
  ]
}
```

프로필은 중복 없는 최대 80개 단서이며, 각 사실이 일부 슬롯을 공평하게 갖도록
round-robin으로 만든다. 긴 첫 문장이 전체 sparse 예산을 독점하지 않는다.

## 검색 시 지켜야 할 규칙

- dense는 **사실 단위**, sparse는 **곡 단위**로 색인한다.
- sparse는 기존 곡 검색용 BM25 통계와 섞지 않고 context corpus로 별도 fit한다.
- dense 문서 ID는 `record_id`, sparse 문서 ID는 `profile_id`를 사용한다.
- 곡명과 가수명은 sparse 프로필에는 한 번만 둔다. 사실별 dense 벡터에는
  `가수 - 곡명 | 카테고리: 사실` 형식의 짧은 정체성 prefix를 포함하고 결과는
  `song_id`로 원곡과 연결한다.
- 같은 곡에서 여러 사실이 검색돼도 합산하지 않는다. 1차 버전은
  `max_per_song`으로 사실 수가 많은 곡의 편향을 막는다.
- context 단서가 뚜렷한 질의(OST, 특정 방송/캐릭터, 밈, 제작 비화 등)는 context
  후보 비중을 높인다. 일반 분위기·가사 질의는 기존 검색을 주 경로로 유지한다.
- 최종 응답/리랭킹에는 최고 점수를 만든 `evidence_text`를 함께 전달해 왜 해당 곡이
  검색됐는지 설명 가능하게 한다.

## 생성 및 재생성

크롤링과 함께 생성:

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context \
  --song-ids 837567 1698598
```

전체 카탈로그 실행은 기본적으로 실행당 최대 40곡만 선택한다. 동일한 명령을
재실행하면 현재 `meta.json`과 artifact 상태를 다시 검사하여 완료된 곡과
`needs_review` 곡을 건너뛰고 다음 미처리 곡을 이어서 수집한다. HTTP 요청 예산이
먼저 끝나거나 접근이 중지되면 배치를 즉시 종료하며, 남은 수천 곡을 한 줄씩
`deferred`로 출력하지 않는다. 평소 이어받기 실행에는 `--force`를 사용하지 않는다.

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context
```

현재 artifact를 만들 수 없는 곡은 `artifacts/context/unresolved.json`에 별도로
게시된다. 이 파일에는 전체 미해결 ID와 `needs_review`, 수집 오류, artifact 대기,
미수집 ID가 구분되어 있다. `needs_review`는 다른 곡의 동명 문서를 잘못 색인하지
않도록 검색 artifact를 만들지 않으며, 명시적으로 다시 확인할 때만 재시도한다.

### 동명이곡·앨범 문서 결합 규칙

페이지 URL 모양만으로 곡을 승인하지 않는다. 수집 대상에는 곡명·가수 외에 앨범,
발매 연도와 안전하게 정리한 제목 별칭을 함께 전달하고, 다음 순서로 결합한다.

1. `곡명(가수)` 전용 문서는 강한 일치로 승인한다.
2. 제목만 있는 문서나 `곡명(노래)` 문서는 개요·곡정보·표에서 가수와 음악 관계를
   다시 확인한다.
3. 동음이의·앨범 문서는 제목, 가수, 앨범 단서가 함께 맞는 **한 개의 곡 섹션**만
   선택한다.
4. 같은 강도의 후보 섹션이 둘 이상이면 자동 선택하지 않고 `needs_review`로 둔다.
5. 섹션이 선택된 경우 그 하위 `여담`만 정제한다. 이웃 곡의 여담은 fact나 context
   artifact에 들어가지 않는다.

따라서 `page_title_mismatch`, `title_only_page_artist_not_verified`,
`disambiguation_page_multiple_song_sections`였던 기존 결과는 새 결합 규칙으로 다시
판정할 가치가 있다. 다만 `needs_review` 전부를 승인하는 규칙은 아니며, 독립적인
가수·앨범 근거가 부족한 곡은 계속 검토 대상으로 남는 것이 정상이다.

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context \
  --song-ids <ID> --retry-needs-review --url-map <JSON>
```

자동 후보는 `제목(가수 별칭)`, `제목(노래)`, `제목` 순서로 확인한다.
`제목(노래)`는 접미사만으로 승인하지 않고, 개요·상세·소개·곡정보 또는 제목 앞
본문에서 음악 관계 표현과 가수 별칭이 함께 확인될 때만 해당 곡에 연결한다.
기존 버전에서 `not_found`로 저장된 곡을 새 후보 규칙으로 다시 확인할 때는 정확한
ID만 지정해 재수집한다.

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context \
  --song-ids 30806580 --retry-not-found --refresh
```

직접 확인한 URL을 사용할 경우 `song_id -> URL` JSON을 `--url-map`으로 추가한다.
`--retry-not-found`만 반복 실행하면 실제로 문서가 없는 곡도 매번 다시 선택되므로,
전체 재검토는 먼저 고정된 ID 목록을 만든 다음 작은 배치로 나누어 실행한다.

전체 실행의 기본 40곡보다 다른 배치 크기가 필요하면 `--limit`을 지정한다. 요청
간격과 실행당 HTTP 요청 예산은 별도 안전장치이므로 대량 수집을 위해 무리하게
늘리지 않는다.

### 전체 결과 감사

수집 후에는 네트워크 요청이나 `meta.json` 수정 없이 전체 카탈로그를 감사한다.

```bash
docker compose exec backend python -m \
  src.crawler.scripts_py.audit_namuwiki_context
```

기본 출력은 `artifacts/context/audit/`이며 다음 파일을 만든다.

- `summary.json`: 상태·오류 코드·위험도 집계
- `inventory.jsonl`: 모든 곡의 상태, source, fact 수, artifact 최신성, 결합 근거
- `needs_review_groups.json`: `needs_review` 전곡을 원인별로 묶은 목록
- `manual_review.csv`: 고위험 전곡과 원인별/조회수별 표본을 합친 제한된 검토 큐
- `catalogue_errors.json`: 읽기 실패, 중복 ID, 손상된 run report

`manual_review.csv`의 `decision`, `manual_url`, `review_note`는 사람이 기록하는 빈
열이다. 감사기는 자동으로 원본을 고치지 않는다. CI나 최종 배포 전에는 고위험 또는
치명적 결함이나 미완료 coverage에 실패하도록 `--strict`를 붙인다. 아직
`needs_review`가 남은 중간 점검에서는 옵션 없이 실행하는 것이 정상이다.

```bash
docker compose exec backend python -m \
  src.crawler.scripts_py.audit_namuwiki_context --strict
```

기존 `needs_review`를 새 규칙으로 재판정할 때는 먼저 dry-run으로 선택 범위를 보고,
기본 40곡 배치를 반복하거나 검토한 ID/URL만 지정한다. 캐시가 있으면 같은 snapshot을
재사용하고, 후보가 추가로 필요한 경우에만 네트워크 요청을 쓴다.

```bash
docker compose exec backend python -m \
  src.crawler.scripts_py.backfill_namuwiki_context \
  --retry-needs-review --dry-run

docker compose exec backend python -m \
  src.crawler.scripts_py.backfill_namuwiki_context \
  --retry-needs-review
```

네트워크 요청 없이 현재 `meta.json`으로 v6 아티팩트만 재생성:

```bash
docker compose exec backend python -m src.crawler.scripts_py.backfill_namuwiki_context \
  --song-ids 837567 1698598 --artifacts-only --force
```

샌드박스 데이터에서 `--artifact-dir`을 생략하면 출력도 샌드박스 옆으로 간다.
실데이터 아티팩트와 테스트 fixture가 섞이지 않도록 `--data-dir`에서 기본 출력
경로를 계산한다.

fact dense와 song sparse 생성:

```bash
docker compose exec backend python -m src.embedding.cli.embed_context_dense
docker compose exec backend python -m src.embedding.cli.fit_context_bm25
```

전체 context 수집 완료 후에는 두 명령 모두 `--require-complete-context`로 실행한다.
BM25 세부 계약은 [`namuwiki_context_sparse.md`](namuwiki_context_sparse.md)를
참조한다.

## 다음 구현 단계

1. 별도 namespace 또는 별도 index에 dense fact와 sparse song profile을 upsert한다.
2. 질의에서 context category/고유명사/숫자 단서를 추출한다.
3. context dense/sparse 순위를 RRF로 결합한다.
4. fact hit를 `song_id`별 `max`로 집계하고 기존 곡 후보와 융합한다.
5. 실제 모호 질의 평가셋으로 context 가중치와 top-k를 조정한다.
