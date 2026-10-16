# v58 — 가사 판(Mongo full_lyrics 캐시) TTL 만료 시 동기 전체 재적재 수정 (2026-10-16 밤)

워크트리 `/private/tmp/vague-finder-lyrics-cache-refresh` · 브랜치 `fix/lyrics-cache-refresh` · 기준 main `9b69dcc`(PR #71 v56 병합). PR #72(v57 리랭킹 예산)는 열려 있고 이 작업과 파일이 겹치지 않는다(독립, 어느 순서로 병합해도 된다). 커밋 전.

v56 운영 검증(§7-②)에서 발견한 결함: `LyricsExactSearchService`는 적재 완료 시각부터 TTL 300초(`LYRICS_CACHE_TTL_SECONDS`)를 세고 검색으로 연장되지 않는다. 만료 후 첫 가사 요청이 `_load_lock` 안에서 Mongo 전체(5,143곡 `full_lyrics`)를 다시 읽었고 이 PC→Atlas 기준 **약 350초**가 걸렸으며, 그동안 다른 가사 요청은 같은 잠금을 기다렸다(v56 A n007 다섯째 요청 350.6초 · D t014 351.3초, v57에서도 예열 5.7분). v57로 리랭킹 예산을 고쳐도 이 재적재가 겹치면 화면 제한 60초를 넘겼다.

TTL 숫자를 늘리는 것으로 끝내지 않았다. **판이 있으면 만료 뒤에도 요청은 기다리지 않고 지금 판으로 검색하며, 백그라운드 갱신을 한 번만 돌려 새 판이 완성되면 원자적으로 교체한다**(stale-while-revalidate).

## 1. 기존 동작과 변경 후 상태 전이

### 수정 전(`origin/main`)

```
판 없음 ──search()──▶ _load_lock 안에서 Mongo 전체 읽기(블로킹) ──▶ 판 게시
판 있음·TTL 안 ──search()──▶ 지금 판
판 있음·TTL 지남 ──search()──▶ _load_lock 안에서 Mongo 전체 읽기(블로킹 ≈350초, 다른 가사 요청도 잠금 대기) ──▶ 새 판 게시
refresh() ──▶ _load_lock 잡고 판 = None (다음 검색이 블로킹 적재)
```

### 수정 후

```
판 없음(최초) ──search()──▶ _load_lock 안에서 자료원 전체 읽기(블로킹, 기존 계약) ──▶ 세대 같으면 게시
                             실패 → 예외가 그대로 올라감(라우터가 "가사 경로 실패(빈 결과로 대체)" 기록) — 조용히 생략하지 않음
판 있음·TTL 안 ──search()──▶ 지금 판 (잠금 한 번, 마이크로초)
판 있음·TTL 지남 ──search()──▶ 지금 판 **즉시** + (갱신 중 아님 · 닫히지 않음 · 재시도 대기 아님이면) `lyrics_refresh` 스레드 1개 시작
    갱신 스레드: 어느 잠금도 안 잡고 자료원 전체 읽기 → 새 객체로 판 완성 → _state_lock 아래서 세대 확인 뒤 교체(성공) 
               / 예외 → 지금 판 유지, 실패 시각·오류 형식 기록, retry 간격 동안 재시작 금지
               / 종료·refresh()로 세대 변경 → 다음 문서에서 취소(cursor 닫음), 끝까지 읽었어도 게시 안 함
refresh() ──▶ 세대 += 1, 판 = None, 실패 대기 해제 (다음 검색이 블로킹 적재; 진행 중 갱신 결과는 폐기)
close() ──▶ closed, 세대 += 1, 갱신 스레드 join(timeout) — 시간 안에 안 끝나면 False(취소 완료 아님)
closed 뒤 search() ──▶ 마지막 판이 있으면 그 판(새 갱신 없음) / 없으면 LyricsServiceClosed
TTL = 0 ──▶ 캐시 없음(기존 경계값): 매 검색이 동기로 다시 읽고 백그라운드 없음
```

잠금 둘: `_state_lock`은 판·세대·갱신 상태를 바꾸는 짧은 구간만(자료원 읽는 동안 절대 안 잡음), `_load_lock`은 최초(블로킹) 적재만 직렬화. 갱신 스레드는 요청마다가 아니라 **갱신 한 번에 하나**(TTL 주기당 최대 1개), `daemon=True`, 핸들은 서비스가 쥐고 `close()`에서 join한다. 여러 서버 프로세스 간 갱신 통합은 범위 밖(프로세스마다 각자 갱신).

## 2. 최초 적재 · TTL 만료 · 명시적 무효화의 차이

| 경우 | 요청이 기다리는가 | 자료원 읽기 | 판 교체 | 실패 시 |
|---|---|---|---|---|
| 최초 적재(판 없음) | **예**(블로킹, 기존 계약 — 예열이 보통 먼저 낸다) | 요청 스레드, `_load_lock` | 세대 같고 닫히지 않았으면 게시 | 예외 전파(라우터가 경로 실패로 기록, 다음 검색이 다시 시도) |
| TTL 만료(판 있음) | **아니오** — 지금 판으로 즉시 | 백그라운드 1개, 잠금 없음 | 완성 뒤 원자 교체 | 지금 판 유지, `retry_seconds` 동안 재시작 금지 |
| 명시적 `refresh()` | 그 호출은 즉시 반환; **다음 검색이** 블로킹 | 다음 검색이 최초 적재와 같은 길 | 진행 중이던 갱신(옛 세대) 결과는 **폐기** | 최초 적재와 같음 |
| TTL = 0 | 예(매 검색) | 요청 스레드 | 매번 교체 | 예외 전파 |

**이번 수정이 해결하지 않은 것**: 최초 적재(서버 기동 뒤 예열 lyrics 단계, 이번 환경 실측 약 5.7~6분)는 그대로다. 예열이 끝나기 전 들어온 가사 요청은 예전처럼 적재를 기다린다. 이는 "판이 없으면 가사 경로를 빈 결과로 조용히 생략하지 않는다"는 계약을 유지한 결과이며, 증분 읽기·기동 시 디스크 캐시 등은 별도 작업이다.

## 3. 갱신 실패 · 재시도 · 종료 정책

- **실패**: 자료원 예외(연결·cursor 오류 등)는 갱신 스레드 안에서 잡아 `refresh_failures += 1`, `last_refresh_error`(**예외 형식 이름만**, 원인 사슬 최대 3단 — 메시지는 싣지 않는다. pymongo 메시지에는 호스트·포트·계정이 섞일 수 있고 이 값은 `/health`에 나간다. 로그도 같다), `refresh_failed_at`을 기록하고 **지금 판을 유지**한다. 만들던 목록은 버려져 **부분 판은 게시되지 않는다**. cursor는 `finally`에서 닫는다.
- **재시도 간격**: `LYRICS_REFRESH_RETRY_SECONDS`(새 설정, 기본 60초; 생성자 `refresh_retry_seconds`). 실패 뒤 이 시간 동안은 만료 요청이 와도 갱신을 시작하지 않는다(요청마다 전체 읽기가 몰리는 폭주 방지). 지나면 다음 만료 요청이 다시 한 번 시작한다. 성공하면 실패 기록을 지운다. `refresh()`도 실패 대기를 지운다(사용자가 다시 읽으라고 한 것).
- **오래된 자료 한계**: 만료 뒤 갱신이 끝나기까지(정상 약 350초)와 실패가 이어지는 동안은 **마지막 성공 판**으로 답한다. **유한한 상한은 보장되지 않는다** — 갱신은 가사 요청이 있어야 시작하고(요청이 없으면 만료된 판이 그대로 남는다), 실패하면 재시도 간격마다 다음 요청이 다시 시작할 뿐이다. 그 사이 Mongo에 들어온 곡·고친 가사는 검색·인용에 반영되지 않는다. `/health`의 `lyrics_cache.age_seconds`·`last_success_epoch`·`refresh_failures`·`last_refresh_error`·`retry_in_seconds`로 확인한다.
- **확인 창구**: `LyricsExactSearchService.status()` → `/health`의 `lyrics_cache`(`state` empty/fresh/stale/no_cache, `documents`, `generation`, `age_seconds`, `ttl_seconds`, `last_success_epoch`, `refreshing`, `initial_loads_running`, `refresh_running_seconds`, `refresh_successes/failures/cancelled/discarded`, `last_refresh_error`, `last_failure_age_seconds`, `retry_seconds`, `retry_in_seconds`, `mongo_socket_timeout_seconds`, `mongo_batch_size`, `close_wait_seconds`, `closed`). 서비스가 아직 안 만들어졌으면 `{"state": "not_created"}`(만들지 않는다 — Mongo를 열지 않는다).
- **종료**: lifespan이 `warmup.stop_and_wait` **뒤**(예열 lyrics 단계가 최초 적재 중일 수 있음), 벡터 클라이언트 닫기 **앞**에 `close_lyrics_exact_search_service()`를 부른다. `close()`는 closed·세대 변경으로 새 갱신을 막고 돌던 갱신은 **다음 문서에서** 취소(cursor 닫음)되며, 끝까지 읽었어도 게시하지 않는다. `close()`의 기본 대기는 **유한**하다: 가사 읽기는 전용 Mongo 클라이언트의 socket 제한(한 배치 기준, 아래) 안에 반환되거나 실패하므로 **갱신 스레드와 진행 중 최초 적재**를 "socket 제한 + 여유 15초"(`close_wait_seconds`, 기본 75초) 한도로 기다리고 끝나지 않으면 False(경고 로그)를 돌려준다. 닫힌 뒤에는 같은 잠금 안에서 전용 Mongo 클라이언트 생성을 거부하므로(`LyricsServiceClosed`) 종료 뒤 재개된 적재가 클라이언트를 새로 열어 남기지 않는다. 이 한도는 **가사 서비스 안의 대기 한도**이지 서버 전체 종료 시간이 아니다 — Mongo 클라이언트 정리와 다른 종료 단계(벡터 클라이언트·피드백 저장소 등)를 포함한 전체 종료 시간은 재지 않았다 — join만 끊는 것이 아니라 읽기 자체가 제한으로 끝난다는 점이 전제이고, False는 그 전제가 깨졌다는 신호다(주입 자료원처럼 제한이 없는 읽기가 그렇다). 그 뒤 전용 클라이언트를 닫는다. pymongo cursor는 한 배치(getMore)를 받는 중에는 멈출 수 없으므로 취소 단위는 배치다.
- **DB 제한(가사 읽기 한정)**: 공용 Mongo 클라이언트(`src/common/mongodb.py`)는 `serverSelectionTimeoutMS=5000`·`connectTimeoutMS=5000`만 있고 socket 제한이 없어 Mongo가 멎으면 읽기가 돌아오지 않는다. 그래서 가사 서비스는 **전용** `MongoClient`(`_make_lyrics_mongo_client`: 같은 URI·DB, `socketTimeoutMS` = `LYRICS_MONGO_SOCKET_TIMEOUT_SECONDS` 기본 60초, `appname=vaguefinder-lyrics`)로 읽고, cursor는 `batch_size` = `LYRICS_MONGO_BATCH_SIZE` 기본 200으로 고정한다. socket 제한은 **한 왕복(getMore 한 배치)** 기준이므로 배치 크기를 함께 고정해 왕복 하나의 크기를 예측 가능하게 둔다(v56 실측 5,143곡 350초 ≈ 68ms/곡 → 200곡 ≈ 14초 < 60초). 공용 클라이언트와 다른 Mongo 작업(피드백 저장소 등)에는 영향이 없다. 제한에 걸리면 pymongo가 `NetworkTimeout`을 던져 갱신은 실패(지금 판 유지·재시도 대기)로, 최초 적재는 예외로 끝난다. 전용 클라이언트는 `close()`에서 닫는다. 서버 측 누적 제한(`max_time_ms`)은 두지 않았다(총 적재 시간을 실측 없이 정하면 정상 적재를 끊을 수 있다).

## 4. 검색과 인용의 스냅샷 일관성

- `search(..., snapshot_out=)`은 **이 검색이 본 판**을 그대로 호출부에 넘긴다(계약 유지). 라우트는 그 판을 `excerpt()`에 넘겨 인용한다. 갱신 중이든 교체 뒤든 **요청이 검색한 판에서만** 인용이 뜬다 — 가사 원문·`lyrics_sha16`·매칭 위치가 한 판 안에서 나온다.
- 백그라운드 갱신은 **새 객체**(`LyricsSnapshot`은 frozen dataclass)를 만들고 참조만 바꾸므로, 나간 판은 그 판을 쥔 요청이 끝날 때까지 그대로 산다. `refresh()`·`close()`도 나간 판을 건드리지 않는다.
- 판에 `generation`을 붙여 어느 세대의 읽기인지 남긴다(무효화 뒤 옛 세대 결과가 게시되지 않는 근거).

## 5. 수정 전후 재현

### 5-1. 단위 재현(`repro_cache_refresh.py`, `repro_before_after.json`)

수정 전 코드는 `git show origin/main:src/retrieval/lyrics_exact_search.py`를 그대로 올려 쓴다. 자료원은 호출마다 첫 문서 뒤 **3초** 멈추는 가짜(합성 301곡, Atlas 350초를 축소), TTL 1초, 만료 뒤 요청 2개(0.3초 간격).

| 단계 | 수정 전 | 수정 후 |
|---|---|---|
| 1 최초 적재(블로킹) | 3.018초, 판 v1 | 3.024초, 판 v1 |
| 2 만료 뒤 첫 요청 | **3.021초**, 판 v2(재적재 기다림) | **0.009초**, 판 v1(지금 판) |
| 3 그 사이 둘째 요청 | **2.716초**(잠금 대기), 판 v2 | **0.002초**, 판 v1 |
| 4 갱신 완료 뒤 요청 | 0.001초, 판 v2 | 0.001초, 판 v2 |
| 자료원 호출 수 / 두 요청 벽시계 | 2 / 3.023초 | 2 / 0.314초 |

같은 자료에서 곡 ID·일치 유형·인용 구간은 전후 동일(`same_ids_types_spans_before_vs_after: true`). 인용 판 해시는 단계 2·3에서 수정 전은 새 판(검색도 새 판), 수정 후는 옛 판(검색도 옛 판)으로 **검색과 인용이 늘 같은 판**이다.

### 5-2. HTTP 재현(`run_check.py`, `http_check_{new,old}.{json,csv}`)

실제 FastAPI 앱(`src.backend.main:app`, `v58_app.py`가 그대로 import)에 **합성 가사 자료원**(`v58_runtime.FakeLyricsSource`, 제어 파일로 normal/gate/fail)을 꽂고 실제 Gemini·Claude·Mongo 호출은 예외로 차단(`guard` 셋 모두 blocked). 분석기는 검증 질의 하나에 고정 분석(가사 intent + verbatim 단서), 리랭커는 가짜 SDK(요청은 `use_rerank=false`). Qdrant는 v56 Docker 서버(`vague-finder-qdrant-v56`, 별칭 2개 확인) 서버 모드, 예열 켬. **검증값**: TTL 20초 · 재시도 15초(운영 기본 300/60초를 바꾼 것이 아니라 350초를 기다리지 않기 위한 환경변수). new(수정 후, 8058) · old(수정 전 코드 `V58_OLD_SERVICE=1`, 8059) 두 서버를 같은 Qdrant에 붙여 같은 시나리오를 보냈다. 가사 원문은 기록하지 않았다(판 해시·곡 ID·시간만).

| 요청 | 조건 | 수정 전(old) | 수정 후(new) |
|---|---|---|---|
| R1 | 정상(예열 적재 뒤) | 155ms, 판 b18f… | 546ms(첫 요청 모델 비용 포함), 판 b18f… |
| R2 · R2b | TTL 만료 뒤 동시 2요청, 갱신은 gate에 막힘(25초 뒤 열림) | **25,109ms · 24,805ms**(둘 다 재적재 완료까지 대기), 판 3db5…(새 판) | **173ms · 58ms**, 판 b18f…(지금 판), 상태 `stale`·`refreshing=true`·`refresh_running_seconds=0.4` |
| R3 | 곧바로 한 번 더 | 469ms(이미 교체됨) | 58ms, 판 b18f…, 자료원 호출 수 그대로(갱신 1개) |
| R4 | gate 열고 갱신 완료 뒤 | 49ms, 판 3db5… | 88ms, **판 3db5…**(교체됨), `refresh_successes` +1 |
| R5 | fail 모드로 TTL 만료 뒤 | 166ms, **가사 경로 실패 → 대상 곡 사라짐**(`target_rank=null`, 서버 로그 "가사 구절 일치 경로 실패 (빈 결과로 대체)") | 518ms, **판 3db5… 유지**(대상 1위·인용 유지), `refresh_failures=1`, `last_refresh_error=ConnectionError…` |
| R6 | 5초 뒤(재시도 간격 안) | 160ms, 다시 실패(요청마다 읽기) | 171ms, 판 유지, **자료원 호출 수 변화 없음** |
| R7 | normal(v3)로 바꾸고 간격 경과 뒤 | 182ms, 판 18d0…(블로킹 재적재 — 가짜는 빠름) | 243ms, 판 3db5…(갱신 시작) |
| R8 | 갱신 완료 뒤 | 51ms, 판 18d0… | 52ms, **판 18d0…** |
| R9 | gate 모드(v4)로 만료 뒤(종료 시나리오) | — (생략) | 149ms, 판 18d0…, 갱신 gate에 막힘 |
| 종료 | SIGTERM | 0.42초 뒤 종료 | 3초 뒤에도 **살아 있음**(갱신 읽기를 기다림) → gate 열자 **0.2초 뒤 종료**(SIGTERM 뒤 3.23초). 자료원 로그 call 6: `yielded 2, exhausted false`로 cursor 닫힘 = 다음 문서에서 취소, 게시 없음 |

18요청 전부 200, 대상 곡은 fail 모드의 old R5·R6을 빼면 늘 1위, 인용 구절은 매번 원문과 글자 단위로 같았다(`quote_matches_target_text=true`). new의 R1은 예열 적재 56초 뒤라 그 자체가 만료 요청이었고(정상 자료원, 즉시 갱신 성공) 이것이 `refresh_successes` 첫 1건이다.

### 5-3. HTTP 재현에서 확인하지 못한 것

gate는 파일 폴링(0.1초)이라 "Mongo 배치 안에서 막힘"의 모양만 흉내 낸다. 실제 pymongo cursor의 배치 경계·네트워크 지연에서 취소가 몇 초 걸리는지는 재지 않았다(§7).

## 6. 실행한 테스트

- 신규 `tests/test_lyrics_cache_refresh.py` **38건** 통과(3회 반복 안정, 약 4.5초; 리뷰 1차 4건·2차 3건·3차 2건 포함 — §10). 가짜 자료원(`FakeSource`: gate 이벤트·실패 지점·호출/닫기 횟수)·가짜 시계(`FakeClock`)·실제 라우트+실제 `SearchRouter`(가짜 Qdrant/임베더)를 쓴 TestClient HTTP 3건 포함. 필수 사례 대응:
  - 최초 적재 성공·실패(예외 전파·다음 검색 재시도), 동시 최초 적재는 읽기 1회
  - TTL 전 불필요한 적재 없음
  - 만료 뒤 갱신이 막혀 있어도 기존 결과 즉시 반환(<1초), 상태 `stale/refreshing`
  - 동시 12요청에도 갱신 1회, 요청마다 스레드 생성 없음
  - 갱신 성공 뒤 새 요청은 새 판
  - 갱신 도중 자료원 오류 → 부분 판 미게시, cursor 닫힘
  - 실패 뒤 기존 자료 유지·재시도 간격 준수·간격 뒤 정상 복구, 간격 설정 가능
  - 옛 판으로 검색한 요청은 교체 뒤에도 같은 판에서 인용(단위·HTTP 둘 다, 요청 중간 교체 포함)
  - 명시적 `refresh()`: 판 폐기·다음 검색 블로킹, 진행 중 백그라운드 갱신 결과 폐기, 최초 적재 중 `refresh()`도 덮어쓰지 않음, `refresh()`가 갱신을 기다리지 않음
  - 종료: 돌던 갱신 취소(`close(0.1)`→False, gate 열면 True), 종료 뒤 끝난 갱신 미게시, 닫힌 뒤 마지막 판으로 응답·새 갱신 없음, 판 없으면 `LyricsServiceClosed`, 멱등
  - 빈 자료원(빈 판도 판), TTL 0(캐시 없음·동기·스레드 없음), 기존 `documents=` 생성자
  - 자료 같으면 exact/fuzzy 결과·순위·인용 해시 유지
  - 의존성: `lyrics_cache_status()`·`close_lyrics_exact_search_service()`가 서비스를 만들지 않음, 환경변수 `LYRICS_REFRESH_RETRY_SECONDS` 반영, lifespan 순서 warmup → lyrics → vector
- 관련 기존 테스트 **399 통과·67 skip**: `test_lyric_evidence`(기존 `refresh()`·판 고정·`_source_documents` 덮어쓰기 테스트 그대로 통과) · `test_search_explain` · `test_warmup` · `test_qdrant_backend` · `test_v56_rerank_fallback_paths` · `test_user_explanation` · `test_feedback` · `test_feedback_retrieval` · `context/test_context_api_candidate_proof` · `context/test_context_api_evidence_step7`
- 전체 `tests/`: §6 끝의 결과 줄 참고(아래 "전체 실행" 항목)
- 단위 재현 `repro_cache_refresh.py`(§5-1), HTTP 재현 `run_check.py`(§5-2) 각 1회

전체 실행: `venv/bin/python -m pytest tests/ -q` **6,409 통과 · 130 skip · 실패 0**(리뷰 3차 반영 뒤 최종, 약 8분; 1차 보고 시점 6,400 · 1차 리뷰 반영 뒤 6,404 · 2차 반영 뒤 6,407도 실패 0). `tests/context/test_context_artifacts_v4.py::…test_default_artifact_path_follows_custom_raw_sandbox` 1건은 deselect — 메인 체크아웃에서도 실패하는 경로 의존 테스트(`/tmp`→`/private/tmp`, v57에서 확인)로 이번 변경과 무관하다.

## 7. 미검증 범위 · 남은 위험

- **실제 Mongo 성능**: 이번 검증은 실제 Mongo 호출 0건이다. 운영에서 갱신 1회는 여전히 전곡 읽기(약 350초, 이 PC→Atlas)이며 그동안 `lyrics_refresh` 스레드 1개와 메모리(판 2벌: 지금 판 + 만드는 중인 목록)를 더 쓴다. TTL은 **적재(갱신) 완료 시각부터** 다시 세므로, 지속 요청·실패 없음 기준으로 약 300초 유지 → 350초 갱신(그동안 옛 판 제공) → 다시 300초 유지의 주기가 된다. 운영 TTL을 바꿀지는 별도 결정(이 수정은 TTL 값을 바꾸지 않았다). **socket 제한 60초·배치 200은 실제 Atlas에서 재지 않았다** — 한 배치가 60초를 넘는 환경이면 정상 적재도 실패하므로 운영 투입 전 `LYRICS_MONGO_SOCKET_TIMEOUT_SECONDS`·`LYRICS_MONGO_BATCH_SIZE`를 실측으로 확인해야 한다.
- **오래된 자료**: 만료 뒤 갱신이 끝나기까지, 그리고 실패가 이어지는 동안 마지막 성공 판으로 답한다. 갱신은 요청이 있어야 시작하므로 **유한한 최대 시간은 보장되지 않는다**(§3).
- **종료 지연**: 갱신·최초 적재가 Mongo 배치를 받는 중이면 가사 서비스의 `close()`는 최대 "socket 제한 60초 + 여유 15초"를 기다린다(Mongo가 멎어도 유한). 그 안에 끝나지 않으면 경고와 함께 종료가 이어지고 daemon 스레드는 프로세스와 함께 사라진다 — 이 경우는 제한이 작동하지 않은 이상 상황이다. 이것은 갱신 스레드·최초 적재의 대기 한도이고, Mongo 클라이언트 정리와 다른 종료 단계를 더한 **서버 전체 종료 시간은 검증하지 않았다**. 최초 적재(예열)는 `warmup.stop_and_wait`가 제한 없이 기다리는 기존 정책 그대로이나, 읽기 자체는 같은 전용 클라이언트라 socket 제한으로 끝난다.
- **세대 번호 읽기**: 갱신 스레드는 문서마다 `self._generation != generation`을 잠금 없이 읽는다(CPython int 읽기, 늦게 봐도 다음 문서에서 멈춘다). 게시 판정은 `_state_lock` 아래에서 한다.
- **refresh()와 블로킹 적재의 겹침**: `refresh()`는 `_load_lock`을 잡지 않으므로(분 단위 대기 금지) 최초 적재 중 호출되면 그 적재는 자기 요청에만 쓰이고 게시되지 않으며, 다음 검색이 다시 읽는다(읽기 2회). 드문 경로이며 테스트로 고정했다.
- **HTTP 검증 범위**: 서버 1프로세스·요청 2개 동시까지, 합성 301곡, 운영 질의가 아닌 고정 분석 1건. dev/test 정확도는 재측정하지 않았다(가사 판 자료가 같으면 결과가 같음을 단위 테스트로 고정). D(보조 경로)·리랭킹 켠 요청은 보내지 않았다(가사 경로와 무관).
- **`/health` 응답 키 추가**(`lyrics_cache`): 외부 도구가 이 키를 모른다. 건수·시각·상태·오류 형식만 있고 가사·연결 정보는 없다.
- 남은 운영 결함(별도 작업): v56 §7-③ 부분 실패·후보 밖 id가 `model_applied` · `HYBRID_SCAN_LIMIT` 5000 < 5,143 · 서버 모드 Context 별칭 기동 검사 없음 · 최초 적재 약 6분(§2).

## 8. 재현 방법

```
# 단위 재현(실제 Mongo·모델 없음, 약 15초)
venv/bin/python experiments/reranking/results_v58_lyrics_cache_refresh/repro_cache_refresh.py --slow 3 --ttl 1

# HTTP 재현(v56 Docker Qdrant 서버 127.0.0.1:6356 필요 — docker start vague-finder-qdrant-v56)
OUT=experiments/reranking/results_v58_lyrics_cache_refresh
zsh $OUT/start_server.sh new                 # 수정 후, 8058
PORT=8059 zsh $OUT/start_server.sh old old   # 수정 전 코드(origin/main), 8059
venv/bin/python $OUT/run_check.py --name new --port 8058 --ttl 20 --retry 15
venv/bin/python $OUT/run_check.py --name old --port 8059 --ttl 20 --retry 15 --old --gate-after 25
# 드라이버가 끝에 SIGTERM으로 서버를 내린다. 결과 http_check_{new,old}.json/.csv, 자료원 로그 source_calls_*.jsonl(로컬)

# 테스트
venv/bin/python -m pytest tests/test_lyrics_cache_refresh.py tests/test_lyric_evidence.py tests/test_search_explain.py tests/test_warmup.py tests/test_qdrant_backend.py -q
```

## 9. 파일

- 서비스: `src/retrieval/lyrics_exact_search.py`(판 수명·백그라운드 갱신·`status()`·`close()`·`documents_source`/`clock` 주입), `src/backend/api/dependencies.py`(`LYRICS_REFRESH_RETRY_SECONDS`·`LYRICS_MONGO_SOCKET_TIMEOUT_SECONDS`·`LYRICS_MONGO_BATCH_SIZE`, `lyrics_cache_status()`, `close_lyrics_exact_search_service()`), `src/backend/main.py`(lifespan 종료 순서, `/health.lyrics_cache`)
- 테스트: `tests/test_lyrics_cache_refresh.py`
- 이 폴더: `RUN_INFO.md` · `repro_cache_refresh.py` · `repro_before_after.json` · `run_check.py` · `v58_app.py` · `v58_runtime.py` · `v56_runtime.py`(v56 사본, 모델 차단) · `start_server.sh` · `http_check_new.{json,csv}` · `http_check_old.{json,csv}`. git 제외(`.gitignore`): `logs/` · `ctl/` · `progress.txt` · `timing_*.jsonl` · `fake_calls_*.jsonl` · `source_calls_*.jsonl`
- 바꾸지 않은 것: 운영 `.env`·기본값(TTL 300초 그대로), 원본 색인·DB, 보조 경로·가중·후보 수·프롬프트, 리랭킹 부분 실패 표시, `HYBRID_SCAN_LIMIT`, 평가기(`evaluate_search_accuracy`는 가사 서비스를 쓰지 않음)

## 10. 리뷰 반영(2026-10-16 밤, 1차 보고 뒤)

| # | 지적 | 반영 |
|---|---|---|
| P1 | Mongo 응답이 멎으면 갱신 스레드가 DB 읽기에서 돌아오지 않고 lifespan이 `close()`를 무기한 기다린다 — 요청이 없어도 종료가 멈출 수 있다 | 가사 읽기 **전용** MongoClient에 `socketTimeoutMS`(기본 60초)와 cursor `batch_size`(기본 200)를 둬 한 왕복이 제한 안에 끝나거나 실패하게 했다(공용 클라이언트·다른 Mongo 작업은 그대로). `close()` 기본 대기 = 제한 + 15초, 넘기면 False 경고(§3). join만 끊는 방식이 아니다 — 읽기 자체가 끝난다. 테스트: 전용 클라이언트 생성 인자·배치 크기·닫기, 제한 없는 주입 자료원에서 기본 대기 뒤 False |
| P2 | `_error_label()`이 `str(exc)`를 160자로 잘라 `/health`에 그대로 노출 | 예외 **형식 이름만**(원인 사슬 최대 3단) 저장·로그. 테스트: 호스트·계정이 든 메시지가 상태 JSON과 로그에 없음 |
| P2 | 무효화된 옛 세대 갱신의 늦은 실패가 새 세대의 재시도 대기를 건다 | 실패 처리에도 성공 처리와 같은 세대·종료 확인을 넣어 옛 세대 실패는 `refresh_discarded`로 폐기(대기·오류 기록 없음). 테스트: 옛 갱신 시작 → `refresh()` → 새 판 적재 → 옛 갱신 늦게 실패 → 새 판 만료 시 갱신이 정상 시작 |
| P2(2차) | `_get_mongo_client`가 `_closed`를 보지 않아 종료 뒤 재개된 최초 적재가 클라이언트를 새로 열 수 있고, `close()`가 최초 적재를 추적하지 않아 True를 돌려줬다 | 클라이언트 생성을 `_state_lock` 안에서 `closed`면 거부(`LyricsServiceClosed`). 진행 중 최초 적재 수(`_blocking_loads`, 상태 `initial_loads_running`)를 세고 `close()`가 갱신 스레드와 같은 한도 안에서 조건변수로 기다린다(끝나기 전이면 False). 닫힘으로 취소된 최초 적재는 `LyricsServiceClosed`로 끝나고 cursor를 닫는다. 테스트: 클라이언트 생성 직전에 멈춘 적재 → `close(0.3)`=False → 재개 → 팩토리 호출 0·`LyricsServiceClosed` → `close()`=True; gate에 막힌 최초 적재의 취소·cursor 닫힘 |
| P2(2차) | cursor 정리 실패 경로의 `exc_info=True`가 DEBUG 로그에 메시지 원문을 남긴다 | 예외 형식 이름만 남기도록 통일. 테스트: 호스트·계정이 든 메시지로 `close()`가 실패해도 로그에 형식 이름만 |
| P2(3차) | `_load_blocking`이 `_closed` 확인 뒤 잠금을 풀고 다시 잡아 `_blocking_loads`를 올려, 그 틈에 `close()`가 "진행 중 적재 0건"으로 True를 돌려주고 적재가 종료 뒤 자료원 읽기를 시작할 수 있었다 | 종료 확인·세대 확보·적재 등록을 **한 잠금 구간**으로 합쳤다 — 종료가 먼저면 거부(`LyricsServiceClosed`), 등록이 먼저면 `close()`가 기다린다. 테스트: `_state_lock`에 해제 훅을 끼워 적재 스레드가 그 임계 구역을 벗어나는 순간 `close(0.2)`를 끼워 넣는 결정적 재현(등록 1건을 보고 False, 적재는 `LyricsServiceClosed`) + 150회 반복 경합에서 "close() True 뒤 자료원 읽기 시작" 0건. 잠금을 다시 나눈 변형에서는 결정적 재현 테스트가 실패함을 확인했다 |
| 표현(2차) | "최대 75초에 종료" | 갱신 스레드·최초 적재의 대기 한도로 정정. 전체 종료 시간은 미검증(§3·§7) |
| 표현 | "350초 > TTL 300초라 거의 항상 갱신 중" | TTL은 적재 완료부터 다시 센다 — 약 300초 유지 + 350초 갱신 주기로 정정(§7) |
| 표현 | 오래된 자료 제공 시간의 "최대" 공식 | 갱신은 요청이 있어야 시작하므로 유한한 상한이 보장되지 않는다고 정정(§3·§7) |

반영 뒤 재검증(1차): 신규 테스트 33건(3회 반복) · 관련 기존 399건 통과. (2차): 신규 36건(3회 반복) · 관련 기존 399건 통과. (3차): 신규 38건(3회 반복) · 관련 기존 399건 통과, 전체 `tests/`는 §6 참고. 2차 변경은 종료 경로(클라이언트 생성 거부·최초 적재 대기)뿐이라 HTTP 재실행은 하지 않았다 — HTTP 종료 시나리오는 백그라운드 갱신 경로이고 2차 변경 전후 동작이 같다 · 전체 `tests/` 결과와 HTTP 재실행은 §6·§5-2 끝의 "리뷰 반영 뒤 재실행" 참고.

**리뷰 반영 뒤 HTTP 재실행(new, `http_check_new.{json,csv}`가 이 실행; 1차 실행은 `logs/http_check_new_run1.*`로 로컬 보존)**: 가드 셋 모두 blocked(전용 가사 클라이언트 생성 자리 포함), `/v58/status`의 `mongo_socket_timeout_seconds=60 · close_wait_seconds=75`. 10요청 전부 200, 대상 곡 1위, 인용 구절 원문 일치. R1 97ms(판 b18f…) → gate 만료 뒤 R2·R2b **157ms · 93ms**(판 b18f…, `stale/refreshing`) → R3 75ms(자료원 호출 수 그대로) → R4 105ms 판 3db5… → fail 모드 R5 158ms 판 유지·`refresh_failures=1`·**`last_refresh_error="ConnectionError"`(형식 이름만, 메시지 없음)** → R6 164ms 호출 수 그대로 → R7 155ms 갱신 시작 → R8 50ms 판 18d0… → R9 146ms gate 막힘 → SIGTERM 3초 뒤 살아 있음 → gate 열자 종료(SIGTERM 뒤 3.22초), 자료원 call 5 `yielded 2, exhausted false`(다음 문서에서 취소, 게시 없음). 1차 실행과 같은 모양이다. old 서버는 다시 돌리지 않았다(수정 전 코드는 바뀌지 않음).
