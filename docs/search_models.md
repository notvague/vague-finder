# 검색 모델 설정 — Gemini on Vertex AI (전용)

2026-10-15부터 운영용 생성형 AI 호출은 **전부 Vertex AI의 Gemini**다 — 크롤링 정제·댓글 선별·Context 추출,
질의 분석, 리랭킹. 화면의 Claude / Gemini 선택(#63)은 철회했다: API 본문에 `model_provider`가 없고
`GET /api/v1/search/models`도 없다. 서비스는 Claude 클라이언트·Claude 리랭커를 쓰지 않는다.

Claude 비교 실험의 보고서(#62, `experiments/sonnet_comparison/`)와 실험 브랜치(`experiment/haiku-5-5` ·
`experiment/sonnet-5-5` · `experiment/haiku-structured`)는 그대로다. 재현은 그 브랜치의 커밋(`8ea9706` · `cd9cf11`)에서
한다 — main에는 Claude 모듈이 없다. 당시 조건(AI Studio 저장본과의 비교)은 보고서에 그대로 남긴다.

## 설정

```dotenv
GCP_PROJECT_ID=<크레딧이 연결된 프로젝트>
GCP_LOCATION=global
GEMINI_RETRIEVAL_BACKEND=vertex      # 호환용 — 비워도 같다. 옛 api_key·ai_studio·auto는 설정 오류
GEMINI_MODEL_NAME=gemini-3.1-flash-lite        # 질의 분석·리랭킹
GEMINI_CRAWL_MODEL_NAME=gemini-3.5-flash-lite  # 정제·댓글 선별·Context 추출
RERANKER_BACKEND=gemini_listwise
```

- `GEMINI_API_KEY`·`GOOGLE_API_KEY`는 쓰지 않는다. 남아 있으면 한 번 경고하고 무시한다 — 지운다.
  호출부가 키를 넘겨 AI Studio로 나가는 경로도 없다(`make_genai_client`에 키 인자가 없다).
- 인증은 ADC(Application Default Credentials)다. 크레딧이 있는 Google 계정으로
  `gcloud auth application-default login`을 한 번 실행한다. **브라우저 로그인이나 `gcloud auth login`을 바꾼 것만으로는
  Python 쪽 인증이 바뀌지 않는다** — `gcloud auth list`(CLI 계정)와 ADC 파일(`~/.config/gcloud/application_default_credentials.json`)은
  별개다. ADC는 프로세스에서 한 번만 읽어 모든 클라이언트가 공유한다.
- 프로젝트는 결제 계정(체험 크레딧)에 연결돼 있고 Vertex AI API(`aiplatform.googleapis.com`)가 켜져 있어야 하며,
  ADC 계정이 그 프로젝트에 `roles/aiplatform.user` 이상이어야 한다. 모델 이름은 환경변수로 바꾼다 — 새 프로젝트에서
  지원되는지는 아래 기동 검사가 확인한다.

## 기동·시작 검사와 실패 시 동작

- **서버**(`main.lifespan`): `RERANKER_BACKEND` 값 → 경로 설정 → 프로젝트 → ADC → 검색 모델 무료 호출(countTokens 1회, 토큰 과금 없음)
  순으로 확인한다. 설정·권한 오류(`GeminiConfigError`: HTTP 400·401·403·404, 프로젝트·ADC 없음, 경로 설정 오타)면 서버가 **뜨지 않는다**.
  ADC 파일이 있어도 토큰 갱신이 `invalid_grant`(철회·만료·계정 변경)로 거절되면 재로그인이 필요한 설정 오류로 멈춘다.
  429(RESOURCE_EXHAUSTED)·5xx·연결 불가·인증 서버 일시 장애(google-auth가 retryable로 표시)는 설정 오류가 아니다 — 확인 호출은
  SDK가 최대 3회(1~4초 대기) 다시 시도하고, 그래도 남으면 경고만 남기고 뜬다(요청 경로에서 다시 시도한다).
- `RERANKER_BACKEND`는 비우거나 `gemini_listwise` / `cross_encoder`다. 비우면 `gemini_listwise`, `cross_encoder`는 명시했을 때만이고,
  제거된 `claude_listwise`·오타는 기동·측정 시작에서 멈춘다(조용히 Cross-Encoder로 내려가지 않는다). Vertex 프로젝트가 없을 때의
  CE 폴백은 그대로이며 로그·runinfo(`reranker.backend`)에 남는다.
- **분석 캐시 생성·측정 CLI**(`build_analysis_cache` · `evaluate_search_accuracy` · `evaluate_clarification`): Gemini를 부를 실행이면
  시작 전에 같은 검사를 하고, 설정 오류면 멈춘다. 시작할 때 `리랭커: …`와 `Gemini 경로: vertex location=… model=… probe=ok` 한 줄이 찍힌다.
- **요청 중 실패**는 기존 폴백 그대로다 — 질의 분석은 재시도 뒤 규칙 폴백(`confidence=0.0`), 리랭킹은 검색 순서 유지 + `rerank_outcome=failed`.
  다른 공급자로 바꿔 타지 않고, 실패를 정상 리랭킹으로 기록하지 않는다(`rerank_status_counts.failed`).
- 시간 제한은 그대로다: 질의 분석 전체 20초(`QUERY_ANALYSIS_TIMEOUT_SECONDS`), 리랭킹 호출 20초·전체 예산 30초
  (`GEMINI_RERANK_TIMEOUT_SECONDS` · `GEMINI_RERANK_BUDGET_SECONDS`), 화면 검색 60초. 서버 종료 때 분석기의 Gemini 연결 풀을 닫는다.

## 기록과 평가

- 측정 runinfo의 `gemini`에 `backend=vertex` · `location` · `setting`이 남는다. 분석 캐시 지문(`meta.analyzer`)에도
  `gemini_backend` · `gemini_location`이 들어간다. 프로젝트 ID는 어디에도 적지 않는다.
- 2026-10-15 이전 캐시(`analysis_cache_v06/v09/v10_dev.json`, `results_v46_*/analysis_cache_*_ctx.json`)는 전부 AI Studio 분석이다.
  경로 키가 없는 캐시는 `api_key`로 본다. `build_analysis_cache`는 `--output`의 캐시가 **다른 경로**면 `--force`·`--reuse-despite-drift`로도
  덮어쓰거나 이어 채우지 않고 다른 경로를 요구하며, 경로가 같아도 조건이 어긋나면 플래그 없이는 멈춘다(`--force`는 `--query-ids`와 함께 못 쓴다).
  **Vertex 질의 분석까지 포함한 평가는 새 캐시를 따로 만든다** — 옛 캐시·결과를 덮어쓰지 않는다. 옛 캐시를 읽어 측정하는 것은 그대로 된다.
- AI Studio로 잰 옛 기준선(v32~v47 · 봉인 test v34 · 재질문 v09)과 Vertex 결과는 같은 모델 이름이라도 출력이 달라
  (`results_clarify_v11_prebonus`) 직접 비교하지 않는다. Vertex 기준선은 `experiments/reranking/results_v49_vertex_baseline/RUN_INFO.md`
  — 같은 분석 캐시·같은 후보에서 리랭커 경로만 바꿔도 dev 176 중 21건의 순위가 움직였다(Hit@1 72 → 75 · Hit@10 113 → 115, 두 번 동일).
  **Vertex 전체 경로(분석·리랭킹 모두, 2026-10-15) 기준선: dev 176 후보 119 · Hit@1 80 · Hit@5 105 · Hit@10 114 · MRR 0.522, 두 번 동일,
  리랭킹 실패 0, 리랭킹 중앙값 약 5.0초 · p95 약 7초.** Context 색인 없는 5,143곡 색인·현행 스위치 기준이며, 이후 설정 비교는 이 캐시·색인에 고정해 견준다.
- 봉인 test(v10 42건)는 열지 않는다.
