# Claude Sonnet 5.5 리랭킹 비교 — 시범 30건 · dev 176건 (2026-10-14)

Sonnet이 추가 비용만큼 검색 품질을 올리는지 본다. 서비스 기본은 Gemini 그대로이고, 리랭킹만 쟀다.
Haiku 실험(`experiments/haiku_comparison/RUN_INFO.md`)의 같은 질의 · 같은 후보 · 같은 프롬프트에 모델만 바꿨다.
순서: ① 자유 형식(Haiku와 같은 조건) 시범 30건 두 번 → ② structured outputs만 바꾼 시범 30건 → ③ 기준을 넘어 dev 176건 두 번.

## 0. 결론 먼저

- **자유 형식(①)은 형식이 문제였다.** 응답의 26%가 JSON 앞에 설명을 쓰고(같은 질의에서 반복), 다시 묻기 7회 · 검색 순서 폴백 1건
  (첫 보고서의 "6회"는 집계 오류 — §10).
- **structured outputs(②③)로 형식 문제가 없어졌다.** 176건 두 번 응답 322개가 모두 JSON, 형식 때문에 다시 묻기 0, 폴백 0.
  후보를 하나라도 빠뜨린 응답은 2 · 2(파서가 1곡씩 채움). 없는 id를 한 줄 끼운 응답이 18 · 12 있으나 거의 모두 relevance 0인 채움 줄이라 파서가 버린다.
- **dev 176건(서비스 제한 20 · 30초)에서 Gemini 저장본보다 일관되게 높다.** Top-1 79 · 78(Gemini 72) · Top-5 107 · 106(103) ·
  Hit@10 116 · 116(113) · MRR 0.523 · 0.521(0.492). 세 세트 모두 Gemini 이상, 두 번 모두 같은 방향.
  다만 dev에서만 쟀고, 질의별 우열 22 : 10 · 22 : 11은 부호 검정 p = 0.05 · 0.08(Top-1만 보면 14 : 7 · 14 : 8, p = 0.19 · 0.29) — 결정적이지 않다.
- **운영 지표**: 폴백 0/306, 리랭킹 중앙값 8.1초 · p95 12.9초 · 최대 16.9 · 29.2초(Gemini 4.3~4.5 · 5.5~6.5 · 9.4~25.7초).
  20초 제한에 걸린 호출 1/323(다시 물어 29.2초에 성공 — 30초 예산 안). 같은 설정 두 번에서 정답 순위가 같은 질의 168/176(Gemini 176, Haiku 154).
- **비용**: 모델을 부른 질의당 $0.046 · $0.048(Gemini 약 $0.0050의 약 9.5배). 검색 1,000번에 약 $41(Gemini 약 $4.3 · Haiku 약 $2.8).
- **판단**(사용자 기준 — 비용보다 품질 · 속도 · 안정성): 품질 ○(일관된 우위, 크기는 Top-1 +4%p), 안정성 ○(폴백 0 · 반복 168/176),
  **속도 ×**(중앙값 +3.7초 · p95 약 2배, 30초 예산 여유가 0.8초까지 줄어든 질의 1건). 남은 쟁점은 속도다 — 받아들일 수 있으면 리랭커 후보로
  올리되, 완성된 Context 색인의 최종 기준선 위에서 다시 잰 뒤 봉인 v10 test 42건으로 한 번 확인한다 — §9.
- Sonnet 사용액 **$19.32**(한도 $50, 무효 처리한 실행 $1.06 포함). Google 신규 호출 **0회**.

## 1. 비교 조건

| | 내용 |
|---|---|
| 기준 커밋 | `8ea9706`(experiment/haiku-5-5 — Haiku 결과 커밋). 이 브랜치 `experiment/sonnet-5-5`는 그 위에서 시작했다 |
| 질의 | 시범: `experiments/haiku_comparison/pilot_queries.csv` dev 30건(v06 · v09 · v10 각 10). 확대: dev 176건 전부(v06 57 · v09 59 · v10 60). 봉인 test는 실행하지 않았다 |
| 분석 | `experiments/reranking/results_v46_main_after_context/analysis_cache_{세트}_dev_ctx.json` — Haiku · Gemini 저장본과 같은 캐시. 폴백 0 |
| 색인 | `artifacts/qdrant_5143_baseline_backup`을 이 워크트리에 새로 복사(`artifacts/qdrant_snapshot_5143_pre45`, 무료 점검용 `_b`). 파일 해시가 Haiku 워크트리의 복사본 · 원본 백업과 같다 |
| 가사 DB | 내용 해시 `e0d58f28f1f4a563`(5,143곡) — 모든 유효 실행 · 저장된 Gemini · Haiku 실행이 같다. 실행기가 시작 전에 확인한다(§8) |
| 후보 | 30건 · 176건 모두 `candidate_digest`가 Gemini 저장본 · Haiku 실행과 같다(유효 실행 전부) |
| 정책 생략 | 시범 7건 · 176건 중 23건(image_dominant 등) — 모든 모델이 같다. 모델을 부른 질의 23 · 153건 |
| 프롬프트 · 후처리 | 같다. 같은 질의의 프롬프트 지문 · 입력 토큰 수가 Haiku와 같다(24/24) |
| 요청 설정 | adaptive thinking · effort medium · max_tokens 16,000 · 샘플링 값 안 보냄 · 코드 울타리 벗김. ②③은 여기에 structured outputs(`output_config.format`)만 더했다 |
| 시간 제한 | ①과 Haiku: 요청 60초 · 리랭킹 90초. **②③과 Gemini 저장본: 서비스 제한 20초 · 30초** |
| 순서 | 유료 실행은 한 번에 하나 · 세트도 하나씩. 무료 점검(모든 호출 거절)만 다른 색인 복사본으로 나란히 돌렸다 |

비교 대상(질의별 detail · 응답 원문은 git에서 제외된 자료라 Haiku 워크트리 `../vague-finder-haiku-5-5`에 있다 — 지우지 않는다):

| 이름 | 날짜 · 경로 | 제한 | 원문 · 토큰 |
|---|---|---|---|
| Gemini 저장본 rep1(후보 기준) · rep2 — 176건 | 10/13 21:41 · 22:03, AI Studio | 20 · 30 | 둘 다 없음 |
| Gemini 재호출 `pilot_gemini_metered` — 시범 30건 | 10/14 11:14, AI Studio(Haiku 실험 때) | 60 · 90 | 토큰만 |
| Haiku `pilot_claude_adaptive_medium` — 30건 | 10/14 11:19 | 60 · 90 | 둘 다 |
| Haiku `dev176_claude_adaptive_medium` · `_rep2` — 176건 | 10/14 11:28 · 11:52 | 60 · 90 | 둘 다 |
| Sonnet ① `pilot_sonnet_adaptive_medium` · `_rep2` | 10/14 13:37 · 13:44 | 60 · 90 | 둘 다(프롬프트 포함) |
| Sonnet ② `pilot_sonnet_structured` | 10/14 16:16 | 20 · 30 | 둘 다 |
| Sonnet ③ `dev176_sonnet_structured` · `_rep2` | 10/14 16:26 · 16:51 | 20 · 30 | 둘 다 |

이번에 Gemini는 한 번도 부르지 않았다. Gemini 숫자는 모두 저장된 결과다.

## 2. 공식 문서로 확인한 것 (2026-10-14)

- 모델 ID `claude-sonnet-5-5`(날짜 없는 고정 ID). <https://platform.claude.com/docs/en/about-claude/models/overview>
- 가격(100만 토큰당): 입력 $2 · 5분 캐시 쓰기 $2.50 · 1시간 캐시 쓰기 $4 · 캐시 읽기 $0.10 · 출력 $10. 긴 프롬프트 단가 없음.
  <https://platform.claude.com/docs/en/about-claude/pricing>
- thinking: adaptive 지원. Sonnet 5.5에서 생각을 끄려면 `"disabled"`가 아니라 `{"type":"between_tools"}` — 쓰지 않았다.
- effort: low~max 모두 지원(`output_config.effort`). **API 기본값은 high**(Haiku 5.5는 medium) — medium을 직접 보냈다. thinking은 `max_tokens`에 포함된다.
  <https://platform.claude.com/docs/en/build-with-claude/effort>
- structured outputs(`output_config.format` · `json_schema`): Sonnet 5.5 지원. 문법은 답 본문에만 걸리고 thinking에는 걸리지 않는다(thinking과 함께 쓸 수 있다).
  객체마다 `additionalProperties: false`, 수치 · 길이 제약 · 1을 넘는 `minItems`는 쓸 수 없다. 형식 안내 system 프롬프트가 붙어 입력 토큰이 늘고
  (설정 확인에서 약 350토큰), 스키마를 처음 쓸 때 문법 컴파일 지연(24시간 캐시)이 있다. 거절 · `max_tokens` 도달이면 스키마를 따르지 않을 수 있다.
  <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- 설정 확인 호출 3회: medium(200) · 리랭킹 스키마를 붙인 medium(200, 스키마 모양 그대로 답함). 그 전에 1회가 **xhigh로 나갔다** —
  Claude Code 세션이 `CLAUDE_EFFORT=xhigh`를 셸에 내보내는데 이름이 어댑터 변수와 같아 `env.sh`의 `${CLAUDE_EFFORT:-medium}`이 물려받았다.
  이제 `env.sh`는 바깥 `CLAUDE_*`를 물려받지 않는다(`SONNET_*`로만 바꾼다). 모든 측정 실행은 effort medium으로 나갔다(호출 기록 · runinfo).
  Haiku 측정 실행도 모두 의도한 effort로 기록돼 있다 — Haiku 보고서의 smoke 1건(`disabled` + `xhigh` → 400)은 같은 원인이었을 가능성이 높다.

## 3. Google 신규 호출 차단

Vertex AI 크레딧이 끝나, 이번 실험에서 허용한 유료 호출은 Anthropic뿐이다. 설정값에 기대지 않고 **프로세스 안에서 거절**했다
(`google_guard.py`, 측정은 모두 `run_guarded.py`로).

| 겹 | 하는 일 |
|---|---|
| 자격 증명 비우기 | 프로젝트 코드 import 전에 `GEMINI_API_KEY` · `GOOGLE_API_KEY` · `GCP_PROJECT_ID` · `GOOGLE_CLOUD_PROJECT` · `GOOGLE_APPLICATION_CREDENTIALS` 등을 **빈 값**으로(빈 값이라야 `load_dotenv`가 다시 채우지 않는다). `CLOUDSDK_CONFIG`는 빈 임시 폴더. `.env` · gcloud 설정은 건드리지 않았다 |
| 클라이언트 생성 거절 | `google.genai.Client` · `make_genai_client` · `google.auth.default` → `GoogleCallBlocked`(**BaseException** — `except Exception` 폴백이 삼키지 못한다) |
| 주소 거절 | `socket.getaddrinfo`가 Google 도메인을 찾으면 같은 예외. 그 밖의 주소는 상위 도메인으로 세어 보고에 남긴다 |
| 조용한 대체 막기 | 분석 캐시 없는 실행 · `--allow-fallback-analysis` · `RERANKER_BACKEND≠claude_listwise` · Claude 키 없음 · **가사 DB를 못 읽거나 내용이 다름**을 시작 전에 거절, 분석기를 만들려 하면 멈춤 |

Google 클라이언트는 모두 `genai.Client` 한 곳에서 만들어진다(`src/common/gemini_client.py` — 질의 분석기 · Gemini 리랭커 · 크롤링 정제 ·
Context 추출기). 이번 평가 경로는 분석 캐시를 쓰므로 분석기를 만들지 않고, 리랭커는 Claude 어댑터를 받는다. 임베딩은 로컬 모델이다.
서버 예열은 Gemini를 부르지 않고, 기동은 `GEMINI_RETRIEVAL_BACKEND` 값의 형식만 본다(서버는 띄우지 않았다 — 코드로만 확인).

검증: 가드 시험 14건(새 프로세스마다 — 생성 · 팩토리 4종 · Google 주소 5종 · `.env` 뒤에도 빈 값 · 실행기 거절 4가지 · 가사 DB 거절) 통과.
모든 실행 · 세트의 가드 보고(`results/rerank/*/guard_{세트}.json`) — **막은 시도 0, 찾은 주소는 anthropic.com · mongodb.net(가사 DB) · huggingface.co뿐.**
runinfo의 Gemini 경로가 `none`으로 남는 것은 가드가 키를 비웠기 때문이다(Haiku 실행은 `api_key`).

## 4. 예산 · 호출 기록 · 시간 제한 · 형식 (`src/common/claude_client.py`, `src/retrieval/claude_listwise_reranker.py`)

모두 실제 모델 없이 mock으로 먼저 확인했다.

| 고친 것 | 전 | 후 |
|---|---|---|
| 단가를 모르는 모델 | 비용 None → 0 | 예산이 걸려 있으면 부르기 전에 거절 |
| 호출 전 상한 | 입력 = 글자 수 × 1토큰 | 입력 = **UTF-8 바이트 + 2,048 (+ 스키마 바이트)**, 입력 단가 = 입력 · 캐시 쓰기 중 비싼 쪽, 출력 = `max_tokens`(thinking 포함) |
| 동시 실행 | 기록만 보고 판단 | 부르기 전에 상한을 **예약**, 끝나면 실제 비용으로. 파일 잠금 안에서 — 스레드 · 프로세스끼리도. 죽은 예약은 상한으로 계속 셈 |
| 답을 못 받은 호출 | 시간 초과만 상한 | 0은 보내기 전 연결 실패 · 4xx뿐. 시간 초과 · 취소 · 보낸 뒤 끊김 · 5xx · usage 없는 200은 상한(`cost_basis`) |
| 요청 시간 | `aio`가 없어 리랭커 취소 경로를 건너뜀, 읽기 제한(조각 하나)만 | 루프 스레드에서 **요청 전체 시간**을 끊고, `aio` 면으로 리랭커의 `wait_for(min(요청 제한, 남은 예산))`가 진행 중인 요청을 취소. 연결 재사용 |
| structured outputs | 없음 | `CLAUDE_STRUCTURED_OUTPUT=json_schema`면 리랭커가 순위 호출에만 `RANKING_SCHEMA`를 붙이고 어댑터가 `output_config.format`으로 보낸다. 프롬프트 글자 · 파서 · 후처리는 그대로. 후보 id를 enum으로 묶지 않았다(형식만 바꾸려고) |

mock 시험: `tests/test_claude_client.py` +17 · `tests/test_claude_listwise_reranker.py` +3(리랭킹 예산 0.5초가 요청 제한 60초보다
먼저 끝나면 2초 안에 검색 순서 + failed · 스키마는 순위 호출에만 붙고 프롬프트는 같음 · 스키마가 제약을 지킴 등).

실제 기록으로 본 상한: Sonnet 435회(usage 있는 434회) 모두 실제 비용 / 상한 ≤ 0.21, 입력 토큰 / 입력 상한 ≤ 0.43. Haiku 1,107회를 같은 공식(글자 수 기준 — 더
작게 잡힌다)으로 다시 계산해도 ≤ 0.37 · ≤ 0.63 — **상한이 실제를 밑돈 호출 0**. 끝난 뒤 열린 예약 $0.

## 5. 시범 30건 — 자유 형식(①)과 structured outputs(②)

| | Top-1 | Top-5 | Hit@10 | MRR@10 | 폴백 | 다시 묻기 | 리랭킹 중앙값 · p95 · 최대 | 질의당 |
|---|---|---|---|---|---|---|---|---|
| 리랭킹 전 | 9 | 15 | 20 | 0.397 | | | | |
| Gemini 저장 · 재호출(3회 같음) | 12 | 19 | 22 | 0.509 | 0 | 0 | 4.6 · 5.4 · 6.6초(20 · 30) | $0.0050 |
| Haiku 3회 | 11~12 | 19 | 21~23 | 0.496~0.518 | 0 | 0 | 12.9 · 16.5 · 17.8초 | $0.0032 |
| Sonnet ① 자유 형식 2회 | 13 · 13 | 19 | 23 · 23 | 0.532 · 0.533 | **1 · 1** | 7 · 7 | 7.8 · 19.1 · 33.2초 / 7.5 · 19.1 · 33.7초 | $0.058 |
| **Sonnet ② structured**(20 · 30초) | **14** | 19 | 23 | **0.563** | **0** | **0** | 8.0 · 11.7 · 12.7초 | $0.047 |

형식(응답 원문 — 파서가 채운 응답은 완전한 응답으로 세지 않는다):

| | 응답 | 그대로 JSON | 울타리 벗긴 뒤 | 후보 30곡 정확히 한 번씩 | 리랭커가 쓴 응답: 완전 · 파서가 채움 · 없는 id/중복을 버림 |
|---|---|---|---|---|---|
| Haiku 시범 | 24 | 11 | 24 | 20 | 19 · 2(1곡 · 23곡) · 2 (23건 중) |
| Sonnet ① 1회 · 2회 | 31 · 31 | **23 · 23** | 23 · 23 | 19 · 22 | 18 · 1 · 3 / 21 · 0 · 1 (22건 중 — 1건은 쓸 응답이 없었다) |
| **Sonnet ②** | 24 | **24** | 24 | **24** | **23 · 0 · 0** (23건 중) |
| Gemini | 원문 없음 | (JSON 스위치 — 재호출 24회에 다시 묻기 0) | | 모름 | 모름 |

- ①의 형식 실패: 응답 8/31(두 번 모두 같은 질의 q312 · n031 · n038 · t025 · t042 · n096×3)이 JSON 앞에 설명을 썼다 — "The drama is
  태양의 후예 … Its OSTs include …", "The film is 봄날은 간다 (2001) …"처럼 바깥 지식을 짚어 추론한 뒤 JSON을 썼다. thinking 토큰은 0.
  n096('라면 먹을래요?' 영화 주제가)은 세 번 모두 그래서 검색 순서로 돌아갔다(정답이 후보 밖이라 지표 영향은 없다).
- ②에서는 형식 실패가 없고, ①에서 설명을 앞세우던 질의도 첫 응답에 JSON으로 답했다. 지표도 ①보다 낮아지지 않았다(Top-1 14 · MRR 0.563).
  thinking 토큰이 조금 생겼다(호출당 34 — ①은 0).

**176건으로 넓히는 기준**(②를 돌리기 전에 정한 것): 모델을 부른 23건에서 ⓐ 모든 응답이 JSON · 형식 때문에 다시 묻기 0 · 폴백 0,
ⓑ 파서가 채우거나 버린 응답이 ① 수준(2건) 이하, ⓒ Top-1 ≥ 12 · Hit@10 ≥ 22 · MRR ≥ 0.509(Gemini 이상), ⓓ 서비스 제한에서 시간 초과 폴백 0.
②는 넷 다 넘었다 → 176건 두 번.

①의 두 번째 실행을 고른 근거(①을 돌린 뒤): 지시의 두 갈래("품질 · 형식이 괜찮으면 반복" / "지연이 주요 문제면 low") 어느 쪽도 맞지 않았다
(형식 26% 실패, 지연은 다시 묻기 때문). effort를 바꾸면 형식 문제 위에 변수가 하나 더 붙고, 형식 실패가 되풀이되는지가 다음 실험을 정하므로
같은 설정을 한 번 더 돌렸다 — 형식 실패 · 다시 묻기 · 폴백이 두 번 모두 같았다. adaptive + low는 돌리지 않았다.

## 6. dev 176건 — structured outputs 두 번(③), 서비스 제한 20 · 30초

### 6-1. 품질 (후보 30 안에 정답 118건 — 모든 실행 같음)

| | Top-1 | Top-5 | Hit@10 | MRR@10 | 폴백 |
|---|---|---|---|---|---|
| 리랭킹 전 | 44 | 77 | 98 | 0.343 | |
| Gemini 저장 2회(순위 176/176 같음) | 72 | 103 | 113 | 0.492 | 0 · 0 |
| Haiku 2회(60 · 90초) | 69 · 70 | 106 · 101 | 117 · 114 | 0.486 · 0.481 | 0 · 0 |
| **Sonnet structured 2회** | **79 · 78** | **107 · 106** | **116 · 116** | **0.523 · 0.521** | **0 · 0** |

세트별(Top-1 · Hit@10 · MRR):

| | v06(57) | v09(59) | v10(60) |
|---|---|---|---|
| Gemini | 28 · 41 · 0.563 | 26 · 38 · 0.512 | 18 · 34 · 0.406 |
| Haiku 2회 | 25~26 · 41~43 · 0.552~0.553 | 23~24 · 38~39 · 0.471~0.491 | 20~21 · 35 · 0.419~0.423 |
| Sonnet 2회 | **31 · 42 · 0.613** (두 번 같음) | 27 · 39 · 0.522 | 21 · 20 · 35 · 0.439 · 0.434 |

질의별(Gemini 저장본 대비, 1회 · 2회): 나음 22 · 22, 같음 144 · 143, 나쁨 10 · 11. Top-1을 새로 얻음 14 · 14, 잃음 7 · 8. Top-10에 새로 듦 3
(q205 · n091 · t008), 빠짐 0. 부호 검정(두 쪽): 나음/나쁨 p = 0.05 · 0.08, Top-1 얻음/잃음 p = 0.19 · 0.29. Haiku 1회 대비 나음 21 · 22, 나쁨 10 · 9.

- Top-1을 새로 얻은 질의에는 **곡에 대한 바깥 지식**을 쓰는 것이 많다 — 드라마 · 사극 OST(c707 · n031 · t037), "노래방에서 꼭 부르는"
  · "전 세계가 따라 춘" 같은 인지도(c601 · n097 · t004 · t008), 표지 묘사(n089 · t101 · t104).
- 잃은 질의에는 **가수 이름을 직접 댄 질의**가 셋 있다(q208 휘성 · q218 민경훈 · t084 잔나비 — Gemini 1위, Sonnet 2~4위). 나머지는 표지 · 소리 묘사.
  30건 · 176건 어디서도 "이 종류만 Sonnet" 하고 고를 깔끔한 경계는 보이지 않았다.

### 6-2. 시간 · 실패 (서비스 제한 그대로)

| | 리랭킹 중앙값 · p95 · 최대 | API 호출 중앙값 · p95 · 최대 | 20초 넘김 · 30초 넘김 | 폴백 | 시간 초과로 끊긴 호출 |
|---|---|---|---|---|---|
| Gemini 저장 rep1 · rep2 (10/13 밤) | 4.3 · 5.5 · 9.4초 / 4.5 · 6.5 · 25.7초 | 기록 없음 | 0 · 0 / 1 · 0 | 0 · 0 | 모름 |
| **Sonnet 1회 · 2회** (10/14 오후) | **8.1 · 12.9 · 16.9초 / 8.1 · 12.9 · 29.2초** | 7.9 · 12.0 · 14.8초 / 8.0 · 11.6 · 15.1초 | 0 · 0 / 1 · 0 | **0 · 0** | 0 · **1**(n082) |

- 2회차 n082(분홍 바탕 표지 질의)의 첫 호출이 20초 요청 제한에서 끊겼고(`cancelled` — 상한 비용으로 셈), 다시 물어 리랭킹 전체 29.2초에
  성공했다. 30초 예산을 0.8초 남겼다 — 한 번 더 느렸으면 폴백이었다. 323회 중 1회(0.3%).
- 지연 비용: 중앙값 +3.7초, p95 약 2배. Haiku(60 · 90초 측정)는 중앙값 13.5~13.7초 · 20초 넘김 6 · 2였다.
- **조건 차이**: Gemini 저장본은 전날 밤, Sonnet은 다음 날 오후다. Gemini 리랭커는 호출마다 연결을 새로 맺고, Sonnet 어댑터는 연결을 재사용한다.
  이번에 Gemini를 다시 잴 수 없어 같은 시각 값은 없다.

### 6-3. 형식 (176건, 응답 원문)

| | 응답 | JSON | 후보 30곡 정확히 한 번씩 | 리랭커가 쓴 153건: 완전 · 파서가 채움 · 없는 id/중복을 버림 | 없는 id를 낀 응답 · 중복 id 응답 |
|---|---|---|---|---|---|
| Haiku 2회(자유 형식 + 울타리 벗김) | 163 · 161 | 161 · 161 | 128 · 130 | 122 · 12 · 19 / 123 · 22 · 8 | — |
| **Sonnet structured 2회** | 161 · 161 | **161 · 161** | 143 · 149 | **135 · 2 · 16 / 142 · 2 · 9** | 18 · 12 / 1 · 3 |

- 없는 id는 거의 모두 **relevance 0인 채움 줄**이다(28/28 · 18/19) — 실제 id 끝에 `x`를 붙이거나 앞부분만 쓴 꼴이 절반, 나머지는 `10000000`
  같은 숫자. 파서가 버리므로 나머지 후보의 상대 순서는 그대로다. 같은 접두의 실제 id가 그 줄보다 **뒤에** 나온 경우가 5 · 5 있다(낮은 순위 구간).
- 파서가 빠진 후보를 채운 응답은 2 · 2(각 1곡). Haiku는 12 · 22(한 응답에서 최대 26 · 30곡).

### 6-4. 반복 변동

| | 정답 순위가 같은 질의 | Top-1 여부가 갈린 질의 | Top-10 여부가 갈린 질의 | 상위 10곡 목록이 통째로 같은 질의 |
|---|---|---|---|---|
| Gemini 2회 | 176/176 | 0 | 0 | 175/176 |
| Haiku 2회 | 154/176 | 5 | 3 | 24/176 |
| **Sonnet structured 2회** | **168/176** | 1(t039) | 0 | 24/176 |

Sonnet은 정답 순위는 거의 같게 내지만(온도 0을 줄 수 없어) 정답이 아닌 곡의 순서는 자주 바뀐다.

### 6-5. 토큰 · 비용 (정가, 호출 기록에서 — 다시 묻기 · 끊긴 호출 포함)

| | 호출 | 입력 · 출력 토큰(성공 호출당) | thinking | 실행 비용 | 모델을 부른 질의당 | 검색 1,000번(질의의 87%가 모델 호출) |
|---|---|---|---|---|---|---|
| Gemini | 기록 없음 | (시범 재호출: 10,598 · 1,420) | 0 | (시범 단가로 약 $0.77) | 약 $0.0050(시범 재호출 기준 추정) | 약 $4.3 |
| Haiku 2회 | 163 · 161 | 16,013 · 2,920 | 1,038 | $0.499 · $0.491 | $0.0033 · $0.0032 | 약 $2.8 |
| **Sonnet structured 2회** | 161 · 162 | 16,344 · 1,144~1,159 | 33~38 | **$7.105 · $7.391** | **$0.046 · $0.048** | **약 $41** |

- 입력 토큰은 자유 형식보다 호출당 약 350 많다(structured outputs의 형식 안내 system 프롬프트). 대신 다시 묻기가 사라져 질의당 $0.058 → $0.047.
- 같은 프롬프트에서 Gemini는 Claude보다 입력 토큰을 약 35% 적게 센다(토크나이저가 다르다) — 토큰 수로 속도를 견주지 않는다.
- 176건 실측은 처음 예상한 두 번 약 $18보다 적은 $14.50이었다(다시 묻기가 사라져서).

### 6-6. 이 실험 전체가 쓴 것 (`results/claude_usage.jsonl`, 한도 $50)

| | 비용 |
|---|---|
| 설정 확인 3회 | $0.0026 |
| ① 자유 형식 시범 2회 | $2.6857 |
| ② structured 시범 — 무효 처리한 1회(§8) · 다시 잰 1회 | $1.0608 · $1.0704 |
| ③ dev 176건 2회 | $14.4967 |
| **합계** | **$19.316** (호출 435회 — 상한으로 센 호출 1) |

## 7. 대표 사례 (순위: Gemini | Haiku 2회 | Sonnet structured 2회, `-`는 10위 밖)

- 나아진 것: c707 후각이 특별한 주인공 드라마 OST 2 | 6 · - | **1 · 1** · n031 「천국의 계단」 OST 2 | 2 · 2 | **1 · 1** ·
  t008 이별 후 취해서 우는 남성 듀오 R&B - | 1 · 1 | **1 · 1** · q209 공기반 소리반 여성 솔로 OST 5 | 2 · 2 | **1 · 1**
- 나빠진 것: q208 휘성 · 원곡이 팝송 1 | 2 · 2 | 4 · 3 · n029 낮고 허스키한 여성 · 남자 아이돌 작사 1 | 4 · 5 | 7 · 6 ·
  t084 잔나비 · 제목 꼴 1 | 2 · 1 | 3 · 3
- 전체 질의별 순위는 `results/compare_dev176_per_query.csv`.

## 8. 무효 처리한 실행 — 가사 DB 연결 실패

16:09에 돌린 structured 시범(`results/rerank/pilot_sonnet_structured_service`, $1.06)은 비교에서 뺐다. 시스템의 **첫 DNS 서버가 가사 DB(MongoDB
Atlas)의 SRV 레코드를 돌려주지 않아**(두 번째 DNS 서버와 다른 공개 DNS는 돌려줬다 · 13:37~13:50 실행은 문제없었다) 가사 DB를 읽지 못했고,
평가기는 가사 구절 일치 경로를 **빈 결과로 대체한 채 측정을 이어 갔다**. 가사 단서가 있는 q209 · t025의 후보 지문이 달라 집계기가 잡았다.
- 그 뒤 실행은 이 프로세스의 SRV 조회만 시스템의 두 번째 DNS 서버(`SRV_DNS_NAMESERVERS=168.126.63.1`)로 물었다 — 시스템 DNS 설정은 바꾸지 않았다.
- 실행기가 시작 전에 가사 DB를 평가기와 같은 방법으로 읽어 내용 해시(`EXPECT_LYRICS_SHA`)를 확인하고, 못 읽거나 다르면 모델을 부르기 전에 멈춘다
  (시험 · 실제 거절 확인). ②③의 모든 세트가 해시 `e0d58f28f1f4a563`을 확인하고 시작했다.
- 형식 결과는 참고로만: 그 실행도 응답 23/23이 JSON, 다시 묻기 0이었다.
- 그 실행의 runinfo 오류 문구에 가사 DB 클러스터 주소가 찍혀 있어 그 부분만 지웠다(`<가사 DB 클러스터 주소 — 지움>`).

## 9. 판단

- **품질**: structured outputs를 붙인 Sonnet은 dev 176건에서 처음으로 Gemini보다 **일관되게** 높다(두 번 · 세 세트 · 질의별 22 : 10~11).
  크기는 Top-1 +6~7(+4%p) · MRR +0.03 · Hit@10 +3. 다만 ① dev에서만 쟀고(이 dev 세트로 Gemini 기준 설정을 다듬어 왔다), ② 부호 검정이
  경계선이며(p 0.05~0.08, Top-1만 보면 0.19~0.29), ③ 봉인 test는 열지 않았다.
- **운영(서비스 제한)**: 폴백 0/306 · 실패율 0%. p95 12.9초로 20초 안이지만, 20초에 끊긴 호출 1회가 30초 예산을 0.8초 남기고 성공했다 —
  느린 날에는 폴백이 생길 수 있다. 중앙값 +3.7초는 사용자가 느끼는 검색 시간에 그대로 더해진다.
- **반복 변동**: 정답 순위 168/176이 같다 — Haiku(154)보다 안정적이고 Gemini(176)보다는 덜하다.
- **비용**(참고 — 사용자는 비용보다 품질 · 속도 · 안정성을 보기로 했다): 검색 1,000번에 약 $41(Gemini 약 $4.3).
- **권고**: 판단을 가르는 것은 속도다. 리랭킹이 검색마다 중앙값 3.7초 늘고 p95가 약 2배다. 이것을 받아들일 수 있으면 Sonnet structured를
  리랭커 후보로 올린다. 다만 기본값을 바로 바꾸지는 않는다.
  1. 이번 비교는 과거 조건(Context 없는 v46 후보 · 분석 캐시)에서 쟀다. 완성된 Context 색인 위의 최종 기준선이 아직 없으므로(NEXT_WORK §1),
     그 기준선을 정할 때 Gemini와 Sonnet을 같은 조건으로 함께 잰다 — 그때는 Gemini 재호출이 필요하므로 호출 허용을 먼저 정한다.
  2. 보지 않은 질의 확인은 봉인 v10 test 42건으로 — 최종 코드 · 색인 · 설정을 고정한 뒤 한 번만(v09 test 38건은 이미 평가해 홀드아웃이 아니다).
  3. 운영 쪽: 요청 제한 20초는 그대로 두되 30초 예산의 여유가 얇았던 사례(§6-2)를 기억한다. Claude 리랭커만 바꿔도 질의 분석은 Gemini 경로로 남는다
     — Google 없이 서비스 전체가 동작하는 것은 아니다.
- **어려운 질의에만 Sonnet**: 이득이 바깥 지식(OST · 인지도) 질의에 많고 가수 이름을 댄 질의에서 잃는 경향은 보이지만, 경계가 깔끔하지 않고
  고를 기준(라우터)을 새로 만들어야 한다. 지금 근거로는 권하지 않는다.

## 10. 재지 못한 것

- **Gemini 신규 측정 없음**(이번 실험의 규칙) — Gemini 숫자는 모두 저장본이다. 같은 시각의 Gemini 시간, Gemini 응답 원문의 후보 완전성,
  Gemini 176건의 실제 토큰 · 비용(시범 30건 단가로 추정)은 모른다.
- 봉인 test(v10 42건), 질의 분석 · 정제 · 댓글 선별, 재질문 2턴, effort low · high, Sonnet thinking 강제(high 이상), 프롬프트 수정, Batch · 프롬프트 캐시.
- 서비스 서버에서의 실제 검색 시간(분석 + 검색 + 리랭킹)은 재지 않았다 — 리랭킹 단계 시간만이다.
- 자유 형식 Sonnet(①)은 60 · 90초 제한으로 쟀다. 호출 하나가 20초를 넘은 적이 없어 서비스 제한이었어도 결과가 같았을 것이다(추정).
- 전체 pytest: 돌리지 않았다. 고친 코드의 시험만 자격 증명을 비우고 네트워크를 막은 채 돌렸다 — 97 통과. Gemini 클라이언트를 실제로 만드는 시험
  9건(이번에 고치지 않은 파일)은 가드가 생성을 거절해 뺐다. 처음 한 번은 플러그인을 만들기 전에 Claude 시험 두 파일(48건, 모두 mock)을
  `.env`가 읽힌 상태로 돌렸다 — 네트워크로 나간 호출은 없다.
- `.env`의 Claude 키가 어느 조직 · 크레딧에 묶였는지, 잔액은 확인하지 않았다. 비용은 모두 정가 계산이다.
- **집계 수정**: 첫 보고서는 다시 묻기를 "같은 프롬프트 앞 4,000자 지문의 두 번째 이후 호출"로 셌다. 프롬프트가 4,000자보다 짧으면(t025 —
  후보가 적은 보조 호출) 정정 문장이 지문에 들어가 새 프롬프트로 세어져 ①의 다시 묻기가 6회로 나왔다 — 실제는 7회(최초 24 + 다시 묻기 7 = 31).
  이제 프롬프트 원문이 있으면 정정 문장으로 센다(Sonnet 실행 전부). 원문이 없는 Haiku · Gemini는 옛 방식이며 집계표에 방식을 함께 적는다.
  비용 · 검색 지표는 바뀌지 않는다.

## 11. 실행 방법

```zsh
cd ../vague-finder-sonnet-5-5      # 이 워크트리. .env · venv · data는 원본 체크아웃을 연결, 색인은 복사본

# 0) 시험 — 자격 증명을 비우고 네트워크를 막은 채(모델 호출 없음)
GOOGLE_GUARD_REPORT=experiments/sonnet_comparison/results/guard_pytest.json PYTHONPATH=experiments/sonnet_comparison \
  venv/bin/python -m pytest -p pytest_offline -p no:cacheprovider -q \
  tests/test_claude_client.py tests/test_claude_listwise_reranker.py tests/test_gemini_listwise_reranker.py \
  experiments/sonnet_comparison/test_google_guard.py \
  --deselect tests/test_gemini_listwise_reranker.py::test_client_is_built_with_http_timeout

# 공통: 가사 DB SRV 조회가 시스템 첫 DNS에서 실패하면 이 프로세스만 다른 DNS로(시스템 설정은 그대로)
export SRV_DNS_NAMESERVERS=168.126.63.1

# 1) 모델을 부르지 않는 점검 — 모든 호출을 예산으로 거절시키고 후보 지문 · 가드 · 가사 DB만 본다(다른 색인 복사본이면 나란히 돌려도 된다)
QDRANT_PATH=artifacts/qdrant_snapshot_5143_pre45_b NAME=dryrun_dev176_no_call ALL=1 SONNET_STRUCTURED=json_schema \
  SONNET_BUDGET_USD=0.000001 SONNET_USAGE_LOG=experiments/sonnet_comparison/results/rerank/dryrun_dev176_no_call/usage_refused.jsonl \
  GEMINI_RERANK_TIMEOUT_SECONDS=20 GEMINI_RERANK_BUDGET_SECONDS=30 zsh experiments/sonnet_comparison/run_rerank.sh

# 2) 설정 확인 1회(유료, 약 $0.002) — SONNET_STRUCTURED를 주면 리랭킹 스키마를 붙인다
source experiments/sonnet_comparison/env.sh && CLAUDE_STRUCTURED_OUTPUT=json_schema \
  GOOGLE_GUARD_REPORT=experiments/sonnet_comparison/results/guard_check_structured.json CLAUDE_USAGE_RUN=setup/check_structured \
  $PY experiments/sonnet_comparison/check_settings.py

# 3) 자유 형식 시범(①, 60 · 90초) — 한 번 약 $1.35
NAME=pilot_sonnet_adaptive_medium zsh experiments/sonnet_comparison/run_rerank.sh
# 4) structured 시범(②, 서비스 제한) — 약 $1.07
NAME=pilot_sonnet_structured SONNET_STRUCTURED=json_schema GEMINI_RERANK_TIMEOUT_SECONDS=20 GEMINI_RERANK_BUDGET_SECONDS=30 \
  zsh experiments/sonnet_comparison/run_rerank.sh
# 5) dev 176건(③) — 한 번 약 $7.2, 25분
NAME=dev176_sonnet_structured ALL=1 SONNET_STRUCTURED=json_schema GEMINI_RERANK_TIMEOUT_SECONDS=20 GEMINI_RERANK_BUDGET_SECONDS=30 \
  zsh experiments/sonnet_comparison/run_rerank.sh
#   effort를 바꿀 때는 SONNET_EFFORT=low처럼(CLAUDE_EFFORT는 Claude Code 셸 값과 이름이 겹쳐 env.sh가 무시한다)

# 6) 집계 — 모델 · DB 없음. Gemini · Haiku 자료는 Haiku 워크트리에서 읽는다(--haiku-results)
venv/bin/python experiments/sonnet_comparison/compare_sonnet.py \
  --sonnet pilot_sonnet_adaptive_medium pilot_sonnet_adaptive_medium_rep2 pilot_sonnet_structured \
  --repeat-groups pilot_sonnet_adaptive_medium,pilot_sonnet_adaptive_medium_rep2 --output pilot
venv/bin/python experiments/sonnet_comparison/compare_sonnet.py --queries dev176 \
  --sonnet dev176_sonnet_structured dev176_sonnet_structured_rep2 \
  --repeat-groups dev176_sonnet_structured,dev176_sonnet_structured_rep2 --output dev176
```

산출물(`results/`): `compare_{pilot,dev176}_{summary,format,per_query,head_to_head,repeat,by_set,ledger}.csv`, 호출 기록 `claude_usage.jsonl`
(토큰 · 비용 · 시간 — 본문 없음), 가드 보고 `guard_*.json`. git 제외(로컬에만): 응답 · 프롬프트 원문 `claude_raw.jsonl`(가사 요약이 들어 있다),
질의별 detail · explain · 로그, 예산 예약 · 잠금 파일(`claude_usage.jsonl.pending` — 지우지 않는다).

### 바꾼 파일

- `src/common/claude_client.py` — §4(상한 · 예약 · 실패 비용 · 전체 시간 제한 · `aio` · structured outputs). 기본 경로(Gemini)는 이 모듈을 쓰지 않는다
- `src/retrieval/claude_listwise_reranker.py` — `RANKING_SCHEMA` · `CLAUDE_STRUCTURED_OUTPUT=json_schema`일 때만 순위 호출에 스키마
- `tests/test_claude_client.py` · `tests/test_claude_listwise_reranker.py` — mock 시험 +20, 비용 없는 기록을 0으로 세던 시험 고침
- `.gitignore` — Sonnet 결과 규칙(Haiku와 같은 원칙)
- `experiments/sonnet_comparison/` — `env.sh` · `run_rerank.sh` · `run_guarded.py` · `google_guard.py` · `pytest_offline.py` · `test_google_guard.py` ·
  `check_settings.py` · `compare_sonnet.py` · 이 문서 · `results/`
- Haiku 결과 파일 · main은 고치지 않았다
