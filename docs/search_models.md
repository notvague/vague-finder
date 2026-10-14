# 검색 모델 설정

기본 검색 모델은 Claude Sonnet이다. 화면의 선택창 또는 `POST /api/v1/search` 본문의
`model_provider`(`claude` / `gemini`)로 **질의 분석과 리랭킹을 함께** 선택한다.
생략하면 `SEARCH_MODEL_PROVIDER`를 사용한다. 설정 변경 뒤 서버를 재시작한다.

```dotenv
SEARCH_MODEL_PROVIDER=claude
ANTHROPIC_API_KEY=...
CLAUDE_MODEL_NAME=claude-sonnet-5-5
VF_CLAUDE_THINKING=adaptive   # 이 둘만 VF_ 접두어 — CLAUDE_EFFORT는 Claude Code가 셸에 내보내는 이름과 겹친다
VF_CLAUDE_EFFORT=medium
CLAUDE_STRUCTURED_OUTPUT=json_schema
CLAUDE_USAGE_LOG=artifacts/claude_usage.jsonl
# CLAUDE_BUDGET_USD=50

GEMINI_API_KEY=...
GEMINI_MODEL_NAME=gemini-3.1-flash-lite
```

`CLAUDE_API`도 기존 실험과 호환하며 두 키 이름이 모두 있으면 이 값을 우선한다.
키는 서버에만 둔다. `GET /api/v1/search/models`는 모델 이름, 기본 선택, 키 설정 여부만 반환한다.
키가 없는 모델 선택은 503이다. 호출 실패 시 다른 공급자로 바꾸지 않는다.
기본 모델(`SEARCH_MODEL_PROVIDER`)의 키가 없으면 서버가 **기동에서** 멈춘다 — 정상으로 떠 있다가
`model_provider`를 생략한 검색마다 503이 나는 것보다 낫다. Gemini 키만 있으면 `SEARCH_MODEL_PROVIDER=gemini`로 둔다.
기존 규칙 분석/검색 순서 폴백과 서버 시간 제한은 유지한다. 화면 검색 제한은 분석 20초 +
리랭킹 30초 + 검색 여유를 고려해 60초로 두며, 질문 조회는 8초다.

화면에서 바꾼 선택은 다음 새 검색부터 적용된다. 진행 중 대화의 추가 질문·거절·출생 연도 재검색은
응답의 `model_provider`를 유지한다. API 클라이언트도 같은 값을 다음 검색에 보낸다.
모델을 바꾸어 새로 비교하려면 `prior_analysis`와 답변/거절 상태를 비우고 첫 검색부터 시작한다.
서로 다른 모델 요청은 분석기·라우터를 분리하고 임베딩 모델과 Qdrant 클라이언트는 공유한다.

Vertex AI는 제거했다. Gemini SDK에 `vertexai=False`를 명시하므로 GCP 프로젝트, ADC 또는
`GOOGLE_GENAI_USE_VERTEXAI`가 남아 있어도 Vertex로 연결되지 않는다.
옛 `GEMINI_RETRIEVAL_BACKEND=vertex`는 설정 오류이므로 삭제한다. 크롤링의 Gemini 호출도 AI Studio만 사용한다.
크롤링 모델은 `GEMINI_CRAWL_MODEL_NAME`으로 별도 설정하며, 검색 화면의 선택과 무관하다.

## 평가와 비용

Sonnet 실험의 116/176 Hit@10은 **저장된 Gemini 분석 + Sonnet 리랭킹** 결과다.
Claude 분석까지 적용한 이번 검색 경로의 품질을 보장하는 수치가 아니다. Context 색인과 함께 dev에서
별도의 분석 캐시와 기준선을 만들어 평가한다. 기존 Gemini 캐시는 덮어쓰지 않고 봉인 test는 열지 않는다.
캐시 지문·피드백·평가 기록에 선택 공급자와 모델 설정을 남긴다.

분석 캐시의 조건 비교(drift)는 서버 기본 모델이 아니라 **그 캐시의 모델** 기준이다 — 기본이 Claude여도 얼려 둔
Gemini 캐시(`provider`가 없는 옛 캐시는 모두 Gemini)는 어긋난 것으로 보지 않고, `build_analysis_cache`가 빈 항목을
채울 때도 그 캐시의 모델을 쓴다. 새 캐시의 모델은 `--model-provider`, 없으면 `SEARCH_MODEL_PROVIDER`다.
측정 runinfo에는 `model_provider`(실시간 분석·기본 리랭커의 모델)가 남고, Gemini를 부르지 않은 실행은
`gemini.backend`가 `unused`다 — 기준선과 비교하기 전에 두 값을 함께 본다.

CLI는 기본적으로 `SEARCH_MODEL_PROVIDER`를 따른다. 과거 리랭커만 재현할 때는
`RERANKER_BACKEND=gemini_listwise|claude_listwise|cross_encoder`를 명시한다.
이 CLI 덮어쓰기는 웹 요청의 모델 선택에는 적용하지 않는다.
리랭킹 시간 제한·후보 수·가중치 등 기존 `GEMINI_RERANK_*` 옵션은 두 listwise 모델에 공통 적용된다.
Google Search 옵션을 켠 채 Claude를 선택하면 설정 오류를 낸다.

Claude 리랭킹에는 실험에서 검증한 JSON 스키마를 기본 적용한다. 후보 ID 누락·중복 처리는 공용 파서가
담당한다. 질의 분석은 기존 프롬프트·검증·재시도를 유지하고 응답 전체를 감싼 코드 울타리를 벗긴다.
자유 형식 리랭킹 재현 시 `CLAUDE_STRUCTURED_OUTPUT=off`를 사용한다.

`CLAUDE_BUDGET_USD`는 사용량 로그의 누적 비용과 진행 중 요청의 예약 상한을 제한한다.
Console 잔액이나 자동 충전 설정을 바꾸지 않으며, 응답 없는 요청은 상한 추정으로 기록한다.
검색 전체 비용에는 분석 호출과 재시도가 포함된다. 기존 실험의 리랭킹 단가만으로 계산하지 않는다.
