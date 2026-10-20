# docs

최상위에는 지금도 쓰는 안내 문서만 둔다. 끝난 작업의 수정·검증 기록은 `history/<주제>/`로 옮긴다.

## 파일명 규칙

- 소문자 snake_case(`vector_backend.md`). 브랜드 자산(`brand/`)은 기존 이름을 따른다.
- 기록 문서(`history/`)는 날짜를 아는 것은 `YYYY-MM-DD_<주제>.md`, 단계 순서로 진행한 것은 `stepNN_<순번>_<주제>.md`.
  짝을 이루는 검증 결과 json은 문서와 같은 앞부분을 쓴다(`…_fix.md` ↔ `…_local_checks.json`).
- 코드·설정·실험 기록이 경로를 직접 쓰는 문서(`data_policy.md`, `feedback.md`, `vector_backend.md`,
  `namuwiki_context_vector_db.md`, `eval/queries.json` 등)는 옮기거나 이름을 바꾸지 않는다.

## 안내 문서

| 문서 | 내용 |
| --- | --- |
| [data_policy.md](data_policy.md) | 수집 데이터 이용 범위 — 저작권·초상권 |
| [feedback.md](feedback.md) | 정답 피드백 수집 · 검토 · 색인 · 검색 재사용 |
| [search_models.md](search_models.md) | 검색 모델 설정 — Gemini on Vertex AI |
| [vector_backend.md](vector_backend.md) | 벡터 DB(Qdrant) 로컬·서버 모드 |
| [namuwiki_context_search.md](namuwiki_context_search.md) | 나무위키 배경지식 검색 설계 |
| [namuwiki_context_dense.md](namuwiki_context_dense.md) | 나무위키 Context KoE5 임베딩 |
| [namuwiki_context_sparse.md](namuwiki_context_sparse.md) | 나무위키 Context BM25 |
| [namuwiki_context_vector_db.md](namuwiki_context_vector_db.md) | 나무위키 Context Qdrant 적재 |
| [manual_namuwiki_review_resolution.md](manual_namuwiki_review_resolution.md) | 나무위키 수동 검토 완료 절차 |
| [crawling_strategy.md](crawling_strategy.md) | 데이터 수집 전략 |
| [git_convention.md](git_convention.md) | Git·PR 규칙 |

## 폴더

| 폴더 | 내용 |
| --- | --- |
| `eval/` | 평가 질의(`queries.json`)와 작성 규칙·양식 |
| `data_schemas/` | 곡 메타데이터 스키마 |
| `brand/` | 로고·파비콘 키트([가이드](brand/README.md)) |
| `images/` | README용 이미지 |
| `history/namuwiki_context/` | 나무위키 Context 8~10단계와 2026-10-05~07 수정·검증 기록([목록](history/namuwiki_context/README.md)) |
