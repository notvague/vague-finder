# 나무위키 Context 작업 기록

Context 검색을 붙이며 남긴 단계별 평가와 수정·검증 기록이다. 끝난 작업이라 고치지 않는다.
현재 절차와 설계는 상위 폴더의 `namuwiki_context_*.md`를 본다.

원래 `docs/` 최상위에 있던 파일을 옮기고 이름만 바꿨다(2026-10-20). 원래 이름에 날짜가 없던 문서는
날짜를 추정하지 않고 단계 번호로 순서를 매겼다. 순서는 본문의 "이전 수정" 언급으로 정했다.

## 8~10단계

| 문서 | 내용 |
| --- | --- |
| [step08_1_evaluation.md](step08_1_evaluation.md) | 부분 코퍼스 Context 검색 평가 |
| [step08_2_nw008_recovery.md](step08_2_nw008_recovery.md) | nw008 분석 폴백 복구 |
| [step08_3_nw013_modality_guard_recovery.md](step08_3_nw013_modality_guard_recovery.md) | 전체 dev 평가 복구: nw013 |
| [step08_4_test_nw016_recovery.md](step08_4_test_nw016_recovery.md) | test 평가 복구: nw016 |
| [step09_full_scope_reload.md](step09_full_scope_reload.md) | 전체 범위 확정, 재생성, Qdrant 최종 적재 |
| [step10_1_evaluation_freeze.md](step10_1_evaluation_freeze.md) | 전체 코퍼스 검색 평가와 설정 동결 |
| [step10_2_blind_failure_followup.md](step10_2_blind_failure_followup.md) | 최초 블라인드 실패 이후 진단 |
| [step10_3_after_fix_diagnosis.md](step10_3_after_fix_diagnosis.md) | 첫 블라인드 실패 이후 원인 분리 |
| [step10_4_candidate_cut_fix.md](step10_4_candidate_cut_fix.md) | 후보 절단과 사실 근거 검증 수정 |
| [step10_5_fixed_validation.md](step10_5_fixed_validation.md) | 고정 설정 재평가와 새 독립 평가 |

## 2026-10-05 ~ 10-07 수정·검증

| 문서 | 내용 |
| --- | --- |
| [2026-10-05_fixed_followup.md](2026-10-05_fixed_followup.md) | 고정 설정 후속 검증 |
| [2026-10-05_feedback_capture_fix.md](2026-10-05_feedback_capture_fix.md) | 평가 로그 첨부 복구 |
| [2026-10-05_audio_spatial_fix.md](2026-10-05_audio_spatial_fix.md) | 오디오 background/foreground 검증 수정 |
| [2026-10-05_cover_dance_fix.md](2026-10-05_cover_dance_fix.md) | 커버댄스 분류 오류 수정과 재평가 |
| [2026-10-05_broadcast_fact_fix.md](2026-10-05_broadcast_fact_fix.md) | 방송명과 외부 사실 단서 경계 수정 |
| [2026-10-06_cover_contest_fix.md](2026-10-06_cover_contest_fix.md) | 커버 경연·영상과 앨범 표지의 경계 수정 |
| [2026-10-06_cover_meaning_fix.md](2026-10-06_cover_meaning_fix.md) | 분석 경계 보완 및 실제 검색 적용 검증 |
| [2026-10-06_event_detail_fix.md](2026-10-06_event_detail_fix.md) | 단서·근거 세부 조건 수정 및 최종 실행 |
| [2026-10-06_api_finish.md](2026-10-06_api_finish.md) | Context API 확인 마무리 |
| [2026-10-06_final_validation.md](2026-10-06_final_validation.md) | Context 검색 최종 검증 기록 |
| [2026-10-06_review_faq.md](2026-10-06_review_faq.md) | 검토 질문과 근거 |
| [2026-10-06_media_scope_fix.md](2026-10-06_media_scope_fix.md) · [json](2026-10-06_media_scope_local_checks.json) | 작품 단서 범위 수정 및 재검증 |
| [2026-10-06_media_specificity_fix.md](2026-10-06_media_specificity_fix.md) · [json](2026-10-06_media_specificity_local_checks.json) | 상세 단서가 일반 OST 검색으로 축약되는 회귀 수정 |
| [2026-10-06_review_cleanup.md](2026-10-06_review_cleanup.md) · [json](2026-10-06_review_cleanup_local_checks.json) | 코드 리뷰 보완과 새 분석 재검증 |
| [2026-10-07_api_candidate_proof_fix.md](2026-10-07_api_candidate_proof_fix.md) · [json](2026-10-07_api_candidate_local_checks.json) | Context API 최종 검증 보완 |
| [2026-10-07_lyric_scope_fix.md](2026-10-07_lyric_scope_fix.md) · [json](2026-10-07_lyric_scope_local_checks.json) | 가사 인용과 오디오·이미지 단서 구분 수정 |

## 날짜 미상

| 문서 | 내용 |
| --- | --- |
| [query_analysis_review_validation.md](query_analysis_review_validation.md) | Context 질의 분석 리뷰 검증 |

`2026-10-06_final_validation.md`·`2026-10-06_review_faq.md`가 가리키는 `context_validation_queries_20261006.csv`는
레포에 올라온 적이 없다.
