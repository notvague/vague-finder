# 감정 필드(major_emotion) 색인 변경 A/B — 제외 결정의 근거

**결론: 현재 76개 평가 질의에서는 검색 개선을 확인하지 못했다.** 그래서 이 변경을 색인에서
뺐다. 크롤러·데이터 품질 개선(한국어 감정 프롬프트, 태그 근거 정규화, 멜론 댓글 수정)만 남긴다.

일반적으로 효과가 없다는 것까지 입증한 건 아니다. 76개 중 감정 표현이 든 질의는 8개뿐이라
검출력이 낮다. 다만 리랭킹 후 순위가 오른 질의는 없고(q310은 1차 검색에서만 3→2) 두 건은
내려갔으니, 이번 PR에서 제외할 근거로는 충분하다.

'감정 표현이 든 질의'는 질의 원문이나 분석의 `korean_tags`에 감정 어간 12개
(`reproduce/build_tables.py`의 `EMOTION_STEMS`) 중 하나가 든 질의다. 이 어간 목록에는 통제
어휘 14종 중 사랑·자신감·열정·분노·몽환·희망이 없어서, 예를 들어 q316(태그 `희망`)과
q214·q219(`몽환`)는 세지 않았다.

## 무엇을 비교했나

`community_feedback.major_emotion`은 검색에서 두 군데에 쓰인다.

- **색인**: sparse passage(BM25)의 감정 줄. dense passage(KoE5)에는 없다.
- **색인 밖**: 벡터 DB 메타데이터를 거쳐 기본 리랭커(Cross-Encoder, `reranker.py`의 `대표 감정`
  줄) 문서와 검색 API 응답(`major_emotion`)에 그대로 들어간다.

수집분 961곡의 값이 전부 영어(45종)였고, 이를 한국어 통제 어휘로 옮겨 **색인하는** 변경을 검토했다.
이 A/B는 passage만 바꿨다. 두 조건 모두 리랭커에는 저장된 영어 값이 들어갔다.

| 조건 | sparse passage의 감정 줄 |
|---|---|
| **A** | 저장된 영어 값 그대로 (`Sadness`) |
| **B** | 태그 근거로 정한 한국어 값 + 형용사형 확장 (`슬픔 슬픈`) |

형용사형은 Kiwi 표제어 문제 때문에 붙였다. 질의 `슬픈 노래`는 `슬프다`로 분석되는데
색인 값 `슬픔`은 `슬픔`으로 남아 서로 맞지 않는다. B의 코드는 `reproduce/arm_B.patch`에 있다.

패치는 측정 당시 코드를 그대로 보존한 것이라, 그 안의 주석과 docstring은 측정 전에 쓴 설명이다.
다음 세 설명은 이 문서의 결과와 맞지 않으며, 결론은 이 문서를 따른다.

- 감정 값이 sparse passage에만 쓰인다는 설명 — 리랭커 문서와 API 응답에도 들어간다.
- 영어 값을 비워도 영어일 때와 같다는 설명 — 줄이 빠지면 문서 길이가 달라진다. q109의 순위
  변화도 그 영향으로 보인다.
- 다시 크롤링하지 않아도 색인이 바로잡힌다는 설명 — 이 A/B에서는 개선을 확인하지 못했다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 측정 일시 | 2026-09-17 ~ 18 |
| 코드 기준 | `feat/qdrant-backend` @ `6053bdd` (main에는 Qdrant 어댑터가 없어 이 브랜치에서 측정) |
| 두 조건의 코드 차이 | `passage_builder.py`의 감정 줄 + `emotion_vocab.py`뿐 (`arm_B.patch`) |
| 코퍼스 | 905곡 (임베딩이 있는 곡). 감정 줄이 바뀐 곡 905, 그중 B에서 근거가 없어 줄을 비운 곡 22 (`passage_emotion_changes.csv`) |
| 벡터 DB | Qdrant 로컬, 조건마다 따로 적재 |
| BM25 파라미터 | **조건마다 자기 passage로 다시 맞춤** (`fit_params.py`). 질의 IDF가 파라미터의 `doc_freq`에서 나오므로 적재와 검색이 같은 파일을 쓰게 했다 |
| 질의 분석 | Gemini 분석을 **한 번만** 받아 두 조건에 같게 주입 (`reproduce/analyses.json`, 76/76 동일 확인) |
| 평가 질의 | **`feat/qdrant-backend`의** `experiments/reranking/eval_queries_v05.csv` (v0.5.1) 76건, dev 53 + test 23 |
| top_k / candidate_k | 10 / 30 |
| 리랭커 | Cross-Encoder `dragonkue/bge-reranker-v2-m3-ko` (6053bdd에는 리랭커 전환 설정이 없다). `RERANKER_SPREAD_REF=0.02`, `RERANKER_WEIGHT=0.90`, `RERANKER_LOW_CONF_TOP_N=5`. SPREAD_REF는 코드 기본값이 0이라, 설정하지 않으면 결합 가중치와 재정렬 방식이 달라져 리랭킹 후 수치가 재현되지 않는다. 상세 결과에서 `rerank_spread / rerank_confidence`가 0.02로 확인된다 |
| 가사 정확일치 | MongoDB `vaguefinder.songs` (`MONGO_URI`·`MONGO_DB_NAME`). 측정 뒤 확인한 상태는 905곡, 모두 가사 있음. 리랭킹 후 상위 결과의 가사 일치 유형은 두 조건 모두 exact 22·fuzzy 2·phonetic 1건이다 |
| dense·이미지·오디오 벡터 | 두 조건 동일 |

**평가 질의 버전 주의.** main의 같은 파일(v0.5.0)은 c607·c704·c707 세 질의의 문구가 다르다.
`analyses.json`은 질의 문구를 키로 쓰므로 main의 CSV로는 73/76건만 맞는다. 재현은 반드시
`feat/qdrant-backend` @ `6053bdd`에서 한다.

**토큰화 주의.** 두 조건의 텍스트 차이는 감정 줄뿐이지만, Kiwi가 passage 전체를 한 번에
분석하므로 감정 줄 자체나 인접 줄의 토큰이 문맥에 따라 달라질 수 있다. 감정 줄을 따로 분석한
토큰과 비교하면 905곡 중 120곡에서 그런 차이가 있었다.

A도 파라미터를 다시 맞췄으므로 **A의 수치는 기존 측정(공식 기준선 `results_v05`, `results_v07` 등
— Pinecone namespace `dev`, 질의 세트도 다름)과 같지 않다.** 이 표는 A와 B를 서로 비교하는
용도로만 쓴다.

두 조건 모두 76건을 끝까지 돌았다(상세 결과 76행·고유 질의 76건·분석 누락 0건). 실행 당시
`run_ab.sh`는 종료코드를 잘못 읽어 늘 `exit=0`을 찍었다. 지금 스크립트는 고쳐 두었다.

## 결과 (리랭킹 후)

| 지표 | A (영어) | B (한국어+확장) | 차이 |
|---|---|---|---|
| Hit@1 | 0.461 | 0.447 | −1건 |
| Hit@5 | 0.671 | 0.658 | −1건 |
| **Hit@10** | **0.776** | **0.776** | 0 |
| MRR@10 | 0.564 | 0.557 | −0.007 |
| nDCG@10 | 0.615 | 0.609 | −0.005 |
| 후보 Recall@30 | 0.947 | 0.947 | 0 |

### 질의별 순위 (`rank_changes.csv`)

76건 중 순위가 달라진 질의는 4건이다. **최종 지표를 바꾼 건 q109와 c706 두 건이다.**
q310은 1차 검색에서만 올라갔고 리랭킹 후 순위는 같다.

| 질의 | split | 감정 표현 | 후보@30 | 1차 검색 | 리랭킹 후 |
|---|---|---|---|---|---|
| q109 | test | 없음 | 1 → 2 | 1 → 2 | **1 → 2** |
| c706 | dev | 있음 | 5 → 6 | 5 → 6 | **5 → 6** |
| q310 | dev | 있음 | 3 → 2 | 3 → 2 | 2 → 2 |
| q316 | dev | 없음 | 23 → 24 | 10위 밖 | 10위 밖 |

(감정 표현 열은 위 정의를 따른다. q316은 태그에 `희망`이 있지만 어간 목록에 없어 '없음'이다.)

## 관측된 원인

**감정 질의 토큰이 A passage에 얼마나 있었는지, 어디서 왔는지는 토큰마다 다르다.**
`설레다`·`추억`·`위로`는 A에서도 흔했고(269~821곡), `슬프다`·`그립다`·`기쁘다`는 그보다
적었다(905곡 중 170·107·14곡). (`emotion_token_sources.csv`, `query_token_idf.csv`)

| 질의 토큰 | A에서 있던 곡 | 태그에 있음 | 가사에 있음 | 둘 다 아님(요약 등) |
|---|---|---|---|---|
| 설레다 | 269 | 249 | 39 | 5 |
| 추억 | 821 | 772 | 138 | 37 |
| 위로 | 508 | 290 | 46 | 194 |
| 슬프다 | 170 | 2 | 124 | 45 |
| 그립다 | 107 | 2 | 94 | 12 |
| 기쁘다 | 14 | 0 | 14 | 0 |

(한 곡이 태그와 가사 양쪽에 있을 수 있어 칸의 합이 곡 수보다 클 수 있다.)

`설레다`·`추억`·`위로`는 원래 한국어인 태그에 이미 흔해서, 감정 줄을 한국어로 옮겨도 더해지는
토큰이 적다. 반면 형용사 표제어인 `슬프다`·`그립다`·`기쁘다`는 주로 **가사**에서 왔다.

**형용사형 확장이 이 토큰들의 IDF를 낮췄다.**

| 질의 토큰 | 문서빈도 A → B | IDF A → B |
|---|---|---|
| 슬프다 | 170 → 428 (47%) | 1.67 → 0.75 |
| 기쁘다 | 14 → 137 | 4.13 → 1.89 |
| 그립다 | 107 → 124 | 2.13 → 1.98 |

IDF가 줄어든 것 자체는 결함이 아니다(문서빈도가 늘면 IDF는 줄어든다). 다만 A에서 `슬프다`는
가사 등에 그 표현이 실제로 들어 있는 170곡에만 있었는데, B에서 Sadness 364곡 모두에 `슬픈`이
붙으면서(그중 106곡은 이미 있었다) 258곡이 새로 더해져 428곡이 됐다. 이 질의어로 곡을 가려내는
힘이 줄었다.

IDF 표는 Pinecone `BM25Encoder`와 같은 식으로 계산했다. 코퍼스에 없는 토큰은 인코더처럼
df=1로 계산하며(`화난` 단독), 문서빈도 열에는 실제 값 0을 둔다.

순위가 바뀐 두 질의를 곡 단위로 보면 다음과 같다(조건별 상세 결과의 상위 곡 목록과,
조건별 passage를 토큰화해 센 값).

- **c706** (정답 「눈물」, Sadness) — 가사 단서가 없어 sparse 질의는 `korean_tags`
  (`여성보컬 랩 발라드 슬픈`)이고, 토큰은 `여성 보컬 랩 발라드 슬프다`다. B에서 정답보다 새로
  위로 올라온 「나만 안되는 연애」와 「미워도 사랑하니까」(둘 다 Sadness)는 A passage에 `슬프다`가
  없다가 감정 줄의 `슬픈` 확장으로 이 질의 토큰을 처음 얻었다(tf 0 → 1). 정답은 A에서도 `슬프다`가
  있었다(tf 2 → 3). `슬프다`의 IDF가 1.67 → 0.75로 줄면서 정답이 이 토큰으로 앞서던 차이가
  좁혀진 것으로 보인다. 세 곡 모두 passage가 한 토큰씩만 늘어 길이 보정의 영향은 작을 것으로 본다.
  1~3위인 「두사랑」「있다 없으니까」「착해 빠졌어」는 A에서도 정답보다 위에 있었고, A에서 4위로
  정답보다 위였던 「거북이」는 B에서 상위 10위 밖으로 밀려났다. 두 곡이 올라오고 한 곡이 빠져
  5위에서 6위가 됐다.
- **q109** (정답 「Hello」, Sadness) — 질의에 감정 표현이 없어 겹치는 감정 토큰이 없다. B에서
  「Shock」(Intensity)은 근거가 없어 감정 줄이 빠졌고 「Hello」에는 `슬픔 슬픈`이 붙었다. 질의와
  겹치지 않는 토큰으로 문서가 길어지고 짧아지면서 BM25 길이 보정이 두 곡의 순서를 바꾼 것으로 보인다.

## 이번 PR에서의 처리

- **제외**: `passage_builder`의 감정 색인 변경, 형용사형 확장 함수, 전용 테스트
- **유지**: 한국어 감정 프롬프트, 태그 근거 정규화(크롤러가 LLM 응답을 받을 때 적용), 멜론 댓글
  수정 3건, 미리보기가 기본인 댓글 엔티티 backfill
- **함께 고침**: 정규화가 근거 없는 값을 비우면 임베딩 전 메타 검증이 빈 문자열로 곡 전체를
  탈락시키고 있었다. `major_emotion`만 빈 문자열을 허용했다. 검증기의 키 존재·문자열 자료형·실패
  문구 검사는 그대로 둔다. 다만 크롤러 성공 경로는 이 필드의 누락·null·실패 문구·목록 밖 값도
  빈 문자열로 저장하므로, 크롤러 출력에서 이 필드로 걸리는 것은 재시도를 모두 실패한 대체값
  `분석실패`뿐이다.

**기존 배포 색인은 이번 PR에서 다시 적재할 필요가 없다.** 다만 앞으로 한국어 감정값으로 수집한
곡은 영어 값이던 기존 곡과 값이 달라진다(예: `Sadness` 대신 `슬픔`). 지금 passage 로직으로
색인하면 감정 줄의 토큰이 달라지고, 리랭커 문서의 `대표 감정` 줄과 API 응답에도 영어·한국어 값이
섞인다. 리랭커 쪽 변화는 이 A/B에서 재지 않았으므로, 새로 수집한 곡이 코퍼스에 들어갈 때 BM25
토큰과 함께 리랭커 결과도 확인해야 한다.

## 별도 과제 — BM25 파라미터 불일치

측정 중에 커밋된 파라미터와 `feat/qdrant-backend` @ `6053bdd`의 passage(A 조건) 사이의 불일치를
발견했다. **이번 PR에서는 파라미터를 바꾸지 않는다.**

| 파라미터 | n_docs | avgdl | 고유 토큰 |
|---|---|---|---|
| 커밋된 `artifacts/bm25_params.json` (main·`feat/qdrant-backend` 동일) | 905 | 314.74 | 14,330 |
| A 조건(`6053bdd`) passage로 다시 맞춘 값 | 905 | 378.57 | 16,547 |
| (참고) main @ `eda0697` passage로 다시 맞춘 값 | 905 | 313.95 | 14,275 |

앨범 정보 복원(`5d18183`, `feat/qdrant-backend`에만 있고 main에는 없음. sparse passage에 `album`·
`album_summary`·`sentiment_summary`를 되돌린 변경)으로 passage가 길어졌는데 파라미터가 갱신되지
않았다는 **가설**이 있다. main의 passage로 같은 905곡에 다시 맞추면 커밋된 값과 가깝지만(doc_freq
13,554개 일치) 차이(avgdl 0.79, 고유 토큰 55개)가 남으므로 원인을 확정하지는 않는다. 확정하려면
다음을 대조해야 한다.

- 코퍼스 구성 — n_docs는 같지만 포함된 곡 목록이 같은지
- 토크나이저 — Kiwi 버전·사용자 사전(`_DOMAIN_USER_WORDS`)이 파라미터 생성 시점과 같은지
- passage 버전 — 파라미터를 만든 시점의 `passage_builder`가 무엇이었는지

파라미터 파일만 바꾸면 문서 벡터와 질의 IDF가 어긋난다. **문서 sparse 벡터와 검색용 파라미터를
함께 갱신하는 실험**으로 따로 진행해서 변경 전후를 비교한다.

## 파일

| 파일 | 내용 | 커밋 |
|---|---|---|
| `A_english/`·`B_korean_expanded/search_eval_all_summary.csv` | 조건별 지표 요약 | O |
| `A_english/`·`B_korean_expanded/search_eval_all_detail.csv` | 질의별 상세(분석·상위 곡 목록) | X (`.gitignore`, 용량) |
| `rank_changes.csv` | 76개 질의의 단계별 순위 A/B | O |
| `query_token_idf.csv` | 감정 질의 토큰의 문서빈도·IDF A/B | O |
| `emotion_token_sources.csv` | 감정 질의 토큰의 출처(태그·가사·그 밖) | O |
| `passage_emotion_changes.csv` | 감정 줄 A 값 → B 값, 곡 수 | O |
| `reproduce/analyses.json` | 두 조건에 주입한 Gemini 질의 분석 | O |
| `reproduce/*.py`, `run_ab.sh`, `arm_B.patch` | 재현 스크립트와 B 조건 코드 | O |

`analyses.json`을 커밋하는 이유: Gemini 분석은 실행마다 달라져서, 같은 분석 없이는 A/B를 다시
돌려도 같은 조건이 되지 않는다. 내용은 커밋된 평가 질의에서만 나온다(`lyric_clues`가 있는 8개
질의의 단서 12개가 모두 질의 원문 안의 문구다).

passage 원문(`passages_*.json`)은 **가사 전문이 들어 있어 커밋하지 않는다.** 거기서 만든 파라미터
파일(`params_A.json`·`params_B.json`)에는 가사가 없고 해시된 토큰 id와 문서빈도만 들어 있다(커밋된
`artifacts/bm25_params.json`과 같은 형식). passage에서 `fit_params.py`로 다시 만들 수 있어 커밋하지
않는다.

## 재현

`feat/qdrant-backend` @ `6053bdd`에서 한다. 아래 `$REPO`는 곡 데이터(`data/`)·임베딩
(`artifacts/embeddings/` 등)·`.env`가 있는 작업 사본, `$W`는 빈 작업폴더, `$RP`는 이 폴더의
`reproduce/` 경로다. **세 경로는 모두 절대경로로 지정한다** — 1단계의 `git -C`와 3·6단계의 `cd`
뒤에서 그대로 다시 쓰기 때문이다. `python`은 프로젝트 venv의 파이썬이다.

`$REPO/.env`에는 가사 정확일치에 쓰는 `MONGO_URI`·`MONGO_DB_NAME`이 있어야 한다(MongoDB `songs`
컬렉션에 곡과 가사가 적재된 상태). `run_ab.sh`는 `.env`가 없으면 멈추고, 리랭커 설정은 `.env`와
무관하게 측정 당시 값으로 고정한다.

```bash
# 1) worktree 두 개. B에만 arm_B.patch를 적용한다.
git -C "$REPO" worktree add --detach "$W/wA" 6053bdd
git -C "$REPO" worktree add --detach "$W/wB" 6053bdd
git -C "$W/wB" apply "$RP/arm_B.patch"

# 2) 곡 데이터와 임베딩을 연결한다. artifacts/는 bm25_params.json이 추적 파일이라 폴더째 링크하지 않는다.
for w in wA wB; do
  ln -s "$REPO/data" "$W/$w/data"
  for item in embeddings map audio_mean.npy audio_mean_manifest.json; do
    ln -s "$REPO/artifacts/$item" "$W/$w/artifacts/$item"
  done
done

# 3) 조건마다 passage → BM25 파라미터 → Qdrant 적재
for arm in A B; do
  (cd "$W/w$arm" && PYTHONPATH=. python "$RP/dump_passages.py" "$W/passages_w$arm.json")
  (cd "$W/w$arm" && PYTHONPATH=. python "$RP/fit_params.py" "$W/passages_w$arm.json" "$W/params_$arm.json")
  (cd "$W/w$arm" && VECTOR_BACKEND=qdrant PYTHONPATH=. python -m src.vector_db.cli.qdrant_load \
     --recreate --qdrant-path "$W/qdrant_$arm" --bm25-params "$W/params_$arm.json")
done

# 4) 질의 분석. 커밋된 analyses.json을 쓰면 원래 측정과 같은 조건이 된다.
cp "$RP/analyses.json" "$W/analyses.json"
#    새로 받으려면 키를 읽은 뒤 실행한다. 키가 없거나 분석이 fallback이면 prefetch.py가 멈춘다.
#    (cd "$W/wA" && set -a && . "$REPO/.env" && set +a && PYTHONPATH=. python "$RP/prefetch.py" "$W/analyses.json")

# 5) 평가 (A → B). 저장된 분석을 주입하므로 Gemini를 부르지 않는다.
"$RP/run_ab.sh" "$W" "$REPO"

# 6) 표 다시 만들기
(cd "$W/wB" && PYTHONPATH=. python "$RP/build_tables.py" "$W" "$W/tables")
(cd "$W/wA" && PYTHONPATH=. python "$RP/token_sources.py" "$W/passages_wA.json" "$W/tables/emotion_token_sources.csv")
```

`build_tables.py`는 `arm_B.patch`의 `emotion_index_terms`가 필요해서 wB에서 돌린다. 스크립트는
CSV를 UTF-8 BOM·LF로 쓰고, 커밋된 파일도 같은 형식이다. 이 절차로 `rank_changes.csv`·
`query_token_idf.csv`·`passage_emotion_changes.csv`·`emotion_token_sources.csv`가 커밋된 파일과
바이트 단위로 같게 나오는 것을 확인했다.
