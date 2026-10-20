# 산출물 명세 — `description_residual_v1`(앨범 소개 원문 중 색인 문서에 없는 낱말)

생성 2026-10-20 · 코드 `src/retrieval/description_aux.py`(v76 워크트리, 커밋 전) · 운영 자료는 읽기만 했다.

## 1. 규칙(`desc_residual_v1`)

곡마다:

1. 소개 원문 `D = metadata.album_description.strip()`. 비었으면 문서 없음.
2. 색인 문서 `P = build_sparse_passage(song)`(텍스트 색인 BM25 문서와 같은 함수).
3. `tok(x) = normalize_for_bm25(x).split()`(색인과 같은 Kiwi 정규화).
4. 잔여 `R = [t for t in tok(D) if t not in set(tok(P))]`(중복 횟수 유지). 비었으면 문서 없음.
5. 묶음 id = sha256(D)[:16] — 같은 소개를 공유하는 곡(같은 앨범)은 같은 묶음.

BM25 통계는 잔여 문서 전체로 센다(문서 수 · 문서 빈도 · 평균 길이). 낱말 사전·질의·정답은 쓰지 않는다.

## 2. 입력과 코드 지문

| 항목 | 값 |
|---|---|
| 코퍼스 | `data/all_songs.jsonl` 5,143곡, sha256 `63891fd3fa6155f73e2f7496ec5fdff414937c67ea2373c5e0ff18470568d9de` |
| 색인 일치 확인 | v76 Qdrant(텍스트 컬렉션)의 sparse 벡터 5,143/5,143 = 이 jsonl의 `build_sparse_passage` + `artifacts/bm25_params.json`(sha1 앞 16자 `4e66576fcb1de1f5`) 인코딩 |
| `korean_bm25_tokenizer.py` sha256 | `af0c6afc4224907aaf7ac8a59be26083c6bffa99c6908733db0e50b31efbfcd3` |
| `passage_builder.py` sha256 | `cb21374ed7bf7fc62185694e3e0f0435659ce94640701441d91c55dc4acd7c0f` |
| kiwipiepy | 0.23.2 |

## 3. 산출물

| 항목 | 값 |
|---|---|
| 파일 | `artifacts/description_aux/description_residual_v1.json.gz`(Git 제외 경로) + `description_residual_v1.manifest.json` |
| 크기 | 2.1MB(gz) |
| sha256 | `11d72680dfe8285c6d8c2559cb3389d1c7014bef78a24ea766656591826eb0ad` |
| 결정성 | 같은 입력으로 두 번 만들어 같은 sha(gzip `mtime=0`, 키 정렬) |
| 소개 있는 곡 · 문서 | 4,873 · 4,856(소개는 있으나 잔여가 빈 곡 17) |
| 소개 묶음 | 3,893 |
| 잔여 토큰 · 어휘 | 899,211 · 43,293 · 역색인 항목 638,410 |
| 적재 | 0.23초 · 프로세스 RSS +81MB(gz 해제·JSON 해석 잔여 포함, 같은 노트북) |

## 4. 다시 만들기(워크트리 루트)

```bash
venv/bin/python -m src.retrieval.description_aux build --songs data/all_songs.jsonl \
  --out artifacts/description_aux/description_residual_v1.json.gz     # 이미 있으면 덮어쓰지 않는다(종료 코드 2)
```

코퍼스·`passage_builder`·토크나이저·Kiwi 판이 바뀌면 잔여가 달라진다 — 그때는 새로 만들고 sha를 다시 고정한다(색인을 다시 만들 때 함께). 서비스에서 켜면(`DESCRIPTION_AUX_WEIGHT>0`) manifest의 sha와 파일이 다를 때 기동 오류다.
