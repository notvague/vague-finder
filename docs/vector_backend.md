# 벡터 DB (Qdrant)

검색은 Qdrant를 쓴다. 기본은 **로컬 파일 모드**라 서버를 따로 띄울 필요가 없다.

```bash
QDRANT_PATH=artifacts/qdrant          # 생략 가능 (기본값)
# QDRANT_URL=http://localhost:6333    # 서버/클라우드를 쓸 때만. 있으면 QDRANT_PATH는 무시
# QDRANT_API_KEY=...
```

검색 코드는 `client.Index(name).query(...)` 결과에서 `matches`의 `id / score / metadata`만
읽는다. 이 호출 모양은 `src/vector_db/qdrant_backend.py`가 제공한다.

## 처음 받았을 때 (팀원용)

**임베딩 벡터와 곡 데이터는 레포에 없다.** 가사·앨범 커버·음원에서 뽑은 것이라 커밋하지
않는다. 아래 네 단계를 거치면 검색이 돈다.

**1. 패키지**

```bash
venv/bin/pip install -r requirements.txt
```

**2. 드라이브에서 받을 것** (`vague-finder/` 폴더, 2026-09-19 952곡 기준)

| 받는 것 | 놓을 위치 |
|---|---|
| `embeddings_20260919_952.zip` | `artifacts/embeddings/` 아래에 풀기 |
| `all_songs_20260919_952.jsonl` | `data/all_songs.jsonl`로 이름을 바꿔 두기 |

푼 모양은 이렇다. 모델 이름 폴더는 자동으로 찾으므로 이름이 달라도 된다.

```
artifacts/embeddings/text_dense/<모델>/<song_id>.npy
artifacts/embeddings/image/<모델>/<song_id>.npy
artifacts/embeddings/audio/<모델>/<song_id>.npy
```

**3. `.env` 설정** — `.env.example`을 복사해 채운다

```bash
cp .env.example .env     # GEMINI_API_KEY 확인
```

**4. 적재**

```bash
venv/bin/python -m src.vector_db.cli.qdrant_load --recreate
```

디렉터리가 하나라도 없으면 명령이 **실패로 끝난다**(예전에는 조용히 건너뛰고 성공으로
끝나 빈 컬렉션이 만들어졌다). BM25 sparse는 파일이 아니라 이 단계에서 만들어진다.

이제 `uvicorn src.backend.main:app --reload`로 서버를 띄우거나 측정을 돌릴 수 있다.
가사 정확 일치 검색만 MongoDB가 따로 필요하고, 없으면 그 경로만 비고 나머지는 동작한다.

만들어지는 것 — `artifacts/qdrant/` (git 추적 안 함)

| 컬렉션 | 내용 |
|---|---|
| `vaguefinder-text-1024-koe5__dev` | KoE5 dense 1024 + BM25 sparse |
| `vaguefinder-image-768__dev` | SigLIP2 768 |
| `vaguefinder-audio-512__dev` | CLAP 512 |

컬렉션 이름은 `<이름>__<네임스페이스>`다. `QDRANT_TEXT_COLLECTION` · `QDRANT_IMAGE_COLLECTION` ·
`QDRANT_AUDIO_COLLECTION` · `QDRANT_NAMESPACE`로 바꿀 수 있지만, 바꾸면 다시 적재해야 한다.

나무위키 배경지식은 기존 곡 단위 컬렉션에 섞지 않는다. 사실 단위 Dense와 곡 단위
Sparse를 별도 세대 컬렉션에 안전하게 게시하는 절차는
[`namuwiki_context_vector_db.md`](namuwiki_context_vector_db.md)를 따른다.

## 점수 계산

| 경로 | 방식 |
|---|---|
| 텍스트 | alpha로 스케일한 dense·sparse 내적의 합. Qdrant에는 이 합산이 없어 dense·sparse를 따로 조회해 **코드에서 더한다**. 두 조회 모두 컬렉션 전수를 훑어(`QDRANT_HYBRID_SCAN_LIMIT`, 기본 5000) 잘림이 없다 |
| 이미지·오디오 | cosine (저장 벡터가 전부 L2 정규화라 dot과 동일) |
| 메타데이터 필터 | `$eq` `$and` `$gte`(문장형 제목) `$or`(의성어 제목)를 Qdrant 필터로 변환(범위는 Range, OR는 should). **모르는 연산자는 예외를 던진다** |
| 검색 방식 | 로컬은 전수 비교, 서버 모드는 `exact=True`. 근사 검색은 측정할 때마다 순위를 흔든다 |

## 주의

- `artifacts/qdrant/`는 **적재 산출물**이라 커밋하지 않는다. 벡터가 바뀌면 다시 적재한다.
- 로컬 모드는 한 프로세스만 디렉터리를 연다. 서버와 측정 스크립트를 동시에 돌리려면
  각각 다른 `QDRANT_PATH`를 쓰거나 서버 모드(`QDRANT_URL`)를 쓴다.
- BM25 sparse는 `artifacts/bm25_params.json`과 짝이다. 한쪽만 바꾸면 점수가 어긋난다.
- BM25 인코딩은 `pinecone-text` 라이브러리를 쓴다. 이름만 같고 Pinecone 서비스와는 무관하다.

## 이력 — Pinecone에서 옮긴 이유 (2026-09)

처음에는 관리형 Pinecone을 썼다. 2026-09-16 Qdrant로 옮겼고, 2026-09-29 Pinecone 코드와
설정을 저장소에서 모두 뺐다.

- Pinecone Starter는 **월 전송 1GB**다. 2026-09-16 측정 도중 소진돼 모든 읽기가 429로
  막혔고, 검색 결과가 0개가 됐다. 검색 1회당 약 0.3MB라 1GB ≈ 3,000회뿐이다.
- 이 프로젝트는 측정이 중심이다(하네스 한 번에 수백 회 검색). 질의 수를 줄이는 대신
  한도가 없는 쪽으로 옮겼다.
- 곡이 수천 곡 규모라 로컬 전수 검색이 빠르고 정확하다.
- 검색 코드가 쓰던 `Index(name).query(...)` 호출 모양은 그대로 두어, 옮길 때 검색·재질문·
  리랭킹 코드를 바꾸지 않았다.
