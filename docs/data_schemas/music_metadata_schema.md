# Music Metadata Data Schema

이 문서는 **Crawler(이연우) -> LLM Processor(이연우) -> Embedding/Search(최정현)** 간의 데이터 전달을 위한 통합 JSON 포맷을 정의합니다.

모든 모듈은 아래의 스키마로 통일합니다

## JSON Structure

````json
{
  "id": "BzYnNdJhZQw",
  "metadata": {
    // 1. Basic Info (Source: iTunes - Korean Store)
    "trackName": "밤편지",
    "trackCensoredName": "밤편지",
    "artistName": "아이유",
    "collectionName": "밤편지 - Single",
    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/.../600x600bb.jpg",
    "trackTimeMillis": 253293,
    "primaryGenreName": "K-Pop",
    "releaseDate": "2017-03-24T07:00:00Z",
    "trackExplicitness": "notExplicit",

    // 2. Lyrics Data (Source: Genius)
    "lyrics": "[Verse 1]\n(가사 전문 — 예시에서는 생략)",

    // 3. Reaction Data (Source: YouTube)
    "video_url": "https://www.youtube.com/watch?v=BzYnNdJhZQw",
    "comments": [
      "(댓글 1 — 예시에서는 생략)",
      "(댓글 2 — 예시에서는 생략)"
    ],
    "like_count": 857106,
    "view_count": 115830679,

    // 4. AI Analysis (Source: Gemini 1.5 Flash -> Refined)
    // Categories: Time, Weather, Activity, Vibe, Emotion
    "mood_tags": [
      "밤",
      "잠들기 전",
      "잔잔한",
      "어쿠스틱",
      "그리움",
      "위로",
      "몽환적인"
    ],
    "scene_summary": "깊은 밤, 잠들기 전 사랑하는 이를 그리워하며 듣는 잔잔하고 몽환적인 위로 발라드",
    "lyrics_highlight": "(핵심 가사 한두 줄 — 예시에서는 생략)"
  }
}

## Field Descriptions

| Field | Source | Description |
| :--- | :--- | :--- |
| `id` | YouTube | YouTube Video ID (File Name Key) |
| `trackName` | iTunes | 곡 제목 (Korean) |
| `artistName` | iTunes | 아티스트 이름 (Korean) |
| `collectionName` | iTunes | 앨범명 |
| `artworkUrl100` | iTunes | 앨범 커버 URL (High Res) |
| `trackTimeMillis` | iTunes | 곡 길이 (밀리초) |
| `primaryGenreName` | iTunes | 대표 장르 |
| `releaseDate` | iTunes | 발매일 (ISO Format) |
| `trackExplicitness` | iTunes | 유해성 여부 (explicit/cleaned/not) |
| `lyrics` | Genius | 가사 전체 |
| `video_url` | YouTube | 유튜브 영상 링크 |
| `comments` | YouTube | 필터링된 댓글 리스트 (Top 30) |
| `like_count` | YouTube | 좋아요 수 |
| `view_count` | YouTube | 조회수 |
| `mood_tags` | **LLM** | 감성 태그 (시간/날씨/행동/분위기/감정 포함) |
| `scene_summary` | **LLM** | 태그를 통합한 상황 묘사 한 문장 |
| `lyrics_highlight` | **LLM** | 핵심 가사 한 두 줄 (하이라이트) |
| `external_context.namuwiki.match_status` | Namuwiki | 외부 문서 매칭 결과 (`matched`, `ambiguous`, `not_found`, `access_error`) |
| `external_context.namuwiki.match_type` | Namuwiki | 매칭된 문서 유형 (`song_page`, `album_section`, `artist_section`, `other_section`) |
| `external_context.namuwiki.page_title` | Namuwiki | 매칭된 나무위키 문서 제목 |
| `external_context.namuwiki.source_url` | Namuwiki | 원본 문서 URL |
| `external_context.namuwiki.context_usable` | Pipeline | 검색 맥락으로 활용 가능한지 여부 (`yes`, `no`) |
| `external_context.namuwiki.context_tags` | **LLM** | 원문에 명시적으로 등장하는 외부 맥락 고유명사 및 핵심 표현 |
| `external_context.namuwiki.fact_summary` | **LLM** | 원문에 명시된 곡 관련 사실을 요약한 문장 |

## External Context (Namuwiki)

외부 문서에서 수집한 곡 관련 맥락 정보를 `external_context.namuwiki`에 저장합니다.

```json
"external_context": {
  "namuwiki": {
    "match_status": "matched",
    "match_type": "song_page",
    "page_title": "거짓말(빅뱅)",
    "source_url": "https://namu.wiki/w/...",
    "context_usable": "yes",
    "context_tags": [
      "BIGBANG",
      "Always",
      "거짓말",
      "용감한 형제"
    ],
    "fact_summary": "2007년 8월 16일 발매된 빅뱅의 첫 번째 미니 앨범 Always의 타이틀곡입니다."
  }
}
`external_context.namuwiki.context_tags`는 외부 문서에 명시된 고유명사·사실 맥락이며, 기존 감성/상황 기반 태그와 구분합니다.

## LLM Processing Rules (For Data Engineer)

LLM 프롬프팅 시 아래 규칙을 준수하여 태그와 `scene_summary`를 생성해야 합니다.

context_usable이 yes인 데이터만 검색 passage에 사용합니다.
context_tags와 fact_summary는 원문에 명시된 정보만 사용합니다.
원문에 없는 상황, 감정, 장소, 활용 맥락 등을 추론하여 생성하지 않습니다.
context_excerpt와 같은 원문 본문은 최종 meta.json 및 검색 passage에 저장하지 않습니다.
context_tags와 fact_summary는 sparse passage에 사용합니다.
context_tags와 fact_summary는 refine_data.py 단계에서 생성되므로, 그 앞 단계인 나무위키 매칭 결과표에서는 비어 있는 것이 정상입니다.


### 1. Tagging Rules (변별력 확보)

아래 카테고리의 단어가 `mood_tags`, `context_tags`에 최대한 많이 포함되어야 합니다.

- **시간**: 낮 / 밤 / 새벽 / 저녁
- **날씨/환경**: 비 / 맑음 / 네온 / 실내 / 도시 / 바다
- **행동**: 혼자 걷기 / 드라이브 / 이어폰 끼기 / 창가에 앉기
- **감정/무드**: 우울 / 청량 / 몽환 / 잔잔
- **음악 성격**: 인디 곡 / 드라이브 음악 / 신스 사운드

### 2. Scene Summary Generation Logic

태그와 하이라이트 가사를 바탕으로 **"사용자가 기억으로 말할 법한 한 문장"**을 생성합니다.

- **규칙**: `{시간} + {날씨} + {행동} + {감정/성격} 곡` 형태로 일관성 있게 생성
- **예시**:
  - 입력: `mood_tags`=[우울, 인디], `context_tags`=[비, 새벽, 창가]
  - 출력(`scene_summary`): "비 오는 새벽에 창가에 앉아 듣기 좋은 우울한 인디 곡"

## Embedding Strategy (For AI Researcher)

벡터(`values`) 생성 시 입력 텍스트(Passage) 구성:

> **`scene_summary`를 맨 앞에 배치**하여 검색 정확도를 극대화합니다.

```text
{scene_summary} Title: {title}, Artist: {artist}, Highlight: {lyrics_highlight}
````
