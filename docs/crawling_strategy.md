# Data Collection Strategy & Guide

이 문서는 **이연우 (Data Engineer)** 님을 위한 편향되지 않는 데이터 수집 전략 및 가이드라인입니다.

## 1. Overall Goal

- **Target Volume**: 총 **5,000곡**
- **Objective**: 다양한 상황과 감정에서 고르게 검색되는 "Balanced Dataset" 구축

## 2. Theme Distribution

데이터 쏠림을 방지하기 위해 아래 비율에 맞춰 수집을 진행합니다.

| Priority | Theme                      | Ratio   | Count       | Keywords (검색어 예시)                                                               |
| :------: | -------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------ |
|  **1**   | **비 / 새벽 / 감성**       | **30%** | **1,500곡** | `비 오는 날 플리`, `새벽 감성 노래`, `잔잔한 인디 음악`, `잠들기 전 듣기 좋은 곡`    |
|  **2**   | **드라이브 / 신남 / 청량** | **30%** | **1,500곡** | `드라이브 플리`, `여행 갈 때 듣는 노래`, `청량한 아이돌 노래`, `노동요`, `기분 전환` |
|  **3**   | **이별 / 우울 / 슬픔**     | **20%** | **1,000곡** | `이별 노래 모음`, `눈물 나는 발라드`, `우울할 때 듣는 노래`, `헤어진 연인을 위한`    |
|  **4**   | **OST / 노래방 / 명곡**    | **20%** | **1,000곡** | `드라마 OST 명곡`, `노래방 인기차트`, `2000년대 발라드`, `싸이월드 BGM`              |

## 3. Data Collection Pipeline (New Strategy)

우리는 데이터의 **정확성(Accuracy)**과 **풍부함(Richness)**을 모두 잡기 위해 4단계 하이브리드 파이프라인을 사용합니다.

### Step 1. Metadata (From iTunes API) 🍎

- **Role**: "기준 데이터(Ground Truth)" 확보
- **Action**: 검색어를 iTunes API에 질의하여 **공식 메타데이터**를 가져옵니다.
- **Acquired Data**:
  - `title` (Clean): "밤편지" (not "[MV] 아이유...")
  - `artist` (Clean): "IU"
  - `album_cover`: 고화질 앨범 커버 (100x100 -> 600x600)
  - `release_date`: 발매일 ("2017-03-24")
  - `genre`: 장르 ("K-Pop")
  - `preview_url`: 30초 미리듣기 오디오 (Optional)

### Step 2. Lyrics (From Genius API) 🎤

- **Role**: "가사(Text)" 확보
- **Action**: Step 1에서 얻은 `artist` + `title`로 Genius를 검색합니다.
- **Acquired Data**:
  - `lyrics`: 노래 가사 전체

### Step 3. Community (From YouTube via yt-dlp) 💬

- **Role**: "감성(Mood) & 반응(Reaction)" 확보
- **Action**: `artist` + `title`로 유튜브를 검색하여 상위 영상의 댓글을 수집합니다.
- **Acquired Data**:
  - `comments`: 사용자지 설 (필터링됨)
  - `view_count`: 대중성 지표

### Step 4. Refinement (LLM) 🧠

- **Role**: "최종 태깅(Tagging)"
- **Action**: 위 3가지 데이터를 LLM에 입력하여 최종 메타데이터를 생성합니다.
- **Output**: `mood_tags`, `situation_tags`, `scene_summary`
