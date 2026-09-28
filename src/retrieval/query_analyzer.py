"""
[Retrieval] Query Analyzer
설명: 한국어 자연어 질의를 Gemini로 분석하여 의도 분류, 한국어 태그 추출,
      CLAP/SigLIP2 전용 영어 질의, 모달리티 가중치를 한 번에 생성합니다.
작성자: 황찬혁 (Full)
생성일: 2026-05-23
"""
import os
import re
import asyncio
import json
import logging
import threading
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai

from src.retrieval import timing
from src.backend.schemas.query import (
    QueryAnalysis,
    TitleConstraints,
    TitleMeaningClue,
)
from src.retrieval.lyrics_query import extract_lyric_clues, normalize_lyric_surface
from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
    extract_audio_evidence_text,
    fallback_modality_payload,
    has_explicit_audio_clue,
    has_explicit_visual_clue,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _reference_year() -> int:
    """상대적 시기 해석 기준 연도. 평가 재현성을 위해 환경변수로 고정 가능."""
    raw = os.getenv("SEARCH_REFERENCE_YEAR", "").strip()
    if raw.isdigit() and 1900 <= int(raw) <= 2100:
        return int(raw)
    return time.gmtime().tm_year

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a Korean music search query analyzer.

Given a Korean natural language music search query, output a single JSON object.

## Output fields

| field | type | description |
|---|---|---|
| intent_type | string | one of: "mood", "place", "time", "genre", "artist", "lyrics", "mixed". Use "lyrics" when the query explicitly asks about words/phrases that appear *in the lyrics* (e.g. "가사에 '봄'이 나오는 노래"). |
| korean_tags | string[] | concise Korean noun keywords for BM25 (NO spaces inside each tag, max 8 items). Include mood/scene/genre/vocal keywords. Do NOT put song titles or artist names here. |
| lyric_keywords | string[] | ONLY the exact words the user explicitly says must appear in the lyrics (e.g. "가사에 '찰나'라는 단어"→["찰나"]). Empty list if none. |
| lyric_clues | object[] | Structured lyric clues. Each object is {{"text": string, "kind": "verbatim"|"partial"|"phonetic"|"semantic", "variants": string[], "confidence": 0-1, "source": "model"}}. Use verbatim for a quoted continuous lyric, partial for an unquoted continuous fragment the user remembers as sung, phonetic for heard sounds/misheard foreign words, and semantic for a paraphrased lyric meaning. For phonetic clues keep the heard Korean in text and put 1-5 plausible original-language spellings in variants. Otherwise variants=[]. |
| lyric_semantic_query | string | A concise Korean paraphrase of lyric meaning when the user describes the story/content rather than actual sung words. Empty string when none. |
| song_title | string | The song title if the query names a specific song, KEEPING original spacing (e.g. "사랑은 늘 도망가"). Empty string if none. |
| title_constraints | object | Structural clues explicitly remembered about the title, NOT the title itself. Fields: script ("latin","hangul","hanja","numeric","mixed" or null), char_count (int|null), word_count (int|null), contains_number (bool|null), repeated_char (bool|null), repeated_word (bool|null), confidence (0-1). Only fill clues explicitly stated or strongly implied by the user's memory. |
| title_meaning_clue | object | Meaning/category remembered about an unknown title, separate from title_constraints. Fields: {{"kind": "foreign_person_name"|"person_name"|"place_name"|"sentence"|"question"|"onomatopoeia"|"object_name"|"other"|null, "text": string, "confidence": 0-1}}. Example: "외국 사람 이름 같은 제목" is foreign_person_name. Empty when the query does not explicitly describe the title's meaning/type. |
| release_era | object | Release-period memory. Fields: {{"start_year": int|null, "end_year": int|null, "confidence": 0-1}}. Absolute periods may have high confidence. Resolve relative periods against reference year {reference_year}, but use a deliberately broad range and low confidence for vague "최근/요즘" memories. Do not fill for timeless words such as "옛날/예전" alone. |
| artist_type | object | Stable identity of the credited artist, not the arrangement of this song. Fields: {{"values": string[], "confidence": 0-1}}. Allowed values: "솔로", "그룹", "듀오", "밴드". Fill only when the query describes the artist/team itself. |
| performance_clues | object | What is heard inside this song, separate from artist_type. Fields: {{"vocal_count": "solo"|"duet"|"multiple"|"choir"|null, "vocal_roles": string[], "sound_ensemble": string[], "confidence": 0-1}}. Preserve singing/rap/featuring/narration/chorus roles and band/acoustic/orchestral/electronic/live/a-cappella sound clues here. |
| artist_name | string | The artist name if the query names a specific artist, KEEPING original spacing (e.g. "임영웅"). Empty string if none. |
| artist_name_alt | string[] | ALL plausible written forms of the artist used in Korean music databases (Melon). Include Korean, English (no space), English (with space). e.g. "뉴진스" → ["NewJeans","New Jeans"], "방탄소년단" → ["BTS","Bangtan"], "에스파" → ["aespa"], "아이브" → ["IVE","IVE (아이브)"]. Empty list if artist_name is empty. |
| vocal_gender | string or null | "남성" if the user asks for a male/man's song, "여성" for female, "혼성" for mixed. null if not specified. |
| genre | string | The genre if explicitly named (e.g. "발라드", "랩/힙합", "R&B/Soul"). Empty string if not specified. |
| image_english_query | string | **SigLIP2-only visual prompt.** Describe only what can physically be seen in the album artwork: colors, objects/people, composition, material/texture, photography/illustration style, and visible typography. Users may omit words such as "album", "cover", or "표지" while describing a coherent static artwork; that still requires a visual prompt. Empty string only when there is no album-art evidence. |
| audio_english_query | string | **CLAP-only auditory prompt.** Describe only what can physically be heard in the waveform: vocal timbre/count/roles, instruments, sound effects, genre, tempo, rhythm, dynamics, production texture, and auditory mood. Empty string when the user gives no explicit auditory clue. |
| has_visual_clue | boolean | `true` when the query describes album-art appearance — color, drawing/illustration, photo, material/texture, static layout, visible title/typography, or depicted objects. Do not require the literal noun "album/cover/표지" when several coherent artwork details make the intent clear. A plain mood/atmosphere, listening imagery, lyrics scene, animation/drama/movie scene, music video, or stage visual is NOT album art. When false, image_english_query must be "" and image weight must be 0. |
| modality_weights | object | {{"text": float, "image": float, "audio": float}} — must sum to 1.0. A modality with an empty dedicated English query must have weight 0. |
| text_alpha | float | Within the text path, dense(KoE5 semantic) vs sparse(BM25 keyword) balance. 1.0=pure semantic, 0.0=pure keyword. Use LOW (≤0.4) when the query demands exact words/proper nouns (lyric_keywords present, artist name, song title); use HIGH (≥0.6) for vague mood/atmosphere queries. |
| confidence | float | 0.0–1.0 |

## Title constraint rules

Title constraints are different from song_title.

- "제목이 알파벳 세 글자였던 것 같아"
  -> script="latin", char_count=3
- "영어 단어 하나였어"
  -> script="latin", word_count=1
- "제목에 숫자가 들어갔어"
  -> contains_number=true
- "같은 영어 알파벳 두 글자였어"
  -> script="latin", char_count=2, repeated_char=true
- "같은 단어를 두 번 반복하는 제목"
  -> repeated_word=true
- "영어 한 글자 제목"
  -> script="latin", char_count=1

Do NOT guess the actual song title from these clues.
Do NOT invent title constraints that the user did not mention.
Phrases such as "것 같아", "아마", "기억에는" should lower
title_constraints.confidence, but the constraints should still be preserved.

Title meaning clues are also memories, not exact titles. Keep them separate:

- "외국 사람 이름 같은 제목" -> kind="foreign_person_name"
- "사람 이름이 제목" -> kind="person_name"
- "도시나 나라 이름 같은 제목" -> kind="place_name"
- "문장처럼 긴 제목" -> kind="sentence"
- "질문하는 제목" -> kind="question"
- "의성어 같은 제목" -> kind="onomatopoeia"
- Never convert these descriptions into song_title or a hard title constraint.

## Release era and artist type rules

- "2000년대" -> start_year=2000, end_year=2009
- "2000년대 초반" -> 2000-2003
- "2000년대 중반" -> 2004-2006
- "2000년대 후반" -> 2007-2009
- "2000년대 중후반" -> 2004-2009
- "2010년대 중반" -> 2014-2016
- "2009년쯤" -> 2009-2009 with lower confidence
- Resolve "올해/작년/재작년/N년 전" against reference year {reference_year}.
- "신곡/막 나온 곡" may use {reference_year_minus_one}-{reference_year} with medium confidence.
- "최근/요즘 나온/근래" may use a broad {recent_start_year}-{reference_year}
  range with low confidence. Never narrow a vague relative memory to one year.
- Do not convert timeless expressions such as "옛날" or "예전" alone.
- "솔로 가수" -> artist_type.values=["솔로"]
- "아이돌 그룹", "걸그룹", "보이그룹", "3인조 그룹" -> ["그룹"]
- "2인조 듀오", "듀오 팀" -> ["듀오"]
- "록 밴드", "인디밴드가 부른 곡" -> ["밴드"]
- "남자가/여자가 부르는" describes the vocal, not artist_type.
- "남녀 듀엣 곡", "둘이 함께 부르는" describes this song's performance,
  not necessarily a credited duo artist.
- "남자는 랩하고 여자는 노래", "피처링 보컬", "코러스/합창/나레이션"
  belong only in performance_clues.
- "밴드 사운드/밴드처럼 들리는/밴드 편곡" belongs only in
  performance_clues.sound_ensemble, not artist_type.
- Never infer group/duo solely from several voices being heard in one song.
- If the user says "솔로? 밴드?" preserve both values and lower confidence.
- These fields are soft memories. Never treat them as mandatory filters.

## Modality-specific embedding prompt rules

The two English embedding queries are independent projections of the user's
evidence, NOT two general summaries. Never copy one into the other and never
translate the whole query into both fields.

### audio_english_query (CLAP)

- Include only evidence that could be recognized by listening to the audio:
  vocal gender/timbre/count/roles, singing vs rap, instruments, effects,
  melody, harmony, tempo, rhythm, energy, dynamics, production texture,
  acoustic environment, genre, and mood as expressed by sound.
- Preserve relational arrangement details, not just a bag of instrument names:
  which sound is nearly alone or dominant, which instrument stays underneath,
  what is absent, the section where a sound enters, and whether layers build or
  fade over time. Preserve vocal delivery such as pleading, whisper-like, or
  restrained singing. These distinctions are often the strongest CLAP clues.
- Exclude album-cover colors, objects, people in a photo, illustration style,
  composition, background/foreground, typography, visible letters/numbers,
  and every phrase such as "album cover", "cover art", or "pictured".
- Exclude song/artist names, release era, popularity/platform history, title
  structure, exact lyric words, and lyric-story meaning unless the user also
  gives a separate audible property. Those clues belong to text retrieval.
- Do not invent a known artist's or song's sound. If the query only names an
  artist/title or only describes a cover, return "".

### image_english_query (SigLIP2)

- Include only album-art evidence that could be recognized by
  looking at the image: color palette, visible subjects/objects, spatial
  composition, photography/drawing/pixel/collage style, material/texture,
  background, and visible typography or symbols.
- The cover noun may be omitted. A cohesive static-artwork description such as
  "손으로 그린 꽃, 거친 종이 질감, 손글씨 제목" is still album-art evidence:
  set has_visual_clue=true and generate image_english_query.
- Do NOT reinterpret music-video/animation/drama/movie/stage scenes, scenes
  described in lyrics, or an image merely evoked while listening as cover art.
- Exclude vocal gender/timbre, singing/rap, instruments as sounds, genre,
  tempo, rhythm, melody, dynamics, production, lyrics, and musical mood that
  is not explicitly tied to the cover's visual style.
- An instrument or singer may appear only when the user says it is visibly
  shown on the cover (e.g. "표지에 기타를 든 남자 사진"). Describe the
  visible person/object, never the sound it might make.
- If has_visual_clue=false, return "" exactly. Never turn a listening scene
  such as "비 오는 날 듣기 좋은" into rain cover art.

### Mixed-query invariant

For a query with both cover and sound clues, produce two disjoint prompts.
Example: "별이 가득한 밤하늘 표지, 남성 보컬 피아노 발라드" ->
- image_english_query: "Album cover showing a star-filled night sky."
- audio_english_query: "Male vocals in a gentle piano-led ballad."

The audio prompt must not mention stars/night-sky/cover, and the image prompt
must not mention male vocals/piano/ballad. The same rule applies to every mixed
query regardless of clue order or sentence structure.

## Modality weight guidelines

- **High text** (≥0.7): genre name, artist name, or specific album/song queries
  e.g. "아이유 발라드" → text=0.8, image=0.0, audio=0.2
- **High image** (>=0.3): ONLY when the query describes album artwork (has_visual_clue=true), including a coherent static-artwork description that omits the cover noun
  e.g. "분홍 하트가 그려진 표지", "파란 도트 그림 앨범" → text=0.4, image=0.6, audio=0.0
  A mood/atmosphere phrase is NOT a cover description: "새벽 카페 감성", "비 오는 창가" -> has_visual_clue=false, image=0.0
- **High audio** (≥0.3): instrument or sound-texture queries
  e.g. "어쿠스틱 기타 잔잔한", "피아노 재즈" → text=0.5, image=0.0, audio=0.5
- **Detailed audio memory**: when there is no exact title/artist/lyrics/cover clue
  and the user gives several audible attributes or an arrangement relation,
  use audio>=0.5 even if a vague release era such as "최근" is also present.
  Example: "통기타 한 대만 들리다가 후렴부터 현악기가 점점 쌓이고
  여자가 속삭이듯 부름" -> preserve sparse guitar, breathy female delivery,
  and the gradual string build in audio_english_query; use text=0.5, audio=0.5.
- **Lyrics intent**: query asks about specific words IN the lyrics
  e.g. "가사에 '봄'이 나오는 노래" -> text=1.0, image=0.0, audio=0.0
  → Also use LOW text_alpha (≤0.3) so BM25 sparse keyword matching dominates
- A lyric fragment does NOT require quotation marks. Phrases such as
  "너를 사랑해도 되겠니라는 가사가 있었어" and
  "가사에 머리부터 발끝까지라는 말이 나와" contain continuous partial lyrics.
- Preserve verbatim/partial lyric text exactly as the user wrote it. Never stem,
  split, summarize, translate, or correct it inside lyric_clues.
- For heard sounds such as "아파운더웨이 하는 가사", keep the original sound
  in phonetic.text and put plausible original-language spellings such as
  "I found the way" in that clue's variants. Do not silently replace the user's form.
- Descriptions such as "상대를 밀어내면서도 그리워하는 내용" are semantic,
  not verbatim. Put the concise meaning in lyric_semantic_query.
- intent_type="lyrics" is reserved for actual sung surface words:
  verbatim, partial, or phonetic lyric clues.
- When verbatim/partial clues exist, use intent_type="lyrics" and text_alpha≤0.15.
- When only phonetic clues exist, still use intent_type="lyrics", but keep
  text_alpha around 0.50-0.60 so an uncertain spelling cannot collapse dense retrieval.
- A semantic description of lyric meaning is not an exact lyrics intent.
  Use intent_type="mixed" and text_alpha>=0.60.
- Preserve audio weights when the query describes instruments, sound effects,
  beeps, whistles, vocal texture, or production sounds.
- Pure cover-only query: audio_english_query="" and audio=0.0. Do not infer a
  soundtrack from colors or illustration style.
- Artist/title/lyrics/metadata-only query with no audible clue:
  audio_english_query="" and audio=0.0.
- Any empty modality query must have that modality's weight set to 0.0.
- **Balanced**: mixed or mood queries with no strong single axis

## Few-shot examples

Query: "2000년대 여자 그룹 댄스곡인데 제목이 알파벳 세 글자였던 것 같아"
{{
  "intent_type": "mixed",
  "korean_tags": ["2000년대", "걸그룹", "댄스곡"],
  "lyric_keywords": [],
  "song_title": "",
  "title_constraints": {{
    "script": "latin",
    "char_count": 3,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.75
  }},
  "release_era": {{
    "start_year": 2000,
    "end_year": 2009,
    "confidence": 0.9
  }},
  "artist_type": {{
    "values": ["그룹"],
    "confidence": 0.9
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": "여성",
  "genre": "",
  "image_english_query": "",
  "audio_english_query": "An upbeat, fast-paced Korean girl-group dance track with energetic vocals.",
  "has_visual_clue": false,
  "modality_weights": {{
    "text": 0.7,
    "image": 0.0,
    "audio": 0.3
  }},
  "text_alpha": 0.4,
  "confidence": 0.9
}}

Query: "비 오는 날 듣기 좋은 재즈"
{{
  "intent_type": "mood",
  "korean_tags": ["비", "재즈", "감성", "잔잔한", "새벽"],
  "lyric_keywords": [],
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": null,
  "genre": "재즈",
  "image_english_query": "",
  "audio_english_query": "Soothing jazz with mellow piano, brushed drums, and an intimate melancholic atmosphere.",
  "has_visual_clue": false,
  "modality_weights": {{"text": 0.5, "image": 0.0, "audio": 0.5}},
  "text_alpha": 0.7,
  "confidence": 0.95
}}

Query: "뉴진스 노래 추천해줘"
{{
  "intent_type": "artist",
  "korean_tags": ["걸그룹", "K팝", "청량"],
  "lyric_keywords": [],
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "artist_name": "뉴진스",
  "artist_name_alt": ["NewJeans", "New Jeans"],
  "vocal_gender": "여성",
  "genre": "",
  "image_english_query": "",
  "audio_english_query": "",
  "has_visual_clue": false,
  "modality_weights": {{"text": 1.0, "image": 0.0, "audio": 0.0}},
  "text_alpha": 0.15,
  "confidence": 0.98
}}

Query: "임영웅의 사랑은 늘 도망가"
{{
  "intent_type": "artist",
  "korean_tags": ["발라드", "남성솔로"],
  "lyric_keywords": [],
  "song_title": "사랑은 늘 도망가",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "artist_name": "임영웅",
  "artist_name_alt": ["임영웅", "Lim Young-woong"],
  "vocal_gender": "남성",
  "genre": "발라드",
  "image_english_query": "",
  "audio_english_query": "",
  "has_visual_clue": false,
  "modality_weights": {{"text": 1.0, "image": 0.0, "audio": 0.0}},
  "text_alpha": 0.2,
  "confidence": 0.99
}}

Query: "앨범 표지에 밤하늘에 별이 그려져있고, 남자 발라드 노래 가사에 찰나라는 단어가 포함되어있어"
{{
  "intent_type": "lyrics",
  "korean_tags": ["밤하늘", "별", "남성보컬", "발라드"],
  "lyric_keywords": ["찰나"],
  "lyric_clues": [
    {{"text": "찰나", "kind": "partial", "variants": [], "confidence": 0.95, "source": "model"}}
  ],
  "lyric_semantic_query": "",
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": "남성",
  "genre": "발라드",
  "image_english_query": "Album cover showing a star-filled night sky.",
  "audio_english_query": "A tender, emotional male-vocal ballad.",
  "has_visual_clue": true,
  "modality_weights": {{"text": 0.5, "image": 0.35, "audio": 0.15}},
  "text_alpha": 0.15,
  "confidence": 0.92
}}

Query: "분홍색 하트가 화면 중앙에 크게 있고 컬러풀한 일러스트 앨범 표지"
{{
  "intent_type": "place",
  "korean_tags": ["일러스트", "컬러풀", "하트"],
  "lyric_keywords": [],
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": null,
  "genre": "",
  "image_english_query": "Album cover with a large pink heart in the center and a colorful, playful illustration style.",
  "audio_english_query": "",
  "has_visual_clue": true,
  "modality_weights": {{"text": 0.35, "image": 0.65, "audio": 0.0}},
  "text_alpha": 0.5,
  "confidence": 0.9
}}

Query: "꽤 옛날 노래인데 가사에 너를 사랑해도 되겠니라는 말이 있었어. 남성 솔로이고 멜로디는 신났던 것 같아"
{{
  "intent_type": "lyrics",
  "korean_tags": ["옛날노래", "남성솔로", "미성", "신나는멜로디"],
  "lyric_keywords": [],
  "lyric_clues": [
    {{"text": "너를 사랑해도 되겠니", "kind": "partial", "variants": [], "confidence": 0.95, "source": "model"}}
  ],
  "lyric_semantic_query": "",
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": ["솔로"],
    "confidence": 0.75
  }},
  "performance_clues": {{
    "vocal_count": "solo",
    "vocal_roles": ["남성노래"],
    "sound_ensemble": [],
    "confidence": 0.8
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": "남성",
  "genre": "",
  "image_english_query": "",
  "audio_english_query": "An upbeat song with one male singer and a light, thin vocal timbre.",
  "has_visual_clue": false,
  "modality_weights": {{"text": 0.9, "image": 0.0, "audio": 0.1}},
  "text_alpha": 0.1,
  "confidence": 0.95
}}

Query: "남녀가 같이 부르는 노래인데 가사에서 아파운더웨이처럼 들렸어. 밴드 사운드였던 것 같아"
{{
  "intent_type": "lyrics",
  "korean_tags": ["남녀보컬", "밴드사운드"],
  "lyric_keywords": [],
  "lyric_clues": [
    {{"text": "아파운더웨이", "kind": "phonetic", "variants": ["I found the way", "I found a way"], "confidence": 0.72, "source": "model"}}
  ],
  "lyric_semantic_query": "",
  "song_title": "",
  "title_constraints": {{
    "script": null,
    "char_count": null,
    "word_count": null,
    "contains_number": null,
    "repeated_char": null,
    "repeated_word": null,
    "confidence": 0.0
  }},
  "release_era": {{
    "start_year": null,
    "end_year": null,
    "confidence": 0.0
  }},
  "artist_type": {{
    "values": [],
    "confidence": 0.0
  }},
  "performance_clues": {{
    "vocal_count": "duet",
    "vocal_roles": ["남성노래", "여성노래"],
    "sound_ensemble": ["밴드사운드"],
    "confidence": 0.85
  }},
  "artist_name": "",
  "artist_name_alt": [],
  "vocal_gender": "혼성",
  "genre": "",
  "image_english_query": "",
  "audio_english_query": "A male-female vocal duet over a full band arrangement.",
  "has_visual_clue": false,
  "modality_weights": {{"text": 0.65, "image": 0.0, "audio": 0.35}},
  "text_alpha": 0.55,
  "confidence": 0.9
}}

Now analyze:
Query: "{query}"
"""


# ---------------------------------------------------------------------------
# QueryAnalyzer
# ---------------------------------------------------------------------------

class QueryAnalyzer:
    """
    Gemini를 호출해 QueryAnalysis를 반환한다.
    API 키 미설정 또는 3회 연속 실패 시 confidence=0.0 fallback을 반환하며
    전용 영어 질의를 규칙 기반으로 안전하게 분리한다. 인식할 수 없는 단서는
    공용 원문으로 두 모델을 오염시키지 않고 text-only로 폴백한다.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
        # **분석 전체에 쓸 수 있는 시간.** 한 번의 호출이 아니라 재시도와 대기까지
        # 합친 값이다. 제한이 없으면 Gemini가 응답하지 않을 때 요청이 끝나지 않고,
        # 화면은 "검색 중…"에서 영원히 멈춘다(2026-09-24 실패 경로 확인).
        #
        # 기본 20초의 근거: 측정된 분석 중앙값이 2.6초다(experiments/latency/run_v01).
        # 정상 호출은 이 예산의 1/7도 쓰지 않으므로 멀쩡한 질의를 자를 일이 없고,
        # 막혔을 때는 20초 안에 규칙 폴백으로 내려와 검색을 마친다.
        self._budget_seconds = float(
            os.getenv("QUERY_ANALYSIS_TIMEOUT_SECONDS", "20")
        )
        self._client: Optional[genai.Client] = None
        # analyze()가 스레드에서 돌기 시작하면 이 지연 생성이 겹칠 수 있다.
        # 잠그지 않으면 두 스레드가 모두 None을 보고 클라이언트를 두 벌 만든다.
        self._client_lock = threading.Lock()
        if not self._api_key:
            logger.warning("[QueryAnalyzer] GEMINI_API_KEY 미설정 — fallback 모드로 동작합니다.")

    @property
    def _gemini(self) -> genai.Client:
        """첫 호출에서 한 번만 만든다.

        이중 확인: 잠금 밖에서 먼저 보고, 없을 때만 잠근다. 만들어진 뒤에는
        매 질의마다 잠금을 잡지 않는다(`dependencies.singleton`과 같은 이유·같은 꼴).
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def analysis_budget_seconds(self) -> float:
        """이 분석기가 쓰기로 한 시간. 호출부가 **벽시계 상한**을 걸 때 기준이 된다."""
        return self._budget_seconds

    # -- 한 번의 시도를 이루는 조각들 (동기·비동기 고리가 같이 쓴다) --------

    def _prompt(self, query: str) -> str:
        reference_year = _reference_year()
        return _PROMPT_TEMPLATE.format(
            query=query,
            reference_year=reference_year,
            reference_year_minus_one=reference_year - 1,
            recent_start_year=reference_year - 8,
        )

    def _config(self, remaining: float) -> dict:
        return {
            "response_mime_type": "application/json",
            # 재현성 확보: 동일 query 가 매번 같은 분석을 내도록 고정.
            # temperature=0 + top_p=1 = greedy decoding.
            # (C-1: 베이스라인 비교에서 비결정성으로 인한 회귀/개선 노이즈 차단)
            "temperature": 0.0,
            "top_p": 1.0,
            # 남은 예산만큼만 기다린다. 이 라이브러리는 **밀리초**로 받는다.
            #
            # **이것만으로는 총 시간이 묶이지 않는다.** httpx의 timeout은 연결·읽기
            # 같은 단계마다 따로 세는 값이고, 읽기는 *조각 하나*를 기다리는 시간이다.
            # 응답이 조금씩 도착하면 이 값을 넘겨 끝난다(0.1초 제한에 0.68초 확인).
            # 그래서 벽시계 상한은 호출부가 따로 건다(`routes/search.py`의 `_analyze`).
            "http_options": {"timeout": max(1, int(remaining * 1000))},
        }

    def _parse(self, query: str, response: Any) -> QueryAnalysis:
        raw: dict = json.loads(response.text)
        raw = _apply_lyric_safeguards(query, raw)
        raw = _apply_metadata_safeguards(query, raw)
        raw = apply_modality_query_safeguards(query, raw)
        raw["original_query"] = query
        analysis = QueryAnalysis(**raw)
        logger.debug(
            "[QueryAnalyzer] model=%s intent=%s "
            "weights=(t=%.2f i=%.2f a=%.2f) image_en='%s' audio_en='%s'",
            self._model_name,
            analysis.intent_type,
            analysis.modality_weights.text,
            analysis.modality_weights.image,
            analysis.modality_weights.audio,
            analysis.image_english_query,
            analysis.audio_english_query,
        )
        logger.info(
            "[QueryAnalyzer] Gemini 분석 성공 model=%s confidence=%.2f",
            self._model_name,
            analysis.confidence,
        )
        return analysis

    @staticmethod
    def _correction_for(exc: Exception, attempt: int) -> str:
        """이 실패를 다음 시도에 어떻게 알릴 것인가. 로그도 여기서 남긴다.

        모달리티 경계를 어긴 경우에만 새 정정을 만든다. 나머지 실패에서는 **빈
        문자열**을 돌려주는데, 이것은 "정정 없음"이 아니라 **"바꿀 것 없음"**이다 —
        호출부는 앞서 만든 정정을 그대로 들고 다음 시도로 간다.
        """
        if isinstance(exc, ModalityQueryValidationError):
            logger.warning(
                "[QueryAnalyzer] 모달리티 질의 검증 실패 (시도 %d/3): %s",
                attempt + 1,
                exc,
            )
            return (
                "\n\n## REQUIRED CORRECTION\n"
                "Your previous JSON violated the modality boundary: "
                f"{exc}. Regenerate the entire JSON object. Keep physical "
                "album-art details only in image_english_query, including "
                "coherent static artwork descriptions that omit the cover noun, "
                "and keep audible "
                "waveform details only in audio_english_query. Do not reuse a "
                "shared translation.\n"
            )
        if isinstance(exc, json.JSONDecodeError):
            logger.warning(
                "[QueryAnalyzer] JSON 파싱 실패 (시도 %d/3): %s", attempt + 1, exc
            )
        else:
            logger.warning(
                "[QueryAnalyzer] Gemini 호출 실패 (시도 %d/3): %s", attempt + 1, exc
            )
        return ""

    def _give_up(self, query: str, attempts: int, spent: bool) -> QueryAnalysis:
        if spent:
            logger.warning(
                "[QueryAnalyzer] 분석 예산 %.0f초를 다 썼다 (%d회 시도) → fallback: '%s'",
                self._budget_seconds,
                attempts,
                query,
            )
        else:
            logger.warning(
                "[QueryAnalyzer] %d회 실패 → fallback 반환: '%s'", attempts, query
            )
        return _fallback(query)

    # -- 두 개의 고리. 순서와 조건은 같고 기다리는 방식만 다르다 ------------

    def analyze(self, query: str) -> QueryAnalysis:
        """한국어 질의를 분석해 QueryAnalysis를 반환한다. **동기다.**

        평가 스크립트와 캐시 생성이 이 길로 온다. 비동기 라우트는 `analyze_async`를
        쓴다 — 시간이 다 됐을 때 **진행 중인 호출까지 끊으려면** 취소가 전파돼야
        하고, 동기 호출은 그게 안 된다(스레드를 버릴 수는 있어도 멈출 수는 없다).
        """
        if not self._api_key:
            return _fallback(query)

        prompt, correction = self._prompt(query), ""
        # 재시도와 대기를 **합쳐서** 여기까지다. 시도마다 제한을 걸면 3회 × 제한
        # 만큼 늘어나 결국 같은 문제가 커진 채로 남는다.
        deadline = time.monotonic() + self._budget_seconds
        attempts = 0
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._give_up(query, attempts, spent=True)
            attempts += 1
            try:
                # 시도마다 따로 남긴다. 성공 요청과 재시도 요청의 시간이 한 칸에
                # 섞이면 "분석이 느리다"와 "분석이 한 번 실패했다"를 구분할 수 없다.
                with timing.step("analysis.gemini", attempt=attempt + 1):
                    response = self._gemini.models.generate_content(
                        model=self._model_name,
                        contents=prompt + correction,
                        config=self._config(remaining),
                    )
                return self._parse(query, response)
            except Exception as exc:
                # **새 지시가 있을 때만 바꾼다.** 통신·JSON 오류로 앞서 만든 정정을
                # 지우면 마지막 시도가 최초 프롬프트로 돌아가 같은 모달리티 오류를
                # 그대로 되풀이한다(정정 포함 여부가 False→True→False가 됐다).
                correction = self._correction_for(exc, attempt) or correction

            if attempt < 2:
                # 대기도 예산 안이다. 남은 시간보다 오래 자면 제한을 넘긴다.
                nap = min(1.0, max(0.0, deadline - time.monotonic()))
                with timing.step("analysis.retry_sleep", attempt=attempt + 1):
                    time.sleep(nap)

        return self._give_up(query, attempts, spent=False)

    async def analyze_async(self, query: str) -> QueryAnalysis:
        """같은 분석을 **취소할 수 있는 형태로** 한다.

        `asyncio.wait_for`로 묶으면 시간이 다 됐을 때 진행 중인 HTTP 호출까지
        취소된다. 동기판을 스레드에 얹어 놓고 기다림만 끊으면, 버려진 스레드가
        계속 돌면서 실행기 자리를 붙들고 결과는 버려진다.
        """
        if not self._api_key:
            return _fallback(query)

        prompt, correction = self._prompt(query), ""
        deadline = time.monotonic() + self._budget_seconds
        attempts = 0
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._give_up(query, attempts, spent=True)
            attempts += 1
            try:
                with timing.step("analysis.gemini", attempt=attempt + 1):
                    response = await self._gemini.aio.models.generate_content(
                        model=self._model_name,
                        contents=prompt + correction,
                        config=self._config(remaining),
                    )
                return self._parse(query, response)
            except Exception as exc:
                # 동기판과 같은 이유로 **유지**한다(위 주석 참고).
                correction = self._correction_for(exc, attempt) or correction

            if attempt < 2:
                nap = min(1.0, max(0.0, deadline - time.monotonic()))
                with timing.step("analysis.retry_sleep", attempt=attempt + 1):
                    await asyncio.sleep(nap)

        return self._give_up(query, attempts, spent=False)


def rule_fallback(query: str) -> QueryAnalysis:
    """Gemini 없이 규칙만으로 만든 분석. **공개 이름이다.**

    시간이 다 돼 분석을 포기한 호출부(`routes/search.py`)도 같은 것을 쓴다 —
    폴백이 두 종류가 되면 `looks_like_fallback` 판정이 갈린다.
    """
    return _fallback(query)


def _fallback(query: str) -> QueryAnalysis:
    """Gemini 불가 시 모달리티별 보수적 규칙 프롬프트를 반환한다."""
    raw = _apply_metadata_safeguards(
        query,
        _apply_lyric_safeguards(
            query,
            {
                "intent_type": "mixed",
                "lyric_keywords": [],
                "lyric_clues": [],
                "lyric_semantic_query": "",
                "text_alpha": 0.5,
            },
        ),
    )
    modality = fallback_modality_payload(
        query,
        raw["performance_clues"],
    )
    return QueryAnalysis(
        original_query=query,
        intent_type=raw["intent_type"],
        korean_tags=raw.get("korean_tags", []),
        lyric_keywords=raw["lyric_keywords"],
        lyric_clues=raw["lyric_clues"],
        lyric_semantic_query=raw["lyric_semantic_query"],
        song_title="",
        title_constraints=TitleConstraints(),
        title_meaning_clue=TitleMeaningClue(
            **raw["title_meaning_clue"]
        ),
        release_era=raw["release_era"],
        artist_type=raw["artist_type"],
        performance_clues=raw["performance_clues"],
        artist_name="",
        artist_name_alt=[],
        vocal_gender=None,
        genre="",
        image_english_query=modality["image_english_query"],
        audio_english_query=modality["audio_english_query"],
        has_visual_clue=modality["has_visual_clue"],
        modality_weights=modality["modality_weights"],
        text_alpha=raw["text_alpha"],
        confidence=0.0,
    )


def _apply_lyric_safeguards(query: str, raw: dict) -> dict:
    """모델 출력에 규칙 기반 가사 추출을 병합하고 검색 invariant를 강제한다."""
    enriched = dict(raw)
    merged: dict[str, dict] = {}
    surface_kinds = {"verbatim", "partial", "phonetic"}
    lyric_cues = ("가사", "가삿말", "노랫말", "구절", "후렴")
    phonetic_context = any(
        token in query
        for token in (
            "들리",
            "들었",
            "들었던",
            "처럼",
            "같이 들",
            "발음",
            "소리",
            "외국어",
            "영어 가사",
        )
    )

    def add_clue(item: dict) -> None:
        text = str(item.get("text") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not text or kind not in {"verbatim", "partial", "phonetic", "semantic"}:
            return
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 1.0))))
        except (TypeError, ValueError):
            confidence = 1.0
        source = item.get("source")
        if source not in {"rule", "model", "fallback"}:
            source = "model"
        normalized = normalize_lyric_surface(text)
        if not normalized:
            return
        raw_variants = item.get("variants") or []
        if isinstance(raw_variants, str):
            raw_variants = [raw_variants]
        elif not isinstance(raw_variants, (list, tuple, set)):
            raw_variants = []
        variants: list[str] = []
        variant_keys: set[str] = {normalized}
        for raw_variant in raw_variants:
            variant = str(raw_variant or "").strip()
            variant_key = normalize_lyric_surface(variant)
            if not variant_key or variant_key in variant_keys:
                continue
            variant_keys.add(variant_key)
            variants.append(variant)
            if len(variants) >= 8:
                break
        # 모델이 ``꽤 옛날 노래인데 가사에 실제 구절``처럼 검색 문맥까지
        # 가사로 반환한 경우는 버린다. 규칙 추출 결과는 별도 안전 검사를 통과한다.
        if source == "model" and kind in surface_kinds and any(
            cue in text for cue in lyric_cues
        ):
            return
        # 같은 표면 문자열을 verbatim/partial로 중복 검색하지 않는다.
        key = (
            f"surface:{normalized}"
            if kind in surface_kinds
            else f"semantic:{normalized}"
        )
        current = merged.get(key)
        candidate_priority = (
            kind == "verbatim",
            kind == "phonetic" and phonetic_context,
            confidence,
            source == "rule",
        )
        current_priority = (
            current["kind"] == "verbatim",
            current["kind"] == "phonetic" and phonetic_context,
            float(current["confidence"]),
            current["source"] == "rule",
        ) if current is not None else None
        combined_variants = list(
            dict.fromkeys(
                [
                    *(current.get("variants", []) if current else []),
                    *variants,
                ]
            )
        )[:8]
        if current is None or candidate_priority > current_priority:
            merged[key] = {
                "text": text,
                "kind": kind,
                "variants": combined_variants,
                "confidence": confidence,
                "source": source,
            }
        elif combined_variants != current.get("variants", []):
            current["variants"] = combined_variants

    for item in enriched.get("lyric_clues") or []:
        if isinstance(item, dict):
            add_clue(item)

    for clue in extract_lyric_clues(query):
        add_clue(
            {
                "text": clue.text,
                "kind": clue.kind,
                "confidence": clue.confidence,
                "source": "rule",
            }
        )

    # 구버전 프롬프트/모델이 연속 구절을 lyric_keywords에 넣어도 표면 검색에 참여시킨다.
    raw_keywords = enriched.get("lyric_keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]
    elif not isinstance(raw_keywords, (list, tuple, set)):
        raw_keywords = []
    lyric_keywords = [
        str(value).strip()
        for value in raw_keywords
        if str(value).strip()
    ]
    if any(cue in query for cue in lyric_cues):
        existing_surface = [
            normalize_lyric_surface(item["text"])
            for item in merged.values()
            if item["kind"] in surface_kinds
        ]
        for keyword in lyric_keywords:
            normalized_keyword = normalize_lyric_surface(keyword)
            if not normalized_keyword or any(cue in keyword for cue in lyric_cues):
                continue
            # 모델이 연속 구절과 그 구성 단어를 모두 반환하면 가장 긴 연속 구절만
            # exact 검색에 사용한다. 짧은 단어 OR 검색으로 후보가 폭증하는 것을 막는다.
            if any(
                normalized_keyword != phrase
                and normalized_keyword in phrase
                for phrase in existing_surface
            ):
                continue
            add_clue(
                {
                    "text": keyword,
                    "kind": "partial",
                    "confidence": 0.90,
                    "source": "model",
                }
            )

    clues = list(merged.values())
    rule_surfaces = [
        normalize_lyric_surface(clue["text"])
        for clue in clues
        if clue["source"] == "rule" and clue["kind"] in surface_kinds
    ]
    clues = [
        clue
        for clue in clues
        if not (
            clue["source"] == "model"
            and clue["kind"] in surface_kinds
            and any(
                normalize_lyric_surface(clue["text"]) != rule_surface
                and normalize_lyric_surface(clue["text"]) in rule_surface
                for rule_surface in rule_surfaces
            )
        )
    ]
    # 구형 검색/평가 코드도 가사 구절을 관찰할 수 있도록 확정적인 표면 단서는
    # lyric_keywords에 역호환 형태로 함께 유지한다. phonetic은 정확 문자열이라고
    # 단정할 수 없으므로 제외한다.
    lyric_keywords = [
        clue["text"]
        for clue in clues
        if clue["kind"] in {"verbatim", "partial"}
    ]
    deduped_keywords: list[str] = []
    keyword_keys: set[str] = set()
    for keyword in lyric_keywords:
        key = normalize_lyric_surface(keyword)
        if key and key not in keyword_keys:
            keyword_keys.add(key)
            deduped_keywords.append(keyword)

    semantic_query = str(enriched.get("lyric_semantic_query") or "").strip()
    has_exact_surface = any(
        clue["kind"] in {"verbatim", "partial"}
        for clue in clues
    )
    has_phonetic = any(clue["kind"] == "phonetic" for clue in clues)
    has_lexical = bool(has_exact_surface or has_phonetic or deduped_keywords)
    has_semantic = bool(
        semantic_query
        or any(clue["kind"] == "semantic" for clue in clues)
    )

    try:
        current_alpha = float(enriched.get("text_alpha", 0.5))
    except (TypeError, ValueError):
        current_alpha = 0.5

    enriched["lyric_clues"] = clues
    enriched["lyric_keywords"] = deduped_keywords
    enriched["lyric_semantic_query"] = semantic_query

    if has_lexical:
        enriched["intent_type"] = "lyrics"
        if has_exact_surface or deduped_keywords:
            enriched["text_alpha"] = min(current_alpha, 0.15)
        else:
            # 들리는 대로 적은 표기는 철자가 불확실하다. BM25에 전부 걸면
            # 원어 가사와 어휘가 달라 후보가 사라지므로 dense를 절반 이상 유지한다.
            enriched["text_alpha"] = min(0.60, max(current_alpha, 0.50))

    elif has_semantic:
        # 줄거리식 가사 설명은 exact/BM25 가사 검색 의도가 아니다.
        if enriched.get("intent_type") == "lyrics":
            enriched["intent_type"] = "mixed"

        # 의미 설명은 KoE5 dense 검색을 중심으로 처리한다.
        enriched["text_alpha"] = max(current_alpha, 0.60)

    return enriched

# ---------------------------------------------------------------------------
# Release era / artist type safeguards
# ---------------------------------------------------------------------------

_DECADE_RE = re.compile(
    r"(?P<decade>(?:19|20)\d0)년대"
    r"(?:\s*(?P<phase>초중반|중후반|초반|중반|후반))?"
)

_SHORT_DECADE_RE = re.compile(
    r"(?<!\d)(?P<decade>\d{2})년대"
    r"(?:\s*(?P<phase>초중반|중후반|초반|중반|후반))?"
)

_EXACT_YEAR_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})년(?!대)"
)

_YEARS_AGO_RE = re.compile(
    r"(?P<count>\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)"
    r"\s*년\s*(?P<approx>쯤|정도)?\s*전"
)

_KOREAN_NUMBER = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}


def _release_era_dict(
    start_year: int,
    end_year: int,
    confidence: float,
) -> dict:
    start = max(1900, min(2100, int(start_year)))
    end = max(start, min(2100, int(end_year)))
    return {
        "start_year": start,
        "end_year": end,
        "confidence": confidence,
    }


def _extract_release_era(query: str) -> dict | None:
    reference_year = _reference_year()
    uncertain = any(
        token in query
        for token in (
            "같아",
            "같은데",
            "쯤",
            "정도",
            "기억",
        )
    )

    confidence = 0.75 if uncertain else 0.9

    exact = _EXACT_YEAR_RE.search(query)

    if exact:
        year = int(exact.group("year"))
        return _release_era_dict(year, year, confidence)

    decade_match = _DECADE_RE.search(query)
    if decade_match is not None:
        decade = int(decade_match.group("decade"))
        phase = decade_match.group("phase")
    else:
        short_decade_match = _SHORT_DECADE_RE.search(query)
        if short_decade_match is not None:
            short = int(short_decade_match.group("decade"))
            decade = 2000 + short if short <= 30 else 1900 + short
            phase = short_decade_match.group("phase")
        else:
            decade = -1
            phase = None

    ranges = {
        None: (decade, decade + 9),
        "초반": (decade, decade + 3),
        "초중반": (decade, decade + 6),
        "중반": (decade + 4, decade + 6),
        "후반": (decade + 7, decade + 9),
        "중후반": (decade + 4, decade + 9),
    }

    if decade >= 0:
        start_year, end_year = ranges[phase]
        return _release_era_dict(start_year, end_year, confidence)

    # 명시적인 상대 연도는 기준 연도에 대해 결정론적으로 해석한다.
    if "재작년" in query:
        return _release_era_dict(reference_year - 2, reference_year - 2, 0.85)
    if "작년" in query:
        return _release_era_dict(reference_year - 1, reference_year - 1, 0.85)
    if "올해" in query or "금년" in query:
        return _release_era_dict(reference_year, reference_year, 0.90)

    years_ago = _YEARS_AGO_RE.search(query)
    if years_ago is not None:
        raw_count = years_ago.group("count")
        count = int(raw_count) if raw_count.isdigit() else _KOREAN_NUMBER[raw_count]
        center = reference_year - count
        if years_ago.group("approx") or uncertain:
            return _release_era_dict(center - 1, center + 1, 0.60)
        return _release_era_dict(center, center, 0.75)

    if re.search(r"몇\s*년\s*(?:쯤|정도)?\s*전", query):
        return _release_era_dict(reference_year - 8, reference_year - 2, 0.35)

    if re.search(r"(?:최신곡?|신곡|막\s*나온\s*(?:곡|노래))", query):
        return _release_era_dict(reference_year - 1, reference_year, 0.65)

    if re.search(
        r"(?:최근|요즘|요새|근래)(?:에)?(?:\s*(?:나온|발매된?|유행한?))?"
        r"\s*(?:곡|노래|음악)?|얼마\s*안\s*된\s*(?:곡|노래)",
        query,
    ):
        # '최근'은 사람마다 폭이 크게 달라 후보 제거에 쓰면 위험하다.
        # 8년 창 + 낮은 confidence로 soft boost만 제공한다.
        return _release_era_dict(reference_year - 8, reference_year, 0.35)

    return None


def _extract_artist_type(query: str) -> dict | None:
    # 곡의 편성/보컬 역할을 말하는 표현은 credited artist의 타입 증거에서 제거한다.
    identity_text = re.sub(
        r"밴드\s*(?:사운드|느낌|편곡|반주|스타일|풍|연주)|"
        r"밴드처럼\s*들리(?:는|던|고)?|밴드\s*같이\s*들리(?:는|던|고)?",
        " ",
        query,
    )
    identity_text = re.sub(
        r"(?:남녀|두\s*사람|둘이)\s*(?:가|이|는)?\s*(?:함께|같이)?\s*"
        r"(?:부르(?:는|던)?|노래하(?:는|던)?|듀엣)|"
        r"듀엣\s*(?:곡|노래|무대|파트|보컬)",
        " ",
        identity_text,
    )
    identity_text = re.sub(
        r"(?:기타|피아노|악기)\s*솔로|솔로\s*(?:파트|가창|보컬\s*파트)",
        " ",
        identity_text,
    )

    values: list[str] = []

    if re.search(
        r"(?:남성|여성|남자|여자)?\s*솔로(?:\s*(?:가수|아티스트|싱어|"
        r"뮤지션|곡|노래|앨범|데뷔|활동))?",
        identity_text,
    ):
        values.append("솔로")

    if re.search(
        r"(?:걸\s*그룹|걸그룹|보이\s*그룹|보이그룹|아이돌\s*그룹|"
        r"혼성\s*그룹|다인조\s*(?:그룹|팀)|"
        r"(?:[3-9]|세|네|다섯|여섯|일곱|여덟|아홉)\s*인조\s*(?:그룹|팀|아이돌)|"
        r"그룹\s*(?:가수|아티스트|팀|멤버|출신|곡|노래|음악|이|가|의|은|는)?)",
        identity_text,
    ):
        values.append("그룹")

    if re.search(
        r"(?:듀오(?:\s*(?:가수|아티스트|팀|그룹|곡|노래|활동))?|"
        r"2\s*인조(?:\s*(?:듀오|그룹|팀|가수))?|"
        r"두\s*명으로\s*(?:이루어진|이뤄진|된)\s*(?:듀오|팀|그룹)|"
        r"듀엣\s*(?:팀|그룹|가수))",
        identity_text,
    ):
        values.append("듀오")

    if re.search(
        r"(?:(?:인디|록|락|혼성)\s*)?밴드(?:\s*(?:가수|그룹|팀|멤버|"
        r"출신|곡|노래|음악)|가|는|의|이)?",
        identity_text,
    ):
        values.append("밴드")

    values = list(dict.fromkeys(values))

    if not values:
        return None

    confidence = 0.55 if len(values) > 1 else 0.85

    if any(
        token in query
        for token in ("같아", "같은데", "인가", "?")
    ):
        confidence = min(confidence, 0.7)

    return {
        "values": values,
        "confidence": confidence,
    }


def _extract_title_meaning_clue(query: str) -> dict | None:
    """명시적으로 말한 제목 의미/유형만 결정론적으로 구조화한다.

    모델이 일반 곡 설명을 제목 설명으로 오해하면 별도 후보 경로가 불필요하게
    켜질 수 있으므로, 제목/곡명/노래 이름 문맥이 확인된 경우에만 반환한다.
    """
    if not re.search(r"(?:제목|곡명|노래\s*이름)", query):
        return None

    patterns = (
        (
            "foreign_person_name",
            r"(?:(?:외국|해외|서양|영어권)\s*(?:사람|인물|남자|여자|가수)?\s*이름|"
            r"외국인\s*이름)",
            "외국 사람 이름 같은 제목",
        ),
        (
            "person_name",
            r"(?:사람|인물|남자|여자|가수)\s*이름",
            "사람 이름 같은 제목",
        ),
        (
            "place_name",
            r"(?:도시|나라|국가|지역|지명|장소)\s*(?:이름|명칭)",
            "장소 이름 같은 제목",
        ),
        (
            "sentence",
            r"(?:긴\s*)?문장\s*(?:같|형태|처럼)|문장형",
            "문장 같은 제목",
        ),
        (
            "question",
            r"(?:질문|의문문|물어보는\s*말)\s*(?:같|형태|처럼)?",
            "질문 같은 제목",
        ),
        (
            "onomatopoeia",
            r"(?:의성어|의태어|효과음|소리\s*나는\s*말)\s*(?:같|형태|처럼)?",
            "의성어 같은 제목",
        ),
        (
            "object_name",
            r"(?:물건|사물|음식|동물|식물)\s*(?:이름|명칭)",
            "사물 이름 같은 제목",
        ),
    )
    uncertain = any(
        token in query
        for token in ("같아", "같은데", "같았", "아마", "기억")
    )
    for kind, pattern, text in patterns:
        if re.search(pattern, query):
            return {
                "kind": kind,
                "text": text,
                "confidence": 0.65 if uncertain else 0.85,
            }
    return None


def _extract_performance_clues(query: str) -> dict:
    """곡 안에서 들리는 보컬 역할/인원/편성을 고정 규칙으로 보존한다."""
    # 표지에 보이는 피아노/기타/가수 사진을 실제 사운드 단서로 오인하지 않는다.
    query = extract_audio_evidence_text(query)
    vocal_count: str | None = None
    roles: list[str] = []
    ensembles: list[str] = []

    if re.search(r"(?:합창|콰이어|성가대|대합창)", query):
        vocal_count = "choir"
    elif re.search(
        r"(?:듀엣|둘이|두\s*(?:명|사람)이?\s*(?:같이|함께)?\s*부르|"
        r"남녀(?:가|이)?\s*(?:같이|함께)\s*부르)",
        query,
    ):
        vocal_count = "duet"
    elif re.search(r"(?:여러\s*명이|여럿이|다\s*같이|떼창|다중\s*보컬)", query):
        vocal_count = "multiple"
    elif re.search(r"(?:혼자\s*부르|한\s*명이\s*부르|독창|단독\s*보컬|솔로\s*보컬)", query):
        vocal_count = "solo"

    role_patterns = (
        ("남성랩", r"남(?:자|성)(?:가|이|은|는)?\s*(?:랩|래핑)|남성\s*래퍼"),
        ("여성랩", r"여(?:자|성)(?:가|이|은|는)?\s*(?:랩|래핑)|여성\s*래퍼"),
        (
            "남성노래",
            r"남(?:자|성)(?:가|이|은|는)?\s*.{0,20}?"
            r"(?:노래하|부르|보컬|가창)|남성\s*보컬",
        ),
        (
            "여성노래",
            r"여(?:자|성)(?:가|이|은|는)?\s*.{0,20}?"
            r"(?:노래하|부르|보컬|가창)|여성\s*보컬",
        ),
        ("피처링보컬", r"(?:피처링|featuring|feat\.?|객원\s*보컬)"),
        ("나레이션", r"(?:나레이션|내레이션|낭독|말하듯\s*(?:부르|읽))"),
        ("코러스", r"(?:코러스|백업\s*보컬|백킹\s*보컬)"),
        ("합창", r"(?:합창|콰이어|성가대|떼창)"),
    )
    for name, pattern in role_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            roles.append(name)

    if re.search(
        r"남녀(?:가|이)?\s*(?:(?:같이|함께)\s*(?:부르|노래)|듀엣)",
        query,
    ):
        roles.extend(["남성노래", "여성노래"])

    # "남녀 듀엣"이라는 총칭 뒤에 역할을 더 구체적으로 말한 경우에는
    # 구체 역할을 우선한다(남자는 랩, 여자는 노래 → 남성노래를 중복 생성하지 않음).
    explicit_male_singing = bool(
        re.search(
            r"남(?:자|성)(?:가|이|은|는)?\s*.{0,20}?"
            r"(?:노래하|부르|보컬|가창)",
            query,
        )
    )
    explicit_female_singing = bool(
        re.search(
            r"여(?:자|성)(?:가|이|은|는)?\s*.{0,20}?"
            r"(?:노래하|부르|보컬|가창)",
            query,
        )
    )
    if "남성랩" in roles and not explicit_male_singing:
        roles = [role for role in roles if role != "남성노래"]
    if "여성랩" in roles and not explicit_female_singing:
        roles = [role for role in roles if role != "여성노래"]

    ensemble_patterns = (
        ("밴드사운드", r"밴드\s*(?:사운드|느낌|편곡|반주|스타일|풍|연주)|밴드처럼\s*들리"),
        ("어쿠스틱", r"(?:어쿠스틱|통기타|언플러그드)"),
        ("오케스트라", r"(?:오케스트라|관현악|웅장한\s*현악)"),
        ("전자음악", r"(?:전자음|일렉트로닉|신스|신디사이저|EDM)"),
        ("라이브", r"(?:라이브|공연장|관객\s*소리)"),
        ("아카펠라", r"(?:아카펠라|무반주\s*보컬)"),
        ("브라스", r"(?:브라스|트럼펫|트롬본|색소폰)"),
        ("스트링", r"(?:스트링|현악|바이올린|첼로)"),
        ("피아노", r"(?:피아노|건반)"),
        ("기타", r"(?:기타\s*소리|기타\s*리프|일렉\s*기타|통기타)"),
        ("퍼커션", r"(?:퍼커션|드럼|타악기)"),
        ("휘파람", r"(?:휘파람|휘슬링?|whistl(?:e|ing))"),
        ("비프음", r"(?:비프음?|삐\s*(?:소리|하는|하고|삐)?|beeps?)"),
        ("벨소리", r"(?:벨소리|종소리|차임|초인종)"),
        ("박수", r"(?:박수\s*소리|손뼉|claps?)"),
        ("핑거스냅", r"(?:핑거\s*스냅|손가락\s*튕기|스냅\s*소리)"),
        ("사이렌", r"(?:사이렌|경보음)"),
        ("전화음", r"(?:전화\s*(?:벨|연결음|통화음)|수화기\s*소리)"),
        ("자연음", r"(?:빗소리|파도\s*소리|새\s*소리|바람\s*소리|물\s*소리)"),
        ("플루트", r"(?:플루트|플룻|피리\s*소리)"),
        ("하모니카", r"(?:하모니카|하프\s*소리)"),
        ("국악", r"(?:국악|판소리|가야금|해금|대금|장구|사물놀이)"),
        ("인도풍", r"(?:인도\s*(?:음악|풍|느낌)|볼리우드|시타르)"),
        ("중동풍", r"(?:중동|아랍)\s*(?:음악|풍|느낌)"),
    )
    for name, pattern in ensemble_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            ensembles.append(name)

    roles = list(dict.fromkeys(roles))
    ensembles = list(dict.fromkeys(ensembles))
    has_clue = bool(vocal_count or roles or ensembles)
    uncertain = any(token in query for token in ("같아", "같은데", "듯", "아마"))
    return {
        "vocal_count": vocal_count,
        "vocal_roles": roles,
        "sound_ensemble": ensembles,
        "confidence": 0.70 if has_clue and uncertain else (0.90 if has_clue else 0.0),
    }


def _apply_metadata_safeguards(
    query: str,
    raw: dict,
) -> dict:
    enriched = dict(raw)

    rule_era = _extract_release_era(query)

    if rule_era is not None:
        enriched["release_era"] = rule_era

    elif not isinstance(
        enriched.get("release_era"),
        dict,
    ):
        enriched["release_era"] = {
            "start_year": None,
            "end_year": None,
            "confidence": 0.0,
        }

    rule_artist_type = _extract_artist_type(query)

    if rule_artist_type is not None:
        enriched["artist_type"] = rule_artist_type
    else:
        # artist_type은 후보 부스트에 직접 쓰인다. 모델이 '여러 명이 부름',
        # '밴드 사운드'를 그룹/밴드 정체성으로 오해한 값은 보존하지 않는다.
        enriched["artist_type"] = {
            "values": [],
            "confidence": 0.0,
        }

    # 제목 의미는 잘못 켜졌을 때 별도 후보 경로를 발생시키므로 모델 추측값을
    # 그대로 신뢰하지 않는다. 명시적인 제목 문맥을 규칙이 확인한 경우만 보존한다.
    rule_title_meaning = _extract_title_meaning_clue(query)
    enriched["title_meaning_clue"] = rule_title_meaning or {
        "kind": None,
        "text": "",
        "confidence": 0.0,
    }

    rule_performance = _extract_performance_clues(query)
    model_performance = enriched.get("performance_clues")
    if not isinstance(model_performance, dict):
        model_performance = {}
    has_audio_evidence = has_explicit_audio_clue(
        query,
        rule_performance,
    )
    if not has_audio_evidence:
        # 표지에 보이는 악기/인물을 모델이 공연 단서로 추측했더라도 버린다.
        model_performance = {}

    def _clean_string_list(value: object) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        return list(
            dict.fromkeys(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        )[:12]

    model_count = model_performance.get("vocal_count")
    if model_count not in {"solo", "duet", "multiple", "choir"}:
        model_count = None
    vocal_count = rule_performance["vocal_count"] or model_count
    roles = _clean_string_list(
        [
            *_clean_string_list(model_performance.get("vocal_roles")),
            *rule_performance["vocal_roles"],
        ]
    )
    role_aliases = {
        "남성보컬": "남성노래",
        "남자보컬": "남성노래",
        "남자노래": "남성노래",
        "여성보컬": "여성노래",
        "여자보컬": "여성노래",
        "여자노래": "여성노래",
        "내레이션": "나레이션",
        "백업보컬": "코러스",
        "백킹보컬": "코러스",
    }

    roles = list(
        dict.fromkeys(
            role_aliases.get(role, role)
            for role in roles
        )
    )
    ensembles = _clean_string_list(
        [
            *_clean_string_list(model_performance.get("sound_ensemble")),
            *rule_performance["sound_ensemble"],
        ]
    )
    try:
        model_performance_confidence = float(
            model_performance.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        model_performance_confidence = 0.0
    enriched["performance_clues"] = {
        "vocal_count": vocal_count,
        "vocal_roles": roles,
        "sound_ensemble": ensembles,
        "confidence": min(
            1.0,
            max(
                rule_performance["confidence"],
                model_performance_confidence if (vocal_count or roles or ensembles) else 0.0,
            ),
        ),
    }

    has_male_role = any(role.startswith("남성") for role in roles)
    has_female_role = any(role.startswith("여성") for role in roles)
    if has_male_role and has_female_role:
        enriched["vocal_gender"] = "혼성"
    elif has_male_role:
        enriched["vocal_gender"] = "남성"
    elif has_female_role:
        enriched["vocal_gender"] = "여성"

    explicit_performer_gender = bool(
        re.search(
            r"(?:(?:남자|남성|여자|여성)\s*(?:가수|보컬|그룹|솔로|아이돌|"
            r"래퍼|노래)|걸\s*그룹|걸그룹|보이\s*그룹|보이그룹)",
            query,
        )
    )
    visible_gendered_person = bool(
        re.search(
            r"(?:남자|남성|여자|여성)(?:의)?\s*(?:얼굴|인물|사진|초상)",
            query,
        )
    )
    if (
        has_explicit_visual_clue(query)
        and visible_gendered_person
        and not has_audio_evidence
        and not explicit_performer_gender
    ):
        # 커버 속 인물의 성별은 보컬 성별 증거가 아니다.
        enriched["vocal_gender"] = None

    # 모델이 누락하더라도 BM25에 유효한 음색·가창 태그는
    # 사용자 원문에서 결정적으로 복구한다.
    korean_tags = _clean_string_list(
        enriched.get("korean_tags")
    )

    audio_tag_rules = (
        (
            "호소력",
            r"(?:호소(?:하듯|하는|력)|애원하듯|감정을?\s*쏟)",
        ),
        (
            "허스키",
            r"(?:허스키|거친\s*목소리)",
        ),
        (
            "미성",
            r"(?:미성|얇은\s*목소리|가느다란\s*목소리|청아한\s*목소리)",
        ),
        (
            "가성",
            r"(?:가성|팔세토|falsetto)",
        ),
    )

    audio_evidence_text = extract_audio_evidence_text(query)

    for tag, pattern in audio_tag_rules:
        if re.search(pattern, audio_evidence_text, re.IGNORECASE):
            korean_tags.append(tag)

    enriched["korean_tags"] = list(
        dict.fromkeys(korean_tags)
    )[:8]

    return enriched
