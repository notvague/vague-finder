"""
[Step 3] LLM Refinement Module (Gemini 정제)

역할:
- Step1/2에서 모은 데이터(앨범소개/가사/댓글)를 기반으로
- Gemini 모델에 프롬프트를 보내서 JSON 형태의 정제 결과를 얻는다.

정제 결과 예시(반환 dict):
{
  "mood_tags": [...],
  "album_summary": "...",
  "comment_context": "...",
  "lyrics_summary": "...",
  "scene_summary": "...",
  "lyrics_highlight": "..."
}

주의:
- 네트워크/쿼터 문제로 실패할 수 있어 재시도 로직(3회)이 들어가 있음
"""
import os
import json
import logging
import time
from typing import Dict, List, Optional, Any

from google import genai
from dotenv import load_dotenv

# 로깅 설정
# logging configured by main
logger = logging.getLogger(__name__)

# .env 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 키가 없으면 경고
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY가 설정되지 않았습니다.")


def _ensure_list(value: Any) -> List[str]:
    """
    LLM 응답이 list / str / None 등으로 올 수 있으므로
    항상 list[str]로 정규화
    """
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _flatten_unique(*lists: List[str]) -> List[str]:
    """
    여러 태그 리스트를 합쳐서 중복 제거한 flat list 반환
    """
    seen = set()
    out = []
    for lst in lists:
        for item in lst:
            key = item.strip()
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def generate_prompt(
    metadata: Dict, 
    lyrics: Optional[str], 
    melon_comments: List[str],
    youtube_comments: List[str]
) -> str:
    """
    Gemini에게 보낼 "프롬프트 문자열" 생성 함수.

    입력:
    - metadata: 멜론에서 가져온 메타데이터 dict
    - lyrics: 가사 문자열
    - melon_comments: 멜론 댓글(필터링된 상위 리스트)
    - youtube_comments: 유튜브 댓글(필터링된 상위 리스트)

    출력:
    - 모델에게 던질 긴 prompt string
    """
    # 곡명/가수명: metadata 키가 원본/다른 소스일 수 있으니 fallback까찌 고려
    track = metadata.get("title", metadata.get("trackName", "Unknown"))
    artist = metadata.get("artist", metadata.get("artistName", "Unknown"))
    
    # 앨범소개
    album_desc = metadata.get("album_desc", "")
    genre = metadata.get("genre", "")
    album_name = metadata.get("album_name", "")

    playlist_titles = metadata.get("melon_playlist_titles", []) or []
    playlist_tags = metadata.get("melon_playlist_tags", []) or []

    playlist_title_block = "\n".join([f"- {t}" for t in playlist_titles[:5]]) if playlist_titles else "- 없음"
    playlist_tag_block = " ".join([f"#{t}" for t in playlist_tags[:15]]) if playlist_tags else "없음"
    
    # 텍스트 길이 제한
    lyrics_snippet = lyrics[:1500] if lyrics else "가사 없음" # 1500->1000
    album_desc_snippet = album_desc[:1200] if album_desc else "정보 없음"
    
    # 댓글 블록 생성 (Top 10개만 사용)
    m_block = "\n".join([f"- {c}" for c in melon_comments[:10]])
    y_block = "\n".join([f"- {c}" for c in youtube_comments[:10]])
    
    # 실제 프롬프트
    prompt = f"""
role: You are a "Music Vibe Analyst" building a dataset for a Korean music retrieval system.

SYSTEM GOAL:
This dataset will later be embedded for search.
Users often search songs by vague memory, emotion, atmosphere, scene, time, weather, color, instrument feel, or life context.

Examples of real user queries:
- "파란색 앨범표지의 우울한 느낌의 인디밴드 노래"
- "비 오는 밤에 혼자 듣기 좋은 먹먹한 곡"
- "겨울 공기 같은 서늘한 발라드"
- "퇴근길 버스에서 듣던 위로되는 남자 노래"
- "이별 후에 생각나는 잔잔한 피아노곡"

INPUT DATA:
1. Song: {artist} - {track}
2. Genre: {genre}
3. Album: {album_name}
4. Album Intro:
{album_desc_snippet}
...
5. Lyrics:
{lyrics_snippet}
...
6. Melon DJ Playlist Titles:
{playlist_title_block}
7. Melon DJ Playlist Tags:
{playlist_tag_block}
8. User Comments:
   [Melon Reviews]:
{m_block}
   [YouTube Reactions]:
{y_block}

TASK:
Analyze the song and output JSON fields for search-oriented metadata.
We need both:
1) the original summary fields
2) additional fine-grained structured fields

IMPORTANT RULES:
- Use Korean for all output values.
- Do NOT mention unrelated people names, fandom jokes, meme context or other song names.
- Be concrete and retrieval-friendly.
- Prefer search-useful words over poetic filler.
- If some field is uncertain, infer conservatively from lyrics/comments/album intro/genre/playlist titles/playlist tags.
- The user may remember visual, emotional, atmospheric, situational, or sound-related clues.

REQUIRED ORIGINAL FIELDS:
1. "mood_tags"
   - Flat merged tag list useful for search.
   - Must mix time, weather, activity, vibe, emotion, and situational memory cues.

2. "scene_summary"
   - Describe the SONG'S REPRESENTATIVE SCENE OR VISUAL IDENTITY.
   - Focus on the song itself, not listener behavior.
   - This should feel like the core cinematic / visual scene of the song.
   - Include representative landscape, time, atmosphere, and emotional tone if possible.
   - Do NOT write about "people often listen when..." here.
   - 1~2 Korean sentences.

Good example:
"늦은 밤 창가에 머물며 반딧불 같은 빛을 바라보는 조용한 풍경이 떠오른다. 사라질 듯한 사랑을 조심스럽게 붙잡는 서정적이고 아련한 분위기가 중심에 있다."

3. "album_summary"
   - A short Korean summary of album intro / album context.

4. "lyrics_summary"
   - Summarize the emotional core of the lyrics in 2 short Korean sentences.

5. "comment_context"
- Summarize how listeners tend to remember, revisit, or search for this song.
- Focus on SEARCHABLE CONTEXT, not general praise.
- Include concrete clues when possible:
  - when they listen to it (e.g. 밤, 새벽, 겨울, 비 오는 날)
  - in what situation they seek it out (e.g. 이별 후, 혼자 있을 때, 퇴근길, 잠 못 드는 밤)
  - what emotional or relationship memory it connects to
- Write in 1~2 short Korean sentences.
- Avoid artist praise, vocal praise, popularity statements, and vague compliments.
- Prefer retrieval-friendly context over abstract admiration.
- When generating "comment_context", prioritize comments that mention
  specific situations, memories, times, weather, places, or relationship contexts.

Good example:
"잠 못 드는 밤, 혼자 조용히 그리운 사람을 떠올릴 때 찾게 되는 곡으로 받아들여진다. 직접 말하지 못한 마음과 지난 사랑의 여운을 정리할 때 어울린다는 반응이 많다."

Bad example:
"많은 사람들이 좋아하고 위로를 받는 곡이다."

6. "lyrics_highlight"
   - The most memorable 1~2 lines of lyrics (Korean).

7. "time_weather_tags"
   - Short tags for time / season / weather.
   - Examples: ["새벽", "밤", "비", "겨울", "흐린 날"]

8. "place_activity_tags"
   - Short tags for place / activity / usage context.
   - Examples: ["창가", "버스", "드라이브", "산책", "퇴근길", "혼자 듣기"]

9. "emotion_tags"
   - Short emotion tags.
   - Examples: ["먹먹함", "쓸쓸함", "위로", "그리움", "설렘"]

10. "vibe_tags"
   - Short vibe / texture / atmosphere tags.
   - Examples: ["몽환적", "잔잔한", "서늘한", "따뜻한", "서정적"]

11. "relation_context_tags"
   - Short situation / relation / life-event tags.
   - Examples: ["이별", "짝사랑", "기다림", "재회", "혼자", "위로"]

12. "color_tags"
   - Short color / visual-tone tags inferred from the song's emotional imagery.
   - Examples: ["파랑", "회색", "보라", "검정", "하양"]
   - Only include colors that feel meaningfully connected to the song.

13. "visual_imagery"
   - 3~6 short visual images or scene fragments.
   - Examples:
     ["비 오는 창가", "회색 하늘", "새벽 버스 창문", "겨울 공기", "빈 밤거리"]

14. "sound_tags"
   - Short sound / arrangement / vocal-feel tags inferred from genre, lyrics mood, comments, and general presentation.
   - Examples: ["피아노 중심", "잔잔한 밴드 사운드", "호소력 짙은 보컬", "서정적인 멜로디"]

15. "search_style_summary"
- VERY IMPORTANT.
- Write ONE Korean sentence that sounds like a realistic user search query or retrieval description.
- This should be shorter and more query-like than scene_summary and comment_context.
- It should combine the most searchable clues:
  emotion + vibe + time/weather/place/activity + sound/genre if available.
- Do NOT write it like a review.
- Do NOT write it like a poetic scene paragraph.
- Do NOT write listener behavior summary here.
- Make it sound like something a user would type to find the song.

Good example:
"잠 못 오는 밤 혼자 창가에서 듣기 좋은 아련하고 서정적인 발라드"

Difference between fields:
- scene_summary = the song's representative cinematic scene
- comment_context = how listeners actually revisit/search the song in life situations
- search_style_summary = compressed retrieval-style query sentence

OUTPUT FORMAT (JSON ONLY):
{{
  "mood_tags": [...],
  "scene_summary": "...",
  "album_summary": "...",
  "lyrics_summary": "...",
  "comment_context": "...",
  "lyrics_highlight": "...",
  "time_weather_tags": [...],
  "place_activity_tags": [...],
  "emotion_tags": [...],
  "vibe_tags": [...],
  "relation_context_tags": [...],
  "color_tags": [...],
  "visual_imagery": [...],
  "sound_tags": [...],
  "search_style_summary": "..."
}}
"""
    return prompt


def normalize_refined_result(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 출력 스키마를 안정적으로 정규화
    - 기존 필드 유지
    - 신규 구조화 필드 추가
    - mood_tags는 세부 태그를 합친 flat list로 재구성
    """
    time_weather_tags = _ensure_list(result_json.get("time_weather_tags"))
    place_activity_tags = _ensure_list(result_json.get("place_activity_tags"))
    emotion_tags = _ensure_list(result_json.get("emotion_tags"))
    vibe_tags = _ensure_list(result_json.get("vibe_tags"))
    relation_context_tags = _ensure_list(result_json.get("relation_context_tags"))
    color_tags = _ensure_list(result_json.get("color_tags"))
    visual_imagery = _ensure_list(result_json.get("visual_imagery"))
    sound_tags = _ensure_list(result_json.get("sound_tags"))

    mood_tags_from_model = _ensure_list(result_json.get("mood_tags"))

    mood_tags = _flatten_unique(
        mood_tags_from_model,
        time_weather_tags,
        place_activity_tags,
        emotion_tags,
        vibe_tags,
        relation_context_tags,
        color_tags,
    )

    return {
        # 기존 필드 유지
        "mood_tags": mood_tags,
        "scene_summary": str(result_json.get("scene_summary", "") or "").strip(),
        "album_summary": str(result_json.get("album_summary", "") or "").strip(),
        "lyrics_summary": str(result_json.get("lyrics_summary", "") or "").strip(),
        "comment_context": str(result_json.get("comment_context", "") or "").strip(),
        "lyrics_highlight": str(result_json.get("lyrics_highlight", "") or "").strip(),

        # 신규 세분화 필드
        "time_weather_tags": time_weather_tags,
        "place_activity_tags": place_activity_tags,
        "emotion_tags": emotion_tags,
        "vibe_tags": vibe_tags,
        "relation_context_tags": relation_context_tags,
        "color_tags": color_tags,
        "visual_imagery": visual_imagery,
        "sound_tags": sound_tags,
        "search_style_summary": str(result_json.get("search_style_summary", "") or "").strip(),
    }


def refine_data(metadata: Dict, lyrics: Optional[str], reaction: Optional[Dict]) -> Dict:
    """
    Gemini를 호출해서 정제 결과 JSON(dict)를 반환하는 함수.

    - 실패하면 {} 또는 "분석실패" 기본값을 반환한다.
    - main.py에서는 refined.get(...)로 접근하므로 dict를 반환하는 것이 중요.
    """
    # 1) Melon 댓글은 metadata 안에 있음
    melon_comments = metadata.get("melon_comments", [])
    
    # 2) YouTube 댓글은 reaction 딕셔너리 안에 있음
    youtube_comments = reaction.get("comments", []) if reaction else []

    # 3) 프롬프트 생성
    prompt = generate_prompt(metadata, lyrics, melon_comments, youtube_comments)
    
    # 4) API 키 없으면 즉시 실패
    # === Gemini 모드 ===
    if not GEMINI_API_KEY:
        logger.error("Gemini API Key 누락! .env파일 확인요망")
        return {}
        
    # 5) 재시도 설정 (유료 플랜이므로 안정성을 위해 3회 재시도)
    max_retries = 3

    # 6) Gemini 클라이언트 생성
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # 7) 재시도 루프
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            result_json = json.loads(response.text)
            normalized = normalize_refined_result(result_json)

            logger.info("LLM 응답 생성 완료 (Gemini)")
            return normalized
            
        # 시도 실패
        except Exception as e:
            logger.warning(f"Gemini API 호출 중 오류 (시도 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # 2초 대기 후 재시도
                
    logger.error("최대 재시도 횟수 초과. 분석 실패.")
    # 실패 시 기본값 반환
    return {
        # 기존 필드
        "mood_tags": ["분석실패", "API_ERROR"],
        "scene_summary": "API 호출 한도 초과로 분석하지 못했습니다.",
        "album_summary": "분석 실패",
        "comment_context": "분석 실패",
        "lyrics_summary": "분석 실패",
        "lyrics_highlight": "분석 실패",

        # 신규 필드
        "time_weather_tags": [],
        "place_activity_tags": [],
        "emotion_tags": [],
        "vibe_tags": [],
        "relation_context_tags": [],
        "color_tags": [],
        "visual_imagery": [],
        "sound_tags": [],
        "search_style_summary": "분석 실패",
    }


def build_comment_selection_prompt(
        batch_comments: List[Dict[str, Any]],
        source_name: str = "YouTube"
) -> str:
    """
    댓글 배치 판별용 프롬프트
    - 음악 감성 검색용 데이터셋 구축 목적
    """

    comment_lines = []
    for idx, item in enumerate(batch_comments):
        text = item.get("text", "")
        likes = item.get("like_count", 0)
        comment_lines.append(f'[{idx}] likes={likes} | text="{text}"')

    joined = "\n".join(comment_lines)

    return f"""
You are building a dataset for a **music search engine based on emotional memory**.

The source of comments is: {source_name}

Users often remember songs like this:

- "a sad indie band song with a blue album cover"
- "a melancholic song that feels like a rainy night"
- "a song that feels like driving alone at night"
- "a quiet song that feels like winter air"

Therefore we want comments that describe **the feeling, scene, or atmosphere of the song**.

GOOD COMMENTS usually contain:

• emotions
  (슬프다, 먹먹하다, 위로된다, 설렌다)

• atmosphere / vibe
  (몽환적, 잔잔한, 따뜻한, 차분한)

• time
  (새벽, 밤, 겨울밤, 해질녘)

• weather
  (비 오는 날, 흐린 날, 눈 오는 날)

• place
  (창가, 버스, 지하철, 카페, 드라이브)

• situations
  (이별 후, 혼자 있을 때, 공부할 때, 산책할 때)

• visual imagery
  (회색 하늘, 파란 느낌, 별빛 같은 노래)

GOOD EXAMPLES:

- "비 오는 새벽에 창가에서 듣기 좋은 노래"
- "이별 후 밤에 혼자 걸으면서 듣기 좋은 곡"
- "회색 하늘 같은 분위기의 슬픈 발라드"
- "겨울밤 공기 같은 몽환적인 노래"

BAD COMMENTS:

- fan praise ("레전드", "최고", "미쳤다")
- appearance praise
- "2024년에 듣는 사람?"
- "보러 옴"
- "파트 확인"
- lyrics copy
- random jokes
- arguments

IMPORTANT:
Prefer comments that describe:
- scenes
- atmosphere
- emotions
- imagery
- weather / time / place / situations
- color-like impressions if present

If a comment only contains fan praise, jokes, viewer check-ins, meme reactions,
or meta discussion, do NOT keep it unless it still clearly describes the song's emotional scene.

TASK:
For each comment decide:

keep:
True only if the comment helps describe
the **emotion, scene, or atmosphere of the song**.

score:
5 = perfect emotional scene description  
4 = strong emotional context  
3 = somewhat useful  
2 = weak but related  
1 = barely useful  
0 = useless

Return JSON ONLY.

Format:

[
  {{
    "index": 0,
    "keep": true,
    "score": 5,
    "reason": "비 오는 밤 장면과 감정이 함께 묘사됨",
    "aspects": ["emotion","weather","time","mood"]
  }}
]

Comments:
{joined}
"""


def select_emotional_comments_with_llm(
    comments: List[Dict[str, Any]],
    target_count: int = 20,
    batch_size: int = 15,
    source_name: str = "YouTube"
) -> List[str]:
    """
    좋아요 순 후보 댓글을 배치 단위로 LLM이 평가해서
    감성 서술형 댓글을 선별한다.

    개선된 fallback 정책:
    1) keep=true 댓글을 우선 사용
    2) 부족하면 keep=false 중 score >= 4 만 추가
    3) 그래도 부족하면 score >= 3 만 추가
    4) 그래도 부족하면 score >= 2 만 추가
    5) score 0~1 은 절대 추가하지 않음

    즉, target_count=20은 "최대 20개" 목표이고,
    품질이 부족하면 20개 미만으로 끝날 수 있다.
    """

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY가 없어 댓글 LLM 선별을 수행할 수 없습니다.")
        return [c.get("text", "") for c in comments[:target_count]]

    if not comments:
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)

    kept_items: List[Dict[str, Any]] = []
    reviewed_items: List[Dict[str, Any]] = []

    # 좋아요 순으로 정렬
    comments = sorted(comments, key=lambda x: x.get("like_count", 0), reverse=True)

    total_candidates = len(comments)
    logger.info(f"[LLM COMMENT SELECT:{source_name}] total_candidates={total_candidates}")

    for start in range(0, total_candidates, batch_size):
        batch = comments[start:start + batch_size]
        if not batch:
            break

        prompt = build_comment_selection_prompt(batch, source_name=source_name)

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            parsed = json.loads(response.text)

            for item in parsed:
                idx = item.get("index")
                if idx is None or not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= len(batch):
                    continue

                original = batch[idx]
                text = original.get("text", "")
                likes = int(original.get("like_count", 0) or 0)

                keep = bool(item.get("keep", False))

                # score는 안전하게 0~5로 정규화
                try:
                    score = int(item.get("score", 0) or 0)
                except Exception:
                    score = 0
                score = max(0, min(5, score))

                reason = str(item.get("reason", ""))
                aspects = item.get("aspects", [])

                record = {
                    "text": text,
                    "like_count": likes,
                    "keep": keep,
                    "score": score,
                    "reason": reason,
                    "aspects": aspects,
                }

                reviewed_items.append(record)
                if keep:
                    kept_items.append(record)

            logger.info(
                f"[LLM COMMENT SELECT] batch=({start}-{start + len(batch) - 1}) "
                f"kept_so_far={len(kept_items)} reviewed_so_far={len(reviewed_items)}"
            )

            # keep=true가 충분히 모였으면 중단
            if len(kept_items) >= target_count:
                break

        except Exception as e:
            logger.warning(f"[LLM COMMENT SELECT] 배치 평가 실패: {e}")

            # 실패한 배치는 reviewed_items에 낮은 점수 fallback 후보로만 넣음
            # score=1 이므로 아래 fallback 단계에서 절대 선택되지 않음
            for item in batch:
                reviewed_items.append({
                    "text": item.get("text", ""),
                    "like_count": int(item.get("like_count", 0) or 0),
                    "keep": False,
                    "score": 1,
                    "reason": "LLM 평가 실패",
                    "aspects": [],
                })

    # -----------------------
    # 1) keep=true 우선 정렬
    # -----------------------
    kept_items_sorted = sorted(
        kept_items,
        key=lambda x: (x.get("score", 0), x.get("like_count", 0)),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    selected_texts = set()

    for item in kept_items_sorted:
        text = item["text"]
        if text in selected_texts:
            continue
        selected.append(item)
        selected_texts.add(text)
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        pool_score3 = sorted(
            [x for x in reviewed_items if x["text"] not in selected_texts and x.get("score", 0) >= 3],
            key=lambda x: (x.get("score", 0), x.get("like_count", 0)),
            reverse=True,
        )
        for item in pool_score3:
            selected.append(item)
            selected_texts.add(item["text"])
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        pool_score2 = sorted(
            [x for x in reviewed_items if x["text"] not in selected_texts and x.get("score", 0) >= 2],
            key=lambda x: (x.get("score", 0), x.get("like_count", 0)),
            reverse=True,
        )
        for item in pool_score2:
            selected.append(item)
            selected_texts.add(item["text"])
            if len(selected) >= target_count:
                break

    if len(selected) == 0:
        logger.warning(f"[LLM COMMENT SELECT:{source_name}] keep/score 기반 선별 결과 0개 -> 안전 fallback 사용")
        emergency_pool = sorted(
            reviewed_items,
            key=lambda x: x.get("like_count", 0),
            reverse=True,
        )
        for item in emergency_pool[:min(target_count, len(emergency_pool))]:
            if item["text"] in selected_texts:
                continue
            selected.append(item)
            selected_texts.add(item["text"])

    if len(selected) == 0:
        logger.warning(f"[LLM COMMENT SELECT:{source_name}] reviewed_items도 0개 -> 원본 후보 fallback 사용")
        for item in comments[:min(target_count, len(comments))]:
            text = item.get("text", "")
            if text in selected_texts:
                continue
            selected.append({
                "text": text,
                "like_count": int(item.get("like_count", 0) or 0),
                "keep": False,
                "score": 0,
                "reason": "raw fallback",
                "aspects": [],
            })
            selected_texts.add(text)

    logger.info(
        f"[LLM COMMENT SELECT:{source_name}] final_selected={len(selected)} / requested={target_count}"
    )

    return [item["text"] for item in selected[:target_count]]