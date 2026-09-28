"""A 조건 passage에서 감정 질의 토큰이 어디서 왔는지(태그·가사·그 밖) 센다.

사용 (A worktree 루트에서):
    PYTHONPATH=. python <reproduce>/token_sources.py <작업폴더>/passages_wA.json <출력 CSV>

passages_wA.json은 dump_passages.py 결과다. 가사가 들어 있으므로 커밋하지 않는다.
여기서 세는 문서 수는 토크나이저 기준이라 BM25 파라미터의 doc_freq와 1곡 정도 다를 수 있다.
"""
import csv
import json
import sys

from src.embedding.text.korean_bm25_tokenizer import tokenize_for_bm25

TOKENS = ["슬프다", "그립다", "기쁘다", "설레다", "추억", "위로"]
TAG_FIELDS = ["mood_tags", "time_weather_tags", "place_activity_tags", "emotion_tags", "vibe_tags",
              "relation_context_tags", "color_tags", "sound_tags", "melon_playlist_tags", "visual_imagery"]

passages = json.load(open(sys.argv[1], encoding="utf-8"))
songs = {}
for line in open("data/all_songs.jsonl", encoding="utf-8"):
    song = json.loads(line)
    songs[str(song.get("song_id"))] = song

cache = {}
def tokens(text):
    if text not in cache:
        cache[text] = set(tokenize_for_bm25(text))
    return cache[text]

rows = []
for token in TOKENS:
    docs = [sid for sid, passage in passages.items() if token in tokens(passage)]
    in_tags = in_lyrics = in_neither = 0
    for sid in docs:
        song = songs[sid]
        semantic = song.get("semantic_analysis") or {}
        community = song.get("community_feedback") or {}
        tag_text = " ".join(sum([semantic.get(k) or [] for k in TAG_FIELDS], []) + (community.get("fans_tags") or []))
        tagged = token in tokens(tag_text)
        lyric = token in tokens((song.get("lyrics_data") or {}).get("full_lyrics") or "")
        in_tags += tagged
        in_lyrics += lyric
        in_neither += not tagged and not lyric
    rows.append({"token": token, "A_docs": len(docs), "in_tags": in_tags,
                 "in_lyrics": in_lyrics, "in_neither_tags_nor_lyrics": in_neither})

with open(sys.argv[2], "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
