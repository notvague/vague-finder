"""피드백 후보 보충의 합성 평가 자료를 만든다. 실제 곡·실제 피드백과 무관하다.

    python experiments/feedback_reuse/synthetic/make_synthetic.py

같은 난수 씨앗으로 `songs.jsonl`(곡 402개) · `queries.csv`(질의 10건) · `cases.jsonl`(사례 14건)을 다시 쓴다.
곡 요약은 장르·분위기·악기·보컬·주제의 조합이라 문장이 어색한 곳이 있다(조사 등) — 검색 기능 확인에는 지장이 없다.

구성: 일반 곡 360개 + 단서 묶음 6개(묶음마다 그 소리가 요약에 적힌 방해 곡 6개 + 요약에 그 소리가 **없는** 정답 곡 1개).
정답 곡은 사용자가 기억하는 단서가 곡 정보에 없어 기존 검색이 후보에 넣지 못하는 경우를 흉내 낸다.
"""
import csv, itertools, json, random
from pathlib import Path

out = Path(__file__).resolve().parent
rng = random.Random(20261013)

genres = ["발라드", "댄스", "록", "힙합", "R&B", "포크", "재즈", "트로트", "인디", "일렉트로니카"]
moods = ["잔잔하고 쓸쓸한", "밝고 경쾌한", "몽환적이고 느린", "강렬하고 빠른", "따뜻하고 포근한", "어둡고 무거운",
         "청량하고 시원한", "애절하고 호소력 있는"]
instruments = ["피아노 반주", "통기타 반주", "일렉 기타 리프", "묵직한 베이스", "신시사이저 패드", "현악 편곡",
               "브라스 섹션", "어쿠스틱 드럼"]
vocals = [("남성", "낮은 남성 보컬"), ("여성", "맑은 여성 보컬"), ("혼성", "남녀가 주고받는 보컬"), ("여성", "허스키한 여성 보컬"),
          ("남성", "고음이 시원한 남성 보컬")]
themes = ["이별 뒤의 그리움", "첫사랑의 설렘", "여름밤의 드라이브", "새벽의 혼잣말", "친구와의 우정", "고향 생각",
          "도시의 외로움", "다시 시작하는 용기"]

songs = []
def add(song_id, title, genre, gender, summary):
    songs.append({"song_id": song_id, "title": title, "artist": f"합성가수 {len(songs) % 37 + 1:02d}", "genre": genre,
                  "vocal_gender": gender, "search_style_summary": summary})

# 일반 곡 360개 — 조합을 고정 난수로 뽑는다(같은 요약이 겹치지 않게)
n = 0
seen = set()
while n < 360:
    genre, mood, inst, theme = rng.choice(genres), rng.choice(moods), rng.choice(instruments), rng.choice(themes)
    gender, vocal = rng.choice(vocals)
    summary = f"{mood} {genre}. {inst}와 {vocal}. {theme}을 노래한다."
    if summary in seen:
        continue
    seen.add(summary)
    n += 1
    add(f"syn{n:03d}", f"합성곡 {n:03d}", genre, gender, summary)

# 단서 묶음 6개 — 묶음마다 방해 곡 6개(요약에 그 소리가 적혀 있다) + 정답 곡 1개(요약에 그 소리가 없다)
families = {
    "whistle": ["휘파람 소리", "박수 소리"], "choir": ["어린이 합창", "종소리"], "phone": ["전화벨 소리", "자동응답기 음성"],
    "crowd": ["관객 함성", "앵콜 외침"], "rain": ["빗소리", "천둥소리"], "count": ["카운트다운", "잠깐의 정적"],
}
targets = {}
for family, (first, second) in families.items():
    for k in range(6):
        n += 1
        genre, mood = rng.choice(genres), rng.choice(moods)
        gender, vocal = rng.choice(vocals)
        where = rng.choice(["도입부에", "후렴에", "간주에", "곡 끝에"])
        add(f"syn{n:03d}", f"합성곡 {n:03d}", genre, gender,
            f"{mood} {genre}. {where} {first if k % 2 == 0 else second}가 들어가고 {vocal}가 이끈다.")
    n += 1
    targets[family] = f"syn{n:03d}"
target_summary = {
    "whistle": ("포크", "남성", "담담한 포크. 통기타 반주와 낮은 남성 보컬. 기차역에서의 작별을 노래한다."),
    "choir": ("인디", "여성", "미디엄 템포의 인디 팝. 신시사이저 패드와 맑은 여성 보컬. 졸업식 날의 기억을 노래한다."),
    "phone": ("R&B", "남성", "느린 R&B. 묵직한 베이스와 낮은 남성 보컬. 연락이 끊긴 사람을 그리워한다."),
    "crowd": ("록", "남성", "록 발라드. 일렉 기타 리프와 어쿠스틱 드럼. 늦은 밤 귀갓길을 노래한다."),
    "rain": ("재즈", "여성", "재즈풍 팝. 브라스 섹션과 허스키한 여성 보컬. 오래된 사진첩을 넘기는 마음을 노래한다."),
    "count": ("일렉트로니카", "혼성", "일렉트로니카. 신시사이저 패드와 남녀가 주고받는 보컬. 낯선 도시의 아침을 노래한다."),
}
for family, song_id in targets.items():
    genre, gender, summary = target_summary[family]
    idx = int(song_id[3:])
    songs.insert(idx - 1, {"song_id": song_id, "title": f"합성곡 {idx:03d}", "artist": f"합성가수 {idx % 37 + 1:02d}",
                           "genre": genre, "vocal_gender": gender, "search_style_summary": summary})
assert len(songs) == 360 + 6 * 7 == 402 and len({s["song_id"] for s in songs}) == 402
(out / "songs.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in songs), encoding="utf-8")

def pick(*needles, skip=0):
    """요약에 이 말들이 모두 든 일반 곡(1~360번) 가운데 skip번째."""
    found = [s for s in songs if int(s["song_id"][3:]) <= 360 and all(n in s["search_style_summary"] for n in needles)]
    return found[skip]["song_id"]

# 평가 질의 — 사례와 **따로 쓴** 문장. 사용자는 분위기·장르를 틀리게 기억하기도 한다.
queries = [
    ("f01", "신나는 노래였던 것 같은데 처음에 휘파람 불고 중간에 다 같이 박수 치는 부분이 있었어", targets["whistle"]),
    ("f02", "여자 가수 노래인데 후렴에서 애들이 떼로 따라 부르고 마지막에 종이 울렸던 걸로 기억해", targets["choir"]),
    ("f03", "전화가 울리면서 시작하고 누가 음성 메시지 남기는 소리가 나오는 슬픈 노래", targets["phone"]),
    ("f04", "노래 끝날 때 사람들이 환호하면서 한 곡 더 해 달라고 외치는 게 그대로 들어가 있던 곡", targets["crowd"]),
    ("f05", "처음부터 끝까지 비 내리는 소리가 깔려 있던 차분한 여자 노래", targets["rain"]),
    ("f06", "한창 달리다가 갑자기 뚝 끊기고 숫자를 세고 나서 다시 터지는 빠른 곡", targets["count"]),
    # 대조 질의 — 사례가 정답을 밀어내면 안 된다
    ("f07", "슬픈 발라드 추천해 줘", pick("쓸쓸한 발라드")),
    ("f08", "여름밤에 드라이브하면서 듣기 좋은 노래", pick("여름밤의 드라이브", "청량하고")),
    ("f09", "휘파람 소리가 들어간 밝은 댄스곡", "syn361"),
    ("f10", "고향 생각이 나는 따뜻한 노래", pick("따뜻하고 포근한", "고향 생각")),
]
with (out / "queries.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["query_id", "split", "query_type", "query", "relevant_ids"])
    for qid, text, rel in queries: w.writerow([qid, "dev", "search", text, rel])

cases = [
    # target — 운영자가 정리했을 법한 단서. 질의와 표현이 다르다
    ("t01", targets["whistle"], "도입부에 휘파람이 나오고 중간에 손뼉 소리가 들어가는 포크 곡", "target"),
    ("t02", targets["choir"], "후렴에 어린이 합창이 들어가고 종소리로 끝나는 곡", "target"),
    ("t03", targets["phone"], "전화벨 소리로 시작하고 자동응답기에 남긴 목소리가 나오는 곡", "target"),
    ("t04", targets["crowd"], "곡 끝에 관객 함성과 앵콜을 외치는 소리가 녹음된 곡", "target"),
    ("t05", targets["rain"], "곡 내내 빗소리가 배경에 깔리는 재즈풍 곡", "target"),
    ("t06", targets["count"], "중간에 멈췄다가 카운트다운을 하고 다시 시작하는 빠른 곡", "target"),
    # 비슷한 단서의 다른 곡
    ("s01", "syn362", "휘파람과 박수 소리가 들어가는 곡", "similar_clue_other_song"),
    ("s02", "syn383", "관객이 환호하는 소리가 들어간 라이브 느낌의 곡", "similar_clue_other_song"),
    # 다른 단서의 같은 곡
    ("o01", targets["whistle"], "가사에 기차역에서 헤어지는 장면이 나오는 곡", "other_clue_same_song"),
    ("o02", targets["rain"], "피아노 전주가 길게 이어지는 곡", "other_clue_same_song"),
    # 잘못된 사유로 승인된 사례 — 단서와 곡이 맞지 않는다
    ("w01", "syn010", "전화가 울리면서 시작하고 음성 메시지가 나오는 곡", "wrong_reason"),
    ("w02", "syn020", "끝날 때 사람들이 한 곡 더를 외치는 곡", "wrong_reason"),
    # 넓은 단서
    ("b01", pick("쓸쓸한 발라드", skip=1), "슬픈 발라드", "broad"),
    ("b02", pick("밝고 경쾌한 댄스"), "신나는 노래", "broad"),
]
(out / "cases.jsonl").write_text("".join(json.dumps(
    {"case_id": c, "song_id": s, "reusable_text": t, "group": g}, ensure_ascii=False) + "\n" for c, s, t, g in cases), encoding="utf-8")
