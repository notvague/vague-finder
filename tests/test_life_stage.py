"""생애 단계 표현("중학교 때·어릴 때")은 사용자 나이에 걸린 시간 단서다 — NEXT_WORK §2-13.

규칙: ① 발매 시기를 추측하지 않는다(release_era 비움) ② korean_tags로 새지 않는다 ③ 임베딩 질의문(search_text)에서 뺀다
④ 가사 내용·곡의 사건으로 쓰인 같은 말은 잡지 않는다 ⑤ 출생 연도를 받으면 창을 만든다.
"""
import pytest

from src.backend.schemas.query import LifeStageClue, QueryAnalysis
from src.retrieval import query_analyzer as qa


@pytest.mark.parametrize("query, stage, text", [
    ("중학교 때 많이 듣던 남자 발라드인데 제목이 기억 안 나", "middle", "중학교 때"),
    ("중딩 시절 유행했던 걸그룹 노래", "middle", "중딩 시절"),
    ("고3 때 야자 끝나고 듣던 노래", "high", "고3 때"),
    ("고등학교 수학여행 버스에서 틀었던 신나는 노래, 남자 그룹", "high", "고등학교 수학여행"),
    ("초등학교 때 엄청 유행했던 댄스곡", "elementary", "초등학교 때"),
    ("대학교 신입생 때 축제에서 다 같이 부르던 걸그룹 댄스곡", "freshman", "대학교 신입생 때"),
    ("대학 다닐 때 자주 듣던 밴드 노래", "college", "대학 다닐 때"),
    ("군대에서 자주 듣던 발라드", "military", "군대에서"),
    ("어릴 때 엄마가 차에서 자주 틀어주던 여자 가수 노래", "childhood", "어릴 때"),
    ("학창 시절에 맨날 듣던 노래", "school", "학창 시절"),
    ("스무 살 때 한창 듣던 노래", "age", "스무 살 때"),
    ("25살 때 자주 들었던 힙합", "age", "25살 때"),
    ("중학교 때 좋아하던 노래인데 여자 솔로였어", "middle", "중학교 때"),   # 곡을 목적어로 받는 '좋아하던'은 시간 단서
    ("고등학교 때 반에서 인기 있던 댄스곡", "high", "고등학교 때"),
    # 활용형 — 리뷰 2차
    ("어릴 때 자주 나오던 여자 가수 노래", "childhood", "어릴 때"),
    ("고등학교 때 제일 좋아했던 노래", "high", "고등학교 때"),
    ("초등학교 다닐 때 듣던 발라드", "elementary", "초등학교 다닐 때"),
    ("고등학교 다닐 때 듣던 발라드", "high", "고등학교 다닐 때"),
    ("중학교 다녔을 때 유행하던 걸그룹 노래", "middle", "중학교 다녔을 때"),
    ("중학교 때 유명했던 노래", "middle", "중학교 때"),             # 곡을 목적어로 받으면 시간 단서
    ("어릴 때 엄마가 들려주던 노래", "childhood", "어릴 때"),
    ("고등학교 때 교실에서 자주 들리던 발라드", "high", "고등학교 때"),
    # 동사와 곡 명사 사이의 수식어, 다른 곡 명사 — 리뷰 참고
    ("중학교 때 유명했던 아이돌 노래", "middle", "중학교 때"),
    ("고등학교 때 들리던 그 노래", "high", "고등학교 때"),
    ("어릴 때 엄마가 들려주던 자장가", "childhood", "어릴 때"),
    ("중학교 때 좋아하던 곡이야", "middle", "중학교 때"),
    ("고등학교 때 유명했던 가요 중에 발라드", "high", "고등학교 때"),
    # 뒤 제한이 흔한 표현을 막지 않는다 — 리뷰 5차
    ("고등학교 때 좋아하던 노래 하나 찾고 싶어요", "high", "고등학교 때"),
    ("중학교 때 유명했던 가요 하나", "middle", "중학교 때"),
    ("중학교 때 좋아했던 곡이에요", "middle", "중학교 때"),
    ("중학교 때 좋아했던 곡으로 기억해요", "middle", "중학교 때"),
    ("중학교 때 좋아했던 곡이다", "middle", "중학교 때"),
    ("중학교 때 좋아했던 곡이고 여자 솔로", "middle", "중학교 때"),
    ("중학교 때 좋아했던 곡이랑 비슷한 느낌", "middle", "중학교 때"),
    ("중학교 때 노래인데 여자 솔로였어", "middle", "중학교 때"),          # 곡 명사가 바로 붙어도 시간 단서
    ("고등학교 때 노래 하루 종일 듣던 거", "high", "고등학교 때"),
    ("어릴 때 유행했던 동요", "childhood", "어릴 때"),
    ("중학교 때 노래였어, 여자 솔로", "middle", "중학교 때"),
    ("고등학교 때 노래 중에 제일 슬픈 거", "high", "고등학교 때"),
    ("고등학교 때 그 노래", "high", "고등학교 때"),
    # 동사가 무엇을 꾸미는지로 판정 — 리뷰 8차(0b921cf). 내용 표지가 동사 앞 목적어면 막지 않는다
    ("중학교 때 가사를 외우면서 듣던 노래", "middle", "중학교 때"),
    ("어릴 때 즐겨 듣던 노래", "childhood", "어릴 때"),                     # 즐겨 먹던(비감지)과 짝
    ("중학교 때 돌려 듣던 시디", "middle", "중학교 때"),                     # 돌려 받은 편지(비감지)와 짝
    ("어릴 때 엄마가 불러주던 노래", "childhood", "어릴 때"),                # 부르던 내 이름(비감지)과 짝
    ("어릴 때 아빠가 차에서 틀어주던 올드팝", "childhood", "어릴 때"),      # 틀어주던 만화(비감지)와 짝
    ("중학교 때 많이 들었는데 제목이 기억 안 나", "middle", "중학교 때"),   # 종결·연결형은 청취가 서술
    ("중학교 때 많이 들었던 것 같아", "middle", "중학교 때"),               # 대명사 머리
    ("중학교 때 듣던 건데 여자 솔로", "middle", "중학교 때"),
    ("중학교 때 듣던 노래 제목이 기억 안 나", "middle", "중학교 때"),
    ("중학교 때 듣던 노래 가사에 비 얘기가 나와", "middle", "중학교 때"),   # 곡 명사가 내용 표지보다 먼저
    ("중학교 때 자주 듣던 가수인데 이름이 기억 안 나", "middle", "중학교 때"),
    ("중학교 때 듣던 추억을 회상하는 가사의 노래", "middle", "중학교 때"),  # '듣던 추억'은 청취의 기억
    ("중학교 때 좋아하던 애가 부르던 노래", "middle", "중학교 때"),
    ("7살 때 듣던 노래", "age", "7살 때"),                                  # 숫자 나이 5~9살도 받는다(리뷰 P3)
    ("일곱 살 때 듣던 노래", "age", "일곱 살 때"),
    ("I don't know 어릴 때 듣던 노래 I can't remember", "childhood", "어릴 때"),   # 어깨점은 따옴표가 아니다
    # 11차 — 10차에서 넣은 검사가 넓어 놓치던 흔한 문장. 조건을 좁혀서 되살린다
    # ① 'X가 나오는'은 부사(많이)·곡 명사 주어(노래가·신곡이·앨범이)에는 적용하지 않는다
    ("중학교 때 길거리에서 많이 나오던 노래", "middle", "중학교 때"),
    ("중학교 때 라디오에 많이 나왔던 노래", "middle", "중학교 때"),
    ("중학교 때 이 노래가 나왔는데 진짜 좋아했어", "middle", "중학교 때"),
    ("중학교 때 신곡이 나왔는데 맨날 들었어", "middle", "중학교 때"),
    ("고등학교 때 그 앨범이 나왔어", "high", "고등학교 때"),
    # ② 보조 용언 검사는 -어·-지·-다 뒤에서만, 다음 어절이 보조 용언 활용형일 때만(보컬·주변·말이야·못 찾겠어는 아니다)
    ("중학교 때 많이 들었는데 못 찾겠어", "middle", "중학교 때"),
    ("중학교 때 많이 들었는데 보컬이 여자였어", "middle", "중학교 때"),
    ("중학교 때 엄청 유행했는데 주변에 아는 사람이 없어", "middle", "중학교 때"),
    ("중학교 때 많이 들었는데 말이야", "middle", "중학교 때"),
    ("어릴 때 엄마가 틀어 주던 노래", "childhood", "어릴 때"),
    # ④ 절 경계로 보는 내용 표지는 계사·인용 꼴(가사인데·얘긴데·이야기야)뿐 — 가사가·가사도·가사는 뒤의 청취 동사는 본다
    ("중학교 때 가사가 좋아서 계속 듣던 노래", "middle", "중학교 때"),
    ("중학교 때 가사도 모르고 따라 부르던 노래", "middle", "중학교 때"),
    ("중학교 때 가사는 모르고 멜로디만 흥얼거리던 노래", "middle", "중학교 때"),
    # 깨지면 안 되는 문장(11차 확인)
    ("열일곱 살 때 듣던 노래", "age", "열일곱 살 때"),
    ("열 살 때 듣던 노래", "age", "열 살 때"),
    ("고등학교 때 유행했던", "high", "고등학교 때"),
    ("어릴 때 TV에서 자주 나오던 노래", "childhood", "어릴 때"),
    ("중학교 때 엄청 유행했다. 제목이 뭐였지", "middle", "중학교 때"),
    # 12차 — 곡 명사 뒤 '한'은 한다·한단·한대·한답일 때만 '하다'(한번·한때·한국·한참·한 곡은 아니다)
    ("중학교 때 듣던 노래 한번만 다시 듣고 싶다", "middle", "중학교 때"),
    ("어릴 때 듣던 노래 한때 유명했던", "childhood", "어릴 때"),
    ("중학교 때 좋아하던 노래 한국 여자 가수", "middle", "중학교 때"),
    ("중학교 때 유행했던 노래 한참 찾았어", "middle", "중학교 때"),
    ("중학교 때 노래 한 곡만 계속 들었어", "middle", "중학교 때"),
    # 12차 — 곡 명사 주어 + 나오-계열은 '내용 아님'까지만, 청취 여부는 종결형·머리 명사가 정한다
    ("중학교 때 이 노래가 나왔는데 진짜 좋아했어", "middle", "중학교 때"),
    ("중학교 때 신곡이 나왔는데 맨날 들었어", "middle", "중학교 때"),
    ("고등학교 때 그 앨범이 나왔어", "high", "고등학교 때"),
    ("중학교 때 길거리에서 많이 나오던 노래", "middle", "중학교 때"),
    # 13차 — 들은·들어본·들어봤·들어보 어간('들어' 통째는 들어가다·들어오다와 겹쳐 넣지 않는다)
    ("중학교 때 많이 들은 노래", "middle", "중학교 때"),
    ("중학교 때 처음 들은 노래", "middle", "중학교 때"),
    ("어릴 때 들은 노래인데 제목이 기억 안 나", "childhood", "어릴 때"),
    ("중학교 때 많이 들어본 노래", "middle", "중학교 때"),
    ("중학교 때 들어봤던 노래", "middle", "중학교 때"),
])
def test_life_stage_is_detected_as_a_time_clue(query, stage, text):
    found = qa._extract_life_stage(query)
    assert found is not None and (found["stage"], found["text"]) == (stage, text)
    assert found["age_from"] <= found["age_to"] and 0 < found["confidence"] <= 1


@pytest.mark.parametrize("query", [
    # 가사 내용 — v09 n001·n039
    "어릴 때 집이 어려워서 엄마가 짜장면이 싫다고 했다는 얘기 나오는 노래 뭐였지?",
    "무한도전에서 MC랑 가수가 팀 짜서 진지하게 불렀던 노래. '나 스무 살 적에' 이런 가사로 시작했던 것 같은데",
    # 곡의 사건 — v10 초안
    "군대 가 있는 동안 역주행해서 차트 1위까지 한 남자 솔로 노래",
    "판타지 소설 속 마법 주문 같은 단어가 제목인 여자 초등학생들에게 인기좋은 걸그룹 노래",
    "대학가요제에서 대상 받고 광고에도 나왔던 노래 있잖아. 잔잔한 발라드였던 것 같아, 2005년쯤",
    "몇 년 지나서 대학 축제 영상 때문에 다시 떴어",
    "비 오는 날 듣기 좋은 재즈",
    # 부사·'좋아하다'만으로는 시간 단서가 아니다 — 리뷰(#35)
    "어릴 때 엄마가 자주 아팠다는 가사 나오는 노래",
    "중학교 때 좋아하던 사람 얘기하는 가사",
    "고등학교 때 많이 싸웠던 친구한테 사과하는 내용의 노래",
    # 목적어 없는 유명·들려·들리는 가사 내용 — 리뷰 3차
    "중학교 때 유명했던 일진 얘기하는 가사",
    "어릴 때 엄마가 들려주던 옛날 이야기 같은 가사",
    "중학교 때 유명했던 가요제 얘기하는 가사",          # '가요제'는 곡이 아니다
    "고등학교 때 좋아하던 노래방 친구 얘기하는 가사",   # '노래방'도
    # 곡 명사가 어절 중간·끝에 있거나 동사·다른 명사의 일부 — 리뷰 4차
    "중학교 때 좋아했던 사람한테 가요 하고 말하는 가사",
    "고등학교 때 좋아하던 남자애가 노래 잘했다는 가사",
    "고등학교 때 좋아하던 애가 노래하는 모습 얘기하는 가사",
    "중학교 때 유명했던 왜곡 보도 얘기하는 가사",
    # 띄어 쓴 '노래 하면서·노래 해주던'은 동사구 — 리뷰 6차
    "중학교 때 좋아하던 오빠가 노래 하면서 울던 가사",
    "어릴 때 유명했던 가수가 노래 해주던 이야기",
    "고등학교 때 노래 하다가 울었던 장면이 나오는 가사",
    # 곡 명사가 다른 명사를 꾸미거나 동사 '가요', 창 끝에서 잘린 '노래방' — 리뷰 7차
    "고등학교 때 음악 선생님이 해준 얘기 같은 가사",
    "어릴 때 노래 대회 나갔던 이야기",
    "어릴 때 살던 동네에 다시 가요 라는 가사",
    "중학교 때 좋아하던 친구랑 매일 같이 놀러 가던 노래방 얘기하는 가사",
    # 동사가 꾸미는 머리 명사가 곡이 아니면 애매 — 잡지 않는다(검색문에서 지우지 않는다). 리뷰 8차(0b921cf), 음악 용법과 짝으로 검증
    "가사에 어릴 때 라디오를 듣던 엄마 이야기가 나와",     # 표현 앞 '가사에' + 머리 '이야기'
    "가사가 어릴 때 듣던 노래 같은 느낌이야",
    "어릴 때 즐겨 먹던 음식 얘기하는 가사",               # ↔ 즐겨 듣던 노래
    "중학교 때 돌려 받은 편지 얘기하는 가사",             # ↔ 돌려 듣던 시디
    "어릴 때 엄마가 부르던 내 이름 얘기하는 가사",        # ↔ 불러주던 노래
    "어릴 때 유행했던 놀이 얘기하는 가사",                # ↔ 유행했던 동요
    "어릴 때 유행했던 놀이",
    "어릴 때 틀어주던 만화 얘기",                         # ↔ 틀어주던 올드팝
    "어릴 때 틀어주던 게임 얘기",                         # '게임'은 대명사 '게'가 아니다
    "어릴 때 할머니한테 듣던 옛날 이야기 같은 가사",
    "어릴 때 듣던 라디오 사연 얘기하는 가사",             # 곡 명사 아닌 머리(라디오) → 애매
    "어릴 때 얘기 나오는 노래",                           # '얘기'가 주어인 나오는
    "어릴 때 집에서 나온 뒤 얘기하는 가사",
    "중학교 때 진짜 좋아했어",                            # 좋아하다는 곡 목적어가 있어야
    "어릴 때 좋아하던 걸 잃어버린 얘기 가사",
    "중학교 때 좋아하던 기억 얘기하는 가사",              # '좋아하던 기억'은 청취의 기억이 아니다
    # 11차 — ①~④를 좁혀도 계속 안 잡혀야 하는 짝
    "어릴 때 엄마가 부르던 내 이름이 나오는 노래",
    "어릴 때 유행했던 놀이가 나오는 노래",
    "어릴 때 듣던 라디오가 나오는 노래",
    "어릴 때 엄마가 나오던 꿈 얘기 가사",
    "어릴 때 얘기 나오는 노래",
    "어릴 때 차마 부르지 못한 이름 얘기 가사",
    "어릴 때 듣지 못했던 말 얘기하는 가사",
    "어릴 때 엄마가 틀어 놓던 만화 얘기",
    "어릴 때 잔소리 듣다 지쳐 집 나간 얘기 가사",
    "어릴 때 집이 가난했다는 가사인데 최근에 나온 노래야",
    "어릴 때 엄마가 돌아가셨다는 가사인데 요즘 유행하는 노래",
    "어릴 때 헤어진 친구 얘긴데 요즘 좋아하는 노래",
    "어릴 때 살던 동네 이야기야 작년에 나온 노래",
    "가사에 어릴 때 라디오를 듣던 엄마 이야기가 나와",
    "어릴 때 음악가가 되고 싶었다는 가사",
    "어릴 때 유행어 따라 하던 얘기 가사",
    "어릴 때 엄마가 틀어주던 만화 생각이 난다는 가사",
    # 12차 — '노래/음악/랩 한다·한대·한답·하면서'는 동사구
    "중학교 때 노래 한다고 설치던 친구 얘기 가사",
    "중학교 때 음악 한다고 집 나간 형 이야기",
    "중학교 때 노래 하면서 놀던 얘기 가사",
    "중학교 때 노래 한다는 애 얘기 가사",
    "중학교 때 음악 한대서 놀림받던 얘기",
    "중학교 때 랩 한답시고 까불던 얘기 가사",
    # 12차 — 곡 명사 주어 + 나오-계열이어도 꾸미는 머리가 곡이 아니면 내용
    "중학교 때 노래가 나오는 인형 선물 받은 얘기 가사",
    "어릴 때 음악이 나오던 오르골 얘기 가사",
    "중학교 때 음악이 나오면 춤추던 친구 얘기 가사",
    "어릴 때 엄마가 부르던 내 이름이 나오는 노래",
    "어릴 때 듣던 라디오가 나오는 노래",
    # 13차 — '들은'의 머리가 곡이 아니거나, 들어오다·들어가다
    "어릴 때 엄마한테 들은 이야기 같은 가사",
    "어릴 때 들은 말이 생각나는 가사",
    "어릴 때 소문으로 들은 얘기 가사",
    "어릴 때 집에 들어온 강아지 얘기 가사",
    "어릴 때 들어가던 골목 얘기 가사",
])
def test_lyric_content_and_song_events_are_not_life_stage(query):
    assert qa._extract_life_stage(query) is None


def _model_raw(**over):
    raw = {
        "intent_type": "mixed", "korean_tags": ["발라드", "남성보컬", "추억", "어린시절", "수학여행"],
        "lyric_keywords": [], "lyric_clues": [], "lyric_semantic_query": "", "text_alpha": 0.5,
        "release_era": {"start_year": 2000, "end_year": 2015, "confidence": 0.4},  # 10/10 실제 모델 출력 — 근거 없는 추측
        "artist_type": {"values": [], "confidence": 0.0},
        "performance_clues": {"vocal_count": None, "vocal_roles": [], "sound_ensemble": [], "confidence": 0.0},
    }
    raw.update(over)
    return raw


def test_safeguards_drop_guessed_era_and_leaked_tags_when_life_stage_present():
    out = qa._apply_metadata_safeguards("중학교 때 많이 듣던 남자 발라드", _model_raw())
    assert out["life_stage"]["stage"] == "middle"
    assert out["release_era"] == {"start_year": None, "end_year": None, "confidence": 0.0}
    # '추억'(일반 회상어)·'어린시절'(생애 단계 질의마다 모델이 다는 회상어)은 빠지고, '수학여행'은 잡힌 단계(중학교)의 말이
    # 아니라 남는다 — 장면 단서일 수 있다
    assert out["korean_tags"] == ["발라드", "남성보컬", "수학여행"]


def test_absolute_era_wins_over_life_stage():
    out = qa._apply_metadata_safeguards("2000년대 중학교 때 듣던 댄스곡", _model_raw())
    assert out["life_stage"]["stage"] == "middle"
    assert (out["release_era"]["start_year"], out["release_era"]["end_year"]) == (2000, 2009)


def test_without_life_stage_tags_and_model_era_are_kept():
    """'어린 시절' 이야기를 담은 가사를 찾는 질의에서는 태그도 시기도 건드리지 않는다."""
    out = qa._apply_metadata_safeguards("어린 시절 이야기를 담은 잔잔한 발라드", _model_raw(release_era={"start_year": 2010, "end_year": 2019, "confidence": 0.5}))
    assert out["life_stage"]["stage"] is None
    assert "어린시절" in out["korean_tags"]
    assert out["release_era"]["start_year"] == 2010


def test_tags_naming_song_facts_survive_even_if_they_contain_a_life_stage_word():
    """'대학가요제'는 곡 정보다 — '대학' 부분 일치로 지우면 안 된다(리뷰)."""
    out = qa._apply_metadata_safeguards("중학교 때 듣던 대학가요제 대상 받은 노래",
                                        _model_raw(korean_tags=["대학가요제", "대상", "중학교", "추억", "학창시절", "수학여행송"]))
    assert out["life_stage"]["stage"] == "middle"
    assert out["korean_tags"] == ["대학가요제", "대상", "수학여행송"]


def test_every_life_stage_phrase_outside_quotes_is_removed_from_search_text():
    """대표는 '고3 때'지만 '야자 끝나고'도 시간 표현이라 검색문에서 빠진다(리뷰)."""
    analysis = qa._fallback("고3 때 야자 끝나고 듣던 노래")
    assert analysis.life_stage.text == "고3 때" and len(analysis.life_stage.spans) == 2
    assert analysis.search_text == "듣던 노래"


def test_quoted_lyric_with_the_same_phrase_is_kept():
    """따옴표 안의 '중학교 때'는 가사다 — 위치로 지우므로 남는다(리뷰)."""
    q = '중학교 때 듣던 노래인데 가사에 "중학교 때 널 만났지"가 나와'
    analysis = qa._fallback(q)
    assert analysis.life_stage.stage == "middle" and analysis.life_stage.spans == [[0, 5]]
    assert analysis.search_text == '듣던 노래인데 가사에 "중학교 때 널 만났지"가 나와'
    # 따옴표 안에만 있으면 시간 단서가 아니다
    assert qa._extract_life_stage('가사에 "중학교 때 널 만났지"가 나오는 노래') is None


def test_context_words_next_to_the_phrase_stay_in_search_text():
    """'버스에서 틀었던'은 장면 단서다 — 시간 표현만 빼고 조사가 매달린 채 남지 않게 한다."""
    analysis = qa._fallback("고등학교 수학여행 버스에서 틀었던 신나는 노래, 남자 그룹")
    assert analysis.life_stage.stage == "high"
    assert analysis.search_text == "버스에서 틀었던 신나는 노래, 남자 그룹"


def test_school_name_with_danil_ttae_is_removed_whole():
    a = qa._fallback("초등학교 다닐 때 듣던 발라드")
    assert (a.life_stage.age_from, a.life_stage.age_to) == (7, 12)
    assert a.search_text == "듣던 발라드"


def test_detached_phrase_without_a_listening_verb_is_lyric_content_and_stays():
    """대표('중학교 때 듣던')가 잡혀도 떨어져 있는 '어릴 때 집이 어려웠다는'은 가사 내용 — 검색문·태그 모두 남긴다(리뷰)."""
    q = "중학교 때 듣던 노래인데 어릴 때 집이 어려웠다는 가사가 나와"
    a = qa._fallback(q)
    assert a.life_stage.stage == "middle" and a.life_stage.spans == [[0, 5]]
    assert a.search_text == "듣던 노래인데 어릴 때 집이 어려웠다는 가사가 나와"
    out = qa._apply_metadata_safeguards(q, _model_raw(korean_tags=["발라드", "어린시절", "중학교", "추억", "가난"]))
    assert out["korean_tags"] == ["발라드", "어린시절", "가난"]


@pytest.mark.parametrize("query", [
    '중학교 때 듣던 노래인데 "어릴 때 집이 어려웠다"는 가사가 나와',   # 인용 안의 표현도 내용 단계
    "어릴 때 듣던 노래인데 어린 시절 집이 어려웠다는 가사가 나와",      # 청취 단계와 내용 단계가 같아도 내용 태그는 남긴다
])
def test_content_tags_survive_when_the_stage_is_also_quoted_or_equals_the_listening_stage(query):
    out = qa._apply_metadata_safeguards(query, _model_raw(korean_tags=["발라드", "어린시절", "추억", "가난"]))
    assert out["life_stage"]["stage"] in ("middle", "childhood")
    assert out["korean_tags"] == ["발라드", "어린시절", "가난"]


_LONG_LYRIC = ("어릴 때 우리 집 앞마당에 피어 있던 꽃들이 하나둘 시들어 가고 엄마는 매일 밤 라디오를 틀어 놓고 "
               "조용히 울었지 나는 그 소리를 들으며 잠이 들곤 했어")


@pytest.mark.parametrize("open_q, close_q", [('"', '"'), ("“", "”"), ("'", "'"), ("「", "」")])
def test_long_quoted_lyric_is_protected_whatever_its_length(open_q, close_q):
    """인용은 짝이 맞는 닫는 따옴표까지다 — 80자 제한 때문에 109자 인용 안의 '어릴 때'가 지워졌다(리뷰)."""
    q = f"어릴 때 듣던 노래인데 가사가 {open_q}{_LONG_LYRIC}{close_q}로 시작해"
    assert len(_LONG_LYRIC) > 80
    a = qa._fallback(q)
    assert a.life_stage.stage == "childhood" and a.life_stage.spans == [[0, 4]]
    assert a.search_text == f"듣던 노래인데 가사가 {open_q}{_LONG_LYRIC}{close_q}로 시작해"


def test_tags_written_in_the_query_as_content_are_kept():
    """'중학교 때 듣던 추억을 회상하는 가사' — 추억·회상은 질의에 내용으로 적혀 있다. 모델이 시간 단서에서 지어낸 회상어만 뺀다(리뷰)."""
    tags = ["추억", "회상", "발라드", "중학교", "어린시절"]
    out = qa._apply_metadata_safeguards("중학교 때 듣던 추억을 회상하는 가사의 노래", _model_raw(korean_tags=tags))
    assert out["life_stage"]["stage"] == "middle"
    assert out["korean_tags"] == ["추억", "회상", "발라드"]
    out = qa._apply_metadata_safeguards("중학교 때 많이 듣던 남자 발라드", _model_raw(korean_tags=tags))
    assert out["korean_tags"] == ["발라드"]


def test_adjacent_phrase_is_still_removed_with_the_representative():
    a = qa._fallback("고3 때 야자 끝나고 듣던 노래")
    assert len(a.life_stage.spans) == 2 and a.search_text == "듣던 노래"


def test_search_text_strips_the_phrase_but_original_query_stays():
    analysis = qa._fallback("중학교 때 많이 듣던 남자 발라드인데 제목이 기억 안 나")
    assert analysis.has_life_stage
    assert analysis.original_query.startswith("중학교 때")
    assert analysis.search_text == "많이 듣던 남자 발라드인데 제목이 기억 안 나"
    plain = qa._fallback("비 오는 날 듣기 좋은 재즈")
    assert not plain.has_life_stage and plain.search_text == plain.original_query


def test_default_life_stage_clue_is_empty():
    a = qa._fallback("비 오는 날 듣기 좋은 재즈")
    assert a.life_stage == LifeStageClue()
    assert not a.has_life_stage and a.search_text == a.original_query
    assert "life_stage" in QueryAnalysis.model_fields  # 캐시·API 응답에 실린다


@pytest.mark.parametrize("life, birth, expected", [
    ({"age_from": 13, "age_to": 15, "confidence": 0.6}, 2001, (2013, 2017, 0.6)),   # 01년생 중학교
    ({"age_from": 13, "age_to": 15, "confidence": 0.6}, 1990, (2002, 2006, 0.6)),   # 90년생 중학교
    ({"age_from": 4, "age_to": 12, "confidence": 0.3}, 2001, (2004, 2014, 0.3)),    # 어릴 때는 넓고 약하게
])
def test_release_era_from_birth_year(life, birth, expected):
    era = qa.release_era_from_birth_year(life, birth)
    assert (era["start_year"], era["end_year"], era["confidence"]) == expected


def test_release_era_from_birth_year_rejects_missing_or_absurd_input():
    assert qa.release_era_from_birth_year({"stage": None}, 2001) is None
    assert qa.release_era_from_birth_year({"age_from": 13, "age_to": 15, "confidence": 0.6}, 1800) is None


# ---------------------------------------------------------------------------
# 9차 리뷰(0ddc833): 확실한 청취 시기만 분리하고 가사 내용은 건드리지 않는다. 감지/비감지를 짝으로, 최종 검색 입력까지 확인한다.
# ---------------------------------------------------------------------------
def _assert_untouched(query, tags=("어린시절", "추억", "발라드")):
    """비감지 질의는 원문·내용 태그가 그대로다."""
    a = qa._fallback(query)
    assert not a.has_life_stage and a.search_text == query
    out = qa._apply_metadata_safeguards(query, _model_raw(korean_tags=list(tags)))
    assert out["life_stage"]["stage"] is None and out["korean_tags"] == list(tags)


@pytest.mark.parametrize("query", [
    # 1. 곡 명사 + 격조사는 뒤 서술어의 논항이다 — 계사(인데·였어)와 다르다
    "어릴 때 음악이 유일한 친구였다는 가사",
    "어릴 때 노래가 꿈이었다는 이야기",
    # 3. 어간 접두 + 비관형형이 곧 청취는 아니다(뜨겁게·듣기·불러도)
    "어릴 때 뜨겁게 사랑했던 사람 이야기하는 가사",
    "어릴 때 듣기 싫던 잔소리 얘기 가사",
    "어릴 때 불러도 대답 없던 이름 이야기",
    # 5. 동사가 꾸미는 명사구의 머리(사람을·놀이를·이름을)에서 멈춘다 — 절을 넘어 뒤의 '노래'를 가져오지 않는다
    "중학교 때 좋아하던 사람을 그리워하는 노래",
    "어릴 때 유행했던 놀이를 소개하는 노래",
    "어릴 때 부르던 내 이름을 그리워하는 노래",
])
def test_review9_content_is_not_a_listening_memory(query):
    _assert_untouched(query)


@pytest.mark.parametrize("query, stage, search_text", [
    ("중학교 때 노래인데 가사가 슬퍼", "middle", "노래인데 가사가 슬퍼"),              # 계사는 그대로 감지
    ("고등학교 때 그 노래", "high", "그 노래"),
    ("중학교 때 좋아하던 사람이 부르던 노래", "middle", "좋아하던 사람이 부르던 노래"),   # 뒤의 독립된 '부르던 노래'
    ("중학교 때 많이 들었는데 제목이 기억 안 나", "middle", "많이 들었는데 제목이 기억 안 나"),
    ("중학교 때 많이 들었던 것 같아", "middle", "많이 들었던 것 같아"),
    ("어릴 때 엄마가 불러주던 노래", "childhood", "엄마가 불러주던 노래"),
    ("어릴 때 듣고 자란 노래", "childhood", "듣고 자란 노래"),                          # 연결형도 꾸미는 머리를 본다
])
def test_review9_positive_controls_reach_search_text(query, stage, search_text):
    a = qa._fallback(query)
    assert a.life_stage.stage == stage and a.search_text == search_text and a.original_query == query
    assert QueryAnalysis.model_validate_json(a.model_dump_json()).search_text == search_text


@pytest.mark.parametrize("open_q, close_q", [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")])
def test_review9_words_inside_quotes_are_not_evidence_for_the_outside_phrase(open_q, close_q):
    query = f"어릴 때 엄마가 {open_q}듣고 있니{close_q}라고 묻는 가사"
    assert qa._quoted_ranges(query)
    _assert_untouched(query)


def test_review9_same_stage_inside_and_outside_quotes():
    q = '어릴 때 듣던 노래인데 "어릴 때 그 골목"이라는 가사가 나와'
    a = qa._fallback(q)
    assert a.life_stage.stage == "childhood" and a.life_stage.spans == [[0, 4]]
    assert a.search_text == '듣던 노래인데 "어릴 때 그 골목"이라는 가사가 나와'
    out = qa._apply_metadata_safeguards(q, _model_raw(korean_tags=["어린시절", "추억", "골목"]))
    assert out["korean_tags"] == ["어린시절", "골목"]           # 인용 안의 같은 단계는 내용 → 어린시절 보존, 추억은 모델 회상어


@pytest.mark.parametrize("number, expected", [
    ("열다섯", 15), ("열 다섯", 15), ("열일곱", 17), ("열아홉", 19), ("쉰다섯", 55), ("쉰 다섯", 55),
    ("다섯", 5), ("일곱", 7), ("스물 다섯", 25), ("서른 다섯", 35), ("5", 5), ("7", 7), ("9", 9), ("15", 15), ("59", 59),
])
def test_review9_numerals_are_read_whole(number, expected):
    """'열다섯'을 5살로 읽고 '열'을 검색문에 남기면 안 된다 — 전체를 읽거나 전체를 미지원."""
    query = f"{number} 살 때 듣던 노래" if not number.isdigit() else f"{number}살 때 듣던 노래"
    life = qa._extract_life_stage(query)
    assert life is not None and life["age_from"] == life["age_to"] == expected
    assert life["text"] == (f"{number} 살 때" if not number.isdigit() else f"{number}살 때")
    assert qa._fallback(query).search_text == "듣던 노래"


@pytest.mark.parametrize("query", ["예순다섯 살 때 듣던 노래", "예순 다섯 살 때 듣던 노래", "60살 때 듣던 노래", "네 살 때 듣던 노래"])
def test_review9_ages_outside_support_are_not_truncated_to_the_unit(query):
    _assert_untouched(query)


@pytest.mark.parametrize("query, tags, expected", [
    ("중학교 때 듣던 추억의 노래", ["추억", "발라드"], ["발라드"]),            # 곡을 꾸미는 시간 수식어
    ("고등학교 때 듣던 옛날 노래", ["옛날", "발라드"], ["발라드"]),
    ("중학교 때 듣던 추억을 회상하는 가사의 노래", ["추억", "회상", "발라드"], ["추억", "회상", "발라드"]),   # 내용 근거(가사) 있음
    ("중학교 때 듣던 노래인데 가사는 고향에 대한 향수를 담았어", ["향수", "고향", "발라드"], ["향수", "고향", "발라드"]),
])
def test_review9_temporal_modifiers_are_stripped_but_described_content_is_kept(query, tags, expected):
    out = qa._apply_metadata_safeguards(query, _model_raw(korean_tags=tags))
    assert out["life_stage"]["stage"] is not None
    assert out["korean_tags"] == expected


# ---------------------------------------------------------------------------
# 10차 리뷰: 9차에서 생긴 회귀(격조사 머리에서 멈춤·창 끝 동사), 절 경계, 'X가 나오는', 보조 용언, 태그 근거 엄격화
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, stage, search_text", [
    # 2-① 뜻이 하나뿐인 청취 동사(듣·들었·들으) 뒤는 격조사 머리에서 멈추지 않고 곡 명사까지 간다
    ("중학교 때 듣던 제목이 기억 안 나는 노래", "middle", "듣던 제목이 기억 안 나는 노래"),
    ("중학교 때 듣던 여자가 부른 노래", "middle", "듣던 여자가 부른 노래"),
    ("중학교 때 듣던 뮤비가 유명한 노래", "middle", "듣던 뮤비가 유명한 노래"),
    ("고등학교 때 유행했던 춤이 유명한 노래", "high", "유행했던 춤이 유명한 노래"),    # 뒤의 독립된 '유명한 노래'
    ("중학교 때 많이 듣던 가사가 슬픈 노래", "middle", "많이 듣던 가사가 슬픈 노래"),
    # 2-② 동사가 질의의 마지막 어절
    ("그 노래 중학교 때 많이 들었던", "middle", "그 노래 많이 들었던"),
    ("이 노래 중학교 때 엄청 유행했던", "middle", "이 노래 엄청 유행했던"),
    ("중학교 때 많이 듣던", "middle", "많이 듣던"),
    # 3 놓치던 것 — 곡 명사가 내용 표지보다 먼저, 보조 용언 '하다', 연결 어미 '지만'
    ("어릴 때 듣던 만화 주제가", "childhood", "듣던 만화 주제가"),
    ("어릴 때 따라 부르던 주제곡", "childhood", "따라 부르던 주제곡"),
    ("어릴 때 자주 듣곤 했어", "childhood", "자주 듣곤 했어"),
    ("어릴 때 많이 들었지만 지금은 안 들어", "childhood", "많이 들었지만 지금은 안 들어"),
    ("어릴 때 유행가 따라 부르던 거", "childhood", "유행가 따라 부르던 거"),
    ("어릴 때 라디오에서 나오던 노래", "childhood", "라디오에서 나오던 노래"),            # 'X에서 나오던'은 청취
])
def test_review10_detected(query, stage, search_text):
    a = qa._fallback(query)
    assert a.life_stage.stage == stage and a.search_text == search_text


@pytest.mark.parametrize("query", [
    # 절 경계 — 앞 절이 가사 내용으로 끝났으면 뒤 절의 동사는 근거가 아니다
    "어릴 때 집이 가난했다는 가사인데 최근에 나온 노래야",
    "중학교 때 좋아하던 친구 얘긴데 요즘 좋아하는 노래",
    "어릴 때 집이 가난했는데 요즘 나온 노래",
    # 'X가 나오는 노래' — 나오는 것은 X다
    "어릴 때 부르던 내 이름이 나오는 노래",
    "어릴 때 유행했던 놀이가 나오는 노래",
    "어릴 때 듣던 라디오가 나오는 노래",
    # 종결 어미처럼 보이지만 부정·보조 용언·연결
    "어릴 때 부르지 못한 이름",
    "어릴 때 듣지 못했던 말",
    "어릴 때 틀어 놓던 만화",
    "어릴 때 잔소리 듣다 지쳐",
    "어릴 때 유행어 따라 하던 친구 얘기",
    "어릴 때 뜬금없이 울던 얘기 가사",
    "어릴 때 듣기 평가 망친 얘기",
    "어릴 때 방에 틀어박혀 울던 가사",
    # 머리 명사가 아닌 '생각'·'노래 한다'
    "어릴 때 틀어주던 만화 생각이 난다는 가사",
    "중학교 때 노래 한다고 설치던 친구 얘기 가사",
    "중학교 때 많이 들었거든 근데 제목을 모르겠어",   # 10차 재확인 — '-거든'은 종결
])
def test_review10_not_detected(query):
    if query.endswith("제목을 모르겠어"):
        a = qa._fallback(query)
        assert a.life_stage.stage == "middle" and a.search_text == "많이 들었거든 근데 제목을 모르겠어"
        return
    _assert_untouched(query)


@pytest.mark.parametrize("query, tags, expected", [
    ("중학교 때 듣던 추억의 노래 가사가 기억 안 나", ["추억", "발라드"], ["발라드"]),     # 3어절 안에 '가사가'가 있어도 추억은 노래를 꾸민다
    ("중학교 때 듣던 추억의 노래 그리고 발라드", ["추억", "발라드"], ["발라드"]),        # '그리고'는 그리워하다가 아니다
    ("고등학교 때 듣던 옛날 노래인데 담임 선생님이 틀어줬어", ["옛날", "발라드"], ["발라드"]),
    ("고등학교 때 듣던 옛날 노래 라디오에 나오는", ["옛날", "발라드"], ["발라드"]),
    ("중학교 때 듣던 노래인데 추억이 담긴 가사", ["추억", "발라드"], ["추억", "발라드"]),
    ("중학교 때 듣던 노래인데 옛날 얘기 나오는 가사", ["옛날", "발라드"], ["옛날", "발라드"]),
    ("중학교 때 듣던 노래인데 고향에 대한 향수를 그리워하는 가사", ["향수", "고향", "발라드"], ["향수", "고향", "발라드"]),
])
def test_review10_tag_evidence_requires_the_word_to_be_the_content_argument(query, tags, expected):
    out = qa._apply_metadata_safeguards(query, _model_raw(korean_tags=tags))
    assert out["life_stage"]["stage"] is not None and out["korean_tags"] == expected
