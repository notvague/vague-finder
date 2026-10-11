"""재질문 전용 — 최초 질의에서 사용자가 **이미 말한** 데이터 슬롯(성별·장르) 판정.

`clarify.pick_question`이 "이미 말한 것은 다시 묻지 않는다"를 적용할 때 쓰는 근거다. 검색에는 쓰지 않는다 —
`QueryAnalysis.vocal_gender`·`genre`(검색용 값)와 분석 프롬프트는 이 모듈과 무관하고, 여기서 찾은 값이 검색
필터나 가짜 답변이 되지도 않는다. 돌려주는 것은 슬롯 이름뿐이다.

**원문이 증거다.** 분석 필드가 채워져 있다는 것만으로 확정하지 않는다(모델 추측일 수 있다). 반대로 분석이
규칙 폴백이라 성별·장르가 비어 있어도 원문에 `남자 아이돌`·`인디`가 있으면 말한 것이다 — 사용자 예시 둘 다
폴백 분석에서는 두 칸이 비어 있었다.

판정 규칙(모두 부분 문자열이 아니라 문맥이다):

- 성별은 **가창자**를 가리키는 표현만 — `남자 아이돌`, `여자 보컬`, `남자 인디 노래`, `남자가 부른`, `남녀 듀엣`,
  `남자 둘 여자 하나`, `보컬이 남자야`. 영상·표지 문장(`뮤비에서 남자가 노래 부르는 장면`)의 인물은 가창자가
  아니다 — 그 절에서는 "~가 부른" 같은 행위 문형을 보지 않고, 가수·보컬·아이돌처럼 가창자 명사가 붙은 표현과
  곡 수식("남자 발라드")만 인정한다. 남자친구·여자친구는 가창자가 아니다.
- 장르는 현재 재질문 선택지(코퍼스 장르)에 대응하는 표현 — 발라드·댄스·힙합·록/락/메탈·R&B/알앤비·인디·포크·
  트로트·재즈·일렉. 단독 `랩`은 장르 진술이 아니라 랩 파트 얘기인 경우가 전부라(평가 질의 17/17) `랩 곡`처럼
  곡을 수식할 때만 본다. `인디`는 세부 장르를 하나로 확정하는 값은 아니지만, 음악의 성격을 이미 말한 것이므로
  지금의 포괄적인 "어떤 장르에 가까웠나요?"를 되풀이하지 않는 근거로 본다(계획서 §1).
  악기·장소·행위(`일렉 기타`, `재즈 카페`, `락 페스티벌`, `댄스 브레이크`)와 제목·가사 언급(`제목에 록이 들어가는`)은
  장르 진술이 아니다.
- 따옴표 안(가사·제목 인용)과 이름으로 지정된 구절(`아티스트명 남자답게`, `남자답게라는 가수`)은 보지 않는다.
  단 `장르는 '인디'야`처럼 장르를 가리키는 따옴표는 값이다.
- 단서 **바로 뒤**의 불확실·부정 표현은 확정이 아니다 — `남자 가수인지 모르겠어`, `남자 보컬일 수도 있어`,
  `발라드인지 록인지 모르겠어`, `여자 가수는 아니야`, `랩 없이`. 부정만으로 반대쪽을 확정하지도 않는다.
  멀리 떨어진 "모르겠어"는 다른 것(제목·곡이 뭔지)에 붙은 말이다 — `남자 아이돌 노래 뭔지 모르겠어`는 성별 확정.
- 문맥 단위로 본다 — `남자 보컬이고 장르는 모르겠어`의 장르 불확실성은 성별을 지우지 않는다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import KNOWN_SLOT_ORDER
from src.retrieval.query_analyzer import _quoted_ranges

# 재질문이 "이미 말했다"로 뺄 수 있는 슬롯 — 요청·응답 스키마와 같은 한 곳의 값. clarify.ALLOWED_SLOTS와 같아야 한다
# (tests/test_known_slots.py가 고정한다).
KNOWN_SLOT_NAMES: Sequence[str] = KNOWN_SLOT_ORDER

# 절 경계 — 문장 부호, 접속어, 연결 어미. 불확실·부정 표현은 같은 절 안에서만 그 단서에 붙는다.
_CLAUSE_BREAK_RE = re.compile(
    r"[,;.!?。！？…\n]+|(?:그런데|하지만|근데|그리고|그래서)|"
    r"(?<=[가-힣])(?:인데|한데|이고|있고|였고|는데|은데|지만|라서|어서|해서)(?=\s)|"
    # 연결 어미 '-고'. 부정의 '말고'·'아니고'는 절을 끝내지 않는다 — 그 부정이 앞 단서에 붙어야 한다("여자 가수 말고 남자 가수")
    r"(?<=[가-힣])(?<!말)(?<!아니)고(?=\s)"
)

# 영상·표지 문맥 — 이 절에서 "~가 부르는" 같은 행위의 주체는 가창자가 아닐 수 있다.
_SCENE_RE = re.compile(
    r"(?:뮤비|뮤직\s*비디오|(?<![A-Za-z])MV(?![A-Za-z])|영상|티저|무대|장면|화면|표지|커버(?!\s*(?:한|했|곡|버전|해|하))|자켓|재킷|"
    r"앨범\s*(?:아트|사진|이미지)|사진|이미지|그림|일러스트|썸네일|직캠|공연|춤추|안무)",
    re.IGNORECASE,
)

# 이름으로 지정된 구절 — "아티스트명 남자답게", "제목이 한 남자", "남자답게라는 가수". 그 어절은 단서가 아니다.
# '아티스트' 단독은 마커가 아니다("남자 아티스트 발라드"는 가창자 표현) — 명·이름이 붙어야 한다.
_NAME_AFTER_MARKER_RE = re.compile(
    r"(?:제목|곡명|곡\s*이름|노래\s*제목|노래\s*이름|아티스트(?:명|\s*이름)|가수(?:명|\s*이름)|"
    r"그룹(?:명|\s*이름)|팀(?:명|\s*이름)|밴드(?:명|\s*이름)|이름)(?:이|은|는|:)?\s*(\S+)"
)
_NAME_BEFORE_MARKER_RE = re.compile(
    r"(\S+?)(?:이라는|라는|이란|란|이라고\s*하는|라고\s*하는)\s*(?:제목|곡|노래|가수|아티스트|그룹|팀|밴드|앨범)"
)

# 따옴표 앞에 이런 말이 붙으면 인용이 아니라 값이다 — "장르는 '인디'야".
_VALUE_QUOTE_LEAD_RE = re.compile(r"(?:장르|성별|보컬|목소리)(?:는|은|가|이|도|:)?\s*$")

# 남자친구·여자친구는 가창자가 아니다.
_GENDER_WORD = r"(?P<g>남자|남성|여자|여성|남녀|혼성)(?!\s*친구)"
# 성별과 가창자 명사 사이에 올 수 있는 수식어·인원 — 무제한 어절을 허용하면 "남자 이름이 제목인 노래"가 걸린다.
_MOD = (
    r"(?:(?:엄청|되게|좀|약간|진짜|정말|꽤|완전|아주|너무|많이|제일|가장|그|어떤|한)\s*)?"
    r"(?:(?:유명한|유명했던|인기\s*많던|인기\s*있던|인기있던|신인|옛날|오래된|최신|요즘|국내|해외|외국|한국|일본|"
    r"어린|젊은|나이\s*든|중년|인디|발라드|힙합|랩|댄스|록|락|메탈|알앤비|R&B|트로트|포크|재즈|솔로|아이돌|밴드|그룹|"
    r"래퍼|보컬|가수|아티스트|섞인|분|애|애들|들|둘|셋|넷|혼자|여러\s*명|\d+\s*명|[두세네]\s*명|\d+\s*인조)\s*){0,4}"
)
# 가창자를 가리키는 명사 — 영상·표지 절 안에서도 인정한다.
_PERFORMER_NOUN = r"(?:가수|보컬|아이돌|솔로|래퍼|밴드|듀오|듀엣|그룹|목소리|음색|보이스|버전|보컬리스트|싱어|아티스트|참가자)"
# 곡을 가리키는 명사 — "남자 인디 노래".
_SONG_NOUN = r"(?:노래|노랜|곡|발라드|힙합|댄스곡|록|락)"
# 가창 동사 활용 — 부르는·부른·불렀·불러·부를·불린.
_SING_VERB = r"(?:부르|부른|부를|불렀|불러|불를|불린|노래하|노래했|노래를|노래만|노래\s*위주|가창|랩하|랩했|랩을|랩이랑|랩과|랩도|커버한|커버했|리메이크한|리메이크했)"
# "~가 부른" 사이의 어절. 다른 주제(제목·이름·가사·다른 성별)로 넘어가면 그 동사는 이 성별의 것이 아니다.
_GAP = r"(?:(?!(?:제목|이름|가사|남자|여자|남성|여성))[가-힣A-Za-z0-9]+\s*){0,5}?"

# 강한 문형 — 영상·표지 절에서도 인정
_GENDER_STRONG_RES = [
    re.compile(_GENDER_WORD + r"\s*" + _MOD + _PERFORMER_NOUN),
    re.compile(r"(?P<g>걸\s*그룹|보이\s*그룹|여돌|남돌)"),
    # 서술어 — "보컬이 남자야", "부른 사람이 남자", "목소리가 여자"
    re.compile(r"(?:보컬|가수|목소리|음색|보이스|싱어|래퍼|부른\s*(?:사람|가수|애|분)|노래\s*부른\s*(?:사람|가수))"
               r"(?:이|가|은|는|도)\s*(?:다\s*|전부\s*|둘\s*다\s*)?" + _GENDER_WORD),
    # 남녀 — "남자랑 여자가", "남자하고 여자가 번갈아"
    re.compile(r"(?P<g>(?:남자|남성)\s*(?:랑|와|과|하고|이랑|,)\s*(?:여자|여성)|(?:여자|여성)\s*(?:랑|와|과|하고|이랑|,)\s*(?:남자|남성))"),
]
# 곡 수식 — "남자 발라드", "남자 노래". 영상 낱말이 같은 절에 있어도 곡 얘기다("뮤비가 인상적인 남자 발라드").
_GENDER_SONG_RE = re.compile(_GENDER_WORD + r"\s*" + _MOD + _SONG_NOUN)
# 인원 — "남자 둘 여자 하나", "여자 3명이", "남성 4인조". 영상·표지 **문장**에서는 보지 않는다 — "회색빛 커버였는데 남자 네 명이
# 서 있었어"처럼 표지 얘기가 앞 절에 있고 인원이 뒤 절에 오는 문장이 흔하다. 행위 문형은 절 단위다(c601 "뮤직비디오가 웃겼고 남자가 불렀어").
_GENDER_COUNT_RE = re.compile(_GENDER_WORD + r"\s*(?:둘|셋|넷|혼자|여러\s*명|\d+\s*명|[두세네]\s*명|한\s*명|하나|\d+\s*인조)")
# 행위 — "남자가 부른", "여자 혼자 불렀어". 영상·표지 절에서는 보지 않는다(화면 속 인물).
# 성별 낱말이 가창의 **대상**이면(를·을·에게·한테·위해 — "여자를 위해 부른", "남자에게 불러주는") 주체가 아니다(리뷰).
_OBJECT_PARTICLE = r"(?!\s*(?:를|을|에게|한테|위해|위한|보고|더러|께|에게서|한테서))"
_GENDER_VERB_RE = re.compile(_GENDER_WORD + _OBJECT_PARTICLE + r"(?:가|이|은|는|의|분이|애가|애들이|들이|랑|와|과)?\s*" + _GAP + _SING_VERB)

_GENDER_VALUE = {"남자": "남성", "남성": "남성", "여자": "여성", "여성": "여성", "남녀": "혼성", "혼성": "혼성",
                 "걸그룹": "여성", "보이그룹": "남성", "여돌": "여성", "남돌": "남성"}

# 장르 — 코퍼스 장르(발라드·댄스·랩/힙합·R&B/Soul·록/메탈·인디음악·포크/블루스·일렉트로니카·성인가요/트로트·재즈)에
# 대응하는 사용자 표현. 팝·케이팝은 넣지 않는다(코퍼스 전체가 그것이다). 단독 '랩'은 넣지 않는다 — 랩 파트 얘기다.
_GENRE_WORDS = (
    "발라드", "발라든", "댄스", "힙합", "랩곡", "랩 곡", "랩 음악", "랩 노래", "알앤비", "알엔비", "R&B", "RnB", "소울", "록", "락", "롹",
    "메탈", "인디", "포크", "블루스", "일렉트로니카", "일렉트로닉", "일렉", "EDM", "트로트", "재즈",
)
# 장르 단어 뒤 — 어절 끝·조사·어미·장르 접미만. 다른 낱말로 이어지면(인디아나·재즈바·소울메이트·메탈릭·일렉기타) 아니다.
_GENRE_TAIL = (
    r"(?=$|[^가-힣A-Za-z0-9]|"
    r"이|가|은|는|을|를|의|도|로|으로|만|랑|이랑|에|와|과|야|예요|이에요|입니다|지|네|임|인|요|고|구나|든|이든|"
    r"곡|음악|노래|풍|스러|같|였|이었|인데|인가|인지|쪽|계열|느낌|스타일|장르|밴드|가수|류|식|처럼|라|란|데|"
    r"발라드|힙합|댄스|록|락|메탈|팝|음)"
)
_GENRE_RE = re.compile(
    r"(?<![가-힣A-Za-z0-9])(?P<genre>" + "|".join(re.escape(w) for w in _GENRE_WORDS) + r")" + _GENRE_TAIL,
    re.IGNORECASE,
)
_GENRE_CANON = {"발라든": "발라드", "락": "록", "롹": "록", "알엔비": "알앤비", "rnb": "알앤비", "r&b": "알앤비",
                "일렉트로닉": "일렉트로니카", "일렉": "일렉트로니카", "edm": "일렉트로니카",
                "랩곡": "힙합", "랩 곡": "힙합", "랩 음악": "힙합", "랩 노래": "힙합"}
# 장르 단어 바로 뒤의 이 낱말은 악기·장소·행위다 — "일렉 기타", "재즈 카페", "락 페스티벌", "댄스 브레이크".
_GENRE_NOT_GENRE_RE = re.compile(
    r"^\s*(?:기타|베이스|드럼|피아노|카페|바(?![가-힣])|클럽|페스티벌|페스티발|브레이크|대회|경연|배틀|동아리|학원|수업|"
    r"선생|댄서|타임|파트|무대|추는|추던|췄|를\s*추|가\s*멋|실력)"
)
# 제목·가사 언급 — "제목에 록이 들어가는", "가사에 발라드라는 말이 나오는"(따옴표 없음).
_GENRE_MENTION_LEAD_RE = re.compile(r"(?:제목|가사|노랫말|곡명)(?:에|에서|이|은|는|중에|속에)?\s*$")
_GENRE_MENTION_TAIL_RE = re.compile(r"^\s*(?:이라는|라는|란|이란)\s*(?:말|단어|가사|제목|구절|글자|낱말)")

# 단서 바로 뒤에 오면 확정이 아닌 표현. 부정(아니야·말고·없이)도 여기 — 부정은 반대쪽을 확정하지 않으므로 똑같이 "모른다"다.
_UNCERTAIN_RE = re.compile(
    r"(?:모르|확실(?:하지|치|히는)|헷갈|긴가민가|가물|"
    r"(?:인지|였는지|이었는지|인가|였나|이었나)(?![가-힣])|"
    r"일\s*수도|일지도|였을\s*수도|이었을\s*수도|일\s*지도|같기도|아닐\s*수도|아닐지도|인가\s*싶|"
    r"기억(?:이)?\s*(?:잘\s*)?(?:안|못)|기억(?:이)?\s*(?:잘\s*)?나지|"
    r"아니(?![가-힣])|아니(?:야|고|었|에요|예요|지|라|다|면|ㄴ)|아닌|말고)"
)
# 슬롯별로 더 보는 것. 성별: 비교("남자 록커처럼")·작사가("남자 아이돌이 써줬다")는 가창자가 아니다. 장르: "랩 없이"·"랩 빼고".
_UNCERTAIN_BY_SLOT = {
    "vocal_gender": re.compile(r"(?:처럼|써\s*줬|써\s*준|작사|작곡|프로듀)"),   # "혼성 3인조 만들어서"의 만들은 결성이라 넣지 않는다
    "genre": re.compile(r"(?:없이|없는|없었|빼고|안\s*(?:들어|나오|했|해))"),
}
# 단서 뒤에 이런 말이 나오면 그 뒤의 "모르겠어"는 **그것**에 붙은 말이다 — "남자 아이돌 노래 제목 모르겠어", "… 뭔지 모르겠어".
_OTHER_TOPIC_RE = {
    "vocal_gender": re.compile(r"(?:제목|이름|가사|연도|년도|시기|앨범|장르|누구|누가|가수가|멜로디|뭔지|뭐였는지|뭐인지|무슨|어떤|찾아|(?:노래|곡)(?=\s|$))"),
    "genre": re.compile(r"(?:제목|이름|가사|연도|년도|시기|앨범|성별|누구|누가|가수|멜로디|뭔지|뭐였는지|뭐인지|무슨|어떤|찾아|(?:노래|곡)(?=\s|$))"),
}
# 불확실·부정은 단서에서 이 어절 수 안에 있어야 붙은 말이다.
_UNCERTAIN_WINDOW_TOKENS = 3


def _mask(text: str, spans: List[tuple]) -> str:
    chars = list(text)
    for start, end in spans:
        for k in range(max(start, 0), min(end, len(chars))):
            if not chars[k].isspace():
                chars[k] = " "
    return "".join(chars)


def _evidence_text(query: str) -> str:
    """인용·이름 구절을 공백으로 지운 원문(길이·위치 보존). 장르를 가리키는 따옴표는 남긴다."""
    text = str(query or "")
    spans: List[tuple] = []
    for start, end in _quoted_ranges(text):
        if _VALUE_QUOTE_LEAD_RE.search(text[:start]):
            spans.append((start, start + 1))   # 여는 따옴표만 지운다 — 안의 값은 단서다
            spans.append((end - 1, end))
            continue
        spans.append((start, end))
    masked = _mask(text, spans)
    for rx in (_NAME_AFTER_MARKER_RE, _NAME_BEFORE_MARKER_RE):
        for m in rx.finditer(masked):
            spans.append(m.span(1))
    return _mask(text, spans)


_SENTENCE_BREAK_RE = re.compile(r"[.!?。！？…\n]+")
# 곡 밖의 인물을 말하는 **절** — 인원 문형("여자 둘이 주인공인 드라마", "남자 둘이 싸우는 내용의 가사")은 여기서 보지 않는다(리뷰).
# 문장 단위로 넓히면 "…우는 내용의 R&B 발라드, 남자 둘이 하는 그룹"(cc008)의 앞 절 때문에 뒤 절의 인원까지 잃는다.
_NON_PERFORMER_CLAUSE_RE = re.compile(r"(?:드라마|영화|가사|내용|주인공|이야기|얘기|스토리|사연|등장|애니|소설|웹툰)")


def _scene_sentences(text: str) -> List[tuple]:
    """영상·표지 낱말이 든 문장의 (시작, 끝) — 인원 문형을 보지 않는 범위."""
    out: List[tuple] = []
    pos = 0
    for m in list(_SENTENCE_BREAK_RE.finditer(text)) + [None]:
        end = m.start() if m is not None else len(text)
        if _SCENE_RE.search(text[pos:end]):
            out.append((pos, end))
        if m is None:
            break
        pos = m.end()
    return out


def _clauses(text: str) -> List[tuple]:
    """(시작, 끝) 절 구간. 경계 어미·부호는 어느 절에도 속하지 않는다."""
    out: List[tuple] = []
    pos = 0
    for m in _CLAUSE_BREAK_RE.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def _is_certain(clause: str, match_end: int, slot: str) -> bool:
    """단서 **바로 뒤**(같은 절, 몇 어절 안)에 불확실·부정 표현이 없는가. 다른 주제가 먼저 나오면 그 뒤는 보지 않는다."""
    tail = clause[match_end:]
    other = _OTHER_TOPIC_RE[slot].search(tail)
    if other is not None:
        tail = tail[: other.start()]
    window = " ".join(tail.split()[:_UNCERTAIN_WINDOW_TOKENS])
    return _UNCERTAIN_RE.search(window) is None and _UNCERTAIN_BY_SLOT[slot].search(window) is None


def _gender_value(group: str) -> str:
    compact = "".join(group.split())
    if re.search(r"남", compact) and re.search(r"여", compact):
        return "혼성"   # "남자랑 여자", "남녀"
    for key, value in _GENDER_VALUE.items():
        if compact.startswith(key):
            return value
    return "혼성"


def _is_genre_statement(clause: str, start: int, end: int) -> bool:
    """악기·장소·행위·제목·가사 언급이 아닌가 — "일렉 기타", "재즈 카페", "제목에 록이", "발라드라는 말"."""
    if _GENRE_NOT_GENRE_RE.search(clause[end:]):
        return False
    if _GENRE_MENTION_LEAD_RE.search(clause[:start]) or _GENRE_MENTION_TAIL_RE.search(clause[end:]):
        return False
    return True


def known_slot_values(query: str) -> Dict[str, str]:
    """원문에서 확정된 슬롯 값. 예: `{"vocal_gender": "남성", "genre": "인디"}`. 없으면 빈 dict."""
    text = _evidence_text(query)
    found: Dict[str, str] = {}

    genders: set = set()
    certain_any = False
    scene_sentences = _scene_sentences(text)
    for start, end in _clauses(text):
        clause = text[start:end]
        scene_clause = _SCENE_RE.search(clause) is not None
        scene_sentence = any(s <= start < e for s, e in scene_sentences)
        patterns = [*_GENDER_STRONG_RES, _GENDER_SONG_RE]
        if not scene_sentence and not _NON_PERFORMER_CLAUSE_RE.search(clause):
            patterns.append(_GENDER_COUNT_RE)
        if not scene_clause:
            patterns.append(_GENDER_VERB_RE)
        for rx in patterns:
            for m in rx.finditer(clause):
                if not _is_certain(clause, m.end(), "vocal_gender"):
                    continue
                certain_any = True
                genders.add(_gender_value(m.group("g")))
    if certain_any:
        if genders == {"남성", "여성"} or "혼성" in genders:
            found["vocal_gender"] = "혼성"      # "남자는 랩, 여자는 노래" — 분석기와 같은 해석
        elif len(genders) == 1:
            found["vocal_gender"] = next(iter(genders))

    for start, end in _clauses(text):
        clause = text[start:end]
        scene = _SCENE_RE.search(clause) is not None
        for m in _GENRE_RE.finditer(clause):
            raw = m.group("genre")
            if scene and raw == "댄스" and not re.match(r"\s*(?:곡|음악|노래)", clause[m.end():]):
                continue   # "뮤비에서 댄스가 멋있던" — 춤이지 장르가 아니다
            if not _is_genre_statement(clause, m.start(), m.end()):
                continue
            if not _is_certain(clause, m.end(), "genre"):
                continue
            found.setdefault("genre", _GENRE_CANON.get(raw.lower(), raw))
    return found


def known_slots_from_text(query: str) -> List[str]:
    """원문에서 이미 말한 슬롯 이름. `KNOWN_SLOT_NAMES` 순서."""
    values = known_slot_values(query)
    return [slot for slot in KNOWN_SLOT_NAMES if slot in values]


def known_slots_from_query(analysis: QueryAnalysis) -> List[str]:
    """분석의 **원문**으로 판정한다. 분석의 vocal_gender·genre 값은 보지 않는다 — 모델 추측일 수 있고, 폴백이면 비어 있다."""
    return known_slots_from_text(analysis.original_query)
