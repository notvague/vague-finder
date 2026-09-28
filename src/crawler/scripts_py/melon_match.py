"""
[Step 1-a] Melon 검색 결과 후보 판정

설명: 멜론 통합검색 결과 중 "시드가 가리키는 원곡"을 고르기 위한 점수 로직.
      네트워크를 타지 않는 순수 함수만 두어 테스트로 고정할 수 있게 한다.

배경: 기존 fetch_melon_song_id는 제목 부분일치 + 아티스트 부분일치를 통과한
      **첫 행을 즉시 반환**했다. 그래서 멜론 검색 순위가 바뀌면 같은 질의가 다른
      음원을 가져왔다.

          2026-03-02  BLACKPINK Kill This Love -> 31717822 (앨범 KILL THIS LOVE)
          2026-07-29  BLACKPINK Kill This Love -> 32591630 (일본판 도쿄돔 라이브)

      일본판은 가사가 일본어 로마자이고 멜론 장르가 J-POP인데, 유튜브는 시드 제목으로
      따로 검색해 한국어 원곡 M/V를 받아왔다. 한 레코드 안에서 텍스트 임베딩과 오디오
      임베딩이 서로 다른 녹음을 가리키게 된다.

      듀엣판도 같은 부류다. 이승철 'My Love (Duet Ver.)'가 들어오면서 vocal_gender가
      '혼성'이 됐는데, 원곡은 남성 솔로 발라드다. vocal_gender는 재질문이 물을 수 있는
      두 슬롯 중 하나이고 회수율이 가장 높은 슬롯이라(v07 개입 9건 중 7건) 조용히
      오염되면 타격이 크다.

작성자: 황찬혁 (Full)
생성일: 2026-09-17
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# 채택 금지. 이 표시가 붙은 후보는 순위와 무관하게 버린다.
HARD = "hard"
# 순위 감점만. 더 나은 후보가 없으면 채택하되 검토 대상으로 남긴다.
SOFT = "soft"

# 제목이 완전히 같을 때 주는 점수.
EXACT_TITLE_BONUS = 100
# 시드 제목 **뒤에 괄호 부제만** 붙은 경우. '작은 것들을 위한 시 (Boy With Luv)
# (Feat. Halsey)'처럼 멜론이 원제에 부제를 달아 둔 정상 곡이 여기 해당한다.
SUBTITLE_BONUS = 60
# 시드 제목 **앞에** 다른 말이 붙은 경우. '좋은날'을 찾았는데 '사랑하기 좋은날'이
# 오는 식이라 다른 곡일 확률이 높다.
PREFIX_MISMATCH_PENALTY = -40
# 이 점수에 못 미치면 채택하되 검토 목록에 남긴다.
REVIEW_THRESHOLD = SUBTITLE_BONUS

# 앨범명에서 찾은 표시는 절반만 반영한다. 앨범은 곡 자체의 속성이 아니라서
# 'Live Op.4 Concert Project'처럼 원래 라이브 음반이 유일한 음원인 경우가 있다.
ALBUM_WEIGHT = 0.5

# (패턴, 점수, 라벨, 강도)
# 패턴은 soften()을 거친 문자열(소문자 + 기호가 공백으로 바뀐 형태)에 적용한다.
VERSION_MARKERS: Sequence[Tuple[str, int, str, str]] = (
    # --- 채택 금지 ---
    (r"\b(japan|japanese)\b", -200, "일본어판", HARD),
    (r"\b(chinese|mandarin|cantonese)\b", -200, "중국어판", HARD),
    (r"\benglish\s*ver", -200, "영어판", HARD),
    (r"\bjp\s*ver", -200, "일본어판", HARD),
    (r"(일본어|중국어|영어)\s*(버전|ver)", -200, "외국어판", HARD),
    # MR은 여기서 보지 않는다. 괄호를 떼어낸 문자열에서 찾으면 경칭 'Mr.'와 구분할 수
    # 없다 — 소녀시대 'Mr.Mr.'는 반주로 잡히고, '밤편지 (MR) [가사]'는 놓친다.
    # is_instrumental()이 괄호·대괄호 구조를 보존한 채 판별한다.
    (r"\b(inst|instrumental)\b|반주", -200, "반주/MR", HARD),
    # --- 순위 감점 ---
    (r"\b(duet)\b|듀엣", -150, "듀엣판", SOFT),
    (r"\b(live|tour|concert|unplugged)\b|실황", -100, "라이브판", SOFT),
    (
        r"\b(remix|mix|acoustic|orchestra|rock\s*ver|ballad\s*ver)\b|어쿠스틱|리믹스",
        -80,
        "편곡판",
        SOFT,
    ),
    # --- 가산 ---
    # 한국어판 표기는 이 프로젝트에서 오히려 찾는 물건이다.
    # 'To My Love (Korean Ver.)'(윤미래), '으르렁 (EXO-K Ver.)'가 여기 해당한다.
    # EXO는 K가 한국어판, M이 중국어판이다.
    (r"\bkorean\s*ver|한국어\s*(버전|ver)|\bexo\s*k\b", 50, "한국어판", SOFT),
    (r"\boriginal\s*ver", 10, "원곡표기", SOFT),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), delta, label, strength)
    for pattern, delta, label, strength in VERSION_MARKERS
)

# 시드가 뭘 요청했든 이 프로젝트에서는 받지 않는다. 앨범에서 발견해도 마찬가지다.
ALWAYS_REJECT_LABELS = frozenset({"일본어판", "중국어판", "영어판", "외국어판", "반주/MR"})

# 녹음 자체가 달라지는 표시. **시드와 후보가 같아야** 한다.
#
# 한쪽만 보고 일괄 제외하면 원래 듀엣곡(시드가 듀엣판을 요청한 경우)까지 놓치고,
# 감점만 하면 원곡 후보가 없을 때 듀엣판이 그대로 저장된다. 그래서 대칭 비교한다.
# 지금 시드 1,000건에서 버전 표기가 붙은 곡은 모두 시드에도 같은 표기가 있다.
#     비,태양을 피하는 방법 (Gtr.Remix) / 렉시,하늘위로 (Remix) / 조규만,다 줄거야 (Acoustic Ver.)
VERSION_LABELS = frozenset({"듀엣판", "라이브판", "편곡판"})

# 한국어판·원곡표기는 우리가 찾는 물건이라 대칭 비교에서 뺀다.
# 시드 'To My Love'에 후보 'To My Love (Korean Ver.)'가 오는 것은 정상이다.

# 멜론이 국내 가수에게 붙이지 않는 장르. 붙어 있으면 다른 나라 음원이다.
# 경고만 하면 제목이 깨끗한 외국어판이 그대로 저장되므로 제외 조건으로 둔다.
# 제외해도 조용히 사라지지 않는다 — 후보가 전부 탈락하면 failed_songs.csv에 남는다.
FOREIGN_GENRES = frozenset({"J-POP", "POP", "월드뮤직"})


def normalize(text: object) -> str:
    """식별용 정규화. 기호와 공백을 모두 지워 순수 문자만 남긴다."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9가-힣]", "", str(text).lower())


def soften(text: object) -> str:
    """표시 탐지용 정규화. 기호를 공백으로 바꿔 단어 경계를 남긴다.

    normalize와 달리 단어 경계가 살아 있어야 `\\blive\\b`가 'ALIVE'나
    'The Livelong Day'에 걸리지 않는다.
    """
    if not text:
        return ""
    lowered = re.sub(r"[^0-9a-z가-힣]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", lowered).strip()


def soften_keep_parens(text: object) -> str:
    """표시 탐지용 정규화에 괄호만 남긴 형태.

    부제는 괄호로 들어오고("작은 것들을 위한 시 (Boy With Luv)"), 다른 곡은 맨
    단어로 늘어난다("Love" -> "Love Wins"). 이 둘을 가르려면 괄호가 필요하다.
    """
    if not text:
        return ""
    lowered = str(text).replace("（", "(").replace("）", ")").lower()
    lowered = re.sub(r"[^0-9a-z가-힣()]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def artist_matches(seed_artist: str, row_artists: Sequence[str]) -> bool:
    """아티스트가 한 명이라도 맞으면 통과. 듀엣/피처링 표기 차이를 흡수한다."""
    target = normalize(seed_artist)
    if not target:
        return False
    return any(
        target in normalize(name) or (normalize(name) and normalize(name) in target)
        for name in row_artists
    )


def title_extra(seed_title: str, row_title: str) -> str:
    """멜론 제목에서 시드 제목을 뺀 나머지.

    시드에 없던 꼬리표만 검사해야 원곡 제목의 일부인 부제에 감점하지 않는다.
    코퍼스 961곡의 제목 괄호 269개를 확인해 보면 대부분이 '초련(初戀)',
    'Sorry, Sorry'처럼 정상적인 부제다.
    """
    soft_seed = soften(seed_title)
    soft_row = soften(row_title)
    if soft_seed and soft_seed in soft_row:
        return soft_row.replace(soft_seed, " ").strip()
    return soft_row


# 괄호와 대괄호를 모두 한 덩어리로 본다. 유튜브 제목은 '[가사]', '[Official]'처럼
# 대괄호를 쓴다.
_CHUNK = re.compile(r"[\(（\[][^)）\]]*[\)）\]]")
_CREDIT_PAREN = re.compile(r"[\(（\[]\s*(feat|prod|narr|with)[^)）\]]*[\)）\]]", re.IGNORECASE)

# 괄호 안이 통째로 반주 표기일 때만 반주로 본다.
_INST_CHUNK = re.compile(
    r"^(inst\.?|instrumental|mr\.?|반주)(\s*(ver\.?|version|버전))?$", re.IGNORECASE
)


def chunks(title: object) -> List[str]:
    """제목에서 괄호·대괄호 덩어리의 속을 꺼낸다."""
    return [chunk.strip("()（）[]").strip() for chunk in _CHUNK.findall(str(title or ""))]


def is_instrumental(title: object) -> bool:
    """반주/MR 음원인지 판별한다.

    괄호 구조를 보존한 채 본다. 괄호를 떼어낸 문자열에서 'mr'을 찾으면
    소녀시대 'Mr.Mr.'가 반주로 잡히고, 반대로 '밤편지 (MR) [가사]'는 MR 뒤에
    '가사'가 이어져 놓친다.
    """
    return any(_INST_CHUNK.match(inner) for inner in chunks(title))


def labels_in(text: object) -> frozenset:
    """문자열에서 걸리는 표시 라벨을 모은다. **정규화 전의 원문**을 넘길 것."""
    if not text:
        return frozenset()

    softened = soften(text)
    labels = {label for pattern, _, label, _ in _COMPILED if pattern.search(softened)}
    if is_instrumental(text):
        labels.add("반주/MR")
    return frozenset(labels)


def _marker_labels(text: str) -> frozenset:
    """이미 soften된 문자열에서 표시 라벨을 모은다(괄호 정보 없음)."""
    if not text:
        return frozenset()
    return frozenset(
        label for pattern, _, label, _ in _COMPILED if pattern.search(text)
    )


def _scan(text: str, weight: float = 1.0) -> Tuple[int, List[str], bool]:
    """표시를 찾아 점수·사유·채택금지 여부를 돌려준다.

    앨범에서 찾은 표시는 점수를 절반만 반영하지만(ALBUM_WEIGHT), **채택 금지는
    그대로 적용한다.** 앨범이 'Song (English Version)'이라고 명시하는데 감점으로만
    낮추면 제목이 깨끗한 외국어판이 통과한다.
    """
    if not text:
        return 0, [], False

    score = 0
    reasons: List[str] = []
    hard = False
    seen: set = set()

    for pattern, delta, label, strength in _COMPILED:
        if not pattern.search(text):
            continue
        if label in seen:
            continue
        seen.add(label)

        applied = int(delta * weight)
        score += applied
        reasons.append(f"{label} {applied:+d}")
        if strength == HARD:
            hard = True

    return score, reasons, hard


@dataclass
class MatchScore:
    score: int
    reasons: List[str] = field(default_factory=list)
    hard_reject: bool = False

    @property
    def needs_review(self) -> bool:
        return not self.hard_reject and self.score < REVIEW_THRESHOLD

    def describe(self) -> str:
        return f"{self.score:+d} ({', '.join(self.reasons) or '표시 없음'})"


def score_candidate(
    seed_title: str,
    row_title: str,
    row_album: str = "",
) -> MatchScore:
    """검색 결과 한 행의 점수를 매긴다. 아티스트 일치는 호출부에서 먼저 거른다."""
    score = 0
    reasons: List[str] = []

    exact = normalize(seed_title) == normalize(row_title)
    if exact:
        score += EXACT_TITLE_BONUS
        reasons.append(f"제목 완전일치 +{EXACT_TITLE_BONUS}")

    extra_score, extra_reasons, hard = _scan(title_extra(seed_title, row_title))
    score += extra_score
    reasons.extend(extra_reasons)

    shape_hard = False
    if not exact:
        score, reasons, shape_hard = _apply_shape_bonus(
            seed_title, row_title, score, reasons, has_marker=bool(extra_reasons)
        )
    hard = hard or shape_hard

    album_score, album_reasons, album_hard = _scan(soften(row_album), weight=ALBUM_WEIGHT)
    score += album_score
    reasons.extend(f"앨범 {r}" for r in album_reasons)
    hard = hard or album_hard

    # 항상 제외할 표시는 **후보 제목 전체**에서 본다. 시드를 뺀 나머지만 보면
    # 시드와 후보가 둘 다 'Song (English Ver.)'일 때 검사를 통과해 버린다.
    # 이 프로젝트는 국내 곡만 다루므로 시드가 요청했더라도 받지 않는다.
    banned = (labels_in(row_title) | labels_in(row_album)) & ALWAYS_REJECT_LABELS
    if banned:
        hard = True
        if not any(label in r for r in reasons for label in banned):
            reasons.append(f"국외/반주 음원({', '.join(sorted(banned))})")

    mismatch = version_mismatch(seed_title, row_title)
    if mismatch:
        hard = True
        reasons.append(mismatch)

    return MatchScore(score=score, reasons=reasons, hard_reject=hard)


# 비교 대상 라벨. 한국어판·원곡표기는 표기 유무가 녹음을 바꾸지 않으므로 뺀다.
CHECKED_LABELS = VERSION_LABELS | ALWAYS_REJECT_LABELS


def version_tags(title: str) -> frozenset:
    """제목에서 '다른 녹음'을 뜻하는 표시 라벨만 뽑는다."""
    return frozenset(version_details(title))


# 같은 뜻인데 다르게 적히는 표시. 이것만 정규화한다.
TOKEN_SYNONYMS = {
    "듀엣": "duet", "어쿠스틱": "acoustic", "리믹스": "remix", "실황": "live",
    "반주": "mr", "일본어": "japan", "중국어": "chinese", "영어": "english",
}

# 버전 표기의 뼈대. 실제 참여자 이름과 섞이지 않게 떼어낸다.
# 'Duet with 선미'와 'Duet 선미'는 같은 녹음이다.
_BOILERPLATE = re.compile(
    r"\b(with|ver|version|버전|and|feat|ft|편|of)\b|[&,]", re.IGNORECASE
)


def _canonical_token(token: str) -> str:
    normalized = normalize(token)
    return TOKEN_SYNONYMS.get(normalized, normalized)


def _detail_of(segment: str) -> Tuple[Dict[str, set], str]:
    """한 덩어리에서 (라벨 -> 표시 토큰) 과 남은 문자열을 뽑는다.

    표시 토큰은 '어떤 종류의 버전인가'(remix / acoustic / duet)이고,
    남은 문자열은 '누가·무엇이 다른가'(선미 / gtr)다. 둘을 따로 비교해야
    'Duet with 선미'와 'Duet 선미'를 같게, 'Duet with 다른가수'를 다르게 볼 수 있다.
    """
    softened = soften(segment)
    tokens: Dict[str, set] = {}
    leftover = softened

    for pattern, _, label, _ in _COMPILED:
        for match in pattern.finditer(softened):
            hit = match.group(0)
            tokens.setdefault(label, set()).add(_canonical_token(hit))
            leftover = leftover.replace(hit, " ")

    return tokens, normalize(_BOILERPLATE.sub(" ", leftover))


def version_details(title: str) -> Dict[str, Dict[str, set]]:
    """라벨별로 표시 토큰과 참여자를 남긴다.

    라벨만 비교하면 서로 다른 녹음이 같은 것으로 판정된다.

        When We Disco (Duet with 선미)  ↔  When We Disco (Duet with 다른가수)
        Song (Acoustic Ver.)           ↔  Song (Remix)

    괄호 밖에 적힌 표시도 토큰은 모은다. 빈 집합으로 두면 비교를 통째로 우회해서
    'Song (Acoustic Ver.)'와 '가수 Song Remix'가 같은 버전이 된다. 다만 참여자는
    담지 않는다 — 괄호 밖은 영상 제목의 부가 문구('MV', 'Official')와 섞인다.
    """
    details: Dict[str, Dict[str, set]] = {}

    def slot(label: str) -> Dict[str, set]:
        return details.setdefault(label, {"tokens": set(), "who": set()})

    for inner in chunks(title):
        if _CREDIT_PAREN.match(f"({inner})"):
            continue
        tokens, who = _detail_of(inner)
        for label, found in tokens.items():
            if label not in CHECKED_LABELS:
                continue
            slot(label)["tokens"] |= found
            if who:
                slot(label)["who"].add(who)

    outside_tokens, _ = _detail_of(_CHUNK.sub(" ", str(title or "")))
    for label, found in outside_tokens.items():
        if label in CHECKED_LABELS:
            slot(label)["tokens"] |= found

    # 반주/MR은 괄호 구조로 판별한다(경칭 'Mr.'와 구분하기 위해).
    if is_instrumental(title):
        slot("반주/MR")["tokens"].add("mr")

    return details


def _sets_agree(left: set, right: set, substring_ok: bool = False) -> bool:
    """한쪽을 알 수 없으면 판단을 보류한다."""
    if not left or not right:
        return True
    if left & right:
        return True
    if substring_ok:
        return any(a in b or b in a for a in left for b in right)
    return False


def version_mismatch(seed_title: str, row_title: str) -> str:
    """시드가 요청한 버전과 후보의 버전이 다르면 사유를 돌려준다.

    감점만 하면 원곡 후보가 없을 때 듀엣판이 그대로 저장되고, 반대로 듀엣을 일괄
    제외하면 시드가 듀엣판을 요청한 곡을 놓친다. 그래서 양쪽을 맞춘다.
    """
    return _compare_versions(seed_title, row_title, "요청", "후보")


def _compare_versions(left_title: str, right_title: str, left_name: str, right_name: str) -> str:
    left, right = version_details(left_title), version_details(right_title)

    only_left = set(left) - set(right)
    only_right = set(right) - set(left)
    if only_left or only_right:
        return (
            f"버전 불일치 ({left_name} {', '.join(sorted(only_left)) or '없음'}"
            f" / {right_name} {', '.join(sorted(only_right)) or '없음'})"
        )

    for label in left:
        # 1) 종류가 다른가 — acoustic vs remix
        lt, rt = left[label]["tokens"], right[label]["tokens"]
        if not _sets_agree(lt, rt, substring_ok=True):
            return (
                f"{label} 종류 불일치 ({left_name} {', '.join(sorted(lt))}"
                f" / {right_name} {', '.join(sorted(rt))})"
            )

        # 2) 상대가 다른가 — duet with 선미 vs duet with 다른가수
        lw, rw = left[label]["who"], right[label]["who"]
        if not _sets_agree(lw, rw, substring_ok=True):
            return (
                f"{label} 대상 불일치 ({left_name} {', '.join(sorted(lw))}"
                f" / {right_name} {', '.join(sorted(rw))})"
            )
    return ""


# 괄호 밖에 맨 단어로 남은 버전 표시의 꼬리말. 'exo k ver'에서 'exo k'를 지우면 'ver'가 남는다.
_VERSION_WORD = re.compile(r"\b(ver|vers|version)\b|버전", re.IGNORECASE)


def _split_around_seed(seed: str, row: str) -> Optional[Tuple[str, str]]:
    """공백 차이를 무시하고 후보 제목에서 시드를 찾아 (앞, 뒤)로 자른다. 없으면 None.

    앞단 필터(normalize)는 공백을 모두 지우고 비교하므로 여기서도 공백 차이를 무시해야 한다.
    그러지 않으면 시드 '작은것들을위한시'와 멜론 '작은 것들을 위한 시 (Boy With Luv)'가
    어긋나 정상 곡이 제외된다. 괄호는 남겨야 부제를 알아볼 수 있어 공백만 무시한다.
    """
    letters = [(index, char) for index, char in enumerate(row) if char != " "]
    target = seed.replace(" ", "")
    if not target:
        return None

    for start in range(len(letters) - len(target) + 1):
        if all(letters[start + offset][1] == target[offset] for offset in range(len(target))):
            head = letters[start][0]
            tail = letters[start + len(target) - 1][0] + 1
            return row[:head].strip(), row[tail:].strip()
    return None


def strip_version_markers(text: str) -> str:
    """인식한 버전 표시와 그 꼬리말(ver)을 떼어 낸다. 곡 이름만 비교하려는 자리에 쓴다."""
    out = str(text or "")
    for pattern, _delta, _label, _strength in _COMPILED:
        out = pattern.sub(" ", out)
    return _VERSION_WORD.sub(" ", out)


def _bare_residue(text: str) -> str:
    """괄호 덩어리와 인식된 버전 표시를 걷어낸 나머지.

    버전 표시가 하나 있다고 제목 검사를 통째로 건너뛰면 'Love Wins (Original Ver.)'처럼
    다른 곡이 통과한다. 표시를 지우고도 남는 맨 글자('Wins', '2')는 다른 곡이라는 뜻이다.
    """
    text = strip_version_markers(_CHUNK.sub(" ", text))
    return re.sub(r"[\s()]+", " ", text).strip()


def _apply_shape_bonus(
    seed_title: str,
    row_title: str,
    score: int,
    reasons: List[str],
    has_marker: bool,
) -> Tuple[int, List[str], bool]:
    """제목이 어떻게 늘어났는지로 가감한다. (점수, 사유, 채택금지)

    괄호 부제가 붙은 것은 같은 곡이다. 그런데 **괄호 없이 맨 단어가 더 붙으면 대개 다른
    곡이다.** 버전 표시(라이브·리믹스 등)가 걸린 경우는 여기서 판단하지 않고 버전 로직에 맡긴다.

    실제로 세 곡이 이렇게 잘못 수집됐다. 모두 시드가 가리킨 곡과 발매 연도·작곡가·분위기가
    다른 별개의 곡이다.
        유리상자 '좋은날'(2002) -> '사랑하기 좋은날'(2023)
        윤현석   'Love'(2000)  -> 'Love Wins'(2016)
        송하예   '니 소식'(2019) -> '니 소식2'(2022, 후속곡)
    코퍼스 961곡을 시드와 대조하면 928곡이 완전일치이고 이 형태는 저 3곡뿐이라, 막아도
    정상 수집을 잃지 않는다. 시드 제목을 줄여 적어 이 규칙에 걸리면 곡을 못 받는 대신
    failed_songs.csv에 사유가 남으므로 사람이 시드를 고칠 수 있다.
    """
    seed = soften_keep_parens(seed_title)
    row = soften_keep_parens(row_title)
    if not seed or not row:
        return score, reasons, False

    split = _split_around_seed(seed, row)
    if split is None:
        # 시드가 후보 제목 안에 없다. 후보가 더 짧은 경우가 대부분이라 판단을 보류한다.
        return score, reasons, False

    head, tail = split
    if not head and not tail:
        return score, reasons, False          # 공백 표기만 다르다

    # 괄호 덩어리와 버전 표시를 걷어내고 남는 맨 글자가 있으면 다른 곡이다.
    residue = _bare_residue(f"{head} {tail}")
    if residue:
        if head:
            score += PREFIX_MISMATCH_PENALTY
            reasons.append(f"제목 앞에 다른 말 '{residue}' {PREFIX_MISMATCH_PENALTY:+d} (다른 곡)")
        else:
            reasons.append(f"제목 뒤에 다른 말 '{residue}' (다른 곡)")
        return score, reasons, True

    if head:
        # 남는 맨 글자는 없지만 시드 앞에 괄호·버전 표시가 붙어 있다. 부제로 보지 않는다.
        score += PREFIX_MISMATCH_PENALTY
        reasons.append(f"제목 앞에 다른 말 {PREFIX_MISMATCH_PENALTY:+d}")
        return score, reasons, False

    # 버전 표시가 걸렸으면 부제 가산은 하지 않는다. 판단은 버전 로직에 맡긴다.
    if not has_marker and tail.startswith("("):
        score += SUBTITLE_BONUS
        reasons.append(f"괄호 부제만 붙음 +{SUBTITLE_BONUS}")
    return score, reasons, False


def check_details(genres: object) -> Tuple[bool, str]:
    """상세 페이지에서만 볼 수 있는 장르로 2차 판정.

    검색 결과 페이지에는 장르 열이 없어서 이 검사는 상세 페이지를 받은 뒤에 한다.
    """
    if isinstance(genres, str):
        values = [g.strip() for g in genres.split(",") if g.strip()]
    else:
        values = [str(g).strip() for g in (genres or []) if str(g).strip()]

    for value in values:
        key = value.upper() if re.search(r"[a-zA-Z]", value) else value
        if key in FOREIGN_GENRES:
            return False, f"국외 장르({value})"

    return True, ""


def title_core(title: str) -> str:
    """제목에서 버전 정보와 크레딧을 떼고 곡 이름만 남긴다.

    '으르렁 (Growl)(EXO-K Ver.)' -> '으르렁 (Growl)'
    'Kill This Love (Japan Version / ... -TOKYO DOME-)' -> 'Kill This Love'

    부제 괄호는 곡 이름의 일부라 남긴다. 표시가 들어 있는 괄호만 떼어낸다.
    """
    def drop(match: "re.Match") -> str:
        chunk = match.group(0)
        if _CREDIT_PAREN.match(chunk) or labels_in(chunk):
            return " "
        return chunk

    return _CHUNK.sub(drop, str(title or ""))


# 단어 안의 아포스트로피. 제목을 감싼 따옴표와 구분해야 한다.
# 무조건 나누면 "I'm Fine"과 "I'm Sorry"가 'i'를, "Don't Cry"와 "Don't Say Goodbye"가
# 'don'을 공통 조각으로 갖게 되어 다른 곡이 같은 제목으로 판정된다.
_INNER_APOSTROPHE = re.compile(r"(?<=[0-9A-Za-z가-힣])['’‘](?=[0-9A-Za-z가-힣])")

# 구분자로 쪼개 나온 조각이 이보다 짧으면 보통 곡 이름이 아니다('M/V' -> 'm', 'v').
_MIN_SEGMENT_LENGTH = 2


def _is_boilerplate_only(segment: str) -> bool:
    """조각이 부가 표현뿐인가. 'M', 'V', 'MV'가 여기 걸린다."""
    return not normalize(_VIDEO_BOILERPLATE.sub(" ", segment))


def _useful_segments(text: str, keep: frozenset = frozenset()) -> List[str]:
    """곡 이름 자리가 될 수 있는 조각만.

    한 글자 조각은 기본적으로 버린다 — 'M/V'를 쪼갠 'm', 'v'가 아무 제목에나 맞아떨어진다.
    다만 **찾는 제목과 같은** 한 글자 조각은 살린다. 한 글자 제목이 실제로 있다
    ('god - 길', "TAEYEON 태연 'I' MV"). 부가 표현뿐인 조각은 그때도 버린다.
    """
    whole = normalize(text)
    useful = []
    for segment in _segments(text):
        compact = normalize(segment)
        if not compact:
            continue
        if len(compact) >= _MIN_SEGMENT_LENGTH or compact == whole:
            useful.append(segment)
        elif compact in keep and not _is_boilerplate_only(segment):
            useful.append(segment)
    return useful


# 영상 제목을 곡 이름 자리로 나누는 구분자.
# 괄호·대괄호·따옴표·세로줄·슬래시는 항상 나눈다. 하이픈과 언더바는 **공백에 둘러싸였을 때만**
# 나눈다 — 'EXO-K', 'U-Go-Girl'처럼 제목 안의 하이픈을 살리려는 것이다.
# 쉼표는 나누지 않는다. '그대가, 그대를...'처럼 곡 이름 안에 쉼표가 있는 경우가 있다.
_TITLE_SPLIT = re.compile(
    r"""[()\[\]（）【】「」『』|ㅣ/~"'“”‘’]+"""
    r"|\s[-–—_]+\s"
    r"|^[-–—_\s]+|[-–—_\s]+$"
)

# 영상 제목에 붙는 부가 표현. 곡 이름 자리에서 떼어 낸다.
_VIDEO_BOILERPLATE = re.compile(
    r"\b(official|officially|mv|m|v|music|video|audio|lyrics?|visualizer|teaser|trailer|"
    r"performance|practice|special|clip|full|ver|version|hd|hq|4k|1080p|720p|color|coded|"
    r"eng|sub|subtitle|karaoke|inst|instrumental)\b"
    r"|뮤직비디오|뮤비|가사|자막|공식|영상|음원|티저|퍼포먼스|안무|무대|풀버전|자체제작",
    re.IGNORECASE,
)


# 앨범 수록 순서 표기. '04. 너랑 나'처럼 곡 이름 앞에 붙는다.
_TRACK_NUMBER = re.compile(r"^\s*\d{1,2}\s*[.)\-]\s*")
# 한글·숫자·공백만으로 이어지는 앞뒤 덩어리. '주지마 Don't'처럼 한국어 제목 옆에 영어 제목이
# 구분자 없이 붙는 표기를 위해 쓴다. 숫자를 남기므로 '니 소식2'는 그대로 '니 소식2'다.
_HANGUL_RUN_HEAD = re.compile(r"^[가-힣0-9\s]+")
_HANGUL_RUN_TAIL = re.compile(r"[가-힣0-9\s]+$")


def _segments(text: str) -> List[str]:
    """제목을 곡 이름 자리 후보로 나눈다. 단어 안의 아포스트로피는 나누지 않는다."""
    cleaned = _INNER_APOSTROPHE.sub("", str(text or ""))
    return [part.strip() for part in _TITLE_SPLIT.split(cleaned) if part and part.strip()]


def _strip_names(text: str, names: Sequence[str]) -> str:
    """가수 이름을 떼어 낸다. 두 글자 미만은 곡 이름을 망가뜨릴 수 있어 건드리지 않는다."""
    out = text
    for name in names:
        for piece in [name] + _segments(name):
            piece = piece.strip()
            if len(normalize(piece)) < 2:
                continue
            out = re.sub(re.escape(piece), " ", out, flags=re.IGNORECASE)
    return out


def song_title_parts(title: str) -> frozenset:
    """곡 이름과 그 조각들. 부제 괄호와 구분자로 나눈 부분을 모두 담는다.

    '으르렁 (Growl)' -> {'으르렁growl', '으르렁', 'growl'}
    """
    core = title_core(title)
    parts = {normalize(core)} | {normalize(segment) for segment in _useful_segments(core)}
    return frozenset(part for part in parts if part)


def video_title_parts(video_title: str, known_artists: Sequence[str] = (),
                      keep: frozenset = frozenset()) -> frozenset:
    """영상 제목에서 곡 이름 자리 후보를 모은다.

    영상 제목에는 가수명·[MV]·Official 같은 부가 표현과 영어 제목이 함께 들어간다
    ('[MV] IU(아이유) _ Through the Night(밤편지)'). 구분자로 나눈 각 조각에 대해
    원본·가수명 제거·부가 표현 제거를 모두 후보로 담는다. 넉넉하게 담아야 정상 영상을
    잃지 않는다 — 걸러내는 힘은 '부분일치가 아니라 일치'를 요구하는 데서 나온다.
    """
    parts = set()
    for segment in _useful_segments(video_title, keep):
        without_names = _strip_names(segment, known_artists) if known_artists else segment
        variants = {segment, without_names}
        for base in (segment, without_names):
            plain = _VIDEO_BOILERPLATE.sub(" ", base)
            variants.add(plain)
            # 버전 표시가 괄호 없이 맨 단어로 붙기도 한다('하늘위로 Remix MV').
            # 이름 비교 자리에서는 떼어 낸다. 버전이 맞는지는 _compare_versions가 따로 본다.
            variants.add(strip_version_markers(plain))
            variants.add(_TRACK_NUMBER.sub("", plain))
            # 한국어 제목 옆에 영어 제목이 구분자 없이 붙는 표기('주지마 Don't')
            for run in (_HANGUL_RUN_HEAD.search(plain), _HANGUL_RUN_TAIL.search(plain)):
                if run:
                    variants.add(run.group(0))
        parts.update(normalize(variant) for variant in variants)
    return frozenset(part for part in parts if part)


def title_conflict(melon_title: str, video_title: str,
                   known_artists: Sequence[str] = ()) -> str:
    """멜론이 고른 곡과 유튜브가 고른 영상이 다른 녹음이면 사유를 돌려준다.

    두 소스를 각각 따로 검색하기 때문에 한 레코드가 서로 다른 녹음을 가리킬 수 있다.
    32591630이 그랬다 — 멜론은 일본판 라이브, 유튜브는 한국어 원곡 M/V.

    곡 이름과 버전 정보를 나눠서 본다. 제목 전체의 포함 관계로 보면 같은 한국어판인
    '으르렁 (Growl)(EXO-K Ver.)'와 '으르렁 (Growl) MV (Korean Ver.)'가 표기 차이만으로
    불일치가 된다. 멜론 후보에서는 한국어판을 우대해 놓고 영상에서 떨어뜨리는 셈이다.
    """
    if not melon_title or not video_title:
        return ""  # 판단 근거가 없으면 경고하지 않는다

    # 1) 곡 이름이 영상 제목의 '곡 이름 자리'에 없는 경우.
    #    부분 문자열로 보면 '좋은날'이 '사랑하기 좋은날'에, '니 소식'이 '니 소식2'에,
    #    'Love'가 'Love Wins'에 들어 있어 다른 곡의 오디오가 붙는다.
    song = song_title_parts(melon_title)
    if song and not (song & video_title_parts(video_title, known_artists, keep=song)):
        return "곡 이름이 영상 제목에 없음"

    # 2) 버전이 어긋나는 경우. 양방향으로, 세부까지 본다 — 한쪽만 보면 멜론 'My Love'와
    #    유튜브 'My Love (Duet Ver.)'가 일치로 판정되고, 라벨만 보면 듀엣 상대가 다른
    #    두 녹음이 같은 것으로 판정된다.
    #    한국어판·원곡표기는 비교 대상이 아니다(표기 유무가 녹음을 바꾸지 않는다).
    return _compare_versions(melon_title, video_title, "멜론", "영상")


def strip_requested_versions(text: str, seed_title: str) -> str:
    """시드가 요청한 버전 표시를 텍스트에서 뺀다.

    '하늘위로 (Remix)'를 요청했으면 영상 제목의 'Remix'는 제외 사유가 아니다. 요청하지 않은
    편곡·커버만 막아야 한다. 코퍼스에도 요청 자체가 리믹스판인 곡이 있다(1621412, 490059).
    """
    out = str(text or "")
    soft_seed = soften(seed_title)
    for pattern, _delta, _label, _strength in _COMPILED:
        if pattern.search(soft_seed):
            out = pattern.sub(" ", out)
    return out


def remove_song_title(video_title: str, melon_title: str) -> str:
    """영상 제목에서 곡 이름 부분을 뺀 나머지.

    키워드 필터가 곡 이름 안의 단어에 걸리지 않게 한다 — 'Piano Man', '시간'처럼
    제외 키워드와 같은 말이 곡 이름인 경우가 있다.
    """
    core = title_core(melon_title)
    split = _split_around_seed(soften_keep_parens(core), soften_keep_parens(video_title))
    if split is None:
        return video_title
    head, tail = split
    return f"{head} {tail}".strip()


def source_titles_agree(melon_title: str, video_title: str,
                        known_artists: Sequence[str] = ()) -> bool:
    """title_conflict의 불리언 래퍼."""
    return not title_conflict(melon_title, video_title, known_artists)
