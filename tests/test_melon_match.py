"""
tests/test_melon_match.py

멜론 검색 결과에서 "원곡"을 고르는 점수 로직의 회귀 테스트.

배경: 같은 질의가 시점에 따라 다른 음원을 가져왔다.

    2026-03-02  BLACKPINK Kill This Love -> 31717822 (앨범 KILL THIS LOVE)
    2026-07-29  BLACKPINK Kill This Love -> 32591630 (일본판 도쿄돔 라이브, J-POP)

기존 코드는 제목 부분일치를 통과한 첫 행을 즉시 반환해서, 멜론 검색 순위가 바뀌면
결과가 따라 바뀌었다. 아래 ROW 상수는 2026-09-17 실제 검색 결과에서 가져온 것이다.

오탐 방지가 절반이다. 코퍼스 961곡 중 제목에 버전 표기가 붙은 곡은 13곡(1.4%)뿐이고,
제목 괄호 269개는 대부분 '초련(初戀)' 같은 정상 부제다. 부제에 감점하면 안 된다.

실행:
    venv/bin/python -m pytest tests/test_melon_match.py -v
"""
from __future__ import annotations

import pytest

from src.crawler.scripts_py.melon_match import (
    EXACT_TITLE_BONUS,
    SUBTITLE_BONUS,
    artist_matches,
    check_details,
    normalize,
    score_candidate,
    title_extra,
)


def best_of(seed_title, rows):
    """hard_reject를 버리고 점수순 1위를 고른다. 호출부와 같은 규칙."""
    scored = [
        (score_candidate(seed_title, title, album), title)
        for title, album in rows
    ]
    alive = [(s, t) for s, t in scored if not s.hard_reject]
    if not alive:
        return None, scored
    alive.sort(key=lambda x: x[0].score, reverse=True)
    return alive[0][1], scored


# 2026-09-17 'BLACKPINK Kill This Love' 검색 결과 (아티스트가 맞는 행만)
KILL_THIS_LOVE_ROWS = [
    ("Kill This Love (Japan Version / BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME-)",
     "BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME- (Live)"),
    ("Kill This Love", "KILL THIS LOVE"),
    ("Kill This Love", "BLACKPINK 2021 'THE SHOW' LIVE"),
]


def test_japan_version_is_rejected_outright() -> None:
    """일본어판은 순위와 무관하게 채택 금지다."""
    japan = score_candidate("Kill This Love", *KILL_THIS_LOVE_ROWS[0])
    assert japan.hard_reject is True
    assert any("일본어판" in r for r in japan.reasons)


def test_kill_this_love_picks_the_studio_original() -> None:
    """일본판이 검색 1위여도 원곡을 골라야 한다.

    2026-07-29 수집분이 일본판(32591630)으로 들어간 사건의 회귀 테스트다.
    """
    winner, _ = best_of("Kill This Love", KILL_THIS_LOVE_ROWS)
    assert winner == "Kill This Love"


def test_live_album_loses_to_studio_album_on_equal_title() -> None:
    """제목이 똑같이 완전일치면 앨범이 승부를 가른다.

    'THE SHOW LIVE' 행은 제목만 보면 원곡과 동점이라 앨범 열이 꼭 필요하다.
    """
    studio = score_candidate("Kill This Love", *KILL_THIS_LOVE_ROWS[1])
    live = score_candidate("Kill This Love", *KILL_THIS_LOVE_ROWS[2])
    assert studio.score > live.score
    assert live.hard_reject is False  # 라이브는 감점만. 유일한 음원이면 쓸 수 있어야 한다


def test_duet_version_loses_to_original() -> None:
    """듀엣판이 vocal_gender를 '혼성'으로 바꿔 놓은 사건의 회귀 테스트.

    이승철 'My Love (Duet Ver.)'(33040923)이 원곡 대신 수집돼 있었다.
    """
    winner, _ = best_of("My Love", [
        ("My Love (Duet Ver.)", "이승철 35주년 기념 앨범 Special 'My Love'"),
        ("My Love", "The Livelong Day"),
    ])
    assert winner == "My Love"


@pytest.mark.parametrize(
    "seed, title, album",
    [
        # EXO는 K가 한국어판, M이 중국어판이다. K를 골라야 한다.
        ("으르렁 (Growl)", "으르렁 (Growl）(EXO-K Ver.)", "The 1st Album 'XOXO' Repackage"),
        # 윤미래 'To My Love'는 한국어판 표기가 붙은 쪽이 우리가 원하는 음원이다.
        ("To My Love", "To My Love (Korean Ver.)", "To My Love"),
    ],
)
def test_korean_version_marker_is_a_bonus(seed, title, album) -> None:
    """한국어판 표기는 감점이 아니라 가산이다."""
    result = score_candidate(seed, title, album)
    assert result.hard_reject is False
    assert result.score > 0, result.describe()


@pytest.mark.parametrize(
    "seed, title, album",
    [
        ("초련(初戀) (Techno Mix) (Feat. 윤진)", "초련(初戀) (Techno Mix) (Feat. 윤진)", "New World"),
        ("편지", "편지 (Original Ver.)", "Trap"),
        ("Sorry, Sorry", "Sorry, Sorry", "Sorry, Sorry"),
        ("D (half moon)", "D (half moon) (Feat. 개코)", "130 Mood : TRBL"),
        ("작은 것들을 위한 시", "작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)",
         "MAP OF THE SOUL : PERSONA"),
    ],
)
def test_normal_titles_are_never_hard_rejected(seed, title, album) -> None:
    """부제·피처링·제작자 표기는 채택 금지 사유가 아니다."""
    assert score_candidate(seed, title, album).hard_reject is False


@pytest.mark.parametrize(
    "seed, title, album",
    [
        ("Sorry, Sorry", "Sorry, Sorry", "Sorry, Sorry"),
        ("작은 것들을 위한 시", "작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)",
         "MAP OF THE SOUL : PERSONA"),
        ("D (half moon)", "D (half moon) (Feat. 개코)", "130 Mood : TRBL"),
    ],
)
def test_feature_and_subtitle_get_no_penalty(seed, title, album) -> None:
    """피처링과 부제에는 감점이 0이어야 한다."""
    result = score_candidate(seed, title, album)
    penalties = [r for r in result.reasons if "-" in r and "완전일치" not in r]
    assert penalties == [], result.describe()


@pytest.mark.parametrize(
    "album",
    [
        "빅뱅 미니앨범 5집 'ALIVE'",     # ALIVE 안의 live
        "The Livelong Day",              # Livelong 안의 live
        "BIGBANG SPECIAL EDITION 'STILL ALIVE'",
    ],
)
def test_live_pattern_respects_word_boundary(album) -> None:
    """단어 경계 덕분에 ALIVE·Livelong은 라이브로 잡히지 않는다."""
    result = score_candidate("FANTASTIC BABY", "FANTASTIC BABY", album)
    assert not any("라이브" in r for r in result.reasons), result.describe()


def test_live_album_is_detected_when_it_is_a_real_word() -> None:
    """실제 라이브 앨범은 잡혀야 한다."""
    result = score_candidate("앤", "앤", "Live Op.4 Concert Project 4th Movement The Album")
    assert any("라이브" in r for r in result.reasons)
    assert result.hard_reject is False  # 유일한 음원일 수 있으므로 감점만


@pytest.mark.parametrize(
    "title",
    [
        "밤편지 (Inst.)",
        "밤편지 (Instrumental)",
        "밤편지 (MR)",
    ],
)
def test_instrumental_is_rejected(title) -> None:
    """반주 음원은 검색 대상이 아니다."""
    assert score_candidate("밤편지", title).hard_reject is True


def test_title_extra_only_returns_the_tail() -> None:
    assert title_extra("Kill This Love", "Kill This Love (Japan Version)") == "japan version"
    assert title_extra("My Love", "My Love (Duet Ver.)") == "duet ver"
    assert title_extra("밤편지", "밤편지") == ""


def test_artist_match_allows_partial_names() -> None:
    assert artist_matches("악뮤", ["악뮤 (AKMU)"]) is True
    assert artist_matches("아이유", ["아이유", "SUGA"]) is True
    assert artist_matches("BLACKPINK", ["The Lullabeats"]) is False


def test_foreign_genre_is_rejected_at_detail_stage() -> None:
    """검색 페이지에는 장르 열이 없어 상세 페이지에서 한 번 더 본다."""
    ok, reason = check_details("J-POP")
    assert ok is False
    assert "J-POP" in reason

    ok, _ = check_details(["댄스"])
    assert ok is True


def test_pop_genre_is_rejected_too() -> None:
    """POP도 국내 가수에게 붙지 않는다. 경고만 하면 외국어판이 그대로 저장된다.

    제외해도 조용히 사라지지 않는다 — 후보가 전부 탈락하면 failed_songs.csv에 남는다.
    """
    ok, reason = check_details("POP")
    assert ok is False
    assert "POP" in reason


# --- 멜론 ↔ 유튜브 교차검증 -------------------------------------------------
# 멜론(가사·메타)과 유튜브(오디오)를 각각 따로 검색하기 때문에 한 레코드가 서로 다른
# 녹음을 가리킬 수 있다. 32591630이 정확히 그랬다.

from src.crawler.scripts_py.melon_match import source_titles_agree


def test_cross_check_catches_the_japan_version_incident() -> None:
    """멜론은 일본판 라이브, 유튜브는 한국어 원곡 M/V를 잡은 실제 사건."""
    assert source_titles_agree(
        "Kill This Love (Japan Version / BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME-)",
        "BLACKPINK - 'Kill This Love' M/V",
    ) is False


@pytest.mark.parametrize(
    "melon_title, video_title",
    [
        ("Kill This Love", "BLACKPINK - 'Kill This Love' M/V"),
        # 피처링 표기는 떼고 비교하므로 표기 차이로 오탐하지 않는다
        ("작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)",
         "BTS (방탄소년단) '작은 것들을 위한 시 (Boy With Luv) (feat. Halsey)' Official MV"),
        ("밤편지", "아이유(IU) - 밤편지 (Through the Night) [Official Music Video]"),
    ],
)
def test_cross_check_passes_matching_recordings(melon_title, video_title) -> None:
    assert source_titles_agree(melon_title, video_title) is True


def test_cross_check_stays_quiet_without_evidence() -> None:
    """유튜브 제목을 못 받은 경우까지 경고하면 노이즈만 는다."""
    assert source_titles_agree("밤편지", "") is True
    assert source_titles_agree("", "무엇이든") is True


# --- 제목이 어떻게 늘어났는지 -------------------------------------------------
# 뒤에 괄호 부제가 붙은 것은 같은 곡, 앞에 다른 말이 붙은 것은 대개 다른 곡이다.
# 실제 수집분에서 나온 오염 사례로 고정한다.

def test_paren_subtitle_is_treated_as_the_same_song() -> None:
    """멜론이 원제에 부제를 달아 둔 정상 곡은 검토 대상이 아니다."""
    result = score_candidate(
        "작은 것들을 위한 시",
        "작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)",
        "MAP OF THE SOUL : PERSONA",
    )
    assert result.needs_review is False, result.describe()


@pytest.mark.parametrize(
    "seed, title, album",
    [
        # 유리상자 '좋은날'(2002, 5집 Favorite) vs '사랑하기 좋은날'(2023, 여행 테마송)
        ("좋은날", "사랑하기 좋은날", "사랑하기 좋은날"),
        # 윤현석 'Love'(2000, 2집 Will) vs 'Love Wins'(2016, 4집 Struggle 타이틀)
        ("Love", "Love Wins", "Love Wins"),
        # 송하예 '니 소식'(2019) vs 후속곡 '니 소식2'(2022)
        ("니 소식", "니 소식2", "니 소식2"),
        # 시드가 제목 안에 들어 있을 뿐 다른 곡인 경우
        ("Love", "들을 수 없는 독백 (Goodbye My Love)", "들을 수 없는 독백"),
        ("Love", "forever love", "밤하늘 별빛 아래서"),
    ],
)
def test_bare_extra_title_text_means_a_different_song(seed, title, album) -> None:
    """괄호 없이 맨 단어가 더 붙으면 다른 곡이다. 검토 표시가 아니라 채택 금지다.

    2026-09-18까지는 감점만 하고 채택했다. 위 세 곡이 실제로 그렇게 수집됐고, 발매 연도·
    작곡가·분위기가 모두 다른 별개의 곡이었다. 코퍼스 961곡을 시드와 대조하면 928곡이
    완전일치라 이 규칙으로 잃는 정상 수집은 없다.
    """
    result = score_candidate(seed, title, album)
    assert result.hard_reject is True, result.describe()


@pytest.mark.parametrize(
    "seed, title, album",
    [
        ("작은 것들을 위한 시", "작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)", "MAP OF THE SOUL : PERSONA"),
        ("으르렁 (Growl)", "으르렁 (Growl)(EXO-K Ver.)", "XOXO"),
        ("초련", "초련(初戀)", "초련"),
        ("밤편지", "밤편지", "Palette"),
    ],
)
def test_subtitles_and_korean_versions_are_still_accepted(seed, title, album) -> None:
    """괄호 부제·한국어판 표기는 같은 곡이다. 새 규칙이 이걸 막으면 안 된다."""
    result = score_candidate(seed, title, album)
    assert result.hard_reject is False, result.describe()


@pytest.mark.parametrize(
    "seed, title, album",
    [
        # 시드가 공백 없이 적혀 있고 멜론 제목에는 공백과 괄호 부제가 둘 다 있는 경우.
        # 앞단 필터(normalize)는 공백을 지우고 비교하므로 모양 검사도 공백을 무시해야 한다.
        ("작은것들을위한시", "작은 것들을 위한 시 (Boy With Luv) (Feat. Halsey)", "MAP OF THE SOUL : PERSONA"),
        ("어떻게이별까지사랑하겠어", "어떻게 이별까지 사랑하겠어 (Feat. 이수현)", "항해"),
    ],
)
def test_spacing_difference_still_reads_as_a_subtitle(seed, title, album) -> None:
    """공백 표기 차이로 정상 곡을 거절하면 안 된다.

    '거절하지 않는다'만으로는 부족하다. 공백을 무시하지 못하면 시드를 찾지 못해 '판단 보류'로
    빠지는데, 그것도 거절은 아니기 때문이다. 부제로 알아봤는지(가산점)까지 확인한다.
    """
    result = score_candidate(seed, title, album)
    assert result.hard_reject is False, result.describe()
    assert result.score >= SUBTITLE_BONUS, result.describe()
    assert any("괄호 부제" in reason for reason in result.reasons), result.describe()
    assert result.needs_review is False, result.describe()


@pytest.mark.parametrize(
    "seed, title, album",
    [
        ("작은것들을위한시", "작은 것들을 위한 시", "MAP OF THE SOUL : PERSONA"),
        ("으르렁(Growl)", "으르렁 (Growl)", "XOXO"),
    ],
)
def test_spacing_difference_alone_is_an_exact_match(seed, title, album) -> None:
    result = score_candidate(seed, title, album)
    assert result.hard_reject is False, result.describe()
    assert result.score >= EXACT_TITLE_BONUS, result.describe()


@pytest.mark.parametrize(
    "seed, title",
    [
        # 버전 표시가 하나 걸렸다고 제목 검사를 통째로 건너뛰면 다른 곡이 통과한다
        ("Love", "Love Wins (Original Ver.)"),
        ("니 소식", "니 소식2 (Korean Ver.)"),
        ("Love", "Forever Love (Original Ver.)"),
        ("좋은날", "사랑하기 좋은날 (Original Ver.)"),
    ],
)
def test_a_version_marker_does_not_excuse_a_different_title(seed, title) -> None:
    """인식한 버전 표시를 지운 뒤에도 남는 맨 글자는 비교해야 한다.

    'Love Wins (Original Ver.)'는 원곡표기가 붙어 있어도 'Wins'가 남으므로 다른 곡이다.
    """
    result = score_candidate(seed, title, "앨범")
    assert result.hard_reject is True, result.describe()


@pytest.mark.parametrize(
    "seed, title",
    [
        ("으르렁", "으르렁 EXO-K Ver."),
        ("으르렁", "으르렁 Korean Ver."),
        ("밤편지", "밤편지 Original Ver."),
    ],
)
def test_version_markers_without_parentheses_are_left_to_the_version_logic(seed, title) -> None:
    """괄호 없이 붙은 말이라도 버전 표시면 모양 규칙이 판단하지 않는다.

    한국어판·원곡표기는 이 프로젝트가 오히려 찾는 물건이다('으르렁 (EXO-K Ver.)'는 한국어판,
    M이 중국어판). 모양 규칙이 먼저 막으면 이 곡들을 통째로 잃는다.
    """
    result = score_candidate(seed, title, "앨범")
    assert result.hard_reject is False, result.describe()
    assert result.score > 0, result.describe()


def test_live_only_release_is_not_rejected_by_the_shape_rule() -> None:
    """버전 표시가 걸린 경우는 버전 로직이 판단한다. 모양 규칙이 먼저 막지 않는다.

    라이브 음원이 유일한 릴리스인 곡이 있어서(김광석 'Live Op.4 Concert Project'),
    라이브는 감점만 하고 채택하되 검토 목록에 남긴다.
    """
    result = score_candidate("사랑했지만", "사랑했지만", "김광석 Live Op.4 Concert Project")
    assert result.hard_reject is False, result.describe()


def test_original_beats_a_prefixed_lookalike() -> None:
    """원곡이 검색 결과에 함께 있으면 확실히 이겨야 한다."""
    winner, _ = best_of("좋은날", [
        ("사랑하기 좋은날", "사랑하기 좋은날"),
        ("좋은날", "Loveday"),
    ])
    assert winner == "좋은날"


def test_shorter_melon_title_is_not_penalised() -> None:
    """멜론 쪽 제목이 더 짧은 경우는 판단을 보류한다(빅마마 '체념 後(후)' 사례)."""
    result = score_candidate("체념 後(후)", "체념", "Like The Bible")
    assert not any("앞에 다른 말" in r for r in result.reasons), result.describe()


# --- 리뷰 반영: 요청한 버전과 후보의 버전을 대칭 비교 ---------------------------
# 감점만 하면 원곡 후보가 없을 때 듀엣판이 그대로 저장되고, 듀엣을 일괄 제외하면
# 시드가 듀엣판을 요청한 곡을 놓친다.

from src.crawler.scripts_py.melon_match import title_conflict, version_mismatch


def test_duet_is_not_accepted_when_the_original_was_requested() -> None:
    """원곡 후보가 없어도 듀엣판으로 대체하지 않는다."""
    only_duet = score_candidate("My Love", "My Love (Duet Ver.)", "이승철 35주년 기념 앨범")
    assert only_duet.hard_reject is True, only_duet.describe()


def test_solo_is_not_accepted_when_the_duet_was_requested() -> None:
    """반대 방향도 막는다. 시드가 듀엣판을 가리키면 솔로판은 다른 녹음이다."""
    only_solo = score_candidate("My Love (Duet Ver.)", "My Love", "MY LOVE")
    assert only_solo.hard_reject is True, only_solo.describe()


def test_requested_version_is_accepted_when_it_matches() -> None:
    """원래 그 버전을 요청한 곡은 그대로 수집돼야 한다.

    실제 시드에 있는 곡들이다. 버전 표기가 붙은 곡은 모두 시드에도 같은 표기가 있다.
    """
    for seed, title, album in [
        ("태양을 피하는 방법 (Gtr.Remix)", "태양을 피하는 방법 (Gtr.Remix)", "Rain 2"),
        ("하늘위로 (Remix)", "하늘위로 (Remix)", "RUSH"),
        ("다 줄거야 (Acoustic Ver.)", "다 줄거야 (Acoustic Ver.)", "I Will Give You All"),
    ]:
        result = score_candidate(seed, title, album)
        assert result.hard_reject is False, f"{seed}: {result.describe()}"


def test_positive_markers_are_not_a_version_mismatch() -> None:
    """한국어판·원곡표기는 우리가 찾는 물건이라 불일치가 아니다."""
    assert version_mismatch("To My Love", "To My Love (Korean Ver.)") == ""
    assert version_mismatch("편지", "편지 (Original Ver.)") == ""
    assert version_mismatch("으르렁 (Growl)", "으르렁 (Growl）(EXO-K Ver.)") == ""


# --- 리뷰 반영: 교차검증을 양방향으로 -----------------------------------------

def test_cross_check_catches_version_only_on_the_video_side() -> None:
    """포함 관계만 보면 멜론 'My Love'가 유튜브 'My Love (Duet Ver.)'에 들어가 버린다."""
    assert title_conflict("My Love", "이승철 - My Love (Duet Ver.) [Official]") != ""
    assert title_conflict("Kill This Love", "BLACKPINK - Kill This Love (Japan Version)") != ""


def test_cross_check_still_catches_the_melon_side() -> None:
    assert title_conflict(
        "Kill This Love (Japan Version / BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME-)",
        "BLACKPINK - 'Kill This Love' M/V",
    ) != ""


def test_cross_check_allows_matching_versions_on_both_sides() -> None:
    """양쪽에 같은 표기가 있으면 일치다."""
    assert title_conflict("하늘위로 (Remix)", "렉시 - 하늘위로 (Remix) MV") == ""


# --- 리뷰 반영: 앨범에만 있는 외국어판 표시 ------------------------------------

def test_foreign_marker_in_album_alone_is_rejected() -> None:
    """제목이 깨끗해도 앨범이 외국어판이라고 명시하면 받지 않는다."""
    result = score_candidate("Song", "Song", "Song (English Version)")
    assert result.hard_reject is True, result.describe()


def test_live_marker_in_album_stays_a_penalty() -> None:
    """라이브·편곡은 앨범에 있어도 감점에 그친다. 유일한 음원일 수 있다."""
    result = score_candidate("앤", "앤", "Live Op.4 Concert Project 4th Movement The Album")
    assert result.hard_reject is False, result.describe()


# --- 검색 결과 파싱 + 필터링 통합 (네트워크 모킹) -------------------------------
# 점수 로직뿐 아니라 "후보 목록에 실제로 들어가는지"까지 봐야 한다. 채택 금지된
# 후보가 목록에 남으면 원곡이 없을 때 그대로 저장된다.

from src.crawler.scripts_py import collect_melon_data as cmd


def _fake_search_html(rows):
    """멜론 통합검색 결과의 최소 재현. 실제 페이지의 셀렉터 구조를 따른다."""
    body = ""
    for song_id, title, artist, album in rows:
        body += f"""
        <tr>
          <td class="t_left"><div class="wrap pd_none">
            <a class="btn btn_icon_detail" href="javascript:melon.link.goSongDetail('{song_id}');">상세정보</a>
            <div class="ellipsis"><a class="fc_gray" href="#">{title}</a></div>
          </div></td>
          <td class="t_left"><div class="wrap wrapArtistName" id="artistName">
            <div class="ellipsis"><a class="fc_mgray" href="javascript:melon.link.goArtistDetail('1');">{artist}</a></div>
          </div></td>
          <td class="t_left"><div class="wrap"><div class="ellipsis">
            <a class="fc_mgray" href="javascript:searchLog('web_tot','SONG','AL','q','1');">{album}</a>
          </div></div></td>
        </tr>"""
    return f"<html><body><table>{body}</table></body></html>"


@pytest.fixture
def fake_melon(monkeypatch):
    def install(rows):
        class Response:
            status_code = 200   # fetch_melon_page는 상태 코드를 본다. 403은 MelonAccessError다.
            text = _fake_search_html(rows)

        monkeypatch.setattr(cmd.requests, "get", lambda *a, **k: Response())
    return install


def test_only_duet_available_yields_no_candidate(fake_melon) -> None:
    """원곡이 없다고 듀엣판으로 대체하면 안 된다. 후보 없음 -> 수집 보류."""
    fake_melon([("1", "My Love (Duet Ver.)", "이승철", "35주년 기념 앨범")])
    assert cmd.fetch_melon_song_candidates("이승철", "My Love") == []


def test_only_solo_available_when_duet_requested(fake_melon) -> None:
    """반대 방향도 같다."""
    fake_melon([("2", "My Love", "이승철", "MY LOVE")])
    assert cmd.fetch_melon_song_candidates("이승철", "My Love (Duet Ver.)") == []


def test_original_is_picked_over_duet(fake_melon) -> None:
    """둘 다 있으면 요청한 쪽을 고른다."""
    fake_melon([
        ("1", "My Love (Duet Ver.)", "이승철", "35주년 기념 앨범"),
        ("2", "My Love", "이승철", "MY LOVE"),
    ])
    picked = cmd.fetch_melon_song_candidates("이승철", "My Love")
    assert [c.song_id for c in picked] == ["2"]


def test_requested_duet_is_collected(fake_melon) -> None:
    """원래 듀엣곡은 놓치지 않아야 한다."""
    fake_melon([
        ("1", "My Love (Duet Ver.)", "이승철", "35주년 기념 앨범"),
        ("2", "My Love", "이승철", "MY LOVE"),
    ])
    picked = cmd.fetch_melon_song_candidates("이승철", "My Love (Duet Ver.)")
    assert [c.song_id for c in picked] == ["1"]


def test_japan_version_never_enters_the_candidate_list(fake_melon) -> None:
    """일본판이 검색 1위여도 후보 목록에서 빠진다."""
    fake_melon([
        ("32591630", "Kill This Love (Japan Version / ... -TOKYO DOME-)", "BLACKPINK",
         "BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME- (Live)"),
        ("31717822", "Kill This Love", "BLACKPINK", "KILL THIS LOVE"),
    ])
    picked = cmd.fetch_melon_song_candidates("BLACKPINK", "Kill This Love")
    assert [c.song_id for c in picked] == ["31717822"]


def test_album_only_foreign_marker_is_excluded(fake_melon) -> None:
    """제목이 깨끗하고 앨범에만 외국어판 표시가 있는 경우."""
    fake_melon([("9", "Song", "가수", "Song (English Version)")])
    assert cmd.fetch_melon_song_candidates("가수", "Song") == []


# --- 리뷰 반영: 곡 이름과 버전 정보를 분리 ------------------------------------
# 제목 전체의 포함 관계로 보면 같은 한국어판이 표기 차이만으로 불일치가 된다.
# 멜론 후보에서는 한국어판을 우대해 놓고 영상에서 떨어뜨리면 앞뒤가 맞지 않는다.

from src.crawler.scripts_py.melon_match import title_core


def test_equivalent_korean_markers_are_not_a_conflict() -> None:
    """EXO-K Ver.와 Korean Ver.는 같은 한국어판이다."""
    assert title_conflict("으르렁 (Growl）(EXO-K Ver.)", "으르렁 (Growl) MV (Korean Ver.)") == ""


@pytest.mark.parametrize(
    "melon_title, video_title",
    [
        ("To My Love (Korean Ver.)", "윤미래 - To My Love"),
        ("편지 (Original Ver.)", "최재훈 - 편지"),
    ],
)
def test_positive_markers_do_not_conflict_with_plain_video(melon_title, video_title) -> None:
    """한국어판·원곡표기는 표기 유무가 녹음을 바꾸지 않는다."""
    assert title_conflict(melon_title, video_title) == ""


def test_title_core_keeps_subtitles_but_drops_versions() -> None:
    """부제 괄호는 곡 이름의 일부라 남기고, 표시가 든 괄호만 뗀다."""
    assert normalize(title_core("으르렁 (Growl）(EXO-K Ver.)")) == normalize("으르렁 (Growl)")
    assert normalize(title_core(
        "Kill This Love (Japan Version / BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME-)"
    )) == normalize("Kill This Love")


def test_version_conflict_is_still_caught_after_core_split() -> None:
    """곡 이름을 분리해도 버전 불일치는 양방향으로 잡혀야 한다."""
    assert title_conflict(
        "Kill This Love (Japan Version / BLACKPINK 2019-2020 WORLD TOUR IN YOUR AREA -TOKYO DOME-)",
        "BLACKPINK - 'Kill This Love' M/V",
    ) != ""
    assert title_conflict("My Love", "이승철 - My Love (Duet Ver.) [Official]") != ""
    assert title_conflict("하늘위로 (Remix)", "렉시 - 하늘위로") != ""


# --- 리뷰 반영: 시드에도 외국어판 표시가 있는 경우 -----------------------------

@pytest.mark.parametrize(
    "seed, title",
    [
        ("Song (English Ver.)", "Song (English Ver.)"),
        ("Song (Japan Version)", "Song (Japan Version)"),
        ("Song (Inst.)", "Song (Inst.)"),
    ],
)
def test_banned_versions_are_rejected_even_when_requested(seed, title) -> None:
    """국내 곡만 다루므로 시드가 외국어판·반주를 가리켜도 받지 않는다.

    시드를 뺀 나머지만 검사하면 시드와 후보가 같을 때 검사를 통과해 버린다.
    """
    result = score_candidate(seed, title, "Album")
    assert result.hard_reject is True, result.describe()


# --- 리뷰 반영: 버전 검사를 자르기 전에 적용 ----------------------------------

from src.crawler.scripts_py import collect_reaction as cr


class _FakeYDL:
    def __init__(self, entries):
        self._entries = entries

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, query, download=False, process=True):
        return {"entries": self._entries}


@pytest.fixture
def fake_youtube(monkeypatch):
    def install(entries):
        monkeypatch.setattr(cr.yt_dlp, "YoutubeDL", lambda *a, **k: _FakeYDL(entries))
    return install


def test_original_below_the_top_three_is_still_found(fake_youtube) -> None:
    """조회수 상위 3개가 듀엣판이고 원곡이 4번째인 경우.

    자른 뒤에 버전을 검사하면 원곡을 아예 보지 못한다.
    """
    fake_youtube([
        {"id": "a", "title": "이승철 - My Love (Duet Ver.)", "channel": "c", "view_count": 1000},
        {"id": "b", "title": "My Love (Duet Ver.) MV", "channel": "c", "view_count": 900},
        {"id": "c", "title": "My Love (Duet Ver.) Official", "channel": "c", "view_count": 800},
        {"id": "d", "title": "이승철 - My Love", "channel": "c", "view_count": 100},
    ])
    picked = cr.fetch_best_video("ytsearch10:이승철 My Love", melon_title="My Love")
    assert [v["id"] for v in picked] == ["d"]


def test_keyword_fallback_does_not_restore_wrong_versions(fake_youtube) -> None:
    """키워드 필터로 전부 빠져도 버전이 어긋난 영상을 되돌리면 안 된다."""
    fake_youtube([
        {"id": "a", "title": "My Love (Duet Ver.) cover", "channel": "c", "view_count": 10},
        {"id": "b", "title": "My Love (Duet Ver.) reaction", "channel": "c", "view_count": 9},
    ])
    assert cr.fetch_best_video("ytsearch10:이승철 My Love", melon_title="My Love") == []


def test_version_filter_is_skipped_without_a_melon_title(fake_youtube) -> None:
    """멜론 제목을 못 받았으면 기존대로 동작한다."""
    fake_youtube([
        {"id": "a", "title": "아무 영상", "channel": "c", "view_count": 5},
    ])
    assert [v["id"] for v in cr.fetch_best_video("ytsearch10:x")] == ["a"]


def test_mr_honorific_is_not_an_instrumental() -> None:
    """후보 제목 전체를 검사하게 되면서 경칭 'Mr.'가 반주로 잡혔다.

    코퍼스 점검에서 애즈원 'Mr. A-Jo'(444798)와 이효리 'Hey Mr. Big'(1903057)이
    통째로 제외되는 것을 발견했다.
    """
    for title in ["Mr. A-Jo", "Hey Mr. Big", "Mr.Chu"]:
        assert score_candidate(title, title, "Album").hard_reject is False, title


def test_standalone_mr_is_still_an_instrumental() -> None:
    for title in ["밤편지 (MR)", "밤편지 (Inst.)", "밤편지 (Instrumental)"]:
        assert score_candidate("밤편지", title).hard_reject is True, title


# --- 리뷰 반영: 반주 판별은 괄호 구조를 보존한 채 -----------------------------
# 괄호를 떼어낸 문자열에서 'mr'을 찾으면 경칭과 구분할 수 없다.

from src.crawler.scripts_py.melon_match import is_instrumental, version_details


@pytest.mark.parametrize(
    "title",
    [
        "Mr.Mr.",            # 소녀시대. 마지막 Mr.이 반주로 잡히던 곡
        "Mr. Chu (On Stage)",  # Apink. 실제 수집 목록에 있다
        "Mr. A-Jo",
        "Hey Mr. Big",
    ],
)
def test_mr_in_the_song_name_is_not_instrumental(title) -> None:
    assert is_instrumental(title) is False, title
    assert score_candidate(title, title, "Album").hard_reject is False, title


@pytest.mark.parametrize(
    "title",
    [
        "밤편지 (MR)",
        "밤편지 (MR) [가사]",       # MR 뒤에 말이 이어져도 잡혀야 한다
        "밤편지 (Inst.)",
        "밤편지 (Instrumental Version)",
    ],
)
def test_bracketed_instrumental_is_detected(title) -> None:
    assert is_instrumental(title) is True, title


def test_instrumental_video_is_rejected_as_a_source() -> None:
    """반주 영상이 유튜브 후보로 통과하면 오디오 임베딩이 반주가 된다."""
    assert title_conflict("밤편지", "아이유 - 밤편지 (MR) [가사]") != ""


# --- 리뷰 반영: 듀엣 상대와 편곡 종류를 보존해 비교 ----------------------------
# 큰 분류(듀엣판/편곡판)만 맞으면 서로 다른 녹음이 같은 것으로 판정된다.

@pytest.mark.parametrize(
    "left, right",
    [
        ("When We Disco (Duet with 선미)", "When We Disco (Duet with 다른가수)"),
        ("Song (Acoustic Ver.)", "Song (Remix)"),
    ],
)
def test_different_details_within_the_same_label_conflict(left, right) -> None:
    assert version_mismatch(left, right) != ""
    assert title_conflict(left, right) != ""


def test_same_detail_agrees() -> None:
    assert version_mismatch("하늘위로 (Remix)", "하늘위로 (Remix)") == ""


def test_detail_comparison_tolerates_equivalent_wording() -> None:
    """포함 관계는 허용한다. 'Remix'와 'Gtr.Remix'는 같은 편곡을 달리 적은 것이다."""
    assert title_conflict("태양을 피하는 방법 (Gtr.Remix)", "비 - 태양을 피하는 방법 (Remix) MV") == ""


def test_bare_marker_still_carries_its_kind() -> None:
    """괄호 밖 표시도 종류는 모은다.

    빈 집합으로 두면 비교를 통째로 우회해서 'Song (Acoustic Ver.)'와
    '가수 Song Remix'가 같은 버전으로 판정된다.
    """
    detail = version_details("하늘위로 Remix")["편곡판"]
    assert detail["tokens"] == {"remix"}
    assert detail["who"] == set()   # 참여자는 영상 제목 부가 문구와 섞여 담지 않는다


# --- 리뷰 반영: 괄호 밖 버전 표시도 비교 --------------------------------------

def test_bare_arrangement_kind_is_compared() -> None:
    """(Remix)처럼 괄호가 있어야만 차단되면 안 된다."""
    assert title_conflict("Song (Acoustic Ver.)", "가수 Song Remix") != ""
    assert version_mismatch("Song (Acoustic Ver.)", "Song Remix") != ""


def test_bare_marker_of_the_same_kind_agrees() -> None:
    assert title_conflict("하늘위로 (Remix)", "렉시 - 하늘위로 Remix MV") == ""
    assert title_conflict("태양을 피하는 방법 (Gtr.Remix)", "비 - 태양을 피하는 방법 Remix MV") == ""


# --- 리뷰 반영: 표기 뼈대와 참여자 이름을 분리 ---------------------------------

def test_duet_partner_matches_regardless_of_wording() -> None:
    """'Duet with 선미'와 'Duet 선미'는 같은 녹음이다."""
    assert version_mismatch("When We Disco (Duet with 선미)", "When We Disco (Duet 선미)") == ""
    assert title_conflict("When We Disco (Duet with 선미)", "박진영 - When We Disco (Duet 선미)") == ""


def test_different_duet_partner_still_conflicts() -> None:
    assert version_mismatch(
        "When We Disco (Duet with 선미)", "When We Disco (Duet with 다른가수)"
    ) != ""


def test_partner_is_separated_from_the_boilerplate() -> None:
    """뼈대(Duet/with/Ver.)를 떼고 참여자만 남아야 한다."""
    left = version_details("When We Disco (Duet with 선미)")["듀엣판"]
    right = version_details("When We Disco (Duet 선미)")["듀엣판"]
    assert left["tokens"] == right["tokens"] == {"duet"}
    assert left["who"] == right["who"] == {"선미"}


def test_unknown_partner_does_not_block() -> None:
    """'(Duet Ver.)'처럼 상대가 안 적힌 경우는 상대 비교를 보류한다."""
    assert version_mismatch("My Love (Duet Ver.)", "My Love (Duet with 누군가)") == ""
