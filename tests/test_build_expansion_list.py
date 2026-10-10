"""
tests/test_build_expansion_list.py

추가 크롤링 목록 만들기(build_expansion_list)의 정규화·중복 제거·멈춤 규칙 테스트.
네트워크를 쓰지 않는다. 멜론 응답은 가짜 세션으로 흉내 낸다.

실행:
    venv/bin/python -m pytest tests/test_build_expansion_list.py -v
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from src.crawler.scripts_py import build_expansion_list as bel
from src.crawler.scripts_py.build_expansion_list import (
    REASON_ADULT,
    REASON_CANDIDATE_DUP,
    REASON_CAP,
    REASON_EXISTING,
    REASON_NOT_CREDITED,
    REASON_VARIANT,
    AliasMap,
    MelonClient,
    Progress,
    Song,
    StopCrawl,
    BudgetReached,
    artist_tokens,
    assemble,
    parse_artist_anchors,
    parse_song_rows,
    pick_artist,
    split_artists,
    title_key,
    variant_reason,
)

ALIASES = AliasMap.from_rows([
    {"artist": "싸이", "alias": "PSY"},
    {"artist": "DAY6", "alias": "데이식스"},
    {"artist": "원필", "alias": "원필 (DAY6)"},
    {"artist": "아이유", "alias": "IU"},
    {"artist": "지코", "alias": "ZICO"},
    {"artist": "블랙핑크", "alias": "BLACKPINK"},
])


# --- 정규화 -------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("에잇 (Prod.&Feat. SUGA of BTS)", "에잇"),
    ("사랑은 늘 도망가 (드라마 '신사와 아가씨' OST)", "사랑은 늘 도망가"),
    ("Hype Boy (2023 Remaster)", "hype boy"),
    ("그대가, 그대를...", "그대가 그대를"),
    ("Love Dive Feat. 누구", "LOVE DIVE"),
    ("ROSÉ", "rose"),
    ("밤편지 [가사]", "밤편지"),
])
def test_title_key_drops_extra_info(a: str, b: str) -> None:
    assert title_key(a) == title_key(b)


def test_title_key_keeps_bracket_only_title() -> None:
    assert title_key("(Intro)") == "intro"


def test_title_key_keeps_with_in_title() -> None:
    assert title_key("Dance With Me") != title_key("Dance")


def test_split_artists_respects_parentheses() -> None:
    assert split_artists("아이유, 지코") == ["아이유", "지코"]
    assert split_artists("싹쓰리 (유두래곤, 린다G, 비룡)") == ["싹쓰리 (유두래곤, 린다G, 비룡)"]
    assert split_artists("철싸 (노홍철 & 싸이), 누구") == ["철싸 (노홍철 & 싸이)", "누구"]


def test_artist_tokens_use_paren_alias_but_not_member_list() -> None:
    assert artist_tokens("TWICE (트와이스)") == {"twice트와이스", "twice", "트와이스"}
    assert "유두래곤" not in artist_tokens("싹쓰리 (유두래곤, 린다G, 비룡)")


def test_alias_unifies_names() -> None:
    assert ALIASES.artist_set("PSY") == ALIASES.artist_set("싸이") == frozenset({"싸이"})
    assert "싸이" in ALIASES.artist_set("싸이 (PSY)")


def test_alias_paren_does_not_merge_member_into_group() -> None:
    # '원필 (DAY6)'의 괄호 속을 alias 열쇠로 쓰면 목록의 DAY6가 원필로 묶인다.
    assert ALIASES.canonical("day6") == "day6"
    assert ALIASES.artist_set("DAY6 (데이식스)") & frozenset({"day6"})
    assert "원필" not in ALIASES.artist_set("DAY6 (데이식스)")


# --- 버전 변형 -----------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "밤편지 (Inst.)", "밤편지 (MR)", "Kill This Love (Live)", "Gentleman (Remix)",
    "봄날 (Acoustic Ver.)", "LOVE DIVE (Japanese Ver.)", "사랑 (Piano Ver.)", "노래 (Sped Up)",
    "노래 - Live", "노래 [Instrumental]",
])
def test_variant_detected(title: str) -> None:
    assert variant_reason(title)


@pytest.mark.parametrize("title", [
    "Live My Life", "Mr.Mr.", "그대니까요 (Duet With 차은주)", "To My Love (Korean Ver.)",
    "에잇 (Prod.&Feat. SUGA of BTS)", "Hype Boy (2023 Remaster)", "Alive", "Remixed Heart",
    "Lunisolar (Original Mix)",
])
def test_variant_not_detected(title: str) -> None:
    assert variant_reason(title) == ""


# --- 조립 ---------------------------------------------------------------------

EXISTING = [
    Song("싸이 (PSY)", "강남스타일", "3853978", "existing"),
    Song("DAY6 (데이식스)", "예뻤어", "1111", "existing"),
    Song("싹쓰리 (유두래곤, 린다G, 비룡)", "다시 여기 바닷가", "32790516", "existing"),
]


def _cand(artist: str, titles: List[str], source: str, start_id: int) -> List[Song]:
    return [Song(artist, t, str(start_id + i), source, rank=i) for i, t in enumerate(titles)]


def _reasons(result) -> dict:
    return {(e.song.artist, e.song.title): e.reason for e in result.excluded}


def test_existing_duplicate_by_id_and_by_key() -> None:
    cands = {"싸이": [
        Song("싸이 (PSY)", "강남스타일", "3853978", "싸이", rank=0),            # 같은 ID
        Song("PSY", "강남스타일 (Remastered)", "999", "싸이", rank=1),          # 다른 ID, 같은 키 + alias
        Song("싸이 (PSY)", "That That (Prod. SUGA)", "998", "싸이", rank=2),
    ]}
    r = assemble(EXISTING, cands, [], ALIASES)
    reasons = _reasons(r)
    assert reasons[("싸이 (PSY)", "강남스타일")] == REASON_EXISTING
    assert reasons[("PSY", "강남스타일 (Remastered)")] == REASON_EXISTING
    assert [s.title for s in r.new_songs] == ["That That (Prod. SUGA)"]


def test_same_title_other_artist_is_not_duplicate() -> None:
    cands = {"아이유": [Song("아이유", "예뻤어", "5000", "아이유", rank=0)]}
    r = assemble(EXISTING, cands, [], ALIASES)
    assert [s.title for s in r.new_songs] == ["예뻤어"]


def test_collab_duplicate_across_artists_keeps_first() -> None:
    cands = {
        "아이유": [Song("아이유, 지코", "협업곡", "7000", "아이유", rank=0)],
        "지코": [Song("지코 (ZICO), 아이유", "협업곡 (Feat. 누구)", "7001", "지코", rank=0)],
    }
    r = assemble([], cands, [], ALIASES)
    assert [s.source for s in r.new_songs] == ["아이유"]
    assert _reasons(r)[("지코 (ZICO), 아이유", "협업곡 (Feat. 누구)")] == REASON_CANDIDATE_DUP


def test_cap_keeps_popularity_order_after_exclusions() -> None:
    titles = [f"곡{i}" for i in range(30)]
    titles[0] = "곡0 (Live)"                 # 변형은 상한 자리를 쓰지 않는다
    cands = {"아이유": _cand("아이유", titles, "아이유", 100)}
    r = assemble([], cands, [], ALIASES, cap=20)
    assert [s.title for s in r.new_songs] == [f"곡{i}" for i in range(1, 21)]
    reasons = [e.reason for e in r.excluded]
    assert reasons.count(REASON_VARIANT) == 1 and reasons.count(REASON_CAP) == 9
    assert r.added_per_artist["아이유"] == 20


def test_must_add_ignores_cap_and_wins_over_candidates() -> None:
    cands = {"아이유": _cand("아이유", [f"곡{i}" for i in range(25)], "아이유", 100)}
    must = [Song("IU", "곡0", "", "must_add"), Song("싸이", "강남스타일", "", "must_add"),
            Song("블랙핑크", "마지막처럼", "", "must_add")]
    r = assemble(EXISTING, cands, must, ALIASES, cap=20)
    titles = [s.title for s in r.new_songs]
    assert titles[:2] == ["곡0", "마지막처럼"]                         # must_add가 먼저
    assert _reasons(r)[("싸이", "강남스타일")] == REASON_EXISTING       # must_add도 중복 검사
    assert _reasons(r)[("아이유", "곡0")] == REASON_CANDIDATE_DUP
    assert r.added_per_artist["아이유"] == 20                            # must_add는 상한 자리를 안 씀
    assert r.must_add_added == 2


def test_adult_and_not_credited_are_excluded() -> None:
    cands = {"아이유": [
        Song("아이유", "19금 곡", "1", "아이유", rank=0, adult=True),
        Song("남의 가수", "피처링만", "2", "아이유", rank=1, credited=False),
    ]}
    r = assemble([], cands, [], ALIASES)
    assert not r.new_songs
    assert {e.reason for e in r.excluded} == {REASON_ADULT, REASON_NOT_CREDITED}


def test_paren_member_list_artist_matches_existing() -> None:
    cands = {"싹쓰리": [Song("싹쓰리 (유두래곤, 린다G, 비룡)", "다시 여기 바닷가", "1", "싹쓰리", rank=0)]}
    assert assemble(EXISTING, cands, [], ALIASES).excluded[0].reason == REASON_EXISTING


REAL_EXISTING = Path("data/songs_3010.csv")


@pytest.mark.skipif(not REAL_EXISTING.is_file(), reason="data/songs_3010.csv 없음(로컬 전용)")
def test_against_real_existing_csv() -> None:
    existing = bel.load_existing(REAL_EXISTING)
    assert len(existing) == 3010
    must = [Song("PSY", "강남스타일", "", "must_add"), Song("싸이", "새로운 가짜곡", "", "must_add")]
    cands = {"싸이": [Song("싸이 (PSY)", "챔피언", "430978", "싸이", rank=0),
                     Song("싸이 (PSY)", "낙원 (Feat. 이재훈)", "77", "싸이", rank=1),
                     Song("싸이 (PSY)", "가짜 신곡", "78", "싸이", rank=2)]}
    r = assemble(existing, cands, must, ALIASES)
    reasons = _reasons(r)
    assert reasons[("PSY", "강남스타일")] == REASON_EXISTING
    assert reasons[("싸이 (PSY)", "챔피언")] == REASON_EXISTING
    assert reasons[("싸이 (PSY)", "낙원 (Feat. 이재훈)")] == REASON_EXISTING
    assert [s.title for s in r.new_songs] == ["새로운 가짜곡", "가짜 신곡"]


# --- 파싱 ---------------------------------------------------------------------

SONG_LIST_HTML = """
<table><tbody>
<tr>
  <td><input type="checkbox" class="input_check" title="Celebrity 곡 선택" value="33077590"/></td>
  <td><a href="javascript:melon.link.goSongDetail('33077590');" class="btn btn_icon_detail">
      <span class="odd_span">Celebrity 상세정보 페이지 이동</span></a></td>
  <td><div id="artistName"><a href="javascript:melon.link.goArtistDetail('261143');" class="fc_mgray">아이유</a></div></td>
</tr>
<tr>
  <td><input type="checkbox" class="input_check" title="협업곡 (Feat. 지코) 곡 선택" value="2"/></td>
  <td><span class="age_19">19금</span></td>
  <td><div id="artistName"><a href="javascript:melon.link.goArtistDetail('261143');" class="fc_mgray">아이유</a>
      <a href="javascript:melon.link.goArtistDetail('545');" class="fc_mgray">지코 (ZICO)</a></div></td>
</tr>
<tr><td>헤더처럼 ID 없는 행</td></tr>
</tbody></table>
"""


def test_parse_song_rows() -> None:
    rows = parse_song_rows(SONG_LIST_HTML)
    assert [r["song_id"] for r in rows] == ["33077590", "2"]
    assert rows[0]["title"] == "Celebrity" and rows[0]["artists"] == [("261143", "아이유")]
    assert rows[1]["adult"] is True
    assert rows[1]["artists"] == [("261143", "아이유"), ("545", "지코 (ZICO)")]


def test_pick_artist_prefers_most_frequent_and_reports_same_name() -> None:
    html = """
    <a href="javascript:melon.link.goArtistDetail('111');">정인</a>
    <a href="javascript:melon.link.goArtistDetail('222');">정인</a>
    <a href="javascript:melon.link.goArtistDetail('222');">정인</a>
    <a href="javascript:melon.link.goArtistDetail('333');">조정치</a>
    """
    pick = pick_artist(parse_artist_anchors(html), {"정인"}, AliasMap())
    assert pick is not None and pick.artist_id == "222" and pick.hits == 2
    assert [o[0] for o in pick.others] == ["111"]


def test_pick_artist_matches_alias_in_parentheses() -> None:
    html = "<a href=\"javascript:melon.link.goArtistDetail('7');\">PSY (싸이)</a>"
    assert pick_artist(parse_artist_anchors(html), {"싸이"}, ALIASES).artist_id == "7"


# --- 멈춤 규칙(가짜 세션) --------------------------------------------------------

class _Resp:
    def __init__(self, status: int, text: str) -> None:
        self.status_code = status
        self.text = text
        self.encoding = "utf-8"


class _FakeSession:
    def __init__(self, responses: List) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


U = "https://www.melon.com/x"
NORMAL = "<html>" + "x" * 300 + "goArtistDetail('1')</html>"


def _client(tmp_path: Path, responses: List, **kw) -> MelonClient:
    progress = Progress(tmp_path / "progress.json")
    client = MelonClient(tmp_path, progress, max_requests=kw.get("max_requests", 300),
                         run_budget=kw.get("run_budget"), delay=(0.0, 0.0), offline=kw.get("offline", False))
    client._session = _FakeSession(responses)
    return client


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(bel.time, "sleep", lambda s: None)


@pytest.mark.parametrize("resp, word", [
    (_Resp(403, NORMAL), "HTTP 403"),
    (_Resp(429, NORMAL), "HTTP 429"),
    (_Resp(406, NORMAL), "HTTP 406"),
    (_Resp(200, ""), "빈 응답"),
    (_Resp(200, NORMAL.replace("</html>", "reCAPTCHA</html>")), "차단/캡차"),
    (_Resp(200, "<html>" + "y" * 400 + "</html>"), "예상과 다른 구조"),
])
def test_unexpected_response_stops_without_retry(tmp_path: Path, resp, word: str) -> None:
    client = _client(tmp_path, [resp, _Resp(200, NORMAL)])
    with pytest.raises(StopCrawl) as e:
        client.get("https://www.melon.com/x", {"q": "a"}, {}, expect="goArtistDetail")
    assert word in e.value.reason
    assert client._session.calls == 1                       # 재시도 없음
    assert client.progress.requests_made == 1
    assert not list((tmp_path / "responses").glob("*.html"))  # 이상한 응답은 캐시하지 않음


def test_connection_error_stops(tmp_path: Path) -> None:
    import requests
    client = _client(tmp_path, [requests.ConnectionError("reset")])
    with pytest.raises(StopCrawl):
        client.get("https://www.melon.com/x", {"q": "a"}, {}, expect="goArtistDetail")
    assert client._session.calls == 1


def test_cache_hit_makes_no_request(tmp_path: Path) -> None:
    client = _client(tmp_path, [_Resp(200, NORMAL)])
    assert client.get("https://www.melon.com/x", {"q": "a"}, {}, expect="goArtistDetail") == NORMAL
    assert client.get("https://www.melon.com/x", {"q": "a"}, {}, expect="goArtistDetail") == NORMAL
    assert client._session.calls == 1 and client.progress.requests_made == 1


def test_empty_result_marker_is_not_a_stop(tmp_path: Path) -> None:
    body = "<html>" + "z" * 300 + "검색결과가 없습니다</html>"
    client = _client(tmp_path, [_Resp(200, body)])
    assert client.get(U, {"q": "a"}, {}, expect="goArtistDetail", empty_markers=bel.SEARCH_EMPTY_MARKERS) == body


def test_request_caps(tmp_path: Path) -> None:
    client = _client(tmp_path, [_Resp(200, NORMAL)] * 3, max_requests=2)
    client.get(U, {"q": "1"}, {}, expect="goArtistDetail")
    client.get(U, {"q": "2"}, {}, expect="goArtistDetail")
    with pytest.raises(BudgetReached):
        client.get(U, {"q": "3"}, {}, expect="goArtistDetail")
    assert client._session.calls == 2

    # 누적 횟수는 진행 파일에 남아 다음 실행에도 적용된다.
    again = _client(tmp_path, [_Resp(200, NORMAL)], max_requests=2)
    with pytest.raises(BudgetReached):
        again.get(U, {"q": "4"}, {}, expect="goArtistDetail")


def test_run_budget(tmp_path: Path) -> None:
    client = _client(tmp_path, [_Resp(200, NORMAL)] * 2, run_budget=1)
    client.get(U, {"q": "1"}, {}, expect="goArtistDetail")
    with pytest.raises(BudgetReached):
        client.get(U, {"q": "2"}, {}, expect="goArtistDetail")


def test_offline_never_requests(tmp_path: Path) -> None:
    client = _client(tmp_path, [], offline=True)
    assert client.get(U, {"q": "1"}, {}, expect="goArtistDetail") is None
    assert client._session.calls == 0
