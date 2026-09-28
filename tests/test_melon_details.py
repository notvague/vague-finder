"""
tests/test_melon_details.py

멜론 상세 페이지 수집(collect_melon_data.fetch_melon_details)의 회귀 테스트.

배경 (2026-09-18 점검): HTTP 상태와 필수 필드를 보지 않고 HTML을 파싱했다. 403 안내
페이지를 넣으면 title="Unknown", genre=""인 사전이 나왔고, 수집 함수는 그것을 채택해
두 번째 후보를 시도하지 않았다.

이제 HTTP 오류 / 페이지 없음 / 파싱 실패를 구분한다. 5xx와 네트워크 오류는 제한적으로
다시 시도하고, 403·429는 MelonAccessError로 올려 배치를 멈추게 한다.

실행:
    venv/bin/python -m pytest tests/test_melon_details.py -v
"""
from __future__ import annotations

import pytest
import requests

from src.crawler.scripts_py import collect_melon_data as cmd
from src.crawler.scripts_py.collect_melon_data import MelonAccessError, SongCandidate
from src.crawler.scripts_py.melon_match import MatchScore

DETAIL_HTML = """
<html><body>
  <div class="song_name">곡명 밤편지</div>
  <div class="artist"><span class="wrap_dtl"><a>아이유</a></span></div>
  <div class="meta"><dl class="list">
    <dt>앨범</dt><dd>Palette</dd>
    <dt>발매일</dt><dd>2017.03.24</dd>
    <dt>장르</dt><dd>발라드</dd>
  </dl></div>
  <div class="lyric">첫 줄<br>둘째 줄</div>
</body></html>
"""
FORBIDDEN_HTML = "<html><body><h1>접근이 제한되었습니다</h1></body></html>"

# 19세 미만 이용불가 곡. 멜론이 곡명 앞에 배지를 붙이고, 가사 영역(.lyric)은 오지 않는다.
# 2026-09-18 실제 페이지(songId=31854689)에서 가져온 구조다.
ADULT_BADGE = ('<span class="bullet_icons age_19 large" title="19세 미만 청소년 이용불가">'
               '<span class="none">19금</span></span>')
ADULT_DETAIL_HTML = """
<html><body>
  <div class="song_name"> <strong class="none">곡명</strong> __BADGE__ BAND </div>
  <div class="artist"><span class="wrap_dtl"><a>창모 (CHANGMO)</a></span></div>
  <div class="meta"><dl class="list">
    <dt>앨범</dt><dd>BAND</dd>
    <dt>발매일</dt><dd>2019.06.08</dd>
    <dt>장르</dt><dd>랩/힙합</dd>
  </dl></div>
</body></html>
""".replace("__BADGE__", ADULT_BADGE)


class Response:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def melon_server(monkeypatch):
    """URL(songId)마다 응답 목록을 순서대로 돌려준다. 목록이 끝나면 마지막 응답을 반복한다."""
    monkeypatch.setattr(cmd.time, "sleep", lambda s: None)
    monkeypatch.setattr(cmd, "fetch_melon_playlists_from_song_page", lambda soup, max_items=2: [])
    calls = []

    def install(routes):
        queues = {key: list(value) for key, value in routes.items()}

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            key = next((k for k in queues if k in url or (params and k in str(params))), None)
            if key is None:
                raise requests.ConnectionError(f"경로 없음: {url}")
            queue = queues[key]
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(cmd.requests, "get", fake_get)
        return calls
    return install


def test_valid_page_is_parsed() -> None:
    details = cmd.parse_melon_details(DETAIL_HTML)
    assert (details["title"], details["artist"], details["genre"]) == ("밤편지", ["아이유"], "발라드")
    assert details["lyrics"] == "첫 줄\n둘째 줄"


def test_403_is_an_access_error_not_a_record(melon_server) -> None:
    melon_server({"songId=1": [Response(403, FORBIDDEN_HTML)]})
    with pytest.raises(MelonAccessError):
        cmd.fetch_melon_details("1")


def test_429_on_search_propagates_to_the_caller(melon_server) -> None:
    melon_server({"search": [Response(429, "")]})
    with pytest.raises(MelonAccessError):
        cmd.fetch_melon_song_candidates("아이유", "밤편지")


@pytest.mark.parametrize(
    "html",
    [
        FORBIDDEN_HTML,                                              # 오류 안내 본문(200으로 왔을 때)
        "<html><body><div class='song_name'>곡명 X</div></body></html>",  # 제목만 있고 가수 없음
        "",
    ],
)
def test_page_without_required_fields_is_rejected(html) -> None:
    """예전에는 title='Unknown', artist=['Unknown']으로 채워 정상 데이터처럼 돌려줬다."""
    assert cmd.parse_melon_details(html) == {}


def test_parse_failure_moves_on_to_the_next_candidate(melon_server) -> None:
    melon_server({
        "songId=1": [Response(200, FORBIDDEN_HTML)],
        "songId=2": [Response(200, DETAIL_HTML)],
        "cmt.melon.com": [Response(200, "{}")],
    })
    monkeypatch_comments = lambda song_id, **kw: []
    cmd.fetch_melon_comments, original = monkeypatch_comments, cmd.fetch_melon_comments
    try:
        candidates = [
            SongCandidate(song_id="1", title="밤편지", match=MatchScore(score=100)),
            SongCandidate(song_id="2", title="밤편지", match=MatchScore(score=90)),
        ]
        result = cmd.collect_melon_data("아이유", "밤편지", candidates=candidates)
    finally:
        cmd.fetch_melon_comments = original
    assert result["id"] == "2"


def test_server_error_is_retried_then_raised_not_swallowed(melon_server) -> None:
    """일시 오류를 빈 사전으로 뭉개면 '이 후보는 틀렸다'로 읽혀 다른 음원이 채택된다."""
    calls = melon_server({"songId=1": [Response(500), Response(502), Response(503)]})
    with pytest.raises(cmd.MelonTransientError):
        cmd.fetch_melon_details("1")
    assert len(calls) == cmd.PAGE_RETRIES + 1


def test_transient_error_does_not_fall_through_to_the_next_candidate(melon_server, monkeypatch) -> None:
    """1위 후보가 잠깐 503이면 2위(같은 곡의 다른 릴리스)를 채택하는 대신 곡 단위로 실패한다.

    예전에는 조용히 2위가 채택됐고, 다음 실행의 중복 검사는 1위 id로 하므로 같은 시드에
    레코드가 둘 남았다. failed_songs.csv·review_songs.csv 어디에도 안 남았다.
    """
    melon_server({"songId=1": [Response(503)], "songId=2": [Response(200, DETAIL_HTML)]})
    monkeypatch.setattr(cmd, "fetch_melon_comments", lambda song_id, **kw: [])
    candidates = [
        SongCandidate(song_id="1", title="밤편지", match=MatchScore(score=100)),
        SongCandidate(song_id="2", title="밤편지", match=MatchScore(score=100)),
    ]
    with pytest.raises(cmd.MelonTransientError):
        cmd.collect_melon_data("아이유", "밤편지", candidates=candidates)


def test_parse_details_never_makes_a_request(monkeypatch) -> None:
    """파서는 순수 함수여야 테스트로 고정된다. 앨범 소개는 fetch_melon_details가 받는다."""
    requested = []
    monkeypatch.setattr(cmd.requests, "get", lambda *a, **k: requested.append(a) or Response(200, ""))
    html = DETAIL_HTML.replace(
        "<dt>앨범</dt><dd>Palette</dd>",
        "<dt>앨범</dt><dd><a href=\"javascript:melon.link.goAlbumDetail('10554246');\">Palette</a></dd>",
    )
    details = cmd.parse_melon_details(html)
    assert requested == []
    assert details["album_id"] == "10554246" and details["album_desc"] == ""


def test_fetch_details_fills_the_album_description(melon_server) -> None:
    html = DETAIL_HTML.replace(
        "<dt>앨범</dt><dd>Palette</dd>",
        "<dt>앨범</dt><dd><a href=\"javascript:melon.link.goAlbumDetail('777');\">Palette</a></dd>",
    )
    melon_server({
        "songId=1": [Response(200, html)],
        "albumId=777": [Response(200, "<div class='dtl_albuminfo'>앨범소개 이 앨범은</div>")],
    })
    details = cmd.fetch_melon_details("1")
    assert "album_id" not in details
    assert details["album_desc"] == "이 앨범은"


def test_server_error_recovers_on_retry(melon_server) -> None:
    melon_server({"songId=1": [Response(500), Response(200, DETAIL_HTML)]})
    assert cmd.fetch_melon_details("1")["title"] == "밤편지"


def test_404_returns_empty_without_retry(melon_server) -> None:
    calls = melon_server({"songId=1": [Response(404)]})
    assert cmd.fetch_melon_details("1") == {}
    assert len(calls) == 1


def test_network_error_is_retried(melon_server, monkeypatch) -> None:
    attempts = []

    def flaky_get(url, params=None, headers=None, timeout=None):
        attempts.append(url)
        if len(attempts) == 1:
            raise requests.ConnectionError("reset")
        return Response(200, DETAIL_HTML)

    monkeypatch.setattr(cmd.requests, "get", flaky_get)
    assert cmd.fetch_melon_details("1")["title"] == "밤편지"
    assert len(attempts) == 2


# --- 19금 곡은 가사를 받을 수 없어 채택하지 않는다 -------------------------------------------
# 수집분 961곡 중 가사가 빈 8곡이 전부 19금이었다. 배지의 '19금' 글자가 제목에도 섞여
# '19금 BAND', '19금 그XX'로 저장됐다.

def test_adult_song_is_detected_and_the_title_is_clean() -> None:
    details = cmd.parse_melon_details(ADULT_DETAIL_HTML)
    assert details["is_adult"] is True
    assert details["title"] == "BAND"          # '19금 BAND'가 아니다


def test_normal_song_is_not_marked_adult() -> None:
    details = cmd.parse_melon_details(DETAIL_HTML)
    assert "is_adult" not in details


def test_adult_candidate_is_rejected_before_the_detail_request(melon_server) -> None:
    """검색 결과 행에도 같은 배지가 있다. 상세 요청 전에 거른다."""
    row = f'''<tr>
        <td><a class="btn_icon_detail" href="javascript:melon.link.goSongDetail('31854689');">상세</a></td>
        <td><div class="ellipsis rank01">{ADULT_BADGE}<a class="fc_gray">BAND</a></div>
            <div id="artistName"><a class="fc_mgray">창모 (CHANGMO)</a></div></td>
        <td></td><td></td><td>BAND</td>
    </tr>'''
    calls = melon_server({"search": [Response(200, f"<html><body><table>{row}</table></body></html>")]})
    assert cmd.fetch_melon_song_candidates("창모", "BAND") == []
    assert len(calls) == 1      # 상세 페이지를 요청하지 않았다


def test_adult_song_found_only_in_the_detail_page_is_skipped(melon_server, monkeypatch) -> None:
    """검색 행에 배지가 없더라도(레이아웃 변경) 상세에서 다시 본다."""
    melon_server({"songId=1": [Response(200, ADULT_DETAIL_HTML)],
                  "songId=2": [Response(200, DETAIL_HTML)]})
    monkeypatch.setattr(cmd, "fetch_melon_comments", lambda song_id, **kw: [])
    candidates = [
        SongCandidate(song_id="1", title="BAND", match=MatchScore(score=100)),
        SongCandidate(song_id="2", title="밤편지", match=MatchScore(score=90)),
    ]
    result = cmd.collect_melon_data("창모", "BAND", candidates=candidates)
    assert result["id"] == "2"


def test_lyric_less_candidate_is_skipped_before_comments_are_fetched(melon_server, monkeypatch) -> None:
    """가사 없는 후보에 댓글 요청·LLM 할당량을 쓰지 않는다. 다음 후보로 넘어간다."""
    no_lyrics = DETAIL_HTML.replace('<div class="lyric">첫 줄<br>둘째 줄</div>', "")
    melon_server({"songId=1": [Response(200, no_lyrics)], "songId=2": [Response(200, DETAIL_HTML)]})
    fetched = []
    monkeypatch.setattr(cmd, "fetch_melon_comments", lambda song_id, **kw: fetched.append(song_id) or [])
    candidates = [
        SongCandidate(song_id="1", title="밤편지", match=MatchScore(score=100)),
        SongCandidate(song_id="2", title="밤편지", match=MatchScore(score=90)),
    ]
    result = cmd.collect_melon_data("아이유", "밤편지", candidates=candidates)
    assert result["id"] == "2"
    assert fetched == ["2"]          # 1번 후보에는 댓글 요청을 보내지 않았다


# --- 낭비 제거: 탈락할 후보에는 앨범 소개를 받지 않는다 ---------------------------------------

def test_album_description_is_not_fetched_for_a_rejected_candidate(melon_server, monkeypatch) -> None:
    """가사 없는 후보에도 앨범 페이지 요청이 나갔다."""
    cmd.reset_album_desc_cache()
    no_lyrics = DETAIL_HTML.replace('<div class="lyric">첫 줄<br>둘째 줄</div>', "").replace(
        "<dt>앨범</dt><dd>Palette</dd>",
        "<dt>앨범</dt><dd><a href=\"javascript:melon.link.goAlbumDetail('777');\">Palette</a></dd>")
    with_lyrics = DETAIL_HTML.replace(
        "<dt>앨범</dt><dd>Palette</dd>",
        "<dt>앨범</dt><dd><a href=\"javascript:melon.link.goAlbumDetail('888');\">Palette</a></dd>")
    calls = melon_server({
        "songId=1": [Response(200, no_lyrics)],
        "songId=2": [Response(200, with_lyrics)],
        "albumId=777": [Response(200, "<div class='dtl_albuminfo'>앨범소개 안 받아야 함</div>")],
        "albumId=888": [Response(200, "<div class='dtl_albuminfo'>앨범소개 이 앨범은</div>")],
    })
    monkeypatch.setattr(cmd, "fetch_melon_comments", lambda song_id, **kw: [])
    candidates = [
        SongCandidate(song_id="1", title="밤편지", match=MatchScore(score=100)),
        SongCandidate(song_id="2", title="밤편지", match=MatchScore(score=90)),
    ]
    result = cmd.collect_melon_data("아이유", "밤편지", candidates=candidates)
    assert result["album_desc"] == "이 앨범은"
    requested = [url for url, _ in calls]
    assert not any("albumId=777" in url for url in requested), requested


def test_album_description_is_cached_within_a_run(melon_server) -> None:
    """같은 앨범의 다른 곡이 같은 페이지를 다시 받지 않는다."""
    cmd.reset_album_desc_cache()
    calls = melon_server({"albumId=777": [Response(200, "<div class='dtl_albuminfo'>앨범소개 본문</div>")]})
    assert cmd.fetch_album_desc("777") == "본문"
    assert cmd.fetch_album_desc("777") == "본문"
    assert len(calls) == 1


def test_a_failed_album_request_is_not_cached(melon_server) -> None:
    """일시 오류로 빈 소개가 된 것을 캐시하면 그 앨범의 모든 곡이 소개 없이 저장된다."""
    cmd.reset_album_desc_cache()
    melon_server({"albumId=777": [Response(503)]})
    assert cmd.fetch_album_desc("777") == ""      # 재시도를 다 써도 503 -> 빈 값

    melon_server({"albumId=777": [Response(200, "<div class='dtl_albuminfo'>앨범소개 본문</div>")]})
    assert cmd.fetch_album_desc("777") == "본문"   # 캐시되지 않았으므로 다시 받는다


def test_adopted_song_that_is_already_collected_stops_before_comments(melon_server, monkeypatch) -> None:
    melon_server({"songId=1": [Response(200, DETAIL_HTML)]})
    fetched = []
    monkeypatch.setattr(cmd, "fetch_melon_comments", lambda song_id, **kw: fetched.append(song_id) or [])
    candidates = [SongCandidate(song_id="1", title="밤편지", match=MatchScore(score=100))]
    with pytest.raises(cmd.AlreadyCollected):
        cmd.collect_melon_data("아이유", "밤편지", candidates=candidates,
                               already_collected=lambda sid: sid == "1")
    assert fetched == []
