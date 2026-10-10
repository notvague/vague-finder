"""
tests/test_youtube_selection.py

유튜브 영상 선택(collect_reaction)의 회귀 테스트.

배경 (2026-09-18 점검):

1. 가수 이름은 검색어에만 들어가고 결과 검증은 곡 제목·버전만 봤다. 조회수순으로 고르니
   요청한 가수보다 조회수가 높은 다른 가수의 동명곡이 먼저 뽑혔다.
2. 댓글이 꺼진 영상은 통째로 건너뛰었다. 후보 전부가 댓글 비활성화면 정상 음원이 있어도
   {}를 돌려줘 곡 전체를 수집하지 않았다.
3. 다운로드 포맷 bestaudio[ext=m4a]/bestaudio가 audio.webm을 남기면 저장은 성공이지만
   완료 판정·Mongo 경로·임베딩 입력은 audio.m4a만 봐서 재실행에서 미완료가 됐다.

yt-dlp는 가짜다. 제목·채널명은 지어낸 것이다.

실행:
    venv/bin/python -m pytest tests/test_youtube_selection.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.crawler.scripts_py import collect_reaction as cr
from src.crawler.scripts_py.melon_match import title_conflict  # noqa: F401  (cr 경유로 씀)

EXCLUDE = cr.EXCLUDE_KEYWORDS      # 프로덕션 목록을 그대로 쓴다


def entry(id_, title, channel="채널", views=0, **extra):
    return {"id": id_, "title": title, "channel": channel, "view_count": views, **extra}


# --- 1. 다른 가수의 동명곡 ---------------------------------------------------------------

def test_other_artists_same_title_loses_to_the_requested_artist() -> None:
    """조회수가 높아도 가수가 다르면 뒤로 간다. 가수 근거가 있는 후보가 있으면 없는 후보는 버린다."""
    picked = cr.select_candidates(
        [
            entry("x", "다른가수 - 밤편지 Official MV", channel="다른가수", views=5_000_000),
            entry("y", "아이유(IU) - 밤편지 Official", channel="1theK", views=1_000_000),
        ],
        melon_title="밤편지", known_artists=["아이유"], exclude_keywords=EXCLUDE,
    )
    assert [e["id"] for e in picked] == ["y"]
    assert picked[0]["_artist_verified"] is True


def test_official_source_outranks_views_among_the_same_artist() -> None:
    picked = cr.select_candidates(
        [
            entry("lyric", "아이유 밤편지 가사 lyrics", channel="가사채널", views=10_000_000),
            entry("mv", "[MV] 아이유 - 밤편지", channel="1theK", views=5_000_000),
            entry("topic", "밤편지", channel="아이유 - Topic", views=100),
        ],
        melon_title="밤편지", known_artists=["아이유"], exclude_keywords=EXCLUDE,
    )
    assert [e["id"] for e in picked] == ["mv", "topic", "lyric"]


def test_no_artist_evidence_anywhere_keeps_view_order_but_flags_for_review() -> None:
    picked = cr.select_candidates(
        [entry("a", "밤편지 Audio", views=10), entry("b", "밤편지 lyrics", views=20)],
        melon_title="밤편지", known_artists=["아이유"], exclude_keywords=EXCLUDE,
    )
    assert [e["id"] for e in picked] == ["b", "a"]
    assert all(e["_artist_verified"] is False for e in picked)


def test_an_arrangement_video_loses_to_the_official_one() -> None:
    """'밤편지 piano'는 곡 이름은 맞지만 원곡 오디오가 아니다. 키워드 단계가 거른다.

    이름 비교로 막지는 않는다. 한국어 제목 옆에 영어 제목이 붙는 정상 표기('주지마 Don't')와
    구조가 같아서, 이름만 보고는 구분할 수 없다.
    """
    picked = cr.select_candidates(
        [entry("a", "밤편지 piano", views=9_000_000),
         entry("b", "아이유 - 밤편지 Official MV", views=100)],
        melon_title="밤편지", known_artists=["아이유"], exclude_keywords=EXCLUDE,
    )
    assert [e["id"] for e in picked] == ["b"]


def test_bilingual_melon_artist_name_matches_either_spelling() -> None:
    """멜론은 '태연 (TAEYEON)'처럼 병기한다. 통째로 비교하면 어느 쪽도 못 찾는다."""
    assert cr.expand_names(["태연 (TAEYEON)"]) == ["태연 (TAEYEON)", "태연", "TAEYEON"]
    assert cr.artist_evidence({"title": "TAEYEON 태연 'Fine' MV"}, ["태연 (TAEYEON)"]) is True
    assert cr.artist_evidence({"title": "Fine (cover by someone)"}, ["태연 (TAEYEON)"]) is False


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("IU", "IU - Through the Night", True),
        ("IU", "radius calculation tutorial", False),   # 짧은 영문 이름은 단어 경계가 필요하다
        ("아이유", "아이유(IU) - 밤편지", True),
        ("V", "V 'Slow Dancing' Official MV", True),
        ("V", "Very old song", False),
    ],
)
def test_short_latin_names_need_a_word_boundary(name, text, expected) -> None:
    assert cr.name_in_text(name, text) is expected


@pytest.mark.parametrize(
    "info, reason",
    [
        ({"artists": ["다른가수"], "title": "밤편지", "channel": "다른가수"}, "영상 아티스트 불일치"),
        ({"artists": ["IU"], "title": "아이유 - 밤편지", "channel": "x"}, ""),   # 제목에 근거가 있으면 통과
        ({"artists": ["아이유"], "title": "밤편지"}, ""),                        # 크레딧 자체가 근거다
        # 한계: 멜론이 '아이유'만 주고 유튜브 크레딧이 'IU'뿐이면 다른 가수로 본다. 제목·채널에
        # 이름이 있거나 멜론 표기가 '아이유 (IU)'면 통과한다. 거른 곡은 failed_songs.csv에 남는다.
        ({"artists": ["IU"], "title": "밤편지"}, "영상 아티스트 불일치"),
        ({"track": "밤편지 (Duet Ver.)", "title": "아이유 - 밤편지"}, "영상 트랙 불일치"),
        ({"title": "밤편지 lyrics"}, ""),                                       # 크레딧이 없으면 거르지 않는다
    ],
)
def test_full_info_rejects_only_positive_contradictions(info, reason) -> None:
    result = cr.video_mismatch(info, "밤편지", ["아이유"])
    assert result.startswith(reason) if reason else result == ""


# --- 2. 댓글 비활성화 -------------------------------------------------------------------

class FakeYDL:
    """검색 질의에는 entries를, watch URL에는 videos[id]를 돌려준다.

    yt-dlp의 지연 댓글 수집을 그대로 흉내 낸다. process=False면 댓글 대신
    __post_extractor(호출 가능)만 남긴다 — 검증에서 탈락한 영상은 이것을 부르지 않는다.
    """

    entries: list = []
    videos: dict = {}
    comment_calls: list = []

    def __init__(self, opts=None):
        self.opts = opts or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, query, download=False, process=True):
        if query.startswith("ytsearch"):
            return {"entries": list(self.entries)}
        video_id = query.split("v=")[-1]
        info = dict(self.videos[video_id])
        info.setdefault("webpage_url", query)
        comments = info.pop("comments", None)
        if process:
            FakeYDL.comment_calls.append(video_id)
            if isinstance(comments, Exception):
                raise comments
            info["comments"] = comments
            return info

        def post_extractor():
            FakeYDL.comment_calls.append(video_id)
            if isinstance(comments, Exception):
                raise comments
            return {"comments": comments}

        info["__post_extractor"] = post_extractor
        return info


@pytest.fixture
def fake_ydl(monkeypatch):
    def install(entries, videos):
        FakeYDL.entries, FakeYDL.videos = entries, videos
        FakeYDL.comment_calls = []
        monkeypatch.setattr(cr.yt_dlp, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(cr, "select_emotional_comments_with_llm",
                            lambda cands, **kw: [c["text"] for c in cands])
    return install


def test_disabled_comments_do_not_discard_a_valid_audio_source(fake_ydl) -> None:
    """오디오는 검증을 통과한 첫 영상, 댓글은 댓글이 있는 다음 영상에서 보충한다."""
    fake_ydl(
        [entry("mv", "아이유 - 밤편지 Official MV", views=100), entry("lyr", "아이유 밤편지 lyrics", views=50)],
        {
            "mv": {"title": "아이유 - 밤편지 Official MV", "view_count": 100, "comments": None},
            "lyr": {"title": "아이유 밤편지 lyrics", "view_count": 50,
                    "comments": [{"text": "비 오는 새벽에 혼자 듣기 좋은 노래예요", "like_count": 3}]},
        },
    )
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지")
    assert result["video_url"] == "https://www.youtube.com/watch?v=mv"
    assert result["view_count"] == 100
    assert result["comments"] == ["비 오는 새벽에 혼자 듣기 좋은 노래예요"]
    assert result["comment_source_url"] == "https://www.youtube.com/watch?v=lyr"


def test_all_candidates_without_comments_still_yield_the_audio(fake_ydl) -> None:
    fake_ydl(
        [entry("mv", "아이유 - 밤편지 Official MV", views=100)],
        {"mv": {"title": "아이유 - 밤편지 Official MV", "view_count": 100, "comments": None}},
    )
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지")
    assert result["video_url"] == "https://www.youtube.com/watch?v=mv"
    assert result["comments"] == []
    assert result["artist_verified"] is True


def test_comment_fetch_failure_falls_through_to_the_next_candidate(fake_ydl) -> None:
    """댓글 수집이 터지면 그 후보만 버린다. 곡 전체를 실패로 만들지 않는다.

    예전에는 댓글 수집이 extract_info 안에서 일어나 그 예외가 곧 '이 후보 탈락'이었다.
    지연 수집으로 바꾼 뒤 예외를 잡지 않으면 유튜브 댓글 API가 한 번 끊길 때마다
    정상 음원이 있는 곡이 단계 실패로 남는다.
    """
    fake_ydl(
        [entry("bad", "아이유 - 밤편지 Official MV", views=900),
         entry("ok", "아이유 - 밤편지 Official Audio", views=10)],
        {
            "bad": {"title": "아이유 - 밤편지 Official MV", "view_count": 900,
                    "comments": RuntimeError("댓글 continuation 실패")},
            "ok": {"title": "아이유 - 밤편지 Official Audio", "view_count": 10,
                   "comments": [{"text": "비 오는 새벽에 혼자 듣기 좋은 노래예요", "like_count": 3}]},
        },
    )
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지")
    # 댓글이 터진 후보는 음원 출처로도 쓰지 않는다(예전 동작과 같다).
    assert result["video_url"] == "https://www.youtube.com/watch?v=ok"
    assert result["comments"] == ["비 오는 새벽에 혼자 듣기 좋은 노래예요"]


def test_every_candidate_failing_on_comments_is_reported_as_no_audio(fake_ydl) -> None:
    fake_ydl(
        [entry("bad", "아이유 - 밤편지 Official MV", views=900)],
        {"bad": {"title": "아이유 - 밤편지 Official MV", "view_count": 900,
                 "comments": RuntimeError("댓글 continuation 실패")}},
    )
    assert cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지") == {}


@pytest.mark.parametrize(
    "broken",
    [
        {"formats": []},
        {"formats": [{"format_id": "251"}]},                               # url 없는 껍데기
        {"formats": [{"format_id": "251", "url": "https://x/a", "has_drm": True}]},
    ],
    ids=["포맷없음", "url없는포맷", "DRM포맷만"],
)
def test_video_without_downloadable_audio_is_skipped(fake_ydl, broken) -> None:
    """process=True가 해 주던 포맷 검사를 대신한다.

    댓글을 아끼려고 process=False로 받으면 yt-dlp의 포맷 선택이 돌지 않는다. 예전에는 그
    단계가 내려받을 수 없는 영상에서 예외를 올려 다음 후보로 넘겨 줬다. 이 검사가 없으면
    그 영상이 음원으로 채택되고 assets 단계에서 실패하며, 재수집해도 같은 영상을 다시 고른다.
    """
    good_comments = [{"text": "비 오는 새벽에 혼자 듣기 좋은 노래예요", "like_count": 3}]
    fake_ydl(
        [entry("bad", "아이유 - 밤편지 Official MV", views=900),
         entry("ok", "아이유 - 밤편지 Official Audio", views=10)],
        {
            "bad": dict({"title": "아이유 - 밤편지 Official MV", "view_count": 900,
                         "comments": list(good_comments)}, **broken),
            "ok": {"title": "아이유 - 밤편지 Official Audio", "view_count": 10,
                   "comments": list(good_comments)},
        },
    )
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지")
    assert result["video_url"] == "https://www.youtube.com/watch?v=ok"
    assert cr.audio_unavailable({"title": "x", **broken})


def test_a_normal_video_with_formats_is_not_rejected() -> None:
    assert cr.audio_unavailable({"formats": [{"format_id": "251", "url": "https://x/a"}]}) == ""
    assert cr.audio_unavailable({}) == ""                        # 포맷 정보가 없으면 판단하지 않는다
    # yt-dlp가 받을 수 있는 것은 거르지 않는다. 예전(process=True)도 통과시켰다.
    assert cr.audio_unavailable({"live_status": "is_live",
                                 "formats": [{"url": "https://x/live.m3u8"}]}) == ""
    assert cr.audio_unavailable({"formats": [{"url": "https://x/a", "has_drm": "maybe"}]}) == ""
    assert cr.audio_unavailable({"formats": [{"fragments": [{"path": "a"}]}]}) == ""


def test_other_artist_in_full_info_is_skipped_for_the_next_candidate(fake_ydl) -> None:
    """flat 검색에는 이름이 없어 통과했더라도 상세 크레딧이 다른 가수면 다음 후보로 간다."""
    fake_ydl(
        [entry("bad", "밤편지 Official MV", views=900), entry("ok", "밤편지 Official Audio", views=10)],
        {
            "bad": {"title": "밤편지 Official MV", "artists": ["다른가수"], "view_count": 900, "comments": []},
            "ok": {"title": "밤편지 Official Audio", "artists": ["아이유"], "view_count": 10, "comments": []},
        },
    )
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지")
    assert result["video_url"] == "https://www.youtube.com/watch?v=ok"


# --- 3. audio.m4a 강제 -------------------------------------------------------------------

class DownloadYDL:
    """download()가 주어진 확장자의 파일을 outtmpl 자리에 쓴다."""

    ext = "m4a"
    body = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 200

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def download(self, urls):
        stem = self.opts["outtmpl"].replace(".%(ext)s", "")
        Path(stem + "." + self.ext).write_bytes(self.body)


def test_webm_result_is_a_failure_and_leaves_no_fragment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr.yt_dlp, "YoutubeDL", DownloadYDL)
    monkeypatch.setattr(DownloadYDL, "ext", "webm")
    target = tmp_path / "audio.m4a"
    assert cr.download_youtube_audio("https://www.youtube.com/watch?v=x", str(target)) is False
    assert list(tmp_path.iterdir()) == []


def test_m4a_result_succeeds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr.yt_dlp, "YoutubeDL", DownloadYDL)
    monkeypatch.setattr(DownloadYDL, "ext", "m4a")
    target = tmp_path / "audio.m4a"
    assert cr.download_youtube_audio("https://www.youtube.com/watch?v=x", str(target)) is True
    assert [p.name for p in tmp_path.iterdir()] == ["audio.m4a"]
    assert cr.looks_like_m4a(target) is True


def test_download_format_depends_on_ffmpeg(monkeypatch) -> None:
    """ffmpeg가 없으면 m4a만 요청해 webm이 생길 여지를 없애고, 있으면 변환 후처리를 붙인다."""
    monkeypatch.setattr(cr.shutil, "which", lambda name: None)
    opts = cr.audio_download_options("/tmp/audio")
    assert opts["format"] == "bestaudio[ext=m4a]"
    assert "postprocessors" not in opts

    monkeypatch.setattr(cr.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    opts = cr.audio_download_options("/tmp/audio")
    assert opts["format"] == "bestaudio[ext=m4a]/bestaudio"
    assert opts["postprocessors"] == [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}]


def test_download_uses_node_runtime() -> None:
    """댓글 수집과 같은 JS 엔진을 쓴다. 없으면 yt-dlp가 지원 중단된 방식으로만 받는다."""
    assert cr.audio_download_options("/tmp/audio")["js_runtimes"] == {"node": {}}


class FlakyYDL(DownloadYDL):
    """앞의 fail_times번은 403을 내고 그다음 정상으로 받는다."""

    fail_times = 0
    error = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    calls = 0

    def download(self, urls):
        type(self).calls += 1
        if type(self).calls <= type(self).fail_times:
            stem = self.opts["outtmpl"].replace(".%(ext)s", "")
            Path(stem + ".m4a.part").write_bytes(b"partial")
            raise cr.yt_dlp.utils.DownloadError(type(self).error)
        super().download(urls)


@pytest.fixture
def flaky(monkeypatch):
    monkeypatch.setattr(cr.yt_dlp, "YoutubeDL", FlakyYDL)
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    monkeypatch.setattr(FlakyYDL, "calls", 0)
    return FlakyYDL


def test_403_is_retried_then_succeeds(tmp_path, flaky, monkeypatch) -> None:
    monkeypatch.setattr(flaky, "fail_times", 2)
    target = tmp_path / "audio.m4a"
    assert cr.download_youtube_audio("https://www.youtube.com/watch?v=x", str(target)) is True
    assert flaky.calls == 3
    assert [p.name for p in tmp_path.iterdir()] == ["audio.m4a"]   # 실패 조각은 남지 않는다


def test_403_gives_up_after_attempts(tmp_path, flaky, monkeypatch) -> None:
    monkeypatch.setattr(flaky, "fail_times", 99)
    target = tmp_path / "audio.m4a"
    assert cr.download_youtube_audio("https://www.youtube.com/watch?v=x", str(target)) is False
    assert flaky.calls == cr.AUDIO_DOWNLOAD_ATTEMPTS
    assert list(tmp_path.iterdir()) == []


def test_non_403_error_is_not_retried(tmp_path, flaky, monkeypatch) -> None:
    monkeypatch.setattr(flaky, "fail_times", 99)
    monkeypatch.setattr(flaky, "error", "ERROR: [youtube] x: Video unavailable")
    target = tmp_path / "audio.m4a"
    assert cr.download_youtube_audio("https://www.youtube.com/watch?v=x", str(target)) is False
    assert flaky.calls == 1


# --- 제목이 비슷한 다른 곡의 영상 --------------------------------------------------------
# 멜론을 맞게 골라도 유튜브에서 다른 곡을 선택할 수 있었다. 곡 이름을 부분 문자열로 비교해서
# '좋은날'이 '사랑하기 좋은날'에, '니 소식'이 '니 소식2'에, 'Love'가 'Love Wins'에 들어
# 있었기 때문이다. 조회수가 더 높으면 그 영상이 선택돼, 멜론 메타·가사에 다른 곡의 오디오가 붙는다.

WRONG_SONG_CASES = [
    ("유리상자", "좋은날", "사랑하기 좋은날"),
    ("송하예", "니 소식", "니 소식2"),
    ("윤현석", "Love", "Love Wins"),
]


def _wrong_and_right(artist, melon, wrong):
    """잘못된 영상의 조회수가 훨씬 높은 검색 결과. 상세 track에도 다른 제목이 들어 있다."""
    entries = [
        entry("bad", f"{artist} - {wrong} Official MV", channel=artist, views=9_000_000),
        entry("ok", f"{artist} - {melon} Official MV", channel=artist, views=50_000),
    ]
    videos = {
        "bad": {"title": entries[0]["title"], "view_count": 9_000_000, "track": wrong,
                "artists": [artist], "comments": [{"text": "비 오는 새벽에 듣기 좋은 노래", "like_count": 3}]},
        "ok": {"title": entries[1]["title"], "view_count": 50_000, "track": melon,
               "artists": [artist], "comments": [{"text": "비 오는 새벽에 듣기 좋은 노래", "like_count": 3}]},
    }
    return entries, videos


@pytest.mark.parametrize("artist, melon, wrong", WRONG_SONG_CASES)
def test_higher_view_count_does_not_win_for_a_different_song(artist, melon, wrong, fake_ydl) -> None:
    entries, videos = _wrong_and_right(artist, melon, wrong)
    fake_ydl(entries, videos)
    result = cr.fetch_youtube_reaction(artist, melon, melon_title=melon, known_artists=[artist])
    assert result["video_url"].endswith("v=ok"), result.get("video_title")
    assert wrong not in result["video_title"]


@pytest.mark.parametrize("artist, melon, wrong", WRONG_SONG_CASES)
def test_only_a_different_song_yields_no_video(artist, melon, wrong, fake_ydl) -> None:
    """원곡 영상이 없으면 수집하지 않는다. 다른 곡의 오디오를 붙이느니 미수집이 낫다."""
    entries, videos = _wrong_and_right(artist, melon, wrong)
    fake_ydl(entries[:1], {"bad": videos["bad"]})
    assert cr.fetch_youtube_reaction(artist, melon, melon_title=melon, known_artists=[artist]) == {}


@pytest.mark.parametrize(
    "melon, video_title",
    [
        ("밤편지", "[MV] IU(아이유) _ Through the Night(밤편지)"),
        ("야생화", "박효신(Park Hyo Shin) - 야생화(Wild Flower) Special Video"),
        ("너랑 나", "IU - 04. 너랑 나 (You & I)"),
        ("주지마", "주지마 Don't"),
        ("Fine", "TAEYEON 태연 'Fine' MV"),
        ("하늘위로 (Remix)", "렉시 - 하늘위로 Remix MV"),
        ("사랑을 했다 (LOVE SCENARIO)", "iKON - '사랑을 했다(LOVE SCENARIO)' M/V"),
    ],
)
def test_real_official_video_titles_still_pass(melon, video_title) -> None:
    """실제 공식 영상 제목들. 가수명·[MV]·영어 제목·수록 순서가 섞여 있어도 통과해야 한다."""
    assert cr.title_conflict(melon, video_title, ["아이유", "박효신", "로꼬", "태연 (TAEYEON)", "렉시", "iKON"]) == ""


@pytest.mark.parametrize(
    "melon, video_title",
    [
        ("Bad Boy", "Red Velvet 레드벨벳 'RBB (Really Bad Boy)' MV"),
        ("Marry Me", "마크툽(MAKTUB), 유연정 - Marry You [Marry Me Part.2]"),
        ("사랑 사랑아", "[MV] DAVICHI(다비치) - This Love(이 사랑)"),
    ],
)
def test_similar_but_different_songs_are_excluded(melon, video_title) -> None:
    """부분일치로 통과하던 실제 사례들. 검색 결과에서 직접 확인한 제목이다."""
    assert cr.title_conflict(melon, video_title, ["Red Velvet", "마크툽 (MAKTUB)", "다비치"]) != ""


@pytest.mark.parametrize(
    "melon, video_title",
    [
        # 한글이 없어 '한글 덩어리' 변형으로는 구제되지 않는 경우들.
        # 시드와 영상의 버전이 같아야 한다 — 버전이 다르면 그건 버전 검사가 따로 거른다.
        ("Butter (Remix)", "BTS - Butter Remix MV"),          # 버전 표시가 맨 단어로 붙음
        ("Through the Night", "IU - 04. Through the Night"),  # 앨범 수록 순서가 앞에 붙음
    ],
)
def test_latin_titles_need_marker_and_track_number_stripping(melon, video_title) -> None:
    """영어 제목은 한글 덩어리 변형의 도움을 못 받는다. 버전 표시와 수록 순서를 직접 떼어야 한다."""
    assert cr.title_conflict(melon, video_title, ["BTS", "IU"]) == ""


def test_a_remix_video_is_still_excluded_for_an_original_seed() -> None:
    """이름이 맞아도 버전이 다르면 거른다. 위 테스트가 버전 검사를 무력화하지 않았는지 확인한다."""
    assert cr.title_conflict("Butter", "BTS - Butter Remix MV", ["BTS"]) != ""


def test_track_metadata_catches_a_song_the_title_hides(fake_ydl) -> None:
    """영상 제목은 맞는데 유튜브 뮤직 track이 다른 곡이면 거른다.

    제목은 업로더가 쓰지만 track은 음원 메타데이터라 더 믿을 만하다.
    """
    entries = [entry("bad", "아이유 - 밤편지 Official MV", channel="아이유", views=9_000_000)]
    fake_ydl(entries, {"bad": {"title": entries[0]["title"], "view_count": 9_000_000,
                               "track": "좋은 날", "artists": ["아이유"], "comments": []}})
    assert cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지",
                                     known_artists=["아이유"]) == {}


def test_track_in_another_language_is_not_a_conflict(fake_ydl) -> None:
    """국내 곡이 영어 제목으로 등록되기도 한다. 영상 제목에 둘 다 있으면 같은 곡이다."""
    title = "[MV] IU(아이유) _ Through the Night(밤편지)"
    entries = [entry("ok", title, channel="1theK", views=100)]
    fake_ydl(entries, {"ok": {"title": title, "view_count": 100, "track": "Through the Night",
                              "artists": ["IU"], "comments": []}})
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지",
                                       known_artists=["아이유 (IU)"])
    assert result.get("video_url", "").endswith("v=ok"), result


# --- 아포스트로피: 단어 안의 것은 구분자가 아니다 -------------------------------------------
# 무조건 나누면 "I'm Fine"과 "I'm Sorry"가 'i'를, "Don't Cry"와 "Don't Say Goodbye"가
# 'don'을 공통 조각으로 갖게 되어 다른 곡이 같은 제목으로 판정됐다.

@pytest.mark.parametrize(
    "melon, video_title",
    [
        ("I'm Fine", "백지영 - I'm Sorry Official MV"),
        ("Don't Cry", "박효신 - Don't Say Goodbye MV"),
        ("I'm Fine", "백지영 - I'm Yours MV"),
    ],
)
def test_word_internal_apostrophe_does_not_create_a_shared_title(melon, video_title) -> None:
    assert cr.title_conflict(melon, video_title, ["백지영", "박효신"]) != ""


@pytest.mark.parametrize(
    "melon, video_title",
    [
        ("I'm Fine", "백지영 - I'm Fine Official MV"),
        ("Don't Cry", "박효신 - Don't Cry MV"),
        ("Fine", "TAEYEON 태연 'Fine' MV"),          # 제목을 감싼 따옴표는 구분자다
    ],
)
def test_quotes_around_a_title_still_split(melon, video_title) -> None:
    assert cr.title_conflict(melon, video_title, ["백지영", "박효신", "태연 (TAEYEON)"]) == ""


@pytest.mark.parametrize(
    "melon, video_title",
    [
        ("봄", "아이유 - 봄 MV"),
        ("길", "god - 길 M/V"),
        # Topic 채널은 영상 제목이 곡 이름 그 자체다. 조각이 한 글자여도 버리면 안 된다.
        ("봄", "봄"),
        ("길", "길"),
        # 부가 문구 없이 한 글자 조각만 분리되는 경우. 길이만 보고 버리면 곡을 잃는다.
        ("길", "god - 길"),
        ("I", "TAEYEON 태연 'I' MV"),
        ("I", "태연 - I (Feat. 버벌진트)"),
    ],
)
def test_one_character_titles_are_not_dropped(melon, video_title) -> None:
    """쪼개서 나온 한 글자 조각('M/V' -> m, v)은 버리되, 찾는 제목과 같은 조각은 살린다."""
    assert cr.title_conflict(melon, video_title, ["아이유", "god", "태연 (TAEYEON)"]) == ""


def test_a_split_fragment_cannot_match_a_one_character_title() -> None:
    """'M/V'의 'v'가 제목 'V'와 맞아떨어지면 안 된다."""
    assert cr.title_conflict("V", "아이유 - 밤편지 M/V", ["아이유"]) != ""


# --- 번역 예외에는 근거가 필요하다 -----------------------------------------------------------

def test_same_language_track_mismatch_is_rejected() -> None:
    """영상 제목에 둘 다 나온다는 것만으로 번역 관계를 인정하면 후속곡이 통과한다."""
    info = {"title": "송하예 - 니 소식 / 니 소식2", "track": "니 소식2", "artists": ["송하예"]}
    assert cr.video_mismatch(info, "니 소식", ["송하예"]) != ""


def test_different_script_track_is_accepted_as_a_translation() -> None:
    info = {"title": "[MV] IU(아이유) _ Through the Night(밤편지)",
            "track": "Through the Night", "artists": ["IU"]}
    assert cr.video_mismatch(info, "밤편지", ["아이유 (IU)"]) == ""


# --- 편곡·커버 판정 후보는 되살리지 않는다 ---------------------------------------------------

def test_an_arrangement_only_result_yields_no_video(fake_ydl) -> None:
    """후보가 편곡 영상 하나뿐이면 수집하지 않는다. 되살리면 원곡 자리에 다른 연주가 들어간다."""
    entries = [entry("p", "아이유 - 밤편지 piano", channel="누군가", views=9_000)]
    fake_ydl(entries, {"p": {"title": entries[0]["title"], "view_count": 9_000,
                             "artists": ["아이유"], "comments": []}})
    assert cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지",
                                     known_artists=["아이유"]) == {}


def test_a_broadcast_only_result_is_restored(fake_ydl) -> None:
    """방송 영상은 오디오가 원곡일 수 있다. 다른 후보가 없으면 되살린다."""
    entries = [entry("k", "아이유 - 밤편지 [열린음악회] KBS", channel="KBS", views=9_000)]
    fake_ydl(entries, {"k": {"title": entries[0]["title"], "view_count": 9_000,
                             "artists": ["아이유"], "comments": []}})
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지",
                                       known_artists=["아이유"])
    assert result.get("video_url", "").endswith("v=k"), result


@pytest.mark.parametrize(
    "melon, video_title",
    [("Piano Man", "빌리 조엘 - Piano Man Official MV"), ("시간", "이적 - 시간 Official MV")],
)
def test_keywords_inside_the_song_title_do_not_exclude_it(melon, video_title) -> None:
    """제외 키워드와 같은 말이 곡 이름인 경우가 있다. 곡 이름 부분은 검사에서 뺀다."""
    picked = cr.select_candidates([entry("a", video_title, views=10)], melon, ["빌리 조엘", "이적"])
    assert [e["id"] for e in picked] == ["a"]


# --- 요청한 버전 표시는 제외 사유가 아니다 ---------------------------------------------------
# 코퍼스에도 요청 자체가 리믹스판인 곡이 있다(렉시 '하늘위로 (Remix)' 1621412,
# 비 '태양을 피하는 방법 (Gtr.Remix)' 490059). 키워드 필터가 그 곡을 버리면 안 된다.

@pytest.mark.parametrize(
    "melon, video_title, artist",
    [
        ("하늘위로 (Remix)", "렉시 - 하늘위로 Remix MV", "렉시"),
        ("태양을 피하는 방법 (Gtr.Remix)", "비 - 태양을 피하는 방법 Gtr.Remix", "비"),
    ],
)
def test_a_requested_remix_is_not_excluded_by_the_keyword_filter(melon, video_title, artist) -> None:
    picked = cr.select_candidates([entry("a", video_title, channel=artist, views=100)],
                                  melon, [artist])
    assert [e["id"] for e in picked] == ["a"]


def test_an_unrequested_remix_is_still_excluded() -> None:
    """요청하지 않은 편곡은 그대로 막는다. 위 예외가 편곡 차단을 무력화하지 않았는지 본다."""
    picked = cr.select_candidates([entry("a", "아이유 - 밤편지 Remix", views=100)],
                                  "밤편지", ["아이유"])
    assert picked == []


@pytest.mark.parametrize(
    "artist, melon, video_title",
    [
        ("god", "길", "god - 길"),
        ("태연", "I", "TAEYEON 태연 'I' MV"),
        ("렉시", "하늘위로 (Remix)", "렉시 - 하늘위로 Remix MV"),
    ],
)
def test_these_songs_survive_the_whole_selection_flow(artist, melon, video_title, fake_ydl) -> None:
    """부분 검사만으로는 부족하다. 영상 선택 전체를 돌려 {}가 아닌지 확인한다."""
    entries = [entry("a", video_title, channel=artist, views=1000)]
    fake_ydl(entries, {"a": {"title": video_title, "view_count": 1000, "artists": [artist],
                             "comments": [{"text": "비 오는 새벽에 듣기 좋은 노래", "like_count": 3}]}})
    result = cr.fetch_youtube_reaction(artist, melon, melon_title=melon,
                                       known_artists=[artist, "태연 (TAEYEON)"])
    assert result.get("video_title") == video_title, result


def test_strip_requested_versions_only_removes_what_the_seed_asked_for() -> None:
    """요청한 버전 표시만 뺀다. 요청하지 않은 편곡 표시까지 빼면 키워드 필터가 무력해진다.

    (요청하지 않은 편곡은 버전 검사가 먼저 막지만, 키워드 필터도 제 몫을 해야 한다.)
    """
    from src.crawler.scripts_py.melon_match import strip_requested_versions

    asked = strip_requested_versions("하늘위로 Remix MV", "하늘위로 (Remix)")
    assert "remix" not in asked.lower()

    not_asked = strip_requested_versions("밤편지 Remix MV", "밤편지")
    assert "remix" in not_asked.lower()

    # 편곡 표시가 아닌 키워드는 어느 쪽이든 남는다
    assert "piano" in strip_requested_versions("밤편지 piano", "밤편지 (Remix)").lower()


def test_comments_are_not_fetched_for_a_rejected_video(fake_ydl) -> None:
    """탈락할 영상에 댓글 요청(최대 1,000개)을 쓰지 않는다.

    상세 메타데이터만 먼저 받아 검증하고, 통과한 영상에서만 댓글을 받는다.
    """
    # 검색 결과로는 구분되지 않고 **상세의 track에서만** 다른 곡으로 드러나는 영상.
    # 조회수가 높아 먼저 시도된다.
    entries = [
        entry("bad", "아이유 - 밤편지 Official MV", channel="아이유", views=9_000_000),
        entry("ok", "아이유(IU) - 밤편지 MV", channel="아이유", views=100),
    ]
    fake_ydl(entries, {
        "bad": {"title": entries[0]["title"], "view_count": 9_000_000, "artists": ["아이유"],
                "track": "좋은 날",
                "comments": [{"text": "이 영상 댓글은 받지 말아야 한다", "like_count": 1}]},
        "ok": {"title": entries[1]["title"], "view_count": 100, "artists": ["아이유"],
               "track": "밤편지",
               "comments": [{"text": "비 오는 새벽에 듣기 좋은 노래", "like_count": 3}]},
    })
    result = cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지",
                                       known_artists=["아이유"])
    assert result["video_url"].endswith("v=ok")
    assert FakeYDL.comment_calls == ["ok"], FakeYDL.comment_calls


def test_comments_are_fetched_once_for_the_accepted_video(fake_ydl) -> None:
    entries = [entry("ok", "아이유 - 밤편지 Official MV", channel="아이유", views=100)]
    fake_ydl(entries, {"ok": {"title": entries[0]["title"], "view_count": 100, "artists": ["아이유"],
                              "comments": [{"text": "비 오는 새벽에 듣기 좋은 노래", "like_count": 3}]}})
    cr.fetch_youtube_reaction("아이유", "밤편지", melon_title="밤편지", known_artists=["아이유"])
    assert FakeYDL.comment_calls == ["ok"]
