import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

from src.crawler.context.fetcher import Fetcher
from src.crawler.context.schemas import FetchAttempt
from src.crawler.context.store import Store
from src.crawler.scripts_py.backfill_namuwiki_context import (
    candidate_target,
    main as backfill_main,
    make_parser,
    run_backfill,
    should_process,
)
from src.embedding.fixtures.meta_validation import validate_meta_document

BIGBANG_URL = "https://namu.wiki/w/거짓말(BIGBANG)"
RULES = b"User-agent: *\nDisallow: /\nAllow: /$\nAllow: /w/\nAllow: /skins/\n"
SHELL = b'<html><body><div>Loading...</div><script src="/skins/espejo/app.js"></script></body></html>'


def rendered_html(title="거짓말", artist="BIGBANG", *, trivia=True):
    body = (
        f"<h2>1. 개요[편집]</h2><p>{artist}의 곡 {title}이다.</p>"
        "<h2>2. 가사[편집]</h2><p>검색에 들어가면 안 되는 가사</p>"
    )
    if trivia:
        body += (
            "<h2>3. 여담[편집]</h2><ul>"
            "<li>뮤직비디오의 내용은 여자가 사건에 휘말리는 줄거리이다.</li>"
            "<li>원래는 솔로곡으로 만들었지만 그룹 타이틀곡으로 바뀌었다.</li>"
            "<li>리듬 게임에 수록되어 있다.</li></ul>"
            "<h3>3.1. 밈화[편집]</h3><p>인터넷 패러디가 화제가 되었다.</p>"
            "<blockquote>인용된 긴 대사는 저장하지 않는다.</blockquote>"
        )
    body += "<h2>4. 관련 문서[편집]</h2><p>다른 문서 설명</p>"
    nested = '<div class="opaque+a">' * 8 + body + "</div>" * 8
    return (
        f"<!doctype html><html><head><title>{title}({artist}) - 나무위키</title></head>"
        f"<body><header>사이트 메뉴</header><div><h1>{title}({artist})</h1>{nested}</div>"
        "<aside>추천 문서</aside><footer>사이트 푸터</footer></body></html>"
    ).encode()


class Response:
    def __init__(self, status, raw=b"", headers=None):
        self.status_code, self.raw = status, raw
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.raw


class Session:
    def __init__(self, *responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url, **kwargs):
        self.calls.append(url)
        if not self.responses:
            raise AssertionError("unexpected network request: " + url)
        return self.responses.pop(0)

    def close(self):
        pass


def robots():
    return Response(200, RULES, {"Content-Type": "text/plain; charset=utf-8"})


class Renderer:
    def __init__(self, raw=None):
        self.raw, self.calls = raw or rendered_html(), []

    def render(self, url, **kwargs):
        self.calls.append(url)
        kwargs["on_navigation"](url)
        return {
            "raw": self.raw,
            "final_url": url,
            "http_status": 200,
            "redirects": [],
            "browser_requests": 3,
        }

    def close(self):
        pass


class NeverFetcher:
    stopped = False
    requests_made = 0

    def fetch(self, *args, **kwargs):
        raise AssertionError("network/cache fetch must not run")

    def close(self):
        pass


class FailedFetcher(NeverFetcher):
    def fetch(self, url, **kwargs):
        return FetchAttempt(
            requested_url=url,
            acquisition_method="http",
            status="network_error",
            error_code="Timeout",
        )


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw_dir = self.root / "raw"
        self.cache_dir = self.root / "context"
        self.fixture = json.loads(
            (Path(__file__).parent / "fixtures/song_meta.json").read_text(encoding="utf-8-sig")
        )

    def write_song(self, meta=None, *, folder="BIGBANG_거짓말_1698598"):
        song_dir = self.raw_dir / folder
        song_dir.mkdir(parents=True)
        path = song_dir / "meta.json"
        value = copy.deepcopy(meta or self.fixture)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        (song_dir / "audio.m4a").write_bytes(b"audio-original")
        (song_dir / "cover.jpg").write_bytes(b"cover-original")
        return song_dir, path

    def args(self, *extra):
        return make_parser().parse_args([
            "--data-dir", str(self.raw_dir),
            "--cache-dir", str(self.cache_dir),
            "--interval", "0",
            *extra,
        ])

    def test_browser_collection_writes_only_small_top_level_schema(self):
        song_dir, path = self.write_song()
        renderer, session = Renderer(), Session(robots(), Response(200, SHELL))

        def factory(store, **kwargs):
            return Fetcher(store, session=session, renderer=renderer, **kwargs)

        report = run_backfill(self.args("--song-ids", "1698598"), fetcher_factory=factory)
        row = report["songs"][0]
        self.assertEqual((row["action"], row["status"], row["fact_count"]), ("updated", "ok", 4))
        updated = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(list(updated)[-1], "namuwiki")
        self.assertNotIn("external_context", updated)
        context = updated["namuwiki"]
        self.assertEqual(set(context), {"schema_version", "status", "source_url", "collected_at", "facts"})
        self.assertEqual(set(context["facts"][0]), {"category", "section", "text"})
        serialized = json.dumps(context, ensure_ascii=False)
        for unwanted in ("attempts", "snapshot_ref", "document_ref", "binding", "footnotes", "diagnostics"):
            self.assertNotIn(unwanted, serialized)
        self.assertEqual((song_dir / "audio.m4a").read_bytes(), b"audio-original")
        self.assertEqual((song_dir / "cover.jpg").read_bytes(), b"cover-original")

    def test_existing_verbose_bigbang_payload_migrates_without_fetch(self):
        meta = copy.deepcopy(self.fixture)
        meta["external_context"] = {
            "another_source": {"keep": True},
            "namuwiki": {
                "schema_version": "namuwiki_meta_v2",
                "status": "collected",
                "sources": [{
                    "url": BIGBANG_URL,
                    "page_title": "거짓말(BIGBANG)",
                    "blocks": [
                        {"section_path": ["개요"], "block_type": "paragraph", "text": "개요 설명"},
                        {"section_path": ["여담"], "block_type": "list_item",
                         "text": "뮤직비디오의 내용은 여자가 다른 사람 대신 감옥에 가는 이야기다."},
                        {"section_path": ["여담"], "block_type": "paragraph",
                         "text": "이 문서의 내용 중 전체 또는 일부는"},
                    ],
                }],
                "attempts": [{"huge": "internal"}],
                "diagnostics": [{"huge": "internal"}],
            },
        }
        _, path = self.write_song(meta)
        original = path.read_bytes()
        report = run_backfill(self.args(), fetcher_factory=lambda *a, **kw: NeverFetcher())
        self.assertEqual(report["songs"][0]["action"], "migrated")
        updated = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(updated["external_context"], {"another_source": {"keep": True}})
        self.assertEqual(updated["namuwiki"]["status"], "ok")
        self.assertEqual(len(updated["namuwiki"]["facts"]), 1)
        backup = self.cache_dir / report["songs"][0]["backup_ref"]
        self.assertEqual(backup.read_bytes(), original)

    def test_cached_rendered_love_page_retries_old_needs_review(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "837567"
        meta["metadata"]["title"] = "사랑했나봐"
        meta["metadata"]["artist"] = ["윤도현"]
        meta["external_context"] = {"namuwiki": {
            "schema_version": "namuwiki_meta_v2",
            "status": "needs_review",
            "attempts": [{"url": "https://namu.wiki/w/사랑했나봐(윤도현)"}],
            "sources": [],
        }}
        _, path = self.write_song(meta, folder="윤도현_사랑했나봐_837567")
        store = Store(self.cache_dir)
        store.save_snapshot(
            rendered_html("사랑했나봐", "윤도현"),
            requested_url="https://namu.wiki/w/사랑했나봐(윤도현)",
            final_url="https://namu.wiki/w/사랑했나봐(윤도현)",
            http_status=200,
            acquisition_method="browser_rendered",
        )
        session = Session()
        report = run_backfill(
            self.args("--song-ids", "837567"),
            fetcher_factory=lambda store, **kw: Fetcher(store, session=session, **kw),
        )
        self.assertEqual(report["songs"][0]["status"], "ok")
        self.assertEqual(session.calls, [])
        self.assertEqual(json.loads(path.read_text())["namuwiki"]["facts"][0]["category"], "music_video")

    def test_wrong_same_title_page_is_reviewed_without_persisting_source_url(self):
        _, path = self.write_song()
        store = Store(self.cache_dir)
        store.save_snapshot(
            rendered_html("다른 노래", "다른 가수"),
            requested_url=BIGBANG_URL,
            final_url=BIGBANG_URL,
            http_status=200,
            acquisition_method="browser_rendered",
        )
        report = run_backfill(
            self.args("--song-ids", "1698598"),
            fetcher_factory=lambda store, **kw: Fetcher(
                store,
                session=Session(),
                **kw,
            ),
        )
        row = report["songs"][0]
        self.assertEqual(row["action"], "needs_review")
        self.assertEqual(row["status"], "needs_review")
        self.assertEqual(row["error_code"], "page_title_mismatch")
        self.assertEqual(report["progress"]["failed_this_run"], 0)
        context = json.loads(path.read_text(encoding="utf-8"))["namuwiki"]
        self.assertNotIn("source_url", context)
        self.assertEqual(context["facts"], [])

    def test_no_trivia_and_not_found_are_terminal_but_core_remains_valid(self):
        _, path = self.write_song()
        Store(self.cache_dir).save_snapshot(
            rendered_html(trivia=False), requested_url=BIGBANG_URL,
            final_url=BIGBANG_URL, http_status=200, acquisition_method="browser_rendered",
        )
        report = run_backfill(
            self.args(),
            fetcher_factory=lambda store, **kw: Fetcher(store, session=Session(), **kw),
        )
        self.assertEqual(report["songs"][0]["status"], "no_trivia")
        updated = json.loads(path.read_text())
        self.assertEqual(updated["namuwiki"]["facts"], [])
        self.assertEqual(validate_meta_document(updated, require_media_files=False), [])
        again = run_backfill(self.args(), fetcher_factory=lambda *a, **kw: self.fail("must skip"))
        self.assertEqual(again["songs"][0]["action"], "skip")

        # A separate song with an explicit known URL can be absent without being
        # moved or rejected by the embedding validator.
        with tempfile.TemporaryDirectory() as directory:
            second = BackfillTests(methodName="runTest")
            second.root = Path(directory)
            second.raw_dir = second.root / "raw"
            second.cache_dir = second.root / "context"
            second.fixture = self.fixture
            song_dir, missing_path = second.write_song()
            session = Session(robots(), Response(404))
            missing = run_backfill(
                second.args(),
                fetcher_factory=lambda store, **kw: Fetcher(store, session=session, **kw),
            )
            self.assertEqual(missing["songs"][0]["status"], "not_found")
            missing_meta = json.loads(missing_path.read_text())
            self.assertEqual(validate_meta_document(missing_meta, song_dir=song_dir), [])
            self.assertTrue(song_dir.is_dir())

    def test_dry_run_and_song_selection_make_no_files_or_requests(self):
        self.write_song()
        meta2 = copy.deepcopy(self.fixture)
        meta2["song_id"] = "837567"
        meta2["metadata"]["title"], meta2["metadata"]["artist"] = "사랑했나봐", ["윤도현"]
        self.write_song(meta2, folder="윤도현_사랑했나봐_837567")
        report = run_backfill(
            self.args("--song-ids", "837567", "--dry-run"),
            fetcher_factory=lambda *a, **kw: self.fail("dry-run must not build fetcher"),
        )
        self.assertEqual(report["selected_count"], 1)
        self.assertEqual(report["songs"][0]["song_id"], "837567")
        self.assertFalse(self.cache_dir.exists())

    def test_failed_forced_refresh_keeps_previous_meta_exactly(self):
        meta = copy.deepcopy(self.fixture)
        meta["namuwiki"] = {
            "schema_version": "namuwiki_v1",
            "status": "ok",
            "source_url": BIGBANG_URL,
            "collected_at": "2026-01-01T00:00:00+00:00",
            "facts": [{"category": "other", "section": "여담", "text": "기존의 유효한 배경 정보다."}],
        }
        _, path = self.write_song(meta)
        original = path.read_bytes()
        report = run_backfill(
            self.args("--force"),
            fetcher_factory=lambda *a, **kw: FailedFetcher(),
        )
        self.assertEqual(report["songs"][0]["action"], "kept_previous")
        self.assertEqual(path.read_bytes(), original)

    def test_compact_v1_schema_is_migrated_not_silently_skipped(self):
        meta = {"namuwiki": {"schema_version": "namuwiki_v1", "status": "ok", "facts": []}}
        self.assertEqual(
            should_process(meta),
            ("migrate", "upgrade_compact_namuwiki_schema"),
        )

    def test_current_needs_review_waits_for_explicit_retry(self):
        meta = {
            "namuwiki": {
                "schema_version": "namuwiki_v3",
                "status": "needs_review",
                "collected_at": "2026-09-22T00:00:00+00:00",
                "facts": [],
                "error_code": "page_title_mismatch",
            }
        }
        self.assertEqual(
            should_process(meta),
            ("skip", "needs_review_manual_resolution"),
        )
        self.assertEqual(
            should_process(meta, retry_needs_review=True),
            ("collect", "retry_incomplete_or_error"),
        )

    def test_generated_urls_use_clean_artist_aliases(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "123456"
        target = candidate_target(meta, Path("BIGBANG_거짓말_123456"), {})
        self.assertEqual([unquote(url) for url in target.urls], [
            "https://namu.wiki/w/거짓말(BIGBANG)",
            "https://namu.wiki/w/거짓말(빅뱅)",
            "https://namu.wiki/w/거짓말(노래)",
            "https://namu.wiki/w/거짓말",
        ])

    def test_generated_urls_include_artist_name_without_internal_spaces(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "1615770"
        meta["metadata"].update({
            "title": "아리랑",
            "artist": ["SG 워너비"],
        })
        target = candidate_target(meta, Path("SG_워너비_아리랑_1615770"), {})

        decoded = [unquote(url) for url in target.urls]
        self.assertEqual(decoded[:2], [
            "https://namu.wiki/w/아리랑(SG 워너비)",
            "https://namu.wiki/w/아리랑(SG워너비)",
        ])
        self.assertIn("https://namu.wiki/w/아리랑(노래)", decoded)

    def test_space_alias_does_not_displace_parenthesized_local_artist(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "602362112"
        meta["metadata"].update({
            "title": "GO! GO!",
            "artist": ["ALPHA DRIVE ONE (알파드라이브원)"],
        })
        target = candidate_target(
            meta,
            Path("ALPHA_DRIVE_ONE_GO_GO_602362112"),
            {},
        )

        self.assertEqual([unquote(url) for url in target.urls[:2]], [
            "https://namu.wiki/w/GO! GO!(ALPHA DRIVE ONE)",
            "https://namu.wiki/w/GO! GO!(알파드라이브원)",
        ])

    def test_space_normalized_artist_candidate_collects_sg_wannabe_page(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "1615770"
        meta["metadata"].update({
            "title": "아리랑",
            "artist": ["SG 워너비"],
        })
        _, path = self.write_song(meta, folder="SG_워너비_아리랑_1615770")
        session = Session(
            robots(),
            Response(404),
            Response(200, rendered_html("아리랑", "SG워너비")),
        )

        report = run_backfill(
            self.args("--song-ids", "1615770"),
            fetcher_factory=lambda store, **kw: Fetcher(
                store,
                session=session,
                **kw,
            ),
        )

        row = report["songs"][0]
        self.assertEqual((row["action"], row["status"]), ("updated", "ok"))
        self.assertGreater(len(json.loads(path.read_text())["namuwiki"]["facts"]), 0)
        self.assertEqual([unquote(url) for url in session.calls[-2:]], [
            "https://namu.wiki/w/아리랑(SG 워너비)",
            "https://namu.wiki/w/아리랑(SG워너비)",
        ])

    def test_generic_candidate_collects_exact_multi_artist_credit_page(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "2534748"
        meta["metadata"].update({
            "title": "원더우먼",
            "artist": ["씨야"],
        })
        _, path = self.write_song(meta, folder="씨야_원더우먼_2534748")
        session = Session(
            robots(),
            Response(404),
            Response(
                200,
                rendered_html("원더우먼", "씨야, 다비치, 티아라"),
            ),
        )

        report = run_backfill(
            self.args("--song-ids", "2534748"),
            fetcher_factory=lambda store, **kw: Fetcher(
                store,
                session=session,
                **kw,
            ),
        )

        row = report["songs"][0]
        self.assertEqual((row["action"], row["status"]), ("updated", "ok"))
        self.assertGreater(len(json.loads(path.read_text())["namuwiki"]["facts"]), 0)
        self.assertEqual([unquote(url) for url in session.calls[-2:]], [
            "https://namu.wiki/w/원더우먼(씨야)",
            "https://namu.wiki/w/원더우먼(노래)",
        ])

    def test_target_carries_album_year_and_bounded_clean_title_fallbacks(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "1525541"
        meta["metadata"].update({
            "title": "별 (Original Dialog Ver.)",
            "artist": ["김아중"],
            "album": "미녀는 괴로워 OST",
            "release_date": "2006.12.14",
        })
        target = candidate_target(
            meta,
            Path("김아중_별_(Original_Dialog_Ver.)_1525541"),
            {},
        )

        decoded = [unquote(url) for url in target.urls]
        self.assertEqual(target.album, "미녀는 괴로워 OST")
        self.assertEqual(target.release_year, 2006)
        self.assertIn("별", target.title_aliases)
        self.assertLessEqual(len(decoded), 8)
        self.assertIn("https://namu.wiki/w/별(김아중)", decoded)
        self.assertIn("https://namu.wiki/w/별(노래)", decoded)

    def test_unknown_parenthetical_is_not_used_as_an_automatic_title_alias(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "123456"
        meta["metadata"].update({
            "title": "나와 같다면 (김장훈)",
            "artist": ["다른 가수"],
        })
        target = candidate_target(meta, Path("다른_가수_나와_같다면_123456"), {})
        self.assertNotIn("나와 같다면", target.title_aliases)
        self.assertFalse(any(
            unquote(url) == "https://namu.wiki/w/나와 같다면(다른 가수)"
            for url in target.urls
        ))

    def test_forced_recheck_keeps_previous_url_then_adds_safe_fallbacks(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "123456"
        meta["namuwiki"] = {
            "schema_version": "namuwiki_v3",
            "status": "no_trivia",
            "source_url": "https://namu.wiki/w/거짓말",
            "collected_at": "2026-09-22T00:00:00+00:00",
            "facts": [],
        }
        target = candidate_target(meta, Path("BIGBANG_거짓말_123456"), {})
        decoded = [unquote(url) for url in target.urls]
        self.assertEqual(decoded[0], "https://namu.wiki/w/거짓말")
        self.assertIn("https://namu.wiki/w/거짓말(BIGBANG)", decoded)
        self.assertIn("https://namu.wiki/w/거짓말(노래)", decoded)

    def test_generic_song_candidate_collects_artist_verified_page(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "30806580"
        meta["metadata"]["title"] = "instagram"
        meta["metadata"]["artist"] = ["DEAN"]
        _, path = self.write_song(meta, folder="DEAN_instagram_30806580")

        generic_page = rendered_html("instagram", "DEAN").replace(
            b"instagram(DEAN)",
            "instagram(노래)".encode(),
        )
        session = Session(
            robots(),
            Response(404),
            Response(200, generic_page),
        )
        report = run_backfill(
            self.args("--song-ids", "30806580"),
            fetcher_factory=lambda store, **kw: Fetcher(
                store,
                session=session,
                **kw,
            ),
        )

        row = report["songs"][0]
        self.assertEqual((row["action"], row["status"]), ("updated", "ok"))
        self.assertEqual(
            unquote(json.loads(path.read_text(encoding="utf-8"))["namuwiki"]["source_url"]),
            "https://namu.wiki/w/instagram(노래)",
        )
        self.assertEqual([unquote(url) for url in session.calls[-2:]], [
            "https://namu.wiki/w/instagram(DEAN)",
            "https://namu.wiki/w/instagram(노래)",
        ])

    def test_collection_scopes_disambiguation_page_to_matching_artist_song(self):
        meta = copy.deepcopy(self.fixture)
        meta["song_id"] = "31131274"
        meta["metadata"].update({
            "title": "Forever Young",
            "artist": ["BLACKPINK"],
            "album": "SQUARE UP",
            "release_date": "2018.06.15",
        })
        _, path = self.write_song(meta, folder="BLACKPINK_Forever_Young_31131274")
        raw = (
            "<!doctype html><html><head><title>Forever Young - 나무위키</title></head>"
            "<body><article><h1>Forever Young</h1>"
            "<h2>1. 음악</h2>"
            "<h3>1.1. BLACKPINK의 노래</h3><p>BLACKPINK의 수록곡이다.</p>"
            "<h4>1.1.1. 여담</h4><p>콘서트 무대에서 자주 공연한 곡이다.</p>"
            "<h3>1.2. 다른 가수의 노래</h3><p>다른 가수의 곡이다.</p>"
            "<h4>1.2.1. 여담</h4><p>인터넷 밈으로 유행한 곡이다.</p>"
            "</article></body></html>"
        ).encode()
        session = Session(robots(), Response(200, raw))

        report = run_backfill(
            self.args("--song-ids", "31131274"),
            fetcher_factory=lambda store, **kw: Fetcher(store, session=session, **kw),
        )

        self.assertEqual(report["songs"][0]["status"], "ok")
        context = json.loads(path.read_text(encoding="utf-8"))["namuwiki"]
        self.assertEqual(len(context["facts"]), 1)
        self.assertIn("콘서트", context["facts"][0]["text"])
        self.assertNotIn("밈", context["facts"][0]["text"])
        stored_report = json.loads(
            (self.cache_dir / report["report_ref"]).read_text(encoding="utf-8")
        )
        identity = stored_report["audits"][0]["attempts"][0]["identity"]
        self.assertTrue(identity["verified"])
        self.assertEqual(identity["reason"], "artist_song_subsection_match")
        self.assertIsNotNone(identity["scope_section_id"])

    def test_invalid_core_is_skipped_and_never_moved(self):
        meta = copy.deepcopy(self.fixture)
        meta["metadata"]["title"] = ""
        song_dir, _ = self.write_song(meta)
        report = run_backfill(self.args(), fetcher_factory=lambda *a, **kw: self.fail("must not fetch"))
        self.assertEqual(report["songs"][0]["action"], "skip")
        self.assertIn("metadata.title", report["songs"][0]["reason"])
        self.assertTrue(song_dir.exists())
        self.assertFalse((self.root / "failed_raw").exists())

    def test_cli_does_not_fail_for_request_budget_deferred_rows(self):
        result = {
            "songs": [{"action": "deferred", "status": "error"}],
            "artifact_manifest": {"invalid_artifact_count": 0},
        }
        with patch(
            "src.crawler.scripts_py.backfill_namuwiki_context.run_backfill",
            return_value=result,
        ):
            self.assertEqual(backfill_main([]), 0)


if __name__ == "__main__":
    unittest.main()
