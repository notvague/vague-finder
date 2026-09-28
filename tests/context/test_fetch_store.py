import gzip
import tempfile
import unittest
from pathlib import Path

import requests

from src.crawler.context.fetcher import Fetcher, article_url, retry_time
from src.crawler.context.schemas import SourceSnapshot
from src.crawler.context.store import Store

URL = "https://namu.wiki/w/노래B"
HTML = '<article><h1>노래B</h1><p>가수A의 곡 노래B다.</p></article>'.encode()


class Response:
    def __init__(self, code, body=b"", headers=None):
        self.status_code, self.body = code, body
        self.headers = headers if headers is not None else {"Content-Type": "text/html; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.body


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers, self.calls = {}, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass


def robots(body=b"User-agent: *\nAllow: /\n"):
    return Response(200, body, {"Content-Type": "text/plain"})


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "context")

    def fetcher(self, responses, **kwargs):
        session = Session(responses)
        return Fetcher(self.store, session=session, interval=0, **kwargs), session

    def test_cache_idempotence_and_saved_html_not_http_success(self):
        ref, source = self.store.save_snapshot(HTML, requested_url=URL, acquisition_method="user_saved_html")
        same_ref, _ = self.store.save_snapshot(HTML, requested_url=URL, acquisition_method="user_saved_html")
        self.assertEqual(ref, same_ref)
        self.assertIsNone(source.http_status)
        self.assertIsNone(source.final_url)
        fetcher, session = self.fetcher([])
        cached = fetcher.fetch(URL)
        self.assertEqual(cached.status, "ok")
        self.assertEqual(cached.acquisition_method, "user_saved_html")
        self.assertIsNone(cached.http_status)
        self.assertEqual(session.calls, [])

    def test_robots_checked_before_page_and_lowercase_headers_supported(self):
        fetcher, session = self.fetcher([robots(), Response(200, HTML, {"content-type": "text/html", "etag": "abc"})])
        result = fetcher.fetch(URL, allow_http=True)
        self.assertEqual(result.status, "ok")
        self.assertEqual(session.calls[0][0], "https://namu.wiki/robots.txt")
        snapshot = SourceSnapshot.model_validate(self.store.read(result.snapshot_ref))
        self.assertEqual(snapshot.etag, "abc")
        self.assertEqual(snapshot.http_status, 200)

    def test_robots_disallow_stops_other_requests_and_persists_pause(self):
        fetcher, session = self.fetcher([robots(b"User-agent: *\nDisallow: /w/\n")])
        first = fetcher.fetch(URL, allow_http=True)
        second = fetcher.fetch("https://namu.wiki/w/다른노래", allow_http=True)
        self.assertEqual(first.error_code, "robots_disallowed")
        self.assertEqual(second.status, "pending")
        self.assertEqual(len(session.calls), 1)
        new_fetcher, new_session = self.fetcher([robots(b"User-agent: *\nDisallow: /w/\n")])
        self.assertEqual(new_fetcher.fetch(URL, allow_http=True).status, "blocked")
        self.assertEqual(len(new_session.calls), 1)

    def test_access_and_challenge_stop_run(self):
        for response in (Response(403), Response(401), Response(200, b'<title>Just a moment...</title>')):
            with self.subTest(response=response.status_code):
                with tempfile.TemporaryDirectory() as directory:
                    store = Store(Path(directory))
                    session = Session([robots(), response])
                    fetcher = Fetcher(store, session=session, interval=0)
                    self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "blocked")
                    self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "pending")
                    self.assertEqual(len(session.calls), 2)

    def test_retry_after_does_not_sleep_or_retry(self):
        sleeps = []
        fetcher, session = self.fetcher([robots(), Response(429, headers={"Retry-After": "7200"})], sleep=sleeps.append)
        result = fetcher.fetch(URL, allow_http=True)
        self.assertEqual(result.status, "rate_limited")
        self.assertEqual(result.retry_after, "7200")
        self.assertTrue(result.retry_at)
        self.assertEqual(sleeps, [])
        self.assertEqual(result.requests_made, 2)

    def test_http_404_distinct_from_network_error(self):
        fetcher, _ = self.fetcher([robots(), Response(404)])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "not_found")

    def test_transient_retry_bounded(self):
        fetcher, session = self.fetcher([robots(), Response(503), Response(503), Response(503)])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "network_error")
        self.assertEqual(len(session.calls), 4)

    def test_timeout_retries_and_request_budget(self):
        fetcher, session = self.fetcher([robots(), requests.Timeout(), requests.Timeout(), requests.Timeout()])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "network_error")
        self.assertEqual(len(session.calls), 4)
        fetcher, session = self.fetcher([robots()], max_requests=1)
        result = fetcher.fetch(URL, allow_http=True)
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.error_code, "request_budget")

    def test_offdomain_redirect_never_requested(self):
        fetcher, session = self.fetcher([robots(), Response(302, headers={"Location": "https://example.com/private"})])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "invalid_response")
        self.assertEqual(len(session.calls), 2)

    def test_valid_redirect_preserves_requested_and_final(self):
        target = "https://namu.wiki/w/노래B(가수A)"
        fetcher, session = self.fetcher([robots(), Response(302, headers={"Location": target}), Response(200, HTML)])
        result = fetcher.fetch(URL, allow_http=True)
        source = SourceSnapshot.model_validate(self.store.read(result.snapshot_ref))
        self.assertEqual(source.requested_url, URL)
        self.assertEqual(source.final_url, target)
        self.assertEqual(source.redirects, [target])

    def test_conditional_get_and_failed_refresh_keep_good_snapshot(self):
        fetcher, _ = self.fetcher([robots(), Response(200, HTML, {"Content-Type": "text/html", "ETag": "version-1"})])
        original = fetcher.fetch(URL, allow_http=True)
        fetcher, session = self.fetcher([robots(), Response(304)])
        result = fetcher.fetch(URL, allow_http=True, refresh=True)
        self.assertEqual(result.snapshot_ref, original.snapshot_ref)
        self.assertEqual(session.calls[-1][1]["headers"]["If-None-Match"], "version-1")
        fetcher, _ = self.fetcher([robots(), Response(404)])
        self.assertEqual(fetcher.fetch(URL, allow_http=True, refresh=True).status, "not_found")
        self.assertEqual(self.store.cached(URL), original.snapshot_ref)

    def test_non_html_oversized_and_invalid_robots(self):
        fetcher, _ = self.fetcher([robots(), Response(200, b"{}", {"Content-Type": "application/json"})])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "invalid_response")
        fetcher, _ = self.fetcher([Response(200, b"<html>not robots</html>")])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).error_code, "robots_invalid_response")
        fetcher, _ = self.fetcher([robots(), Response(200, b"x" * 10_000_001)])
        self.assertEqual(fetcher.fetch(URL, allow_http=True).status, "invalid_response")

    def test_url_validation(self):
        for url in ("http://namu.wiki/w/x", "https://namu.wiki.evil/w/x", "https://namu.wiki@evil/w/x",
                    "https://namu.wiki/w/파일:test", "https://namu.wiki/Search?q=x", "https://namu.wiki/w/x?q=y"):
            with self.assertRaises(ValueError):
                article_url(url)
        self.assertNotEqual(article_url("https://namu.wiki/w/거짓말(BIGBANG)"), article_url("https://namu.wiki/w/거짓말(빅뱅)"))


class StoreTests(unittest.TestCase):
    def test_writer_lock_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            with store.writer():
                with self.assertRaises(RuntimeError):
                    with store.writer():
                        pass
            self.assertFalse(store.path("writer.lock").exists())
            with self.assertRaises(ValueError):
                store.path("../outside.json")

    def test_hash_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            _, source = store.save_snapshot(HTML, requested_url=URL, acquisition_method="user_saved_html")
            store.write_bytes(source.raw_html_ref, gzip.compress(b"different text"))
            with self.assertRaisesRegex(ValueError, "hash"):
                store.raw(source)

    def test_immutable_writes_refuse_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            store.immutable("a.json", {"a": 1})
            store.immutable("a.json", {"a": 1})
            with self.assertRaises(ValueError):
                store.immutable("a.json", {"a": 2})


if __name__ == "__main__":
    unittest.main()
