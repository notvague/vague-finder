"""HTTP with bounded browser rendering for client shells; no challenge bypass."""
from __future__ import annotations

import time
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from protego import Protego
from bs4 import BeautifulSoup

import requests

from .schemas import FetchAttempt, SourceSnapshot, digest, now
from .store import Store

USER_AGENT = "Vague-Finder-Context/1.0 (song-context research collector)"
MAX_BYTES = 10_000_000


def needs_rendering(raw: bytes) -> bool:
    """Detect an empty JS app shell, not an article that happens to say 'Loading'."""
    soup = BeautifulSoup(raw, "html.parser")
    if not soup.find("script"):
        return False
    for node in soup.select("head, script, style, noscript, template"):
        if node.parent is not None:
            node.decompose()
    visible = soup.get_text(" ", strip=True).strip().casefold()
    return not soup.find(re.compile(r"^h[1-6]$")) and (
        visible in {"", "loading", "loading...", "loading…", "로딩 중", "로딩 중..."})


def article_url(value: str, *, strip_query: bool = False) -> str:
    parts = urlsplit(value)

    if (parts.scheme != "https" or parts.netloc.lower() != "namu.wiki"
            or not parts.path.startswith("/w/") or not parts.path[3:]
            ):
        raise ValueError("only https://namu.wiki/w/<document> URLs are allowed")

    if parts.query and not strip_query:
        raise ValueError("query strings are not allowed for seed article URLs")

    title = unquote(parts.path[3:])
    if title.startswith(("분류:", "파일:", "틀:", "사용자:", "토론:", "나무위키:")):
        raise ValueError("not a song-source namespace")
    
    if any(ord(char) < 32 for char in value):
        raise ValueError("invalid URL control character")
    
    return urlunsplit(("https", "namu.wiki", parts.path, "", ""))


def challenge(raw: bytes) -> bool:
    """Access screen indicators, not musical prose containing the word CAPTCHA."""
    text = raw[:500_000].decode("utf-8", errors="replace").lower()
    return any(marker in text for marker in (
        "<title>just a moment", "<title>access denied", "<title>robot challenge",
        "cf-chl-", "id=\"challenge-form\"", "id='challenge-form'", "challenges.cloudflare.com",
        "<title>접근 제한", "<title>접근 차단", "<title>captcha",
    ))


def retry_time(value: str | None) -> str:
    try:
        if value and value.strip().isdigit():
            return (datetime.now(timezone.utc) + timedelta(seconds=int(value))).isoformat()
        if value:
            parsed = parsedate_to_datetime(value)
            return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).isoformat()
    except (ValueError, OverflowError, TypeError):
        pass
    return (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()


class Fetcher:
    def __init__(self, store: Store, *, interval=3.0, timeout=20.0, max_requests=20,
                 session=None, sleep=time.sleep, clock=time.monotonic,
                 render=True, render_timeout=30.0, renderer=None):
        if not 0 <= interval <= 60 or not 0 < timeout <= 60 or max_requests < 1:
            raise ValueError("interval 0..60, timeout 0..60, max_requests >= 1 required")
        self.store, self.interval, self.timeout = store, interval, timeout
        self.max_requests = max_requests
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.8"})
        self.sleep, self.clock = sleep, clock
        self.last_request = None
        self.requests_made = 0
        self.stopped = False
        self.robots = None
        if not 0 < render_timeout <= 60:
            raise ValueError("render_timeout must be > 0 and <= 60")
        self.render_enabled, self.render_timeout = render, render_timeout
        self.renderer = renderer
        self._attempt_start = 0

    def close(self):
        try:
            self.session.close()
        finally:
            if self.renderer:
                self.renderer.close()

    def _reserve_request(self):
        if self.requests_made >= self.max_requests:
            raise RuntimeError("request_budget")
        wait = 0 if self.last_request is None else self.interval - (self.clock() - self.last_request)
        if wait > 60:
            raise RuntimeError("crawl_delay_exceeds_run_wait_budget")
        if wait > 0:
            self.sleep(wait)
        self.requests_made += 1
        self.last_request = self.clock()

    def _get(self, url: str, headers=None):
        self._reserve_request()
        with self.session.get(url, timeout=self.timeout, allow_redirects=False, stream=True,
                              headers=headers or {}) as response:
            raw = bytearray()
            for part in response.iter_content(chunk_size=65536):
                raw.extend(part)
                if len(raw) > MAX_BYTES:
                    raise ValueError("response_too_large")
            return response.status_code, requests.structures.CaseInsensitiveDict(response.headers), bytes(raw)

    def _stop(self, attempt: FetchAttempt):
        self.stopped = True
        attempt.requests_made = self.requests_made - self._attempt_start
        self.store.write("http_pause.json", attempt.model_dump())

    def _access_status(self, attempt, status, headers, raw):
        attempt.http_status = status
        if status in (401, 403) or challenge(raw):
            attempt.status, attempt.error_code = "blocked", "access_or_challenge"
            self._stop(attempt)
            return True
        if status == 429:
            attempt.status, attempt.error_code = "rate_limited", "http_429"
            attempt.retry_after = headers.get("Retry-After")
            attempt.retry_at = retry_time(attempt.retry_after)
            self._stop(attempt)
            return True
        return False

    def _robots_allowed(self, url, attempt):
        if self.robots is None:
            attempt.stage = "robots"
            status, headers, raw = self._get("https://namu.wiki/robots.txt")
            if self._access_status(attempt, status, headers, raw):
                attempt.http_status = status
                return False
            if status not in (200, 404):
                attempt.status, attempt.error_code = "pending", "robots_unavailable"
                self.stopped = True
                return False
            if status == 200 and (b"<html" in raw.lower() or b"<!doctype" in raw.lower()):
                attempt.status, attempt.error_code = "pending", "robots_invalid_response"
                self.stopped = True
                return False
            robots_text = raw.decode("utf-8", errors="replace") if status == 200 else ""
            self.robots = Protego.parse(robots_text)
            delay = self.robots.crawl_delay(USER_AGENT)
            rate = self.robots.request_rate(USER_AGENT)
            self.interval = max(self.interval, delay or 0, rate.seconds / rate.requests if rate else 0)
        if not self.robots.can_fetch(url, USER_AGENT):
            attempt.stage = "robots"
            attempt.status, attempt.error_code = "blocked", "robots_disallowed"
            self._stop(attempt)
            return False
        return True

    def _render_snapshot(self, url, initial_ref, attempt):
        from .browser_renderer import BrowserRenderer, RenderError
        attempt.initial_snapshot_ref = initial_ref
        attempt.stage = "render"
        if not self.render_enabled:
            attempt.status, attempt.error_code = "pending", "render_required_disabled"
            return attempt

        def navigation(target):
            target = article_url(target, strip_query=True)
            if not self._robots_allowed(target, attempt):
                raise RenderError("render_navigation_robots_disallowed", stop=True)
            self._reserve_request()
            return target

        self.renderer = self.renderer or BrowserRenderer()
        initial = SourceSnapshot.model_validate(self.store.read(initial_ref))
        try:
            rendered = self.renderer.render(initial.final_url or url, user_agent=USER_AGENT,
                timeout=self.render_timeout, on_navigation=navigation,
                is_challenge=challenge, max_bytes=MAX_BYTES)
            final_url = article_url(rendered["final_url"], strip_query=True)
            raw = rendered["raw"]
            if needs_rendering(raw):
                raise RenderError("render_returned_client_shell")
            if challenge(raw):
                raise RenderError("render_access_challenge", stop=True)
            ref, _ = self.store.save_snapshot(raw, requested_url=url,
                acquisition_method="browser_rendered", final_url=final_url,
                http_status=rendered["http_status"], redirects=rendered["redirects"],
                headers={"Content-Type": "text/html; charset=utf-8"}, parent_snapshot_ref=initial_ref)
            attempt.acquisition_method = "browser_rendered"
            attempt.http_status, attempt.final_url = rendered["http_status"], final_url
            attempt.redirects, attempt.browser_requests = rendered["redirects"], rendered["browser_requests"]
            attempt.status, attempt.snapshot_ref, attempt.error_code = "ok", ref, None
            return attempt
        except RenderError as exc:
            attempt.stage = "render"
            attempt.http_status, attempt.error_code = exc.status, exc.code
            attempt.status = "pending"
            if exc.status == 404:
                attempt.status = "not_found"
            elif exc.status == 429:
                attempt.status = "rate_limited"
                attempt.retry_after = exc.retry_after
                attempt.retry_at = retry_time(exc.retry_after)
            elif exc.status in {401, 403} or "access_challenge" in exc.code:
                attempt.status = "blocked"
            if exc.stop:
                # Installation faults stop this run, but must not poison the domain pause.
                if attempt.status in {"blocked", "rate_limited"}:
                    self._stop(attempt)
                else:
                    self.stopped = True
            return attempt

    def fetch(self, url: str, *, html_path: Path | None = None, refresh=False,
              allow_http=False) -> FetchAttempt:
        url = article_url(url)
        method = "user_saved_html" if html_path else "http"
        attempt = FetchAttempt(requested_url=url, acquisition_method=method, status="pending")
        before = self.requests_made
        self._attempt_start = before
        try:
            if html_path:
                if html_path.stat().st_size > MAX_BYTES:
                    raise ValueError("saved_html_too_large")
                raw = html_path.read_bytes()
                if not raw.strip() or challenge(raw):
                    attempt.status, attempt.error_code = "invalid_response", "saved_html_empty_or_challenge"
                    return attempt
                ref, _ = self.store.save_snapshot(raw, requested_url=url, acquisition_method=method)
                attempt.status, attempt.snapshot_ref = "ok", ref
                return attempt
            cached_ref = self.store.cached(url, http_only=allow_http)
            cached_shell = False
            if cached_ref and not refresh:
                cached_snapshot = SourceSnapshot.model_validate(self.store.read(cached_ref))
                cached_shell = needs_rendering(self.store.raw(cached_snapshot))
                if not cached_shell:
                    attempt.acquisition_method = cached_snapshot.acquisition_method
                    attempt.http_status, attempt.final_url = cached_snapshot.http_status, cached_snapshot.final_url
                    attempt.status, attempt.cache_hit, attempt.snapshot_ref = "ok", True, cached_ref
                    return attempt
            if not allow_http:
                attempt.error_code = "render_required_offline" if cached_shell else "http_not_enabled_and_no_cache"
                attempt.initial_snapshot_ref = cached_ref
                return attempt
            pause_file = self.store.path("http_pause.json")
            if self.stopped or pause_file.exists():
                pause = self.store.read("http_pause.json") if pause_file.exists() else {}
                if not self.stopped and pause.get("error_code") == "robots_disallowed":
                    # Re-evaluate only robots denial with the corrected parser. Access/429
                    # pauses never use this path. Preserve the previous decision for audit.
                    if not self._robots_allowed(url, attempt):
                        return attempt
                    self.store.write(f"pause_history/{digest(pause)}.json", pause)
                    pause_file.unlink()
                    pause = {}
                retry_at = pause.get("retry_at")
                if self.stopped or (pause and (not retry_at or datetime.fromisoformat(retry_at) > datetime.now(timezone.utc))):
                    attempt.error_code = "domain_paused_inspect_http_pause"
                    attempt.retry_at = retry_at
                    return attempt
            if not self._robots_allowed(url, attempt):
                return attempt
            if cached_shell and not refresh:
                return self._render_snapshot(url, cached_ref, attempt)
            request_headers = {}
            if cached_ref:
                cached = SourceSnapshot.model_validate(self.store.read(cached_ref))
                if cached.etag:
                    request_headers["If-None-Match"] = cached.etag
                if cached.last_modified:
                    request_headers["If-Modified-Since"] = cached.last_modified
            current, retries = url, 0
            for _ in range(7):
                try:
                    attempt.stage = "http"
                    status, headers, raw = self._get(current, request_headers)
                except requests.RequestException:
                    if retries < 2:
                        retries += 1
                        continue
                    raise
                attempt.http_status, attempt.final_url = status, current
                if self._access_status(attempt, status, headers, raw):
                    return attempt
                if status in (301, 302, 303, 307, 308):
                    if len(attempt.redirects) >= 3 or not headers.get("Location"):
                        raise ValueError("invalid_or_excessive_redirects")
                    current = article_url(urljoin(current, headers["Location"]),strip_query=True,)
                    if current in attempt.redirects or current == url:
                        raise ValueError("redirect_loop")
                    if not self._robots_allowed(current, attempt):
                        return attempt
                    attempt.redirects.append(current)
                    request_headers = {}  # Never apply another URL's ETag to the target.
                    continue
                if status in (500, 502, 503, 504) and retries < 2:
                    retries += 1
                    continue
                if status == 304 and cached_ref:
                    if needs_rendering(self.store.raw(SourceSnapshot.model_validate(self.store.read(cached_ref)))):
                        return self._render_snapshot(url, cached_ref, attempt)
                    attempt.status, attempt.cache_hit, attempt.snapshot_ref = "ok", True, cached_ref
                    return attempt
                if status == 404:
                    attempt.status = "not_found"
                    return attempt
                if status != 200:
                    attempt.status, attempt.error_code = "network_error", f"http_{status}"
                    return attempt
                content_type = headers.get("Content-Type", "").lower()
                if not raw.strip() or "html" not in content_type:
                    raise ValueError("empty_or_non_html_response")
                ref, _ = self.store.save_snapshot(
                    raw, requested_url=url, acquisition_method=method, final_url=current,
                    http_status=status, redirects=attempt.redirects, headers=headers,
                    publish_cache=not needs_rendering(raw) or not cached_ref,
                )
                if needs_rendering(raw):
                    return self._render_snapshot(url, ref, attempt)
                attempt.status, attempt.snapshot_ref = "ok", ref
                return attempt
            raise ValueError("response_attempt_budget")
        except requests.RequestException as exc:
            attempt.status, attempt.error_code = "network_error", type(exc).__name__
        except RuntimeError as exc:
            attempt.status, attempt.error_code = "pending", str(exc)
            if str(exc) in {"request_budget", "crawl_delay_exceeds_run_wait_budget"}:
                self.stopped = True
        except (ValueError, OSError) as exc:
            attempt.status, attempt.error_code = "invalid_response", type(exc).__name__ + ": " + str(exc)
        finally:
            attempt.requests_made = self.requests_made - before
        return attempt
