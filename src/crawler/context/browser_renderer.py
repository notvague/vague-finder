"""Bounded ordinary Chromium rendering. No stealth, login or challenge handling.

The fetcher owns robots, navigation budgets, cache publication and status records.
One browser is reused; each document receives a fresh, credential-free context.
"""
from __future__ import annotations


class RenderError(RuntimeError):
    def __init__(self, code, *, status=None, retry_after=None, stop=False):
        super().__init__(code)
        self.code, self.status = code, status
        self.retry_after, self.stop = retry_after, stop


READY = r"""() => {
  const body = document.body;
  const h1 = document.querySelector('h1');
  if (!body || !h1 || !h1.textContent.trim()) return false;
  const outline = [...document.querySelectorAll('h2')].filter(h => /^\s*\d+\./.test(h.textContent));
  const semantic = document.querySelector('.wiki-content, .wiki-inner-content, article, main, [role="main"]');
  if (!outline.length && (!semantic || semantic.textContent.trim().length <= h1.textContent.trim().length)) return false;
  let root = semantic;
  if (!root && outline.length >= 2) {
    root = outline[0].parentElement;
    while (root && !outline.every(h => root.contains(h))) root = root.parentElement;
    if (!root || root === body || root === document.documentElement) return false;
  } else if (!root && outline.length === 1) {
    root = outline[0].parentElement;
    while (root && root !== body && root !== document.documentElement) {
      const hasContent = root.querySelector('p, li, table, blockquote');
      if (hasContent && root.textContent.trim().length > outline[0].textContent.trim().length + 5) break;
      root = root.parentElement;
    }
    if (!root || root === body || root === document.documentElement) return false;
  }
  const text = root.textContent.trim();
  if (!text) return false;
  const signature = text + '\n' + document.querySelectorAll('h2,h3,h4,h5,h6').length;
  const now = performance.now();
  if (!window.__vagueFinderReady || window.__vagueFinderReady.signature !== signature) {
    window.__vagueFinderReady = {signature, since: now};
    return false;
  }
  return now - window.__vagueFinderReady.since >= 800;
}"""


class BrowserRenderer:
    def __init__(self):
        self.runtime = self.browser = None

    def close(self):
        try:
            if self.browser:
                self.browser.close()
        finally:
            if self.runtime:
                self.runtime.stop()
            self.runtime = self.browser = None

    def render(self, url, *, user_agent, timeout, on_navigation, is_challenge,
               max_bytes, max_resources=200):
        try:
            from playwright.sync_api import sync_playwright, Error, TimeoutError
        except ImportError as exc:
            raise RenderError("playwright_package_missing", stop=True) from exc
        if self.browser is None:
            try:
                self.runtime = sync_playwright().start()
                self.browser = self.runtime.chromium.launch(headless=True)
            except Error as exc:
                self.close()
                raise RenderError("browser_launch_failed_install_chromium_and_dependencies", stop=True) from exc
        context = self.browser.new_context(user_agent=user_agent, service_workers="block")
        page = context.new_page()
        requests, navigation_urls, failures = [], [], []
        main_status = None

        def route_request(route):
            request = route.request
            requests.append(request.url)
            if len(requests) > max_resources:
                failures.append(RenderError("render_resource_budget"))
            if failures:
                route.abort()
                return
            if request.is_navigation_request() and request.frame == page.main_frame:
                if len(navigation_urls) >= 4:
                    failures.append(RenderError("render_redirect_budget"))
                    route.abort()
                    return
                try:
                    normalized_url = on_navigation(request.url)
                    navigation_urls.append(normalized_url)
                except (ValueError, RuntimeError) as exc:
                    failures.append(exc)
                    route.abort()
                    return
                # Playwright routing does not intercept every HTTP redirect hop.
                # HTTP fetcher has already resolved normal redirects. Do not let
                # a different browser-only redirect escape URL/robots checks.
                try:
                    response = route.fetch(url=normalized_url, max_redirects=0, timeout=timeout * 1000)
                    if 300 <= response.status < 400:
                        failures.append(RenderError("render_redirect_requires_http_refresh", status=response.status))
                        route.abort()
                    elif len(response.body()) > max_bytes:
                        failures.append(RenderError("rendered_response_too_large", status=response.status))
                        route.abort()
                    else:
                        route.fulfill(response=response)
                except Error:
                    failures.append(RenderError("render_document_request_failed"))
                    route.abort()
                return
            if request.is_navigation_request():
                route.abort()  # Embedded videos/ads are not article text.
                return
            if request.resource_type in {"image", "media", "font"}:
                route.abort()
            else:
                route.continue_()

        def response_received(response):
            nonlocal main_status
            request = response.request
            if request.is_navigation_request() and request.frame == page.main_frame:
                main_status = response.status
                if response.status in {401, 403, 429}:
                    failures.append(RenderError(
                        "render_http_429" if response.status == 429 else "render_access_denied",
                        status=response.status, retry_after=response.headers.get("retry-after"), stop=True))
                elif response.status >= 400:
                    failures.append(RenderError(f"render_http_{response.status}", status=response.status))

        context.route("**/*", route_request)
        page.on("response", response_received)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            if failures:
                raise failures[0]
            if is_challenge(page.content().encode("utf-8")):
                raise RenderError("render_access_challenge", status=main_status, stop=True)
            # Poll in bounded slices so a late access screen/resource limit stops promptly.
            import time
            deadline = time.monotonic() + timeout
            while True:
                if failures:
                    raise failures[0]
                if is_challenge(page.content().encode("utf-8")):
                    raise RenderError("render_access_challenge", status=main_status, stop=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RenderError("render_timeout_no_stable_document", status=main_status)
                try:
                    page.wait_for_function(READY, polling=250, timeout=min(1000, remaining * 1000))
                    break
                except TimeoutError:
                    pass
            if failures:
                raise failures[0]
            raw = page.content().encode("utf-8")
            if len(raw) > max_bytes:
                raise RenderError("rendered_html_too_large", status=main_status)
            if is_challenge(raw):
                raise RenderError("render_access_challenge", status=main_status, stop=True)
            return {"raw": raw, "final_url": page.url, "http_status": main_status,
                    "redirects": navigation_urls[1:], "browser_requests": len(requests)}
        except RenderError:
            raise
        except TimeoutError as exc:
            if failures:
                raise failures[0]
            raise RenderError("render_navigation_timeout", status=main_status) from exc
        except (ValueError, RuntimeError) as exc:
            if failures:
                raise failures[0]
            raise
        except Error as exc:
            if failures:
                raise failures[0]
            raise RenderError("render_navigation_or_browser_error", status=main_status) from exc
        finally:
            context.close()
