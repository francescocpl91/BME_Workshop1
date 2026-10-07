"""Small HTTPS transport with pacing, bounded retries and safe error messages."""
from __future__ import annotations

import json
import socket
import ssl

import certifi
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPSHandler

from .models import RetrievalError


class SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(req.full_url).netloc != urlsplit(newurl).netloc or urlsplit(newurl).scheme != "https":
            raise RetrievalError("network_error", "Refused cross-host or non-HTTPS API redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpClient:
    def __init__(self, min_interval: float = 0, timeout: float = 20, retries: int = 2):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.last_request = 0.0
        self.response_sink = None  # Optional raw-body recorder, before decoding/parsing.
        # Retain OS/custom trust roots and supplement portable Python installations
        # with Mozilla's public CA bundle. Certificate/hostname checks stay enabled.
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        self.opener = build_opener(SameOriginRedirect(), HTTPSHandler(context=context))

    def get(self, url: str, params=None, headers=None) -> str:
        if urlsplit(url).scheme != "https":
            raise RetrievalError("configuration_error", "API connections must use HTTPS.")
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params)
        request = Request(url, headers={"User-Agent": "MarketData/0.1 educational research", **(headers or {})})
        for attempt in range(self.retries + 1):
            delay = self.min_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    body = response.read()
                    self.capture_response(body, response.status, response.headers.get("Content-Type", ""), url)
                    return body.decode("utf-8-sig")
            except HTTPError as exc:
                code = exc.code
                try:
                    body = exc.read()
                    self.capture_response(body, code, exc.headers.get("Content-Type", ""), url)
                finally:
                    exc.close()
                if code in {502, 503, 504} and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                mapping = {
                    400: ("invalid_request", "HTTP 400: check symbol, series key and account configuration."),
                    401: ("authentication_error", "HTTP 401: API credential rejected."),
                    402: ("entitlement_error", "HTTP 402: this request requires additional account access."),
                    403: ("access_denied", "HTTP 403: provider, account or network denied access."),
                    404: ("no_data", "HTTP 404: endpoint/symbol unavailable or no data for this window."),
                    429: ("rate_limited", "HTTP 429: provider rate/credit limit reached; try later."),
                }
                raise RetrievalError(*mapping.get(code, ("http_error", f"Provider HTTP error {code}."))) from None
            except (URLError, TimeoutError, socket.timeout, ConnectionError):
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise RetrievalError("network_error", "Connection failed or timed out; check network/proxy access.") from None
        raise RetrievalError("network_error", "Request failed.")

    def capture_response(self, body, status, content_type, url):
        if self.response_sink is not None:
            self.response_sink(body, status, content_type, url)

    def get_json(self, url: str, params=None, headers=None):
        try:
            return json.loads(self.get(url, params=params, headers=headers))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RetrievalError("invalid_response", "API response was not valid JSON.") from None
