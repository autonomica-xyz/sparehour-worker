"""Worker-facing control-plane client.

``WorkerClient`` is a drop-in ``ControlPlaneClient`` that hardens the HTTP
transport for a worker holding a bearer token against a public endpoint:

- the endpoint is policy-checked (``config.validate_endpoint``), TLS
  certificate verification stays enabled, and proxies/netrc are ignored so
  credentials cannot leak through ambient configuration — the one env
  exception is the standard trust roots ``SSL_CERT_FILE``/``SSL_CERT_DIR``,
  which an operator may set deliberately to trust a private CA;
- every request — including direct ``.client`` calls and absolute artifact
  URLs — is pinned to the configured origin with no embedded userinfo, and
  redirects are never followed, so the Authorization header can never reach
  a third party;
- ``httpx`` transport failures and malformed response bodies surface as
  ``AgentError`` with the token and URL queries redacted, and HTTP error
  statuses no longer echo the remote response body (which could carry
  secrets or injected text into logs).
"""

from __future__ import annotations

import math
import os
import re
import ssl
import urllib.parse
from json import JSONDecodeError
from typing import Any

import certifi
import httpx

from sparehour.agent import AgentError, ControlPlaneClient

from . import config as config_mod


def _effective_port(url: httpx.URL) -> int:
    if url.port is not None:
        return url.port
    return {"https": 443, "http": 80}.get(url.scheme, -1)


_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>)\]]+")


def _strip_url_queries(text: str) -> str:
    """Remove query/fragment components from URLs embedded in a message."""
    def _replace(match: re.Match[str]) -> str:
        try:
            parts = urllib.parse.urlsplit(match.group(0))
        except ValueError:
            return match.group(0)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    return _URL_IN_TEXT.sub(_replace, text)


def _trusted_ssl_context() -> ssl.SSLContext:
    """Verified TLS context: default public roots plus deliberate env roots.

    ``SSL_CERT_FILE``/``SSL_CERT_DIR`` are honored because they are the
    standard mechanism for intentionally trusting a private CA; unlike
    proxies or netrc they cannot redirect the bearer token anywhere.
    Verification stays mandatory in every case.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    cert_file = os.environ.get("SSL_CERT_FILE") or None
    cert_dir = os.environ.get("SSL_CERT_DIR") or None
    if cert_file or cert_dir:
        try:
            context.load_verify_locations(cafile=cert_file, capath=cert_dir)
        except (OSError, ssl.SSLError) as exc:
            raise config_mod.ConfigError(
                f"cannot load SSL_CERT_FILE/SSL_CERT_DIR roots: {exc}"
            ) from exc
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    return context


class _OriginLockedClient(httpx.Client):
    def __init__(self, *args: Any, origin: tuple[str, str, int], sanitize: Any, **kwargs: Any):
        kwargs["follow_redirects"] = False
        super().__init__(*args, **kwargs)
        self._origin = origin
        self._sanitize = sanitize

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        url = request.url
        if url.username or url.password:
            raise AgentError("refusing to send a request with embedded userinfo")
        if (url.scheme, url.host, _effective_port(url)) != self._origin:
            raise AgentError(
                f"refusing to send an authenticated request off-origin to "
                f"{url.scheme}://{url.host} — artifact and API links must stay on the controller origin"
            )
        kwargs["follow_redirects"] = False
        try:
            return super().send(request, **kwargs)
        except httpx.HTTPError as exc:
            detail = self._sanitize(str(exc))
            raise AgentError(f"control-plane request failed ({type(exc).__name__}): {detail}") from exc


class WorkerClient(ControlPlaneClient):
    """ControlPlaneClient pinned to a validated endpoint's origin.

    Signature matches upstream: ``WorkerClient(endpoint, token, *, timeout=…)``.
    ``transport`` is accepted for deterministic fake transports in tests.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        base = config_mod.validate_endpoint(endpoint)
        if not isinstance(token, str) or not token.strip():
            raise config_mod.ConfigError("agent token must be a non-empty string")
        # bool is an int subclass; True/False are not timeouts. NaN and inf
        # would give httpx a meaningless or never-firing deadline.
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise config_mod.ConfigError("timeout must be a positive finite number")
        self._token = token
        # httpx normalizes the port (drops :443/:80); _effective_port covers
        # the explicit-port forms for comparison anyway.
        base_url = httpx.URL(base)
        origin = (base_url.scheme, base_url.host, _effective_port(base_url))
        self.client = _OriginLockedClient(
            base_url=base,
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
            verify=_trusted_ssl_context(),  # ignored when transport is injected
            trust_env=False,  # proxies/netrc must not see the bearer token
            follow_redirects=False,
            transport=transport,
            origin=origin,
            sanitize=self._sanitize,
        )

    def _sanitize(self, text: str) -> str:
        """Strip credentials and URL queries from anything headed for a log."""
        text = _strip_url_queries(text)
        for secret in {self._token, urllib.parse.quote(self._token, safe="")}:
            if secret:
                text = text.replace(secret, "***")
        return text

    def _json(self, response: httpx.Response, operation: str) -> Any:
        # The remote body may contain injected text or leaked internals —
        # report only the status, never echo it.
        if response.status_code < 200 or response.status_code >= 300:
            raise AgentError(f"{operation} failed: HTTP {response.status_code}")
        if not response.content:
            return None
        try:
            return response.json()
        except JSONDecodeError as exc:
            # Malformed bodies are a recoverable remote fault; never let a
            # raw decode error carrying body text escape into slot loops.
            raise AgentError(f"{operation} failed: malformed JSON response") from exc
