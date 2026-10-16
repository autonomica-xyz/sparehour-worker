"""Loopback-only web dashboard for the worker: status, queues, selection, pause.

Plain http.server — no extra dependencies. The dashboard is a local control
surface with a hard boundary:

* it binds to a loopback IP only — never a wildcard or a resolving hostname
  (``localhost`` is normalized to ``127.0.0.1``);
* every /api/* request requires `Authorization: Bearer <token>` where the
  token is a per-run secret unrelated to the control-plane token (the token
  is never accepted in a query string);
* the browser receives the token via the URL fragment (`server.dashboard_url`),
  stores it in sessionStorage, and strips it from the address bar;
* the Host header must be exactly the bound address:port (defeats DNS
  rebinding) and Origin, when present, must be exactly this origin;
* POST bodies must be small `application/json` objects with exact field
  types — routes are validated before any body bytes are read;
* every accepted connection — including incomplete headers and idle
  keep-alive — counts against ``max_concurrency``; excess connections are
  answered 503 and closed before a handler thread is allocated;
* each request (line, headers, body) has one monotonic deadline that
  trickled bytes cannot extend;
* ``server_close()`` (or ``shutdown()``) atomically stops new work, breaks
  stalled reads, and joins request threads and the prefetch worker, so no
  request or prefetch work outlives the server.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import secrets
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from . import config

HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")

MAX_BODY_BYTES = 16 * 1024
MAX_CAPABILITY_CHARS = 256
MAX_MESSAGE_CHARS = 500


_SHED_BODY = b'{"ok": false, "error": "dashboard busy"}'
_SHED_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: application/json\r\n"
    b"Cache-Control: no-store\r\n"
    b"Content-Length: " + str(len(_SHED_BODY)).encode("ascii") + b"\r\n"
    b"Connection: close\r\n\r\n" + _SHED_BODY
)


class _DeadlineReader:
    """``rfile`` replacement enforcing one monotonic deadline across reads.

    A per-operation socket timeout is not a request bound: buffered reads
    loop on ``recv`` and each trickled byte buys a fresh timeout.  Before
    every read this wrapper ``select``s for at most the remaining deadline
    (or the socket timeout, whichever is sooner), then performs a ``recv``.
    """

    __slots__ = ("_sock", "_handler", "_buf")

    def __init__(self, sock: socket.socket, handler: BaseHTTPRequestHandler) -> None:
        self._sock = sock
        self._handler = handler
        self._buf = bytearray()

    def _fill(self) -> None:
        """Block until one chunk arrives or the deadline/socket timeout hits."""
        deadline = self._handler._read_deadline
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        timeout = self._sock.gettimeout()
        wait = timeout if remaining is None else min(remaining, timeout or remaining)
        readable, _, _ = select.select([self._sock], [], [], wait)
        if not readable:
            raise TimeoutError("request read timed out")
        data = self._sock.recv(65536)
        if not data:
            raise ConnectionError("client closed connection")
        self._buf += data

    def _take(self, size: int) -> bytes:
        if size >= len(self._buf):
            out = bytes(self._buf)
            self._buf.clear()
            return out
        out = bytes(self._buf[:size])
        del self._buf[:size]
        return out

    def readline(self, size: int = -1) -> bytes:
        limit = 65536 if size < 0 else size
        out = bytearray()
        while len(out) < limit:
            newline = self._buf.find(b"\n")
            if newline >= 0:
                out += self._take(min(newline + 1, limit - len(out)))
                break
            out += self._take(limit - len(out))
            if len(out) >= limit:
                break
            try:
                self._fill()
            except ConnectionError:
                break  # EOF: return what we have, like a buffered file
        return bytes(out)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            try:
                while True:
                    self._fill()
            except ConnectionError:
                pass
            return self._take(len(self._buf))
        out = bytearray()
        while len(out) < size:
            out += self._take(size - len(out))
            if len(out) >= size:
                break
            try:
                self._fill()
            except ConnectionError:
                break  # EOF: short read; caller sees a truncated body
        return bytes(out)

    def close(self) -> None:  # the socket itself is closed by shutdown_request
        pass


_GET_ROUTES = {"/", "/index.html", "/api/status"}
_POST_ROUTES = {"/api/pause", "/api/select", "/api/prefetch"}


class DashboardServer(ThreadingHTTPServer):
    """Threading HTTP server with auth, bounded concurrency, joined workers.

    Connection admission happens at accept time, before a handler thread
    exists: incomplete-header and idle keep-alive connections count against
    ``max_concurrency``, and excess connections are answered 503 and closed.
    ``server_close()`` denies new work, breaks stalled reads, and joins all
    request threads and the prefetch worker before returning.

    Attributes may be tightened before ``serve_forever`` for tests or
    constrained deployments.
    """

    daemon_threads = True

    request_timeout = 30.0  # seconds; monotonic deadline per request
    max_concurrency = 16  # simultaneous accepted connections

    def __init__(self, host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> None:
        ip = ipaddress.ip_address(host)  # caller already validated loopback
        if ip.version == 6:
            self.address_family = socket.AF_INET6
        self.access_token = secrets.token_urlsafe(24)
        self._inflight = 0
        self._closing = False
        self._lifecycle_cond = threading.Condition()
        self._request_threads: set[threading.Thread] = set()
        self._request_sockets: set[socket.socket] = set()
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_running = False
        super().__init__((host, port), handler)
        bound_host = self.server_address[0]
        shown = f"[{bound_host}]" if ":" in bound_host else bound_host
        self.allowed_hosts = frozenset({f"{shown}:{self.server_port}"})
        self.allowed_origin = f"http://{shown}:{self.server_port}"
        self.dashboard_url = f"http://{shown}:{self.server_port}/#{self.access_token}"

    # -- connection admission --------------------------------------------------

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        """Admit a connection before allocating a handler thread.

        The permit is held for the whole connection — while headers are
        still arriving and while it sits idle between keep-alive requests —
        so those states cannot exhaust the bound.
        """
        with self._lifecycle_cond:
            if self._closing:
                allowed = False
            elif self._inflight >= self.max_concurrency:
                allowed = False
            else:
                self._inflight += 1
                self._request_sockets.add(request)
                allowed = True
        if not allowed:
            self._shed_connection(request)
            return
        thread = threading.Thread(
            target=self._request_thread,
            args=(request, client_address),
            name="dashboard-request",
            daemon=True,
        )
        with self._lifecycle_cond:
            self._request_threads.add(thread)
        try:
            thread.start()
        except Exception:
            with self._lifecycle_cond:
                self._request_threads.discard(thread)
            self._release_connection(request)
            try:
                self.shutdown_request(request)
            except OSError:
                pass
            raise

    def _request_thread(self, request: socket.socket, client_address: Any) -> None:
        thread = threading.current_thread()
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        try:
            self.shutdown_request(request)
        except OSError:
            pass  # socket already shut down by _begin_close
        finally:
            self._release_connection(request, thread)

    def _release_connection(
        self, request: socket.socket, thread: threading.Thread | None = None
    ) -> None:
        with self._lifecycle_cond:
            self._request_sockets.discard(request)
            if thread is not None:
                self._request_threads.discard(thread)
            self._inflight -= 1
            self._lifecycle_cond.notify_all()

    def _shed_connection(self, request: socket.socket) -> None:
        # Answer 503 before the client finishes its request, then close
        # cleanly so the response is readable even with bytes still in flight.
        try:
            request.settimeout(5.0)
            request.sendall(_SHED_RESPONSE)
        except OSError:
            pass
        try:
            request.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        request.close()

    # -- bounded single-flight prefetch -------------------------------------

    def start_prefetch(self, handler: Callable[[], None]) -> bool:
        with self._prefetch_lock:
            if self._prefetch_running or self._closing:
                return False
            self._prefetch_running = True

            def work() -> None:
                try:
                    if not self._closing:
                        handler()
                finally:
                    with self._prefetch_lock:
                        self._prefetch_running = False

            self._prefetch_thread = threading.Thread(
                target=work, name="dashboard-prefetch", daemon=True
            )
            self._prefetch_thread.start()
            return True

    # -- lifecycle ------------------------------------------------------------

    def shutdown(self) -> None:
        self._begin_close()
        super().shutdown()

    def server_close(self) -> None:
        self._begin_close()
        super().server_close()
        self._join_workers()

    def _begin_close(self) -> None:
        """Atomically stop new work and break stalled request reads."""
        with self._lifecycle_cond:
            if self._closing:
                return
            self._closing = True
            sockets = list(self._request_sockets)
        for sock in sockets:
            # Shutting down (not closing) the socket wakes a blocked recv
            # without an fd-reuse race against the handler thread.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _join_workers(self) -> None:
        me = threading.current_thread()
        while True:
            with self._lifecycle_cond:
                pending = [t for t in self._request_threads if t is not me]
                if not pending:
                    break
                self._lifecycle_cond.wait()
        for thread in pending:
            thread.join()
        prefetch = self._prefetch_thread
        if prefetch is not None and prefetch is not me:
            prefetch.join()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Disconnects, read timeouts, and close-time OSErrors are routine;
        # don't spam stderr.
        pass


def make_server(
    *,
    status_provider: Callable[[], dict[str, Any]],
    pause_handler: Callable[[bool], None],
    select_handler: Callable[[str, bool], tuple[bool, str]],
    prefetch_handler: Callable[[], None],
    host: str,
    port: int,
) -> DashboardServer:
    host = config.validate_dashboard_host(str(host))

    class Handler(BaseHTTPRequestHandler):
        server: DashboardServer
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        _read_deadline: float | None = None

        def setup(self) -> None:
            super().setup()
            # Bound the whole request read by one monotonic deadline instead
            # of the per-recv socket timeout (which trickled bytes extend).
            self.rfile.close()
            self.rfile = _DeadlineReader(self.request, self)

        def handle(self) -> None:
            # Bound every read (request line, headers, body, keep-alive idle).
            self.request.settimeout(self.server.request_timeout)
            super().handle()

        def handle_one_request(self) -> None:
            # One deadline covers the request line, headers, and body —
            # plus, equivalently, the idle wait for a keep-alive request.
            timeout = self.server.request_timeout
            self._read_deadline = (
                None if timeout is None else time.monotonic() + timeout
            )
            try:
                super().handle_one_request()
            finally:
                self._read_deadline = None

        # -- responses -------------------------------------------------------

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def _reject(self, code: int, message: str) -> None:
            # Rejections happen before the body is consumed; close the
            # connection so unread bytes can't corrupt the next request.
            self.close_connection = True
            self._json(code, {"ok": False, "error": message})

        # -- guards ------------------------------------------------------------

        def _host_ok(self) -> bool:
            hosts = self.headers.get_all("Host")
            if hosts is not None and len(hosts) > 1:
                self._reject(400, "duplicate Host header")
                return False
            host = self.headers.get("Host", "").strip().lower()
            if host in self.server.allowed_hosts:
                return True
            self._reject(403, "forbidden host")
            return False

        def _origin_ok(self) -> bool:
            origins = self.headers.get_all("Origin")
            if origins is not None and len(origins) > 1:
                self._reject(400, "duplicate Origin header")
                return False
            origin = self.headers.get("Origin")
            if origin is None or origin.strip().lower() == self.server.allowed_origin:
                return True
            self._reject(403, "foreign origin")
            return False

        def _authorized(self) -> bool:
            headers = self.headers.get_all("Authorization")
            if headers is not None and len(headers) > 1:
                self._reject(400, "duplicate Authorization header")
                return False
            header = self.headers.get("Authorization", "")
            # Compare as bytes so non-ASCII header values reject cleanly
            # instead of raising TypeError in compare_digest.
            token = self.server.access_token.encode("utf-8")
            if header.startswith("Bearer ") and hmac.compare_digest(
                header[7:].strip().encode("utf-8"), token
            ):
                return True
            self.close_connection = True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Bearer realm="sparehour-worker-dashboard"')
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            body = json.dumps({"ok": False, "error": "authentication required"}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        # -- body --------------------------------------------------------------

        def _read_body(self) -> dict[str, Any] | None:
            """Validate and read a POST body; replies and returns None on failure."""
            ctype = self.headers.get("Content-Type")
            if (
                ctype is None
                or ctype.split(";", 1)[0].strip().lower() != "application/json"
            ):
                self._reject(415, "expected application/json")
                return None
            transfer = self.headers.get("Transfer-Encoding")
            if transfer is not None and transfer.strip().lower() != "identity":
                self._reject(400, "transfer-encoding not supported")
                return None
            lengths = self.headers.get_all("Content-Length")
            if lengths is None:
                self._reject(411, "content-length required")
                return None
            if len(lengths) != 1:
                self._reject(400, "ambiguous content-length")
                return None
            try:
                length = int(lengths[0].strip())
            except ValueError:
                self._reject(400, "invalid content-length")
                return None
            if length < 0:
                self._reject(400, "invalid content-length")
                return None
            if length > MAX_BODY_BYTES:
                self._reject(413, "body too large")
                return None
            try:
                raw = self.rfile.read(length)
            except OSError:  # socket.timeout included
                self._reject(408, "body read timed out")
                return None
            if len(raw) != length:
                self._reject(400, "truncated body")
                return None
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._reject(400, "body must be utf-8")
                return None
            if not text.strip():
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                self._reject(400, "invalid json")
                return None
            except RecursionError:
                self._reject(400, "json nested too deeply")
                return None
            if not isinstance(parsed, dict):
                self._reject(400, "expected a json object")
                return None
            return parsed

        # -- methods -----------------------------------------------------------

        def _method_not_allowed(self) -> None:
            self._reject(405, "method not allowed")

        def do_GET(self) -> None:
            if not self._host_ok() or not self._origin_ok():
                return
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                # Public shell: static HTML, holds no state or secrets.
                self._send(
                    200, HTML.encode("utf-8"), "text/html; charset=utf-8"
                )
                return
            if path.startswith("/api/") and not self._authorized():
                return
            if path == "/api/status":
                if self.server._closing:
                    self._reject(503, "server closing")
                    return
                try:
                    payload = status_provider()
                except Exception:
                    self._json(500, {"ok": False, "error": "status unavailable"})
                    return
                self._json(200, payload)
            elif path in _POST_ROUTES:
                self._method_not_allowed()
            else:
                self._reject(404, "not found")

        def do_POST(self) -> None:
            if not self._host_ok() or not self._origin_ok():
                return
            path = urlsplit(self.path).path
            if path.startswith("/api/") and not self._authorized():
                return
            if path not in _POST_ROUTES:
                if path in _GET_ROUTES:
                    self._method_not_allowed()
                else:
                    self._reject(404, "not found")
                return
            body = self._read_body()
            if body is None:
                return
            # The body is fully read here; refuse to launch new callbacks
            # once the server has started closing.
            if self.server._closing:
                self._reject(503, "server closing")
                return
            try:
                if path == "/api/pause":
                    paused = body.get("paused")
                    if type(paused) is not bool:
                        self._reject(400, "'paused' must be a boolean")
                        return
                    pause_handler(paused)
                    self._json(200, {"ok": True})
                elif path == "/api/select":
                    capability = body.get("capability")
                    selected = body.get("selected")
                    if (
                        not isinstance(capability, str)
                        or not 0 < len(capability) <= MAX_CAPABILITY_CHARS
                    ):
                        self._reject(
                            400, "'capability' must be a non-empty string"
                        )
                        return
                    if type(selected) is not bool:
                        self._reject(400, "'selected' must be a boolean")
                        return
                    ok, message = select_handler(capability, selected)
                    self._json(
                        200 if ok else 409,
                        {"ok": bool(ok), "message": str(message)[:MAX_MESSAGE_CHARS]},
                    )
                elif path == "/api/prefetch":
                    if self.server.start_prefetch(prefetch_handler):
                        self._json(
                            200, {"ok": True, "message": "prefetch started"}
                        )
                    else:
                        self._json(
                            409,
                            {"ok": False, "error": "prefetch already running"},
                        )
            except Exception:
                # Sanitized: callback failures must not leak internals.
                self._json(500, {"ok": False, "error": "internal error"})

    return DashboardServer(host, port, Handler)
