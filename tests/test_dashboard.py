"""HTTP-boundary tests for the worker dashboard.

Exercises a real listening server: auth, Host/Origin checks, body bounds,
single-flight prefetch, and loopback-only binding. No control plane needed.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import types
from typing import Any

import pytest

from dworker import dashboard


@pytest.fixture
def server():
    """A real dashboard server on 127.0.0.1 with recording stub callbacks."""
    calls: dict[str, list] = {"pause": [], "select": [], "prefetch": []}
    state: dict[str, Any] = {"payload": {"agent_id": "w1", "paused": False}}
    started = threading.Event()
    release = threading.Event()

    def status() -> dict[str, Any]:
        return dict(state["payload"])

    def pause(paused: bool) -> None:
        calls["pause"].append(paused)
        state["payload"]["paused"] = paused

    def select(capability: str, selected: bool) -> tuple[bool, str]:
        calls["select"].append((capability, selected))
        return True, "applied"

    def prefetch() -> None:
        calls["prefetch"].append(1)
        started.set()
        release.wait(10)

    srv = dashboard.make_server(
        status_provider=status,
        pause_handler=pause,
        select_handler=select,
        prefetch_handler=prefetch,
        host="127.0.0.1",
        port=0,
    )
    srv.request_timeout = 2.0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.calls = calls  # type: ignore[attr-defined]
    srv.state = state  # type: ignore[attr-defined]
    srv.prefetch_started = started  # type: ignore[attr-defined]
    srv.prefetch_release = release  # type: ignore[attr-defined]
    yield srv
    release.set()
    srv.shutdown()
    thread.join(5)
    srv.server_close()


def _conn(srv) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(
        srv.server_address[0], srv.server_port, timeout=5
    )


def _req(srv, method: str, path: str, **kw) -> tuple[int, dict]:
    """One request on a fresh connection; returns (status, json-or-text body)."""
    conn = _conn(srv)
    conn.request(method, path, **kw)
    resp = conn.getresponse()
    raw = resp.read()
    status = resp.status
    headers = dict(resp.getheaders())
    conn.close()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, {"_raw": raw.decode("utf-8", "replace"), "_headers": headers}


def _auth(srv) -> dict[str, str]:
    return {"Authorization": f"Bearer {srv.access_token}"}


def _json_headers(srv) -> dict[str, str]:
    return {**_auth(srv), "Content-Type": "application/json"}


# -- serving and binding ----------------------------------------------------


def test_root_page_public_and_token_free(server):
    status, body = _req(server, "GET", "/")
    assert status == 200
    assert "<title>dworker</title>" in body["_raw"]
    assert server.access_token not in body["_raw"]
    # The fragment bootstrap exists so the token never hits the query string.
    assert "location.hash" in body["_raw"]
    assert "sessionStorage" in body["_raw"]


def test_dashboard_url_carries_fragment_token(server):
    url = server.dashboard_url
    assert url.startswith(f"http://127.0.0.1:{server.server_port}/#")
    assert url.rsplit("#", 1)[1] == server.access_token
    # Per-run secret, not derivable from or equal to any fixed value.
    srv2 = dashboard.make_server(
        status_provider=lambda: {},
        pause_handler=lambda p: None,
        select_handler=lambda c, s: (True, ""),
        prefetch_handler=lambda: None,
        host="127.0.0.1",
        port=0,
    )
    try:
        assert srv2.access_token != server.access_token
    finally:
        srv2.server_close()


def test_refuses_non_loopback_and_hostname_bind():
    def make(host: str):
        return dashboard.make_server(
            status_provider=lambda: {},
            pause_handler=lambda p: None,
            select_handler=lambda c, s: (True, ""),
            prefetch_handler=lambda: None,
            host=host,
            port=0,
        )

    for host in ("0.0.0.0", "10.0.0.5", "::", "example.com"):
        with pytest.raises(ValueError, match="loopback"):
            make(host)

    # `localhost` is normalized to the literal loopback, not resolved: the
    # server actually binds 127.0.0.1.
    srv = make("localhost")
    try:
        assert srv.server_address[0] == "127.0.0.1"
        assert srv.dashboard_url.startswith("http://127.0.0.1:")
    finally:
        srv.server_close()


def test_ipv6_loopback_accepted_when_available():
    try:
        srv = dashboard.make_server(
            status_provider=lambda: {"ok": 1},
            pause_handler=lambda p: None,
            select_handler=lambda c, s: (True, ""),
            prefetch_handler=lambda: None,
            host="::1",
            port=0,
        )
    except (OSError, socket.gaierror):
        pytest.skip("no IPv6 loopback on this host")
    try:
        assert srv.dashboard_url.startswith("http://[::1]:")
        conn = http.client.HTTPConnection(
            "::1", srv.server_port, timeout=5
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        conn.request(
            "GET", "/api/status", headers={"Authorization": f"Bearer {srv.access_token}"}
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.close()
        srv.shutdown()
        thread.join(5)
    finally:
        srv.server_close()


# -- authentication ----------------------------------------------------------


def test_api_requires_bearer_token(server):
    for method, path, kw in (
        ("GET", "/api/status", {}),
        ("POST", "/api/pause", {"body": "{}", "headers": {"Content-Type": "application/json"}}),
        ("POST", "/api/prefetch", {"body": "{}", "headers": {"Content-Type": "application/json"}}),
    ):
        status, body = _req(server, method, path, **kw)
        assert status == 401, (method, path)
        assert body["ok"] is False
    assert server.calls["pause"] == []
    assert server.calls["prefetch"] == []


def test_wrong_and_mangled_tokens_rejected(server):
    for bad in ("wrong-token", "", server.access_token + "x"):
        status, _ = _req(
            server, "GET", "/api/status",
            headers={"Authorization": f"Bearer {bad}"},
        )
        assert status == 401
    status, _ = _req(
        server, "GET", "/api/status",
        headers={"Authorization": server.access_token},
    )
    assert status == 401
    # A token in the query string is never accepted.
    status, _ = _req(server, "GET", f"/api/status?token={server.access_token}")
    assert status == 401


def test_status_works_with_token(server):
    status, body = _req(server, "GET", "/api/status", headers=_auth(server))
    assert status == 200
    assert body["agent_id"] == "w1"


# -- Host / Origin ------------------------------------------------------------


def test_host_header_enforced(server):
    status, body = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Host": "evil.example.com"},
    )
    assert status == 403
    status, _ = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Host": "127.0.0.1:1"},
    )
    assert status == 403
    status, _ = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Host": "127.0.0.1"},  # missing port
    )
    assert status == 403
    status, _ = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Host": "localhost:7788"},
    )
    assert status == 403


def test_origin_enforced_on_api(server):
    status, _ = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Origin": "http://evil.example.com"},
    )
    assert status == 403
    status, _ = _req(
        server, "POST", "/api/pause",
        body=json.dumps({"paused": True}),
        headers={**_json_headers(server), "Origin": "http://evil.example.com"},
    )
    assert status == 403
    assert server.calls["pause"] == []
    # Same-origin requests pass.
    status, _ = _req(
        server, "GET", "/api/status",
        headers={**_auth(server), "Origin": server.allowed_origin},
    )
    assert status == 200


def test_foreign_origin_text_plain_mutation_denied(server):
    """Cross-site form/CSRF shape: foreign Origin + non-JSON content type."""
    status, body = _req(
        server, "POST", "/api/pause",
        body='{"paused": true}',
        headers={
            **_auth(server),
            "Origin": "http://attacker.invalid",
            "Content-Type": "text/plain",
        },
    )
    assert status == 403
    assert server.calls["pause"] == []


# -- request validation -------------------------------------------------------


def test_post_requires_json_content_type(server):
    status, _ = _req(
        server, "POST", "/api/pause",
        body='{"paused": true}',
        headers={**_auth(server), "Content-Type": "text/plain"},
    )
    assert status == 415
    status, _ = _req(
        server, "POST", "/api/pause",
        body='{"paused": true}',
        headers=_auth(server),  # no content type at all
    )
    assert status == 415
    assert server.calls["pause"] == []


def test_content_length_validation(server):
    conn = _conn(server)
    conn.putrequest("POST", "/api/pause", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "abc")
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 400
    resp.read()
    conn.close()

    # No Content-Length at all.
    conn = _conn(server)
    conn.putrequest("POST", "/api/pause", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 411
    resp.read()
    conn.close()

    # Oversized body rejected before it is read.
    status, _ = _req(
        server, "POST", "/api/pause",
        body=b"x" * (dashboard.MAX_BODY_BYTES + 1),
        headers=_json_headers(server),
    )
    assert status == 413
    assert server.calls["pause"] == []


def test_body_must_be_utf8_json_object(server):
    for bad in (b"[1,2]", b"null", b'"paused"', b"\xff\xfe{}", b"{"):
        status, _ = _req(
            server, "POST", "/api/pause", body=bad, headers=_json_headers(server)
        )
        assert status == 400, bad
    assert server.calls["pause"] == []


def test_exact_field_types_enforced(server):
    # Truthy strings must not coerce to booleans.
    status, _ = _req(
        server, "POST", "/api/pause",
        body=json.dumps({"paused": "false"}),
        headers=_json_headers(server),
    )
    assert status == 400
    status, _ = _req(
        server, "POST", "/api/pause",
        body=json.dumps({"paused": 1}),
        headers=_json_headers(server),
    )
    assert status == 400
    status, _ = _req(
        server, "POST", "/api/pause",
        body=json.dumps({}),
        headers=_json_headers(server),
    )
    assert status == 400
    assert server.calls["pause"] == []

    # select: capability must be a bounded string, selected a real bool.
    bad_bodies = [
        {"capability": 5, "selected": True},
        {"capability": "", "selected": True},
        {"capability": "x" * 1000, "selected": True},
        {"capability": "cap", "selected": "yes"},
        {"capability": "cap"},
    ]
    for bad in bad_bodies:
        status, _ = _req(
            server, "POST", "/api/select",
            body=json.dumps(bad),
            headers=_json_headers(server),
        )
        assert status == 400, bad
    assert server.calls["select"] == []


def test_unknown_route_404_and_methods(server):
    status, _ = _req(
        server, "GET", "/api/nope", headers=_auth(server)
    )
    assert status == 404
    status, _ = _req(
        server, "POST", "/api/nope",
        body="{}", headers=_json_headers(server),
    )
    assert status == 404
    status, _ = _req(
        server, "GET", "/api/pause", headers=_auth(server)
    )
    assert status == 405
    status, _ = _req(
        server, "POST", "/",
        body="{}", headers=_json_headers(server),
    )
    assert status == 405


def test_unknown_route_rejects_before_reading_body(server):
    """Route is validated first: a body that never arrives still gets a 404."""
    conn = _conn(server)
    conn.putrequest("POST", "/api/nope", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "100")
    conn.endheaders()
    resp = conn.getresponse()  # would hang if the server tried to read the body
    assert resp.status == 404
    resp.read()
    conn.close()


def test_stalled_body_times_out(server):
    conn = http.client.HTTPConnection(
        server.server_address[0], server.server_port, timeout=10
    )
    conn.putrequest("POST", "/api/pause", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "50")
    conn.endheaders(b"{")  # declare 50 bytes, send 1, then stall
    t0 = time.monotonic()
    resp = conn.getresponse()
    assert resp.status == 408
    assert time.monotonic() - t0 < 10
    resp.read()
    conn.close()


# -- API behavior --------------------------------------------------------------


def test_pause_roundtrip_and_select(server):
    status, body = _req(
        server, "POST", "/api/pause",
        body=json.dumps({"paused": True}),
        headers=_json_headers(server),
    )
    assert status == 200 and body["ok"] is True
    assert server.calls["pause"] == [True]
    status, body = _req(server, "GET", "/api/status", headers=_auth(server))
    assert body["paused"] is True

    status, body = _req(
        server, "POST", "/api/select",
        body=json.dumps({"capability": "translate-batch", "selected": True}),
        headers=_json_headers(server),
    )
    assert status == 200 and body["ok"] is True
    assert server.calls["select"] == [("translate-batch", True)]


def test_select_rejection_maps_to_409(server):
    def reject(capability: str, selected: bool) -> tuple[bool, str]:
        return False, "route requires vendor=gpu"

    srv = dashboard.make_server(
        status_provider=lambda: {},
        pause_handler=lambda p: None,
        select_handler=reject,
        prefetch_handler=lambda: None,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _req(
            srv, "POST", "/api/select",
            body=json.dumps({"capability": "c", "selected": True}),
            headers=_json_headers(srv),
        )
        assert status == 409
        assert body["ok"] is False and "vendor" in body["message"]
    finally:
        srv.shutdown()
        thread.join(5)
        srv.server_close()


def test_callback_failure_is_sanitized_500(server):
    def boom(paused: bool) -> None:
        raise RuntimeError("leaky detail: /home/user/secret-token-xyz")

    srv = dashboard.make_server(
        status_provider=lambda: {},
        pause_handler=boom,
        select_handler=lambda c, s: (True, ""),
        prefetch_handler=lambda: None,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _req(
            srv, "POST", "/api/pause",
            body=json.dumps({"paused": True}),
            headers=_json_headers(srv),
        )
        assert status == 500
        assert body["ok"] is False
        assert "secret" not in body["error"] and "leaky" not in body["error"]
    finally:
        srv.shutdown()
        thread.join(5)
        srv.server_close()


# -- prefetch and resource bounds ----------------------------------------------


def test_prefetch_single_flight(server):
    server.prefetch_release.clear()
    status, body = _req(
        server, "POST", "/api/prefetch",
        body="{}", headers=_json_headers(server),
    )
    assert status == 200 and body["ok"] is True
    assert server.prefetch_started.wait(5)
    status, body = _req(
        server, "POST", "/api/prefetch",
        body="{}", headers=_json_headers(server),
    )
    assert status == 409 and body["ok"] is False
    assert server.calls["prefetch"] == [1]
    server.prefetch_release.set()


def test_prefetch_reusable_after_completion(server):
    """A completed prefetch frees the slot for a second run — deterministically."""
    server.prefetch_release.set()  # handler returns immediately

    def prefetch_once() -> None:
        server.prefetch_started.clear()
        status, _ = _req(
            server, "POST", "/api/prefetch",
            body="{}", headers=_json_headers(server),
        )
        assert status == 200
        # Synchronize on the worker, not the response: 200 means the thread
        # was started, not that its callback ran.
        assert server.prefetch_started.wait(5)
        server._prefetch_thread.join(5)
        assert not server._prefetch_running

    prefetch_once()
    prefetch_once()
    assert server.calls["prefetch"] == [1, 1]


def test_server_close_joins_prefetch(server):
    """server_close() returns only after the prefetch callback has exited."""
    server.prefetch_release.clear()
    _req(
        server, "POST", "/api/prefetch",
        body="{}", headers=_json_headers(server),
    )
    assert server.prefetch_started.wait(5)  # callback is blocked mid-run
    closer = threading.Thread(target=server.server_close, daemon=True)
    closer.start()
    closer.join(0.5)
    assert closer.is_alive()  # blocked joining the live callback
    server.prefetch_release.set()
    closer.join(5)
    assert not closer.is_alive()


def test_concurrency_bound(server):
    gate = threading.Event()
    entered = threading.Event()

    def slow_status() -> dict:
        entered.set()
        gate.wait(5)
        return {"agent_id": "w1"}

    slow = dashboard.make_server(
        status_provider=slow_status,
        pause_handler=lambda p: None,
        select_handler=lambda c, s: (True, ""),
        prefetch_handler=lambda: None,
        host="127.0.0.1",
        port=0,
    )
    slow.max_concurrency = 1
    thread = threading.Thread(target=slow.serve_forever, daemon=True)
    thread.start()
    try:
        results = []
        errors = []

        def hit():
            try:
                results.append(
                    _req(slow, "GET", "/api/status", headers=_auth(slow))[0]
                )
            except Exception as exc:  # connection refused/reset counts as shed
                errors.append(exc)

        t1 = threading.Thread(target=hit)
        t1.start()
        assert entered.wait(5)  # t1 holds the single in-flight slot
        t2 = threading.Thread(target=hit)
        t2.start()
        t2.join(5)
        gate.set()
        t1.join(5)
        assert 503 in results or errors
        assert 200 in results
    finally:
        gate.set()
        slow.shutdown()
        thread.join(5)
        slow.server_close()


def _raw_status(srv, request_bytes: bytes) -> int | None:
    """Send a complete raw request; return the response status, or None."""
    sock = socket.create_connection(
        (srv.server_address[0], srv.server_port), timeout=5
    )
    try:
        sock.sendall(request_bytes)
        data = b""
        while True:
            head, sep, rest = data.partition(b"\r\n\r\n")
            if sep:
                wanted = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        wanted = int(line.split(b":", 1)[1])
                if len(rest) >= wanted:
                    break
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            data += chunk
        head = data.split(b"\r\n", 1)[0]
        if not head.startswith(b"HTTP/"):
            return None
        return int(head.split(b" ", 2)[1])
    finally:
        sock.close()


def test_incomplete_header_connections_bounded(server):
    """The cap binds accepted connections, including unauth incomplete ones."""
    server.max_concurrency = 4
    server.request_timeout = 30.0  # held connections must not lapse mid-test
    held = []
    try:
        for _ in range(4):
            sock = socket.create_connection(
                (server.server_address[0], server.server_port), timeout=5
            )
            # Partial headers: no request line completed, no auth, never sent.
            sock.sendall(b"GET /api/status HTTP/1.1\r\nHost: 1")
            held.append(sock)
        for _ in range(50):
            if server._inflight >= 4:
                break
            time.sleep(0.05)
        assert server._inflight == 4

        extra = socket.create_connection(
            (server.server_address[0], server.server_port), timeout=5
        )
        try:
            try:
                data = extra.recv(4096)
            except OSError:
                data = b""
            # Shed before any request bytes: a readable 503, or a closed
            # connection — never a request thread or an auth challenge.
            assert b"503" in data or not data
        finally:
            extra.close()
    finally:
        for sock in held:
            sock.close()


def test_request_deadline_bounds_trickled_headers(server):
    """A header byte per interval must not extend the request deadline."""
    sock = socket.create_connection(
        (server.server_address[0], server.server_port), timeout=5
    )
    t0 = time.monotonic()
    try:
        sock.sendall(b"GET /api/status HTTP/1.1\r\nHost: 1")
        # Trickle one byte roughly every timeout/4 — with a per-operation
        # timeout this would hold the connection past request_timeout.
        while time.monotonic() - t0 < server.request_timeout + 3:
            try:
                sock.sendall(b"0")
            except OSError:
                break  # server closed the connection at the deadline
            time.sleep(server.request_timeout / 4)
        try:
            sock.recv(4096)
        except OSError:
            pass
        assert time.monotonic() - t0 < server.request_timeout + 5
    finally:
        sock.close()


def test_request_deadline_bounds_trickled_body(server):
    """Trickled POST body bytes cannot extend the total request deadline."""
    conn = http.client.HTTPConnection(
        server.server_address[0], server.server_port, timeout=10
    )
    conn.putrequest("POST", "/api/pause", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "50")
    conn.endheaders(b"{")
    t0 = time.monotonic()
    while time.monotonic() - t0 < server.request_timeout + 3:
        try:
            conn.send(b"x")
        except OSError:
            break
        time.sleep(server.request_timeout / 4)
    # The server closed the socket at the deadline: drain a possible 408
    # then confirm EOF/reset follows promptly.
    conn.sock.settimeout(5)
    try:
        conn.sock.recv(4096)
    except OSError:
        pass
    assert time.monotonic() - t0 < server.request_timeout + 5
    assert server.calls["pause"] == []
    conn.close()


def test_server_close_blocks_partial_body_callback(server):
    """Close during a partial authenticated body: no callback may launch."""
    conn = http.client.HTTPConnection(
        server.server_address[0], server.server_port, timeout=10
    )
    conn.putrequest("POST", "/api/pause", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", f"Bearer {server.access_token}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "16")
    conn.endheaders(b'{"paused": tr')  # withhold the tail
    for _ in range(50):
        if server._inflight >= 1:
            break
        time.sleep(0.05)
    assert server._inflight == 1  # a request thread is parked on the body
    server.shutdown()
    server.server_close()  # must return: socket shutdown breaks the read
    try:
        conn.send(b'ue}')
        conn.getresponse()
    except (OSError, http.client.HTTPException):
        pass
    assert server.calls["pause"] == []
    conn.close()


def test_server_close_finishes_stalled_connections(server):
    """server_close() unblocks parked readers and joins request threads."""
    server.request_timeout = 30.0  # parked read can't expire on its own
    sock = socket.create_connection(
        (server.server_address[0], server.server_port), timeout=5
    )
    sock.sendall(b"GET /api/status HTTP/1.1\r\n")
    try:
        for _ in range(50):
            if server._inflight >= 1:
                break
            time.sleep(0.05)
        assert server._inflight == 1
        t0 = time.monotonic()
        server.server_close()
        assert time.monotonic() - t0 < 5  # not stuck on the parked read
        assert server._inflight == 0  # cleanup actually finished
        assert not server._request_threads
    finally:
        sock.close()


def test_non_ascii_authorization_rejected(server):
    conn = _conn(server)
    conn.putrequest("GET", "/api/status", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server.server_port}")
    conn.putheader("Authorization", "Bearer tok\u00e9n")  # latin-1 on the wire
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 401
    resp.read()
    conn.close()


def test_deeply_nested_json_rejected(server):
    """Max-depth nesting: 400 whether it parses to a list or hits recursion."""
    body = b"[" * 8000 + b"1" + b"]" * 8000  # just under MAX_BODY_BYTES
    status, body_obj = _req(
        server, "POST", "/api/pause", body=body, headers=_json_headers(server)
    )
    assert status == 400
    assert body_obj["ok"] is False
    assert server.calls["pause"] == []


def test_json_recursion_error_rejected(server, monkeypatch):
    """Force the RecursionError path: over-nested JSON must get a clean 400."""

    def deep(_text: str) -> Any:
        raise RecursionError("maximum recursion depth exceeded")

    fake_json = types.SimpleNamespace(
        loads=deep, dumps=json.dumps, JSONDecodeError=json.JSONDecodeError
    )
    monkeypatch.setattr(dashboard, "json", fake_json)
    conn = _conn(server)
    conn.request(
        "POST", "/api/pause", body='{"paused": true}', headers=_json_headers(server)
    )
    resp = conn.getresponse()
    assert resp.status == 400
    resp.read()
    conn.close()
    assert server.calls["pause"] == []


def test_duplicate_security_headers_rejected(server):
    auth = f"Bearer {server.access_token}".encode()
    base = (
        b"POST /api/pause HTTP/1.1\r\n"
        b"Host: 127.0.0.1:" + str(server.server_port).encode() + b"\r\n"
        b"Authorization: " + auth + b"\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 16\r\n\r\n"
        b'{"paused": true}'
    )
    dup = base.replace(
        b"Host: 127.0.0.1:" + str(server.server_port).encode(),
        b"Host: 127.0.0.1:" + str(server.server_port).encode()
        + b"\r\nHost: 127.0.0.1:" + str(server.server_port).encode(),
        1,
    )
    assert _raw_status(server, base) == 200
    assert _raw_status(server, dup) == 400

    origin = server.allowed_origin.encode()
    dup_origin = base.replace(
        b"Authorization:",
        b"Origin: " + origin + b"\r\nOrigin: " + origin + b"\r\nAuthorization:",
        1,
    )
    assert _raw_status(server, dup_origin) == 400

    dup_auth = base.replace(
        b"Authorization: " + auth,
        b"Authorization: " + auth + b"\r\nAuthorization: " + auth,
        1,
    )
    assert _raw_status(server, dup_auth) == 400

    # Only the unambiguous request reached the callback.
    assert server.calls["pause"] == [True]
