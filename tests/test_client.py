"""Regression tests for dworker.client.WorkerClient transport boundaries.

All offline: httpx.MockTransport handlers stand in for the control plane.
"""

from __future__ import annotations

import shutil
import socket
import ssl
import subprocess
import threading

import httpx
import pytest

from gpu_orchestrator.agent import AgentError, ControlPlaneClient
from gpu_orchestrator.models import AgentRegistration, SlotInventory

from dworker.client import WorkerClient
from dworker.config import ConfigError

TOKEN = "dgp_testsekrit123"
ENDPOINT = "http://127.0.0.1:8730"


def _client(handler, endpoint=ENDPOINT, **kwargs):
    return WorkerClient(endpoint, TOKEN, transport=httpx.MockTransport(handler), **kwargs)


def _registration() -> AgentRegistration:
    slot = SlotInventory(
        id="s0",
        gpu_index=0,
        gpu_uuid="u",
        vendor="nvidia",
        model="test gpu",
        vram_mb=1024,
    )
    return AgentRegistration(
        agent_id="agent-1",
        provider="local",
        hostname="host",
        cpu_cores=4,
        ram_mb=8192,
        disk_free_mb=10240,
        slots=[slot],
    )


# -- construction ------------------------------------------------------------


def test_is_control_plane_client_subclass():
    client = _client(lambda request: httpx.Response(200))
    try:
        assert isinstance(client, ControlPlaneClient)
        assert isinstance(client.client, httpx.Client)
    finally:
        client.close()


def test_constructor_enforces_endpoint_policy():
    for endpoint in (
        "http://public.example.com",          # plain HTTP off loopback
        "https://user:pw@ctrl.example.com",   # embedded credentials
        "https://ctrl.example.com/?x=1",      # query string
        "ctrl.example.com",                   # no scheme
    ):
        with pytest.raises(ConfigError):
            WorkerClient(endpoint, TOKEN)


def test_constructor_rejects_empty_or_blank_token():
    for token in ("", "   "):
        with pytest.raises(ConfigError, match="token"):
            WorkerClient(ENDPOINT, token)


def test_tls_verification_stays_enabled():
    client = WorkerClient("https://ctrl.example.com", TOKEN)
    try:
        context = client.client._transport._pool._ssl_context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
    finally:
        client.close()


def test_env_proxy_and_netrc_ignored(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.evil.example.com:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.evil.example.com:3128")
    client = _client(lambda request: httpx.Response(200))
    try:
        assert client.client._mounts == {}  # no proxy transports mounted
    finally:
        client.close()


# -- bearer confinement --------------------------------------------------------


def test_bearer_sent_to_origin_requests():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"agent_id": "agent-1"})

    client = _client(handler)
    try:
        client.register(_registration())
    finally:
        client.close()
    assert seen["auth"] == f"Bearer {TOKEN}"


def test_redirect_is_not_followed_and_returns_response():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/v1/agents/register":
            return httpx.Response(302, headers={"location": "https://evil.example.com/steal"})
        return httpx.Response(200, json={})

    client = _client(handler)
    try:
        with pytest.raises(AgentError, match="HTTP 302"):
            client.register(_registration())
        # explicit follow_redirects=True attempts are still refused
        response = client.client.get("/v1/workloads", follow_redirects=True)
        assert response.status_code == 200
    finally:
        client.close()
    assert all("evil.example.com" not in url for url in calls)


def test_absolute_url_to_other_origin_refused():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    client = _client(handler)
    try:
        for hostile in (
            "https://evil.example.com/artifact",
            "http://127.0.0.1:9999/v1/workloads",       # different port = different origin
            "https://127.0.0.1:8730/v1/workloads",      # different scheme
        ):
            with pytest.raises(AgentError, match="off-origin"):
                client.client.get(hostile)
        # same origin (absolute or relative) is allowed and authenticated
        assert client.client.get(f"{ENDPOINT}/v1/workloads").status_code == 200
        assert client.client.get("/v1/workloads").status_code == 200
    finally:
        client.close()
    assert len(requests) == 2
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in requests)


def test_transport_errors_become_agent_error_via_methods_and_direct_calls():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    client = _client(handler)
    try:
        with pytest.raises(AgentError) as caught:
            client.register(_registration())
        assert not isinstance(caught.value, httpx.HTTPError)
        assert TOKEN not in str(caught.value)
        # direct .client access (queues.fetch_workloads path) is normalized too
        with pytest.raises(AgentError) as caught_direct:
            client.client.get("/v1/workloads")
        assert not isinstance(caught_direct.value, httpx.HTTPError)
    finally:
        client.close()


def test_error_message_never_contains_token():
    def handler(request):
        # an error detail that managed to embed the token twice: bare and in a URL
        raise httpx.ConnectError(
            f"cannot reach {request.url} with credential {TOKEN}", request=request
        )

    client = _client(handler)
    try:
        with pytest.raises(AgentError) as caught:
            client.client.get(f"/probe?auth={TOKEN}")
        assert TOKEN not in str(caught.value)
        assert "***" in str(caught.value)          # bare token redacted
        assert "auth=" not in str(caught.value)    # URL query stripped entirely
    finally:
        client.close()


def test_http_error_status_does_not_echo_remote_body():
    secret = "remote-secret-phishing-text"

    def handler(request):
        return httpx.Response(500, text=f"fail: {secret}")

    client = _client(handler)
    try:
        with pytest.raises(AgentError, match="HTTP 500") as caught:
            client.register(_registration())
        assert secret not in str(caught.value)
    finally:
        client.close()


def test_claim_returns_none_and_parsed_json():
    def handler(request):
        if request.url.path.endswith("/claim"):
            return httpx.Response(204)
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    try:
        assert client.claim("slot-1", 90) is None
    finally:
        client.close()


def test_constructor_rejects_non_finite_or_bool_timeout():
    for bad in (True, False, float("nan"), float("inf"), float("-inf"), 0, -1, "60"):
        with pytest.raises(ConfigError, match="timeout"):
            WorkerClient(ENDPOINT, TOKEN, timeout=bad)


def test_embedded_userinfo_refused_even_on_origin():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    client = _client(handler)
    try:
        with pytest.raises(AgentError, match="userinfo"):
            client.client.get(f"http://{TOKEN}@127.0.0.1:8730/v1/workloads")
    finally:
        client.close()
    assert requests == []


def test_malformed_json_body_is_recoverable_agent_error():
    trap = "not json {also token? " + TOKEN

    def handler(request):
        return httpx.Response(200, content=trap.encode())

    client = _client(handler)
    try:
        with pytest.raises(AgentError, match="malformed JSON") as caught:
            client.register(_registration())
        assert TOKEN not in str(caught.value)
        assert trap not in str(caught.value)
    finally:
        client.close()


# -- trusted root environment -------------------------------------------------


def _openssl(*args):
    subprocess.run(["openssl", *map(str, args)], check=True, capture_output=True)


class _TlsHelloServer:
    """Minimal one-shot TLS server: answers every connection with a fixed 200."""

    def __init__(self, cert, key):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(10)
        self.port = listener.getsockname()[1]
        self._context = context
        self._listener = listener
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._done.is_set():
            try:
                conn, _ = self._listener.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                try:
                    with self._context.wrap_socket(conn, server_side=True) as tls:
                        tls.recv(65536)
                        tls.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                            b"Connection: close\r\n\r\n{}"
                        )
                except (ssl.SSLError, OSError):
                    pass  # expected when the client (rightly) rejects the cert

    def close(self):
        self._done.set()
        self._listener.close()
        self._thread.join(timeout=10)


@pytest.fixture
def private_ca(tmp_path):
    """An isolated CA + leaf for 127.0.0.1 minted locally; no external deps."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    ca_key, ca_crt = tmp_path / "ca.key", tmp_path / "ca.crt"
    leaf_key, leaf_csr, leaf_crt = tmp_path / "leaf.key", tmp_path / "leaf.csr", tmp_path / "leaf.crt"
    ext = tmp_path / "leaf.ext"
    ext.write_text("subjectAltName=IP:127.0.0.1\n")
    _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", ca_key, "-out", ca_crt, "-days", "1", "-subj", "/CN=test-ca")
    _openssl("req", "-newkey", "rsa:2048", "-nodes",
             "-keyout", leaf_key, "-out", leaf_csr, "-subj", "/CN=127.0.0.1")
    _openssl("x509", "-req", "-in", leaf_csr, "-CA", ca_crt, "-CAkey", ca_key,
             "-CAcreateserial", "-out", leaf_crt, "-days", "1", "-extfile", ext)
    server = _TlsHelloServer(leaf_crt, leaf_key)
    yield server, ca_crt
    server.close()


def test_private_ca_rejected_without_trusted_root(private_ca):
    """Verification still bites: an unknown CA fails the handshake."""
    server, _ = private_ca
    client = WorkerClient(f"https://127.0.0.1:{server.port}", TOKEN)
    try:
        with pytest.raises(AgentError, match="certificate verify failed"):
            client.client.get("/ping")
    finally:
        client.close()


def test_ssl_cert_file_trusts_private_ca(private_ca, monkeypatch):
    """A deliberately configured SSL_CERT_FILE private CA is honored."""
    server, ca_crt = private_ca
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_crt))
    client = WorkerClient(f"https://127.0.0.1:{server.port}", TOKEN)
    try:
        response = client.client.get("/ping")
        assert response.status_code == 200
    finally:
        client.close()


def test_ssl_cert_dir_trusts_private_ca(private_ca, monkeypatch, tmp_path):
    server, ca_crt = private_ca
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    target = cert_dir / "ca.pem"
    target.write_bytes(ca_crt.read_bytes())
    _openssl("rehash", str(cert_dir))
    monkeypatch.setenv("SSL_CERT_DIR", str(cert_dir))
    client = WorkerClient(f"https://127.0.0.1:{server.port}", TOKEN)
    try:
        assert client.client.get("/ping").status_code == 200
    finally:
        client.close()


def test_hostname_mismatch_still_rejected_with_trusted_ca(private_ca, monkeypatch):
    """Trusting the CA does not weaken hostname validation."""
    server, ca_crt = private_ca
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_crt))
    # 'localhost' resolves to 127.0.0.1 but the leaf only carries IP:127.0.0.1
    client = WorkerClient(f"https://localhost:{server.port}", TOKEN)
    try:
        with pytest.raises(AgentError):
            client.client.get("/ping")
    finally:
        client.close()
