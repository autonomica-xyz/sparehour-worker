"""Dynamic-mode provisioning tests: deterministic pack archives, real
manifest/digest verification, and manager state transitions.

All pack fixtures are real .argosmodel zips built locally and hashed for
real. Every remote read runs through the owned fetch subprocess
(`sparehour_worker.fetch`): tests use genuine loopback TLS servers — a listener
that accepts but never completes TLS, servers trickling headers/body —
with the client's certificate verification narrowed only by trusting the
test CA through the standard SSL_CERT_FILE environment.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse


import pytest
from sparehour.models import SlotInventory

from sparehour_worker import dynamic as dyn

HOST = {"cpu_cores": 16, "ram_mb": 65536, "disk_free_mb": 1000000}

# request path -> archive bytes or ("redirect", target); populated by
# make_pack and served by the loopback TLS archive server.
_ARCHIVES: dict = {}

# PKI material lives in one module temp dir: the fetch child trusts the CA
# via SSL_CERT_FILE (the default verified context), the TLS server loads
# the server cert once per fixture.
_PKI_DIR = Path(tempfile.mkdtemp(prefix="sparehour-worker-test-pki-"))
_CA_FILE = _PKI_DIR / "ca.pem"
_SERVER_CERT = _PKI_DIR / "server.pem"
_SERVER_KEY_FILE = _PKI_DIR / "server.key"
_ARCHIVE_SERVER = None  # bound by _serve_archives while a test runs


@pytest.fixture(autouse=True)
def _serve_archives(monkeypatch):
    """Real loopback TLS server for pack archives and remote catalogs."""
    _ARCHIVES.clear()
    monkeypatch.setenv("SSL_CERT_FILE", str(_CA_FILE))
    requests = []

    def serve(sock, halted):
        target = _drain_request(sock)
        requests.append(target)
        path = urlparse(target).path
        entry = _ARCHIVES.get(path)
        if entry is None:
            sock.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        elif isinstance(entry, tuple) and entry[0] == "redirect":
            location = str(entry[1]).format(port=server.port)
            sock.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: " + location.encode()
                + b"\r\nContent-Length: 0\r\n\r\n"
            )
        elif isinstance(entry, tuple) and entry[0] == "status":
            sock.sendall(
                f"HTTP/1.1 {entry[1]} {entry[2]}\r\n".encode()
                + b"Content-Length: 0\r\n\r\n"
            )
        elif isinstance(entry, tuple) and entry[0] == "stall":
            sock.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                b"Content-Length: " + str(len(entry[1])).encode()
                + b"\r\n\r\n" + entry[1][:8]
            )
            halted.wait(120)  # rest of the body never arrives
        else:
            sock.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                b"Content-Length: " + str(len(entry)).encode() + b"\r\n\r\n" + entry
            )

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    global _ARCHIVE_SERVER
    _ARCHIVE_SERVER = server
    try:
        yield requests
    finally:
        server.close()


def _inv(**overrides):
    base = dict(
        id="s0",
        gpu_index=0,
        gpu_uuid="u",
        model="NVIDIA GeForce RTX 4090",
        vram_mb=24564,
        compute_capability="8.9",
        capabilities=[],
        model_packs=[],
        labels={},
    )
    base.update(overrides)
    return SlotInventory(**base)


def _row(capability="translate-batch", pending=1, packs=(), model_dir=None, sha=None, route="r1"):
    environment = {}
    if model_dir:
        environment["ARGOS_MODEL_DIR"] = model_dir
    if sha:
        environment["ARGOS_MODEL_ARCHIVE_SHA256"] = sha
    return {
        "workload": "w",
        "route": route,
        "capability": capability,
        "model_packs": list(packs),
        "resources": {"vendor": "nvidia", "min_vram_mb": 1},
        "executor_type": "command",
        "executor_detail": "",
        "executor_environment": environment,
        "pending_jobs": pending,
    }


def make_pack(tmp_path: Path, name="argos-en-es-1.0", root="en_es",
              from_code="en", to_code="es",
              files={"sentencepiece.model": b"sentencepiece bytes", "model/model.bin": b"weights"}):
    """Build a real .argosmodel zip and its lock entry (sha256 computed)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            f"{root}/metadata.json",
            json.dumps({"from_code": from_code, "to_code": to_code}),
        )
        for member, data in files.items():
            bundle.writestr(f"{root}/{member}", data)
    archive = tmp_path / f"{name}.argosmodel"
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = buffer.getvalue()
    archive.write_bytes(payload)
    path = f"/{name}.argosmodel"
    lock = {
        "model_pack": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": _ARCHIVE_SERVER.base_url + path,
        "archive_root": root,
        "from_code": from_code,
        "to_code": to_code,
    }
    _ARCHIVES[path] = payload
    return lock, archive


def _plan(rows, **kw):
    defaults = dict(
        slots=[_inv()],
        provider="local",
        host=HOST,
        agent_labels={},
        installed_packs=set(),
        catalog={},
        fallback_dir=Path("/fallback"),
    )
    defaults.update(kw)
    return dyn.plan(rows, **defaults)


def _fetch_children():
    """Live fetch subprocesses owned by this test process (Linux /proc)."""
    running = []
    for entry in Path("/proc").iterdir() if Path("/proc").exists() else []:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            if int(stat.rsplit(") ", 1)[1].split()[1]) != os.getpid():
                continue
            if "sparehour_worker.fetch" in (entry / "cmdline").read_text():
                running.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return running



# -- real loopback transport fixtures -----------------------------------------

# Static loopback PKI (valid until 2126): a CA, a server cert it signs with
# SAN DNS:localhost / IP:127.0.0.1, and a self-signed rogue cert. Clients
# under test keep real certificate verification; only the trust anchor is
# narrowed to this CA.
_CA_PEM = """-----BEGIN CERTIFICATE-----
MIIDJzCCAg+gAwIBAgIUWbgnAenFgrq+CgC/QNbkkspkKOowDQYJKoZIhvcNAQEL
BQAwGjEYMBYGA1UEAwwPZHdvcmtlci10ZXN0LWNhMCAXDTI2MTAxNDE5MDUwOFoY
DzIxMjYwOTIwMTkwNTA4WjAaMRgwFgYDVQQDDA9kd29ya2VyLXRlc3QtY2EwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCfWdBsKSfc2LyVxRwknoBcX6i1
G04If3Wm46/pY+E5g21l21V7zmdZlMoZGFsGYeTCUMOKGlRKo7iISf0yA40NtzA7
xO9l++n0cD81unF77dn8N2xWCIk+JmtSS8/5ywwDj+E5xOMvONZzdXPaZeh1crbg
fhIZsZduYQIYB3Bwb1JPUIjvdILlJP75eK/8apd2/KmmGZosW1yd8XHsxkUUdmhc
Xlm3JkxzModMLCV/rtJ0fSJ9Iv4YrJj3VcbKIeQh4+25MuTTFmniA4iEPxDa0Pgh
+yxWYsauNRBZGMk3qlp0dLbi1cNK9E5XIyUELuPEPpUDrNAbEZG92+x9oIDhAgMB
AAGjYzBhMB0GA1UdDgQWBBTlsWB2MlQOWnkpFdybiS5nsUV2RTAfBgNVHSMEGDAW
gBTlsWB2MlQOWnkpFdybiS5nsUV2RTAPBgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB
/wQEAwIBBjANBgkqhkiG9w0BAQsFAAOCAQEAA67GSwGgppDoM3buXiaRu2aTLIlF
j2NW/IKgGaGYpmwjuc/N1ob5j+IwCi3uEEaebjO8kj5SIg9DI5FYQ01ETk59gt2A
U4JRtwCQ8iWSXgL8F6qNFy2jxehuaJ7a6zjHzv9R7+nMgbF7rG3ZC1ENojg8iKSq
ywjXjIFsb+SMSBuAialq8lIaWoddK8bqvwRSuJTAkaGBJ8fmplQz7bJF6ORoVNBU
xvGI8rKxbv1C2LXBS0XuMJA8SNLjX4R+b7FMaaCXVBxTH73KGHtnBFqa4dkmvOPj
hOY/a9cmQ78V/v80Bg7aQyMxbiGNz97hHdc0WFHEHCXOgR/OLk6ysA/vKA==
-----END CERTIFICATE-----
"""

_SERVER_PEM = """-----BEGIN CERTIFICATE-----
MIIDSzCCAjOgAwIBAgIUd9aOfmGO8i7k27tWft6X1br2mBEwDQYJKoZIhvcNAQEL
BQAwGjEYMBYGA1UEAwwPZHdvcmtlci10ZXN0LWNhMCAXDTI2MTAxNDE5MDUwOFoY
DzIxMjYwOTIwMTkwNTA4WjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQCqQ+hUhTj+rpL3N7r+DgPt6dfZlCzyEm96
qYWdBTuYVTWEUdnOw0bNxzLfG9mzQuCeu5tEtl7rqBlIX4MhLAyJjx0m4Qdg3mU/
o3/GilQBDV5xYNwr/jsmPaWoLqp1kXCTN6Se/hGYRfx05RUyeps7o0mHRLTccU+N
HwHuPXT7WUB07L8HID1YCvgABBWEuBfm8TH2USpwD69v9XDYhDgTXaAhZ4auH0++
+FMwEIdDlq3h7IrKHnxvKVVxbc22NkpjWRf21Ab1SPdwN2UKCnWtLYxrs2FyMb2P
vFzTrQ1jV3noaPLJ2sKVWIQjpCiDQZkw0mZDOihDlpAKSU5J3hGhAgMBAAGjgYww
gYkwGgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMB0GA1UdDgQWBBRGRIdTnlFh2/jr
kXNUuwQsYgA6OTAfBgNVHSMEGDAWgBTlsWB2MlQOWnkpFdybiS5nsUV2RTANBgkq
hkiG9w0BAQsFAAOCAQEAdueSvzAVN/mb+tGw9Rg/Eivi6P2HDD2aTskaNNP8Xn7J
zON+7v6DWdgIwq8qFz/y5Ga/+ZYzqTX7GjG7xoLjaObUHeX7S8rQOL+cW8Gv8oOq
bsGL1jPlb2vG5f6nj79N8G8RmKtw9QM6JCoGvLd6cb1fCVR3bq+gvWM2ur5LuSCS
ZeiM8L3uYje+SB1cYroMICmVWghLvmMyMG0HUqELHRPGMERPxrY1y7eyMPVdbzai
LBfOHvXzUHYXS2EFbNrH8I8Hcma4iBGy35q/MH0NvHgaMwrNxYtSO1VWchIiEKYL
4GIUs1kUsWMZ/bsrDPYM4Q80frtoOt99wNkQLfdbpw==
-----END CERTIFICATE-----
"""

_SERVER_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCqQ+hUhTj+rpL3
N7r+DgPt6dfZlCzyEm96qYWdBTuYVTWEUdnOw0bNxzLfG9mzQuCeu5tEtl7rqBlI
X4MhLAyJjx0m4Qdg3mU/o3/GilQBDV5xYNwr/jsmPaWoLqp1kXCTN6Se/hGYRfx0
5RUyeps7o0mHRLTccU+NHwHuPXT7WUB07L8HID1YCvgABBWEuBfm8TH2USpwD69v
9XDYhDgTXaAhZ4auH0+++FMwEIdDlq3h7IrKHnxvKVVxbc22NkpjWRf21Ab1SPdw
N2UKCnWtLYxrs2FyMb2PvFzTrQ1jV3noaPLJ2sKVWIQjpCiDQZkw0mZDOihDlpAK
SU5J3hGhAgMBAAECggEACU0SiD7LKq3RL/Q+q18LIRLvkb50h8MbjWUXfDN7ZQUv
RPuZV8rsQin8p1pKCxmfE8oWVODXMG2QogGRCX3H7+IeRXqF+2lIDnupQNMw0b+P
9kbEXZqH/dJgy15EkrhqS1R/EkNj+oBE5q2BT1kKGVjVwlqtvEbBVylyOCLp0iAg
HClI7MhpVBK6LW7vKnwteeV5BQNHhqGJoP0dt2pGrisQy+HPvEeKa6m3cMo/Q7/p
t/Mo6oH3t7pGWRe8l+cgloV6hhSG8t0UulIRDheLkoWvjMp8d+kXdEebsDu55vmb
G713v88k9Vilq2oDEG0I51yVywHmI7jnXaFepw15BQKBgQDb38uRa04kuGuoFH9R
Abo7S2THUX4mbS8YUzkNA2GXccGM9kLpMX3+U2ambmuqXiVfcf0pN43XgOtBi+KU
IrTQyFJkE5FXF33SSmVgDUArvPMtl0hSSt5SJEFmkvsDzo6oO1jx22qi/+98QG7T
r/gSXrY6/8KGyuEWPb5GBFFJ1QKBgQDGPX5klidP8/RdCTQsJgg9LBE15xh4Qf1G
k2/58Ng61WMD2hmvIvPxEXX1ZRU/6NINIbKTA23GStyyst6L9bdkSTMqKuUI8p4k
mcDg8VR81Z1xm58UZJW+7NBKIq4/8pnGCZMo3bBuNLg9weu/HNMlxQsAldXUkPny
9vWj+J+inQKBgBTx7a9tnGUNGFxLj5FE69uoI/KExdKdRgY4L1AZCVvm3zwxXVBf
gZQPl9C05ha1nzPHZQlfHGjyA/cbfzsDT1wwKQEViBvWxgqZQazztjA0cKngJmiI
b0M95ukBiV3keFx4QtHPSAo8/TBF5wjNU4FUemGGaZ5zhxuaSBzdXmz9AoGAXPy5
TctU8172L7HChDbuxGroQ8V7p+UkPbeo3a+IbchsIoFqbMlTba+ICix5FDpMVRaK
YgVIt6uj7DqUw0fl1yey1UO7nT/YyEhKvfjVbZzxtjmOHA22ZBFG/o9X9u0ADnGt
UozyVi9qgKV2xHVr4ZqUt5iQ7Z13cO1AecKEds0CgYEAgDbiVvzDv1BaGAe872i9
1qetG5qmw8ANqFnLw1MHhauS0901Z4mPi1B6sxQhgJ9aC4E+Kvh3DDW1LOswWOtf
d73QdlCmM+G8Uffhdq11dH+f1Y0iqcQFekSOLiBkDkxJo68ZVLhNUmjQ8qdEUVsM
7znC57V1qsA7UqV8Y9QFA9c=
-----END PRIVATE KEY-----
"""

_ROGUE_PEM = """-----BEGIN CERTIFICATE-----
MIIDJzCCAg+gAwIBAgIUJQlWKkIu2klStqPfBTHAlDMc1akwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MCAXDTI2MTAxNDE5MDUwOVoYDzIxMjYw
OTIwMTkwNTA5WjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQDJTXguyasYMp7CqnUffvKVqY9Oy+PMGU8lLDqnhe+6
O7bln9C8r/ygDlpiOd2cdYiTkEClEvzw70VW5PTkXpS2yGJfOrNJexuS9L38yPdL
waOJ0RP+UFXm5w99wh3zLbgNOKF50LFShi9xjbadtjvU3XReEPhD4054KfvyYjDT
pWgvSjXCtZKV0EJrCWxqn1ABnmamwPVUHE3Q+It64P35OnVX/yO5E91Q/0cXhk3J
NRy+Z/VsZUz9yVNfPD9MZVXwCEaLD62IqgmPt5Q+7R+FE9IiucPh2dDSwpv2tGKU
K92swfOGMBtgt0bPHEn29z7yrNRAfdKq41HOkSjymV3lAgMBAAGjbzBtMB0GA1Ud
DgQWBBS5HaQzqW0QwC1EV/I/f/SatnufsTAfBgNVHSMEGDAWgBS5HaQzqW0QwC1E
V/I/f/SatnufsTAPBgNVHRMBAf8EBTADAQH/MBoGA1UdEQQTMBGCCWxvY2FsaG9z
dIcEfwAAATANBgkqhkiG9w0BAQsFAAOCAQEAuVKsUVx+wEYcAXV16V+/Zrq3VM5m
k1OypPq1m0ovnbdWNaDh+wqal3rVn2kYZ3y9ABBZz61M9+hszXEf2+JokRqVjDX2
d5MQ+rmX9/ORS/iX6YNM+UFRR5zy8+WoAKSuIw9tTxBirsnzxNycA8UglykdC8u4
t4ktWGCD3RG9AZ4KmLL004m9NxeazDfaby1kBs/YBeCMOjuSQ7SGQ/hzuSJ6thF4
tmyuL4PV2PU/AxZZtKmJPVksC0qWqPYqhusqjdo6KGlkmNN+K0XA0mbx7b7w9OvG
GCxP4isLd2whnxxQEc29qjw8T1WGPWz9WNsYZ9L4tY2SzWqoAjl2WuLvzw==
-----END CERTIFICATE-----
"""

_ROGUE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDJTXguyasYMp7C
qnUffvKVqY9Oy+PMGU8lLDqnhe+6O7bln9C8r/ygDlpiOd2cdYiTkEClEvzw70VW
5PTkXpS2yGJfOrNJexuS9L38yPdLwaOJ0RP+UFXm5w99wh3zLbgNOKF50LFShi9x
jbadtjvU3XReEPhD4054KfvyYjDTpWgvSjXCtZKV0EJrCWxqn1ABnmamwPVUHE3Q
+It64P35OnVX/yO5E91Q/0cXhk3JNRy+Z/VsZUz9yVNfPD9MZVXwCEaLD62IqgmP
t5Q+7R+FE9IiucPh2dDSwpv2tGKUK92swfOGMBtgt0bPHEn29z7yrNRAfdKq41HO
kSjymV3lAgMBAAECggEAZB6zRTYJckYO/0wPZ8RYfAZy3+J5oJP/8MGgRJzRgENd
8rQ7efVfxO2Ol6cnKT+t15LaDCA3DNA8oPzMLl86QPTGNI36CPSBYOR4L+d2U8fy
AkQqWQT0zJykcExhYAcfsPe3HGKnJx/VLfmpV6SA+pzQWU7AY+eEqjBbfpDGrySF
tQTJmVVy/R6wmWNlDd4Z8cSRdGmDDnCujxJtnV+bD6WdF+SqnSp/npgBh1+SVbWp
Q1n7rSnHkh3kHewZH1yxsXGzenuOZb7yheyBU75KTqGEw1EUSFuMauLJokHqzsCX
NKw4nuAYil4J/ghBrbQriZkLbzh68cA7pZthjLc+OwKBgQDyKxcxDY+ztJ/+5vO6
PK2HGffVUsTQ96MwRuZ3hNqqVw/2+FB6X9yzWl2aqU4zeNaUAQ3mfPuvENGShaPO
JlNjjKjEgLZNh7l3BAzK4JFf1YuGIglCijbutxVmJXsjFHo2wJBuqhkPV6Nq3Dps
1mmUEDwkRsaQ5UVYz86B2GZeFwKBgQDUzNpxGfuqW0RtQCwvmnKzA2jz2Mswc90p
XoFkmhj/7dsl9oVrwYuKGgIZIfQfSIDxKg4PddPZn6CZ7oU2JKFfkTuFsul4b0KO
fnbv/emJquRkU8zTTznjDxbWICyLP3BC4di4mqFlpuxsFPtSAydZgz466Wx4qB7d
55JYsFS9YwKBgQCaUfFWdgDkEK+AfWZFgNNH67lt/G+lhpk8c4H8GfJKaG6pADYw
tZPryyMpIsjNrix3ZnCYN38fYYVbbMQk43FyV7lYd3k00M/UUwVf51lci7V8xYXH
HFLDLnJ5hFsC0NLMv0uK6eRMTZm6yBqHLt9a1jmtGFkDDsdxQAOl5OPfAwKBgH3g
m8rRkuQHzudNUZXrJ7Qq7TdpY3BK6WViSOoomWcy3TKMrAm82H0du2YAhrbGnbeK
p/yPN30BBvBnfpczfd3ZPDtvRNn//n/p4RM/wJi15RpP09afg9+Den+5UoAgS5jU
ne09UwGdhza8BppR0UIK95HDqG2LDY84P4KE8+t/AoGAFugc0U8msyVkxK4zkIVU
kXwi/LnAESJK0g6LKUkPrO9JWEfGoRh3Trj0d4UmJ067NEjgkud9B2dkdSCEYLI8
raDbW4szTJ/W9dluDtdtIBhR8Crme4m5PRBKiKrPP3XgNH3La4C6VzHUrLfQiGSk
FZUZNLZlgxc9bCC7Q92E5mE=
-----END PRIVATE KEY-----
"""


_SANFAIL_PEM = """-----BEGIN CERTIFICATE-----
MIIDHjCCAgagAwIBAgIUKog3mDV2Ch5IdDacoMoIAKddp9YwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJb3RoZXJob3N0MCAXDTI2MTAxNDE5NDg1MFoYDzIxMjYw
OTIwMTk0ODUwWjAUMRIwEAYDVQQDDAlvdGhlcmhvc3QwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCUyW4sA7UrgcGTequCqp44G6z0eTSE3ygJbS4Nc7GU
FtHe7Q6K735E3gJ96pVswxY5m+CNCbOlEi4j7OW747J9jnWjUjDLgYhsrzLprtGA
P+Ytx/pUDjHd7GaG+HEriHvYWmRjObIAl1j+qBxkyPIsXCUUVy76fGNsegUXnOoM
O13/7z5/kg7jZLaHl2wRnR2ngRopDUCCea+gek65vTvOeOKCdUG2q3vZrz1fV36N
4ei8z1CUFKUquPYB5NB9dVkYOjuEGgD86rW2YO7PdGRf+uEqayYFfMDxPwvTBv7o
O240GeST0MOj8vLKP0C1dRhGFtWJ8lf8LfIiXAemA2vbAgMBAAGjZjBkMB0GA1Ud
DgQWBBRrrHv8HJqTouVsv+WwyO2Z8YJS3DAfBgNVHSMEGDAWgBRrrHv8HJqTouVs
v+WwyO2Z8YJS3DAUBgNVHREEDTALgglvdGhlcmhvc3QwDAYDVR0TBAUwAwEB/zAN
BgkqhkiG9w0BAQsFAAOCAQEAgG5vNuicNvNntBS6Ic51eQ+J/Rq2G8P8Q05kVbRp
+VYxqUVPnvrY16nq9K2zr0j53j3rKOn1CrJ7EnJPR6Tya/KjhADcsbAwRQeOxhRQ
pgbJV8ddwL67P6nfmGALXE9Sro3/JPRZflGl2W8TJAGC6bQyju2x5EHz9YEFuoMr
6IVQ9OrtGfEEGiKh1AidoJ5R+yf3HE0YHPPgvmUZsx7GuvB5A23mziHYQLkMKQP7
/TwdcCrJAjTSGk5T/qWHLZnsHHERPwHxzLOjKYzVfZSuvsAFZ+YXhiCMBNVA3pX0
ALjzA0HVVVE9DMUUhHYlvzN3sWYESkAjaciK/1w7me3K4A==
-----END CERTIFICATE-----
"""

_SANFAIL_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCUyW4sA7UrgcGT
equCqp44G6z0eTSE3ygJbS4Nc7GUFtHe7Q6K735E3gJ96pVswxY5m+CNCbOlEi4j
7OW747J9jnWjUjDLgYhsrzLprtGAP+Ytx/pUDjHd7GaG+HEriHvYWmRjObIAl1j+
qBxkyPIsXCUUVy76fGNsegUXnOoMO13/7z5/kg7jZLaHl2wRnR2ngRopDUCCea+g
ek65vTvOeOKCdUG2q3vZrz1fV36N4ei8z1CUFKUquPYB5NB9dVkYOjuEGgD86rW2
YO7PdGRf+uEqayYFfMDxPwvTBv7oO240GeST0MOj8vLKP0C1dRhGFtWJ8lf8LfIi
XAemA2vbAgMBAAECggEAEgKOZJbPqZiTW9sio5kUSMcQF+tbvTbHaaeoDt0ROKsV
rbx1j/R5V/CKc1S5e78A9iDYgtGUKdTkzF/+R+6FsQxZzYf9o6AIm6JMDXnyd5KY
MVurr1spoS9RH5yb/sPeRcY0Ih8fEk7G+/eZbjwaU/3LCT86bUAfZZlcDiWQ4s7p
FKO53d3xvrYDA1uf8JPBz9bGzx9LhuhPEPYSzpAuomVGqIPMiCoqAw2SX0medTTR
/FhCMAohgWSSlgL1kBoZSHPzQ2+AwOKyrR5KM4/5y8wRdmNSiiu+pWGgXHpzm5jh
kOmgUxGrJij1RAIsRT1g60eg2YJzgHaineyyAF9dgQKBgQD9AgKl6A57CxiB9fru
/8py84ion7zqt4ok2XtVGUiXc1b3kaWWReYwwr8G93oCv0gnkJ95B8iYhni+1IYn
TJOeoT7Pd8xYFK60PbixZj6Zww5Nw80IXWP9VbxsY043QuBEGeBYXxsrF5QrjEQg
q2gKUEFTJmZGnrloh9+LfsARjQKBgQDnF1ew73ICAc9yvHkypp/1Gj8cv9SAQJtu
Wodg5lIiVj+J9Vy1EUS4Da5Aa1/kPTLqRr0uZVfLS1q08EY0PBHoW+q/NDGJ7WAZ
bHsNfj0jF9A58hwzXx7qT/eGNVyOMQ8rMv3b9iqhqzLKHFTTHB55/M/C7M4NOXGA
kJ1dYxRsBwKBgE4t/9l6Qnc8737BHqAZhFpO7IyGh57Txikq1tRt8qNx7Y6kf9FW
mhG5K8QgVqBgsRAgvVZt1EtlSgxVqXqdTAQ5eSosr4UMJwvlApw2puNmhZX8CfAq
w4n1ozqu2O10ic4hXNSR1thS+eMW/bpGnOiK8hSaigIKa+pkCeyV5w3FAoGBAMUY
i5pFHx/HG3/ocBZbNyJ9eEa09jFM6UR9welLh9Hxbc/kjWWBjPLN+EjJS8gGyl42
ZzQjC3ZAluSr6bSzZzKe8uCRS48v/j6w32LqwIolzVsevB60ilGeUqcM7Bv6i810
eEAa/NFWisMG1vIXkYbJsQm3nyZ4T+4vaYg08T4RAoGBAOIW7QPicq9iKBrRPq+y
u1DzMt0e1T4CTy83QoXXGXujMqKovHy6enggr7AC5o/SJbEgU/n4DZfFt2oIvCd1
KO6FaPz7g0Cv1FdsjsbAFXPoVyVru7qKWvTU67KJLkykeAYXksTA1e4VgfX8fQBV
YU8yZpDm8VSdc0CYpd2vO0Je
-----END PRIVATE KEY-----
"""

# The loopback PKI only authenticates the test server; write it once so the
# fetch child can trust it via SSL_CERT_FILE.
_CA_FILE.write_text(_CA_PEM)
_SERVER_CERT.write_text(_SERVER_PEM)
_SERVER_KEY_FILE.write_text(_SERVER_KEY)
_ROGUE_CERT = _PKI_DIR / "rogue.pem"
_ROGUE_KEY_FILE = _PKI_DIR / "rogue.key"
_ROGUE_CERT.write_text(_ROGUE_PEM)
_ROGUE_KEY_FILE.write_text(_ROGUE_KEY)
_SANFAIL_CERT = _PKI_DIR / "sanfail.pem"
_SANFAIL_KEY_FILE = _PKI_DIR / "sanfail.key"
_SANFAIL_CERT.write_text(_SANFAIL_PEM)
_SANFAIL_KEY_FILE.write_text(_SANFAIL_KEY)
# Trust bundle for the SAN-mismatch test: the self-signed "otherhost" cert
# chains (it is a root anchor) but names no localhost address.
_SANFAIL_BUNDLE = _PKI_DIR / "sanfail-bundle.pem"
_SANFAIL_BUNDLE.write_text(_CA_PEM + _SANFAIL_PEM)


class _SilentTcpListener:
    """Real listener that accepts TCP and never completes TLS."""

    def __init__(self):
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.2)  # accept() must wake for close()
        self.port = self._listener.getsockname()[1]
        self.url = f"https://127.0.0.1:{self.port}/models.json"
        self._held = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drive, daemon=True)
        self._thread.start()

    def _drive(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._held.append(conn)  # held open: peer goes silent

    def close(self):
        self._stop.set()
        self._listener.close()
        for conn in self._held:
            conn.close()
        self._thread.join(5)


class _LoopbackTlsServer:
    """Real TLS server; ``serve(sock, halted)`` drives each connection after
    the handshake — stall, trickle, or reply; ``halted`` sets on teardown."""

    def __init__(self, cert: Path, key: Path, serve):
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(str(cert), str(key))
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self.port = self._listener.getsockname()[1]
        self.base_url = f"https://localhost:{self.port}"
        self.url = f"{self.base_url}/models.json"
        self._serve = serve
        self._held = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drive, daemon=True)
        self._thread.start()

    def _drive(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._held.append(conn)
            try:
                tls = self._context.wrap_socket(conn, server_side=True)
            except OSError:
                continue  # e.g. client cancelled mid-handshake
            self._held.append(tls)
            try:
                self._serve(tls, self._stop)
            except OSError:
                pass  # client wedged/closed the socket

    def close(self):
        self._stop.set()
        self._listener.close()
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass
        self._thread.join(5)


def _drain_request(sock):
    """Read the client's request head; return the request-target."""
    data = b""
    sock.settimeout(10)
    while b"\r\n\r\n" not in data:
        block = sock.recv(65536)
        if not block:
            break
        data += block
    try:
        return data.split(b"\r\n", 1)[0].split(b" ", 2)[1].decode()
    except (IndexError, UnicodeDecodeError):
        return ""


def _cancel_result(source, stop):
    """Run load_catalog in a caller thread; returns DynamicError or result."""
    outcome = []

    def caller():
        try:
            outcome.append(dyn.load_catalog(source, stop=stop))
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=caller, daemon=True)
    thread.start()
    return thread, outcome


def _assert_cancelled(thread, outcome, stop):
    stop.set()
    started = time.monotonic()
    thread.join(5)  # termination + reap is bounded by the poll interval
    elapsed = time.monotonic() - started
    assert not thread.is_alive(), "load_catalog did not return after stop"
    assert elapsed < 5
    (result,) = outcome
    assert isinstance(result, dyn.DynamicError)
    assert "cancelled" in str(result)
    assert not _fetch_children()

# -- pack directory verification -------------------------------------------
def test_verify_pack_dir_passes_real_build(tmp_path):
    lock, archive = make_pack(tmp_path)
    target = tmp_path / "installed" / lock["model_pack"]
    dyn.install_pack(lock, target)
    assert archive.exists()  # fixture file untouched
    assert not list(target.parent.glob("*.argosmodel"))  # temp archive cleaned up
    dyn.verify_pack_dir(target, lock)
    assert dyn.is_pack_installed(target, lock)


def test_install_pack_rejects_non_https_url(tmp_path, _serve_archives):
    lock, _ = make_pack(tmp_path)
    lock = dict(lock, url="http://packs.invalid/model.argosmodel")
    with pytest.raises(dyn.DynamicError):
        dyn.install_pack(lock, tmp_path / "out")
    assert _serve_archives == []  # refused before any request is sent


def test_verify_pack_dir_rejects_junk(tmp_path):
    lock, archive = make_pack(tmp_path)
    dyn.install_pack(lock, tmp_path / "good")
    for entry in (tmp_path / "empty", tmp_path / "with-junk"):
        entry.mkdir()
    (tmp_path / "with-junk" / "stray").write_text("not a pack")
    (tmp_path / "a-file").write_text("x")
    (tmp_path / "a-link").symlink_to(tmp_path / "good")
    for bad in (tmp_path / "empty", tmp_path / "with-junk", tmp_path / "a-file", tmp_path / "a-link"):
        assert not dyn.is_pack_installed(bad, lock)


def test_verify_pack_dir_catches_bitrot_and_manifest_tamper(tmp_path):
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "pack"
    dyn.install_pack(lock, target)

    member = target / "sentencepiece.model"
    member.write_bytes(member.read_bytes() + b"!")
    with pytest.raises(dyn.PackVerificationError, match="digest mismatch"):
        dyn.verify_pack_dir(target, lock)

    dyn.install_pack(lock, target)  # self-heals corrupt member
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_pack"] = "other-pack"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(dyn.PackVerificationError, match="pack id"):
        dyn.verify_pack_dir(target, lock)

    dyn.install_pack(lock, target)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = []
    manifest_path.write_text(json.dumps(manifest))
    assert not dyn.is_pack_installed(target, lock)


def test_verify_pack_dir_rejects_foreign_pack_content(tmp_path):
    """A directory built for a different lock is not 'installed'."""
    lock_a, _ = make_pack(tmp_path, name="pack-a")
    lock_b, _ = make_pack(tmp_path / "b", name="pack-b", root="b",
                          files={"sentencepiece.model": b"different"})
    target = tmp_path / "shared"
    dyn.install_pack(lock_a, target)
    with pytest.raises(dyn.PackVerificationError):
        dyn.verify_pack_dir(target, lock_b)


def test_install_pack_refuses_and_preserves_unowned_target(tmp_path, _serve_archives):
    """Route-controlled paths must never delete foreign content."""
    lock, _ = make_pack(tmp_path)
    file_target = tmp_path / "file-target"
    file_target.write_text("precious")
    dir_target = tmp_path / "dir-target"
    dir_target.mkdir()
    (dir_target / "x").write_text("precious")
    empty_target = tmp_path / "empty-target"
    empty_target.mkdir()
    good = tmp_path / "good"
    dyn.install_pack(lock, good)
    link_target = tmp_path / "link-target"
    link_target.symlink_to(good)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    lock_b, _ = make_pack(tmp_path / "b", name="pack-b", root="b")
    foreign_pack = tmp_path / "foreign-pack"
    dyn.install_pack(lock_b, foreign_pack)  # real pack, but for a different lock
    for path in (file_target, dir_target, empty_target, link_target, dangling, foreign_pack):
        with pytest.raises(dyn.DynamicError, match="refus"):
            dyn.install_pack(lock, path)
    # Everything left byte-for-byte untouched; no download was attempted.
    assert file_target.read_text() == "precious"
    assert (dir_target / "x").read_text() == "precious"
    assert list(empty_target.iterdir()) == []
    assert link_target.is_symlink() and link_target.resolve() == good.resolve()
    assert dangling.is_symlink()
    dyn.verify_pack_dir(foreign_pack, lock_b)
    assert _serve_archives == [urlparse(lock["url"]).path, urlparse(lock_b["url"]).path]  # only the two real installs fetched


def test_install_pack_rebuilds_only_owned_corrupt_pack(tmp_path, _serve_archives):
    """A corrupt dir provably ours is rebuilt via verified staging; the old
    contents survive a failed rebuild — never silently destroyed."""
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "pack"
    dyn.install_pack(lock, target)
    member = target / "sentencepiece.model"
    member.write_bytes(b"corrupted")
    dyn.install_pack(lock, target)  # ownership provable: verified rebuild in place
    dyn.verify_pack_dir(target, lock)

    member.write_bytes(b"corrupted")
    _ARCHIVES.pop(urlparse(lock["url"]).path)  # unreachable: rebuild fails
    with pytest.raises(dyn.DynamicError):
        dyn.install_pack(lock, target)
    assert member.read_bytes() == b"corrupted"  # previous content preserved
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(f".{target.name}-")]

def test_install_pack_refuses_owned_dir_with_unmanaged_content(tmp_path, _serve_archives):
    """A matching manifest is NOT permission to delete: any entry the
    manifest does not account for — unlisted file, unlisted directory,
    planted symlink — makes repair refuse and preserve everything."""
    lock, _ = make_pack(tmp_path)

    def seeded(pack_dir):
        dyn.install_pack(lock, pack_dir)
        (pack_dir / "sentencepiece.model").write_bytes(b"corrupted")  # bitrot
        return pack_dir

    unlisted_file = seeded(tmp_path / "unlisted-file") / "user-notes.txt"
    unlisted_file.write_bytes(b"precious")
    unlisted_dir = seeded(tmp_path / "unlisted-dir") / "user-data"
    unlisted_dir.mkdir()
    (unlisted_dir / "keep.txt").write_bytes(b"precious")
    symlinked = seeded(tmp_path / "symlinked") / "planted-link"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"precious")
    symlinked.symlink_to(outside)

    before = len(_serve_archives)
    for sentinel in (unlisted_file, unlisted_dir, symlinked):
        pack_dir = sentinel.parent
        with pytest.raises(dyn.DynamicError, match="refus"):
            dyn.install_pack(lock, pack_dir)
        # Sentinel and corrupt member both preserved; nothing moved aside.
        assert sentinel.exists()
        assert (pack_dir / "sentencepiece.model").read_bytes() == b"corrupted"
        assert not [p for p in pack_dir.parent.iterdir()
                    if p.name.startswith(f".{pack_dir.name}-")]
    assert unlisted_file.read_bytes() == b"precious"
    assert (unlisted_dir / "keep.txt").read_bytes() == b"precious"
    assert symlinked.is_symlink() and symlinked.resolve() == outside.resolve()
    assert outside.read_bytes() == b"precious"
    assert len(_serve_archives) == before  # no download was even attempted


def test_install_pack_refuses_symlinked_ancestors(tmp_path, _serve_archives):
    lock, _ = make_pack(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "via").symlink_to(real)
    with pytest.raises(dyn.DynamicError, match="symlink"):
        dyn.install_pack(lock, tmp_path / "via" / "pack")
    assert list(real.iterdir()) == []  # nothing created through the alias
    assert _serve_archives == []


def test_verify_pack_dir_rejects_symlink_traversal(tmp_path):
    lock, _ = make_pack(tmp_path)
    real = tmp_path / "real-pack"
    dyn.install_pack(lock, real)
    # Target itself reached through a symlink never verifies.
    link = tmp_path / "as-link"
    link.symlink_to(real)
    with pytest.raises(dyn.PackVerificationError, match="symlink"):
        dyn.verify_pack_dir(link, lock)
    # A symlinked ancestor above the target fails closed too.
    (tmp_path / "via").symlink_to(tmp_path)
    with pytest.raises(dyn.PackVerificationError, match="symlink"):
        dyn.verify_pack_dir(tmp_path / "via" / "real-pack", lock)
    # Member traversal: a symlinked member directory or file is rejected.
    (real / "model" / "model.bin").unlink()
    (real / "model").rmdir()
    (real / "model").symlink_to(tmp_path)
    assert not dyn.is_pack_installed(real, lock)
    (real / "model").unlink()
    (real / "model").mkdir()
    (real / "model" / "model.bin").write_bytes(b"weights")
    (real / "sentencepiece.model").unlink()
    (real / "sentencepiece.model").symlink_to(tmp_path / "other.bin")
    assert not dyn.is_pack_installed(real, lock)


def test_download_is_bounded_and_cancellable(tmp_path, monkeypatch):
    monkeypatch.setattr(dyn, "MAX_ARCHIVE_BYTES", 4)
    _ARCHIVES["/big.argosmodel"] = b"0123456789"
    url = _ARCHIVE_SERVER.base_url + "/big.argosmodel"
    with pytest.raises(dyn.DynamicError, match="maximum size"):
        dyn.install_pack({"url": url}, tmp_path / "sub" / "out")
    assert not list((tmp_path / "sub").iterdir())  # temp archive cleaned up

    lock, archive = make_pack(tmp_path)
    stop = threading.Event()
    stop.set()
    with pytest.raises(dyn.DynamicError, match="cancelled"):
        dyn.install_pack(lock, tmp_path / "stopped", stop=stop)
    assert not (tmp_path / "stopped").exists()


def test_download_rejects_insecure_redirect(tmp_path):
    lock, archive = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    _ARCHIVES[path] = ("redirect", "http://evil.invalid/x")
    with pytest.raises(dyn.DynamicError, match="insecure"):
        dyn.install_pack(lock, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_download_rejects_credentialed_redirect(tmp_path):
    lock, archive = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    _ARCHIVES[path] = ("redirect", "https://user:hunter2@x.invalid/y?tok=s3cr3t")
    with pytest.raises(dyn.DynamicError, match="credentialed") as caught:
        dyn.install_pack(lock, tmp_path / "out")
    # The surfaced note must not echo the userinfo or the query string.
    assert "hunter2" not in str(caught.value)
    assert "s3cr3t" not in str(caught.value)
    assert not (tmp_path / "out").exists()

# -- fetch subprocess: cancellation, deadlines, TLS, error hygiene -------------


def _record_popen(monkeypatch):
    """Record every spawned fetch child (for lifecycle assertions)."""
    spawned = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    return spawned


def test_catalog_cancel_trickled_tls_headers():
    """Verified handshake, then a header block that stalls mid-way: the
    fetch child is terminated, not waited out."""
    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
        halted.wait(60)  # Content-Length never arrives; teardown wakes it

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    try:
        stop = threading.Event()
        thread, outcome = _cancel_result(server.url, stop)
        time.sleep(0.7)  # handshake + request + partial headers
        _assert_cancelled(thread, outcome, stop)
    finally:
        server.close()


def test_catalog_cancel_trickled_tls_body():
    """Verified handshake and headers, then a body that drips forever."""
    body = json.dumps({"models": [{"model_pack": "p"}]}).encode()

    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body[:4]
        )
        halted.wait(60)  # the remaining body bytes never arrive

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    try:
        stop = threading.Event()
        thread, outcome = _cancel_result(server.url, stop)
        time.sleep(0.7)  # headers delivered, child parked in read
        _assert_cancelled(thread, outcome, stop)
    finally:
        server.close()


def test_catalog_cancel_during_tls_handshake(monkeypatch):
    """A peer that accepts TCP but never completes TLS parks the child in
    the handshake; cancel must terminate and reap the whole process —
    DNS, connect, TLS, headers and body all die with it."""
    spawned = _record_popen(monkeypatch)
    listener = _SilentTcpListener()
    try:
        stop = threading.Event()
        thread, outcome = _cancel_result(listener.url, stop)
        deadline = time.monotonic() + 10
        while not spawned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert spawned, "fetch child never spawned"
        _assert_cancelled(thread, outcome, stop)
        assert all(proc.poll() is not None for proc in spawned)
    finally:
        listener.close()


def test_catalog_cancel_during_wedged_connect():
    """A full accept queue drops SYNs, wedging connect indefinitely; the
    poll loop still terminates the child in-bounds."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    stuffed = []
    try:
        # Fill the accept queue until a connect can no longer complete.
        for _ in range(64):
            filler = socket.socket()
            filler.settimeout(0.15)
            try:
                filler.connect(("127.0.0.1", port))
            except OSError:
                break
            stuffed.append(filler)
        else:
            pytest.skip("accept queue would not fill")
        stuffed.append(filler)  # SYN still in flight, keeps the queue full
        stop = threading.Event()
        thread, outcome = _cancel_result(
            f"https://127.0.0.1:{port}/models.json", stop
        )
        time.sleep(0.5)  # child is inside the wedged connect
        _assert_cancelled(thread, outcome, stop)
    finally:
        listener.close()
        for sock in stuffed:
            sock.close()


def test_catalog_deadline_kills_stalled_child(monkeypatch):
    """A stalled fetch dies at the total deadline even without stop."""
    monkeypatch.setattr(dyn, "_CATALOG_DEADLINE_SECONDS", 0.5)
    listener = _SilentTcpListener()
    try:
        start = time.monotonic()
        with pytest.raises(dyn.DynamicError, match="timed out"):
            dyn.load_catalog(listener.url)
        assert time.monotonic() - start < 10
        assert not _fetch_children()
    finally:
        listener.close()


def test_catalog_over_verified_tls_and_real_dns():
    """End-to-end success: child resolves, connects, verifies TLS, streams."""
    payload = {"models": [{"model_pack": "p", "sha256": "0" * 64}]}
    body = json.dumps(payload).encode()

    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    try:
        # 'localhost' (not an IP literal) exercises real resolution.
        result = dyn.load_catalog(server.url)
        assert result["p"]["model_pack"] == "p"
    finally:
        server.close()
    assert not _fetch_children()


def test_catalog_tls_verification_enforced():
    """A self-signed cert not issued by the trusted CA is refused."""
    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    server = _LoopbackTlsServer(_ROGUE_CERT, _ROGUE_KEY_FILE, serve)
    try:
        with pytest.raises(dyn.DynamicError, match="URLError"):
            dyn.load_catalog(server.url)
    finally:
        server.close()
    assert not _fetch_children()


def test_catalog_tls_hostname_mismatch_refused(monkeypatch):
    """A chained cert naming a different SAN is refused: the chain verifies
    (the cert anchors itself in this bundle) but hostname does not."""
    monkeypatch.setenv("SSL_CERT_FILE", str(_SANFAIL_BUNDLE))

    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    server = _LoopbackTlsServer(_SANFAIL_CERT, _SANFAIL_KEY_FILE, serve)
    try:
        url = f"https://127.0.0.1:{server.port}/models.json"
        with pytest.raises(dyn.DynamicError, match="URLError"):
            dyn.load_catalog(url)
    finally:
        server.close()
    assert not _fetch_children()


def test_fetch_child_argv_never_carries_url(monkeypatch):
    """URL/query secrets ride stdin, never argv or the process table."""
    argv_seen = []
    real_popen = subprocess.Popen

    def recording_popen(args, **kwargs):
        argv_seen.append(list(args))
        return real_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    payload = {"models": [{"model_pack": "p"}]}
    body = json.dumps(payload).encode()

    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
            + b"\r\n\r\n" + body
        )

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    try:
        source = server.url + "?sig=s3cr3t"
        result = dyn.load_catalog(source)
        assert result["p"]["model_pack"] == "p"
    finally:
        server.close()
    assert argv_seen, "fetch child never spawned"
    flat = " ".join(arg for argv in argv_seen for arg in argv)
    assert "s3cr3t" not in flat
    assert "sig=" not in flat
    assert "localhost" not in flat  # nothing URL-derived reaches argv


def test_catalog_error_never_echoes_remote_text():
    """An HTTP error leaks only its classified status code, never text."""
    def serve(sock, halted):
        _drain_request(sock)
        sock.sendall(
            b"HTTP/1.1 503 s3cr3t h0stile reason\r\nContent-Length: 0\r\n\r\n"
        )

    server = _LoopbackTlsServer(_SERVER_CERT, _SERVER_KEY_FILE, serve)
    try:
        with pytest.raises(dyn.DynamicError) as caught:
            dyn.load_catalog(server.url + "?sig=s3cr3t")
    finally:
        server.close()
    message = str(caught.value)
    assert "503" in message
    assert "s3cr3t" not in message
    assert "h0stile" not in message
    assert "sig=" not in message


def test_archive_error_never_echoes_remote_text(tmp_path):
    lock, _ = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    _ARCHIVES[path] = ("status", 500, "h0stile reason s3cr3t")
    with pytest.raises(dyn.DynamicError) as caught:
        dyn.install_pack(lock, tmp_path / "out")
    message = str(caught.value)
    assert "500" in message
    assert "h0stile" not in message
    assert "s3cr3t" not in message


def test_archive_trickle_cancel_and_deadline_cleanup(tmp_path):
    """Stop mid-body ends the fetch, kills the child, drops the temp file;
    a stalled body dies at the archive deadline without stop too."""
    lock, archive = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    served = _ARCHIVES[path]
    _ARCHIVES[path] = ("stall", served)  # headers then silence

    stop = threading.Event()
    outcome = []

    def install():
        try:
            dyn.install_pack(
                lock, tmp_path / "packs" / lock["model_pack"], stop=stop
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=install, daemon=True)
    thread.start()
    time.sleep(1.0)  # child is past headers, parked in the stalled body
    stop.set()
    thread.join(5)
    assert not thread.is_alive(), "install_pack did not return after stop"
    (result,) = outcome
    assert isinstance(result, dyn.DynamicError)
    assert "cancelled" in str(result)
    packs_dir = tmp_path / "packs"
    leftovers = list(packs_dir.iterdir()) if packs_dir.exists() else []
    assert not leftovers  # temp archive and staging are gone
    assert not _fetch_children()


def test_archive_deadline_kills_stalled_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dyn, "_ARCHIVE_DEADLINE_SECONDS", 0.5)
    lock, archive = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    _ARCHIVES[path] = ("stall", archive.read_bytes())
    start = time.monotonic()
    with pytest.raises(dyn.DynamicError, match="timed out"):
        dyn.install_pack(lock, tmp_path / "packs" / lock["model_pack"])
    assert time.monotonic() - start < 10
    packs_dir = tmp_path / "packs"
    leftovers = list(packs_dir.iterdir()) if packs_dir.exists() else []
    assert not leftovers
    assert not _fetch_children()


def test_archive_size_cap_still_enforced(tmp_path, monkeypatch):
    """The byte bound fires inside the child and surfaces via the parent."""
    monkeypatch.setattr(dyn, "MAX_ARCHIVE_BYTES", 4)
    _ARCHIVES["/big.argosmodel"] = b"0123456789"
    url = _ARCHIVE_SERVER.base_url + "/big.argosmodel"
    with pytest.raises(dyn.DynamicError, match="maximum size"):
        dyn.install_pack({"url": url}, tmp_path / "sub" / "out")
    assert not list((tmp_path / "sub").iterdir())


def test_manager_drain_mid_archive_fetch_registers_nothing(tmp_path):
    """runtime.stop mid-download: no registration, persist, or leftovers."""
    lock, archive = make_pack(tmp_path)
    path = urlparse(lock["url"]).path
    _ARCHIVES[path] = ("stall", archive.read_bytes())
    runtime = FakeRuntime(
        [_pack_row(lock, tmp_path)], tmp_path, catalog_entries=[lock]
    )
    outcome = []

    def cycle():
        outcome.append(dyn.manage_cycle(runtime))

    thread = threading.Thread(target=cycle, daemon=True)
    thread.start()
    time.sleep(1.0)  # cycle is inside the stalled archive body
    runtime.stop.set()
    thread.join(5)
    assert not thread.is_alive(), "manage_cycle survived drain"
    assert outcome == [False]
    assert runtime.pack_updates == [] and runtime.cap_updates == []
    assert runtime.persisted_caps == []
    packs_dir = tmp_path / "packs"
    leftovers = list(packs_dir.iterdir()) if packs_dir.exists() else []
    assert not leftovers
    assert not _fetch_children()

# -- planning ---------------------------------------------------------------


def test_plan_provisions_each_distinct_target(tmp_path):
    lock, _ = make_pack(tmp_path)
    rows = [
        _row(packs=[lock["model_pack"]], model_dir="/m/one", sha=lock["sha256"], route="r1"),
        _row(packs=[lock["model_pack"]], model_dir="/m/two", sha=lock["sha256"], route="r2"),
    ]
    result = _plan(rows, catalog={lock["model_pack"]: lock})
    targets = {fetch["target"] for fetch in result.fetches}
    assert targets == {Path("/m/one"), Path("/m/two")}
    assert not result.packs[lock["model_pack"]].ready


def test_plan_dedupes_shared_target(tmp_path):
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "shared"
    dyn.install_pack(lock, target)
    rows = [
        _row(packs=[lock["model_pack"]], model_dir=str(target), sha=lock["sha256"], route="r1"),
        _row(packs=[lock["model_pack"]], model_dir=str(target), sha=lock["sha256"], route="r2"),
    ]
    result = _plan(rows, catalog={lock["model_pack"]: lock})
    assert result.fetches == []
    assert result.packs[lock["model_pack"]].ready


def test_plan_rejects_conflicting_route_pins(tmp_path):
    lock, _ = make_pack(tmp_path)
    rows = [
        _row(packs=[lock["model_pack"]], model_dir="/m/one", sha=lock["sha256"], route="r1"),
        _row(packs=[lock["model_pack"]], model_dir="/m/two", sha="f" * 64, route="r2"),
    ]
    with pytest.raises(dyn.DynamicError, match="conflicting"):
        _plan(rows, catalog={lock["model_pack"]: lock})
    # Same conflict even when the pack is already "installed" per config.
    with pytest.raises(dyn.DynamicError):
        _plan(rows, catalog={lock["model_pack"]: lock},
              installed_packs={lock["model_pack"]})


def test_plan_rejects_same_dir_for_two_packs(tmp_path):
    lock_a, _ = make_pack(tmp_path, name="pack-a")
    lock_b, _ = make_pack(tmp_path / "b", name="pack-b", root="b")
    rows = [
        _row(packs=["pack-a"], model_dir="/m/shared", sha=lock_a["sha256"], route="r1"),
        _row(packs=["pack-b"], model_dir="/m/shared", sha=lock_b["sha256"], route="r2"),
    ]
    with pytest.raises(dyn.DynamicError, match="same directory"):
        _plan(rows, catalog={"pack-a": lock_a, "pack-b": lock_b})


def test_plan_unknown_catalog_pack_not_advertised(tmp_path):
    rows = [_row(packs=["ghost-pack"], route="r1")]
    result = _plan(rows, catalog={})
    assert result.capabilities == ["translate-batch"]  # capability still follows demand
    entry = result.packs["ghost-pack"]
    assert not entry.ready and result.fetches == []
    assert any("not in the catalog" in note for note in result.notes)


def test_plan_language_pair_pins_checked(tmp_path):
    lock, _ = make_pack(tmp_path)
    row = _row(packs=[lock["model_pack"]], model_dir="/m/x", sha=lock["sha256"])
    row["executor_environment"]["ARGOS_TARGET_LANGUAGE"] = "fr"
    with pytest.raises(dyn.DynamicError, match="language"):
        _plan([row], catalog={lock["model_pack"]: lock})


def _pin_route(row, lock):
    """Pin the full identity — pack id, archive digest, language pair."""
    row["executor_environment"].update({
        "ARGOS_MODEL_PACK": lock["model_pack"],
        "ARGOS_MODEL_ARCHIVE_SHA256": lock["sha256"],
        "ARGOS_SOURCE_LANGUAGE": lock["from_code"],
        "ARGOS_TARGET_LANGUAGE": lock["to_code"],
    })
    return row


def test_plan_pinned_pack_verifies_without_catalog(tmp_path):
    """Preinstalled dynamic mode needs no catalog: a demanded target that
    verifies fully against the route's pinned identity is advertised."""
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "preinstalled"
    dyn.install_pack(lock, target)
    row = _pin_route(_row(packs=[lock["model_pack"]], model_dir=str(target)), lock)
    result = _plan([row], catalog={})
    entry = result.packs[lock["model_pack"]]
    assert entry.ready and result.fetches == []
    assert result.capabilities == ["translate-batch"]


def test_plan_pinned_pack_missing_or_bitrotted_never_fetches(tmp_path):
    """Pins prove identity but carry no URL: an unverifiable pin-pinned
    target is withheld with a visible reason — never a synthesized fetch.
    Multiple targets are checked against the same merged identity."""
    lock, _ = make_pack(tmp_path)
    bitrotted = tmp_path / "bitrotted"
    dyn.install_pack(lock, bitrotted)
    (bitrotted / "model" / "model.bin").write_bytes(b"corrupted")
    missing = tmp_path / "missing"
    rows = [
        _pin_route(_row(packs=[lock["model_pack"]], model_dir=str(bitrotted), route="r1"), lock),
        _pin_route(_row(packs=[lock["model_pack"]], model_dir=str(missing), route="r2"), lock),
    ]
    result = _plan(rows, catalog={}, installed_packs={lock["model_pack"]})
    assert result.fetches == []
    assert not result.packs[lock["model_pack"]].ready
    assert any("not in the catalog" in note for note in result.notes)
    # A pack whose target holds another pack entirely is refused, not just
    # unverifiable — the same no-fetch guarantee.
    other_lock, _ = make_pack(tmp_path / "other", name="other-pack", root="other")
    foreign = tmp_path / "foreign"
    dyn.install_pack(other_lock, foreign)
    row = _pin_route(_row(packs=[lock["model_pack"]], model_dir=str(foreign)), lock)
    result = _plan([row], catalog={})
    assert result.fetches == [] and not result.packs[lock["model_pack"]].ready
    assert any("refus" in note for note in result.notes)
    dyn.verify_pack_dir(foreign, other_lock)  # foreign pack untouched


def test_plan_pinned_pack_incomplete_pins_not_advertised(tmp_path):
    """A digest pin alone is not enough identity to trust a directory."""
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "preinstalled"
    dyn.install_pack(lock, target)
    row = _row(packs=[lock["model_pack"]], model_dir=str(target), sha=lock["sha256"])
    result = _plan([row], catalog={})
    assert not result.packs[lock["model_pack"]].ready and result.fetches == []
    assert any("not in the catalog" in note for note in result.notes)
    # ...but give the same directory to a catalog and it is a cache hit.
    result = _plan([row], catalog={lock["model_pack"]: lock})
    assert result.packs[lock["model_pack"]].ready


# -- executor prerequisites ---------------------------------------------------


def _plan_row(executor_type, detail, environment=None):
    row = _row()
    row["executor_type"] = executor_type
    row["executor_detail"] = detail
    row["executor_environment"] = environment or {}
    return row


def test_container_route_needs_docker(monkeypatch):
    monkeypatch.setitem(dyn._docker_probe_cache, "at", float("-inf"))
    monkeypatch.setattr(dyn.shutil, "which", lambda name: None)
    result = _plan([_plan_row("container", "img@sha256:ab")])
    assert result.capabilities == []
    assert any("docker" in note for note in result.notes)


def _docker_stub(tmp_path, monkeypatch, body_lines):
    """Install a stub `docker` executable on PATH; returns a counter file
    the stub appends to on each invocation."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    counter = tmp_path / "probes"
    script = bindir / "docker"
    script.write_text(
        "#!/bin/sh\n" + "\n".join(body_lines).replace("COUNTFILE", str(counter)) + "\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setitem(dyn._docker_probe_cache, "at", float("-inf"))
    return counter


def test_container_route_needs_reachable_daemon(tmp_path, monkeypatch):
    """Present binary + inaccessible daemon: capability withheld, and the
    bounded probe is cached so repeated routes don't re-spawn it."""
    counter = _docker_stub(
        tmp_path,
        monkeypatch,
        [
            "echo probe >> COUNTFILE",
            'echo "permission denied while trying to connect to the Docker '
            'daemon socket" >&2',
            "exit 1",
        ],
    )
    row = _plan_row("container", "img@sha256:ab")
    result = _plan([row])
    assert result.capabilities == []
    assert any("docker" in note for note in result.notes)
    assert any("access denied" in note for note in result.notes)

    # Same verdict for a second route/plan pass without another probe.
    again = _plan([_plan_row("container", "img@sha256:ab")])
    assert again.capabilities == []
    assert counter.read_text().count("probe") == 1


def test_container_route_eligible_when_daemon_answers(tmp_path, monkeypatch):
    counter = _docker_stub(
        tmp_path, monkeypatch, ["echo probe >> COUNTFILE", 'echo "27.0.0"', "exit 0"]
    )
    result = _plan([_plan_row("container", "img@sha256:ab")])
    assert result.capabilities == ["translate-batch"]
    assert counter.read_text().count("probe") == 1


def test_container_route_reprobes_after_ttl(tmp_path, monkeypatch):
    counter = _docker_stub(
        tmp_path, monkeypatch, ["echo probe >> COUNTFILE", "exit 1"]
    )
    monkeypatch.setattr(dyn, "_DOCKER_PROBE_TTL_SECONDS", 0)
    _plan([_plan_row("container", "img@sha256:ab")])
    _plan([_plan_row("container", "img@sha256:ab")])
    assert counter.read_text().count("probe") == 2


def test_plugin_route_needs_importable_handler(tmp_path, monkeypatch):
    module = tmp_path / "fake_plugin_mod.py"
    module.write_text("class Executor:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    result = _plan([_plan_row("plugin", "fake_plugin_mod:Executor")])
    assert result.capabilities == ["translate-batch"]

    result = _plan([_plan_row("plugin", "fake_plugin_mod:MissingAttr")])
    assert result.capabilities == []
    assert any("missing or not callable" in note for note in result.notes)

    result = _plan([_plan_row("plugin", "fake_plugin_mod_no_colon")])
    assert result.capabilities == []
    assert any("module:attribute" in note for note in result.notes)


def test_argos_plugin_reports_missing_optional_dependencies():
    pytest.importorskip("sparehour_workloads.argos_translate")
    detail = "sparehour_workloads.argos_translate:ArgosTranslateExecutor"
    result = _plan([_plan_row("plugin", detail)])
    import importlib.util

    if importlib.util.find_spec("ctranslate2") is None:
        assert result.capabilities == []
        assert any("ctranslate2" in note for note in result.notes)
    else:
        assert result.capabilities == ["translate-batch"]


def _serve(handler):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_http_route_requires_reachable_endpoint():
    server, _ = _serve(_OkHandler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        result = _plan([_plan_row("http", url)])
        assert result.capabilities == ["translate-batch"]
    finally:
        server.shutdown()

    # A closed loopback port is deterministically unreachable.
    dead = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    port = dead.server_port
    dead.server_close()
    result = _plan([_plan_row("http", f"http://127.0.0.1:{port}/v1")])
    assert result.capabilities == []
    assert any("unreachable" in note for note in result.notes)


def test_http_route_rejects_non_loopback_plain_http():
    # Rejected by policy before any network probe (deterministic offline).
    for endpoint in ("http://0.0.0.0:9/x", "http://192.0.2.1:9/x", "gopher://x/"):
        result = _plan([_plan_row("http", endpoint)])
        assert result.capabilities == []
        assert result.notes

def test_endpoint_probe_cancel_during_wedged_headers():
    """A silent endpoint parks the probe mid-headers; stop unblocks it via
    the same tracked-socket path the remote fetch uses."""
    listener = _SilentTcpListener()
    try:
        stop = threading.Event()
        outcome = []

        def probe():
            outcome.append(
                dyn._http_endpoint_runnable(
                    f"http://127.0.0.1:{listener.port}/v1", stop
                )
            )

        thread = threading.Thread(target=probe, daemon=True)
        thread.start()
        time.sleep(0.5)  # probe is parked in the header read
        stop.set()
        thread.join(6)
        assert not thread.is_alive(), "endpoint probe survived stop"
        (verdict,) = outcome
        ok, reason = verdict
        assert ok is False
        assert "cancelled" in reason
    finally:
        listener.close()
    assert not _fetch_children()


def test_manager_drain_during_endpoint_probe_registers_nothing(tmp_path):
    """runtime.stop mid-endpoint-probe: cycle drains, nothing registers."""
    listener = _SilentTcpListener()
    try:
        row = _row(
            pending=1,
            packs=[],
            route="r1",
            model_dir=None,
        )
        row["executor_type"] = "http"
        row["executor_detail"] = f"http://127.0.0.1:{listener.port}/v1"
        runtime = FakeRuntime([row], tmp_path)
        outcome = []

        def cycle():
            outcome.append(dyn.manage_cycle(runtime))

        thread = threading.Thread(target=cycle, daemon=True)
        thread.start()
        time.sleep(0.5)  # cycle is inside the wedged endpoint probe
        runtime.stop.set()
        thread.join(6)
        assert not thread.is_alive(), "manage_cycle survived drain"
        assert outcome == [False]
        assert runtime.cap_updates == []
        assert runtime.persisted_caps == []
    finally:
        listener.close()
    assert not _fetch_children()


# -- catalog transport ----------------------------------------------------------


def test_remote_catalog_is_bounded_and_https_only(monkeypatch):
    monkeypatch.setattr(dyn, "_CATALOG_MAX_BYTES", 4)
    _ARCHIVES["/models.json"] = b"x" * 64
    source = _ARCHIVE_SERVER.base_url + "/models.json"
    with pytest.raises(dyn.DynamicError, match="exceeds"):
        dyn.load_catalog(source)

    body = json.dumps({"models": [{"model_pack": "p", "sha256": "x"}]}).encode()
    monkeypatch.setattr(dyn, "_CATALOG_MAX_BYTES", 8 * 1024 * 1024)
    _ARCHIVES["/models.json"] = body
    assert dyn.load_catalog(source)["p"]["model_pack"] == "p"


# -- manager state transitions --------------------------------------------------


class _Notes:
    def __init__(self):
        self.messages = []

    def note(self, message):
        self.messages.append(message)


class FakeRuntime:
    """WorkerRuntime contract surface used by the dynamic manager."""
    def __init__(self, report, workdir, *, capabilities=None, model_packs=None,
                 catalog_entries=None, stop=None):
        # Catalog is a real local file so cycles exercise load_catalog for real.
        catalog_file = ""
        if catalog_entries:
            catalog_path = workdir / "catalog.json"
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.write_text(json.dumps({"models": catalog_entries}))
            catalog_file = str(catalog_path)
        self.config = {
            "model_catalog": catalog_file,
            "dynamic_poll_seconds": 60,
            "provider": "local",
            "labels": {},
            "capabilities": list(capabilities or []),
            "model_packs": list(model_packs or []),
            "workdir": str(workdir),
        }
        self.config_file = workdir / "config.json"
        self.host = dict(HOST)
        self.slots = [_inv()]
        self.lock = threading.RLock()
        self.stop = stop if stop is not None else threading.Event()
        self.status = _Notes()
        self._report = report
        self.pack_updates = []
        self.cap_updates = []
        self.persisted_caps = []

    def queues_report(self, max_age=30.0):
        return self._report

    def update_model_packs(self, packs):
        self.pack_updates.append(list(packs))
        self.config["model_packs"] = list(packs)
        return True, f"model packs now: {packs or 'none'} (registered)"

    def apply_capability_list(self, capabilities, *, persist):
        self.cap_updates.append(list(capabilities))
        if not persist:
            self.persisted_caps.append(False)
        else:
            self.persisted_caps.append(True)
        self.config["capabilities"] = list(capabilities)
        return True, f"capabilities now: {capabilities or 'none'} (registered)"


def _pack_row(lock, tmp_path, capability="translate-batch", pending=1):
    return _row(
        capability=capability,
        pending=pending,
        packs=[lock["model_pack"]],
        model_dir=str(tmp_path / "packs" / lock["model_pack"]),
        sha=lock["sha256"],
    )


def test_manager_installs_pack_and_advertises(tmp_path):
    lock, _ = make_pack(tmp_path)
    runtime = FakeRuntime([_pack_row(lock, tmp_path)], tmp_path, catalog_entries=[lock])
    assert dyn.manage_cycle(runtime) is True
    target = tmp_path / "packs" / lock["model_pack"]
    dyn.verify_pack_dir(target, lock)
    assert runtime.pack_updates == [[lock["model_pack"]]]
    assert runtime.cap_updates == [["translate-batch"]]
    assert runtime.config["model_packs"] == [lock["model_pack"]]
    assert runtime.persisted_caps == [False]  # dynamic selection stays runtime-only

    # Second cycle: pack verifies; update_model_packs still re-registers even
    # though nothing changed — the controller is never left with a stale list.
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates == [[lock["model_pack"]], [lock["model_pack"]]]
    assert runtime.cap_updates[-1] == ["translate-batch"]


def test_manager_drops_stale_then_readvertises_pack(tmp_path):
    """Advertised pack whose directory vanished: un-advertised, rebuilt, re-added."""
    lock, _ = make_pack(tmp_path)
    runtime = FakeRuntime(
        [_pack_row(lock, tmp_path)], tmp_path, model_packs=[lock["model_pack"]],
        catalog_entries=[lock],
    )
    assert dyn.manage_cycle(runtime) is True
    dyn.verify_pack_dir(tmp_path / "packs" / lock["model_pack"], lock)
    assert runtime.pack_updates == [[], [lock["model_pack"]]]

    # Now corrupt the pack and leave a false-positive installed config: the
    # next cycle un-advertises, rebuilds byte-verified, and re-advertises.
    member = tmp_path / "packs" / lock["model_pack"] / "sentencepiece.model"
    member.write_bytes(b"corrupted")
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates[-2:] == [[], [lock["model_pack"]]]
    dyn.verify_pack_dir(tmp_path / "packs" / lock["model_pack"], lock)


def test_manager_never_registers_after_drain(tmp_path):
    lock, _ = make_pack(tmp_path)
    stop = threading.Event()
    stop.set()
    runtime = FakeRuntime([_pack_row(lock, tmp_path)], tmp_path, stop=stop)
    assert dyn.manage_cycle(runtime) is False
    assert runtime.pack_updates == [] and runtime.cap_updates == []
    assert not (tmp_path / "packs").exists()


def test_manager_stops_without_queue_depths(tmp_path):
    runtime = FakeRuntime([_row(pending=None)], tmp_path)
    assert dyn.manage_cycle(runtime) is False
    assert any("perms=dynamic" in note for note in runtime.status.messages)


def test_manager_survives_cycle_errors(tmp_path):
    runtime = FakeRuntime([], tmp_path)
    runtime.queues_report = lambda max_age=0.0: (_ for _ in ()).throw(RuntimeError("boom"))
    assert dyn.manage_cycle(runtime) is True  # one bad cycle, manager continues
    assert any("boom" in note for note in runtime.status.messages)


def test_manager_pin_conflict_aborts_cycle_before_fetch(tmp_path):
    lock, _ = make_pack(tmp_path)
    rows = [
        _row(packs=[lock["model_pack"]], model_dir=str(tmp_path / "a"),
             sha=lock["sha256"], route="r1"),
        _row(packs=[lock["model_pack"]], model_dir=str(tmp_path / "b"),
             sha="f" * 64, route="r2"),
    ]
    runtime = FakeRuntime(rows, tmp_path)
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates == []
    assert any("conflicting" in note for note in runtime.status.messages)


def test_manage_cycle_defaults_to_runtime_stop(tmp_path):
    runtime = FakeRuntime([_row()], tmp_path)
    runtime.stop.set()
    assert dyn.manage_cycle(runtime) is False


def test_manager_refuses_and_preserves_foreign_target(tmp_path):
    """A demanded pack path occupied by foreign data is never deleted,
    never advertised, and surfaces an actionable note."""
    lock, _ = make_pack(tmp_path)
    victim = tmp_path / "packs" / lock["model_pack"]
    victim.mkdir(parents=True)
    (victim / "user-data.txt").write_text("precious")
    runtime = FakeRuntime(
        [_pack_row(lock, tmp_path)], tmp_path,
        model_packs=[lock["model_pack"]], catalog_entries=[lock],
    )
    assert dyn.manage_cycle(runtime) is True
    assert (victim / "user-data.txt").read_text() == "precious"
    # Un-advertised (unsafe target is not eligible) and never re-added.
    assert lock["model_pack"] not in runtime.config["model_packs"]
    assert not any(lock["model_pack"] in update for update in runtime.pack_updates)
    assert any("refus" in note for note in runtime.status.messages)

def test_manager_runs_route_pinned_pack_without_catalog(tmp_path, _serve_archives):
    """Baseline preinstalled mode: model_catalog empty, the demanded target
    verifies against the route's pinned identity — advertised, no download."""
    lock, _ = make_pack(tmp_path)
    target = tmp_path / "packs" / lock["model_pack"]
    dyn.install_pack(lock, target)
    row = _pin_route(_pack_row(lock, tmp_path), lock)
    runtime = FakeRuntime([row], tmp_path, model_packs=[lock["model_pack"]])
    assert dyn.manage_cycle(runtime) is True
    assert _serve_archives == [urlparse(lock["url"]).path]  # only the seeding install fetched
    assert runtime.config["model_packs"] == [lock["model_pack"]]
    assert runtime.cap_updates[-1] == ["translate-batch"]

    # Bitrot with no catalog: un-advertised with a reason, never fetched —
    # identity pins carry no URL, so nothing can be downloaded.
    (target / "sentencepiece.model").write_bytes(b"corrupted")
    assert dyn.manage_cycle(runtime) is True
    assert lock["model_pack"] not in runtime.config["model_packs"]
    assert any("not in the catalog" in note for note in runtime.status.messages)
    assert (target / "sentencepiece.model").read_bytes() == b"corrupted"
    assert _serve_archives == [urlparse(lock["url"]).path]  # still only the seeding install


def test_manager_preserves_sentinel_in_owned_corrupt_pack(tmp_path):
    """Owned-by-manifest pack dir + user file + bitrot: the manager refuses
    repair and preserves the sentinel — nothing is rmtree'd or re-added."""
    lock, _ = make_pack(tmp_path)
    victim = tmp_path / "packs" / lock["model_pack"]
    dyn.install_pack(lock, victim)
    (victim / "sentencepiece.model").write_bytes(b"corrupted")
    (victim / "user-notes.txt").write_text("precious")
    runtime = FakeRuntime(
        [_pack_row(lock, tmp_path)], tmp_path,
        model_packs=[lock["model_pack"]], catalog_entries=[lock],
    )
    assert dyn.manage_cycle(runtime) is True
    assert (victim / "user-notes.txt").read_text() == "precious"
    assert (victim / "sentencepiece.model").read_bytes() == b"corrupted"
    assert lock["model_pack"] not in runtime.config["model_packs"]
    assert not any(lock["model_pack"] in update for update in runtime.pack_updates)
    assert any("refus" in note for note in runtime.status.messages)
    assert not [p for p in victim.parent.iterdir() if p.name.startswith(f".{victim.name}-")]

# -- huggingface-format and service packs ------------------------------------------

_HF_REV = "c" * 40


def _hf_lock(pack="hf-pack", model_id="org/model", revision=_HF_REV, **over):
    lock = {
        "format": "huggingface",
        "model_pack": pack,
        "model_id": model_id,
        "revision": revision,
        "allow_patterns": ["*.json", "*.bin"],
    }
    lock.update(over)
    return lock


def _hf_row(pack="hf-pack", model_id="org/model", revision=_HF_REV,
            model_dir=None, prefix="TEST", **kw):
    """A queue row whose env binds an HF pack identity and (optionally) a dir."""
    row = _row(packs=[pack], **kw)
    row["executor_environment"].update(
        {
            f"{prefix}_MODEL_PACK": pack,
            f"{prefix}_MODEL_ID": model_id,
            f"{prefix}_MODEL_REVISION": revision,
        }
    )
    if model_dir:
        row["executor_environment"][f"{prefix}_MODEL_DIR"] = str(model_dir)
    return row


def _hf_pack_dir(target: Path, pack="hf-pack", model_id="org/model",
                 revision=_HF_REV, files=None):
    """Build a real HF pack dir + manifest for preinstalled-pack tests."""
    mp = pytest.importorskip("sparehour_workloads.model_packs")
    target.mkdir(parents=True, exist_ok=True)
    files = files or {"config.json": b"{}"}
    rows = []
    for name, data in files.items():
        member = target.joinpath(*name.split("/"))
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(data)
        rows.append({
            "path": name, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    rows.sort(key=lambda item: item["path"])
    (target / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "format": "huggingface", "model_pack": pack,
        "model_id": model_id, "revision": revision, "files": rows,
    }))
    return target


def test_plan_hf_pack_provisions_env_dir(tmp_path):
    lock = _hf_lock()
    row = _hf_row(model_dir="/m/hf")
    result = _plan([row], catalog={"hf-pack": lock})
    targets = {fetch["target"] for fetch in result.fetches}
    assert targets == {Path("/m/hf")}
    assert result.fetches[0]["lock"]["format"] == "huggingface"
    assert not result.packs["hf-pack"].ready


def test_plan_hf_pack_verifies_preinstalled_dir(tmp_path):
    pytest.importorskip("sparehour_workloads.model_packs")
    lock = _hf_lock()
    target = _hf_pack_dir(tmp_path / "installed")
    row = _hf_row(model_dir=str(target))
    result = _plan([row], catalog={"hf-pack": lock})
    assert result.fetches == []
    assert result.packs["hf-pack"].ready
    assert result.capabilities == ["translate-batch"]


def test_plan_hf_pack_verifies_without_catalog(tmp_path):
    """Full env pins synthesize an identity; a verified dir is advertised
    with no catalog and no download."""
    pytest.importorskip("sparehour_workloads.model_packs")
    target = _hf_pack_dir(tmp_path / "preinstalled")
    row = _hf_row(model_dir=str(target))
    result = _plan([row], catalog={})
    assert result.fetches == []
    assert result.packs["hf-pack"].ready


def test_plan_hf_pack_mutable_revision_refused(tmp_path):
    row = _hf_row(revision="main", model_dir=str(tmp_path / "p"))
    with pytest.raises(dyn.DynamicError, match="immutable"):
        _plan([row], catalog={})


def test_plan_hf_pack_conflicting_revisions_abort(tmp_path):
    rows = [
        _hf_row(model_dir="/m/a", route="r1"),
        _hf_row(model_dir="/m/b", revision="d" * 40, route="r2"),
    ]
    with pytest.raises(dyn.DynamicError, match="conflicting"):
        _plan(rows, catalog={})


def test_plan_hf_pack_catalog_pin_mismatch(tmp_path):
    lock = _hf_lock(revision="e" * 40)
    row = _hf_row(model_dir="/m/hf")  # pins _HF_REV, catalog disagrees
    with pytest.raises(dyn.DynamicError, match="catalog"):
        _plan([row], catalog={"hf-pack": lock})


def test_plan_unknown_catalog_format_noted_not_argos(tmp_path):
    """A foreign format entry is withheld with a note — never fetched as an
    Argos archive."""
    lock = {"format": "oci", "model_pack": "weird", "image": "img"}
    row = _row(packs=["weird"], model_dir=str(tmp_path / "p"))
    result = _plan([row], catalog={"weird": lock})
    assert result.fetches == []
    assert result.packs["weird"].kind == "unsupported"
    assert not result.packs["weird"].ready
    assert any("unsupported" in note for note in result.notes)


def test_plan_hf_pack_multiple_distinct_targets(tmp_path):
    """WhisperX-style: three packs, three env triples, three targets."""
    packs = {}
    rows = []
    env = {}
    for prefix, pack in (
        ("WHISPERX", "whisperx-large-v3"),
        ("WHISPERX_ALIGN", "whisperx-align-en"),
        ("DIARIZATION", "speaker-diarization"),
    ):
        env[f"{prefix}_MODEL_PACK"] = pack
        env[f"{prefix}_MODEL_ID"] = f"org/{pack}"
        env[f"{prefix}_MODEL_REVISION"] = _HF_REV
        env[f"{prefix}_MODEL_DIR"] = f"/m/{pack}"
        packs[pack] = _hf_lock(pack=pack, model_id=f"org/{pack}")
    row = _row(packs=list(packs), route="whisperx")
    row["executor_environment"] = env
    result = _plan([row], catalog=packs)
    assert {f["target"] for f in result.fetches} == {
        Path(f"/m/{p}") for p in packs
    }
    assert all(entry.kind == "huggingface" for entry in result.packs.values())


def test_plan_hf_same_dir_two_packs_rejected():
    rows = []
    env = {
        "A_MODEL_PACK": "pack-a", "A_MODEL_ID": "org/a",
        "A_MODEL_REVISION": _HF_REV, "A_MODEL_DIR": "/m/shared",
        "B_MODEL_PACK": "pack-b", "B_MODEL_ID": "org/b",
        "B_MODEL_REVISION": _HF_REV, "B_MODEL_DIR": "/m/shared",
    }
    row = _row(packs=["pack-a", "pack-b"], route="r1")
    row["executor_environment"] = env
    catalog = {
        "pack-a": _hf_lock("pack-a", "org/a"),
        "pack-b": _hf_lock("pack-b", "org/b"),
    }
    with pytest.raises(dyn.DynamicError, match="same directory"):
        _plan([row], catalog=catalog)


def test_plan_hf_pack_without_shared_module_noted(tmp_path, monkeypatch):
    """A wheel too old for model_packs notes instead of crashing the plan."""
    lock = _hf_lock()
    monkeypatch.setattr(
        dyn, "_model_packs",
        lambda: (_ for _ in ()).throw(dyn.DynamicError("no module")),
    )
    result = _plan([_hf_row(model_dir=str(tmp_path / "p"))],
                   catalog={"hf-pack": lock})
    assert result.fetches == []
    assert not result.packs["hf-pack"].ready
    assert result.notes


# -- service packs ----------------------------------------------------------


class _JsonHandler(http.server.BaseHTTPRequestHandler):
    """Serves a canned JSON body from a class attribute."""
    payload = b"[]"
    expected_path = None

    def do_GET(self):
        if self.expected_path is not None and self.path != self.expected_path:
            self.send_error(404)
            return
        body = self.payload
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def _json_server(payload, *, path=None):
    handler = type("Handler", (_JsonHandler,), {"payload": payload, "expected_path": path})
    server, _ = _serve(handler)
    return server


def test_service_pack_http_executor_ready_when_up():
    server = _json_server(b"{}")
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        row = _plan_row("http", url)
        row["model_packs"] = ["svc-pack"]
        result = _plan([row])
        assert result.packs["svc-pack"].ready
        assert result.capabilities == ["translate-batch"]
    finally:
        server.shutdown()


def test_service_pack_http_executor_not_ready_when_down():
    dead = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    port = dead.server_port
    dead.server_close()
    row = _plan_row("http", f"http://127.0.0.1:{port}/v1")
    row["model_packs"] = ["svc-pack"]
    result = _plan([row])
    # An unreachable endpoint makes the executor itself unrunnable, so the
    # route is skipped before any pack state is produced.
    assert result.packs == {}
    assert result.fetches == []
    assert result.capabilities == []
    assert any("unreachable" in note for note in result.notes)


def test_service_pack_vllm_served_model_probe(monkeypatch):
    """vLLM pack readiness = GET /v1/models ids == {pinned served model}."""
    pytest.importorskip("sparehour_workloads.qwen")
    server = _json_server(
        json.dumps({"object": "list", "data": [
            {"id": "qwen-pinned", "object": "model"}]}).encode()
    )
    try:
        row = _plan_row(
            "plugin", "sparehour_workloads.qwen:VllmBatchExecutor",
            {
                "VLLM_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "VLLM_SERVED_MODEL": "qwen-pinned",
                "VLLM_MODEL_PACK": "qwen-pack",
            },
        )
        row["model_packs"] = ["qwen-pack"]
        result = _plan([row])
        assert result.packs["qwen-pack"].ready
    finally:
        server.shutdown()


def test_service_pack_vllm_wrong_model_not_ready(monkeypatch):
    pytest.importorskip("sparehour_workloads.qwen")
    server = _json_server(
        json.dumps({"object": "list", "data": [
            {"id": "other-model", "object": "model"},
            {"id": "qwen-pinned", "object": "model"}]}).encode()
    )
    try:
        row = _plan_row(
            "plugin", "sparehour_workloads.qwen:VllmBatchExecutor",
            {
                "VLLM_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "VLLM_SERVED_MODEL": "qwen-pinned",
            },
        )
        row["model_packs"] = ["qwen-pack"]
        result = _plan([row])
        assert not result.packs["qwen-pack"].ready
        assert any("pinned model" in note for note in result.notes)
    finally:
        server.shutdown()


@pytest.mark.parametrize("prefix", ["", "/backend"])
def test_service_pack_libretranslate_languages_probe(prefix):
    pytest.importorskip("sparehour_workloads.libretranslate")
    import importlib.util

    if importlib.util.find_spec("httpx") is None:
        pytest.skip("httpx not installed")
    server = _json_server(
        json.dumps([{"code": "en", "name": "English"},
                    {"code": "es", "name": "Spanish"}]).encode(),
        path=f"{prefix}/languages",
    )
    try:
        row = _plan_row(
            "plugin",
            "sparehour_workloads.libretranslate:LibreTranslateExecutor",
            {"LIBRETRANSLATE_ENDPOINT":
             f"http://127.0.0.1:{server.server_port}{prefix}/translate"},
        )
        row["model_packs"] = ["libretranslate-pack"]
        result = _plan([row])
        assert result.packs["libretranslate-pack"].ready
        assert result.capabilities == ["translate-batch"]
    finally:
        server.shutdown()


def test_service_pack_libretranslate_empty_languages_not_ready():
    pytest.importorskip("sparehour_workloads.libretranslate")
    import importlib.util

    if importlib.util.find_spec("httpx") is None:
        pytest.skip("httpx not installed")
    server = _json_server(b"[]", path="/languages")
    try:
        row = _plan_row(
            "plugin",
            "sparehour_workloads.libretranslate:LibreTranslateExecutor",
            {"LIBRETRANSLATE_ENDPOINT":
             f"http://127.0.0.1:{server.server_port}/translate"},
        )
        row["model_packs"] = ["libretranslate-pack"]
        result = _plan([row])
        assert not result.packs["libretranslate-pack"].ready
        assert any("languages" in note for note in result.notes)
    finally:
        server.shutdown()


def test_json_probe_rejects_non_json_and_non_loopback(tmp_path):
    server = _json_server(b"not json at all")
    try:
        result = dyn._json_probe(
            {"kind": "x", "url": f"http://127.0.0.1:{server.server_port}/x"},
            None,
        )
        assert not result["ok"]
    finally:
        server.shutdown()
    result = dyn._json_probe({"kind": "x", "url": "http://192.0.2.1:9/x"}, None)
    assert not result["ok"]
    result = dyn._json_probe(
        {"kind": "x", "url": "https://user:pw@127.0.0.1/x"}, None
    )
    assert not result["ok"]


def test_service_probe_cancel_during_wedged_response():
    """Stop unblocks a parked service probe; the child is reaped."""
    listener = _SilentTcpListener()
    try:
        stop = threading.Event()
        outcome = []

        def probe():
            outcome.append(
                dyn._json_probe(
                    {"kind": "libretranslate",
                     "url": f"http://127.0.0.1:{listener.port}/languages"},
                    stop,
                )
            )

        thread = threading.Thread(target=probe, daemon=True)
        thread.start()
        time.sleep(0.5)
        stop.set()
        thread.join(6)
        assert not thread.is_alive()
        (result,) = outcome
        assert not result["ok"]
        assert "cancelled" in result["reason"]
    finally:
        listener.close()
    assert not _fetch_children()


def test_manager_hf_pack_full_cycle(tmp_path, monkeypatch):
    """Manager installs an HF pack via the shared module, then advertises."""
    mp = pytest.importorskip("sparehour_workloads.model_packs")
    lock = _hf_lock()
    target = tmp_path / "packs" / "hf-pack"

    def fake_install(lock_arg, target_arg, *, stop=None):
        _hf_pack_dir(target_arg)

    monkeypatch.setattr(mp, "install_hf_pack", fake_install)
    row = _hf_row(model_dir=str(target))
    runtime = FakeRuntime([row], tmp_path, catalog_entries=[lock])
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates == [["hf-pack"]]
    assert runtime.cap_updates == [["translate-batch"]]
    # Second cycle: verified dir stays advertised, no fetch queued.
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates[-1] == ["hf-pack"]


def test_manager_hf_bitrot_unadvertise_rebuild(tmp_path, monkeypatch):
    """Advertised HF pack with corrupted bytes: un-advertised, rebuilt via
    the shared installer, re-advertised — the Argos flow verbatim."""
    mp = pytest.importorskip("sparehour_workloads.model_packs")
    lock = _hf_lock()
    target = tmp_path / "packs" / "hf-pack"
    _hf_pack_dir(target)
    (target / "config.json").write_bytes(b"corrupted")

    def fake_install(lock_arg, target_arg, *, stop=None):
        (target_arg / "config.json").write_bytes(b"{}")

    monkeypatch.setattr(mp, "install_hf_pack", fake_install)
    row = _hf_row(model_dir=str(target))
    runtime = FakeRuntime(
        [row], tmp_path, model_packs=["hf-pack"], catalog_entries=[lock]
    )
    assert dyn.manage_cycle(runtime) is True
    assert runtime.pack_updates == [[], ["hf-pack"]]
    assert runtime.config["model_packs"] == ["hf-pack"]


def test_manager_service_pack_advertised_when_ready(tmp_path):
    """A demanded service pack registers only while its endpoint answers."""
    server = _json_server(
        json.dumps([{"code": "en", "name": "English"}]).encode(),
        path="/languages",
    )
    pytest.importorskip("sparehour_workloads.libretranslate")
    import importlib.util

    if importlib.util.find_spec("httpx") is None:
        server.shutdown()
        pytest.skip("httpx not installed")
    try:
        row = _plan_row(
            "plugin",
            "sparehour_workloads.libretranslate:LibreTranslateExecutor",
            {"LIBRETRANSLATE_ENDPOINT":
             f"http://127.0.0.1:{server.server_port}/translate"},
        )
        row["model_packs"] = ["libretranslate-pack"]
        runtime = FakeRuntime([row], tmp_path)
        assert dyn.manage_cycle(runtime) is True
        assert runtime.config["model_packs"] == ["libretranslate-pack"]
        assert runtime.cap_updates == [["translate-batch"]]
    finally:
        server.shutdown()
