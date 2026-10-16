"""Owned one-shot fetch child for the dynamic manager.

Every remote read the dynamic mode needs — a lock catalog, a pack archive,
or an endpoint reachability HEAD — runs as ``sys.executable -m sparehour_worker.fetch``
inside its own subprocess. Process termination is the cancellation
mechanism: DNS, TCP connect, TLS handshake, headers, and a trickled body
can all park inside the OS where no in-process interrupt reaches them, but
none of them survives ``terminate``.

Protocol (no URL or query secret ever appears in argv):

  stdin:  {"kind": "catalog"|"archive"|"probe", "url": str,
           "destination": str (catalog/archive), "max_bytes": int,
           "timeout": float}
  stdout: one JSON object —
          {"ok": true} or {"ok": false, "kind": TypeName-or-"HTTP status N"}
          or {"ok": false, "policy": sanitized message}

``run`` is the parent half: it owns the child end-to-end, enforces the
stop event and the hard total deadline, and always reaps. Catalog and
archive bodies stream straight to a caller-chosen file so a huge or
hostile body never lands in a pipe the parent must drain; only the bounded
classified result returns on stdout — never remote text.
"""

from __future__ import annotations

import http.client
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

_POLL_SECONDS = 0.25
_TERMINATE_GRACE_SECONDS = 2.0
_MAX_RESPONSE_BYTES = 4096  # the child only ever prints one small JSON line


class FetchFailed(Exception):
    """Stop fired, the deadline passed, or the child could not report."""


# -- child -------------------------------------------------------------------


def _credentialed(url: str) -> bool:
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return False
    return bool(parsed.username or parsed.password)


def _sanitize_url(url: str) -> str:
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return "URL"
    host = parsed.hostname or ""
    host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{host}{parsed.path}"


class _Refused(Exception):
    """A policy refusal the parent surfaces verbatim (already sanitized)."""


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirect targets that downgrade or smuggle credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not str(newurl).startswith("https://"):
            raise _Refused(
                f"refusing insecure redirect to {_sanitize_url(newurl)!r}"
            )
        if _credentialed(newurl):
            raise _Refused(
                f"refusing redirect to credentialed URL {_sanitize_url(newurl)!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Probes never chase redirects: any HTTP response proves a listener."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_bounded(response, destination: str, cap: int, over: str) -> None:
    """Stream the body to ``destination``, refusing past ``cap`` bytes."""
    declared = response.headers.get("Content-Length")
    if declared:
        try:
            if int(declared) > cap:
                raise _Refused(over)
        except ValueError:
            pass  # unparseable length: the read bound below still applies
    total = 0
    with open(destination, "wb") as output:
        while True:
            chunk = response.read1(1024 * 1024)
            if not chunk:
                return
            total += len(chunk)
            if total > cap:
                raise _Refused(over)
            output.write(chunk)


def _error_kind(exc: BaseException) -> str:
    """``TypeName`` or ``HTTP status N`` — never remote-controlled text."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP status {exc.code}"
    if isinstance(exc, http.client.HTTPException) and getattr(exc, "args", None):
        # Some HTTPException subclasses embed a peer-supplied reason.
        return type(exc).__name__
    return type(exc).__name__


def _main() -> int:
    spec = json.loads(sys.stdin.read())
    kind = spec["kind"]
    url = str(spec["url"])
    timeout = float(spec["timeout"])
    handlers: list = []
    if kind == "probe":
        # Loopback-HTTP or HTTPS, HEAD only, no redirects — policy was
        # already enforced by the parent before it spawned us.
        handlers.append(_NoRedirect)
    else:
        if not url.startswith("https://"):
            raise _Refused(f"refusing non-HTTPS URL {_sanitize_url(url)!r}")
        handlers.append(_HttpsOnlyRedirect)
    if _credentialed(url):
        raise _Refused(f"refusing credentialed URL {_sanitize_url(url)!r}")
    request = urllib.request.Request(
        url,
        method="HEAD" if kind == "probe" else "GET",
        headers={"User-Agent": "sparehour-worker-dynamic/1.0"},
    )
    # Verified TLS: the default context does cert+hostname checks and honors
    # SSL_CERT_FILE/SSL_CERT_DIR for site trust anchors.
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        *handlers,
    )
    with opener.open(request, timeout=timeout) as response:
        if kind == "probe":
            pass  # headers alone prove liveness
        elif kind == "catalog":
            max_bytes = int(spec["max_bytes"])
            _read_bounded(
                response, spec["destination"], max_bytes,
                f"model catalog exceeds {max_bytes} bytes",
            )
        else:
            _read_bounded(
                response, spec["destination"], int(spec["max_bytes"]),
                "model archive exceeds maximum size",
            )
    return 0


def _child_entry() -> int:
    try:
        _main()
    except _Refused as exc:
        payload = {"ok": False, "policy": str(exc)}
    except BaseException as exc:  # classified; remote text never crosses
        payload = {"ok": False, "kind": _error_kind(exc)}
    else:
        payload = {"ok": True}
    print(json.dumps(payload))
    return 0


# -- parent -------------------------------------------------------------------


def run(spec: dict, *, stop=None, deadline: float, cancelled: str,
        timed_out: str) -> dict:
    """Run one fetch child to completion; return its result payload.

    Polls the child against ``stop`` and the monotonic ``deadline``; on
    either it terminates, escalates to kill if needed, and always reaps —
    which cancels OS DNS, connect, TLS, header, and body phases alike.
    The child's stdout result is the only cross-process payload: bounded,
    classified JSON, never raw remote text.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "sparehour_worker.fetch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        try:
            proc.stdin.write(json.dumps(spec).encode())
        except (BrokenPipeError, OSError):
            pass  # child already exited; its stdout says why
        proc.stdin.close()
        while True:
            if stop is not None and stop.is_set():
                raise FetchFailed(cancelled)
            if time.monotonic() > deadline:
                raise FetchFailed(timed_out)
            try:
                proc.wait(_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
        output = proc.stdout.read(_MAX_RESPONSE_BYTES + 1)
        if proc.returncode != 0 or len(output) > _MAX_RESPONSE_BYTES:
            raise FetchFailed("fetch child exited without a result")
        try:
            result = json.loads(output)
        except ValueError:
            raise FetchFailed("fetch child returned an unreadable result")
        if not isinstance(result, dict) or "ok" not in result:
            raise FetchFailed("fetch child returned an unreadable result")
        return result
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


if __name__ == "__main__":
    sys.exit(_child_entry())
