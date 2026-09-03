"""Local web dashboard for the worker: status, queues, selection, pause.

Plain http.server — no extra dependencies. Binds to localhost by default;
treat it as a local control surface, not an internet-facing one.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


def make_server(
    *,
    status_provider: Callable[[], dict[str, Any]],
    pause_handler: Callable[[bool], None],
    select_handler: Callable[[str, bool], tuple[bool, str]],
    prefetch_handler: Callable[[], None],
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                try:
                    payload = status_provider()
                    self._send(
                        200,
                        json.dumps(payload).encode("utf-8"),
                        "application/json",
                    )
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b"invalid json", "text/plain")
                return
            if self.path == "/api/pause":
                pause_handler(bool(body.get("paused")))
                self._send(200, b'{"ok":true}', "application/json")
            elif self.path == "/api/select":
                ok, message = select_handler(str(body.get("capability", "")), bool(body.get("selected")))
                self._send(
                    200 if ok else 409,
                    json.dumps({"ok": ok, "message": message}).encode(),
                    "application/json",
                )
            elif self.path == "/api/prefetch":
                threading.Thread(target=prefetch_handler, daemon=True).start()
                self._send(200, b'{"ok":true,"message":"prefetch started"}', "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return ThreadingHTTPServer((host, port), Handler)
