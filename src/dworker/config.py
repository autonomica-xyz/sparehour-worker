"""Worker configuration persisted as a single JSON file (mode 0600)."""

from __future__ import annotations

import json
import os
import socket
import stat
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "endpoint": "",
    "token": "",
    "agent_id": socket.gethostname().split(".")[0] or "worker",
    "provider": "local",
    # Capabilities this worker advertises. Empty selection claims nothing.
    "capabilities": [],
    "model_packs": [],
    "labels": {},
    # Extra vendor=cpu slots for spare CPU compute on GPU-less machines.
    "cpu_slots": 0,
    # SETI@home mode: only claim work while the machine is idle/locked.
    "idle_only": False,
    "idle_after_seconds": 300,
    "workdir": "~/.local/state/dworker/work",
    "dashboard": True,
    "dashboard_host": "127.0.0.1",
    "dashboard_port": 7788,
    "poll_seconds": 1.0,
    "heartbeat_seconds": 15,
    "lease_seconds": 90,
}


def config_path() -> Path:
    root = os.environ.get("DWORKER_CONFIG")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "dworker" / "config.json"


def load(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    config = dict(DEFAULTS)
    if target.is_file():
        stored = json.loads(target.read_text(encoding="utf-8"))
        config.update({key: value for key, value in stored.items() if key in DEFAULTS})
    config.setdefault("agent_id", DEFAULTS["agent_id"])
    return config


def save(config: dict[str, Any], path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    had_file = target.exists()
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not had_file or (target.stat().st_mode & stat.S_IRWXG | stat.S_IRWXO):
        target.chmod(0o600)
    return target


def workdir_path(config: dict[str, Any]) -> Path:
    return Path(config["workdir"]).expanduser()
