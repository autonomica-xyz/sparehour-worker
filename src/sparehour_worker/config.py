"""Worker configuration persisted as a single JSON file (mode 0600).

The file can hold the control-plane token, so writes are atomic and the
file is born private: the payload goes to a same-directory tempfile that is
created with mode 0600 and fsynced, then ``os.replace`` swaps it in. There
is no write-then-chmod window, a failed save leaves the previous file
untouched, and a symlink at the target is replaced rather than written
through — ``save`` never writes credentials to a symlink's target.

``load`` and ``save`` run everything through ``validate``: unknown keys are
dropped, missing keys take defaults, and wrong types or out-of-range values
fail loudly instead of silently healing into surprising behavior.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
    # Dynamic workloads: watch per-route queue depth (requires a token with
    # `perms=dynamic`), fetch missing model packs from the lock catalog, and
    # advertise exactly the capabilities with pending work.
    "dynamic": False,
    "dynamic_poll_seconds": 60,
    "model_catalog": "",
    "workdir": "~/.local/state/sparehour/worker/work",
    "dashboard": True,
    "dashboard_host": "127.0.0.1",
    "dashboard_port": 7788,
    "poll_seconds": 1.0,
    "heartbeat_seconds": 15,
    "lease_seconds": 90,
}

# (min, max) bounds for numeric fields. Both endpoints inclusive.
_RANGES: dict[str, tuple[float, float]] = {
    "cpu_slots": (0, 1024),
    "idle_after_seconds": (0, 30 * 86400),
    "dynamic_poll_seconds": (1, 86400),
    "dashboard_port": (1, 65535),
    "poll_seconds": (0.001, 3600),
    "heartbeat_seconds": (1, 86400),
    "lease_seconds": (1, 7 * 86400),
}

_TRUE_WORDS = {"1", "true", "yes", "on"}
_FALSE_WORDS = {"0", "false", "no", "off"}

# DNS labels: no leading/trailing dot or hyphen, no underscore; literals and
# alternate spellings (hex/octal/dword loopback) never match this.
_HOSTNAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")


class ConfigError(ValueError):
    """Raised for any invalid config key, value, file, or endpoint."""


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_loopback_host(host: str) -> bool:
    """Literal loopback only: `localhost` or an address in 127.0.0.0/8 / ::1.

    Deliberately excludes other hostnames (even ones that resolve to
    loopback), trailing-dot aliases, and alternate IPv4 spellings like
    ``0x7f000001`` — allowlisting a name that merely *resolves* to loopback
    would let an attacker-controlled resolver smuggle an HTTP endpoint.
    """
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def validate_endpoint(endpoint: str) -> str:
    """Normalize and policy-check a control-plane endpoint.

    HTTPS is mandatory; plain HTTP is accepted only for a literal loopback
    host (``localhost`` / 127.0.0.0/8 / ::1) for local development. Embedded
    credentials, query strings, and fragments are rejected, as are hostnames
    that are neither a syntactically valid DNS name nor an IP literal. The
    returned string keeps the endpoint usable verbatim while normalizing
    case and stripping a trailing slash from the path.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ConfigError("endpoint must be a non-empty URL")
    try:
        parts = urlsplit(endpoint.strip())
        port = parts.port  # raises ValueError when out of range or malformed
    except ValueError as exc:
        raise ConfigError(f"invalid endpoint URL: {exc}") from exc
    if parts.scheme not in ("https", "http"):
        raise ConfigError(
            f"endpoint must use https:// (or http:// on loopback), got {parts.scheme or 'no scheme'!r}"
        )
    host = parts.hostname or ""
    if not host:
        raise ConfigError("endpoint URL has no host")
    if parts.username is not None or parts.password is not None:
        raise ConfigError("endpoint must not contain credentials")
    if parts.query or parts.fragment:
        raise ConfigError("endpoint must not contain a query string or fragment")
    if not _is_ip_literal(host) and not _HOSTNAME_RE.match(host):
        raise ConfigError(f"endpoint host {host!r} is not a valid hostname or IP literal")
    if parts.scheme == "http" and not _is_loopback_host(host):
        raise ConfigError(
            f"endpoint host {host!r} requires https:// — plain HTTP is only allowed on loopback"
        )
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    path = parts.path
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""
    return f"{parts.scheme}://{netloc}{path}"


def validate_dashboard_host(host: str) -> str:
    """Normalize and policy-check the dashboard bind host: loopback or nothing.

    ``localhost`` is normalized to the literal ``127.0.0.1`` — a fixed
    spelling, not a DNS lookup — so a saved ``localhost`` still starts.
    IPv4-mapped loopback (``::ffff:127.0.0.1``) normalizes to its mapped
    address.  Everything else is rejected: wildcards, non-loopback literals,
    and hostnames that merely *resolve* to loopback (an attacker-controlled
    resolver could otherwise smuggle an HTTP endpoint).
    """
    normalized = host.strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    if _is_loopback_host(normalized):
        mapped = getattr(ipaddress.ip_address(normalized), "ipv4_mapped", None)
        return str(mapped) if mapped is not None else normalized
    raise ConfigError(f"dashboard host must be a literal loopback IP (127.0.0.1 or ::1), got {_short(host)}")


def _check_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ConfigError(f"{key!r} must be a boolean ({sorted(_TRUE_WORDS)} / {sorted(_FALSE_WORDS)}), got {_short(value)}")


def _check_int(key: str, value: Any) -> int:
    # bool is a subclass of int; True is not a valid slot count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key!r} must be an integer, got {_short(value)}")
    low, high = _RANGES[key]
    if not (low <= value <= high):
        raise ConfigError(f"{key!r} must be between {low:g} and {high:g}, got {value}")
    return value


def _check_float(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key!r} must be a number, got {_short(value)}")
    result = float(value)
    low, high = _RANGES[key]
    if not math.isfinite(result) or not (low <= result <= high):
        raise ConfigError(f"{key!r} must be between {low:g} and {high:g}, got {_short(value)}")
    return result


def _check_str_list(key: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{key!r} must be a list of strings, got {_short(value)}")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{key!r} entries must be non-empty strings, got {_short(item)}")
        result.append(item.strip())
    return result


def _check_labels(key: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{key!r} must be an object of string:string pairs, got {_short(value)}")
    result: dict[str, str] = {}
    for name, label_value in value.items():
        if not isinstance(name, str) or not isinstance(label_value, str):
            raise ConfigError(f"{key!r} entries must map string to string, got {_short(name)}")
        result[name] = label_value
    return result


def _check_endpoint(key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string, got {_short(value)}")
    if not value.strip():
        return ""
    return validate_endpoint(value)


def _check_model_catalog(key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string, got {_short(value)}")
    text = value.strip()
    if not text or "://" not in text:
        return text  # empty, or a local catalog file path
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise ConfigError(f"invalid {key!r} URL: {exc}") from exc
    if parts.scheme != "https" or not parts.hostname:
        raise ConfigError(f"{key!r} must be a local file path or an https:// URL")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{key!r} URL must not contain credentials")
    return text


def _check_dashboard_host(key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string, got {_short(value)}")
    return validate_dashboard_host(value)


_SPECIAL_CHECKS = {
    "endpoint": _check_endpoint,
    "model_catalog": _check_model_catalog,
    "labels": _check_labels,
    "dashboard_host": _check_dashboard_host,
}


def _check(key: str, value: Any) -> Any:
    """Validate (and normalize) one value against its default's type."""
    default = DEFAULTS[key]
    special = _SPECIAL_CHECKS.get(key)
    if special is not None:
        return special(key, value)
    if isinstance(default, bool):
        return _check_bool(key, value)
    if isinstance(default, int):
        return _check_int(key, value)
    if isinstance(default, float):
        return _check_float(key, value)
    if isinstance(default, list):
        return _check_str_list(key, value)
    if isinstance(default, str):
        if not isinstance(value, str):
            shown = "***" if key == "token" else _short(value)
            raise ConfigError(f"{key!r} must be a string, got {shown}")
        return value.strip()
    raise ConfigError(f"{key!r} has no validator")


def coerce(key: str, value: str) -> Any:
    """Parse a CLI string into the typed config value for ``key``.

    Booleans accept only an explicit true/false word — a typo like
    ``idle_only=treu`` is an error, not a silent ``False``. Integers and
    floats must parse as that type (``cpu_slots=2.5`` fails). ``labels``
    takes a JSON object; lists take a comma-separated string.
    """
    if key not in DEFAULTS:
        raise ConfigError(f"unknown config key {key!r}")
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return _check_bool(key, value)
    if isinstance(default, int):
        try:
            parsed = int(value.strip())
        except ValueError:
            raise ConfigError(f"{key!r} expects an integer, got {_short(value)}") from None
        return _check(key, parsed)
    if isinstance(default, float):
        try:
            parsed_f = float(value.strip())
        except ValueError:
            raise ConfigError(f"{key!r} expects a number, got {_short(value)}") from None
        return _check(key, parsed_f)
    if isinstance(default, dict):
        try:
            parsed_d = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{key!r} expects a JSON object: {exc.msg}") from exc
        return _check(key, parsed_d)
    if isinstance(default, list):
        return _check(key, [item.strip() for item in value.split(",") if item.strip()])
    return _check(key, value)



def validate(config: dict[str, Any]) -> dict[str, Any]:
    """Return a complete, checked config: defaults filled, unknown keys dropped."""
    if not isinstance(config, dict):
        raise ConfigError("config must be a JSON object")
    result = {key: _check(key, config.get(key, copy.deepcopy(default)))
              for key, default in DEFAULTS.items()}
    if result["heartbeat_seconds"] >= result["lease_seconds"]:
        raise ConfigError(
            f"'heartbeat_seconds' ({result['heartbeat_seconds']:g}) must be less than "
            f"'lease_seconds' ({result['lease_seconds']:g})"
        )
    return result


def config_path() -> Path:
    root = os.environ.get("SPAREHOUR_WORKER_CONFIG")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sparehour" / "worker.json"


def load(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    config = copy.deepcopy(DEFAULTS)
    if target.is_file():
        try:
            stored = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot load config from {target}: {exc}") from exc
        if not isinstance(stored, dict):
            raise ConfigError(f"config file {target} must contain a JSON object")
        config.update({key: value for key, value in stored.items() if key in DEFAULTS})
    try:
        return validate(config)
    except ConfigError as exc:
        raise ConfigError(f"invalid config in {target}: {exc}") from exc


def save(config: dict[str, Any], path: Path | None = None) -> Path:
    """Atomically persist ``config`` (validated first) as mode-0600 JSON.

    The tempfile is created private in the same directory and swapped in with
    ``os.replace``, so readers never see a half-written or world-readable
    file, a failed save preserves the previous contents, and a symlink at the
    target is atomically replaced instead of being written through.
    """
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(validate(config), indent=2, sort_keys=True) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return target


def workdir_path(config: dict[str, Any]) -> Path:
    return Path(config["workdir"]).expanduser()
