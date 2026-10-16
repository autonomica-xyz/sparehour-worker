"""Dynamic workload mode: watch queue depth, fetch model packs, switch.

Permission model: the control plane only includes per-route `pending_jobs`
in `/v1/workloads` for tokens with the `perms=dynamic` scope. The manager
refuses to run without that signal, so enabling dynamic mode is always a
deliberate grant by the operator, not something any agent token can do.

Switching itself needs no orchestration: a slot advertising N capabilities
claims from whichever matching queue has work, and re-registration preserves
busy slots. This module only decides WHAT to advertise and downloads the
model packs that decision requires.

Trust model: an advertised model pack must correspond to a directory that
passes full manifest + per-file digest verification against the lock entry
the route pins — either the catalog lock, or, when no catalog is configured,
the route's own identity pins (pack id, archive sha256, language pair for
Argos locks; model id + immutable 40-hex revision for huggingface locks).
Fixed model services (LibreTranslate, a warm vLLM server, plain HTTP
executors) are instead "service packs": readiness is the trusted endpoint
answering with the pinned model — they carry no download at all.
Configuration bookkeeping ("installed") is a hint, never proof.
Provisioning only ever replaces directories it can prove are wholly its own:
a builder manifest tied to the lock AND every file, directory, and link in
the tree accounted for by that manifest. Foreign content — unlisted files,
directories, symlinks, or anything reached through a symlink in the path —
is refused and preserved, never deleted.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import importlib
import importlib.util
import ipaddress
import json
import os
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from sparehour_workloads.argos_pack import (
    MAX_ARCHIVE_BYTES,
    build_model_pack_from_lock,
)

from . import fetch as fetch_mod
from . import queues as queues_mod

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # bounded read size for digest checks
_CATALOG_MAX_BYTES = 8 * 1024 * 1024
_ENDPOINT_PROBE_TIMEOUT = 3.0
# Per-request socket timeout and hard total fetch deadline, both enforced
# inside the owned fetch child (timeout) and by the parent's poll against
# it (deadline): process termination cancels DNS, connect, TLS, headers,
# and a trickled body alike.
_CATALOG_SOCKET_TIMEOUT = 30.0
_ARCHIVE_SOCKET_TIMEOUT = 120.0
_CATALOG_DEADLINE_SECONDS = 60.0
_ARCHIVE_DEADLINE_SECONDS = 3600.0
# Service-pack readiness probes (bounded JSON GET via an owned child).
_SERVICE_PROBE_TIMEOUT = 5.0
_SERVICE_PROBE_BODY_MAX = 256 * 1024
_SERVICE_RESULT_MAX_BYTES = 4096
_POLL_SECONDS = 0.25
_TERMINATE_GRACE_SECONDS = 2.0

# Known pack formats in lock-catalog entries. Missing "format" means the
# legacy Argos archive shape; "huggingface" entries pin model_id to an
# immutable 40-hex revision and install through
# sparehour_workloads.model_packs. Anything else is recognized as a
# deliberate catalog entry but never provisioned or advertised.
_PACK_FORMATS = ("argos", "huggingface")

# HF pack environment triples. A route associates a demanded pack with a
# local directory via <P>_MODEL_PACK=<pack> + <P>_MODEL_ID + immutable
# <P>_MODEL_REVISION; the pack installs to <P>_MODEL_DIR. Any
# <P>_MODEL_PACK claim is honored — a new family needs no code change —
# and a catalog entry's model_id also matches an unpinned triple.
_HF_TARGET_ENV = re.compile(r"^(?P<prefix>[A-Z0-9_]+)_MODEL_DIR$")

# Route executor environment keys that pin pack identity. The lock catalog
# must agree with every demanded route's pins — even when the pack is already
# installed or planned — or the catalog is stale for that pack.
_IDENTITY_PINS: tuple[tuple[str, str, str], ...] = (
    ("ARGOS_MODEL_ARCHIVE_SHA256", "sha256", "archive sha256"),
    ("ARGOS_MODEL_PACK", "model_pack", "pack id"),
    ("ARGOS_SOURCE_LANGUAGE", "from_code", "source language"),
    ("ARGOS_TARGET_LANGUAGE", "to_code", "target language"),
)

# Optional dependencies that shipped plugin executors import lazily at job
# time. A findable handler without them advertises work it can never finish.
# A bare module key applies to every handler in it; a module:attribute key
# applies to one handler, so sibling executors sharing a module can carry
# different runtime images.
_PLUGIN_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "sparehour_workloads.argos_translate": ("ctranslate2", "sentencepiece"),
    "sparehour_workloads.sdxl": ("torch", "diffusers"),
    "sparehour_workloads.libretranslate": ("httpx",),
    "sparehour_workloads.transcription:ParakeetExecutor": (
        "nemo",
        "torch",
        "lightning",
    ),
    "sparehour_workloads.transcription:WhisperXExecutor": (
        "whisperx",
        "torch",
        "torchaudio",
        "faster_whisper",
        "ctranslate2",
        "pyannote",
        "transformers",
        "soundfile",
        "nltk",
    ),
    "sparehour_workloads.qwen": ("httpx",),
}

# Bounds for the docker daemon probe: an installed binary means nothing when
# the daemon socket is unreachable, so container routes get a real — short,
# sanitized — `docker info` check. The short TTL keeps a multi-route
# planning pass from spawning one probe per route.
_DOCKER_PROBE_TIMEOUT = 3.0
_DOCKER_PROBE_TTL_SECONDS = 30.0
_docker_probe_cache = {"at": float("-inf"), "result": (False, "not probed")}


class DynamicError(RuntimeError):
    pass


class PackVerificationError(DynamicError):
    """An on-disk pack directory failed manifest/digest verification."""

def _error_kind(exc: BaseException) -> str:
    """``TypeName`` or ``HTTP status N`` — never remote-controlled text."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP status {exc.code}"
    if isinstance(exc, http.client.HTTPException) and getattr(exc, "args", None):
        return type(exc).__name__
    return type(exc).__name__





def _credentialed(url: str) -> bool:
    """True when ``url`` embeds userinfo (user[:password]@host)."""
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return False
    return bool(parsed.username or parsed.password)


def _sanitize_url(url: str) -> str:
    """``url`` with userinfo, params, query, and fragment stripped.

    Route/catalog-controlled URLs carry credentials and signed query strings
    that must never land in status notes or surfaced exceptions.
    """
    try:
        parsed = urlparse(str(url))
        port = parsed.port
    except ValueError:
        return "<unparseable URL>"
    if not parsed.scheme:
        return str(url)  # not a URL (a local path): nothing URL-secret to hide
    netloc = parsed.hostname or ""
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if port is not None:
        netloc += f":{port}"
    return f"{parsed.scheme}://{netloc}{parsed.path}"




def _fetch_to_file(
    spec: dict[str, Any], destination: Path, *,
    stop: Any | None, deadline: float, cancelled: str, timed_out: str,
    failed: str,
) -> None:
    """Stream a remote body to ``destination`` via the owned fetch child.

    ``fetch_mod.run`` polls the child against ``stop`` and the total
    ``deadline``, terminates then kills it, and always reaps — cancelling
    DNS, connect, TLS, header, and body phases alike. The child reports a
    bounded classified result (never remote text): ``policy`` messages are
    already sanitized and surface verbatim; failure kinds are classified
    behind ``failed``.
    """
    try:
        result = fetch_mod.run(
            spec, stop=stop, deadline=deadline,
            cancelled=cancelled, timed_out=timed_out,
        )
    except fetch_mod.FetchFailed as exc:
        raise DynamicError(str(exc)) from exc
    if result.get("ok"):
        return
    policy = result.get("policy")
    if policy:
        raise DynamicError(str(policy))
    raise DynamicError(f"{failed}: {result.get('kind', 'unknown error')}")


def load_catalog(
    source: str, *, stop: Any | None = None
) -> dict[str, dict[str, Any]]:
    """Pack name -> lock entry, from a local file or HTTPS URL.

    Accepts the control plane's lock-catalog format (``{"models": [...]}``)
    or a bare list of lock entries. Entries keep the exact shape
    ``build_model_pack_from_lock`` verifies (url, sha256, archive_root, ...).
    Remote catalogs require HTTPS with certificate validation and are
    bounded in size and total fetch time; ``stop`` (an Event-like) cancels
    an in-flight fetch promptly. Anything else with a URL scheme is
    rejected.
    """
    if not source:
        return {}
    if source.startswith("https://"):
        if _credentialed(source):
            raise DynamicError(
                f"model catalog URL must not embed credentials: {_sanitize_url(source)!r}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".catalog-", suffix=".json"
        )
        os.close(descriptor)
        catalog_file = Path(temp_name)
        try:
            _fetch_to_file(
                {
                    "kind": "catalog",
                    "url": source,
                    "destination": str(catalog_file),
                    "max_bytes": _CATALOG_MAX_BYTES,
                    "timeout": _CATALOG_SOCKET_TIMEOUT,
                },
                catalog_file,
                stop=stop,
                deadline=time.monotonic() + _CATALOG_DEADLINE_SECONDS,
                cancelled="model catalog fetch cancelled",
                timed_out="model catalog fetch timed out",
                failed=f"model catalog fetch failed for {_sanitize_url(source)!r}",
            )
            try:
                document = json.loads(catalog_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DynamicError(
                    f"model catalog fetch failed for {_sanitize_url(source)!r}: "
                    f"{type(exc).__name__}"
                ) from exc
        finally:
            catalog_file.unlink(missing_ok=True)
    elif "://" in source:
        raise DynamicError(
            f"model catalog must be a local file or HTTPS URL: {_sanitize_url(source)!r}"
        )
    else:
        document = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    entries = document.get("models") if isinstance(document, dict) else document
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise DynamicError("model catalog is not a lock-entry list")
    catalog: dict[str, dict[str, Any]] = {}
    for entry in entries:
        pack = entry.get("model_pack")
        if not isinstance(pack, str) or not pack:
            raise DynamicError("model catalog entry is missing model_pack")
        # Explicit format dispatch: missing means the legacy Argos archive
        # shape; a value outside _PACK_FORMATS stays in the catalog so plan()
        # can note it as unsupported instead of treating it as Argos.
        entry = dict(entry)
        entry.setdefault("format", "argos")
        catalog[pack] = entry
    return catalog


# -- pack directory verification -------------------------------------------


def _first_symlink_component(path: Path) -> Path | None:
    """First symlink anywhere in ``path``'s traversal, else ``None``.

    Walks every component from the filesystem root down to the path itself
    with ``lstat`` so a symlinked ancestor — not just a symlinked leaf —
    is caught. Missing components stop the walk: later components can only
    be created as real directories by our own ``mkdir``/builder calls.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            return None  # absent or unreachable: nothing symlinked to report
        if stat.S_ISLNK(info.st_mode):
            return current
        if not stat.S_ISDIR(info.st_mode):
            return None  # a non-directory ends traversal here
    return None


def _pack_manifest(target: Path) -> dict[str, Any] | None:
    """The builder manifest inside ``target``, or ``None`` if unreadable.

    A symlinked manifest is unreadable: following it would let a swapped-in
    file claim ownership of a directory it does not describe.
    """
    manifest_path = target / "manifest.json"
    if manifest_path.is_symlink():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _manifest_identifies(manifest: dict[str, Any], lock: dict[str, Any]) -> bool:
    """True when the manifest's pack id or archive digest ties it to ``lock``."""
    if manifest.get("model_pack") != lock.get("model_pack"):
        return manifest.get("archive_sha256") == str(lock.get("sha256") or "")
    if str(lock.get("format") or "argos") == "huggingface":
        return manifest.get("format") == "huggingface"
    return True


def _managed_paths(manifest: dict[str, Any]) -> tuple[set[PurePosixPath], set[PurePosixPath]] | None:
    """(files, dirs) a real builder manifest claims to own, or ``None``.

    A manifest is only coherent proof of ownership when its file list is a
    nonempty set of safe, unique relative paths — the same invariants
    ``verify_pack_dir`` enforces. Anything else cannot account for the
    directory's contents, so nothing may be deleted on its say-so.
    """
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return None
    file_paths: set[PurePosixPath] = {PurePosixPath("manifest.json")}
    dir_paths: set[PurePosixPath] = set()
    for item in files:
        if not isinstance(item, dict):
            return None
        relative = PurePosixPath(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return None
        if relative in file_paths:
            return None
        file_paths.add(relative)
        for parent in relative.parents:
            if not parent.parts or parent == PurePosixPath("."):
                break
            dir_paths.add(parent)
    if file_paths & dir_paths:
        return None  # one path claimed as both file and directory: incoherent
    return file_paths, dir_paths


def _first_unmanaged_entry(target: Path, manifest: dict[str, Any]) -> Path | None:
    """First on-disk entry the manifest does not account for, else ``None``.

    Ownership of a corrupt pack dir is only proven when EVERYTHING in the
    tree is managed: the manifest itself, every listed member file, and the
    directories that contain them. An unlisted file, an unlisted directory,
    a listed path of the wrong kind, or a symlink anywhere under ``target``
    is unmanaged content — evidence the directory holds data that is not
    ours to delete, so it is reported and never removed.
    """
    managed = _managed_paths(manifest)
    if managed is None:
        return target  # incoherent manifest accounts for nothing on disk
    file_paths, dir_paths = managed
    pending = [target]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue  # unreadable mid-scan: verification fails on its own
        for child in children:
            relative = child.relative_to(target)
            try:
                info = os.lstat(child)
            except OSError:
                continue  # vanished mid-scan: verify/blocker rerun decides
            if stat.S_ISLNK(info.st_mode):
                return child  # a symlink is never managed pack content
            if relative in file_paths:
                if not stat.S_ISREG(info.st_mode):
                    return child  # listed file is not a regular file
                continue
            if relative in dir_paths:
                if not stat.S_ISDIR(info.st_mode):
                    return child  # listed directory is not a directory
                pending.append(child)
                continue
            return child  # unlisted entry: not ours to delete
    return None


def _provision_blocker(target: Path, lock: dict[str, Any]) -> str | None:
    """Why ``install_pack`` must refuse ``target``, or ``None`` if it may build.

    Fail-closed order: a symlink anywhere in the path traversal or anything
    occupying ``target`` that is not provably one of our own pack dirs —
    manifest identity AND fully manifest-accounted contents — blocks
    provisioning. Deleting unowned or unmanaged content is never an option —
    a route-controlled path must not become remote code for unlinking user
    data; the refusal names the offending entry so it is actionable.
    """
    alias = _first_symlink_component(target)
    if alias is not None:
        return (
            f"{alias} is a symlink; refusing to provision {target} through it — "
            "remove or repoint the alias and retry"
        )
    try:
        info = os.lstat(target)
    except OSError:
        return None  # absent: nothing to protect, build is allowed
    if not stat.S_ISDIR(info.st_mode):  # files and symlinks are foreign
        return (
            f"{target} already exists and is not a verified sparehour-worker pack for "
            f"{lock.get('model_pack')!r}; refusing to remove or overwrite it — "
            "clear the path manually or pin a different ARGOS_MODEL_DIR"
        )
    manifest = _pack_manifest(target)
    if manifest is None or not _manifest_identifies(manifest, lock):
        return (
            f"{target} already exists and is not a verified sparehour-worker pack for "
            f"{lock.get('model_pack')!r}; refusing to remove or overwrite it — "
            "clear the path manually or pin a different ARGOS_MODEL_DIR"
        )
    unmanaged = _first_unmanaged_entry(target, manifest)
    if unmanaged is not None:
        return (
            f"{target} is a sparehour-worker pack for {lock.get('model_pack')!r} but holds "
            f"unmanaged content at {unmanaged}; refusing to replace it — "
            "clear the entry or pin a different ARGOS_MODEL_DIR"
        )
    return None


def verify_pack_dir(target: Path, lock: dict[str, Any]) -> None:
    """Full integrity check of an installed pack directory against its lock.
    Verifies the builder's manifest (pack id, archive digest, language pair)
    and every listed member's path safety, size, and SHA-256. ``target``,
    every ancestor component, the manifest, and every member path component
    must be real directories/files — a symlink anywhere in the traversal
    means the check was never on the directory it claims to describe.
    Raises PackVerificationError on the first defect; a passing directory is
    byte-identical to what ``build_model_pack_from_lock`` produced.
    """
    alias = _first_symlink_component(target)
    if alias is not None:
        raise PackVerificationError(f"{target}: path component {alias} is a symlink")
    if not target.is_dir():
        raise PackVerificationError(f"{target}: not a directory")
    manifest_path = target / "manifest.json"
    if manifest_path.is_symlink():
        raise PackVerificationError(f"{target}: manifest is a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackVerificationError(f"{target}: unreadable manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackVerificationError(f"{target}: manifest is not a JSON object")
    if manifest.get("model_pack") != lock.get("model_pack"):
        raise PackVerificationError(f"{target}: manifest pack id does not match lock")
    if manifest.get("archive_sha256") != str(lock.get("sha256") or ""):
        raise PackVerificationError(f"{target}: manifest archive digest does not match lock")
    metadata = manifest.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("from_code") != lock.get("from_code")
        or metadata.get("to_code") != lock.get("to_code")
    ):
        raise PackVerificationError(f"{target}: manifest language pair does not match lock")
    files = manifest.get("files") or []
    if not files:
        raise PackVerificationError(f"{target}: manifest lists no files")
    seen: set[PurePosixPath] = set()
    for item in files:
        if not isinstance(item, dict):
            raise PackVerificationError(f"{target}: manifest entry is not a JSON object")
        relative = PurePosixPath(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise PackVerificationError(f"{target}: unsafe manifest path {relative}")
        if relative in seen:
            raise PackVerificationError(f"{target}: duplicate manifest path {relative}")
        seen.add(relative)
        member = target.joinpath(*relative.parts)
        member_alias = _first_symlink_component(member)
        if member_alias is not None:
            raise PackVerificationError(
                f"{target}: member path component is a symlink: {member_alias}"
            )
        if not member.is_file():
            raise PackVerificationError(f"{target}: member is not a regular file: {relative}")
        digest = hashlib.sha256()
        size = 0
        try:
            with member.open("rb") as handle:
                while chunk := handle.read(_DOWNLOAD_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise PackVerificationError(f"{target}: unreadable member {relative}: {exc}") from exc
        if size != item.get("size") or digest.hexdigest() != item.get("sha256"):
            raise PackVerificationError(f"{target}: digest mismatch for member {relative}")


def is_pack_installed(target: Path, lock: dict[str, Any]) -> bool:
    """True only when ``target`` verifies completely against ``lock``."""
    try:
        verify_pack_dir(target, lock)
    except PackVerificationError:
        return False
    return True


# -- executor prerequisites -------------------------------------------------


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.casefold().strip("[]")
    if host in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _http_endpoint_runnable(
    endpoint: str, stop: Any | None = None
) -> tuple[bool, str]:
    """Safe-and-reachable check for an HTTP model endpoint.

    The probe runs through the same stop/deadline-safe open path as remote
    fetches: DNS, connect, and header reads are interruptible and bounded,
    so a draining manager never waits out a wedged endpoint.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http":
        if not _is_loopback_host(parsed.hostname):
            return False, f"plain-HTTP model endpoint {endpoint!r} is not loopback"
    else:
        return False, f"model endpoint {endpoint!r} must use http(s)"
    if parsed.username or parsed.password:
        return False, "model endpoint must not embed credentials"
    if not parsed.hostname:
        return False, f"model endpoint {endpoint!r} has no host"
    try:
        result = fetch_mod.run(
            {"kind": "probe", "url": endpoint,
             "timeout": _ENDPOINT_PROBE_TIMEOUT},
            stop=stop,
            deadline=time.monotonic() + _ENDPOINT_PROBE_TIMEOUT * 4,
            cancelled="model endpoint probe cancelled",
            timed_out="model endpoint probe timed out",
        )
    except fetch_mod.FetchFailed as exc:
        return False, str(exc)
    if result.get("ok"):
        return True, "endpoint reachable"
    if str(result.get("kind", "")).startswith("HTTP status"):
        return True, "endpoint reachable"  # any HTTP response means a server listens
    if result.get("policy"):
        return False, str(result["policy"])
    return False, (
        f"model endpoint unreachable: {result.get('kind', 'unknown error')}"
    )


def _docker_daemon_runnable() -> tuple[bool, str]:
    """(ok, reason) for whether the docker daemon is actually usable.

    A present binary says nothing about socket access — e.g. this host has
    docker installed but no /var/run/docker.sock permission — so the probe
    runs a bounded `docker info`. Reasons are classified, never the raw
    stderr (local noise, and remote-ish text belongs nowhere in notes).
    """
    cached_at, cached_result = (
        _docker_probe_cache["at"],
        _docker_probe_cache["result"],
    )
    now = time.monotonic()
    if now - cached_at < _DOCKER_PROBE_TTL_SECONDS:
        return cached_result
    docker = shutil.which("docker")
    if docker is None:
        result: tuple[bool, str] = (
            False, "container executor needs docker, which is not installed"
        )
    else:
        try:
            probe = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=_DOCKER_PROBE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            result = (False, "docker daemon did not answer `docker info` in time")
        except FileNotFoundError:
            result = (False, "container executor needs docker, which is not installed")
        except OSError:
            result = (False, "docker daemon probe could not run")
        else:
            if probe.returncode == 0:
                result = (True, "docker daemon reachable")
            elif "permission denied" in (probe.stderr or "").lower():
                result = (
                    False,
                    "docker daemon socket access denied for this user",
                )
            else:
                result = (False, "docker daemon unreachable")
    _docker_probe_cache["at"] = now
    _docker_probe_cache["result"] = result
    return result


def _plugin_runnable(
    detail: str, environment: dict[str, Any], stop: Any | None = None
) -> tuple[bool, str]:
    """Import the handler exactly as the upstream plugin executor does."""
    module_name, separator, attribute = detail.partition(":")
    if not separator or not module_name or not attribute:
        return False, f"plugin handler {detail!r} is not module:attribute"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"plugin module {module_name} is not importable: {exc}"
    handler = getattr(module, attribute, None)
    if not callable(handler):
        return False, f"plugin handler {detail} is missing or not callable"
    dependencies = (
        _PLUGIN_DEPENDENCIES.get(module_name, ())
        + _PLUGIN_DEPENDENCIES.get(f"{module_name}:{attribute}", ())
    )
    for dependency in dependencies:
        try:
            present = importlib.util.find_spec(dependency) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            return False, f"{module_name} needs optional dependency {dependency}"
    if module_name == "sparehour_workloads.libretranslate":
        endpoint = environment.get("LIBRETRANSLATE_ENDPOINT")
        if not endpoint:
            from sparehour_workloads import libretranslate

            endpoint = libretranslate.DEFAULT_ENDPOINT
        return _http_endpoint_runnable(str(endpoint), stop)
    return True, "plugin handler available"


def _executor_runnable(
    executor_type: str,
    executor_detail: str,
    environment: dict[str, Any],
    stop: Any | None = None,
) -> tuple[bool, str]:
    """(ok, reason) for whether this executor can actually run jobs here."""
    if executor_type == "command":
        return True, "command executor is built in"
    if executor_type == "container":
        return _docker_daemon_runnable()
    if executor_type == "http":
        return _http_endpoint_runnable(executor_detail, stop)
    if executor_type == "plugin":
        return _plugin_runnable(executor_detail, environment, stop)
    return False, f"unsupported executor type {executor_type!r}"


# -- service pack probes --------------------------------------------------------

# A "service pack" is a demanded compatibility.model_packs entry served by a
# fixed, already-running model service rather than a downloaded directory.
# Readiness is the trusted endpoint answering AND the service model matching
# the route pin — never a downloaded Argos manifest.
def _service_probe_spec(
    queue: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any] | None:
    """Probe spec for routes served by a fixed model service, or ``None``."""
    executor_type = str(queue.get("executor_type") or "")
    detail = str(queue.get("executor_detail") or "")
    if executor_type == "http":
        return {"kind": "http", "url": detail}
    if executor_type != "plugin":
        return None
    if detail.startswith("sparehour_workloads.libretranslate:"):
        try:
            from sparehour_workloads import libretranslate
        except ImportError:
            return None
        endpoint = environment.get("LIBRETRANSLATE_ENDPOINT") or libretranslate.DEFAULT_ENDPOINT
        return {
            "kind": "libretranslate",
            "url": libretranslate._languages_url(_sanitize_url(str(endpoint))),
        }
    if detail.startswith("sparehour_workloads.qwen:"):
        base = str(environment.get("VLLM_BASE_URL") or "http://127.0.0.1:8000")
        served = environment.get("VLLM_SERVED_MODEL") or environment.get(
            "VLLM_MODEL_PACK"
        )
        return {
            "kind": "vllm",
            "url": f"{base.rstrip('/')}/v1/models",
            "model": str(served) if served else "",
        }
    return None


def _run_probe_child(
    spec: dict[str, Any], *, stop: Any | None, deadline: float,
    cancelled: str, timed_out: str,
) -> dict[str, Any]:
    """Run ``python -m sparehour_worker.dynamic`` as an owned JSON GET child.

    Same lifecycle contract as fetch_mod.run: the spec rides stdin (nothing
    URL- or token-derived reaches argv), the parent polls stop/deadline and
    terminates-then-kills to reap, and only a bounded classified result
    returns — remote text never crosses the pipe.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "sparehour_worker.dynamic"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        try:
            proc.stdin.write(json.dumps(spec).encode())
        except (BrokenPipeError, OSError):
            pass
        proc.stdin.close()
        while True:
            if stop is not None and stop.is_set():
                return {"ok": False, "reason": cancelled}
            if time.monotonic() > deadline:
                return {"ok": False, "reason": timed_out}
            try:
                proc.wait(_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
        output = proc.stdout.read(_SERVICE_RESULT_MAX_BYTES + 1)
        if proc.returncode != 0 or len(output) > _SERVICE_RESULT_MAX_BYTES:
            return {"ok": False, "reason": "probe child exited without a result"}
        try:
            result = json.loads(output)
        except ValueError:
            return {"ok": False, "reason": "probe child returned an unreadable result"}
        if not isinstance(result, dict) or "ok" not in result:
            return {"ok": False, "reason": "probe child returned an unreadable result"}
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


def _json_probe(spec: dict[str, Any], stop: Any | None) -> dict[str, Any]:
    """Bounded, interruptible GET for service-pack readiness.

    Policy is checked in the parent before any spawn: HTTPS anywhere, plain
    HTTP only to loopback, never embedded credentials. The child GETs the
    URL, reads a bounded body, parses JSON, and reports {"ok", "body"} or a
    classified failure.
    """
    url = str(spec.get("url") or "")
    parsed = urlparse(url)
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        pass
    else:
        return {
            "ok": False,
            "reason": f"service probe URL must be HTTPS or loopback HTTP: {_sanitize_url(url)!r}",
        }
    if parsed.username or parsed.password:
        return {"ok": False, "reason": "service probe URL must not embed credentials"}
    if not parsed.hostname:
        return {"ok": False, "reason": "service probe URL has no host"}
    result = _run_probe_child(
        spec, stop=stop,
        deadline=time.monotonic() + _SERVICE_PROBE_TIMEOUT * 4,
        cancelled="service probe cancelled",
        timed_out="service probe timed out",
    )
    if result.get("ok"):
        return {"ok": True, "body": result.get("body")}
    reason = result.get("policy") or result.get("reason") or result.get("kind")
    return {"ok": False, "reason": f"service probe failed: {reason}"}


def _service_pack_ready(spec: dict[str, Any], stop: Any | None) -> tuple[bool, str]:
    """(ok, reason): is the fixed model service answering with the pinned model?"""
    kind = spec["kind"]
    if kind == "http":
        return _http_endpoint_runnable(str(spec["url"]), stop)
    result = _json_probe(spec, stop)
    if not result.get("ok"):
        return False, str(result.get("reason") or "probe failed")
    body = result.get("body")
    if kind == "vllm":
        model = spec.get("model")
        if not model:
            return False, "route does not pin a served model (VLLM_SERVED_MODEL)"
        data = body.get("data") if isinstance(body, dict) else None
        ids = {
            str(entry.get("id"))
            for entry in (data or [])
            if isinstance(entry, dict)
        }
        if ids == {str(model)}:
            return True, "vllm serves the pinned model"
        return False, "vllm endpoint is up but does not serve the pinned model"
    if kind == "libretranslate":
        languages = body if isinstance(body, list) else []
        codes = {
            str(entry.get("code"))
            for entry in languages
            if isinstance(entry, dict) and entry.get("code")
        }
        if not codes:
            return False, "libretranslate answers but serves no languages"
        return True, "libretranslate serving languages"
    return False, f"unknown service probe kind {kind!r}"


def _json_probe_child_entry() -> int:
    """Child entry for `python -m sparehour_worker.dynamic`: one bounded JSON GET."""
    try:
        spec = json.loads(sys.stdin.read())
        url = str(spec["url"])
        timeout = float(spec.get("timeout") or _SERVICE_PROBE_TIMEOUT)
        if not (url.startswith("https://") or url.startswith("http://")):
            raise fetch_mod._Refused(f"refusing probe URL {_sanitize_url(url)!r}")
        if _credentialed(url):
            raise fetch_mod._Refused(f"refusing credentialed URL {_sanitize_url(url)!r}")
        request = urllib.request.Request(
            url, headers={"User-Agent": "sparehour-worker-dynamic/1.0"}
        )
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            fetch_mod._NoRedirect,
        )
        with opener.open(request, timeout=timeout) as response:
            body = response.read(_SERVICE_PROBE_BODY_MAX + 1)
        if len(body) > _SERVICE_PROBE_BODY_MAX:
            raise fetch_mod._Refused("service response exceeds maximum size")
        try:
            payload = json.loads(body)
        except ValueError:
            raise fetch_mod._Refused("service response is not JSON") from None
        result = {"ok": True, "body": payload}
    except fetch_mod._Refused as exc:
        result = {"ok": False, "policy": str(exc)}
    except BaseException as exc:  # classified; remote text never crosses
        result = {"ok": False, "kind": _error_kind(exc)}
    print(json.dumps(result))
    return 0


# -- planning ----------------------------------------------------------------


@dataclass
class PackPlan:
    """Provisioning state for one demanded model pack.

    ``lock`` is the identity each target must verify against: the catalog
    lock when one exists, else a lock synthesized from the routes' identity
    pins (which carry no URL and can never drive a download). ``kind`` is
    the pack format this host must satisfy: "argos", "huggingface",
    "service" (a fixed endpoint serves the model — no local directory), or
    "unsupported" for catalog formats we deliberately never provision.
    """

    lock: dict[str, Any] | None
    kind: str = "argos"
    targets: set[Path] = field(default_factory=set)
    verified: set[Path] = field(default_factory=set)
    probe_specs: list[dict[str, Any]] = field(default_factory=list)
    probes: list[tuple[bool, str]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Advertise only when every demanded target/service check passes."""
        if self.kind == "service":
            return bool(self.probes) and all(ok for ok, _ in self.probes)
        if self.kind == "unsupported":
            return False
        return (
            self.lock is not None
            and bool(self.targets)
            and self.verified == self.targets
            and all(ok for ok, _ in self.probes)
        )


@dataclass
class Plan:
    """One planning pass: what to advertise, fetch, and explain."""

    capabilities: list[str]
    fetches: list[dict[str, Any]]  # {"pack", "lock", "target", "reason"}
    packs: dict[str, PackPlan]
    notes: list[str]


def _check_pack_pins(
    pack: str,
    environment: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    pins_seen: dict[str, dict[str, str]],
    label: str,
) -> None:
    """Validate a demanded route's identity pins for ``pack``.

    Runs for every demanded route — installed or planned packs included —
    because a stale directory or an earlier route proves nothing about this
    route's pins. Raises DynamicError on route-vs-route conflicts and on
    pins that disagree with the catalog entry.
    """
    seen = pins_seen.setdefault(pack, {})
    lock = catalog.get(pack)
    for env_key, lock_key, human in _IDENTITY_PINS:
        pinned = environment.get(env_key)
        if not pinned:
            continue
        pinned = str(pinned)
        previous = seen.get(lock_key)
        if previous is not None and previous != pinned:
            raise DynamicError(
                f"routes pin conflicting {human} for model pack {pack!r} "
                f"({previous!r} vs {pinned!r})"
            )
        seen[lock_key] = pinned
        if lock_key == "model_pack" and pinned != pack:
            raise DynamicError(
                f"{label}: {env_key}={pinned!r} does not match demanded pack {pack!r}"
            )
        if lock is not None and str(lock.get(lock_key) or "") != pinned:
            raise DynamicError(
                f"{label}: catalog {human} for {pack!r} disagrees with route pin "
                f"{pinned!r}; refresh the model catalog"
            )


def _pinned_lock(pack: str, pins: dict[str, str]) -> dict[str, Any] | None:
    """Verification identity synthesized from route pins, or ``None``.

    When the catalog has no entry for a demanded pack, a preinstalled
    directory may still be trusted — but only when the demanding routes pin
    its identity completely: pack id, archive sha256, and language pair.
    Anything less cannot verify a directory against anything, so the pack
    stays unadvertised. The synthesized lock deliberately has no URL:
    identity pins can prove a directory, never source a download.
    """
    lock: dict[str, Any] = {"model_pack": pack}
    for lock_key in ("sha256", "from_code", "to_code"):
        pinned = pins.get(lock_key)
        if not pinned:
            return None
        lock[lock_key] = pinned
    return lock


def _hf_env_claims(environment: dict[str, Any]) -> list[dict[str, str]]:
    """HF pack claims in a route env: MODEL_ID/REVISION/DIR/PACK triples.

    Every ``<P>_MODEL_ID``/``<P>_MODEL_DIR`` key claims the ``<P>`` prefix;
    ``<P>_MODEL_PACK`` explicitly binds the triple to a demanded pack.
    ``ARGOS`` is skipped — its identity pins are the legacy archive shape.
    """
    prefixes: set[str] = set()
    for key in environment:
        for suffix in ("_MODEL_ID", "_MODEL_DIR"):
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                if prefix and prefix != "ARGOS":
                    prefixes.add(prefix)
    claims = []
    for prefix in sorted(prefixes):
        claims.append(
            {
                "prefix": prefix,
                "pack": str(environment.get(f"{prefix}_MODEL_PACK") or ""),
                "id": str(environment.get(f"{prefix}_MODEL_ID") or ""),
                "revision": str(environment.get(f"{prefix}_MODEL_REVISION") or ""),
                "dir": str(environment.get(f"{prefix}_MODEL_DIR") or ""),
            }
        )
    return claims


def _merge_pin(
    seen: dict[str, str], key: str, value: str, human: str, pack: str
) -> None:
    previous = seen.get(key)
    if previous is not None and previous != value:
        raise DynamicError(
            f"routes pin conflicting {human} for model pack {pack!r} "
            f"({previous!r} vs {value!r})"
        )
    seen[key] = value


def _check_hf_claims(
    pack: str,
    claims: list[dict[str, str]],
    lock: dict[str, Any] | None,
    pins_seen: dict[str, dict[str, str]],
    label: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Match HF env claims to ``pack``; return (claims, merged pins).

    A claim binds to the pack via an explicit ``<P>_MODEL_PACK=<pack>``, or
    implicitly when the catalog lock's model_id matches. Every bound claim
    must agree with the lock's pinned model_id/revision and with claims on
    other routes — disagreement is a DynamicError, same as Argos pins.
    Bound claims also carry the trusted pin identity that can synthesize a
    lock when the catalog has none.
    """
    seen = pins_seen.setdefault(pack, {})
    bound: list[dict[str, str]] = []
    for claim in claims:
        explicit = bool(claim["pack"])
        if explicit:
            if claim["pack"] != pack:
                continue  # names a sibling pack, e.g. WHISPERX_ALIGN_MODEL_PACK
            bound.append(claim)
        elif lock is not None and claim["id"] and claim["id"] == str(
            lock.get("model_id") or ""
        ):
            bound.append(claim)
        else:
            continue
    for claim in bound:
        if claim["id"]:
            _merge_pin(seen, "model_id", claim["id"], "model id", pack)
            if lock is not None and str(lock.get("model_id") or "") != claim["id"]:
                raise DynamicError(
                    f"{label}: catalog model_id for {pack!r} disagrees with route "
                    f"pin {claim['id']!r}; refresh the model catalog"
                )
        elif lock is not None:
            raise DynamicError(
                f"{label}: {claim['prefix']}_MODEL_ID is empty for demanded "
                f"hf pack {pack!r}"
            )
        if claim["revision"]:
            _merge_pin(seen, "revision", claim["revision"], "revision", pack)
            if lock is not None and str(lock.get("revision") or "") != claim["revision"]:
                raise DynamicError(
                    f"{label}: catalog revision for {pack!r} disagrees with route "
                    f"pin {claim['revision']!r}; refresh the model catalog"
                )
        elif lock is None:
            continue  # incomplete identity noted later as unverifiable
        else:
            raise DynamicError(
                f"{label}: {claim['prefix']}_MODEL_REVISION missing for demanded "
                f"hf pack {pack!r}"
            )
    return bound, seen


def _hf_locked(pack: str, pins: dict[str, str]) -> dict[str, Any] | None:
    """Synthesize an hf lock from merged route pins, or ``None``.

    Route pins prove identity (model_id + immutable revision) but never
    source a download — there is deliberately no endpoint field here.
    """
    model_id = pins.get("model_id")
    revision = pins.get("revision")
    if not model_id or not revision:
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DynamicError(
            f"hf pack revision must be an immutable 40-hex commit, got {revision!r}"
        )
    return {
        "format": "huggingface",
        "model_pack": pack,
        "model_id": model_id,
        "revision": revision,
        "allow_patterns": ["*"],
    }





def plan(
    report: list[dict[str, Any]],
    *,
    slots: list[Any],
    provider: str,
    host: dict[str, Any],
    agent_labels: dict[str, str],
    installed_packs: set[str],
    catalog: dict[str, dict[str, Any]],
    fallback_dir: Path,
    stop: Any | None = None,
) -> Plan:
    """Pure decision: capabilities to advertise, pack targets to provision.
    Pack kinds: Argos locks verify a downloaded directory against the
    archive lock; ``format: huggingface`` locks verify a snapshot directory
    against the pinned model_id + immutable 40-hex revision (route env
    triples ``<P>_MODEL_PACK/ID/REVISION/DIR`` associate packs to targets);
    uncataloged packs claimed by a fixed model service (a plugin adapter
    like libretranslate/qwen or an http executor) are "service" packs —
    readiness is the trusted endpoint answering with the pinned model, and
    no Argos manifest is ever required. Catalog entries with an unknown
    format are noted and withheld rather than treated as Argos.


    A capability is advertised when at least one of its routes has pending
    jobs, matches this host's hardware (vendor/VRAM/labels — the current
    capability selection and installed packs are deliberately ignored, they
    are what this function decides), and its executor is runnable here.

    Every demanded pack contributes every distinct target directory its
    routes pin. A pack is "ready" only when ALL of its targets pass full
    manifest/digest verification; missing or stale targets are queued for
    verified rebuild — config bookkeeping alone never counts as installed.

    Verification identity comes from the catalog lock when the pack has one.
    With no catalog entry the pack may still run if every target verifies
    against the routes' merged identity pins (preinstalled packs need no
    catalog) — but pins carry no URL, so a pin-verified target that fails
    or is absent can never be fetched. Packs with neither a catalog entry
    nor a fully pin-verified directory are noted and withheld.

    Executor readiness checks may probe the environment (docker daemon,
    HTTP endpoints); ``stop`` interrupts a wedged probe so a draining
    manager never waits one out.
    """
    capabilities: set[str] = set()
    fetches: list[dict[str, Any]] = []
    packs: dict[str, PackPlan] = {}
    notes: list[str] = []
    pins_seen: dict[str, dict[str, str]] = {}
    dir_owner: dict[Path, str] = {}
    slot_dicts = [
        slot.model_dump() if hasattr(slot, "model_dump") else dict(slot) for slot in slots
    ]
    for queue in report:
        if not queue.get("pending_jobs"):
            continue
        label = f"{queue.get('workload', '?')}/{queue.get('route', '?')}"
        capability = str(queue.get("capability") or "")
        requirements = queues_mod.requirements_for(queue)
        hardware_only = {
            key: value
            for key, value in requirements.items()
            if key not in ("capability", "model_packs")
        }
        eligible = False
        last_reason = "no slots discovered"
        for slot in slot_dicts:
            ok, last_reason = queues_mod.route_matches(
                hardware_only,
                slot,
                provider=provider,
                host=host,
                agent_labels=agent_labels,
            )
            if ok:
                eligible = True
                break
        if not eligible:
            notes.append(
                f"{label} [{capability or '-'}]: skipped — hardware ineligible ({last_reason})"
            )
            continue
        environment = dict(queue.get("executor_environment") or {})
        runnable, reason = _executor_runnable(
            str(queue.get("executor_type") or ""),
            str(queue.get("executor_detail") or ""),
            environment,
            stop,
        )
        if not runnable:
            notes.append(f"{label} [{capability or '-'}]: skipped — {reason}")
            continue
        if capability:
            capabilities.add(capability)
        for pack in queue.get("model_packs") or []:
            pack = str(pack)
            lock = catalog.get(pack)
            fmt = str(lock.get("format") or "argos") if lock is not None else None
            claims = _hf_env_claims(environment)
            service_spec = _service_probe_spec(queue, environment)
            entry = packs.get(pack)
            if entry is None:
                entry = packs[pack] = PackPlan(
                    lock=lock,
                    kind=fmt or "argos",
                )
            # Kind conflicts across routes demand different provisioning for
            # the same pack name — never resolve silently.
            if fmt is not None and fmt not in _PACK_FORMATS:
                if entry.kind != "unsupported":
                    entry.kind = "unsupported"
                    notes.append(
                        f"{label}: catalog format {fmt!r} for model pack {pack!r} "
                        "is unsupported — cannot provision or advertise it"
                    )
                continue
            if fmt == "huggingface" or (lock is None and any(
                claim["pack"] == pack for claim in claims
            )):
                if entry.kind != "huggingface":
                    entry.kind = "huggingface"
                bound, pins = _check_hf_claims(
                    pack, claims, lock, pins_seen, label
                )
                for claim in bound:
                    if not claim["dir"]:
                        continue
                    target = Path(claim["dir"])
                    owner = dir_owner.get(target)
                    if owner is not None and owner != pack:
                        raise DynamicError(
                            f"routes pin packs {owner!r} and {pack!r} to the "
                            f"same directory {target}"
                        )
                    dir_owner[target] = pack
                    entry.targets.add(target)
                if not entry.targets:
                    # No env dir pin: the pack lands under the workdir.
                    target = Path(fallback_dir).expanduser() / pack
                    owner = dir_owner.get(target)
                    if owner is not None and owner != pack:
                        raise DynamicError(
                            f"routes pin packs {owner!r} and {pack!r} to the "
                            f"same directory {target}"
                        )
                    dir_owner[target] = pack
                    entry.targets.add(target)
            elif (
                fmt == "argos"
                or environment.get("ARGOS_MODEL_DIR")
                or entry.targets
                or (lock is None and service_spec is None)
            ):
                # Legacy Argos path: an env ARGOS_MODEL_DIR or the fallback
                # workdir directory, verified against archive identity pins.
                if entry.kind == "service" or entry.kind == "huggingface":
                    raise DynamicError(
                        f"{label}: pack {pack!r} mixes argos and "
                        f"{entry.kind} provisioning across routes"
                    )
                if any(claim["pack"] == pack for claim in claims):
                    raise DynamicError(
                        f"{label}: route binds hf pins to pack {pack!r} but the "
                        "catalog classifies it as an argos pack"
                    )
                _check_pack_pins(pack, environment, catalog, pins_seen, label)
                if environment.get("ARGOS_MODEL_DIR"):
                    target = Path(str(environment["ARGOS_MODEL_DIR"]))
                else:
                    target = Path(fallback_dir).expanduser() / pack
                owner = dir_owner.get(target)
                if owner is not None and owner != pack:
                    raise DynamicError(
                        f"routes pin packs {owner!r} and {pack!r} to the same directory {target}"
                    )
                dir_owner[target] = pack
                entry.targets.add(target)
            else:
                # Service pack: a fixed model service is the readiness
                # contract; there is no local directory to verify.
                entry.kind = "service"
            # A demanded pack served partly by a fixed service also needs its
            # endpoint probe — the pack dir alone never proves the service.
            if service_spec is not None and service_spec not in entry.probe_specs:
                entry.probe_specs.append(service_spec)
    # Verify only after the whole pass: a route's pins merge into
    # pins_seen as routes are visited, so per-target verification deferred
    # to here sees every demanded route's identity, not a route-order prefix.
    for pack, entry in packs.items():
        for spec in entry.probe_specs:
            ok, reason = _service_pack_ready(spec, stop)
            entry.probes.append((ok, reason))
            if not ok:
                notes.append(f"model pack {pack}: service not ready — {reason}")
        if entry.kind == "service" or entry.kind == "unsupported":
            if entry.kind == "service" and not entry.probe_specs:
                notes.append(
                    f"model pack {pack!r} needs a fixed model service but its "
                    "routes pin no probeable endpoint — cannot advertise it"
                )
            continue
        if entry.lock is None:
            if entry.kind == "huggingface":
                entry.lock = _hf_locked(pack, pins_seen.get(pack) or {})
            else:
                # No catalog entry: fall back to the routes' merged identity
                # pins so a preinstalled directory can still prove itself.
                entry.lock = _pinned_lock(pack, pins_seen.get(pack) or {})
        fetchable = pack in catalog  # pinned locks have no URL, ever
        if entry.kind == "huggingface" and fetchable and entry.lock is not None:
            # A malformed hf catalog lock can never verify or install: fail
            # the pack early with the reason instead of a doomed fetch.
            try:
                _model_packs().normalize_hf_lock(entry.lock)
            except Exception as exc:
                detail = (
                    str(exc)
                    if isinstance(exc, DynamicError) or _model_pack_error(exc)
                    else _error_kind(exc)
                )
                notes.append(f"model pack {pack}: invalid hf lock — {detail}")
                entry.lock = None
                fetchable = False
        for target in sorted(entry.targets):
            if entry.lock is None:
                notes.append(
                    f"model pack {pack!r} is not in the catalog and its routes "
                    "do not pin a complete identity — cannot verify, "
                    "provision, or advertise it"
                )
                break
            try:
                _verify_pack_target(target, entry.lock)
            except PackVerificationError as exc:
                blocker = _pack_provision_blocker(target, entry.lock)
                if blocker is not None:
                    # Unsafe/unowned/unmanaged target: install_pack will
                    # refuse, so do not queue a doomed fetch — surface the
                    # blocker instead.
                    notes.append(f"model pack {pack} not provisioned — {blocker}")
                    continue
                if not fetchable:
                    notes.append(
                        f"model pack {pack}: target {target} failed "
                        f"verification against the route pins ({exc}) and the "
                        "pack is not in the catalog — cannot download it"
                    )
                    continue
                fetches.append(
                    {"pack": pack, "lock": entry.lock, "target": target, "reason": str(exc)}
                )
                if pack in installed_packs:
                    notes.append(
                        f"model pack {pack}: advertised install failed verification "
                        f"({exc}); reprovisioning"
                    )
            else:
                entry.verified.add(target)
                if pack not in installed_packs:
                    notes.append(
                        f"model pack {pack}: verified at {target} but not advertised yet"
                    )
    return Plan(sorted(capabilities), fetches, packs, notes)


def _model_packs():
    """Lazy access to the shared hf-pack verifier/installer."""
    try:
        from sparehour_workloads import model_packs
    except ImportError as exc:
        raise DynamicError(
            "hf model packs need sparehour_workloads.model_packs "
            "(upgrade the installed sparehour wheel)"
        ) from exc
    return model_packs


def _model_pack_error(exc: BaseException) -> bool:
    """True for authored failures from the shared hf-pack module."""
    try:
        mp = _model_packs()
    except DynamicError:
        return False
    return isinstance(exc, mp.ModelPackError)


def _verify_pack_target(target: Path, lock: dict[str, Any]) -> None:
    """Format-dispatched integrity check for one pack directory."""
    if str(lock.get("format") or "argos") != "huggingface":
        verify_pack_dir(target, lock)
        return
    try:
        mp = _model_packs()
        mp.verify_hf_pack(
            target,
            model_pack=str(lock.get("model_pack") or ""),
            model_id=str(lock.get("model_id") or ""),
            revision=str(lock.get("revision") or ""),
        )
    except DynamicError as exc:
        raise PackVerificationError(str(exc)) from exc
    except Exception as exc:
        if _model_pack_error(exc):
            raise PackVerificationError(str(exc)) from exc
        raise


def _pack_provision_blocker(target: Path, lock: dict[str, Any]) -> str | None:
    """Format-dispatched ownership guard for a failed pack directory."""
    if str(lock.get("format") or "argos") == "huggingface":
        try:
            mp = _model_packs()
        except DynamicError as exc:
            return str(exc)
        return mp.hf_pack_provision_blocker(target, lock)
    return _provision_blocker(target, lock)


def _download(url: str, destination: Path, *, stop: Any | None) -> None:
    """HTTPS-only, byte-bounded, cancellable archive fetch.

    URL-bearing failures are re-raised with the URL sanitized — redirects and
    catalog entries can carry userinfo or signed query strings that must not
    reach status notes — and transport failures are classified, never
    interpolated, so remote-controlled text cannot leak either.
    """
    if not str(url).startswith("https://"):
        raise DynamicError(f"model archive URL must use HTTPS: {_sanitize_url(url)!r}")
    if _credentialed(url):
        raise DynamicError(
            f"model archive URL must not embed credentials: {_sanitize_url(url)!r}"
        )
    _fetch_to_file(
        {
            "kind": "archive",
            "url": url,
            "destination": str(destination),
            "max_bytes": MAX_ARCHIVE_BYTES,
            "timeout": _ARCHIVE_SOCKET_TIMEOUT,
        },
        destination,
        stop=stop,
        deadline=time.monotonic() + _ARCHIVE_DEADLINE_SECONDS,
        cancelled="model download cancelled",
        timed_out="model archive fetch timed out",
        failed=f"model archive fetch failed for {_sanitize_url(url)!r}",
    )


def install_pack(lock: dict[str, Any], target: Path, *, stop: Any | None = None) -> None:
    """Format-dispatched install: ``format: huggingface`` locks download a
    pinned HF snapshot via sparehour_workloads.model_packs; anything
    else is the legacy Argos archive path."""
    if str(lock.get("format") or "argos") == "huggingface":
        mp = _model_packs()
        try:
            mp.install_hf_pack(lock, Path(target), stop=stop)
        except mp.ModelPackError as exc:
            raise DynamicError(str(exc)) from exc
        return
    _install_argos_pack(lock, target, stop=stop)


def _install_argos_pack(lock: dict[str, Any], target: Path, *, stop: Any | None = None) -> None:
    """Verified download + extract. Idempotent: a verified dir is a cache hit.

    This runs under route-controlled paths, so it NEVER deletes content it
    did not create. A symlink anywhere in ``target``'s traversal, or any
    existing file/symlink/directory that is not provably one of our own pack
    dirs — manifest identity plus fully manifest-accounted contents —
    raises DynamicError and leaves the path untouched. A corrupt-but-owned
    pack dir is rebuilt only through a verified staging directory swapped
    in atomically; if the swap or rebuild fails, the previous contents are
    preserved.
    """
    try:
        verify_pack_dir(target, lock)
        return  # already installed and intact
    except PackVerificationError:
        pass
    blocker = _provision_blocker(target, lock)
    if blocker is not None:
        raise DynamicError(f"cannot install model pack {lock.get('model_pack')!r}: {blocker}")
    url = str(lock.get("url") or "")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DynamicError(f"{target}: cannot create parent directories: {exc}") from exc
    descriptor, archive_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".argosmodel", dir=target.parent
    )
    os.close(descriptor)
    archive = Path(archive_name)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-new-", dir=target.parent))
    staging.rmdir()  # reserve a fresh sibling name for the builder's rename
    try:
        _download(url, archive, stop=stop)
        if stop is not None and stop.is_set():
            raise DynamicError("model install cancelled")
        try:
            build_model_pack_from_lock(lock, staging, archive_path=archive)
        except FileExistsError as exc:
            raise DynamicError(f"{staging} appeared during install; not overwriting") from exc
        if os.path.lexists(target):
            _swap_verified_pack(staging, target, lock)
        else:
            staging.rename(target)
    finally:
        archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)


def _swap_verified_pack(staging: Path, target: Path, lock: dict[str, Any]) -> None:
    """Atomically replace owned-but-corrupt ``target`` with verified ``staging``.

    The old directory moves aside first, the new one lands in a single
    rename, and the backup is kept if any step fails — the previous content
    is never destroyed by a rebuild that could not complete. The manifest
    identity/ownership proof is re-checked on the moved-aside backup before
    it is deleted: content that changed between the blocker check and the
    swap is preserved, never removed.
    """
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent))
    backup.rmdir()  # reserve the name; the old dir renames onto it
    os.rename(target, backup)
    try:
        staging.rename(target)
    except OSError:
        try:
            os.rename(backup, target)
        except OSError as restore_exc:
            raise DynamicError(
                f"{target}: install failed and the previous pack could not be "
                f"restored; preserved at {backup}: {restore_exc}"
            ) from restore_exc
        raise DynamicError(
            f"{target}: could not move verified pack into place; previous "
            "contents restored"
        )
    # The verified pack is in place; delete the backup only while it is
    # still provably wholly managed — content swapped in or grown between
    # the blocker check and now is preserved, not removed.
    manifest = _pack_manifest(backup)
    if (
        manifest is None
        or not _manifest_identifies(manifest, lock)
        or _first_unmanaged_entry(backup, manifest) is not None
    ):
        raise DynamicError(
            f"{target}: installed the verified pack, but the replaced directory "
            f"held unmanaged content; preserved at {backup} instead of deleting"
        )
    try:
        shutil.rmtree(backup)
    except OSError:
        pass  # leftover backup is inert; the verified pack is in place


# -- manager -------------------------------------------------------------------


def _stopped(stop: Any | None) -> bool:
    return stop is not None and stop.is_set()


def _runtime_lock(runtime: Any):
    lock = getattr(runtime, "lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def _manage_cycle(runtime: Any, stop: Any | None, cache: dict[str, Any]) -> bool:
    report = runtime.queues_report(max_age=0.0)
    if report and all(queue.get("pending_jobs") is None for queue in report):
        runtime.status.note(
            "dynamic: no queue depths in /v1/workloads — token lacks "
            "perms=dynamic; manager stopping"
        )
        return False
    source = str(runtime.config.get("model_catalog") or "")
    catalog: dict[str, dict[str, Any]] = {}
    if source:
        now = time.monotonic()
        if cache["catalog"] is None or now - cache["loaded_at"] >= cache["refresh"]:
            cache["catalog"] = catalog = load_catalog(source, stop=stop)
            cache["loaded_at"] = now
        else:
            catalog = cache["catalog"]
    if _stopped(stop):
        return False
    with _runtime_lock(runtime):
        slots = list(runtime.slots)
        provider = str(runtime.config["provider"])
        agent_labels = dict(runtime.config["labels"])
        installed_packs = set(runtime.config["model_packs"])
        fallback_dir = Path(str(runtime.config["workdir"])).expanduser() / "model-packs"
    result = plan(
        report,
        slots=slots,
        provider=provider,
        host=runtime.host,
        agent_labels=agent_labels,
        installed_packs=installed_packs,
        catalog=catalog,
        fallback_dir=fallback_dir,
        stop=stop,
    )
    for note in result.notes:
        runtime.status.note(f"dynamic: {note}")
    # Un-advertise demanded packs whose targets failed verification BEFORE
    # rebuilding: while a pack is advertised, the controller may route jobs
    # that would die on the corrupt directory mid-rebuild.
    with _runtime_lock(runtime):
        current = set(runtime.config["model_packs"])
    stale = {name for name, entry in result.packs.items() if not entry.ready} & current
    if stale:
        ok, message = runtime.update_model_packs(sorted(current - stale))
        runtime.status.note(f"dynamic: {message}")
        if _stopped(stop):
            return False
    for fetch in result.fetches:
        if _stopped(stop):
            return False
        runtime.status.note(
            f"dynamic: fetching model pack {fetch['pack']} ({fetch['reason']})"
        )
        try:
            install_pack(fetch["lock"], fetch["target"], stop=stop)
        except DynamicError as exc:  # keep provisioning other packs
            runtime.status.note(f"dynamic: model pack {fetch['pack']} failed: {exc}")
            continue
        except Exception as exc:  # classified: exception text is untrusted
            runtime.status.note(
                f"dynamic: model pack {fetch['pack']} failed: {_error_kind(exc)}"
            )
            continue
        result.packs[fetch["pack"]].verified.add(fetch["target"])
        runtime.status.note(
            f"dynamic: model pack {fetch['pack']} installed at {fetch['target']}"
        )
    if _stopped(stop):
        return False  # drained mid-cycle: no saves or registration afterwards
    with _runtime_lock(runtime):
        current = set(runtime.config["model_packs"])
    # Advertise demanded packs only when every target verifies; drop demanded
    # packs that failed or went stale; leave undemanded packs to the operator.
    desired = sorted(
        (current - set(result.packs))
        | {name for name, entry in result.packs.items() if entry.ready}
    )
    ok, message = runtime.update_model_packs(desired)
    runtime.status.note(f"dynamic: {message}")
    if _stopped(stop):
        return False
    # A failed re-registration is a bad cycle, not a reason to stop forever:
    # the next pass retries. Capabilities update best-effort in any case.
    ok, message = runtime.apply_capability_list(result.capabilities, persist=False)
    runtime.status.note(f"dynamic: {message}")
    return True


def manage_cycle(runtime: Any, stop: Any | None = None, *, _cache: dict[str, Any] | None = None) -> bool:
    """Run one report→provision→advertise pass; never raises.

    Returns False when the manager should stop (the token cannot see queue
    depths, or shutdown was requested mid-cycle). ``stop`` defaults to
    ``runtime.stop`` so a synchronous call — e.g. once before the first
    claim cycle — honours the same drain signal as the loop.
    """
    if stop is None:
        stop = getattr(runtime, "stop", None)
    if _stopped(stop):
        return False
    cache = (
        _cache
        if _cache is not None
        else {"catalog": None, "loaded_at": 0.0, "refresh": 3600.0}
    )
    try:
        return _manage_cycle(runtime, stop, cache)
    except DynamicError as exc:
        runtime.status.note(f"dynamic: {exc}")
    except Exception as exc:  # a bad cycle must not kill the manager
        runtime.status.note(f"dynamic: {type(exc).__name__}: {exc}")
    return True


def manager_loop(
    runtime: Any, stop: Any | None = None, *, catalog_refresh_seconds: float = 3600.0
) -> None:
    """Background loop behind `sparehour-worker run` when config dynamic=true."""
    if stop is None:
        stop = getattr(runtime, "stop", None)
    if stop is None:
        stop = threading.Event()  # unmanaged: poll forever like before
    interval = max(5.0, float(runtime.config["dynamic_poll_seconds"]))
    cache = {"catalog": None, "loaded_at": 0.0, "refresh": float(catalog_refresh_seconds)}
    runtime.status.note("dynamic: manager started")
    while True:
        if stop.wait(interval):
            return  # drained between cycles: no saves or registration
        if not manage_cycle(runtime, stop, _cache=cache):
            return


if __name__ == "__main__":
    # Owned JSON-probe child for service-pack readiness checks.
    sys.exit(_json_probe_child_entry())
