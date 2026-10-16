"""Idle and screen-lock detection for opportunistic computing.

A machine counts as idle only when we can *prove* nobody is interacting with
it: every relevant local session is locked, or the most recently active
session has been idle for at least ``idle_after_seconds``. One active or
unclassifiable session outvotes any number of locked/idle ones.

Headless vs. unknown:
- Confirmed headless — logind answers and reports no local interactive
  sessions (only remote/background ones or none at all) — stays idle:
  servers should contribute spare compute continuously.
- Unknown — logind unreachable, session properties unreadable, or no input
  timer on macOS/Windows — fails closed: the machine does NOT claim work.
  There is no "assume idle" fallback.

Signals by platform:
- Linux: loginctl LockedHint/IdleHint/IdleSinceHint(Monotonic) per local
  user session, plus xprintidle for X11 displays. Remote sessions and
  non-user classes (greeter, lock-screen, background, manager) are excluded.
  IdleSinceHint is CLOCK_REALTIME microseconds; IdleSinceHintMonotonic is
  CLOCK_MONOTONIC microseconds — monotonic is preferred (immune to wall
  clock steps). Formatted loginctl timestamps are rejected, not parsed.
- macOS: ioreg IOConsoleLocked + HIDIdleTime (nanoseconds).
- Windows: GetLastInputInfo/GetTickCount via ctypes with explicit unsigned
  32-bit restype and modulo-2**32 subtraction (ticks wrap every ~49.7 days);
  locked when OpenInputDesktop fails. HDESK is pointer-sized, so restype is
  c_void_p and the handle is always passed back to CloseDesktop.
"""

from __future__ import annotations

import ctypes
import os
import plistlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

_MICROSECONDS = 1_000_000.0

# logind session properties _session_detail needs. Values are matched by
# name: `loginctl show-session` prints `Key=value` lines in DBus order,
# which is NOT the order the -p flags were given in.
_SESSION_PROPS = (
    "LockedHint",
    "IdleHint",
    "IdleSinceHint",
    "IdleSinceHintMonotonic",
    "Remote",
    "Class",
)

# logind session classes that never represent a human at a local console.
# Everything else (including "" — fail closed) is treated as interactive.
_NON_USER_CLASSES = {
    "greeter",
    "lock-screen",
    "background",
    "background-light",
    "manager",
    "manager-early",
}

_DESKTOP_READOBJECTS = 0x0001


@dataclass(frozen=True)
class IdleState:
    idle: bool
    locked: bool | None  # None = unknown on this platform
    idle_seconds: float | None
    reason: str


def is_idle(idle_after_seconds: float, now: float | None = None) -> IdleState:
    """Whether the machine may take opportunistic work right now."""
    clock_now = float(now) if now is not None else time.time()
    try:
        if sys.platform.startswith("linux"):
            return _linux_state(idle_after_seconds, clock_now)
        if sys.platform == "darwin":
            return _macos_state(idle_after_seconds)
        if sys.platform == "win32":
            return _windows_state(idle_after_seconds)
    except Exception:
        pass
    return IdleState(False, None, None, "activity signal unavailable; failing closed")


def screen_locked() -> bool | None:
    """Best-effort lock probe; None when the platform cannot answer."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_locked()
        if sys.platform == "darwin":
            return _macos_locked()
        if sys.platform == "win32":
            return _windows_locked()
    except Exception:
        return None
    return None


def idle_seconds() -> float | None:
    """Seconds since the last input, or None when no signal exists."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_idle_seconds()
        if sys.platform == "darwin":
            return _macos_idle_seconds()
        if sys.platform == "win32":
            return _windows_idle_seconds()
    except Exception:
        return None
    return None


# -- shared helpers -------------------------------------------------------


def _run(command: list[str], timeout: float = 5.0) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return result.stdout if result.returncode == 0 else None


def _monotonic_us() -> int:
    """CLOCK_MONOTONIC in microseconds — the unit IdleSinceHintMonotonic uses."""
    return time.monotonic_ns() // 1000


def _duration_state(seconds: float, threshold: float, *, locked: bool | None, source: str) -> IdleState:
    if seconds >= threshold:
        return IdleState(True, locked, seconds, f"idle for {int(seconds)}s ({source})")
    return IdleState(
        False, locked, seconds, f"active (idle {int(seconds)}s < {int(threshold)}s via {source})"
    )


# -- Linux: logind sessions + xprintidle ----------------------------------


def _sessions() -> list[str] | None:
    """logind session IDs, or None when logind cannot be queried."""
    out = _run(["loginctl", "list-sessions", "--no-legend"])
    if out is None:
        return None
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def _session_detail(session: str) -> dict[str, str] | None:
    """Selected properties for one session; None when the probe fails.

    `loginctl show-session -p X -p Y` prints `Key=value` lines, but the
    order follows the DBus property table, not the -p order — matching by
    position silently assigns the wrong values. Parse the names and fail
    closed (return None) when any needed property is absent or a line is
    malformed.
    """
    command = ["loginctl", "show-session", session]
    for prop in _SESSION_PROPS:
        command.append(f"--property={prop}")
    out = _run(command)
    if out is None:
        return None
    props: dict[str, str] = {}
    for line in out.splitlines():
        if "=" not in line:
            return None
        name, _, value = line.partition("=")
        props[name] = value
    if any(prop not in props for prop in _SESSION_PROPS):
        return None
    return {prop: props[prop] for prop in _SESSION_PROPS}


def _local_sessions() -> tuple[list[str] | None, list[dict[str, str]], bool]:
    """(all session ids | None when logind fails, relevant sessions, probe_failed).

    Relevant = local (Remote != yes) interactive classes. An unreadable
    session never counts as absent — probe_failed forces fail closed.
    """
    sids = _sessions()
    if sids is None:
        return None, [], False
    relevant: list[dict[str, str]] = []
    failed = False
    for sid in sids:
        props = _session_detail(sid)
        if props is None:
            failed = True
            continue
        if props.get("Remote") == "yes" or props.get("Class") in _NON_USER_CLASSES:
            continue
        relevant.append(props)
    return sids, relevant, failed


def _parse_us(text: str | None) -> int | None:
    """Integer microseconds, or None for missing/formatted timestamps."""
    try:
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def _hint_idle_seconds(props: dict[str, str], clock_now: float) -> float | None:
    """Seconds a session has been idle, from logind's idle-since hints.

    IdleSinceHint is CLOCK_REALTIME microseconds, IdleSinceHintMonotonic is
    CLOCK_MONOTONIC microseconds — the monotonic hint is preferred because
    wall-clock steps cannot corrupt it. Unparseable hints (e.g. formatted
    `loginctl show-session` timestamps) return None instead of a bogus
    duration.
    """
    mono = _parse_us(props.get("IdleSinceHintMonotonic"))
    if mono is not None:
        return max(0.0, (_monotonic_us() - mono) / _MICROSECONDS)
    real = _parse_us(props.get("IdleSinceHint"))
    if real is not None:
        return max(0.0, clock_now - real / _MICROSECONDS)
    return None


def _session_kind(props: dict[str, str], clock_now: float) -> tuple[str, float | None]:
    """Classify one local session: ("locked" | "idle" | "active" | "unknown", seconds).

    An "active" session keeps the machine busy no matter what other sessions
    report. "unknown" means logind gave no usable hint — callers must fail
    closed rather than guess.
    """
    if props.get("LockedHint") == "yes":
        # A locked screen is conclusive; no idle duration needed.
        return "locked", None
    if props.get("IdleHint") == "no":
        return "active", 0.0
    if props.get("IdleHint") == "yes":
        duration = _hint_idle_seconds(props, clock_now)
        return ("idle", duration) if duration is not None else ("unknown", None)
    return "unknown", None


def _x11_idle_seconds() -> float | None:
    """xprintidle input timer for a reachable X display; None otherwise."""
    if not os_display() or not shutil.which("xprintidle"):
        return None
    out = _run(["xprintidle"])
    if out is not None and out.strip().isdigit():
        return int(out.strip()) / 1000.0
    return None


def _linux_state(idle_after: float, clock_now: float) -> IdleState:
    sids, relevant, probe_failed = _local_sessions()
    x11 = _x11_idle_seconds()
    if sids is None:
        # logind unreachable: only a working X11 timer can prove (in)activity.
        if x11 is None:
            return IdleState(
                False, None, None, "no activity signal (logind and X11 unavailable); failing closed"
            )
        return _duration_state(x11, idle_after, locked=None, source="X11")
    if probe_failed:
        return IdleState(
            False, None, None, "could not inspect all logind sessions; failing closed"
        )
    if not relevant:
        # Confirmed headless: logind answered and no local interactive
        # session exists — only remote/background sessions or none at all.
        if x11 is not None:
            return _duration_state(x11, idle_after, locked=None, source="X11")
        return IdleState(
            True, None, None, f"no local interactive sessions ({len(sids)} remote/system); headless"
        )
    kinds = [_session_kind(props, clock_now) for props in relevant]
    unlocked_seen = any(props.get("LockedHint") == "no" for props in relevant)
    if any(kind == "active" for kind, _ in kinds):
        return IdleState(False, False if unlocked_seen else None, 0.0, "active local session")
    if any(kind == "unknown" for kind, _ in kinds):
        return IdleState(
            False, False if unlocked_seen else None, None,
            "interactive session state unknown; failing closed",
        )
    locked_count = sum(1 for kind, _ in kinds if kind == "locked")
    if locked_count == len(relevant):
        return IdleState(True, True, None, "all local sessions locked")
    # Some sessions are unlocked but idle-hinted: the machine is idle only if
    # the *least* idle session has crossed the threshold. The X11 input
    # timer, when present, is one more local signal in the same minimum.
    durations = [d for _, d in kinds if d is not None]
    if x11 is not None:
        durations.append(x11)
    if not durations:
        return IdleState(
            False, False if unlocked_seen else None, None,
            "no usable idle durations; failing closed",
        )
    return _duration_state(min(durations), idle_after, locked=False if unlocked_seen else None,
                           source="logind")


def _linux_locked() -> bool | None:
    sids, relevant, probe_failed = _local_sessions()
    if sids is None or probe_failed or not relevant:
        return None
    if all(_session_kind(props, time.time())[0] == "locked" for props in relevant):
        return True
    return False if any(props.get("LockedHint") == "no" for props in relevant) else None


def _linux_idle_seconds() -> float | None:
    sids, relevant, probe_failed = _local_sessions()
    x11 = _x11_idle_seconds()
    if sids is None:
        return x11
    if probe_failed or not relevant:
        return x11
    kinds = [_session_kind(props, time.time()) for props in relevant]
    if any(kind == "unknown" for kind, _ in kinds):
        return None
    durations = [d for kind, d in kinds if kind != "locked" and d is not None]
    if x11 is not None:
        durations.append(x11)
    return min(durations) if durations else None


# -- macOS: console lock + HID idle time ----------------------------------


def _macos_locked() -> bool | None:
    """IOConsoleLocked from the I/O Registry root; None when unavailable."""
    out = _run(["ioreg", "-a", "-n", "Root", "-d", "1"])
    if not out:
        return None
    try:
        parsed = plistlib.loads(out.encode("utf-8"))
    except Exception:
        return None
    entries = parsed if isinstance(parsed, list) else [parsed]
    for entry in entries:
        if isinstance(entry, dict) and "IOConsoleLocked" in entry:
            return bool(entry["IOConsoleLocked"])
    return None


def _macos_idle_seconds() -> float | None:
    out = _run(["ioreg", "-c", "IOHIDSystem", "-d", "1"])
    if not out:
        return None
    for line in out.splitlines():
        if "HIDIdleTime" in line:
            try:
                nanoseconds = int("".join(char for char in line.split("=")[-1] if char.isdigit()))
                return nanoseconds / 1e9
            except ValueError:
                continue
    return None


def _macos_state(idle_after: float) -> IdleState:
    locked = _macos_locked()
    if locked:
        return IdleState(True, True, None, "console locked")
    seconds = _macos_idle_seconds()
    if seconds is None:
        return IdleState(False, locked, None, "no HID idle-time signal; failing closed")
    return _duration_state(seconds, idle_after, locked=locked, source="HIDIdleTime")


# -- Windows: last-input tick + input desktop -----------------------------


def _windows_idle_seconds() -> float | None:
    """Seconds since last input via GetLastInputInfo/GetTickCount.

    Both values are unsigned 32-bit tick counts; subtraction is modulo 2**32
    so the ~49.7-day wrap cannot yield a negative duration. restype is set
    explicitly — the ctypes default (signed c_int) would misread ticks above
    0x7FFFFFFF, which occurs after ~24.9 days of uptime.
    """

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    get_last_input = user32.GetLastInputInfo
    get_last_input.restype = ctypes.c_bool
    get_last_input.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
    get_tick_count = kernel32.GetTickCount
    get_tick_count.restype = ctypes.c_uint32
    get_tick_count.argtypes = []
    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO))
    if not get_last_input(ctypes.byref(info)):
        return None
    elapsed_ms = (int(get_tick_count()) - info.dwTime) % (1 << 32)
    return elapsed_ms / 1000.0


def _windows_locked() -> bool | None:
    """True when the input desktop cannot be opened (workstation locked).

    HDESK is pointer-sized: without restype=c_void_p a 64-bit handle
    truncates to c_int and a locked desktop could read as open. The handle
    is always returned to CloseDesktop so probes never leak handles.
    """
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        open_desktop = user32.OpenInputDesktop
        open_desktop.restype = ctypes.c_void_p
        open_desktop.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        close_desktop = user32.CloseDesktop
        close_desktop.restype = ctypes.c_bool
        close_desktop.argtypes = [ctypes.c_void_p]
        desktop = open_desktop(0, False, _DESKTOP_READOBJECTS)
        if not desktop:
            return True
        close_desktop(desktop)
        return False
    except Exception:
        return None


def _windows_state(idle_after: float) -> IdleState:
    locked = _windows_locked()
    if locked:
        return IdleState(True, True, None, "input desktop inaccessible (locked)")
    seconds = _windows_idle_seconds()
    if seconds is None:
        return IdleState(False, locked, None, "GetLastInputInfo unavailable; failing closed")
    return _duration_state(seconds, idle_after, locked=locked, source="GetLastInputInfo")


def os_display() -> str | None:
    return os.environ.get("DISPLAY")
