"""Best-effort idle and screen-lock detection for opportunistic computing.

A machine counts as idle when the screen is locked, or when no input has
happened for `idle_after_seconds`. Headless machines (no local sessions) are
always idle — servers should contribute spare compute continuously.

Signals by platform:
- Linux: loginctl LockedHint/IdleHint per session, xprintidle under X11.
- macOS: ioreg HIDIdleTime (idle only; lock state is not probed).
- Windows: GetLastInputInfo via ctypes; locked when the input desktop is gone.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class IdleState:
    idle: bool
    locked: bool | None  # None = unknown on this platform
    idle_seconds: float | None
    reason: str


def is_idle(idle_after_seconds: float, now: float | None = None) -> IdleState:
    locked = screen_locked()
    if locked:
        return IdleState(True, True, None, "screen locked")
    seconds = idle_seconds()
    if seconds is None:
        return IdleState(True, None, None, "no local activity signal (headless?) — treating as idle")
    if seconds >= idle_after_seconds:
        return IdleState(True, False, seconds, f"idle for {int(seconds)}s")
    return IdleState(False, False, seconds, f"active (idle {int(seconds)}s < {int(idle_after_seconds)}s)")


def screen_locked() -> bool | None:
    try:
        if sys.platform.startswith("linux"):
            return _linux_locked()
        if sys.platform == "win32":
            return _windows_locked()
    except Exception:
        return None
    return None  # macOS: lock probing not worth the fragility; idle time covers it


def idle_seconds() -> float | None:
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


def _run(command: list[str], timeout: float = 5.0) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return result.stdout if result.returncode == 0 else None


def _sessions() -> list[str]:
    out = _run(["loginctl", "list-sessions", "--no-legend"])
    if out is None:
        return []
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def _session_property(session: str, prop: str) -> str | None:
    out = _run(["loginctl", "show-session", session, f"--property={prop}", "--value"])
    return out.strip() if out else None


def _linux_locked() -> bool | None:
    sessions = _sessions()
    if not sessions:
        return None
    hints = [_session_property(sid, "LockedHint") for sid in sessions]
    values = {hint for hint in hints if hint is not None}
    if not values:
        return None
    return "yes" in values


def _linux_idle_seconds() -> float | None:
    # X11: xprintidle is per-display and the most reliable input timer.
    if shutil.which("xprintidle") and os_display():
        out = _run(["xprintidle"])
        if out is not None and out.strip().isdigit():
            return int(out.strip()) / 1000.0
    # logind fallback: sessions that logind already marked idle.
    best: float | None = None
    for sid in _sessions():
        if _session_property(sid, "IdleHint") == "yes":
            since = _session_property(sid, "IdleSinceHint")
            if since:
                try:
                    best = max(best or 0.0, time.time() - float(since))
                except ValueError:
                    continue
    return best


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


def _windows_idle_seconds() -> float | None:
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO))
    if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
        return (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0  # type: ignore[attr-defined]
    return None


def _windows_locked() -> bool | None:
    # The input desktop is inaccessible while the workstation is locked.
    try:
        desktop = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0001)  # type: ignore[attr-defined]
        if desktop:
            ctypes.windll.user32.CloseDesktop(desktop)  # type: ignore[attr-defined]
            return False
        return True
    except Exception:
        return None


def os_display() -> str | None:
    return os.environ.get("DISPLAY")
