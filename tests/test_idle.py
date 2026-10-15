"""Idle detector regressions: session veto, timestamps, fail-closed behavior.

All probes are stubbed at the module seams (_sessions/_session_detail/_run/
windll/os_display) so the platform logic is exercised without a display,
logind, or Windows host.
"""

from __future__ import annotations

import ctypes
import plistlib
import sys
import time
from types import SimpleNamespace

import pytest

from dworker import idle


def _props(**overrides) -> dict[str, str]:
    """A local user session property dict; override individual logind fields."""
    props = {name: "" for name in idle._SESSION_PROPS}
    props.update({"Remote": "no", "Class": "user"})
    props.update({name: str(value) for name, value in overrides.items()})
    return props


# The order `loginctl show-session` actually emits properties in: the DBus
# vtable order, deliberately NOT the -p request order. Emitting real
# Key=value lines in this order exercises the parser; emitting request
# order would hide a positional (zip) parsing bug.
_DBUS_ORDER = (
    "Remote",
    "Class",
    "IdleHint",
    "IdleSinceHint",
    "IdleSinceHintMonotonic",
    "LockedHint",
)


def _show_session_output(props: dict[str, str] | str) -> str:
    if isinstance(props, str):
        return props  # raw output for malformed/partial-output tests
    return "".join(f"{name}={props[name]}\n" for name in _DBUS_ORDER if name in props)


def _patch_linux(monkeypatch, sessions: dict[str, dict | str | None], *, sids="auto"):
    """Stub logind at the _run seam with DBus-order Key=value output.

    sids=None simulates an unreachable logind. X11 is dark (no DISPLAY).
    """
    monkeypatch.setattr(sys, "platform", "linux")
    session_ids = list(sessions) if sids == "auto" else sids

    def fake_run(cmd, timeout=5.0):
        if cmd[0] != "loginctl":
            return None
        if "list-sessions" in cmd:
            if session_ids is None:
                return None
            return "".join(f"{sid} user seat0\n" for sid in session_ids)
        if "show-session" in cmd:
            props = sessions.get(cmd[2])
            return None if props is None else _show_session_output(props)
        return None

    monkeypatch.setattr(idle, "_run", fake_run)
    monkeypatch.setattr(idle, "os_display", lambda: None)
    monkeypatch.setattr(idle.shutil, "which", lambda name: f"/usr/bin/{name}")


def _idle_since_monotonic(seconds_ago: float) -> str:
    return str(idle._monotonic_us() - int(seconds_ago * 1_000_000))


def _idle_since_realtime(seconds_ago: float) -> str:
    return str(int((time.time() - seconds_ago) * 1_000_000))


# -- multi-session decisions ----------------------------------------------


def test_active_session_vetoes_locked_and_idle_sessions(monkeypatch):
    _patch_linux(
        monkeypatch,
        {
            "1": _props(LockedHint="yes"),  # locked VT
            "2": _props(IdleHint="yes", IdleSinceHintMonotonic=_idle_since_monotonic(7200)),
            "3": _props(IdleHint="no"),  # actively in use
        },
    )
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds == 0.0
    assert "active" in state.reason


def test_locked_session_does_not_veto_active_one(monkeypatch):
    # Regression: previously any locked session authorized idle.
    _patch_linux(
        monkeypatch,
        {"1": _props(LockedHint="yes"), "2": _props(IdleHint="no")},
    )
    assert not idle.is_idle(300).idle


def test_all_local_sessions_locked_is_idle(monkeypatch):
    _patch_linux(
        monkeypatch,
        {"1": _props(LockedHint="yes"), "2": _props(LockedHint="yes")},
    )
    state = idle.is_idle(300)
    assert state.idle
    assert state.locked is True
    assert "locked" in state.reason


def test_idle_threshold_uses_least_idle_session(monkeypatch):
    # One session idle for 4000s, another for only 120s: the machine is NOT
    # idle at a 300s threshold because someone was active 2 minutes ago.
    _patch_linux(
        monkeypatch,
        {
            "1": _props(IdleHint="yes", IdleSinceHintMonotonic=_idle_since_monotonic(4000)),
            "2": _props(IdleHint="yes", IdleSinceHintMonotonic=_idle_since_monotonic(120)),
        },
    )
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds == pytest.approx(120, abs=5)

    state = idle.is_idle(100)
    assert state.idle
    assert state.idle_seconds == pytest.approx(120, abs=5)


def test_locked_and_idle_sessions_combine(monkeypatch):
    # Locked VT plus an idle-hinted session past the threshold -> idle.
    _patch_linux(
        monkeypatch,
        {
            "1": _props(LockedHint="yes"),
            "2": _props(IdleHint="yes", IdleSinceHintMonotonic=_idle_since_monotonic(600)),
        },
    )
    state = idle.is_idle(300)
    assert state.idle
    assert state.idle_seconds == pytest.approx(600, abs=5)


# -- logind timestamp handling ---------------------------------------------


def test_idle_since_hint_is_microseconds(monkeypatch):
    # IdleSinceHint is CLOCK_REALTIME *microseconds*; dividing wrongly (or
    # subtracting seconds from microseconds) must not pass.
    _patch_linux(
        monkeypatch,
        {"1": _props(IdleHint="yes", IdleSinceHint=_idle_since_realtime(600))},
    )
    state = idle.is_idle(300, now=time.time())
    assert state.idle
    assert state.idle_seconds == pytest.approx(600, abs=5)


def test_monotonic_hint_preferred_over_realtime(monkeypatch):
    # A bogus realtime hint (epoch 0) must not inflate the idle duration
    # when the trustworthy monotonic hint is present.
    _patch_linux(
        monkeypatch,
        {
            "1": _props(
                IdleHint="yes",
                IdleSinceHint="1",
                IdleSinceHintMonotonic=_idle_since_monotonic(10),
            )
        },
    )
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds == pytest.approx(10, abs=5)


def test_formatted_idle_hint_fails_closed(monkeypatch):
    # `loginctl show-session` (without --value) style timestamps are not
    # numeric; treating them as 0-idle would hide an active user.
    _patch_linux(
        monkeypatch,
        {"1": _props(IdleHint="yes", IdleSinceHint="Mon 2026-10-12 08:15:00 UTC")},
    )
    state = idle.is_idle(300)
    assert not state.idle
    assert "unknown" in state.reason


# -- show-session property parsing -------------------------------------------


def test_session_properties_match_by_name_not_position(monkeypatch):
    # loginctl emits extra properties and in an order unrelated to the -p
    # flags; matching positionally misassigns every value (e.g. IdleHint
    # receives the Class string). Parse by name and read a real duration.
    _patch_linux(
        monkeypatch,
        {
            "1": (
                "Name=user\n"
                "IdleSinceHintMonotonic="
                + _idle_since_monotonic(600)
                + "\n"
                "LockedHint=no\n"
                "Type=x11\n"
                "IdleHint=yes\n"
                "Remote=no\n"
                "IdleSinceHint=\n"
                "Class=user\n"
            )
        },
    )
    state = idle.is_idle(300)
    assert state.idle
    assert state.idle_seconds == pytest.approx(600, abs=5)


def test_missing_session_property_fails_closed(monkeypatch):
    # An incomplete property set must fail closed, never guess that the
    # session is inactive.
    _patch_linux(monkeypatch, {"1": "Remote=no\nClass=user\nIdleHint=yes\n"})
    state = idle.is_idle(300)
    assert not state.idle
    assert "inspect" in state.reason


def test_malformed_session_line_fails_closed(monkeypatch):
    _patch_linux(
        monkeypatch,
        {
            "1": (
                "Remote=no\nClass=user\nIdleHint=no\n"
                "IdleSinceHintMonotonic=0\nLockedHint=no\n"
                "corrupt line without separator\n"
                "IdleSinceHint=0\n"
            )
        },
    )
    state = idle.is_idle(300)
    assert not state.idle
    assert "inspect" in state.reason


# -- unknown vs. confirmed headless ----------------------------------------


def test_session_probe_failure_fails_closed(monkeypatch):
    # logind lists a session but properties are unreadable: it may be an
    # active user, so the machine must NOT be claimed as idle.
    _patch_linux(monkeypatch, {"1": None})
    state = idle.is_idle(300)
    assert not state.idle
    assert "inspect" in state.reason


def test_logind_unavailable_fails_closed(monkeypatch):
    _patch_linux(monkeypatch, {}, sids=None)
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds is None
    assert "failing closed" in state.reason


def test_confirmed_headless_stays_idle(monkeypatch):
    # logind answered with zero sessions: genuinely headless -> idle.
    _patch_linux(monkeypatch, {})
    state = idle.is_idle(300)
    assert state.idle
    assert "headless" in state.reason


def test_remote_and_nonuser_sessions_do_not_gate(monkeypatch):
    # A remote SSH session (even "active") and a greeter session are not
    # local interactive use; with nothing else, the machine is headless.
    _patch_linux(
        monkeypatch,
        {
            "ssh": _props(Remote="yes", IdleHint="no"),
            "gdm": _props(Class="greeter"),
            "bg": _props(Class="background"),
        },
    )
    state = idle.is_idle(300)
    assert state.idle
    assert "headless" in state.reason


def test_unsupported_platform_fails_closed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "sunos5")
    state = idle.is_idle(300)
    assert not state.idle
    assert "failing closed" in state.reason


# -- X11 signal -------------------------------------------------------------


def test_x11_recent_input_blocks_idle(monkeypatch):
    _patch_linux(monkeypatch, {})
    loginctl_run = idle._run
    monkeypatch.setattr(idle, "os_display", lambda: ":0")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] == "xprintidle":
            return "5000\n"
        return loginctl_run(cmd, timeout)

    monkeypatch.setattr(idle, "_run", fake_run)
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds == pytest.approx(5.0)


def test_x11_idle_past_threshold_allows_idle_without_sessions(monkeypatch):
    _patch_linux(monkeypatch, {})
    loginctl_run = idle._run
    monkeypatch.setattr(idle, "os_display", lambda: ":0")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] == "xprintidle":
            return "600000\n"
        return loginctl_run(cmd, timeout)

    monkeypatch.setattr(idle, "_run", fake_run)
    state = idle.is_idle(300)
    assert state.idle
    assert state.idle_seconds == pytest.approx(600.0)


# -- macOS ------------------------------------------------------------------


def _patch_macos(monkeypatch, *, locked=None, hid_seconds=None):
    """Stub ioreg: plist for the Root console query, text for IOHIDSystem."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] != "ioreg":
            return None
        if "-a" in cmd:
            if locked is None:
                return None
            return plistlib.dumps({"IOConsoleLocked": locked}).decode()
        if hid_seconds is None:
            return None
        return f'"HIDIdleTime" = {int(hid_seconds * 1e9)}\n'

    monkeypatch.setattr(idle, "_run", fake_run)


def test_macos_locked_console_is_idle(monkeypatch):
    _patch_macos(monkeypatch, locked=True, hid_seconds=10)
    state = idle.is_idle(300)
    assert state.idle
    assert state.locked is True


def test_macos_uses_hid_idle_time(monkeypatch):
    _patch_macos(monkeypatch, locked=False, hid_seconds=10)
    assert not idle.is_idle(300).idle
    _patch_macos(monkeypatch, locked=False, hid_seconds=400)
    state = idle.is_idle(300)
    assert state.idle
    assert state.idle_seconds == pytest.approx(400)


def test_macos_missing_signal_fails_closed(monkeypatch):
    _patch_macos(monkeypatch, locked=None, hid_seconds=None)
    state = idle.is_idle(300)
    assert not state.idle
    assert "failing closed" in state.reason


# -- Windows -----------------------------------------------------------------


def _stub_win32(monkeypatch, *, tick, last_input=None, desktop=1):
    """Fake ctypes.windll with function objects that accept restype/argtypes."""
    monkeypatch.setattr(sys, "platform", "win32")
    calls: dict[str, list] = {"closed": []}

    def GetLastInputInfo(buf):
        buf._obj.dwTime = last_input
        return 1 if last_input is not None else 0

    def GetTickCount():
        return tick

    def OpenInputDesktop(flags, inherit, access):
        calls["open_args"] = (flags, inherit, access)
        return desktop

    def CloseDesktop(handle):
        calls["closed"].append(handle)
        return True

    funcs = SimpleNamespace(
        GetLastInputInfo=GetLastInputInfo,
        GetTickCount=GetTickCount,
        OpenInputDesktop=OpenInputDesktop,
        CloseDesktop=CloseDesktop,
    )
    windll = SimpleNamespace(user32=funcs, kernel32=funcs)
    monkeypatch.setattr(idle.ctypes, "windll", windll, raising=False)
    return calls, funcs


def test_windows_tick_wraps_unsigned(monkeypatch):
    # Tick wrapped past 0xFFFFFFFF: 50 - 0xFFFFFF00 must be +306ms, never
    # negative or a huge bogus duration.
    _stub_win32(monkeypatch, tick=50, last_input=0xFFFFFF00, desktop=1)
    assert idle._windows_idle_seconds() == pytest.approx(0.306)


def test_windows_tick_count_declared_unsigned(monkeypatch):
    # Above 0x7FFFFFFF a signed restype would read the tick as negative;
    # the function must be declared c_uint32.
    _, funcs = _stub_win32(monkeypatch, tick=0x80000010, last_input=0x80000000)
    assert idle._windows_idle_seconds() == pytest.approx(0.016)
    assert funcs.GetTickCount.restype is ctypes.c_uint32


def test_windows_desktop_handles_are_pointer_sized_and_closed(monkeypatch):
    big_handle = 0x00000001_00000001  # truncated by a c_int restype
    calls, funcs = _stub_win32(monkeypatch, tick=0, last_input=0, desktop=big_handle)
    assert idle._windows_locked() is False
    assert calls["closed"] == [big_handle]  # the handle was actually closed
    assert funcs.OpenInputDesktop.restype is ctypes.c_void_p
    assert funcs.CloseDesktop.argtypes == [ctypes.c_void_p]


def test_windows_inaccessible_desktop_means_locked(monkeypatch):
    _stub_win32(monkeypatch, tick=0, last_input=0, desktop=None)
    state = idle.is_idle(300)
    assert state.idle
    assert state.locked is True


def test_windows_missing_input_info_fails_closed(monkeypatch):
    _stub_win32(monkeypatch, tick=0, last_input=None, desktop=1)
    state = idle.is_idle(300)
    assert not state.idle
    assert state.idle_seconds is None
    assert "failing closed" in state.reason
