"""CLI lifecycle regression tests — offline, deterministic fakes.

The real ``WorkerRuntime`` and ``IdleGatedRunner`` run here; only the outer
boundaries are stubbed: hardware probes (``discover_slots``/``host_resources``)
and the control-plane client transport. Tests therefore exercise the actual
CLI wiring — overrides applied before discovery, dynamic provisioning before
the first claim, drain propagation, join ordering, and config persistence —
and assert consumer-visible outcomes, never error-message wording.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sparehour.agent import AgentError
from sparehour.models import SlotInventory
from sparehour_workloads.argos_pack import build_model_pack_from_lock

from sparehour_worker import cli
from sparehour_worker import config as config_mod
from sparehour_worker import dynamic as dyn
from sparehour_worker import runner as runner_mod

TOKEN = "test-token-0123456789abcdef"
ENDPOINT = "https://controller.example.com"


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    config = dict(config_mod.DEFAULTS)
    config.update(
        endpoint=ENDPOINT,
        token=TOKEN,
        workdir=str(tmp_path / "work"),
        dashboard=False,
    )
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _cpu_slot(index: int, capabilities: list[str], model_packs: list[str]) -> SlotInventory:
    return SlotInventory(
        id=f"cpu-{index}",
        gpu_index=index,
        gpu_uuid=f"cpu-{index}",
        vendor="cpu",
        model="cpu (x86_64)",
        vram_mb=16384,
        capabilities=list(capabilities),
        model_packs=list(model_packs),
    )


class _Control:
    """Per-test knobs for the fake control plane."""

    def __init__(self) -> None:
        self.workloads: list[dict] = []
        self.jobs: list[dict] = []
        self.http_error: BaseException | None = None
        self.claim_error: AgentError | None = None
        self.on_claim: Any = None
        self.clients: list[FakeClient] = []
        self.events: list[tuple] = []


class FakeHTTP:
    """Stand-in for the httpx.Client inside WorkerClient.

    Serves /healthz and /v1/workloads out of the control object.
    """

    def __init__(self, state: _Control):
        self.state = state

    def get(self, path: str):
        if self.state.http_error is not None:
            raise self.state.http_error
        if path == "/v1/workloads":
            payload = self.state.workloads
        else:
            payload = {"workloads": len(self.state.workloads)}
        return SimpleNamespace(json=lambda: payload, status_code=200, text="")


class FakeClient:
    """WorkerClient stand-in recording lifecycle calls and registrations."""

    def __init__(self, endpoint: str, token: str, *, state: _Control, **_kw):
        self.endpoint = endpoint
        self.state = state
        self.client = FakeHTTP(state)
        self.registrations: list[Any] = []
        self.heartbeats: list[Any] = []
        self.completed: list[str] = []
        self.closed = False
        state.clients.append(self)

    def register(self, registration: Any) -> dict:
        self.registrations.append(registration)
        return {}

    def heartbeat_agent(self, agent_id: str, *, draining=None) -> None:
        self.heartbeats.append(draining)

    def heartbeat_job(self, *args: Any, **kw: Any) -> bool:
        return False

    def claim(self, slot_id: str, lease_seconds: int):
        if self.state.claim_error is not None:
            raise self.state.claim_error
        self.state.events.append(("claim", slot_id))
        if self.state.on_claim is not None:
            self.state.on_claim(slot_id)
        return self.state.jobs.pop(0) if self.state.jobs else None

    def confirm(self, *args: Any, **kw: Any) -> None:
        pass

    def complete(self, job_id: str, *args: Any, **kw: Any) -> None:
        self.state.events.append(("complete", job_id))
        self.completed.append(job_id)

    def fail(self, job_id: str, slot_id: str, token: str, code: str, message: str) -> None:
        self.state.events.append(("fail", job_id, code))

    def upload_output(self, *args: Any, **kw: Any) -> dict:
        raise AgentError("unexpected artifact upload")

    def release(self, *args: Any, **kw: Any) -> None:
        pass

    def close(self) -> None:
        self.state.events.append(("client-closed",))
        self.closed = True


def _fake_discover(agent_id, *, capabilities, model_packs, labels, cpu_slots):
    """Honors the real discovery contract: slots reflect the *current*
    cpu_slots/capabilities/model_packs values, so tests verify the CLI applies
    overrides before the runtime is built."""
    slots = [
        _cpu_slot(i, capabilities, model_packs) for i in range(int(cpu_slots))
    ]
    return slots, ["fake probe"]


class RecordingRuntime(cli.WorkerRuntime):
    """Real WorkerRuntime with recorded instances and a closed flag."""

    instances: list["RecordingRuntime"] = []

    def __init__(self, *args: Any, **kw: Any):
        self.closed = False
        RecordingRuntime.instances.append(self)
        super().__init__(*args, **kw)

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.fixture(autouse=True)
def cp(monkeypatch):
    """Sandbox the control plane and hardware probes around the real runtime."""
    state = _Control()
    monkeypatch.setattr(cli, "WorkerRuntime", RecordingRuntime)
    monkeypatch.setattr(
        runner_mod,
        "WorkerClient",
        lambda *a, **kw: FakeClient(*a, state=state, **kw),
    )
    monkeypatch.setattr(runner_mod.hardware, "discover_slots", _fake_discover)
    monkeypatch.setattr(
        runner_mod.hardware,
        "host_resources",
        lambda workdir=None: {
            "cpu_cores": 8,
            "ram_mb": 16384,
            "disk_free_mb": 100000,
        },
    )
    RecordingRuntime.instances = []
    return state


def _last_runtime() -> RecordingRuntime:
    return RecordingRuntime.instances[-1]


def _run_args(config_path: Path, *extra: str) -> list[str]:
    return ["--config", str(config_path), "run", "--once", "--no-dashboard", *extra]


# -- run: CPU override ------------------------------------------------------


def test_cpu_slots_override_creates_slots_and_does_not_persist(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=0)
    assert cli.main(_run_args(config_path, "--cpu-slots", "2")) == 0

    client = cp.clients[-1]
    registration = client.registrations[-1]
    assert len(registration.slots) == 2
    assert all(slot.vendor == "cpu" for slot in registration.slots)
    assert ("claim", "cpu-0") in cp.events
    assert ("claim", "cpu-1") in cp.events
    # The override is runtime-only; the saved config keeps cpu_slots=0.
    assert json.loads(config_path.read_text())["cpu_slots"] == 0
    assert _last_runtime().closed and client.closed


def test_cpu_slots_zero_override_removes_saved_slots(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=3)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path, "--cpu-slots", "0"))
    assert exc.value.code == 1
    runtime = _last_runtime()
    assert runtime.slots == []  # override applied before discovery
    assert runtime.closed       # and the client was still cleaned up
    assert cp.clients[-1].closed


def test_no_slots_fails_cleanly_and_closes_client(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=0)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    runtime = _last_runtime()
    assert runtime.closed and runtime.client.closed
    assert runtime.client.registrations == []  # nothing reached the controller


def test_negative_cpu_slots_rejected_before_runtime(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=1)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path, "--cpu-slots=-2"))
    assert exc.value.code == 1
    assert RecordingRuntime.instances == []  # never constructed
    assert cp.clients == []                  # no credentials touched the wire


# -- run: error paths -------------------------------------------------------


def test_claim_failure_fails_sanitized_and_closes(tmp_path, capsys, cp):
    cp.claim_error = AgentError(f"boom token={TOKEN}")
    config_path = _write_config(tmp_path, cpu_slots=1)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.err
    assert "Traceback" not in captured.err
    runtime = _last_runtime()
    assert runtime.closed and runtime.client.closed


def test_run_without_credentials_fails(tmp_path, cp):
    config_path = _write_config(tmp_path, token="")
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    assert RecordingRuntime.instances == []
    assert cp.clients == []


def test_queues_error_fails_sanitized_and_closes(tmp_path, capsys, cp):
    cp.http_error = RuntimeError(f"listing blew up {TOKEN}")
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(config_path), "queues"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.err
    runtime = _last_runtime()
    assert runtime.closed and runtime.client.closed


def test_runtime_partial_construction_closes_client(tmp_path, capsys, monkeypatch, cp):
    """A failure after the control-plane client exists must still release it."""

    def broken_discover(*a, **kw):
        raise RuntimeError(f"probe blew up {TOKEN}")

    monkeypatch.setattr(runner_mod.hardware, "discover_slots", broken_discover)
    config_path = _write_config(tmp_path, cpu_slots=1)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    assert TOKEN not in capsys.readouterr().err
    assert cp.clients[-1].closed  # created, then released by startup cleanup


# -- doctor ------------------------------------------------------------------


def test_doctor_unreachable_is_sanitized_nonzero(tmp_path, capsys, cp):
    cp.http_error = httpx.ConnectError(f"connection refused for {TOKEN}")
    config_path = _write_config(tmp_path)
    assert cli.main(["--config", str(config_path), "doctor"]) == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err
    assert _last_runtime().closed


# -- run: dynamic mode -------------------------------------------------------


def _workload(*, capability="translate-batch", pending=1, packs=(), environment=None):
    """One workload in the /v1/workloads wire shape."""
    return {
        "name": "translate",
        "version": "1",
        "routes": {
            "r1": {
                "compatibility": {
                    "capability": capability,
                    "model_packs": list(packs),
                },
                "resources": {"vendor": "cpu"},
                "executor": {
                    "type": "command",
                    "command": ["echo", "work"],
                    "environment": dict(environment or {}),
                },
                "pending_jobs": pending,
            }
        },
    }


def _pack(tmp_path: Path, name="argos-en-es-1.0", root="en_es",
          from_code="en", to_code="es"):
    """A real .argosmodel archive plus its lock entry (sha256 computed)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            f"{root}/metadata.json",
            json.dumps({"from_code": from_code, "to_code": to_code}),
        )
        bundle.writestr(f"{root}/sentencepiece.model", b"sentencepiece bytes")
        bundle.writestr(f"{root}/model/model.bin", b"weights")
    payload = buffer.getvalue()
    archive = tmp_path / f"{name}.argosmodel"
    archive.write_bytes(payload)
    lock = {
        "model_pack": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": f"https://packs.invalid/{name}.argosmodel",
        "archive_root": root,
        "from_code": from_code,
        "to_code": to_code,
    }
    return lock, archive


def _pack_environment(lock: dict, target: Path) -> dict[str, str]:
    return {
        "ARGOS_MODEL_DIR": str(target),
        "ARGOS_MODEL_ARCHIVE_SHA256": lock["sha256"],
        "ARGOS_MODEL_PACK": lock["model_pack"],
        "ARGOS_SOURCE_LANGUAGE": lock["from_code"],
        "ARGOS_TARGET_LANGUAGE": lock["to_code"],
    }


def _command_job(job_id="job-1", delay=0.0):
    """A real command-executor job writing an empty output manifest."""
    script = (
        "import os, time\n"
        f"time.sleep({delay!r})\n"
        "open(os.environ['SPAREHOUR_OUTPUT_MANIFEST'], 'w').write('{\"files\": []}')\n"
    )
    return {
        "id": job_id,
        "lease_token": "tok",
        "workload": "translate",
        "route": "r1",
        "payload": {},
        "executor": {
            "type": "command",
            "command": [sys.executable, "-c", script],
        },
        "output_contract": [],
    }


@pytest.mark.parametrize("saved_capabilities", [["translate-batch"], []])
def test_dynamic_once_registers_provisioned_inventory_and_keeps_overrides(
    tmp_path, cp, saved_capabilities
):
    """--once dynamic run against the real manager cycle.

    The demanded route pins a pre-installed pack; the synchronous cycle must
    advertise the pack + capability and register the result, the runner's own
    registration must carry the *provisioned* inventory (never the stale
    pre-cycle snapshot), the demanded job must actually execute, and the saved
    config must gain the pack without gaining runtime-only CLI overrides.
    """
    lock, archive = _pack(tmp_path)
    target = tmp_path / "packs" / lock["model_pack"]
    build_model_pack_from_lock(lock, target, archive_path=archive)
    dyn.verify_pack_dir(target, lock)  # fixture sanity: really installed
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"models": [lock]}), encoding="utf-8")

    cp.workloads = [
        _workload(packs=[lock["model_pack"]], environment=_pack_environment(lock, target))
    ]
    cp.jobs = [_command_job()]
    config_path = _write_config(
        tmp_path,
        cpu_slots=0,
        dynamic=True,
        model_catalog=str(catalog),
        capabilities=saved_capabilities,
        model_packs=[],
    )
    argv = _run_args(config_path, "--cpu-slots", "2", "--idle-after", "60")
    assert cli.main(argv) == 0

    client = cp.clients[-1]
    final = client.registrations[-1]
    # The last registration advertises what the provisioning cycle produced:
    # the demanded pack and its capability on every slot.
    assert all(slot.model_packs == [lock["model_pack"]] for slot in final.slots)
    assert all(slot.capabilities == ["translate-batch"] for slot in final.slots)
    assert len(final.slots) == 2
    # And the claimed job ran the real executor to completion.
    assert client.completed == ["job-1"]

    saved = json.loads(config_path.read_text())
    assert saved["model_packs"] == [lock["model_pack"]]  # provisioning persisted
    assert saved["capabilities"] == saved_capabilities   # dynamic picks stay runtime-only
    assert saved["cpu_slots"] == 0                       # --cpu-slots never persisted
    assert saved["idle_after_seconds"] == 300            # nor --idle-after
    assert _last_runtime().closed and client.closed


def test_dynamic_requires_queue_depth_permission(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=1, dynamic=True)
    cp.workloads = [_workload(pending=None)]
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    client = cp.clients[-1]
    assert _last_runtime().closed and client.closed
    assert client.registrations == []  # never advertised without permission


def test_dynamic_requires_visible_workloads(tmp_path, cp):
    config_path = _write_config(tmp_path, cpu_slots=1, dynamic=True)
    with pytest.raises(SystemExit) as exc:
        cli.main(_run_args(config_path))
    assert exc.value.code == 1
    client = cp.clients[-1]
    assert _last_runtime().closed and client.closed
    assert client.registrations == []


def test_sigterm_drain_cancels_manager_while_job_finishes(tmp_path, monkeypatch, cp):
    """runner.request_drain (SIGTERM) must propagate to runtime.stop mid-run:
    the dynamic manager exits promptly, the already-running job still
    completes, and the client closes only after everything else."""
    events = cp.events

    def fake_manager(runtime, stop, **kw):
        events.append(("manager-start",))
        stop.wait(30)  # wakes only when the shared cancellation event is set
        events.append(("manager-stopped",))

    monkeypatch.setattr(dyn, "manager_loop", fake_manager)

    armed = threading.Event()

    def send_sigterm(_slot_id):
        # Inside the first claim: the drain lands while the job still runs.
        if not armed.is_set():
            armed.set()
            threading.Timer(
                0.05, lambda: os.kill(os.getpid(), signal.SIGTERM)
            ).start()

    cp.on_claim = send_sigterm
    cp.workloads = [_workload()]
    cp.jobs = [_command_job(delay=1.0)]
    config_path = _write_config(tmp_path, cpu_slots=1, dynamic=True)
    assert cli.main(_run_args(config_path)) == 0

    assert True in cp.clients[-1].heartbeats  # drain heartbeat reached the client
    # The manager observed the drain *while* the job was still running —
    # the shared stop event — not only after runner.run() returned.
    assert events.index(("manager-stopped",)) < events.index(("complete", "job-1"))
    assert events.index(("complete", "job-1")) < events.index(("client-closed",))
    assert _last_runtime().closed


def test_shutdown_joins_dynamic_manager_before_closing_client(tmp_path, monkeypatch, cp):
    """The dynamic manager thread is joined completely before client.close —
    even a manager that lingers in teardown must be finished first."""
    events = cp.events
    teardown_done = threading.Event()

    def fake_manager(runtime, stop, **kw):
        stop.wait(30)
        events.append(("manager-teardown",))
        time.sleep(0.25)  # linger past any warn-and-continue join bound
        teardown_done.set()
        events.append(("manager-done",))

    monkeypatch.setattr(dyn, "manager_loop", fake_manager)
    cp.workloads = [_workload()]
    config_path = _write_config(tmp_path, cpu_slots=1, dynamic=True)
    assert cli.main(_run_args(config_path)) == 0
    assert teardown_done.is_set()
    assert events.index(("manager-done",)) < events.index(("client-closed",))
    assert _last_runtime().closed


# -- run: dashboard ----------------------------------------------------------


class FakeServer:
    def __init__(self, url: str = "http://127.0.0.1:7788/#dash-secret-token"):
        self.dashboard_url = url
        self.shutdown_called = threading.Event()
        self.closed = False
        self.threads: list[threading.Thread] = []

    def serve_forever(self):
        self.shutdown_called.wait(30)

    def shutdown(self):
        self.shutdown_called.set()

    def server_close(self):
        self.closed = True


def _patch_dashboard(monkeypatch, server_or_error):
    import sparehour_worker.dashboard as dashboard_mod

    def make_server(**kw):
        if isinstance(server_or_error, BaseException):
            raise server_or_error
        return server_or_error

    monkeypatch.setattr(dashboard_mod, "make_server", make_server)


def test_dashboard_bind_failure_still_runs_compute(tmp_path, monkeypatch, capsys, cp):
    config_path = _write_config(tmp_path, cpu_slots=1, dashboard=True, dashboard_port=7788)
    _patch_dashboard(monkeypatch, OSError(98, "Address already in use"))
    argv = [a for a in _run_args(config_path) if a != "--no-dashboard"]
    assert cli.main(argv) == 0
    assert capsys.readouterr().err  # the bind failure is reported somewhere
    runtime = _last_runtime()
    assert ("claim", "cpu-0") in cp.events  # compute ran regardless
    assert runtime.closed


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.0.2.1", "dashboard.example.com"]
)
def test_unsafe_dashboard_host_rejected_before_any_work(
    tmp_path, monkeypatch, capsys, cp, host
):
    """A non-loopback dashboard_host must die in config validation: no
    runtime, no client, no bind attempt, no token leak, no config rewrite."""
    config_path = _write_config(
        tmp_path, cpu_slots=1, dashboard=True, dashboard_host=host
    )
    before = config_path.read_bytes()

    calls = []

    import sparehour_worker.dashboard as dashboard_mod

    monkeypatch.setattr(
        dashboard_mod, "make_server", lambda **kw: calls.append(kw)
    )
    argv = [a for a in _run_args(config_path) if a != "--no-dashboard"]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err   # no secret disclosure
    assert config_path.read_bytes() == before         # config untouched
    assert RecordingRuntime.instances == []           # rejected before runtime
    assert cp.clients == []                           # no control-plane client
    assert calls == []                                # and no bind attempted


def test_dashboard_security_rejection_is_fatal(tmp_path, monkeypatch, capsys, cp):
    """Config validation passes but the server rejects the security config:
    the run must abort cleanly, never continue exposed."""
    config_path = _write_config(tmp_path, cpu_slots=1, dashboard=True)
    _patch_dashboard(monkeypatch, ValueError("not a loopback configuration"))
    argv = [a for a in _run_args(config_path) if a != "--no-dashboard"]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 1
    assert TOKEN not in capsys.readouterr().err
    runtime = _last_runtime()
    assert runtime.closed and runtime.client.closed
    assert ("claim", "cpu-0") not in cp.events  # compute never ran exposed


def test_dashboard_access_url_private_and_cleaned_up(tmp_path, monkeypatch, capsys, cp):
    server = FakeServer()
    config_path = _write_config(tmp_path, cpu_slots=1, dashboard=True)
    _patch_dashboard(monkeypatch, server)

    recorded: dict = {}
    real_write = cli._write_access_file

    def spy_write(config_file, url):
        path = real_write(config_file, url)
        recorded["mode"] = path.stat().st_mode & 0o777
        recorded["url"] = path.read_text().strip()
        return path

    monkeypatch.setattr(cli, "_write_access_file", spy_write)
    argv = [a for a in _run_args(config_path) if a != "--no-dashboard"]
    assert cli.main(argv) == 0

    # Written atomically, mode 0600, containing the full access URL.
    assert recorded["mode"] == 0o600
    assert recorded["url"] == server.dashboard_url
    assert "#dash-secret-token" in recorded["url"]

    # Non-TTY output must not disclose the secret fragment; only the base.
    captured = capsys.readouterr()
    assert "dash-secret-token" not in captured.out
    assert "127.0.0.1:7788" in captured.out

    # Shutdown removed the instance-owned file and closed the server.
    url_file = cli._dashboard_access_file(config_path)
    assert not url_file.exists()
    assert server.shutdown_called.is_set() and server.closed
    assert _last_runtime().closed


def test_dashboard_access_file_kept_when_replaced(tmp_path, monkeypatch):
    """A file not written by this run (another instance's) must survive."""

    class ReplacingServer(FakeServer):
        def server_close(self):
            cli._dashboard_access_file(config_path).write_text("other-instance\n")
            super().server_close()

    config_path = _write_config(tmp_path, cpu_slots=1, dashboard=True)
    _patch_dashboard(monkeypatch, ReplacingServer())
    argv = [a for a in _run_args(config_path) if a != "--no-dashboard"]
    assert cli.main(argv) == 0
    url_file = cli._dashboard_access_file(config_path)
    assert url_file.read_text() == "other-instance\n"


# -- config command ------------------------------------------------------------


def test_config_get_token_fully_redacted(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    assert cli.main(["--config", str(config_path), "config", "get", "token"]) == 0
    out = capsys.readouterr().out
    assert "<redacted>" in out
    assert TOKEN not in out


def test_config_dump_redacts_token(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    assert cli.main(["--config", str(config_path), "config"]) == 0
    out = capsys.readouterr().out
    assert TOKEN not in out
    assert json.loads(out)["token"] == "<redacted>"


def test_config_set_malformed_value_fails_and_preserves(tmp_path, capsys):
    config_path = _write_config(tmp_path, cpu_slots=1)
    with pytest.raises(SystemExit):
        cli.main(["--config", str(config_path), "config", "set", "cpu_slots=abc"])
    assert json.loads(config_path.read_text())["cpu_slots"] == 1


def test_config_set_coerces_int(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    assert cli.main(["--config", str(config_path), "config", "set", "cpu_slots=4"]) == 0
    assert json.loads(config_path.read_text())["cpu_slots"] == 4


# -- login --------------------------------------------------------------------


def test_login_saves_private_config(tmp_path):
    config_path = tmp_path / "config.json"
    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "login",
                "--endpoint",
                ENDPOINT,
                "--token",
                TOKEN,
            ]
        )
        == 0
    )
    saved = json.loads(config_path.read_text())
    assert saved["token"] == TOKEN
    assert saved["endpoint"] == ENDPOINT
    assert config_path.stat().st_mode & 0o077 == 0
