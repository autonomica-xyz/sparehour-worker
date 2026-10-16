"""Runtime lifecycle: executor device wiring, outcomes, rollback, prefetch.

Exercises the real IdleGatedRunner/WorkerRuntime code paths against an
in-memory control-plane client and a fake `docker` executable that records
its argv/environment — no daemon, no GPU, no network.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from sparehour.agent import AgentConfig, AgentError, ExecutionError
from sparehour.models import AgentRegistration, SlotInventory

from sparehour_worker import runner as runner_mod
from sparehour_worker.client import WorkerClient
from sparehour_worker.runner import IdleGatedRunner, WorkerRuntime, WorkerStatus


# -- fixtures ---------------------------------------------------------------


def _slot(vendor="cpu", index=0, caps=None, packs=None, **kw):
    base = dict(
        id=f"w-{vendor}-{index}",
        gpu_index=index,
        gpu_uuid="GPU-0123456789abcdef" if vendor == "amd" else f"uuid-{vendor}-{index}",
        vendor=vendor,
        model="AMD GPU" if vendor == "amd" else f"{vendor} unit",
        vram_mb=8192,
        capabilities=list(caps if caps is not None else ["demo"]),
        model_packs=list(packs or []),
        labels={},
    )
    base.update(kw)
    return SlotInventory(**base)


_MANIFEST = (
    "import os;"
    "open(os.environ['SPAREHOUR_OUTPUT_MANIFEST'],'w').write('{\"files\":[]}')"
)
_ENV_DUMP = (
    "import os,json;"
    "json.dump({k:os.environ.get(k) for k in "
    "['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']},"
    "open(os.environ['SPAREHOUR_OUTPUT_DIR']+'/env.json','w'));"
    + _MANIFEST
)


def _job(command=None, *, executor=None, job_id="job-1"):
    return {
        "id": job_id,
        "lease_token": "tok",
        "workload": "demo",
        "route": "r1",
        "payload": {},
        "executor": executor
        or {"type": "command", "command": command or [sys.executable, "-c", _MANIFEST]},
        "output_contract": [{"name": "out", "max_bytes": 1024}],
    }


class FakeClient:
    """In-memory stand-in for WorkerClient: records lifecycle calls."""

    def __init__(self):
        self.registrations = []
        self.claim_calls = 0
        self.jobs: list[dict] = []
        self.confirmed: list[str] = []
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.uploaded: list[str] = []
        self.heartbeats: list[str] = []
        self.confirm_error: AgentError | None = None
        self.complete_error: AgentError | None = None
        self.claim_errors: list[AgentError] = []

    def register(self, registration):
        self.registrations.append(registration)
        return {"ok": True}

    def heartbeat_agent(self, agent_id, *, draining=None):
        return None

    def claim(self, slot_id, lease_seconds):
        self.claim_calls += 1
        if self.claim_errors:
            raise self.claim_errors.pop(0)
        return self.jobs.pop(0) if self.jobs else None

    def confirm(self, job_id, slot_id, lease_token):
        if self.confirm_error is not None:
            exc, self.confirm_error = self.confirm_error, None
            raise exc
        self.confirmed.append(job_id)

    def heartbeat_job(self, job_id, slot_id, lease_token, lease_seconds):
        return False

    def complete(self, job_id, slot_id, lease_token, outputs):
        if self.complete_error is not None:
            exc, self.complete_error = self.complete_error, None
            raise exc
        self.completed.append(job_id)

    def fail(self, job_id, slot_id, lease_token, code, message):
        self.failed.append((job_id, code))

    def upload_output(self, job_id, slot_id, lease_token, *, name, path, media_type):
        self.uploaded.append(name)
        return {"name": name, "uri": f"remote://{name}"}

    def release(self, *args, **kwargs):
        return None

    def close(self):
        pass


class _LeaseStub:
    """Lease check stand-in for direct executor invocation."""

    def __init__(self, error=None, wait_for=None):
        self.error = error
        self.wait_for = wait_for  # callable polled once before first check

    def check(self):
        if self.wait_for is not None:
            gate, self.wait_for = self.wait_for, None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not gate():
                time.sleep(0.02)
        if self.error is not None:
            raise self.error


def _runner(client, tmp_path, slots=None):
    status = WorkerStatus(agent_id="w")
    config = AgentConfig(
        registration=AgentRegistration(
            agent_id="w",
            provider="local",
            hostname="test",
            cpu_cores=4,
            ram_mb=1024,
            disk_free_mb=1024,
            labels={},
            slots=slots or [_slot()],
        ),
        workdir_root=tmp_path / "work",
        heartbeat_seconds=90,
        lease_seconds=180,
        poll_seconds=0.02,
    )
    config.workdir_root.mkdir(parents=True, exist_ok=True)  # run() does this
    return IdleGatedRunner(client, config, status)


FAKE_DOCKER = """\
#!{python}
import json, os, sys, time
args = sys.argv[1:]
record = os.environ.get("FAKE_DOCKER_RECORD")
entry = {{"argv": args, "env": {{k: os.environ.get(k) for k in (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES", "SPAREHOUR_JOB_ID")}}}}
if record:
    with open(record, "a") as fh:
        fh.write(json.dumps(entry) + "\\n")
if args and args[0] == "run":
    for a in args:
        if a.startswith("type=bind,src="):
            src = a.split("src=", 1)[1].split(",", 1)[0]
            with open(os.path.join(src, "outputs.json"), "w") as fh:
                fh.write('{{"files": []}}')
if args and args[0] in ("run", "pull") and os.environ.get("FAKE_DOCKER_SLEEP"):
    time.sleep(float(os.environ["FAKE_DOCKER_SLEEP"]))
sys.exit(0)
"""


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "docker"
    script.write_text(FAKE_DOCKER.format(python=sys.executable))
    script.chmod(0o755)
    record = tmp_path / "docker-log.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_DOCKER_RECORD", str(record))
    return record


def _docker_calls(record: Path):
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines() if line.strip()]


def _container_job(**executor_overrides):
    executor = {
        "type": "container",
        "image": "img@sha256:" + "a" * 64,
        "command": ["run-job"],
        "environment": {"MODEL_DIR": "/models"},
    }
    executor.update(executor_overrides)
    return _job(executor=executor)


# -- executor device wiring -------------------------------------------------


def test_cpu_container_gets_no_gpu_passthrough(tmp_path, fake_docker):
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    outputs = runner._execute_process(
        _slot("cpu"), _container_job(), workdir, _LeaseStub(), container=True
    )
    assert outputs == []
    calls = _docker_calls(fake_docker)
    assert len(calls) == 1 and calls[0]["argv"][0] == "run"
    argv = calls[0]["argv"]
    assert "--gpus" not in argv
    assert "/dev/kfd" not in argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in argv
    # Host-side GPU variables are cleared, so nothing leaks via -e either.
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0]["env"]["HIP_VISIBLE_DEVICES"] == ""
    assert calls[0]["env"]["ROCR_VISIBLE_DEVICES"] == ""


def test_nvidia_container_keeps_gpus_flag(tmp_path, fake_docker):
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    runner._execute_process(
        _slot("nvidia", gpu_uuid="GPU-nv0"), _container_job(), workdir, _LeaseStub(), container=True
    )
    argv = _docker_calls(fake_docker)[0]["argv"]
    gpus_at = argv.index("--gpus")
    assert argv[gpus_at + 1] == "device=GPU-nv0"
    assert "/dev/kfd" not in argv


def test_amd_container_uses_rocr_passthrough(tmp_path, fake_docker, monkeypatch):
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: str(p) in ("/dev/kfd", "/dev/dri") or real_exists(p)
    )
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    slot = _slot("amd", gpu_uuid="GPU-0123456789abcdef")
    runner._execute_process(slot, _container_job(), workdir, _LeaseStub(), container=True)
    calls = _docker_calls(fake_docker)
    argv = calls[0]["argv"]
    assert "--gpus" not in argv
    assert "/dev/kfd" in argv and "/dev/dri" in argv
    assert "video" in argv and "render" in argv
    # Visibility variables are forwarded into the container (-e NAME).
    assert "ROCR_VISIBLE_DEVICES" in argv and "HIP_VISIBLE_DEVICES" in argv
    assert calls[0]["env"]["ROCR_VISIBLE_DEVICES"] == "GPU-0123456789abcdef"
    assert calls[0]["env"]["HIP_VISIBLE_DEVICES"] == "0"


def test_amd_command_uses_stable_uuid_not_ordinal(tmp_path, monkeypatch):
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: str(p) in ("/dev/kfd", "/dev/dri") or real_exists(p)
    )
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    slot = _slot("amd", index=3, gpu_uuid="GPU-ffeeddccbbaa0011")
    runner._execute_process(
        slot,
        _job(command=[sys.executable, "-c", _ENV_DUMP]),
        workdir,
        _LeaseStub(),
        container=False,
    )
    env = json.loads((workdir / "env.json").read_text())
    assert env["ROCR_VISIBLE_DEVICES"] == "GPU-ffeeddccbbaa0011"
    assert env["HIP_VISIBLE_DEVICES"] == "0"  # index into the filtered set
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_amd_execution_fails_clearly_without_rocm_nodes(tmp_path, monkeypatch):
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: False if str(p).startswith("/dev/") else real_exists(p)
    )
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    with pytest.raises(ExecutionError) as caught:
        runner._execute_process(
            _slot("amd"),
            _job(command=[sys.executable, "-c", _MANIFEST]),
            workdir,
            _LeaseStub(),
            container=False,
        )
    assert caught.value.code == "executor_unavailable"
    assert "/dev/kfd" in str(caught.value)


def test_amd_execution_rejects_fallback_pci_uuid(tmp_path, monkeypatch):
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    with pytest.raises(ExecutionError) as caught:
        runner._execute_process(
            _slot("amd", gpu_uuid="pci-0000:03:00.0"),
            _job(command=[sys.executable, "-c", _MANIFEST]),
            workdir,
            _LeaseStub(),
            container=False,
        )
    assert caught.value.code == "executor_unavailable"
    assert "unique_id" in str(caught.value) or "UUID" in str(caught.value)


def test_container_requires_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()
    with pytest.raises(ExecutionError) as caught:
        runner._execute_process(
            _slot("cpu"), _container_job(), workdir, _LeaseStub(), container=True
        )
    assert caught.value.code == "executor_unavailable"


def test_cancel_stops_named_container(tmp_path, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_SLEEP", "60")
    runner = _runner(FakeClient(), tmp_path)
    workdir = tmp_path / "jobdir"
    workdir.mkdir()

    def container_started():
        return any(
            call["argv"][:1] == ["run"] for call in _docker_calls(fake_docker)
        )

    lease = _LeaseStub(
        error=ExecutionError("cancelled", "job cancellation was requested"),
        wait_for=container_started,
    )
    with pytest.raises(ExecutionError):
        runner._execute_process(
            _slot("cpu"), _container_job(), workdir, lease, container=True
        )
    calls = [call["argv"] for call in _docker_calls(fake_docker)]
    name = "sparehour-job-1"
    assert calls[0][0] == "run"
    assert ["stop", "-t", "2", name] in calls
    assert ["rm", "-f", name] in calls


# -- slot loop lifecycle ----------------------------------------------------


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_once_mode_completes_and_counts_success(tmp_path):
    client = FakeClient()
    client.jobs = [_job()]
    runner = _runner(client, tmp_path)
    runner._slot_loop(runner.config.registration.slots[0], once=True)
    assert client.completed == ["job-1"]
    assert runner.status.completed == 1
    assert runner.status.failed == 0


def test_once_mode_reports_executor_failure(tmp_path):
    client = FakeClient()
    client.jobs = [_job(command=[sys.executable, "-c", "import sys;sys.exit(3)"])]
    runner = _runner(client, tmp_path)
    runner._slot_loop(runner.config.registration.slots[0], once=True)
    assert client.failed == [("job-1", "executor_failed")]
    assert runner.status.completed == 0
    assert runner.status.failed == 1


def test_completion_transport_error_counts_as_failed(tmp_path):
    client = FakeClient()
    client.jobs = [_job()]
    client.complete_error = AgentError("connection reset")
    runner = _runner(client, tmp_path)
    runner._slot_loop(runner.config.registration.slots[0], once=True)
    assert client.completed == []
    assert client.failed == [("job-1", "artifact_transport_failed")]
    assert runner.status.completed == 0
    assert runner.status.failed == 1


def test_continuous_loop_recovers_from_confirm_agent_error(tmp_path):
    client = FakeClient()
    client.jobs = [_job(job_id="j1"), _job(job_id="j2")]
    client.confirm_error = AgentError("confirm rejected")
    runner = _runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
    )
    thread.start()
    assert _wait_for(lambda: bool(client.completed))
    runner.stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    # j1's confirm was rejected (counted as a failed attempt); j2 completed.
    assert client.completed == ["j2"]
    assert runner.status.completed == 1
    assert runner.status.failed == 1


def test_continuous_loop_recovers_from_claim_agent_error(tmp_path):
    client = FakeClient()
    client.claim_errors = [AgentError("claim timeout"), AgentError("claim refused")]
    client.jobs = [_job()]
    runner = _runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
    )
    thread.start()
    assert _wait_for(lambda: bool(client.completed))
    runner.stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert client.claim_calls >= 3
    assert client.completed == ["job-1"]


def test_drain_completes_the_running_job(tmp_path):
    client = FakeClient()
    client.jobs = [
        _job(
            command=[
                sys.executable,
                "-c",
                "import os,time;time.sleep(0.3);" + _MANIFEST,
            ]
        )
    ]
    runner = _runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
    )
    thread.start()
    assert _wait_for(lambda: bool(client.confirmed))
    runner.stop.set()  # drain: already-running work must finish
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert client.completed == ["job-1"]
    assert runner.status.completed == 1


# -- inherited heartbeat/lease worker lifetime ---------------------------------


class StallableClient(FakeClient):
    """FakeClient whose heartbeat transports stall on caller-held gates.

    Events record ordering across threads: a terminal report must never be
    appended while a heartbeat request is still inside its transport call.
    """

    def __init__(self, agent_gate=None, job_gate=None, job_pre_gate=None):
        super().__init__()
        self.agent_gate = agent_gate
        self.job_gate = job_gate
        # Suspends a heartbeat_job call before it even reaches the recorded
        # transport — the "late first heartbeat" shape, where the keeper's
        # call arrives after its stop event was already set.
        self.job_pre_gate = job_pre_gate
        self.events: list[str] = []

    def heartbeat_agent(self, agent_id, *, draining=None):
        if draining is None:
            self.events.append("agent-heartbeat-entered")
            if self.agent_gate is not None:
                self.agent_gate.wait(timeout=10)
            self.events.append("agent-heartbeat-returned")
        return super().heartbeat_agent(agent_id, draining=draining)

    def heartbeat_job(self, job_id, slot_id, lease_token, lease_seconds):
        self.events.append("job-heartbeat-called")
        if self.job_pre_gate is not None:
            self.job_pre_gate.wait(timeout=10)
        self.events.append("job-heartbeat-entered")
        if self.job_gate is not None:
            self.job_gate.wait(timeout=10)
        self.events.append("job-heartbeat-returned")
        return super().heartbeat_job(job_id, slot_id, lease_token, lease_seconds)

    def complete(self, job_id, slot_id, lease_token, outputs):
        self.events.append("complete-entered")
        super().complete(job_id, slot_id, lease_token, outputs)

    def fail(self, job_id, slot_id, lease_token, code, message):
        self.events.append("fail-entered")
        super().fail(job_id, slot_id, lease_token, code, message)


def _fast_runner(client, tmp_path):
    """Runner with a 1s heartbeat so a stalled request outlives upstream's
    timed joins (heartbeat_seconds + 1) inside the test budget."""
    runner = _runner(client, tmp_path)
    runner.config = replace(
        runner.config, heartbeat_seconds=1, lease_seconds=15
    )
    return runner


def test_run_joins_stalled_heartbeat_worker_before_returning(tmp_path):
    """A drained worker cannot return while an agent heartbeat is in flight."""
    gate = threading.Event()
    client = StallableClient(agent_gate=gate)
    client.jobs = [
        _job(command=[sys.executable, "-c", "import time;time.sleep(1.5);" + _MANIFEST])
    ]
    runner = _fast_runner(client, tmp_path)
    done = threading.Event()
    run_errors: list[BaseException] = []

    def _run():
        try:
            runner.run(once=True)
        except BaseException as exc:
            run_errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_run, name="run", daemon=True)
    thread.start()
    try:
        assert _wait_for(lambda: "agent-heartbeat-entered" in client.events)
        assert _wait_for(lambda: client.completed == ["job-1"])
        # Slot work has finished. Outlast upstream's heartbeat_seconds+1 join.
        time.sleep(2.5)
        assert not done.is_set()
        assert "agent-heartbeat-returned" not in client.events
        gate.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert "agent-heartbeat-returned" in client.events
        assert not run_errors
        assert runner.status.completed == 1
        assert not runner.client._heartbeat_threads
        assert not runner.client._lease_workers
    finally:
        gate.set()
        runner.stop.set()
        thread.join(timeout=10)



def test_complete_is_fenced_behind_released_lease_heartbeat(tmp_path):
    """A lease heartbeat blocked in transport must not race the terminal
    complete() report: the wrapper joins the job's lease worker first, even
    when _LeaseKeeper.stop's timed join has already expired."""
    gate = threading.Event()
    client = StallableClient(job_gate=gate)
    client.jobs = [
        _job(command=[sys.executable, "-c", "import time;time.sleep(1.5);" + _MANIFEST])
    ]
    runner = _fast_runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, True), name="slot-test", daemon=True
    )
    thread.start()
    try:
        assert _wait_for(lambda: "job-heartbeat-entered" in client.events)
        # The proxy tracked the keeper's own worker at the _execute boundary.
        assert _wait_for(lambda: "job-1" in runner.client._lease_workers)
        worker, _stop = runner.client._lease_workers["job-1"]
        # Executor finishes (~1.5s) and lease.stop's timed join expires
        # (hb+1 = 2s): without the fence, complete() would post at ~4s while
        # the heartbeat is still stalled inside the transport.
        time.sleep(3.2)
        assert "job-heartbeat-returned" not in client.events
        assert "complete-entered" not in client.events
        gate.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        # The heartbeat's transport call finished before the terminal report.
        assert client.events.index("job-heartbeat-returned") < client.events.index(
            "complete-entered"
        )
        assert client.completed == ["job-1"]
        assert runner.status.completed == 1
        # The known worker was actually joined — it provably cannot
        # heartbeat again — and its record was reclaimed.
        assert not worker.is_alive()
        assert "job-1" not in runner.client._lease_workers
    finally:
        gate.set()
        runner.stop.set()
        thread.join(timeout=10)


def test_fail_report_is_fenced_behind_released_lease_heartbeat(tmp_path):
    """The failure path gets the same fence: a stalled lease heartbeat must
    finish before fail() posts."""
    gate = threading.Event()
    client = StallableClient(job_gate=gate)
    client.jobs = [
        _job(
            command=[
                sys.executable,
                "-c",
                "import time,sys;time.sleep(1.5);sys.exit(3)",
            ]
        )
    ]
    runner = _fast_runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, True), name="slot-test", daemon=True
    )
    thread.start()
    try:
        assert _wait_for(lambda: "job-heartbeat-entered" in client.events)
        assert _wait_for(lambda: "job-1" in runner.client._lease_workers)
        worker, _stop = runner.client._lease_workers["job-1"]
        time.sleep(3.2)  # past executor exit and the expired timed join
        assert "job-heartbeat-returned" not in client.events
        assert "fail-entered" not in client.events
        gate.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert client.events.index("job-heartbeat-returned") < client.events.index(
            "fail-entered"
        )
        assert client.failed == [("job-1", "executor_failed")]
        assert runner.status.failed == 1
        assert not worker.is_alive()
        assert "job-1" not in runner.client._lease_workers
    finally:
        gate.set()
        runner.stop.set()
        thread.join(timeout=10)


def test_late_first_lease_heartbeat_cannot_run_after_terminal_report(tmp_path):
    """A keeper thread that reaches heartbeat_job only after its stop event
    was already set (the late-first-heartbeat shape) must still land before
    complete(): the fence joins the known worker, so no per-job tombstone
    gate is needed to keep the ordering."""
    pre = threading.Event()
    client = StallableClient(job_pre_gate=pre)
    client.jobs = [
        _job(command=[sys.executable, "-c", "import time;time.sleep(1.5);" + _MANIFEST])
    ]
    runner = _fast_runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, True), name="slot-test", daemon=True
    )
    thread.start()
    try:
        # The keeper's first heartbeat call is suspended before it reaches
        # the recorded transport.
        assert _wait_for(lambda: "job-heartbeat-called" in client.events)
        assert _wait_for(lambda: "job-1" in runner.client._lease_workers)
        worker, stop = runner.client._lease_workers["job-1"]
        # Executor exits at ~1.5s; stop is set and its timed join (hb+1=2s)
        # expires ~3.5s in while the keeper is still pre-transport.
        assert _wait_for(stop.is_set)
        time.sleep(4.0)
        assert "job-heartbeat-entered" not in client.events
        assert "complete-entered" not in client.events
        # Releasing the pre-gate now lets the late first heartbeat start
        # only while the terminal fence is already joining its worker.
        pre.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert client.events.index("job-heartbeat-returned") < client.events.index(
            "complete-entered"
        )
        assert client.completed == ["job-1"]
        assert runner.status.completed == 1
        assert not worker.is_alive()
        assert "job-1" not in runner.client._lease_workers
    finally:
        pre.set()
        runner.stop.set()
        thread.join(timeout=10)


def test_completed_jobs_leave_no_retained_lease_state(tmp_path):
    """Continuous running must keep per-job worker state bounded: once a
    job's keeper is fenced and joined, its record is reclaimed — even when a
    later job reuses the same id (what an id-tombstone would wrongly block).
    """
    claim_gate = threading.Event()

    class ClaimGatedClient(FakeClient):
        def claim(self, slot_id, lease_seconds):
            if self.claim_calls >= 1:
                claim_gate.wait(timeout=10)
            return super().claim(slot_id, lease_seconds)

        def heartbeat_job(self, job_id, slot_id, lease_token, lease_seconds):
            self.heartbeats.append(job_id)
            return super().heartbeat_job(job_id, slot_id, lease_token, lease_seconds)

    client = ClaimGatedClient()
    # Same job id twice plus a drain job to prove reuse is unaffected.
    client.jobs = [
        _job(command=[sys.executable, "-c", "import time;time.sleep(1.5);" + _MANIFEST]),
        _job(command=[sys.executable, "-c", "import time;time.sleep(1.5);" + _MANIFEST]),
    ]
    runner = _fast_runner(client, tmp_path)
    slot = runner.config.registration.slots[0]
    thread = threading.Thread(
        target=runner._slot_loop, args=(slot, False), name="slot-test", daemon=True
    )
    thread.start()
    try:
        # job-1 heartbeats at least once, completes, and leaves no state
        # behind while the second claim is still parked on the gate — the
        # worker is still running, so this is a live-state invariant.
        assert _wait_for(lambda: client.completed == ["job-1"])
        assert not runner.client._lease_workers
        claim_gate.set()
        assert _wait_for(lambda: "job-1" in runner.client._lease_workers)
        worker, _stop = runner.client._lease_workers["job-1"]
        assert _wait_for(lambda: len(client.completed) == 2)
        runner.stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        # The reused id heartbeated on its own keeper and was reclaimed too.
        assert len(client.heartbeats) >= 2
        assert not worker.is_alive()
        assert not runner.client._lease_workers
        assert runner.status.completed == 2
    finally:
        claim_gate.set()
        runner.stop.set()
        thread.join(timeout=10)


# -- WorkerRuntime: lock, persistence, rollback ------------------------------


def _config(tmp_path, **overrides):
    base = dict(
        endpoint="http://localhost:1",
        token="tok",
        agent_id="w",
        provider="local",
        capabilities=["saved-cap"],
        model_packs=[],
        labels={},
        cpu_slots=1,
        idle_only=False,
        idle_after_seconds=300,
        dynamic=False,
        dynamic_poll_seconds=60,
        model_catalog="",
        workdir=str(tmp_path / "work"),
        dashboard=False,
        dashboard_host="127.0.0.1",
        dashboard_port=7788,
        poll_seconds=0.02,
        heartbeat_seconds=15,
        lease_seconds=90,
    )
    base.update(overrides)
    return base


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    def fake_discover(agent_id, *, capabilities, model_packs, labels, cpu_slots):
        return (
            [_slot("cpu", index=index, caps=capabilities, packs=model_packs)
             for index in range(max(1, cpu_slots))],
            ["fake discovery"],
        )

    monkeypatch.setattr(runner_mod.hardware, "discover_slots", fake_discover)
    rt = WorkerRuntime(_config(tmp_path), tmp_path / "config.json")
    rt.client.close()
    rt.client = FakeClient()
    try:
        yield rt
    finally:
        rt.close()


def test_apply_capability_list_rolls_back_on_registration_failure(runtime):
    def rejecting(registration):
        raise AgentError("endpoint unreachable")

    runtime.client.register = rejecting
    ok, message = runtime.apply_capability_list(["other-cap"], persist=False)
    assert not ok
    assert "rejected" in message or "unreachable" in message
    assert runtime.config["capabilities"] == ["saved-cap"]
    assert all(s.capabilities == ["saved-cap"] for s in runtime.slots)


def test_update_model_packs_reregisters_without_capability_change(runtime):
    # Unchanged capabilities take the early-return path...
    ok, _ = runtime.apply_capability_list(["saved-cap"], persist=False)
    assert ok and runtime.client.registrations == []
    # ...but a new pack must still reach the controller via registration.
    ok, _ = runtime.update_model_packs(["pack-a"])
    assert ok
    assert len(runtime.client.registrations) == 1
    slot = runtime.client.registrations[0].slots[0]
    assert slot.model_packs == ["pack-a"]
    assert slot.capabilities == ["saved-cap"]


def test_update_model_packs_rolls_back_local_inventory(runtime):
    def rejecting(registration):
        raise AgentError("endpoint unreachable")

    runtime.client.register = rejecting
    ok, _ = runtime.update_model_packs(["pack-a"])
    assert not ok
    assert runtime.config["model_packs"] == []
    assert all(s.model_packs == [] for s in runtime.slots)


def test_saved_selection_not_leaked_by_dynamic_pack_save(runtime):
    # Dynamic mode diverges the runtime selection without persisting it.
    ok, _ = runtime.apply_capability_list(["dyn-cap"], persist=False)
    assert ok
    ok, _ = runtime.update_model_packs(["pack-a"])
    assert ok
    saved = json.loads(runtime.config_file.read_text())
    assert saved["capabilities"] == ["saved-cap"]
    assert saved["model_packs"] == ["pack-a"]
    assert runtime.config["capabilities"] == ["dyn-cap"]


def test_persisted_selection_updates_saved_file(runtime):
    ok, _ = runtime.set_capability("new-cap", True)
    assert ok
    saved = json.loads(runtime.config_file.read_text())
    assert saved["capabilities"] == ["saved-cap", "new-cap"]


def test_save_failure_rolls_back_selection_and_registration(runtime, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    runtime.config_file = blocker / "config.json"  # mkdir will fail
    ok, message = runtime.apply_capability_list(["cap-z"], persist=True)
    assert not ok
    assert "save" in message
    assert runtime.config["capabilities"] == ["saved-cap"]
    # Rollback re-registered the previous selection.
    last = runtime.client.registrations[-1]
    assert all(s.capabilities == ["saved-cap"] for s in last.slots)


def test_save_config_restores_run_overrides(runtime):
    # CLI rewrote cpu_slots/idle_only for this run only; the file must keep
    # the saved values when a later pack update persists the config.
    overridden = WorkerRuntime(
        {**runtime.config, "cpu_slots": 4, "idle_only": True},
        runtime.config_file,
        saved_overrides={"cpu_slots": 0, "idle_only": False},
    )
    overridden.client.close()
    overridden.client = FakeClient()
    try:
        ok, _ = overridden.update_model_packs(["pack-a"])
        assert ok
        saved = json.loads(runtime.config_file.read_text())
        assert saved["cpu_slots"] == 0
        assert saved["idle_only"] is False
    finally:
        overridden.close()


def test_constructor_closes_client_on_partial_failure(tmp_path, monkeypatch):
    instances = []

    class StubClient:
        def __init__(self, endpoint, token):
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runner_mod, "WorkerClient", StubClient)

    def failing_discover(agent_id, **kwargs):
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(runner_mod.hardware, "discover_slots", failing_discover)
    with pytest.raises(RuntimeError, match="discovery failed"):
        WorkerRuntime(_config(tmp_path), tmp_path / "config.json")
    assert instances and instances[0].closed


def test_capability_changes_serialized_through_lock(runtime):
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def blocking_register(registration):
        entered.set()
        release.wait(timeout=10)
        observed.append(tuple(registration.slots[0].capabilities))

    runtime.client.register = blocking_register
    t1 = threading.Thread(
        target=runtime.apply_capability_list, args=(["cap-one"],), kwargs={"persist": False}
    )
    t1.start()
    assert entered.wait(10)
    t2 = threading.Thread(
        target=runtime.apply_capability_list, args=(["cap-two"],), kwargs={"persist": False}
    )
    t2.start()
    time.sleep(0.3)
    assert observed == []  # second caller is still waiting on runtime.lock
    release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert observed == [("cap-one",), ("cap-two",)]
    assert runtime.config["capabilities"] == ["cap-two"]


# -- prefetch ---------------------------------------------------------------


def _container_queue(capability="saved-cap"):
    return {
        "workload": "w",
        "version": "1",
        "description": "",
        "route": "r1",
        "capability": capability,
        "model_packs": [],
        "resources": {"vendor": "cpu"},
        "executor_type": "container",
        "executor_detail": "img@sha256:" + "a" * 64,
        "executor_environment": {},
        "pending_jobs": 1,
    }


def test_prefetch_without_docker_skips_pulls(runtime, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(runtime, "queues_report", lambda max_age=0.0: [_container_queue()])
    lines = runtime.prefetch_images()
    assert any("docker not found" in line for line in lines)
    assert not any("pull img" in line for line in lines)



def test_prefetch_images_cancelled_by_stop(runtime, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/docker")
    monkeypatch.setattr(runtime, "queues_report", lambda max_age=0.0: [_container_queue()])
    runtime.stop.set()
    lines = runtime.prefetch_images()
    assert any("skipped (worker stopping)" in line for line in lines)


def test_docker_pull_is_bounded(runtime, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_SLEEP", "60")
    started = time.monotonic()
    verdict = runtime._docker_pull("img@sha256:" + "a" * 64, timeout=0.5)
    assert "exceeded" in verdict
    assert time.monotonic() - started < 30


def test_docker_pull_aborts_when_runtime_stops(runtime, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_SLEEP", "60")
    runtime.stop.set()
    verdict = runtime._docker_pull("img@sha256:" + "a" * 64, timeout=600)
    assert "cancelled" in verdict


def test_prefetch_queue_listing_failure_is_recorded(runtime, monkeypatch):
    # An HTTP/transport error from /v1/workloads must not kill the prefetch
    # thread silently: the dashboard needs a FAILED line and the running
    # flag must reset. The exception text echoes the remote response body
    # (and could carry credentials) — only a classified summary is stored.
    remote_body = "down for maintenance agent-token-xyz"
    monkeypatch.setattr(
        runtime,
        "queues_report",
        lambda max_age=0.0: (_ for _ in ()).throw(
            RuntimeError(f"workload listing failed: HTTP 503: {remote_body}")
        ),
    )
    lines = runtime.prefetch_images()
    assert runtime.status.prefetch_running is False
    assert any("FAILED" in line and "503" in line for line in lines)
    assert remote_body not in "\n".join(lines)
    log_text = "\n".join(runtime.status.prefetch_log)
    assert "FAILED" in log_text and "503" in log_text
    assert remote_body not in log_text
    assert any("prefetch failed" in note for note in runtime.status.notes)
    assert remote_body not in "\n".join(runtime.status.notes)


def test_prefetch_agent_error_is_recorded(runtime, monkeypatch):
    def raising(max_age=0.0):
        raise AgentError("connection refused to http://controller:8080")

    monkeypatch.setattr(runtime, "queues_report", raising)
    runtime.prefetch_images()
    assert runtime.status.prefetch_running is False
    assert any("FAILED" in line for line in runtime.status.prefetch_log)
    assert "refused" not in "\n".join(runtime.status.prefetch_log)


def test_prefetch_http_error_never_leaks_response_body(runtime):
    """End-to-end: /v1/workloads returns a 503 whose body carries a token and
    a signed URL. fetch_workloads must surface only the HTTP status, so the
    classified note/log lines cannot contain the remote body."""
    secret_body = (
        "diagnostic dump token=dgp_workersekret retry "
        "https://media.evil.example/upload?signature=AKIA-SECRET"
    )

    def handler(request):
        return httpx.Response(503, text=secret_body)

    client = WorkerClient(
        "http://127.0.0.1:8730",
        "dgp_workersekret",
        transport=httpx.MockTransport(handler),
    )
    runtime.client.close()
    runtime.client = client
    try:
        lines = runtime.prefetch_images()
    finally:
        runtime.client = FakeClient()
        client.close()
    assert runtime.status.prefetch_running is False
    assert any("FAILED" in line and "503" in line for line in lines)
    assert "AKIA-SECRET" not in "\n".join(lines)
    assert "dgp_workersekret" not in "\n".join(lines)
    assert "AKIA-SECRET" not in "\n".join(runtime.status.prefetch_log)
    assert "dgp_workersekret" not in "\n".join(runtime.status.prefetch_log)
    assert "AKIA-SECRET" not in "\n".join(runtime.status.notes)
    assert "dgp_workersekret" not in "\n".join(runtime.status.notes)


def test_prefetch_recovers_after_listing_failure(runtime, monkeypatch):
    # A failed listing leaves no stuck state: the next prefetch runs.
    calls = {"n": 0}

    def flaky(max_age=0.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("workload listing failed: HTTP 500: boom")
        return [_container_queue()]

    monkeypatch.setattr(runtime, "queues_report", flaky)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    first = runtime.prefetch_images()
    assert any("FAILED" in line for line in first)
    runtime.prefetch_images()
    assert runtime.status.prefetch_running is False
    assert any("docker not found" in line for line in runtime.status.prefetch_log)


# -- registration freshness ----------------------------------------------------


def test_run_registers_current_inventory_not_stale_config(runtime, tmp_path):
    # The runner's AgentConfig captures a pre-provisioning snapshot missing
    # the current capability and model pack; upstream AgentRunner.run
    # registers whatever config.registration holds. The bound runner must
    # instead register the locked runtime's current inventory, and the slot
    # threads/heartbeat must run from that same registration.
    stale_config = AgentConfig(
        registration=runtime.registration(),
        workdir_root=tmp_path / "work",
        heartbeat_seconds=90,
        lease_seconds=180,
        poll_seconds=0.02,
    )
    runner = IdleGatedRunner(
        runtime.client,
        stale_config,
        status=runtime.status,
        runtime=runtime,
    )
    # Simulate the dynamic sync provisioning cycle landing after the runner
    # was constructed (the real CLI ordering): a new capability and a model
    # pack that the construction-time registration never saw.
    ok, _ = runtime.apply_capability_list(["provisioned-cap"], persist=False)
    assert ok
    ok, _ = runtime.update_model_packs(["pack-a"])
    assert ok
    assert runtime.config["capabilities"] == ["provisioned-cap"]
    assert runtime.config["model_packs"] == ["pack-a"]

    runner.stop.set()
    runner.run(once=True)

    sent = runtime.client.registrations[-1]
    for slot in sent.slots:
        assert slot.capabilities == ["provisioned-cap"]
        assert slot.model_packs == ["pack-a"]
    # The runner's live config is exactly the registration it sent, so
    # heartbeats and slot threads see the provisioned inventory.
    for slot in runner.config.registration.slots:
        assert slot.capabilities == ["provisioned-cap"]
        assert slot.model_packs == ["pack-a"]


def test_initial_registration_cannot_lose_to_concurrent_update(runtime, tmp_path):
    # Deterministic version of the startup race: the initial registration's
    # send is frozen mid-flight while a dynamic inventory update is issued.
    # Under the old design (snapshot refresh, unlocked send) the stale
    # initial register would land last and revert the controller's view.
    # With the locked snapshot+send transaction the update is guaranteed to
    # be the controller's latest state.
    frozen_send = threading.Event()
    release = threading.Event()
    sent_at_freeze = []

    def blocking_register(registration):
        if not frozen_send.is_set():
            frozen_send.set()
            sent_at_freeze.append(registration)
            release.wait(timeout=10)
        # No gate on later sends: the dynamic update's own register must
        # record immediately once it gets the lock.
        runtime.client.recorded.append(registration)

    runtime.client.recorded = []
    runtime.client.register = blocking_register

    stale_config = AgentConfig(
        registration=runtime.registration(),
        workdir_root=tmp_path / "work",
        heartbeat_seconds=90,
        lease_seconds=180,
        poll_seconds=0.02,
    )
    runner = IdleGatedRunner(
        runtime.client,
        stale_config,
        status=runtime.status,
        runtime=runtime,
    )
    runner.stop.set()

    run_errors: list[BaseException] = []

    def run_initial():
        try:
            runner.run(once=True)
        except BaseException as exc:  # surfaced below
            run_errors.append(exc)

    run_thread = threading.Thread(target=run_initial)
    run_thread.start()
    # The initial register is frozen on the wire holding runtime.lock.
    assert frozen_send.wait(10)
    assert sent_at_freeze, "runner must register through runtime.client"

    # Attempt the dynamic update while the initial send is still in flight:
    # a new capability and a new model pack, in one updater pass.
    update_result: list[tuple[bool, str]] = []

    def update():
        ok_cap, _ = runtime.apply_capability_list(["new-cap"], persist=False)
        ok_pack, _ = runtime.update_model_packs(["pack-z"])
        update_result.append((ok_cap and ok_pack, "dynamic update"))

    update_thread = threading.Thread(target=update)
    update_thread.start()
    # Give the updater a real chance to interleave; it must be blocked on
    # the lock, not racing past the frozen send.
    time.sleep(0.2)
    assert update_result == []
    release.set()
    run_thread.join(timeout=10)
    update_thread.join(timeout=10)
    assert not run_errors
    assert update_result and update_result[0][0]

    # Controller-latest is the update, and the initial send was itself the
    # locked-current snapshot (old caps here, since the update waited).
    initial_sent = sent_at_freeze[0]
    assert all(s.capabilities == ["saved-cap"] for s in initial_sent.slots)
    assert all(s.model_packs == [] for s in initial_sent.slots)
    latest = runtime.client.recorded[-1]
    for slot in latest.slots:
        assert slot.capabilities == ["new-cap"]
        assert slot.model_packs == ["pack-z"]
    for slot in runtime.slots:
        assert slot.capabilities == ["new-cap"]
        assert slot.model_packs == ["pack-z"]
    # The runner's adopted config equals what the initial send posted —
    # the registration it actually used, not the constructor's snapshot.
    assert runner.config.registration.slots == initial_sent.slots


def test_standalone_runner_registers_config_inventory(tmp_path):
    # Without a bound runtime the runner keeps standalone behavior: the
    # construction-time registration is sent verbatim through the proxy.
    client = FakeClient()
    runner = _runner(client, tmp_path, slots=[_slot(caps=["mine"], packs=["p1"])])
    runner.stop.set()
    runner.run(once=True)
    assert len(client.registrations) == 1
    sent = client.registrations[0]
    assert sent is runner.config.registration
    assert sent.slots[0].capabilities == ["mine"]
    assert sent.slots[0].model_packs == ["p1"]
    assert runner.client.registration_hook is None
