"""Worker runtime: registration, idle-gated claim loop, prefetch, status.

Reuses the control plane's AgentRunner (claim/confirm/heartbeat/upload/
complete lifecycle and the http/plugin executors). This module adds:
- idle/lock gating before claiming (SETI@home mode),
- one process/container executor for all vendors with per-vendor device
  wiring (NVIDIA `--gpus`, AMD ROCr device passthrough, CPU without GPU
  device access) and explicit named-container teardown on cancel/timeout,
- a live status snapshot for the dashboard with true success/failed counts,
- one locked snapshot+send transaction for every published registration —
  the runner's initial register and live capability/model-pack updates
  alike — with rollback on failure, so a slow send can never overwrite
  newer inventory (the control plane preserves busy slots),
- full joins of the inherited heartbeat/lease request workers before
  run() returns and before each job's terminal report — upstream only
  time-joins them, so without this a stalled request could outlive the
  client or a late lease heartbeat could race complete/fail.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparehour.agent import (
    AgentConfig,
    AgentError,
    AgentRunner,
    ExecutionError,
    _load_output_manifest,
    _process_request,
)
from sparehour.models import AgentRegistration, SlotInventory

from . import config as config_mod
from . import hardware, idle, queues as queues_mod
from .client import WorkerClient

# Docker container names the runner owns; used to force-stop the workload
# container itself on cancellation/timeout (killing only the `docker run`
# CLI would leave a detached container holding the GPU).
_CONTAINER_PREFIX = "sparehour-"

# Upper bound on one `docker pull` during prefetch; network stalls must not
# pin the dashboard's prefetch worker (or `sparehour-worker prefetch`) forever.
PREFETCH_PULL_TIMEOUT = 600.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


@dataclass
class WorkerStatus:
    """Thread-safe snapshot the CLI and dashboard read."""

    agent_id: str
    started_at: float = field(default_factory=time.time)
    paused: bool = False
    idle_only: bool = False
    idle_state: str = "unknown"
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed: int = 0
    failed: int = 0
    started_jobs: int = 0
    prefetch_running: bool = False
    notes: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    prefetch_log: list[str] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note(self, message: str) -> None:
        with self._lock:
            self.notes.appendleft(f"{_now()} {message}")

    def slot_state(self, slot_id: str, state: str, job: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.slots[slot_id] = {
                "state": state,
                "job": (
                    {
                        "id": job.get("id"),
                        "workload": job.get("workload"),
                        "route": job.get("route"),
                        "started_at": _now(),
                        "started_at_seconds": time.time(),
                    }
                    if job
                    else None
                ),
            }

    def job_started(self, slot_id: str, job: dict[str, Any]) -> None:
        with self._lock:
            self.started_jobs += 1
        self.slot_state(slot_id, "running", job)
        self.note(f"slot {slot_id}: claimed job {job.get('id')} ({job.get('workload')})")

    def job_finished(self, slot_id: str, job: dict[str, Any], outcome: str) -> None:
        """Record the terminal controller result: 'completed' or 'failed'.

        'completed' means the control plane accepted complete(); anything
        else — reported failure, cancelled lease, or a lifecycle/confirm
        AgentError that ended the job — counts as a failed attempt.
        """
        with self._lock:
            if outcome == "completed":
                self.completed += 1
            else:
                self.failed += 1
        self.slot_state(slot_id, "idle")
        self.note(f"slot {slot_id}: {outcome} job {job.get('id')}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "started_at": self.started_at,
                "paused": self.paused,
                "idle_only": self.idle_only,
                "idle_state": self.idle_state,
                "slots": {key: dict(value) for key, value in self.slots.items()},
                "completed": self.completed,
                "failed": self.failed,
                "started_jobs": self.started_jobs,
                "prefetch_running": self.prefetch_running,
                "notes": list(self.notes),
                "prefetch_log": list(self.prefetch_log),
                "uptime_seconds": int(time.time() - self.started_at),
            }


class _OutcomeClient:
    """Recording/lifecycle wrapper around the control-plane client.

    ``AgentRunner._run_job`` deliberately swallows job outcomes (it reports
    complete/fail and returns), so the terminal report is recorded here, per
    claiming thread, for the slot loop to fold into WorkerStatus. Every
    other method/attribute (``client`` transport handle included) delegates
    straight through.

    ``register`` is the one intercepted upstream call: a bound runner
    (``runtime`` passed to IdleGatedRunner) sets ``registration_hook`` so the
    initial registration goes through the runtime's locked snapshot+send
    transaction instead of posting a config snapshot that could overwrite a
    concurrent dynamic update.

    The wrapper also owns the lifetime bookkeeping upstream cannot provide.
    ``IdleGatedRunner._execute`` hands every job's ``_LeaseKeeper`` here via
    ``track_lease`` — the keeper owns the only thread that ever calls
    ``heartbeat_job`` for its job, so its worker thread and stop event are
    the fence's exact, already-known target (no job-id tombstones, no
    caller-thread snapshots). The periodic agent heartbeat is tracked by
    thread instead (``draining=None`` — the dedicated loop thread; the
    register-time ``draining=False`` send and ``request_drain``'s
    ``draining=True`` send run on ordinary callers). Fences are then applied
    at both boundaries:

    - ``complete``/``fail``/``release`` set the keeper's stop event, join
      its worker thread without a bound, and drop the record — once joined,
      the worker provably cannot heartbeat again, so a late lease heartbeat
      can never land after the terminal report and no per-job state outlives
      the job. ``_LeaseKeeper.stop``'s timed join may expire while a
      heartbeat request is still blocked; this join is bounded by the
      client's own request timeout instead.
    - ``join_workers`` — called from ``IdleGatedRunner.run``'s finally,
      after upstream has set every stop event — marks all workers closed
      and fully reaps them, so the client is never closed under a live
      background request.

    Joining skips the current and main threads, so a caller-side drain can
    never deadlock on itself.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._local = threading.local()
        self.registration_hook: Any | None = None
        self._workers_lock = threading.Lock()
        self._heartbeat_threads: set[threading.Thread] = set()
        # job_id -> (lease worker thread, its stop event); captured at the
        # _execute boundary so every tracked worker is a known thread that
        # terminates, not an unbounded set of retired job ids.
        self._lease_workers: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._workers_closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def register(self, registration: AgentRegistration) -> None:
        if self.registration_hook is not None:
            self.registration_hook(registration)
        else:
            self._inner.register(registration)

    def heartbeat_agent(self, agent_id: str, *, draining: bool | None = None) -> None:
        # Only the periodic heartbeat-loop thread calls with draining=None;
        # register-time and request_drain heartbeats pass draining explicitly.
        if draining is not None:
            self._inner.heartbeat_agent(agent_id, draining=draining)
            return
        with self._workers_lock:
            if self._workers_closed:
                # Woke between the closed mark and this call: the lease/
                # heartbeat stop events are already set, so the worker exits
                # now and the fence stays exact — no post-close request.
                return
            self._heartbeat_threads.add(threading.current_thread())
        self._inner.heartbeat_agent(agent_id)

    def heartbeat_job(
        self, job_id: str, slot_id: str, lease_token: str, lease_seconds: int
    ) -> bool:
        with self._workers_lock:
            if self._workers_closed:
                # Woke between the closed mark and this call: every stop
                # event is already set, so the keeper exits now rather than
                # racing a heartbeat past shutdown.
                return False
        return self._inner.heartbeat_job(job_id, slot_id, lease_token, lease_seconds)

    def track_lease(self, lease: Any) -> None:
        """Record a job's lease keeper's worker/stop pair for fencing.

        Called by ``IdleGatedRunner._execute`` with the keeper upstream just
        started in ``_run_job``. Bounded compatibility: relies only on
        ``_LeaseKeeper.job_id``/``_stop``/``_thread`` — the same internals
        ``stop()`` itself uses — not on any module-global monkeypatching.
        """
        with self._workers_lock:
            self._lease_workers[lease.job_id] = (lease._thread, lease._stop)

    def complete(self, job_id: str, slot_id: str, lease_token: str, outputs: Any) -> None:
        self._fence_lease(job_id)
        self._inner.complete(job_id, slot_id, lease_token, outputs)
        self._local.outcome = "completed"

    def fail(self, job_id: str, slot_id: str, lease_token: str, code: str, message: str) -> None:
        self._fence_lease(job_id)
        self._inner.fail(job_id, slot_id, lease_token, code, message)
        self._local.outcome = "failed"

    def release(self, job_id: str, slot_id: str, lease_token: str, reason: str) -> None:
        self._fence_lease(job_id)
        self._inner.release(job_id, slot_id, lease_token, reason)

    def pop_outcome(self) -> str | None:
        outcome = getattr(self._local, "outcome", None)
        self._local.outcome = None
        return outcome

    def _fence_lease(self, job_id: str) -> None:
        """Fully reap this job's lease-heartbeat worker before its terminal post.

        ``_LeaseKeeper.stop``'s timed join may expire while a heartbeat
        request is still blocked. Because ``track_lease`` captured the
        keeper's actual worker and stop event, fencing is exact: set the
        stop event (idempotent; upstream already set it), join the worker
        once without a timeout — its exit is bounded by the client's request
        timeout — then drop the record. A joined thread can never heartbeat
        again, so no closing-marker has to outlive the job and a completed
        job leaves no retained worker or tombstone.
        """
        with self._workers_lock:
            record = self._lease_workers.get(job_id)
        if record is None:
            return
        worker, stop = record
        stop.set()
        current = threading.current_thread()
        if worker is not current and worker.is_alive():
            worker.join()
        if not worker.is_alive():
            with self._workers_lock:
                self._lease_workers.pop(job_id, None)

    def join_workers(self) -> None:
        """Join every recorded heartbeat/lease worker; skips self and main.

        Preconditions (all established by ``AgentRunner.run``'s finally
        before this is invoked): the heartbeat stop event and every lease
        stop event are set. The closed mark makes any not-yet-registered
        heartbeat call a no-op; surviving workers are only finishing an
        in-flight request — bounded by the client's request timeout — then
        exit. The main thread and arbitrary drain callers (which only send
        ``draining`` heartbeats and are never recorded) are excluded, and no
        runtime lock is held while joining.
        """
        current = threading.current_thread()
        main = threading.main_thread()
        with self._workers_lock:
            self._workers_closed = True
        while True:
            with self._workers_lock:
                threads = set(self._heartbeat_threads)
                stops: list[threading.Event] = []
                for worker, stop in self._lease_workers.values():
                    threads.add(worker)
                    stops.append(stop)
                live = [
                    thread
                    for thread in threads
                    if thread.is_alive() and thread is not current and thread is not main
                ]
            # Idempotent stop-event pokes guarantee exit even if a keeper
            # outlived its slot thread's terminal report path.
            for stop in stops:
                stop.set()
            if not live:
                # Workers can no longer register (closed gate); reclaim
                # every dead worker's record.
                with self._workers_lock:
                    self._heartbeat_threads = {
                        thread
                        for thread in self._heartbeat_threads
                        if thread.is_alive()
                    }
                    self._lease_workers = {
                        job_id: record
                        for job_id, record in self._lease_workers.items()
                        if record[0].is_alive()
                    }
                return
            for thread in live:
                thread.join()

    def open_workers(self) -> None:
        """Re-arm worker tracking for a fresh ``run()`` after a previous
        close drained every recorded worker."""
        with self._workers_lock:
            self._workers_closed = False


class IdleGatedRunner(AgentRunner):
    """AgentRunner that only claims work while the machine is spare."""

    def __init__(
        self,
        client: Any,
        config: AgentConfig,
        status: WorkerStatus,
        runtime: "WorkerRuntime | None" = None,
    ):
        super().__init__(client, config)
        self.client = _OutcomeClient(client)
        self.status = status
        self.idle_only = False
        self.paused_event = threading.Event()
        self.idle_after_seconds = 300.0
        self._runtime = runtime
        if runtime is not None:
            self.client.registration_hook = self._register_with_runtime

    def _register_with_runtime(self, _stale: AgentRegistration) -> None:
        """Serviced instead of the upstream initial ``client.register``.

        AgentConfig is frozen and captured at construction; dynamic mode may
        have re-provisioned capabilities/model packs before run() — or while
        this call is in flight. The runtime's transaction therefore holds
        ``runtime.lock`` across snapshot *and* send: the posted registration
        is the current inventory, and no concurrent update can be clobbered
        by a stale in-flight register. The config the runner then uses for
        heartbeats and slot threads is replaced with the registration that
        was actually sent. A rejected registration propagates as AgentError
        like the upstream path. Ignores the stale config payload it is
        invoked with; a runtime-less runner never installs this hook.
        """
        runtime = self._runtime
        assert runtime is not None  # hook is only installed when bound
        sent = runtime.reregister()
        self.config = replace(self.config, registration=sent)

    def run(self, *, once: bool = False) -> None:
        """Upstream run, then a full join of inherited request workers.

        Upstream's finally sets the heartbeat stop event and performs only
        timed joins on the heartbeat/lease threads — a worker can still be
        inside a client request when run() returns or raises, and the CLI
        would then close the shared client beneath it. Every stop event is
        already set at that point, so joining the recorded workers here is
        bounded by the client's request timeout and guarantees no request
        outlives run(). The terminal-report path is fenced too: each job's
        lease worker is joined before its complete/fail post inside the
        client wrapper. Runs on error exits as well, via finally.
        """
        self.client.open_workers()
        try:
            super().run(once=once)
        finally:
            self.client.join_workers()

    def _idle_ok(self) -> bool:
        state = idle.is_idle(self.idle_after_seconds)
        self.status.idle_state = state.reason
        return state.idle

    def _slot_loop(self, slot: SlotInventory, once: bool) -> None:
        """Upstream claim loop plus gating, status, and outcome accounting.

        Like upstream, AgentError from claim *or* job lifecycle (confirm,
        completion/failure reporting) is recoverable in continuous mode:
        the slot notes the failure and polls again instead of stopping every
        slot. ``once`` mode still propagates so the CLI reports the failure.
        """
        while not self.stop.is_set():
            if self.paused_event.is_set():
                self.status.slot_state(slot.id, "paused")
                if once:
                    return
                self.stop.wait(2.0)
                continue
            if self.idle_only and not self._idle_ok():
                self.status.slot_state(slot.id, "waiting-idle")
                if once:
                    return
                self.stop.wait(5.0)
                continue
            job = None
            try:
                job = self.client.claim(slot.id, self.config.lease_seconds)
                if job is not None:
                    self.status.job_started(slot.id, job)
                    self._run_job(slot, job)
            except AgentError as exc:
                if once:
                    raise
                self.status.note(f"slot {slot.id}: {exc}")
                self.stop.wait(self.config.poll_seconds)
            finally:
                if job is not None:
                    self.status.job_finished(slot.id, job, self.client.pop_outcome() or "failed")
            if job is None:
                if once:
                    return
                self.status.slot_state(slot.id, "idle")
                self.stop.wait(self.config.poll_seconds)
                continue
            if once:
                return

    def _execute(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: Any,
    ) -> list[dict[str, Any]]:
        """Track this job's lease worker, then run the inherited executor.

        Upstream ``_run_job`` creates and starts the ``_LeaseKeeper`` just
        before dispatching here, and its worker thread is the only source of
        ``heartbeat_job`` calls for the job. Registering it hands the
        terminal-report fence the actual thread/stop event to join — no
        per-job tombstone needed and the bookkeeping is reclaimed when the
        joined worker provably cannot heartbeat again.
        """
        self.client.track_lease(lease)
        return super()._execute(slot, job, workdir, lease)

    # -- process/container executor --------------------------------------

    def _execute_process(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: Any,
        *,
        container: bool,
    ) -> list[dict[str, Any]]:
        """One executor for all vendors; only GPU wiring differs.

        Semantics match upstream: request.json + SPAREHOUR_* environment,
        {agent_python} substitution for host commands, lease-checked polling
        against a deadline, log-tail error reporting, outputs.json manifest.
        Vendor wiring:
          nvidia   CUDA_VISIBLE_DEVICES=index; container gets --gpus device=
          amd      ROCR_VISIBLE_DEVICES=<stable GPU uuid>, HIP_VISIBLE_DEVICES=0
                   (index zero into the ROCr-filtered set), CUDA hidden;
                   container gets /dev/kfd + /dev/dri + video/render groups
          cpu      all GPU-visibility variables cleared; container gets no
                   device flags and NVIDIA_VISIBLE_DEVICES=void so the
                   nvidia-container runtime cannot inject GPUs
        """
        vendor = str(slot.vendor).casefold()
        if container and not shutil.which("docker"):
            raise ExecutionError("executor_unavailable", "docker is required for container routes")
        if vendor == "amd":
            if str(slot.gpu_uuid).startswith("pci-"):
                raise ExecutionError(
                    "executor_unavailable",
                    f"AMD slot {slot.id} has no stable GPU UUID (fallback "
                    f"{slot.gpu_uuid!r}); ROCR_VISIBLE_DEVICES requires the "
                    "sysfs unique_id — cannot select this GPU safely",
                )
            self._check_rocm(container)
        executor = job["executor"]
        request_path = workdir / "request.json"
        request_path.write_text(
            json.dumps(_process_request(job, workdir, container=container), sort_keys=True),
            encoding="utf-8",
        )
        output_manifest = workdir / "outputs.json"
        environment = {
            **os.environ,
            **executor.get("environment", {}),
            **self._device_environment(slot),
            "SPAREHOUR_JOB_ID": job["id"],
            "SPAREHOUR_REQUEST_JSON": str(request_path),
            "SPAREHOUR_OUTPUT_DIR": str(workdir),
            "SPAREHOUR_OUTPUT_MANIFEST": str(output_manifest),
        }
        command = [
            sys.executable if argument == "{agent_python}" else argument
            for argument in executor["command"]
        ]
        container_name = ""
        if container:
            container_name = f"{_CONTAINER_PREFIX}{job['id'][:48]}"
            container_environment = []
            for name in sorted(executor.get("environment", {})):
                container_environment.extend(["-e", name])
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                *self._container_device_args(slot),
                "--mount",
                f"type=bind,src={workdir},dst=/job",
                "-e",
                "SPAREHOUR_JOB_ID",
                "-e",
                "SPAREHOUR_REQUEST_JSON=/job/request.json",
                "-e",
                "SPAREHOUR_OUTPUT_DIR=/job",
                "-e",
                "SPAREHOUR_OUTPUT_MANIFEST=/job/outputs.json",
                *container_environment,
                executor["image"],
                *executor["command"],
            ]
        stdout_path = workdir / "executor.stdout.log"
        stderr_path = workdir / "executor.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            timeout = float(executor.get("timeout_seconds") or 3600)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    lease.check()
                    wait = min(1.0, max(0.0, deadline - time.monotonic()))
                    if wait <= 0:
                        raise ExecutionError(
                            "execution_timeout", f"executor exceeded {timeout:g}s"
                        )
                    try:
                        process.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                # The container is daemon-owned: kill the named container
                # itself, then the local `docker run` process group.
                if container_name:
                    _stop_container(container_name)
                _terminate_process(process)
                raise
        if process.returncode != 0:
            stdout = _tail(stdout_path)
            stderr = _tail(stderr_path)
            message = (stderr or stdout or f"exit code {process.returncode}")[-3000:]
            code = "gpu_oom" if "out of memory" in message.casefold() else "executor_failed"
            raise ExecutionError(code, message)
        return _load_output_manifest(output_manifest, workdir)

    @staticmethod
    def _device_environment(slot: SlotInventory) -> dict[str, str]:
        vendor = str(slot.vendor).casefold()
        if vendor == "amd":
            # ROCR_VISIBLE_DEVICES selects by the stable GPU uuid (not the
            # PCI ordinal); HIP_VISIBLE_DEVICES then indexes the filtered
            # set — always 0 for the single-slot assignment.
            return {
                "CUDA_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": slot.gpu_uuid,
                "HIP_VISIBLE_DEVICES": "0",
            }
        if vendor == "cpu":
            return {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            }
        return {"CUDA_VISIBLE_DEVICES": str(slot.gpu_index)}

    @staticmethod
    def _container_device_args(slot: SlotInventory) -> list[str]:
        vendor = str(slot.vendor).casefold()
        if vendor == "amd":
            return [
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video",
                "--group-add",
                "render",
                "-e",
                "ROCR_VISIBLE_DEVICES",
                "-e",
                "HIP_VISIBLE_DEVICES",
            ]
        if vendor == "cpu":
            # No device flags, and belt-and-suspenders against the
            # nvidia-container runtime injecting every GPU when it sees the
            # toolkit on the host.
            return ["-e", "NVIDIA_VISIBLE_DEVICES=void"]
        return ["--gpus", f"device={slot.gpu_uuid}"]

    @staticmethod
    def _check_rocm(container: bool) -> None:
        missing = [path for path in ("/dev/kfd", "/dev/dri") if not Path(path).exists()]
        if missing:
            raise ExecutionError(
                "executor_unavailable",
                f"ROCm device nodes missing ({', '.join(missing)}); "
                "the amdgpu driver must be loaded for AMD GPU execution",
            )
        if container and not os.access("/dev/kfd", os.R_OK | os.W_OK):
            raise ExecutionError(
                "executor_unavailable", "/dev/kfd exists but is not accessible by this user"
            )


def _stop_container(name: str) -> None:
    """Force-remove a named workload container; best-effort, bounded."""
    for args in (["stop", "-t", "2"], ["rm", "-f"]):
        try:
            subprocess.run(
                ["docker", *args, name],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue


def _terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            process.wait()


def _tail(path: Path, max_bytes: int = 16 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _prefetch_error_summary(exc: Exception) -> str:
    """Classified one-line cause for a failed queue listing.

    Exception text may echo the remote response body or embed credentials,
    so only the exception type and an HTTP status (when the message carries
    one, e.g. ``workload listing failed: HTTP 503: ...``) are surfaced.
    """
    code = getattr(exc, "code", None)
    summary = code if isinstance(code, str) and code else type(exc).__name__
    status = re.search(r"HTTP (\d{3})", str(exc))
    if status:
        summary = f"{summary}, HTTP {status.group(1)}"
    return summary[:120]


class WorkerRuntime:
    """Everything the CLI and dashboard need, bound to one config file.

    Threading: ``lock`` (an RLock) serializes capability/model-pack changes,
    slot rebuilds, registrations, and config persistence. ``stop`` is a
    runtime-level cancellation event observed by prefetch and long-running
    background work; ``close()`` sets it so in-flight work cancels
    cooperatively before the client is closed.

    ``saved_overrides`` carries the pre-override values for any config keys
    the caller (CLI) rewrote for this run only; ``save_config()`` writes
    them back so runtime-only overrides never reach disk.
    """

    def __init__(
        self,
        config: dict[str, Any],
        config_file: Path,
        *,
        saved_overrides: dict[str, Any] | None = None,
    ):
        self.config = config
        self.config_file = config_file
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.client = WorkerClient(config["endpoint"], config["token"])
        try:
            self._init_after_client(saved_overrides)
        except Exception:
            # Nobody else holds the client yet; only the constructor can
            # release it on a partial-construction failure.
            self.client.close()
            raise

    def _init_after_client(self, saved_overrides: dict[str, Any] | None) -> None:
        config = self.config
        workdir = Path(config["workdir"]).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        self.host = hardware.host_resources(workdir)
        self.host["rocm_kfd"] = Path("/dev/kfd").exists()
        self.host["rocm_dri"] = Path("/dev/dri").exists()
        self.slots, self.hardware_notes = hardware.discover_slots(
            config["agent_id"],
            capabilities=list(config["capabilities"]),
            model_packs=list(config["model_packs"]),
            labels=dict(config["labels"]),
            cpu_slots=int(config["cpu_slots"]),
        )
        # The operator's persisted selection: dynamic mode may diverge
        # config["capabilities"] at runtime, but save_config() must write
        # back only what the operator saved. saved_overrides restores any
        # other keys the caller rewrote for this run only.
        self._saved_capabilities = list(config["capabilities"])
        self._saved_overrides = dict(saved_overrides or {})
        self.status = WorkerStatus(agent_id=config["agent_id"])
        self._queues_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])

    # -- registration -----------------------------------------------------

    def registration(self) -> AgentRegistration:
        with self.lock:
            return AgentRegistration(
                agent_id=self.config["agent_id"],
                provider=self.config["provider"],
                hostname=os.uname().nodename if hasattr(os, "uname") else "localhost",
                cpu_cores=self.host["cpu_cores"],
                ram_mb=int(self.host["ram_mb"]),
                disk_free_mb=int(self.host["disk_free_mb"]),
                labels=dict(self.config["labels"]),
                slots=list(self.slots),
            )

    def reregister(self) -> AgentRegistration:
        """Register the *current* inventory atomically; return what was sent.

        Snapshot (``self.registration()``) and send both happen inside
        ``self.lock``, so no capability/model-pack mutation can interleave:
        a dynamic update either lands first (and is what gets sent) or waits
        and then publishes newer state itself. Used by live re-registration
        and by IdleGatedRunner's initial registration hook.
        """
        with self.lock:
            registration = self.registration()
            self.client.register(registration)
            return registration

    # -- persistence ------------------------------------------------------

    def save_config(self) -> Path:
        """Persist config, writing the saved (not runtime/dynamic) selection."""
        with self.lock:
            payload = dict(self.config)
            payload["capabilities"] = list(self._saved_capabilities)
            payload.update(self._saved_overrides)
            return config_mod.save(payload, self.config_file)

    # -- capabilities / model packs ----------------------------------------

    def set_capability(self, capability: str, selected: bool) -> tuple[bool, str]:
        return self.set_capability_list([capability], selected)

    def set_capability_list(self, capabilities: list[str], selected: bool) -> tuple[bool, str]:
        with self.lock:
            current = list(self.config["capabilities"])
            if selected:
                wanted = list(dict.fromkeys(current + capabilities))
            else:
                wanted = [item for item in current if item not in set(capabilities)]
            if wanted == current:
                return True, f"capabilities unchanged: {wanted or 'none'}"
            return self.apply_capability_list(wanted, persist=True)

    def apply_capability_list(self, capabilities: list[str], *, persist: bool) -> tuple[bool, str]:
        """Set the advertised capabilities and re-register live.

        Running jobs are never interrupted (busy slots are preserved by the
        control plane). persist=False keeps the selection runtime-only — used
        by dynamic mode, which re-decides every cycle and must not rewrite
        the operator's saved selection on exit.
        """
        with self.lock:
            previous = list(self.config["capabilities"])
            wanted = list(dict.fromkeys(capabilities))
            if wanted == previous:
                return True, f"capabilities unchanged: {wanted or 'none'}"
            ok, message = self._apply_locked(wanted, list(self.config["model_packs"]))
            if not ok:
                return ok, message
            if persist:
                old_saved = self._saved_capabilities
                self._saved_capabilities = wanted
                try:
                    self.save_config()
                except OSError as exc:
                    self._saved_capabilities = old_saved
                    self._apply_locked(previous, list(self.config["model_packs"]))
                    return False, f"registered but could not save config: {exc}"
            self.status.note(f"capabilities now: {wanted or 'none'}")
            return True, f"capabilities now: {wanted or 'none'} (registered)"

    def update_model_packs(self, packs: list[str]) -> tuple[bool, str]:
        """Advertise an updated model-pack set, persist it, re-register.

        Unlike apply_capability_list this always re-registers when the pack
        set changes — even with unchanged capabilities — because the control
        plane only learns pack inventory through registration. On a
        registration failure the local pack list and slot inventory roll
        back so a retry stays possible. Persistence writes the saved
        capability selection, never the dynamic runtime one.
        """
        with self.lock:
            wanted = sorted(dict.fromkeys(str(pack) for pack in packs))
            previous = list(self.config["model_packs"])
            if wanted == previous:
                return True, f"model packs unchanged: {wanted or 'none'}"
            ok, message = self._apply_locked(list(self.config["capabilities"]), wanted)
            if not ok:
                return False, message
            try:
                self.save_config()
            except OSError as exc:
                self._apply_locked(list(self.config["capabilities"]), previous)
                return False, f"registered but could not save config: {exc}"
            self.status.note(f"model packs now: {wanted or 'none'}")
            return True, f"model packs now: {wanted or 'none'} (registered)"

    def _apply_locked(self, capabilities: list[str], model_packs: list[str]) -> tuple[bool, str]:
        """Mutate inventory, rebuild slots, re-register; rollback on failure.

        Caller must hold ``self.lock``. On registration failure the prior
        config values and slot inventory are restored.
        """
        previous_caps = list(self.config["capabilities"])
        previous_packs = list(self.config["model_packs"])
        previous_slots = self.slots
        self.config["capabilities"] = list(capabilities)
        self.config["model_packs"] = list(model_packs)
        self._rebuild_slots()
        try:
            self.reregister()
        except AgentError as exc:
            self.config["capabilities"] = previous_caps
            self.config["model_packs"] = previous_packs
            self.slots = previous_slots
            return False, f"control plane rejected registration: {exc}"
        return True, "registered"

    def _rebuild_slots(self) -> None:
        self.slots, _ = hardware.discover_slots(
            self.config["agent_id"],
            capabilities=list(self.config["capabilities"]),
            model_packs=list(self.config["model_packs"]),
            labels=dict(self.config["labels"]),
            cpu_slots=int(self.config["cpu_slots"]),
        )

    # -- queues -----------------------------------------------------------

    def queues_report(self, max_age: float = 30.0) -> list[dict[str, Any]]:
        cached_at, cached = self._queues_cache
        if time.time() - cached_at > max_age:
            queues = queues_mod.expand_queues(queues_mod.fetch_workloads(self.client))
            self._queues_cache = (time.time(), queues)
            cached = queues
        with self.lock:
            selected = set(self.config["capabilities"])
            slots = list(self.slots)
            provider = str(self.config["provider"])
            agent_labels = dict(self.config["labels"])
        report = []
        for queue in cached:
            requirements = queues_mod.requirements_for(queue)
            best: tuple[bool, str] | None = None
            for slot in slots:
                match = queues_mod.route_matches(
                    requirements,
                    slot.model_dump(),
                    provider=provider,
                    host=self.host,
                    agent_labels=agent_labels,
                )
                if best is None or (match[0] and not best[0]):
                    best = match
                if best[0]:
                    break
            ok, reason = best or (False, "no slots")
            compatible, compat_reason = queues_mod.hardware_compatibility(
                queue, slots, provider=provider, host=self.host, agent_labels=agent_labels,
            )
            report.append(
                {
                    **queue,
                    "selected": queue["capability"] in selected,
                    "eligible": ok,
                    "match_reason": reason,
                    "compatible": compatible,
                    "compat_reason": compat_reason,
                }
            )
        return report

    # -- prefetch ---------------------------------------------------------

    def prefetch_images(self, *, pull_timeout: float = PREFETCH_PULL_TIMEOUT) -> list[str]:
        """docker-pull every digest-pinned image behind selected capabilities.

        Runs synchronously; concurrency is owned by the caller (the dashboard
        runs it on a single-flight background thread). Cancellable via
        ``self.stop``; each pull is bounded by pull_timeout.
        """
        lines: list[str] = []
        images: set[str] = set()
        others: set[str] = set()
        with self.lock:
            self.status.prefetch_running = True
        listing_error = ""
        try:
            try:
                queues = self.queues_report(max_age=0.0)
            except Exception as exc:
                # Network/HTTP/parse failures here must not kill the daemon
                # thread silently: record a classified line so the dashboard
                # shows the outage. Exception text can echo the remote
                # response body or the agent token — sanitize, never log raw.
                listing_error = _prefetch_error_summary(exc)
                lines.append(f"FAILED: could not list queues: {listing_error}")
                queues = []
            for queue in queues:
                if queue["capability"] not in set(self.config["capabilities"]):
                    continue
                if queue["executor_type"] == "container" and queue.get("executor_detail"):
                    images.add(queue["executor_detail"])
                elif queue["executor_type"] == "plugin":
                    others.add(
                        f"{queue['workload']}/{queue['route']}: plugin {queue['executor_detail']} "
                        f"(install its model pack manually, then sparehour-worker model-packs add)"
                    )
                elif queue["executor_type"] == "http":
                    others.add(
                        f"{queue['workload']}/{queue['route']}: needs a local model server "
                        f"at {queue['executor_detail']}"
                    )
            if images and not shutil.which("docker"):
                lines.append("docker not found — container routes cannot run/prefetch here")
            elif not shutil.which("docker"):
                lines.append("docker not found — nothing to pull")
            else:
                for image in sorted(images):
                    if self.stop.is_set():
                        lines.append(f"pull {image}: skipped (worker stopping)")
                        continue
                    verdict = self._docker_pull(image, pull_timeout)
                    lines.append(f"pull {image}: {verdict}")
            lines.extend(f"note: {line}" for line in sorted(others))
            if not lines:
                lines.append("nothing to prefetch for the current capability selection")
            with self.lock:
                self.status.prefetch_log = [f"{_now()} {line}" for line in lines]
            self.status.note(
                "prefetch finished" if not listing_error
                else f"prefetch failed: {listing_error}"
            )
            return lines
        finally:
            with self.lock:
                self.status.prefetch_running = False

    def _docker_pull(self, image: str, timeout: float) -> str:
        try:
            process = subprocess.Popen(
                ["docker", "pull", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return f"FAILED: docker not runnable: {exc}"
        deadline = time.monotonic() + max(1.0, timeout)
        stderr = ""
        while True:
            if self.stop.is_set():
                _terminate_process(process)
                return "cancelled (worker stopping)"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                return f"FAILED: pull exceeded {timeout:g}s"
            try:
                _, stderr = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode == 0:
            return "ok"
        return f"FAILED: {(stderr or '').strip()[-200:] or f'exit {process.returncode}'}"

    def close(self) -> None:
        self.stop.set()
        self.client.close()
