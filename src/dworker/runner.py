"""Worker runtime: registration, idle-gated claim loop, prefetch, status.

Reuses the control plane's AgentRunner (claim/confirm/heartbeat/execute/
upload/complete lifecycle and the command/container/http/plugin executors).
This subclass only adds:
- idle/lock gating before claiming (SETI@home mode),
- an AMD-friendly process/container executor (ROCm visibility + device
  passthrough instead of nvidia-container `--gpus`),
- a live status snapshot for the dashboard,
- live capability re-registration (the control plane preserves busy slots).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_orchestrator.agent import (
    AgentConfig,
    AgentError,
    AgentRunner,
    ControlPlaneClient,
    ExecutionError,
)
from gpu_orchestrator.models import AgentRegistration, SlotInventory

from . import hardware, idle, queues as queues_mod


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
    started_jobs: int = 0
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

    def job_finished(self, slot_id: str, job: dict[str, Any]) -> None:
        with self._lock:
            self.completed += 1
        self.slot_state(slot_id, "idle")
        self.note(f"slot {slot_id}: finished job {job.get('id')}")

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
                "started_jobs": self.started_jobs,
                "notes": list(self.notes),
                "prefetch_log": list(self.prefetch_log),
                "uptime_seconds": int(time.time() - self.started_at),
            }


class IdleGatedRunner(AgentRunner):
    """AgentRunner that only claims work while the machine is spare."""

    def __init__(self, client: ControlPlaneClient, config: AgentConfig, status: WorkerStatus):
        super().__init__(client, config)
        self.status = status
        self.idle_only = False
        self.paused_event = threading.Event()
        self.idle_after_seconds = 300.0

    def _idle_ok(self) -> bool:
        state = idle.is_idle(self.idle_after_seconds)
        self.status.idle_state = state.reason
        return state.idle

    def _slot_loop(self, slot: SlotInventory, once: bool) -> None:
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
            try:
                job = self.client.claim(slot.id, self.config.lease_seconds)
            except AgentError as exc:
                if once:
                    raise
                self.status.note(f"slot {slot.id}: claim failed: {exc}")
                self.stop.wait(self.config.poll_seconds)
                continue
            if job is None:
                if once:
                    return
                self.status.slot_state(slot.id, "idle")
                self.stop.wait(self.config.poll_seconds)
                continue
            self.status.job_started(slot.id, job)
            try:
                self._run_job(slot, job)
            finally:
                self.status.job_finished(slot.id, job)
            if once:
                return

    def _execute_process(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: Any,
        *,
        container: bool,
    ) -> list[dict[str, Any]]:
        if str(slot.vendor).casefold() != "amd":
            return super()._execute_process(slot, job, workdir, lease, container=container)
        return self._execute_process_amd(slot, job, workdir, lease, container=container)

    def _execute_process_amd(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: Any,
        *,
        container: bool,
    ) -> list[dict[str, Any]]:
        """AMD variant: ROCm device visibility instead of nvidia-container.

        Duplicates ~30 lines of the parent executor to swap `--gpus` for
        /dev/kfd + /dev/dri passthrough. Upstream into the control-plane
        executor when AMD workers are common.
        """
        executor = job["executor"]
        request_path = workdir / "request.json"
        request_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        output_manifest = workdir / "outputs.json"
        environment = {
            **os.environ,
            **executor.get("environment", {}),
            "CUDA_VISIBLE_DEVICES": str(slot.gpu_index),
            "HIP_VISIBLE_DEVICES": str(slot.gpu_index),
            "ROCR_VISIBLE_DEVICES": str(slot.gpu_index),
            "DGPU_JOB_ID": job["id"],
            "DGPU_REQUEST_JSON": str(request_path),
            "DGPU_OUTPUT_DIR": str(workdir),
            "DGPU_OUTPUT_MANIFEST": str(output_manifest),
        }
        command = [
            sys.executable if argument == "{agent_python}" else argument
            for argument in executor["command"]
        ]
        if container:
            container_environment = []
            for name in sorted(executor.get("environment", {})):
                container_environment.extend(["-e", name])
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"dgpu-{job['id'][:48]}",
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video",
                "--group-add",
                "render",
                "--mount",
                f"type=bind,src={workdir},dst=/job",
                "-e",
                "HIP_VISIBLE_DEVICES",
                "-e",
                "DGPU_JOB_ID",
                "-e",
                "DGPU_REQUEST_JSON=/job/request.json",
                "-e",
                "DGPU_OUTPUT_DIR=/job",
                "-e",
                "DGPU_OUTPUT_MANIFEST=/job/outputs.json",
                *container_environment,
                executor["image"],
                *executor["command"],
            ]
        stdout_path = workdir / "executor.stdout.log"
        stderr_path = workdir / "executor.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdout=out,
                stderr=err,
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
                self._terminate(process)
                raise
        if process.returncode != 0:
            message = _tail(stderr_path) or _tail(stdout_path) or f"exit code {process.returncode}"
            code = "gpu_oom" if "out of memory" in message.casefold() else "executor_failed"
            raise ExecutionError(code, message[-3000:])
        from gpu_orchestrator.agent import _load_output_manifest

        return _load_output_manifest(output_manifest, workdir)

    @staticmethod
    def _terminate(process: subprocess.Popen[str], grace: float = 10.0) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _tail(path: Path, max_bytes: int = 16 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


class WorkerRuntime:
    """Everything the CLI and dashboard need, bound to one config file."""

    def __init__(self, config: dict[str, Any], config_file: Path):
        self.config = config
        self.config_file = config_file
        self.client = ControlPlaneClient(config["endpoint"], config["token"])
        workdir = Path(config["workdir"]).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        self.host = hardware.host_resources(workdir)
        self.slots, self.hardware_notes = hardware.discover_slots(
            config["agent_id"],
            capabilities=list(config["capabilities"]),
            model_packs=list(config["model_packs"]),
            labels=dict(config["labels"]),
            cpu_slots=int(config["cpu_slots"]),
        )
        self.status = WorkerStatus(agent_id=config["agent_id"])
        self._queues_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])

    # -- registration -----------------------------------------------------

    def registration(self) -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.config["agent_id"],
            provider=self.config["provider"],
            hostname=os.uname().nodename if hasattr(os, "uname") else "localhost",
            cpu_cores=self.host["cpu_cores"],
            ram_mb=int(self.host["ram_mb"]),
            disk_free_mb=int(self.host["disk_free_mb"]),
            labels=dict(self.config["labels"]),
            slots=self.slots,
        )

    def reregister(self) -> None:
        """Re-register with current config (safe: busy slots are preserved)."""
        self.client.register(self.registration())

    # -- capabilities -----------------------------------------------------

    def set_capability(self, capability: str, selected: bool) -> tuple[bool, str]:
        return self.set_capability_list([capability], selected)

    def set_capability_list(self, capabilities: list[str], selected: bool) -> tuple[bool, str]:
        current = list(self.config["capabilities"])
        if selected:
            wanted = list(dict.fromkeys(current + capabilities))
        else:
            wanted = [item for item in current if item not in set(capabilities)]
        if wanted == current:
            return True, f"capabilities unchanged: {wanted or 'none'}"
        previous = current
        self.config["capabilities"] = wanted
        self._rebuild_slots()
        try:
            self.reregister()
        except AgentError as exc:
            self.config["capabilities"] = previous
            self._rebuild_slots()
            return False, f"control plane rejected registration: {exc}"
        from . import config as config_mod

        config_mod.save(self.config, self.config_file)
        self.status.note(f"capabilities now: {wanted or 'none'}")
        return True, f"capabilities now: {wanted or 'none'} (registered)"

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
        selected = set(self.config["capabilities"])
        report = []
        for queue in cached:
            requirements = queues_mod.requirements_for(queue)
            best: tuple[bool, str] | None = None
            for slot in self.slots:
                match = queues_mod.route_matches(
                    requirements,
                    slot.model_dump(),
                    provider=self.config["provider"],
                    host=self.host,
                    agent_labels=dict(self.config["labels"]),
                )
                if best is None or (match[0] and not best[0]):
                    best = match
                if best[0]:
                    break
            ok, reason = best or (False, "no slots")
            report.append(
                {
                    **queue,
                    "selected": queue["capability"] in selected,
                    "eligible": ok,
                    "match_reason": reason,
                }
            )
        return report

    # -- prefetch ---------------------------------------------------------

    def prefetch_images(self) -> list[str]:
        """docker-pull every digest-pinned image behind selected capabilities."""
        lines: list[str] = []
        images: set[str] = set()
        others: set[str] = set()
        for queue in self.queues_report(max_age=0.0):
            if queue["capability"] not in set(self.config["capabilities"]):
                continue
            if queue["executor_type"] == "container" and queue.get("executor_detail"):
                images.add(queue["executor_detail"])
            elif queue["executor_type"] == "plugin":
                others.add(
                    f"{queue['workload']}/{queue['route']}: plugin {queue['executor_detail']} "
                    f"(install its model pack manually, then dworker model-packs add)"
                )
            elif queue["executor_type"] == "http":
                others.add(
                    f"{queue['workload']}/{queue['route']}: needs a local model server "
                    f"at {queue['executor_detail']}"
                )
        if not shutil.which("docker"):
            lines.append("docker not found — container routes cannot run/prefetch here")
        for image in sorted(images):
            lines.append(f"pulling {image} ...")
            result = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
            verdict = "ok" if result.returncode == 0 else f"FAILED: {result.stderr.strip()[-200:]}"
            lines[-1] = f"pull {image}: {verdict}"
        lines.extend(f"note: {line}" for line in sorted(others))
        if not lines:
            lines.append("nothing to prefetch for the current capability selection")
        self.status.prefetch_log = [f"{_now()} {line}" for line in lines]
        self.status.note("prefetch finished")
        return lines

    def close(self) -> None:
        self.client.close()
