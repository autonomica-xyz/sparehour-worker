"""Hardware discovery: NVIDIA (nvidia-smi), AMD (sysfs), and CPU-only slots.

The control plane accepts any vendor string; workloads pick the vendor they
need (`nvidia`, `amd`, `cpu`). One slot per physical GPU plus optional
synthetic CPU slots for spare compute on GPU-less machines.

Registration semantics:
- AMD `gpu_uuid` is `GPU-<16 hex>` derived from sysfs `unique_id` — the same
  GPU UUID the ROCm stack prints (rocm-smi/rocminfo) and ROCR_VISIBLE_DEVICES
  accepts. Cards without a nonzero unique_id keep a `pci-<BDF>` identity;
  that address is discoverable but is NOT a ROCm device selector, so the
  runner must refuse ROCm execution for it rather than substitute an
  ordinal. `gpu_index` is only the filtered sysfs enumeration order, never
  a ROCm index.
- Reported cores/RAM are *usable* capacity: process CPU affinity and
  cgroup v1/v2 cpu/memory limits cap the host-wide readings.
- CPU slots partition host RAM (total / slots) instead of every slot
  claiming all of it.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from sparehour.agent import AgentError, discover_nvidia_slots
from sparehour.models import SlotInventory

_SYS_DRM = Path("/sys/class/drm")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_MOUNTS = Path("/proc/mounts")
_PROC_CGROUP = Path("/proc/self/cgroup")


def host_resources(workdir: Path | None = None) -> dict[str, float]:
    """Usable CPU cores, RAM MB, and free disk MB for registration.

    Cores/RAM reflect CPU affinity and cgroup v1/v2 limits so a constrained
    worker (container, systemd slice, taskset) does not advertise capacity
    it cannot use.
    """
    cpu_quota, mem_bytes = _cgroup_limits()
    cpu = _effective_cpu_cores()
    if cpu_quota is not None:
        cpu = min(cpu, cpu_quota)
    ram = _total_ram_mb()
    if mem_bytes is not None and mem_bytes > 0:
        limit_mb = int(mem_bytes) // (1024 * 1024)
        ram = min(ram, limit_mb) if ram > 0 else limit_mb
    free = shutil.disk_usage(workdir or Path.home()).free // (1024 * 1024)
    return {"cpu_cores": cpu, "ram_mb": ram, "disk_free_mb": int(free)}


def _effective_cpu_cores() -> float:
    """Scheduler-visible CPU count: affinity mask when narrower."""
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = 0
    detected = os.cpu_count() or 1
    usable = affinity if affinity > 0 else detected
    return float(max(1, usable))


def _total_ram_mb() -> int:
    if sys.platform == "win32":
        win = _windows_ram_mb()
        if win is not None:
            return win
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
            return int(out) // (1024 * 1024)
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0
    if os.path.exists("/proc/meminfo"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 0


def _windows_ram_mb() -> int | None:
    """Physical RAM via GlobalMemoryStatusEx; None when unavailable."""

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    try:
        memory_status = ctypes.windll.kernel32.GlobalMemoryStatusEx  # type: ignore[attr-defined]
        memory_status.restype = ctypes.c_bool
        memory_status.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not memory_status(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys) // (1024 * 1024)
    except Exception:
        return None


# -- cgroup limits ---------------------------------------------------------


def _read_limited(path: Path, parse: Callable[[str], float | None]) -> float | None:
    """parse() the file's text; None when unreadable or unconstrained."""
    try:
        return parse(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _v2_cpu_limit(text: str) -> float | None:
    parts = text.split()
    if len(parts) < 2 or parts[0] == "max":
        return None
    try:
        quota, period = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _v1_cpu_limit(quota_text: str, period_text: str | None) -> float | None:
    try:
        quota = int(quota_text)
    except ValueError:
        return None
    if quota <= 0:
        return None
    try:
        period = int(period_text) if period_text else 100_000
    except ValueError:
        period = 100_000
    if period <= 0:
        period = 100_000
    return quota / period


def _v2_mem_limit(text: str) -> float | None:
    if text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return float(value) if value > 0 else None


def _v1_mem_limit(text: str) -> float | None:
    try:
        value = int(text)
    except ValueError:
        return None
    # kernels report an enormous sentinel when unconstrained
    return float(value) if 0 < value < (1 << 60) else None


def _cgroup_mounts(mounts: Path) -> tuple[Path | None, list[tuple[set[str], Path]]]:
    """(cgroup v2 unified mount, [(v1 controller set, hierarchy root)])."""
    v2_root: Path | None = None
    v1: list[tuple[set[str], Path]] = []
    try:
        lines = mounts.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return v2_root, v1
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        destination = Path(fields[1])
        if fields[2] == "cgroup2":
            v2_root = destination
        elif fields[2] == "cgroup" and len(fields) >= 4:
            options = fields[3].split(",")
            v1.append(({opt for opt in options if opt and opt != "rw"}, destination))
    return v2_root, v1


def _self_cgroup_rels(proc_cgroup: Path) -> list[tuple[frozenset[str], str]]:
    """(controller set, relative cgroup path) entries from /proc/self/cgroup."""
    rels: list[tuple[frozenset[str], str]] = []
    try:
        lines = proc_cgroup.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rels
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = frozenset(c for c in parts[1].split(",") if c)
        rels.append((controllers, parts[2]))
    return rels


def _ancestors(node: Path, root: Path) -> list[Path]:
    """node plus every ancestor up to and including the hierarchy root."""
    chain: list[Path] = []
    while True:
        chain.append(node)
        if node == node.parent or node == root:
            break
        node = node.parent
    return chain


def _cgroup_limits(
    *,
    root: Path | None = None,
    mounts: Path | None = None,
    proc_cgroup: Path | None = None,
) -> tuple[float | None, int | None]:
    """(cpu quota in cores, memory limit in bytes) — each None when unlimited.

    Checks both the cgroup v2 unified hierarchy (cpu.max / memory.max) and
    v1 hierarchies (cpu.cfs_quota_us/cpu.cfs_period_us, memory.limit_in_bytes).
    Ancestor limits apply to descendants, so the minimum along the chain wins.
    """
    root = root or _CGROUP_ROOT
    mounts = mounts or _MOUNTS
    proc_cgroup = proc_cgroup or _PROC_CGROUP
    v2_root, v1 = _cgroup_mounts(mounts)
    if v2_root is None and (root / "cgroup.controllers").is_file():
        v2_root = root
    rels = _self_cgroup_rels(proc_cgroup)
    cpu_limits: list[float] = []
    mem_limits: list[float] = []

    unified_rel = "/"
    for controllers, path in rels:
        if not controllers:
            unified_rel = path
            break

    if v2_root is not None:
        nodes = _ancestors(v2_root / unified_rel.lstrip("/"), v2_root)
        for node in nodes:
            value = _read_limited(node / "cpu.max", _v2_cpu_limit)
            if value is not None:
                cpu_limits.append(value)
            value = _read_limited(node / "memory.max", _v2_mem_limit)
            if value is not None:
                mem_limits.append(value)

    for options, hierarchy in v1:
        rel = "/"
        for controllers, path in rels:
            if controllers & options:
                rel = path
                break
        nodes = _ancestors(hierarchy / rel.lstrip("/"), hierarchy)
        if "cpu" in options:
            for node in nodes:
                quota_text: str | None = None
                try:
                    quota_text = (node / "cpu.cfs_quota_us").read_text(
                        encoding="utf-8"
                    ).strip()
                except OSError:
                    continue
                try:
                    period_text: str | None = (node / "cpu.cfs_period_us").read_text(
                        encoding="utf-8"
                    ).strip()
                except OSError:
                    period_text = None
                value = _v1_cpu_limit(quota_text, period_text)
                if value is not None:
                    cpu_limits.append(value)
        if "memory" in options:
            for node in nodes:
                value = _read_limited(node / "memory.limit_in_bytes", _v1_mem_limit)
                if value is not None:
                    mem_limits.append(value)

    return (min(cpu_limits) if cpu_limits else None,
            int(min(mem_limits)) if mem_limits else None)



# -- slot discovery --------------------------------------------------------


def discover_slots(
    agent_id: str,
    *,
    capabilities: list[str],
    model_packs: list[str],
    labels: dict[str, str],
    cpu_slots: int,
) -> tuple[list[SlotInventory], list[str]]:
    """All slots plus human-readable notes about what was found/skipped."""
    notes: list[str] = []
    slots: list[SlotInventory] = []
    try:
        nvidia = discover_nvidia_slots(
            agent_id, capabilities=capabilities, model_packs=model_packs, labels=labels
        )
        slots += nvidia
        notes.append(f"nvidia: {len(nvidia)} GPU slot(s)")
    except AgentError as exc:
        notes.append(f"nvidia: none ({exc})")
    except Exception as exc:
        # Malformed nvidia-smi output (bad row shape, non-numeric fields,
        # failed model validation) must not cost us AMD/CPU discovery.
        notes.append(f"nvidia: none (invalid nvidia-smi output: {exc})")
    amd = discover_amd_slots(
        agent_id, capabilities=capabilities, model_packs=model_packs, labels=labels
    )
    slots += amd
    notes.append(f"amd: {len(amd)} GPU slot(s)")
    if amd:
        if not Path("/dev/kfd").exists():
            notes.append(
                "amd: /dev/kfd missing — ROCm compute is unavailable until the "
                "amdgpu driver exposes it (jobs on amd slots will fail to start)"
            )
        if not (shutil.which("rocminfo") or shutil.which("rocm-smi")):
            notes.append(
                "amd: no ROCm tools (rocminfo/rocm-smi) on PATH — install ROCm "
                "before advertising amd capability"
            )
        for slot in amd:
            if not str(slot.gpu_uuid).startswith("GPU-"):
                notes.append(
                    f"amd: {slot.id} has no sysfs unique_id ({slot.gpu_uuid}) — "
                    "install/enable amdgpu with unique_id support or exclude this "
                    "card; ROCm jobs cannot be mapped to it safely"
                )
    total_ram = int(host_resources()["ram_mb"])
    per_slot_ram = max(1, total_ram // cpu_slots) if cpu_slots > 0 else 0
    for index in range(cpu_slots):
        slots.append(
            SlotInventory(
                id=f"{agent_id}-cpu-{index}",
                gpu_index=index,
                gpu_uuid=f"{agent_id}-cpu-{index}",
                vendor="cpu",
                model=f"cpu ({os.uname().machine})" if hasattr(os, "uname") else "cpu",
                # SlotInventory requires vram_mb > 0; for CPU slots it is the
                # slot's RAM share so vendor=cpu routes get a matching signal.
                # Slots run concurrently, so each advertises RAM/cpu_slots
                # rather than the whole host.
                vram_mb=per_slot_ram,
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels=dict(labels),
            )
        )
    if cpu_slots:
        notes.append(
            f"cpu: {cpu_slots} slot(s), {per_slot_ram} MB RAM each (vendor=cpu)"
        )
    return slots, notes


def discover_amd_slots(
    agent_id: str,
    *,
    capabilities: list[str],
    model_packs: list[str],
    labels: dict[str, str],
    sysfs_root: Path | None = None,
) -> list[SlotInventory]:
    """AMD GPUs from the amdgpu sysfs tree; ROCm not required to register.

    Cards with a nonzero sysfs ``unique_id`` get a stable ``GPU-<16 hex>``
    ``gpu_uuid`` — the same GPU UUID ROCm tools display and
    ROCR_VISIBLE_DEVICES accepts. Without it the card falls back to a
    ``pci-<BDF>`` identity: it stays discoverable, but it is not a ROCm
    device selector and the runner must fail rather than guess an index.
    ``gpu_index`` here is only the filtered sysfs enumeration order.
    """
    sysfs_root = sysfs_root if sysfs_root is not None else _SYS_DRM
    slots: list[SlotInventory] = []
    for card in sorted(sysfs_root.glob("card[0-9]*")):
        device = card / "device"
        if not (device / "vendor").is_file():
            continue
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            klass = (device / "class").read_text().strip()
            if not klass.startswith("0x03"):
                continue  # not a display/GPU device
        except OSError:
            continue
        vram = _read_int(device / "mem_info_vram_total") // (1024 * 1024)
        if vram <= 0:
            continue  # not an amdgpu-managed GPU (e.g. iGPU without driver info)
        model = _amd_model_name(device)
        uuid = _amd_gpu_uuid(device)
        slots.append(
            SlotInventory(
                id=f"{agent_id}-amd-{len(slots)}",
                gpu_index=len(slots),
                gpu_uuid=uuid,
                vendor="amd",
                model=model,
                vram_mb=vram,
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels=dict(labels),
            )
        )
    return slots


def _amd_gpu_uuid(device: Path) -> str:
    """Stable ``GPU-<16 hex>`` from sysfs unique_id, else ``pci-<BDF>``.

    unique_id is the same value ROCm calls the GPU UUID; a zero or absent
    value means the driver did not assign one, so the PCI bus address keeps
    the card discoverable (and debuggable) without pretending to be a ROCm
    identity.
    """
    unique = _read_text(device / "unique_id")
    if unique:
        try:
            value = int(unique, 16) & 0xFFFFFFFFFFFFFFFF
        except ValueError:
            value = 0
        if value:
            return f"GPU-{value:016x}"
    return f"pci-{device.resolve().name}"


def _amd_model_name(device: Path) -> str:
    name = _read_text(device / "product_name")
    if name and not name.startswith("0x"):
        return name
    vendor_id = _read_text(device / "vendor") or "0x1002"
    device_id = _read_text(device / "device") or "unknown"
    return f"AMD GPU {vendor_id}:{device_id}"


def _read_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
        return value or None
    except OSError:
        return None


def _read_int(path: Path) -> int:
    value = _read_text(path)
    try:
        return int(value or "0", 0)
    except ValueError:
        return 0
