"""Hardware discovery: NVIDIA (nvidia-smi), AMD (sysfs), and CPU-only slots.

The control plane accepts any vendor string; workloads pick the vendor they
need (`nvidia`, `amd`, `cpu`). One slot per physical GPU plus optional
synthetic CPU slots for spare compute on GPU-less machines.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from gpu_orchestrator.agent import AgentError, discover_nvidia_slots
from gpu_orchestrator.models import SlotInventory

_SYS_DRM = Path("/sys/class/drm")


def host_resources(workdir: Path | None = None) -> dict[str, float]:
    """CPU cores, RAM MB, and free disk MB for the agent registration."""
    cpu = float(os.cpu_count() or 1)
    ram = _total_ram_mb()
    free = shutil.disk_usage(workdir or Path.home()).free // (1024 * 1024)
    return {"cpu_cores": cpu, "ram_mb": ram, "disk_free_mb": int(free)}


def _total_ram_mb() -> int:
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
        slots += discover_nvidia_slots(
            agent_id, capabilities=capabilities, model_packs=model_packs, labels=labels
        )
        notes.append(f"nvidia: {len(slots)} GPU slot(s)")
    except AgentError as exc:
        notes.append(f"nvidia: none ({exc})")
    amd = discover_amd_slots(
        agent_id, capabilities=capabilities, model_packs=model_packs, labels=labels
    )
    slots += amd
    notes.append(f"amd: {len(amd)} GPU slot(s)")
    total_ram = _total_ram_mb()
    for index in range(cpu_slots):
        slots.append(
            SlotInventory(
                id=f"{agent_id}-cpu-{index}",
                gpu_index=index,
                gpu_uuid=f"{agent_id}-cpu-{index}",
                vendor="cpu",
                model=f"cpu ({os.uname().machine})" if hasattr(os, "uname") else "cpu",
                # SlotInventory requires vram_mb > 0; for CPU slots it is the
                # host RAM so vendor=cpu routes have a matching signal.
                vram_mb=max(1, total_ram),
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels=dict(labels),
            )
        )
    if cpu_slots:
        notes.append(f"cpu: {cpu_slots} slot(s) (vendor=cpu)")
    return slots, notes


def discover_amd_slots(
    agent_id: str,
    *,
    capabilities: list[str],
    model_packs: list[str],
    labels: dict[str, str],
    sysfs_root: Path = _SYS_DRM,
) -> list[SlotInventory]:
    """AMD GPUs from the amdgpu sysfs tree; ROCm not required to register.

    Sysfs-first keeps this driver-version independent; if a future kernel
    hides product names, fall back to `rocm-smi --showproductname --csv`.
    """
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
        model = _amd_model_name(device)
        uuid = _read_text(device / "unique_id") or f"pci-{device.resolve().name}"
        if vram <= 0:
            continue  # not an amdgpu-managed GPU (e.g. iGPU without driver info)
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
