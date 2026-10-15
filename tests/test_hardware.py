"""Hardware discovery regressions: identity, capacity limits, fault isolation."""

from __future__ import annotations

import ctypes
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dworker import hardware


def _add_gpu(root: Path, bdf: str, *, unique_id: str | None = None, vram_gb: int = 8) -> None:
    """Mirrors sysfs: cardN/device is a symlink to the PCI device dir."""
    pci = root / "_pci" / bdf
    pci.mkdir(parents=True)
    (pci / "vendor").write_text("0x1002\n")
    (pci / "class").write_text("0x030000\n")
    (pci / "device").write_text("0x744c\n")
    (pci / "mem_info_vram_total").write_text(f"{vram_gb * (1 << 30)}\n")
    if unique_id is not None:
        (pci / "unique_id").write_text(f"{unique_id}\n")
    card = root / f"card{len(list(root.glob('card[0-9]*')))}"
    card.mkdir()
    (card / "device").symlink_to(pci)


# -- AMD identity ------------------------------------------------------------


def test_amd_unique_id_becomes_stable_gpu_uuid(tmp_path: Path):
    _add_gpu(tmp_path, "0000:03:00.0", unique_id="a1b2c3d4e5f60708")
    slots = hardware.discover_amd_slots(
        "tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path
    )
    assert len(slots) == 1
    assert slots[0].gpu_uuid == "GPU-a1b2c3d4e5f60708"


def test_amd_unique_id_is_left_padded(tmp_path: Path):
    _add_gpu(tmp_path, "0000:03:00.0", unique_id="ff12")
    slots = hardware.discover_amd_slots(
        "tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path
    )
    assert slots[0].gpu_uuid == "GPU-000000000000ff12"


@pytest.mark.parametrize("bad", ["0", "0x0", "deadbeef-not-hex"])
def test_amd_zero_or_malformed_unique_id_falls_back_to_pci(tmp_path: Path, bad: str):
    _add_gpu(tmp_path, "0000:03:00.0", unique_id=bad)
    slots = hardware.discover_amd_slots(
        "tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path
    )
    assert slots[0].gpu_uuid == "pci-0000:03:00.0"


def test_amd_missing_unique_id_falls_back_to_pci(tmp_path: Path):
    _add_gpu(tmp_path, "0000:41:00.0")
    slots = hardware.discover_amd_slots(
        "tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path
    )
    assert slots[0].gpu_uuid == "pci-0000:41:00.0"
    # The PCI fallback is deliberately not a ROCm selector prefix.
    assert not slots[0].gpu_uuid.startswith("GPU-")


def test_amd_gpu_index_is_enumeration_order_not_rocm_identity(tmp_path: Path):
    # card2 sorts before card10; gpu_index must track discovered slots, and
    # the BDF/UUID — not the ordinal — is what maps to a device.
    _add_gpu(tmp_path, "0000:41:00.0", unique_id="01")
    _add_gpu(tmp_path, "0000:03:00.0", unique_id="02")
    slots = hardware.discover_amd_slots(
        "tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path
    )
    assert [slot.gpu_index for slot in slots] == [0, 1]
    assert {slot.gpu_uuid for slot in slots} == {
        "GPU-0000000000000001",
        "GPU-0000000000000002",
    }


def test_pci_fallback_note_is_actionable(monkeypatch, tmp_path: Path):
    _add_gpu(tmp_path, "0000:03:00.0")  # no unique_id
    monkeypatch.setattr(hardware, "_SYS_DRM", tmp_path)
    monkeypatch.setattr(
        hardware, "discover_nvidia_slots", lambda *a, **k: (_ for _ in ()).throw(
            hardware.AgentError("no nvidia-smi")
        )
    )
    slots, notes = hardware.discover_slots(
        "tower", capabilities=[], model_packs=[], labels={}, cpu_slots=0
    )
    assert len(slots) == 1
    assert any("unique_id" in note and "cannot be mapped" in note for note in notes)


# -- discovery fault isolation ----------------------------------------------


@pytest.mark.parametrize(
    "runner",
    [
        # non-numeric numeric column -> ValueError inside upstream parsing
        lambda *a, **k: SimpleNamespace(
            stdout="0, GPU-abc, RTX 4090, notanumber, 8.9\n"
        ),
        # negative VRAM -> pydantic validation error
        lambda *a, **k: SimpleNamespace(
            stdout="0, GPU-abc, RTX 4090, -5, 8.9\n"
        ),
        # row with wrong column count -> upstream AgentError
        lambda *a, **k: SimpleNamespace(stdout="garbage\n"),
        # nvidia-smi not installed -> OSError -> upstream AgentError
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    ],
    ids=["bad-vram", "negative-vram", "bad-row", "missing-binary"],
)
def test_malformed_nvidia_discovery_keeps_amd_and_cpu(
    monkeypatch, tmp_path: Path, runner
):
    """Broken nvidia-smi output must not cost us AMD/CPU slots."""
    real_nvidia = hardware.discover_nvidia_slots
    monkeypatch.setattr(
        hardware,
        "discover_nvidia_slots",
        functools.partial(real_nvidia, runner=runner),
    )
    _add_gpu(tmp_path, "0000:03:00.0", unique_id="deadbeef")
    monkeypatch.setattr(hardware, "_SYS_DRM", tmp_path)
    slots, notes = hardware.discover_slots(
        "tower", capabilities=[], model_packs=[], labels={}, cpu_slots=2
    )
    vendors = [slot.vendor for slot in slots]
    assert vendors == ["amd", "cpu", "cpu"]
    assert any(note.startswith("nvidia: none") for note in notes)


# -- capacity: affinity and cgroup limits ------------------------------------


def test_cpu_affinity_caps_reported_cores(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 64)
    assert hardware._effective_cpu_cores() == 2.0


def test_host_resources_reports_affinity_not_host_total(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(hardware, "_total_ram_mb", lambda: 8192)
    monkeypatch.setattr(hardware, "_cgroup_limits", lambda: (None, None))
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=4 * (1 << 30)),
    )
    resources = hardware.host_resources(Path.cwd())
    assert resources["cpu_cores"] == 2.0


def _cgroup_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "cgroup"
    mounts = tmp_path / "mounts"
    proc_cgroup = tmp_path / "proc_cgroup"
    return root, mounts, proc_cgroup


def test_cgroup_v2_limits(monkeypatch, tmp_path: Path):
    root, mounts, proc_cgroup = _cgroup_fixture(tmp_path)
    (root / "sub").mkdir(parents=True)
    (root / "cgroup.controllers").write_text("cpu memory\n")
    (root / "cpu.max").write_text("200000 100000\n")  # 2 cores
    (root / "memory.max").write_text(str(4 * (1 << 30)))
    (root / "sub" / "cpu.max").write_text("100000 100000\n")  # tighter: 1 core
    (root / "sub" / "memory.max").write_text("max")
    mounts.write_text(f"cgroup2 {root} cgroup2 rw 0 0\n")
    proc_cgroup.write_text("0::/sub\n")
    cpu, mem = hardware._cgroup_limits(root=root, mounts=mounts, proc_cgroup=proc_cgroup)
    assert cpu == pytest.approx(1.0)
    assert mem == 4 * (1 << 30)


def test_cgroup_v1_limits(monkeypatch, tmp_path: Path):
    root, mounts, proc_cgroup = _cgroup_fixture(tmp_path)
    cpu_dir = tmp_path / "cgcpu"
    mem_dir = tmp_path / "cgmem"
    (cpu_dir / "u").mkdir(parents=True)
    (mem_dir / "u").mkdir(parents=True)
    (cpu_dir / "u" / "cpu.cfs_quota_us").write_text("150000\n")
    (cpu_dir / "u" / "cpu.cfs_period_us").write_text("100000\n")
    (mem_dir / "u" / "memory.limit_in_bytes").write_text(str(2 * (1 << 30)))
    mounts.write_text(
        f"cgroup {cpu_dir} cgroup rw,cpu,cpuacct 0 0\n"
        f"cgroup {mem_dir} cgroup rw,memory 0 0\n"
    )
    proc_cgroup.write_text("3:cpu,cpuacct:/u\n2:memory:/u\n")
    cpu, mem = hardware._cgroup_limits(root=root, mounts=mounts, proc_cgroup=proc_cgroup)
    assert cpu == pytest.approx(1.5)
    assert mem == 2 * (1 << 30)


def test_cgroup_unlimited_reports_none(tmp_path: Path):
    root, mounts, proc_cgroup = _cgroup_fixture(tmp_path)
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory\n")
    (root / "cpu.max").write_text("max 100000\n")
    (root / "memory.max").write_text("max\n")
    mounts.write_text(f"cgroup2 {root} cgroup2 rw 0 0\n")
    proc_cgroup.write_text("0::/\n")
    assert hardware._cgroup_limits(root=root, mounts=mounts, proc_cgroup=proc_cgroup) == (
        None,
        None,
    )


def test_v1_memory_unconstrained_sentinel_is_none():
    assert hardware._v1_mem_limit("9223372036854771712") is None
    assert hardware._v1_mem_limit(str(4 * (1 << 30))) == float(4 * (1 << 30))
    assert hardware._v1_mem_limit("not-a-number") is None


def test_host_resources_applies_cgroup_ram(monkeypatch):
    monkeypatch.setattr(hardware, "_effective_cpu_cores", lambda: 16.0)
    monkeypatch.setattr(hardware, "_total_ram_mb", lambda: 65536)
    monkeypatch.setattr(hardware, "_cgroup_limits", lambda: (4.0, 8 * (1 << 30)))
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1 << 40),
    )
    resources = hardware.host_resources(Path.cwd())
    assert resources["cpu_cores"] == 4.0
    assert resources["ram_mb"] == 8192


def test_cpu_slots_partition_host_ram(monkeypatch):
    monkeypatch.setattr(hardware, "_cgroup_limits", lambda: (None, None))
    monkeypatch.setattr(hardware, "_total_ram_mb", lambda: 65536)
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1 << 40),
    )
    monkeypatch.setattr(
        hardware, "discover_nvidia_slots", lambda *a, **k: (_ for _ in ()).throw(
            hardware.AgentError("no nvidia")
        )
    )
    monkeypatch.setattr(hardware, "discover_amd_slots", lambda *a, **k: [])
    slots, notes = hardware.discover_slots(
        "tower", capabilities=[], model_packs=[], labels={}, cpu_slots=4
    )
    assert len(slots) == 4
    # Each CPU slot advertises its share, not the whole host.
    assert all(slot.vram_mb == 16384 for slot in slots)
    assert all(slot.vendor == "cpu" for slot in slots)


# -- Windows RAM ---------------------------------------------------------------


def test_windows_global_memory_status(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def GlobalMemoryStatusEx(buf):
        status = buf._obj
        assert status.dwLength == ctypes.sizeof(type(status))
        status.ullTotalPhys = 32 * (1 << 30)
        return True

    windll = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=GlobalMemoryStatusEx))
    monkeypatch.setattr(hardware.ctypes, "windll", windll, raising=False)
    assert hardware._windows_ram_mb() == 32768


def test_windows_ram_feeds_total(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def GlobalMemoryStatusEx(buf):
        buf._obj.ullTotalPhys = 8 * (1 << 30)
        return True

    windll = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=GlobalMemoryStatusEx))
    monkeypatch.setattr(hardware.ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(hardware.os.path, "exists", lambda p: False)
    assert hardware._total_ram_mb() == 8192
