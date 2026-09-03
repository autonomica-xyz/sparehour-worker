"""Offline checks for dworker core logic. No control plane, no GPU needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dworker import config as config_mod
from dworker import queues as queues_mod
from dworker.hardware import discover_amd_slots


def _slot(**overrides):
    base = {
        "id": "s",
        "gpu_index": 0,
        "gpu_uuid": "u",
        "vendor": "nvidia",
        "model": "NVIDIA GeForce RTX 4090",
        "vram_mb": 24564,
        "compute_capability": "8.9",
        "capabilities": ["demo-gpu"],
        "model_packs": [],
        "labels": {},
    }
    base.update(overrides)
    return base


HOST = {"cpu_cores": 16, "ram_mb": 65536, "disk_free_mb": 1000000}
REQ = {"gpu_count": 1, "vendor": "nvidia", "min_vram_mb": 1, "capability": "demo-gpu"}


def match(requirements, slot, **kw):
    return queues_mod.route_matches(
        requirements, slot, provider="local", host=HOST, agent_labels={}, **kw
    )


def test_route_requires_selected_capability():
    ok, reason = match(REQ, _slot(capabilities=[]))
    assert not ok and "not selected" in reason


def test_route_vendor_and_vram_checks():
    ok, reason = match(REQ, _slot(vendor="amd"))
    assert not ok and "wants" in reason
    ok, reason = match({**REQ, "min_vram_mb": 32768}, _slot())
    assert not ok and "needs" in reason


def test_route_gpu_model_allowlist_fnmatch():
    req = {**REQ, "gpu_models": ["*4090*"]}
    assert match(req, _slot())[0]
    ok, _ = match(req, _slot(model="AMD Radeon RX 7900 XTX", vendor="nvidia"))
    assert not ok


def test_route_labels_and_packs():
    ok, reason = match({**REQ, "required_labels": {"tier": "trusted"}}, _slot())
    assert not ok and "label" in reason
    ok, reason = match({**REQ, "model_packs": ["sdxl-base"]}, _slot())
    assert not ok and "sdxl-base" in reason


def test_cpu_slot_matches_cpu_vendor_route():
    req = {"vendor": "cpu", "min_vram_mb": 0, "capability": "cpu-batch"}
    slot = _slot(vendor="cpu", model="cpu (x86_64)", vram_mb=32768, capabilities=["cpu-batch"])
    assert match(req, slot)[0]


def test_expand_queues_and_requirements():
    workloads = [
        {
            "name": "demo-gpu",
            "version": "1",
            "description": "demo",
            "routes": {
                "command": {
                    "compatibility": {"capability": "demo-gpu", "model_packs": []},
                    "resources": {"vendor": "nvidia", "min_vram_mb": 1},
                    "executor": {"type": "command", "command": ["x"]},
                }
            },
        }
    ]
    queues = queues_mod.expand_queues(workloads)
    assert len(queues) == 1
    requirements = queues_mod.requirements_for(queues[0])
    assert requirements["capability"] == "demo-gpu"
    assert requirements["vendor"] == "nvidia"


def test_amd_sysfs_discovery(tmp_path: Path):
    drm = tmp_path / "drm"
    gpu = drm / "card0" / "device"
    gpu.mkdir(parents=True)
    (gpu / "vendor").write_text("0x1002\n")
    (gpu / "class").write_text("0x030000\n")
    (gpu / "device").write_text("0x744c\n")
    (gpu / "mem_info_vram_total").write_text("26843545600\n")  # 24 GB -> 25600 MB
    (gpu / "product_name").write_text("AMD Radeon RX 7900 XTX\n")
    audio = drm / "card1" / "device"
    audio.mkdir(parents=True)
    (audio / "vendor").write_text("0x1002\n")
    (audio / "class").write_text("0x040300\n")  # audio device, skipped
    slots = discover_amd_slots("tower", capabilities=["rocm"], model_packs=[], labels={}, sysfs_root=drm)
    assert len(slots) == 1
    assert slots[0].vendor == "amd"
    assert slots[0].model == "AMD Radeon RX 7900 XTX"
    assert slots[0].vram_mb == 25600
    assert slots[0].gpu_uuid  # unique_id missing -> pci fallback
    assert slots[0].capabilities == ["rocm"]


def test_amd_sysfs_falls_back_to_pci_name(tmp_path: Path):
    gpu = tmp_path / "card1" / "device"
    gpu.mkdir(parents=True)
    (gpu / "vendor").write_text("0x1002\n")
    (gpu / "class").write_text("0x030000\n")
    (gpu / "device").write_text("0x67df\n")
    (gpu / "mem_info_vram_total").write_text("8589934592\n")
    slots = discover_amd_slots("tower", capabilities=[], model_packs=[], labels={}, sysfs_root=tmp_path)
    assert slots[0].model == "AMD GPU 0x1002:0x67df"


def test_idle_decision():
    from dworker.idle import IdleState as S

    # Locked screen is idle regardless of threshold.
    assert S(True, True, None, "screen locked").idle
    assert S(True, False, 400, "idle for 400s").idle
    assert not S(False, False, 10, "active").idle
    # No signal (headless server) counts as idle.
    assert S(True, None, None, "headless").idle


def test_config_roundtrip_and_permissions(tmp_path: Path):
    path = tmp_path / "config.json"
    config = dict(config_mod.DEFAULTS)
    config.update({"endpoint": "https://x", "token": "sekrit", "capabilities": ["demo-gpu"]})
    config_mod.save(config, path)
    loaded = config_mod.load(path)
    assert loaded["endpoint"] == "https://x"
    assert loaded["capabilities"] == ["demo-gpu"]
    assert path.stat().st_mode & 0o077 == 0


def test_config_load_ignores_unknown_keys(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"endpoint": "https://x", "malicious": True}))
    loaded = config_mod.load(path)
    assert "malicious" not in loaded
