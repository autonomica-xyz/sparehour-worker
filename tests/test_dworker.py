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


# -- dynamic mode ---------------------------------------------------------

from pathlib import Path as _P  # noqa: E402

from gpu_orchestrator.models import SlotInventory  # noqa: E402

from dworker import dynamic as dyn  # noqa: E402


def _inv(**overrides):
    base = dict(
        id="s0",
        gpu_index=0,
        gpu_uuid="u",
        model="NVIDIA GeForce RTX 4090",
        vram_mb=24564,
        compute_capability="8.9",
        capabilities=[],
        model_packs=[],
        labels={},
    )
    base.update(overrides)
    return SlotInventory(**base)


def _row(capability="translate-batch", pending=1, packs=(), model_dir=None, sha=None, route="r1"):
    environment = {}
    if model_dir:
        environment["ARGOS_MODEL_DIR"] = model_dir
    if sha:
        environment["ARGOS_MODEL_ARCHIVE_SHA256"] = sha
    return {
        "workload": "w",
        "route": route,
        "capability": capability,
        "model_packs": list(packs),
        "resources": {"vendor": "nvidia", "min_vram_mb": 1},
        "executor_type": "command",
        "executor_detail": "",
        "executor_environment": environment,
        "pending_jobs": pending,
    }


LOCK = {"model_pack": "argos-en-es-1.0", "sha256": "a" * 64, "url": "https://x/m.argosmodel",
        "archive_root": "en_es", "from_code": "en", "to_code": "es"}


def _plan(rows, **kw):
    defaults = dict(
        slots=[_inv()],
        provider="local",
        host=HOST,
        agent_labels={},
        installed_packs=set(),
        catalog={LOCK["model_pack"]: LOCK},
        fallback_dir=_P("/fallback"),
    )
    defaults.update(kw)
    return dyn.plan(rows, **defaults)


def test_plan_advertises_only_demanded_eligible_capabilities():
    rows = [
        _row(capability="translate-batch", pending=5),
        _row(capability="transcribe", pending=0, route="r2"),
        _row(capability="sdxl", pending=9, route="r3"),  # demanded but wants AMD
    ]
    rows[2]["resources"] = {"vendor": "amd", "min_vram_mb": 1}
    result = _plan(rows)
    assert result.capabilities == ["translate-batch"]
    assert result.fetches == []
    assert any("sdxl" in note for note in result.notes)


def test_plan_fetches_missing_packs_with_route_dir(tmp_path):
    sha = "a" * 64
    target = tmp_path / "models" / "en_es"
    rows = [_row(packs=["argos-en-es-1.0"], model_dir=str(target), sha=sha)]
    result = _plan(rows)
    assert result.capabilities == ["translate-batch"]
    assert result.fetches[0]["pack"] == "argos-en-es-1.0"
    assert result.fetches[0]["target"] == target
    assert not result.packs["argos-en-es-1.0"].ready
    # "installed" only in config bookkeeping is not proof: a missing or
    # junk directory is still queued for a verified rebuild.
    assert _plan(rows, installed_packs={"argos-en-es-1.0"}).fetches
    target.mkdir(parents=True)
    (target / "junk").write_text("not a model pack")
    # A junk directory is foreign data: never deleted, never provisioned —
    # the plan notes the refusal and queues no doomed fetch for it.
    result = _plan(rows, installed_packs={"argos-en-es-1.0"})
    assert result.fetches == [] and not result.packs["argos-en-es-1.0"].ready
    assert (target / "junk").read_text() == "not a model pack"
    assert any("refus" in note for note in result.notes)
    # no route dir -> fallback location is provisioned
    rows = [_row(packs=["argos-en-es-1.0"], sha=sha)]
    result = _plan(rows)
    assert [fetch["target"] for fetch in result.fetches] == [_P("/fallback") / "argos-en-es-1.0"]
    # no packs -> nothing fetched
    assert _plan([_row()]).fetches == []


def test_plan_rejects_catalog_that_disagrees_with_route_hash():
    rows = [_row(packs=["argos-en-es-1.0"], model_dir="/opt/models/en_es", sha="b" * 64)]
    with pytest.raises(dyn.DynamicError):
        _plan(rows)


def test_plan_skips_plugin_routes_without_handler():
    rows = [_row()]
    rows[0]["executor_type"] = "plugin"
    rows[0]["executor_detail"] = "no_such_module_xyz:Executor"
    result = _plan(rows)
    assert result.capabilities == []
    assert any("not importable" in note for note in result.notes)


def test_load_catalog_rejects_insecure_source():
    with pytest.raises(dyn.DynamicError):
        dyn.load_catalog("http://mirror.example/catalog.json")
    with pytest.raises(dyn.DynamicError):
        dyn.load_catalog("ftp://mirror.example/catalog.json")


def test_load_catalog_rejects_credentialed_url_and_leaks_nothing():
    source = "https://agent:hunter2@mirror.example/catalog.json?sig=s3cr3t"
    with pytest.raises(dyn.DynamicError, match="credentials") as caught:
        dyn.load_catalog(source)
    assert "hunter2" not in str(caught.value)
    assert "s3cr3t" not in str(caught.value)

