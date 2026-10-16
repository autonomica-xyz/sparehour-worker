"""Matcher boundary tests: multi-GPU rejection and ROCm prerequisites.

These mirror the control plane's ``_slot_matches`` contract — the worker's
client-side preview must refuse exactly what the server refuses, and must
not advertise AMD eligibility when the ROCm device nodes are absent.
"""

from __future__ import annotations

import httpx
import pytest
from sparehour.agent import AgentError

from sparehour_worker import cli as cli_mod
from sparehour_worker import queues as queues_mod
from sparehour_worker.client import WorkerClient



def _slot(**overrides):
    base = {
        "id": "w-0",
        "gpu_index": 0,
        "gpu_uuid": "GPU-aaaabbbbccccdddd",
        "vendor": "nvidia",
        "model": "Test GPU",
        "vram_mb": 16384,
        "compute_capability": "9.0",
        "capabilities": ["demo"],
        "model_packs": [],
        "labels": {},
    }
    base.update(overrides)
    return base


HOST = {"cpu_cores": 16, "ram_mb": 65536, "disk_free_mb": 1000000}
REQ = {"vendor": "nvidia", "capability": "demo", "min_vram_mb": 1}


def match(requirements, slot, host=HOST):
    return queues_mod.route_matches(
        requirements, slot, provider="local", host=host, agent_labels={}
    )


def test_gpu_count_one_and_absent_match():
    assert match({**REQ, "gpu_count": 1}, _slot()) == (True, "matches")
    assert match(REQ, _slot()) == (True, "matches")


def test_multi_gpu_requirement_explicitly_rejected():
    ok, reason = match({**REQ, "gpu_count": 2}, _slot())
    assert not ok
    assert "gpu_count" in reason


def test_multi_gpu_rejection_takes_priority_over_other_fields():
    # Even a slot that satisfies every other requirement must not match.
    ok, _ = match({**REQ, "gpu_count": 8, "min_vram_mb": 1}, _slot())
    assert not ok


def test_amd_route_rejected_when_rocm_nodes_missing():
    host = {**HOST, "rocm_kfd": False}
    ok, reason = match(
        {"vendor": "amd", "capability": "demo", "min_vram_mb": 1},
        _slot(vendor="amd", model="AMD Instinct", capabilities=["demo"]),
        host=host,
    )
    assert not ok
    assert "ROCm" in reason or "kfd" in reason


def test_amd_route_rejected_on_fallback_pci_uuid():
    amd_slot = _slot(
        vendor="amd", model="AMD Instinct", gpu_uuid="pci-0000:03:00.0",
        capabilities=["demo"],
    )
    ok, reason = match(
        {"vendor": "amd", "capability": "demo", "min_vram_mb": 1},
        amd_slot,
        host={**HOST, "rocm_kfd": True},
    )
    assert not ok
    assert "UUID" in reason


def test_amd_route_matches_when_rocm_nodes_present_or_unknown():
    amd_slot = _slot(vendor="amd", model="AMD Instinct", capabilities=["demo"])
    req = {"vendor": "amd", "capability": "demo", "min_vram_mb": 1}
    assert match(req, amd_slot, host={**HOST, "rocm_kfd": True})[0]
    # Absent key = unmeasured host (e.g. unit tests) → do not block.
    assert match(req, amd_slot)[0]


def test_cpu_slot_never_matches_gpu_route_and_vice_versa():
    cpu_slot = _slot(vendor="cpu", model="cpu", capabilities=["demo"])
    assert not match(REQ, cpu_slot)[0]
    gpu_slot = _slot()
    assert not match({"vendor": "cpu", "capability": "demo"}, gpu_slot)[0]
    assert match({"vendor": "cpu", "capability": "demo"}, cpu_slot)[0]


def test_requirements_for_carries_gpu_count():
    queue = {
        "workload": "w",
        "route": "r",
        "capability": "demo",
        "model_packs": [],
        "resources": {"vendor": "nvidia", "gpu_count": 2},
    }
    requirements = queues_mod.requirements_for(queue)
    assert requirements["gpu_count"] == 2
    ok, _ = match(requirements, _slot())
    assert not ok



def test_hardware_compatibility_ignores_selection_and_packs():
    """The dashboard's compatible/independent-of-selected signal: an
    unselected capability on hardware that suits still reports compatible,
    while a real hardware miss reports its reason — not 'not selected'."""
    queue = {
        "workload": "w",
        "route": "r",
        "capability": "unselected",
        "model_packs": "irrelevant",
        "resources": {"vendor": "nvidia", "min_vram_mb": 1},
    }
    ok, reason = queues_mod.hardware_compatibility(
        queue, [_slot()], provider="local", host=HOST, agent_labels={}
    )
    assert ok and reason == "matches"

    ok, reason = queues_mod.hardware_compatibility(
        queue, [_slot(vendor="cpu")], provider="local", host=HOST, agent_labels={}
    )
    assert not ok and "not selected" not in reason

    queue["resources"]["min_vram_mb"] = 10**9
    ok, reason = queues_mod.hardware_compatibility(
        queue, [_slot()], provider="local", host=HOST, agent_labels={}
    )
    assert not ok and "MB" in reason

# -- fetch_workloads: sanitized control-plane boundary -------------------------

TOKEN = "dgp_testsekrit123"
ENDPOINT = "http://127.0.0.1:8730"
SECRET_BODY = (
    'controller exploded; diagnostic dump token=dgp_testsekrit123 '
    'retry https://media.evil.example/upload?signature=AKIA-SECRET&x=1'
)


def _fetch_client(handler):
    return WorkerClient(ENDPOINT, TOKEN, transport=httpx.MockTransport(handler))


def test_fetch_workloads_http_error_never_echoes_body():
    """A 503 body carrying a token and a signed URL must reach neither the
    raised exception (CLI-visible via _clean_message) nor anything the
    dynamic manager would record as a status note."""

    def handler(request):
        return httpx.Response(503, text=SECRET_BODY)

    client = _fetch_client(handler)
    try:
        with pytest.raises(AgentError) as caught:
            queues_mod.fetch_workloads(client)
    finally:
        client.close()
    message = str(caught.value)
    assert "HTTP 503" in message
    for secret in (TOKEN, "AKIA-SECRET", "media.evil.example"):
        assert secret not in message
        assert secret not in repr(caught.value)
        # The exception must survive the CLI's sanitizer unchanged-clean.
        assert secret not in cli_mod._clean_message(caught.value)
        assert secret not in cli_mod._clean_message(
            caught.value, {"token": "another-token"}
        )


def test_fetch_workloads_malformed_json_is_sanitized():
    def handler(request):
        return httpx.Response(200, text=SECRET_BODY + " {not json")

    client = _fetch_client(handler)
    try:
        with pytest.raises(AgentError) as caught:
            queues_mod.fetch_workloads(client)
    finally:
        client.close()
    assert "malformed" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert "AKIA-SECRET" not in str(caught.value)


def test_fetch_workloads_unexpected_shape_is_sanitized():
    # Right type family, wrong shape — echoed shape content must not leak.
    def handler(request):
        return httpx.Response(200, json={"workloads": SECRET_BODY})

    client = _fetch_client(handler)
    try:
        with pytest.raises(AgentError) as caught:
            queues_mod.fetch_workloads(client)
    finally:
        client.close()
    assert "unexpected" in str(caught.value) or "malformed" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert "AKIA-SECRET" not in str(caught.value)


def test_fetch_workloads_success_preserves_wire_shape():
    workloads = [
        {
            "name": "translate",
            "version": "1",
            "routes": {
                "r1": {
                    "compatibility": {"capability": "translate-batch"},
                    "executor": {"type": "command", "command": ["echo"]},
                    "pending_jobs": 3,
                }
            },
        }
    ]

    def handler(request):
        assert request.url.path == "/v1/workloads"
        return httpx.Response(200, json=workloads)

    client = _fetch_client(handler)
    try:
        assert queues_mod.fetch_workloads(client) == workloads
    finally:
        client.close()
    queues = queues_mod.expand_queues(workloads)
    assert queues[0]["workload"] == "translate"
    assert queues[0]["route"] == "r1"
    assert queues[0]["pending_jobs"] == 3


def test_fetch_workloads_empty_body_returns_empty_list():
    def handler(request):
        return httpx.Response(204)

    client = _fetch_client(handler)
    try:
        assert queues_mod.fetch_workloads(client) == []
    finally:
        client.close()


def test_expand_queues_forwards_environment_and_inputs():
    """Queue rows carry the route's executor environment (model dirs/IDs the
    provisioner needs) and the workload's declared inputs (materialization
    requirement for input-consuming routes)."""
    workloads = [
        {
            "name": "transcribe",
            "version": "1",
            "inputs": [{"name": "audio", "media_types": ["audio/wav"],
                        "max_bytes": 1024}],
            "routes": {
                "parakeet": {
                    "compatibility": {
                        "capability": "transcribe-parakeet",
                        "model_packs": ["parakeet-tdt-0.6b-v3"],
                    },
                    "executor": {
                        "type": "plugin",
                        "handler": "sparehour_workloads.transcription:ParakeetExecutor",
                        "environment": {
                            "PARAKEET_MODEL_DIR": "/models/parakeet",
                            "PARAKEET_MODEL_ID": "nvidia/parakeet-tdt-0.6b-v3",
                            "PARAKEET_MODEL_REVISION": "a" * 40,
                        },
                    },
                    "pending_jobs": 1,
                }
            },
        }
    ]
    queues = queues_mod.expand_queues(workloads)
    assert len(queues) == 1
    env = queues[0]["executor_environment"]
    assert env["PARAKEET_MODEL_DIR"] == "/models/parakeet"
    assert env["PARAKEET_MODEL_REVISION"] == "a" * 40
    assert queues[0]["inputs"] == workloads[0]["inputs"]
