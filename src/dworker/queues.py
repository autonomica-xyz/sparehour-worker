"""Queue discovery: list control-plane routes and match them to local slots.

Mirrors the control plane's `_slot_matches` so the worker can show, before
registering, which queues this machine could serve and why not the others.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from gpu_orchestrator.agent import ControlPlaneClient


def fetch_workloads(client: ControlPlaneClient) -> list[dict[str, Any]]:
    response = client.client.get("/v1/workloads")
    if response.status_code >= 300:
        raise RuntimeError(f"workload listing failed: HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def expand_queues(workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten workloads into one row per route (a route == a queue)."""
    queues: list[dict[str, Any]] = []
    for workload in workloads:
        for route_name, route in (workload.get("routes") or {}).items():
            compatibility = route.get("compatibility") or {}
            executor = route.get("executor") or {}
            queues.append(
                {
                    "workload": workload["name"],
                    "version": workload.get("version", ""),
                    "description": workload.get("description", ""),
                    "route": route_name,
                    "capability": compatibility.get("capability", ""),
                    "model_packs": compatibility.get("model_packs") or [],
                    "resources": route.get("resources") or {},
                    "executor_type": executor.get("type", ""),
                    "executor_detail": _executor_detail(executor),
                }
            )
    return sorted(queues, key=lambda queue: (queue["workload"], queue["route"]))


def _executor_detail(executor: dict[str, Any]) -> str:
    kind = executor.get("type", "")
    if kind == "container":
        return str(executor.get("image") or "unpinned image?")
    if kind == "http":
        return str(executor.get("endpoint") or "")
    if kind == "plugin":
        return str(executor.get("handler") or "")
    if kind == "command":
        return " ".join(map(str, executor.get("command") or []))[:80]
    return kind


def requirements_for(queue: dict[str, Any]) -> dict[str, Any]:
    """Combined requirement dict in the same shape as the job snapshot."""
    resources = dict(queue["resources"])
    resources["capability"] = queue["capability"]
    resources["model_packs"] = queue["model_packs"]
    return resources


def route_matches(
    requirements: dict[str, Any],
    slot: dict[str, Any],
    *,
    provider: str,
    host: dict[str, Any],
    agent_labels: dict[str, str],
) -> tuple[bool, str]:
    """(ok, failing-reason) for one slot against one route's requirements."""
    capability = requirements.get("capability")
    if capability and capability not in set(slot.get("capabilities") or []):
        return False, f"capability {capability!r} not selected"
    vendor = str(requirements.get("vendor") or "").casefold()
    if vendor != str(slot.get("vendor", "")).casefold():
        return False, f"wants {vendor!r} GPU, slot is {slot.get('vendor')!r}"
    if int(slot.get("vram_mb", 0)) < int(requirements.get("min_vram_mb") or 0):
        return (
            False,
            f"needs {int(requirements.get('min_vram_mb') or 0)} MB, slot has {slot.get('vram_mb')}",
        )
    minimum_cc = _version_tuple(requirements.get("min_compute_capability"))
    if minimum_cc and _version_tuple(slot.get("compute_capability")) < minimum_cc:
        return False, f"needs compute capability >= {requirements.get('min_compute_capability')}"
    if float(host.get("cpu_cores", 0)) < float(requirements.get("min_cpu_cores") or 0):
        return False, f"needs {requirements.get('min_cpu_cores')} CPU cores"
    if int(host.get("ram_mb", 0)) < int(requirements.get("min_ram_mb") or 0):
        return False, f"needs {requirements.get('min_ram_mb')} MB RAM"
    if int(host.get("disk_free_mb", 0)) < int(requirements.get("min_disk_mb") or 0):
        return False, f"needs {requirements.get('min_disk_mb')} MB disk"
    providers = {str(value).casefold() for value in requirements.get("allowed_providers") or []}
    if providers and str(provider).casefold() not in providers:
        return False, f"provider {provider!r} not allowed"
    if not set(requirements.get("model_packs") or []).issubset(
        set(slot.get("model_packs") or [])
    ):
        missing = set(requirements.get("model_packs") or []) - set(slot.get("model_packs") or [])
        return False, f"missing model pack(s): {', '.join(sorted(missing))}"
    model = str(slot.get("model", "")).casefold()
    allowed = [str(value).casefold() for value in requirements.get("gpu_models") or []]
    excluded = [str(value).casefold() for value in requirements.get("excluded_gpu_models") or []]
    if allowed and not any(fnmatch(model, pattern) for pattern in allowed):
        return False, f"GPU model {model!r} not in allowlist"
    if any(fnmatch(model, pattern) for pattern in excluded):
        return False, f"GPU model {model!r} excluded"
    labels = {**agent_labels, **(slot.get("labels") or {})}
    for key, value in (requirements.get("required_labels") or {}).items():
        if labels.get(key) != value:
            return False, f"requires label {key}={value}"
    return True, "matches"


def _version_tuple(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return ()
