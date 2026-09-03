"""dworker — command line for the opportunistic distributed worker."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from . import __version__, config as config_mod, idle
from .runner import AgentConfig, IdleGatedRunner, WorkerRuntime


def _runtime(args: argparse.Namespace) -> WorkerRuntime:
    config = config_mod.load(_config_file(args))
    if not config["endpoint"] or not config["token"]:
        fail("no credentials configured — run: dworker login --endpoint URL --token-file FILE")
    return WorkerRuntime(config, _config_file(args))


def _config_file(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "config", None) or config_mod.config_path())


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _print_queues(report: list[dict[str, Any]]) -> None:
    if not report:
        print("  (no workloads visible to this token)")
        return
    name_width = max(len(f"{q['workload']}/{q['route']}") for q in report)
    for queue in report:
        mark = "+" if queue["selected"] else " "
        flag = "eligible " if queue["eligible"] else "no      "
        print(
            f"  [{mark}] {flag} {queue['workload']}/{queue['route']:<{name_width}}"
            f"  {queue['capability']:<24} {queue['executor_type']:<9} {queue['match_reason']}"
        )
    print("\n  [+]=advertised capability. Toggle with: dworker select|deselect <capability>")


def cmd_login(args: argparse.Namespace) -> int:
    config = config_mod.load(_config_file(args))
    config["endpoint"] = args.endpoint.rstrip("/")
    token = args.token or _token_from_file(args) or _token_prompt()
    if not token:
        fail("no token provided (--token, --token-file, or DWORKER_TOKEN)")
    config["token"] = token
    if args.agent_id:
        config["agent_id"] = args.agent_id
    path = config_mod.save(config, _config_file(args))
    print(f"saved credentials to {path} (agent_id={config['agent_id']})")
    return 0


def _token_from_file(args: argparse.Namespace) -> str:
    if args.token_file:
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    import os

    return os.environ.get("DWORKER_TOKEN", "").strip()


def _token_prompt() -> str:
    try:
        return getpass.getpass("agent token: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    print(f"dworker {__version__}  config={runtime.config_file}")
    print(f"control plane: {runtime.config['endpoint']}", end=" ")
    try:
        health = runtime.client.client.get("/healthz").json()
        print(f"OK ({health.get('workloads', '?')} workloads)")
    except Exception as exc:
        print(f"UNREACHABLE: {exc}")
        return 1
    try:
        runtime.queues_report(max_age=0.0)
        print("token: accepted")
    except Exception as exc:
        print(f"token: REJECTED: {exc}")
        return 1
    for note in runtime.hardware_notes:
        print(f"hardware: {note}")
    host = runtime.host
    print(
        f"hardware: host cpu={host['cpu_cores']:.0f} cores ram={host['ram_mb']} MB"
        f" disk_free={host['disk_free_mb']} MB"
    )
    for slot in runtime.slots:
        print(
            f"  slot {slot.id}: {slot.vendor} {slot.model} ({slot.vram_mb} MB)"
            + (f" cc={slot.compute_capability}" if slot.compute_capability else "")
        )
    print(f"docker: {'available' if shutil.which('docker') else 'MISSING (container routes unusable)'}")
    state = idle.is_idle(float(runtime.config["idle_after_seconds"]))
    print(
        f"idle detection: {'locked' if state.locked else 'unlocked'}, "
        f"idle={state.idle_seconds if state.idle_seconds is not None else '?'}s -> "
        f"{'IDLE' if state.idle else 'ACTIVE'} ({state.reason})"
    )
    print(f"selection: capabilities={runtime.config['capabilities'] or 'none'} "
          f"model_packs={runtime.config['model_packs'] or 'none'} "
          f"idle_only={runtime.config['idle_only']} "
          f"cpu_slots={runtime.config['cpu_slots']}")
    print("queues:")
    _print_queues(runtime.queues_report(max_age=0.0))
    runtime.close()
    return 0


def cmd_queues(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    _print_queues(runtime.queues_report(max_age=0.0))
    runtime.close()
    return 0


def _cmd_select(args: argparse.Namespace, selected: bool) -> int:
    runtime = _runtime(args)
    if not args.capabilities:
        print(f"selected capabilities: {runtime.config['capabilities'] or 'none'}")
        runtime.close()
        return 0
    ok, message = runtime.set_capability_list(args.capabilities, selected)
    print(message)
    runtime.close()
    return 0 if ok else 1


def cmd_prefetch(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    for line in runtime.prefetch_images():
        print(line)
    runtime.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    if not runtime.slots:
        runtime.close()
        fail(
            "no slots to register: select capabilities (dworker select <capability>), "
            "have a GPU (nvidia-smi / amdgpu sysfs), or set cpu_slots > 0"
        )
    if args.idle_only:
        runtime.config["idle_only"] = True
    if args.idle_after is not None:
        runtime.config["idle_after_seconds"] = int(args.idle_after)
    if args.cpu_slots is not None:
        runtime.config["cpu_slots"] = int(args.cpu_slots)

    runner = IdleGatedRunner(
        runtime.client,
        AgentConfig(
            registration=runtime.registration(),
            workdir_root=config_mod.workdir_path(runtime.config),
            heartbeat_seconds=int(runtime.config["heartbeat_seconds"]),
            lease_seconds=int(runtime.config["lease_seconds"]),
            poll_seconds=float(runtime.config["poll_seconds"]),
        ),
        status=runtime.status,
    )
    runner.idle_only = bool(runtime.config["idle_only"])
    runner.idle_after_seconds = float(runtime.config["idle_after_seconds"])
    runtime.status.idle_only = runner.idle_only

    dashboard_enabled = bool(runtime.config["dashboard"]) and not args.no_dashboard
    if dashboard_enabled:
        _start_dashboard(runtime, runner, args)

    runtime.status.note("worker started")
    try:
        runner.run(once=args.once)
    finally:
        runtime.close()
    return 0


def _start_dashboard(runtime: WorkerRuntime, runner: IdleGatedRunner, args: argparse.Namespace) -> None:
    from . import dashboard

    def status_payload() -> dict[str, Any]:
        state = idle.is_idle(runner.idle_after_seconds)
        snapshot = runtime.status.snapshot()
        snapshot.update(
            {
                "endpoint": runtime.config["endpoint"],
                "capabilities": runtime.config["capabilities"],
                "model_packs": runtime.config["model_packs"],
                "docker": bool(shutil.which("docker")),
                "idle_ok": state.idle,
                "idle_reason": state.reason,
                "queues": runtime.queues_report(),
            }
        )
        for slot in runtime.slots:
            snapshot["slots"].setdefault(
                slot.id,
                {"state": "registered", "job": None, "hardware": ""},
            )["hardware"] = f"{slot.vendor} {slot.model} ({slot.vram_mb} MB)"
        return snapshot

    def pause(paused: bool) -> None:
        runner.paused_event.set() if paused else runner.paused_event.clear()
        runtime.status.paused = paused
        runtime.status.note("paused" if paused else "resumed")

    def select(capability: str, selected: bool) -> tuple[bool, str]:
        if not capability:
            return False, "capability required"
        return runtime.set_capability(capability, selected)

    def prefetch() -> None:
        runtime.prefetch_images()

    server = dashboard.make_server(
        status_provider=status_payload,
        pause_handler=pause,
        select_handler=select,
        prefetch_handler=prefetch,
        host=str(runtime.config["dashboard_host"]),
        port=int(runtime.config["dashboard_port"]),
    )
    thread = threading.Thread(target=server.serve_forever, name="dashboard", daemon=True)
    thread.start()
    print(
        f"dashboard: http://{runtime.config['dashboard_host']}:{runtime.config['dashboard_port']}",
        flush=True,
    )


def cmd_config(args: argparse.Namespace) -> int:
    config = config_mod.load(_config_file(args))
    words = args.action
    if words and words[0] == "set":
        if len(words) != 2:
            fail("usage: dworker config set KEY=VALUE")
        key, separator, value = words[1].partition("=")
        if not separator or key not in config_mod.DEFAULTS:
            fail(f"unknown config key {key!r} (known: {', '.join(sorted(config_mod.DEFAULTS))})")
        config[key] = _coerce(key, value, config_mod.DEFAULTS[key])
        config_mod.save(config, _config_file(args))
        print(f"{key} = {config[key]!r} (saved)")
    elif words and words[0] == "get":
        if len(words) != 2:
            fail("usage: dworker config get KEY")
        print(config.get(words[1], ""))
    elif not words:
        shown = dict(config)
        if shown.get("token"):
            shown["token"] = shown["token"][:6] + "…" + shown["token"][-4:]
        print(json.dumps(shown, indent=2, sort_keys=True))
    else:
        fail(f"usage: dworker config [set KEY=VALUE | get KEY] (got: {' '.join(words)})")
    return 0


def _coerce(key: str, value: str, default: Any) -> Any:
    try:
        if isinstance(default, bool):
            return value.lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, list):
            return [item.strip() for item in value.split(",") if item.strip()]
    except ValueError:
        fail(f"invalid value for {key!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dworker", description=__doc__)
    parser.add_argument("--config", help="config file path (default: ~/.config/dworker/config.json)")
    parser.add_argument("--version", action="version", version=f"dworker {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser("login", help="store control-plane endpoint and agent token")
    login.add_argument("--endpoint", required=True, help="e.g. https://gpu-control.example.com")
    login.add_argument("--token", help="agent bearer token (or DWORKER_TOKEN, or prompt)")
    login.add_argument("--token-file", help="file containing the token")
    login.add_argument("--agent-id", help="defaults to hostname")
    login.set_defaults(func=cmd_login)

    for name, help_text in (("doctor", "diagnose hardware, credentials, queues, idle detection"),):
        doctor = commands.add_parser(name, help=help_text)
        doctor.set_defaults(func=cmd_doctor)

    queues = commands.add_parser("queues", help="list queues and whether this machine matches")
    queues.set_defaults(func=cmd_queues)

    select = commands.add_parser("select", help="advertise capability(ies): dworker select demo-gpu")
    select.add_argument("capabilities", nargs="*")
    select.set_defaults(func=lambda args: _cmd_select(args, True))

    deselect = commands.add_parser("deselect", help="stop advertising capability(ies)")
    deselect.add_argument("capabilities", nargs="*")
    deselect.set_defaults(func=lambda args: _cmd_select(args, False))

    prefetch = commands.add_parser("prefetch", help="docker-pull images for selected capabilities")
    prefetch.set_defaults(func=cmd_prefetch)

    run = commands.add_parser("run", help="register slots and work queues (the SETI loop)")
    run.add_argument("--once", action="store_true", help="claim at most one job per slot, then exit")
    run.add_argument("--idle-only", action="store_true", help="only claim while machine is idle/locked")
    run.add_argument("--idle-after", type=int, metavar="SECONDS", help="idle threshold")
    run.add_argument("--cpu-slots", type=int, help="override cpu_slots for this run")
    run.add_argument("--no-dashboard", action="store_true", help="do not serve the local dashboard")
    run.set_defaults(func=cmd_run)

    config = commands.add_parser("config", help="show config, or: config set KEY=VALUE / get KEY")
    config.add_argument("action", nargs="*", help="set KEY=VALUE | get KEY | (nothing to dump)")
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
