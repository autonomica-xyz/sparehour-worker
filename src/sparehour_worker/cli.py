"""Sparehour Worker — lend spare GPU and CPU capacity to a Sparehour control plane."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import __version__, config as config_mod, idle
from .runner import AgentConfig, IdleGatedRunner, WorkerRuntime


def _config_file(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "config", None) or config_mod.config_path())


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _clean_message(exc: BaseException, config: dict[str, Any] | None = None) -> str:
    """Short, sanitized error text: type name, message, secrets removed."""
    text = f"{type(exc).__name__}: {exc}"
    if config is not None:
        token = str(config.get("token") or "")
        if token:
            text = text.replace(token, "<redacted>")
    return " ".join(text.split())[:400]


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = _config_file(args)
    try:
        config = config_mod.load(path)
        return config_mod.validate(config)
    except Exception as exc:
        fail(f"invalid config at {path}: {_clean_message(exc)}")
    raise AssertionError("unreachable")


def _runtime(args: argparse.Namespace) -> WorkerRuntime:
    config = _load_config(args)
    if not config["endpoint"] or not config["token"]:
        fail("no credentials configured — run: sparehour-worker login --endpoint URL --token-file FILE")
    try:
        return WorkerRuntime(config, _config_file(args))
    except Exception as exc:
        fail(f"worker startup failed: {_clean_message(exc, config)}")
    raise AssertionError("unreachable")


def _coerce(key: str, value: Any) -> Any:
    try:
        result = config_mod.coerce(key, str(value))
    except Exception as exc:
        fail(f"invalid value for {key!r}: {_clean_message(exc)}")
    if key in {"cpu_slots", "idle_after_seconds"}:
        try:
            if int(result) < 0:
                fail(f"invalid value for {key!r}: must be >= 0")
        except (TypeError, ValueError):
            fail(f"invalid value for {key!r}: expected a number, got {result!r}")
    return result


def _print_queues(report: list[dict[str, Any]]) -> None:
    if not report:
        print("  No workloads visible to this access token.")
        return
    name_width = max(len(f"{q['workload']} / {q['route']}") for q in report)
    for queue in report:
        name = f"{queue['workload']} / {queue['route']}"
        selected = queue["selected"]
        compatible = queue.get("compatible", queue["eligible"])
        if selected:
            readiness = "ready" if queue["eligible"] else queue["match_reason"]
        else:
            readiness = "compatible" if compatible else queue.get("compat_reason", queue["match_reason"])
        print(f"  {'on' if selected else 'off':<4} {name:<{name_width}}  [{queue['capability']}]  {readiness}")
    print("\n  Select work: sparehour-worker select <capability> (shown in brackets)")
    print("  Stop taking work: sparehour-worker deselect <capability>")


def cmd_login(args: argparse.Namespace) -> int:
    config = _load_config(args)
    try:
        config["endpoint"] = config_mod.validate_endpoint(args.endpoint.rstrip("/"))
    except Exception as exc:
        fail(f"invalid endpoint: {_clean_message(exc)}")
    token = args.token or _token_from_file(args) or _token_prompt()
    if not token:
        fail("no token provided (--token, --token-file, or SPAREHOUR_WORKER_TOKEN)")
    config["token"] = token
    if args.agent_id:
        config["agent_id"] = args.agent_id
    try:
        path = config_mod.save(config, _config_file(args))
    except Exception as exc:
        fail(f"cannot save credentials: {_clean_message(exc)}")
    print(f"saved credentials to {path} (agent_id={config['agent_id']})")
    return 0


def _token_from_file(args: argparse.Namespace) -> str:
    if args.token_file:
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    return os.environ.get("SPAREHOUR_WORKER_TOKEN", "").strip()


def _token_prompt() -> str:
    try:
        return getpass.getpass("agent token: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    try:
        print(f"Sparehour Worker {__version__} · {runtime.config['agent_id']}")
        print(f"  Configuration  {runtime.config_file}\n")
        try:
            health = runtime.client.client.get("/healthz").json()
            print(f"  + Control plane  {runtime.config['endpoint']} · {health.get('workloads', '?')} workloads")
        except Exception as exc:
            print(f"  ! Control plane  {_clean_message(exc, runtime.config)}")
            return 1
        try:
            report = runtime.queues_report(max_age=0.0)
            print("  + Access token   accepted")
        except Exception as exc:
            print(f"  ! Access token   {_clean_message(exc, runtime.config)}")
            return 1
        host = runtime.host
        print(
            f"  - Host           {host['cpu_cores']:.0f} cores · "
            f"{host['ram_mb'] / 1024:.1f} GB RAM · "
            f"{host['disk_free_mb'] / 1024:.1f} GB free disk"
        )
        for slot in runtime.slots:
            print(
                f"  - Slot           {slot.id} · {slot.model} · {slot.vram_mb / 1024:.1f} GB"
                + (f" · compute {slot.compute_capability}" if slot.compute_capability else "")
            )
        for note in runtime.hardware_notes:
            print(f"  - Hardware note  {note}")
        docker = bool(shutil.which("docker"))
        print(f"  {'+' if docker else '!'} Docker         {'available' if docker else 'not installed — container routes unavailable'}")
        state = idle.is_idle(float(runtime.config["idle_after_seconds"]))
        policy = "only when idle" if runtime.config["idle_only"] else "whenever work is queued"
        print(f"  - When to work   {policy}")
        print(f"  - Idle signal    {'idle' if state.idle else 'not idle'} · {state.reason}")
        print(f"  - Selection      {'dynamic' if runtime.config['dynamic'] else 'manual'}")
        print(f"  - Model packs    {', '.join(runtime.config['model_packs']) or 'none installed'}")
        print("\nWorkloads")
        _print_queues(report)
        return 0
    finally:
        runtime.close()


def cmd_queues(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    try:
        _print_queues(runtime.queues_report(max_age=0.0))
    except Exception as exc:
        fail(f"queue listing failed: {_clean_message(exc, runtime.config)}")
    finally:
        runtime.close()
    return 0


def _cmd_select(args: argparse.Namespace, selected: bool) -> int:
    runtime = _runtime(args)
    try:
        if not args.capabilities:
            print(f"selected capabilities: {runtime.config['capabilities'] or 'none'}")
            return 0
        ok, message = runtime.set_capability_list(args.capabilities, selected)
        print(message)
        return 0 if ok else 1
    except Exception as exc:
        fail(f"selection failed: {_clean_message(exc, runtime.config)}")
    finally:
        runtime.close()
    return 1


def cmd_prefetch(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    try:
        for line in runtime.prefetch_images():
            print(line)
    except Exception as exc:
        fail(f"prefetch failed: {_clean_message(exc, runtime.config)}")
    finally:
        runtime.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if not config["endpoint"] or not config["token"]:
        fail("no credentials configured — run: sparehour-worker login --endpoint URL --token-file FILE")
    # CLI overrides must be in place BEFORE WorkerRuntime discovers slots:
    # --cpu-slots (including --cpu-slots 0) changes the registered inventory,
    # and idle settings gate the very first claim attempt. The pre-override
    # values are handed to the runtime so its config persistence (dynamic
    # model-pack saves) never writes these runtime-only overrides to disk.
    saved_overrides: dict[str, Any] = {}
    if args.idle_only:
        saved_overrides["idle_only"] = config["idle_only"]
        config["idle_only"] = True
    if args.idle_after is not None:
        saved_overrides["idle_after_seconds"] = config["idle_after_seconds"]
        config["idle_after_seconds"] = _coerce("idle_after_seconds", args.idle_after)
    if args.cpu_slots is not None:
        saved_overrides["cpu_slots"] = config["cpu_slots"]
        config["cpu_slots"] = _coerce("cpu_slots", args.cpu_slots)
    try:
        runtime = WorkerRuntime(config, _config_file(args), saved_overrides=saved_overrides)
    except Exception as exc:
        fail(f"worker startup failed: {_clean_message(exc, config)}")
    return _run_worker(runtime, once=bool(args.once), no_dashboard=bool(args.no_dashboard))


def _run_worker(runtime: WorkerRuntime, *, once: bool, no_dashboard: bool) -> int:
    runner: IdleGatedRunner | None = None
    dynamic_thread: threading.Thread | None = None
    dashboard_handles: tuple[Any, threading.Thread] | None = None
    try:
        if not runtime.slots:
            fail(
                "no slots to register: select capabilities (sparehour-worker select <capability>), "
                "have a GPU (nvidia-smi / amdgpu sysfs), or set cpu_slots > 0"
            )
        try:
            agent_config = AgentConfig(
                registration=runtime.registration(),
                workdir_root=config_mod.workdir_path(runtime.config),
                heartbeat_seconds=int(runtime.config["heartbeat_seconds"]),
                lease_seconds=int(runtime.config["lease_seconds"]),
                poll_seconds=float(runtime.config["poll_seconds"]),
            )
            runner = IdleGatedRunner(
                runtime.client,
                agent_config,
                status=runtime.status,
                # The runtime association makes the initial registration
                # atomic and current: a locked snapshot of the inventory is
                # taken and registered at send time, so a concurrent or
                # just-finished dynamic provisioning cycle can never be
                # overwritten by a stale pre-provisioning registration.
                runtime=runtime,
            )
        except Exception as exc:
            fail(f"invalid worker settings: {_clean_message(exc, runtime.config)}")
        # Share one cancellation event between runner and runtime: a signal
        # drain (runner.request_drain sets runner.stop) then also cancels the
        # dynamic manager and in-flight prefetch/pulls immediately — while
        # already-running jobs still finish, since executors don't poll stop.
        runner.stop = runtime.stop
        runner.idle_only = bool(runtime.config["idle_only"])
        runner.idle_after_seconds = float(runtime.config["idle_after_seconds"])
        runtime.status.idle_only = runner.idle_only

        if bool(runtime.config["dynamic"]):
            dynamic_thread = _start_dynamic(runtime)

        if bool(runtime.config["dashboard"]) and not no_dashboard:
            dashboard_handles = _start_dashboard(runtime, runner)

        runtime.status.note("worker started")
        try:
            runner.run(once=once)
        except Exception as exc:
            fail(f"worker failed: {_clean_message(exc, runtime.config)}")
        return 0
    finally:
        # Stop sources first so background loops exit, then shut the dashboard
        # (which joins in-flight prefetch work), then the dynamic manager, and
        # only then release the control-plane client. Threads are joined
        # completely — shutdown is bounded by the cooperative stop plus
        # bounded network/subprocess operations — never warned-and-abandoned,
        # so client.close() cannot run under live background work.
        runtime.stop.set()
        if dashboard_handles is not None:
            server, dashboard_thread = dashboard_handles
            _stop_dashboard(server, dashboard_thread, runtime)
        if dynamic_thread is not None:
            dynamic_thread.join()
        runtime.close()


def _start_dynamic(runtime: WorkerRuntime) -> threading.Thread:
    """Provision once synchronously, then run the dynamic manager.

    The first cycle runs inline so demanded routes are advertised before the
    runner's first claims — otherwise `--once` could report a false no-job
    success while the manager was still waiting out its initial interval.
    """
    from . import dynamic as dynamic_mod

    try:
        report = runtime.queues_report(max_age=0.0)
    except Exception as exc:
        fail(f"cannot check dynamic mode: {_clean_message(exc, runtime.config)}")
    if report and all(queue.get("pending_jobs") is None for queue in report):
        fail(
            "dynamic=true but this token cannot see queue depths — reissue it "
            "with: sparehour token --role agent --subject NAME --scope perms=dynamic"
        )
    if not report:
        fail("dynamic=true but the control plane exposes no workloads")
    if not str(runtime.config["model_catalog"] or ""):
        print("dynamic: no model_catalog set — packs missing locally cannot be fetched")
    try:
        dynamic_mod.manage_cycle(runtime)
    except Exception as exc:
        # A failed first cycle is not fatal: the manager retries. For --once
        # the demand was simply not provisioned this run.
        print(f"dynamic: initial provisioning failed: {_clean_message(exc, runtime.config)}",
              file=sys.stderr)
    thread = threading.Thread(
        target=dynamic_mod.manager_loop,
        args=(runtime, runtime.stop),
        name="dynamic",
    )
    thread.start()
    return thread


def _dashboard_access_file(config_file: Path) -> Path:
    return config_file.parent / f"{config_file.name}.dashboard-url"


def _write_access_file(config_file: Path, url: str) -> Path:
    """Atomically write the dashboard access URL, mode 0600, beside the config."""
    target = _dashboard_access_file(config_file)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{url}\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _start_dashboard(
    runtime: WorkerRuntime, runner: IdleGatedRunner
) -> tuple[Any, threading.Thread] | None:
    """Bind the dashboard and return (server, thread) cleanup handles.

    A bind/address failure is reported but does not stop compute — the
    dashboard is optional. Invalid security configuration (e.g. a non-loopback
    bind host) is a fatal error, never silently ignored.
    """
    from . import dashboard

    def status_payload() -> dict[str, Any]:
        state = idle.is_idle(runner.idle_after_seconds)
        snapshot = runtime.status.snapshot()
        snapshot.update(
            {
                "endpoint": runtime.config["endpoint"],
                "capabilities": runtime.config["capabilities"],
                "model_packs": runtime.config["model_packs"],
                "dynamic": bool(runtime.config["dynamic"]),
                "docker": bool(shutil.which("docker")),
                "idle_ok": state.idle,
                "idle_reason": state.reason,
                "queues": runtime.queues_report(),
            }
        )
        for slot in runtime.slots:
            info = snapshot["slots"].setdefault(slot.id, {"state": "registered", "job": None})
            info["name"] = f"{'CPU' if slot.vendor == 'cpu' else 'GPU'} {slot.gpu_index}"
            info["hardware"] = f"{slot.model} · {slot.vram_mb / 1024:g} GB"
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

    host = str(runtime.config["dashboard_host"])
    try:
        port = int(runtime.config["dashboard_port"])
    except (TypeError, ValueError):
        fail(f"invalid dashboard_port: {runtime.config['dashboard_port']!r}")
    try:
        server = dashboard.make_server(
            status_provider=status_payload,
            pause_handler=pause,
            select_handler=select,
            prefetch_handler=prefetch,
            host=host,
            port=port,
        )
    except OSError as exc:
        # Port in use / unresolvable address: optional UI is unavailable but
        # compute must still run.
        print(
            f"dashboard unavailable ({host}:{port}: {_clean_message(exc)}) — continuing without it",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        # Rejected security configuration (non-loopback host, bad token, …).
        fail(f"invalid dashboard configuration: {_clean_message(exc)}")
    thread = threading.Thread(target=server.serve_forever, name="dashboard", daemon=True)
    thread.start()
    url = server.dashboard_url
    try:
        url_file = _write_access_file(runtime.config_file, url)
    except OSError as exc:
        print(
            f"warning: cannot write dashboard access file: {_clean_message(exc)}",
            file=sys.stderr,
        )
        url_file = _dashboard_access_file(runtime.config_file)
    if sys.stdout.isatty():
        # The fragment token is a secret: only print it on an interactive TTY.
        print(f"dashboard: {url}", flush=True)
    else:
        print(
            f"dashboard: http://{host}:{port} — access URL written to {url_file} (mode 0600)",
            flush=True,
        )
    return (server, thread)


def _stop_dashboard(server: Any, thread: threading.Thread, runtime: WorkerRuntime) -> None:
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()  # also joins in-flight prefetch work
    except Exception:
        pass
    thread.join()  # shutdown() guarantees serve_forever returns
    # Remove the access file only if it still belongs to this instance —
    # another run sharing the config path may have replaced it.
    url_file = _dashboard_access_file(runtime.config_file)
    try:
        if url_file.is_file() and url_file.read_text(encoding="utf-8").strip() == str(
            server.dashboard_url
        ):
            url_file.unlink()
    except OSError:
        pass


def cmd_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    words = args.action
    if words and words[0] == "set":
        if len(words) != 2:
            fail("usage: sparehour-worker config set KEY=VALUE")
        key, separator, value = words[1].partition("=")
        if not separator or key not in config_mod.DEFAULTS:
            fail(f"unknown config key {key!r} (known: {', '.join(sorted(config_mod.DEFAULTS))})")
        config[key] = _coerce(key, value)
        try:
            config_mod.save(config, _config_file(args))
        except Exception as exc:
            fail(f"cannot save config: {_clean_message(exc)}")
        print(f"{key} = {_redacted(key, config[key])!r} (saved)")
    elif words and words[0] == "get":
        if len(words) != 2:
            fail("usage: sparehour-worker config get KEY")
        print(_redacted(words[1], config.get(words[1], "")))
    elif not words:
        shown = dict(config)
        if shown.get("token"):
            shown["token"] = "<redacted>"
        print(json.dumps(shown, indent=2, sort_keys=True))
    else:
        fail(f"usage: sparehour-worker config [set KEY=VALUE | get KEY] (got: {' '.join(words)})")
    return 0


def _redacted(key: str, value: Any) -> Any:
    """Never disclose the agent token — not even a prefix/suffix."""
    if key == "token" and value:
        return "<redacted>"
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparehour-worker", description=__doc__)
    parser.add_argument("--config", help="config file path (default: ~/.config/sparehour/worker.json)")
    parser.add_argument("--version", action="version", version=f"sparehour-worker {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser("login", help="store control-plane endpoint and agent token")
    login.add_argument("--endpoint", required=True, help="e.g. https://gpu-control.example.com")
    login.add_argument("--token", help="agent bearer token (or SPAREHOUR_WORKER_TOKEN, or prompt)")
    login.add_argument("--token-file", help="file containing the token")
    login.add_argument("--agent-id", help="defaults to hostname")
    login.set_defaults(func=cmd_login)

    for name, help_text in (("doctor", "diagnose hardware, credentials, queues, idle detection"),):
        doctor = commands.add_parser(name, help=help_text)
        doctor.set_defaults(func=cmd_doctor)

    queues = commands.add_parser("queues", help="list queues and whether this machine matches")
    queues.set_defaults(func=cmd_queues)

    select = commands.add_parser("select", help="advertise capability(ies): sparehour-worker select demo-gpu")
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
