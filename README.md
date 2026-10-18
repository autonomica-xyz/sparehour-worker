# Sparehour Worker

**Sparehour Worker** is the worker side of
[Sparehour](https://github.com/autonomica-xyz/sparehour): an open-source CLI
and local dashboard that lets a workstation or server contribute spare
NVIDIA, AMD, or CPU compute to a control plane — SETI@home-style, claiming
independent jobs only when you tell it to. The controller (a separate,
GPL-3.0-or-later package) defines trusted workload routes, stores jobs
durably, and schedules at most one job per worker slot; `sparehour-worker`
registers slots, advertises the capabilities you select, and runs the jobs
it claims.

![Sparehour Worker local dashboard](assets/screenshots/worker-dashboard.png)

*The dark local dashboard uses Autonomica colours and system sans typography:
live worker status, per-slot jobs, workload switches, and hardware compatibility.*

## Historical demo video

[Watch the 60-second worker demonstration](assets/videos/worker-demo.mp4):
real GPU inventory, workload selection, pause/resume, and a dispatched image
job returning its result. Captioned and edited for length; runtime and model
weights are provisioned separately.
The recording predates the Sparehour rebrand; its interface and burned-in
captions show the previous names and design. The screenshot above is current.

[Poster, subtitles, and sharing guide](assets/videos/README.md).

## What it does

- **Inventory** — detects NVIDIA GPUs via `nvidia-smi`, AMD GPUs via the
  amdgpu sysfs tree, and configurable CPU slots (`cpu_slots=N`), and
  reports usable cores/RAM after affinity and cgroup limits.
- **Queue visibility** — `sparehour-worker doctor` and `sparehour-worker queues` show every
  route the controller offers and whether this machine matches (vendor,
  VRAM, compute capability, GPU model, model packs, labels), with the
  reason when it does not.
- **Explicit selection** — you choose what to advertise:
  `sparehour-worker select translate-batch`, or workload switches in the
  dashboard. Switches re-register live; a running job always finishes.
- **Idle gating** — with `idle_only=true` the worker claims only while the
  machine is confirmed idle: all local sessions locked, or quiet past
  `idle_after_seconds`. Detection reads logind/X11 on Linux,
  `IOConsoleLocked`/`HIDIdleTime` on macOS, and `GetLastInputInfo` on
  Windows. Ambiguous or missing signals fail closed — no new claiming.
  Idle gating does not preempt jobs already running.
- **Dynamic mode** — with `dynamic=true` and an agent token carrying
  `perms=dynamic`, the worker watches per-route queue depth and advertises
  exactly the capabilities that have pending work, downloading and
  SHA-256-verifying demanded model packs before the first claim.
- **Prefetch** — `sparehour-worker prefetch` docker-pulls the digest-pinned images
  behind your selected capabilities and reports what model packs or local
  model servers a route still needs.
- **Dashboard** — during `sparehour-worker run`, a status page binds a **literal
  loopback address only** (default `http://127.0.0.1:7788`); every `/api/*`
  call needs a per-run bearer token delivered in the URL fragment and
  stored mode-0600 in `<config-file>.dashboard-url`.
  Hardware compatibility is shown separately from selection and model-pack
  readiness. Routes share one switch per capability; long route lists expand
  under “more routes”, with individual requirements and executor details kept
  available. Polling preserves open disclosures and focused switches.

## Requirements

- Python 3.10 or newer. GPU drivers and inference runtimes are
  platform-specific; Linux is recommended for GPU workers.
- The separate `sparehour` package, installed from source or a wheel.
- For container routes, Docker and access to its daemon.
- For AMD execution, a ROCm-capable runtime, accessible `/dev/kfd` and
  `/dev/dri`, and a card with a nonzero sysfs `unique_id`.

Install each route's executor, runtime and reviewed model packs before
advertising it. ML dependencies and weights are not part of the base worker
install. The controller's [workload guide](https://github.com/autonomica-xyz/sparehour/blob/main/workloads/README.md)
and [runtime profiles](https://github.com/autonomica-xyz/sparehour/tree/main/deploy)
describe those prerequisites. Bundled ML routes target NVIDIA; AMD and CPU
capacity need compatible routes configured by your controller operator.

## Install

Both packages install from source. The worker depends on the control-plane
package, so install that first:

```bash
git clone https://github.com/autonomica-xyz/sparehour.git sparehour
git clone https://github.com/autonomica-xyz/sparehour-worker.git sparehour-worker
python3 -m venv ~/.local/share/sparehour-worker/venv
~/.local/share/sparehour-worker/venv/bin/pip install ./sparehour ./sparehour-worker
export PATH="$HOME/.local/share/sparehour-worker/venv/bin:$PATH"
```

For always-on hosts, [`deploy/sparehour-worker.service`](deploy/sparehour-worker.service) is
an example per-user systemd unit matching this installation. Follow its
session-environment and lingering instructions for workstation idle detection.

## Upgrading an existing worker

This is a coordinated, breaking rebrand, not an alias layer. Install the matching
`sparehour` controller package and `sparehour-worker` worker package together;
Python imports are now `sparehour`, `sparehour_workloads`, and `sparehour_worker`.
The repositories are now `autonomica-xyz/sparehour` and
`autonomica-xyz/sparehour-worker`. Existing checkout directories can keep their names.

- Drain existing work before upgrading the controller and its workers. Existing
  databases, artifacts, models, credentials, and services are not moved or
  modified automatically.
- Copy your previous worker settings into `~/.config/sparehour/worker.json`
  (or keep an explicit `--config PATH`). Preserve mode 0600. Update the stored
  endpoint/token using `sparehour-worker login` and a freshly issued scoped token.
- Use `SPAREHOUR_WORKER_CONFIG` and `SPAREHOUR_WORKER_TOKEN` instead of the old
  worker environment variables. The default work directory is
  `~/.local/state/sparehour/worker/work`; move cached work explicitly if needed.
- Replace the old user service with `deploy/sparehour-worker.service` and restart
  it only after the matching controller is available. Reopen its newly written
  dashboard URL; old browser access keys do not carry across runs.
- Publish the controller cutover before worker CI runs. CI checks out
  `autonomica-xyz/sparehour`; set repository variable
  `SPAREHOUR_CONTROLLER_REF` to the matching controller commit to pin it.


## First run

From the controller checkout, issue an agent token using the same controller
database as the running service. Scope it to the intended workloads; repeat
`--scope` per value. Never give a worker an admin or client token.

```bash
umask 077
mkdir -p state
.venv/bin/sparehour token --role agent --subject tower-a \
    --scope routes=translate-batch --expires-hours 720 > state/tower-a.token
```

Transfer that file privately to the worker host as `token.txt`. Replace the
example endpoint below with your controller's HTTPS address.

Then on the worker host:

```bash
sparehour-worker login --endpoint https://gpu-control.example.com \
    --token-file token.txt --agent-id tower-a
sparehour-worker doctor                       # hardware, token, idle probe, queue matrix
sparehour-worker select translate-batch       # advertise a capability
sparehour-worker run                          # claim loop + dashboard on 127.0.0.1:7788
```

Open the full dashboard access URL printed in an interactive terminal, or
read it from `~/.config/sparehour/worker.json.dashboard-url`. Its private
per-run token is separate from the controller credential; do not share the
URL. The plain localhost URL alone does not authenticate a browser.

Useful variations:

```bash
sparehour-worker config set idle_only=true        # claim only while confirmed idle
sparehour-worker config set idle_after_seconds=600
sparehour-worker config set cpu_slots=4           # GPU-less box serving vendor=cpu routes
sparehour-worker run --cpu-slots 2                # one-run override, nothing saved
sparehour-worker run --once                       # smoke test: one job per slot, then exit
```

Dynamic (queue-driven) selection needs the depth signal on the token
(`--scope perms=dynamic`) plus `sparehour-worker config set dynamic=true` and a
`model_catalog` pointing at a lock catalog (local file or HTTPS URL) in the
controller's `models/*.lock.json` format; install the `[models]` extra for
Hugging Face pack downloads.

## Security and trust limits

- The endpoint must be HTTPS; only loopback endpoints are exempt. TLS
  verification is never disabled (use `SSL_CERT_FILE`/`SSL_CERT_DIR` for a
  private CA), and model downloads never receive the controller token.
- The controller and its workload registry are the code authority:
  executors are trusted commands, plugin handlers, or digest-pinned
  container images chosen by the registry — job submitters cannot pick
  executors or inject code/models. Run the worker under a user whose
  privileges you are comfortable granting the controller operator.
- The dashboard is not a remote management surface: literal loopback bind,
  per-run token, strict origin checks. Do not proxy or expose it.
- Provider credentials (Vast, Salad, …) live on the controller and are
  never sent to workers.
- One job per slot: a host with N registered slots runs at most N
  independent jobs concurrently — this is not multi-GPU training or model
  sharding.
- Fixed local model services (LibreTranslate, vLLM) are operator-started
  loopback services; the worker checks readiness but does not daemonize
  them, and one fixed endpoint must not be advertised as backed by
  several GPU slots.

## Commands

| command | purpose |
|---------|---------|
| `login` | store endpoint + agent token (config is mode 0600; `config` redacts it) |
| `doctor` | connectivity, hardware, token, idle probe, queue match table |
| `queues` | just the queue matrix |
| `select` / `deselect` | toggle advertised capabilities (live) |
| `prefetch` | docker-pull images for selected capabilities |
| `run` | the worker loop (`--once`, `--idle-only`, `--idle-after`, `--cpu-slots`, `--no-dashboard`) |
| `config` | get/set settings (`config set cpu_slots=4`); file: `~/.config/sparehour/worker.json` |

Key config keys: `capabilities` (empty claims nothing), `cpu_slots`,
`idle_only`, `idle_after_seconds`, `dynamic`, `model_catalog`, `workdir`,
`labels` (string→string placement labels), `dashboard_host`,
`dashboard_port`, `poll_seconds`, `heartbeat_seconds`, and `lease_seconds`.

## Known ceilings

- Job-level failures — for example a translation job requesting a language
  pair the local LibreTranslate hasn't loaded, or bad job input — are reported
  to the coordinator as failed jobs and the worker keeps running and claims
  the next job. Worker-level faults (model server down, GPU context errors,
  model load failures from other plugins) still drain: the running job is
  reported failed and the worker stops claiming.
- Capabilities are per-host, not per-GPU: a host advertises the same set
  on every slot.
- Idle detection fails closed: an unclassifiable session or a platform
  with no signal means "not idle" — `idle_only` then simply claims
  nothing.
- Dynamic selection never rewrites your saved capabilities. Downloaded
  model packs remain on disk until you prune them; container images are
  cached by Docker after the first pull.

## License

MIT — see [LICENSE](LICENSE). This worker depends on the `sparehour`
control-plane package, which is **GPL-3.0-or-later** and distributed
separately; installing this worker alongside it does not change its
license.
