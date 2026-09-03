# distributed-worker (dworker)

The worker side of [distributed-gpu](../distributed-gpu): a CLI + local web
dashboard that turns any workstation or server into a SETI@home-style
opportunistic compute node. Workers register their GPUs (NVIDIA or AMD) or
spare CPU, advertise the capabilities you select, and claim jobs from the
control plane whenever the machine has nothing better to do.

## What it does

- **Credentials in, queues out** — `dworker login` stores the endpoint and
  scoped agent token; `doctor`/`queues`/dashboard show every workload route
  the control plane offers and whether *this* machine matches (vendor, VRAM,
  compute capability, GPU model, model packs, labels), with the reason when
  it doesn't.
- **Selection** — you pick which capabilities to advertise
  (`dworker select demo-gpu`, or the Run/Stop buttons in the dashboard).
  Toggles re-register live; running jobs are never interrupted.
- **NVIDIA + AMD hosts** — NVIDIA slots come from `nvidia-smi`, AMD slots from
  the amdgpu sysfs tree (no ROCm needed to register). AMD jobs get ROCm device
  visibility (`HIP_VISIBLE_DEVICES`) and `/dev/kfd` + `/dev/dri` container
  passthrough instead of nvidia-container `--gpus`.
- **CPU-only spare compute** — `config set cpu_slots=N` registers N
  `vendor=cpu` slots so GPU-less boxes can serve `vendor: cpu` routes.
- **Idle / screen-lock detection** — `idle_only=true` claims work only while
  the screen is locked or input has been quiet for `idle_after_seconds`
  (default 300). Linux uses loginctl + xprintidle, macOS ioreg HID idle time,
  Windows GetLastInputInfo/OpenInputDesktop. Headless machines are always
  idle. A job that already started always finishes.
- **Model/image prefetch** — `dworker prefetch` docker-pulls every
  digest-pinned image behind your selected capabilities and tells you what
  plugin model packs or local model servers a route still needs.
- **Dashboard** — `dworker run` serves a local page (default
  `http://127.0.0.1:7788`) with slots, live job state, idle status, the queue
  matrix with Run/Stop toggles, pause/resume, and prefetch.

## Install

The worker reuses the control plane's client and executor machinery, so
install that package first (from your controller checkout or its wheel):

```bash
python3 -m venv .venv
.venv/bin/pip install /path/to/distributed-gpu   # or its wheel from dist/
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## First run on a host

```bash
# On the controller: issue a scoped agent token
#   dgpu token --role agent --subject tower-a --scope routes=demo-gpu

dworker login --endpoint https://gpu-control.example.com --token-file token.txt --agent-id tower-a
dworker doctor                 # hardware, token, idle detection, queue matrix
dworker select demo-gpu        # advertise the capability
dworker run                    # claim loop + dashboard on 127.0.0.1:7788
```

SETI@home mode (only when the machine is idle/locked):

```bash
dworker config set idle_only=true
dworker config set idle_after_seconds=600
```

Smoke test one job per slot then exit: `dworker run --once`.

For always-on servers, see `deploy/dworker.service`.

## Commands

| command | purpose |
|---------|---------|
| `login` | store endpoint + agent token (file is chmod 600) |
| `doctor` | connectivity, hardware, token, idle probe, queue match table |
| `queues` | just the queue matrix |
| `select` / `deselect CAP…` | toggle advertised capabilities (live) |
| `prefetch` | docker-pull images for selected capabilities |
| `run` | the worker loop (`--once`, `--idle-only`, `--idle-after`, `--cpu-slots`, `--no-dashboard`) |
| `config` | get/set any setting (`config set cpu_slots=4`) |

## Configuration

`~/.config/dworker/config.json` (override with `--config` or `DWORKER_CONFIG`):

| key | default | meaning |
|-----|---------|---------|
| `endpoint`, `token`, `agent_id` | — | control plane access |
| `capabilities` | `[]` | advertised capabilities; empty claims nothing |
| `model_packs` | `[]` | model packs installed on this host |
| `labels` | `{}` | placement labels (agent + slots) |
| `cpu_slots` | `0` | synthetic `vendor=cpu` slots |
| `idle_only` | `false` | only claim while idle/locked |
| `idle_after_seconds` | `300` | idle threshold |
| `dashboard_host` / `dashboard_port` | `127.0.0.1` / `7788` | local dashboard |

## Known ceilings

- AMD `container` jobs require the control plane image to actually support
  ROCm (the passthrough is handled here; image content is route policy).
- Idle detection is per-machine best effort; Wayland without logind idle
  hints falls back to "always idle" (which is the safe direction for a
  workstation you also use interactively: it keeps claiming while you work —
  set `idle_only=false` if you never want that).
- Capabilities are per-host, not per-GPU; a host advertises the same set on
  every slot (same as the controller's own agent).

## License

MIT — see [LICENSE](LICENSE). Note: this worker depends on the
[distributed-gpu](../distributed-gpu) control-plane package, which must be
available (checkout or wheel) until it is published.
