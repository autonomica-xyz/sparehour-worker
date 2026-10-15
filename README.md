# distributed-worker (dworker)

The worker side of [distributed-gpu](../distributed-gpu): a CLI + local web
dashboard that turns any workstation or server into a SETI@home-style
opportunistic compute node. Workers register their GPUs (NVIDIA or AMD) or
spare CPU, advertise the capabilities you select, and claim jobs from the
control plane whenever the machine has nothing better to do.

## What it does

- **Credentials in, queues out** — `dworker login` stores the endpoint and
  scoped agent token in a mode-0600 file; `doctor`/`queues`/dashboard show
  every workload route the control plane offers and whether *this* machine
  matches (vendor, VRAM, compute capability, GPU model, model packs,
  labels), with the reason when it doesn't.
- **Selection** — you pick which capabilities to advertise
  (`dworker select demo-gpu`, or the Run/Stop buttons in the dashboard).
  Toggles re-register live; running jobs are never interrupted.
- **NVIDIA + AMD hosts** — NVIDIA slots come from `nvidia-smi`, AMD slots
  from the amdgpu sysfs tree (no ROCm needed to *register*). Executing AMD
  jobs additionally requires an accessible `/dev/kfd` and `/dev/dri`, plus a
  stable GPU identity: the runner selects the device with
  `ROCR_VISIBLE_DEVICES=<GPU-<16 hex>>` derived from sysfs `unique_id` (the
  same UUID rocm-smi/rocminfo print) and passes `/dev/kfd` + `/dev/dri`
  through to containers. A card without a nonzero `unique_id` keeps only a
  `pci-<BDF>` identity — it stays discoverable for diagnostics, but the
  runner refuses ROCm execution for it rather than guess a device ordinal.
- **CPU-only spare compute** — `dworker config set cpu_slots=N` registers N
  `vendor=cpu` slots so GPU-less boxes can serve `vendor: cpu` routes.
  Reported cores/RAM are usable capacity — CPU affinity and cgroup v1/v2
  cpu/memory limits cap the host-wide readings — and the N slots partition
  host RAM so concurrent claims do not each advertise all of it.
  `dworker run --cpu-slots M` overrides the count for that run only.
- **Idle / screen-lock detection** — `idle_only=true` claims work only while
  the machine is confirmed idle: all local interactive sessions locked, or
  quiet past `idle_after_seconds` (default 300). Linux reads logind session
  classes, `LockedHint`, and `IdleSinceHint`/`IdleSinceHintMonotonic`
  timestamps (plus xprintidle for a reachable X display); macOS reads
  `IOConsoleLocked` + `HIDIdleTime` via ioreg; Windows uses
  GetLastInputInfo/GetTickCount + OpenInputDesktop. Only a *confirmed*
  headless machine (logind answered and no local interactive session
  exists, and no X11 timer) counts as idle; when the signal is missing or
  ambiguous the probe **fails closed** — no claiming — so the worker never
  steals cycles from a session it cannot see. A job that already started
  always finishes.
- **Model/image prefetch** — `dworker prefetch` (or the dashboard button)
  docker-pulls every digest-pinned image behind your selected capabilities
  and tells you what plugin model packs or local model servers a route
  still needs. Prefetch is single-flight per run, each `docker pull` is
  bounded (600 s default), and a worker stop cancels it instead of
  stranding a pull.
- **Dynamic workloads** — with `dynamic=true` and a token carrying
  `perms=dynamic`, the worker watches per-route queue depth and advertises
  exactly the capabilities that have pending work, switching queues as
  demand moves. (`perms=dynamic` grants only that depth signal; it does not
  widen what the token may do.) Before the first claim — including a
  `run --once` — one provisioning cycle runs synchronously. Demanded model
  packs install into *every* target directory the routes name
  (`ARGOS_MODEL_DIR` or the workload's family-specific `*_MODEL_DIR`, with a
  `<workdir>/model-packs/<pack>` fallback), SHA-256-verified per file.
  Verification identity comes from the `model_catalog` lock entry, or —
  with no catalog configured — from the route's own identity pins, so
  preinstalled packs still run but can never be downloaded without a
  catalog. Provisioning never deletes data it did not create: a target
  directory holding anything beyond the pack's own manifest-listed content
  (or a symlinked path component) is refused and preserved. Dynamic
  switching is runtime-only — it never rewrites your saved `capabilities`,
  and CLI overrides like `--cpu-slots` never reach disk either; the file
  keeps what the operator saved. The manager does not delete downloaded
  packs; prune `model-packs/` manually if disk matters. Cancel/drain stops
  the manager and prefetch immediately; running compute still drains.
- **Dashboard** — `dworker run` serves a page on a **literal loopback
  address only** (default `http://127.0.0.1:7788`; any other bind address
  is rejected). Every `/api/*` call requires `Authorization: Bearer
  <token>` where the token is a per-run secret unrelated to the
  control-plane token, delivered to your browser in the URL fragment. The
  full URL is written atomically, mode 0600, next to the config file as
  `config.json.dashboard-url`, and printed only on an interactive TTY. Host
  and Origin headers must match the bound origin, and request bodies are
  strictly typed JSON (exact booleans/strings — no coercion). The status
  snapshot it serves carries true succeeded/failed job counts from
  confirmed controller outcomes.

## Install

The worker reuses the control plane's client and executor machinery, so
install that package first (from your controller checkout or its wheel).

Canonical service install (matches `deploy/dworker.service`):

```bash
python3 -m venv ~/.local/share/dworker/venv
~/.local/share/dworker/venv/bin/pip install /path/to/distributed-gpu   # or its wheel from dist/
~/.local/share/dworker/venv/bin/pip install /path/to/distributed-worker
```

Development install (separate, activated environment):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install /path/to/distributed-gpu
pip install -e '.[dev]'
pytest
```

## First run on a host

```bash
# On the controller: issue a least-privilege, per-agent scoped token.
# Repeat --scope per value (never CSV); cap it to the routes you select and
# give it an expiry.
#   dgpu token --role agent --subject tower-a \
#       --scope routes=demo-gpu --expires-hours 720
# Never hand the worker an admin or client token: the control plane decides
# which executor each route runs, and an agent token only registers slots
# and claims work for its subject and scoped capabilities.

dworker login --endpoint https://gpu-control.example.com --token-file token.txt --agent-id tower-a
dworker doctor                 # hardware, token, idle detection, queue matrix
dworker select demo-gpu        # advertise the capability
dworker run                    # claim loop + dashboard on 127.0.0.1:7788
```

SETI@home mode (only when the machine is confirmed idle/locked):

```bash
dworker config set idle_only=true
dworker config set idle_after_seconds=600
```

Dynamic workloads (queue-depth-driven selection):

```bash
# On the controller: grant the depth signal on the same agent token —
# repeat --scope, do not comma-join values.
dgpu token --role agent --subject tower-a \
    --scope routes=demo-gpu --scope routes=demo-cpu \
    --scope perms=dynamic --expires-hours 720
# On the worker:
dworker login --endpoint https://gpu-control.example.com --token-file token.txt
dworker config set dynamic=true
dworker config set model_catalog=https://gpu-control.example.com/locks/argos-popular.json
dworker run
```

The lock catalog is the control plane's `models/*.lock.json` format
(`{"models": [...]}`) — serve it over HTTPS or point at a local file; HTTP
URLs and URLs carrying credentials are refused. Argos entries pin an archive
URL and SHA-256; Hugging Face entries use `format: "huggingface"`, `model_id`,
an immutable 40-hex `revision`, and reviewed `allow_patterns`. Provisioning
checks the source file identities and writes a per-file SHA-256 manifest.
Install the `[models]` extra for Hugging Face provisioning; the large inference
runtimes remain separate, pinned profiles under
[`distributed-gpu/deploy`](../distributed-gpu/deploy).

Model packs install into the family-specific directory named by the route
(`ARGOS_MODEL_DIR`, `SDXL_MODEL_DIR`, `PARAKEET_MODEL_DIR`,
`WHISPERX_MODEL_DIR`, alignment/diarization directories, or `VLLM_MODEL_DIR`)
— make that parent writable by the worker user — or into
`<workdir>/model-packs/<pack>` when no directory is pinned. Plugin routes
whose handler module isn't installed are skipped; packs that are neither in
the catalog nor already pin-verified on disk are skipped too; both show up
in the notes.

LibreTranslate and vLLM are operator-started, loopback-only model services;
dynamic mode checks readiness but does not create a service daemon. Provision
the fixed Qwen pack before starting vLLM, then enable its sleep/wake profile
from [`deploy/llm/requirements.txt`](../distributed-gpu/deploy/llm/requirements.txt).
The worker retains warm plugin instances, releases GPU models on handler
switches, and drains after a failed release rather than allocating the next
model beside one whose memory ownership is uncertain.

Smoke test one job per slot then exit: `dworker run --once`. Dynamic mode
provisions before the first claim, so `--once` sees real demand.

For always-on servers, see `deploy/dworker.service`.

## Connecting to a public controller

- **Endpoint must be HTTPS.** `dworker login` and the config loader reject
  non-HTTPS endpoints except loopback (`localhost`, `127.0.0.0/8`, `::1`),
  which is allowed for local development. TLS verification is never
  disabled; to trust a private CA set `SSL_CERT_FILE`/`SSL_CERT_DIR`.
  The controller bearer transport ignores ambient proxy and netrc settings;
  model downloads never receive the controller token.
- **Terminate TLS where the controller runs.** A reverse proxy
  (nginx/Caddy/Apache or a cloud LB) terminating TLS and forwarding to the
  controller is the normal deployment; the worker only needs the public
  `https://` origin. Keep firewall rules tight: expose only the controller
  port, not the worker — the worker makes outbound calls only.
- **Least-privilege tokens.** Issue one token per agent `--subject`, repeat
  `--scope routes=CAPABILITY` for each advertised capability, add
  `--scope perms=dynamic` only if the host runs dynamic mode, and set
  `--expires-hours`. Rotate with `dgpu revoke-token`. Worker tokens are
  never admin tokens.
- **Executor trust.** The controller registry chooses trusted commands,
  plugin handlers, and digest-pinned container images; job submitters do not
  choose executors. Commands and plugins are local code, not sandboxed by
  this worker. Treat the controller and registry as code authority with the
  worker user's privileges.
- **Dashboard stays local.** The dashboard binds literal loopback only,
  and its bearer token is a separate per-run secret stored in the
  mode-0600 `config.json.dashboard-url` file. It is not a remote
  management surface; do not proxy or expose it.

## Commands

| command | purpose |
|---------|---------|
| `login` | store endpoint + agent token (config file is mode 0600; `config` output redacts the token) |
| `doctor` | connectivity, hardware, token, idle probe, queue match table |
| `queues` | just the queue matrix |
| `select` / `deselect CAP…` | toggle advertised capabilities (live) |
| `prefetch` | docker-pull images for selected capabilities |
| `run` | the worker loop (`--once`, `--idle-only`, `--idle-after`, `--cpu-slots`, `--no-dashboard`) |
| `config` | get/set any setting (`config set cpu_slots=4`) |

## Configuration

`~/.config/dworker/config.json` (override with `--config` or
`DWORKER_CONFIG`). Saves are validated, atomic (temp file + rename), and
mode 0600. Values are typed: booleans accept only
true/false/yes/no/on/off/1/0, integers are range-checked, `labels` must be
a string→string map — malformed values are rejected, not coerced.

| key | default | meaning |
|-----|---------|---------|
| `endpoint`, `token`, `agent_id` | — | control plane access |
| `provider` | `local` | provider label reported to the control plane |
| `capabilities` | `[]` | advertised capabilities; empty claims nothing |
| `model_packs` | `[]` | model packs installed on this host |
| `labels` | `{}` | placement labels (agent + slots) |
| `cpu_slots` | `0` | synthetic `vendor=cpu` slots (RAM is partitioned across them) |
| `idle_only` | `false` | only claim while confirmed idle/locked |
| `idle_after_seconds` | `300` | idle threshold |
| `dynamic` | `false` | queue-depth-driven capability switching (needs `perms=dynamic` token) |
| `dynamic_poll_seconds` | `60` | how often the dynamic manager re-checks depth |
| `model_catalog` | `""` | lock catalog (local file or HTTPS URL) for model-pack downloads |
| `workdir` | `~/.local/state/dworker/work` | job working directories and pack fallback |
| `dashboard` | `true` | serve the dashboard during `run` |
| `dashboard_host` / `dashboard_port` | `127.0.0.1` / `7788` | dashboard bind — literal loopback only |
| `poll_seconds`, `heartbeat_seconds`, `lease_seconds` | `1` / `15` / `90` | control-plane loop timings |

## Known ceilings

- AMD execution requires `/dev/kfd` and `/dev/dri` accessible by the
  worker user and a card with a nonzero sysfs `unique_id`; otherwise amd
  slots register but cannot run ROCm work.
- Idle detection fails closed: a session it cannot classify, or a platform
  with no signal at all, means "not idle". Unconfirmed machines never look
  idle — `idle_only` on an unrecognized setup simply claims nothing until
  you fix detection or disable `idle_only`.
- Capabilities are per-host, not per-GPU; a host advertises the same set on
  every slot (same as the controller's own agent).
  A fixed LibreTranslate/vLLM endpoint is not a per-GPU server pool: expose
  only its GPU to that worker (for example, a one-GPU worker container).
  Do not advertise one service as independently backed by several GPU slots.
- Dynamic mode is runtime-only selection: it does not rewrite the saved
  `capabilities` config, and the first container job for a newly demanded
  capability pays the image pull (subsequent jobs hit the docker cache).

## License

MIT — see [LICENSE](LICENSE). Note: this worker depends on the
[distributed-gpu](../distributed-gpu) control-plane package, which is
GPL-3.0-or-later and is distributed separately — its license is unchanged
by this project's license, and it must be available (checkout or wheel)
until it is published.
