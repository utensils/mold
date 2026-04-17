# `mold runpod` — native RunPod CLI

`mold runpod` manages RunPod cloud GPU pods end-to-end from the same binary
you use for local generation. Create a pod, connect to it, stream logs,
track spend, and (with `mold runpod run`) create-generate-save in a single
command.

Compared to the [Docker & RunPod](./docker) guide — which shows the manual
pod-creation flow via `runpodctl` or the web console — this guide covers
the integrated workflow.

## Setup

Generate a RunPod API key at
[runpod.io/console/user/settings](https://www.runpod.io/console/user/settings)
(Settings → **API Keys**, "Read/Write" scope). Then:

```bash
# Option 1: persist to config.toml
mold config set runpod.api_key <your-key>

# Option 2: env var (overrides config)
export RUNPOD_API_KEY=<your-key>

# Verify
mold runpod doctor
```

`mold runpod doctor` checks the key, the REST endpoint, and your RunPod
balance + spend rate.

## The killer feature — `mold runpod run`

Generate an image on a fresh or reused pod with one command:

```bash
mold runpod run "a cinematic low-angle of a tiny steel robot"
```

What this does:

1. If a warm pod exists (created by a prior `run`), reuses it.
2. Otherwise creates a new pod with smart defaults:
   - picks the cheapest GPU with High or Medium stock (4090 → 5090 → L40S → A100),
   - selects the matching `ghcr.io/utensils/mold` image tag,
   - retries across datacenters if scheduling stalls.
3. Waits for the mold server inside the pod to be reachable, streaming a
   readiness progress bar.
4. Calls `/api/generate/stream` — SSE events drive a live progress display
   (model pull, weight load, denoise steps).
5. Saves the output to `./mold-outputs/runpod-<pod-id>-<timestamp>.png`
   (directory auto-created, `.gitignore`'d by default).
6. Leaves the pod **warm** for reuse on the next `run`. Pass `--keep` to
   leave it running explicitly, or set `runpod.auto_teardown = true` in
   config to delete after each generation.

### Common flags

```bash
mold runpod run "a cat" --model flux-dev:q4        # preload a specific model
mold runpod run "a cat" --gpu 5090                 # force a GPU family
mold runpod run "a cat" --dc US-IL-1               # pin a datacenter
mold runpod run "a cat" --keep                     # don't park pod for reuse
mold runpod run "a cat" --steps 28 --seed 42       # forward standard gen flags
mold runpod run "a cat" --output-dir ./renders     # custom save path
```

## Manual pod management

```bash
# Discovery
mold runpod gpus                       # GPU types with aggregate stock
mold runpod gpus --json
mold runpod datacenters --gpu "RTX 5090"
mold runpod usage --since 7d           # spend summary + pod history

# Lifecycle
mold runpod create --gpu 5090 --volume 50  # smart defaults fill the rest
mold runpod create --dry-run               # print plan without creating
mold runpod list
mold runpod get <pod-id>
mold runpod stop <pod-id>                  # pause billing, keep storage
mold runpod start <pod-id>                 # resume
mold runpod delete <pod-id>                # tear down

# Connecting
mold runpod connect <pod-id>                    # print export MOLD_HOST=…
eval "$(mold runpod connect <pod-id>)"          # exec the export in your shell

# Observability
mold runpod logs <pod-id>                       # one-shot
mold runpod logs <pod-id> --follow              # tail (polls every 2s)
```

## Smart defaults explained

When `mold runpod create` (or `run`) is invoked without `--gpu`/`--dc`:

1. `RunPodClient::gpu_types()` aggregates the highest stock signal per GPU
   across all datacenters (via GraphQL — the REST API doesn't expose
   this).
2. The cheapest family with **High** or **Medium** stock wins, from the
   preference list `4090 > 5090 > L40S > A100`.
3. The image tag is derived from the GPU family:
   - Ada (4090, L40S) → `:latest`
   - Ampere (A100, 3090) → `:latest-sm80`
   - Blackwell (5090) → `:latest-sm120`
4. Datacenter is left unset; RunPod's scheduler picks any machine it can
   place. If that fails, `ensure_pod` retries across stock-ranked DCs with
   a 90-second schedule timeout per attempt, deleting stuck pods before
   moving on.

## Configuration reference

`config.toml` (`~/.config/mold/config.toml` or `~/.mold/config.toml`)
supports a `[runpod]` section:

```toml
[runpod]
# api_key = "rpa_..."                   # Prefer RUNPOD_API_KEY env var
default_gpu = "RTX 5090"                # Override auto-pick
default_datacenter = "EUR-IS-2"
default_network_volume_id = "nv-abc123" # Attach to every new pod
auto_teardown = false                   # true = delete after each `run`
auto_teardown_idle_mins = 20            # Idle reap window
cost_alert_usd = 3.0                    # Abort if session spend exceeds
# endpoint = "https://rest.runpod.io/v1"  # Override (mostly for testing)
```

All keys are settable via `mold config set runpod.<key> <value>`.

## Env-var precedence

| Variable         | Purpose                           |
| ---------------- | --------------------------------- |
| `RUNPOD_API_KEY` | Overrides `config.runpod.api_key` |

Other runpod settings are config-only (no env-var override) — they rarely
change between runs.

## State files

`mold runpod` persists two files under `$MOLD_HOME/` (default `~/.mold/`):

- `runpod-state.json` — warm-pod pointer (`last_pod_id`, timestamps, cached
  GPU + cost). Used by `run` for reuse detection.
- `runpod-history.jsonl` — append-only log of pod lifetime events with
  cost and prompt metadata. Used by `mold runpod usage --since <win>`.

Delete these any time to reset state; they're caches, not sources of truth.

## NixOS integration

The `services.mold` module supports a `runpodApiKeyFile` option:

```nix
services.mold = {
  enable = true;
  package = inputs.mold.packages.${system}.default;
  runpodApiKeyFile = config.age.secrets.runpod-key.path;
};
```

The key is read via `ExecStartPre` and injected into the service
environment — never written into the Nix store. Same pattern as
`hfTokenFile` and `apiKeyFile`.

## REST vs GraphQL

`RunPodClient` hits RunPod's REST API at `https://rest.runpod.io/v1/`
for pod lifecycle (create/list/get/stop/start/delete/logs) and uses
the GraphQL endpoint at `https://api.runpod.io/graphql` for account
info, GPU catalog, and datacenter availability — those aren't exposed
via REST.

Both paths use the same API key (`Authorization: Bearer …`).

## Troubleshooting

**"pod didn't schedule within 90s"** — the datacenter likely has no real
capacity despite a High/Medium stock signal. `mold runpod run` will
automatically try the next candidate. If all fail, RunPod is out of
capacity for that GPU family right now. Retry later, pick a different
GPU, or fall back to a local GPU host.

**"value must be one of …" from `/pods`** — you pinned a datacenter that
isn't in RunPod's REST enum whitelist. GraphQL exposes datacenters that
REST doesn't accept. Omit `--dc` to let RunPod pick.

**"RunPod /user 401" or "…403"** — stale/invalid/missing API key. Run
`mold runpod doctor` to confirm. Regenerate at
[runpod.io/console/user/settings](https://www.runpod.io/console/user/settings).

**Orphaned pods after Ctrl-C** — `mold runpod run` persists `last_pod_id`
**before** waiting for readiness, so `mold runpod list` always surfaces
zombie pods. Delete with `mold runpod delete <id>`.
