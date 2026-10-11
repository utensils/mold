# `mold runpod`: native RunPod CLI

`mold runpod` manages RunPod cloud GPU pods end-to-end from the same binary
you use for local generation. Create a pod, connect to it, hand off logs to the
RunPod console, track spend, and (with `mold runpod run`) manage already-paired generation
in a single command.

Compared to the [Docker & RunPod](./docker) guide (which shows the manual
pod-creation flow via `runpodctl` or the web console) this guide covers
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

## Pair before generation

RunPod's provider API remains available for creating, listing and managing paid
resources. Provider status does not establish a trusted Mold connection. Create
a host explicitly, then use its trusted RunPod console to configure an
authenticated upgraded server and the [encrypted relay](/deployment/relay).
Set that host's `MOLD_PUBLIC_URL` to its RunPod proxy origin so the CLI can
associate the saved pairing with the tracked pod.

On the host, run `mold pairing create --output /private/invitation.json`. Transfer
the short-lived invitation privately and run
`mold pair --invite-file /private/invitation.json` on your client. Then:

```bash
eval "$(mold runpod connect <pod-id> --check)"
mold runpod run "a cinematic low-angle of a tiny steel robot"
```

`runpod run` requires an existing paired pod selected by `MOLD_HOST` or the
tracked pod. It checks that exact running pod and refuses GPU, datacenter or
volume mismatches before creating or deleting any resource. Use explicit
`create`, then pair and connect, when you want a different host. It streams generation and media through the encrypted client;
`connect --check` verifies application readiness through that same connection.
Changing the host creates a new trust decision and requires a new invitation.
A running pod is not deleted just because it refuses an old plaintext proxy
probe. The provider's web console remains the supported logs and owner-control
surface; raw public HTTP and browser media URLs are no longer an API transport.

The encrypted Studio supports Library browsing, thumbnails, video Range reads,
exports and progress. Open the trusted Studio and pair the host. API keys from
the old proxy cannot be silently migrated into a secure host identity.

### Common flags

```bash
mold runpod run "a cat" --model flux-dev:q4        # preload a specific model
mold runpod run "a cat" --gpu 5090                 # require the paired GPU family
mold runpod run "a cat" --dc US-IL-1               # require the paired datacenter
mold runpod run "a cat" --network-volume <id>       # require the paired volume
mold runpod run "a cat" --keep                     # keep the pod running (skips auto-teardown / idle reap)
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

# Persistent network volumes
mold runpod network-volume list
mold runpod network-volume get <volume-id>
mold runpod network-volume create --name models --size 100 --dc US-KS-2
mold runpod network-volume update <volume-id> --size 200
mold runpod network-volume update <volume-id> --name shared-models
mold runpod network-volume delete <volume-id>       # permanently deletes all data

# Connecting
mold runpod connect <pod-id>                    # print export MOLD_HOST=…
eval "$(mold runpod connect <pod-id>)"          # exec the export in your shell

# Observability
mold runpod logs <pod-id>                       # validate pod + print console handoff
```

## Smart defaults explained

When a network volume is selected, it replaces the ordinary Pod workspace
disk at `/workspace`. Mold forces Secure Cloud and pins deployment to the
volume's datacenter, even if `--cloud`, `--dc`, or their config defaults say
otherwise. Production currently accepts 10–3999 GB. A volume can be renamed
and grown within that range, but cannot shrink.
It remains billable independently of pods until explicitly deleted, and must be
detached by deleting its pod before the volume itself can be deleted.

When `mold runpod create` (or `run`) is invoked without `--gpu`/`--dc`:

1. `RunPodClient::gpu_types()` aggregates the highest stock signal per GPU
   across all datacenters (via GraphQL; the REST API doesn't expose
   this).
2. Each candidate must also clear the VRAM the requested model is estimated
   to need (`estimated_vram_need_gb`, 18 GB when no model is named). The
   first family with **High** or **Medium** stock that clears it wins,
   walking a ten-entry ladder in ascending-VRAM order so small models do not
   overshoot: RTX 4090, RTX 3090, RTX 5090, L40, L40S, RTX A6000,
   A100 PCIe, A100 SXM, H100 NVL, H100 SXM. If none matches, any
   High-stock "interesting" GPU with enough VRAM is taken as a fallback;
   otherwise the command fails and asks for an explicit `--gpu`.
3. The displayed image target comes from the shared GPU table:
   - A30/A100 and generic Ampere → `:<version>-sm80`
   - A2/A10/A16/A40, RTX A4000–A6000, RTX 3050–3090 → `:<version>-sm86`
   - Ada (4090, L40S) → `:<version>`
   - H100/H200 → `:<version>-sm90`
   - B200/B300 → `:<version>-sm100`
   - named RTX PRO/GeForce RTX 50-series → `:<version>-sm120`
   - ambiguous generic Blackwell → `:<version>` instead of guessing between
     incompatible datacenter and consumer targets
     Rolling/source/Nix binaries submit mutable `latest*` tags. Stable official
     binaries fetch the exact release/source digest manifest and submit
     `ghcr.io/utensils/mold@sha256:…`; missing or inconsistent manifests fail
     closed instead of falling back to a tag. An explicit custom stable
     repository must itself be supplied as `@sha256`.
     B200 support is simulated, not hardware-qualified; this routing does not
     represent a real B200 acceptance run.
     GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.
     RunPod CLI and desktop reject those Grace systems before constructing or
     submitting an image reference.
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

Other runpod settings are config-only (no env-var override); they rarely
change between runs.

## State files

`mold runpod` persists two files under `$MOLD_HOME/` (default `~/.mold/`):

- `runpod-state.json`: warm-pod pointer (`last_pod_id`, timestamps, cached
  GPU + cost). Used by `run` for reuse detection.
- `runpod-history.jsonl`: append-only log of pod lifetime events with
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

The key is read via `ExecStartPre` and injected into the service environment.
It is never written into the Nix store. The same pattern applies to
`hfTokenFile`, `civitaiTokenFile`, and `apiKeyFile`.

## REST vs GraphQL

`RunPodClient` hits RunPod's REST API at `https://rest.runpod.io/v1/` for pod
lifecycle (create/list/get/stop/start/delete) and network-volume management,
and uses the GraphQL endpoint at `https://api.runpod.io/graphql` for account
info, GPU catalog, and datacenter availability; those aren't exposed via
REST. Container/system logs remain a RunPod console-only surface.

Both paths use the same API key (`Authorization: Bearer …`).

## Troubleshooting

**Pod is not ready or does not match the requested GPU**: `runpod run` refuses
that host without replacing it or creating another paid pod. Inspect the
provider console, explicitly create a suitable replacement if needed, then
pair and connect it before generation.

**"value must be one of …" from `/pods`**: you pinned a datacenter that
isn't in RunPod's REST enum whitelist. GraphQL exposes datacenters that
REST doesn't accept. Omit `--dc` to let RunPod pick.

**"RunPod /user 401" or "…403"**: stale/invalid/missing API key. Run
`mold runpod doctor` to confirm. Regenerate at
[runpod.io/console/user/settings](https://www.runpod.io/console/user/settings).

**Pod logs**: RunPod does not expose container/system logs through the Pod
REST API. `mold runpod logs <pod-id>` validates the Pod, prints its state and
the console URL, then tells you to open that Pod's **Logs** panel. `--follow`
is retained for compatibility but cannot stream console-only logs.

**Resources after Ctrl-C**: `mold runpod run` uses an already paired pod;
explicitly created pods remain visible in `mold runpod list`. Delete a pod you
no longer need with `mold runpod delete <id>`.
