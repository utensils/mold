# Docker & RunPod

Run mold on any NVIDIA GPU host with Docker, including cloud GPU providers like
[RunPod](https://www.runpod.io/).

## Building

::: code-group

```bash [Ada]
docker build -t mold-server .
```

```bash [Hopper]
docker build --build-arg CUDA_COMPUTE_CAP=90 -t mold-server-h100 .
```

```bash [Ampere]
docker build --build-arg CUDA_COMPUTE_CAP=80 -t mold-server-a100 .
```

```bash [Blackwell]
docker build --build-arg CUDA_COMPUTE_CAP=120 -t mold-server-b200 .
```

:::

## Pre-Built Images

Images are published to GHCR on every push to `main` and on version tags:

```bash
# Ada (RTX 4090) — default
docker pull ghcr.io/utensils/mold:latest

# Ampere (A100)
docker pull ghcr.io/utensils/mold:latest-sm80

# Blackwell (RTX 5090)
docker pull ghcr.io/utensils/mold:latest-sm120
```

## Running

::: code-group

```bash [Basic]
docker run --gpus all -p 7680:7680 ghcr.io/utensils/mold:latest
```

```bash [With Models]
docker run --gpus all -p 7680:7680 \
  -v ~/.mold:/workspace/.mold \
  ghcr.io/utensils/mold:latest
```

:::

## RunPod Deployment

The pre-built GHCR images are the fastest path. Pick the tag that matches your
GPU's compute capability:

| GPU family                 | Image tag                            |
| -------------------------- | ------------------------------------ |
| Ada (RTX 4090, L40S)       | `ghcr.io/utensils/mold:latest`       |
| Ampere (A100, RTX 3090)    | `ghcr.io/utensils/mold:latest-sm80`  |
| Blackwell (RTX 5090, B200) | `ghcr.io/utensils/mold:latest-sm120` |

### Option 1 — Web Console

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods) → **Deploy**.
2. Pick a GPU (RTX 4090 is the sweet spot — see [GPUs](#recommended-gpus)).
3. **Container image**: `ghcr.io/utensils/mold:latest`.
4. **HTTP ports**: `7680`.
5. **Volume**: mount at `/workspace` (50 GB is plenty for a test; attach a
   **network volume** there if you want models to survive pod deletion).
6. **Environment variables** (optional): `MOLD_DEFAULT_MODEL=flux2-klein:q8`,
   `MOLD_LOG=info`, `HF_TOKEN=<your-token>` for gated models.
7. Deploy. When the pod reaches `RUNNING` and the container starts, mold
   is reachable at `https://<pod-id>-7680.proxy.runpod.net`.

### Option 2 — `runpodctl` CLI

Install from [runpod.io docs](https://docs.runpod.io/runpodctl/install-runpodctl)
(or use our Nix devshell, which ships it), then authenticate once:

```bash
runpodctl doctor   # prompts for API key, saves to ~/.runpod/config.toml
```

Spin up a pod:

```bash
runpodctl pod create \
  --name mold-server \
  --image ghcr.io/utensils/mold:latest \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --cloud-type SECURE \
  --gpu-count 1 \
  --container-disk-in-gb 20 \
  --volume-in-gb 50 \
  --volume-mount-path /workspace \
  --ports "7680/http,22/tcp" \
  --env '{"MOLD_DEFAULT_MODEL":"flux2-klein:q8","MOLD_LOG":"info"}'
```

The response is JSON — grab `.id` for subsequent commands.

::: tip Community vs Secure cloud
`--cloud-type COMMUNITY` is ~30-50 % cheaper but 4090 stock is often "Low" —
you'll get a `This machine does not have the resources to deploy your pod`
error. Retry with `SECURE` if that happens.
:::

::: warning Pick a datacenter with stock
`runpodctl gpu list` and `runpodctl datacenter list` both expose
`stockStatus`. If it is empty or `Low`, the pod can be "Rented" but never
actually schedule — `uptimeSeconds` stays at `0` and `runtime` is `null`.
Pin to a `High`-stock datacenter with `--data-center-ids <id>`. The GraphQL
backend only accepts a single datacenter id; if you pass more, `runpodctl`
silently uses the first.

When 4090 stock is thin, the RTX 5090 (`NVIDIA GeForce RTX 5090`, image
`ghcr.io/utensils/mold:latest-sm120`) is usually easier to get.
:::

### Connecting Once the Pod is Up

```bash
POD=<pod-id>
export MOLD_HOST="https://${POD}-7680.proxy.runpod.net"

# Health checks
curl "$MOLD_HOST/health"
curl "$MOLD_HOST/api/status" | jq

# Generate
mold run "a cinematic portrait"

# Open the bundled web gallery
open "$MOLD_HOST/"   # macOS; use `xdg-open` on Linux
```

::: tip Web gallery is bundled
The image includes a Vue 3 gallery SPA at `/opt/mold/web` — visiting
`https://${POD}-7680.proxy.runpod.net/` in a browser lists every output
in the server's output directory with real thumbnails (MP4 first frames
included), metadata panels, and download / copy-prompt actions. Set
`MOLD_GALLERY_ALLOW_DELETE=1` in the pod env to enable the delete button.
:::

::: warning Proxy Timeout
RunPod's Cloudflare proxy has a **100-second timeout**. Use the SSE streaming
endpoint (`/api/generate/stream`) for long generations — mold's client does this
automatically.
:::

### Teardown

```bash
runpodctl pod stop   <pod-id>   # pause billing (keeps storage)
runpodctl pod start  <pod-id>   # resume
runpodctl pod delete <pod-id>   # fully remove
```

### Network Volume (Optional)

For long-lived setups, create a [network
volume](https://docs.runpod.io/pods/storage/create-network-volumes) and mount
it at `/workspace`. The entrypoint (`docker/start.sh`) auto-detects this and
relocates:

- Models: `/workspace/.mold/models`
- HuggingFace cache: `/workspace/.cache/huggingface`
- Config: `/workspace/.mold/config.toml`

All data survives pod restarts — and pod **deletion**, so you can spin up a new
pod on a different GPU without re-downloading 10+ GB of weights.

### Environment Variables

| Variable                | Default | Description                                                       |
| ----------------------- | ------- | ----------------------------------------------------------------- |
| `MOLD_HOME`             | auto    | Base mold directory (auto-detected from `/workspace`)             |
| `MOLD_PORT`             | `7680`  | Server port                                                       |
| `MOLD_LOG`              | `info`  | Log level                                                         |
| `MOLD_DEFAULT_MODEL`    | —       | Default model tag (**not pre-pulled** — fetched on first request) |
| `MOLD_MODELS_DIR`       | —       | Override models path                                              |
| `MOLD_API_KEY`          | —       | API key for authentication (`X-Api-Key` header required)          |
| `MOLD_RATE_LIMIT`       | —       | Per-IP rate limit (e.g., `10/min`)                                |
| `MOLD_RATE_LIMIT_BURST` | —       | Burst allowance override (defaults to 2x rate)                    |
| `HF_TOKEN`              | —       | HuggingFace token for gated model repos                           |
| `MOLD_WEB_DIR`          | `/opt/mold/web` | Path to the bundled web gallery SPA                       |
| `MOLD_GALLERY_ALLOW_DELETE` | —   | `1` to enable `DELETE /api/gallery/image/:name`                   |

### HuggingFace Token (`HF_TOKEN`)

Some models are gated on HuggingFace and require a token:

- **LTX-2 / LTX-2.3** — Gemma 3 text encoder is gated
- **FLUX.1-dev** — non-schnell FLUX weights are gated
- Any model whose manifest points at a gated HF repo

Public models (`flux-schnell`, `flux2-klein`, `sd15`, `sdxl`, `qwen-image`,
`z-image`, `wuerstchen`) work without a token.

Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
Use a **fine-grained token** with `Read access to public gated repos you've been
granted access to` — do not use a full-access token on cloud GPUs.

Three ways to deliver the token into a RunPod pod, in order of preference:

#### 1. Network volume + HF cache (recommended for long-lived setups)

SSH into a pod mounted on a network volume and run:

```bash
huggingface-cli login
```

The token lands at `/workspace/.cache/huggingface/token`. Because
`docker/start.sh` exports `HF_HOME=/workspace/.cache/huggingface`, every future
pod on the same volume picks it up automatically. The token never appears in
`runpodctl pod get` output or pod JSON.

#### 2. RunPod Secrets (web console)

In the console: **Settings → Secrets → Create Secret** named `HF_TOKEN`. Then
reference it from the pod's env vars:

```text
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
```

Works from `runpodctl pod create --env` as well — RunPod expands the
<span v-pre>`{{ RUNPOD_SECRET_* }}`</span> template at container start. The
token is redacted in pod listings.

#### 3. Plain `--env` flag (quick and dirty)

```bash
runpodctl pod create \
  --image ghcr.io/utensils/mold:latest-sm120 \
  --gpu-id "NVIDIA GeForce RTX 5090" \
  --env '{"HF_TOKEN":"hf_abcdef...","MOLD_DEFAULT_MODEL":"ltx-2"}' \
  ...
```

Token appears in shell history, `runpodctl pod get` output, and the console env
panel. Fine for personal throwaway pods, not for shared or audited access.

::: tip SSH sessions auto-inherit the token
`docker/start.sh` copies env vars matching `MOLD_*`, `HF_*`, `CUDA*`, and
`LD_LIBRARY*` into `/etc/rp_environment`, which bash SSH sessions source on
login. Once the token is in the pod environment, `huggingface-cli` and `curl`
inside an SSH session pick it up without re-exporting.
:::

### Recommended GPUs

Rates below are RunPod's published list prices and drift over time — always
confirm with `runpodctl gpu list` or the console before deploying.

| GPU       | VRAM  | Community $/hr | Secure $/hr | Image tag       | Notes                               |
| --------- | ----- | -------------- | ----------- | --------------- | ----------------------------------- |
| RTX 4090  | 24 GB | ~$0.34         | ~$0.69      | `:latest`       | Best value when stock is available  |
| RTX 5090  | 32 GB | ~$0.49         | ~$0.99      | `:latest-sm120` | Usually better stock than 4090      |
| L40S      | 48 GB | ~$0.40         | ~$0.86      | `:latest`       | Full BF16 FLUX without offload      |
| A100 80GB | 80 GB | ~$0.79         | ~$1.64      | `:latest-sm80`  | Maximum headroom, multi-model swaps |

### Alternative: REST API (no CLI required)

`runpodctl` wraps RunPod's REST API at `https://rest.runpod.io/v1/`. If you want
to provision pods from CI, a deploy script, or anywhere the CLI is inconvenient,
hit the API directly:

```bash
export RP=https://rest.runpod.io/v1
export H="Authorization: Bearer $RUNPOD_API_KEY"

# Create pod
curl -sS -H "$H" -H "Content-Type: application/json" $RP/pods -d '{
  "name": "mold-server",
  "imageName": "ghcr.io/utensils/mold:latest-sm120",
  "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
  "cloudType": "SECURE",
  "dataCenterIds": ["EUR-IS-2"],
  "gpuCount": 1,
  "containerDiskInGb": 20,
  "volumeInGb": 50,
  "volumeMountPath": "/workspace",
  "ports": ["7680/http", "22/tcp"],
  "env": { "MOLD_DEFAULT_MODEL": "flux2-klein:q8" }
}' | jq

# Lifecycle
curl -sS -X POST   -H "$H" $RP/pods/<pod-id>/stop
curl -sS -X POST   -H "$H" $RP/pods/<pod-id>/start
curl -sS -X DELETE -H "$H" $RP/pods/<pod-id>
```

Unlike the CLI, the REST API honors every id in `"dataCenterIds"` as a valid
scheduling candidate. Full reference:
[rest.runpod.io/v1/docs](https://rest.runpod.io/v1/docs).

## Monitoring

Docker images include the `metrics` feature. Scrape `GET /metrics` for
Prometheus-format metrics (HTTP request rates, generation duration, queue depth,
GPU memory, uptime). The endpoint is excluded from auth and rate limiting.

## Image Details

The Dockerfile uses a multi-stage build:

1. **Builder** — `nvidia/cuda:12.8.1-devel-ubuntu22.04` with Rust and cargo
2. **Runtime** — `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3.4 GB image, 33 MB
   binary)

`libcuda.so.1` (the NVIDIA driver) is injected at runtime by the NVIDIA
Container Toolkit — the image cannot run without GPU access.
