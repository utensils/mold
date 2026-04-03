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

### 1. Push Your Image

```bash
docker tag mold-server your-registry/mold-server
docker push your-registry/mold-server
```

Or use the pre-built GHCR images directly.

### 2. Create a Pod Template

- **Container image**: `ghcr.io/utensils/mold:latest`
- **HTTP port**: `7680`
- Attach a **network volume** for persistent model storage

### 3. Generate from Anywhere

```bash
MOLD_HOST=https://<pod-id>-7680.proxy.runpod.net mold run "a cat"
```

### Network Volume

The entrypoint auto-detects RunPod network volumes at `/workspace`:

- Models persist at `/workspace/.mold/models`
- HuggingFace cache at `/workspace/.cache/huggingface`
- All data survives pod restarts

### Environment Variables

| Variable                | Default | Description                                              |
| ----------------------- | ------- | -------------------------------------------------------- |
| `MOLD_HOME`             | auto    | Base mold directory (auto-detected from `/workspace`)    |
| `MOLD_PORT`             | `7680`  | Server port                                              |
| `MOLD_LOG`              | `info`  | Log level                                                |
| `MOLD_DEFAULT_MODEL`    | —       | Default model to load                                    |
| `MOLD_MODELS_DIR`       | —       | Override models path                                     |
| `MOLD_API_KEY`          | —       | API key for authentication (`X-Api-Key` header required) |
| `MOLD_RATE_LIMIT`       | —       | Per-IP rate limit (e.g., `10/min`)                       |
| `MOLD_RATE_LIMIT_BURST` | —       | Burst allowance override (defaults to 2x rate)           |
| `HF_TOKEN`              | —       | HuggingFace token for gated model repos                  |

### Recommended GPUs

| GPU       | VRAM  | $/hr  | Notes                          |
| --------- | ----- | ----- | ------------------------------ |
| RTX 4090  | 24 GB | $0.34 | Best value, all models work    |
| L40S      | 48 GB | $0.40 | Full BF16 FLUX without offload |
| A100 80GB | 80 GB | $0.79 | Maximum headroom               |

::: tip Proxy Timeout
RunPod's Cloudflare proxy has a 100-second timeout. Use the SSE streaming endpoint (`/api/generate/stream`) for long generations.
:::

## Image Details

The Dockerfile uses a multi-stage build:

1. **Builder** — `nvidia/cuda:12.8.1-devel-ubuntu22.04` with Rust and cargo
2. **Runtime** — `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3.4 GB image, 33 MB
   binary)

`libcuda.so.1` (the NVIDIA driver) is injected at runtime by the NVIDIA
Container Toolkit — the image cannot run without GPU access.
