# Getting Started

mold began as a single-binary command-line tool for generating AI images and
video on your own GPU, and it remains **CLI-native**. No cloud API, Python
environment, or heavyweight runtime is required.

The command line is the stable product foundation: core workflows accept
predictable arguments, stdin, files, and environment variables; return useful
stdout, stderr, exit status, and machine-readable output; and work the same for
a person, shell script, CI job, or agent. Desktop, web, TUI, iPhone, REST/SSE,
and MCP clients extend that engine rather than hiding it.

## How It Works

```
mold run "a cat"
  │
  ├─ Server running? → send request over HTTP
  │
  └─ No server? → load model locally on GPU
       ├─ Encode prompt (T5/CLIP text encoders)
       ├─ Denoise latent (transformer/UNet)
       ├─ Decode pixels (VAE)
       └─ Save PNG
```

mold tries to connect to a running `mold serve` instance first. If no server is
available, it falls back to local GPU inference — auto-downloading the model if
needed.

![Tea house — generated with FLUX Dev Q4](/gallery/flux-dev-teahouse.png)

## Built for automation

Mold composes with ordinary Unix tools:

```bash
# Prompt on stdin, image bytes on stdout
echo "a cinematic city at night" | mold run flux-schnell | viu -

# Reproducible batch work from a script or CI job
mold run flux-dev:q4 "product photo" --seed 42 --output artifact.png

# Point the same CLI or an agent at a remote GPU
MOLD_HOST=http://gpu-server:7680 mold run "a storyboard frame"
```

For tool-calling agents, use `mold mcp`, the
[OpenClaw skill](/guide/openclaw), or the documented [REST/SSE API](/api/).

## Quick Start

```bash
# Install (auto-detects your GPU)
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh

# Generate your first image
mold run "a sunset over mountains"

# That's it — the model downloads on first run (~12GB for flux-schnell:q8)
```

For the latest rolling CLI build from `main`, add `MOLD_CHANNEL=nightly` on
the `sh` side of the install pipe.

## What You Get

- **Broad model support** — FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein
  and Dev, Qwen-Image, Qwen-Image-Edit, Wuerstchen v2, LTX Video,
  LTX-2 / LTX-2.3, Wan 2.1/2.2, and MiniMax H3 (compact FL2VA; generation on
  supported SM89 CUDA builds)
- **txt2img, img2img, multimodal edit, inpainting, ControlNet** — all in one binary
- **Image upscaling** — Real-ESRGAN super-resolution (2x/4x) via CLI, server API, or TUI
- **Pipe-friendly** — `mold run "a cat" | viu -` just works
- **Client-server** — run the GPU part on one machine, generate from anywhere
- **Native apps** — a local/multi-host desktop studio and a remote-only iPhone
  companion
- **Prompt expansion** — short prompts become detailed via local LLM
- **LoRA adapters** — apply fine-tuned styles across FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image, Qwen-Image-Edit, Wan, and Z-Image
- **PNG metadata** — generation parameters embedded for reproducibility

## Requirements

- **NVIDIA GPU** with CUDA or **Apple Silicon** with Metal
- Models auto-download on first use (~2–30 GB depending on model)

## Next Steps

- [Installation](/guide/installation) — all the ways to install mold
- [Configuration](/guide/configuration) — environment variables, config file
- [Generating Images](/guide/generating) — full usage guide
- [Desktop App](/guide/desktop) — local and multi-host native studio
- [iPhone App](/guide/iphone) — remote Create, Library, Models, Machines, and
  Settings
- [Machines](/guide/machines) — connect, discover, and monitor web hosts
- [Feature Support](/guide/feature-matrix) — which model families support which
  features
- [Remote Workflows](/guide/remote-workflows) — laptop-to-GPU-server setups
- [Performance](/guide/performance) — speed and VRAM tuning
- [Custom Models & LoRA](/guide/custom-models) — manual config and adapter
  workflows
- [Models](/models/) — which model to pick for your use case
