# LTX-2.5

Mold runs the official LTX-2.5 split checkpoints natively through Candle. The
same request and capability contract serves the CLI, HTTP API, web, desktop,
TUI, and the shared iPhone/Android surface. Metal is the native Apple backend;
CUDA is the native NVIDIA backend.

## Pick a model

The short names default to the smaller distilled INT8 ConvRot pack:

```bash
mold pull ltx-2.5-22b-distilled
mold run ltx-2.5-22b-distilled "a fox crosses a wet neon street" \
  --frames 121 --fps 24 --audio --format mp4
```

`ltx-2.5-22b-distilled` resolves to
`ltx-2.5-22b-distilled:int8-conv`. It downloads about 40.0 GB (37.2 GiB) and is
the default because its transformer and Gemma 4 encoder are substantially
smaller than BF16. The full distilled BF16 packs download about 71.4 GB
(66.5 GiB):

```bash
# BF16 with the conventional video VAE
mold pull ltx-2.5-22b-distilled:bf16-conv

# BF16 with the diffusion video VAE
mold pull ltx-2.5-22b-distilled:bf16
```

Dev variants use the same suffixes and add the official distilled LoRA, about
8.9 GB. Every pack also owns the matching Gemma 4 encoder, audio VAE/vocoder,
duration head, and spatial and temporal upscalers. `/api/models` marks a pack
`runtime_ready: true` only after all of those components pass qualification.
Auto and Most capable routing exclude a host that explicitly reports an
incomplete pack.

All weights are gated and downloaded from the official
[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) repository.
See the [license boundary](architecture/ltx-2.5-license.md) before commercial
use or redistribution.

## Duration and audio

Explicit duration remains the reproducible default. LTX frame counts use the
`8n+1` grid:

```bash
mold run ltx-2.5-22b-distilled "a one-take dance rehearsal" \
  --frames 121 --fps 24 --audio --format mp4
```

The official duration head can instead choose a 1–20 second clip. The CLI flag
and Studio switch omit `frames`; they do not manufacture a guessed frame
count:

```bash
mold run ltx-2.5-22b-distilled "a complete product reveal" \
  --predict-duration --fps 24 --audio --format mp4
```

Surfaces show this option only when the selected host advertises both
`supports_duration_prediction: true` and `runtime_ready: true`. Saved metadata
and Use as prompt preserve whether duration was explicit or predicted.

## Native multishot and Mold Sequence

LTX-2.5 native multishot is prompt/model behavior inside one generated clip.
Describe shots in temporal order, following the official
[prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide).
It is distinct from **Mold Sequence**, which authors separate clip requests,
persists them as a durable chain job, and stitches their outputs. A native
multishot prompt should stay a One shot request unless separate resumable clips
and explicit seams are wanted.

## Supported and deferred paths

Supported now:

- distilled and dev transformers in BF16 and ComfyUI-compatible INT8 ConvRot;
- conventional and diffusion video VAEs;
- Gemma 4 conditioning, explicit or predicted duration, synchronized audio,
  source/keyframe conditioning, guidance overrides, LoRAs, and offload;
- native spatial/temporal two-stage upscaling and ordinary Mold Sequence jobs;
- discovery, pull, readiness, ownership-aware removal, metadata, and reuse on
  every Mold surface.

Deferred and fail-closed:

- NVFP4 execution (the official file is known but not exposed as runnable);
- HDR/EXR, IC-LoRA, Retake, and LipDub adapters until LTX-2.5-specific weights
  and parity are qualified;
- Dynamic Frame Rate and prompt-enhancer product controls;
- redistribution of model weights in Mold releases.

## Reference snapshots

The implementation and parity fixtures are pinned to:

- [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) at
  `400fd31054597515f47125691032c04b1c3ee24e`;
- [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
  at `15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d`;
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) at
  `a1079ba16f2674734b065eb036fbfdddaa321a4d`;
- [Diffusers](https://github.com/huggingface/diffusers) at
  `95c0d467cc2a4770b71fa25a117320377e6eb08f`.

Retained parity artifacts, logs, prompts, seeds, and run manifests live under
`/Volumes/ExternalStorage/mold2/output/verification/ltx-2.5/`. Downloaded model
files remain under `/Volumes/ExternalStorage/mold2` and are not cleanup data.

Run `scripts/capture-ltx25-metal-verification.sh` on Apple Silicon to capture a
machine-readable INT8 Metal report. The capture validates the pinned upstream
clones, download-time SHA-256 markers, decoded retained media, and focused Rust
parity/regression gates. It only reads existing model and media files; reports
and logs are added under the verification directory without deleting renders.

The executable compact-checkpoint A/B row is Mold versus ComfyUI on MPS. The
official PyTorch and Diffusers BF16 pipelines remain static tensor/config
oracles because neither directly loads the Comfy INT8 ConvRot checkpoint.
CUDA qualification is performed on its dedicated host. BF16 remains available
and retained, but a stopped BF16 runtime run is recorded as operator-deferred,
not misreported as passing.
