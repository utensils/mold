# LTX-2.5

Mold runs the official LTX-2.5 split checkpoints natively through Candle. The
same request and capability contract serves the CLI, HTTP API, web, desktop,
TUI, and the shared iPhone/Android surface. This qualification covers the
compact distilled INT8 ConvRot pack on Apple Metal. CUDA qualification is
completed separately on an NVIDIA host.

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

Seven transformer-only GGUF tiers from
[`Abiray/LTX-2.5-Distilled-GGUF`](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)
are also pinned and runnable: `:q3-k-s`, `:q3` (Q3_K_M), `:q4-k-s`, `:q4`
(Q4_K_M), `:q5`, `:q6`, and `:q8`. They reuse the official packed INT8
Gemma 4, Conv VAE, audio VAE/vocoder, duration head, and both latent
upscalers. The quantized weights stay compact at rest — block linears load as
ggml `QTensor`s and dequantize per forward on CUDA by default
(`MOLD_LTX2_QMATMUL=1` opts into candle's quantized fast path; Metal keeps
`QMatMul`) — so adaptive residency prices the tiers at their file sizes:
Q4_K_M's ~15.7 GB transformer sits fully resident on a 24 GB card. LoRAs
apply as a parallel low-rank branch, never merged into the quantized weight;
full-weight `.diff` deltas are refused with a pointer at the safetensors
packs.

Official companion weights are gated and downloaded from
[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5). The public
GGUF mirror does not remove the underlying LTX-2.x license obligations.
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

## Implemented and qualified paths

Executed on Apple Metal with retained media and machine-readable reports:

- the distilled ComfyUI-compatible INT8 ConvRot transformer with the
  conventional video VAE;
- Gemma 4 conditioning with explicit frames, including synchronized-audio MP4
  and silent T2V APNG outputs.

Implemented and covered by focused planning, parsing, or unit contracts, but
not claimed as executed Metal qualification by this report:

- predicted duration, source/keyframe conditioning, guidance overrides, LoRAs,
  and offload;
- native spatial/temporal two-stage upscaling and ordinary Mold Sequence jobs;
- discovery, pull, readiness, ownership-aware removal, metadata, and reuse
  across Mold surfaces.

Deferred and fail-closed:

- BF16 execution on Metal; its approximately 71.4 GB split packs remain
  downloadable and checksum-qualified;
- the diffusion-video-VAE packs, which are BF16 and therefore part of the
  deferred Metal runtime row;
- CUDA runtime qualification, which belongs to the dedicated NVIDIA campaign;
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
parity/regression gates. First run
`scripts/capture-ltx25-comfy-metal-reference.sh` to retain the exact-weight
ComfyUI MPS graph, history, log, and manifest used by the report. A completed
reference also retains its decoded clip; a run that exceeds the guarded budget
is recorded as operator-deferred with the blocking operator and sampler
progress instead of being reported as passing. Neither capture deletes models
or renders.

The compact-checkpoint reference row is Mold versus ComfyUI on MPS. On the
qualification host, ComfyUI loaded the exact INT8 ConvRot weights and reached
the official sampler, but PyTorch sent `aten::_int_mm` to CPU and the run
remained at `0/8` when the 60-minute guard stopped it. The retained row is
therefore `operator_deferred`, not a completed visual A/B. The official
PyTorch and Diffusers BF16 pipelines remain static tensor/config oracles
because neither directly loads the Comfy INT8 ConvRot checkpoint. CUDA
qualification is performed on its dedicated host. BF16 remains available and
retained, but a stopped BF16 runtime run is recorded as operator-deferred, not
misreported as passing.
