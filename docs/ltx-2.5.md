# LTX-2.5

Mold runs the official LTX-2.5 split checkpoints natively through Candle. The
same request and capability contract serves the CLI, HTTP API, web, desktop,
TUI, and the shared iPhone/Android surface. Apple Metal qualification covers
the compact distilled INT8 ConvRot pack plus the Q3_K_M, Q4_K_M, and Q6 GGUF
tiers. CUDA has a separate completed qualification campaign on NVIDIA hosts.

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

On Metal, adaptive residency reserves `max(15% of installed RAM, 8 GiB)` for
macOS and other applications, retains only the packed blocks inside the live
remainder, and streams overflow tensors from disk through bounded release
fences. If the fixed transformer weights, request activations, runtime
headroom, and one streamed block exceed that live remainder, Mold refuses
before transformer allocation. A failed Metal release fence after an
allocation OOM also stops the retry ladder. Neither rule changes CUDA.

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

Surfaces show this option only when the selected host advertises
`supports_duration_prediction: true` and does not explicitly report
`runtime_ready: false`; `runtime_ready` is optional, so a host that omits it
still offers the switch. Saved metadata and Use as prompt preserve whether
duration was explicit or predicted.

### Video-only rendering

`--video-only` (`GenerateRequest.video_only: true`, the Advanced "Video only"
toggle on web and desktop) skips the audio-video transformer's audio branch
entirely, the way upstream's video-only configurator omits it
([#1037](https://github.com/utensils/mold/issues/1037)). It is
output-changing, so it is never inferred or defaulted, and is refused
alongside `enable_audio: true`, conditioning audio, the text-to-audio
pipeline, and `extend_video`. Mold Sequence's chain wire does not carry
`video_only` yet — a sequence clip always renders with its ordinary audio
behavior.

## Native multishot and Mold Sequence

LTX-2.5 native multishot is prompt/model behavior inside one generated clip.
Describe shots in temporal order, following the official
[prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide).
It is distinct from **Mold Sequence**, which authors separate clip requests,
persists them as a durable chain job, and stitches their outputs. A native
multishot prompt should stay a One shot request unless separate resumable clips
and explicit seams are wanted.

## Implemented and qualified paths

Executed on Apple Metal with retained evidence:

- the distilled ComfyUI-compatible INT8 ConvRot transformer with the
  conventional video VAE;
- Gemma 4 conditioning with explicit frames, including synchronized-audio MP4
  and silent T2V APNG outputs;
- Q3_K_M (`:q3`), Q4_K_M (`:q4`), and Q6_K (`:q6`) GGUF transformers at
  512x512, 9 frames, and 8 steps. Fixed-seed visual inspection confirmed the
  requested yellow tram, blue neon reflections, and red umbrellas in all three
  tiers after the Gemma 4 RMSNorm fix;
- Q3_K_M at the full 97-frame single-clip envelope, exported as MP4.

On the 48 GiB Apple M4 Max qualification host, process RSS peaked at 13.14 GiB
for Q3_K_M, 13.53 GiB for Q4_K_M, and 13.95 GiB for Q6_K during the matched
512x512 runs. The compact INT8 ConvRot comparison peaked at 19.43 GiB. These
are whole-process unified-memory measurements, not Metal-only VRAM figures.
The memory guard did not fire, and every output was retained in the Mold
Library.

The sealed machine-readable Metal report covers the INT8 audio and silent rows
from the qualification rendering commit. GGUF qualification retains its clips,
logs, TSV measurements, and inspected contact sheets beside that report; it is
not represented as a GGUF row inside the INT8 report.

Implemented and covered by focused planning, parsing, or unit contracts, but
not claimed as executed Metal qualification by this report:

- the `:q3-k-s`, `:q4-k-s`, `:q5`, and `:q8` GGUF tiers;
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
`/Volumes/ExternalStorage/mold2/output/verification/ltx-2.5/`. The retained
Metal report for the qualification rendering commit is
`ltx25-metal-int8-verification-20260831T152436Z.json`; it seals source commit
`8ac394ecdab96953a209bbe8f51e90d9e5ceaaf6`, the three focused Rust gates, and
the retained INT8 audio and silent media. Downloaded model files remain under
`/Volumes/ExternalStorage/mold2` and are not cleanup data.

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

The completed CUDA campaign has the same two halves on its dedicated NVIDIA
hosts: `scripts/capture-ltx25-cuda-verification.sh` runs (`--run`) and then
seals (`--seal`, the default) every
`scripts/fixtures/ltx25-cuda-matrix.json` row into a
`mold.ltx25.cuda.verification.v1` report, checked by
`scripts/validate-ltx25-cuda-report.py` against
`docs/qualification/ltx25-cuda-verification.schema.json`, which re-hashes every
retained log, manifest, and media file.
`scripts/capture-ltx25-comfy-cuda-reference.sh` retains the exact-weight
ComfyUI CUDA oracle for those rows, and
`scripts/provision-ltx25-comfy-oracle.sh` provisions its pinned clone and venv
under the gitignored `tmp/`. Like the Metal capture, these are UAT tooling: they
ship nothing and delete no models or renders.

Both CUDA capture scripts require an explicit existing `MOLD_HOME` and
`MOLD_MODELS_DIR`; the model store may live under `MOLD_HOME/models`. Set
`LTX25_GPU_INDEX` to the physical GPU to observe (default 0), and start the
scratch mold server with `CUDA_VISIBLE_DEVICES` set to that GPU's full UUID.
The runner verifies this binding; the ComfyUI capture sets it for its own
process. Keep the production service separate from this scratch server.
The ComfyUI preflight requires 50% **or** 64 GiB available host memory and
aborts below both 20% and 16 GiB, while retaining its 48 GiB process RSS and
one-hour runtime limits. Test gates clear `MOLD_*` settings and isolate the
configuration directory. Reports accept filesystem-identical bind-mount
aliases and still verify every retained checksum.

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


### CUDA follow-up oracle (2026-09-05)

The pinned ComfyUI GGUF Q4 reference completed its 8-step first pass and
3-step spatial refinement on one L40S. Its retained manifest is
`/storage/mold/uat-cuda-reliability/oracle-home/output/verification/ltx-2.5/cuda/comfyui/reference-20260905T234810Z/manifest.json`.
The inspected frame contains brass automata on a wet reflective surface,
consistent with the subject and setting of the fixture prompt. This establishes
an executable reference, not pixel parity with Mold. ComfyUI stayed pinned at
`a1079ba16f2674734b065eb036fbfdddaa321a4d`, with ComfyUI-GGUF at
`6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.

For this NixOS scratch environment, Triton needed
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` and `C_INCLUDE_PATH` pointing to
the venv interpreter's real Python include directory. The torch wheel's
`site-packages/nvidia/*/lib` directories preceded the devshell libraries in
`LD_LIBRARY_PATH`, so cuDNN loaded its matching sublibraries. Earlier failed
attempts remain beside the successful capture.


The CUDA matrix's `default` profile is the historical **math qualification
baseline**, pinned with `MOLD_ATTN=math`; it is not the shipping video default,
which can select FlashAttention when compiled. The F32, INT8-dequant and QMatMul
profiles also pin math, and the Flash profile pins flash. Captures verify these
values against the running server, including absence of other profile knobs.
The startup `attention backend policy resolved requested=Some(...)` event
records that input; each render must still supply its actual `ltx2 attention
path=...` event and matching gallery metadata. A policy event alone is never
proof of the executed path.
