# Performance

mold performance depends mostly on three things:

1. model family and quantization
2. your GPU memory headroom
3. whether offloading or CPU text encoders are in play

This page gives practical expectations, not a formal benchmark suite. Exact
timings vary by GPU, driver, storage speed, and whether a model is already
loaded.

## Representative Starting Points

Reference hardware: RTX 4090 class GPU, warm model cache, default resolution.

| Model                               | Typical Steps | Ballpark Time | Notes                                                                                    |
| ----------------------------------- | ------------- | ------------- | ---------------------------------------------------------------------------------------- |
| `flux-schnell:q8`                   | 4             | ~8-12s        | Fastest high-quality default                                                             |
| `flux-dev:q4`                       | 25            | ~20-40s       | Better quality, slower denoising                                                         |
| `z-image-turbo:q8`                  | 9             | ~10-20s       | Strong quality/speed trade-off                                                           |
| `sdxl-turbo:fp16`                   | 4             | ~3-8s         | Very fast when you want 1024 output                                                      |
| `sd15:fp16`                         | 25            | ~5-15s        | Lightest full-featured family                                                            |
| `ltx-video-0.9.6-distilled:bf16`    | 8             | ~30-90s       | Recommended current video default                                                        |
| `ltx-video-0.9.8-2b-distilled:bf16` | 7+3           | ~30-90s       | Newer checkpoint family, full multiscale refine                                          |
| `ltx-2-19b-distilled:fp8`           | 8             | ~2-6 min      | Joint audio-video; native Rust FP8 path                                                  |
| `ltx-2.3-22b-distilled:fp8`         | 8             | ~3-8 min      | Larger native joint audio-video path                                                     |
| `wan21-t2v-1.3b:bf16`               | 30            | ~3.5 min      | 480p16; measured 209 s at 33 frames                                                      |
| `wan21-t2v-1.3b:turbo`              | 3             | ~40 s         | 480p16 DMD distill; measured 40.8 s at 81 frames on an L40S (base: 196 s)                |
| `wan21-t2v-14b:q8`                  | 30            | ~15 min       | 480p16; measured 877 s at 33 frames, 20.4 GB                                             |
| `wan22-ti2v-5b:fp16`                | 20            | ~2-4 min      | 246 s T2V 720p24 (4090) / 105 s I2V 480p, 49f; 258.2 s T2V on an L40S with the cache off |
| `wan22-ti2v-5b:turbo`               | 4             | ~80 s         | 720p24 Self-Forcing distill; measured 80.9 s at 121 frames on an L40S                    |
| `wan22-ti2v-5b:dmd`                 | 3             | ~85 s         | 720p24 DMD distill, text-to-video only, shift-5 table; 85.0 s at 121 frames on an L40S   |
| `wan22-i2v-a14b:q5`                 | 4             | ~3.5 min      | Two-expert Lightning tier; 199 s at 53 frames                                            |

## What Slows Things Down

### Video generation

Video generation is significantly slower than image generation. Even distilled
video models still process a 3D latent over frames × height × width, and VAE
decode remains materially slower than image models due to 3D convolutions.

Reducing frame count (`--frames 9`) or step count (`--steps 20`) helps.
Reducing resolution has a large impact since the latent sequence length scales
as frames × height × width.

LTX-2 is slower again than `ltx-video` because it carries the joint
audio-video stack, larger checkpoints, staged native loading, and a larger
conditioning surface. Treat it as a quality-first workflow, not a quick draft
path.

On a 24 GB RTX 4090-class card, the practical local path is the distilled FP8
checkpoint with native layer streaming enabled. mold currently uses the
compatible `fp8-cast` path there rather than Hopper-only
`fp8-scaled-mm`/TensorRT-LLM.

Wan's fast tiers are the A14B 4-step Lightning pairs (`wan22-*-a14b:q5` and
`:q4`), the single-expert `wan22-ti2v-5b:turbo` (4-step Self-Forcing,
guidance 1.0, 121-frame default measured at 160.7 s on an RTX 4090), and two
FastVideo DMD 3-step distills: `wan21-t2v-1.3b:turbo` (the 2.1 1.3B) and
`wan22-ti2v-5b:dmd` (the 2.2 TI2V-5B). The DMD pair is the strictest of the
group: each walks exactly three published rungs (timesteps 1000 / 757 / 522)
on its own flow-match table — shift-8 for the 1.3B, shift-5 for the 5B, since
each distillation trained against a different one; the 5B's shift-5 is the
distill's own table, not the flow shift 8.0 the rest of the family renders at — and on both, steps,
guidance, sample solver, and flow shift are all fixed; a request that sets
any of them is refused rather than silently ignored, because a DMD student
is re-noised between rungs and a UniPC or Euler pass over it is a different
render, not a slower one. At `(30 x 2) / 3` the 1.3B tier is 20x fewer
transformer forwards than its base tier; at `(20 x 2) / 3` the 5B tier is
about 13.3x fewer than its base tier. On A14B, two 14B experts alternate
with one resident at a time, so VRAM is the larger expert, and guidance 1.0
skips the unconditional pass so each of the four steps is one forward. The
A14B GGUF tiers default to their measured RTX 4090
envelopes: 81 frames for `:q5`/`:q4` (partial block offload parks trailing
transformer blocks in host RAM automatically; 81 frames at 832x480 measured
at 316–318 s with a ~15.7–17.3 GiB peak), 73 frames for `:q8`, and 45 for
`:fp8` (fp8 cannot park; the byte round-trip is GGUF-only).
`MOLD_WAN_OFFLOAD_BLOCKS=N` pins the parked-block count (0 disables). The
timing rows above are single-configuration RTX 4090 measurements, not a
support matrix.

CUDA and Apple Metal are both supported backends for local LTX-2 runs. Metal
is performance-qualified on Apple Silicon for the 19B/22B distilled FP8 tiers,
but remains slower than a comparable CUDA card; streamed FP8 widening trades
speed for fitting a 19B–22B model in unified memory. CPU exists for
correctness-oriented native coverage and can be extremely slow.

LTX-2.5 GGUF on Metal keeps only the packed transformer blocks that fit after
preserving a live macOS safety floor. Overflow blocks stay in the GGUF file and
are read one tensor at a time; bounded Metal command-buffer fences release each
streaming window before more temporary weights can accumulate. If memory
pressure changes during loading, Mold demotes resident blocks toward full disk
streaming instead of retaining the original split. This does not change CUDA's
residency or synchronization cadence. If the fixed weights, request
activations, runtime headroom, and one streamed block cannot fit without
crossing the live macOS floor, Mold fails before transformer allocation.

Measured on a 48 GiB Apple M4 Max at 512x512, 9 frames, and 8 steps, whole-
process RSS peaked at 13.14 GiB for Q3_K_M, 13.53 GiB for Q4_K_M, and 13.95 GiB
for Q6_K. The compact INT8 ConvRot route peaked at 19.43 GiB under the same
request. All four completed without the memory guard firing. These are
single-host unified-memory measurements, not Metal-only VRAM figures or a
promise for untested tiers.

### Offloading

`--offload` uses mold-owned block streaming for FLUX, Flux.2, Z-Image,
Qwen-Image, LTX-2, Wan, and SD3 paths where implemented. FLUX, Flux.2, Z-Image,
and Qwen-Image keep the blocks that fit on GPU and stream only the remainder.
LTX-2 and SD3 use full block streaming when offload is forced. Wan parks
trailing transformer blocks in host RAM as raw quantized bytes (GGUF tiers
only); it engages automatically when the render's activation budget exceeds
free VRAM, `--offload`/`MOLD_OFFLOAD=1` parks every block, and
`MOLD_WAN_OFFLOAD_BLOCKS=N` pins the count exactly (0 disables).

Use it when a model otherwise would not fit. Do not use it when the model
already fits comfortably in VRAM. Progress output reports resident blocks,
streamed blocks, resident GB, streamed GB per denoise pass, and reserved
headroom.

### CPU text encoders

mold may place text encoders on CPU when VRAM is tight. That reduces memory
pressure, but prompt encoding takes longer.

You can also force the choice with `--device-text-encoders cpu` on `mold run`
(or the web UI's **Placement** panel, or `MOLD_PLACE_TEXT_ENCODERS=cpu`). This
is often the single biggest VRAM win short of quantization: FLUX's T5 is ~10
GB, SD3.5's triple-encoder stack is larger, and freeing that budget lets the
transformer stay fully resident without triggering block-level offload. Encoding
moves from ≈200 ms to ≈2 s on typical CPU; negligible at 20+ denoising steps,
painful at 4.

For FLUX, Flux.2, Z-Image, and Qwen-Image specifically, you can also pin
individual components: `--device-transformer gpu:1 --device-vae cpu` (two-GPU
split with decode on host memory), `--device-t5 cpu` (FLUX only, keeps CLIP-L
on GPU), etc. See [Configuration → Per-component device placement](./configuration.md#per-component-device-placement)
for the full matrix.

If your GPU has headroom, `--eager` can improve repeat generation speed by
keeping more components resident.

For Qwen-Image, `auto` uses quantized Qwen2.5-VL GGUF text encoders when the
heavier BF16 text stack would be a poor fit. On CUDA local one-shot runs, that
avoids loading the full BF16 encoder on CPU when BF16 does not fit on GPU. On
Apple Metal/MPS, it is mainly a memory-responsiveness improvement, not a
promise of higher throughput.

For `qwen-image-edit`, mold also stages the Qwen2.5-VL vision tower only while
building edit conditioning. Quantized `--qwen2-variant` values reduce the
language-side footprint further, so short edit runs do not keep the full
multimodal stack resident between requests.

### Tier 1 knobs

For edge cases, mold exposes opt-in runtime knobs in
[Configuration → Generation](/guide/configuration#generation):
`MOLD_KEEP_TE_RAM`, `MOLD_LORA_BYPASS`, `MOLD_VAE_TILED`, `MOLD_ATTN`, and
`MOLD_ATTN_CHUNK`. Treat them as targeted controls after you have tried model
quantization, `--device-text-encoders cpu`, and `--offload`.

### Cold starts

The first request for a model pays for:

- model weight loading
- tokenizer setup
- possible prompt expansion model loading

The second request is usually faster unless the model or encoder was dropped to
save memory.

## Practical Tuning

| Goal                    | Use this first                                           |
| ----------------------- | -------------------------------------------------------- |
| Faster iteration        | `flux-schnell:q8`, `sdxl-turbo:fp16`, or `sd15:fp16`     |
| Lower VRAM              | smaller quantization or `--offload`                      |
| Better repeat latency   | keep the same model loaded; try `--eager` if VRAM allows |
| Faster remote workflow  | keep `mold serve` running on the GPU host                |
| Smaller startup penalty | pre-pull models with `mold pull`                         |

## Example Tuning Workflow

```bash
# Start with a fast baseline
mold run flux-schnell:q8 "studio product photo"

# Move up in quality if the baseline is good enough operationally
mold run flux-dev:q6 "studio product photo"

# Only use adaptive offload when necessary
mold run flux-dev:bf16 "studio product photo" --offload
```

## Benchmarking Your Own Setup

The most honest benchmark is your own prompt mix. Use fixed seeds and a warm
model:

```bash
time mold run flux-schnell:q8 "a product photo" --seed 42
time mold run flux-dev:q4 "a product photo" --seed 42
```

For remote setups, also compare local CLI latency against the server’s
`generation_time_ms` from the SSE `complete` event to separate network time from
pure inference time.

## macOS Metal memory

`mold system metal-memory status [--json]` inspects this machine, ignoring
`MOLD_HOST`. Explicit root-only `set <MiB> [--persist]` and `reset [--persist]`
administer its system-wide limit; never run the server as root. Use
`mold gpu list --json` for the inference host's effective capacity and headroom.
Zero means automatic; increases may require restarting an idle inference process.
