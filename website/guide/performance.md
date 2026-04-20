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

| Model                               | Typical Steps | Ballpark Time | Notes                                           |
| ----------------------------------- | ------------- | ------------- | ----------------------------------------------- |
| `flux-schnell:q8`                   | 4             | ~8-12s        | Fastest high-quality default                    |
| `flux-dev:q4`                       | 25            | ~20-40s       | Better quality, slower denoising                |
| `z-image-turbo:q8`                  | 9             | ~10-20s       | Strong quality/speed trade-off                  |
| `sdxl-turbo:fp16`                   | 4             | ~3-8s         | Very fast when you want 1024 output             |
| `sd15:fp16`                         | 25            | ~5-15s        | Lightest full-featured family                   |
| `ltx-video-0.9.6-distilled:bf16`    | 8             | ~30-90s       | Recommended current video default               |
| `ltx-video-0.9.8-2b-distilled:bf16` | 7+3           | ~30-90s       | Newer checkpoint family, full multiscale refine |
| `ltx-2-19b-distilled:fp8`           | 8             | ~2-6 min      | Joint audio-video; native Rust FP8 path         |
| `ltx-2.3-22b-distilled:fp8`         | 8             | ~3-8 min      | Larger native joint audio-video path            |

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

CUDA is the supported backend for real local LTX-2 runs. CPU exists for
correctness-oriented native coverage and can be extremely slow. Metal is
explicitly unsupported for this family.

### Offloading

`--offload` can drop FLUX-class VRAM usage from roughly 24 GB to roughly 2-4 GB,
but it is usually 3-5x slower.

Use it when a model otherwise would not fit. Do not use it when the model
already fits comfortably in VRAM.

### CPU text encoders

mold may place text encoders on CPU when VRAM is tight. That reduces memory
pressure, but prompt encoding takes longer.

You can also force the choice with `--device-text-encoders cpu` on `mold run`
(or the web UI's **Placement** panel, or `MOLD_PLACE_TEXT_ENCODERS=cpu`). This
is often the single biggest VRAM win short of quantization: FLUX's T5 is ~10
GB, SD3.5's triple-encoder stack is larger, and freeing that budget lets the
transformer stay fully resident without triggering block-level offload (which
is 3–5× slower per step). Encoding moves from ≈200 ms to ≈2 s on typical CPU
— negligible at 20+ denoising steps, painful at 4.

For FLUX, Flux.2, Z-Image, and Qwen-Image specifically, you can also pin
individual components: `--device-transformer gpu:1 --device-vae cpu` (two-GPU
split with decode on host memory), `--device-t5 cpu` (FLUX only, keeps CLIP-L
on GPU), etc. See [Configuration → Per-component device placement](./configuration.md#per-component-device-placement)
for the full matrix.

If your GPU has headroom, `--eager` can improve repeat generation speed by
keeping more components resident.

For Qwen-Image on Apple Metal/MPS, `auto` now prefers quantized Qwen2.5-VL
GGUF text encoders before falling back to the heavier BF16 text stack. That is
mainly a memory-responsiveness improvement, not a promise of higher throughput.

For `qwen-image-edit`, mold also stages the Qwen2.5-VL vision tower only while
building edit conditioning. Quantized `--qwen2-variant` values reduce the
language-side footprint further, so short edit runs do not keep the full
multimodal stack resident between requests.

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

# Only use offload when necessary
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
