# Troubleshooting

Common issues when running mold locally or against a remote GPU host.

## Out of Memory / VRAM Errors

If generation fails with an out-of-memory message:

- Add `--offload` to use adaptive transformer block offload.
- Use a smaller quantization such as `:q6`, `:q4`, or a lighter family like
  `flux2-klein`.
- Lower `--width` and `--height`.
- Avoid `--eager` unless you know your card has enough headroom.

Examples:

```bash
mold run flux-dev:q4 "a portrait" --offload
mold run z-image-turbo:q4 "a city at dusk" --width 768 --height 768
```

## Which Model Fits My GPU?

| GPU VRAM | Good Starting Choices                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 4-6 GB   | `flux2-klein:q4`, `sd15:fp16`                                                                              |
| 8-10 GB  | `flux-dev:q4`, `flux-schnell:q4`, `z-image-turbo:q4`, `sdxl-turbo:fp16`                                    |
| 12-16 GB | `flux-schnell:q8`, `flux-dev:q6`, `z-image-turbo:q8`, `qwen-image:q4`, `qwen-image-2512:q4`                |
| 24 GB    | `qwen-image:q4`, `qwen-image-2512:q4`, `qwen-image-edit-2511:q4`, `flux-dev:bf16`, most quantized variants |
| 48 GB+   | Full BF16 variants with more room for eager loading                                                        |

As a rule, quantized FLUX and Z-Image variants are the easiest place to start.
For the Qwen family on a 24 GB card, start with `qwen-image:q4`,
`qwen-image-2512:q4`, or `qwen-image-edit-2511:q4`. On the current mold
validation machine, Qwen GGUF variants `q2` through `q6` were validated at
`1024x1024`, while `q8` was validated at `768x768`.

If Qwen prompt conditioning or edit setup makes the machine unresponsive, keep
the model the same and try a quantized Qwen2 path explicitly:

```bash
mold run qwen-image:q2 "your prompt" --qwen2-variant q6
mold run qwen-image:q2 "your prompt" --qwen2-variant q4
mold run qwen-image-edit-2511:q4 "make the chair red leather" --image chair.png --qwen2-variant q4
```

`auto` already prefers the lighter path when BF16 would be too heavy. Only
force `--qwen2-variant bf16` if you are deliberately comparing the larger
resident encoder behavior.

## Connection Refused

If `mold run` cannot reach the server:

- Run `mold ps` to check server status or detect local mold processes.
- Start the server with `mold serve`.
- Verify `MOLD_HOST` points at the right machine and port.

```bash
mold ps
MOLD_HOST=http://gpu-host:7680 mold run "a cat"
```

If no server is reachable, `mold run` may fall back to local inference when the
binary includes GPU support.

## Slow Generation

Slow generation is often expected when mold is preserving VRAM:

- `--offload` can reduce VRAM dramatically. FLUX, Flux.2, Z-Image, and
  Qwen-Image keep fitting blocks GPU-resident and stream only the remainder;
  LTX-2 and SD3 full-stream transformer blocks when offload is forced.
- Text encoders may be placed on CPU automatically when VRAM is tight.
- `--eager` can improve throughput if your GPU has enough free memory.

If you want maximum speed, use a smaller model that fits fully on the card
without offloading.

## Temporarily Unschedulable Models

On multi-GPU servers, a model that OOMs on more than one worker can be marked
temporarily unschedulable. Generation returns an error naming that state instead
of repeatedly cycling every queued job through the same failing GPUs.

Check:

```bash
mold ps
curl http://localhost:7680/api/status
```

Then wait for the cooldown, lower the request size, choose a smaller
quantization, or force a lower-memory path such as adaptive `--offload` or
`--device-text-encoders cpu`.

## Worker Degraded State

If `/api/status` shows a GPU worker with `"state": "degraded"`, that worker hit
several consecutive failures and is cooling down briefly. New jobs route to
healthy workers when possible. Server logs include the original error; inspect
them before changing models or deleting files.

## Model Download Problems

For gated Hugging Face repos, set `HF_TOKEN` before running `mold pull`:

```bash
export HF_TOKEN=hf_...
mold pull flux-dev:q4
```

If an interrupted download leaves the model marked as incomplete:

- retry `mold pull <model>`
- or remove the partial download with `mold rm <model>`

`mold pull` uses a `.pulling` marker to track incomplete downloads, so
`mold list` can show the state accurately.

## Wrong GPU Architecture or Device

If a Docker or Nix build targets the wrong NVIDIA architecture, rebuild with the
correct `CUDA_COMPUTE_CAP` or choose the matching package variant.

Examples:

```bash
docker build --build-arg CUDA_COMPUTE_CAP=120 -t mold-server-b200 .
nix build .#mold-sm120
```

For local debugging, `MOLD_DEVICE=cpu` forces CPU execution. That is mostly
useful for diagnosis, not for real image generation performance.

## Advanced Performance Knobs

The main opt-in knobs are documented in
[Configuration → Performance knobs](/guide/configuration#performance-knobs).
Start there for `MOLD_KEEP_TE_RAM`, `MOLD_LORA_BYPASS`, `MOLD_VAE_TILED`,
`MOLD_ATTN`, and `MOLD_ATTN_CHUNK` instead of guessing from log messages.
