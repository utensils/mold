# Qwen-Image

Qwen2.5-VL text encoder with a 3D causal VAE (2D temporal-slice) and
flow-matching with classifier-free guidance.

![Winter cabin — Qwen-Image 2512 Q4](/gallery/qwen-image-cabin.png)
_"A snowy mountain cabin at twilight, warm orange light pouring from the windows, aurora borealis in the sky above"_ — **qwen-image-2512:q4**, 50 steps, seed 888

![Overgrown greenhouse — Qwen-Image 2512 Q4](/gallery/qwen-image-greenhouse.png)
_"An abandoned greenhouse overgrown with exotic flowers and vines, cracked glass roof letting in shafts of golden light, butterflies and hummingbirds, lush and magical"_ — **qwen-image-2512:q4**, 50 steps, seed 2024

![Hot air balloon — Qwen-Image 2512 Q4](/gallery/qwen-image-balloon.png)
_"A colorful hot air balloon floating over a misty valley at sunrise, the balloon has the word MOLD written on the side"_ — **qwen-image-2512:q4**, 50 steps, seed 314

- **Developer**: [Alibaba / Qwen Team](https://huggingface.co/Qwen)
- **License**: Apache 2.0
- **Upstream releases**:
  [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image),
  [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512),
  [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- **GGUF sources**:
  [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf),
  [unsloth/Qwen-Image-2512-GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF),
  [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF),
  [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF)
- **Few-step distill sources**:
  [realrebelai/Qwen_Image_Flash_GGUF](https://huggingface.co/realrebelai/Qwen_Image_Flash_GGUF)
  (quants of [nvidia/Qwen-Image-Flash](https://huggingface.co/nvidia/Qwen-Image-Flash)),
  [QuantStack/Qwen-Image-Distill-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Distill-GGUF)
  (quants of DiffSynth-Studio's Qwen-Image-Distill-Full),
  [Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF)
  (quants of Phr00t's Qwen-Image-Edit-Rapid-AIO)

## Stable GGUF Variants

`mold` supports two quantized Qwen lines:

- `qwen-image:*` uses the base `Qwen/Qwen-Image` release with GGUF transformers from `city96/Qwen-Image-gguf`
- `qwen-image-2512:*` uses `Qwen/Qwen-Image-2512` with GGUF transformers from `unsloth/Qwen-Image-2512-GGUF`

The Qwen-Image text encoder itself is also selectable now:

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

On Apple Metal/MPS, `auto` prefers quantized Qwen2.5-VL GGUF text encoders
(`q6`, then `q4`) to avoid the BF16 text-encoder memory spike. CUDA `auto`
prefers BF16 when enough headroom remains after the transformer load and falls
back to quantized GGUF variants when that resident encoder would be too heavy.

### Base Qwen-Image

| Model           | Steps | Size    | Validated On 24 GB | Notes                                |
| --------------- | ----- | ------- | ------------------ | ------------------------------------ |
| `qwen-image:q8` | 50    | 21.8 GB | `768x768`          | Highest-quality GGUF tier            |
| `qwen-image:q6` | 50    | 16.8 GB | `1024x1024`        | Quality/size trade-off               |
| `qwen-image:q5` | 50    | 14.9 GB | `1024x1024`        | Dynamic `K_M` quant                  |
| `qwen-image:q4` | 50    | 13.1 GB | `1024x1024`        | Stable 24 GB choice                  |
| `qwen-image:q3` | 50    | 9.7 GB  | `1024x1024`        | Lower bitrate, still prompt-faithful |
| `qwen-image:q2` | 50    | 7.1 GB  | `1024x1024`        | Smallest published base GGUF         |

### Qwen-Image-2512

| Model                | Steps | Size    | Validated On 24 GB | Notes                                |
| -------------------- | ----- | ------- | ------------------ | ------------------------------------ |
| `qwen-image-2512:q8` | 50    | 21.8 GB | `768x768`          | Highest-quality 2512 GGUF tier       |
| `qwen-image-2512:q6` | 50    | 16.8 GB | `1024x1024`        | Quality/size trade-off               |
| `qwen-image-2512:q5` | 50    | 15.0 GB | `1024x1024`        | Dynamic `K_M` quant                  |
| `qwen-image-2512:q4` | 50    | 13.2 GB | `1024x1024`        | Stable 24 GB choice                  |
| `qwen-image-2512:q3` | 50    | 9.9 GB  | `1024x1024`        | Lower bitrate, still prompt-faithful |
| `qwen-image-2512:q2` | 50    | 7.3 GB  | `1024x1024`        | Smallest published 2512 GGUF         |

### Qwen-Image-Edit-2511

`qwen-image-edit-2511:*` is the edit-family sibling of Qwen-Image. It uses
repeatable `--image` inputs instead of img2img `--strength`, supports negative
prompts, and targets output dimensions derived from the first input image at
roughly `1024x1024` area.

| Model                       | Steps | Size    | Notes                                 |
| --------------------------- | ----- | ------- | ------------------------------------- |
| `qwen-image-edit-2511:q8`   | 50    | 21.8 GB | Highest-quality GGUF tier             |
| `qwen-image-edit-2511:q6`   | 50    | 16.9 GB | Quality/size trade-off                |
| `qwen-image-edit-2511:q5`   | 50    | 15.0 GB | Dynamic `K_M` quant                   |
| `qwen-image-edit-2511:q4`   | 50    | 13.2 GB | Stable 24 GB GGUF target              |
| `qwen-image-edit-2511:q3`   | 50    | 9.9 GB  | Lower bitrate, still relatively small |
| `qwen-image-edit-2511:q2`   | 50    | 7.5 GB  | Smallest published edit GGUF          |
| `qwen-image-edit-2511:bf16` | 50    | 40.9 GB | Sharded BF16 edit transformer         |

### Few-step distilled variants

These are step-distilled merges of the base transformers, so they reuse the
same shared VAE / Qwen2.5-VL components and run CFG-free at guidance `1.0`.

`qwen-image-flash:*` also runs its own packaged scheduler — NVIDIA ships
`use_dynamic_shifting=false`, `shift=3.0`, `shift_terminal=null` for the
four-step trajectory — rather than base Qwen-Image's resolution-dependent
schedule. The Distill-Full and Rapid AIO merges are transformer-only exports
with no scheduler of their own, so they keep the base contract.

| Model                      | Steps | Guidance | Size    | Notes                                                     |
| -------------------------- | ----- | -------- | ------- | --------------------------------------------------------- |
| `qwen-image-flash:q8`      | 4     | 1.0      | 21.8 GB | NVIDIA DMD2 4-step distill of base Qwen-Image             |
| `qwen-image-flash:q4`      | 4     | 1.0      | 11.7 GB | Same distill, the 24 GB-friendly tier                     |
| `qwen-image-distill:q8`    | 15    | 1.0      | 21.8 GB | DiffSynth Distill-Full merge, closer to base fidelity     |
| `qwen-image-distill:q4`    | 15    | 1.0      | 13.1 GB | Same merge, the 24 GB-friendly tier                       |
| `qwen-image-edit-rapid:q4` | 8     | 1.0      | 13.3 GB | 8-step edit merge; `qwen-image-edit` family, `--image` in |

::: danger 18+ NSFW
`qwen-image-edit-rapid:q4` tracks the `v23` release of Phr00t's Rapid AIO merge
(`v23/Qwen-Rapid-NSFW-v23_Q4_K.gguf`), which upstream classifies
`not-for-all-audiences`. It is an uncensored community merge, and mold flags it
as mature: `/api/models[].nsfw` is `true` for it, so every Models surface shows
the `18+ NSFW` badge.
:::

::: warning Distillation trade-off
`qwen-image-flash:*` collapses the schedule to four steps. Dense small text,
hair-fine detail, and very complex scenes may degrade against base Qwen-Image.
Use `qwen-image-distill:*` (15 steps) when you want most of the speed-up with
more of the base model's fidelity, and base `qwen-image:*` at 50 steps when the
prompt depends on fine text or dense structure.
:::

```bash
mold run qwen-image-flash:q4 "your prompt here"
mold run qwen-image-distill:q4 "your prompt here"
mold run qwen-image-edit-rapid:q4 "make the sky stormy" --image input.png
```

::: tip Edit Path
`qwen-image-edit-2511` runs a real multimodal edit path: Qwen2.5-VL condition
images are patchified through the vision tower, source-image latents are packed
and concatenated with output-noise tokens, and true CFG uses norm rescaling.
Quantized `--qwen2-variant` values are supported for the edit family through a
GGUF Qwen2.5 language path plus the staged Qwen2.5-VL vision tower used for
image conditioning. On CUDA, quantized edit transformers run true CFG as two
passes rather than doubling the packed output-and-conditioning sequence; this
is the stable path at non-square resolutions and is used even when extra VRAM
is available.
:::

::: tip Recommended Stable Quant Paths
On a 24 GB card, `qwen-image:q4` and `qwen-image-2512:q4` are the safest
starting points for native-quality GGUF inference. `q6` and `q5` also work
well at `1024x1024`, while `q8` is currently validated at `768x768`.

```bash
mold pull qwen-image:q4
mold run qwen-image:q4 "your prompt here"

mold pull qwen-image-2512:q4
mold run qwen-image-2512:q4 "your prompt here"
```

:::

::: tip Apple Silicon
On Apple Silicon, leave `--qwen2-variant` unset first. Metal `auto` will prefer
the quantized Qwen2.5-VL text encoder path for Qwen-Image automatically.

```bash
mold run qwen-image:q2 "your prompt here" --preview
```

To compare explicitly:

```bash
mold run qwen-image:q2 "your prompt here" --qwen2-variant q6
mold run qwen-image:q2 "your prompt here" --qwen2-variant q4
```

:::

::: tip CUDA local runs
On CUDA, `auto` keeps BF16 for the Qwen2.5-VL text encoder when there is enough
headroom. If not, local one-shot runs use the quantized Q4 GGUF encoder instead
of loading the full BF16 text stack on CPU.

Use `--qwen2-variant bf16` only when you deliberately want the BF16 comparison.
:::

## Defaults

- **Resolution**: 1328x1328
- **Guidance**: 4.0
- **Steps**: 50

On the 24 GB validation machine used for mold development:

- `q2` through `q6` were validated at `1024x1024`
- `q8` was validated at `768x768`
- `qwen-image-2512:q4` still ran out of memory at `1328x1328`

## Negative Prompts

Qwen-Image supports negative prompts via `--negative-prompt`.

For the GGUF quantized paths above, the best prompt adherence came from using
no default negative prompt at all. Start without one and only add a negative
prompt if you need to push the image away from a specific failure mode.

The upstream Chinese negative prompt is more appropriate for BF16 / FP8 paths:

```bash
mold run qwen-image:fp8 "a cat" --negative-prompt "低分辨率，低画质，肢体畸形，手指畸形"
```

::: warning
The upstream Chinese negative prompt can hurt GGUF prompt adherence. Avoid
using it by default with `qwen-image:q2` through `qwen-image:q8` or
`qwen-image-2512:q2` through `qwen-image-2512:q8`.
:::

## Other Qwen Variants

`mold` also exposes higher-VRAM Qwen paths such as `qwen-image:bf16`,
`qwen-image:fp8`, `qwen-image-lightning:fp8` (4 steps, 20.4 GB), and
`qwen-image-lightning:fp8-8step` (8 steps, 20.5 GB). Those are separate from the
GGUF quantized matrix above and have different memory and scheduler behavior.

The few-step GGUF distills — `qwen-image-flash:{q8,q4}`,
`qwen-image-distill:{q8,q4}`, and `qwen-image-edit-rapid:q4` — are listed with
their steps and sizes under
[Few-step distilled variants](#few-step-distilled-variants) above.

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1328  | 1328   | 1:1 (native) |
| 1024  | 1024   | 1:1          |
| 1152  | 896    | 9:7          |
| 896   | 1152   | 7:9          |
| 1216  | 832    | 19:13        |
| 832   | 1216   | 13:19        |
| 1344  | 768    | 7:4          |
| 768   | 1344   | 4:7          |
| 1664  | 928    | ~16:9        |
| 928   | 1664   | ~9:16        |
| 768   | 768    | 1:1 (small)  |
| 512   | 512    | 1:1 (small)  |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.
