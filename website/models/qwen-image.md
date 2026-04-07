# Qwen-Image

Qwen2.5-VL text encoder with a 3D causal VAE (2D temporal-slice) and
flow-matching with classifier-free guidance.

![Winter cabin — Qwen-Image 2512 Q4](/gallery/qwen-image-cabin.png)
_"A snowy mountain cabin at twilight, warm orange light pouring from the windows, aurora borealis in the sky above"_ — **qwen-image-2512:q4**, 50 steps, seed 888

![Hot air balloon — Qwen-Image 2512 Q4](/gallery/qwen-image-balloon.png)
_"A colorful hot air balloon floating over a misty valley at sunrise, the balloon has the word MOLD written on the side"_ — **qwen-image-2512:q4**, 50 steps, seed 314

- **Developer**: [Alibaba / Qwen Team](https://huggingface.co/Qwen)
- **License**: Apache 2.0
- **Upstream releases**:
  [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image),
  [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
- **GGUF sources**:
  [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf),
  [unsloth/Qwen-Image-2512-GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF),
  [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF)

## Stable GGUF Variants

`mold` supports two quantized Qwen lines:

- `qwen-image:*` uses the base `Qwen/Qwen-Image` release with GGUF transformers from `city96/Qwen-Image-gguf`
- `qwen-image-2512:*` uses `Qwen/Qwen-Image-2512` with GGUF transformers from `unsloth/Qwen-Image-2512-GGUF`

The Qwen-Image text encoder itself is also selectable now:

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

On Apple Metal/MPS, `auto` prefers quantized Qwen2.5-VL GGUF text encoders
(`q6`, then `q4`) to avoid the BF16 text-encoder memory spike. CUDA `auto`
stays on the existing BF16 path unless you explicitly select a Qwen2 variant.

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
`qwen-image:fp8`, `qwen-image-lightning:fp8`, and `qwen-image-lightning:fp8-8step`.
Those are separate from the GGUF quantized matrix above and have different
memory and scheduler behavior.

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
