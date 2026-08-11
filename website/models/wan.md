# Wan Video

Text-to-video generation from [Alibaba's Wan team](https://github.com/Wan-Video),
based on a flow-matching DiT with a UMT5-XXL text encoder and a causal 3D
video VAE that streams decoding one latent frame at a time. mold implements
the family natively in Rust.

- **Developer**: [Wan-AI](https://huggingface.co/Wan-AI)
- **License**: Apache 2.0
- **Reference**: [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) ·
  [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)

> **Note**: Video output defaults to MP4. Also supports GIF, WebP, and APNG
> via `--format`. Frame count must be 4n+1 (77, 81, 121, ...) due to the
> VAE's 4x temporal compression. Width and height must be multiples of 16 —
> except `wan22-ti2v-5b`, whose 2.2 VAE requires multiples of 32. Wan
> generation currently targets CUDA; CPU runs are correctness-only.

## Variants

| Model                 | Steps | Approx total pull | Notes                                        |
| --------------------- | ----- | ----------------- | -------------------------------------------- |
| `wan21-t2v-1.3b:bf16` | 30    | ~14.5 GB          | 480p text-to-video; smallest, fastest pull   |
| `wan21-t2v-14b:q5`    | 30    | ~23 GB            | Q5_K_M 2.1 14B; 480p text-to-video           |
| `wan21-t2v-14b:q8`    | 30    | ~27.5 GB          | Q8_0 2.1 14B; the 2.1 quality tier           |
| `wan22-ti2v-5b:fp16`  | 20    | ~22.8 GB          | 720p24 text- and image-to-video              |
| `wan22-ti2v-5b:q8`    | 20    | ~18 GB            | Q8_0 5B; 8-12 GB cards at reduced settings   |
| `wan22-t2v-a14b:q5`   | 4     | ~36 GB            | 480p16 text-to-video, 4-step Lightning tier  |
| `wan22-t2v-a14b:q8`   | 20    | ~42 GB            | Same weights at Q8_0, no distill             |
| `wan22-t2v-a14b:q4`   | 4     | ~33 GB            | Q4_K_M Lightning; 12-16 GB needs reduced use |
| `wan22-i2v-a14b:q5`   | 4     | ~36 GB            | 480p16 image-to-video, 4-step Lightning tier |
| `wan22-i2v-a14b:q8`   | 20    | ~42 GB            | Same weights at Q8_0, no distill             |
| `wan22-i2v-a14b:q4`   | 4     | ~33 GB            | Q4_K_M Lightning; 12-16 GB needs reduced use |

Totals include the shared UMT5-XXL encoder (~11.4 GB), tokenizer, and the
variant's VAE. The encoder is shared across every Wan model under
`shared/wan/`, so a second Wan pull only fetches the checkpoint and VAE.

### A14B is two models

Wan 2.2 A14B is a mixture of experts along the _noise_ axis: two complete 14B
transformers, one trained for the early, structural part of the schedule and
one for the late, detail part. mold loads the high-noise expert first, switches
once when the schedule crosses the boundary (timestep 875 for T2V, 900 for
I2V), and drops each expert before loading its partner — so **VRAM is the
larger of the two experts, not their sum** (~10.8 GB at `:q5`, ~15.4 GB at
`:q8`). Disk is the sum, which is why the pull totals are large.

Admission prices a wan render from its frame count, not just its resolution.
Video memory is dominated by the token count — `((frames − 1) / 4 + 1)` latent
frames times the patch grid — so the same 832×480 shape costs several times
more at 81 frames than at 17. The server predicts that peak from the
checkpoint's own header before the UMT5 encode and the expert load, and refuses
a shape that cannot fit rather than failing part-way through the denoise. The
model is calibrated against measured peaks on an RTX 4090 and validated on a
second checkpoint it was not fitted to. A rejection names frames first, because
that is the most effective lever, and suggests the next quantized tier down
rather than the one that just failed.

Community A14B adapters are published the same way: a high-noise file and a
low-noise file, distilled together and explicitly not interchangeable. Bind one
to its expert with `--lora file.safetensors@high` (or `@low`), or the additive
per-entry `expert` field on the API. mold infers the binding from the dominant
filename conventions (`high_noise`, `HighNoise`, `HIGH`) when the field is
absent and says so in the progress output — an adapter with no expert marker
still applies to both experts, which is right for a genuinely unpaired one. A
single-expert checkpoint refuses an explicit `expert` rather than ignoring it.

The `:q5` tier additionally pulls lightx2v's 4-step distill — a separate
adapter for each expert — and defaults to guidance 1.0. That is not a weak
setting: at guidance ≤ 1 mold skips the unconditional pass entirely, so each
step is one forward instead of two. Four steps at one forward each is where
the tier's speed comes from.

## Usage

```bash
# 480p, 81 frames @ 16 fps (defaults)
mold run wan21-t2v-1.3b "a red fox trotting through fresh snow, golden hour"

# 720p24, 121 frames — Wan 2.2 5B
mold run wan22-ti2v-5b "aerial view of waves breaking on a black sand beach" \
  --width 1280 --height 704 --frames 121 --fps 24

# Wan 2.2 A14B, 4-step Lightning tier
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"

# A14B image-to-video from a still
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png

# Single-frame text-to-image — Wan 2.2 as a still-image model
mold run wan22-t2v-a14b:q5 "a lighthouse at dusk, volumetric fog" \
  --frames 1 --output still.png

# First/last-frame interpolation: anchor both endpoints (A14B I2V or TI2V-5B)
mold run wan22-i2v-a14b:q5 "the sapling grows into an oak" \
  --image sapling.png --last-image oak.png

# The fp8-scaled quality tier — the 20-step recipe with ~2.6 GB more VRAM headroom
mold run wan22-t2v-a14b:fp8 "storm waves crash over the lighthouse"
```

The `:fp8` tier is not faster than `:q8` — measured on an RTX 4090 at
33f/832x480 under identical settings, both run ~28 s/step (the denoise is
compute-bound, not weight-decode-bound) — but its peak VRAM is 17,646 MiB
against `:q8`'s 20,278 MiB. The trade-off: fp8-scaled weights refuse LoRA
stacks (merging would re-round every targeted weight to three mantissa
bits), so adapters — the Lightning distills included — need the GGUF or
bf16 tiers.

## Where an A14B step actually goes

A kernel-level audit (RTX 4090, 33f/832x480 — 14,040 video tokens) attributes
the quality-tier step. Under the device-synced profiler the step measures
~42.5 s, of which **dense self-attention SDPA is ~21 s**, the quantized
matmuls ~7 s — the GGUF fast path (MMQ) engages for every shipped quant mix,
and forcing the dequantize-per-forward fallback triples that bucket to ~21 s
while leaving attention untouched — and the BF16↔F32 boundary casts are
~1.5 s. Per-phase syncs inflate the many small ops far more than the 80
large attention kernels, so against the real ~28 s step SDPA's share sits
between half and roughly three quarters: attention dominates. Weight
size barely matters: measured denoise time is **28.2 s/step at `:q8`
quality, 30.2 at `:q5` quality** (its Lightning adapter runs as a per-step
parallel branch on GGUF), and **15.9 at `:q5` fast** (guidance 1 skips the
uncond forward — exactly half a CFG step). Two diagnostic env knobs ship for
re-running the audit: `MOLD_WAN_STEP_PROFILE=1` prints a per-phase,
device-synced timing line per denoise step, and `MOLD_WAN_FORCE_DMMV=1`
forces the quantized-matmul fallback for A/B comparison — neither belongs in
production use.

### FlashAttention on Wan

A `--features cuda,flash-attn` build routes the Wan DiT's self- and cross-attention through candle-flash-attn v2 and defaults `MOLD_ATTN` to `flash`. Measured on an RTX 4090 with the same binary, only the backend varying (`wan22-t2v-a14b:q5`, 53 frames at 832x480):

| Backend | Peak VRAM  | Wall clock |
| ------- | ---------- | ---------- |
| `flash` | 21,354 MiB | 75.3 s     |
| `math`  | 22,250 MiB | 158.4 s    |

So flash is worth **2.1x on speed** and only ~900 MiB on peak. That is the opposite of the usual expectation, and it is why longer clips are not unlocked by switching backends: at 81 frames the estimate is ~27.7 GB against ~24.8 GB usable, and flash's measured per-token saving extrapolates to about 1.3 GB — not the ~3 GB that would be needed. Reaching 81 frames on a 24 GB card needs partial block offload, which is not wired for this family yet. Note also that `flash-attn` ships in no release artifact, so this is a source-build configuration.

At `--frames 1` Wan renders a still: png/jpeg output is admitted (and png is
the default there), the image embeds the same `mold:parameters` provenance as
every image family, and the gallery treats it as an upscale-eligible still.
Upstream defines its `t2i-14B` task as the same weights at `frame_num=1`; any
frame count above 1 keeps the video-only output contract.

Wan checkpoints were tuned against a specific long Chinese negative prompt;
mold applies it automatically whenever a request carries no negative at all.
Every surface now shows it: `/api/models` advertises the tuned default per
model (`default_negative_prompt`), the web/desktop/iPhone Negative field and
the TUI's Advanced → Negative editor prefill it, and editing the text
replaces it. Clearing the field (or passing `--no-negative` on the CLI, or
`negative_prompt: none` on Discord) sends an explicit empty negative, which
the engine honors as a real empty uncond instead of re-applying the default.
An untouched field stays absent on the wire, so older servers behave exactly
as before. Saved gallery metadata records the negative that actually
conditioned the render, so Library details and "Reuse settings" are truthful.

## Source-image contracts

Wan checkpoints split three ways, and `/api/models` advertises which through
the additive per-model `source_image` field so every surface offers exactly
what the checkpoint accepts:

- **`unsupported`** — `wan21-t2v-1.3b`, `wan21-t2v-14b:*`, `wan22-t2v-a14b:*`:
  pure text-to-video; a supplied image is rejected at admission.
- **`optional`** — `wan22-ti2v-5b:*`: text-to-video, or the source pinned as
  frame 0 through latent inpainting.
- **`required`** — `wan22-i2v-a14b:*`: the image is half the model input;
  admission rejects a request without one.

Installed `cv:`/`hf:` wan checkpoints classify from their own tensor shapes —
the same read the engine performs — never from their names.

## First/last-frame interpolation

`--image` + `--last-image` (wire: a two-entry `keyframes` list anchoring
pixel frames 0 and F-1) renders motion between two stills — upstream's FLF2V
task, ComfyUI's `WanFirstLastFrameToVideo`. A14B I2V drives the 36-channel
mask contract with the endpoint flag in mask channel 3; TI2V-5B pins both
endpoint latent frames through the same inpaint path diffusers' `last_image`
uses. Any other keyframe layout is refused at admission — the family has no
mid-clip keyframe path.

## Defaults and limits

| Property   | `wan21-t2v-1.3b`  | `wan21-t2v-14b:*` | `wan22-ti2v-5b`     | `wan22-*-a14b:q5` | `wan22-*-a14b:q8` |
| ---------- | ----------------- | ----------------- | ------------------- | ----------------- | ----------------- |
| Resolution | 832x480 / 480x832 | 832x480 / 480x832 | 1280x704 / 704x1280 | 832x480           | 832x480           |
| Frames     | 81 @ 16 fps       | 81 @ 16 fps       | 121 @ 24 fps        | 53 @ 16 fps       | 33 @ 16 fps       |
| Steps      | 30                | 30                | 20                  | 4                 | 20                |
| Guidance   | 6.0               | 6.0               | 5.0                 | 1.0 (no CFG pass) | per-expert¹       |
| Flow shift | 8.0               | 8.0               | 8.0                 | 5.0               | 5.0               |
| Sampler    | FlowUniPC (bh2)   | FlowUniPC (bh2)   | FlowUniPC (bh2)     | FlowUniPC (bh2)   | FlowUniPC (bh2)   |

¹ The `:q8` quality tier advertises guidance 3.5, but by default mold applies
upstream's **per-expert** scales, switching at the same boundary as the expert
swap: T2V runs 4.0 while the high-noise expert is resident and 3.0 after the
boundary; I2V runs 3.5 throughout (`wan_{t2v,i2v}_A14B.py`
`sample_guide_scale`). Passing an explicit `--guidance` pins that one scale
for the whole schedule — except an explicit 3.5 on the quality tier, which is
indistinguishable from the default on the wire and selects the per-expert
pair. The Lightning tiers (default 1.0) treat every value, 3.5 included, as
an explicit uniform choice.

The A14B frame defaults are the measured 24 GB envelope, not the checkpoint's
trained 81-frame clip length: on an RTX 4090 the `:q5` pair peaks at
23,975 MiB rendering 53 frames at 832x480 (81 frames peaked at 23.0 GB and
then ran out of memory), and the `:q8` pair's ~4.6 GB larger resident expert
moves its edge to ~33 frames. Larger cards simply pass `--frames 81`.
Reclaiming 81-frame clips on 24 GB is tracked in
[#776](https://github.com/utensils/mold/issues/776).

The sampler schedule matches the one lightx2v's Lightning distills were
trained against (diffusers' flow-UniPC grid), so the 4-step tier reproduces
its published timesteps exactly.

## Recipe controls

Three request-level knobs reproduce published Wan recipes; each stays absent
by default so the tier defaults above remain authoritative.

**Flow shift** (`--sample-shift`, env fallback `MOLD_WAN_SHIFT`) is the
family's primary quality/character knob — upstream ships per-task values from
3.0 to 16, diffusers documents 2.0–5.0 for low resolutions and 7.0–12.0 for
high, Lightning wants 5, upstream's 720p quality A14B T2V wants 12, and
ComfyUI templates ship 8. Precedence is request > env > per-tier default, so
two queued jobs can run different shifts on one server.

**Sample solver** (`--sample-solver unipc|euler|dpm++`, env fallback
`MOLD_WAN_SOLVER`, wire slot `scheduler`) selects the denoise algorithm:

- `unipc` (default) — FlowUniPC order-2 predictor-corrector, the UAT'd
  recipe every existing seed reproduces.
- `euler` — plain flow Euler over the same grid; the solver the lightx2v
  4-step Lightning distills were tuned for. At 4 steps, order 2 vs order 1
  is a real output difference.
- `dpm++` — upstream's `FlowDPMSolverMultistepScheduler` (order 2,
  dpmsolver++ midpoint) over its own sigma grid, which starts at exactly 1.0
  (first DiT timestep 1000) — useful for A/B-ing quality-tier renders
  against upstream, golden-pinned to `fm_solvers.py`.

**Distill strength** (`--distill-strength high=X,low=Y`, or one number for
both experts) scales the manifest-shipped Lightning adapters per expert. The
fast tier's documented failure mode is reduced motion / grayish output; the
community mitigation (lightx2v-acknowledged) is high-noise strength 1.5–2.0
with low at 1.0, and/or 5–6 steps at guidance 1. A strength on a tier that
ships no distill in that slot (the `:q8` quality tier) is refused, not
ignored.

## Quantized checkpoints and adapters

A14B ships as GGUF. Quantized weights stay quantized in memory and dequantize
inside the matmul, which is what keeps a 14B expert at ~10.8 GB rather than
~28 GB. A LoRA cannot be merged into a weight in that state without
requantizing it, so on GGUF mold applies adapters as a parallel branch
instead — low-rank for A/B pairs, dense for full-weight `.diff` deltas — the
same arithmetic, applied at full precision, with no load cost. On bf16
safetensors the adapter is merged as the weights are read; fp8-scaled
checkpoints refuse adapter stacks rather than re-round their weights.

Community adapters that carry `.diff` (full weight delta) and `.diff_b`
(bias delta) tensors alongside or instead of their low-rank pairs — Kijai's
Wan 2.1 lightx2v extractions, lightx2v's distill pairs, musubi-tuner-trained
Civitai LoRAs — load fully: `W' = W + strength·diff`, `b' = b +
strength·diff_b`, matching ComfyUI. The kohya alpha never rescales a
full-weight delta (a delta has no rank), and a delta naming a tensor the
checkpoint does not have still refuses the whole adapter rather than
applying the part that matches.

`*_fp8_e4m3fn_scaled` safetensors also load: the weights stay 1 byte per
parameter and dequantize per call against their per-module scale. The `e5m2`
variants some repositories publish beside them are refused by name — mold
reads the e4m3 flavour only.

### Checkpoint key layouts

Three safetensors key layouts load, and mold picks between them by reading the
file's own header — never its filename:

- **Upstream / ComfyUI** names at the file root (`blocks.0.self_attn.q.*`).
- **Comfy-Org repacks**, the same names under a `model.diffusion_model.` prefix.
- **diffusers** `WanTransformer3DModel` exports (`blocks.0.attn1.to_q.*`,
  `condition_embedder.*`, `ffn.net.0.proj`), translated at load through the
  same rename table the golden parity test uses — including diffusers'
  `norm2`/`norm3` swap.

`patch_embedding.weight` is spelled identically in all three, so the
self-attention query projection is what actually distinguishes them. A
checkpoint matching none of the layouts is refused by name at load rather than
constructing a transformer from default weights and rendering noise.

## Discovery

The models in the table above install by name. Community Wan fine-tunes are
additionally discoverable in the catalog — open **Models → Discover** in Mold
Studio and search for `wan`, then install the row, or pull a Civitai version id
directly:

```bash
mold pull wan22-t2v-a14b:q5   # manifest name — the A14B fast tier
mold pull wan21-t2v-1.3b      # bare names resolve their default tag
mold pull cv:<version-id>     # a catalog row
```

`mold pull` takes manifest names and catalog ids (`cv:…`, `hf:…`). A catalog id
has to name a row the catalog actually supports, which is not the same as any
Hugging Face repository — `hf:Wan-AI/Wan2.2-T2V-A14B`, for instance, is the
upstream aggregate repository, not a single runnable checkpoint; the A14B
manifest tiers above are its supported route. When in doubt, install from
**Models → Discover**, which only lists rows this build can run.

Every Wan checkpoint in the wild ships the transformer alone, so a catalog
install also pulls the shared UMT5-XXL encoder and the matching VAE. Those are
the same files the manifest models use, under `shared/wan/`, so a second Wan
install reuses them.

**A14B fine-tunes install as a pair.** Civitai publishes the two A14B experts
as separate model versions of one model (`… HighNoise` / `… LowNoise`,
`HIGH Q8` / `LOW Q8`, and similar). mold pairs the high-noise version with its
low-noise sibling into one install: the catalog shows one row per pair, either
expert's `cv:` id resolves to the same two-file download, and the installed
model denoises with both experts exactly like the manifest tiers (switching at
timestep 875 for T2V, 900 for I2V). A version whose counterpart cannot be
identified with confidence — merged "all-in-one" republications, a
high-noise-only upload, ambiguous naming — stays visible but is refused with
the reason rather than installed as a single expert, which would render
silently wrong. If one half of an installed pair goes missing, the row reports
not-installed and re-running the install resumes just the missing half.

What the catalog deliberately does not offer:

- **Wan 2.1 image-to-video** conditions through a CLIP-vision cross-attention
  branch mold's transformer does not implement, so the download would install
  and then fail to generate.
- **Wan 2.5 and 2.7** are later architectures with no mold engine.
- **GGUF Civitai rows** — the Civitai path is safetensors-only; the GGUF A14B
  tiers ship through the manifest names above.
- **4-bit (NF4/NVFP4) safetensors** — the Wan loader reads dense, scaled-FP8,
  and GGUF weights only, so these versions are dropped rather than offered as
  multi-gigabyte downloads that fail at load.

Wan 2.1 text-to-video at either size, TI2V-5B, and paired A14B fine-tunes
install from the catalog normally.

## Roadmap

Remaining Wan work is tracked in the
[Wan Video milestone](https://github.com/utensils/mold/milestone/4).

The Wan ecosystem is much wider than text-to-video and image-to-video —
VACE, Fun-Control, camera, Phantom, track, audio-driven (S2V, HuMo,
InfiniteTalk, WanDancer), and character animation (Animate, SCAIL) all exist
upstream. Every one of them carries an explicit decision in the
**[ComfyUI parity ledger](https://github.com/utensils/mold/blob/main/docs/architecture/wan-comfyui-parity-ledger.md)**:
supported, earning an engine, deferred with the blocker named, or dropped
with the reason. If a Wan variant you use is not in this page's model table,
that ledger says why and what would change it.

One naming trap worth stating here: ComfyUI's **Fun Inpaint** is the
first/last-frame contract under a different brand name. In mold it is
`--image` plus `--last-image` (the `keyframes` pair), not a separate model.
