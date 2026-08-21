# MiniMax H3

MiniMax H3 is an audio-video generation family from
[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3). Mold can discover,
download, verify, repair, inventory, and remove two compact Comfy variants. The
files are downloaded directly from their pinned Hugging Face repositories;
Mold does not bundle or mirror the weights.

::: warning CUDA FL2VA is the first supported runtime
Both compact variants can be downloaded on any Mold host. Mold's SM89 CUDA
release can run the compact FL2VA model for the supported request profile below.
Ref2VA execution and the CPU backend remain unavailable. Broader request
shapes also remain unavailable until those paths are implemented and
tested; Mold reports that limitation normally rather than treating it as a
licensing or authorization failure.
:::

::: info Apple Metal is a correctness-only path in progress
The Apple Silicon execution path exists as of #1164 — family-scoped BF16, a
folded audio-VAE reduction, chunked dense attention sized so the score matrix
fits a Metal buffer, the portable INT8 ConvRot arm, and fp8-scaled weights
refused by name because candle has no Metal fp8 widening kernel. It is
advertised as **correctness-only**, the same tier Wan and LTX-2 landed on
before their performance qualification. Metal is not yet a runnable H3 route:
the public runtime profile is still SM89 CUDA, and Metal execution waits on
qualification against real Apple Silicon hardware. Expect Metal to be slow when
it lands — the reference MLX port measures minutes per step at 5 s — so this is
a portability path, not a speed one.
:::

## Compact variants

| Model                                                 | Task                                              | Total pull | Runtime status                       |
| ----------------------------------------------------- | ------------------------------------------------- | ---------: | ------------------------------------ |
| `minimax-h3-fl2va:comfy-pruned-int8`                  | First/last-frame conditioning with audio          |  42.482 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step`      | FL2VA + reviewed Turbo 8-step LoRA (9 steps)      |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p` | FL2VA + reviewed Turbo 4-step 768p LoRA (5 steps) |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-ref2va:comfy-pruned-int8`                 | Reference media to video with audio               |  42.482 GB | Downloadable; execution unavailable  |

Pull a variant from the CLI, or install it from **Models → Discover** in Mold
Studio:

```bash
mold pull minimax-h3-fl2va:comfy-pruned-int8
mold pull minimax-h3-fl2va:comfy-pruned-int8-turbo-8step
mold pull minimax-h3-ref2va:comfy-pruned-int8
```

The files are revision-pinned and SHA-256 verified before Mold marks the model
complete. Raw repository IDs, custom manifests, configured aliases, and live
catalog recipes cannot substitute for any registered graph.

## Reviewed Turbo tiers

A Turbo tier is a reviewed LoRA adapter overlaid on the **same** compact INT8
FL2VA checkpoint — nothing about the base artifact contract relaxes, and the
only request axis a tier moves is its fixed step count. Each Turbo model tag
pulls the complete base stack plus one pinned adapter
(1,956,193,000 bytes for 8-step, 1,956,192,992 bytes for 4-step 768p, stored
once under `shared/minimax-h3/loras/` and shared by both tags):

- `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step` — 9 terminal-inclusive
  sampler grid points (8 model evaluations)
- `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p` — 5 terminal-inclusive
  sampler grid points (4 model evaluations)

A Turbo tag stores its base stack in the base checkpoint's own directory
(`minimax-h3-fl2va-comfy-pruned-int8/` plus the shared family bucket), so a
machine that already has `minimax-h3-fl2va:comfy-pruned-int8` installed
downloads only the ~1.96 GB adapter — and pulling a Turbo tag first means a
later base pull downloads nothing. Because the shared bytes genuinely
constitute a complete base install, a Turbo-only pull also makes the base tag
read as installed. Removal ref-counts every complete install: removing a
Turbo tag deletes only its adapter while the base stack has another owner,
and removing the base keeps every file a fully installed Turbo tag still
uses (a Turbo tag missing its adapter owns nothing). Freeing everything is
two removals — the Turbo tag, then the base — and each removal reports the
kept files with the tags that still use them.

Selecting a Turbo model resolves the base INT8 transformer, the tier's pinned
adapter, and the tier's reviewed step count with no extra configuration. The
`MOLD_H3_TURBO_ADAPTER` / `MOLD_H3_TURBO_TIER` environment pair is a
capture-scope UAT override honored only by `h3-private-uat` builds; ordinary
builds refuse a set pair rather than letting two selection authorities
disagree.

## Download size and sources

Each compact variant has the same component graph except for its task-specific
transformer:

| Component                                              |              Bytes |  Decimal size | Upstream source                                                                                                     |
| ------------------------------------------------------ | -----------------: | ------------: | ------------------------------------------------------------------------------------------------------------------- |
| Task transformer                                       |     20,970,379,616 |     20.970 GB | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/eb8a16107c595128b3a578f82d2ce2f75920c355) |
| Qwen3-VL NVFP4-AWQ text encoder                        |     15,687,142,551 |     15.687 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| FP16 video VAE                                         |      5,207,808,496 |      5.208 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| FP32 audio VAE                                         |        605,254,808 |      0.605 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| Tokenizer, processor, scheduler, and component configs |         11,504,847 |      0.012 GB | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/bfc8ed0353f5a9733be73e6b2c98ec0948195b86) |
| **One complete variant**                               | **42,482,090,318** | **42.482 GB** | Both pinned repositories                                                                                            |

A Turbo tag adds one adapter to this graph:
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/dc559027db79c174125df4d827db55cd11178860)
`loras/` at pinned revision `dc559027db79c174125df4d827db55cd11178860`
(1,956,193,000 bytes for `-turbo-8step`, 1,956,192,992 bytes for
`-turbo-4step-768p`), bringing one complete Turbo variant to 44,438,283,318 or
44,438,283,310 bytes (44.438 GB).

The encoder, VAEs, and common support files are shared between the variants.
After one complete variant is installed, adding the other downloads its 20.970
GB transformer and 546-byte task config. Both variants together occupy 63.452
GB (63,452,470,480 bytes) of model payloads, excluding filesystem and Hugging
Face cache overhead.

Sizes above are decimal gigabytes (`1 GB = 1,000,000,000 bytes`) and describe
downloads and disk use, not peak VRAM. They come from Mold's registered,
full-file manifest identities rather than estimates from repository listings.

## Supported FL2VA request

The initial compact CUDA implementation supports this request profile:

- an SM89 CUDA GPU with sufficient VRAM and the H3 attention/runtime operators enabled
- `1344x768`, batch size 1
- exactly 124 frames at 24 fps
- exactly 21 terminal-inclusive sampler grid points (20 model evaluations) for
  the base model; a reviewed Turbo tag instead requires exactly its tier's own
  count (9 for `-turbo-8step`, 5 for `-turbo-4step-768p`)
- one required first-frame image and no last-frame endpoint
- MP4 output with synchronized generated audio
- a prompt of roughly 1,000 tokens or fewer: the reviewed conditioner sequence
  budgets 2,048 rows, of which the first-frame image's vision pads and label
  take 1,014, and a longer prompt is refused immediately with its exact budget
  named rather than after artifact verification

When a first-frame image is attached without explicit `--width`/`--height`,
the CLI and Discord builders submit the fixed `1344x768` envelope regardless
of the source's aspect ratio — the engine fits the frame internally. The
aspect-derived short-edge canvas applies only to the hidden official BF16
reference.

Every reviewed compact and Turbo tag advertises exactly this envelope, so
Create on web, desktop, and iPhone offers the single `1344x768` canvas with the
tier's step count and 124 frames already fixed, and an off-envelope request is
refused at submission instead of after the model loads.

Mold rejects rather than silently resizing, rerouting, changing steps, dropping
the source image, or falling back to another backend. A downloaded checkpoint
can remain stored on an unsupported host; Create and request routing become
available only when that host advertises the matching CUDA runtime capability.
The public SM89 runtime uses a source-controlled conservative memory profile;
it does not require private authorization or qualification-record files.

## License and support boundary

H3 model weights and upstream model assets use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE),
at pinned revision `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`, not
Mold's MIT license. Review those terms for your intended use. Mold's source code
and H3 integration remain under Mold's repository license. Mold downloads the
weights directly from the pinned upstream repositories and verifies every file;
it does not bundle or mirror the payloads in Mold releases. The project's
[license and integration record](https://github.com/utensils/mold/blob/main/docs/architecture/minimax-h3-authorization.md)
documents the completed governance decision.

The completed project review authorizes H3 use in every territory and across
Mold's CLI, server/API, Discord, desktop, web, iPhone, TUI, gallery,
remote-client, shared-server, and hosted paths. It also covers generated-output
distribution and model distribution or redistribution. Technical availability
remains limited to routes Mold has implemented and qualified; authorization
does not make an unsupported task, device, or request shape runnable.

The license link and notice in this guide and Mold's README are the project's
required user-facing license, attribution, disclosure, downstream-term, and
acceptable-use delivery. Mold does not require a separate clickthrough,
geolocation check, H3-specific generated-content label, downstream contract, or
surface-specific acceptable-use control. Existing Mold authentication,
validation, capability, safety, and operational controls continue to apply.

The official BF16 checkpoints remain hidden qualification references. Their
much larger artifact graphs are not public Mold download options.
