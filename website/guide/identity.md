# Identity Photos

Keep one person's face across arbitrary prompts, scenes, and poses. Give mold a
reference photograph and it conditions the render on that person's identity —
this is [PuLID](https://github.com/ToTheBeginning/PuLID), running natively in
Rust, on **both FLUX and SDXL** with a different adapter for each.

```bash
mold run flux-dev:q4 "a candid photograph of a person on a beach at sunset" \
  --id-image ~/photos/portrait.jpg

mold run sdxl-base:fp16 "a candid photograph of a person on a beach at sunset" \
  --id-image ~/photos/portrait.jpg
```

::: warning License
The face detector and recognizer are InsightFace **pretrained models**, which
are licensed for **non-commercial research purposes only**. Mold does not bundle
them and will not download them until you have explicitly accepted those terms.
The checkpoints identity conditions — FLUX.1-dev under the
[FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md),
the qualified SDXL checkpoints under their own respective licenses — are
separate from this gate. See [Licenses](#licenses) below.
:::

## Which models

Being an SDXL checkpoint is not enough on its own: qualification is an
enumerated list, not a family match, because a checkpoint's own fine-tuning can
retrain the conditioning PuLID was trained against.

| Model                 | Family | Bundle       |
| --------------------- | ------ | ------------ |
| `flux-dev:q4`         | FLUX   | `pulid-flux` |
| `flux-dev:q8`         | FLUX   | `pulid-flux` |
| `sdxl-base:fp16`      | SDXL   | `pulid-sdxl` |
| `juggernaut-xl:fp16`  | SDXL   | `pulid-sdxl` |
| `realvis-xl:fp16`     | SDXL   | `pulid-sdxl` |
| `dreamshaper-xl:fp16` | SDXL   | `pulid-sdxl` |

Every other SDXL checkpoint refuses an identity request rather than rendering
one badly:

- **Turbo / Lightning tiers** — distilled for few-step sampling, a different
  training objective than PuLID's adapter was fit against.
- **Playground v2/v2.5** — trained with an EDM objective rather than SDXL's own.
- **The two Pony derivatives** — retrained conditioning that PuLID's SDXL
  adapter was never fit against.

The live, authoritative answer is always the server's own
`GET /api/models[].supports_identity` — this table is a convenience, not a
second source of truth, and a client should read that field rather than
hard-code this list.

## Setup

Identity conditioning needs three things.

**1. A binary built with the `pulid` feature.** Every official release has it —
the Nix packages, the AUR source packages, and the release tarballs. If you
built from source yourself:

```bash
cargo build --release -p mold-ai --features cuda,pulid   # or metal,pulid
```

`protoc` must be on `PATH` at build time (`nix develop` provides it; otherwise
`protobuf-compiler` / `brew install protobuf`).

**2. A qualified checkpoint.** See [Which models](#which-models) above. Every
other checkpoint refuses the request rather than render it without the face.

```bash
mold pull flux-dev:q4
mold pull sdxl-base:fp16
```

**3. The matching PuLID bundle.** `pulid-flux` and `pulid-sdxl` are two
files-only bundles that share one `pulid` family, one storage root, and four of
their five files — the face detector, recognizer, parser, and vision tower are
identical downloads either way, so a machine that already has one bundle pulls
only the other's adapter (about 984 MB for `pulid-sdxl`, ~1.14 GB for
`pulid-flux`; ~2.2 GB total for both from a clean machine):

```bash
mold pull pulid-flux --accept-license insightface-antelopev2
mold pull pulid-sdxl --accept-license insightface-antelopev2
```

Without `--accept-license`, the pull prints the terms and stops. That is
deliberate: mold will not accept a third-party licence on your behalf. The
acceptance covers both bundles — accept it once and either `mold pull`
proceeds.

`mold licenses` lists what you have accepted on this machine. Acceptance is
recorded per machine, in `$MOLD_HOME`; a remote server needs its own.

## Usage

`--id-weight` and `--id-start-step` mean the same thing on both families.

```bash
# FLUX
mold run flux-dev:q4 "an astronaut in a diner" --id-image face.jpg

# SDXL
mold run sdxl-base:fp16 "an astronaut in a diner" --id-image face.jpg
mold run juggernaut-xl:fp16 "a studio portrait" --id-image face.jpg --id-weight 0.8

# Several references of the same person, averaged into one identity
mold run flux-dev:q4 "a chef in a kitchen" \
  --id-image front.jpg --id-image side.jpg --id-image smiling.jpg

# Weaker identity, more prompt freedom
mold run flux-dev:q4 "a Renaissance oil portrait" --id-image face.jpg --id-weight 0.6

# True CFG (FLUX only): a real negative branch instead of FLUX's distilled guidance
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg \
  --true-cfg 2.0 --guidance 1.0 --negative-prompt "blurry, cartoon"

# SDXL takes an ordinary negative prompt with no extra flag
mold run sdxl-base:fp16 "a hiker on a ridge" --id-image face.jpg \
  --negative-prompt "blurry, cartoon"

# Let the composition settle first, then apply the face
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg --id-start-step 4

# Remote — the server holds the bundle, you send the photograph
MOLD_HOST=http://gpu-box:7680 mold run flux-dev:q4 "a chef in a kitchen" \
  --id-image face.jpg
```

### Flags

| Flag               | Default | Meaning                                                                                                                   |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--id-image`       | —       | Reference photograph. PNG or JPEG, at most 16 MiB, 8192 px per axis, 32 MP. **Repeatable**, up to 4 times.                |
| `--id-weight`      | `1.0`   | Identity strength, `0.0`–`3.0`. Around `0.6`–`0.8` trades likeness for prompt adherence; above `1.2` starts to look waxy. |
| `--id-start-step`  | `0`     | First denoise step identity is applied from. Must be below `--steps`.                                                     |
| `--true-cfg`       | `1.0`   | **FLUX only.** True classifier-free guidance scale, `1.0`–`10.0`. `1.0` is off. Requires `--id-image`. Refused on SDXL.   |
| `--cfg-start-step` | `1`     | **FLUX only.** First denoise step the true-CFG negative branch runs at. Must be below `--steps`. Requires `--true-cfg`.   |

**`--id-weight 0` is completely inert.** Nothing is pulled, decoded, loaded, or
extracted, and the render is byte-identical to the same seed with no identity
flags at all, on either family. That is the falsification case: if the two
differ, the injection is doing something it should not be.

### Several photographs

Repeat `--id-image` (up to four times) to give mold more than one reference of
the **same person**. Each photograph is run through the whole pipeline
independently and the resulting identity tokens are averaged, which is how
[PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI) combines references.
Two or three photographs from different angles usually hold the likeness better
across poses than one does. This works identically on FLUX and SDXL.

The whole set has budgets of its own beside the per-photograph ones: at most
4 images, 32 MiB of encoded bytes, and 64 MP in total. A photograph with no
detectable face refuses the whole request and names which one — dropping it
silently would change the face that renders with nothing to show for it.

Averaging is order-independent, but the saved provenance records every
photograph's name and SHA-256 in the order you gave them.

### True CFG (FLUX only)

FLUX.1-dev is guidance-distilled: it runs a single forward per step and
`--guidance` steers it without a real negative branch, which is why
`--negative-prompt` normally does nothing on FLUX. `--true-cfg` restores actual
classifier-free guidance for identity renders — from `--cfg-start-step`
onwards, each step also runs a second forward over your negative prompt and the
_unconditional_ identity, and the two predictions are combined.

Upstream's own advice, which mold follows rather than enforces: when you turn
true CFG on, drop `--guidance` to `1.0`. Leaving the distilled guidance high
while a real CFG scale is also applied stacks two guidance mechanisms.

It costs close to twice the denoise time — two forwards per step instead of one
— and about 150 MB of extra VRAM, which admission charges before the render is
accepted rather than discovering mid-denoise.

`--true-cfg 1.0` is inert in exactly the way `--id-weight 0` is: the branch is
never constructed and the render is bit-identical to one that never named it.

**SDXL never has this problem.** SDXL is not guidance-distilled — its ordinary
`--guidance` already **is** the classifier-free scale, and `--negative-prompt`
always works, with no flag needed. Naming `--true-cfg` or `--cfg-start-step` on
an SDXL identity request is refused rather than silently ignored, because
accepting them would either mean nothing or double the guidance the render
already performs. PuLID's SDXL adapter conditions its negative branch on the
unconditional identity automatically, with no separate control.

### Older servers

Both several photographs and true CFG are additive request fields, and a
server that predates them does not reject them — it **ignores** them. Sending
several photographs to such a server would render with no face at all; sending
`--true-cfg` would render the ordinary distilled path with no negative branch.
Neither would tell you.

So `mold run` asks first. When the render target does not advertise
`capabilities.identity`, the request is **refused by name** rather than
submitted:

```
$ mold run flux-dev:q4 "a chef" --id-image a.jpg --id-image b.jpg
error: http://gpu-box:7680 does not support more than one identity photograph,
       and sending several to it would render with no face at all. Use a single
       --id-image, or upgrade that server.
```

A single `--id-image` needs no such check — every identity-capable server has
always understood it, on either family — and `--local` is unaffected. If the
server cannot be reached **at all**, nothing is refused: the ordinary local
fallback takes over, and both shapes work in full there. A server that _is_
reachable but cannot answer the check is treated as not supporting them,
because that is exactly how an older one behaves.

### Choosing a photograph

- One clearly visible face, looking roughly at the camera. If several faces are
  found, mold conditions on the largest and says so.
- A tight crop is not required — mold detects and aligns the face itself.
- Phone photographs are fine: EXIF orientation and embedded colour profiles are
  both honoured.
- Resolution beyond about 1024 px buys nothing; the face is resampled to 512 px.

## What actually happens

Extraction is identical for both families — the same detector, recognizer,
parser, vision tower, and IDFormer produce the same shape of 32 identity
tokens. Only the injection into the image model differs:

```
your photograph
      |  SCRFD detects the face and five landmarks       (GPU)
      |  ArcFace embeds the aligned 112x112 crop         (GPU)
      |  BiSeNet segments the aligned 512x512 crop       (GPU)
      |    background -> white, face -> greyscale
      |  EVA02-CLIP-L-14-336 encodes that masked crop    (GPU)
      |  IDFormer resamples both into 32 identity tokens (GPU)
      v
  [32 x 2048] identity
      |
      |  FLUX: 20 cross-attention modules            SDXL: 70 cross-attention
      |  injected between the transformer's           modules injected into the
      |  double-/single-stream blocks (GPU)            UNet's attn2 layers      (GPU)
      v
   your print
```

The mask is not a background _removal_: the face is greyscaled too, so the
vision tower sees shape and no colour at all. That is what PuLID was trained
against, and skipping it costs about a thousandfold in how closely the
extracted identity matches the reference implementation.

### What differs between the families

|                                   | FLUX                                                        | SDXL                                               |
| --------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
| Adapter file                      | `pulid_flux_v0.9.1.safetensors`                             | `pulid_v1.1.safetensors`                           |
| Injection point                   | 20 modules between transformer double-/single-stream blocks | 70 `attn2` cross-attention modules across the UNet |
| `--guidance`                      | distilled conditioning scalar                               | the classifier-free scale                          |
| `--true-cfg` / `--cfg-start-step` | supported                                                   | refused                                            |
| `--negative-prompt`               | only takes effect with `--true-cfg`                         | always active                                      |
| Extra VRAM                        | ~1.25 GB                                                    | ~850 MB                                            |

### When it runs, and what it costs

Everything up to the 32 tokens runs **once per request**, on the same GPU that
will render the print, as the first thing that GPU does — before the checkpoint
is even loaded. A batch of eight siblings extracts the face once and all eight
reuse the identical value.

It is deliberately over before the render begins. The detector, the recognizer,
the parser, the vision tower, and the IDFormer are each built, run, and fully
released in turn, so none of them is ever resident beside the checkpoint or
beside the identity adapter. Mold shows it as its own **Extracting face
identity** stage, and the queue learns how long it takes on your hardware and
includes it in the time estimates it shows you.

Measured on an Apple M4 Max, one extraction is about **0.4 s** on the GPU and
about **1.9 s** on a CPU-only host; on an NVIDIA L40S it is about **0.6 s**,
against **6 s** on that machine's CPU. Rendering the same face again within a
session is under **2 ms** — the identity is cached in memory, keyed on the
photograph, the family, and every model file involved, so a repaired or
updated bundle never serves a stale face, and one photograph never serves one
family the other's embedding. Nothing is written to disk: a face embedding is
biometric data, and mold keeps it only for as long as the server runs.

On the GPU, identity costs extra VRAM beside the checkpoint — about **1.25 GB**
on FLUX (the adapter's twenty cross-attention modules plus their activations)
and about **850 MB** on SDXL (its seventy, smaller modules) — and roughly 10%
of denoise time. The extraction itself needs about **0.7 GB** more while it
runs, which mold reserves for you when the job is queued and releases before
the checkpoint loads.

## Provenance

A print rendered with an identity records it. Saved metadata and the gallery row
carry the reference photograph's **file name** and its **SHA-256**, plus the
weight and start step that were applied — never the photograph itself, and never
your directory layout. A print made from several photographs records every name
and digest in request order, and a true-CFG print records the scale and start
step the branch actually ran with.

```bash
mold info ~/.mold/output/mold-flux-dev-q4-1.png
```

A print with no identity records no identity fields at all, so a knob on an
ordinary render can never read as conditioning that did not happen.

## Limitations

Each of these is refused with a specific message rather than silently rendered,
on both families:

- **Qualified checkpoints only**: see [Which models](#which-models).
- **No LoRA** alongside an identity.
- **No img2img** alongside an identity.
- Video families, Flux.2, Z-Image, Qwen-Image, SD 1.5/3.5, and Wuerstchen do
  not support identity at all.

- **True CFG needs an identity, and is FLUX only**: `--true-cfg` and
  `--cfg-start-step` are qualified only alongside an active identity (a
  photograph and a non-zero `--id-weight`) on FLUX. They are refused on an
  ordinary FLUX render rather than accepted and ignored, and refused outright
  on an SDXL identity request.
- **Older servers refuse both** multi-photograph and true-CFG requests. They
  are gated on `GET /api/capabilities` → `identity`; see
  [Older servers](#older-servers).
- Multiple photographs and true CFG are **CLI and API only** so far. The web,
  desktop, iPhone, TUI, and Discord surfaces still offer a single photograph and
  no true-CFG control, on either family.

## Removing it

```bash
mold rm pulid-flux
mold rm pulid-sdxl
```

Each removes its own adapter. Removing one keeps the four shared extraction
artifacts — the vision tower, face detector, recognizer, and parser mold
derived from them — installed for as long as the other bundle needs them; only
removing both deletes those shared files too. Your recorded licence acceptance
stays either way — `mold licenses` shows it, and it is what lets a later
`mold pull pulid-flux` or `mold pull pulid-sdxl` proceed without the flag.

## Licenses

| Artifact                                                                         | License                                                                                   |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `pulid_flux_v0.9.1.safetensors`                                                  | Apache-2.0 ([guozinan/PuLID](https://huggingface.co/guozinan/PuLID))                      |
| `pulid_v1.1.safetensors`                                                         | Apache-2.0 ([guozinan/PuLID](https://huggingface.co/guozinan/PuLID))                      |
| `EVA02_CLIP_L_336_psz14_s6B.pt`                                                  | MIT ([QuanSun/EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP))                         |
| `scrfd_10g_bnkps.onnx`, `glintr100.onnx`                                         | InsightFace pretrained models — **non-commercial research only**                          |
| `parsing_bisenet.pth`                                                            | MIT ([facexlib](https://github.com/xinntao/facexlib))                                     |
| `flux-dev:q4` / `flux-dev:q8`                                                    | FLUX.1-dev Non-Commercial License                                                         |
| `sdxl-base:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, `dreamshaper-xl:fp16` | Each checkpoint's own license — see the [model catalog](https://utensils.io/mold/models/) |

Mold ships none of these; it downloads them on request, and refuses the
InsightFace pair until acceptance is recorded. The InsightFace _code_ is MIT;
the _weights_ are not, and that distinction is the reason for the gate.

## Troubleshooting

**"this server was built without PuLID face-identity support"** — the binary
does not link the feature. Use an official release, or rebuild with
`--features …,pulid`.

**"no face was detected in the identity image"** — the detector found nothing
above its threshold. Try a photograph where the face is larger, better lit, or
closer to frontal.

**"does not support face-identity conditioning"** — the model is not one of the
qualified checkpoints in [Which models](#which-models).

**"identity photo 2 of 3: no face was detected"** — one photograph of a set is
unusable. The number is the position you gave it in.

**"id_image and id_images are the same field in two shapes"** — an API client
sent both. Put every photograph in `id_images`, or a single one in `id_image`.

**"true_cfg and cfg_start_step are qualified only alongside active
face-identity conditioning"** — `--true-cfg` was used without `--id-image`, or
with `--id-weight 0`, on FLUX.

**"true_cfg is not supported for this model"** — `--true-cfg` was used against
an SDXL identity request. Drop it; SDXL's `--guidance` is already the
classifier-free scale.

**"no PuLID bundle resolved"** — run `mold pull pulid-flux` (for FLUX) or
`mold pull pulid-sdxl` (for SDXL) with `--accept-license
insightface-antelopev2` on the machine that renders, which for a remote run is
the server, not your laptop.
