# Identity Photos

Keep one person's face across arbitrary prompts, scenes, and poses. Give mold a
reference photograph and it conditions the render on that person's identity —
this is [PuLID-FLUX](https://github.com/ToTheBeginning/PuLID), running natively
in Rust.

```bash
mold run flux-dev:q4 "a candid photograph of a person on a beach at sunset" \
  --id-image ~/photos/portrait.jpg
```

::: warning License
The face detector and recognizer are InsightFace **pretrained models**, which
are licensed for **non-commercial research purposes only**. Mold does not bundle
them and will not download them until you have explicitly accepted those terms.
The FLUX.1-dev checkpoints they condition are separately under the
[FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
See [Licenses](#licenses) below.
:::

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

**2. A qualified checkpoint.** Identity is qualified for `flux-dev:q4` and
`flux-dev:q8` only. Other FLUX tiers, and every other family, refuse the request
rather than render it without the face.

```bash
mold pull flux-dev:q4
```

**3. The PuLID bundle**, four auxiliary artifacts totalling about 2.1 GB:

```bash
mold pull pulid-flux --accept-license insightface-antelopev2
```

Without `--accept-license`, the pull prints the terms and stops. That is
deliberate: mold will not accept a third-party licence on your behalf.

`mold licenses` lists what you have accepted on this machine. Acceptance is
recorded per machine, in `$MOLD_HOME`; a remote server needs its own.

## Usage

```bash
# The whole feature
mold run flux-dev:q4 "an astronaut in a diner" --id-image face.jpg

# Weaker identity, more prompt freedom
mold run flux-dev:q4 "a Renaissance oil portrait" --id-image face.jpg --id-weight 0.6

# Let the composition settle first, then apply the face
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg --id-start-step 4

# Remote — the server holds the bundle, you send the photograph
MOLD_HOST=http://gpu-box:7680 mold run flux-dev:q4 "a chef in a kitchen" \
  --id-image face.jpg
```

### Flags

| Flag              | Default | Meaning                                                                                                                   |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--id-image`      | —       | Reference photograph. PNG or JPEG, at most 16 MiB, 8192 px per axis, 32 MP.                                               |
| `--id-weight`     | `1.0`   | Identity strength, `0.0`–`3.0`. Around `0.6`–`0.8` trades likeness for prompt adherence; above `1.2` starts to look waxy. |
| `--id-start-step` | `0`     | First denoise step identity is applied from. Must be below `--steps`.                                                     |

**`--id-weight 0` is completely inert.** Nothing is pulled, decoded, loaded, or
extracted, and the render is byte-identical to the same seed with no identity
flags at all. That is the falsification case: if the two differ, the injection
is doing something it should not be.

### Choosing a photograph

- One clearly visible face, looking roughly at the camera. If several faces are
  found, mold conditions on the largest and says so.
- A tight crop is not required — mold detects and aligns the face itself.
- Phone photographs are fine: EXIF orientation and embedded colour profiles are
  both honoured.
- Resolution beyond about 1024 px buys nothing; the face is resampled to 512 px.

## What actually happens

```
your photograph
      |  SCRFD detects the face and five landmarks       (CPU)
      |  ArcFace embeds the aligned 112x112 crop         (CPU)
      |  EVA02-CLIP-L-14-336 encodes the 512x512 crop    (CPU)
      |  IDFormer resamples both into 32 identity tokens (CPU)
      v
  [32 x 2048] identity
      |
      |  20 cross-attention modules injected between the
      |  FLUX transformer blocks, at every denoise step   (GPU)
      v
   your print
```

Everything up to the 32 tokens runs **once**, on the CPU, when the request is
admitted — before the model is even placed on a GPU. A batch of eight siblings
extracts the face once and all eight reuse the identical value. That also means
identity extraction never competes with the text encoders for memory: it has
finished and released its ~1.4 GB before the render starts.

On the GPU, identity costs about **1.25 GB** of VRAM beside the checkpoint (the
adapter's twenty cross-attention modules plus their activations) and roughly
10% of denoise time.

## Provenance

A print rendered with an identity records it. Saved metadata and the gallery row
carry the reference photograph's **file name** and its **SHA-256**, plus the
weight and start step that were applied — never the photograph itself, and never
your directory layout.

```bash
mold info ~/.mold/output/mold-flux-dev-q4-1.png
```

A print with no identity records no identity fields at all, so a knob on an
ordinary render can never read as conditioning that did not happen.

## Limitations

Milestone 1 is deliberately narrow. Each of these is refused with a specific
message rather than silently rendered:

- **Qualified tiers only**: `flux-dev:q4` and `flux-dev:q8`.
- **No LoRA** alongside an identity.
- **No img2img** alongside an identity.
- Video families, Flux.2, Z-Image, Qwen-Image, SD, and Wuerstchen do not support
  identity at all.

Also not implemented: facexlib's background mask (upstream applies one before
the vision tower), and fusing several photographs of the same person into one
stronger identity.

## Removing it

```bash
mold rm pulid-flux
```

This deletes all four downloaded artifacts and the vision tower mold derived
from them on first use. Your recorded licence acceptance stays — `mold licenses`
shows it, and it is what lets a later `mold pull pulid-flux` proceed without the
flag.

## Licenses

| Artifact                                 | License                                                              |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `pulid_flux_v0.9.1.safetensors`          | Apache-2.0 ([guozinan/PuLID](https://huggingface.co/guozinan/PuLID)) |
| `EVA02_CLIP_L_336_psz14_s6B.pt`          | MIT ([QuanSun/EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP))    |
| `scrfd_10g_bnkps.onnx`, `glintr100.onnx` | InsightFace pretrained models — **non-commercial research only**     |
| `flux-dev:q4` / `flux-dev:q8`            | FLUX.1-dev Non-Commercial License                                    |

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
two qualified tiers.

**"no PuLID bundle resolved"** — run
`mold pull pulid-flux --accept-license insightface-antelopev2` on the machine
that renders, which for a remote run is the server, not your laptop.
