# #1225 — doc deltas staged for consolidation

Every change in this file belongs in a document another agent also edits this
cycle (`CLAUDE.md` / `AGENTS.md`, `README.md`,
`crates/mold-cli/src/skill/SKILL.md`, `THIRD_PARTY_NOTICES.md`). They are
staged here so #1225 does not conflict with its siblings. **Apply them and
delete this file** — nothing reads it, and it is not a permanent record; the
permanent record is `docs/architecture/pulid-face-extraction.md`.

Landed in the tree already, so they are NOT repeated here:
`docs/architecture/pulid-face-extraction.md`, `website/guide/identity.md`,
`crates/mold-inference/testdata/pulid/README.md`,
`crates/mold-inference/testdata/pulid/faces/README.md`,
`changelog.d/pulid-fidelity.md`.

---

## 1. `THIRD_PARTY_NOTICES.md`

New section, to sit immediately after "## InsightFace antelopev2 (face
detection and recognition)" — it is the third upstream in the same pipeline
and the contrast with antelopev2's terms is the point.

```markdown
## facexlib (face parsing)

PuLID masks the aligned face crop before its vision tower sees it, using the
BiSeNet face parser published by
[facexlib](https://github.com/xinntao/facexlib)
(`pulid/pipeline_flux.py:53`, `:161-170`). Mold pulls the released checkpoint
`parsing_bisenet.pth` as part of the `pulid-flux` bundle, from the Hugging Face
mirror [`leonelhs/facexlib`](https://huggingface.co/leonelhs/facexlib) whose
LFS object is byte-identical to facexlib's own GitHub release
(<https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth>,
sha256 `468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567`,
53,289,463 bytes, verified 2026-08-21).

Unlike the InsightFace pretrained models above, facexlib places **no**
non-commercial restriction on its released weights: the project is MIT and its
licence covers the repository as published. `mold pull pulid-flux` therefore
requires no recorded acceptance for this file.

    MIT License

    Copyright (c) 2020 Xintao Wang

Mold ports two facexlib source files rather than vendoring them —
`facexlib/parsing/bisenet.py` and `facexlib/parsing/resnet.py`, as
`crates/mold-inference/src/identity/parsing.rs` — and follows
`facexlib/utils/face_restoration_helper.py` for the 512 crop's template and
border. Every ported function names the upstream file and line range it
follows. No facexlib source file is vendored, and mold ships no Python.
```

Also update the PuLID section's asset list if it enumerates four artifacts;
the bundle is now five.

---

## 2. `README.md`

Wherever the PuLID bundle's artifact count or size appears, it is now **five
artifacts, about 2.2 GB** (was four, about 2.1 GB). If the README lists them,
add:

```markdown
- `parsing_bisenet.pth` — facexlib's BiSeNet face parser (MIT), which masks
  the aligned crop before the vision tower sees it.
```

The licence sentence stays accurate as written: only the two InsightFace
antelopev2 models require `--accept-license insightface-antelopev2`.

---

## 3. `crates/mold-cli/src/skill/SKILL.md`

No flag, endpoint, or env var changed, so the only edit is factual: the
`pulid-flux` bundle is five files, and a user upgrading an existing install
needs a repair pull. Suggested wording wherever the bundle is introduced:

```markdown
`mold pull pulid-flux` fetches five auxiliary artifacts (~2.2 GB). An install
made before mold 0.24 has four; re-running the pull fetches only the missing
face parser.
```

---

## 4. `CLAUDE.md` / `AGENTS.md`

Replace the last two sentences of the **Identity-photo invariant** bullet's
neighbour — the `PuLID face extraction is a CPU candle-onnx path…` bullet —
where it says the 512 crop "carries no BiSeNet mask (both #1225)". Proposed
replacement for that clause, and one new bullet after it.

Edit inside the existing "PuLID face extraction is a CPU `candle-onnx` path"
bullet:

> Deliberate, measured deviations: the 512 crop uses SCRFD's landmarks rather
> than facexlib's RetinaFace, and replaces `cv2.LMEDS` with its least-squares
> refinement (measured max element delta 1.14e-5).

(i.e. drop "carries no BiSeNet mask (both #1225)" — the mask now exists.)

New bullet, to follow it:

> - **PuLID's face crop is masked before the tower sees it, and the parser is
>   a candle port because the op gate said so.** `pipeline_flux.py:161-170` is
>   one policy in `identity/parsing.rs`: facexlib's BiSeNet segments the
>   aligned 512 crop, the background labels `[0, 16, 18, 7, 8, 9, 14, 15]`
>   (hair, 17, is NOT one) become exact white, and the face becomes Rec. 601
>   greyscale — the tower conditions on shape, never colour. The parser's own
>   input takes the **ImageNet** statistics while the tower's takes the
>   **OpenAI CLIP** ones, on the same crop; two normalizations, one image. The
>   final logit upsample and its argmax are FUSED
>   (`bilinear_align_corners_argmax`) because interpolating logits can elect a
>   class that wins at no low-resolution sample, so upsampling the labels
>   instead is a different function rather than an optimization. It is a
>   candle port and not a fourth ONNX graph because the same machine-derived
>   Step-0 gate — `pulid_face_probe gate <graph.onnx>` — refuses a real opset-11
>   export on three counts that are missing evaluators rather than exporter
>   idioms: `MaxPool` pads, `Resize mode=linear`, and `Resize
>   coordinate_transformation_mode=align_corners`. Its weights are facexlib's
>   own pinned release, converted once to safetensors through
>   `encoders/pickle_convert.rs` (formerly `eva_clip_convert.rs`, now
>   describing each conversion as data) — and that release is a **legacy**
>   pre-1.6 `torch.save` archive candle cannot read, so
>   `encoders/legacy_pth.rs` reads the container while candle's own `Stack`
>   still parses every pickle in it. Pinning a stranger's re-save was the
>   alternative and is not one. **Both derived artifacts reach their loaders as
>   `AuthenticatedArtifact` — a descriptor resolved once through a `Dir`,
>   mapped, and pinned on that mapping — never as a `PathBuf`**, because
>   hashing a path and reopening it resolves one name twice and a rename needs
>   only the PARENT's write bit, which the model-storage rule lets a shared
>   root grant; the parser's production code names no path type at all, and a
>   structural test keeps it that way. The PuLID adapter stays a pathname load
>   on purpose: it is a manifest file verified at download, with no fresher
>   authentication to discard. **RetinaFace stays unimplemented on evidence,
>   not on deferral**: with the mask in place, swapping SCRFD's landmarks for
>   facexlib's moves the final identity by at most 2.8e-3 of peak, against the
>   1.2–1.5e-2 the mask itself was worth. Mold runs ONE detection and gives the
>   same face to both crops — upstream takes the largest for ArcFace and the
>   most CENTRAL for the 512 crop, from two detectors, so on a group photograph
>   its two halves describe two people. Numbers, the gate output, and the
>   acceptance pins: `docs/architecture/pulid-face-extraction.md`.

Also, in the same file's PuLID asset sentence, `pulid_assets::PulidPaths` now
resolves **five** files, and `IdentityAssetDigests` carries a fifth digest —
the derived parser — so a different parser is a different identity.
