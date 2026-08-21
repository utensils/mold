# Staged documentation for #1223 (PuLID end to end)

Three files this PR deliberately does NOT edit — `CLAUDE.md` (and its `AGENTS.md`
symlink), `crates/mold-cli/src/skill/SKILL.md`, and `README.md` — because
several PRs collided on them the same day. The text they need is here; apply it
in one consolidated docs commit on `main` after this PR merges, then **delete
this file**.

Everything else this issue required is already in its final home: the website
guide (`website/guide/identity.md` plus its nav entry), the changelog fragment
(`changelog.d/pulid-flux.md`), the architecture reference
(`docs/architecture/pulid.md`), and the UAT record
(`docs/architecture/pulid-uat.md`).

---

## 1. `CLAUDE.md` — append to the existing PuLID invariant bullets

> **Identity is extracted once per request, and its lifetime is the whole
> invariant.** When the effective `id_weight` is above zero, `mold_core::identity`
> owns the contract (qualified tiers `flux-dev:q4`/`:q8`, the `0.0..=3.0` weight,
> the bounded decode limits, and `IDENTITY_RUNTIME_READY`, which is now `true` so
> a `pulid` build both advertises and executes). Admission resolves the reference
> photograph EXACTLY ONCE, on the CPU, in
> `variant_dependencies::prepare_inputs_for_devices` — after the per-device loop,
> before batch fan-out — through `mold_inference::identity::extraction`, and
> stores the immutable `FrozenIdentityEmbedding` (plain little-endian `f32` plus
> source SHA, the four asset digests, and a fingerprint over all three) on
> `PreparedExecutionInputs`. That struct is what the batch parent clones into
> every child, so "one extraction per parent, reused by every sibling on every
> device" is structural. The scheduler re-prepares dependencies for EVERY pending
> job including batch children, so a child is handed its parent's frozen value
> through `DependencyPreparationContext::frozen_identity` and skips extraction
> entirely; `compose_prepared_generation` carries it across as a backstop, and
> placement preview (`ExistingOnly`) extracts nothing at all. The GPU worker
> installs it through `InferenceEngine::install_identity_embedding` before EVERY
> dispatch and clears it otherwise — the engine is cached and the identity is
> not, so a stale embedding would condition the next print on the previous
> person; the default trait impl REFUSES a populated embedding rather than
> dropping it, and only `FluxEngine` overrides. Forced-local installs at the same
> point in `local_engine::build_local_engine_from_plan`, from the same
> `prepare_local_execution_inputs` path, which is what makes local/remote parity
> structural. Weight zero performs no pull, decode, load, or extraction and must
> stay byte-identical to an unconditioned render (the falsification case from
> `tmp/sdcpp/docs/pulid.md`). Because extraction completes and releases its
> ~1.4 GB before a device is leased, it cannot overlap the T5/CLIP encode peak —
> a stronger guarantee than a scheduled slot, and the reason no new typed
> learned-scheduling phase was added. Memory is measured, not declared:
> `IDENTITY_VRAM_OVERHEAD_BYTES` is 1.25 GB (the adapter's own resident
> arithmetic plus bounded cross-attention activations), and the detector,
> recognizer, and EVA02-CLIP tower are all `is_host_only` because they never
> reach the generation device. The CLI reads `--id-image` through
> `commands::identity`: `open_regular_file_no_follow`, bounded from the
> descriptor before allocating, read from that exact descriptor, then
> `validate_id_image_bytes` — all before any request bytes exist; `id_image_name`
> is the file's own name and never the client path. The `pulid` feature ships ON
> in every release recipe (`flake.nix`, `release.yml`, both source PKGBUILDs,
> which now need `protobuf`/`protoc` for `candle-onnx`'s `prost-build`); the
> feature only decides whether the binary LINKS the stack, and the licence gate
> is the recorded InsightFace acceptance enforced at download time. `mold rm
> pulid-flux` deletes the derived EVA safetensors and its sidecar, whose names
> live in `mold_core::pulid_assets` because removal cannot see `mold-inference`.

## 2. `crates/mold-cli/src/skill/SKILL.md` — replace the identity section

The current section (around "Face identity (PuLID)") says identity is *"Not
executable yet on any build"*. Replace it with:

> ### Face identity (PuLID-FLUX)
>
> Keep one person's face across arbitrary prompts. Qualified for `flux-dev:q4`
> and `flux-dev:q8`, on CUDA and Metal, in every official release build.
>
> ```bash
> # One-time setup: the bundle is licence-gated and will not download without this
> mold pull pulid-flux --accept-license insightface-antelopev2
>
> mold run flux-dev:q4 "an astronaut in a diner" --id-image face.jpg
> mold run flux-dev:q4 "a Renaissance portrait" --id-image face.jpg --id-weight 0.6
> mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg --id-start-step 4
> ```
>
> | Flag | Default | Range | Notes |
> | --- | --- | --- | --- |
> | `--id-image <path>` | — | PNG/JPEG, ≤16 MiB, ≤8192 px/axis, ≤32 MP | Reference photograph |
> | `--id-weight <f>` | `1.0` | `0.0`–`3.0` | `0` is completely inert: nothing pulled, loaded, or extracted, and the render is byte-identical to no identity at all |
> | `--id-start-step <n>` | `0` | `< --steps` | First denoise step identity applies from |
>
> Refused, by name rather than silently: any other model, a LoRA alongside an
> identity, img2img alongside an identity, and a build without the `pulid`
> feature. Works over `$MOLD_HOST` and with `--local`; the bundle and the licence
> acceptance must be on the machine that RENDERS, which for a remote run is the
> server. Saved metadata records the photograph's file name, its SHA-256, and the
> applied weight and start step — never the photograph. `mold rm pulid-flux`
> removes the bundle and the derived vision tower. Full guide:
> `website/guide/identity.md`.

## 3. `README.md`

**Feature bullet** — add beside the other generation features:

> - **Face identity (PuLID-FLUX)** — keep one person's face across arbitrary
>   prompts with `--id-image`, on `flux-dev:q4` and `flux-dev:q8`. Pure Rust:
>   SCRFD, ArcFace, EVA02-CLIP, and the IDFormer feed twenty cross-attention
>   modules inside the FLUX transformer.

**Model/qualification note** — wherever supported models are tabulated, mark
`flux-dev:q4` and `flux-dev:q8` as the identity-qualified tiers.

**License note** — add to the existing third-party licensing paragraph:

> Face identity additionally downloads two InsightFace **pretrained models**
> (`scrfd_10g_bnkps`, `glintr100`), which are licensed for **non-commercial
> research purposes only** — the InsightFace *code* is MIT, the *weights* are
> not. Mold ships neither and refuses to download them until you record
> acceptance with `mold pull pulid-flux --accept-license insightface-antelopev2`;
> `mold licenses` lists what has been accepted. The PuLID adapter is Apache-2.0
> and the EVA02-CLIP tower is MIT.

## 4. `THIRD_PARTY_NOTICES.md`

Checked: it already carries PuLID (Apache-2.0), EVA-CLIP (MIT), and the
InsightFace pretrained-model restriction, added by #1220. No change is needed
for this PR — the artifact set did not grow. The only new on-disk artifact is
`eva02_clip_l_336_vision.safetensors`, which is mold's own conversion of the
already-listed EVA-CLIP checkpoint and carries its licence.
