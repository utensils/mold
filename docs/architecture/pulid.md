# PuLID-FLUX: asset and encoder lifecycle

PuLID conditions a FLUX render on a face. This document records how its four
assets get onto disk, how the two that are models become runnable tensors, and
which invariants hold along the way. It is the reference for the
"PuLID-FLUX: functional" milestone; the FLUX cross-attention integration is not
covered here because it does not exist yet.

## The bundle is four unrelated artifacts

`pulid-flux` is a **hidden, auxiliary, files-only** manifest
(`mold_core::manifest::pulid_manifests`). It is not a checkpoint: it never
appears in a model picker, never becomes a default model, and deliberately
cannot resolve to a `ModelPaths`, because `ModelPaths` requires a generator and
none of these four files is one.

| Component | File | Source | Role |
| --- | --- | --- | --- |
| `IdentityAdapter` | `pulid_flux_v0.9.1.safetensors` | `guozinan/PuLID` | IDFormer + FLUX cross-attention weights |
| `IdentityVisionEncoder` | `EVA02_CLIP_L_336_psz14_s6B.pt` | `QuanSun/EVA-CLIP` | Vision tower — a **conversion input**, never loaded directly |
| `FaceDetector` | `scrfd_10g_bnkps.onnx` | InsightFace antelopev2 mirror | Face detection |
| `FaceRecognizer` | `glintr100.onnx` | InsightFace antelopev2 mirror | ArcFace identity embedding |

All four land under `models/shared/pulid/`.
`mold_core::pulid_assets::pulid_paths` returns `Some` only when **every** one
is complete, so holding a `PulidPaths` means holding a runnable bundle;
`missing_pulid_files` is the repair signal.

The two ONNX files are InsightFace *pretrained models*, which are
non-commercial-research-only even though the InsightFace code is MIT. Mold does
not bundle them and refuses to download them until the user has recorded
acceptance. The other two are Apache-2.0 (PuLID) and MIT (EVA-CLIP) and need no
acceptance.

## The vision tower is converted, never loaded as a pickle

BAAI ships EVA02-CLIP as a torch pickle. **Mold's runtime never reads a
pickle.** `encoders::eva_clip_convert` is the single place one is opened, and
what it produces is ordinary safetensors that everything downstream loads
through a normal `VarBuilder`.

```
EVA02_CLIP_L_336_psz14_s6B.pt          856 MB, f16, 712 tensors (visual + text)
  |  ensure_eva_clip_vision_safetensors
  v
eva02_clip_l_336_vision.safetensors    ~609 MB, f16, 514 tensors, `visual.` stripped
eva02_clip_l_336_vision.json           provenance only, never trusted
```

### What the conversion does

1. **Open no-follow and retain.** `mold_core::secure_file` walks every parent
   component with `O_NOFOLLOW | O_DIRECTORY` and opens the file itself
   `O_NOFOLLOW`, then checks it is a regular file.
2. **Copy the descriptor's bytes into a private file, hashing that same
   stream**, and require the manifest's pinned source SHA-256.
3. **Parse the private copy**, never the caller's pathname.
4. **Keep `visual.*` only**, minus the 48 per-block RoPE buffers, which are
   byte-identical copies of the shared `visual.rope.*` tables (upstream builds
   one `VisionRotaryEmbeddingFast` and gives it to every block). The single
   top-level pair is retained so mold's derived table can be checked against
   the checkpoint's own numbers.
5. **Write atomically and deterministically.** The bytes are built inside the
   private staging directory and then `rename`d into place — same directory,
   therefore same filesystem, therefore a real rename — fsynced first, with the
   parent directory fsynced after. `safetensors`' own `prepare` sorts tensors
   before laying out the buffer, so the byte image does not depend on read
   order, which is what makes the derived pin meaningful.
6. **Record provenance** in a sidecar named after the artifact. Nothing reads
   it back to decide anything.

The conversion is a **re-container, not a cast**: f16 stays f16, so the derived
file is ~609 MB rather than 1.2 GB, and the loading `VarBuilder` picks the
compute dtype.

### Why a private copy, and not an inode fence

candle's `PthTensors` re-opens its file **by pathname** for every tensor ("We
hope that the file has not changed since first reading it",
`pickle.rs:770-772`). Hashing the retained descriptor and then handing candle
the original pathname authenticates nothing: the name can be renamed away and
back between the hash and any of those opens, and the parse would read bytes
the hash never saw.

Re-checking `(device, inode)` on a fresh no-follow open either side of the
parse does not close it either. It samples the pathname at two instants, and
candle re-opens hundreds of times between them.

A `/dev/fd/N` pathname derived from the retained descriptor is the obvious
alternative and does not work: on macOS opening `/dev/fd/N` is `dup(N)`, so
candle's second open would inherit an exhausted offset and read nothing.

So the bytes are copied out of the descriptor into a file only mold can reach,
and hashed on that same stream. The digest and the parse observe identical
bytes by construction. The cost is one transient 856 MB copy, on an
install-time path that is about to write 609 MB anyway.

The staging directory is created with `mkdirat` — which fails rather than
reusing an existing entry — at mode `0o700`, and is removed on every exit path.
That exclusivity is what makes it safe for `serialize_to_file` and the pickle
reader to open paths inside it by name.

### Why the source copy is not staged in the model root

A `0o700` directory is only as private as its parent. Renaming an entry needs
write permission on the **containing** directory, so in a group-writable model
root another member can rename our staging directory away *after* we verified
its contents and drop an unpinned `source.pt` at the pathname `PthTensors`
keeps re-opening. The derived-output pin would still catch the resulting
weights — but only after the pickle parser had already consumed
attacker-chosen bytes, which is the wrong place to find out.

So the source copy is staged under a root chosen by policy, not by convention.
`private_staging_root_candidates` offers `$XDG_RUNTIME_DIR` (a per-user
`0o700` tmpfs on systemd Linux) and then `std::env::temp_dir()` (which honours
`$TMPDIR`: a per-user `0o700` directory on macOS, `/tmp` at sticky `1777` on
most Linux systems). `secure_dir::parent_policy` accepts exactly two shapes,
and **the sticky test has to come first**:

- **Sticky**, owned by us or by root. Sticky restricts rename and unlink to
  each entry's own owner, the directory's owner, or root, so no other
  unprivileged user can touch our entries however open the write bits are.
  Checking ownership first instead rejects root-owned `1777` `/tmp` — which is
  the *only* candidate on a headless box with neither `XDG_RUNTIME_DIR` nor
  `TMPDIR` set — and breaks first-use conversion outright. The owner still
  matters, just less: sticky lets the directory's owner rename entries too, so
  a sticky directory belonging to some other unprivileged user is refused.
- **Not sticky**, owned by us, and not writable by group or other. `0o700` and
  `0o755` both qualify.

If none qualifies the conversion fails closed, naming every candidate and its
reason, rather than silently falling back to the model root.

There is deliberately no `MOLD_*` variable for this: `TMPDIR` is the standard
knob, and a private one would have to be registered in
`ENGINE_SHAPING_VARIABLES` for something that is not engine shaping.

The private root is often a tmpfs sized as a fraction of RAM, and 856 MB is
enough that a default `/tmp` can genuinely be too small, so free space is
checked before the copy starts and the error names `TMPDIR` as the remedy —
rather than surfacing ENOSPC most of the way through.

The **output** staging directory still sits beside the destination, because a
publish must `renameat` within one filesystem. That is safe there: nothing ever
hands its pathname to anyone, and both endpoints of the rename are retained
descriptors.

### Why every publish is a `renameat` between descriptors

Two separate hazards, one helper.

`std::fs::write` **follows** a symlink at its destination, so a link
pre-planted at the sidecar or the weights path would redirect the write into a
file of the attacker's choosing. `rename` replaces the link itself. Every write
in the module — weights and sidecar alike — goes through the one `publish`
helper for that reason, and two tests plant a symlink over a victim file and
assert the victim is untouched.

But a *pathname* rename is still not enough, and this is the subtler half.
Renaming an entry needs write permission on the **parent** directory, not on
the entry, so in the group-writable model root that `CLAUDE.md` explicitly
supports another member can rename our 0o700 staging directory away and drop
their own at the same name — between the hash and the publish. A pathname
rename would then publish their file under our authenticated digest.

So both endpoints are retained directory descriptors and the publish is a
`renameat` through them. Descriptors refer to inodes, so this reaches the
directory we created no matter what its name now points at.
`encoders::secure_dir` holds that primitive; the staging directory is created
with `mkdirat` and then proven to be ours (uid and mode) to close the window
before the `openat`. Afterwards the published file's `(device, inode)` is
re-read through the destination's parent descriptor and compared with the
staged file's, so the artifact the caller receives is provably the one that was
hashed.

`serialize_to_file` insists on a pathname and is the one step that is not
descriptor-bound. That is a liveness concern only: if the staging name were
stolen mid-write the bytes would land in the impostor, and the next step —
which re-opens through the retained staging descriptor — would fail to find
them and error. It cannot silently succeed on someone else's bytes, because the
hash and the publish both go through that descriptor.
`a_stolen_staging_name_cannot_substitute_the_published_bytes` performs exactly
this swap through a test hook placed between the hash and the publish.

### Why the sidecar is not trusted

The sidecar is written by mold. Anything able to tamper with the derived
weights could rewrite it to match, so authenticating the weights against it
would mean authenticating them against the attacker's own claim.

`DERIVED_SHA256` is a compiled-in constant instead. The conversion is
deterministic, so converting the pinned source always produces exactly those
bytes; the weight-gated `conversion_is_deterministic_on_the_pinned_source`
re-derives the constant from the real checkpoint, so a `safetensors` layout
change or a re-uploaded source fails loudly rather than silently shipping
different weights. The sidecar stays as provenance for a human.

### When it runs

`ensure_eva_clip_vision_safetensors(&PulidPaths)` converts **on first use** and
is idempotent on the *bytes*: a derived file is reused only when it hashes to
`DERIVED_SHA256`. Anything else — missing, truncated, tampered with, or
carrying a forged sidecar — reconverts from the pinned source, and a source
that fails its own pin errors rather than falling back to what is on disk. A
fresh conversion that does not reproduce the pin is a hard error, not a silent
reconvert on every later call.

This is deliberately convert-on-first-use rather than a download post-hook.
Admission calls it once it has resolved a complete bundle. Hanging an 856 MB
pickle read off the download path would couple asset installation to model
loading for no benefit, and the reuse check costs one hash of a local file.

## The encoders

```
                       aligned 512x512 face (RGB, [0,1] planar CHW)
                                  |
              eva_clip_preprocess |  bicubic 336, antialiased, a = -0.5
                                  |  then (x - CLIP_MEAN) / CLIP_STD
                                  v
                          [1, 3, 336, 336] f32
                                  |
                 eva_clip_vision  |  EVA02-CLIP-L-14-336, 24 blocks
                                  v
      +---------------------------+----------------------------+
      |                                                        |
  5 x [1, 577, 1024]                                    [1, 768] L2-normalized
  entering blocks 4/8/12/16/20                          visual.head of CLS
      |                                                        |
      |                          arcface [1, 512] ---- cat ----+
      |                                                        |
      |                                                  [1, 1280]
      |                                                        |
      +----------------> flux::pulid_encoder (IDFormer) <------+
                                  |
                                  v
                           [1, 32, 2048]
```

### Preprocessing (`encoders::eva_clip_preprocess`)

Three details are load-bearing and none is visible from the call site:

- The resize runs on the **float** tensor in `[0, 1]`, not on `u8` pixels, and
  is **not clamped**.
- `torchvision.transforms.functional.resize` defaults to `antialias=True` for
  tensors, so the filter support widens with the downscale ratio. The `image`
  crate's `FilterType::CatmullRom` is the same cubic family but a different
  `a`, and is not a substitute.
- That cubic's `a` is **-0.5**, not the -0.75 PyTorch's *non*-antialiased
  bicubic uses. Verified against torchvision directly: -0.5 reproduces it to
  1.5e-5 in f32, -0.75 is off by 6.4e-2.

### Vision tower (`encoders::eva_clip_vision`)

Ported from `eva_clip/eva_vit_model.py`, not adapted from candle's `eva2.rs` —
that model shares the attention and SwiGLU shapes but is a fixed-448 ImageNet
classifier with a different weight layout and no hidden-state taps.

Fixed by `EVA02-CLIP-L-14-336.json`: image 336, patch 14 (24x24 = 576 patches
+ CLS = 577 tokens), width 1024, 24 layers, 16 heads of 64, MLP hidden
`int(1024 * 2.6667)` = **2730** (upstream truncates), LayerNorm eps 1e-6, no
layer scale, pre-norm.

The parts that are easy to get wrong:

- **RoPE is 2D and interleaved.** A 16x16 trained grid is interpolated to
  24x24 purely through the position ramp (`arange(24) / 24 * 16`). Each of the
  two spatial axes contributes 32 of the 64 head columns. `rotate_half` pairs
  **adjacent** lanes, not split halves — the wrong convention type-checks and
  produces a plausible, wrong embedding.
- **RoPE skips CLS.** Only tokens 1.. are rotated.
- **Sub-LN attention.** q/k/v are separate biasless linears; `q_bias` and
  `v_bias` are supplied out of band and **there is no k bias at all**. An
  `inner_attn_ln` sits between the attention output and `proj`.
- **The hidden states are taken entering blocks 4/8/12/16/20**, not leaving
  them (`eva_vit_model.py:526` appends before running the block). A one-block
  shift still yields five correctly shaped tensors.
- **The pooled feature is the normed CLS token**, because
  `global_average_pool` is false for this config, so `fc_norm` does not exist.
  The pipeline then L2-normalizes the `head` projection.

The tower is ~609 MB and follows the crate's **drop-and-reload** rule: build,
encode, drop. Nothing caches it.

### IDFormer (`flux::pulid_encoder`)

Ported from `pulid/encoders_transformer.py`, loaded from the `pulid_encoder.*`
tensors of the adapter. It is a resampler run five times: 32 learned latents
are concatenated with five identity tokens **once**, then each vision scale
drives two `[PerceiverAttention, FeedForward]` layers in order. The identity
tokens stay in the key/value context for every scale, which is why they are
built before the loop.

- `PerceiverAttention`'s key/value input is `cat(context, latents)` — the
  latents attend to themselves as well as the context.
- Its scale is `dim_head^-0.25` applied to **both** q and k before the matmul,
  which is what upstream ships.
- The L2 normalization of the CLIP projection happens in the tower's working
  dtype, not in f32. Upstream's `torch.norm` / `torch.div`
  (`pipeline_flux.py:178-179`) operate on the tensor the tower returned, and
  the tower ran in `weight_dtype` (`:176`) — bfloat16 by default. Widening
  looks like a free accuracy win and is a divergence; it would also leave an
  f32 half in the concatenation with the ArcFace embedding.
- Its softmax is widened to f32 and cast straight back
  (`torch.softmax(weight.float(), -1).type(weight.dtype)`,
  `encoders_transformer.py:114`). This is not a rounding detail: BF16 carries
  8 mantissa bits, and running the softmax in it is measurably 4x further from
  the f32 reference than widening (5.98e-3 against 1.51e-3 of scale). It is
  also upstream-specific — the EVA02 tower's own attention does a plain
  `attn.softmax(dim=-1)` with no cast (`eva_vit_model.py:236`) and mold matches
  that, so the two must not be "harmonized".
- `proj_out` is a bare parameter used as `latents @ proj_out`, stored
  `[1024, 2048]`. It must **not** be transposed the way an `nn.Linear` weight
  would be.
- The activations are `nn.LeakyReLU()` (slope 0.01) in the mapping MLPs and
  exact erf `nn.GELU` in the feed-forwards.

## Parity coverage

`crates/mold-inference/testdata/pulid/README.md` records the goldens, the
tensor mapping in full, the measured errors, and how to run the tests. In
summary: the hermetic tests (shapes, RoPE against the checkpoint's own buffer,
SwiGLU gate order, the resize kernel, the CLIP constants, and every conversion
safety property) run in the ordinary suite; the parity tests against the real
checkpoints are `#[ignore]` behind `MOLD_TEST_PULID_ASSETS`, so CI stays
hermetic.

Measured against upstream, CPU f32: the IDFormer output matches to 1.5e-7 of
its scale, the tower's CLS projection to 1.3e-5 absolute on a unit vector, and
the raw hidden states to ~1e-4 of their own peak magnitude — the last of these
being f32 accumulation amplified by EVA02's extreme activations rather than a
port defect.

## The extraction lifetime (#1223)

The encoders above are the *computation*. What #1223 added is the *lifetime*,
and the lifetime is the part with teeth: an identity must be resolved exactly
once per parent request, before batch fan-out, and every sibling and every
denoise step must reuse that one value.

### Where it runs, and why there

`mold_inference::identity::extraction::extract_identity_embedding` composes the
whole stack — SCRFD, ArcFace, `ensure_eva_clip_vision_safetensors`, the tower,
the IDFormer — into one `[1, 32, 2048]` answer. It is called from exactly one
place: `variant_dependencies::prepare_inputs_for_devices`, **after** the
per-device loop and **before** anything fans out.

That position is chosen, not incidental.

- **After the device loop**, because asset *paths* are per-device (a
  mixed-capacity host can select different encoder variants per GPU) but an
  identity is not. It is a function of the request's own bytes and is identical
  on every device, so resolving it inside the loop would compute the same value
  N times.
- **Before fan-out**, because `PreparedExecutionInputs` is precisely what
  `batch_runtime::submit_child` clones into every `BatchChildExecution`. Storing
  the embedding there makes "one extraction, every sibling" structural rather
  than a convention somebody has to keep.
- **On the CPU**, because at admission the scheduler has not leased a device
  yet, so there is no GPU to run it on. (Until #1227 there was a second reason:
  `candle-onnx` materializes every initializer on `Device::Cpu` and refuses
  anything else. The two face networks are now resident `candle` modules that
  take a device — `identity::scrfd_net`, `identity::arcface_net` — so only the
  admission-ordering reason remains, and `docs/architecture/pulid-perf.md` §3
  designs the phase that would lift it.) That constraint turns into the
  guarantee the issue asked for: extraction's ~1.4 GB peak is allocated and
  released before the job is dispatched, so it *cannot* overlap the T5/CLIP
  encode peak. No scheduled slot and no new typed learned-scheduling phase was
  needed, because the two peaks cannot coexist.

### The frozen value

`mold_core::identity::FrozenIdentityEmbedding` is deliberately plain data:
little-endian `f32` bytes, not a `candle` tensor. That makes it `Clone`, `Eq`,
`Hash`, and device-independent, which is what lets one extraction serve a batch
that fans out across several GPUs. It carries the reference photograph's
SHA-256, the four asset digests, and a fingerprint over all three — the
fingerprint being what a test can compare across siblings to prove nothing
re-extracted. Its `Debug` redacts the values: they are a biometric derivative
and must never reach a log line, an error body, or a probe payload.

At 32 x 2048 f32 it is 256 KiB, which is why carrying it per child costs
nothing worth optimizing.

### The three ways it could have leaked, and what stops each

1. **The scheduler re-prepares dependencies for EVERY pending job**, batch
   children included, and `compose_prepared_generation` replaces
   `pending.prepared_inputs` wholesale. Left alone, a `batch_size = 4` parent
   would extract five times and then discard the parent's value. So the child's
   frozen identity is handed to preparation through
   `DependencyPreparationContext::frozen_identity`, which short-circuits the
   extraction entirely, and `compose_prepared_generation` carries it across as a
   backstop for a preparer that ignored the context.
2. **Placement preview** runs the same preparation path read-only. It must never
   spend seconds and 1.4 GB on a probe, and it does not need to: identity
   changes the *memory demand*, which `memory_preflight` charges from the
   request, not from the extracted value. `DependencyMaterializationPolicy::
   ExistingOnly` therefore extracts nothing.
3. **The engine is cached across requests and the identity is not.** The GPU
   worker calls `InferenceEngine::install_identity_embedding` before EVERY
   dispatch — with `Some` when the plan froze one and `None` otherwise. Passing
   `None` is not an optimization, it is the clear; without it a cached
   `FluxEngine` would condition the next print on the previous print's face. The
   default trait implementation REFUSES a populated embedding rather than
   dropping it, because a family that cannot condition on a face must not render
   a print that silently has none; only `FluxEngine` overrides it.

Forced-local installs at the same point
(`mold-cli/src/commands/local_engine.rs::build_local_engine_from_plan`) from the
same `prepare_local_execution_inputs` path, which is what makes local/remote
parity structural rather than reviewed.

### Weight zero

An explicit `id_weight` of 0 is inert at every layer:
`identity_dependencies::request_needs_identity_assets` plans no assets,
`identity_extraction::resolve_identity_embedding` returns `None` without
counting an extraction, `memory_preflight` charges nothing, and
`flux::identity::identity_request` returns `None` so the denoise loop takes the
exact code path an unconditioned build takes. The falsification test from
`tmp/sdcpp/docs/pulid.md` — same seed, `--id-weight 0`, byte-identical output —
is therefore structural rather than numerical.

### Memory

Measured, replacing #1220's declared 2.3 GB placeholder:

| Where | Bytes | What |
| --- | --- | --- |
| Device | 839,270,400 | 20 cross-attention modules at FLUX.1's geometry, f16/bf16 |
| Device | ~410,000,000 | cross-attention activation headroom at 1024x1024 |
| **Device total** | **1,250,000,000** | `IDENTITY_VRAM_OVERHEAD_BYTES` |
| Host | 16,923,827 | `scrfd_10g_bnkps.onnx` |
| Host | 260,665,334 | `glintr100.onnx` |
| Host | 856,461,210 | the EVA02-CLIP `.pt` (609 MB derived, plus f32 activations) |

The old placeholder charged the tower and the IDFormer to VRAM on the assumption
that the extractor would run on the generation device. It does not, and
over-charging VRAM by ~1 GB parks renders a card could actually run.
`ComponentRole::IdentityVisionEncoder` is consequently `is_host_only`, alongside
the two ONNX graphs; `IdentityAdapter` deliberately is not, because it is the
one identity artifact that IS device-resident for the whole denoise.

### Removal

`mold rm pulid-flux` now deletes the derived
`eva02_clip_l_336_vision.safetensors` and its sidecar as well as the four
manifest files. Both names live in `mold_core::pulid_assets`
(`DERIVED_VISION_FILENAME`, `DERIVED_VISION_SIDECAR_FILENAME`) rather than in
`encoders::eva_clip_convert`, because removal is in `mold-core` and cannot see
`mold-inference`; the converter reads them from there so the two can never name
different files.

## Not yet built

- facexlib's BiSeNet background mask, which upstream applies before the vision
  tower (`PuLID/pulid/pipeline_flux.py:145-170`). Issue #1225.
- Fusing several reference photographs into one stronger identity.
- Identity on tiers other than `flux-dev:q4` / `:q8`, alongside a LoRA, or
  alongside img2img. All three are refused by name at the request contract; a
  milestone-2 qualification pass owns them.
