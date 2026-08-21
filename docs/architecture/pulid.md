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
eva02_clip_l_336_vision.json           { source_sha256, derived_sha256, derived_filename }
```

### What the conversion does

1. **Open no-follow and retain.** `mold_core::secure_file` walks every parent
   component with `O_NOFOLLOW | O_DIRECTORY` and opens the file itself
   `O_NOFOLLOW`, then checks it is a regular file.
2. **Hash through the descriptor**, not through the pathname, and require the
   manifest's pinned source SHA-256.
3. **Fence the pathname.** candle's `PthTensors` re-opens by pathname for every
   tensor, so `(device, inode)` is re-checked on a fresh no-follow open both
   before and after candle reads. The retained descriptor keeps the inode
   allocated for that whole window, so it cannot be recycled: a swap-and-swap
   back would have to hand back an inode number that is provably still ours.

   A `/dev/fd/N` pathname derived from the retained descriptor would be the
   obvious alternative and does not work — on macOS opening `/dev/fd/N` is
   `dup(N)`, so candle's second open would inherit an exhausted offset and read
   nothing.
4. **Keep `visual.*` only**, minus the 48 per-block RoPE buffers, which are
   byte-identical copies of the shared `visual.rope.*` tables (upstream builds
   one `VisionRotaryEmbeddingFast` and gives it to every block). The single
   top-level pair is retained so mold's derived table can be checked against
   the checkpoint's own numbers.
5. **Write atomically and deterministically.** A sibling `.staging` file — same
   directory, therefore same filesystem, therefore a real `rename` — is
   fsynced, then renamed, then the parent directory is fsynced. `safetensors`'
   own `prepare` sorts tensors before laying out the buffer, so the byte image
   does not depend on read order and the recorded digest is meaningful. A
   `.staging` file left by an interrupted run was never published and is
   replaced outright, never resumed.
6. **Record the derived SHA-256** in a sidecar named after the artifact.

The conversion is a **re-container, not a cast**: f16 stays f16, so the derived
file is ~609 MB rather than 1.2 GB, and the loading `VarBuilder` picks the
compute dtype.

### When it runs

`ensure_eva_clip_vision_safetensors(&PulidPaths)` converts **on first use** and
is idempotent — a derived file whose SHA-256 matches its sidecar is accepted
as-is; a missing file, a missing sidecar, or a mismatched digest reconverts,
because a half-written or hand-edited artifact must never be loaded as weights.

This is deliberately convert-on-first-use rather than a download post-hook.
Admission calls it once it has resolved a complete bundle. Hanging an 856 MB
pickle read off the download path would couple asset installation to model
loading for no benefit, and the idempotence check makes the cost a single
`stat` plus one hash on every later call.

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

## Not yet built

- FLUX cross-attention integration (`pulid_ca.*` in the adapter) and the
  generation path.
- Face detection and alignment (SCRFD + ArcFace); issue #1222.
- Any admission-side call to `ensure_eva_clip_vision_safetensors`. The function
  exists and is idempotent; nothing invokes it yet.
- Removal of the derived artifact. `mold rm pulid-flux` deletes what
  `pulid_storage_paths` lists, which is the four manifest files — the derived
  `eva02_clip_l_336_vision.safetensors` and its sidecar are not among them and
  would be left behind as a ~609 MB orphan. Harmless today because nothing
  produces them yet, but this must be fixed in the same change that wires the
  conversion into admission.
