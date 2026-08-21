# PuLID face extraction — runtime decision, parity, and deferrals

Issue [#1222](https://github.com/utensils/mold/issues/1222). This is the
decision record for how mold detects a face and produces the identity
embedding PuLID-FLUX conditions on, in pure Rust.

Scope: SCRFD detection, five-point alignment, the ArcFace embedding, and the
512×512 crop the EVA vision tower takes in
[#1229](https://github.com/utensils/mold/issues/1229). The IDFormer, the
adapter, and the FLUX attention hooks are other issues'.

Code: `crates/mold-inference/src/identity/`, behind the `pulid` feature.

---

## Step 0 — the runtime decision

**Decision: `candle-onnx` runs both graphs, unchanged, with one mold-side graph
normalization.** No hand-ported iResNet100, no cached evaluator, no `ort`, no
Python, no native ONNX runtime.

The gate had two halves and both had to pass.

### Half 1 — the op gate

Every distinct `(op_type, attributes)` in both pinned `ModelProto`s, checked
against the ops `candle_onnx::simple_eval` implements
(`candle-onnx/src/eval.rs`, the `node.op_type.as_str()` match). The inventory
is machine-derived from the model bytes; the capability set is transcribed into
`identity/onnx_inventory.rs::CANDLE_ONNX_SUPPORTED_OPS` because mold cannot
introspect another crate's `match`.

`scrfd_10g_bnkps.onnx` (opset 11, 16,923,827 bytes,
`5838f7fe…b5b91`):

| op | count | notable attributes |
| --- | --- | --- |
| Conv | 58 | `group=1`, symmetric `pads`, `strides` ∈ {1, 2} |
| Relu | 36 | |
| Add | 16 | |
| Transpose / Reshape | 9 / 9 | `perm=2,3,0,1` |
| Shape / Gather / Unsqueeze / Slice / Concat | 6 / 4 / 4 / 2 / 2 | |
| Sigmoid | 3 | |
| Mul | 3 | |
| AveragePool | 3 | **`ceil_mode=1`** |
| Resize | 2 | `mode=nearest`, `nearest_mode=floor`, `coordinate_transformation_mode=asymmetric` |
| MaxPool | 1 | `ceil_mode=0` |

`glintr100.onnx` (opset 11, 260,665,334 bytes, `4ab1d643…4cdf`):

| op | count | notable attributes |
| --- | --- | --- |
| Conv | 103 | `group=1` |
| BatchNormalization | 51 | `epsilon=1e-5`, inference mode |
| PRelu | 50 | |
| Add | 49 | |
| Flatten | 1 | `axis=1` |
| Gemm | 1 | `alpha=1`, `beta=1`, `transB=1` |

**Result: pass.** Every op has an evaluator arm, and every attribute value
falls inside what that arm accepts. `Resize` in particular lands exactly on
candle-onnx's only supported combination — nearest/floor/asymmetric
(`eval.rs:2325-2344`).

Two things the gate recorded rather than waved through:

* **`AveragePool.ceil_mode=1` is silently ignored.** `eval.rs:491-525` reads
  `kernel_shape`, `pads`, `strides`, `dilations`, and `auto_pad`, and never
  looks at `ceil_mode`; `Tensor::avg_pool2d_with_stride` floors. Flooring and
  ceiling agree exactly when every pooled extent divides the stride, and mold
  pins the detector input at 640×640, where SCRFD's five halvings all land on
  even extents (640 → 320 → 160 → 80 → 40 → 20). The precondition is a
  function (`pinned_input_makes_pooling_exact`) and a test, not a comment. A
  future detector size that is not a multiple of 32 would break this and must
  not be added without revisiting it.
* **`Resize`'s empty `scales` is a real candle-onnx defect** — see below.

### The one candle-onnx defect, and the mold-side normalization

The gate passed on ops but the first evaluation failed:

```
SCRFD graph evaluation failed
Scales and sizes cannot both be set for Resize operation
```

Both `Resize` nodes read `['382', '392', '392', '399']`. `392` is a **single
zero-length `float` initializer** standing in for both the absent `roi` and the
absent `scales` — the idiom ONNX's own `Resize` spec prescribes ("if `sizes` is
needed, set `scales` to an empty tensor") and the one PyTorch and MXNet
exporters emit.

`candle-onnx` tests presence on the input **name**, not the tensor:

```rust
// candle-onnx/src/eval.rs:2291-2296
let scales = if node.input.len() > 2 && !node.input[2].is_empty() {
    Some(get(&node.input[2])?)
} else {
    None
};
```

`node.input[2]` is the `String` `"392"`, which is not empty, so an absent
`scales` reads as supplied, `sizes` is supplied too, and `:2312-2314` bails.

**Upstream fix (a separate candle-fork PR, deliberately not folded into this
work):** treat a zero-element tensor as absent in both the `scales` and `sizes`
presence checks — e.g. `.filter(|t| t.elem_count() > 0)` after the `get`.
Exact location: `candle-onnx/src/eval.rs`, the `"Resize"` arm, lines 2291-2301.
That is the whole change; no new op and no new attribute support is required.

**Until it lands**, `identity/onnx_graph.rs::normalize_empty_optional_resize_inputs`
rewrites `Resize`'s optional `roi`/`scales`/`sizes` inputs to the empty-string
form when they name a zero-element initializer. It is scoped to `Resize` only,
because that is where the ONNX spec explicitly gives an empty tensor the
meaning "unspecified"; a blanket rewrite of every empty input would be wrong.
Five unit tests pin the scope, including that the required data input is never
rewritten.

### Half 2 — the latency gate

Budget: **p95 ≤ 2.0 s per image**, warm, over 20 measured runs after 5 warmups,
on named halcyon and plato configurations. Warm is the only meaningful number
here: `simple_eval` re-materializes every initializer and retains every
intermediate on each call (`eval.rs:191-232`, `:249-257`), so the second call
costs what the thousandth does and there is nothing to amortize.

Measured with `cargo build --release -p mold-ai-inference --features
dev-bins,pulid --bin pulid_face_probe`, then `pulid_face_probe bench <dir>`.
"per image" is one SCRFD detection at 640×640 plus one ArcFace embedding. The
probe **exits non-zero on a failed gate**, like its `inventory` subcommand, so
CI, a bisect, or a `&&` chain can act on the verdict.

Use the full 5-warmup / 20-run protocol. A short run is not a measurement: at
`--runs 2` the p95 is simply the larger of two samples, and one scheduler stall
on a busy machine reports a blown budget. The same host that measured 588.4 ms
p95 under a load average of 83 with the real protocol reported 2533.7 ms from
two.

**halcyon** — Apple M4 Max, 16 cores, 48 GiB, macOS (aarch64-darwin):

| stage | p50 | p95 | max |
| --- | --- | --- | --- |
| SCRFD | 174.3 ms | 180.8 ms | 181.3 ms |
| ArcFace | 222.2 ms | 234.9 ms | 253.6 ms |
| **per image** | **395.3 ms** | **415.7 ms** | 432.8 ms |

Peak RSS 1113 MiB; decode + construct 585 ms (once, at load).

An earlier run on the same host taken while it was under a load average of 34
(parallel agent builds) measured p95 = 1565.8 ms — still inside budget, and
recorded here because it is the realistic worst case for a developer machine
doing other work.

**plato** — 128-core x86_64 NixOS, 1.5 TiB RAM, 4× L40S (idle, load average
4.07), CPU evaluation:

| stage | p50 | p95 | max |
| --- | --- | --- | --- |
| SCRFD | 502.8 ms | 536.4 ms | 543.5 ms |
| ArcFace | 906.4 ms | 1050.7 ms | 1095.5 ms |
| **per image** | **1411.9 ms** | **1574.5 ms** | 1639.0 ms |

Peak RSS 849 MiB; decode + construct 572 ms.

**Result: pass on both.** halcyon has 4.8× headroom, plato 1.27×. plato is the
tighter configuration despite far more cores — candle's CPU backend does not
scale past a modest thread count on these convolution shapes, so core count
buys almost nothing. Its narrow margin is the number to watch: a checkpoint
change, a candle regression, or a busier host could push it over, and the
`pulid_face_probe bench` subcommand exists to re-measure rather than re-argue.

### What was NOT needed

* **A hand-ported iResNet100 in candle.** It remains the fallback if the
  latency gate ever fails on a target configuration; the port is mechanical
  (`recognition/arcface_torch/backbones/iresnet.py`: Conv → BN → PReLU → Add,
  Flatten → FC → BN, no global pooling) and would load weights straight from
  the ONNX initializers, which `candle-onnx` can still parse.
* **A cached evaluator** that materializes initializers once. Attractive
  (~260 MB of re-copied `glintr100` weights per call), but it is a candle-onnx
  change, and the budget is met without it.
* **GPU execution.** `candle-onnx` is CPU-only by construction — `get_tensor`
  places every initializer on `Device::Cpu` (`eval.rs:191-232`) and the `Gemm`
  arm builds its `alpha`/`beta` tensors there too (`:1794-1798`).
  `IdentityExtractor::load` rejects a non-CPU device explicitly rather than
  demoting silently.

### Build-system consequence

`candle-onnx`'s build script drives `prost-build`, which shells out to
`protoc`. `pkgs.protobuf` was added to the flake devshell. **No release recipe
enables `pulid` yet**; whichever issue first ships it must also add `protobuf`
to the crane `nativeBuildInputs` in `flake.nix` and to any CI job that builds
with the feature.

---

## The pipeline

```
image bytes
  → EXIF-oriented, ICC-corrected decode              img_utils.rs
  → letterbox to 640x640, top-left, zero fill      scrfd.py:459-470
  → blob: (x - 127.5) / 128.0, planar RGB          scrfd.py:164
  → candle-onnx simple_eval  ->  9 tensors
  → anchor decode, strides 8/16/32, 2 anchors      scrfd.py:158-225
  → score >= 0.5, NMS IoU <= 0.4 (inclusive extents)  scrfd.py:352-380
  → largest bbox wins (+ warning if several)       pipeline_flux.py:127-129
  ├─ Umeyama fit -> arcface_dst 112, warp          face_align.py:6-29
  │    → blob: (x - 127.5) / 127.5                 arcface_onnx.py:37-40, 78-81
  │    → glintr100  ->  raw 512-d embedding
  └─ Umeyama fit -> facexlib FFHQ 512, warp        face_restoration_helper.py:73-74, 242-259
       → 512x512 crop for #1229
```

### The photograph is righted before the detector sees it

`extract` decodes through `img_utils::decode_oriented_srgb`, the same path
LTX-2 still conditioning uses (it lived in `ltx2/preprocess.rs` until #1222
needed it and was lifted into `img_utils`, which now re-exports to LTX-2 under
its original name). A phone photograph carries its rotation in an EXIF tag
rather than in the pixels; handing SCRFD the raw buffer gives it a sideways
face, which it either misses outright or locates in a frame every crop then
inherits. Upstream orients too — `cv2.imread` applies EXIF orientation unless
`IMREAD_IGNORE_ORIENTATION` is set.

One thing that decode does and `cv2.imread` does not: convert an embedded ICC
profile to sRGB. Deliberate. An untagged sRGB image — every parity fixture —
takes the identical path, so parity is unaffected, and a tagged one would
otherwise have its colors misread into the embedding.
`raja-chari-official-portrait.exif6.jpg` is the regression fixture: the same
portrait stored 1000x800 with orientation 6. It detects landmarks within
0.111 px of the upright original's, at ArcFace cosine 0.999245.

### Both graphs are authenticated on every load

`load_onnx_model` takes the expected SHA-256 and refuses a mismatch *before*
decoding, against the digest of the same retained descriptor the bytes are read
from. The pin comes from `mold_core::pulid_assets::pulid_manifest()` via
`onnx_graph::pinned_sha256`, never a second copy in this crate.

The downloader's `.sha256-verified` marker is not a substitute: it records that
the bytes were correct *when they landed*, and says nothing about the bytes
now. A marker sitting beside a since-modified model is exactly the state
someone with write access to the models directory would leave behind — and
these two graphs are executed code in every sense that matters.
`IdentityExtractor::from_paths` always supplies the manifest's pins and has no
unverified variant; tools that inspect arbitrary graphs pass `None` and get no
extractor out of it.

### The embedding is RAW, not L2-normalized

#1222's summary line says "512-d, L2-normalized". Upstream says otherwise, and
upstream is what the IDFormer was trained against:
`insightface/model_zoo/arcface_onnx.py:63-66` stores the raw network output on
the face, `PuLID/pulid/pipeline_flux.py:130` reads exactly that
(`face_info['embedding']`), and `:156-158` moves it to the device **without**
normalizing — in visible contrast to the EVA branch at `:177-178`, which *is*
normalized before the two halves are concatenated.

`ArcFaceEmbedding::raw` is therefore the value that travels to #1229.
`l2_normalized()` is offered beside it for cosine comparisons. Measured raw
norms on the fixture set are 16.3–21.9, so the difference is not cosmetic:
handing #1229 a unit vector would silently change the IDFormer's input scale.

### The two crops share one detection

PuLID does not: it takes the ArcFace embedding from InsightFace's SCRFD and the
512 crop's landmarks from **facexlib's RetinaFace**
(`pipeline_flux.py:145-147`, `get_face_landmarks_5(only_center_face=True)`),
then masks that crop's background with BiSeNet (`:161-170`). #1222 scopes both
out. Mold warps the 512 crop from the **same SCRFD landmarks** and applies **no
mask**.

This is a named, measured divergence, not a silent one. See "Deferred", below.

---

## Ports, and the one deliberate deviation

Every function cites the upstream file and line range it follows. Two choices
are worth stating outright.

### Pixel operations are OpenCV's, not a Rust image crate's

`identity/warp.rs` implements `cv2.resize(INTER_LINEAR)` and
`cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT)` directly.
`image::imageops::resize` scales its triangle-filter support with the ratio, so
a 4× downscale averages an 8-tap window where OpenCV takes exactly two taps at
`src = (dst + 0.5) * scale - 0.5`. The difference is many LSBs on a real
photograph, and every landmark the detector reports is a function of those
pixels. `imageproc` was not added: it is a candle-workspace dependency, not a
mold one, and two functions do not justify a new tree in the Nix vendor set.

mold evaluates both in `f64` and rounds once; OpenCV uses 5-bit fixed-point
interpolation weights. The measured cost of that difference is in the parity
table below.

### facexlib's LMEDS fit is replaced by its least-squares refinement

`face_restoration_helper.py:242-244` fits the 512 template with
`cv2.estimateAffinePartial2D(landmark, template, method=cv2.LMEDS)`. LMEDS is a
**randomized** robust estimator: OpenCV draws random 2-point subsets, scores
each by median squared residual, then refines the best model by least squares
over its inliers. A faithful port would be neither deterministic nor stable
across OpenCV versions.

mold performs the refinement step alone — the least-squares 4-DOF similarity,
which is exactly Umeyama's closed form, since Umeyama's solution *is* the LS
optimum for a similarity. facexlib's own comment says it reached for LMEDS "for
the equivalence to skimage transform" (`face_utils.py:167`), i.e. it wanted the
LS fit. With five landmarks from one detector there is no outlier for the
robust step to reject.

Measured against real `cv2.LMEDS` output on the fixture set: **max element-wise
difference 1.14e-5** on a matrix whose translation terms are hundreds of
pixels. The deviation is free.

---

## Parity

Captured by `crates/mold-inference/testdata/pulid/capture_goldens.py` from the SHA-pinned models
(onnxruntime 1.29.0, OpenCV 5.0.0, insightface 1.0.1), on four public-domain
portraits — see `crates/mold-inference/testdata/pulid/faces/README.md` for licenses and sources.

**Hermetic** (no weights; the committed landmarks are enough):

| check | tolerance | measured worst |
| --- | --- | --- |
| `m112` vs skimage `SimilarityTransform` | 1e-4 | 1.74e-5 |
| `m512` vs skimage | 1e-4 | 1.74e-5 |
| `m512` vs `cv2.estimateAffinePartial2D(LMEDS)` | 1e-4 | 1.14e-5 |
| 112 crop vs `cv2.warpAffine` — mean abs | 0.6 / 255 | 0.229 |
| 112 crop vs `cv2.warpAffine` — p99.9 abs | 4 LSB | 2 |
| 512 crop vs facexlib's warp — mean abs | 0.6 / 255 | 0.190 |
| 512 crop vs facexlib's warp — p99.9 abs | 4 LSB | 2 |

**Weight-gated** (`MOLD_TEST_PULID_ASSETS`):

| check | tolerance | measured worst |
| --- | --- | --- |
| landmark position vs InsightFace | 1.0 px | **0.232 px** |
| bbox corner | 2.0 px | inside |
| detection score | 0.02 | inside |
| ArcFace cosine vs InsightFace | ≥ 0.99 | **0.999384** |

Per face: Frank Rubio 0.999384, Kayla Barron 0.999773, Mae Jemison 0.999871,
Raja Chari 0.999774.

Running them:

```bash
# hermetic
nix develop -c cargo test -p mold-ai-inference --features pulid

# with the antelopev2 weights present
MOLD_TEST_PULID_ASSETS=/path/to/antelopev2 \
  nix develop -c cargo test --release -p mold-ai-inference --features pulid \
  --test pulid_face_parity -- --ignored --nocapture
```

The weights are InsightFace pretrained models, **non-commercial research use
only**. They are never committed, and mold refuses to download them without a
recorded license acceptance (`THIRD_PARTY_NOTICES.md`).

---

## Deferred, and what decides it

Owned by [#1225](https://github.com/utensils/mold/issues/1225), named here so
neither is silent:

1. **facexlib's RetinaFace detector** for the 512 crop's landmarks. Mold uses
   SCRFD's. Also note upstream selects the **centre** face for that crop
   (`only_center_face=True`) while selecting the **largest** for ArcFace; mold
   uses the largest for both, from one detection.
2. **The BiSeNet background mask** applied to the 512 crop before the EVA
   tower (`pipeline_flux.py:161-170`: background labels are replaced with
   white and the face is converted to grey). Mold passes the unmasked colour
   crop.

#1222's fidelity gate decides whether #1225 moves into this milestone: run the
full mold extractor (this issue plus #1229) against the pinned Python pipeline
and record the final IDFormer-output error plus a rendered-identity check
(ArcFace cosine between the reference photo and a generated face). That gate
cannot be evaluated until #1229 lands the EVA tower and the IDFormer.

Also not done here: the CLI's `--id-image` path handling
([#1223](https://github.com/utensils/mold/issues/1223)).

---

## Files

| path | what |
| --- | --- |
| `identity/mod.rs` | `IdentityExtractor`, `IdentityFeatures`, oriented decode, face selection, typed errors |
| `identity/onnx_inventory.rs` | Step 0's op/attribute gate |
| `identity/onnx_graph.rs` | descriptor-fenced load, pin verification, `Resize` normalization |
| `identity/scrfd.rs` | letterbox blob, anchor decode, NMS, detection |
| `identity/align.rs` | both templates, Umeyama fit, residuals |
| `identity/arcface.rs` | `norm_crop`, blob, 512-d embedding |
| `identity/warp.rs` | OpenCV-convention resize and affine warp |
| `img_utils::decode_oriented_srgb` | the crate's one EXIF/ICC decode, shared with LTX-2 |
| `bin/pulid_face_probe.rs` | the inventory and benchmark tool this record cites |
| `tests/pulid_face_parity.rs` | hermetic + weight-gated parity |
| `crates/mold-inference/testdata/pulid/` | inventory fixture, faces, goldens, capture scripts |
