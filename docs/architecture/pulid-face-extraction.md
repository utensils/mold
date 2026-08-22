# PuLID face extraction — runtime decision, parity, and deferrals

Issue [#1222](https://github.com/utensils/mold/issues/1222). This is the
decision record for how mold detects a face and produces the identity
embedding PuLID-FLUX conditions on, in pure Rust.

Scope: SCRFD detection, five-point alignment, the ArcFace embedding, and the
512×512 crop the EVA vision tower takes in
[#1229](https://github.com/utensils/mold/issues/1229). The IDFormer, the
adapter, and the FLUX attention hooks are other issues'.

Code: `crates/mold-inference/src/identity/`, behind the `pulid` feature.

> **Step 0's runtime choice was superseded by
> [#1227](https://github.com/utensils/mold/issues/1227).** `candle-onnx` no
> longer evaluates either graph at request time: `identity/scrfd_net.rs` and
> `identity/arcface_net.rs` run them as resident `candle` modules built once
> from the same SHA-pinned files, and `candle_onnx::simple_eval` survives only
> as the parity oracle those modules are tested against. Everything else on this
> page still stands — the op inventory, the pinned digests, the alignment and
> warp goldens, the tolerances, and the deferrals — and the measured latencies
> below remain the baselines #1227 compares against.
> `docs/architecture/pulid-perf.md` §4 records what that swap actually bought,
> including the part of Step 0's reasoning it falsified.

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
two. The probe enforces this rather than trusting the reader: `--runs 0` is
refused outright (a percentile of no samples used to report 0 ms and PASS), and
anything below the protocol prints an ADVISORY banner instead of GATE.

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
decoding, against the digest of the exact bytes it decodes.

That last clause is load-bearing and cost a review round. Retaining the
descriptor is not enough: hashing it and then reading it is **two** passes over
the file, and on shared storage an in-place write landing between them
authenticates one set of bytes and executes another. The private
`AuthenticatedBytes` type makes that unrepresentable — a single `read_to_end`,
a digest of the buffer it produced, and no way to obtain a digest and a buffer
that did not come from the same call. `mold_core::secure_file::sha256_open_file`
is deliberately unused here for exactly that reason: it takes a `&File`, which
necessarily leaves the bytes to a second read. A structural test pins the
module to one read.

That single read is also **bounded**, which is what makes it safe to perform at
all. A digest can only be computed from bytes already in memory, so an
unbounded `read_to_end` on an oversized or sparse replacement file exhausts
memory before the mismatch it was about to find can be reported — the process
dies instead of refusing the file. The manifest pins each artifact's exact
`size_bytes` alongside its digest, so `PinnedArtifact` carries both from one
lookup (a caller cannot pair one component's digest with another's length), the
retained descriptor is `fstat`ed first, and an unexpected length is refused as
`ArtifactSizeError` before a byte is read. The read is then capped at the
expected length **+ 1**, so a file that grows between the stat and the read
overshoots and is refused rather than silently truncated to a prefix that
happens to parse. Loads without a pin — the inventory and benchmark tools,
which take arbitrary paths by design — get the same treatment at
`UNPINNED_MAX_BYTES` (1 GiB, four times the largest graph mold loads). The pin comes from `mold_core::pulid_assets::pulid_manifest()` via
`onnx_graph::pinned_sha256`, never a second copy in this crate.

The downloader's `.sha256-verified` marker is not a substitute: it records that
the bytes were correct *when they landed*, and says nothing about the bytes
now. A marker sitting beside a since-modified model is exactly the state
someone with write access to the models directory would leave behind — and
these two graphs are executed code in every sense that matters.
`IdentityExtractor::from_paths` always supplies the manifest's pins and has no
unverified variant; tools that inspect arbitrary graphs pass `None` and get no
extractor out of it.

This is complementary to placement-time verification — the download path's pin
check, and the per-file dependency verification landing in
[#1242](https://github.com/utensils/mold/pull/1242) — and must not be folded
into it. Placement-time checking proves a file as it is materialized and
accepts an existing `.sha256-verified` marker without rehashing: right for
materialization, and exactly the assumption the load-time check exists to stop
relying on. One proves what landed; the other proves what runs.

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
then masks that crop's background with BiSeNet (`:161-170`). #1222 scoped both
out; #1225 measured both and implemented the mask. Mold warps the 512 crop
from the **same SCRFD landmarks** and applies the **same mask**.

The remaining divergence — one detector rather than two — is named and
measured, not silent. See "#1225", below.

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

## #1225 — the two deferrals, resolved

#1222 named two divergences from upstream rather than hiding them: the BiSeNet
background mask, and the second detector upstream fits its 512 crop with.
[#1225](https://github.com/utensils/mold/issues/1225) measured both end to end
and closed them differently. **The mask is now implemented. RetinaFace is
not, on the evidence below.**

### What the two cost, measured

Upstream's own pipeline, run in a scratch venv on the four committed
portraits, produces the `[1, 32, 2048]` value FLUX is conditioned on. Every
row below is mold's crop and mold's landmarks fed through THAT pipeline, so
the only variable is the thing named — the port itself contributes nothing to
these numbers.

Reference = upstream exactly: RetinaFace's centre-face landmarks, masked.

| variant | rel. error of peak | cosine vs upstream |
| --- | --- | --- |
| SCRFD landmarks, **unmasked** (mold before #1225) | 1.2–1.5e-2 | 0.99916–0.99945 |
| RetinaFace landmarks, **unmasked** | 1.3–1.5e-2 | 0.99918–0.99950 |
| SCRFD landmarks, **masked** (mold after #1225) | 0.9–2.8e-3 | 0.99996–0.99998 |

Read the first two rows together: swapping the detector while unmasked changes
almost nothing, so the detector was never what the gap was made of. The mask
is, and applying it closes 80–93% of it.

The third row is also the answer on RetinaFace. With the mask in place, the
detector choice is worth at most 2.8e-3 of peak — an order of magnitude below
what the mask was worth, and below the level at which a second detector, a
second 109 MB download, and a second set of five landmarks could be justified.
The geometric difference is real but small: RetinaFace's landmarks sit 1.8–2.7
px from SCRFD's on average (5.5 px worst), which moves the 512 crop by 6–13
mean LSB.

Then mold's actual port, against upstream's masked pipeline on the same
photographs — this is the acceptance pin, `the_identity_matches_upstream_end_to_end`:

| face | relative error of peak |
| --- | --- |
| Frank Rubio | 7.0e-7 |
| Kayla Barron | 4.3e-7 |
| Mae Jemison | 1.0e-5 |
| Raja Chari | 5.4e-7 |

So mold now sits ~100x closer to upstream than the RetinaFace divergence it
declines to close, and ~1000x closer than it did unmasked.

### Face selection: one detection, largest, both crops

Upstream selects **differently for its two halves**: the largest face for
ArcFace (`pipeline_flux.py:127-129`) and the most CENTRAL one for the 512
crop, because facexlib is called with `only_center_face=True` (`:145`, and
`face_restoration_helper.py:152-163`). On a group photograph those are
different people, and the identity it builds is half of one face and half of
another. That is a defect, not a contract.

Mold runs one detection, takes the largest by area, hands the same face to
both crops, and reports the choice through `x-mold-request-warning` when there
was more than one. `one_detection_serves_both_crops_even_when_centre_and_largest_disagree`
constructs the disagreeing case explicitly so the divergence stays deliberate.
On a single-face photograph — every fixture, and the overwhelming majority of
real reference photographs — the two rules agree by construction, which is why
the table above is measurable at all.

### Step 0, again: why the parser is a candle port

The face stack's other two models are ONNX run through
`candle_onnx::simple_eval`, so the first thing #1225 tried was the same for
BiSeNet. The gate is the one this document already describes, run over a real
`torch.onnx.export(..., opset_version=11)` of `facexlib.parsing.bisenet.BiSeNet`
through the probe's new `gate` subcommand:

```text
$ pulid_face_probe gate bisenet_opset11.onnx
=== bisenet_opset11.onnx  sha256=176d6ce2…9bfd9  52598436 bytes
    opset: {"": 11}
    …
    x1    Resize   coordinate_transformation_mode=align_corners cubic_coeff_a=-0.75 mode=linear nearest_mode=floor
    x3    Resize   coordinate_transformation_mode=asymmetric    cubic_coeff_a=-0.75 mode=nearest nearest_mode=floor
    x1    MaxPool  ceil_mode=0 dilations=1,1 kernel_shape=3,3 pads=1,1,1,1 strides=2,2
    OP GATE: FAIL
      - candle-onnx's `MaxPool` rejects pads=1,1,1,1 (only all-zero pads (eval.rs:472-476, :507-511))
      - candle-onnx's `Resize` rejects mode=linear (only `nearest` (eval.rs:2325))
      - candle-onnx's `Resize` rejects coordinate_transformation_mode=align_corners (only `asymmetric` (eval.rs:2333))
```

Reproduce it with `crates/mold-inference/testdata/pulid/export_bisenet_onnx.py`.

None of the three is an exporter idiom mold could normalize away the way the
empty-`scales` rewrite handles SCRFD's. The `MaxPool` padding is ResNet18's
stem (`resnet.py:54`) and both `Resize` restrictions are the final logit
upsample (`bisenet.py:135`), so all three are load-bearing and all three are
missing *evaluator support*. Closing them is three separate candle-onnx
changes — a padded pooling arm, a bilinear kernel, and the `align_corners`
coordinate transform — weighed against porting a network that is 191 tensors
of `Conv → BatchNorm → ReLU`.

**Decision: port it.** `identity/parsing.rs`. The weights then arrive the way
the EVA02-CLIP tower's do — a pinned torch pickle, converted once to
safetensors, loaded through an ordinary `VarBuilder` — so mold's runtime still
never reads a pickle.

One thing that decision dragged in. facexlib published `parsing_bisenet.pth`
in 2020, in the **legacy** flat `torch.save` container, and
`candle_core::pickle::PthTensors` reads only the modern zip one. There is no
newer release of those weights, so `encoders/legacy_pth.rs` reads the
container: five sequential pickles, then `i64` element count plus raw
elements per storage key, in the sorted order the fifth pickle lists. Only the
container is new — every pickle inside it is parsed by candle's own `Stack`,
so the opcode surface mold trusts is unchanged. The alternative was pinning a
stranger's re-save, which moves the provenance from "facexlib published these
bytes" to "someone says these are facexlib's weights".

The one detail worth knowing before touching that file: the archive opens with
a pickle of a **ten-byte** magic number, and candle's `Long1` arm accumulates
into an `i64` with `<< (i * 8)`, which panics in a debug build at `i = 8`. The
reader compares the fixed 21-byte preamble as bytes instead, which both avoids
that and is stricter than reading the value.

### The derived artifacts reach their loaders as mappings, not as names

The two ONNX graphs are authenticated by `AuthenticatedBytes` — one bounded
read, a digest of that buffer, no way to obtain a digest and a buffer from
different calls. The two DERIVED artifacts (the vision tower and the parser)
need the same property and cannot use the same mechanism, because 609 MB is
not a thing to hold twice.

`pickle_convert::AuthenticatedArtifact` is that property for a file: the final
component is resolved exactly once, through a `Dir` descriptor
(`openat` + `O_NOFOLLOW`), the descriptor is mapped, and the pin is checked
against **that mapping**. `ensure_eva_clip_vision_safetensors` and
`ensure_bisenet_parser_safetensors` hand out that value and never a `PathBuf`,
and both loaders — `EvaClipVisionTower::from_authenticated` and
`BiSeNetParser::from_authenticated` — read it through
`VarBuilder::from_slice_safetensors`.

The gap this closes is specific and was found in review. Hashing a path and
then reopening it for `from_mmaped_safetensors` resolves one name twice, and
**renaming an entry needs write permission on the containing directory, not on
the file** — exactly the grant `CLAUDE.md`'s model-storage rule says a shared
model root may legitimately hand out. A second member could therefore let the
digest check pass and then swap the file the loader opened. Mapping the
retained descriptor makes that unrepresentable: after the rename the mapping
still refers to the inode that was hashed.

`a_renamed_artifact_cannot_be_handed_to_a_loader` performs that rename and
asserts both halves — a handle already held still reads what it verified, and a
fresh open of the same name is refused on its digest.
`the_parser_cannot_be_loaded_from_a_bare_path` is the structural guard, in the
style of `onnx_graph`'s: the parser's production code never names a path type
at all, so no constructor can accept one.

One load in the extraction is deliberately still by pathname — the PuLID
adapter, in `compose_identity_tokens`. It is a MANIFEST file, verified when it
was downloaded, and mold does not re-hash 1.1 GB per request; there is no
fresher authentication there to discard by reopening a name. That distinction
is what the two comments at those call sites say, and it is why the structural
test scopes itself rather than banning path loads outright.

### The mask itself

`pipeline_flux.py:161-170`, and every clause matters:

* The parser's input is normalized with the **ImageNet** statistics (`:163`),
  while the tower's is normalized with the **OpenAI CLIP** ones (`:174`), on
  the same crop. Two normalizations, one image.
* Background labels are `[0, 16, 18, 7, 8, 9, 14, 15]` — hair (17) is NOT one
  of them — replaced with exact white, not with facexlib's border grey. The
  border grey fills pixels the warp had no source for; white fills pixels the
  parser called not-face.
* The face is converted to **greyscale** (Rec. 601 luma, `:113-116`), not
  passed through. A port that left the face in colour still produces a
  plausible-looking image.
* The upsample-then-argmax order is load-bearing and the two are fused in
  `bilinear_align_corners_argmax` so it cannot be taken the other way round:
  interpolating logits can elect a class that wins at no low-resolution
  sample, so upsampling the labels instead is a different function.

### Parity, and the cost

Against facexlib on the four committed portraits, from the same 512 crop:

| check | tolerance | measured worst |
| --- | --- | --- |
| per-class pixel count, as a fraction of the crop | 1e-4 | < 5e-7 (every one of 262 144 labels agrees) |
| probed labels differing (of 512) | 1 | 0 |
| masked crop, mean abs channel delta | 0.02 / 255 | 0.0001 |
| masked crop, fraction of channels differing at all | 1e-3 | 9.2e-5 |
| masked, resized, CLIP-normalized tensor, max abs | 1e-3 | 4.3e-5 |
| final `[1, 32, 2048]` identity, relative to peak | 5e-5 | 1.0e-5 |

Latency, re-measured with the parser as its own stage — the 2.0 s budget is
per extraction, not per model. halcyon, release build, 5 warmups / 20 runs,
under a load average of **19.5** (three agents building; the idle numbers
above were taken on a quiet machine, and this is the honest working figure).

Measured on the **pre-#1227** stack, so the SCRFD and ArcFace rows are the
`candle-onnx` baselines the note at the top of this page describes rather than
what ships today. The BiSeNet row is unaffected either way — the parser never
went through `candle-onnx` — and it is the number #1225 contributes:

| stage | p50 | p95 | max |
| --- | --- | --- | --- |
| SCRFD | 185.1 ms | 276.8 ms | 285.1 ms |
| ArcFace | 235.2 ms | 307.7 ms | 395.1 ms |
| BiSeNet | 156.3 ms | 244.5 ms | 247.8 ms |
| **per image** | **576.1 ms** | **810.0 ms** | 855.9 ms |

Peak RSS 1179 MiB; decode + construct + convert 799.9 ms, once, at load.
**PASS**, with 2.5x headroom on a busy machine. The parser is the cheapest of
the three stages, which is what a 53 MB ResNet18 should be.

Since #1227 the parser is timed by the extraction itself rather than by a
probe-local loop: `ComposeStage::Parse` brackets materializing the parser,
segmenting the crop, and applying the mask, and `pulid_face_probe bench --full`
reports it as `bisenet` beside `eva-build`, `eva-forward`, `idformer-build` and
`idformer-fwd`. It is charged apart from `EvaBuild` deliberately — it is a
second network, not the tower's setup — and it is included in that command's
`per-image` total.

### Still out of scope

* **facexlib's RetinaFace detector.** Declined on the measurement above, not
  deferred again. If a future change makes the residual matter — a
  higher-fidelity milestone, or a checkpoint more sensitive to crop geometry —
  the number to beat is 2.8e-3 of peak.
* **BiSeNet's auxiliary heads** (`conv_out16`, `conv_out32`). They are training
  outputs; PuLID reads `[0]` only. They are retained in the derived
  safetensors, which is a faithful re-container, and never constructed.
* A rendered-identity ArcFace cosine between the reference photograph and a
  generated face. That is a GPU measurement of the whole pipeline including
  the adapter, and belongs with the milestone's UAT rather than with the
  extraction; the `[1, 32, 2048]` pin above is what bounds this issue's own
  contribution to it.

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
| `identity/parsing.rs` | the BiSeNet port, the label upsample, and PuLID's mask |
| `encoders/legacy_pth.rs` | the pre-1.6 `torch.save` container |
| `encoders/pickle_convert.rs` | both pinned pickles, converted once to safetensors, and `AuthenticatedArtifact` |
| `img_utils::decode_oriented_srgb` | the crate's one EXIF/ICC decode, shared with LTX-2 |
| `bin/pulid_face_probe.rs` | the inventory and benchmark tool this record cites |
| `tests/pulid_face_parity.rs` | hermetic + weight-gated parity |
| `crates/mold-inference/testdata/pulid/` | inventory fixture, faces, goldens, capture scripts |
