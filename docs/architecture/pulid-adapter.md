# PuLID cross-attention adapter

How mold injects a face identity into a FLUX render, and why the four
transformer variants share exactly one injection policy.

`docs/architecture/pulid.md` covers the asset bundle and the identity
*encoders*. This document covers what happens to the resulting embedding inside
the denoise loop.

## What the adapter is

PuLID-FLUX does not fine-tune the transformer. It adds a stack of twenty small
cross-attention modules between the transformer's blocks. Each takes the
current image tokens as the query and a 32-token, 2048-wide identity embedding
as key and value, and adds its scaled output back onto the image stream:

```text
img = img + id_weight * ca[ca_idx](id_embeds, img)
```

One module — `PerceiverAttentionCA` — is a LayerNorm on each input, a
`3072 → 2048` query projection, a `2048 → 4096` key/value projection split in
half, 16 heads of 128, softmax attention, and a `2048 → 3072` output
projection. It carries no bias anywhere. mold's port is
`crates/mold-inference/src/flux/pulid.rs`, from
`ToTheBeginning/PuLID` `pulid/encoders_transformer.py:29-72`.

Attention goes through `crate::attention`, not a private matmul, so the Metal
auto-chunking and `MOLD_ATTN` policy that bound every other FLUX-family score
matrix apply here too.

### Where the modules attach

| stream | rule | blocks | modules |
| --- | --- | --- | --- |
| double | after every 2nd block | 0, 2, … 18 | 0–9 |
| single | after every 4th block | 0, 4, … 36 | 10–19 |

`ceil(depth / 2) + ceil(depth_single / 4)` — 10 + 10 = 20 for FLUX.1's 19
double and 38 single blocks. `ca_idx` advances **only on an injection** and
never resets between the two loops, which is why the single-stream modules
continue the double-stream numbering rather than starting over.

The count the transformer shape implies is checked against the count the
checkpoint actually carries at load time. A PuLID v1.1 file, which renames the
prefix from `pulid_ca.*` to `id_adapter_attn_layers.*`, is therefore refused by
name instead of loading zero modules and rendering an unconditioned image.

By the time the single-stream loop runs, text and image tokens share one
tensor. Only the image slice — `xs[:, txt_len..]` — is conditioned; the text
prefix is spliced back bit-for-bit (`flux/model.py:141-146`).

## One policy, four variants

`FluxTransformer` has four arms and they reach their block loops three
different ways:

| variant | how the hook gets in |
| --- | --- |
| `BF16` | the candle fork's `Flux::forward_with_hook` |
| `Quantized` | the same, on `quantized_model::Flux` |
| `QuantizedBypass` | mold's own loop in `quantized_transformer.rs` |
| `Offloaded` | mold's own loop in `offload.rs` |

All four drive the same `PulidBlockHook`, which implements the fork's
`BlockHook` trait (`candle_transformers::models::flux::BlockHook`, added in
utensils/candle#6). The upstream models take `&dyn BlockHook`; mold's two take
`Option<&dyn BlockHook>`, because a `None` hook has to execute the untouched
loop rather than a no-op implementation of one.

`crates/mold-inference/src/flux/pulid_variants.rs` renders a synthetic 4-double
/ 8-single transformer through all four arms and asserts they agree
numerically, which is what makes the per-variant claims below claims about one
model rather than four.

## The zero-weight rule

An effective `id_weight` of 0, and every step before `id_start_step`, must
render **bit-identically** to a request that never mentioned identity. This is
stable-diffusion.cpp's own falsification test
(`tmp/sdcpp/docs/pulid.md`, "Verification"): if A (no PuLID) and B (PuLID at
weight 0) differ, the injection is computing something it should not.

mold satisfies it structurally rather than numerically. The gate is
`PulidRuntime::hook_for_step`, which yields `None` in both cases, and every
arm of the denoise dispatch answers `None` by calling the variant's ordinary
`forward` — the exact call a build with no identity request makes. There is no
"multiply by zero and add", no allocation, and no second transformer load: the
comparison holds on the *same* route.

The same rule reaches further up the stack. `mold_core::identity` refuses an
identity request on a build without the `pulid` feature, and
`mold-server/src/identity_dependencies.rs` plans no assets, starts no download,
and charges no memory for a request whose effective weight is 0. The engine's
`identity_request` predicate in `flux/identity.rs` is the same test, so the
three layers cannot disagree about what "asks for a face" means.

## Residency

The adapter is **always fully resident**. It is not streamed with the offloaded
blocks, even on the path that exists to fit a 24 GB transformer into 2–4 GB.
It is ~1.14 GB of fp16, and every one of its twenty modules is touched on every
step: streaming it would pay twenty host→device copies per step to reclaim
memory the block schedule has already accounted for.

`flux/identity.rs` owns the lifecycle, following the same drop-and-reload
discipline as the text encoders. The rule is that **the adapter is resident
only while the transformer is**:

- Loaded lazily on the first request that conditions on a face.
- Kept across subsequent conditioned requests that agree on device, dtype, and
  transformer shape.
- Dropped the moment a request does not condition on a face.
- Dropped at the end of any render the transformer did not survive — the
  sequential path always, the eager path unless `MOLD_FLUX_KEEP_TRANSFORMER`
  or the quantized stay-hot path kept it. `FluxEngine::generate` applies this
  once, at the end, so neither in-render drop site carries its own copy of the
  rule and neither can forget it.
- **Released by `unload()`.** This one is not an optimisation. `ModelCache`
  parks an engine by calling `unload()`, setting the entry's `vram_bytes` to
  0, and keeping the engine alive in the cache. An adapter that survived that
  would be ~1.7 GB of device memory nothing accounts for, and the next model
  switch would size its preflight against a number wrong by the whole adapter —
  failing admission or OOMing. `Drop` needs no help: the `Arc` dies with the
  engine.

`FluxEngine::identity_resident_bytes()` is the accounting-visible form of all
of the above. The `InferenceEngine` trait has no resident-bytes method to fold
it into, so it is exposed on the engine directly and pinned by hermetic tests
that assert an `unload()` leaves it at zero.

The dtype is the transformer's *working* dtype, not its weight dtype: the
quantized paths run their state tensors in f32, so the adapter is loaded in f32
there and in bf16 on the dense path. A dtype or shape change rebuilds rather
than silently feeding the transformer the wrong precision.

## The embedding seam

`IdentityEmbedding` is a `[1, 32, 2048]` newtype, validated once at the
boundary instead of at each of the twenty injection sites. It is the seam the
face extractor plugs into: the detector → ArcFace → EVA-CLIP → IDFormer stack
produces one of these and installs it with
`FluxEngine::set_identity_embedding`, and nothing downstream changes.

Until that lands, a face-conditioned request with no embedding installed is an
**explicit error**. Accept-and-ignore is not an option for the same reason it
is not one at the request contract: the print would come back without the face
and nothing would say so.

For bring-up and for comparison against the oracle, `IdentityEmbedding` also
loads from a safetensors file and from stable-diffusion.cpp's `.pulidembd`
gguf container (a single `pulid_id` tensor), so the identical identity can be
pushed through both implementations.

## References

- `ToTheBeginning/PuLID` — the reference implementation.
  `pulid/encoders_transformer.py:29-72` for the module,
  `flux/model.py:83-85` for the intervals, `flux/model.py:116-147` for the two
  injection sites.
- stable-diffusion.cpp (`tmp/sdcpp`) — the executable oracle. It consumes the
  same `pulid_flux_v0.9.1.safetensors` mold downloads and runs on Metal, so a
  disagreement can be settled by running both rather than by reading both.
  `src/model/adapter/pulid.hpp`,
  `src/model/diffusion/flux.hpp:993-1004` (module count) and `:1120-1161`
  (injection), `docs/pulid.md`.
- `crates/mold-inference/testdata/pulid/README.md` — the captured goldens, the
  weight hash they came from, and the measured agreement behind each tolerance.
