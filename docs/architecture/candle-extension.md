# Candle extension boundary

There must be only one `candle_core::Tensor` type in a build graph.
`Tensor` and `Error` are nominal types: a second copy of Candle under any
package name is a second, unrelated type universe, and every call site that
hands a tensor across the boundary stops compiling.

Mold satisfies that rule by pinning **every** Candle crate — `candle-core`,
`candle-nn`, `candle-transformers`, `candle-flash-attn`, and `candle-onnx` — to
one revision of the `utensils/candle` fork. Its published packages are renamed
(`candle-core-mold`, `candle-nn-mold`, `candle-transformers-mold`); the
ecosystem crates in the same tree keep their upstream names but depend on the
renamed core, so the whole set is one identity.

## Ownership rule

Put a change in `mold-ai-candle` when it can be expressed with Candle's public
API. Patch Candle only when the change must alter a backend, storage primitive,
or private implementation. Prefer an upstream contribution whenever the change
is generally useful.

| Capability                                      | Owner                      | Reason                                                                                                                                                                                                                                                                               |
| ----------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LTX-Video model, sampler, and VAE               | `mold-ai-candle`           | Mold consumes and evolves this model; it uses public tensor and NN APIs.                                                                                                                                                                                                             |
| Stable Diffusion component configs              | `mold-ai-candle`           | These are Mold's single-file loading policy, not framework accessors.                                                                                                                                                                                                                |
| GGUF builder from in-memory `QTensor`s          | `mold-ai-candle`           | A small application adapter over public `QTensor` and `VarBuilder` APIs.                                                                                                                                                                                                             |
| Compact quantize-then-transfer helper           | `mold-ai-candle`           | Public APIs can move quantized bytes without retaining an extra dense GPU tensor.                                                                                                                                                                                                    |
| Native CUDA FP8 dtype and matmul                | Candle compatibility patch | Requires backend dtype, storage, CUDA kernel, and cast changes.                                                                                                                                                                                                                      |
| Stable Diffusion VAE mode and bounded attention | Candle compatibility patch | Changes Candle model internals; both are candidates for focused upstream PRs.                                                                                                                                                                                                        |
| Wuerstchen timestep dtype correction            | Candle compatibility patch | Changes Candle model internals and is independently upstreamable.                                                                                                                                                                                                                    |
| GGUF 5-D tensor cap raise                       | Candle compatibility patch | `GGUF_MAX_TENSOR_DIMS` is a private reader const; the video-model GGUF ecosystem (city96 converter lineage: Wan patch embeddings are Conv3d weights) legitimately emits 5-D tensors that n_dims-as-u32 permits. One-const change with a round-trip test; independently upstreamable. |

Focused upstream submissions track each exit independently:

- VAE deterministic mode: [huggingface/candle#3841](https://github.com/huggingface/candle/pull/3841)
- Stable Diffusion attention bounds: [huggingface/candle#3842](https://github.com/huggingface/candle/pull/3842)
- Wuerstchen timestep dtype: [huggingface/candle#3843](https://github.com/huggingface/candle/pull/3843)
- Native CUDA FP8 matmul: [huggingface/candle#3844](https://github.com/huggingface/candle/pull/3844)
- Stable AArch64 FP16 vector storage: [huggingface/candle#3845](https://github.com/huggingface/candle/pull/3845)

The former Metal quantized-matmul override is deliberately absent. Upstream's
fused kernels already accumulate in F32, upstream fixed the valid BF16 GGML
mapping, and Mold had no focused regression proving that dequantizing every
matmul was correct. Reintroducing it requires a reproducible failing test and a
targeted upstreamable fix.

## Dependency shape

Every Candle crate in the workspace and desktop Cargo roots is a direct git
dependency on ONE `utensils/candle` revision. `[patch.crates-io]` cannot be
used for this: a patch must keep the patched package's name, and the fork
publishes the renamed `candle-*-mold` identities that carry the merged Metal
kernels LTX-2.5 needs (#1393). The root `Cargo.toml` still patches `cudarc`,
which is a same-name fork and therefore patchable.

`mold-ai-inference` imports application extensions from `mold-ai-candle` and
ordinary framework models from `candle-transformers`. Feature forwarding is:

```text
mold-ai-inference cuda/metal
  -> mold-ai-candle cuda/metal
  -> candle-core + candle-nn + candle-transformers cuda/metal
```

**Moving the pin means moving all of it.** A Candle crate left on crates.io
depends on the upstream-named `candle-core` and drags a second identity into
the graph. That is exactly how #1399 broke every CUDA build for four
consecutive `main` merges: #1393 moved `candle-core`/`-nn`/`-transformers` to
the fork and left `crates/mold-candle`'s `candle-flash-attn` at
`version = "=0.11.0"`, so `candle_flash_attn::flash_attn` was being handed
`candle-core-mold` tensors it could not accept.

`scripts/tests/candle-single-identity.sh` enforces this on the release-contract
CI route (it runs on pull requests; the `--features flash-attn` compile job does
not). It asserts that every declared Candle dependency names the same fork URL
and revision, that none names a `branch`, `tag`, or `path` source (a `version`
key beside `git`+`rev` is permitted, because cargo ignores it while the git
source wins locally and a published manifest requires one), and — the assertion
that actually holds, because a manifest audit cannot see a transitive consumer —
that no `Cargo.lock` resolves any `candle-*` package from a registry, a path, or
a second revision.

Do not add a second Candle source, a duplicate git revision, or a local copy of
a Candle backend. `cargo tree -d` should not report Candle packages from
multiple sources, and Flash Attention must compile without a private cfg flag.

## Distribution boundary

Mold no longer publishes workspace crates to crates.io. Releases use GitHub
artifacts, Nix/FlakeHub, Docker, AUR, and source builds, which preserve the pinned
backend. Existing registry versions are historical, not a current installation
channel.

Cargo strips Git sources when publishing and requires registry version
requirements instead. Adding those requirements alone is insufficient: the
fork revision may not be published, and upstream registry packages such as
`candle-flash-attn` use a different Candle package identity. Substituting them
would break the shared Tensor type and discard backend fixes.

The obsolete registry publisher and tag-triggered publish job are removed.
`scripts/tests/crates-publish-contract.sh` guards against restoring registry
publication or installation instructions while preserving supported release
jobs. release-plz retains version PRs and tags with registry publishing disabled.

## Compatibility revision lifecycle

The current compatibility source is revision
`13022a3b3d8a0f5545e439e058e9e41a290d3da4` of `utensils/candle`, which contains
the renamed `candle-core-mold` / `candle-nn-mold` / `candle-transformers-mold`
packages every Mold cargo root pins. No manifest names a branch — the identity
script rejects a `branch =` source outright — so moving the compatibility source
means moving that revision. Its commits are intentionally independent by
concern. For each remaining patch:

1. Add or retain a focused regression test.
2. Open a narrowly scoped upstream PR.
3. Move the pin to the fork revision that contains the upstream commit.
4. Remove the local commit and verify CPU plus the affected GPU backend.
5. Once no compatibility commits remain, resolve directly from an upstream
   revision or, if the rename is retired too, from crates.io.

Updating the pin requires editing every manifest that declares a Candle crate
and refreshing the root and desktop lockfiles together. A force-push that
rewrites a revision a Mold release consumed is prohibited; publish a new
revision instead.

## Upgrade checklist

- Confirm one Candle source with `cargo tree -d` and `cargo tree -i candle-core`.
- Run `cargo test -p mold-ai-candle`.
- Run inference checks for the default backend and the platform GPU backend.
- Compile the release feature combination, including `flash-attn` on CUDA CI.
- Exercise one GGUF LoRA path and one Stable Diffusion single-file path on
  hardware before release.
- Record upstream PRs and any remaining patch exit criteria here.
