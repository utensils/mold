# Z-Image Metal VAE decode qualification

## Status — 2026-09-05

Candle candidate `bedc287458e0d890dd6ed1c298c99e991e066fe1` passes the isolated
whole-1024 decoder comparison and visual check below. The Mold integration
passes targeted crate validation; the proactive latent-120 cap and existing Candle
pin remain in place. This record does not qualify 1152-pixel decoding or Wan.

The candidate bounds Metal im2col and GEMM-result workspaces rather than
changing the VAE's image tiling or receptive field. Spatial partitioning follows
Candle's `candle-core/src/cpu_backend/conv2d.rs::conv2d_tiled`: every output
patch reads the original full input. Small convolutions retain the existing
path. Large convolutions reclaim temporary results after bounded batches of
tiles, using the precise-current completion wait. Global indices use `size_t`
and each dispatch fits the kernel's `uint` grid.

Command completion checks errors after waiting and checks earlier producers
before accepting a successful readback. Qualification-only allocation limits,
terminal diagnostics and the compact-bucket experiment are excluded from the
candidate commit. The original silent-corruption mechanism was not reproduced
under the bounded guards, so this is not confirmation of a residency/driver
root cause. `ResidencySet::commit` does not return a discarded `Result`.

## Raw comparisons

Real VAE weights were used with byte-identical deterministic latent inputs.
Output comparisons precede clipping or PNG conversion. The acceptance gate is
relative L2 below `1e-3`, globally and in every quadrant, with finite output.

| Whole latent / output side | Candle Metal vs CPU relative L2 |
|---|---:|
| 96 / 768 | 2.35308e-6 |
| 104 / 832 | 2.90430e-6 |
| 112 / 896 | 4.03264e-6 |
| 120 / 960 | 4.49140e-6 |
| 128 / 1024 | 5.23749e-6 |

The weights-free convolution ladder at latent 128 also passes, at `2.68094e-6`.
The independent sd.cpp whole-128 decoder passes against CPU at `3.48184e-4`;
its Metal GEMM uses mixed precision, unlike the Candle F32 comparison.

The VAE checkpoint is
`shared/z-image/vae/diffusion_pytorch_model.safetensors`, SHA256
`f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`.
The executable reference is sd.cpp
`6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`, ggml
`e20c3a14aa70ee84ca58499814206dd08d8026bc`, with the full decode from
`src/model/vae/auto_encoder_kl.hpp` and its diffusion-to-VAE normalization.
No reference implementation ships in Mold.

## Real-latent visual check

The reference generated a red ceramic teapot on a wooden table beside a
window, using the existing Z-Image Turbo Q4_K and Qwen3 Q8_0 checkpoints,
seed 42, eight steps, guidance 1 and a 1024×1024 canvas. Disk parameter
residency released unused weights between phases; mmap and VAE tiling were
not enabled. The final denoiser latent was captured before VAE normalization.
Its SHA256 is
`0371db5e232e8ba6b07b9a584fe90ecfe8cc0bcbb3cb52ddd705feb22e3291fb`.

Candle Metal versus CPU decoding of that exact latent gives relative L2
`3.58728e-6`, maximum absolute error `3.47793e-5`, and passing quadrants.
All three full-resolution images were opened and visually inspected. The
reference, CPU and Metal images show the same coherent teapot, lid, handle,
spout, window lighting and wood grain. No seams, missing regions, stripe
fields or visible decode corruption were observed.

The images are retained in the user's external Mold library:

```text
/Volumes/ExternalStorage/mold2/output/1040-20260905-real128-reference.png
/Volumes/ExternalStorage/mold2/output/1040-20260905-real128-cpu.png
/Volumes/ExternalStorage/mold2/output/1040-20260905-real128-metal.png
```

Each carries embedded metadata and qualification provenance, plus a JSON
sidecar. This is an isolated decoder comparison using a reference-generated
latent, not a claim that a full Mold render has been qualified.

## Safety evidence and limitations

- Every Metal decoder run used a native 12 GiB allocation ceiling and required
  28 GiB free-plus-inactive memory before launch. The host guard aborts below
  12 GiB available, on non-normal kernel pressure, on swap growth above
  256 MiB, or at its deadline. Runs were serialized with an atomic lock.
- Latent 144 / 1152 pixels passed CPU decoding but was refused on Metal:
  11,397,234,688 allocated bytes plus a 2,147,483,648-byte request exceeded
  the native ceiling. It is neither a corruption result nor qualified.
- The first full reference attempt stopped at 11.923 GiB available before
  producing a latent or image. Recorded samples showed normal pressure and no swap growth.
  The single authorized disk-backed retry succeeded, with minimum available
  memory 17.09 GiB, normal pressure and no swap growth in recorded samples.
- Baseline refusals were retained. Authorized filesystem-cache purge restored
  headroom before distinct retries; no thresholds, kernel limits or unrelated
  applications were changed.
- Model files, build targets, raw diagnostics and generated media were on the
  external volume. Existing checkpoints were used without downloads, and user
  database/settings were preserved. Owned processes and the lock were verified
  cleared before returning the campaign slot.

The operator evidence bundle is
`/Volumes/ExternalStorage/mold-1040-qualification/`: `UAT-RESULTS.md`,
`evidence/FINAL-RESULTS.json`, per-run commands/samples, raw inputs/outputs,
comparison scripts, source patches and guard refusals. Timings include loading
and compilation and are not decoder-only benchmarks.

## Recovery integration

`vae_recovery` holds the injectable ordering contract; `vae_tiling` retains
its public mode path and cached environment resolution. Existing callers keep
the infallible cleanup API. Z-Image selects the new fallible policy only when
the actual VAE device is Metal, and completes each attempt before returning
pixels. This currently applies where the retained proactive cap does not select
the legacy tiled branch. Only recognized OOMs authorize retry. Repeated Metal cleanup OOMs are
consumed; unrelated errors propagate. Eager CPU fallback remains GPU-only and
loads fresh F32 VAE weights from the original latent.

CPU/CUDA retain their previous decode ordering and environment-policy behavior,
including CUDA's direct CPU fallback and cleanup error propagation. Tests use
injected decode/completion/cleanup failures without creating GPU devices.
Validation passes 11 recovery tests, 20 VAE-tiling tests and 39 Z-Image pipeline
tests, plus Metal-enabled crate Clippy, workspace Rust formatting, CI routing
and the single-Candle-identity contract. Independent review found no remaining
source blocker. The coordinated Candle publication/pin, cap removal, final
review and exact-head CI remain required before shipping.
