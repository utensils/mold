---
paths:
  - "crates/mold-inference/src/hunyuan3d/paint_*.rs"
  - "crates/mold-inference/src/hunyuan3d/uv.rs"
  - "crates/mold-inference/vendor/xatlas/**"
  - "crates/mold-inference/build.rs"
  - "crates/mold-candle/src/stable_diffusion/**"
---

# Hunyuan3D paint numerics and UV unwrapping

Moved from the root CLAUDE.md; loaded only when working under the paths above.

- **Paint shares the SD VAE implementation.** `mold_candle::stable_diffusion::vae`
  owns the VAE used by SD1.5, SDXL, SD3 and Hunyuan3D paint. The original posterior
  API preserves SD behavior; paint opts into Diffusers' log-variance bounds and
  supplies its own posterior noise. Paint's published `.bin` weights are parsed
  in Rust and the loader requires every checkpoint tensor to be consumed. The
  campaign qualification ledger distinguishes component parity from completed
  end-to-end paint support.
  `VaeNumerics::Diffusers` is paint's explicit numerical policy: PyTorch's
  normalization/statistics and SiLU rounding boundaries, with a public Candle
  CUDA GroupNorm operation for half precision. `AutoEncoderKL::new` keeps
  `VaeNumerics::Candle` for existing SD callers. Neither compilation nor a paint
  render changes a process-global arithmetic switch.
  `stable_diffusion::normalization::DiffusersGroupNorm` is shared by paint VAE
  and UNet components; epsilon belongs to the layer (VAE and spatial attention
  `1e-6`, UNet residual blocks `1e-5`), including the half-rounded CUDA epsilon.

- **Paint spatial caches follow Tencent's dtype boundaries.** `paint_unet` captures
  reference norm1 at sixteen sites and consumes twelve skips newest-first as
  `[hidden, skip]`. `paint_positions` quantizes in half even for F32 inference;
  F16 maps retain zeroed invalid pixels between scales, whereas F32 maps are
  converted afresh. Integer valid counts round to half before division, and
  final coordinates use ties-to-even. Never replace this with a dtype-neutral
  average or mutate caller-owned maps. Full float32 network parity does not
  close the separate half-precision or full texture-generation gates.
- **Paint UniPC is the VP v-prediction recipe, not Wan's flow solver.**
  `paint_sampler` preserves the sample dtype at every tensor operation, zero-SNR
  beta rescaling, trailing NumPy timesteps, and conversion before correction.
  Its left scalar products deliberately follow PyTorch's different CPU/CUDA
  half rounding. `paint_guidance` keeps both guidance updates separate; folding
  the reference branch algebraically changes half output. The upstream default
  call supplies no camera azimuths, so view weights are all one despite the
  renderer's different camera angles.
- **Paint Linear rounding depends on the incoming layout.**
  `mold_candle::stable_diffusion::linear::forward` is shared by the opt-in paint
  VAE and paint UNet. Torch fuses bias for 2D or contiguous ND inputs; a
  non-contiguous ND input rounds the matrix product before adding bias. Spatial
  inputs must keep BCHW -> B,C,HW -> transpose strides: Candle reshape after
  permute copies contiguous, silently selecting the wrong rounding boundary.
- **A paint conditioning cache belongs to its loaded denoiser and request.**
  `paint_denoiser::PreparedPaint` borrows its owning model and retains the reference
  network, projected DINO and position tables once per request. Guidance repeats
  geometry across three branches, zeros only the first two DINO inputs, and uses
  reference scales `[0,1,1]`. The fifteen-step driver receives explicit initial
  noise and calls cancellation before conditioning and after every sampler step.
  Cancellation never leaves a reusable cache on the model.
- **UV unwrapping is the narrow native exception, and it is compiled the way the oracle compiles it.** `mesh-texture` builds vendored xatlas `f700c779`, exactly the version in the 2.1 oracle’s xatlas-python 0.0.9 — and `build.rs` now passes what that oracle's CMake `Release`/`CXX_STANDARD 17` passes: `-std=c++17 -O3 -DNDEBUG`. Omitting `NDEBUG` leaves all 155 `XA_DEBUG_ASSERT` sites compiled in (a measured ~17% of the unwrap) and is a divergence from the pinned revision, not a safety net. It also builds `-DXA_MULTITHREADED=0`, upstream's own switch: mold submits ONE mesh, so one connected shape is one chart group and exactly one worker ever has work, while the threaded scheduler spawns `hardware_concurrency() - 1` threads with no cap and waits in a bare `yield()` loop — measured 127 threads and 3.06 cores against 1.00 core, for 7% more wall clock and byte-identical UVs.
  `xatlas.h` is unmodified; `xatlas.cpp` carries ONE mold change, every hunk marked `MOLD DIVERGENCE`, with the complete diff in `vendor/xatlas/mold-cancellable-merge.patch` and the rationale in that directory's `README.md`. `segment::ClusteredCharts::mergeCharts` rescans every chart pair after each merge and advances no counter, so upstream neither reports nor interrupts it; on a non-manifold shape it runs for minutes to hours with the progress callback never invoked, which is why a user's cancel was never observed (#1666). `Progress::poll()` re-fires the callback at the percent already reported — `update()` fires only when the whole percent changes, so during that phase `cancel` is never even written and reading it is reading a flag nobody sets. Scheduling only: with nothing cancelling, every merge decision and every output UV is unchanged, pinned by the 1e-7 oracle test and by byte-identical output across patched and unpatched builds on real meshes.
  The Rust wrapper validates geometry, preserves every seam-corner attribute and polls cancellation across native threads; `uv::unwrap_reporting` forwards xatlas's own `ProgressCategory` and percent, which `bridge.cpp` used to receive and discard. Inference, samplers and texture baking remain Rust/Candle. Enabling the build feature alone does not advertise a paint engine.

Half cuDNN convolution accumulation (also paint-relevant) lives in `inference.md`.
