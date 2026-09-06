# Third-Party Notices

## Diffusers VAE numerical path and PyTorch CUDA normalization

The opt-in VAE blocks in `src/stable_diffusion/vae/precision.rs` follow
Hugging Face Diffusers v0.30.0, commit
`8a79d8ec3973e78065f13638eefc0dc7d4dc6009`, specifically `models/resnet.py`,
`models/unets/unet_2d_blocks.py` and `models/attention_processor.py`.

    Copyright 2024 The HuggingFace Team. All rights reserved.

The Rust adaptation omits training and time conditioning, bounds attention
allocations, and makes numerical policy explicit. Diffusers is Apache-2.0;
the full text is retained in `LICENSE-APACHE-2.0`.

Shared `src/stable_diffusion/normalization.rs` and its CUDA implementation
follow PyTorch v2.5.1's CUDA `group_norm_kernel.cu` rounding
boundaries; SiLU follows `ActivationSiluKernel.cu` opmath precision. Shared
`src/stable_diffusion/linear.rs` follows `aten/src/ATen/native/Linear.cpp`'s
layout-dependent fused versus separate bias-rounding boundary. PyTorch's
copyright notices and BSD license terms are retained in `LICENSE-PYTORCH`.
The implementation uses public Candle tensor operations and ships no PyTorch
or Python runtime. Existing SD callers retain Candle's numerical policy.
The shared normalization accepts the layer epsilon explicitly so the VAE's
`1e-6` and paint UNet residual blocks' `1e-5` use the same implementation.

## FerrisMind/candle-video (LTX-Video port)

Mold's legacy LTX-Video transformer, 3D causal video VAE, and flow-match Euler
scheduler in `crates/mold-candle/src/ltx_video/` (`transformer.rs`, `vae.rs`,
`sampling.rs`) were ported from
[candle-video](https://github.com/FerrisMind/candle-video) by FerrisMind,
`src/models/ltx_video/{ltx_transformer,vae,scheduler}.rs`, as of upstream
commit `d7208cbbb07a8bb3bf286ffa447616335eecd2e5` (the tip when mold's port
landed on 2026-04-04). candle-video is itself a Rust port of the Hugging Face
[diffusers](https://github.com/huggingface/diffusers) LTX-Video implementation
(`transformer_ltx.py`, `autoencoder_kl_ltx.py`,
`scheduling_flow_match_euler_discrete.py`), which is also Apache-2.0. The LTX-2
video transformer and video VAE in
`crates/mold-inference/src/ltx2/model/{video_transformer,video_vae}.rs` were
adapted from mold's copy and retain portions of that code.

    Copyright 2025 FerrisMind

candle-video is licensed under the Apache License, Version 2.0
(<https://github.com/FerrisMind/candle-video/blob/main/LICENSE>). Mold's copies
retain attribution, identify their changes in each file header (standalone
models without the upstream pipeline traits, flash-attn gates and debug output
removed, LTX-2 positional-embedding and config changes), and are distributed
under Apache-2.0 alongside Mold's MIT license; the complete license text is
included in `crates/mold-candle/LICENSE-APACHE-2.0` and
`crates/mold-inference/LICENSE-APACHE-2.0`, and ships in both published crates
(`mold-ai-candle`, `mold-ai-inference`). Mold's own LTX-Video pipeline, single-file loaders,
latent upsampler, and the rest of the LTX-2 runtime are original work.

## comfy-kitchen INT8 CUDA reference

The MiniMax H3 native INT8 CUDA kernels and cuBLASLt operation layout are
adapted from comfy-kitchen 0.2.26, commit
`255a43879fe57bbcbecfdb273b46d772b00c5a90`, specifically
`backends/cuda/ops/int8_linear.cu` and
`backends/cuda/ops/cublas_gemm_int8.cu`.

    Copyright (c) 2025 Comfy Org. All rights reserved.
    Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Those sources are licensed under the Apache License, Version 2.0. Mold's copy
retains attribution and identifies its changes; the complete license text is
included in `LICENSE-APACHE-2.0`.
