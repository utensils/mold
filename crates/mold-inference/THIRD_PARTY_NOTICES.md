# Third-Party Notices

## Diffusers paint UNet

`src/hunyuan3d/paint_conv.rs`, `src/hunyuan3d/paint_unet.rs` and
`src/hunyuan3d/paint_sampler.rs` adapt Hugging Face Diffusers v0.30.0,
commit `8a79d8ec3973e78065f13638eefc0dc7d4dc6009`, specifically
`models/resnet.py`, `models/downsampling.py`, `models/upsampling.py`,
`models/unets/unet_2d_condition.py`, `models/embeddings.py` and
`models/transformers/transformer_2d.py` and
`schedulers/scheduling_unipc_multistep.py`.
The Rust port implements the fixed inference recipe, explicit timestep
conditioning and bounded output-size validation; training branches are omitted.

    Copyright 2024 The HuggingFace Team. All rights reserved.

Diffusers is Apache-2.0; the full license is retained in `LICENSE-APACHE-2.0`.
No Python or PyTorch runtime ships with this implementation.

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
included in `LICENSE-APACHE-2.0` in this crate and in the published
`mold-ai-candle` crate. Mold's own LTX-Video pipeline, single-file loaders,
latent upsampler, and the rest of the LTX-2 runtime are original work.

The complete repository-wide list lives in `THIRD_PARTY_NOTICES.md` at the
mold repository root.

## xatlas (optional mesh UV unwrapping)

The `mesh-texture` feature builds unmodified xatlas at revision
`f700c7790aaa030e794b52ba7791a05c085faf0c`, the version used by the
Hunyuan3D 2.1 reference's xatlas-python 0.0.9. Copyright (c) 2018–2020
Jonathan Young, MIT licence. Sources and complete licence are retained in
`crates/mold-inference/vendor/xatlas/`. The C ABI bridge is mold-owned code.
No Python runtime is linked or invoked by mold.

## OpenCV Navier–Stokes texture fill

`crates/mold-inference/src/hunyuan3d/paint_ns_fill.rs` ports the RGB radius-three
branch of OpenCV 4.10.0 `modules/photo/src/inpaint.cpp`, revision
`71d3237a093b60a27601c20e9ee6c3e52154e8b1`. It preserves fill ordering and
rounding and adds bounds validation and cancellation. The original notice is:

    For Open Source Computer Vision Library

    Copyright (C) 2000, Intel Corporation, all rights reserved.
    Third party copyrights are property of their respective icvers.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    * Redistribution's of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    * Redistribution's in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    * The name of Intel Corporation may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    This software is provided by the copyright holders and contributors "as is" and
    any express or implied warranties, including, but not limited to, the implied
    warranties of merchantability and fitness for a particular purpose are disclaimed.
    In no event shall the Intel Corporation or contributors be liable for any direct,
    indirect, incidental, special, exemplary, or consequential damages
    (including, but not limited to, procurement of substitute goods or services;
    loss of use, data, or profits; or business interruption) however caused
    and on any theory of liability, whether in contract, strict liability,
    or tort (including negligence or otherwise) arising in any way out of
    the use of this software, even if advised of the possibility of such damage.
