# Third-Party Notices

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
included in `crates/mold-candle/LICENSE-APACHE-2.0` and in the published
`mold-ai-candle` crate. Mold's own LTX-Video pipeline, single-file loaders,
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
included in `crates/mold-candle/LICENSE-APACHE-2.0` and in the published
`mold-ai-candle` crate.

## Pillow image resampling algorithm

The pure-Rust MiniMax H3 endpoint resampler in
`crates/mold-inference/src/minimax_h3/pipeline.rs` independently expresses the
numeric behavior of Pillow's U8 LANCZOS resampler. The reference is Pillow
12.3.0, commit `bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d`,
`src/libImaging/Resample.c`. No Pillow source file or C implementation is
vendored in Mold.

The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

## torchaudio sinc resampling algorithm

The pure-Rust MiniMax H3 reference-audio resampler in
`crates/mold-inference/src/reference_media.rs` independently expresses the
numeric behavior of torchaudio's default Hann-windowed sinc resampler. The
reference is torchaudio 2.8.0, commit
`6e1c7fe9ff6d82b8665d0a46d859d3357d2ebaaa`,
`src/torchaudio/functional/functional.py`. No torchaudio source file or Python
implementation is vendored in Mold.

torchaudio is licensed under the BSD 2-Clause License:

    Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
    All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
