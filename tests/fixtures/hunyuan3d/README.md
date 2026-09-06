# Hunyuan3D component oracle

`transformer21.safetensors` contains synthetic random weights, inputs and the
expected output of Tencent's unchanged Hunyuan3D 2.1 transformer, executed on an
NVIDIA L40S in float32. It contains no pretrained weights. Configuration, seed,
source revision and tolerance are recorded in `transformer21.json`.

Regenerate with `scripts/capture-hunyuan3d21-component.py --help` using the pinned
Tencent checkout and an oracle-only PyTorch environment. Keep the generated
original in the campaign evidence directory before copying a new fixture here.
The CPU Candle test compares the complete forward result, including attention
packing, timestep convention, skip connections, sparse experts and output sign.
A component pass does not establish pretrained or full-pipeline qualification.

`uv-tetrahedron.json` is xatlas-python 0.0.9's exact `parametrize` result for a
synthetic tetrahedron. Regenerate with `scripts/capture-hunyuan3d-uv-reference.py`.
The native xatlas revision is recorded in the fixture and vendored-source README.

`paint-vae-tiny-weights.safetensors`, `paint-vae-tiny.bin` and
`paint-vae-tiny.safetensors` are synthetic Diffusers 0.30 VAE weights plus input,
posterior and decode tensors. Regenerate using
`scripts/capture-hunyuan3d-paint-vae.py --tiny-only`; strict float32 captures disable
TF32. No pretrained weights are checked in. Retain each original capture and
metadata in the campaign evidence directory before copying fixtures.

`paint-vae-opmath.safetensors` records PyTorch 2.5.1 CUDA half-precision
normalization, SiLU and biased linear outputs. GroupNorm cases exercise the
32-thread and 512-thread reductions plus the separate spatial-one path. The
native CUDA test compares normalization exactly, while the portable tensor
fallback has a separately declared rounding tolerance. Regenerate with the
same VAE capture script using `--device cuda --tiny-only`.

`paint-pth-unexpected-int32.bin` contains one float32 tensor named `expected` and
an int32 tensor named `unexpected`, generated with PyTorch 2.5.1. Candle's lenient
parser silently omits the latter; the paint reader must reject the checkpoint.
`paint-pth-negative-offset.bin` modifies a one-tensor PyTorch checkpoint to carry
a signed storage offset of -1. It must return an error before Candle's offset
arithmetic. Peer-review reproducers and original bytes are retained under
`paint-pth-review-v1/` and `paint-pth-review-v2/` in the campaign evidence root.

`paint-projector-tiny.safetensors` contains synthetic Tencent image-projector
weights and pooled/token inputs and outputs. Regenerate with
`scripts/capture-hunyuan3d-paint-projector.py` without checkpoint arguments.

`paint-attention.safetensors` contains synthetic Tencent material self-attention,
reference attention, rotary multiview attention and plain/cross-attention weights
and outputs in float32/float16, plus position tables. Regenerate with
`scripts/capture-hunyuan3d-paint-attention.py` without `--checkpoint`. Five heads,
two batches, two materials and three views expose the reference processor's
concatenate-values-before-head-reshape ordering. Its script also captures installed
weights at production dimensions; those larger files stay in the campaign
evidence directory and are never checked in as synthetic fixtures.
