# MiniMax H3 Comfy quantization source audit

Status: source-only portable primitives; runtime activation remains blocked.

This audit did not access MiniMax H3 model payloads, safetensors headers, or
generated outputs. It used only public implementation source. The model's
license gate in `mold-core` remains authoritative, H3 factory registration is
still absent, and the Comfy checkpoint candidate continues to reject every
runtime backend.

## Pinned implementation sources

- ComfyUI commit `a464ac33588ae182f81a090d910cfbf21e255b73`.
- ComfyUI pins `comfy-kitchen==0.2.26` in `requirements.txt`.
- comfy-kitchen tag `v0.2.26`, commit
  `255a43879fe57bbcbecfdb273b46d772b00c5a90`.
- Both implementation repositories publish their source under Apache-2.0.

The relevant ComfyUI authorities are `comfy/quant_ops.py`, `comfy/ops.py`,
`comfy/sd1_clip.py`, and `comfy/text_encoders/minimax.py`. The relevant
comfy-kitchen authorities are `tensor/int8.py`, `tensor/int8_utils.py`,
`tensor/nvfp4.py`, `backends/eager/quantization.py`, and `float_utils.py`.

## Pruned DiT INT8 ConvRot

Comfy names the representation `int8_tensorwise`, but a ConvRot weight is
specifically quantized per output channel. comfy-kitchen rejects ConvRot unless
`per_channel=true`; the resulting scale shape is `[out_features, 1]`. Mold must
therefore reject a scalar or flat `[out_features]` scale for this layout.

For each 256-wide input group, comfy-kitchen builds the normalized regular
Hadamard matrix from the symmetric 4x4 seed and Kronecker products. Offline
conversion stores

```text
W_rot = W * H^T
Q_w[row] = round(W_rot[row] / s_w[row])
```

The accelerated forward rotates activations online, dynamically quantizes each
activation row, performs an INT8 accumulation, and applies the activation and
weight scales. The portable Mold path deliberately uses the source-defined
dequantized alternative instead:

```text
y = (x * H) * (Q_w * s_w)^T + bias
```

Because `H` is symmetric and orthonormal, this equals a full-precision linear
operation with the reconstructed `W = (Q_w * s_w) * H`. Output-row chunking
keeps the dense F32 staging bound explicit. It does not claim numerical parity
with Comfy's additional lossy activation quantization or fused CUDA kernel.

Candle has no signed-I8 tensor dtype. Mold retains checkpoint INT8 data as
unaltered two's-complement bytes in CPU U8 storage and widens each byte through
signed interpretation when staging a chunk. A numeric U8 cast would corrupt
negative weights and is never used.

## Qwen3-VL NVFP4-AWQ

Comfy's NVFP4 storage consists of:

- high-nibble-first E2M1 values, two weights per U8;
- one FP8-E4M3 block scale per 16 logical input values, stored in cuBLAS
  `SWIZZLE_32_4_4` order;
- one F32 tensor scale (`weight_scale_2`); and
- an optional ModelOpt AWQ-style `pre_quant_scale` applied to the input.

The H3 Comfy conditioner contract is the AWQ variant, so Mold requires an
explicit input-scale vector with exactly `in_features` entries. It applies that
vector before the linear operation. Comfy constructs text encoders with
`full_precision_mm=true`, so their NVFP4 representation is a storage and
streaming optimization: weights dequantize before matrix multiplication even
on hardware that offers native NVFP4 compute. Mold follows that ordering and
does not substitute an unqualified Blackwell kernel.

For packed byte `b`, logical even column `2i` uses `b >> 4` and odd column
`2i+1` uses `b & 0x0f`. Reconstruction is

```text
W[row, col] = E2M1[nibble] * E4M3[block] * tensor_scale
y = (x * pre_quant_scale) * W^T + bias
```

Mold unswizzles the block scales once, retains compressed U8 weights on the
host, and stages only a bounded number of output rows for each F32 Candle
matrix multiplication. The same portable path is typed for CPU, Metal, and
CUDA execution devices; native quantized kernels remain out of scope.

## Synthetic qualification and remaining gates

The committed tests use tiny deterministic synthetic tensors only. They prove:

- regular Hadamard symmetry and orthonormality;
- ConvRot forward equivalence to explicit dequantization;
- strict `[out_features, 1]` INT8 scale authority and signed-byte handling;
- high-nibble-first E2M1 decoding, tensor/block scale application, and
  multi-tile scale unswizzling against a fixed comfy-kitchen layout oracle;
- AWQ input scaling occurs before the dequantized linear operation; and
- source-dtype-aware encoded byte accounting, CPU/Metal forward parity, and
  fail-closed invalid dtypes, shapes, scales, and zero-sized staging plans.

Before any production activation, separate authorization and qualification
work must still provide an approved H3 checkpoint identity, verify its complete
content digest, connect an immutable loader/frozen placement to these
primitives, measure real quality and memory, validate representative long
sequences, and explicitly register the runtime factory. None of those gates is
satisfied by this source-only change.
