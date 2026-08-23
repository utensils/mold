# MiniMax H3 NVFP4 layout probe

Companion to `minimax-h3-nvfp4-layout-probe.json`, which is the verbatim output
of one run of the unregistered example
`crates/mold-candle/examples/h3_nvfp4_layout_probe.rs`. It answers part 1 of
issue #1317: whether the published NVFP4 FL2VA weights can be read at all, how
far they sit from the BF16 base, and what moving their dequantize onto the
device would buy.

Reproduce with:

```
cargo build --example h3_nvfp4_layout_probe -p mold-ai-candle --features cuda --release
./target/release/examples/h3_nvfp4_layout_probe \
  <nvfp4.safetensors> <int8_convrot.safetensors> <pruned_bf16.safetensors> \
  --blocks 0,25,49 --rows 4096 --cost-rows 37296
```

## Run provenance

| | |
| --- | --- |
| Host | hal9000 — i9-13900K, 64 GB RAM |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, driver 580.142 |
| Kernel | Linux 6.12.93 |
| Toolchain | rustc 1.95.0 (59807616e 2026-04-14), release profile, `--features cuda` |
| Commit | `51db581654649638de7f417cfef148d73f510ff9` (`origin/main`) |
| Probed blocks | 0, 25, 49 x {`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`} |

A production `mold serve` shared the GPU during the run but rendered nothing;
peak probe VRAM was about 5.5 GB of the 23.6 GB free.

### Inputs

| Role | Path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| NVFP4 FL2VA (third-party `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` @ `908eccad`) | `/storage-fast/mold/probe-1317/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` | 12,528,636,865 | `6ab7f0c48141e7919b32f925ca3def22e06a6aebeb9e0b6f5a0be0fe8409976f` |
| INT8 ConvRot FL2VA (mold's pinned Comfy-Org object) | `/storage-fast/mold/models/minimax-h3-fl2va-comfy-pruned-int8/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| Pruned BF16 FL2VA (Comfy-Org) | `/storage-fast/mold/probe-1317/minimax_h3_fl2va_pruned_bf16.safetensors` | 40,225,724,176 | `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4` |

All three files agree on everything neither quantization touched: the probe
compares every tensor the three carry at one name, dtype, and shape, and 332 of
332 are byte-identical in all three. That is what licenses reading the BF16 file
as the common base for both quantizations, including the NVFP4 one from a
different publisher.

The NVFP4 artifact carries 65 bytes of non-tensor data past the end of its
payload (`\nL2P_bypass_...\n`). Mold's production opener,
`open_h3_comfy_published_int8_checkpoint`, refuses the file for exactly that
reason, so the probe does its own bounded header read that tolerates and reports
trailing bytes while keeping every other structural rule strict.

## Verdict 1 — the block scales are swizzled

**Swizzled.** Mean relative Frobenius error against the BF16 base is **0.0948**
under the swizzled hypothesis and **0.5693** under the natural one, across all
twelve probed linears with no overlap between the two populations. So
`comfy_quant.rs`'s unconditional `unswizzle_nvfp4_scales` is right on H3-shaped
tensors.

That settles a second, wider assumption at the same time. `mold-inference`'s own
`nvfp4::dequant_nvfp4_to_bf16_cpu` also calls `unswizzle_block_scales`
unconditionally, and it is the dequantize both LTX-2's video transformer and
Flux.2's transformer use for their NVFP4 weights. Neither had numerical
confirmation on real comfy-kitchen tensors; this run is one, on a checkpoint
from a different publisher than either of them.

The header could not have answered this: every probed linear has `out_features %
128 == 0` and `in_features/16 % 4 == 0`, so the swizzled and natural scale
shapes coincide with zero padding.

Two details differ from #1317's prediction and are worth recording. The wrong
hypothesis lands at 0.40 – 1.00 rather than the predicted ~1.41 — the 128x4
swizzle permutes scales *within* a 512-element tile, and neighbouring blocks of
one weight have similar magnitudes, so the damage is large but well short of
decorrelated. And the *right* hypothesis lands at ~0.095, not the predicted
0.03–0.06. There is still no ambiguous middle: the two populations are 4x apart.

| Block | Linear | swizzled vs BF16 | natural vs BF16 | probe vs library max delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | `attn.qkv_proj` | 0.09473 | 0.53364 | 0.0e+00 |
| 0 | `attn.out_proj` | 0.09581 | 0.71851 | 0.0e+00 |
| 0 | `mlp.fc1` | 0.09518 | 0.55147 | 0.0e+00 |
| 0 | `mlp.fc2` | 0.09470 | 0.42953 | 0.0e+00 |
| 25 | `attn.qkv_proj` | 0.09482 | 0.40065 | 0.0e+00 |
| 25 | `attn.out_proj` | 0.09556 | 0.52548 | 0.0e+00 |
| 25 | `mlp.fc1` | 0.09510 | 0.44020 | 0.0e+00 |
| 25 | `mlp.fc2` | 0.09468 | 0.44359 | 0.0e+00 |
| 49 | `attn.qkv_proj` | 0.09410 | 0.55746 | 0.0e+00 |
| 49 | `attn.out_proj` | 0.09578 | 0.66609 | 0.0e+00 |
| 49 | `mlp.fc1` | 0.09459 | 0.56245 | 0.0e+00 |
| 49 | `mlp.fc2` | 0.09311 | 1.00269 | 0.0e+00 |

The last column cross-checks the probe's own index math against the shipped
`H3ComfyNvfp4AwqLinear`, which unswizzles unconditionally at construction: the
two dequantized weights are identical to the last bit on every probed linear.
The probe's FP8-E4M3 table also agrees with candle's own `F8E4M3 -> F32` cast on
all 256 byte values.

## Verdict 2 — NVFP4 is ~7x further from BF16 than INT8 ConvRot is

**No-go under #1317's own rule.** In activation space NVFP4 sits 0.0931 – 0.0958
from the BF16 reference while INT8 ConvRot sits 0.0125 – 0.0150, a ratio of
6.39x to 7.63x. The stated rule — *if `nvfp4_vs_bf16` exceeds `2 x int8_vs_bf16`
in activation space, the layout is not worth shipping as-is* — fails by a wide
margin on every one of the twelve linears.

Read this as a statement about the **third-party NVFP4 requantization**, not
about the swizzle: verdict 1 already establishes the layout is being read
correctly, and reading it the other way costs another 6x. What the numbers say
is that this particular NVFP4 artifact is a much coarser quantization of the
same BF16 base than mold's pinned INT8 ConvRot object, which is unsurprising —
4-bit blocks of 16 with an FP8 scale against 8-bit rows with a Hadamard rotation
and an F32 scale. Row cosine stays high (0.9946 minimum), so the error is
broadly magnitude noise rather than a few destroyed rows, but it is roughly an
order of magnitude more of it.

Weight space, relative Frobenius against the BF16 base:

| Block | Linear | NVFP4 vs BF16 | INT8 vs BF16 | NVFP4 vs INT8 | NVFP4 row cos min | NVFP4 row cos p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `attn.qkv_proj` | 0.09473 | 0.00884 | 0.09431 | 0.995115 | 0.995525 |
| 0 | `attn.out_proj` | 0.09581 | 0.01200 | 0.09505 | 0.994641 | 0.995414 |
| 0 | `mlp.fc1` | 0.09518 | 0.00882 | 0.09478 | 0.995059 | 0.995482 |
| 0 | `mlp.fc2` | 0.09470 | 0.00937 | 0.09423 | 0.995246 | 0.995518 |
| 25 | `attn.qkv_proj` | 0.09482 | 0.00883 | 0.09442 | 0.995104 | 0.995501 |
| 25 | `attn.out_proj` | 0.09556 | 0.00981 | 0.09506 | 0.994921 | 0.995433 |
| 25 | `mlp.fc1` | 0.09510 | 0.00883 | 0.09469 | 0.995070 | 0.995474 |
| 25 | `mlp.fc2` | 0.09468 | 0.00934 | 0.09422 | 0.995209 | 0.995520 |
| 49 | `attn.qkv_proj` | 0.09410 | 0.00887 | 0.09369 | 0.995141 | 0.995578 |
| 49 | `attn.out_proj` | 0.09578 | 0.01121 | 0.09513 | 0.994934 | 0.995417 |
| 49 | `mlp.fc1` | 0.09459 | 0.00885 | 0.09418 | 0.995026 | 0.995503 |
| 49 | `mlp.fc2` | 0.09311 | 0.00952 | 0.09262 | 0.995336 | 0.995591 |

Activation space, fixed-seed N(0,1) at 4,096 rows (the real per-forward count,
`REVIEWED_MAX_TARGET_VIDEO_ROWS` = 37,296, is the extrapolation basis for the
cost section below and does not change these ratios):

| Block | Linear | NVFP4 vs BF16 | INT8 vs BF16 | ratio |
| ---: | --- | ---: | ---: | ---: |
| 0 | `attn.qkv_proj` | 0.09475 | 0.01249 | 7.58x |
| 0 | `attn.out_proj` | 0.09578 | 0.01500 | 6.39x |
| 0 | `mlp.fc1` | 0.09519 | 0.01248 | 7.63x |
| 0 | `mlp.fc2` | 0.09471 | 0.01324 | 7.15x |
| 25 | `attn.qkv_proj` | 0.09484 | 0.01249 | 7.59x |
| 25 | `attn.out_proj` | 0.09556 | 0.01329 | 7.19x |
| 25 | `mlp.fc1` | 0.09510 | 0.01248 | 7.62x |
| 25 | `mlp.fc2` | 0.09468 | 0.01322 | 7.16x |
| 49 | `attn.qkv_proj` | 0.09411 | 0.01251 | 7.52x |
| 49 | `attn.out_proj` | 0.09576 | 0.01437 | 6.66x |
| 49 | `mlp.fc1` | 0.09460 | 0.01251 | 7.56x |
| 49 | `mlp.fc2` | 0.09311 | 0.01334 | 6.98x |

These are the two *shipped execution paths*, not two weight errors:
`H3ComfyInt8ConvRotLinear::forward_reference` is W8A8 and dynamically quantizes
the activation as well, which is why its activation-space error (0.0125 –
0.0150) sits above its weight-space error (0.0088 – 0.0120), while
`H3ComfyNvfp4AwqLinear::forward_dequantized` keeps full-precision activations
and lands at essentially its weight error. INT8 carries the extra activation
term and is still ~7x closer to BF16.

## Verdict 3 — the host dequantize is the cost, and a device arm removes it

Measured on `blocks.0.attn.qkv_proj` (`[21504, 5376]`) at 37,296 rows, 2 warmups
and 5 timed runs, F32 activations. The INT8 comparison takes the native cuBLASLt
arm (`select_h3_int8_linear_kind` -> `NativeCudaInt8`).

| Path | ms / linear (fastest) | mean | extrapolated denoise (s) |
| --- | ---: | ---: | ---: |
| `H3ComfyNvfp4AwqLinear::forward_dequantized` (shipped, host scalar loop) | 297.6 | 300.1 | 536 |
| `H3ComfyInt8ConvRotLinear::forward_reference` (native cuBLASLt INT8) | 160.8 | 161.3 | 289 |
| Prototype device dequantize + F32 matmul | 257.0 | 261.8 | 463 |
| Prototype device dequantize + BF16 matmul | 101.2 | 102.3 | 182 |
| Prototype device dequantize alone, no matmul | 41.3 | 41.3 | — |

Every figure is the fastest of 5 timed runs after 2 warmups, each individually
synchronized so it is one call's latency rather than a pipelined average.
hal9000 is a shared development box; this run was taken at a 1-minute load
average of 4.2, and three consecutive runs between load 4 and 15 agreed to
within 2% on every row. Earlier attempts at load 40–60 inflated the whole table
by roughly 2x while leaving the ratios between rows broadly intact. The
verdict-1 and verdict-2 numbers are exact and load-independent either way.

Extrapolation basis is #1317's: `per_linear_ms x 4 linears x 50 blocks x 9
terminal-inclusive steps`, against the reviewed Turbo 8-step baseline of 730 s.
These are hypothetical figures for an NVFP4 *transformer*; today's DiT is the
INT8 object and only the Qwen3-VL conditioner is NVFP4.

Stage attribution for the shipped path, with a device synchronize between each
stage (so these serialize work the real forward overlaps, and each is an upper
bound on its own stage rather than an addend):

| Stage | ms |
| --- | ---: |
| host scalar dequantize | 101.7 |
| host to device | 134.8 |
| matmul | 159.7 |

The device arm collapses the first two stages into 41.3 ms of device work
against the 237 ms they cost on the host — a 5.7x reduction, and the result
#1317 part 2 is after.

What remains is the GEMM, and it decides more than the dequantize does. An F32
matmul over the dequantized weight comes to 1.60x the native INT8 path — the F32
arithmetic, not the unpacking, is what costs — while casting the dequantized
weight to BF16 first brings the whole linear to 0.63x, i.e. below the native
INT8 path rather than above it.

Treat that last figure as encouraging rather than decisive. It is one linear in
isolation: it charges nothing for the resident weight an NVFP4 transformer would
still have to hold, its INT8 comparator additionally quantizes its activations
on every call, and a real block interleaves attention and norms that this
measurement omits entirely. What it does establish is that an NVFP4 DiT is not
automatically slower than today's INT8 one once the dequantize leaves the host,
which is the premise part 2 rests on.

What is not in doubt at all is the **conditioner**, which is NVFP4 today and
whose only alternative to the device arm is exactly the 102 ms single-threaded
host scalar loop above, per linear, on every prompt encode. End to end, the
shipped path costs 298 ms per linear against the prototype's 101 ms — 2.9x.

## Blocking finding for part 2 — U8 `index_select` ids cannot address entry 255

#1317 part 2 proposes two 256-entry F32 nibble lookup tables driven by
`index_select` over U8 ids, on the grounds that candle's CUDA backend accepts U8
ids directly (`candle-core/src/cuda_backend/mod.rs:459`). It does — but
`candle-kernels/src/indexing.cu:60` reserves `max_value<I>()` as a zero-padding
sentinel:

```c
if (ids[id_i] == max_value<I>()) {
  out[dst_i] = static_cast<T>(0);
} else { ... }
```

For `uint8_t` ids that sentinel is **255**, so a 256-entry table can never
return its last entry. `0xff` is an ordinary NVFP4 payload byte — two `-6.0`
E2M1 nibbles — and the probed `blocks.0.attn.qkv_proj` weight contains 122,820
of them. Built that way, the arm silently zeroes 245,640 of 115,605,504 weights
(0.21%), with no error anywhere.

The probe measures that arm as a negative result and then uses U32 ids, which
are cast on the device from the U8 packed tensor: the packed weight stays U8 at
rest, so the 10.84 GB residency argument is untouched, and only a transient id
buffer is widened. With U32 ids the device arm is **bit-identical** to the
shipped host loop across all 115,605,504 elements of that weight, which is the
gate part 2 asks for.

The cheaper alternative — keeping U8 ids and adding a `packed == 0xff`
correction term — is left to part 2 to evaluate; it trades one 4x-wide cast for
two extra elementwise passes.

## Scope

This example is unregistered on purpose: `mold-candle`'s `Cargo.toml` has no
`[[example]]` table, so autodiscovery is the only thing that builds it and it
reaches no release surface. It registers no engine, changes no shipping path,
and its unit tests run under `cargo test --example h3_nvfp4_layout_probe -p
mold-ai-candle --features cuda`.
