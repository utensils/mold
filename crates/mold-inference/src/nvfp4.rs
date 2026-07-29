//! NVFP4 (NVIDIA FP4) portable dequantization.
//!
//! NVFP4 stores each weight as three layers of quantization:
//!
//! 1. `weight`: 4-bit E2M1 nibbles, two packed per `U8` byte, shape `[N, K/2]`.
//! 2. `weight_scale`: per-block FP8-E4M3 scale, one per 16 K-values, shape `[N, K/16]`.
//! 3. `weight_scale_2`: per-tensor F32 scalar.
//!
//! Full-precision reconstruction is `W[n,k] = E2M1[bits] × E4M3[block] × F32_tensor_scale`.
//!
//! The portable runtime performs the FP4×FP8-block product into f32, applies
//! the per-tensor F32 scale, then stores a BF16 CPU cache for streaming matmul.
//! Native Blackwell FP4 tensor-core execution is gated behind
//! `MOLD_NVFP4_BACKEND=native` and is intentionally unavailable until validated
//! on real sm_120 hardware.
//!
//! Pack order: within each U8 the **high** nibble (`>> 4`) holds K=2i (even),
//! the **low** nibble (`& 0x0F`) holds K=2i+1 (odd). Matches ComfyUI's
//! `comfy_kitchen.float_utils.pack_uint4`:
//! `packed[i] = (data[2i] << 4) | data[2i+1]`,
//! and the eager `dequantize_nvfp4` `torch.stack([hi, lo], dim=-1)` ordering.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

/// Block of 16 weights share one FP8-E4M3 scale.
pub const NVFP4_BLOCK_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Nvfp4Backend {
    Portable,
    #[allow(dead_code)]
    Native,
}

/// Resolve the runtime NVFP4 backend from `MOLD_NVFP4_BACKEND`.
///
/// `auto` currently means the portable BF16 streaming backend everywhere. The
/// native backend must not silently activate until it has passed numerical and
/// generation validation on sm_120/Blackwell hardware.
pub(crate) fn resolve_nvfp4_backend(device: &Device) -> candle_core::Result<Nvfp4Backend> {
    let value = crate::runtime_env::value("MOLD_NVFP4_BACKEND");
    resolve_nvfp4_backend_value(device, value.as_deref())
}

fn resolve_nvfp4_backend_value(
    device: &Device,
    value: Option<&str>,
) -> candle_core::Result<Nvfp4Backend> {
    let value = value.unwrap_or("auto");
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "portable" => Ok(Nvfp4Backend::Portable),
        "native" => Err(candle_core::Error::Msg(format!(
            "MOLD_NVFP4_BACKEND=native requires sm_120/Blackwell NVFP4 tensor-core support; \
             current device/build ({device:?}) supports only the portable BF16 streaming backend"
        ))),
        other => Err(candle_core::Error::Msg(format!(
            "invalid MOLD_NVFP4_BACKEND={other:?}; expected auto, portable, or native"
        ))),
    }
}

/// Apply the cuBLAS SWIZZLE_32_4_4 tiled layout (forward of
/// `unswizzle_block_scales`). Mirrors `comfy_kitchen.float_utils.to_blocked`
/// with `flatten=False`. Used by tests to construct swizzled fixtures; not
/// hot-path for production loading.
#[cfg(test)]
pub fn swizzle_block_scales(
    natural: &[f32],
    num_rows: usize,
    num_cols_blocks: usize,
) -> Result<Vec<f32>> {
    if natural.len() != num_rows * num_cols_blocks {
        bail!(
            "swizzle_block_scales: expected {} elements for [{}, {}], got {}",
            num_rows * num_cols_blocks,
            num_rows,
            num_cols_blocks,
            natural.len()
        );
    }
    let n_row_blocks = num_rows.div_ceil(128);
    let n_col_blocks = num_cols_blocks.div_ceil(4);
    let padded_rows = n_row_blocks * 128;
    let padded_cols = n_col_blocks * 4;
    let mut swizzled = vec![0.0f32; padded_rows * padded_cols];
    for r in 0..num_rows {
        let i = r / 128;
        let h = r % 128;
        let a = h / 32;
        let b = h % 32;
        for q in 0..num_cols_blocks {
            let j = q / 4;
            let w = q % 4;
            let c = a * 4 + w;
            let t = i * n_col_blocks + j;
            let flat = t * 512 + b * 16 + c;
            swizzled[flat] = natural[r * num_cols_blocks + q];
        }
    }
    Ok(swizzled)
}

/// Reverse the cuBLAS SWIZZLE_32_4_4 tiled layout used by ComfyUI's
/// `comfy_kitchen.tensor.TensorCoreNVFP4Layout` for NVFP4 block scales.
///
/// The converter stores scales in a layout sized
/// `(roundup(num_rows, 128), roundup(num_cols_blocks, 4))` where `num_cols_blocks
/// = K / 16`. The bytes inside that tile are interleaved by the cuBLAS scaled-mm
/// recipe `SWIZZLE_32_4_4`. Reference: `comfy_kitchen.float_utils.from_blocked`.
///
/// Forward mapping for a natural element at `(r, q)` (r in `0..num_rows`, q in
/// `0..num_cols_blocks`):
/// - `i = r / 128`,  `a = (r % 128) / 32`,  `b = r % 32`
/// - `j = q / 4`,    `w = q % 4`
/// - `c = a * 4 + w`,  `t = i * n_col_blocks + j`
/// - `flat = t * 512 + b * 16 + c`
/// - stored index is `flat` interpreted in row-major over `(padded_rows, padded_cols)`.
pub fn unswizzle_block_scales(
    swizzled: &[f32],
    num_rows: usize,
    num_cols_blocks: usize,
) -> Result<Vec<f32>> {
    let n_row_blocks = num_rows.div_ceil(128);
    let n_col_blocks = num_cols_blocks.div_ceil(4);
    let padded_rows = n_row_blocks * 128;
    let padded_cols = n_col_blocks * 4;
    let expected = padded_rows * padded_cols;
    if swizzled.len() != expected {
        bail!(
            "unswizzle_block_scales: expected {expected} elements ({padded_rows}×{padded_cols}) for natural [{num_rows}, {num_cols_blocks}], got {}",
            swizzled.len()
        );
    }

    let mut natural = vec![0.0f32; num_rows * num_cols_blocks];
    for r in 0..num_rows {
        let i = r / 128;
        let h = r % 128;
        let a = h / 32;
        let b = h % 32;
        for q in 0..num_cols_blocks {
            let j = q / 4;
            let w = q % 4;
            let c = a * 4 + w;
            let t = i * n_col_blocks + j;
            let flat = t * 512 + b * 16 + c;
            // flat is the row-major index into the (padded_rows, padded_cols) storage.
            natural[r * num_cols_blocks + q] = swizzled[flat];
        }
    }
    Ok(natural)
}

/// Signed E2M1 lookup, indexed by 4-bit nibble.
/// bits = [sign(1) | exp(2) | mantissa(1)], exponent bias = 1.
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Convert one FP8-E4M3 byte to f32. Exponent bias 7, no infinity (S.1111.111 = NaN).
///
/// Production code uses candle's `Tensor::to_dtype(DType::F32)` on F8E4M3 tensors
/// (correct, fast, and integrated with the rest of the cast machinery). This
/// helper exists for tests + spec documentation: the unit test confirms the
/// per-byte mapping mold relies on, independent of candle's implementation.
#[allow(dead_code)]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xF) as i32;
    let mant_bits = byte & 0x7;
    if exp == 0xF && mant_bits == 0x7 {
        return f32::NAN;
    }
    let mant = mant_bits as f32;
    if exp == 0 {
        sign * (mant / 8.0) * (1.0 / 64.0)
    } else {
        sign * (1.0 + mant / 8.0) * (2.0_f32).powi(exp - 7)
    }
}

/// Dequantize NVFP4 packed weights × per-block scales (already converted to f32).
///
/// The per-tensor `weight_scale_2` is intentionally NOT applied here — callers
/// store it as a sidecar and apply it during matmul.
///
/// `n_rows` is the logical output dim (N); `n_cols` is the logical input dim (K).
/// `packed` must be `[N, K/2]` in row-major byte order.
/// `block_scales` must be `[N, K/16]` in row-major f32 order. Production
/// callers obtain it from candle's `tensor.to_dtype(DType::F32)` on the
/// FP8-E4M3 `weight_scale` tensor; tests construct it directly via
/// `e4m3_to_f32(byte)`.
pub fn dequantize_nvfp4_to_f32(
    packed: &[u8],
    block_scales: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f32>> {
    if !n_cols.is_multiple_of(NVFP4_BLOCK_SIZE) {
        bail!("NVFP4: K ({n_cols}) must be a multiple of block size {NVFP4_BLOCK_SIZE}");
    }
    let pack_cols = n_cols / 2;
    let scale_cols = n_cols / NVFP4_BLOCK_SIZE;
    let expected_packed = n_rows * pack_cols;
    let expected_scales = n_rows * scale_cols;
    if packed.len() != expected_packed {
        bail!(
            "NVFP4: packed buffer is {} bytes, expected {} for [{}, {}]",
            packed.len(),
            expected_packed,
            n_rows,
            pack_cols
        );
    }
    if block_scales.len() != expected_scales {
        bail!(
            "NVFP4: block_scales is {} entries, expected {} for [{}, {}]",
            block_scales.len(),
            expected_scales,
            n_rows,
            scale_cols
        );
    }

    let mut out = vec![0.0f32; n_rows * n_cols];
    let bytes_per_block = NVFP4_BLOCK_SIZE / 2;

    for n in 0..n_rows {
        let p_row = &packed[n * pack_cols..(n + 1) * pack_cols];
        let s_row = &block_scales[n * scale_cols..(n + 1) * scale_cols];
        let o_row = &mut out[n * n_cols..(n + 1) * n_cols];

        for (block_idx, &s) in s_row.iter().enumerate() {
            let p_off = block_idx * bytes_per_block;
            let o_off = block_idx * NVFP4_BLOCK_SIZE;
            for j in 0..bytes_per_block {
                let byte = p_row[p_off + j];
                let hi = (byte >> 4) as usize;
                let lo = (byte & 0x0F) as usize;
                // Matches ComfyUI pack_uint4: HIGH nibble = even index (2j),
                // LOW nibble = odd index (2j+1).
                o_row[o_off + 2 * j] = E2M1_LUT[hi] * s;
                o_row[o_off + 2 * j + 1] = E2M1_LUT[lo] * s;
            }
        }
    }
    Ok(out)
}

/// Dequantize a full, unsliced NVFP4 weight to a BF16 tensor on CPU.
///
/// `packed` must be U8 `[N, K/2]`. `block_scales` must be F8E4M3 in the
/// cuBLAS `SWIZZLE_32_4_4` tiled layout produced by ComfyUI-style NVFP4
/// converters. `tensor_scale` is applied before the BF16 cast.
pub(crate) fn dequant_nvfp4_to_bf16_cpu(
    packed: &Tensor,
    block_scales: &Tensor,
    tensor_scale: f32,
) -> candle_core::Result<Tensor> {
    let packed_dims = packed.dims();
    let scale_dims = block_scales.dims();
    if packed_dims.len() != 2 || scale_dims.len() != 2 {
        candle_core::bail!(
            "NVFP4 streaming: rank mismatch — packed {:?}, scales {:?}",
            packed_dims,
            scale_dims,
        );
    }
    let n_rows = packed_dims[0];
    let n_cols = packed_dims[1] * 2;
    let num_cols_blocks = n_cols / NVFP4_BLOCK_SIZE;
    let packed_bytes: Vec<u8> = packed.flatten_all()?.to_vec1()?;
    let swizzled_scales: Vec<f32> = block_scales
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let scales_f32 = unswizzle_block_scales(&swizzled_scales, n_rows, num_cols_blocks)
        .map_err(|err| candle_core::Error::Msg(format!("NVFP4 unswizzle: {err}")))?;
    let mut dequant = dequantize_nvfp4_to_f32(&packed_bytes, &scales_f32, n_rows, n_cols)
        .map_err(|err| candle_core::Error::Msg(format!("NVFP4 streaming dequant: {err}")))?;
    for value in &mut dequant {
        *value *= tensor_scale;
    }
    let f32_t = Tensor::from_vec(dequant, (n_rows, n_cols), &Device::Cpu)?;
    f32_t.to_dtype(DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn nvfp4_backend_auto_and_portable_resolve_to_portable() {
        let device = Device::Cpu;
        assert_eq!(
            resolve_nvfp4_backend_value(&device, None).unwrap(),
            Nvfp4Backend::Portable
        );
        assert_eq!(
            resolve_nvfp4_backend_value(&device, Some("auto")).unwrap(),
            Nvfp4Backend::Portable
        );
        assert_eq!(
            resolve_nvfp4_backend_value(&device, Some("portable")).unwrap(),
            Nvfp4Backend::Portable
        );
    }

    #[test]
    fn nvfp4_backend_native_requires_blackwell() {
        let err = resolve_nvfp4_backend_value(&Device::Cpu, Some("native"))
            .expect_err("native must fail on CPU");
        assert!(err.to_string().contains("requires sm_120/Blackwell"));
    }

    #[test]
    fn nvfp4_backend_rejects_invalid_value() {
        let err = resolve_nvfp4_backend_value(&Device::Cpu, Some("banana"))
            .expect_err("invalid backend must fail");
        assert!(err.to_string().contains("MOLD_NVFP4_BACKEND"));
    }

    #[test]
    fn e2m1_lut_matches_spec() {
        assert_eq!(E2M1_LUT[0], 0.0);
        assert_eq!(E2M1_LUT[1], 0.5);
        assert_eq!(E2M1_LUT[2], 1.0);
        assert_eq!(E2M1_LUT[3], 1.5);
        assert_eq!(E2M1_LUT[4], 2.0);
        assert_eq!(E2M1_LUT[5], 3.0);
        assert_eq!(E2M1_LUT[6], 4.0);
        assert_eq!(E2M1_LUT[7], 6.0);
        for i in 0..8 {
            assert_eq!(
                E2M1_LUT[i + 8],
                -E2M1_LUT[i],
                "negative half mirrors positive"
            );
        }
    }

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), 0.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
        // 0x38 = S.0111.000: sign=0, exp=7 (bias=7 → 2^0), mantissa=0 → 1.0
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        // 0xB8 = -1.0
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        // 0x40 = S.1000.000: exp=8 → 2^1=2, 1+0/8 = 1.0 → 2.0
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        // 0x30 = S.0110.000: exp=6 → 2^-1, 1.0 → 0.5
        assert_eq!(e4m3_to_f32(0x30), 0.5);
        // 0x7E = S.1111.110: exp=15, mantissa=6 → 1.75 × 2^8 = 448 (max finite)
        assert!((e4m3_to_f32(0x7E) - 448.0).abs() < 1e-3);
        // 0x01 = subnormal: 1/8 × 2^-6 = 0.001953125
        assert!((e4m3_to_f32(0x01) - 0.001953125).abs() < 1e-9);
    }

    /// Helper: convert a slice of E4M3 bytes to f32 scales for primitive callers.
    fn scales_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.iter().map(|&b| e4m3_to_f32(b)).collect()
    }

    #[test]
    fn dequant_single_block_produces_expected_values() {
        // One block of 16 weights, all nibble = 0b0010 (E2M1 = +1.0).
        // Block scale 0x38 (= 1.0). Tensor scale not applied here.
        // Expected: 16 ones.
        let packed: Vec<u8> = vec![0x22; 8]; // each byte = (1,1) nibble pair
        let scales = scales_f32(&[0x38]);
        let out = dequantize_nvfp4_to_f32(&packed, &scales, 1, 16).unwrap();
        assert_eq!(out.len(), 16);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "out[{i}] = {v}, want 1.0");
        }
    }

    #[test]
    fn dequant_pack_order_high_nibble_is_even_index() {
        // One byte 0x71: high nibble = 7 (= +6.0) for K=0 (even),
        // low nibble = 1 (= +0.5) for K=1 (odd). Matches ComfyUI pack_uint4.
        // Block scale 0x38 (= 1.0). Pack rest of block with zeros.
        let mut packed = vec![0u8; 8];
        packed[0] = 0x71;
        let scales = scales_f32(&[0x38]);
        let out = dequantize_nvfp4_to_f32(&packed, &scales, 1, 16).unwrap();
        assert_eq!(out[0], 6.0, "K=0 should come from high nibble (= 6.0)");
        assert_eq!(out[1], 0.5, "K=1 should come from low nibble (= 0.5)");
        for &v in &out[2..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_applies_per_block_scale_independently() {
        // Two blocks of 16 weights. Block 0 nibble = 2 (1.0), scale 0x38 (1.0).
        // Block 1 nibble = 2 (1.0), scale 0x40 (2.0). Expect first 16 = 1.0, next 16 = 2.0.
        let packed = vec![0x22u8; 16]; // 32 weights
        let scales = scales_f32(&[0x38, 0x40]);
        let out = dequantize_nvfp4_to_f32(&packed, &scales, 1, 32).unwrap();
        for &v in &out[..16] {
            assert!((v - 1.0).abs() < 1e-6);
        }
        for &v in &out[16..32] {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_negative_nibbles_produce_negatives() {
        // Nibble 0xF = -6.0, scale = 1.0. Verify sign carries.
        let packed = vec![0xFFu8; 8];
        let scales = scales_f32(&[0x38]);
        let out = dequantize_nvfp4_to_f32(&packed, &scales, 1, 16).unwrap();
        for &v in &out {
            assert_eq!(v, -6.0);
        }
    }

    #[test]
    fn dequant_rejects_misaligned_k() {
        let res = dequantize_nvfp4_to_f32(&[0; 8], &scales_f32(&[0x38]), 1, 15);
        assert!(res.is_err(), "K=15 (not multiple of 16) must fail");
    }

    #[test]
    fn dequant_rejects_packed_size_mismatch() {
        let res = dequantize_nvfp4_to_f32(&[0; 4], &scales_f32(&[0x38]), 1, 16);
        assert!(res.is_err());
    }

    #[test]
    fn dequant_rejects_scale_size_mismatch() {
        let res = dequantize_nvfp4_to_f32(&[0; 8], &scales_f32(&[0x38, 0x38]), 1, 16);
        assert!(res.is_err());
    }

    #[test]
    fn unswizzle_round_trips_to_blocked() {
        // 1 row-block × 2 col-blocks: natural shape (128, 8). Fill with unique
        // markers so any swap shows up.
        let num_rows = 128;
        let num_cols_blocks = 8;
        let mut natural = Vec::with_capacity(num_rows * num_cols_blocks);
        for r in 0..num_rows {
            for q in 0..num_cols_blocks {
                natural.push((r * 1000 + q) as f32);
            }
        }
        let swizzled = swizzle_block_scales(&natural, num_rows, num_cols_blocks).unwrap();
        let restored = unswizzle_block_scales(&swizzled, num_rows, num_cols_blocks).unwrap();
        assert_eq!(restored, natural);
    }

    #[test]
    fn unswizzle_handles_multi_row_block() {
        // 3 row-blocks × 5 col-blocks: natural (384, 20).
        let num_rows = 384;
        let num_cols_blocks = 20;
        let mut natural = Vec::with_capacity(num_rows * num_cols_blocks);
        for r in 0..num_rows {
            for q in 0..num_cols_blocks {
                natural.push(((r as i64 * 31 + q as i64 * 7) % 1024) as f32);
            }
        }
        let swizzled = swizzle_block_scales(&natural, num_rows, num_cols_blocks).unwrap();
        let restored = unswizzle_block_scales(&swizzled, num_rows, num_cols_blocks).unwrap();
        assert_eq!(restored, natural);
    }

    #[test]
    fn unswizzle_klein9b_img_mlp_shape_round_trips() {
        // Real shape from cv:2759597's img_mlp.0: weight is [24576, 2048] →
        // num_cols_blocks = 2048/16 = 128. Pick a smaller proportional shape
        // for runtime: scale by 1/8: rows=3072 (24 row-blocks), cols=16
        // (4 col-blocks). Same divisibility class.
        let num_rows = 3072;
        let num_cols_blocks = 16;
        let mut natural = vec![0.0f32; num_rows * num_cols_blocks];
        for (i, v) in natural.iter_mut().enumerate() {
            *v = ((i * 17) % 31) as f32 - 15.0;
        }
        let swizzled = swizzle_block_scales(&natural, num_rows, num_cols_blocks).unwrap();
        let restored = unswizzle_block_scales(&swizzled, num_rows, num_cols_blocks).unwrap();
        assert_eq!(restored, natural);
    }

    #[test]
    fn unswizzle_rejects_size_mismatch() {
        let bad = vec![0.0f32; 100];
        let r = unswizzle_block_scales(&bad, 128, 8);
        assert!(r.is_err());
    }

    /// Document the precision-loss difference that motivates the sidecar
    /// tensor_scale design over baking the scale before the FP8 cast.
    ///
    /// Setup: a single block of 16 weights, all at FP4=+1.0 with FP8 block
    /// scale 1.0 — so the f32 dequant product (pre-tensor-scale) is 1.0
    /// for every element. Use `tensor_scale = 1.35e-4`, the order-of-magnitude
    /// scale observed in cv:2759597's NVFP4 layers.
    ///
    /// - Baked path: f32_product ← f32_product * tensor_scale = 1.35e-4,
    ///   then cast → F8E4M3. F8E4M3's smallest normal is `2^-6 ≈ 1.56e-2`,
    ///   smallest subnormal `≈ 1.95e-3`, so 1.35e-4 rounds to ZERO. The
    ///   round-trip produces 0.0 — total signal loss for *every* element of
    ///   *every* layer with `tensor_scale < 2e-3`. This is the textured-noise
    ///   mechanism that the previous FP8-storage attempt exhibited.
    ///
    /// - Sidecar path (current): f32_product (= 1.0) cast → F8E4M3 (= 1.0
    ///   exactly), then multiplied by the F32 tensor scale at forward time.
    ///   The cast is lossless for values at FP8's exact representable points
    ///   like 1.0, and rounds with ~1/8 relative error in the worst case.
    #[test]
    fn fp8_round_trip_zeroes_when_tensor_scale_baked_below_subnormal() {
        use candle_core::{DType, Device, Tensor};
        let dev = Device::Cpu;
        let tensor_scale = 1.35e-4_f32; // representative cv:2759597 magnitude
        let n = 16;

        // Both paths share this f32 dequant product (FP4 1.0 × block scale 1.0).
        let dequant_f32: Vec<f32> = vec![1.0; n];

        // Baked path: scale-then-cast.
        let baked: Vec<f32> = dequant_f32.iter().map(|v| v * tensor_scale).collect();
        let baked_t = Tensor::from_vec(baked, (1, n), &dev).unwrap();
        let baked_after_fp8: Vec<f32> = baked_t
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Cast undershoots FP8's smallest subnormal — every value rounds to 0.
        for (i, &v) in baked_after_fp8.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "baked path[{i}] = {v}, expected 0.0 (1.35e-4 < FP8 subnormal threshold)",
            );
        }

        // Sidecar path: cast f32_product → FP8 first, multiply by tensor_scale
        // at forward in higher precision. The FP8 cast is lossless at 1.0.
        let sidecar_t = Tensor::from_vec(dequant_f32, (1, n), &dev).unwrap();
        let sidecar_after_fp8: Vec<f32> = sidecar_t
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &v) in sidecar_after_fp8.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "sidecar path[{i}] = {v}, expected 1.0 (FP8 represents 1.0 exactly)",
            );
            // After applying the per-tensor scale at "forward" time:
            let reconstructed = v * tensor_scale;
            assert!(
                (reconstructed - tensor_scale).abs() < 1e-9,
                "sidecar reconstruction[{i}] = {reconstructed}, expected {tensor_scale}",
            );
        }
    }
}
