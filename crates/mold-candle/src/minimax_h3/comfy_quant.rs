//! Source-pinned, artifact-free Comfy quantization primitives for MiniMax H3.
//!
//! The execution contracts in this module are validated against ComfyUI
//! `a464ac33588ae182f81a090d910cfbf21e255b73` and its exact
//! `comfy-kitchen==0.2.26` dependency at
//! `255a43879fe57bbcbecfdb273b46d772b00c5a90`. They operate only on tensors
//! supplied by the caller and do not discover, open, download, or register H3
//! artifacts.
//!
//! Four representations/execution policies intentionally remain distinct:
//!
//! - The pruned DiT stores INT8 weights after a regular, normalized, 256-wide
//!   ConvRot transform and carries one F32 scale per output row. Its source
//!   reference rotates activations in the input dtype, performs dynamic
//!   per-row INT8 QDQ, and applies both F32 scales. CUDA uses the pinned
//!   INT8-to-INT32 cuBLASLt boundary; CPU and Metal stream portable F32
//!   output-row chunks without retaining a dense weight. That linear is the
//!   family-neutral [`crate::comfy_int8`] primitive (LTX-2.5's `int8-conv`
//!   packs share the format); this module re-exports it under the `H3`
//!   names its call sites and qualification tests use, with the H3 storage
//!   contract (CPU-resident, staged per forward) and the H3 rule that a
//!   biased linear takes the portable arm both unchanged.
//! - The pruned DiT's scaled FP8 matrices retain E4M3 weights plus scalar F32
//!   weight/input scales. Their reference path preserves the source QDQ order
//!   and accumulates against bounded, reconstructed F32 weight chunks.
//! - The Qwen3-VL INT8 ConvRot variant reconstructs bounded weight chunks and
//!   runs an ordinary floating-point matmul; it does not quantize activations.
//! - The Qwen3-VL NVFP4 variant stores high-nibble-first weights with
//!   swizzled FP8-E4M3 block scales and an F32 tensor scale. Its selective
//!   AWQ-style `pre_quant_scale` is applied when present. Comfy deliberately
//!   selects full-precision matrix multiplication for text encoders, so the
//!   forward dequantizes bounded weight chunks before the linear operation.
//!   The MATMUL is one arm on every backend; the DEQUANTIZATION dispatches
//!   through `select_h3_nvfp4_linear_kind` — CUDA widens the packed bytes on
//!   the device through `index_select` lookup tables, Metal and CPU keep the
//!   host scalar loop — and the two arms are bit-identical.
//!
//! These are reusable Candle operations, not runtime activation authority.
//! Compact artifacts may be downloaded independently, but execution remains
//! limited to Mold's separately qualified runtime route.

use candle::{DType, Device, Result, Tensor};
use float8::F8E4M3 as f8e4m3;
use std::sync::OnceLock;

/// The family-neutral INT8 ConvRot linear under its H3 name; see
/// [`crate::comfy_int8`].
pub use crate::comfy_int8::ComfyInt8ConvRotLinear as H3ComfyInt8ConvRotLinear;
/// The family-neutral INT8 arm selector's answer, under its H3 name.
pub use crate::comfy_int8::Int8LinearKind as H3Int8LinearKind;
#[cfg(all(test, any(feature = "h3", feature = "h3-private-uat")))]
use crate::comfy_int8::{
    accelerator_signed_widening_workspace_upper_bound,
    NATIVE_INT8_CUBLAS_WORKSPACE_BYTES as H3_NATIVE_INT8_CUBLAS_WORKSPACE_BYTES,
};
use crate::comfy_int8::{
    checked_round_up, ensure_floating, ensure_output_dtype, finish_linear, flattened_input,
    select_int8_linear_kind,
};
#[cfg(test)]
use crate::comfy_int8::{regular_hadamard, regular_hadamard_values, round_ties_even};

/// Comfy's released H3 INT8 checkpoints require this exact ConvRot group.
pub const H3_COMFY_CONVROT_GROUP_SIZE: usize = crate::comfy_int8::CONVROT_GROUP_SIZE;

/// NVFP4 uses one FP8-E4M3 scale per 16 logical values.
pub const H3_COMFY_NVFP4_BLOCK_SIZE: usize = 16;

/// Device bytes held by one [`Nvfp4DeviceTables`]: the `[256, 2]` nibble
/// table, the `[256]` F8E4M3 widening table, and the `[1, 1]` tensor scale.
const NVFP4_DEVICE_TABLE_BYTES: usize = (256 * 2 + 256 + 1) * std::mem::size_of::<f32>();

/// Bounded default for portable dequantized weight staging.
pub const H3_COMFY_PORTABLE_ROW_CHUNK: usize = crate::comfy_int8::PORTABLE_ROW_CHUNK;

/// Resolve the INT8 ConvRot arm for one H3 linear.
///
/// The H3 DiT's rule is the older, stricter one: a bias sends the whole
/// linear to the portable arm, because the native kernel folds none and
/// `forward_reference` adds a bias only on the portable path. Everything else
/// is [`select_int8_linear_kind`]'s answer — Metal always takes the portable
/// arm (the cuBLASLt kernel has no Metal twin, and unlike Qwen-Image's
/// `QMatMul` there is no candle-side Metal quantized kernel to qualify), which
/// is also why the H3 Metal tier is `CorrectnessOnly`: the portable arm
/// re-uploads and widens the packed weight chunk on every forward.
pub fn select_h3_int8_linear_kind(
    device: &Device,
    native_kernel_compiled: bool,
    has_bias: bool,
    in_features: usize,
    out_features: usize,
) -> H3Int8LinearKind {
    if has_bias {
        H3Int8LinearKind::PortableQuantizeDequantize
    } else {
        select_int8_linear_kind(device, native_kernel_compiled, in_features, out_features)
    }
}

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Which arm dequantizes one NVFP4-AWQ weight chunk.
///
/// Both arms compute exactly `E2M1[nibble] * widen(block_scale) *
/// tensor_scale`, in that association, in `f32`. They are bit-identical, not
/// merely close: `nvfp4_device_dequantize_is_bit_identical_to_the_host_loop`
/// compares `to_bits()` element for element, and #1317's layout probe made the
/// same comparison over all 115,605,504 elements of the shipped
/// `attn.qkv_proj` weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3Nvfp4LinearKind {
    /// Unpack the chunk in a host scalar loop and upload one dense `f32` row
    /// block. Traffic is 4 bytes per logical weight.
    PortableHostDequantize,
    /// Upload the packed payload bytes and the unswizzled FP8-E4M3 scale
    /// bytes and widen both on the device through `index_select` lookup
    /// tables. Traffic is 0.5 bytes per logical weight plus one scale byte per
    /// sixteen, and the scalar loop disappears entirely.
    DeviceLookupDequantize,
}

/// Resolve the NVFP4-AWQ dequantization arm for one linear.
///
/// CUDA takes the device arm; every other backend keeps the host loop. Metal
/// is deliberately excluded even though candle's Metal `index_select` accepts
/// `U32` ids: the equivalence this change rests on is a bit-for-bit comparison
/// against the host loop, and there is no Metal device in reach of CI or of
/// the machine that qualified this path, so a Metal arm would ship unmeasured.
/// Adding it is a one-line change plus a `cfg(feature = "metal")` twin of that
/// test, run on Apple hardware.
///
/// Like `select_h3_int8_linear_kind` this is a pure function of the device, so
/// a machine with no CUDA device can still pin the CUDA answer in a test.
pub fn select_h3_nvfp4_linear_kind(device: &Device) -> H3Nvfp4LinearKind {
    if device.is_cuda() {
        H3Nvfp4LinearKind::DeviceLookupDequantize
    } else {
        H3Nvfp4LinearKind::PortableHostDequantize
    }
}

/// Per-output-row-chunk device staging charge for one NVFP4-AWQ linear, as a
/// pure function of the arm and the logical shape.
///
/// This is the accounting authority: [`H3ComfyNvfp4AwqLinear`]'s methods
/// delegate to it, and a memory authority holding only shapes — no tensors —
/// calls it directly, exactly as the INT8 policy's
/// `max_dequantization_workspace_bytes` does.
///
/// [`H3Nvfp4LinearKind::PortableHostDequantize`] uploads one dense `f32` row
/// block and nothing else. [`H3Nvfp4LinearKind::DeviceLookupDequantize`]
/// uploads half a byte per weight and widens on the device, so it stages more
/// device scratch; `P` below is the chunk's PADDED element count, `rows *
/// padded_in_features`, because every intermediate is built at the padded
/// width and only the result is narrowed back to `in_features`:
///
/// | term                              | bytes                    |
/// |-----------------------------------|--------------------------|
/// | packed payload upload (`U8`)      | `P / 2`                  |
/// | widened lookup ids (`U32`)        | `2P`                     |
/// | `[N, 2]` nibble gather (`F32`)    | `4P`                     |
/// | block-scale bytes (`U8`)          | `P / 16`                 |
/// | block-scale ids (`U32`)           | `P / 4`                  |
/// | gathered block scales (`F32`)     | `P / 4`                  |
/// | block-scaled product (`F32`)      | `4P`                     |
/// | tensor-scaled result (`F32`)      | `4 * rows * in_features` |
/// | shared lookup tables              | `3076`                   |
///
/// It is the SUM of every device tensor the arm materializes, not the observed
/// concurrent peak, so it holds whatever order the caching allocator retires
/// the temporaries in — roughly `15.06 P` against the portable arm's `4 * rows
/// * in_features`. Over-charging is the correct direction for admission.
pub fn h3_nvfp4_weight_staging_bytes(
    kind: H3Nvfp4LinearKind,
    out_features: usize,
    in_features: usize,
    rows_per_chunk: usize,
) -> Result<usize> {
    if rows_per_chunk == 0 {
        candle::bail!("MiniMax H3 NVFP4 row chunk must be positive")
    }
    if out_features == 0 || in_features == 0 {
        candle::bail!("MiniMax H3 NVFP4 dimensions must be positive")
    }
    let overflow = || candle::Error::Msg("MiniMax H3 NVFP4 staging size overflows".into());
    let rows = rows_per_chunk.min(out_features);
    let logical_elements = rows.checked_mul(in_features).ok_or_else(overflow)?;
    let f32_width = std::mem::size_of::<f32>();
    let logical_bytes = logical_elements
        .checked_mul(f32_width)
        .ok_or_else(overflow)?;
    if kind == H3Nvfp4LinearKind::PortableHostDequantize {
        return Ok(logical_bytes);
    }
    let padded_in = checked_round_up(in_features, H3_COMFY_NVFP4_BLOCK_SIZE, "NVFP4 input")?;
    let padded_elements = rows.checked_mul(padded_in).ok_or_else(overflow)?;
    let packed_elements = padded_elements / 2;
    let block_count = padded_elements / H3_COMFY_NVFP4_BLOCK_SIZE;
    let u32_width = std::mem::size_of::<u32>();
    [
        Some(packed_elements),
        packed_elements.checked_mul(u32_width),
        padded_elements.checked_mul(f32_width),
        Some(block_count),
        block_count.checked_mul(u32_width),
        block_count.checked_mul(f32_width),
        padded_elements.checked_mul(f32_width),
        Some(logical_bytes),
        Some(NVFP4_DEVICE_TABLE_BYTES),
    ]
    .into_iter()
    .try_fold(0usize, |total, term| {
        total
            .checked_add(term.ok_or_else(overflow)?)
            .ok_or_else(overflow)
    })
}

/// Lookup tables shared by every chunk of one NVFP4 device dequantization.
///
/// Built once per `forward_dequantized` / `dequantize_weight` call rather than
/// once per chunk (the widest shipped projection is 100 chunks at the default
/// row block) and never cached across calls: three kilobytes of host-to-device
/// traffic per linear is cheaper than a device-keyed global with a lifetime.
struct Nvfp4DeviceTables {
    /// `[256, 2]` `f32`: row `b` is `[E2M1[b >> 4], E2M1[b & 0x0f]]`.
    ///
    /// One `index_select` over this table yields both nibbles of every packed
    /// byte in source column order, because row-major `[N, 2]` flattens to
    /// `high(0), low(0), high(1), low(1), ...` — exactly the even/odd column
    /// rule the host loop applies.
    nibbles: Tensor,
    /// `[256]` `f32` image of every F8E4M3 encoding, verbatim from
    /// [`f8e4m3_widening_table`]. The two NaN encodings are present and
    /// unreachable: the constructor refuses a weight whose scales include one.
    scales: Tensor,
    /// `[1, 1]` `f32` tensor scale, applied as its own multiply so the
    /// association matches the host loop's `(value * block) * tensor`.
    tensor_scale: Tensor,
}

/// Widen packed payload bytes into `index_select` ids.
///
/// The id dtype is **not** a free choice. `candle-kernels/src/indexing.cu:60`
/// reserves `max_value<I>()` as a zero-padding sentinel, so a 256-entry table
/// driven by `U8` ids can never return its 256th entry — and `0xff` is an
/// ordinary NVFP4 payload byte, two `-6.0` E2M1 nibbles, occurring 122,820
/// times in the shipped `attn.qkv_proj` weight alone. Casting to `U32` on the
/// device moves the sentinel to `u32::MAX`, which no widened byte can reach.
/// The stored weight stays `U8` at rest and the upload stays one byte per two
/// weights; only this transient id buffer is four bytes wide.
fn nvfp4_lookup_ids(packed: Tensor) -> Result<Tensor> {
    packed.to_dtype(DType::U32)
}

/// Widened image of all 256 F8E4M3 bit patterns.
///
/// F8E4M3 has exactly 256 encodings and every one has an exact `f32` image, so
/// this table is a lossless replacement for widening the checkpoint's scales
/// into a host-resident `f32` cache. It is built through Candle's own
/// `to_dtype` conversion — never a hand-rolled FP8 decode — so the table can
/// never drift from the conversion it replaces.
fn f8e4m3_widening_table() -> Result<&'static [f32; 256]> {
    static TABLE: OnceLock<std::result::Result<[f32; 256], String>> = OnceLock::new();
    TABLE
        .get_or_init(|| build_f8e4m3_widening_table().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|message| candle::Error::Msg(message.clone()))
}

fn build_f8e4m3_widening_table() -> Result<[f32; 256]> {
    let widened = Tensor::from_vec(
        (0..=u8::MAX).map(f8e4m3::from_bits).collect::<Vec<_>>(),
        256,
        &Device::Cpu,
    )?
    .to_dtype(DType::F32)?
    .to_vec1::<f32>()?;
    let mut table = [0.0f32; 256];
    table.copy_from_slice(&widened);
    Ok(table)
}

/// CPU-backed per-tensor FP8-E4M3 linear used by Comfy's scaled H3 DiT.
///
/// Both scales use the source convention, not its reciprocal:
///
/// ```text
/// q(x) = fp8(clamp(x / input_scale, -448, 448))
/// y = (f32(q(x)) * input_scale)
///     @ (f32(weight) * weight_scale)^T + bias
/// ```
///
/// The portable reference path deliberately executes the reconstructed
/// multiplication in F32. It establishes the exact quantize/dequantize and
/// scale ordering without claiming a qualified native FP8 kernel. Construction
/// accepts only caller-supplied tensors and never discovers or reads an H3
/// artifact.
#[derive(Clone, Debug)]
pub struct H3ComfyFp8ScaledLinear {
    weight: Tensor,
    weight_scale: f32,
    input_scale: f32,
    out_features: usize,
    in_features: usize,
}

impl H3ComfyFp8ScaledLinear {
    pub fn new(weight: Tensor, weight_scale: Tensor, input_scale: Tensor) -> Result<Self> {
        if !weight.device().is_cpu()
            || !weight_scale.device().is_cpu()
            || !input_scale.device().is_cpu()
        {
            candle::bail!("MiniMax H3 portable scaled FP8 storage must remain on CPU")
        }
        if weight.dtype() != DType::F8E4M3 {
            candle::bail!(
                "MiniMax H3 scaled FP8 weight must use F8E4M3 storage, got {:?}",
                weight.dtype()
            )
        }
        let (out_features, in_features) = weight.dims2()?;
        if out_features == 0 || in_features == 0 {
            candle::bail!("MiniMax H3 scaled FP8 weight dimensions must be positive")
        }
        let scalar = |value: &Tensor, role: &str| -> Result<f32> {
            if value.dtype() != DType::F32 || value.rank() != 0 {
                candle::bail!(
                    "MiniMax H3 scaled FP8 {role} must use an exact rank-0 F32 source tensor"
                )
            }
            let value = value.to_scalar::<f32>()?;
            if !value.is_finite() || value <= 0.0 {
                candle::bail!("MiniMax H3 scaled FP8 {role} must be finite and positive")
            }
            Ok(value)
        };
        Ok(Self {
            weight,
            weight_scale: scalar(&weight_scale, "weight scale")?,
            input_scale: scalar(&input_scale, "input scale")?,
            out_features,
            in_features,
        })
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub const fn weight_scale(&self) -> f32 {
        self.weight_scale
    }

    pub const fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Source-encoded weight and its two mandatory F32 scalar sidecars.
    pub fn encoded_weight_bytes(&self) -> Result<usize> {
        self.out_features
            .checked_mul(self.in_features)
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 scaled FP8 byte count overflows".into()))
    }

    /// Dense F32 device-weight bytes staged for one output-row chunk.
    pub fn portable_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 scaled FP8 row chunk must be positive")
        }
        rows_per_chunk
            .min(self.out_features)
            .checked_mul(self.in_features)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                candle::Error::Msg("MiniMax H3 scaled FP8 staging size overflows".into())
            })
    }

    fn quantize_dequantize_input(&self, input: &Tensor) -> Result<Tensor> {
        let input = input.to_dtype(DType::F32)?;
        let scale = Tensor::new(self.input_scale, input.device())?;
        input
            .broadcast_div(&scale)?
            .clamp(-448.0f32, 448.0f32)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&scale)
    }

    fn dequantize_rows(&self, start: usize, rows: usize, device: &Device) -> Result<Tensor> {
        self.weight
            .narrow(0, start, rows)?
            .to_dtype(DType::F32)?
            .affine(self.weight_scale as f64, 0.0)?
            .to_device(device)
    }

    /// Execute the source-defined scaled FP8 reference operation with bounded
    /// output-row weight staging and F32 accumulation.
    pub fn forward_reference(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 scaled FP8 row chunk must be positive")
        }
        let device = input.device();
        // candle has no Metal F8E4M3 cast kernel, so the round trip in
        // `quantize_dequantize_input` would fail with a raw
        // "Metal contiguous to_dtype F32 F8E4M3 not implemented" from inside a
        // linear, naming neither the checkpoint nor the fix. Refuse by name,
        // the same rule Wan applies to its fp8-scaled tiers.
        if device.is_metal() {
            candle::bail!(
                "MiniMax H3 scaled FP8 weights do not run on Metal — candle has no Metal fp8 \
                 widening kernel. Use the INT8 ConvRot or BF16 tier of this checkpoint instead."
            )
        }
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let quantized_input = self.quantize_dequantize_input(&flat)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(self.out_features - start);
            let weight = self.dequantize_rows(start, rows, device)?;
            chunks.push(quantized_input.matmul(&weight.t()?.contiguous()?)?);
        }
        finish_linear(
            chunks,
            bias,
            output_dtype,
            output_shape,
            self.out_features,
            device,
        )
    }
}

/// Gather the logical scale grid out of the swizzled tile layout.
///
/// This is index arithmetic only, so it is generic over the stored element:
/// production reads the checkpoint's raw F8E4M3 bytes, while the pinned
/// comfy-kitchen oracle tests exercise the same gather over `f32`.
fn unswizzle_nvfp4_scales<T: Copy + Default>(
    swizzled: &[T],
    logical_rows: usize,
    logical_columns: usize,
) -> Result<Vec<T>> {
    let row_blocks = logical_rows.div_ceil(128);
    let column_blocks = logical_columns.div_ceil(4);
    let padded_rows = row_blocks
        .checked_mul(128)
        .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 scale rows overflow".into()))?;
    let padded_columns = column_blocks
        .checked_mul(4)
        .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 scale columns overflow".into()))?;
    let expected = padded_rows
        .checked_mul(padded_columns)
        .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 scale size overflows".into()))?;
    if swizzled.len() != expected {
        candle::bail!(
            "MiniMax H3 NVFP4 scale storage has {} elements, expected {expected}",
            swizzled.len()
        )
    }
    let logical_elements = logical_rows.checked_mul(logical_columns).ok_or_else(|| {
        candle::Error::Msg("MiniMax H3 NVFP4 logical scale size overflows".into())
    })?;
    let mut natural = vec![T::default(); logical_elements];
    for row in 0..logical_rows {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let quarter = row_in_block / 32;
        let lane = row_in_block % 32;
        for column in 0..logical_columns {
            let column_block = column / 4;
            let column_in_block = column % 4;
            let swizzled_column = quarter * 4 + column_in_block;
            let tile = row_block * column_blocks + column_block;
            let source = tile * 512 + lane * 16 + swizzled_column;
            natural[row * logical_columns + column] = swizzled[source];
        }
    }
    Ok(natural)
}

/// CPU-backed tensorwise-INT8 embedding used by the selected H3 conditioner.
///
/// Candle has no signed-I8 tensor dtype, so an authenticated loader preserves
/// the safetensors payload byte-for-byte in U8 storage. Lookup widens each
/// selected byte through two's-complement `i8` interpretation and applies the
/// corresponding F32 row scale. Only requested token rows are materialized in
/// floating point; the complete 151,936 x 5,120 table is never dequantized.
#[derive(Clone, Debug)]
pub struct H3ComfyInt8TensorwiseEmbedding {
    weight_bytes: Tensor,
    row_scales: Tensor,
    vocabulary: usize,
    hidden_size: usize,
}

impl H3ComfyInt8TensorwiseEmbedding {
    pub fn new(weight_bytes: Tensor, row_scales: Tensor) -> Result<Self> {
        if !weight_bytes.device().is_cpu() || !row_scales.device().is_cpu() {
            candle::bail!("MiniMax H3 portable INT8 embedding storage must remain on CPU")
        }
        if weight_bytes.dtype() != DType::U8 {
            candle::bail!(
                "MiniMax H3 INT8 embedding payload must retain raw U8 bytes, got {:?}",
                weight_bytes.dtype()
            )
        }
        if row_scales.dtype() != DType::F32 {
            candle::bail!(
                "MiniMax H3 INT8 embedding row scales must be F32, got {:?}",
                row_scales.dtype()
            )
        }
        let (vocabulary, hidden_size) = weight_bytes.dims2()?;
        if vocabulary == 0 || hidden_size == 0 {
            candle::bail!("MiniMax H3 INT8 embedding dimensions must be positive")
        }
        if row_scales.dims() != [vocabulary, 1] {
            candle::bail!(
                "MiniMax H3 INT8 embedding scales must have shape [{vocabulary}, 1], got {:?}",
                row_scales.dims()
            )
        }
        let scales = row_scales.flatten_all()?.to_vec1::<f32>()?;
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            candle::bail!("MiniMax H3 INT8 embedding scales must be finite and positive")
        }
        Ok(Self {
            weight_bytes,
            row_scales,
            vocabulary,
            hidden_size,
        })
    }

    pub const fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn encoded_weight_bytes(&self) -> Result<usize> {
        self.vocabulary
            .checked_mul(self.hidden_size)
            .and_then(|bytes| {
                self.vocabulary
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|scales| bytes.checked_add(scales))
            })
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 INT8 embedding bytes overflow".into()))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        output_dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        if input_ids.dtype() != DType::U32 {
            candle::bail!(
                "MiniMax H3 INT8 embedding input ids must be U32, got {:?}",
                input_ids.dtype()
            )
        }
        let input_shape = input_ids.dims().to_vec();
        if input_shape.is_empty() {
            candle::bail!("MiniMax H3 INT8 embedding input ids must have rank at least one")
        }
        let ids = input_ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        if ids.is_empty() {
            candle::bail!("MiniMax H3 INT8 embedding input ids cannot be empty")
        }
        let scales = self.row_scales.flatten_all()?.to_vec1::<f32>()?;
        let mut rows =
            Vec::with_capacity(ids.len().checked_mul(self.hidden_size).ok_or_else(|| {
                candle::Error::Msg("MiniMax H3 INT8 embedding output size overflows".into())
            })?);
        for id in ids {
            let row = id as usize;
            if row >= self.vocabulary {
                candle::bail!(
                    "MiniMax H3 INT8 embedding token id {row} exceeds vocabulary {}",
                    self.vocabulary
                )
            }
            let bytes = self
                .weight_bytes
                .narrow(0, row, 1)?
                .flatten_all()?
                .to_vec1::<u8>()?;
            let scale = scales[row];
            rows.extend(bytes.into_iter().map(|byte| (byte as i8) as f32 * scale));
        }
        let mut output_shape = input_shape;
        output_shape.push(self.hidden_size);
        Tensor::from_vec(rows, output_shape.as_slice(), device)?.to_dtype(output_dtype)
    }
}

/// CPU-backed NVFP4 weight with H3's selective AWQ input transform.
#[derive(Clone, Debug)]
pub struct H3ComfyNvfp4AwqLinear {
    packed_weight: Tensor,
    /// The checkpoint's own one-byte F8E4M3 scales, unswizzled into the logical
    /// grid. Widening happens per lookup through [`f8e4m3_widening_table`];
    /// retaining an `f32` copy here quadrupled the host footprint of the
    /// shipped Qwen3-VL conditioner by 4.57 GB for no arithmetic difference.
    natural_block_scales: Vec<u8>,
    tensor_scale: f32,
    pre_quant_scale: Option<Tensor>,
    out_features: usize,
    in_features: usize,
    padded_out_features: usize,
    padded_in_features: usize,
}

impl H3ComfyNvfp4AwqLinear {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        packed_weight: Tensor,
        block_scales: Tensor,
        tensor_scale: Tensor,
        pre_quant_scale: Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        Self::new_with_optional_awq(
            packed_weight,
            block_scales,
            tensor_scale,
            Some(pre_quant_scale),
            out_features,
            in_features,
        )
    }

    /// Construct one published NVFP4 projection. ModelOpt AWQ smoothing is
    /// selective in the H3 Qwen artifact: `None` is the exact identity input
    /// transform used by 250 projections, while the attention output and MLP
    /// down projections provide a mandatory vector validated by the artifact
    /// loading policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_optional_awq(
        packed_weight: Tensor,
        block_scales: Tensor,
        tensor_scale: Tensor,
        pre_quant_scale: Option<Tensor>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        if !packed_weight.device().is_cpu()
            || !block_scales.device().is_cpu()
            || !tensor_scale.device().is_cpu()
            || pre_quant_scale
                .as_ref()
                .is_some_and(|scale| !scale.device().is_cpu())
        {
            candle::bail!("MiniMax H3 portable NVFP4-AWQ storage must remain on CPU")
        }
        if out_features == 0 || in_features == 0 {
            candle::bail!("MiniMax H3 NVFP4-AWQ dimensions must be positive")
        }
        if packed_weight.dtype() != DType::U8 {
            candle::bail!(
                "MiniMax H3 NVFP4 weight must use packed U8 storage, got {:?}",
                packed_weight.dtype()
            )
        }
        if block_scales.dtype() != DType::F8E4M3 {
            candle::bail!(
                "MiniMax H3 NVFP4 block scales must use F8E4M3 storage, got {:?}",
                block_scales.dtype()
            )
        }
        if tensor_scale.dtype() != DType::F32
            || !(tensor_scale.rank() == 0 || tensor_scale.dims() == [1])
        {
            candle::bail!("MiniMax H3 NVFP4 tensor scale must be F32 with source shape [] or [1]")
        }
        if let Some(pre_quant_scale) = &pre_quant_scale {
            ensure_floating(pre_quant_scale.dtype(), "NVFP4 AWQ pre_quant_scale")?;
            if pre_quant_scale.dims() != [in_features] {
                candle::bail!(
                    "MiniMax H3 NVFP4 AWQ pre_quant_scale must have shape [{in_features}], got {:?}",
                    pre_quant_scale.dims()
                )
            }
        }

        let padded_out_features = checked_round_up(out_features, 16, "NVFP4 output")?;
        let padded_in_features = checked_round_up(in_features, 16, "NVFP4 input")?;
        let packed_columns = padded_in_features / 2;
        if packed_weight.dims() != [padded_out_features, packed_columns] {
            candle::bail!(
                "MiniMax H3 NVFP4 packed weight expected shape [{padded_out_features}, {packed_columns}], got {:?}",
                packed_weight.dims()
            )
        }
        let blocks_per_row = padded_in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let scale_rows = checked_round_up(padded_out_features, 128, "NVFP4 scale output")?;
        let scale_columns = checked_round_up(blocks_per_row, 4, "NVFP4 scale input")?;
        if block_scales.dims() != [scale_rows, scale_columns] {
            candle::bail!(
                "MiniMax H3 NVFP4 block scales expected swizzled shape [{scale_rows}, {scale_columns}], got {:?}",
                block_scales.dims()
            )
        }
        let tensor_scale = tensor_scale.flatten_all()?.to_vec1::<f32>()?[0];
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            candle::bail!("MiniMax H3 NVFP4 tensor scale must be finite and positive")
        }
        if let Some(pre_quant_scale) = &pre_quant_scale {
            let awq = pre_quant_scale.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            if awq.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
                candle::bail!("MiniMax H3 AWQ input scales must be finite and positive")
            }
        }
        let swizzled = block_scales
            .flatten_all()?
            .to_vec1::<f8e4m3>()?
            .into_iter()
            .map(|scale| scale.to_bits())
            .collect::<Vec<u8>>();
        let natural_block_scales =
            unswizzle_nvfp4_scales(&swizzled, padded_out_features, blocks_per_row)?;
        // The old check widened every scale and tested it individually. The
        // byte cache reaches the identical verdict from one pass that records
        // which encodings occur followed by a scan of just those table entries,
        // so `-0.0` (0x80) is still accepted and the two NaN encodings
        // (0x7f / 0xff) are still refused.
        let table = f8e4m3_widening_table()?;
        let mut present = [false; 256];
        for bits in &natural_block_scales {
            present[*bits as usize] = true;
        }
        if present
            .iter()
            .zip(table.iter())
            .any(|(present, scale)| *present && (!scale.is_finite() || *scale < 0.0))
        {
            candle::bail!("MiniMax H3 NVFP4 block scales must be finite and nonnegative")
        }
        Ok(Self {
            packed_weight,
            natural_block_scales,
            tensor_scale,
            pre_quant_scale,
            out_features,
            in_features,
            padded_out_features,
            padded_in_features,
        })
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Source-encoded checkpoint bytes represented by the packed weight and
    /// quantization sidecars.
    ///
    /// The AWQ vector is charged at its incoming F16, BF16, or F32 dtype; it is
    /// converted to F32 only while validating or executing a forward pass. The
    /// FP8 block scales are charged at their source byte width, which is also
    /// exactly what the unswizzled host cache retains.
    pub fn encoded_weight_bytes(&self) -> Result<usize> {
        let packed = self
            .padded_out_features
            .checked_mul(self.padded_in_features / 2)
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 byte count overflows".into()))?;
        let scale_rows =
            checked_round_up(self.padded_out_features, 128, "NVFP4 encoded scale output")?;
        let scale_columns = checked_round_up(
            self.padded_in_features / H3_COMFY_NVFP4_BLOCK_SIZE,
            4,
            "NVFP4 encoded scale input",
        )?;
        let scales = scale_rows
            .checked_mul(scale_columns)
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 scale count overflows".into()))?;
        let awq_scale = self.pre_quant_scale.as_ref().map_or(Ok(0), |scale| {
            self.in_features
                .checked_mul(scale.dtype().size_in_bytes())
                .ok_or_else(|| {
                    candle::Error::Msg("MiniMax H3 NVFP4 AWQ byte count overflows".into())
                })
        })?;
        packed
            .checked_add(scales)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_add(awq_scale))
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 byte count overflows".into()))
    }

    /// Dense F32 device-weight bytes staged for one output-row chunk by
    /// [`H3Nvfp4LinearKind::PortableHostDequantize`].
    ///
    /// This deliberately excludes the already-resident packed host tensor,
    /// its unswizzled byte scale cache, and the final output tensor.
    pub fn portable_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 NVFP4 row chunk must be positive")
        }
        rows_per_chunk
            .min(self.out_features)
            .checked_mul(self.in_features)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 NVFP4 staging size overflows".into()))
    }

    /// Upper bound on the transient device bytes
    /// [`H3Nvfp4LinearKind::DeviceLookupDequantize`] materializes for one
    /// output-row chunk. See [`h3_nvfp4_weight_staging_bytes`] for the terms.
    pub fn device_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        h3_nvfp4_weight_staging_bytes(
            H3Nvfp4LinearKind::DeviceLookupDequantize,
            self.out_features,
            self.in_features,
            rows_per_chunk,
        )
    }

    /// The larger of the two arms' per-chunk staging charges.
    ///
    /// Memory authorities read this rather than either arm directly: which arm
    /// runs is a property of the device the forward lands on, and a plan that
    /// priced only the portable arm would under-charge every CUDA render.
    pub fn max_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        Ok(self
            .portable_weight_staging_bytes(rows_per_chunk)?
            .max(self.device_weight_staging_bytes(rows_per_chunk)?))
    }

    /// Build the per-call lookup tables for the device arm.
    fn device_tables(&self, device: &Device) -> Result<Nvfp4DeviceTables> {
        let mut nibbles = Vec::with_capacity(512);
        for byte in 0..=u8::MAX {
            nibbles.push(E2M1_LUT[(byte >> 4) as usize]);
            nibbles.push(E2M1_LUT[(byte & 0x0f) as usize]);
        }
        Ok(Nvfp4DeviceTables {
            nibbles: Tensor::from_vec(nibbles, (256, 2), device)?,
            scales: Tensor::from_slice(f8e4m3_widening_table()?.as_slice(), 256, device)?,
            tensor_scale: Tensor::from_vec(vec![self.tensor_scale], (1, 1), device)?,
        })
    }

    fn dequantize_rows(
        &self,
        start: usize,
        rows: usize,
        device: &Device,
        tables: Option<&Nvfp4DeviceTables>,
    ) -> Result<Tensor> {
        match tables {
            Some(tables) => self.dequantize_rows_device(start, rows, device, tables),
            None => self.dequantize_rows_host(start, rows, device),
        }
    }

    /// Widen one output-row chunk on the device.
    ///
    /// The arithmetic is the host loop's, term for term and in the same
    /// association: `E2M1[nibble]`, times the widened block scale, times the
    /// tensor scale. Every step is `f32` and every step is IEEE
    /// round-to-nearest on both backends, which is why the result is
    /// bit-identical rather than merely close.
    fn dequantize_rows_device(
        &self,
        start: usize,
        rows: usize,
        device: &Device,
        tables: &Nvfp4DeviceTables,
    ) -> Result<Tensor> {
        let packed_columns = self.padded_in_features / 2;
        let blocks_per_row = self.padded_in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        // `to_device` copies the whole underlying storage and keeps the
        // layout, so a `narrow`ed view would upload the entire weight. Take
        // the chunk's bytes on the host, which is what the layout-respecting
        // `to_vec1` does, and upload exactly those.
        let packed = self
            .packed_weight
            .narrow(0, start, rows)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let block_scales = Tensor::from_slice(
            &self.natural_block_scales[start * blocks_per_row..(start + rows) * blocks_per_row],
            rows * blocks_per_row,
            device,
        )?;
        let block_scales = tables
            .scales
            .index_select(&nvfp4_lookup_ids(block_scales)?, 0)?
            .reshape((rows, blocks_per_row, 1))?;
        let packed = Tensor::from_vec(packed, rows * packed_columns, device)?;
        let widened = tables
            .nibbles
            .index_select(&nvfp4_lookup_ids(packed)?, 0)?
            .reshape((rows, blocks_per_row, H3_COMFY_NVFP4_BLOCK_SIZE))?;
        let scaled = widened
            .broadcast_mul(&block_scales)?
            .reshape((rows, self.padded_in_features))?;
        drop(widened);
        drop(block_scales);
        let scaled = if self.padded_in_features == self.in_features {
            scaled
        } else {
            scaled.narrow(1, 0, self.in_features)?
        };
        scaled.broadcast_mul(&tables.tensor_scale)
    }

    fn dequantize_rows_host(&self, start: usize, rows: usize, device: &Device) -> Result<Tensor> {
        let table = f8e4m3_widening_table()?;
        let packed_columns = self.padded_in_features / 2;
        let blocks_per_row = self.padded_in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let packed = self
            .packed_weight
            .narrow(0, start, rows)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let output_elements = rows.checked_mul(self.in_features).ok_or_else(|| {
            candle::Error::Msg("MiniMax H3 NVFP4 dequantized chunk size overflows".into())
        })?;
        let mut output = vec![0.0; output_elements];
        for row in 0..rows {
            let packed_row = &packed[row * packed_columns..(row + 1) * packed_columns];
            let scales = &self.natural_block_scales
                [(start + row) * blocks_per_row..(start + row + 1) * blocks_per_row];
            for column in 0..self.in_features {
                let byte = packed_row[column / 2];
                let nibble = if column.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                output[row * self.in_features + column] = E2M1_LUT[nibble as usize]
                    * table[scales[column / H3_COMFY_NVFP4_BLOCK_SIZE] as usize]
                    * self.tensor_scale;
            }
        }
        Tensor::from_vec(output, (rows, self.in_features), device)
    }

    pub fn dequantize_weight(
        &self,
        output_dtype: DType,
        device: &Device,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 NVFP4 row chunk must be positive")
        }
        let tables = self.dequantize_tables(device)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(self.out_features - start);
            chunks.push(self.dequantize_rows(start, rows, device, tables.as_ref())?);
        }
        Tensor::cat(&chunks, 0)?.to_dtype(output_dtype)
    }

    /// Resolve the arm once per call and, for the device arm, build its shared
    /// lookup tables once rather than once per chunk.
    fn dequantize_tables(&self, device: &Device) -> Result<Option<Nvfp4DeviceTables>> {
        match select_h3_nvfp4_linear_kind(device) {
            H3Nvfp4LinearKind::PortableHostDequantize => Ok(None),
            H3Nvfp4LinearKind::DeviceLookupDequantize => Ok(Some(self.device_tables(device)?)),
        }
    }

    /// Comfy text encoders multiply by the ModelOpt AWQ input scale and then
    /// use a full-precision matrix multiplication over dequantized NVFP4
    /// weights. This method preserves that ordering and bounds weight staging.
    ///
    /// NVFP4 still has no native matmul kernel in mold, so the linear itself
    /// is one arm on every backend: an ordinary full-precision `f32` matmul
    /// over a dequantized weight chunk. The DEQUANTIZATION is dispatched, by
    /// `select_h3_nvfp4_linear_kind` — CUDA widens the packed bytes on the
    /// device through `index_select` lookup tables (#1317), and Metal and CPU
    /// keep the host scalar loop. Both arms are bit-identical, so this is a
    /// cost choice and never a numerical one, and the matmul dtype is
    /// unchanged.
    ///
    /// The FP8-E4M3 block scales still never reach a device AS FP8: they are
    /// unswizzled on the host at construction, retained at their source byte
    /// width, and widened either through a 256-entry host table or — on the
    /// device arm — by uploading those same bytes and gathering the identical
    /// table with `index_select`. So this path needs no Metal or CUDA fp8 cast
    /// and remains exempt from the fp8 refusal `H3ComfyFp8ScaledLinear`
    /// carries.
    ///
    /// The one trap the device arm has to dodge is candle's own: `U8`
    /// `index_select` ids cannot address a 256-entry table, because
    /// `candle-kernels/src/indexing.cu:60` reserves `u8::MAX` as a
    /// zero-padding sentinel and `0xff` is an ordinary NVFP4 payload byte. See
    /// [`nvfp4_lookup_ids`].
    pub fn forward_dequantized(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 NVFP4 row chunk must be positive")
        }
        let device = input.device();
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let rows = flat.dim(0)?;
        let scaled = flat
            .to_dtype(DType::F32)?
            .reshape((rows, self.in_features))?;
        let scaled = if let Some(pre_quant_scale) = &self.pre_quant_scale {
            let awq = pre_quant_scale
                .to_device(device)?
                .to_dtype(DType::F32)?
                .reshape((1, self.in_features))?;
            scaled.broadcast_mul(&awq)?
        } else {
            scaled
        };
        let tables = self.dequantize_tables(device)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let width = rows_per_chunk.min(self.out_features - start);
            let weight = self.dequantize_rows(start, width, device, tables.as_ref())?;
            chunks.push(scaled.matmul(&weight.t()?.contiguous()?)?);
        }
        finish_linear(
            chunks,
            bias,
            output_dtype,
            output_shape,
            self.out_features,
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float8::F8E4M3 as f8e4m3;

    fn max_error(left: &Tensor, right: &Tensor) -> Result<f32> {
        let left = left.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let right = right
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(left
            .into_iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max))
    }

    fn swizzle_scales(natural: &[f32], rows: usize, columns: usize) -> Vec<f32> {
        let row_blocks = rows.div_ceil(128);
        let column_blocks = columns.div_ceil(4);
        let mut swizzled = vec![0.0; row_blocks * 128 * column_blocks * 4];
        for row in 0..rows {
            let row_block = row / 128;
            let row_in_block = row % 128;
            let quarter = row_in_block / 32;
            let lane = row_in_block % 32;
            for column in 0..columns {
                let column_block = column / 4;
                let column_in_block = column % 4;
                let swizzled_column = quarter * 4 + column_in_block;
                let tile = row_block * column_blocks + column_block;
                swizzled[tile * 512 + lane * 16 + swizzled_column] =
                    natural[row * columns + column];
            }
        }
        swizzled
    }

    #[test]
    fn scaled_fp8_reference_applies_exact_source_scale_order() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![1.0f32, 2.0, -1.0, 0.5, -2.0, 0.5, 1.0, 4.0],
            (2, 4),
            &device,
        )?
        .to_dtype(DType::F8E4M3)?;
        let linear = H3ComfyFp8ScaledLinear::new(
            weight,
            Tensor::new(0.25f32, &device)?,
            Tensor::new(0.5f32, &device)?,
        )?;
        let input = Tensor::from_vec(vec![0.5f32, 1.0, -0.5, 0.25], (1, 4), &device)?;
        let bias = Tensor::from_vec(vec![0.125f32, -0.25], 2, &device)?;
        let output = linear.forward_reference(&input, Some(&bias), DType::F32, 1)?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![0.90625, -0.25]]);
        assert_eq!(linear.weight_scale(), 0.25);
        assert_eq!(linear.input_scale(), 0.5);
        assert_eq!(linear.encoded_weight_bytes()?, 16);
        assert_eq!(linear.portable_weight_staging_bytes(1)?, 16);
        Ok(())
    }

    /// The INT8 arm is chosen from the device, the compiled kernel, and the
    /// weight's own shape. Metal and CPU can never reach the CUDA kernel, and
    /// a build without it never reaches it either — which is what makes the
    /// portable arm the honest description of the Metal tier.
    #[test]
    fn the_int8_arm_is_native_only_for_a_compiled_cuda_kernel() {
        // Metal and CPU take the portable arm however the binary was built.
        for compiled in [false, true] {
            assert_eq!(
                select_h3_int8_linear_kind(&Device::Cpu, compiled, false, 256, 512),
                H3Int8LinearKind::PortableQuantizeDequantize
            );
        }
        // A build that omitted the kernel must not claim it even on CUDA.
        assert_eq!(
            select_h3_int8_linear_kind(&Device::Cpu, false, false, 256, 512),
            H3Int8LinearKind::PortableQuantizeDequantize
        );
    }

    /// A Metal device must take the portable arm, and the selector must say so
    /// without needing one — the same reason `zimage`'s `select_linear_kind` is
    /// a pure function.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_never_selects_the_native_int8_kernel() {
        let metal = Device::new_metal(0).unwrap();
        assert_eq!(
            select_h3_int8_linear_kind(&metal, true, false, 256, 512),
            H3Int8LinearKind::PortableQuantizeDequantize
        );
    }

    /// candle has no Metal fp8 widening kernel, so an fp8-scaled linear is
    /// refused by name rather than erroring from inside the cast.
    #[cfg(feature = "metal")]
    #[test]
    fn an_fp8_scaled_linear_is_refused_on_metal_by_name() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], (1, 4), &device)?
            .to_dtype(DType::F8E4M3)?;
        let linear = H3ComfyFp8ScaledLinear::new(
            weight,
            Tensor::new(0.25f32, &device)?,
            Tensor::new(0.5f32, &device)?,
        )?;
        let metal = Device::new_metal(0).unwrap();
        let input = Tensor::from_vec(vec![0.5f32, 1.0, -0.5, 0.25], (1, 4), &metal)?;
        let error = linear
            .forward_reference(&input, None, DType::F32, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Metal"), "{error}");
        assert!(error.contains("fp8"), "{error}");
        assert!(error.contains("INT8 ConvRot or BF16"), "{error}");
        Ok(())
    }

    #[test]
    fn scaled_fp8_reference_clamps_before_cast_and_reapplies_input_scale() -> Result<()> {
        let device = Device::Cpu;
        let weight =
            Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device)?.to_dtype(DType::F8E4M3)?;
        let linear = H3ComfyFp8ScaledLinear::new(
            weight,
            Tensor::new(1.0f32, &device)?,
            Tensor::new(0.5f32, &device)?,
        )?;
        let input = Tensor::from_vec(vec![250.0f32, -250.0], (1, 2), &device)?;
        assert_eq!(
            linear
                .forward_reference(&input, None, DType::F32, 1)?
                .to_vec2::<f32>()?,
            vec![vec![224.0]]
        );
        Ok(())
    }

    #[test]
    fn scaled_fp8_rejects_implicit_or_invalid_dtype_and_scale_contracts() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((2, 4), DType::F32, &device)?.to_dtype(DType::F8E4M3)?;
        let scalar = Tensor::new(0.5f32, &device)?;
        let rank_one = Tensor::from_vec(vec![0.5f32], 1, &device)?;
        assert!(
            H3ComfyFp8ScaledLinear::new(weight.clone(), rank_one, scalar.clone())
                .unwrap_err()
                .to_string()
                .contains("rank-0 F32")
        );
        let zero = Tensor::new(0.0f32, &device)?;
        assert!(
            H3ComfyFp8ScaledLinear::new(weight.clone(), scalar.clone(), zero)
                .unwrap_err()
                .to_string()
                .contains("finite and positive")
        );
        for invalid in [-1.0f32, f32::INFINITY, f32::NAN] {
            assert!(H3ComfyFp8ScaledLinear::new(
                weight.clone(),
                scalar.clone(),
                Tensor::new(invalid, &device)?,
            )
            .unwrap_err()
            .to_string()
            .contains("finite and positive"));
        }
        let half_scale = scalar.to_dtype(DType::F16)?;
        assert!(H3ComfyFp8ScaledLinear::new(weight, half_scale, scalar)
            .unwrap_err()
            .to_string()
            .contains("rank-0 F32"));
        let dense = Tensor::ones((2, 4), DType::F32, &device)?;
        assert!(H3ComfyFp8ScaledLinear::new(
            dense,
            Tensor::new(1.0f32, &device)?,
            Tensor::new(1.0f32, &device)?,
        )
        .unwrap_err()
        .to_string()
        .contains("F8E4M3"));
        Ok(())
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn synthetic_int8_forward(device: &Device) -> Result<Tensor> {
        let columns = H3_COMFY_CONVROT_GROUP_SIZE;
        let weight_values = (0..2 * columns)
            .map(|index| (((index * 19 + 7) % 29) as i8 - 14) as u8)
            .collect::<Vec<_>>();
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(weight_values, (2, columns), &Device::Cpu)?,
            Tensor::from_vec(vec![0.125f32, 0.25], (2, 1), &Device::Cpu)?,
        )?;
        let input_values = (0..2 * columns)
            .map(|index| (index as f32 % 17.0 - 8.0) / 5.0)
            .collect::<Vec<_>>();
        linear
            .forward_reference(
                &Tensor::from_vec(input_values, (2, columns), device)?,
                None,
                DType::F32,
                1,
            )?
            .to_device(&Device::Cpu)
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn synthetic_nvfp4_forward(device: &Device) -> Result<Tensor> {
        let out_features = 2;
        let in_features = 16;
        let padded_rows = 16;
        let mut packed = vec![0u8; padded_rows * in_features / 2];
        packed[0] = 0x71;
        packed[in_features / 2] = 0xaf;
        let natural_scales = vec![1.0f32; padded_rows];
        let block_scales = Tensor::from_vec(
            swizzle_scales(&natural_scales, padded_rows, 1),
            (128, 4),
            &Device::Cpu,
        )?
        .to_dtype(DType::F8E4M3)?;
        let linear = H3ComfyNvfp4AwqLinear::new(
            Tensor::from_vec(packed, (padded_rows, in_features / 2), &Device::Cpu)?,
            block_scales,
            Tensor::new(0.5f32, &Device::Cpu)?,
            Tensor::from_vec(
                (0..in_features)
                    .map(|column| if column == 0 { 2.0f32 } else { 1.0f32 })
                    .collect(),
                in_features,
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?,
            out_features,
            in_features,
        )?;
        linear
            .forward_dequantized(
                &Tensor::from_vec(vec![1.0f32; in_features], (1, in_features), device)?,
                None,
                DType::F32,
                1,
            )?
            .to_device(&Device::Cpu)
    }

    #[test]
    fn regular_hadamard_matches_pinned_comfy_kitchen_seed_and_kron_order() -> Result<()> {
        let h4 = regular_hadamard_values(4)?;
        assert_eq!(
            h4,
            vec![
                0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5,
            ]
        );

        let h16 = regular_hadamard_values(16)?;
        let row0 = [
            1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
        ]
        .map(|value| value * 0.25);
        let row5 = [
            1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
        ]
        .map(|value| value * 0.25);
        assert_eq!(&h16[..16], &row0);
        assert_eq!(&h16[5 * 16..6 * 16], &row5);
        Ok(())
    }

    #[test]
    fn regular_hadamard_is_symmetric_and_orthonormal() -> Result<()> {
        let device = Device::Cpu;
        let hadamard = regular_hadamard(&device)?;
        assert_eq!(max_error(&hadamard, &hadamard.t()?)?, 0.0);
        let identity = hadamard.matmul(&hadamard.t()?.contiguous()?)?;
        let expected = Tensor::eye(H3_COMFY_CONVROT_GROUP_SIZE, DType::F32, &device)?;
        assert!(max_error(&identity, &expected)? <= 1e-6);
        Ok(())
    }

    #[test]
    fn ties_to_even_rounding_matches_pytorch_half_step_contract() -> Result<()> {
        let input = Tensor::new(
            &[-3.5f32, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
            &Device::Cpu,
        )?;
        let expected = vec![-4.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0];
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(
                round_ties_even(&input.to_dtype(dtype)?)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                expected,
                "ties-to-even mismatch for {dtype:?}"
            );
        }
        assert_ne!(
            input.round()?.to_vec1::<f32>()?,
            round_ties_even(&input)?.to_vec1::<f32>()?,
            "the regression must distinguish Candle's half-away-from-zero primitive"
        );
        Ok(())
    }

    #[test]
    fn int8_convrot_forward_matches_explicit_dequantized_weight() -> Result<()> {
        let device = Device::Cpu;
        let rows = 3;
        let columns = H3_COMFY_CONVROT_GROUP_SIZE * 2;
        let quantized = (0..rows * columns)
            .map(|index| (((index * 37 + 11) % 31) as i8 - 15) as u8)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(quantized, (rows, columns), &device)?;
        let scale = Tensor::from_vec(vec![0.125f32, 0.25, 0.5], (rows, 1), &device)?;
        let linear = H3ComfyInt8ConvRotLinear::new(weight, scale)?;
        let input = Tensor::from_vec(
            (0..2 * columns)
                .map(|index| (index as f32 % 23.0 - 11.0) / 8.0)
                .collect(),
            (1, 2, columns),
            &device,
        )?;
        let bias = Tensor::from_vec(vec![0.5f32, -0.25, 1.0], rows, &device)?;
        let actual = linear.forward_dequantized(&input, Some(&bias), DType::F32, 2)?;
        let dense = linear.dequantize_weight(DType::F32, &device, 1)?;
        let expected = input
            .reshape((2, columns))?
            .matmul(&dense.t()?.contiguous()?)?
            .broadcast_add(&bias.reshape((1, rows))?)?
            .reshape((1, 2, rows))?;
        assert!(max_error(&actual, &expected)? <= 2e-4);
        assert_eq!(linear.encoded_weight_bytes()?, rows * columns + rows * 4);
        assert_eq!(linear.portable_weight_staging_bytes(2)?, 2 * columns * 4);
        assert!(linear.portable_weight_staging_bytes(0).is_err());
        assert!(linear
            .forward_dequantized(&input, Some(&bias), DType::F32, 0)
            .is_err());
        Ok(())
    }

    #[test]
    fn int8_convrot_reference_matches_comfy_dynamic_row_qdq_order() -> Result<()> {
        let device = Device::Cpu;
        let rows = 3;
        let columns = H3_COMFY_CONVROT_GROUP_SIZE;
        let raw = (0..rows * columns)
            .map(|index| (((index * 29 + 5) % 37) as i8 - 18).to_ne_bytes()[0])
            .collect::<Vec<_>>();
        let scales = Tensor::from_vec(vec![0.015625f32, 0.03125, 0.0625], (rows, 1), &device)?;
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(raw, (rows, columns), &device)?,
            scales.clone(),
        )?;
        let input = Tensor::from_vec(
            (0..2 * columns)
                .map(|index| ((index * 13 % 97) as f32 - 48.0) / 31.0)
                .collect::<Vec<_>>(),
            (2, columns),
            &device,
        )?;
        let actual = linear.forward_reference(&input, None, DType::F32, 2)?;

        let rotated = input
            .reshape((2, columns))?
            .matmul(&regular_hadamard(&device)?)?
            .reshape((2, columns))?;
        let input_scale = rotated
            .abs()?
            .max_keepdim(1)?
            .affine(1.0 / 127.0, 0.0)?
            .clamp(1e-30f32, f32::MAX)?;
        let scaled = rotated.broadcast_div(&input_scale)?;
        let quantized = round_ties_even(&scaled)?.clamp(-128.0f32, 127.0f32)?;
        let signed = linear.signed_rows(0, rows, &device)?;
        let expected = quantized
            .matmul(&signed.t()?.contiguous()?)?
            .broadcast_mul(&input_scale.broadcast_mul(&scales.t()?)?)?;
        assert_eq!(max_error(&actual, &expected)?, 0.0);
        assert!(
            max_error(
                &actual,
                &linear.forward_dequantized(&input, None, DType::F32, 2)?
            )? > 0.0,
            "dynamic activation quantization must remain distinct from dense dequantization"
        );
        Ok(())
    }

    #[test]
    fn qwen_int8_weight_only_forward_dequantizes_before_float_matmul() -> Result<()> {
        let device = Device::Cpu;
        let rows = 3;
        let columns = H3_COMFY_CONVROT_GROUP_SIZE;
        let quantized = (0..rows * columns)
            .map(|index| (((index * 17 + 3) % 23) as i8 - 11) as u8)
            .collect::<Vec<_>>();
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(quantized, (rows, columns), &device)?,
            Tensor::from_vec(vec![0.03125f32, 0.0625, 0.125], (rows, 1), &device)?,
        )?;
        let input = Tensor::from_vec(
            (0..2 * columns)
                .map(|index| (index as f32 % 19.0 - 9.0) / 16.0)
                .collect(),
            (1, 2, columns),
            &device,
        )?;
        let bias = Tensor::from_vec(vec![0.125f32, -0.25, 0.5], rows, &device)?;
        let actual = linear.forward_weight_only(&input, Some(&bias), DType::F32, 2)?;
        let dense = linear.dequantize_weight(DType::F32, &device, 1)?;
        let expected = input
            .reshape((2, columns))?
            .matmul(&dense.t()?.contiguous()?)?
            .broadcast_add(&bias.reshape((1, rows))?)?
            .reshape((1, 2, rows))?;
        assert!(max_error(&actual, &expected)? <= 2e-5);

        let mut perturbed = input.flatten_all()?.to_vec1::<f32>()?;
        perturbed[0] += 1.0 / 65_536.0;
        let perturbed = Tensor::from_vec(perturbed, (1, 2, columns), &device)?;
        let changed = linear.forward_weight_only(&perturbed, Some(&bias), DType::F32, 2)?;
        assert!(max_error(&actual, &changed)? > 0.0);
        Ok(())
    }

    #[test]
    fn int8_convrot_rejects_scalar_and_flat_row_scales() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::zeros((2, H3_COMFY_CONVROT_GROUP_SIZE), DType::U8, &device)?;
        let scalar = Tensor::new(0.5f32, &device)?;
        assert!(H3ComfyInt8ConvRotLinear::new(weight.clone(), scalar)
            .unwrap_err()
            .to_string()
            .contains("[2, 1]"));
        let flat = Tensor::new(&[0.5f32, 0.25], &device)?;
        assert!(H3ComfyInt8ConvRotLinear::new(weight, flat)
            .unwrap_err()
            .to_string()
            .contains("[2, 1]"));
        let weight = Tensor::zeros((2, H3_COMFY_CONVROT_GROUP_SIZE), DType::U8, &device)?;
        let nonpositive = Tensor::from_vec(vec![0.5f32, 0.0], (2, 1), &device)?;
        assert!(H3ComfyInt8ConvRotLinear::new(weight, nonpositive)
            .unwrap_err()
            .to_string()
            .contains("finite and positive"));
        Ok(())
    }

    #[test]
    fn int8_convrot_widens_raw_twos_complement_bytes_as_signed() -> Result<()> {
        assert_signed_rows(&Device::Cpu)
    }

    #[cfg(feature = "h3-private-uat")]
    #[test]
    fn int8_convrot_workspace_charges_every_device_widening_intermediate() -> Result<()> {
        let elements = 17 * H3_COMFY_CONVROT_GROUP_SIZE;
        assert_eq!(
            accelerator_signed_widening_workspace_upper_bound(elements)?,
            u64::try_from(
                elements * (2 * std::mem::size_of::<u8>() + 4 * std::mem::size_of::<f32>())
            )
            .unwrap()
        );
        Ok(())
    }

    #[cfg(feature = "h3-private-uat")]
    #[test]
    fn int8_convrot_workspace_prices_native_cuda_lifetimes() -> Result<()> {
        let outputs = 16;
        let input_rows = 3;
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::zeros(
                (outputs, H3_COMFY_CONVROT_GROUP_SIZE),
                DType::U8,
                &Device::Cpu,
            )?,
            Tensor::ones((outputs, 1), DType::F32, &Device::Cpu)?,
        )?;
        let hadamard_f32 = H3_COMFY_CONVROT_GROUP_SIZE.pow(2) * 4;
        let hadamard_bf16 = H3_COMFY_CONVROT_GROUP_SIZE.pow(2) * 2;
        let rotated_bf16 = input_rows * H3_COMFY_CONVROT_GROUP_SIZE * 2;
        let packed_weight = outputs * H3_COMFY_CONVROT_GROUP_SIZE;
        let weight_scales = outputs * 4;
        let quantized_input = input_rows * H3_COMFY_CONVROT_GROUP_SIZE;
        let input_scales = input_rows * 4;
        let accumulator = input_rows * outputs * 4;
        let output = input_rows * outputs * 2;
        let expected = hadamard_f32
            + hadamard_bf16
            + rotated_bf16
            + packed_weight
            + weight_scales
            + quantized_input
            + input_scales
            + accumulator
            + H3_NATIVE_INT8_CUBLAS_WORKSPACE_BYTES
            + output;
        assert_eq!(
            linear.reference_workspace_upper_bound(
                input_rows,
                DType::BF16,
                DType::BF16,
                H3_COMFY_PORTABLE_ROW_CHUNK,
                false,
                true,
            )?,
            expected as u64
        );
        assert!(linear
            .reference_workspace_upper_bound(
                input_rows,
                DType::BF16,
                DType::BF16,
                H3_COMFY_PORTABLE_ROW_CHUNK,
                true,
                true,
            )
            .is_err());
        Ok(())
    }

    fn assert_signed_rows(device: &Device) -> Result<()> {
        let mut bytes = vec![0u8; H3_COMFY_CONVROT_GROUP_SIZE];
        bytes[..4].copy_from_slice(&[0x80, 0xff, 0x00, 0x7f]);
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(bytes, (1, H3_COMFY_CONVROT_GROUP_SIZE), &Device::Cpu)?,
            Tensor::ones((1, 1), DType::F32, &Device::Cpu)?,
        )?;
        let signed = linear.signed_rows(0, 1, device)?.to_vec2::<f32>()?;
        assert_eq!(&signed[0][..4], &[-128.0, -1.0, 0.0, 127.0]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn int8_convrot_device_staging_preserves_signed_bytes_on_metal() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        assert_signed_rows(&device)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn int8_convrot_device_staging_preserves_signed_bytes_on_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        assert_signed_rows(&device)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn int8_convrot_native_cuda_matches_portable_reference() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let columns = H3_COMFY_CONVROT_GROUP_SIZE;
        let outputs = 8;
        let raw = (0..outputs * columns)
            .map(|index| (((index * 37 + 11) % 251) as i16 - 125) as i8 as u8)
            .collect::<Vec<_>>();
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(raw, (outputs, columns), &cpu)?,
            Tensor::from_vec(
                (0..outputs)
                    .map(|index| (index + 1) as f32 / 256.0)
                    .collect::<Vec<_>>(),
                (outputs, 1),
                &cpu,
            )?,
        )?;
        let values = (0..3 * columns)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 37.0)
            .collect::<Vec<_>>();
        let expected = linear.forward_reference(
            &Tensor::from_vec(values.clone(), (3, columns), &cpu)?,
            None,
            DType::F32,
            4,
        )?;
        let actual = linear
            .forward_reference(
                &Tensor::from_vec(values.clone(), (3, columns), &cuda)?,
                None,
                DType::F32,
                4,
            )?
            .to_device(&cpu)?;
        assert!(max_error(&actual, &expected)? <= 1e-4);

        // Exact BF16 values from the source-pinned PyTorch/Comfy operation:
        // BF16 ConvRot, rowwise QDQ, signed INT8 accumulation, then ordered
        // F32 activation-scale and weight-scale multiplication.
        let expected_bf16 = vec![
            -8.8125, 11.5625, 9.9375, -17.75, -35.0, -29.125, -4.71875, 11.75, -8.25, 14.125,
            -1.4453125, -23.375, -51.25, 43.75, -3.921875, -16.375, -14.4375, 8.375, 38.25, -59.25,
            -58.25, 39.5, 56.25, -84.5,
        ];
        let actual_bf16 = linear
            .forward_reference(
                &Tensor::from_vec(values, (3, columns), &cuda)?.to_dtype(DType::BF16)?,
                None,
                DType::BF16,
                4,
            )?
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual_bf16, expected_bf16);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn portable_quantized_forwards_match_metal() -> Result<()> {
        let Ok(metal) = Device::new_metal(0) else {
            return Ok(());
        };
        assert!(
            max_error(
                &synthetic_int8_forward(&Device::Cpu)?,
                &synthetic_int8_forward(&metal)?
            )? <= 2e-4
        );
        assert!(
            max_error(
                &synthetic_nvfp4_forward(&Device::Cpu)?,
                &synthetic_nvfp4_forward(&metal)?
            )? <= 2e-4
        );
        Ok(())
    }

    /// Every F8E4M3 class the constructor admits, one of each: `+0`, `-0`, the
    /// two smallest subnormals, an exact power of two, `1.0`, and the maximum
    /// finite encoding `0x7e` (448.0). The two NaN encodings (`0x7f`, `0xff`)
    /// are deliberately absent — the constructor refuses a weight carrying
    /// one, which is also why the device arm's 256-entry scale table can hold
    /// them unreachably.
    const NVFP4_SCALE_BIT_CLASSES: [u8; 7] = [0x00, 0x80, 0x01, 0x02, 0x38, 0x3c, 0x7e];

    /// A weight whose LIVE packed payload — the first `out_features` rows,
    /// the only ones any arm reads — covers `0x00`, `0xff`, `0x0f`, and
    /// `0xf0`, with `0xff` present in both the first chunk and the trailing
    /// one. `out_features` (5) and `in_features` (40) are both off the padding
    /// grid, so a trailing short chunk and the `padded_in_features` narrow
    /// both run.
    fn nvfp4_byte_coverage_linear(device: &Device) -> Result<H3ComfyNvfp4AwqLinear> {
        let out_features = 5;
        let in_features = 40;
        let padded_out = 16;
        let padded_in = 48;
        let packed_columns = padded_in / 2;
        let blocks_per_row = padded_in / H3_COMFY_NVFP4_BLOCK_SIZE;
        let mut packed: Vec<u8> = (0..padded_out * packed_columns)
            .map(|index| ((index * 13) % 256) as u8)
            .collect();
        packed[0] = 0x00;
        packed[1] = 0xff;
        packed[2] = 0x0f;
        packed[3] = 0xf0;
        // Also in the trailing one-row chunk, so a chunked run cannot pass by
        // never reaching the sentinel after the first block.
        packed[(out_features - 1) * packed_columns + 5] = 0xff;
        // Only the first `out_features` rows are ever dequantized; the padded
        // tail is never read, so coverage has to be asserted over the live
        // region or a U8-id regression slips through.
        let live = &packed[..out_features * packed_columns];
        for byte in [0x00u8, 0xff, 0x0f, 0xf0] {
            assert!(live.contains(&byte), "live rows must cover {byte:#04x}");
        }
        let packed = Tensor::from_vec(packed, (padded_out, packed_columns), device)?;
        let natural: Vec<u8> = (0..padded_out * blocks_per_row)
            .map(|index| NVFP4_SCALE_BIT_CLASSES[index % NVFP4_SCALE_BIT_CLASSES.len()])
            .collect();
        let swizzled = swizzle_scale_bits(&natural, padded_out, blocks_per_row);
        let block_scales = Tensor::from_vec(
            swizzled
                .into_iter()
                .map(f8e4m3::from_bits)
                .collect::<Vec<_>>(),
            (128, 4),
            device,
        )?;
        let awq = Tensor::from_vec(
            (0..in_features)
                .map(|column| 0.75 + (column as f32) / 64.0)
                .collect::<Vec<f32>>(),
            in_features,
            device,
        )?;
        H3ComfyNvfp4AwqLinear::new(
            packed,
            block_scales,
            Tensor::new(0.123_456_79f32, device)?,
            awq,
            out_features,
            in_features,
        )
    }

    #[test]
    fn nvfp4_device_nibble_table_reproduces_the_host_lookup_order() -> Result<()> {
        let linear = nvfp4_byte_coverage_linear(&Device::Cpu)?;
        let tables = linear.device_tables(&Device::Cpu)?;
        let nibbles = tables.nibbles.to_vec2::<f32>()?;
        assert_eq!(nibbles.len(), 256);
        for byte in 0..=u8::MAX {
            // The host loop takes the HIGH nibble for even logical columns and
            // the low nibble for odd ones; `[N, 2]` flattens in that order.
            assert_eq!(
                nibbles[byte as usize],
                vec![
                    E2M1_LUT[(byte >> 4) as usize],
                    E2M1_LUT[(byte & 0x0f) as usize]
                ],
                "nibble table row {byte:#04x}"
            );
        }
        // Signed zero survives the table, so the arm cannot be reformulated as
        // an `affine` that adds 0.0 to the product.
        assert!(nibbles[0x80][0].is_sign_negative());
        assert_eq!(
            tables.scales.to_vec1::<f32>()?[..0x7f],
            f8e4m3_widening_table()?[..0x7f]
        );
        assert_eq!(tables.tensor_scale.dims(), [1, 1]);
        Ok(())
    }

    #[test]
    fn nvfp4_lookup_ids_escape_the_u8_index_select_sentinel() -> Result<()> {
        // `candle-kernels/src/indexing.cu:60` zeroes any id equal to
        // `max_value<I>()`. `0xff` is an ordinary payload byte — two -6.0
        // nibbles — so a U8 id would silently decode it to 0.0.
        let sentinel = u8::MAX;
        assert_eq!(E2M1_LUT[(sentinel >> 4) as usize], -6.0);
        assert_eq!(E2M1_LUT[(sentinel & 0x0f) as usize], -6.0);
        let packed = Tensor::from_vec(vec![0x00u8, 0x0f, 0xf0, 0xff], 4, &Device::Cpu)?;
        let ids = nvfp4_lookup_ids(packed)?;
        assert_eq!(ids.dtype(), DType::U32);
        assert_eq!(ids.to_vec1::<u32>()?, vec![0, 15, 240, 255]);
        Ok(())
    }

    /// Why the `U8`-id alternative has to be a SELECT and not an addition.
    ///
    /// #1317's part-1 record left "keep `U8` ids and add a `packed == 0xff`
    /// correction" open as the cheaper option. It is not merely cheaper — it
    /// is WRONG: `index_select` zeroes the sentinel, so every other byte needs
    /// a `+0.0`, and `-0.0 + 0.0` is `+0.0` in every IEEE rounding mode. E2M1
    /// entry 8 IS `-0.0`, so an additive repair flips the sign of every
    /// negative zero in the weight. That is not caught by any tolerance test,
    /// which is exactly the failure mode the bit-for-bit gate exists for.
    #[test]
    fn nvfp4_u8_id_repair_by_addition_loses_negative_zero() {
        let negative_zero = E2M1_LUT[8];
        assert!(negative_zero.is_sign_negative() && negative_zero == 0.0);
        // What an additive repair does to a byte that is not the sentinel.
        assert_eq!((negative_zero + 0.0f32).to_bits(), 0.0f32.to_bits());
        assert_ne!((negative_zero + 0.0f32).to_bits(), negative_zero.to_bits());
        // The `U32` cast never needs a repair term at all.
        assert!(u32::from(u8::MAX) < u32::MAX);
    }

    /// The device-arm gate is only as good as its fixture, and both halves of
    /// what it must cover are invisible in the assertion itself. Pin them on
    /// the CPU so a fixture change that quietly removes either fails here
    /// rather than by silently weakening a CUDA-only test.
    #[test]
    fn nvfp4_byte_coverage_fixture_reaches_both_sentinel_and_signed_zero() -> Result<()> {
        let linear = nvfp4_byte_coverage_linear(&Device::Cpu)?;
        let dense = linear
            .dequantize_weight(DType::F32, &Device::Cpu, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // A `0xff` payload byte, whose two `-6.0` nibbles are what a `U8` id
        // would silently zero.
        assert!(dense.iter().any(|value| *value < 0.0));
        // A signed zero, which an additive `0xff` repair would flip.
        assert!(dense
            .iter()
            .any(|value| *value == 0.0 && value.is_sign_negative()));
        assert!(dense.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn nvfp4_linear_kind_is_device_only_on_cuda() {
        assert_eq!(
            select_h3_nvfp4_linear_kind(&Device::Cpu),
            H3Nvfp4LinearKind::PortableHostDequantize
        );
    }

    #[test]
    fn nvfp4_device_staging_upper_bounds_the_portable_charge() -> Result<()> {
        let linear = nvfp4_byte_coverage_linear(&Device::Cpu)?;
        // rows = 2, in = 40, padded_in = 48 => P = 96, logical = 80.
        // 48 + 192 + 384 + 6 + 24 + 24 + 384 + 320 + 3076.
        assert_eq!(linear.device_weight_staging_bytes(2)?, 4_458);
        assert_eq!(linear.portable_weight_staging_bytes(2)?, 320);
        assert_eq!(linear.max_weight_staging_bytes(2)?, 4_458);
        // The row chunk saturates at `out_features`, on both arms.
        assert_eq!(
            linear.device_weight_staging_bytes(usize::MAX)?,
            linear.device_weight_staging_bytes(5)?
        );
        assert!(linear.device_weight_staging_bytes(0).is_err());
        assert!(linear.max_weight_staging_bytes(0).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_device_dequantize_is_bit_identical_to_the_host_loop() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        assert_eq!(
            select_h3_nvfp4_linear_kind(&cuda),
            H3Nvfp4LinearKind::DeviceLookupDequantize
        );
        let host = nvfp4_byte_coverage_linear(&Device::Cpu)?;
        // `dequantize_weight`'s device argument selects the arm, so this is a
        // genuine arm-versus-arm comparison and not a device round trip.
        for rows_per_chunk in [1, 2, 5, 64] {
            let expected = host
                .dequantize_weight(DType::F32, &Device::Cpu, rows_per_chunk)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = host
                .dequantize_weight(DType::F32, &cuda, rows_per_chunk)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(expected.len(), 5 * 40);
            let mismatched = expected
                .iter()
                .zip(&actual)
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            assert_eq!(
                mismatched,
                0,
                "chunk {rows_per_chunk}: {mismatched} of {} elements differ in bits",
                expected.len()
            );
        }
        // The forward keeps today's F32 matmul, so its own residual is the
        // ordinary CPU-versus-cuBLAS accumulation-order difference, bounded by
        // the same 2e-4 `portable_quantized_forwards_match_cuda` uses.
        let input = Tensor::from_vec(
            (0..3 * 40)
                .map(|index| (index as f32) / 37.0 - 1.0)
                .collect::<Vec<f32>>(),
            (3, 40),
            &Device::Cpu,
        )?;
        let cpu = host.forward_dequantized(&input, None, DType::F32, 2)?;
        let gpu = host.forward_dequantized(&input.to_device(&cuda)?, None, DType::F32, 2)?;
        assert!(max_error(&cpu, &gpu.to_device(&Device::Cpu)?)? <= 2e-4);
        Ok(())
    }

    /// Time both NVFP4 dequantization arms on one CUDA device, at #1317's
    /// probe shape, driving the same chunk loop and the same F32 matmul that
    /// `forward_dequantized` runs. Ignored by default: it allocates ~600 MB of
    /// VRAM and takes several seconds.
    ///
    /// `cargo test -p mold-ai-candle --features cuda --release --
    ///  nvfp4_device_dequantize_benchmark --ignored --nocapture`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "measures a CUDA device; run explicitly"]
    fn nvfp4_device_dequantize_benchmark() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        // #1317's probe shape, then the widest released Qwen conditioner
        // projection (`mlp.down_proj`) at the sequence an FL2VA render
        // actually encodes: 2,033 output-text rows plus 4,032 vision rows, the
        // pair the `qwen_activation_workspace_bytes` grant was measured over.
        for (out_features, in_features, activation_rows) in [
            (21_504usize, 5_376usize, 4_096usize),
            (5_120, 25_600, 6_065),
        ] {
            nvfp4_benchmark_shape(&cuda, out_features, in_features, activation_rows)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_benchmark_shape(
        cuda: &Device,
        out_features: usize,
        in_features: usize,
        activation_rows: usize,
    ) -> Result<()> {
        use std::time::Instant;

        let packed_columns = in_features / 2;
        let blocks_per_row = in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let packed: Vec<u8> = (0..out_features * packed_columns)
            .map(|index| ((index * 13) % 256) as u8)
            .collect();
        let packed = Tensor::from_vec(packed, (out_features, packed_columns), &Device::Cpu)?;
        let natural: Vec<u8> = (0..out_features * blocks_per_row)
            .map(|index| NVFP4_SCALE_BIT_CLASSES[index % NVFP4_SCALE_BIT_CLASSES.len()])
            .collect();
        let swizzled = swizzle_scale_bits(&natural, out_features, blocks_per_row);
        let block_scales = Tensor::from_vec(
            swizzled
                .into_iter()
                .map(f8e4m3::from_bits)
                .collect::<Vec<_>>(),
            (out_features, blocks_per_row.next_multiple_of(4)),
            &Device::Cpu,
        )?;
        let linear = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
            packed,
            block_scales,
            Tensor::new(0.012_345f32, &Device::Cpu)?,
            None,
            out_features,
            in_features,
        )?;
        let activations = Tensor::rand(
            -1.0f32,
            1.0f32,
            (activation_rows, in_features),
            &Device::Cpu,
        )?
        .to_device(cuda)?;

        // Arm 2 is the alternative #1317's part-1 record left open: keep the
        // ids `U8` and repair the `0xff` sentinel afterwards. It has to be a
        // SELECT, not an addition — see
        // `nvfp4_u8_id_repair_by_addition_loses_negative_zero`, which is why
        // "one extra elementwise pass" was never the real trade — so it costs
        // an `eq`, a broadcast, and a `where_cond` against the `U32` cast's
        // single kernel, over the same buffers.
        let corrected = |packed: &Tensor, tables: &Nvfp4DeviceTables| -> Result<Tensor> {
            let elements = packed.elem_count();
            let gathered = tables.nibbles.index_select(packed, 0)?;
            let repair = Tensor::full(-6.0f32, (elements, 2), cuda)?;
            packed
                .eq(u8::MAX)?
                .reshape((elements, 1))?
                .broadcast_as((elements, 2))?
                .contiguous()?
                .where_cond(&repair, &gathered)
        };

        let run = |arm: usize| -> Result<f64> {
            let tables = (arm > 0).then(|| linear.device_tables(cuda)).transpose()?;
            let started = Instant::now();
            let mut chunks = Vec::new();
            for start in (0..out_features).step_by(H3_COMFY_PORTABLE_ROW_CHUNK) {
                let rows = H3_COMFY_PORTABLE_ROW_CHUNK.min(out_features - start);
                let weight = match (arm, tables.as_ref()) {
                    (2, Some(tables)) => {
                        let packed = Tensor::from_vec(
                            linear
                                .packed_weight
                                .narrow(0, start, rows)?
                                .flatten_all()?
                                .to_vec1::<u8>()?,
                            rows * packed_columns,
                            cuda,
                        )?;
                        let scales = tables
                            .scales
                            .index_select(
                                &Tensor::from_slice(
                                    &linear.natural_block_scales
                                        [start * blocks_per_row..(start + rows) * blocks_per_row],
                                    rows * blocks_per_row,
                                    cuda,
                                )?,
                                0,
                            )?
                            .reshape((rows, blocks_per_row, 1))?;
                        corrected(&packed, tables)?
                            .reshape((rows, blocks_per_row, H3_COMFY_NVFP4_BLOCK_SIZE))?
                            .broadcast_mul(&scales)?
                            .reshape((rows, in_features))?
                            .broadcast_mul(&tables.tensor_scale)?
                    }
                    _ => linear.dequantize_rows(start, rows, cuda, tables.as_ref())?,
                };
                chunks.push(activations.matmul(&weight.t()?.contiguous()?)?);
            }
            drop(Tensor::cat(&chunks, 1)?);
            cuda.synchronize()?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };

        let mut fastest = [f64::INFINITY; 3];
        for (arm, best) in fastest.iter_mut().enumerate() {
            for _ in 0..2 {
                run(arm)?;
            }
            for _ in 0..10 {
                *best = best.min(run(arm)?);
            }
        }
        println!(
            "nvfp4 [{out_features}, {in_features}] x {activation_rows} rows, chunk \
             {H3_COMFY_PORTABLE_ROW_CHUNK}: host arm {:.1} ms, device arm (U32 ids) {:.1} ms \
             ({:.2}x), device arm (U8 ids + 0xff repair) {:.1} ms ({:.2}x); staging/chunk host \
             {} B, device {} B",
            fastest[0],
            fastest[1],
            fastest[0] / fastest[1],
            fastest[2],
            fastest[0] / fastest[2],
            linear.portable_weight_staging_bytes(H3_COMFY_PORTABLE_ROW_CHUNK)?,
            linear.device_weight_staging_bytes(H3_COMFY_PORTABLE_ROW_CHUNK)?,
        );
        // Deliberately no timing assertion: this is a report, and a wall-clock
        // threshold on a shared GPU is a flaky gate, not a contract. The
        // arms' equality is gated by
        // `nvfp4_device_dequantize_is_bit_identical_to_the_host_loop`.
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn portable_quantized_forwards_match_cuda() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        assert!(
            max_error(
                &synthetic_int8_forward(&Device::Cpu)?,
                &synthetic_int8_forward(&cuda)?
            )? <= 2e-4
        );
        assert!(
            max_error(
                &synthetic_nvfp4_forward(&Device::Cpu)?,
                &synthetic_nvfp4_forward(&cuda)?
            )? <= 2e-4
        );
        Ok(())
    }

    #[test]
    fn nvfp4_awq_dequant_and_forward_are_high_nibble_first() -> Result<()> {
        let device = Device::Cpu;
        let out_features = 2;
        let in_features = 16;
        let padded_rows = 16;
        let blocks_per_row = 1;
        let mut packed = vec![0u8; padded_rows * in_features / 2];
        packed[0] = 0x71; // +6.0 then +0.5 in the first logical row.
        packed[in_features / 2] = 0xaf; // -1.0 then -6.0 in the second row.
        let packed = Tensor::from_vec(packed, (padded_rows, in_features / 2), &device)?;
        let mut natural_scales = vec![1.0f32; padded_rows * blocks_per_row];
        natural_scales[0] = 2.0;
        natural_scales[1] = 0.5;
        let swizzled = swizzle_scales(&natural_scales, padded_rows, blocks_per_row);
        let scales = Tensor::from_vec(swizzled, (128, 4), &device)?.to_dtype(DType::F8E4M3)?;
        // comfy-kitchen's pinned loader/sample uses the one-element source
        // encoding; the constructor also accepts an actual rank-0 scalar.
        let tensor_scale = Tensor::from_vec(vec![0.5f32], 1, &device)?;
        let awq = Tensor::from_vec(
            (0..in_features)
                .map(|column| if column == 0 { 2.0f32 } else { 1.0f32 })
                .collect(),
            in_features,
            &device,
        )?
        .to_dtype(DType::F16)?;
        let scalar_linear = H3ComfyNvfp4AwqLinear::new(
            packed.clone(),
            scales.clone(),
            Tensor::new(0.5f32, &device)?,
            awq.clone(),
            out_features,
            in_features,
        )?;
        let linear = H3ComfyNvfp4AwqLinear::new(
            packed,
            scales,
            tensor_scale,
            awq,
            out_features,
            in_features,
        )?;
        let dense = linear.dequantize_weight(DType::F32, &device, 1)?;
        assert_eq!(
            max_error(
                &dense,
                &scalar_linear.dequantize_weight(DType::F32, &device, 1)?
            )?,
            0.0
        );
        let dense = dense.to_vec2::<f32>()?;
        assert_eq!(dense[0][0], 6.0);
        assert_eq!(dense[0][1], 0.5);
        assert_eq!(dense[1][0], -0.25);
        assert_eq!(dense[1][1], -1.5);

        let input = Tensor::from_vec(vec![1.0f32; in_features], (1, in_features), &device)?;
        let actual = linear.forward_dequantized(&input, None, DType::F32, 1)?;
        // AWQ doubles only input column zero before the dequantized matmul.
        assert_eq!(actual.to_vec2::<f32>()?, vec![vec![12.5, -2.0]]);
        assert_eq!(
            linear.encoded_weight_bytes()?,
            padded_rows * in_features / 2 + 128 * 4 + 4 + in_features * 2
        );
        assert_eq!(linear.portable_weight_staging_bytes(1)?, in_features * 4);
        assert!(linear.portable_weight_staging_bytes(0).is_err());
        assert!(linear
            .forward_dequantized(&input, None, DType::F32, 0)
            .is_err());
        Ok(())
    }

    #[test]
    fn nvfp4_scale_unswizzle_matches_pinned_comfy_kitchen_oracle() -> Result<()> {
        let rows = 129;
        let columns = 5;
        let mut blocked = vec![0.0f32; 256 * 8];
        // These offsets are a fixed oracle from comfy-kitchen's `to_blocked`
        // at 255a43879fe57bbcbecfdb273b46d772b00c5a90. They deliberately do
        // not use Mold's inverse `swizzle_scales` test helper.
        let fixtures = [
            (0, 0, 0, 1.0),
            (3, 0, 3, 2.0),
            (512, 0, 4, 3.0),
            (16, 1, 0, 4.0),
            (498, 31, 2, 5.0),
            (4, 32, 0, 6.0),
            (503, 63, 3, 7.0),
            (9, 64, 1, 8.0),
            (12, 96, 0, 9.0),
            (511, 127, 3, 10.0),
            (1024, 128, 0, 11.0),
            (1536, 128, 4, 12.0),
        ];
        for (offset, _, _, value) in fixtures {
            blocked[offset] = value;
        }
        let natural = unswizzle_nvfp4_scales(&blocked, rows, columns)?;
        for (_, row, column, value) in fixtures {
            assert_eq!(natural[row * columns + column], value);
        }
        assert_eq!(natural[2 * columns + 2], 0.0);
        Ok(())
    }

    #[test]
    fn nvfp4_scale_swizzle_round_trips_multiple_tiles() -> Result<()> {
        let rows = 144;
        let columns = 7;
        let natural = (0..rows * columns)
            .map(|index| ((index * 13 + 5) % 251) as f32)
            .collect::<Vec<_>>();
        let swizzled = swizzle_scales(&natural, rows, columns);
        assert_eq!(unswizzle_nvfp4_scales(&swizzled, rows, columns)?, natural);
        Ok(())
    }

    /// Every F8E4M3 bit pattern this loader accepts: the 127 finite
    /// non-negative patterns plus `0x80`, which is `-0.0` and therefore passes
    /// the `< 0.0` check. The negative finite patterns and the two NaN
    /// encodings (`0x7f` / `0xff`) are refused at construction, so no accepted
    /// scale cache can contain them.
    fn accepted_f8e4m3_bit_patterns() -> Vec<u8> {
        (0x00u8..=0x7e).chain(std::iter::once(0x80u8)).collect()
    }

    fn swizzle_scale_bits(natural: &[u8], rows: usize, columns: usize) -> Vec<u8> {
        let row_blocks = rows.div_ceil(128);
        let column_blocks = columns.div_ceil(4);
        let mut swizzled = vec![0u8; row_blocks * 128 * column_blocks * 4];
        for row in 0..rows {
            let row_block = row / 128;
            let row_in_block = row % 128;
            let quarter = row_in_block / 32;
            let lane = row_in_block % 32;
            for column in 0..columns {
                let column_block = column / 4;
                let column_in_block = column % 4;
                let swizzled_column = quarter * 4 + column_in_block;
                let tile = row_block * column_blocks + column_block;
                swizzled[tile * 512 + lane * 16 + swizzled_column] =
                    natural[row * columns + column];
            }
        }
        swizzled
    }

    /// The gate for #1316: the host scale cache narrowed from `f32` to the
    /// checkpoint's own byte plus a widening table, so every dequantized
    /// element must stay **bit**-identical, never merely close.
    ///
    /// The expectation is derived independently of the cache: the same
    /// `Tensor::to_dtype(F32)` widening the loader used to perform is applied
    /// here to the natural (already unswizzled) scale grid, and the dense
    /// weight is composed by hand as `E2M1_LUT * widened_scale * tensor_scale`.
    /// A `-0.0` scale meeting a `-0.0` E2M1 code is exactly why this compares
    /// `f32::to_bits` rather than values.
    #[test]
    fn nvfp4_dequantization_is_bit_identical_across_every_accepted_scale_pattern() -> Result<()> {
        let device = Device::Cpu;
        let out_features = 8;
        let in_features = 256;
        let padded_rows = 16;
        let packed_columns = in_features / 2;
        let blocks_per_row = in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let tensor_scale = 0.375f32;

        // 16 padded rows x 16 blocks = 256 scales, so each of the 128 accepted
        // bit patterns appears twice: 0x00, 0x80 (-0.0), the subnormals
        // 0x01..=0x07, and 0x7e (448.0, the largest finite E4M3 value).
        let patterns = accepted_f8e4m3_bit_patterns();
        assert_eq!(patterns.len(), 128);
        let natural_bits = (0..padded_rows * blocks_per_row)
            .map(|index| patterns[index % patterns.len()])
            .collect::<Vec<_>>();
        let block_scales = Tensor::from_vec(
            swizzle_scale_bits(&natural_bits, padded_rows, blocks_per_row)
                .into_iter()
                .map(f8e4m3::from_bits)
                .collect::<Vec<_>>(),
            (128, 16),
            &device,
        )?;
        assert_eq!(block_scales.dtype(), DType::F8E4M3);

        let packed = (0..padded_rows * packed_columns)
            .map(|index| ((index * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let awq = (0..in_features)
            .map(|column| 1.0 + (column % 5) as f32 / 4.0)
            .collect::<Vec<_>>();
        let linear = H3ComfyNvfp4AwqLinear::new(
            Tensor::from_vec(packed.clone(), (padded_rows, packed_columns), &device)?,
            block_scales,
            Tensor::new(tensor_scale, &device)?,
            Tensor::from_vec(awq.clone(), in_features, &device)?,
            out_features,
            in_features,
        )?;

        // Independent expectation: candle's own F8E4M3 -> F32 widening applied
        // to the natural grid, then the published dequantization arithmetic in
        // the published order.
        let widened = Tensor::from_vec(
            natural_bits
                .iter()
                .map(|bits| f8e4m3::from_bits(*bits))
                .collect::<Vec<_>>(),
            natural_bits.len(),
            &device,
        )?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
        let mut expected = vec![0.0f32; out_features * in_features];
        for row in 0..out_features {
            for column in 0..in_features {
                let byte = packed[row * packed_columns + column / 2];
                let nibble = if column.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                expected[row * in_features + column] = E2M1_LUT[nibble as usize]
                    * widened[row * blocks_per_row + column / H3_COMFY_NVFP4_BLOCK_SIZE]
                    * tensor_scale;
            }
        }

        // A chunk width that does not divide the row count exercises the
        // partial trailing chunk as well.
        let actual = linear
            .dequantize_weight(DType::F32, &device, 3)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "dequantized element {index} differs bitwise: {actual} vs {expected}"
            );
        }

        // The forward pass consumes the same weights, so its output is bitwise
        // reproducible from the independently composed dense weight.
        let input = Tensor::from_vec(
            (0..in_features)
                .map(|column| (column as f32 % 7.0 - 3.0) / 8.0)
                .collect::<Vec<_>>(),
            (1, in_features),
            &device,
        )?;
        let reference = input
            .broadcast_mul(&Tensor::from_vec(awq, (1, in_features), &device)?)?
            .matmul(
                &Tensor::from_vec(expected, (out_features, in_features), &device)?
                    .t()?
                    .contiguous()?,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let forward = linear
            .forward_dequantized(&input, None, DType::F32, out_features)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (actual, expected)) in forward.iter().zip(&reference).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "forward element {index} differs bitwise: {actual} vs {expected}"
            );
        }

        // Source-encoded accounting prices the scales at their checkpoint byte
        // width and must not move with the host representation.
        assert_eq!(
            linear.encoded_weight_bytes()?,
            padded_rows * in_features / 2 + 128 * 16 + 4 + in_features * 4
        );
        Ok(())
    }

    /// The point of #1316: the host cache is one byte per checkpoint scale, so
    /// the retained representation costs exactly what the source encodes. The
    /// memory authority charges `nvfp4_block_scale_bytes` unexpanded on the
    /// strength of this.
    #[test]
    fn nvfp4_host_scale_cache_is_one_byte_per_logical_source_scale() -> Result<()> {
        let device = Device::Cpu;
        let out_features = 8;
        let in_features = 256;
        let padded_rows = 16;
        let blocks_per_row = in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let linear = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
            Tensor::zeros((padded_rows, in_features / 2), DType::U8, &device)?,
            Tensor::ones((128, 16), DType::F32, &device)?.to_dtype(DType::F8E4M3)?,
            Tensor::new(1.0f32, &device)?,
            None,
            out_features,
            in_features,
        )?;
        assert_eq!(
            std::mem::size_of_val(&linear.natural_block_scales[..]),
            padded_rows * blocks_per_row
        );
        Ok(())
    }

    /// The 256-entry table is Candle's own widening, so a hand-rolled decode
    /// can never drift from it. Zero, negative zero, the subnormals, and the
    /// largest finite encoding are named because they are the values the
    /// construction-time validation branches on.
    #[test]
    fn f8e4m3_widening_table_is_candles_own_conversion() -> Result<()> {
        let table = f8e4m3_widening_table()?;
        let widened = Tensor::from_vec(
            (0..=u8::MAX).map(f8e4m3::from_bits).collect::<Vec<_>>(),
            256,
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
        for (bits, expected) in widened.iter().enumerate() {
            assert_eq!(
                table[bits].to_bits(),
                expected.to_bits(),
                "bits {bits:#04x}"
            );
        }
        assert_eq!(table[0x00].to_bits(), 0.0f32.to_bits());
        assert_eq!(table[0x80].to_bits(), (-0.0f32).to_bits());
        // -0.0 survives the construction-time "finite and nonnegative" check
        // exactly as it did when the cache widened every scale eagerly.
        assert!(table[0x80].is_finite() && table[0x80] >= 0.0);
        assert_eq!(table[0x01], 2.0f32.powi(-9));
        assert_eq!(table[0x7e], 448.0);
        assert!(table[0x7f].is_nan());
        assert!(table[0xff].is_nan());
        Ok(())
    }

    #[test]
    fn nvfp4_without_selective_awq_scale_uses_identity_input_transform() -> Result<()> {
        let device = Device::Cpu;
        let out_features = 1;
        let in_features = 16;
        let padded_rows = 16;
        let mut packed = vec![0_u8; padded_rows * in_features / 2];
        packed[0] = 0x22;
        let block_scales = Tensor::from_vec(
            swizzle_scales(&vec![1.0; padded_rows], padded_rows, 1),
            (128, 4),
            &device,
        )?
        .to_dtype(DType::F8E4M3)?;
        let linear = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
            Tensor::from_vec(packed, (padded_rows, in_features / 2), &device)?,
            block_scales,
            Tensor::new(1.0_f32, &device)?,
            None,
            out_features,
            in_features,
        )?;
        let output = linear.forward_dequantized(
            &Tensor::from_vec(vec![1.0_f32; in_features], (1, in_features), &device)?,
            None,
            DType::F32,
            1,
        )?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![2.0]]);
        assert_eq!(linear.encoded_weight_bytes()?, 16 * 8 + 128 * 4 + 4);
        Ok(())
    }

    #[test]
    fn int8_embedding_widens_signed_bytes_and_only_materializes_selected_rows() -> Result<()> {
        let device = Device::Cpu;
        let embedding = H3ComfyInt8TensorwiseEmbedding::new(
            Tensor::from_vec(vec![0_u8, 1, 127, 128, 255, 2, 254, 64], (2, 4), &device)?,
            Tensor::from_vec(vec![0.5_f32, 2.0], (2, 1), &device)?,
        )?;
        let output = embedding.forward(
            &Tensor::from_vec(vec![1_u32, 0], (1, 2), &device)?,
            DType::F32,
            &device,
        )?;
        assert_eq!(
            output.to_vec3::<f32>()?,
            vec![vec![
                vec![-2.0, 4.0, -4.0, 128.0],
                vec![0.0, 0.5, 63.5, -64.0],
            ]]
        );
        assert_eq!(embedding.vocabulary(), 2);
        assert_eq!(embedding.hidden_size(), 4);
        assert_eq!(embedding.encoded_weight_bytes()?, 16);
        Ok(())
    }

    #[test]
    fn nvfp4_awq_rejects_missing_shape_authority_and_bad_scales() -> Result<()> {
        let device = Device::Cpu;
        let packed = Tensor::zeros((16, 8), DType::U8, &device)?;
        let block_scales = Tensor::ones((128, 4), DType::F32, &device)?.to_dtype(DType::F8E4M3)?;
        let tensor_scale = Tensor::new(1.0f32, &device)?;
        let wrong_awq = Tensor::ones(15, DType::F32, &device)?;
        assert!(H3ComfyNvfp4AwqLinear::new(
            packed.clone(),
            block_scales.clone(),
            tensor_scale.clone(),
            wrong_awq,
            2,
            16,
        )
        .unwrap_err()
        .to_string()
        .contains("pre_quant_scale"));
        let awq = Tensor::ones(16, DType::F32, &device)?;
        let rank_two_tensor_scale = Tensor::ones((1, 1), DType::F32, &device)?;
        assert!(H3ComfyNvfp4AwqLinear::new(
            packed.clone(),
            block_scales.clone(),
            rank_two_tensor_scale,
            awq.clone(),
            2,
            16,
        )
        .unwrap_err()
        .to_string()
        .contains("source shape [] or [1]"));
        let zero_tensor_scale = Tensor::new(0.0f32, &device)?;
        assert!(H3ComfyNvfp4AwqLinear::new(
            packed.clone(),
            block_scales.clone(),
            zero_tensor_scale,
            awq.clone(),
            2,
            16,
        )
        .unwrap_err()
        .to_string()
        .contains("finite and positive"));
        let zero_awq = Tensor::zeros(16, DType::F32, &device)?;
        assert!(H3ComfyNvfp4AwqLinear::new(
            packed.clone(),
            block_scales.clone(),
            tensor_scale,
            zero_awq,
            2,
            16,
        )
        .unwrap_err()
        .to_string()
        .contains("AWQ input scales"));
        let scalar_int8 = Tensor::new(1u8, &device)?;
        assert!(
            H3ComfyNvfp4AwqLinear::new(packed, block_scales, scalar_int8, awq, 2, 16,)
                .unwrap_err()
                .to_string()
                .contains("source shape [] or [1]")
        );
        Ok(())
    }
}
