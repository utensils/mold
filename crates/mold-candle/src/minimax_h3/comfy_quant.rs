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
//!   per-row INT8 QDQ, streams output-row chunks, and applies both F32 scales
//!   without retaining a dense weight.
//! - The pruned DiT's scaled FP8 matrices retain E4M3 weights plus scalar F32
//!   weight/input scales. Their reference path preserves the source QDQ order
//!   and accumulates against bounded, reconstructed F32 weight chunks.
//! - The Qwen3-VL INT8 ConvRot variant reconstructs bounded weight chunks and
//!   runs an ordinary floating-point matmul; it does not quantize activations.
//! - The Qwen3-VL NVFP4 variant stores high-nibble-first weights with
//!   swizzled FP8-E4M3 block scales and an F32 tensor scale. Its selective
//!   AWQ-style `pre_quant_scale` is applied when present. Comfy deliberately
//!   selects full-precision matrix multiplication for text encoders, so the
//!   portable forward dequantizes weight chunks before the linear operation.
//!
//! These are reusable Candle operations, not runtime activation authority.
//! H3 remains hidden behind Mold's compliance gate and unavailable factory.

use candle::{DType, Device, Result, Tensor};

/// Comfy's released H3 INT8 checkpoints require this exact ConvRot group.
pub const H3_COMFY_CONVROT_GROUP_SIZE: usize = 256;

/// NVFP4 uses one FP8-E4M3 scale per 16 logical values.
pub const H3_COMFY_NVFP4_BLOCK_SIZE: usize = 16;

/// Bounded default for portable dequantized weight staging.
pub const H3_COMFY_PORTABLE_ROW_CHUNK: usize = 256;

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn ensure_floating(dtype: DType, role: &str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        other => candle::bail!("MiniMax H3 {role} must be F32, F16, or BF16, got {other:?}"),
    }
}

fn ensure_output_dtype(dtype: DType) -> Result<()> {
    ensure_floating(dtype, "quantized linear output")
}

fn checked_round_up(value: usize, multiple: usize, role: &str) -> Result<usize> {
    if multiple == 0 {
        candle::bail!("MiniMax H3 {role} alignment must be positive")
    }
    value
        .checked_add(multiple - 1)
        .map(|value| value / multiple * multiple)
        .ok_or_else(|| candle::Error::Msg(format!("MiniMax H3 {role} alignment overflows")))
}

fn flattened_input(input: &Tensor, in_features: usize) -> Result<(Tensor, Vec<usize>)> {
    ensure_floating(input.dtype(), "quantized linear input")?;
    let mut output_shape = input.dims().to_vec();
    let Some(last) = output_shape.last_mut() else {
        candle::bail!("MiniMax H3 quantized linear input must have rank at least one")
    };
    if *last != in_features {
        candle::bail!(
            "MiniMax H3 quantized linear expected {in_features} input features, got {}",
            *last
        )
    }
    *last = 0;
    let rows = input.dims()[..input.rank() - 1]
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul(*value))
        .ok_or_else(|| candle::Error::Msg("MiniMax H3 input row count overflows".into()))?;
    if rows == 0 {
        candle::bail!("MiniMax H3 quantized linear input cannot be empty")
    }
    Ok((input.reshape((rows, in_features))?, output_shape))
}

fn finish_linear(
    chunks: Vec<Tensor>,
    bias: Option<&Tensor>,
    output_dtype: DType,
    mut output_shape: Vec<usize>,
    out_features: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut output = Tensor::cat(&chunks, 1)?;
    if let Some(bias) = bias {
        ensure_floating(bias.dtype(), "quantized linear bias")?;
        if bias.dims() != [out_features] {
            candle::bail!(
                "MiniMax H3 quantized linear bias must have shape [{out_features}], got {:?}",
                bias.dims()
            )
        }
        let bias = bias
            .to_device(device)?
            .to_dtype(DType::F32)?
            .reshape((1, out_features))?;
        output = output.broadcast_add(&bias)?;
    }
    ensure_output_dtype(output_dtype)?;
    *output_shape
        .last_mut()
        .expect("flattened_input established a nonempty shape") = out_features;
    output.to_dtype(output_dtype)?.reshape(&*output_shape)
}

fn regular_hadamard_values(size: usize) -> Result<Vec<f32>> {
    if size < 4 || !size.is_power_of_two() || !size.trailing_zeros().is_multiple_of(2) {
        candle::bail!(
            "MiniMax H3 ConvRot group size must be a power of four at least four, got {size}"
        )
    }
    const H4: [[f32; 4]; 4] = [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];
    let mut matrix = H4.iter().flatten().copied().collect::<Vec<_>>();
    let mut width = 4usize;
    while width < size {
        let next_width = width
            .checked_mul(4)
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 Hadamard dimension overflows".into()))?;
        let elements = next_width.checked_mul(next_width).ok_or_else(|| {
            candle::Error::Msg("MiniMax H3 Hadamard element count overflows".into())
        })?;
        let mut next = vec![0.0; elements];
        for row in 0..width {
            for column in 0..width {
                let value = matrix[row * width + column];
                for h4_row in 0..4 {
                    for h4_column in 0..4 {
                        next[(row * 4 + h4_row) * next_width + column * 4 + h4_column] =
                            value * H4[h4_row][h4_column];
                    }
                }
            }
        }
        matrix = next;
        width = next_width;
    }
    let normalization = 1.0 / (size as f32).sqrt();
    matrix.iter_mut().for_each(|value| *value *= normalization);
    Ok(matrix)
}

fn regular_hadamard(device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        regular_hadamard_values(H3_COMFY_CONVROT_GROUP_SIZE)?,
        (H3_COMFY_CONVROT_GROUP_SIZE, H3_COMFY_CONVROT_GROUP_SIZE),
        device,
    )
}

/// Match `torch.round`: nearest integer with half-way values rounded to even.
///
/// Candle's built-in `round` follows Rust/C `round` and sends half-way values
/// away from zero. Comfy's dynamic INT8 QDQ uses PyTorch's ties-to-even rule,
/// so that primitive is not source-equivalent at exact half steps.
fn round_ties_even(input: &Tensor) -> Result<Tensor> {
    ensure_floating(input.dtype(), "ties-to-even input")?;
    let lower = input.floor()?;
    let fraction = input.broadcast_sub(&lower)?;
    let greater_than_half = fraction.gt(0.5)?.to_dtype(input.dtype())?;
    let exactly_half = fraction.eq(0.5)?.to_dtype(input.dtype())?;
    let even_below = lower.affine(0.5, 0.0)?.floor()?.affine(2.0, 0.0)?;
    let odd_lower = lower.broadcast_sub(&even_below)?;
    let increment = greater_than_half.broadcast_add(&exactly_half.broadcast_mul(&odd_lower)?)?;
    lower.broadcast_add(&increment)
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

/// CPU-backed INT8 ConvRot weight for the Comfy pruned H3 DiT or Qwen
/// conditioner.
///
/// Candle does not expose a signed-I8 tensor dtype, so the checkpoint's exact
/// two's-complement bytes are retained in a U8 tensor and widened explicitly
/// when a row chunk is staged. No numeric U8 conversion is permitted.
///
/// `weight_scale` is deliberately strict: ConvRot quantization is per output
/// channel in comfy-kitchen, so its only accepted shape is `[out_features, 1]`.
/// A scalar scale is not source-equivalent and is rejected.
#[derive(Clone, Debug)]
pub struct H3ComfyInt8ConvRotLinear {
    weight: Tensor,
    weight_scale: Tensor,
    out_features: usize,
    in_features: usize,
}

impl H3ComfyInt8ConvRotLinear {
    pub fn new(weight: Tensor, weight_scale: Tensor) -> Result<Self> {
        if !weight.device().is_cpu() || !weight_scale.device().is_cpu() {
            candle::bail!("MiniMax H3 portable INT8 ConvRot storage must remain on CPU")
        }
        if weight.dtype() != DType::U8 {
            candle::bail!(
                "MiniMax H3 INT8 ConvRot weight must use raw two's-complement U8 storage, got {:?}",
                weight.dtype()
            )
        }
        if weight_scale.dtype() != DType::F32 {
            candle::bail!(
                "MiniMax H3 INT8 ConvRot scale must use F32 storage, got {:?}",
                weight_scale.dtype()
            )
        }
        let (out_features, in_features) = weight.dims2()?;
        if out_features == 0
            || in_features == 0
            || !in_features.is_multiple_of(H3_COMFY_CONVROT_GROUP_SIZE)
        {
            candle::bail!(
                "MiniMax H3 INT8 ConvRot weight must be nonempty and its input width must be divisible by {}",
                H3_COMFY_CONVROT_GROUP_SIZE
            )
        }
        if weight_scale.dims() != [out_features, 1] {
            candle::bail!(
                "MiniMax H3 INT8 ConvRot scale must have source shape [{out_features}, 1], got {:?}",
                weight_scale.dims()
            )
        }
        let scales = weight_scale.flatten_all()?.to_vec1::<f32>()?;
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            candle::bail!("MiniMax H3 INT8 ConvRot scales must be finite and positive")
        }
        Ok(Self {
            weight,
            weight_scale,
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

    /// Source-encoded checkpoint bytes represented by this weight and row scale.
    pub fn encoded_weight_bytes(&self) -> Result<usize> {
        let weight = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 INT8 byte count overflows".into()))?;
        let scales = self
            .out_features
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 INT8 scale bytes overflow".into()))?;
        weight
            .checked_add(scales)
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 INT8 byte count overflows".into()))
    }

    /// Dense F32 device-weight bytes staged for one output-row chunk.
    ///
    /// This deliberately excludes the already-resident compressed host tensor
    /// and the final output tensor, which have separate lifetimes.
    pub fn portable_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 INT8 row chunk must be positive")
        }
        rows_per_chunk
            .min(self.out_features)
            .checked_mul(self.in_features)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 INT8 staging size overflows".into()))
    }

    /// Conservative peak bytes allocated by one production W8A8 reference
    /// call, excluding its borrowed input. This mirrors the tensors and chunk
    /// accumulation in `forward_reference` and is capture-only authority.
    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn reference_workspace_upper_bound(
        &self,
        input_rows: usize,
        input_dtype: DType,
        output_dtype: DType,
        rows_per_chunk: usize,
        has_bias: bool,
    ) -> Result<u64> {
        if input_rows == 0 || rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 reference workspace requires positive rows and chunk size")
        }
        let input_bytes = input_dtype.size_in_bytes();
        let output_bytes = output_dtype.size_in_bytes();
        let chunk = rows_per_chunk.min(self.out_features);
        let checked = |values: &[usize]| -> Result<u64> {
            let bytes = values.iter().try_fold(1usize, |total, value| {
                total.checked_mul(*value).ok_or_else(|| {
                    candle::Error::Msg("MiniMax H3 reference workspace size overflows".into())
                })
            })?;
            u64::try_from(bytes)
                .map_err(|_| candle::Error::Msg("MiniMax H3 workspace exceeds u64".into()))
        };
        let activation_input = checked(&[input_rows, self.in_features, input_bytes])?;
        let activation_f32 = checked(&[input_rows, self.in_features, std::mem::size_of::<f32>()])?;
        let row_input = checked(&[input_rows, input_bytes])?;
        let row_f32 = checked(&[input_rows, std::mem::size_of::<f32>()])?;
        let hadamard_f32 = checked(&[
            H3_COMFY_CONVROT_GROUP_SIZE,
            H3_COMFY_CONVROT_GROUP_SIZE,
            std::mem::size_of::<f32>(),
        ])?;
        let hadamard_input = checked(&[
            H3_COMFY_CONVROT_GROUP_SIZE,
            H3_COMFY_CONVROT_GROUP_SIZE,
            input_bytes,
        ])?;
        let quantized_weight = checked(&[chunk, self.in_features, std::mem::size_of::<f32>()])?;
        let weight_scale = checked(&[chunk, std::mem::size_of::<f32>()])?;
        let chunk_f32 = checked(&[input_rows, chunk, std::mem::size_of::<f32>()])?;
        let chunk_output = checked(&[input_rows, chunk, output_bytes])?;
        let output_chunks = checked(&[input_rows, self.out_features, output_bytes])?;
        let concatenated_output = output_chunks;
        let bias_workspace = if has_bias {
            checked(&[self.out_features, std::mem::size_of::<f32>(), 2])?
                .checked_add(output_chunks)
                .ok_or_else(|| {
                    candle::Error::Msg("MiniMax H3 bias workspace sum overflows".into())
                })?
        } else {
            0
        };
        [
            // regular_hadamard F32 plus its input-dtype cast.
            hadamard_f32,
            hadamard_input,
            // Rotated input; abs; max(input); max(F32); affine; clamp; the
            // input-dtype scale cast; and broadcast division.
            activation_input,
            activation_input,
            row_input,
            row_f32,
            row_f32,
            row_f32,
            row_input,
            activation_input,
            // round -> clamp -> F32 conversion can retain every chained
            // full-activation result through the statement.
            activation_input,
            activation_input,
            activation_f32,
            // A chunk may retain original+contiguous weight, row scale,
            // combined scale, matmul, scaled matmul, and cast output.
            quantized_weight,
            quantized_weight,
            weight_scale,
            chunk_f32,
            chunk_f32,
            chunk_f32,
            chunk_output,
            // Tensor::cat allocates a full result while every chunk lives.
            output_chunks,
            concatenated_output,
            bias_workspace,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                candle::Error::Msg("MiniMax H3 reference workspace sum overflows".into())
            })
        })
    }

    fn signed_rows(&self, start: usize, rows: usize, device: &Device) -> Result<Tensor> {
        let values = self
            .weight
            .narrow(0, start, rows)?
            .flatten_all()?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|byte| i8::from_ne_bytes([byte]) as f32)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, self.in_features), device)
    }

    fn dequantize_rows(
        &self,
        start: usize,
        rows: usize,
        output_dtype: DType,
        device: &Device,
        hadamard: &Tensor,
    ) -> Result<Tensor> {
        let groups = self.in_features / H3_COMFY_CONVROT_GROUP_SIZE;
        let grouped_rows = rows.checked_mul(groups).ok_or_else(|| {
            candle::Error::Msg("MiniMax H3 INT8 grouped weight rows overflow".into())
        })?;
        let quantized = self.signed_rows(start, rows, device)?;
        let scales = self
            .weight_scale
            .narrow(0, start, rows)?
            .to_device(device)?
            .reshape((rows, 1))?;
        quantized
            .broadcast_mul(&scales)?
            // Candle 0.11 materializes the broadcasted RHS for
            // `broadcast_matmul`. Flatten groups into the row dimension so the
            // fixed 256x256 Hadamard stays singular and the bounded staging
            // calculation remains authoritative.
            .reshape((grouped_rows, H3_COMFY_CONVROT_GROUP_SIZE))?
            .matmul(hadamard)?
            .reshape((rows, self.in_features))?
            .to_dtype(output_dtype)
    }

    /// Reconstruct the dense weight in the original, unrotated basis.
    pub fn dequantize_weight(
        &self,
        output_dtype: DType,
        device: &Device,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 INT8 row chunk must be positive")
        }
        let hadamard = regular_hadamard(device)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(self.out_features - start);
            chunks.push(self.dequantize_rows(start, rows, output_dtype, device, &hadamard)?);
        }
        Tensor::cat(&chunks, 0)
    }

    /// Comfy's Qwen INT8 layout is weight-only: each ConvRot row chunk is
    /// reconstructed in the input dtype, then a normal floating-point linear
    /// operation runs without dynamically quantizing the activation. This is
    /// intentionally distinct from the DiT's optional fused W8A8 execution.
    pub fn forward_weight_only(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 INT8 row chunk must be positive")
        }
        let compute_dtype = input.dtype();
        ensure_floating(compute_dtype, "Qwen INT8 weight-only compute")?;
        let device = input.device();
        let (flat, mut output_shape) = flattened_input(input, self.in_features)?;
        let flat = flat.to_dtype(compute_dtype)?;
        let hadamard = regular_hadamard(device)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(self.out_features - start);
            let weight = self.dequantize_rows(start, rows, compute_dtype, device, &hadamard)?;
            chunks.push(flat.matmul(&weight.t()?.contiguous()?)?);
        }
        let mut output = Tensor::cat(&chunks, 1)?;
        if let Some(bias) = bias {
            ensure_floating(bias.dtype(), "Qwen INT8 weight-only bias")?;
            if bias.dims() != [self.out_features] {
                candle::bail!(
                    "MiniMax H3 Qwen INT8 weight-only bias must have shape [{}], got {:?}",
                    self.out_features,
                    bias.dims()
                )
            }
            output = output.broadcast_add(
                &bias
                    .to_device(device)?
                    .to_dtype(compute_dtype)?
                    .reshape((1, self.out_features))?,
            )?;
        }
        *output_shape
            .last_mut()
            .expect("flattened_input established a nonempty shape") = self.out_features;
        output.to_dtype(output_dtype)?.reshape(&*output_shape)
    }

    /// Portable low-memory forward equivalent to multiplying by the exact
    /// dequantized ConvRot weight. It does not claim parity with Comfy's lossy
    /// dynamic activation quantizer or its fused CUDA kernel.
    pub fn forward_dequantized(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 INT8 row chunk must be positive")
        }
        let device = input.device();
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let rows = flat.dim(0)?;
        let groups = self.in_features / H3_COMFY_CONVROT_GROUP_SIZE;
        let grouped_rows = rows.checked_mul(groups).ok_or_else(|| {
            candle::Error::Msg("MiniMax H3 INT8 grouped activation rows overflow".into())
        })?;
        let hadamard = regular_hadamard(device)?;
        let rotated = flat
            .to_dtype(DType::F32)?
            .reshape((grouped_rows, H3_COMFY_CONVROT_GROUP_SIZE))?
            .matmul(&hadamard)?
            .reshape((rows, self.in_features))?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let width = rows_per_chunk.min(self.out_features - start);
            let quantized = self.signed_rows(start, width, device)?;
            let scales = self
                .weight_scale
                .narrow(0, start, width)?
                .to_device(device)?
                .reshape((1, width))?;
            chunks.push(
                rotated
                    .matmul(&quantized.t()?.contiguous()?)?
                    .broadcast_mul(&scales)?,
            );
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

    /// Execute Comfy's source-defined INT8 ConvRot W8A8 reference order.
    ///
    /// Activations are rotated in their input dtype, dynamically quantized
    /// per row with `absmax / 127`, rounded and clamped to signed INT8, then
    /// accumulated against the checkpoint's packed signed bytes. Scales are
    /// applied in F32 before each bounded output-row chunk is converted to the
    /// requested output dtype. This mirrors comfy-kitchen's eager fallback
    /// while retaining neither a dense block weight nor the full output-width
    /// accumulator.
    pub fn forward_reference(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
        rows_per_chunk: usize,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        if rows_per_chunk == 0 {
            candle::bail!("MiniMax H3 INT8 row chunk must be positive")
        }
        let device = input.device();
        let input_dtype = input.dtype();
        ensure_floating(input_dtype, "INT8 ConvRot activation")?;
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let rows = flat.dim(0)?;
        let groups = self.in_features / H3_COMFY_CONVROT_GROUP_SIZE;
        let grouped_rows = rows.checked_mul(groups).ok_or_else(|| {
            candle::Error::Msg("MiniMax H3 INT8 grouped activation rows overflow".into())
        })?;
        let hadamard = regular_hadamard(device)?.to_dtype(input_dtype)?;
        let rotated = flat
            .to_dtype(input_dtype)?
            .reshape((grouped_rows, H3_COMFY_CONVROT_GROUP_SIZE))?
            .matmul(&hadamard)?
            .reshape((rows, self.in_features))?;
        let input_scale = rotated
            .abs()?
            .max_keepdim(1)?
            .to_dtype(DType::F32)?
            .affine(1.0 / 127.0, 0.0)?
            .clamp(1e-30f32, f32::MAX)?;
        // comfy-kitchen casts the F32 scale back to the activation dtype for
        // the division before PyTorch's ties-to-even rounding and clamping to
        // the signed-I8 interval.
        let scaled_input = rotated.broadcast_div(&input_scale.to_dtype(input_dtype)?)?;
        let quantized_input = round_ties_even(&scaled_input)?
            .clamp(-128.0f32, 127.0f32)?
            .to_dtype(DType::F32)?;
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let width = rows_per_chunk.min(self.out_features - start);
            let quantized_weight = self.signed_rows(start, width, device)?;
            let weight_scale = self
                .weight_scale
                .narrow(0, start, width)?
                .to_device(device)?
                .reshape((1, width))?;
            let scale = input_scale.broadcast_mul(&weight_scale)?;
            chunks.push(
                quantized_input
                    .matmul(&quantized_weight.t()?.contiguous()?)?
                    .broadcast_mul(&scale)?
                    .to_dtype(output_dtype)?,
            );
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

fn unswizzle_nvfp4_scales(
    swizzled: &[f32],
    logical_rows: usize,
    logical_columns: usize,
) -> Result<Vec<f32>> {
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
    let mut natural = vec![0.0; logical_elements];
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
    natural_block_scales: Vec<f32>,
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
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let natural_block_scales =
            unswizzle_nvfp4_scales(&swizzled, padded_out_features, blocks_per_row)?;
        if natural_block_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
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
    /// FP8 block scales are charged at their source byte width even though this
    /// portable implementation retains an unswizzled F32 cache for execution.
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

    /// Dense F32 device-weight bytes staged for one output-row chunk.
    ///
    /// This deliberately excludes the already-resident packed host tensor,
    /// its unswizzled scale cache, and the final output tensor.
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

    fn dequantize_rows(&self, start: usize, rows: usize, device: &Device) -> Result<Tensor> {
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
                    * scales[column / H3_COMFY_NVFP4_BLOCK_SIZE]
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
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(self.out_features - start);
            chunks.push(self.dequantize_rows(start, rows, device)?);
        }
        Tensor::cat(&chunks, 0)?.to_dtype(output_dtype)
    }

    /// Comfy text encoders multiply by the ModelOpt AWQ input scale and then
    /// use a full-precision matrix multiplication over dequantized NVFP4
    /// weights. This method preserves that ordering and bounds weight staging.
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
        let mut chunks = Vec::new();
        for start in (0..self.out_features).step_by(rows_per_chunk) {
            let width = rows_per_chunk.min(self.out_features - start);
            let weight = self.dequantize_rows(start, width, device)?;
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
        let mut bytes = vec![0u8; H3_COMFY_CONVROT_GROUP_SIZE];
        bytes[..4].copy_from_slice(&[0x80, 0xff, 0x00, 0x7f]);
        let linear = H3ComfyInt8ConvRotLinear::new(
            Tensor::from_vec(bytes, (1, H3_COMFY_CONVROT_GROUP_SIZE), &Device::Cpu)?,
            Tensor::ones((1, 1), DType::F32, &Device::Cpu)?,
        )?;
        let signed = linear.signed_rows(0, 1, &Device::Cpu)?.to_vec2::<f32>()?;
        assert_eq!(&signed[0][..4], &[-128.0, -1.0, 0.0, 127.0]);
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
