//! Family-neutral Comfy `int8_tensorwise` + ConvRot linear primitives.
//!
//! One INT8 ConvRot weight is the same object whether it sits in the pruned
//! MiniMax H3 DiT, the H3 Qwen conditioner, or an LTX-2.5 `int8-conv` pack:
//! signed INT8 bytes after a regular, normalized, 256-wide ConvRot transform,
//! one F32 scale per output row, and — at execution time — Comfy's W8A8 order
//! (rotate the activation in its input dtype, dynamically quantize it per row
//! with `absmax / 127`, accumulate INT8 x INT8 into INT32, apply both scales
//! in F32). The execution contracts here are validated against ComfyUI
//! `a464ac33588ae182f81a090d910cfbf21e255b73` and its exact
//! `comfy-kitchen==0.2.26` dependency at
//! `255a43879fe57bbcbecfdb273b46d772b00c5a90`; the CUDA arm is the pinned
//! cuBLASLt INT8-to-INT32 kernel in [`cuda`], and every other backend streams
//! portable F32 output-row chunks without retaining a dense weight.
//!
//! This module was promoted out of `minimax_h3::comfy_quant` so LTX-2 could
//! reuse it; `minimax_h3` keeps `H3`-prefixed aliases so its call sites and
//! qualification tests are unchanged. Two things are new here and only here:
//! [`ComfyInt8ConvRotLinear::new_on_device`], which retains the packed weight
//! on the execution device instead of re-uploading it per forward (the H3
//! contract, CPU storage plus per-forward staging, is [`new`] and is
//! untouched), and [`ComfyInt8ConvRotLinear::forward`], which adds a bias
//! AFTER the GEMM so a biased linear can still take the native kernel — the
//! kernel itself folds no bias, and [`select_int8_linear_kind`] says nothing
//! about one.
//!
//! These are reusable Candle operations, not runtime activation authority:
//! they operate only on tensors supplied by the caller and never discover,
//! open, download, or register a checkpoint.
//!
//! [`new`]: ComfyInt8ConvRotLinear::new

use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::native_int8_linear;

/// Comfy's INT8 ConvRot checkpoints require this exact rotation group.
pub const CONVROT_GROUP_SIZE: usize = 256;

/// Bounded default for portable dequantized weight staging.
pub const PORTABLE_ROW_CHUNK: usize = 256;

/// Workspace offered to the source-matched cuBLASLt INT8 heuristic.
///
/// Compiled whenever the workspace accounting is, not only when the kernel is:
/// `reference_workspace_upper_bound` has a `native_cuda` arm that a Metal or
/// CPU H3 build still has to be able to name, even though it never takes it.
#[cfg(any(feature = "cuda", feature = "h3", feature = "h3-private-uat"))]
pub(crate) const NATIVE_INT8_CUBLAS_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// Whether this build linked the cuBLASLt INT8 kernel.
///
/// The one place the feature flag is read for this decision, so every caller
/// hands [`select_int8_linear_kind`] the same answer.
pub const fn native_int8_kernel_compiled() -> bool {
    cfg!(feature = "cuda")
}

/// Which arm executes one INT8 ConvRot linear.
///
/// The choice is a pure function of the device, the compiled feature set, and
/// the weight's own shape — never of the calling surface — mirroring
/// `zimage`/`qwen_image`'s `select_linear_kind`. Keeping it a function rather
/// than an inline `cfg!` is what lets Metal's arm be pinned by a test on a
/// machine that has no Metal device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int8LinearKind {
    /// cuBLASLt signed INT8 -> INT32 GEMM with a fused dequantize.
    NativeCudaInt8,
    /// Portable quantize/dequantize: rotate, dynamically quantize the
    /// activation, accumulate against the packed signed bytes in F32 over
    /// bounded output-row chunks, then apply both scales. This is the arm
    /// Metal and CPU take, and it is exact against the same reference the
    /// CUDA kernel matches.
    PortableQuantizeDequantize,
}

/// Resolve the INT8 ConvRot arm for one linear.
///
/// Metal always takes the portable arm: [`cuda`]'s cuBLASLt kernel has no
/// Metal twin, and unlike Qwen-Image's `QMatMul` there is no candle-side Metal
/// quantized kernel to qualify — so this is a correctness fallback by
/// construction, not a tuning choice. It is also why the H3 Metal tier is
/// `CorrectnessOnly`: the portable arm re-uploads and widens the packed weight
/// chunk on every forward.
///
/// This answers for the KERNEL CALL only — the cuBLASLt layout descriptors
/// need both extents to be multiples of four, and the kernel folds no bias.
/// Whether a biased linear may still take the native arm is the caller's
/// question: [`ComfyInt8ConvRotLinear::forward`] adds the bias after the GEMM,
/// while `minimax_h3`'s `select_h3_int8_linear_kind` keeps the older rule
/// that a bias sends the whole linear to the portable arm.
pub fn select_int8_linear_kind(
    device: &Device,
    native_kernel_compiled: bool,
    in_features: usize,
    out_features: usize,
) -> Int8LinearKind {
    if native_kernel_compiled
        && device.is_cuda()
        && in_features.is_multiple_of(4)
        && out_features.is_multiple_of(4)
    {
        Int8LinearKind::NativeCudaInt8
    } else {
        Int8LinearKind::PortableQuantizeDequantize
    }
}

pub(crate) fn ensure_floating(dtype: DType, role: &str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        other => candle::bail!("Comfy {role} must be F32, F16, or BF16, got {other:?}"),
    }
}

pub(crate) fn ensure_output_dtype(dtype: DType) -> Result<()> {
    ensure_floating(dtype, "quantized linear output")
}

pub(crate) fn checked_round_up(value: usize, multiple: usize, role: &str) -> Result<usize> {
    if multiple == 0 {
        candle::bail!("Comfy {role} alignment must be positive")
    }
    value
        .checked_add(multiple - 1)
        .map(|value| value / multiple * multiple)
        .ok_or_else(|| candle::Error::Msg(format!("Comfy {role} alignment overflows")))
}

pub(crate) fn flattened_input(input: &Tensor, in_features: usize) -> Result<(Tensor, Vec<usize>)> {
    ensure_floating(input.dtype(), "quantized linear input")?;
    let mut output_shape = input.dims().to_vec();
    let Some(last) = output_shape.last_mut() else {
        candle::bail!("Comfy quantized linear input must have rank at least one")
    };
    if *last != in_features {
        candle::bail!(
            "Comfy quantized linear expected {in_features} input features, got {}",
            *last
        )
    }
    *last = 0;
    let rows = input.dims()[..input.rank() - 1]
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul(*value))
        .ok_or_else(|| candle::Error::Msg("Comfy input row count overflows".into()))?;
    if rows == 0 {
        candle::bail!("Comfy quantized linear input cannot be empty")
    }
    Ok((input.reshape((rows, in_features))?, output_shape))
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn accelerator_signed_widening_workspace_upper_bound(elements: usize) -> Result<u64> {
    let raw = elements
        .checked_mul(std::mem::size_of::<u8>())
        .ok_or_else(|| candle::Error::Msg("Comfy raw INT8 staging size overflows".into()))?;
    let widened = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle::Error::Msg("Comfy widened INT8 staging size overflows".into()))?;
    raw.checked_mul(2)
        .and_then(|bytes| {
            widened
                .checked_mul(4)
                .and_then(|wide| bytes.checked_add(wide))
        })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| candle::Error::Msg("Comfy signed widening workspace overflows".into()))
}

pub(crate) fn finish_linear(
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
                "Comfy quantized linear bias must have shape [{out_features}], got {:?}",
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

pub(crate) fn regular_hadamard_values(size: usize) -> Result<Vec<f32>> {
    if size < 4 || !size.is_power_of_two() || !size.trailing_zeros().is_multiple_of(2) {
        candle::bail!("Comfy ConvRot group size must be a power of four at least four, got {size}")
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
            .ok_or_else(|| candle::Error::Msg("Comfy Hadamard dimension overflows".into()))?;
        let elements = next_width
            .checked_mul(next_width)
            .ok_or_else(|| candle::Error::Msg("Comfy Hadamard element count overflows".into()))?;
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

pub(crate) fn regular_hadamard(device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        regular_hadamard_values(CONVROT_GROUP_SIZE)?,
        (CONVROT_GROUP_SIZE, CONVROT_GROUP_SIZE),
        device,
    )
}

/// Match `torch.round`: nearest integer with half-way values rounded to even.
///
/// Candle's built-in `round` follows Rust/C `round` and sends half-way values
/// away from zero. Comfy's dynamic INT8 QDQ uses PyTorch's ties-to-even rule,
/// so that primitive is not source-equivalent at exact half steps.
pub(crate) fn round_ties_even(input: &Tensor) -> Result<Tensor> {
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

/// One Comfy INT8 ConvRot weight: packed signed bytes plus one F32 scale per
/// output row.
///
/// Candle does not expose a signed-I8 tensor dtype, so the checkpoint's exact
/// two's-complement bytes are retained in a U8 tensor and widened explicitly
/// when a row chunk is staged. No numeric U8 conversion is permitted.
///
/// `weight_scale` is deliberately strict: ConvRot quantization is per output
/// channel in comfy-kitchen, so its only accepted shape is `[out_features, 1]`.
/// A scalar scale is not source-equivalent and is rejected.
///
/// Two storage contracts share this one type. [`Self::new`] is the MiniMax H3
/// contract: the weight lives on the CPU and every forward stages it onto the
/// execution device (the H3 memory model prices exactly that traffic).
/// [`Self::new_on_device`] retains the packed bytes wherever the caller put
/// them — for LTX-2.5 that is the CUDA device, once, at load — so a forward
/// uploads nothing. Every method below is storage-agnostic: `to_device` onto
/// the device the weight already lives on is a no-op clone.
#[derive(Clone, Debug)]
pub struct ComfyInt8ConvRotLinear {
    weight: Tensor,
    weight_scale: Tensor,
    out_features: usize,
    in_features: usize,
}

impl ComfyInt8ConvRotLinear {
    /// CPU-resident storage, staged onto the execution device per forward.
    pub fn new(weight: Tensor, weight_scale: Tensor) -> Result<Self> {
        if !weight.device().is_cpu() || !weight_scale.device().is_cpu() {
            candle::bail!("Comfy portable INT8 ConvRot storage must remain on CPU")
        }
        Self::validated(weight, weight_scale)
    }

    /// Storage retained on whichever device the caller supplies — the weight
    /// and its scales must already share one.
    ///
    /// This is what lets a resident LTX-2.5 block run the W8A8 kernel without
    /// an upload per forward: the packed bytes are the kernel's own operand,
    /// so a device-resident weight costs one byte per parameter and nothing
    /// per step.
    pub fn new_on_device(weight: Tensor, weight_scale: Tensor) -> Result<Self> {
        if !weight.device().same_device(weight_scale.device()) {
            candle::bail!(
                "Comfy INT8 ConvRot weight and scales must share a device, got {:?} and {:?}",
                weight.device().location(),
                weight_scale.device().location()
            )
        }
        Self::validated(weight, weight_scale)
    }

    fn validated(weight: Tensor, weight_scale: Tensor) -> Result<Self> {
        if weight.dtype() != DType::U8 {
            candle::bail!(
                "Comfy INT8 ConvRot weight must use raw two's-complement U8 storage, got {:?}",
                weight.dtype()
            )
        }
        if weight_scale.dtype() != DType::F32 {
            candle::bail!(
                "Comfy INT8 ConvRot scale must use F32 storage, got {:?}",
                weight_scale.dtype()
            )
        }
        let (out_features, in_features) = weight.dims2()?;
        if out_features == 0 || in_features == 0 || !in_features.is_multiple_of(CONVROT_GROUP_SIZE)
        {
            candle::bail!(
                "Comfy INT8 ConvRot weight must be nonempty and its input width must be divisible by {}",
                CONVROT_GROUP_SIZE
            )
        }
        if weight_scale.dims() != [out_features, 1] {
            candle::bail!(
                "Comfy INT8 ConvRot scale must have source shape [{out_features}, 1], got {:?}",
                weight_scale.dims()
            )
        }
        let scales = weight_scale.flatten_all()?.to_vec1::<f32>()?;
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            candle::bail!("Comfy INT8 ConvRot scales must be finite and positive")
        }
        Ok(Self {
            weight,
            weight_scale,
            out_features,
            in_features,
        })
    }

    /// The device holding the packed weight and its scales.
    pub fn device(&self) -> &Device {
        self.weight.device()
    }

    /// The packed two's-complement bytes, `U8 [out_features, in_features]`.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// The per-output-row scales, `F32 [out_features, 1]`.
    pub fn weight_scale(&self) -> &Tensor {
        &self.weight_scale
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
            .ok_or_else(|| candle::Error::Msg("Comfy INT8 byte count overflows".into()))?;
        let scales = self
            .out_features
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| candle::Error::Msg("Comfy INT8 scale bytes overflow".into()))?;
        weight
            .checked_add(scales)
            .ok_or_else(|| candle::Error::Msg("Comfy INT8 byte count overflows".into()))
    }

    /// Dense F32 device-weight bytes staged for one output-row chunk.
    ///
    /// This deliberately excludes the already-resident compressed host tensor
    /// and the final output tensor, which have separate lifetimes.
    pub fn portable_weight_staging_bytes(&self, rows_per_chunk: usize) -> Result<usize> {
        if rows_per_chunk == 0 {
            candle::bail!("Comfy INT8 row chunk must be positive")
        }
        rows_per_chunk
            .min(self.out_features)
            .checked_mul(self.in_features)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| candle::Error::Msg("Comfy INT8 staging size overflows".into()))
    }

    /// Conservative peak bytes allocated by one production W8A8 reference
    /// call, excluding its borrowed input. This mirrors the tensors and chunk
    /// accumulation in `forward_reference` and is capture-only authority.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn reference_workspace_upper_bound(
        &self,
        input_rows: usize,
        input_dtype: DType,
        output_dtype: DType,
        rows_per_chunk: usize,
        has_bias: bool,
        native_cuda: bool,
    ) -> Result<u64> {
        if input_rows == 0 || rows_per_chunk == 0 {
            candle::bail!("Comfy reference workspace requires positive rows and chunk size")
        }
        let input_bytes = input_dtype.size_in_bytes();
        let output_bytes = output_dtype.size_in_bytes();
        let chunk = rows_per_chunk.min(self.out_features);
        let checked = |values: &[usize]| -> Result<u64> {
            let bytes = values.iter().try_fold(1usize, |total, value| {
                total.checked_mul(*value).ok_or_else(|| {
                    candle::Error::Msg("Comfy reference workspace size overflows".into())
                })
            })?;
            u64::try_from(bytes)
                .map_err(|_| candle::Error::Msg("Comfy workspace exceeds u64".into()))
        };
        let activation_input = checked(&[input_rows, self.in_features, input_bytes])?;
        let activation_f32 = checked(&[input_rows, self.in_features, std::mem::size_of::<f32>()])?;
        let row_input = checked(&[input_rows, input_bytes])?;
        let row_f32 = checked(&[input_rows, std::mem::size_of::<f32>()])?;
        let hadamard_f32 = checked(&[
            CONVROT_GROUP_SIZE,
            CONVROT_GROUP_SIZE,
            std::mem::size_of::<f32>(),
        ])?;
        let hadamard_input = checked(&[CONVROT_GROUP_SIZE, CONVROT_GROUP_SIZE, input_bytes])?;
        if native_cuda {
            if has_bias {
                candle::bail!("Comfy native INT8 CUDA workspace does not accept bias")
            }
            let rotated_input = activation_input;
            let packed_weight = checked(&[
                self.out_features,
                self.in_features,
                std::mem::size_of::<u8>(),
            ])?;
            let weight_scales = checked(&[self.out_features, std::mem::size_of::<f32>()])?;
            let quantized_input =
                checked(&[input_rows, self.in_features, std::mem::size_of::<i8>()])?;
            let input_scales = checked(&[input_rows, std::mem::size_of::<f32>()])?;
            let accumulator =
                checked(&[input_rows, self.out_features, std::mem::size_of::<i32>()])?;
            let output = checked(&[input_rows, self.out_features, output_bytes])?;
            return [
                hadamard_f32,
                hadamard_input,
                rotated_input,
                packed_weight,
                weight_scales,
                quantized_input,
                input_scales,
                accumulator,
                u64::try_from(NATIVE_INT8_CUBLAS_WORKSPACE_BYTES)
                    .map_err(|_| candle::Error::Msg("Comfy cuBLAS workspace exceeds u64".into()))?,
                output,
            ]
            .into_iter()
            .try_fold(0u64, |total, bytes| {
                total.checked_add(bytes).ok_or_else(|| {
                    candle::Error::Msg("Comfy native CUDA workspace sum overflows".into())
                })
            });
        }
        let signed_widening_workspace = accelerator_signed_widening_workspace_upper_bound(
            chunk.checked_mul(self.in_features).ok_or_else(|| {
                candle::Error::Msg("Comfy signed widening element count overflows".into())
            })?,
        )?;
        let weight_scale = checked(&[chunk, std::mem::size_of::<f32>()])?;
        let chunk_f32 = checked(&[input_rows, chunk, std::mem::size_of::<f32>()])?;
        let chunk_output = checked(&[input_rows, chunk, output_bytes])?;
        let output_chunks = checked(&[input_rows, self.out_features, output_bytes])?;
        let concatenated_output = output_chunks;
        let bias_workspace = if has_bias {
            checked(&[self.out_features, std::mem::size_of::<f32>(), 2])?
                .checked_add(output_chunks)
                .ok_or_else(|| candle::Error::Msg("Comfy bias workspace sum overflows".into()))?
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
            // Accelerator signed widening can retain the transferred raw U8,
            // U8 comparison, F32 comparison cast, affine result, unsigned F32
            // source, and returned F32 result. Charge that complete pipeline
            // before the remaining linear intermediates.
            signed_widening_workspace,
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
            total
                .checked_add(bytes)
                .ok_or_else(|| candle::Error::Msg("Comfy reference workspace sum overflows".into()))
        })
    }

    pub(crate) fn signed_rows(&self, start: usize, rows: usize, device: &Device) -> Result<Tensor> {
        let raw = self.weight.narrow(0, start, rows)?;
        if device.is_cpu() {
            let values = raw
                .flatten_all()?
                .to_vec1::<u8>()?
                .into_iter()
                .map(|byte| i8::from_ne_bytes([byte]) as f32)
                .collect::<Vec<_>>();
            return Tensor::from_vec(values, (rows, self.in_features), device);
        }

        // Preserve the checkpoint's exact two's-complement bytes during the
        // host-to-device transfer, then widen on the execution device. The
        // former path widened into a host Vec<f32> first, quadrupling PCIe
        // traffic and doing one scalar CPU conversion for every weight byte
        // on every H3 transformer evaluation.
        let unsigned = raw.to_device(device)?.to_dtype(DType::F32)?;
        let wrapped = unsigned
            .gt(127.0)?
            .to_dtype(DType::F32)?
            .affine(-256.0, 0.0)?;
        unsigned.broadcast_add(&wrapped)
    }

    fn dequantize_rows(
        &self,
        start: usize,
        rows: usize,
        output_dtype: DType,
        device: &Device,
        hadamard: &Tensor,
    ) -> Result<Tensor> {
        let groups = self.in_features / CONVROT_GROUP_SIZE;
        let grouped_rows = rows
            .checked_mul(groups)
            .ok_or_else(|| candle::Error::Msg("Comfy INT8 grouped weight rows overflow".into()))?;
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
            .reshape((grouped_rows, CONVROT_GROUP_SIZE))?
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
            candle::bail!("Comfy INT8 row chunk must be positive")
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
            candle::bail!("Comfy INT8 row chunk must be positive")
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
                    "Comfy Qwen INT8 weight-only bias must have shape [{}], got {:?}",
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
            candle::bail!("Comfy INT8 row chunk must be positive")
        }
        let device = input.device();
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let rows = flat.dim(0)?;
        let groups = self.in_features / CONVROT_GROUP_SIZE;
        let grouped_rows = rows.checked_mul(groups).ok_or_else(|| {
            candle::Error::Msg("Comfy INT8 grouped activation rows overflow".into())
        })?;
        let hadamard = regular_hadamard(device)?;
        let rotated = flat
            .to_dtype(DType::F32)?
            .reshape((grouped_rows, CONVROT_GROUP_SIZE))?
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

    /// Execute Comfy's source-defined INT8 ConvRot W8A8 order.
    ///
    /// Activations are rotated in their input dtype, dynamically quantized
    /// per row with `absmax / 127`, rounded and clamped to signed INT8, then
    /// accumulated against the checkpoint's packed signed bytes. CUDA performs
    /// native signed INT8-to-INT32 multiplication and applies both scales in
    /// F32. CPU and Metal mirror the eager fallback with bounded output-row
    /// chunks, retaining neither a dense block weight nor the full-width
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
            candle::bail!("Comfy INT8 row chunk must be positive")
        }
        let device = input.device();
        let input_dtype = input.dtype();
        ensure_floating(input_dtype, "INT8 ConvRot activation")?;
        let (flat, output_shape) = flattened_input(input, self.in_features)?;
        let rotated = self.rotated_activation(&flat)?;
        // The H3 rule: a bias sends the whole linear to the portable arm,
        // because the kernel folds none and this path adds it only there.
        let kind = if bias.is_some() {
            Int8LinearKind::PortableQuantizeDequantize
        } else {
            select_int8_linear_kind(
                device,
                native_int8_kernel_compiled(),
                self.in_features,
                self.out_features,
            )
        };
        #[cfg(not(feature = "cuda"))]
        debug_assert_eq!(kind, Int8LinearKind::PortableQuantizeDequantize);
        #[cfg(feature = "cuda")]
        if kind == Int8LinearKind::NativeCudaInt8 {
            let weight = self.weight.to_device(device)?;
            let weight_scale = self.weight_scale.to_device(device)?;
            let mut output_shape = output_shape;
            *output_shape
                .last_mut()
                .expect("flattened_input established a nonempty shape") = self.out_features;
            return cuda::native_int8_linear(&rotated, &weight, &weight_scale, output_dtype)?
                .reshape(&*output_shape);
        }
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

    /// Rotate one flattened `[rows, in_features]` activation by the regular
    /// 256-wide Hadamard in the activation's own dtype — comfy-kitchen's
    /// `convrot` step, which runs before the dynamic INT8 quantizer.
    fn rotated_activation(&self, flat: &Tensor) -> Result<Tensor> {
        let device = flat.device();
        let input_dtype = flat.dtype();
        let rows = flat.dim(0)?;
        let groups = self.in_features / CONVROT_GROUP_SIZE;
        let grouped_rows = rows.checked_mul(groups).ok_or_else(|| {
            candle::Error::Msg("Comfy INT8 grouped activation rows overflow".into())
        })?;
        let hadamard = regular_hadamard(device)?.to_dtype(input_dtype)?;
        flat.to_dtype(input_dtype)?
            .reshape((grouped_rows, CONVROT_GROUP_SIZE))?
            .matmul(&hadamard)?
            .reshape((rows, self.in_features))
    }

    /// Comfy's W8A8 order with the bias applied AFTER the GEMM.
    ///
    /// This is [`Self::forward_reference`]'s arithmetic with one difference:
    /// a bias no longer disqualifies the native kernel. The kernel folds no
    /// bias, so a biased linear runs the INT8 GEMM to an F32 result, adds the
    /// bias there, and narrows once — the same F32 add-then-narrow the
    /// portable arm's `finish_linear` performs — while an unbiased linear
    /// narrows inside the kernel exactly as before. The activation is
    /// quantized identically on both arms (dynamic per-row `absmax / 127`),
    /// so a biased and an unbiased linear of the same weight see the same
    /// INT8 operands.
    ///
    /// The arm is [`select_int8_linear_kind`]'s answer for the execution
    /// device; the packed weight is staged onto that device if it lives
    /// elsewhere and used in place if it already does, so this method is the
    /// one to call on a [`Self::new_on_device`] weight.
    pub fn forward(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        output_dtype: DType,
    ) -> Result<Tensor> {
        ensure_output_dtype(output_dtype)?;
        let device = input.device();
        ensure_floating(input.dtype(), "INT8 ConvRot activation")?;
        if let Some(bias) = bias {
            ensure_floating(bias.dtype(), "quantized linear bias")?;
            if bias.dims() != [self.out_features] {
                candle::bail!(
                    "Comfy quantized linear bias must have shape [{}], got {:?}",
                    self.out_features,
                    bias.dims()
                )
            }
        }
        let kind = select_int8_linear_kind(
            device,
            native_int8_kernel_compiled(),
            self.in_features,
            self.out_features,
        );
        let gemm = match kind {
            Int8LinearKind::NativeCudaInt8 => {
                #[cfg(not(feature = "cuda"))]
                {
                    candle::bail!("Comfy native INT8 kernel selected in a build without CUDA")
                }
                #[cfg(feature = "cuda")]
                {
                    let (flat, mut output_shape) = flattened_input(input, self.in_features)?;
                    let rotated = self.rotated_activation(&flat)?;
                    let weight = self.weight.to_device(device)?;
                    let weight_scale = self.weight_scale.to_device(device)?;
                    // Narrow inside the kernel when nothing follows the GEMM;
                    // keep F32 when a bias does, so the add happens before
                    // the one narrowing.
                    let gemm_dtype = if bias.is_some() {
                        DType::F32
                    } else {
                        output_dtype
                    };
                    *output_shape
                        .last_mut()
                        .expect("flattened_input established a nonempty shape") = self.out_features;
                    cuda::native_int8_linear(&rotated, &weight, &weight_scale, gemm_dtype)?
                        .reshape(&*output_shape)?
                }
            }
            Int8LinearKind::PortableQuantizeDequantize => {
                let portable_dtype = if bias.is_some() {
                    DType::F32
                } else {
                    output_dtype
                };
                self.forward_reference(input, None, portable_dtype, PORTABLE_ROW_CHUNK)?
            }
        };
        let Some(bias) = bias else {
            return Ok(gemm);
        };
        let bias = bias.to_device(device)?.to_dtype(DType::F32)?;
        gemm.broadcast_add(&bias)?.to_dtype(output_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn max_error(left: &Tensor, right: &Tensor) -> Result<f32> {
        let left = left.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let right = right
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(left.len(), right.len());
        Ok(left
            .into_iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max))
    }

    fn bf16_bits(tensor: &Tensor) -> Result<Vec<u16>> {
        Ok(tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(half::bf16::to_bits)
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn f32_bits(tensor: &Tensor) -> Result<Vec<u32>> {
        Ok(tensor
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// An `[8, 256]` weight whose rows use every scale in a short cycle, plus
    /// a bias with both signs — the smallest shape that satisfies the native
    /// kernel's multiple-of-four rule on both extents.
    fn fixture(device: &Device) -> Result<(ComfyInt8ConvRotLinear, Tensor)> {
        let columns = CONVROT_GROUP_SIZE;
        let outputs = 8;
        let raw = (0..outputs * columns)
            .map(|index| (((index * 37 + 11) % 251) as i16 - 125) as i8 as u8)
            .collect::<Vec<_>>();
        let scales = (0..outputs)
            .map(|index| (index + 1) as f32 / 256.0)
            .collect::<Vec<_>>();
        let linear = ComfyInt8ConvRotLinear::new_on_device(
            Tensor::from_vec(raw, (outputs, columns), device)?,
            Tensor::from_vec(scales, (outputs, 1), device)?,
        )?;
        let bias = Tensor::from_vec(
            (0..outputs)
                .map(|index| (index as f32 - 3.5) * 0.75)
                .collect::<Vec<_>>(),
            outputs,
            device,
        )?;
        Ok((linear, bias))
    }

    fn activation(device: &Device) -> Result<Tensor> {
        let columns = CONVROT_GROUP_SIZE;
        Tensor::from_vec(
            (0..3 * columns)
                .map(|index| ((index * 17 % 257) as f32 - 128.0) / 37.0)
                .collect::<Vec<_>>(),
            (3, columns),
            device,
        )
    }

    #[test]
    fn the_arm_is_native_only_for_a_compiled_kernel_on_cuda() {
        // Off CUDA the answer is portable however the binary was built and
        // whatever the shape.
        for compiled in [false, true] {
            for (in_features, out_features) in [(256, 512), (256, 6), (255, 8)] {
                assert_eq!(
                    select_int8_linear_kind(&Device::Cpu, compiled, in_features, out_features),
                    Int8LinearKind::PortableQuantizeDequantize
                );
            }
        }
    }

    #[test]
    fn new_on_device_accepts_cpu_storage_and_shares_the_h3_validation() -> Result<()> {
        let (linear, _) = fixture(&Device::Cpu)?;
        assert!(linear.device().is_cpu());
        assert_eq!(linear.weight().dtype(), DType::U8);
        assert_eq!(linear.weight_scale().dims(), &[8, 1]);
        // The same contracts `new` enforces: U8 bytes, F32 `[out, 1]` scales,
        // a 256-divisible input width, finite positive scales.
        assert!(ComfyInt8ConvRotLinear::new_on_device(
            Tensor::zeros((8, 255), DType::U8, &Device::Cpu)?,
            Tensor::ones((8, 1), DType::F32, &Device::Cpu)?
        )
        .is_err());
        assert!(ComfyInt8ConvRotLinear::new_on_device(
            Tensor::zeros((8, 256), DType::U8, &Device::Cpu)?,
            Tensor::ones(8, DType::F32, &Device::Cpu)?
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn forward_without_bias_is_the_reference_forward() -> Result<()> {
        let (linear, _) = fixture(&Device::Cpu)?;
        let input = activation(&Device::Cpu)?;
        let expected = linear.forward_reference(&input, None, DType::F32, PORTABLE_ROW_CHUNK)?;
        let actual = linear.forward(&input, None, DType::F32)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn forward_with_bias_is_the_reference_forward_plus_bias_narrowed_once() -> Result<()> {
        let (linear, bias) = fixture(&Device::Cpu)?;
        let input = activation(&Device::Cpu)?;
        let reference = linear.forward_reference(&input, None, DType::F32, PORTABLE_ROW_CHUNK)?;
        let expected = reference.broadcast_add(&bias)?;

        let actual = linear.forward(&input, Some(&bias), DType::F32)?;
        assert_eq!(actual.dims(), &[3, 8]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );

        // The bias is added in F32 and the result narrowed once, so a BF16
        // request is the F32 answer narrowed — never a BF16 GEMM plus a BF16
        // bias, which would round twice.
        let narrowed = linear.forward(&input, Some(&bias), DType::BF16)?;
        assert_eq!(narrowed.dtype(), DType::BF16);
        assert_eq!(bf16_bits(&narrowed)?, bf16_bits(&expected)?);

        // Rank-3 activations keep their leading shape.
        let batched = linear.forward(
            &input.reshape((1, 3, CONVROT_GROUP_SIZE))?,
            Some(&bias),
            DType::F32,
        )?;
        assert_eq!(batched.dims(), &[1, 3, 8]);
        Ok(())
    }

    #[test]
    fn forward_refuses_a_bias_of_the_wrong_shape() -> Result<()> {
        let (linear, _) = fixture(&Device::Cpu)?;
        let input = activation(&Device::Cpu)?;
        let wrong = Tensor::zeros(7, DType::F32, &Device::Cpu)?;
        assert!(linear.forward(&input, Some(&wrong), DType::F32).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_arm_on_cuda_needs_the_kernel_and_multiple_of_four_extents() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        assert_eq!(
            select_int8_linear_kind(&cuda, true, 256, 8),
            Int8LinearKind::NativeCudaInt8
        );
        assert_eq!(
            select_int8_linear_kind(&cuda, false, 256, 8),
            Int8LinearKind::PortableQuantizeDequantize
        );
        assert_eq!(
            select_int8_linear_kind(&cuda, true, 256, 6),
            Int8LinearKind::PortableQuantizeDequantize
        );
        assert!(native_int8_kernel_compiled());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn new_on_device_refuses_a_weight_and_scales_on_different_devices() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        assert!(ComfyInt8ConvRotLinear::new_on_device(
            Tensor::zeros((8, 256), DType::U8, &cuda)?,
            Tensor::ones((8, 1), DType::F32, &Device::Cpu)?
        )
        .is_err());
        // The H3 constructor still refuses device storage outright.
        assert!(ComfyInt8ConvRotLinear::new(
            Tensor::zeros((8, 256), DType::U8, &cuda)?,
            Tensor::ones((8, 1), DType::F32, &cuda)?
        )
        .is_err());
        Ok(())
    }

    /// The device-resident form runs the very same kernel over the very same
    /// bytes the CPU-staged form uploads per forward, so the two are
    /// bit-identical — with and without a bias, in F32 and narrowed.
    #[cfg(feature = "cuda")]
    #[test]
    fn device_resident_forward_is_bit_identical_to_cpu_staged() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (cpu_linear, bias) = fixture(&Device::Cpu)?;
        let staged = ComfyInt8ConvRotLinear::new(
            cpu_linear.weight().clone(),
            cpu_linear.weight_scale().clone(),
        )?;
        let resident = ComfyInt8ConvRotLinear::new_on_device(
            cpu_linear.weight().to_device(&cuda)?,
            cpu_linear.weight_scale().to_device(&cuda)?,
        )?;
        assert!(resident.device().is_cuda());
        let input = activation(&cuda)?;
        let bias = bias.to_device(&cuda)?;
        for (bias, dtype) in [
            (None, DType::F32),
            (Some(&bias), DType::F32),
            (None, DType::BF16),
            (Some(&bias), DType::BF16),
        ] {
            let staged_out = staged.forward(&input, bias, dtype)?;
            let resident_out = resident.forward(&input, bias, dtype)?;
            assert_eq!(staged_out.dtype(), dtype);
            assert_eq!(
                f32_bits(&staged_out)?,
                f32_bits(&resident_out)?,
                "bias {} dtype {dtype:?}",
                bias.is_some()
            );
        }
        Ok(())
    }

    /// The native arm with a bias is the portable W8A8 reference plus that
    /// bias: same quantized operands, F32 add, one narrowing.
    #[cfg(feature = "cuda")]
    #[test]
    fn native_biased_forward_matches_portable_reference_plus_bias() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (cpu_linear, bias) = fixture(&Device::Cpu)?;
        let input = activation(&Device::Cpu)?;
        let expected = cpu_linear
            .forward_reference(&input, None, DType::F32, PORTABLE_ROW_CHUNK)?
            .broadcast_add(&bias)?;

        let resident = ComfyInt8ConvRotLinear::new_on_device(
            cpu_linear.weight().to_device(&cuda)?,
            cpu_linear.weight_scale().to_device(&cuda)?,
        )?;
        assert_eq!(
            select_int8_linear_kind(&cuda, native_int8_kernel_compiled(), 256, 8),
            Int8LinearKind::NativeCudaInt8
        );
        let cuda_input = input.to_device(&cuda)?;
        let cuda_bias = bias.to_device(&cuda)?;
        let actual = resident
            .forward(&cuda_input, Some(&cuda_bias), DType::F32)?
            .to_device(&Device::Cpu)?;
        assert!(max_error(&actual, &expected)? <= 1e-4);

        // Narrowed once, after the bias: the BF16 answer is the F32 answer
        // rounded, so it sits within half a BF16 ulp of it at these
        // magnitudes (|y| < 128 ⇒ ulp ≤ 0.5).
        let narrowed = resident
            .forward(&cuda_input, Some(&cuda_bias), DType::BF16)?
            .to_device(&Device::Cpu)?;
        assert_eq!(bf16_bits(&narrowed)?, bf16_bits(&actual)?);
        Ok(())
    }
}
