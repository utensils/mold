//! Quantized (GGUF) Qwen-Image transformer with device-specific linear dispatch.
//!
//! **CUDA**: GPU-resident weights whose GGML dtype candle's MMQ/MMVQ kernels
//! accept run through `QMatMul`, so the checkpoint is never dequantized — the
//! int8 kernels consume the BF16 activation and return BF16. Weights the
//! kernels cannot take (an `IQ*` or float-stored tensor), CPU-staged weights,
//! and `MOLD_QWEN_QMATMUL=0` keep the per-forward dequant-to-BF16 arm, because
//! candle's `dequantize_matmul` fallback reads the activation as `f32` and
//! therefore errors on BF16. All computation stays in BF16 matching the
//! model's training dtype.
//!
//! **Metal**: uses candle's `QMatMul`-backed `Linear` which avoids per-forward
//! full dequantization (faster on Metal). Computation in F32 since Metal's QMatMul
//! dequantizes to F32 internally.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_transformers::models::z_image::transformer::apply_rotary_emb;
use mold_candle::quantized::VarBuilder;
use mold_candle::quantized_nn::Linear as QMatMulLinear;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::transformer::{QwenImageConfig, MAX_PERIOD};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
pub(crate) const ROPE_CACHE_LEN: usize = 4096;

/// CUDA: BF16 (matches training dtype, halves activation memory vs F32).
/// Metal: F32 (QMatMul dequantizes to F32 internally on Metal).
fn working_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

fn debug_stage(stage: &str) {
    if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
        eprintln!("[qwen-quantized] {stage}");
    }
}

/// Device class a `VarBuilder` resolves to, so the linear-arm decision stays a
/// pure function that can be exercised for CUDA without a CUDA device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearDevice {
    Cuda,
    Metal,
    Other,
}

impl LinearDevice {
    fn of(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Other
        }
    }
}

/// Which implementation a quantized linear resolves to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenLinearKind {
    /// Weight stays quantized; candle's kernels consume it directly.
    QMatMul,
    /// Full per-forward dequantization to the working dtype.
    Dequant,
}

/// GGML dtypes candle's CUDA MMQ/MMVQ kernels accept
/// (`candle-core/src/quantized/fast_mmq.rs`, `supports`). Anything else falls
/// through to `dequantize_matmul`, which reads the activation as `f32` and so
/// errors on the BF16 activations this engine runs — those weights must keep
/// the per-forward dequant arm.
pub(crate) fn cuda_mmq_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// `MOLD_QWEN_QMATMUL`: `0` restores the per-forward dequantization arm on
/// CUDA. Unset — or any other value — keeps the quantized fast path, so a
/// typo degrades to the shipped behavior rather than to the slow one.
pub(crate) fn parse_qwen_qmatmul(value: Option<&str>) -> bool {
    !matches!(value.map(str::trim), Some("0"))
}

/// Process-frozen `MOLD_QWEN_QMATMUL`, read once per process through the
/// admission-frozen environment.
fn qwen_qmatmul_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = parse_qwen_qmatmul(crate::runtime_env::value("MOLD_QWEN_QMATMUL").as_deref());
        if !enabled {
            tracing::warn!(
                "qwen-image: MOLD_QWEN_QMATMUL=0 — quantized linears dequantize per forward"
            );
        }
        enabled
    })
}

/// The whole linear-arm decision, as a pure function.
///
/// Metal has always used `QMatMul`. CUDA joins it only when every kernel
/// precondition holds: the weight's GGML dtype is one the MMQ/MMVQ kernels
/// accept, the weight is resident on the device the activations live on (a
/// CPU-staged weight would hit candle's `unreachable!` in the CUDA matmul),
/// and the escape hatch is not forcing the old arm. Everything else — CPU
/// included — dequantizes per forward.
pub(crate) fn select_linear_kind(
    device: LinearDevice,
    weight_dtype: GgmlDType,
    weight_on_target_device: bool,
    qmatmul_enabled: bool,
) -> QwenLinearKind {
    match device {
        LinearDevice::Metal => QwenLinearKind::QMatMul,
        LinearDevice::Cuda
            if qmatmul_enabled && weight_on_target_device && cuda_mmq_supported(weight_dtype) =>
        {
            QwenLinearKind::QMatMul
        }
        _ => QwenLinearKind::Dequant,
    }
}

/// Device-dispatched quantized linear layer.
///
/// CUDA: `QMatMul` when the weight is MMQ-eligible, otherwise dequantizes the
/// weight to BF16 per forward (temporary ~72MB peak).
/// Metal: uses QMatMul (weight stays quantized, dequant inside kernel).
enum QwenLinear {
    /// Per-forward BF16 dequantization — correct dtype for CUDA.
    Dequant {
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
    },
    /// QMatMul-backed — avoids full dequant, faster on Metal.
    QMatMul(QMatMulLinear),
}

impl Module for QwenLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dequant { weight, bias } => {
                let dtype = working_dtype(x.device());
                let x = x.to_dtype(dtype)?;
                let w = if weight.device().is_cpu() && !x.device().is_cpu() {
                    weight
                        .dequantize(&Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_device(x.device())?
                } else {
                    weight.dequantize(x.device())?.to_dtype(dtype)?
                };
                let bias = bias
                    .as_ref()
                    .map(|b| b.to_device(x.device())?.to_dtype(dtype))
                    .transpose()?;
                candle_nn::Linear::new(w, bias).forward(&x)
            }
            Self::QMatMul(inner) => {
                // Both CUDA fast paths decline a non-contiguous rhs, and the
                // fallback they decline into cannot read a BF16 activation.
                if x.is_contiguous() {
                    inner.forward(x)
                } else {
                    inner.forward(&x.contiguous()?)
                }
            }
        }
    }
}

fn qlinear(vb: &VarBuilder, name: &str) -> Result<QwenLinear> {
    let vb = vb.pp(name);
    let weight = vb.get_no_shape("weight")?;
    let device = vb.device();
    // Metal: F32 bias matches QMatMul's F32 output.
    // CUDA: BF16 bias matches both the dequant arm and MMQ's BF16 output.
    let dtype = working_dtype(device);
    let bias = match vb.get_no_shape("bias") {
        Ok(b) => Some(b.dequantize(device)?.to_dtype(dtype)?),
        Err(_) => None,
    };
    match select_linear_kind(
        LinearDevice::of(device),
        weight.dtype(),
        weight.device().same_device(device),
        qwen_qmatmul_enabled(),
    ) {
        QwenLinearKind::QMatMul => Ok(QwenLinear::QMatMul(QMatMulLinear::from_arc(weight, bias)?)),
        QwenLinearKind::Dequant => Ok(QwenLinear::Dequant { weight, bias }),
    }
}

#[derive(Debug, Clone)]
struct DynamicRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl DynamicRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let weight = self.weight.to_device(xs.device())?.to_dtype(dtype)?;
        xs.broadcast_mul(&weight)
    }
}

/// Dequantize a small 1D weight vector for RmsNorm (norm weights are tiny).
fn dequant_rms_norm(vb: &VarBuilder, name: &str, eps: f64) -> Result<DynamicRmsNorm> {
    let dtype = working_dtype(vb.device());
    let weight = vb
        .pp(name)
        .get_no_shape("weight")?
        .dequantize(vb.device())?
        .to_dtype(dtype)?;
    Ok(DynamicRmsNorm { weight, eps })
}

#[derive(Debug, Clone)]
struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RopeCacheDevice {
    Cpu,
    Cuda,
    Metal,
}

/// Image RoPE tables are `seq_len x total_half` — the large half of the cache —
/// and depend only on the latent grid, never on the prompt length. Keying them
/// on the text length (as one combined key used to) re-materialised them for
/// every new prompt-pair length and, once split CFG stopped padding the two
/// streams to a common length, would have doubled that again.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ImageRopeKey {
    frame: usize,
    height: usize,
    width: usize,
    device: RopeCacheDevice,
}

/// Text RoPE is `pos_cos.narrow(0, max_vid_index, txt_len)` — a `txt_len x
/// total_half` slice, negligible next to the image tables, and a strict prefix
/// of any longer slice at the same `max_vid_index`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextRopeKey {
    max_vid_index: usize,
    txt_len: usize,
    device: RopeCacheDevice,
}

impl RopeCacheDevice {
    fn from_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Cpu
        }
    }
}

/// `(cos, sin)` for one axis group.
type RopeTables = (Tensor, Tensor);

#[derive(Debug)]
pub(crate) struct QwenRopeEmbedder {
    axes_dims: Vec<usize>,
    axis_half_dims: Vec<usize>,
    axis_offsets: Vec<usize>,
    pos_cos: Tensor,
    pos_sin: Tensor,
    neg_cos: Tensor,
    neg_sin: Tensor,
    dtype: DType,
    image_cache: Mutex<HashMap<ImageRopeKey, RopeTables>>,
    text_cache: Mutex<HashMap<TextRopeKey, RopeTables>>,
}

impl Clone for QwenRopeEmbedder {
    fn clone(&self) -> Self {
        Self {
            axes_dims: self.axes_dims.clone(),
            axis_half_dims: self.axis_half_dims.clone(),
            axis_offsets: self.axis_offsets.clone(),
            pos_cos: self.pos_cos.clone(),
            pos_sin: self.pos_sin.clone(),
            neg_cos: self.neg_cos.clone(),
            neg_sin: self.neg_sin.clone(),
            dtype: self.dtype,
            image_cache: Mutex::new(HashMap::new()),
            text_cache: Mutex::new(HashMap::new()),
        }
    }
}

impl QwenRopeEmbedder {
    pub(crate) fn new(
        theta: f64,
        axes_dims: Vec<usize>,
        cpu_device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut axis_half_dims = Vec::with_capacity(axes_dims.len());
        let mut axis_offsets = Vec::with_capacity(axes_dims.len());
        let mut running_offset = 0;
        for &dim in &axes_dims {
            if dim % 2 != 0 {
                candle_core::bail!("Qwen RoPE axis dim {dim} must be even");
            }
            axis_offsets.push(running_offset);
            let half_dim = dim / 2;
            axis_half_dims.push(half_dim);
            running_offset += half_dim;
        }

        let pos_index: Vec<f32> = (0..ROPE_CACHE_LEN).map(|i| i as f32).collect();
        let neg_index: Vec<f32> = (0..ROPE_CACHE_LEN)
            .rev()
            .map(|i| -(i as i32) as f32 - 1.0)
            .collect();

        let mut pos_cos_parts = Vec::with_capacity(axes_dims.len());
        let mut pos_sin_parts = Vec::with_capacity(axes_dims.len());
        let mut neg_cos_parts = Vec::with_capacity(axes_dims.len());
        let mut neg_sin_parts = Vec::with_capacity(axes_dims.len());
        for &dim in &axes_dims {
            let (pos_cos, pos_sin) = Self::rope_params(&pos_index, dim, theta, cpu_device)?;
            let (neg_cos, neg_sin) = Self::rope_params(&neg_index, dim, theta, cpu_device)?;
            pos_cos_parts.push(pos_cos);
            pos_sin_parts.push(pos_sin);
            neg_cos_parts.push(neg_cos);
            neg_sin_parts.push(neg_sin);
        }

        Ok(Self {
            axes_dims,
            axis_half_dims,
            axis_offsets,
            pos_cos: Tensor::cat(&pos_cos_parts, D::Minus1)?,
            pos_sin: Tensor::cat(&pos_sin_parts, D::Minus1)?,
            neg_cos: Tensor::cat(&neg_cos_parts, D::Minus1)?,
            neg_sin: Tensor::cat(&neg_sin_parts, D::Minus1)?,
            dtype,
            image_cache: Mutex::new(HashMap::new()),
            text_cache: Mutex::new(HashMap::new()),
        })
    }

    fn rope_params(
        index: &[f32],
        dim: usize,
        theta: f64,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|k| 1.0 / (theta as f32).powf(k as f32 / dim as f32))
            .collect();
        let index = Tensor::from_vec(index.to_vec(), index.len(), device)?;
        let inv_freq = Tensor::from_vec(inv_freq, dim / 2, device)?;
        let freqs = index.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    fn axis_slice(&self, table: &Tensor, axis: usize) -> Result<Tensor> {
        table.narrow(1, self.axis_offsets[axis], self.axis_half_dims[axis])
    }

    fn leading_axis_freqs(&self, table: &Tensor, axis: usize, len: usize) -> Result<Tensor> {
        if len > ROPE_CACHE_LEN {
            candle_core::bail!("Qwen RoPE length {len} exceeds cache size {ROPE_CACHE_LEN}");
        }
        self.axis_slice(table, axis)?.narrow(0, 0, len)
    }

    fn leading_axis_freqs_with_offset(
        &self,
        table: &Tensor,
        axis: usize,
        len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        if offset + len > ROPE_CACHE_LEN {
            candle_core::bail!(
                "Qwen RoPE slice [{}..{}) exceeds cache size {ROPE_CACHE_LEN}",
                offset,
                offset + len
            );
        }
        self.axis_slice(table, axis)?.narrow(0, offset, len)
    }

    fn centered_axis_freqs(
        &self,
        pos_table: &Tensor,
        neg_table: &Tensor,
        axis: usize,
        len: usize,
    ) -> Result<Tensor> {
        if len > ROPE_CACHE_LEN {
            candle_core::bail!("Qwen RoPE length {len} exceeds cache size {ROPE_CACHE_LEN}");
        }
        let pos_len = len / 2;
        let neg_len = len - pos_len;
        let pos_axis = self.axis_slice(pos_table, axis)?;
        let neg_axis = self.axis_slice(neg_table, axis)?;
        match (neg_len, pos_len) {
            (0, _) => pos_axis.narrow(0, 0, pos_len),
            (_, 0) => neg_axis.narrow(0, ROPE_CACHE_LEN - neg_len, neg_len),
            _ => Tensor::cat(
                &[
                    neg_axis.narrow(0, ROPE_CACHE_LEN - neg_len, neg_len)?,
                    pos_axis.narrow(0, 0, pos_len)?,
                ],
                0,
            ),
        }
    }

    fn to_target(&self, tensor: Tensor, device: &Device) -> Result<Tensor> {
        tensor.to_device(device)?.to_dtype(self.dtype)
    }

    /// Text RoPE for one `(max_vid_index, txt_len)` pair, cached separately
    /// from the image tables so a new prompt length never re-materialises them.
    fn text_tables(
        &self,
        max_vid_index: usize,
        txt_len: usize,
        device: &Device,
    ) -> Result<RopeTables> {
        if max_vid_index + txt_len > ROPE_CACHE_LEN {
            candle_core::bail!(
                "Qwen text RoPE slice [{}..{}) exceeds cache size {}",
                max_vid_index,
                max_vid_index + txt_len,
                ROPE_CACHE_LEN
            );
        }
        let key = TextRopeKey {
            max_vid_index,
            txt_len,
            device: RopeCacheDevice::from_device(device),
        };
        if let Some(cached) = self.text_cache.lock().unwrap().get(&key) {
            return Ok(cached.clone());
        }
        let value = (
            self.to_target(self.pos_cos.narrow(0, max_vid_index, txt_len)?, device)?,
            self.to_target(self.pos_sin.narrow(0, max_vid_index, txt_len)?, device)?,
        );
        self.text_cache.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }

    pub(crate) fn forward(
        &self,
        frame: usize,
        height: usize,
        width: usize,
        max_txt_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        if self.axes_dims.len() != 3 {
            candle_core::bail!(
                "Qwen RoPE expects exactly 3 axes, got {}",
                self.axes_dims.len()
            );
        }

        let max_vid_index = (height / 2).max(width / 2);
        let (txt_cos, txt_sin) = self.text_tables(max_vid_index, max_txt_seq_len, device)?;

        let image_key = ImageRopeKey {
            frame,
            height,
            width,
            device: RopeCacheDevice::from_device(device),
        };
        if let Some((img_cos, img_sin)) = self.image_cache.lock().unwrap().get(&image_key) {
            return Ok((img_cos.clone(), img_sin.clone(), txt_cos, txt_sin));
        }

        let frame_cos = self.leading_axis_freqs(&self.pos_cos, 0, frame)?;
        let frame_sin = self.leading_axis_freqs(&self.pos_sin, 0, frame)?;
        let height_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 1, height)?;
        let height_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 1, height)?;
        let width_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 2, width)?;
        let width_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 2, width)?;

        let frame_half = self.axis_half_dims[0];
        let height_half = self.axis_half_dims[1];
        let width_half = self.axis_half_dims[2];
        let total_half = frame_half + height_half + width_half;
        let seq_len = frame * height * width;

        let img_cos = Tensor::cat(
            &[
                frame_cos
                    .reshape((frame, 1, 1, frame_half))?
                    .expand((frame, height, width, frame_half))?,
                height_cos.reshape((1, height, 1, height_half))?.expand((
                    frame,
                    height,
                    width,
                    height_half,
                ))?,
                width_cos
                    .reshape((1, 1, width, width_half))?
                    .expand((frame, height, width, width_half))?,
            ],
            D::Minus1,
        )?
        .reshape((seq_len, total_half))?;
        let img_sin = Tensor::cat(
            &[
                frame_sin
                    .reshape((frame, 1, 1, frame_half))?
                    .expand((frame, height, width, frame_half))?,
                height_sin.reshape((1, height, 1, height_half))?.expand((
                    frame,
                    height,
                    width,
                    height_half,
                ))?,
                width_sin
                    .reshape((1, 1, width, width_half))?
                    .expand((frame, height, width, width_half))?,
            ],
            D::Minus1,
        )?
        .reshape((seq_len, total_half))?;

        let image_tables = (
            self.to_target(img_cos, device)?,
            self.to_target(img_sin, device)?,
        );
        self.image_cache
            .lock()
            .unwrap()
            .insert(image_key, image_tables.clone());
        Ok((image_tables.0, image_tables.1, txt_cos, txt_sin))
    }

    pub(crate) fn forward_shapes(
        &self,
        img_shapes: &[(usize, usize, usize)],
        max_txt_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        if img_shapes.is_empty() {
            candle_core::bail!("img_shapes must contain at least one image shape");
        }
        if self.axes_dims.len() != 3 {
            candle_core::bail!(
                "Qwen RoPE expects exactly 3 axes, got {}",
                self.axes_dims.len()
            );
        }

        let frame_half = self.axis_half_dims[0];
        let height_half = self.axis_half_dims[1];
        let width_half = self.axis_half_dims[2];
        let total_half = frame_half + height_half + width_half;

        let mut img_cos_parts = Vec::with_capacity(img_shapes.len());
        let mut img_sin_parts = Vec::with_capacity(img_shapes.len());
        let mut max_vid_index = 0usize;

        for (shape_index, &(frame, height, width)) in img_shapes.iter().enumerate() {
            let frame_cos =
                self.leading_axis_freqs_with_offset(&self.pos_cos, 0, frame, shape_index)?;
            let frame_sin =
                self.leading_axis_freqs_with_offset(&self.pos_sin, 0, frame, shape_index)?;
            let height_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 1, height)?;
            let height_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 1, height)?;
            let width_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 2, width)?;
            let width_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 2, width)?;
            let seq_len = frame * height * width;

            img_cos_parts.push(
                Tensor::cat(
                    &[
                        frame_cos
                            .reshape((frame, 1, 1, frame_half))?
                            .expand((frame, height, width, frame_half))?,
                        height_cos.reshape((1, height, 1, height_half))?.expand((
                            frame,
                            height,
                            width,
                            height_half,
                        ))?,
                        width_cos
                            .reshape((1, 1, width, width_half))?
                            .expand((frame, height, width, width_half))?,
                    ],
                    D::Minus1,
                )?
                .reshape((seq_len, total_half))?,
            );
            img_sin_parts.push(
                Tensor::cat(
                    &[
                        frame_sin
                            .reshape((frame, 1, 1, frame_half))?
                            .expand((frame, height, width, frame_half))?,
                        height_sin.reshape((1, height, 1, height_half))?.expand((
                            frame,
                            height,
                            width,
                            height_half,
                        ))?,
                        width_sin
                            .reshape((1, 1, width, width_half))?
                            .expand((frame, height, width, width_half))?,
                    ],
                    D::Minus1,
                )?
                .reshape((seq_len, total_half))?,
            );
            max_vid_index = max_vid_index.max(height / 2).max(width / 2);
        }

        // The multi-shape image tables are not cached (an edit request's shape
        // list is effectively per-request); the text half still is.
        let (txt_cos, txt_sin) = self.text_tables(max_vid_index, max_txt_seq_len, device)?;

        Ok((
            self.to_target(
                Tensor::cat(&img_cos_parts.iter().collect::<Vec<_>>(), 0)?,
                device,
            )?,
            self.to_target(
                Tensor::cat(&img_sin_parts.iter().collect::<Vec<_>>(), 0)?,
                device,
            )?,
            txt_cos,
            txt_sin,
        ))
    }

    /// Number of cached image tables. Test-only observation of the cache split.
    #[cfg(test)]
    pub(crate) fn image_cache_len(&self) -> usize {
        self.image_cache.lock().unwrap().len()
    }
}

pub(crate) fn build_edit_modulation_index(
    img_shapes: &[(usize, usize, usize)],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let output_tokens = img_shapes
        .first()
        .map(|(f, h, w)| f * h * w)
        .unwrap_or_default();
    let condition_tokens: usize = img_shapes.iter().skip(1).map(|(f, h, w)| f * h * w).sum();
    let row_len = output_tokens + condition_tokens;
    let mut flat = Vec::with_capacity(batch_size * row_len);
    for batch_idx in 0..batch_size {
        let base = (batch_idx as u32) * 2;
        flat.extend(std::iter::repeat_n(base, output_tokens));
        flat.extend(std::iter::repeat_n(base + 1, condition_tokens));
    }
    Tensor::from_vec(flat, (batch_size, row_len), device)
}

pub(crate) fn select_modulation_params(
    mod_params: &Tensor,
    modulate_index: &Tensor,
) -> Result<Tensor> {
    let (batch, seq_len) = modulate_index.dims2()?;
    let hidden = mod_params.dim(1)?;
    mod_params
        .index_select(&modulate_index.flatten_all()?, 0)?
        .reshape((batch, seq_len, hidden))
}

struct TimestepProjEmbeddings {
    linear1: QwenLinear,
    linear2: QwenLinear,
}

impl TimestepProjEmbeddings {
    fn new(vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("time_text_embed").pp("timestep_embedder");
        Ok(Self {
            linear1: qlinear(&vb, "linear_1")?,
            linear2: qlinear(&vb, "linear_2")?,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let half = FREQUENCY_EMBEDDING_SIZE / 2;
        let freqs = Tensor::arange(0u32, half as u32, t.device())?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

struct ApproximateGelu {
    proj: QwenLinear,
}

impl ApproximateGelu {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: qlinear(&vb, "proj")?,
        })
    }
}

impl Module for ApproximateGelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.proj)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)
    }
}

struct FeedForward {
    act: ApproximateGelu,
    out: QwenLinear,
}

impl FeedForward {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: ApproximateGelu::new(vb.pp("net").pp("0"))?,
            out: qlinear(&vb.pp("net"), "2")?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(x)?.apply(&self.out)
    }
}

struct QkNorm {
    norm_q: DynamicRmsNorm,
    norm_k: DynamicRmsNorm,
}

impl QkNorm {
    fn new(eps: f64, vb: &VarBuilder, q_name: &str, k_name: &str) -> Result<Self> {
        Ok(Self {
            norm_q: dequant_rms_norm(vb, q_name, eps)?,
            norm_k: dequant_rms_norm(vb, k_name, eps)?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.norm_q.forward(q)?, self.norm_k.forward(k)?))
    }
}

struct JointAttention {
    to_q: QwenLinear,
    to_k: QwenLinear,
    to_v: QwenLinear,
    to_out: QwenLinear,
    add_q_proj: QwenLinear,
    add_k_proj: QwenLinear,
    add_v_proj: QwenLinear,
    add_out_proj: QwenLinear,
    qk_norm: QkNorm,
    added_qk_norm: QkNorm,
    n_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: qlinear(&vb, "to_q")?,
            to_k: qlinear(&vb, "to_k")?,
            to_v: qlinear(&vb, "to_v")?,
            to_out: qlinear(&vb.pp("to_out"), "0")?,
            add_q_proj: qlinear(&vb, "add_q_proj")?,
            add_k_proj: qlinear(&vb, "add_k_proj")?,
            add_v_proj: qlinear(&vb, "add_v_proj")?,
            add_out_proj: qlinear(&vb, "to_add_out")?,
            qk_norm: QkNorm::new(1e-6, &vb, "norm_q", "norm_k")?,
            added_qk_norm: QkNorm::new(1e-6, &vb, "norm_added_q", "norm_added_k")?,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        bias: Option<&Tensor>,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        img_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _, _) = img_hidden.dims3()?;
        let txt_seq_len = txt_hidden.dim(1)?;

        let q_img = img_hidden.apply(&self.to_q)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k_img = img_hidden.apply(&self.to_k)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v_img = img_hidden.apply(&self.to_v)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;

        let q_txt = txt_hidden.apply(&self.add_q_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k_txt = txt_hidden.apply(&self.add_k_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v_txt = txt_hidden.apply(&self.add_v_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;

        let (q_img, k_img) = self.qk_norm.forward(&q_img, &k_img)?;
        let (q_txt, k_txt) = self.added_qk_norm.forward(&q_txt, &k_txt)?;

        let q_img = apply_rotary_emb(&q_img, img_cos, img_sin)?;
        let k_img = apply_rotary_emb(&k_img, img_cos, img_sin)?;
        let q_txt = apply_rotary_emb(&q_txt, txt_cos, txt_sin)?;
        let k_txt = apply_rotary_emb(&k_txt, txt_cos, txt_sin)?;

        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = super::attention::joint_attention(&q, &k, &v, scale, bias)?;

        let total_seq_len = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((batch, total_seq_len, ()))?;
        // contiguous() required: narrow() creates non-contiguous strided views;
        // candle's matmul expects contiguous input on CUDA.
        let txt_attn = attn.narrow(1, 0, txt_seq_len)?.contiguous()?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?.contiguous()?;

        Ok((
            img_attn.apply(&self.to_out)?,
            txt_attn.apply(&self.add_out_proj)?,
        ))
    }
}

struct QwenImageTransformerBlock {
    img_norm1: LayerNormNoParams,
    img_norm2: LayerNormNoParams,
    txt_norm1: LayerNormNoParams,
    txt_norm2: LayerNormNoParams,
    attn: JointAttention,
    img_mlp: FeedForward,
    txt_mlp: FeedForward,
    img_mod: QwenLinear,
    txt_mod: QwenLinear,
}

impl QwenImageTransformerBlock {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            img_norm1: LayerNormNoParams::new(1e-6),
            img_norm2: LayerNormNoParams::new(1e-6),
            txt_norm1: LayerNormNoParams::new(1e-6),
            txt_norm2: LayerNormNoParams::new(1e-6),
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: FeedForward::new(vb.pp("img_mlp"))?,
            txt_mlp: FeedForward::new(vb.pp("txt_mlp"))?,
            img_mod: qlinear(&vb.pp("img_mod"), "1")?,
            txt_mod: qlinear(&vb.pp("txt_mod"), "1")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        bias: Option<&Tensor>,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;
        let temb = temb.silu()?;
        let img_mod = temb.apply(&self.img_mod)?;
        let img_mod = if let Some(modulate_index) = modulate_index {
            select_modulation_params(&img_mod, modulate_index)?
        } else {
            img_mod.unsqueeze(1)?
        };
        let txt_temb = if modulate_index.is_some() {
            temb.narrow(0, 0, txt_hidden.dim(0)?)?
        } else {
            temb.clone()
        };
        let txt_mod = txt_temb.apply(&self.txt_mod)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(6, D::Minus1)?;
        let txt_chunks = txt_mod.chunk(6, D::Minus1)?;
        let (
            img_shift_msa,
            img_scale_msa,
            img_gate_msa,
            img_shift_mlp,
            img_scale_mlp,
            img_gate_mlp,
        ) = (
            &img_chunks[0],
            &img_chunks[1],
            &img_chunks[2],
            &img_chunks[3],
            &img_chunks[4],
            &img_chunks[5],
        );
        let (
            txt_shift_msa,
            txt_scale_msa,
            txt_gate_msa,
            txt_shift_mlp,
            txt_scale_mlp,
            txt_gate_mlp,
        ) = (
            &txt_chunks[0],
            &txt_chunks[1],
            &txt_chunks[2],
            &txt_chunks[3],
            &txt_chunks[4],
            &txt_chunks[5],
        );

        let img_attn_in = self
            .img_norm1
            .forward(img_hidden)?
            .broadcast_mul(&(img_scale_msa + 1.0)?)?
            .broadcast_add(img_shift_msa)?;
        let txt_attn_in = self
            .txt_norm1
            .forward(txt_hidden)?
            .broadcast_mul(&(txt_scale_msa + 1.0)?)?
            .broadcast_add(txt_shift_msa)?;
        let (img_attn, txt_attn) = self.attn.forward(
            &img_attn_in,
            &txt_attn_in,
            bias,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            img_seq_len,
        )?;

        // Match the BF16 path and upstream Qwen masking semantics: any text
        // mask is consumed inside attention and text-conditioning, not
        // multiplied back into each residual update.
        let img_hidden = (img_hidden + img_gate_msa.broadcast_mul(&img_attn)?)?;
        let txt_hidden = (txt_hidden + txt_gate_msa.broadcast_mul(&txt_attn)?)?;

        let img_mlp_in = self
            .img_norm2
            .forward(&img_hidden)?
            .broadcast_mul(&(img_scale_mlp + 1.0)?)?
            .broadcast_add(img_shift_mlp)?;
        let txt_mlp_in = self
            .txt_norm2
            .forward(&txt_hidden)?
            .broadcast_mul(&(txt_scale_mlp + 1.0)?)?
            .broadcast_add(txt_shift_mlp)?;
        let img_ff = self.img_mlp.forward(&img_mlp_in)?;

        let img_hidden = (&img_hidden + img_gate_mlp.broadcast_mul(&img_ff)?)?;
        let txt_hidden =
            (&txt_hidden + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mlp_in)?)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

struct OutputLayer {
    norm_final: LayerNormNoParams,
    adaln_linear: QwenLinear,
    linear: QwenLinear,
}

impl OutputLayer {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            adaln_linear: qlinear(&vb.pp("norm_out"), "linear")?,
            linear: qlinear(&vb, "proj_out")?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        let scale = chunks[0].unsqueeze(1)?;
        let shift = chunks[1].unsqueeze(1)?;
        let x = self
            .norm_final
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        x.apply(&self.linear)
    }
}

pub(crate) struct QuantizedQwenImageTransformer2DModel {
    time_embed: TimestepProjEmbeddings,
    img_in: QwenLinear,
    txt_in: QwenLinear,
    txt_norm: DynamicRmsNorm,
    blocks: Vec<QwenImageTransformerBlock>,
    rope_embedder: QwenRopeEmbedder,
    output_layer: OutputLayer,
    cfg: QwenImageConfig,
    supports_cfg_batching: bool,
}

impl QuantizedQwenImageTransformer2DModel {
    pub fn new(
        cfg: &QwenImageConfig,
        vb: VarBuilder,
        device: &Device,
        supports_cfg_batching: bool,
    ) -> Result<Self> {
        let time_embed = TimestepProjEmbeddings::new(vb.clone())?;
        let img_in = qlinear(&vb, "img_in")?;
        let txt_in = qlinear(&vb, "txt_in")?;
        let txt_norm = dequant_rms_norm(&vb, "txt_norm", cfg.norm_eps)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(cfg, vb_blocks.pp(i))?);
        }

        // RoPE source tables stay on CPU; device-local views are cached by shape.
        let rope_dtype = working_dtype(device);
        let rope_embedder = QwenRopeEmbedder::new(
            10000.0,
            cfg.axes_dims_rope.clone(),
            &Device::Cpu,
            rope_dtype,
        )?;
        let output_layer = OutputLayer::new(vb)?;

        Ok(Self {
            time_embed,
            img_in,
            txt_in,
            txt_norm,
            blocks,
            rope_embedder,
            output_layer,
            cfg: cfg.clone(),
            supports_cfg_batching,
        })
    }

    pub fn supports_cfg_batching(&self) -> bool {
        self.supports_cfg_batching
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let out_dtype = x.dtype();
        let device = x.device();

        // CUDA: BF16 (matches training dtype, halves activation memory).
        // Metal/CPU: F32 (QMatMul dequantizes to F32 internally on Metal).
        let dtype = working_dtype(device);
        let x = x.to_dtype(dtype)?;
        let t = t.to_dtype(dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(dtype)?;
        debug_stage("inputs prepared");

        let (batch, channels, height, width) = x.dims4()?;
        let patch_size = self.cfg.patch_size;
        let temb = self.time_embed.forward(&t)?;
        debug_stage("time embedding");

        let height_patches = height / patch_size;
        let width_patches = width / patch_size;
        let x_packed = x
            .reshape((
                batch,
                channels,
                height_patches,
                patch_size,
                width_patches,
                patch_size,
            ))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((
                batch,
                height_patches * width_patches,
                channels * patch_size * patch_size,
            ))?
            .contiguous()?;

        let mut img = x_packed.apply(&self.img_in)?;
        debug_stage("image stem");
        let txt_normed = self.txt_norm.forward(&encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;
        debug_stage("text stem");

        let h_tokens = height / patch_size;
        let w_tokens = width / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;
        debug_stage("rope");

        let key_bias = super::attention::joint_key_bias(
            encoder_attention_mask,
            height_patches * width_patches,
            dtype,
            device,
        )?;
        let key_bias = super::attention::hoist_bias_for_device(
            key_bias,
            self.cfg.num_attention_heads,
            txt_seq_len + height_patches * width_patches,
            dtype,
            device,
        )?;

        for (i, block) in self.blocks.iter().enumerate() {
            (img, txt) = block.forward(
                &img,
                &txt,
                key_bias.as_ref(),
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                None,
            )?;
            if i == 0 || i + 1 == self.blocks.len() {
                debug_stage(&format!("block {}", i + 1));
            }
        }

        let img_out = self.output_layer.forward(&img, &temb)?;
        debug_stage("output layer");
        let out_channels = self.cfg.out_channels;
        let x_out = img_out
            .reshape((
                batch,
                height_patches,
                width_patches,
                out_channels,
                patch_size,
                patch_size,
            ))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((batch, out_channels, height, width))?
            .contiguous()?;

        x_out.to_dtype(out_dtype)
    }

    pub fn forward_packed(
        &self,
        packed_hidden_states: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        let out_dtype = packed_hidden_states.dtype();
        let device = packed_hidden_states.device();
        let dtype = working_dtype(device);
        let mut timestep = t.to_dtype(dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(dtype)?;
        let packed_hidden_states = packed_hidden_states.to_dtype(dtype)?;
        let batch = packed_hidden_states.dim(0)?;

        let modulate_index = if self.cfg.zero_cond_t {
            timestep = Tensor::cat(&[&timestep, &(timestep.zeros_like()?)], 0)?;
            Some(build_edit_modulation_index(img_shapes, batch, device)?)
        } else {
            None
        };

        let temb = self.time_embed.forward(&timestep)?;
        let mut img = packed_hidden_states.apply(&self.img_in)?;
        let txt_normed = self.txt_norm.forward(&encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward_shapes(img_shapes, txt_seq_len, device)?;

        let key_bias =
            super::attention::joint_key_bias(encoder_attention_mask, img.dim(1)?, dtype, device)?;
        let key_bias = super::attention::hoist_bias_for_device(
            key_bias,
            self.cfg.num_attention_heads,
            txt_seq_len + img.dim(1)?,
            dtype,
            device,
        )?;

        for block in &self.blocks {
            (img, txt) = block.forward(
                &img,
                &txt,
                key_bias.as_ref(),
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                modulate_index.as_ref(),
            )?;
        }

        let out_temb = if self.cfg.zero_cond_t {
            temb.narrow(0, 0, batch)?
        } else {
            temb
        };
        self.output_layer
            .forward(&img, &out_temb)?
            .to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_edit_modulation_index, cuda_mmq_supported, parse_qwen_qmatmul, select_linear_kind,
        select_modulation_params, ImageRopeKey, LinearDevice, QwenLinear, QwenLinearKind,
        QwenRopeEmbedder, RopeCacheDevice, TextRopeKey,
    };
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Module, Tensor};
    use std::sync::Arc;

    /// The ten types `fast_mmq::supports` accepts, and the ones it does not.
    /// A weight outside the accepted set reaches `dequantize_matmul`, which
    /// cannot read the BF16 activation, so it must never select `QMatMul`.
    #[test]
    fn cuda_mmq_supported_matches_candle_kernel_contract() {
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            assert!(cuda_mmq_supported(dtype), "{dtype:?} must reach MMQ");
        }
        for dtype in [
            GgmlDType::F16,
            GgmlDType::F32,
            GgmlDType::BF16,
            GgmlDType::Q8_1,
            GgmlDType::Q8K,
        ] {
            assert!(!cuda_mmq_supported(dtype), "{dtype:?} must keep dequant");
        }
    }

    #[test]
    fn qwen_qmatmul_env_parses() {
        assert!(parse_qwen_qmatmul(None));
        assert!(parse_qwen_qmatmul(Some("1")));
        assert!(parse_qwen_qmatmul(Some("")));
        // A value we do not understand keeps the shipped fast path.
        assert!(parse_qwen_qmatmul(Some("garbage")));
        assert!(!parse_qwen_qmatmul(Some("0")));
        assert!(!parse_qwen_qmatmul(Some(" 0 ")));
    }

    #[test]
    fn cuda_selects_qmatmul_only_when_every_kernel_precondition_holds() {
        let choose = |dtype, same_device, enabled| {
            select_linear_kind(LinearDevice::Cuda, dtype, same_device, enabled)
        };
        assert_eq!(
            choose(GgmlDType::Q6K, true, true),
            QwenLinearKind::QMatMul,
            "an MMQ dtype resident on the CUDA device is the whole point"
        );
        assert_eq!(
            choose(GgmlDType::Q6K, true, false),
            QwenLinearKind::Dequant,
            "MOLD_QWEN_QMATMUL=0 must restore the old arm"
        );
        assert_eq!(
            choose(GgmlDType::Q6K, false, true),
            QwenLinearKind::Dequant,
            "a CPU-staged weight would hit candle's unreachable! in the CUDA matmul"
        );
        assert_eq!(
            choose(GgmlDType::F16, true, true),
            QwenLinearKind::Dequant,
            "a float-stored tensor falls through to a fallback that rejects BF16"
        );
    }

    #[test]
    fn metal_keeps_qmatmul_and_cpu_keeps_dequant_regardless_of_the_switch() {
        for enabled in [true, false] {
            assert_eq!(
                select_linear_kind(LinearDevice::Metal, GgmlDType::Q4K, true, enabled),
                QwenLinearKind::QMatMul,
            );
            assert_eq!(
                select_linear_kind(LinearDevice::Other, GgmlDType::Q4K, true, enabled),
                QwenLinearKind::Dequant,
            );
        }
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The flip must be the same operation, not a different one: both arms
    /// wrap the same quantized weight and must agree to quantization error.
    /// The tolerance is relative because the quantized arm also quantizes the
    /// activation (Q8 vec-dot on CPU, int8 MMQ on CUDA) — this pins that the
    /// two arms compute the same linear layer, not that they are bit-equal.
    #[test]
    fn qmatmul_and_dequant_agree_on_cpu() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (64, 128), &device).unwrap();
        let bias = Tensor::randn(0f32, 1f32, 64, &device).unwrap();
        let quantized = Arc::new(QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap());

        let dequant = QwenLinear::Dequant {
            weight: quantized.clone(),
            bias: Some(bias.clone()),
        };
        let qmatmul = QwenLinear::QMatMul(
            mold_candle::quantized_nn::Linear::from_arc(quantized, Some(bias)).unwrap(),
        );

        let x = Tensor::randn(0f32, 1f32, (2, 5, 128), &device).unwrap();
        let a = dequant.forward(&x).unwrap();
        let b = qmatmul.forward(&x).unwrap();
        let scale = max_abs(&a);
        let diff = max_abs(&(a - b).unwrap());
        assert!(
            diff < 0.05 * scale,
            "arms disagree by {diff} against an output scale of {scale}"
        );
    }

    /// A transposed activation must not reach candle's kernels un-materialized:
    /// both CUDA fast paths decline a non-contiguous rhs and the fallback they
    /// decline into cannot read a BF16 activation.
    #[test]
    fn qmatmul_arm_materializes_a_non_contiguous_activation() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (8, 64), &device).unwrap();
        let quantized = Arc::new(QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap());
        let layer = QwenLinear::QMatMul(
            mold_candle::quantized_nn::Linear::from_arc(quantized, None).unwrap(),
        );

        let x = Tensor::randn(0f32, 1f32, (64, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        assert_eq!(layer.forward(&x).unwrap().dims(), &[3, 8]);
    }

    #[test]
    fn rope_cache_device_detects_cpu() {
        assert!(matches!(
            RopeCacheDevice::from_device(&Device::Cpu),
            RopeCacheDevice::Cpu
        ));
    }

    #[test]
    fn rope_cache_key_includes_shape_and_device() {
        let a = ImageRopeKey {
            frame: 1,
            height: 64,
            width: 64,
            device: RopeCacheDevice::Cpu,
        };
        let b = ImageRopeKey {
            frame: 1,
            height: 64,
            width: 64,
            device: RopeCacheDevice::Metal,
        };
        assert_ne!(a, b);

        let text_a = TextRopeKey {
            max_vid_index: 32,
            txt_len: 512,
            device: RopeCacheDevice::Cpu,
        };
        let text_b = TextRopeKey {
            max_vid_index: 32,
            txt_len: 256,
            device: RopeCacheDevice::Cpu,
        };
        assert_ne!(text_a, text_b);
    }

    /// The image tables are the expensive half and do not depend on the prompt
    /// length, so two prompts at one resolution must share a single entry.
    ///
    /// Before the split this was keyed on `max(cond_len, uncond_len)`, so every
    /// new prompt pair re-materialised a `seq_len x total_half` pair — and once
    /// split CFG stopped padding the two streams to a common length it would
    /// have produced two entries per generation instead of one.
    #[test]
    fn rope_image_table_is_shared_across_text_lengths() {
        let device = Device::Cpu;
        let embedder =
            QwenRopeEmbedder::new(10000.0, vec![16, 56, 56], &device, DType::F32).unwrap();

        let (cos_a, sin_a, txt_cos_a, _) = embedder.forward(1, 8, 8, 20, &device).unwrap();
        let (cos_b, sin_b, txt_cos_b, _) = embedder.forward(1, 8, 8, 12, &device).unwrap();

        assert_eq!(
            embedder.image_cache_len(),
            1,
            "two prompt lengths at one resolution must share one image entry"
        );
        assert_eq!(cos_a.dims(), cos_b.dims());
        let diff = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        assert_eq!(diff(&cos_a, &cos_b), 0.0);
        assert_eq!(diff(&sin_a, &sin_b), 0.0);

        // Text RoPE is a prefix slice: the shorter stream sees exactly what the
        // longer one saw at those positions, so slicing never moves a real token.
        assert_eq!(txt_cos_a.dims(), &[20, 64]);
        assert_eq!(txt_cos_b.dims(), &[12, 64]);
        assert_eq!(diff(&txt_cos_a.narrow(0, 0, 12).unwrap(), &txt_cos_b), 0.0);

        // A different resolution is a different image entry.
        embedder.forward(1, 8, 16, 12, &device).unwrap();
        assert_eq!(embedder.image_cache_len(), 2);
    }

    #[test]
    fn edit_modulation_index_is_precomputed_per_batch_on_device() {
        let index = build_edit_modulation_index(&[(1, 1, 2), (1, 1, 1)], 2, &Device::Cpu).unwrap();
        let rows = index.to_vec2::<u32>().unwrap();
        assert_eq!(rows, vec![vec![0, 0, 1], vec![2, 2, 3]]);
    }

    #[test]
    fn select_modulation_params_uses_precomputed_indices_without_rebasing() {
        let mod_params =
            Tensor::from_vec(vec![10f32, 20., 30., 40.], (4, 1), &Device::Cpu).unwrap();
        let index = Tensor::from_vec(vec![0u32, 0, 1, 2, 2, 3], (2, 3), &Device::Cpu).unwrap();
        let selected = select_modulation_params(&mod_params, &index).unwrap();
        let rows = selected.squeeze(2).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(rows, vec![vec![10.0, 10.0, 20.0], vec![30.0, 30.0, 40.0]]);
    }
}
