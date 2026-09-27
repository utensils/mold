//! Tier-agnostic linears for the Qwen Image 2.1 transformer.
//!
//! One transformer type serves every published tier. The format is decided
//! once, from the checkpoint header
//! ([`crate::artifact_format::probe_qwen_image21_transformer`]), and the
//! transformer asks [`Q21WeightSource`] for linears BY NAME; the answer is a
//! [`Q21Linear`] whose arm matches the checkpoint:
//!
//! | Tier | Arm | Forward |
//! |---|---|---|
//! | `bf16` (diffusers shards) | [`Q21Weight::Dense`] | `candle_nn::Linear` — bit-identical to the pre-tier loader |
//! | GGUF (`q8`…`q2`, leejet or unsloth) | [`Q21Weight::Quant`] | [`crate::quantized_linear::QuantizedLinear`]: per-forward dequant on CUDA by default, `QMatMul` behind `MOLD_QWEN_IMAGE21_QMATMUL=1`, `QMatMul` on Metal |
//! | `int8-conv` (Comfy-Org) | [`Q21Weight::Int8`] | `mold_candle::comfy_int8` W8A8 — the cuBLASLt INT8 kernel on CUDA, the portable reference elsewhere |
//! | `fp8` (unsloth torchao) | [`Q21Weight::Fp8`] | widen F8E4M3 per forward, per-row scale on the OUTPUT (the Qwen-Image 2512 FP8 rule) |
//!
//! Every arm carries the same LoRA adapter slot. LoRA on Qwen Image 2.1 is
//! ALWAYS bypass (`W x + s·B(A x)`), never merged: Viggle's turbo LoRA is
//! specified unmerged, merging into BF16 is lossy, and a quantized arm could
//! only merge through a dequant→requant round trip. [`Q21Linear::set_adapters`]
//! swaps a stack without rebuilding the base weight.
//!
//! The ComfyUI and GGUF tiers FUSE the MLP's gate and up projections into one
//! `img_mlp.gate_up` linear with the gate rows first — ComfyUI
//! `comfy/ldm/qwen_image21/model.py:53-63` builds `[gate; up]` and
//! `comfy/ops.py:950-952` (`_swiglu_eager`) reads it as
//! `gate, up = x.chunk(2, dim=-1); silu(gate) * up`, where `up` is diffusers'
//! `proj` (`transformer_qwenimage21.py:206-212`). [`Q21GateUp`] keeps a fused
//! checkpoint fused — one GEMM, one dequant — and [`split_gate_up`] is the one
//! place the halves are named.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor, D};
use mold_candle::comfy_int8::{ComfyInt8ConvRotLinear, CONVROT_GROUP_SIZE};

use crate::artifact_format::QwenImage21TransformerFormat;
use crate::flux::lora_bypass::{apply_adapters, FusedSlice, LinearLoraAdapter};
use crate::quantized_linear::{parse_qmatmul_flag, QuantizedLinear, QuantizedLinearKind};

/// Opt CUDA into candle's quantized `QMatMul` fast path for the GGUF tiers.
///
/// Engine-shaping: the arms differ in numerics (int8 MMQ against a BF16 GEMM
/// over dequantized weights), transient memory and step latency, so it is
/// listed in `runtime_env::ENGINE_SHAPING_VARIABLES` and classified by
/// mold-server's `runtime_semantic_variable`.
pub(crate) const QMATMUL_ENV: &str = "MOLD_QWEN_IMAGE21_QMATMUL";

/// Parse [`QMATMUL_ENV`] — the shared `parse_qmatmul_flag`, so mold-server's
/// canonicalization and the engine read the same decision.
pub(crate) fn parse_qwen_image21_qmatmul(value: Option<&str>) -> bool {
    parse_qmatmul_flag(value)
}

/// Process-frozen [`QMATMUL_ENV`].
pub(crate) fn qmatmul_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = parse_qwen_image21_qmatmul(crate::runtime_env::value(QMATMUL_ENV).as_deref());
        if enabled {
            tracing::info!(
                "qwen-image-2.1: {QMATMUL_ENV}=1 — candle's quantized CUDA fast path enabled \
                 for GGUF tiers"
            );
        }
        enabled
    })
}

/// The base weight of one linear, by checkpoint encoding.
#[derive(Clone, Debug)]
pub(crate) enum Q21Weight {
    /// Plain float weight at the working dtype.
    Dense(candle_nn::Linear),
    /// A GGUF tensor through the shared quantized dispatch.
    Quant(QuantizedLinear),
    /// Comfy INT8 ConvRot, device-resident packed bytes plus row scales.
    Int8 {
        linear: Arc<ComfyInt8ConvRotLinear>,
        bias: Option<Tensor>,
    },
    /// torchao F8E4M3 weight with its per-output-row F32 scale (`[out]`).
    Fp8 {
        weight: Tensor,
        row_scale: Tensor,
        bias: Option<Tensor>,
    },
}

/// What a [`Q21Linear`] resolved to, for logs and tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Q21LinearKind {
    Dense,
    GgufDequant,
    GgufQMatMul,
    /// A float tensor stored inside a GGUF, dequantized once at load.
    GgufDense,
    Int8ConvRot,
    Fp8,
}

/// A Qwen Image 2.1 linear: one base [`Q21Weight`] plus a bypass LoRA stack.
#[derive(Clone, Debug)]
pub(crate) struct Q21Linear {
    weight: Q21Weight,
    in_features: usize,
    out_features: usize,
    adapters: Vec<LinearLoraAdapter>,
}

impl Q21Linear {
    /// Wrap an already-built dense linear (the BF16 path and tests).
    pub(crate) fn dense(linear: candle_nn::Linear) -> Result<Self> {
        let (out_features, in_features) = linear.weight().dims2()?;
        Ok(Self::from_weight(
            Q21Weight::Dense(linear),
            in_features,
            out_features,
        ))
    }

    fn from_weight(weight: Q21Weight, in_features: usize, out_features: usize) -> Self {
        Self {
            weight,
            in_features,
            out_features,
            adapters: Vec::new(),
        }
    }

    /// Build the torchao FP8 arm. `row_scale` is the checkpoint's
    /// `_weight_scale` (`[out, 1]` or `[out]`).
    pub(crate) fn fp8(weight: Tensor, row_scale: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        anyhow::ensure!(
            weight.dtype() == DType::F8E4M3,
            "Qwen Image 2.1 FP8 weight must be F8E4M3, got {:?}",
            weight.dtype()
        );
        anyhow::ensure!(
            row_scale.elem_count() == out_features,
            "Qwen Image 2.1 FP8 scale has {} values for {out_features} output rows",
            row_scale.elem_count()
        );
        let row_scale = row_scale.reshape(out_features)?.to_dtype(DType::F32)?;
        Ok(Self::from_weight(
            Q21Weight::Fp8 {
                weight,
                row_scale,
                bias,
            },
            in_features,
            out_features,
        ))
    }

    /// Build the Comfy INT8 ConvRot arm from packed two's-complement bytes
    /// (`U8 [out, in]`) and `F32 [out, 1]` row scales on one device.
    pub(crate) fn int8(packed: Tensor, scales: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = ComfyInt8ConvRotLinear::new_on_device(packed, scales)?;
        let (in_features, out_features) = (linear.in_features(), linear.out_features());
        Ok(Self::from_weight(
            Q21Weight::Int8 {
                linear: Arc::new(linear),
                bias,
            },
            in_features,
            out_features,
        ))
    }

    /// Build the GGUF arm through the shared quantized dispatch.
    pub(crate) fn quantized(
        weight: Arc<QTensor>,
        device: &Device,
        dtype: DType,
        qmatmul: bool,
    ) -> Result<Self> {
        let (out_features, in_features) = weight.shape().dims2()?;
        let linear = QuantizedLinear::new(weight, None, device, dtype, qmatmul)?;
        Ok(Self::from_weight(
            Q21Weight::Quant(linear),
            in_features,
            out_features,
        ))
    }

    pub(crate) fn in_features(&self) -> usize {
        self.in_features
    }

    pub(crate) fn out_features(&self) -> usize {
        self.out_features
    }

    pub(crate) fn kind(&self) -> Q21LinearKind {
        match &self.weight {
            Q21Weight::Dense(_) => Q21LinearKind::Dense,
            Q21Weight::Quant(linear) => match linear.kind() {
                QuantizedLinearKind::Dequant => Q21LinearKind::GgufDequant,
                QuantizedLinearKind::QMatMul => Q21LinearKind::GgufQMatMul,
                QuantizedLinearKind::Dense => Q21LinearKind::GgufDense,
            },
            Q21Weight::Int8 { .. } => Q21LinearKind::Int8ConvRot,
            Q21Weight::Fp8 { .. } => Q21LinearKind::Fp8,
        }
    }

    /// The installed bypass stack.
    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    pub(crate) fn adapters(&self) -> &[LinearLoraAdapter] {
        &self.adapters
    }

    /// Replace the bypass stack. The base weight is untouched, so a new
    /// request's LoRA set is an adapter swap, never a transformer rebuild.
    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    pub(crate) fn set_adapters(&mut self, adapters: Vec<LinearLoraAdapter>) -> Result<()> {
        for adapter in &adapters {
            let (rank, in_features) = adapter.down.dims2()?;
            let (rows, up_rank) = adapter.up.dims2()?;
            anyhow::ensure!(
                in_features == self.in_features && rank == up_rank,
                "LoRA adapter (down {:?}, up {:?}) does not fit a {}x{} linear",
                adapter.down.dims(),
                adapter.up.dims(),
                self.out_features,
                self.in_features
            );
            let (offset, length) = adapter
                .fused_slice
                .map_or((0, self.out_features), |slice| (slice.offset, slice.length));
            anyhow::ensure!(
                length == rows && offset + length <= self.out_features,
                "LoRA adapter writes rows [{offset}, {}) of a {}-row linear",
                offset + rows,
                self.out_features
            );
        }
        self.adapters = adapters;
        Ok(())
    }

    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    pub(crate) fn clear_adapters(&mut self) {
        self.adapters.clear();
    }

    /// Device bytes the installed adapters hold.
    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    pub(crate) fn adapter_bytes(&self) -> u64 {
        self.adapters
            .iter()
            .map(|adapter| {
                let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
                bytes(&adapter.down) + bytes(&adapter.up)
            })
            .sum()
    }

    /// The base weight's forward, with no adapters.
    fn base_forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Q21Weight::Dense(linear) => Ok(linear.forward(x)?),
            Q21Weight::Quant(linear) => Ok(linear.forward(x)?),
            Q21Weight::Int8 { linear, bias } => Ok(linear.forward(x, bias.as_ref(), x.dtype())?),
            Q21Weight::Fp8 {
                weight,
                row_scale,
                bias,
            } => fp8_forward(x, weight, row_scale, bias.as_ref()),
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.base_forward(x)?;
        if self.adapters.is_empty() {
            Ok(out)
        } else {
            apply_adapters(&self.adapters, x, out)
        }
    }
}

impl Module for Q21Linear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Q21Linear::forward(self, xs).map_err(|error| candle_core::Error::Msg(format!("{error:#}")))
    }
}

/// The Qwen-Image 2512 FP8 forward (`qwen_image/transformer.rs` `QwenLinear::Fp8`):
/// widen the F8 slab to the activation dtype, one GEMM, then the per-row scale
/// rides the (much smaller) output — `x @ (w·s)ᵀ == (x @ wᵀ)·s` for a scale
/// that is constant along each output row — so the forward keeps exactly one
/// full-size pass, the widening cast itself.
fn fp8_forward(
    x: &Tensor,
    weight: &Tensor,
    row_scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let w = weight.to_dtype(dtype)?.t()?;
    let out = match *x.dims() {
        [b1, b2, m, k] => x
            .reshape((b1 * b2 * m, k))?
            .matmul(&w)?
            .reshape((b1, b2, m, ()))?,
        [batch, m, k] => x
            .reshape((batch * m, k))?
            .matmul(&w)?
            .reshape((batch, m, ()))?,
        _ => x.matmul(&w)?,
    };
    let out = out.broadcast_mul(&row_scale.to_dtype(dtype)?)?;
    match bias {
        Some(bias) => Ok(out.broadcast_add(&bias.to_dtype(dtype)?)?),
        None => Ok(out),
    }
}

/// Split a fused `gate_up` output into `(gate, up)` — gate rows FIRST.
///
/// ComfyUI `comfy/ops.py:950-952`: `gate, up = x.chunk(2, dim=-1)`; the
/// checkpoint's fused rows are `[gate_layer; proj]`
/// (`comfy/ldm/qwen_image21/model.py:53-55`), and ComfyUI's LoRA loader maps
/// the same halves (`comfy/lora.py:331-333`: gate `(0, half)`, proj `(half,
/// half)`).
pub(crate) fn split_gate_up(fused: &Tensor, hidden: usize) -> Result<(Tensor, Tensor)> {
    let width = fused.dim(D::Minus1)?;
    anyhow::ensure!(
        width == 2 * hidden,
        "Qwen Image 2.1 gate_up output is {width} wide, expected 2 x {hidden}"
    );
    Ok((
        fused.narrow(D::Minus1, 0, hidden)?,
        fused.narrow(D::Minus1, hidden, hidden)?,
    ))
}

/// The MLP's input projections, fused or split as the checkpoint stores them.
#[derive(Clone, Debug)]
pub(crate) enum Q21GateUp {
    /// Diffusers layout (BF16, FP8): separate `gate_layer` and `proj`.
    Split { gate: Q21Linear, proj: Q21Linear },
    /// ComfyUI/GGUF layout: one `gate_up` linear, gate rows first.
    Fused { gate_up: Q21Linear, hidden: usize },
}

impl Q21GateUp {
    #[allow(dead_code)] // the LoRA installer sizes gate/proj adapters with it
    pub(crate) fn hidden(&self) -> usize {
        match self {
            Self::Split { gate, .. } => gate.out_features(),
            Self::Fused { hidden, .. } => *hidden,
        }
    }

    /// `(gate_layer(x), proj(x))` — diffusers' names for the two halves.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Split { gate, proj } => Ok((gate.forward(x)?, proj.forward(x)?)),
            Self::Fused { gate_up, hidden } => split_gate_up(&gate_up.forward(x)?, *hidden),
        }
    }

    /// `silu(gate_layer(x)) * proj(x)` — the input to `img_mlp.out`
    /// (diffusers `transformer_qwenimage21.py:212`).
    pub(crate) fn swiglu(&self, x: &Tensor) -> Result<Tensor> {
        let (gate, up) = self.forward(x)?;
        Ok((candle_nn::ops::silu(&gate)? * up)?)
    }

    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    /// Install the bypass stacks that target `gate_layer` and `proj`. On a
    /// fused checkpoint each lands on its own half of `gate_up`'s output.
    pub(crate) fn set_adapters(
        &mut self,
        gate_adapters: Vec<LinearLoraAdapter>,
        proj_adapters: Vec<LinearLoraAdapter>,
    ) -> Result<()> {
        match self {
            Self::Split { gate, proj } => {
                gate.set_adapters(gate_adapters)?;
                proj.set_adapters(proj_adapters)
            }
            Self::Fused { gate_up, hidden } => {
                let hidden = *hidden;
                let place = |adapters: Vec<LinearLoraAdapter>, offset: usize| {
                    adapters
                        .into_iter()
                        .map(|mut adapter| {
                            anyhow::ensure!(
                                adapter.fused_slice.is_none(),
                                "a gate_layer/proj LoRA adapter must address its whole half"
                            );
                            adapter.fused_slice = Some(FusedSlice {
                                offset,
                                length: hidden,
                            });
                            Ok(adapter)
                        })
                        .collect::<Result<Vec<_>>>()
                };
                let mut stack = place(gate_adapters, 0)?;
                stack.extend(place(proj_adapters, hidden)?);
                gate_up.set_adapters(stack)
            }
        }
    }

    #[allow(dead_code)] // the LoRA installer (qwen_image21::lora) calls it
    pub(crate) fn clear_adapters(&mut self) {
        match self {
            Self::Split { gate, proj } => {
                gate.clear_adapters();
                proj.clear_adapters();
            }
            Self::Fused { gate_up, .. } => gate_up.clear_adapters(),
        }
    }
}

/// Where a Qwen Image 2.1 transformer's tensors come from.
enum Q21Backend {
    Safetensors(MmapedSafetensors),
    /// A resident GGUF builder, already `pp`'d past unsloth's prefix.
    Gguf(mold_candle::quantized::VarBuilder),
    /// An ordinary dense builder (synthetic tests, or any caller that already
    /// holds one): every linear is the Dense arm, bit-identical to
    /// `candle_nn::linear_no_bias` over the same builder.
    Dense(candle_nn::VarBuilder<'static>),
}

/// The one loader every Qwen Image 2.1 tier goes through.
///
/// Construction reads the format from the header; every accessor answers in
/// the checkpoint's own terms (a fused `gate_up` stays fused, a GGUF block
/// stays quantized), so the transformer never branches on the tier.
pub(crate) struct Q21WeightSource {
    format: QwenImage21TransformerFormat,
    backend: Q21Backend,
    device: Device,
    dtype: DType,
    qmatmul: bool,
    /// GGUF block types seen, for the tier label.
    gguf_types: Vec<GgmlDType>,
}

impl Q21WeightSource {
    /// Open `paths` (all shards of one checkpoint), probing the format from
    /// the first. `dtype` is the transformer's working dtype.
    pub(crate) fn open(
        paths: &[PathBuf],
        device: &Device,
        dtype: DType,
        qmatmul: bool,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let first = paths
            .first()
            .context("Qwen Image 2.1 transformer has no checkpoint files")?;
        let format =
            crate::artifact_format::probe_qwen_image21_transformer(first).map_err(|failure| {
                anyhow::anyhow!(
                    "cannot read the Qwen Image 2.1 transformer format of {}: {failure:?}",
                    first.display()
                )
            })?;
        if format == QwenImage21TransformerFormat::TorchaoFp8 && device.is_metal() {
            bail!(
                "qwen-image-2.1:fp8 needs CUDA: candle's Metal backend has no F8E4M3 cast \
                 kernel to widen its weights. Use qwen-image-2.1:int8-conv or a GGUF tier on Metal."
            );
        }
        let backend = match format {
            QwenImage21TransformerFormat::Gguf {
                diffusion_model_prefix,
            } => {
                anyhow::ensure!(
                    paths.len() == 1,
                    "a Qwen Image 2.1 GGUF transformer is one file, got {}",
                    paths.len()
                );
                let vb = crate::weight_loader::load_gguf_var_builder(
                    first,
                    device,
                    "Qwen Image 2.1 transformer",
                    progress,
                )?;
                Q21Backend::Gguf(if diffusion_model_prefix {
                    vb.pp("model").pp("diffusion_model")
                } else {
                    vb
                })
            }
            _ => {
                let refs = paths.iter().map(PathBuf::as_path).collect::<Vec<_>>();
                // SAFETY: the mapping is read-only and lives as long as the
                // source; the files are verified model artifacts.
                let st = unsafe { MmapedSafetensors::multi(&refs) }.with_context(|| {
                    format!("mapping Qwen Image 2.1 transformer {}", first.display())
                })?;
                Q21Backend::Safetensors(st)
            }
        };
        let gguf_types = match &backend {
            Q21Backend::Gguf(vb) => {
                let mut types = vb
                    .tensors()
                    .values()
                    .map(|tensor| tensor.dtype())
                    .filter(|dtype| {
                        !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
                    })
                    .collect::<Vec<_>>();
                types.sort_by_key(|dtype| format!("{dtype:?}"));
                types.dedup();
                types
            }
            Q21Backend::Safetensors(_) | Q21Backend::Dense(_) => Vec::new(),
        };
        Ok(Self {
            format,
            backend,
            device: device.clone(),
            dtype,
            qmatmul,
            gguf_types,
        })
    }

    /// Wrap a dense `VarBuilder`. Its dtype is the working dtype.
    pub(crate) fn from_var_builder(vb: candle_nn::VarBuilder<'static>) -> Self {
        Self {
            format: QwenImage21TransformerFormat::Bf16,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            backend: Q21Backend::Dense(vb),
            qmatmul: false,
            gguf_types: Vec::new(),
        }
    }

    /// The whole source as a path builder at its root.
    pub(crate) fn root(&self) -> Q21Vb<'_> {
        Q21Vb {
            source: self,
            prefix: String::new(),
        }
    }

    /// Whether a CUDA GGUF linear may have taken the QMatMul arm, which is
    /// what the per-step finiteness guard names.
    pub(crate) fn qmatmul_guard(&self) -> bool {
        self.qmatmul && matches!(self.format, QwenImage21TransformerFormat::Gguf { .. })
    }

    #[cfg_attr(not(test), allow(dead_code))] // qualification surface
    pub(crate) fn format(&self) -> QwenImage21TransformerFormat {
        self.format
    }

    /// The tier as a log/error label: `bf16`, `int8-conv`, `fp8`, or
    /// `gguf (Q4K+Q5K+…)` naming every quantized block type the file mixes.
    pub(crate) fn tier_label(&self) -> String {
        if self.gguf_types.is_empty() {
            self.format.label().to_string()
        } else {
            let types = self
                .gguf_types
                .iter()
                .map(|dtype| format!("{dtype:?}"))
                .collect::<Vec<_>>()
                .join("+");
            format!("gguf ({types})")
        }
    }

    /// Whether the checkpoint carries tensor `name` (in the logical, unprefixed
    /// key space — for a torchao FP8 linear the `.weight` name answers for its
    /// `._weight_qdata`).
    pub(crate) fn contains(&self, name: &str) -> bool {
        match &self.backend {
            Q21Backend::Gguf(vb) => vb.contains_key(name),
            Q21Backend::Dense(vb) => vb.contains_tensor(name),
            Q21Backend::Safetensors(st) => {
                st.get(name).is_ok()
                    || name
                        .strip_suffix(".weight")
                        .is_some_and(|base| st.get(&format!("{base}._weight_qdata")).is_ok())
            }
        }
    }

    /// A dense (non-linear) tensor — norm scales and the like — at `dtype`.
    pub(crate) fn tensor(&self, name: &str, dtype: DType) -> Result<Tensor> {
        match &self.backend {
            Q21Backend::Safetensors(st) => Ok(st
                .load(name, &self.device)
                .with_context(|| format!("Qwen Image 2.1 tensor {name}"))?
                .to_dtype(dtype)?),
            Q21Backend::Gguf(vb) => Ok(vb
                .get_no_shape(name)?
                .dequantize(&self.device)?
                .to_dtype(dtype)?),
            Q21Backend::Dense(vb) => Ok(vb.get_unchecked_dtype(name, dtype)?),
        }
    }

    /// The linear `<prefix>.weight` (and `<prefix>.bias` when present), in the
    /// arm the checkpoint encodes it in. `in_features`/`out_features` are the
    /// architecture's, checked against the file.
    pub(crate) fn linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Q21Linear> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let linear = match &self.backend {
            Q21Backend::Gguf(vb) => {
                anyhow::ensure!(
                    !vb.contains_key(&bias_name),
                    "Qwen Image 2.1 GGUF linear {prefix} carries a bias the architecture has none of"
                );
                let weight = vb.get((out_features, in_features), &weight_name)?;
                Q21Linear::quantized(weight, &self.device, self.dtype, self.qmatmul)?
            }
            Q21Backend::Dense(vb) => Q21Linear::dense(candle_nn::linear_no_bias(
                in_features,
                out_features,
                vb.pp(prefix),
            )?)?,
            Q21Backend::Safetensors(st) => {
                let bias = st
                    .get(&bias_name)
                    .is_ok()
                    .then(|| st.load(&bias_name, &self.device))
                    .transpose()?;
                let qdata = format!("{prefix}._weight_qdata");
                if st.get(&qdata).is_ok() {
                    let weight = st.load(&qdata, &self.device)?;
                    let scale = st.load(&format!("{prefix}._weight_scale"), &self.device)?;
                    Q21Linear::fp8(weight, scale, bias)?
                } else {
                    let view = st
                        .get(&weight_name)
                        .with_context(|| format!("Qwen Image 2.1 linear {weight_name}"))?;
                    if format!("{:?}", view.dtype()) == "I8" {
                        let [rows, cols] = view.shape() else {
                            bail!("Qwen Image 2.1 INT8 weight {weight_name} is not rank 2");
                        };
                        anyhow::ensure!(
                            cols % CONVROT_GROUP_SIZE == 0,
                            "Qwen Image 2.1 INT8 weight {weight_name} width {cols} is not a \
                             multiple of the ConvRot group"
                        );
                        let packed = Tensor::from_slice(view.data(), (*rows, *cols), &self.device)?;
                        let scales = st
                            .load(&format!("{prefix}.weight_scale"), &self.device)?
                            .to_dtype(DType::F32)?
                            .reshape((*rows, 1))?;
                        Q21Linear::int8(packed, scales, bias)?
                    } else {
                        let weight = st.load(&weight_name, &self.device)?.to_dtype(self.dtype)?;
                        let bias = bias.map(|b| b.to_dtype(self.dtype)).transpose()?;
                        Q21Linear::dense(candle_nn::Linear::new(weight, bias))?
                    }
                }
            }
        };
        anyhow::ensure!(
            linear.in_features() == in_features && linear.out_features() == out_features,
            "Qwen Image 2.1 linear {prefix} is {}x{}, expected {out_features}x{in_features}",
            linear.out_features(),
            linear.in_features()
        );
        Ok(linear)
    }

    /// The MLP input projections under `mlp_prefix` (`transformer_blocks.N.img_mlp`),
    /// fused `gate_up` or split `gate_layer`/`proj` as the checkpoint stores them.
    pub(crate) fn gate_up(&self, mlp_prefix: &str, dim: usize, hidden: usize) -> Result<Q21GateUp> {
        let fused = format!("{mlp_prefix}.gate_up");
        if self.contains(&format!("{fused}.weight")) {
            Ok(Q21GateUp::Fused {
                gate_up: self.linear(&fused, dim, 2 * hidden)?,
                hidden,
            })
        } else {
            Ok(Q21GateUp::Split {
                gate: self.linear(&format!("{mlp_prefix}.gate_layer"), dim, hidden)?,
                proj: self.linear(&format!("{mlp_prefix}.proj"), dim, hidden)?,
            })
        }
    }

    /// Reconstruct the full-precision weight of the diffusers-named linear
    /// `prefix` as `F32 [out, in]` on `device`, whatever the tier stores.
    ///
    /// A fused checkpoint answers `…img_mlp.gate_layer` / `…img_mlp.proj` from
    /// the matching half of `gate_up`. This is the qualification surface: it is
    /// what the parity tests compare against the BF16 shards.
    #[cfg_attr(not(test), allow(dead_code))] // qualification surface
    pub(crate) fn dequantized_weight(&self, prefix: &str, device: &Device) -> Result<Tensor> {
        let half = |fused_base: &str, second: bool| -> Result<Tensor> {
            let fused = self.dequantized_weight(fused_base, device)?;
            let rows = fused.dim(0)? / 2;
            Ok(fused.narrow(0, if second { rows } else { 0 }, rows)?)
        };
        if let Some(base) = prefix.strip_suffix(".gate_layer") {
            let fused = format!("{base}.gate_up");
            if self.contains(&format!("{fused}.weight")) {
                return half(&fused, false);
            }
        }
        if let Some(base) = prefix.strip_suffix(".proj") {
            let fused = format!("{base}.gate_up");
            if self.contains(&format!("{fused}.weight")) {
                return half(&fused, true);
            }
        }
        let weight_name = format!("{prefix}.weight");
        match &self.backend {
            Q21Backend::Gguf(vb) => Ok(vb
                .get_no_shape(&weight_name)?
                .dequantize(device)?
                .to_dtype(DType::F32)?),
            Q21Backend::Dense(vb) => Ok(vb
                .get_unchecked_dtype(&weight_name, DType::F32)?
                .to_device(device)?),
            Q21Backend::Safetensors(st) => {
                let qdata = format!("{prefix}._weight_qdata");
                if st.get(&qdata).is_ok() {
                    let weight = st.load(&qdata, &Device::Cpu)?.to_dtype(DType::F32)?;
                    let rows = weight.dim(0)?;
                    let scale = st
                        .load(&format!("{prefix}._weight_scale"), &Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .reshape((rows, 1))?;
                    return Ok(weight.broadcast_mul(&scale)?.to_device(device)?);
                }
                let view = st.get(&weight_name)?;
                if format!("{:?}", view.dtype()) == "I8" {
                    let [rows, cols] = view.shape() else {
                        bail!("Qwen Image 2.1 INT8 weight {weight_name} is not rank 2");
                    };
                    let packed = Tensor::from_slice(view.data(), (*rows, *cols), device)?;
                    let scales = st
                        .load(&format!("{prefix}.weight_scale"), device)?
                        .to_dtype(DType::F32)?
                        .reshape((*rows, 1))?;
                    let linear = ComfyInt8ConvRotLinear::new_on_device(packed, scales)?;
                    return Ok(linear.dequantize_weight(
                        DType::F32,
                        device,
                        mold_candle::comfy_int8::PORTABLE_ROW_CHUNK,
                    )?);
                }
                Ok(st.load(&weight_name, device)?.to_dtype(DType::F32)?)
            }
        }
    }

    /// Every linear-weight name in the checkpoint's logical key space,
    /// normalized to diffusers' names (a fused `gate_up` answers as its
    /// `gate_layer` and `proj` halves). Used by the parity tests to cover the
    /// whole file rather than a sample.
    #[cfg_attr(not(test), allow(dead_code))] // qualification surface
    pub(crate) fn logical_linear_names(&self) -> Vec<String> {
        let names: Vec<String> = match &self.backend {
            Q21Backend::Gguf(vb) => {
                let prefix = vb.key("");
                // Every rank-2 `.weight` is a linear in this architecture; the
                // norm scales are rank 1.
                vb.tensors()
                    .iter()
                    .filter(|(_, tensor)| tensor.shape().rank() == 2)
                    .filter_map(|(key, _)| key.strip_prefix(prefix.as_str()))
                    .filter_map(|key| key.strip_suffix(".weight"))
                    .map(str::to_string)
                    .collect()
            }
            Q21Backend::Dense(_) => Vec::new(),
            Q21Backend::Safetensors(st) => st
                .tensors()
                .into_iter()
                .filter_map(|(key, view)| {
                    if let Some(base) = key.strip_suffix("._weight_qdata") {
                        return Some(base.to_string());
                    }
                    let base = key.strip_suffix(".weight")?;
                    (view.shape().len() == 2).then(|| base.to_string())
                })
                .collect(),
        };
        let mut logical = BTreeMap::new();
        for name in names {
            if let Some(base) = name.strip_suffix(".gate_up") {
                logical.insert(format!("{base}.gate_layer"), ());
                logical.insert(format!("{base}.proj"), ());
            } else {
                logical.insert(name, ());
            }
        }
        logical.into_keys().collect()
    }
}

/// A `VarBuilder`-shaped view of a [`Q21WeightSource`]: a dotted prefix plus
/// the accessors a transformer module needs. It is what the transformer's
/// constructors take, so the one `linear()` helper answers in whatever arm the
/// checkpoint encodes.
#[derive(Clone)]
pub(crate) struct Q21Vb<'a> {
    source: &'a Q21WeightSource,
    prefix: String,
}

impl<'a> Q21Vb<'a> {
    pub(crate) fn pp<S: ToString>(&self, segment: S) -> Self {
        let segment = segment.to_string();
        Self {
            source: self.source,
            prefix: if self.prefix.is_empty() {
                segment
            } else {
                format!("{}.{segment}", self.prefix)
            },
        }
    }

    fn name(&self, leaf: &str) -> String {
        if self.prefix.is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{leaf}", self.prefix)
        }
    }

    /// A dense tensor at the working dtype, shape-checked like
    /// `VarBuilder::get`.
    pub(crate) fn get<S: Into<candle_core::Shape>>(&self, shape: S, leaf: &str) -> Result<Tensor> {
        let name = self.name(leaf);
        let tensor = self.source.tensor(&name, self.source.dtype)?;
        let expected = shape.into();
        anyhow::ensure!(
            tensor.shape() == &expected,
            "Qwen Image 2.1 tensor {name} is {:?}, expected {expected:?}",
            tensor.shape()
        );
        Ok(tensor)
    }

    /// The linear at this prefix.
    pub(crate) fn linear(&self, in_features: usize, out_features: usize) -> Result<Q21Linear> {
        self.source.linear(&self.prefix, in_features, out_features)
    }

    /// The MLP input projections at this (`img_mlp`) prefix.
    pub(crate) fn gate_up(&self, dim: usize, hidden: usize) -> Result<Q21GateUp> {
        self.source.gate_up(&self.prefix, dim, hidden)
    }
}

/// Fail a denoise step whose prediction is not finite, naming the tier.
///
/// Qwen-Image's GGUF tiers have returned 100% NaN through candle's MMQ kernels
/// (`docs/architecture/qwen-mmq-nan.md`), which renders as a solid-black image
/// rather than an error. `x - x` is `0` for every finite element and `NaN` for
/// an infinite or NaN one, so its sum is exactly zero iff the tensor is finite
/// — one elementwise pass and a reduction, with no overflow on large but finite
/// predictions (a sum of squares could overflow and false-positive).
pub(crate) fn ensure_finite_prediction(
    prediction: &Tensor,
    step: usize,
    tier: &str,
    qmatmul: bool,
) -> Result<()> {
    let probe = prediction
        .broadcast_sub(prediction)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    if probe == 0.0 {
        return Ok(());
    }
    let hint = if qmatmul {
        format!(
            "{QMATMUL_ENV}=1 routed its linears through candle's quantized CUDA kernels; unset it \
             to take the per-forward dequant arm"
        )
    } else {
        "the quantized kernels are not implicated; report the tier and the canvas".to_string()
    };
    bail!(
        "Qwen Image 2.1 {tier} produced a non-finite prediction at denoise step {}; {hint}",
        step + 1
    )
}

#[cfg(test)]
mod tests;
