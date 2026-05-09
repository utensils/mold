//! Block-level GPU offloading for FLUX transformers.
//!
//! Streams transformer blocks one at a time between CPU and GPU during each
//! denoising step. Reduces peak VRAM from ~24GB (full BF16 dev model) to ~2-4GB
//! at the cost of 3-5x slower inference.
//!
//! Self-contained: defines its own block types and forward logic so no patches
//! to candle-transformers are needed.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};

use crate::flux::lora_bypass::{LoraLinear, LoraRegistry};
use crate::flux::pinned::{
    largest_block_size_bytes, pinned_cap_bytes, prefetch_enabled_from_env, try_pin_to_host,
    PinnedMemoryTracker, PinnedRegion,
};
use crate::progress::ProgressReporter;

#[cfg(feature = "cuda")]
use std::sync::Arc;

// Re-export Config and EmbedNd — these are public types with public constructors.
use candle_transformers::models::flux::model::{Config, EmbedNd};

#[cfg(feature = "cuda")]
type PrefetchStream = Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>;
#[cfg(feature = "cuda")]
type PrefetchBuffer = candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>;

// ── Reimplemented candle-internal helpers ────────────────────────────────────

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        anyhow::bail!("{dim} is odd");
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(candle_core::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(candle_core::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // Single dispatch point — FlashAttention / SDPA / math is selected at
    // process start via `MOLD_ATTN` and the `flash-attn` cargo feature.
    Ok(crate::attention::attention_default_scale(q, k, v)?)
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    Ok((fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())?)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    Ok(x.transpose(1, 2)?.flatten_from(2)?)
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

// ── Device-transfer helpers ──────────────────────────────────────────────────

fn linear_to_device(l: &Linear, dev: &Device) -> Result<Linear> {
    let w = l.weight().to_device(dev)?;
    let b = l.bias().map(|b| b.to_device(dev)).transpose()?;
    Ok(Linear::new(w, b))
}

/// Move a `Linear`'s weights to `dev`, then look up bypass-mode LoRA
/// adapters for `key` in `registry` and attach them. The adapters
/// already live on the runtime device (see `LoraRegistry::build`),
/// so attaching is a clone of small `Tensor` handles — no copy.
fn lora_linear_to_device(
    l: &Linear,
    dev: &Device,
    registry: Option<&LoraRegistry>,
    key: &str,
) -> Result<LoraLinear> {
    let inner = linear_to_device(l, dev)?;
    let adapters = registry
        .map(|r| r.adapters_for(key).to_vec())
        .unwrap_or_default();
    if adapters.is_empty() {
        Ok(LoraLinear::Plain(inner))
    } else {
        Ok(LoraLinear::WithAdapters { inner, adapters })
    }
}

fn layer_norm_to_device(ln: &LayerNorm, dev: &Device) -> Result<LayerNorm> {
    let w = ln.weight().to_device(dev)?;
    match ln.bias() {
        Some(b) => Ok(LayerNorm::new(w, b.to_device(dev)?, 1e-6)),
        None => Ok(LayerNorm::new_no_bias(w, 1e-6)),
    }
}

fn rms_norm_to_device(rn: &RmsNorm, dev: &Device) -> Result<RmsNorm> {
    let inner = rn.clone().into_inner();
    Ok(RmsNorm::new(inner.weight().to_device(dev)?, 1e-6))
}

// ── Block-weight visitors (for pinning + size sums) ──────────────────────────
//
// `try_pin_visit_*` walk every CPU-resident base weight in a block. We use
// closures rather than returning a Vec<&Tensor> so the same traversal logic
// drives byte-counting and pinning without allocating a temporary index.

/// Visit every `Linear`, `LayerNorm`, and `RmsNorm` base weight in a
/// `DoubleBlock`. Returns the running total of `n_bytes` across all visits.
fn visit_double_block_weights<F>(b: &DoubleBlock, mut f: F) -> usize
where
    F: FnMut(&Tensor) -> usize,
{
    let mut total = 0usize;
    total += f(b.img_mod.lin.weight());
    if let Some(t) = b.img_mod.lin.bias() {
        total += f(t);
    }
    total += f(b.img_norm1.weight());
    if let Some(t) = b.img_norm1.bias() {
        total += f(t);
    }
    total += f(b.img_attn.qkv.weight());
    if let Some(t) = b.img_attn.qkv.bias() {
        total += f(t);
    }
    // RmsNorm exposes `clone().into_inner().weight()` which is awkward; we
    // pin the wrapped weight directly via the same accessor used in
    // `rms_norm_to_device`.
    total += f(b.img_attn.query_norm.clone().into_inner().weight());
    total += f(b.img_attn.key_norm.clone().into_inner().weight());
    total += f(b.img_attn.proj.weight());
    if let Some(t) = b.img_attn.proj.bias() {
        total += f(t);
    }
    total += f(b.img_norm2.weight());
    if let Some(t) = b.img_norm2.bias() {
        total += f(t);
    }
    total += f(b.img_mlp.lin1.weight());
    if let Some(t) = b.img_mlp.lin1.bias() {
        total += f(t);
    }
    total += f(b.img_mlp.lin2.weight());
    if let Some(t) = b.img_mlp.lin2.bias() {
        total += f(t);
    }
    total += f(b.txt_mod.lin.weight());
    if let Some(t) = b.txt_mod.lin.bias() {
        total += f(t);
    }
    total += f(b.txt_norm1.weight());
    if let Some(t) = b.txt_norm1.bias() {
        total += f(t);
    }
    total += f(b.txt_attn.qkv.weight());
    if let Some(t) = b.txt_attn.qkv.bias() {
        total += f(t);
    }
    total += f(b.txt_attn.query_norm.clone().into_inner().weight());
    total += f(b.txt_attn.key_norm.clone().into_inner().weight());
    total += f(b.txt_attn.proj.weight());
    if let Some(t) = b.txt_attn.proj.bias() {
        total += f(t);
    }
    total += f(b.txt_norm2.weight());
    if let Some(t) = b.txt_norm2.bias() {
        total += f(t);
    }
    total += f(b.txt_mlp.lin1.weight());
    if let Some(t) = b.txt_mlp.lin1.bias() {
        total += f(t);
    }
    total += f(b.txt_mlp.lin2.weight());
    if let Some(t) = b.txt_mlp.lin2.bias() {
        total += f(t);
    }
    total
}

/// Visit every base weight in a `SingleBlock`.
fn visit_single_block_weights<F>(b: &SingleBlock, mut f: F) -> usize
where
    F: FnMut(&Tensor) -> usize,
{
    let mut total = 0usize;
    total += f(b.linear1.weight());
    if let Some(t) = b.linear1.bias() {
        total += f(t);
    }
    total += f(b.linear2.weight());
    if let Some(t) = b.linear2.bias() {
        total += f(t);
    }
    total += f(b.query_norm.clone().into_inner().weight());
    total += f(b.key_norm.clone().into_inner().weight());
    total += f(b.pre_norm.weight());
    if let Some(t) = b.pre_norm.bias() {
        total += f(t);
    }
    total += f(b.modulation.lin.weight());
    if let Some(t) = b.modulation.lin.bias() {
        total += f(t);
    }
    total
}

/// Bytes consumed by a single tensor — `elem_count() * dtype.size_in_bytes()`.
fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

// ── Prefetch stream + buffer init ────────────────────────────────────────────

/// Bring up a non-default CUDA stream + a reusable byte buffer sized to the
/// largest block. Allocating up-front means subsequent block prefetches
/// never invoke `cudaMalloc`, which otherwise dominates short-step inference.
///
/// On non-CUDA builds this entire function is compiled out (the caller is
/// already gated by `#[cfg(feature = "cuda")]`).
#[cfg(feature = "cuda")]
fn init_prefetch(
    gpu_device: &Device,
    largest_block_bytes: usize,
) -> Result<(Option<PrefetchStream>, Option<PrefetchBuffer>)> {
    let cuda_dev = match gpu_device.as_cuda_device() {
        Ok(d) => d,
        Err(_) => return Ok((None, None)),
    };
    let ctx = cuda_dev.cuda_stream().context().clone();
    let stream = match ctx.new_stream() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("FLUX offload: failed to create prefetch stream ({e:?}) — falling back to single-stream");
            return Ok((None, None));
        }
    };
    if largest_block_bytes == 0 {
        return Ok((Some(stream), None));
    }
    // Allocate the destination buffer on the prefetch stream so freeing
    // happens on the same stream and we don't surprise candle's
    // event-tracker.
    let buf = match unsafe { stream.alloc::<u8>(largest_block_bytes) } {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                "FLUX offload: prefetch buffer alloc failed ({largest_block_bytes} bytes, {e:?}) — \
                 falling back to single-stream"
            );
            return Ok((Some(stream), None));
        }
    };
    Ok((Some(stream), Some(buf)))
}

// ── Self-contained block types ───────────────────────────────────────────────

struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin: candle_nn::linear(dim, 3 * dim, vb.pp("lin"))?,
        })
    }
    /// Move to `dev` and attach bypass-mode LoRA adapters keyed at
    /// `<base>.lin.weight` (e.g. `single_blocks.7.modulation.lin.weight`).
    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<GpuModulation1> {
        Ok(GpuModulation1 {
            lin: lora_linear_to_device(
                &self.lin,
                dev,
                registry,
                &format!("{base_key}.lin.weight"),
            )?,
        })
    }
}

struct GpuModulation1 {
    lin: LoraLinear,
}

impl GpuModulation1 {
    fn forward(&self, vec_: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let pre = vec_.silu()?;
        let ys = self.lin.forward(&pre)?.unsqueeze(1)?.chunk(3, D::Minus1)?;
        Ok((ys[0].clone(), ys[1].clone(), ys[2].clone()))
    }
}

struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin: candle_nn::linear(dim, 6 * dim, vb.pp("lin"))?,
        })
    }
    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<GpuModulation2> {
        Ok(GpuModulation2 {
            lin: lora_linear_to_device(
                &self.lin,
                dev,
                registry,
                &format!("{base_key}.lin.weight"),
            )?,
        })
    }
}

struct GpuModulation2 {
    lin: LoraLinear,
}

impl GpuModulation2 {
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        vec_: &Tensor,
    ) -> Result<((Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))> {
        let pre = vec_.silu()?;
        let ys = self.lin.forward(&pre)?.unsqueeze(1)?.chunk(6, D::Minus1)?;
        Ok((
            (ys[0].clone(), ys[1].clone(), ys[2].clone()),
            (ys[3].clone(), ys[4].clone(), ys[5].clone()),
        ))
    }
}

struct SelfAttention {
    qkv: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn load(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = candle_nn::linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let query_norm = vb.get(head_dim, "norm.query_norm.scale")?;
        let key_norm = vb.get(head_dim, "norm.key_norm.scale")?;
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            query_norm: RmsNorm::new(query_norm, 1e-6),
            key_norm: RmsNorm::new(key_norm, 1e-6),
            proj,
            num_heads,
        })
    }
    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<GpuSelfAttention> {
        Ok(GpuSelfAttention {
            qkv: lora_linear_to_device(
                &self.qkv,
                dev,
                registry,
                &format!("{base_key}.qkv.weight"),
            )?,
            query_norm: rms_norm_to_device(&self.query_norm, dev)?,
            key_norm: rms_norm_to_device(&self.key_norm, dev)?,
            proj: lora_linear_to_device(
                &self.proj,
                dev,
                registry,
                &format!("{base_key}.proj.weight"),
            )?,
            num_heads: self.num_heads,
        })
    }
}

struct GpuSelfAttention {
    qkv: LoraLinear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    proj: LoraLinear,
    num_heads: usize,
}

impl GpuSelfAttention {
    fn qkv_split(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv.forward(xs)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.query_norm)?;
        let k = k.apply(&self.key_norm)?;
        Ok((q, k, v))
    }
}

struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn load(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: candle_nn::linear(in_sz, mlp_sz, vb.pp("0"))?,
            lin2: candle_nn::linear(mlp_sz, in_sz, vb.pp("2"))?,
        })
    }
    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<GpuMlp> {
        Ok(GpuMlp {
            lin1: lora_linear_to_device(
                &self.lin1,
                dev,
                registry,
                &format!("{base_key}.0.weight"),
            )?,
            lin2: lora_linear_to_device(
                &self.lin2,
                dev,
                registry,
                &format!("{base_key}.2.weight"),
            )?,
        })
    }
}

struct GpuMlp {
    lin1: LoraLinear,
    lin2: LoraLinear,
}

impl GpuMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.lin1.forward(xs)?.gelu()?;
        self.lin2.forward(&h)
    }
}

/// FLUX double-stream block — processes image and text streams in parallel
/// with cross-attention.
pub(crate) struct DoubleBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        Ok(Self {
            img_mod: Modulation2::load(h, vb.pp("img_mod"))?,
            img_norm1: layer_norm(h, vb.pp("img_norm1"))?,
            img_attn: SelfAttention::load(h, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?,
            img_norm2: layer_norm(h, vb.pp("img_norm2"))?,
            img_mlp: Mlp::load(h, mlp_sz, vb.pp("img_mlp"))?,
            txt_mod: Modulation2::load(h, vb.pp("txt_mod"))?,
            txt_norm1: layer_norm(h, vb.pp("txt_norm1"))?,
            txt_attn: SelfAttention::load(h, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?,
            txt_norm2: layer_norm(h, vb.pp("txt_norm2"))?,
            txt_mlp: Mlp::load(h, mlp_sz, vb.pp("txt_mlp"))?,
        })
    }

    /// Stream this block onto `dev` and bind any bypass-mode LoRA
    /// adapters keyed under `double_blocks.{idx}.…`.
    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        idx: usize,
    ) -> Result<GpuDoubleBlock> {
        let base = format!("double_blocks.{idx}");
        Ok(GpuDoubleBlock {
            img_mod: self
                .img_mod
                .to_device(dev, registry, &format!("{base}.img_mod"))?,
            img_norm1: layer_norm_to_device(&self.img_norm1, dev)?,
            img_attn: self
                .img_attn
                .to_device(dev, registry, &format!("{base}.img_attn"))?,
            img_norm2: layer_norm_to_device(&self.img_norm2, dev)?,
            img_mlp: self
                .img_mlp
                .to_device(dev, registry, &format!("{base}.img_mlp"))?,
            txt_mod: self
                .txt_mod
                .to_device(dev, registry, &format!("{base}.txt_mod"))?,
            txt_norm1: layer_norm_to_device(&self.txt_norm1, dev)?,
            txt_attn: self
                .txt_attn
                .to_device(dev, registry, &format!("{base}.txt_attn"))?,
            txt_norm2: layer_norm_to_device(&self.txt_norm2, dev)?,
            txt_mlp: self
                .txt_mlp
                .to_device(dev, registry, &format!("{base}.txt_mlp"))?,
        })
    }
}

/// GPU-resident, LoRA-aware double-stream block. Built fresh each step
/// from a CPU [`DoubleBlock`] via `to_device`; lives only for the
/// duration of one block forward.
struct GpuDoubleBlock {
    img_mod: GpuModulation2,
    img_norm1: LayerNorm,
    img_attn: GpuSelfAttention,
    img_norm2: LayerNorm,
    img_mlp: GpuMlp,
    txt_mod: GpuModulation2,
    txt_norm1: LayerNorm,
    txt_attn: GpuSelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: GpuMlp,
}

impl GpuDoubleBlock {
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ((img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2)) = self.img_mod.forward(vec_)?;
        let ((txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2)) = self.txt_mod.forward(vec_)?;

        // QKV for both streams
        let img_modulated = img
            .apply(&self.img_norm1)?
            .broadcast_mul(&(&img_sc1 + 1.)?)?
            .broadcast_add(&img_s1)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv_split(&img_modulated)?;

        let txt_modulated = txt
            .apply(&self.txt_norm1)?
            .broadcast_mul(&(&txt_sc1 + 1.)?)?
            .broadcast_add(&txt_s1)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv_split(&txt_modulated)?;

        // Cross-attention
        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        // Image residual
        let img = (img + img_g1.broadcast_mul(&self.img_attn.proj.forward(&img_attn_out)?)?)?;
        let img_ff = img
            .apply(&self.img_norm2)?
            .broadcast_mul(&(&img_sc2 + 1.)?)?
            .broadcast_add(&img_s2)?;
        let img = (&img + img_g2.broadcast_mul(&self.img_mlp.forward(&img_ff)?)?)?;

        // Text residual
        let txt = (txt + txt_g1.broadcast_mul(&self.txt_attn.proj.forward(&txt_attn_out)?)?)?;
        let txt_ff = txt
            .apply(&self.txt_norm2)?
            .broadcast_mul(&(&txt_sc2 + 1.)?)?
            .broadcast_add(&txt_s2)?;
        let txt = (&txt + txt_g2.broadcast_mul(&self.txt_mlp.forward(&txt_ff)?)?)?;

        Ok((img, txt))
    }
}

/// FLUX single-stream block — processes combined image+text stream.
pub(crate) struct SingleBlock {
    linear1: Linear,
    linear2: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h / cfg.num_heads;
        Ok(Self {
            linear1: candle_nn::linear(h, h * 3 + mlp_sz, vb.pp("linear1"))?,
            linear2: candle_nn::linear(h + mlp_sz, h, vb.pp("linear2"))?,
            query_norm: {
                let w = vb.get(head_dim, "norm.query_norm.scale")?;
                RmsNorm::new(w, 1e-6)
            },
            key_norm: {
                let w = vb.get(head_dim, "norm.key_norm.scale")?;
                RmsNorm::new(w, 1e-6)
            },
            pre_norm: layer_norm(h, vb.pp("pre_norm"))?,
            modulation: Modulation1::load(h, vb.pp("modulation"))?,
            h_sz: h,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn to_device(
        &self,
        dev: &Device,
        registry: Option<&LoraRegistry>,
        idx: usize,
    ) -> Result<GpuSingleBlock> {
        let base = format!("single_blocks.{idx}");
        Ok(GpuSingleBlock {
            linear1: lora_linear_to_device(
                &self.linear1,
                dev,
                registry,
                &format!("{base}.linear1.weight"),
            )?,
            linear2: lora_linear_to_device(
                &self.linear2,
                dev,
                registry,
                &format!("{base}.linear2.weight"),
            )?,
            query_norm: rms_norm_to_device(&self.query_norm, dev)?,
            key_norm: rms_norm_to_device(&self.key_norm, dev)?,
            pre_norm: layer_norm_to_device(&self.pre_norm, dev)?,
            modulation: self
                .modulation
                .to_device(dev, registry, &format!("{base}.modulation"))?,
            h_sz: self.h_sz,
            mlp_sz: self.mlp_sz,
            num_heads: self.num_heads,
        })
    }
}

/// GPU-resident, LoRA-aware single-stream block.
struct GpuSingleBlock {
    linear1: LoraLinear,
    linear2: LoraLinear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    pre_norm: LayerNorm,
    modulation: GpuModulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl GpuSingleBlock {
    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (shift, scale, gate) = self.modulation.forward(vec_)?;
        let x_mod = xs
            .apply(&self.pre_norm)?
            .broadcast_mul(&(&scale + 1.)?)?
            .broadcast_add(&shift)?;
        let x_mod = self.linear1.forward(&x_mod)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.query_norm)?;
        let k = k.apply(&self.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output_in = Tensor::cat(&[attn, mlp.gelu()?], 2)?;
        let output = self.linear2.forward(&output_in)?;
        Ok((xs + gate.broadcast_mul(&output)?)?)
    }
}

/// Last layer: AdaLN modulation → linear projection.
struct FinalLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl FinalLayer {
    fn load(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm(h_sz, vb.pp("norm_final"))?,
            linear: candle_nn::linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?,
            ada_ln_modulation: candle_nn::linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm_to_device(&self.norm_final, dev)?,
            linear: linear_to_device(&self.linear, dev)?,
            ada_ln_modulation: linear_to_device(&self.ada_ln_modulation, dev)?,
        })
    }
    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        Ok(xs.apply(&self.linear)?)
    }
}

// ── Main offloaded transformer ───────────────────────────────────────────────

/// BF16 FLUX transformer with blocks on CPU, streamed to GPU one at a time.
pub(crate) struct OffloadedFluxTransformer {
    // Stem layers on GPU permanently (~50MB). Stem isn't a typical LoRA
    // target so we leave them as raw `Linear`. If a future FLUX LoRA
    // does target `img_in` / `txt_in`, promote these to `LoraLinear` and
    // extend `map_lora_key` to recognise them.
    img_in: Linear,
    txt_in: Linear,
    time_in: StemMlpEmbedder,
    vector_in: StemMlpEmbedder,
    guidance_in: Option<StemMlpEmbedder>,
    pe_embedder: EmbedNd,
    final_layer: FinalLayer,
    // Blocks on CPU
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    gpu_device: Device,
    /// Bypass-mode LoRA stack. None when no LoRAs are active. Adapters
    /// already live on `gpu_device` so block-stream cycles never have
    /// to copy them — only the base `Linear` weights move.
    lora_registry: Option<LoraRegistry>,
    /// `cuMemHostRegister`'d regions backing every CPU-resident block
    /// weight. Held for the lifetime of the transformer so the underlying
    /// page-locks survive across forward passes. Empty on Metal/CPU and
    /// when pinning was disabled or capped.
    #[allow(dead_code)]
    pinned_regions: Vec<PinnedRegion>,
    /// Side stream for async H2D prefetching of block N+1 while block N
    /// computes on the device's primary stream. None on Metal/CPU and
    /// when `MOLD_OFFLOAD_PREFETCH=off`.
    ///
    /// **Implementation note:** candle-core-mold's `Tensor::to_device`
    /// dispatches H2D through the device's *primary* stream — there is
    /// no public API to redirect a single tensor transfer onto a
    /// different stream. Hooking the actual block-prefetch onto this
    /// side stream therefore requires either a `Tensor::from_storage` +
    /// manual `cuMemcpyHtoDAsync_v2` path (one branch per dtype) or a
    /// candle-core-mold patch exposing a stream override on `to_device`.
    /// The stream itself is created and held here so the follow-up that
    /// wires the manual transfer path can simply consume it; pinning
    /// alone (Phase 1) already captures the bulk of the H2D speedup.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    prefetch_stream: Option<PrefetchStream>,
    /// Reusable destination buffer sized to the largest block on this
    /// model. Allocated once on the prefetch stream so block-level H2D
    /// never `cudaMalloc`s. None when prefetching is disabled.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    prefetch_buffer: Option<PrefetchBuffer>,
}

impl OffloadedFluxTransformer {
    /// Load the full FLUX transformer from safetensors on CPU, then move stem to GPU.
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        gpu_device: &Device,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        progress.info("Loading transformer blocks on CPU…");

        // Load stem on CPU, then move to GPU.
        // We use our own StemMlpEmbedder (2 Linears + SiLU) since candle's
        // MlpEmbedder has private fields and no to_device method.
        let img_in = linear_to_device(
            &candle_nn::linear(cfg.in_channels, cfg.hidden_size, vb.pp("img_in"))?,
            gpu_device,
        )?;
        let txt_in = linear_to_device(
            &candle_nn::linear(cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"))?,
            gpu_device,
        )?;
        let time_in =
            StemMlpEmbedder::load(256, cfg.hidden_size, vb.pp("time_in"))?.to_device(gpu_device)?;
        let vector_in = StemMlpEmbedder::load(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"))?
            .to_device(gpu_device)?;
        let guidance_in = if cfg.guidance_embed {
            Some(
                StemMlpEmbedder::load(256, cfg.hidden_size, vb.pp("guidance_in"))?
                    .to_device(gpu_device)?,
            )
        } else {
            None
        };

        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());

        let final_layer =
            FinalLayer::load(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?
                .to_device(gpu_device)?;

        // Load blocks on CPU
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            double_blocks.push(DoubleBlock::load(cfg, vb_d.pp(idx))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleBlock::load(cfg, vb_s.pp(idx))?);
        }

        progress.info(&format!(
            "Offloading: {} double + {} single blocks on CPU, stem on GPU",
            double_blocks.len(),
            single_blocks.len(),
        ));

        // Per-block byte sizes — feeds both the prefetch-buffer sizing and
        // the pinning summary log.
        let mut block_sizes: Vec<usize> =
            Vec::with_capacity(double_blocks.len() + single_blocks.len());
        for b in &double_blocks {
            block_sizes.push(visit_double_block_weights(b, tensor_bytes));
        }
        for b in &single_blocks {
            block_sizes.push(visit_single_block_weights(b, tensor_bytes));
        }

        // ── Phase 1: pin every CPU-resident block weight ────────────────
        let tracker = PinnedMemoryTracker::new(pinned_cap_bytes());
        let mut pinned_regions: Vec<PinnedRegion> = Vec::new();
        let mut pin_visit = |t: &Tensor| -> usize {
            match try_pin_to_host(t, &tracker) {
                Ok(Some(region)) => {
                    pinned_regions.push(region);
                    0
                }
                Ok(None) => 0,
                Err(e) => {
                    tracing::debug!("try_pin_to_host failed: {e:?} (continuing)");
                    0
                }
            }
        };
        for b in &double_blocks {
            visit_double_block_weights(b, &mut pin_visit);
        }
        for b in &single_blocks {
            visit_single_block_weights(b, &mut pin_visit);
        }

        // ── Phase 2: optionally bring up a prefetch stream + buffer ────
        let prefetch_on = prefetch_enabled_from_env() && gpu_device.is_cuda();
        let largest_block = largest_block_size_bytes(&block_sizes);

        #[cfg(feature = "cuda")]
        let (prefetch_stream, prefetch_buffer) = if prefetch_on {
            init_prefetch(gpu_device, largest_block)?
        } else {
            (None, None)
        };

        // Single-line INFO so users can confirm both levers are running.
        let pinned_gb = tracker.used_bytes() as f64 / 1_000_000_000.0;
        if pinned_regions.is_empty() {
            progress.info(&format!(
                "FLUX offload: prefetch={} (largest block {:.1} MB) — pinning skipped \
                 (no CUDA / unsupported tensors)",
                if prefetch_on { "on" } else { "off" },
                largest_block as f64 / 1_000_000.0,
            ));
        } else {
            progress.info(&format!(
                "FLUX offload: pinned {:.2} GB across {} tensors, prefetch={} \
                 (largest block {:.1} MB)",
                pinned_gb,
                pinned_regions.len(),
                if prefetch_on { "on" } else { "off" },
                largest_block as f64 / 1_000_000.0,
            ));
        }

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            final_layer,
            double_blocks,
            single_blocks,
            gpu_device: gpu_device.clone(),
            lora_registry: None,
            pinned_regions,
            #[cfg(feature = "cuda")]
            prefetch_stream,
            #[cfg(feature = "cuda")]
            prefetch_buffer,
        })
    }

    /// Install a bypass-mode LoRA stack. Adapters fire each block-step
    /// on top of the base matmul output — no base-weight rebuild, no
    /// CPU-side dequant→merge→requant. Pass `None` to clear.
    ///
    /// Cheap: just stashes the registry handle. The actual binding to
    /// each `LoraLinear` happens lazily on the next `forward()` when
    /// blocks stream onto the GPU.
    pub(crate) fn set_lora_registry(&mut self, registry: Option<LoraRegistry>) {
        self.lora_registry = registry;
    }

    /// True when at least one bypass-mode adapter is installed.
    #[allow(dead_code)]
    pub(crate) fn has_loras(&self) -> bool {
        self.lora_registry
            .as_ref()
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    }

    /// Run the full FLUX forward pass with block-level streaming.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = img.dtype();
        let registry = self.lora_registry.as_ref();

        // Positional encoding
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };

        // Stem projections (on GPU)
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;

        // Timestep + guidance + vector embedding
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double blocks: stream each from CPU → GPU. LoRA adapters are
        // already GPU-resident (in `lora_registry`); streaming a block
        // copies only the base Linear weights.
        for (i, block) in self.double_blocks.iter().enumerate() {
            let gpu_block = block.to_device(&self.gpu_device, registry, i)?;
            (img, txt) = gpu_block.forward(&img, &txt, &vec_, &pe)?;
            self.gpu_device.synchronize()?;
            drop(gpu_block);
            tracing::trace!("double block {i} done");
        }

        // Single blocks: stream each from CPU → GPU
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.dim(1)?;
        for (i, block) in self.single_blocks.iter().enumerate() {
            let gpu_block = block.to_device(&self.gpu_device, registry, i)?;
            img = gpu_block.forward(&img, &vec_, &pe)?;
            self.gpu_device.synchronize()?;
            drop(gpu_block);
            tracing::trace!("single block {i} done");
        }

        // Final layer (on GPU)
        let img = img.i((.., txt_len..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

/// Simple MLP embedder (2 Linears + SiLU) for stem layers.
/// Reimplemented here because candle's `MlpEmbedder` has private fields.
struct StemMlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl StemMlpEmbedder {
    fn load(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_layer: candle_nn::linear(in_sz, h_sz, vb.pp("in_layer"))?,
            out_layer: candle_nn::linear(h_sz, h_sz, vb.pp("out_layer"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            in_layer: linear_to_device(&self.in_layer, dev)?,
            out_layer: linear_to_device(&self.out_layer, dev)?,
        })
    }
}

impl Module for StemMlpEmbedder {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}
