//! Hunyuan3D 2.0 "vecset" shape VAE — **decode only**.
//!
//! This VAE does not decode to pixels. It decodes to a *function*: given the
//! 1-D latent token sequence the DiT denoised (`[B, embed_dim, num_latents]`,
//! i.e. 64 x 3072 for every 2.0 checkpoint), it answers "is this 3-D point
//! inside the surface?" for an arbitrary batch of query points. The caller
//! evaluates that function on a dense `(res + 1)^3` grid to obtain an occupancy
//! field, which [`super::mesh`] then turns into triangles.
//!
//! # Why decode only
//!
//! The encoder (`PointCrossAttention`, farthest-point sampling, sharp-edge
//! sampling) exists to turn a *mesh* into latents. Image-to-3D never has a mesh
//! to encode — the latents come out of the flow-matching DiT — so the encoder
//! is dead weight and is deliberately not ported. `pre_kl` and the diagonal
//! Gaussian are part of that same encode path and are likewise absent.
//!
//! # Shape of the computation
//!
//! ```text
//! latents [B, 64, 3072] --movedim(-2,-1)--> [B, 3072, 64]
//!     -> post_kl (Linear 64->1024)
//!     -> Transformer (16 self-attention resblocks, width 1024, 16 heads)   <-- ONCE
//!     =  prepared [B, 3072, 1024]
//!
//! queries [B, N, 3]
//!     -> FourierEmbedder (51 dims) -> query_proj (Linear 51->1024)
//!     -> ONE ResidualCrossAttentionBlock against `prepared`               <-- PER CHUNK
//!     -> ln_post -> output_proj (Linear 1024->1)  =  logits [B, N]
//! ```
//!
//! The split between [`ShapeVae::prepare_latents`] and
//! [`ShapeVae::decode_queries`] is load-bearing: the 16-layer transformer is
//! independent of the query points, so it runs once per generation while only
//! the single cross-attention block runs per chunk. Fusing them would re-run
//! the transformer thousands of times.
//!
//! # Memory
//!
//! At `octree_resolution = 256` the grid is `257^3 = 16_974_593` points. Held
//! all at once that is **~204 MB** of F32 query coordinates plus **~68 MB** of
//! F32 logits, before any activation. Every intermediate inside the cross
//! attention block is `N x 1024` (~69 GB at full resolution), so materialising
//! the whole grid in one [`ShapeVae::decode_queries`] call is not an option.
//! Callers walk the grid in chunks (upstream's default is 8000-10000 points)
//! via [`query_grid_chunk`], which never allocates the full coordinate buffer;
//! [`query_grid`] is provided for tests and small resolutions and documents the
//! cost above. Hoist [`ShapeVae::prepare_cross_kv`] out of the chunk loop and
//! call [`ShapeVae::decode_queries_cached`] to avoid re-projecting the 3072
//! latent tokens through `c_kv` on every chunk (upstream's `kv_cache` flag,
//! `vae.py:667-676`).
//!
//! # Upstream references
//!
//! Ported from ComfyUI `comfy/ldm/hunyuan3d/vae.py` (the executable oracle):
//! `ShapeVAE` (899), `ShapeVAE.decode` (966), `VanillaVolumeDecoder` (427),
//! `FourierEmbedder` (459), `CrossAttentionDecoder` (846), `Transformer` (811),
//! `ResidualAttentionBlock` (781), `MultiheadAttention` (751),
//! `QKVMultiheadAttention` (723), `ResidualCrossAttentionBlock` (687),
//! `MultiheadCrossAttention` (646), `QKVMultiheadCrossAttention` (610),
//! `MLP` (592). Config values are cross-checked against the `config.yaml`
//! shipped beside each Tencent checkpoint.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_no_bias, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder,
};

use crate::attention::attention;

/// LayerNorm epsilon used by every `norm_layer(...)` inside the blocks.
///
/// `vae.py` passes `eps=1e-6` explicitly at every block-level call site
/// (`vae.py:711-713`, `vae.py:797-799`, `vae.py:622-623`).
const BLOCK_LN_EPS: f64 = 1e-6;

/// `CrossAttentionDecoder.ln_post` is a bare `ops.LayerNorm(width)`
/// (`vae.py:876`) and therefore keeps torch's default epsilon.
const LN_POST_EPS: f64 = 1e-5;

/// Geometry + hyper-parameters of a Hunyuan3D 2.0 shape VAE.
///
/// Only the decode-relevant fields are modelled; the encoder knobs
/// (`pc_size`, `point_feats`, `downsample_ratio`, `num_encoder_layers`, ...)
/// are intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeVaeConfig {
    /// Number of latent tokens the DiT produces (3072 for every 2.0 variant).
    /// Informational: nothing in the decode path is sized by it.
    pub num_latents: usize,
    /// Latent channel count, the input width of `post_kl`.
    pub embed_dim: usize,
    /// Transformer / decoder hidden width.
    pub width: usize,
    /// Attention heads (shared by the self- and cross-attention stacks).
    pub heads: usize,
    /// Self-attention resblock count.
    pub num_decoder_layers: usize,
    /// Fourier frequency count. `2^0 .. 2^(num_freqs-1)`.
    pub num_freqs: usize,
    /// Multiply the frequencies by pi. `false` for every 2.0 checkpoint.
    pub include_pi: bool,
    /// Bias on `c_qkv` / `c_q` / `c_kv`. `false` for every 2.0 checkpoint.
    pub qkv_bias: bool,
    /// Per-head LayerNorm on q and k. `true` for every 2.0 checkpoint.
    pub qk_norm: bool,
    /// MLP expansion inside the self-attention resblocks (`vae.py:801`).
    pub mlp_expand_ratio: usize,
    /// MLP expansion inside the geo decoder's cross-attention block.
    pub geo_decoder_mlp_expand_ratio: usize,
    /// Apply `ln_post` before `output_proj`. `true` for every 2.0 checkpoint.
    pub geo_decoder_ln_post: bool,
    /// Occupancy channels out of `output_proj`. Always 1 (`label_type` is
    /// `"binary"`, so the single channel is a logit).
    pub out_channels: usize,
    /// Latent scaling from the checkpoint's `config.yaml`. See
    /// [`ShapeVae::unscale_latents`].
    pub scale_factor: f64,
}

impl Default for ShapeVaeConfig {
    fn default() -> Self {
        Self::v2_0()
    }
}

impl ShapeVaeConfig {
    /// Published 2.1 shape config, and ComfyUI's Hunyuan3Dv2_1 latent format.
    pub fn v2_1() -> Self {
        Self {
            num_latents: 4096,
            scale_factor: 1.003_950_615_875_240_3,
            ..Self::v2_0()
        }
    }

    /// `hunyuan3d-dit-v2-0` and `hunyuan3d-dit-v2-0-turbo`.
    pub fn v2_0() -> Self {
        Self {
            num_latents: 3072,
            embed_dim: 64,
            width: 1024,
            heads: 16,
            num_decoder_layers: 16,
            num_freqs: 8,
            include_pi: false,
            qkv_bias: false,
            qk_norm: true,
            // `geo_decoder_downsample_ratio` is 1 for every 2.0 checkpoint, so
            // the `latents_proj` branch of `CrossAttentionDecoder.__init__`
            // (`vae.py:869-870`) never materialises and `width`/`heads` reach
            // the geo decoder undivided.
            mlp_expand_ratio: 4,
            geo_decoder_mlp_expand_ratio: 4,
            geo_decoder_ln_post: true,
            out_channels: 1,
            scale_factor: 0.999_094_304_262_252_9,
        }
    }

    /// `hunyuan3d-dit-v2-0-mini-turbo`. Identical geometry; the only difference
    /// is the latent scale factor.
    pub fn v2_0_mini() -> Self {
        Self {
            scale_factor: 1.018_813_714_239_540_4,
            ..Self::v2_0()
        }
    }

    /// Per-head dimension. The q/k LayerNorms are sized by this.
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

fn block_ln(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    layer_norm(
        size,
        LayerNormConfig {
            eps: BLOCK_LN_EPS,
            remove_mean: true,
            affine: true,
        },
        vb,
    )
}

/// Sin/cos positional embedding for the query points.
///
/// Ported from `FourierEmbedder` (`vae.py:459-553`). With `logspace = true`
/// (the only mode any 2.0 checkpoint uses) the frequencies are
/// `2^0 .. 2^(num_freqs - 1)`, optionally scaled by pi.
///
/// The output layout is `cat([x, sin(embed), cos(embed)], -1)` where `embed` is
/// `[..., input_dim * num_freqs]` laid out **input-dim major** — that is,
/// `x0*f0, x0*f1, ..., x0*f7, x1*f0, ...` — because upstream broadcasts
/// `x[..., None] * frequencies` and flattens the trailing two axes
/// (`vae.py:546-547`). Getting that order wrong silently rotates the learned
/// `query_proj` and produces a plausible-looking but wrong surface.
#[derive(Debug, Clone)]
pub struct FourierEmbedder {
    frequencies: Vec<f32>,
    input_dim: usize,
    include_input: bool,
    out_dim: usize,
}

impl FourierEmbedder {
    /// `logspace = true` is implied; no 2.0 checkpoint uses the linear mode.
    pub fn new(num_freqs: usize, input_dim: usize, include_input: bool, include_pi: bool) -> Self {
        let pi = std::f64::consts::PI;
        let frequencies: Vec<f32> = (0..num_freqs)
            .map(|i| {
                let f = 2f64.powi(i as i32);
                (if include_pi { f * pi } else { f }) as f32
            })
            .collect();
        // `FourierEmbedder.get_dims` (`vae.py:527-531`): the raw input rides
        // along when `include_input` is set *or* when there are no frequencies
        // at all.
        let temp = usize::from(include_input || num_freqs == 0);
        let out_dim = input_dim * (num_freqs * 2 + temp);
        Self {
            frequencies,
            input_dim,
            include_input,
            out_dim,
        }
    }

    /// Embedding width. 51 for the shipped `num_freqs = 8, input_dim = 3,
    /// include_input = true` configuration — which is exactly the input width
    /// of `geo_decoder.query_proj`.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Number of frequency bands.
    pub fn num_freqs(&self) -> usize {
        self.frequencies.len()
    }

    /// `[..., input_dim] -> [..., out_dim]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if self.frequencies.is_empty() {
            return Ok(xs.clone());
        }
        let last = xs.dim(D::Minus1)?;
        if last != self.input_dim {
            candle_core::bail!(
                "FourierEmbedder expects a trailing dim of {}, got {last}",
                self.input_dim
            );
        }
        let freqs = Tensor::from_slice(
            self.frequencies.as_slice(),
            self.frequencies.len(),
            xs.device(),
        )?
        .to_dtype(xs.dtype())?;

        // `(x[..., None] * frequencies).view(*x.shape[:-1], -1)` — vae.py:546.
        let scaled = xs.unsqueeze(D::Minus1)?.broadcast_mul(&freqs)?;
        let mut dims = xs.dims().to_vec();
        let trailing = dims.pop().unwrap_or(0) * self.frequencies.len();
        dims.push(trailing);
        let embed = scaled.reshape(dims)?;

        let sin = embed.sin()?;
        let cos = embed.cos()?;
        if self.include_input {
            Tensor::cat(&[xs, &sin, &cos], D::Minus1)
        } else {
            Tensor::cat(&[&sin, &cos], D::Minus1)
        }
    }
}

/// `MLP` (`vae.py:592-608`): `c_proj(gelu(c_fc(x)))`.
///
/// `nn.GELU()` with no arguments is the exact erf formulation, not the tanh
/// approximation, so this uses `gelu_erf`.
#[derive(Debug)]
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(width: usize, expand_ratio: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: linear(width, width * expand_ratio, vb.pp("c_fc"))?,
            c_proj: linear(width * expand_ratio, width, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu_erf()?)
    }
}

/// Optional per-head q/k LayerNorm shared by both attention flavours.
///
/// `QKVMultiheadAttention` / `QKVMultiheadCrossAttention` build these as
/// `norm_layer(width // heads, elementwise_affine=True, eps=1e-6)`
/// (`vae.py:622-623`, `vae.py:735-736`) — LayerNorm with **weight and bias**,
/// not RmsNorm. They normalise the trailing per-head axis of a
/// `[B, N, H, head_dim]` tensor, i.e. *before* the head permute.
#[derive(Debug)]
struct QkNorm {
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
}

impl QkNorm {
    fn new(head_dim: usize, enabled: bool, vb: VarBuilder) -> Result<Self> {
        if !enabled {
            return Ok(Self {
                q_norm: None,
                k_norm: None,
            });
        }
        Ok(Self {
            q_norm: Some(block_ln(head_dim, vb.pp("q_norm"))?),
            k_norm: Some(block_ln(head_dim, vb.pp("k_norm"))?),
        })
    }

    fn apply_q(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.q_norm {
            Some(n) => n.forward(xs),
            None => Ok(xs.clone()),
        }
    }

    fn apply_k(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.k_norm {
            Some(n) => n.forward(xs),
            None => Ok(xs.clone()),
        }
    }
}

/// `[B, N, H, head_dim] -> [B, H, N, head_dim]` (upstream's
/// `t.permute(0, 2, 1, 3)`).
fn to_bhnd(xs: &Tensor) -> Result<Tensor> {
    xs.transpose(1, 2)?.contiguous()
}

/// `[B, H, N, head_dim] -> [B, N, H * head_dim]` (upstream's
/// `out.transpose(1, 2).reshape(bs, n_ctx, -1)`).
fn from_bhnd(xs: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = xs.dims4()?;
    xs.transpose(1, 2)?.contiguous()?.reshape((b, n, h * d))
}

/// `MultiheadAttention` + `QKVMultiheadAttention` (`vae.py:723-778`).
#[derive(Debug)]
struct SelfAttention {
    c_qkv: Linear,
    c_proj: Linear,
    norms: QkNorm,
    heads: usize,
}

impl SelfAttention {
    fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        let width = cfg.width;
        let c_qkv = if cfg.qkv_bias {
            linear(width, width * 3, vb.pp("c_qkv"))?
        } else {
            linear_no_bias(width, width * 3, vb.pp("c_qkv"))?
        };
        Ok(Self {
            c_qkv,
            // `c_proj` is a plain `ops.Linear(width, width)` and always keeps
            // its bias, regardless of `qkv_bias` (vae.py:762).
            c_proj: linear(width, width, vb.pp("c_proj"))?,
            norms: QkNorm::new(cfg.head_dim(), cfg.qk_norm, vb.pp("attention"))?,
            heads: cfg.heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let qkv = self.c_qkv.forward(xs)?;
        let width = qkv.dim(D::Minus1)?;
        // `attn_ch = width // heads // 3` then `view(bs, n_ctx, heads, -1)` and
        // a 3-way split of the trailing axis (vae.py:740-744).
        let head_dim = width / self.heads / 3;
        let qkv = qkv.reshape((b, n, self.heads, 3 * head_dim))?;
        let q = self
            .norms
            .apply_q(&qkv.narrow(3, 0, head_dim)?.contiguous()?)?;
        let k = self
            .norms
            .apply_k(&qkv.narrow(3, head_dim, head_dim)?.contiguous()?)?;
        let v = qkv.narrow(3, 2 * head_dim, head_dim)?.contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = attention(&to_bhnd(&q)?, &to_bhnd(&k)?, &to_bhnd(&v)?, scale as f32)?;
        self.c_proj.forward(&from_bhnd(&out)?)
    }
}

/// `ResidualAttentionBlock` (`vae.py:781-808`).
#[derive(Debug)]
struct ResidualAttentionBlock {
    ln_1: LayerNorm,
    attn: SelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl ResidualAttentionBlock {
    fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: block_ln(cfg.width, vb.pp("ln_1"))?,
            attn: SelfAttention::new(cfg, vb.pp("attn"))?,
            ln_2: block_ln(cfg.width, vb.pp("ln_2"))?,
            mlp: Mlp::new(cfg.width, cfg.mlp_expand_ratio, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.ln_1.forward(xs)?)?)?;
        &xs + self.mlp.forward(&self.ln_2.forward(&xs)?)?
    }
}

/// `MultiheadCrossAttention` + `QKVMultiheadCrossAttention`
/// (`vae.py:610-685`).
///
/// `c_kv` projects the *data* (the prepared latents) to `2 * width`; the
/// per-head split is `attn_ch = width // heads // 2` where `width` there is the
/// **kv** width, so `k` and `v` each land on `cfg.head_dim()` (vae.py:629-635).
#[derive(Debug)]
struct CrossAttention {
    c_q: Linear,
    c_kv: Linear,
    c_proj: Linear,
    norms: QkNorm,
    heads: usize,
}

impl CrossAttention {
    fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        let width = cfg.width;
        let (c_q, c_kv) = if cfg.qkv_bias {
            (
                linear(width, width, vb.pp("c_q"))?,
                linear(width, width * 2, vb.pp("c_kv"))?,
            )
        } else {
            (
                linear_no_bias(width, width, vb.pp("c_q"))?,
                linear_no_bias(width, width * 2, vb.pp("c_kv"))?,
            )
        };
        Ok(Self {
            c_q,
            c_kv,
            c_proj: linear(width, width, vb.pp("c_proj"))?,
            norms: QkNorm::new(cfg.head_dim(), cfg.qk_norm, vb.pp("attention"))?,
            heads: cfg.heads,
        })
    }

    /// The query-independent half: `c_kv(data)`. Upstream caches exactly this
    /// behind its `kv_cache` flag (`vae.py:667-676`).
    fn project_kv(&self, data: &Tensor) -> Result<Tensor> {
        self.c_kv.forward(data)
    }

    /// `x` is `[B, Nq, width]`, `kv` is a [`Self::project_kv`] output
    /// `[B, Nd, 2 * width]`.
    fn forward_with_kv(&self, xs: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let (b, n_ctx, _) = xs.dims3()?;
        let (_, n_data, kv_width) = kv.dims3()?;
        let head_dim = kv_width / self.heads / 2;

        let q = self
            .c_q
            .forward(xs)?
            .reshape((b, n_ctx, self.heads, head_dim))?;
        let kv = kv.reshape((b, n_data, self.heads, 2 * head_dim))?;
        let q = self.norms.apply_q(&q.contiguous()?)?;
        let k = self
            .norms
            .apply_k(&kv.narrow(3, 0, head_dim)?.contiguous()?)?;
        let v = kv.narrow(3, head_dim, head_dim)?.contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = attention(&to_bhnd(&q)?, &to_bhnd(&k)?, &to_bhnd(&v)?, scale as f32)?;
        self.c_proj.forward(&from_bhnd(&out)?)
    }
}

/// `ResidualCrossAttentionBlock` (`vae.py:687-721`).
///
/// The norm placement is the part worth reading twice
/// (`vae.py:717-721`):
///
/// ```text
/// x = x + attn(ln_1(x), ln_2(data))
/// x = x + mlp(ln_3(x))
/// ```
///
/// `ln_2` normalises the **data** (the prepared latents), not the queries, and
/// `ln_3` — not `ln_2` — is the pre-MLP norm on the query stream. A port that
/// reuses `ln_2` for the MLP the way `ResidualAttentionBlock` does will load
/// the wrong tensor and still run.
#[derive(Debug)]
struct ResidualCrossAttentionBlock {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    ln_3: LayerNorm,
    attn: CrossAttention,
    mlp: Mlp,
}

impl ResidualCrossAttentionBlock {
    fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: block_ln(cfg.width, vb.pp("ln_1"))?,
            ln_2: block_ln(cfg.width, vb.pp("ln_2"))?,
            ln_3: block_ln(cfg.width, vb.pp("ln_3"))?,
            attn: CrossAttention::new(cfg, vb.pp("attn"))?,
            mlp: Mlp::new(cfg.width, cfg.geo_decoder_mlp_expand_ratio, vb.pp("mlp"))?,
        })
    }

    /// `c_kv(ln_2(data))` — everything in the block that does not depend on the
    /// queries.
    fn project_kv(&self, data: &Tensor) -> Result<Tensor> {
        self.attn.project_kv(&self.ln_2.forward(data)?)
    }

    fn forward_with_kv(&self, xs: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward_with_kv(&self.ln_1.forward(xs)?, kv)?)?;
        &xs + self.mlp.forward(&self.ln_3.forward(&xs)?)?
    }
}

/// `CrossAttentionDecoder` (`vae.py:846-896`), minus the `downsample_ratio != 1`
/// `latents_proj` branch that no 2.0 checkpoint carries.
#[derive(Debug)]
struct GeoDecoder {
    fourier: FourierEmbedder,
    query_proj: Linear,
    cross_attn_decoder: ResidualCrossAttentionBlock,
    ln_post: Option<LayerNorm>,
    output_proj: Linear,
}

impl GeoDecoder {
    fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        let fourier = FourierEmbedder::new(cfg.num_freqs, 3, true, cfg.include_pi);
        let ln_post = if cfg.geo_decoder_ln_post {
            Some(layer_norm(
                cfg.width,
                LayerNormConfig {
                    eps: LN_POST_EPS,
                    remove_mean: true,
                    affine: true,
                },
                vb.pp("ln_post"),
            )?)
        } else {
            None
        };
        Ok(Self {
            query_proj: linear(fourier.out_dim(), cfg.width, vb.pp("query_proj"))?,
            fourier,
            cross_attn_decoder: ResidualCrossAttentionBlock::new(cfg, vb.pp("cross_attn_decoder"))?,
            ln_post,
            output_proj: linear(cfg.width, cfg.out_channels, vb.pp("output_proj"))?,
        })
    }
}

/// Hunyuan3D 2.0 shape VAE decoder.
#[derive(Debug)]
pub struct ShapeVae {
    cfg: ShapeVaeConfig,
    post_kl: Linear,
    resblocks: Vec<ResidualAttentionBlock>,
    geo_decoder: GeoDecoder,
}

impl ShapeVae {
    /// `vb` must already be scoped to the checkpoint's `vae.` prefix.
    pub fn new(cfg: &ShapeVaeConfig, vb: VarBuilder) -> Result<Self> {
        let post_kl = linear(cfg.embed_dim, cfg.width, vb.pp("post_kl"))?;
        let vb_t = vb.pp("transformer").pp("resblocks");
        let mut resblocks = Vec::with_capacity(cfg.num_decoder_layers);
        for i in 0..cfg.num_decoder_layers {
            resblocks.push(ResidualAttentionBlock::new(cfg, vb_t.pp(i))?);
        }
        Ok(Self {
            cfg: *cfg,
            post_kl,
            resblocks,
            geo_decoder: GeoDecoder::new(cfg, vb.pp("geo_decoder"))?,
        })
    }

    /// The configuration this decoder was built from.
    pub fn config(&self) -> &ShapeVaeConfig {
        &self.cfg
    }

    /// The query embedder, exposed so callers (and tests) can size a
    /// `query_proj` input or pre-embed a reusable grid.
    pub fn fourier_embedder(&self) -> &FourierEmbedder {
        &self.geo_decoder.fourier
    }

    /// Divide DiT latents by `scale_factor`.
    ///
    /// Both references apply this, in different places, which is easy to miss:
    /// `ShapeVAE.decode` itself never touches `scale_factor` (`comfy/sd.py:866`
    /// even constructs `ShapeVAE()` with the class default and drops the
    /// checkpoint's value), because in ComfyUI the division happens one layer
    /// up — `LatentFormat.process_out` is `latent / self.scale_factor`
    /// (`comfy/latent_formats.py:18`), called by `process_latent_out`
    /// (`comfy/model_base.py:378`) on the sampler's output before the VAE sees
    /// it, using the per-checkpoint `Hunyuan3Dv2` / `Hunyuan3Dv2mini` values.
    /// Tencent does the same division inline in
    /// `hy3dgen/shapegen/pipelines.py`.
    ///
    /// It stays a separate call rather than hiding inside
    /// [`Self::prepare_latents`] because that is where the two references put
    /// it: it belongs to the sampler's output contract, not to the decoder.
    pub fn unscale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents.affine(1.0 / self.cfg.scale_factor, 0.0)
    }

    /// `post_kl` + the 16-layer self-attention transformer. **Run once per
    /// generation**, then feed the result to every
    /// [`Self::decode_queries`] chunk.
    ///
    /// `latents` is `[B, embed_dim, num_latents]` — channels *before* tokens,
    /// which is how the DiT emits them. `ShapeVAE.decode` (`vae.py:967`) opens
    /// with `self.post_kl(latents.movedim(-2, -1))`, so the transpose to
    /// `[B, num_latents, embed_dim]` happens here and is load-bearing: feeding
    /// an already-transposed tensor makes `post_kl` see 3072 "channels" and
    /// fail, or worse, silently succeed on a square input.
    ///
    /// Returns `[B, num_latents, width]`.
    pub fn prepare_latents(&self, latents: &Tensor) -> Result<Tensor> {
        // vae.py:967 — `latents.movedim(-2, -1)`.
        let xs = latents.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut xs = self.post_kl.forward(&xs)?;
        for block in &self.resblocks {
            xs = block.forward(&xs)?;
        }
        Ok(xs)
    }

    /// Project the prepared latents through the cross-attention block's
    /// `ln_2` + `c_kv` once, so a chunked decode does not redo it per chunk.
    ///
    /// This is upstream's `kv_cache` (`vae.py:667-676`) made explicit instead of
    /// stateful. Pass the result to [`Self::decode_queries_cached`].
    pub fn prepare_cross_kv(&self, prepared: &Tensor) -> Result<Tensor> {
        self.geo_decoder.cross_attn_decoder.project_kv(prepared)
    }

    /// Cross-attend one chunk of query points against the prepared latents.
    ///
    /// `queries` is `[B, N, 3]` in model space (the caller's grid, typically
    /// bounded by +/-1.01); `prepared` is a [`Self::prepare_latents`] output.
    /// Returns `[B, N]` occupancy **logits** — `label_type` is `"binary"`, so
    /// there is no sigmoid here and the mesher thresholds the raw logit.
    ///
    /// Prefer [`Self::decode_queries_cached`] in a chunk loop.
    pub fn decode_queries(&self, queries: &Tensor, prepared: &Tensor) -> Result<Tensor> {
        let kv = self.prepare_cross_kv(prepared)?;
        self.decode_queries_cached(queries, &kv)
    }

    /// [`Self::decode_queries`] with the kv projection hoisted out of the loop.
    pub fn decode_queries_cached(&self, queries: &Tensor, cross_kv: &Tensor) -> Result<Tensor> {
        let geo = &self.geo_decoder;
        // `self.query_proj(self.fourier_embedder(queries).to(latents.dtype))`
        // — vae.py:888.
        let embedded = geo.fourier.forward(queries)?.to_dtype(cross_kv.dtype())?;
        let xs = geo.query_proj.forward(&embedded)?;
        let xs = geo.cross_attn_decoder.forward_with_kv(&xs, cross_kv)?;
        let xs = match &geo.ln_post {
            Some(ln) => ln.forward(&xs)?,
            None => xs,
        };
        let occ = geo.output_proj.forward(&xs)?;
        // `out_channels` is 1; drop the singleton so callers get [B, N].
        occ.i((.., .., 0))
    }

    /// Fold a flat `[B, (res + 1)^3]` logit sequence — the concatenation of
    /// every [`Self::decode_queries`] chunk, in [`query_grid`] order — into the
    /// grid layout the mesher expects.
    ///
    /// `VanillaVolumeDecoder` views the concatenated logits as
    /// `(B, R, R, R)` in `ij` order, i.e. axes `(x, y, z)` (`vae.py:455`).
    /// TWO upstream moves then sit between that tensor and the mesher, and
    /// both are reproduced here because each one alone rotates the mesh:
    ///
    /// 1. `ShapeVAE.decode` returns `grid_logits.movedim(-2, -1)`
    ///    (`vae.py:976`), swapping the last two axes to `(x, z, y)`.
    /// 2. ComfyUI never calls `ShapeVAE.decode` directly: `VAEDecodeHunyuan3D`
    ///    goes through the generic `comfy.sd.VAE.decode` wrapper, which ends
    ///    every decode — image, video or voxel — with `movedim(1, -1)`
    ///    (`comfy/sd.py:1277`, the channels-last step meant for pictures).
    ///    On a `(B, x, z, y)` grid that moves `x` to the end: `(z, y, x)`.
    ///
    /// The mesher (`nodes_hunyuan3d.py:229-413`) therefore walks a `[z][y][x]`
    /// grid, emits vertex columns in that order and finishes with
    /// `torch.fliplr`, so its final vertices are `(x, y, z)` — the raw query
    /// coordinates, with every transpose cancelled. Porting only the first
    /// move (as this function once did) hands glTF a mesh whose axes are the
    /// cyclic permutation `(y, z, x)` of the oracle's, which is how the first
    /// real render came out lying on its side. Result is `[B, R, R, R]`
    /// indexed `[b, z, y, x]`.
    pub fn reshape_grid_logits(logits: &Tensor, octree_resolution: usize) -> Result<Tensor> {
        let (b, n) = logits.dims2()?;
        let r = octree_resolution + 1;
        if n != r * r * r {
            candle_core::bail!(
                "expected {} logits for resolution {octree_resolution}, got {n}",
                r * r * r
            );
        }
        logits
            .reshape((b, r, r, r))?
            // `(x, y, z)` -> `(x, z, y)`: `ShapeVAE.decode`'s own movedim.
            .transpose(D::Minus2, D::Minus1)?
            // `(x, z, y)` -> `(z, y, x)`: the VAE wrapper's channels-last
            // movedim(1, -1).
            .permute((0, 2, 3, 1))?
            .contiguous()
    }
}

/// Number of query points in the grid for `octree_resolution`.
pub fn query_grid_len(octree_resolution: usize) -> usize {
    let r = octree_resolution + 1;
    r * r * r
}

/// `i`-th coordinate of `torch.linspace(-bounds, bounds, steps)`.
///
/// Torch pins both endpoints exactly; computing `start + i * step` alone drifts
/// on the last sample, which would put the outermost grid plane slightly inside
/// the bounding box.
fn linspace_at(bounds: f32, steps: usize, i: usize) -> f32 {
    if steps <= 1 {
        return -bounds;
    }
    if i == 0 {
        return -bounds;
    }
    if i + 1 == steps {
        return bounds;
    }
    let start = -bounds as f64;
    let step = (bounds as f64 - start) / (steps - 1) as f64;
    (start + i as f64 * step) as f32
}

/// A contiguous slice of the query grid: points `start .. start + len` of
/// [`query_grid`], as `[len, 3]`.
///
/// This is the allocation-friendly entry point for a chunked decode — it never
/// builds the full `(res + 1)^3 x 3` buffer. A `start` at or past the end
/// yields a `[0, 3]` tensor.
pub fn query_grid_chunk(
    octree_resolution: usize,
    bounds: f32,
    start: usize,
    len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let steps = octree_resolution + 1;
    let total = query_grid_len(octree_resolution);
    let start = start.min(total);
    let len = len.min(total - start);

    let mut data = Vec::with_capacity(len * 3);
    for idx in start..start + len {
        // `meshgrid(x, y, z, indexing="ij")` then `stack(..., -1).reshape(-1, 3)`
        // (vae.py:440-441) — x is the slowest axis, z the fastest.
        let iz = idx % steps;
        let iy = (idx / steps) % steps;
        let ix = idx / (steps * steps);
        data.push(linspace_at(bounds, steps, ix));
        data.push(linspace_at(bounds, steps, iy));
        data.push(linspace_at(bounds, steps, iz));
    }
    Tensor::from_vec(data, (len, 3), device)?.to_dtype(dtype)
}

/// The full `(res + 1)^3 x 3` query grid, exactly as
/// `VanillaVolumeDecoder.__call__` builds it (`vae.py:436-441`): a per-axis
/// `torch.linspace(-bounds, bounds, res + 1)`, `meshgrid(..., indexing="ij")`,
/// `stack(..., -1).reshape(-1, 3)`.
///
/// Upstream's default `bounds` is `1.01` (`vae.py:970`).
///
/// **This allocates the whole grid.** At `octree_resolution = 256` that is
/// 16_974_593 points — ~204 MB in F32. Use [`query_grid_chunk`] in the decode
/// loop; this exists for tests and low resolutions.
pub fn query_grid(
    octree_resolution: usize,
    bounds: f32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    query_grid_chunk(
        octree_resolution,
        bounds,
        0,
        query_grid_len(octree_resolution),
        device,
        dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic small pseudo-random values; no external rng dependency and
    /// no reliance on a device-side seed.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (self.0 >> 40) as f32 / 16_777_216.0; // [0, 1)
            (bits - 0.5) * 0.4
        }

        fn tensor(&mut self, shape: (usize, usize), device: &Device) -> Tensor {
            let n = shape.0 * shape.1;
            let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
            Tensor::from_vec(data, shape, device).expect("test tensor")
        }

        fn vec1(&mut self, n: usize, device: &Device) -> Tensor {
            let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
            Tensor::from_vec(data, n, device).expect("test tensor")
        }
    }

    fn tiny_config() -> ShapeVaeConfig {
        ShapeVaeConfig {
            num_latents: 8,
            embed_dim: 4,
            width: 32,
            heads: 2,
            num_decoder_layers: 2,
            ..ShapeVaeConfig::v2_0()
        }
    }

    fn synthetic_weights(cfg: &ShapeVaeConfig, device: &Device) -> HashMap<String, Tensor> {
        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut map = HashMap::new();
        let w = cfg.width;
        let hd = cfg.head_dim();
        let ff = cfg.width * cfg.mlp_expand_ratio;

        let lin = |map: &mut HashMap<String, Tensor>,
                   rng: &mut Lcg,
                   prefix: &str,
                   out: usize,
                   inp: usize,
                   bias: bool| {
            map.insert(format!("{prefix}.weight"), rng.tensor((out, inp), device));
            if bias {
                map.insert(format!("{prefix}.bias"), rng.vec1(out, device));
            }
        };
        let ln = |map: &mut HashMap<String, Tensor>, prefix: &str, n: usize| {
            map.insert(
                format!("{prefix}.weight"),
                Tensor::ones(n, DType::F32, device).expect("ones"),
            );
            map.insert(
                format!("{prefix}.bias"),
                Tensor::zeros(n, DType::F32, device).expect("zeros"),
            );
        };

        lin(&mut map, &mut rng, "post_kl", w, cfg.embed_dim, true);
        for i in 0..cfg.num_decoder_layers {
            let p = format!("transformer.resblocks.{i}");
            ln(&mut map, &format!("{p}.ln_1"), w);
            ln(&mut map, &format!("{p}.ln_2"), w);
            lin(
                &mut map,
                &mut rng,
                &format!("{p}.attn.c_qkv"),
                3 * w,
                w,
                cfg.qkv_bias,
            );
            lin(&mut map, &mut rng, &format!("{p}.attn.c_proj"), w, w, true);
            ln(&mut map, &format!("{p}.attn.attention.q_norm"), hd);
            ln(&mut map, &format!("{p}.attn.attention.k_norm"), hd);
            lin(&mut map, &mut rng, &format!("{p}.mlp.c_fc"), ff, w, true);
            lin(&mut map, &mut rng, &format!("{p}.mlp.c_proj"), w, ff, true);
        }

        let fourier_dim = 3 * (cfg.num_freqs * 2 + 1);
        lin(
            &mut map,
            &mut rng,
            "geo_decoder.query_proj",
            w,
            fourier_dim,
            true,
        );
        let c = "geo_decoder.cross_attn_decoder";
        ln(&mut map, &format!("{c}.ln_1"), w);
        ln(&mut map, &format!("{c}.ln_2"), w);
        ln(&mut map, &format!("{c}.ln_3"), w);
        lin(
            &mut map,
            &mut rng,
            &format!("{c}.attn.c_q"),
            w,
            w,
            cfg.qkv_bias,
        );
        lin(
            &mut map,
            &mut rng,
            &format!("{c}.attn.c_kv"),
            2 * w,
            w,
            cfg.qkv_bias,
        );
        lin(&mut map, &mut rng, &format!("{c}.attn.c_proj"), w, w, true);
        ln(&mut map, &format!("{c}.attn.attention.q_norm"), hd);
        ln(&mut map, &format!("{c}.attn.attention.k_norm"), hd);
        lin(&mut map, &mut rng, &format!("{c}.mlp.c_fc"), ff, w, true);
        lin(&mut map, &mut rng, &format!("{c}.mlp.c_proj"), w, ff, true);
        ln(&mut map, "geo_decoder.ln_post", w);
        lin(
            &mut map,
            &mut rng,
            "geo_decoder.output_proj",
            cfg.out_channels,
            w,
            true,
        );

        map
    }

    fn tiny_vae(device: &Device) -> (ShapeVaeConfig, ShapeVae) {
        tiny_vae_on(device, DType::F32)
    }

    /// The weights are authored on the CPU in F32 and cast + moved by
    /// `VarBuilder::from_tensors` as the model asks for them, so one
    /// generator serves every device and dtype.
    fn tiny_vae_on(device: &Device, dtype: DType) -> (ShapeVaeConfig, ShapeVae) {
        let cfg = tiny_config();
        let vb = VarBuilder::from_tensors(synthetic_weights(&cfg, &Device::Cpu), dtype, device);
        let vae = ShapeVae::new(&cfg, vb).expect("build tiny ShapeVae");
        (cfg, vae)
    }

    /// Latents in the shape the DiT emits, deterministic and small. Only the
    /// accelerator forward tests use it, so a build without either GPU
    /// feature would otherwise see it as dead code.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn tiny_latents(cfg: &ShapeVaeConfig, device: &Device, dtype: DType) -> Tensor {
        let mut rng = Lcg(0x0FF1_CE00_1234_5678);
        let data: Vec<f32> = (0..cfg.embed_dim * cfg.num_latents)
            .map(|_| rng.next_f32())
            .collect();
        Tensor::from_vec(data, (1, cfg.embed_dim, cfg.num_latents), &Device::Cpu)
            .expect("latents")
            .to_device(device)
            .expect("move")
            .to_dtype(dtype)
            .expect("cast")
    }

    #[test]
    fn fourier_embedder_out_dim_matches_query_proj_input() {
        let with_input = FourierEmbedder::new(8, 3, true, false);
        assert_eq!(with_input.out_dim(), 51);
        let without_input = FourierEmbedder::new(8, 3, false, false);
        assert_eq!(without_input.out_dim(), 48);
    }

    #[test]
    fn fourier_embedder_layout_is_input_dim_major() {
        let device = Device::Cpu;
        let embedder = FourierEmbedder::new(8, 3, true, false);
        let xs = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (1, 1, 3), &device).expect("input");
        let out = embedder.forward(&xs).expect("embed");
        assert_eq!(out.dims(), &[1, 1, 51]);

        let v = out
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        // The raw input rides in front.
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6 && v[2].abs() < 1e-6);
        // Then 24 sines: dim-major, so v[3..11] are x0 * [1, 2, 4, ... 128].
        for (i, expected) in [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
            .into_iter()
            .enumerate()
        {
            assert!(
                (v[3 + i] - expected.sin()).abs() < 1e-5,
                "sin band {i}: {} vs {}",
                v[3 + i],
                expected.sin()
            );
            assert!(
                (v[27 + i] - expected.cos()).abs() < 1e-5,
                "cos band {i}: {} vs {}",
                v[27 + i],
                expected.cos()
            );
        }
        // x1 and x2 are zero, so their sin bands vanish and cos bands are 1.
        for i in 8..24 {
            assert!(v[3 + i].abs() < 1e-6, "sin slot {i} should be 0");
            assert!((v[27 + i] - 1.0).abs() < 1e-6, "cos slot {i} should be 1");
        }

        // include_input = false drops the leading 3 and keeps the rest.
        let no_input = FourierEmbedder::new(8, 3, false, false);
        let out = no_input.forward(&xs).expect("embed");
        assert_eq!(out.dims(), &[1, 1, 48]);
    }

    #[test]
    fn query_grid_matches_ij_meshgrid() {
        let device = Device::Cpu;
        let res = 4usize;
        let grid = query_grid(res, 1.01, &device, DType::F32).expect("grid");
        assert_eq!(grid.dims(), &[query_grid_len(res), 3]);
        assert_eq!(query_grid_len(res), 125);

        let v = grid
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert_eq!(&v[0..3], &[-1.01, -1.01, -1.01]);
        let n = v.len();
        assert_eq!(&v[n - 3..], &[1.01, 1.01, 1.01]);

        // indexing="ij": the fastest-varying axis is the LAST one, so the
        // second point differs from the first only in z.
        assert_eq!(v[3], -1.01);
        assert_eq!(v[4], -1.01);
        assert!(v[5] > -1.01, "third component should advance first");

        // Chunking the grid reproduces it exactly.
        let chunk = query_grid_chunk(res, 1.01, 60, 10, &device, DType::F32).expect("chunk");
        let cv = chunk
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert_eq!(cv.as_slice(), &v[60 * 3..70 * 3]);
    }

    #[test]
    fn tiny_shape_vae_round_trip_shapes_are_finite() {
        let device = Device::Cpu;
        let (cfg, vae) = tiny_vae(&device);

        let latents = Tensor::ones((1, cfg.embed_dim, cfg.num_latents), DType::F32, &device)
            .expect("latents")
            .affine(0.1, -0.05)
            .expect("scale");
        let prepared = vae.prepare_latents(&latents).expect("prepare");
        assert_eq!(prepared.dims(), &[1, cfg.num_latents, cfg.width]);

        let queries = query_grid(3, 1.01, &device, DType::F32)
            .expect("grid")
            .unsqueeze(0)
            .expect("batch");
        let n = queries.dim(1).expect("n");
        let logits = vae.decode_queries(&queries, &prepared).expect("decode");
        assert_eq!(logits.dims(), &[1, n]);

        let values = logits
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );

        let grid = ShapeVae::reshape_grid_logits(&logits, 3).expect("reshape");
        assert_eq!(grid.dims(), &[1, 4, 4, 4]);
    }

    /// Pins both upstream moves at once. A logit tagged with its query index
    /// as `100 x + 10 y + z` must land at `[b, z, y, x]`: the ShapeVAE's
    /// `movedim(-2, -1)` (`vae.py:976`) followed by the VAE wrapper's
    /// `movedim(1, -1)` (`comfy/sd.py:1277`). Reproducing only the first
    /// leaves `x` in front and rotates every mesh.
    #[test]
    fn reshape_grid_logits_hands_the_mesher_upstreams_channels_last_grid() {
        let r = 3usize;
        let mut tagged = Vec::with_capacity(r * r * r);
        for x in 0..r {
            for y in 0..r {
                for z in 0..r {
                    tagged.push((100 * x + 10 * y + z) as f32);
                }
            }
        }
        let flat = Tensor::from_vec(tagged, (1, r * r * r), &Device::Cpu).expect("flat");
        let grid = ShapeVae::reshape_grid_logits(&flat, r - 1).expect("reshape");
        assert_eq!(grid.dims(), &[1, r, r, r]);
        let values = grid
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        for a0 in 0..r {
            for a1 in 0..r {
                for a2 in 0..r {
                    let got = values[(a0 * r + a1) * r + a2];
                    let (z, y, x) = (a0, a1, a2);
                    assert_eq!(
                        got,
                        (100 * x + 10 * y + z) as f32,
                        "grid[{a0}][{a1}][{a2}] must hold the logit of query (x={x}, y={y}, z={z})"
                    );
                }
            }
        }
    }

    #[test]
    fn decode_is_invariant_to_query_chunking() {
        let device = Device::Cpu;
        let (cfg, vae) = tiny_vae(&device);

        let latents = Tensor::ones((1, cfg.embed_dim, cfg.num_latents), DType::F32, &device)
            .expect("latents")
            .affine(0.1, -0.05)
            .expect("scale");
        let prepared = vae.prepare_latents(&latents).expect("prepare");
        let kv = vae.prepare_cross_kv(&prepared).expect("kv");

        let mut rng = Lcg(0xC0FF_EE00_1234_5678);
        let points: Vec<f32> = (0..300).map(|_| rng.next_f32() * 5.0).collect();
        let queries = Tensor::from_vec(points, (1, 100, 3), &device).expect("queries");

        let whole = vae
            .decode_queries_cached(&queries, &kv)
            .expect("whole")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");

        let mut chunked = Vec::with_capacity(100);
        for start in (0..100).step_by(25) {
            let chunk = queries.narrow(1, start, 25).expect("narrow");
            chunked.extend(
                vae.decode_queries_cached(&chunk, &kv)
                    .expect("chunk")
                    .flatten_all()
                    .expect("flat")
                    .to_vec1::<f32>()
                    .expect("vec"),
            );
        }

        assert_eq!(whole.len(), chunked.len());
        for (i, (a, b)) in whole.iter().zip(chunked.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "point {i}: whole={a} chunked={b} (chunked decode must be a no-op)"
            );
        }

        // The uncached entry point must agree with the cached one too.
        let uncached = vae
            .decode_queries(&queries, &prepared)
            .expect("uncached")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        for (a, b) in whole.iter().zip(uncached.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// The whole decode — `post_kl`, the self-attention stack, the Fourier
    /// embedder, and the cross-attention — has to run on Metal in F16, which
    /// is the dtype `super::backend::compute_dtype` picks for every
    /// accelerator. The query grid is cast to the compute dtype too, exactly
    /// as `comfy/ldm/hunyuan3d/vae.py:442` casts it to `latents.dtype`.
    #[cfg(feature = "metal")]
    #[test]
    fn tiny_vae_decode_runs_on_metal_in_f16() {
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let (cfg, reference) = tiny_vae_on(&cpu, DType::F32);
        let (_, vae) = tiny_vae_on(&metal, DType::F16);

        let want = {
            let latents = tiny_latents(&cfg, &cpu, DType::F32);
            let queries = query_grid(3, 1.01, &cpu, DType::F32)
                .expect("grid")
                .unsqueeze(0)
                .expect("batch");
            let prepared = reference.prepare_latents(&latents).expect("prepare");
            reference
                .decode_queries(&queries, &prepared)
                .expect("cpu decode")
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec")
        };

        let latents = tiny_latents(&cfg, &metal, DType::F16);
        let queries = query_grid(3, 1.01, &metal, DType::F16)
            .expect("grid")
            .unsqueeze(0)
            .expect("batch");
        let prepared = vae.prepare_latents(&latents).expect("prepare");
        let logits = vae
            .decode_queries(&queries, &prepared)
            .expect("metal decode");
        assert_eq!(logits.dims(), &[1, query_grid_len(3)]);

        let got = logits
            .to_dtype(DType::F32)
            .expect("widen")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert_eq!(got.len(), want.len());
        for (index, (metal_value, cpu_value)) in got.iter().zip(&want).enumerate() {
            assert!(metal_value.is_finite(), "point {index} is not finite");
            assert!(
                (metal_value - cpu_value).abs() < 5e-2,
                "point {index}: metal F16 {metal_value} vs cpu F32 {cpu_value}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_vae_decode_runs_on_cuda_in_f16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let (cfg, reference) = tiny_vae_on(&cpu, DType::F32);
        let (_, vae) = tiny_vae_on(&cuda, DType::F16);

        let want = {
            let latents = tiny_latents(&cfg, &cpu, DType::F32);
            let queries = query_grid(3, 1.01, &cpu, DType::F32)
                .expect("grid")
                .unsqueeze(0)
                .expect("batch");
            let prepared = reference.prepare_latents(&latents).expect("prepare");
            reference
                .decode_queries(&queries, &prepared)
                .expect("cpu decode")
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec")
        };

        let latents = tiny_latents(&cfg, &cuda, DType::F16);
        let queries = query_grid(3, 1.01, &cuda, DType::F16)
            .expect("grid")
            .unsqueeze(0)
            .expect("batch");
        let prepared = vae.prepare_latents(&latents).expect("prepare");
        let logits = vae
            .decode_queries(&queries, &prepared)
            .expect("cuda decode");
        assert_eq!(logits.dims(), &[1, query_grid_len(3)]);

        let got = logits
            .to_dtype(DType::F32)
            .expect("widen")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert_eq!(got.len(), want.len());
        for (index, (cuda_value, cpu_value)) in got.iter().zip(&want).enumerate() {
            assert!(cuda_value.is_finite(), "point {index} is not finite");
            assert!(
                (cuda_value - cpu_value).abs() < 5e-2,
                "point {index}: cuda F16 {cuda_value} vs cpu F32 {cpu_value}"
            );
        }
    }

    #[test]
    fn config_variants_differ_only_in_scale_factor() {
        let base = ShapeVaeConfig::v2_0();
        let mini = ShapeVaeConfig::v2_0_mini();
        assert_eq!(base.width, 1024);
        assert_eq!(base.head_dim(), 64);
        assert!((base.scale_factor - 0.999_094_304_262_252_9).abs() < 1e-15);
        assert!((mini.scale_factor - 1.018_813_714_239_540_4).abs() < 1e-15);
        assert_eq!(
            ShapeVaeConfig {
                scale_factor: base.scale_factor,
                ..mini
            },
            base
        );
    }
}
