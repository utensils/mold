//! DINOv2-giant — the only conditioner Hunyuan3D 2.0 has.
//!
//! Hunyuan3D has no text encoder: the DiT in [`super::transformer`] is
//! cross-attended to the token sequence this tower produces from a single
//! (background-removed) source image, and the negative branch is literally
//! `zeros_like` of that sequence (`comfy_extras/nodes_hunyuan3d.py:47-52`).
//! So this file is the whole conditioning stack.
//!
//! ## Upstream references
//!
//! * `comfy/image_encoders/dino2.py` — the authoritative port target. Every
//!   submodule below mirrors one of its classes: `Dino2PatchEmbeddings`,
//!   `Dino2Embeddings` (including `interpolate_pos_encoding`), `BertAttention`
//!   + `Dino2AttentionOutput` (`comfy/text_encoders/bert.py:5-21`),
//!     `LayerScale`, `SwiGLUFFN`, `Dinov2MLP`, `Dino2Block`, `Dino2Encoder`,
//!     `Dinov2Model`.
//! * `comfy/image_encoders/dino2_giant.json` — the config [`Dinov2Config::giant`]
//!   reproduces. `comfy/clip_vision.py:139-141` selects it by probing for
//!   `encoder.layer.39.layer_scale2.lambda1`, which is how we know a 40-layer
//!   checkpoint is the giant.
//! * `hy3dgen/shapegen/preprocessors.py` (`Tencent-Hunyuan/Hunyuan3D-2`) —
//!   `ImageProcessorV2.recenter` / `load_image`, mirrored by [`preprocess`].
//! * `hy3dgen/shapegen/models/conditioner.py` — `ImageEncoder.__init__`'s
//!   `Resize(image_size, BILINEAR, antialias=True)` + `CenterCrop(image_size)`,
//!   which is the SECOND resize [`preprocess`] performs. The letterbox size
//!   and the encoder size are different numbers on the 1.1B tiers (512 and
//!   518) and equal on the mini tier (1022), so [`preprocess`] takes both.
//!
//! The DA3 extensions in `dino2.py` (2-D RoPE, QK-norm, alternating
//! cross-view attention, camera tokens) are all keyed off config fields that
//! the giant config does not set (`alt_start`/`qknorm_start`/`rope_start`
//! default to `-1`), so they are deliberately absent here.
//!
//! ## Weight layout
//!
//! [`Dinov2Model::new`] takes a [`VarBuilder`] already scoped to the
//! checkpoint's `conditioner.main_image_encoder.model.` prefix, so the names
//! below are relative. Read off the `model.fp16.safetensors` header of
//! `tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0`:
//!
//! ```text
//! embeddings.cls_token                                 [1, 1, 1536]
//! embeddings.mask_token                                [1, 1536]        (unused)
//! embeddings.position_embeddings                       [1, 1370, 1536]  (1 + 37*37)
//! embeddings.patch_embeddings.projection.weight        [1536, 3, 14, 14]
//! embeddings.patch_embeddings.projection.bias          [1536]
//! encoder.layer.{0..39}.norm1.{weight,bias}            [1536]
//! encoder.layer.{0..39}.attention.attention.{query,key,value}.weight  [1536, 1536]
//! encoder.layer.{0..39}.attention.attention.{query,key,value}.bias    [1536]
//! encoder.layer.{0..39}.attention.output.dense.{weight,bias}
//! encoder.layer.{0..39}.layer_scale1.lambda1           [1536]
//! encoder.layer.{0..39}.norm2.{weight,bias}            [1536]
//! encoder.layer.{0..39}.mlp.weights_in.weight          [8192, 1536]   (2 * 4096)
//! encoder.layer.{0..39}.mlp.weights_out.weight         [1536, 4096]
//! encoder.layer.{0..39}.layer_scale2.lambda1           [1536]
//! layernorm.{weight,bias}                              [1536]
//! ```
//!
//! `embeddings.mask_token` is a pre-training parameter that nothing reads;
//! `dino2.py:236` keeps it only so a strict `load_state_dict` accepts the key.
//! A `VarBuilder` never has to be exhaustive, so we simply do not ask for it.

// Nothing in the crate calls this yet — the engine that loads a Hunyuan3D
// checkpoint lands with the rest of the pipeline. Until then every item here
// is reachable only from this file's tests.
#![allow(dead_code)]

use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv2d, LayerNorm, Linear, VarBuilder};
use image::DynamicImage;

/// ImageNet statistics, as `dino2_giant.json`'s `image_mean` / `image_std` and
/// `DinoImageEncoder.{mean,std}` in `hy3dgen/shapegen/models/conditioner.py`
/// both spell them.
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `ImageProcessorV2.load_image`'s default (`preprocessors.py:69`). Note this
/// is *not* `recenter`'s own `0.2` default — Hunyuan3D always overrides it.
pub const BORDER_RATIO: f32 = 0.15;

/// Cubic coefficient of `torch.nn.functional.interpolate(mode="bicubic")`,
/// which is what `dino2.py:264` resamples the position grid with. PyTorch's
/// `get_cubic_upsample_coefficients` hard-codes `A = -0.75`; the antialiased
/// `torchvision` resize used elsewhere in this crate uses `-0.5`, and the two
/// are not interchangeable.
const POS_EMBED_CUBIC_A: f64 = -0.75;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The upstream convention for resampling a stored position grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionInterpolation {
    /// ComfyUI's inherited DINO scale-factor workaround (adds 0.1).
    ComfyScale,
    /// Transformers 4.46 uses an explicit output size, without the offset.
    TransformersSize,
}

/// DINO inference configuration. Training-only dropout and initialization
/// settings are omitted; the two MLP variants fix their own activation.
#[derive(Debug, Clone, PartialEq)]
pub struct Dinov2Config {
    pub position_interpolation: PositionInterpolation,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    /// Side length the *stored* position embeddings were trained at. The
    /// forward pass accepts any multiple of `patch_size` and interpolates.
    pub image_size: usize,
    pub layer_norm_eps: f64,
    pub use_swiglu_ffn: bool,
    pub qkv_bias: bool,
    /// `mlp_ratio`; only consulted by the non-SwiGLU branch, where the hidden
    /// width is exactly `hidden_size * mlp_ratio`.
    pub mlp_ratio: usize,
}

impl Dinov2Config {
    /// Paint's pinned Transformers 4.46 DINO wrapper uses size-based position
    /// interpolation at a 224-pixel crop, unlike shape's ComfyUI convention.
    pub fn paint_giant() -> Self {
        Self {
            position_interpolation: PositionInterpolation::TransformersSize,
            ..Self::giant()
        }
    }
    /// Tencent's Hunyuan3D 2.1 conditioner: DINOv2-large, verified against
    /// the bundled checkpoint's 1024-wide embeddings and 24 encoder layers.
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            use_swiglu_ffn: false,
            ..Self::giant()
        }
    }

    /// `comfy/image_encoders/dino2_giant.json`, verbatim. The Hunyuan3D
    /// checkpoints ship the same numbers in their own `config.yaml`.
    pub fn giant() -> Self {
        Self {
            position_interpolation: PositionInterpolation::ComfyScale,
            hidden_size: 1536,
            num_hidden_layers: 40,
            num_attention_heads: 24,
            patch_size: 14,
            image_size: 518,
            layer_norm_eps: 1e-6,
            use_swiglu_ffn: true,
            qkv_bias: true,
            mlp_ratio: 4,
        }
    }

    /// Patches per side at the stored resolution — `sqrt` of the position
    /// grid, 37 for the giant.
    pub fn patch_grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// `1 + patch_grid()^2`, the stored `position_embeddings` length.
    pub fn stored_tokens(&self) -> usize {
        1 + self.patch_grid() * self.patch_grid()
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `SwiGLUFFN.__init__` (`dino2.py:85-87`): four times the width, then two
    /// thirds of that, then rounded *up* to a multiple of eight. For the giant
    /// this is `((1536 * 4) * 2 / 3 + 7) / 8 * 8 = 4096`, which is why
    /// `mlp.weights_in.weight` is `[8192, 1536]` and `weights_out.weight` is
    /// `[1536, 4096]`.
    fn swiglu_hidden(&self) -> usize {
        let wide = self.hidden_size * 4;
        (wide * 2 / 3).div_ceil(8) * 8
    }
}

// ---------------------------------------------------------------------------
// Position-embedding interpolation
// ---------------------------------------------------------------------------

/// One output sample's four source indices and their weights.
fn cubic_taps(source: f64, size: usize) -> ([usize; 4], [f32; 4]) {
    // `cubic_convolution1` / `cubic_convolution2` from PyTorch's
    // `UpSampleBicubic2d`.
    fn near(x: f64) -> f64 {
        ((POS_EMBED_CUBIC_A + 2.0) * x - (POS_EMBED_CUBIC_A + 3.0)) * x * x + 1.0
    }
    fn far(x: f64) -> f64 {
        ((POS_EMBED_CUBIC_A * x - 5.0 * POS_EMBED_CUBIC_A) * x + 8.0 * POS_EMBED_CUBIC_A) * x
            - 4.0 * POS_EMBED_CUBIC_A
    }

    let base = source.floor();
    let t = source - base;
    let weights = [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)];

    // `upsample_get_value_bounded` clamps the window to the border. Bicubic
    // is also the one mode whose source index is *not* clamped to zero first
    // (`area_pixel_compute_source_index`), so `base` is signed here.
    let base = base as i64;
    let last = size as i64 - 1;
    let indices = [
        (base - 1).clamp(0, last) as usize,
        base.clamp(0, last) as usize,
        (base + 1).clamp(0, last) as usize,
        (base + 2).clamp(0, last) as usize,
    ];
    (indices, weights.map(|w| w as f32))
}

/// Resample the second-from-last axis of an `[outer, in_len, channels]` buffer.
fn resample_axis(
    input: &[f32],
    outer: usize,
    in_len: usize,
    channels: usize,
    out_len: usize,
    scale: f64,
) -> Vec<f32> {
    let taps: Vec<([usize; 4], [f32; 4])> = (0..out_len)
        // `align_corners=False`: `src = (dst + 0.5) / scale - 0.5`.
        .map(|dst| cubic_taps((dst as f64 + 0.5) / scale - 0.5, in_len))
        .collect();

    let mut output = vec![0.0_f32; outer * out_len * channels];
    for row in 0..outer {
        let source_base = row * in_len * channels;
        let target_base = row * out_len * channels;
        for (dst, (indices, weights)) in taps.iter().enumerate() {
            let target = target_base + dst * channels;
            for (channel, value) in output[target..target + channels].iter_mut().enumerate() {
                let mut sum = 0.0_f32;
                for (&index, &weight) in indices.iter().zip(weights.iter()) {
                    sum += input[source_base + index * channels + channel] * weight;
                }
                *value = sum;
            }
        }
    }
    output
}

/// Swap the two leading axes of an `[h, w, channels]` buffer.
fn transpose_hw(data: &[f32], height: usize, width: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; data.len()];
    for row in 0..height {
        for column in 0..width {
            let source = (row * width + column) * channels;
            let target = (column * height + row) * channels;
            output[target..target + channels].copy_from_slice(&data[source..source + channels]);
        }
    }
    output
}

/// Bicubic resample of a `[grid, grid, channels]` patch-position grid onto
/// `[rows, columns, channels]`, mirroring `Dino2Embeddings.interpolate_pos_encoding`
/// (`dino2.py:255-270`).
///
/// The `+ 0.1` in the scale factor is upstream's floating-point rounding
/// workaround, inherited from the original DINOv2 repository: PyTorch derives
/// the output size as `floor(grid * scale)`, and `(rows + 0.1) / grid` is the
/// smallest perturbation that reliably lands on `rows`. It is not cosmetic —
/// the sampling grid is `1 / scale` spaced, so keeping it reproduces upstream's
/// samples and dropping it does not.
///
/// Separable, and always in f32 (upstream casts the parameter to `float32`
/// before interpolating regardless of the model dtype).
fn interpolate_patch_positions(
    grid: &[f32],
    grid_side: usize,
    channels: usize,
    rows: usize,
    columns: usize,
    convention: PositionInterpolation,
) -> Vec<f32> {
    let offset = match convention {
        PositionInterpolation::ComfyScale => 0.1,
        PositionInterpolation::TransformersSize => 0.,
    };
    let scale_rows = (rows as f64 + offset) / grid_side as f64;
    let scale_columns = (columns as f64 + offset) / grid_side as f64;

    // Columns first, on the natural [row, column, channel] layout.
    let horizontal = resample_axis(grid, grid_side, grid_side, channels, columns, scale_columns);
    // Then rows, by transposing so they occupy the resampled axis.
    let transposed = transpose_hw(&horizontal, grid_side, columns, channels);
    let vertical = resample_axis(&transposed, columns, grid_side, channels, rows, scale_rows);
    transpose_hw(&vertical, columns, rows, channels)
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

/// `Dino2PatchEmbeddings` — a stride-`patch_size` convolution, flattened to a
/// token sequence.
#[derive(Debug)]
struct PatchEmbeddings {
    projection: Conv2d,
}

impl PatchEmbeddings {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let projection = candle_nn::conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("projection"),
        )?;
        Ok(Self { projection })
    }

    /// `[B, 3, H, W]` -> `[B, (H/p)*(W/p), C]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        Ok(self
            .projection
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?
            .contiguous()?)
    }
}

/// `Dino2Embeddings` — patch tokens, a prepended CLS token, and the additive
/// (optionally interpolated) position grid.
#[derive(Debug)]
struct Embeddings {
    position_interpolation: PositionInterpolation,
    patch_embeddings: PatchEmbeddings,
    cls_token: Tensor,
    position_embeddings: Tensor,
    hidden_size: usize,
    patch_size: usize,
}

impl Embeddings {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        let patch_embeddings = PatchEmbeddings::new(cfg, vb.pp("patch_embeddings"))?;
        let cls_token = vb.get((1, 1, cfg.hidden_size), "cls_token")?;
        let position_embeddings = vb.get(
            (1, cfg.stored_tokens(), cfg.hidden_size),
            "position_embeddings",
        )?;
        Ok(Self {
            position_interpolation: cfg.position_interpolation,
            patch_embeddings,
            cls_token,
            position_embeddings,
            hidden_size: cfg.hidden_size,
            patch_size: cfg.patch_size,
        })
    }

    /// The position grid resampled onto the `height x width` pixel input's
    /// patch grid, as `[1, 1 + rows*columns, C]` in `dtype` on `device`.
    ///
    /// The CLS position is copied through untouched; only the patch grid is
    /// resampled (`dino2.py:258-259`).
    fn interpolate_pos_encoding(
        &self,
        height: usize,
        width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let channels = self.hidden_size;
        let stored = self.position_embeddings.dim(1)?;
        let patches = stored - 1;
        let grid_side = (patches as f64).sqrt().round() as usize;
        ensure!(
            grid_side * grid_side == patches,
            "DINOv2 position grid must be square, got {patches} patches"
        );

        let rows = height / self.patch_size;
        let columns = width / self.patch_size;
        ensure!(
            rows > 0 && columns > 0,
            "input {height}x{width} is smaller than one {}px patch",
            self.patch_size
        );

        let full = self
            .position_embeddings
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let class_position = full.i((.., 0..1, ..))?;
        let patch_grid = full.i((.., 1.., ..))?.flatten_all()?.to_vec1::<f32>()?;

        let resampled = interpolate_patch_positions(
            &patch_grid,
            grid_side,
            channels,
            rows,
            columns,
            self.position_interpolation,
        );
        let resampled = Tensor::from_vec(resampled, (1, rows * columns, channels), &Device::Cpu)?;

        Ok(Tensor::cat(&[&class_position, &resampled], 1)?
            .to_device(device)?
            .to_dtype(dtype)?)
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (batch, _, height, width) = pixel_values.dims4()?;
        let patches = self.patch_embeddings.forward(pixel_values)?;
        let cls = self
            .cls_token
            .to_dtype(patches.dtype())?
            .broadcast_as((batch, 1, self.hidden_size))?
            .contiguous()?;
        let tokens = Tensor::cat(&[&cls, &patches], 1)?;

        let positions = if tokens.dim(1)? == self.position_embeddings.dim(1)? {
            self.position_embeddings.to_dtype(tokens.dtype())?
        } else {
            self.interpolate_pos_encoding(height, width, tokens.dtype(), tokens.device())?
        };
        Ok(tokens.broadcast_add(&positions)?)
    }
}

/// `BertAttention` + `Dino2AttentionOutput`. Three separate projections, not a
/// fused QKV — the checkpoint stores them under
/// `attention.attention.{query,key,value}` and `attention.output.dense`.
#[derive(Debug)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        let size = cfg.hidden_size;
        let inner = vb.pp("attention");
        let projection = |name: &str| -> Result<Linear> {
            Ok(if cfg.qkv_bias {
                candle_nn::linear(size, size, inner.pp(name))?
            } else {
                candle_nn::linear_no_bias(size, size, inner.pp(name))?
            })
        };
        Ok(Self {
            query: projection("query")?,
            key: projection("key")?,
            value: projection("value")?,
            // `Dino2AttentionOutput.dense` always carries a bias.
            output: candle_nn::linear(size, size, vb.pp("output").pp("dense"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        let shape = (batch, tokens, self.num_heads, self.head_dim);
        let split = |projected: Tensor| -> Result<Tensor> {
            Ok(projected.reshape(shape)?.transpose(1, 2)?.contiguous()?)
        };
        let q = split(self.query.forward(xs)?)?;
        let k = split(self.key.forward(xs)?)?;
        let v = split(self.value.forward(xs)?)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attended = crate::attention::attention(&q, &k, &v, scale as f32)?;
        let merged = attended.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            tokens,
            self.num_heads * self.head_dim,
        ))?;
        Ok(self.output.forward(&merged)?)
    }
}

/// `LayerScale` — a learned per-channel gain applied to a residual branch
/// *before* it is added back (`dino2.py:110-112`).
#[derive(Debug)]
struct LayerScale {
    lambda1: Tensor,
}

impl LayerScale {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lambda1: vb.get(cfg.hidden_size, "lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.broadcast_mul(&self.lambda1.to_dtype(xs.dtype())?)?)
    }
}

/// The feed-forward branch. The giant is SwiGLU; the smaller DINOv2 variants
/// are a plain GELU MLP, and both are kept so [`Dinov2Config::use_swiglu_ffn`]
/// means something.
#[derive(Debug)]
enum FeedForward {
    /// `SwiGLUFFN`: one fused in-projection to `2 * hidden`, split down the
    /// middle, `silu(first) * second`, then the out-projection. The gate is
    /// the *first* half — `x1, x2 = x.chunk(2, -1); silu(x1) * x2`
    /// (`dino2.py:92-94`). Swapping the halves compiles and produces garbage.
    Swiglu {
        weights_in: Linear,
        weights_out: Linear,
    },
    /// `Dinov2MLP`: `fc2(gelu(fc1(x)))`.
    Mlp { fc1: Linear, fc2: Linear },
}

impl FeedForward {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        if cfg.use_swiglu_ffn {
            let hidden = cfg.swiglu_hidden();
            Ok(Self::Swiglu {
                weights_in: candle_nn::linear(cfg.hidden_size, 2 * hidden, vb.pp("weights_in"))?,
                weights_out: candle_nn::linear(hidden, cfg.hidden_size, vb.pp("weights_out"))?,
            })
        } else {
            let hidden = cfg.hidden_size * cfg.mlp_ratio;
            Ok(Self::Mlp {
                fc1: candle_nn::linear(cfg.hidden_size, hidden, vb.pp("fc1"))?,
                fc2: candle_nn::linear(hidden, cfg.hidden_size, vb.pp("fc2"))?,
            })
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Swiglu {
                weights_in,
                weights_out,
            } => {
                let projected = weights_in.forward(xs)?;
                let hidden = projected.dim(D::Minus1)? / 2;
                let gate = projected.narrow(D::Minus1, 0, hidden)?;
                let up = projected.narrow(D::Minus1, hidden, hidden)?;
                Ok(weights_out.forward(&(gate.silu()? * up)?)?)
            }
            // `torch.nn.functional.gelu` is the exact (erf) formulation;
            // candle's `gelu` is the tanh approximation, so this must be
            // `gelu_erf`.
            Self::Mlp { fc1, fc2 } => Ok(fc2.forward(&fc1.forward(xs)?.gelu_erf()?)?),
        }
    }
}

/// `Dino2Block` — pre-norm attention and FFN, each scaled by its `LayerScale`
/// before the residual add.
#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attention: Attention,
    layer_scale1: LayerScale,
    norm2: LayerNorm,
    mlp: FeedForward,
    layer_scale2: LayerScale,
}

impl Block {
    fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm1"))?,
            attention: Attention::new(cfg, vb.pp("attention"))?,
            layer_scale1: LayerScale::new(cfg, vb.pp("layer_scale1"))?,
            norm2: candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm2"))?,
            mlp: FeedForward::new(cfg, vb.pp("mlp"))?,
            layer_scale2: LayerScale::new(cfg, vb.pp("layer_scale2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended = self.attention.forward(&self.norm1.forward(xs)?)?;
        let xs = (xs + self.layer_scale1.forward(&attended)?)?;
        let expanded = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        Ok((&xs + self.layer_scale2.forward(&expanded)?)?)
    }
}

/// `Dinov2Model` — the vision tower Hunyuan3D conditions on.
#[derive(Debug)]
pub struct Dinov2Model {
    embeddings: Embeddings,
    layers: Vec<Block>,
    layernorm: LayerNorm,
    patch_size: usize,
}

impl Dinov2Model {
    /// `vb` must already be scoped to `conditioner.main_image_encoder.model.`.
    pub fn new(cfg: &Dinov2Config, vb: VarBuilder) -> Result<Self> {
        ensure!(
            cfg.num_attention_heads > 0
                && cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
                && cfg.patch_size > 0,
            "invalid DINOv2 config: {cfg:?}"
        );
        let embeddings = Embeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = vb.pp("encoder").pp("layer");
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| Block::new(cfg, encoder.pp(index)))
            .collect::<Result<Vec<_>>>()?;
        let layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        Ok(Self {
            embeddings,
            layers,
            layernorm,
            patch_size: cfg.patch_size,
        })
    }

    /// `[B, 3, H, W]` -> the final-normed **last hidden state**,
    /// `[B, 1 + (H/p)*(W/p), hidden_size]`.
    ///
    /// The CLS token is kept: `Hunyuan3Dv2Conditioning` feeds
    /// `clip_vision_output.last_hidden_state` straight through
    /// (`comfy_extras/nodes_hunyuan3d.py:47-52`), and
    /// `ClipVisionModel.encode_image` fills that from `Dinov2Model.forward`'s
    /// first return value, i.e. `layernorm(x)` including token 0.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (_, channels, height, width) = pixel_values.dims4()?;
        ensure!(channels == 3, "expected 3 input channels, got {channels}");
        ensure!(
            height.is_multiple_of(self.patch_size) && width.is_multiple_of(self.patch_size),
            "input {height}x{width} is not a multiple of the {}px patch size",
            self.patch_size
        );

        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(self.layernorm.forward(&xs)?)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Paint's pinned BitImageProcessor: shortest edge 256 with Pillow BICUBIC,
/// centered 224 crop, then ImageNet rescaling/normalization. The caller supplies
/// the already composed RGB appearance image; this stage never frames a mesh
/// silhouette or performs background removal.
pub fn preprocess_paint(image: &image::RgbImage) -> Result<Tensor> {
    let shortest = image.width().min(image.height());
    ensure!(shortest > 0, "paint appearance image is empty");
    let width = u32::try_from(u64::from(image.width()) * 256 / u64::from(shortest))?;
    let height = u32::try_from(u64::from(image.height()) * 256 / u64::from(shortest))?;
    let resized = crate::pillow_resize::resize(
        image,
        width,
        height,
        crate::pillow_resize::Filter::Bicubic,
        &mut || Ok(()),
    )?;
    let crop = image::imageops::crop_imm(&resized, (width - 224) / 2, (height - 224) / 2, 224, 224)
        .to_image();
    let mut pixels = vec![0.; 3 * 224 * 224];
    for (index, pixel) in crop.pixels().enumerate() {
        for channel in 0..3 {
            pixels[channel * 224 * 224 + index] =
                (f32::from(pixel[channel]) / 255. - IMAGE_MEAN[channel]) / IMAGE_STD[channel];
        }
    }
    Ok(Tensor::from_vec(pixels, (1, 3, 224, 224), &Device::Cpu)?)
}

/// Letterbox to a square of side `letterbox`, resize to `encoder`, and
/// normalize — the tensor [`Dinov2Model::forward`] wants, as
/// `[1, 3, encoder, encoder]`.
///
/// **The two sizes are different numbers and both come from the checkpoint's
/// `config.yaml`.** `image_processor.params.size` is the letterbox square and
/// `conditioner.params.main_image_encoder.kwargs.image_size` is what the
/// tower receives: 512 and 518 on the 1.1B tiers, 1022 and 1022 on the mini
/// tier. Upstream's `ImageEncoder.__init__`
/// (`hy3dgen/shapegen/models/conditioner.py`) is where the second step lives —
/// a `transforms.Resize(image_size, BILINEAR, antialias=True)` followed by
/// `CenterCrop(image_size)`, applied after the letterbox. ComfyUI reaches the
/// same 518 from the other direction (`comfy/clip_vision.py:38`, `:68`, over
/// `comfy/image_encoders/dino2_giant.json`'s `"image_size": 518`).
///
/// Conflating them is not a quality question: 512 is not a multiple of the
/// 14px patch size, so [`Dinov2Model::forward`] refuses it outright. The
/// centre crop is a no-op here — the letterbox already produced a square, so
/// the resize lands exactly on `encoder` in both axes — and the second resize
/// is skipped entirely when the two sizes agree, so the mini tier keeps its
/// existing single-resample bytes.
///
/// Mirrors `ImageProcessorV2.recenter` + `load_image`
/// (`hy3dgen/shapegen/preprocessors.py:31-84`): the opaque content is cropped
/// to its alpha bounding box, scaled so its longer side is
/// `int(max(H, W) * (1 - border_ratio))`, and centred on a **white** square of
/// side `max(H, W)`. Nothing is cropped away — a centre crop would eat the
/// silhouette this model is being asked to reconstruct.
///
/// Upstream then normalizes twice and cancels itself out: `array_to_tensor`
/// maps to `[-1, 1]`, and `ImageEncoder.forward` immediately undoes it with
/// `(image - low) / (high - low)` for `value_range = (-1, 1)` before
/// `Normalize(mean, std)`. The net transform is `(x / 255 - mean) / std`,
/// which is also exactly what ComfyUI's `clip_preprocess` does, so that is
/// what is implemented here.
///
/// Two filter substitutions, both deliberate and both sub-quantization-noise
/// on a photograph:
///   * the content resize is `image`'s `Triangle` (support scaled by the
///     downsampling ratio) where upstream has `cv2.INTER_AREA`;
///   * the letterbox resize is `CatmullRom` (Keys cubic, `a = -0.5`) where
///     upstream has `cv2.INTER_CUBIC` (`a = -0.75`).
///
/// The encoder resize is `Triangle`, matching torchvision's `BILINEAR`. It is
/// an upscale on every shipped tier (512 to 518), so `antialias=True` has
/// nothing to do and the two agree.
///
/// An image with no alpha channel is treated as fully opaque, which makes the
/// bounding box the whole frame and the operation a plain letterbox.
pub fn preprocess(
    image: &DynamicImage,
    letterbox: u32,
    encoder: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    ensure!(letterbox > 0, "letterbox size must be positive");
    ensure!(encoder > 0, "encoder size must be positive");
    let square = letterbox_square(image, BORDER_RATIO)?;
    let resized = image::imageops::resize(
        &square,
        letterbox,
        letterbox,
        image::imageops::FilterType::CatmullRom,
    );
    // Skipped when the sizes agree: an identity resample still filters, and
    // the mini tier's bytes must not move.
    let resized = if encoder == letterbox {
        resized
    } else {
        image::imageops::resize(
            &resized,
            encoder,
            encoder,
            image::imageops::FilterType::Triangle,
        )
    };

    let side = encoder as usize;
    let plane = side * side;
    let mut planar = vec![0.0_f32; 3 * plane];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = y as usize * side + x as usize;
        for (channel, raw) in pixel.0.iter().enumerate() {
            planar[channel * plane + offset] =
                (*raw as f32 / 255.0 - IMAGE_MEAN[channel]) / IMAGE_STD[channel];
        }
    }
    Ok(Tensor::from_vec(planar, (1, 3, side, side), device)?.to_dtype(dtype)?)
}

/// `ImageProcessorV2.recenter`, returning the composited RGB square.
fn letterbox_square(image: &DynamicImage, border_ratio: f32) -> Result<image::RgbImage> {
    ensure!(
        (0.0..1.0).contains(&border_ratio),
        "border_ratio must be in [0, 1), got {border_ratio}"
    );
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    ensure!(width > 0 && height > 0, "source image is empty");
    let opaque = !image.color().has_alpha();

    // Bounding box of the non-zero mask. Upstream then slices
    // `image[x_min:x_max, y_min:y_max]`, i.e. *exclusive* of the maxima, so
    // the extents below are `max - min` and not `max - min + 1`.
    let alpha_at = |x: u32, y: u32| -> u8 {
        if opaque {
            255
        } else {
            rgba.get_pixel(x, y).0[3]
        }
    };
    let (mut row_min, mut row_max) = (u32::MAX, 0_u32);
    let (mut column_min, mut column_max) = (u32::MAX, 0_u32);
    for y in 0..height {
        for x in 0..width {
            if alpha_at(x, y) != 0 {
                row_min = row_min.min(y);
                row_max = row_max.max(y);
                column_min = column_min.min(x);
                column_max = column_max.max(x);
            }
        }
    }
    if row_min == u32::MAX {
        bail!("input image is empty: every pixel is fully transparent");
    }
    let content_height = row_max - row_min;
    let content_width = column_max - column_min;
    if content_height == 0 || content_width == 0 {
        bail!("input image is empty: opaque content is {content_width}x{content_height}");
    }

    let side = width.max(height);
    let desired = (side as f32 * (1.0 - border_ratio)) as u32;
    let scale = desired as f32 / content_height.max(content_width) as f32;
    let scaled_height = (content_height as f32 * scale) as u32;
    let scaled_width = (content_width as f32 * scale) as u32;
    ensure!(
        scaled_height > 0 && scaled_width > 0,
        "letterboxed content collapsed to {scaled_width}x{scaled_height}"
    );

    let cropped =
        image::imageops::crop_imm(&rgba, column_min, row_min, content_width, content_height)
            .to_image();
    let scaled = image::imageops::resize(
        &cropped,
        scaled_width,
        scaled_height,
        image::imageops::FilterType::Triangle,
    );

    // `result` starts as zeros and is composited over a white background, so
    // every pixel the content does not cover ends up white.
    let mut canvas = image::RgbImage::from_pixel(side, side, image::Rgb([255, 255, 255]));
    let top = (side - scaled_height) / 2;
    let left = (side - scaled_width) / 2;
    for (x, y, pixel) in scaled.enumerate_pixels() {
        let alpha = pixel.0[3] as f32 / 255.0;
        let target = canvas.get_pixel_mut(left + x, top + y);
        for (channel, value) in target.0.iter_mut().enumerate() {
            let blended = pixel.0[channel] as f32 * alpha + 255.0 * (1.0 - alpha);
            // numpy's `clip(0, 255).astype(np.uint8)` truncates.
            *value = blended.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(canvas)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The giant config is a contract with the checkpoint, not a preference:
    /// every number here is load-bearing for a key or a shape.
    #[test]
    fn giant_config_matches_upstream_json() {
        let cfg = Dinov2Config::giant();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.image_size, 518);
        assert_eq!(cfg.layer_norm_eps, 1e-6);
        assert!(cfg.use_swiglu_ffn);
        assert!(cfg.qkv_bias);
        assert_eq!(cfg.head_dim(), 64);
        // 37 x 37 + 1 == the checkpoint's [1, 1370, 1536].
        assert_eq!(cfg.patch_grid(), 37);
        assert_eq!(cfg.stored_tokens(), 1370);
        // `mlp.weights_in.weight` is [8192, 1536], `weights_out` is [1536, 4096].
        assert_eq!(cfg.swiglu_hidden(), 4096);
    }

    /// `Dinov2Model::new` must ask for exactly the names in the checkpoint. The
    /// synthetic `VarBuilder` below is built from a literal key list, so a
    /// renamed or dropped path fails to load rather than silently reading
    /// zeros.
    fn synthetic_config() -> Dinov2Config {
        Dinov2Config {
            position_interpolation: PositionInterpolation::ComfyScale,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            patch_size: 2,
            image_size: 8,
            layer_norm_eps: 1e-6,
            use_swiglu_ffn: true,
            qkv_bias: true,
            mlp_ratio: 4,
        }
    }

    #[test]
    fn paint_position_resize_matches_installed_transformers_oracle() {
        let mut tensors = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-dino-position.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let expected = tensors.remove("expected").unwrap();
        let mut cfg = Dinov2Config::paint_giant();
        cfg.hidden_size = 4;
        cfg.image_size = 6;
        cfg.patch_size = 2;
        for (name, shape) in [
            ("cls_token", vec![1, 1, 4]),
            ("patch_embeddings.projection.weight", vec![4, 3, 2, 2]),
            ("patch_embeddings.projection.bias", vec![4]),
        ] {
            tensors.insert(
                name.into(),
                Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let actual = Embeddings::new(&cfg, vb)
            .unwrap()
            .interpolate_pos_encoding(10, 8, DType::F32, &Device::Cpu)
            .unwrap();
        let error = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 3e-6, "paint position resize max error {error}");
    }

    #[test]
    fn paint_preprocessing_matches_actual_dino_processor() {
        let tensors = candle_core::safetensors::load_buffer(
            include_bytes!(
                "../../../../tests/fixtures/hunyuan3d/paint-dino-preprocess.safetensors"
            ),
            &Device::Cpu,
        )
        .unwrap();
        let source = image::RgbImage::from_raw(
            17,
            11,
            tensors["source"]
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
        )
        .unwrap();
        let actual = preprocess_paint(&source).unwrap();
        let delta = (actual - &tensors["expected"])
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(delta < 1e-6, "paint preprocessing error {delta}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires retained pretrained paint DINO weights and CUDA oracle"]
    fn pretrained_paint_dino_matches_tencent() -> Result<()> {
        let checkpoint = std::env::var("MOLD_PAINT_DINO_CHECKPOINT")?;
        let fixture = std::env::var("MOLD_PAINT_DINO_ORACLE")?;
        let output = std::path::PathBuf::from(std::env::var("MOLD_PAINT_DINO_RESULT")?);
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(fixture, &device)?;
        for (dtype, key, max_tolerance, rms_tolerance) in [
            (DType::F32, "expected_f32", 0.001, 0.00005),
            (DType::F16, "expected_f16", 0.2, 0.03),
        ] {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(&checkpoint),
                dtype,
                &device,
                "paint DINO qualification",
                &crate::progress::ProgressReporter::default(),
            )?;
            let model = Dinov2Model::new(&Dinov2Config::paint_giant(), vb)?;
            let actual = model.forward(&tensors["pixels"].to_dtype(dtype)?)?;
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("actual".to_string(), actual.clone())]),
                output.join(format!("{key}.safetensors")),
            )?;
            let delta = (actual.to_dtype(DType::F32)? - tensors[key].to_dtype(DType::F32)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            ensure!(delta.iter().all(|x| x.is_finite()), "nonfinite DINO output");
            let max = delta.iter().map(|x| x.abs()).fold(0., f32::max);
            let rms = (delta.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                / delta.len() as f64)
                .sqrt();
            eprintln!("paint DINO {dtype:?}: max_abs={max}, rms={rms}");
            ensure!(
                max < max_tolerance && rms < rms_tolerance,
                "paint DINO {dtype:?} diverges"
            );
        }
        Ok(())
    }

    /// Deterministic, small, and zero-mean-ish so a forward pass exercises
    /// real arithmetic instead of collapsing to zeros.
    fn deterministic(shape: &[usize], seed: &mut u64) -> Tensor {
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| {
                *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
            })
            .collect();
        Tensor::from_vec(values, shape, &Device::Cpu).expect("cpu tensor from a known shape")
    }

    fn synthetic_weights(cfg: &Dinov2Config) -> HashMap<String, Tensor> {
        let mut seed = 0x5eed_1234_u64;
        let mut weights = HashMap::new();
        let mut put = |name: String, shape: &[usize], seed: &mut u64| {
            weights.insert(name, deterministic(shape, seed));
        };
        let size = cfg.hidden_size;

        put("embeddings.cls_token".into(), &[1, 1, size], &mut seed);
        put(
            "embeddings.position_embeddings".into(),
            &[1, cfg.stored_tokens(), size],
            &mut seed,
        );
        put(
            "embeddings.patch_embeddings.projection.weight".into(),
            &[size, 3, cfg.patch_size, cfg.patch_size],
            &mut seed,
        );
        put(
            "embeddings.patch_embeddings.projection.bias".into(),
            &[size],
            &mut seed,
        );

        let hidden = cfg.swiglu_hidden();
        for layer in 0..cfg.num_hidden_layers {
            let p = format!("encoder.layer.{layer}");
            for norm in ["norm1", "norm2"] {
                put(format!("{p}.{norm}.weight"), &[size], &mut seed);
                put(format!("{p}.{norm}.bias"), &[size], &mut seed);
            }
            for projection in ["query", "key", "value"] {
                let base = format!("{p}.attention.attention.{projection}");
                put(format!("{base}.weight"), &[size, size], &mut seed);
                put(format!("{base}.bias"), &[size], &mut seed);
            }
            put(
                format!("{p}.attention.output.dense.weight"),
                &[size, size],
                &mut seed,
            );
            put(
                format!("{p}.attention.output.dense.bias"),
                &[size],
                &mut seed,
            );
            put(format!("{p}.layer_scale1.lambda1"), &[size], &mut seed);
            put(format!("{p}.layer_scale2.lambda1"), &[size], &mut seed);
            put(
                format!("{p}.mlp.weights_in.weight"),
                &[2 * hidden, size],
                &mut seed,
            );
            put(format!("{p}.mlp.weights_in.bias"), &[2 * hidden], &mut seed);
            put(
                format!("{p}.mlp.weights_out.weight"),
                &[size, hidden],
                &mut seed,
            );
            put(format!("{p}.mlp.weights_out.bias"), &[size], &mut seed);
        }
        put("layernorm.weight".into(), &[size], &mut seed);
        put("layernorm.bias".into(), &[size], &mut seed);
        weights
    }

    fn synthetic_model(cfg: &Dinov2Config) -> Dinov2Model {
        synthetic_model_on(cfg, &Device::Cpu, DType::F32)
    }

    /// The weights are authored on the CPU in F32; `VarBuilder::from_tensors`
    /// casts and moves each one as the model asks for it.
    fn synthetic_model_on(cfg: &Dinov2Config, device: &Device, dtype: DType) -> Dinov2Model {
        let vb = VarBuilder::from_tensors(synthetic_weights(cfg), dtype, device);
        Dinov2Model::new(cfg, vb).expect("synthetic weights cover every requested key")
    }

    #[test]
    fn forward_at_the_stored_resolution_keeps_the_cls_token() {
        let cfg = synthetic_config();
        let model = synthetic_model(&cfg);
        let mut seed = 99;
        let pixels = deterministic(&[1, 3, cfg.image_size, cfg.image_size], &mut seed);

        let out = model.forward(&pixels).expect("forward");
        // 8/2 = 4 patches per side, plus the CLS token.
        assert_eq!(out.dims(), &[1, 1 + 16, cfg.hidden_size]);

        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "forward produced a non-finite value"
        );
        // A layer-normed output cannot be uniformly zero unless the weights
        // are, which they are not.
        assert!(values.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn forward_off_resolution_interpolates_the_position_grid() {
        let cfg = synthetic_config();
        let model = synthetic_model(&cfg);
        let mut seed = 7;
        // 16x16 -> 8x8 patches, against a stored 4x4 grid.
        let pixels = deterministic(&[2, 3, 16, 16], &mut seed);

        let out = model.forward(&pixels).expect("forward");
        assert_eq!(out.dims(), &[2, 1 + 64, cfg.hidden_size]);
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// The vision tower runs in F16 on Metal, at an off-resolution input so
    /// `interpolate_pos_encoding` executes against a Metal tensor rather than
    /// taking the equal-length short circuit. That path drops to the CPU in
    /// F32 to resample and must come back on the right device and dtype; a
    /// CPU-only test cannot tell the two apart.
    #[cfg(feature = "metal")]
    #[test]
    fn synthetic_dinov2_forward_runs_on_metal_in_f16() {
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        let cfg = synthetic_config();
        let model = synthetic_model_on(&cfg, &metal, DType::F16);
        let mut seed = 0x5EED_FEED;
        // 16x16 -> 8x8 patches against a stored 4x4 grid: off-resolution.
        let pixels = deterministic(&[2, 3, 16, 16], &mut seed)
            .to_device(&metal)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out = model.forward(&pixels).expect("forward");
        assert_eq!(out.dims(), &[2, 1 + 64, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::F16);
        assert!(out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn synthetic_dinov2_forward_runs_on_cuda_in_f16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        let cfg = synthetic_config();
        let model = synthetic_model_on(&cfg, &cuda, DType::F16);
        let mut seed = 0x5EED_FEEE;
        let pixels = deterministic(&[2, 3, 16, 16], &mut seed)
            .to_device(&cuda)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out = model.forward(&pixels).expect("forward");
        assert_eq!(out.dims(), &[2, 1 + 64, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::F16);
        assert!(out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn forward_rejects_a_ragged_input() {
        let cfg = synthetic_config();
        let model = synthetic_model(&cfg);
        let mut seed = 3;
        let pixels = deterministic(&[1, 3, 9, 8], &mut seed);
        assert!(model.forward(&pixels).is_err());
    }

    #[test]
    fn position_interpolation_has_the_target_shape() {
        let cfg = Dinov2Config::giant();
        let grid = cfg.patch_grid();
        let channels = 4;
        let source = vec![0.25_f32; grid * grid * channels];
        // The mini-turbo checkpoint conditions at 1022px == 73 patches.
        let rows = 1022 / cfg.patch_size;
        let columns = rows;
        assert_eq!(rows, 73);

        let out = interpolate_patch_positions(
            &source,
            grid,
            channels,
            rows,
            columns,
            PositionInterpolation::ComfyScale,
        );
        assert_eq!(out.len(), rows * columns * channels);
        // A constant field must survive: the four cubic weights sum to one.
        for value in &out {
            assert!(
                (value - 0.25).abs() < 1e-5,
                "constant grid drifted to {value}"
            );
        }
    }

    /// A golden check against an independent transcription of PyTorch's
    /// `upsample_bicubic2d` (`aten/src/ATen/native/UpSample.h`:
    /// `get_cubic_upsample_coefficients` with `A = -0.75`,
    /// `area_pixel_compute_source_index` with `align_corners=false`,
    /// `upsample_get_value_bounded`'s border clamp), evaluated as the full
    /// non-separable 4x4 tensor product on a 4x4 -> 6x5 resample of
    /// `grid[i][j] = 4i + j`.
    ///
    /// This pins four things a shape assertion cannot: the cubic coefficient,
    /// the `(dst + 0.5) / scale - 0.5` source mapping, the `+ 0.1` inside that
    /// scale, and the row/column axis order of the two separable passes (the
    /// grid is deliberately not symmetric, and the output is not square).
    /// Note the first sample is negative: Keys' cubic with `a = -0.75`
    /// overshoots, so it is not interchangeable with the antialiased `a = -0.5`
    /// resize used elsewhere in this crate, and a bilinear stand-in cannot
    /// produce these numbers either.
    #[test]
    fn position_interpolation_matches_the_pytorch_bicubic_reference() {
        const GRID: usize = 4;
        const ROWS: usize = 6;
        const COLUMNS: usize = 5;
        #[rustfmt::skip]
        // f64 so the transcribed samples keep their printed precision.
        const GOLDEN: [f64; ROWS * COLUMNS] = [
            -0.4182968,  0.2308243,  1.1166090,  1.9723543,  2.6668614,
             1.4995455,  2.1486666,  3.0344512,  3.8901965,  4.5847036,
             4.6660093,  5.3151304,  6.2009151,  7.0566603,  7.7511675,
             6.9236432,  7.5727643,  8.4585490,  9.3142942, 10.0088014,
            10.1224489, 10.7715700, 11.6573547, 12.5131000, 13.2076071,
            12.1907972, 12.8399183, 13.7257029, 14.5814482, 15.2759553,
        ];

        let source: Vec<f32> = (0..GRID * GRID).map(|index| index as f32).collect();
        let out = interpolate_patch_positions(
            &source,
            GRID,
            1,
            ROWS,
            COLUMNS,
            PositionInterpolation::ComfyScale,
        );

        assert_eq!(out.len(), GOLDEN.len());
        for (index, (actual, expected)) in out.iter().zip(GOLDEN.iter()).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() < 2e-5,
                "sample {index}: expected {expected}, got {actual}"
            );
        }
    }

    /// The channel axis must ride along untouched: interleaving two grids into
    /// one two-channel buffer has to give the same answer as resampling each
    /// on its own.
    #[test]
    fn position_interpolation_keeps_channels_independent() {
        const GRID: usize = 5;
        const ROWS: usize = 9;
        const COLUMNS: usize = 7;

        let first: Vec<f32> = (0..GRID * GRID).map(|i| i as f32).collect();
        let second: Vec<f32> = (0..GRID * GRID).map(|i| (i as f32) * -0.5 + 3.0).collect();
        let interleaved: Vec<f32> = first
            .iter()
            .zip(second.iter())
            .flat_map(|(a, b)| [*a, *b])
            .collect();

        let a = interpolate_patch_positions(
            &first,
            GRID,
            1,
            ROWS,
            COLUMNS,
            PositionInterpolation::ComfyScale,
        );
        let b = interpolate_patch_positions(
            &second,
            GRID,
            1,
            ROWS,
            COLUMNS,
            PositionInterpolation::ComfyScale,
        );
        let both = interpolate_patch_positions(
            &interleaved,
            GRID,
            2,
            ROWS,
            COLUMNS,
            PositionInterpolation::ComfyScale,
        );

        for index in 0..ROWS * COLUMNS {
            assert!((both[index * 2] - a[index]).abs() < 1e-5);
            assert!((both[index * 2 + 1] - b[index]).abs() < 1e-5);
        }
    }

    #[test]
    fn preprocess_normalizes_a_white_image_to_the_channel_constants() {
        // Fully opaque and uniformly white: the letterbox border is white too,
        // so every output sample is (1 - mean) / std.
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            48,
            image::Rgb([255, 255, 255]),
        ));
        let out = preprocess(&image, 28, 28, &Device::Cpu, DType::F32).expect("preprocess");
        assert_eq!(out.dims(), &[1, 3, 28, 28]);

        for channel in 0..3 {
            let expected = (1.0 - IMAGE_MEAN[channel]) / IMAGE_STD[channel];
            let plane = out
                .i((0, channel))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for value in plane {
                assert!(
                    (value - expected).abs() < 1e-4,
                    "channel {channel}: expected {expected}, got {value}"
                );
            }
        }
    }

    /// The border must be background and the subject must survive: this is the
    /// property a centre crop would violate.
    #[test]
    fn preprocess_letterboxes_instead_of_cropping() {
        // A wide image whose opaque content is a red block; the transparent
        // remainder must not shift the subject off-centre.
        let mut rgba = image::RgbaImage::from_pixel(120, 60, image::Rgba([0, 0, 0, 0]));
        for y in 10..50 {
            for x in 20..100 {
                rgba.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
            }
        }
        let image = DynamicImage::ImageRgba8(rgba);
        let out = preprocess(&image, 56, 56, &Device::Cpu, DType::F32).expect("preprocess");
        assert_eq!(out.dims(), &[1, 3, 56, 56]);

        let red = out.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
        let green = out.i((0, 1)).unwrap().to_vec2::<f32>().unwrap();
        let white_red = (1.0 - IMAGE_MEAN[0]) / IMAGE_STD[0];
        let subject_green = (0.0 - IMAGE_MEAN[1]) / IMAGE_STD[1];

        // Corners are border: white.
        for (row, column) in [(0, 0), (0, 55), (55, 0), (55, 55)] {
            assert!(
                (red[row][column] - white_red).abs() < 1e-3,
                "corner ({row},{column}) is not background: {}",
                red[row][column]
            );
        }
        // The centre is the subject: red, so the green channel is at 0/255.
        assert!(
            (green[28][28] - subject_green).abs() < 1e-3,
            "centre is not the subject: {}",
            green[28][28]
        );
        // And the subject is genuinely inset by the border ratio rather than
        // filling the frame: 15% of 120 is 18px, so column 4 of 56 (~8.6px on
        // the 120px canvas) is still background.
        assert!(
            (red[28][2] - white_red).abs() < 1e-3,
            "no border was left on the long axis: {}",
            red[28][2]
        );
    }

    #[test]
    fn preprocess_rejects_a_fully_transparent_image() {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            32,
            32,
            image::Rgba([1, 2, 3, 0]),
        ));
        assert!(preprocess(&image, 28, 28, &Device::Cpu, DType::F32).is_err());
    }

    /// The letterbox size and the encoder size are two different numbers on
    /// the 1.1B tiers, and it is the second one the tower must receive.
    ///
    /// `hunyuan3d-dit-v2-0/config.yaml` sets `image_processor.params.size` to
    /// 512 and `conditioner.params.main_image_encoder.kwargs.image_size` to
    /// 518; `ImageEncoder.__init__` in `hy3dgen/shapegen/models/conditioner.py`
    /// applies a `Resize(518, BILINEAR, antialias=True)` plus
    /// `CenterCrop(518)` after the letterbox. ComfyUI lands on the same 518
    /// (`comfy/clip_vision.py:38`, `:68`, over
    /// `comfy/image_encoders/dino2_giant.json`'s `"image_size": 518`).
    ///
    /// Handing DINOv2 the 512 letterbox instead is not a subtle quality loss:
    /// 512 is not a multiple of the 14px patch size, so `Dinov2Model::forward`
    /// refuses it and image encoding fails outright.
    #[test]
    fn preprocess_resizes_the_letterbox_to_the_encoder_size() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            48,
            image::Rgb([255, 255, 255]),
        ));
        let out = preprocess(&image, 512, 518, &Device::Cpu, DType::F32).expect("preprocess");
        assert_eq!(out.dims(), &[1, 3, 518, 518]);
        assert_eq!(518 % Dinov2Config::giant().patch_size, 0);

        // A uniformly white source stays white through both resizes, so the
        // channel constants still pin the normalization.
        for channel in 0..3 {
            let expected = (1.0 - IMAGE_MEAN[channel]) / IMAGE_STD[channel];
            let plane = out
                .i((0, channel))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for value in plane {
                assert!(
                    (value - expected).abs() < 1e-4,
                    "channel {channel}: expected {expected}, got {value}"
                );
            }
        }
    }

    /// The mini tier letterboxes and encodes at the same 1022, so the second
    /// resize must not run at all — an identity resample is still a resample,
    /// and it would change the bytes the tower sees on the tier that was
    /// already correct.
    #[test]
    fn preprocess_skips_the_second_resize_when_sizes_agree() {
        let mut rgba = image::RgbaImage::from_pixel(120, 60, image::Rgba([0, 0, 0, 0]));
        for y in 10..50 {
            for x in 20..100 {
                rgba.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
            }
        }
        let image = DynamicImage::ImageRgba8(rgba);

        let same = preprocess(&image, 56, 56, &Device::Cpu, DType::F32).expect("preprocess");
        assert_eq!(same.dims(), &[1, 3, 56, 56]);

        // Bit-identical to the single-resize path this test's fixture was
        // written against.
        let reference = preprocess_letterbox_only(&image, 56);
        let a = same.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (got, want)) in a.iter().zip(&reference).enumerate() {
            assert_eq!(
                got, want,
                "sample {index}: the equal-size path must not resample twice"
            );
        }
    }

    /// The letterbox-and-normalize path with no second resize, transcribed so
    /// the test above compares against something independent of `preprocess`'s
    /// own branch.
    fn preprocess_letterbox_only(image: &DynamicImage, target: u32) -> Vec<f32> {
        let square = letterbox_square(image, BORDER_RATIO).expect("letterbox");
        let resized = image::imageops::resize(
            &square,
            target,
            target,
            image::imageops::FilterType::CatmullRom,
        );
        let side = target as usize;
        let plane = side * side;
        let mut planar = vec![0.0_f32; 3 * plane];
        for (x, y, pixel) in resized.enumerate_pixels() {
            let offset = y as usize * side + x as usize;
            for (channel, raw) in pixel.0.iter().enumerate() {
                planar[channel * plane + offset] =
                    (*raw as f32 / 255.0 - IMAGE_MEAN[channel]) / IMAGE_STD[channel];
            }
        }
        planar
    }
}
