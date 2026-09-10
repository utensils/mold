//! OpenCLIP CLIP-ViT-H-14 vision tower — the image encoder classic IP-Adapter
//! conditions on.
//!
//! The checkpoint is `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`'s vision half, as
//! republished by `h94/IP-Adapter` under `models/image_encoder/`. It is stored
//! in Hugging Face's `CLIPVisionModelWithProjection` layout, so this file is a
//! port of `transformers` 4.57.3 `models/clip/modeling_clip.py`:
//!
//! - `:715-760` `CLIPVisionTransformer` (embeddings, `pre_layrnorm`, encoder,
//!   CLS, `post_layernorm`)
//! - `:131-201` `CLIPVisionEmbeddings`
//! - `:368-413` `CLIPEncoderLayer`
//! - `:254-350` `eager_attention_forward` / `CLIPAttention`
//! - `:353-365` `CLIPMLP`
//! - `:1098-1158` `CLIPVisionModelWithProjection` (the bias-free
//!   `visual_projection` on top)
//!
//! ComfyUI's `comfy/clip_vision_config_h.json` is the same architecture read
//! back as a config, and `comfy/clip_vision.py:88-112`
//! (`convert_to_transformers`) is the authority for the tensor names an
//! OpenCLIP-native export maps onto — `ln_pre` -> `vision_model.pre_layrnorm`,
//! `ln_post` -> `vision_model.post_layernorm`, `proj^T` ->
//! `visual_projection.weight`.
//!
//! ## Why this is a port and not a call into the candle fork
//!
//! `candle-transformers`' `models/clip/vision_model.rs` has the right shape,
//! but its `Activation` enum (`models/clip/text_model.rs:16-26`) carries
//! exactly one variant, `QuickGelu`. ViT-H/14's config says `hidden_act:
//! "gelu"`, which `transformers` resolves to `GELUActivation`
//! (`activations.py:66-85`, `ACT2CLS` at `:315`) — `nn.functional.gelu`, the
//! **exact erf** form, not the tanh approximation and not QuickGelu. Three
//! activations, three different functions, and picking the wrong one produces
//! a plausible embedding that is quietly wrong everywhere. The fork is
//! upstream-tracking and stays untouched.
//!
//! Two smaller divergences from that same file are deliberate here and both
//! follow HF rather than the fork: the attention scale is applied to the score
//! matrix *after* the QK matmul (`modeling_clip.py:265`, the fork pre-scales
//! the query), and the softmax is widened to f32 and cast back (`:268`).
//! Neither is observable in f32; both are, in f16.
//!
//! ## What leaves the module
//!
//! [`OpenClipVisionOutput`] carries the two things IP-Adapter reads, and
//! nothing else:
//!
//! - `image_embeds` — the projected CLS token, `[batch, 1024]`. This is what
//!   classic IP-Adapter conditions on (`diffusers`
//!   `pipeline_stable_diffusion.py:532`,
//!   `self.image_encoder(image).image_embeds`).
//! - `penultimate_hidden_states` — the encoder's output after layer 30 of 32,
//!   `[batch, 257, 1280]`, un-normalized. IP-Adapter Plus takes this instead
//!   (`pipeline_stable_diffusion.py:522`, `.hidden_states[-2]`); wiring it now
//!   costs one `clone` of a tensor the loop already holds.
//!
//! The tower is ~632 M parameters, so it follows the crate's drop-and-reload
//! rule: build it, encode, drop it. Nothing here caches.

// The IP-Adapter pipeline that consumes this module lands separately; until
// that consumer exists every item here is reachable only from tests, so the
// dead-code lint would otherwise force either a premature `pub` surface or a
// stub caller. This mirrors `eva_clip_vision`'s note for the same reason.
#![allow(dead_code)]

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

/// Square input edge in pixels (`clip_vision_config_h.json`, `image_size`).
pub(crate) const IMAGE_SIZE: usize = 224;
/// Patch edge in pixels (`patch_size`).
pub(crate) const PATCH_SIZE: usize = 14;
/// Residual width (`hidden_size`).
pub(crate) const EMBED_DIM: usize = 1280;
/// MLP width (`intermediate_size`).
pub(crate) const INTERMEDIATE_SIZE: usize = 5120;
/// Transformer depth (`num_hidden_layers`).
pub(crate) const NUM_LAYERS: usize = 32;
/// `num_attention_heads`.
pub(crate) const NUM_HEADS: usize = 16;
/// CLIP joint-embedding width, the `visual_projection` output
/// (`projection_dim`).
pub(crate) const PROJECTION_DIM: usize = 1024;
/// `layer_norm_eps`. Every LayerNorm in the tower — the two around the encoder
/// and the two per layer — shares it (`modeling_clip.py:722`, `:724`, `:373`,
/// `:375`).
const LAYER_NORM_EPS: f64 = 1e-5;

/// Everything IP-Adapter reads off the tower.
#[derive(Debug, Clone)]
pub(crate) struct OpenClipVisionOutput {
    /// `[batch, 1024]` — `visual_projection(post_layernorm(hidden[:, 0]))`.
    pub(crate) image_embeds: Tensor,
    /// `[batch, 257, 1280]` — the residual stream entering the LAST encoder
    /// layer, i.e. `hidden_states[-2]`, before `post_layernorm`.
    pub(crate) penultimate_hidden_states: Tensor,
}

/// `CLIPVisionConfig` for this checkpoint, kept as a struct rather than as
/// loose constants for one reason: it lets the tests build a two-layer,
/// eight-wide tower and assert the real shapes at every stage without
/// allocating 632 M parameters. Production has exactly one instance,
/// [`OpenClipVisionConfig::vit_h_14`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenClipVisionConfig {
    pub(crate) embed_dim: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) projection_dim: usize,
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
}

impl OpenClipVisionConfig {
    /// `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, verbatim from
    /// `ComfyUI/comfy/clip_vision_config_h.json` (which is that repo's own
    /// `config.json` `vision_config`).
    pub(crate) const fn vit_h_14() -> Self {
        Self {
            embed_dim: EMBED_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            num_hidden_layers: NUM_LAYERS,
            num_attention_heads: NUM_HEADS,
            projection_dim: PROJECTION_DIM,
            image_size: IMAGE_SIZE,
            patch_size: PATCH_SIZE,
        }
    }

    /// `modeling_clip.py:286` — `hidden_size // num_attention_heads`.
    const fn head_dim(&self) -> usize {
        self.embed_dim / self.num_attention_heads
    }

    /// Patch-grid edge, `image_size // patch_size` (16 for ViT-H/14 at 224).
    const fn grid_size(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// `modeling_clip.py:149-150` — `(image_size // patch_size) ** 2 + 1`, the
    /// `+ 1` being the CLS token. 257 here.
    const fn num_positions(&self) -> usize {
        self.grid_size() * self.grid_size() + 1
    }

    /// The head split has to be exact; a config whose width is not divisible by
    /// its head count reshapes into garbage rather than failing
    /// (`modeling_clip.py:287-291` raises for the same reason).
    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_attention_heads > 0 && self.embed_dim.is_multiple_of(self.num_attention_heads),
            "embed_dim {} is not divisible by num_attention_heads {}",
            self.embed_dim,
            self.num_attention_heads
        );
        ensure!(
            self.patch_size > 0 && self.image_size.is_multiple_of(self.patch_size),
            "image_size {} is not a whole number of {}px patches",
            self.image_size,
            self.patch_size
        );
        Ok(())
    }
}

/// `CLIPVisionEmbeddings` (`modeling_clip.py:131-215`).
///
/// The position embedding is an `nn.Embedding` indexed by a registered
/// `arange` buffer (`:152`), so the lookup is the identity on the weight and
/// mold broadcast-adds the `[num_positions, embed_dim]` matrix directly.
/// `interpolate_pos_encoding` (`:154-193`) is not ported: it exists to run the
/// tower off its trained resolution, and [`OpenClipVisionTower::forward`]
/// refuses anything but `image_size` square, which is the branch upstream takes
/// by default (`:169-170`).
#[derive(Debug)]
struct Embeddings {
    patch_embedding: Conv2d,
    class_embedding: Tensor,
    position_embedding: Tensor,
    embed_dim: usize,
}

impl Embeddings {
    fn new(vb: VarBuilder, config: &OpenClipVisionConfig) -> Result<Self> {
        // `:141-147`: kernel == stride == patch_size, and **no bias**.
        let patch_embedding = Conv2d::new(
            vb.get(
                (config.embed_dim, 3, config.patch_size, config.patch_size),
                "patch_embedding.weight",
            )?,
            None,
            Conv2dConfig {
                stride: config.patch_size,
                ..Default::default()
            },
        );
        Ok(Self {
            patch_embedding,
            // `:139` — an `nn.Parameter` of rank 1, not a `[1, 1, D]` token.
            class_embedding: vb.get(config.embed_dim, "class_embedding")?,
            position_embedding: vb.get(
                (config.num_positions(), config.embed_dim),
                "position_embedding.weight",
            )?,
            embed_dim: config.embed_dim,
        })
    }

    /// `:195-215` — conv, flatten the grid, tokens last, prepend CLS, add the
    /// absolute position embedding.
    fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let batch = pixels.dim(0)?;
        let patches = self
            .patch_embedding
            .forward(pixels)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let class_embeds = self.class_embedding.expand((batch, 1, self.embed_dim))?;
        Ok(Tensor::cat(&[class_embeds, patches], 1)?
            .broadcast_add(&self.position_embedding)?
            .contiguous()?)
    }
}

/// `CLIPAttention` (`modeling_clip.py:278-350`) with `eager_attention_forward`
/// (`:254-275`) inlined. Bidirectional: the vision tower passes neither an
/// attention mask nor a causal mask, so there is no mask term at all.
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(vb: VarBuilder, config: &OpenClipVisionConfig) -> Result<Self> {
        let dim = config.embed_dim;
        // `:296-299`: all four projections are square and all four carry a bias.
        let load = |name: &str| -> Result<Linear> {
            Ok(Linear::new(
                vb.get((dim, dim), &format!("{name}.weight"))?,
                Some(vb.get(dim, &format!("{name}.bias"))?),
            ))
        };
        Ok(Self {
            q_proj: load("q_proj")?,
            k_proj: load("k_proj")?,
            v_proj: load("v_proj")?,
            out_proj: load("out_proj")?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            embed_dim: dim,
            // `:292` — `head_dim ** -0.5`.
            scale: (config.head_dim() as f64).powf(-0.5),
        })
    }

    fn split_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        Ok(xs
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        let q = self.split_heads(&self.q_proj.forward(xs)?)?;
        let k = self.split_heads(&self.k_proj.forward(xs)?)?;
        let v = self.split_heads(&self.v_proj.forward(xs)?)?;

        // `:265` — scale the SCORES, not the query. Algebraically the same,
        // and not the same in f16: the fork's CLIP pre-scales `q`, which
        // rounds once per projected element instead of once per score.
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        // `:268` — softmax in f32 whatever the weights are, then back.
        let weights = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())?;
        let attended =
            weights
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, tokens, self.embed_dim))?;
        Ok(self.out_proj.forward(&attended)?)
    }
}

/// `CLIPMLP` (`modeling_clip.py:353-365`).
///
/// `hidden_act` is `"gelu"`, which is `ACT2FN["gelu"] == GELUActivation`
/// (`activations.py:315`, `:66-85`) — `nn.functional.gelu`, i.e.
/// `x * 0.5 * (1 + erf(x / sqrt(2)))`. candle's `Tensor::gelu` is the *tanh*
/// approximation; the exact one is `gelu_erf`. QuickGelu, which the rest of
/// OpenAI's CLIP family uses and which is the only variant the candle fork's
/// CLIP implements, is a third function again.
#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, config: &OpenClipVisionConfig) -> Result<Self> {
        let (dim, hidden) = (config.embed_dim, config.intermediate_size);
        let load = |name: &str, out: usize, inp: usize| -> Result<Linear> {
            Ok(Linear::new(
                vb.get((out, inp), &format!("{name}.weight"))?,
                Some(vb.get(out, &format!("{name}.bias"))?),
            ))
        };
        Ok(Self {
            fc1: load("fc1", hidden, dim)?,
            fc2: load("fc2", dim, hidden)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.fc2.forward(&self.fc1.forward(xs)?.gelu_erf()?)?)
    }
}

/// `CLIPEncoderLayer` (`modeling_clip.py:368-413`) — pre-norm, two residuals.
#[derive(Debug)]
struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn new(vb: VarBuilder, config: &OpenClipVisionConfig) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(config.embed_dim, vb.pp("layer_norm1"))?,
            self_attn: Attention::new(vb.pp("self_attn"), config)?,
            layer_norm2: layer_norm(config.embed_dim, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), config)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.self_attn.forward(&self.layer_norm1.forward(xs)?)?)?;
        Ok((&xs + self.mlp.forward(&self.layer_norm2.forward(&xs)?)?)?)
    }
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        LAYER_NORM_EPS,
    ))
}

/// The tower.
///
/// Build it from a `VarBuilder` rooted at the **top of the checkpoint**, the
/// way `h94/IP-Adapter`'s `models/image_encoder/model.safetensors` is written:
/// everything under `vision_model.` plus the single top-level
/// `visual_projection.weight`. That is `CLIPVisionModelWithProjection`'s own
/// module tree (`modeling_clip.py:1098-1111`), so no key rewriting happens
/// anywhere in mold.
#[derive(Debug)]
pub(crate) struct OpenClipVisionTower {
    embeddings: Embeddings,
    pre_layrnorm: LayerNorm,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    visual_projection: Linear,
    config: OpenClipVisionConfig,
    device: Device,
    dtype: DType,
}

impl OpenClipVisionTower {
    /// Build from installed, manifest-verified safetensors.
    ///
    /// `dtype` is the tower's WORKING dtype and belongs to the caller, exactly
    /// as it does for [`super::eva_clip_vision::EvaClipVisionTower`]: the
    /// arithmetic is identical either way, but widening a checkpoint stored
    /// narrow costs a full pass over the weights, and candle has no narrow CPU
    /// kernels worth using. The safetensors stay mmap'd — this is a
    /// build-encode-drop tower, so the file-backed pages are the point.
    ///
    /// # Safety
    ///
    /// `VarBuilder::from_mmaped_safetensors` maps the files; the caller must
    /// not have another writer mutating them for the tower's lifetime. Same
    /// contract every other engine in the crate takes on its weights.
    pub(crate) fn from_installed<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
            .context("mapping the CLIP-ViT-H-14 vision tower")?;
        Self::new(vb, OpenClipVisionConfig::vit_h_14(), device)
    }

    pub(crate) fn new(
        vb: VarBuilder,
        config: OpenClipVisionConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let dtype = vb.dtype();
        // `modeling_clip.py:1105-1106`: `CLIPVisionModelWithProjection` keeps
        // the inner transformer at `vision_model`, so the encoder layers live
        // at `vision_model.encoder.layers.N` and the projection sits OUTSIDE
        // that subtree.
        let vision = vb.pp("vision_model");
        let layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::new(vision.pp(format!("encoder.layers.{index}")), &config))
            .collect::<Result<Vec<_>>>()
            .context("failed to build a CLIP-ViT-H-14 encoder layer")?;
        Ok(Self {
            embeddings: Embeddings::new(vision.pp("embeddings"), &config)?,
            // `:722` — HF really does spell it `pre_layrnorm`. The typo is in
            // the published weights, so it is the wire name.
            pre_layrnorm: layer_norm(config.embed_dim, vision.pp("pre_layrnorm"))?,
            layers,
            post_layernorm: layer_norm(config.embed_dim, vision.pp("post_layernorm"))?,
            // `:1108` — `nn.Linear(hidden_size, projection_dim, bias=False)`.
            visual_projection: Linear::new(
                vb.get(
                    (config.projection_dim, config.embed_dim),
                    "visual_projection.weight",
                )?,
                None,
            ),
            config,
            device: device.clone(),
            dtype,
        })
    }

    pub(crate) fn config(&self) -> &OpenClipVisionConfig {
        &self.config
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    /// `pixels` is `[batch, 3, 224, 224]`, already resized, cropped and
    /// normalized by [`super::clip_image_preprocess`].
    pub(crate) fn forward(&self, pixels: &Tensor) -> Result<OpenClipVisionOutput> {
        let edge = self.config.image_size;
        let (batch, channels, height, width) = pixels.dims4()?;
        // `modeling_clip.py:197-200` raises on exactly this, because the
        // position embedding is trained for one grid and silently broadcasts
        // for none.
        ensure!(
            channels == 3 && height == edge && width == edge,
            "CLIP-ViT-H-14 expects [batch, 3, {edge}, {edge}], got \
             [{batch}, {channels}, {height}, {width}]"
        );
        ensure!(
            self.config.num_hidden_layers >= 2,
            "a penultimate hidden state needs at least two encoder layers, this config has {}",
            self.config.num_hidden_layers
        );
        let pixels = pixels.to_dtype(self.dtype)?.to_device(&self.device)?;

        // `:742-743`.
        let mut hidden = self
            .pre_layrnorm
            .forward(&self.embeddings.forward(&pixels)?)?;

        // `hidden_states[-2]` is the output of layer `num_hidden_layers - 2`
        // — HF's `hidden_states` tuple begins with the embedding output, so it
        // has `num_hidden_layers + 1` entries and index `-2` is the LAST
        // layer's INPUT. ComfyUI reaches the same tensor from the other side
        // (`clip_model.py`, `intermediate_output = len(self.layers) - 2`,
        // captured after running that layer). Reading it as the last layer's
        // output instead is a one-layer shift that still type-checks.
        let penultimate_index = self.config.num_hidden_layers - 2;
        let mut penultimate = None;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            if index == penultimate_index {
                penultimate = Some(hidden.clone());
            }
        }
        let penultimate_hidden_states = penultimate.context("the encoder ran no layers")?;

        // `:752-753` — pool the CLS token, THEN normalize it. LayerNorm is
        // per-token, so normalizing the whole sequence first would agree; this
        // way the final norm runs on one token instead of 257.
        let pooled = self.post_layernorm.forward(&hidden.i((.., 0, ..))?)?;
        let image_embeds = self.visual_projection.forward(&pooled)?;

        Ok(OpenClipVisionOutput {
            image_embeds,
            penultimate_hidden_states,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A deterministic, dependency-free value stream. The tests need weights
    /// that are not zero (a zero LayerNorm gain collapses every stage to zeros
    /// and hides a mis-ordered forward) and they must not depend on candle's
    /// global RNG, which other tests in this crate re-seed.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes' 64-bit LCG; only the high bits are used.
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        }

        fn tensor(&mut self, dims: &[usize]) -> Tensor {
            let len: usize = dims.iter().product();
            let values: Vec<f32> = (0..len).map(|_| self.next_f32()).collect();
            Tensor::from_vec(values, dims, &Device::Cpu).unwrap()
        }
    }

    /// Two layers, eight wide, 4x4 images of 2x2 patches: five tokens, four
    /// heads of two. Small enough to build in a test, large enough that every
    /// reshape in the forward is non-trivial.
    fn tiny_config() -> OpenClipVisionConfig {
        OpenClipVisionConfig {
            embed_dim: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            projection_dim: 6,
            image_size: 4,
            patch_size: 2,
        }
    }

    /// Every tensor `OpenClipVisionTower::new` is allowed to ask for, at the
    /// exact names and shapes the published checkpoint uses. Because
    /// `VarBuilder::from_tensors` fails on a missing key, this map IS the
    /// tensor-name contract: a renamed or re-nested load fails here rather
    /// than at whatever hour someone first points the loader at 2.5 GB of
    /// real weights.
    fn weights(config: &OpenClipVisionConfig) -> HashMap<String, Tensor> {
        let mut rng = Lcg(0x5eed_1105);
        let mut map = HashMap::new();
        let (dim, hidden) = (config.embed_dim, config.intermediate_size);
        let mut put = |name: &str, dims: &[usize], rng: &mut Lcg| {
            map.insert(name.to_string(), rng.tensor(dims));
        };
        put(
            "vision_model.embeddings.patch_embedding.weight",
            &[dim, 3, config.patch_size, config.patch_size],
            &mut rng,
        );
        put("vision_model.embeddings.class_embedding", &[dim], &mut rng);
        put(
            "vision_model.embeddings.position_embedding.weight",
            &[config.num_positions(), dim],
            &mut rng,
        );
        for norm in ["pre_layrnorm", "post_layernorm"] {
            put(&format!("vision_model.{norm}.weight"), &[dim], &mut rng);
            put(&format!("vision_model.{norm}.bias"), &[dim], &mut rng);
        }
        for index in 0..config.num_hidden_layers {
            let prefix = format!("vision_model.encoder.layers.{index}");
            for norm in ["layer_norm1", "layer_norm2"] {
                put(&format!("{prefix}.{norm}.weight"), &[dim], &mut rng);
                put(&format!("{prefix}.{norm}.bias"), &[dim], &mut rng);
            }
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                put(
                    &format!("{prefix}.self_attn.{projection}.weight"),
                    &[dim, dim],
                    &mut rng,
                );
                put(
                    &format!("{prefix}.self_attn.{projection}.bias"),
                    &[dim],
                    &mut rng,
                );
            }
            put(
                &format!("{prefix}.mlp.fc1.weight"),
                &[hidden, dim],
                &mut rng,
            );
            put(&format!("{prefix}.mlp.fc1.bias"), &[hidden], &mut rng);
            put(
                &format!("{prefix}.mlp.fc2.weight"),
                &[dim, hidden],
                &mut rng,
            );
            put(&format!("{prefix}.mlp.fc2.bias"), &[dim], &mut rng);
        }
        put(
            "visual_projection.weight",
            &[config.projection_dim, dim],
            &mut rng,
        );
        map
    }

    fn tiny_tower(config: &OpenClipVisionConfig) -> OpenClipVisionTower {
        let vb = VarBuilder::from_tensors(weights(config), DType::F32, &Device::Cpu);
        OpenClipVisionTower::new(vb, *config, &Device::Cpu).unwrap()
    }

    /// The published arithmetic, restated so a "tidy-up" of the constants has
    /// to disagree with a test rather than with a 1.3 GB download.
    #[test]
    fn the_geometry_matches_the_published_config() {
        let config = OpenClipVisionConfig::vit_h_14();
        assert_eq!(config.grid_size(), 16, "224 / 14");
        // 16 x 16 patches plus CLS. The position embedding is [257, 1280] and
        // an off-by-one here fails to load rather than mis-rendering, which is
        // exactly why it is worth pinning.
        assert_eq!(config.num_positions(), 257);
        assert_eq!(config.num_positions(), (IMAGE_SIZE / PATCH_SIZE).pow(2) + 1);
        assert_eq!(config.head_dim(), 80, "1280 / 16");
        assert_eq!(config.head_dim() * config.num_attention_heads, EMBED_DIM);
        // 4x the residual width, unlike ViT-L's 4096/1024.
        assert_eq!(config.intermediate_size, 4 * config.embed_dim);
        config.validate().unwrap();
    }

    /// A width that does not split evenly across the heads reshapes into
    /// nonsense instead of failing, so the config refuses it up front.
    #[test]
    fn an_unsplittable_config_is_refused() {
        let mut config = tiny_config();
        config.num_attention_heads = 3;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("not divisible"), "unexpected error: {error}");

        let mut config = tiny_config();
        config.patch_size = 3;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("whole number"), "unexpected error: {error}");
    }

    /// Shapes at every stage, on the real forward. The tiny config keeps this
    /// hermetic; the ratios it exercises (tokens = grid^2 + 1, heads split,
    /// MLP widening, projection narrowing) are the same ones ViT-H/14 uses.
    #[test]
    fn every_stage_has_the_shape_the_next_one_expects() {
        let config = tiny_config();
        let tower = tiny_tower(&config);
        let mut rng = Lcg(7);
        let pixels = rng.tensor(&[2, 3, config.image_size, config.image_size]);

        // Embeddings: [batch, grid^2 + 1, embed_dim].
        let embedded = tower.embeddings.forward(&pixels).unwrap();
        assert_eq!(embedded.dims(), &[2, config.num_positions(), 8]);
        assert_eq!(config.num_positions(), 5, "2x2 patches plus CLS");

        // The encoder is shape-preserving, layer by layer.
        let normed = tower.pre_layrnorm.forward(&embedded).unwrap();
        assert_eq!(normed.dims(), embedded.dims());
        let attended = tower.layers[0]
            .self_attn
            .forward(&tower.layers[0].layer_norm1.forward(&normed).unwrap())
            .unwrap();
        assert_eq!(attended.dims(), embedded.dims());
        let mlp = tower.layers[0].mlp.forward(&normed).unwrap();
        assert_eq!(mlp.dims(), embedded.dims());
        assert_eq!(
            tower.layers[0].mlp.fc1.forward(&normed).unwrap().dims(),
            &[2, 5, 16],
            "fc1 widens to intermediate_size"
        );

        let output = tower.forward(&pixels).unwrap();
        assert_eq!(
            output.penultimate_hidden_states.dims(),
            &[2, config.num_positions(), 8]
        );
        assert_eq!(
            output.image_embeds.dims(),
            &[2, config.projection_dim],
            "the projection pools the sequence away"
        );
        assert!(output
            .image_embeds
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    /// The real geometry, asserted without building the real tower: 257 tokens
    /// in, 1024 out.
    #[test]
    fn the_published_config_produces_257_tokens_and_a_1024_embedding() {
        let config = OpenClipVisionConfig::vit_h_14();
        assert_eq!(config.num_positions(), 257);
        assert_eq!(config.projection_dim, 1024);
        assert_eq!(config.embed_dim, 1280);
        assert_eq!(config.num_hidden_layers, 32);
        // `hidden_states[-2]` is the 31st layer's output on a 32-layer stack.
        assert_eq!(config.num_hidden_layers - 2, 30);
    }

    /// `penultimate_hidden_states` is the input to the LAST layer, not its
    /// output. Both are correctly shaped, so only a re-run can tell them apart.
    #[test]
    fn the_penultimate_state_is_the_last_layers_input() {
        let config = tiny_config();
        let tower = tiny_tower(&config);
        let mut rng = Lcg(11);
        let pixels = rng.tensor(&[1, 3, config.image_size, config.image_size]);
        let output = tower.forward(&pixels).unwrap();

        // Re-run everything but the final layer by hand.
        let mut hidden = tower
            .pre_layrnorm
            .forward(&tower.embeddings.forward(&pixels).unwrap())
            .unwrap();
        for layer in tower.layers.iter().take(config.num_hidden_layers - 1) {
            hidden = layer.forward(&hidden).unwrap();
        }
        let expected = hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = output
            .penultimate_hidden_states
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, expected, "the tap is off by a layer");

        // ...and it is genuinely not the final state, so the assertion above
        // is not vacuous.
        let final_state = tower
            .layers
            .last()
            .unwrap()
            .forward(&hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(actual, final_state);
    }

    /// `hidden_act: "gelu"` is the exact erf form. candle's `gelu` is the tanh
    /// approximation and the fork's CLIP only knows QuickGelu; all three agree
    /// to about 1e-3 and none of them is a substitute for another.
    #[test]
    fn the_mlp_activation_is_the_exact_erf_gelu() {
        let device = Device::Cpu;
        let identity = |width: usize, scale: f32| {
            let mut data = vec![0.0_f32; width * width];
            for i in 0..width {
                data[i * width + i] = scale;
            }
            Tensor::from_vec(data, (width, width), &device).unwrap()
        };
        let zeros = Tensor::zeros(3, DType::F32, &device).unwrap();
        let mlp = Mlp {
            fc1: Linear::new(identity(3, 1.0), Some(zeros.clone())),
            fc2: Linear::new(identity(3, 1.0), Some(zeros)),
        };
        let xs = Tensor::from_vec(vec![-2.0_f32, 0.5, 1.75], (1, 1, 3), &device).unwrap();
        let actual = mlp
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let erf = xs
            .gelu_erf()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (actual, expected) in actual.iter().zip(&erf) {
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }

        // The tanh approximation differs by ~1e-3 here — small, everywhere,
        // and exactly the kind of drift that reads as "close enough" in a
        // spot check.
        let tanh = xs
            .gelu()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gap = actual
            .iter()
            .zip(&tanh)
            .fold(0.0_f32, |peak, (a, b)| peak.max((a - b).abs()));
        assert!(gap > 1e-4, "the activation choice is not observable: {gap}");

        // QuickGelu (`x * sigmoid(1.702x)`) is further still.
        let quick = (&xs * candle_nn::ops::sigmoid(&(&xs * 1.702_f64).unwrap()).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gap = actual
            .iter()
            .zip(&quick)
            .fold(0.0_f32, |peak, (a, b)| peak.max((a - b).abs()));
        assert!(gap > 1e-3, "quickgelu is not observable: {gap}");
    }

    /// Attention scales the scores, not the query (`modeling_clip.py:265`),
    /// and softmax runs in f32. The scale itself is `head_dim ** -0.5`.
    #[test]
    fn attention_uses_the_published_scale_and_normalizes_its_rows() {
        let config = tiny_config();
        let tower = tiny_tower(&config);
        let attention = &tower.layers[0].self_attn;
        assert!((attention.scale - (config.head_dim() as f64).powf(-0.5)).abs() < 1e-12);
        assert_eq!(config.head_dim(), 2, "8 wide over 4 heads");
        assert!(
            (attention.scale - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
            "head_dim 2 -> 1/sqrt(2)"
        );
        // The real tower: 1280 / 16 = 80, so 80 ** -0.5.
        let published = OpenClipVisionConfig::vit_h_14();
        assert!(((published.head_dim() as f64).powf(-0.5) - 0.111_803_398_874_989_5).abs() < 1e-12);

        // A bidirectional softmax makes every row a distribution; a stray mask
        // or a missing softmax would not.
        let mut rng = Lcg(3);
        let xs = rng.tensor(&[1, config.num_positions(), config.embed_dim]);
        let q = attention
            .split_heads(&attention.q_proj.forward(&xs).unwrap())
            .unwrap();
        let k = attention
            .split_heads(&attention.k_proj.forward(&xs).unwrap())
            .unwrap();
        let scores = (q
            .matmul(&k.transpose(D::Minus2, D::Minus1).unwrap())
            .unwrap()
            * attention.scale)
            .unwrap();
        let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        for row in weights.reshape(((), 5)).unwrap().to_vec2::<f32>().unwrap() {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "row sums to {total}");
            assert!(row.iter().all(|w| *w > 0.0), "a mask leaked in");
        }
        assert_eq!(
            scores.dims(),
            &[1, 4, 5, 5],
            "[batch, heads, tokens, tokens]"
        );
    }

    /// Whole-tower parity against `transformers`' own `CLIPVisionModelWithProjection`.
    ///
    /// Everything above this test says what the port *should* do; this one is
    /// the only thing that says it does. The oracle is a scratch venv running
    /// `transformers` 5.17.0 on `torch` 2.14.0+cpu, driven with
    /// [`tiny_config`]'s geometry and with `load_state_dict` fed the tensors
    /// [`weights`] generates — the Python side reimplements [`Lcg`] so both
    /// halves see the same bytes, and it reported no missing and no unexpected
    /// keys, which is itself the tensor-name check.
    ///
    /// Between them the twelve numbers below pin the whole forward: the
    /// biasless patch conv, CLS ordering, the position embedding, both
    /// `pre_layrnorm` and `post_layernorm` (misplacing either moves every
    /// value), the score-side attention scale, the erf gelu, pooling the CLS
    /// token before the projection, the biasless projection itself, and the
    /// `hidden_states[-2]` tap.
    ///
    /// Tolerance is 2e-6: candle and torch order the same f32 reductions
    /// differently, and nothing structural survives a bound that tight.
    #[test]
    fn the_tower_matches_transformers() {
        let config = tiny_config();
        let tower = tiny_tower(&config);
        let pixels = Lcg(13).tensor(&[1, 3, config.image_size, config.image_size]);
        let output = tower.forward(&pixels).unwrap();

        let image_embeds = output
            .image_embeds
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = [
            -0.012_090_974_f32,
            -0.310_298_77,
            -0.408_486_07,
            -0.400_623_95,
            0.227_755_64,
            0.254_794_24,
        ];
        assert_eq!(image_embeds.len(), expected.len());
        for (index, (actual, expected)) in image_embeds.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-6,
                "image_embeds[{index}]: {actual} vs {expected}"
            );
        }

        let penultimate = output
            .penultimate_hidden_states
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(penultimate.len(), config.num_positions() * config.embed_dim);
        let expected = [
            0.117_059_21_f32,
            0.409_348_6,
            1.042_601_3,
            0.051_179_632,
            0.306_222_92,
            0.301_364_78,
            -0.828_628_8,
            0.578_175_8,
        ];
        for (index, (actual, expected)) in penultimate.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-6,
                "penultimate[{index}]: {actual} vs {expected}"
            );
        }
        // Whole-tensor statistics as well: a defect that misses the first
        // eight elements still moves these.
        let total: f32 = penultimate.iter().sum();
        assert!((total - 7.584_802_6).abs() < 1e-4, "sum {total}");
        let peak = penultimate
            .iter()
            .fold(0.0_f32, |peak, v| peak.max(v.abs()));
        assert!((peak - 1.079_257_7).abs() < 2e-6, "peak {peak}");
    }

    /// A wrongly sized input has to be refused rather than broadcast against a
    /// position embedding trained for one grid.
    #[test]
    fn a_wrongly_shaped_input_is_refused() {
        let config = tiny_config();
        let tower = tiny_tower(&config);
        let mut rng = Lcg(5);
        let error = tower
            .forward(&rng.tensor(&[1, 3, config.image_size, config.image_size + 2]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("expects [batch, 3,"), "unexpected: {error}");
    }
}
