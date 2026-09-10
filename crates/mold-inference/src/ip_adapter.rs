//! IP-Adapter image prompting for SD1.5 and SDXL.
//!
//! IP-Adapter (`tencent-ailab/IP-Adapter`) conditions a render on a reference
//! PICTURE rather than a face: a CLIP-Vision ViT-H/14 tower encodes the image,
//! a small projection turns that one embedding into a handful of image tokens,
//! and every `attn2` module grows a second key/value stream keyed off those
//! tokens. Upstream calls it "decoupled cross-attention" because the text
//! branch is left exactly as it was and the image branch is simply added on:
//!
//! ```text
//! attended = attended + scale * attention(q, to_k_ip(tokens), to_v_ip(tokens))
//! ```
//!
//! Ported from the executable oracle, stable-diffusion.cpp —
//! `src/model/common/block.hpp:385-392` for the injection (note it lands
//! BEFORE `to_out_0`, which is exactly where mold's [`CrossAttentionHook`]
//! fires) and `src/model/adapter/ip_adapter.hpp:12-34` for the projection.
//! Cross-checked against diffusers' `IPAdapterAttnProcessor2_0`
//! (`models/attention_processor.py:4408-4560`) and `ImageProjection`
//! (`models/embeddings.py:1515-1536`).
//!
//! Three properties are load-bearing:
//!
//! * **One module serves both families.** The only thing that differs between
//!   SD1.5 and SDXL is the UNet's `cross_attention_dim` (768 vs 2048) and its
//!   `attn2` geometry (16 modules vs 70), and both come out of the
//!   [`UNet2DConditionModelConfig`] the caller already has. There is no
//!   per-family adapter type, because there is no per-family arithmetic.
//! * **Nothing is transcribed from the checkpoint's shape.** The token count
//!   is `proj.out_features / cross_attention_dim` and the CLIP width is
//!   `proj.in_features`, read off the tensors exactly as
//!   `ip_adapter.hpp:169-175` reads them. A file with a different token count
//!   therefore loads rather than being refused by a hard-coded 4.
//! * **A zero scale is structurally inert.** [`IpAdapterRuntime::hook_for_step`]
//!   yields `None`, the denoise loop calls the UNet's ordinary `forward`, and
//!   the render is bit-identical to one that named no reference image at all —
//!   the same gate PuLID uses, and the same falsification test.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::attention::CrossAttentionHook;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig;

use crate::sd_attn_layout::{plan_attn_layers, AttnLayerSite};

/// Leading module name the checkpoint stores the per-layer projections under.
///
/// `ip_adapter.<i>.to_k_ip.weight`, where `<i>` counts `attn2` modules ONLY —
/// see [`AttnLayerSite::ip_index`], which is where that differs from PuLID.
pub const ADAPTER_PREFIX: &str = "ip_adapter";

/// Leading module name for the image projection.
pub const IMAGE_PROJ_PREFIX: &str = "image_proj";

/// Width of the CLIP-Vision embedding the projection consumes.
///
/// ViT-H/14's `visual_projection` output. Not a hard requirement — the loader
/// reads the real width off `proj.weight` — but the value every shipped
/// `vit-h` adapter has, and what the image tower must therefore produce.
pub const CLIP_EMBED_DIM: usize = 1024;

/// The image tokens one reference picture becomes.
///
/// `ImageProjModel` is `Linear(clip_dim, num_tokens * ctx_dim)` reshaped to
/// `[batch, num_tokens, ctx_dim]`, then layer-normed
/// (`ip_adapter.hpp:18-33`). Classic adapters emit 4; the Plus adapters
/// replace this with a Resampler emitting 16, which is why the count is read
/// from the file rather than assumed.
#[derive(Debug)]
pub struct IpImageProjection {
    proj: Linear,
    norm: LayerNorm,
    tokens: usize,
    context_dim: usize,
    clip_dim: usize,
}

/// The dimensions an adapter file carries, read off its own tensor shapes.
///
/// Upstream derives every one of these rather than assuming
/// (`ip_adapter.hpp:165-175`), and so does mold: a Plus checkpoint's 16 tokens
/// and a classic one's 4 differ only in `proj.weight`'s output width, and a
/// hard-coded 4 would refuse the former for no reason.
///
/// It is a separate step from building the module because a `VarBuilder` can
/// only be asked for a tensor whose shape you already know. The safetensors
/// header is what knows, which is exactly the `tensor_storage_map` upstream
/// consults before it constructs anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpAdapterShape {
    /// Image tokens one reference picture becomes.
    pub tokens: usize,
    /// CLIP-Vision embedding width the projection consumes.
    pub clip_dim: usize,
}

impl IpAdapterShape {
    /// Derive from `image_proj.proj.weight`'s shape against a UNet's width.
    ///
    /// A file whose output width is not a whole multiple of the context dim is
    /// not this UNet's adapter — pairing `ip-adapter_sdxl_vit-h` (2048) with
    /// SD1.5 (768) lands exactly here — and is refused by name rather than
    /// reshaped into nonsense.
    pub fn from_proj_dims(out_dim: usize, clip_dim: usize, context_dim: usize) -> Result<Self> {
        if context_dim == 0 {
            bail!("this UNet advertises a zero-width cross-attention context");
        }
        if !out_dim.is_multiple_of(context_dim) {
            bail!(
                "image_proj.proj projects to {out_dim}, which is not a whole number of \
                 {context_dim}-wide tokens — this adapter belongs to a different UNet"
            );
        }
        let tokens = out_dim / context_dim;
        if tokens == 0 {
            bail!("image_proj.proj projects to fewer than one {context_dim}-wide token");
        }
        Ok(Self { tokens, clip_dim })
    }

    /// Read the shape out of an adapter file's header.
    pub fn from_safetensors(path: &std::path::Path, context_dim: usize) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading IP-Adapter header {}", path.display()))?;
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes)
            .with_context(|| format!("parsing IP-Adapter header {}", path.display()))?;
        let name = format!("{IMAGE_PROJ_PREFIX}.proj.weight");
        let info = metadata
            .tensors()
            .get(&name)
            .cloned()
            .with_context(|| format!("{} carries no {name}", path.display()))?;
        let dims = info.shape.clone();
        let [out_dim, clip_dim] = dims[..] else {
            bail!("{name} is {}-D, expected 2-D", dims.len());
        };
        Self::from_proj_dims(out_dim, clip_dim, context_dim)
    }
}

impl IpImageProjection {
    /// Build from a `VarBuilder` rooted at `image_proj`, at a known shape.
    pub fn load(vb: VarBuilder, context_dim: usize, shape: IpAdapterShape) -> Result<Self> {
        let out_dim = shape.tokens * context_dim;
        Ok(Self {
            // `Linear(clip_dim, num_tokens * ctx_dim, bias = true)` —
            // `ip_adapter.hpp:20`. The bias is real; `to_k_ip`/`to_v_ip` below
            // are the ones without.
            proj: candle_nn::linear(shape.clip_dim, out_dim, vb.pp("proj"))?,
            norm: candle_nn::layer_norm(context_dim, 1e-5, vb.pp("norm"))?,
            tokens: shape.tokens,
            context_dim,
            clip_dim: shape.clip_dim,
        })
    }

    /// Image tokens this projection emits per reference picture.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// CLIP-Vision embedding width this projection consumes.
    pub fn clip_dim(&self) -> usize {
        self.clip_dim
    }

    /// `[batch, clip_dim]` -> `[batch, tokens, context_dim]`.
    ///
    /// `ip_adapter.hpp:25-33`: project, reshape, layer-norm — in that order.
    /// Normalising before the reshape would norm across the concatenated
    /// tokens instead of within each one.
    pub fn forward(&self, image_embeds: &Tensor) -> Result<Tensor> {
        let (batch, width) = image_embeds
            .dims2()
            .context("image embeds must be [batch, clip_dim]")?;
        if width != self.clip_dim {
            bail!(
                "image embeds are {width}-wide but this adapter's projection reads \
                 {}-wide CLIP embeddings",
                self.clip_dim
            );
        }
        let projected = self.proj.forward(image_embeds)?;
        let reshaped = projected.reshape((batch, self.tokens, self.context_dim))?;
        Ok(self.norm.forward(&reshaped)?)
    }
}

/// One module's image key/value projections.
///
/// `Linear(ip_dim, inner_dim, bias = false)` twice —
/// `block.hpp:326-328`, and diffusers' `IPAdapterAttnProcessor2_0.__init__`
/// (`attention_processor.py:4444-4449`).
#[derive(Debug)]
pub struct IpAttnLayer {
    to_k_ip: Linear,
    to_v_ip: Linear,
    site: AttnLayerSite,
}

impl IpAttnLayer {
    pub fn load(site: AttnLayerSite, context_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_k_ip: candle_nn::linear_no_bias(context_dim, site.hidden_size, vb.pp("to_k_ip"))?,
            to_v_ip: candle_nn::linear_no_bias(context_dim, site.hidden_size, vb.pp("to_v_ip"))?,
            site,
        })
    }

    /// Where this module sits and what shape it is.
    pub fn site(&self) -> AttnLayerSite {
        self.site
    }

    /// `[b, n, heads * dim_head]` -> `[b, heads, n, dim_head]`.
    ///
    /// The BHND layout [`crate::attention`] expects, and upstream's own
    /// `view(batch, -1, heads, head_dim).transpose(1, 2)`
    /// (`attention_processor.py:4514-4517`).
    fn split_heads(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        xs.reshape((b, n, self.site.heads, self.site.dim_head()))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// The image branch's contribution, unscaled: `[batch, seq, hidden_size]`.
    ///
    /// `query` is `to_q`'s own output, the tensor the hook is handed before the
    /// head split — the same tensor the text branch attended with, which is the
    /// whole point of "decoupled": one query, two key/value streams.
    pub fn ip_hidden_states(&self, tokens: &Tensor, query: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = query.dims3()?;
        let k = self.to_k_ip.forward(tokens)?;
        let v = self.to_v_ip.forward(tokens)?;

        let q = self.split_heads(query)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // `F.scaled_dot_product_attention`'s default scale — upstream passes
        // no explicit `scale=` (`attention_processor.py:4560`).
        let scale = (self.site.dim_head() as f64).powf(-0.5) as f32;
        let attn = crate::attention::attention(&q, &k, &v, scale)?;
        attn.transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.site.hidden_size))
    }

    /// `attended + scale * ip_hidden_states(...)` (`block.hpp:391`).
    pub fn inject(
        &self,
        tokens: &Tensor,
        query: &Tensor,
        attended: &Tensor,
        scale: f32,
    ) -> candle_core::Result<Tensor> {
        let delta = self.ip_hidden_states(tokens, query)?;
        attended + (delta * f64::from(scale))?
    }
}

/// Every image projection for one UNet, indexed by hook position.
#[derive(Debug)]
pub struct IpAdapter {
    image_proj: IpImageProjection,
    layers: Vec<IpAttnLayer>,
    context_dim: usize,
    dtype: DType,
}

impl IpAdapter {
    /// Load from `ip-adapter_sd15.safetensors` or
    /// `ip-adapter_sdxl_vit-h.safetensors`.
    ///
    /// Opened by pathname, for the reasons recorded on
    /// [`crate::sdxl::pulid::SdxlPulidAdapter::load`] and tracked as #1308 —
    /// this is a manifest file whose pinned digest the download verified, and
    /// hardening one loader of several would give a single contract two
    /// answers.
    pub fn load(
        path: &std::path::Path,
        config: &UNet2DConditionModelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // SAFETY: the same mmap contract every other mold safetensors loader
        // relies on — the file must not be mutated while the engine holds it.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&path.to_path_buf()),
                dtype,
                device,
            )
            .with_context(|| format!("reading IP-Adapter {}", path.display()))?
        };
        let shape = IpAdapterShape::from_safetensors(path, config.cross_attention_dim)?;
        Self::from_var_builder(vb, config, shape)
            .with_context(|| format!("loading IP-Adapter {}", path.display()))
    }

    /// Build the stack from a `VarBuilder` rooted at the file's top level.
    ///
    /// The inventory is checked against the layout the UNet config implies in
    /// BOTH directions, for the reason the PuLID loader documents: an
    /// SD1.5-shaped plan against the SDXL file would find its planned indices
    /// present — they are a prefix — load 16 of the file's 70 modules, and
    /// render an image conditioned on a fraction of the reference. Here the
    /// index space is dense from zero, so the surplus check is simply "is
    /// there an `ip_adapter.<n>` above the plan".
    pub fn from_var_builder(
        vb: VarBuilder,
        config: &UNet2DConditionModelConfig,
        shape: IpAdapterShape,
    ) -> Result<Self> {
        let sites = plan_attn_layers(config);
        if sites.is_empty() {
            bail!("this UNet has no cross-attention modules to condition");
        }
        let context_dim = config.cross_attention_dim;
        let image_proj = IpImageProjection::load(vb.pp(IMAGE_PROJ_PREFIX), context_dim, shape)?;

        let adapter = vb.pp(ADAPTER_PREFIX);
        let planned = sites.len();
        let mut missing = Vec::new();
        let mut layers = Vec::with_capacity(planned);
        for site in &sites {
            let index = site.ip_index();
            let entry = adapter.pp(index.to_string());
            if !entry.contains_tensor("to_k_ip.weight") {
                missing.push(index);
                continue;
            }
            layers.push(IpAttnLayer::load(*site, context_dim, entry)?);
        }
        if !missing.is_empty() {
            bail!(
                "this IP-Adapter is missing {} of the {planned} cross-attention modules this \
                 UNet has (first absent: ip_adapter.{}); it belongs to a different architecture",
                missing.len(),
                missing[0]
            );
        }
        // The file's own index space is dense from zero, so one probe past the
        // plan is enough to catch an SDXL checkpoint loaded against an SD1.5
        // plan — the case where every planned index IS present.
        if adapter.contains_tensor(&format!("{planned}.to_k_ip.weight")) {
            bail!(
                "this IP-Adapter carries more than the {planned} cross-attention modules this \
                 UNet has; it belongs to a different architecture"
            );
        }

        Ok(Self {
            image_proj,
            layers,
            context_dim,
            dtype: vb.dtype(),
        })
    }

    /// Modules this adapter conditions.
    pub fn module_count(&self) -> usize {
        self.layers.len()
    }

    /// Image tokens one reference picture becomes.
    pub fn tokens(&self) -> usize {
        self.image_proj.tokens()
    }

    /// The UNet width this adapter was built against.
    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    /// The dtype its weights were read at.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The image projection, for turning a CLIP embedding into tokens.
    pub fn image_proj(&self) -> &IpImageProjection {
        &self.image_proj
    }

    /// One module, by hook index.
    pub fn layer(&self, hook_index: usize) -> Option<&IpAttnLayer> {
        self.layers.get(hook_index)
    }

    /// Device+host bytes these projections occupy once resident.
    ///
    /// Two `[hidden_size, context_dim]` matrices per module, plus the
    /// projection's `[tokens * context_dim, clip_dim]` and its bias and norm.
    /// The arithmetic rather than a measured constant, so a preflight charge
    /// pinned against it cannot drift when the layer table changes.
    pub fn resident_bytes(&self) -> u64 {
        let width = self.dtype.size_in_bytes() as u64;
        let attn: u64 = self
            .layers
            .iter()
            .map(|layer| 2 * layer.site().hidden_size as u64 * self.context_dim as u64)
            .sum();
        let proj = (self.image_proj.tokens() * self.context_dim) as u64
            * self.image_proj.clip_dim() as u64
            + (self.image_proj.tokens() * self.context_dim) as u64
            + 2 * self.context_dim as u64;
        (attn + proj) * width
    }
}

/// The projected image tokens and the strength they inject at.
///
/// Belongs to the REQUEST, not to the loaded UNet — which is why it is a
/// separate type from [`IpAdapter`]. One denoise loop is the only place both
/// are in scope.
#[derive(Debug)]
pub struct IpAdapterContext {
    tokens: Tensor,
    scale: f32,
}

impl IpAdapterContext {
    /// Project a CLIP-Vision embedding into this UNet's token space.
    ///
    /// `image_embeds` is `[batch, clip_dim]` — the POOLED, PROJECTED embedding
    /// (`visual_projection(post_layernorm(hidden[:, 0]))`), not hidden states.
    /// The Plus adapters take hidden states instead, which is a different
    /// projection module and a later change.
    pub fn new(
        adapter: &IpAdapter,
        image_embeds: &Tensor,
        scale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let embeds = image_embeds.to_device(device)?.to_dtype(dtype)?;
        let tokens = adapter.image_proj().forward(&embeds)?;
        Ok(Self { tokens, scale })
    }

    /// Wrap already-projected tokens, for a caller that projected once and is
    /// reusing them across a batch.
    pub fn from_tokens(tokens: Tensor, scale: f32) -> Self {
        Self { tokens, scale }
    }

    /// `[batch, tokens, context_dim]`.
    pub fn tokens(&self) -> &Tensor {
        &self.tokens
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Whether this context contributes anything at all.
    ///
    /// `block.hpp:385` gates on exactly this: `ctx->ip_scale != 0.0f`.
    pub fn is_active(&self) -> bool {
        self.scale != 0.0
    }
}

/// The adapter and the context, paired for the length of one denoise loop.
#[derive(Debug, Clone, Copy)]
pub struct IpAdapterRuntime<'a> {
    adapter: &'a IpAdapter,
    context: &'a IpAdapterContext,
}

impl<'a> IpAdapterRuntime<'a> {
    pub fn new(adapter: &'a IpAdapter, context: &'a IpAdapterContext) -> Self {
        Self { adapter, context }
    }

    /// The hook for this step, or `None` when the reference contributes
    /// nothing.
    ///
    /// `None` is the whole gate: the denoise loop answers it by calling the
    /// UNet's ordinary `forward`, so an inactive render executes the exact
    /// code a build with no reference image executes. Bit-identity is
    /// structural rather than numerical.
    ///
    /// IP-Adapter has no start-step control upstream, so the step is not read
    /// — the parameter is there so a caller cannot accidentally hoist the
    /// call out of the loop and silently lose a future one.
    pub fn hook_for_step(&self, _step: usize) -> Option<IpAdapterHook<'a>> {
        self.context.is_active().then_some(IpAdapterHook {
            adapter: self.adapter,
            context: self.context,
        })
    }

    pub fn adapter(&self) -> &'a IpAdapter {
        self.adapter
    }
}

/// Injects image features into every planned cross-attention module.
#[derive(Debug, Clone, Copy)]
pub struct IpAdapterHook<'a> {
    adapter: &'a IpAdapter,
    context: &'a IpAdapterContext,
}

impl IpAdapterHook<'_> {
    /// Broadcast the reference tokens across a CFG batch.
    ///
    /// mold runs `[uncond, cond]` as ONE forward, so a `[1, tokens, dim]`
    /// context has to cover both rows. Upstream runs the two branches
    /// separately and hands each the same tokens, so broadcasting reproduces
    /// it exactly — the image prompt applies to both branches, unlike PuLID's
    /// true CFG where the negative branch takes a DIFFERENT embedding.
    fn broadcast(&self, batch: usize) -> candle_core::Result<Tensor> {
        let tokens = self.context.tokens();
        let have = tokens.dim(0)?;
        if have == batch {
            return Ok(tokens.clone());
        }
        if have == 1 {
            return tokens
                .broadcast_as((batch, tokens.dim(1)?, tokens.dim(2)?))?
                .contiguous();
        }
        candle_core::bail!("IP-Adapter tokens carry {have} rows but the forward runs {batch}")
    }
}

impl CrossAttentionHook for IpAdapterHook<'_> {
    fn cross_attention(
        &self,
        index: usize,
        query: &Tensor,
        attended: &Tensor,
        heads: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        let Some(layer) = self.adapter.layer(index) else {
            // The cursor walked past the plan: the UNet running this forward is
            // not the one the layer table was built from. Refusing beats
            // conditioning a prefix of the modules and returning a picture.
            candle_core::bail!(
                "IP-Adapter was planned for {} cross-attention modules but the UNet reached \
                 index {index}",
                self.adapter.module_count()
            );
        };
        if layer.site().heads != heads {
            candle_core::bail!(
                "IP-Adapter module {index} was planned for {} heads but the UNet reports {heads}",
                layer.site().heads
            );
        }
        let batch = query.dim(0)?;
        let tokens = self.broadcast(batch)?;
        Ok(Some(layer.inject(
            &tokens,
            query,
            attended,
            self.context.scale(),
        )?))
    }
}

/// Every planned `ip_adapter.<i>` name, for an inventory check or a test.
pub fn planned_module_names(config: &UNet2DConditionModelConfig) -> BTreeMap<usize, String> {
    plan_attn_layers(config)
        .into_iter()
        .map(|site| {
            (
                site.hook_index,
                format!("{ADAPTER_PREFIX}.{}", site.ip_index()),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn sd15_config() -> UNet2DConditionModelConfig {
        mold_candle::stable_diffusion::sd15_unet()
    }

    fn sdxl_config() -> UNet2DConditionModelConfig {
        mold_candle::stable_diffusion::sdxl_unet()
    }

    /// A checkpoint shaped exactly as `plan_attn_layers` says it should be.
    ///
    /// Built rather than downloaded: the real files are 44 MB and 700 MB, and
    /// what these tests check is the LAYOUT and the arithmetic, which a
    /// synthetic file expresses exactly as well. Golden numerics against the
    /// real weights are the oracle fixture's job.
    ///
    /// Weights are drawn from a normal rather than left at `VarMap`'s zero
    /// default: with zero projections every injection is the identity, and the
    /// "a live scale changes the render" half of the bit-identity argument
    /// would pass on an adapter that does nothing at all.
    fn tensor(varmap: &VarMap, shape: (usize, usize), name: &str) {
        varmap
            .get(
                shape,
                name,
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
    }

    fn vector(varmap: &VarMap, len: usize, name: &str) {
        varmap
            .get(
                len,
                name,
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
    }

    /// Populate a varmap with an adapter for `config`, optionally with one
    /// module more than the plan calls for.
    fn populate(
        varmap: &VarMap,
        config: &UNet2DConditionModelConfig,
        tokens: usize,
        clip_dim: usize,
        surplus: bool,
        skip: Option<usize>,
    ) {
        let context_dim = config.cross_attention_dim;
        tensor(
            varmap,
            (tokens * context_dim, clip_dim),
            &format!("{IMAGE_PROJ_PREFIX}.proj.weight"),
        );
        vector(
            varmap,
            tokens * context_dim,
            &format!("{IMAGE_PROJ_PREFIX}.proj.bias"),
        );
        vector(
            varmap,
            context_dim,
            &format!("{IMAGE_PROJ_PREFIX}.norm.weight"),
        );
        vector(
            varmap,
            context_dim,
            &format!("{IMAGE_PROJ_PREFIX}.norm.bias"),
        );

        let sites = plan_attn_layers(config);
        for site in &sites {
            if skip == Some(site.ip_index()) {
                continue;
            }
            for name in ["to_k_ip", "to_v_ip"] {
                tensor(
                    varmap,
                    (site.hidden_size, context_dim),
                    &format!("{ADAPTER_PREFIX}.{}.{name}.weight", site.ip_index()),
                );
            }
        }
        if surplus {
            let last = sites.last().expect("a UNet with cross-attention");
            for name in ["to_k_ip", "to_v_ip"] {
                tensor(
                    varmap,
                    (last.hidden_size, context_dim),
                    &format!("{ADAPTER_PREFIX}.{}.{name}.weight", sites.len()),
                );
            }
        }
    }

    fn build(
        varmap: &VarMap,
        config: &UNet2DConditionModelConfig,
        tokens: usize,
        clip_dim: usize,
    ) -> Result<IpAdapter> {
        IpAdapter::from_var_builder(
            VarBuilder::from_varmap(varmap, DType::F32, &Device::Cpu),
            config,
            IpAdapterShape { tokens, clip_dim },
        )
    }

    fn synthetic(
        config: &UNet2DConditionModelConfig,
        tokens: usize,
        clip_dim: usize,
    ) -> (VarMap, IpAdapter) {
        let varmap = VarMap::new();
        populate(&varmap, config, tokens, clip_dim, false, None);
        let adapter = build(&varmap, config, tokens, clip_dim)
            .expect("the synthetic checkpoint matches the plan");
        (varmap, adapter)
    }

    #[test]
    fn both_families_load_their_own_geometry() {
        let (_map, sd15) = synthetic(&sd15_config(), 4, CLIP_EMBED_DIM);
        assert_eq!(sd15.module_count(), 16);
        assert_eq!(sd15.context_dim(), 768);
        assert_eq!(sd15.tokens(), 4);

        // SDXL's side is asserted from the PLAN rather than by materialising
        // weights: 70 modules at up to [1280, 2048] is ~1.5 GB of tensors to
        // prove an arithmetic fact the layer table already owns, and the real
        // file's inventory is pinned by the fixture test in `sdxl::pulid`.
        let sdxl = sdxl_config();
        assert_eq!(plan_attn_layers(&sdxl).len(), 70);
        assert_eq!(sdxl.cross_attention_dim, 2048);
        assert_eq!(planned_module_names(&sdxl).len(), 70);
    }

    /// The token count is the file's, not a constant.
    ///
    /// `ip_adapter.hpp:174` derives it as `out_dim / ctx_dim` precisely so a
    /// Plus checkpoint's 16 tokens load through the same path. Hard-coding 4
    /// would refuse them.
    #[test]
    fn the_token_count_comes_from_the_checkpoint() {
        for tokens in [1usize, 4, 16] {
            let (_map, adapter) = synthetic(&sd15_config(), tokens, CLIP_EMBED_DIM);
            assert_eq!(adapter.tokens(), tokens);
            let embeds = Tensor::zeros((1, CLIP_EMBED_DIM), DType::F32, &Device::Cpu).unwrap();
            let projected = adapter.image_proj().forward(&embeds).unwrap();
            assert_eq!(projected.dims(), &[1, tokens, 768]);
        }
    }

    /// An SD1.5 adapter against the SDXL plan finds every planned index
    /// present — they are a prefix — so a one-directional check would accept
    /// it, load 16 of 70 modules, and render a picture conditioned on a
    /// fraction of the reference. Both directions are checked.
    #[test]
    fn a_checkpoint_for_a_different_unet_is_refused_in_both_directions() {
        // Surplus: one module more than this UNet has. This is the direction
        // that matters, because an SDXL file's planned indices are a SUPERSET
        // of SD1.5's — every index an SD1.5-shaped plan looks for is present,
        // so only a probe past the plan can tell the two apart.
        let varmap = VarMap::new();
        populate(&varmap, &sd15_config(), 4, CLIP_EMBED_DIM, true, None);
        let error = build(&varmap, &sd15_config(), 4, CLIP_EMBED_DIM)
            .expect_err("a surplus module must be refused");
        assert!(error.to_string().contains("more than"), "{error}");

        // Missing: a truncated or partially-converted file.
        let varmap = VarMap::new();
        populate(&varmap, &sd15_config(), 4, CLIP_EMBED_DIM, false, Some(7));
        let error = build(&varmap, &sd15_config(), 4, CLIP_EMBED_DIM)
            .expect_err("an absent module must be refused");
        let message = error.to_string();
        assert!(message.contains("missing"), "{message}");
        assert!(message.contains("ip_adapter.7"), "{message}");
    }

    /// Pairing the SDXL adapter with SD1.5 is caught before a single module is
    /// read, by the projection width alone.
    ///
    /// This is the real cross-family mistake — the two files sit in sibling
    /// directories of one repo — and the earliest honest place to catch it is
    /// `image_proj.proj`, whose output is `tokens * cross_attention_dim`.
    /// 4 * 2048 is not a whole number of 768-wide tokens.
    #[test]
    fn the_wrong_family_adapter_is_caught_by_its_projection_width() {
        assert!(IpAdapterShape::from_proj_dims(4 * 2048, CLIP_EMBED_DIM, 768).is_err());
        assert!(IpAdapterShape::from_proj_dims(4 * 768, CLIP_EMBED_DIM, 2048).is_err());
        // Each with its own UNet resolves.
        assert!(IpAdapterShape::from_proj_dims(4 * 768, CLIP_EMBED_DIM, 768).is_ok());
        assert!(IpAdapterShape::from_proj_dims(4 * 2048, CLIP_EMBED_DIM, 2048).is_ok());
    }

    /// A projection whose output is not a whole number of tokens is not this
    /// UNet's adapter, and reshaping it would silently produce garbage tokens.
    #[test]
    fn a_projection_that_does_not_divide_into_tokens_is_refused() {
        // The SDXL adapter's projection is 4 * 2048 wide. Against SD1.5's 768
        // that divides with a remainder — exactly what pairing
        // `ip-adapter_sdxl_vit-h` with SD1.5 produces, and the earliest point
        // it can be caught.
        let error = IpAdapterShape::from_proj_dims(4 * 2048, CLIP_EMBED_DIM, 768)
            .expect_err("a non-multiple projection must be refused");
        assert!(error.to_string().contains("whole number"), "{error}");

        // The matching pairs still resolve, and to the token count the file
        // actually carries.
        assert_eq!(
            IpAdapterShape::from_proj_dims(4 * 768, CLIP_EMBED_DIM, 768).unwrap(),
            IpAdapterShape {
                tokens: 4,
                clip_dim: CLIP_EMBED_DIM
            }
        );
        assert_eq!(
            IpAdapterShape::from_proj_dims(16 * 2048, CLIP_EMBED_DIM, 2048)
                .unwrap()
                .tokens,
            16
        );
    }

    /// A zero scale is inert ARITHMETICALLY as well as structurally.
    ///
    /// The structural gate is `hook_for_step` yielding `None`; this pins the
    /// other half, so a caller that installs the hook anyway still cannot
    /// change a pixel.
    #[test]
    fn a_zero_scale_leaves_the_attention_output_untouched() {
        let (_map, adapter) = synthetic(&sd15_config(), 4, CLIP_EMBED_DIM);
        let device = Device::Cpu;
        let layer = adapter.layer(0).expect("module 0");
        let hidden = layer.site().hidden_size;
        let tokens = Tensor::randn(0f32, 1.0, (1, 4, 768), &device).unwrap();
        let query = Tensor::randn(0f32, 1.0, (1, 6, hidden), &device).unwrap();
        let attended = Tensor::randn(0f32, 1.0, (1, 6, hidden), &device).unwrap();

        let injected = layer.inject(&tokens, &query, &attended, 0.0).unwrap();
        assert_eq!(
            injected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            attended.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let live = layer.inject(&tokens, &query, &attended, 1.0).unwrap();
        assert_ne!(
            live.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            attended.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "a live scale must actually change the output"
        );
    }

    /// A `[1, ...]` context covers a `[uncond, cond]` forward.
    ///
    /// mold runs CFG as one doubled batch where upstream runs two passes, and
    /// upstream hands the SAME tokens to both — so broadcasting reproduces it.
    #[test]
    fn the_tokens_broadcast_across_a_cfg_batch() {
        let (_map, adapter) = synthetic(&sd15_config(), 4, CLIP_EMBED_DIM);
        let device = Device::Cpu;
        let tokens = Tensor::randn(0f32, 1.0, (1, 4, 768), &device).unwrap();
        let context = IpAdapterContext::from_tokens(tokens, 1.0);
        let hook = IpAdapterRuntime::new(&adapter, &context)
            .hook_for_step(0)
            .expect("a live scale yields a hook");

        let hidden = adapter.layer(0).unwrap().site().hidden_size;
        let query = Tensor::randn(0f32, 1.0, (2, 6, hidden), &device).unwrap();
        let attended = Tensor::randn(0f32, 1.0, (2, 6, hidden), &device).unwrap();
        let out = hook
            .cross_attention(0, &query, &attended, adapter.layer(0).unwrap().site().heads)
            .unwrap()
            .expect("the hook replaces the attention output");
        assert_eq!(out.dims(), attended.dims());
    }

    /// The gate the whole bit-identity claim rests on.
    #[test]
    fn a_zero_scale_yields_no_hook_at_all() {
        let (_map, adapter) = synthetic(&sd15_config(), 4, CLIP_EMBED_DIM);
        let tokens = Tensor::zeros((1, 4, 768), DType::F32, &Device::Cpu).unwrap();
        let inert = IpAdapterContext::from_tokens(tokens.clone(), 0.0);
        assert!(IpAdapterRuntime::new(&adapter, &inert)
            .hook_for_step(0)
            .is_none());
        let live = IpAdapterContext::from_tokens(tokens, 0.5);
        assert!(IpAdapterRuntime::new(&adapter, &live)
            .hook_for_step(0)
            .is_some());
    }

    /// A hook handed an index or a head count it was not planned for refuses
    /// rather than conditioning the wrong module.
    #[test]
    fn a_hook_refuses_what_it_was_not_planned_for() {
        let (_map, adapter) = synthetic(&sd15_config(), 4, CLIP_EMBED_DIM);
        let device = Device::Cpu;
        let tokens = Tensor::zeros((1, 4, 768), DType::F32, &device).unwrap();
        let context = IpAdapterContext::from_tokens(tokens, 1.0);
        let hook = IpAdapterRuntime::new(&adapter, &context)
            .hook_for_step(0)
            .unwrap();
        let site = adapter.layer(0).unwrap().site();
        let query = Tensor::zeros((1, 6, site.hidden_size), DType::F32, &device).unwrap();
        let attended = query.clone();

        assert!(hook
            .cross_attention(adapter.module_count(), &query, &attended, site.heads)
            .is_err());
        assert!(hook
            .cross_attention(0, &query, &attended, site.heads + 1)
            .is_err());
    }

    /// The names the loader looks for are the names the file has.
    #[test]
    fn the_planned_names_are_dense_from_zero() {
        for config in [sd15_config(), sdxl_config()] {
            let names = planned_module_names(&config);
            let count = plan_attn_layers(&config).len();
            assert_eq!(names.len(), count);
            let suffixes: std::collections::BTreeSet<usize> = names
                .values()
                .map(|name| name.rsplit('.').next().unwrap().parse().unwrap())
                .collect();
            assert_eq!(suffixes, (0..count).collect());
        }
    }
}
