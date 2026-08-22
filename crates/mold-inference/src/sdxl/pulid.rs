//! PuLID v1.1 identity cross-attention adapter for SDXL.
//!
//! PuLID conditions an SDXL render on a face by mixing a second key/value
//! stream into every one of the UNet's text cross-attentions. Each `attn2`
//! module keeps its own query, projects the 32-token identity embedding through
//! a bias-free `id_to_k` / `id_to_v` pair, attends the query against that, and
//! adds the result onto the text attention output **before** `to_out`:
//!
//! ```text
//! attended = attended + id_scale * sdpa(q, id_to_k(id), id_to_v(id))
//! ```
//!
//! Ported from `ToTheBeginning/PuLID`
//! (`1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa`):
//! `pulid/attention_processor.py:275-422` (`IDAttnProcessor2_0`, whose identity
//! branch is `:355-379`) and `pulid/pipeline_v1_1.py:129-149`
//! (`hack_unet_attn_layers`, which decides WHICH modules get weights and in
//! what order). The module globals that branch reads are pinned at
//! `NUM_ZERO = 0` and `ORTHO = ORTHO_v2 = False`
//! (`attention_processor.py:23-25`), so the shipped arithmetic is the plain
//! additive combination at `:378` — the two orthogonal-projection variants are
//! research code that this checkpoint was never trained under, and
//! `crates/mold-inference/testdata/pulid_sdxl/capture_attn_goldens.py` asserts
//! those three globals still hold upstream before it captures anything.
//!
//! Three properties are load-bearing.
//!
//! * **The layer table is a permutation, not an offset.** The checkpoint keys
//!   its weights by diffusers' `unet.attn_processors` position, which walks
//!   `down_blocks -> up_blocks -> mid_block`; candle's UNet forward — and
//!   therefore the hook index — walks `down_blocks -> mid_block -> up_blocks`.
//!   Reading `id_adapter_attn_layers.<hook>` would silently condition the mid
//!   block on the up blocks' weights. [`plan_attn_layers`] derives the mapping
//!   from the UNet config and a fixture test pins it against the checkpoint's
//!   own tensor inventory.
//! * **Zero identity work happens when identity is off.** The gate lives in
//!   [`SdxlPulidRuntime::hook_for_step`]: before `id_start_step`, or at an
//!   effective `id_weight` of 0, it yields `None` and the denoise loop calls
//!   the UNet's ordinary `forward`. Bit-identity with a vanilla SDXL render is
//!   therefore a property of the control flow, not of an arithmetic
//!   coincidence.
//! * **The identity rides the CFG batch.** Upstream runs two separate UNet
//!   passes and hands the negative one `uncond_id_embedding`
//!   (`pipeline_v1_1.py:306-316`); mold runs ONE `[uncond, cond]` batched
//!   forward, so the embedding is concatenated on dim 0 in the same order. A
//!   `[1, 32, 2048]` embedding against a batch-2 query would broadcast the
//!   conditional identity onto the negative branch and quietly halve the
//!   guidance the render was asked for.
//!
//! Attention runs through [`crate::attention`] so the Metal auto-chunking and
//! `MOLD_ATTN` rules that bound every other score matrix in the engine apply
//! here too.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::attention::CrossAttentionHook;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig;

/// Identity tokens in a PuLID embedding.
///
/// The IDFormer concatenates its 32 learned queries with 5 per-image identity
/// tokens internally, but slices back to `num_queries` before returning
/// (`pulid/encoders_transformer.py`), so what reaches an attention layer is 32.
pub const ID_TOKENS: usize = 32;

/// Width of one identity token — the IDFormer's `proj_out` output width.
pub const ID_TOKEN_DIM: usize = 2048;

/// Leading module name PuLID's SDXL checkpoint stores the injection weights
/// under (`pipeline_v1_1.py:151-163` splits the file by it).
pub const ADAPTER_PREFIX: &str = "id_adapter_attn_layers";

/// One hooked cross-attention module: where it sits in the forward pass, and
/// which checkpoint index carries its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnLayerSite {
    /// Position among `attn2` modules in the order candle's UNet forward
    /// visits them — `down_blocks -> mid_block -> up_blocks` — which is
    /// exactly the `index` the [`CrossAttentionHook`] receives.
    pub hook_index: usize,
    /// Position in diffusers' `unet.attn_processors`, which interleaves
    /// `attn1` and `attn2` and walks `down_blocks -> up_blocks -> mid_block`.
    /// This is the `<i>` in `id_adapter_attn_layers.<i>`.
    pub processor_index: usize,
    /// The module's own channel width; `id_to_k` / `id_to_v` are
    /// `[hidden_size, ID_TOKEN_DIM]`.
    pub hidden_size: usize,
    /// Attention heads this module splits `hidden_size` across.
    pub heads: usize,
}

impl AttnLayerSite {
    /// Width of one head. `IDAttnProcessor2_0` derives it the same way
    /// (`attention_processor.py:338`, `inner_dim // attn.heads`).
    pub fn dim_head(&self) -> usize {
        self.hidden_size / self.heads
    }
}

/// A contiguous run of `attn2` modules that share a width and a head count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Region {
    Down(usize),
    Mid,
    Up(usize),
}

#[derive(Debug, Clone, Copy)]
struct RegionShape {
    region: Region,
    /// `attn2` modules this region contributes.
    count: usize,
    hidden_size: usize,
    heads: usize,
}

/// The `attn2` layout of one UNet, in hook order, with each module's
/// checkpoint index.
///
/// Derived from the UNet config rather than transcribed, so the SD1.5 geometry
/// (16 modules) and the SDXL geometry (70) come out of the same arithmetic and
/// a config change cannot leave a hard-coded table behind. The two orders are
/// built separately and joined on `(region, local index)`:
///
/// * diffusers registers `down_blocks`, then `up_blocks`, then `mid_block`,
///   because that is `UNet2DConditionModel.__init__`'s attribute order and
///   `attn_processors` walks `named_children` — and within each transformer
///   block it registers `attn1` before `attn2`.
/// * candle's `forward` runs `down_blocks`, then `mid_block`, then `up_blocks`
///   (`unet_2d.rs`'s "3. down / 4. mid / 5. up"), which is the order the hook
///   cursor counts in.
pub fn plan_attn_layers(config: &UNet2DConditionModelConfig) -> Vec<AttnLayerSite> {
    let n_blocks = config.blocks.len();
    let mut downs = Vec::new();
    let mut ups = Vec::new();
    for index in 0..n_blocks {
        let block = config.blocks[index];
        if let Some(transformer_layers) = block.use_cross_attn {
            downs.push(RegionShape {
                region: Region::Down(index),
                count: config.layers_per_block * transformer_layers,
                hidden_size: block.out_channels,
                heads: block.attention_head_dim,
            });
        }
        // `up_blocks[i]` is built from `blocks[n - 1 - i]` and carries one more
        // resnet layer than its down-block mirror (`unet_2d.rs`'s
        // `num_layers: config.layers_per_block + 1`).
        let mirrored = config.blocks[n_blocks - 1 - index];
        if let Some(transformer_layers) = mirrored.use_cross_attn {
            ups.push(RegionShape {
                region: Region::Up(index),
                count: (config.layers_per_block + 1) * transformer_layers,
                hidden_size: mirrored.out_channels,
                heads: mirrored.attention_head_dim,
            });
        }
    }
    // The mid block is always cross-attentional and always takes the last
    // block's width, head count, and transformer depth (`unet_2d.rs`'s
    // `mid_transformer_layers_per_block`, mirroring diffusers' own
    // `unet_2d_condition.py:462`).
    let mid = config.blocks.last().map(|block| RegionShape {
        region: Region::Mid,
        count: block.use_cross_attn.unwrap_or(1),
        hidden_size: block.out_channels,
        heads: block.attention_head_dim,
    });

    // Diffusers registration order assigns the checkpoint indices.
    let mut processor_of: BTreeMap<(Region, usize), usize> = BTreeMap::new();
    let mut next_processor = 0usize;
    let diffusers_order = downs
        .iter()
        .chain(ups.iter())
        .chain(mid.iter())
        .copied()
        .collect::<Vec<_>>();
    for shape in &diffusers_order {
        for local in 0..shape.count {
            // Every transformer block registers `attn1` then `attn2`.
            next_processor += 1;
            processor_of.insert((shape.region, local), next_processor);
            next_processor += 1;
        }
    }

    // Candle's forward order assigns the hook indices.
    let candle_order = downs
        .iter()
        .chain(mid.iter())
        .chain(ups.iter())
        .copied()
        .collect::<Vec<_>>();
    let mut sites = Vec::with_capacity(next_processor / 2);
    for shape in &candle_order {
        for local in 0..shape.count {
            let processor_index = processor_of[&(shape.region, local)];
            sites.push(AttnLayerSite {
                hook_index: sites.len(),
                processor_index,
                hidden_size: shape.hidden_size,
                heads: shape.heads,
            });
        }
    }
    sites
}

/// One module's identity key/value projections.
///
/// `IDAttnProcessor2_0.__init__` (`attention_processor.py:285-291`) builds
/// exactly these two, both `nn.Linear(cross_attention_dim, hidden_size,
/// bias=False)`.
#[derive(Debug)]
pub struct IdAttnLayer {
    id_to_k: Linear,
    id_to_v: Linear,
    site: AttnLayerSite,
}

impl IdAttnLayer {
    /// Load one module from `vb`, which must already be scoped to an
    /// `id_adapter_attn_layers.{processor_index}` prefix.
    pub fn load(site: AttnLayerSite, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            id_to_k: candle_nn::linear_no_bias(ID_TOKEN_DIM, site.hidden_size, vb.pp("id_to_k"))?,
            id_to_v: candle_nn::linear_no_bias(ID_TOKEN_DIM, site.hidden_size, vb.pp("id_to_v"))?,
            site,
        })
    }

    /// Where this module sits and what shape it is.
    pub fn site(&self) -> AttnLayerSite {
        self.site
    }

    /// `[b, n, heads * dim_head]` -> `[b, heads, n, dim_head]`, the BHND layout
    /// [`crate::attention`] expects. Mirrors upstream's
    /// `view(batch, -1, heads, head_dim).transpose(1, 2)`
    /// (`attention_processor.py:341-345, 368-369`).
    fn split_heads(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        xs.reshape((b, n, self.site.heads, self.site.dim_head()))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// The identity branch's contribution, unscaled.
    ///
    /// `query` is `to_q`'s own output, `[batch, seq, hidden_size]`, exactly the
    /// tensor upstream reshapes at `attention_processor.py:341`. Returns
    /// `[batch, seq, hidden_size]`, ready to be scaled and added onto the text
    /// attention output.
    pub fn id_hidden_states(
        &self,
        id_embeds: &Tensor,
        query: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = query.dims3()?;
        let id_k = self.id_to_k.forward(id_embeds)?;
        let id_v = self.id_to_v.forward(id_embeds)?;

        let q = self.split_heads(query)?;
        let k = self.split_heads(&id_k)?;
        let v = self.split_heads(&id_v)?;

        // `F.scaled_dot_product_attention`'s default scale
        // (`attention_processor.py:373-375`, no explicit `scale=`).
        let scale = (self.site.dim_head() as f64).powf(-0.5) as f32;
        let attn = crate::attention::attention(&q, &k, &v, scale)?;
        attn.transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.site.hidden_size))
    }

    /// `attended + id_weight * id_hidden_states(...)`
    /// (`attention_processor.py:378`).
    pub fn inject(
        &self,
        id_embeds: &Tensor,
        query: &Tensor,
        attended: &Tensor,
        id_weight: f32,
    ) -> candle_core::Result<Tensor> {
        let delta = self.id_hidden_states(id_embeds, query)?;
        attended + (delta * f64::from(id_weight))?
    }
}

/// Every identity projection for one UNet, indexed by hook position.
#[derive(Debug)]
pub struct SdxlPulidAdapter {
    layers: Vec<IdAttnLayer>,
    dtype: DType,
}

impl SdxlPulidAdapter {
    /// Load the adapter from PuLID's `pulid_v1.1.safetensors`.
    ///
    /// The file also carries the IDFormer (`id_adapter.*`), which this ignores:
    /// only `id_adapter_attn_layers.*` is read. `dtype` is the UNet's working
    /// dtype — the file ships f16 and the `VarBuilder` casts on read.
    pub fn load(
        path: &Path,
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
            .with_context(|| format!("reading PuLID adapter {}", path.display()))?
        };
        Self::from_var_builder(vb, config)
            .with_context(|| format!("loading PuLID adapter {}", path.display()))
    }

    /// Build the stack from a `VarBuilder` rooted at the file's top level.
    ///
    /// The checkpoint's inventory is checked against the layout the UNet config
    /// implies BEFORE anything is read, in both directions: every planned
    /// module must be present, and no `id_adapter_attn_layers.<i>` the plan did
    /// not name may exist. A one-directional check would accept the FLUX
    /// checkpoint's silence as agreement, and an SD1.5-shaped plan against the
    /// SDXL file would load 16 of its 70 modules and render an image
    /// conditioned on a fifth of the face.
    pub fn from_var_builder(vb: VarBuilder, config: &UNet2DConditionModelConfig) -> Result<Self> {
        let sites = plan_attn_layers(config);
        if sites.is_empty() {
            bail!("this UNet has no cross-attention modules to condition");
        }
        let planned: std::collections::BTreeSet<usize> =
            sites.iter().map(|site| site.processor_index).collect();
        let highest = sites
            .iter()
            .map(|site| site.processor_index)
            .max()
            .unwrap_or(0);
        // Scan a little PAST the highest planned index. Without the margin an
        // SD1.5-shaped plan reads the SDXL checkpoint's first 16 modules — its
        // planned indices are a prefix of the SDXL file's — and every orphan
        // sits above the bound where nothing looks for it. The margin is two
        // full transformer blocks' worth of processor slots, and the
        // checkpoint's indices are dense in `attn1`/`attn2` pairs, so a real
        // adapter never has that much silence in the middle.
        const ORPHAN_SCAN_MARGIN: usize = 4;
        let mut missing = Vec::new();
        let mut unexpected = Vec::new();
        for index in 0..=highest.saturating_add(ORPHAN_SCAN_MARGIN) {
            let present = vb.contains_tensor(&format!("{ADAPTER_PREFIX}.{index}.id_to_k.weight"));
            match (planned.contains(&index), present) {
                (true, false) => missing.push(index),
                (false, true) => unexpected.push(index),
                _ => {}
            }
        }
        if !missing.is_empty() || !unexpected.is_empty() {
            bail!(
                "PuLID adapter does not match this UNet: {} planned cross-attention modules, \
                 {} missing ({:?}), {} unexpected ({:?}); this is not a pulid_v1.1 checkpoint \
                 for this architecture",
                sites.len(),
                missing.len(),
                missing.iter().take(8).collect::<Vec<_>>(),
                unexpected.len(),
                unexpected.iter().take(8).collect::<Vec<_>>(),
            );
        }

        let dtype = vb.dtype();
        let mut layers = Vec::with_capacity(sites.len());
        for site in sites {
            layers.push(
                IdAttnLayer::load(
                    site,
                    vb.pp(format!("{ADAPTER_PREFIX}.{}", site.processor_index)),
                )
                .with_context(|| {
                    format!(
                        "loading {ADAPTER_PREFIX}.{} (hook index {})",
                        site.processor_index, site.hook_index
                    )
                })?,
            );
        }
        Ok(Self { layers, dtype })
    }

    /// Number of hooked cross-attention modules.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Always false in practice — an empty adapter is refused at load — but
    /// clippy asks for it beside `len`.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The dtype the modules were loaded at.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// One module, by its hook index. Exposed so a parity test can drive a
    /// single module against the upstream golden without a UNet.
    pub fn layer(&self, hook_index: usize) -> Option<&IdAttnLayer> {
        self.layers.get(hook_index)
    }

    /// The module whose weights live at `id_adapter_attn_layers.<index>`.
    pub fn layer_for_processor(&self, processor_index: usize) -> Option<&IdAttnLayer> {
        self.layers
            .iter()
            .find(|layer| layer.site.processor_index == processor_index)
    }

    /// Bytes this stack occupies on its device.
    ///
    /// Not a nicety: the adapter is device-resident and nothing else in the
    /// engine accounts for it, so whatever classifies the engine's residency
    /// has to be able to see it. 681,574,400 bytes at SDXL's geometry in
    /// f16/bf16 —
    /// `mold_server::memory_preflight::IDENTITY_SDXL_VRAM_OVERHEAD_BYTES` is
    /// pinned against this arithmetic.
    ///
    /// Computed from the geometry rather than by walking the tensors, because
    /// the two linears are bias-free and their shapes are exactly the layer
    /// table's.
    pub fn resident_bytes(&self) -> u64 {
        let elements: usize = self
            .layers
            .iter()
            .map(|layer| 2 * layer.site.hidden_size * ID_TOKEN_DIM)
            .sum();
        elements as u64 * self.dtype.size_in_bytes() as u64
    }
}

/// The identity signal a render conditions on.
///
/// A newtype rather than a bare `Tensor` so the `[*, 32, 2048]` contract is
/// checked once, at the boundary, instead of at each of the seventy injection
/// sites.
#[derive(Debug, Clone)]
pub struct SdxlIdentityEmbedding {
    tensor: Tensor,
}

impl SdxlIdentityEmbedding {
    /// Wrap a tensor, accepting `[32, 2048]` or `[1, 32, 2048]`.
    pub fn new(tensor: Tensor) -> Result<Self> {
        let tensor = match tensor.dims() {
            [ID_TOKENS, ID_TOKEN_DIM] => tensor.unsqueeze(0)?,
            [1, ID_TOKENS, ID_TOKEN_DIM] => tensor,
            other => bail!(
                "PuLID identity embedding must be [{ID_TOKENS}, {ID_TOKEN_DIM}] or \
                 [1, {ID_TOKENS}, {ID_TOKEN_DIM}], got {other:?}"
            ),
        };
        Ok(Self {
            tensor: tensor.to_dtype(DType::F32)?,
        })
    }

    /// Rehydrate the identity admission froze for this request.
    pub fn from_frozen(frozen: &mold_core::identity::FrozenIdentityEmbedding) -> Result<Self> {
        Self::new(Tensor::from_vec(
            frozen.values(),
            (1, ID_TOKENS, ID_TOKEN_DIM),
            &Device::Cpu,
        )?)
    }

    /// Rehydrate the UNCONDITIONAL identity frozen beside the real one.
    ///
    /// SDXL needs it for every classifier-free render, which is every render at
    /// a guidance above 1 — unlike FLUX, where it is the true-CFG opt-in.
    pub fn uncond_from_frozen(
        frozen: &mold_core::identity::FrozenIdentityEmbedding,
    ) -> Result<Option<Self>> {
        let Some(values) = frozen.uncond_values() else {
            return Ok(None);
        };
        Ok(Some(Self::new(Tensor::from_vec(
            values,
            (1, ID_TOKENS, ID_TOKEN_DIM),
            &Device::Cpu,
        )?)?))
    }

    /// The `[1, 32, 2048]` tensor, on the CPU in f32.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

/// Identity conditioning frozen for one render.
#[derive(Debug, Clone)]
pub struct SdxlPulidContext {
    /// `[batch, 32, 2048]` on the UNet's device, in its working dtype. `batch`
    /// matches the denoise batch exactly: 2 (`[uncond, cond]`) under
    /// classifier-free guidance, 1 without it.
    pub id_embeds: Tensor,
    pub id_weight: f32,
    pub start_step: usize,
}

impl SdxlPulidContext {
    /// Build the batched conditioning one render drives.
    ///
    /// `uncond` is required exactly when the render runs a CFG batch, because
    /// mold's SDXL denoise concatenates `[uncond, cond]` into ONE forward while
    /// upstream runs two. Passing `None` there is an error rather than a
    /// broadcast: `[1, 32, 2048]` against a batch-2 query broadcasts the
    /// conditional identity onto the negative branch, which cancels most of the
    /// identity out of the guided result without failing.
    pub fn new(
        embedding: &SdxlIdentityEmbedding,
        uncond: Option<&SdxlIdentityEmbedding>,
        id_weight: f32,
        start_step: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cond = embedding.tensor().to_device(device)?.to_dtype(dtype)?;
        let id_embeds = match uncond {
            None => cond,
            Some(uncond) => {
                let uncond = uncond.tensor().to_device(device)?.to_dtype(dtype)?;
                // `[uncond, cond]`, the order `denoise_loop` concatenates its
                // latents and its text embeddings in.
                Tensor::cat(&[&uncond, &cond], 0)?
            }
        };
        Ok(Self {
            id_embeds,
            id_weight,
            start_step,
        })
    }

    /// Whether this context contributes anything at `step`.
    pub fn active_at_step(&self, step: usize) -> bool {
        self.id_weight != 0.0 && step >= self.start_step
    }
}

/// The adapter and the context, paired for the length of one denoise loop.
#[derive(Debug, Clone, Copy)]
pub struct SdxlPulidRuntime<'a> {
    adapter: &'a SdxlPulidAdapter,
    context: &'a SdxlPulidContext,
}

impl<'a> SdxlPulidRuntime<'a> {
    pub fn new(adapter: &'a SdxlPulidAdapter, context: &'a SdxlPulidContext) -> Self {
        Self { adapter, context }
    }

    /// The hook for `step`, or `None` when identity contributes nothing.
    ///
    /// `None` is the whole gate. The denoise loop answers it by calling the
    /// UNet's ordinary `forward`, so a gated step executes the exact code a
    /// build with no identity request executes — which is what makes
    /// bit-identity structural rather than numerical.
    pub fn hook_for_step(&self, step: usize) -> Option<SdxlPulidHook<'a>> {
        self.context.active_at_step(step).then_some(SdxlPulidHook {
            adapter: self.adapter,
            context: self.context,
        })
    }

    pub fn adapter(&self) -> &'a SdxlPulidAdapter {
        self.adapter
    }

    pub fn context(&self) -> &'a SdxlPulidContext {
        self.context
    }
}

/// Injects identity features into every text cross-attention.
#[derive(Debug, Clone, Copy)]
pub struct SdxlPulidHook<'a> {
    adapter: &'a SdxlPulidAdapter,
    context: &'a SdxlPulidContext,
}

impl CrossAttentionHook for SdxlPulidHook<'_> {
    fn cross_attention(
        &self,
        index: usize,
        query: &Tensor,
        attended: &Tensor,
        heads: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        let layer = self.adapter.layer(index).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "PuLID identity layer {index} is out of range ({} loaded); the adapter was built \
                 for a different UNet than the one running",
                self.adapter.len()
            ))
        })?;
        // The UNet is the authority on its own head count. A disagreement means
        // the layer table was derived from a different config than the one that
        // built the modules, and the reshape below would silently produce a
        // valid-shaped, wrong answer.
        if layer.site.heads != heads {
            return Err(candle_core::Error::Msg(format!(
                "PuLID identity layer {index} was planned for {} heads but the UNet reports \
                 {heads}",
                layer.site.heads
            )));
        }
        let id_embeds = broadcast_identity(&self.context.id_embeds, query.dim(0)?)?;
        layer
            .inject(&id_embeds, query, attended, self.context.id_weight)
            .map(Some)
    }
}

/// Match the identity batch to the UNet's.
///
/// The context is built at the render's batch, so this is normally the identity
/// function. It exists for the one case the denoise loop cannot rule out
/// statically — a UNet forward whose batch differs from the one the context was
/// frozen at — where a single-entry identity is repeated rather than left to
/// broadcast implicitly, and anything else is an error rather than a guess.
fn broadcast_identity(id_embeds: &Tensor, batch: usize) -> candle_core::Result<Tensor> {
    let have = id_embeds.dim(0)?;
    if have == batch {
        return Ok(id_embeds.clone());
    }
    if have == 1 {
        return id_embeds
            .expand((batch, ID_TOKENS, ID_TOKEN_DIM))?
            .contiguous();
    }
    Err(candle_core::Error::Msg(format!(
        "PuLID identity embedding carries {have} batch entries but the UNet forward is running \
         {batch}"
    )))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::Var;
    use candle_nn::VarMap;
    use candle_transformers::models::stable_diffusion::attention::HookCursor;
    use candle_transformers::models::stable_diffusion::unet_2d::{
        BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
    };
    use std::sync::Mutex;

    fn block(out_channels: usize, use_cross_attn: Option<usize>, heads: usize) -> BlockConfig {
        BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim: heads,
        }
    }

    /// SDXL's published UNet geometry
    /// (`stabilityai/stable-diffusion-xl-base-1.0/unet/config.json`), which is
    /// what `StableDiffusionConfig::sdxl` builds.
    pub(crate) fn sdxl_config() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            blocks: vec![
                block(320, None, 5),
                block(640, Some(2), 10),
                block(1280, Some(10), 20),
            ],
            center_input_sample: false,
            cross_attention_dim: 2048,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: None,
            use_linear_projection: true,
        }
    }

    /// SD1.5's UNet geometry, pinned only so the layout arithmetic is exercised
    /// on a second shape. No PuLID-SD1.5 checkpoint exists.
    fn sd15_config() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            blocks: vec![
                block(320, Some(1), 8),
                block(640, Some(1), 8),
                block(1280, Some(1), 8),
                block(1280, None, 8),
            ],
            cross_attention_dim: 768,
            ..UNet2DConditionModelConfig::default()
        }
    }

    /// A UNet small enough to construct on a CPU in milliseconds, with the same
    /// SHAPE of layout SDXL has: one attention-free block, one cross-attention
    /// block with more than one transformer layer, and a mid block.
    fn tiny_config() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            blocks: vec![block(32, None, 2), block(64, Some(2), 4)],
            center_input_sample: false,
            cross_attention_dim: 2048,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 1,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: None,
            use_linear_projection: true,
        }
    }

    fn layer_map_fixture(name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/pulid_sdxl")
            .join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        serde_json::from_str(&text).expect("the layer map is JSON")
    }

    /// Every `attn2` entry of a committed layer map, in traversal order.
    fn fixture_attn2(name: &str) -> Vec<(usize, usize, usize, String)> {
        layer_map_fixture(name)["layers"]
            .as_array()
            .expect("layers is an array")
            .iter()
            .filter(|entry| entry["kind"] == "attn2")
            .map(|entry| {
                (
                    entry["processor_index"].as_u64().unwrap() as usize,
                    entry["hidden_size"].as_u64().unwrap() as usize,
                    entry["heads"].as_u64().unwrap() as usize,
                    entry["module_name"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    /// The permutation is the whole correctness argument, so it is pinned
    /// against the checkpoint's own tensor inventory rather than against
    /// itself. `attn_layer_map.json` was captured by enumerating the REAL
    /// `diffusers.UNet2DConditionModel.attn_processors`, which is the traversal
    /// `hack_unet_attn_layers` walks and therefore the index PuLID's checkpoint
    /// keys are in.
    #[test]
    fn the_sdxl_layer_table_matches_upstreams_own_processor_enumeration() {
        let sites = plan_attn_layers(&sdxl_config());
        let fixture = fixture_attn2("attn_layer_map.json");
        assert_eq!(sites.len(), 70);
        assert_eq!(fixture.len(), 70);

        // The fixture is in DIFFUSERS order; the plan is in CANDLE order. They
        // must be the same SET, matched on the checkpoint index.
        let mut by_processor: std::collections::BTreeMap<usize, &AttnLayerSite> =
            std::collections::BTreeMap::new();
        for site in &sites {
            assert_eq!(site.hook_index, sites[site.hook_index].hook_index);
            assert!(
                by_processor.insert(site.processor_index, site).is_none(),
                "processor index {} planned twice",
                site.processor_index
            );
        }
        for (processor_index, hidden_size, heads, module_name) in &fixture {
            let site = by_processor.get(processor_index).unwrap_or_else(|| {
                panic!("{module_name} (index {processor_index}) was not planned")
            });
            assert_eq!(site.hidden_size, *hidden_size, "{module_name}");
            assert_eq!(site.heads, *heads, "{module_name}");
            assert_eq!(site.dim_head(), 64, "{module_name}");
        }

        // And the ORDER is a genuine permutation, not the identity: candle
        // visits the mid block second, diffusers registers it last.
        let hook_order: Vec<usize> = sites.iter().map(|site| site.processor_index).collect();
        let diffusers_order: Vec<usize> = fixture.iter().map(|(index, ..)| *index).collect();
        assert_ne!(
            hook_order, diffusers_order,
            "reading id_adapter_attn_layers.<hook_index> would be a silent mis-wiring, so the \
             two orders must actually differ"
        );
        // The three golden layers, spelled out: down block 1's first attn2 is
        // hook 0, the mid block's first is hook 24, up block 0's first is
        // hook 34.
        assert_eq!(sites[0].processor_index, 1);
        assert_eq!(sites[24].processor_index, 121);
        assert_eq!(sites[34].processor_index, 49);
    }

    #[test]
    fn the_sd15_layer_table_matches_upstreams_own_processor_enumeration() {
        let sites = plan_attn_layers(&sd15_config());
        let fixture = fixture_attn2("attn_layer_map_sd15.json");
        assert_eq!(sites.len(), 16);
        assert_eq!(fixture.len(), 16);

        let planned: std::collections::BTreeSet<usize> =
            sites.iter().map(|site| site.processor_index).collect();
        let captured: std::collections::BTreeSet<usize> =
            fixture.iter().map(|(index, ..)| *index).collect();
        assert_eq!(planned, captured);
        for (processor_index, hidden_size, _, module_name) in &fixture {
            let site = sites
                .iter()
                .find(|site| site.processor_index == *processor_index)
                .unwrap();
            assert_eq!(site.hidden_size, *hidden_size, "{module_name}");
        }
    }

    /// Records the `(index, heads)` sequence a real UNet forward hands the
    /// hook, so the plan is checked against candle's own traversal rather than
    /// against a reading of it.
    #[derive(Default)]
    struct RecordingHook {
        seen: Mutex<Vec<(usize, usize, Vec<usize>)>>,
    }

    impl CrossAttentionHook for RecordingHook {
        fn cross_attention(
            &self,
            index: usize,
            query: &Tensor,
            _attended: &Tensor,
            heads: usize,
        ) -> candle_core::Result<Option<Tensor>> {
            self.seen
                .lock()
                .unwrap()
                .push((index, heads, query.dims().to_vec()));
            Ok(None)
        }
    }

    /// The hook index a module receives IS its position in candle's traversal,
    /// and `plan_attn_layers` reproduces that order and those head counts
    /// exactly. Hermetic: a tiny UNet on zeroed weights, no checkpoint.
    #[test]
    fn the_plan_reproduces_candles_own_traversal_order() {
        let device = Device::Cpu;
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let unet = UNet2DConditionModel::new(vb, 4, 4, false, config.clone())
            .expect("a tiny UNet constructs on zeroed weights");

        let hook = RecordingHook::default();
        let xs = Tensor::zeros((1, 4, 16, 16), DType::F32, &device).unwrap();
        let context =
            Tensor::zeros((1, 7, config.cross_attention_dim), DType::F32, &device).unwrap();
        unet.forward_with_hook(&xs, 1.0, &context, &hook)
            .expect("the tiny UNet runs");

        let seen = hook.seen.lock().unwrap().clone();
        let plan = plan_attn_layers(&config);
        assert_eq!(seen.len(), plan.len(), "{seen:?}");
        for (position, (index, heads, query_dims)) in seen.iter().enumerate() {
            assert_eq!(*index, position, "the cursor must count in traversal order");
            assert_eq!(plan[position].hook_index, position);
            assert_eq!(
                plan[position].heads, *heads,
                "hook {position}: the plan and the UNet disagree on the head count"
            );
            assert_eq!(
                query_dims[2], plan[position].hidden_size,
                "hook {position}: the plan and the UNet disagree on the width"
            );
        }

        // The tiny geometry's own permutation, worked out from the two
        // registration orders: down(2), mid(2), up(4) in candle; down(2),
        // up(4), mid(2) in diffusers.
        let processors: Vec<usize> = plan.iter().map(|site| site.processor_index).collect();
        assert_eq!(processors, vec![1, 3, 13, 15, 5, 7, 9, 11]);
    }

    /// Build a synthetic adapter for `config` without touching a checkpoint.
    pub(crate) fn synthetic_adapter(
        config: &UNet2DConditionModelConfig,
        device: &Device,
    ) -> SdxlPulidAdapter {
        let varmap = VarMap::new();
        {
            let mut data = varmap.data().lock().unwrap();
            for site in plan_attn_layers(config) {
                for name in ["id_to_k", "id_to_v"] {
                    let tensor =
                        Tensor::zeros((site.hidden_size, ID_TOKEN_DIM), DType::F32, device)
                            .unwrap();
                    data.insert(
                        format!("{ADAPTER_PREFIX}.{}.{name}.weight", site.processor_index),
                        Var::from_tensor(&tensor).unwrap(),
                    );
                }
            }
        }
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        SdxlPulidAdapter::from_var_builder(vb, config).expect("the synthetic adapter loads")
    }

    #[test]
    fn a_checkpoint_that_does_not_match_the_unet_is_refused_by_name() {
        let device = Device::Cpu;
        // An SDXL-shaped adapter against an SD1.5-shaped UNet: the file carries
        // 70 modules, the plan names 16, and every unplanned index is an
        // orphan. A one-directional presence check would accept this.
        let varmap = VarMap::new();
        {
            let mut data = varmap.data().lock().unwrap();
            for site in plan_attn_layers(&sdxl_config()) {
                for name in ["id_to_k", "id_to_v"] {
                    let tensor =
                        Tensor::zeros((site.hidden_size, ID_TOKEN_DIM), DType::F32, &device)
                            .unwrap();
                    data.insert(
                        format!("{ADAPTER_PREFIX}.{}.{name}.weight", site.processor_index),
                        Var::from_tensor(&tensor).unwrap(),
                    );
                }
            }
        }
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let error = SdxlPulidAdapter::from_var_builder(vb, &sd15_config())
            .expect_err("an SDXL adapter is not an SD1.5 adapter");
        let message = format!("{error:#}");
        assert!(message.contains("unexpected"), "{message}");

        // And an empty file is refused for the modules it is missing.
        let empty = VarBuilder::zeros(DType::F32, &device);
        let error = SdxlPulidAdapter::from_var_builder(empty, &sdxl_config())
            .expect_err("a file with no adapter tensors is not a PuLID checkpoint");
        assert!(format!("{error:#}").contains("missing"), "{error:#}");
    }

    /// The memory charge is derived from the layer table, so the two cannot
    /// drift. 681,574,400 bytes is `2 x 2048 x (10 x 640 + 60 x 1280) x 2`.
    #[test]
    fn the_sdxl_adapters_resident_bytes_are_the_layer_tables_own_arithmetic() {
        let sites = plan_attn_layers(&sdxl_config());
        let narrow = sites.iter().filter(|site| site.hidden_size == 640).count();
        let wide = sites.iter().filter(|site| site.hidden_size == 1280).count();
        assert_eq!((narrow, wide), (10, 60));

        let elements: usize = sites
            .iter()
            .map(|site| 2 * site.hidden_size * ID_TOKEN_DIM)
            .sum();
        assert_eq!(elements, 340_787_200);
        assert_eq!(elements * 2, 681_574_400, "f16/bf16 bytes");
        assert_eq!(elements * 4, 1_363_148_800, "f32 bytes");
    }

    fn embedding(value: f32, device: &Device) -> SdxlIdentityEmbedding {
        SdxlIdentityEmbedding::new(
            Tensor::full(value, (1, ID_TOKENS, ID_TOKEN_DIM), device).unwrap(),
        )
        .unwrap()
    }

    /// Upstream runs two UNet passes and gives the negative one the
    /// unconditional identity; mold runs one batched `[uncond, cond]` pass, so
    /// the embedding has to be batched in the same order.
    #[test]
    fn the_cfg_batch_carries_the_unconditional_identity_first() {
        let device = Device::Cpu;
        let cond = embedding(1.0, &device);
        let uncond = embedding(-1.0, &device);

        let batched =
            SdxlPulidContext::new(&cond, Some(&uncond), 1.0, 0, &device, DType::F32).unwrap();
        assert_eq!(batched.id_embeds.dims(), &[2, ID_TOKENS, ID_TOKEN_DIM]);
        let rows = batched.id_embeds.to_vec3::<f32>().unwrap();
        assert_eq!(rows[0][0][0], -1.0, "row 0 is the unconditional branch");
        assert_eq!(rows[1][0][0], 1.0, "row 1 is the conditional branch");

        let single = SdxlPulidContext::new(&cond, None, 1.0, 0, &device, DType::F32).unwrap();
        assert_eq!(single.id_embeds.dims(), &[1, ID_TOKENS, ID_TOKEN_DIM]);
    }

    /// The gate is control flow, not arithmetic: an inactive step yields no
    /// hook at all, so the denoise loop runs the UNet's ordinary `forward`.
    #[test]
    fn an_inactive_step_yields_no_hook() {
        let device = Device::Cpu;
        let adapter = synthetic_adapter(&tiny_config(), &device);
        let cond = embedding(1.0, &device);

        let delayed = SdxlPulidContext::new(&cond, None, 1.0, 3, &device, DType::F32).unwrap();
        let runtime = SdxlPulidRuntime::new(&adapter, &delayed);
        for step in 0..3 {
            assert!(runtime.hook_for_step(step).is_none(), "step {step}");
        }
        assert!(runtime.hook_for_step(3).is_some());

        let zero = SdxlPulidContext::new(&cond, None, 0.0, 0, &device, DType::F32).unwrap();
        let runtime = SdxlPulidRuntime::new(&adapter, &zero);
        for step in 0..8 {
            assert!(
                runtime.hook_for_step(step).is_none(),
                "a zero weight must never install a hook"
            );
        }
    }

    /// A hook driven over a real UNet forward must leave the output untouched
    /// when its weights are zero — the numerical mirror of the control-flow
    /// gate above, and what proves the injection is additive rather than
    /// replacing the text attention.
    #[test]
    fn a_zero_weighted_adapter_leaves_the_unet_output_unchanged() {
        let device = Device::Cpu;
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let unet = UNet2DConditionModel::new(vb, 4, 4, false, config.clone()).unwrap();
        let adapter = synthetic_adapter(&config, &device);
        let cond = embedding(1.0, &device);
        let context = SdxlPulidContext::new(&cond, None, 1.0, 0, &device, DType::F32).unwrap();
        let runtime = SdxlPulidRuntime::new(&adapter, &context);
        let hook = runtime.hook_for_step(0).expect("an active step");

        let xs = Tensor::ones((1, 4, 16, 16), DType::F32, &device).unwrap();
        let encoder =
            Tensor::ones((1, 7, config.cross_attention_dim), DType::F32, &device).unwrap();
        let plain = unet.forward(&xs, 1.0, &encoder).unwrap();
        let hooked = unet.forward_with_hook(&xs, 1.0, &encoder, &hook).unwrap();
        let difference = (&plain - &hooked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            difference, 0.0,
            "zero id_to_k/id_to_v weights inject nothing at all"
        );

        // And the cursor really did visit every module: a hook that silently
        // did nothing would pass the assertion above for the wrong reason.
        let cursor = HookCursor::new(&hook);
        assert_eq!(cursor.visited(), 0);
    }

    /// Parity for the identity branch against upstream's own
    /// `IDAttnProcessor2_0`, on the pinned `pulid_v1.1.safetensors`.
    ///
    /// Weight-gated, mirroring `tests/pulid_adapter_parity.rs`:
    ///
    /// ```text
    /// MOLD_TEST_PULID_ASSETS=/path/to/pulid \
    ///   cargo test --release -p mold-ai-inference --features pulid \
    ///     --lib sdxl::pulid -- --ignored --nocapture --test-threads=1
    /// ```
    ///
    /// The goldens come from
    /// `testdata/pulid_sdxl/capture_attn_goldens.py`, which asserts upstream's
    /// `NUM_ZERO`/`ORTHO`/`ORTHO_v2` module globals are still
    /// `0`/`False`/`False` before it captures anything — so these numbers are
    /// the branch this checkpoint was trained under, not a variant.
    mod golden_parity {
        use super::*;
        use crate::pulid_fixtures::{
            gather_probe, pulid_asset, scale_relative_error, sdxl_golden, DeterministicStream,
            GoldenStats, SEED_SDXL_ATTN_ATTENDED, SEED_SDXL_ATTN_ID, SEED_SDXL_ATTN_PROBE,
            SEED_SDXL_ATTN_QUERY,
        };

        const GOLDEN_FILE: &str = "attn_goldens.safetensors";
        /// `capture_attn_goldens.py`'s `BATCH` and `SEQ`.
        const BATCH: usize = 2;
        const SEQ: usize = 64;
        /// The three layers the capture chose: one per UNet region, covering
        /// both widths.
        const LAYERS: [usize; 3] = [1, 121, 49];

        /// The README measures f16 INPUT sensitivity at 1.4e-4 to 2.0e-4
        /// relative; a port compared against the f32 golden must sit far
        /// below it. FLUX's `PerceiverAttentionCA` golden lands at 1.3e-5
        /// relative, and this arithmetic is strictly simpler — two bias-free
        /// linears and one attention — so the budget is set an order of
        /// magnitude under the input-sensitivity floor. A transposed
        /// projection, a wrong head split, or a missing scale moves it by
        /// whole percent.
        const TOLERANCE: f32 = 2.0e-5;

        fn adapter() -> SdxlPulidAdapter {
            let path = pulid_asset("pulid_v1.1.safetensors");
            SdxlPulidAdapter::load(&path, &sdxl_config(), DType::F32, &Device::Cpu)
                .expect("the pinned v1.1 adapter loads against SDXL's UNet")
        }

        fn assert_golden(label: &str, probe_seed: u64, actual: &Tensor) {
            let stats = GoldenStats::load_sdxl(GOLDEN_FILE, &format!("{label}.stats"));
            stats.assert_matches(&GoldenStats::measure(actual), 1e-4, label);

            let expected = sdxl_golden(GOLDEN_FILE, &format!("{label}.probe"))
                .to_vec1::<f32>()
                .unwrap();
            let observed = gather_probe(actual, probe_seed);
            let error = scale_relative_error(&observed, &expected, stats.peak);
            println!("{label}: {error:.3e} of the {} scale", stats.peak);
            assert!(error < TOLERANCE, "{label} drifted by {error}");
        }

        #[test]
        #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
        fn the_identity_branch_matches_upstream() {
            let device = Device::Cpu;
            let adapter = adapter();
            assert_eq!(adapter.len(), 70);

            for processor_index in LAYERS {
                let layer = adapter
                    .layer_for_processor(processor_index)
                    .unwrap_or_else(|| panic!("processor {processor_index} is loaded"));
                let site = layer.site();
                let idx = processor_index as u64;

                let query = DeterministicStream::new(SEED_SDXL_ATTN_QUERY + idx)
                    .tensor(&[BATCH, SEQ, site.hidden_size], &device);
                let attended = DeterministicStream::new(SEED_SDXL_ATTN_ATTENDED + idx)
                    .tensor(&[BATCH, SEQ, site.hidden_size], &device);
                let id_embeds = DeterministicStream::new(SEED_SDXL_ATTN_ID + idx)
                    .tensor(&[BATCH, ID_TOKENS, ID_TOKEN_DIM], &device);

                let id_hidden = layer.id_hidden_states(&id_embeds, &query).unwrap();
                assert_eq!(id_hidden.dims(), &[BATCH, SEQ, site.hidden_size]);
                assert_golden(
                    &format!("attn{processor_index}.id_hidden_states"),
                    SEED_SDXL_ATTN_PROBE + idx + 1,
                    &id_hidden,
                );

                for (scale, tag) in [(1.0_f32, "s1p0"), (0.7_f32, "s0p7")] {
                    let combined = layer.inject(&id_embeds, &query, &attended, scale).unwrap();
                    assert_golden(
                        &format!("attn{processor_index}.combined_{tag}"),
                        SEED_SDXL_ATTN_PROBE + idx,
                        &combined,
                    );
                }
            }
        }

        /// The adapter the pinned checkpoint carries is exactly the one SDXL's
        /// UNet asks for — no missing modules, no orphans, and the layer table
        /// the load used is the one `plan_attn_layers` derives.
        #[test]
        #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
        fn the_pinned_checkpoint_carries_exactly_the_planned_modules() {
            let adapter = adapter();
            let planned = plan_attn_layers(&sdxl_config());
            assert_eq!(adapter.len(), planned.len());
            for site in &planned {
                let loaded = adapter
                    .layer(site.hook_index)
                    .expect("every hook is loaded");
                assert_eq!(loaded.site(), *site);
            }
            // f32 here because the test loads at f32; production loads at the
            // UNet's f16/bf16, which is the figure the memory charge uses.
            assert_eq!(adapter.resident_bytes(), 1_363_148_800);
        }
    }

    /// A hook built for one UNet must refuse another rather than reading a
    /// neighbouring layer's projections.
    #[test]
    fn a_hook_refuses_an_index_or_head_count_it_was_not_planned_for() {
        let device = Device::Cpu;
        let adapter = synthetic_adapter(&tiny_config(), &device);
        let cond = embedding(1.0, &device);
        let context = SdxlPulidContext::new(&cond, None, 1.0, 0, &device, DType::F32).unwrap();
        let runtime = SdxlPulidRuntime::new(&adapter, &context);
        let hook = runtime.hook_for_step(0).unwrap();

        let site = adapter.layer(0).unwrap().site();
        let query = Tensor::zeros((1, 5, site.hidden_size), DType::F32, &device).unwrap();
        let attended = query.clone();

        let error = hook
            .cross_attention(adapter.len(), &query, &attended, site.heads)
            .expect_err("an out-of-range index is an error");
        assert!(format!("{error}").contains("out of range"), "{error}");

        let error = hook
            .cross_attention(0, &query, &attended, site.heads + 1)
            .expect_err("a head-count disagreement is an error");
        assert!(format!("{error}").contains("heads"), "{error}");
    }
}
