//! Image-prompt (IP-Adapter) conditioning state for the SD1.5 and SDXL
//! engines.
//!
//! ## Why ONE module for two families
//!
//! [`crate::sdxl::identity`] is the template this mirrors, and the reason it
//! is a per-family module is that PuLID's contract IS per-family: the FLUX
//! adapter takes a true-CFG branch the SDXL one refuses, and the SDXL one
//! requires an unconditional embedding the FLUX one treats as opt-in. None of
//! that is true here. [`crate::ip_adapter`]'s own header records why — "the
//! only thing that differs between SD1.5 and SDXL is the UNet's
//! `cross_attention_dim` and its `attn2` geometry, and both come out of the
//! [`UNet2DConditionModelConfig`] the caller already has" — so a second copy
//! of this file under `sd15/` and `sdxl/` would be two identical residency
//! policies that can drift, keyed on a distinction the arithmetic does not
//! make. It sits beside [`crate::sd_attn_layout`] for the same reason that
//! module does: what it describes is a fact about an SD-lineage UNet, not
//! about one engine.
//!
//! ## Residency: two artifacts, two lifetimes
//!
//! The bundle is a ~2.5 GB CLIP-ViT-H/14 tower and a small stack of per-layer
//! key/value projections (44 MB on SD1.5, ~700 MB on SDXL). They are NOT
//! resident on the same schedule, and treating them as one bundle is the
//! mistake this module exists to avoid:
//!
//! * The **tower** runs exactly once per request, before the denoise, and
//!   produces a `[1, 1024]` embedding. It is built, forwarded, and dropped
//!   inside [`SdReferenceState::encode`] — the drop-and-reload discipline the
//!   text encoders and the PuLID face stack already follow, and for a much
//!   sharper reason here: 2.5 GB held across a denoise loop is 2.5 GB the UNet
//!   activations and the VAE decode's conv2d intermediates have to compete
//!   with, on a family whose whole appeal is fitting small cards.
//! * The **projections** are touched at every one of the UNet's 16 (SD1.5) or
//!   70 (SDXL) cross-attention modules on every step, so they stay resident
//!   across matching requests exactly as the PuLID adapter does, keyed on
//!   device, dtype, and module count.
//!
//! Release follows [`crate::sdxl::identity`]'s rule verbatim, and the reason
//! is not optional: `ModelCache` parks an engine by calling `unload()`, which
//! zeroes the entry's `vram_bytes` while keeping the engine cached, so an
//! adapter that survived parking is device memory nothing accounts for.
//!
//! ## The zero-weight falsification
//!
//! An effective weight of 0, or a request carrying no reference picture at
//! all, plans nothing, decodes nothing, loads nothing, and drops whatever was
//! resident. [`ResolvedReference::runtime`]'s `hook_for_step` then yields
//! `None` for a zero scale as a second, arithmetic-independent gate, and the
//! denoise loop answers `None` by calling the UNet's ordinary `forward` — so
//! bit-identity with an unreferenced render is STRUCTURAL, the same argument
//! `id_weight: 0` rests on, pinned by
//! `a_null_hook_is_bit_identical_to_the_plain_forward` below.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::attention::CrossAttentionHook;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig;
use mold_core::generation_profile::reference_images_for_recipe;
use mold_core::ip_adapter_assets::{ImagePromptFamily, IpAdapterPaths};
use mold_core::GenerateRequest;

use crate::encoders::clip_image_preprocess::preprocess_rgb8;
use crate::encoders::openclip_vision::OpenClipVisionTower;
use crate::ip_adapter::{IpAdapter, IpAdapterContext, IpAdapterRuntime};
use crate::progress::ProgressReporter;
use crate::sd_attn_layout::plan_attn_layers;

/// What a request asked for, once the recipe's advertised default is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReferenceRequest {
    pub(crate) weight: f32,
}

/// Read the image-prompt fields off a request.
///
/// `None` covers the three cases that must plan nothing, load nothing, and
/// render exactly what a request with no reference picture renders: no
/// `edit_images` at all, an empty list, and an explicit `reference_weight` of
/// 0.
///
/// The default comes from [`reference_images_for_recipe`] rather than from a
/// constant read directly, because that profile is what the host ADVERTISED
/// to the client as `capabilities.reference_images.weight` — the sentence in
/// [`GenerateRequest::reference_weight`]'s own doc comment is "absent means
/// the recipe's advertised default", and reading the advertisement is the only
/// way that stays true if the bounds are ever retuned. A recipe whose profile
/// carries no weight has no adapter to drive, so it conditions nothing.
pub(crate) fn reference_request(
    req: &GenerateRequest,
    family: ImagePromptFamily,
    model: &str,
) -> Option<ReferenceRequest> {
    let images = req.edit_images.as_ref()?;
    if images.iter().all(Vec::is_empty) {
        return None;
    }
    let weight = req.reference_weight.or_else(|| {
        reference_images_for_recipe(family.family(), model)
            .weight
            .map(|control| control.default)
    })?;
    if weight == 0.0 {
        return None;
    }
    Some(ReferenceRequest {
        weight: weight as f32,
    })
}

/// The adapter plus this request's projected image tokens, alive for one
/// denoise loop.
#[derive(Debug)]
pub(crate) struct ResolvedReference {
    adapter: Arc<IpAdapter>,
    context: IpAdapterContext,
}

impl ResolvedReference {
    pub(crate) fn runtime(&self) -> IpAdapterRuntime<'_> {
        IpAdapterRuntime::new(&self.adapter, &self.context)
    }

    /// Cross-attention modules the render is driving, for the progress line.
    pub(crate) fn module_count(&self) -> usize {
        self.adapter.module_count()
    }

    /// Image tokens one reference picture became, for the progress line.
    ///
    /// Read off the checkpoint rather than assumed, so a Plus adapter's 16
    /// reports 16 — see [`IpAdapter::tokens`].
    pub(crate) fn tokens(&self) -> usize {
        self.adapter.tokens()
    }
}

/// A resident projection stack and the exact shape it was built for.
struct ResidentAdapter {
    adapter: Arc<IpAdapter>,
    device: Device,
    dtype: DType,
    /// The module count the UNet config implied when this was built. Cheaper
    /// to compare than the whole config and sufficient: two SD-family UNets
    /// with the same cross-attention layout take the same adapter.
    modules: usize,
}

impl ResidentAdapter {
    fn matches(&self, device: &Device, dtype: DType, modules: usize) -> bool {
        self.device.same_device(device) && self.dtype == dtype && self.modules == modules
    }
}

/// The engine's image-prompt state.
pub(crate) struct SdReferenceState {
    /// Which architecture's adapter this engine takes. Held rather than read
    /// off [`Self::assets`] because the question has to be answerable when no
    /// bundle was prepared: that is exactly the case that must name the
    /// missing bundle instead of silently rendering unreferenced.
    family: ImagePromptFamily,
    /// Concrete IP-Adapter asset paths admission froze, or `None` when the
    /// bundle was not planned for this engine.
    assets: Option<IpAdapterPaths>,
    resident: Option<ResidentAdapter>,
}

impl SdReferenceState {
    pub(crate) fn new(family: ImagePromptFamily, assets: Option<IpAdapterPaths>) -> Self {
        Self {
            family,
            assets,
            resident: None,
        }
    }

    /// Release the projection stack's device memory.
    ///
    /// Every path that stops classifying the engine as GPU-resident must call
    /// this. `Drop` needs no help — the `Arc` dies with the engine — but
    /// parking does.
    pub(crate) fn drop_adapter(&mut self) {
        self.resident = None;
    }

    /// Device bytes the resident projections occupy, or 0 when none are.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident
            .as_ref()
            .map_or(0, |resident| resident.adapter.resident_bytes())
    }

    /// The adapter path admission froze, for a diagnostic.
    pub(crate) fn adapter_path(&self) -> Option<&std::path::Path> {
        self.assets.as_ref().map(|assets| assets.adapter.as_path())
    }

    /// Resolve image-prompt conditioning for one render.
    ///
    /// Returns `None` for every request that names no reference picture, and
    /// drops the projections on the way out so an unreferenced render does not
    /// keep them alive.
    ///
    /// The vision tower runs BEFORE the projections are loaded, not after.
    /// Both orders produce the same tokens; only this one keeps the 2.5 GB
    /// tower from ever coexisting with the projection stack, which on SDXL is
    /// a further ~700 MB. The cost is that a mismatched adapter file is
    /// diagnosed one encode late — a misconfiguration, against a peak every
    /// render pays.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve(
        &mut self,
        req: &GenerateRequest,
        model: &str,
        use_cfg: bool,
        device: &Device,
        dtype: DType,
        config: &UNet2DConditionModelConfig,
        progress: &ProgressReporter,
    ) -> Result<Option<ResolvedReference>> {
        let Some(asked) = reference_request(req, self.family, model) else {
            self.drop_adapter();
            return Ok(None);
        };
        let images = req
            .edit_images
            .as_ref()
            .expect("reference_request answered Some only for a populated list");
        // `capabilities.reference_images.max_count` is 1 for both families and
        // admission enforces it, so this is a plan-drift warning rather than a
        // refusal: a second authority disagreeing with admission about how many
        // pictures are legal is worse than a render that used the first one.
        if images.len() > 1 {
            tracing::warn!(
                count = images.len(),
                "IP-Adapter conditions on one reference picture; using the first"
            );
        }

        let embeds = self.encode(&images[0], device, dtype, progress)?;
        let adapter = self.ensure_adapter(device, dtype, config)?;
        let context =
            IpAdapterContext::new(&adapter, &embeds, use_cfg, asked.weight, device, dtype)
                .context("projecting the reference image into this UNet's token space")?;
        Ok(Some(ResolvedReference { adapter, context }))
    }

    /// Decode the reference picture and encode it to a CLIP image embedding.
    ///
    /// The tower is built, forwarded, and dropped inside this function. That
    /// is the whole point of the function existing: the borrow checker cannot
    /// enforce "drop before the denoise", but a scope can, and 632 M
    /// parameters surviving into the denoise loop is not a leak any counter
    /// would report.
    fn encode(
        &self,
        bytes: &[u8],
        device: &Device,
        dtype: DType,
        progress: &ProgressReporter,
    ) -> Result<Tensor> {
        let assets = self.assets()?;
        progress.checkpoint()?;
        let label = "Encoding the reference image";
        progress.stage_start(label);
        let started = std::time::Instant::now();

        // `decode_oriented_srgb` is the crate's single EXIF/ICC path: a phone
        // photograph stores its rotation in a tag, and an unoriented buffer
        // hands the tower a sideways picture whose embedding describes a
        // sideways picture. Same argument as the PuLID face crop's. It yields
        // interleaved 8-bit RGB, which is exactly what `preprocess_rgb8`
        // consumes — going through `DynamicImage` would re-wrap the same
        // buffer to unwrap it again.
        let image = crate::img_utils::decode_oriented_srgb(bytes)
            .context("decoding the reference image")?;
        let (width, height) = (image.width() as usize, image.height() as usize);
        let pixels = preprocess_rgb8(image.as_raw(), height, width, device)
            .context("preprocessing the reference image for CLIP-ViT-H-14")?;
        drop(image);

        let embeds = {
            let tower = OpenClipVisionTower::from_installed(
                std::slice::from_ref(&assets.vision_encoder),
                dtype,
                device,
            )
            .context("loading the CLIP-ViT-H-14 reference image encoder")?;
            // `preprocess_rgb8` builds f32 on purpose (the normalization
            // constants are published in f32); the tower's working dtype is the
            // engine's, so the cast belongs here rather than in the shared
            // preprocessor every consumer would then have to undo.
            let pixels = pixels.to_dtype(tower.dtype())?;
            let output = tower
                .forward(&pixels)
                .context("encoding the reference image")?;
            // Classic IP-Adapter conditions on the POOLED, PROJECTED embedding
            // (`diffusers` `pipeline_stable_diffusion.py:532`,
            // `self.image_encoder(image).image_embeds`). The penultimate hidden
            // states beside it are the Plus adapters' input and are dropped
            // here with the tower.
            output.image_embeds
        };
        progress.stage_done(label, started.elapsed());
        progress.checkpoint()?;
        Ok(embeds)
    }

    fn assets(&self) -> Result<&IpAdapterPaths> {
        let assets = self.assets.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "this request attaches a reference image but no IP-Adapter bundle was prepared \
                 for this engine; pull {} and retry",
                self.family.manifest()
            )
        })?;
        // A plan that froze the other architecture's bundle would load a file
        // whose `image_proj.proj` does not divide into this UNet's tokens, and
        // `IpAdapterShape::from_proj_dims` would refuse it by arithmetic. Say
        // so by name first: the arithmetic answer does not mention which
        // bundle was planned.
        if assets.family != self.family {
            bail!(
                "this engine takes the {:?} IP-Adapter but the prepared bundle is {:?}",
                self.family,
                assets.family
            );
        }
        Ok(assets)
    }

    fn ensure_adapter(
        &mut self,
        device: &Device,
        dtype: DType,
        config: &UNet2DConditionModelConfig,
    ) -> Result<Arc<IpAdapter>> {
        let modules = plan_attn_layers(config).len();
        if let Some(resident) = &self.resident {
            if resident.matches(device, dtype, modules) {
                return Ok(Arc::clone(&resident.adapter));
            }
        }
        let path = self.assets()?.adapter.clone();
        let adapter = Arc::new(
            IpAdapter::load(&path, config, dtype, device)
                .context("loading the IP-Adapter image-prompt projections")?,
        );
        self.resident = Some(ResidentAdapter {
            adapter: Arc::clone(&adapter),
            device: device.clone(),
            dtype,
            modules,
        });
        Ok(adapter)
    }

    /// Install a resident adapter without reading a checkpoint.
    #[cfg(test)]
    pub(crate) fn install_resident_for_test(
        &mut self,
        adapter: Arc<IpAdapter>,
        device: Device,
        dtype: DType,
        modules: usize,
    ) {
        self.resident = Some(ResidentAdapter {
            adapter,
            device,
            dtype,
            modules,
        });
    }
}

/// Two cross-attention hooks driving one forward pass.
///
/// SDXL is the only UNet that can carry both adapters at once — PuLID v1.1
/// identity and IP-Adapter image prompting are independent contracts on
/// independent request fields, and nothing in `mold_core::identity` or
/// `capabilities.reference_images` makes one exclude the other. They COMPOSE
/// rather than conflict, and the arithmetic is why:
///
/// ```text
/// PuLID          attended + id_scale * attention(q, id_to_k(id), id_to_v(id))
/// IP-Adapter     attended + ip_scale * attention(q, to_k_ip(img), to_v_ip(img))
/// ```
///
/// (`PuLID/pulid/attention_processor.py:378` and `stable-diffusion.cpp`
/// `src/model/common/block.hpp:391`.) Both are additive deltas computed from
/// the SAME `query` — `to_q`'s own output — and NEITHER reads `attended`, so
/// chaining them yields `attended + id + ip` whichever order they run in. That
/// is also what diffusers produces when an IP-Adapter processor is stacked on
/// a base processor that already injects: each `AttnProcessor` adds its own
/// branch onto the text attention output before `to_out`. Refusing the
/// combination would be a policy invented here, not one upstream states.
///
/// The `None` path is preserved through the composition on purpose: a layer
/// whose members all decline replaces nothing, so the denoise loop's
/// `forward` / `forward_with_hook` branch stays a decision about whether ANY
/// adapter is live, and the zero-weight bit-identity argument survives
/// stacking.
pub(crate) struct LayeredCrossAttentionHook<'a> {
    first: &'a dyn CrossAttentionHook,
    second: &'a dyn CrossAttentionHook,
}

impl<'a> LayeredCrossAttentionHook<'a> {
    pub(crate) fn new(
        first: &'a dyn CrossAttentionHook,
        second: &'a dyn CrossAttentionHook,
    ) -> Self {
        Self { first, second }
    }
}

impl CrossAttentionHook for LayeredCrossAttentionHook<'_> {
    fn cross_attention(
        &self,
        index: usize,
        query: &Tensor,
        attended: &Tensor,
        heads: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        let after_first = self.first.cross_attention(index, query, attended, heads)?;
        // The second hook sees the first's replacement when there was one, so
        // its own delta lands on top rather than discarding it.
        let intermediate = after_first.as_ref().unwrap_or(attended);
        match self
            .second
            .cross_attention(index, query, intermediate, heads)?
        {
            Some(replaced) => Ok(Some(replaced)),
            None => Ok(after_first),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_nn::VarBuilder;
    use candle_transformers::models::stable_diffusion::unet_2d::{
        BlockConfig, UNet2DConditionModel,
    };
    use std::cell::Cell;

    fn block(out_channels: usize, use_cross_attn: Option<usize>, heads: usize) -> BlockConfig {
        BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim: heads,
        }
    }

    /// An SD1.5-SHAPED UNet small enough to build on a CPU in milliseconds.
    ///
    /// Shrunken rather than real for the reason `sdxl::pulid::tests` shrinks
    /// its own: the production geometry is 16 modules at up to 1280 wide
    /// against a 768-wide context, and what these tests check is the ROUTING —
    /// that a `None` hook takes the same code the plain forward takes, and
    /// that a live one does not. `use_linear_projection: false` is SD1.5's own
    /// setting (`mold_candle::stable_diffusion::sd15_unet`) and is kept
    /// because it changes which projection the transformer block applies
    /// around attention, i.e. the path the hook fires inside.
    pub(crate) fn sd15_tiny_config() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            blocks: vec![block(32, None, 2), block(64, Some(2), 4)],
            center_input_sample: false,
            cross_attention_dim: 64,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 1,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }

    /// A UNet on random (not zero) weights.
    ///
    /// Zeroed weights make every attention output zero, and a bit-identity
    /// claim against zero is a claim about nothing: two different code paths
    /// both returning zeros would pass. `VarMap`'s `Randn` init gives the
    /// comparison something to be identical ABOUT.
    fn tiny_unet(config: &UNet2DConditionModelConfig, device: &Device) -> UNet2DConditionModel {
        UNet2DConditionModel::new(synthetic_weights(device, None), 4, 4, false, config.clone())
            .expect("a tiny UNet builds on synthetic weights")
    }

    /// A `VarBuilder` whose every tensor is derived from its own name.
    ///
    /// `known` is the inventory `contains_tensor` admits to. `None` means "any
    /// name", which is what a model constructor wants; the adapter loader
    /// PROBES one index past its plan to catch a checkpoint for a different
    /// UNet, so building an adapter needs a backend that can say no.
    fn synthetic_weights(
        device: &Device,
        known: Option<std::collections::BTreeSet<String>>,
    ) -> VarBuilder<'static> {
        VarBuilder::from_backend(
            Box::new(SyntheticBackend { known }),
            DType::F32,
            device.clone(),
        )
    }

    /// A `SimpleBackend` that answers every `get` with deterministic pseudo
    /// noise derived from the tensor's NAME.
    ///
    /// `VarBuilder::zeros` would make the comparison vacuous (see
    /// [`tiny_unet`]) and `VarMap` with a `Randn` init would make it
    /// irreproducible across runs, which is the wrong trade for a test whose
    /// whole claim is exact equality. Name-derived values are both non-trivial
    /// and stable.
    struct SyntheticBackend {
        known: Option<std::collections::BTreeSet<String>>,
    }

    impl SyntheticBackend {
        /// xorshift64*, seeded from the tensor name: dense, non-trivial, and
        /// identical on every run and every platform.
        fn values(shape: &candle_core::Shape, name: &str) -> Vec<f32> {
            let mut state = name
                .bytes()
                .fold(0x9e37_79b9_7f4a_7c15_u64, |acc, byte| {
                    acc.rotate_left(7) ^ u64::from(byte).wrapping_mul(0x0100_0000_01b3)
                })
                .max(1);
            let mut values = Vec::with_capacity(shape.elem_count());
            for _ in 0..shape.elem_count() {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
                values.push((bits as f32 / 16_777_216.0 - 0.5) * 0.1);
            }
            values
        }
    }

    impl candle_nn::var_builder::SimpleBackend for SyntheticBackend {
        fn get(
            &self,
            shape: candle_core::Shape,
            name: &str,
            _init: candle_nn::Init,
            dtype: DType,
            device: &Device,
        ) -> candle_core::Result<Tensor> {
            let values = Self::values(&shape, name);
            Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
        }

        fn get_unchecked(
            &self,
            name: &str,
            _dtype: DType,
            _device: &Device,
        ) -> candle_core::Result<Tensor> {
            candle_core::bail!("the synthetic backend has no shape for {name}")
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.known.as_ref().is_none_or(|known| known.contains(name))
        }
    }

    /// A hook that observes every module and replaces nothing.
    #[derive(Default)]
    struct NullHook {
        seen: Cell<usize>,
    }

    impl CrossAttentionHook for NullHook {
        fn cross_attention(
            &self,
            _index: usize,
            _query: &Tensor,
            _attended: &Tensor,
            _heads: usize,
        ) -> candle_core::Result<Option<Tensor>> {
            self.seen.set(self.seen.get() + 1);
            Ok(None)
        }
    }

    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn unet_inputs(config: &UNet2DConditionModelConfig, device: &Device) -> (Tensor, Tensor) {
        let xs = Tensor::arange(0f32, 4.0 * 16.0 * 16.0, device)
            .unwrap()
            .reshape((1, 4, 16, 16))
            .unwrap();
        let context = Tensor::arange(0f32, (7 * config.cross_attention_dim) as f32, device)
            .unwrap()
            .reshape((1, 7, config.cross_attention_dim))
            .unwrap();
        ((xs / 1024.0).unwrap(), (context / 1024.0).unwrap())
    }

    /// **The gate the whole zero-weight claim rests on.**
    ///
    /// `IpAdapterRuntime::hook_for_step` answers `None` at a zero scale and the
    /// denoise loop answers `None` by calling `forward`, so an unreferenced
    /// render is bit-identical BY CONSTRUCTION. That argument is only worth
    /// anything if the hooked forward is itself neutral when the hook declines
    /// — otherwise merely COMPILING the routing would move pixels on every
    /// SD1.5 render mold has ever made, referenced or not.
    ///
    /// Both hooked entry points are checked, because SD1.5's loop uses both:
    /// `forward_with_hook` and, when a ControlNet is loaded,
    /// `forward_with_additional_residuals_and_hook`.
    #[test]
    fn a_null_hook_is_bit_identical_to_the_plain_forward() {
        let device = Device::Cpu;
        let config = sd15_tiny_config();
        let unet = tiny_unet(&config, &device);
        let (xs, context) = unet_inputs(&config, &device);

        let plain = unet.forward(&xs, 3.0, &context).unwrap();
        let hook = NullHook::default();
        let hooked = unet.forward_with_hook(&xs, 3.0, &context, &hook).unwrap();
        assert_eq!(
            hook.seen.get(),
            plan_attn_layers(&config).len(),
            "the hook must be offered every planned cross-attention module"
        );
        assert_eq!(
            flat(&hooked),
            flat(&plain),
            "a hook that replaces nothing must not move a single bit"
        );

        // And the same through the ControlNet entry point, with residuals
        // actually present — the arm SD1.5's four-way match adds.
        let residuals = down_block_residuals(&config, &device);
        let mid = Tensor::full(0.25f32, (1, 64, 8, 8), &device).unwrap();
        let plain = unet
            .forward_with_additional_residuals(&xs, 3.0, &context, Some(&residuals), Some(&mid))
            .unwrap();
        let hook = NullHook::default();
        let hooked = unet
            .forward_with_additional_residuals_and_hook(
                &xs,
                3.0,
                &context,
                Some(&residuals),
                Some(&mid),
                &hook,
            )
            .unwrap();
        assert_eq!(hook.seen.get(), plan_attn_layers(&config).len());
        assert_eq!(
            flat(&hooked),
            flat(&plain),
            "a null hook must be inert on the ControlNet path too"
        );
    }

    /// The down-block residual stack this tiny UNet consumes.
    ///
    /// Shapes are dictated by the config: one resnet per block plus a
    /// downsampler between them, at each block's own width and resolution.
    fn down_block_residuals(config: &UNet2DConditionModelConfig, device: &Device) -> Vec<Tensor> {
        let _ = config;
        vec![
            Tensor::full(0.01f32, (1, 32, 16, 16), device).unwrap(),
            Tensor::full(0.02f32, (1, 32, 16, 16), device).unwrap(),
            Tensor::full(0.03f32, (1, 32, 8, 8), device).unwrap(),
            Tensor::full(0.04f32, (1, 64, 8, 8), device).unwrap(),
        ]
    }

    /// Build an IP-Adapter for `config` without touching a checkpoint.
    pub(crate) fn synthetic_adapter(
        config: &UNet2DConditionModelConfig,
        tokens: usize,
        clip_dim: usize,
        device: &Device,
    ) -> IpAdapter {
        let known = plan_attn_layers(config)
            .iter()
            .flat_map(|site| {
                ["to_k_ip", "to_v_ip"].map(|name| {
                    format!(
                        "{}.{}.{name}.weight",
                        crate::ip_adapter::ADAPTER_PREFIX,
                        site.ip_index()
                    )
                })
            })
            .collect();
        IpAdapter::from_var_builder(
            synthetic_weights(device, Some(known)),
            config,
            crate::ip_adapter::IpAdapterShape { tokens, clip_dim },
        )
        .expect("the synthetic adapter matches the plan")
    }

    /// The other half of the falsification: a live weight must actually move
    /// the prediction. Without this, a routing that silently dropped the hook
    /// would pass every bit-identity assertion above.
    #[test]
    fn a_live_reference_weight_changes_the_prediction() {
        let device = Device::Cpu;
        let config = sd15_tiny_config();
        let unet = tiny_unet(&config, &device);
        let (xs, context) = unet_inputs(&config, &device);
        let adapter = synthetic_adapter(&config, 4, 16, &device);

        let tokens = Tensor::arange(0f32, (4 * config.cross_attention_dim) as f32, &device)
            .unwrap()
            .reshape((1, 4, config.cross_attention_dim))
            .unwrap();

        let plain = unet.forward(&xs, 3.0, &context).unwrap();

        let inert = IpAdapterContext::from_tokens(tokens.clone(), 0.0);
        assert!(
            IpAdapterRuntime::new(&adapter, &inert)
                .hook_for_step(0)
                .is_none(),
            "a zero weight must yield no hook at all"
        );

        let live = IpAdapterContext::from_tokens((tokens / 1024.0).unwrap(), 1.0);
        let runtime = IpAdapterRuntime::new(&adapter, &live);
        let hook = runtime
            .hook_for_step(0)
            .expect("a live weight yields a hook");
        let conditioned = unet.forward_with_hook(&xs, 3.0, &context, &hook).unwrap();
        assert_ne!(
            flat(&conditioned),
            flat(&plain),
            "a live reference weight must change the prediction"
        );
    }

    /// PuLID and IP-Adapter both add a delta computed from the same query and
    /// neither reads `attended`, so layering them is exactly the sum of the
    /// two deltas — and in either order.
    ///
    /// This is the assertion behind the decision to COMPOSE rather than refuse
    /// the combination: if the two were order-sensitive, the SDXL loop would
    /// be choosing an arithmetic nobody upstream specified.
    #[test]
    fn layering_two_additive_hooks_sums_their_deltas_in_either_order() {
        let device = Device::Cpu;

        /// A hook that adds a fixed per-index constant, standing in for one
        /// adapter's branch: the property under test is the COMPOSITION, and
        /// a real adapter's arithmetic would only obscure it.
        struct Bias(f32);
        impl CrossAttentionHook for Bias {
            fn cross_attention(
                &self,
                _index: usize,
                _query: &Tensor,
                attended: &Tensor,
                _heads: usize,
            ) -> candle_core::Result<Option<Tensor>> {
                Ok(Some((attended + f64::from(self.0))?))
            }
        }

        let attended = Tensor::arange(0f32, 12.0, &device)
            .unwrap()
            .reshape((1, 3, 4))
            .unwrap();
        let query = attended.clone();
        let (a, b) = (Bias(0.25), Bias(-0.75));

        let forward = LayeredCrossAttentionHook::new(&a, &b)
            .cross_attention(0, &query, &attended, 2)
            .unwrap()
            .expect("a live layer replaces");
        let reversed = LayeredCrossAttentionHook::new(&b, &a)
            .cross_attention(0, &query, &attended, 2)
            .unwrap()
            .expect("a live layer replaces");
        assert_eq!(flat(&forward), flat(&reversed));
        let expected = (&attended + (0.25 - 0.75)).unwrap();
        assert_eq!(flat(&forward), flat(&expected));

        // A layer whose members all decline replaces nothing, so the denoise
        // loop's `forward` / `forward_with_hook` branch keeps its meaning.
        let (null_a, null_b) = (NullHook::default(), NullHook::default());
        assert!(LayeredCrossAttentionHook::new(&null_a, &null_b)
            .cross_attention(0, &query, &attended, 2)
            .unwrap()
            .is_none());
        assert_eq!((null_a.seen.get(), null_b.seen.get()), (1, 1));

        // One live member still replaces, and the null one does not discard it.
        let live = LayeredCrossAttentionHook::new(&null_a, &a)
            .cross_attention(0, &query, &attended, 2)
            .unwrap()
            .expect("one live member replaces");
        assert_eq!(flat(&live), flat(&(&attended + 0.25f64).unwrap()));
    }

    /// The layout the SD1.5 engine hands the resolver is the PRODUCTION
    /// UNet's, not `UNet2DConditionModelConfig::default()`.
    ///
    /// `sd15::pipeline`'s `load_controlnet` builds the CONTROLNET
    /// architecture from `default()`, and it sits a few lines from where the
    /// adapter is resolved — so the near-miss is a live hazard, not a
    /// hypothetical one. The trap is that the two configs plan the SAME 16
    /// module positions at the same widths and head counts: only
    /// `cross_attention_dim` differs (1280 against SD1.5's 768), and that is
    /// precisely the number `to_k_ip`, `to_v_ip` and `image_proj.proj` are
    /// sized on. A module-count comparison would notice nothing.
    #[test]
    fn the_sd15_layout_is_the_production_unet_not_the_controlnet_default() {
        let production = mold_candle::stable_diffusion::sd15_unet();
        assert_eq!(
            plan_attn_layers(&production).len(),
            16,
            "SD1.5 has 16 cross-attention modules"
        );
        assert_eq!(production.cross_attention_dim, 768);
        let controlnet = UNet2DConditionModelConfig::default();
        assert_eq!(
            plan_attn_layers(&controlnet).len(),
            plan_attn_layers(&production).len(),
            "the two configs plan the same module positions, which is why the \
             mistake is invisible to a count"
        );
        assert_ne!(
            controlnet.cross_attention_dim, production.cross_attention_dim,
            "the ControlNet default's context width is what makes it the wrong \
             layout to plan an adapter from"
        );
    }

    fn request(edit_images: Option<Vec<Vec<u8>>>) -> GenerateRequest {
        let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "sd15:fp16",
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance": 7.5,
        }))
        .expect("the minimal generate-request wire shape");
        req.edit_images = edit_images;
        req
    }

    #[test]
    fn a_request_with_no_reference_image_asks_for_nothing() {
        for images in [None, Some(Vec::new()), Some(vec![Vec::new()])] {
            assert_eq!(
                reference_request(&request(images), ImagePromptFamily::Sd15, "sd15:fp16"),
                None
            );
        }
    }

    #[test]
    fn a_reference_image_alone_takes_the_advertised_default() {
        let advertised = reference_images_for_recipe("sd15", "sd15:fp16")
            .weight
            .expect("sd15 advertises a reference weight")
            .default;
        assert_eq!(
            reference_request(
                &request(Some(vec![vec![0x89]])),
                ImagePromptFamily::Sd15,
                "sd15:fp16"
            ),
            Some(ReferenceRequest {
                weight: advertised as f32
            })
        );
    }

    #[test]
    fn an_explicit_zero_weight_asks_for_nothing() {
        let mut req = request(Some(vec![vec![0x89]]));
        req.reference_weight = Some(0.0);
        assert_eq!(
            reference_request(&req, ImagePromptFamily::Sd15, "sd15:fp16"),
            None,
            "reference_weight 0 must be indistinguishable from a plain request"
        );
    }

    fn state_with_adapter(device: &Device) -> SdReferenceState {
        let config = sd15_tiny_config();
        let adapter = Arc::new(synthetic_adapter(&config, 4, 16, device));
        let mut state = SdReferenceState::new(ImagePromptFamily::Sd15, None);
        state.install_resident_for_test(
            adapter,
            device.clone(),
            DType::F32,
            plan_attn_layers(&config).len(),
        );
        state
    }

    #[test]
    fn an_unreferenced_request_drops_a_resident_adapter() {
        let device = Device::Cpu;
        let config = sd15_tiny_config();
        let mut state = state_with_adapter(&device);
        assert!(state.resident_bytes() > 0);

        let resolved = state
            .resolve(
                &request(None),
                "sd15:fp16",
                false,
                &device,
                DType::F32,
                &config,
                &ProgressReporter::default(),
            )
            .unwrap();
        assert!(resolved.is_none());
        assert_eq!(
            state.resident_bytes(),
            0,
            "an unreferenced render must not keep the projections alive"
        );
    }

    #[test]
    fn a_referenced_request_without_a_prepared_bundle_names_it() {
        let device = Device::Cpu;
        let config = sd15_tiny_config();
        let mut state = state_with_adapter(&device);
        let error = state
            .resolve(
                &request(Some(vec![vec![0x89, 0x50, 0x4e, 0x47]])),
                "sd15:fp16",
                false,
                &device,
                DType::F32,
                &config,
                &ProgressReporter::default(),
            )
            .expect_err("no bundle means no adapter");
        let message = format!("{error:#}");
        assert!(message.contains("ip-adapter"), "{message}");
        assert!(state.adapter_path().is_none());
    }
}
