//! Face-identity conditioning state for the SDXL engine.
//!
//! Keeps the PuLID adapter's residency policy in one place instead of spread
//! across `pipeline.rs`'s two render paths. The engine holds exactly one of
//! these; both paths ask it the same question and get back either a
//! [`ResolvedSdxlIdentity`] or `None`.
//!
//! Residency follows the same drop-and-reload discipline the FLUX adapter and
//! the text encoders use, for the same reason: the adapter is ~682 MB on the
//! device in f16/bf16, and it is useless to a render nobody asked to
//! condition. It is loaded on the first identity-conditioned request, kept
//! across subsequent ones that agree on device, dtype, and UNet shape, and
//! released whenever the engine stops holding a UNet — an unconditioned
//! request, the drop before VAE decode, or an `unload()`. That last one is not
//! optional: `ModelCache` parks by calling `unload()`, zeroing the entry's
//! `vram_bytes` while keeping the engine cached, so an adapter that survived it
//! is device memory nothing accounts for.
//!
//! Two things differ from the FLUX state and both come from SDXL being
//! classifier-free rather than guidance-distilled.
//!
//! * There is no true-CFG branch. `true_cfg` / `cfg_start_step` are refused at
//!   the request boundary (`mold_core::identity::TRUE_CFG_FLUX_ONLY`), because
//!   SDXL's ordinary `guidance` already IS the classifier-free scale.
//! * The unconditional identity is not an opt-in. Upstream drives the negative
//!   pass with `uncond_id_embedding` as a matter of course
//!   (`PuLID/pulid/pipeline_v1_1.py:306-316`), so a CFG render REQUIRES it and
//!   a missing one is an error rather than a silently unconditioned branch.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig;
use mold_core::pulid_assets::PulidPaths;
use mold_core::GenerateRequest;

use super::pulid::{
    plan_attn_layers, SdxlIdentityEmbedding, SdxlPulidAdapter, SdxlPulidContext, SdxlPulidRuntime,
};

/// What a request asked for, once the contract defaults are applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IdentityRequest {
    pub(crate) id_weight: f32,
    pub(crate) start_step: usize,
}

/// Read the identity fields off a request.
///
/// `None` covers both "no identity fields at all" and an explicit `id_weight`
/// of 0 — the two cases that must plan nothing, load nothing, and render
/// exactly what a vanilla SDXL request renders. The thresholds come from
/// `mold_core::identity` so the value applied is the value the request contract
/// advertised.
pub(crate) fn identity_request(req: &GenerateRequest) -> Option<IdentityRequest> {
    if !mold_core::identity::request_mentions_identity(req) {
        return None;
    }
    let id_weight = mold_core::identity::effective_id_weight(req);
    if id_weight == 0.0 {
        return None;
    }
    Some(IdentityRequest {
        id_weight: id_weight as f32,
        start_step: mold_core::identity::effective_id_start_step(req) as usize,
    })
}

/// The adapter plus the request's conditioning, alive for one denoise loop.
#[derive(Debug)]
pub(crate) struct ResolvedSdxlIdentity {
    adapter: Arc<SdxlPulidAdapter>,
    context: SdxlPulidContext,
}

impl ResolvedSdxlIdentity {
    pub(crate) fn runtime(&self) -> SdxlPulidRuntime<'_> {
        SdxlPulidRuntime::new(&self.adapter, &self.context)
    }

    /// Cross-attention modules the render is driving, for the progress line.
    pub(crate) fn module_count(&self) -> usize {
        self.adapter.len()
    }
}

/// A resident adapter and the exact shape it was built for.
struct ResidentAdapter {
    adapter: Arc<SdxlPulidAdapter>,
    device: Device,
    dtype: DType,
    /// The module count the UNet config implied when this was built. Cheaper to
    /// compare than the whole config and sufficient: two SD-family UNets with
    /// the same cross-attention layout take the same adapter.
    modules: usize,
}

impl ResidentAdapter {
    fn matches(&self, device: &Device, dtype: DType, modules: usize) -> bool {
        self.device.same_device(device) && self.dtype == dtype && self.modules == modules
    }
}

/// The engine's identity state.
#[derive(Default)]
pub(crate) struct SdxlIdentityState {
    /// Concrete PuLID asset paths admission froze, or `None` when the bundle
    /// was not planned for this engine.
    assets: Option<PulidPaths>,
    /// The identity signal the next conditioned request uses.
    pending_embedding: Option<SdxlIdentityEmbedding>,
    /// The unconditional identity for the next request's negative CFG branch.
    /// Installed and cleared by the same call as `pending_embedding`, so the
    /// two can never describe different requests.
    pending_uncond_embedding: Option<SdxlIdentityEmbedding>,
    resident: Option<ResidentAdapter>,
}

impl SdxlIdentityState {
    pub(crate) fn new(assets: Option<PulidPaths>) -> Self {
        Self {
            assets,
            pending_embedding: None,
            pending_uncond_embedding: None,
            resident: None,
        }
    }

    /// Install the identity signal the next conditioned request will use, and
    /// the unconditional one beside it.
    ///
    /// One setter for both, because an unconditional embedding left over from a
    /// previous request would condition THIS render's negative branch on the
    /// previous person's absence — and the clear is what every dispatch
    /// performs, so it must not be possible to clear one and not the other.
    pub(crate) fn set_embedding(
        &mut self,
        embedding: Option<SdxlIdentityEmbedding>,
        uncond: Option<SdxlIdentityEmbedding>,
    ) {
        self.pending_embedding = embedding;
        self.pending_uncond_embedding = uncond;
    }

    /// Release the adapter's device memory.
    ///
    /// Every path that stops classifying the engine as GPU-resident must call
    /// this. `Drop` needs no help — the `Arc` dies with the engine — but
    /// parking does.
    pub(crate) fn drop_adapter(&mut self) {
        self.resident = None;
    }

    /// Device bytes the resident adapter occupies, or 0 when none is.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident
            .as_ref()
            .map_or(0, |resident| resident.adapter.resident_bytes())
    }

    /// Resolve identity conditioning for one render.
    ///
    /// Returns `None` for every request that does not condition on a face, and
    /// drops the adapter on the way out so an unconditioned render does not
    /// keep ~682 MB alive.
    ///
    /// `use_cfg` is the render's own `cfg_active(guidance)` answer, not a
    /// guess: it decides whether the identity is batched `[uncond, cond]` to
    /// match the doubled latent, and whether the unconditional half is
    /// required at all.
    pub(crate) fn resolve(
        &mut self,
        req: &GenerateRequest,
        use_cfg: bool,
        device: &Device,
        dtype: DType,
        config: &UNet2DConditionModelConfig,
    ) -> Result<Option<ResolvedSdxlIdentity>> {
        let Some(asked) = identity_request(req) else {
            self.drop_adapter();
            return Ok(None);
        };

        let embedding = self.pending_embedding.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "this request asks for face-identity conditioning but no identity embedding \
                 has been extracted for it"
            )
        })?;
        // Upstream conditions the negative pass on the unconditional identity
        // (`pipeline_v1_1.py:306-316`). Rendering it unconditioned instead
        // would leave most of the identity cancelled out of the guided result
        // — a plausible print of the wrong person, with nothing to see.
        let uncond = if use_cfg {
            Some(self.pending_uncond_embedding.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "this request runs classifier-free guidance but no unconditional identity \
                     embedding was frozen for it; the negative branch would render unconditioned"
                )
            })?)
        } else {
            None
        };

        let adapter = self.ensure_adapter(device, dtype, config)?;
        let context = SdxlPulidContext::new(
            &embedding,
            uncond.as_ref(),
            asked.id_weight,
            asked.start_step,
            device,
            dtype,
        )?;
        Ok(Some(ResolvedSdxlIdentity { adapter, context }))
    }

    fn ensure_adapter(
        &mut self,
        device: &Device,
        dtype: DType,
        config: &UNet2DConditionModelConfig,
    ) -> Result<Arc<SdxlPulidAdapter>> {
        let modules = plan_attn_layers(config).len();
        if let Some(resident) = &self.resident {
            if resident.matches(device, dtype, modules) {
                return Ok(Arc::clone(&resident.adapter));
            }
        }
        let path = self
            .assets
            .as_ref()
            .map(|assets| assets.adapter.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "this request asks for face-identity conditioning but no PuLID adapter was \
                     prepared for this engine; pull the pulid-sdxl bundle and retry"
                )
            })?;
        let adapter = Arc::new(
            SdxlPulidAdapter::load(&path, config, dtype, device)
                .with_context(|| "loading the PuLID identity adapter")?,
        );
        self.resident = Some(ResidentAdapter {
            adapter: Arc::clone(&adapter),
            device: device.clone(),
            dtype,
            modules,
        });
        Ok(adapter)
    }

    /// The adapter path admission froze, for a diagnostic.
    pub(crate) fn adapter_path(&self) -> Option<&PathBuf> {
        self.assets.as_ref().map(|assets| &assets.adapter)
    }

    /// Install a resident adapter without loading 682 MB of real weights.
    #[cfg(test)]
    pub(crate) fn install_resident_for_test(
        &mut self,
        adapter: Arc<SdxlPulidAdapter>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::identity::{ID_START_STEP_DEFAULT, ID_WEIGHT_DEFAULT};

    fn request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
        }))
        .expect("the minimal generate-request wire shape")
    }

    fn conditioned() -> GenerateRequest {
        let mut req = request();
        req.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        req
    }

    #[test]
    fn a_request_with_no_identity_fields_asks_for_nothing() {
        assert_eq!(identity_request(&request()), None);
    }

    #[test]
    fn an_image_alone_takes_the_contract_defaults() {
        assert_eq!(
            identity_request(&conditioned()),
            Some(IdentityRequest {
                id_weight: ID_WEIGHT_DEFAULT as f32,
                start_step: ID_START_STEP_DEFAULT as usize,
            })
        );
    }

    #[test]
    fn an_explicit_zero_weight_asks_for_nothing() {
        let mut req = conditioned();
        req.id_weight = Some(0.0);
        assert_eq!(
            identity_request(&req),
            None,
            "id_weight 0 must be byte-indistinguishable from a plain request"
        );
    }

    fn state_with_adapter(device: &Device) -> SdxlIdentityState {
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let adapter = Arc::new(super::super::pulid::tests::synthetic_adapter(
            &config, device,
        ));
        let mut state = SdxlIdentityState::default();
        state.install_resident_for_test(
            adapter,
            device.clone(),
            DType::F32,
            plan_attn_layers(&config).len(),
        );
        state
    }

    #[test]
    fn an_unconditioned_request_drops_a_resident_adapter() {
        let device = Device::Cpu;
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let mut state = state_with_adapter(&device);
        assert!(state.resident_bytes() > 0);

        let resolved = state
            .resolve(&request(), true, &device, DType::F32, &config)
            .unwrap();
        assert!(resolved.is_none());
        assert_eq!(
            state.resident_bytes(),
            0,
            "an unconditioned render must not keep the adapter alive"
        );
    }

    #[test]
    fn a_conditioned_request_without_an_embedding_is_an_explicit_error() {
        let device = Device::Cpu;
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let mut state = state_with_adapter(&device);
        let error = state
            .resolve(&conditioned(), true, &device, DType::F32, &config)
            .expect_err("a conditioned request with no embedding is an error");
        assert!(
            format!("{error:#}").contains("no identity embedding"),
            "{error:#}"
        );
    }

    fn embedding(value: f32) -> SdxlIdentityEmbedding {
        SdxlIdentityEmbedding::new(
            candle_core::Tensor::full(
                value,
                (
                    1,
                    mold_core::identity::ID_EMBEDDING_TOKENS,
                    mold_core::identity::ID_EMBEDDING_DIM,
                ),
                &Device::Cpu,
            )
            .unwrap(),
        )
        .unwrap()
    }

    /// A CFG render whose unconditional half was never frozen must fail loudly:
    /// broadcasting the conditional identity onto the negative branch cancels
    /// most of the identity out of the guided result without erroring.
    #[test]
    fn a_cfg_render_without_the_unconditional_identity_is_an_explicit_error() {
        let device = Device::Cpu;
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let mut state = state_with_adapter(&device);
        state.set_embedding(Some(embedding(1.0)), None);

        let error = state
            .resolve(&conditioned(), true, &device, DType::F32, &config)
            .expect_err("a CFG render needs the unconditional identity");
        assert!(
            format!("{error:#}").contains("unconditional identity"),
            "{error:#}"
        );

        // Without CFG the negative branch does not exist, so its absence is
        // not an error and the identity stays unbatched.
        let resolved = state
            .resolve(&conditioned(), false, &device, DType::F32, &config)
            .unwrap()
            .expect("a non-CFG identity render resolves");
        assert_eq!(resolved.runtime().context().id_embeds.dim(0).unwrap(), 1);
    }

    #[test]
    fn a_cfg_render_batches_the_identity_to_match_the_doubled_latent() {
        let device = Device::Cpu;
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let mut state = state_with_adapter(&device);
        state.set_embedding(Some(embedding(1.0)), Some(embedding(-1.0)));

        let resolved = state
            .resolve(&conditioned(), true, &device, DType::F32, &config)
            .unwrap()
            .expect("a CFG identity render resolves");
        let id_embeds = &resolved.runtime().context().id_embeds;
        assert_eq!(id_embeds.dim(0).unwrap(), 2);
        let rows = id_embeds.to_vec3::<f32>().unwrap();
        assert_eq!(rows[0][0][0], -1.0, "row 0 is the unconditional branch");
        assert_eq!(rows[1][0][0], 1.0, "row 1 is the conditional branch");
        assert_eq!(resolved.module_count(), plan_attn_layers(&config).len());
    }

    #[test]
    fn a_conditioned_request_without_prepared_assets_names_the_missing_bundle() {
        let device = Device::Cpu;
        let config = super::super::pulid::tests::sdxl_tiny_config();
        let mut state = SdxlIdentityState::new(None);
        state.set_embedding(Some(embedding(1.0)), Some(embedding(-1.0)));
        let error = state
            .resolve(&conditioned(), true, &device, DType::F32, &config)
            .expect_err("no assets means no adapter");
        assert!(format!("{error:#}").contains("pulid-sdxl"), "{error:#}");
        assert!(state.adapter_path().is_none());
    }
}
