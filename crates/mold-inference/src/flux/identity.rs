//! Face-identity conditioning state for the FLUX engine.
//!
//! Keeps the PuLID adapter's residency policy in one place instead of spread
//! across `pipeline.rs`'s two denoise paths. The engine holds exactly one of
//! these; both paths ask it the same question and get back either a
//! [`ResolvedIdentity`] or `None`.
//!
//! Residency follows the same drop-and-reload discipline the text encoders
//! use, for the same reason: the adapter is ~1.14 GB of fp16 that is useless
//! to a render nobody asked to condition. It is loaded on the first
//! identity-conditioned request, kept across subsequent ones that agree on
//! device, dtype, and transformer shape, and dropped the moment a request does
//! not condition on a face. Unlike the transformer's blocks it is never
//! streamed — see `flux::offload::OffloadedFluxTransformer::forward_with_hook`
//! for why.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_transformers::models::flux;
use mold_core::pulid_assets::PulidPaths;
use mold_core::GenerateRequest;

use crate::flux::pulid::{IdentityEmbedding, PulidAdapter, PulidContext, PulidRuntime};

/// What a request asked for, once the contract defaults are applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IdentityRequest {
    pub(crate) id_weight: f32,
    pub(crate) start_step: usize,
}

/// Read the identity fields off a request.
///
/// `None` covers both "no identity fields at all" and an explicit
/// `id_weight` of 0 — the two cases that must plan nothing, load nothing, and
/// render exactly what a vanilla FLUX request renders. The thresholds come
/// from `mold_core::identity` so the value applied is the value the request
/// contract advertised.
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
pub(crate) struct ResolvedIdentity {
    adapter: Arc<PulidAdapter>,
    context: PulidContext,
}

impl ResolvedIdentity {
    pub(crate) fn runtime(&self) -> PulidRuntime<'_> {
        PulidRuntime::new(&self.adapter, &self.context)
    }
}

/// A resident adapter and the exact shape it was built for.
struct ResidentAdapter {
    adapter: Arc<PulidAdapter>,
    device: Device,
    dtype: DType,
    depth: usize,
    depth_single_blocks: usize,
}

impl ResidentAdapter {
    fn matches(&self, device: &Device, dtype: DType, cfg: &flux::model::Config) -> bool {
        self.device.same_device(device)
            && self.dtype == dtype
            && self.depth == cfg.depth
            && self.depth_single_blocks == cfg.depth_single_blocks
    }
}

/// The engine's identity state.
#[derive(Default)]
pub(crate) struct EngineIdentityState {
    /// Concrete PuLID asset paths admission froze, or `None` when the bundle
    /// was not planned for this engine.
    assets: Option<PulidPaths>,
    /// The identity signal the next conditioned request uses.
    ///
    /// This is the seam #1223 fills: the face extractor produces an
    /// [`IdentityEmbedding`] from `req.id_image` and installs it here, after
    /// which nothing downstream changes. Until then only a test or dev harness
    /// sets it, which is why a conditioned request with no embedding is an
    /// explicit error rather than a silently unconditioned render.
    pending_embedding: Option<IdentityEmbedding>,
    resident: Option<ResidentAdapter>,
}

impl EngineIdentityState {
    pub(crate) fn new(assets: Option<PulidPaths>) -> Self {
        Self {
            assets,
            pending_embedding: None,
            resident: None,
        }
    }

    /// Install the identity signal the next conditioned request will use.
    pub(crate) fn set_embedding(&mut self, embedding: Option<IdentityEmbedding>) {
        self.pending_embedding = embedding;
    }

    pub(crate) fn adapter_path(&self) -> Option<&PathBuf> {
        self.assets.as_ref().map(|assets| &assets.adapter)
    }

    /// Release the adapter's memory.
    pub(crate) fn drop_adapter(&mut self) {
        self.resident = None;
    }

    /// Whether an adapter is currently resident.
    #[cfg(test)]
    pub(crate) fn is_resident(&self) -> bool {
        self.resident.is_some()
    }

    /// Resolve identity conditioning for one request.
    ///
    /// Returns `None` for every request that does not condition on a face, and
    /// drops the adapter on the way out so an unconditioned render does not
    /// keep 1.14 GB alive. A conditioned request loads the adapter if the
    /// resident one does not match this device, dtype, and transformer shape.
    pub(crate) fn resolve(
        &mut self,
        req: &GenerateRequest,
        device: &Device,
        dtype: DType,
        cfg: &flux::model::Config,
    ) -> Result<Option<ResolvedIdentity>> {
        let Some(asked) = identity_request(req) else {
            self.drop_adapter();
            return Ok(None);
        };

        let embedding = self.pending_embedding.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "this request asks for face-identity conditioning but no identity embedding \
                 has been extracted for it; the PuLID face extractor is not wired into the \
                 engine yet"
            )
        })?;

        let adapter = self.ensure_adapter(device, dtype, cfg)?;
        let context =
            PulidContext::new(&embedding, asked.id_weight, asked.start_step, device, dtype)?;
        Ok(Some(ResolvedIdentity { adapter, context }))
    }

    fn ensure_adapter(
        &mut self,
        device: &Device,
        dtype: DType,
        cfg: &flux::model::Config,
    ) -> Result<Arc<PulidAdapter>> {
        if let Some(resident) = &self.resident {
            if resident.matches(device, dtype, cfg) {
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
                     prepared for this engine; pull the pulid-flux bundle and retry"
                )
            })?;
        let adapter = Arc::new(
            PulidAdapter::load(&path, cfg.depth, cfg.depth_single_blocks, dtype, device)
                .with_context(|| "loading the PuLID identity adapter")?,
        );
        self.resident = Some(ResidentAdapter {
            adapter: Arc::clone(&adapter),
            device: device.clone(),
            dtype,
            depth: cfg.depth,
            depth_single_blocks: cfg.depth_single_blocks,
        });
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use mold_core::identity::{ID_START_STEP_DEFAULT, ID_WEIGHT_DEFAULT};

    /// Built through the wire shape rather than an exhaustive struct literal:
    /// these tests care about four identity fields, and a literal would need
    /// editing every time an unrelated request field lands.
    fn request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
            "strength": 0.75,
        }))
        .expect("the minimal generate-request wire shape")
    }

    #[test]
    fn a_request_with_no_identity_fields_asks_for_nothing() {
        assert_eq!(identity_request(&request()), None);
    }

    #[test]
    fn an_image_alone_takes_the_contract_defaults() {
        let mut req = request();
        req.id_image = Some(vec![1, 2, 3]);
        let asked = identity_request(&req).expect("identity requested");
        assert_eq!(asked.id_weight, ID_WEIGHT_DEFAULT as f32);
        assert_eq!(asked.start_step, ID_START_STEP_DEFAULT as usize);
    }

    #[test]
    fn an_explicit_zero_weight_asks_for_nothing() {
        let mut req = request();
        req.id_image = Some(vec![1, 2, 3]);
        req.id_weight = Some(0.0);
        assert_eq!(
            identity_request(&req),
            None,
            "a zero weight must plan no identity work at all"
        );
    }

    #[test]
    fn a_delayed_start_is_carried_through() {
        let mut req = request();
        req.id_image = Some(vec![1, 2, 3]);
        req.id_weight = Some(0.8);
        req.id_start_step = Some(4);
        let asked = identity_request(&req).expect("identity requested");
        assert_eq!(asked.id_weight, 0.8);
        assert_eq!(asked.start_step, 4);
    }

    #[test]
    fn an_unconditioned_request_drops_a_resident_adapter() {
        let mut state = EngineIdentityState::new(None);
        // Stand in for a resident adapter without loading 1.14 GB of weights.
        state.resident = Some(ResidentAdapter {
            adapter: Arc::new(crate::flux::pulid::tests::synthetic_adapter(
                crate::flux::pulid::tests::tiny_config(),
                4,
                8,
                DType::F32,
                &Device::Cpu,
            )),
            device: Device::Cpu,
            dtype: DType::F32,
            depth: 4,
            depth_single_blocks: 8,
        });
        assert!(state.is_resident());

        let cfg = flux::model::Config::dev();
        let resolved = state
            .resolve(&request(), &Device::Cpu, DType::F32, &cfg)
            .expect("an unconditioned request never fails");
        assert!(resolved.is_none());
        assert!(
            !state.is_resident(),
            "the adapter must not survive a request that does not condition on a face"
        );
    }

    #[test]
    fn a_conditioned_request_without_an_embedding_is_an_explicit_error() {
        let mut state = EngineIdentityState::new(None);
        let mut req = request();
        req.id_image = Some(vec![1, 2, 3]);
        let cfg = flux::model::Config::dev();
        let error = state
            .resolve(&req, &Device::Cpu, DType::F32, &cfg)
            .expect_err("no embedding is an error, never a silent unconditioned render");
        assert!(
            error.to_string().contains("no identity embedding"),
            "{error}"
        );
    }

    #[test]
    fn a_conditioned_request_without_prepared_assets_names_the_missing_bundle() {
        let mut state = EngineIdentityState::new(None);
        state.set_embedding(Some(
            IdentityEmbedding::new(
                Tensor::zeros(
                    (
                        crate::flux::pulid::ID_TOKENS,
                        crate::flux::pulid::ID_TOKEN_DIM,
                    ),
                    DType::F32,
                    &Device::Cpu,
                )
                .unwrap(),
            )
            .unwrap(),
        ));
        let mut req = request();
        req.id_image = Some(vec![1, 2, 3]);
        let cfg = flux::model::Config::dev();
        let error = state
            .resolve(&req, &Device::Cpu, DType::F32, &cfg)
            .expect_err("no adapter path is an error");
        assert!(error.to_string().contains("no PuLID adapter"), "{error}");
    }

    #[test]
    fn a_resident_adapter_is_reused_only_for_a_matching_shape() {
        let resident = ResidentAdapter {
            adapter: Arc::new(crate::flux::pulid::tests::synthetic_adapter(
                crate::flux::pulid::tests::tiny_config(),
                4,
                8,
                DType::F32,
                &Device::Cpu,
            )),
            device: Device::Cpu,
            dtype: DType::F32,
            depth: 4,
            depth_single_blocks: 8,
        };
        let mut cfg = flux::model::Config::dev();
        cfg.depth = 4;
        cfg.depth_single_blocks = 8;
        assert!(resident.matches(&Device::Cpu, DType::F32, &cfg));
        assert!(
            !resident.matches(&Device::Cpu, DType::BF16, &cfg),
            "a dtype change must rebuild — the transformer's working dtype is what it feeds"
        );
        cfg.depth = 19;
        assert!(
            !resident.matches(&Device::Cpu, DType::F32, &cfg),
            "a different block count needs a different module count"
        );
    }
}
