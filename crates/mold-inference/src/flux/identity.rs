//! Face-identity conditioning state for the FLUX engine.
//!
//! Keeps the PuLID adapter's residency policy in one place instead of spread
//! across `pipeline.rs`'s two denoise paths. The engine holds exactly one of
//! these; both paths ask it the same question and get back either a
//! [`ResolvedIdentity`] or `None`.
//!
//! Residency follows the same drop-and-reload discipline the text encoders
//! use, for the same reason: the adapter is ~1.7 GB on the device at FLUX.1's
//! geometry in f32 (~840 MB in bf16), and it is useless to a render nobody
//! asked to condition. It is loaded on the first identity-conditioned request,
//! kept across subsequent ones that agree on device, dtype, and transformer
//! shape, and released whenever the engine stops holding a transformer — an
//! unconditioned request, a render the transformer did not survive, or an
//! `unload()`. That last one is not optional: `ModelCache` parks by calling
//! `unload()`, zeroing the entry's `vram_bytes` while keeping the engine
//! cached, so an adapter that survived it is device memory nothing accounts
//! for and the next model switch sizes its preflight against a number that is
//! wrong by the whole adapter. [`EngineIdentityState::resident_bytes`] is the
//! accounting-visible form of that guarantee.
//!
//! Unlike the transformer's blocks it is never streamed — see
//! `flux::offload::OffloadedFluxTransformer::forward_with_hook` for why.

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

/// The identity conditioning for one render, and the engine slot it came from.
///
/// The adapter is reachable through **two** owners — the render's
/// [`ResolvedIdentity`] and [`EngineIdentityState`]'s resident slot — so
/// releasing one of them frees nothing. This handle exists so a render never
/// holds those two references separately and cannot release half of them: it
/// is what the denoise loop is handed, and [`Self::release`] drops both at
/// once.
///
/// That matters at exactly one moment. Every FLUX path that drops the
/// transformer before VAE decode does it to create decode headroom — the
/// sequential and offloaded paths always, the eager path unless
/// `MOLD_FLUX_KEEP_TRANSFORMER` keeps it hot — and those are the constrained
/// machines that chose those paths in the first place. An adapter still alive
/// there is 0.8–1.7 GB the VAE's conv2d intermediates have to compete with, on
/// the render that could least afford it.
pub(crate) struct RenderIdentity<'a> {
    resolved: Option<ResolvedIdentity>,
    state: &'a mut EngineIdentityState,
}

impl RenderIdentity<'_> {
    /// The runtime the denoise loop drives, or `None` when this render does
    /// not condition on a face — which is the whole PuLID gate.
    pub(crate) fn runtime(&self) -> Option<PulidRuntime<'_>> {
        self.resolved.as_ref().map(ResolvedIdentity::runtime)
    }

    /// Whether this render conditions on a face at all.
    pub(crate) fn is_active(&self) -> bool {
        self.resolved.is_some()
    }

    /// Cross-attention modules the render is driving, for the progress line.
    pub(crate) fn module_count(&self) -> usize {
        self.resolved
            .as_ref()
            .map_or(0, |resolved| resolved.runtime().adapter().len())
    }

    /// Release the adapter's device memory now.
    ///
    /// Called at the transformer drop point, before the device sync and the
    /// VAE load, so the freed bytes are actually available to the decode.
    /// Idempotent, and harmless on a render that never conditioned.
    /// `FluxEngine::generate`'s end-of-render check remains the backstop for
    /// any path that returns without reaching a drop point at all.
    pub(crate) fn release(&mut self) {
        self.resolved = None;
        self.state.drop_adapter();
    }

    /// Device bytes still held, across both references.
    #[cfg(test)]
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.state.resident_bytes()
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

    /// Release the adapter's device memory.
    ///
    /// Every path that stops classifying the engine as GPU-resident must call
    /// this. `Drop` needs no help — the `Arc` dies with the engine — but
    /// parking does: `ModelCache` calls `unload()`, sets the entry's
    /// `vram_bytes` to 0, and keeps the engine alive in the cache, so an
    /// adapter that survived would be device memory nothing accounts for.
    pub(crate) fn drop_adapter(&mut self) {
        self.resident = None;
    }

    /// Device bytes the resident adapter occupies, or 0 when none is.
    ///
    /// This is the accounting-visible form of [`Self::drop_adapter`]: a caller
    /// that wants to know whether the engine is really holding nothing asks
    /// this rather than trusting that a drop happened.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident
            .as_ref()
            .map_or(0, |resident| resident.adapter.resident_bytes())
    }

    /// A render handle over whatever is currently resident, with a synthetic
    /// context — the same TWO references `resolve_for_render` produces (the
    /// engine slot and the render's own clone), so a test can prove both are
    /// released rather than only the one it can see.
    #[cfg(test)]
    pub(crate) fn render_identity_for_test(&mut self) -> RenderIdentity<'_> {
        let resolved = self.resident.as_ref().map(|resident| ResolvedIdentity {
            adapter: Arc::clone(&resident.adapter),
            context: crate::flux::pulid::tests::synthetic_context(
                resident.adapter.config(),
                1.0,
                0,
                resident.dtype,
                &resident.device,
            ),
        });
        RenderIdentity {
            resolved,
            state: self,
        }
    }

    /// The resident adapter handle, for a test that wants to watch its
    /// reference count.
    #[cfg(test)]
    pub(crate) fn resident_adapter_for_test(&self) -> Option<Arc<PulidAdapter>> {
        self.resident
            .as_ref()
            .map(|resident| Arc::clone(&resident.adapter))
    }

    /// Install a resident adapter without loading 1.7 GB of real weights.
    #[cfg(test)]
    pub(crate) fn install_resident_for_test(
        &mut self,
        adapter: Arc<PulidAdapter>,
        device: Device,
        dtype: DType,
        depth: usize,
        depth_single_blocks: usize,
    ) {
        self.resident = Some(ResidentAdapter {
            adapter,
            device,
            dtype,
            depth,
            depth_single_blocks,
        });
    }

    /// Resolve identity conditioning for one render.
    ///
    /// This is what the render paths call. The result borrows the engine's
    /// resident slot, so the render cannot end up holding the adapter through
    /// a reference the transformer drop point does not know about — see
    /// [`RenderIdentity`].
    pub(crate) fn resolve_for_render(
        &mut self,
        req: &GenerateRequest,
        device: &Device,
        dtype: DType,
        cfg: &flux::model::Config,
    ) -> Result<RenderIdentity<'_>> {
        let resolved = self.resolve(req, device, dtype, cfg)?;
        Ok(RenderIdentity {
            resolved,
            state: self,
        })
    }

    /// Resolve identity conditioning for one request.
    ///
    /// Returns `None` for every request that does not condition on a face, and
    /// drops the adapter on the way out so an unconditioned render does not
    /// keep ~1.7 GB alive. A conditioned request loads the adapter if the
    /// resident one does not match this device, dtype, and transformer shape.
    fn resolve(
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
pub(crate) mod tests {
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

    /// Stands in for a resident adapter without loading 1.7 GB of weights.
    pub(crate) fn state_holding_an_adapter() -> EngineIdentityState {
        let mut state = EngineIdentityState::new(None);
        state.install_resident_for_test(
            Arc::new(crate::flux::pulid::tests::synthetic_adapter(
                crate::flux::pulid::tests::tiny_config(),
                4,
                8,
                DType::F32,
                &Device::Cpu,
            )),
            Device::Cpu,
            DType::F32,
            4,
            8,
        );
        state
    }

    #[test]
    fn an_unconditioned_request_drops_a_resident_adapter() {
        let mut state = state_holding_an_adapter();
        assert!(state.resident_bytes() > 0);

        let cfg = flux::model::Config::dev();
        let resolved = state
            .resolve(&request(), &Device::Cpu, DType::F32, &cfg)
            .expect("an unconditioned request never fails");
        assert!(resolved.is_none());
        assert_eq!(
            state.resident_bytes(),
            0,
            "the adapter must not survive a request that does not condition on a face"
        );
    }

    /// The sequential and offloaded paths drop the transformer unconditionally
    /// before VAE decode, and call `release()` at that exact point. Both of the
    /// render's references have to go: releasing only one frees nothing.
    #[test]
    fn releasing_a_render_drops_both_references_to_the_adapter() {
        let mut state = state_holding_an_adapter();
        let watched = state
            .resident_adapter_for_test()
            .expect("the fixture holds an adapter");
        let mut identity = state.render_identity_for_test();
        assert!(identity.is_active());
        assert!(identity.module_count() > 0);
        assert!(identity.runtime().is_some());
        assert_eq!(
            Arc::strong_count(&watched),
            3,
            "the engine slot, the render's clone, and this test's watch"
        );

        identity.release();

        assert_eq!(identity.resident_bytes(), 0);
        assert!(!identity.is_active());
        assert!(identity.runtime().is_none());
        assert_eq!(
            Arc::strong_count(&watched),
            1,
            "the engine slot and the render's clone must both go"
        );

        // Idempotent: `generate`'s end-of-render backstop runs after this.
        identity.release();
        assert_eq!(identity.resident_bytes(), 0);
    }

    /// A render that never conditioned holds nothing, so the release the drop
    /// point performs unconditionally must be harmless.
    #[test]
    fn releasing_an_inactive_render_is_harmless() {
        let mut state = EngineIdentityState::new(None);
        let mut identity = state.render_identity_for_test();
        assert!(!identity.is_active());
        assert_eq!(identity.module_count(), 0);
        assert!(identity.runtime().is_none());
        identity.release();
        assert_eq!(identity.resident_bytes(), 0);
    }

    /// The parking path: `ModelCache` calls `unload()`, zeroes the entry's
    /// `vram_bytes`, and keeps the engine cached. An adapter that survived
    /// that would be device memory nothing accounts for.
    #[test]
    fn dropping_the_adapter_zeroes_the_accounted_bytes() {
        let mut state = state_holding_an_adapter();
        let before = state.resident_bytes();
        assert!(before > 0, "the fixture must actually hold something");
        state.drop_adapter();
        assert_eq!(state.resident_bytes(), 0);
        // Idempotent: a second release on an already-empty state is a no-op,
        // because several paths may classify the engine as gone.
        state.drop_adapter();
        assert_eq!(state.resident_bytes(), 0);
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
