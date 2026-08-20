//! Authorization-bound Qwen conditioner adapter for private MiniMax H3 UAT.
//!
//! This module is compiled only with `h3-private-uat`. It joins the exact
//! opened-file-bound NVFP4-AWQ runtime to the already-landed runtime-neutral
//! FL2VA/Ref2VA presentation contracts. It does not register an engine, find
//! or download weights, alter capabilities, or create a public factory path.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    build_ref2va_presentation, load_h3_qwen_nvfp4_conditioner_from_authority,
    open_h3_qwen_nvfp4_authority_after_authorization, pack_qwen_vision_u8, qwen_mrope_positions,
    released_h3_qwen_nvfp4_output_tensor_bytes, released_h3_qwen_nvfp4_runtime_memory_facts,
    released_h3_qwen_nvfp4_runtime_memory_facts_for_placement, ConditionerCheckpoint, GridThw,
    H3AuthenticatedQwenNvfp4Authority, H3ConditionerInput, H3ModalityTag, H3QwenNvfp4LoadObserver,
    H3QwenNvfp4RuntimeError, H3QwenNvfp4RuntimeMemoryFacts, H3QwenNvfp4RuntimePlacement,
    H3RawTokenizer, H3VisionInput, LoadedH3QwenNvfp4Conditioner, PackedVisionPatches,
    RefPresentationKind, H3_QWEN_NVFP4_AWQ_POLICY_SHA256, H3_QWEN_NVFP4_AWQ_SHA256,
    H3_SELECTED_LANGUAGE_LAYERS,
};
use mold_core::minimax_h3::Task;
use std::path::Path;

use super::backend::{
    prepare_fl2va_conditioner_input, H3BackendArtifactLease, H3BackendExecutionLease,
    H3CandleBackendDevice,
};
use super::pipeline::ref2va::H3ReferencePresentation;
use super::pipeline::{
    H3PipelineCheckpoint, H3PipelineEvent, H3PipelinePhase, H3PreparedEndpoint, H3TextConditioning,
};
use super::private_qwen_support::H3PrivateQwenSupport;
use super::reference_media::{H3NormalizedReference, H3ReferenceMediaAdapter};
use super::H3ConditionerLease;
use crate::{
    FrozenH3FactoryAuthority, H3FactoryConditionerPlacement, H3FactoryQuantizationAuthority,
};

const QWEN_MAXIMUM_TOKENS: usize = 262_144;
const QWEN_SPATIAL_MERGE_SIZE: usize = 2;
const QWEN_VISION_LAYER_COUNT: usize = 27;
const QWEN_VISION_CHECKPOINTS_PER_PASS: usize = 1 + QWEN_VISION_LAYER_COUNT;

/// Private admission route used to derive the exact retained representation
/// before a concrete CUDA device is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3PrivateQwenLoaderMemoryRoute {
    Cuda,
    Cpu,
}

/// Exact parameter authority consumed by the private loader's pre-load and
/// post-load memory gate. This intentionally omits staging and I/O facts,
/// which remain authenticated runtime evidence but are not resident charges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct H3PrivateQwenLoaderMemoryAuthority {
    pub source_parameter_bytes: u64,
    pub host_resident_parameter_bytes: u64,
    pub device_resident_parameter_bytes: u64,
    /// Largest single tensor the loader reads, and the raw header it retains.
    /// Both are anonymous host bytes the target budget must derive from here
    /// rather than choose; `validate_parameter_memory` re-pins them against
    /// the authenticated runtime facts before any tensor is allocated.
    pub maximum_tensor_staging_bytes: u64,
    pub retained_raw_header_bytes: u64,
}

/// Payload-free authority for authenticating the private Qwen checkpoint
/// before the scheduler commits any CUDA allocation. It binds the future
/// concrete route without retaining or constructing a Candle [`Device`].
pub(crate) struct H3PrivateQwenOpenRouteAuthority {
    route: H3PrivateQwenLoaderMemoryRoute,
    device_id: String,
    device_ordinal: usize,
    compute_capability: Option<(u16, u16)>,
    factory_identity_sha256: String,
    execution_fingerprint: String,
    component_set_identity_sha256: String,
    conditioner_component_content_sha256: String,
    conditioner_component_validation_sha256: String,
    support_identity_sha256: String,
    weight_identity_sha256: String,
    weight_header_identity_sha256: String,
    weight_policy_identity_sha256: String,
    memory_facts: H3QwenNvfp4RuntimeMemoryFacts,
}

impl H3PrivateQwenOpenRouteAuthority {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        authority: &FrozenH3FactoryAuthority,
        support: &H3PrivateQwenSupport,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: Option<(u16, u16)>,
        weight_identity_sha256: &str,
        weight_header_identity_sha256: &str,
        weight_policy_identity_sha256: &str,
    ) -> Result<Self> {
        let route = match authority.conditioner_placement() {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                H3PrivateQwenLoaderMemoryRoute::Cuda
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => H3PrivateQwenLoaderMemoryRoute::Cpu,
        };
        let placement = route.placement();
        let memory_facts = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(placement)?;
        let (component_content, component_validation) = authority.conditioner_component_authority();
        let captured = Self {
            route,
            device_id: device_id.into(),
            device_ordinal,
            compute_capability,
            factory_identity_sha256: authority.identity_sha256().into(),
            execution_fingerprint: authority.execution_fingerprint().into(),
            component_set_identity_sha256: authority.component_set_identity_sha256().into(),
            conditioner_component_content_sha256: component_content.into(),
            conditioner_component_validation_sha256: component_validation.into(),
            support_identity_sha256: support.support_identity_sha256().into(),
            weight_identity_sha256: weight_identity_sha256.into(),
            weight_header_identity_sha256: weight_header_identity_sha256.into(),
            weight_policy_identity_sha256: weight_policy_identity_sha256.into(),
            memory_facts,
        };
        captured.revalidate(authority, support)?;
        Ok(captured)
    }

    fn revalidate(
        &self,
        authority: &FrozenH3FactoryAuthority,
        support: &H3PrivateQwenSupport,
    ) -> Result<()> {
        support.revalidate()?;
        authority.validate_engine_seam(
            support.model(),
            self.device_ordinal,
            authority.block_offload(),
        )?;
        let qwen_policy = match authority.quantization() {
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                qwen_policy_sha256,
                ..
            } => qwen_policy_sha256,
            H3FactoryQuantizationAuthority::OfficialBf16 => {
                bail!("private H3 Qwen open route changed to the official BF16 layout")
            }
        };
        let (component_content, component_validation) = authority.conditioner_component_authority();
        let expected_route = match authority.conditioner_placement() {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                H3PrivateQwenLoaderMemoryRoute::Cuda
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => H3PrivateQwenLoaderMemoryRoute::Cpu,
        };
        let expected_memory =
            released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(expected_route.placement())?;
        if self.route != expected_route
            || self.device_id != authority.device_id()
            || self.device_ordinal != authority.device_ordinal()
            || self.compute_capability != authority.compute_capability()
            || self.factory_identity_sha256 != authority.identity_sha256()
            || self.execution_fingerprint != authority.execution_fingerprint()
            || self.component_set_identity_sha256 != authority.component_set_identity_sha256()
            || self.conditioner_component_content_sha256 != component_content
            || self.conditioner_component_validation_sha256 != component_validation
            || self.support_identity_sha256 != support.support_identity_sha256()
            || self.weight_identity_sha256 != H3_QWEN_NVFP4_AWQ_SHA256
            || self.weight_policy_identity_sha256 != H3_QWEN_NVFP4_AWQ_POLICY_SHA256
            || self.weight_policy_identity_sha256 != qwen_policy.as_str()
            || !valid_sha256(&self.weight_header_identity_sha256)
            || self.memory_facts != expected_memory
        {
            bail!("private H3 payload-free Qwen open route changed before authentication")
        }
        validate_parameter_memory(authority, &self.memory_facts)
    }

    const fn placement(&self) -> H3QwenNvfp4RuntimePlacement {
        self.route.placement()
    }
}

impl H3PrivateQwenLoaderMemoryRoute {
    const fn placement(self) -> H3QwenNvfp4RuntimePlacement {
        match self {
            Self::Cuda => H3QwenNvfp4RuntimePlacement::Accelerated,
            Self::Cpu => H3QwenNvfp4RuntimePlacement::Cpu,
        }
    }
}

/// Return the same released facts the private loader derives at its
/// authority-before-open boundary, without requiring a local CUDA device.
pub fn released_h3_private_qwen_loader_memory_authority(
    route: H3PrivateQwenLoaderMemoryRoute,
) -> Result<H3PrivateQwenLoaderMemoryAuthority> {
    let placement = match route {
        H3PrivateQwenLoaderMemoryRoute::Cuda => H3QwenNvfp4RuntimePlacement::Accelerated,
        H3PrivateQwenLoaderMemoryRoute::Cpu => H3QwenNvfp4RuntimePlacement::Cpu,
    };
    let facts = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(placement)?;
    Ok(H3PrivateQwenLoaderMemoryAuthority {
        source_parameter_bytes: facts.effective_parameter_bytes,
        host_resident_parameter_bytes: facts.host_resident_parameter_bytes,
        device_resident_parameter_bytes: facts.device_resident_parameter_bytes,
        maximum_tensor_staging_bytes: facts.maximum_tensor_staging_bytes,
        retained_raw_header_bytes: facts.retained_raw_header_bytes,
    })
}

/// Return the released conditioner's exact BF16 output-copy charge for an
/// admitted text sequence. Server tests consume this across the crate boundary
/// so their CPU overlap ledger cannot silently diverge from the loader.
pub fn released_h3_private_qwen_output_tensor_bytes(text_rows: u64) -> Result<u64> {
    Ok(released_h3_qwen_nvfp4_output_tensor_bytes(text_rows)?)
}

/// Apply the exact memory gate used immediately before and after private Qwen
/// loading. The server's private-UAT bridge uses this to prove its frozen
/// authority is consumable before it can reach an artifact path.
pub fn validate_h3_private_qwen_loader_memory_authority(
    authority: &FrozenH3FactoryAuthority,
    route: H3PrivateQwenLoaderMemoryRoute,
) -> Result<()> {
    let placement = match route {
        H3PrivateQwenLoaderMemoryRoute::Cuda => H3QwenNvfp4RuntimePlacement::Accelerated,
        H3PrivateQwenLoaderMemoryRoute::Cpu => H3QwenNvfp4RuntimePlacement::Cpu,
    };
    let facts = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(placement)?;
    validate_parameter_memory(authority, &facts)
}

/// Attempt-scoped conditioner reservation issued for the same frozen factory
/// authority as the complete H3 worker transaction.
pub(crate) trait H3PrivateQwenConditionerLease: H3ConditionerLease {
    fn factory_identity_sha256(&self) -> &str;
    fn execution_fingerprint(&self) -> &str;
    fn is_active(&self) -> bool;
}

/// Artifact authority independent from the support loader and Qwen runtime.
///
/// # Safety
///
/// Implementers must keep the complete logical conditioner component and the
/// exact support/checkpoint objects named below immutable until the lease is
/// dropped. Returning digests copied from a loaded object without validating
/// them against the frozen admission record violates this contract.
pub(crate) unsafe trait H3PrivateQwenArtifactLease: H3BackendArtifactLease {
    fn factory_identity_sha256(&self) -> &str;
    fn conditioner_component_content_sha256(&self) -> &str;
    fn conditioner_component_validation_sha256(&self) -> &str;
    fn support_identity_sha256(&self) -> &str;
    fn weight_identity_sha256(&self) -> &str;
    fn weight_header_identity_sha256(&self) -> &str;
    fn weight_policy_identity_sha256(&self) -> &str;
}

/// Field order is load-bearing. `model` is explicitly taken before releasing
/// the conditioner lease. The adapter owns only lightweight execution and
/// artifact projections; the singular underlying leases remain owned by the
/// outer attempt and cannot be cloned or dropped here.
struct H3PrivateQwenResources<M, S, C>
where
    C: H3PrivateQwenConditionerLease,
{
    model: Option<M>,
    support: S,
    conditioner_lease: C,
    conditioner_released: bool,
}

impl<M, S, C> H3PrivateQwenResources<M, S, C>
where
    C: H3PrivateQwenConditionerLease,
{
    fn release_conditioner(&mut self) {
        if let Some(model) = self.model.take() {
            drop(model);
        }
        if !self.conditioner_released {
            self.conditioner_lease.release();
            self.conditioner_released = true;
        }
    }
}

impl<M, S, C> Drop for H3PrivateQwenResources<M, S, C>
where
    C: H3PrivateQwenConditionerLease,
{
    fn drop(&mut self) {
        self.release_conditioner();
    }
}

/// One-shot private Qwen component. Successful or failed encoding always
/// drops model allocations and releases the conditioner reservation before
/// the denoising phase can start.
pub(crate) struct H3PrivateQwenAdapter<'authority, C, E, A>
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateQwenArtifactLease,
{
    resources: H3PrivateQwenResources<LoadedH3QwenNvfp4Conditioner, H3PrivateQwenSupport, C>,
    execution_lease: E,
    artifact_lease: A,
    authority: &'authority FrozenH3FactoryAuthority,
}

/// Authenticate the exact Qwen checkpoint without constructing model tensors.
/// This is the atomic preparation counterpart of `load_authorized`: it uses
/// the same frozen authority, lease graph, memory facts, and bounded polling,
/// then returns the non-Clone opened object for the target-budget binder.
pub(crate) fn open_authorized_private_qwen_authority(
    weights_path: &Path,
    support: &H3PrivateQwenSupport,
    route: &H3PrivateQwenOpenRouteAuthority,
    authority: &FrozenH3FactoryAuthority,
    checkpoint: &mut dyn H3PipelineCheckpoint,
) -> Result<H3AuthenticatedQwenNvfp4Authority> {
    route.revalidate(authority, support)?;
    let mut observer =
        H3PrivateQwenLoadCheckpoint::new(checkpoint, H3PipelinePhase::QwenLoadChunk, || {
            route.revalidate(authority, support)
        });
    // SAFETY: the payload-free route binds the same immutable factory,
    // support, artifact, placement, and owner facts that concrete leases must
    // re-prove at load time. This open path creates no model tensors.
    let opened = unsafe {
        open_h3_qwen_nvfp4_authority_after_authorization(
            weights_path,
            route.placement(),
            &mut observer,
        )
    };
    let opened = observer.finish(opened)?;
    opened.revalidate()?;
    if opened.artifact_identity_sha256() != route.weight_identity_sha256
        || opened.header_identity_sha256() != route.weight_header_identity_sha256
        || opened.policy_identity_sha256() != route.weight_policy_identity_sha256
        || opened.memory_facts() != &route.memory_facts
    {
        bail!("authenticated H3 Qwen opened evidence differs from its payload-free route")
    }
    route.revalidate(authority, support)?;
    Ok(opened)
}

impl<'authority, C, E, A> H3PrivateQwenAdapter<'authority, C, E, A>
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateQwenArtifactLease,
{
    /// The only production construction path. All independent frozen, lease,
    /// support, route, policy, and memory bindings are checked before the
    /// loader callback is invoked and therefore before the checkpoint path is
    /// opened or any weight storage is allocated.
    pub(crate) fn load_authorized(
        weights_path: &Path,
        support: H3PrivateQwenSupport,
        conditioner_lease: C,
        execution_lease: E,
        artifact_lease: A,
        authority: &'authority FrozenH3FactoryAuthority,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Self> {
        let mut resources = H3PrivateQwenResources {
            model: None,
            support,
            conditioner_lease,
            conditioner_released: false,
        };
        let device = resources.conditioner_lease.device();
        let placement = if device.is_cpu() {
            H3QwenNvfp4RuntimePlacement::Cpu
        } else {
            H3QwenNvfp4RuntimePlacement::Accelerated
        };
        let memory_facts = released_h3_qwen_nvfp4_runtime_memory_facts(device)?;
        validate_preload_authorities(
            &resources.support,
            &resources.conditioner_lease,
            &execution_lease,
            &artifact_lease,
            authority,
            &memory_facts,
        )?;

        let opened = {
            let mut observer = H3PrivateQwenLoadCheckpoint::new(
                checkpoint,
                H3PipelinePhase::QwenLoadChunk,
                || {
                    poll_active_authorities(
                        &resources.support,
                        &resources.conditioner_lease,
                        &execution_lease,
                        &artifact_lease,
                        authority,
                    )
                },
            );
            // SAFETY: every immutable attempt/lease binding was validated
            // above and is polled at each bounded authentication checkpoint.
            let result = unsafe {
                open_h3_qwen_nvfp4_authority_after_authorization(
                    weights_path,
                    placement,
                    &mut observer,
                )
            };
            observer.finish(result)?
        };
        validate_opened_qwen_authority(&opened, &artifact_lease, authority, &memory_facts)?;
        poll_active_authorities(
            &resources.support,
            &resources.conditioner_lease,
            &execution_lease,
            &artifact_lease,
            authority,
        )?;

        let loaded = {
            let config = resources.support.conditioner_config();
            let device = resources.conditioner_lease.device();
            let mut observer = H3PrivateQwenLoadCheckpoint::new(
                checkpoint,
                H3PipelinePhase::QwenLoadChunk,
                || {
                    poll_active_authorities(
                        &resources.support,
                        &resources.conditioner_lease,
                        &execution_lease,
                        &artifact_lease,
                        authority,
                    )
                },
            );
            // SAFETY: `opened` is the non-Clone authority validated at the
            // allocation boundary above. It is consumed exactly once here.
            let result = unsafe {
                load_h3_qwen_nvfp4_conditioner_from_authority(opened, config, device, &mut observer)
            };
            observer.finish(result)?
        };
        validate_output_dtype(loaded.dtype_profile().output_dtype)?;
        resources.model = Some(loaded);
        let adapter = Self {
            resources,
            execution_lease,
            artifact_lease,
            authority,
        };
        adapter.validate_active_authorities()?;
        Ok(adapter)
    }

    /// Consume an already-authenticated, non-Clone Qwen authority. Unlike the
    /// legacy private helper above, this path cannot reopen a caller-selected
    /// pathname between target-budget preparation and allocation.
    pub(crate) fn load_authorized_from_opened(
        opened: H3AuthenticatedQwenNvfp4Authority,
        support: H3PrivateQwenSupport,
        conditioner_lease: C,
        execution_lease: E,
        artifact_lease: A,
        authority: &'authority FrozenH3FactoryAuthority,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Self> {
        let mut resources = H3PrivateQwenResources {
            model: None,
            support,
            conditioner_lease,
            conditioner_released: false,
        };
        let memory_facts = opened.memory_facts().clone();
        let expected_placement = if resources.conditioner_lease.device().is_cpu() {
            H3QwenNvfp4RuntimePlacement::Cpu
        } else {
            H3QwenNvfp4RuntimePlacement::Accelerated
        };
        if opened.placement() != expected_placement {
            bail!("private H3 opened Qwen placement differs before allocation")
        }
        validate_preload_authorities(
            &resources.support,
            &resources.conditioner_lease,
            &execution_lease,
            &artifact_lease,
            authority,
            &memory_facts,
        )?;
        validate_opened_qwen_authority(&opened, &artifact_lease, authority, &memory_facts)?;
        poll_active_authorities(
            &resources.support,
            &resources.conditioner_lease,
            &execution_lease,
            &artifact_lease,
            authority,
        )?;

        let loaded = {
            let config = resources.support.conditioner_config();
            let device = resources.conditioner_lease.device();
            let mut observer = H3PrivateQwenLoadCheckpoint::new(
                checkpoint,
                H3PipelinePhase::QwenLoadChunk,
                || {
                    poll_active_authorities(
                        &resources.support,
                        &resources.conditioner_lease,
                        &execution_lease,
                        &artifact_lease,
                        authority,
                    )
                },
            );
            // SAFETY: the opened authority and all attempt projections were
            // validated above and remain polled at every tensor-load boundary.
            let result = unsafe {
                load_h3_qwen_nvfp4_conditioner_from_authority(opened, config, device, &mut observer)
            };
            observer.finish(result)?
        };
        validate_output_dtype(loaded.dtype_profile().output_dtype)?;
        resources.model = Some(loaded);
        let adapter = Self {
            resources,
            execution_lease,
            artifact_lease,
            authority,
        };
        adapter.validate_active_authorities()?;
        Ok(adapter)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_with(
        weights_path: &Path,
        support: H3PrivateQwenSupport,
        conditioner_lease: C,
        execution_lease: E,
        artifact_lease: A,
        authority: &'authority FrozenH3FactoryAuthority,
        checkpoint: &mut dyn H3PipelineCheckpoint,
        loader: impl FnOnce(
            &Path,
            &mold_candle::minimax_h3::H3ConditionerConfig,
            &Device,
            &mut dyn H3QwenNvfp4LoadObserver,
        ) -> std::result::Result<
            LoadedH3QwenNvfp4Conditioner,
            H3QwenNvfp4RuntimeError,
        >,
    ) -> Result<Self> {
        let mut resources = H3PrivateQwenResources {
            model: None,
            support,
            conditioner_lease,
            conditioner_released: false,
        };
        let memory_facts =
            released_h3_qwen_nvfp4_runtime_memory_facts(resources.conditioner_lease.device())?;
        let loaded = invoke_authorized_loader(
            || {
                validate_preload_authorities(
                    &resources.support,
                    &resources.conditioner_lease,
                    &execution_lease,
                    &artifact_lease,
                    authority,
                    &memory_facts,
                )
            },
            || {
                let mut observer = H3PrivateQwenLoadCheckpoint::new(
                    checkpoint,
                    H3PipelinePhase::QwenLoadChunk,
                    || {
                        poll_active_authorities(
                            &resources.support,
                            &resources.conditioner_lease,
                            &execution_lease,
                            &artifact_lease,
                            authority,
                        )
                    },
                );
                let result = loader(
                    weights_path,
                    resources.support.conditioner_config(),
                    resources.conditioner_lease.device(),
                    &mut observer,
                );
                observer.finish(result)
            },
        )?;
        validate_output_dtype(loaded.dtype_profile().output_dtype)?;
        resources.model = Some(loaded);
        let adapter = Self {
            resources,
            execution_lease,
            artifact_lease,
            authority,
        };
        adapter.validate_active_authorities()?;
        Ok(adapter)
    }

    pub(crate) fn model(&self) -> &str {
        self.resources.support.model()
    }

    pub(crate) const fn task(&self) -> Task {
        self.resources.support.task()
    }

    pub(crate) fn support_identity_sha256(&self) -> &str {
        self.resources.support.support_identity_sha256()
    }

    pub(crate) fn execution_lease(&self) -> &E {
        &self.execution_lease
    }

    pub(crate) fn artifact_lease(&self) -> &A {
        &self.artifact_lease
    }

    pub(crate) fn encode_fl2va(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        let prepared = (|| {
            self.require_task(Task::Fl2va)?;
            self.validate_active_authorities()?;
            prepare_fl2va_conditioner_input(
                self.resources.support.tokenizer(),
                prompt,
                endpoints,
                self.resources.conditioner_lease.device(),
            )
        })();
        self.encode_prepared(prepared, checkpoint)
    }

    pub(crate) fn encode_ref2va(
        &mut self,
        prompt: &str,
        references: &[H3ReferencePresentation],
        media: &H3ReferenceMediaAdapter,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        let prepared = (|| {
            self.require_task(Task::Ref2va)?;
            self.validate_active_authorities()?;
            prepare_ref2va_conditioner_input(
                self.resources.support.tokenizer(),
                prompt,
                references,
                media,
                self.resources.conditioner_lease.device(),
            )
        })();
        self.encode_prepared(prepared, checkpoint)
    }

    fn require_task(&self, task: Task) -> Result<()> {
        if self.resources.support.task() != task || self.authority.task() != task {
            bail!("private H3 Qwen adapter was invoked for the wrong task partition")
        }
        Ok(())
    }

    fn encode_prepared(
        &mut self,
        prepared: Result<(H3ConditionerInput, Vec<H3ModalityTag>)>,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        let prepared = complete_preparation(prepared, |(input, _)| {
            self.validate_active_authorities()?;
            validate_prepared_authority(self.authority, input)
        });
        let (input, tags) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.resources.release_conditioner();
                return Err(error);
            }
        };
        let vision_passes = usize::from(input.image.is_some()) + usize::from(input.video.is_some());
        let encode_result = {
            let model = self
                .resources
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("private H3 Qwen conditioner was already released"))?;
            encode_with_typed_checkpoint(
                checkpoint,
                vision_passes,
                || {
                    poll_active_authorities(
                        &self.resources.support,
                        &self.resources.conditioner_lease,
                        &self.execution_lease,
                        &self.artifact_lease,
                        self.authority,
                    )
                },
                |callback| model.encode(&input, callback),
            )
        };
        let encode_result = complete_encode(encode_result, || self.validate_active_authorities());

        // Model storage must drop before C is released on success, failure,
        // cancellation, or post-encode revocation. E/A are lightweight
        // projections over the outer attempt and are validated before and
        // after tensor transfer.
        self.resources.release_conditioner();
        let states = encode_result?;
        validate_output_dtype(states.dtype())?;
        self.validate_continuing_authorities()?;
        let states = states.to_device(self.execution_lease.device())?;
        self.validate_continuing_authorities()?;
        Ok(H3TextConditioning {
            states,
            tags,
            #[cfg(test)]
            lifetime_probe: None,
        })
    }

    fn validate_active_authorities(&self) -> Result<()> {
        let model = self
            .resources
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("private H3 Qwen conditioner was already released"))?;
        self.resources.support.revalidate()?;
        model.revalidate_artifact()?;
        self.authority.validate_engine_seam(
            self.resources.support.model(),
            self.authority.device_ordinal(),
            self.authority.block_offload(),
        )?;

        let loaded = LoadedBindingClaims {
            model: self.resources.support.model().to_owned(),
            task: self.resources.support.task(),
            support_identity_sha256: self.resources.support.support_identity_sha256().to_owned(),
            weight_identity_sha256: model.artifact_identity_sha256().to_owned(),
            weight_header_identity_sha256: model.header_identity_sha256().to_owned(),
            weight_policy_identity_sha256: model.policy_identity_sha256().to_owned(),
            resident_language_layers: model.resident_language_layers(),
        };
        let frozen = frozen_binding_claims(self.authority)?;
        let leases = lease_binding_claims(
            &self.resources.conditioner_lease,
            &self.execution_lease,
            &self.artifact_lease,
            model
                .device()
                .same_device(self.resources.conditioner_lease.device()),
        );
        validate_binding(&loaded, &frozen, &leases)?;
        validate_parameter_memory(self.authority, model.memory_facts())
    }

    pub(crate) fn validate_continuing_authorities(&self) -> Result<()> {
        self.resources.support.revalidate()?;
        self.authority.validate_engine_seam(
            self.resources.support.model(),
            self.authority.device_ordinal(),
            self.authority.block_offload(),
        )?;
        let (component_content, component_validation) =
            self.authority.conditioner_component_authority();
        let qwen_policy = match self.authority.quantization() {
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                qwen_policy_sha256,
                ..
            } => qwen_policy_sha256,
            H3FactoryQuantizationAuthority::OfficialBf16 => {
                bail!("private H3 Qwen authority changed after encoding")
            }
        };
        if !self.execution_lease.is_active()
            || self.execution_lease.lease_id().trim().is_empty()
            || self.execution_lease.device_id() != self.authority.device_id()
            || self.execution_lease.execution_fingerprint()
                != self.authority.execution_fingerprint()
            || self.execution_lease.backend()
                != H3CandleBackendDevice::from_compute_capability(
                    self.authority.compute_capability(),
                )
            || !self
                .execution_lease
                .backend()
                .matches_candle(self.execution_lease.device())
            || !self.artifact_lease.is_active()
            || self.artifact_lease.factory_identity_sha256() != self.authority.identity_sha256()
            || self.artifact_lease.component_set_identity()
                != self.authority.component_set_identity_sha256()
            || self.artifact_lease.conditioner_component_content_sha256() != component_content
            || self
                .artifact_lease
                .conditioner_component_validation_sha256()
                != component_validation
            || self.artifact_lease.support_identity_sha256()
                != self.resources.support.support_identity_sha256()
            || self.artifact_lease.weight_identity_sha256() != H3_QWEN_NVFP4_AWQ_SHA256
            || self.artifact_lease.weight_policy_identity_sha256()
                != H3_QWEN_NVFP4_AWQ_POLICY_SHA256
            || self.artifact_lease.weight_policy_identity_sha256() != qwen_policy
            || require_sha256(
                self.artifact_lease.weight_header_identity_sha256(),
                "weight header",
            )
            .is_err()
        {
            bail!("private H3 Qwen execution or artifact authority was revoked after encoding")
        }
        Ok(())
    }
}

fn complete_preparation<T>(
    prepared: Result<T>,
    validate: impl FnOnce(&T) -> Result<()>,
) -> Result<T> {
    let prepared = prepared?;
    validate(&prepared)?;
    Ok(prepared)
}

fn complete_encode<T>(encoded: Result<T>, validate: impl FnOnce() -> Result<()>) -> Result<T> {
    match encoded {
        Ok(value) => {
            validate()?;
            Ok(value)
        }
        Err(first_error) => {
            // Poll after a failed callback as well, but preserve the first
            // typed callback/Candle error if execution continued after it.
            let _ = validate();
            Err(first_error)
        }
    }
}

fn invoke_authorized_loader<T>(
    validate_preload: impl FnOnce() -> Result<()>,
    loader: impl FnOnce() -> Result<T>,
) -> Result<T> {
    validate_preload()?;
    loader()
}

#[derive(Clone, Debug)]
struct LoadedBindingClaims {
    model: String,
    task: Task,
    support_identity_sha256: String,
    weight_identity_sha256: String,
    weight_header_identity_sha256: String,
    weight_policy_identity_sha256: String,
    resident_language_layers: usize,
}

#[derive(Clone, Debug)]
struct FrozenBindingClaims {
    model: String,
    task: Task,
    factory_identity_sha256: String,
    component_set_identity_sha256: String,
    conditioner_component_content_sha256: String,
    conditioner_component_validation_sha256: String,
    device_id: String,
    execution_fingerprint: String,
    conditioner_placement: H3FactoryConditionerPlacement,
    qwen_policy_identity_sha256: String,
    compute_capability: Option<(u16, u16)>,
}

#[derive(Clone, Debug)]
struct LeaseBindingClaims {
    conditioner_factory_identity_sha256: String,
    conditioner_device_id: String,
    conditioner_execution_fingerprint: String,
    conditioner_lease_id: String,
    conditioner_active: bool,
    execution_device_id: String,
    execution_fingerprint: String,
    execution_lease_id: String,
    execution_backend: H3CandleBackendDevice,
    execution_active: bool,
    artifact_factory_identity_sha256: String,
    artifact_component_set_identity_sha256: String,
    artifact_component_content_sha256: String,
    artifact_component_validation_sha256: String,
    artifact_support_identity_sha256: String,
    artifact_weight_identity_sha256: String,
    artifact_weight_header_identity_sha256: String,
    artifact_weight_policy_identity_sha256: String,
    artifact_active: bool,
    loaded_matches_conditioner_device: bool,
    conditioner_is_cpu: bool,
    conditioner_is_cuda: bool,
    conditioner_matches_execution_device: bool,
    execution_is_cuda: bool,
}

fn frozen_binding_claims(authority: &FrozenH3FactoryAuthority) -> Result<FrozenBindingClaims> {
    let qwen_policy = match authority.quantization() {
        H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            qwen_policy_sha256,
            ..
        } => qwen_policy_sha256,
        H3FactoryQuantizationAuthority::OfficialBf16 => {
            bail!("private H3 NVFP4 Qwen adapter requires the frozen Comfy layout")
        }
    };
    let (component_content, component_validation) = authority.conditioner_component_authority();
    Ok(FrozenBindingClaims {
        model: authority.canonical_model().to_owned(),
        task: authority.task(),
        factory_identity_sha256: authority.identity_sha256().to_owned(),
        component_set_identity_sha256: authority.component_set_identity_sha256().to_owned(),
        conditioner_component_content_sha256: component_content.to_owned(),
        conditioner_component_validation_sha256: component_validation.to_owned(),
        device_id: authority.device_id().to_owned(),
        execution_fingerprint: authority.execution_fingerprint().to_owned(),
        conditioner_placement: authority.conditioner_placement(),
        qwen_policy_identity_sha256: qwen_policy.to_owned(),
        compute_capability: authority.compute_capability(),
    })
}

fn lease_binding_claims<C, E, A>(
    conditioner: &C,
    execution: &E,
    artifact: &A,
    loaded_matches_conditioner_device: bool,
) -> LeaseBindingClaims
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateQwenArtifactLease,
{
    LeaseBindingClaims {
        conditioner_factory_identity_sha256: conditioner.factory_identity_sha256().to_owned(),
        conditioner_device_id: conditioner.device_id().to_owned(),
        conditioner_execution_fingerprint: conditioner.execution_fingerprint().to_owned(),
        conditioner_lease_id: conditioner.lease_id().to_owned(),
        conditioner_active: conditioner.is_active(),
        execution_device_id: execution.device_id().to_owned(),
        execution_fingerprint: execution.execution_fingerprint().to_owned(),
        execution_lease_id: execution.lease_id().to_owned(),
        execution_backend: execution.backend(),
        execution_active: execution.is_active(),
        artifact_factory_identity_sha256: artifact.factory_identity_sha256().to_owned(),
        artifact_component_set_identity_sha256: artifact.component_set_identity().to_owned(),
        artifact_component_content_sha256: artifact
            .conditioner_component_content_sha256()
            .to_owned(),
        artifact_component_validation_sha256: artifact
            .conditioner_component_validation_sha256()
            .to_owned(),
        artifact_support_identity_sha256: artifact.support_identity_sha256().to_owned(),
        artifact_weight_identity_sha256: artifact.weight_identity_sha256().to_owned(),
        artifact_weight_header_identity_sha256: artifact.weight_header_identity_sha256().to_owned(),
        artifact_weight_policy_identity_sha256: artifact.weight_policy_identity_sha256().to_owned(),
        artifact_active: artifact.is_active(),
        loaded_matches_conditioner_device,
        conditioner_is_cpu: conditioner.device().is_cpu(),
        conditioner_is_cuda: conditioner.device().is_cuda(),
        conditioner_matches_execution_device: conditioner.device().same_device(execution.device()),
        execution_is_cuda: execution.device().is_cuda(),
    }
}

fn validate_parameter_memory(
    authority: &FrozenH3FactoryAuthority,
    facts: &H3QwenNvfp4RuntimeMemoryFacts,
) -> Result<()> {
    if authority.qwen_parameter_bytes() != facts.effective_parameter_bytes {
        bail!(
            "private H3 Qwen frozen parameter authority is {}, authenticated runtime requires exactly {} source-encoded bytes",
            authority.qwen_parameter_bytes(),
            facts.effective_parameter_bytes
        )
    }
    if authority.qwen_host_resident_parameter_bytes() != facts.host_resident_parameter_bytes
        || authority.qwen_device_resident_parameter_bytes() != facts.device_resident_parameter_bytes
    {
        bail!(
            "private H3 Qwen frozen residency is host/device {}/{}, authenticated runtime requires exactly {}/{} bytes",
            authority.qwen_host_resident_parameter_bytes(),
            authority.qwen_device_resident_parameter_bytes(),
            facts.host_resident_parameter_bytes,
            facts.device_resident_parameter_bytes
        )
    }
    // The host ledger charges the loader's own transients and retained
    // metadata. Bind both to the authenticated runtime facts here, at the same
    // seam that pins parameter residency, so a target budget cannot invent
    // them: `maximum_tensor_staging_bytes` is the largest single tensor the
    // loader reads (held twice, as the read `Vec` and its `from_raw_buffer`
    // copy) and `retained_raw_header_bytes` is the parsed header it keeps.
    if authority.qwen_maximum_tensor_staging_bytes() != facts.maximum_tensor_staging_bytes
        || authority.qwen_retained_raw_header_bytes() != facts.retained_raw_header_bytes
    {
        bail!(
            "private H3 Qwen frozen loader staging/header authority is {}/{}, authenticated runtime requires exactly {}/{} bytes",
            authority.qwen_maximum_tensor_staging_bytes(),
            authority.qwen_retained_raw_header_bytes(),
            facts.maximum_tensor_staging_bytes,
            facts.retained_raw_header_bytes
        )
    }
    let residency_is_bound = match authority.conditioner_placement() {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
            facts.host_resident_parameter_bytes > 0 && facts.device_resident_parameter_bytes > 0
        }
        H3FactoryConditionerPlacement::HostCpuThenDrop => {
            facts.host_resident_parameter_bytes == facts.retained_parameter_bytes
                && facts.device_resident_parameter_bytes == 0
        }
    };
    if !residency_is_bound {
        bail!("private H3 Qwen runtime residency differs from frozen conditioner placement")
    }
    Ok(())
}

fn validate_output_dtype(dtype: DType) -> Result<()> {
    if dtype == DType::BF16 {
        Ok(())
    } else {
        bail!("private H3 Qwen output dtype is {dtype:?}, exact admission requires BF16 states")
    }
}

fn validate_preload_authorities<C, E, A>(
    support: &H3PrivateQwenSupport,
    conditioner: &C,
    execution: &E,
    artifact: &A,
    authority: &FrozenH3FactoryAuthority,
    memory_facts: &H3QwenNvfp4RuntimeMemoryFacts,
) -> Result<()>
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateQwenArtifactLease,
{
    support.revalidate()?;
    authority.validate_engine_seam(
        support.model(),
        authority.device_ordinal(),
        authority.block_offload(),
    )?;
    if artifact.weight_identity_sha256() != H3_QWEN_NVFP4_AWQ_SHA256
        || artifact.weight_policy_identity_sha256() != H3_QWEN_NVFP4_AWQ_POLICY_SHA256
    {
        bail!("private H3 Qwen artifact lease is not the released NVFP4-AWQ object")
    }
    validate_parameter_memory(authority, memory_facts)?;
    let loaded = LoadedBindingClaims {
        model: support.model().to_owned(),
        task: support.task(),
        support_identity_sha256: support.support_identity_sha256().to_owned(),
        weight_identity_sha256: artifact.weight_identity_sha256().to_owned(),
        weight_header_identity_sha256: artifact.weight_header_identity_sha256().to_owned(),
        weight_policy_identity_sha256: artifact.weight_policy_identity_sha256().to_owned(),
        resident_language_layers: H3_SELECTED_LANGUAGE_LAYERS,
    };
    let frozen = frozen_binding_claims(authority)?;
    let leases = lease_binding_claims(conditioner, execution, artifact, true);
    validate_binding(&loaded, &frozen, &leases)
}

fn validate_opened_qwen_authority<A>(
    opened: &H3AuthenticatedQwenNvfp4Authority,
    artifact: &A,
    authority: &FrozenH3FactoryAuthority,
    expected_memory: &H3QwenNvfp4RuntimeMemoryFacts,
) -> Result<()>
where
    A: H3PrivateQwenArtifactLease,
{
    opened.revalidate()?;
    require_sha256(opened.identity_sha256(), "opened authority")?;
    if opened.artifact_file_bytes() == 0
        || opened.artifact_identity_sha256() != artifact.weight_identity_sha256()
        || opened.header_identity_sha256() != artifact.weight_header_identity_sha256()
        || opened.policy_identity_sha256() != artifact.weight_policy_identity_sha256()
        || opened.memory_facts() != expected_memory
    {
        bail!(
            "authenticated H3 Qwen opened evidence differs from the frozen artifact or memory authority"
        )
    }
    validate_parameter_memory(authority, opened.memory_facts())
}

/// Cheap checkpoint poll. Expensive descriptor rehash/reopen validation runs
/// at the preload, pre-encode, and post-encode boundaries; every bounded load
/// and model checkpoint still proves all live lease identities and routes.
fn poll_active_authorities<C, E, A>(
    support: &H3PrivateQwenSupport,
    conditioner: &C,
    execution: &E,
    artifact: &A,
    authority: &FrozenH3FactoryAuthority,
) -> Result<()>
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateQwenArtifactLease,
{
    let (component_content, component_validation) = authority.conditioner_component_authority();
    let qwen_policy = match authority.quantization() {
        H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            qwen_policy_sha256,
            ..
        } => qwen_policy_sha256,
        H3FactoryQuantizationAuthority::OfficialBf16 => {
            bail!("private H3 Qwen authority changed to the official BF16 layout")
        }
    };
    let conditioner_route_matches = match authority.conditioner_placement() {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
            conditioner.device_id() == authority.device_id()
                && conditioner.device().is_cuda()
                && conditioner.device().same_device(execution.device())
        }
        H3FactoryConditionerPlacement::HostCpuThenDrop => {
            conditioner.device_id() == "cpu" && conditioner.device().is_cpu()
        }
    };
    if support.model() != authority.canonical_model()
        || support.task() != authority.task()
        || !conditioner.is_active()
        || conditioner.lease_id().trim().is_empty()
        || conditioner.factory_identity_sha256() != authority.identity_sha256()
        || conditioner.execution_fingerprint() != authority.execution_fingerprint()
        || !conditioner_route_matches
        || !execution.is_active()
        || execution.lease_id().trim().is_empty()
        || execution.device_id() != authority.device_id()
        || execution.execution_fingerprint() != authority.execution_fingerprint()
        || execution.backend()
            != H3CandleBackendDevice::from_compute_capability(authority.compute_capability())
        || !execution.backend().matches_candle(execution.device())
        || !artifact.is_active()
        || artifact.factory_identity_sha256() != authority.identity_sha256()
        || artifact.component_set_identity() != authority.component_set_identity_sha256()
        || artifact.conditioner_component_content_sha256() != component_content
        || artifact.conditioner_component_validation_sha256() != component_validation
        || artifact.support_identity_sha256() != support.support_identity_sha256()
        || artifact.weight_identity_sha256() != H3_QWEN_NVFP4_AWQ_SHA256
        || artifact.weight_policy_identity_sha256() != H3_QWEN_NVFP4_AWQ_POLICY_SHA256
        || artifact.weight_policy_identity_sha256() != qwen_policy
        || require_sha256(artifact.weight_header_identity_sha256(), "weight header").is_err()
    {
        bail!("private H3 Qwen authority was revoked or cross-routed at a checkpoint")
    }
    Ok(())
}

fn validate_binding(
    loaded: &LoadedBindingClaims,
    frozen: &FrozenBindingClaims,
    leases: &LeaseBindingClaims,
) -> Result<()> {
    for (label, digest) in [
        ("support", loaded.support_identity_sha256.as_str()),
        ("weight", loaded.weight_identity_sha256.as_str()),
        (
            "weight header",
            loaded.weight_header_identity_sha256.as_str(),
        ),
        (
            "weight policy",
            loaded.weight_policy_identity_sha256.as_str(),
        ),
        ("factory", frozen.factory_identity_sha256.as_str()),
        (
            "component set",
            frozen.component_set_identity_sha256.as_str(),
        ),
        (
            "conditioner component content",
            frozen.conditioner_component_content_sha256.as_str(),
        ),
        (
            "conditioner component validation",
            frozen.conditioner_component_validation_sha256.as_str(),
        ),
        ("execution", frozen.execution_fingerprint.as_str()),
    ] {
        require_sha256(digest, label)?;
    }
    if loaded.model != frozen.model || loaded.task != frozen.task {
        bail!("private H3 Qwen loaded model/task differs from frozen factory authority")
    }
    if loaded.resident_language_layers != H3_SELECTED_LANGUAGE_LAYERS {
        bail!("private H3 Qwen runtime did not retain exactly layers 0-49")
    }
    if loaded.support_identity_sha256 != leases.artifact_support_identity_sha256
        || loaded.weight_identity_sha256 != leases.artifact_weight_identity_sha256
        || loaded.weight_header_identity_sha256 != leases.artifact_weight_header_identity_sha256
        || loaded.weight_policy_identity_sha256 != leases.artifact_weight_policy_identity_sha256
    {
        bail!("private H3 Qwen support/checkpoint came from a different artifact lease")
    }
    if loaded.weight_policy_identity_sha256 != frozen.qwen_policy_identity_sha256 {
        bail!("private H3 Qwen checkpoint policy differs from frozen factory quantization")
    }
    if leases.conditioner_factory_identity_sha256 != frozen.factory_identity_sha256
        || leases.artifact_factory_identity_sha256 != frozen.factory_identity_sha256
        || leases.artifact_component_set_identity_sha256 != frozen.component_set_identity_sha256
        || leases.artifact_component_content_sha256 != frozen.conditioner_component_content_sha256
        || leases.artifact_component_validation_sha256
            != frozen.conditioner_component_validation_sha256
    {
        bail!("private H3 Qwen lease differs from frozen component/factory authority")
    }
    if !leases.conditioner_active
        || leases.conditioner_lease_id.trim().is_empty()
        || leases.conditioner_execution_fingerprint != frozen.execution_fingerprint
        || !leases.loaded_matches_conditioner_device
    {
        bail!("private H3 Qwen conditioner lease is inactive or cross-routed")
    }
    let conditioner_route_matches = match frozen.conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
            leases.conditioner_device_id == frozen.device_id
                && leases.conditioner_is_cuda
                && leases.conditioner_matches_execution_device
        }
        H3FactoryConditionerPlacement::HostCpuThenDrop => {
            leases.conditioner_device_id == "cpu" && leases.conditioner_is_cpu
        }
    };
    if !conditioner_route_matches {
        bail!("private H3 Qwen loaded device differs from frozen conditioner placement")
    }
    let expected_backend =
        H3CandleBackendDevice::from_compute_capability(frozen.compute_capability);
    if !leases.execution_active
        || leases.execution_lease_id.trim().is_empty()
        || leases.execution_device_id != frozen.device_id
        || leases.execution_fingerprint != frozen.execution_fingerprint
        || leases.execution_backend != expected_backend
        || !leases.execution_is_cuda
    {
        bail!("private H3 Qwen execution lease differs from the frozen CUDA route")
    }
    if !leases.artifact_active {
        bail!("private H3 Qwen artifact lease was revoked")
    }
    Ok(())
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        bail!("private H3 Qwen {label} identity is not lowercase SHA-256")
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_prepared_authority(
    authority: &FrozenH3FactoryAuthority,
    input: &H3ConditionerInput,
) -> Result<()> {
    let (text_rows, vision_rows, minimum_live_activation_bytes) =
        prepared_qwen_activation_floor(input)?;
    if text_rows != authority.qwen_output_text_rows() || vision_rows != authority.qwen_vision_rows()
    {
        bail!(
            "private H3 Qwen prepared rows ({text_rows} text, {vision_rows} vision) differ from frozen admission ({}, {})",
            authority.qwen_output_text_rows(),
            authority.qwen_vision_rows()
        )
    }
    if authority.qwen_activation_workspace_bytes() < minimum_live_activation_bytes {
        bail!(
            "private H3 Qwen activation authority is undercharged: {} bytes frozen, {minimum_live_activation_bytes} bytes are unavoidably live before encode",
            authority.qwen_activation_workspace_bytes()
        )
    }
    Ok(())
}

/// Exact unavoidable live storage at the encode boundary: all prepared input
/// tensors plus the returned BF16 `[1, sequence, 5120]` states, which coexist.
/// This is a lower bound, not a claim about measured peak allocator or kernel
/// workspace use; admission's independently qualified workspace must cover it.
fn prepared_qwen_activation_floor(input: &H3ConditionerInput) -> Result<(u64, u64, u64)> {
    let (batch, sequence) = input.input_ids.dims2()?;
    if batch != 1 || sequence == 0 || input.input_ids.dtype() != candle_core::DType::U32 {
        bail!("private H3 Qwen input ids must be nonempty U32 [1, sequence]")
    }
    if input.position_ids.dims() != [3, 1, sequence]
        || input.position_ids.dtype() != candle_core::DType::U32
    {
        bail!("private H3 Qwen positions must be U32 [3, 1, sequence]")
    }
    let mut prepared_bytes = tensor_storage_bytes(&input.input_ids)?
        .checked_add(tensor_storage_bytes(&input.position_ids)?)
        .ok_or_else(|| anyhow!("private H3 Qwen prepared input bytes overflow"))?;
    let mut vision_rows = 0_u64;
    for vision in [input.image.as_ref(), input.video.as_ref()]
        .into_iter()
        .flatten()
    {
        let (patches, patch_width) = vision.pixel_values.dims2()?;
        let (grids, columns) = vision.grid_thw.dims2()?;
        if patches == 0
            || patch_width != 1_536
            || vision.pixel_values.dtype() != candle_core::DType::F32
            || grids == 0
            || columns != 3
            || vision.grid_thw.dtype() != candle_core::DType::U32
        {
            bail!("private H3 Qwen vision inputs differ from released processor geometry")
        }
        vision_rows = vision_rows
            .checked_add(u64::try_from(patches)?)
            .ok_or_else(|| anyhow!("private H3 Qwen vision row count overflows"))?;
        let pixel_bytes = tensor_storage_bytes(&vision.pixel_values)?;
        let grid_bytes = tensor_storage_bytes(&vision.grid_thw)?;
        prepared_bytes = prepared_bytes
            .checked_add(pixel_bytes)
            .and_then(|bytes| bytes.checked_add(grid_bytes))
            .ok_or_else(|| anyhow!("private H3 Qwen prepared vision bytes overflow"))?;
    }
    let text_rows = u64::try_from(sequence)?;
    let output_bytes = released_h3_private_qwen_output_tensor_bytes(text_rows)?;
    let minimum_live_activation_bytes = prepared_bytes
        .checked_add(output_bytes)
        .ok_or_else(|| anyhow!("private H3 Qwen live activation bytes overflow"))?;
    Ok((text_rows, vision_rows, minimum_live_activation_bytes))
}

fn tensor_storage_bytes(tensor: &Tensor) -> Result<u64> {
    u64::try_from(tensor.elem_count())?
        .checked_mul(u64::try_from(tensor.dtype().size_in_bytes())?)
        .ok_or_else(|| anyhow!("private H3 Qwen tensor storage bytes overflow"))
}

fn encode_with_typed_checkpoint(
    checkpoint: &mut dyn H3PipelineCheckpoint,
    vision_passes: usize,
    mut validate_authorities: impl FnMut() -> Result<()>,
    encode: impl FnOnce(
        &mut dyn FnMut(ConditionerCheckpoint) -> candle_core::Result<()>,
    ) -> candle_core::Result<Tensor>,
) -> Result<Tensor> {
    let mut mapper = ConditionerProgressMapper::new(vision_passes)?;
    let mut first_error = None;
    let output = encode(&mut |conditioner_checkpoint| {
        if first_error.is_some() {
            return Err(candle_core::Error::Msg(
                "private H3 Qwen callback continued after cancellation".into(),
            ));
        }
        let result = validate_authorities()
            .and_then(|()| mapper.map(conditioner_checkpoint))
            .and_then(|event| checkpoint.checkpoint(event));
        if let Err(error) = result {
            first_error = Some(error);
            return Err(candle_core::Error::Msg(
                "private H3 Qwen checkpoint callback failed".into(),
            ));
        }
        Ok(())
    });
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(output?)
}

struct ConditionerProgressMapper {
    vision_passes: usize,
    completed_vision_passes: usize,
    current_vision_layer: Option<usize>,
    token_seen: bool,
    language_started: bool,
    language_layer: usize,
    last_completed: usize,
    total: usize,
}

impl ConditionerProgressMapper {
    fn new(vision_passes: usize) -> Result<Self> {
        if vision_passes > 2 {
            bail!("private H3 Qwen supports at most image and video vision passes")
        }
        let total = 1_usize
            .checked_add(
                vision_passes
                    .checked_mul(QWEN_VISION_CHECKPOINTS_PER_PASS)
                    .ok_or_else(|| anyhow!("private H3 Qwen progress total overflows"))?,
            )
            .and_then(|value| value.checked_add(H3_SELECTED_LANGUAGE_LAYERS))
            .ok_or_else(|| anyhow!("private H3 Qwen progress total overflows"))?;
        Ok(Self {
            vision_passes,
            completed_vision_passes: 0,
            current_vision_layer: None,
            token_seen: false,
            language_started: false,
            language_layer: 0,
            last_completed: 0,
            total,
        })
    }

    fn map(&mut self, checkpoint: ConditionerCheckpoint) -> Result<H3PipelineEvent> {
        let completed = match checkpoint {
            ConditionerCheckpoint::TokenEmbedding
                if !self.token_seen
                    && self.completed_vision_passes == 0
                    && !self.language_started =>
            {
                self.token_seen = true;
                1
            }
            ConditionerCheckpoint::VisionPatchEmbedding
                if self.token_seen
                    && !self.language_started
                    && self.current_vision_layer.is_none()
                    && self.completed_vision_passes < self.vision_passes =>
            {
                self.current_vision_layer = Some(0);
                1 + self.completed_vision_passes * QWEN_VISION_CHECKPOINTS_PER_PASS + 1
            }
            ConditionerCheckpoint::VisionLayer { completed, total }
                if self.token_seen
                    && !self.language_started
                    && total == QWEN_VISION_LAYER_COUNT
                    && completed > 0
                    && completed <= total
                    && self.current_vision_layer == Some(completed - 1) =>
            {
                self.current_vision_layer = if completed == total {
                    self.completed_vision_passes += 1;
                    None
                } else {
                    Some(completed)
                };
                1 + (self.completed_vision_passes - usize::from(completed == total))
                    * QWEN_VISION_CHECKPOINTS_PER_PASS
                    + 1
                    + completed
            }
            ConditionerCheckpoint::LanguageLayer { completed, total }
                if self.token_seen
                    && self.completed_vision_passes == self.vision_passes
                    && self.current_vision_layer.is_none()
                    && total == H3_SELECTED_LANGUAGE_LAYERS
                    && completed > 0
                    && completed <= total
                    && completed == self.language_layer + 1 =>
            {
                self.language_started = true;
                self.language_layer = completed;
                1 + self.vision_passes * QWEN_VISION_CHECKPOINTS_PER_PASS + completed
            }
            other => bail!("private H3 Qwen emitted an invalid progress checkpoint {other:?}"),
        };
        if completed <= self.last_completed || completed > self.total {
            bail!("private H3 Qwen progress was non-monotonic or exceeded its exact total")
        }
        self.last_completed = completed;
        Ok(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncodeChunk,
            completed,
            total: self.total,
        })
    }
}

fn prepare_ref2va_conditioner_input(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    references: &[H3ReferencePresentation],
    media: &H3ReferenceMediaAdapter,
    device: &Device,
) -> Result<(H3ConditionerInput, Vec<H3ModalityTag>)> {
    let expected_indices = references
        .iter()
        .map(|reference| reference.index)
        .collect::<Vec<_>>();
    if expected_indices
        .iter()
        .enumerate()
        .any(|(offset, index)| usize::try_from(*index).ok() != Some(offset + 1))
        || media.normalized_indices() != expected_indices
    {
        bail!("private H3 Qwen reference order differs from normalized media authority")
    }
    let normalized = references
        .iter()
        .map(|reference| {
            media
                .normalized(reference.index)
                .ok_or_else(|| anyhow!("private H3 Qwen reference has no normalized media"))
        })
        .collect::<Result<Vec<_>>>()?;
    prepare_ref2va_conditioner_input_from_normalized(
        tokenizer,
        prompt,
        references,
        &normalized,
        device,
    )
}

fn prepare_ref2va_conditioner_input_from_normalized(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    references: &[H3ReferencePresentation],
    normalized: &[&H3NormalizedReference],
    device: &Device,
) -> Result<(H3ConditionerInput, Vec<H3ModalityTag>)> {
    if references.len() != normalized.len() {
        bail!("private H3 Qwen reference presentation/media counts differ")
    }
    let mut images = VisionAccumulator::default();
    let mut videos = VisionAccumulator::default();
    let mut presentation = Vec::with_capacity(references.len());
    for (offset, (reference, normalized)) in references.iter().zip(normalized).enumerate() {
        let expected_index = u32::try_from(offset + 1).context("H3 reference index exceeds u32")?;
        if reference.index != expected_index {
            bail!("private H3 Qwen reference presentation order changed")
        }
        match (&reference.presentation.kind, *normalized) {
            (
                RefPresentationKind::Image { vision_tokens },
                H3NormalizedReference::Image {
                    image,
                    vision_tokens: normalized_tokens,
                },
            ) if !reference.presentation.has_audio && vision_tokens == normalized_tokens => {
                let packed = pack_qwen_vision_u8(
                    image.as_raw(),
                    1,
                    usize::try_from(image.height())?,
                    usize::try_from(image.width())?,
                )?;
                if packed.grid.merged_tokens(QWEN_SPATIAL_MERGE_SIZE)? != *vision_tokens {
                    bail!("private H3 Qwen image presentation changed its vision-token geometry")
                }
                images.push(packed)?;
            }
            (
                RefPresentationKind::Video { blocks },
                H3NormalizedReference::Video {
                    qwen_frames,
                    qwen_source_indices,
                    qwen_block_timestamps_seconds,
                    audio,
                    ..
                },
            ) if reference.presentation.has_audio == audio.is_some() => {
                let first = qwen_frames
                    .first()
                    .ok_or_else(|| anyhow!("private H3 Qwen video has no sampled frames"))?;
                if qwen_source_indices.len() != qwen_frames.len()
                    || qwen_source_indices.windows(2).any(|pair| pair[0] > pair[1])
                    || qwen_frames
                        .iter()
                        .any(|frame| frame.dimensions() != first.dimensions())
                {
                    bail!("private H3 Qwen video sample order or geometry changed")
                }
                let mut rgb = Vec::new();
                for frame in qwen_frames {
                    rgb.extend_from_slice(frame.as_raw());
                }
                let packed = pack_qwen_vision_u8(
                    &rgb,
                    qwen_frames.len(),
                    usize::try_from(first.height())?,
                    usize::try_from(first.width())?,
                )?;
                let per_block = GridThw {
                    temporal: 1,
                    height: packed.grid.height,
                    width: packed.grid.width,
                }
                .merged_tokens(QWEN_SPATIAL_MERGE_SIZE)?;
                if blocks.len() != packed.grid.temporal
                    || qwen_block_timestamps_seconds.len() != blocks.len()
                    || blocks.iter().zip(qwen_block_timestamps_seconds).any(
                        |(block, expected_timestamp)| {
                            block.vision_tokens != per_block
                                || block.timestamp_seconds.to_bits() != expected_timestamp.to_bits()
                        },
                    )
                {
                    bail!(
                        "private H3 Qwen video presentation changed its temporal block timing or geometry"
                    )
                }
                videos.push(packed)?;
            }
            (RefPresentationKind::Audio, H3NormalizedReference::Audio { audio })
                if reference.presentation.has_audio && audio.samples_per_channel() > 0 => {}
            _ => bail!(
                "private H3 Qwen reference presentation differs from normalized media authority"
            ),
        }
        presentation.push(reference.presentation.clone());
    }

    let presentation =
        build_ref2va_presentation(tokenizer, prompt, &presentation, QWEN_MAXIMUM_TOKENS)?;
    let positions = qwen_mrope_positions(
        &presentation.qwen_mm_types,
        &images.grids,
        &videos.grids,
        QWEN_SPATIAL_MERGE_SIZE,
    )?;
    let sequence = presentation.token_ids.len();
    let input_ids = Tensor::from_vec(presentation.token_ids, (1, sequence), device)?;
    let position_ids = Tensor::from_vec(
        positions.into_iter().flatten().collect::<Vec<_>>(),
        (3, 1, sequence),
        device,
    )?;
    Ok((
        H3ConditionerInput {
            input_ids,
            position_ids,
            image: images.finish(device)?,
            video: videos.finish(device)?,
        },
        presentation.h3_tags,
    ))
}

#[derive(Default)]
struct VisionAccumulator {
    values: Vec<f32>,
    grids: Vec<GridThw>,
    patch_width: Option<usize>,
    patch_count: usize,
}

impl VisionAccumulator {
    fn push(&mut self, packed: PackedVisionPatches) -> Result<()> {
        if self
            .patch_width
            .replace(packed.patch_width)
            .is_some_and(|width| width != packed.patch_width)
        {
            bail!("private H3 Qwen references produced different patch widths")
        }
        self.patch_count = self
            .patch_count
            .checked_add(packed.patch_count)
            .ok_or_else(|| anyhow!("private H3 Qwen patch count overflow"))?;
        self.grids.push(packed.grid);
        self.values.extend(packed.values);
        Ok(())
    }

    fn finish(self, device: &Device) -> Result<Option<H3VisionInput>> {
        if self.grids.is_empty() {
            if !self.values.is_empty() || self.patch_count != 0 || self.patch_width.is_some() {
                bail!("private H3 Qwen empty vision accumulator retained patches")
            }
            return Ok(None);
        }
        let patch_width = self
            .patch_width
            .context("private H3 Qwen vision patches lost their width")?;
        let grid_values = self
            .grids
            .iter()
            .flat_map(|grid| {
                [
                    u32::try_from(grid.temporal),
                    u32::try_from(grid.height),
                    u32::try_from(grid.width),
                ]
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(H3VisionInput {
            pixel_values: Tensor::from_vec(self.values, (self.patch_count, patch_width), device)?,
            grid_thw: Tensor::from_vec(grid_values, (self.grids.len(), 3), device)?,
        }))
    }
}

/// Loading observer that retains a typed cancellation from an inference
/// checkpoint while satisfying Candle's boolean load callback.
struct H3PrivateQwenLoadCheckpoint<'a, V> {
    checkpoint: &'a mut dyn H3PipelineCheckpoint,
    phase: H3PipelinePhase,
    validate_authorities: V,
    first_error: Option<anyhow::Error>,
}

impl<'a, V> H3PrivateQwenLoadCheckpoint<'a, V>
where
    V: FnMut() -> Result<()>,
{
    fn new(
        checkpoint: &'a mut dyn H3PipelineCheckpoint,
        phase: H3PipelinePhase,
        validate_authorities: V,
    ) -> Self {
        Self {
            checkpoint,
            phase,
            validate_authorities,
            first_error: None,
        }
    }

    fn finish<T>(self, result: std::result::Result<T, H3QwenNvfp4RuntimeError>) -> Result<T> {
        if let Some(error) = self.first_error {
            Err(error)
        } else {
            Ok(result?)
        }
    }
}

impl<V> H3QwenNvfp4LoadObserver for H3PrivateQwenLoadCheckpoint<'_, V>
where
    V: FnMut() -> Result<()>,
{
    fn should_cancel(&mut self, event: &mold_candle::minimax_h3::H3QwenNvfp4LoadEvent) -> bool {
        if self.first_error.is_some() {
            return true;
        }
        let (completed, total) = match event {
            mold_candle::minimax_h3::H3QwenNvfp4LoadEvent::Authenticating {
                completed_bytes,
                total_bytes,
            }
            | mold_candle::minimax_h3::H3QwenNvfp4LoadEvent::LoadingTensor {
                completed_bytes,
                total_bytes,
                ..
            } => (*completed_bytes, *total_bytes),
        };
        let total = usize::try_from(total).unwrap_or(usize::MAX).max(1);
        let completed = usize::try_from(completed).unwrap_or(usize::MAX).min(total);
        let result = (self.validate_authorities)().and_then(|()| {
            self.checkpoint.checkpoint(H3PipelineEvent {
                phase: self.phase,
                completed,
                total,
            })
        });
        if let Err(error) = result {
            self.first_error = Some(error);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    #[cfg(feature = "cuda")]
    use std::path::PathBuf;
    #[cfg(feature = "cuda")]
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use candle_core::DType;
    use image::{Rgb, RgbImage};
    use mold_candle::minimax_h3::{
        H3QwenNvfp4LoadEvent, PresentationError, RefPresentation, VideoPresentationBlock,
    };
    use mold_core::minimax_h3 as contract;

    use super::*;
    use crate::attention::{AttentionBackend, AttentionChunkPolicy};
    use crate::progress::{is_inference_cancelled, InferenceCancelled};
    use crate::{H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole};

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn test_authority(
        qwen_parameter_bytes: u64,
        qwen_activation_workspace_bytes: u64,
        qwen_output_text_rows: u64,
        qwen_vision_rows: u64,
    ) -> FrozenH3FactoryAuthority {
        let runtime_facts = released_h3_qwen_nvfp4_runtime_memory_facts(&Device::Cpu).unwrap();
        let components = [
            H3FactoryComponentRole::Conditioner,
            H3FactoryComponentRole::Transformer,
            H3FactoryComponentRole::VisualVae,
            H3FactoryComponentRole::AudioVae,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, role)| {
            H3FactoryComponentAuthority::new(
                role,
                sha(char::from(b'1' + index as u8)),
                sha(char::from(b'5' + index as u8)),
            )
            .unwrap()
        })
        .collect();
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: sha('9'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes,
            qwen_host_resident_parameter_bytes: runtime_facts.host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: runtime_facts.device_resident_parameter_bytes,
            qwen_activation_workspace_bytes,
            qwen_maximum_tensor_staging_bytes: runtime_facts.maximum_tensor_staging_bytes,
            qwen_retained_raw_header_bytes: runtime_facts.retained_raw_header_bytes,
            qwen_output_text_rows,
            qwen_vision_rows,
            condition_visual_rows: 0,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "private-test-kernel".into(),
            attention_qualification_sha256: sha('a'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            attention_runtime: None,
            prepared_attempt: None,
            execution_budget_echo: None,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('b'),
                qwen_policy_sha256: H3_QWEN_NVFP4_AWQ_POLICY_SHA256.into(),
                pruned_adaln_table_sha256: sha('c'),
                turbo_adapter: None,
            },
            components,
        })
        .unwrap()
    }

    fn binding_claims() -> (LoadedBindingClaims, FrozenBindingClaims, LeaseBindingClaims) {
        let loaded = LoadedBindingClaims {
            model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            support_identity_sha256: sha('1'),
            weight_identity_sha256: sha('2'),
            weight_header_identity_sha256: sha('3'),
            weight_policy_identity_sha256: sha('4'),
            resident_language_layers: 50,
        };
        let frozen = FrozenBindingClaims {
            model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            factory_identity_sha256: sha('5'),
            component_set_identity_sha256: sha('6'),
            conditioner_component_content_sha256: sha('7'),
            conditioner_component_validation_sha256: sha('8'),
            device_id: "gpu-0".into(),
            execution_fingerprint: sha('9'),
            conditioner_placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
            qwen_policy_identity_sha256: sha('4'),
            compute_capability: Some((8, 9)),
        };
        let leases = LeaseBindingClaims {
            conditioner_factory_identity_sha256: sha('5'),
            conditioner_device_id: "gpu-0".into(),
            conditioner_execution_fingerprint: sha('9'),
            conditioner_lease_id: "conditioner-1".into(),
            conditioner_active: true,
            execution_device_id: "gpu-0".into(),
            execution_fingerprint: sha('9'),
            execution_lease_id: "execution-1".into(),
            execution_backend: H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            execution_active: true,
            artifact_factory_identity_sha256: sha('5'),
            artifact_component_set_identity_sha256: sha('6'),
            artifact_component_content_sha256: sha('7'),
            artifact_component_validation_sha256: sha('8'),
            artifact_support_identity_sha256: sha('1'),
            artifact_weight_identity_sha256: sha('2'),
            artifact_weight_header_identity_sha256: sha('3'),
            artifact_weight_policy_identity_sha256: sha('4'),
            artifact_active: true,
            loaded_matches_conditioner_device: true,
            conditioner_is_cpu: false,
            conditioner_is_cuda: true,
            conditioner_matches_execution_device: true,
            execution_is_cuda: true,
        };
        (loaded, frozen, leases)
    }

    #[test]
    fn independent_loaded_frozen_and_lease_authorities_cross_bind() {
        let (loaded, frozen, leases) = binding_claims();
        validate_binding(&loaded, &frozen, &leases).unwrap();

        let mut wrong_model = loaded.clone();
        wrong_model.model = contract::REF2VA_COMFY.into();
        wrong_model.task = Task::Ref2va;
        assert!(validate_binding(&wrong_model, &frozen, &leases)
            .unwrap_err()
            .to_string()
            .contains("model/task"));

        let mut wrong_pair = leases.clone();
        wrong_pair.artifact_support_identity_sha256 = sha('a');
        assert!(validate_binding(&loaded, &frozen, &wrong_pair)
            .unwrap_err()
            .to_string()
            .contains("different artifact lease"));

        let mut wrong_factory = leases.clone();
        wrong_factory.artifact_component_validation_sha256 = sha('b');
        assert!(validate_binding(&loaded, &frozen, &wrong_factory)
            .unwrap_err()
            .to_string()
            .contains("component/factory"));
    }

    #[test]
    fn revoked_conditioner_execution_and_artifact_leases_fail_closed() {
        let (loaded, frozen, leases) = binding_claims();
        for revoke in [
            (|claims: &mut LeaseBindingClaims| claims.conditioner_active = false)
                as fn(&mut LeaseBindingClaims),
            |claims| claims.execution_active = false,
            |claims| claims.artifact_active = false,
        ] {
            let mut revoked = leases.clone();
            revoke(&mut revoked);
            assert!(validate_binding(&loaded, &frozen, &revoked).is_err());
        }
    }

    #[test]
    fn exact_parameter_authority_matches_authenticated_source_not_retained_representation() {
        let facts = released_h3_qwen_nvfp4_runtime_memory_facts(&Device::Cpu).unwrap();
        let exact = test_authority(facts.effective_parameter_bytes, 1, 1, 0);
        validate_parameter_memory(&exact, &facts).unwrap();
        assert_ne!(
            facts.effective_parameter_bytes,
            facts.retained_parameter_bytes
        );
        for under_or_over in [
            facts.effective_parameter_bytes - 1,
            facts.effective_parameter_bytes + 1,
        ] {
            let wrong = test_authority(under_or_over, 1, 1, 0);
            assert!(validate_parameter_memory(&wrong, &facts).is_err());
        }
    }

    #[test]
    fn output_dtype_authority_accepts_only_bf16() {
        validate_output_dtype(DType::BF16).unwrap();
        for wrong in [DType::F16, DType::F32, DType::F64] {
            assert!(validate_output_dtype(wrong)
                .unwrap_err()
                .to_string()
                .contains("requires BF16"));
        }
    }

    #[test]
    fn invalid_preload_authority_never_invokes_loader_callback() {
        let called = Cell::new(false);
        let error = invoke_authorized_loader::<()>(
            || bail!("revoked before load"),
            || {
                called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("revoked before load"));
        assert!(!called.get());

        let facts = released_h3_qwen_nvfp4_runtime_memory_facts(&Device::Cpu).unwrap();
        let undercharged = test_authority(facts.effective_parameter_bytes - 1, 1, 1, 0);
        let error = invoke_authorized_loader::<()>(
            || validate_parameter_memory(&undercharged, &facts),
            || {
                called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires exactly"));
        assert!(!called.get());

        let (loaded, frozen, mut leases) = binding_claims();
        leases.conditioner_active = false;
        let error = invoke_authorized_loader::<()>(
            || validate_binding(&loaded, &frozen, &leases),
            || {
                called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("inactive or cross-routed"));
        assert!(!called.get());
    }

    struct CancelCheckpoint;

    impl H3PipelineCheckpoint for CancelCheckpoint {
        fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
            Err(InferenceCancelled.into())
        }
    }

    #[test]
    fn candle_encode_callback_preserves_typed_inference_cancellation() {
        let error = encode_with_typed_checkpoint(
            &mut CancelCheckpoint,
            0,
            || Ok(()),
            |callback| {
                callback(ConditionerCheckpoint::TokenEmbedding)?;
                Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)
            },
        )
        .unwrap_err();
        assert!(is_inference_cancelled(&error));
    }

    #[test]
    fn load_observer_preserves_typed_inference_cancellation() {
        let mut checkpoint = CancelCheckpoint;
        let mut observer = H3PrivateQwenLoadCheckpoint::new(
            &mut checkpoint,
            H3PipelinePhase::QwenLoadChunk,
            || Ok(()),
        );
        assert!(
            observer.should_cancel(&H3QwenNvfp4LoadEvent::Authenticating {
                completed_bytes: 1,
                total_bytes: 2,
            })
        );
        let error = observer
            .finish::<()>(Err(H3QwenNvfp4RuntimeError::Contract("cancelled".into())))
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
    }

    #[test]
    fn load_observer_polls_authority_at_every_checkpoint_and_stops_on_revocation() {
        let active = Cell::new(true);
        let validations = Cell::new(0);
        let mut checkpoint = RecordingCheckpoint::default();
        let mut observer = H3PrivateQwenLoadCheckpoint::new(
            &mut checkpoint,
            H3PipelinePhase::QwenLoadChunk,
            || {
                validations.set(validations.get() + 1);
                if active.get() {
                    Ok(())
                } else {
                    bail!("artifact authority revoked while loading")
                }
            },
        );
        let first = H3QwenNvfp4LoadEvent::Authenticating {
            completed_bytes: 1,
            total_bytes: 3,
        };
        assert!(!observer.should_cancel(&first));
        active.set(false);
        let second = H3QwenNvfp4LoadEvent::Authenticating {
            completed_bytes: 2,
            total_bytes: 3,
        };
        assert!(observer.should_cancel(&second));
        assert!(observer.should_cancel(&second));
        let error = observer
            .finish::<()>(Err(H3QwenNvfp4RuntimeError::Contract(
                "loader continued".into(),
            )))
            .unwrap_err();
        assert!(error.to_string().contains("revoked while loading"));
        assert_eq!(validations.get(), 2);
    }

    #[test]
    fn encode_and_load_callbacks_preserve_the_first_typed_error_when_work_continues() {
        let error = encode_with_typed_checkpoint(
            &mut CancelCheckpoint,
            0,
            || Ok(()),
            |callback| {
                let _ = callback(ConditionerCheckpoint::TokenEmbedding);
                let _ = callback(ConditionerCheckpoint::LanguageLayer {
                    completed: 1,
                    total: H3_SELECTED_LANGUAGE_LAYERS,
                });
                Err(candle_core::Error::Msg("later Candle failure".into()))
            },
        )
        .unwrap_err();
        assert!(is_inference_cancelled(&error));

        let mut checkpoint = CancelCheckpoint;
        let validation_calls = Cell::new(0);
        let mut observer = H3PrivateQwenLoadCheckpoint::new(
            &mut checkpoint,
            H3PipelinePhase::QwenLoadChunk,
            || {
                validation_calls.set(validation_calls.get() + 1);
                Ok(())
            },
        );
        let first = H3QwenNvfp4LoadEvent::Authenticating {
            completed_bytes: 1,
            total_bytes: 2,
        };
        assert!(observer.should_cancel(&first));
        assert!(observer.should_cancel(&first));
        let error = observer
            .finish::<()>(Err(H3QwenNvfp4RuntimeError::Contract(
                "later load failure".into(),
            )))
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(validation_calls.get(), 1);
    }

    #[derive(Default)]
    struct RecordingCheckpoint(Vec<H3PipelineEvent>);

    impl H3PipelineCheckpoint for RecordingCheckpoint {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.0.push(event);
            Ok(())
        }
    }

    #[test]
    fn mixed_image_video_progress_is_monotonic_and_uses_the_exact_total() {
        let mut checkpoint = RecordingCheckpoint::default();
        let states = encode_with_typed_checkpoint(
            &mut checkpoint,
            2,
            || Ok(()),
            |callback| {
                callback(ConditionerCheckpoint::TokenEmbedding)?;
                for _ in 0..2 {
                    callback(ConditionerCheckpoint::VisionPatchEmbedding)?;
                    for completed in 1..=QWEN_VISION_LAYER_COUNT {
                        callback(ConditionerCheckpoint::VisionLayer {
                            completed,
                            total: QWEN_VISION_LAYER_COUNT,
                        })?;
                    }
                }
                for completed in 1..=H3_SELECTED_LANGUAGE_LAYERS {
                    callback(ConditionerCheckpoint::LanguageLayer {
                        completed,
                        total: H3_SELECTED_LANGUAGE_LAYERS,
                    })?;
                }
                Tensor::zeros((1, 1, 5_120), DType::BF16, &Device::Cpu)
            },
        )
        .unwrap();
        assert_eq!(states.dims3().unwrap(), (1, 1, 5_120));
        let expected_total = 1 + 2 * QWEN_VISION_CHECKPOINTS_PER_PASS + H3_SELECTED_LANGUAGE_LAYERS;
        assert_eq!(checkpoint.0.last().unwrap().completed, expected_total);
        assert!(checkpoint
            .0
            .iter()
            .all(|event| event.total == expected_total));
        assert!(checkpoint
            .0
            .windows(2)
            .all(|events| events[0].completed < events[1].completed));
    }

    #[test]
    fn progress_totals_distinguish_zero_one_and_two_vision_passes() {
        for (passes, expected) in [
            (0, 1 + H3_SELECTED_LANGUAGE_LAYERS),
            (
                1,
                1 + QWEN_VISION_CHECKPOINTS_PER_PASS + H3_SELECTED_LANGUAGE_LAYERS,
            ),
            (
                2,
                1 + 2 * QWEN_VISION_CHECKPOINTS_PER_PASS + H3_SELECTED_LANGUAGE_LAYERS,
            ),
        ] {
            assert_eq!(
                ConditionerProgressMapper::new(passes).unwrap().total,
                expected
            );
        }
    }

    #[test]
    fn authority_revocation_during_encode_is_polled_at_every_checkpoint() {
        let active = Cell::new(true);
        let mut checkpoint = RecordingCheckpoint::default();
        let error = encode_with_typed_checkpoint(
            &mut checkpoint,
            0,
            || {
                if active.get() {
                    Ok(())
                } else {
                    bail!("execution authority revoked during encode")
                }
            },
            |callback| {
                callback(ConditionerCheckpoint::TokenEmbedding)?;
                active.set(false);
                let _ = callback(ConditionerCheckpoint::LanguageLayer {
                    completed: 1,
                    total: H3_SELECTED_LANGUAGE_LAYERS,
                });
                let _ = callback(ConditionerCheckpoint::LanguageLayer {
                    completed: 2,
                    total: H3_SELECTED_LANGUAGE_LAYERS,
                });
                Tensor::zeros((1, 1, 5_120), DType::BF16, &Device::Cpu)
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("revoked during encode"));
        assert_eq!(checkpoint.0.len(), 1);
    }

    #[test]
    fn preparation_and_post_encode_revocation_fail_before_downstream_use() {
        let active = Cell::new(true);
        let prepared = {
            active.set(false);
            Ok::<_, anyhow::Error>(42_u32)
        };
        let prepared = complete_preparation(prepared, |_| {
            if active.get() {
                Ok(())
            } else {
                bail!("revoked during preparation")
            }
        });
        assert!(prepared
            .unwrap_err()
            .to_string()
            .contains("revoked during preparation"));

        active.set(true);
        let encoded = {
            active.set(false);
            Ok::<_, anyhow::Error>(84_u32)
        };
        let encoded = complete_encode(encoded, || {
            if active.get() {
                Ok(())
            } else {
                bail!("revoked after encode")
            }
        });
        assert!(encoded
            .unwrap_err()
            .to_string()
            .contains("revoked after encode"));
    }

    fn prepared_input(
        sequence: usize,
        image_patches: Option<usize>,
        video_patches: Option<usize>,
    ) -> H3ConditionerInput {
        let vision = |patches: usize| H3VisionInput {
            pixel_values: Tensor::zeros((patches, 1_536), DType::F32, &Device::Cpu).unwrap(),
            grid_thw: Tensor::ones((1, 3), DType::U32, &Device::Cpu).unwrap(),
        };
        H3ConditionerInput {
            input_ids: Tensor::zeros((1, sequence), DType::U32, &Device::Cpu).unwrap(),
            position_ids: Tensor::zeros((3, 1, sequence), DType::U32, &Device::Cpu).unwrap(),
            image: image_patches.map(vision),
            video: video_patches.map(vision),
        }
    }

    #[test]
    fn prepared_activation_floor_binds_exact_rows_and_rejects_undercharge_before_encode() {
        let input = prepared_input(19, Some(8), Some(12));
        let (text_rows, vision_rows, floor) = prepared_qwen_activation_floor(&input).unwrap();
        assert_eq!((text_rows, vision_rows), (19, 20));
        assert_eq!(floor, 10_256 * 19 + 6_144 * 20 + 12 * 2);

        let facts = released_h3_qwen_nvfp4_runtime_memory_facts(&Device::Cpu).unwrap();
        let exact = test_authority(
            facts.effective_parameter_bytes,
            floor,
            text_rows,
            vision_rows,
        );
        validate_prepared_authority(&exact, &input).unwrap();

        let undercharged = test_authority(
            facts.effective_parameter_bytes,
            floor - 1,
            text_rows,
            vision_rows,
        );
        let encode_called = Cell::new(false);
        let result = validate_prepared_authority(&undercharged, &input).map(|()| {
            encode_called.set(true);
        });
        assert!(result.unwrap_err().to_string().contains("undercharged"));
        assert!(!encode_called.get());

        let wrong_rows = test_authority(
            facts.effective_parameter_bytes,
            floor,
            text_rows + 1,
            vision_rows,
        );
        assert!(validate_prepared_authority(&wrong_rows, &input)
            .unwrap_err()
            .to_string()
            .contains("differ from frozen admission"));
    }

    #[test]
    fn prepared_activation_floor_rejects_malformed_released_dtypes_and_geometry() {
        let mut malformed = prepared_input(3, None, None);
        malformed.position_ids = Tensor::zeros((3, 1, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(prepared_qwen_activation_floor(&malformed).is_err());

        let mut malformed = prepared_input(3, Some(2), None);
        malformed.image.as_mut().unwrap().pixel_values =
            Tensor::zeros((2, 1_535), DType::F32, &Device::Cpu).unwrap();
        assert!(prepared_qwen_activation_floor(&malformed).is_err());
    }

    struct ExactTokenizer;

    impl H3RawTokenizer for ExactTokenizer {
        fn encode_raw(&self, text: &str) -> std::result::Result<Vec<u32>, PresentationError> {
            let id = match text {
                "<Picture 1>: " => 10,
                "<Audio 1>: " => 20,
                "<Audio 2>: " => 21,
                "<Video 1>: " => 30,
                "<0.2 seconds>" => 31,
                "<1.0 seconds>" => 32,
                "prompt" => 40,
                other => {
                    return Err(PresentationError::Tokenizer(format!(
                        "unexpected test piece {other:?}"
                    )))
                }
            };
            Ok(vec![id])
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|vision_start|>" => Some(151_652),
                "<|vision_end|>" => Some(151_653),
                "<|image_pad|>" => Some(151_655),
                "<|video_pad|>" => Some(151_656),
                _ => None,
            }
        }
    }

    #[test]
    fn ref2va_builder_preserves_mixed_reference_order_and_exact_mrope() {
        let image = RgbImage::from_pixel(64, 32, Rgb([16, 32, 48]));
        let video_frames = (0..3)
            .map(|value| RgbImage::from_pixel(64, 32, Rgb([value, 64, 96])))
            .collect::<Vec<_>>();
        let audio = crate::reference_media::NormalizedStereoAudio {
            channels: [vec![0.0; 32], vec![0.0; 32]],
            sample_rate: 32_000,
        };
        let normalized = [
            H3NormalizedReference::Image {
                image,
                vision_tokens: 2,
            },
            H3NormalizedReference::Audio {
                audio: audio.clone(),
            },
            H3NormalizedReference::Video {
                vae_frames: Vec::new(),
                qwen_frames: video_frames,
                qwen_source_indices: vec![0, 12, 24],
                qwen_block_timestamps_seconds: vec![0.25, 1.0],
                audio: Some(audio),
            },
        ];
        let references = [
            H3ReferencePresentation {
                index: 1,
                presentation: RefPresentation {
                    kind: RefPresentationKind::Image { vision_tokens: 2 },
                    has_audio: false,
                },
            },
            H3ReferencePresentation {
                index: 2,
                presentation: RefPresentation {
                    kind: RefPresentationKind::Audio,
                    has_audio: true,
                },
            },
            H3ReferencePresentation {
                index: 3,
                presentation: RefPresentation {
                    kind: RefPresentationKind::Video {
                        blocks: vec![
                            VideoPresentationBlock {
                                timestamp_seconds: 0.25,
                                vision_tokens: 2,
                            },
                            VideoPresentationBlock {
                                timestamp_seconds: 1.0,
                                vision_tokens: 2,
                            },
                        ],
                    },
                    has_audio: true,
                },
            },
        ];
        let normalized_refs = normalized.iter().collect::<Vec<_>>();
        let (input, tags) = prepare_ref2va_conditioner_input_from_normalized(
            &ExactTokenizer,
            "prompt",
            &references,
            &normalized_refs,
            &Device::Cpu,
        )
        .unwrap();
        let ids = input.input_ids.to_vec2::<u32>().unwrap().remove(0);
        assert_eq!(
            ids,
            vec![
                10, 151_652, 151_655, 151_655, 151_653, 20, 21, 30, 31, 151_652, 151_656, 151_656,
                151_653, 32, 151_652, 151_656, 151_656, 151_653, 40,
            ]
        );
        assert_eq!(tags.len(), ids.len());
        let positions = input.position_ids.to_vec3::<u32>().unwrap();
        assert_eq!(&positions[0][0][2..4], &[2, 2]);
        assert_eq!(&positions[1][0][2..4], &[2, 2]);
        assert_eq!(&positions[2][0][2..4], &[2, 3]);
        assert_eq!(
            input
                .image
                .as_ref()
                .unwrap()
                .grid_thw
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![1, 2, 4]]
        );
        assert_eq!(
            input
                .video
                .as_ref()
                .unwrap()
                .grid_thw
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![2, 2, 4]]
        );

        let mut reordered = references.clone();
        reordered.swap(0, 1);
        assert!(prepare_ref2va_conditioner_input_from_normalized(
            &ExactTokenizer,
            "prompt",
            &reordered,
            &normalized_refs,
            &Device::Cpu,
        )
        .is_err());

        let mut retimed = references.clone();
        let RefPresentationKind::Video { blocks } = &mut retimed[2].presentation.kind else {
            unreachable!()
        };
        blocks[1].timestamp_seconds = 1.25;
        let error = prepare_ref2va_conditioner_input_from_normalized(
            &ExactTokenizer,
            "prompt",
            &retimed,
            &normalized_refs,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(error.to_string().contains("timing or geometry"));
    }

    #[derive(Clone)]
    struct DropProbe {
        name: &'static str,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.events.lock().unwrap().push(self.name);
        }
    }

    struct ProbeConditionerLease {
        probe: DropProbe,
        device: Device,
    }

    impl H3ConditionerLease for ProbeConditionerLease {
        fn lease_id(&self) -> &str {
            "conditioner"
        }
        fn device_id(&self) -> &str {
            "cpu"
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn release(&mut self) {
            self.probe
                .events
                .lock()
                .unwrap()
                .push("conditioner-release");
        }
    }

    impl H3PrivateQwenConditionerLease for ProbeConditionerLease {
        fn factory_identity_sha256(&self) -> &str {
            "factory"
        }
        fn execution_fingerprint(&self) -> &str {
            "execution"
        }
        fn is_active(&self) -> bool {
            true
        }
    }

    struct ProbeExecutionLease {
        probe: DropProbe,
        device: Device,
    }

    impl H3BackendExecutionLease for ProbeExecutionLease {
        fn lease_id(&self) -> &str {
            "execution"
        }
        fn device_id(&self) -> &str {
            "cpu"
        }
        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            }
        }
        fn execution_fingerprint(&self) -> &str {
            "execution"
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn is_active(&self) -> bool {
            true
        }
    }

    struct ProbeArtifactLease {
        probe: DropProbe,
    }

    unsafe impl H3BackendArtifactLease for ProbeArtifactLease {
        fn component_set_identity(&self) -> &str {
            "components"
        }
        fn is_active(&self) -> bool {
            true
        }
    }

    unsafe impl H3PrivateQwenArtifactLease for ProbeArtifactLease {
        fn factory_identity_sha256(&self) -> &str {
            "factory"
        }
        fn conditioner_component_content_sha256(&self) -> &str {
            "content"
        }
        fn conditioner_component_validation_sha256(&self) -> &str {
            "validation"
        }
        fn support_identity_sha256(&self) -> &str {
            "support"
        }
        fn weight_identity_sha256(&self) -> &str {
            "weight"
        }
        fn weight_header_identity_sha256(&self) -> &str {
            "header"
        }
        fn weight_policy_identity_sha256(&self) -> &str {
            "policy"
        }
    }

    #[test]
    fn model_allocations_drop_before_support_and_conditioner_lease() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let probe = |name| DropProbe {
            name,
            events: Arc::clone(&events),
        };
        drop(H3PrivateQwenResources {
            model: Some(probe("model")),
            support: probe("support-descriptors"),
            conditioner_lease: ProbeConditionerLease {
                probe: probe("conditioner-lease"),
                device: Device::Cpu,
            },
            conditioner_released: false,
        });
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "model",
                "conditioner-release",
                "support-descriptors",
                "conditioner-lease",
            ]
        );
    }

    #[test]
    fn one_shot_qwen_resources_borrow_outer_execution_and_artifact_owners() {
        for encode_fails in [false, true] {
            let events = Arc::new(Mutex::new(Vec::new()));
            let probe = |name| DropProbe {
                name,
                events: Arc::clone(&events),
            };
            let execution = ProbeExecutionLease {
                probe: probe("execution-lease"),
                device: Device::Cpu,
            };
            let artifact = ProbeArtifactLease {
                probe: probe("artifact-lease"),
            };
            let execution_borrow = &execution;
            let artifact_borrow = &artifact;
            let mut resources = H3PrivateQwenResources {
                model: Some(probe("model")),
                support: probe("support-descriptors"),
                conditioner_lease: ProbeConditionerLease {
                    probe: probe("conditioner-lease"),
                    device: Device::Cpu,
                },
                conditioner_released: false,
            };
            let result = if encode_fails {
                Err::<(), _>(anyhow!("synthetic encode failure"))
            } else {
                Ok(())
            };
            let _ = complete_encode(result, || Ok(()));
            resources.release_conditioner();
            drop(resources);

            assert!(execution_borrow.is_active());
            assert!(artifact_borrow.is_active());
            assert_eq!(execution_borrow.lease_id(), "execution");
            assert_eq!(artifact_borrow.component_set_identity(), "components");
            assert_eq!(
                *events.lock().unwrap(),
                vec![
                    "model",
                    "conditioner-release",
                    "support-descriptors",
                    "conditioner-lease",
                ]
            );

            drop(execution);
            drop(artifact);
            let events = events.lock().unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| **event == "execution-lease")
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| **event == "artifact-lease")
                    .count(),
                1
            );
        }
    }

    #[cfg(feature = "cuda")]
    struct RealConditionerLease {
        factory: String,
        execution: String,
        device_id: String,
        device: Device,
        active: Arc<AtomicBool>,
    }

    #[cfg(feature = "cuda")]
    impl H3ConditionerLease for RealConditionerLease {
        fn lease_id(&self) -> &str {
            "private-qwen-smoke"
        }
        fn device_id(&self) -> &str {
            &self.device_id
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn release(&mut self) {
            self.active.store(false, Ordering::SeqCst);
        }
    }

    #[cfg(feature = "cuda")]
    impl H3PrivateQwenConditionerLease for RealConditionerLease {
        fn factory_identity_sha256(&self) -> &str {
            &self.factory
        }
        fn execution_fingerprint(&self) -> &str {
            &self.execution
        }
        fn is_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    #[cfg(feature = "cuda")]
    struct RealExecutionLease {
        execution: String,
        device_id: String,
        device: Device,
    }

    #[cfg(feature = "cuda")]
    impl H3BackendExecutionLease for RealExecutionLease {
        fn lease_id(&self) -> &str {
            "private-execution-smoke"
        }
        fn device_id(&self) -> &str {
            &self.device_id
        }
        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            }
        }
        fn execution_fingerprint(&self) -> &str {
            &self.execution
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn is_active(&self) -> bool {
            true
        }
    }

    #[cfg(feature = "cuda")]
    struct RealArtifactLease {
        factory: String,
        components: String,
        content: String,
        validation: String,
        support: String,
        weight: String,
        header: String,
        policy: String,
        active: Arc<AtomicBool>,
    }

    #[cfg(feature = "cuda")]
    impl Drop for RealArtifactLease {
        fn drop(&mut self) {
            self.active.store(false, Ordering::SeqCst);
        }
    }

    #[cfg(feature = "cuda")]
    unsafe impl H3BackendArtifactLease for RealArtifactLease {
        fn component_set_identity(&self) -> &str {
            &self.components
        }
        fn is_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    #[cfg(feature = "cuda")]
    unsafe impl H3PrivateQwenArtifactLease for RealArtifactLease {
        fn factory_identity_sha256(&self) -> &str {
            &self.factory
        }
        fn conditioner_component_content_sha256(&self) -> &str {
            &self.content
        }
        fn conditioner_component_validation_sha256(&self) -> &str {
            &self.validation
        }
        fn support_identity_sha256(&self) -> &str {
            &self.support
        }
        fn weight_identity_sha256(&self) -> &str {
            &self.weight
        }
        fn weight_header_identity_sha256(&self) -> &str {
            &self.header
        }
        fn weight_policy_identity_sha256(&self) -> &str {
            &self.policy
        }
    }

    #[cfg(feature = "cuda")]
    struct NoopCheckpoint;

    #[cfg(feature = "cuda")]
    impl H3PipelineCheckpoint for NoopCheckpoint {
        fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
            Ok(())
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires authorized private H3 artifacts and a CUDA device"]
    fn authorized_private_qwen_adapter_constructs_and_encodes_text() {
        let models_root = std::env::var_os("MOLD_H3_MODELS_ROOT")
            .map(PathBuf::from)
            .expect("set MOLD_H3_MODELS_ROOT to the authorized private models root");
        let weights_path = std::env::var_os("MOLD_H3_QWEN_PATH")
            .map(PathBuf::from)
            .expect("set MOLD_H3_QWEN_PATH to the authorized NVFP4-AWQ checkpoint");
        let model_name =
            std::env::var("MOLD_H3_MODEL").unwrap_or_else(|_| contract::FL2VA_COMFY.to_string());
        assert_eq!(model_name, contract::FL2VA_COMFY);
        let header_identity = std::env::var("MOLD_H3_QWEN_HEADER_SHA256")
            .expect("set MOLD_H3_QWEN_HEADER_SHA256 from the authorized artifact record");
        require_sha256(&header_identity, "authorized Qwen header").unwrap();
        let device = Device::new_cuda(0).unwrap();
        let support = super::super::private_qwen_support::load_qualified_private_qwen_support(
            &models_root,
            &model_name,
        )
        .unwrap();
        let support_identity = support.support_identity_sha256().to_owned();
        let prompt = "private construction smoke";
        let (prepared, _) =
            prepare_fl2va_conditioner_input(support.tokenizer(), prompt, &[], &device).unwrap();
        let (qwen_output_text_rows, qwen_vision_rows, activation_floor) =
            prepared_qwen_activation_floor(&prepared).unwrap();
        drop(prepared);
        let memory_facts = released_h3_qwen_nvfp4_runtime_memory_facts(&device).unwrap();
        assert_eq!(memory_facts.host_resident_parameter_bytes, 19_066_444_664);
        assert_eq!(memory_facts.device_resident_parameter_bytes, 1_191_583_200);
        assert_eq!(memory_facts.retained_parameter_bytes, 20_258_027_864);
        let execution = sha('a');
        let conditioner_content = sha('1');
        let conditioner_validation = sha('5');
        let components = [
            H3FactoryComponentRole::Conditioner,
            H3FactoryComponentRole::Transformer,
            H3FactoryComponentRole::VisualVae,
            H3FactoryComponentRole::AudioVae,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, role)| {
            H3FactoryComponentAuthority::new(
                role,
                if role == H3FactoryComponentRole::Conditioner {
                    conditioner_content.clone()
                } else {
                    sha(char::from(b'2' + index as u8))
                },
                if role == H3FactoryComponentRole::Conditioner {
                    conditioner_validation.clone()
                } else {
                    sha(char::from(b'6' + index as u8))
                },
            )
            .unwrap()
        })
        .collect();
        let authority = FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: model_name,
            device_id: "cuda:0".into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: execution.clone(),
            conditioner_placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
            qwen_parameter_bytes: memory_facts.effective_parameter_bytes,
            qwen_host_resident_parameter_bytes: memory_facts.host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: memory_facts.device_resident_parameter_bytes,
            qwen_activation_workspace_bytes: activation_floor,
            qwen_maximum_tensor_staging_bytes: memory_facts.maximum_tensor_staging_bytes,
            qwen_retained_raw_header_bytes: memory_facts.retained_raw_header_bytes,
            qwen_output_text_rows,
            qwen_vision_rows,
            condition_visual_rows: 0,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "private-smoke-kernel".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            attention_runtime: None,
            prepared_attempt: None,
            execution_budget_echo: None,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: H3_QWEN_NVFP4_AWQ_POLICY_SHA256.into(),
                pruned_adaln_table_sha256: sha('d'),
                turbo_adapter: None,
            },
            components,
        })
        .unwrap();
        let factory = authority.identity_sha256().to_owned();
        let component_set = authority.component_set_identity_sha256().to_owned();
        let active = Arc::new(AtomicBool::new(true));
        let conditioner_lease = RealConditionerLease {
            factory: factory.clone(),
            execution: execution.clone(),
            device_id: "cuda:0".into(),
            device: device.clone(),
            active: Arc::clone(&active),
        };
        let execution_lease = RealExecutionLease {
            execution,
            device_id: "cuda:0".into(),
            device: device.clone(),
        };
        let artifact_active = Arc::new(AtomicBool::new(true));
        let artifact_lease = RealArtifactLease {
            factory,
            components: component_set,
            content: conditioner_content,
            validation: conditioner_validation,
            support: support_identity,
            weight: H3_QWEN_NVFP4_AWQ_SHA256.into(),
            header: header_identity,
            policy: H3_QWEN_NVFP4_AWQ_POLICY_SHA256.into(),
            active: Arc::clone(&artifact_active),
        };
        let mut checkpoint = NoopCheckpoint;
        let mut adapter = H3PrivateQwenAdapter::load_authorized(
            &weights_path,
            support,
            conditioner_lease,
            &execution_lease,
            artifact_lease,
            &authority,
            &mut checkpoint,
        )
        .unwrap();
        let conditioning = adapter.encode_fl2va(prompt, &[], &mut checkpoint).unwrap();
        let (batch, rows, width) = conditioning.states.dims3().unwrap();
        assert_eq!((batch, width), (1, 5_120));
        assert_eq!(conditioning.states.dtype(), DType::BF16);
        assert!(rows > 0);
        assert!(!active.load(Ordering::SeqCst));
        assert!(execution_lease.is_active());
        assert!(artifact_active.load(Ordering::SeqCst));
        drop(adapter);
        assert!(execution_lease.is_active());
        assert!(!artifact_active.load(Ordering::SeqCst));
    }
}
