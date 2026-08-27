//! Narrow server boundary for one private MiniMax H3 owner attempt.
//!
//! The concrete inference value is intentionally hidden behind a non-Clone
//! trait object. It may ride the final `GpuJob` owner handoff, but it cannot be
//! copied into scheduler state, a registry row, or the generic model cache.

use crate::h3_attempt::H3AttemptScope;

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
pub(crate) const H3_PRIVATE_PARTITION_REJECTED: &str = "MINIMAX_H3_PRIVATE_PARTITION_REJECTED";
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
pub(crate) const H3_PRIVATE_RUNTIME_UNAVAILABLE: &str = "MINIMAX_H3_PRIVATE_RUNTIME_UNAVAILABLE";

/// Payload-free proof that one authenticated request crossed the deliberately
/// narrow private ingress partition. This value may be cloned through queue
/// and dependency-preparation state; it contains neither the API key/auth
/// marker nor prompt, media, reference, or filesystem payloads.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateIngressGrant {
    canonical_model: String,
    task: mold_core::minimax_h3::Task,
    authenticated_identity_sha256: String,
    instance_identity_sha256: String,
    partition_identity_sha256: String,
    policy_identity_sha256: String,
    request_authority_sha256: String,
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
impl H3PrivateIngressGrant {
    #[cfg(test)]
    pub(crate) fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    #[cfg_attr(all(test, not(feature = "h3-private-uat")), allow(dead_code))]
    pub(crate) fn authority_identity_sha256(&self) -> String {
        ingress_digest(
            b"mold.minimax-h3.private-ingress-grant.v1\0",
            &[
                self.canonical_model.as_bytes(),
                h3_task_label(self.task),
                self.authenticated_identity_sha256.as_bytes(),
                self.instance_identity_sha256.as_bytes(),
                self.partition_identity_sha256.as_bytes(),
                self.policy_identity_sha256.as_bytes(),
                self.request_authority_sha256.as_bytes(),
            ],
        )
    }

    /// Stable authenticated subject used only for client-operation
    /// idempotency. Runtime policy, server instance, and the materialized
    /// execution request belong to the sealed replay authority instead: a
    /// later policy change must not make the same client operation conflict
    /// with the batch it already admitted.
    pub(crate) fn idempotency_subject_sha256(&self) -> &str {
        &self.authenticated_identity_sha256
    }

    /// Payload sealed by the queue-media AEAD and bound to owner + job ID.
    /// The subject is a one-way identity digest, while the authority digest
    /// binds the exact request, instance, task, partition, and current policy.
    pub(crate) fn durable_replay_envelope(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&DurableH3ReplayEnvelope {
            version: 1,
            authenticated_identity_sha256: self.authenticated_identity_sha256.clone(),
            authority_identity_sha256: self.authority_identity_sha256(),
        })
        .map_err(|error| format!("MiniMax H3 replay authority serialization failed: {error}"))
    }

    pub(crate) fn validate_for_request(
        &self,
        request: &mold_core::GenerateRequest,
        instance_id: &str,
    ) -> Result<(), String> {
        self.validate_bound_request(request)?;
        if self.instance_identity_sha256 != private_instance_identity_sha256(instance_id) {
            return Err("MiniMax H3 ingress server instance changed".into());
        }
        Ok(())
    }

    /// Revalidate the immutable request portion when the current server
    /// instance was already cross-bound into the prepared-input authority.
    pub(crate) fn validate_bound_request(
        &self,
        request: &mold_core::GenerateRequest,
    ) -> Result<(), String> {
        if request.model != self.canonical_model
            || mold_core::minimax_h3::task_for_model(&request.model) != Some(self.task)
        {
            return Err(
                "MiniMax H3 ingress authority task or model changed after authentication".into(),
            );
        }
        let request_authority_sha256 = request_authority_sha256(request)?;
        if request_authority_sha256 != self.request_authority_sha256 {
            return Err("MiniMax H3 request changed after authenticated ingress".into());
        }
        if self.authenticated_identity_sha256.len() != 64
            || self.instance_identity_sha256.len() != 64
            || self.partition_identity_sha256
                != private_ingress_partition_identity_sha256(self.task)
        {
            return Err("MiniMax H3 ingress route or instance identity changed".into());
        }
        if self.policy_identity_sha256 != private_ingress_policy_identity_sha256(self.task) {
            return Err("MiniMax H3 ingress policy identity changed".into());
        }
        Ok(())
    }

    /// Rebind only the seed resolution performed by immutable inference
    /// admission evidence. Every other serialized request field must remain
    /// exactly equal to the authenticated submission.
    pub(crate) fn rebind_resolved_request(
        &self,
        submitted: &mold_core::GenerateRequest,
        resolved: &mold_core::GenerateRequest,
        instance_id: &str,
    ) -> Result<Self, String> {
        self.validate_for_request(submitted, instance_id)?;
        let mut expected = submitted.clone();
        match (submitted.seed, resolved.seed) {
            (None, Some(seed)) => expected.seed = Some(seed),
            (Some(submitted), Some(resolved)) if submitted == resolved => {}
            _ => return Err("MiniMax H3 admission returned an invalid seed transition".into()),
        }
        if request_authority_sha256(&expected)? != request_authority_sha256(resolved)? {
            return Err("MiniMax H3 admission changed more than the resolved request seed".into());
        }
        let mut rebound = self.clone();
        rebound.request_authority_sha256 = request_authority_sha256(resolved)?;
        rebound.validate_for_request(resolved, instance_id)?;
        Ok(rebound)
    }

    /// Rebind authority after trusted, post-acknowledgement server preparation
    /// materializes defaults, expansion results, and server-owned paths. The
    /// original request must still match the authenticated envelope exactly;
    /// the prepared value is then reclassified through the same partition and
    /// current-runtime gates rather than accepting a caller-supplied mutation.
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn rebind_server_prepared_request(
        &self,
        submitted: &mold_core::GenerateRequest,
        prepared: &mold_core::GenerateRequest,
        instance_id: &str,
    ) -> Result<Self, crate::routes::ApiError> {
        self.validate_for_request(submitted, instance_id)
            .map_err(|error| {
                crate::routes::ApiError::with_code(
                    error,
                    H3_PRIVATE_PARTITION_REJECTED,
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                )
            })?;
        build_h3_private_ingress_grant(
            prepared,
            instance_id,
            self.authenticated_identity_sha256.clone(),
            reviewed_h3_private_runtime_available,
        )?
        .ok_or_else(|| {
            crate::routes::ApiError::with_code(
                "prepared MiniMax H3 request left its authenticated partition",
                H3_PRIVATE_PARTITION_REJECTED,
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            )
        })
    }

    #[cfg(test)]
    fn authenticated_identity_sha256(&self) -> &str {
        &self.authenticated_identity_sha256
    }

    #[cfg(test)]
    fn instance_identity_sha256(&self) -> &str {
        &self.instance_identity_sha256
    }

    #[cfg(test)]
    fn partition_identity_sha256(&self) -> &str {
        &self.partition_identity_sha256
    }

    #[cfg(test)]
    fn policy_identity_sha256(&self) -> &str {
        &self.policy_identity_sha256
    }

    #[cfg(test)]
    fn request_authority_sha256(&self) -> &str {
        &self.request_authority_sha256
    }
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DurableH3ReplayEnvelope {
    version: u16,
    authenticated_identity_sha256: String,
    authority_identity_sha256: String,
}

/// Classify the only private H3 HTTP partition before activation, artifact
/// discovery, path resolution, or scheduler mutation. Non-H3 requests return
/// `None` and retain the existing public ingress gates unchanged.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
#[cfg_attr(all(test, not(feature = "h3-private-uat")), allow(dead_code))]
pub(crate) fn classify_h3_private_ingress(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    instance_id: &str,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    classify_h3_private_ingress_with_runtime(
        request,
        authenticated,
        instance_id,
        reviewed_h3_private_runtime_available,
    )
}

/// Capture durable authenticated authority without requiring the inference
/// runtime to be healthy yet. Runtime qualification is rechecked by restore
/// before the feeder publishes the job to the scheduler.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn capture_durable_h3_private_ingress(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    instance_id: &str,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    classify_h3_private_ingress_with_runtime(request, authenticated, instance_id, |_| true)
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn classify_h3_private_ingress_with_runtime(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    instance_id: &str,
    runtime_available: impl FnOnce(mold_core::minimax_h3::Task) -> bool,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    #[cfg(not(feature = "h3"))]
    use axum::http::StatusCode;

    let Some(_contract) = mold_core::minimax_h3::capability_contract_for_model(&request.model)
    else {
        return Ok(None);
    };
    // A pinned H3 identity mold has no engine arm for is not this boundary's
    // to refuse (#1354). This classifier runs ahead of `model_activation`, and
    // it used to claim every identity `capability_contract_for_model` resolves
    // — which is every H3 row — so an `official-bf16` reference or a
    // pruned-NVFP4 tag was answered `422 …PARTITION_REJECTED` while its own
    // `/api/models` row promised `501 MINIMAX_H3_RUNTIME_UNAVAILABLE` and a
    // sentence about a missing weight-layout loader. That is #1276 reappearing
    // through the private path, so the layout-level case defers and the public
    // authority answers.
    //
    // Layout-level ONLY, and both exclusions are load-bearing. Deferring on
    // `UnsupportedTask` would hand the private Ref2VA ingress seam to public
    // activation, and deferring on `EngineNotBuilt` would take the whole
    // `h3-private-uat` runtime off its own path (mold-core's `h3` edge is off
    // there, so every compact FL2VA row reports that obstacle). This asks
    // `mold_core`'s own predicate rather than re-deriving the set from
    // `contract.layout`, because it is the exact question `model_activation`
    // asks before it answers `RuntimeUnavailable`; anything looser would defer
    // an identity that then answers a licensing refusal instead.
    //
    // It sits ahead of the credential check on purpose. A build without public
    // `h3` demands an API key before it classifies anything, and answering 401
    // here would hide the row's own reason behind a gate protecting nothing —
    // there is no runtime to protect.
    if mold_core::is_pinned_unrunnable_minimax_h3_identity(&request.model) {
        return Ok(None);
    }
    #[cfg(not(feature = "h3"))]
    let authenticated_identity_sha256 = authenticated.map_or_else(
        || {
            Err(crate::routes::ApiError::with_code(
                "API key authentication is required for MiniMax H3 private generation",
                "UNAUTHORIZED",
                StatusCode::UNAUTHORIZED,
            ))
        },
        |authenticated| {
            Ok(ingress_digest(
                b"mold.minimax-h3.private-authenticated-identity.v1\0",
                &[authenticated.durable_identity.as_bytes()],
            ))
        },
    )?;
    #[cfg(feature = "h3")]
    let authenticated_identity_sha256 = ingress_digest(
        b"mold.minimax-h3.public-ingress-identity.v1\0",
        &[instance_id.as_bytes()],
    );
    #[cfg(feature = "h3")]
    let _ = authenticated;

    build_h3_private_ingress_grant(
        request,
        instance_id,
        authenticated_identity_sha256,
        runtime_available,
    )
}

/// Reconstruct a payload-free private ingress grant for a durably admitted
/// row. The caller supplies only the opaque authenticated-subject digest that
/// was bound to the original admission row; every mutable security fact is
/// rechecked against the current process before the grant is returned.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn restore_durable_h3_private_ingress(
    request: &mold_core::GenerateRequest,
    envelope: &[u8],
    instance_id: &str,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    restore_durable_h3_private_ingress_with_runtime(
        request,
        envelope,
        instance_id,
        reviewed_h3_private_runtime_available,
    )
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn restore_durable_h3_private_ingress_with_runtime(
    request: &mold_core::GenerateRequest,
    envelope: &[u8],
    instance_id: &str,
    runtime_available: impl FnOnce(mold_core::minimax_h3::Task) -> bool,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    use axum::http::StatusCode;

    let envelope: DurableH3ReplayEnvelope = serde_json::from_slice(envelope).map_err(|_| {
        crate::routes::ApiError::with_code(
            "durable MiniMax H3 admission authority is invalid",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        )
    })?;
    let valid_digest = |digest: &str| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if envelope.version != 1
        || !valid_digest(&envelope.authenticated_identity_sha256)
        || !valid_digest(&envelope.authority_identity_sha256)
    {
        return Err(crate::routes::ApiError::with_code(
            "durable MiniMax H3 admission authority is invalid",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    let restored = build_h3_private_ingress_grant(
        request,
        instance_id,
        envelope.authenticated_identity_sha256,
        runtime_available,
    )?;
    let Some(grant) = restored else {
        return Err(crate::routes::ApiError::with_code(
            "durable MiniMax H3 admission authority no longer names an H3 request",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    };
    if grant.authority_identity_sha256() != envelope.authority_identity_sha256 {
        return Err(crate::routes::ApiError::with_code(
            "durable MiniMax H3 admission authority does not match this request",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    Ok(Some(grant))
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn build_h3_private_ingress_grant(
    request: &mold_core::GenerateRequest,
    instance_id: &str,
    authenticated_identity_sha256: String,
    runtime_available: impl FnOnce(mold_core::minimax_h3::Task) -> bool,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    use axum::http::StatusCode;

    let Some(contract) = mold_core::minimax_h3::capability_contract_for_model(&request.model)
    else {
        return Ok(None);
    };
    if mold_core::is_pinned_unrunnable_minimax_h3_identity(&request.model) {
        return Ok(None);
    }
    let output_format = request
        .output_format
        .unwrap_or(mold_core::OutputFormat::Mp4);
    let references_match_task = match contract.task {
        mold_core::minimax_h3::Task::Fl2va => request.references.is_none(),
        mold_core::minimax_h3::Task::Ref2va => request
            .references
            .as_ref()
            .is_some_and(|references| !references.is_empty()),
    };
    let exact_partition = request.model == contract.canonical_model
        && mold_core::minimax_h3::is_reviewed_compact_model(contract.canonical_model)
        && contract.layout == mold_core::minimax_h3::Layout::ComfyPrunedInt8ConvrotNvfp4Awq
        && request.batch_size == 1
        && output_format == mold_core::OutputFormat::Mp4
        && request.expand != Some(true)
        && request.upscale_model.is_none()
        && references_match_task;
    if !exact_partition {
        return Err(crate::routes::ApiError::with_code(
            "MiniMax H3 accepts only its supported compact task partition and conditioning contract",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    if !runtime_available(contract.task) {
        return Err(crate::routes::ApiError::with_code(
            "MiniMax H3 runtime is unavailable for this task or server build",
            H3_PRIVATE_RUNTIME_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }

    let partition_identity_sha256 = private_ingress_partition_identity_sha256(contract.task);
    let request_authority_sha256 = request_authority_sha256(request).map_err(|error| {
        crate::routes::ApiError::with_code(
            error,
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        )
    })?;
    Ok(Some(H3PrivateIngressGrant {
        canonical_model: contract.canonical_model.to_string(),
        task: contract.task,
        authenticated_identity_sha256,
        instance_identity_sha256: private_instance_identity_sha256(instance_id),
        partition_identity_sha256,
        policy_identity_sha256: private_ingress_policy_identity_sha256(contract.task),
        request_authority_sha256,
    }))
}

/// Fail closed until the inference facade contains a compiled runtime
/// qualification for this exact task partition. This check performs no path
/// access.
///
/// The task literal that used to sit beside this call is gone (#825): the
/// public build now carries a compiled qualification for BOTH the FL2VA
/// boundary-endpoint route and the Ref2VA ordered-reference route, and
/// `reviewed_h3_private_runtime_available_for_task` is the one authority that
/// says which. Restating a task here would let this gate and the runtime
/// disagree, which is exactly what it exists to prevent.
#[cfg(feature = "h3")]
fn reviewed_h3_private_runtime_available(task: mold_core::minimax_h3::Task) -> bool {
    mold_inference::reviewed_h3_private_runtime_available_for_task(task)
}

#[cfg(all(not(feature = "h3"), feature = "h3-private-uat"))]
fn reviewed_h3_private_runtime_available(task: mold_core::minimax_h3::Task) -> bool {
    mold_inference::reviewed_h3_private_runtime_available_for_task(task)
}

#[cfg(all(test, not(any(feature = "h3", feature = "h3-private-uat"))))]
#[allow(dead_code)]
fn reviewed_h3_private_runtime_available(_task: mold_core::minimax_h3::Task) -> bool {
    false
}

/// Build the presentation-only capability for the one supported task.
///
/// This deliberately performs no artifact hashing and grants no runtime
/// authority. Generation admission separately authenticates the authorization
/// record, qualification record, and every model component. Omission is the
/// safe response when API-key authentication or either external authority file
/// is unavailable.
///
/// Scoped like every other item here: only a build that compiles the H3
/// runtime — or the bare `cfg(test)` build that exercises its fail-closed
/// arm — has a caller for it. `h3-private-bridge` alone is the structural
/// scheduler/server seam and never advertises, so compiling this there is
/// dead code the workspace lint gate rejects.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
#[cfg_attr(all(test, not(feature = "h3-private-uat")), allow(dead_code))]
pub(crate) fn advertised_h3_private_capability(
    api_key_auth_enabled: bool,
    models_root: &std::path::Path,
    device_state: &mold_core::DeviceState,
) -> Option<mold_core::MiniMaxH3Capability> {
    #[cfg(feature = "h3")]
    {
        let _ = api_key_auth_enabled;
        if !cfg!(feature = "mp4") {
            return None;
        }
        let routes = device_state
            .devices
            .iter()
            .filter(|device| {
                matches!(
                    device.backend,
                    mold_core::GpuBackend::Cuda | mold_core::GpuBackend::Metal
                ) && device.schedulable
                    && device.ordinal.is_some()
            })
            .filter_map(|device| {
                let compute_capability = match device.backend {
                    mold_core::GpuBackend::Cuda => {
                        let (major, minor) =
                            device.compute_capability.as_deref()?.split_once('.')?;
                        Some((major.parse().ok()?, minor.parse().ok()?))
                    }
                    mold_core::GpuBackend::Metal => None,
                };
                Some(mold_inference::H3PrivatePresentationRoute {
                    device_id: &device.id,
                    device_ordinal: device.ordinal?,
                    compute_capability,
                })
            })
            .collect::<Vec<_>>();
        let authority = mold_inference::authenticate_h3_public_presentation(&routes).ok()?;
        if authority.task() != mold_core::minimax_h3::Task::Fl2va
            || authority.canonical_model() != mold_core::minimax_h3::FL2VA_COMFY
        {
            return None;
        }
        build_fl2va_capability(models_root)
    }
    #[cfg(not(feature = "h3"))]
    {
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        {
            if !api_key_auth_enabled || !cfg!(feature = "mp4") {
                return None;
            }
            let paths = H3PrivateUatPathSet::resolve(models_root.to_path_buf());
            let routes = device_state
                .devices
                .iter()
                .filter(|device| {
                    device.backend == mold_core::GpuBackend::Cuda
                        && device.schedulable
                        && device.ordinal.is_some()
                        && device.compute_capability.is_some()
                })
                .filter_map(|device| {
                    let (major, minor) = device.compute_capability.as_deref()?.split_once('.')?;
                    Some(mold_inference::H3PrivatePresentationRoute {
                        device_id: &device.id,
                        device_ordinal: device.ordinal?,
                        // This route set is filtered to CUDA devices above, so the
                        // architecture is always known; `None` identifies Metal.
                        compute_capability: Some((major.parse().ok()?, minor.parse().ok()?)),
                    })
                })
                .collect::<Vec<_>>();
            let authority = mold_inference::authenticate_h3_private_presentation(
                models_root,
                &paths.authorization_record,
                &paths.runtime_qualification_record,
                &routes,
            )
            .ok()?;
            if authority.task() != mold_core::minimax_h3::Task::Fl2va
                || authority.canonical_model() != mold_core::minimax_h3::FL2VA_COMFY
            {
                return None;
            }
            build_fl2va_capability(models_root)
        }
        #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
        {
            let _ = (api_key_auth_enabled, models_root, device_state);
            None
        }
    }
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn build_fl2va_capability(models_root: &std::path::Path) -> Option<mold_core::MiniMaxH3Capability> {
    use mold_core::manifest::{find_manifest, storage_path, ModelComponent};

    #[derive(Clone, Copy)]
    enum Group {
        Transformer,
        Qwen,
        Processor,
        VideoVae,
        AudioVae,
    }

    fn group(component: ModelComponent, filename: &str) -> Group {
        match component {
            ModelComponent::Transformer
            | ModelComponent::TransformerShard
            | ModelComponent::TaskConfig => Group::Transformer,
            ModelComponent::TextEncoder | ModelComponent::TextTokenizer => Group::Qwen,
            ModelComponent::Vae => Group::VideoVae,
            ModelComponent::AudioVae => Group::AudioVae,
            ModelComponent::ModelConfig if filename.starts_with("text_encoder/") => Group::Qwen,
            ModelComponent::ModelConfig if filename.starts_with("vae/") => Group::VideoVae,
            ModelComponent::ModelConfig if filename.starts_with("audio_vae/") => Group::AudioVae,
            ModelComponent::Processor
            | ModelComponent::VideoScheduler
            | ModelComponent::AudioScheduler
            | ModelComponent::ModelConfig => Group::Processor,
            _ => Group::Processor,
        }
    }

    struct ComponentAccumulator {
        id: &'static str,
        display_name: &'static str,
        kind: &'static str,
        role: &'static str,
        scope: &'static str,
        size_bytes: u64,
        installed: bool,
    }

    let manifest = find_manifest(mold_core::minimax_h3::FL2VA_COMFY)?;
    let mut components = [
        ComponentAccumulator {
            id: "minimax-h3-fl2va-transformer",
            display_name: "FL2VA pruned INT8 transformer",
            kind: "checkpoint",
            role: "transformer",
            scope: "fl2va",
            size_bytes: 0,
            installed: true,
        },
        ComponentAccumulator {
            id: "minimax-h3-shared-qwen",
            display_name: "Qwen3-VL NVFP4-AWQ conditioner",
            kind: "text-encoder",
            role: "qwen",
            scope: "shared",
            size_bytes: 0,
            installed: true,
        },
        ComponentAccumulator {
            id: "minimax-h3-shared-processor",
            display_name: "MiniMax H3 processor and schedules",
            kind: "tokenizer",
            role: "processor",
            scope: "shared",
            size_bytes: 0,
            installed: true,
        },
        ComponentAccumulator {
            id: "minimax-h3-shared-video-vae",
            display_name: "MiniMax H3 FP16 video VAE",
            kind: "vae",
            role: "video-vae",
            scope: "shared",
            size_bytes: 0,
            installed: true,
        },
        ComponentAccumulator {
            id: "minimax-h3-shared-audio-vae",
            display_name: "MiniMax H3 FP32 stereo AudioVAE",
            kind: "vae",
            role: "audio-vae",
            scope: "shared",
            size_bytes: 0,
            installed: true,
        },
    ];
    for file in &manifest.files {
        let index = match group(file.component, &file.hf_filename) {
            Group::Transformer => 0,
            Group::Qwen => 1,
            Group::Processor => 2,
            Group::VideoVae => 3,
            Group::AudioVae => 4,
        };
        components[index].size_bytes = components[index].size_bytes.checked_add(file.size_bytes)?;
        let path = models_root.join(storage_path(manifest, file));
        components[index].installed &= path.symlink_metadata().is_ok_and(|metadata| {
            metadata.file_type().is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() == file.size_bytes
        });
    }
    if components.iter().any(|component| component.size_bytes == 0) {
        return None;
    }

    let component_ids = components
        .iter()
        .map(|component| component.id.to_string())
        .collect();
    let generation_profile_sha256 = reviewed_h3_private_generation_profile()?.0.profile_hash;
    let base_installed = components.iter().all(|component| component.installed);
    // Reviewed Turbo variants ride the same partition additively: older
    // clients parse exactly one partition per task, so a Turbo tier must not
    // become a second `fl2va` entry. A variant advertises its adapter's own
    // install state and carries its reviewed request envelope only once the
    // base stack and the adapter are both landed.
    //
    // The filter is load-bearing, not tidiness: this builds the FL2VA
    // partition, whose `required_endpoint` is "first". A Ref2VA Turbo tier
    // listed here would be advertised with a first-frame conditioning
    // contract, and `authenticated_h3_private_model_rows` would then replace
    // that model's ordinary ordered-reference row with it.
    let turbo = mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .filter(|tier| {
            mold_core::minimax_h3::task_for_model(tier.model)
                == Some(mold_core::minimax_h3::Task::Fl2va)
        })
        .map(|tier| {
            let manifest = find_manifest(tier.model)?;
            let adapter = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::DistilledLora)?;
            let path = models_root.join(storage_path(manifest, adapter));
            let installed = path.symlink_metadata().is_ok_and(|metadata| {
                metadata.file_type().is_file()
                    && !metadata.file_type().is_symlink()
                    && metadata.len() == adapter.size_bytes
            });
            let request = (installed && base_installed)
                .then(|| {
                    let (profile, _, _) =
                        reviewed_h3_private_generation_profile_for(tier.model, tier.steps)?;
                    Some(mold_core::MiniMaxH3RequestCapability {
                        width: mold_core::minimax_h3::DEFAULT_WIDTH,
                        height: mold_core::minimax_h3::DEFAULT_HEIGHT,
                        frames: mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                        fps: mold_core::minimax_h3::FIXED_FPS,
                        steps: tier.steps,
                        batch_size: 1,
                        output_format: "mp4".into(),
                        required_endpoint: "first".into(),
                        generation_profile_sha256: profile.profile_hash,
                    })
                })
                .flatten();
            Some(mold_core::MiniMaxH3TurboVariantCapability {
                model: tier.model.into(),
                display_name: format!("MiniMax H3 FL2VA {}", tier.display_label),
                tier: tier.display_label.into(),
                adapter_size_bytes: tier.adapter_size_bytes,
                installed,
                request,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(mold_core::MiniMaxH3Capability {
        runtime_available: true,
        qualification: mold_core::MiniMaxH3QualificationCapability {
            backend: "cuda-or-metal".into(),
            metal_supported: true,
            minimum_host_ram_bytes: crate::h3_admission::H3_CURATED_HOST_RAM_RECOMMENDATION_BYTES,
            // Measured bounds from the first real FL2VA render (2026-08-19,
            // hal9000 RTX 4090 SM89, 1216 s) cut the denoise workspace caps
            // from guesses to `observed x 1.15` rounded to 64 MiB, dropping
            // max(attention, FFN) 15.30 -> 8.79 GB. With the token refiner
            // streamed as well, the predicted denoise peak falls from the
            // ~19.4 GB this comment previously recorded to ~12.1 GB.
            //
            // The tier stays 24 GiB anyway. ~12.1 GB would clear a 16 GiB
            // card, but no 16 GiB render has ever happened, and advertising a
            // tier no hardware has qualified is exactly how the original
            // 40 GiB L40S-era figure got here — in the other direction, where
            // it merely refused work rather than promising a 42 GB download
            // that admission then declines. Drop to 16 GiB when a 16 GiB card
            // has actually rendered, not before. Admission gates on the exact
            // per-attempt budget regardless; this tier is advisory.
            minimum_vram_bytes: 24 * 1024 * 1024 * 1024,
            attention_profile:
                "FlashAttention v2 BF16 on CUDA SM89; chunked dense BF16 correctness on Metal"
                    .into(),
            quantization_profile: "Comfy pruned INT8-convrot + Qwen NVFP4-AWQ".into(),
        },
        partitions: vec![mold_core::MiniMaxH3PartitionCapability {
            task: "fl2va".into(),
            model: mold_core::minimax_h3::FL2VA_COMFY.into(),
            display_name: "MiniMax H3 FL2VA".into(),
            runtime_available: true,
            tier: "Compact 24 GiB VRAM".into(),
            component_ids,
            request: Some(mold_core::MiniMaxH3RequestCapability {
                width: mold_core::minimax_h3::DEFAULT_WIDTH,
                height: mold_core::minimax_h3::DEFAULT_HEIGHT,
                frames: mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                fps: mold_core::minimax_h3::FIXED_FPS,
                steps: mold_core::minimax_h3::COMFY_DEFAULT_STEPS,
                batch_size: 1,
                output_format: "mp4".into(),
                required_endpoint: "first".into(),
                generation_profile_sha256,
            }),
            turbo,
        }],
        components: components
            .into_iter()
            .map(|component| mold_core::MiniMaxH3ComponentCapability {
                id: component.id.into(),
                display_name: component.display_name.into(),
                kind: component.kind.into(),
                role: component.role.into(),
                scope: component.scope.into(),
                size_bytes: component.size_bytes,
                state: if component.installed {
                    "installed"
                } else {
                    "missing"
                }
                .into(),
            })
            .collect(),
    })
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn reviewed_h3_private_generation_profile() -> Option<(mold_core::GenerationProfileSet, u32, u64)> {
    reviewed_h3_private_generation_profile_for(
        mold_core::minimax_h3::FL2VA_COMFY,
        mold_core::minimax_h3::COMFY_DEFAULT_STEPS,
    )
}

/// Whether THIS BUILD's H3 runtime authenticates a stored qualification
/// record rather than minting one per request.
///
/// A private-UAT build without the public `h3` feature reads a record file
/// (FL2VA) or the compiled capture-scope profile (Ref2VA), each pinning one
/// canvas, one clip length, and one step count by equality. Everything it
/// advertises must be those exact values; the public runtime mints its
/// envelope from the request and advertises the whole compact rule.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
const fn private_h3_record_runtime() -> bool {
    cfg!(feature = "h3-private-uat") && !cfg!(feature = "h3")
}

/// The reviewed single-quality-point profile for one compact FL2VA identity.
/// A Turbo tag renders the same canvas/duration envelope; the only axis its
/// reviewed tier moves is the fixed step count.
#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn reviewed_h3_private_generation_profile_for(
    model: &str,
    steps: u32,
) -> Option<(mold_core::GenerationProfileSet, u32, u64)> {
    use mold_core::generation_profile::{ControlMode, FpsControl, ResolutionDomain};

    let width = mold_core::minimax_h3::DEFAULT_WIDTH;
    let height = mold_core::minimax_h3::DEFAULT_HEIGHT;
    let frames = mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES;
    let fps = mold_core::minimax_h3::FIXED_FPS;
    let mut profile = mold_core::resolve_generation_profile(mold_core::GenerationProfileInput {
        model,
        family: mold_core::minimax_h3::FAMILY,
        sub_family: None,
        default_width: width,
        default_height: height,
        default_steps: steps,
        default_guidance: 0.0,
        default_frames: Some(frames),
        default_fps: Some(fps),
        default_negative_prompt: None,
        source_image: Some(mold_core::SourceImageCapability::Required),
        supports_sequence: false,
        supports_extend: false,
        // Same single authority as the row: H3 declares synchronized audio.
        supports_audio: mold_core::catalog::declared_audio_capability(
            mold_core::minimax_h3::FAMILY,
            model,
        ) == Some(true),
    });
    let recipe = profile.recipes.first_mut()?;
    // WHICH RUNTIME IS BEHIND THIS ROW decides what it may advertise, and the
    // two differ on EVERY axis — spatial included.
    //
    // The public `h3` runtime MINTS its qualification envelope from the
    // request (`private_server::public_runtime_envelope_for_shape`), so it
    // admits the whole compact canvas rule, the family frame grid, and the
    // base tier's step range.
    //
    // A private-UAT build without `h3` authenticates a STORED qualification
    // record instead, and that record pins one canvas, one clip length, and
    // one step count by equality — `precheck_private_h3_record_canvas` refuses
    // anything else before the artifact pass and
    // `validate_prepared_with_adapter` refuses it again afterwards. Advertising
    // a range there would offer sizes that runtime cannot accept, which is
    // exactly the advertise-one-thing-enforce-another split this profile
    // exists to prevent. So it keeps the record's own fixed shape: ONE
    // recommended canvas, and ceilings that are that canvas's own numbers.
    let private_record_runtime = private_h3_record_runtime();
    // The record's canvas is the same `width`/`height` this profile defaults
    // to, which is what `capture_runtime_envelope` and the reviewed FL2VA
    // record both carry.
    let record_canvas = [(width, height)];
    let reviewed_canvases: &[(u32, u32)] = if private_record_runtime {
        &record_canvas
    } else {
        mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES
    };
    let pixels = if private_record_runtime {
        u64::from(width) * u64::from(height)
    } else {
        mold_core::minimax_h3::reviewed_compact_max_pixels()
    };
    let max_axis = if private_record_runtime {
        width.max(height)
    } else {
        mold_core::minimax_h3::reviewed_compact_max_axis_pixels()
    };
    let (min_aspect, max_aspect) = if private_record_runtime {
        let aspect = f64::from(width) / f64::from(height);
        (aspect, aspect)
    } else {
        mold_core::minimax_h3::reviewed_compact_aspect_bounds()
    };
    if private_record_runtime {
        recipe.resolution.domain = ResolutionDomain::Buckets;
        recipe.resolution.off_bucket = Some(mold_core::generation_profile::OffBucketPolicy::Reject);
        recipe.resolution.min_width = width;
        recipe.resolution.min_height = height;
    } else {
        // A RANGE, not a bucket set: the compact runtime admits any aligned
        // canvas inside its area ceiling and the memory estimate decides what
        // fits.
        recipe.resolution.domain = ResolutionDomain::Dynamic;
        recipe.resolution.off_bucket = None;
        let min_axis = mold_core::minimax_h3::reviewed_compact_min_axis_pixels();
        recipe.resolution.min_width = min_axis;
        recipe.resolution.min_height = min_axis;
    }
    recipe.resolution.max_pixels = pixels;
    recipe.resolution.max_axis_pixels = Some(max_axis);
    recipe.resolution.min_aspect_ratio = Some(min_aspect);
    recipe.resolution.max_aspect_ratio = Some(max_aspect);
    // Keep the generated profile's canonical reduced-ratio identities (`7:4`
    // for 1344x768, `1:1` for 768x768). Create surfaces render these labels
    // inside a compact aspect chip; replacing them with prose here made one
    // server-only presentation fork wrap outside the control even though
    // every surface already shares the profile.
    // The presets keep the generated profile's own `recommended` tier, which
    // every surface hides (`studio/lib/outputShape.ts`): a mark that sits on
    // every pill says nothing, and "Reviewed" on an H3 size pill said less —
    // the qualification it named is not a property of the pixels. The
    // `Default` mark still comes from the recipe default.
    let mut reviewed_groups = recipe.resolution.aspect_groups.clone();
    for group in &mut reviewed_groups {
        group
            .presets
            .retain(|preset| reviewed_canvases.contains(&(preset.width, preset.height)));
    }
    reviewed_groups.retain(|group| !group.presets.is_empty());
    if reviewed_groups
        .iter()
        .map(|group| group.presets.len())
        .sum::<usize>()
        != reviewed_canvases.len()
    {
        return None;
    }
    recipe.resolution.aspect_groups = reviewed_groups;
    // A Turbo tier's step count is its distilled adapter's own schedule
    // length and stays fixed; the base tier takes the compact range, unless a
    // stored record is the authority (above), which pins one count.
    let fixed_steps =
        private_record_runtime || mold_core::minimax_h3::turbo_tier_for_model(model).is_some();
    recipe.steps.default = steps;
    recipe.steps.min = if fixed_steps {
        steps
    } else {
        mold_core::minimax_h3::COMPACT_MIN_STEPS
    };
    recipe.steps.max = if fixed_steps {
        steps
    } else {
        mold_core::minimax_h3::COMPACT_MAX_STEPS
    };
    recipe.steps.step = 1;
    recipe.steps.recommended = vec![steps];
    recipe.steps.mode = if fixed_steps {
        ControlMode::Fixed
    } else {
        ControlMode::Adjustable
    };
    let temporal = recipe.temporal.as_mut()?;
    temporal.frames.default = frames;
    if private_record_runtime {
        temporal.frames.min = frames;
        temporal.frames.max = frames;
        temporal.frames.step = 1;
        temporal.frames.recommended = vec![frames];
        temporal.frames.mode = ControlMode::Fixed;
        temporal.max_duration_seconds = Some(frames.div_ceil(fps));
    } else {
        temporal.frames.min = mold_core::minimax_h3::MIN_FRAMES;
        temporal.frames.max = mold_core::minimax_h3::MAX_FRAMES;
        temporal.frames.step = mold_core::minimax_h3::FRAME_STEP;
        temporal.frames.recommended = vec![
            mold_core::minimax_h3::MIN_FRAMES,
            frames,
            mold_core::minimax_h3::MAX_FRAMES,
        ];
        temporal.frames.mode = ControlMode::Adjustable;
        temporal.max_duration_seconds = Some(mold_core::minimax_h3::MAX_FRAMES.div_ceil(fps));
    }
    temporal.fps = FpsControl::Fixed { value: fps };
    recipe.defaults.width = width;
    recipe.defaults.height = height;
    recipe.defaults.steps = steps;
    recipe.defaults.frames = Some(frames);
    recipe.defaults.fps = Some(fps);
    recipe.capabilities.source_image = Some(mold_core::SourceImageCapability::Required);
    let alignment = recipe.resolution.alignment;
    mold_core::qualify_generation_profile_delivery(
        &mut profile,
        mold_core::GenerationDeliveryCapabilities::new(true, cfg!(feature = "webp")),
    );
    (profile.recipes.len() == 1).then_some((profile, alignment, pixels))
}

/// Convert the authenticated presentation graph into the sole executable model
/// row clients may submit. A row exists only when every exact component is
/// already landed and the supported partition carries the admitted envelope.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn authenticated_h3_private_model_row(
    capability: &mold_core::MiniMaxH3Capability,
) -> Option<mold_core::ModelInfoExtended> {
    if !capability.runtime_available || capability.partitions.len() != 1 {
        return None;
    }
    let partition = capability.partitions.first()?;
    let request = partition.request.as_ref()?;
    if partition.task != "fl2va"
        || partition.model != mold_core::minimax_h3::FL2VA_COMFY
        || !partition.runtime_available
        || request.width != mold_core::minimax_h3::DEFAULT_WIDTH
        || request.height != mold_core::minimax_h3::DEFAULT_HEIGHT
        || request.frames != mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES
        || request.fps != mold_core::minimax_h3::FIXED_FPS
        || request.steps != mold_core::minimax_h3::COMFY_DEFAULT_STEPS
        || request.batch_size != 1
        || request.output_format != "mp4"
        || request.required_endpoint != "first"
        || partition.component_ids.len() != capability.components.len()
    {
        return None;
    }

    let mut seen = std::collections::HashSet::new();
    let mut disk_usage_bytes = 0_u64;
    for component_id in &partition.component_ids {
        if !seen.insert(component_id.as_str()) {
            return None;
        }
        let component = capability
            .components
            .iter()
            .find(|component| component.id == *component_id)?;
        if component.state != "installed" {
            return None;
        }
        disk_usage_bytes = disk_usage_bytes.checked_add(component.size_bytes)?;
    }
    if disk_usage_bytes == 0 {
        return None;
    }

    let (generation_profile, alignment, pixels) = reviewed_h3_private_generation_profile()?;
    if request.generation_profile_sha256 != generation_profile.profile_hash {
        return None;
    }

    h3_model_row(
        &partition.model,
        &partition.display_name,
        "Compact FL2VA CUDA runtime with synchronized audio; supported first-frame quality profile.",
        request,
        disk_usage_bytes,
        (generation_profile, alignment, pixels),
    )
}

/// Every executable model row the authenticated capability advertises: the
/// base FL2VA partition plus each reviewed Turbo variant whose adapter is
/// landed and whose advertised envelope matches its reviewed tier exactly.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn authenticated_h3_private_model_rows(
    capability: &mold_core::MiniMaxH3Capability,
) -> Vec<mold_core::ModelInfoExtended> {
    let Some(base) = authenticated_h3_private_model_row(capability) else {
        return Vec::new();
    };
    let base_disk = base.disk_usage_bytes.unwrap_or(0);
    let mut rows = vec![base];
    let Some(partition) = capability.partitions.first() else {
        return rows;
    };
    for variant in &partition.turbo {
        let Some(tier) = mold_core::minimax_h3::turbo_tier_for_model(&variant.model) else {
            continue;
        };
        let Some(request) = variant.request.as_ref() else {
            continue;
        };
        if !variant.installed
            || request.width != mold_core::minimax_h3::DEFAULT_WIDTH
            || request.height != mold_core::minimax_h3::DEFAULT_HEIGHT
            || request.frames != mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES
            || request.fps != mold_core::minimax_h3::FIXED_FPS
            || request.steps != tier.steps
            || request.batch_size != 1
            || request.output_format != "mp4"
            || request.required_endpoint != "first"
        {
            continue;
        }
        let Some((profile, alignment, pixels)) =
            reviewed_h3_private_generation_profile_for(tier.model, tier.steps)
        else {
            continue;
        };
        if request.generation_profile_sha256 != profile.profile_hash {
            continue;
        }
        let Some(disk_usage_bytes) = base_disk.checked_add(variant.adapter_size_bytes) else {
            continue;
        };
        if let Some(row) = h3_model_row(
            &variant.model,
            &variant.display_name,
            "Compact FL2VA CUDA runtime with synchronized audio; reviewed Turbo distillation quality profile.",
            request,
            disk_usage_bytes,
            (profile, alignment, pixels),
        ) {
            rows.push(row);
        }
    }
    rows
}

/// One executable compact-FL2VA `/api/models` row. Shared by the base
/// partition and its reviewed Turbo variants so the two can never advertise
/// diverging envelope shapes.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn h3_model_row(
    model: &str,
    display_name: &str,
    description: &str,
    request: &mold_core::MiniMaxH3RequestCapability,
    disk_usage_bytes: u64,
    (generation_profile, alignment, pixels): (mold_core::GenerationProfileSet, u32, u64),
) -> Option<mold_core::ModelInfoExtended> {
    Some(mold_core::ModelInfoExtended {
        // The bridge only ever re-publishes a reviewed, executable partition.
        runtime_available: Some(true),
        runtime_unavailable_reason: None,
        info: mold_core::ModelInfo {
            name: model.to_string(),
            family: mold_core::minimax_h3::FAMILY.into(),
            size_gb: disk_usage_bytes as f32 / 1_000_000_000.0,
            is_loaded: false,
            last_used: None,
            // The row remains the sole executable recipe, while its acquisition
            // provenance is the same pinned Hugging Face repo as the manifest.
            hf_repo: mold_core::minimax_h3::COMFY_REPO.into(),
        },
        defaults: mold_core::ModelDefaults {
            default_steps: request.steps,
            default_guidance: 0.0,
            default_width: request.width,
            default_height: request.height,
            description: description.into(),
            default_frames: Some(request.frames),
            default_fps: Some(request.fps),
            // The compact stack takes the FAMILY frame grid; `request.frames`
            // is the default clip length, not a ceiling. A private-UAT build
            // without `h3` authenticates a stored record that pins one clip
            // length, so its row must say so — same rule as the profile above.
            min_frames: Some(if private_h3_record_runtime() {
                request.frames
            } else {
                mold_core::minimax_h3::MIN_FRAMES
            }),
            max_frames: Some(if private_h3_record_runtime() {
                request.frames
            } else {
                mold_core::minimax_h3::MAX_FRAMES
            }),
            max_runtime_seconds: Some(if private_h3_record_runtime() {
                request.frames.div_ceil(request.fps)
            } else {
                mold_core::minimax_h3::MAX_DURATION_SECONDS
            }),
            max_frames_absolute: Some(if private_h3_record_runtime() {
                request.frames
            } else {
                mold_core::minimax_h3::MAX_FRAMES
            }),
            frame_step: Some(if private_h3_record_runtime() {
                1
            } else {
                mold_core::minimax_h3::FRAME_STEP
            }),
            frame_offset: Some(if private_h3_record_runtime() {
                0
            } else {
                mold_core::minimax_h3::FRAME_OFFSET
            }),
            // `pixels` already follows the runtime (it comes from the
            // profile builder), and the two spatial fields the row states on
            // its own follow it the same way: a stored-record build admits
            // exactly `request`'s canvas, so the ceiling it advertises is
            // that canvas's own, not the compact rule's.
            max_pixels: Some(pixels),
            max_axis_pixels: Some(if private_h3_record_runtime() {
                request.width.max(request.height)
            } else {
                mold_core::minimax_h3::reviewed_compact_max_axis_pixels()
            }),
            default_negative_prompt: None,
            // The recommended canvases, default first — the same slice the
            // profile above advertises as presets. A row that named only the
            // default let a client that reads `recommended_dimensions` rather
            // than the profile offer one size while the profile offered
            // several; a stored-record build is the case where one really is
            // all there is.
            recommended_dimensions: if private_h3_record_runtime() {
                vec![mold_core::RecommendedDimensions {
                    width: request.width,
                    height: request.height,
                }]
            } else {
                mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES
                    .iter()
                    .map(|&(width, height)| mold_core::RecommendedDimensions { width, height })
                    .collect()
            },
            dimension_alignment: Some(alignment),
        },
        downloaded: true,
        disk_usage_bytes: Some(disk_usage_bytes),
        remaining_download_bytes: Some(0),
        display_name: Some(display_name.to_string()),
        kind: Some("checkpoint".into()),
        modality: Some("video".into()),
        nsfw: None,
        // The family's own declaration, never a literal: the manifest row for
        // this same identity derives from it too, so the executable row and
        // the cold row can never advertise diverging audio contracts (#841).
        supports_audio: mold_core::catalog::declared_audio_capability(
            mold_core::minimax_h3::FAMILY,
            model,
        ),
        // These readiness fields describe the split LTX-2.5 runtime contract;
        // H3 remains executable through its own authenticated bridge.
        supports_duration_prediction: None,
        runtime_ready: None,
        runtime_readiness_error: None,
        supports_identity: Some(false),
        supports_extend: Some(false),
        extend_default_overlap_frames: None,
        supports_sequence: Some(false),
        guidance_capabilities: Some(mold_core::GuidanceCapabilities {
            adjustable: false,
            supports_negative_prompt: false,
            fixed_scale: Some(0.0),
        }),
        source_image: Some(mold_core::SourceImageCapability::Required),
        generation_profile: Some(generation_profile),
    })
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn ingress_digest(domain: &[u8], values: &[&[u8]]) -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update(domain);
    for value in values {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn h3_task_label(task: mold_core::minimax_h3::Task) -> &'static [u8] {
    match task {
        mold_core::minimax_h3::Task::Fl2va => b"fl2va",
        mold_core::minimax_h3::Task::Ref2va => b"ref2va",
    }
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn private_instance_identity_sha256(instance_id: &str) -> String {
    ingress_digest(
        b"mold.minimax-h3.private-server-instance.v1\0",
        &[instance_id.as_bytes()],
    )
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn private_ingress_policy_identity_sha256(task: mold_core::minimax_h3::Task) -> String {
    let (model, references) = match task {
        mold_core::minimax_h3::Task::Fl2va => (
            mold_core::minimax_h3::FL2VA_COMFY,
            b"no-references".as_slice(),
        ),
        mold_core::minimax_h3::Task::Ref2va => (
            mold_core::minimax_h3::REF2VA_COMFY,
            b"ordered-heterogeneous-references".as_slice(),
        ),
    };
    ingress_digest(
        b"mold.minimax-h3.private-ingress-policy.v1\0",
        &[
            model.as_bytes(),
            h3_task_label(task),
            b"comfy-pruned-int8",
            b"batch-1",
            b"mp4",
            b"no-expand",
            b"no-upscale",
            references,
            b"api-key-authenticated",
        ],
    )
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn private_ingress_partition_identity_sha256(task: mold_core::minimax_h3::Task) -> String {
    let (model, references) = match task {
        mold_core::minimax_h3::Task::Fl2va => (
            mold_core::minimax_h3::FL2VA_COMFY,
            b"no-references".as_slice(),
        ),
        mold_core::minimax_h3::Task::Ref2va => (
            mold_core::minimax_h3::REF2VA_COMFY,
            b"ordered-heterogeneous-references".as_slice(),
        ),
    };
    ingress_digest(
        b"mold.minimax-h3.private-ingress-partition.v1\0",
        &[
            model.as_bytes(),
            h3_task_label(task),
            b"comfy-pruned-int8",
            b"batch-1",
            b"mp4",
            b"no-expand",
            b"no-upscale",
            references,
        ],
    )
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn request_authority_sha256(request: &mold_core::GenerateRequest) -> Result<String, String> {
    let mut canonical =
        crate::queue_media_runtime::ZeroizingGenerateRequest::from_owned(request.clone());
    if canonical.audio_file_path.is_some() {
        canonical.audio_file_path = Some("<durable-media:audio-file-path>".into());
    }
    if canonical.source_video_path.is_some() {
        canonical.source_video_path = Some("<durable-media:source-video-path>".into());
    }
    if canonical.extend_video_path.is_some() {
        canonical.extend_video_path = Some("<durable-media:extend-video-path>".into());
    }
    let serialized = zeroize::Zeroizing::new(serde_json::to_vec(&*canonical).map_err(|error| {
        format!("MiniMax H3 request authority could not be serialized: {error}")
    })?);
    Ok(ingress_digest(
        b"mold.minimax-h3.private-request-authority.v1\0",
        &[&serialized],
    ))
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn pin_private_preview_seed(
    request: &mut mold_core::GenerateRequest,
) -> Result<(), crate::routes::ApiError> {
    use sha2::{Digest, Sha256};

    if request.seed.is_some() {
        return Ok(());
    }
    let serialized = serde_json::to_vec(request).map_err(|error| {
        crate::routes::ApiError::validation(format!(
            "MiniMax H3 placement-preview seed authority could not be serialized: {error}"
        ))
    })?;
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-placement-preview-seed.v1\0");
    digest.update((serialized.len() as u64).to_le_bytes());
    digest.update(serialized);
    let bytes: [u8; 32] = digest.finalize().into();
    request.seed = Some(u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("SHA-256 prefix always contains eight bytes"),
    ));
    Ok(())
}

/// Substitute redacted FL2VA endpoints in a placement probe.
///
/// Placement previews redact media, so a present endpoint arrives as zero
/// bytes (`source_image: ""` on the wire). Admission decodes endpoints
/// exactly as a real submission would, which refused every H3 preview with
/// "endpoint is empty" even though the client validated the real image
/// locally. The probe substitutes a solid placeholder at the request's own
/// target geometry so the preview prices the same shape; the real submission
/// still decodes the real bytes at admission. Only present-but-empty
/// endpoints are touched — real bytes and absent fields pass through.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn substitute_redacted_preview_endpoints(request: &mut mold_core::GenerateRequest) {
    if mold_core::minimax_h3::task_for_model(&request.model)
        != Some(mold_core::minimax_h3::Task::Fl2va)
    {
        return;
    }
    // Never allocate for a shape admission will refuse anyway. The contract
    // bounds alignment, pixel count, and aspect ratio, so an unauthenticated
    // preview cannot drive the probe into building an oversized placeholder;
    // a contract-invalid request keeps its redacted endpoints and admission
    // reports the same contract error it always did.
    if mold_core::minimax_h3::validate_request_contract(request, mold_core::minimax_h3::Task::Fl2va)
        .is_err()
    {
        return;
    }
    let (width, height) = (request.width, request.height);
    let mut placeholder: Option<Vec<u8>> = None;
    let mut fill = |slot: &mut Vec<u8>| {
        if slot.is_empty() {
            *slot = placeholder
                .get_or_insert_with(|| mold_inference::h3_placeholder_endpoint_png(width, height))
                .clone();
        }
    };
    if let Some(image) = request.source_image.as_mut() {
        fill(image);
    }
    for keyframe in request.keyframes.iter_mut().flatten() {
        fill(&mut keyframe.image);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PreparedAttemptFacts {
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) execution_identity_sha256: String,
    pub(crate) prepared_attempt_identity_sha256: String,
    pub(crate) target_budget_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) admission_evidence_identity_sha256: String,
    pub(crate) artifact_qualification_identity_sha256: String,
    pub(crate) runtime_qualification_identity_sha256: String,
    pub(crate) work_identity_sha256: String,
    pub(crate) cancellation_scope_identity_sha256: String,
    pub(crate) memory_ledger_sequence: u64,
    pub(crate) consumption_identity_sha256: String,
    pub(crate) predicted_device_peak_bytes: u64,
    pub(crate) predicted_host_increment_bytes: u64,
    pub(crate) media: H3PreparedMediaContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3TerminalIdentityEcho {
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) execution_identity_sha256: String,
    pub(crate) prepared_attempt_identity_sha256: String,
    pub(crate) target_budget_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) admission_evidence_identity_sha256: String,
    pub(crate) artifact_qualification_identity_sha256: String,
    pub(crate) runtime_qualification_identity_sha256: String,
    pub(crate) consumption_identity_sha256: String,
    pub(crate) media: H3PreparedMediaContract,
    pub(crate) duration_ms: u64,
    pub(crate) audio_sample_rate: u32,
    pub(crate) audio_channels: u16,
    pub(crate) synchronized_audio_video: bool,
    pub(crate) pipeline_provenance_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PreparedMediaContract {
    pub(crate) canonical_model: String,
    pub(crate) task: mold_core::minimax_h3::Task,
    pub(crate) mode: mold_core::minimax_h3::Mode,
    pub(crate) seed: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) fps: u32,
    /// Fingerprint of the target-duration prepared reference metadata in
    /// exact request order. Ref2VA requires this; FL2VA must leave it absent.
    pub(crate) reference_fingerprint_sha256: Option<String>,
    /// Fingerprint of the probed descriptor set that owns the private staged
    /// files. This separately detects byte/probe replacement even when the
    /// target-duration prepared shapes are unchanged.
    pub(crate) resolved_reference_fingerprint_sha256: Option<String>,
    pub(crate) reference_count: u32,
}

impl H3PreparedMediaContract {
    pub(crate) fn from_request(
        request: &mold_core::GenerateRequest,
        resolved_reference_fingerprint_sha256: Option<&str>,
    ) -> Result<Self, String> {
        let task = mold_core::minimax_h3::task_for_model(&request.model)
            .ok_or_else(|| "MiniMax H3 prepared media lost its task partition".to_string())?;
        let mode = match task {
            mold_core::minimax_h3::Task::Fl2va => {
                mold_core::minimax_h3::validate_request_contract(request, task)
            }
            mold_core::minimax_h3::Task::Ref2va => {
                mold_core::minimax_h3::validate_resolved_request_contract(request, task)
            }
        }
        .map_err(|error| format!("{}: {}", error.code, error.message))?;
        let seed = request
            .seed
            .ok_or_else(|| "MiniMax H3 prepared media has no frozen seed".to_string())?;
        let frames = request
            .frames
            .unwrap_or(mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES);
        let (reference_fingerprint_sha256, resolved_fingerprint, reference_count) = match task {
            mold_core::minimax_h3::Task::Fl2va => {
                if resolved_reference_fingerprint_sha256.is_some() {
                    return Err(
                        "MiniMax H3 FL2VA cannot inherit resolved Ref2VA authority".to_string()
                    );
                }
                (None, None, 0)
            }
            mold_core::minimax_h3::Task::Ref2va => {
                let references = request.references.as_deref().ok_or_else(|| {
                    "MiniMax H3 Ref2VA prepared media lost ordered references".to_string()
                })?;
                let reference_count = u32::try_from(references.len())
                    .map_err(|_| "MiniMax H3 Ref2VA reference count exceeded u32".to_string())?;
                let resolved_fingerprint =
                    resolved_reference_fingerprint_sha256.ok_or_else(|| {
                        "MiniMax H3 Ref2VA prepared media lost resolved reference authority"
                            .to_string()
                    })?;
                let resolved_metadata = references
                    .iter()
                    .enumerate()
                    .map(|(index, reference)| reference.redacted_metadata_lossless(index))
                    .collect::<Vec<_>>();
                let expected_resolved =
                    mold_core::generation_reference_fingerprint(&resolved_metadata);
                if expected_resolved != resolved_fingerprint {
                    return Err(
                        "MiniMax H3 Ref2VA resolved reference order or artifact identity changed"
                            .to_string(),
                    );
                }
                let prepared_metadata = references
                    .iter()
                    .enumerate()
                    .map(|(index, reference)| {
                        reference.redacted_metadata_lossless_for_target(index, frames)
                    })
                    .collect::<Vec<_>>();
                (
                    Some(mold_core::generation_reference_fingerprint(
                        &prepared_metadata,
                    )),
                    Some(resolved_fingerprint.to_string()),
                    reference_count,
                )
            }
        };
        Ok(Self {
            canonical_model: request.model.clone(),
            task,
            mode,
            seed,
            width: request.width,
            height: request.height,
            frames,
            fps: mold_core::minimax_h3::FIXED_FPS,
            reference_fingerprint_sha256,
            resolved_reference_fingerprint_sha256: resolved_fingerprint,
            reference_count,
        })
    }

    pub(crate) fn validate_for_request(
        &self,
        request: &mold_core::GenerateRequest,
        resolved_reference_fingerprint_sha256: Option<&str>,
    ) -> Result<(), String> {
        let expected = Self::from_request(request, resolved_reference_fingerprint_sha256)?;
        if &expected != self {
            return Err(
                "MiniMax H3 prepared media changed from its ordered request authority".to_string(),
            );
        }
        Ok(())
    }
}

pub(crate) struct H3ClaimedRunOutput {
    pub(crate) response: mold_core::GenerateResponse,
    pub(crate) identity_echo: H3TerminalIdentityEcho,
}

/// One owner-supplied allocation notification. The closure is owned so a
/// private inference adapter can move it into its own one-shot commit guard
/// without borrowing scheduler state across the runtime call.
#[cfg_attr(
    all(
        feature = "h3-private-bridge",
        not(feature = "h3-private-uat"),
        not(test)
    ),
    allow(dead_code)
)]
pub(crate) struct H3AllocationCommit {
    callback: Option<Box<dyn FnOnce() -> anyhow::Result<()> + Send>>,
}

impl H3AllocationCommit {
    pub(crate) fn new(callback: impl FnOnce() -> anyhow::Result<()> + Send + 'static) -> Self {
        Self {
            callback: Some(Box::new(callback)),
        }
    }

    #[cfg_attr(
        all(
            feature = "h3-private-bridge",
            not(feature = "h3-private-uat"),
            not(test)
        ),
        allow(dead_code)
    )]
    pub(crate) fn commit_once(&mut self) -> anyhow::Result<()> {
        let callback = self
            .callback
            .take()
            .ok_or_else(|| anyhow::anyhow!("MiniMax H3 allocation commit was already consumed"))?;
        callback()
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn into_inference(self) -> mold_inference::H3PrivateAllocationCommit {
        let mut commit = self;
        mold_inference::H3PrivateAllocationCommit::new(move || commit.commit_once())
    }
}

/// One prepared, allocation-free attempt. Implementations consume their
/// internal authority on the first call and must reject a replay.
pub(crate) trait H3PreparedAttempt: Send {
    fn facts(&self) -> H3PreparedAttemptFacts;

    fn run_once(
        &mut self,
        scope: H3AttemptScope<'_>,
        progress: &mut mold_inference::progress::ProgressReporter,
        allocation_commit: H3AllocationCommit,
    ) -> anyhow::Result<H3ClaimedRunOutput>;
}

pub(crate) type BoxedH3PreparedAttempt = Box<dyn H3PreparedAttempt>;

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[derive(Clone, Debug)]
pub(crate) struct H3PrivateUatPathSet {
    pub(crate) models_root: std::path::PathBuf,
    pub(crate) staging_root: std::path::PathBuf,
    pub(crate) authorization_record: std::path::PathBuf,
    pub(crate) runtime_qualification_record: std::path::PathBuf,
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl H3PrivateUatPathSet {
    pub(crate) fn resolve(models_root: std::path::PathBuf) -> Self {
        #[cfg(feature = "h3")]
        let default_runtime_root = mold_core::Config::mold_dir()
            .unwrap_or_else(|| models_root.clone())
            .join("h3-runtime");
        #[cfg(not(feature = "h3"))]
        let default_runtime_root = mold_core::Config::mold_dir()
            .unwrap_or_else(|| models_root.clone())
            .join("h3-private-uat");
        let uat_root = std::env::var_os("MOLD_H3_PRIVATE_UAT_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or(default_runtime_root);
        Self {
            models_root,
            staging_root: private_uat_path("MOLD_H3_STAGING_ROOT", uat_root.join("staging")),
            authorization_record: private_uat_path(
                "MOLD_H3_AUTHORIZATION_RECORD",
                uat_root.join("authorization-record.json"),
            ),
            runtime_qualification_record: private_uat_path(
                "MOLD_H3_RUNTIME_QUALIFICATION_RECORD",
                uat_root.join("runtime-qualification.json"),
            ),
        }
    }

    /// Create the server-owned staging root when it is missing.
    ///
    /// Admission strictly validates this directory (real and canonical), but
    /// nothing ever created it, so a
    /// fresh public-H3 deployment could never admit a request — the VAE load
    /// plan failed with "cannot inspect private H3 staging root". Creation is
    /// owner-only and non-recursive in spirit: an existing directory is left
    /// exactly as the operator set it, and failures stay silent here because
    /// admission's own validation reports the authoritative error.
    ///
    /// The only lib caller is `feature = "h3"`-gated on purpose — the
    /// private-UAT campaign layout stays fail-closed, its hand-built scope
    /// must already exist — so the campaign build (`h3-private-uat` without
    /// `h3`) compiles this method dead; tests still exercise it there.
    #[cfg_attr(not(feature = "h3"), allow(dead_code))]
    pub(crate) fn ensure_staging_root(&self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            if !self.staging_root.exists() {
                let _ = std::fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(&self.staging_root);
            }
        }
    }

    pub(crate) fn inference_paths(&self) -> mold_inference::H3PrivateFl2VaUatPaths<'_> {
        mold_inference::H3PrivateFl2VaUatPaths {
            models_root: &self.models_root,
            staging_root: &self.staging_root,
            authorization_record: &self.authorization_record,
            runtime_qualification_record: &self.runtime_qualification_record,
        }
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) struct InferenceH3PreparedAttempt {
    inner: mold_inference::H3PrivateFl2VaPreparedAttempt,
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl InferenceH3PreparedAttempt {
    pub(crate) fn boxed(
        inner: mold_inference::H3PrivateFl2VaPreparedAttempt,
    ) -> BoxedH3PreparedAttempt {
        Box::new(Self { inner })
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl H3PreparedAttempt for InferenceH3PreparedAttempt {
    fn facts(&self) -> H3PreparedAttemptFacts {
        inference_facts(self.inner.facts())
    }

    fn run_once(
        &mut self,
        scope: H3AttemptScope<'_>,
        progress: &mut mold_inference::progress::ProgressReporter,
        allocation_commit: H3AllocationCommit,
    ) -> anyhow::Result<H3ClaimedRunOutput> {
        let context = scope.private_run_context()?;
        let output = self
            .inner
            .run_once(context, progress, allocation_commit.into_inference())?;
        Ok(H3ClaimedRunOutput {
            response: output.response,
            identity_echo: H3TerminalIdentityEcho {
                device_id: output.identity_echo.device_id,
                device_ordinal: output.identity_echo.device_ordinal,
                execution_identity_sha256: output.identity_echo.execution_fingerprint,
                prepared_attempt_identity_sha256: output
                    .identity_echo
                    .prepared_attempt_identity_sha256,
                target_budget_identity_sha256: output.identity_echo.target_budget_identity_sha256,
                component_set_identity_sha256: output.identity_echo.component_set_identity_sha256,
                admission_evidence_identity_sha256: output
                    .identity_echo
                    .admission_evidence_identity_sha256,
                artifact_qualification_identity_sha256: output
                    .identity_echo
                    .artifact_qualification_identity_sha256,
                runtime_qualification_identity_sha256: output
                    .identity_echo
                    .runtime_qualification_identity_sha256,
                consumption_identity_sha256: output.identity_echo.consumption_identity_sha256,
                media: inference_media(output.identity_echo.media),
                duration_ms: output.identity_echo.duration_ms,
                audio_sample_rate: output.identity_echo.audio_sample_rate,
                audio_channels: output.identity_echo.audio_channels,
                synchronized_audio_video: output.identity_echo.synchronized_audio_video,
                pipeline_provenance_sha256: output.identity_echo.pipeline_provenance_sha256,
            },
        })
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn inference_facts(facts: &mold_inference::H3PrivateFl2VaAttemptFacts) -> H3PreparedAttemptFacts {
    H3PreparedAttemptFacts {
        device_id: facts.device_id.clone(),
        device_ordinal: facts.device_ordinal,
        execution_identity_sha256: facts.execution_fingerprint.clone(),
        prepared_attempt_identity_sha256: facts.prepared_attempt_identity_sha256.clone(),
        target_budget_identity_sha256: facts.target_budget_identity_sha256.clone(),
        component_set_identity_sha256: facts.component_set_identity_sha256.clone(),
        admission_evidence_identity_sha256: facts.admission_evidence_identity_sha256().to_string(),
        artifact_qualification_identity_sha256: facts
            .artifact_qualification_identity_sha256()
            .to_string(),
        runtime_qualification_identity_sha256: facts
            .runtime_qualification_identity_sha256()
            .to_string(),
        work_identity_sha256: facts.work_identity_sha256().to_string(),
        cancellation_scope_identity_sha256: facts.cancellation_scope_identity_sha256().to_string(),
        memory_ledger_sequence: facts.memory_ledger_sequence(),
        consumption_identity_sha256: facts.consumption_identity_sha256().to_string(),
        predicted_device_peak_bytes: facts.predicted_device_peak_bytes,
        predicted_host_increment_bytes: facts.predicted_host_increment_bytes,
        media: inference_media(facts.media.clone()),
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn inference_media(media: mold_inference::H3PrivateFl2VaMediaContract) -> H3PreparedMediaContract {
    H3PreparedMediaContract {
        canonical_model: media.canonical_model,
        task: media.task,
        mode: media.mode,
        seed: media.seed,
        width: media.width,
        height: media.height,
        frames: media.frames,
        fps: media.fps,
        // Carried through from the prepared attempt, never restated here:
        // the whole point of the frozen-owner comparison is that the two
        // sides derived the same reference authority independently.
        reference_fingerprint_sha256: media.reference_fingerprint_sha256,
        resolved_reference_fingerprint_sha256: media.resolved_reference_fingerprint_sha256,
        reference_count: media.reference_count,
    }
}

/// Atomically prepare and attach one private attempt on the accepting GPU
/// owner. The scheduler only ever carried an empty slot; every opened file,
/// prepared tensor, and final attempt identity originates in the hidden
/// inference facade after the second plan fence.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn prepare_for_owner(
    worker: &crate::gpu_pool::GpuWorker,
    fence: &crate::scheduler::LeaseFence,
    job: &mut crate::gpu_pool::GpuJob,
    cancellation: mold_inference::InferenceCancellationToken,
    available_host_headroom_bytes: u64,
) -> Result<(), String> {
    if job.h3_prepared_attempt.is_some() {
        return Err("MiniMax H3 owner received a prepared attempt before final dispatch".into());
    }
    // Keyed on the same per-task availability predicate ingress advertises:
    // FL2VA on every reviewed build, Ref2VA only in the developer campaign
    // build, everything else refused.
    match mold_core::minimax_h3::task_for_model(&job.request.model) {
        Some(task) if reviewed_h3_private_runtime_available(task) => {}
        _ => {
            return Err(
                "MiniMax H3 owner preparation accepts only a reviewed task partition".into(),
            )
        }
    }
    let plan = job
        .execution_plan
        .as_ref()
        .ok_or_else(|| "MiniMax H3 preparation lacked an execution plan".to_string())?;
    let frozen_factory = plan
        .engine_config
        .h3_factory_authority
        .as_ref()
        .ok_or_else(|| "MiniMax H3 preparation lacked factory authority".to_string())?;
    let prepared_inputs = job.prepared_execution_inputs.as_ref().ok_or_else(|| {
        "MiniMax H3 preparation lost its allocation-free admission evidence".to_string()
    })?;
    let ingress_grant = prepared_inputs
        .h3_private_ingress_grant
        .as_ref()
        .ok_or_else(|| "MiniMax H3 preparation lost its ingress grant".to_string())?;
    ingress_grant.validate_bound_request(&job.request)?;
    let expected_media = H3PreparedMediaContract::from_request(
        &job.request,
        job.resolved_references
            .as_ref()
            .map(crate::reference_uploads::ResolvedReferenceSet::fingerprint),
    )?;
    let compute_capability = worker.gpu.compute_capability;
    let device_id = crate::scheduler::worker_device_id(worker);
    let admission_evidence = prepared_inputs
        .h3_private_admission_by_device
        .get(&device_id)
        .ok_or_else(|| {
            format!("MiniMax H3 preparation has no admission evidence for '{device_id}'")
        })?;
    validate_private_h3_live_owner_route(
        admission_evidence.device_id(),
        admission_evidence.device_ordinal(),
        admission_evidence.compute_capability(),
        &device_id,
        worker.gpu.ordinal,
        compute_capability,
    )?;
    if fence.work_id != job.id
        || fence.device_id != device_id
        || plan.device_ordinal != worker.gpu.ordinal
        || frozen_factory.device_id() != device_id
        || frozen_factory.device_ordinal() != worker.gpu.ordinal
    {
        return Err("MiniMax H3 preparation no longer owns the accepted fence".into());
    }

    let prepared_route = prepared_inputs
        .by_device
        .get(&device_id)
        .ok_or_else(|| format!("MiniMax H3 preparation has no prepared route for '{device_id}'"))?;
    if prepared_route.engine_paths != plan.engine_paths
        || prepared_route.engine_config != plan.engine_config
        || frozen_factory != admission_evidence.base_factory_authority()
        || plan.model_fingerprint != admission_evidence.component_set_identity_sha256()
        || plan.execution_fingerprint != admission_evidence.execution_fingerprint()
        || plan.predicted_vram_peak_bytes != admission_evidence.predicted_device_peak_bytes()
        || plan.admitted_available_vram_bytes
            != admission_evidence.admitted_available_device_bytes()
        || plan.predicted_host_increment_bytes
            != admission_evidence.predicted_host_increment_bytes()
    {
        return Err("MiniMax H3 owner plan differs from immutable admission evidence".into());
    }
    let (available_device_bytes, available_host_headroom_bytes) =
        crate::gpu_worker::prepare_private_h3_allocation_boundary(
            worker,
            &job.model,
            plan.predicted_vram_peak_bytes,
            plan.predicted_host_increment_bytes,
            available_host_headroom_bytes,
        )
        .map_err(|error| error.error)?;
    admission_evidence
        .validate_for(
            &job.request,
            &device_id,
            worker.gpu.ordinal,
            compute_capability,
            available_device_bytes,
            available_host_headroom_bytes,
        )
        .map_err(|error| {
            format!("MiniMax H3 live owner evidence no longer validates: {error:#}")
        })?;

    let work_identity_sha256 = crate::h3_attempt::private_work_identity_sha256(&fence.work_id);
    let cancellation_scope_identity_sha256 =
        crate::h3_attempt::private_cancellation_scope_identity_sha256(
            &work_identity_sha256,
            &fence.device_id,
            fence.owner_epoch,
            fence.state_version,
            fence.plan_version,
            fence.worker_generation,
            fence.memory_sample_generation,
            fence.memory_ledger_sequence,
        );

    let paths = H3PrivateUatPathSet::resolve(plan.engine_config.artifact_root.clone());

    let progress_tx = job.progress_tx.clone();
    let mut progress = mold_inference::progress::ProgressReporter::default();
    progress.set_callback(Box::new(move |event| {
        crate::gpu_worker::record_h3_progress(event, progress_tx.as_ref());
    }));
    // Bind the staged Ref2VA references immediately before preparation — the
    // reopen re-derives the frozen factory request through the same decoder
    // admission used, so it needs the same verified bindings. FL2VA jobs
    // carry no resolved set and bind nothing.
    let reference_bindings = job
        .resolved_references
        .as_ref()
        .map(|references| references.inference_bindings(&job.request, Some(&cancellation)))
        .transpose()
        .map_err(|error| {
            format!("MiniMax H3 owner preparation could not bind its staged references: {error}")
        })?
        .unwrap_or_default();
    progress.set_cancellation_token(cancellation);
    let prepared = mold_inference::prepare_h3_private_fl2va_attempt(
        mold_inference::H3PrivateFl2VaPrepareInput {
            request: &job.request,
            frozen_factory,
            admission_evidence,
            paths: paths.inference_paths(),
            references: reference_bindings,
            owner_fence: mold_inference::H3PrivateFl2VaOwnerFenceFacts {
                work_identity_sha256: work_identity_sha256.clone(),
                cancellation_scope_identity_sha256: cancellation_scope_identity_sha256.clone(),
                device_id: device_id.clone(),
                device_ordinal: worker.gpu.ordinal,
                compute_capability,
                memory_ledger_sequence: fence.memory_ledger_sequence,
                admission_evidence_identity_sha256: admission_evidence
                    .identity_sha256()
                    .to_string(),
                artifact_qualification_identity_sha256: admission_evidence
                    .artifact_qualification_identity_sha256()
                    .to_string(),
                runtime_qualification_identity_sha256: admission_evidence
                    .runtime_qualification_identity_sha256()
                    .to_string(),
                prepared_attempt_identity_sha256: admission_evidence
                    .prepared_attempt_identity_sha256()
                    .to_string(),
                target_budget_identity_sha256: admission_evidence
                    .target_budget_identity_sha256()
                    .to_string(),
                predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
                predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
            },
        },
        &progress,
    );
    progress.clear_cancellation_token();
    progress.clear_callback();
    let prepared = prepared.map_err(private_prepare_error_message)?;

    let boxed = InferenceH3PreparedAttempt::boxed(prepared);
    let facts = boxed.facts();
    if facts.device_id != device_id
        || facts.device_ordinal != worker.gpu.ordinal
        || facts.execution_identity_sha256 != admission_evidence.execution_fingerprint()
        || facts.prepared_attempt_identity_sha256
            != admission_evidence.prepared_attempt_identity_sha256()
        || facts.target_budget_identity_sha256 != admission_evidence.target_budget_identity_sha256()
        || facts.component_set_identity_sha256 != admission_evidence.component_set_identity_sha256()
        || facts.admission_evidence_identity_sha256 != admission_evidence.identity_sha256()
        || facts.artifact_qualification_identity_sha256
            != admission_evidence.artifact_qualification_identity_sha256()
        || facts.runtime_qualification_identity_sha256
            != admission_evidence.runtime_qualification_identity_sha256()
        || facts.work_identity_sha256 != work_identity_sha256
        || facts.cancellation_scope_identity_sha256 != cancellation_scope_identity_sha256
        || facts.memory_ledger_sequence != fence.memory_ledger_sequence
        || facts.predicted_device_peak_bytes != admission_evidence.predicted_device_peak_bytes()
        || facts.predicted_host_increment_bytes
            != admission_evidence.predicted_host_increment_bytes()
        || facts.consumption_identity_sha256.len() != 64
        || !facts
            .consumption_identity_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || facts.media != expected_media
    {
        return Err("MiniMax H3 prepared attempt changed from the frozen owner admission".into());
    }
    job.h3_prepared_attempt = Some(boxed);
    Ok(())
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn private_prepare_error_message(
    error: mold_inference::H3PrivateFl2VaPrepareError,
) -> String {
    match error {
        mold_inference::H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification => {
            "MiniMax H3 runtime has no reviewed runtime qualification".to_string()
        }
        mold_inference::H3PrivateFl2VaPrepareError::InsufficientHostHeadroom(shortfall) => {
            shortfall.to_string()
        }
        mold_inference::H3PrivateFl2VaPrepareError::InsufficientDeviceHeadroom(shortfall) => {
            shortfall.to_string()
        }
        mold_inference::H3PrivateFl2VaPrepareError::InvalidEvidence(reason) => {
            format!("MiniMax H3 preparation evidence was rejected: {reason}")
        }
    }
}

/// The host-memory shortfall a preparation refusal turns on, if that is what it
/// turned on.
///
/// Read by the admission retry (#1289) so it can release mold's own model cache
/// and ask again. Deliberately a match on the variant rather than a search of
/// the rendered sentence: prose is not evidence, and only a host shortfall may
/// be answered by eviction — a device shortfall would evict the cache for
/// memory eviction cannot supply.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn private_prepare_host_shortfall(
    error: &mold_inference::H3PrivateFl2VaPrepareError,
) -> Option<mold_inference::H3PrivateHostHeadroomShortfall> {
    match error {
        mold_inference::H3PrivateFl2VaPrepareError::InsufficientHostHeadroom(shortfall) => {
            Some(*shortfall)
        }
        _ => None,
    }
}

/// The device half of the same typed question. A device shortfall cannot be
/// answered by releasing mold's own cache — eviction supplies host bytes —
/// but it CAN be answered by waiting for the card, which is a park, not a
/// hold.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn private_prepare_device_shortfall(
    error: &mold_inference::H3PrivateFl2VaPrepareError,
) -> Option<mold_inference::H3PrivateDeviceHeadroomShortfall> {
    match error {
        mold_inference::H3PrivateFl2VaPrepareError::InsufficientDeviceHeadroom(shortfall) => {
            Some(*shortfall)
        }
        _ => None,
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn private_uat_path(name: &str, fallback: std::path::PathBuf) -> std::path::PathBuf {
    std::env::var_os(name)
        .map(std::path::PathBuf::from)
        .unwrap_or(fallback)
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn validate_private_h3_live_owner_route(
    expected_device_id: &str,
    expected_device_ordinal: usize,
    expected_compute_capability: Option<(u16, u16)>,
    actual_device_id: &str,
    actual_device_ordinal: usize,
    actual_compute_capability: Option<(u16, u16)>,
) -> Result<(), String> {
    if expected_device_id != actual_device_id
        || expected_device_ordinal != actual_device_ordinal
        || expected_compute_capability != actual_compute_capability
    {
        return Err("MiniMax H3 GPU route changed after allocation-free admission".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod presentation_tests {
    /// The advertised presets are exactly the canvases `mold_core`
    /// recommends, grouped by their canonical reduced-ratio identity (`7:4`
    /// for 1344x768, `1:1` for the squares). They are RECOMMENDATIONS on a
    /// continuous range, not a bucket set.
    #[test]
    fn reviewed_profile_keeps_the_canonical_exact_aspect_identity() {
        let (profile, _, pixels) = super::reviewed_h3_private_generation_profile()
            .expect("the reviewed H3 profile must resolve");
        let recipe = profile.default_recipe().expect("one reviewed recipe");

        let advertised = recipe
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| group.presets.iter())
            .map(|preset| (preset.width, preset.height))
            .collect::<Vec<_>>();
        // Order follows the aspect grouping, so compare as sets. A
        // stored-record build advertises the record's canvas alone.
        let mut sorted = advertised.clone();
        sorted.sort_unstable();
        let mut expected = if super::private_h3_record_runtime() {
            vec![(
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT,
            )]
        } else {
            mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES.to_vec()
        };
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        // The default canvas still leads.
        assert_eq!(
            advertised.first().copied(),
            Some((
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT
            ))
        );
        // A range, not a bucket set — unless a stored record is this
        // build's authority, which pins one canvas by equality.
        if super::private_h3_record_runtime() {
            assert_eq!(
                recipe.resolution.domain,
                mold_core::generation_profile::ResolutionDomain::Buckets
            );
        } else {
            assert_eq!(
                recipe.resolution.domain,
                mold_core::generation_profile::ResolutionDomain::Dynamic
            );
            assert_eq!(recipe.resolution.off_bucket, None);
        }
        for group in &recipe.resolution.aspect_groups {
            assert_eq!(group.id, group.label);
            for preset in &group.presets {
                // The bridge no longer overrides the tier: a "Reviewed" mark
                // on every size pill is noise, and the generated profile's
                // own `recommended` is the one every surface hides.
                assert_eq!(preset.tier, "recommended");
            }
        }
        assert_eq!(
            recipe
                .resolution
                .aspect_groups
                .iter()
                .map(|group| group.id.as_str())
                .collect::<Vec<_>>(),
            if super::private_h3_record_runtime() {
                vec!["7:4"]
            } else {
                vec!["7:4", "20:11", "16:9", "1:1", "4:7", "11:20"]
            }
        );

        // Every ceiling is the canvas RULE's own, so a client that reads a
        // ceiling rather than the preset list is never handed a number
        // smaller than a canvas admission accepts. A stored-record build has
        // no rule to read — every axis is the record's own value — and that
        // arm is pinned by
        // `the_advertised_profile_matches_this_build_s_runtime_authority`.
        if !super::private_h3_record_runtime() {
            assert_eq!(pixels, mold_core::minimax_h3::reviewed_compact_max_pixels());
            assert_eq!(recipe.resolution.max_pixels, pixels);
            assert_eq!(
                recipe.resolution.max_axis_pixels,
                Some(mold_core::minimax_h3::reviewed_compact_max_axis_pixels())
            );
            assert_eq!(
                recipe.resolution.min_width,
                mold_core::minimax_h3::reviewed_compact_min_axis_pixels()
            );
            assert_eq!(recipe.resolution.min_height, recipe.resolution.min_width);
            let (min_aspect, max_aspect) = mold_core::minimax_h3::reviewed_compact_aspect_bounds();
            assert_eq!(recipe.resolution.min_aspect_ratio, Some(min_aspect));
            assert_eq!(recipe.resolution.max_aspect_ratio, Some(max_aspect));
        }

        // The default the row seeds a form with stays the historical canvas.
        assert_eq!(recipe.defaults.width, mold_core::minimax_h3::DEFAULT_WIDTH);
        assert_eq!(
            recipe.defaults.height,
            mold_core::minimax_h3::DEFAULT_HEIGHT
        );

        // The frame and step axes are ranges for the base tier, on the
        // per-request-minted runtime. The defaults hold either way.
        let temporal = recipe.temporal.as_ref().expect("a temporal block");
        assert_eq!(
            temporal.frames.default,
            mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES
        );
        assert_eq!(
            recipe.steps.default,
            mold_core::minimax_h3::COMFY_DEFAULT_STEPS
        );
        if !super::private_h3_record_runtime() {
            assert_eq!(temporal.frames.min, mold_core::minimax_h3::MIN_FRAMES);
            assert_eq!(temporal.frames.max, mold_core::minimax_h3::MAX_FRAMES);
            assert_eq!(temporal.frames.step, mold_core::minimax_h3::FRAME_STEP);
            assert_eq!(
                temporal.frames.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
            assert_eq!(recipe.steps.min, mold_core::minimax_h3::COMPACT_MIN_STEPS);
            assert_eq!(recipe.steps.max, mold_core::minimax_h3::COMPACT_MAX_STEPS);
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
        }
    }

    /// What the row advertises must match WHICH RUNTIME is behind it.
    ///
    /// The public `h3` runtime mints its qualification envelope per request,
    /// so it may advertise the whole compact rule. A private-UAT build without
    /// `h3` authenticates a STORED record that pins one canvas, one clip
    /// length, and one step count by equality — `precheck_private_h3_record_
    /// canvas` refuses anything else before the ~37 GB artifact pass — so it
    /// must advertise exactly those. One predicate answers for both, and this
    /// test asserts the arm this build actually compiled.
    #[test]
    fn the_advertised_profile_matches_this_build_s_runtime_authority() {
        let (profile, _, pixels) = super::reviewed_h3_private_generation_profile()
            .expect("the reviewed H3 profile must resolve");
        let recipe = profile.default_recipe().expect("one reviewed recipe");
        let temporal = recipe.temporal.as_ref().expect("a temporal block");
        let presets = recipe
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| group.presets.iter())
            .map(|preset| (preset.width, preset.height))
            .collect::<Vec<_>>();
        let record = (
            mold_core::minimax_h3::DEFAULT_WIDTH,
            mold_core::minimax_h3::DEFAULT_HEIGHT,
        );

        if super::private_h3_record_runtime() {
            // A stored record is the authority: every axis is its own value,
            // SPATIAL included. `precheck_private_h3_record_canvas` accepts
            // only the record's exact canvas, so advertising the compact
            // rule's ceilings would offer sizes it refuses.
            assert_eq!(presets, vec![record]);
            assert_eq!(pixels, u64::from(record.0) * u64::from(record.1));
            assert_eq!(recipe.resolution.max_pixels, pixels);
            assert_eq!(
                recipe.resolution.max_axis_pixels,
                Some(record.0.max(record.1))
            );
            assert_eq!(recipe.resolution.min_width, record.0);
            assert_eq!(recipe.resolution.min_height, record.1);
            let aspect = f64::from(record.0) / f64::from(record.1);
            assert_eq!(recipe.resolution.min_aspect_ratio, Some(aspect));
            assert_eq!(recipe.resolution.max_aspect_ratio, Some(aspect));
            assert_eq!(
                recipe.resolution.domain,
                mold_core::generation_profile::ResolutionDomain::Buckets
            );
            assert_eq!(
                recipe.resolution.off_bucket,
                Some(mold_core::generation_profile::OffBucketPolicy::Reject)
            );
            assert_eq!(
                temporal.frames.mode,
                mold_core::generation_profile::ControlMode::Fixed
            );
            assert_eq!(
                temporal.frames.min,
                mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES
            );
            assert_eq!(
                temporal.frames.max,
                mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES
            );
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Fixed
            );
            assert_eq!(recipe.steps.min, recipe.steps.max);
            assert_eq!(recipe.steps.min, mold_core::minimax_h3::COMFY_DEFAULT_STEPS);
        } else {
            // The per-request-minted runtime: the whole compact rule.
            let mut sorted = presets.clone();
            sorted.sort_unstable();
            let mut expected = mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES.to_vec();
            expected.sort_unstable();
            assert_eq!(sorted, expected);
            assert_eq!(pixels, mold_core::minimax_h3::reviewed_compact_max_pixels());
            assert_eq!(recipe.resolution.max_pixels, pixels);
            assert_eq!(
                recipe.resolution.max_axis_pixels,
                Some(mold_core::minimax_h3::reviewed_compact_max_axis_pixels())
            );
            assert_eq!(
                recipe.resolution.min_width,
                mold_core::minimax_h3::MIN_COMPACT_AXIS_PIXELS
            );
            assert_eq!(
                recipe.resolution.min_height,
                mold_core::minimax_h3::MIN_COMPACT_AXIS_PIXELS
            );
            let (min_aspect, max_aspect) = mold_core::minimax_h3::reviewed_compact_aspect_bounds();
            assert_eq!(recipe.resolution.min_aspect_ratio, Some(min_aspect));
            assert_eq!(recipe.resolution.max_aspect_ratio, Some(max_aspect));
            assert_eq!(
                recipe.resolution.domain,
                mold_core::generation_profile::ResolutionDomain::Dynamic
            );
            assert_eq!(recipe.resolution.off_bucket, None);
            assert_eq!(
                temporal.frames.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
            assert_eq!(temporal.frames.min, mold_core::minimax_h3::MIN_FRAMES);
            assert_eq!(temporal.frames.max, mold_core::minimax_h3::MAX_FRAMES);
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
            assert_eq!(recipe.steps.min, mold_core::minimax_h3::COMPACT_MIN_STEPS);
            assert_eq!(recipe.steps.max, mold_core::minimax_h3::COMPACT_MAX_STEPS);
        }
    }

    /// The predicate itself: a private-UAT build without public `h3` is the
    /// only configuration whose authority is a stored record.
    #[test]
    fn only_a_private_uat_build_without_h3_reads_a_stored_record() {
        assert_eq!(
            super::private_h3_record_runtime(),
            cfg!(feature = "h3-private-uat") && !cfg!(feature = "h3")
        );
        if cfg!(feature = "h3") {
            assert!(!super::private_h3_record_runtime());
        }
    }

    /// A Turbo tier's step count IS its distilled adapter's schedule length,
    /// so it stays a fixed control while the base tier's became a range.
    #[test]
    fn a_turbo_tier_keeps_its_exact_step_count() {
        for tier in mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS {
            let (profile, _, _) =
                super::reviewed_h3_private_generation_profile_for(tier.model, tier.steps)
                    .expect("every reviewed Turbo tier resolves a profile");
            let recipe = profile.default_recipe().expect("one reviewed recipe");
            assert_eq!(recipe.steps.min, tier.steps, "{}", tier.model);
            assert_eq!(recipe.steps.max, tier.steps, "{}", tier.model);
            assert_eq!(recipe.steps.default, tier.steps, "{}", tier.model);
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Fixed,
                "{}",
                tier.model
            );
            // The frame axis is still the family grid: only the step axis is
            // the adapter's property. A stored-record build pins it instead.
            let temporal = recipe.temporal.as_ref().expect("a temporal block");
            if super::private_h3_record_runtime() {
                assert_eq!(
                    temporal.frames.max,
                    mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES
                );
            } else {
                assert_eq!(temporal.frames.max, mold_core::minimax_h3::MAX_FRAMES);
            }
        }
    }
}

#[cfg(all(test, any(feature = "h3", feature = "h3-private-uat")))]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn redacted_preview_endpoints_are_substituted_only_when_present_and_empty() {
        // The fixture has to satisfy the FL2VA request contract, or every
        // assertion below would be measuring the same early return: guidance
        // must be 0 and strength 1 (their serde defaults are 3.5 and 0.75,
        // neither of which H3 accepts), and a frame-0 keyframe beside a
        // `source_image` is a duplicate first-frame boundary.
        fn request(model: &str) -> mold_core::GenerateRequest {
            serde_json::from_value(serde_json::json!({
                "prompt": "probe",
                "model": model,
                "width": mold_core::minimax_h3::DEFAULT_WIDTH,
                "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
                "steps": mold_core::minimax_h3::DEFAULT_STEPS,
                "guidance": 0.0,
                "strength": 1.0,
                "batch_size": 1,
                "output_format": "mp4"
            }))
            .expect("test request must deserialize")
        }
        // The final frame of the default clip: FL2VA keyframes may target
        // only frame 0 or this one.
        let last_frame = mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES - 1;

        // A redacted first frame is substituted; a boundary carrying real
        // bytes is left exactly as it arrived.
        let mut fl2va = request(mold_core::minimax_h3::FL2VA_COMFY);
        fl2va.source_image = Some(Vec::new());
        fl2va.keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: last_frame,
            image: vec![1, 2, 3],
            name: None,
        }]);
        super::substitute_redacted_preview_endpoints(&mut fl2va);
        let substituted = fl2va.source_image.clone().unwrap();
        assert!(!substituted.is_empty());
        // Real bytes are never replaced.
        assert_eq!(fl2va.keyframes.as_deref().unwrap()[0].image, vec![1, 2, 3]);

        // Redacted keyframes are substituted too, and one placeholder serves
        // every endpoint of the same request.
        let mut boundaries = request(mold_core::minimax_h3::FL2VA_COMFY);
        boundaries.keyframes = Some(vec![
            mold_core::KeyframeCondition {
                frame: 0,
                image: Vec::new(),
                name: None,
            },
            mold_core::KeyframeCondition {
                frame: last_frame,
                image: Vec::new(),
                name: None,
            },
        ]);
        super::substitute_redacted_preview_endpoints(&mut boundaries);
        let keyframes = boundaries.keyframes.as_deref().unwrap();
        assert_eq!(keyframes[0].image, substituted);
        assert_eq!(keyframes[1].image, substituted);

        // Non-FL2VA requests pass through untouched.
        let mut ref2va = request(mold_core::minimax_h3::REF2VA_COMFY);
        ref2va.source_image = Some(Vec::new());
        super::substitute_redacted_preview_endpoints(&mut ref2va);
        assert_eq!(ref2va.source_image.as_deref(), Some(&[][..]));

        // An absent endpoint stays absent — presence is contract-relevant.
        let mut absent = request(mold_core::minimax_h3::FL2VA_COMFY);
        super::substitute_redacted_preview_endpoints(&mut absent);
        assert!(absent.source_image.is_none());

        // A contract-invalid geometry never allocates a placeholder — the
        // unauthenticated preview path must not be drivable into large
        // synchronous allocations, and admission reports the contract error.
        let mut oversized = request(mold_core::minimax_h3::FL2VA_COMFY);
        oversized.width = 65_504;
        oversized.height = 65_504;
        oversized.source_image = Some(Vec::new());
        super::substitute_redacted_preview_endpoints(&mut oversized);
        assert_eq!(oversized.source_image.as_deref(), Some(&[][..]));
    }

    #[test]
    fn staging_root_is_created_owner_only_when_missing_and_left_alone_when_present() {
        let temporary = tempfile::tempdir().unwrap();
        let staging = temporary.path().join("h3-runtime").join("staging");
        let paths = super::H3PrivateUatPathSet {
            models_root: temporary.path().join("models"),
            staging_root: staging.clone(),
            authorization_record: temporary.path().join("authorization-record.json"),
            runtime_qualification_record: temporary.path().join("runtime-qualification.json"),
        };

        paths.ensure_staging_root();
        let metadata = fs::symlink_metadata(&staging).unwrap();
        assert!(metadata.is_dir());
        // Fresh runtime-owned storage remains private by default.
        assert_eq!(metadata.permissions().mode() & 0o777, 0o700);

        // A pre-existing directory keeps its mode — ensure never loosens or
        // tightens an operator-managed staging root.
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o755)).unwrap();
        paths.ensure_staging_root();
        let unchanged = fs::symlink_metadata(&staging).unwrap();
        assert_eq!(unchanged.permissions().mode() & 0o777, 0o755);
    }

    fn private_file(path: &std::path::Path, bytes: u64) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .unwrap();
        file.set_len(bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }

    fn capability_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let models = root.path().join("models");
        let manifest =
            mold_core::manifest::find_manifest(mold_core::minimax_h3::FL2VA_COMFY).unwrap();
        for file in &manifest.files {
            private_file(
                &models.join(mold_core::manifest::storage_path(manifest, file)),
                file.size_bytes,
            );
        }
        (root, models)
    }

    #[test]
    fn authenticated_capability_advertises_only_the_reviewed_fl2va_partition() {
        let (_root, models) = capability_fixture();
        let capability = super::build_fl2va_capability(&models)
            .expect("complete manifest projection must advertise");

        assert!(capability.runtime_available);
        assert_eq!(capability.qualification.backend, "cuda-or-metal");
        assert!(capability.qualification.metal_supported);
        assert_eq!(capability.partitions.len(), 1);
        assert_eq!(capability.partitions[0].task, "fl2va");
        assert_eq!(
            capability.partitions[0].model,
            mold_core::minimax_h3::FL2VA_COMFY
        );
        assert_eq!(capability.components.len(), 5);
        assert!(capability
            .components
            .iter()
            .all(|component| component.state == "installed"));
        assert_eq!(
            capability.partitions[0].component_ids.len(),
            capability.components.len()
        );
        assert_eq!(
            capability.partitions[0].request,
            Some(mold_core::MiniMaxH3RequestCapability {
                width: mold_core::minimax_h3::DEFAULT_WIDTH,
                height: mold_core::minimax_h3::DEFAULT_HEIGHT,
                frames: mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                fps: mold_core::minimax_h3::FIXED_FPS,
                steps: mold_core::minimax_h3::COMFY_DEFAULT_STEPS,
                batch_size: 1,
                output_format: "mp4".into(),
                required_endpoint: "first".into(),
                generation_profile_sha256: capability.partitions[0]
                    .request
                    .as_ref()
                    .unwrap()
                    .generation_profile_sha256
                    .clone(),
            })
        );
    }

    #[test]
    fn authenticated_model_row_is_exactly_the_reviewed_quality_envelope() {
        let (_root, models) = capability_fixture();
        let capability = super::build_fl2va_capability(&models).unwrap();
        let row = super::authenticated_h3_private_model_row(&capability)
            .expect("complete authenticated partition must produce one row");

        assert_eq!(row.info.name, mold_core::minimax_h3::FL2VA_COMFY);
        assert_eq!(row.info.family, mold_core::minimax_h3::FAMILY);
        assert!(row.downloaded);
        assert_eq!(row.info.hf_repo, mold_core::minimax_h3::COMFY_REPO);
        assert_eq!(row.defaults.default_width, 1344);
        assert_eq!(row.defaults.default_height, 768);
        assert_eq!(row.defaults.default_frames, Some(124));
        assert_eq!(row.defaults.min_frames, Some(124));
        assert_eq!(row.defaults.max_frames, Some(124));
        assert_eq!(row.defaults.default_steps, 21);
        assert_eq!(
            row.source_image,
            Some(mold_core::SourceImageCapability::Required)
        );
        let profile = row.generation_profile.unwrap();
        assert_eq!(
            capability.partitions[0]
                .request
                .as_ref()
                .unwrap()
                .generation_profile_sha256,
            profile.profile_hash
        );
        let recipe = profile.default_recipe().unwrap();
        assert_eq!(
            recipe.resolution.domain,
            if super::private_h3_record_runtime() {
                mold_core::generation_profile::ResolutionDomain::Buckets
            } else {
                mold_core::generation_profile::ResolutionDomain::Dynamic
            }
        );
        // Presets are grouped by aspect, so their order is the grouping's;
        // compare as a set. A stored-record build advertises the record's
        // canvas alone, because that is the only one it admits.
        let mut advertised = recipe
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| group.presets.iter())
            .map(|preset| (preset.width, preset.height))
            .collect::<Vec<_>>();
        advertised.sort_unstable();
        let mut expected = if super::private_h3_record_runtime() {
            vec![(
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT,
            )]
        } else {
            mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES.to_vec()
        };
        expected.sort_unstable();
        assert_eq!(advertised, expected);
        // The default canvas leads either way, in its own 7:4 group; only a
        // stored-record build has that group to itself.
        assert_eq!(recipe.resolution.aspect_groups[0].id, "7:4");
        assert_eq!(recipe.resolution.aspect_groups[0].label, "7:4");
        assert_eq!(recipe.resolution.aspect_groups[0].presets[0].width, 1344);
        assert_eq!(recipe.resolution.aspect_groups[0].presets[0].height, 768);
        if super::private_h3_record_runtime() {
            assert_eq!(recipe.resolution.aspect_groups.len(), 1);
            assert_eq!(recipe.resolution.aspect_groups[0].presets.len(), 1);
        }
        // The row's spatial metadata follows the same runtime authority as
        // the profile: a stored-record build admits one canvas, so that is
        // the only thing it may recommend or bound.
        let advertised_dimensions = row
            .defaults
            .recommended_dimensions
            .iter()
            .map(|entry| (entry.width, entry.height))
            .collect::<Vec<_>>();
        if super::private_h3_record_runtime() {
            let record = (
                mold_core::minimax_h3::DEFAULT_WIDTH,
                mold_core::minimax_h3::DEFAULT_HEIGHT,
            );
            assert_eq!(advertised_dimensions, vec![record]);
            assert_eq!(
                row.defaults.max_pixels,
                Some(u64::from(record.0) * u64::from(record.1))
            );
            assert_eq!(row.defaults.max_axis_pixels, Some(record.0.max(record.1)));
        } else {
            // The same slice, in order, default first — the row does not
            // group by aspect.
            assert_eq!(
                advertised_dimensions,
                mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES.to_vec()
            );
            assert_eq!(
                row.defaults.max_pixels,
                Some(mold_core::minimax_h3::reviewed_compact_max_pixels())
            );
            assert_eq!(
                row.defaults.max_axis_pixels,
                Some(mold_core::minimax_h3::reviewed_compact_max_axis_pixels())
            );
        }
        assert_eq!(
            recipe.resolution.aspect_groups[0].presets[0].tier,
            "recommended"
        );
        assert_eq!(recipe.steps.default, 21);
        let temporal = recipe.temporal.as_ref().unwrap();
        assert_eq!(temporal.frames.default, 124);
        if super::private_h3_record_runtime() {
            // A stored record pins every axis by equality.
            assert_eq!(recipe.steps.min, 21);
            assert_eq!(recipe.steps.max, 21);
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Fixed
            );
            assert_eq!(temporal.frames.min, 124);
            assert_eq!(temporal.frames.max, 124);
            assert_eq!(
                temporal.frames.mode,
                mold_core::generation_profile::ControlMode::Fixed
            );
        } else {
            // The per-request-minted runtime advertises the compact ranges.
            assert_eq!(recipe.steps.min, mold_core::minimax_h3::COMPACT_MIN_STEPS);
            assert_eq!(recipe.steps.max, mold_core::minimax_h3::COMPACT_MAX_STEPS);
            assert_eq!(
                recipe.steps.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
            assert_eq!(temporal.frames.min, mold_core::minimax_h3::MIN_FRAMES);
            assert_eq!(temporal.frames.max, mold_core::minimax_h3::MAX_FRAMES);
            assert_eq!(
                temporal.frames.mode,
                mold_core::generation_profile::ControlMode::Adjustable
            );
        }
    }

    fn place_turbo_adapter(
        models: &std::path::Path,
        tier: &mold_core::minimax_h3::TurboManifestTier,
    ) {
        let manifest = mold_core::manifest::find_manifest(tier.model).unwrap();
        let adapter = manifest
            .files
            .iter()
            .find(|file| file.component == mold_core::manifest::ModelComponent::DistilledLora)
            .unwrap();
        private_file(
            &models.join(mold_core::manifest::storage_path(manifest, adapter)),
            adapter.size_bytes,
        );
    }

    #[test]
    fn turbo_variants_ride_the_partition_additively_and_gate_on_the_adapter() {
        let (_root, models) = capability_fixture();

        // Base stack only: both reviewed variants advertise as not installed
        // and carry no request envelope, and the model rows stay base-only.
        let capability = super::build_fl2va_capability(&models).unwrap();
        assert_eq!(capability.partitions.len(), 1, "one partition per task");
        let turbo = &capability.partitions[0].turbo;
        assert_eq!(
            turbo.len(),
            mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS.len()
        );
        for variant in turbo {
            assert!(!variant.installed);
            assert!(variant.request.is_none());
        }
        let rows = super::authenticated_h3_private_model_rows(&capability);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].info.name, mold_core::minimax_h3::FL2VA_COMFY);

        // Landing the 8-step adapter makes exactly that variant executable:
        // its envelope moves only the step axis, its profile hash is its own,
        // and its model row advertises the tier's reviewed step default.
        let eight_step = &mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS[0];
        assert_eq!(
            eight_step.model,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP
        );
        place_turbo_adapter(&models, eight_step);
        let capability = super::build_fl2va_capability(&models).unwrap();
        let turbo = &capability.partitions[0].turbo;
        let variant = turbo
            .iter()
            .find(|variant| variant.model == eight_step.model)
            .unwrap();
        assert!(variant.installed);
        let request = variant.request.as_ref().unwrap();
        assert_eq!(request.steps, eight_step.steps);
        assert_eq!(request.width, mold_core::minimax_h3::DEFAULT_WIDTH);
        assert_eq!(request.height, mold_core::minimax_h3::DEFAULT_HEIGHT);
        assert_eq!(
            request.frames,
            mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES
        );
        let base_request = capability.partitions[0].request.as_ref().unwrap();
        assert_ne!(
            request.generation_profile_sha256,
            base_request.generation_profile_sha256
        );
        assert!(turbo.iter().any(|variant| variant.model
            == mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P
            && !variant.installed));

        let rows = super::authenticated_h3_private_model_rows(&capability);
        assert_eq!(rows.len(), 2);
        let turbo_row = rows
            .iter()
            .find(|row| row.info.name == eight_step.model)
            .unwrap();
        assert_eq!(turbo_row.defaults.default_steps, eight_step.steps);
        assert_eq!(turbo_row.defaults.default_width, 1344);
        assert_eq!(turbo_row.defaults.default_height, 768);
        assert!(turbo_row.downloaded);
        assert_eq!(
            turbo_row.disk_usage_bytes.unwrap(),
            rows[0].disk_usage_bytes.unwrap() + eight_step.adapter_size_bytes
        );
        let profile = turbo_row.generation_profile.as_ref().unwrap();
        assert_eq!(profile.profile_hash, request.generation_profile_sha256);
        let recipe = profile.default_recipe().unwrap();
        assert_eq!(recipe.steps.min, eight_step.steps);
        assert_eq!(recipe.steps.max, eight_step.steps);

        // A widened variant envelope is refused rather than advertised.
        let mut widened = capability.clone();
        widened.partitions[0].turbo[0]
            .request
            .as_mut()
            .unwrap()
            .steps = mold_core::minimax_h3::COMFY_DEFAULT_STEPS;
        assert_eq!(
            super::authenticated_h3_private_model_rows(&widened).len(),
            1
        );
    }

    #[test]
    fn capability_reports_missing_components_without_inventing_download_authority() {
        let (_root, models) = capability_fixture();
        let manifest =
            mold_core::manifest::find_manifest(mold_core::minimax_h3::FL2VA_COMFY).unwrap();
        let transformer = manifest
            .files
            .iter()
            .find(|file| file.component == mold_core::manifest::ModelComponent::Transformer)
            .unwrap();
        fs::remove_file(models.join(mold_core::manifest::storage_path(manifest, transformer)))
            .unwrap();
        let capability = super::build_fl2va_capability(&models).unwrap();
        assert_eq!(
            capability
                .components
                .iter()
                .find(|component| component.role == "transformer")
                .unwrap()
                .state,
            "missing"
        );
        assert!(super::authenticated_h3_private_model_row(&capability).is_none());
    }

    #[test]
    fn model_row_rejects_any_widened_partition_axis() {
        let (_root, models) = capability_fixture();
        let capability = super::build_fl2va_capability(&models).unwrap();
        for mutate in [
            |value: &mut mold_core::MiniMaxH3RequestCapability| value.width += 64,
            |value: &mut mold_core::MiniMaxH3RequestCapability| value.frames += 17,
            |value: &mut mold_core::MiniMaxH3RequestCapability| value.steps += 1,
        ] {
            let mut widened = capability.clone();
            mutate(widened.partitions[0].request.as_mut().unwrap());
            assert!(super::authenticated_h3_private_model_row(&widened).is_none());
        }
    }

    #[test]
    fn missing_reviewed_runtime_qualification_stays_a_typed_fail_closed_rejection() {
        assert_eq!(
            super::private_prepare_error_message(
                mold_inference::H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification,
            ),
            "MiniMax H3 runtime has no reviewed runtime qualification",
        );
    }
}

#[cfg(test)]
mod structural_tests {
    use super::BoxedH3PreparedAttempt;
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::PermissionsExt;

    const INSTANCE_ID: &str = "test-server-instance";

    #[test]
    fn private_capability_is_auth_bound_fl2va_only_and_manifest_derived() {
        fn sparse_file(path: &std::path::Path, bytes: u64) {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .unwrap();
            file.set_len(bytes).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
        }

        let root = tempfile::tempdir().unwrap();
        let models = root.path().join("models");
        let manifest =
            mold_core::manifest::find_manifest(mold_core::minimax_h3::FL2VA_COMFY).unwrap();
        for file in &manifest.files {
            sparse_file(
                &models.join(mold_core::manifest::storage_path(manifest, file)),
                file.size_bytes,
            );
        }
        let capability = super::build_fl2va_capability(&models).unwrap();
        assert_eq!(capability.partitions.len(), 1);
        assert_eq!(capability.partitions[0].task, "fl2va");
        assert_eq!(
            capability.partitions[0].model,
            mold_core::minimax_h3::FL2VA_COMFY
        );
        assert_eq!(capability.components.len(), 5);
        assert!(capability
            .components
            .iter()
            .all(|component| component.state == "installed"));
        assert!(capability.qualification.metal_supported);
    }

    fn request(model: &str) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "private prompt bytes must not enter the ingress grant",
            "model": model,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": mold_core::minimax_h3::DEFAULT_STEPS,
            "batch_size": 1,
            "output_format": "mp4"
        }))
        .expect("test request must deserialize")
    }

    fn authenticated() -> crate::auth::ApiKeyAuthenticated {
        crate::auth::ApiKeyAuthenticated {
            identity: "process-local-auth-marker".to_string(),
            durable_identity: "restart-stable-auth-marker".to_string(),
        }
    }

    fn ref2va_request(swapped: bool) -> mold_core::GenerateRequest {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
        };

        let provenance = |name: &str, byte: u8| GenerationReferenceProvenance {
            name: Some(name.to_string()),
            sha256: Some(format!("{byte:02x}").repeat(32)),
        };
        let image = GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance("subject.png", 1),
            mime_type: "image/png".to_string(),
            width: 1024,
            height: 768,
        };
        let audio = GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance("voice.wav", 2),
            mime_type: "audio/wav".to_string(),
            duration_ms: 2_000,
            sample_rate: 48_000,
            channels: 1,
            sample_count: Some(96_000),
        };
        let mut request = request(mold_core::minimax_h3::REF2VA_COMFY);
        request.seed = Some(7);
        request.guidance = 0.0;
        request.strength = 1.0;
        request.enable_audio = Some(true);
        request.references = Some(if swapped {
            vec![audio, image]
        } else {
            vec![image, audio]
        });
        request
    }

    #[test]
    fn opaque_attempt_is_non_clone_and_absent_from_scheduler_registry_and_cache_types() {
        trait AmbiguousIfClone<A> {
            fn assert_not_implemented() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        struct Marker;
        impl<T: Clone> AmbiguousIfClone<Marker> for T {}
        <BoxedH3PreparedAttempt as AmbiguousIfClone<_>>::assert_not_implemented();

        let gpu_pool = include_str!("gpu_pool.rs");
        assert!(gpu_pool.contains(
            "h3_prepared_attempt: Option<crate::h3_private_bridge::BoxedH3PreparedAttempt>"
        ));
        for (label, source) in [
            ("scheduler", include_str!("scheduler/mod.rs")),
            ("job registry", include_str!("job_registry.rs")),
            ("model cache", include_str!("model_cache.rs")),
        ] {
            assert!(
                !source.contains("BoxedH3PreparedAttempt"),
                "{label} must not retain the opaque private H3 attempt type",
            );
        }
    }

    #[test]
    fn non_h3_ingress_keeps_the_public_path_and_never_reads_private_runtime_state() {
        let runtime_checked = std::cell::Cell::new(false);
        let grant = super::classify_h3_private_ingress_with_runtime(
            &request("flux-schnell:q8"),
            None,
            INSTANCE_ID,
            |_| {
                runtime_checked.set(true);
                true
            },
        )
        .expect("non-H3 classification must remain a no-op");

        assert!(grant.is_none());
        assert!(!runtime_checked.get());
    }

    /// #1354: the private boundary claims only identities mold can RUN.
    ///
    /// Every pinned identity with no engine arm for its weight layout defers
    /// to `model_activation`, which answers the same 501 sentence the model's
    /// own `/api/models` row publishes. The deferral happens before the
    /// credential check and before the runtime lookup, so it is the same
    /// answer on a public `h3` build and on the private UAT build.
    #[test]
    fn pinned_unrunnable_h3_identities_defer_to_public_model_activation() {
        for model in [
            mold_core::minimax_h3::FL2VA_OFFICIAL,
            mold_core::minimax_h3::REF2VA_OFFICIAL,
            mold_core::minimax_h3::FL2VA_COMFY_NVFP4,
            mold_core::minimax_h3::REF2VA_COMFY_NVFP4,
        ] {
            assert!(
                mold_core::is_pinned_unrunnable_minimax_h3_identity(model),
                "{model} must be a pinned unrunnable identity",
            );
            let runtime_checked = std::cell::Cell::new(false);
            let grant = super::classify_h3_private_ingress_with_runtime(
                &request(model),
                None,
                INSTANCE_ID,
                |_| {
                    runtime_checked.set(true);
                    true
                },
            )
            .unwrap_or_else(|error| {
                panic!("{model} must defer rather than refuse: {}", error.code)
            });

            assert!(grant.is_none(), "{model}");
            assert!(!runtime_checked.get(), "{model}");
        }
    }

    /// The deferral is layout-level only: the reviewed compact task partition
    /// — including Ref2VA, whose private ingress seam this boundary owns —
    /// stays this classifier's to answer.
    #[test]
    fn the_reviewed_compact_partition_is_still_claimed_by_the_private_boundary() {
        for model in [
            mold_core::minimax_h3::FL2VA_COMFY,
            mold_core::minimax_h3::REF2VA_COMFY,
        ] {
            assert!(
                !mold_core::is_pinned_unrunnable_minimax_h3_identity(model),
                "{model} must stay off the deferral path",
            );
        }
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn exact_h3_partition_requires_api_key_authentication_before_runtime_lookup() {
        use axum::response::IntoResponse;

        let runtime_checked = std::cell::Cell::new(false);
        let error = super::classify_h3_private_ingress_with_runtime(
            &request(mold_core::minimax_h3::FL2VA_COMFY),
            None,
            INSTANCE_ID,
            |_| {
                runtime_checked.set(true);
                true
            },
        )
        .expect_err("unauthenticated H3 ingress must fail closed");

        assert_eq!(error.code, "UNAUTHORIZED");
        assert_eq!(
            error.into_response().status(),
            axum::http::StatusCode::UNAUTHORIZED
        );
        assert!(!runtime_checked.get());
    }

    #[test]
    #[cfg(feature = "h3")]
    fn public_h3_partition_requires_runtime_but_not_api_key_authentication() {
        let runtime_checked = std::cell::Cell::new(false);
        let grant = super::classify_h3_private_ingress_with_runtime(
            &request(mold_core::minimax_h3::FL2VA_COMFY),
            None,
            INSTANCE_ID,
            |_| {
                runtime_checked.set(true);
                true
            },
        )
        .expect("public H3 classification")
        .expect("public H3 grant");

        assert!(runtime_checked.get());
        grant
            .validate_for_request(&request(mold_core::minimax_h3::FL2VA_COMFY), INSTANCE_ID)
            .expect("public grant binds the exact request and server");
    }

    #[test]
    fn ingress_grant_rebinds_only_an_inference_resolved_seed() {
        let submitted = request(mold_core::minimax_h3::FL2VA_COMFY);
        assert!(submitted.seed.is_none());
        let grant = super::classify_h3_private_ingress_with_runtime(
            &submitted,
            Some(&authenticated()),
            INSTANCE_ID,
            |_| true,
        )
        .expect("exact authenticated partition")
        .expect("private grant");

        let mut resolved = submitted.clone();
        resolved.seed = Some(42);
        let rebound = grant
            .rebind_resolved_request(&submitted, &resolved, INSTANCE_ID)
            .expect("one omitted seed may be resolved once");
        assert!(grant.validate_for_request(&resolved, INSTANCE_ID).is_err());
        rebound
            .validate_for_request(&resolved, INSTANCE_ID)
            .expect("rebound grant must bind the exact resolved request");

        let mut mutated = resolved.clone();
        mutated.prompt.push_str(" changed");
        assert!(rebound.validate_for_request(&mutated, INSTANCE_ID).is_err());
        assert!(grant
            .rebind_resolved_request(&submitted, &mutated, INSTANCE_ID)
            .is_err());
    }

    #[test]
    fn ref2va_ingress_is_task_scoped_and_server_instance_bound() {
        use axum::response::IntoResponse;

        let request = ref2va_request(false);
        let seen_task = std::cell::Cell::new(None);
        let grant = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&authenticated()),
            INSTANCE_ID,
            |task| {
                seen_task.set(Some(task));
                task == mold_core::minimax_h3::Task::Ref2va
            },
        )
        .expect("exact Ref2VA partition must classify with injected authority")
        .expect("exact Ref2VA partition must return a grant");
        assert_eq!(seen_task.get(), Some(mold_core::minimax_h3::Task::Ref2va));
        grant
            .validate_for_request(&request, INSTANCE_ID)
            .expect("unchanged task and server instance must validate");
        assert!(grant
            .validate_for_request(&request, "replacement-server-instance")
            .is_err());

        let error = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&authenticated()),
            INSTANCE_ID,
            |task| task == mold_core::minimax_h3::Task::Fl2va,
        )
        .expect_err("an FL2VA qualification must not authorize Ref2VA");
        assert_eq!(error.code, super::H3_PRIVATE_RUNTIME_UNAVAILABLE);
        assert_eq!(
            error.into_response().status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn ref2va_preparation_binds_swapped_order_and_resolved_artifact_identity() {
        let original = ref2va_request(false);
        let original_resolved =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&original);
        let original_media = super::H3PreparedMediaContract::from_request(
            &original,
            Some(original_resolved.fingerprint()),
        )
        .expect("resolved Ref2VA request must freeze ordered authority");

        let swapped = ref2va_request(true);
        let swapped_resolved =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&swapped);
        let swapped_media = super::H3PreparedMediaContract::from_request(
            &swapped,
            Some(swapped_resolved.fingerprint()),
        )
        .expect("swapped Ref2VA request must freeze its distinct order");

        assert_ne!(
            original_media.reference_fingerprint_sha256,
            swapped_media.reference_fingerprint_sha256
        );
        assert_ne!(
            original_media.resolved_reference_fingerprint_sha256,
            swapped_media.resolved_reference_fingerprint_sha256
        );
        assert!(original_media
            .validate_for_request(&swapped, Some(swapped_resolved.fingerprint()))
            .is_err());

        let mut replaced = original.clone();
        let first = replaced
            .references
            .as_mut()
            .and_then(|references| references.first_mut())
            .expect("fixture reference");
        match first {
            mold_core::GenerationReference::Image { provenance, .. }
            | mold_core::GenerationReference::Video { provenance, .. }
            | mold_core::GenerationReference::Audio { provenance, .. } => {
                provenance.sha256 = Some("03".repeat(32));
            }
        }
        assert!(super::H3PreparedMediaContract::from_request(
            &replaced,
            Some(original_resolved.fingerprint()),
        )
        .is_err());

        let metadata = mold_core::OutputMetadata::from_generate_request(&original, 7, None, "test");
        let durable = metadata.references.expect("durable ordered references");
        assert_eq!(
            durable.iter().map(|item| item.index).collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            mold_core::generation_reference_fingerprint(&durable),
            original_media
                .reference_fingerprint_sha256
                .as_deref()
                .expect("prepared reference fingerprint")
        );
        let serialized = serde_json::to_string(&durable).unwrap();
        for private in ["descriptor", "upload", "server_path", "api_key"] {
            assert!(!serialized.contains(private));
        }

        let replay_request: mold_core::GenerateRequest =
            serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        let replay_resolved =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(
                &replay_request,
            );
        let replay_media = super::H3PreparedMediaContract::from_request(
            &replay_request,
            Some(replay_resolved.fingerprint()),
        )
        .expect("descriptor-only durable replay must retain exact order identity");
        assert_eq!(replay_media, original_media);
    }

    /// A REVIEWED identity submitted outside its partition still fails closed
    /// here. The pinned identities mold has no loader for used to be in this
    /// list too; since #1354 they defer to `model_activation` instead, which
    /// is `pinned_unrunnable_h3_identities_defer_to_public_model_activation`.
    #[test]
    fn wrong_h3_partitions_are_rejected_before_runtime_or_artifact_work() {
        let auth = authenticated();
        let mut wrong_conditioning = request(mold_core::minimax_h3::FL2VA_COMFY);
        wrong_conditioning.references = Some(Vec::new());
        for mut invalid in [
            request(mold_core::minimax_h3::REF2VA_COMFY),
            wrong_conditioning,
        ] {
            let runtime_checked = std::cell::Cell::new(false);
            let error = super::classify_h3_private_ingress_with_runtime(
                &invalid,
                Some(&auth),
                INSTANCE_ID,
                |_| {
                    runtime_checked.set(true);
                    true
                },
            )
            .expect_err("wrong H3 task/layout must fail closed");
            assert_eq!(error.code, super::H3_PRIVATE_PARTITION_REJECTED);
            assert!(!runtime_checked.get());

            invalid.model = mold_core::minimax_h3::FL2VA_COMFY.to_string();
            invalid.references = None;
            invalid.batch_size = 2;
            let error = super::classify_h3_private_ingress_with_runtime(
                &invalid,
                Some(&auth),
                INSTANCE_ID,
                |_| {
                    runtime_checked.set(true);
                    true
                },
            )
            .expect_err("batch H3 ingress must fail closed");
            assert_eq!(error.code, super::H3_PRIVATE_PARTITION_REJECTED);
            assert!(!runtime_checked.get());
        }
    }

    #[test]
    fn accepted_ingress_grant_is_cloneable_and_payload_free() {
        let auth = authenticated();
        let request = request(mold_core::minimax_h3::FL2VA_COMFY);
        let grant = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&auth),
            INSTANCE_ID,
            |_| true,
        )
        .expect("exact private partition must classify")
        .expect("exact private partition must return a grant");
        let cloned = grant.clone();

        assert_eq!(cloned.canonical_model(), mold_core::minimax_h3::FL2VA_COMFY);
        assert_eq!(cloned.authenticated_identity_sha256().len(), 64);
        assert_eq!(cloned.instance_identity_sha256().len(), 64);
        assert_eq!(cloned.partition_identity_sha256().len(), 64);
        assert_eq!(cloned.policy_identity_sha256().len(), 64);
        assert_eq!(cloned.request_authority_sha256().len(), 64);
        cloned
            .validate_for_request(&request, INSTANCE_ID)
            .expect("unmodified canonical request must retain its grant");
        let mut mutated = request.clone();
        mutated.prompt.push_str(" changed");
        assert!(cloned.validate_for_request(&mutated, INSTANCE_ID).is_err());
        let debug = format!("{cloned:?}");
        assert!(!debug.contains(&auth.identity));
        assert!(!debug.contains(&request.prompt));
    }

    #[test]
    fn idempotency_subject_is_stable_but_replay_authority_is_exact() {
        let auth = authenticated();
        let request = request(mold_core::minimax_h3::FL2VA_COMFY);
        let admitted = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&auth),
            INSTANCE_ID,
            |_| true,
        )
        .unwrap()
        .unwrap();

        let restarted_auth = crate::auth::ApiKeyAuthenticated {
            identity: "new-process-local-auth-marker".to_string(),
            durable_identity: auth.durable_identity.clone(),
        };
        let same_client_other_instance = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&restarted_auth),
            "different-instance",
            |_| true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            admitted.idempotency_subject_sha256(),
            same_client_other_instance.idempotency_subject_sha256(),
            "client-operation idempotency must not conflict after a server restart"
        );
        assert_ne!(
            admitted.authority_identity_sha256(),
            same_client_other_instance.authority_identity_sha256(),
            "durable replay authority must remain bound to the admitting instance"
        );

        let other_auth = crate::auth::ApiKeyAuthenticated {
            identity: "different-process-local-auth-marker".to_string(),
            durable_identity: "different-restart-stable-auth-marker".to_string(),
        };
        let other_client = super::classify_h3_private_ingress_with_runtime(
            &request,
            Some(&other_auth),
            INSTANCE_ID,
            |_| true,
        )
        .unwrap()
        .unwrap();
        assert_ne!(
            admitted.idempotency_subject_sha256(),
            other_client.idempotency_subject_sha256(),
            "different authenticated clients must never share idempotency authority"
        );

        let envelope = admitted.durable_replay_envelope().unwrap();
        let restored = super::restore_durable_h3_private_ingress_with_runtime(
            &request,
            &envelope,
            INSTANCE_ID,
            |_| true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            restored.authority_identity_sha256(),
            admitted.authority_identity_sha256()
        );
        restored
            .validate_for_request(&request, INSTANCE_ID)
            .expect("durable replay must revalidate the exact request");

        let mut changed = request;
        changed.prompt.push_str(" changed");
        assert!(restored
            .validate_for_request(&changed, INSTANCE_ID)
            .is_err());
    }

    #[test]
    fn durable_path_staging_does_not_change_h3_request_authority() {
        let auth = authenticated();
        let mut submitted = request(mold_core::minimax_h3::FL2VA_COMFY);
        submitted.source_video_path = Some("/submitted/source.mp4".into());
        submitted.audio_file_path = Some("/submitted/audio.wav".into());
        let grant = super::classify_h3_private_ingress_with_runtime(
            &submitted,
            Some(&auth),
            INSTANCE_ID,
            |_| true,
        )
        .unwrap()
        .unwrap();

        let mut hydrated = submitted.clone();
        hydrated.source_video_path = Some("/private/runtime/source.mp4".into());
        hydrated.audio_file_path = Some("/private/runtime/audio.wav".into());
        grant
            .validate_for_request(&hydrated, INSTANCE_ID)
            .expect("AEAD-bound staging paths are not caller request authority");

        hydrated.prompt.push_str(" changed");
        assert!(grant.validate_for_request(&hydrated, INSTANCE_ID).is_err());
    }

    #[test]
    fn durable_h3_envelope_rejects_non_digest_authority() {
        let error = super::restore_durable_h3_private_ingress(
            &request(mold_core::minimax_h3::FL2VA_COMFY),
            br#"{"version":1,"authenticated_identity_sha256":"not-a-digest","authority_identity_sha256":"not-a-digest"}"#,
            INSTANCE_ID,
        )
        .expect_err("malformed durable authority must fail closed");
        assert_eq!(error.code, super::H3_PRIVATE_PARTITION_REJECTED);
    }

    #[test]
    fn private_owner_route_rejects_changed_compute_capability() {
        super::validate_private_h3_live_owner_route(
            "cuda:0",
            0,
            Some((8, 9)),
            "cuda:0",
            0,
            Some((8, 9)),
        )
        .expect("unchanged live route must validate");
        let error = super::validate_private_h3_live_owner_route(
            "cuda:0",
            0,
            Some((8, 9)),
            "cuda:0",
            0,
            Some((9, 0)),
        )
        .expect_err("changed compute capability must fail the second owner fence");
        assert!(error.contains("GPU route changed"));
    }
}
