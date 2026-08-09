//! Narrow server boundary for one private MiniMax H3 owner attempt.
//!
//! The concrete inference value is intentionally hidden behind a non-Clone
//! trait object. It may ride the final `GpuJob` owner handoff, but it cannot be
//! copied into scheduler state, a registry row, or the generic model cache.

use crate::h3_attempt::H3AttemptScope;

#[cfg(any(test, feature = "h3-private-uat"))]
pub(crate) const H3_PRIVATE_PARTITION_REJECTED: &str = "MINIMAX_H3_PRIVATE_PARTITION_REJECTED";
#[cfg(any(test, feature = "h3-private-uat"))]
pub(crate) const H3_PRIVATE_RUNTIME_UNAVAILABLE: &str = "MINIMAX_H3_PRIVATE_RUNTIME_UNAVAILABLE";

/// Payload-free proof that one authenticated request crossed the deliberately
/// narrow private ingress partition. This value may be cloned through queue
/// and dependency-preparation state; it contains neither the API key/auth
/// marker nor prompt, media, reference, or filesystem payloads.
#[cfg(any(test, feature = "h3-private-uat"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateIngressGrant {
    canonical_model: String,
    authenticated_identity_sha256: String,
    partition_identity_sha256: String,
    policy_identity_sha256: String,
    request_authority_sha256: String,
}

#[cfg(any(test, feature = "h3-private-uat"))]
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
                self.authenticated_identity_sha256.as_bytes(),
                self.partition_identity_sha256.as_bytes(),
                self.policy_identity_sha256.as_bytes(),
                self.request_authority_sha256.as_bytes(),
            ],
        )
    }

    pub(crate) fn validate_for_request(
        &self,
        request: &mold_core::GenerateRequest,
    ) -> Result<(), String> {
        if request.model != self.canonical_model {
            return Err("MiniMax H3 ingress authority model changed after authentication".into());
        }
        let request_authority_sha256 = request_authority_sha256(request)?;
        if request_authority_sha256 != self.request_authority_sha256 {
            return Err("MiniMax H3 request changed after authenticated ingress".into());
        }
        if self.authenticated_identity_sha256.len() != 64
            || self.partition_identity_sha256 != private_ingress_partition_identity_sha256()
        {
            return Err("MiniMax H3 private ingress identity is malformed".into());
        }
        if self.policy_identity_sha256 != private_ingress_policy_identity_sha256() {
            return Err("MiniMax H3 private ingress policy identity changed".into());
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
    ) -> Result<Self, String> {
        self.validate_for_request(submitted)?;
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
        rebound.validate_for_request(resolved)?;
        Ok(rebound)
    }

    #[cfg(test)]
    fn authenticated_identity_sha256(&self) -> &str {
        &self.authenticated_identity_sha256
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

/// Classify the only private H3 HTTP partition before activation, artifact
/// discovery, path resolution, or scheduler mutation. Non-H3 requests return
/// `None` and retain the existing public ingress gates unchanged.
#[cfg(any(test, feature = "h3-private-uat"))]
#[cfg_attr(all(test, not(feature = "h3-private-uat")), allow(dead_code))]
pub(crate) fn classify_h3_private_ingress(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    classify_h3_private_ingress_with_runtime(
        request,
        authenticated,
        reviewed_h3_private_runtime_available,
    )
}

#[cfg(any(test, feature = "h3-private-uat"))]
fn classify_h3_private_ingress_with_runtime(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    runtime_available: impl FnOnce() -> bool,
) -> Result<Option<H3PrivateIngressGrant>, crate::routes::ApiError> {
    use axum::http::StatusCode;

    let Some(contract) = mold_core::minimax_h3::capability_contract_for_model(&request.model)
    else {
        return Ok(None);
    };
    let authenticated = authenticated.ok_or_else(|| {
        crate::routes::ApiError::with_code(
            "API key authentication is required for MiniMax H3 private generation",
            "UNAUTHORIZED",
            StatusCode::UNAUTHORIZED,
        )
    })?;

    let output_format = request
        .output_format
        .unwrap_or(mold_core::OutputFormat::Mp4);
    let exact_partition = request.model == contract.canonical_model
        && contract.canonical_model == mold_core::minimax_h3::FL2VA_COMFY
        && contract.task == mold_core::minimax_h3::Task::Fl2va
        && contract.layout == mold_core::minimax_h3::Layout::ComfyPrunedInt8ConvrotNvfp4Awq
        && request.batch_size == 1
        && output_format == mold_core::OutputFormat::Mp4
        && request.expand != Some(true)
        && request.upscale_model.is_none()
        && request.references.is_none();
    if !exact_partition {
        return Err(crate::routes::ApiError::with_code(
            "MiniMax H3 private ingress accepts only authenticated FL2VA Comfy batch-1 MP4 requests without expansion, upscaling, or ordered references",
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    if !runtime_available() {
        return Err(crate::routes::ApiError::with_code(
            "MiniMax H3 private runtime has no reviewed server admission implementation",
            H3_PRIVATE_RUNTIME_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }

    let partition_identity_sha256 = private_ingress_partition_identity_sha256();
    let request_authority_sha256 = request_authority_sha256(request).map_err(|error| {
        crate::routes::ApiError::with_code(
            error,
            H3_PRIVATE_PARTITION_REJECTED,
            StatusCode::UNPROCESSABLE_ENTITY,
        )
    })?;
    Ok(Some(H3PrivateIngressGrant {
        canonical_model: contract.canonical_model.to_string(),
        authenticated_identity_sha256: ingress_digest(
            b"mold.minimax-h3.private-authenticated-identity.v1\0",
            &[authenticated.identity.as_bytes()],
        ),
        partition_identity_sha256,
        policy_identity_sha256: private_ingress_policy_identity_sha256(),
        request_authority_sha256,
    }))
}

/// Fail closed until the inference facade contains at least one exact reviewed
/// private-runtime qualification record. This check performs no path access.
#[cfg(feature = "h3-private-uat")]
fn reviewed_h3_private_runtime_available() -> bool {
    mold_inference::reviewed_h3_private_runtime_available()
}

#[cfg(all(test, not(feature = "h3-private-uat")))]
#[allow(dead_code)]
fn reviewed_h3_private_runtime_available() -> bool {
    false
}

#[cfg(any(test, feature = "h3-private-uat"))]
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

#[cfg(any(test, feature = "h3-private-uat"))]
fn private_ingress_policy_identity_sha256() -> String {
    ingress_digest(
        b"mold.minimax-h3.private-ingress-policy.v1\0",
        &[
            mold_core::minimax_h3::FL2VA_COMFY.as_bytes(),
            b"fl2va",
            b"comfy-pruned-int8",
            b"batch-1",
            b"mp4",
            b"no-expand",
            b"no-upscale",
            b"no-references",
            b"api-key-authenticated",
        ],
    )
}

#[cfg(any(test, feature = "h3-private-uat"))]
fn private_ingress_partition_identity_sha256() -> String {
    ingress_digest(
        b"mold.minimax-h3.private-ingress-partition.v1\0",
        &[
            mold_core::minimax_h3::FL2VA_COMFY.as_bytes(),
            b"fl2va",
            b"comfy-pruned-int8",
            b"batch-1",
            b"mp4",
            b"no-expand",
            b"no-upscale",
            b"no-references",
        ],
    )
}

#[cfg(any(test, feature = "h3-private-uat"))]
fn request_authority_sha256(request: &mold_core::GenerateRequest) -> Result<String, String> {
    let serialized = serde_json::to_vec(request).map_err(|error| {
        format!("MiniMax H3 request authority could not be serialized: {error}")
    })?;
    Ok(ingress_digest(
        b"mold.minimax-h3.private-request-authority.v1\0",
        &[&serialized],
    ))
}

#[cfg(feature = "h3-private-uat")]
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

    #[cfg(feature = "h3-private-uat")]
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

#[cfg(feature = "h3-private-uat")]
#[derive(Clone, Debug)]
pub(crate) struct H3PrivateUatPathSet {
    pub(crate) models_root: std::path::PathBuf,
    pub(crate) staging_root: std::path::PathBuf,
    pub(crate) authorization_record: std::path::PathBuf,
    pub(crate) runtime_qualification_record: std::path::PathBuf,
}

#[cfg(feature = "h3-private-uat")]
impl H3PrivateUatPathSet {
    pub(crate) fn resolve(models_root: std::path::PathBuf) -> Self {
        let uat_root = std::env::var_os("MOLD_H3_PRIVATE_UAT_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                mold_core::Config::mold_dir()
                    .unwrap_or_else(|| models_root.clone())
                    .join("h3-private-uat")
            });
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

    pub(crate) fn inference_paths(&self) -> mold_inference::H3PrivateFl2VaUatPaths<'_> {
        mold_inference::H3PrivateFl2VaUatPaths {
            models_root: &self.models_root,
            staging_root: &self.staging_root,
            authorization_record: &self.authorization_record,
            runtime_qualification_record: &self.runtime_qualification_record,
        }
    }
}

#[cfg(feature = "h3-private-uat")]
pub(crate) struct InferenceH3PreparedAttempt {
    inner: mold_inference::H3PrivateFl2VaPreparedAttempt,
}

#[cfg(feature = "h3-private-uat")]
impl InferenceH3PreparedAttempt {
    pub(crate) fn boxed(
        inner: mold_inference::H3PrivateFl2VaPreparedAttempt,
    ) -> BoxedH3PreparedAttempt {
        Box::new(Self { inner })
    }
}

#[cfg(feature = "h3-private-uat")]
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

#[cfg(feature = "h3-private-uat")]
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

#[cfg(feature = "h3-private-uat")]
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
    }
}

/// Atomically prepare and attach one private attempt on the accepting GPU
/// owner. The scheduler only ever carried an empty slot; every opened file,
/// prepared tensor, and final attempt identity originates in the hidden
/// inference facade after the second plan fence.
#[cfg(feature = "h3-private-uat")]
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
    if mold_core::minimax_h3::task_for_model(&job.request.model)
        != Some(mold_core::minimax_h3::Task::Fl2va)
    {
        return Err("MiniMax H3 private UAT accepts only the sealed FL2VA partition".into());
    }
    let plan = job
        .execution_plan
        .as_ref()
        .ok_or_else(|| "MiniMax H3 private preparation lacked an execution plan".to_string())?;
    let frozen_factory = plan
        .engine_config
        .h3_factory_authority
        .as_ref()
        .ok_or_else(|| "MiniMax H3 private preparation lacked factory authority".to_string())?;
    let prepared_inputs = job.prepared_execution_inputs.as_ref().ok_or_else(|| {
        "MiniMax H3 private preparation lost its allocation-free admission evidence".to_string()
    })?;
    let ingress_grant = prepared_inputs
        .h3_private_ingress_grant
        .as_ref()
        .ok_or_else(|| "MiniMax H3 private preparation lost its ingress grant".to_string())?;
    ingress_grant.validate_for_request(&job.request)?;
    let compute_capability = worker.gpu.compute_capability.ok_or_else(|| {
        "MiniMax H3 private preparation requires exact CUDA compute capability".to_string()
    })?;
    let device_id = crate::scheduler::worker_device_id(worker);
    let admission_evidence = prepared_inputs
        .h3_private_admission_by_device
        .get(&device_id)
        .ok_or_else(|| {
            format!("MiniMax H3 private preparation has no admission evidence for '{device_id}'")
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
        return Err("MiniMax H3 private preparation no longer owns the accepted fence".into());
    }

    let prepared_route = prepared_inputs.by_device.get(&device_id).ok_or_else(|| {
        format!("MiniMax H3 private preparation has no prepared route for '{device_id}'")
    })?;
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
        return Err(
            "MiniMax H3 owner plan differs from immutable private admission evidence".into(),
        );
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
    progress.set_cancellation_token(cancellation);
    let prepared = mold_inference::prepare_h3_private_fl2va_attempt(
        mold_inference::H3PrivateFl2VaPrepareInput {
            request: &job.request,
            frozen_factory,
            admission_evidence,
            paths: paths.inference_paths(),
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
    let expected_mode = mold_core::minimax_h3::validate_request_contract(
        &job.request,
        mold_core::minimax_h3::Task::Fl2va,
    )
    .map_err(|error| format!("{}: {}", error.code, error.message))?;
    let expected_frames = job
        .request
        .frames
        .unwrap_or(mold_core::minimax_h3::MIN_FRAMES);
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
        || facts.media.canonical_model != mold_core::minimax_h3::FL2VA_COMFY
        || facts.media.task != mold_core::minimax_h3::Task::Fl2va
        || facts.media.mode != expected_mode
        || facts.media.width != job.request.width
        || facts.media.height != job.request.height
        || facts.media.frames != expected_frames
        || facts.media.fps != mold_core::minimax_h3::FIXED_FPS
        || job.request.seed != Some(facts.media.seed)
    {
        return Err("MiniMax H3 prepared attempt changed from the frozen owner admission".into());
    }
    job.h3_prepared_attempt = Some(boxed);
    Ok(())
}

#[cfg(feature = "h3-private-uat")]
pub(crate) fn private_prepare_error_message(
    error: mold_inference::H3PrivateFl2VaPrepareError,
) -> String {
    match error {
        mold_inference::H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification => {
            "MiniMax H3 private runtime has no reviewed runtime qualification".to_string()
        }
        mold_inference::H3PrivateFl2VaPrepareError::InvalidEvidence(reason) => {
            format!("MiniMax H3 private preparation evidence was rejected: {reason}")
        }
    }
}

#[cfg(feature = "h3-private-uat")]
fn private_uat_path(name: &str, fallback: std::path::PathBuf) -> std::path::PathBuf {
    std::env::var_os(name)
        .map(std::path::PathBuf::from)
        .unwrap_or(fallback)
}

#[cfg(any(test, feature = "h3-private-uat"))]
fn validate_private_h3_live_owner_route(
    expected_device_id: &str,
    expected_device_ordinal: usize,
    expected_compute_capability: (u16, u16),
    actual_device_id: &str,
    actual_device_ordinal: usize,
    actual_compute_capability: (u16, u16),
) -> Result<(), String> {
    if expected_device_id != actual_device_id
        || expected_device_ordinal != actual_device_ordinal
        || expected_compute_capability != actual_compute_capability
    {
        return Err(
            "MiniMax H3 CUDA route changed after allocation-free private admission".to_string(),
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "h3-private-uat"))]
mod tests {
    #[test]
    fn missing_reviewed_runtime_qualification_stays_a_typed_fail_closed_rejection() {
        assert_eq!(
            super::private_prepare_error_message(
                mold_inference::H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification,
            ),
            "MiniMax H3 private runtime has no reviewed runtime qualification",
        );
    }
}

#[cfg(test)]
mod structural_tests {
    use super::BoxedH3PreparedAttempt;

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
        }
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
            || {
                runtime_checked.set(true);
                true
            },
        )
        .expect("non-H3 classification must remain a no-op");

        assert!(grant.is_none());
        assert!(!runtime_checked.get());
    }

    #[test]
    fn exact_h3_partition_requires_api_key_authentication_before_runtime_lookup() {
        use axum::response::IntoResponse;

        let runtime_checked = std::cell::Cell::new(false);
        let error = super::classify_h3_private_ingress_with_runtime(
            &request(mold_core::minimax_h3::FL2VA_COMFY),
            None,
            || {
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
    fn ingress_grant_rebinds_only_an_inference_resolved_seed() {
        let submitted = request(mold_core::minimax_h3::FL2VA_COMFY);
        assert!(submitted.seed.is_none());
        let grant = super::classify_h3_private_ingress_with_runtime(
            &submitted,
            Some(&authenticated()),
            || true,
        )
        .expect("exact authenticated partition")
        .expect("private grant");

        let mut resolved = submitted.clone();
        resolved.seed = Some(42);
        let rebound = grant
            .rebind_resolved_request(&submitted, &resolved)
            .expect("one omitted seed may be resolved once");
        assert!(grant.validate_for_request(&resolved).is_err());
        rebound
            .validate_for_request(&resolved)
            .expect("rebound grant must bind the exact resolved request");

        let mut mutated = resolved.clone();
        mutated.prompt.push_str(" changed");
        assert!(rebound.validate_for_request(&mutated).is_err());
        assert!(grant.rebind_resolved_request(&submitted, &mutated).is_err());
    }

    #[test]
    fn wrong_h3_partitions_are_rejected_before_runtime_or_artifact_work() {
        let auth = authenticated();
        for mut invalid in [
            request(mold_core::minimax_h3::REF2VA_COMFY),
            request(mold_core::minimax_h3::FL2VA_OFFICIAL),
        ] {
            let runtime_checked = std::cell::Cell::new(false);
            let error =
                super::classify_h3_private_ingress_with_runtime(&invalid, Some(&auth), || {
                    runtime_checked.set(true);
                    true
                })
                .expect_err("wrong H3 task/layout must fail closed");
            assert_eq!(error.code, super::H3_PRIVATE_PARTITION_REJECTED);
            assert!(!runtime_checked.get());

            invalid.model = mold_core::minimax_h3::FL2VA_COMFY.to_string();
            invalid.batch_size = 2;
            let error =
                super::classify_h3_private_ingress_with_runtime(&invalid, Some(&auth), || {
                    runtime_checked.set(true);
                    true
                })
                .expect_err("batch H3 ingress must fail closed");
            assert_eq!(error.code, super::H3_PRIVATE_PARTITION_REJECTED);
            assert!(!runtime_checked.get());
        }
    }

    #[test]
    fn accepted_ingress_grant_is_cloneable_and_payload_free() {
        let auth = authenticated();
        let request = request(mold_core::minimax_h3::FL2VA_COMFY);
        let grant = super::classify_h3_private_ingress_with_runtime(&request, Some(&auth), || true)
            .expect("exact private partition must classify")
            .expect("exact private partition must return a grant");
        let cloned = grant.clone();

        assert_eq!(cloned.canonical_model(), mold_core::minimax_h3::FL2VA_COMFY);
        assert_eq!(cloned.authenticated_identity_sha256().len(), 64);
        assert_eq!(cloned.partition_identity_sha256().len(), 64);
        assert_eq!(cloned.policy_identity_sha256().len(), 64);
        assert_eq!(cloned.request_authority_sha256().len(), 64);
        cloned
            .validate_for_request(&request)
            .expect("unmodified canonical request must retain its grant");
        let mut mutated = request.clone();
        mutated.prompt.push_str(" changed");
        assert!(cloned.validate_for_request(&mutated).is_err());
        let debug = format!("{cloned:?}");
        assert!(!debug.contains(&auth.identity));
        assert!(!debug.contains(&request.prompt));
    }

    #[test]
    fn private_owner_route_rejects_changed_compute_capability() {
        super::validate_private_h3_live_owner_route("cuda:0", 0, (8, 9), "cuda:0", 0, (8, 9))
            .expect("unchanged live route must validate");
        let error =
            super::validate_private_h3_live_owner_route("cuda:0", 0, (8, 9), "cuda:0", 0, (9, 0))
                .expect_err("changed compute capability must fail the second owner fence");
        assert!(error.contains("CUDA route changed"));
    }
}
