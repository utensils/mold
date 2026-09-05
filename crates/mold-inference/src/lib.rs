pub(crate) mod adaptive_offload;
pub mod artifact_format;
pub mod attention;
pub mod audio;
pub mod av_media;
mod batch;
mod cache;
pub(crate) mod cfg_plus_ddim;
pub mod chain;
pub mod controlnet;
pub mod conv_policy;
pub mod device;
mod encoders;
pub mod engine;
pub(crate) mod engine_base;
pub mod error;
#[cfg(feature = "expand")]
pub mod expand;
mod factory;
pub mod flux;
pub mod flux2;
mod h3_factory;
/// Hunyuan3D 2.0 image-to-3D shape generation (#1495).
pub mod hunyuan3d;
/// PuLID face detection and identity embedding (#1222). Gated on the
/// `pulid` feature, which is what pulls `candle-onnx` into the build.
#[cfg(feature = "pulid")]
pub mod identity;
mod image;
pub(crate) mod img2img;
pub mod img_utils;
pub mod latent_preview;
pub mod loader;
pub mod ltx2;
pub mod ltx_video;
pub mod metal_memory;
/// Reductions on shapes candle's Metal backend gets right.
///
/// The implementation lives in `mold-candle` because the H3 audio VAE needs it
/// too and `mold-candle` cannot depend on this crate. This alias keeps every
/// existing `crate::metal_reduce::…` call site pointing at that one authority.
pub(crate) use mold_candle::metal_reduce;
// The deterministic scheduler contract lands ahead of the license-gated H3
// engine. Keeping these primitives crate-private avoids advertising a runnable
// family while leaving exact math and conditioner lifecycle ready for the
// future pipeline to consume.
#[allow(dead_code)]
pub(crate) mod minimax_h3;
/// Release every cached MiniMax H3 conditioner output, reporting the host
/// bytes handed back. The server's reclaim paths call this; see
/// `minimax_h3::conditioner_cache` for the reclaim contract. It is the whole
/// exported surface of that cache — the bytes it reports ARE what the map held.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub use minimax_h3::conditioner_cache::h3_conditioner_cache_clear;
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub use minimax_h3::pipeline::placeholder_endpoint_png as h3_placeholder_endpoint_png;
#[cfg(any(all(feature = "h3", feature = "dev-bins"), feature = "h3-private-uat"))]
pub use minimax_h3::private_qualification;
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[doc(hidden)]
pub use minimax_h3::private_qwen::{
    released_h3_private_qwen_loader_memory_authority, released_h3_private_qwen_output_tensor_bytes,
    validate_h3_private_qwen_loader_memory_authority, H3PrivateQwenLoaderMemoryAuthority,
    H3PrivateQwenLoaderMemoryRoute,
};
#[cfg(any(all(feature = "h3", feature = "dev-bins"), feature = "h3-private-uat"))]
#[doc(hidden)]
pub use minimax_h3::private_runtime_qualification::{
    produce_h3_private_runtime_qualification_candidate, H3PrivateRuntimeQualificationCandidate,
    H3_PRIVATE_RUNTIME_RECORD_PRODUCER_MARKER,
};
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[doc(hidden)]
pub const fn h3_private_runtime_code_identity_sha256() -> &'static str {
    minimax_h3::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256
}
#[cfg(feature = "h3")]
#[doc(hidden)]
pub use minimax_h3::private_server::authenticate_h3_public_presentation;
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[doc(hidden)]
pub use minimax_h3::private_server::{
    authenticate_h3_private_presentation, authenticate_h3_private_runtime_qualification,
    prepare_h3_private_fl2va_admission, prepare_h3_private_fl2va_attempt,
    reviewed_h3_private_runtime_available, reviewed_h3_private_runtime_available_for_task,
    H3PrivateAllocationCommit, H3PrivateDeviceHeadroomShortfall, H3PrivateFl2VaAdmissionEvidence,
    H3PrivateFl2VaAdmissionInput, H3PrivateFl2VaAttemptFacts, H3PrivateFl2VaMediaContract,
    H3PrivateFl2VaOwnerFenceFacts, H3PrivateFl2VaPrepareError, H3PrivateFl2VaPrepareInput,
    H3PrivateFl2VaPreparedAttempt, H3PrivateFl2VaRunContext, H3PrivateFl2VaRunOutput,
    H3PrivateFl2VaRuntimeBounds, H3PrivateFl2VaTerminalIdentityEcho, H3PrivateFl2VaUatPaths,
    H3PrivateHostHeadroomShortfall, H3PrivatePresentationAuthority, H3PrivatePresentationRoute,
    H3PrivateRuntimeQualificationAuthority, H3PrivateSchedulerLedgerIdentity,
};
pub mod model_registry;
pub(crate) mod nvfp4;
pub(crate) mod playground_edm;
pub mod progress;
#[cfg(test)]
pub(crate) mod pulid_fixtures;
#[cfg(test)]
pub(crate) mod test_support;

pub(crate) mod quantized_dmmv;
pub(crate) mod quantized_linear;
pub mod qwen_image;
pub(crate) mod reference_media;
pub mod runtime_env;
pub mod scheduler;
pub mod sd15;
pub mod sd3;
pub mod sdxl;
pub mod shared_pool;
pub mod upscaler;
pub mod vae_tiling;
// The T2V engine is live, but the family ships in layers: the 14B/A14B and
// I2V configs, the diffusers-format rename table, and several accessors are
// built and tested here ahead of the layers that consume them.
#[allow(dead_code)]
pub mod wan;
pub use wan::pipeline::distill_is_active as wan_distill_is_active;
/// Narrow public surface of the wan module: the checkpoint-header
/// source-image classification `/api/models` advertises (#772).
pub use wan::pipeline::source_image_capability as wan_source_image_capability;
pub use wan::step_cache::requested_threshold as wan_requested_step_cache_threshold;
pub(crate) mod weight_loader;
pub mod wuerstchen;
pub mod zimage;

pub use batch::{
    batch_execution_capability_for_family, production_batch_capabilities,
    production_family_capabilities, production_family_capability_for_family,
    validate_runtime_batch_capability, BackendApplicability, BackendQualification,
    ComponentPlacementCapability, DeterminismGuarantee, FamilyBatchCapability, MediaKind,
    QualificationReference, SeedContract, TiledVaeCapability, WorkflowCapabilities,
};
pub use engine::{
    with_inference_cancellation, BatchExecutionCapability, GenerationReferenceBinding,
    InferenceEngine, LoadStrategy,
};
pub use error::InferenceError;
pub use factory::{
    create_engine, create_engine_with_frozen_config, create_engine_with_pool,
    factory_family_availability, FactoryFamilyAvailability, FrozenEngineConfig,
};
pub use flux::FluxEngine;
pub use flux2::Flux2Engine;
pub use h3_factory::{
    expected_h3_factory_prepared_attempt_identity, expected_h3_factory_prepared_request_identity,
    expected_h3_factory_raw_checkpoint_identity, expected_h3_factory_target_budget_identity,
    h3_factory_activation_prerequisites, media_model_matches_h3_authority,
    FrozenH3FactoryAuthority, H3FactoryActivationPrerequisite, H3FactoryArtifactHostInput,
    H3FactoryArtifactHostRole, H3FactoryAttentionInput, H3FactoryAuthorityInput,
    H3FactoryBlockMemoryInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
    H3FactoryConditionerPlacement, H3FactoryEndpointAnchor, H3FactoryEndpointInput,
    H3FactoryEndpointPreprocess, H3FactoryExecutionBudgetEchoInput, H3FactoryPreparedAttemptInput,
    H3FactoryPreparedRequestInput, H3FactoryPreparedRowsInput, H3FactoryQuantizationAuthority,
    H3FactoryRawCheckpointInput, H3FactoryTargetBudgetInput, H3FactoryTargetDenoiseCopyPolicy,
    H3FactoryTargetLoadDropPolicy,
};
pub use ltx2::Ltx2Engine;
pub use ltx_video::LtxVideoEngine;
pub use model_registry::known_models;
pub use mold_candle::minimax_h3::{
    H3AttentionActivation, H3AttentionBackend, H3AttentionDType, H3AttentionDevice,
    H3AttentionKernel, H3AttentionModelContract,
};
pub use progress::{
    is_inference_cancelled, InferenceCancellationToken, InferenceCancelled, ProgressEvent,
    ProgressPhase,
};
pub use qwen_image::QwenImageEngine;
pub use sd15::SD15Engine;
pub use sd3::SD3Engine;
pub use sdxl::SDXLEngine;
pub use upscaler::{create_upscale_engine, with_upscale_cancellation, UpscaleEngine};
pub use wuerstchen::WuerstchenEngine;
pub use zimage::ZImageEngine;

/// Compile-time proof of whether either non-shipping FlashAttention feature
/// reached the Mold-owned Candle dependency graph.
///
/// Release entry points consume this through `black_box` so the selected
/// marker remains inspectable after LTO and stripping.
#[inline(never)]
pub fn h3_attention_release_provenance_marker() -> &'static str {
    std::hint::black_box(mold_candle::minimax_h3::h3_attention_release_provenance_marker())
}

/// Compile-time acceleration backend label for provenance columns
/// (gallery DB rows, embedded metadata). Lives here because the
/// `cuda`/`metal` features are defined on this crate and forwarded by
/// every binary, so feature unification guarantees one answer per build.
pub fn compiled_backend_label() -> &'static str {
    std::hint::black_box(h3_attention_release_provenance_marker());
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    }
}

#[cfg(test)]
mod backend_label_tests {
    #[test]
    fn backend_label_is_a_known_backend() {
        assert!(["cuda", "metal", "cpu"].contains(&super::compiled_backend_label()));
    }

    #[test]
    #[cfg(not(any(feature = "flash-attn", feature = "h3-attention-rc")))]
    fn ordinary_build_reports_both_attention_features_omitted() {
        assert_eq!(
            super::h3_attention_release_provenance_marker(),
            "mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted"
        );
    }
}
