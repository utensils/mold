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
mod image;
pub(crate) mod img2img;
pub mod img_utils;
pub mod latent_preview;
pub mod loader;
pub mod ltx2;
pub mod ltx_video;
// The deterministic scheduler contract lands ahead of the license-gated H3
// engine. Keeping these primitives crate-private avoids advertising a runnable
// family while leaving exact math and conditioner lifecycle ready for the
// future pipeline to consume.
#[allow(dead_code)]
pub(crate) mod minimax_h3;
pub mod model_registry;
pub(crate) mod nvfp4;
pub mod progress;
pub mod qwen_image;
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
pub(crate) mod wan;
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
    with_inference_cancellation, BatchExecutionCapability, InferenceEngine, LoadStrategy,
};
pub use error::InferenceError;
pub use factory::{
    create_engine, create_engine_with_frozen_config, create_engine_with_pool,
    factory_family_availability, FactoryFamilyAvailability, FrozenEngineConfig,
};
pub use flux::FluxEngine;
pub use flux2::Flux2Engine;
pub use ltx2::Ltx2Engine;
pub use ltx_video::LtxVideoEngine;
pub use model_registry::known_models;
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

/// Compile-time acceleration backend label for provenance columns
/// (gallery DB rows, embedded metadata). Lives here because the
/// `cuda`/`metal` features are defined on this crate and forwarded by
/// every binary, so feature unification guarantees one answer per build.
pub fn compiled_backend_label() -> &'static str {
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
}
