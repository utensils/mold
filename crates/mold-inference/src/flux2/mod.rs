pub(crate) mod lora;
pub(crate) mod pipeline;
pub(crate) mod quantized_transformer;
pub(crate) mod sampling;
pub mod single_file;
pub mod text_encoder_residency;
pub(crate) mod transformer;
pub(crate) mod vae;

pub use pipeline::Flux2Engine;
pub use single_file::{detect_format, Flux2SingleFileFormat};
/// The CFG budget gate and the activation model it is charged with are
/// re-exported because the server's execution plan has to reach the same
/// verdict the engine does. Both are pure functions, so the planner asking
/// THEM — rather than re-deriving the arithmetic from a different estimator —
/// is what keeps the recorded execution class and the executed render from
/// disagreeing.
pub use transformer::{
    flux2_activation_bytes_for, flux2_cfg_batching, Flux2CfgBatching, Flux2Config,
};

/// The batch-2 activation term a PLAN charges the FLUX.2 CFG budget gate with.
///
/// `flux2::pipeline::resolve_cfg_batching` charges
/// [`flux2_activation_bytes_for`] at `batch = 2` over the render's own token
/// count, and this is the same call with the same model — asked before a
/// prompt has been encoded or a device leased, which is the only difference
/// between the two sites:
///
/// * TEXT rows are the fixed `FLUX2_KLEIN_MAX_LENGTH`. Every Klein prompt is
///   truncated and right-padded to that width before it leaves
///   `encoders::qwen3`, so the planner knows the number exactly rather than
///   estimating it, and the engine's length gate is always satisfied.
/// * IMAGE rows are [`crate::device::flux_token_count`], the one authority for
///   "how many tokens does this canvas pack into", which the engine's own
///   residency budget already uses. References are deliberately NOT counted:
///   they are concatenated inside `denoise`, after the point at which the
///   engine samples `state.img.dim(1)` for this same estimate, so counting
///   them here would make the plan charge more than the render does.
/// * The WIDTH is bf16's. A FLUX.2 render reaching a batched CFG step is on
///   CUDA by construction (the engine reads its VRAM total through a CUDA
///   ordinal and answers `Sequential` for every other device location), and
///   the model reads nothing from the dtype but its byte width, which f16 and
///   bf16 share.
/// * The BACKEND is [`crate::device::flux_effective_attention_backend`], the
///   device-blind mirror of the engine's own `effective_backend_under` that
///   the FLUX.2 residency budget is already priced with.
///
/// Charging a DIFFERENT estimator here — an area model, say — is the failure
/// this function exists to prevent: the gate is a comparison, and two sides
/// that share the comparison but not its inputs still disagree on every card
/// whose total falls between their two answers.
pub fn flux2_cfg_plan_activation_bytes(cfg: &Flux2Config, width: u32, height: u32) -> u64 {
    let img_tokens = crate::device::flux_token_count(width, height);
    let tokens = (crate::encoders::qwen3::FLUX2_KLEIN_MAX_LENGTH as u64).saturating_add(img_tokens);
    flux2_activation_bytes_for(
        cfg,
        usize::try_from(tokens).unwrap_or(usize::MAX),
        2,
        candle_core::DType::BF16,
        crate::device::flux_effective_attention_backend(),
    )
}

/// The transformer geometry a plan charges for an undistilled FLUX.2 [klein]
/// base checkpoint, or `None` for a name that is not one.
///
/// This is the engine's own name fallback
/// (`Flux2Engine::resolve_config`), restricted to the names that can reach it.
/// A base tier is a `klein-base` name by
/// `mold_core::validation::is_flux2_base_model`, so the engine's earlier arms
/// — `flux2-dev`, and the plain `klein`/GGUF default — resolve the same two
/// configurations this does, on the same `9b` test.
///
/// The engine prefers the checkpoint's own header where it can read one
/// (`single_file::detect_hidden_size`), and this does not: a plan may be
/// resolved before a byte has landed. For every published base tier the two
/// agree by construction — `flux2-klein-base-9b:*` is the 4096-wide
/// checkpoint and `flux2-klein-base:*` the 3072-wide one — and
/// `every_published_klein_base_tier_resolves_its_own_geometry` pins that. A
/// checkpoint whose name contradicts its header is the one divergence, and it
/// is a mislabeled file rather than a shape this has to model.
pub fn flux2_base_tier_config(model_name: &str) -> Option<Flux2Config> {
    if !mold_core::validation::is_flux2_base_model(model_name) {
        return None;
    }
    Some(if model_name.to_ascii_lowercase().contains("9b") {
        Flux2Config::klein_9b()
    } else {
        Flux2Config::klein()
    })
}

#[cfg(test)]
mod plan_budget_tests {
    use super::*;

    /// The plan's activation term IS the engine's, at the engine's own
    /// arguments. Written as a differential rather than a golden number so it
    /// keeps holding when the activation model is re-fitted.
    #[test]
    fn the_plan_activation_term_is_the_engines_own_call() {
        for cfg in [
            Flux2Config::klein(),
            Flux2Config::klein_9b(),
            Flux2Config::dev(),
        ] {
            for (width, height) in [(512, 512), (1024, 1024), (1536, 1024), (2048, 2048)] {
                let tokens = crate::encoders::qwen3::FLUX2_KLEIN_MAX_LENGTH
                    + (width as usize / 16) * (height as usize / 16);
                assert_eq!(
                    flux2_cfg_plan_activation_bytes(&cfg, width, height),
                    flux2_activation_bytes_for(
                        &cfg,
                        tokens,
                        2,
                        candle_core::DType::BF16,
                        crate::device::flux_effective_attention_backend(),
                    ),
                    "{width}x{height} at hidden {}",
                    cfg.hidden_size
                );
            }
        }
    }

    /// Every published base tier resolves the geometry its checkpoint
    /// actually has, which is what lets the plan skip the header read the
    /// engine performs.
    #[test]
    fn every_published_klein_base_tier_resolves_its_own_geometry() {
        let mut seen = 0;
        for manifest in mold_core::manifest::known_manifests() {
            let Some(config) = flux2_base_tier_config(&manifest.name) else {
                continue;
            };
            seen += 1;
            let expected = if manifest.name.contains("9b") {
                4096
            } else {
                3072
            };
            assert_eq!(
                config.hidden_size, expected,
                "{} must charge its own transformer width",
                manifest.name
            );
        }
        assert!(seen >= 2, "both base tiers must be covered, saw {seen}");
        // A distilled tier, [dev], and an opaque catalog id are not base
        // checkpoints and never reach the budget at all.
        for name in [
            "flux2-klein:q8",
            "flux2-klein-9b:bf16",
            "flux2-dev:q8",
            "cv:3143864",
        ] {
            assert!(flux2_base_tier_config(name).is_none(), "{name}");
        }
    }
}
