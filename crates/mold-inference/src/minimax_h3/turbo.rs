//! Reviewed MiniMax H3 Turbo tier contracts and their runtime resolution.
//!
//! A Turbo tier is a LoRA adapter overlaid on the *same* compact INT8 ConvRot
//! checkpoint, so nothing about the base artifact contract relaxes. What a tier
//! adds is a distillation triple — sampler kind, terminal-inclusive step count,
//! and video shift — plus the adapter's own authenticated identity and its
//! resident cost.
//!
//! Selection is by model identity: the reviewed Turbo manifest tags
//! (`mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS`) name a tier and
//! pin its adapter file inside the model's own manifest, together with the
//! repository and revision that tier's adapter is published at — Comfy-Org's
//! `loras/` re-hosts or lightx2v's repository root. The historical
//! `MOLD_H3_TURBO_ADAPTER` (path) + `MOLD_H3_TURBO_TIER` (tier id) pair
//! remains a capture-scope UAT override honored only under the
//! `h3-private-uat` feature; ordinary builds refuse a set pair outright so
//! there is never a second, contradictory selection authority. Both variables
//! stay registered engine-shaping variables.

use anyhow::{anyhow, bail, Context, Result};
use mold_candle::minimax_h3::{
    authenticate_h3_turbo_lora_adapter, H3ComfyNeverCancel, H3TurboLoraContract,
    H3TurboLoraRuntime, H3TurboLoraTier,
};

use crate::h3_factory::H3FactoryTurboAdapterAuthority;

use super::sampler::{H3SamplerKind, H3_TURBO_768P_VIDEO_SHIFT, H3_VIDEO_SHIFT};

/// Environment variable naming the reviewed adapter file.
pub(crate) const TURBO_ADAPTER_PATH_VARIABLE: &str = "MOLD_H3_TURBO_ADAPTER";
/// Environment variable naming which reviewed tier that file must be.
pub(crate) const TURBO_ADAPTER_TIER_VARIABLE: &str = "MOLD_H3_TURBO_TIER";

/// The distillation contract of one reviewed tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct H3TurboTierContract {
    pub(crate) tier: H3TurboLoraTier,
    /// Terminal-inclusive grid points: `evaluations + 1`. A "Turbo 8-step"
    /// checkpoint is therefore 9 mold steps and exactly 8 forwards.
    pub(crate) grid_points: u32,
    pub(crate) sampler_kind: H3SamplerKind,
    pub(crate) video_shift: f32,
}

/// Every reviewed tier's contract.
///
/// All five tiers select `KSamplerSelect: euler` over Comfy's
/// `BasicScheduler("simple")` grid in their published reference workflows. Only
/// the 768p-trained tiers move the video shift, which the 4-step tier's own
/// Diffusers documentation passes as `--video-shift 6` and which every
/// LightX2V 768p config repeats.
pub(crate) const REVIEWED_TURBO_TIERS: &[H3TurboTierContract] = &[
    H3TurboTierContract {
        tier: H3TurboLoraTier::Fl2v8StepV10,
        grid_points: 9,
        sampler_kind: H3SamplerKind::ComfyEuler,
        video_shift: H3_VIDEO_SHIFT,
    },
    H3TurboTierContract {
        tier: H3TurboLoraTier::Fl2v768p4StepV10,
        grid_points: 5,
        sampler_kind: H3SamplerKind::ComfyEuler,
        video_shift: H3_TURBO_768P_VIDEO_SHIFT,
    },
    H3TurboTierContract {
        tier: H3TurboLoraTier::Ref2v4StepV10,
        grid_points: 5,
        sampler_kind: H3SamplerKind::ComfyEuler,
        video_shift: H3_VIDEO_SHIFT,
    },
    H3TurboTierContract {
        tier: H3TurboLoraTier::Fl2v768p4StepV11,
        grid_points: 5,
        sampler_kind: H3SamplerKind::ComfyEuler,
        video_shift: H3_TURBO_768P_VIDEO_SHIFT,
    },
    H3TurboTierContract {
        tier: H3TurboLoraTier::Fl2v768p8StepV10,
        grid_points: 9,
        sampler_kind: H3SamplerKind::ComfyEuler,
        video_shift: H3_TURBO_768P_VIDEO_SHIFT,
    },
];

/// Look one contract up by its stable id. This is the lookup the frozen
/// factory authority uses, so the authority and the runtime can never disagree
/// about a tier's distillation triple.
pub(crate) fn reviewed_contract_for_stable_id(
    tier_stable_id: &str,
) -> Option<&'static H3TurboTierContract> {
    let trimmed = tier_stable_id.trim();
    REVIEWED_TURBO_TIERS
        .iter()
        .find(|contract| contract.tier.stable_id() == trimmed)
}

pub(crate) fn turbo_tier_contract(tier: H3TurboLoraTier) -> Result<&'static H3TurboTierContract> {
    REVIEWED_TURBO_TIERS
        .iter()
        .find(|contract| contract.tier == tier)
        .ok_or_else(|| anyhow!("MiniMax H3 Turbo tier {tier:?} has no reviewed contract"))
}

/// Resolve a tier from its stable id or from a short alias.
///
/// The stable id is the durable form; the short aliases exist only so the
/// interim environment knob is usable by hand.
pub(crate) fn parse_turbo_tier(value: &str) -> Result<H3TurboLoraTier> {
    let trimmed = value.trim();
    let matched = H3TurboLoraTier::ALL.into_iter().find(|tier| {
        tier.stable_id().eq_ignore_ascii_case(trimmed)
            || short_tier_alias(*tier).eq_ignore_ascii_case(trimmed)
    });
    matched.ok_or_else(|| {
        anyhow!(
            "MiniMax H3 Turbo tier {value:?} is not reviewed; expected one of {}",
            H3TurboLoraTier::ALL
                .into_iter()
                .map(short_tier_alias)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

pub(crate) const fn short_tier_alias(tier: H3TurboLoraTier) -> &'static str {
    match tier {
        H3TurboLoraTier::Fl2v8StepV10 => "fl2v-8step",
        H3TurboLoraTier::Fl2v768p4StepV10 => "fl2v-4step-768p",
        H3TurboLoraTier::Ref2v4StepV10 => "ref2v-4step",
        H3TurboLoraTier::Fl2v768p4StepV11 => "fl2v-4step-768p-v1.1",
        H3TurboLoraTier::Fl2v768p8StepV10 => "fl2v-8step-768p",
    }
}

/// Resolve the manifest-selected Turbo tier and adapter path for a model
/// identity, if the identity is a reviewed Turbo tag.
///
/// The adapter path is derived from the tag's own manifest through
/// `storage_path`, so admission, `mold pull`, and repair all agree on the one
/// shared on-disk copy.
pub(crate) fn manifest_turbo_selection(
    model: &str,
    models_root: &std::path::Path,
) -> Result<Option<(std::path::PathBuf, H3TurboLoraTier)>> {
    let Some(manifest_tier) = mold_core::minimax_h3::turbo_tier_for_model(model) else {
        return Ok(None);
    };
    let tier = parse_turbo_tier(manifest_tier.tier_stable_id)?;
    Ok(Some((
        manifest_adapter_path(models_root, manifest_tier)?,
        tier,
    )))
}

/// The manifest-pinned on-disk location of one reviewed tier's adapter.
fn manifest_adapter_path(
    models_root: &std::path::Path,
    manifest_tier: &mold_core::minimax_h3::TurboManifestTier,
) -> Result<std::path::PathBuf> {
    let manifest = mold_core::manifest::find_manifest(manifest_tier.model).ok_or_else(|| {
        anyhow!(
            "MiniMax H3 Turbo model {} has no registered manifest",
            manifest_tier.model
        )
    })?;
    let adapter = manifest
        .files
        .iter()
        .find(|file| file.component == mold_core::manifest::ModelComponent::DistilledLora)
        .ok_or_else(|| {
            anyhow!(
                "MiniMax H3 Turbo manifest {} pins no adapter file",
                manifest_tier.model
            )
        })?;
    Ok(models_root.join(mold_core::manifest::storage_path(manifest, adapter)))
}

/// The one Turbo selection rule shared by admission and the transformer load.
///
/// - A reviewed Turbo model tag selects its manifest-pinned adapter.
/// - The `MOLD_H3_TURBO_ADAPTER`/`MOLD_H3_TURBO_TIER` pair is a capture-scope
///   UAT override: under the `h3-private-uat` feature it wins over (or, for a
///   base model, supplies) the selection. In every other build a set pair is
///   a hard error — never silently ignored — because two selection
///   authorities that can disagree must not both stay live.
pub(crate) fn resolve_turbo_selection(
    model: &str,
    models_root: &std::path::Path,
) -> Result<Option<(std::path::PathBuf, H3TurboLoraTier)>> {
    let manifest_selection = manifest_turbo_selection(model, models_root)?;
    let env_selection = requested_turbo_selection()?;
    match (manifest_selection, env_selection) {
        (selection, None) => Ok(selection),
        (_, Some(env)) if cfg!(feature = "h3-private-uat") => Ok(Some(env)),
        (Some(_), Some(_)) => bail!(
            "{TURBO_ADAPTER_PATH_VARIABLE}/{TURBO_ADAPTER_TIER_VARIABLE} contradict the \
             manifest-selected Turbo model {model}; unset the environment pair — it is a \
             capture-scope UAT override honored only under the h3-private-uat feature"
        ),
        // The tag list is formatted from the manifest tier table rather than
        // written out, so a new tier can never leave this sentence stale (it
        // already omitted the Ref2VA tag for a whole release).
        (None, Some(_)) => bail!(
            "{TURBO_ADAPTER_PATH_VARIABLE}/{TURBO_ADAPTER_TIER_VARIABLE} are a capture-scope \
             UAT override honored only under the h3-private-uat feature; select a reviewed \
             Turbo model tag ({}) instead",
            mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS
                .iter()
                .map(|tier| tier.model)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The selection the environment currently requests, if any.
///
/// Both variables must be present together: a path with no tier cannot be
/// authenticated against a pinned digest, and a tier with no path has nothing
/// to authenticate.
pub(crate) fn requested_turbo_selection() -> Result<Option<(std::path::PathBuf, H3TurboLoraTier)>> {
    let path = crate::runtime_env::value(TURBO_ADAPTER_PATH_VARIABLE)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let tier = crate::runtime_env::value(TURBO_ADAPTER_TIER_VARIABLE)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    match (path, tier) {
        (None, None) => Ok(None),
        (Some(path), Some(tier)) => Ok(Some((path.into(), parse_turbo_tier(&tier)?))),
        (Some(_), None) => bail!(
            "{TURBO_ADAPTER_PATH_VARIABLE} requires {TURBO_ADAPTER_TIER_VARIABLE}; a Turbo adapter is only admissible against a reviewed tier's pinned digest"
        ),
        (None, Some(_)) => bail!(
            "{TURBO_ADAPTER_TIER_VARIABLE} requires {TURBO_ADAPTER_PATH_VARIABLE}; there is no adapter file to authenticate"
        ),
    }
}

/// Build the factory authority for an authenticated adapter contract.
///
/// Only the file's own facts are supplied; the distillation triple comes from
/// the reviewed tier table inside the constructor.
pub(crate) fn turbo_adapter_authority(
    contract: &H3TurboLoraContract,
    resident_device_bytes: u64,
    device_staging_peak_bytes: u64,
    host_staging_peak_bytes: u64,
) -> Result<H3FactoryTurboAdapterAuthority> {
    H3FactoryTurboAdapterAuthority::for_reviewed_tier(
        contract.tier().stable_id(),
        contract.adapter_identity_sha256(),
        contract.content_sha256(),
        resident_device_bytes,
        device_staging_peak_bytes,
        host_staging_peak_bytes,
    )
}

/// Authenticate the requested adapter at admission and describe it.
///
/// This reads and digests the whole file but allocates no device memory: the
/// deltas are materialized later, in the transformer-load phase, which is where
/// the budget charges them. The resident and staging figures are derived from
/// the validated structure rather than assumed, so the budget term and the
/// eventual allocation agree.
pub(crate) fn resolve_turbo_authority_for_request(
    model: &str,
    models_root: &std::path::Path,
) -> Result<Option<H3FactoryTurboAdapterAuthority>> {
    let Some((path, tier)) = resolve_turbo_selection(model, models_root)? else {
        return Ok(None);
    };
    let contract = authenticate_h3_turbo_lora_adapter(&path, tier, &H3ComfyNeverCancel)
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| {
            format!(
                "failed to authenticate the reviewed MiniMax H3 Turbo adapter at {}",
                path.display()
            )
        })?;
    let inspection = contract.inspection();
    let mut resident_device_bytes = 0u64;
    let mut widest_matrix_bytes = 0u64;
    let mut widest_module_bytes = 0u64;
    for module in inspection.modules.values() {
        let mut module_bytes = 0u64;
        for reference in [&module.lora_a, &module.lora_b] {
            let bytes = reference.data_offsets[1] - reference.data_offsets[0];
            resident_device_bytes = resident_device_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("MiniMax H3 Turbo resident byte count overflows"))?;
            widest_matrix_bytes = widest_matrix_bytes.max(bytes);
            module_bytes = module_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("MiniMax H3 Turbo module byte count overflows"))?;
        }
        widest_module_bytes = widest_module_bytes.max(module_bytes);
    }
    let host_staging_peak_bytes = widest_matrix_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow!("MiniMax H3 Turbo host staging byte count overflows"))?;
    turbo_adapter_authority(
        &contract,
        resident_device_bytes,
        // The widest module's transposed copies live beside their originals
        // during the upload; see `H3TurboLoraRuntime::device_staging_peak_bytes`.
        widest_module_bytes,
        host_staging_peak_bytes,
    )
    .map(Some)
}

/// Load the reviewed adapter's deltas onto the execution device.
///
/// The file is authenticated a second time here, on purpose: admission's digest
/// proves what was *offered*, and this one proves what is actually being read
/// into the transformer. Between the two the file could have been replaced, and
/// the deltas must come from the exact descriptor whose digest was verified.
pub(crate) fn load_reviewed_turbo_runtime(
    authority: &H3FactoryTurboAdapterAuthority,
    models_root: &std::path::Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    cancellation: std::sync::Arc<dyn mold_candle::minimax_h3::H3ComfyInt8Cancellation>,
) -> Result<H3TurboLoraRuntime> {
    let tier = parse_turbo_tier(authority.tier_stable_id())?;
    // Re-resolve the selection with the same rule admission used, keyed by the
    // frozen tier. A UAT env override must still name the admitted tier; an
    // env pair set in an ordinary build stays a hard error rather than a
    // silently divergent authority; otherwise the tier's own manifest tag
    // names the shared on-disk adapter.
    let path = match requested_turbo_selection()? {
        Some((path, env_tier)) if cfg!(feature = "h3-private-uat") => {
            if env_tier.stable_id() != authority.tier_stable_id() {
                bail!(
                    "MiniMax H3 Turbo selection changed from {} to {} after admission",
                    authority.tier_stable_id(),
                    env_tier.stable_id()
                )
            }
            path
        }
        Some(_) => bail!(
            "{TURBO_ADAPTER_PATH_VARIABLE}/{TURBO_ADAPTER_TIER_VARIABLE} are a capture-scope \
             UAT override honored only under the h3-private-uat feature"
        ),
        None => {
            let manifest_tier = mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS
                .iter()
                .find(|manifest_tier| manifest_tier.tier_stable_id == tier.stable_id())
                .ok_or_else(|| {
                    anyhow!(
                        "MiniMax H3 Turbo tier {} has no manifest tag and no UAT selection",
                        tier.stable_id()
                    )
                })?;
            manifest_adapter_path(models_root, manifest_tier)?
        }
    };
    // The frozen authority's distillation triple must still be the reviewed
    // one for this tier. The constructor derives it, so this catches a value
    // mutated between admission and load rather than a bad build.
    let reviewed = turbo_tier_contract(tier)?;
    if authority.resolved_sampler_kind() != reviewed.sampler_kind
        || authority.grid_points() != reviewed.grid_points
        || authority.video_shift().to_bits() != reviewed.video_shift.to_bits()
    {
        bail!(
            "MiniMax H3 Turbo authority for {} carries {}/{} steps/shift {}, reviewed is {}/{} steps/shift {}",
            tier.stable_id(),
            authority.resolved_sampler_kind().as_str(),
            authority.grid_points(),
            authority.video_shift(),
            reviewed.sampler_kind.as_str(),
            reviewed.grid_points,
            reviewed.video_shift
        )
    }
    let runtime = if device.is_metal() {
        H3TurboLoraRuntime::open_metal_streamed(&path, tier, device, dtype, cancellation)
    } else {
        H3TurboLoraRuntime::open(&path, tier, device, dtype, cancellation.as_ref())
    }
    .map_err(|error| anyhow!("{error}"))?;
    if runtime.adapter_identity_sha256() != authority.adapter_identity_sha256()
        || runtime.content_sha256() != authority.adapter_content_sha256()
    {
        bail!("MiniMax H3 Turbo adapter changed between admission and transformer load")
    }
    let resident_cost_matches = if runtime.is_metal_streamed() {
        runtime
            .device_bytes()
            .checked_add(runtime.streamed_main_block_device_bytes())
            .is_some_and(|peak| peak <= authority.resident_device_bytes())
    } else {
        runtime.device_bytes() == authority.resident_device_bytes()
    };
    if !resident_cost_matches
        || runtime.device_staging_peak_bytes() != authority.device_staging_peak_bytes()
        || runtime.host_staging_peak_bytes() != authority.host_staging_peak_bytes()
    {
        bail!(
            "MiniMax H3 Turbo adapter costs {} resident / {} device staging / {} host staging bytes, admission charged {} / {} / {}",
            runtime.device_bytes(),
            runtime.device_staging_peak_bytes(),
            runtime.host_staging_peak_bytes(),
            authority.resident_device_bytes(),
            authority.device_staging_peak_bytes(),
            authority.host_staging_peak_bytes()
        )
    }
    Ok(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reviewed_tier_has_exactly_one_contract() {
        assert_eq!(REVIEWED_TURBO_TIERS.len(), H3TurboLoraTier::ALL.len());
        for tier in H3TurboLoraTier::ALL {
            let contract = turbo_tier_contract(tier).unwrap();
            assert_eq!(contract.tier, tier);
            // Every Turbo workflow selects euler over the Comfy simple grid.
            assert_eq!(contract.sampler_kind, H3SamplerKind::ComfyEuler);
            assert!(contract.sampler_kind.uses_comfy_simple_grid());
            assert!(contract.sampler_kind.uses_euler_update());
            assert!(contract.video_shift.is_finite() && contract.video_shift > 0.0);
        }
    }

    /// mold `steps` is terminal-inclusive, so an N-step tier is N+1 steps.
    #[test]
    fn reviewed_step_counts_are_the_published_evaluation_counts_plus_one() {
        let expected = [
            (H3TurboLoraTier::Fl2v8StepV10, 9u32, H3_VIDEO_SHIFT),
            (
                H3TurboLoraTier::Fl2v768p4StepV10,
                5,
                H3_TURBO_768P_VIDEO_SHIFT,
            ),
            (H3TurboLoraTier::Ref2v4StepV10, 5, H3_VIDEO_SHIFT),
            (
                H3TurboLoraTier::Fl2v768p4StepV11,
                5,
                H3_TURBO_768P_VIDEO_SHIFT,
            ),
            (
                H3TurboLoraTier::Fl2v768p8StepV10,
                9,
                H3_TURBO_768P_VIDEO_SHIFT,
            ),
        ];
        for (tier, grid_points, video_shift) in expected {
            let contract = turbo_tier_contract(tier).unwrap();
            assert_eq!(contract.grid_points, grid_points, "{tier:?}");
            assert_eq!(contract.video_shift, video_shift, "{tier:?}");
            // The schedule must build and must not collapse.
            let schedule = super::super::sampler::H3DualSchedule::new_for_sampler_with_video_shift(
                usize::try_from(grid_points).unwrap(),
                contract.sampler_kind,
                video_shift,
            )
            .unwrap();
            assert_eq!(
                schedule.counts().transformer_evaluations,
                usize::try_from(grid_points).unwrap() - 1
            );
        }
        // Only the 768p-trained tiers move the shift.
        assert_ne!(H3_TURBO_768P_VIDEO_SHIFT, H3_VIDEO_SHIFT);
    }

    #[test]
    fn tiers_parse_from_stable_ids_and_short_aliases() {
        for tier in H3TurboLoraTier::ALL {
            assert_eq!(parse_turbo_tier(tier.stable_id()).unwrap(), tier);
            assert_eq!(parse_turbo_tier(short_tier_alias(tier)).unwrap(), tier);
            assert_eq!(
                parse_turbo_tier(&format!("  {}  ", short_tier_alias(tier))).unwrap(),
                tier
            );
        }
        let error = parse_turbo_tier("fl2v-2step").unwrap_err().to_string();
        assert!(error.contains("not reviewed"), "{error}");
        assert!(error.contains("fl2v-8step"), "{error}");
        assert!(error.contains("fl2v-8step-768p"), "{error}");
        assert!(error.contains("fl2v-4step-768p-v1.1"), "{error}");
    }

    #[test]
    fn the_authority_derives_its_distillation_contract_from_the_reviewed_tier() {
        for tier in H3TurboLoraTier::ALL {
            let reviewed = turbo_tier_contract(tier).unwrap();
            let authority = H3FactoryTurboAdapterAuthority::for_reviewed_tier(
                tier.stable_id(),
                &"a".repeat(64),
                tier.content_sha256(),
                1_956_118_528,
                20_643_840,
                33_030_144,
            )
            .unwrap();
            // The caller never supplied any of these; they came from the table.
            assert_eq!(authority.grid_points(), reviewed.grid_points);
            assert_eq!(authority.video_shift(), reviewed.video_shift);
            assert_eq!(authority.resolved_sampler_kind(), reviewed.sampler_kind);
            assert_eq!(authority.tier_stable_id(), tier.stable_id());
            assert_eq!(authority.resident_device_bytes(), 1_956_118_528);
            assert_eq!(authority.device_staging_peak_bytes(), 20_643_840);
            assert_eq!(authority.host_staging_peak_bytes(), 33_030_144);
        }
    }

    #[test]
    fn an_unreviewed_tier_can_never_mint_an_authority() {
        let error = H3FactoryTurboAdapterAuthority::for_reviewed_tier(
            "minimax-h3.turbo-lora.fl2v-2step.v1",
            &"a".repeat(64),
            &"b".repeat(64),
            1,
            1,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not a reviewed tier"), "{error}");
    }

    /// The mold-core manifest tier table and the mold-candle runtime tier
    /// table describe the same reviewed artifacts. This is the pin that keeps
    /// acquisition (manifest name, file identity, default steps) and runtime
    /// (tier authentication, distillation triple) from drifting apart.
    #[test]
    fn manifest_tiers_pin_the_exact_reviewed_runtime_tiers() {
        for manifest_tier in mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS {
            let tier = parse_turbo_tier(manifest_tier.tier_stable_id).unwrap();
            assert_eq!(tier.stable_id(), manifest_tier.tier_stable_id);
            assert_eq!(tier.repository_path(), manifest_tier.adapter_hf_filename);
            // Provenance is per tier on BOTH sides: the acquisition row and
            // the runtime tier must name the same repository and revision, or
            // a download and an authentication describe different artifacts.
            assert_eq!(
                tier.source_repository(),
                manifest_tier.adapter_hf_repo,
                "{tier:?}"
            );
            assert_eq!(
                tier.source_revision(),
                manifest_tier.adapter_hf_revision,
                "{tier:?}"
            );
            assert_eq!(tier.file_bytes(), manifest_tier.adapter_size_bytes);
            assert_eq!(tier.content_sha256(), manifest_tier.adapter_sha256);
            let contract = turbo_tier_contract(tier).unwrap();
            assert_eq!(contract.grid_points, manifest_tier.steps);
        }
        assert_eq!(
            mold_candle::minimax_h3::H3_TURBO_LORA_SOURCE_REVISION,
            mold_core::minimax_h3::COMFY_TURBO_LORA_REVISION
        );
        assert_eq!(
            mold_candle::minimax_h3::H3_TURBO_LORA_REPOSITORY,
            mold_core::minimax_h3::COMFY_REPO
        );
        assert_eq!(
            mold_candle::minimax_h3::H3_TURBO_LORA_LIGHTX2V_REPOSITORY,
            mold_core::minimax_h3::LIGHTX2V_REPO
        );
        assert_eq!(
            mold_candle::minimax_h3::H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION,
            mold_core::minimax_h3::LIGHTX2V_REVISION
        );
    }

    #[test]
    fn manifest_selection_resolves_the_shared_family_adapter_path() {
        let root = std::path::Path::new("/models");
        for manifest_tier in mold_core::minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS {
            let (path, tier) = manifest_turbo_selection(manifest_tier.model, root)
                .unwrap()
                .expect("turbo tag selects a tier");
            assert_eq!(tier.stable_id(), manifest_tier.tier_stable_id);
            assert_eq!(
                path,
                root.join("shared").join("minimax-h3").join("loras").join(
                    std::path::Path::new(manifest_tier.adapter_hf_filename)
                        .file_name()
                        .unwrap()
                )
            );
        }
        assert!(
            manifest_turbo_selection(mold_core::minimax_h3::FL2VA_COMFY, root)
                .unwrap()
                .is_none()
        );
    }

    /// The env pair and a manifest-selected Turbo tag are two selection
    /// authorities; outside the capture-scope `h3-private-uat` feature a set
    /// pair is refused instead of silently losing or winning.
    /// Serializes the tests that read or write the process-global env pair.
    static TURBO_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn env_pair_is_uat_only_and_contradicts_a_manifest_selection() {
        let _lock = TURBO_ENV_LOCK.lock().unwrap();
        struct EnvPairGuard;
        impl Drop for EnvPairGuard {
            fn drop(&mut self) {
                std::env::remove_var(TURBO_ADAPTER_PATH_VARIABLE);
                std::env::remove_var(TURBO_ADAPTER_TIER_VARIABLE);
            }
        }
        let _guard = EnvPairGuard;
        std::env::set_var(TURBO_ADAPTER_PATH_VARIABLE, "/uat/adapter.safetensors");
        std::env::set_var(TURBO_ADAPTER_TIER_VARIABLE, "fl2v-8step");

        let root = std::path::Path::new("/models");
        let with_manifest =
            resolve_turbo_selection(mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P, root);
        let env_only = resolve_turbo_selection(mold_core::minimax_h3::FL2VA_COMFY, root);
        if cfg!(feature = "h3-private-uat") {
            // Capture-scope override: the env pair wins in both shapes.
            for result in [with_manifest, env_only] {
                let (path, tier) = result.unwrap().expect("UAT env pair selects");
                assert_eq!(path, std::path::PathBuf::from("/uat/adapter.safetensors"));
                assert_eq!(tier, H3TurboLoraTier::Fl2v8StepV10);
            }
        } else {
            let contradiction = with_manifest.unwrap_err().to_string();
            assert!(contradiction.contains("contradict"), "{contradiction}");
            let refused = env_only.unwrap_err().to_string();
            assert!(refused.contains("h3-private-uat"), "{refused}");
            assert!(
                refused.contains(mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP),
                "{refused}"
            );
        }
    }

    /// With the env pair unset, the manifest tag is the whole selection and a
    /// base model selects nothing.
    #[test]
    fn manifest_tags_select_without_any_environment() {
        let _lock = TURBO_ENV_LOCK.lock().unwrap();
        std::env::remove_var(TURBO_ADAPTER_PATH_VARIABLE);
        std::env::remove_var(TURBO_ADAPTER_TIER_VARIABLE);
        let root = std::path::Path::new("/models");
        assert!(
            resolve_turbo_selection(mold_core::minimax_h3::FL2VA_COMFY, root)
                .unwrap()
                .is_none()
        );
        let (_, tier) =
            resolve_turbo_selection(mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP, root)
                .unwrap()
                .expect("turbo tag selects");
        assert_eq!(tier, H3TurboLoraTier::Fl2v8StepV10);
    }

    #[test]
    fn a_reviewed_tiers_lookup_is_by_stable_id_only() {
        for tier in H3TurboLoraTier::ALL {
            assert_eq!(
                reviewed_contract_for_stable_id(tier.stable_id())
                    .unwrap()
                    .tier,
                tier
            );
            // The short alias is an interim convenience for the env knob, not
            // an identity the frozen authority accepts.
            assert!(reviewed_contract_for_stable_id(short_tier_alias(tier)).is_none());
        }
        assert!(reviewed_contract_for_stable_id("").is_none());
    }
}
