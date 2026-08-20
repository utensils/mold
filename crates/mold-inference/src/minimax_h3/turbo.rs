//! Reviewed MiniMax H3 Turbo tier contracts and their runtime resolution.
//!
//! A Turbo tier is a LoRA adapter overlaid on the *same* compact INT8 ConvRot
//! checkpoint, so nothing about the base artifact contract relaxes. What a tier
//! adds is a distillation triple — sampler kind, terminal-inclusive step count,
//! and video shift — plus the adapter's own authenticated identity and its
//! resident cost.
//!
//! Selection is currently by `MOLD_H3_TURBO_ADAPTER` (path) and
//! `MOLD_H3_TURBO_TIER` (tier id). Both are registered engine-shaping
//! variables. Manifests replace this knob in the follow-up; the tier table and
//! the authority it produces do not change when they do.

use anyhow::{anyhow, bail, Context, Result};
use mold_candle::minimax_h3::{
    authenticate_h3_turbo_lora_adapter, H3ComfyNeverCancel, H3TurboLoraContract,
    H3TurboLoraRuntime, H3TurboLoraTier,
};

use crate::h3_factory::{H3FactorySamplerKind, H3FactoryTurboAdapterAuthority};

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
/// All three tiers select `KSamplerSelect: euler` over Comfy's
/// `BasicScheduler("simple")` grid in their published reference workflows. Only
/// the 768p 4-step tier moves the video shift, which its own Diffusers
/// documentation passes as `--video-shift 6`.
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
];

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
pub(crate) fn turbo_adapter_authority(
    contract: &H3TurboLoraContract,
    host_staging_peak_bytes: u64,
    resident_device_bytes: u64,
) -> Result<H3FactoryTurboAdapterAuthority> {
    let tier = contract.tier();
    let tier_contract = turbo_tier_contract(tier)?;
    let authority = H3FactoryTurboAdapterAuthority {
        tier_stable_id: tier.stable_id().into(),
        adapter_identity_sha256: contract.adapter_identity_sha256().into(),
        adapter_content_sha256: contract.content_sha256().into(),
        sampler_kind: H3FactorySamplerKind::from_runtime(tier_contract.sampler_kind),
        grid_points: tier_contract.grid_points,
        video_shift_bits: tier_contract.video_shift.to_bits(),
        resident_device_bytes,
        host_staging_peak_bytes,
    };
    Ok(authority)
}

/// Authenticate the requested adapter at admission and describe it.
///
/// This reads and digests the whole file but allocates no device memory: the
/// deltas are materialized later, in the transformer-load phase, which is where
/// the budget charges them. The resident and staging figures are derived from
/// the validated structure rather than assumed, so the budget term and the
/// eventual allocation agree.
pub(crate) fn resolve_requested_turbo_authority() -> Result<Option<H3FactoryTurboAdapterAuthority>>
{
    let Some((path, tier)) = requested_turbo_selection()? else {
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
    for module in inspection.modules.values() {
        for reference in [&module.lora_a, &module.lora_b] {
            let bytes = reference.data_offsets[1] - reference.data_offsets[0];
            resident_device_bytes = resident_device_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("MiniMax H3 Turbo resident byte count overflows"))?;
            widest_matrix_bytes = widest_matrix_bytes.max(bytes);
        }
    }
    let host_staging_peak_bytes = widest_matrix_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow!("MiniMax H3 Turbo host staging byte count overflows"))?;
    turbo_adapter_authority(&contract, host_staging_peak_bytes, resident_device_bytes).map(Some)
}

/// Load the reviewed adapter's deltas onto the execution device.
///
/// The file is authenticated a second time here, on purpose: admission's digest
/// proves what was *offered*, and this one proves what is actually being read
/// into the transformer. Between the two the file could have been replaced, and
/// the deltas must come from the exact descriptor whose digest was verified.
pub(crate) fn load_reviewed_turbo_runtime(
    authority: &H3FactoryTurboAdapterAuthority,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<H3TurboLoraRuntime> {
    let Some((path, tier)) = requested_turbo_selection()? else {
        bail!("MiniMax H3 Turbo authority is present but no adapter selection is configured")
    };
    if tier.stable_id() != authority.tier_stable_id {
        bail!(
            "MiniMax H3 Turbo selection changed from {} to {} after admission",
            authority.tier_stable_id,
            tier.stable_id()
        )
    }
    let runtime = H3TurboLoraRuntime::open(&path, tier, device, dtype, &H3ComfyNeverCancel)
        .map_err(|error| anyhow!("{error}"))?;
    if runtime.adapter_identity_sha256() != authority.adapter_identity_sha256
        || runtime.content_sha256() != authority.adapter_content_sha256
    {
        bail!("MiniMax H3 Turbo adapter changed between admission and transformer load")
    }
    if runtime.device_bytes() != authority.resident_device_bytes
        || runtime.host_staging_peak_bytes() != authority.host_staging_peak_bytes
    {
        bail!(
            "MiniMax H3 Turbo adapter costs {} resident / {} staging bytes, admission charged {} / {}",
            runtime.device_bytes(),
            runtime.host_staging_peak_bytes(),
            authority.resident_device_bytes,
            authority.host_staging_peak_bytes
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
        // Only the 768p tier moves the shift.
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
    }

    #[test]
    fn the_authority_carries_its_tier_distillation_contract() {
        // Build the authority shape directly; `resolve_requested_turbo_authority`
        // needs a real adapter file, which the mold-candle tests own.
        let contract = turbo_tier_contract(H3TurboLoraTier::Fl2v768p4StepV10).unwrap();
        let authority = H3FactoryTurboAdapterAuthority {
            tier_stable_id: H3TurboLoraTier::Fl2v768p4StepV10.stable_id().into(),
            adapter_identity_sha256: "a".repeat(64),
            adapter_content_sha256: H3TurboLoraTier::Fl2v768p4StepV10
                .content_sha256()
                .to_owned(),
            sampler_kind: H3FactorySamplerKind::from_runtime(contract.sampler_kind),
            grid_points: contract.grid_points,
            video_shift_bits: contract.video_shift.to_bits(),
            resident_device_bytes: 1_956_118_528,
            host_staging_peak_bytes: 33_030_144,
        };
        assert_eq!(authority.grid_points, 5);
        assert_eq!(authority.video_shift(), H3_TURBO_768P_VIDEO_SHIFT);
        assert_eq!(authority.resolved_sampler_kind(), H3SamplerKind::ComfyEuler);
    }
}
