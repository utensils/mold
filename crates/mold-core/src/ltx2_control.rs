use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ltx2ControlProfile {
    Ltx2_19bDistilled,
    Ltx2_23_22bDistilled,
}

impl Ltx2ControlProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ltx2_19bDistilled => "LTX-2 19B distilled",
            Self::Ltx2_23_22bDistilled => "LTX-2.3 22B distilled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ltx2ControlAdapter {
    pub id: &'static str,
    pub label: &'static str,
    pub guide: &'static str,
    pub profile: Ltx2ControlProfile,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
    pub download_model: &'static str,
}

pub const LTX2_CONTROL_ADAPTERS: &[Ltx2ControlAdapter] = &[
    Ltx2ControlAdapter {
        id: "union",
        label: "Union control",
        guide: "A frame-aligned Canny, depth, or pose guide video.",
        profile: Ltx2ControlProfile::Ltx2_19bDistilled,
        hf_repo: "Lightricks/LTX-2-19b-IC-LoRA-Union-Control",
        hf_filename: "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
        size_bytes: 654_465_296,
        sha256: "cd342e0dcf7754d8b36135b5e768b0f0e820703acd372adc49e53de0b00a931b",
        download_model: "ltx2-control-union-19b",
    },
    Ltx2ControlAdapter {
        id: "pose",
        label: "Pose control",
        guide: "A frame-aligned pose or OpenPose guide video.",
        profile: Ltx2ControlProfile::Ltx2_19bDistilled,
        hf_repo: "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
        hf_filename: "ltx-2-19b-ic-lora-pose-control.safetensors",
        size_bytes: 654_465_256,
        sha256: "61816bf0985d4470c456160deec65df69188ae45d553e5aa8f1252fc543bc8aa",
        download_model: "ltx2-control-pose-19b",
    },
    Ltx2ControlAdapter {
        id: "detailer",
        label: "Detailer",
        guide: "The source video whose details and textures should be refined.",
        profile: Ltx2ControlProfile::Ltx2_19bDistilled,
        hf_repo: "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
        hf_filename: "ltx-2-19b-ic-lora-detailer.safetensors",
        size_bytes: 2_617_401_920,
        sha256: "05efdae9e472e06d168e122f5ebb890e7ef348cc047cf9876da6504c36d7d0e2",
        download_model: "ltx2-control-detailer-19b",
    },
    Ltx2ControlAdapter {
        id: "union",
        label: "Union control",
        guide: "A frame-aligned Canny, depth, or pose guide video.",
        profile: Ltx2ControlProfile::Ltx2_23_22bDistilled,
        hf_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        hf_filename: "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        size_bytes: 654_465_352,
        sha256: "a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18",
        download_model: "ltx2-control-union-23",
    },
    Ltx2ControlAdapter {
        id: "motion-track",
        label: "Motion track",
        guide: "A video with colored spline overlays marking the desired trajectories.",
        profile: Ltx2ControlProfile::Ltx2_23_22bDistilled,
        hf_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        hf_filename: "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        size_bytes: 327_309_314,
        sha256: "e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca",
        download_model: "ltx2-control-motion-track-23",
    },
];

pub fn normalize_control_id(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

pub fn adapters_for_profile(
    profile: Ltx2ControlProfile,
) -> impl Iterator<Item = &'static Ltx2ControlAdapter> {
    LTX2_CONTROL_ADAPTERS
        .iter()
        .filter(move |adapter| adapter.profile == profile)
}

pub fn resolve_control_adapter(
    profile: Ltx2ControlProfile,
    id: &str,
) -> Result<&'static Ltx2ControlAdapter, String> {
    let id = normalize_control_id(id);
    adapters_for_profile(profile)
        .find(|adapter| adapter.id == id)
        .ok_or_else(|| {
            let valid = adapters_for_profile(profile)
                .map(|adapter| adapter.id)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "IC-LoRA control '{id}' is not compatible with {}; choose one of: {valid}",
                profile.label()
            )
        })
}

/// Resolve the built-in control profile from the effective runtime model
/// configuration. This deliberately does not inspect the public model ID:
/// catalog models use opaque `cv:` / `hf:` identifiers, while the resolved
/// configuration retains the checkpoint architecture and distilled schedule.
pub fn control_profile_for_config(config: &ModelConfig) -> Result<Ltx2ControlProfile, String> {
    if config.family.as_deref() != Some("ltx2") {
        return Err("IC-LoRA controls require an LTX-2 model".to_string());
    }
    if config.is_schnell != Some(true) {
        return Err(
            "built-in IC-LoRA controls require an effective distilled LTX-2 profile".to_string(),
        );
    }

    let architecture_paths = [
        config.transformer.as_deref(),
        config.vae.as_deref(),
        config.spatial_upscaler.as_deref(),
    ];
    if architecture_paths
        .iter()
        .flatten()
        .any(|path| path.contains("ltx-2.3"))
    {
        return Ok(Ltx2ControlProfile::Ltx2_23_22bDistilled);
    }
    if architecture_paths
        .iter()
        .flatten()
        .any(|path| path.contains("ltx-2"))
    {
        return Ok(Ltx2ControlProfile::Ltx2_19bDistilled);
    }
    Err(
        "the installed checkpoint architecture is unknown; select an LTX-2 19B distilled or LTX-2.3 22B distilled profile"
            .to_string(),
    )
}

/// Resolve built-in manifests through an explicit profile table, then fall
/// back to the effective runtime config for custom and opaque catalog IDs.
/// The public model ID is never pattern-matched.
pub fn control_profile_for_model(
    model: &str,
    config: &ModelConfig,
) -> Result<Ltx2ControlProfile, String> {
    let canonical = crate::manifest::resolve_model_name(model);
    let manifest_profile = match canonical.as_str() {
        "ltx-2-19b-distilled:fp8" => Some(Ltx2ControlProfile::Ltx2_19bDistilled),
        "ltx-2.3-22b-distilled:fp8" | "ltx-2.3-22b-distilled:bf16" => {
            Some(Ltx2ControlProfile::Ltx2_23_22bDistilled)
        }
        _ => None,
    };
    if let Some(profile) = manifest_profile {
        if config.family.as_deref() == Some("ltx2") {
            return Ok(profile);
        }
    }
    control_profile_for_config(config)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Ltx2ControlAdapterInfo {
    pub id: String,
    pub label: String,
    pub guide: String,
    pub size_bytes: u64,
    pub installed: bool,
    pub download_model: String,
    pub download_repo: String,
    pub download_filename: String,
    pub download_sha256: String,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn registry_has_unique_profile_and_id_pairs() {
        let mut seen = HashSet::new();
        for adapter in LTX2_CONTROL_ADAPTERS {
            assert!(seen.insert((adapter.profile, adapter.id)));
            assert_eq!(adapter.sha256.len(), 64);
            assert!(adapter.hf_filename.ends_with(".safetensors"));
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn official_artifact_identities_are_exact() {
        let actual = LTX2_CONTROL_ADAPTERS
            .iter()
            .map(|adapter| {
                (
                    adapter.hf_repo,
                    adapter.hf_filename,
                    adapter.size_bytes,
                    adapter.sha256,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                (
                    "Lightricks/LTX-2-19b-IC-LoRA-Union-Control",
                    "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
                    654_465_296,
                    "cd342e0dcf7754d8b36135b5e768b0f0e820703acd372adc49e53de0b00a931b",
                ),
                (
                    "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
                    "ltx-2-19b-ic-lora-pose-control.safetensors",
                    654_465_256,
                    "61816bf0985d4470c456160deec65df69188ae45d553e5aa8f1252fc543bc8aa",
                ),
                (
                    "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
                    "ltx-2-19b-ic-lora-detailer.safetensors",
                    2_617_401_920,
                    "05efdae9e472e06d168e122f5ebb890e7ef348cc047cf9876da6504c36d7d0e2",
                ),
                (
                    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                    654_465_352,
                    "a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18",
                ),
                (
                    "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
                    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
                    327_309_314,
                    "e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca",
                ),
            ]
        );
    }

    #[test]
    fn compatibility_matrix_is_exact() {
        let controls_19b = adapters_for_profile(Ltx2ControlProfile::Ltx2_19bDistilled)
            .map(|adapter| adapter.id)
            .collect::<Vec<_>>();
        assert_eq!(controls_19b, ["union", "pose", "detailer"]);

        let controls_23 = adapters_for_profile(Ltx2ControlProfile::Ltx2_23_22bDistilled)
            .map(|adapter| adapter.id)
            .collect::<Vec<_>>();
        assert_eq!(controls_23, ["union", "motion-track"]);

        for id in ["union", "motion-track", "pose", "detailer"] {
            assert_eq!(
                resolve_control_adapter(Ltx2ControlProfile::Ltx2_19bDistilled, id).is_ok(),
                matches!(id, "union" | "pose" | "detailer"),
                "unexpected LTX-2 19B compatibility for {id}"
            );
            assert_eq!(
                resolve_control_adapter(Ltx2ControlProfile::Ltx2_23_22bDistilled, id).is_ok(),
                matches!(id, "union" | "motion-track"),
                "unexpected LTX-2.3 compatibility for {id}"
            );
        }
    }

    #[test]
    fn ids_are_normalized_without_guessing_aliases() {
        assert_eq!(normalize_control_id(" Motion_Track "), "motion-track");
        assert!(
            resolve_control_adapter(Ltx2ControlProfile::Ltx2_23_22bDistilled, "MOTION_TRACK")
                .is_ok()
        );
        assert!(
            resolve_control_adapter(Ltx2ControlProfile::Ltx2_23_22bDistilled, "motion").is_err()
        );
    }

    #[test]
    fn effective_config_is_the_profile_authority() {
        let config = ModelConfig {
            family: Some("ltx2".into()),
            is_schnell: Some(true),
            transformer: Some("/models/catalog/opaque/ltx-2.3-22b-distilled.safetensors".into()),
            ..Default::default()
        };
        assert_eq!(
            control_profile_for_config(&config).unwrap(),
            Ltx2ControlProfile::Ltx2_23_22bDistilled
        );

        let dev = ModelConfig {
            is_schnell: Some(false),
            ..config
        };
        assert!(control_profile_for_config(&dev)
            .unwrap_err()
            .contains("distilled"));
    }

    #[test]
    fn built_in_manifest_profiles_do_not_require_landed_paths() {
        let config = ModelConfig {
            family: Some("ltx2".into()),
            is_schnell: Some(true),
            ..Default::default()
        };
        assert_eq!(
            control_profile_for_model("ltx-2-19b-distilled:fp8", &config).unwrap(),
            Ltx2ControlProfile::Ltx2_19bDistilled
        );
        assert_eq!(
            control_profile_for_model("ltx-2.3-22b-distilled:fp8", &config).unwrap(),
            Ltx2ControlProfile::Ltx2_23_22bDistilled
        );
        assert!(control_profile_for_model("ltx-2.3-22b-dev:bf16", &config).is_err());
    }
}
