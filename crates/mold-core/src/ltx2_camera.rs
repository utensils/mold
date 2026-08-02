use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ltx2CameraProfile {
    Ltx2_19b,
}

impl Ltx2CameraProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ltx2_19b => "LTX-2 19B",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ltx2CameraControlPreset {
    pub id: &'static str,
    pub label: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
    pub download_model: &'static str,
}

pub const LTX2_CAMERA_CONTROLS: &[Ltx2CameraControlPreset] = &[
    Ltx2CameraControlPreset {
        id: "dolly-in",
        label: "Dolly in",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
        hf_filename: "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        size_bytes: 327_309_208,
        sha256: "0da937d528ea619c95859ad0c5cab012c7213fe058b7cb289e2a513ec351f237",
        download_model: "ltx2-camera-control-dolly-in-19b",
    },
    Ltx2CameraControlPreset {
        id: "dolly-left",
        label: "Dolly left",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        hf_filename: "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        size_bytes: 327_309_208,
        sha256: "fdd28ee1a53e5413d74b90064c7ec0fe98b78457b5da06c0c9c4ea5810d1f5f6",
        download_model: "ltx2-camera-control-dolly-left-19b",
    },
    Ltx2CameraControlPreset {
        id: "dolly-out",
        label: "Dolly out",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
        hf_filename: "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        size_bytes: 327_309_208,
        sha256: "554c30e64fa57b6c98e059f1b3ccabc50ad85195aab0911aed317b714f8d644d",
        download_model: "ltx2-camera-control-dolly-out-19b",
    },
    Ltx2CameraControlPreset {
        id: "dolly-right",
        label: "Dolly right",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
        hf_filename: "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        size_bytes: 327_309_208,
        sha256: "ee1a3a34dc21c6b61a4c5775756dd0fab8700e06dd35658c852c0127202aa5dd",
        download_model: "ltx2-camera-control-dolly-right-19b",
    },
    Ltx2CameraControlPreset {
        id: "jib-down",
        label: "Jib down",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
        hf_filename: "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        size_bytes: 2_214_978_664,
        sha256: "5a2f72b231841f998a46b949271fa0bcec21012d4238a53e7eb4762bc828bee3",
        download_model: "ltx2-camera-control-jib-down-19b",
    },
    Ltx2CameraControlPreset {
        id: "jib-up",
        label: "Jib up",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
        hf_filename: "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        size_bytes: 2_214_978_664,
        sha256: "c4e8bd628238caf20079515ba2bce5770fa1108e6ed8c1d403a3179a68dce77f",
        download_model: "ltx2-camera-control-jib-up-19b",
    },
    Ltx2CameraControlPreset {
        id: "static",
        label: "Static",
        hf_repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
        hf_filename: "ltx-2-19b-lora-camera-control-static.safetensors",
        size_bytes: 2_214_978_664,
        sha256: "6b79aad7ecdd60aef07f39177d1ac225a6608806086af94848436fec432e2d0d",
        download_model: "ltx2-camera-control-static-19b",
    },
];

pub fn normalize_camera_control_id(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

pub fn camera_controls_for_profile(
    profile: Ltx2CameraProfile,
) -> impl Iterator<Item = &'static Ltx2CameraControlPreset> {
    match profile {
        Ltx2CameraProfile::Ltx2_19b => LTX2_CAMERA_CONTROLS.iter(),
    }
}

pub fn resolve_camera_control_preset(id: &str) -> Result<&'static Ltx2CameraControlPreset, String> {
    let id = normalize_camera_control_id(id);
    LTX2_CAMERA_CONTROLS
        .iter()
        .find(|preset| preset.id == id)
        .ok_or_else(|| {
            let valid = LTX2_CAMERA_CONTROLS
                .iter()
                .map(|preset| preset.id)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown LTX-2 camera-control preset '{id}' (expected one of: {valid})")
        })
}

/// Resolve camera-control compatibility from the effective runtime model
/// configuration. Official camera-control LoRAs are published only for the
/// LTX-2 19B architecture.
pub fn camera_profile_for_config(config: &ModelConfig) -> Result<Ltx2CameraProfile, String> {
    if config.family.as_deref() != Some("ltx2") {
        return Err("camera controls require an LTX-2 model".to_string());
    }
    camera_profile_for_artifact_paths([
        config.transformer.as_deref(),
        config.vae.as_deref(),
        config.spatial_upscaler.as_deref(),
    ])
}

/// The architecture sniff behind `camera_profile_for_config`, over resolved
/// artifact paths rather than a `ModelConfig`.
///
/// The engine holds `ModelPaths`, not a `ModelConfig`, and used to make this
/// call with `model_name.contains("ltx-2.3")` — which is wrong in both
/// directions for opaque `cv:` / `hf:` catalog IDs, whose names contain no
/// architecture at all. Routing both callers through one function keeps the
/// answer identical wherever it is asked.
pub fn camera_profile_for_artifact_paths<'a>(
    architecture_paths: impl IntoIterator<Item = Option<&'a str>>,
) -> Result<Ltx2CameraProfile, String> {
    let architecture_paths: Vec<&str> = architecture_paths.into_iter().flatten().collect();
    if architecture_paths
        .iter()
        .any(|path| path.contains("ltx-2.3"))
    {
        return Err(
            "camera-control presets are currently published for LTX-2 19B only".to_string(),
        );
    }
    if architecture_paths.iter().any(|path| path.contains("ltx-2")) {
        return Ok(Ltx2CameraProfile::Ltx2_19b);
    }
    Err("the installed checkpoint architecture is unknown; select an LTX-2 19B profile".to_string())
}

/// Resolve built-in manifests through an explicit profile table, then fall
/// back to effective runtime configuration for opaque catalog IDs.
pub fn camera_profile_for_model(
    model: &str,
    config: &ModelConfig,
) -> Result<Ltx2CameraProfile, String> {
    let canonical = crate::manifest::resolve_model_name(model);
    if matches!(
        canonical.as_str(),
        "ltx-2-19b-dev:fp8" | "ltx-2-19b-distilled:fp8"
    ) && config.family.as_deref() == Some("ltx2")
    {
        return Ok(Ltx2CameraProfile::Ltx2_19b);
    }
    if canonical.starts_with("ltx-2.3-") {
        return Err(
            "camera-control presets are currently published for LTX-2 19B only".to_string(),
        );
    }
    camera_profile_for_config(config)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Ltx2CameraControlInfo {
    pub id: String,
    pub label: String,
    pub size_bytes: u64,
    pub installed: bool,
    pub download_model: String,
    pub download_repo: String,
    pub download_filename: String,
    pub download_sha256: String,
}

/// The detailed form of `/api/capabilities/ltx2-camera-controls`, returned
/// only for `?detail=1`.
///
/// The bare-array form cannot distinguish "this checkpoint has no published
/// presets" from "the request failed", so every client hard-coded its own
/// guess at the server's policy — and guessed wrong for the
/// unknown-architecture case, which used to 422 into a silent empty picker.
/// This carries the server's own reason instead. It is opt-in so the array
/// response older clients parse stays byte-identical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Ltx2CameraControlAvailability {
    pub controls: Vec<Ltx2CameraControlInfo>,
    pub supported: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// `studio/lib/cameraMotion.ts` mirrors this registry so a clip's camera
    /// choice can be recovered from a chain script or a saved print with no
    /// server in the loop. That matching falls back to the human label, so a
    /// drifted label silently stops restoring the user's choice.
    #[test]
    fn camera_motion_ts_mirror_matches_the_rust_registry() {
        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));
        let path = format!("{workspace}/studio/lib/cameraMotion.ts");
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        let list = source
            .split_once("export const CAMERA_MOTION_PRESETS = [")
            .expect("cameraMotion.ts must declare CAMERA_MOTION_PRESETS")
            .1
            .split_once("] as const;")
            .expect("CAMERA_MOTION_PRESETS must end in `] as const;`")
            .0;

        for preset in LTX2_CAMERA_CONTROLS {
            let entry = format!("{{ id: \"{}\", label: \"{}\" }}", preset.id, preset.label);
            assert!(
                list.contains(&entry),
                "studio/lib/cameraMotion.ts is missing `{entry}`"
            );
        }
        assert_eq!(
            list.matches("{ id:").count(),
            LTX2_CAMERA_CONTROLS.len(),
            "studio/lib/cameraMotion.ts declares a different number of presets than the registry"
        );
    }

    /// Compatibility comes from the resolved artifacts, never the model name.
    /// An opaque catalog ID carries no architecture, so a name-substring test
    /// both admitted LTX-2.3 and rejected a 19B install reached that way.
    #[test]
    fn camera_profile_reads_artifact_paths_not_the_model_name() {
        assert!(
            camera_profile_for_artifact_paths([Some(
                "/models/cv-2752735/ltx-2.3-22b-distilled.safetensors"
            )])
            .unwrap_err()
            .contains("LTX-2 19B only"),
            "an opaque catalog ID pointing at LTX-2.3 artifacts must be rejected"
        );
        assert_eq!(
            camera_profile_for_artifact_paths([Some(
                "/models/cv-3063794/ltx-2-19b-distilled-fp8.safetensors"
            )])
            .unwrap(),
            Ltx2CameraProfile::Ltx2_19b,
            "an opaque catalog ID pointing at 19B artifacts must be accepted"
        );
        assert!(
            camera_profile_for_artifact_paths([Some("/models/mystery/weights.safetensors")])
                .unwrap_err()
                .contains("unknown")
        );
    }

    #[test]
    fn registry_has_unique_normalized_ids_and_download_models() {
        let mut ids = HashSet::new();
        let mut download_models = HashSet::new();
        for preset in LTX2_CAMERA_CONTROLS {
            assert!(ids.insert(preset.id));
            assert!(download_models.insert(preset.download_model));
            assert_eq!(normalize_camera_control_id(preset.id), preset.id);
            assert_eq!(preset.sha256.len(), 64);
            assert!(preset.sha256.chars().all(|ch| ch.is_ascii_hexdigit()));
            assert!(preset.hf_filename.ends_with(".safetensors"));
        }
        assert_eq!(ids.len(), 7);
    }

    #[test]
    fn official_artifact_identities_are_exact() {
        let actual = LTX2_CAMERA_CONTROLS
            .iter()
            .map(|preset| {
                (
                    preset.id,
                    preset.hf_repo,
                    preset.hf_filename,
                    preset.size_bytes,
                    preset.sha256,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                (
                    "dolly-in",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
                    "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                    327_309_208,
                    "0da937d528ea619c95859ad0c5cab012c7213fe058b7cb289e2a513ec351f237",
                ),
                (
                    "dolly-left",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
                    "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
                    327_309_208,
                    "fdd28ee1a53e5413d74b90064c7ec0fe98b78457b5da06c0c9c4ea5810d1f5f6",
                ),
                (
                    "dolly-out",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
                    "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
                    327_309_208,
                    "554c30e64fa57b6c98e059f1b3ccabc50ad85195aab0911aed317b714f8d644d",
                ),
                (
                    "dolly-right",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
                    "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
                    327_309_208,
                    "ee1a3a34dc21c6b61a4c5775756dd0fab8700e06dd35658c852c0127202aa5dd",
                ),
                (
                    "jib-down",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
                    "ltx-2-19b-lora-camera-control-jib-down.safetensors",
                    2_214_978_664,
                    "5a2f72b231841f998a46b949271fa0bcec21012d4238a53e7eb4762bc828bee3",
                ),
                (
                    "jib-up",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
                    "ltx-2-19b-lora-camera-control-jib-up.safetensors",
                    2_214_978_664,
                    "c4e8bd628238caf20079515ba2bce5770fa1108e6ed8c1d403a3179a68dce77f",
                ),
                (
                    "static",
                    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
                    "ltx-2-19b-lora-camera-control-static.safetensors",
                    2_214_978_664,
                    "6b79aad7ecdd60aef07f39177d1ac225a6608806086af94848436fec432e2d0d",
                ),
            ]
        );
    }

    #[test]
    fn preset_ids_are_normalized_without_guessing_aliases() {
        assert_eq!(normalize_camera_control_id(" Dolly_In "), "dolly-in");
        assert!(resolve_camera_control_preset("DOLLY_IN").is_ok());
        assert!(resolve_camera_control_preset("dolly").is_err());
    }

    #[test]
    fn profile_resolution_accepts_only_ltx2_19b() {
        let built_in = ModelConfig {
            family: Some("ltx2".into()),
            is_schnell: Some(true),
            ..Default::default()
        };
        assert_eq!(
            camera_profile_for_model("ltx-2-19b-distilled:fp8", &built_in).unwrap(),
            Ltx2CameraProfile::Ltx2_19b
        );
        assert_eq!(
            camera_profile_for_model("ltx-2-19b-dev:fp8", &built_in).unwrap(),
            Ltx2CameraProfile::Ltx2_19b
        );
        assert!(
            camera_profile_for_model("ltx-2.3-22b-distilled:fp8", &built_in)
                .unwrap_err()
                .contains("19B only")
        );

        let opaque_19b = ModelConfig {
            transformer: Some("/models/catalog/ltx-2-19b-distilled.safetensors".into()),
            ..built_in.clone()
        };
        assert_eq!(
            camera_profile_for_model("hf:opaque", &opaque_19b).unwrap(),
            Ltx2CameraProfile::Ltx2_19b
        );

        let dev = ModelConfig {
            is_schnell: Some(false),
            ..opaque_19b
        };
        assert_eq!(
            camera_profile_for_config(&dev).unwrap(),
            Ltx2CameraProfile::Ltx2_19b
        );
    }
}
