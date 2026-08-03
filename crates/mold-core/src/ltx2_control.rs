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
    /// Whether Hugging Face requires an accepted licence before serving this
    /// repository. Surfaced to clients so they can say "accept the licence and
    /// set `HF_TOKEN`" *before* a download fails with a 403.
    pub gated: bool,
    /// Extra files the adapter cannot run without, beyond `hf_filename`.
    ///
    /// The HDR adapter ships pre-computed prompt embeddings alongside its
    /// weights, so an adapter is not always exactly one file.
    pub extra_files: &'static [Ltx2ControlAdapterFile],
}

/// A companion file an adapter needs on disk alongside its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ltx2ControlAdapterFile {
    pub hf_filename: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
}

impl Ltx2ControlAdapter {
    /// Every file this adapter needs, weights first.
    pub fn files(&self) -> impl Iterator<Item = Ltx2ControlAdapterFile> + '_ {
        std::iter::once(Ltx2ControlAdapterFile {
            hf_filename: self.hf_filename,
            size_bytes: self.size_bytes,
            sha256: self.sha256,
        })
        .chain(self.extra_files.iter().copied())
    }

    /// Total bytes to download, across every file.
    pub fn total_size_bytes(&self) -> u64 {
        self.files().map(|file| file.size_bytes).sum()
    }

    /// The adapter's pre-computed text-embedding companion, if it ships one.
    ///
    /// Upstream's HDR pipeline takes these embeddings instead of encoding a
    /// prompt, so the engine needs to find the file by name rather than
    /// hardcoding it. Returns `None` for every adapter that encodes normally.
    pub fn scene_embeddings_filename(&self) -> Option<&'static str> {
        self.extra_files
            .iter()
            .map(|file| file.hf_filename)
            .find(|name| name.contains("scene-emb"))
    }
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
        gated: false,
        extra_files: &[],
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
        gated: false,
        extra_files: &[],
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
        gated: false,
        extra_files: &[],
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
        gated: false,
        extra_files: &[],
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
        gated: false,
        extra_files: &[],
    },
    Ltx2ControlAdapter {
        id: "lipdub",
        label: "Lip dub",
        guide: "A reference video with speech; the mouth is re-timed to new audio.",
        profile: Ltx2ControlProfile::Ltx2_23_22bDistilled,
        // Upstream's README still names the repository LTX-2.3-22b-IC-LoRA-LipDub,
        // which now 307-redirects to DubIt. Pin the destination so the download
        // does not depend on a redirect being honoured.
        hf_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
        hf_filename: "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
        size_bytes: 2_466_665_072,
        sha256: "fc415b12cb639e78511bc264f85080c2f7b188e334c1d9fade76b310e2bc419c",
        download_model: "ltx2-control-lipdub-23",
        gated: true,
        extra_files: &[],
    },
    Ltx2ControlAdapter {
        id: "hdr",
        label: "HDR",
        guide: "An SDR reference video; the render is re-graded to linear HDR.",
        profile: Ltx2ControlProfile::Ltx2_23_22bDistilled,
        hf_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
        hf_filename: "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
        size_bytes: 327_309_312,
        sha256: "c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e",
        download_model: "ltx2-control-hdr-23",
        gated: true,
        // The HDR pipeline runs from pre-computed prompt embeddings rather
        // than encoding a prompt, so this file is not optional.
        extra_files: &[Ltx2ControlAdapterFile {
            hf_filename: "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
            size_bytes: 12_583_096,
            sha256: "78bffa6049bae2649a4365ec8769db88052c21348d643e8fc1ce6d483d994c5b",
        }],
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
    /// Total bytes across every file the adapter needs, not just its weights.
    pub size_bytes: u64,
    pub installed: bool,
    pub download_model: String,
    pub download_repo: String,
    pub download_filename: String,
    pub download_sha256: String,
    /// Whether Hugging Face requires an accepted licence first. Additive:
    /// absent on servers that predate gated adapters, which clients read as
    /// "not gated" — the same answer those servers would have given.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gated: bool,
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
        assert_eq!(seen.len(), 7);
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
                (
                    "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
                    "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
                    2_466_665_072,
                    "fc415b12cb639e78511bc264f85080c2f7b188e334c1d9fade76b310e2bc419c",
                ),
                (
                    "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
                    "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
                    327_309_312,
                    "c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e",
                ),
            ]
        );
    }

    /// A multi-file adapter must report every file it needs, and its
    /// advertised size must be the whole download — quoting only the weights
    /// would understate the HDR adapter by its 12 MB embeddings file.
    #[test]
    fn adapters_enumerate_every_file_they_need() {
        let hdr = LTX2_CONTROL_ADAPTERS
            .iter()
            .find(|a| a.id == "hdr")
            .expect("hdr adapter is registered");
        let files: Vec<_> = hdr.files().collect();
        assert_eq!(files.len(), 2, "HDR ships weights plus scene embeddings");
        assert_eq!(files[0].hf_filename, hdr.hf_filename, "weights come first");
        assert!(files[1].hf_filename.contains("scene-emb"));
        assert_eq!(
            hdr.total_size_bytes(),
            files.iter().map(|f| f.size_bytes).sum::<u64>()
        );
        assert!(hdr.total_size_bytes() > hdr.size_bytes);

        // Single-file adapters are unchanged.
        for adapter in LTX2_CONTROL_ADAPTERS.iter().filter(|a| a.id != "hdr") {
            assert_eq!(adapter.files().count(), 1, "{}", adapter.id);
            assert_eq!(adapter.total_size_bytes(), adapter.size_bytes);
        }
    }

    /// Gating is per-adapter, and the generated manifests must carry it — a
    /// gated file that claims otherwise fails late with an opaque 403 instead
    /// of the "accept the licence, set HF_TOKEN" guidance the downloader has.
    #[test]
    fn gated_adapters_propagate_to_their_manifests() {
        for adapter in LTX2_CONTROL_ADAPTERS {
            let manifest = crate::manifest::find_manifest(adapter.download_model)
                .unwrap_or_else(|| panic!("{} has no manifest", adapter.id));
            assert_eq!(
                manifest.files.len(),
                adapter.files().count(),
                "{} manifest file count",
                adapter.id
            );
            for file in &manifest.files {
                assert_eq!(file.gated, adapter.gated, "{} gating", adapter.id);
            }
        }
        let gated: Vec<_> = LTX2_CONTROL_ADAPTERS
            .iter()
            .filter(|a| a.gated)
            .map(|a| a.id)
            .collect();
        assert_eq!(gated, ["lipdub", "hdr"]);
    }

    /// The 2.3 adapters are for the 22B distilled profile; offering one on a
    /// 19B checkpoint would download gigabytes that cannot load.
    #[test]
    fn new_adapters_resolve_only_for_the_22b_profile() {
        for id in ["lipdub", "hdr"] {
            assert!(resolve_control_adapter(Ltx2ControlProfile::Ltx2_23_22bDistilled, id).is_ok());
            assert!(resolve_control_adapter(Ltx2ControlProfile::Ltx2_19bDistilled, id).is_err());
        }
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
        assert_eq!(controls_23, ["union", "motion-track", "lipdub", "hdr"]);

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
