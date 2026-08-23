use std::fmt;
use std::path::Path;

use crate::minimax_h3;

use serde::{Deserialize, Serialize};

/// Stable machine-readable reason returned while MiniMax H3 authorization is absent.
pub const MINIMAX_H3_AUTHORIZATION_REQUIRED: &str = "MINIMAX_H3_AUTHORIZATION_REQUIRED";

/// Repository record that owns the authorization decision.
pub const MINIMAX_H3_AUTHORIZATION_ISSUE_URL: &str = "https://github.com/utensils/mold/issues/831";

/// License revision reviewed when this policy was introduced.
pub const MINIMAX_H3_LICENSE_URL: &str = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/\
bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE";

/// Content identity of the reviewed license bytes, pinned by the H3
/// conformance manifest and required by any future authorization record.
pub const MINIMAX_H3_LICENSE_SHA256: &str =
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelActivation {
    Available,
    ComplianceGated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelActivationError;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelAccessCapabilities {
    #[serde(default)]
    pub restrictions: Vec<ModelAccessRestriction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelAccessRestriction {
    pub code: String,
    pub family: String,
    pub message: String,
    pub license_url: String,
    pub authorization_url: String,
}

impl fmt::Display for ModelActivationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MiniMax H3 support is compliance-gated and is not activated in this build \
             ({MINIMAX_H3_AUTHORIZATION_REQUIRED}). See {MINIMAX_H3_AUTHORIZATION_ISSUE_URL}"
        )
    }
}

impl std::error::Error for ModelActivationError {}

impl ModelActivationError {
    pub fn restriction(self) -> ModelAccessRestriction {
        minimax_h3_restriction()
    }
}

pub fn model_access_capabilities() -> ModelAccessCapabilities {
    ModelAccessCapabilities {
        restrictions: Vec::new(),
    }
}

fn minimax_h3_restriction() -> ModelAccessRestriction {
    ModelAccessRestriction {
        code: MINIMAX_H3_AUTHORIZATION_REQUIRED.to_string(),
        family: "minimax-h3".to_string(),
        message: ModelActivationError.to_string(),
        license_url: MINIMAX_H3_LICENSE_URL.to_string(),
        authorization_url: MINIMAX_H3_AUTHORIZATION_ISSUE_URL.to_string(),
    }
}

/// Return the activation state for a model identity and its resolved family.
///
/// The family is required when the public identifier is opaque (for example a
/// `cv:` catalog ID). Callers that have resolved catalog metadata must pass it.
pub fn model_activation(identifier: &str, family: Option<&str>) -> ModelActivation {
    if is_reviewed_minimax_h3_model(identifier) {
        ModelActivation::Available
    } else if is_minimax_h3_identity(identifier) || family.is_some_and(is_minimax_h3_identity) {
        ModelActivation::ComplianceGated
    } else {
        ModelActivation::Available
    }
}

/// Whether an identity may appear in ordinary model-discovery surfaces.
///
/// This is a convenience view over [`model_activation`], not a second policy
/// table. Catalog-family lists and other non-error-producing discovery paths
/// use it so a compliance-gated family cannot leak through static taxonomy
/// while mutating ingress paths continue to use [`require_model_activation`].
pub fn model_activation_available(identifier: &str, family: Option<&str>) -> bool {
    model_activation(identifier, family) == ModelActivation::Available
}

/// Return whether a model may be discovered and acquired from its pinned
/// upstream source.
///
/// MiniMax H3's reviewed authorization permits upstream-direct downloads and
/// local storage, but execution remains independently gated by
/// [`model_activation`]. Keeping those authorities separate prevents a
/// downloadable checkpoint from becoming an implicit runtime approval.
pub fn model_acquisition(identifier: &str, family: Option<&str>) -> ModelActivation {
    if is_reviewed_minimax_h3_acquisition_identity(identifier) {
        ModelActivation::Available
    } else if cfg!(feature = "h3")
        && (is_minimax_h3_identity(identifier) || family.is_some_and(is_minimax_h3_identity))
    {
        ModelActivation::ComplianceGated
    } else {
        model_activation(identifier, family)
    }
}

pub fn model_acquisition_available(identifier: &str, family: Option<&str>) -> bool {
    model_acquisition(identifier, family) == ModelActivation::Available
}

pub fn require_model_acquisition(
    identifier: &str,
    family: Option<&str>,
) -> Result<(), ModelActivationError> {
    match model_acquisition(identifier, family) {
        ModelActivation::Available => Ok(()),
        ModelActivation::ComplianceGated => Err(ModelActivationError),
    }
}

pub fn require_model_activation(
    identifier: &str,
    family: Option<&str>,
) -> Result<(), ModelActivationError> {
    match model_activation(identifier, family) {
        ModelActivation::Available => Ok(()),
        ModelActivation::ComplianceGated => Err(ModelActivationError),
    }
}

/// Validate one exact source-controlled manifest for runtime use.
///
/// A reviewed H3 manifest deliberately contains raw upstream repository and
/// filename identities that remain forbidden as caller-selected models. Those
/// locators are safe only when the name, family, and complete pinned file set
/// exactly match the static registry. Value identity is required because
/// durable queue replay reconstructs the reviewed manifest across processes;
/// pointer identity cannot survive a restart.
pub fn is_exact_registered_manifest(manifest: &crate::manifest::ModelManifest) -> bool {
    crate::manifest::find_manifest(&manifest.name).is_some_and(|registered| {
        registered.name == manifest.name
            && registered.family == manifest.family
            && registered.files == manifest.files
    })
}

pub fn require_registered_manifest_activation(
    manifest: &crate::manifest::ModelManifest,
) -> Result<(), ModelActivationError> {
    require_model_activation(&manifest.name, Some(&manifest.family))?;
    let contains_gated_source = manifest.files.iter().any(|file| {
        require_model_activation(&file.hf_repo, Some(&manifest.family)).is_err()
            || require_model_activation(&file.hf_filename, Some(&manifest.family)).is_err()
    });
    if !contains_gated_source {
        return Ok(());
    }
    if is_reviewed_minimax_h3_model(&manifest.name) && is_exact_registered_manifest(manifest) {
        Ok(())
    } else {
        Err(ModelActivationError)
    }
}

/// Return the activation state for one concrete model artifact path.
///
/// `artifact_root` is a caller-owned trust boundary such as
/// [`crate::Config::resolved_models_dir`]. Its own path components describe
/// storage placement, not model identity, so only the artifact-relative suffix
/// is inspected when `path` is contained by that root. Paths outside the root
/// remain fail-closed and are inspected in full.
///
/// This distinction matters when an operator deliberately names `MOLD_HOME`
/// after a UAT target (for example `/.../minimax-h3`): an ordinary FLUX file
/// below that root is not H3, while a nested `MiniMax-H3/...` artifact still is.
pub fn model_artifact_activation(
    path: &Path,
    artifact_root: Option<&Path>,
    family: Option<&str>,
) -> ModelActivation {
    let identity_path = artifact_root
        .and_then(|root| path.strip_prefix(root).ok())
        .unwrap_or(path);
    let is_h3 = is_minimax_h3_identity(&identity_path.to_string_lossy())
        || family.is_some_and(is_minimax_h3_identity);
    if cfg!(feature = "h3") && is_h3 {
        ModelActivation::Available
    } else if is_h3 {
        ModelActivation::ComplianceGated
    } else {
        ModelActivation::Available
    }
}

pub fn require_model_artifact_activation(
    path: &Path,
    artifact_root: Option<&Path>,
    family: Option<&str>,
) -> Result<(), ModelActivationError> {
    match model_artifact_activation(path, artifact_root, family) {
        ModelActivation::Available => Ok(()),
        ModelActivation::ComplianceGated => Err(ModelActivationError),
    }
}

fn is_minimax_h3_identity(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().chars().fold(
        String::with_capacity(value.len()),
        |mut out, ch| {
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
            } else if !out.ends_with('-') {
                out.push('-');
            }
            out
        },
    );
    let needle = "minimax-h3";
    let separated_alias = normalized.match_indices(needle).any(|(start, _)| {
        let before = normalized[..start].chars().next_back();
        let after = normalized[start + needle.len()..].chars().next();
        before.is_none_or(|ch| !ch.is_ascii_alphanumeric())
            && after.is_none_or(|ch| !ch.is_ascii_alphanumeric())
    });
    separated_alias || normalized.split('-').any(is_minimax_h3_compact_alias)
}

fn is_reviewed_minimax_h3_acquisition_identity(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    minimax_h3::resolve_model_name(&normalized).is_some()
}

pub fn is_reviewed_minimax_h3_model(value: &str) -> bool {
    minimax_h3::is_reviewed_compact_model(value)
}

fn is_minimax_h3_compact_alias(token: &str) -> bool {
    fn has_class_suffix(value: &str, prefix: &str) -> bool {
        value.strip_prefix(prefix).is_some_and(|suffix| {
            suffix
                .chars()
                .next()
                .is_none_or(|ch| ch.is_ascii_alphabetic())
        })
    }

    has_class_suffix(token, "minimaxh3") || has_class_suffix(token, "autoencoderklminimaxh3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "h3"))]
    fn minimax_h3_aliases_and_task_variants_are_compliance_gated() {
        for identifier in [
            "minimax-h3",
            "MiniMax H3",
            "MiniMax-H3-FL2VA",
            "minimax_h3_ref2va:bf16",
            "MiniMaxH3",
            "MiniMaxH3Scheduler",
            "MiniMaxH3Transformer3DModel",
            "AutoencoderKLMiniMaxH3",
            "hf:MiniMaxAI/MiniMax-H3",
            "hf:MiniMaxAI/MiniMaxH3",
            "hf:Comfy-Org/MiniMax-H3",
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main",
        ] {
            assert_eq!(
                model_activation(identifier, None),
                ModelActivation::ComplianceGated,
                "{identifier}"
            );
        }

        assert_eq!(
            model_activation("cv:42", Some("MiniMax-H3")),
            ModelActivation::ComplianceGated
        );
    }

    #[test]
    fn unrelated_models_and_h3_lookalikes_remain_available() {
        for identifier in [
            "flux-dev:q8",
            "my-h3-model",
            "h3",
            "minimax-h30",
            "minimaxh30",
            "notminimax-h3",
            "notminimaxh3",
        ] {
            assert_eq!(
                model_activation(identifier, None),
                ModelActivation::Available,
                "{identifier}"
            );
        }
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn discovery_availability_is_a_view_of_the_activation_authority() {
        assert!(!model_activation_available(
            "minimax-h3",
            Some("minimax-h3")
        ));
        assert!(model_activation_available("flux", Some("flux")));
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn reviewed_h3_models_are_ordinary_activation_identities() {
        for identifier in [
            "minimax-h3-fl2va:comfy-pruned-int8",
            "minimax-h3-ref2va:comfy-pruned-int8",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
        ] {
            assert_eq!(
                model_acquisition(identifier, Some("minimax-h3")),
                ModelActivation::Available,
                "{identifier}"
            );
            assert_eq!(
                model_activation(identifier, Some("minimax-h3")),
                ModelActivation::Available,
                "{identifier}"
            );
        }

        assert_eq!(
            model_activation("minimax-h3", Some("minimax-h3")),
            ModelActivation::ComplianceGated
        );

        for official in [minimax_h3::FL2VA_OFFICIAL, minimax_h3::REF2VA_OFFICIAL] {
            assert_eq!(
                model_acquisition(official, Some("minimax-h3")),
                ModelActivation::Available,
                "{official}"
            );
            assert_eq!(
                model_activation(official, Some("minimax-h3")),
                ModelActivation::ComplianceGated,
                "{official}"
            );
        }

        for unreviewed in [
            "hf:Comfy-Org/MiniMax-H3",
            "minimax-h3:custom",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step",
            "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step",
            "transformer/high_noise.safetensors",
        ] {
            assert_eq!(
                model_acquisition(unreviewed, Some("minimax-h3")),
                ModelActivation::ComplianceGated,
                "{unreviewed}"
            );
        }
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn artifact_policy_ignores_h3_named_storage_root_but_not_nested_identity() {
        let artifact_root = Path::new("/Volumes/ExternalStorage/mold-uat/minimax-h3/models");
        let flux = artifact_root.join("flux-dev/transformer/model.safetensors");
        let h3 = artifact_root.join("custom/MiniMax-H3/transformer/model.safetensors");

        assert_eq!(
            model_artifact_activation(&flux, Some(artifact_root), Some("flux")),
            ModelActivation::Available
        );
        assert_eq!(
            model_artifact_activation(&h3, Some(artifact_root), Some("custom")),
            ModelActivation::ComplianceGated
        );
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn artifact_policy_inspects_full_paths_outside_its_trusted_root() {
        let artifact_root = Path::new("/srv/mold/models");
        let external = Path::new("/Volumes/MiniMax-H3/weights.safetensors");

        assert_eq!(
            model_artifact_activation(external, Some(artifact_root), None),
            ModelActivation::ComplianceGated
        );
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn rejection_is_stable_and_does_not_echo_the_supplied_identifier() {
        let secretish_identifier = "hf:MiniMaxAI/MiniMax-H3?token=do-not-echo";
        let error = require_model_activation(secretish_identifier, None).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(MINIMAX_H3_AUTHORIZATION_REQUIRED));
        assert!(message.contains(MINIMAX_H3_AUTHORIZATION_ISSUE_URL));
        assert!(!message.contains(secretish_identifier));
    }

    #[test]
    #[cfg(not(feature = "h3"))]
    fn capabilities_do_not_advertise_a_family_wide_h3_restriction() {
        let capabilities = model_access_capabilities();
        assert!(capabilities.restrictions.is_empty());

        let round_trip: ModelAccessCapabilities =
            serde_json::from_str(&serde_json::to_string(&capabilities).unwrap()).unwrap();
        assert_eq!(round_trip, capabilities);
    }

    #[test]
    #[cfg(feature = "h3")]
    fn public_h3_feature_activates_only_the_exact_runtime_partition() {
        assert_eq!(
            model_activation(minimax_h3::FL2VA_COMFY, Some("minimax-h3")),
            ModelActivation::Available
        );
        for identifier in [
            "minimax-h3",
            "hf:Comfy-Org/MiniMax-H3",
            "MiniMaxH3Transformer3DModel",
        ] {
            assert_eq!(
                model_activation(identifier, Some("minimax-h3")),
                ModelActivation::ComplianceGated,
                "{identifier}"
            );
        }
        assert_eq!(
            model_activation(minimax_h3::REF2VA_COMFY, Some("minimax-h3")),
            ModelActivation::Available
        );
        assert!(model_access_capabilities().restrictions.is_empty());
        assert_eq!(
            model_artifact_activation(
                Path::new("/models/minimax-h3/transformer.safetensors"),
                Some(Path::new("/models")),
                Some("minimax-h3")
            ),
            ModelActivation::Available
        );
    }

    #[test]
    #[cfg(feature = "h3")]
    fn public_h3_feature_keeps_acquisition_on_reviewed_manifests() {
        for reviewed in [
            "minimax-h3-fl2va:comfy-pruned-int8",
            "minimax-h3-ref2va:comfy-pruned-int8",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
        ] {
            assert_eq!(
                model_acquisition(reviewed, Some("minimax-h3")),
                ModelActivation::Available
            );
            assert_eq!(
                model_activation(reviewed, Some("minimax-h3")),
                ModelActivation::Available
            );
        }
        for official in [minimax_h3::FL2VA_OFFICIAL, minimax_h3::REF2VA_OFFICIAL] {
            assert_eq!(
                model_acquisition(official, Some("minimax-h3")),
                ModelActivation::Available
            );
            assert_eq!(
                model_activation(official, Some("minimax-h3")),
                ModelActivation::ComplianceGated
            );
        }
        for unreviewed in [
            "hf:Comfy-Org/MiniMax-H3",
            "minimax-h3:custom",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step",
        ] {
            assert_eq!(
                model_acquisition(unreviewed, Some("minimax-h3")),
                ModelActivation::ComplianceGated
            );
        }
    }

    #[test]
    fn only_an_exact_registered_h3_manifest_may_bind_its_raw_sources() {
        let registered = crate::manifest::find_manifest(minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P)
            .expect("reviewed H3 Turbo manifest");
        require_registered_manifest_activation(registered)
            .expect("the exact source-controlled manifest must activate");

        let copied = registered.clone();
        require_registered_manifest_activation(&copied)
            .expect("durable replay must preserve exact manifest authority by value");
        let mut changed = copied;
        changed.files[0].hf_filename.push_str(".changed");
        assert!(require_registered_manifest_activation(&changed).is_err());
        assert!(
            require_model_activation("hf:Comfy-Org/MiniMax-H3", Some(minimax_h3::FAMILY),).is_err()
        );
    }
}
