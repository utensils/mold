use std::fmt;
use std::path::Path;

use crate::minimax_h3;

use serde::{Deserialize, Serialize};

/// Stable machine-readable reason returned while MiniMax H3 authorization is absent.
pub const MINIMAX_H3_AUTHORIZATION_REQUIRED: &str = "MINIMAX_H3_AUTHORIZATION_REQUIRED";

/// Stable machine-readable reason returned for a model mold may download and
/// store but has no engine arm for.
///
/// Deliberately distinct from [`MINIMAX_H3_AUTHORIZATION_REQUIRED`]: that one
/// is a licensing statement carrying a license URL and an authorization
/// record, and reporting it for unwritten code tells the user to go read a
/// legal document about a problem that is ours.
pub const MINIMAX_H3_RUNTIME_UNAVAILABLE: &str = "MINIMAX_H3_RUNTIME_UNAVAILABLE";
pub const LTX25_GGUF_RUNTIME_UNAVAILABLE: &str = "LTX25_GGUF_RUNTIME_UNAVAILABLE";

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
    /// Acquirable and storable, but this build has no runtime for it.
    ///
    /// Three things land here, and the carried reason is what tells them
    /// apart: the pinned `official-bf16` references and the pruned NVFP4
    /// compact layout (no engine arm for the weight layout), every Ref2VA
    /// identity (no qualified route on a released build), and *every* H3
    /// identity on a binary compiled without the `h3` feature. `mold pull`,
    /// inventory, repair, and `mold rm` all work; only execution is refused.
    RuntimeUnavailable(minimax_h3::RuntimeUnavailableReason),
    /// LTX-2.5 GGUF acquisition is supported, but native QTensor execution is not.
    Ltx25GgufRuntimeUnavailable,
}

/// Why an activation was refused. Carried by [`ModelActivationError`] so the
/// message and the HTTP status can tell a licensing refusal apart from a
/// missing implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationRefusal {
    ComplianceGated,
    RuntimeUnavailable(minimax_h3::RuntimeUnavailableReason),
    Ltx25GgufRuntimeUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelActivationError(pub ActivationRefusal);

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
        match self.0 {
            // Byte-for-byte the message this type has always produced.
            ActivationRefusal::ComplianceGated => write!(
                formatter,
                "MiniMax H3 support is compliance-gated and is not activated in this build \
                 ({MINIMAX_H3_AUTHORIZATION_REQUIRED}). See {MINIMAX_H3_AUTHORIZATION_ISSUE_URL}"
            ),
            // Deliberately carries no license or authorization URL: nothing
            // about this refusal is the user's to resolve. The sentence comes
            // from `minimax_h3::RuntimeUnavailableReason`, the single
            // authority the `/api/models` row also publishes, so the refusal
            // and the row can never name different obstacles.
            ActivationRefusal::RuntimeUnavailable(reason) => write!(
                formatter,
                "{} ({MINIMAX_H3_RUNTIME_UNAVAILABLE})",
                reason.message()
            ),
            ActivationRefusal::Ltx25GgufRuntimeUnavailable => write!(
                formatter,
                "{} ({LTX25_GGUF_RUNTIME_UNAVAILABLE})",
                crate::ltx25_manifest::GGUF_RUNTIME_UNAVAILABLE_REASON
            ),
        }
    }
}

impl std::error::Error for ModelActivationError {}

impl ModelActivationError {
    pub const fn refusal(self) -> ActivationRefusal {
        self.0
    }

    /// The family-wide access restriction this refusal publishes, if any.
    ///
    /// `None` for [`ActivationRefusal::RuntimeUnavailable`]: a missing engine
    /// arm is a per-identity fact about this build, and
    /// [`ModelAccessRestriction`] is family-scoped. Publishing one would gate
    /// every H3 identity in every client.
    pub fn restriction(self) -> Option<ModelAccessRestriction> {
        match self.0 {
            ActivationRefusal::ComplianceGated => Some(minimax_h3_restriction()),
            ActivationRefusal::RuntimeUnavailable(_) => None,
            ActivationRefusal::Ltx25GgufRuntimeUnavailable => None,
        }
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
        message: ModelActivationError(ActivationRefusal::ComplianceGated).to_string(),
        license_url: MINIMAX_H3_LICENSE_URL.to_string(),
        authorization_url: MINIMAX_H3_AUTHORIZATION_ISSUE_URL.to_string(),
    }
}

/// Return the activation state for a model identity and its resolved family.
///
/// The family is required when the public identifier is opaque (for example a
/// `cv:` catalog ID). Callers that have resolved catalog metadata must pass it.
pub fn model_activation(identifier: &str, family: Option<&str>) -> ModelActivation {
    let canonical = crate::manifest::resolve_model_name(identifier);
    if crate::ltx25_manifest::is_gguf_manifest(&canonical) {
        ModelActivation::Ltx25GgufRuntimeUnavailable
    } else if is_reviewed_minimax_h3_model(identifier) {
        // Reviewed for acquisition, which is a separate authority from
        // execution. Whether *this* build can run it is
        // `minimax_h3::model_runtime_availability`'s answer and nothing
        // else's: Ref2VA has no qualified route on any released binary, and
        // only the sm89 recipe compiles the engine at all (#1276).
        minimax_h3_runtime_activation(identifier)
    } else if is_pinned_unrunnable_minimax_h3_identity(identifier) {
        // A pinned H3 manifest identity mold may download but cannot execute:
        // the `official-bf16` qualification references and the pruned NVFP4
        // compact layout. Refusing these as compliance-gated would hand the
        // user a license URL for a missing engine arm.
        minimax_h3_runtime_activation(identifier)
    } else if is_minimax_h3_identity(identifier) || family.is_some_and(is_minimax_h3_identity) {
        ModelActivation::ComplianceGated
    } else {
        ModelActivation::Available
    }
}

/// Project the H3 runtime authority onto the activation vocabulary.
///
/// Fails closed on both halves: an identity the runtime authority cannot
/// resolve reports [`minimax_h3::RuntimeUnavailableReason::UnsupportedLayout`]
/// there, and this reads it verbatim rather than inventing a second answer.
fn minimax_h3_runtime_activation(identifier: &str) -> ModelActivation {
    match minimax_h3::model_runtime_availability(identifier) {
        minimax_h3::RuntimeAvailability::Available => ModelActivation::Available,
        minimax_h3::RuntimeAvailability::Unavailable(reason) => {
            ModelActivation::RuntimeUnavailable(reason)
        }
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
    let canonical = crate::manifest::resolve_model_name(identifier);
    if crate::ltx25_manifest::is_gguf_manifest(&canonical)
        || is_reviewed_minimax_h3_acquisition_identity(identifier)
    {
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
        ModelActivation::ComplianceGated => {
            Err(ModelActivationError(ActivationRefusal::ComplianceGated))
        }
        ModelActivation::RuntimeUnavailable(reason) => Err(ModelActivationError(
            ActivationRefusal::RuntimeUnavailable(reason),
        )),
        ModelActivation::Ltx25GgufRuntimeUnavailable => Err(ModelActivationError(
            ActivationRefusal::Ltx25GgufRuntimeUnavailable,
        )),
    }
}

pub fn require_model_activation(
    identifier: &str,
    family: Option<&str>,
) -> Result<(), ModelActivationError> {
    match model_activation(identifier, family) {
        ModelActivation::Available => Ok(()),
        ModelActivation::ComplianceGated => {
            Err(ModelActivationError(ActivationRefusal::ComplianceGated))
        }
        ModelActivation::RuntimeUnavailable(reason) => Err(ModelActivationError(
            ActivationRefusal::RuntimeUnavailable(reason),
        )),
        ModelActivation::Ltx25GgufRuntimeUnavailable => Err(ModelActivationError(
            ActivationRefusal::Ltx25GgufRuntimeUnavailable,
        )),
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
        Err(ModelActivationError(ActivationRefusal::ComplianceGated))
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
        ModelActivation::ComplianceGated => {
            Err(ModelActivationError(ActivationRefusal::ComplianceGated))
        }
        // `model_artifact_activation` classifies a path, never a model
        // identity, so it has no way to observe a missing engine arm.
        ModelActivation::RuntimeUnavailable(reason) => Err(ModelActivationError(
            ActivationRefusal::RuntimeUnavailable(reason),
        )),
        ModelActivation::Ltx25GgufRuntimeUnavailable => Err(ModelActivationError(
            ActivationRefusal::Ltx25GgufRuntimeUnavailable,
        )),
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

/// Whether this is an exact pinned H3 manifest identity that mold cannot run.
///
/// Exactness is the point. `is_reviewed_minimax_h3_acquisition_identity`
/// resolves aliases, so `minimax-h3` answers `true` there — but that alias
/// names the reviewed, runnable compact FL2VA model, and reporting it as
/// "no runtime" would be false. Requiring the normalized input to equal its
/// own canonical form keeps aliases on the existing compliance path, which
/// is the same not-alias-resolving rule `is_reviewed_compact_model` uses.
///
/// Public because it is also the one authority the server's private H3
/// ingress boundary asks before it claims a request (#1354). That boundary
/// runs ahead of [`model_activation`], so it has to know which identities
/// this function routes here — otherwise it answers its own partition
/// refusal for a checkpoint whose `/api/models` row promises a runtime
/// sentence, which is #1276 re-appearing through the private path.
pub fn is_pinned_unrunnable_minimax_h3_identity(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    minimax_h3::resolve_model_name(&normalized).is_some_and(|canonical| canonical == normalized)
        && !is_reviewed_minimax_h3_model(&normalized)
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
    fn ltx25_gguf_acquires_but_refuses_execution_before_queueing() {
        for identifier in [
            crate::ltx25_manifest::DISTILLED_Q3_K_S,
            crate::ltx25_manifest::DISTILLED_Q3,
            crate::ltx25_manifest::DISTILLED_Q4_K_S,
            crate::ltx25_manifest::DISTILLED_Q4,
            crate::ltx25_manifest::DISTILLED_Q5,
            crate::ltx25_manifest::DISTILLED_Q6,
            crate::ltx25_manifest::DISTILLED_Q8,
        ] {
            assert_eq!(
                model_acquisition(identifier, Some(crate::ltx25_manifest::FAMILY)),
                ModelActivation::Available,
                "{identifier}"
            );
            assert_eq!(
                model_activation(identifier, Some(crate::ltx25_manifest::FAMILY)),
                ModelActivation::Ltx25GgufRuntimeUnavailable,
                "{identifier}"
            );
            let error = require_model_activation(identifier, Some(crate::ltx25_manifest::FAMILY))
                .expect_err("GGUF execution must be refused before admission");
            assert_eq!(
                error.refusal(),
                ActivationRefusal::Ltx25GgufRuntimeUnavailable
            );
            assert!(error.to_string().contains(LTX25_GGUF_RUNTIME_UNAVAILABLE));
        }
    }

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
        use minimax_h3::RuntimeUnavailableReason;

        // Acquisition and execution are separate authorities, and #1276 is
        // the case where they disagree: every reviewed identity still
        // downloads on this build, and none of them runs, because this build
        // was compiled without the engine.
        for identifier in [
            "minimax-h3-fl2va:comfy-pruned-int8",
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
                ModelActivation::RuntimeUnavailable(RuntimeUnavailableReason::EngineNotBuilt),
                "{identifier}"
            );
        }

        // Ref2VA downloads on every build and executes wherever the engine
        // is linked (#825). The refusal a build without it gives names the
        // build recipe, because that is the only obstacle left.
        assert_eq!(
            model_acquisition("minimax-h3-ref2va:comfy-pruned-int8", Some("minimax-h3")),
            ModelActivation::Available
        );
        assert_eq!(
            model_activation("minimax-h3-ref2va:comfy-pruned-int8", Some("minimax-h3")),
            if minimax_h3::engine_is_built() {
                ModelActivation::Available
            } else {
                ModelActivation::RuntimeUnavailable(RuntimeUnavailableReason::EngineNotBuilt)
            }
        );

        // An alias names the reviewed compact model, so it stays on the
        // compliance path rather than claiming mold has no runtime for it.
        assert_eq!(
            model_activation("minimax-h3", Some("minimax-h3")),
            ModelActivation::ComplianceGated
        );

        // Pinned identities mold may download and store but has no engine
        // arm for. They are refused as `RuntimeUnavailable`, never as a
        // licensing problem: nothing about the refusal is the user's to
        // resolve, and the message carries no license or authorization URL.
        for unrunnable in [
            minimax_h3::FL2VA_OFFICIAL,
            minimax_h3::REF2VA_OFFICIAL,
            minimax_h3::FL2VA_COMFY_NVFP4,
            minimax_h3::REF2VA_COMFY_NVFP4,
        ] {
            assert_eq!(
                model_acquisition(unrunnable, Some("minimax-h3")),
                ModelActivation::Available,
                "{unrunnable}"
            );
            assert_eq!(
                model_activation(unrunnable, Some("minimax-h3")),
                ModelActivation::RuntimeUnavailable(RuntimeUnavailableReason::UnsupportedLayout),
                "{unrunnable}"
            );
            let error = require_model_activation(unrunnable, Some("minimax-h3"))
                .expect_err("execution must be refused");
            assert_eq!(
                error.refusal(),
                ActivationRefusal::RuntimeUnavailable(RuntimeUnavailableReason::UnsupportedLayout)
            );
            assert!(error.restriction().is_none(), "{unrunnable}");
            let message = error.to_string();
            assert!(
                message.contains(MINIMAX_H3_RUNTIME_UNAVAILABLE),
                "{unrunnable}: {message}"
            );
            assert!(
                !message.contains(MINIMAX_H3_AUTHORIZATION_REQUIRED)
                    && !message.contains(MINIMAX_H3_LICENSE_URL)
                    && !message.contains(MINIMAX_H3_AUTHORIZATION_ISSUE_URL),
                "a missing engine arm must not be reported as a licensing refusal: {message}"
            );
        }

        for unreviewed in [
            "hf:Comfy-Org/MiniMax-H3",
            "minimax-h3:custom",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step",
            "minimax-h3-ref2va:comfy-pruned-int8-turbo-8step",
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
        // Ref2VA is a reviewed acquisition identity whose ordered-reference
        // route was qualified in #825, so a build carrying the engine runs
        // it and a build without one refuses as a missing implementation,
        // never as a licensing problem (#1276).
        assert_eq!(
            model_activation(minimax_h3::REF2VA_COMFY, Some("minimax-h3")),
            if minimax_h3::engine_is_built() {
                ModelActivation::Available
            } else {
                ModelActivation::RuntimeUnavailable(
                    minimax_h3::RuntimeUnavailableReason::EngineNotBuilt,
                )
            }
        );
        assert_eq!(
            model_acquisition(minimax_h3::REF2VA_COMFY, Some("minimax-h3")),
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
        // The binding question this test owns is "may this manifest's raw
        // upstream locators be used at all", which only an exact registered
        // manifest may answer yes to. On a build with no engine the answer is
        // separately no for a reason that has nothing to do with binding
        // (#1276), so assert the *difference* between the exact manifest and
        // a mutated one rather than an unconditional Ok.
        let exact = require_registered_manifest_activation(registered);
        if minimax_h3::engine_is_built() {
            exact.expect("the exact source-controlled manifest must activate");
        } else {
            assert_eq!(
                exact.unwrap_err().refusal(),
                ActivationRefusal::RuntimeUnavailable(
                    minimax_h3::RuntimeUnavailableReason::EngineNotBuilt
                )
            );
        }

        let copied = registered.clone();
        assert_eq!(
            require_registered_manifest_activation(&copied).map_err(ModelActivationError::refusal),
            exact.map_err(ModelActivationError::refusal),
            "durable replay must preserve exact manifest authority by value"
        );
        let mut changed = copied;
        changed.files[0].hf_filename.push_str(".changed");
        // The binding rule itself is value identity against the registry, and
        // it holds on every build — a runtime answer never widens it.
        assert!(is_exact_registered_manifest(registered));
        assert!(!is_exact_registered_manifest(&changed));
        require_registered_manifest_activation(&changed)
            .expect_err("a mutated manifest may never bind raw upstream sources");
        if minimax_h3::engine_is_built() {
            assert_eq!(
                require_registered_manifest_activation(&changed)
                    .unwrap_err()
                    .refusal(),
                ActivationRefusal::ComplianceGated
            );
        }
        assert!(
            require_model_activation("hf:Comfy-Org/MiniMax-H3", Some(minimax_h3::FAMILY),).is_err()
        );
    }
}
