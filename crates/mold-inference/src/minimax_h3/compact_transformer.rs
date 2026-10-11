//! Per-ARTIFACT selection of the compact DiT one admitted H3 identity runs.
//!
//! The compact runtime used to pick its transformer from the TASK alone:
//! FL2VA meant Comfy-Org's FL2VA file, Ref2VA meant Comfy-Org's Ref2VA file.
//! A distilled base tier such as FastVideo's FastH3 V2 breaks that pairing —
//! it executes the FL2VA task face (same conditioner, VAEs, presentation, and
//! conditioning tensors, so the conditioner cache key stays shared) on a
//! different published DiT. [`H3CompactTransformerSelection`] is the one
//! answer every storage, qualification, admission, and reopen site asks
//! instead of re-deriving a task -> artifact constant.
//!
//! Selection is DATA: an admitted model's own manifest names its Transformer
//! row, and that row's exact `(hf_repo, hf_filename)` must match one row of
//! [`REVIEWED_COMPACT_TRANSFORMERS`] whose size and SHA-256 pins agree. A
//! Turbo tag's manifest carries its base stack's Transformer row, so it
//! resolves to the base selection exactly as before, and every pre-existing
//! storage, qualification, and frozen identity stays byte-identical. The
//! FastH3 manifest tag (`mold_core::minimax_h3::FL2VA_FASTH3_V2_INT8`) names
//! the FastVideo file in its own Transformer row and selects it here.

use anyhow::{anyhow, bail, Result};
use mold_candle::minimax_h3::{
    H3ComfyPublishedArtifact, H3TransformerTask, H3_FASTH3_V2_INT8_SOURCE_PATH,
};
use mold_core::manifest::{find_manifest, ModelComponent, ModelFile, ModelManifest};
use mold_core::minimax_h3::{self as contract, Task};

pub(crate) const FL2VA_TRANSFORMER_SOURCE: &str =
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
pub(crate) const REF2VA_TRANSFORMER_SOURCE: &str =
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors";

/// Every published compact DiT the H3 runtime executes, keyed on the exact
/// repository and repository-relative path a manifest Transformer row names.
const REVIEWED_COMPACT_TRANSFORMERS: &[(&str, &str, H3ComfyPublishedArtifact)] = &[
    (
        contract::COMFY_REPO,
        FL2VA_TRANSFORMER_SOURCE,
        H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
    ),
    (
        contract::COMFY_REPO,
        REF2VA_TRANSFORMER_SOURCE,
        H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
    ),
    (
        contract::FASTVIDEO_FASTH3_REPO,
        H3_FASTH3_V2_INT8_SOURCE_PATH,
        H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot,
    ),
];

pub(crate) const fn candle_task(task: Task) -> H3TransformerTask {
    match task {
        Task::Fl2va => H3TransformerTask::T2VaFl2Va,
        Task::Ref2va => H3TransformerTask::Ref2Va,
    }
}

/// The compact DiT one admitted H3 identity executes, and where its file is
/// declared. `Copy` and payload-free: it names an artifact, it does not
/// authenticate one — storage resolution, qualification, and the opened
/// checkpoint still enforce the pinned size, digest, and header.
#[derive(Clone, Copy)]
pub(crate) struct H3CompactTransformerSelection {
    task: Task,
    published_artifact: H3ComfyPublishedArtifact,
    /// The manifest whose `Transformer` row names the artifact file: the
    /// task's base compact manifest for Comfy-Org's DiTs, the admitted tag's
    /// own manifest for a substituted DiT.
    manifest_model: &'static str,
    /// The resolved manifest for a substituted DiT (`None` for a base
    /// selection, which re-finds its pinned base manifest by name). Not part
    /// of equality — `manifest_model` names it.
    manifest: Option<&'static ModelManifest>,
    source: &'static str,
}

impl PartialEq for H3CompactTransformerSelection {
    fn eq(&self, other: &Self) -> bool {
        self.task == other.task
            && self.published_artifact == other.published_artifact
            && self.manifest_model == other.manifest_model
            && self.source == other.source
    }
}

impl Eq for H3CompactTransformerSelection {}

impl std::fmt::Debug for H3CompactTransformerSelection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3CompactTransformerSelection")
            .field("task", &self.task)
            .field("published_artifact", &self.published_artifact)
            .field("manifest_model", &self.manifest_model)
            .field("source", &self.source)
            .finish()
    }
}

impl H3CompactTransformerSelection {
    /// The Comfy-Org compact DiT of `task` — what every reviewed identity
    /// executed before per-artifact selection existed.
    pub(crate) const fn base_for_task(task: Task) -> Self {
        match task {
            Task::Fl2va => Self {
                task,
                published_artifact: H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
                manifest_model: contract::FL2VA_COMFY,
                manifest: None,
                source: FL2VA_TRANSFORMER_SOURCE,
            },
            Task::Ref2va => Self {
                task,
                published_artifact: H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
                manifest_model: contract::REF2VA_COMFY,
                manifest: None,
                source: REF2VA_TRANSFORMER_SOURCE,
            },
        }
    }

    /// Resolve the DiT an admitted (or frozen) model identity executes, from
    /// that identity's own manifest. Admission and reopen both call this on
    /// the request's model, so a frozen attempt reopens the same artifact it
    /// was admitted on; the frozen component digests fence any disagreement.
    pub(crate) fn for_model(model: &str) -> Result<Self> {
        let canonical = contract::resolve_model_name(model)
            .ok_or_else(|| anyhow!("MiniMax H3 compact transformer selection names {model:?}"))?;
        let manifest = find_manifest(canonical)
            .ok_or_else(|| anyhow!("MiniMax H3 {canonical} has no pinned manifest"))?;
        Self::for_manifest(manifest)
    }

    /// The selection a pinned H3 manifest declares through its Transformer row.
    pub(crate) fn for_manifest(manifest: &'static ModelManifest) -> Result<Self> {
        let contract = contract::manifest_contract(manifest)
            .ok_or_else(|| anyhow!("MiniMax H3 manifest {} has no contract", manifest.name))?;
        if manifest.family != contract::FAMILY {
            bail!("MiniMax H3 compact transformer selection requires an H3 manifest")
        }
        let task = contract.task;
        let file = transformer_row(manifest)?;
        let (_, source, artifact) = REVIEWED_COMPACT_TRANSFORMERS
            .iter()
            .copied()
            .find(|(repo, path, _)| file.hf_repo == *repo && file.hf_filename == *path)
            .ok_or_else(|| {
                anyhow!(
                    "MiniMax H3 manifest {} names an unreviewed compact transformer {}/{}",
                    manifest.name,
                    file.hf_repo,
                    file.hf_filename
                )
            })?;
        if artifact.task() != candle_task(task)
            || file.size_bytes != artifact.file_bytes()
            || file.sha256 != Some(artifact.content_sha256())
        {
            bail!(
                "MiniMax H3 manifest {} pins {} differently from its {artifact:?} authority",
                manifest.name,
                file.hf_filename
            )
        }
        let base = Self::base_for_task(task);
        if artifact == base.published_artifact {
            // A base or Turbo tag: resolve storage exactly where it always
            // did — the task's base manifest — so storage, qualification,
            // and frozen identities are unchanged.
            return Ok(base);
        }
        Ok(Self {
            task,
            published_artifact: artifact,
            manifest_model: manifest.name.as_str(),
            manifest: Some(manifest),
            source,
        })
    }

    pub(crate) const fn task(&self) -> Task {
        self.task
    }

    pub(crate) const fn published_artifact(&self) -> H3ComfyPublishedArtifact {
        self.published_artifact
    }

    pub(crate) const fn transformer_task(&self) -> H3TransformerTask {
        self.published_artifact.task()
    }

    pub(crate) const fn manifest_model(&self) -> &'static str {
        self.manifest_model
    }

    pub(crate) const fn source(&self) -> &'static str {
        self.source
    }

    /// True for the task's Comfy-Org DiT, i.e. no substitution.
    pub(crate) fn is_task_base(&self) -> bool {
        *self == Self::base_for_task(self.task)
    }

    /// The exact manifest and Transformer row this selection resolves from.
    pub(crate) fn transformer_file(&self) -> Result<(&'static ModelManifest, &'static ModelFile)> {
        let manifest = match self.manifest {
            Some(manifest) => manifest,
            None => find_manifest(self.manifest_model).ok_or_else(|| {
                anyhow!(
                    "MiniMax H3 transformer manifest {} is not pinned",
                    self.manifest_model
                )
            })?,
        };
        let file = transformer_row(manifest)?;
        if file.hf_filename != self.source {
            bail!(
                "MiniMax H3 transformer manifest {} no longer names {}",
                self.manifest_model,
                self.source
            )
        }
        if Self::for_manifest(manifest)? != *self {
            bail!("MiniMax H3 transformer selection differs from its manifest authority")
        }
        Ok((manifest, file))
    }

    /// Whether an opened/inspected artifact is the one this selection names.
    pub(crate) fn require_artifact(&self, artifact: H3ComfyPublishedArtifact) -> Result<()> {
        if artifact != self.published_artifact || artifact.task() != self.transformer_task() {
            bail!(
                "private H3 prepared attempt requires the exact {:?} INT8 ConvRot transformer {:?}, got {artifact:?}",
                self.task,
                self.published_artifact
            )
        }
        Ok(())
    }
}

/// Whether `artifact` is a reviewed compact DiT of `task` (any selection).
pub(crate) fn is_reviewed_compact_transformer(
    task: Task,
    artifact: H3ComfyPublishedArtifact,
) -> bool {
    artifact.task() == candle_task(task)
        && REVIEWED_COMPACT_TRANSFORMERS
            .iter()
            .any(|(_, _, reviewed)| *reviewed == artifact)
}

fn transformer_row(manifest: &'static ModelManifest) -> Result<&'static ModelFile> {
    let rows = manifest
        .files
        .iter()
        .filter(|file| file.component == ModelComponent::Transformer)
        .collect::<Vec<_>>();
    let [file] = rows.as_slice() else {
        bail!(
            "MiniMax H3 manifest {} must name exactly one compact transformer",
            manifest.name
        )
    };
    Ok(file)
}

/// The registered FastH3 V2 manifest: the FL2VA compact stack whose
/// Transformer row names FastVideo's distilled DiT.
#[cfg(test)]
pub(crate) fn test_fasth3_manifest() -> &'static ModelManifest {
    find_manifest(contract::FL2VA_FASTH3_V2_INT8).expect("the FastH3 tag is registered")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_candle::minimax_h3::{H3_FASTH3_REPOSITORY, H3_FASTH3_SOURCE_REVISION};

    #[test]
    fn mold_core_and_mold_candle_pin_the_same_fasth3_source() {
        assert_eq!(contract::FASTVIDEO_FASTH3_REPO, H3_FASTH3_REPOSITORY);
        assert_eq!(
            contract::FASTVIDEO_FASTH3_REVISION,
            H3_FASTH3_SOURCE_REVISION
        );
        assert_eq!(
            contract::repo_revision(contract::FASTVIDEO_FASTH3_REPO),
            Some(H3_FASTH3_SOURCE_REVISION)
        );
    }

    /// Every reviewed identity that is not a distilled base tier — both bases
    /// and every Turbo tag — resolves to its task's base selection, so
    /// nothing pre-existing moves.
    #[test]
    fn every_reviewed_compact_model_selects_its_task_base() {
        for model in contract::REVIEWED_COMPACT_MODELS
            .iter()
            .filter(|model| contract::distilled_base_tier_for_model(model).is_none())
        {
            let task = contract::task_for_model(model).unwrap();
            let selection = H3CompactTransformerSelection::for_model(model).unwrap();
            assert_eq!(
                selection,
                H3CompactTransformerSelection::base_for_task(task),
                "{model}"
            );
            assert!(selection.is_task_base());
            let (manifest, file) = selection.transformer_file().unwrap();
            assert_eq!(manifest.name, contract::base_compact_model_for_task(task));
            assert_eq!(file.hf_filename, selection.source());
        }
        let fl2va = H3CompactTransformerSelection::base_for_task(Task::Fl2va);
        assert_eq!(
            fl2va.published_artifact(),
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot
        );
        assert_eq!(fl2va.transformer_task(), H3TransformerTask::T2VaFl2Va);
        let ref2va = H3CompactTransformerSelection::base_for_task(Task::Ref2va);
        assert_eq!(
            ref2va.published_artifact(),
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot
        );
        assert_eq!(ref2va.transformer_task(), H3TransformerTask::Ref2Va);
    }

    #[test]
    fn a_manifest_naming_the_fasth3_file_selects_the_fasth3_artifact() {
        let manifest = test_fasth3_manifest();
        assert_eq!(manifest.name, contract::FL2VA_FASTH3_V2_INT8);
        let selection = H3CompactTransformerSelection::for_manifest(manifest).unwrap();
        // Selecting by the registered tag is the same answer, and mold-core's
        // tier row pins exactly mold-candle's published artifact.
        assert_eq!(
            H3CompactTransformerSelection::for_model(contract::FL2VA_FASTH3_V2_INT8).unwrap(),
            selection
        );
        let tier = contract::distilled_base_tier_for_model(contract::FL2VA_FASTH3_V2_INT8).unwrap();
        let artifact = H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot;
        assert_eq!(tier.transformer_hf_filename, H3_FASTH3_V2_INT8_SOURCE_PATH);
        assert_eq!(tier.transformer_size_bytes, artifact.file_bytes());
        assert_eq!(tier.transformer_sha256, artifact.content_sha256());
        assert_eq!(candle_task(tier.task), artifact.task());
        assert_eq!(selection.task(), Task::Fl2va);
        assert_eq!(
            selection.published_artifact(),
            H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot
        );
        assert_eq!(selection.transformer_task(), H3TransformerTask::T2VaFl2Va);
        assert_eq!(selection.manifest_model(), manifest.name);
        assert_eq!(selection.source(), H3_FASTH3_V2_INT8_SOURCE_PATH);
        assert!(!selection.is_task_base());
        assert_ne!(
            selection,
            H3CompactTransformerSelection::base_for_task(Task::Fl2va)
        );
        selection
            .require_artifact(H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot)
            .unwrap();
        assert!(selection
            .require_artifact(H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot)
            .is_err());
        assert!(is_reviewed_compact_transformer(
            Task::Fl2va,
            H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot
        ));
        assert!(!is_reviewed_compact_transformer(
            Task::Ref2va,
            H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot
        ));
        assert!(!is_reviewed_compact_transformer(
            Task::Fl2va,
            H3ComfyPublishedArtifact::Fl2VaPrunedBf16
        ));
    }

    #[test]
    fn mispinned_cross_task_or_unknown_transformer_rows_fail_closed() {
        let mutate = |edit: fn(&mut ModelFile)| {
            let mut manifest = test_fasth3_manifest().clone();
            let row = manifest
                .files
                .iter_mut()
                .find(|file| file.component == ModelComponent::Transformer)
                .unwrap();
            edit(row);
            H3CompactTransformerSelection::for_manifest(Box::leak(Box::new(manifest)))
        };
        assert!(mutate(|row| row.size_bytes -= 1).is_err());
        assert!(mutate(|row| row.sha256 =
            Some("e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"))
        .is_err());
        assert!(mutate(|row| row.hf_repo = contract::COMFY_REPO.to_owned()).is_err());
        assert!(mutate(|row| row.hf_filename.push('x')).is_err());

        // The FastH3 DiT is an FL2VA-face artifact; a Ref2VA manifest naming
        // it is refused.
        let mut ref2va = find_manifest(contract::REF2VA_COMFY).unwrap().clone();
        let artifact = H3ComfyPublishedArtifact::FastH3V2Fl2VaPrunedInt8ConvRot;
        let row = ref2va
            .files
            .iter_mut()
            .find(|file| file.component == ModelComponent::Transformer)
            .unwrap();
        row.hf_repo = contract::FASTVIDEO_FASTH3_REPO.to_owned();
        row.hf_filename = H3_FASTH3_V2_INT8_SOURCE_PATH.to_owned();
        row.size_bytes = artifact.file_bytes();
        row.sha256 = Some(artifact.content_sha256());
        assert!(H3CompactTransformerSelection::for_manifest(Box::leak(Box::new(ref2va))).is_err());

        assert!(H3CompactTransformerSelection::for_model("minimax-h3-fl2va:nope").is_err());
        assert!(H3CompactTransformerSelection::for_model(contract::FL2VA_OFFICIAL).is_err());
    }
}
