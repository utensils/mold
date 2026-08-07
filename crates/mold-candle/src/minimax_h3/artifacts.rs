use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::config::{H3ConditionerConfig, H3_FULL_LANGUAGE_LAYERS, H3_SELECTED_LANGUAGE_LAYERS};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ArtifactRole {
    ArchitectureConfig,
    Tokenizer,
    TokenizerConfig,
    ProcessorConfig,
    VideoProcessorConfig,
    CheckpointIndex,
    CheckpointShard(usize),
}

/// The two released BF16 conditioner namespaces are deliberately distinct.
/// The official Transformers checkpoint retains the nested
/// `model.language_model`/`model.visual` prefixes, while Comfy's layer-50
/// checkpoint removes only those two wrapper components. A checkpoint must
/// identify as exactly one layout; mixed namespaces are never repaired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConditionerWeightLayout {
    Official,
    ComfyLayer50,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFingerprint {
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenArtifact {
    pub role: ArtifactRole,
    pub path: PathBuf,
    pub fingerprint: ArtifactFingerprint,
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("failed to read {role:?} artifact {path}: {source}")]
    Io {
        role: ArtifactRole,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{role:?} artifact changed after planning: expected {expected:?}, found {found:?}")]
    FingerprintChanged {
        role: ArtifactRole,
        expected: ArtifactFingerprint,
        found: ArtifactFingerprint,
    },
    #[error("duplicate MiniMax H3 artifact role {0:?}")]
    DuplicateRole(ArtifactRole),
    #[error("missing MiniMax H3 artifact role {0:?}")]
    MissingRole(ArtifactRole),
    #[error("MiniMax H3 checkpoint key accounting failed: {0}")]
    CheckpointKeys(String),
}

impl FrozenArtifact {
    pub fn freeze(role: ArtifactRole, path: impl Into<PathBuf>) -> Result<Self, ArtifactError> {
        let path = path.into();
        let fingerprint = fingerprint_path(&role, &path)?;
        Ok(Self {
            role,
            path,
            fingerprint,
        })
    }

    /// Re-check identity immediately before mmap/load. A matching filename is
    /// not sufficient authority for a 51.5 GB component.
    pub fn verify(&self) -> Result<(), ArtifactError> {
        let found = fingerprint_path(&self.role, &self.path)?;
        if found != self.fingerprint {
            return Err(ArtifactError::FingerprintChanged {
                role: self.role.clone(),
                expected: self.fingerprint.clone(),
                found,
            });
        }
        Ok(())
    }
}

fn fingerprint_path(
    role: &ArtifactRole,
    path: &Path,
) -> Result<ArtifactFingerprint, ArtifactError> {
    let mut file = File::open(path).map_err(|source| ArtifactError::Io {
        role: role.clone(),
        path: path.to_path_buf(),
        source,
    })?;
    let size_bytes = file
        .metadata()
        .map_err(|source| ArtifactError::Io {
            role: role.clone(),
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| ArtifactError::Io {
            role: role.clone(),
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let sha256 = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(ArtifactFingerprint { sha256, size_bytes })
}

#[derive(Clone, Debug)]
pub struct ConditionerArtifacts {
    artifacts: BTreeMap<ArtifactRole, FrozenArtifact>,
}

impl ConditionerArtifacts {
    pub fn new(artifacts: impl IntoIterator<Item = FrozenArtifact>) -> Result<Self, ArtifactError> {
        let mut by_role = BTreeMap::new();
        for artifact in artifacts {
            let role = artifact.role.clone();
            if by_role.insert(role.clone(), artifact).is_some() {
                return Err(ArtifactError::DuplicateRole(role));
            }
        }
        for role in [
            ArtifactRole::ArchitectureConfig,
            ArtifactRole::Tokenizer,
            ArtifactRole::TokenizerConfig,
            ArtifactRole::ProcessorConfig,
            ArtifactRole::VideoProcessorConfig,
        ] {
            if !by_role.contains_key(&role) {
                return Err(ArtifactError::MissingRole(role));
            }
        }
        if !by_role
            .keys()
            .any(|role| matches!(role, ArtifactRole::CheckpointShard(_)))
        {
            return Err(ArtifactError::MissingRole(ArtifactRole::CheckpointShard(0)));
        }
        Ok(Self { artifacts: by_role })
    }

    pub fn get(&self, role: &ArtifactRole) -> Option<&FrozenArtifact> {
        self.artifacts.get(role)
    }

    pub fn checkpoint_shards(&self) -> impl Iterator<Item = &FrozenArtifact> {
        self.artifacts.iter().filter_map(|(role, artifact)| {
            matches!(role, ArtifactRole::CheckpointShard(_)).then_some(artifact)
        })
    }

    pub fn verify_all(&self) -> Result<(), ArtifactError> {
        self.artifacts.values().try_for_each(FrozenArtifact::verify)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointKeyReport {
    pub materialized: BTreeSet<String>,
    pub intentionally_skipped: BTreeSet<String>,
}

/// Account for every released checkpoint key. Only the final norm, LM head,
/// and decoder layers 50-63 are legal non-resident tensors.
pub fn validate_checkpoint_keys(
    names: impl IntoIterator<Item = impl Into<String>>,
) -> Result<CheckpointKeyReport, ArtifactError> {
    validate_checkpoint_keys_and_layout(names).map(|(report, _)| report)
}

pub(super) fn validate_checkpoint_keys_and_layout(
    names: impl IntoIterator<Item = impl Into<String>>,
) -> Result<(CheckpointKeyReport, ConditionerWeightLayout), ArtifactError> {
    let expected_materialized = expected_materialized_keys();
    let expected_skipped = expected_skipped_keys();
    let raw: BTreeSet<String> = names.into_iter().map(Into::into).collect();
    let layout = detect_checkpoint_layout(&raw)?;
    let actual = raw
        .iter()
        .map(|name| canonical_checkpoint_name(name, layout))
        .collect::<BTreeSet<_>>();

    let missing: Vec<_> = expected_materialized.difference(&actual).cloned().collect();
    let unexpected: Vec<_> = actual
        .difference(&expected_materialized)
        .filter(|name| !expected_skipped.contains(*name) && !is_regenerable_rotary_buffer(name))
        .cloned()
        .collect();
    let skipped: BTreeSet<_> = actual
        .intersection(&expected_skipped)
        .cloned()
        .chain(
            actual
                .iter()
                .filter(|name| is_regenerable_rotary_buffer(name))
                .cloned(),
        )
        .collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ArtifactError::CheckpointKeys(format!(
            "{} materialized keys missing (first: {:?}); {} unexpected keys (first: {:?})",
            missing.len(),
            missing.first(),
            unexpected.len(),
            unexpected.first()
        )));
    }
    Ok((
        CheckpointKeyReport {
            materialized: actual
                .intersection(&expected_materialized)
                .cloned()
                .collect(),
            intentionally_skipped: skipped,
        },
        layout,
    ))
}

fn detect_checkpoint_layout(
    names: &BTreeSet<String>,
) -> Result<ConditionerWeightLayout, ArtifactError> {
    let official = names.contains("model.language_model.embed_tokens.weight")
        || names.iter().any(|name| name.starts_with("model.visual."));
    let comfy = names.contains("model.embed_tokens.weight")
        || names.iter().any(|name| name.starts_with("visual."));
    match (official, comfy) {
        (true, false) => Ok(ConditionerWeightLayout::Official),
        (false, true) => Ok(ConditionerWeightLayout::ComfyLayer50),
        (true, true) => Err(ArtifactError::CheckpointKeys(
            "checkpoint mixes official and Comfy layer-50 namespaces".into(),
        )),
        (false, false) => Err(ArtifactError::CheckpointKeys(
            "checkpoint has no exact official or Comfy layer-50 namespace marker".into(),
        )),
    }
}

pub(super) fn canonical_checkpoint_name(name: &str, layout: ConditionerWeightLayout) -> String {
    match layout {
        ConditionerWeightLayout::Official => name.to_string(),
        ConditionerWeightLayout::ComfyLayer50 => {
            if let Some(suffix) = name.strip_prefix("model.") {
                format!("model.language_model.{suffix}")
            } else if let Some(suffix) = name.strip_prefix("visual.") {
                format!("model.visual.{suffix}")
            } else {
                name.to_string()
            }
        }
    }
}

fn is_regenerable_rotary_buffer(name: &str) -> bool {
    matches!(
        name,
        "model.language_model.rotary_emb.inv_freq" | "model.visual.rotary_pos_emb.inv_freq"
    )
}

fn expected_materialized_keys() -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    keys.insert("model.language_model.embed_tokens.weight".into());
    for layer in 0..H3_SELECTED_LANGUAGE_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            keys.insert(format!("{prefix}.{suffix}"));
        }
    }
    keys.insert("model.visual.patch_embed.proj.weight".into());
    keys.insert("model.visual.patch_embed.proj.bias".into());
    keys.insert("model.visual.pos_embed.weight".into());
    for layer in 0..27 {
        let prefix = format!("model.visual.blocks.{layer}");
        for suffix in [
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ] {
            keys.insert(format!("{prefix}.{suffix}"));
        }
    }
    for prefix in std::iter::once("model.visual.merger".to_string())
        .chain((0..3).map(|index| format!("model.visual.deepstack_merger_list.{index}")))
    {
        for suffix in [
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
            "norm.weight",
            "norm.bias",
        ] {
            keys.insert(format!("{prefix}.{suffix}"));
        }
    }
    keys
}

fn expected_skipped_keys() -> BTreeSet<String> {
    let mut keys = BTreeSet::from([
        "lm_head.weight".to_string(),
        "model.language_model.norm.weight".to_string(),
    ]);
    for layer in H3_SELECTED_LANGUAGE_LAYERS..H3_FULL_LANGUAGE_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            keys.insert(format!("{prefix}.{suffix}"));
        }
    }
    keys
}

pub(super) fn expected_checkpoint_shapes(
    config: &H3ConditionerConfig,
) -> BTreeMap<String, Vec<usize>> {
    let text = &config.text_config;
    let vision = &config.vision_config;
    let mut shapes = BTreeMap::new();
    shapes.insert(
        "model.language_model.embed_tokens.weight".into(),
        vec![text.vocab_size, text.hidden_size],
    );
    shapes.insert(
        "model.language_model.norm.weight".into(),
        vec![text.hidden_size],
    );
    shapes.insert(
        "lm_head.weight".into(),
        vec![text.vocab_size, text.hidden_size],
    );
    let q_width = text.num_attention_heads * text.head_dim;
    let kv_width = text.num_key_value_heads * text.head_dim;
    for layer in 0..H3_FULL_LANGUAGE_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![text.hidden_size]),
            ("post_attention_layernorm.weight", vec![text.hidden_size]),
            ("self_attn.q_proj.weight", vec![q_width, text.hidden_size]),
            ("self_attn.k_proj.weight", vec![kv_width, text.hidden_size]),
            ("self_attn.v_proj.weight", vec![kv_width, text.hidden_size]),
            ("self_attn.o_proj.weight", vec![text.hidden_size, q_width]),
            ("self_attn.q_norm.weight", vec![text.head_dim]),
            ("self_attn.k_norm.weight", vec![text.head_dim]),
            (
                "mlp.gate_proj.weight",
                vec![text.intermediate_size, text.hidden_size],
            ),
            (
                "mlp.up_proj.weight",
                vec![text.intermediate_size, text.hidden_size],
            ),
            (
                "mlp.down_proj.weight",
                vec![text.hidden_size, text.intermediate_size],
            ),
        ] {
            shapes.insert(format!("{prefix}.{suffix}"), shape);
        }
    }

    shapes.insert(
        "model.visual.patch_embed.proj.weight".into(),
        vec![
            vision.hidden_size,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ],
    );
    shapes.insert(
        "model.visual.patch_embed.proj.bias".into(),
        vec![vision.hidden_size],
    );
    shapes.insert(
        "model.visual.pos_embed.weight".into(),
        vec![vision.num_position_embeddings, vision.hidden_size],
    );
    for layer in 0..vision.depth {
        let prefix = format!("model.visual.blocks.{layer}");
        for (suffix, shape) in [
            (
                "attn.qkv.weight",
                vec![vision.hidden_size * 3, vision.hidden_size],
            ),
            ("attn.qkv.bias", vec![vision.hidden_size * 3]),
            (
                "attn.proj.weight",
                vec![vision.hidden_size, vision.hidden_size],
            ),
            ("attn.proj.bias", vec![vision.hidden_size]),
            (
                "mlp.linear_fc1.weight",
                vec![vision.intermediate_size, vision.hidden_size],
            ),
            ("mlp.linear_fc1.bias", vec![vision.intermediate_size]),
            (
                "mlp.linear_fc2.weight",
                vec![vision.hidden_size, vision.intermediate_size],
            ),
            ("mlp.linear_fc2.bias", vec![vision.hidden_size]),
            ("norm1.weight", vec![vision.hidden_size]),
            ("norm1.bias", vec![vision.hidden_size]),
            ("norm2.weight", vec![vision.hidden_size]),
            ("norm2.bias", vec![vision.hidden_size]),
        ] {
            shapes.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    let merged = vision.hidden_size * vision.spatial_merge_size.pow(2);
    for (prefix, norm_width) in
        std::iter::once(("model.visual.merger".to_string(), vision.hidden_size)).chain(
            (0..vision.deepstack_visual_indexes.len()).map(|index| {
                (
                    format!("model.visual.deepstack_merger_list.{index}"),
                    merged,
                )
            }),
        )
    {
        for (suffix, shape) in [
            ("norm.weight", vec![norm_width]),
            ("norm.bias", vec![norm_width]),
            ("linear_fc1.weight", vec![merged, merged]),
            ("linear_fc1.bias", vec![merged]),
            ("linear_fc2.weight", vec![vision.out_hidden_size, merged]),
            ("linear_fc2.bias", vec![vision.out_hidden_size]),
        ] {
            shapes.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn layer_tail_is_accounted_but_never_materialized() {
        let all = expected_materialized_keys()
            .into_iter()
            .chain(expected_skipped_keys())
            .collect::<Vec<_>>();
        let report = validate_checkpoint_keys(all).unwrap();
        assert_eq!(report.materialized.len(), 902);
        assert_eq!(report.intentionally_skipped.len(), 156);
        assert!(!report.materialized.contains("lm_head.weight"));
        assert!(!report
            .materialized
            .iter()
            .any(|name| name.starts_with("model.language_model.layers.50.")));
    }

    #[test]
    fn comfy_layer_50_namespace_is_exact_and_normalized() {
        let comfy = expected_materialized_keys().into_iter().map(|name| {
            name.strip_prefix("model.language_model.")
                .map(|suffix| format!("model.{suffix}"))
                .or_else(|| {
                    name.strip_prefix("model.visual.")
                        .map(|suffix| format!("visual.{suffix}"))
                })
                .unwrap_or(name)
        });
        let (report, layout) = validate_checkpoint_keys_and_layout(comfy).unwrap();
        assert_eq!(layout, ConditionerWeightLayout::ComfyLayer50);
        assert_eq!(report.materialized.len(), 902);
        assert!(report.intentionally_skipped.is_empty());
        assert!(report
            .materialized
            .contains("model.language_model.layers.49.self_attn.q_proj.weight"));
    }

    #[test]
    fn mixed_official_and_comfy_namespaces_are_rejected() {
        let names = expected_materialized_keys()
            .into_iter()
            .chain(["model.embed_tokens.weight".to_string()]);
        let error = validate_checkpoint_keys(names).unwrap_err();
        assert!(error.to_string().contains("mixes official and Comfy"));
    }

    #[test]
    fn text_only_qwen_checkpoint_is_rejected() {
        let text_only = expected_materialized_keys()
            .into_iter()
            .filter(|name| !name.starts_with("model.visual."));
        let error = validate_checkpoint_keys(text_only).unwrap_err();
        assert!(error.to_string().contains("351 materialized keys missing"));
    }

    #[test]
    fn unknown_tail_key_is_not_silently_skipped() {
        let all = expected_materialized_keys()
            .into_iter()
            .chain(expected_skipped_keys())
            .chain(["model.language_model.layers.64.self_attn.q_proj.weight".into()]);
        let error = validate_checkpoint_keys(all).unwrap_err();
        assert!(error.to_string().contains("1 unexpected keys"));
    }

    #[test]
    fn frozen_fingerprint_detects_same_path_replacement() {
        let path = std::env::temp_dir().join(format!(
            "mold-h3-artifact-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(&path, b"original").unwrap();
        let artifact = FrozenArtifact::freeze(ArtifactRole::Tokenizer, &path).unwrap();
        fs::write(&path, b"replacement").unwrap();
        assert!(matches!(
            artifact.verify(),
            Err(ArtifactError::FingerprintChanged { .. })
        ));
        let _ = fs::remove_file(path);
    }
}
