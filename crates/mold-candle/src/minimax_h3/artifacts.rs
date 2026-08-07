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
    role: ArtifactRole,
    path: PathBuf,
    fingerprint: ArtifactFingerprint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactVerificationProgress {
    pub role: ArtifactRole,
    pub artifact_bytes_verified: u64,
    pub artifact_bytes_total: u64,
    pub total_bytes_verified: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct VerifiedArtifactMetadata {
    identities: BTreeMap<ArtifactRole, FileMetadataIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileMetadataIdentity {
    size_bytes: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
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
    #[error("{role:?} artifact size changed: expected {expected} bytes, found {found} bytes")]
    SizeChanged {
        role: ArtifactRole,
        expected: u64,
        found: u64,
    },
    #[error("{role:?} artifact does not carry the released pinned identity: expected {expected:?}, found {found:?}")]
    PinnedIdentityMismatch {
        role: ArtifactRole,
        expected: ArtifactFingerprint,
        found: ArtifactFingerprint,
    },
    #[error("invalid pinned fingerprint for {role:?}: {message}")]
    InvalidPinnedFingerprint { role: ArtifactRole, message: String },
    #[error("MiniMax H3 artifact verification was cancelled")]
    VerificationCancelled,
    #[error("MiniMax H3 pinned artifact byte count overflow")]
    ByteCountOverflow,
    #[error("{role:?} artifact changed during or after content verification: {path}")]
    MetadataChanged { role: ArtifactRole, path: PathBuf },
    #[error("duplicate MiniMax H3 artifact role {0:?}")]
    DuplicateRole(ArtifactRole),
    #[error("missing MiniMax H3 artifact role {0:?}")]
    MissingRole(ArtifactRole),
    #[error("MiniMax H3 checkpoint key accounting failed: {0}")]
    CheckpointKeys(String),
}

impl FrozenArtifact {
    /// Declare one exact expected artifact identity. Content is hashed once,
    /// with cooperative cancellation and progress, when the complete artifact
    /// set is prepared. Keeping construction free of file reads avoids hashing
    /// 51.5-66.7 GB checkpoints once at planning and again at execution.
    pub fn pinned(
        role: ArtifactRole,
        path: impl Into<PathBuf>,
        fingerprint: ArtifactFingerprint,
    ) -> Result<Self, ArtifactError> {
        validate_fingerprint(&role, &fingerprint)?;
        if let Some(expected) = released_support_fingerprint(&role) {
            if fingerprint != expected {
                return Err(ArtifactError::PinnedIdentityMismatch {
                    role,
                    expected,
                    found: fingerprint,
                });
            }
        }
        Ok(Self {
            role,
            path: path.into(),
            fingerprint,
        })
    }

    pub fn role(&self) -> &ArtifactRole {
        &self.role
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fingerprint(&self) -> &ArtifactFingerprint {
        &self.fingerprint
    }
}

fn validate_fingerprint(
    role: &ArtifactRole,
    fingerprint: &ArtifactFingerprint,
) -> Result<(), ArtifactError> {
    if fingerprint.size_bytes == 0 {
        return Err(ArtifactError::InvalidPinnedFingerprint {
            role: role.clone(),
            message: "size must be nonzero".into(),
        });
    }
    if fingerprint.sha256.len() != 64
        || !fingerprint
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ArtifactError::InvalidPinnedFingerprint {
            role: role.clone(),
            message: "sha256 must be 64 lowercase hexadecimal characters".into(),
        });
    }
    Ok(())
}

fn released_support_fingerprint(role: &ArtifactRole) -> Option<ArtifactFingerprint> {
    // MiniMaxAI/MiniMax-H3 at the license-reviewed revision
    // bfc8ed0353f5a9733be73e6b2c98ec0948195b86. These are, respectively,
    // text_encoder/config.json, tokenizer/{tokenizer.json,tokenizer_config.json},
    // and text_encoder/{preprocessor_config.json,video_preprocessor_config.json}.
    // Content authentication is intentional: structural parsing alone cannot
    // establish that the released tokenizer and processors are being used.
    let (size_bytes, sha256) = match role {
        ArtifactRole::ArchitectureConfig => (
            1_474,
            "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427",
        ),
        ArtifactRole::Tokenizer => (
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        ArtifactRole::TokenizerConfig => (
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
        ),
        ArtifactRole::ProcessorConfig => (
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        ArtifactRole::VideoProcessorConfig => (
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
        ArtifactRole::CheckpointIndex | ArtifactRole::CheckpointShard(_) => return None,
    };
    Some(ArtifactFingerprint {
        sha256: sha256.into(),
        size_bytes,
    })
}

fn metadata_identity(
    role: &ArtifactRole,
    path: &Path,
) -> Result<FileMetadataIdentity, ArtifactError> {
    let metadata = std::fs::metadata(path).map_err(|source| ArtifactError::Io {
        role: role.clone(),
        path: path.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(FileMetadataIdentity {
            size_bytes: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(FileMetadataIdentity {
            size_bytes: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

fn fingerprint_path_with_progress(
    artifact: &FrozenArtifact,
    verified_before: u64,
    total_bytes: u64,
    progress: &mut dyn FnMut(ArtifactVerificationProgress) -> Result<(), ArtifactError>,
) -> Result<(ArtifactFingerprint, FileMetadataIdentity), ArtifactError> {
    let role = &artifact.role;
    let path = &artifact.path;
    let before = metadata_identity(role, path)?;
    if before.size_bytes != artifact.fingerprint.size_bytes {
        return Err(ArtifactError::SizeChanged {
            role: role.clone(),
            expected: artifact.fingerprint.size_bytes,
            found: before.size_bytes,
        });
    }
    let mut file = File::open(path).map_err(|source| ArtifactError::Io {
        role: role.clone(),
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut artifact_bytes_verified = 0_u64;
    progress(ArtifactVerificationProgress {
        role: role.clone(),
        artifact_bytes_verified,
        artifact_bytes_total: before.size_bytes,
        total_bytes_verified: verified_before,
        total_bytes,
    })?;
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
        artifact_bytes_verified += read as u64;
        progress(ArtifactVerificationProgress {
            role: role.clone(),
            artifact_bytes_verified,
            artifact_bytes_total: before.size_bytes,
            total_bytes_verified: verified_before + artifact_bytes_verified,
            total_bytes,
        })?;
    }
    let sha256 = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let found = ArtifactFingerprint {
        sha256,
        size_bytes: before.size_bytes,
    };
    if found != artifact.fingerprint {
        return Err(ArtifactError::FingerprintChanged {
            role: role.clone(),
            expected: artifact.fingerprint.clone(),
            found,
        });
    }
    let after = metadata_identity(role, path)?;
    if after != before {
        return Err(ArtifactError::MetadataChanged {
            role: role.clone(),
            path: path.clone(),
        });
    }
    Ok((artifact.fingerprint.clone(), after))
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
        for artifact in by_role.values() {
            if let Some(expected) = released_support_fingerprint(&artifact.role) {
                if artifact.fingerprint != expected {
                    return Err(ArtifactError::PinnedIdentityMismatch {
                        role: artifact.role.clone(),
                        expected,
                        found: artifact.fingerprint.clone(),
                    });
                }
            }
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

    pub fn expected_bytes(&self) -> Result<u64, ArtifactError> {
        self.artifacts.values().try_fold(0_u64, |total, artifact| {
            total
                .checked_add(artifact.fingerprint.size_bytes)
                .ok_or(ArtifactError::ByteCountOverflow)
        })
    }

    pub(super) fn verify_all_with_progress(
        &self,
        progress: &mut dyn FnMut(ArtifactVerificationProgress) -> Result<(), ArtifactError>,
    ) -> Result<VerifiedArtifactMetadata, ArtifactError> {
        let total_bytes = self.expected_bytes()?;
        let mut verified_before = 0_u64;
        let mut identities = BTreeMap::new();
        for artifact in self.artifacts.values() {
            let (_, identity) =
                fingerprint_path_with_progress(artifact, verified_before, total_bytes, progress)?;
            verified_before = verified_before
                .checked_add(artifact.fingerprint.size_bytes)
                .ok_or(ArtifactError::ByteCountOverflow)?;
            identities.insert(artifact.role.clone(), identity);
        }
        Ok(VerifiedArtifactMetadata { identities })
    }

    pub(super) fn verify_metadata(
        &self,
        verified: &VerifiedArtifactMetadata,
    ) -> Result<(), ArtifactError> {
        for artifact in self.artifacts.values() {
            let expected = verified
                .identities
                .get(&artifact.role)
                .ok_or_else(|| ArtifactError::MissingRole(artifact.role.clone()))?;
            if &metadata_identity(&artifact.role, &artifact.path)? != expected {
                return Err(ArtifactError::MetadataChanged {
                    role: artifact.role.clone(),
                    path: artifact.path.clone(),
                });
            }
        }
        Ok(())
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
    let missing_official_tail: Vec<_> = if layout == ConditionerWeightLayout::Official {
        expected_skipped.difference(&actual).cloned().collect()
    } else {
        Vec::new()
    };
    let forbidden_comfy_tail: Vec<_> = if layout == ConditionerWeightLayout::ComfyLayer50 {
        actual.intersection(&expected_skipped).cloned().collect()
    } else {
        Vec::new()
    };
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
    if !missing.is_empty()
        || !missing_official_tail.is_empty()
        || !forbidden_comfy_tail.is_empty()
        || !unexpected.is_empty()
    {
        return Err(ArtifactError::CheckpointKeys(format!(
            "{} materialized keys missing (first: {:?}); {} official tail keys missing (first: {:?}); {} tail keys forbidden in Comfy layer-50 layout (first: {:?}); {} unexpected keys (first: {:?})",
            missing.len(),
            missing.first(),
            missing_official_tail.len(),
            missing_official_tail.first(),
            forbidden_comfy_tail.len(),
            forbidden_comfy_tail.first(),
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

    fn test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-h3-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn fingerprint(bytes: &[u8]) -> ArtifactFingerprint {
        ArtifactFingerprint {
            sha256: Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            size_bytes: bytes.len() as u64,
        }
    }

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
    fn official_and_comfy_tail_key_sets_cannot_be_crossed() {
        let truncated_official = expected_materialized_keys();
        let error = validate_checkpoint_keys_and_layout(truncated_official).unwrap_err();
        assert!(error.to_string().contains("156 official tail keys missing"));

        let comfy_with_tail = expected_materialized_keys()
            .into_iter()
            .chain(expected_skipped_keys())
            .map(|name| {
                name.strip_prefix("model.language_model.")
                    .map(|suffix| format!("model.{suffix}"))
                    .or_else(|| {
                        name.strip_prefix("model.visual.")
                            .map(|suffix| format!("visual.{suffix}"))
                    })
                    .unwrap_or(name)
            });
        let error = validate_checkpoint_keys_and_layout(comfy_with_tail).unwrap_err();
        assert!(error
            .to_string()
            .contains("156 tail keys forbidden in Comfy layer-50 layout"));
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
    fn verified_metadata_detects_same_path_replacement_without_rehashing() {
        let path = test_path("metadata-replacement");
        let original = b"original";
        fs::write(&path, original).unwrap();
        let artifact = FrozenArtifact::pinned(
            ArtifactRole::CheckpointShard(0),
            &path,
            fingerprint(original),
        )
        .unwrap();
        let (_, verified) =
            fingerprint_path_with_progress(&artifact, 0, original.len() as u64, &mut |_| Ok(()))
                .unwrap();
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"replaced").unwrap();
        assert!(matches!(
            metadata_identity(artifact.role(), artifact.path()),
            Ok(found) if found != verified
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn large_artifact_hashing_is_cooperatively_cancellable() {
        let path = test_path("cancel-hash");
        let bytes = vec![7_u8; 17 * 1024 * 1024];
        fs::write(&path, &bytes).unwrap();
        let artifact =
            FrozenArtifact::pinned(ArtifactRole::CheckpointShard(0), &path, fingerprint(&bytes))
                .unwrap();
        let mut observed = Vec::new();
        let error =
            fingerprint_path_with_progress(&artifact, 0, bytes.len() as u64, &mut |event| {
                observed.push(event.total_bytes_verified);
                if event.artifact_bytes_verified >= 8 * 1024 * 1024 {
                    Err(ArtifactError::VerificationCancelled)
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert!(matches!(error, ArtifactError::VerificationCancelled));
        assert!(observed.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(observed
            .last()
            .is_some_and(|bytes| *bytes < 17 * 1024 * 1024));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn every_tokenizer_and_processor_role_requires_the_released_digest() {
        for role in [
            ArtifactRole::ArchitectureConfig,
            ArtifactRole::Tokenizer,
            ArtifactRole::TokenizerConfig,
            ArtifactRole::ProcessorConfig,
            ArtifactRole::VideoProcessorConfig,
        ] {
            let expected = released_support_fingerprint(&role).unwrap();
            assert!(FrozenArtifact::pinned(role.clone(), "/unused", expected.clone()).is_ok());
            let wrong = ArtifactFingerprint {
                sha256: "0".repeat(64),
                size_bytes: expected.size_bytes,
            };
            assert!(matches!(
                FrozenArtifact::pinned(role.clone(), "/unused", wrong),
                Err(ArtifactError::PinnedIdentityMismatch { role: found, .. }) if found == role
            ));
        }
    }

    #[test]
    fn same_size_corrupt_tokenizer_cannot_satisfy_the_pinned_identity() {
        let role = ArtifactRole::Tokenizer;
        let expected = released_support_fingerprint(&role).unwrap();
        let path = test_path("corrupt-tokenizer");
        fs::write(&path, vec![0_u8; expected.size_bytes as usize]).unwrap();
        let artifact = FrozenArtifact::pinned(role, &path, expected).unwrap();
        assert!(matches!(
            fingerprint_path_with_progress(
                &artifact,
                0,
                artifact.fingerprint().size_bytes,
                &mut |_| Ok(())
            ),
            Err(ArtifactError::FingerprintChanged { .. })
        ));
        let _ = fs::remove_file(path);
    }
}
