//! Private exact-BF16 MiniMax H3 Qwen layer-50 capture adapter.
//!
//! This binary is unreachable without `dev-bins,h3-private-uat` and refuses
//! execution unless an indexed CUDA device opens successfully. It loads only
//! the frozen official BF16 conditioner through `load_bf16_conditioner`; no
//! quantized deployment loader or production engine is reachable here.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    load_bf16_conditioner, ArtifactFingerprint, ArtifactRole, ConditionerArtifacts, FrozenArtifact,
    H3ConditionerInput, H3VisionInput,
};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const REQUEST_SCHEMA: &str = "mold.minimax-h3.qwen-layer50-capture-request.v1";
const RAW_OUTPUT_SCHEMA: &str = "mold.minimax-h3.qwen-layer50-raw-output.v1";
const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";
const OFFICIAL_MODEL_REVISION: &str = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86";
const OFFICIAL_TEXT_ENCODER_INDEX_SHA256: &str =
    "06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a";
const OFFICIAL_LICENSE_SHA256: &str =
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44";
const REVIEWED_AUTHORIZATION_SHA256: &str =
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d";
const CAPTURE_MARKER: &str = "mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1";
const INPUT_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.qwen-layer50-input.v1\0";
const REQUIRED_SCOPES: [&str; 3] = [
    "checkpoint-execution",
    "fixture-capture",
    "generated-output-retention",
];
const MAX_REQUEST_BYTES: u64 = 32 * 1024 * 1024;
const MAX_AUTHORIZATION_BYTES: u64 = 64 * 1024;
const MAX_AUTHORIZATION_SOURCE_BYTES: u64 = 1024 * 1024;
const MAX_RAW_OUTPUT_BYTES: u64 = 128 * 1024 * 1024;

fn usage() -> &'static str {
    "usage: h3_qwen_layer50_capture --model-root <absolute-official-model> \
     --fixture-root <absolute-external-root> --repository-root <absolute-repository-root> \
     --authorization-record <absolute-record.json> --request <absolute-request.json> \
     --raw-output <absolute-new-output.json> --source-revision <40-hex> \
     --device <cuda:index>"
}

#[derive(Debug)]
struct Arguments {
    model_root: PathBuf,
    fixture_root: PathBuf,
    repository_root: PathBuf,
    authorization_record: PathBuf,
    request: PathBuf,
    raw_output: PathBuf,
    source_revision: String,
    device_index: usize,
    device_label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    size_bytes: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    user: u32,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(unix)]
    links: u64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            size_bytes: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            user: metadata.uid(),
            #[cfg(unix)]
            mode: metadata.mode(),
            #[cfg(unix)]
            modified_seconds: metadata.mtime(),
            #[cfg(unix)]
            modified_nanoseconds: metadata.mtime_nsec(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
            #[cfg(unix)]
            links: metadata.nlink(),
            #[cfg(not(unix))]
            modified: metadata.modified().ok(),
        }
    }
}

#[derive(Debug)]
struct OpenedInputSnapshot {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    sha256: String,
}

impl OpenedInputSnapshot {
    fn read_bounded(
        path: &Path,
        maximum: u64,
        label: &str,
        require_owner_readonly: bool,
    ) -> Result<(Self, Vec<u8>)> {
        let requested =
            fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
        if requested.file_type().is_symlink() || !requested.file_type().is_file() {
            bail!("{label} must be a direct regular file")
        }
        let file = open_regular_file_no_follow(path)
            .with_context(|| format!("failed to open {label} without following links"))?;
        let identity = FileIdentity::from_metadata(&file.metadata()?);
        if identity != FileIdentity::from_metadata(&requested) || identity.size_bytes > maximum {
            bail!("{label} changed while opening or exceeds its size bound")
        }
        #[cfg(unix)]
        if require_owner_readonly {
            // SAFETY: `geteuid` has no preconditions and only reads process state.
            let effective_uid = unsafe { libc::geteuid() };
            if identity.user != effective_uid || identity.mode & 0o777 != 0o400 {
                bail!("{label} must be immutable and readable only by the process owner")
            }
        }
        #[cfg(not(unix))]
        if require_owner_readonly {
            bail!("owner-only immutable capture inputs require a Unix host")
        }
        let sha256 =
            sha256_open_file(&file).with_context(|| format!("failed to authenticate {label}"))?;
        let mut reader = file.try_clone()?;
        let mut bytes = Vec::with_capacity(identity.size_bytes as usize);
        reader
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {label}"))?;
        if bytes.len() as u64 != identity.size_bytes || sha256_bytes(&bytes) != sha256 {
            bail!("{label} changed while it was read")
        }
        let snapshot = Self {
            path: path.to_path_buf(),
            file,
            identity,
            sha256,
        };
        snapshot.revalidate(label)?;
        Ok((snapshot, bytes))
    }

    fn revalidate(&self, label: &str) -> Result<()> {
        let descriptor_identity = FileIdentity::from_metadata(&self.file.metadata()?);
        let current = open_regular_file_no_follow(&self.path)
            .with_context(|| format!("failed to reopen {label} without following links"))?;
        let path_identity = FileIdentity::from_metadata(&current.metadata()?);
        if descriptor_identity != self.identity
            || path_identity != self.identity
            || sha256_open_file(&self.file)? != self.sha256
        {
            bail!("{label} changed after authentication")
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PathSnapshot {
    path: PathBuf,
    identity: FileIdentity,
}

impl PathSnapshot {
    fn capture(path: &Path, label: &str) -> Result<Self> {
        let requested =
            fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
        if requested.file_type().is_symlink() || !requested.file_type().is_file() {
            bail!("{label} must be a direct regular file")
        }
        let file = open_regular_file_no_follow(path)
            .with_context(|| format!("failed to open {label} without following links"))?;
        let identity = FileIdentity::from_metadata(&file.metadata()?);
        if identity != FileIdentity::from_metadata(&requested) {
            bail!("{label} changed while opening")
        }
        Ok(Self {
            path: path.to_path_buf(),
            identity,
        })
    }

    fn revalidate(&self, label: &str) -> Result<()> {
        let file = open_regular_file_no_follow(&self.path)
            .with_context(|| format!("failed to reopen {label} without following links"))?;
        if FileIdentity::from_metadata(&file.metadata()?) != self.identity {
            bail!("{label} changed after authentication")
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationRecord {
    schema_version: String,
    family: String,
    decision: String,
    license_revision: String,
    license_sha256: String,
    approved_scopes: Vec<String>,
    source_document_path: PathBuf,
    source_document_sha256: String,
    review_reference: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureRequest {
    schema_version: String,
    authorization_document_sha256: String,
    source_revision: String,
    model_revision: String,
    cases: Vec<CaptureCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCase {
    case_id: String,
    kind: String,
    input_sha256: String,
    token_ids: Vec<u32>,
    position_ids: [Vec<u32>; 3],
    image: Option<VisionPayload>,
    video: Option<VisionPayload>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VisionPayload {
    shape: [usize; 2],
    grid_thw: [usize; 3],
    bfloat16_le_base64: String,
}

#[derive(Debug, Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct RawCaptureOutput<'a> {
    schema_version: &'static str,
    claim_marker: &'static str,
    authorization_document_sha256: &'a str,
    source_revision: &'a str,
    model_revision: &'static str,
    device: &'a str,
    dtype: &'static str,
    resident_language_layers: usize,
    cases: Vec<RawCaseOutput>,
}

#[derive(Debug, Serialize)]
struct RawCaseOutput {
    case_id: String,
    input_sha256: String,
    shape: Vec<usize>,
    bfloat16_le_base64: String,
}

fn parse_args() -> Result<Arguments> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--help" || argument == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        if !argument.starts_with("--") {
            bail!("unexpected positional argument {argument:?}; {}", usage())
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {argument}; {}", usage()))?;
        if value.starts_with("--") || values.insert(argument.clone(), value).is_some() {
            bail!("invalid or duplicate argument {argument}; {}", usage())
        }
    }
    if values.len() != 8 {
        bail!("all capture arguments are required; {}", usage())
    }
    let device_label = values.remove("--device").context(usage())?;
    let device_index = device_label
        .strip_prefix("cuda:")
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| anyhow!("capture device must be indexed CUDA, such as cuda:0"))?
        .parse::<usize>()
        .context("CUDA device index is outside the supported integer domain")?;
    let arguments = Arguments {
        model_root: PathBuf::from(values.remove("--model-root").context(usage())?),
        fixture_root: PathBuf::from(values.remove("--fixture-root").context(usage())?),
        repository_root: PathBuf::from(values.remove("--repository-root").context(usage())?),
        authorization_record: PathBuf::from(
            values.remove("--authorization-record").context(usage())?,
        ),
        request: PathBuf::from(values.remove("--request").context(usage())?),
        raw_output: PathBuf::from(values.remove("--raw-output").context(usage())?),
        source_revision: values.remove("--source-revision").context(usage())?,
        device_index,
        device_label,
    };
    if !values.is_empty() || !valid_lower_hex(&arguments.source_revision, 40) {
        bail!(
            "unknown argument or non-canonical source revision; {}",
            usage()
        )
    }
    Ok(arguments)
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_existing(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        bail!("{label} must be an absolute canonical path")
    }
    if fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label}"))?
        .file_type()
        .is_symlink()
    {
        bail!("{label} must not be a symbolic link")
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {label}"))?;
    if canonical != path {
        bail!("{label} must not contain aliases or symbolic-link components")
    }
    Ok(canonical)
}

fn require_external(path: &Path, repository_root: &Path, label: &str) -> Result<()> {
    if path.starts_with(repository_root) {
        bail!("{label} must live outside the Mold repository")
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn require_owner_only_directory(path: &Path, label: &str) -> Result<()> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        bail!("{label} must be a direct directory")
    }
    #[cfg(unix)]
    {
        // SAFETY: `geteuid` has no preconditions and only reads process state.
        let effective_uid = unsafe { libc::geteuid() };
        if metadata.uid() != effective_uid || metadata.mode() & 0o077 != 0 {
            bail!("{label} must be accessible only by the process owner")
        }
    }
    #[cfg(not(unix))]
    bail!("private capture output requires a Unix host");
    Ok(())
}

fn validate_authorization(
    path: &Path,
    repository_root: &Path,
) -> Result<(String, Vec<OpenedInputSnapshot>)> {
    require_external(path, repository_root, "authorization record")?;
    let (record_snapshot, encoded) = OpenedInputSnapshot::read_bounded(
        path,
        MAX_AUTHORIZATION_BYTES,
        "authorization record",
        false,
    )?;
    let record: AuthorizationRecord = serde_json::from_slice(&encoded)
        .context("authorization record is not valid exact-schema JSON")?;
    let scopes = record
        .approved_scopes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if record.schema_version != AUTHORIZATION_SCHEMA
        || record.family != "minimax-h3"
        || record.decision != "approved"
        || record.license_revision != OFFICIAL_MODEL_REVISION
        || record.license_sha256 != OFFICIAL_LICENSE_SHA256
        || REQUIRED_SCOPES.iter().any(|scope| !scopes.contains(scope))
        || scopes.len() != record.approved_scopes.len()
        || record.source_document_sha256 != REVIEWED_AUTHORIZATION_SHA256
        || record.review_reference.trim() != record.review_reference
        || record.review_reference.is_empty()
    {
        bail!("authorization record does not cover the reviewed exact-BF16 capture")
    }
    let source = canonical_existing(
        &record.source_document_path,
        "authorization source document",
    )?;
    require_external(&source, repository_root, "authorization source document")?;
    if source == path || source.parent() != path.parent() {
        bail!("authorization source document must be a distinct sibling of its record")
    }
    let (source_snapshot, source_bytes) = OpenedInputSnapshot::read_bounded(
        &source,
        MAX_AUTHORIZATION_SOURCE_BYTES,
        "authorization source document",
        false,
    )?;
    if sha256_bytes(&source_bytes) != record.source_document_sha256 {
        bail!("authorization source document hash differs from its record")
    }
    Ok((
        record.source_document_sha256,
        vec![record_snapshot, source_snapshot],
    ))
}

fn metadata_lines(
    model_root: &Path,
    relative_path: &str,
) -> Result<(Vec<String>, OpenedInputSnapshot)> {
    let path = model_root
        .join(".cache/huggingface/download")
        .join(format!("{relative_path}.metadata"));
    let path = canonical_existing(&path, "Hugging Face component metadata")?;
    if !path.starts_with(model_root) {
        bail!("Hugging Face component metadata escapes the official model root")
    }
    let (snapshot, encoded) =
        OpenedInputSnapshot::read_bounded(&path, 4096, "Hugging Face component metadata", false)?;
    let text = std::str::from_utf8(&encoded).context("component metadata is not UTF-8")?;
    let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
    if lines.first().map(String::as_str) != Some(OFFICIAL_MODEL_REVISION) {
        bail!("component metadata differs from the official model revision")
    }
    Ok((lines, snapshot))
}

fn metadata_sha256(
    model_root: &Path,
    relative_path: &str,
) -> Result<(String, OpenedInputSnapshot)> {
    let (lines, snapshot) = metadata_lines(model_root, relative_path)?;
    let digest = lines
        .get(1)
        .map(String::as_str)
        .filter(|value| valid_lower_hex(value, 64))
        .ok_or_else(|| anyhow!("checkpoint metadata lacks a canonical SHA-256"))?;
    Ok((digest.to_owned(), snapshot))
}

fn frozen(
    role: ArtifactRole,
    path: PathBuf,
    fingerprint: ArtifactFingerprint,
) -> Result<FrozenArtifact> {
    FrozenArtifact::pinned(role, path, fingerprint).map_err(Into::into)
}

fn conditioner_artifacts(
    model_root: &Path,
) -> Result<(
    ConditionerArtifacts,
    Vec<OpenedInputSnapshot>,
    Vec<PathSnapshot>,
)> {
    let fixed = [
        (ArtifactRole::ArchitectureConfig, "text_encoder/config.json"),
        (ArtifactRole::Tokenizer, "tokenizer/tokenizer.json"),
        (
            ArtifactRole::TokenizerConfig,
            "tokenizer/tokenizer_config.json",
        ),
        (
            ArtifactRole::ProcessorConfig,
            "processor/preprocessor_config.json",
        ),
        (
            ArtifactRole::VideoProcessorConfig,
            "processor/video_preprocessor_config.json",
        ),
    ];
    let mut artifacts = Vec::new();
    let mut opened_inputs = Vec::new();
    let mut checkpoint_inputs = Vec::new();
    for (role, relative) in fixed {
        let path = canonical_existing(&model_root.join(relative), "official model component")?;
        if !path.starts_with(model_root) {
            bail!("official model component escapes its snapshot")
        }
        let (snapshot, encoded) = OpenedInputSnapshot::read_bounded(
            &path,
            16 * 1024 * 1024,
            "official model component",
            false,
        )?;
        artifacts.push(frozen(
            role,
            path,
            ArtifactFingerprint {
                sha256: snapshot.sha256.clone(),
                size_bytes: encoded.len() as u64,
            },
        )?);
        opened_inputs.push(snapshot);
    }

    let index_relative = "text_encoder/model.safetensors.index.json";
    let index_path = canonical_existing(&model_root.join(index_relative), "text encoder index")?;
    let (index_snapshot, index_bytes) = OpenedInputSnapshot::read_bounded(
        &index_path,
        16 * 1024 * 1024,
        "text encoder index",
        false,
    )?;
    let (_, index_metadata_snapshot) = metadata_lines(model_root, index_relative)?;
    if sha256_bytes(&index_bytes) != OFFICIAL_TEXT_ENCODER_INDEX_SHA256 {
        bail!("text encoder index differs from the reviewed manifest authority")
    }
    let index: CheckpointIndex =
        serde_json::from_slice(&index_bytes).context("text encoder index is malformed")?;
    let shard_names = index.weight_map.into_values().collect::<BTreeSet<_>>();
    if shard_names.len() != 14 {
        bail!("official text encoder index must name exactly 14 checkpoint shards")
    }
    artifacts.push(frozen(
        ArtifactRole::CheckpointIndex,
        index_path,
        ArtifactFingerprint {
            sha256: sha256_bytes(&index_bytes),
            size_bytes: index_bytes.len() as u64,
        },
    )?);
    opened_inputs.push(index_snapshot);
    opened_inputs.push(index_metadata_snapshot);
    for (index, name) in shard_names.into_iter().enumerate() {
        if Path::new(&name).components().count() != 1 || !name.ends_with(".safetensors") {
            bail!("text encoder index contains a non-canonical shard path")
        }
        let relative = format!("text_encoder/{name}");
        let path = canonical_existing(&model_root.join(&relative), "text encoder shard")?;
        if !path.starts_with(model_root) {
            bail!("text encoder shard escapes its snapshot")
        }
        let path_snapshot = PathSnapshot::capture(&path, "text encoder shard")?;
        let size_bytes = path_snapshot.identity.size_bytes;
        let (sha256, metadata_snapshot) = metadata_sha256(model_root, &relative)?;
        artifacts.push(frozen(
            ArtifactRole::CheckpointShard(index),
            path,
            ArtifactFingerprint { sha256, size_bytes },
        )?);
        checkpoint_inputs.push(path_snapshot);
        opened_inputs.push(metadata_snapshot);
    }
    Ok((
        ConditionerArtifacts::new(artifacts)?,
        opened_inputs,
        checkpoint_inputs,
    ))
}

fn put_u64(digest: &mut Sha256, value: usize) -> Result<()> {
    let value = u64::try_from(value).context("capture input length exceeds u64")?;
    digest.update(value.to_le_bytes());
    Ok(())
}

fn put_string(digest: &mut Sha256, value: &str) -> Result<()> {
    put_u64(digest, value.len())?;
    digest.update(value.as_bytes());
    Ok(())
}

fn decode_vision(payload: &VisionPayload) -> Result<(Vec<u16>, usize)> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&payload.bfloat16_le_base64)
        .context("vision BF16 payload is not canonical base64")?;
    if !bytes.len().is_multiple_of(2) {
        bail!("vision BF16 payload has an odd byte count")
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    let expected = payload.shape[0]
        .checked_mul(payload.shape[1])
        .ok_or_else(|| anyhow!("vision payload shape overflows"))?;
    if values.len() != expected || payload.shape.contains(&0) || payload.grid_thw.contains(&0) {
        bail!("vision BF16 payload shape is inconsistent")
    }
    Ok((values, bytes.len()))
}

fn case_identity(case: &CaptureCase) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(INPUT_IDENTITY_DOMAIN);
    put_string(&mut digest, &case.case_id)?;
    put_string(&mut digest, &case.kind)?;
    put_u64(&mut digest, case.token_ids.len())?;
    for value in &case.token_ids {
        digest.update(value.to_le_bytes());
    }
    for axis in &case.position_ids {
        put_u64(&mut digest, axis.len())?;
        for value in axis {
            digest.update(value.to_le_bytes());
        }
    }
    for vision in [&case.image, &case.video] {
        match vision {
            Some(payload) => {
                digest.update([1]);
                for value in payload.shape {
                    put_u64(&mut digest, value)?;
                }
                for value in payload.grid_thw {
                    put_u64(&mut digest, value)?;
                }
                let (bits, byte_count) = decode_vision(payload)?;
                put_u64(&mut digest, byte_count)?;
                for value in bits {
                    digest.update(value.to_le_bytes());
                }
            }
            None => digest.update([0]),
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn vision_tensor(payload: &VisionPayload, device: &Device) -> Result<H3VisionInput> {
    let (bits, _) = decode_vision(payload)?;
    let values = bits
        .into_iter()
        .map(|bits| f32::from_bits(u32::from(bits) << 16))
        .collect::<Vec<_>>();
    let pixel_values = Tensor::from_vec(values, (payload.shape[0], payload.shape[1]), device)?;
    let grid = payload
        .grid_thw
        .iter()
        .map(|value| u32::try_from(*value).context("vision grid exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let grid_thw = Tensor::from_vec(grid, (1, 3), device)?;
    Ok(H3VisionInput {
        pixel_values,
        grid_thw,
    })
}

fn conditioner_input(case: &CaptureCase, device: &Device) -> Result<H3ConditionerInput> {
    let sequence = case.token_ids.len();
    if sequence == 0 || case.position_ids.iter().any(|axis| axis.len() != sequence) {
        bail!("capture case has empty or inconsistent Qwen input dimensions")
    }
    let input_ids = Tensor::from_vec(case.token_ids.clone(), (1, sequence), device)?;
    let positions = case
        .position_ids
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let position_ids = Tensor::from_vec(positions, (3, 1, sequence), device)?;
    Ok(H3ConditionerInput {
        input_ids,
        position_ids,
        image: case
            .image
            .as_ref()
            .map(|payload| vision_tensor(payload, device))
            .transpose()?,
        video: case
            .video
            .as_ref()
            .map(|payload| vision_tensor(payload, device))
            .transpose()?,
    })
}

fn bfloat16_base64(tensor: &Tensor) -> Result<String> {
    let values = tensor
        .to_dtype(DType::BF16)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let encoded = value.to_bits();
        if encoded & 0xffff != 0 {
            bail!("BF16 capture did not round-trip through exact F32 representation")
        }
        bytes.extend_from_slice(&((encoded >> 16) as u16).to_le_bytes());
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

fn exclusive_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut destination = options
        .open(path)
        .with_context(|| format!("failed to create capture output {}", path.display()))?;
    destination
        .write_all(bytes)
        .context("failed to write capture output")?;
    destination
        .sync_all()
        .context("failed to sync capture output")
}

fn run() -> Result<()> {
    let arguments = parse_args()?;
    let repository_root = canonical_existing(&arguments.repository_root, "repository root")?;
    let fixture_root = canonical_existing(&arguments.fixture_root, "fixture root")?;
    let model_root = canonical_existing(&arguments.model_root, "official model root")?;
    let authorization_record =
        canonical_existing(&arguments.authorization_record, "authorization record")?;
    let request_path = canonical_existing(&arguments.request, "capture request")?;
    for (path, label) in [
        (&fixture_root, "fixture root"),
        (&model_root, "official model root"),
        (&authorization_record, "authorization record"),
        (&request_path, "capture request"),
    ] {
        require_external(path, &repository_root, label)?;
    }
    if !request_path.starts_with(&fixture_root) {
        bail!("capture request must live under the external fixture root")
    }
    let output_parent = arguments
        .raw_output
        .parent()
        .ok_or_else(|| anyhow!("raw output has no parent"))?;
    let output_parent = canonical_existing(output_parent, "raw output parent")?;
    require_external(&output_parent, &repository_root, "raw output parent")?;
    if !output_parent.starts_with(&fixture_root) || arguments.raw_output.exists() {
        bail!("raw output must be a new file under the external fixture root")
    }
    require_owner_only_directory(&output_parent, "raw output parent")?;

    let (authorization_sha, authorization_inputs) =
        validate_authorization(&authorization_record, &repository_root)?;
    let (request_snapshot, request_bytes) = OpenedInputSnapshot::read_bounded(
        &request_path,
        MAX_REQUEST_BYTES,
        "capture request",
        true,
    )?;
    let request: CaptureRequest = serde_json::from_slice(&request_bytes)
        .context("capture request is not exact-schema JSON")?;
    if request.schema_version != REQUEST_SCHEMA
        || request.authorization_document_sha256 != authorization_sha
        || request.source_revision != arguments.source_revision
        || request.model_revision != OFFICIAL_MODEL_REVISION
        || request.cases.len() != 2
    {
        bail!("capture request does not bind the authorized source and model")
    }
    let case_ids = request
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let kinds = request
        .cases
        .iter()
        .map(|case| case.kind.as_str())
        .collect::<BTreeSet<_>>();
    if case_ids.len() != 2 || kinds != BTreeSet::from(["multimodal", "text-only"]) {
        bail!("capture request must contain one text-only and one multimodal case")
    }
    for case in &request.cases {
        if !valid_lower_hex(&case.input_sha256, 64) || case_identity(case)? != case.input_sha256 {
            bail!("capture request input identity is invalid")
        }
        match case.kind.as_str() {
            "text-only" if case.image.is_none() && case.video.is_none() => {}
            "multimodal" if case.image.is_some() && case.video.is_some() => {}
            _ => bail!("capture case kind does not match its conditioning payload"),
        }
    }

    let device = Device::new_cuda(arguments.device_index).with_context(|| {
        format!(
            "failed to open {} for exact-BF16 capture",
            arguments.device_label
        )
    })?;
    let (artifacts, model_inputs, checkpoint_inputs) = conditioner_artifacts(&model_root)?;
    eprintln!("authenticating and loading the official exact-BF16 Qwen conditioner");
    // SAFETY: every path is canonical, the loader authenticates the complete
    // frozen artifact set, and this capture process never mutates model files.
    let loaded = unsafe { load_bf16_conditioner(&artifacts, &device) }
        .context("failed to load the official exact-BF16 Qwen conditioner")?;
    let profile = loaded.model.dtype_profile();
    if profile.parameter_dtype != DType::BF16
        || profile.output_dtype != DType::BF16
        || loaded.model.resident_language_layers() != 50
    {
        bail!("loaded conditioner is not the exact BF16 unnormalized layer-50 authority")
    }

    let mut outputs = Vec::with_capacity(request.cases.len());
    for case in &request.cases {
        let input = conditioner_input(case, &device)?;
        let mut last_checkpoint = None;
        let activation = loaded.model.encode(&input, &mut |checkpoint| {
            if last_checkpoint != Some(checkpoint) {
                eprintln!("{}: {checkpoint:?}", case.case_id);
                last_checkpoint = Some(checkpoint);
            }
            Ok(())
        })?;
        let shape = activation.dims().to_vec();
        if shape != [1, case.token_ids.len(), 5120] || activation.dtype() != DType::BF16 {
            bail!("conditioner emitted the wrong layer-50 shape or dtype")
        }
        outputs.push(RawCaseOutput {
            case_id: case.case_id.clone(),
            input_sha256: case.input_sha256.clone(),
            shape,
            bfloat16_le_base64: bfloat16_base64(&activation)?,
        });
    }
    request_snapshot.revalidate("capture request")?;
    for input in &authorization_inputs {
        input.revalidate("authorization input")?;
    }
    for input in &model_inputs {
        input.revalidate("official model metadata")?;
    }
    for input in &checkpoint_inputs {
        input.revalidate("official text encoder shard")?;
    }
    let output = RawCaptureOutput {
        schema_version: RAW_OUTPUT_SCHEMA,
        claim_marker: CAPTURE_MARKER,
        authorization_document_sha256: &authorization_sha,
        source_revision: &arguments.source_revision,
        model_revision: OFFICIAL_MODEL_REVISION,
        device: &arguments.device_label,
        dtype: "bfloat16",
        resident_language_layers: loaded.model.resident_language_layers(),
        cases: outputs,
    };
    let mut encoded = serde_json::to_vec_pretty(&output)?;
    encoded.push(b'\n');
    if encoded.len() as u64 > MAX_RAW_OUTPUT_BYTES {
        bail!("raw capture output exceeds its emitted-size bound")
    }
    exclusive_write(&arguments.raw_output, &encoded)
}

fn main() -> Result<()> {
    run()
}
