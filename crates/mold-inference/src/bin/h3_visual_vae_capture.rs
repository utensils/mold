//! Private MiniMax H3 visual-VAE capture adapter.
//!
//! This binary is unreachable without `dev-bins,h3-private-uat,cuda`. It
//! authenticates the reviewed Diffusers F32 checkpoint only after validating
//! the external authorization record, executes the released F32-encode / FP16
//! CUDA-decode recipe with exact math attention, and writes no media or model
//! data inside the repository.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, IndexOp, Tensor};
use mold_candle::minimax_h3::{
    inspect_visual_vae_config, validate_diffusers_weight_index, DecodeComputePolicy,
    MiniMaxH3VisualVae, MiniMaxH3VisualVaeConfig, NoopVisualVaeObserver,
    NoopVisualWeightReadObserver, UnitRgbPixels, ValidatedVisualVaeWeights, VisualAttentionBackend,
    VisualVaeCaptureEvidence,
};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const REQUEST_SCHEMA: &str = "mold.minimax-h3.visual-vae-capture-request.v1";
const RAW_OUTPUT_SCHEMA: &str = "mold.minimax-h3.visual-vae-raw-output.v1";
const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";
const OFFICIAL_MODEL_REVISION: &str = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86";
const OFFICIAL_VISUAL_INDEX_SHA256: &str =
    "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca";
const OFFICIAL_VISUAL_CONFIG_SHA256: &str =
    "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745";
const OFFICIAL_LICENSE_SHA256: &str =
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44";
const REVIEWED_AUTHORIZATION_SHA256: &str =
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d";
const CAPTURE_MARKER: &str = "mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1";
const INPUT_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.visual-vae-input.v1\0";
const PATTERN_ID: &str = "rgb-affine-uint8-v1";
const REQUIRED_SCOPES: [&str; 3] = [
    "checkpoint-execution",
    "fixture-capture",
    "generated-output-retention",
];
const MAX_REQUEST_BYTES: u64 = 64 * 1024;
const MAX_AUTHORIZATION_BYTES: u64 = 64 * 1024;
const MAX_AUTHORIZATION_SOURCE_BYTES: u64 = 1024 * 1024;
const MAX_RAW_OUTPUT_BYTES: u64 = 128 * 1024 * 1024;
const CASE_FRAMES: usize = 22;
const LATENT_FRAMES: usize = 7;
const CASE_HEIGHT: usize = 320;
const CASE_WIDTH: usize = 320;
const LATENT_HEIGHT: usize = 20;
const LATENT_WIDTH: usize = 20;
const SEAM_FRAMES: [usize; 3] = [0, 10, 21];
const OFFICIAL_VISUAL_SHARDS: [(&str, &str, &str); 3] = [
    (
        "official-video-vae-shard-00001-of-00003",
        "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374",
    ),
    (
        "official-video-vae-shard-00002-of-00003",
        "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
        "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
    ),
    (
        "official-video-vae-shard-00003-of-00003",
        "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
        "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",
    ),
];

fn usage() -> &'static str {
    "usage: h3_visual_vae_capture --model-root <absolute-official-model> \
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
        let sha256 = sha256_open_file(&file)?;
        let mut reader = file.try_clone()?;
        let mut bytes = Vec::with_capacity(identity.size_bytes as usize);
        reader.read_to_end(&mut bytes)?;
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
        let current = open_regular_file_no_follow(&self.path)
            .with_context(|| format!("failed to reopen {label} without following links"))?;
        if FileIdentity::from_metadata(&self.file.metadata()?) != self.identity
            || FileIdentity::from_metadata(&current.metadata()?) != self.identity
            || sha256_open_file(&self.file)? != self.sha256
        {
            bail!("{label} changed after authentication")
        }
        Ok(())
    }

    fn hash_exact(path: &Path, expected_sha256: &str, label: &str) -> Result<Self> {
        let requested = fs::symlink_metadata(path)?;
        if requested.file_type().is_symlink() || !requested.file_type().is_file() {
            bail!("{label} must be a direct regular file")
        }
        let file = open_regular_file_no_follow(path)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?);
        if identity != FileIdentity::from_metadata(&requested) {
            bail!("{label} changed while opening")
        }
        let sha256 = sha256_open_file(&file)?;
        if sha256 != expected_sha256 {
            bail!("{label} differs from its revision-bound LFS identity")
        }
        let snapshot = Self {
            path: path.to_path_buf(),
            file,
            identity,
            sha256,
        };
        snapshot.revalidate(label)?;
        Ok(snapshot)
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
    checkpoint_components: Vec<CheckpointComponent>,
    cases: Vec<CaptureCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointComponent {
    id: String,
    relative_path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCase {
    case_id: String,
    input_sha256: String,
    pattern: String,
    frames: usize,
    height: usize,
    width: usize,
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
    component_index_sha256: &'static str,
    component_fingerprint_sha256: &'a str,
    device: &'a str,
    stored_weight_dtype: &'static str,
    encoder_compute_dtype: &'static str,
    decoder_compute_dtype: &'static str,
    attention_backend: &'static str,
    cases: Vec<RawCaseOutput>,
}

#[derive(Debug, Serialize)]
struct RawCaseOutput {
    case_id: String,
    input_sha256: String,
    tensors: Vec<RawTensor>,
}

#[derive(Debug, Serialize)]
struct RawTensor {
    key: &'static str,
    shape: Vec<usize>,
    dtype: &'static str,
    little_endian_base64: String,
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
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
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
    let metadata = fs::symlink_metadata(path)?;
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
        bail!("authorization record does not cover the reviewed visual-VAE capture")
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
    let lines = std::str::from_utf8(&encoded)?.lines().collect::<Vec<_>>();
    if lines.first().copied() != Some(OFFICIAL_MODEL_REVISION) {
        bail!("component metadata differs from the official model revision")
    }
    Ok((lines.into_iter().map(str::to_owned).collect(), snapshot))
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
        .ok_or_else(|| anyhow!("component metadata lacks a canonical SHA-256"))?;
    Ok((digest.to_owned(), snapshot))
}

fn authenticate_visual_component(
    model_root: &Path,
    checkpoint_components: &[CheckpointComponent],
) -> Result<(
    MiniMaxH3VisualVaeConfig,
    ValidatedVisualVaeWeights,
    String,
    Vec<OpenedInputSnapshot>,
)> {
    let config_path = canonical_existing(&model_root.join("vae/config.json"), "visual VAE config")?;
    let index_path = canonical_existing(
        &model_root.join("vae/diffusion_pytorch_model.safetensors.index.json"),
        "visual VAE index",
    )?;
    let (config_snapshot, config_bytes) =
        OpenedInputSnapshot::read_bounded(&config_path, 1024 * 1024, "visual VAE config", false)?;
    let (index_snapshot, index_bytes) = OpenedInputSnapshot::read_bounded(
        &index_path,
        16 * 1024 * 1024,
        "visual VAE index",
        false,
    )?;
    if sha256_bytes(&config_bytes) != OFFICIAL_VISUAL_CONFIG_SHA256
        || sha256_bytes(&index_bytes) != OFFICIAL_VISUAL_INDEX_SHA256
    {
        bail!("visual VAE config or index differs from the reviewed authority")
    }
    let index: CheckpointIndex = serde_json::from_slice(&index_bytes)?;
    let shard_names = index.weight_map.into_values().collect::<BTreeSet<_>>();
    if shard_names.len() != 3 {
        bail!("official visual VAE index must name exactly three checkpoint shards")
    }
    let mut snapshots = vec![config_snapshot, index_snapshot];
    for relative in [
        "vae/config.json".to_owned(),
        "vae/diffusion_pytorch_model.safetensors.index.json".to_owned(),
    ] {
        let (_, metadata) = metadata_lines(model_root, &relative)?;
        snapshots.push(metadata);
    }
    if checkpoint_components.len() != OFFICIAL_VISUAL_SHARDS.len() {
        bail!("capture request lacks the complete reviewed visual shard authority")
    }
    for (component, (expected_id, expected_relative, expected_sha256)) in
        checkpoint_components.iter().zip(OFFICIAL_VISUAL_SHARDS)
    {
        if component.id != expected_id
            || component.relative_path != expected_relative
            || component.sha256 != expected_sha256
        {
            bail!("capture request visual shard authority differs from the reviewed pins")
        }
    }
    let expected_names = OFFICIAL_VISUAL_SHARDS
        .iter()
        .map(|(_, relative, _)| {
            Path::new(relative)
                .file_name()
                .and_then(|value| value.to_str())
                .expect("reviewed visual shard paths are valid UTF-8")
        })
        .collect::<BTreeSet<_>>();
    if shard_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_names
    {
        bail!("official visual VAE index differs from the reviewed shard set")
    }
    for (_, relative, expected_sha256) in OFFICIAL_VISUAL_SHARDS {
        let name = Path::new(relative)
            .file_name()
            .and_then(|value| value.to_str())
            .expect("reviewed visual shard paths are valid UTF-8");
        if Path::new(name).components().count() != 1 || !name.ends_with(".safetensors") {
            bail!("visual VAE index contains a non-canonical shard path")
        }
        let path = canonical_existing(&model_root.join(relative), "visual VAE shard")?;
        let (metadata_sha256, metadata) = metadata_sha256(model_root, relative)?;
        if metadata_sha256 != expected_sha256 {
            bail!("visual VAE shard metadata differs from the reviewed pin")
        }
        snapshots.push(OpenedInputSnapshot::hash_exact(
            &path,
            expected_sha256,
            "visual VAE shard",
        )?);
        snapshots.push(metadata);
    }
    let config = inspect_visual_vae_config(&config_path)?;
    let validated = validate_diffusers_weight_index(&config, &index_path, &model_root.join("vae"))?;
    let fingerprint = validated.component_fingerprint(&mut NoopVisualWeightReadObserver)?;
    Ok((config, validated, fingerprint, snapshots))
}

fn put_u64(digest: &mut Sha256, value: usize) -> Result<()> {
    digest.update(u64::try_from(value)?.to_le_bytes());
    Ok(())
}

fn put_string(digest: &mut Sha256, value: &str) -> Result<()> {
    put_u64(digest, value.len())?;
    digest.update(value.as_bytes());
    Ok(())
}

fn case_identity(case: &CaptureCase) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(INPUT_IDENTITY_DOMAIN);
    put_string(&mut digest, &case.case_id)?;
    put_string(&mut digest, &case.pattern)?;
    put_u64(&mut digest, case.frames)?;
    put_u64(&mut digest, case.height)?;
    put_u64(&mut digest, case.width)?;
    put_u64(&mut digest, 256)?;
    put_u64(&mut digest, 64)?;
    for value in [64usize, 256, SEAM_FRAMES[0], SEAM_FRAMES[1], SEAM_FRAMES[2]] {
        put_u64(&mut digest, value)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_cases(cases: &[CaptureCase]) -> Result<()> {
    let expected = [(
        "visual-vae-320x320x22-seed42-v1",
        CASE_FRAMES,
        CASE_HEIGHT,
        CASE_WIDTH,
    )];
    if cases.len() != expected.len() {
        bail!("capture request must contain the reviewed visual case")
    }
    for (case, (id, frames, height, width)) in cases.iter().zip(expected) {
        if case.case_id != id
            || case.pattern != PATTERN_ID
            || (case.frames, case.height, case.width) != (frames, height, width)
            || !valid_lower_hex(&case.input_sha256, 64)
            || case_identity(case)? != case.input_sha256
        {
            bail!("capture request visual case geometry or identity drifted")
        }
    }
    Ok(())
}

fn unit_rgb_pattern(case: &CaptureCase, device: &Device) -> Result<UnitRgbPixels> {
    let count = 3usize
        .checked_mul(case.frames)
        .and_then(|value| value.checked_mul(case.height))
        .and_then(|value| value.checked_mul(case.width))
        .ok_or_else(|| anyhow!("visual capture pattern dimensions overflow"))?;
    let mut values = Vec::with_capacity(count);
    for channel in 0..3 {
        for frame in 0..case.frames {
            for y in 0..case.height {
                for x in 0..case.width {
                    let value = match channel {
                        0 => 3 * x + 5 * y + 19 * frame,
                        1 => 7 * x + 11 * y + 17 + 23 * frame,
                        2 => 13 * x + 17 * y + 29 + 31 * frame,
                        _ => unreachable!(),
                    } % 256;
                    values.push(value as f32 / 255.0);
                }
            }
        }
    }
    UnitRgbPixels::new(Tensor::from_vec(
        values,
        (1, 3, case.frames, case.height, case.width),
        device,
    )?)
    .map_err(Into::into)
}

fn capture_tensor(key: &'static str, tensor: &Tensor, expected: DType) -> Result<RawTensor> {
    if tensor.dtype() != expected {
        bail!(
            "{key} has {:?} data instead of {expected:?}",
            tensor.dtype()
        )
    }
    let flattened = tensor.flatten_all()?;
    let (dtype, bytes) = match expected {
        DType::F32 => {
            let values = flattened.to_vec1::<f32>()?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                if !value.is_finite() {
                    bail!("{key} contains non-finite FP32 evidence")
                }
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ("float32", bytes)
        }
        DType::F16 => {
            let values = flattened.to_vec1::<half::f16>()?;
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for value in values {
                if !value.is_finite() {
                    bail!("{key} contains non-finite FP16 evidence")
                }
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            ("float16", bytes)
        }
        _ => bail!("unsupported raw visual evidence dtype {expected:?}"),
    };
    Ok(RawTensor {
        key,
        shape: tensor.dims().to_vec(),
        dtype,
        little_endian_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn tile_seam_probe(decoded: &Tensor) -> Result<Tensor> {
    if decoded.dims5()? != (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH)
        || decoded.dtype() != DType::F32
    {
        bail!("decoded visual evidence does not match the reviewed seam canvas")
    }
    let probe_axis = [0usize, 63, 64, 255, 256, 319];
    let mut values = Vec::new();
    for frame in SEAM_FRAMES {
        for channel in 0..3 {
            for seam in [64usize, 256] {
                for probe in probe_axis {
                    values.push(
                        decoded
                            .i((0, channel, frame, seam - 1, probe))?
                            .to_scalar::<f32>()?,
                    );
                    values.push(
                        decoded
                            .i((0, channel, frame, seam, probe))?
                            .to_scalar::<f32>()?,
                    );
                    values.push(
                        decoded
                            .i((0, channel, frame, probe, seam - 1))?
                            .to_scalar::<f32>()?,
                    );
                    values.push(
                        decoded
                            .i((0, channel, frame, probe, seam))?
                            .to_scalar::<f32>()?,
                    );
                }
            }
        }
    }
    Tensor::from_vec(values.clone(), values.len(), &Device::Cpu).map_err(Into::into)
}

fn raw_case(case: &CaptureCase, evidence: VisualVaeCaptureEvidence) -> Result<RawCaseOutput> {
    let latent_shape = (1, 24, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH);
    if evidence.normalized_pixels.dims5()? != (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH)
        || evidence.posterior_moments.dims5()?
            != (1, 48, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
        || evidence.posterior_seed42_noise.dims5()? != latent_shape
        || evidence.posterior_sample.dims5()? != latent_shape
        || evidence.posterior_fp16_roundtrip.dims5()? != latent_shape
        || evidence.normalized_latents.dims5()? != latent_shape
        || evidence.decoded_pixels.dims5()? != (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH)
    {
        bail!("captured visual tensors differ from the reviewed temporal case")
    }
    let tile_seams = tile_seam_probe(&evidence.decoded_pixels)?;
    Ok(RawCaseOutput {
        case_id: case.case_id.clone(),
        input_sha256: case.input_sha256.clone(),
        tensors: vec![
            capture_tensor(
                "pixel-normalization",
                &evidence.normalized_pixels,
                DType::F32,
            )?,
            capture_tensor("posterior-moments", &evidence.posterior_moments, DType::F32)?,
            capture_tensor(
                "posterior-seed-42",
                &evidence.posterior_seed42_noise,
                DType::F32,
            )?,
            capture_tensor("posterior-sample", &evidence.posterior_sample, DType::F32)?,
            capture_tensor(
                "posterior-fp16-roundtrip",
                &evidence.posterior_fp16_roundtrip,
                DType::F16,
            )?,
            capture_tensor(
                "latent-normalization",
                &evidence.normalized_latents,
                DType::F32,
            )?,
            capture_tensor("decoded-frames", &evidence.decoded_pixels, DType::F32)?,
            capture_tensor("tile-seams", &tile_seams, DType::F32)?,
        ],
    })
}

fn exclusive_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut destination = options.open(path)?;
    destination.write_all(bytes)?;
    destination.sync_all()?;
    Ok(())
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
    let output_parent = canonical_existing(
        arguments
            .raw_output
            .parent()
            .ok_or_else(|| anyhow!("raw output has no parent"))?,
        "raw output parent",
    )?;
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
    {
        bail!("capture request does not bind the authorized source and visual model")
    }
    validate_cases(&request.cases)?;

    let device = Device::new_cuda(arguments.device_index).with_context(|| {
        format!(
            "failed to open {} for visual-VAE capture",
            arguments.device_label
        )
    })?;
    let (config, validated, component_fingerprint, model_inputs) =
        authenticate_visual_component(&model_root, &request.checkpoint_components)?;
    let model = MiniMaxH3VisualVae::from_validated(
        config,
        &validated,
        &device,
        DecodeComputePolicy::Reference,
        VisualAttentionBackend::Math,
        &mut NoopVisualVaeObserver,
    )?;
    if model.stored_weight_dtype() != DType::F32
        || model.decode_compute_dtype() != DType::F16
        || model.attention_backend() != VisualAttentionBackend::Math
    {
        bail!("loaded visual VAE differs from the F32-storage/FP16-decode authority")
    }
    let mut cases = Vec::with_capacity(request.cases.len());
    for case in &request.cases {
        eprintln!("capturing {}", case.case_id);
        let evidence = model.capture_condition_evidence_unit(
            unit_rgb_pattern(case, &device)?,
            &mut NoopVisualVaeObserver,
        )?;
        cases.push(raw_case(case, evidence)?);
    }

    request_snapshot.revalidate("capture request")?;
    for input in &authorization_inputs {
        input.revalidate("authorization input")?;
    }
    for input in &model_inputs {
        input.revalidate("official visual VAE input")?;
    }
    let final_fingerprint = validated.component_fingerprint(&mut NoopVisualWeightReadObserver)?;
    if final_fingerprint != component_fingerprint {
        bail!("official visual VAE fingerprint changed during capture")
    }
    let output = RawCaptureOutput {
        schema_version: RAW_OUTPUT_SCHEMA,
        claim_marker: CAPTURE_MARKER,
        authorization_document_sha256: &authorization_sha,
        source_revision: &arguments.source_revision,
        model_revision: OFFICIAL_MODEL_REVISION,
        component_index_sha256: OFFICIAL_VISUAL_INDEX_SHA256,
        component_fingerprint_sha256: &component_fingerprint,
        device: &arguments.device_label,
        stored_weight_dtype: "float32",
        encoder_compute_dtype: "float32",
        decoder_compute_dtype: "float16",
        attention_backend: "math",
        cases,
    };
    let mut encoded = serde_json::to_vec_pretty(&output)?;
    encoded.push(b'\n');
    if encoded.len() as u64 > MAX_RAW_OUTPUT_BYTES {
        bail!("raw visual capture output exceeds its emitted-size bound")
    }
    exclusive_write(&arguments.raw_output, &encoded)
}

fn main() -> Result<()> {
    run()
}
