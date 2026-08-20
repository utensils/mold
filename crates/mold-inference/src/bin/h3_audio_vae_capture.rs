//! Private exact-FP32 MiniMax H3 AudioVAE capture adapter.
//!
//! This binary is unreachable without `dev-bins,h3-private-uat`. It refuses
//! non-CUDA execution, authenticates the reviewed external authorization and
//! Comfy folded FP32 checkpoint, and exposes only deterministic capture JSON.
//! It is not a model registry, download path, inference engine, or CLI surface.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    inspect_audio_vae_config_bytes, load_validated_audio_vae, AudioSoundtrackAssociation,
    AudioTensorLayout, AudioVaePhase, StereoWaveform, LATENT_CHANNELS, LATENT_ROWS_PER_SECOND,
    SAMPLES_PER_LATENT, SAMPLE_RATE,
};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const REQUEST_SCHEMA: &str = "mold.minimax-h3.audio-vae-capture-request.v1";
const RAW_OUTPUT_SCHEMA: &str = "mold.minimax-h3.audio-vae-raw-output.v1";
const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";
const OFFICIAL_MODEL_REVISION: &str = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86";
const OFFICIAL_AUDIO_CONFIG_SHA256: &str =
    "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6";
const COMFY_AUDIO_CHECKPOINT_SHA256: &str =
    "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48";
const COMFY_AUDIO_CHECKPOINT_BYTES: u64 = 605_254_808;
const OFFICIAL_LICENSE_SHA256: &str =
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44";
const REVIEWED_AUTHORIZATION_SHA256: &str =
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d";
const CAPTURE_MARKER: &str = "mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1";
const INPUT_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.audio-vae-input.v1\0";
const REVIEWED_CASE_ID: &str = "audio-vae-stereo-batch-v1";
const REVIEWED_BATCH: usize = 2;
const REVIEWED_CHANNELS: usize = 2;
const REVIEWED_SAMPLES: usize = 2_401;
const REQUIRED_SCOPES: [&str; 3] = [
    "checkpoint-execution",
    "fixture-capture",
    "generated-output-retention",
];
const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_AUTHORIZATION_BYTES: u64 = 64 * 1024;
const MAX_AUTHORIZATION_SOURCE_BYTES: u64 = 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_RAW_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;

fn usage() -> &'static str {
    "usage: h3_audio_vae_capture --config <absolute-official-config.json> \
     --weights <absolute-comfy-fp32.safetensors> --fixture-root <absolute-external-root> \
     --repository-root <absolute-repository-root> --authorization-record <absolute-record.json> \
     --request <absolute-request.json> --raw-output <absolute-new-output.json> \
     --source-revision <40-hex> --device <cuda:index>"
}

#[derive(Debug)]
struct Arguments {
    config: PathBuf,
    weights: PathBuf,
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

#[derive(Debug)]
struct CheckpointSnapshot {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    sha256: String,
}

impl CheckpointSnapshot {
    fn authenticate(path: &Path) -> Result<Self> {
        let requested =
            fs::symlink_metadata(path).context("failed to inspect AudioVAE checkpoint")?;
        if requested.file_type().is_symlink() || !requested.file_type().is_file() {
            bail!("AudioVAE checkpoint must be a direct regular file")
        }
        let file = open_regular_file_no_follow(path)
            .context("failed to open AudioVAE checkpoint without following links")?;
        let identity = FileIdentity::from_metadata(&file.metadata()?);
        if identity != FileIdentity::from_metadata(&requested)
            || identity.size_bytes != COMFY_AUDIO_CHECKPOINT_BYTES
        {
            bail!("AudioVAE checkpoint identity or byte length differs from the reviewed artifact")
        }
        let sha256 = sha256_open_file(&file).context("failed to hash AudioVAE checkpoint")?;
        if sha256 != COMFY_AUDIO_CHECKPOINT_SHA256 {
            bail!("AudioVAE checkpoint differs from the reviewed Comfy FP32 artifact")
        }
        let snapshot = Self {
            path: path.to_path_buf(),
            file,
            identity,
            sha256,
        };
        snapshot.revalidate()?;
        Ok(snapshot)
    }

    fn revalidate(&self) -> Result<()> {
        let descriptor_identity = FileIdentity::from_metadata(&self.file.metadata()?);
        let current = open_regular_file_no_follow(&self.path)
            .context("failed to reopen AudioVAE checkpoint without following links")?;
        if descriptor_identity != self.identity
            || FileIdentity::from_metadata(&current.metadata()?) != self.identity
            || sha256_open_file(&self.file)? != self.sha256
        {
            bail!("AudioVAE checkpoint changed after authentication")
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
    config_sha256: String,
    checkpoint_sha256: String,
    cases: Vec<CaptureCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCase {
    case_id: String,
    input_sha256: String,
    sample_rate: usize,
    batch: usize,
    channels: usize,
    samples: usize,
    waveform_f32_le_base64: String,
}

#[derive(Debug, Serialize)]
struct RawCaptureOutput<'a> {
    schema_version: &'static str,
    claim_marker: &'static str,
    authorization_document_sha256: &'a str,
    source_revision: &'a str,
    model_revision: &'static str,
    config_sha256: &'static str,
    checkpoint_sha256: &'static str,
    device: &'a str,
    dtype: &'static str,
    tensor_layout: &'static str,
    sample_rate: usize,
    samples_per_latent: usize,
    latent_rows_per_second: usize,
    encode_phases: Vec<&'static str>,
    decode_phases: Vec<&'static str>,
    cases: Vec<RawCaseOutput>,
}

#[derive(Debug, Serialize)]
struct RawCaseOutput {
    case_id: String,
    input_sha256: String,
    input_shape: [usize; 3],
    packed_mono_shape: [usize; 3],
    normalized_latent_shape: [usize; 4],
    normalized_latents_f32_le_base64: String,
    decoded_pcm_shape: [usize; 3],
    decoded_pcm_f32_le_base64: String,
}

fn parse_args() -> Result<Arguments> {
    let mut raw = env::args().skip(1);
    let mut values = BTreeMap::new();
    while let Some(flag) = raw.next() {
        if !flag.starts_with("--") {
            bail!(usage())
        }
        let value = raw.next().ok_or_else(|| anyhow!(usage()))?;
        if values.insert(flag, value).is_some() {
            bail!("capture argument was supplied more than once")
        }
    }
    let take = |name: &str| values.get(name).cloned().ok_or_else(|| anyhow!(usage()));
    let config = PathBuf::from(take("--config")?);
    let weights = PathBuf::from(take("--weights")?);
    let fixture_root = PathBuf::from(take("--fixture-root")?);
    let repository_root = PathBuf::from(take("--repository-root")?);
    let authorization_record = PathBuf::from(take("--authorization-record")?);
    let request = PathBuf::from(take("--request")?);
    let raw_output = PathBuf::from(take("--raw-output")?);
    let source_revision = take("--source-revision")?;
    let device_label = take("--device")?;
    let known = BTreeSet::from([
        "--config",
        "--weights",
        "--fixture-root",
        "--repository-root",
        "--authorization-record",
        "--request",
        "--raw-output",
        "--source-revision",
        "--device",
    ]);
    if values.keys().any(|key| !known.contains(key.as_str()))
        || !valid_lower_hex(&source_revision, 40)
    {
        bail!(usage())
    }
    let raw_index = device_label
        .strip_prefix("cuda:")
        .ok_or_else(|| anyhow!("capture device must be an indexed CUDA device"))?;
    let device_index = raw_index
        .parse::<usize>()
        .context("capture CUDA index is invalid")?;
    if raw_index != device_index.to_string() {
        bail!("capture CUDA index is not canonical")
    }
    for (path, label) in [
        (&config, "config"),
        (&weights, "weights"),
        (&fixture_root, "fixture root"),
        (&repository_root, "repository root"),
        (&authorization_record, "authorization record"),
        (&request, "request"),
        (&raw_output, "raw output"),
    ] {
        if !path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
        {
            bail!("{label} path must be absolute and normalized")
        }
    }
    Ok(Arguments {
        config,
        weights,
        fixture_root,
        repository_root,
        authorization_record,
        request,
        raw_output,
        source_revision,
        device_index,
        device_label,
    })
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_existing(path: &Path, label: &str) -> Result<PathBuf> {
    let canonical = fs::canonicalize(path).with_context(|| format!("failed to resolve {label}"))?;
    if !canonical.is_absolute() {
        bail!("{label} did not resolve to an absolute path")
    }
    Ok(canonical)
}

fn require_external(path: &Path, repository_root: &Path, label: &str) -> Result<()> {
    if path.starts_with(repository_root) {
        bail!("{label} must live outside the Mold repository")
    }
    Ok(())
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

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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
        bail!("authorization record does not cover the reviewed exact-FP32 capture")
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

fn put_u64(digest: &mut Sha256, value: usize) -> Result<()> {
    digest.update(
        u64::try_from(value)
            .context("capture input length exceeds u64")?
            .to_le_bytes(),
    );
    Ok(())
}

fn put_string(digest: &mut Sha256, value: &str) -> Result<()> {
    put_u64(digest, value.len())?;
    digest.update(value.as_bytes());
    Ok(())
}

fn decode_f32_payload(case: &CaptureCase) -> Result<(Vec<u8>, Vec<f32>)> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&case.waveform_f32_le_base64)
        .context("waveform FP32 payload is not canonical base64")?;
    let elements = case
        .batch
        .checked_mul(case.channels)
        .and_then(|value| value.checked_mul(case.samples))
        .ok_or_else(|| anyhow!("waveform shape overflows"))?;
    if bytes.len() != elements * 4 {
        bail!("waveform FP32 payload differs from its shape")
    }
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("waveform FP32 payload contains a non-finite value")
    }
    Ok((bytes, values))
}

fn case_identity(case: &CaptureCase) -> Result<String> {
    let (bytes, _) = decode_f32_payload(case)?;
    let mut digest = Sha256::new();
    digest.update(INPUT_IDENTITY_DOMAIN);
    put_string(&mut digest, &case.case_id)?;
    put_u64(&mut digest, case.sample_rate)?;
    put_u64(&mut digest, case.batch)?;
    put_u64(&mut digest, case.channels)?;
    put_u64(&mut digest, case.samples)?;
    put_u64(&mut digest, bytes.len())?;
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn f32_base64(tensor: &Tensor) -> Result<String> {
    if tensor.dtype() != DType::F32 {
        bail!("AudioVAE capture tensor is not FP32")
    }
    let values = tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("AudioVAE capture tensor contains a non-finite value")
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

fn phase_name(phase: AudioVaePhase) -> &'static str {
    match phase {
        AudioVaePhase::EncodePreprocess => "encode-preprocess",
        AudioVaePhase::EncodeDac => "encode-dac",
        AudioVaePhase::EncodeProjection => "encode-projection",
        AudioVaePhase::EncodePosterior => "encode-posterior",
        AudioVaePhase::DecodeProjection => "decode-projection",
        AudioVaePhase::DecodeBigVgan => "decode-bigvgan",
        AudioVaePhase::DecodeOutput => "decode-output",
    }
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
    let config_path = canonical_existing(&arguments.config, "official AudioVAE config")?;
    let weights_path = canonical_existing(&arguments.weights, "Comfy AudioVAE checkpoint")?;
    let authorization_record =
        canonical_existing(&arguments.authorization_record, "authorization record")?;
    let request_path = canonical_existing(&arguments.request, "capture request")?;
    for (path, label) in [
        (&fixture_root, "fixture root"),
        (&config_path, "official AudioVAE config"),
        (&weights_path, "Comfy AudioVAE checkpoint"),
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
    let (config_snapshot, config_bytes) = OpenedInputSnapshot::read_bounded(
        &config_path,
        MAX_CONFIG_BYTES,
        "official AudioVAE config",
        false,
    )?;
    if config_snapshot.sha256 != OFFICIAL_AUDIO_CONFIG_SHA256 {
        bail!("official AudioVAE config differs from the manifest authority")
    }
    let config = inspect_audio_vae_config_bytes(&config_bytes)
        .context("official AudioVAE config is not the frozen Mold contract")?;
    config.validate_execution_dtype(DType::F32)?;
    if config.sample_rate != SAMPLE_RATE
        || config.hop_length()? != SAMPLES_PER_LATENT
        || config.latent_rows_per_second()? != LATENT_ROWS_PER_SECOND
        || config.latent_channels != LATENT_CHANNELS
    {
        bail!("AudioVAE config differs from the 32 kHz / 40 Hz FP32 authority")
    }
    let checkpoint_snapshot = CheckpointSnapshot::authenticate(&weights_path)?;

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
        || request.config_sha256 != OFFICIAL_AUDIO_CONFIG_SHA256
        || request.checkpoint_sha256 != COMFY_AUDIO_CHECKPOINT_SHA256
        || request.cases.len() != 1
    {
        bail!("capture request does not bind the authorized source and FP32 artifacts")
    }
    let case = request.cases.first().expect("length checked");
    if case.case_id != REVIEWED_CASE_ID
        || case.sample_rate != SAMPLE_RATE
        || case.batch != REVIEWED_BATCH
        || case.channels != REVIEWED_CHANNELS
        || case.samples != REVIEWED_SAMPLES
        || !valid_lower_hex(&case.input_sha256, 64)
        || case_identity(case)? != case.input_sha256
    {
        bail!("capture request differs from the reviewed deterministic stereo batch")
    }

    let device = Device::new_cuda(arguments.device_index).with_context(|| {
        format!(
            "failed to open {} for exact-FP32 capture",
            arguments.device_label
        )
    })?;
    eprintln!("authenticating and loading the reviewed Comfy FP32 AudioVAE");
    // SAFETY: the direct checkpoint was authenticated, its immutable identity
    // is retained and revalidated below, and this process never mutates it.
    let audio_vae = unsafe {
        load_validated_audio_vae(
            &weights_path,
            config.clone(),
            AudioTensorLayout::ComfyFolded,
            DType::F32,
            &device,
        )
    }
    .context("failed to load the reviewed Comfy FP32 AudioVAE")?;

    let (_, waveform_values) = decode_f32_payload(case)?;
    let waveform_tensor = Tensor::from_vec(
        waveform_values,
        (case.batch, case.channels, case.samples),
        &device,
    )?;
    let waveform = StereoWaveform::new(
        waveform_tensor,
        case.sample_rate,
        AudioSoundtrackAssociation::StandaloneReference { reference_index: 1 },
    )?;
    let packed_mono_shape = waveform.pack_mono_batch()?.dims3()?;
    let mut encode_phase_values = Vec::new();
    let latents = audio_vae.encode_with_phase_checkpoint(&waveform, &mut |phase| {
        encode_phase_values.push(phase);
        eprintln!("{}: {}", case.case_id, phase_name(phase));
        Ok(())
    })?;
    let mut decode_phase_values = Vec::new();
    let decoded = audio_vae.decode_with_phase_checkpoint(
        &latents,
        AudioSoundtrackAssociation::Generated,
        &mut |phase| {
            decode_phase_values.push(phase);
            eprintln!("{}: {}", case.case_id, phase_name(phase));
            Ok(())
        },
    )?;
    let expected_encode = [
        AudioVaePhase::EncodePreprocess,
        AudioVaePhase::EncodeDac,
        AudioVaePhase::EncodeProjection,
        AudioVaePhase::EncodePosterior,
    ];
    let expected_decode = [
        AudioVaePhase::DecodeProjection,
        AudioVaePhase::DecodeBigVgan,
        AudioVaePhase::DecodeOutput,
    ];
    if encode_phase_values != expected_encode || decode_phase_values != expected_decode {
        bail!("AudioVAE capture did not traverse the reviewed phase checkpoints")
    }
    let latent_shape = latents.normalized().dims4()?;
    let decoded_shape = decoded.samples().dims3()?;
    let padded_samples = config.padded_samples(case.samples)?;
    if latent_shape
        != (
            case.batch,
            LATENT_CHANNELS,
            case.channels,
            padded_samples / SAMPLES_PER_LATENT,
        )
        || decoded_shape != (case.batch, case.channels, padded_samples)
    {
        bail!("AudioVAE capture emitted the wrong stereo, latent, or decoded geometry")
    }

    request_snapshot.revalidate("capture request")?;
    config_snapshot.revalidate("official AudioVAE config")?;
    checkpoint_snapshot.revalidate()?;
    for input in &authorization_inputs {
        input.revalidate("authorization input")?;
    }
    let output = RawCaptureOutput {
        schema_version: RAW_OUTPUT_SCHEMA,
        claim_marker: CAPTURE_MARKER,
        authorization_document_sha256: &authorization_sha,
        source_revision: &arguments.source_revision,
        model_revision: OFFICIAL_MODEL_REVISION,
        config_sha256: OFFICIAL_AUDIO_CONFIG_SHA256,
        checkpoint_sha256: COMFY_AUDIO_CHECKPOINT_SHA256,
        device: &arguments.device_label,
        dtype: "float32",
        tensor_layout: "comfy-folded-weight-norm",
        sample_rate: SAMPLE_RATE,
        samples_per_latent: SAMPLES_PER_LATENT,
        latent_rows_per_second: LATENT_ROWS_PER_SECOND,
        encode_phases: encode_phase_values.into_iter().map(phase_name).collect(),
        decode_phases: decode_phase_values.into_iter().map(phase_name).collect(),
        cases: vec![RawCaseOutput {
            case_id: case.case_id.clone(),
            input_sha256: case.input_sha256.clone(),
            input_shape: [case.batch, case.channels, case.samples],
            packed_mono_shape: [
                packed_mono_shape.0,
                packed_mono_shape.1,
                packed_mono_shape.2,
            ],
            normalized_latent_shape: [
                latent_shape.0,
                latent_shape.1,
                latent_shape.2,
                latent_shape.3,
            ],
            normalized_latents_f32_le_base64: f32_base64(latents.normalized())?,
            decoded_pcm_shape: [decoded_shape.0, decoded_shape.1, decoded_shape.2],
            decoded_pcm_f32_le_base64: f32_base64(decoded.samples())?,
        }],
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
