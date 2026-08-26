//! Encrypted, file-first storage for media attached to durable queue jobs.
//!
//! The store deliberately has no queue database knowledge. A caller supplies
//! the queue owner and job identity, and those values plus the random media-set
//! identity are bound into every STREAM record's associated data. The manifest
//! is encrypted along with the media and authenticated by a consuming final
//! record before any plaintext staging path is returned.

use aead_stream::{DecryptorBE32, EncryptorBE32, StreamBE32};
use base64::Engine as _;
use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

const STORE_DIR: &str = "queue-media";
const STORE_VERSION_DIR: &str = "v1";
const KEY_FILE: &str = "master.key";
const GENERATION_ADMISSION_KEY_FILE: &str = "generation-admission.key";
const MAGIC: &[u8; 8] = b"MOLDQMS1";
const FORMAT_VERSION: u16 = 1;
const V2_MAGIC: &[u8; 8] = b"MOLDQMS2";
const V2_FORMAT_VERSION: u16 = 2;
const PROJECTION_VERSION: u16 = 1;
const PROJECTION_PLAINTEXT_BYTES: usize = 64;
const PROJECTION_NONCE_BYTES: usize = 24;
const PROJECTION_CIPHERTEXT_BYTES: usize = PROJECTION_PLAINTEXT_BYTES + AEAD_TAG_BYTES;
const PROJECTION_HEADER_BYTES: usize =
    V2_MAGIC.len() + PROJECTION_NONCE_BYTES + PROJECTION_CIPHERTEXT_BYTES;
const OPERATION_RECEIPT_VERSION: u16 = 1;
const OPERATION_RECEIPT_PLAINTEXT_BYTES: usize = 2 + 2 + 64;
const ADMISSION_AUTHORITY_VERSION: u16 = 1;
const MAX_ADMISSION_AUTHORITY_PLAINTEXT_BYTES: usize = 1024;
const NONCE_PREFIX_BYTES: usize = 19;
const KEY_BYTES: usize = 32;
const CHUNK_BYTES: usize = 1024 * 1024;
const AEAD_TAG_BYTES: usize = 16;
const DATA_HEADER_BYTES: usize = 9;
const MAX_CIPHERTEXT_FRAME: usize = CHUNK_BYTES + DATA_HEADER_BYTES + AEAD_TAG_BYTES;
const BUNDLE_SUFFIX: &str = ".qms";
const OPERATION_FINGERPRINT_VERSION_SHA256_V1: u16 = 1;
const RUNTIME_STAGING_PREFIX: &str = "runtime-";
const RUNTIME_STAGING_CLAIM: &str = ".claim.lock";
const RUNTIME_STAGING_SWEEP: &str = ".sweep.lock";
const JOB_CLEANUP_LOCK: &str = ".cleanup.lock";
pub(crate) const PROJECTED_EDIT_DIMENSION_SLOTS: usize =
    mold_core::validation::FLUX2_DEV_MAX_REFERENCE_IMAGES;
const PROJECTION_EDIT_SLOTS_END: usize = 20 + PROJECTED_EDIT_DIMENSION_SLOTS * 9;

const _: () = assert!(PROJECTION_EDIT_SLOTS_END <= 56);

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SealFrameKind {
    Begin,
    Data,
    Manifest,
    Final,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SealTestFailure {
    MediaRead,
    Encrypt(SealFrameKind),
    Write(SealFrameKind),
    HardLink,
    HardLinkCollision,
    DestinationSync,
    StagingUnlink,
    StagingSync,
}

#[cfg(test)]
const TEST_COLLISION_BYTES: &[u8] = b"pre-existing collision must survive";

#[cfg(test)]
thread_local! {
    static SEAL_TEST_FAILURE: std::cell::Cell<Option<SealTestFailure>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
struct SealTestFailureGuard;

#[cfg(test)]
impl Drop for SealTestFailureGuard {
    fn drop(&mut self) {
        SEAL_TEST_FAILURE.set(None);
    }
}

#[cfg(test)]
fn inject_seal_test_failure(failure: SealTestFailure) -> SealTestFailureGuard {
    SEAL_TEST_FAILURE.set(Some(failure));
    SealTestFailureGuard
}

#[cfg(test)]
fn take_seal_test_failure(failure: SealTestFailure) -> bool {
    SEAL_TEST_FAILURE.get() == Some(failure) && {
        SEAL_TEST_FAILURE.set(None);
        true
    }
}

#[cfg(test)]
fn injected_seal_error(failure: SealTestFailure) -> Result<(), QueueMediaError> {
    if take_seal_test_failure(failure) {
        Err(std::io::Error::other(format!("injected queue-media failure: {failure:?}")).into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
fn seal_frame_kind(plaintext: &[u8]) -> SealFrameKind {
    match plaintext.first() {
        Some(b'B') => SealFrameKind::Begin,
        Some(b'D') => SealFrameKind::Data,
        Some(b'M') => SealFrameKind::Manifest,
        Some(b'F') => SealFrameKind::Final,
        _ => panic!("seal plaintext has an unknown record kind"),
    }
}

type Cipher = XChaCha20Poly1305;
type StreamNonce = aead_stream::Nonce<Cipher, StreamBE32<Cipher>>;

#[derive(Debug, thiserror::Error)]
pub enum QueueMediaError {
    #[error("queue-media I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("queue-media manifest encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("queue-media authentication failed")]
    Authentication,
    #[error("queue-media store is corrupt: {0}")]
    Corrupt(String),
    #[error("queue-media security requirement is unavailable: {0}")]
    SecurityUnavailable(String),
    #[error("queue-media path failed a security check: {0}")]
    InsecurePath(String),
    #[error("queue-media master key is missing while stored media exists")]
    MissingKeyWithExistingStore,
    #[error("queue-media master key does not exist")]
    MissingKey,
    #[error("durable-generation admission key is missing while authenticated receipts exist")]
    MissingAdmissionKeyWithReceipts,
    #[error("queue-media set already exists for owner {owner_id} job {job_id}")]
    JobAlreadySealed { owner_id: String, job_id: String },
    #[error("queue-media set was not found")]
    NotFound,
    #[error("invalid queue-media identity: {0}")]
    InvalidIdentity(String),
    #[error("queue-media scheduling projection is unavailable: {0:?}")]
    ProjectionUnavailable(QueueMediaProjectionFailure),
    #[error("V2 queue media must be hydrated through its authenticated mixed sinks")]
    MixedSinkHydrationRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMediaProjectionFailure {
    LegacyV1,
    Missing,
    Malformed,
    Authentication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDisposition {
    Loaded,
    Initialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMediaSecurityMode {
    UnixOwnerOnly,
    WindowsDpapiCurrentUser,
}

pub struct OpenedQueueMediaStore {
    pub store: QueueMediaStore,
    pub key_disposition: KeyDisposition,
}

#[derive(Clone)]
pub struct QueueMediaStore {
    root: PathBuf,
    key: Arc<Zeroizing<[u8; KEY_BYTES]>>,
    #[cfg(unix)]
    runtime_staging: Arc<QueueMediaRuntimeStaging>,
    #[cfg(test)]
    inspection_calls: Arc<AtomicUsize>,
}

#[cfg(unix)]
#[derive(Debug)]
struct QueueMediaRuntimeStaging {
    root: PathBuf,
    _claim: File,
}

/// A job lock keeps a shared claim on the cleanup namespace for its complete
/// lifetime. Terminal cleanup takes that claim exclusively before unlinking a
/// lock path, so a waiter can never continue on an unlinked lock inode while a
/// new caller creates a second lock for the same job.
struct QueueMediaJobLock {
    _cleanup_claim: File,
    _job: File,
}

#[cfg(unix)]
impl Drop for QueueMediaRuntimeStaging {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(%error, "failed to remove queue-media runtime plaintext staging");
            }
        }
    }
}

impl fmt::Debug for QueueMediaStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueueMediaStore")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MediaSetRef {
    pub owner_id: String,
    pub job_id: String,
    pub set_id: String,
}

pub struct SealMediaBytes {
    bytes: Zeroizing<Vec<u8>>,
    #[cfg(test)]
    zeroized: Option<Arc<AtomicBool>>,
}

impl SealMediaBytes {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Zeroizing::new(bytes),
            #[cfg(test)]
            zeroized: None,
        }
    }

    fn as_slice(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    #[cfg(test)]
    fn with_zeroize_probe(bytes: Vec<u8>, zeroized: Arc<AtomicBool>) -> Self {
        Self {
            bytes: Zeroizing::new(bytes),
            zeroized: Some(zeroized),
        }
    }
}

impl fmt::Debug for SealMediaBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SealMediaBytes")
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl Drop for SealMediaBytes {
    fn drop(&mut self) {
        self.bytes.zeroize();
        #[cfg(test)]
        if let Some(zeroized) = &self.zeroized {
            zeroized.store(self.bytes.iter().all(|byte| *byte == 0), Ordering::SeqCst);
        }
    }
}

#[derive(Debug)]
pub enum SealMediaSource {
    OpenFile(File),
    Bytes(SealMediaBytes),
}

/// A path-shaped seal source that has already passed the supported-platform
/// no-follow regular-file check. Keeping the `File` opaque prevents callers
/// from constructing an unchecked path source while allowing admission to
/// finish every fallible open before it releases its plaintext scrub owner.
#[cfg(unix)]
pub(crate) struct PreopenedSealMediaPath(File);

#[derive(Debug)]
pub struct SealMedia {
    pub role: String,
    pub name: String,
    pub source: SealMediaSource,
    pub sink: QueueMediaSink,
}

impl Drop for SealMedia {
    fn drop(&mut self) {
        self.role.zeroize();
        self.name.zeroize();
    }
}

impl SealMedia {
    #[cfg(unix)]
    pub(crate) fn preopen_path(
        path: &std::path::Path,
    ) -> Result<PreopenedSealMediaPath, QueueMediaError> {
        mold_core::secure_file::open_regular_file_no_follow(path)
            .map(PreopenedSealMediaPath)
            .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))
    }

    #[cfg(unix)]
    pub(crate) fn from_preopened_path(
        role: impl Into<String>,
        name: impl Into<String>,
        source: PreopenedSealMediaPath,
    ) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::OpenFile(source.0),
            sink: QueueMediaSink::PrivateStaging,
        }
    }

    #[cfg(unix)]
    pub fn path(
        role: impl Into<String>,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Result<Self, QueueMediaError> {
        let mut path = path.into();
        let source = Self::preopen_path(&path);
        scrub_path_buf(&mut path);
        Ok(Self::from_preopened_path(role, name, source?))
    }

    pub fn bytes(role: impl Into<String>, name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::Bytes(SealMediaBytes::new(bytes)),
            sink: QueueMediaSink::Memory,
        }
    }

    #[cfg(test)]
    fn bytes_with_zeroize_probe(
        role: impl Into<String>,
        name: impl Into<String>,
        bytes: Vec<u8>,
        zeroized: Arc<AtomicBool>,
    ) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::Bytes(SealMediaBytes::with_zeroize_probe(bytes, zeroized)),
            sink: QueueMediaSink::Memory,
        }
    }

    pub fn bytes_to_private_staging(
        role: impl Into<String>,
        name: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::Bytes(SealMediaBytes::new(bytes)),
            sink: QueueMediaSink::PrivateStaging,
        }
    }
}

#[cfg(unix)]
fn scrub_path_buf(path: &mut PathBuf) {
    use std::os::unix::ffi::OsStringExt as _;

    let mut bytes = std::mem::take(path).into_os_string().into_vec();
    bytes.zeroize();
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaManifestEntry {
    pub role: String,
    pub name: String,
    pub size_bytes: u64,
    pub sha256_hex: String,
    pub sink: QueueMediaSink,
}

impl fmt::Debug for MediaManifestEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaManifestEntry")
            .field("size_bytes", &self.size_bytes)
            .field("sink", &self.sink)
            .finish_non_exhaustive()
    }
}

/// A fingerprint of caller-defined canonical operation bytes.
///
/// The store deliberately does not define or persist the canonical operation.
/// It only keeps this value inside the encrypted, authenticated manifest so a
/// caller can resolve an ambiguous seal without putting media-derived hashes in
/// plaintext queue state.
#[derive(Clone, PartialEq, Eq)]
pub struct QueueMediaOperationFingerprint {
    version: u16,
    sha256_hex: String,
}

impl fmt::Debug for QueueMediaOperationFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueueMediaOperationFingerprint")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl QueueMediaOperationFingerprint {
    pub fn sha256_v1(canonical_operation: &[u8]) -> Self {
        Self::from_sha256_v1_digest(Sha256::digest(canonical_operation).into())
    }

    pub(crate) fn from_sha256_v1_digest(digest: [u8; 32]) -> Self {
        Self {
            version: OPERATION_FINGERPRINT_VERSION_SHA256_V1,
            sha256_hex: hex_encode(&digest),
        }
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    /// Compare authenticated operation identities without data-dependent early
    /// returns. Both values are fixed-width by construction/validation.
    pub fn constant_time_eq(&self, other: &Self) -> bool {
        use subtle::ConstantTimeEq as _;

        bool::from(
            self.version
                .to_be_bytes()
                .ct_eq(&other.version.to_be_bytes())
                & self
                    .sha256_hex
                    .as_bytes()
                    .ct_eq(other.sha256_hex.as_bytes()),
        )
    }
}

/// Randomized authenticated ciphertext stored in the existing opaque batch
/// outcome column. Debug output is deliberately redacted.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QueueMediaOperationReceipt(String);

impl QueueMediaOperationReceipt {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(encoded: impl Into<String>) -> Result<Self, QueueMediaError> {
        let encoded = encoded.into();
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .map_err(|_| QueueMediaError::Authentication)?;
        if bytes.len()
            != PROJECTION_NONCE_BYTES + OPERATION_RECEIPT_PLAINTEXT_BYTES + AEAD_TAG_BYTES
        {
            return Err(QueueMediaError::Authentication);
        }
        Ok(Self(encoded))
    }
}

/// Randomized authenticated ciphertext binding a model-family admission
/// envelope to one queue owner and job. The plaintext is interpreted only by
/// the model-family boundary; the media store supplies confidentiality,
/// integrity, and cross-row swap protection.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QueueMediaAdmissionAuthority(String);

impl QueueMediaAdmissionAuthority {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(encoded: impl Into<String>) -> Result<Self, QueueMediaError> {
        let encoded = encoded.into();
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .map_err(|_| QueueMediaError::Authentication)?;
        if bytes.len() < PROJECTION_NONCE_BYTES + 2 + AEAD_TAG_BYTES
            || bytes.len()
                > PROJECTION_NONCE_BYTES
                    + 2
                    + MAX_ADMISSION_AUTHORITY_PLAINTEXT_BYTES
                    + AEAD_TAG_BYTES
        {
            return Err(QueueMediaError::Authentication);
        }
        Ok(Self(encoded))
    }
}

impl fmt::Debug for QueueMediaAdmissionAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("QueueMediaAdmissionAuthority(<redacted>)")
    }
}

impl fmt::Debug for QueueMediaOperationReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("QueueMediaOperationReceipt(<redacted>)")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectedImageDimensions {
    Known { width: u32, height: u32 },
    UnreadableHeader,
}

/// Authenticated, payload-free facts used before a worker/device lease.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QueueMediaProjection {
    pub source_image: bool,
    pub source_video_inline: bool,
    pub source_video_path: bool,
    pub extend_video_inline: bool,
    pub extend_video_path: bool,
    pub keyframe_count: u32,
    pub identity_present: bool,
    pub identity_photograph_count: u32,
    pub edit_image_count: u32,
    pub edit_images: Vec<ProjectedImageDimensions>,
    pub mask_image: bool,
    pub control_image: bool,
    pub audio_inline: bool,
    pub audio_path: bool,
}

impl QueueMediaProjection {
    pub fn has_keyframes(&self) -> bool {
        self.keyframe_count > 0
    }

    pub fn edit_image_count(&self) -> usize {
        self.edit_image_count as usize
    }

    pub fn has_visual_conditioning(&self) -> bool {
        self.source_image
            || self.source_video_inline
            || self.source_video_path
            || self.extend_video_inline
            || self.extend_video_path
            || self.has_keyframes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueMediaSink {
    Memory,
    PrivateStaging,
}

#[derive(Clone, PartialEq, Eq)]
pub struct MediaSetManifest {
    pub media_set: MediaSetRef,
    pub operation_fingerprint: Option<QueueMediaOperationFingerprint>,
    pub entries: Vec<MediaManifestEntry>,
}

impl fmt::Debug for MediaSetManifest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaSetManifest")
            .field(
                "has_operation_fingerprint",
                &self.operation_fingerprint.is_some(),
            )
            .field("entry_count", &self.entries.len())
            .finish_non_exhaustive()
    }
}

pub struct DecryptedMedia {
    pub role: String,
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256_hex: String,
}

impl fmt::Debug for DecryptedMedia {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecryptedMedia")
            .field("size_bytes", &self.size_bytes)
            .finish_non_exhaustive()
    }
}

pub struct DecryptedMediaSet {
    pub manifest: MediaSetManifest,
    pub files: Vec<DecryptedMedia>,
    root: Option<PathBuf>,
    #[cfg(unix)]
    _runtime_staging: Arc<QueueMediaRuntimeStaging>,
}

impl fmt::Debug for DecryptedMediaSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecryptedMediaSet")
            .field("manifest_entry_count", &self.manifest.entries.len())
            .field("file_count", &self.files.len())
            .field("has_staging_root", &self.root.is_some())
            .finish_non_exhaustive()
    }
}

pub enum DecryptedQueueMediaPayload {
    Bytes(Vec<u8>),
    PrivatePath(PathBuf),
}

impl fmt::Debug for DecryptedQueueMediaPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => formatter
                .debug_struct("Bytes")
                .field("len", &bytes.len())
                .finish_non_exhaustive(),
            Self::PrivatePath(_) => formatter.write_str("PrivatePath(<redacted>)"),
        }
    }
}

pub struct DecryptedQueueMedia {
    pub role: String,
    pub name: String,
    pub payload: DecryptedQueueMediaPayload,
}

impl fmt::Debug for DecryptedQueueMedia {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecryptedQueueMedia")
            .field("payload", &self.payload)
            .finish_non_exhaustive()
    }
}

pub struct DecryptedQueueMediaSet {
    pub manifest: MediaSetManifest,
    pub media: Vec<DecryptedQueueMedia>,
    root: Option<PathBuf>,
    #[cfg(unix)]
    _runtime_staging: Arc<QueueMediaRuntimeStaging>,
}

impl fmt::Debug for DecryptedQueueMediaSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecryptedQueueMediaSet")
            .field("manifest_entry_count", &self.manifest.entries.len())
            .field("media_count", &self.media.len())
            .field("has_staging_root", &self.root.is_some())
            .finish_non_exhaustive()
    }
}

struct PlaintextStagingGuard {
    path: Option<PathBuf>,
}

struct SealPublicationGuard {
    staging: PathBuf,
    active: Option<PathBuf>,
    committed: bool,
}

impl SealPublicationGuard {
    fn new(staging: PathBuf) -> Self {
        Self {
            staging,
            active: None,
            committed: false,
        }
    }

    fn published(&mut self, active: PathBuf) {
        self.active = Some(active);
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for SealPublicationGuard {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(active) = self.active.take() {
            cleanup_owned_publication_path(&active);
        }
        cleanup_owned_publication_path(&self.staging);
    }
}

fn cleanup_owned_publication_path(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "failed to roll back queue-media publication");
        }
    }
    if let Some(parent) = path.parent() {
        if let Err(error) = crate::dir_sync::sync_directory(parent) {
            tracing::warn!(%error, path = %parent.display(), "failed to sync queue-media rollback");
        }
    }
}

impl PlaintextStagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn path(&self) -> &Path {
        self.path.as_deref().expect("staging guard is armed")
    }

    fn rename_to(&mut self, destination: PathBuf) -> Result<(), QueueMediaError> {
        fs::rename(self.path(), &destination)?;
        self.path = Some(destination);
        Ok(())
    }

    fn release(mut self) -> PathBuf {
        self.path.take().expect("staging guard is armed")
    }
}

impl Drop for PlaintextStagingGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

#[cfg(unix)]
fn establish_runtime_staging(parent: &Path) -> Result<QueueMediaRuntimeStaging, QueueMediaError> {
    ensure_private_dir(parent)?;
    let sweep_path = parent.join(RUNTIME_STAGING_SWEEP);
    let sweep = open_or_create_private_file(&sweep_path)?;
    sweep.lock_exclusive().map_err(|error| {
        QueueMediaError::SecurityUnavailable(format!(
            "cannot lock the queue-media staging claim mutex: {error}"
        ))
    })?;

    sweep_dead_runtime_staging_roots(parent)?;

    let root = parent.join(format!("{RUNTIME_STAGING_PREFIX}{}", random_hex(16)?));
    ensure_private_dir(&root)?;
    let claim_path = root.join(RUNTIME_STAGING_CLAIM);
    let claim = match create_private_file(&claim_path) {
        Ok(claim) => claim,
        Err(error) => {
            let _ = fs::remove_dir_all(&root);
            return Err(error);
        }
    };
    if let Err(error) = claim.try_lock_exclusive() {
        let _ = fs::remove_dir_all(&root);
        return Err(QueueMediaError::SecurityUnavailable(format!(
            "cannot claim the queue-media runtime staging root: {error}"
        )));
    }
    drop(sweep);
    Ok(QueueMediaRuntimeStaging {
        root,
        _claim: claim,
    })
}

#[cfg(unix)]
fn sweep_runtime_staging_parent(parent: &Path) -> Result<(), QueueMediaError> {
    ensure_private_dir(parent)?;
    let sweep = open_or_create_private_file(&parent.join(RUNTIME_STAGING_SWEEP))?;
    sweep.lock_exclusive().map_err(|error| {
        QueueMediaError::SecurityUnavailable(format!(
            "cannot lock the queue-media staging sweep mutex: {error}"
        ))
    })?;
    sweep_dead_runtime_staging_roots(parent)
}

#[cfg(unix)]
fn sweep_dead_runtime_staging_roots(parent: &Path) -> Result<(), QueueMediaError> {
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(set_id) = name.strip_prefix(RUNTIME_STAGING_PREFIX) else {
            continue;
        };
        if !valid_set_id(set_id) {
            continue;
        }
        let file_type = entry.file_type()?;
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        let root = entry.path();
        let claim_path = root.join(RUNTIME_STAGING_CLAIM);
        let claim = match mold_core::secure_file::open_regular_file_no_follow(&claim_path) {
            Ok(claim) => claim,
            Err(_) => continue,
        };
        match claim.try_lock_exclusive() {
            Ok(()) => {
                if let Err(error) = fs::remove_dir_all(&root) {
                    if error.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(%error, "failed to sweep dead queue-media staging root");
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => {
                tracing::warn!(%error, "could not inspect queue-media staging liveness");
            }
        }
    }
    Ok(())
}

impl DecryptedQueueMediaSet {
    pub fn close(mut self) -> Result<(), QueueMediaError> {
        self.remove_staging()
    }

    fn remove_staging(&mut self) -> Result<(), QueueMediaError> {
        if let Some(root) = self.root.take() {
            match fs::remove_dir_all(&root) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error.into()),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for DecryptedQueueMediaSet {
    fn drop(&mut self) {
        for item in &mut self.media {
            if let DecryptedQueueMediaPayload::Bytes(bytes) = &mut item.payload {
                bytes.zeroize();
            }
        }
        let _ = self.remove_staging();
    }
}

impl DecryptedMediaSet {
    /// Removes the private plaintext staging directory immediately.
    pub fn close(mut self) -> Result<(), QueueMediaError> {
        self.remove_staging()
    }

    fn remove_staging(&mut self) -> Result<(), QueueMediaError> {
        if let Some(root) = self.root.take() {
            match fs::remove_dir_all(&root) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error.into()),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for DecryptedMediaSet {
    fn drop(&mut self) {
        let _ = self.remove_staging();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrecognizedStoreEntry {
    pub path: PathBuf,
    pub set_id_hint: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreInspection {
    pub active: Vec<MediaSetRef>,
    pub retired: Vec<MediaSetRef>,
    pub staging: Vec<MediaSetRef>,
    pub unrecognized: Vec<UnrecognizedStoreEntry>,
}

/// One queue-media owner directory observed without traversing it.
///
/// Startup uses this only to report roots it does not own. A malformed or
/// symlink root has no trusted owner hint and is never followed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoreOwnerRoot {
    pub owner_id_hint: Option<String>,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireManifest {
    format_version: u16,
    owner_id: String,
    job_id: String,
    set_id: String,
    operation_fingerprint: Option<WireOperationFingerprint>,
    entries: Vec<WireManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireOperationFingerprint {
    version: u16,
    sha256_hex: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireManifestEntry {
    index: u32,
    role: String,
    name: String,
    size_bytes: u64,
    sha256_hex: String,
    chunk_count: u32,
    #[serde(default)]
    sink: Option<WireMediaSink>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireMediaSink {
    Memory,
    PrivateStaging,
}

impl From<QueueMediaSink> for WireMediaSink {
    fn from(value: QueueMediaSink) -> Self {
        match value {
            QueueMediaSink::Memory => Self::Memory,
            QueueMediaSink::PrivateStaging => Self::PrivateStaging,
        }
    }
}

impl From<WireMediaSink> for QueueMediaSink {
    fn from(value: WireMediaSink) -> Self {
        match value {
            WireMediaSink::Memory => Self::Memory,
            WireMediaSink::PrivateStaging => Self::PrivateStaging,
        }
    }
}

#[derive(Debug)]
struct DataObservation {
    index: u32,
    size_bytes: u64,
    sha256_hex: String,
    chunk_count: u32,
    sink: Option<QueueMediaSink>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoredState {
    Active,
    Retired,
    Staging,
}

impl StoredState {
    fn directory(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Retired => "retired",
            Self::Staging => "staging",
        }
    }
}

impl QueueMediaStore {
    /// Load or atomically initialize the durable-generation receipt key.
    /// This key deliberately shares neither readiness nor bytes with encrypted
    /// request media, so media-free admission survives a damaged media key.
    pub(crate) fn generation_admission_key(
        mold_home: impl AsRef<Path>,
        receipt_evidence_exists: bool,
    ) -> Result<Zeroizing<[u8; KEY_BYTES]>, QueueMediaError> {
        let mold_home = mold_home.as_ref();
        ensure_existing_directory(mold_home)?;
        let container = mold_home.join(STORE_DIR);
        ensure_private_dir(&container)?;
        let key_path = container.join(GENERATION_ADMISSION_KEY_FILE);
        if symlink_metadata_optional(&key_path)?.is_some() {
            load_master_key(&key_path)
        } else if receipt_evidence_exists {
            Err(QueueMediaError::MissingAdmissionKeyWithReceipts)
        } else {
            initialize_master_key(&key_path).map(|(key, _)| key)
        }
    }

    /// Opens the store, initializing a key only when no stored payload exists.
    pub fn open(mold_home: impl AsRef<Path>) -> Result<OpenedQueueMediaStore, QueueMediaError> {
        Self::open_or_initialize(mold_home, true)
    }

    /// Opens the store without permission to create a missing master key.
    pub fn open_existing(mold_home: impl AsRef<Path>) -> Result<Self, QueueMediaError> {
        Ok(Self::open_or_initialize(mold_home, false)?.store)
    }

    /// Opens a store and reports whether this call initialized its master key.
    ///
    /// A missing key is never regenerated over active, retired, or interrupted
    /// staging payloads. A present but malformed/inaccessible key always fails.
    pub fn open_or_initialize(
        mold_home: impl AsRef<Path>,
        allow_initialize: bool,
    ) -> Result<OpenedQueueMediaStore, QueueMediaError> {
        let mold_home = mold_home.as_ref();
        ensure_existing_directory(mold_home)?;
        let container = mold_home.join(STORE_DIR);
        let version_root = container.join(STORE_VERSION_DIR);
        let key_path = container.join(KEY_FILE);
        let key_existed = symlink_metadata_optional(&key_path)?.is_some();

        ensure_private_dir(&container)?;
        for path in [
            version_root.clone(),
            version_root.join("active"),
            version_root.join("retired"),
            version_root.join("staging"),
            version_root.join("locks"),
            version_root.join("ephemeral"),
        ] {
            ensure_private_dir(&path)?;
        }

        #[cfg(unix)]
        sweep_runtime_staging_parent(&version_root.join("ephemeral"))?;

        let existing_payload = store_contains_payload(&version_root)?;
        if !key_existed && existing_payload {
            return Err(QueueMediaError::MissingKeyWithExistingStore);
        }
        if !key_existed && !allow_initialize {
            return Err(QueueMediaError::MissingKey);
        }

        let (key, key_disposition) = if key_existed {
            (load_master_key(&key_path)?, KeyDisposition::Loaded)
        } else {
            initialize_master_key(&key_path)?
        };
        #[cfg(unix)]
        let runtime_staging = Arc::new(establish_runtime_staging(&version_root.join("ephemeral"))?);
        let store = Self {
            root: version_root,
            key: Arc::new(key),
            #[cfg(unix)]
            runtime_staging,
            #[cfg(test)]
            inspection_calls: Arc::new(AtomicUsize::new(0)),
        };
        store.cleanup_empty_job_artifacts_at_startup();
        Ok(OpenedQueueMediaStore {
            store,
            key_disposition,
        })
    }

    pub fn security_mode() -> Result<QueueMediaSecurityMode, QueueMediaError> {
        #[cfg(unix)]
        {
            Ok(QueueMediaSecurityMode::UnixOwnerOnly)
        }
        #[cfg(windows)]
        {
            Err(QueueMediaError::SecurityUnavailable(
                "Windows durable media hydration is not implemented".into(),
            ))
        }
        #[cfg(not(any(unix, windows)))]
        {
            Err(QueueMediaError::SecurityUnavailable(
                "no owner-only key protection is implemented for this platform".into(),
            ))
        }
    }

    /// Whether this platform can authenticate and hydrate every supported V2
    /// sink. DPAPI key protection alone is insufficient: admission remains
    /// dark until mixed memory/private-path hydration is implemented.
    pub(crate) const fn supports_mixed_hydration() -> bool {
        cfg!(unix)
    }

    /// Seals exactly one fresh, non-content-addressed bundle for a queue job.
    pub fn seal(
        &self,
        owner_id: &str,
        job_id: &str,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        self.seal_inner(owner_id, job_id, None, None, media)
    }

    /// Seals a bundle whose encrypted manifest carries a versioned operation
    /// fingerprint for ambiguity-safe idempotency checks.
    pub fn seal_with_operation_fingerprint(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: &QueueMediaOperationFingerprint,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        self.seal_inner(owner_id, job_id, Some(operation_fingerprint), None, media)
    }

    /// Seal a V2 bundle with a bounded authenticated projection before the
    /// media stream. This is the only format eligible for deferred scheduling.
    pub fn seal_v2_with_operation_fingerprint(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: &QueueMediaOperationFingerprint,
        projection: &QueueMediaProjection,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        self.seal_inner(
            owner_id,
            job_id,
            Some(operation_fingerprint),
            Some(projection),
            media,
        )
    }

    pub fn seal_operation_receipt_v1(
        &self,
        owner_id: &str,
        operation_id: &str,
        fingerprint: &QueueMediaOperationFingerprint,
    ) -> Result<QueueMediaOperationReceipt, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("operation", operation_id)?;
        validate_operation_fingerprint(WireOperationFingerprint {
            version: fingerprint.version,
            sha256_hex: fingerprint.sha256_hex.clone(),
        })?;
        let mut plaintext = Zeroizing::new([0_u8; OPERATION_RECEIPT_PLAINTEXT_BYTES]);
        plaintext[..2].copy_from_slice(&OPERATION_RECEIPT_VERSION.to_be_bytes());
        plaintext[2..4].copy_from_slice(&fingerprint.version.to_be_bytes());
        plaintext[4..].copy_from_slice(fingerprint.sha256_hex.as_bytes());
        let mut nonce = [0_u8; PROJECTION_NONCE_BYTES];
        random_fill(&mut nonce)?;
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let nonce = XNonce::try_from(nonce.as_slice()).expect("fixed-size nonce");
        let ciphertext = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext.as_slice(),
                    aad: &operation_receipt_aad(owner_id, operation_id),
                },
            )
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut receipt = Vec::with_capacity(nonce.len() + ciphertext.len());
        receipt.extend_from_slice(&nonce);
        receipt.extend_from_slice(&ciphertext);
        Ok(QueueMediaOperationReceipt(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(receipt),
        ))
    }

    pub fn open_operation_receipt_v1(
        &self,
        owner_id: &str,
        operation_id: &str,
        receipt: &QueueMediaOperationReceipt,
    ) -> Result<QueueMediaOperationFingerprint, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("operation", operation_id)?;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(receipt.0.as_bytes())
            .map_err(|_| QueueMediaError::Authentication)?;
        if encoded.len()
            != PROJECTION_NONCE_BYTES + OPERATION_RECEIPT_PLAINTEXT_BYTES + AEAD_TAG_BYTES
        {
            return Err(QueueMediaError::Authentication);
        }
        let (nonce, ciphertext) = encoded.split_at(PROJECTION_NONCE_BYTES);
        let nonce = XNonce::try_from(nonce).expect("validated nonce length");
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let plaintext = Zeroizing::new(
            cipher
                .decrypt(
                    &nonce,
                    Payload {
                        msg: ciphertext,
                        aad: &operation_receipt_aad(owner_id, operation_id),
                    },
                )
                .map_err(|_| QueueMediaError::Authentication)?,
        );
        if plaintext.len() != OPERATION_RECEIPT_PLAINTEXT_BYTES
            || u16::from_be_bytes(plaintext[..2].try_into().expect("sized"))
                != OPERATION_RECEIPT_VERSION
        {
            return Err(QueueMediaError::Authentication);
        }
        let version = u16::from_be_bytes(plaintext[2..4].try_into().expect("sized"));
        let sha256_hex = std::str::from_utf8(&plaintext[4..])
            .map_err(|_| QueueMediaError::Authentication)?
            .to_owned();
        validate_operation_fingerprint(WireOperationFingerprint {
            version,
            sha256_hex,
        })
        .map_err(|_| QueueMediaError::Authentication)
    }

    pub fn seal_admission_authority_v1(
        &self,
        owner_id: &str,
        job_id: &str,
        payload: &[u8],
    ) -> Result<QueueMediaAdmissionAuthority, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("job", job_id)?;
        if payload.is_empty() || payload.len() > MAX_ADMISSION_AUTHORITY_PLAINTEXT_BYTES {
            return Err(QueueMediaError::Authentication);
        }
        let mut plaintext = Zeroizing::new(Vec::with_capacity(2 + payload.len()));
        plaintext.extend_from_slice(&ADMISSION_AUTHORITY_VERSION.to_be_bytes());
        plaintext.extend_from_slice(payload);
        let mut nonce = [0_u8; PROJECTION_NONCE_BYTES];
        random_fill(&mut nonce)?;
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let nonce_ref = XNonce::try_from(nonce.as_slice()).expect("fixed-size nonce");
        let ciphertext = cipher
            .encrypt(
                &nonce_ref,
                Payload {
                    msg: plaintext.as_slice(),
                    aad: &admission_authority_aad(owner_id, job_id),
                },
            )
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut sealed = Vec::with_capacity(nonce.len() + ciphertext.len());
        sealed.extend_from_slice(&nonce);
        sealed.extend_from_slice(&ciphertext);
        Ok(QueueMediaAdmissionAuthority(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sealed),
        ))
    }

    pub fn open_admission_authority_v1(
        &self,
        owner_id: &str,
        job_id: &str,
        authority: &QueueMediaAdmissionAuthority,
    ) -> Result<Zeroizing<Vec<u8>>, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("job", job_id)?;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(authority.0.as_bytes())
            .map_err(|_| QueueMediaError::Authentication)?;
        if encoded.len() < PROJECTION_NONCE_BYTES + 2 + AEAD_TAG_BYTES
            || encoded.len()
                > PROJECTION_NONCE_BYTES
                    + 2
                    + MAX_ADMISSION_AUTHORITY_PLAINTEXT_BYTES
                    + AEAD_TAG_BYTES
        {
            return Err(QueueMediaError::Authentication);
        }
        let (nonce, ciphertext) = encoded.split_at(PROJECTION_NONCE_BYTES);
        let nonce = XNonce::try_from(nonce).expect("validated nonce length");
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut plaintext = Zeroizing::new(
            cipher
                .decrypt(
                    &nonce,
                    Payload {
                        msg: ciphertext,
                        aad: &admission_authority_aad(owner_id, job_id),
                    },
                )
                .map_err(|_| QueueMediaError::Authentication)?,
        );
        if plaintext.len() < 3
            || u16::from_be_bytes(plaintext[..2].try_into().expect("sized"))
                != ADMISSION_AUTHORITY_VERSION
        {
            return Err(QueueMediaError::Authentication);
        }
        plaintext.drain(..2);
        Ok(plaintext)
    }

    fn seal_inner(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: Option<&QueueMediaOperationFingerprint>,
        projection: Option<&QueueMediaProjection>,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        let result =
            self.seal_inner_locked(owner_id, job_id, operation_fingerprint, projection, media);
        if result.is_err() {
            self.cleanup_job_artifacts(owner_id, job_id);
        }
        result
    }

    fn seal_inner_locked(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: Option<&QueueMediaOperationFingerprint>,
        projection: Option<&QueueMediaProjection>,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("job", job_id)?;
        for item in &media {
            validate_manifest_label("role", &item.role)?;
            validate_manifest_label("name", &item.name)?;
        }

        let lock = self.lock_job(owner_id, job_id)?;
        if self.job_has_bundle(StoredState::Active, owner_id, job_id)?
            || self.job_has_bundle(StoredState::Retired, owner_id, job_id)?
            || self.job_has_bundle(StoredState::Staging, owner_id, job_id)?
        {
            drop(lock);
            return Err(QueueMediaError::JobAlreadySealed {
                owner_id: owner_id.into(),
                job_id: job_id.into(),
            });
        }

        let set_id = random_hex(16)?;
        let media_set = MediaSetRef {
            owner_id: owner_id.into(),
            job_id: job_id.into(),
            set_id,
        };
        let staging_path = self.bundle_path(StoredState::Staging, &media_set);
        ensure_private_dir(staging_path.parent().expect("bundle has parent"))?;
        let file = create_private_file(&staging_path)?;
        let mut publication = SealPublicationGuard::new(staging_path.clone());
        self.seal_file(
            &media_set,
            operation_fingerprint.cloned(),
            projection,
            &media,
            file,
        )?;

        let destination = self.bundle_path(StoredState::Active, &media_set);
        ensure_private_dir(destination.parent().expect("bundle has parent"))?;
        #[cfg(test)]
        injected_seal_error(SealTestFailure::HardLink)?;
        #[cfg(test)]
        if take_seal_test_failure(SealTestFailure::HardLinkCollision) {
            let mut collision = create_private_file(&destination)?;
            collision.write_all(TEST_COLLISION_BYTES)?;
            collision.sync_all()?;
        }
        match fs::hard_link(&staging_path, &destination) {
            Ok(()) => publication.published(destination.clone()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(QueueMediaError::JobAlreadySealed {
                    owner_id: owner_id.into(),
                    job_id: job_id.into(),
                });
            }
            Err(error) => return Err(error.into()),
        }
        #[cfg(test)]
        injected_seal_error(SealTestFailure::DestinationSync)?;
        crate::dir_sync::sync_directory(destination.parent().expect("bundle has parent"))?;
        #[cfg(test)]
        injected_seal_error(SealTestFailure::StagingUnlink)?;
        fs::remove_file(&staging_path)?;
        #[cfg(test)]
        injected_seal_error(SealTestFailure::StagingSync)?;
        crate::dir_sync::sync_directory(staging_path.parent().expect("bundle has parent"))?;
        drop(lock);
        publication.commit();
        Ok(media_set)
    }

    /// Authenticates every record and returns only the encrypted manifest.
    pub fn load(&self, media_set: &MediaSetRef) -> Result<MediaSetManifest, QueueMediaError> {
        self.decode_bundle(media_set, None)
            .map(|decoded| decoded.manifest)
    }

    /// Authenticate only the fixed-width first V2 record. No media frame or
    /// trailing manifest byte is read by this operation.
    pub fn open_projection(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<QueueMediaProjection, QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let path = self
            .locate_bundle(media_set)?
            .ok_or(QueueMediaError::NotFound)?;
        let mut file = mold_core::secure_file::open_regular_file_no_follow(&path)
            .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?;
        self.open_projection_from_reader(media_set, &mut file)
    }

    fn open_projection_from_reader(
        &self,
        media_set: &MediaSetRef,
        reader: &mut impl Read,
    ) -> Result<QueueMediaProjection, QueueMediaError> {
        let mut header = [0_u8; PROJECTION_HEADER_BYTES];
        if let Err(error) = reader.read_exact(&mut header[..MAGIC.len()]) {
            return Err(if error.kind() == std::io::ErrorKind::UnexpectedEof {
                QueueMediaError::ProjectionUnavailable(QueueMediaProjectionFailure::Missing)
            } else {
                error.into()
            });
        }
        if &header[..MAGIC.len()] == MAGIC {
            return Err(QueueMediaError::ProjectionUnavailable(
                QueueMediaProjectionFailure::LegacyV1,
            ));
        }
        if let Err(error) = reader.read_exact(&mut header[MAGIC.len()..]) {
            return Err(if error.kind() == std::io::ErrorKind::UnexpectedEof {
                QueueMediaError::ProjectionUnavailable(QueueMediaProjectionFailure::Missing)
            } else {
                error.into()
            });
        }
        if &header[..V2_MAGIC.len()] != V2_MAGIC {
            return Err(QueueMediaError::ProjectionUnavailable(
                QueueMediaProjectionFailure::Malformed,
            ));
        }
        let nonce =
            XNonce::try_from(&header[V2_MAGIC.len()..V2_MAGIC.len() + PROJECTION_NONCE_BYTES])
                .expect("fixed projection nonce length");
        let ciphertext = &header[V2_MAGIC.len() + PROJECTION_NONCE_BYTES..];
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let plaintext = cipher
            .decrypt(
                &nonce,
                Payload {
                    msg: ciphertext,
                    aad: &projection_aad(media_set),
                },
            )
            .map_err(|_| {
                QueueMediaError::ProjectionUnavailable(QueueMediaProjectionFailure::Authentication)
            })?;
        decode_projection(&plaintext).map_err(|_| {
            QueueMediaError::ProjectionUnavailable(QueueMediaProjectionFailure::Malformed)
        })
    }

    /// Authenticates the complete bundle before returning its encrypted-at-rest
    /// operation fingerprint.
    pub fn open_operation_fingerprint(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<Option<QueueMediaOperationFingerprint>, QueueMediaError> {
        Ok(self.load(media_set)?.operation_fingerprint)
    }

    /// Decode a V2 bundle into mixed sinks. Inline and identity values remain
    /// in process memory; only entries sealed as path-shaped media receive an
    /// owner-only ephemeral path.
    #[cfg(unix)]
    pub fn decrypt_mixed(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<DecryptedQueueMediaSet, QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let path = self
            .locate_bundle(media_set)?
            .ok_or(QueueMediaError::NotFound)?;
        let partial = self
            .runtime_staging
            .root
            .join(format!("{}.partial", random_hex(16)?));
        ensure_private_dir(&partial)?;
        let mut staging = PlaintextStagingGuard::new(partial);
        let decoded = self.decode_v2_from_path(media_set, &path, Some(staging.path()), true)?;
        crate::dir_sync::sync_directory(staging.path())?;
        let partial = staging.path().to_path_buf();
        let ready = partial.with_extension("ready");
        staging.rename_to(ready.clone())?;
        crate::dir_sync::sync_directory(&self.runtime_staging.root)?;
        let mut memory = decoded.memory;
        let mut media = Vec::with_capacity(decoded.manifest.entries.len());
        for (index, entry) in decoded.manifest.entries.iter().enumerate() {
            let payload = match entry.sink {
                QueueMediaSink::Memory => DecryptedQueueMediaPayload::Bytes(
                    memory
                        .remove(&(index as u32))
                        .ok_or(QueueMediaError::Authentication)?
                        .into_vec(),
                ),
                QueueMediaSink::PrivateStaging => {
                    DecryptedQueueMediaPayload::PrivatePath(ready.join(format!("{index:08}.media")))
                }
            };
            media.push(DecryptedQueueMedia {
                role: entry.role.clone(),
                name: entry.name.clone(),
                payload,
            });
        }
        if !memory.is_empty() {
            return Err(QueueMediaError::Corrupt(
                "decoded memory payload has no manifest entry".into(),
            ));
        }
        Ok(DecryptedQueueMediaSet {
            manifest: decoded.manifest,
            media,
            root: Some(staging.release()),
            _runtime_staging: Arc::clone(&self.runtime_staging),
        })
    }

    #[cfg(not(unix))]
    pub fn decrypt_mixed(
        &self,
        _media_set: &MediaSetRef,
    ) -> Result<DecryptedQueueMediaSet, QueueMediaError> {
        Err(QueueMediaError::SecurityUnavailable(
            "mixed queue-media hydration requires verified private staging support".into(),
        ))
    }

    /// Authenticates the complete bundle before publishing a private plaintext
    /// staging directory to the caller.
    #[cfg(unix)]
    pub fn decrypt_to_private_staging(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<DecryptedMediaSet, QueueMediaError> {
        let partial = self
            .runtime_staging
            .root
            .join(format!("{}.partial", random_hex(16)?));
        ensure_private_dir(&partial)?;
        let mut staging = PlaintextStagingGuard::new(partial);
        let decoded = self.decode_bundle(media_set, Some(staging.path()))?;
        crate::dir_sync::sync_directory(staging.path())?;
        let partial = staging.path().to_path_buf();
        let ready = partial.with_extension("ready");
        staging.rename_to(ready.clone())?;
        crate::dir_sync::sync_directory(&self.runtime_staging.root)?;
        let files = decoded
            .manifest
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| DecryptedMedia {
                role: entry.role.clone(),
                name: entry.name.clone(),
                path: ready.join(format!("{index:08}.media")),
                size_bytes: entry.size_bytes,
                sha256_hex: entry.sha256_hex.clone(),
            })
            .collect();
        Ok(DecryptedMediaSet {
            manifest: decoded.manifest,
            files,
            root: Some(staging.release()),
            _runtime_staging: Arc::clone(&self.runtime_staging),
        })
    }

    /// Windows key material is DPAPI-protected, but Rust's portable directory
    /// APIs cannot prove that a new plaintext directory has a current-user-only
    /// DACL. Refuse plaintext release until that proof is implemented.
    #[cfg(windows)]
    pub fn decrypt_to_private_staging(
        &self,
        _media_set: &MediaSetRef,
    ) -> Result<DecryptedMediaSet, QueueMediaError> {
        Err(QueueMediaError::SecurityUnavailable(
            "private plaintext staging requires a verified current-user-only Windows DACL".into(),
        ))
    }

    #[cfg(not(any(unix, windows)))]
    pub fn decrypt_to_private_staging(
        &self,
        _media_set: &MediaSetRef,
    ) -> Result<DecryptedMediaSet, QueueMediaError> {
        Err(QueueMediaError::SecurityUnavailable(
            "private plaintext staging is unavailable on this platform".into(),
        ))
    }

    pub fn retire(&self, media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
        self.move_bundle(media_set, StoredState::Active, StoredState::Retired)
    }

    pub fn restore(&self, media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
        self.move_bundle(media_set, StoredState::Retired, StoredState::Active)
    }

    /// Permanently deletes a set. Active sets cross the durable retired fence
    /// before unlink, so deletion never bypasses the lifecycle ordering.
    pub fn delete(&self, media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let result = (|| {
            let _lock = self.lock_job(&media_set.owner_id, &media_set.job_id)?;
            let active = self.bundle_path(StoredState::Active, media_set);
            let retired = self.bundle_path(StoredState::Retired, media_set);
            if let Some(metadata) = symlink_metadata_optional(&active)? {
                if !metadata.is_file() || metadata.file_type().is_symlink() {
                    return Err(QueueMediaError::InsecurePath(active.display().to_string()));
                }
                ensure_private_dir(retired.parent().expect("bundle has parent"))?;
                if symlink_metadata_optional(&retired)?.is_some() {
                    return Err(QueueMediaError::Corrupt(
                        "set exists in both active and retired states".into(),
                    ));
                }
                fs::rename(&active, &retired)?;
                crate::dir_sync::sync_directory(retired.parent().expect("bundle has parent"))?;
                crate::dir_sync::sync_directory(active.parent().expect("bundle has parent"))?;
            }
            let metadata = symlink_metadata_optional(&retired)?.ok_or(QueueMediaError::NotFound)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(QueueMediaError::InsecurePath(retired.display().to_string()));
            }
            fs::remove_file(&retired)?;
            crate::dir_sync::sync_directory(retired.parent().expect("bundle has parent"))?;
            Ok(())
        })();
        if result.is_ok() || matches!(&result, Err(QueueMediaError::NotFound)) {
            self.cleanup_job_artifacts(&media_set.owner_id, &media_set.job_id);
        }
        result
    }

    /// Deletes a fully authenticated interrupted publication from staging.
    pub fn delete_staging(&self, media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let result = (|| {
            let _lock = self.lock_job(&media_set.owner_id, &media_set.job_id)?;
            self.decode_bundle_at_state(media_set, StoredState::Staging)?;
            let staging = self.bundle_path(StoredState::Staging, media_set);
            fs::remove_file(&staging)?;
            crate::dir_sync::sync_directory(staging.parent().expect("bundle has parent"))?;
            Ok(())
        })();
        if result.is_ok() || matches!(&result, Err(QueueMediaError::NotFound)) {
            self.cleanup_job_artifacts(&media_set.owner_id, &media_set.job_id);
        }
        result
    }

    /// Enumerates and authenticates one owner's sets. Malformed entries are
    /// reported and left untouched so startup/GC can make an explicit choice.
    pub fn inspect_owner(&self, owner_id: &str) -> StoreInspection {
        #[cfg(test)]
        self.inspection_calls.fetch_add(1, Ordering::Relaxed);
        let mut report = StoreInspection::default();
        if !Self::supports_mixed_hydration() {
            report.unrecognized.push(UnrecognizedStoreEntry {
                path: self.root.clone(),
                set_id_hint: None,
                reason: "durable media hydration is unavailable on this platform".into(),
            });
            return report;
        }
        if let Err(error) = validate_identity("owner", owner_id) {
            report.unrecognized.push(UnrecognizedStoreEntry {
                path: self.root.clone(),
                set_id_hint: None,
                reason: error.to_string(),
            });
            return report;
        }
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            self.inspect_owner_state(owner_id, state, &mut report);
        }
        sort_inspection(&mut report);
        report
    }

    #[cfg(test)]
    pub(crate) fn inspection_calls(&self) -> usize {
        self.inspection_calls.load(Ordering::Relaxed)
    }

    /// Enumerates every structurally valid owner directory and reports all
    /// unknown entries without mutating them.
    pub fn inspect_all(&self) -> StoreInspection {
        let mut report = StoreInspection::default();
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            let state_root = self.root.join(state.directory());
            let entries = match fs::read_dir(&state_root) {
                Ok(entries) => entries,
                Err(error) => {
                    report.unrecognized.push(UnrecognizedStoreEntry {
                        path: state_root,
                        set_id_hint: None,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        report.unrecognized.push(UnrecognizedStoreEntry {
                            path: state_root.clone(),
                            set_id_hint: None,
                            reason: error.to_string(),
                        });
                        continue;
                    }
                };
                let path = entry.path();
                let owner = entry
                    .file_name()
                    .to_str()
                    .and_then(decode_component)
                    .filter(|owner| validate_identity("owner", owner).is_ok());
                match owner {
                    Some(owner) if entry.file_type().is_ok_and(|kind| kind.is_dir()) => {
                        self.inspect_owner_state(&owner, state, &mut report);
                    }
                    _ => report.unrecognized.push(UnrecognizedStoreEntry {
                        path,
                        set_id_hint: None,
                        reason: "invalid owner directory".into(),
                    }),
                }
            }
        }
        sort_inspection(&mut report);
        report
    }

    /// Enumerate direct owner roots without opening or traversing any owner
    /// other than `claimed_owner_id`.
    pub fn unclaimed_owner_roots(&self, claimed_owner_id: &str) -> Vec<StoreOwnerRoot> {
        let mut roots = std::collections::BTreeSet::new();
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            let state_root = self.root.join(state.directory());
            let entries = match fs::read_dir(&state_root) {
                Ok(entries) => entries,
                Err(error) => {
                    roots.insert(StoreOwnerRoot {
                        owner_id_hint: None,
                        description: format!(
                            "could not enumerate {} owner roots: {error}",
                            state.directory()
                        ),
                    });
                    continue;
                }
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        roots.insert(StoreOwnerRoot {
                            owner_id_hint: None,
                            description: format!(
                                "could not read an {} owner root: {error}",
                                state.directory()
                            ),
                        });
                        continue;
                    }
                };
                let owner = entry
                    .file_name()
                    .to_str()
                    .and_then(decode_component)
                    .filter(|owner| validate_identity("owner", owner).is_ok());
                let is_directory = entry.file_type().is_ok_and(|kind| kind.is_dir());
                match owner {
                    Some(owner) if owner == claimed_owner_id && is_directory => {}
                    Some(owner) if is_directory => {
                        roots.insert(StoreOwnerRoot {
                            owner_id_hint: Some(owner),
                            description: format!(
                                "unclaimed {} queue-media owner root",
                                state.directory()
                            ),
                        });
                    }
                    _ => {
                        roots.insert(StoreOwnerRoot {
                            owner_id_hint: None,
                            description: format!(
                                "unsafe or malformed {} queue-media owner root",
                                state.directory()
                            ),
                        });
                    }
                }
            }
        }
        roots.into_iter().collect()
    }

    fn cleanup_empty_job_artifacts_at_startup(&self) {
        let mut candidates = std::collections::BTreeSet::new();
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            collect_uuid_job_directories(&self.root.join(state.directory()), &mut candidates);
        }
        collect_uuid_job_locks(&self.root.join("locks"), &mut candidates);
        for (owner_id, job_id) in candidates {
            self.cleanup_job_artifacts(&owner_id, &job_id);
        }
    }

    fn seal_file(
        &self,
        media_set: &MediaSetRef,
        operation_fingerprint: Option<QueueMediaOperationFingerprint>,
        projection: Option<&QueueMediaProjection>,
        media: &[SealMedia],
        file: File,
    ) -> Result<(), QueueMediaError> {
        if let Some(projection) = projection {
            return self.seal_file_v2(media_set, operation_fingerprint, projection, media, file);
        }
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        let mut nonce_bytes = [0_u8; NONCE_PREFIX_BYTES];
        random_fill(&mut nonce_bytes)?;
        writer.write_all(&nonce_bytes)?;
        let nonce = StreamNonce::from(nonce_bytes);
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut encryptor = EncryptorBE32::from_aead(cipher, &nonce);
        let mut ordinal = 0_u32;
        let mut manifest_entries = Vec::with_capacity(media.len());

        for (index, item) in media.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| {
                QueueMediaError::Corrupt("too many media entries for the stream format".into())
            })?;
            let mut reader: Box<dyn Read> = match &item.source {
                SealMediaSource::OpenFile(file) => {
                    let mut file = file.try_clone()?;
                    file.seek(SeekFrom::Start(0))?;
                    Box::new(file)
                }
                SealMediaSource::Bytes(bytes) => Box::new(std::io::Cursor::new(bytes.as_slice())),
            };
            let mut digest = Sha256::new();
            let mut size_bytes = 0_u64;
            let mut chunk_count = 0_u32;
            let mut buffer = Zeroizing::new(vec![0_u8; CHUNK_BYTES]);
            loop {
                #[cfg(test)]
                injected_seal_error(SealTestFailure::MediaRead)?;
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
                size_bytes = size_bytes
                    .checked_add(read as u64)
                    .ok_or_else(|| QueueMediaError::Corrupt("media size overflow".into()))?;
                let mut plaintext = Zeroizing::new(Vec::with_capacity(DATA_HEADER_BYTES + read));
                plaintext.push(b'D');
                plaintext.extend_from_slice(&index.to_be_bytes());
                plaintext.extend_from_slice(&chunk_count.to_be_bytes());
                plaintext.extend_from_slice(&buffer[..read]);
                write_encrypted_frame(
                    &mut writer,
                    &mut encryptor,
                    media_set,
                    &mut ordinal,
                    plaintext,
                )?;
                chunk_count = chunk_count.checked_add(1).ok_or_else(|| {
                    QueueMediaError::Corrupt("media chunk counter overflow".into())
                })?;
            }
            manifest_entries.push(WireManifestEntry {
                index,
                role: item.role.clone(),
                name: item.name.clone(),
                size_bytes,
                sha256_hex: hex_encode(&digest.finalize()),
                chunk_count,
                sink: None,
            });
        }

        let manifest = WireManifest {
            format_version: FORMAT_VERSION,
            owner_id: media_set.owner_id.clone(),
            job_id: media_set.job_id.clone(),
            set_id: media_set.set_id.clone(),
            operation_fingerprint: operation_fingerprint.map(|fingerprint| {
                WireOperationFingerprint {
                    version: fingerprint.version,
                    sha256_hex: fingerprint.sha256_hex,
                }
            }),
            entries: manifest_entries,
        };
        let manifest_bytes = Zeroizing::new(serde_json::to_vec(&manifest)?);
        let manifest_digest = Sha256::digest(&*manifest_bytes);
        for chunk in manifest_bytes.chunks(CHUNK_BYTES) {
            let mut plaintext = Zeroizing::new(Vec::with_capacity(1 + chunk.len()));
            plaintext.push(b'M');
            plaintext.extend_from_slice(chunk);
            write_encrypted_frame(
                &mut writer,
                &mut encryptor,
                media_set,
                &mut ordinal,
                plaintext,
            )?;
        }
        let mut final_plaintext = Zeroizing::new(Vec::with_capacity(1 + 8 + 32));
        final_plaintext.push(b'F');
        final_plaintext.extend_from_slice(&(manifest_bytes.len() as u64).to_be_bytes());
        final_plaintext.extend_from_slice(&manifest_digest);
        let aad = frame_aad(
            media_set,
            ordinal,
            true,
            final_plaintext.len() + AEAD_TAG_BYTES,
        );
        #[cfg(test)]
        injected_seal_error(SealTestFailure::Encrypt(SealFrameKind::Final))?;
        let ciphertext = encryptor
            .encrypt_last(aead_stream::aead::Payload {
                msg: &final_plaintext,
                aad: &aad,
            })
            .map_err(|_| QueueMediaError::Authentication)?;
        #[cfg(test)]
        injected_seal_error(SealTestFailure::Write(SealFrameKind::Final))?;
        write_frame(&mut writer, true, &ciphertext)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }

    fn seal_file_v2(
        &self,
        media_set: &MediaSetRef,
        operation_fingerprint: Option<QueueMediaOperationFingerprint>,
        projection: &QueueMediaProjection,
        media: &[SealMedia],
        file: File,
    ) -> Result<(), QueueMediaError> {
        let mut writer = BufWriter::new(file);
        writer.write_all(V2_MAGIC)?;

        let projection_plaintext = Zeroizing::new(encode_projection(projection)?);
        let mut projection_nonce = [0_u8; PROJECTION_NONCE_BYTES];
        random_fill(&mut projection_nonce)?;
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let nonce = XNonce::try_from(projection_nonce.as_slice()).expect("fixed-size nonce");
        let projection_ciphertext = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: &projection_plaintext[..],
                    aad: &projection_aad(media_set),
                },
            )
            .map_err(|_| QueueMediaError::Authentication)?;
        debug_assert_eq!(projection_ciphertext.len(), PROJECTION_CIPHERTEXT_BYTES);
        writer.write_all(&projection_nonce)?;
        writer.write_all(&projection_ciphertext)?;

        let mut nonce_bytes = [0_u8; NONCE_PREFIX_BYTES];
        random_fill(&mut nonce_bytes)?;
        writer.write_all(&nonce_bytes)?;
        let nonce = StreamNonce::from(nonce_bytes);
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut encryptor = EncryptorBE32::from_aead(cipher, &nonce);
        // Projection is record ordinal zero; stream records begin at one.
        let mut ordinal = 1_u32;
        let mut manifest_entries = Vec::with_capacity(media.len());

        for (index, item) in media.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| {
                QueueMediaError::Corrupt("too many media entries for the stream format".into())
            })?;
            let begin_plaintext = Zeroizing::new(vec![
                b'B',
                index.to_be_bytes()[0],
                index.to_be_bytes()[1],
                index.to_be_bytes()[2],
                index.to_be_bytes()[3],
                match item.sink {
                    QueueMediaSink::Memory => 0,
                    QueueMediaSink::PrivateStaging => 1,
                },
            ]);
            write_encrypted_frame_for_version(
                &mut writer,
                &mut encryptor,
                media_set,
                V2_FORMAT_VERSION,
                &mut ordinal,
                begin_plaintext,
            )?;
            let mut reader: Box<dyn Read> = match &item.source {
                SealMediaSource::OpenFile(file) => {
                    let mut file = file.try_clone()?;
                    file.seek(SeekFrom::Start(0))?;
                    Box::new(file)
                }
                SealMediaSource::Bytes(bytes) => Box::new(std::io::Cursor::new(bytes.as_slice())),
            };
            let mut digest = Sha256::new();
            let mut size_bytes = 0_u64;
            let mut chunk_count = 0_u32;
            let mut buffer = Zeroizing::new(vec![0_u8; CHUNK_BYTES]);
            loop {
                #[cfg(test)]
                injected_seal_error(SealTestFailure::MediaRead)?;
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
                size_bytes = size_bytes
                    .checked_add(read as u64)
                    .ok_or_else(|| QueueMediaError::Corrupt("media size overflow".into()))?;
                let mut plaintext = Zeroizing::new(Vec::with_capacity(DATA_HEADER_BYTES + read));
                plaintext.push(b'D');
                plaintext.extend_from_slice(&index.to_be_bytes());
                plaintext.extend_from_slice(&chunk_count.to_be_bytes());
                plaintext.extend_from_slice(&buffer[..read]);
                write_encrypted_frame_for_version(
                    &mut writer,
                    &mut encryptor,
                    media_set,
                    V2_FORMAT_VERSION,
                    &mut ordinal,
                    plaintext,
                )?;
                chunk_count = chunk_count.checked_add(1).ok_or_else(|| {
                    QueueMediaError::Corrupt("media chunk counter overflow".into())
                })?;
            }
            manifest_entries.push(WireManifestEntry {
                index,
                role: item.role.clone(),
                name: item.name.clone(),
                size_bytes,
                sha256_hex: hex_encode(&digest.finalize()),
                chunk_count,
                sink: Some(item.sink.into()),
            });
        }

        let manifest = WireManifest {
            format_version: V2_FORMAT_VERSION,
            owner_id: media_set.owner_id.clone(),
            job_id: media_set.job_id.clone(),
            set_id: media_set.set_id.clone(),
            operation_fingerprint: operation_fingerprint.map(|fingerprint| {
                WireOperationFingerprint {
                    version: fingerprint.version,
                    sha256_hex: fingerprint.sha256_hex,
                }
            }),
            entries: manifest_entries,
        };
        let manifest_bytes = Zeroizing::new(serde_json::to_vec(&manifest)?);
        let manifest_digest = Sha256::digest(&*manifest_bytes);
        for chunk in manifest_bytes.chunks(CHUNK_BYTES) {
            let mut plaintext = Zeroizing::new(Vec::with_capacity(1 + chunk.len()));
            plaintext.push(b'M');
            plaintext.extend_from_slice(chunk);
            write_encrypted_frame_for_version(
                &mut writer,
                &mut encryptor,
                media_set,
                V2_FORMAT_VERSION,
                &mut ordinal,
                plaintext,
            )?;
        }
        let mut final_plaintext = Zeroizing::new(Vec::with_capacity(1 + 8 + 32));
        final_plaintext.push(b'F');
        final_plaintext.extend_from_slice(&(manifest_bytes.len() as u64).to_be_bytes());
        final_plaintext.extend_from_slice(&manifest_digest);
        let aad = frame_aad_for_version(
            media_set,
            V2_FORMAT_VERSION,
            ordinal,
            true,
            final_plaintext.len() + AEAD_TAG_BYTES,
        );
        #[cfg(test)]
        injected_seal_error(SealTestFailure::Encrypt(SealFrameKind::Final))?;
        let ciphertext = encryptor
            .encrypt_last(aead_stream::aead::Payload {
                msg: &final_plaintext,
                aad: &aad,
            })
            .map_err(|_| QueueMediaError::Authentication)?;
        #[cfg(test)]
        injected_seal_error(SealTestFailure::Write(SealFrameKind::Final))?;
        write_frame(&mut writer, true, &ciphertext)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }

    fn decode_bundle(
        &self,
        media_set: &MediaSetRef,
        output: Option<&Path>,
    ) -> Result<DecodedBundle, QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let path = self
            .locate_bundle(media_set)?
            .ok_or(QueueMediaError::NotFound)?;
        self.decode_bundle_from_path(media_set, &path, output)
    }

    fn decode_bundle_from_path(
        &self,
        media_set: &MediaSetRef,
        path: &Path,
        output: Option<&Path>,
    ) -> Result<DecodedBundle, QueueMediaError> {
        let file = mold_core::secure_file::open_regular_file_no_follow(path)
            .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0_u8; MAGIC.len()];
        reader.read_exact(&mut magic).map_err(map_truncation)?;
        if &magic == V2_MAGIC {
            if output.is_some() {
                return Err(QueueMediaError::MixedSinkHydrationRequired);
            }
            return self.decode_v2_from_reader(media_set, reader, output, false);
        }
        if &magic != MAGIC {
            return Err(QueueMediaError::Corrupt("unknown bundle format".into()));
        }
        let mut nonce_bytes = [0_u8; NONCE_PREFIX_BYTES];
        reader
            .read_exact(&mut nonce_bytes)
            .map_err(map_truncation)?;
        let nonce = StreamNonce::from(nonce_bytes);
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut decryptor = Some(DecryptorBE32::from_aead(cipher, &nonce));
        let mut ordinal = 0_u32;
        let mut manifest_bytes = Zeroizing::new(Vec::new());
        let mut observations = Vec::new();
        let mut current: Option<ObservedFile> = None;
        let mut manifest_started = false;
        let mut saw_final = false;

        while let Some((is_final, ciphertext)) = read_frame(&mut reader)? {
            let aad = frame_aad(media_set, ordinal, is_final, ciphertext.len());
            let stream = decryptor.take().expect("stream exists until final frame");
            let plaintext = Zeroizing::new(
                if is_final {
                    stream.decrypt_last(aead_stream::aead::Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    })
                } else {
                    let mut stream = stream;
                    let result = stream.decrypt_next(aead_stream::aead::Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    });
                    decryptor = Some(stream);
                    result
                }
                .map_err(|_| QueueMediaError::Authentication)?,
            );
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| QueueMediaError::Corrupt("stream counter overflow".into()))?;

            if is_final {
                finalize_observation(&mut current, &mut observations)?;
                validate_final_record(&plaintext, &manifest_bytes)?;
                let mut trailing = [0_u8; 1];
                if reader.read(&mut trailing)? != 0 {
                    return Err(QueueMediaError::Corrupt(
                        "bytes follow the final authenticated record".into(),
                    ));
                }
                saw_final = true;
                break;
            }
            match plaintext.first().copied() {
                Some(b'D') if !manifest_started => {
                    consume_data_record(&plaintext, output, &mut current, &mut observations)?
                }
                Some(b'M') => {
                    manifest_started = true;
                    finalize_observation(&mut current, &mut observations)?;
                    manifest_bytes.extend_from_slice(&plaintext[1..]);
                }
                Some(b'D') => {
                    return Err(QueueMediaError::Corrupt(
                        "media record follows manifest data".into(),
                    ));
                }
                _ => return Err(QueueMediaError::Corrupt("unknown stream record".into())),
            }
        }
        if !saw_final {
            return Err(QueueMediaError::Authentication);
        }
        let wire: WireManifest = serde_json::from_slice(&manifest_bytes)?;
        let manifest = validate_manifest(media_set, wire, &observations, output)?;
        manifest_bytes.zeroize();
        Ok(DecodedBundle {
            manifest,
            memory: BTreeMap::new(),
        })
    }

    fn decode_v2_from_path(
        &self,
        media_set: &MediaSetRef,
        path: &Path,
        output: Option<&Path>,
        mixed: bool,
    ) -> Result<DecodedBundle, QueueMediaError> {
        let file = mold_core::secure_file::open_regular_file_no_follow(path)
            .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0_u8; V2_MAGIC.len()];
        reader.read_exact(&mut magic).map_err(map_truncation)?;
        if &magic != V2_MAGIC {
            return Err(QueueMediaError::ProjectionUnavailable(if &magic == MAGIC {
                QueueMediaProjectionFailure::LegacyV1
            } else {
                QueueMediaProjectionFailure::Malformed
            }));
        }
        self.decode_v2_from_reader(media_set, reader, output, mixed)
    }

    fn decode_v2_from_reader(
        &self,
        media_set: &MediaSetRef,
        mut reader: BufReader<File>,
        output: Option<&Path>,
        mixed: bool,
    ) -> Result<DecodedBundle, QueueMediaError> {
        let mut projection_nonce = [0_u8; PROJECTION_NONCE_BYTES];
        reader
            .read_exact(&mut projection_nonce)
            .map_err(map_truncation)?;
        let mut projection_ciphertext = [0_u8; PROJECTION_CIPHERTEXT_BYTES];
        reader
            .read_exact(&mut projection_ciphertext)
            .map_err(map_truncation)?;
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let nonce = XNonce::try_from(projection_nonce.as_slice()).expect("fixed-size nonce");
        let projection_plaintext = cipher
            .decrypt(
                &nonce,
                Payload {
                    msg: &projection_ciphertext,
                    aad: &projection_aad(media_set),
                },
            )
            .map_err(|_| QueueMediaError::Authentication)?;
        let _projection = decode_projection(&projection_plaintext)?;

        let mut nonce_bytes = [0_u8; NONCE_PREFIX_BYTES];
        reader
            .read_exact(&mut nonce_bytes)
            .map_err(map_truncation)?;
        let nonce = StreamNonce::from(nonce_bytes);
        let cipher = Cipher::new_from_slice(self.key.as_ref().as_ref())
            .map_err(|_| QueueMediaError::Authentication)?;
        let mut decryptor = Some(DecryptorBE32::from_aead(cipher, &nonce));
        let mut ordinal = 1_u32;
        let mut manifest_bytes = Zeroizing::new(Vec::new());
        let mut observations = Vec::new();
        let mut current: Option<V2ObservedFile> = None;
        let mut manifest_started = false;
        let mut saw_final = false;
        let mut memory = BTreeMap::new();

        while let Some((is_final, ciphertext)) = read_frame(&mut reader)? {
            let aad = frame_aad_for_version(
                media_set,
                V2_FORMAT_VERSION,
                ordinal,
                is_final,
                ciphertext.len(),
            );
            let stream = decryptor.take().expect("stream exists until final frame");
            let plaintext = Zeroizing::new(
                if is_final {
                    stream.decrypt_last(aead_stream::aead::Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    })
                } else {
                    let mut stream = stream;
                    let result = stream.decrypt_next(aead_stream::aead::Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    });
                    decryptor = Some(stream);
                    result
                }
                .map_err(|_| QueueMediaError::Authentication)?,
            );
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| QueueMediaError::Corrupt("stream counter overflow".into()))?;

            if is_final {
                finalize_v2_observation(&mut current, &mut observations, &mut memory)?;
                validate_final_record(&plaintext, &manifest_bytes)?;
                let mut trailing = [0_u8; 1];
                if reader.read(&mut trailing)? != 0 {
                    return Err(QueueMediaError::Corrupt(
                        "bytes follow the final authenticated record".into(),
                    ));
                }
                saw_final = true;
                break;
            }
            match plaintext.first().copied() {
                Some(b'B') if !manifest_started => begin_v2_observation(
                    &plaintext,
                    output,
                    mixed,
                    &mut current,
                    &mut observations,
                    &mut memory,
                )?,
                Some(b'D') if !manifest_started => {
                    consume_v2_data_record(&plaintext, &mut current)?
                }
                Some(b'M') => {
                    manifest_started = true;
                    finalize_v2_observation(&mut current, &mut observations, &mut memory)?;
                    manifest_bytes.extend_from_slice(&plaintext[1..]);
                }
                Some(b'B' | b'D') => {
                    return Err(QueueMediaError::Corrupt(
                        "media record follows manifest data".into(),
                    ));
                }
                _ => return Err(QueueMediaError::Corrupt("unknown stream record".into())),
            }
        }
        if !saw_final {
            return Err(QueueMediaError::Authentication);
        }
        let wire: WireManifest = serde_json::from_slice(&manifest_bytes)?;
        let manifest = validate_manifest(media_set, wire, &observations, output)?;
        manifest_bytes.zeroize();
        Ok(DecodedBundle { manifest, memory })
    }

    fn lock_job(&self, owner_id: &str, job_id: &str) -> Result<QueueMediaJobLock, QueueMediaError> {
        let locks_root = self.root.join("locks");
        let cleanup_claim = open_or_create_private_file(&locks_root.join(JOB_CLEANUP_LOCK))?;
        FileExt::lock_shared(&cleanup_claim)?;
        let owner_dir = locks_root.join(encode_component(owner_id));
        ensure_private_dir(&owner_dir)?;
        let lock_path = owner_dir.join(format!("{}.lock", encode_component(job_id)));
        let lock = open_or_create_private_file(&lock_path)?;
        lock.lock_exclusive()?;
        Ok(QueueMediaJobLock {
            _cleanup_claim: cleanup_claim,
            _job: lock,
        })
    }

    fn cleanup_job_artifacts(&self, owner_id: &str, job_id: &str) {
        if let Err(error) = self.cleanup_job_artifacts_inner(owner_id, job_id) {
            tracing::warn!(
                %error,
                owner_id,
                job_id,
                "left queue-media job artifacts untouched because cleanup was not provably safe"
            );
        }
    }

    fn cleanup_job_artifacts_inner(
        &self,
        owner_id: &str,
        job_id: &str,
    ) -> Result<(), QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("job", job_id)?;
        ensure_private_dir(&self.root)?;
        let locks_root = self.root.join("locks");
        ensure_private_dir(&locks_root)?;
        let cleanup_claim = open_or_create_private_file(&locks_root.join(JOB_CLEANUP_LOCK))?;
        cleanup_claim.lock_exclusive()?;

        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            let state_root = self.root.join(state.directory());
            ensure_private_dir(&state_root)?;
            let owner_dir = state_root.join(encode_component(owner_id));
            if let Some(metadata) = symlink_metadata_optional(&owner_dir)? {
                verify_private_directory_metadata(&owner_dir, &metadata)?;
            }
            let job_dir = owner_dir.join(encode_component(job_id));
            if !remove_empty_private_directory(&job_dir)? {
                return Ok(());
            }
        }

        let lock_owner = locks_root.join(encode_component(owner_id));
        match symlink_metadata_optional(&lock_owner)? {
            None => {}
            Some(metadata) => {
                verify_private_directory_metadata(&lock_owner, &metadata)?;
                let lock_path = lock_owner.join(format!("{}.lock", encode_component(job_id)));
                if let Some(metadata) = symlink_metadata_optional(&lock_path)? {
                    if !metadata.is_file() || metadata.file_type().is_symlink() {
                        return Err(QueueMediaError::InsecurePath(
                            lock_path.display().to_string(),
                        ));
                    }
                    let lock = mold_core::secure_file::open_regular_file_no_follow(&lock_path)
                        .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?;
                    verify_private_lock_file(&lock_path, &lock)?;
                    lock.lock_exclusive()?;
                    fs::remove_file(&lock_path)?;
                    crate::dir_sync::sync_directory(&lock_owner)?;
                }
            }
        }

        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            let owner_dir = self
                .root
                .join(state.directory())
                .join(encode_component(owner_id));
            remove_empty_private_directory(&owner_dir)?;
        }
        remove_empty_private_directory(&lock_owner)?;
        Ok(())
    }

    fn move_bundle(
        &self,
        media_set: &MediaSetRef,
        source_state: StoredState,
        destination_state: StoredState,
    ) -> Result<(), QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let _lock = self.lock_job(&media_set.owner_id, &media_set.job_id)?;
        let source = self.bundle_path(source_state, media_set);
        let source_metadata =
            symlink_metadata_optional(&source)?.ok_or(QueueMediaError::NotFound)?;
        if !source_metadata.is_file() || source_metadata.file_type().is_symlink() {
            return Err(QueueMediaError::InsecurePath(source.display().to_string()));
        }
        let destination = self.bundle_path(destination_state, media_set);
        ensure_private_dir(destination.parent().expect("bundle has parent"))?;
        if symlink_metadata_optional(&destination)?.is_some() {
            return Err(QueueMediaError::Corrupt(
                "both lifecycle states contain the same set".into(),
            ));
        }
        fs::rename(&source, &destination)?;
        crate::dir_sync::sync_directory(destination.parent().expect("bundle has parent"))?;
        crate::dir_sync::sync_directory(source.parent().expect("bundle has parent"))?;
        Ok(())
    }

    fn locate_bundle(&self, media_set: &MediaSetRef) -> Result<Option<PathBuf>, QueueMediaError> {
        let active = self.bundle_path(StoredState::Active, media_set);
        let retired = self.bundle_path(StoredState::Retired, media_set);
        let active_exists = symlink_metadata_optional(&active)?.is_some();
        let retired_exists = symlink_metadata_optional(&retired)?.is_some();
        match (active_exists, retired_exists) {
            (true, false) => Ok(Some(active)),
            (false, true) => Ok(Some(retired)),
            (false, false) => Ok(None),
            (true, true) => Err(QueueMediaError::Corrupt(
                "set exists in both active and retired states".into(),
            )),
        }
    }

    fn bundle_path(&self, state: StoredState, media_set: &MediaSetRef) -> PathBuf {
        self.root
            .join(state.directory())
            .join(encode_component(&media_set.owner_id))
            .join(encode_component(&media_set.job_id))
            .join(format!("{}{BUNDLE_SUFFIX}", media_set.set_id))
    }

    fn job_has_bundle(
        &self,
        state: StoredState,
        owner_id: &str,
        job_id: &str,
    ) -> Result<bool, QueueMediaError> {
        let directory = self
            .root
            .join(state.directory())
            .join(encode_component(owner_id))
            .join(encode_component(job_id));
        match fs::read_dir(directory) {
            Ok(mut entries) => Ok(entries.next().transpose()?.is_some()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn inspect_owner_state(
        &self,
        owner_id: &str,
        state: StoredState,
        report: &mut StoreInspection,
    ) {
        let owner_path = self
            .root
            .join(state.directory())
            .join(encode_component(owner_id));
        let owner_metadata = match fs::symlink_metadata(&owner_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => {
                report.unrecognized.push(UnrecognizedStoreEntry {
                    path: owner_path,
                    set_id_hint: None,
                    reason: error.to_string(),
                });
                return;
            }
        };
        if owner_metadata.file_type().is_symlink() || !owner_metadata.is_dir() {
            report.unrecognized.push(UnrecognizedStoreEntry {
                path: owner_path,
                set_id_hint: None,
                reason: "owner root is not a direct directory".into(),
            });
            return;
        }
        let jobs = match fs::read_dir(&owner_path) {
            Ok(jobs) => jobs,
            Err(error) => {
                report.unrecognized.push(UnrecognizedStoreEntry {
                    path: owner_path,
                    set_id_hint: None,
                    reason: error.to_string(),
                });
                return;
            }
        };
        for job_entry in jobs {
            let job_entry = match job_entry {
                Ok(entry) => entry,
                Err(error) => {
                    report.unrecognized.push(UnrecognizedStoreEntry {
                        path: owner_path.clone(),
                        set_id_hint: None,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            let job_path = job_entry.path();
            let job_id = job_entry
                .file_name()
                .to_str()
                .and_then(decode_component)
                .filter(|job| validate_identity("job", job).is_ok());
            let Some(job_id) = job_id else {
                report.unrecognized.push(UnrecognizedStoreEntry {
                    path: job_path,
                    set_id_hint: None,
                    reason: "invalid job directory".into(),
                });
                continue;
            };
            if !job_entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                report.unrecognized.push(UnrecognizedStoreEntry {
                    path: job_path,
                    set_id_hint: None,
                    reason: "job entry is not a directory".into(),
                });
                continue;
            }
            let bundles = match fs::read_dir(&job_path) {
                Ok(bundles) => bundles,
                Err(error) => {
                    report.unrecognized.push(UnrecognizedStoreEntry {
                        path: job_path,
                        set_id_hint: None,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            let mut bundle_entries = Vec::new();
            for bundle_entry in bundles {
                match bundle_entry {
                    Ok(entry) => bundle_entries.push(entry),
                    Err(error) => report.unrecognized.push(UnrecognizedStoreEntry {
                        path: job_path.clone(),
                        set_id_hint: None,
                        reason: error.to_string(),
                    }),
                }
            }
            if bundle_entries.len() > 1 {
                for bundle_entry in bundle_entries {
                    report.unrecognized.push(UnrecognizedStoreEntry {
                        path: bundle_entry.path(),
                        set_id_hint: set_id_hint(&bundle_entry.file_name()),
                        reason: "job directory contains multiple bundle entries".into(),
                    });
                }
                continue;
            }
            for bundle_entry in bundle_entries {
                let path = bundle_entry.path();
                let set_id = set_id_hint(&bundle_entry.file_name());
                let Some(set_id) = set_id else {
                    report.unrecognized.push(UnrecognizedStoreEntry {
                        path,
                        set_id_hint: None,
                        reason: "invalid bundle filename".into(),
                    });
                    continue;
                };
                let reference = MediaSetRef {
                    owner_id: owner_id.into(),
                    job_id: job_id.clone(),
                    set_id,
                };
                match self.decode_bundle_at_state(&reference, state) {
                    Ok(()) => match state {
                        StoredState::Active => report.active.push(reference),
                        StoredState::Retired => report.retired.push(reference),
                        StoredState::Staging => report.staging.push(reference),
                    },
                    Err(error) => report.unrecognized.push(UnrecognizedStoreEntry {
                        path,
                        set_id_hint: Some(reference.set_id),
                        reason: error.to_string(),
                    }),
                }
            }
        }
    }

    fn decode_bundle_at_state(
        &self,
        media_set: &MediaSetRef,
        state: StoredState,
    ) -> Result<(), QueueMediaError> {
        let expected = self.bundle_path(state, media_set);
        let other = self.bundle_path(
            match state {
                StoredState::Active => StoredState::Retired,
                StoredState::Retired => StoredState::Active,
                StoredState::Staging => StoredState::Active,
            },
            media_set,
        );
        if state != StoredState::Staging && symlink_metadata_optional(&other)?.is_some() {
            return Err(QueueMediaError::Corrupt(
                "set exists in both lifecycle states".into(),
            ));
        }
        if symlink_metadata_optional(&expected)?.is_none() {
            return Err(QueueMediaError::NotFound);
        }
        if state == StoredState::Staging {
            self.decode_bundle_from_path(media_set, &expected, None)
                .map(|_| ())
        } else {
            self.decode_bundle(media_set, None).map(|_| ())
        }
    }
}

#[derive(Debug)]
struct DecodedBundle {
    manifest: MediaSetManifest,
    memory: BTreeMap<u32, SensitiveBytes>,
}

#[derive(Debug)]
struct SensitiveBytes(Vec<u8>);

impl SensitiveBytes {
    fn into_vec(mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

impl Drop for SensitiveBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

struct ObservedFile {
    index: u32,
    next_chunk: u32,
    size_bytes: u64,
    digest: Sha256,
    output: Option<File>,
}

struct V2ObservedFile {
    index: u32,
    next_chunk: u32,
    size_bytes: u64,
    digest: Sha256,
    sink: QueueMediaSink,
    output: Option<File>,
    memory: Option<SensitiveBytes>,
}

fn begin_v2_observation(
    plaintext: &[u8],
    output_root: Option<&Path>,
    mixed: bool,
    current: &mut Option<V2ObservedFile>,
    observations: &mut Vec<DataObservation>,
    memory: &mut BTreeMap<u32, SensitiveBytes>,
) -> Result<(), QueueMediaError> {
    if plaintext.len() != 6 {
        return Err(QueueMediaError::Corrupt(
            "invalid media begin record".into(),
        ));
    }
    finalize_v2_observation(current, observations, memory)?;
    let index = u32::from_be_bytes(plaintext[1..5].try_into().expect("sized"));
    if observations
        .last()
        .is_some_and(|previous| index <= previous.index)
    {
        return Err(QueueMediaError::Corrupt(
            "media file ordering is not strictly increasing".into(),
        ));
    }
    let sink = match plaintext[5] {
        0 => QueueMediaSink::Memory,
        1 => QueueMediaSink::PrivateStaging,
        _ => return Err(QueueMediaError::Corrupt("invalid media sink".into())),
    };
    let write_to_disk = output_root.is_some() && (!mixed || sink == QueueMediaSink::PrivateStaging);
    let output = write_to_disk
        .then(|| {
            create_private_file(
                &output_root
                    .expect("write-to-disk requires root")
                    .join(format!("{index:08}.media")),
            )
        })
        .transpose()?;
    *current = Some(V2ObservedFile {
        index,
        next_chunk: 0,
        size_bytes: 0,
        digest: Sha256::new(),
        sink,
        output,
        memory: (mixed && sink == QueueMediaSink::Memory).then(|| SensitiveBytes(Vec::new())),
    });
    Ok(())
}

fn consume_v2_data_record(
    plaintext: &[u8],
    current: &mut Option<V2ObservedFile>,
) -> Result<(), QueueMediaError> {
    if plaintext.len() < DATA_HEADER_BYTES {
        return Err(QueueMediaError::Corrupt("short media record".into()));
    }
    let index = u32::from_be_bytes(plaintext[1..5].try_into().expect("sized"));
    let chunk = u32::from_be_bytes(plaintext[5..9].try_into().expect("sized"));
    let file = current
        .as_mut()
        .ok_or_else(|| QueueMediaError::Corrupt("media data precedes begin record".into()))?;
    if file.index != index || file.next_chunk != chunk {
        return Err(QueueMediaError::Corrupt(
            "media chunk ordering is not contiguous".into(),
        ));
    }
    let bytes = &plaintext[DATA_HEADER_BYTES..];
    file.digest.update(bytes);
    file.size_bytes = file
        .size_bytes
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| QueueMediaError::Corrupt("media size overflow".into()))?;
    file.next_chunk = file
        .next_chunk
        .checked_add(1)
        .ok_or_else(|| QueueMediaError::Corrupt("media chunk counter overflow".into()))?;
    if let Some(output) = &mut file.output {
        output.write_all(bytes)?;
    }
    if let Some(memory) = &mut file.memory {
        memory.0.extend_from_slice(bytes);
    }
    Ok(())
}

fn finalize_v2_observation(
    current: &mut Option<V2ObservedFile>,
    observations: &mut Vec<DataObservation>,
    memory: &mut BTreeMap<u32, SensitiveBytes>,
) -> Result<(), QueueMediaError> {
    let Some(mut file) = current.take() else {
        return Ok(());
    };
    if let Some(output) = &mut file.output {
        output.sync_all()?;
    }
    if let Some(bytes) = file.memory.take() {
        if memory.insert(file.index, bytes).is_some() {
            return Err(QueueMediaError::Corrupt(
                "duplicate in-memory media payload".into(),
            ));
        }
    }
    observations.push(DataObservation {
        index: file.index,
        size_bytes: file.size_bytes,
        sha256_hex: hex_encode(&file.digest.finalize()),
        chunk_count: file.next_chunk,
        sink: Some(file.sink),
    });
    Ok(())
}

fn consume_data_record(
    plaintext: &[u8],
    output_root: Option<&Path>,
    current: &mut Option<ObservedFile>,
    observations: &mut Vec<DataObservation>,
) -> Result<(), QueueMediaError> {
    if plaintext.len() < DATA_HEADER_BYTES {
        return Err(QueueMediaError::Corrupt("short media record".into()));
    }
    let index = u32::from_be_bytes(plaintext[1..5].try_into().expect("sized slice"));
    let chunk = u32::from_be_bytes(plaintext[5..9].try_into().expect("sized slice"));
    if current.as_ref().is_some_and(|file| file.index != index) {
        let previous = current.as_ref().expect("checked").index;
        if index <= previous {
            return Err(QueueMediaError::Corrupt(
                "media file ordering is not strictly increasing".into(),
            ));
        }
        finalize_observation(current, observations)?;
    }
    if current.is_none() {
        if chunk != 0 {
            return Err(QueueMediaError::Corrupt(
                "media file does not begin at chunk zero".into(),
            ));
        }
        let output = output_root
            .map(|root| create_private_file(&root.join(format!("{index:08}.media"))))
            .transpose()?;
        *current = Some(ObservedFile {
            index,
            next_chunk: 0,
            size_bytes: 0,
            digest: Sha256::new(),
            output,
        });
    }
    let file = current.as_mut().expect("initialized");
    if chunk != file.next_chunk {
        return Err(QueueMediaError::Corrupt(
            "media chunk ordering is not contiguous".into(),
        ));
    }
    let bytes = &plaintext[DATA_HEADER_BYTES..];
    file.digest.update(bytes);
    file.size_bytes = file
        .size_bytes
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| QueueMediaError::Corrupt("media size overflow".into()))?;
    file.next_chunk = file
        .next_chunk
        .checked_add(1)
        .ok_or_else(|| QueueMediaError::Corrupt("media chunk counter overflow".into()))?;
    if let Some(output) = &mut file.output {
        output.write_all(bytes)?;
    }
    Ok(())
}

fn finalize_observation(
    current: &mut Option<ObservedFile>,
    observations: &mut Vec<DataObservation>,
) -> Result<(), QueueMediaError> {
    let Some(mut file) = current.take() else {
        return Ok(());
    };
    if let Some(output) = &mut file.output {
        output.sync_all()?;
    }
    observations.push(DataObservation {
        index: file.index,
        size_bytes: file.size_bytes,
        sha256_hex: hex_encode(&file.digest.finalize()),
        chunk_count: file.next_chunk,
        sink: None,
    });
    Ok(())
}

fn validate_final_record(plaintext: &[u8], manifest_bytes: &[u8]) -> Result<(), QueueMediaError> {
    if plaintext.len() != 1 + 8 + 32 || plaintext[0] != b'F' {
        return Err(QueueMediaError::Corrupt(
            "invalid final authentication record".into(),
        ));
    }
    let length = u64::from_be_bytes(plaintext[1..9].try_into().expect("sized slice"));
    if length != manifest_bytes.len() as u64 {
        return Err(QueueMediaError::Corrupt(
            "manifest length does not match final record".into(),
        ));
    }
    let digest = Sha256::digest(manifest_bytes);
    if digest.as_slice() != &plaintext[9..] {
        return Err(QueueMediaError::Authentication);
    }
    Ok(())
}

fn validate_manifest(
    expected: &MediaSetRef,
    manifest: WireManifest,
    observations: &[DataObservation],
    output_root: Option<&Path>,
) -> Result<MediaSetManifest, QueueMediaError> {
    if !matches!(manifest.format_version, FORMAT_VERSION | V2_FORMAT_VERSION)
        || manifest.owner_id != expected.owner_id
        || manifest.job_id != expected.job_id
        || manifest.set_id != expected.set_id
    {
        return Err(QueueMediaError::Authentication);
    }
    let observed: BTreeMap<u32, &DataObservation> =
        observations.iter().map(|item| (item.index, item)).collect();
    if observed.len() != observations.len() {
        return Err(QueueMediaError::Corrupt(
            "duplicate media observations".into(),
        ));
    }
    let operation_fingerprint = manifest
        .operation_fingerprint
        .map(validate_operation_fingerprint)
        .transpose()?;
    let empty_digest = hex_encode(&Sha256::digest([]));
    let mut public_entries = Vec::with_capacity(manifest.entries.len());
    let format_version = manifest.format_version;
    for (expected_index, entry) in manifest.entries.into_iter().enumerate() {
        let expected_index = u32::try_from(expected_index)
            .map_err(|_| QueueMediaError::Corrupt("manifest has too many entries".into()))?;
        if entry.index != expected_index {
            return Err(QueueMediaError::Corrupt(
                "manifest entry ordering is invalid".into(),
            ));
        }
        validate_manifest_label("role", &entry.role)?;
        validate_manifest_label("name", &entry.name)?;
        let sink = match (format_version, entry.sink) {
            (FORMAT_VERSION, None) => QueueMediaSink::PrivateStaging,
            (V2_FORMAT_VERSION, Some(sink)) => sink.into(),
            _ => {
                return Err(QueueMediaError::Corrupt(
                    "manifest media sink does not match bundle version".into(),
                ))
            }
        };
        match observed.get(&entry.index) {
            Some(actual)
                if actual.size_bytes == entry.size_bytes
                    && actual.sha256_hex == entry.sha256_hex
                    && actual.chunk_count == entry.chunk_count
                    && actual.sink.is_none_or(|actual_sink| actual_sink == sink) => {}
            None if entry.size_bytes == 0
                && entry.chunk_count == 0
                && entry.sha256_hex == empty_digest =>
            {
                if let Some(root) = output_root {
                    let file =
                        create_private_file(&root.join(format!("{expected_index:08}.media")))?;
                    file.sync_all()?;
                }
            }
            _ => {
                return Err(QueueMediaError::Authentication);
            }
        }
        public_entries.push(MediaManifestEntry {
            role: entry.role,
            name: entry.name,
            size_bytes: entry.size_bytes,
            sha256_hex: entry.sha256_hex,
            sink,
        });
    }
    if observations
        .iter()
        .any(|item| item.index as usize >= public_entries.len())
    {
        return Err(QueueMediaError::Corrupt(
            "media record has no manifest entry".into(),
        ));
    }
    Ok(MediaSetManifest {
        media_set: expected.clone(),
        operation_fingerprint,
        entries: public_entries,
    })
}

fn validate_operation_fingerprint(
    fingerprint: WireOperationFingerprint,
) -> Result<QueueMediaOperationFingerprint, QueueMediaError> {
    if fingerprint.version != OPERATION_FINGERPRINT_VERSION_SHA256_V1
        || fingerprint.sha256_hex.len() != 64
        || !fingerprint
            .sha256_hex
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(QueueMediaError::Corrupt(
            "unsupported or malformed operation fingerprint".into(),
        ));
    }
    Ok(QueueMediaOperationFingerprint {
        version: fingerprint.version,
        sha256_hex: fingerprint.sha256_hex,
    })
}

fn write_encrypted_frame(
    writer: &mut impl Write,
    encryptor: &mut EncryptorBE32<Cipher>,
    media_set: &MediaSetRef,
    ordinal: &mut u32,
    plaintext: Zeroizing<Vec<u8>>,
) -> Result<(), QueueMediaError> {
    write_encrypted_frame_for_version(
        writer,
        encryptor,
        media_set,
        FORMAT_VERSION,
        ordinal,
        plaintext,
    )
}

fn write_encrypted_frame_for_version(
    writer: &mut impl Write,
    encryptor: &mut EncryptorBE32<Cipher>,
    media_set: &MediaSetRef,
    format_version: u16,
    ordinal: &mut u32,
    plaintext: Zeroizing<Vec<u8>>,
) -> Result<(), QueueMediaError> {
    #[cfg(test)]
    let frame_kind = seal_frame_kind(plaintext.as_slice());
    #[cfg(test)]
    injected_seal_error(SealTestFailure::Encrypt(frame_kind))?;
    let ciphertext_len = plaintext
        .len()
        .checked_add(AEAD_TAG_BYTES)
        .ok_or_else(|| QueueMediaError::Corrupt("frame length overflow".into()))?;
    let aad = frame_aad_for_version(media_set, format_version, *ordinal, false, ciphertext_len);
    let ciphertext = encryptor
        .encrypt_next(aead_stream::aead::Payload {
            msg: plaintext.as_slice(),
            aad: &aad,
        })
        .map_err(|_| QueueMediaError::Authentication)?;
    #[cfg(test)]
    injected_seal_error(SealTestFailure::Write(frame_kind))?;
    write_frame(writer, false, &ciphertext)?;
    *ordinal = ordinal
        .checked_add(1)
        .ok_or_else(|| QueueMediaError::Corrupt("stream counter overflow".into()))?;
    Ok(())
}

fn write_frame(
    writer: &mut impl Write,
    is_final: bool,
    ciphertext: &[u8],
) -> Result<(), QueueMediaError> {
    let length = u32::try_from(ciphertext.len())
        .map_err(|_| QueueMediaError::Corrupt("encrypted frame exceeds format".into()))?;
    writer.write_all(&[u8::from(is_final)])?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(ciphertext)?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<Option<(bool, Vec<u8>)>, QueueMediaError> {
    let mut flag = [0_u8; 1];
    match reader.read_exact(&mut flag) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    if flag[0] > 1 {
        return Err(QueueMediaError::Corrupt("invalid frame final flag".into()));
    }
    let mut length = [0_u8; 4];
    reader.read_exact(&mut length).map_err(map_truncation)?;
    let length = u32::from_be_bytes(length) as usize;
    if !(AEAD_TAG_BYTES..=MAX_CIPHERTEXT_FRAME).contains(&length) {
        return Err(QueueMediaError::Corrupt(
            "encrypted frame length is invalid".into(),
        ));
    }
    let mut ciphertext = vec![0_u8; length];
    reader.read_exact(&mut ciphertext).map_err(map_truncation)?;
    Ok(Some((flag[0] == 1, ciphertext)))
}

fn frame_aad(
    media_set: &MediaSetRef,
    ordinal: u32,
    is_final: bool,
    ciphertext_len: usize,
) -> Vec<u8> {
    frame_aad_for_version(media_set, FORMAT_VERSION, ordinal, is_final, ciphertext_len)
}

fn frame_aad_for_version(
    media_set: &MediaSetRef,
    format_version: u16,
    ordinal: u32,
    is_final: bool,
    ciphertext_len: usize,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(
        64 + media_set.owner_id.len() + media_set.job_id.len() + media_set.set_id.len(),
    );
    aad.extend_from_slice(b"mold.queue-media.stream");
    aad.extend_from_slice(&format_version.to_be_bytes());
    append_aad_field(&mut aad, media_set.owner_id.as_bytes());
    append_aad_field(&mut aad, media_set.job_id.as_bytes());
    append_aad_field(&mut aad, media_set.set_id.as_bytes());
    aad.extend_from_slice(&ordinal.to_be_bytes());
    aad.push(u8::from(is_final));
    aad.extend_from_slice(&(ciphertext_len as u64).to_be_bytes());
    aad
}

fn projection_aad(media_set: &MediaSetRef) -> Vec<u8> {
    let mut aad = Vec::with_capacity(
        64 + media_set.owner_id.len() + media_set.job_id.len() + media_set.set_id.len(),
    );
    aad.extend_from_slice(b"mold.queue-media.projection");
    aad.extend_from_slice(&V2_FORMAT_VERSION.to_be_bytes());
    append_aad_field(&mut aad, media_set.owner_id.as_bytes());
    append_aad_field(&mut aad, media_set.job_id.as_bytes());
    append_aad_field(&mut aad, media_set.set_id.as_bytes());
    aad.extend_from_slice(&0_u32.to_be_bytes());
    aad
}

fn operation_receipt_aad(owner_id: &str, operation_id: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(64 + owner_id.len() + operation_id.len());
    aad.extend_from_slice(b"mold.queue-media.operation-receipt");
    aad.extend_from_slice(&OPERATION_RECEIPT_VERSION.to_be_bytes());
    append_aad_field(&mut aad, owner_id.as_bytes());
    append_aad_field(&mut aad, operation_id.as_bytes());
    aad
}

fn admission_authority_aad(owner_id: &str, job_id: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(64 + owner_id.len() + job_id.len());
    aad.extend_from_slice(b"mold.queue-media.admission-authority.v1");
    aad.extend_from_slice(&(owner_id.len() as u64).to_be_bytes());
    aad.extend_from_slice(owner_id.as_bytes());
    aad.extend_from_slice(&(job_id.len() as u64).to_be_bytes());
    aad.extend_from_slice(job_id.as_bytes());
    aad
}

fn encode_projection(
    projection: &QueueMediaProjection,
) -> Result<[u8; PROJECTION_PLAINTEXT_BYTES], QueueMediaError> {
    let expected_dimension_slots = projection
        .edit_image_count()
        .min(PROJECTED_EDIT_DIMENSION_SLOTS);
    if projection.edit_images.len() != expected_dimension_slots {
        return Err(QueueMediaError::Corrupt(format!(
            "projection edit-image count {} requires {expected_dimension_slots} dimension slots, found {}",
            projection.edit_image_count,
            projection.edit_images.len(),
        )));
    }
    if projection.identity_present != (projection.identity_photograph_count > 0) {
        return Err(QueueMediaError::Corrupt(
            "projection identity presence disagrees with its photograph count".into(),
        ));
    }
    if projection.edit_images.iter().any(|dimensions| {
        matches!(
            dimensions,
            ProjectedImageDimensions::Known { width: 0, .. }
                | ProjectedImageDimensions::Known { height: 0, .. }
        )
    }) {
        return Err(QueueMediaError::Corrupt(
            "projection contains zero image dimensions".into(),
        ));
    }
    let mut bytes = [0_u8; PROJECTION_PLAINTEXT_BYTES];
    bytes[..2].copy_from_slice(&PROJECTION_VERSION.to_be_bytes());
    let mut flags = 0_u32;
    for (bit, present) in [
        projection.source_image,
        projection.source_video_inline,
        projection.source_video_path,
        projection.extend_video_inline,
        projection.extend_video_path,
        projection.identity_present,
        projection.mask_image,
        projection.control_image,
        projection.audio_inline,
        projection.audio_path,
    ]
    .into_iter()
    .enumerate()
    {
        if present {
            flags |= 1 << bit;
        }
    }
    bytes[4..8].copy_from_slice(&flags.to_be_bytes());
    bytes[8..12].copy_from_slice(&projection.keyframe_count.to_be_bytes());
    bytes[12..16].copy_from_slice(&projection.identity_photograph_count.to_be_bytes());
    bytes[16..20].copy_from_slice(&projection.edit_image_count.to_be_bytes());
    for (index, dimensions) in projection.edit_images.iter().enumerate() {
        let offset = 20 + index * 9;
        match dimensions {
            ProjectedImageDimensions::Known { width, height } => {
                bytes[offset] = 2;
                bytes[offset + 1..offset + 5].copy_from_slice(&width.to_be_bytes());
                bytes[offset + 5..offset + 9].copy_from_slice(&height.to_be_bytes());
            }
            ProjectedImageDimensions::UnreadableHeader => bytes[offset] = 1,
        }
    }
    Ok(bytes)
}

fn decode_projection(bytes: &[u8]) -> Result<QueueMediaProjection, QueueMediaError> {
    if bytes.len() != PROJECTION_PLAINTEXT_BYTES
        || u16::from_be_bytes(bytes[..2].try_into().expect("sized")) != PROJECTION_VERSION
        || bytes[2..4] != [0, 0]
        || bytes[PROJECTION_EDIT_SLOTS_END..]
            .iter()
            .any(|byte| *byte != 0)
    {
        return Err(QueueMediaError::Corrupt("invalid projection record".into()));
    }
    let flags = u32::from_be_bytes(bytes[4..8].try_into().expect("sized"));
    if flags & !0x03ff != 0 {
        return Err(QueueMediaError::Corrupt(
            "projection contains unknown flags".into(),
        ));
    }
    let edit_image_count = u32::from_be_bytes(bytes[16..20].try_into().expect("sized"));
    let dimension_count = (edit_image_count as usize).min(PROJECTED_EDIT_DIMENSION_SLOTS);
    let mut edit_images = Vec::with_capacity(dimension_count);
    for index in 0..PROJECTED_EDIT_DIMENSION_SLOTS {
        let offset = 20 + index * 9;
        let slot = &bytes[offset..offset + 9];
        if index >= dimension_count {
            if slot.iter().any(|byte| *byte != 0) {
                return Err(QueueMediaError::Corrupt(
                    "unused projection dimension slot is nonzero".into(),
                ));
            }
            continue;
        }
        edit_images.push(match slot[0] {
            1 if slot[1..].iter().all(|byte| *byte == 0) => {
                ProjectedImageDimensions::UnreadableHeader
            }
            2 => {
                let width = u32::from_be_bytes(slot[1..5].try_into().expect("sized"));
                let height = u32::from_be_bytes(slot[5..9].try_into().expect("sized"));
                if width == 0 || height == 0 {
                    return Err(QueueMediaError::Corrupt(
                        "projection contains zero image dimensions".into(),
                    ));
                }
                ProjectedImageDimensions::Known { width, height }
            }
            _ => {
                return Err(QueueMediaError::Corrupt(
                    "invalid projection dimension slot".into(),
                ))
            }
        });
    }
    let bit = |index| flags & (1_u32 << index) != 0_u32;
    let projection = QueueMediaProjection {
        source_image: bit(0),
        source_video_inline: bit(1),
        source_video_path: bit(2),
        extend_video_inline: bit(3),
        extend_video_path: bit(4),
        keyframe_count: u32::from_be_bytes(bytes[8..12].try_into().expect("sized")),
        identity_present: bit(5),
        identity_photograph_count: u32::from_be_bytes(bytes[12..16].try_into().expect("sized")),
        edit_image_count,
        edit_images,
        mask_image: bit(6),
        control_image: bit(7),
        audio_inline: bit(8),
        audio_path: bit(9),
    };
    if projection.identity_present != (projection.identity_photograph_count > 0) {
        return Err(QueueMediaError::Corrupt(
            "projection identity presence disagrees with its photograph count".into(),
        ));
    }
    Ok(projection)
}

fn append_aad_field(aad: &mut Vec<u8>, bytes: &[u8]) {
    aad.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    aad.extend_from_slice(bytes);
}

fn validate_media_set_ref(media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
    validate_identity("owner", &media_set.owner_id)?;
    validate_identity("job", &media_set.job_id)?;
    if !valid_set_id(&media_set.set_id) {
        return Err(QueueMediaError::InvalidIdentity(
            "set id is not 32 lowercase hexadecimal characters".into(),
        ));
    }
    Ok(())
}

fn sort_inspection(report: &mut StoreInspection) {
    report.active.sort();
    report.retired.sort();
    report.staging.sort();
    report.unrecognized.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.set_id_hint.cmp(&right.set_id_hint))
            .then_with(|| left.reason.cmp(&right.reason))
    });
}

fn validate_identity(kind: &str, value: &str) -> Result<(), QueueMediaError> {
    if value.is_empty() || value.contains('\0') {
        return Err(QueueMediaError::InvalidIdentity(format!(
            "{kind} id must be nonempty and contain no NUL"
        )));
    }
    Ok(())
}

fn validate_manifest_label(kind: &str, value: &str) -> Result<(), QueueMediaError> {
    if value.is_empty() || value.contains('\0') {
        return Err(QueueMediaError::Corrupt(format!(
            "manifest {kind} must be nonempty and contain no NUL"
        )));
    }
    Ok(())
}

fn valid_set_id(value: &str) -> bool {
    value.len() == 32
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn set_id_hint(file_name: &std::ffi::OsStr) -> Option<String> {
    file_name
        .to_str()
        .and_then(|name| name.strip_suffix(BUNDLE_SUFFIX))
        .filter(|set_id| valid_set_id(set_id))
        .map(ToOwned::to_owned)
}

fn encode_component(value: &str) -> String {
    hex_encode(value.as_bytes())
}

fn decode_component(value: &str) -> Option<String> {
    if !value.len().is_multiple_of(2) || !value.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().as_chunks::<2>().0 {
        let high = hex_value(pair[0])?;
        let low = hex_value(pair[1])?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).ok()
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn random_hex(bytes: usize) -> Result<String, QueueMediaError> {
    let mut random = Zeroizing::new(vec![0_u8; bytes]);
    random_fill(&mut random)?;
    Ok(hex_encode(&random))
}

fn random_fill(bytes: &mut [u8]) -> Result<(), QueueMediaError> {
    getrandom::fill(bytes).map_err(|error| QueueMediaError::SecurityUnavailable(error.to_string()))
}

fn map_truncation(error: std::io::Error) -> QueueMediaError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        QueueMediaError::Authentication
    } else {
        error.into()
    }
}

fn symlink_metadata_optional(path: &Path) -> Result<Option<fs::Metadata>, QueueMediaError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn ensure_existing_directory(path: &Path) -> Result<(), QueueMediaError> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(QueueMediaError::InsecurePath(path.display().to_string()));
    }
    Ok(())
}

fn ensure_private_dir(path: &Path) -> Result<(), QueueMediaError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => verify_private_directory_metadata(path, &metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path.parent().ok_or_else(|| {
                QueueMediaError::InsecurePath(format!("{} has no parent", path.display()))
            })?;
            if fs::symlink_metadata(parent).is_err() {
                ensure_private_dir(parent)?;
            }
            match create_directory_owner_only(path) {
                Ok(()) => crate::dir_sync::sync_directory(parent)?,
                Err(QueueMediaError::Io(error))
                    if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            let metadata = fs::symlink_metadata(path)?;
            verify_private_directory_metadata(path, &metadata)
        }
        Err(error) => Err(error.into()),
    }
}

/// Remove only a current-user private directory proven empty without
/// following links. `true` means the path is absent after the call; `false`
/// means it contains evidence and was deliberately retained.
fn remove_empty_private_directory(path: &Path) -> Result<bool, QueueMediaError> {
    let Some(metadata) = symlink_metadata_optional(path)? else {
        return Ok(true);
    };
    verify_private_directory_metadata(path, &metadata)?;
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Ok(false);
    }
    fs::remove_dir(path)?;
    if let Some(parent) = path.parent() {
        crate::dir_sync::sync_directory(parent)?;
    }
    Ok(true)
}

fn collect_uuid_job_directories(
    state_root: &Path,
    candidates: &mut std::collections::BTreeSet<(String, String)>,
) {
    let Ok(metadata) = fs::symlink_metadata(state_root) else {
        return;
    };
    if verify_private_directory_metadata(state_root, &metadata).is_err() {
        return;
    }
    let Ok(owners) = fs::read_dir(state_root) else {
        return;
    };
    for owner in owners.flatten() {
        let owner_path = owner.path();
        let Some(owner_id) = decoded_canonical_uuid_component(&owner.file_name()) else {
            continue;
        };
        let Ok(metadata) = fs::symlink_metadata(&owner_path) else {
            continue;
        };
        if verify_private_directory_metadata(&owner_path, &metadata).is_err() {
            continue;
        }
        let Ok(jobs) = fs::read_dir(&owner_path) else {
            continue;
        };
        for job in jobs.flatten() {
            let Some(job_id) = decoded_canonical_uuid_component(&job.file_name()) else {
                continue;
            };
            candidates.insert((owner_id.clone(), job_id));
        }
    }
}

fn collect_uuid_job_locks(
    locks_root: &Path,
    candidates: &mut std::collections::BTreeSet<(String, String)>,
) {
    let Ok(metadata) = fs::symlink_metadata(locks_root) else {
        return;
    };
    if verify_private_directory_metadata(locks_root, &metadata).is_err() {
        return;
    }
    let Ok(owners) = fs::read_dir(locks_root) else {
        return;
    };
    for owner in owners.flatten() {
        let owner_path = owner.path();
        let Some(owner_id) = decoded_canonical_uuid_component(&owner.file_name()) else {
            continue;
        };
        let Ok(metadata) = fs::symlink_metadata(&owner_path) else {
            continue;
        };
        if verify_private_directory_metadata(&owner_path, &metadata).is_err() {
            continue;
        }
        let Ok(locks) = fs::read_dir(&owner_path) else {
            continue;
        };
        for lock in locks.flatten() {
            let Some(name) = lock.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            let Some(encoded_job) = name.strip_suffix(".lock") else {
                continue;
            };
            let Some(job_id) = decode_component(encoded_job).filter(|value| {
                uuid::Uuid::parse_str(value).is_ok_and(|uuid| uuid.to_string() == *value)
            }) else {
                continue;
            };
            candidates.insert((owner_id.clone(), job_id));
        }
    }
}

fn decoded_canonical_uuid_component(value: &std::ffi::OsStr) -> Option<String> {
    value
        .to_str()
        .and_then(decode_component)
        .filter(|value| uuid::Uuid::parse_str(value).is_ok_and(|uuid| uuid.to_string() == *value))
}

fn verify_private_lock_file(path: &Path, file: &File) -> Result<(), QueueMediaError> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != 0 {
        return Err(QueueMediaError::InsecurePath(format!(
            "{} is not an empty private lock file",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if metadata.nlink() != 1 {
            return Err(QueueMediaError::InsecurePath(format!(
                "{} is not a singly-linked lock file",
                path.display()
            )));
        }
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(insecure_private_path(
                path,
                &metadata,
                0o600,
                "a current-user-owned 0600 lock file",
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn create_directory_owner_only(path: &Path) -> Result<(), QueueMediaError> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700).create(path)?;
    Ok(())
}

#[cfg(not(unix))]
fn create_directory_owner_only(path: &Path) -> Result<(), QueueMediaError> {
    fs::create_dir(path)?;
    Ok(())
}

/// POSIX single-quoting: everything inside `'...'` is literal, and an embedded
/// quote is closed, escaped, and reopened.
#[cfg(unix)]
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Describe a refused private path in terms an operator can act on.
///
/// The required state alone ("must be owned by the current user with mode
/// 0700") sends the reader to the source to work out what is actually wrong.
/// Anything that walks the mold data root — an ACL pass, `chmod -R`, a restore
/// that drops modes, `rsync` without `-p` — widens these paths, so the message
/// names the observed owner and mode beside the expected ones and prints the
/// exact repair for whichever half diverged.
#[cfg(unix)]
fn insecure_private_path(
    path: &Path,
    metadata: &fs::Metadata,
    expected_mode: u32,
    subject: &str,
) -> QueueMediaError {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let euid = unsafe { libc::geteuid() };
    let uid = metadata.uid();
    let mode = metadata.permissions().mode() & 0o7777;
    let mut observed = Vec::new();
    let mut repairs = Vec::new();
    // The path is interpolated into a command an operator is invited to paste,
    // and MOLD_HOME may legitimately contain a space or a shell metacharacter,
    // so it is quoted and the argument list is terminated. An unquoted
    // `/srv/Mold Data` is two operands and the "exact repair" would be wrong.
    let quoted = shell_quote(&path.display().to_string());
    if uid != euid {
        observed.push(format!("uid {uid} (expected {euid})"));
        repairs.push(format!("chown -- {euid} {quoted}"));
    }
    if mode & 0o077 != 0 {
        observed.push(format!("mode {mode:04o} (expected {expected_mode:04o})"));
        repairs.push(format!("chmod -- {expected_mode:04o} {quoted}"));
    }
    if observed.is_empty() {
        observed.push("an unexpected file type".to_string());
    }
    let mut message = format!(
        "{} must be {subject}: found {}",
        path.display(),
        observed.join(", ")
    );
    if !repairs.is_empty() {
        message.push_str(&format!("; repair with: {}", repairs.join(" && ")));
    }
    QueueMediaError::InsecurePath(message)
}

fn verify_private_directory_metadata(
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<(), QueueMediaError> {
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(QueueMediaError::InsecurePath(path.display().to_string()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(insecure_private_path(
                path,
                metadata,
                0o700,
                "a current-user-owned 0700 directory",
            ));
        }
    }
    Ok(())
}

fn store_contains_payload(version_root: &Path) -> Result<bool, QueueMediaError> {
    let Some(metadata) = symlink_metadata_optional(version_root)? else {
        return Ok(false);
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Ok(true);
    }
    for entry in fs::read_dir(version_root)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => return Ok(true),
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return Ok(true);
        };
        match name {
            "active" | "retired" | "staging" => {
                if tree_contains_non_directory_entry(&entry.path())? {
                    return Ok(true);
                }
            }
            "ephemeral" => {
                if runtime_staging_contains_payload(&entry.path())? {
                    return Ok(true);
                }
            }
            // Lock files carry no encrypted payload or obligation. A stale
            // lock cannot justify regenerating over media, which is covered by
            // the four globally scanned state trees above.
            "locks" if entry.file_type().is_ok_and(|kind| kind.is_dir()) => {}
            _ => return Ok(true),
        }
    }
    Ok(false)
}

fn runtime_staging_contains_payload(root: &Path) -> Result<bool, QueueMediaError> {
    let Some(metadata) = symlink_metadata_optional(root)? else {
        return Ok(false);
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Ok(true);
    }
    for entry in fs::read_dir(root)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => return Ok(true),
        };
        if entry.file_name() == std::ffi::OsStr::new(RUNTIME_STAGING_SWEEP)
            && entry.file_type().is_ok_and(|kind| kind.is_file())
        {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn tree_contains_non_directory_entry(root: &Path) -> Result<bool, QueueMediaError> {
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(_) => return Ok(true),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Ok(true);
    }
    for entry in fs::read_dir(root)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => return Ok(true),
        };
        if tree_contains_non_directory_entry(&entry.path())? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn initialize_master_key(
    key_path: &Path,
) -> Result<(Zeroizing<[u8; KEY_BYTES]>, KeyDisposition), QueueMediaError> {
    let mut key = Zeroizing::new([0_u8; KEY_BYTES]);
    random_fill(key.as_mut())?;
    let mut protected = Zeroizing::new(protect_master_key(&key)?);
    let temporary = key_path.with_extension(format!("tmp-{}", random_hex(8)?));
    let mut file = create_private_file(&temporary)?;
    file.write_all(&protected)?;
    file.sync_all()?;
    match fs::hard_link(&temporary, key_path) {
        Ok(()) => {
            crate::dir_sync::sync_directory(key_path.parent().expect("key has parent"))?;
            fs::remove_file(&temporary)?;
            protected.zeroize();
            Ok((key, KeyDisposition::Initialized))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(&temporary)?;
            key.zeroize();
            protected.zeroize();
            Ok((load_master_key(key_path)?, KeyDisposition::Loaded))
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error.into())
        }
    }
}

fn load_master_key(path: &Path) -> Result<Zeroizing<[u8; KEY_BYTES]>, QueueMediaError> {
    verify_private_key_path(path)?;
    let mut file = mold_core::secure_file::open_regular_file_no_follow(path)
        .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?;
    let mut protected = Zeroizing::new(Vec::new());
    file.read_to_end(&mut protected)?;
    unprotect_master_key(&protected)
}

#[cfg(unix)]
fn verify_private_key_path(path: &Path) -> Result<(), QueueMediaError> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(QueueMediaError::InsecurePath(format!(
            "{} must be a regular file",
            path.display()
        )));
    }
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err(insecure_private_path(
            path,
            &metadata,
            0o600,
            "a current-user-owned 0600 regular file",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_private_key_path(path: &Path) -> Result<(), QueueMediaError> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(QueueMediaError::InsecurePath(path.display().to_string()));
    }
    Ok(())
}

#[cfg(unix)]
fn protect_master_key(key: &[u8; KEY_BYTES]) -> Result<Vec<u8>, QueueMediaError> {
    Ok(key.to_vec())
}

#[cfg(unix)]
fn unprotect_master_key(bytes: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>, QueueMediaError> {
    let key: [u8; KEY_BYTES] = bytes
        .try_into()
        .map_err(|_| QueueMediaError::Corrupt("master key has an invalid length".into()))?;
    Ok(Zeroizing::new(key))
}

#[cfg(windows)]
fn protect_master_key(key: &[u8; KEY_BYTES]) -> Result<Vec<u8>, QueueMediaError> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: KEY_BYTES as u32,
        pbData: key.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let success = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if success == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let protected = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) };
    let mut bytes = b"MOLDQKDP1".to_vec();
    bytes.extend_from_slice(protected);
    unsafe { LocalFree(output.pbData.cast()) };
    Ok(bytes)
}

#[cfg(windows)]
fn unprotect_master_key(bytes: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>, QueueMediaError> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let payload = bytes
        .strip_prefix(b"MOLDQKDP1")
        .ok_or_else(|| QueueMediaError::Corrupt("master key has an invalid format".into()))?;
    let input = CRYPT_INTEGER_BLOB {
        cbData: u32::try_from(payload.len())
            .map_err(|_| QueueMediaError::Corrupt("master key is oversized".into()))?,
        pbData: payload.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let success = unsafe {
        CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if success == 0 {
        return Err(QueueMediaError::Corrupt(format!(
            "master key cannot be unprotected for this Windows user: {}",
            std::io::Error::last_os_error()
        )));
    }
    let unprotected = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) };
    let result = unprotected
        .try_into()
        .map(Zeroizing::new)
        .map_err(|_| QueueMediaError::Corrupt("unprotected master key has invalid length".into()));
    unsafe { LocalFree(output.pbData.cast()) };
    result
}

#[cfg(not(any(unix, windows)))]
fn protect_master_key(_key: &[u8; KEY_BYTES]) -> Result<Vec<u8>, QueueMediaError> {
    Err(QueueMediaError::SecurityUnavailable(
        "master-key protection is unavailable on this platform".into(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn unprotect_master_key(_bytes: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>, QueueMediaError> {
    Err(QueueMediaError::SecurityUnavailable(
        "master-key protection is unavailable on this platform".into(),
    ))
}

#[cfg(unix)]
fn create_private_file(path: &Path) -> Result<File, QueueMediaError> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(Into::into)
}

#[cfg(not(unix))]
fn create_private_file(path: &Path) -> Result<File, QueueMediaError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(Into::into)
}

#[cfg(unix)]
fn open_or_create_private_file(path: &Path) -> Result<File, QueueMediaError> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(QueueMediaError::InsecurePath(path.display().to_string()));
    }
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    if metadata.nlink() != 1 {
        return Err(QueueMediaError::InsecurePath(format!(
            "{} is not a singly-linked private file",
            path.display()
        )));
    }
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err(insecure_private_path(
            path,
            &metadata,
            0o600,
            "a current-user-owned 0600 private file",
        ));
    }
    Ok(file)
}

#[cfg(not(unix))]
fn open_or_create_private_file(path: &Path) -> Result<File, QueueMediaError> {
    let metadata = symlink_metadata_optional(path)?;
    if metadata.is_some_and(|item| item.file_type().is_symlink() || !item.is_file()) {
        return Err(QueueMediaError::InsecurePath(path.display().to_string()));
    }
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    #[test]
    fn admission_key_is_created_only_without_receipt_evidence() {
        let home = tempfile::tempdir().unwrap();
        let first = QueueMediaStore::generation_admission_key(home.path(), false).unwrap();
        let loaded = QueueMediaStore::generation_admission_key(home.path(), true).unwrap();
        assert_eq!(first.as_ref(), loaded.as_ref());

        std::fs::remove_file(
            home.path()
                .join(STORE_DIR)
                .join(GENERATION_ADMISSION_KEY_FILE),
        )
        .unwrap();
        assert!(matches!(
            QueueMediaStore::generation_admission_key(home.path(), true),
            Err(QueueMediaError::MissingAdmissionKeyWithReceipts)
        ));
        assert!(!home
            .path()
            .join(STORE_DIR)
            .join(GENERATION_ADMISSION_KEY_FILE)
            .exists());
    }

    #[test]
    fn corrupt_admission_key_is_never_regenerated() {
        let home = tempfile::tempdir().unwrap();
        QueueMediaStore::generation_admission_key(home.path(), false).unwrap();
        let key_path = home
            .path()
            .join(STORE_DIR)
            .join(GENERATION_ADMISSION_KEY_FILE);
        std::fs::write(&key_path, [9_u8; 7]).unwrap();
        assert!(QueueMediaStore::generation_admission_key(home.path(), true).is_err());
        assert_eq!(std::fs::read(key_path).unwrap(), [9_u8; 7]);
    }

    struct CountingReader<R> {
        inner: R,
        bytes_read: usize,
    }

    impl<R> CountingReader<R> {
        fn new(inner: R) -> Self {
            Self {
                inner,
                bytes_read: 0,
            }
        }
    }

    impl<R: Read> Read for CountingReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let read = self.inner.read(buffer)?;
            self.bytes_read += read;
            Ok(read)
        }
    }

    fn open_store(home: &Path) -> QueueMediaStore {
        QueueMediaStore::open(home).unwrap().store
    }

    fn media_bytes(size: usize) -> Vec<u8> {
        (0..size).map(|index| (index % 251) as u8).collect()
    }

    fn bundle_bytes(store: &QueueMediaStore, reference: &MediaSetRef) -> Vec<u8> {
        fs::read(store.bundle_path(StoredState::Active, reference)).unwrap()
    }

    fn frame_ranges(bytes: &[u8]) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut offset = MAGIC.len() + NONCE_PREFIX_BYTES;
        while offset < bytes.len() {
            let start = offset;
            assert!(offset + 5 <= bytes.len());
            let length =
                u32::from_be_bytes(bytes[offset + 1..offset + 5].try_into().unwrap()) as usize;
            offset += 5 + length;
            assert!(offset <= bytes.len());
            ranges.push(start..offset);
        }
        ranges
    }

    fn projection() -> QueueMediaProjection {
        QueueMediaProjection {
            source_image: true,
            source_video_inline: true,
            source_video_path: true,
            extend_video_inline: false,
            extend_video_path: true,
            keyframe_count: 2,
            identity_present: true,
            identity_photograph_count: 3,
            edit_image_count: 2,
            edit_images: vec![
                ProjectedImageDimensions::Known {
                    width: 640,
                    height: 480,
                },
                ProjectedImageDimensions::UnreadableHeader,
            ],
            mask_image: true,
            control_image: true,
            audio_inline: true,
            audio_path: true,
        }
    }

    fn seal_test_bundle(
        store: &QueueMediaStore,
        v2: bool,
        job_id: &str,
        media: Vec<SealMedia>,
    ) -> Result<MediaSetRef, QueueMediaError> {
        if v2 {
            store.seal_v2_with_operation_fingerprint(
                "owner",
                job_id,
                &QueueMediaOperationFingerprint::sha256_v1(b"seal test"),
                &QueueMediaProjection {
                    source_image: true,
                    ..Default::default()
                },
                media,
            )
        } else {
            store.seal("owner", job_id, media)
        }
    }

    fn job_directory(
        store: &QueueMediaStore,
        state: StoredState,
        owner: &str,
        job: &str,
    ) -> PathBuf {
        store
            .root
            .join(state.directory())
            .join(encode_component(owner))
            .join(encode_component(job))
    }

    fn job_lock_path(store: &QueueMediaStore, owner: &str, job: &str) -> PathBuf {
        store
            .root
            .join("locks")
            .join(encode_component(owner))
            .join(format!("{}.lock", encode_component(job)))
    }

    #[test]
    fn terminal_delete_reclaims_empty_job_directories_and_lock_idempotently() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let reference = store
            .seal(
                &owner,
                &job,
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .unwrap();
        let lock = job_lock_path(&store, &owner, &job);
        assert!(lock.is_file());

        store.delete(&reference).unwrap();
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            assert!(!job_directory(&store, state, &owner, &job).exists());
        }
        assert!(!lock.exists());

        store.cleanup_job_artifacts(&owner, &job);
        store.cleanup_job_artifacts(&owner, &job);
        assert!(!lock.exists());
    }

    #[test]
    fn rejected_seal_reclaims_empty_job_directories_and_lock() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let failure = inject_seal_test_failure(SealTestFailure::Write(SealFrameKind::Data));
        assert!(store
            .seal(
                &owner,
                &job,
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .is_err());
        drop(failure);

        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            assert!(!job_directory(&store, state, &owner, &job).exists());
        }
        assert!(!job_lock_path(&store, &owner, &job).exists());
    }

    #[test]
    fn startup_reclaims_stale_empty_uuid_job_directories_and_lock() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            ensure_private_dir(&job_directory(&store, state, &owner, &job)).unwrap();
        }
        let lock = job_lock_path(&store, &owner, &job);
        ensure_private_dir(lock.parent().unwrap()).unwrap();
        create_private_file(&lock).unwrap();
        drop(store);

        let reopened = open_store(home.path());
        for state in [
            StoredState::Active,
            StoredState::Retired,
            StoredState::Staging,
        ] {
            assert!(!job_directory(&reopened, state, &owner, &job).exists());
        }
        assert!(!lock.exists());
    }

    #[test]
    fn terminal_cleanup_retains_nonempty_suspicious_job_evidence() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let reference = store
            .seal(
                &owner,
                &job,
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .unwrap();
        let suspicious =
            job_directory(&store, StoredState::Staging, &owner, &job).join("foreign.evidence");
        fs::write(&suspicious, b"retain me").unwrap();

        store.delete(&reference).unwrap();
        assert_eq!(fs::read(&suspicious).unwrap(), b"retain me");
        assert!(job_lock_path(&store, &owner, &job).exists());
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_never_follows_or_unlinks_a_suspicious_lock_symlink() {
        use std::os::unix::fs::symlink;

        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let target = home.path().join("foreign-lock-target");
        fs::write(&target, b"foreign evidence").unwrap();
        let lock = job_lock_path(&store, &owner, &job);
        ensure_private_dir(lock.parent().unwrap()).unwrap();
        symlink(&target, &lock).unwrap();

        store.cleanup_job_artifacts(&owner, &job);
        assert!(fs::symlink_metadata(&lock)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(target).unwrap(), b"foreign evidence");
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_never_descends_through_a_symlinked_owner_root() {
        use std::os::unix::fs::symlink;

        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let owner = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let outside_owner = home.path().join("outside-owner");
        let outside_job = outside_owner.join(encode_component(&job));
        ensure_private_dir(&outside_job).unwrap();
        let owner_link = store
            .root
            .join(StoredState::Active.directory())
            .join(encode_component(&owner));
        symlink(&outside_owner, &owner_link).unwrap();

        store.cleanup_job_artifacts(&owner, &job);
        assert!(fs::symlink_metadata(&owner_link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(outside_job.is_dir());
    }

    #[test]
    fn consuming_seal_zeroizes_owned_inline_bytes_on_success_and_every_frame_failure() {
        for v2 in [false, true] {
            let home = tempfile::tempdir().unwrap();
            let store = open_store(home.path());
            let success_probe = Arc::new(AtomicBool::new(false));
            seal_test_bundle(
                &store,
                v2,
                "success",
                vec![SealMedia::bytes_with_zeroize_probe(
                    "source_image",
                    "scalar",
                    b"success sentinel".to_vec(),
                    Arc::clone(&success_probe),
                )],
            )
            .unwrap();
            assert!(success_probe.load(Ordering::SeqCst));

            let mut failures = vec![
                SealTestFailure::MediaRead,
                SealTestFailure::Encrypt(SealFrameKind::Data),
                SealTestFailure::Write(SealFrameKind::Data),
                SealTestFailure::Encrypt(SealFrameKind::Manifest),
                SealTestFailure::Write(SealFrameKind::Manifest),
                SealTestFailure::Encrypt(SealFrameKind::Final),
                SealTestFailure::Write(SealFrameKind::Final),
            ];
            if v2 {
                failures.extend([
                    SealTestFailure::Encrypt(SealFrameKind::Begin),
                    SealTestFailure::Write(SealFrameKind::Begin),
                ]);
            }
            for (index, failure) in failures.into_iter().enumerate() {
                let job_id = format!("failure-{v2}-{index}");
                let zeroized = Arc::new(AtomicBool::new(false));
                let injection = inject_seal_test_failure(failure);
                assert!(seal_test_bundle(
                    &store,
                    v2,
                    &job_id,
                    vec![SealMedia::bytes_with_zeroize_probe(
                        "source_image",
                        "scalar",
                        b"failure sentinel".to_vec(),
                        Arc::clone(&zeroized),
                    )],
                )
                .is_err());
                drop(injection);
                assert!(zeroized.load(Ordering::SeqCst), "{failure:?} in V2={v2}");
                assert!(!store
                    .job_has_bundle(StoredState::Active, "owner", &job_id)
                    .unwrap());
                assert!(!store
                    .job_has_bundle(StoredState::Staging, "owner", &job_id)
                    .unwrap());
                seal_test_bundle(
                    &store,
                    v2,
                    &job_id,
                    vec![SealMedia::bytes(
                        "source_image",
                        "scalar",
                        b"retry sentinel".to_vec(),
                    )],
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn publication_failures_roll_back_only_this_attempt_and_allow_retry() {
        for (index, failure) in [
            SealTestFailure::HardLink,
            SealTestFailure::DestinationSync,
            SealTestFailure::StagingUnlink,
            SealTestFailure::StagingSync,
        ]
        .into_iter()
        .enumerate()
        {
            let home = tempfile::tempdir().unwrap();
            let store = open_store(home.path());
            let job_id = format!("publication-{index}");
            let zeroized = Arc::new(AtomicBool::new(false));
            let injection = inject_seal_test_failure(failure);
            assert!(seal_test_bundle(
                &store,
                true,
                &job_id,
                vec![SealMedia::bytes_with_zeroize_probe(
                    "source_image",
                    "scalar",
                    b"publication sentinel".to_vec(),
                    Arc::clone(&zeroized),
                )],
            )
            .is_err());
            drop(injection);
            assert!(zeroized.load(Ordering::SeqCst), "{failure:?}");
            assert!(!store
                .job_has_bundle(StoredState::Active, "owner", &job_id)
                .unwrap());
            assert!(!store
                .job_has_bundle(StoredState::Staging, "owner", &job_id)
                .unwrap());
            seal_test_bundle(
                &store,
                true,
                &job_id,
                vec![SealMedia::bytes(
                    "source_image",
                    "scalar",
                    b"retry sentinel".to_vec(),
                )],
            )
            .unwrap();
        }
    }

    #[test]
    fn publication_collision_preserves_preexisting_destination() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let zeroized = Arc::new(AtomicBool::new(false));
        let injection = inject_seal_test_failure(SealTestFailure::HardLinkCollision);
        assert!(matches!(
            seal_test_bundle(
                &store,
                true,
                "collision",
                vec![SealMedia::bytes_with_zeroize_probe(
                    "source_image",
                    "scalar",
                    b"candidate bytes".to_vec(),
                    Arc::clone(&zeroized),
                )],
            ),
            Err(QueueMediaError::JobAlreadySealed { .. })
        ));
        drop(injection);
        assert!(zeroized.load(Ordering::SeqCst));

        let report = store.inspect_owner("owner");
        assert_eq!(report.unrecognized.len(), 1);
        let collision = &report.unrecognized[0].path;
        assert_eq!(fs::read(collision).unwrap(), TEST_COLLISION_BYTES);
        assert!(!store
            .job_has_bundle(StoredState::Staging, "owner", "collision")
            .unwrap());
        fs::remove_file(collision).unwrap();
        crate::dir_sync::sync_directory(collision.parent().unwrap()).unwrap();
        seal_test_bundle(
            &store,
            true,
            "collision",
            vec![SealMedia::bytes(
                "source_image",
                "scalar",
                b"retry bytes".to_vec(),
            )],
        )
        .unwrap();
    }

    #[test]
    fn projection_reader_consumes_only_the_exact_authenticated_prefix() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let expected = projection();
        let v2 = store
            .seal_v2_with_operation_fingerprint(
                "owner",
                "job-v2-prefix",
                &QueueMediaOperationFingerprint::sha256_v1(b"projection prefix"),
                &expected,
                vec![SealMedia::bytes(
                    "source_image",
                    "scalar",
                    media_bytes(CHUNK_BYTES + 17),
                )],
            )
            .unwrap();
        let v2_bundle = bundle_bytes(&store, &v2);
        assert!(v2_bundle.len() > PROJECTION_HEADER_BYTES);
        let mut v2_reader = CountingReader::new(std::io::Cursor::new(v2_bundle));
        assert_eq!(
            store
                .open_projection_from_reader(&v2, &mut v2_reader)
                .unwrap(),
            expected
        );
        assert_eq!(v2_reader.bytes_read, PROJECTION_HEADER_BYTES);

        let v1 = store
            .seal(
                "owner",
                "job-v1-prefix",
                vec![SealMedia::bytes(
                    "source_image",
                    "scalar",
                    media_bytes(CHUNK_BYTES + 17),
                )],
            )
            .unwrap();
        let v1_bundle = bundle_bytes(&store, &v1);
        let mut v1_reader = CountingReader::new(std::io::Cursor::new(v1_bundle));
        assert!(matches!(
            store.open_projection_from_reader(&v1, &mut v1_reader),
            Err(QueueMediaError::ProjectionUnavailable(
                QueueMediaProjectionFailure::LegacyV1
            ))
        ));
        assert_eq!(v1_reader.bytes_read, MAGIC.len());
    }

    #[test]
    fn operation_receipts_are_randomized_bound_and_constant_time_comparable() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(b"complete request bytes");
        let different = QueueMediaOperationFingerprint::sha256_v1(b"different request bytes");
        let first = store
            .seal_operation_receipt_v1("owner-a", "operation-a", &fingerprint)
            .unwrap();
        let second = store
            .seal_operation_receipt_v1("owner-a", "operation-a", &fingerprint)
            .unwrap();

        assert_ne!(first, second);
        assert!(!first.as_str().contains(fingerprint.sha256_hex()));
        let opened = store
            .open_operation_receipt_v1("owner-a", "operation-a", &first)
            .unwrap();
        assert!(opened.constant_time_eq(&fingerprint));
        assert!(!opened.constant_time_eq(&different));
        assert!(store
            .open_operation_receipt_v1("owner-b", "operation-a", &first)
            .is_err());
        assert!(store
            .open_operation_receipt_v1("owner-a", "operation-b", &first)
            .is_err());
        let other_home = tempfile::tempdir().unwrap();
        let other_store = open_store(other_home.path());
        assert!(other_store
            .open_operation_receipt_v1("owner-a", "operation-a", &first)
            .is_err());

        let mut tampered = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(first.as_str())
            .unwrap();
        *tampered.last_mut().unwrap() ^= 1;
        let tampered = QueueMediaOperationReceipt::parse(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(tampered),
        )
        .unwrap();
        assert!(store
            .open_operation_receipt_v1("owner-a", "operation-a", &tampered)
            .is_err());
    }

    #[test]
    fn admission_authority_is_encrypted_and_bound_to_owner_job_and_store() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let payload = br#"{"version":1,"authenticated_identity_sha256":"identity-sentinel"}"#;
        let first = store
            .seal_admission_authority_v1("owner-a", "job-a", payload)
            .unwrap();
        let second = store
            .seal_admission_authority_v1("owner-a", "job-a", payload)
            .unwrap();

        assert_ne!(first, second);
        assert!(!first.as_str().contains("identity-sentinel"));
        assert_eq!(
            store
                .open_admission_authority_v1("owner-a", "job-a", &first)
                .unwrap()
                .as_slice(),
            payload
        );
        assert!(store
            .open_admission_authority_v1("owner-b", "job-a", &first)
            .is_err());
        assert!(store
            .open_admission_authority_v1("owner-a", "job-b", &first)
            .is_err());
        let other_home = tempfile::tempdir().unwrap();
        assert!(open_store(other_home.path())
            .open_admission_authority_v1("owner-a", "job-a", &first)
            .is_err());

        let mut tampered = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(first.as_str())
            .unwrap();
        *tampered.last_mut().unwrap() ^= 1;
        let tampered = QueueMediaAdmissionAuthority::parse(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(tampered),
        )
        .unwrap();
        assert!(store
            .open_admission_authority_v1("owner-a", "job-a", &tampered)
            .is_err());
    }

    #[test]
    fn v2_projection_is_a_bounded_first_record_and_v1_refuses_projection() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(b"operation");
        let expected = projection();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner",
                "job-v2",
                &fingerprint,
                &expected,
                vec![SealMedia::bytes("source_image", "scalar", vec![9; 4096])],
            )
            .unwrap();
        assert_eq!(store.open_projection(&reference).unwrap(), expected);

        let path = store.bundle_path(StoredState::Active, &reference);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        let mut tail = [0_u8; 1];
        file.read_exact(&mut tail).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[tail[0] ^ 1]).unwrap();
        file.sync_all().unwrap();
        assert_eq!(store.open_projection(&reference).unwrap(), expected);
        assert!(store.load(&reference).is_err());

        let legacy = store
            .seal(
                "owner",
                "job-v1",
                vec![SealMedia::bytes("source", "scalar", vec![1])],
            )
            .unwrap();
        assert!(matches!(
            store.open_projection(&legacy),
            Err(QueueMediaError::ProjectionUnavailable(
                QueueMediaProjectionFailure::LegacyV1
            ))
        ));
    }

    #[test]
    fn projection_keeps_unbounded_edit_count_separate_from_flux_dimension_slots() {
        let dimensions =
            vec![ProjectedImageDimensions::UnreadableHeader; PROJECTED_EDIT_DIMENSION_SLOTS];
        let projection = QueueMediaProjection {
            edit_image_count: 5,
            edit_images: dimensions.clone(),
            ..Default::default()
        };
        let encoded = encode_projection(&projection).unwrap();
        assert_eq!(decode_projection(&encoded).unwrap(), projection);

        let noncanonical = QueueMediaProjection {
            edit_image_count: 5,
            edit_images: dimensions[..PROJECTED_EDIT_DIMENSION_SLOTS - 1].to_vec(),
            ..Default::default()
        };
        assert!(matches!(
            encode_projection(&noncanonical),
            Err(QueueMediaError::Corrupt(_))
        ));

        let mut unused_slot = encode_projection(&QueueMediaProjection {
            edit_image_count: 1,
            edit_images: vec![ProjectedImageDimensions::UnreadableHeader],
            ..Default::default()
        })
        .unwrap();
        unused_slot[20 + 9] = 1;
        assert!(matches!(
            decode_projection(&unused_slot),
            Err(QueueMediaError::Corrupt(_))
        ));
    }

    #[cfg(unix)]
    fn dead_runtime_root(parent: &Path, digit: char) -> PathBuf {
        let root = parent.join(format!(
            "{RUNTIME_STAGING_PREFIX}{}",
            digit.to_string().repeat(32)
        ));
        ensure_private_dir(&root).unwrap();
        create_private_file(&root.join(RUNTIME_STAGING_CLAIM)).unwrap();
        fs::write(root.join("plaintext.media"), b"sigkill-shaped plaintext").unwrap();
        root
    }

    #[cfg(unix)]
    #[test]
    fn runtime_staging_preserves_live_peers_and_cleans_on_final_clone_drop() {
        let home = tempfile::tempdir().unwrap();
        let first = open_store(home.path());
        let first_root = first.runtime_staging.root.clone();
        let first_clone = first.clone();
        let second = open_store(home.path());
        let second_root = second.runtime_staging.root.clone();

        assert!(first_root.is_dir());
        assert!(second_root.is_dir());
        drop(first);
        assert!(first_root.is_dir(), "a live clone retains the claim");
        drop(second);
        assert!(!second_root.exists());
        assert!(first_root.is_dir());
        drop(first_clone);
        assert!(!first_root.exists());
    }

    #[cfg(unix)]
    #[test]
    fn mixed_decryption_retains_runtime_root_until_returned_set_drops() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join("source-video.mp4");
        fs::write(&source, b"lease-owned-video").unwrap();
        let store = open_store(home.path());
        let runtime_root = store.runtime_staging.root.clone();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner",
                "job-mixed-lifetime",
                &QueueMediaOperationFingerprint::sha256_v1(b"mixed lifetime"),
                &QueueMediaProjection {
                    source_video_path: true,
                    ..Default::default()
                },
                vec![SealMedia::path("source_video_path", "scalar", &source).unwrap()],
            )
            .unwrap();
        let decrypted = store.decrypt_mixed(&reference).unwrap();
        let staged = match &decrypted.media[0].payload {
            DecryptedQueueMediaPayload::PrivatePath(path) => path.clone(),
            _ => panic!("path-shaped media must use private staging"),
        };

        drop(store);
        assert!(runtime_root.is_dir());
        assert_eq!(fs::read(&staged).unwrap(), b"lease-owned-video");
        drop(decrypted);
        assert!(!staged.exists());
        assert!(!runtime_root.exists());
    }

    #[cfg(unix)]
    #[test]
    fn secret_bearing_debug_output_exposes_only_structural_metadata() {
        let home = tempfile::tempdir().unwrap();
        let private_path = home.path().join("private-path-debug-sentinel.bin");
        fs::write(&private_path, b"private-path-payload-sentinel").unwrap();
        let store = open_store(home.path());
        let fingerprint =
            QueueMediaOperationFingerprint::sha256_v1(b"operation-fingerprint-sentinel");
        let fingerprint_digest = fingerprint.sha256_hex().to_string();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner-debug-sentinel",
                "job-debug-sentinel",
                &fingerprint,
                &QueueMediaProjection {
                    identity_present: true,
                    identity_photograph_count: 1,
                    ..Default::default()
                },
                vec![
                    SealMedia::bytes(
                        "role-debug-sentinel",
                        "name-debug-sentinel",
                        vec![222, 173, 190, 239],
                    ),
                    SealMedia::path(
                        "path-role-debug-sentinel",
                        "path-name-debug-sentinel",
                        &private_path,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
        let mixed = store.decrypt_mixed(&reference).unwrap();
        let entry_digest = mixed.manifest.entries[0].sha256_hex.clone();

        let receipt_bytes =
            vec![0x7d; PROJECTION_NONCE_BYTES + OPERATION_RECEIPT_PLAINTEXT_BYTES + AEAD_TAG_BYTES];
        let receipt_encoded =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(receipt_bytes);
        let receipt = QueueMediaOperationReceipt::parse(receipt_encoded.clone()).unwrap();

        let legacy_reference = store
            .seal(
                "legacy-owner-debug-sentinel",
                "legacy-job-debug-sentinel",
                vec![SealMedia::path(
                    "legacy-role-debug-sentinel",
                    "legacy-name-debug-sentinel",
                    &private_path,
                )
                .unwrap()],
            )
            .unwrap();
        let legacy = store.decrypt_to_private_staging(&legacy_reference).unwrap();

        let outputs = [
            format!("{fingerprint:?}"),
            format!("{:?}", mixed.manifest.entries[0]),
            format!("{:?}", mixed.manifest),
            format!("{:?}", mixed.media[0].payload),
            format!("{:?}", mixed.media[1].payload),
            format!("{:?}", mixed.media[0]),
            format!("{:?}", mixed.media[1]),
            format!("{mixed:?}"),
            format!("{:?}", legacy.files[0]),
            format!("{legacy:?}"),
            format!("{receipt:?}"),
        ];
        let debug = outputs.join("\n");

        for secret in [
            "owner-debug-sentinel",
            "job-debug-sentinel",
            reference.set_id.as_str(),
            "role-debug-sentinel",
            "name-debug-sentinel",
            "path-role-debug-sentinel",
            "path-name-debug-sentinel",
            "private-path-debug-sentinel",
            "legacy-owner-debug-sentinel",
            "legacy-job-debug-sentinel",
            legacy_reference.set_id.as_str(),
            "legacy-role-debug-sentinel",
            "legacy-name-debug-sentinel",
            fingerprint_digest.as_str(),
            entry_digest.as_str(),
            receipt_encoded.as_str(),
            "222, 173, 190, 239",
        ] {
            assert!(!debug.contains(secret), "debug leaked {secret}: {debug}");
        }
        assert!(debug.contains("version: 1"));
        assert!(debug.contains("entry_count: 2"));
        assert!(debug.contains("media_count: 2"));
        assert!(debug.contains("file_count: 1"));
        assert!(debug.contains("Bytes { len: 4"));
        assert!(debug.contains("PrivatePath(<redacted>)"));
        assert!(debug.contains("QueueMediaOperationReceipt(<redacted>)"));
    }

    #[cfg(unix)]
    #[test]
    fn legacy_decryption_retains_runtime_root_until_returned_set_drops() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join("legacy-source.bin");
        fs::write(&source, b"legacy-lease-owned-bytes").unwrap();
        let store = open_store(home.path());
        let runtime_root = store.runtime_staging.root.clone();
        let reference = store
            .seal(
                "owner",
                "job-legacy-lifetime",
                vec![SealMedia::path("source", "scalar", &source).unwrap()],
            )
            .unwrap();
        let decrypted = store.decrypt_to_private_staging(&reference).unwrap();
        let staged = decrypted.files[0].path.clone();

        drop(store);
        assert!(runtime_root.is_dir());
        assert_eq!(fs::read(&staged).unwrap(), b"legacy-lease-owned-bytes");
        drop(decrypted);
        assert!(!staged.exists());
        assert!(!runtime_root.exists());
    }

    #[cfg(unix)]
    #[test]
    fn startup_sweeps_only_proven_dead_runtime_roots() {
        use std::os::unix::fs::symlink;

        let home = tempfile::tempdir().unwrap();
        let initial = open_store(home.path());
        let parent = initial.root.join("ephemeral");
        drop(initial);

        let dead = dead_runtime_root(&parent, 'a');
        let untracked = parent.join(format!("{RUNTIME_STAGING_PREFIX}{}", "b".repeat(32)));
        ensure_private_dir(&untracked).unwrap();
        fs::write(untracked.join("plaintext.media"), b"untracked").unwrap();
        let outside = tempfile::tempdir().unwrap();
        let symlink_root = parent.join(format!("{RUNTIME_STAGING_PREFIX}{}", "c".repeat(32)));
        symlink(outside.path(), &symlink_root).unwrap();

        let reopened = open_store(home.path());
        assert!(!dead.exists(), "an unlocked tracked root is proven dead");
        assert!(
            untracked.is_dir(),
            "an untracked root is never guessed dead"
        );
        assert!(symlink_root
            .symlink_metadata()
            .unwrap()
            .file_type()
            .is_symlink());
        drop(reopened);
    }

    #[cfg(unix)]
    #[test]
    fn control_locks_and_dead_plaintext_do_not_block_safe_empty_store_rekey() {
        let home = tempfile::tempdir().unwrap();
        let initial = QueueMediaStore::open(home.path()).unwrap();
        assert_eq!(initial.key_disposition, KeyDisposition::Initialized);
        let parent = initial.store.root.join("ephemeral");
        drop(initial);
        assert!(parent.join(RUNTIME_STAGING_SWEEP).is_file());

        let dead = dead_runtime_root(&parent, 'd');
        fs::remove_file(home.path().join(STORE_DIR).join(KEY_FILE)).unwrap();
        let reopened = QueueMediaStore::open(home.path()).unwrap();
        assert_eq!(reopened.key_disposition, KeyDisposition::Initialized);
        assert!(!dead.exists());
    }

    #[cfg(unix)]
    #[test]
    fn v2_mixed_decoder_keeps_inline_bytes_in_memory_and_drops_private_paths() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let source = home.path().join("source-video.mp4");
        fs::write(&source, b"path-shaped-private-video").unwrap();
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(b"operation");
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner",
                "job-mixed",
                &fingerprint,
                &projection(),
                vec![
                    SealMedia::bytes("identity_image", "scalar", b"face-bytes".to_vec()),
                    SealMedia::path("source_video_path", "scalar", &source).unwrap(),
                ],
            )
            .unwrap();
        let hydrated = store.decrypt_mixed(&reference).unwrap();
        assert!(matches!(
            store.decrypt_to_private_staging(&reference),
            Err(QueueMediaError::MixedSinkHydrationRequired)
        ));
        assert!(matches!(
            &hydrated.media[0].payload,
            DecryptedQueueMediaPayload::Bytes(bytes) if bytes == b"face-bytes"
        ));
        let staged = match &hydrated.media[1].payload {
            DecryptedQueueMediaPayload::PrivatePath(path) => path.clone(),
            _ => panic!("path-shaped media was not staged"),
        };
        assert_eq!(fs::read(&staged).unwrap(), b"path-shaped-private-video");
        let root = staged.parent().unwrap().to_path_buf();
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        drop(hydrated);
        assert!(!root.exists());
    }

    #[test]
    fn initializes_once_and_reopens_the_same_key() {
        let home = tempfile::tempdir().unwrap();
        let opened = QueueMediaStore::open(home.path()).unwrap();
        assert_eq!(opened.key_disposition, KeyDisposition::Initialized);
        let first_key = **opened.store.key;
        drop(opened);
        let reopened = QueueMediaStore::open(home.path()).unwrap();
        assert_eq!(reopened.key_disposition, KeyDisposition::Loaded);
        assert_eq!(**reopened.store.key, first_key);
        #[cfg(unix)]
        assert_eq!(
            QueueMediaStore::security_mode().unwrap(),
            QueueMediaSecurityMode::UnixOwnerOnly
        );
        #[cfg(windows)]
        {
            assert!(matches!(
                QueueMediaStore::security_mode(),
                Err(QueueMediaError::SecurityUnavailable(_))
            ));
            assert!(!QueueMediaStore::supports_mixed_hydration());
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let key = home.path().join(STORE_DIR).join(KEY_FILE);
            assert_eq!(
                fs::metadata(key).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(home.path().join(STORE_DIR))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn open_existing_does_not_create_a_missing_key() {
        let home = tempfile::tempdir().unwrap();
        assert!(matches!(
            QueueMediaStore::open_existing(home.path()),
            Err(QueueMediaError::MissingKey)
        ));
        assert!(!home.path().join(STORE_DIR).join(KEY_FILE).exists());
    }

    #[cfg(unix)]
    #[test]
    fn round_trip_streams_large_paths_and_keeps_the_manifest_encrypted() {
        let home = tempfile::tempdir().unwrap();
        let input = home.path().join("large-source.bin");
        let expected = media_bytes(CHUNK_BYTES * 5 + 117);
        fs::write(&input, &expected).unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "queue-owner",
                "job-large",
                vec![
                    SealMedia::path("first_frame", "secret-visible-name.png", &input).unwrap(),
                    SealMedia::bytes("mask", "empty-mask.bin", Vec::new()),
                ],
            )
            .unwrap();

        let encrypted = bundle_bytes(&store, &reference);
        assert!(!encrypted
            .windows("secret-visible-name.png".len())
            .any(|window| window == b"secret-visible-name.png"));
        assert!(!encrypted
            .windows("first_frame".len())
            .any(|window| window == b"first_frame"));
        let plaintext_digest = hex_encode(&Sha256::digest(&expected));
        assert!(!encrypted
            .windows(plaintext_digest.len())
            .any(|window| window == plaintext_digest.as_bytes()));
        let manifest = store.load(&reference).unwrap();
        assert_eq!(manifest.entries.len(), 2);
        assert_eq!(manifest.entries[0].size_bytes, expected.len() as u64);
        assert_eq!(manifest.entries[1].size_bytes, 0);

        let decrypted = store.decrypt_to_private_staging(&reference).unwrap();
        assert_eq!(fs::read(&decrypted.files[0].path).unwrap(), expected);
        assert_eq!(
            fs::read(&decrypted.files[1].path).unwrap(),
            Vec::<u8>::new()
        );
        let staging_root = decrypted.files[0].path.parent().unwrap().to_path_buf();
        drop(decrypted);
        assert!(!staging_root.exists());
    }

    #[test]
    fn operation_fingerprint_is_versioned_authenticated_and_encrypted_at_rest() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let canonical_operation = b"operation-v1:secret-source-digest-and-role";
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(canonical_operation);
        let reference = store
            .seal_with_operation_fingerprint(
                "owner",
                "ambiguous-singleton",
                &fingerprint,
                vec![SealMedia::bytes("source", "secret-name", vec![1, 2, 3])],
            )
            .unwrap();

        let encrypted = bundle_bytes(&store, &reference);
        assert!(!encrypted
            .windows(canonical_operation.len())
            .any(|window| window == canonical_operation));
        assert!(!encrypted
            .windows(fingerprint.sha256_hex().len())
            .any(|window| window == fingerprint.sha256_hex().as_bytes()));

        let loaded = store.load(&reference).unwrap();
        assert_eq!(loaded.operation_fingerprint, Some(fingerprint.clone()));
        assert_eq!(
            store.open_operation_fingerprint(&reference).unwrap(),
            Some(fingerprint.clone())
        );
        assert_eq!(fingerprint.version(), 1);

        let without_fingerprint = store
            .seal(
                "owner",
                "ordinary-job",
                vec![SealMedia::bytes("source", "one", vec![4])],
            )
            .unwrap();
        assert_eq!(
            store
                .open_operation_fingerprint(&without_fingerprint)
                .unwrap(),
            None
        );
    }

    #[test]
    fn identical_media_for_distinct_jobs_is_never_deduplicated() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let media = || vec![SealMedia::bytes("source", "same.bin", b"same".to_vec())];
        let first = store.seal("owner", "job-1", media()).unwrap();
        let second = store.seal("owner", "job-2", media()).unwrap();
        assert_ne!(first.set_id, second.set_id);
        assert_ne!(bundle_bytes(&store, &first), bundle_bytes(&store, &second));
    }

    #[test]
    fn a_job_can_have_only_one_bundle_even_after_retirement() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let media = || vec![SealMedia::bytes("source", "one", vec![1])];
        let reference = store.seal("owner", "job", media()).unwrap();
        assert!(matches!(
            store.seal("owner", "job", media()),
            Err(QueueMediaError::JobAlreadySealed { .. })
        ));
        store.retire(&reference).unwrap();
        assert!(matches!(
            store.seal("owner", "job", media()),
            Err(QueueMediaError::JobAlreadySealed { .. })
        ));
    }

    #[test]
    fn wrong_owner_job_and_set_bindings_fail_authentication() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .unwrap();
        let source = store.bundle_path(StoredState::Active, &reference);
        for (owner, job, set) in [
            ("other", "job", reference.set_id.clone()),
            ("owner", "other", reference.set_id.clone()),
            ("owner", "job", "0".repeat(32)),
        ] {
            let wrong = MediaSetRef {
                owner_id: owner.into(),
                job_id: job.into(),
                set_id: set,
            };
            let destination = store.bundle_path(StoredState::Active, &wrong);
            ensure_private_dir(destination.parent().unwrap()).unwrap();
            fs::copy(&source, &destination).unwrap();
            assert!(matches!(
                store.load(&wrong),
                Err(QueueMediaError::Authentication)
            ));
            fs::remove_file(destination).unwrap();
        }
    }

    #[test]
    fn a_bundle_cannot_be_opened_with_another_store_key() {
        let first_home = tempfile::tempdir().unwrap();
        let first_store = open_store(first_home.path());
        let reference = first_store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .unwrap();
        let second_home = tempfile::tempdir().unwrap();
        let second_store = open_store(second_home.path());
        let destination = second_store.bundle_path(StoredState::Active, &reference);
        ensure_private_dir(destination.parent().unwrap()).unwrap();
        fs::copy(
            first_store.bundle_path(StoredState::Active, &reference),
            destination,
        )
        .unwrap();
        assert!(matches!(
            second_store.load(&reference),
            Err(QueueMediaError::Authentication)
        ));
    }

    #[test]
    fn tampering_and_truncation_never_release_plaintext() {
        for mutation in ["tamper", "truncate"] {
            let home = tempfile::tempdir().unwrap();
            let store = open_store(home.path());
            let reference = store
                .seal(
                    "owner",
                    "job",
                    vec![SealMedia::bytes(
                        "source",
                        "one",
                        media_bytes(CHUNK_BYTES + 8),
                    )],
                )
                .unwrap();
            let path = store.bundle_path(StoredState::Active, &reference);
            let mut bytes = fs::read(&path).unwrap();
            if mutation == "tamper" {
                *bytes.last_mut().unwrap() ^= 0x80;
            } else {
                bytes.truncate(bytes.len() - 7);
            }
            fs::write(&path, bytes).unwrap();
            assert!(store.load(&reference).is_err());
            assert!(store.decrypt_to_private_staging(&reference).is_err());
            let runtime_entries = fs::read_dir(&store.runtime_staging.root)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            assert_eq!(
                runtime_entries,
                vec![std::ffi::OsString::from(RUNTIME_STAGING_CLAIM)]
            );
        }
    }

    #[test]
    fn reordered_authenticated_records_are_rejected() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes(
                    "source",
                    "large",
                    media_bytes(CHUNK_BYTES * 2 + 5),
                )],
            )
            .unwrap();
        let path = store.bundle_path(StoredState::Active, &reference);
        let mut bytes = fs::read(&path).unwrap();
        let ranges = frame_ranges(&bytes);
        assert!(ranges.len() >= 4);
        assert_eq!(ranges[0].len(), ranges[1].len());
        let first = bytes[ranges[0].clone()].to_vec();
        let second = bytes[ranges[1].clone()].to_vec();
        bytes[ranges[0].clone()].copy_from_slice(&second);
        bytes[ranges[1].clone()].copy_from_slice(&first);
        fs::write(path, bytes).unwrap();
        assert!(matches!(
            store.load(&reference),
            Err(QueueMediaError::Authentication)
        ));
    }

    #[test]
    fn missing_or_corrupt_key_with_existing_media_fails_closed() {
        for corrupt in [false, true] {
            let home = tempfile::tempdir().unwrap();
            let store = open_store(home.path());
            store
                .seal(
                    "owner",
                    "job",
                    vec![SealMedia::bytes("source", "one", vec![1])],
                )
                .unwrap();
            drop(store);
            let key_path = home.path().join(STORE_DIR).join(KEY_FILE);
            if corrupt {
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&key_path)
                    .unwrap();
                file.write_all(&[7_u8; KEY_BYTES - 1]).unwrap();
                file.sync_all().unwrap();
                assert!(matches!(
                    QueueMediaStore::open(home.path()),
                    Err(QueueMediaError::Corrupt(_))
                ));
            } else {
                fs::remove_file(&key_path).unwrap();
                assert!(matches!(
                    QueueMediaStore::open(home.path()),
                    Err(QueueMediaError::MissingKeyWithExistingStore)
                ));
                assert!(!key_path.exists());
            }
        }
    }

    #[test]
    fn key_initialization_requires_global_payload_emptiness() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "another-owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        drop(store);
        let key_path = home.path().join(STORE_DIR).join(KEY_FILE);
        fs::remove_file(&key_path).unwrap();
        assert!(matches!(
            QueueMediaStore::open(home.path()),
            Err(QueueMediaError::MissingKeyWithExistingStore)
        ));
        assert!(!key_path.exists());

        // Even a structurally unknown entry is evidence, not permission to
        // replace the key and make its contents permanently unreadable.
        fs::remove_file(
            home.path()
                .join(STORE_DIR)
                .join(STORE_VERSION_DIR)
                .join("active")
                .join(encode_component(&reference.owner_id))
                .join(encode_component(&reference.job_id))
                .join(format!("{}{BUNDLE_SUFFIX}", reference.set_id)),
        )
        .unwrap();
        fs::write(
            home.path()
                .join(STORE_DIR)
                .join(STORE_VERSION_DIR)
                .join("unknown-entry"),
            b"unknown",
        )
        .unwrap();
        assert!(matches!(
            QueueMediaStore::open(home.path()),
            Err(QueueMediaError::MissingKeyWithExistingStore)
        ));
    }

    #[test]
    fn media_free_directory_remnants_do_not_prevent_key_reinitialization() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        store.delete(&reference).unwrap();
        drop(store);
        fs::remove_file(home.path().join(STORE_DIR).join(KEY_FILE)).unwrap();
        let reopened = QueueMediaStore::open(home.path()).unwrap();
        assert_eq!(reopened.key_disposition, KeyDisposition::Initialized);
    }

    #[test]
    fn retirement_restore_and_delete_are_file_first_lifecycle_operations() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        let separately_deleted = store
            .seal(
                "owner",
                "delete-active",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        store.delete(&separately_deleted).unwrap();
        assert!(matches!(
            store.load(&separately_deleted),
            Err(QueueMediaError::NotFound)
        ));
        store.retire(&reference).unwrap();
        assert_eq!(store.load(&reference).unwrap().entries.len(), 1);
        store.restore(&reference).unwrap();
        store.retire(&reference).unwrap();
        store.delete(&reference).unwrap();
        assert!(matches!(
            store.load(&reference),
            Err(QueueMediaError::NotFound)
        ));
    }

    #[test]
    fn interrupted_staging_is_authenticated_inspectable_and_deletable() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = MediaSetRef {
            owner_id: "owner".into(),
            job_id: "interrupted-job".into(),
            set_id: random_hex(16).unwrap(),
        };
        let staging = store.bundle_path(StoredState::Staging, &reference);
        ensure_private_dir(staging.parent().unwrap()).unwrap();
        let file = create_private_file(&staging).unwrap();
        store
            .seal_file(
                &reference,
                None,
                None,
                &[SealMedia::bytes("source", "one", vec![1, 2, 3])],
                file,
            )
            .unwrap();
        let report = store.inspect_owner("owner");
        assert_eq!(report.staging, vec![reference.clone()]);
        assert!(report.unrecognized.is_empty());
        store.delete_staging(&reference).unwrap();
        assert!(!staging.exists());
    }

    #[test]
    fn unclaimed_owner_root_enumeration_never_descends_into_peer_roots() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        store
            .seal(
                "claimed-owner",
                "claimed-job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        store
            .seal(
                "peer-owner",
                "peer-job",
                vec![SealMedia::bytes("source", "one", vec![2])],
            )
            .unwrap();

        let roots = store.unclaimed_owner_roots("claimed-owner");
        assert!(roots.iter().any(|root| {
            root.owner_id_hint.as_deref() == Some("peer-owner")
                && root.description.contains("active")
        }));
        assert!(roots
            .iter()
            .all(|root| root.owner_id_hint.as_deref() != Some("claimed-owner")));
    }

    #[cfg(unix)]
    #[test]
    fn inspection_never_descends_into_a_symlinked_claimed_owner_root() {
        use std::os::unix::fs::symlink;

        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1, 2, 3])],
            )
            .unwrap();
        let bundle = store.bundle_path(StoredState::Active, &reference);
        let owner_root = bundle.parent().unwrap().parent().unwrap().to_path_buf();
        let target = home.path().join("owner-root-target");
        fs::rename(&owner_root, &target).unwrap();
        symlink(&target, &owner_root).unwrap();
        let target_bundle = target
            .join(encode_component(&reference.job_id))
            .join(format!("{}{BUNDLE_SUFFIX}", reference.set_id));
        let before = fs::read(&target_bundle).unwrap();

        let report = store.inspect_owner(&reference.owner_id);

        assert!(report.active.is_empty());
        assert!(report.retired.is_empty());
        assert!(report.staging.is_empty());
        assert_eq!(report.unrecognized.len(), 1);
        assert_eq!(report.unrecognized[0].path, owner_root);
        assert_eq!(
            report.unrecognized[0].reason,
            "owner root is not a direct directory"
        );
        assert!(fs::symlink_metadata(&owner_root)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(target_bundle).unwrap(), before);
    }

    #[test]
    fn inspection_retains_tampered_and_symlink_entries_as_unrecognized() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        let path = store.bundle_path(StoredState::Active, &reference);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[byte[0] ^ 1]).unwrap();
        file.sync_all().unwrap();
        let report = store.inspect_owner("owner");
        assert!(report.active.is_empty());
        assert_eq!(report.unrecognized.len(), 1);
        assert_eq!(
            report.unrecognized[0].set_id_hint.as_deref(),
            Some(reference.set_id.as_str())
        );
        assert!(path.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let second_set = "a".repeat(32);
            let link = path
                .parent()
                .unwrap()
                .join(format!("{second_set}{BUNDLE_SUFFIX}"));
            symlink(&path, &link).unwrap();
            let report = store.inspect_owner("owner");
            assert!(report
                .unrecognized
                .iter()
                .any(|entry| entry.set_id_hint.as_deref() == Some(second_set.as_str())));
            assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
        }
    }

    #[test]
    fn inspection_rejects_multiple_bundles_for_one_job_as_structurally_invalid() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        let second_set_id = "a".repeat(32);
        let second = MediaSetRef {
            set_id: second_set_id.clone(),
            ..reference.clone()
        };
        fs::copy(
            store.bundle_path(StoredState::Active, &reference),
            store.bundle_path(StoredState::Active, &second),
        )
        .unwrap();

        let report = store.inspect_owner("owner");
        assert!(report.active.is_empty());
        assert_eq!(report.unrecognized.len(), 2);
        assert!(report
            .unrecognized
            .iter()
            .any(|entry| entry.set_id_hint.as_deref() == Some(reference.set_id.as_str())));
        assert!(report
            .unrecognized
            .iter()
            .any(|entry| entry.set_id_hint.as_deref() == Some(second_set_id.as_str())));
    }

    /// A widened queue-media directory is the single most likely way this
    /// store turns itself off in the field: anything that walks the mold data
    /// root (an ACL pass, `chmod -R`, a restore that drops modes, `rsync`
    /// without `-p`) hits it. The refusal must therefore carry the observed
    /// state and the exact repair, not only the required state.
    #[cfg(unix)]
    #[test]
    fn a_widened_store_directory_names_the_observed_mode_and_its_repair() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        drop(open_store(home.path()));
        let store_dir = home.path().join(STORE_DIR);
        fs::set_permissions(&store_dir, fs::Permissions::from_mode(0o770)).unwrap();

        let Err(QueueMediaError::InsecurePath(message)) = QueueMediaStore::open(home.path()) else {
            panic!("a group-writable store directory must be refused");
        };
        assert!(
            message.contains(&store_dir.display().to_string()),
            "{message}"
        );
        assert!(message.contains("mode 0770"), "{message}");
        assert!(message.contains("expected 0700"), "{message}");
        assert!(
            message.contains(&format!("chmod -- 0700 '{}'", store_dir.display())),
            "{message}"
        );
        assert!(!message.contains("chown"), "{message}");
    }

    /// The repair is meant to be pasted, and `MOLD_HOME` may legitimately hold
    /// a space or a quote. An unquoted path turns one operand into two and the
    /// "exact repair" silently becomes the wrong command.
    #[cfg(unix)]
    #[test]
    fn a_repair_command_quotes_a_path_holding_shell_metacharacters() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        let awkward = home.path().join("Mold Data; rm -rf $HOME");
        fs::create_dir_all(&awkward).unwrap();
        drop(open_store(&awkward));
        let store_dir = awkward.join(STORE_DIR);
        fs::set_permissions(&store_dir, fs::Permissions::from_mode(0o770)).unwrap();

        let Err(QueueMediaError::InsecurePath(message)) = QueueMediaStore::open(&awkward) else {
            panic!("a group-writable store directory must be refused");
        };
        assert!(
            message.contains(&format!("chmod -- 0700 '{}'", store_dir.display())),
            "{message}"
        );
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
    }

    #[cfg(unix)]
    #[test]
    fn a_widened_master_key_names_the_observed_mode_and_its_repair() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        drop(open_store(home.path()));
        let key_path = home.path().join(STORE_DIR).join(KEY_FILE);
        fs::set_permissions(&key_path, fs::Permissions::from_mode(0o640)).unwrap();

        let Err(QueueMediaError::InsecurePath(message)) = QueueMediaStore::open(home.path()) else {
            panic!("a group-readable master key must be refused");
        };
        assert!(
            message.contains(&key_path.display().to_string()),
            "{message}"
        );
        assert!(message.contains("mode 0640"), "{message}");
        assert!(message.contains("expected 0600"), "{message}");
        assert!(
            message.contains(&format!("chmod -- 0600 '{}'", key_path.display())),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn insecure_key_mode_and_symlink_media_are_rejected() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        drop(store);
        let key_path = home.path().join(STORE_DIR).join(KEY_FILE);
        fs::set_permissions(&key_path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            QueueMediaStore::open(home.path()),
            Err(QueueMediaError::InsecurePath(_))
        ));
        fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)).unwrap();

        let source = home.path().join("source");
        let link = home.path().join("source-link");
        fs::write(&source, b"source").unwrap();
        symlink(&source, &link).unwrap();
        let store = open_store(home.path());
        assert!(matches!(
            SealMedia::path("source", "source", link).and_then(|media| store.seal(
                "owner",
                "job",
                vec![media]
            )),
            Err(QueueMediaError::InsecurePath(_))
        ));
    }
}
