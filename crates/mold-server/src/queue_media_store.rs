//! Encrypted, file-first storage for media attached to durable queue jobs.
//!
//! The store deliberately has no queue database knowledge. A caller supplies
//! the queue owner and job identity, and those values plus the random media-set
//! identity are bound into every STREAM record's associated data. The manifest
//! is encrypted along with the media and authenticated by a consuming final
//! record before any plaintext staging path is returned.

use aead_stream::{DecryptorBE32, EncryptorBE32, StreamBE32};
use chacha20poly1305::{KeyInit, XChaCha20Poly1305};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

const STORE_DIR: &str = "queue-media";
const STORE_VERSION_DIR: &str = "v1";
const KEY_FILE: &str = "master.key";
const MAGIC: &[u8; 8] = b"MOLDQMS1";
const FORMAT_VERSION: u16 = 1;
const NONCE_PREFIX_BYTES: usize = 19;
const KEY_BYTES: usize = 32;
const CHUNK_BYTES: usize = 1024 * 1024;
const AEAD_TAG_BYTES: usize = 16;
const DATA_HEADER_BYTES: usize = 9;
const MAX_CIPHERTEXT_FRAME: usize = CHUNK_BYTES + DATA_HEADER_BYTES + AEAD_TAG_BYTES;
const BUNDLE_SUFFIX: &str = ".qms";
const OPERATION_FINGERPRINT_VERSION_SHA256_V1: u16 = 1;

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
    #[error("queue-media set already exists for owner {owner_id} job {job_id}")]
    JobAlreadySealed { owner_id: String, job_id: String },
    #[error("queue-media set was not found")]
    NotFound,
    #[error("invalid queue-media identity: {0}")]
    InvalidIdentity(String),
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

#[derive(Debug, Clone)]
pub enum SealMediaSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SealMedia {
    pub role: String,
    pub name: String,
    pub source: SealMediaSource,
}

impl SealMedia {
    pub fn path(
        role: impl Into<String>,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::Path(path.into()),
        }
    }

    pub fn bytes(role: impl Into<String>, name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            source: SealMediaSource::Bytes(bytes),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaManifestEntry {
    pub role: String,
    pub name: String,
    pub size_bytes: u64,
    pub sha256_hex: String,
}

/// A fingerprint of caller-defined canonical operation bytes.
///
/// The store deliberately does not define or persist the canonical operation.
/// It only keeps this value inside the encrypted, authenticated manifest so a
/// caller can resolve an ambiguous seal without putting media-derived hashes in
/// plaintext queue state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueMediaOperationFingerprint {
    version: u16,
    sha256_hex: String,
}

impl QueueMediaOperationFingerprint {
    pub fn sha256_v1(canonical_operation: &[u8]) -> Self {
        Self {
            version: OPERATION_FINGERPRINT_VERSION_SHA256_V1,
            sha256_hex: hex_encode(&Sha256::digest(canonical_operation)),
        }
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSetManifest {
    pub media_set: MediaSetRef,
    pub operation_fingerprint: Option<QueueMediaOperationFingerprint>,
    pub entries: Vec<MediaManifestEntry>,
}

#[derive(Debug)]
pub struct DecryptedMedia {
    pub role: String,
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256_hex: String,
}

#[derive(Debug)]
pub struct DecryptedMediaSet {
    pub manifest: MediaSetManifest,
    pub files: Vec<DecryptedMedia>,
    root: Option<PathBuf>,
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
}

#[derive(Debug)]
struct DataObservation {
    index: u32,
    size_bytes: u64,
    sha256_hex: String,
    chunk_count: u32,
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

        let existing_payload = store_contains_payload(&version_root)?;
        if !key_existed && existing_payload {
            return Err(QueueMediaError::MissingKeyWithExistingStore);
        }
        if !key_existed && !allow_initialize {
            return Err(QueueMediaError::MissingKey);
        }

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

        let (key, key_disposition) = if key_existed {
            (load_master_key(&key_path)?, KeyDisposition::Loaded)
        } else {
            initialize_master_key(&key_path)?
        };
        Ok(OpenedQueueMediaStore {
            store: Self {
                root: version_root,
                key: Arc::new(key),
            },
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
            Ok(QueueMediaSecurityMode::WindowsDpapiCurrentUser)
        }
        #[cfg(not(any(unix, windows)))]
        {
            Err(QueueMediaError::SecurityUnavailable(
                "no owner-only key protection is implemented for this platform".into(),
            ))
        }
    }

    /// Seals exactly one fresh, non-content-addressed bundle for a queue job.
    pub fn seal(
        &self,
        owner_id: &str,
        job_id: &str,
        media: &[SealMedia],
    ) -> Result<MediaSetRef, QueueMediaError> {
        self.seal_inner(owner_id, job_id, None, media)
    }

    /// Seals a bundle whose encrypted manifest carries a versioned operation
    /// fingerprint for ambiguity-safe idempotency checks.
    pub fn seal_with_operation_fingerprint(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: &QueueMediaOperationFingerprint,
        media: &[SealMedia],
    ) -> Result<MediaSetRef, QueueMediaError> {
        self.seal_inner(owner_id, job_id, Some(operation_fingerprint), media)
    }

    fn seal_inner(
        &self,
        owner_id: &str,
        job_id: &str,
        operation_fingerprint: Option<&QueueMediaOperationFingerprint>,
        media: &[SealMedia],
    ) -> Result<MediaSetRef, QueueMediaError> {
        validate_identity("owner", owner_id)?;
        validate_identity("job", job_id)?;
        for item in media {
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
        let result = self.seal_file(
            &media_set,
            operation_fingerprint.cloned(),
            media,
            &staging_path,
        );
        if result.is_err() {
            let _ = fs::remove_file(&staging_path);
        }
        result?;

        let destination = self.bundle_path(StoredState::Active, &media_set);
        ensure_private_dir(destination.parent().expect("bundle has parent"))?;
        match fs::hard_link(&staging_path, &destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(&staging_path);
                return Err(QueueMediaError::JobAlreadySealed {
                    owner_id: owner_id.into(),
                    job_id: job_id.into(),
                });
            }
            Err(error) => return Err(error.into()),
        }
        crate::dir_sync::sync_directory(destination.parent().expect("bundle has parent"))?;
        fs::remove_file(&staging_path)?;
        crate::dir_sync::sync_directory(staging_path.parent().expect("bundle has parent"))?;
        drop(lock);
        Ok(media_set)
    }

    /// Authenticates every record and returns only the encrypted manifest.
    pub fn load(&self, media_set: &MediaSetRef) -> Result<MediaSetManifest, QueueMediaError> {
        self.decode_bundle(media_set, None)
            .map(|decoded| decoded.manifest)
    }

    /// Authenticates the complete bundle before returning its encrypted-at-rest
    /// operation fingerprint.
    pub fn open_operation_fingerprint(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<Option<QueueMediaOperationFingerprint>, QueueMediaError> {
        Ok(self.load(media_set)?.operation_fingerprint)
    }

    /// Authenticates the complete bundle before publishing a private plaintext
    /// staging directory to the caller.
    #[cfg(unix)]
    pub fn decrypt_to_private_staging(
        &self,
        media_set: &MediaSetRef,
    ) -> Result<DecryptedMediaSet, QueueMediaError> {
        let partial = self
            .root
            .join("ephemeral")
            .join(format!("{}.partial", random_hex(16)?));
        ensure_private_dir(&partial)?;
        let decoded = match self.decode_bundle(media_set, Some(&partial)) {
            Ok(decoded) => decoded,
            Err(error) => {
                let _ = fs::remove_dir_all(&partial);
                return Err(error);
            }
        };
        crate::dir_sync::sync_directory(&partial)?;
        let ready = partial.with_extension("ready");
        fs::rename(&partial, &ready)?;
        crate::dir_sync::sync_directory(&self.root.join("ephemeral"))?;
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
            root: Some(ready),
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
    }

    /// Deletes a fully authenticated interrupted publication from staging.
    pub fn delete_staging(&self, media_set: &MediaSetRef) -> Result<(), QueueMediaError> {
        validate_media_set_ref(media_set)?;
        let _lock = self.lock_job(&media_set.owner_id, &media_set.job_id)?;
        self.decode_bundle_at_state(media_set, StoredState::Staging)?;
        let staging = self.bundle_path(StoredState::Staging, media_set);
        fs::remove_file(&staging)?;
        crate::dir_sync::sync_directory(staging.parent().expect("bundle has parent"))?;
        Ok(())
    }

    /// Enumerates and authenticates one owner's sets. Malformed entries are
    /// reported and left untouched so startup/GC can make an explicit choice.
    pub fn inspect_owner(&self, owner_id: &str) -> StoreInspection {
        let mut report = StoreInspection::default();
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

    fn seal_file(
        &self,
        media_set: &MediaSetRef,
        operation_fingerprint: Option<QueueMediaOperationFingerprint>,
        media: &[SealMedia],
        staging_path: &Path,
    ) -> Result<(), QueueMediaError> {
        let file = create_private_file(staging_path)?;
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
                SealMediaSource::Path(path) => Box::new(
                    mold_core::secure_file::open_regular_file_no_follow(path)
                        .map_err(|error| QueueMediaError::InsecurePath(error.to_string()))?,
                ),
                SealMediaSource::Bytes(bytes) => Box::new(std::io::Cursor::new(bytes)),
            };
            let mut digest = Sha256::new();
            let mut size_bytes = 0_u64;
            let mut chunk_count = 0_u32;
            let mut buffer = vec![0_u8; CHUNK_BYTES];
            loop {
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
                size_bytes = size_bytes
                    .checked_add(read as u64)
                    .ok_or_else(|| QueueMediaError::Corrupt("media size overflow".into()))?;
                let mut plaintext = Vec::with_capacity(DATA_HEADER_BYTES + read);
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
            buffer.zeroize();
            manifest_entries.push(WireManifestEntry {
                index,
                role: item.role.clone(),
                name: item.name.clone(),
                size_bytes,
                sha256_hex: hex_encode(&digest.finalize()),
                chunk_count,
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
        let mut manifest_bytes = Zeroizing::new(serde_json::to_vec(&manifest)?);
        let manifest_digest = Sha256::digest(&*manifest_bytes);
        for chunk in manifest_bytes.chunks(CHUNK_BYTES) {
            let mut plaintext = Vec::with_capacity(1 + chunk.len());
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
        let mut final_plaintext = Vec::with_capacity(1 + 8 + 32);
        final_plaintext.push(b'F');
        final_plaintext.extend_from_slice(&(manifest_bytes.len() as u64).to_be_bytes());
        final_plaintext.extend_from_slice(&manifest_digest);
        let aad = frame_aad(
            media_set,
            ordinal,
            true,
            final_plaintext.len() + AEAD_TAG_BYTES,
        );
        let ciphertext = encryptor
            .encrypt_last(aead_stream::aead::Payload {
                msg: &final_plaintext,
                aad: &aad,
            })
            .map_err(|_| QueueMediaError::Authentication)?;
        write_frame(&mut writer, true, &ciphertext)?;
        final_plaintext.zeroize();
        manifest_bytes.zeroize();
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

        loop {
            let Some((is_final, ciphertext)) = read_frame(&mut reader)? else {
                break;
            };
            let aad = frame_aad(media_set, ordinal, is_final, ciphertext.len());
            let stream = decryptor.take().expect("stream exists until final frame");
            let mut plaintext = if is_final {
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
            .map_err(|_| QueueMediaError::Authentication)?;
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| QueueMediaError::Corrupt("stream counter overflow".into()))?;

            if is_final {
                finalize_observation(&mut current, &mut observations)?;
                validate_final_record(&plaintext, &manifest_bytes)?;
                plaintext.zeroize();
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
            plaintext.zeroize();
        }
        if !saw_final {
            return Err(QueueMediaError::Authentication);
        }
        let wire: WireManifest = serde_json::from_slice(&manifest_bytes)?;
        let manifest = validate_manifest(media_set, wire, &observations, output)?;
        manifest_bytes.zeroize();
        Ok(DecodedBundle { manifest })
    }

    fn lock_job(&self, owner_id: &str, job_id: &str) -> Result<File, QueueMediaError> {
        let owner_dir = self.root.join("locks").join(encode_component(owner_id));
        ensure_private_dir(&owner_dir)?;
        let lock_path = owner_dir.join(format!("{}.lock", encode_component(job_id)));
        let lock = open_or_create_private_file(&lock_path)?;
        lock.lock_exclusive()?;
        Ok(lock)
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
        let jobs = match fs::read_dir(&owner_path) {
            Ok(jobs) => jobs,
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
}

struct ObservedFile {
    index: u32,
    next_chunk: u32,
    size_bytes: u64,
    digest: Sha256,
    output: Option<File>,
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
    if manifest.format_version != FORMAT_VERSION
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
        match observed.get(&entry.index) {
            Some(actual)
                if actual.size_bytes == entry.size_bytes
                    && actual.sha256_hex == entry.sha256_hex
                    && actual.chunk_count == entry.chunk_count => {}
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
    mut plaintext: Vec<u8>,
) -> Result<(), QueueMediaError> {
    let ciphertext_len = plaintext
        .len()
        .checked_add(AEAD_TAG_BYTES)
        .ok_or_else(|| QueueMediaError::Corrupt("frame length overflow".into()))?;
    let aad = frame_aad(media_set, *ordinal, false, ciphertext_len);
    let ciphertext = encryptor
        .encrypt_next(aead_stream::aead::Payload {
            msg: &plaintext,
            aad: &aad,
        })
        .map_err(|_| QueueMediaError::Authentication)?;
    plaintext.zeroize();
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
    let mut aad = Vec::with_capacity(
        64 + media_set.owner_id.len() + media_set.job_id.len() + media_set.set_id.len(),
    );
    aad.extend_from_slice(b"mold.queue-media.stream");
    aad.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
    append_aad_field(&mut aad, media_set.owner_id.as_bytes());
    append_aad_field(&mut aad, media_set.job_id.as_bytes());
    append_aad_field(&mut aad, media_set.set_id.as_bytes());
    aad.extend_from_slice(&ordinal.to_be_bytes());
    aad.push(u8::from(is_final));
    aad.extend_from_slice(&(ciphertext_len as u64).to_be_bytes());
    aad
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
    for pair in value.as_bytes().chunks_exact(2) {
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
            create_directory_owner_only(path)?;
            crate::dir_sync::sync_directory(parent)?;
            let metadata = fs::symlink_metadata(path)?;
            verify_private_directory_metadata(path, &metadata)
        }
        Err(error) => Err(error.into()),
    }
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
            return Err(QueueMediaError::InsecurePath(format!(
                "{} is not owned by the current user with mode 0700",
                path.display()
            )));
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
            "active" | "retired" | "staging" | "ephemeral" => {
                if tree_contains_non_directory_entry(&entry.path())? {
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
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(QueueMediaError::InsecurePath(format!(
            "{} must be a current-user-owned 0600 regular file",
            path.display()
        )));
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
        assert_eq!(
            QueueMediaStore::security_mode().unwrap(),
            QueueMediaSecurityMode::WindowsDpapiCurrentUser
        );

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
                &[
                    SealMedia::path("first_frame", "secret-visible-name.png", &input),
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
                &[SealMedia::bytes("source", "secret-name", vec![1, 2, 3])],
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
                &[SealMedia::bytes("source", "one", vec![4])],
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
        let first = store.seal("owner", "job-1", &media()).unwrap();
        let second = store.seal("owner", "job-2", &media()).unwrap();
        assert_ne!(first.set_id, second.set_id);
        assert_ne!(bundle_bytes(&store, &first), bundle_bytes(&store, &second));
    }

    #[test]
    fn a_job_can_have_only_one_bundle_even_after_retirement() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let media = [SealMedia::bytes("source", "one", vec![1])];
        let reference = store.seal("owner", "job", &media).unwrap();
        assert!(matches!(
            store.seal("owner", "job", &media),
            Err(QueueMediaError::JobAlreadySealed { .. })
        ));
        store.retire(&reference).unwrap();
        assert!(matches!(
            store.seal("owner", "job", &media),
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
                &[SealMedia::bytes("source", "one", vec![1, 2, 3])],
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
                &[SealMedia::bytes("source", "one", vec![1, 2, 3])],
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
                    &[SealMedia::bytes(
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
            assert!(fs::read_dir(store.root.join("ephemeral"))
                .unwrap()
                .next()
                .is_none());
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
                &[SealMedia::bytes(
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
                    &[SealMedia::bytes("source", "one", vec![1])],
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
                &[SealMedia::bytes("source", "one", vec![1])],
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
                &[SealMedia::bytes("source", "one", vec![1])],
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
                &[SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        let separately_deleted = store
            .seal(
                "owner",
                "delete-active",
                &[SealMedia::bytes("source", "one", vec![1])],
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
        store
            .seal_file(
                &reference,
                None,
                &[SealMedia::bytes("source", "one", vec![1, 2, 3])],
                &staging,
            )
            .unwrap();
        let report = store.inspect_owner("owner");
        assert_eq!(report.staging, vec![reference.clone()]);
        assert!(report.unrecognized.is_empty());
        store.delete_staging(&reference).unwrap();
        assert!(!staging.exists());
    }

    #[test]
    fn inspection_retains_tampered_and_symlink_entries_as_unrecognized() {
        let home = tempfile::tempdir().unwrap();
        let store = open_store(home.path());
        let reference = store
            .seal(
                "owner",
                "job",
                &[SealMedia::bytes("source", "one", vec![1])],
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
                &[SealMedia::bytes("source", "one", vec![1])],
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
            store.seal("owner", "job", &[SealMedia::path("source", "source", link)]),
            Err(QueueMediaError::InsecurePath(_))
        ));
    }
}
