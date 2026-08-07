//! Authenticated, bounded ingress for MiniMax H3 reference media.
//!
//! Bearer handles are carried only in headers on stable URLs. The store keeps
//! only salted handle digests, stages under the shared `MOLD_HOME` cache, and
//! converts every public authority into a private [`ResolvedReferenceSet`]
//! before a request may enter queue or recovery code.

use axum::{
    body::Body,
    extract::{Extension, State},
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use mold_core::{
    minimax_h3, GenerationReference, GenerationReferenceAuthority, GenerationReferenceMetadata,
    GenerationReferenceProvenance, ReferenceUploadCompleteResponse, ReferenceUploadSessionRequest,
    ReferenceUploadSessionResponse, ReferenceUploadSlot,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::io::AsyncWriteExt as _;
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use crate::{auth::ApiKeyAuthenticated, routes::ApiError, state::AppState};

pub const UPLOAD_HANDLE_HEADER: &str = "x-mold-reference-upload";
pub const SESSION_HANDLE_HEADER: &str = "x-mold-reference-upload-session";
pub const MAX_REFERENCE_UPLOAD_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSION_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_HOST_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSION_REQUEST_BYTES: usize = 256 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY: usize = 4;
pub const MAX_CONCURRENT_REFERENCE_UPLOADS: usize = 4;
pub const REFERENCE_UPLOAD_SESSION_TTL: Duration = Duration::from_secs(30 * 60);

#[derive(Clone)]
pub struct ReferenceUploadStore {
    inner: Arc<tokio::sync::Mutex<StoreInner>>,
    writers: Arc<tokio::sync::Semaphore>,
    root: Arc<PathBuf>,
    _lifetime: Arc<StoreLifetime>,
    resolved_bytes: Arc<AtomicU64>,
    handle_salt: Arc<[u8; 32]>,
}

struct StoreLifetime {
    root: PathBuf,
}

impl Drop for StoreLifetime {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("failed to remove reference-upload runtime staging: {error}");
            }
        }
    }
}

impl std::fmt::Debug for ReferenceUploadStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadStore")
            .field("root", &"<private MOLD_HOME cache>")
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct StoreInner {
    sessions: HashMap<String, UploadSession>,
    upload_sessions: HashMap<String, String>,
    reserved_bytes: u64,
}

struct UploadSession {
    identity: String,
    scope_sha256: String,
    expires_at_ms: u64,
    dir: PathBuf,
    cancel: CancellationToken,
    slots: HashMap<String, UploadSlot>,
    reserved_bytes: u64,
    consuming: bool,
}

struct UploadSlot {
    reference: u32,
    descriptor: GenerationReference,
    path: PathBuf,
    state: UploadState,
}

enum UploadState {
    Empty,
    Uploading {
        reserved_bytes: u64,
    },
    Complete {
        size_bytes: u64,
        metadata: Box<GenerationReferenceMetadata>,
    },
}

struct UploadSource {
    path: PathBuf,
    metadata: GenerationReferenceMetadata,
}

/// Private resolved media authority. Paths never enter JSON, queue metadata,
/// logs, or durable recovery state; dropping the last owner removes staging.
pub struct ResolvedReferenceSet {
    root: PathBuf,
    entries: Vec<ResolvedReference>,
    fingerprint: String,
    _quota: ResolvedQuotaLease,
    _store_lifetime: Arc<StoreLifetime>,
}

struct ResolvedQuotaLease {
    reserved_bytes: u64,
    counter: Arc<AtomicU64>,
}

impl ResolvedQuotaLease {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self {
            reserved_bytes: 0,
            counter,
        }
    }

    fn adopt_reserved(&mut self, bytes: u64) {
        self.reserved_bytes = self.reserved_bytes.saturating_add(bytes);
    }
}

impl Drop for ResolvedQuotaLease {
    fn drop(&mut self) {
        self.counter
            .fetch_sub(self.reserved_bytes, Ordering::AcqRel);
    }
}

pub struct ResolvedReference {
    pub metadata: GenerationReferenceMetadata,
    pub path: PathBuf,
}

impl std::fmt::Debug for ResolvedReferenceSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedReferenceSet")
            .field("entries", &self.entries.len())
            .field("fingerprint", &self.fingerprint)
            .field("root", &"<redacted>")
            .finish()
    }
}

impl ResolvedReferenceSet {
    pub fn entries(&self) -> &[ResolvedReference] {
        &self.entries
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

impl Drop for ResolvedReferenceSet {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("failed to remove private reference staging: {error}");
            }
        }
    }
}

impl ReferenceUploadStore {
    pub fn from_mold_home() -> Self {
        let mold_home = mold_core::Config::mold_dir().unwrap_or_else(|| PathBuf::from(".mold"));
        let mold_home = if mold_home.is_absolute() {
            mold_home
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(mold_home)
        };
        let root = mold_home
            .join("cache")
            .join("reference-uploads")
            .join(format!("runtime-{}", uuid::Uuid::new_v4()));
        let mut handle_salt = [0_u8; 32];
        getrandom::fill(&mut handle_salt).expect("OS randomness is required for upload handles");
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(StoreInner::default())),
            writers: Arc::new(tokio::sync::Semaphore::new(
                MAX_CONCURRENT_REFERENCE_UPLOADS,
            )),
            root: Arc::new(root.clone()),
            _lifetime: Arc::new(StoreLifetime { root }),
            resolved_bytes: Arc::new(AtomicU64::new(0)),
            handle_salt: Arc::new(handle_salt),
        }
    }

    #[cfg(test)]
    fn at(root: PathBuf) -> Self {
        let mut store = Self::from_mold_home();
        store.root = Arc::new(root.clone());
        store._lifetime = Arc::new(StoreLifetime { root });
        store
    }

    #[cfg(test)]
    pub(crate) fn staging_exists(&self) -> bool {
        self.root.exists()
    }

    fn digest_handle(&self, domain: &[u8], handle: &str) -> String {
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update(self.handle_salt.as_ref());
        digest.update(handle.as_bytes());
        format!("{:x}", digest.finalize())
    }

    fn session_digest(&self, handle: &str) -> String {
        self.digest_handle(b"mold.reference-upload.session.v1\0", handle)
    }

    fn upload_digest(&self, handle: &str) -> String {
        self.digest_handle(b"mold.reference-upload.slot.v1\0", handle)
    }

    pub(crate) fn scope_sha256(
        &self,
        request: &mold_core::GenerateRequest,
    ) -> Result<String, ApiError> {
        request_scope_sha256(request)
    }

    async fn ensure_roots(&self) -> Result<(), ApiError> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || create_private_directories(&root))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!(
                    "failed to prepare private reference staging: {error:#}"
                ))
            })
    }

    async fn purge_expired(&self) {
        let now = unix_time_ms();
        let dirs = {
            let mut inner = self.inner.lock().await;
            let expired = inner
                .sessions
                .iter()
                .filter_map(|(digest, session)| {
                    (session.expires_at_ms <= now && !session.consuming).then_some(digest.clone())
                })
                .collect::<Vec<_>>();
            expired
                .into_iter()
                .filter_map(|digest| remove_session_locked(&mut inner, &digest))
                .collect::<Vec<_>>()
        };
        for dir in dirs {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
    }

    pub async fn create_session(
        &self,
        identity: &str,
        instance_id: &str,
        payload: ReferenceUploadSessionRequest,
    ) -> Result<ReferenceUploadSessionResponse, ApiError> {
        self.purge_expired().await;
        let references = payload.request.references.as_deref().ok_or_else(|| {
            ApiError::structured(
                "reference upload session requires ordered reference descriptors",
                "MINIMAX_H3_REFERENCE_REQUIRED",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("references".to_string()),
            )
        })?;
        minimax_h3::validate_reference_descriptors(references).map_err(ApiError::reference)?;
        let requested_count = payload.upload_references.len();
        let requested = payload
            .upload_references
            .into_iter()
            .collect::<HashSet<_>>();
        if requested.is_empty()
            || requested.len() != requested_count
            || requested.len() > references.len()
        {
            return Err(ApiError::structured(
                "upload_references must name at least one unique one-based reference",
                "MINIMAX_H3_REFERENCE_UPLOAD_SELECTION",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("upload_references".to_string()),
            ));
        }
        if requested
            .iter()
            .any(|reference| *reference == 0 || *reference as usize > references.len())
        {
            return Err(ApiError::structured(
                "upload_references contains an out-of-range one-based reference",
                "MINIMAX_H3_REFERENCE_UPLOAD_SELECTION",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("upload_references".to_string()),
            ));
        }

        self.ensure_roots().await?;
        let session_handle = random_handle("mrs_")?;
        let session_digest = self.session_digest(&session_handle);
        let session_dir = self.root.join(format!("session-{}", uuid::Uuid::new_v4()));
        let dir = session_dir.clone();
        tokio::task::spawn_blocking(move || create_private_directory(&dir))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!(
                    "failed to create private reference session: {error:#}"
                ))
            })?;

        let expires_at_ms =
            unix_time_ms().saturating_add(REFERENCE_UPLOAD_SESSION_TTL.as_millis() as u64);
        let scope_sha256 = request_scope_sha256(&payload.request)?;
        let mut response_slots = Vec::with_capacity(requested.len());
        let mut slots = HashMap::new();
        for reference in requested.iter().copied() {
            let handle = random_handle("mru_")?;
            let digest = self.upload_digest(&handle);
            response_slots.push(ReferenceUploadSlot { reference, handle });
            slots.insert(
                digest,
                UploadSlot {
                    reference,
                    descriptor: references[reference as usize - 1].clone(),
                    path: session_dir.join(format!("reference-{reference}.media")),
                    state: UploadState::Empty,
                },
            );
        }
        response_slots.sort_by_key(|slot| slot.reference);

        let mut inner = self.inner.lock().await;
        let owned_sessions = inner
            .sessions
            .values()
            .filter(|session| session.identity == identity)
            .count();
        if owned_sessions >= MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY {
            drop(inner);
            let _ = tokio::fs::remove_dir_all(session_dir).await;
            return Err(ApiError::with_code(
                "too many active reference upload sessions",
                "REFERENCE_UPLOAD_SESSION_LIMIT",
                StatusCode::TOO_MANY_REQUESTS,
            ));
        }
        for upload_digest in slots.keys() {
            inner
                .upload_sessions
                .insert(upload_digest.clone(), session_digest.clone());
        }
        inner.sessions.insert(
            session_digest,
            UploadSession {
                identity: identity.to_string(),
                scope_sha256: scope_sha256.clone(),
                expires_at_ms,
                dir: session_dir,
                cancel: CancellationToken::new(),
                slots,
                reserved_bytes: 0,
                consuming: false,
            },
        );
        Ok(ReferenceUploadSessionResponse {
            instance_id: instance_id.to_string(),
            expires_at_ms,
            request_scope_sha256: scope_sha256,
            session_handle,
            uploads: response_slots,
        })
    }

    pub async fn upload(
        &self,
        identity: &str,
        instance_id: &str,
        handle: &str,
        content_type: &str,
        content_length: u64,
        body: Body,
    ) -> Result<ReferenceUploadCompleteResponse, ApiError> {
        self.purge_expired().await;
        if content_length == 0 || content_length > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                format!(
                    "reference upload Content-Length must be in 1..={MAX_REFERENCE_UPLOAD_FILE_BYTES}"
                ),
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        let upload_digest = self.upload_digest(handle);
        let (session_digest, path, descriptor, reference, cancel) = {
            let mut inner = self.inner.lock().await;
            let session_digest = inner
                .upload_sessions
                .get(&upload_digest)
                .cloned()
                .ok_or_else(unknown_upload)?;
            let host_reserved = inner
                .reserved_bytes
                .saturating_add(self.resolved_bytes.load(Ordering::Acquire));
            let values = {
                let session = inner
                    .sessions
                    .get_mut(&session_digest)
                    .ok_or_else(unknown_upload)?;
                if session.identity != identity || session.consuming {
                    return Err(unknown_upload());
                }
                if session.reserved_bytes.saturating_add(content_length)
                    > MAX_REFERENCE_UPLOAD_SESSION_BYTES
                    || host_reserved.saturating_add(content_length)
                        > MAX_REFERENCE_UPLOAD_HOST_BYTES
                {
                    return Err(ApiError::with_code(
                        "reference upload quota exceeded",
                        "REFERENCE_UPLOAD_QUOTA",
                        StatusCode::PAYLOAD_TOO_LARGE,
                    ));
                }
                let slot = session
                    .slots
                    .get_mut(&upload_digest)
                    .ok_or_else(unknown_upload)?;
                if !matches!(slot.state, UploadState::Empty) {
                    return Err(ApiError::with_code(
                        "reference upload handle has already been used",
                        "REFERENCE_UPLOAD_ALREADY_USED",
                        StatusCode::CONFLICT,
                    ));
                }
                let declared_type = reference_mime_type(&slot.descriptor);
                if !mime_matches(declared_type, content_type) {
                    return Err(reference_error(
                        slot.reference,
                        "mime_type",
                        "MINIMAX_H3_REFERENCE_MEDIA_TYPE",
                        format!(
                            "reference {} Content-Type does not match its descriptor",
                            slot.reference
                        ),
                    ));
                }
                slot.state = UploadState::Uploading {
                    reserved_bytes: content_length,
                };
                session.reserved_bytes += content_length;
                (
                    slot.path.clone(),
                    slot.descriptor.clone(),
                    slot.reference,
                    session.cancel.clone(),
                )
            };
            inner.reserved_bytes += content_length;
            (session_digest, values.0, values.1, values.2, values.3)
        };

        let _permit = self.writers.clone().acquire_owned().await.map_err(|_| {
            ApiError::generation_unavailable("reference upload service is shutting down")
        })?;
        let result = stream_to_private_file(&path, content_length, body, &cancel).await;
        let digest = match result {
            Ok(digest) => digest,
            Err(error) => {
                self.rollback_upload(&session_digest, &upload_digest).await;
                let _ = tokio::fs::remove_file(&path).await;
                return Err(error);
            }
        };
        let probe_path = path.clone();
        let probe_descriptor = descriptor.clone();
        let probe = tokio::task::spawn_blocking(move || {
            probe_reference(&probe_path, &probe_descriptor, reference, &digest)
        })
        .await
        .map_err(|error| ApiError::internal(format!("reference probe task failed: {error}")));
        let metadata = match probe {
            Ok(Ok(metadata)) => metadata,
            Ok(Err(error)) | Err(error) => {
                self.rollback_upload(&session_digest, &upload_digest).await;
                let _ = tokio::fs::remove_file(&path).await;
                return Err(error);
            }
        };

        let mut inner = self.inner.lock().await;
        let session = inner
            .sessions
            .get_mut(&session_digest)
            .ok_or_else(unknown_upload)?;
        let slot = session
            .slots
            .get_mut(&upload_digest)
            .ok_or_else(unknown_upload)?;
        let reserved_bytes = match slot.state {
            UploadState::Uploading { reserved_bytes } => reserved_bytes,
            _ => return Err(unknown_upload()),
        };
        slot.state = UploadState::Complete {
            size_bytes: reserved_bytes,
            metadata: Box::new(metadata.clone()),
        };
        Ok(ReferenceUploadCompleteResponse {
            instance_id: instance_id.to_string(),
            reference: slot.reference,
            metadata,
        })
    }

    async fn rollback_upload(&self, session_digest: &str, upload_digest: &str) {
        let mut inner = self.inner.lock().await;
        let mut release = 0;
        if let Some(session) = inner.sessions.get_mut(session_digest) {
            if let Some(slot) = session.slots.get_mut(upload_digest) {
                if let UploadState::Uploading { reserved_bytes } = slot.state {
                    release = reserved_bytes;
                    slot.state = UploadState::Empty;
                    session.reserved_bytes = session.reserved_bytes.saturating_sub(release);
                }
            }
        }
        inner.reserved_bytes = inner.reserved_bytes.saturating_sub(release);
    }

    pub async fn cancel_session(&self, identity: &str, handle: &str) -> Result<(), ApiError> {
        self.purge_expired().await;
        let digest = self.session_digest(handle);
        let dir = {
            let mut inner = self.inner.lock().await;
            let session = inner.sessions.get(&digest).ok_or_else(unknown_session)?;
            if session.identity != identity || session.consuming {
                return Err(unknown_session());
            }
            remove_session_locked(&mut inner, &digest).ok_or_else(unknown_session)?
        };
        let _ = tokio::fs::remove_dir_all(dir).await;
        Ok(())
    }

    async fn upload_sources_for_request(
        &self,
        identity: &str,
        request: &mold_core::GenerateRequest,
        frozen_scope_sha256: Option<&str>,
    ) -> Result<(Option<String>, HashMap<String, UploadSource>), ApiError> {
        let Some(references) = request.references.as_deref() else {
            return Ok((None, HashMap::new()));
        };
        let handles = references
            .iter()
            .enumerate()
            .filter_map(|(index, reference)| match reference.media() {
                GenerationReferenceAuthority::Upload { handle } => Some((
                    u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1),
                    handle,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        if handles.is_empty() {
            return Ok((None, HashMap::new()));
        }
        let scope = match frozen_scope_sha256 {
            Some(scope) => scope.to_string(),
            None => request_scope_sha256(request)?,
        };
        let mut inner = self.inner.lock().await;
        let mut session_digest = None::<String>;
        let mut sources = HashMap::new();
        let mut seen = HashSet::new();
        for (reference, handle) in handles {
            let upload_digest = self.upload_digest(handle);
            if !seen.insert(upload_digest.clone()) {
                return Err(reference_error(
                    reference,
                    "media.handle",
                    "REFERENCE_UPLOAD_HANDLE_REUSED",
                    "a reference upload handle cannot be reused within one request",
                ));
            }
            let candidate = inner
                .upload_sessions
                .get(&upload_digest)
                .cloned()
                .ok_or_else(unknown_upload)?;
            if session_digest
                .as_ref()
                .is_some_and(|current| current != &candidate)
            {
                return Err(reference_error(
                    reference,
                    "media.handle",
                    "REFERENCE_UPLOAD_SESSION_MISMATCH",
                    "all upload references in one request must come from one bound session",
                ));
            }
            session_digest.get_or_insert(candidate.clone());
            let session = inner.sessions.get(&candidate).ok_or_else(unknown_upload)?;
            if session.identity != identity || session.scope_sha256 != scope || session.consuming {
                return Err(unknown_upload());
            }
            let slot = session
                .slots
                .get(&upload_digest)
                .ok_or_else(unknown_upload)?;
            let UploadState::Complete {
                size_bytes,
                metadata,
            } = &slot.state
            else {
                return Err(ApiError::structured(
                    format!("reference {} upload is not complete", slot.reference),
                    "REFERENCE_UPLOAD_INCOMPLETE",
                    StatusCode::CONFLICT,
                    Some(slot.reference),
                    Some("media.handle".to_string()),
                ));
            };
            debug_assert!(*size_bytes > 0);
            sources.insert(
                upload_digest,
                UploadSource {
                    path: slot.path.clone(),
                    metadata: metadata.as_ref().clone(),
                },
            );
        }
        let digest = session_digest.expect("upload handles yielded a session");
        let session = inner.sessions.get_mut(&digest).ok_or_else(unknown_upload)?;
        if sources.len() != session.slots.len() {
            return Err(ApiError::structured(
                "every upload slot in the bound session must be consumed together",
                "REFERENCE_UPLOAD_SESSION_INCOMPLETE",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("references".to_string()),
            ));
        }
        session.consuming = true;
        Ok((Some(digest), sources))
    }

    async fn reserve_resolved_bytes(
        &self,
        lease: &mut ResolvedQuotaLease,
        bytes: u64,
    ) -> Result<(), ApiError> {
        let inner = self.inner.lock().await;
        let active = inner
            .reserved_bytes
            .saturating_add(self.resolved_bytes.load(Ordering::Acquire));
        if active.saturating_add(bytes) > MAX_REFERENCE_UPLOAD_HOST_BYTES {
            return Err(ApiError::with_code(
                "reference staging quota exceeded",
                "REFERENCE_UPLOAD_QUOTA",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        if bytes > 0 {
            self.resolved_bytes.fetch_add(bytes, Ordering::AcqRel);
            lease.adopt_reserved(bytes);
        }
        Ok(())
    }

    async fn finish_consumed_session(
        &self,
        digest: Option<&str>,
        success: bool,
        quota: &mut ResolvedQuotaLease,
    ) {
        let Some(digest) = digest else { return };
        let dir = {
            let mut inner = self.inner.lock().await;
            if success {
                let transferred = inner
                    .sessions
                    .get_mut(digest)
                    .map(|session| {
                        let transferred = session.reserved_bytes;
                        session.reserved_bytes = 0;
                        transferred
                    })
                    .unwrap_or_default();
                inner.reserved_bytes = inner.reserved_bytes.saturating_sub(transferred);
                self.resolved_bytes.fetch_add(transferred, Ordering::AcqRel);
                quota.adopt_reserved(transferred);
                remove_session_locked(&mut inner, digest)
            } else {
                if let Some(session) = inner.sessions.get_mut(digest) {
                    session.consuming = false;
                }
                None
            }
        };
        if let Some(dir) = dir {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
    }

    pub async fn resolve_request(
        &self,
        identity: &str,
        request: &mut mold_core::GenerateRequest,
        media_roots: &[PathBuf],
        frozen_scope_sha256: Option<&str>,
    ) -> Result<Option<ResolvedReferenceSet>, ApiError> {
        self.purge_expired().await;
        let Some(references) = request.references.clone() else {
            return Ok(None);
        };
        minimax_h3::validate_references(&references).map_err(ApiError::reference)?;
        let (session_digest, upload_sources) = self
            .upload_sources_for_request(identity, request, frozen_scope_sha256)
            .await?;
        let mut quota = ResolvedQuotaLease::new(self.resolved_bytes.clone());
        self.ensure_roots().await?;
        let resolved_root = self.root.join(format!("resolved-{}", uuid::Uuid::new_v4()));
        let root = resolved_root.clone();
        if let Err(error) = tokio::task::spawn_blocking(move || create_private_directory(&root))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!("failed to create resolved staging: {error:#}"))
            })
        {
            self.finish_consumed_session(session_digest.as_deref(), false, &mut quota)
                .await;
            return Err(error);
        }

        let result = resolve_all_references(
            &resolved_root,
            &references,
            media_roots,
            &upload_sources,
            self,
            &mut quota,
        )
        .await;
        let (entries, descriptors) = match result {
            Ok(value) => value,
            Err(error) => {
                let _ = tokio::fs::remove_dir_all(&resolved_root).await;
                self.finish_consumed_session(session_digest.as_deref(), false, &mut quota)
                    .await;
                return Err(error);
            }
        };
        self.finish_consumed_session(session_digest.as_deref(), true, &mut quota)
            .await;
        let metadata = entries
            .iter()
            .map(|entry| entry.metadata.clone())
            .collect::<Vec<_>>();
        let fingerprint = mold_core::generation_reference_fingerprint(&metadata);
        request.references = Some(descriptors);
        Ok(Some(ResolvedReferenceSet {
            root: resolved_root,
            entries,
            fingerprint,
            _quota: quota,
            _store_lifetime: self._lifetime.clone(),
        }))
    }
}

async fn resolve_all_references(
    root: &Path,
    references: &[GenerationReference],
    media_roots: &[PathBuf],
    upload_sources: &HashMap<String, UploadSource>,
    store: &ReferenceUploadStore,
    quota: &mut ResolvedQuotaLease,
) -> Result<(Vec<ResolvedReference>, Vec<GenerationReference>), ApiError> {
    let mut entries = Vec::with_capacity(references.len());
    let mut descriptors = Vec::with_capacity(references.len());
    for (index, reference) in references.iter().enumerate() {
        let one_based = u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1);
        let target = root.join(format!("reference-{one_based}.media"));
        let metadata = match reference.media() {
            GenerationReferenceAuthority::Upload { handle } => {
                let digest = store.upload_digest(handle);
                let source = upload_sources.get(&digest).ok_or_else(unknown_upload)?;
                tokio::fs::hard_link(&source.path, &target)
                    .await
                    .map_err(|error| {
                        ApiError::internal(format!("failed to bind uploaded reference: {error}"))
                    })?;
                source.metadata.clone()
            }
            GenerationReferenceAuthority::Inline { data } => {
                let bytes = data.clone();
                store
                    .reserve_resolved_bytes(quota, bytes.len() as u64)
                    .await?;
                let target_clone = target.clone();
                let reference_clone = reference.clone();
                tokio::task::spawn_blocking(move || {
                    write_private_bytes(&target_clone, &bytes)?;
                    let digest = format!("{:x}", Sha256::digest(&bytes));
                    probe_reference(&target_clone, &reference_clone, one_based, &digest)
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??
            }
            GenerationReferenceAuthority::ServerPath { path } => {
                let source =
                    mold_core::resolve_server_media_path(path, media_roots).map_err(|error| {
                        reference_error(one_based, "media.path", "MINIMAX_H3_REFERENCE_PATH", error)
                    })?;
                let (source, source_bytes) = tokio::task::spawn_blocking(move || {
                    let source = crate::batch_transaction::open_regular_file_no_follow(&source)
                        .map_err(|error| {
                            ApiError::validation(format!(
                                "failed to safely open reference: {error:#}"
                            ))
                        })?;
                    let source_bytes = source
                        .metadata()
                        .map_err(|error| {
                            ApiError::validation(format!("failed to inspect reference: {error}"))
                        })?
                        .len();
                    if source_bytes > MAX_REFERENCE_UPLOAD_FILE_BYTES {
                        return Err(ApiError::with_code(
                            "server-local reference exceeds the per-file quota",
                            "REFERENCE_UPLOAD_TOO_LARGE",
                            StatusCode::PAYLOAD_TOO_LARGE,
                        ));
                    }
                    Ok((source, source_bytes))
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??;
                store.reserve_resolved_bytes(quota, source_bytes).await?;
                let target_clone = target.clone();
                let reference_clone = reference.clone();
                tokio::task::spawn_blocking(move || {
                    let digest = copy_private_bounded(source, &target_clone)?;
                    probe_reference(&target_clone, &reference_clone, one_based, &digest)
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??
            }
            GenerationReferenceAuthority::Descriptor => {
                return Err(reference_error(
                    one_based,
                    "media.authority",
                    "MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY",
                    "descriptor authority cannot enter generation",
                ));
            }
        };
        descriptors.push(reference_from_metadata(&metadata));
        entries.push(ResolvedReference {
            metadata,
            path: target,
        });
    }
    Ok((entries, descriptors))
}

fn remove_session_locked(inner: &mut StoreInner, digest: &str) -> Option<PathBuf> {
    let session = inner.sessions.remove(digest)?;
    session.cancel.cancel();
    for upload_digest in session.slots.keys() {
        inner.upload_sessions.remove(upload_digest);
    }
    inner.reserved_bytes = inner.reserved_bytes.saturating_sub(session.reserved_bytes);
    Some(session.dir)
}

fn request_scope_sha256(request: &mold_core::GenerateRequest) -> Result<String, ApiError> {
    let mut scoped = request.clone();
    scoped.references = scoped.references.as_ref().map(|references| {
        references
            .iter()
            .map(reference_as_descriptor)
            .collect::<Vec<_>>()
    });
    let wire = serde_json::to_vec(&scoped)
        .map_err(|error| ApiError::internal(format!("failed to bind upload request: {error}")))?;
    let mut digest = Sha256::new();
    digest.update(b"mold.reference-upload.request-scope.v1\0");
    digest.update(wire);
    Ok(format!("{:x}", digest.finalize()))
}

fn reference_as_descriptor(reference: &GenerationReference) -> GenerationReference {
    match reference {
        GenerationReference::Image {
            provenance,
            mime_type,
            width,
            height,
            ..
        } => GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
        },
        GenerationReference::Video {
            provenance,
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => GenerationReference::Video {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
            frame_count: *frame_count,
            duration_ms: *duration_ms,
            fps: *fps,
            has_audio: *has_audio,
            audio_duration_ms: *audio_duration_ms,
            audio_sample_count: *audio_sample_count,
            audio_sample_rate: *audio_sample_rate,
            audio_channels: *audio_channels,
        },
        GenerationReference::Audio {
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            duration_ms: *duration_ms,
            sample_rate: *sample_rate,
            channels: *channels,
            sample_count: *sample_count,
        },
    }
}

fn reference_from_metadata(metadata: &GenerationReferenceMetadata) -> GenerationReference {
    let provenance = GenerationReferenceProvenance {
        name: metadata.name.clone(),
        sha256: Some(metadata.sha256.clone()),
    };
    match metadata.kind {
        mold_core::GenerationReferenceKind::Image => GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            width: metadata.width.unwrap_or_default(),
            height: metadata.height.unwrap_or_default(),
        },
        mold_core::GenerationReferenceKind::Video => GenerationReference::Video {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            width: metadata.width.unwrap_or_default(),
            height: metadata.height.unwrap_or_default(),
            frame_count: metadata.frame_count,
            duration_ms: metadata.duration_ms.unwrap_or_default(),
            fps: metadata.fps.unwrap_or_default(),
            has_audio: metadata.has_audio,
            audio_duration_ms: metadata.audio_duration_ms,
            audio_sample_count: metadata.audio_sample_count,
            audio_sample_rate: metadata.audio_sample_rate,
            audio_channels: metadata.audio_channels,
        },
        mold_core::GenerationReferenceKind::Audio => GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            duration_ms: metadata.duration_ms.unwrap_or_default(),
            sample_rate: metadata.sample_rate.unwrap_or_default(),
            channels: metadata.channels.unwrap_or_default(),
            sample_count: metadata.sample_count,
        },
    }
}

async fn stream_to_private_file(
    path: &Path,
    expected: u64,
    body: Body,
    cancel: &CancellationToken,
) -> Result<String, ApiError> {
    let target = path.to_path_buf();
    let file = tokio::task::spawn_blocking(move || {
        crate::batch_transaction::create_private_file_no_follow(&target)
    })
    .await
    .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
    .map_err(|error| {
        ApiError::internal(format!("failed to create reference staging: {error:#}"))
    })?;
    let mut file = tokio::fs::File::from_std(file);
    let mut stream = body.into_data_stream();
    let mut written = 0_u64;
    let mut digest = Sha256::new();
    loop {
        let next = tokio::select! {
            _ = cancel.cancelled() => {
                return Err(ApiError::cancelled("reference upload session was cancelled"));
            }
            next = stream.next() => next,
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| {
            ApiError::validation(format!("reference upload body failed: {error}"))
        })?;
        written = written.saturating_add(chunk.len() as u64);
        if written > expected || written > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                "reference upload exceeded its declared or maximum size",
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        digest.update(&chunk);
        file.write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))?;
    }
    if written != expected {
        return Err(ApiError::validation(format!(
            "reference upload ended at {written} bytes; expected {expected}"
        )));
    }
    file.flush()
        .await
        .map_err(|error| ApiError::internal(format!("failed to flush reference: {error}")))?;
    file.sync_all()
        .await
        .map_err(|error| ApiError::internal(format!("failed to fsync reference: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn write_private_bytes(path: &Path, bytes: &[u8]) -> Result<(), ApiError> {
    let mut file =
        crate::batch_transaction::create_private_file_no_follow(path).map_err(|error| {
            ApiError::internal(format!("failed to create reference staging: {error:#}"))
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))
}

fn copy_private_bounded(mut source: std::fs::File, target: &Path) -> Result<String, ApiError> {
    let mut target =
        crate::batch_transaction::create_private_file_no_follow(target).map_err(|error| {
            ApiError::internal(format!("failed to create reference staging: {error:#}"))
        })?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    let mut digest = Sha256::new();
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|error| ApiError::validation(format!("failed to read reference: {error}")))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                "server-local reference exceeds the per-file quota",
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        digest.update(&buffer[..read]);
        target
            .write_all(&buffer[..read])
            .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))?;
    }
    target
        .sync_all()
        .map_err(|error| ApiError::internal(format!("failed to fsync reference: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn probe_reference(
    path: &Path,
    expected: &GenerationReference,
    reference: u32,
    digest: &str,
) -> Result<GenerationReferenceMetadata, ApiError> {
    let declared_digest = expected.provenance().sha256.as_deref();
    if declared_digest.is_some_and(|declared| !declared.eq_ignore_ascii_case(digest)) {
        return Err(reference_error(
            reference,
            "provenance.sha256",
            "MINIMAX_H3_REFERENCE_DIGEST_MISMATCH",
            format!("reference {reference}: declared sha256 does not match staged bytes"),
        ));
    }
    let provenance = GenerationReferenceProvenance {
        name: expected.provenance().name.clone(),
        sha256: Some(digest.to_string()),
    };
    let canonical = match expected {
        GenerationReference::Image {
            mime_type,
            width,
            height,
            ..
        } => {
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let reader = image::ImageReader::new(BufReader::new(file))
                .with_guessed_format()
                .map_err(|error| unsupported_media(reference, error))?;
            let format = reader
                .format()
                .ok_or_else(|| unsupported_media(reference, "unknown image format"))?;
            let observed_mime = image_mime(format)
                .ok_or_else(|| unsupported_media(reference, "unsupported image format"))?;
            let (observed_width, observed_height) = reader
                .into_dimensions()
                .map_err(|error| unsupported_media(reference, error))?;
            require_equal(reference, "mime_type", mime_type.as_str(), observed_mime)?;
            require_equal(reference, "width", *width, observed_width)?;
            require_equal(reference, "height", *height, observed_height)?;
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: observed_mime.to_string(),
                width: observed_width,
                height: observed_height,
            }
        }
        GenerationReference::Video {
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => {
            require_equal(reference, "mime_type", mime_type.as_str(), "video/mp4")?;
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let probe = mold_inference::ltx2::media::probe_video_file(file)
                .map_err(|error| unsupported_media(reference, error))?;
            require_equal(reference, "width", *width, probe.width)?;
            require_equal(reference, "height", *height, probe.height)?;
            let observed_frame_count = probe
                .frames
                .filter(|frames| *frames > 0)
                .ok_or_else(|| unsupported_media(reference, "video frame count is unavailable"))?;
            if let Some(declared) = *frame_count {
                require_equal(reference, "frame_count", declared, observed_frame_count)?;
            }
            if (*fps - f64::from(probe.fps)).abs() > 0.51 {
                return Err(drift_error(reference, "fps", *fps, probe.fps));
            }
            let observed_duration = probe
                .duration_ms
                .ok_or_else(|| unsupported_media(reference, "video duration is unavailable"))?;
            let tolerance = (1_000_u64 / u64::from(probe.fps.max(1))).max(50);
            if duration_ms.abs_diff(observed_duration) > tolerance {
                return Err(drift_error(
                    reference,
                    "duration_ms",
                    *duration_ms,
                    observed_duration,
                ));
            }
            require_equal(reference, "has_audio", *has_audio, probe.has_audio)?;
            let observed_audio = if probe.has_audio {
                Some(
                    mold_inference::ltx2::media::probe_decoded_mp4_audio_file(path)
                        .map_err(|error| unsupported_media(reference, error))?
                        .ok_or_else(|| {
                            unsupported_media(
                                reference,
                                "MP4 declares audio but decoded no soundtrack samples",
                            )
                        })?,
                )
            } else {
                None
            };
            let observed_audio_duration = observed_audio.map(|audio| {
                audio
                    .samples_per_channel
                    .saturating_mul(1_000)
                    .div_ceil(u64::from(audio.sample_rate))
            });
            if let (Some(declared), Some(observed)) = (*audio_duration_ms, observed_audio_duration)
            {
                if declared.abs_diff(observed) > tolerance {
                    return Err(drift_error(
                        reference,
                        "audio_duration_ms",
                        declared,
                        observed,
                    ));
                }
            }
            if let Some(audio) = observed_audio {
                if let Some(declared) = *audio_sample_count {
                    require_equal(
                        reference,
                        "audio_sample_count",
                        declared,
                        audio.samples_per_channel,
                    )?;
                }
                if let Some(declared) = *audio_sample_rate {
                    require_equal(reference, "audio_sample_rate", declared, audio.sample_rate)?;
                }
                if let Some(declared) = *audio_channels {
                    require_equal(reference, "audio_channels", declared, audio.channels)?;
                }
            }
            GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "video/mp4".to_string(),
                width: probe.width,
                height: probe.height,
                frame_count: Some(observed_frame_count),
                duration_ms: observed_duration,
                fps: f64::from(probe.fps),
                has_audio: probe.has_audio,
                audio_duration_ms: observed_audio_duration,
                audio_sample_count: observed_audio.map(|audio| audio.samples_per_channel),
                audio_sample_rate: observed_audio.map(|audio| audio.sample_rate),
                audio_channels: observed_audio.map(|audio| audio.channels),
            }
        }
        GenerationReference::Audio {
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => {
            if !matches!(
                mime_type.trim().to_ascii_lowercase().as_str(),
                "audio/wav" | "audio/x-wav" | "audio/wave"
            ) {
                return Err(unsupported_media(
                    reference,
                    "only RIFF/WAVE audio is accepted by reference ingress",
                ));
            }
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let wav = probe_wav(file).map_err(|error| unsupported_media(reference, error))?;
            require_equal(reference, "sample_rate", *sample_rate, wav.sample_rate)?;
            require_equal(reference, "channels", *channels, wav.channels)?;
            if let Some(declared) = *sample_count {
                require_equal(reference, "sample_count", declared, wav.sample_count)?;
            }
            if duration_ms.abs_diff(wav.duration_ms) > 2 {
                return Err(drift_error(
                    reference,
                    "duration_ms",
                    *duration_ms,
                    wav.duration_ms,
                ));
            }
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "audio/wav".to_string(),
                duration_ms: wav.duration_ms,
                sample_rate: wav.sample_rate,
                channels: wav.channels,
                sample_count: Some(wav.sample_count),
            }
        }
    };
    minimax_h3::reference_prepared_shape(&canonical).map_err(ApiError::reference)?;
    canonical
        .redacted_metadata(reference.saturating_sub(1) as usize)
        .ok_or_else(|| ApiError::internal("validated reference metadata lost its digest"))
}

#[derive(Debug)]
struct WavProbe {
    duration_ms: u64,
    sample_rate: u32,
    channels: u16,
    sample_count: u64,
}

fn probe_wav(file: std::fs::File) -> anyhow::Result<WavProbe> {
    let file_len = file.metadata()?.len();
    anyhow::ensure!(file_len <= MAX_REFERENCE_UPLOAD_FILE_BYTES);
    let mut reader = BufReader::new(file);
    let mut header = [0_u8; 12];
    reader.read_exact(&mut header)?;
    anyhow::ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "not a RIFF/WAVE file"
    );
    let riff_end = 8_u64
        .checked_add(u64::from(u32::from_le_bytes(header[4..8].try_into()?)))
        .ok_or_else(|| anyhow::anyhow!("WAVE container length overflowed"))?;
    anyhow::ensure!(
        riff_end >= 12 && riff_end <= file_len,
        "WAVE container is truncated"
    );
    let mut sample_rate = None;
    let mut channels = None;
    let mut byte_rate = None;
    let mut block_align = None;
    let mut data_bytes = None;
    while reader.stream_position()?.saturating_add(8) <= riff_end {
        let mut chunk = [0_u8; 8];
        reader.read_exact(&mut chunk)?;
        let length = u32::from_le_bytes(chunk[4..].try_into()?) as u64;
        let data_start = reader.stream_position()?;
        let padded_length = length
            .checked_add(length % 2)
            .ok_or_else(|| anyhow::anyhow!("WAVE chunk length overflowed"))?;
        let next_chunk = data_start
            .checked_add(padded_length)
            .ok_or_else(|| anyhow::anyhow!("WAVE chunk offset overflowed"))?;
        anyhow::ensure!(
            next_chunk <= riff_end && next_chunk <= file_len,
            "WAVE chunk is truncated"
        );
        match &chunk[..4] {
            b"fmt " => {
                anyhow::ensure!(length >= 16, "WAVE fmt chunk is truncated");
                let mut format = [0_u8; 16];
                reader.read_exact(&mut format)?;
                let encoding = u16::from_le_bytes(format[..2].try_into()?);
                anyhow::ensure!(
                    matches!(encoding, 1 | 3),
                    "unsupported WAVE sample encoding"
                );
                let observed_channels = u16::from_le_bytes(format[2..4].try_into()?);
                let observed_sample_rate = u32::from_le_bytes(format[4..8].try_into()?);
                let observed_byte_rate = u32::from_le_bytes(format[8..12].try_into()?);
                let observed_block_align = u16::from_le_bytes(format[12..14].try_into()?);
                let bits_per_sample = u16::from_le_bytes(format[14..16].try_into()?);
                anyhow::ensure!(
                    observed_channels > 0
                        && observed_sample_rate > 0
                        && observed_block_align > 0
                        && bits_per_sample > 0
                        && bits_per_sample % 8 == 0,
                    "WAVE format fields are invalid"
                );
                let expected_align = observed_channels
                    .checked_mul(bits_per_sample / 8)
                    .ok_or_else(|| anyhow::anyhow!("WAVE block alignment overflowed"))?;
                anyhow::ensure!(
                    observed_block_align == expected_align,
                    "WAVE block alignment is inconsistent"
                );
                anyhow::ensure!(
                    observed_byte_rate
                        == observed_sample_rate.saturating_mul(u32::from(observed_block_align)),
                    "WAVE byte rate is inconsistent"
                );
                channels = Some(observed_channels);
                sample_rate = Some(observed_sample_rate);
                byte_rate = Some(observed_byte_rate);
                block_align = Some(observed_block_align);
            }
            b"data" => {
                data_bytes = Some(length);
            }
            _ => {}
        }
        reader.seek(SeekFrom::Start(next_chunk))?;
        if sample_rate.is_some() && data_bytes.is_some() {
            break;
        }
    }
    let sample_rate = sample_rate
        .filter(|rate| *rate > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE sample rate is missing"))?;
    let channels = channels
        .filter(|channels| *channels > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE channel count is missing"))?;
    let byte_rate = u64::from(
        byte_rate
            .filter(|rate| *rate > 0)
            .ok_or_else(|| anyhow::anyhow!("WAVE byte rate is missing"))?,
    );
    let block_align =
        u64::from(block_align.ok_or_else(|| anyhow::anyhow!("WAVE block alignment is missing"))?);
    let data_bytes = data_bytes
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE data chunk is missing or empty"))?;
    anyhow::ensure!(
        data_bytes % block_align == 0,
        "WAVE data is not frame-aligned"
    );
    let duration_ms = data_bytes.saturating_mul(1_000).div_ceil(byte_rate);
    Ok(WavProbe {
        duration_ms,
        sample_rate,
        channels,
        sample_count: data_bytes / block_align,
    })
}

fn image_mime(format: image::ImageFormat) -> Option<&'static str> {
    match format {
        image::ImageFormat::Png => Some("image/png"),
        image::ImageFormat::Jpeg => Some("image/jpeg"),
        image::ImageFormat::WebP => Some("image/webp"),
        image::ImageFormat::Gif => Some("image/gif"),
        _ => None,
    }
}

fn require_equal<T: PartialEq + std::fmt::Display>(
    reference: u32,
    field: &str,
    declared: T,
    observed: T,
) -> Result<(), ApiError> {
    if declared == observed {
        Ok(())
    } else {
        Err(drift_error(reference, field, declared, observed))
    }
}

fn drift_error(
    reference: u32,
    field: &str,
    declared: impl std::fmt::Display,
    observed: impl std::fmt::Display,
) -> ApiError {
    reference_error(
        reference,
        field,
        "MINIMAX_H3_REFERENCE_DESCRIPTOR_DRIFT",
        format!(
            "reference {reference}: declared {field}={declared} does not match probed {observed}"
        ),
    )
}

fn unsupported_media(reference: u32, error: impl std::fmt::Display) -> ApiError {
    reference_error(
        reference,
        "mime_type",
        "MINIMAX_H3_REFERENCE_MEDIA_UNSUPPORTED",
        format!("reference {reference}: media probe failed: {error}"),
    )
}

fn reference_error(
    reference: u32,
    field: &str,
    code: &str,
    message: impl Into<String>,
) -> ApiError {
    ApiError::structured(
        message,
        code,
        StatusCode::UNPROCESSABLE_ENTITY,
        Some(reference),
        Some(field.to_string()),
    )
}

fn unknown_upload() -> ApiError {
    ApiError::with_code(
        "unknown, expired, or unauthorized reference upload handle",
        "REFERENCE_UPLOAD_NOT_FOUND",
        StatusCode::NOT_FOUND,
    )
}

fn unknown_session() -> ApiError {
    ApiError::with_code(
        "unknown, expired, or unauthorized reference upload session",
        "REFERENCE_UPLOAD_SESSION_NOT_FOUND",
        StatusCode::NOT_FOUND,
    )
}

fn random_handle(prefix: &str) -> Result<String, ApiError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|error| ApiError::internal(format!("failed to mint upload handle: {error}")))?;
    Ok(format!("{prefix}{}", URL_SAFE_NO_PAD.encode(bytes)))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn reference_mime_type(reference: &GenerationReference) -> &str {
    match reference {
        GenerationReference::Image { mime_type, .. }
        | GenerationReference::Video { mime_type, .. }
        | GenerationReference::Audio { mime_type, .. } => mime_type,
    }
}

fn mime_matches(declared: &str, actual: &str) -> bool {
    let declared = declared.trim().to_ascii_lowercase();
    let actual = actual
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    declared == actual
        || matches!(
            (declared.as_str(), actual.as_str()),
            ("audio/wav", "audio/x-wav")
                | ("audio/x-wav", "audio/wav")
                | ("audio/wave", "audio/wav")
        )
}

fn create_private_directories(root: &Path) -> anyhow::Result<()> {
    crate::batch_transaction::create_private_directories_no_follow(root)
}

fn create_private_directory(path: &Path) -> anyhow::Result<()> {
    crate::batch_transaction::create_private_directory_no_follow(path)
}

fn authenticated_identity(
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
) -> Result<String, ApiError> {
    authenticated
        .map(|Extension(authenticated)| authenticated.identity.clone())
        .ok_or_else(|| {
            ApiError::with_code(
                "API key authentication is required for reference uploads",
                "UNAUTHORIZED",
                StatusCode::UNAUTHORIZED,
            )
        })
}

#[utoipa::path(
    post,
    path = "/api/generate/reference-upload-sessions",
    tag = "generation",
    request_body = ReferenceUploadSessionRequest,
    responses(
        (status = 200, description = "Request-bound one-use upload handles", body = ReferenceUploadSessionResponse),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 422, description = "Invalid ordered reference descriptor"),
        (status = 451, description = "MiniMax H3 use is not legally activated"),
    )
)]
pub(crate) async fn create_reference_upload_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    Json(payload): Json<ReferenceUploadSessionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    if payload.request.batch_size > 1 {
        return Err(ApiError::with_code(
            "MiniMax H3 references do not support durable server batches yet",
            "MINIMAX_H3_REFERENCE_BATCH_UNSUPPORTED",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    // Compliance must run before directory/session allocation or download.
    crate::routes::require_server_generation_request_activation(&state, &payload.request, None)
        .await?;
    if minimax_h3::resolve_model_name(&payload.request.model).is_none() {
        return Err(ApiError::validation(
            "reference upload sessions are only valid for MiniMax H3",
        ));
    }
    let response = state
        .reference_uploads
        .create_session(&identity, &state.instance_id, payload)
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}

#[utoipa::path(
    put,
    path = "/api/generate/reference-upload",
    tag = "generation",
    params(
        ("X-Mold-Reference-Upload" = String, Header, description = "One-use upload handle advertised by server capabilities"),
        ("Content-Length" = u64, Header, description = "Exact bounded media byte length"),
        ("Content-Type" = String, Header, description = "Declared media type, verified by content probe"),
    ),
    responses(
        (status = 200, description = "Content-sniffed reference metadata", body = ReferenceUploadCompleteResponse),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 404, description = "Unknown, expired, or unauthorized upload handle"),
        (status = 413, description = "Upload exceeds a file, session, or host quota"),
        (status = 422, description = "Media differs from its bound descriptor"),
    )
)]
pub(crate) async fn upload_reference(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
    body: Body,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let handle = required_secret_header(&headers, UPLOAD_HANDLE_HEADER)?;
    let content_length = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| {
            ApiError::with_code(
                "reference uploads require an exact Content-Length",
                "REFERENCE_UPLOAD_LENGTH_REQUIRED",
                StatusCode::LENGTH_REQUIRED,
            )
        })?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            ApiError::with_code(
                "reference uploads require Content-Type",
                "REFERENCE_UPLOAD_CONTENT_TYPE_REQUIRED",
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            )
        })?;
    let response = state
        .reference_uploads
        .upload(
            &identity,
            &state.instance_id,
            handle,
            content_type,
            content_length,
            body,
        )
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}

#[utoipa::path(
    delete,
    path = "/api/generate/reference-upload-sessions",
    tag = "generation",
    params(
        ("X-Mold-Reference-Upload-Session" = String, Header, description = "Session handle advertised by server capabilities"),
    ),
    responses(
        (status = 204, description = "Session cancelled and staged bytes removed"),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 404, description = "Unknown, expired, or unauthorized session"),
    )
)]
pub(crate) async fn cancel_reference_upload_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
) -> Result<StatusCode, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let handle = required_secret_header(&headers, SESSION_HANDLE_HEADER)?;
    state
        .reference_uploads
        .cancel_session(&identity, handle)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

fn required_secret_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str, ApiError> {
    let value = headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty() && value.len() <= minimax_h3::MAX_REFERENCE_UPLOAD_HANDLE_BYTES
        })
        .ok_or_else(unknown_upload)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::GenerationReferenceKind;

    fn png_reference(
        media: GenerationReferenceAuthority,
        sha256: Option<String>,
    ) -> GenerationReference {
        GenerationReference::Image {
            media,
            provenance: GenerationReferenceProvenance {
                name: Some("anchor.png".to_string()),
                sha256,
            },
            mime_type: "image/png".to_string(),
            width: 2,
            height: 2,
        }
    }

    fn png_bytes() -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    fn request(reference: GenerationReference) -> mold_core::GenerateRequest {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "reference print",
            "model": minimax_h3::REF2VA_COMFY,
            "width": minimax_h3::DEFAULT_WIDTH,
            "height": minimax_h3::DEFAULT_HEIGHT,
            "steps": minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "batch_size": 1,
            "frames": minimax_h3::MIN_FRAMES,
            "fps": minimax_h3::FIXED_FPS,
            "output_format": "mp4"
        }))
        .unwrap();
        request.references = Some(vec![reference]);
        request
    }

    #[tokio::test]
    async fn upload_session_streams_and_resolves_to_private_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = png_bytes();
        let byte_count = bytes.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let descriptor = png_reference(GenerationReferenceAuthority::Descriptor, Some(digest));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        assert!(!format!("{session:?}").contains(&handle));
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &handle,
                "image/png",
                byte_count,
                Body::from(bytes),
            )
            .await
            .unwrap();
        assert_eq!(complete.metadata.kind, GenerationReferenceKind::Image);
        assert!(complete.metadata.prepared_shape.is_some());

        let mut final_request = request(png_reference(
            GenerationReferenceAuthority::Upload { handle },
            Some(complete.metadata.sha256.clone()),
        ));
        let resolved = store
            .resolve_request("auth-a", &mut final_request, &[], None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolved.entries().len(), 1);
        assert!(resolved.entries()[0].path.is_file());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), byte_count);
        assert!(matches!(
            final_request.references.as_ref().unwrap()[0].media(),
            GenerationReferenceAuthority::Descriptor
        ));
        assert!(!format!("{resolved:?}").contains(dir.path().to_string_lossy().as_ref()));
        drop(resolved);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn digest_drift_is_structured_and_partial_file_is_not_reusable() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let descriptor = png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some("00".repeat(32)),
        );
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let bytes = png_bytes();
        let error = store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "image/png",
                bytes.len() as u64,
                Body::from(bytes),
            )
            .await
            .unwrap_err();
        assert_eq!(error.reference, Some(1));
        assert_eq!(error.field.as_deref(), Some("provenance.sha256"));
    }

    #[test]
    fn request_scope_redacts_upload_handles() {
        let digest = "11".repeat(32);
        let descriptor = request(png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some(digest.clone()),
        ));
        let upload = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: "secret-handle".to_string(),
            },
            Some(digest),
        ));
        assert_eq!(
            request_scope_sha256(&descriptor).unwrap(),
            request_scope_sha256(&upload).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn private_staging_rejects_symlinked_parent_components() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let actual = dir.path().join("actual");
        std::fs::create_dir(&actual).unwrap();
        let link = dir.path().join("link");
        symlink(&actual, &link).unwrap();

        assert!(create_private_directories(&link.join("runtime")).is_err());
        assert!(!actual.join("runtime").exists());
    }

    #[test]
    fn wav_probe_rejects_declared_data_past_the_file_boundary() {
        let mut file = tempfile::tempfile().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&36_u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&32_000_u32.to_le_bytes());
        bytes.extend_from_slice(&128_000_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4_096_u32.to_le_bytes());
        file.write_all(&bytes).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        assert!(probe_wav(file)
            .unwrap_err()
            .to_string()
            .contains("truncated"));
    }
}
