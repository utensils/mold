//! Resumable, owner-bound Library video artifacts. Payloads never enter queue media.
use crate::{batch_transaction::ArchivedChildIdentity, routes::ApiError, state::AppState};
use anyhow::{ensure, Context, Result};
use axum::{
    body::{Body, Bytes},
    extract::{Extension, Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    routing::{get, post, put},
    Json, Router,
};
use mold_core::video_artifacts::{VideoArtifactFile, VideoJobArtifacts};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, Write},
    path::{Path as FsPath, PathBuf},
    sync::{Arc, OnceLock},
};

pub(crate) const CHUNK_BYTES: usize = 4 * 1024 * 1024;
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    Provenance,
    Pcm,
}
impl Role {
    fn filename(self) -> &'static str {
        match self {
            Self::Provenance => "provenance.jsonl",
            Self::Pcm => "final-audio.pcm",
        }
    }
    fn expected(self, assets: &VideoJobArtifacts) -> Result<&VideoArtifactFile> {
        match self {
            Self::Provenance => Ok(&assets.provenance),
            Self::Pcm => assets
                .pcm
                .as_ref()
                .map(|v| &v.file)
                .context("this movie has no PCM artifact"),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Offsets {
    #[serde(with = "mold_core::video_job::decimal_u64")]
    pub provenance: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcm: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Status {
    pub handle: String,
    pub archive_identity_sha256: String,
    pub offsets: Offsets,
    pub complete: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    owner: ArchivedChildIdentity,
    artifacts: VideoJobArtifacts,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Begin {
    pub archive_identity_sha256: String,
    pub artifacts: VideoJobArtifacts,
}
#[derive(Default, Deserialize)]
pub(crate) struct Offset {
    #[serde(default)]
    offset_bytes: u64,
}
fn owner_hash(owner: &ArchivedChildIdentity) -> Result<String> {
    crate::gallery_source_media::archive_identity_sha256(owner)
        .map_err(|e| anyhow::anyhow!(e.error))
}
fn handle(owner: &ArchivedChildIdentity, artifacts: &VideoJobArtifacts) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(owner, artifacts))?)
    ))
}
fn private_dir(path: &FsPath) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => {
            // Persist every newly created namespace component, not just leaf files.
            if let Some(parent) = path.parent() {
                crate::dir_sync::sync_directory(parent)?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(e) => return Err(e.into()),
    }
    check_dir(path)
}
fn check_dir(path: &FsPath) -> Result<()> {
    let m = std::fs::symlink_metadata(path)?;
    ensure!(
        m.is_dir() && !m.file_type().is_symlink(),
        "artifact directory is not private regular storage"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            m.permissions().mode() & 0o077 == 0,
            "artifact directory is not private"
        );
    }
    Ok(())
}
fn import_dir(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
    create: bool,
) -> Result<PathBuf> {
    assets.validate().map_err(anyhow::Error::msg)?;
    let mut path = root.to_path_buf();
    for component in [
        None,
        Some("imports".to_owned()),
        Some(owner_hash(owner)?),
        Some(assets.bundle_id.clone()),
    ] {
        if let Some(component) = component {
            path.push(component);
        }
        if create {
            private_dir(&path)?;
        } else {
            check_dir(&path)?;
        }
    }
    Ok(path)
}
fn read_json<T: serde::de::DeserializeOwned>(path: &FsPath) -> Result<T> {
    let file = mold_core::secure_file::open_regular_file_no_follow(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            file.metadata()?.permissions().mode() & 0o077 == 0,
            "artifact descriptor is not private"
        );
    }
    ensure!(
        file.metadata()?.len() <= 16 * 1024,
        "artifact descriptor exceeds 16 KiB"
    );
    let mut bytes = Vec::new();
    file.take(16 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 16 * 1024, "artifact descriptor grew");
    Ok(serde_json::from_slice(&bytes)?)
}
fn write_json(path: &FsPath, name: &str, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(
        bytes.len() <= 16 * 1024,
        "artifact descriptor exceeds 16 KiB"
    );
    let mut temp = tempfile::NamedTempFile::new_in(path)?;
    temp.write_all(&bytes)?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(path.join(name))?;
    crate::dir_sync::sync_directory(path)?;
    Ok(())
}
fn rw(path: &FsPath, create: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(create);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "artifact is not a regular file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "artifact file is not private"
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            metadata.file_attributes() & 0x400 == 0,
            "artifact is a reparse point"
        );
    }
    Ok(file)
}
struct Import {
    path: PathBuf,
    owner: ArchivedChildIdentity,
    assets: VideoJobArtifacts,
    _lock: File,
}
impl Import {
    fn lock(
        root: &FsPath,
        owner: &ArchivedChildIdentity,
        assets: &VideoJobArtifacts,
        create: bool,
    ) -> Result<Self> {
        let path = import_dir(root, owner, assets, create)?;
        let lock = rw(&path.join("transfer.lock"), create)?;
        fs2::FileExt::lock_exclusive(&lock)?;
        Ok(Self {
            path,
            owner: owner.clone(),
            assets: assets.clone(),
            _lock: lock,
        })
    }
    fn check(&self) -> Result<()> {
        let stored: Manifest = read_json(&self.path.join("transfer.json"))?;
        ensure!(
            stored.version == 1 && stored.owner == self.owner && stored.artifacts == self.assets,
            "artifact transfer owner or descriptor changed"
        );
        Ok(())
    }
    fn initialize(&self) -> Result<()> {
        if self.path.join("transfer.json").try_exists()? {
            self.check()?;
        } else {
            write_json(
                &self.path,
                "transfer.json",
                &Manifest {
                    version: 1,
                    owner: self.owner.clone(),
                    artifacts: self.assets.clone(),
                },
            )?;
        }
        if self.path.join("manifest.json").try_exists()? {
            let stored: VideoJobArtifacts = read_json(&self.path.join("manifest.json"))?;
            ensure!(stored == self.assets, "artifact manifest changed");
        } else {
            write_json(&self.path, "manifest.json", &self.assets)?;
        }
        let sealed = self.sealed()?;
        for role in self.roles() {
            let mut file = rw(&self.path.join(role.filename()), true)?;
            let expected = role.expected(&self.assets)?;
            // A completed but unsealed corrupt upload is explicitly repairable on begin.
            // Partial uploads keep their durable offset; published seals are immutable.
            if !sealed && file.metadata()?.len() >= expected.size_bytes {
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 64 * 1024];
                let mut count = 0u64;
                loop {
                    let n = file.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    count = count
                        .checked_add(n as u64)
                        .context("artifact length overflow")?;
                    if count > expected.size_bytes {
                        break;
                    }
                    hash.update(&buffer[..n]);
                }
                if count != expected.size_bytes
                    || format!("{:x}", hash.finalize()) != expected.sha256
                {
                    file.set_len(0)?;
                }
            }
            file.sync_all()?;
        }
        crate::dir_sync::sync_directory(&self.path)?;
        Ok(())
    }
    fn roles(&self) -> Vec<Role> {
        if self.assets.pcm.is_some() {
            vec![Role::Provenance, Role::Pcm]
        } else {
            vec![Role::Provenance]
        }
    }
    fn sealed(&self) -> Result<bool> {
        if !self.path.join("owner.json").try_exists()? {
            return Ok(false);
        }
        let owner: ArchivedChildIdentity = read_json(&self.path.join("owner.json"))?;
        ensure!(
            owner == self.owner,
            "artifact seal belongs to another owner"
        );
        Ok(true)
    }
    fn status(&self) -> Result<Status> {
        self.check()?;
        let size = |role: Role| -> Result<u64> {
            let f = mold_core::secure_file::open_regular_file_no_follow(
                &self.path.join(role.filename()),
            )?;
            let size = f.metadata()?.len();
            ensure!(
                size <= role.expected(&self.assets)?.size_bytes,
                "artifact progress exceeds committed size"
            );
            Ok(size)
        };
        let complete = self.sealed()?;
        if complete {
            verify_files(&self.path, &self.owner, &self.assets, &mut || Ok(()))?;
        }
        Ok(Status {
            handle: handle(&self.owner, &self.assets)?,
            archive_identity_sha256: owner_hash(&self.owner)?,
            offsets: Offsets {
                provenance: size(Role::Provenance)?,
                pcm: self
                    .assets
                    .pcm
                    .as_ref()
                    .map(|_| size(Role::Pcm).map(|n| n.to_string()))
                    .transpose()?,
            },
            complete,
        })
    }
    fn chunk(&self, role: Role, offset: u64, bytes: &[u8]) -> Result<Status> {
        self.check()?;
        ensure!(!self.sealed()?, "completed artifact imports are immutable");
        ensure!(
            !bytes.is_empty() && bytes.len() <= CHUNK_BYTES,
            "artifact chunk must contain 1 to 4 MiB"
        );
        let expected = role.expected(&self.assets)?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .context("artifact offset overflow")?;
        ensure!(
            end <= expected.size_bytes,
            "chunk exceeds the committed artifact size"
        );
        let mut file = rw(&self.path.join(role.filename()), false)?;
        let current = file.metadata()?.len();
        if offset < current {
            ensure!(
                end <= current,
                "chunk overlaps unfinished progress; refresh its offset"
            );
            file.seek(std::io::SeekFrom::Start(offset))?;
            let mut existing = vec![0; bytes.len()];
            file.read_exact(&mut existing)?;
            ensure!(
                existing == bytes,
                "retried chunk differs from committed bytes"
            );
        } else {
            ensure!(
                offset == current,
                "artifact chunk leaves a hole; refresh its offset"
            );
            file.seek(std::io::SeekFrom::Start(offset))?;
            file.write_all(bytes)?;
        }
        file.sync_all()?;
        self.status()
    }
    fn verify(&self) -> Result<()> {
        self.check()?;
        for role in self.roles() {
            rw(&self.path.join(role.filename()), false)?.sync_all()?;
        }
        verify_files(&self.path, &self.owner, &self.assets, &mut || Ok(())).map(|_| ())
    }
    fn seal(&self) -> Result<Status> {
        self.check()?;
        if !self.sealed()? {
            write_json(&self.path, "manifest.json", &self.assets).or_else(|e| {
                let current: VideoJobArtifacts = read_json(&self.path.join("manifest.json"))?;
                ensure!(current == self.assets, "artifact descriptor changed: {e}");
                Ok(())
            })?;
            write_json(&self.path, "owner.json", &self.owner)?;
        }
        self.status()
    }
}
#[cfg(test)]
fn begin(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
) -> Result<Status> {
    let import = Import::lock(root, owner, assets, true)?;
    import.initialize()?;
    import.status()
}
fn owned_directory(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
) -> Result<PathBuf> {
    assets.validate().map_err(anyhow::Error::msg)?;
    check_dir(root)?;
    // A repaired owner-scoped import takes precedence over an old flat bundle.
    let imported = root
        .join("imports")
        .join(owner_hash(owner)?)
        .join(&assets.bundle_id);
    if imported.join("owner.json").try_exists()? {
        let imported = import_dir(root, owner, assets, false)?;
        let recorded: ArchivedChildIdentity = read_json(&imported.join("owner.json"))?;
        ensure!(recorded == *owner, "artifact owner changed");
        return Ok(imported);
    }
    let flat = root.join(&assets.bundle_id);
    check_dir(&flat)?;
    let recorded: ArchivedChildIdentity = read_json(&flat.join("owner.json"))?;
    ensure!(recorded == *owner, "artifact owner changed");
    Ok(flat)
}
fn open_files(path: &FsPath, assets: &VideoJobArtifacts) -> Result<Vec<(File, VideoArtifactFile)>> {
    check_dir(path)?;
    let manifest: VideoJobArtifacts = read_json(&path.join("manifest.json"))?;
    ensure!(
        manifest == *assets,
        "artifact manifest differs from published metadata"
    );
    [Role::Provenance, Role::Pcm]
        .into_iter()
        .filter(|r| !matches!(r, Role::Pcm) || assets.pcm.is_some())
        .map(|r| {
            let expected = r.expected(assets)?.clone();
            let file =
                mold_core::secure_file::open_regular_file_no_follow(&path.join(r.filename()))?;
            ensure!(
                file.metadata()?.len() == expected.size_bytes,
                "artifact byte count differs from published metadata"
            );
            Ok((file, expected))
        })
        .collect()
}
pub(crate) fn verify_files(
    path: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<String> {
    let mut files = open_files(path, assets)?;
    let facts = files
        .iter()
        .map(|(f, _)| crate::batch_transaction::ArchiveFileFacts::from_metadata(&f.metadata()?))
        .collect::<Result<Vec<_>>>()?;
    let key = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(path, owner, assets, &facts))?)
    );
    // Unix ctime/device/inode facts prevent same-size rewrites from inheriting proof.
    // The private receipt survives process restart and has no inventory-size cache cap.
    #[cfg(unix)]
    if read_json::<String>(&path.join("verified-facts.json"))
        .ok()
        .as_ref()
        == Some(&key)
    {
        return Ok(key);
    }
    for (index, (file, expected)) in files.iter_mut().enumerate() {
        let mut sha = Sha256::new();
        let mut count = 0u64;
        let mut buffer = [0; 64 * 1024];
        loop {
            checkpoint()?;
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            count = count
                .checked_add(n as u64)
                .context("artifact length overflow")?;
            ensure!(count <= expected.size_bytes, "artifact grew");
            sha.update(&buffer[..n]);
        }
        ensure!(
            count == expected.size_bytes && format!("{:x}", sha.finalize()) == expected.sha256,
            "artifact checksum mismatch"
        );
        ensure!(
            crate::batch_transaction::ArchiveFileFacts::from_metadata(&file.metadata()?)?
                == facts[index],
            "artifact changed during verification"
        );
    }
    let bytes = serde_json::to_vec(&key)?;
    let mut receipt = tempfile::NamedTempFile::new_in(path)?;
    receipt.write_all(&bytes)?;
    receipt.as_file().sync_all()?;
    receipt.persist(path.join("verified-facts.json"))?;
    crate::dir_sync::sync_directory(path)?;
    Ok(key)
}
pub(crate) fn evidence(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
) -> Result<String> {
    let path = owned_directory(root, owner, assets)?;
    verify_files(&path, owner, assets, &mut || Ok(()))
}
fn source_chunk(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
    role: Role,
    offset: u64,
) -> Result<Vec<u8>> {
    let path = owned_directory(root, owner, assets)?;
    let manifest: VideoJobArtifacts = read_json(&path.join("manifest.json"))?;
    ensure!(manifest == *assets, "artifact descriptor changed");
    let expected = role.expected(assets)?;
    let mut file =
        mold_core::secure_file::open_regular_file_no_follow(&path.join(role.filename()))?;
    let before = crate::batch_transaction::ArchiveFileFacts::from_metadata(&file.metadata()?)?;
    ensure!(
        file.metadata()?.len() == expected.size_bytes && offset < expected.size_bytes,
        "artifact offset or size differs from committed authority"
    );
    file.seek(std::io::SeekFrom::Start(offset))?;
    let size = (expected.size_bytes - offset).min(CHUNK_BYTES as u64) as usize;
    let mut bytes = vec![0; size];
    file.read_exact(&mut bytes)?;
    ensure!(
        crate::batch_transaction::ArchiveFileFacts::from_metadata(&file.metadata()?)? == before,
        "artifact changed during chunk read"
    );
    Ok(bytes)
}
pub(crate) fn release_import(
    root: &FsPath,
    owner: &ArchivedChildIdentity,
    assets: &VideoJobArtifacts,
) -> Result<bool> {
    let path = root
        .join("imports")
        .join(owner_hash(owner)?)
        .join(&assets.bundle_id);
    if !path.try_exists()? {
        return Ok(false);
    }
    let exact = import_dir(root, owner, assets, false)?;
    let manifest: Manifest = read_json(&exact.join("transfer.json"))?;
    ensure!(
        manifest.owner == *owner && manifest.artifacts == *assets,
        "artifact cleanup owner differs"
    );
    std::fs::remove_dir_all(&exact)?;
    crate::dir_sync::sync_directory(exact.parent().unwrap())?;
    Ok(true)
}

fn api(error: impl std::fmt::Display) -> ApiError {
    tracing::warn!(error=%error,"Library video artifacts unavailable");
    ApiError::with_code("This video’s saved audio or clip history is missing, incomplete, or failed its integrity check.","VIDEO_ARTIFACTS_UNAVAILABLE",StatusCode::CONFLICT)
}
fn fence(headers: &HeaderMap, name: &str, expected: &str) -> Result<(), ApiError> {
    if headers.get(name).and_then(|v| v.to_str().ok()) != Some(expected) {
        return Err(ApiError::with_code(format!("The captured {name} identity changed or was not supplied; refresh this Library copy."),"VIDEO_ARTIFACT_FENCE_CHANGED",StatusCode::CONFLICT));
    }
    Ok(())
}
fn authorize(
    auth: Option<&Extension<crate::auth::AuthState>>,
    key: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(), ApiError> {
    if !crate::gallery_source_media::private_media_authorized(auth, key) {
        return Err(ApiError::with_code(
            "Video artifact transfer requires authentication.",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    Ok(())
}
fn target(
    state: &AppState,
    filename: &str,
    headers: &HeaderMap,
    source: bool,
) -> Result<(PathBuf, ArchivedChildIdentity, VideoJobArtifacts), ApiError> {
    fence(
        headers,
        if source {
            "x-mold-source-instance"
        } else {
            "x-mold-destination-instance"
        },
        state.instance_id.as_str(),
    )?;
    crate::routes::validate_gallery_filename(filename)?;
    let config = state.config.blocking_read();
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("Library output is disabled"));
    }
    let output = config.effective_output_dir();
    drop(config);
    let entry = state
        .gallery_publication_gate
        .validated_transfer_entry(&output, filename)
        .map_err(api)?
        .ok_or_else(|| ApiError::not_found("Library output is missing"))?;
    let identity = owner_hash(&entry.identity).map_err(api)?;
    fence(headers, "x-mold-gallery-archive-identity", &identity)?;
    let assets = entry
        .record
        .metadata
        .video_job
        .and_then(|v| v.artifacts)
        .ok_or_else(|| {
            ApiError::not_found("This video has no publication-owned artifact descriptor")
        })?;
    assets.validate().map_err(api)?;
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .ok_or_else(|| api("captured metadata database unavailable"))?;
    Ok((
        crate::video_archive_assets::root_from_db(db).map_err(api)?,
        entry.identity,
        assets,
    ))
}
fn slots(source: bool) -> Arc<tokio::sync::Semaphore> {
    static SOURCE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    static SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    (if source { &SOURCE } else { &SLOTS })
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
}
async fn permit(source: bool) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    slots(source).acquire_owned().await.map_err(api)
}
struct Payload {
    bytes: Vec<u8>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}
impl AsRef<[u8]> for Payload {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
async fn chunk_get(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    key: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((filename, role)): Path<(String, Role)>,
    Query(offset): Query<Offset>,
    headers: HeaderMap,
) -> Result<(HeaderMap, Body), ApiError> {
    authorize(auth.as_ref(), key.as_ref())?;
    let permit = permit(true).await?;
    let (bytes, permit) = tokio::task::spawn_blocking(move || {
        let _read = state.gallery_publication_gate.blocking_read();
        let (root, owner, assets) = target(&state, &filename, &headers, true)?;
        let bytes = source_chunk(&root, &owner, &assets, role, offset.offset_bytes).map_err(api)?;
        Ok::<_, ApiError>((bytes, permit))
    })
    .await
    .map_err(api)??;
    let mut response = HeaderMap::new();
    response.insert(
        "cache-control",
        HeaderValue::from_static("private, no-store"),
    );
    response.insert(
        "content-type",
        HeaderValue::from_static("application/octet-stream"),
    );
    Ok((
        response,
        Body::from(Bytes::from_owner(Payload {
            bytes,
            _permit: permit,
        })),
    ))
}
#[derive(Clone)]
enum Action {
    Begin(VideoJobArtifacts, String),
    Status(String),
    Chunk(String, Role, u64, Bytes),
    Complete(String),
}
async fn mutate(
    state: AppState,
    filename: String,
    headers: HeaderMap,
    action: Action,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<Json<Status>, ApiError> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (root, owner, assets) = target(&state, &filename, &headers, false)?;
        if let Action::Begin(wanted, identity) = &action {
            if wanted != &assets || identity != &owner_hash(&owner).map_err(api)? {
                return Err(api(
                    "artifact begin descriptor differs from committed destination metadata",
                ));
            }
        }
        let import = Import::lock(&root, &owner, &assets, matches!(&action, Action::Begin(..)))
            .map_err(api)?;
        if let Action::Status(h) | Action::Chunk(h, ..) | Action::Complete(h) = &action {
            if h != &handle(&owner, &assets).map_err(api)? {
                return Err(api("artifact transfer handle belongs to another output"));
            }
        }
        // Recovery may hash a full staged file. Keep the exact import lock, but
        // never hold the global publication gate while reading unbounded payloads.
        let prepared_status = match &action {
            Action::Begin(..) => {
                import.initialize().map_err(api)?;
                Some(import.status().map_err(api)?)
            }
            Action::Status(_) => Some(import.status().map_err(api)?),
            _ => None,
        };
        {
            let _read = state.gallery_publication_gate.blocking_read();
            let (_, current, current_assets) = target(&state, &filename, &headers, false)?;
            if current != owner || current_assets != assets {
                return Err(api("destination archive changed"));
            }
            match &action {
                Action::Begin(..) | Action::Status(_) => {
                    return Ok(Json(
                        prepared_status
                            .ok_or_else(|| api("artifact read preparation unavailable"))?,
                    ))
                }
                Action::Chunk(_, role, offset, bytes) => {
                    return Ok(Json(import.chunk(*role, *offset, bytes).map_err(api)?))
                }
                Action::Complete(_) => {}
            }
        }
        // Hash large files without holding the global Gallery publication gate.
        import.verify().map_err(api)?;
        let _read = state.gallery_publication_gate.blocking_read();
        let (_, current, current_assets) = target(&state, &filename, &headers, false)?;
        if current != owner || current_assets != assets {
            return Err(api(
                "destination archive changed during artifact verification",
            ));
        }
        Ok(Json(import.seal().map_err(api)?))
    })
    .await
    .map_err(api)?
}
async fn begin_post(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    key: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path(filename): Path<String>,
    headers: HeaderMap,
    Json(spec): Json<Begin>,
) -> Result<Json<Status>, ApiError> {
    authorize(auth.as_ref(), key.as_ref())?;
    let p = permit(false).await?;
    mutate(
        state,
        filename,
        headers,
        Action::Begin(spec.artifacts, spec.archive_identity_sha256),
        p,
    )
    .await
}
async fn status_get(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    key: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((filename, handle)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Json<Status>, ApiError> {
    authorize(auth.as_ref(), key.as_ref())?;
    let p = permit(false).await?;
    mutate(state, filename, headers, Action::Status(handle), p).await
}
async fn chunk_put(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    key: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((filename, handle, role)): Path<(String, String, Role)>,
    Query(offset): Query<Offset>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Status>, ApiError> {
    authorize(auth.as_ref(), key.as_ref())?;
    let p = permit(false).await?;
    let bytes = axum::body::to_bytes(body, CHUNK_BYTES).await.map_err(|_| {
        ApiError::with_code(
            "Artifact chunks are limited to 4 MiB.",
            "VIDEO_ARTIFACT_CHUNK_TOO_LARGE",
            StatusCode::PAYLOAD_TOO_LARGE,
        )
    })?;
    mutate(
        state,
        filename,
        headers,
        Action::Chunk(handle, role, offset.offset_bytes, bytes),
        p,
    )
    .await
}
async fn complete_post(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    key: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((filename, handle)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Json<Status>, ApiError> {
    authorize(auth.as_ref(), key.as_ref())?;
    let p = permit(false).await?;
    mutate(state, filename, headers, Action::Complete(handle), p).await
}
pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/api/gallery/video-artifacts/:filename/transfer",
            post(begin_post),
        )
        .route(
            "/api/gallery/video-artifacts/:filename/transfer/:handle",
            get(status_get),
        )
        .route(
            "/api/gallery/video-artifacts/:filename/transfer/:handle/complete",
            post(complete_post),
        )
        .route(
            "/api/gallery/video-artifacts/:filename/transfer/:handle/:role",
            put(chunk_put),
        )
        .route(
            "/api/gallery/video-artifacts/:filename/:role",
            get(chunk_get),
        )
        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024))
        .layer(axum::middleware::map_response(
            |mut response: axum::response::Response| async move {
                response.headers_mut().insert(
                    "cache-control",
                    HeaderValue::from_static("private, no-store"),
                );
                response
            },
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn home() -> tempfile::TempDir {
        let home = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(home.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        home
    }
    fn owner(name: &str) -> ArchivedChildIdentity {
        ArchivedChildIdentity {
            parent_id: name.into(),
            attempt_generation: 0,
            child_index: 0,
            final_name: format!("{name}.mp4"),
            checksum_sha256: "a".repeat(64),
            size_bytes: 100,
        }
    }
    fn assets(bytes: &[u8]) -> VideoJobArtifacts {
        VideoJobArtifacts {
            version: 1,
            bundle_id: "b".repeat(64),
            provenance: VideoArtifactFile {
                sha256: format!("{:x}", Sha256::digest(bytes)),
                size_bytes: bytes.len() as u64,
            },
            pcm: None,
        }
    }
    #[test]
    fn begin_persists_exact_owner_descriptor_and_zero_offsets() {
        let home = home();
        let owner = owner("first");
        let assets = assets(b"one\n");
        let status = begin(home.path(), &owner, &assets).unwrap();
        assert!(!status.complete);
        assert_eq!(status.offsets.provenance, 0);
        assert_eq!(
            begin(home.path(), &owner, &assets).unwrap().handle,
            status.handle
        );
    }
    #[test]
    fn partial_retries_restart_and_duplicate_names_keep_separate_owners() {
        let home = home();
        let a = owner("first");
        let b = owner("second");
        let assets = assets(b"abcdef");
        begin(home.path(), &a, &assets).unwrap();
        {
            let upload = Import::lock(home.path(), &a, &assets, false).unwrap();
            upload.chunk(Role::Provenance, 0, b"abc").unwrap();
            assert_eq!(
                upload
                    .chunk(Role::Provenance, 0, b"abc")
                    .unwrap()
                    .offsets
                    .provenance,
                3
            );
            assert!(upload.chunk(Role::Provenance, 0, b"xyz").is_err());
            assert!(upload.chunk(Role::Provenance, 4, b"e").is_err());
        }
        assert_eq!(
            begin(home.path(), &a, &assets).unwrap().offsets.provenance,
            3
        );
        let upload = Import::lock(home.path(), &a, &assets, false).unwrap();
        upload.chunk(Role::Provenance, 3, b"def").unwrap();
        upload.verify().unwrap();
        assert!(upload.seal().unwrap().complete);
        assert!(upload.chunk(Role::Provenance, 0, b"abc").is_err());
        drop(upload);
        assert!(!begin(home.path(), &b, &assets).unwrap().complete);
        assert_eq!(
            source_chunk(home.path(), &a, &assets, Role::Provenance, 0).unwrap(),
            b"abcdef"
        );
        assert!(source_chunk(home.path(), &b, &assets, Role::Provenance, 0).is_err());
    }
    #[test]
    fn corruption_and_partial_cleanup_never_establish_completion() {
        let home = home();
        let owner = owner("first");
        let assets = assets(b"abcd");
        begin(home.path(), &owner, &assets).unwrap();
        let upload = Import::lock(home.path(), &owner, &assets, false).unwrap();
        upload.chunk(Role::Provenance, 0, b"bad!").unwrap();
        assert!(upload.verify().is_err());
        assert!(!upload.status().unwrap().complete);
        drop(upload);
        assert!(release_import(home.path(), &owner, &assets).unwrap());
        assert!(import_dir(home.path(), &owner, &assets, false).is_err());
    }
    #[test]
    fn large_pcm_streams_in_bounded_chunks_survives_restart_and_source_removal() {
        let source = home();
        let destination = home();
        let owner = owner("long");
        let chunk = vec![37u8; CHUNK_BYTES];
        let count = 9u64;
        let mut hash = Sha256::new();
        for _ in 0..count {
            hash.update(&chunk);
        }
        let mut assets = assets(b"provenance\n");
        assets.pcm = Some(mold_core::video_artifacts::VideoPcmArtifact {
            file: VideoArtifactFile {
                sha256: format!("{:x}", hash.finalize()),
                size_bytes: count * CHUNK_BYTES as u64,
            },
            sample_rate: 48000,
            channels: 2,
            frames: count * CHUNK_BYTES as u64 / 8,
        });
        begin(source.path(), &owner, &assets).unwrap();
        {
            let upload = Import::lock(source.path(), &owner, &assets, false).unwrap();
            upload.chunk(Role::Provenance, 0, b"provenance\n").unwrap();
            for i in 0..count {
                upload
                    .chunk(Role::Pcm, i * CHUNK_BYTES as u64, &chunk)
                    .unwrap();
            }
            upload.verify().unwrap();
            upload.seal().unwrap();
        }
        begin(destination.path(), &owner, &assets).unwrap();
        for role in [Role::Provenance, Role::Pcm] {
            let mut offset = 0;
            while offset < role.expected(&assets).unwrap().size_bytes {
                let bytes = source_chunk(source.path(), &owner, &assets, role, offset).unwrap();
                assert!(bytes.len() <= CHUNK_BYTES);
                // Reopening the durable transfer on every chunk exercises process-restart offsets.
                let upload = Import::lock(destination.path(), &owner, &assets, false).unwrap();
                upload.chunk(role, offset, &bytes).unwrap();
                offset += bytes.len() as u64;
            }
        }
        assert!(evidence(destination.path(), &owner, &assets).is_err());
        let upload = Import::lock(destination.path(), &owner, &assets, false).unwrap();
        upload.verify().unwrap();
        assert!(upload.seal().unwrap().complete);
        drop(upload);
        source.close().unwrap();
        assert!(evidence(destination.path(), &owner, &assets).is_ok());
        assert_eq!(
            source_chunk(
                destination.path(),
                &owner,
                &assets,
                Role::Pcm,
                8 * CHUNK_BYTES as u64
            )
            .unwrap(),
            chunk
        );
        assert!(crate::video_archive_assets::release(destination.path(), &assets, &owner).unwrap());
        assert!(evidence(destination.path(), &owner, &assets).is_err());
    }
    #[test]
    fn begin_repairs_only_corrupt_complete_staging_and_keeps_valid_seals_immutable() {
        let home = home();
        let owner = owner("repair");
        let assets = assets(b"good");
        begin(home.path(), &owner, &assets).unwrap();
        let upload = Import::lock(home.path(), &owner, &assets, false).unwrap();
        upload.chunk(Role::Provenance, 0, b"bad!").unwrap();
        drop(upload);
        assert_eq!(
            begin(home.path(), &owner, &assets)
                .unwrap()
                .offsets
                .provenance,
            0
        );
        let upload = Import::lock(home.path(), &owner, &assets, false).unwrap();
        upload.chunk(Role::Provenance, 0, b"good").unwrap();
        upload.verify().unwrap();
        upload.seal().unwrap();
        drop(upload);
        assert!(begin(home.path(), &owner, &assets).unwrap().complete);
    }
    #[tokio::test]
    async fn offer_and_checkpoint_require_verified_artifacts_and_repair_after_movie_import() {
        use tower::ServiceExt;
        let home = home();
        let gallery = tempfile::tempdir().unwrap();
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(
            mold_db::MetadataDb::open(&home.path().join("mold.db")).unwrap(),
        ));
        state.config.write().await.output_dir = Some(gallery.path().to_string_lossy().into());
        let request: mold_core::GenerateRequest =
            serde_json::from_value(serde_json::json!({"prompt":"fixture","model":"test","width":64,"height":64,"steps":1,"guidance":1.0})).unwrap();
        let mut metadata =
            mold_core::OutputMetadata::from_generate_request(&request, 1, None, "test");
        let assets = assets(b"rows\n");
        metadata.video_job=Some(serde_json::from_value(serde_json::json!({"id":"c7f8554e-53e7-4bd1-a203-346b72032e31","manifest_sha256":"a".repeat(64),"clip_count":"1","part_count":"1","added_frames":"24","source_frames":"0","total_frames":"24","join_mode":"continuous","artifacts":assets})).unwrap());
        let record = mold_db::GenerationRecord::from_save(
            FsPath::new("unused"),
            "clip.mp4",
            mold_core::OutputFormat::Mp4,
            metadata,
            mold_db::RecordSource::Server,
            1,
        );
        let mut transaction = crate::batch_transaction::GalleryImportTransaction::begin(
            gallery.path(),
            "artifact-fixture",
            0,
            serde_json::json!({"kind":"fixture"}),
            record,
        )
        .unwrap();
        std::fs::write(transaction.staging_path(), b"exact movie").unwrap();
        transaction.seal_staged_file().unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&state.gallery_publication_gate, state.metadata_db.clone())
            .await
            .unwrap();
        let inspect = |state: AppState| async move {
            tokio::task::spawn_blocking(move || {
                serde_json::to_value(
                    crate::gallery_media_transfer::offer_for(&state, "clip.mp4").unwrap(),
                )
                .unwrap()
            })
            .await
            .unwrap()
        };
        let offer = inspect(state.clone()).await;
        assert_eq!(offer["video_artifacts_complete"], false);
        let archive = offer["archive_identity_sha256"].as_str().unwrap();
        let app = router().with_state(state.clone());
        let send = |method: &str, path: String, body: Vec<u8>| {
            axum::http::Request::builder()
                .method(method)
                .uri(path)
                .header("x-mold-destination-instance", state.instance_id.as_str())
                .header("x-mold-gallery-archive-identity", archive)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        };
        let response = app
            .clone()
            .oneshot(send(
                "POST",
                "/api/gallery/video-artifacts/clip.mp4/transfer".into(),
                serde_json::to_vec(
                    &serde_json::json!({"archive_identity_sha256":archive,"artifacts":assets}),
                )
                .unwrap(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let status: Status = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 16384)
                .await
                .unwrap(),
        )
        .unwrap();
        let response = app
            .clone()
            .oneshot(send(
                "PUT",
                format!(
                    "/api/gallery/video-artifacts/clip.mp4/transfer/{}/provenance?offset_bytes=0",
                    status.handle
                ),
                b"rows\n".to_vec(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            inspect(state.clone()).await["video_artifacts_complete"],
            false
        );
        let response = app
            .oneshot(send(
                "POST",
                format!(
                    "/api/gallery/video-artifacts/clip.mp4/transfer/{}/complete",
                    status.handle
                ),
                vec![],
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            inspect(state.clone()).await["video_artifacts_complete"],
            true
        );
        let (_, axum::Json(checkpoint)) = crate::gallery_sync::bounded_checkpoint(
            State(state),
            None,
            None,
            Json(serde_json::from_value(serde_json::json!({"filenames":["clip.mp4"]})).unwrap()),
        )
        .await
        .unwrap();
        assert!(serde_json::to_value(checkpoint).unwrap()["revisions"]["clip.mp4"].is_string());
    }
    #[tokio::test]
    async fn router_refuses_wrong_installation_and_does_not_cache_private_errors() {
        use tower::ServiceExt;
        let state = AppState::for_tests();
        let app = router().with_state(state.clone());
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/gallery/video-artifacts/clip.mp4/provenance?offset_bytes=0")
                    .header("x-mold-source-instance", "replaced-installation")
                    .header("x-mold-gallery-archive-identity", "a".repeat(64))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(response.headers()["cache-control"], "private, no-store");
        let body = axum::body::to_bytes(response.into_body(), 16384)
            .await
            .unwrap();
        assert!(std::str::from_utf8(&body)
            .unwrap()
            .contains("VIDEO_ARTIFACT_FENCE_CHANGED"));
    }
    #[test]
    fn archive_fence_requires_exact_captured_identity() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-gallery-archive-identity",
            HeaderValue::from_static("old"),
        );
        assert!(fence(&headers, "x-mold-gallery-archive-identity", "new").is_err());
        assert!(fence(&HeaderMap::new(), "x-mold-gallery-archive-identity", "new").is_err());
    }
    #[test]
    fn verified_inventory_does_not_rescan_after_more_than_128_movies() {
        let home = home();
        let assets = assets(b"row");
        for index in 0..130 {
            let owner = owner(&format!("movie-{index}"));
            begin(home.path(), &owner, &assets).unwrap();
            let upload = Import::lock(home.path(), &owner, &assets, false).unwrap();
            upload.chunk(Role::Provenance, 0, b"row").unwrap();
            upload.verify().unwrap();
            upload.seal().unwrap();
        }
        let owner = owner("movie-0");
        let path = owned_directory(home.path(), &owner, &assets).unwrap();
        verify_files(&path, &owner, &assets, &mut || {
            anyhow::bail!("unchanged inventory reread payload")
        })
        .unwrap();
        // Same-size corruption changes ctime and must invalidate durable verification.
        std::fs::write(path.join("provenance.jsonl"), b"bad").unwrap();
        assert!(verify_files(&path, &owner, &assets, &mut || Ok(())).is_err());
    }
    #[test]
    fn roles_are_a_strict_file_allowlist() {
        for role in ["../owner.json", "manifest.json", "audio", "provenance:0"] {
            assert!(serde_json::from_value::<Role>(serde_json::json!(role)).is_err());
        }
    }
}
