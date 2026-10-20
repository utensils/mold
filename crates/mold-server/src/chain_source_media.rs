//! Durable encrypted authored inputs for chain jobs.
//!
//! The chain manifest and SQLite projection intentionally contain only the
//! scrubbed request. A small job-local index maps that canonical request to an
//! opaque encrypted queue-media set. Generated stage intermediates never enter
//! this store.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context};
use mold_core::chain::ChainRequest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::queue_media_store::{MediaSetRef, QueueMediaStore, SealMedia};

const INDEX_FILE: &str = "source-media.json";
const OWNER: &str = "chain-jobs";

#[cfg(test)]
thread_local! {
    static FAIL_INDEX_PUBLICATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_NEXT_DELETE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct CreateRollback {
    jobs_root: std::path::PathBuf,
    job_dir: std::path::PathBuf,
    armed: bool,
}

impl CreateRollback {
    pub(crate) fn new(jobs_root: &Path, job_dir: &Path) -> Self {
        Self {
            jobs_root: jobs_root.to_path_buf(),
            job_dir: job_dir.to_path_buf(),
            armed: true,
        }
    }

    pub(crate) fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for CreateRollback {
    fn drop(&mut self) {
        if self.armed {
            let _ = release_all(&self.jobs_root, &self.job_dir);
            let _ = std::fs::remove_file(self.job_dir.join(INDEX_FILE));
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SourceMediaIndex {
    version: u8,
    requests: BTreeMap<String, MediaSetRef>,
}

pub(crate) fn persist_scrubbed(
    jobs_root: &Path,
    job_dir: &Path,
    job_id: &str,
    mut request: ChainRequest,
) -> anyhow::Result<ChainRequest> {
    // Client-provided identities never authorize bytes or survive an input removal.
    for stage in &mut request.stages {
        stage.authored_media_sha256 = None;
    }
    stamp_authored_media_identity(&mut request)?;
    let retained_tail = crate::chain_continuation::retained_frames(job_dir, &request)?;
    let mut total_bytes = retained_tail
        .iter()
        .map(|bytes| bytes.len() as u64)
        .sum::<u64>();
    for stage in &request.stages {
        total_bytes = total_bytes
            .checked_add(
                stage
                    .source_image
                    .as_ref()
                    .map_or(0, |bytes| bytes.len() as u64),
            )
            .context("chain media size overflow")?;
        for keyframe in stage.keyframes.iter().flatten() {
            total_bytes = total_bytes
                .checked_add(keyframe.image.len() as u64)
                .context("chain media size overflow")?;
        }
        for reference in stage.generation_references.iter().flatten() {
            if let mold_core::GenerationReferenceAuthority::Inline { data } = reference.media() {
                total_bytes = total_bytes
                    .checked_add(data.len() as u64)
                    .context("chain media size overflow")?;
            }
        }
    }
    let mut media = take_authored_media(&mut request)?;
    for (index, bytes) in retained_tail.into_iter().enumerate() {
        media.push(SealMedia::bytes(
            format!("chain_continuation_tail:{index}"),
            "continuation-tail",
            bytes,
        ));
    }
    if request.continue_from.is_some() && (media.len() > 64 || total_bytes > 512 * 1024 * 1024) {
        bail!("linked part retained conditioning exceeds 64 members or 512 MiB");
    }
    if media.is_empty() {
        return Ok(request);
    }
    let digest = request_digest(&request)?;
    let store = open_store(jobs_root)?;
    // Queue-media deliberately permits one active bundle per storage job.
    // A chain can have multiple durable authored revisions, so each revision
    // receives a fresh store-local identity while the sidecar remains the
    // chain authority that relates it to `job_id`.
    let storage_job_id = format!("{job_id}-{}", uuid::Uuid::new_v4().simple());
    let media_set = store
        .seal(OWNER, &storage_job_id, media)
        .context("sealing chain authored source media")?;
    let mut index = read_index(job_dir)?.unwrap_or_default();
    index.version = 1;
    let replaced = index.requests.insert(digest, media_set.clone());
    if let Err(error) = write_index(job_dir, &index) {
        let _ = delete_set(&store, &media_set, false);
        return Err(error);
    }
    if let Some(replaced) = replaced {
        if replaced != media_set {
            let _ = delete_set(&store, &replaced, false);
        }
    }
    Ok(request)
}

pub(crate) fn scrub(request: ChainRequest) -> ChainRequest {
    // Only authenticated hydrated requests enter this infallible projection.
    try_scrub(request).expect("chain media identity serialization")
}

pub(crate) fn try_scrub(mut request: ChainRequest) -> anyhow::Result<ChainRequest> {
    stamp_authored_media_identity(&mut request)?;
    for stage in &mut request.stages {
        stage.source_image = None;
        if let Some(keyframes) = &mut stage.keyframes {
            for keyframe in keyframes {
                keyframe.image.clear();
            }
        }
        if let Some(references) = &mut stage.generation_references {
            for reference in references {
                *reference_media_mut(reference) =
                    mold_core::GenerationReferenceAuthority::Descriptor;
            }
        }
    }
    Ok(request)
}

fn read_authored_member(path: &Path, remaining: &mut u64) -> anyhow::Result<Vec<u8>> {
    let file = mold_core::secure_file::open_regular_file_no_follow(path)
        .context("opening authenticated chain source media")?;
    let mut bytes = Vec::new();
    file.take(*remaining + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > *remaining {
        bail!("authenticated chain source media exceeds the per-part budget");
    }
    *remaining -= bytes.len() as u64;
    Ok(bytes)
}

pub(crate) fn hydrate(
    jobs_root: &Path,
    job_dir: &Path,
    mut request: ChainRequest,
) -> anyhow::Result<ChainRequest> {
    let Some(index) = read_index(job_dir)? else {
        if request.continue_from.is_some() && request.motion_tail_frames > 0 {
            bail!("linked part has no sealed continuation recovery authority");
        }
        ensure_hydrated(&request)?;
        return Ok(request);
    };
    let Some(media_set) = media_revision(&index, &request)? else {
        if request.continue_from.is_some() && request.motion_tail_frames > 0 {
            bail!("linked part media revision is missing");
        }
        ensure_hydrated(&request)?;
        return Ok(request);
    };
    let decrypted = open_store(jobs_root)?
        .decrypt_to_private_staging(media_set)
        .context("hydrating chain authored source media")?;
    let mut seen = BTreeSet::new();
    let retained_tail = crate::chain_continuation::retained_frames(job_dir, &request)?;
    let mut tail_members = 0usize;
    let mut remaining = mold_core::chain::MAX_CHAIN_INLINE_MEDIA_BYTES as u64;
    for item in &decrypted.files {
        if !seen.insert(&item.role) {
            bail!("duplicate chain source-media role");
        }
        if let Some(position) = item.role.strip_prefix("chain_continuation_tail:") {
            let index: usize = position
                .parse()
                .context("invalid continuation tail index")?;
            let expected = retained_tail
                .get(index)
                .context("unexpected sealed continuation tail member")?;
            let bytes = read_authored_member(&item.path, &mut remaining)?;
            if index.to_string() != position || bytes != *expected {
                bail!("sealed continuation tail differs from successor ownership");
            }
            tail_members += 1;
            continue;
        }
        let parts = item.role.split(':').collect::<Vec<_>>();
        let stage_idx: usize = parts
            .get(1)
            .context("missing source-media stage")?
            .parse()
            .context("invalid source-media stage")?;
        let stage = request
            .stages
            .get_mut(stage_idx)
            .context("source-media stage outside request")?;
        let bytes = read_authored_member(&item.path, &mut remaining)?;
        match parts.as_slice() {
            ["stage_source", _] => stage.source_image = Some(bytes),
            ["stage_keyframe", _, position] => {
                let position: usize = position.parse().context("invalid keyframe position")?;
                let keyframe = stage
                    .keyframes
                    .as_mut()
                    .and_then(|frames| frames.get_mut(position))
                    .context("sealed keyframe outside persisted request")?;
                keyframe.image = bytes;
            }
            ["stage_reference", _, position] => {
                let position: usize = position.parse().context("invalid reference position")?;
                let reference = stage
                    .generation_references
                    .as_mut()
                    .and_then(|refs| refs.get_mut(position))
                    .context("sealed reference outside persisted request")?;
                let digest = format!("{:x}", Sha256::digest(&bytes));
                if reference.content_sha256().as_deref() != Some(digest.as_str()) {
                    bail!("sealed reference differs from persisted digest");
                }
                *reference_media_mut(reference) =
                    mold_core::GenerationReferenceAuthority::Inline { data: bytes };
            }
            _ => bail!("unexpected chain source-media role; refusing hydration"),
        }
    }
    if tail_members != retained_tail.len() {
        bail!("continuation conditioning is missing from sealed recovery authority");
    }
    ensure_hydrated(&request)?;
    Ok(request)
}

pub(crate) fn release_all(jobs_root: &Path, job_dir: &Path) -> anyhow::Result<()> {
    let Some(index) = read_index(job_dir)? else {
        return Ok(());
    };
    let store = open_store(jobs_root)?;
    for media_set in index.requests.values() {
        match delete_set(&store, media_set, false) {
            Ok(()) | Err(crate::queue_media_store::QueueMediaError::NotFound) => {}
            Err(error) => return Err(error).context("releasing chain authored source media"),
        }
    }
    Ok(())
}

/// Reclaim only chain-owned encrypted sets that no durable chain sidecar
/// references. Any unreadable sidecar or unrecognized owner entry fails
/// closed before mutation; ordinary queue and gallery owners are never
/// enumerated.
pub(crate) fn reconcile_orphans(jobs_root: &Path) -> anyhow::Result<usize> {
    let mut live = BTreeSet::new();
    for entry in std::fs::read_dir(jobs_root)
        .with_context(|| format!("enumerating chain jobs root '{}'", jobs_root.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        if let Some(index) = read_index(&entry.path())? {
            live.extend(index.requests.into_values());
        }
    }

    let store = open_store(jobs_root)?;
    let inspection = store.inspect_owner(OWNER);
    if !inspection.unrecognized.is_empty() {
        bail!(
            "chain source-media reconciliation found {} unrecognized store entries",
            inspection.unrecognized.len()
        );
    }
    let mut deleted = 0;
    for media_set in inspection.active.iter().chain(&inspection.retired) {
        if !live.contains(media_set) {
            match delete_set(&store, media_set, false) {
                Ok(()) | Err(crate::queue_media_store::QueueMediaError::NotFound) => deleted += 1,
                Err(error) => return Err(error).context("releasing orphaned chain source media"),
            }
        }
    }
    for media_set in &inspection.staging {
        if !live.contains(media_set) {
            match delete_set(&store, media_set, true) {
                Ok(()) | Err(crate::queue_media_store::QueueMediaError::NotFound) => deleted += 1,
                Err(error) => {
                    return Err(error).context("releasing interrupted chain source-media staging")
                }
            }
        }
    }
    Ok(deleted)
}

pub(crate) fn handoff_current_to_gallery(
    db: &mold_db::MetadataDb,
    jobs_root: &Path,
    job_dir: &Path,
    gallery_identity: &crate::batch_transaction::ArchivedChildIdentity,
    output_dir: &Path,
    request: &ChainRequest,
    gate: &crate::batch_transaction::GalleryPublicationGate,
) -> anyhow::Result<()> {
    if !(request.continue_from.is_some() && request.motion_tail_frames > 0)
        && request.stages.iter().all(|stage| {
            stage.source_image.is_none()
                && stage.keyframes.is_none()
                && stage.generation_references.is_none()
                && stage.authored_media_sha256.is_none()
        })
    {
        return Ok(());
    }
    let Some(index) = read_index(job_dir)? else {
        return Ok(());
    };
    // Retake overlays preserve the canonical sealed input revision.
    let media_set = if let Some(media_set) = media_revision(&index, &scrub(request.clone()))? {
        media_set
    } else {
        let manifest = mold_core::chain_job::ChainJobManifest::read_from_dir(job_dir)
            .context("reading canonical media authority during gallery handoff")?;
        media_revision(&index, &scrub(manifest.request()?))?
            .context("current chain media revision has no sealed recovery authority")?
    };
    let store = open_store(jobs_root)?;
    let bindings =
        gate.bind_retained_media_for_output(output_dir, gallery_identity, media_set, |pin_id| {
            store
                .pin_for_gallery_item(media_set, pin_id)
                .map(|_| ())
                .map_err(Into::into)
        })?;
    let canonical = std::fs::canonicalize(output_dir).unwrap_or_else(|_| output_dir.into());
    for (filename, _) in bindings {
        let retained = gate
            .retained_media_for_item(&canonical, &filename)?
            .map(|(_, pins)| pins)
            .unwrap_or_default();
        let projection = retained
            .into_iter()
            .map(|binding| mold_db::gallery_media::GalleryMediaBinding {
                output_dir: canonical.to_string_lossy().into_owned(),
                filename: filename.clone(),
                pin_id: binding.pin_id,
                media_set_id: binding.media_set.set_id,
                owner_uuid: binding.media_set.owner_id,
                job_id: binding.media_set.job_id,
            })
            .collect::<Vec<_>>();
        mold_db::gallery_media::replace_for_item(
            db,
            &canonical.to_string_lossy(),
            &filename,
            &projection,
        )?;
    }
    Ok(())
}

fn stage_has_payload(stage: &mold_core::chain::ChainStage) -> bool {
    stage.source_image.is_some()
        || stage
            .keyframes
            .as_ref()
            .is_some_and(|frames| frames.iter().any(|frame| !frame.image.is_empty()))
        || stage.generation_references.as_ref().is_some_and(|refs| {
            refs.iter().any(|reference| {
                matches!(
                    reference.media(),
                    mold_core::GenerationReferenceAuthority::Inline { .. }
                )
            })
        })
}

/// Content identity is computed before any payload is removed. The previous
/// set remains addressable if publishing the next manifest fails or crashes.
fn stamp_authored_media_identity(request: &mut ChainRequest) -> anyhow::Result<()> {
    for stage in &mut request.stages {
        let has_payload = stage_has_payload(stage);
        if !has_payload {
            continue;
        }
        if let Some(references) = &mut stage.generation_references {
            for reference in references {
                if let mold_core::GenerationReferenceAuthority::Inline { data } = reference.media()
                {
                    let digest = format!("{:x}", Sha256::digest(data));
                    if reference
                        .provenance()
                        .sha256
                        .as_ref()
                        .is_some_and(|expected| !expected.eq_ignore_ascii_case(&digest))
                    {
                        bail!("chain reference payload differs from supplied digest");
                    }
                    reference_provenance_mut(reference).sha256 = Some(digest);
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"mold.chain.authored-media.v1\0");
        for (label, bytes) in [
            ("source", serde_json::to_vec(&stage.source_image)?),
            ("keyframes", serde_json::to_vec(&stage.keyframes)?),
            (
                "references",
                serde_json::to_vec(&stage.generation_references)?,
            ),
        ] {
            hash.update((label.len() as u64).to_le_bytes());
            hash.update(label.as_bytes());
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        stage.authored_media_sha256 = Some(format!("{:x}", hash.finalize()));
    }
    Ok(())
}

fn reference_media_mut(
    reference: &mut mold_core::GenerationReference,
) -> &mut mold_core::GenerationReferenceAuthority {
    use mold_core::GenerationReference::*;
    match reference {
        Image { media, .. }
        | NamedImage { media, .. }
        | Video { media, .. }
        | Audio { media, .. }
        | Mesh { media, .. } => media,
    }
}

fn reference_provenance_mut(
    reference: &mut mold_core::GenerationReference,
) -> &mut mold_core::GenerationReferenceProvenance {
    use mold_core::GenerationReference::*;
    match reference {
        Image { provenance, .. }
        | NamedImage { provenance, .. }
        | Video { provenance, .. }
        | Audio { provenance, .. }
        | Mesh { provenance, .. } => provenance,
    }
}

fn ensure_hydrated(request: &ChainRequest) -> anyhow::Result<()> {
    let mut observed = request.clone();
    stamp_authored_media_identity(&mut observed)?;
    for (stage, restored) in request.stages.iter().zip(&observed.stages) {
        if stage.authored_media_sha256.is_some()
            && stage.authored_media_sha256 != restored.authored_media_sha256
        {
            bail!("chain source media differs from its persisted content authority");
        }
    }
    for stage in &request.stages {
        if stage.authored_media_sha256.is_some() && !stage_has_payload(stage) {
            bail!("chain source media is missing its durable recovery authority");
        }
        if stage
            .keyframes
            .as_ref()
            .is_some_and(|frames| frames.iter().any(|frame| frame.image.is_empty()))
        {
            bail!("chain boundary media is missing its durable recovery authority");
        }
        if stage.generation_references.as_ref().is_some_and(|refs| {
            refs.iter().any(|r| {
                !matches!(
                    r.media(),
                    mold_core::GenerationReferenceAuthority::Inline { .. }
                )
            })
        }) {
            bail!("chain reference media is missing its durable recovery authority");
        }
    }
    Ok(())
}

fn take_authored_media(request: &mut ChainRequest) -> anyhow::Result<Vec<SealMedia>> {
    let mut media = Vec::new();
    for (idx, stage) in request.stages.iter_mut().enumerate() {
        if let Some(bytes) = stage.source_image.take() {
            media.push(SealMedia::bytes(
                format!("stage_source:{idx}"),
                "authored-source",
                bytes,
            ));
        }
        if let Some(keyframes) = &mut stage.keyframes {
            for (position, keyframe) in keyframes.iter_mut().enumerate() {
                if keyframe.image.is_empty() {
                    bail!("cannot seal empty chain keyframe");
                }
                media.push(SealMedia::bytes(
                    format!("stage_keyframe:{idx}:{position}"),
                    "authored-keyframe",
                    std::mem::take(&mut keyframe.image),
                ));
            }
        }
        if let Some(references) = &mut stage.generation_references {
            for (position, reference) in references.iter_mut().enumerate() {
                let authority = std::mem::replace(
                    reference_media_mut(reference),
                    mold_core::GenerationReferenceAuthority::Descriptor,
                );
                let mold_core::GenerationReferenceAuthority::Inline { data } = authority else {
                    bail!("chain references require inline media before durable sealing");
                };
                let digest = format!("{:x}", Sha256::digest(&data));
                if reference
                    .provenance()
                    .sha256
                    .as_ref()
                    .is_some_and(|expected| !expected.eq_ignore_ascii_case(&digest))
                {
                    bail!("chain reference payload differs from supplied digest");
                }
                reference_provenance_mut(reference).sha256 = Some(digest);
                media.push(SealMedia::bytes(
                    format!("stage_reference:{idx}:{position}"),
                    "authored-reference",
                    data,
                ));
            }
        }
    }
    Ok(media)
}

/// Older creation sealed before resolving an omitted base seed. Accept only
/// that exact historical digest: every authored byte hash, setting, prompt and
/// predecessor binding remains identical. Never scan for a similar revision.
fn media_revision<'a>(
    index: &'a SourceMediaIndex,
    request: &ChainRequest,
) -> anyhow::Result<Option<&'a MediaSetRef>> {
    if let Some(media) = index.requests.get(&request_digest(request)?) {
        return Ok(Some(media));
    }
    if request.seed.is_some() {
        let mut legacy = request.clone();
        legacy.seed = None;
        return Ok(index.requests.get(&request_digest(&legacy)?));
    }
    Ok(None)
}

fn request_digest(request: &ChainRequest) -> anyhow::Result<String> {
    let canonical = serde_json::to_vec(request)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn open_store(jobs_root: &Path) -> anyhow::Result<QueueMediaStore> {
    let mold_home = jobs_root
        .parent()
        .context("chain jobs root has no MOLD_HOME parent")?;
    Ok(QueueMediaStore::open(mold_home)?.store)
}

fn delete_set(
    store: &QueueMediaStore,
    media_set: &MediaSetRef,
    staging: bool,
) -> Result<(), crate::queue_media_store::QueueMediaError> {
    #[cfg(test)]
    if FAIL_NEXT_DELETE.with(|flag| flag.replace(false)) {
        return Err(std::io::Error::other("injected chain source-media delete failure").into());
    }
    if staging {
        store.delete_staging(media_set)
    } else {
        store.delete(media_set)
    }
}

fn read_index(job_dir: &Path) -> anyhow::Result<Option<SourceMediaIndex>> {
    let path = job_dir.join(INDEX_FILE);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
            format!("reading chain source-media index '{}'", path.display())
        })?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading '{}'", path.display())),
    }
}

fn write_index(job_dir: &Path, index: &SourceMediaIndex) -> anyhow::Result<()> {
    use std::io::Write as _;

    let path = job_dir.join(INDEX_FILE);
    let temp = job_dir.join(format!(".{INDEX_FILE}.{}.tmp", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec(index)?;
    let mut file =
        std::fs::File::create(&temp).with_context(|| format!("creating '{}'", temp.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("writing '{}'", temp.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing '{}'", temp.display()))?;
    drop(file);
    #[cfg(test)]
    if FAIL_INDEX_PUBLICATION.with(|flag| flag.get()) {
        let _ = std::fs::remove_file(&temp);
        anyhow::bail!("injected chain source-media index publication failure");
    }
    std::fs::rename(&temp, &path).with_context(|| format!("publishing '{}'", path.display()))?;
    crate::dir_sync::sync_directory(job_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn authenticated_member_reads_bound_aggregate_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("member");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let mut remaining = 5;
        assert_eq!(
            super::read_authored_member(&path, &mut remaining).unwrap(),
            [1, 2, 3]
        );
        assert_eq!(remaining, 2);
        assert!(super::read_authored_member(&path, &mut remaining).is_err());
    }
    use super::*;
    use mold_core::chain::{ChainStage, TransitionMode};
    use mold_core::OutputFormat;

    #[test]
    fn legacy_random_seed_seal_recovers_only_the_identical_authored_revision() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("jobs");
        let job = root.join("legacy-random-seed");
        std::fs::create_dir_all(&job).unwrap();
        let mut original = request();
        original.seed = None;
        let sealed = persist_scrubbed(&root, &job, "legacy-random-seed", original.clone()).unwrap();
        let mut persisted = sealed;
        persisted.materialize_seed();
        assert_eq!(
            hydrate(&root, &job, persisted.clone()).unwrap().stages[0].source_image,
            original.stages[0].source_image
        );
        let mut changed = persisted.clone();
        changed.stages[0].prompt.push_str(" changed");
        assert!(hydrate(&root, &job, changed).is_err());
        let mut changed = persisted.clone();
        changed.stages[0].authored_media_sha256 = Some("different bytes".into());
        assert!(hydrate(&root, &job, changed).is_err());
        let mut changed = persisted;
        changed.continue_from = Some(mold_core::ChainContinuation {
            job_id: "other-parent".into(),
            finalize_take: 2,
        });
        assert!(hydrate(&root, &job, changed).is_err());
    }

    fn request() -> ChainRequest {
        let stage = |prompt: &str, source_image| ChainStage {
            prompt: prompt.into(),
            frames: 9,
            source_image,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
            generation_references: None,
            keyframes: None,
            authored_media_sha256: None,
        };
        ChainRequest {
            source_fit: None,
            continue_from: None,
            offload: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![
                stage("one", Some(vec![1, 2, 3])),
                stage("two", Some(vec![4, 5])),
            ],
            motion_tail_frames: 1,
            width: 64,
            height: 64,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            ephemeral: false,
            placement: None,
            title: None,
            tags: None,
            collection: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    #[test]
    fn retry_scrubbing_refuses_mismatched_reference_digest_without_panicking() {
        let mut request = request();
        let mut reference = mold_core::GenerationReference::Image {
            media: mold_core::GenerationReferenceAuthority::Inline {
                data: vec![1, 2, 3],
            },
            provenance: Default::default(),
            mime_type: "image/png".into(),
            width: 16,
            height: 16,
        };
        reference_provenance_mut(&mut reference).sha256 = Some("0".repeat(64));
        request.stages[0].generation_references = Some(vec![reference]);
        assert!(try_scrub(request).is_err());
    }

    #[test]
    fn keyframes_and_ordered_references_are_sealed_and_recover_without_upload_authority() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-media");
        std::fs::create_dir_all(&job).unwrap();
        let mut request = request();
        request.stages[0].keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 8,
            image: b"secret endpoint".to_vec(),
            name: Some("end.png".into()),
        }]);
        request.stages[1].generation_references =
            Some(vec![mold_core::GenerationReference::Image {
                media: mold_core::GenerationReferenceAuthority::Inline {
                    data: b"secret reference".to_vec(),
                },
                provenance: Default::default(),
                mime_type: "image/png".into(),
                width: 16,
                height: 16,
            }]);
        let scrubbed = persist_scrubbed(&jobs, &job, "job-media", request).unwrap();
        let json = serde_json::to_string(&scrubbed).unwrap();
        assert!(!json.contains("secret"));
        assert!(scrubbed.stages[0].keyframes.as_ref().unwrap()[0]
            .image
            .is_empty());
        assert!(matches!(
            scrubbed.stages[1].generation_references.as_ref().unwrap()[0].media(),
            mold_core::GenerationReferenceAuthority::Descriptor
        ));
        let restored = hydrate(&jobs, &job, scrubbed.clone()).unwrap();
        assert_eq!(
            restored.stages[0].keyframes.as_ref().unwrap()[0].image,
            b"secret endpoint"
        );
        assert!(
            matches!(restored.stages[1].generation_references.as_ref().unwrap()[0].media(), mold_core::GenerationReferenceAuthority::Inline { data } if data == b"secret reference")
        );
        // Candidate sealing precedes manifest publication. If publication
        // crashes, the old request must still hydrate its old endpoint bytes.
        let mut edited = restored.clone();
        edited.stages[0].keyframes.as_mut().unwrap()[0].image = b"replacement endpoint".to_vec();
        let new_revision = persist_scrubbed(&jobs, &job, "job-media", edited).unwrap();
        assert_ne!(
            new_revision.stages[0].authored_media_sha256,
            scrubbed.stages[0].authored_media_sha256
        );
        assert_eq!(
            hydrate(&jobs, &job, scrubbed.clone()).unwrap().stages[0]
                .keyframes
                .as_ref()
                .unwrap()[0]
                .image,
            b"secret endpoint"
        );
        assert_eq!(
            hydrate(&jobs, &job, new_revision).unwrap().stages[0]
                .keyframes
                .as_ref()
                .unwrap()[0]
                .image,
            b"replacement endpoint"
        );
        std::fs::remove_file(job.join(INDEX_FILE)).unwrap();
        assert!(hydrate(&jobs, &job, scrubbed).is_err());
    }

    #[test]
    fn encrypted_round_trip_scrubs_durable_request_and_preserves_stage_roles() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let scrubbed = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        assert!(scrubbed
            .stages
            .iter()
            .all(|stage| stage.source_image.is_none()));
        let persisted = std::fs::read_to_string(job.join(INDEX_FILE)).unwrap();
        assert!(!persisted.contains("1,2,3"));
        let hydrated = hydrate(&jobs, &job, scrubbed).unwrap();
        assert_eq!(
            hydrated.stages[0].source_image.as_deref(),
            Some(&[1, 2, 3][..])
        );
        assert_eq!(
            hydrated.stages[1].source_image.as_deref(),
            Some(&[4, 5][..])
        );
    }

    #[test]
    fn missing_sidecar_is_legacy_and_does_not_invent_media_from_provenance() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("legacy");
        std::fs::create_dir_all(&job).unwrap();
        let mut legacy = request();
        for stage in &mut legacy.stages {
            stage.source_image = None;
        }
        assert_eq!(hydrate(&jobs, &job, legacy.clone()).unwrap(), legacy);
    }

    #[test]
    fn release_removes_the_encrypted_bundle_idempotently() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let scrubbed = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        release_all(&jobs, &job).unwrap();
        release_all(&jobs, &job).unwrap();
        assert!(hydrate(&jobs, &job, scrubbed).is_err());
    }

    #[test]
    fn create_rollback_releases_a_sealed_set_after_later_failure() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let scrubbed = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        {
            let _rollback = CreateRollback::new(&jobs, &job);
        }
        assert!(!job.join(INDEX_FILE).exists());
        assert!(hydrate(&jobs, &job, scrubbed).is_err());
    }

    #[test]
    fn failed_replacement_keeps_the_durable_old_index_and_bundle() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let scrubbed = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        let mut replacement = request();
        replacement.stages[0].source_image = Some(vec![9, 9, 9]);
        FAIL_INDEX_PUBLICATION.with(|flag| flag.set(true));
        let result = persist_scrubbed(&jobs, &job, "job-1", replacement);
        FAIL_INDEX_PUBLICATION.with(|flag| flag.set(false));
        assert!(result.is_err());
        let hydrated = hydrate(&jobs, &job, scrubbed).unwrap();
        assert_eq!(
            hydrated.stages[0].source_image.as_deref(),
            Some(&[1, 2, 3][..])
        );
    }

    #[test]
    fn media_only_revisions_keep_old_authority_until_job_gc() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let old = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        let mut replacement = request();
        replacement.stages[0].source_image = Some(vec![9, 9, 9]);
        replacement.stages[0].authored_media_sha256 = Some("caller-forged".into());
        let new = persist_scrubbed(&jobs, &job, "job-1", replacement).unwrap();
        assert_ne!(
            old.stages[0].authored_media_sha256,
            new.stages[0].authored_media_sha256
        );
        assert_eq!(
            open_store(&jobs).unwrap().inspect_owner(OWNER).active.len(),
            2
        );
        assert_eq!(reconcile_orphans(&jobs).unwrap(), 0);
        assert_eq!(
            hydrate(&jobs, &job, old).unwrap().stages[0]
                .source_image
                .as_deref(),
            Some(&[1, 2, 3][..])
        );
        assert_eq!(
            hydrate(&jobs, &job, new).unwrap().stages[0]
                .source_image
                .as_deref(),
            Some(&[9, 9, 9][..])
        );
        release_all(&jobs, &job).unwrap();
        assert!(open_store(&jobs)
            .unwrap()
            .inspect_owner(OWNER)
            .active
            .is_empty());
    }

    #[test]
    fn startup_reconciliation_collects_failed_new_bundle_cleanup_with_no_live_sidecar() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        FAIL_INDEX_PUBLICATION.with(|flag| flag.set(true));
        FAIL_NEXT_DELETE.with(|flag| flag.set(true));
        assert!(persist_scrubbed(&jobs, &job, "job-1", request()).is_err());
        FAIL_INDEX_PUBLICATION.with(|flag| flag.set(false));
        assert_eq!(
            open_store(&jobs).unwrap().inspect_owner(OWNER).active.len(),
            1
        );
        assert_eq!(reconcile_orphans(&jobs).unwrap(), 1);
        assert!(open_store(&jobs)
            .unwrap()
            .inspect_owner(OWNER)
            .active
            .is_empty());
    }
}
