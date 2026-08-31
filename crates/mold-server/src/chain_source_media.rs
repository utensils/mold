//! Durable encrypted authored inputs for chain jobs.
//!
//! The chain manifest and SQLite projection intentionally contain only the
//! scrubbed request. A small job-local index maps that canonical request to an
//! opaque encrypted queue-media set. Generated stage intermediates never enter
//! this store.

use std::collections::{BTreeMap, BTreeSet};
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
    let media = take_authored_media(&mut request);
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

pub(crate) fn scrub(mut request: ChainRequest) -> ChainRequest {
    for stage in &mut request.stages {
        stage.source_image = None;
    }
    request
}

pub(crate) fn hydrate(
    jobs_root: &Path,
    job_dir: &Path,
    mut request: ChainRequest,
) -> anyhow::Result<ChainRequest> {
    let Some(index) = read_index(job_dir)? else {
        return Ok(request);
    };
    let digest = request_digest(&request)?;
    let Some(media_set) = index.requests.get(&digest) else {
        return Ok(request);
    };
    let decrypted = open_store(jobs_root)?
        .decrypt_to_private_staging(media_set)
        .context("hydrating chain authored source media")?;
    for item in &decrypted.files {
        let Some(stage_idx) = item.role.strip_prefix("stage_source:") else {
            bail!(
                "unexpected chain source-media role '{}'; refusing hydration",
                item.role
            );
        };
        let stage_idx: usize = stage_idx
            .parse()
            .context("invalid chain source-media stage role")?;
        let stage = request
            .stages
            .get_mut(stage_idx)
            .context("chain source-media stage is outside the persisted request")?;
        stage.source_image = Some(
            std::fs::read(&item.path)
                .context("reading authenticated chain source media from private staging")?,
        );
    }
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
    job_id: &str,
    output_dir: &Path,
    request: &ChainRequest,
    gate: &crate::batch_transaction::GalleryPublicationGate,
) -> anyhow::Result<()> {
    let Some(index) = read_index(job_dir)? else {
        return Ok(());
    };
    let digest = request_digest(&scrub(request.clone()))?;
    let Some(media_set) = index.requests.get(&digest) else {
        return Ok(());
    };
    let store = open_store(jobs_root)?;
    let bindings = gate.bind_retained_media_for_job(output_dir, job_id, media_set, |pin_id| {
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

fn take_authored_media(request: &mut ChainRequest) -> Vec<SealMedia> {
    request
        .stages
        .iter_mut()
        .enumerate()
        .filter_map(|(idx, stage)| {
            stage.source_image.take().map(|bytes| {
                SealMedia::bytes(format!("stage_source:{idx}"), "authored-source", bytes)
            })
        })
        .collect()
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
    use super::*;
    use mold_core::chain::{ChainStage, TransitionMode};
    use mold_core::OutputFormat;

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
        };
        ChainRequest {
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
        assert!(hydrate(&jobs, &job, scrubbed).unwrap().stages[0]
            .source_image
            .is_none());
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
    fn startup_reconciliation_retries_failed_replacement_cleanup_without_deleting_live_set() {
        let home = tempfile::tempdir().unwrap();
        let jobs = home.path().join("jobs");
        let job = jobs.join("job-1");
        std::fs::create_dir_all(&job).unwrap();
        let _ = persist_scrubbed(&jobs, &job, "job-1", request()).unwrap();
        let mut replacement = request();
        replacement.stages[0].source_image = Some(vec![9, 9, 9]);
        FAIL_NEXT_DELETE.with(|flag| flag.set(true));
        let scrubbed = persist_scrubbed(&jobs, &job, "job-1", replacement).unwrap();
        assert_eq!(
            open_store(&jobs).unwrap().inspect_owner(OWNER).active.len(),
            2
        );

        FAIL_NEXT_DELETE.with(|flag| flag.set(true));
        assert!(reconcile_orphans(&jobs).is_err());
        assert_eq!(
            open_store(&jobs).unwrap().inspect_owner(OWNER).active.len(),
            2
        );
        assert_eq!(reconcile_orphans(&jobs).unwrap(), 1);
        assert_eq!(
            open_store(&jobs).unwrap().inspect_owner(OWNER).active.len(),
            1
        );
        assert_eq!(
            hydrate(&jobs, &job, scrubbed).unwrap().stages[0]
                .source_image
                .as_deref(),
            Some(&[9, 9, 9][..])
        );
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
