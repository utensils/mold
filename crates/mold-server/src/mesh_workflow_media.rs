//! Encrypted authored inputs owned by a durable mesh workflow.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context};
use mold_core::{GenerateRequest, GenerationReference, GenerationReferenceAuthority};
use serde::{Deserialize, Serialize};

use crate::queue_media::{self, ProcessPrivateAuthorities};
use crate::queue_media_store::{MediaSetRef, QueueMediaOperationFingerprint, QueueMediaStore};

const INDEX_FILE: &str = "source-media.json";
const OWNER: &str = "mesh-workflows";

pub(crate) struct CreateRollback {
    root: std::path::PathBuf,
    workflow_dir: std::path::PathBuf,
    armed: bool,
}

impl CreateRollback {
    pub(crate) fn new(root: &Path, workflow_dir: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            workflow_dir: workflow_dir.to_path_buf(),
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
            let _ = release_all(&self.root, &self.workflow_dir);
            let _ = std::fs::remove_dir_all(&self.workflow_dir);
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MediaIndex {
    version: u8,
    requests: BTreeMap<String, StoredRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredRequest {
    scope: String,
    request_json: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    media_set: Option<MediaSetRef>,
}

/// Seal one already-resolved request and return its payload-free form for the
/// portable workflow manifest. Upload handles and server paths must have been
/// resolved before this boundary.
pub(crate) fn persist_request(
    workflows_root: &Path,
    workflow_dir: &Path,
    workflow_id: &str,
    label: &str,
    request: GenerateRequest,
    staged: Option<&crate::reference_uploads::StagedReferences>,
) -> anyhow::Result<GenerateRequest> {
    let scope = format!("{workflow_id}-{label}");
    let extracted = queue_media::extract_request_media(
        scope.clone(),
        request,
        &ProcessPrivateAuthorities::none(),
        staged,
    )
    .context("extracting mesh workflow request media")?;
    let projection = queue_media::project_request_media(extracted.media())
        .context("projecting mesh workflow request media")?;
    let (request_json, media) = extracted.into_parts();
    let seal_inputs =
        queue_media::into_seal_media(media).context("preparing mesh workflow request media")?;
    let media_set = if seal_inputs.is_empty() {
        None
    } else {
        Some(
            open_store(workflows_root)?
                .seal_v2_with_operation_fingerprint(
                    OWNER,
                    &scope,
                    &QueueMediaOperationFingerprint::sha256_v1(request_json.as_bytes()),
                    &projection,
                    seal_inputs,
                )
                .context("sealing mesh workflow request media")?,
        )
    };
    let mut index = read_index(workflow_dir)?.unwrap_or_default();
    index.version = 1;
    if index
        .requests
        .insert(
            label.to_string(),
            StoredRequest {
                scope,
                request_json: request_json.clone(),
                media_set,
            },
        )
        .is_some()
    {
        bail!("mesh workflow request label '{label}' was persisted twice");
    }
    write_index(workflow_dir, &index)?;
    serde_json::from_str(&request_json).context("reading scrubbed mesh workflow request")
}

/// Recover a request as ordinary inline authority so the normal durable
/// generation admission boundary can validate and seal its child job.
pub(crate) fn hydrate_for_admission(
    workflows_root: &Path,
    workflow_dir: &Path,
    label: &str,
) -> anyhow::Result<GenerateRequest> {
    let index = read_index(workflow_dir)?.context("mesh workflow media index is missing")?;
    let stored = index
        .requests
        .get(label)
        .with_context(|| format!("mesh workflow request '{label}' is missing"))?;
    let Some(media_set) = stored.media_set.as_ref() else {
        return serde_json::from_str(&stored.request_json)
            .context("reading payload-free mesh workflow request");
    };
    let store = open_store(workflows_root)?;
    let mut decrypted = store
        .decrypt_mixed(media_set)
        .context("decrypting mesh workflow request media")?;
    let media = queue_media::decrypted_media_into_opaque(&stored.scope, &mut decrypted)
        .context("binding mesh workflow request media")?;
    let (mut request, reference_paths) =
        queue_media::rehydrate_request_media(&stored.scope, &stored.request_json, media)
            .context("rehydrating mesh workflow request")?;
    let references = request.references.as_deref_mut().unwrap_or(&mut []);
    if references.len() != reference_paths.len() {
        bail!("mesh workflow reference descriptors do not match retained media");
    }
    for (reference, path) in references.iter_mut().zip(reference_paths) {
        let bytes = std::fs::read(path).context("reading retained workflow reference")?;
        *reference_media_mut(reference) = GenerationReferenceAuthority::Inline { data: bytes };
    }
    Ok(request)
}

pub(crate) fn release_all(workflows_root: &Path, workflow_dir: &Path) -> anyhow::Result<()> {
    let Some(index) = read_index(workflow_dir)? else {
        return Ok(());
    };
    let store = open_store(workflows_root)?;
    for stored in index.requests.values() {
        if let Some(media_set) = stored.media_set.as_ref() {
            match store.delete(media_set) {
                Ok(()) | Err(crate::queue_media_store::QueueMediaError::NotFound) => {}
                Err(error) => return Err(error).context("releasing mesh workflow media"),
            }
        }
    }
    Ok(())
}

pub(crate) fn purge_claimed(workflows_root: &Path, workflow_dir: &Path) -> anyhow::Result<()> {
    if workflow_dir.parent() != Some(workflows_root) {
        bail!("mesh workflow directory is outside its storage root");
    }
    release_all(workflows_root, workflow_dir)?;
    match std::fs::remove_dir_all(workflow_dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("removing claimed mesh workflow directory"),
    }
}

fn reference_media_mut(reference: &mut GenerationReference) -> &mut GenerationReferenceAuthority {
    match reference {
        GenerationReference::Image { media, .. }
        | GenerationReference::NamedImage { media, .. }
        | GenerationReference::Video { media, .. }
        | GenerationReference::Audio { media, .. }
        | GenerationReference::Mesh { media, .. } => media,
    }
}

fn open_store(workflows_root: &Path) -> anyhow::Result<QueueMediaStore> {
    let mold_home = workflows_root
        .parent()
        .context("mesh workflow root has no MOLD_HOME parent")?;
    Ok(QueueMediaStore::open(mold_home)?.store)
}

fn read_index(workflow_dir: &Path) -> anyhow::Result<Option<MediaIndex>> {
    let path = workflow_dir.join(INDEX_FILE);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
            format!("reading mesh workflow media index '{}'", path.display())
        })?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading '{}'", path.display())),
    }
}

fn write_index(workflow_dir: &Path, index: &MediaIndex) -> anyhow::Result<()> {
    use std::io::Write as _;
    std::fs::create_dir_all(workflow_dir)?;
    let path = workflow_dir.join(INDEX_FILE);
    let temporary = workflow_dir.join(format!(".{INDEX_FILE}.{}.tmp", uuid::Uuid::new_v4()));
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec(index)?)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, &path)?;
    crate::dir_sync::sync_directory(workflow_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(source_image: Option<Vec<u8>>) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "hunyuan3d-2.0:fp16",
            "width": 0,
            "height": 0,
            "steps": 5,
            "guidance": 5.0,
            "seed": 1,
            "output_format": "glb"
        }))
        .unwrap();
        request.source_image = source_image;
        request
    }

    #[test]
    fn authored_bytes_are_encrypted_outside_the_index_and_rehydrate_for_admission() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("mesh-workflows");
        let dir = root.join("workflow");
        std::fs::create_dir_all(&dir).unwrap();
        let scrubbed = persist_request(
            &root,
            &dir,
            "workflow",
            "texture",
            request(Some(vec![1, 2, 3, 4])),
            None,
        )
        .unwrap();
        assert!(scrubbed.source_image.is_none());
        let index = std::fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert!(!index.contains("AQIDBA"));
        let hydrated = hydrate_for_admission(&root, &dir, "texture").unwrap();
        assert_eq!(hydrated.source_image, Some(vec![1, 2, 3, 4]));
        release_all(&root, &dir).unwrap();
    }
}
