//! Explicit export of one held request, with its original durable media.
//! This is never used by queue listings or automatic host selection.
use anyhow::{bail, Context};
use std::io::Read;

const MAX_PORTABLE_MEMBER_BYTES: u64 = 32 * 1024 * 1024;

fn read_original(path: &std::path::Path, remaining: &mut u64) -> anyhow::Result<Vec<u8>> {
    let file = mold_core::secure_file::open_regular_file_no_follow(path)
        .context("Original source media is unavailable")?;
    let mut bytes = Vec::new();
    file.take(*remaining + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > *remaining {
        bail!("An original media member exceeds the portable input budget");
    }
    *remaining -= bytes.len() as u64;
    Ok(bytes)
}

use mold_core::{GenerateRequest, GenerationReference, GenerationReferenceAuthority};
use mold_db::generation_queue::QueueRowState;

use crate::queue_journal::QueueJournal;
use crate::queue_media_runtime::ZeroizingGenerateRequest;

fn remaining_budget(request: &GenerateRequest) -> anyhow::Result<u64> {
    let mut remaining = MAX_PORTABLE_MEMBER_BYTES;
    let scalar = [
        &request.source_image,
        &request.id_image,
        &request.mask_image,
        &request.control_image,
        &request.audio_file,
        &request.source_video,
        &request.extend_video,
    ];
    let lengths = scalar
        .into_iter()
        .flatten()
        .map(Vec::len)
        .chain(request.id_images.iter().flatten().map(Vec::len))
        .chain(request.edit_images.iter().flatten().map(Vec::len))
        .chain(
            request
                .keyframes
                .iter()
                .flatten()
                .map(|frame| frame.image.len()),
        )
        .chain(request.references.iter().flatten().filter_map(
            |reference| match reference.media() {
                GenerationReferenceAuthority::Inline { data } => Some(data.len()),
                _ => None,
            },
        ));
    for length in lengths {
        remaining = remaining
            .checked_sub(length as u64)
            .context("Original media exceeds the portable input budget")?;
    }
    Ok(remaining)
}

pub(crate) fn export_request(
    journal: &QueueJournal,
    id: &str,
    db: Option<&mold_db::MetadataDb>,
) -> anyhow::Result<Vec<u8>> {
    let row = journal.row(id)?.context("held job no longer exists")?;
    if row.state != QueueRowState::Held
        && !(row.state == QueueRowState::Paused && journal.transfer_reserved(id)?)
    {
        bail!("Only a held job can be sent to another machine");
    }
    let mut request = ZeroizingGenerateRequest::from_owned(
        serde_json::from_str::<GenerateRequest>(&row.request_json)?,
    );
    let lease = if let Some(set_id) = row.media_set_id {
        let lifecycle = journal
            .queue_media_lifecycle()
            .context("Original media is unavailable")?;
        let media = lifecycle.deferred_media(crate::queue_media_store::MediaSetRef {
            owner_id: row.owner_uuid,
            job_id: row.id,
            set_id,
        })?;
        Some(media.hydrate_into(id, &mut request)?)
    } else {
        None
    };
    let mut remaining = remaining_budget(&request)?;
    if let Some(lease) = &lease {
        if let Some(references) = lease.references(&request)? {
            let paths = references.entries();
            for (reference, entry) in request.references.iter_mut().flatten().zip(paths) {
                let data = read_original(&entry.path, &mut remaining)
                    .context("Could not read an original reference")?;
                let media = match reference {
                    GenerationReference::Image { media, .. }
                    | GenerationReference::NamedImage { media, .. }
                    | GenerationReference::Video { media, .. }
                    | GenerationReference::Audio { media, .. }
                    | GenerationReference::Mesh { media, .. } => media,
                };
                *media = GenerationReferenceAuthority::Inline { data };
            }
        }
    }
    if let Some(db) = db {
        crate::routes::resolve_collection_reference(db, &mut request.collection);
    }
    portable_request(&mut request, lease.is_some())?;
    // Serialize while the zeroizing request and all authenticated staging
    // holds are alive. The response is explicitly no-store at the HTTP seam.
    Ok(serde_json::to_vec(&*request)?)
}

pub(crate) fn validate_portability(journal: &QueueJournal, id: &str) -> anyhow::Result<()> {
    let row = journal.row(id)?.context("Job no longer exists")?;
    let request = ZeroizingGenerateRequest::from_owned(serde_json::from_str::<GenerateRequest>(
        &row.request_json,
    )?);
    if row.media_set_id.is_none()
        && (request.audio_file_path.is_some()
            || request.source_video_path.is_some()
            || request.extend_video_path.is_some())
    {
        bail!("Original path-only media is unavailable for safe movement. Reuse settings and attach it again.");
    }
    if request.lora.is_some() || request.loras.as_ref().is_some_and(|rows| !rows.is_empty()) {
        bail!("This job uses a machine-local LoRA and cannot be moved. Reuse its settings on the destination.");
    }
    if request.hdr_exr_dir.is_some() || request.mesh_workflow.is_some() {
        bail!("This job belongs to a machine-local workflow and cannot be moved independently.");
    }
    Ok(())
}

fn portable_request(
    request: &mut GenerateRequest,
    authenticated_paths: bool,
) -> anyhow::Result<()> {
    // A machine-local adapter or EXR directory has no portable ingress form.
    // Refuse it by name rather than silently changing the requested pixels.
    if request.lora.is_some() || request.loras.as_ref().is_some_and(|rows| !rows.is_empty()) {
        bail!("This job uses a machine-local LoRA. Install and select that adapter on the destination before resubmitting.");
    }
    if request.hdr_exr_dir.is_some() || request.mesh_workflow.is_some() {
        bail!("This job belongs to a machine-local workflow and cannot be sent independently");
    }
    let mut remaining = remaining_budget(request)?;
    for (path, bytes) in [
        (&mut request.audio_file_path, &mut request.audio_file),
        (&mut request.source_video_path, &mut request.source_video),
        (&mut request.extend_video_path, &mut request.extend_video),
    ] {
        if let Some(path) = path.take() {
            if !authenticated_paths {
                bail!("Original path-only media has no retained authentication. Reuse settings and attach the original media again.");
            }
            if bytes.is_some() {
                bail!("The original media has conflicting authorities");
            }
            *bytes = Some(read_original(std::path::Path::new(&path), &mut remaining)?);
        }
    }
    if request.references.iter().flatten().any(|reference| {
        !matches!(
            reference.media(),
            GenerationReferenceAuthority::Inline { .. }
        )
    }) {
        bail!("The original reference media is unavailable");
    }
    // Accelerator identities and collection ids belong to the source host.
    request.placement = None;
    request.batch_size = 1;
    if let Some(collection) = &mut request.collection {
        if collection.id.is_some() && collection.name.is_none() {
            bail!("This job names a source-only collection. Reuse its settings to choose a destination collection.");
        }
        collection.id = None;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn singular_identity_and_edit_images_share_the_portable_media_budget() {
        for identity in [true, false] {
            let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "model":"ltx-2-19b-distilled:fp8", "prompt":"original", "width":512,
                "height":512, "steps":4
            }))
            .unwrap();
            request.source_image = Some(vec![1]);
            let image = vec![0; MAX_PORTABLE_MEMBER_BYTES as usize];
            if identity {
                request.id_image = Some(image);
            } else {
                request.edit_images = Some(vec![image]);
            }
            assert!(
                remaining_budget(&request).is_err(),
                "every inline image counts toward one aggregate budget"
            );
        }
    }
    #[test]
    fn legacy_mutable_path_is_refused_without_retained_media_authentication() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("changed.mp4");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8","prompt":"original","width":512,"height":512,"steps":4,"source_video_path":path})).unwrap();
        assert!(portable_request(&mut request, false).is_err());
    }

    #[test]
    fn portable_media_reads_share_one_bounded_budget() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("original");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let mut remaining = 5;
        assert_eq!(read_original(&path, &mut remaining).unwrap(), [1, 2, 3]);
        assert_eq!(remaining, 2);
        assert!(read_original(&path, &mut remaining).is_err());
        assert_eq!(remaining, 2);
    }

    #[test]
    fn transfer_preserves_pixels_and_clears_source_device_affinity() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "minimax-h3-fl2va:comfy-pruned-int8", "prompt": "exact words",
            "seed": 42, "width": 1344, "height": 768, "frames": 121, "steps": 4, "output_format": "mp4",
            "source_image": "AQID", "batch_size": 1
        })).unwrap();
        portable_request(&mut request, false).unwrap();
        assert_eq!(request.prompt, "exact words");
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.source_image, Some(vec![1, 2, 3]));
        assert_eq!(request.placement, None);
    }
    #[test]
    fn transfer_refuses_unresolved_media_instead_of_losing_conditioning() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "minimax-h3-ref2va:comfy-pruned-int8", "prompt": "exact words", "width": 32, "height": 32, "steps": 4, "batch_size": 1, "output_format": "mp4",
            "references": [{"kind":"image", "media":{"authority":"descriptor"}, "mime_type":"image/png", "width":32,"height":32}]
        })).unwrap();
        assert!(portable_request(&mut request, false).is_err());
    }
}
