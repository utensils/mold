//! Explicit export of one held request, with its original durable media.
//! This is never used by queue listings or automatic host selection.
use anyhow::{bail, Context};
use mold_core::{GenerateRequest, GenerationReference, GenerationReferenceAuthority};
use mold_db::generation_queue::QueueRowState;

use crate::queue_journal::QueueJournal;
use crate::queue_media_runtime::ZeroizingGenerateRequest;

pub(crate) fn export_request(
    journal: &QueueJournal,
    id: &str,
    db: Option<&mold_db::MetadataDb>,
) -> anyhow::Result<Vec<u8>> {
    let row = journal.row(id)?.context("held job no longer exists")?;
    if row.state != QueueRowState::Held {
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
    if let Some(lease) = &lease {
        if let Some(references) = lease.references(&request)? {
            let paths = references.entries();
            for (reference, entry) in request.references.iter_mut().flatten().zip(paths) {
                let data =
                    std::fs::read(&entry.path).context("Could not read an original reference")?;
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
    portable_request(&mut request)?;
    // Serialize while the zeroizing request and all authenticated staging
    // holds are alive. The response is explicitly no-store at the HTTP seam.
    Ok(serde_json::to_vec(&*request)?)
}

fn portable_request(request: &mut GenerateRequest) -> anyhow::Result<()> {
    // A machine-local adapter or EXR directory has no portable ingress form.
    // Refuse it by name rather than silently changing the requested pixels.
    if request.lora.is_some() || request.loras.as_ref().is_some_and(|rows| !rows.is_empty()) {
        bail!("This job uses a machine-local LoRA. Install and select that adapter on the destination before resubmitting.");
    }
    if request.hdr_exr_dir.is_some() || request.mesh_workflow.is_some() {
        bail!("This job belongs to a machine-local workflow and cannot be sent independently");
    }
    for (path, bytes) in [
        (&mut request.audio_file_path, &mut request.audio_file),
        (&mut request.source_video_path, &mut request.source_video),
        (&mut request.extend_video_path, &mut request.extend_video),
    ] {
        if let Some(path) = path.take() {
            if bytes.is_some() {
                bail!("The original media has conflicting authorities");
            }
            *bytes = Some(std::fs::read(path).context("Original source media is unavailable")?);
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
    fn transfer_preserves_pixels_and_clears_source_device_affinity() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "minimax-h3-fl2va:comfy-pruned-int8", "prompt": "exact words",
            "seed": 42, "width": 1344, "height": 768, "frames": 121, "steps": 4, "output_format": "mp4",
            "source_image": "AQID", "batch_size": 1
        })).unwrap();
        portable_request(&mut request).unwrap();
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
        assert!(portable_request(&mut request).is_err());
    }
}
