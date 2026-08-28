//! The persisted form of a generation request.
//!
//! The durable queue never stores request media: `scrub_request_media` is
//! the ONE list of fields the journal strips (and zeroizes) before a row is
//! written, and everything that must compare a hydrated request with its
//! persisted row hashes THIS form — the durable H3 ingress grant
//! (`request_authority_sha256`) and the H3 admission evidence's request
//! identity both do. It lives in `mold-core` so `mold-inference` can hash the
//! same form the server persists; two copies of this list drifted once and
//! held every H3 first-frame job at resolve (#1423).

use zeroize::Zeroize;

use crate::GenerateRequest;

/// Wipe every request field that the durable queue-media overlay can restore.
///
/// This is intentionally the same exhaustive authority set as
/// `extract_request_fields`. Attempt-scoped runtime guards call it before
/// releasing private staging, and zeroizing request clones call it on every
/// success/error exit from downstream worker ownership. Reference DESCRIPTORS
/// are deliberately left alone: they are settings, not media — byte-identical
/// to what `OutputMetadata.references` persists — and no public authority can
/// be present on them past admission.
pub fn scrub_request_media(request: &mut GenerateRequest) {
    fn scrub_bytes(value: &mut Option<Vec<u8>>) {
        if let Some(bytes) = value {
            bytes.zeroize();
        }
        *value = None;
    }

    fn scrub_text(value: &mut Option<String>) {
        if let Some(text) = value {
            text.zeroize();
        }
        *value = None;
    }

    fn scrub_byte_collection(value: &mut Option<Vec<Vec<u8>>>) {
        if let Some(items) = value {
            for item in items.iter_mut() {
                item.zeroize();
            }
            items.clear();
        }
        *value = None;
    }

    fn scrub_text_collection(value: &mut Option<Vec<String>>) {
        if let Some(items) = value {
            for item in items.iter_mut() {
                item.zeroize();
            }
            items.clear();
        }
        *value = None;
    }

    scrub_bytes(&mut request.source_image);
    scrub_text(&mut request.source_image_name);
    scrub_bytes(&mut request.id_image);
    scrub_text(&mut request.id_image_name);
    scrub_byte_collection(&mut request.id_images);
    scrub_text_collection(&mut request.id_image_names);
    scrub_byte_collection(&mut request.edit_images);
    scrub_bytes(&mut request.mask_image);
    scrub_bytes(&mut request.control_image);
    scrub_bytes(&mut request.audio_file);
    scrub_text(&mut request.audio_file_path);
    scrub_bytes(&mut request.source_video);
    scrub_text(&mut request.source_video_path);
    scrub_bytes(&mut request.extend_video);
    scrub_text(&mut request.extend_video_path);
    if let Some(keyframes) = &mut request.keyframes {
        for keyframe in keyframes.iter_mut() {
            keyframe.image.zeroize();
            if let Some(name) = &mut keyframe.name {
                name.zeroize();
            }
        }
        keyframes.clear();
    }
    request.keyframes = None;
    scrub_text(&mut request.hdr_exr_dir);
    if let Some(lora) = &mut request.lora {
        lora.path.zeroize();
    }
    request.lora = None;
    if let Some(loras) = &mut request.loras {
        for lora in loras.iter_mut() {
            lora.path.zeroize();
        }
        loras.clear();
    }
    request.loras = None;
}

/// The request as the durable queue persists it: a clone with every media
/// payload scrubbed. Hash this, never the hydrated request, when the hash
/// must survive the round trip through the journal.
pub fn persisted_request_form(request: &GenerateRequest) -> GenerateRequest {
    let mut persisted = request.clone();
    scrub_request_media(&mut persisted);
    persisted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_form_is_the_scrubbed_request_and_is_idempotent() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a lighthouse in a storm",
            "model": "minimax-h3-fl2va:comfy-pruned-int8",
            "width": 768,
            "height": 768,
            "steps": 4,
            "source_image": "AQID",
            "source_image_name": "first.png",
        }))
        .unwrap();
        assert_eq!(request.source_image.as_deref(), Some(&[1_u8, 2, 3][..]));
        let persisted = persisted_request_form(&request);
        assert!(persisted.source_image.is_none());
        assert!(persisted.source_image_name.is_none());
        let as_json = |value: &GenerateRequest| serde_json::to_value(value).unwrap();
        assert_eq!(
            as_json(&persisted_request_form(&persisted)),
            as_json(&persisted)
        );
        scrub_request_media(&mut request);
        assert_eq!(as_json(&request), as_json(&persisted));
    }
}
