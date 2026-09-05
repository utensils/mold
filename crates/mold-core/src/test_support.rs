#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A minimal, fully-populated [`crate::types::GenerateRequest`] for tests that
/// only care about one field. Kept here so the exhaustive struct literal lives
/// in one place instead of being copied into every test module.
#[cfg(test)]
pub(crate) fn minimal_generate_request(model: &str) -> crate::types::GenerateRequest {
    crate::types::GenerateRequest {
        mesh: None,
        video_only: None,
        title: None,
        tags: None,
        collection: None,
        source_fit: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: "a red apple".to_string(),
        negative_prompt: None,
        model: model.to_string(),
        width: 1024,
        height: 1024,
        steps: 4,
        guidance: 0.0,
        seed: Some(42),
        batch_size: 1,
        output_format: Some(crate::types::OutputFormat::Png),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        source_image_name: None,
        edit_images: None,
        references: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        save_to_gallery: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: None,
        pipeline: None,
        ic_lora_control: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
        id_image: None,
        id_image_name: None,
        id_weight: None,
        id_start_step: None,
        id_images: None,
        id_image_names: None,
        true_cfg: None,
        cfg_start_step: None,
    }
}

/// The inode `ctime` of `path` as `(seconds, nanoseconds)`, for pairing with
/// [`wait_until_ctime_moves`] around a tamper write.
#[cfg(unix)]
pub(crate) fn ctime_of(path: &std::path::Path) -> (i64, i64) {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path).unwrap();
    (metadata.ctime(), metadata.ctime_nsec())
}

/// Spin until `path`'s `ctime` differs from `from`.
///
/// Linux stamps inodes from the coarse clock (`ktime_get_coarse_real_ts64`,
/// a 1–4 ms tick), so a same-length in-place write that lands inside the tick
/// of the previous stamp leaves `ctime` byte-identical and the identity memos
/// under test cannot tell the two files apart — which is a property of the
/// clock, not of the memo. The nudge is `set_permissions` with the file's
/// current mode: a `chmod` always marks the status-change time and touches
/// nothing else — never `mtime`, the length, or the inode — so the test's
/// tamper stays exactly the tamper it describes. Panics if `ctime` never
/// advances within two seconds: a filesystem that will not move it is a
/// broken assumption, not a flake to retry.
#[cfg(unix)]
pub(crate) fn wait_until_ctime_moves(path: &std::path::Path, from: (i64, i64)) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let metadata = std::fs::metadata(path).unwrap();
        if ctime_of(path) != from {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "ctime of {} stayed at {from:?} for two seconds despite chmod nudges",
            path.display()
        );
        std::fs::set_permissions(path, metadata.permissions()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Non-Unix platforms keep no inode change time, and the identity memos there
/// key on `(len, mtime)` instead — nothing to wait for.
#[cfg(not(unix))]
pub(crate) fn ctime_of(_path: &std::path::Path) -> (i64, i64) {
    (0, 0)
}

#[cfg(not(unix))]
pub(crate) fn wait_until_ctime_moves(_path: &std::path::Path, _from: (i64, i64)) {}
