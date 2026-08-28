//! Thin CLI wrapper around the process-wide metadata DB handle that now
//! lives in `mold-db`. Keeps the `record_local_save` helper the CLI uses
//! on its local generation path; everything else delegates.

use std::path::Path;

use mold_core::{GenerateRequest, OutputFormat, OutputMetadata, VideoData};
use mold_db::{MetadataDb, RecordSource};

/// Install the `Config::load_or_default()` post-load hook so every freshly
/// loaded config is overlaid with DB-backed user preferences, and run the
/// one-shot `config.toml → DB` import on first call. Delegates to
/// `mold_db` so `mold-server` and `mold-discord` standalone binaries can
/// call the same entry point.
pub fn install_config_db_hooks() {
    mold_db::config_sync::install_config_post_load_hook();
}

/// Borrow the process-wide metadata DB. Open errors are logged once and
/// then suppressed — the CLI must keep working without persistence.
pub fn handle() -> Option<&'static MetadataDb> {
    mold_db::global_db()
}

/// Compile-time backend label for rows written from the CLI's local path.
fn backend_label() -> Option<String> {
    Some(mold_inference::compiled_backend_label().to_string())
}

/// Persist a metadata row for a file the CLI just wrote locally.
///
/// `saved_path` is the on-disk file (used to derive `output_dir + filename`).
/// `req` carries prompt / dimensions / lora; `seed_used` and
/// `generation_time_ms` come from the engine's response. When
/// `actual_dims` is set, it overwrites the requested dimensions so the
/// row describes the file that exists (e.g. post-upscale).
/// `video` supplies completed runtime metadata that cannot be inferred from
/// the request, notably an Auto-selected LTX-2 pipeline.
///
/// Best-effort: errors are logged and discarded. Returns `false` when the
/// DB is disabled or open failed, true otherwise.
pub fn record_local_save(
    saved_path: &Path,
    req: &GenerateRequest,
    seed_used: u64,
    generation_time_ms: u64,
    format: OutputFormat,
    actual_dims: Option<(u32, u32)>,
    video: Option<&VideoData>,
) -> bool {
    let metadata = metadata_for_local_save(req, seed_used, video);
    record_local_save_metadata(
        saved_path,
        metadata,
        generation_time_ms,
        format,
        actual_dims,
    )
}

fn metadata_for_local_save(
    req: &GenerateRequest,
    seed_used: u64,
    video: Option<&VideoData>,
) -> OutputMetadata {
    let mut metadata = OutputMetadata::from_generate_request(
        req,
        seed_used,
        None,
        mold_core::build_info::version_string(),
    );
    if let Some(video) = video {
        metadata.apply_video_output(video);
    }
    metadata
}

/// Like [`record_local_save`] but with caller-built metadata — the chain
/// command uses this to carry the structured per-clip provenance block that
/// a synthetic single-clip `GenerateRequest` cannot express.
pub fn record_local_save_metadata(
    saved_path: &Path,
    mut metadata: OutputMetadata,
    generation_time_ms: u64,
    format: OutputFormat,
    actual_dims: Option<(u32, u32)>,
) -> bool {
    let Some(db) = handle() else {
        return false;
    };
    // Resolve to an absolute path so two galleries with the same filename
    // (e.g. `out.png` in two cwds) don't collide on the unique index.
    let abs = std::fs::canonicalize(saved_path).unwrap_or_else(|_| saved_path.to_path_buf());
    let Some(filename) = abs
        .file_name()
        .and_then(|f| f.to_str())
        .map(|s| s.to_string())
    else {
        return false;
    };
    let Some(output_dir) = abs.parent() else {
        return false;
    };
    if let Some((w, h)) = actual_dims {
        metadata.apply_output_dimensions(w, h);
    }
    mold_db::persist::record_saved_output(
        db,
        output_dir,
        &filename,
        &abs,
        &mold_db::persist::OutputRecordParams {
            format,
            metadata: &metadata,
            source: RecordSource::Cli,
            generation_time_ms: Some(generation_time_ms as i64),
            backend: backend_label().as_deref(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateRequest, OutputFormat};
    use mold_db::GenerationRecord;

    fn req() -> GenerateRequest {
        // GenerateRequest doesn't impl Default — easiest minimal builder is
        // a JSON literal with all serde-required fields, then let optional
        // fields stay unset.
        serde_json::from_str(
            r#"{
                "prompt":"a stoic owl",
                "model":"flux-dev:q4",
                "width":1024,
                "height":1024,
                "steps":20,
                "guidance":4.0
            }"#,
        )
        .unwrap()
    }

    /// Direct round-trip through `MetadataDb` to mirror what the CLI helper
    /// would write — keeps the DB schema honest for the CLI's column
    /// expectations without depending on the global `OnceLock` handle.
    #[test]
    fn round_trip_constructs_record_from_request_and_seed() {
        let dir = tempfile::tempdir().unwrap();
        let saved = dir.path().join("mold-flux-dev-q4-1.png");
        std::fs::write(&saved, b"fake-bytes").unwrap();

        let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();

        let metadata = OutputMetadata::from_generate_request(
            &req(),
            42,
            None,
            mold_core::build_info::version_string(),
        );
        let mut rec = GenerationRecord::from_save(
            dir.path(),
            "mold-flux-dev-q4-1.png",
            OutputFormat::Png,
            metadata,
            RecordSource::Cli,
            1_700_000_000_000,
        );
        rec.stat_from_disk(&saved);
        rec.generation_time_ms = Some(2_500);
        rec.backend = backend_label();

        db.upsert(&rec).unwrap();
        let got = db
            .get(dir.path(), "mold-flux-dev-q4-1.png")
            .unwrap()
            .unwrap();
        assert_eq!(got.metadata.prompt, "a stoic owl");
        assert_eq!(got.metadata.seed, 42);
        assert_eq!(got.source, RecordSource::Cli);
        assert_eq!(got.generation_time_ms, Some(2_500));
        assert_eq!(got.file_size_bytes, Some(b"fake-bytes".len() as i64));
    }

    #[test]
    fn backend_label_is_non_empty() {
        let label = backend_label().unwrap();
        assert!(["cuda", "metal", "cpu"].contains(&label.as_str()));
    }

    #[test]
    fn local_video_metadata_uses_the_runtime_resolved_pipeline() {
        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: Vec::new(),
            format: OutputFormat::Mp4,
            width: 1216,
            height: 704,
            frames: 97,
            fps: 24,
            pipeline: Some(mold_core::Ltx2PipelineMode::TwoStageHq),
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: Vec::new(),
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };

        let metadata = metadata_for_local_save(&req(), 42, Some(&video));

        assert_eq!(
            metadata.pipeline,
            Some(mold_core::Ltx2PipelineMode::TwoStageHq),
            "the runtime response is authoritative when an Auto request omitted pipeline"
        );
    }

    /// The CLI's own local-save path records identity provenance, so a print
    /// rendered through `--local` is as attributable as one the server saved.
    /// Both go through `OutputMetadata::from_generate_request`, which is why
    /// this is a round trip rather than a second implementation.
    #[test]
    fn a_local_save_records_the_identity_provenance_it_rendered_with() {
        let dir = tempfile::tempdir().unwrap();
        let saved = dir.path().join("mold-flux-dev-q4-1.png");
        std::fs::write(&saved, b"fake-bytes").unwrap();
        let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();

        let mut request = req();
        request.id_image = Some(b"pretend-png".to_vec());
        request.id_image_name = Some("portrait.png".to_string());
        request.id_weight = Some(0.85);
        request.id_start_step = Some(2);

        let metadata = metadata_for_local_save(&request, 42, None);
        assert_eq!(metadata.id_image_name.as_deref(), Some("portrait.png"));
        assert_eq!(metadata.id_weight, Some(0.85));
        assert_eq!(metadata.id_start_step, Some(2));
        assert_eq!(
            metadata.id_image_sha256.as_deref(),
            Some(mold_core::identity::id_image_sha256(b"pretend-png").as_str()),
            "the recorded digest must identify the exact reference photograph"
        );

        let mut rec = GenerationRecord::from_save(
            dir.path(),
            "mold-flux-dev-q4-1.png",
            OutputFormat::Png,
            metadata,
            RecordSource::Cli,
            1_700_000_000_000,
        );
        rec.stat_from_disk(&saved);
        db.upsert(&rec).unwrap();

        let got = db
            .get(dir.path(), "mold-flux-dev-q4-1.png")
            .unwrap()
            .expect("the row round-trips");
        assert_eq!(got.metadata.id_image_name.as_deref(), Some("portrait.png"));
        assert_eq!(got.metadata.id_weight, Some(0.85));
        assert_eq!(got.metadata.id_start_step, Some(2));
        // The photograph itself is never stored — only its digest.
        let serialized = serde_json::to_string(&got.metadata).unwrap();
        assert!(!serialized.contains("pretend-png"), "{serialized}");
    }

    /// A print with no identity records none, so a bare knob on an ordinary
    /// render can never read as conditioning that did not happen.
    #[test]
    fn a_local_save_without_identity_records_no_identity_fields() {
        let metadata = metadata_for_local_save(&req(), 42, None);
        assert!(metadata.id_image_name.is_none());
        assert!(metadata.id_image_sha256.is_none());
        assert!(metadata.id_weight.is_none());
        assert!(metadata.id_start_step.is_none());
    }
}
