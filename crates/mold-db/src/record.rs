use mold_core::{OutputFormat, OutputMetadata};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// What surface inserted a row. Used for diagnostics and to distinguish
/// reconciled / backfilled rows from real generation events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordSource {
    /// Written by `mold serve` after a successful `/api/generate*` call.
    Server,
    /// Written by the CLI's local generation path (`mold run --local` or local fallback).
    Cli,
    /// Written by the TUI's local generation path.
    Tui,
    /// Imported by the startup reconciliation walk from a file already on disk.
    Backfill,
    /// Catch-all for rows whose origin was lost (e.g. an upgrade migrating
    /// older data into the table).
    Unknown,
}

impl RecordSource {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordSource::Server => "server",
            RecordSource::Cli => "cli",
            RecordSource::Tui => "tui",
            RecordSource::Backfill => "backfill",
            RecordSource::Unknown => "unknown",
        }
    }

    /// Parse a stored DB string back into a `RecordSource`. Unknown values
    /// (e.g. from a future schema) round-trip as [`RecordSource::Unknown`]
    /// rather than failing.
    pub fn parse(s: &str) -> Self {
        match s {
            "server" => RecordSource::Server,
            "cli" => RecordSource::Cli,
            "tui" => RecordSource::Tui,
            "backfill" => RecordSource::Backfill,
            _ => RecordSource::Unknown,
        }
    }
}

/// One row in the `generations` table — a saved gallery file plus its
/// generation metadata.
///
/// `id` is `None` for unpersisted records and `Some(_)` after `upsert`.
/// `output_dir` is stored as an absolute path so we can disambiguate
/// identical filenames in different gallery directories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub id: Option<i64>,
    pub filename: String,
    pub output_dir: String,
    pub created_at_ms: i64,
    pub file_mtime_ms: Option<i64>,
    pub file_size_bytes: Option<i64>,
    pub format: OutputFormat,
    pub metadata: OutputMetadata,
    pub generation_time_ms: Option<i64>,
    pub backend: Option<String>,
    pub hostname: Option<String>,
    pub source: RecordSource,
    /// True when [`metadata`] was synthesized from the filename (no embedded
    /// `mold:parameters` chunk). Mirrors [`mold_core::GalleryImage::metadata_synthetic`].
    pub metadata_synthetic: bool,
    /// User-owned display title. Seeded from the generation request on
    /// insert and editable afterwards; a reconcile refresh never resets it.
    #[serde(default)]
    pub title: Option<String>,
    /// User-owned favorite flag. Survives reconcile refreshes.
    #[serde(default)]
    pub favorite: bool,
    /// When the print was moved to `<output_dir>/.trash/`; `None` for a
    /// live print. The row keeps its `(output_dir, filename)` identity while
    /// trashed. Survives reconcile refreshes.
    #[serde(default)]
    pub trashed_at_ms: Option<i64>,
}

impl GenerationRecord {
    /// Construct a record for a file we just wrote, with no on-disk stat yet.
    /// Caller may run [`Self::stat_from_disk`] afterward to fill mtime/size.
    pub fn from_save(
        output_dir: &Path,
        filename: impl Into<String>,
        format: OutputFormat,
        metadata: OutputMetadata,
        source: RecordSource,
        created_at_ms: i64,
    ) -> Self {
        Self {
            id: None,
            filename: filename.into(),
            output_dir: output_dir.to_string_lossy().into_owned(),
            created_at_ms,
            file_mtime_ms: None,
            file_size_bytes: None,
            format,
            generation_time_ms: None,
            backend: None,
            hostname: None,
            source,
            metadata_synthetic: false,
            // The user-editable title starts as the creation-time title the
            // request carried (embedded in `mold:parameters`), so mirrors and
            // reconcile-from-disk recover it; later edits live on the row only.
            title: metadata.title.clone(),
            favorite: false,
            trashed_at_ms: None,
            metadata,
        }
    }

    /// True when the print is live (not in the trash).
    pub fn is_live(&self) -> bool {
        self.trashed_at_ms.is_none()
    }

    /// Update [`Self::file_mtime_ms`] and [`Self::file_size_bytes`] from a
    /// fresh `stat()`. Best-effort: errors are silently ignored.
    pub fn stat_from_disk(&mut self, path: &Path) {
        if let Ok(meta) = std::fs::metadata(path) {
            self.file_size_bytes = Some(meta.len() as i64);
            if let Ok(modified) = meta.modified() {
                if let Ok(d) = modified.duration_since(std::time::UNIX_EPOCH) {
                    self.file_mtime_ms = Some(d.as_millis() as i64);
                }
            }
        }
    }

    /// Convert to the wire shape returned by `/api/gallery`. Uses
    /// `file_mtime_ms` as the displayed timestamp (seconds), falling back
    /// to `created_at_ms` so synthetic / freshly-inserted rows still sort
    /// reasonably.
    pub fn to_gallery_image(&self) -> mold_core::GalleryImage {
        let timestamp = self
            .file_mtime_ms
            .or(Some(self.created_at_ms))
            .map(|ms| (ms / 1000) as u64)
            .unwrap_or(0);
        mold_core::GalleryImage {
            filename: self.filename.clone(),
            metadata: self.metadata.clone(),
            timestamp,
            format: Some(self.format),
            size_bytes: self.file_size_bytes.map(|n| n as u64),
            metadata_synthetic: self.metadata_synthetic,
            // Row title (user-editable) wins over the creation-time metadata
            // title; tags/collections/purge_at are applied by the server's
            // post-overlay enrichment, never here.
            title: self.title.clone().or_else(|| self.metadata.title.clone()),
            tags: Vec::new(),
            favorite: self.favorite,
            collections: Vec::new(),
            trashed_at: self.trashed_at_ms.map(|ms| (ms / 1000) as u64),
            purge_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> OutputMetadata {
        OutputMetadata {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "p".into(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            model: "m".into(),
            seed: 1,
            steps: 2,
            guidance: 3.0,
            width: 4,
            height: 5,
            generation_width: Some(4),
            generation_height: Some(5),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            chain_job_id: None,
            chain: None,
            version: "v".into(),
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
        }
    }

    #[test]
    fn record_source_roundtrips() {
        for src in [
            RecordSource::Server,
            RecordSource::Cli,
            RecordSource::Tui,
            RecordSource::Backfill,
            RecordSource::Unknown,
        ] {
            assert_eq!(RecordSource::parse(src.as_str()), src);
        }
    }

    #[test]
    fn from_save_constructs_unpersisted_row() {
        let rec = GenerationRecord::from_save(
            Path::new("/tmp/out"),
            "x.png",
            OutputFormat::Png,
            meta(),
            RecordSource::Cli,
            10,
        );
        assert_eq!(rec.id, None);
        assert_eq!(rec.filename, "x.png");
        assert_eq!(rec.output_dir, "/tmp/out");
        assert_eq!(rec.source, RecordSource::Cli);
        assert!(!rec.metadata_synthetic);
    }

    /// Rows serialized before v20 (e.g. a server snapshot written by an
    /// older build) carry none of the organization fields; they must
    /// deserialize with neutral defaults instead of failing.
    #[test]
    fn organization_fields_default_when_absent_from_serialized_record() {
        let rec = GenerationRecord::from_save(
            Path::new("/o"),
            "f.png",
            OutputFormat::Png,
            meta(),
            RecordSource::Server,
            5_000,
        );
        let mut value = serde_json::to_value(&rec).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("title");
        obj.remove("favorite");
        obj.remove("trashed_at_ms");
        let back: GenerationRecord = serde_json::from_value(value).unwrap();
        assert_eq!(back.title, None);
        assert!(!back.favorite);
        assert_eq!(back.trashed_at_ms, None);
        assert!(back.is_live());
    }

    #[test]
    fn to_gallery_image_prefers_mtime_when_present() {
        let mut rec = GenerationRecord::from_save(
            Path::new("/o"),
            "f.png",
            OutputFormat::Png,
            meta(),
            RecordSource::Server,
            5_000,
        );
        rec.file_mtime_ms = Some(20_000);
        let gi = rec.to_gallery_image();
        assert_eq!(gi.timestamp, 20);
    }
}
