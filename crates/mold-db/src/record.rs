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
#[derive(Debug, Clone)]
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
            metadata,
            generation_time_ms: None,
            backend: None,
            hostname: None,
            source,
            metadata_synthetic: false,
        }
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> OutputMetadata {
        OutputMetadata {
            prompt: "p".into(),
            negative_prompt: None,
            original_prompt: None,
            model: "m".into(),
            seed: 1,
            steps: 2,
            guidance: 3.0,
            width: 4,
            height: 5,
            strength: None,
            scheduler: None,
            lora: None,
            lora_scale: None,
            frames: None,
            fps: None,
            version: "v".into(),
        }
    }

    #[test]
    fn record_source_roundtrips() {
        for src in [
            RecordSource::Server,
            RecordSource::Cli,
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
