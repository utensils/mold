//! Shared "record a generation output that was just written to disk"
//! helper. The server queue/GPU-worker paths and the CLI local path all
//! build the same `GenerationRecord` tail (stat, timing, hostname,
//! backend, upsert) — this is the single home so a column can't silently
//! stop being populated on one path while the other keeps working.

use std::path::Path;

use mold_core::{OutputFormat, OutputMetadata};

use crate::record::{GenerationRecord, RecordSource};
use crate::MetadataDb;

/// Provenance and timing for one saved output.
pub struct OutputRecordParams<'a> {
    pub format: OutputFormat,
    pub metadata: &'a OutputMetadata,
    pub source: RecordSource,
    pub generation_time_ms: Option<i64>,
    pub backend: Option<&'a str>,
}

/// Build and upsert a `GenerationRecord` for a file already written to
/// `output_dir/filename` (stat'd via `on_disk`). Path policy is the
/// caller's: the server passes its configured dir verbatim (the gallery
/// unique index keys on that exact string), the CLI canonicalizes first.
///
/// Best-effort: upsert errors are logged and swallowed; returns `false`
/// on failure. The file on disk stays the source of truth.
pub fn record_saved_output(
    db: &MetadataDb,
    output_dir: &Path,
    filename: &str,
    on_disk: &Path,
    params: &OutputRecordParams<'_>,
) -> bool {
    let mut rec = GenerationRecord::from_save(
        output_dir,
        filename,
        params.format,
        params.metadata.clone(),
        params.source,
        mold_core::time::now_epoch_ms(),
    );
    rec.stat_from_disk(on_disk);
    rec.generation_time_ms = params.generation_time_ms;
    rec.hostname = hostname_string();
    rec.backend = params.backend.map(|s| s.to_string());
    if let Err(e) = db.upsert(&rec) {
        tracing::warn!("metadata DB upsert failed for {}: {e:#}", rec.filename);
        return false;
    }
    true
}

/// Best-effort hostname for the `hostname` DB column. Falls back to `None`.
pub fn hostname_string() -> Option<String> {
    hostname::get().ok().and_then(|s| s.into_string().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> OutputMetadata {
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{
                "prompt":"a stoic owl",
                "model":"flux-dev:q4",
                "width":1024,
                "height":1024,
                "steps":20,
                "guidance":4.0
            }"#,
        )
        .unwrap();
        OutputMetadata::from_generate_request(&req, 42, None, "test")
    }

    #[test]
    fn record_saved_output_populates_provenance_columns() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("mold-flux-dev-q4-1700000000000.png");
        std::fs::write(&file, b"fake-bytes").unwrap();
        let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();

        let ok = record_saved_output(
            &db,
            dir.path(),
            "mold-flux-dev-q4-1700000000000.png",
            &file,
            &OutputRecordParams {
                format: OutputFormat::Png,
                metadata: &metadata(),
                source: RecordSource::Cli,
                generation_time_ms: Some(2_500),
                backend: Some("cpu"),
            },
        );
        assert!(ok);

        let got = db
            .get(dir.path(), "mold-flux-dev-q4-1700000000000.png")
            .unwrap()
            .unwrap();
        assert_eq!(got.metadata.prompt, "a stoic owl");
        assert_eq!(got.source, RecordSource::Cli);
        assert_eq!(got.generation_time_ms, Some(2_500));
        assert_eq!(got.backend.as_deref(), Some("cpu"));
        assert_eq!(got.file_size_bytes, Some(b"fake-bytes".len() as i64));
        // created_at is stamped from the wall clock in epoch millis.
        assert!(
            got.created_at_ms > 1_767_225_600_000,
            "{}",
            got.created_at_ms
        );
    }
}
