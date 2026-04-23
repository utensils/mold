//! Thin CLI wrapper around the process-wide metadata DB handle that now
//! lives in `mold-db`. Keeps the `record_local_save` helper the CLI uses
//! on its local generation path; everything else delegates.

use std::path::Path;

use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
use mold_db::{GenerationRecord, MetadataDb, RecordSource};

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
    if cfg!(feature = "cuda") {
        Some("cuda".into())
    } else if cfg!(feature = "metal") {
        Some("metal".into())
    } else {
        Some("cpu".into())
    }
}

/// Persist a metadata row for a file the CLI just wrote locally.
///
/// `saved_path` is the on-disk file (used to derive `output_dir + filename`).
/// `req` carries prompt / dimensions / lora; `seed_used` and
/// `generation_time_ms` come from the engine's response.
///
/// Best-effort: errors are logged and discarded. Returns `false` when the
/// DB is disabled or open failed, true otherwise.
pub fn record_local_save(
    saved_path: &Path,
    req: &GenerateRequest,
    seed_used: u64,
    generation_time_ms: u64,
    format: OutputFormat,
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
    let metadata = OutputMetadata::from_generate_request(
        req,
        seed_used,
        None,
        mold_core::build_info::version_string(),
    );
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let mut rec = GenerationRecord::from_save(
        output_dir,
        filename,
        format,
        metadata,
        RecordSource::Cli,
        now_ms,
    );
    rec.stat_from_disk(&abs);
    rec.generation_time_ms = Some(generation_time_ms as i64);
    rec.backend = backend_label();
    rec.hostname = hostname::get().ok().and_then(|s| s.into_string().ok());
    if let Err(e) = db.upsert(&rec) {
        tracing::warn!("metadata DB upsert failed for {}: {e:#}", abs.display());
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateRequest, OutputFormat};

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
}
