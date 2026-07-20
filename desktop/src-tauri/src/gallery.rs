use std::io::{Read, Seek, SeekFrom};

use serde::Serialize;
use tauri::http::{header, Request, Response, StatusCode};

#[derive(Debug, Serialize)]
pub struct ImportedSourceImage {
    filename: String,
    base64: String,
    metadata: Option<mold_core::OutputMetadata>,
}

fn import_source_image_from_path(path: &std::path::Path) -> Result<ImportedSourceImage, String> {
    use base64::Engine;

    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "The dropped image has no valid filename.".to_string())?
        .to_string();
    let reader = image::ImageReader::open(path)
        .map_err(|error| format!("Couldn't open the dropped image: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("Couldn't identify the dropped image: {error}"))?;
    let format = match reader.format() {
        Some(image::ImageFormat::Png) => mold_core::OutputFormat::Png,
        Some(image::ImageFormat::Jpeg) => mold_core::OutputFormat::Jpeg,
        _ => return Err("Drop a PNG or JPEG image.".into()),
    };
    reader
        .into_dimensions()
        .map_err(|error| format!("Couldn't decode the dropped image: {error}"))?;

    let bytes =
        std::fs::read(path).map_err(|error| format!("Couldn't read the dropped image: {error}"))?;
    let metadata = mold_db::metadata_io::read_embedded(path, format);
    Ok(ImportedSourceImage {
        filename,
        base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        metadata,
    })
}

/// Read an OS-dropped still and its embedded Mold generation metadata. The
/// command validates the file's decoded format instead of trusting its suffix.
#[tauri::command]
pub async fn import_source_image(path: String) -> Result<ImportedSourceImage, String> {
    tauri::async_runtime::spawn_blocking(move || {
        import_source_image_from_path(std::path::Path::new(&path))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn output_dir() -> Option<std::path::PathBuf> {
    let config = mold_core::Config::load_or_default();
    (!config.is_output_disabled()).then(|| config.effective_output_dir())
}

fn valid_filename(filename: &str) -> bool {
    !filename.is_empty()
        && filename != "."
        && filename != ".."
        && !filename.contains('/')
        && !filename.contains('\\')
        && !filename.contains('\0')
}

fn scan(dir: &std::path::Path) -> Vec<mold_core::GalleryImage> {
    let mut images: Vec<_> = mold_db::scan::scan_output_dir(dir)
        .filter_map(|item| match item {
            mold_db::scan::ScanItem::Valid(file) => Some(file),
            _ => None,
        })
        .map(|file| {
            let timestamp = file.timestamp_secs();
            let size_bytes = file.size_u64();
            let (metadata, synthetic) = mold_db::metadata_io::read_or_synthesize(
                &file.path,
                file.format,
                &file.filename,
                timestamp,
            );
            mold_core::GalleryImage {
                filename: file.filename,
                metadata,
                timestamp,
                format: Some(file.format),
                size_bytes: Some(size_bytes),
                metadata_synthetic: synthetic,
            }
        })
        .collect();
    images.sort_by_key(|image| std::cmp::Reverse(image.timestamp));
    images
}

#[tauri::command]
pub async fn local_gallery_list() -> Result<Vec<mold_core::GalleryImage>, String> {
    let Some(dir) = output_dir() else {
        return Ok(Vec::new());
    };
    tauri::async_runtime::spawn_blocking(move || {
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        if let Some(db) = mold_db::global_db() {
            let rows = db.list(Some(&dir)).map_err(|error| format!("{error:#}"))?;
            if !rows.is_empty() {
                return Ok(rows.iter().map(|row| row.to_gallery_image()).collect());
            }
        }
        Ok(scan(&dir))
    })
    .await
    .map_err(|error| error.to_string())?
}

/// First free name for a new save: `name.png`, then `name-2.png`, `-3`, …
/// Pure over the `exists` probe so it's testable without hitting a real dir.
fn unique_output_path(dir: &std::path::Path, filename: &str) -> std::path::PathBuf {
    let candidate = dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let (stem, ext) = match filename.rsplit_once('.') {
        Some((s, e)) => (s, e),
        None => (filename, ""),
    };
    for n in 2.. {
        let name = if ext.is_empty() {
            format!("{stem}-{n}")
        } else {
            format!("{stem}-{n}.{ext}")
        };
        let candidate = dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("the counter loop always yields a free name")
}

/// True when `path` holds exactly `bytes` — size gate first, then a chunked
/// streaming compare so a large video is never buffered twice in memory.
fn file_matches_bytes(path: &std::path::Path, bytes: &[u8]) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() != bytes.len() as u64 {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut offset = 0usize;
    let mut buf = [0u8; 1 << 16];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return offset == bytes.len(),
            Ok(n) => {
                if bytes.get(offset..offset + n) != Some(&buf[..n]) {
                    return false;
                }
                offset += n;
            }
            Err(_) => return false,
        }
    }
}

/// Write generated bytes into this Mac's output dir — auto-save of remote
/// results and the gallery's "Save to this Mac". The bytes are the encoded
/// output file, so the local gallery reads embedded provenance back out of
/// it; `metadata` optionally carries the origin server's recorded metadata
/// for formats that embed nothing (video). Returns the saved filename.
///
/// Saving the same print twice is idempotent: when the target name already
/// holds a byte-identical file, no `-2` copy is minted — the existing file
/// is re-recorded and its name returned, preserving cross-host identity.
#[tauri::command]
pub async fn save_output_bytes(
    filename: String,
    data_b64: String,
    metadata: Option<Box<mold_core::OutputMetadata>>,
) -> Result<String, String> {
    if !valid_filename(&filename) {
        return Err("Invalid filename.".into());
    }
    let dir = output_dir().ok_or_else(|| "Local output is disabled.".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .map_err(|e| format!("Invalid image data: {e}"))?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let existing = dir.join(&filename);
        // Idempotence requires byte-identical content, not just a matching
        // name and length — a different print under the same name must not
        // be silently dropped or re-recorded with the wrong provenance.
        let path = if file_matches_bytes(&existing, &bytes) {
            existing
        } else {
            let path = unique_output_path(&dir, &filename);
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
            path
        };

        let saved_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&filename)
            .to_string();
        // Best-effort DB row so the file shows in the local gallery
        // immediately instead of waiting for the next reconcile walk.
        // Embedded metadata wins (it is the file's own record); the caller's
        // wire metadata covers formats that embed nothing; filename
        // synthesis remains the last resort.
        if let (Some(db), Some(format)) = (
            mold_db::global_db(),
            mold_db::metadata_io::format_from_path(&path),
        ) {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let row_metadata = match mold_db::metadata_io::read_embedded(&path, format) {
                Some(embedded) => embedded,
                None => match metadata {
                    Some(provided) => *provided,
                    None => mold_db::metadata_io::synthesize_from_filename(&saved_name, timestamp),
                },
            };
            let _ = mold_db::persist::record_saved_output(
                db,
                &dir,
                &saved_name,
                &path,
                &mold_db::persist::OutputRecordParams {
                    format,
                    metadata: &row_metadata,
                    source: mold_db::RecordSource::Backfill,
                    generation_time_ms: None,
                    backend: None,
                },
            );
        }
        Ok(saved_name)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn local_gallery_delete(filename: String) -> Result<(), String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let dir = output_dir().ok_or_else(|| "Local gallery is disabled.".to_string())?;
    let path = dir.join(&filename);
    if path.exists() {
        let root = dir.canonicalize().map_err(|error| error.to_string())?;
        let candidate = path.canonicalize().map_err(|error| error.to_string())?;
        if !candidate.starts_with(&root) {
            return Err("Invalid gallery filename.".into());
        }
        std::fs::remove_file(&candidate).map_err(|error| error.to_string())?;
    }
    if let Some(db) = mold_db::global_db() {
        let _ = db.delete(&dir, &filename);
    }
    Ok(())
}

fn content_type(filename: &str) -> &'static str {
    match filename
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" | "apng" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

fn error_response(status: StatusCode, message: &str) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(message.as_bytes().to_vec())
        .expect("valid protocol response")
}

pub fn protocol_response(request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let encoded = request.uri().path().trim_start_matches('/');
    let filename = match percent_encoding::percent_decode_str(encoded).decode_utf8() {
        Ok(filename) if valid_filename(&filename) => filename.into_owned(),
        _ => return error_response(StatusCode::BAD_REQUEST, "Invalid gallery filename."),
    };
    let Some(dir) = output_dir() else {
        return error_response(StatusCode::NOT_FOUND, "Local gallery is disabled.");
    };
    let path = dir.join(&filename);
    let safe_path = match (dir.canonicalize(), path.canonicalize()) {
        (Ok(root), Ok(candidate)) if candidate.starts_with(&root) => candidate,
        _ => return error_response(StatusCode::NOT_FOUND, "Gallery file not found."),
    };
    let Ok(mut file) = std::fs::File::open(&safe_path) else {
        return error_response(StatusCode::NOT_FOUND, "Gallery file not found.");
    };
    let Ok(metadata) = file.metadata() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read gallery file.",
        );
    };
    let total = metadata.len();
    let range = match request.headers().get(header::RANGE) {
        Some(value) => match value
            .to_str()
            .map_err(|_| ())
            .and_then(|value| parse_byte_range(value, total))
        {
            Ok(range) => range,
            Err(()) => {
                return Response::builder()
                    .status(StatusCode::RANGE_NOT_SATISFIABLE)
                    .header(header::CONTENT_RANGE, format!("bytes */{total}"))
                    .header(header::ACCEPT_RANGES, "bytes")
                    .body(Vec::new())
                    .expect("valid range error response")
            }
        },
        None => None,
    };
    let (status, start, end) = range
        .map(|(start, end)| (StatusCode::PARTIAL_CONTENT, start, end))
        .unwrap_or((StatusCode::OK, 0, total.saturating_sub(1)));
    let length = if total == 0 { 0 } else { end - start + 1 };
    if file.seek(SeekFrom::Start(start)).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not seek gallery file.",
        );
    }
    let mut body = vec![0; length as usize];
    if file.read_exact(&mut body).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read gallery file.",
        );
    }
    let mut response = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type(&filename))
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, length.to_string())
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{total}"),
        );
    }
    response.body(body).expect("valid protocol response")
}

fn parse_byte_range(value: &str, total: u64) -> Result<Option<(u64, u64)>, ()> {
    let Some(spec) = value.strip_prefix("bytes=") else {
        return Ok(None);
    };
    if total == 0 {
        return Err(());
    }
    let first = spec.split(',').next().ok_or(())?.trim();
    let (start, end) = first.split_once('-').ok_or(())?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().map_err(|_| ())?;
        if suffix == 0 {
            return Err(());
        }
        let length = suffix.min(total);
        return Ok(Some((total - length, total - 1)));
    }
    let start = start.parse::<u64>().map_err(|_| ())?;
    if start >= total {
        return Err(());
    }
    let end = if end.is_empty() {
        total - 1
    } else {
        end.parse::<u64>().map_err(|_| ())?.min(total - 1)
    };
    if end < start {
        return Err(());
    }
    Ok(Some((start, end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_a_valid_png_as_base64() {
        use base64::Engine;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.png");
        image::RgbImage::from_pixel(2, 1, image::Rgb([12, 34, 56]))
            .save(&path)
            .unwrap();

        let imported = import_source_image_from_path(&path).unwrap();
        assert_eq!(imported.filename, "source.png");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(imported.base64)
                .unwrap(),
            std::fs::read(path).unwrap()
        );
        assert!(imported.metadata.is_none());
    }

    #[test]
    fn rejects_non_image_drops() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an image").unwrap();

        assert_eq!(
            import_source_image_from_path(&path).unwrap_err(),
            "Drop a PNG or JPEG image."
        );
    }

    #[test]
    fn file_matches_bytes_requires_identical_content() {
        let dir = std::env::temp_dir().join(format!("mold-fmb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.bin");
        std::fs::write(&path, b"hello world").unwrap();

        assert!(file_matches_bytes(&path, b"hello world"));
        // Same length, different content — the size gate alone must not pass.
        assert!(!file_matches_bytes(&path, b"hello_world"));
        assert!(!file_matches_bytes(&path, b"hello"));
        assert!(!file_matches_bytes(
            &dir.join("missing.bin"),
            b"hello world"
        ));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gallery_protocol_rejects_path_traversal() {
        assert!(!valid_filename("../secrets.json"));
        assert!(!valid_filename("nested/image.png"));
        assert!(valid_filename("mold-flux-1.png"));
    }

    #[test]
    fn parses_open_ended_and_suffix_byte_ranges() {
        assert_eq!(parse_byte_range("bytes=5-", 10), Ok(Some((5, 9))));
        assert_eq!(parse_byte_range("bytes=-4", 10), Ok(Some((6, 9))));
        assert_eq!(parse_byte_range("bytes=-40", 10), Ok(Some((0, 9))));
    }

    #[test]
    fn rejects_unsatisfiable_or_malformed_byte_ranges() {
        assert_eq!(parse_byte_range("bytes=10-", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=8-2", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=-0", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=nope", 10), Err(()));
    }

    #[test]
    fn unique_output_path_counts_up_past_collisions() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "print.png"),
            dir.path().join("print.png")
        );
        std::fs::write(dir.path().join("print.png"), b"x").unwrap();
        std::fs::write(dir.path().join("print-2.png"), b"x").unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "print.png"),
            dir.path().join("print-3.png")
        );
        // Extension-less names still get numbered rather than clobbered.
        std::fs::write(dir.path().join("raw"), b"x").unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "raw"),
            dir.path().join("raw-2")
        );
    }
}
