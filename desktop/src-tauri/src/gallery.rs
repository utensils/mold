use std::io::{Read, Seek, SeekFrom};

use tauri::http::{header, Request, Response, StatusCode};

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
}
