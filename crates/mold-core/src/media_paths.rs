use std::path::PathBuf;

/// Suffix appended to an output's filename to name its animated GIF
/// preview sidecar. The server writes `<filename>.preview.gif` into its
/// preview cache and the TUI reconstructs the same name — one constant
/// so the two can't drift.
pub const PREVIEW_GIF_SUFFIX: &str = ".preview.gif";

/// Preview-sidecar filename for a gallery output (e.g.
/// `mold-ltx2-1234.mp4` → `mold-ltx2-1234.mp4.preview.gif`).
pub fn preview_gif_filename(filename: &str) -> String {
    format!("{filename}{PREVIEW_GIF_SUFFIX}")
}

/// Suffix the TUI appends when caching a thumbnail. The server's cache in the
/// same directory uses a plain `.png` instead — see
/// [`audio_waveform_thumbnail_paths`], which writes both so an audio print has
/// a tile wherever it is opened.
pub const TUI_THUMBNAIL_SUFFIX: &str = ".thumb.png";

/// Both thumbnail-cache paths an audio output needs, given the shared
/// `<mold_dir>/cache/thumbnails` directory.
///
/// Audio has no raster frame, so neither the server's on-demand thumbnailer
/// nor the TUI's `image::open` can produce one — the waveform PNG has to be
/// written at save time. The two consumers name their cache entries
/// differently (`<file>.png` for the server route, `<file>.thumb.png` for the
/// TUI), so a saver writes both rather than guessing which surface will open
/// the print.
pub fn audio_waveform_thumbnail_paths(
    thumbnail_dir: &std::path::Path,
    filename: &str,
) -> [PathBuf; 2] {
    [
        thumbnail_dir.join(format!("{filename}.png")),
        thumbnail_dir.join(format!("{filename}{TUI_THUMBNAIL_SUFFIX}")),
    ]
}

fn expand_home(path: &str) -> PathBuf {
    if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from(path))
    } else if let Some(rest) = path.strip_prefix("~/") {
        dirs::home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| PathBuf::from(path))
    } else {
        PathBuf::from(path)
    }
}

/// Resolve a trusted server-local media path under one of the configured roots.
///
/// Both roots and the target are canonicalized before comparison, so lexical
/// `..` tricks and symlinks that escape the allow roots are rejected.
pub fn resolve_server_media_path(
    requested: &str,
    allow_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    if requested.trim().is_empty() {
        return Err("server-local media path must not be empty".to_string());
    }
    if allow_roots.is_empty() {
        return Err(
            "server-local media paths require configured media_roots or MOLD_MEDIA_ROOTS"
                .to_string(),
        );
    }

    let target = expand_home(requested)
        .canonicalize()
        .map_err(|e| format!("server-local media path not found: {e}"))?;
    if !target.is_file() {
        return Err("server-local media path must point to a file".to_string());
    }

    let mut saw_existing_root = false;
    for root in allow_roots {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        if !root.is_dir() {
            continue;
        }
        saw_existing_root = true;
        if target.starts_with(&root) {
            return Ok(target);
        }
    }

    if saw_existing_root {
        Err("server-local media path is outside configured media_roots".to_string())
    } else {
        Err("no configured media_roots exist on disk".to_string())
    }
}

pub fn parse_media_roots_env(value: &str) -> Vec<PathBuf> {
    std::env::split_paths(value)
        .filter(|path| !path.as_os_str().is_empty())
        .collect()
}

pub fn configured_media_roots(paths: &[String]) -> Vec<PathBuf> {
    paths.iter().map(|path| expand_home(path)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_server_media_path_accepts_file_under_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("clip.mp4");
        std::fs::write(&file, b"mp4").unwrap();

        let resolved = resolve_server_media_path(&file.to_string_lossy(), &[dir.path().into()])
            .expect("file under root should resolve");

        assert_eq!(resolved, file.canonicalize().unwrap());
    }

    #[test]
    fn resolve_server_media_path_rejects_missing_roots() {
        let err = resolve_server_media_path("/tmp/clip.mp4", &[]).unwrap_err();
        assert!(err.contains("media_roots"), "got: {err}");
    }

    #[test]
    fn resolve_server_media_path_rejects_nonexistent_target() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.mp4");

        let err = resolve_server_media_path(&missing.to_string_lossy(), &[dir.path().into()])
            .unwrap_err();

        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn resolve_server_media_path_rejects_directory_target() {
        let dir = tempfile::tempdir().unwrap();

        let err = resolve_server_media_path(&dir.path().to_string_lossy(), &[dir.path().into()])
            .unwrap_err();

        assert!(err.contains("file"), "got: {err}");
    }

    #[test]
    fn resolve_server_media_path_rejects_parent_escape() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file = outside.path().join("clip.mp4");
        std::fs::write(&file, b"mp4").unwrap();
        let escaped = root.path().join("..").join(
            outside
                .path()
                .file_name()
                .expect("tempdir should have a leaf"),
        );
        let escaped = escaped.join("clip.mp4");

        let err = resolve_server_media_path(&escaped.to_string_lossy(), &[root.path().into()])
            .unwrap_err();

        assert!(err.contains("outside"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn resolve_server_media_path_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file = outside.path().join("clip.mp4");
        std::fs::write(&file, b"mp4").unwrap();
        let link = root.path().join("linked.mp4");
        symlink(&file, &link).unwrap();

        let err =
            resolve_server_media_path(&link.to_string_lossy(), &[root.path().into()]).unwrap_err();

        assert!(err.contains("outside"), "got: {err}");
    }

    #[test]
    fn preview_gif_filename_appends_suffix() {
        assert_eq!(
            preview_gif_filename("mold-ltx2-1234.mp4"),
            "mold-ltx2-1234.mp4.preview.gif"
        );
    }
}
