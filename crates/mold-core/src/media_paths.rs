use std::path::PathBuf;

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
}
