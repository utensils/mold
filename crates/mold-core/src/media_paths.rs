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

/// Revision of the mesh poster renderer.
///
/// A mesh poster is not read out of the print the way a waveform or a video
/// frame is — it is RENDERED from the geometry, so it changes when the
/// renderer's camera, framing, lighting or palette change while the `.glb`'s
/// own mtime and size do not. Every cache downstream keys on one of two
/// things and neither would notice: the sidecar is addressed by NAME, and the
/// clients key their tile caches on the opaque `media_version`. So the
/// revision goes into both, and this is the constant both derive from —
/// `mesh_poster_thumbnail_paths` infixes it into the name, and
/// `mold_server::thumbnails::MESH_POSTER_REVISION_SUFFIX` is this string with
/// a leading `:` for the wire, pinned to this one by a test there.
///
/// Bump it in the same change as any alteration to the poster's pixels. `p2`
/// is the shared sweep framing: the poster is now fit to the
/// rotation-invariant bound that also frames the turntable and the
/// interactive viewer, so every mesh print rendered before it is drawn at a
/// different size.
pub const MESH_POSTER_REVISION: &str = "p2";

/// Both thumbnail-cache paths a mesh output needs.
///
/// Close in shape to [`audio_waveform_thumbnail_paths`] and for a related
/// reason: a mesh has no raster frame, so neither the server's on-demand
/// thumbnailer nor the TUI's `image::open` can produce a tile. The poster PNG
/// is rendered at save time and written to both names, because the two
/// consumers spell their cache entries differently and a saver must not guess
/// which surface will open the print first.
///
/// Where it PARTS from audio is [`MESH_POSTER_REVISION`], which is infixed
/// into the server's name (`<file>.p2.png`). A waveform is a transcription of
/// bytes that do not change; a poster is a render, so a pre-revision sidecar
/// has to MISS rather than be served verbatim forever. Making the name carry
/// the revision is what turns "the poster renderer changed" into an ordinary
/// cache miss for the route, `ensure_mesh_poster`, the desktop's offline
/// tiles, and the save-time writers all at once — none of them needs to know
/// a revision exists.
///
/// The TUI's name (`<file>.thumb.png`) deliberately does NOT carry it. The
/// TUI resolves that name itself, from its own `thumbnail_path`, without
/// reading this module, so revisioning it would leave every locally generated
/// mesh print with no TUI tile at all. It self-heals instead: a re-render
/// triggered by the server name's miss rewrites BOTH sidecars.
pub fn mesh_poster_thumbnail_paths(
    thumbnail_dir: &std::path::Path,
    filename: &str,
) -> [PathBuf; 2] {
    [
        thumbnail_dir.join(format!("{filename}.{MESH_POSTER_REVISION}.png")),
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

    /// The server's mesh sidecar carries the poster revision and the TUI's
    /// does not, and both differ from the audio pair they used to share.
    #[test]
    fn the_mesh_poster_sidecar_carries_the_renderer_revision() {
        let dir = std::path::Path::new("/cache");
        let [server, tui] = mesh_poster_thumbnail_paths(dir, "chair.glb");
        assert_eq!(
            server,
            dir.join(format!("chair.glb.{MESH_POSTER_REVISION}.png"))
        );
        assert_eq!(tui, dir.join("chair.glb.thumb.png"));

        // A pre-revision poster is therefore at a name nothing reads any
        // more, which is exactly how "the renderer changed" becomes a miss.
        assert_ne!(server, dir.join("chair.glb.png"));

        // Audio is untouched: its tile is a transcription of bytes that do
        // not change, so its sidecar has no revision to carry.
        let [audio_server, audio_tui] = audio_waveform_thumbnail_paths(dir, "take.wav");
        assert_eq!(audio_server, dir.join("take.wav.png"));
        assert_eq!(audio_tui, dir.join("take.wav.thumb.png"));
    }

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

    /// Titled prints carry a `~<slug>` in the stem. Sidecar naming is a
    /// plain suffix append, so the separator must survive untouched in both
    /// the preview and thumbnail cache names.
    #[test]
    fn sidecar_names_preserve_title_slug_separator() {
        let filename = "mold-ltx2-1700000000000~smurf-village.mp4";
        assert_eq!(
            preview_gif_filename(filename),
            "mold-ltx2-1700000000000~smurf-village.mp4.preview.gif"
        );
        let dir = std::path::Path::new("/cache/thumbnails");
        let [server, tui] = audio_waveform_thumbnail_paths(dir, filename);
        assert_eq!(
            server,
            dir.join("mold-ltx2-1700000000000~smurf-village.mp4.png")
        );
        assert_eq!(
            tui,
            dir.join("mold-ltx2-1700000000000~smurf-village.mp4.thumb.png")
        );
        // `~` is an ordinary path character: no home expansion, one component.
        assert_eq!(
            server
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .matches('~')
                .count(),
            1
        );
        assert_eq!(expand_home(filename), PathBuf::from(filename));
    }
}
