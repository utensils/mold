//! Offer to move Mold into /Applications when it is launched from a place that
//! poisons macOS Launch Services.
//!
//! macOS resolves a notification's app icon (and update/quarantine behaviour) by
//! `CFBundleIdentifier` through Launch Services. When Mold is run straight from a
//! mounted `.dmg` (`/Volumes/Mold/Mold.app`) or from a Gatekeeper App
//! Translocation path, that transient bundle registers `com.utensils.mold`. Once
//! the disk image is ejected the registration goes stale, `usernoted` can no
//! longer load the icon, and notifications fall back to a generic placeholder —
//! even for a later copy dragged into /Applications. Getting the user to run from
//! /Applications on first launch prevents the stale registration entirely.
//!
//! Only the pure decision (`relocation_reason`) and path helpers are unit tested;
//! the copy/relaunch and native dialog are thin side-effecting wrappers.

use std::path::{Path, PathBuf};

/// Why the running bundle should be relocated to /Applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocateReason {
    /// Running directly from a read-only mounted disk image.
    DiskImage,
    /// Running from a Gatekeeper App Translocation quarantine path.
    Translocated,
}

impl RelocateReason {
    /// User-facing explanation, tied to the concrete breakage each cause creates.
    pub fn dialog_body(self) -> &'static str {
        match self {
            RelocateReason::DiskImage => {
                "Mold is running from its disk image. Move it to your Applications \
folder to finish installing — otherwise automatic updates and the app icon on \
notifications won't work correctly. You can eject the disk image afterwards."
            }
            RelocateReason::Translocated => {
                "macOS is running Mold from a temporary quarantine location. Move it \
to your Applications folder so automatic updates and the app icon on \
notifications work correctly."
            }
        }
    }
}

/// Decide whether to offer relocation. Pure so the policy is unit tested.
///
/// A writable volume under `/Volumes` (an external drive) is deliberately left
/// alone — only a read-only image (a `.dmg`) or a translocated launch prompts.
pub fn relocation_reason(
    bundle: &Path,
    translocated: bool,
    under_volumes: bool,
    volume_read_only: bool,
    applications_dirs: &[PathBuf],
) -> Option<RelocateReason> {
    // Already installed in a standard Applications folder — never prompt.
    if applications_dirs.iter().any(|dir| bundle.starts_with(dir)) {
        return None;
    }
    if translocated {
        return Some(RelocateReason::Translocated);
    }
    if under_volumes && volume_read_only {
        return Some(RelocateReason::DiskImage);
    }
    None
}

/// Walk up an executable path to the enclosing `.app` bundle.
/// `.../Mold.app/Contents/MacOS/mold-desktop` → `.../Mold.app`.
pub fn bundle_path_from_exe(exe: &Path) -> Option<PathBuf> {
    let mut current = Some(exe);
    while let Some(path) = current {
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with(".app"))
        {
            return Some(path.to_path_buf());
        }
        current = path.parent();
    }
    None
}

/// A path is translocated when macOS interposed an `AppTranslocation` container.
pub fn path_is_translocated(bundle: &Path) -> bool {
    bundle
        .components()
        .any(|component| component.as_os_str() == "AppTranslocation")
}

#[cfg(target_os = "macos")]
fn applications_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("/Applications")];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("Applications"));
    }
    dirs
}

/// True when the filesystem backing `path` is mounted read-only (a disk image).
#[cfg(target_os = "macos")]
fn volume_is_read_only(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is zeroed
    // before the call, which fills it on success.
    unsafe {
        let mut stat: libc::statfs = std::mem::zeroed();
        libc::statfs(c_path.as_ptr(), &mut stat) == 0
            && stat.f_flags & (libc::MNT_RDONLY as u32) != 0
    }
}

/// Copy the running bundle into /Applications with `ditto` (which preserves the
/// code signature and extended attributes that `cp -R` would strip). Returns the
/// installed path.
#[cfg(target_os = "macos")]
fn relocate_to_applications(bundle: &Path) -> Result<PathBuf, String> {
    let name = bundle
        .file_name()
        .ok_or_else(|| "bundle has no name".to_string())?;
    let target = PathBuf::from("/Applications").join(name);
    if target.exists() {
        std::fs::remove_dir_all(&target)
            .map_err(|e| format!("couldn't replace {}: {e}", target.display()))?;
    }
    let status = std::process::Command::new("/usr/bin/ditto")
        .arg(bundle)
        .arg(&target)
        .status()
        .map_err(|e| format!("failed to run ditto: {e}"))?;
    if !status.success() {
        return Err(format!("ditto exited with {status}"));
    }
    Ok(target)
}

/// On macOS, if Mold was launched from a disk image or a translocation path,
/// offer to move it to /Applications and relaunch from there. No-op elsewhere,
/// in dev builds (which run from `target/`), and when already installed.
#[cfg(target_os = "macos")]
pub fn maybe_offer_relocation(app: &tauri::AppHandle) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    // Dev builds run from `target/…`; never nag while iterating.
    if cfg!(debug_assertions) {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(bundle) = bundle_path_from_exe(&exe) else {
        return;
    };
    let Some(reason) = relocation_reason(
        &bundle,
        path_is_translocated(&bundle),
        bundle.starts_with("/Volumes/"),
        volume_is_read_only(&bundle),
        &applications_dirs(),
    ) else {
        return;
    };

    // blocking_show must not run on the main thread; the setup hook does.
    let app = app.clone();
    std::thread::spawn(move || {
        let accepted = app
            .dialog()
            .message(reason.dialog_body())
            .title("Move Mold to Applications?")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Move to Applications".into(),
                "Not Now".into(),
            ))
            .blocking_show();
        if !accepted {
            return;
        }
        match relocate_to_applications(&bundle) {
            Ok(target) => {
                // Wait for this instance to exit before relaunching so the
                // single-instance lock is free — otherwise `open` just refocuses
                // the dying disk-image instance and the new copy never starts.
                let pid = std::process::id();
                let _ = std::process::Command::new("/bin/sh")
                    .arg("-c")
                    .arg(format!(
                        "while kill -0 {pid} 2>/dev/null; do sleep 0.2; done; \
                         open {}",
                        shell_single_quote(&target.to_string_lossy()),
                    ))
                    .spawn();
                app.exit(0);
            }
            Err(error) => {
                tracing::warn!("relocation to /Applications failed: {error}");
                let _ = app
                    .dialog()
                    .message(format!(
                        "Mold couldn't move itself to Applications:\n{error}\n\n\
                         Drag Mold to your Applications folder manually, then eject \
                         the disk image."
                    ))
                    .title("Couldn't move Mold")
                    .kind(MessageDialogKind::Error)
                    .blocking_show();
            }
        }
    });
}

/// Wrap a string in single quotes for `/bin/sh`, escaping embedded quotes.
#[cfg(target_os = "macos")]
fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apps() -> Vec<PathBuf> {
        vec![
            PathBuf::from("/Applications"),
            PathBuf::from("/Users/me/Applications"),
        ]
    }

    #[test]
    fn installed_bundle_is_never_relocated() {
        assert_eq!(
            relocation_reason(
                Path::new("/Applications/Mold.app"),
                false,
                false,
                false,
                &apps()
            ),
            None
        );
        assert_eq!(
            relocation_reason(
                Path::new("/Users/me/Applications/Mold.app"),
                false,
                false,
                false,
                &apps()
            ),
            None
        );
    }

    #[test]
    fn disk_image_launch_offers_relocation() {
        assert_eq!(
            relocation_reason(
                Path::new("/Volumes/Mold/Mold.app"),
                false,
                true,
                true,
                &apps()
            ),
            Some(RelocateReason::DiskImage)
        );
    }

    #[test]
    fn writable_external_volume_is_left_alone() {
        // An external SSD mounts under /Volumes but is writable — not a disk image.
        assert_eq!(
            relocation_reason(
                Path::new("/Volumes/SSD/Mold.app"),
                false,
                true,
                false,
                &apps()
            ),
            None
        );
    }

    #[test]
    fn translocation_offers_relocation_and_outranks_disk_image() {
        assert_eq!(
            relocation_reason(
                Path::new("/private/var/folders/x/y/T/AppTranslocation/UUID/d/Mold.app"),
                true,
                false,
                false,
                &apps()
            ),
            Some(RelocateReason::Translocated)
        );
        // Even a translocated read-only image reports Translocated first.
        assert_eq!(
            relocation_reason(
                Path::new("/Volumes/Mold/Mold.app"),
                true,
                true,
                true,
                &apps()
            ),
            Some(RelocateReason::Translocated)
        );
    }

    #[test]
    fn arbitrary_writable_location_does_not_prompt() {
        // e.g. a nix-store path or a downloaded copy sitting on the boot volume.
        assert_eq!(
            relocation_reason(
                Path::new("/nix/store/abc-mold-desktop/Applications/Mold.app"),
                false,
                false,
                false,
                &apps()
            ),
            None
        );
    }

    #[test]
    fn bundle_path_is_derived_from_the_executable() {
        assert_eq!(
            bundle_path_from_exe(Path::new(
                "/Volumes/Mold/Mold.app/Contents/MacOS/mold-desktop"
            )),
            Some(PathBuf::from("/Volumes/Mold/Mold.app"))
        );
        assert_eq!(
            bundle_path_from_exe(Path::new("/usr/local/bin/mold-desktop")),
            None
        );
    }

    #[test]
    fn translocation_is_detected_by_container_component() {
        assert!(path_is_translocated(Path::new(
            "/private/var/folders/x/y/T/AppTranslocation/UUID/d/Mold.app"
        )));
        assert!(!path_is_translocated(Path::new("/Applications/Mold.app")));
    }

    #[test]
    fn dialog_bodies_are_distinct_and_nonempty() {
        let dmg = RelocateReason::DiskImage.dialog_body();
        let translocated = RelocateReason::Translocated.dialog_body();
        assert!(!dmg.is_empty() && !translocated.is_empty());
        assert_ne!(dmg, translocated);
    }
}
