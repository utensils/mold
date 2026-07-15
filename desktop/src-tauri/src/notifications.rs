use tauri::Manager;

#[cfg(target_os = "macos")]
fn bundled_icon(resource_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let path = resource_dir.join("icon.icns");
    path.is_file().then_some(path)
}

/// Send a notification through the platform path that can preserve Mold's
/// identity icon. Other platforms (and unbundled macOS development builds)
/// return `false` so the frontend can use Tauri's cross-platform fallback.
#[tauri::command]
pub async fn send_native_notification(
    app: tauri::AppHandle,
    title: String,
    body: Option<String>,
) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
        let Some(icon) = bundled_icon(&resource_dir) else {
            return Ok(false);
        };
        let icon = icon.to_string_lossy().into_owned();
        let identifier = app.config().identifier.clone();

        tauri::async_runtime::spawn_blocking(move || {
            // This is process-global and may report AlreadySet if another
            // notification raced us. Either way, the existing value is Mold.
            let _ = mac_notification_sys::set_application(&identifier);
            let mut notification = mac_notification_sys::Notification::new();
            notification
                .title(&title)
                .message(body.as_deref().unwrap_or(""))
                .app_icon(&icon);
            notification.send().map(|_| true).map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, title, body);
        Ok(false)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::bundled_icon;

    #[test]
    fn uses_the_bundled_macos_icon_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let icon = dir.path().join("icon.icns");
        std::fs::write(&icon, b"test icon").unwrap();

        assert_eq!(bundled_icon(dir.path()), Some(icon));
    }

    #[test]
    fn leaves_unbundled_development_runs_to_the_fallback() {
        let dir = tempfile::tempdir().unwrap();

        assert_eq!(bundled_icon(dir.path()), None);
    }
}
