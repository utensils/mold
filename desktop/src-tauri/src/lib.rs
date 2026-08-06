pub mod clipboard;
pub mod commands;
pub mod connection;
pub mod gallery;
#[cfg(target_os = "macos")]
mod macos_window;
pub mod menu;
pub mod mold_home;
pub mod notifications;
pub mod relocate;
pub mod runpod;
pub mod secrets;
pub mod server;
pub mod settings;
pub mod source_stash;
pub mod updater;

use tauri::Manager;

fn app_window_title(is_development: bool) -> &'static str {
    if is_development {
        "Mold - dev"
    } else {
        "Mold"
    }
}

#[cfg(target_os = "linux")]
fn preferred_linux_gdk_backend(
    wayland_display: Option<&str>,
    x11_display: Option<&str>,
    configured_backend: Option<&str>,
) -> Option<&'static str> {
    if configured_backend.is_none() && wayland_display.is_some() && x11_display.is_some() {
        Some("x11")
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn preferred_linux_dmabuf_setting(configured: Option<&str>) -> Option<&'static str> {
    configured.is_none().then_some("1")
}

pub fn run() {
    #[cfg(target_os = "linux")]
    if let Some(backend) = preferred_linux_gdk_backend(
        std::env::var("WAYLAND_DISPLAY").ok().as_deref(),
        std::env::var("DISPLAY").ok().as_deref(),
        std::env::var("GDK_BACKEND").ok().as_deref(),
    ) {
        // WebKitGTK can hit compositor protocol errors on some Wayland stacks.
        // XWayland is widely available and users can still override GDK_BACKEND.
        std::env::set_var("GDK_BACKEND", backend);
    }
    #[cfg(target_os = "linux")]
    if let Some(setting) = preferred_linux_dmabuf_setting(
        std::env::var("WEBKIT_DISABLE_DMABUF_RENDERER")
            .ok()
            .as_deref(),
    ) {
        // WebKitGTK's DMA-BUF renderer can create a native shell but leave the
        // webview blank when GBM allocation fails under NVIDIA compositors.
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", setting);
    }

    // One-shot config.toml → DB migration + DB overlay on Config::load,
    // exactly like every other mold binary's main(). The shared Mold-home
    // bootstrap pointer is resolved inside Config before either store opens.
    let saved_home_unavailable = std::env::var_os("MOLD_HOME").is_none()
        && match mold_core::Config::read_saved_mold_dir() {
            Ok(Some(path)) => !path.is_dir(),
            Ok(None) => false,
            Err(_) => true,
        };
    if !saved_home_unavailable {
        mold_db::config_sync::install_config_post_load_hook();
    }

    // An absent external drive must not be recreated as an empty Mold home.
    // Keep recovery logs beside the bootstrap pointer so Settings can launch
    // and let the user reconnect or replace the unavailable location.
    let config = if saved_home_unavailable {
        mold_core::Config::default()
    } else {
        mold_core::Config::load_or_default()
    };
    let log_dir = saved_home_unavailable
        .then(mold_core::Config::mold_home_pointer_path)
        .flatten()
        .and_then(|path| path.parent().map(|parent| parent.join("recovery-logs")))
        .unwrap_or_else(|| config.resolved_log_dir());
    let log_guard = mold_server::logging::init_tracing_file_only(&config.logging, "info", log_dir);

    tauri::Builder::default()
        .register_uri_scheme_protocol("mold-local", |context, request| {
            let state = context.app_handle().state::<commands::AppState>();
            gallery::protocol_response(&state, request)
        })
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(move |app| {
            // If Mold was launched from a disk image or a translocation path,
            // offer to move it to /Applications first — running from a transient
            // bundle poisons the Launch Services icon lookup notifications use.
            #[cfg(target_os = "macos")]
            {
                relocate::maybe_offer_relocation(app.handle());
                notifications::install_notification_delegate(app.handle());
            }
            if let Some(window) = app.get_webview_window("main") {
                window.set_title(app_window_title(cfg!(debug_assertions)))?;
            }
            menu::set_process_name(cfg!(debug_assertions));
            let menu = menu::build(app.handle(), cfg!(debug_assertions))?;
            app.set_menu(menu)?;
            let settings_store = commands::SettingsStore::load(app.handle())?;
            app.manage(updater::UpdaterState::default());
            let app_data = app.path().app_data_dir()?;
            let secrets = secrets::SecretStore::new(app_data);
            // Retire remote-primary installs: re-home the ex-primary as a
            // connected host so the built-in engine is always the internal
            // primary. Runs once (idempotent) before anything reads settings.
            {
                let mut current = settings_store.current.lock().expect("settings mutex");
                if settings::migrate_remote_primary(&mut current, &secrets) {
                    if let Err(e) = settings::save(&settings_store.path, &current) {
                        tracing::warn!("failed to persist remote-primary migration: {e}");
                    }
                }
            }
            app.manage(settings_store);
            let local_api_key = secrets.local_server_api_key()?;
            // mold-server reads auth from the environment when the server
            // thread starts. Resolve and export the persistent key first.
            std::env::set_var("MOLD_API_KEY", &local_api_key);
            app.manage(commands::AppState {
                conn: tokio::sync::Mutex::new(connection::Conn::Off),
                local_server: tokio::sync::Mutex::new(commands::LocalServer::Off),
                local_api_key,
                secrets,
            });
            // Keep the tracing appender alive for the app's lifetime.
            app.manage(log_guard);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::app_settings_get,
            commands::app_settings_set,
            commands::get_connection,
            commands::get_mold_home,
            commands::change_mold_home,
            commands::ensure_local_server,
            commands::start_local_engine,
            commands::stop_local_engine,
            commands::forget_remote_host,
            commands::test_remote_host,
            commands::discover_servers,
            commands::get_output_dir,
            commands::set_dock_badge,
            notifications::send_native_notification,
            notifications::take_notification_action,
            commands::reveal_output_file,
            commands::open_logs_dir,
            updater::check_for_updates,
            updater::install_pending_update,
            gallery::local_gallery_list,
            gallery::local_gallery_delete,
            gallery::import_source_image,
            gallery::save_output_bytes,
            gallery::save_image_as,
            gallery::local_output_file_path,
            source_stash::source_stash_put,
            source_stash::source_stash_get,
            clipboard::clipboard_write_image,
            commands::secret_get,
            commands::secret_set,
            commands::secret_clear,
            runpod::runpod_overview,
            runpod::runpod_create,
            runpod::runpod_network_volume_create,
            runpod::runpod_network_volume_update,
            runpod::runpod_network_volume_delete,
            runpod::runpod_start,
            runpod::runpod_stop,
            runpod::runpod_delete,
        ])
        .on_window_event(|window, event| {
            #[cfg(target_os = "macos")]
            if let tauri::WindowEvent::Focused(focused) = event {
                if let Err(error) = macos_window::set_traffic_light_focus(window, *focused) {
                    tracing::warn!("failed to update inactive traffic lights: {error}");
                }
            }
            if let tauri::WindowEvent::Destroyed = event {
                // Best-effort engine shutdown; chain jobs are crash-safe and
                // resumable, so a hard exit is acceptable as fallback.
                let state = window.state::<commands::AppState>();
                let engine = {
                    let mut local = match state.local_server.try_lock() {
                        Ok(local) => local,
                        Err(_) => return,
                    };
                    match std::mem::replace(&mut *local, commands::LocalServer::Off) {
                        commands::LocalServer::Embedded(engine) => Some(engine),
                        commands::LocalServer::External { .. } | commands::LocalServer::Off => None,
                    }
                };
                if let Some(mut engine) = engine {
                    let url = format!("{}/api/shutdown", engine.base_url());
                    let key = state.local_api_key.clone();
                    std::thread::spawn(move || {
                        let _ = reqwest::blocking::Client::new()
                            .post(&url)
                            .header("X-Api-Key", &key)
                            .timeout(std::time::Duration::from_secs(3))
                            .send();
                        if !engine.join(std::time::Duration::from_secs(5)) {
                            tracing::warn!("embedded engine remained alive during app teardown");
                        }
                    });
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running mold desktop");
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{preferred_linux_dmabuf_setting, preferred_linux_gdk_backend};

    #[test]
    fn linux_prefers_xwayland_when_both_displays_are_available() {
        assert_eq!(
            preferred_linux_gdk_backend(Some("wayland-1"), Some(":0"), None),
            Some("x11")
        );
        assert_eq!(
            preferred_linux_gdk_backend(Some("wayland-1"), Some(":0"), Some("wayland")),
            None
        );
        assert_eq!(
            preferred_linux_gdk_backend(Some("wayland-1"), None, None),
            None
        );
    }

    #[test]
    fn linux_disables_webkit_dmabuf_unless_explicitly_configured() {
        assert_eq!(preferred_linux_dmabuf_setting(None), Some("1"));
        assert_eq!(preferred_linux_dmabuf_setting(Some("0")), None);
    }
}

#[cfg(test)]
mod title_tests {
    use super::app_window_title;

    #[test]
    fn development_window_title_is_distinct_from_release() {
        assert_eq!(app_window_title(true), "Mold - dev");
        assert_eq!(app_window_title(false), "Mold");
    }
}
