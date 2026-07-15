pub mod clipboard;
pub mod commands;
pub mod connection;
pub mod gallery;
pub mod menu;
pub mod notifications;
pub mod runpod;
pub mod secrets;
pub mod server;
pub mod settings;
pub mod updater;

use tauri::Manager;

pub fn run() {
    // One-shot config.toml → DB migration + DB overlay on Config::load,
    // exactly like every other mold binary's main().
    mold_db::config_sync::install_config_post_load_hook();

    // The desktop app owns no terminal — log to ~/.mold/logs only.
    let config = mold_core::Config::load_or_default();
    let log_guard = mold_server::logging::init_tracing_file_only(
        &config.logging,
        "info",
        config.resolved_log_dir(),
    );

    tauri::Builder::default()
        .register_uri_scheme_protocol("mold-local", |_context, request| {
            gallery::protocol_response(request)
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
            let menu = menu::build(app.handle())?;
            app.set_menu(menu)?;
            app.manage(commands::SettingsStore::load(app.handle())?);
            app.manage(updater::UpdaterState::default());
            let app_data = app.path().app_data_dir()?;
            let secrets = secrets::SecretStore::new(app_data);
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
            commands::ensure_local_server,
            commands::start_local_engine,
            commands::stop_local_engine,
            commands::set_remote_host,
            commands::forget_remote_host,
            commands::test_remote_host,
            commands::discover_servers,
            commands::get_output_dir,
            commands::set_dock_badge,
            notifications::send_native_notification,
            commands::reveal_output_file,
            commands::open_logs_dir,
            updater::check_for_updates,
            updater::install_pending_update,
            updater::take_update_recovery,
            updater::confirm_update_healthy,
            gallery::local_gallery_list,
            gallery::local_gallery_delete,
            gallery::save_output_bytes,
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
                if let Some(engine) = engine {
                    let url = format!("{}/api/shutdown", engine.base_url());
                    let key = state.local_api_key.clone();
                    std::thread::spawn(move || {
                        let _ = reqwest::blocking::Client::new()
                            .post(&url)
                            .header("X-Api-Key", &key)
                            .timeout(std::time::Duration::from_secs(3))
                            .send();
                        engine.join(std::time::Duration::from_secs(5));
                    });
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running mold desktop");
}
