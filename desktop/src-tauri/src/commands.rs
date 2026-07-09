use std::path::PathBuf;
use std::sync::Mutex;

use tauri::Manager;

use crate::settings::{self, AppSettings};

pub struct SettingsStore {
    pub path: PathBuf,
    pub current: Mutex<AppSettings>,
}

impl SettingsStore {
    pub fn load(app: &tauri::AppHandle) -> anyhow::Result<Self> {
        let path = app.path().app_data_dir()?.join("settings.json");
        let current = Mutex::new(settings::load(&path));
        Ok(Self { path, current })
    }
}

#[tauri::command]
pub fn app_settings_get(store: tauri::State<'_, SettingsStore>) -> AppSettings {
    store.current.lock().expect("settings mutex").clone()
}

#[tauri::command]
pub fn app_settings_set(
    store: tauri::State<'_, SettingsStore>,
    settings: AppSettings,
) -> Result<(), String> {
    settings::save(&store.path, &settings).map_err(|e| e.to_string())?;
    *store.current.lock().expect("settings mutex") = settings;
    Ok(())
}
