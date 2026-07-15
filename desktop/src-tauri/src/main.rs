// Prevents an extra console window on Windows in release; harmless elsewhere.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mold_desktop_lib::updater::retire_legacy_supervisor_if_present();
    mold_desktop_lib::run()
}
