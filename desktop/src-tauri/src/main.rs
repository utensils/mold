// Prevents an extra console window on Windows in release; harmless elsewhere.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if mold_desktop_lib::updater::run_supervisor_if_requested() {
        return;
    }
    mold_desktop_lib::run()
}
