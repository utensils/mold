// Prevents an extra console window on Windows in release; harmless elsewhere.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Keep the same positive exclusion provenance as the CLI in CUDA desktop
    // artifacts so AppImage and Nix publication checks can inspect it.
    std::hint::black_box(mold_server::h3_attention_release_provenance_marker());
    mold_desktop_lib::updater::retire_legacy_supervisor_if_present();
    mold_desktop_lib::run()
}
