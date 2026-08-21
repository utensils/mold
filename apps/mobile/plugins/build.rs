const COMMANDS: &[&str] = &["set_api_key", "get_api_key", "delete_api_key"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();
}
