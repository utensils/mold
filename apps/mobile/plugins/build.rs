const COMMANDS: &[&str] = &[
    "set_api_key",
    "get_api_key",
    "delete_api_key",
    "discover_mold_hosts",
    "copy_image_to_clipboard",
    "save_image_to_photos",
    "save_video_to_photos",
    "share_exported_animation",
    "pick_identity_photo",
    "set_mobile_appearance",
    "scan_pairing_code",
    "cancel_pairing_scan",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();
}
