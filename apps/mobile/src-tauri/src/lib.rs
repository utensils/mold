//! Thin native shell for Mold on iPhone. Generation and gallery logic use the
//! shared Vue wire types while the phone connects only to remote Mold servers.

mod discovery;
mod keychain;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            discovery::discover_mold_hosts,
            keychain::keychain_set_api_key,
            keychain::keychain_get_api_key,
            keychain::keychain_delete_api_key,
        ])
        .run(tauri::generate_context!())
        .expect("error while running mold-mobile");
}
