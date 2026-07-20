//! Thin native shell for Mold on iPhone. Generation and gallery logic use the
//! shared Vue wire types while the phone connects only to remote Mold servers.

mod appearance;
mod discovery;
mod keychain;

fn app_context() -> tauri::Context<tauri::Wry> {
    tauri::generate_context!()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            appearance::set_mobile_appearance,
            discovery::discover_mold_hosts,
            keychain::keychain_set_api_key,
            keychain::keychain_get_api_key,
            keychain::keychain_delete_api_key,
        ])
        .run(app_context())
        .expect("error while running mold-mobile");
}

#[cfg(test)]
mod tests {
    use tauri::utils::assets::AssetKey;

    #[test]
    fn packaged_frontend_contains_mobile_entry() {
        let context = super::app_context();
        let index = context
            .assets()
            .get(&AssetKey::from("index.html"))
            .expect("Tauri frontend assets must include /index.html");
        let index = std::str::from_utf8(index.as_ref()).expect("index.html must be UTF-8");

        assert!(
            index.contains("mold-mobile-entry-v1"),
            "embedded index.html must be the Mold mobile entry"
        );
        assert!(
            index.contains("maximum-scale=1") && index.contains("user-scalable=no"),
            "embedded mobile viewport must disable iPhone document zoom"
        );
        assert!(
            context
                .assets()
                .get(&AssetKey::from("index.mobile.html"))
                .is_none(),
            "the source filename must be renamed before Tauri embeds it"
        );
    }

    #[test]
    fn ios_uses_view_controller_status_bar_appearance() {
        let plist = include_str!("../Info.ios.plist");
        assert!(
            plist.contains("<key>UIViewControllerBasedStatusBarAppearance</key>\n  <true/>"),
            "iOS must let the Tauri view controller update status-bar appearance"
        );
    }
}
