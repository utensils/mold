//! Thin native shell for Mold on iPhone. Generation and gallery logic use the
//! shared Vue wire types while the phone connects only to remote Mold servers.

mod appearance;
mod discovery;
mod keychain;
mod media;
mod viewport;

fn app_context() -> tauri::Context<tauri::Wry> {
    tauri::generate_context!()
}

#[cfg(target_os = "ios")]
fn install_network_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("the iPhone app must install its rustls crypto provider before networking");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "ios")]
    install_network_crypto_provider();

    tauri::Builder::default()
        .plugin(tauri_plugin_deep_link::init())
        .setup(|_app| {
            #[cfg(target_os = "ios")]
            {
                media::cleanup_stale_animation_exports();
                _app.handle().plugin(tauri_plugin_barcode_scanner::init())?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            appearance::set_mobile_appearance,
            discovery::discover_mold_hosts,
            keychain::keychain_set_api_key,
            keychain::keychain_get_api_key,
            keychain::keychain_delete_api_key,
            media::copy_image_to_clipboard,
            media::save_image_to_photos,
            media::save_video_to_photos,
            media::share_exported_animation,
            viewport::restore_mobile_viewport,
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

    #[test]
    fn ios_declares_add_only_photos_access_for_explicit_saves() {
        let plist = include_str!("../Info.ios.plist");
        assert!(plist.contains("<key>NSPhotoLibraryAddUsageDescription</key>"));
        assert!(plist.contains("Save generated images and videos"));
    }

    #[test]
    fn ios_declares_camera_access_only_for_pairing_scans() {
        let plist = include_str!("../Info.ios.plist");
        assert!(plist.contains("<key>NSCameraUsageDescription</key>"));
        assert!(plist.contains("Scan a one-time Mold host pairing code."));
    }

    #[test]
    fn ios_registers_mobile_pairing_deep_links() {
        let config = include_str!("../tauri.conf.json");
        let generated_plist = include_str!("../gen/apple/mold-mobile_iOS/Info.plist");
        assert!(config.contains("\"deep-link\""));
        assert!(config.contains("\"scheme\": [\"mold\"]"));
        assert!(generated_plist.contains("<key>CFBundleURLTypes</key>"));
        assert!(generated_plist.contains("<string>mold</string>"));
    }

    #[cfg(target_os = "ios")]
    #[test]
    fn ios_installs_crypto_provider_before_building_network_clients() {
        super::install_network_crypto_provider();
        reqwest::Client::builder()
            .build()
            .expect("reqwest must be usable after iPhone startup installs the provider");
    }
}
