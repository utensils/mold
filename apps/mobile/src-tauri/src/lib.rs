//! Thin native shell for Mold on iPhone and Android. Generation and gallery
//! logic use the shared Vue wire types while the phone connects only to remote
//! Mold servers.

mod appearance;
mod background_task;
mod context_menu;
mod discovery;
mod identity;
mod keychain;
mod media;
mod pairing_scanner;
mod viewport;

#[cfg(target_os = "android")]
use tauri_plugin_mold_mobile_native as mobile_native;

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

    let builder = tauri::Builder::default()
        .manage(background_task::MobileBackgroundTaskState::default())
        .plugin(tauri_plugin_deep_link::init());
    #[cfg(target_os = "android")]
    let builder = builder.plugin(mobile_native::init());

    builder
        .setup(|_app| {
            #[cfg(any(target_os = "ios", target_os = "android"))]
            {
                #[cfg(target_os = "ios")]
                media::cleanup_stale_media_exports();
                _app.handle().plugin(tauri_plugin_barcode_scanner::init())?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            appearance::set_mobile_appearance,
            background_task::begin_mobile_background_task,
            background_task::end_mobile_background_task,
            context_menu::extend_gallery_context_menu,
            discovery::discover_mold_hosts,
            keychain::keychain_set_api_key,
            keychain::keychain_get_api_key,
            keychain::keychain_delete_api_key,
            identity::pick_identity_photo,
            media::copy_image_to_clipboard,
            media::save_image_to_photos,
            media::save_video_to_photos,
            media::share_exported_animation,
            media::save_export_to_mold_folder,
            pairing_scanner::scan_android_pairing_code,
            pairing_scanner::cancel_android_pairing_scan,
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

    /// "Save to Mold folder" writes `<Documents>/Mold`; without these two
    /// keys the folder exists but never appears in Files ▸ On My iPhone, so
    /// the save would be real and invisible.
    #[test]
    fn ios_exposes_the_documents_folder_in_the_files_app() {
        let plist = include_str!("../Info.ios.plist");
        assert!(plist.contains("<key>UIFileSharingEnabled</key>\n  <true/>"));
        assert!(plist.contains("<key>LSSupportsOpeningDocumentsInPlace</key>\n  <true/>"));
        let generated_plist = include_str!("../gen/apple/mold-mobile_iOS/Info.plist");
        assert!(generated_plist.contains("<key>UIFileSharingEnabled</key>"));
        assert!(generated_plist.contains("<key>LSSupportsOpeningDocumentsInPlace</key>"));
    }

    #[test]
    fn ios_declares_camera_access_only_for_pairing_scans() {
        let plist = include_str!("../Info.ios.plist");
        assert!(plist.contains("<key>NSCameraUsageDescription</key>"));
        assert!(plist.contains("Scan a one-time Mold host pairing code."));
    }

    #[test]
    fn ios_registers_mobile_pairing_deep_links() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).expect("valid Tauri config");
        let generated_plist = include_str!("../gen/apple/mold-mobile_iOS/Info.plist");
        assert_eq!(
            config["plugins"]["deep-link"]["mobile"][0]["scheme"][0],
            "mold"
        );
        assert!(generated_plist.contains("<key>CFBundleURLTypes</key>"));
        assert!(generated_plist.contains("<string>mold</string>"));
    }

    #[test]
    fn native_capabilities_cover_ios_and_android() {
        let capability: serde_json::Value =
            serde_json::from_str(include_str!("../capabilities/default.json"))
                .expect("valid mobile capability");
        let platforms = capability["platforms"]
            .as_array()
            .expect("mobile capability platforms");
        assert!(platforms.iter().any(|platform| platform == "iOS"));
        assert!(platforms.iter().any(|platform| platform == "android"));
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
