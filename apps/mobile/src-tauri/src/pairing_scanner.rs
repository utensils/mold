#[derive(serde::Serialize)]
pub struct PairingScanResult {
    pub content: String,
}

#[tauri::command]
pub async fn scan_android_pairing_code(app: tauri::AppHandle) -> Result<PairingScanResult, String> {
    #[cfg(target_os = "android")]
    {
        use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;
        let response = app
            .mold_mobile_native()
            .scan_pairing_code()
            .await
            .map_err(|error| error.to_string())?;
        return Ok(PairingScanResult {
            content: response.content,
        });
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Err("the native pairing scanner is available only on Android".to_owned())
    }
}

#[tauri::command]
pub async fn cancel_android_pairing_scan(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "android")]
    {
        use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;
        return app
            .mold_mobile_native()
            .cancel_pairing_scan()
            .await
            .map_err(|error| error.to_string());
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        Err("the native pairing scanner is available only on Android".to_owned())
    }
}
